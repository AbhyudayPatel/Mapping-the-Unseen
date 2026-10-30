"""Thin async wrapper over ``httpx.AsyncClient`` for the forensic engine.

Responsibilities (all timing is monotonic and relative to request start):

* a shared client factory (keep-alive so TLS/slow-start don't contaminate timing);
* retry with exponential backoff + jitter on 429/500/502/503/504, honouring the
  ``Retry-After`` header, plus transient transport errors;
* a monotonic timing helper (:class:`MonotonicTimer`) that records
  ``t_first_byte`` / ``t_end`` and, for streaming SSE, per-chunk arrival times
  into a :class:`forensics.core.types.Timing`;
* parsing of ``Server-Timing``-style headers into ``timing.server_ms``;
* :func:`request_json`, which NEVER raises on HTTP/transport errors (it returns
  the error text so error responses can be treated as forensic evidence);
* :func:`stream_sse`, an async generator over parsed SSE ``data:`` payloads that
  fills a shared :class:`Timing` as chunks arrive.

This module is NOT a workflow script: it may (and does) call
``time.monotonic()`` / ``time.time()`` at runtime to measure real latency.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator, Mapping

import httpx

from forensics.core.types import Chunk, Timing

# --------------------------------------------------------------------------- retry policy

#: HTTP status codes that are safe to retry (rate limiting + transient 5xx).
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass
class RetryPolicy:
    """Exponential-backoff-with-jitter retry configuration.

    Delay for attempt ``k`` (0-indexed) is ``min(max_delay, base_delay * 2**k)``
    split into a fixed half plus a uniform-random half ("equal jitter"). When the
    server sends ``Retry-After`` it takes precedence (capped at ``max_delay``).
    """

    max_retries: int = 4
    base_delay: float = 0.5          # seconds
    max_delay: float = 30.0          # seconds
    statuses: frozenset[int] = RETRY_STATUSES


def _backoff_delay(attempt: int, retry_after: float | None, policy: RetryPolicy,
                   rng: random.Random | None = None) -> float:
    """Compute the sleep before the next attempt (seconds)."""
    if retry_after is not None:
        return max(0.0, min(retry_after, policy.max_delay))
    r = rng or random
    ceiling = min(policy.max_delay, policy.base_delay * (2 ** attempt))
    if ceiling <= 0:
        return 0.0
    return ceiling / 2.0 + r.uniform(0.0, ceiling / 2.0)


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    val = None
    for k, v in headers.items():
        if k.lower() == "retry-after":
            val = v
            break
    if val is None:
        return None
    val = val.strip()
    try:
        return max(0.0, float(val))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(val)
        if dt is not None:
            return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        pass
    return None


# --------------------------------------------------------------------------- timing

class MonotonicTimer:
    """Records event times in seconds relative to a monotonic start into a Timing.

    Use one timer per HTTP attempt. ``t_start`` is captured from
    ``time.monotonic()`` at construction; ``wall_start`` stores the absolute epoch
    time (for cross-request drift/load analysis).
    """

    def __init__(self, timing: Timing | None = None) -> None:
        self.timing = timing if timing is not None else Timing()
        self.t_start = time.monotonic()
        if self.timing.wall_start is None:
            self.timing.wall_start = time.time()

    def elapsed(self) -> float:
        """Seconds since the timer started."""
        return time.monotonic() - self.t_start

    def mark_first_byte(self) -> float:
        """Record the moment response headers were received (first byte)."""
        t = self.elapsed()
        if self.timing.t_first_byte is None:
            self.timing.t_first_byte = t
        return t

    def mark_end(self) -> float:
        """Record the moment the full response was received."""
        self.timing.t_end = self.elapsed()
        return self.timing.t_end

    def add_chunk(self, text: str, n_tokens: int | None = None,
                  t: float | None = None) -> Chunk:
        """Append a streamed chunk and update first-token time if it has content."""
        at = self.elapsed() if t is None else t
        chunk = Chunk(t=at, text=text, n_tokens=n_tokens)
        self.timing.chunks.append(chunk)
        if text and self.timing.t_first_token is None:
            self.timing.t_first_token = at
        return chunk


# --------------------------------------------------------------------------- server-timing

#: Vendor headers that carry server-side processing time (value in milliseconds
#: unless a unit suffix says otherwise).
_SERVER_TIME_HEADERS: tuple[str, ...] = (
    "openai-processing-ms",
    "x-envoy-upstream-service-time",
    "x-processing-time",
    "x-response-time",
    "x-total-time-ms",
    "x-total-time",
)


def _parse_ms(value: str) -> float | None:
    """Parse a duration string into milliseconds (handles ``ms`` / ``s`` suffixes)."""
    s = str(value).strip().lower()
    try:
        if s.endswith("ms"):
            return float(s[:-2])
        if s.endswith("s"):
            return float(s[:-1]) * 1000.0
        return float(s)
    except ValueError:
        return None


def parse_server_timing(headers: Mapping[str, str]) -> float | None:
    """Extract server-side processing time (milliseconds) from response headers.

    Understands the standard ``Server-Timing`` header (sums ``dur=`` metrics, or
    prefers a ``total`` metric) plus common vendor headers. Returns None if no
    server time is exposed.
    """
    h = {str(k).lower(): v for k, v in dict(headers).items()}

    st = h.get("server-timing")
    if st:
        total = re.search(r"total[^,]*?dur=([0-9.]+)", st, re.IGNORECASE)
        if total:
            return float(total.group(1))
        durs = re.findall(r"dur=([0-9.]+)", st)
        if durs:
            return sum(float(d) for d in durs)

    for name in _SERVER_TIME_HEADERS:
        if name in h:
            ms = _parse_ms(h[name])
            if ms is not None:
                return ms
    return None


# --------------------------------------------------------------------------- client factory

def make_client(
    base_url: str = "",
    *,
    timeout: float = 60.0,
    max_keepalive_connections: int = 8,
    max_connections: int = 16,
    http2: bool = False,
    headers: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Create a shared ``AsyncClient`` with keep-alive tuned for timing work.

    Reusing one client (one connection pool) keeps TCP/TLS handshake and TCP
    slow-start out of per-request latency, which matters for the timing probes.
    """
    limits = httpx.Limits(
        max_keepalive_connections=max_keepalive_connections,
        max_connections=max_connections,
    )
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        limits=limits,
        http2=http2,
        headers=dict(headers) if headers else None,
        **kwargs,
    )


# --------------------------------------------------------------------------- helpers

def _build_kwargs(headers: Mapping[str, str] | None, json_body: Any,
                  timeout: float | None) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if headers:
        kw["headers"] = dict(headers)
    if json_body is not None:
        kw["json"] = json_body
    if timeout is not None:
        kw["timeout"] = timeout
    return kw


def _decode_body(raw: bytes, content_type: str) -> tuple[Any, str | None]:
    """Best-effort decode of a response body. Returns (body, decode_error)."""
    if not raw:
        return None, None
    text = raw.decode("utf-8", errors="replace")
    ct = (content_type or "").lower()
    looks_json = "json" in ct or text.lstrip()[:1] in "{["
    if looks_json:
        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            return text, f"json decode error: {e}"
    return text, None


def _error_text(status: int, body: Any) -> str:
    """Build a compact error string from a >=400 response body."""
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            msg = err.get("message") or err.get("code") or json.dumps(err)[:400]
        else:
            msg = str(err)[:400]
    elif body is None:
        msg = ""
    else:
        msg = str(body)[:400]
    return f"HTTP {status}: {msg}".rstrip(": ")


# --------------------------------------------------------------------------- request_json

async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Mapping[str, str] | None = None,
    json_body: Any = None,
    timeout: float | None = 60.0,
    *,
    policy: RetryPolicy | None = None,
    rng: random.Random | None = None,
) -> tuple[int | None, dict[str, str], Any, Timing, str | None]:
    """Perform a JSON request, returning ``(status, headers, body, timing, error)``.

    Never raises on HTTP or transport errors: a failed request yields
    ``error`` set to a human-readable string (and ``status`` None for transport
    failures). Retries 429/5xx and transient transport errors per ``policy``,
    honouring ``Retry-After``. Timing is measured monotonically; the returned
    Timing belongs to the final attempt.
    """
    policy = policy or RetryPolicy()
    timer = MonotonicTimer()
    last_error: str | None = None

    for attempt in range(policy.max_retries + 1):
        timer = MonotonicTimer()
        try:
            request = client.build_request(
                method.upper(), url, **_build_kwargs(headers, json_body, timeout)
            )
            resp = await client.send(request, stream=True)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            timer.mark_end()
            if attempt < policy.max_retries:
                await asyncio.sleep(_backoff_delay(attempt, None, policy, rng))
                continue
            return None, {}, None, timer.timing, last_error

        timer.mark_first_byte()
        try:
            raw = await resp.aread()
        finally:
            await resp.aclose()
        timer.mark_end()

        status = resp.status_code
        resp_headers = {k: v for k, v in resp.headers.items()}
        timer.timing.server_ms = parse_server_timing(resp.headers)
        body, decode_err = _decode_body(raw, resp.headers.get("content-type", ""))

        if status in policy.statuses and attempt < policy.max_retries:
            last_error = _error_text(status, body)
            delay = _backoff_delay(attempt, _parse_retry_after(resp.headers), policy, rng)
            await asyncio.sleep(delay)
            continue

        error: str | None = None
        if status >= 400:
            error = _error_text(status, body)
        elif decode_err is not None:
            error = decode_err
        return status, resp_headers, body, timer.timing, error

    # Exhausted retries on retryable statuses.
    return None, {}, None, timer.timing, last_error


# --------------------------------------------------------------------------- SSE streaming

def _sse_delta_text(obj: Any) -> str:
    """Best-effort extraction of the visible text from one OpenAI-style SSE object."""
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
            txt = first.get("text")
            if isinstance(txt, str):
                return txt
    return ""


async def stream_sse(
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str] | None = None,
    json_body: Any = None,
    timeout: float | None = 120.0,
    *,
    method: str = "POST",
    policy: RetryPolicy | None = None,
    timing: Timing | None = None,
    rng: random.Random | None = None,
) -> AsyncIterator[tuple[Any, float]]:
    """Stream a Server-Sent-Events response, yielding ``(data_json, arrival_t)``.

    Each yielded tuple is the parsed JSON of one ``data:`` line and its arrival
    time (seconds since request start). The provided ``timing`` (created if None
    and accessible to the caller via the ``timing`` argument) is filled with
    ``t_first_byte``, ``t_first_token``, per-chunk arrival times and ``t_end``.

    Retries only the initial connection (before any data is yielded); once the
    stream is flowing a mid-stream failure ends iteration. Non-2xx responses end
    the stream without yielding (inspect ``timing`` / send a probe request via
    :func:`request_json` to capture the error body).
    """
    policy = policy or RetryPolicy()
    if timing is None:
        timing = Timing()

    for attempt in range(policy.max_retries + 1):
        timer = MonotonicTimer(timing)
        try:
            request = client.build_request(
                method.upper(), url, **_build_kwargs(headers, json_body, timeout)
            )
            resp = await client.send(request, stream=True)
        except (httpx.TimeoutException, httpx.TransportError):
            timer.mark_end()
            if attempt < policy.max_retries:
                await asyncio.sleep(_backoff_delay(attempt, None, policy, rng))
                continue
            return

        timer.mark_first_byte()

        if resp.status_code in policy.statuses and attempt < policy.max_retries:
            retry_after = _parse_retry_after(resp.headers)
            await resp.aclose()
            await asyncio.sleep(_backoff_delay(attempt, retry_after, policy, rng))
            continue

        timing.server_ms = parse_server_timing(resp.headers)

        if resp.status_code >= 400:
            await resp.aread()
            await resp.aclose()
            timer.mark_end()
            return

        try:
            async for line in resp.aiter_lines():
                arrival = timer.elapsed()
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                timer.add_chunk(_sse_delta_text(obj), t=arrival)
                yield obj, arrival
        finally:
            await resp.aclose()
            timer.mark_end()
        return
