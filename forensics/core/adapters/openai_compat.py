"""OpenAI-v1-compatible generative adapter (Azure ``.../openai/v1`` and friends).

This is the real network adapter used against the live Azure endpoint surveyed in
``research/azure_capabilities.md``.  A single class serves every family on that
endpoint (classic Azure OpenAI, the gpt-5.x/6 reasoning line, Foundry MaaS
partner models such as DeepSeek/Kimi, and the Fireworks-backed ``FW-`` models)
because it discovers each model's quirks *adaptively* instead of hard-coding a
table:

* it builds a normal ``/chat/completions`` body from a :class:`GenRequest`;
* if the request 400s with a machine-parsable complaint (``max_tokens`` vs
  ``max_completion_tokens``; ``top_logprobs`` cap; unsupported
  ``temperature`` / ``top_p`` / ``logit_bias`` / ``logprobs``; a
  ``reasoning_effort`` "Supported values are: ..." list; or "the requested
  operation is unsupported", meaning the model is ``/responses``-only), it
  **parses the error, adjusts, and retries**;
* every learned constraint is cached per-model on the instance, so later calls
  skip the negotiation entirely.

Like every adapter it never raises on an API/transport error: a failed call
comes back as a :class:`GenResponse` with ``error``/``status`` set (error texts
are forensic evidence).  Timing, per-token logprobs (with byte lists and, for
Fireworks models, ``token_id``), usage detail (cached / reasoning tokens, and
the full raw usage including Azure ``latency_checkpoint``) and streaming
chunk-arrival times are all captured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from forensics.core.adapters.base import Adapter
from forensics.core.httpclient import make_client, request_json, stream_sse
from forensics.core.types import (
    Capabilities, GenRequest, GenResponse, Message, Timing, TokenLogprob,
    TopToken, Usage,
)

#: Max adaptation rounds for a single request (enough for several stacked quirks).
_MAX_ADAPT = 6

#: Ordering of reasoning-effort levels, low→high, for choosing a nearest allowed value.
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh")

#: Body keys we know how to strip when a family rejects them generically.
_KNOWN_PARAMS = (
    "temperature", "top_p", "logit_bias", "logprobs", "top_logprobs", "seed",
    "n", "stop", "reasoning_effort", "presence_penalty", "frequency_penalty",
)


# --------------------------------------------------------------------------- constraints
@dataclass
class _ModelConstraints:
    """Per-model quirks learned from 400 errors and cached on the adapter."""

    max_tokens_param: str = "max_tokens"          # or "max_completion_tokens"
    use_responses: bool = False                   # family D: route to /responses
    max_top_logprobs: int | None = None           # numeric cap parsed from the error
    allowed_reasoning_efforts: list[str] | None = None
    dropped_params: set[str] = field(default_factory=set)   # chat body keys to strip
    responses_drop_logprobs: bool = False         # /responses: no include-logprobs
    resp_drop_temperature: bool = False           # /responses: no temperature

    def snapshot(self) -> tuple[Any, ...]:
        """Hashable view used to detect whether an adaptation made progress."""
        return (
            self.max_tokens_param, self.use_responses, self.max_top_logprobs,
            tuple(self.allowed_reasoning_efforts or ()),
            tuple(sorted(self.dropped_params)),
            self.responses_drop_logprobs, self.resp_drop_temperature,
        )


# --------------------------------------------------------------------------- helpers
def _lower_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


def _flt(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _negative(low: str) -> bool:
    """True if the error text signals a rejected/unsupported field."""
    return any(s in low for s in (
        "unsupported", "not supported", "unrecognized", "does not support",
        "not permitted", "unknown parameter", "only the default", "not allowed",
    ))


# --------------------------------------------------------------------------- adapter
class OpenAICompatAdapter(Adapter):
    """Adaptive adapter for OpenAI-v1-compatible chat / responses endpoints."""

    kind = "generative"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        auth_style: str = "api-key",
        api_version: str | None = None,
        extra_headers: dict[str, str] | None = None,
        name: str | None = None,
        *,
        timeout: float = 120.0,
    ) -> None:
        super().__init__()
        if auth_style not in ("api-key", "bearer"):
            raise ValueError(f"auth_style must be 'api-key' or 'bearer', got {auth_style!r}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.auth_style = auth_style
        self.api_version = api_version
        self.extra_headers = dict(extra_headers or {})
        self.name = name or f"openai-compat:{model}"
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._constraints: dict[str, _ModelConstraints] = {}

    # ------------------------------------------------------------------ plumbing
    def _con(self) -> _ModelConstraints:
        con = self._constraints.get(self.model)
        if con is None:
            con = _ModelConstraints()
            self._constraints[self.model] = con
        return con

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"content-type": "application/json"}
        if self.auth_style == "bearer":
            h["authorization"] = f"Bearer {self.api_key}"
        else:
            h["api-key"] = self.api_key
        h.update(self.extra_headers)
        return h

    def _url(self, path: str) -> str:
        url = f"{self.base_url}/{path}"
        if self.api_version:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api-version={self.api_version}"
        return url

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = make_client(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(base_url=self.base_url, model=self.model, auth_style=self.auth_style)
        return d

    # ------------------------------------------------------------------ effort pick
    @staticmethod
    def _pick_effort(requested: str, allowed: list[str] | None) -> str:
        if not allowed or requested in allowed:
            return requested
        if requested in _EFFORT_ORDER:
            ri = _EFFORT_ORDER.index(requested)
            ranked = [a for a in allowed if a in _EFFORT_ORDER]
            if ranked:
                return min(ranked, key=lambda a: abs(_EFFORT_ORDER.index(a) - ri))
        return allowed[0]

    # ================================================================== body building
    def _messages_payload(self, req: GenRequest) -> list[dict[str, str]]:
        if req.messages is not None:
            return [{"role": m.role, "content": m.content} for m in req.messages]
        if req.prompt is not None:
            return [{"role": "user", "content": req.prompt}]
        return [{"role": "user", "content": ""}]

    def _build_chat_body(self, req: GenRequest, con: _ModelConstraints) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model,
                                "messages": self._messages_payload(req)}
        body[con.max_tokens_param] = req.max_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.seed is not None:
            body["seed"] = req.seed
        if req.n and req.n != 1:
            body["n"] = req.n
        if req.stop:
            body["stop"] = req.stop
        if req.logit_bias:
            body["logit_bias"] = {str(k): float(v) for k, v in req.logit_bias.items()}
        if req.logprobs:
            body["logprobs"] = True
            if req.top_logprobs is not None:
                tl = req.top_logprobs
                if con.max_top_logprobs is not None:
                    tl = min(tl, con.max_top_logprobs)
                body["top_logprobs"] = tl
        if req.reasoning_effort:
            body["reasoning_effort"] = self._pick_effort(
                req.reasoning_effort, con.allowed_reasoning_efforts)
        if req.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        if req.extra:
            body.update(req.extra)
        for p in con.dropped_params:
            body.pop(p, None)
        return body

    def _build_responses_body(self, req: GenRequest, con: _ModelConstraints) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model}
        if req.messages is not None:
            body["input"] = [{"role": m.role, "content": m.content} for m in req.messages]
        else:
            body["input"] = req.prompt if req.prompt is not None else ""
        body["max_output_tokens"] = req.max_tokens
        if req.temperature is not None and not con.resp_drop_temperature:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.seed is not None:
            body["seed"] = req.seed
        if req.reasoning_effort:
            body["reasoning"] = {
                "effort": self._pick_effort(req.reasoning_effort, con.allowed_reasoning_efforts)
            }
        if req.logprobs and not con.responses_drop_logprobs:
            body["include"] = ["message.output_text.logprobs"]
            if req.top_logprobs is not None:
                tl = req.top_logprobs
                if con.max_top_logprobs is not None:
                    tl = min(tl, con.max_top_logprobs)
                body["top_logprobs"] = tl
        if req.extra:
            body.update(req.extra)
        for p in con.dropped_params:
            body.pop(p, None)
        return body

    # ================================================================== error parsing
    def _mark_drop(self, con: _ModelConstraints, param: str) -> bool:
        param = param.strip().strip("'\"")
        if not param or param == "max_tokens":  # max_tokens handled by param swap
            return False
        changed = False
        if param == "logprobs":
            for k in ("logprobs", "top_logprobs"):
                if k not in con.dropped_params:
                    con.dropped_params.add(k)
                    changed = True
        elif param not in con.dropped_params:
            con.dropped_params.add(param)
            changed = True
        return changed

    def _adapt_chat(self, err: str, con: _ModelConstraints) -> bool:
        """Mutate ``con`` from a chat 400 error; return True if something changed."""
        low = err.lower()

        # Family D: /chat/completions unsupported -> switch to /responses.
        if "operation is unsupported" in low or ("unsupported" in low and "operation" in low):
            if not con.use_responses:
                con.use_responses = True
                return True
            return False

        changed = False

        # max_tokens -> max_completion_tokens.
        if (con.max_tokens_param == "max_tokens" and "max_tokens" in low
                and ("max_completion_tokens" in low or _negative(low))):
            con.max_tokens_param = "max_completion_tokens"
            changed = True

        # top_logprobs numeric cap (kept: logprobs still allowed).
        if "top_logprobs" in low or ("logprob" in low and "less than or equal" in low):
            m = (re.search(r"less than or equal to (\d+)", low)
                 or re.search(r"between \d+ and (\d+)", low)
                 or re.search(r"at most (\d+)", low))
            if m:
                cap = int(m.group(1))
                if con.max_top_logprobs is None or cap < con.max_top_logprobs:
                    con.max_top_logprobs = cap
                    changed = True

        # logprobs unsupported outright (reasoning-class models).
        if ("logprob" in low and _negative(low)
                and not any(s in low for s in ("less than or equal", "between", "at most"))):
            changed |= self._mark_drop(con, "logprobs")

        # reasoning_effort: learn the allowed value list, or drop if unrecognised.
        if "reasoning_effort" in low or "reasoning effort" in low:
            idx = low.find("supported values are")
            if idx >= 0:
                vals = [v for v in re.findall(r"'([^']+)'", err[idx:]) if v]
                if vals and con.allowed_reasoning_efforts != vals:
                    con.allowed_reasoning_efforts = vals
                    changed = True
            elif _negative(low):
                changed |= self._mark_drop(con, "reasoning_effort")

        # temperature / top_p / logit_bias explicit rejections.
        if "temperature" in low and (_negative(low) or "unsupported value" in low):
            changed |= self._mark_drop(con, "temperature")
        if "top_p" in low and _negative(low):
            changed |= self._mark_drop(con, "top_p")
        if "logit_bias" in low and (_negative(low) or "rejected" in low):
            changed |= self._mark_drop(con, "logit_bias")

        # Generic quoted "unsupported parameter: 'x'" fallback.
        for m in re.finditer(
            r"(?:unsupported parameter|unknown parameter|unrecognized request argument"
            r"|unsupported value)[:\s]+'?([a-z_][a-z0-9_]*)'?", low
        ):
            changed |= self._mark_drop(con, m.group(1))
        return changed

    def _adapt_responses(self, err: str, con: _ModelConstraints) -> bool:
        low = err.lower()
        changed = False
        if "logprob" in low and _negative(low):
            if not con.responses_drop_logprobs:
                con.responses_drop_logprobs = True
                changed = True
        if "temperature" in low and (_negative(low) or "unsupported value" in low):
            if not con.resp_drop_temperature:
                con.resp_drop_temperature = True
                changed = True
        if ("reasoning" in low or "effort" in low) and "supported values are" in low:
            idx = low.find("supported values are")
            vals = [v for v in re.findall(r"'([^']+)'", err[idx:]) if v]
            if vals and con.allowed_reasoning_efforts != vals:
                con.allowed_reasoning_efforts = vals
                changed = True
        if "top_logprobs" in low:
            m = (re.search(r"less than or equal to (\d+)", low)
                 or re.search(r"between \d+ and (\d+)", low))
            if m:
                cap = int(m.group(1))
                if con.max_top_logprobs is None or cap < con.max_top_logprobs:
                    con.max_top_logprobs = cap
                    changed = True
        for m in re.finditer(
            r"(?:unsupported parameter|unknown parameter|unrecognized request argument)"
            r"[:\s]+'?([a-z_][a-z0-9_]*)'?", low
        ):
            p = m.group(1)
            if p and p not in con.dropped_params:
                con.dropped_params.add(p)
                changed = True
        return changed

    # ================================================================== generate
    async def generate(self, req: GenRequest) -> GenResponse:
        con = self._con()
        if con.use_responses:
            return await self._responses_flow(req, con)
        return await self._chat_flow(req, con)

    async def _chat_flow(self, req: GenRequest, con: _ModelConstraints) -> GenResponse:
        resp: GenResponse = GenResponse(error="no attempt made")
        for _ in range(_MAX_ADAPT + 1):
            body = self._build_chat_body(req, con)
            if req.stream:
                resp = await self._chat_stream(body)
                if resp.ok and (resp.timing.chunks or resp.text
                                or resp.usage.prompt_tokens is not None):
                    return resp
                # Empty stream => probe non-streaming to surface the real error/body.
                probe_body = dict(body)
                probe_body.pop("stream", None)
                probe_body.pop("stream_options", None)
                probe = await self._chat_once(probe_body)
                if probe.ok:
                    return probe  # transient streaming glitch; return valid result
                resp = probe
            else:
                resp = await self._chat_once(body)
                if resp.ok:
                    return resp

            if resp.status != 400 or not resp.error:
                return resp
            before = con.snapshot()
            self._adapt_chat(resp.error, con)
            if con.use_responses:
                return await self._responses_flow(req, con)
            if con.snapshot() == before:
                return resp  # nothing more we can adjust
        return resp

    async def _responses_flow(self, req: GenRequest, con: _ModelConstraints) -> GenResponse:
        resp: GenResponse = GenResponse(error="no attempt made")
        for _ in range(_MAX_ADAPT + 1):
            body = self._build_responses_body(req, con)
            resp = await self._responses_once(body)
            if resp.ok:
                return resp
            if resp.status != 400 or not resp.error:
                return resp
            before = con.snapshot()
            self._adapt_responses(resp.error, con)
            if con.snapshot() == before:
                return resp
        return resp

    # ------------------------------------------------------------------ chat HTTP
    async def _chat_once(self, body: dict[str, Any]) -> GenResponse:
        client = await self._get_client()
        status, headers, rbody, timing, error = await request_json(
            client, "POST", self._url("chat/completions"),
            headers=self._headers(), json_body=body, timeout=self.timeout,
        )
        return self._parse_chat(status, headers, rbody, timing, error)

    def _parse_chat(self, status: int | None, headers: Any, body: Any,
                    timing: Timing, error: str | None) -> GenResponse:
        resp = GenResponse(status=status, timing=timing, raw=body,
                           headers=_lower_headers(headers))
        if error is not None:
            resp.error = error
            resp.error_body = body
            return resp
        if not isinstance(body, dict):
            resp.error = f"unexpected chat response body: {type(body).__name__}"
            resp.error_body = body
            return resp

        resp.model_reported = body.get("model")
        resp.system_fingerprint = body.get("system_fingerprint")
        choices = body.get("choices") if isinstance(body.get("choices"), list) else []
        texts: list[str] = []
        for ch in choices:
            if not isinstance(ch, dict):
                texts.append("")
                continue
            msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
            content = msg.get("content")
            if content is None:
                content = ch.get("text")
            texts.append(content or "")
        resp.choices_text = texts
        resp.text = texts[0] if texts else ""
        if choices and isinstance(choices[0], dict):
            resp.finish_reason = choices[0].get("finish_reason")
            resp.tokens = self._parse_logprobs(choices[0].get("logprobs"))
        resp.usage = self._parse_usage(body.get("usage"))
        self._apply_server_timing(resp)
        return resp

    # ------------------------------------------------------------------ chat streaming
    async def _chat_stream(self, body: dict[str, Any]) -> GenResponse:
        client = await self._get_client()
        timing = Timing()
        resp = GenResponse(timing=timing)
        raw_chunks: list[Any] = []
        choice_parts: dict[int, list[str]] = {}
        logprob_content: list[Any] = []
        usage_obj: dict[str, Any] | None = None
        finish: str | None = None
        model: str | None = None
        sysfp: str | None = None
        try:
            async for obj, _t in stream_sse(
                client, self._url("chat/completions"), headers=self._headers(),
                json_body=body, timeout=self.timeout, timing=timing,
            ):
                raw_chunks.append(obj)
                if not isinstance(obj, dict):
                    continue
                if obj.get("model"):
                    model = obj["model"]
                if obj.get("system_fingerprint"):
                    sysfp = obj["system_fingerprint"]
                u = obj.get("usage")
                if isinstance(u, dict):
                    usage_obj = dict(u)
                    lc = obj.get("latency_checkpoint")
                    if lc is not None and "latency_checkpoint" not in usage_obj:
                        usage_obj["latency_checkpoint"] = lc
                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue  # Azure prompt-filter first chunk / final usage chunk
                for ch in choices:
                    if not isinstance(ch, dict):
                        continue
                    idx = ch.get("index", 0)
                    delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
                    piece = delta.get("content")
                    if piece is None:
                        piece = ch.get("text")
                    if piece:
                        choice_parts.setdefault(idx, []).append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    lp = ch.get("logprobs")
                    if idx == 0 and isinstance(lp, dict) and isinstance(lp.get("content"), list):
                        logprob_content.extend(lp["content"])
        except (httpx.HTTPError, ValueError) as exc:
            resp.error = f"{type(exc).__name__}: {exc}"

        if choice_parts:
            hi = max(choice_parts)
            resp.choices_text = ["".join(choice_parts.get(i, [])) for i in range(hi + 1)]
        resp.text = resp.choices_text[0] if resp.choices_text else ""
        resp.finish_reason = finish
        resp.model_reported = model
        resp.system_fingerprint = sysfp
        resp.usage = self._parse_usage(usage_obj)
        resp.raw = raw_chunks
        if logprob_content:
            resp.tokens = self._parse_logprobs({"content": logprob_content})
        self._apply_server_timing(resp)
        if resp.error is None:
            resp.status = 200
        return resp

    # ------------------------------------------------------------------ responses HTTP
    async def _responses_once(self, body: dict[str, Any]) -> GenResponse:
        client = await self._get_client()
        status, headers, rbody, timing, error = await request_json(
            client, "POST", self._url("responses"),
            headers=self._headers(), json_body=body, timeout=self.timeout,
        )
        return self._parse_responses(status, headers, rbody, timing, error)

    def _parse_responses(self, status: int | None, headers: Any, body: Any,
                         timing: Timing, error: str | None) -> GenResponse:
        resp = GenResponse(status=status, timing=timing, raw=body,
                           headers=_lower_headers(headers))
        if error is not None:
            resp.error = error
            resp.error_body = body
            return resp
        if not isinstance(body, dict):
            resp.error = f"unexpected responses body: {type(body).__name__}"
            resp.error_body = body
            return resp

        resp.model_reported = body.get("model")
        text_parts: list[str] = []
        tokens: list[TokenLogprob] | None = None
        output = body.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for c in item.get("content") or []:
                    if not isinstance(c, dict) or c.get("type") != "output_text":
                        continue
                    text_parts.append(c.get("text") or "")
                    lp = c.get("logprobs")
                    if isinstance(lp, list) and tokens is None:
                        tokens = self._parse_logprobs({"content": lp})
        if not text_parts and isinstance(body.get("output_text"), str):
            text_parts.append(body["output_text"])
        resp.text = "".join(text_parts)
        resp.choices_text = [resp.text]
        resp.tokens = tokens
        st = body.get("status")
        if st == "incomplete":
            det = body.get("incomplete_details") or {}
            resp.finish_reason = det.get("reason") if isinstance(det, dict) else "incomplete"
        elif st == "completed":
            resp.finish_reason = "stop"
        elif isinstance(st, str):
            resp.finish_reason = st
        resp.usage = self._parse_usage(body.get("usage"))
        self._apply_server_timing(resp)
        return resp

    # ================================================================== parsing helpers
    def _parse_logprobs(self, lp: Any) -> list[TokenLogprob] | None:
        if not isinstance(lp, dict):
            return None
        content = lp.get("content")
        if not isinstance(content, list):
            return None
        out: list[TokenLogprob] = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            top: list[TopToken] = []
            for t in entry.get("top_logprobs") or []:
                if not isinstance(t, dict):
                    continue
                top.append(TopToken(
                    token=t.get("token", ""),
                    logprob=_flt(t.get("logprob")),
                    bytes=t.get("bytes"),
                    token_id=t.get("token_id"),
                ))
            out.append(TokenLogprob(
                token=entry.get("token", ""),
                logprob=_flt(entry.get("logprob")),
                bytes=entry.get("bytes"),
                token_id=entry.get("token_id"),
                top=top,
            ))
        return out or None

    def _parse_usage(self, u: Any) -> Usage:
        if not isinstance(u, dict):
            return Usage()
        ptd = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
        ctd = u.get("completion_tokens_details") if isinstance(u.get("completion_tokens_details"), dict) else {}
        itd = u.get("input_tokens_details") if isinstance(u.get("input_tokens_details"), dict) else {}
        otd = u.get("output_tokens_details") if isinstance(u.get("output_tokens_details"), dict) else {}

        prompt = u.get("prompt_tokens")
        if prompt is None:
            prompt = u.get("input_tokens")
        completion = u.get("completion_tokens")
        if completion is None:
            completion = u.get("output_tokens")

        cached = ptd.get("cached_tokens")
        if cached is None:
            cached = itd.get("cached_tokens")
        reasoning = ctd.get("reasoning_tokens")
        if reasoning is None:
            reasoning = otd.get("reasoning_tokens")
        if reasoning is None:
            reasoning = u.get("reasoning_tokens")  # MaaS stream final usage

        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=u.get("total_tokens"),
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            raw=dict(u),  # keep everything, incl. Azure latency_checkpoint
        )

    def _apply_server_timing(self, resp: GenResponse) -> None:
        """Prefer Azure engine timing (usage.latency_checkpoint) over header time."""
        raw = resp.usage.raw if resp.usage else None
        lc = raw.get("latency_checkpoint") if isinstance(raw, dict) else None
        if isinstance(lc, dict):
            for k in ("engine_ttlt_ms", "service_ttlt_ms"):
                v = lc.get(k)
                if isinstance(v, (int, float)):
                    resp.timing.server_ms = float(v)
                    return

    # ================================================================== embeddings
    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        status, _headers, rbody, _timing, error = await request_json(
            client, "POST", self._url("embeddings"),
            headers=self._headers(),
            json_body={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        if error is not None:
            raise RuntimeError(f"embed() failed (status {status}): {error}")
        if not isinstance(rbody, dict) or not isinstance(rbody.get("data"), list):
            raise RuntimeError("embed() got an unexpected response body")
        rows = sorted(rbody["data"], key=lambda d: d.get("index", 0))
        return [list(d.get("embedding") or []) for d in rows]

    # ================================================================== capabilities
    def static_capabilities(self) -> Capabilities:
        con = self._constraints.get(self.model)
        return Capabilities(
            kind="generative",
            chat=True,
            completion=None,
            logprobs=True,
            max_top_logprobs=(con.max_top_logprobs if con and con.max_top_logprobs else 20),
            logprob_bytes=True,
            logit_bias=True,
            logit_bias_affects_logprobs=None,
            logit_bias_max=100.0,
            streaming=True,
            stream_usage=True,
            temperature=True,
            top_p=True,
            seed=True,
            n=True,
            stop=True,
            system_role=True,
            assistant_prefill=None,
            usage_prompt_tokens=True,
            usage_cached_tokens=True,
            usage_reasoning_tokens=None,
            reasoning_effort=None,
            system_fingerprint=None,
            server_timing_header=None,
            max_tokens_param=(con.max_tokens_param if con else "max_tokens"),
            token_ids_known=None,
            notes=[f"OpenAI-compat endpoint; model={self.model}; auth={self.auth_style}"],
        )


# --------------------------------------------------------------------------- self-test
async def _self_test() -> int:
    """Small live smoke test against the Azure endpoint in ``.env``. Returns exit code."""
    import os

    from forensics.config import load_env

    load_env()
    base = os.environ["AZURE_OPENAI_BASE_URL"]
    key = os.environ["AZURE_OPENAI_API_KEY"]

    def mk(model: str) -> OpenAICompatAdapter:
        return OpenAICompatAdapter(base, key, model, auth_style="api-key")

    results: list[str] = []
    ok = True

    # 1) gpt-4o-mini: logprobs top 5, 8 tokens -> tokens with top lists.
    a1 = mk("gpt-4o-mini")
    r1 = await a1.generate(GenRequest.chat(
        "Reply with exactly one word: the capital of France.",
        max_tokens=8, logprobs=True, top_logprobs=5, temperature=0.0))
    await a1.aclose()
    assert r1.ok, f"gpt-4o-mini failed: {r1.error}"
    assert r1.tokens, "gpt-4o-mini returned no per-token logprobs"
    assert r1.tokens[0].top, "gpt-4o-mini token has no top list"
    results.append(f"gpt-4o-mini={r1.model_reported} tok0={r1.tokens[0].token!r} "
                   f"top={len(r1.tokens[0].top)}")

    # 2) gpt-5.4-nano: should auto-swap max_tokens -> max_completion_tokens.
    a2 = mk("gpt-5.4-nano")
    r2 = await a2.generate(GenRequest.chat(
        "Reply with exactly one word: the capital of France.", max_tokens=16))
    swap = a2._constraints["gpt-5.4-nano"].max_tokens_param
    await a2.aclose()
    assert r2.ok, f"gpt-5.4-nano failed: {r2.error}"
    assert swap == "max_completion_tokens", f"expected token-param swap, got {swap}"
    results.append(f"gpt-5.4-nano={r2.model_reported} param={swap}")

    # 3) DeepSeek-V3.2: tiny logit_bias + logprobs -> ok.
    a3 = mk("DeepSeek-V3.2")
    r3 = await a3.generate(GenRequest.chat(
        "Reply with exactly one word: the capital of France.",
        max_tokens=8, logprobs=True, top_logprobs=5, logit_bias={1000: 5.0},
        temperature=0.0))
    await a3.aclose()
    assert r3.ok, f"DeepSeek-V3.2 failed: {r3.error}"
    assert r3.tokens, "DeepSeek-V3.2 returned no per-token logprobs"
    results.append(f"DeepSeek-V3.2={r3.model_reported} tok0={r3.tokens[0].token!r} "
                   f"lb_ok=1")

    # 4) streaming gpt-4o-mini: chunks + ttft.
    a4 = mk("gpt-4o-mini")
    r4 = await a4.generate(GenRequest.chat(
        "Count: one two three four five.", max_tokens=16, stream=True))
    await a4.aclose()
    assert r4.timing.chunks, "streaming returned no chunks"
    assert r4.timing.ttft is not None, "streaming ttft not set"
    results.append(f"stream={r4.model_reported} chunks={len(r4.timing.chunks)} "
                   f"ttft={r4.timing.ttft:.3f}s")

    print("PASS | " + " | ".join(results))
    return 0 if ok else 1


if __name__ == "__main__":
    import asyncio
    import sys

    sys.exit(asyncio.run(_self_test()))
