"""Tests for the run infrastructure: Recorder, Budget and httpclient.

Uses httpx.MockTransport so the HTTP tests are hermetic (no network). asyncio
tests run without a marker because pyproject sets ``asyncio_mode = "auto"``.
"""
from __future__ import annotations

import json

import httpx

from forensics.core.budget import Budget
from forensics.core.httpclient import (
    MonotonicTimer,
    RetryPolicy,
    parse_server_timing,
    request_json,
    stream_sse,
)
from forensics.core.recorder import Recorder
from forensics.core.types import (
    DecisionResponse,
    GenRequest,
    GenResponse,
    Timing,
    TokenLogprob,
    Usage,
)


# --------------------------------------------------------------------------- Recorder
def test_recorder_roundtrip_and_redaction(tmp_path):
    rec = Recorder(str(tmp_path))

    req = GenRequest.chat("hello")
    resp1 = GenResponse(
        text="world",
        usage=Usage(prompt_tokens=3, completion_tokens=1),
        headers={"Authorization": "Bearer sk-super-secret", "content-type": "application/json"},
        tokens=[TokenLogprob(token="world", logprob=-0.12)],
    )
    resp2 = GenResponse(text="!", headers={"api-key": "sk-two"})
    rec.record("generate", req, resp1, probe="p01")
    rec.record("generate", req, resp2, probe="p01")

    rec.meta({"target": {"api_key": "sk-xyz", "base_url": "http://example"}, "note": "run"})

    obs = list(Recorder.iter_observations(str(tmp_path)))
    assert len(obs) == 2
    assert obs[0]["kind"] == "generate" and obs[0]["probe"] == "p01"

    # Secret header redacted, non-secret header preserved.
    assert obs[0]["response"]["headers"]["Authorization"] == "***"
    assert obs[0]["response"]["headers"]["content-type"] == "application/json"
    assert obs[1]["response"]["headers"]["api-key"] == "***"

    # Forensic payload (token string) must NOT be redacted despite "token" marker.
    assert obs[0]["response"]["tokens"][0]["token"] == "world"

    meta = Recorder.read_meta(str(tmp_path))
    assert meta["target"]["api_key"] == "***"
    assert meta["target"]["base_url"] == "http://example"
    assert meta["note"] == "run"


# --------------------------------------------------------------------------- Budget
def test_budget_accounting_and_gating():
    b = Budget(max_requests=3, price_in_per_m=1.0, price_out_per_m=2.0)
    req = GenRequest.chat("hi there")

    assert b.can_spend(req) is True

    # prompt_tokens / completion_tokens naming
    b.charge(req, GenResponse(usage=Usage(prompt_tokens=100, completion_tokens=50)))
    # input_tokens / output_tokens via usage.raw fallback (alternate naming)
    b.charge(req, GenResponse(usage=Usage(raw={"input_tokens": 200, "output_tokens": 25})))
    # DecisionResponse with standard usage
    b.charge(req, DecisionResponse(usage=Usage(prompt_tokens=10, completion_tokens=5)))

    s = b.summary()
    assert s["requests"] == 3
    assert s["input_tokens"] == 310
    assert s["output_tokens"] == 80
    # cost = 310/1e6 * 1.0 + 80/1e6 * 2.0
    assert abs(s["est_cost_usd"] - (310e-6 + 160e-6)) < 1e-12
    assert s["remaining"]["requests"] == 0

    # Request ceiling reached -> no more spending.
    assert b.can_spend(req) is False


def test_budget_soft_input_token_gating():
    b = Budget(max_input_tokens=5)
    small = GenRequest.chat("hi")            # ~1 token estimate
    big = GenRequest.chat("a" * 1000)        # ~250 token estimate
    assert b.can_spend(small) is True
    assert b.can_spend(big) is False


# --------------------------------------------------------------------------- Timing math
def test_monotonic_timer_and_timing_math():
    timing = Timing()
    timer = MonotonicTimer(timing)
    timer.mark_first_byte()
    timer.add_chunk("", t=0.10)          # empty chunk: no first-token yet
    timer.add_chunk("Hel", t=0.20)       # first content -> first_token = 0.20
    timer.add_chunk("lo", t=0.35)
    timer.mark_end()

    assert timing.t_first_byte is not None
    assert timing.t_first_token == 0.20
    assert timing.ttft == 0.20
    assert timing.t_end is not None
    # inter-token latencies only over content chunks: 0.35 - 0.20
    itls = timing.inter_token_latencies()
    assert len(itls) == 1
    assert abs(itls[0] - 0.15) < 1e-9


def test_parse_server_timing_headers():
    assert parse_server_timing({"openai-processing-ms": "123.4"}) == 123.4
    assert parse_server_timing({"x-envoy-upstream-service-time": "42"}) == 42.0
    assert parse_server_timing({"x-response-time": "1.5s"}) == 1500.0
    # Standard Server-Timing: prefer a "total" metric.
    st = 'cache;desc="hit";dur=2.0, total;dur=57.5'
    assert parse_server_timing({"Server-Timing": st}) == 57.5
    # No total -> sum of dur metrics.
    st2 = "db;dur=10, app;dur=5"
    assert parse_server_timing({"server-timing": st2}) == 15.0
    assert parse_server_timing({"content-type": "application/json"}) is None


# --------------------------------------------------------------------------- httpclient (mocked)
async def test_request_json_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate limited"})
        return httpx.Response(
            200, json={"ok": True}, headers={"openai-processing-ms": "12.5"}
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        status, headers, body, timing, error = await request_json(
            client, "POST", "/v1/chat", json_body={"a": 1},
            policy=RetryPolicy(max_retries=3, base_delay=0.0),
        )

    assert calls["n"] == 2                    # retried exactly once
    assert status == 200 and error is None
    assert body == {"ok": True}
    assert timing.server_ms == 12.5
    assert timing.t_first_byte is not None and timing.t_end is not None


async def test_request_json_http_error_does_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad input"}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        status, headers, body, timing, error = await request_json(
            client, "POST", "/v1/chat", json_body={"a": 1},
            policy=RetryPolicy(max_retries=2, base_delay=0.0),
        )

    assert status == 400
    assert error is not None and "bad input" in error
    assert body == {"error": {"message": "bad input"}}


async def test_stream_sse_yields_and_builds_timing():
    sse = (
        b'data: {"choices":[{"delta":{"content":"He"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse
        )

    transport = httpx.MockTransport(handler)
    timing = Timing()
    got: list[tuple] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async for obj, arrival in stream_sse(
            client, "/v1/stream", json_body={"stream": True}, timing=timing
        ):
            got.append((obj, arrival))

    assert len(got) == 2
    assert got[0][0]["choices"][0]["delta"]["content"] == "He"
    assert len(timing.chunks) == 2
    assert timing.t_first_byte is not None
    assert timing.t_first_token is not None
    assert timing.t_end is not None
    # arrival times are non-decreasing
    assert got[1][1] >= got[0][1]


if __name__ == "__main__":
    raise SystemExit(json.dumps({"hint": "run with: python -m pytest tests/test_infra.py -q"}))
