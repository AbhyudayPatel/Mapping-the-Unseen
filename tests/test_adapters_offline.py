"""Offline tests for the replay and HTTP-template adapters (no network).

* ReplayAdapter: record a couple of observations with the real Recorder, then check
  that replaying the same requests returns the reconstructed responses and that an
  unrecorded request yields (or raises) the no-match sentinel.
* HTTPTemplateAdapter: config ``from_dict`` / ``to_dict`` round-trip, plus an
  end-to-end generate against an ``httpx.MockTransport`` (template fill + extraction).

asyncio tests run marker-free (pyproject sets ``asyncio_mode = "auto"``).
"""
from __future__ import annotations

import json

import httpx
import pytest

from forensics.core.adapters.http_template import (
    OPENAI_CHAT_PRESET,
    HTTPTemplateAdapter,
    TemplateConfig,
    fill_template,
    resolve_path,
)
from forensics.core.adapters.replay import (
    NoRecordedResponse,
    ReplayAdapter,
    gen_response_from_dict,
    request_hash,
)
from forensics.core.recorder import Recorder
from forensics.core.types import (
    Chunk,
    DecisionAnswer,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    GenRequest,
    GenResponse,
    Timing,
    TokenLogprob,
    TopToken,
    Usage,
)


# =========================================================================== replay
def _record_two_generates(run_dir: str) -> tuple[GenRequest, GenRequest, GenRequest]:
    rec = Recorder(run_dir)
    req1 = GenRequest.chat("What is the capital of France?", max_tokens=4)
    resp1 = GenResponse(
        text="Paris",
        choices_text=["Paris"],
        finish_reason="stop",
        model_reported="gpt-4o-mini-2024-07-18",
        system_fingerprint="fp_abc",
        usage=Usage(prompt_tokens=7, completion_tokens=1, total_tokens=8),
        tokens=[TokenLogprob(token="Paris", logprob=-0.01, bytes=[80],
                             token_id=1234,
                             top=[TopToken(token="Paris", logprob=-0.01, token_id=1234),
                                  TopToken(token="London", logprob=-6.5)])],
        timing=Timing(t_first_byte=0.05, t_end=0.2, server_ms=120.0),
        headers={"x-ms-region": "East US 2"},
        status=200,
    )
    req2 = GenRequest.chat("Say hi", max_tokens=8, temperature=0.7)
    resp2 = GenResponse(
        text="hi there",
        choices_text=["hi there"],
        finish_reason="stop",
        model_reported="gpt-4o-mini-2024-07-18",
        usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        timing=Timing(chunks=[Chunk(t=0.1, text="hi"), Chunk(t=0.15, text=" there")]),
        status=200,
    )
    rec.record("generate", req1, resp1, probe="p01")
    rec.record("generate", req2, resp2, probe="p01")

    unknown = GenRequest.chat("An unrecorded prompt", max_tokens=16)
    return req1, req2, unknown


async def test_replay_matches_recorded_generates(tmp_path):
    run_dir = str(tmp_path)
    req1, req2, unknown = _record_two_generates(run_dir)

    replay = ReplayAdapter(run_dir)

    r1 = await replay.generate(req1)
    assert r1.ok
    assert r1.text == "Paris"
    assert r1.finish_reason == "stop"
    assert r1.model_reported == "gpt-4o-mini-2024-07-18"
    assert r1.system_fingerprint == "fp_abc"
    assert r1.usage.prompt_tokens == 7 and r1.usage.completion_tokens == 1
    # per-token logprobs reconstructed into real dataclasses
    assert r1.tokens is not None and r1.tokens[0].token == "Paris"
    assert r1.tokens[0].token_id == 1234
    assert r1.tokens[0].top[1].token == "London"
    assert r1.timing.server_ms == 120.0

    r2 = await replay.generate(req2)
    assert r2.text == "hi there"
    assert [c.text for c in r2.timing.chunks] == ["hi", " there"]

    # unrecorded request -> graceful no-match sentinel
    r3 = await replay.generate(unknown)
    assert not r3.ok
    assert r3.error == "no recorded response"


async def test_replay_strict_raises_on_unknown(tmp_path):
    run_dir = str(tmp_path)
    _, _, unknown = _record_two_generates(run_dir)
    replay = ReplayAdapter(run_dir, strict=True)
    with pytest.raises(NoRecordedResponse):
        await replay.generate(unknown)


async def test_replay_capabilities_from_log(tmp_path):
    run_dir = str(tmp_path)
    _record_two_generates(run_dir)
    caps = ReplayAdapter(run_dir).static_capabilities()
    assert caps.kind == "generative"
    assert caps.chat is True
    assert caps.logprobs is True
    assert caps.logprob_bytes is True
    assert caps.token_ids_known is True
    assert caps.streaming is True            # req2 carried chunks
    assert caps.usage_prompt_tokens is True
    assert any("recorded models" in n for n in caps.notes)


async def test_replay_decision_roundtrip(tmp_path):
    run_dir = str(tmp_path)
    rec = Recorder(run_dir)
    q = {"sentiment": DecisionQuestion(type="choice", instructions="Pick one",
                                       criteria={"pos": "positive", "neg": "negative"})}
    dreq = DecisionRequest(state="I love this", questions=q)
    dresp = DecisionResponse(
        answers={"sentiment": DecisionAnswer(type="choice", value="pos", confidence=0.83,
                                             probabilities={"pos": 0.9, "neg": 0.1})},
        usage=Usage(prompt_tokens=280, completion_tokens=38),
        model_reported="jev-1.13.0",
        status=200,
    )
    rec.record("decide", dreq, dresp, probe="p20")

    replay = ReplayAdapter(run_dir)
    assert replay.kind == "decision"
    out = await replay.decide(dreq)
    assert out.ok
    assert out.answers["sentiment"].value == "pos"
    assert out.answers["sentiment"].probabilities == {"pos": 0.9, "neg": 0.1}
    assert out.model_reported == "jev-1.13.0"

    miss = await replay.decide(DecisionRequest(state="other", questions=q))
    assert miss.error == "no recorded response"


def test_request_hash_ignores_tag(tmp_path):
    a = GenRequest.chat("same prompt", max_tokens=4)
    b = GenRequest.chat("same prompt", max_tokens=4)
    b.tag = "different-label"
    assert request_hash("generate", a) == request_hash("generate", b)
    c = GenRequest.chat("same prompt", max_tokens=5)   # semantic change
    assert request_hash("generate", a) != request_hash("generate", c)


def test_gen_response_from_dict_handles_inf_sentinel():
    # to_jsonable writes -inf as the string "-inf"; reconstruction must restore it.
    d = {"text": "x", "tokens": [{"token": "x", "logprob": "-inf"}]}
    resp = gen_response_from_dict(d)
    assert resp.tokens is not None and resp.tokens[0].logprob == float("-inf")


# ===================================================================== http_template
def test_template_config_roundtrip():
    cfg = TemplateConfig.from_dict(OPENAI_CHAT_PRESET)
    d = cfg.to_dict()
    # fully JSON-serialisable (presets are saved to disk as JSON)
    text = json.dumps(d)
    assert "api_key" not in json.loads(text).get("body_template", {})  # no secret leaked
    # round-trips exactly, and is idempotent
    cfg2 = TemplateConfig.from_dict(d)
    assert cfg2 == cfg
    assert TemplateConfig.from_dict(cfg2.to_dict()) == cfg2


def test_adapter_roundtrip_and_no_key_in_dict():
    a = HTTPTemplateAdapter.from_dict(OPENAI_CHAT_PRESET, api_key="sk-secret")
    d = a.to_dict()
    # the API key is never serialised
    assert "sk-secret" not in json.dumps(d)
    b = HTTPTemplateAdapter.from_dict(d)
    assert b.to_dict() == d


def test_resolve_path_helpers():
    body = {"choices": [{"message": {"content": "hi"}}, {"message": {"content": "yo"}}],
            "usage": {"prompt_tokens": 11}}
    assert resolve_path(body, "choices[0].message.content") == "hi"
    assert resolve_path(body, "choices.0.message.content") == "hi"
    assert resolve_path(body, "choices[-1].message.content") == "yo"
    assert resolve_path(body, "usage.prompt_tokens") == 11
    assert resolve_path(body, "len(choices)") == 2
    assert resolve_path(body, "missing.path[3]") is None


def test_fill_template_typed_and_embedded():
    from forensics.core.types import Message
    req = GenRequest(messages=[Message("system", "be brief"), Message("user", "hello")],
                     max_tokens=5, temperature=0.0)
    subs = {"messages_json": [{"role": "user", "content": "hello"}], "max_tokens": 5,
            "temperature": 0.0, "system": "be brief"}
    tmpl = {"messages": "{messages_json}", "max_tokens": "{max_tokens}",
            "temperature": "{temperature}", "note": "system says: {system}"}
    out = fill_template(tmpl, subs)
    assert out["messages"] == [{"role": "user", "content": "hello"}]   # typed list
    assert out["max_tokens"] == 5 and isinstance(out["max_tokens"], int)
    assert out["temperature"] == 0.0
    assert out["note"] == "system says: be brief"                      # embedded string


def _openai_style_body() -> dict:
    return {
        "id": "chatcmpl-x",
        "model": "gpt-4o-mini-2024-07-18",
        "system_fingerprint": "fp_369e662417",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Paris"},
            "logprobs": {"content": [
                {"token": "Paris", "logprob": -0.002, "bytes": [80, 97, 114, 105, 115],
                 "top_logprobs": [{"token": "Paris", "logprob": -0.002},
                                  {"token": "London", "logprob": -8.5}]},
            ]},
        }],
        "usage": {"prompt_tokens": 14, "completion_tokens": 1, "total_tokens": 15,
                  "latency_checkpoint": {"engine_ttft_ms": 23}},
    }


async def test_http_template_generate_against_mock_transport():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_style_body(),
                              headers={"x-ms-region": "East US 2"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = {
        **OPENAI_CHAT_PRESET,
        "url": "http://mock/openai/v1/chat/completions",
        "headers": {"api-key": "{api_key}", "content-type": "application/json"},
        "api_key_env": None,
    }
    adapter = HTTPTemplateAdapter.from_dict(cfg, api_key="unit-test-key", client=client)
    try:
        resp = await adapter.generate(GenRequest.chat("capital of France?", max_tokens=4))
    finally:
        await client.aclose()

    # extraction
    assert resp.ok
    assert resp.text == "Paris"
    assert resp.choices_text == ["Paris"]
    assert resp.finish_reason == "stop"
    assert resp.model_reported == "gpt-4o-mini-2024-07-18"
    assert resp.system_fingerprint == "fp_369e662417"
    assert resp.usage.prompt_tokens == 14 and resp.usage.total_tokens == 15
    assert resp.usage.raw["latency_checkpoint"]["engine_ttft_ms"] == 23
    assert resp.tokens is not None
    assert resp.tokens[0].token == "Paris"
    assert resp.tokens[0].bytes == [80, 97, 114, 105, 115]
    assert resp.tokens[0].top[1].token == "London"

    # template fill (typed messages + int max_tokens) and header substitution
    assert captured["api_key"] == "unit-test-key"
    assert captured["body"]["messages"] == [{"role": "user", "content": "capital of France?"}]
    assert captured["body"]["max_tokens"] == 4
    assert captured["body"]["model"] == "gpt-4o-mini"


async def test_http_template_http_error_does_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "unsupported_parameter"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = {**OPENAI_CHAT_PRESET, "url": "http://mock/v1/chat/completions",
           "api_key_env": None, "max_retries": 0}
    adapter = HTTPTemplateAdapter.from_dict(cfg, api_key="k", client=client)
    try:
        resp = await adapter.generate(GenRequest.chat("hi"))
    finally:
        await client.aclose()

    assert not resp.ok
    assert resp.status == 400
    assert "unsupported_parameter" in (resp.error or "")


if __name__ == "__main__":
    raise SystemExit(json.dumps({"hint": "python -m pytest tests/test_adapters_offline.py -q"}))
