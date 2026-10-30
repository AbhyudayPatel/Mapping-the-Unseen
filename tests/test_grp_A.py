"""Group-A probe tests (P01 interface, P02 tokenizer, P03 hidden-size, P04 precision).

Every probe is validated against the SimulatedAdapter with KNOWN ground truth in fast mode
(<= ~30 requests). We assert each probe recovers the right estimate or moves belief in the
right direction.
"""
from __future__ import annotations

import asyncio

import pytest

from forensics.core.adapters.simulated import (
    SimConfig, SimulatedAdapter, sim_decoder_dense, sim_ssm,
)
from forensics.probes.base import ProbeContext
from forensics.probes.p01_interface import InterfaceStackProbe, detect_stack
from forensics.probes.p02_tokenizer import TokenizerFingerprintProbe
from forensics.probes.p03_logits import HiddenSizeProbe
from forensics.probes.p04_precision import LogitPrecisionProbe


def _run(coro):
    return asyncio.run(coro)


def _ctx(adapter: SimulatedAdapter, **config) -> ProbeContext:
    return ProbeContext(adapter, config={"fast": True, **config})


def _long_responder():
    fixed = " ".join(f"token{i}" for i in range(400))
    return lambda _p: fixed


# =========================================================================== P01
def test_p01_detect_stack_pure():
    # Pydantic-v2 422 + Envoy/Cloudflare/Azure/SGLang/vLLM signatures.
    hdr = {"server": "cloudflare", "cf-ray": "abc", "x-envoy-upstream-service-time": "12",
           "x-ms-served-model": "gpt-4o-2024"}
    body = {"detail": [{"type": "missing", "loc": ["body", "state"], "msg": "Field required"}]}
    sig = detect_stack(hdr, body, {"routing": {"serving_pipereplica": "d-prefill-r13-dp0"}})
    s = set(sig["signals"])
    assert {"cloudflare", "envoy", "azure", "fastapi_pydantic_v2"} <= s
    assert "prefill" in str(sig.get("disaggregation", "")).lower()
    vllm = detect_stack({}, {"object": "error", "message": "bad"}, None)
    assert "vllm" in vllm["signals"]
    sglang = detect_stack({}, None, {"choices": [{"matched_stop": 2}]})
    assert "sglang" in sglang["signals"]


def test_p01_single_model_no_drift():
    cfg = SimConfig(hidden_size=64, vocab_size=4000, max_context=131072)
    probe = InterfaceStackProbe()
    res = _run(probe.run(_ctx(SimulatedAdapter(cfg))))
    assert res.status == "ok"
    row = res.tables["stack"][0]
    assert row["model_reported"] == "sim-decoder_only"
    assert row["system_fingerprint_present"] is True
    assert row["system_fingerprint_drift"] is False
    assert row["model_aliasing"] is None  # no requested model supplied
    ev = next(e for e in res.evidence if e.property == "system.topology")
    # a stable single fingerprint with no aliasing favors a single model
    assert ev.likelihoods["single_model"] >= ev.likelihoods["router_or_cascade"]
    assert ev.likelihoods["single_model"] >= ev.likelihoods["ensemble"]


def test_p01_aliasing_and_error_body():
    # small context so the overflow probe is cheap and returns the typed error body
    cfg = SimConfig(hidden_size=64, vocab_size=4000, max_context=120,
                    chat_template_overhead=7, per_message_overhead=4)
    probe = InterfaceStackProbe()
    res = _run(probe.run(_ctx(SimulatedAdapter(cfg),
                              requested_model="gpt-4o", p01_overflow_words=200)))
    row = res.tables["stack"][0]
    assert row["model_aliasing"] is True
    assert row["context_limit"] == 120
    assert "openai_style_error" in row["serving_signals"]
    err = res.tables["errors"][0]
    assert err["status"] == 400 and err["context_limit"] == 120
    ev = next(e for e in res.evidence if e.property == "system.topology")
    # aliasing (asked X, got Y) shifts belief toward a router/cascade
    assert ev.likelihoods["router_or_cascade"] > ev.likelihoods["single_model"]


# =========================================================================== P02
def test_p02_recovers_o200k():
    probe = TokenizerFingerprintProbe()
    res = _run(probe.run(_ctx(SimulatedAdapter(sim_decoder_dense()))))
    assert res.status == "ok"
    tok = res.shared["tokenizer"]
    assert "o200k" in tok["family"]
    # empty single-message baseline overhead = template(7) + per-message(4)
    assert tok["chat_template_overhead"] == 11
    top = res.tables["tokenizer_scores"][0]
    assert "o200k" in top["family"] and top["l1"] == 0
    ev = next(e for e in res.evidence if e.property == "tokenizer.family")
    assert max(ev.likelihoods, key=ev.likelihoods.get) == top["family"]
    vocab = next(e for e in res.estimates if e.quantity == "dims.vocab_size")
    assert 150_000 <= vocab.value <= 210_000


def test_p02_recovers_cl100k_and_discriminates():
    probe = TokenizerFingerprintProbe()
    res = _run(probe.run(_ctx(SimulatedAdapter(sim_ssm()))))  # cl100k_base
    scores = {r["family"]: r for r in res.tables["tokenizer_scores"]}
    winner = res.shared["tokenizer"]["family"]
    assert "cl100k" in winner
    # cl100k must beat o200k on exact matches for a cl100k target
    o200k = next((v for k, v in scores.items() if "o200k_base" in k), None)
    assert o200k is not None
    assert scores[winner]["exact"] > o200k["exact"]
    # exact ties (e.g. cl100k-derived vocabularies) are published, not silently dropped:
    # every tied family has the winner's (exact, L1) and the canonical base one is reported
    ties = res.shared["tokenizer"]["tied_with"]
    assert isinstance(ties, list) and winner not in ties
    for fam in ties:
        assert (scores[fam]["exact"], scores[fam]["l1"]) ==             (scores[winner]["exact"], scores[winner]["l1"])
        assert scores[fam]["tied_with_winner"] is True
    ev = next(e for e in res.evidence if e.property == "tokenizer.family")
    assert all(ev.likelihoods[f] == ev.likelihoods[winner] for f in ties)


# =========================================================================== P03
# P03 is validated on the BARE simulator (NO continuation responder): the corrected
# method reconstructs each row from POSITION-0 logits only, which depend on the prompt
# alone and are therefore identical across logit_bias groups and repeated calls. It must
# not depend on continuations being deterministic under bias.
def _p03_cfg(d: int, dtype: str, bias_affects: bool = True) -> SimConfig:
    return SimConfig(arch_family="decoder_only", hidden_size=d, vocab_size=4000,
                     logit_dtype=dtype, max_top_logprobs=20,
                     logit_bias_affects_logprobs=bias_affects, determinism_seed=11)


@pytest.mark.parametrize("d", [64, 128, 256])
def test_p03_hidden_size_fp32_exact(d):
    # Bare simulator (no responder), generous ctx (fast disabled) -> exact recovery.
    adapter = SimulatedAdapter(_p03_cfg(d, "fp32"))
    res = _run(HiddenSizeProbe().run(_ctx(adapter, fast=False, p03_expected_d=d)))
    assert res.status == "ok"
    est = next(e for e in res.estimates if e.quantity == "dims.hidden_size")
    assert est.value == d, f"expected exact {d}, got {est.value}"
    acq = res.tables["acquisition"][0]
    assert acq["path"] == "logit_bias_pos0"
    assert acq["rows"] > d and acq["cols"] > d  # matrix exceeds d in both dimensions
    assert acq["requests"] == res.n_requests > 0  # request count reported
    # cost formula: requests == n_prompts * n_groups
    assert acq["requests"] <= (acq["rows"] + 1) * acq["n_groups"]


@pytest.mark.parametrize("d", [64, 128])
def test_p03_hidden_size_bf16_within_2(d):
    # Bare simulator, bf16 quantisation -> within +-2 of the truth.
    adapter = SimulatedAdapter(_p03_cfg(d, "bf16"))
    res = _run(HiddenSizeProbe().run(_ctx(adapter, fast=False, p03_expected_d=d)))
    assert res.status == "ok"
    est = next(e for e in res.estimates if e.quantity == "dims.hidden_size")
    assert abs(est.value - d) <= 2, f"expected {d}+-2, got {est.value}"
    assert res.shared.get("hidden_size") == est.value
    # output.head evidence should favor a generative LM head (softmax bottleneck)
    ev = next(e for e in res.evidence if e.property == "output.head")
    assert ev.likelihoods["generative_lm_head"] >= ev.likelihoods["classification_heads"]


def test_p03_reasoning_style_skipped():
    cfg = SimConfig(arch_family="decoder_only", hidden_size=128, expose_logprobs=False)
    adapter = SimulatedAdapter(cfg)
    probe = HiddenSizeProbe()
    ok, _ = probe.applicable(_ctx(adapter))
    assert ok is False  # requires logprobs
    res = _run(probe.run(_ctx(adapter)))
    assert res.status == "skipped"
    est = next(e for e in res.estimates if e.quantity == "dims.hidden_size")
    assert est.value is None and est.identifiability == 5


def test_p03_topk_only_is_graceful():
    # bias does not affect logprobs -> fall back to top-k union; must not crash.
    adapter = SimulatedAdapter(_p03_cfg(64, "bf16", bias_affects=False))
    adapter.register_responder(_long_responder())
    res = _run(HiddenSizeProbe().run(_ctx(adapter, p03_expected_d=64)))
    assert res.status in {"ok", "partial"}
    assert res.tables["acquisition"][0]["path"] == "topk_union"


def test_p03_fast_budget_insufficient_is_graceful():
    # Fast mode caps requests far below n_prompts*n_groups: the probe must NOT crash and
    # must report None (or a lower bound) at identifiability >= 4, with a recommendation.
    adapter = SimulatedAdapter(_p03_cfg(128, "fp32"))  # bare sim, bias affects logprobs
    res = _run(HiddenSizeProbe().run(_ctx(adapter, p03_expected_d=128)))  # fast=True
    assert res.status in {"partial", "ok"}
    est = next(e for e in res.estimates if e.quantity == "dims.hidden_size")
    assert est.value is None or est.value <= 128         # None or an honest lower bound
    assert est.identifiability >= 4                       # weak / assumption / non-identifiable
    assert res.shared.get("hidden_size") in (None, est.value)
    assert any("recommend" in n.lower() or "budget" in n.lower() for n in res.notes)


def test_p03_engine_recovers_hidden_size():
    # Engine-level: run_forensics on a bare simulator must surface dims.hidden_size ~ 128.
    from forensics.core.budget import Budget
    from forensics.inference.engine import run_forensics

    cfg = SimConfig(hidden_size=128, vocab_size=4000, max_context=8192, logit_dtype="fp32")
    adapter = SimulatedAdapter(cfg)
    budget = Budget(max_requests=8000)
    config = {"p03_expected_d": 128, "thorough": True}
    result = _run(run_forensics(adapter, config=config, budget=budget, seed=1))
    hs = [e for e in result.estimates
          if e.quantity == "dims.hidden_size" and e.value is not None]
    assert hs, "expected a dims.hidden_size estimate from the engine run"
    assert any(abs(int(e.value) - 128) <= 2 for e in hs), \
        f"expected hidden_size ~128, got {[e.value for e in hs]}"


# =========================================================================== P04
@pytest.mark.parametrize("truth,expected", [
    ("fp32", "fp32"),
    ("bf16", "bf16"),
    ("fp16", "fp16"),
    ("fp8", "fp8_or_lower"),
])
def test_p04_logit_dtype(truth, expected):
    cfg = SimConfig(arch_family="decoder_only", hidden_size=128, vocab_size=4000,
                    logit_dtype=truth, max_top_logprobs=20, determinism_seed=4)
    res = _run(LogitPrecisionProbe().run(_ctx(SimulatedAdapter(cfg))))
    est = next(e for e in res.estimates if e.quantity == "out.logit_dtype")
    assert est.value == expected, f"truth {truth} -> {est.value}, expected {expected}"
    ev = next(e for e in res.evidence if e.property == "out.logit_precision")
    assert max(ev.likelihoods, key=ev.likelihoods.get) == expected
