"""Decision-model fixes found by the live Jev run.

* P01 / P02 have real decision paths (``ctx.decide``) instead of crashing on ``ctx.gen``.
* P22 decision timing (state length, #questions, prefix cache, active params).
* P18 question isolation is decisive (K secrets, both contrasts).
* Family matching: prior-only beliefs are not constraints, a measurement is required and
  scores reflect constraint coverage.
"""
from __future__ import annotations

import asyncio
import dataclasses

from forensics.core.adapters.simulated import SimConfig, SimulatedAdapter, sim_jev
from forensics.inference.belief import Belief
from forensics.inference.family_match import (
    INSUFFICIENT_NOTE, evidence_backed, match_families, score_ranked,
)
from forensics.inference.hypotheses import PROPERTIES
from forensics.knowledge.archdb import ArchDB
from forensics.probes.base import Estimate, Evidence, ProbeContext
from forensics.probes.p01_interface import InterfaceStackProbe, alias_kind
from forensics.probes.p02_tokenizer import TokenizerFingerprintProbe
from forensics.probes.p18_decision import DecisionIsolationProbe
from forensics.probes.p22_decision_timing import DecisionTimingProbe


def _run(probe, adapter, **config):
    ctx = ProbeContext(adapter, config={"fast": True, **config}, seed=3)
    ctx.current_probe = probe.id
    return asyncio.run(probe.run(ctx))


def _posterior(prop: str, evidence) -> dict[str, float]:
    b = Belief(PROPERTIES, kind="decision")
    b.apply_all([e for e in evidence if e.property == prop])
    return b.posterior[prop]


def _ev(res, prop):
    return [e for e in res.evidence if e.property == prop]


def _est(res, q):
    return next((e for e in res.estimates if e.quantity == q), None)


# =========================================================================== P01
def test_p01_decision_path_fingerprints_stack():
    res = _run(InterfaceStackProbe(), SimulatedAdapter(sim_jev()), requested_model="sim-latest")
    assert res.status == "ok" and res.n_requests <= 8
    row = res.tables["stack"][0]
    sig = set(row["serving_signals"].split(","))
    assert {"cloudflare", "envoy", "fastapi_pydantic_v2", "sanitized_400"} <= sig
    assert row["request_id_header"] == "x-typesafe-request-id"
    assert row["multiple_worker_processes"] is True          # key order varies (4 workers)
    assert row["alias_kind"] == "version_alias"
    assert row["validation_before_model"] is True
    errs = {e["status"]: e for e in res.tables["errors"]}
    assert errs[422]["shape"] == "pydantic_v2_list" and errs[400]["message"] == "Invalid request."
    post = _posterior("system.topology", res.evidence)
    assert max(post, key=post.get) == "single_model"


def test_p01_decision_single_worker_stable_order():
    cfg = dataclasses.replace(sim_jev(), decision_workers=1)
    res = _run(InterfaceStackProbe(), SimulatedAdapter(cfg))
    assert res.tables["stack"][0]["multiple_worker_processes"] is False


def test_alias_kind():
    assert alias_kind("jev-latest", "jev-1.13.0") == "version_alias"
    assert alias_kind("gpt-4o", "gpt-4o-2024-08-06") == "same"
    assert alias_kind("gpt-4o", "llama-3-70b") == "different_model"


# =========================================================================== P02
def test_p02_decision_path_custom_tokenizer_is_not_crowned():
    res = _run(TokenizerFingerprintProbe(), SimulatedAdapter(sim_jev()))
    assert res.status == "ok" and res.n_requests <= 30
    assert _est(res, "tokenizer.family").value == "unknown_custom"
    # 260 template + 7 question wrapper + 6 tokens of the minimal noul
    assert _est(res, "tokenizer.chat_template_overhead").value == 273
    assert _est(res, "dims.vocab_size") is None               # never a wrong measured vocab
    assert any("custom or pruned vocabulary; closest:" in n for n in res.notes)
    ev = _ev(res, "tokenizer.family")[0]
    assert max(ev.likelihoods, key=ev.likelihoods.get) == "unknown_custom"
    assert ev.stats["exact_rate"] < 0.85
    feats = {r["feature"]: r["observed"] for r in res.tables["features"]}
    assert feats["digit splitting"] == "1 token per digit"
    assert res.shared["tokenizer"]["family"] == "unknown_custom"


def test_p02_decision_path_matches_public_tokenizer():
    cfg = SimConfig(arch_family="decision", tokenizer_name="o200k_base",
                    chat_template_overhead=260, per_message_overhead=7, expose_logprobs=False)
    res = _run(TokenizerFingerprintProbe(), SimulatedAdapter(cfg))
    fam = _est(res, "tokenizer.family").value
    assert "o200k" in fam
    assert _est(res, "dims.vocab_size") is not None
    ev = _ev(res, "tokenizer.family")[0]
    assert ev.stats["exact_rate"] >= 0.85


# =========================================================================== P22
def test_p22_decision_timing_on_sim_jev():
    res = _run(DecisionTimingProbe(), SimulatedAdapter(sim_jev()))
    assert res.status == "ok" and res.n_requests <= 30
    assert _winner(res, "attn.long_range") == "full_quadratic"
    assert _winner(res, "serving.prompt_caching") == "yes"
    assert _winner(res, "decision.question_coupling") == "independent"
    assert _winner(res, "decision.scoring_mechanism") == "single_pass_heads"
    tps = _est(res, "timing.prefill_tokens_per_s")
    assert tps is not None and 1.0e5 < tps.value < 3.0e5     # sim: ~5 us/token + attention
    ap = _est(res, "dims.active_params")
    assert ap.level == "speculative" and ap.ci[0] < 10.0 < ap.ci[1]   # true 10B, wide CI
    assert res.tables["server_ms_vs_state"] and res.series["server_ms_vs_questions"]["x"]
    for ev in res.evidence:
        assert ev.alternatives and all(v > 0 for v in ev.likelihoods.values())


def test_p22_linear_prefill_and_no_cache():
    # a sliding window saturates the attention term (linear prefill); no prompt cache
    cfg = dataclasses.replace(sim_jev(), sliding_window=256, prompt_caching=False,
                              timing_jitter_s=0.002)
    res = _run(DecisionTimingProbe(), SimulatedAdapter(cfg))
    lr = _ev(res, "attn.long_range")[0]
    assert lr.likelihoods["full_quadratic"] < 1.0 and lr.strength <= 0.25
    assert _winner(res, "serving.prompt_caching") == "no"


def _winner(res, prop):
    post = _posterior(prop, res.evidence)
    return max(post, key=post.get)


# =========================================================================== P18 isolation
def test_isolation_is_decisive_on_isolated_branches():
    res = _run(DecisionIsolationProbe(), SimulatedAdapter(sim_jev()))
    assert res.status == "ok" and res.n_requests <= 30
    ev = _ev(res, "decision.question_coupling")[0]
    assert ev.stats["n"] >= 3 and ev.stats["k_state"] == ev.stats["n"]
    assert ev.stats["k_sibling"] == 0
    post = _posterior("decision.question_coupling", res.evidence)
    assert post["independent"] >= 0.9, post


def test_isolation_detects_coupled_prompt():
    cfg = dataclasses.replace(sim_jev(), decision_isolated=False)
    res = _run(DecisionIsolationProbe(), SimulatedAdapter(cfg))
    post = _posterior("decision.question_coupling", res.evidence)
    assert post["coupled"] >= 0.9, post


# =========================================================================== family matching
def _backed_belief(**tops: str) -> Belief:
    b = Belief(PROPERTIES, kind="generative")
    for prop, val in tops.items():
        lk = {v: (1.0 if v == val else 0.05) for v in PROPERTIES[prop].values}
        b.update(Evidence(property=prop, likelihoods=lk, summary="test", probe_id="T"))
    return b


def test_prior_only_belief_and_no_measurements_gives_no_matches():
    b = Belief(PROPERTIES, kind="decision")
    fams, notes = match_families(b, [])
    assert fams == [] and any(n.startswith(INSUFFICIENT_NOTE) for n in notes)
    # the prior (kv_sharing 60% = prior 60%) is not evidence
    assert evidence_backed(b, "attn.kv_sharing")[0] is None


def test_engine_fallback_follows_same_rules():
    from forensics.inference.engine import _Engine
    eng = _Engine(SimulatedAdapter(sim_jev()), {}, None, None, None, 0, None)
    out = eng.fallback_family_matches(Belief(PROPERTIES, kind="decision"), [])
    assert out == [] and any(INSUFFICIENT_NOTE in n for n in eng.notes)


def test_strong_measurements_give_llama3_top_with_high_score():
    b = _backed_belief(**{"arch.family": "decoder_only", "attn.kv_sharing": "gqa_or_mqa"})
    ests = [Estimate("dims.hidden_size", 4096, level="inferred"),
            Estimate("dims.vocab_size", 128256, level="estimated")]
    fams, _ = match_families(b, ests)
    assert fams and "llama-3" in fams[0].name.lower(), fams[:3]
    assert fams[0].score >= 0.6
    assert {"hidden_size", "vocab_size", "arch_type", "kv_sharing"} <= set(fams[0].matched)


def test_prior_only_kv_sharing_is_not_a_constraint():
    b = _backed_belief(**{"arch.family": "decoder_only"})       # kv_sharing stays at prior
    fams, notes = match_families(b, [Estimate("dims.hidden_size", 4096, level="inferred")])
    assert fams and all("kv_sharing" not in f.matched for f in fams)


def test_weak_constraints_give_low_scores():
    db = ArchDB.load()
    only_arch = score_ranked(db.nearest({"arch_type": "decoder"}, top_k=None))
    assert only_arch and max(f.score for f in only_arch) < 0.5
    two_weak = score_ranked(db.nearest({"arch_type": "decoder", "kv_sharing": "gqa_or_mqa"},
                                       top_k=None))
    assert max(f.score for f in two_weak) < 0.5                 # never 1.000
    b = _backed_belief(**{"arch.family": "decoder_only"})
    fams, _ = match_families(b, [Estimate("context.max_input_tokens", 131072, level="observed")])
    assert fams and max(f.score for f in fams) < 0.5
