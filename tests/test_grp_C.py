"""Group C probe tests: P10 determinism, P11 MoE, P12 knowledge, P13 reasoning.

Each probe is validated against the SimulatedAdapter with KNOWN ground truth (via the
``sim_*`` presets and hand-set SimConfig knobs / responders), asserting the probe moves
belief in the RIGHT direction or recovers the right estimate.  All runs use
``config={"fast": True}`` so the suite stays cheap (<=~30 requests per probe).
"""
from __future__ import annotations

import asyncio

import numpy as np

from forensics.core.adapters.simulated import (
    SimConfig, SimulatedAdapter, sim_decoder_dense, sim_decoder_moe,
)
from forensics.probes.base import NON_IDENTIFIABLE, ProbeContext
from forensics.probes.p10_determinism import DeterminismProbe
from forensics.probes.p11_moe import MoEProbe
from forensics.probes import p12_knowledge as P12
from forensics.probes.p12_knowledge import KnowledgeProbe
from forensics.probes import p13_reasoning as P13
from forensics.probes.p13_reasoning import ReasoningProbe


def _run(coro):
    return asyncio.run(coro)


def _ctx(adapter, shared=None):
    ctx = ProbeContext(adapter, config={"fast": True})
    if shared:
        ctx.shared.update(shared)
    return ctx


def _posterior(res, prop: str, prior=None) -> dict[str, float]:
    """Combine a result's Evidence on one property into a posterior (documented semantics:
    likelihood ** strength, floors, unlisted values = mean of listed)."""
    post = dict(prior or {})
    for ev in res.evidence:
        if ev.property != prop:
            continue
        vals = ev.likelihoods
        neutral = float(np.mean(list(vals.values()))) if vals else 1.0
        for k in list(post) or list(vals):
            post.setdefault(k, 0.5)
        for k in post:
            lk = max(vals.get(k, neutral), 1e-6)
            post[k] *= lk ** ev.strength
    s = sum(post.values()) or 1.0
    return {k: v / s for k, v in post.items()}


def _est(res, quantity):
    got = [e for e in res.estimates if e.quantity == quantity]
    return got[0] if got else None


# =========================================================================== P10
def test_p10_detects_batch_noninvariance():
    """batch_invariant=False -> jitter detected (favour 'no'); True -> deterministic ('yes')."""
    res_det = _run(DeterminismProbe().run(_ctx(SimulatedAdapter(sim_decoder_dense()))))
    res_jit = _run(DeterminismProbe().run(_ctx(SimulatedAdapter(sim_decoder_moe()))))
    assert res_det.n_requests <= 30 and res_jit.n_requests <= 30

    post_det = _posterior(res_det, "serving.batch_invariant", {"yes": 0.3, "no": 0.7})
    post_jit = _posterior(res_jit, "serving.batch_invariant", {"yes": 0.3, "no": 0.7})

    # deterministic stack -> believe it IS batch-invariant
    assert post_det["yes"] > 0.5, post_det
    # non-invariant stack -> believe it is NOT
    assert post_jit["no"] > 0.5, post_jit
    assert post_det["yes"] > post_jit["yes"]

    row_det = res_det.tables["repeat_summary"][0]
    row_jit = res_jit.tables["repeat_summary"][0]
    assert row_det["flavour"] == "deterministic"
    assert row_jit["detected_nondeterminism"] is True
    assert row_jit["logprob_jitter_std"] > row_det["logprob_jitter_std"]


# =========================================================================== P11
def test_p11_moe_posterior_higher_for_moe():
    res_dense = _run(MoEProbe().run(_ctx(SimulatedAdapter(sim_decoder_dense()))))
    res_moe = _run(MoEProbe().run(_ctx(SimulatedAdapter(sim_decoder_moe()))))
    assert res_dense.n_requests <= 30 and res_moe.n_requests <= 30

    post_dense = _posterior(res_dense, "arch.sparsity", {"moe": 0.5, "dense": 0.5})
    post_moe = _posterior(res_moe, "arch.sparsity", {"moe": 0.5, "dense": 0.5})

    assert post_moe["moe"] > post_dense["moe"], (post_moe, post_dense)
    assert post_moe["moe"] > 0.5
    assert post_dense["moe"] < 0.5


def test_p11_capacity_speed_mismatch_from_shared():
    """When timing (active) and knowledge (total) disagree strongly, favour MoE harder."""
    a = SimulatedAdapter(sim_decoder_moe())
    ctx = _ctx(a, shared={"knowledge.total_params_b": 47.0,
                          "timing.active_params_b": 13.0})
    res = _run(MoEProbe().run(ctx))
    cap_ev = [e for e in res.evidence
              if e.property == "arch.sparsity" and e.stats.get("ratio")]
    assert any(e.stats["ratio"] > 1.8 for e in cap_ev), \
        [e.stats for e in res.evidence if e.property == "arch.sparsity"]
    post = _posterior(res, "arch.sparsity", {"moe": 0.5, "dense": 0.5})
    assert post["moe"] > 0.7, post


def test_p11_expert_estimates_non_identifiable():
    res = _run(MoEProbe().run(_ctx(SimulatedAdapter(sim_decoder_moe()))))
    ne = _est(res, "dims.n_experts")
    ae = _est(res, "dims.active_experts")
    assert ne is not None and ae is not None
    assert ne.identifiability == NON_IDENTIFIABLE
    assert ne.level == "speculative"


# =========================================================================== P12
def _knowledge_responder(adapter, cutoff_year=2023):
    """Simulate a model whose recall depends on knowledge_scale and a training cutoff."""
    def responder(rendered: str):
        m = P12._MARK_Q.search(rendered)
        if m:
            f = P12.FACT_BY_ID.get(m.group(1))
            if f is None:
                return None
            return f["a"] if f["rarity"] <= adapter.cfg.knowledge_scale else "I do not know."
        m = P12._MARK_EV.search(rendered)
        if m:
            e = P12.EVENT_BY_ID.get(m.group(1))
            if e is None:
                return None
            return e["a"] if e["year"] <= cutoff_year else "I do not know."
        return None
    return responder


def _run_knowledge(knowledge_scale: float, cutoff_year: int = 2023):
    cfg = sim_decoder_dense()
    cfg.knowledge_scale = knowledge_scale
    a = SimulatedAdapter(cfg)
    a.register_responder(_knowledge_responder(a, cutoff_year))
    res = _run(KnowledgeProbe().run(_ctx(a)))
    return res


def test_p12_recovers_monotone_curve_and_ordered_params():
    res_small = _run_knowledge(0.3)
    res_large = _run_knowledge(0.8)
    assert res_small.n_requests <= 30 and res_large.n_requests <= 30

    # monotone decreasing accuracy vs rarity in both cases
    for res in (res_small, res_large):
        tiers = res.tables["recall_by_tier"]
        accs = [t["accuracy"] for t in tiers]
        assert accs[0] >= accs[-1]              # easiest tier at least as good as hardest
        assert accs[-1] < accs[0]              # and strictly falls off at the rare end
        assert res.shared["knowledge"]["spearman"] <= 0.0

    small = _est(res_small, "dims.total_params")
    large = _est(res_large, "dims.total_params")
    assert small.level == "speculative"
    assert small.identifiability >= 3          # weakly / assumption-only identifiable
    # larger stored knowledge -> larger parameter estimate, with CI ordered
    assert small.value < large.value
    assert small.ci[0] < large.ci[0]
    assert small.ci[1] < large.ci[1]


def test_p12_recovers_training_cutoff():
    res = _run_knowledge(0.8, cutoff_year=2023)
    cut = _est(res, "knowledge.cutoff")
    assert cut is not None
    assert int(str(cut.value)) == 2023


def test_p12_publishes_total_params_shared():
    res = _run_knowledge(0.8)
    assert res.shared.get("knowledge.total_params_b", 0) > 0


# =========================================================================== P13
def _khop_responder(capability: int):
    def responder(rendered: str):
        m = P13._MARK.search(rendered)
        if not m:
            return None
        d, s = int(m.group(1)), int(m.group(2))
        _, ans = P13.build_khop(d, s)
        return ans if d <= capability else "nobody"
    return responder


def _run_reasoning(hidden: bool, mult: float, capability: int):
    cfg = sim_decoder_dense()
    cfg.hidden_reasoning = hidden
    cfg.reasoning_token_multiplier = mult
    a = SimulatedAdapter(cfg)
    a.register_responder(_khop_responder(capability))
    return _run(ReasoningProbe().run(_ctx(a)))


def test_p13_detects_hidden_reasoning_and_ratio():
    res = _run_reasoning(hidden=True, mult=2.0, capability=3)
    assert res.n_requests <= 30

    post = _posterior(res, "system.hidden_reasoning", {"yes": 0.4, "no": 0.6})
    assert post["yes"] > 0.5, post

    ratio = _est(res, "reasoning.hidden_tokens_ratio")
    assert ratio.value > 0
    assert abs(ratio.value - 2.0) < 0.6        # recovers the reasoning_token_multiplier
    assert ratio.identifiability == 1          # DIRECT: usage reports the tokens


def test_p13_recovers_max_hop_depth():
    for cap in (2, 3):
        res = _run_reasoning(hidden=True, mult=1.0, capability=cap)
        md = _est(res, "reasoning.max_hop_depth")
        assert md.value == cap, (cap, md.value,
                                 res.tables["depth_summary"])


def test_p13_no_hidden_reasoning_when_absent():
    res = _run_reasoning(hidden=False, mult=0.0, capability=3)
    post = _posterior(res, "system.hidden_reasoning", {"yes": 0.4, "no": 0.6})
    assert post["no"] > 0.5, post
    ratio = _est(res, "reasoning.hidden_tokens_ratio")
    assert ratio.value == 0.0
