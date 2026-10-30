"""Group-E probe tests: P18 (decision/Jev), P19 (architecture family), P20 (multimodality).

Every probe is validated against a :class:`SimulatedAdapter` whose hidden ``SimConfig`` is
the known ground truth, using ``config={"fast": True}``.  The assertions check that each
probe moves belief in the right direction / recovers the right estimate:

* P18 against ``sim_jev`` — additive template (260 tokens) + state prefix sharing;
  independent question branches; a joint softmax over options; semantic (order-invariant)
  option scoring; the 2-decimal probability grid; and a non-autoregressive readout.
* P19 against ``sim_decoder_dense`` / ``sim_ssm`` / ``sim_encoder`` — the ``arch.family``
  posterior favours decoder_only / ssm_or_linear / encoder_only respectively.
* P20 against ``sim_decoder_dense`` (text-only) — ``modal.input`` posterior favours
  text_only (the simulator silently ignores image/audio parts → the documented moderate
  strength path), with ``modal.output`` = text_only and ``modal.vision_encoder`` = none.

Belief updating here mirrors the ``Evidence`` contract: posterior ∝ prior · Π
likelihood(value)**strength, with unlisted values taking the mean of the listed ones.
"""
from __future__ import annotations

import asyncio

from forensics.core.adapters.simulated import (
    SimulatedAdapter, sim_decoder_dense, sim_encoder, sim_jev, sim_ssm,
)
from forensics.inference.hypotheses import PROPERTIES, normalized_priors
from forensics.probes import p18_decision as p18
from forensics.probes import p19_archfamily as p19
from forensics.probes import p20_multimodal as p20
from forensics.probes.base import ProbeContext

_LONG_REPLY = ("The inventor built a marvelous flying machine out of brass gears cogs and "
               "shimmering copper wire and then soared above the clouds happily for many hours "
               "on end throughout the long bright summer day")


# --------------------------------------------------------------------------- harness
def _run(probe: object, adapter: SimulatedAdapter, seed: int = 3, fast: bool = True):
    ctx = ProbeContext(adapter, config=({"fast": True} if fast else {}), seed=seed)
    res = asyncio.run(probe.run(ctx))  # type: ignore[attr-defined]
    return res, ctx


def _posterior(prop_key: str, evidences) -> dict[str, float]:
    prop = PROPERTIES[prop_key]
    post = dict(normalized_priors(prop))
    for ev in evidences:
        if ev.property != prop_key:
            continue
        listed = ev.likelihoods
        assert listed, f"{ev.probe_id}: empty likelihoods for {prop_key}"
        assert all(v > 0 for v in listed.values()), f"{ev.probe_id}: likelihoods must have a floor > 0"
        assert 0.0 < ev.strength <= 1.0, f"{ev.probe_id}: strength out of range"
        neutral = sum(listed.values()) / len(listed)
        for v in list(post):
            post[v] *= (listed.get(v, neutral)) ** ev.strength
    s = sum(post.values()) or 1.0
    return {k: v / s for k, v in post.items()}


def _winner(prop_key: str, evidences) -> str:
    post = _posterior(prop_key, evidences)
    return max(post, key=post.get)


def _estimate(res, quantity: str):
    for e in res.estimates:
        if e.quantity == quantity:
            return e
    return None


def _validate_evidence(res) -> None:
    """Common contract checks: known property keys, floors, alternatives, strength range."""
    for ev in res.evidence:
        assert ev.property in PROPERTIES, f"unknown property {ev.property}"
        assert ev.likelihoods and all(v > 0 for v in ev.likelihoods.values())
        assert 0.0 < ev.strength <= 1.0
        assert ev.alternatives, f"{ev.probe_id}: evidence should list confounders"


# =========================================================================== P18
def test_p18_additivity_recovers_template_and_sharing():
    res, ctx = _run(p18.DecisionAdditivityProbe(), SimulatedAdapter(sim_jev()))
    assert res.status == "ok"
    est = _estimate(res, "tokenizer.chat_template_overhead")
    assert est is not None and abs(int(est.value) - 260) <= 2, est
    # State billed once ⇒ evidence favours independent (shared-prefix) branches.
    assert _winner("decision.question_coupling", res.evidence) == "independent"
    _validate_evidence(res)
    assert res.n_requests <= 30


def test_p18_question_isolation_is_independent():
    add, _ = _run(p18.DecisionAdditivityProbe(), SimulatedAdapter(sim_jev()))
    iso, _ = _run(p18.DecisionIsolationProbe(), SimulatedAdapter(sim_jev()))
    assert iso.status == "ok"
    ev = [e for e in iso.evidence if e.property == "decision.question_coupling"][0]
    # sibling secret invisible, state secret visible
    assert ev.stats["d_sibling"] < 0.02 <= ev.stats["d_state"]
    assert _winner("decision.question_coupling", add.evidence + iso.evidence) == "independent"
    _validate_evidence(iso)
    assert iso.n_requests <= 30


def test_p18_choice_set_is_joint_softmax():
    res, _ = _run(p18.DecisionChoiceSetProbe(), SimulatedAdapter(sim_jev()))
    assert res.status == "ok"
    sm = [e for e in res.evidence if e.property == "decision.scoring_mechanism"][0]
    assert sm.stats["mass_taken"] > 0.05, "adding options should redistribute probability mass"
    assert _winner("decision.scoring_mechanism", res.evidence) == "single_pass_heads"
    _validate_evidence(res)
    assert res.n_requests <= 30


def test_p18_option_position_is_semantic():
    res, _ = _run(p18.DecisionOptionPositionProbe(), SimulatedAdapter(sim_jev()))
    assert res.status == "ok"
    assert _winner("decision.option_processing", res.evidence) == "semantic"
    _validate_evidence(res)
    assert res.n_requests <= 30


def test_p18_probability_grid_is_two_decimals():
    res, _ = _run(p18.DecisionProbabilityGridProbe(), SimulatedAdapter(sim_jev()))
    est = _estimate(res, "decision.probability_resolution")
    assert est is not None and abs(float(est.value) - 0.01) < 1e-9, est
    assert res.n_requests <= 30


def test_p18_output_readout_is_non_autoregressive():
    res, _ = _run(p18.DecisionReadoutProbe(), SimulatedAdapter(sim_jev()))
    assert res.status == "ok"
    assert _winner("output.head", res.evidence) == "classification_heads"
    assert _winner("decision.scoring_mechanism", res.evidence) == "single_pass_heads"
    _validate_evidence(res)
    assert res.n_requests <= 30


def test_p18_probes_applicable_to_decision_only():
    ctx_dec = ProbeContext(SimulatedAdapter(sim_jev()))
    ctx_gen = ProbeContext(SimulatedAdapter(sim_decoder_dense()))
    for cls in (p18.DecisionAdditivityProbe, p18.DecisionIsolationProbe,
                p18.DecisionChoiceSetProbe, p18.DecisionOptionPositionProbe,
                p18.DecisionProbabilityGridProbe, p18.DecisionReadoutProbe):
        assert cls().applicable(ctx_dec)[0] is True
        assert cls().applicable(ctx_gen)[0] is False


# =========================================================================== P19
def _run_p19(cfg_factory, long_sweep: bool = True) -> list:
    """Run both P19 probes against a fresh sim (with a multi-token responder); return evidence.
    long_sweep reaches ~32k tokens, where quadratic attention becomes visible at all."""
    adapter = SimulatedAdapter(cfg_factory())
    adapter.register_responder(lambda _p: _LONG_REPLY)
    conf = {"fast": True}
    if long_sweep:
        conf["p19_word_counts"] = [200, 1_000, 3_000, 6_000, 12_000, 24_000]
    ctx = ProbeContext(adapter, config=conf, seed=7)
    evidence = []
    reqs = 0
    for cls in (p19.GenerationCapabilityProbe, p19.PrefillScalingProbe):
        res = asyncio.run(cls().run(ctx))
        evidence += res.evidence
        _validate_evidence(res)
        reqs = ctx.n_requests
    assert reqs <= 30, f"P19 fast budget exceeded: {reqs}"
    return evidence


def test_p19_decoder_only_family():
    ev = _run_p19(sim_decoder_dense)
    assert _winner("arch.family", ev) == "decoder_only"
    assert _winner("output.head", ev) == "generative_lm_head"
    assert _winner("train.objective", ev) == "causal_lm"
    assert _winner("serving.streaming_granularity", ev) == "token"


def test_p19_ssm_or_linear_family():
    # A long (~72k-token) sweep with zero curvature is strong physical evidence against full
    # attention. From a 5% SSM prior (vs 62% decoder) one probe should shift belief a lot and
    # favour SSM in likelihood — but need not overturn the prior on its own.
    ev = _run_p19(sim_ssm)
    prior = normalized_priors(PROPERTIES["arch.family"])
    post = _posterior("arch.family", ev)
    assert post["ssm_or_linear"] >= 3 * prior["ssm_or_linear"], post
    assert post["ssm_or_linear"] / post["decoder_only"] >=         3 * prior["ssm_or_linear"] / prior["decoder_only"]
    assert _winner("output.head", ev) == "generative_lm_head"


def test_p19_short_sweep_is_only_weak_evidence():
    # a fast, short sweep must not strongly claim SSM just because prefill looks linear
    adapter = SimulatedAdapter(sim_ssm())
    ctx = ProbeContext(adapter, config={"fast": True}, seed=7)
    res = asyncio.run(p19.PrefillScalingProbe().run(ctx))
    fam = [e for e in res.evidence if e.property == "arch.family"]
    assert all(e.strength <= 0.3 for e in fam), [e.strength for e in fam]
    for e in fam:
        assert e.likelihoods.get("decoder_only", 0) >= 0.3   # never floored by a linear fit


def test_p19_encoder_only_family():
    ev = _run_p19(sim_encoder)
    assert _winner("arch.family", ev) == "encoder_only"
    assert _winner("output.head", ev) == "classification_heads"
    assert _winner("train.objective", ev) in ("masked_lm", "contrastive_or_classifier")


# =========================================================================== P20
def test_p20_input_is_text_only():
    adapter = SimulatedAdapter(sim_decoder_dense())
    res, ctx = _run(p20.MultimodalInputProbe(), adapter)
    assert res.status == "ok"
    assert _winner("modal.input", res.evidence) == "text_only"
    assert _winner("modal.vision_encoder", res.evidence) == "none"
    # Documented moderate-strength path (sim ignores parts rather than rejecting them).
    mi = [e for e in res.evidence if e.property == "modal.input"][0]
    assert mi.strength < 0.6, "ignored-not-rejected should be moderate strength"
    _validate_evidence(res)
    assert res.n_requests <= 30


def test_p20_output_is_text_only():
    res, _ = _run(p20.MultimodalOutputProbe(), SimulatedAdapter(sim_decoder_dense()))
    assert res.status == "ok"
    assert _winner("modal.output", res.evidence) == "text_only"
    _validate_evidence(res)
    assert res.n_requests <= 30


def test_p20_input_text_only_when_not_fast():
    res, _ = _run(p20.MultimodalInputProbe(), SimulatedAdapter(sim_decoder_dense()), fast=False)
    assert _winner("modal.input", res.evidence) == "text_only"


def test_p20_requires_generative():
    assert p20.MultimodalInputProbe().kinds == ("generative",)
    ctx_dec = ProbeContext(SimulatedAdapter(sim_jev()))
    assert p20.MultimodalInputProbe().applicable(ctx_dec)[0] is False
