"""Group-D probe tests (P14 alignment, P15 sampling, P16 topology, P17 metamorphic).

Every probe is validated against the SimulatedAdapter with KNOWN ground truth supplied via
``SimConfig.refusal_topics`` and ``SimConfig.responders``.  Each test asserts the probe
moves belief in the RIGHT direction (its Evidence likelihood ratio favours the true value)
or recovers the right structural fact.  All probes run with ``config={'fast': True}``.
"""
from __future__ import annotations

import asyncio
import hashlib
import re

import numpy as np

from forensics.core.adapters.simulated import (
    SimConfig, SimulatedAdapter, sim_decoder_dense, sim_decoder_moe,
)
from forensics.probes.base import ProbeContext
from forensics.probes.p14_alignment import (
    BaseVsInstruct, EntropyCollapse, RefusalStyle, SelfIdentification,
)
from forensics.probes.p15_sampling import SamplingPenalties, TemperatureFidelity
from forensics.probes.p16_topology import (
    LatencyMixture, RetrievalDetection, SafetyMiddleware, ToolOrchestration,
)
from forensics.probes.p17_metamorphic import MetamorphicInvariance


def _run(coro):
    return asyncio.run(coro)


def _ctx(adapter, seed: int = 0, **cfg) -> ProbeContext:
    c = {"fast": True}
    c.update(cfg)
    return ProbeContext(adapter, config=c, seed=seed)


def _ev(res, prop):
    for e in res.evidence:
        if e.property == prop:
            return e
    raise AssertionError(f"no evidence for {prop}; got {[e.property for e in res.evidence]}")


def _dense(**mut) -> SimConfig:
    cfg = sim_decoder_dense()
    for k, v in mut.items():
        setattr(cfg, k, v)
    return cfg


# ============================================================ P14 alignment
def test_p14_base_vs_instruct_direction():
    # instruct: obeys the strict one-word format
    ad = SimulatedAdapter(_dense())
    ad.register_responder(lambda p: "Blue")
    res = _run(BaseVsInstruct().run(_ctx(ad)))
    assert res.status == "ok"
    ev = _ev(res, "post.instruction_tuning")
    assert ev.likelihoods["instruct"] > ev.likelihoods["base"]

    # base: continues with long prose, ignoring the format
    adb = SimulatedAdapter(_dense(instruction_tuned=False))
    adb.register_responder(
        lambda p: "and then the long tale wandered on across many rolling green hills "
                  "and distant valleys beneath a wide and pale evening sky")
    resb = _run(BaseVsInstruct().run(_ctx(adb)))
    evb = _ev(resb, "post.instruction_tuning")
    assert evb.likelihoods["base"] > evb.likelihoods["instruct"]


def test_p14_refusal_detects_safety_rl():
    ad = SimulatedAdapter(_dense(refusal_topics=["bomb", "malware", "poison"]))
    ad.register_responder(lambda p: "Sure, here is a helpful and friendly answer for you.")
    res = _run(RefusalStyle().run(_ctx(ad)))
    assert res.status == "ok"
    ev = _ev(res, "post.preference_optimization")
    assert ev.likelihoods["rlhf_or_dpo"] > ev.likelihoods["none"]
    rows = {r["class"]: r for r in res.tables["refusal"]}
    assert rows["unsafe"]["rate"] == 1.0
    assert rows["benign"]["rate"] == 0.0


def test_p14_identity_distillation_signal():
    ad = SimulatedAdapter(_dense())
    ad.register_responder(lambda p: "I am ChatGPT, a large language model trained by OpenAI.")
    res = _run(SelfIdentification().run(_ctx(ad)))
    ev = _ev(res, "post.distillation_signal")
    assert ev.likelihoods["distilled_from_other"] > ev.likelihoods["none"]
    assert res.shared.get("self_identified_family") == "openai"

    ad2 = SimulatedAdapter(_dense())
    ad2.register_responder(lambda p: "I am a helpful AI assistant, here to help you today.")
    res2 = _run(SelfIdentification().run(_ctx(ad2)))
    ev2 = _ev(res2, "post.distillation_signal")
    assert ev2.likelihoods["none"] > ev2.likelihoods["distilled_from_other"]


def test_p14_entropy_collapse():
    # collapsed: always the same number => RLHF mode collapse
    ad = SimulatedAdapter(_dense())
    ad.register_responder(lambda p: "7")
    res = _run(EntropyCollapse().run(_ctx(ad)))
    ev = _ev(res, "post.preference_optimization")
    assert ev.likelihoods["rlhf_or_dpo"] > ev.likelihoods["none"]
    assert res.tables["random_number"][0]["entropy_normalised"] < 0.2

    # diverse: uniform-ish => little collapse
    rng = np.random.default_rng(5)
    add = SimulatedAdapter(_dense())
    add.register_responder(lambda p: str(int(rng.integers(1, 101))))
    resd = _run(EntropyCollapse().run(_ctx(add)))
    evd = _ev(resd, "post.preference_optimization")
    assert evd.likelihoods["none"] > evd.likelihoods["rlhf_or_dpo"]


# ============================================================ P15 sampling
def _softmax_sampler(adapter, seed: int = 999):
    rng = np.random.default_rng(seed)

    def sampler(rendered: str):
        h = adapter._hidden(rendered + "\x1f#0")
        ids, z = adapter._pool_logits(h)
        z = np.asarray(z, dtype=float)
        p = np.exp(z - z.max())
        p = p / p.sum()
        j = int(rng.choice(len(ids), p=p))
        s = adapter.tok.token_str(ids[j])
        return s if s.strip() else "x"

    return sampler


def test_p15_temperature_fidelity_matches_softmax():
    ad = SimulatedAdapter(_dense())               # batch_invariant=True
    ad.register_responder(_softmax_sampler(ad))
    res = _run(TemperatureFidelity().run(_ctx(ad)))
    assert res.status == "ok"
    s = res.tables["summary"][0]
    assert s["tv_distance"] < 0.35
    assert s["temperature_fidelity"] > 0.6
    assert s["mode_agreement"] is True
    ev = _ev(res, "serving.batch_invariant")
    assert ev.likelihoods["yes"] > ev.likelihoods["no"]   # dense is batch-invariant


def test_p15_batch_noninvariant_moe():
    ad = SimulatedAdapter(sim_decoder_moe())      # batch_invariant=False (load jitter)
    res = _run(TemperatureFidelity().run(_ctx(ad)))
    ev = _ev(res, "serving.batch_invariant")
    assert ev.likelihoods["no"] > ev.likelihoods["yes"]


def test_p15_penalties_on_deterministic_sim():
    ad = SimulatedAdapter(_dense())
    res = _run(SamplingPenalties().run(_ctx(ad)))
    checks = {r["check"]: r for r in res.tables["sampling_checks"]}
    assert checks["temperature_0_determinism"]["result"] is True
    # the sim implements neither nucleus truncation nor a repetition penalty
    assert checks["top_p_truncation"]["result"] in ("indeterminate", "none")
    assert checks["repetition_penalty"]["result"] in (None, False)
    ev = _ev(res, "serving.batch_invariant")
    assert ev.likelihoods["yes"] > ev.likelihoods["no"]


# ============================================================ P16 topology
def test_p16_latency_single_model_unimodal():
    cfg = _dense(timing_jitter_s=0.002)
    ad = SimulatedAdapter(cfg)
    ad.register_responder(lambda p: "ok")         # constant output length => one mode
    res = _run(LatencyMixture().run(_ctx(ad, seed=1)))
    assert res.shared["latency_modes"] == 1
    ev = _ev(res, "system.topology")
    assert ev.likelihoods["single_model"] > ev.likelihoods["router_or_cascade"]


def test_p16_latency_router_bimodal():
    cfg = _dense(timing_jitter_s=0.002)
    ad = SimulatedAdapter(cfg)

    def router(rendered: str):
        # digit-bearing prompts get a long (slow) reply => a second latency mode
        if any(c.isdigit() for c in rendered):
            return "this is a considerably longer response describing many details at length here"
        return "ok"

    ad.register_responder(router)
    res = _run(LatencyMixture().run(_ctx(ad, seed=2)))
    assert res.shared["latency_modes"] >= 2
    ev = _ev(res, "system.topology")
    assert ev.likelihoods["router_or_cascade"] > ev.likelihoods["single_model"]


def test_p16_safety_full_latency_not_prefilter():
    cfg = _dense(refusal_topics=["nerve agent", "hack", "thermite"], timing_jitter_s=0.0)
    ad = SimulatedAdapter(cfg)
    ad.register_responder(lambda p: "Here is a clear, friendly explanation of the topic.")
    res = _run(SafetyMiddleware().run(_ctx(ad)))
    assert res.shared["refusal_rate_borderline"] == 1.0
    ev = _ev(res, "system.safety_middleware")
    # borderline refused at full model latency => a pre-filter is ruled out
    assert ev.likelihoods["pre_filter"] < ev.likelihoods["none"]
    assert ev.likelihoods["pre_filter"] < ev.likelihoods["post_filter"]
    rows = {r["class"]: r for r in res.tables["safety"]}
    assert rows["benign"].get("median_latency_s") is not None


def test_p16_retrieval_detection():
    ad = SimulatedAdapter(_dense())
    ad.register_responder(
        lambda p: "Yes, this week the Fictionland Prize went to Dr Ada Vance for quantum art.")
    res = _run(RetrievalDetection().run(_ctx(ad)))
    ev = _ev(res, "system.retrieval")
    assert ev.likelihoods["retrieval_augmented"] > ev.likelihoods["none"]

    ad2 = SimulatedAdapter(_dense())
    ad2.register_responder(
        lambda p: "I do not have information about events after my training cutoff.")
    res2 = _run(RetrievalDetection().run(_ctx(ad2)))
    ev2 = _ev(res2, "system.retrieval")
    assert ev2.likelihoods["none"] > ev2.likelihoods["retrieval_augmented"]


def test_p16_tools_exact_arithmetic():
    ad = SimulatedAdapter(_dense())

    def calc(rendered: str):
        m = re.search(r"Compute (\d+) \* (\d+)", rendered)
        return str(int(m.group(1)) * int(m.group(2))) if m else None

    ad.register_responder(calc)
    res = _run(ToolOrchestration().run(_ctx(ad)))
    ev = _ev(res, "system.tool_orchestration")
    assert ev.likelihoods["yes"] > ev.likelihoods["no"]

    ad2 = SimulatedAdapter(_dense())
    ad2.register_responder(lambda p: "12345")     # wrong answers => no tool
    res2 = _run(ToolOrchestration().run(_ctx(ad2)))
    ev2 = _ev(res2, "system.tool_orchestration")
    assert ev2.likelihoods["no"] > ev2.likelihoods["yes"]


# ============================================================ P17 metamorphic
def test_p17_metamorphic_stable_answers_low_distance():
    ad = SimulatedAdapter(_dense())
    ad.register_responder(lambda p: "answer")     # invariant to every transform
    res = _run(MetamorphicInvariance().run(_ctx(ad)))
    fp = res.shared["fingerprint_metamorphic"]
    assert isinstance(fp, list) and len(fp) == 6            # populated fingerprint vector
    assert max(fp) < 0.2                                    # stable answers => low drift
    inv_mean = res.tables["summary"][0]["mean_answer_instability"]
    assert inv_mean < 0.05
    ev = _ev(res, "post.instruction_tuning")
    assert ev.likelihoods["instruct"] > ev.likelihoods["base"]


def test_p17_metamorphic_discriminates_brittle_model():
    ad = SimulatedAdapter(_dense())

    def brittle(rendered: str):
        return "w" + hashlib.blake2b(rendered.encode("utf-8"), digest_size=2).hexdigest()

    ad.register_responder(brittle)
    res = _run(MetamorphicInvariance().run(_ctx(ad)))
    fp = res.shared["fingerprint_metamorphic"]
    assert len(fp) == 6
    sen_mean = res.tables["summary"][0]["mean_answer_instability"]
    assert sen_mean > 0.5                                   # every transform changes the answer
