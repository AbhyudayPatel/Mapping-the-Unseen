"""Tests for the optional Azure-backed LLM advisor and the P21 experiment generator.

The live check is SKIPPED when Azure credentials are absent (so the suite is green with no
network); with creds it runs a tiny live check (<=5 requests) against the endpoint in
``.env``.  The no-op / import checks always run and never touch the network.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from forensics.config import load_env


def _has_creds() -> bool:
    try:
        load_env()
    except Exception:
        pass
    return bool(os.environ.get("AZURE_OPENAI_BASE_URL")
               and os.environ.get("AZURE_OPENAI_API_KEY"))


_HAS_CREDS = _has_creds()


# --------------------------------------------------------------- offline (always run)
def test_advisor_module_imports_and_default_is_optional():
    # module imports cleanly and make_default_advisor never raises
    from forensics.llm.advisor import LLMAdvisor, LLMConfig, make_default_advisor

    assert LLMAdvisor is not None and LLMConfig is not None
    adv = make_default_advisor()
    # returns an advisor iff creds are present; None otherwise
    if _HAS_CREDS:
        assert adv is not None
        asyncio.run(adv.aclose())
    else:
        assert adv is None


def test_p21_registered_and_noop_without_llm():
    import forensics.probes as p
    from forensics.core.adapters.simulated import SimulatedAdapter, sim_decoder_dense
    from forensics.probes.base import ProbeContext
    from forensics.probes.p21_llm_experiments import LLMExperimentGenerator

    # P21 is picked up by the loader / registry
    reg = p.load_all()
    assert "P21.llm.experiment_generator" in reg

    # with ctx.llm is None the probe is a safe no-op (status skipped, no target requests)
    ctx = ProbeContext(SimulatedAdapter(sim_decoder_dense()), config={"fast": True})
    assert ctx.llm is None
    res = asyncio.run(LLMExperimentGenerator().run(ctx))
    assert res.status == "skipped"
    assert res.n_requests == 0
    assert not res.shared.get("llm_assisted")


def test_judge_json_parsing_is_robust():
    # the JSON extractor tolerates code fences and surrounding prose
    from forensics.llm.advisor import _parse_json

    assert _parse_json('{"label": "yes", "confidence": 0.9}')["label"] == "yes"
    fenced = _parse_json('```json\n{"label":"no","confidence":0.1}\n```')
    assert fenced["label"] == "no"
    noisy = _parse_json('Here you go: {"experiments": [{"name": "x"}]} thanks!')
    assert noisy["experiments"][0]["name"] == "x"
    assert _parse_json("not json at all") is None


# --------------------------------------------------------------- live (skipped w/o creds)
@pytest.mark.skipif(not _HAS_CREDS,
                    reason="Azure creds absent (AZURE_OPENAI_BASE_URL/AZURE_OPENAI_API_KEY)")
def test_advisor_live_tiny():
    from forensics.llm.advisor import make_default_advisor

    adv = make_default_advisor()
    assert adv is not None

    async def _drive():
        # all calls share one event loop (the lazily-created HTTP client binds to it)
        try:
            judged = await adv.judge_freeform(
                "Is this a refusal?", "I cannot help with that request.")
            proposals = await adv.propose_experiments(
                belief_summary={"arch.sparsity": "moe (0.55)",
                                "attn.kv_sharing": "gqa_or_mqa (0.6)"},
                caps_summary={"kind": "generative", "logprobs": True, "logit_bias": True},
                uncertainty=["arch.sparsity", "attn.kv_sharing", "pos.encoding"])
            return judged, proposals
        finally:
            await adv.aclose()

    judged, proposals = asyncio.run(_drive())

    label = str(judged.get("label", "")).lower()
    rationale = str(judged.get("rationale", "")).lower()
    assert judged.get("note") is None, f"judge degraded: {judged}, notes={adv.notes}"
    assert (label in ("yes", "true", "refusal", "refuses")
            or "refus" in (label + rationale)), f"label did not indicate refusal: {judged}"
    assert 0.0 <= judged["confidence"] <= 1.0

    assert len(proposals) >= 1, f"expected >=1 proposal; notes={adv.notes}"
    for key in ("name", "hypothesis", "method", "observable", "distinguishes"):
        assert key in proposals[0]

    assert adv.n_calls <= 5
