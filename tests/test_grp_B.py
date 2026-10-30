"""Ground-truth tests for probe group B (P05-P09) against the SimulatedAdapter.

Each probe is run in fast mode against a simulator whose hidden config is known, and we
assert the probe recovers the right estimate / moves belief in the right direction.  Where a
probe needs the model to actually answer its prompt format (needle recall, predictable vs
random decoding), we register a ``SimConfig.responder`` that emulates the ground-truth
behaviour so the physically-modelled side-channels (window recall, spec-decode speed-up,
prompt-cache accounting) fire.
"""
from __future__ import annotations

import asyncio
import random
import re

import numpy as np

from forensics.probes.base import ProbeContext
from forensics.probes.p05_context import ContextLimitsProbe
from forensics.probes.p06_positional import NeedleRecallProbe
from forensics.probes.p07_timing import TimingScalingProbe
from forensics.probes.p08_speculative import SpeculativeDecodingProbe
from forensics.probes.p09_caching import PromptCacheProbe
from forensics.core.adapters.simulated import (
    SimConfig, SimulatedAdapter, sim_decoder_dense, sim_ssm,
)


# --------------------------------------------------------------------------- helpers
def _run(coro):
    return asyncio.run(coro)


def _ctx(adapter, seed: int = 0, **cfg) -> ProbeContext:
    conf = {"fast": True}
    conf.update(cfg)
    return ProbeContext(adapter, config=conf, seed=seed)


def _est(res, quantity):
    for e in res.estimates:
        if e.quantity == quantity:
            return e
    return None


def _ev(res, prop):
    for e in res.evidence:
        if e.property == prop:
            return e
    return None


# =========================================================================== P05
def test_p05_recovers_input_limit_from_error_and_output_cap():
    cfg = sim_decoder_dense()
    cfg.max_context = 8192
    cfg.max_output = 4096
    cfg.max_tokens_policy = "clamp"      # provider silently caps generation (e.g. Fireworks)
    adapter = SimulatedAdapter(cfg)
    adapter.register_responder(lambda p: "word " * 9000)   # long reply -> output cap visible

    # allow generation up to 8k so the 4096 completion cap is actually reached
    res = _run(ContextLimitsProbe().run(_ctx(adapter, seed=1, p05_max_gen_tokens=8192)))
    assert res.status == "ok"

    ein = _est(res, "context.max_input_tokens")
    assert ein is not None
    assert ein.value == 8192                       # read exactly from the error text
    assert ein.level == "observed" and ein.identifiability == 1

    eout = _est(res, "context.max_output_tokens")
    assert eout is not None
    assert 4000 <= eout.value <= 4200              # completion saturates at max_output

    assert any("explicit_error" in n for n in res.notes)
    assert res.n_requests <= 30


def test_p05_oracle_reads_context_from_vllm_style_error():
    # vLLM/SGLang/serverless: an absurd max_tokens is rejected WITH the context limit
    cfg = sim_decoder_dense()
    cfg.max_context = 131_072
    res = _run(ContextLimitsProbe().run(_ctx(SimulatedAdapter(cfg), seed=3)))
    ein = _est(res, "context.max_input_tokens")
    assert ein.value == 131_072 and ein.level == "observed"
    assert res.n_requests <= 12          # no giant prompts needed


def test_p05_fast_mode_reports_lower_bounds_not_false_precision():
    # no limit stated anywhere + a cap on how far we search -> lower bounds with open CIs
    cfg = sim_decoder_dense()
    cfg.max_context = 200_000
    cfg.max_output = 4096
    cfg.max_tokens_policy = "clamp"      # no limit stated anywhere
    adapter = SimulatedAdapter(cfg)
    adapter.register_responder(lambda p: "word " * 9000)
    res = _run(ContextLimitsProbe().run(_ctx(adapter, seed=2, p05_max_probe_tokens=5000)))
    ein = _est(res, "context.max_input_tokens")
    assert ein is not None and ein.ci[1] == float("inf"), ein
    assert ein.level != "observed"
    eout = _est(res, "context.max_output_tokens")
    assert eout is not None and eout.ci[1] == float("inf")   # fast: 1024-token gen cap


# =========================================================================== P06
def _needle_adapter(window):
    cfg = SimConfig(arch_family="decoder_only", hidden_size=512, max_context=30000,
                    sliding_window=window, tokenizer_name="o200k_base")
    adapter = SimulatedAdapter(cfg)
    tok = adapter.tok

    def responder(rendered: str):
        m = re.search(r"harbor passcode is (\d{4,6})", rendered)
        if not m:
            return None
        dist = tok.count(rendered[m.end():])       # tokens between needle and the end
        if window is not None and dist > window:    # needle fell out of the attention window
            return "I do not know."
        return f"The harbor passcode is {m.group(1)}."

    adapter.register_responder(responder)
    return adapter


def test_p06_detects_sliding_window_cliff():
    adapter = _needle_adapter(window=1024)
    res = _run(NeedleRecallProbe().run(_ctx(adapter, seed=3)))

    ev = _ev(res, "attn.long_range")
    assert ev is not None and ev.stats["cliff"] is True
    assert ev.likelihoods["sliding_or_local"] > ev.likelihoods["full_quadratic"]

    sw = _est(res, "context.sliding_window")
    assert sw is not None
    assert 512 <= sw.value <= 1600                 # cliff localises near the true window 1024


def test_p06_full_attention_has_no_cliff():
    adapter = _needle_adapter(window=None)
    res = _run(NeedleRecallProbe().run(_ctx(adapter, seed=3)))

    ev = _ev(res, "attn.long_range")
    assert ev is not None and ev.stats["cliff"] is False
    assert ev.likelihoods["full_quadratic"] > ev.likelihoods["sliding_or_local"]
    assert _est(res, "context.sliding_window") is None


# =========================================================================== P07
def test_p07_attention_family_direction_and_active_params():
    dec_cfg = sim_decoder_dense(); dec_cfg.prompt_caching = False
    ssm_cfg = sim_ssm(); ssm_cfg.prompt_caching = False
    dec = SimulatedAdapter(dec_cfg)
    ssm = SimulatedAdapter(ssm_cfg)

    dec_res = _run(TimingScalingProbe().run(_ctx(dec, seed=5)))
    ssm_res = _run(TimingScalingProbe().run(_ctx(ssm, seed=5)))

    dec_ev = _ev(dec_res, "attn.long_range")
    ssm_ev = _ev(ssm_res, "attn.long_range")
    assert dec_ev is not None and ssm_ev is not None
    # quadratic prefill (decoder) vs linear prefill (ssm): posterior direction differs
    assert dec_ev.likelihoods["full_quadratic"] > ssm_ev.likelihoods["full_quadratic"]
    assert dec_ev.stats["quadratic_significant"] is True
    assert ssm_ev.stats["quadratic_significant"] is False

    # active-parameter inversion: WIDE CI, but it must bracket the simulator's truth
    dec_ap = _est(dec_res, "dims.active_params")
    ssm_ap = _est(ssm_res, "dims.active_params")
    assert dec_ap is not None and dec_ap.ci[0] <= 8.0 <= dec_ap.ci[1]
    assert ssm_ap is not None and ssm_ap.ci[0] <= 2.8 <= ssm_ap.ci[1]

    rtt = _est(dec_res, "timing.rtt_ms")
    assert rtt is not None and rtt.value > 0


# =========================================================================== P08
def _spec_adapter(spec: bool):
    cfg = sim_decoder_dense()
    cfg.speculative_decoding = spec
    cfg.prompt_caching = False
    adapter = SimulatedAdapter(cfg)

    def responder(rendered: str):
        low = rendered.lower()
        if "random hex" in low:                    # genuinely high-entropy output
            rng = random.Random(7)
            return " ".join("%02x" % rng.randint(0, 255) for _ in range(48))
        if "count upward" in low or "repeat" in low:
            return "1 " * 48                        # predictable / low-entropy output
        return None

    adapter.register_responder(responder)
    return adapter


def test_p08_detects_speculative_decoding():
    adapter = _spec_adapter(spec=True)
    res = _run(SpeculativeDecodingProbe().run(_ctx(adapter, seed=2)))
    ev = _ev(res, "serving.speculative_decoding")
    assert ev is not None
    assert res.shared["speculative"]["detected"] is True
    assert ev.likelihoods["yes"] > ev.likelihoods["no"]
    su = _est(res, "serving.spec_decoding_speedup")
    assert su is not None and su.value >= 2.0


def test_p08_no_speculative_decoding():
    adapter = _spec_adapter(spec=False)
    res = _run(SpeculativeDecodingProbe().run(_ctx(adapter, seed=2)))
    ev = _ev(res, "serving.speculative_decoding")
    assert ev is not None
    assert res.shared["speculative"]["detected"] is False
    assert ev.likelihoods["no"] > ev.likelihoods["yes"]
    su = _est(res, "serving.spec_decoding_speedup")
    assert su is not None and su.value < 1.5


# =========================================================================== P09
def _cache_adapter(caching: bool):
    cfg = sim_decoder_dense()
    cfg.prompt_caching = caching
    cfg.cache_block_tokens = 256
    adapter = SimulatedAdapter(cfg)
    adapter.register_responder(lambda p: "ok")
    return adapter


def test_p09_detects_prompt_caching_and_block_size():
    adapter = _cache_adapter(caching=True)
    res = _run(PromptCacheProbe().run(_ctx(adapter, seed=4)))
    ev = _ev(res, "serving.prompt_caching")
    assert ev is not None
    assert res.shared["prompt_cache"]["detected"] is True
    assert ev.likelihoods["yes"] > ev.likelihoods["no"]
    blk = _est(res, "serving.cache_block_tokens")
    assert blk is not None and blk.value == 256


def test_p09_no_prompt_caching_when_disabled():
    adapter = _cache_adapter(caching=False)
    res = _run(PromptCacheProbe().run(_ctx(adapter, seed=4)))
    ev = _ev(res, "serving.prompt_caching")
    assert ev is not None
    assert res.shared["prompt_cache"]["detected"] is False
    assert ev.likelihoods["no"] > ev.likelihoods["yes"]
    assert _est(res, "serving.cache_block_tokens") is None
