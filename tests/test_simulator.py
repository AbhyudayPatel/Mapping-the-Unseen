"""Physics-fidelity tests for the ground-truth SimulatedAdapter.

These assert that the *observable* side-channels really encode the known hidden config, so
that probe tests validated against this simulator are meaningful:

* logit matrix rank == hidden_size (softmax bottleneck, P03),
* prefill timing is quadratic for a decoder but linear for an SSM (P07),
* bf16 logits land on a coarser lattice than fp32 (P04),
* an over-length prompt returns an error whose text carries the context limit,
* a cached second call is faster than the first.
"""
from __future__ import annotations

import asyncio

import numpy as np

from forensics.core.adapters.simulated import (
    SimConfig, SimulatedAdapter, quantize_logits, sim_decoder_dense, sim_jev, sim_ssm,
    to_bf16,
)
from forensics.core.types import DecisionQuestion, DecisionRequest, GenRequest


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- P03: rank == d
def test_logit_matrix_rank_equals_hidden_size():
    d = 32
    cfg = SimConfig(arch_family="decoder_only", hidden_size=d, vocab_size=4000,
                    logit_dtype="fp32", determinism_seed=7)
    adapter = SimulatedAdapter(cfg)

    m = 128                      # columns (> d)
    n_prompts = 130              # rows (> d)
    cols = list(range(m))
    Z = np.stack([adapter.synthesize_logits(f"unique probe prompt #{i} lorem ipsum", cols)
                  for i in range(n_prompts)])

    # kill the constant unembedding-bias / offset direction, then SVD
    Zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    ratios = s[:-1] / np.maximum(s[1:], 1e-30)
    est_rank = int(np.argmax(ratios)) + 1

    assert est_rank == d, f"SVD recovered rank {est_rank}, expected {d}"
    assert s[d - 1] / s[d] > 1e3, "singular-value gap at d is not sharp"


# --------------------------------------------------------------------------- P07: prefill curvature
def _prefill_curvature(cfg: SimConfig) -> float:
    """Fit server_ms = a + b*p + c*p^2 over a prompt-length sweep; return c."""
    cfg.timing_jitter_s = 0.0
    cfg.prompt_caching = False                  # defeat caching (shared prefixes would flatten it)
    cfg.responders = [lambda _p: "ok"]          # fixed 1-token reply -> isolate prefill
    adapter = SimulatedAdapter(cfg)
    ps, ts = [], []
    for k in (60, 120, 240, 480, 960, 1600, 2400, 3200):
        prompt = ("token " * k).strip()
        resp = _run(adapter.generate(GenRequest(prompt=prompt, max_tokens=1)))
        ps.append(resp.usage.prompt_tokens)
        ts.append(resp.timing.server_ms)
    c, b, a = np.polyfit(np.array(ps, float), np.array(ts, float), 2)
    return float(c)


def test_prefill_quadratic_decoder_linear_ssm():
    c_dec = _prefill_curvature(sim_decoder_dense())
    c_ssm = _prefill_curvature(sim_ssm())
    assert c_dec > 0, f"decoder prefill should be quadratic (c={c_dec:.3e})"
    assert abs(c_ssm) < c_dec * 1e-3, f"ssm prefill should be ~linear (c={c_ssm:.3e} vs {c_dec:.3e})"


# --------------------------------------------------------------------------- P04: dtype lattice
def test_bf16_coarser_lattice_than_fp32():
    cfg = SimConfig(hidden_size=64, vocab_size=8000, determinism_seed=3)
    adapter = SimulatedAdapter(cfg)
    z = adapter.synthesize_logits("precision staircase probe", list(range(3000)),
                                  apply_dtype=False)

    def min_gap(vals: np.ndarray) -> float:
        u = np.unique(np.sort(vals))
        return float(np.min(np.diff(u)))

    b = to_bf16(z)
    f = quantize_logits(z, "fp32")
    assert min_gap(b) > min_gap(f), "bf16 tread should exceed fp32 tread"
    assert np.unique(b).size < np.unique(f).size, "bf16 should tie more values than fp32"


# --------------------------------------------------------------------------- context overflow
def test_context_overflow_error_contains_limit():
    limit = 100
    cfg = SimConfig(max_context=limit, chat_template_overhead=0, per_message_overhead=0)
    adapter = SimulatedAdapter(cfg)
    prompt = "word " * 300                      # ~300 tokens > 100
    resp = _run(adapter.generate(GenRequest(prompt=prompt, max_tokens=8)))
    assert not resp.ok
    assert resp.status == 400
    assert str(limit) in (resp.error or ""), f"limit {limit} not in error: {resp.error!r}"


# --------------------------------------------------------------------------- prompt cache
def test_cached_second_call_is_faster():
    cfg = sim_decoder_dense()
    cfg.timing_jitter_s = 0.0
    cfg.responders = [lambda _p: "ok"]
    adapter = SimulatedAdapter(cfg)
    prompt = "context sentence number seven. " * 90     # comfortably > cache_block_tokens

    first = _run(adapter.generate(GenRequest(prompt=prompt, max_tokens=1)))
    second = _run(adapter.generate(GenRequest(prompt=prompt, max_tokens=1)))

    assert (first.usage.cached_tokens or 0) == 0
    assert (second.usage.cached_tokens or 0) > 0
    assert second.timing.server_ms < first.timing.server_ms, (
        f"cache hit not faster: {second.timing.server_ms} !< {first.timing.server_ms}")


# --------------------------------------------------------------------------- smoke: logprobs
def test_generate_logprobs_shape_and_streaming():
    adapter = SimulatedAdapter(sim_decoder_dense())
    adapter.register_responder(lambda _p: "hello world from the simulator")
    resp = _run(adapter.generate(GenRequest.chat("hi", logprobs=True, top_logprobs=5,
                                                 max_tokens=8, stream=True)))
    assert resp.ok and resp.tokens
    assert all(len(t.top) == 5 for t in resp.tokens)
    assert all(t.logprob <= 1e-6 for t in resp.tokens)      # log-probabilities are <= 0
    assert resp.timing.chunks and resp.timing.t_first_token is not None
    assert resp.headers.get("x-sim-arch") == "decoder_only"


# --------------------------------------------------------------------------- smoke: digit-split
def test_digit_split_tokenizer_counts_single_digits():
    adapter = SimulatedAdapter(sim_jev())
    assert adapter.tok.count("1234567890") == 10        # one token per digit
    assert adapter.tok.count("hello world") == 2


# --------------------------------------------------------------------------- smoke: decision
def test_decision_choice_confidence_and_sum():
    adapter = SimulatedAdapter(sim_jev())
    assert adapter.static_capabilities().kind == "decision"
    req = DecisionRequest(
        state="The customer was charged twice for the same subscription.",
        questions={"route": DecisionQuestion(
            type="choice", instructions="Which team should handle this?",
            criteria={"billing": "money issues", "sales": "purchases",
                      "technical": "bugs"})},
    )
    resp = _run(adapter.decide(req))
    assert resp.ok
    ans = resp.answers["route"]
    assert ans.value in {"billing", "sales", "technical"}
    assert abs(sum(ans.probabilities.values()) - 1.0) < 1e-9
    assert 0.0 <= ans.confidence <= 1.0
    assert resp.usage.prompt_tokens and resp.usage.prompt_tokens > 260
