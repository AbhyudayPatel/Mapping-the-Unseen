"""P08 — Speculative / multi-token decoding (streaming inter-token latency).

Speculative decoding (a small draft model proposes several tokens that the target verifies
in one pass) is nearly free on *predictable* text — most drafts are accepted, so tokens
arrive in bursts and the effective per-token latency drops sharply — but gives no speed-up
on *high-entropy* text where drafts are rejected.  So we contrast two streamed generations:

* **predictable** — ask the model to count / repeat, producing low-entropy output;
* **high-entropy** — ask for random hex, producing high-entropy output.

If per-token latency is much lower on the predictable stream (and its inter-token latencies
are bursty / bimodal), the serving stack is doing speculative (or other multi-token)
decoding.  ⇒ ``serving.speculative_decoding``; the latency ratio is
``serving.spec_decoding_speedup``.
"""
from __future__ import annotations

import numpy as np

from forensics.probes.base import (
    STATISTICAL, WEAK, BudgetExhausted, Estimate, Evidence, Probe, ProbeContext,
    ProbeResult, register,
)
from forensics.core.types import GenRequest
from forensics.stats import bootstrap_ci, burstiness, permutation_test

_PRED = ("Count upward one number per token, slowly and steadily: "
         "1 2 3 4 5 6 7 8 9 10 11 12 and keep going in the same simple pattern.")
_ENT = ("Output a stream of completely random hexadecimal tokens with no pattern "
        "whatsoever, e.g. 9f 3a c1 07 e2 ... just random hex, nothing else.")


def _visible(resp) -> int:
    u = resp.usage
    if u.raw and isinstance(u.raw, dict) and u.raw.get("visible_tokens") is not None:
        return int(u.raw["visible_tokens"])
    return int((u.completion_tokens or 0) - (u.reasoning_tokens or 0))


def _buffered(resp) -> bool:
    """Chunks flushed in bursts (median gap < 1 ms) are proxy/CDN buffering, not decode
    steps — client timing then says nothing about speculative decoding."""
    gaps = resp.timing.inter_token_latencies()
    return bool(gaps) and float(np.median(gaps)) < 0.001


def _per_token_s(resp) -> float | None:
    from forensics.probes.p07_timing import _server_tbt_s
    tbt = _server_tbt_s(resp)            # server-reported time-between-tokens (Azure)
    if tbt is not None:
        return tbt
    if _buffered(resp):
        return None
    t = resp.timing
    if t.t_first_token is None or t.t_end is None:
        return None
    n = _visible(resp)
    if n < 2:
        return None
    return max(0.0, (t.t_end - t.t_first_token)) / (n - 1)


def _token_itls(resp) -> list[float]:
    """Per-token inter-arrival series: a chunk delivering k tokens => one gap + (k-1) ~zeros."""
    out: list[float] = []
    prev = resp.timing.t_first_token
    for c in resp.timing.chunks:
        if not c.text:
            continue
        if prev is not None:
            out.append(max(0.0, c.t - prev))
        k = c.n_tokens or 1
        out.extend([0.0] * max(0, k - 1))
        prev = c.t
    return out


@register
class SpeculativeDecodingProbe(Probe):
    id = "P08.serving.speculative"
    title = "Speculative / multi-token decoding via ITL on predictable vs random text"
    category = "serving"
    kinds = ("generative",)
    informs = {"serving.speculative_decoding": 0.8, "serving.streaming_granularity": 0.3}
    estimates = ("serving.spec_decoding_speedup",)
    requires = ("streaming",)
    cost = 12

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        res = self.result()
        reps = ctx.scale(5, minimum=3)
        max_tokens = 64

        pred_pt: list[float] = []
        ent_pt: list[float] = []
        pred_itls: list[float] = []
        ent_itls: list[float] = []
        rows = []
        try:
            for r in range(reps):
                rp = await ctx.gen(GenRequest.chat(_PRED, max_tokens=max_tokens, stream=True,
                                                   tag=self.id))
                re_ = await ctx.gen(GenRequest.chat(_ENT, max_tokens=max_tokens, stream=True,
                                                    tag=self.id))
                pp, ep = _per_token_s(rp), _per_token_s(re_)
                if pp is not None:
                    pred_pt.append(pp)
                    if not _buffered(rp):
                        pred_itls.extend(_token_itls(rp))
                if ep is not None:
                    ent_pt.append(ep)
                    if not _buffered(re_):
                        ent_itls.extend(_token_itls(re_))
                rows.append({"rep": r, "pred_per_tok_ms": None if pp is None else pp * 1000,
                             "ent_per_tok_ms": None if ep is None else ep * 1000,
                             "pred_chunks": len(rp.timing.chunks),
                             "ent_chunks": len(re_.timing.chunks)})
        except BudgetExhausted as e:
            res.notes.append(f"budget exhausted: {e}")
            res.status = "partial"

        res.tables["itl"] = rows
        if not pred_pt or not ent_pt:
            res.status = res.status if res.status == "partial" else "skipped"
            res.notes.append("no usable decode timing: the stream is buffered (chunks arrive in "
                             "bursts) and no server time-between-tokens is reported — "
                             "speculative decoding is not identifiable from this interface")
            res.n_requests = ctx.n_requests
            return res

        med_pred = float(np.median(pred_pt))
        med_ent = float(np.median(ent_pt))
        speedup = med_ent / med_pred if med_pred > 0 else 1.0
        burst = burstiness(pred_itls) if len(pred_itls) >= 4 else {"bimodality_coef": float("nan"),
                                                                   "cv": float("nan")}
        bimodal = burst["bimodality_coef"] == burst["bimodality_coef"] and burst["bimodality_coef"] > 0.555

        p = permutation_test(np.array(ent_pt), np.array(pred_pt), stat="mean_diff",
                             n=2000, rng=int(ctx.rng.integers(1 << 30))) if (
            len(pred_pt) >= 2 and len(ent_pt) >= 2) else 1.0

        detected = speedup >= 1.5 and (bimodal or p < 0.1)
        if detected:
            yes, no = 1.0, _floor(min(0.8, 1.0 / max(speedup, 1.0)))
        else:
            yes, no = 0.3, 1.0
        res.add_evidence(Evidence(
            property="serving.speculative_decoding",
            likelihoods={"yes": _floor(yes), "no": _floor(no)},
            summary=f"predictable decode {speedup:.2f}x faster than random"
                    + (", bursty ITL" if bimodal else ""),
            level="inferred", strength=0.8,
            stats={"speedup": speedup, "p_value": p, "bimodality_coef": burst["bimodality_coef"],
                   "cv": burst["cv"], "median_pred_ms": med_pred * 1000,
                   "median_ent_ms": med_ent * 1000, "test": "permutation(mean_diff)"},
            alternatives=["draft accepted rate depends on content, not only on serving",
                          "output-length caps or KV growth could confound raw ITL"]))

        # streaming granularity hint (bursts => multi-token chunks)
        pred_chunk_toks = np.mean([r["pred_chunks"] for r in rows]) if rows else 0
        if detected:
            res.add_evidence(Evidence(
                property="serving.streaming_granularity",
                likelihoods={"multi_token_chunks": _floor(0.8), "token": _floor(0.3)},
                summary="predictable stream delivers multi-token bursts", level="inferred",
                strength=0.4))

        lo, hi, _ = bootstrap_ci(np.array(ent_pt), np.median, n_boot=1000, level=0.9,
                                 rng=int(ctx.rng.integers(1 << 30)))
        # convert the entropy-median CI into a speedup CI via the predictable median
        res.add_estimate(Estimate(
            quantity="serving.spec_decoding_speedup", value=round(speedup, 3),
            ci=(lo / med_pred if med_pred > 0 else speedup,
                hi / med_pred if med_pred > 0 else speedup),
            level="estimated" if detected else "inferred",
            method="median per-token latency ratio (high-entropy / predictable)",
            identifiability=STATISTICAL if detected else WEAK, unit="x",
            assumptions=["equal prompt lengths so decode baselines match"]))

        res.series["per_token"] = {
            "x": ["predictable", "high_entropy"], "y": [med_pred * 1000, med_ent * 1000],
            "xlabel": "content", "ylabel": "per-token ms", "kind": "bar"}
        res.shared["speculative"] = {"detected": bool(detected), "speedup": speedup}
        res.n_requests = ctx.n_requests
        return res


def _floor(v: float) -> float:
    return max(0.02, min(1.0, float(v)))
