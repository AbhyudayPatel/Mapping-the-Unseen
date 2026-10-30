"""P04 — numeric precision (logit dtype) from the ULP lattice of reported logprobs.

Logits are computed in some dtype (fp32/bf16/fp16/fp8) and rounded onto that dtype's grid
*before* the log-softmax, so the set of achievable logit differences is a lattice whose step
is one ULP of the dtype at the relevant magnitude. Within a single position, differences of
top-k logprobs cancel the per-row ``logZ`` offset and equal raw logit differences, so their
finest consistent spacing (relative to the value scale) pins the mantissa-bit count and thus
the dtype (see ``research/techniques_logits_tokenizer.md`` §6):

    relative ULP ~= 2^-mantissa:  fp8 E4M3 ~ 2^-3 (coarse, heavy tie-ing), bf16 ~ 2^-7/2^-8,
    fp16 ~ 2^-10, fp32 ~ 2^-23 (effectively continuous).

We collect within-position pairwise logprob differences, estimate the ULP step as the
smallest well-supported spacing, normalise by the difference scale, and classify.
``requires=('logprobs',)``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from forensics.core.types import GenRequest
from forensics.probes.base import (
    STATISTICAL, WEAK, Evidence, Estimate, Probe, ProbeContext, ProbeResult, register,
)

# centres of log10(relative-ULP) measured against the reference simulator
_CENTERS = {"fp8_or_lower": -0.30, "bf16": -1.35, "fp16": -2.25, "fp32": -3.5}
_WIDTH = 0.6


def _classify(rel: float) -> str:
    if rel > 0.15:
        return "fp8_or_lower"
    if rel > 0.02:
        return "bf16"
    if rel > 0.0025:
        return "fp16"
    return "fp32"


def estimate_ulp(diffs: np.ndarray) -> dict[str, Any]:
    """Estimate the lattice step and relative ULP from a bag of logit differences."""
    ad = np.abs(np.asarray(diffs, dtype=float))
    ad = ad[ad > 1e-12]
    if ad.size < 4:
        return {"step": None, "rel": None, "scale": None, "n_unique": int(ad.size)}
    scale = float(np.median(ad))
    # On a dtype grid every difference is an integer multiple of the ULP, so the smallest
    # nonzero difference IS the ULP (rounding to 12 dp collapses float64 transport noise).
    # A truly continuous grid (fp32) yields an arbitrarily small min -> classified fp32.
    vals = np.unique(np.round(ad, 12))
    vals = vals[vals > 1e-12]
    step = float(vals.min())
    rel = step / scale if scale > 0 else None
    return {"step": step, "rel": rel, "scale": scale, "n_unique": int(vals.size)}


@register
class LogitPrecisionProbe(Probe):
    id = "P04.precision.logit_dtype"
    title = "Logit numeric dtype from the ULP lattice of logprobs"
    category = "precision"
    kinds = ("generative",)
    informs = {"out.logit_precision": 0.8}
    estimates = ("out.logit_dtype",)
    requires = ("logprobs",)
    cost = 6

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ctx.current_probe = self.id
        res = self.result()
        if not getattr(ctx.caps, "logprobs", None):
            res.status = "skipped"
            res.notes.append("no logprobs; logit dtype not observable")
            return res

        k = max(2, min(int(ctx.caps.max_top_logprobs or 5), 20))
        n_calls = 3 if ctx.fast() else 6
        max_tokens = 12 if ctx.fast() else 24
        diffs: list[float] = []
        n_lp = 0
        for i in range(n_calls):
            prompt = f"precision staircase probe {i}: describe entropy briefly"
            resp = await ctx.gen(GenRequest.chat(
                prompt, logprobs=True, top_logprobs=k, max_tokens=max_tokens,
                temperature=0.7, seed=i))
            if not resp.ok or not resp.tokens:
                continue
            for tok in resp.tokens:
                lps = np.array([tt.logprob for tt in (tok.top or [])], dtype=float)
                n_lp += lps.size
                if lps.size >= 2:
                    d = lps[:, None] - lps[None, :]
                    diffs.extend(d[np.triu_indices(lps.size, k=1)].tolist())

        est = estimate_ulp(np.array(diffs))
        res.tables["ulp"] = [{"step": est["step"], "rel_ulp": est["rel"],
                              "scale": est["scale"], "n_unique": est["n_unique"],
                              "n_logprobs": n_lp, "n_diffs": len(diffs)}]

        if est["rel"] is None:
            # too few / all-tied values -> extremely coarse grid
            res.status = "partial"
            winner = "fp8_or_lower"
            lk = {"fp8_or_lower": 1.0, "fp16": 0.3, "bf16": 0.3, "fp32": 0.1}
            res.add_evidence(Evidence(
                property="out.logit_precision", likelihoods=lk,
                summary="logprobs heavily tied (very coarse grid)", level="inferred",
                strength=0.4, stats=est, alternatives=["provider rounds reported logprobs"]))
            res.add_estimate(Estimate(quantity="out.logit_dtype", value=winner,
                                      level="inferred", method="ULP lattice (degenerate)",
                                      identifiability=WEAK))
            res.n_requests = ctx.n_requests
            return res

        rel = est["rel"]
        winner = _classify(rel)
        logrel = math.log10(rel)
        lk: dict[str, float] = {}
        for dt, c in _CENTERS.items():
            lk[dt] = max(0.03, math.exp(-0.5 * ((logrel - c) / _WIDTH) ** 2))
        lk[winner] = 1.0  # bucket decision dominates

        res.add_evidence(Evidence(
            property="out.logit_precision",
            likelihoods=lk,
            summary=(f"relative ULP ~ {rel:.2e} (step {est['step']:.2e}, scale "
                     f"{est['scale']:.2e}) -> {winner}"),
            level="inferred", strength=0.7,
            stats={"rel_ulp": rel, "step": est["step"], "scale": est["scale"],
                   "n_unique": est["n_unique"], "mantissa_bits_est": round(-logrel / 0.301, 1)},
            alternatives=["provider upcasts logits to fp32 for transport (compute dtype hidden)",
                          "temperature scaling shifts the staircase",
                          "post-softmax rounding of logprobs changes the lattice"],
        ))
        res.add_estimate(Estimate(
            quantity="out.logit_dtype", value=winner,
            level="inferred", method="relative ULP of within-position logprob differences",
            identifiability=STATISTICAL,
            assumptions=["reported logprobs preserve the compute-dtype grid",
                         "logit magnitudes are O(1-30)"],
            alternatives=[d for d in _CENTERS if d != winner],
        ))
        res.shared["logit_dtype"] = winner
        res.n_requests = ctx.n_requests
        return res
