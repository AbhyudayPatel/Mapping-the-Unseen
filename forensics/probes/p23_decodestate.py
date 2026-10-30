"""P23 — decode-state growth: the transformer-vs-SSM split at realistic context lengths.

The generation test (P19) establishes *autoregressive decoding* but cannot separate a
KV-attention decoder from a constant-state SSM decoder: both emit tokens left to right.
Their per-token decode cost differs mechanistically, and the difference is visible at
contexts of only a few hundred tokens — far inside every deployment cap:

* **attention decoder**: each decode step reads a KV cache whose size grows with the
  context L, so per-token decode time increases with L;
* **constant-state decoder** (SSM / linear attention): the recurrent state has fixed
  size, so per-token decode time is (up to scheduler noise) flat in L.

Two measurement channels, in preference order:

1. **streaming ITL** (when the API streams): median inter-chunk latency of a
   16-token decode at each context length L, on fresh-nonce prompts.  Per-token
   times need no pairing at all; this is the same statistic P07 reports as
   ``timing.kv_growth_us_per_token``, here turned into family evidence.
2. **same-body suffix pairs** (non-streaming fallback): one shared fresh body B(L)
   and two one-token suffixes; ``t(B#a, n_b) - t(B#a, n_a)`` cancels prefill
   *exactly* (identical token count, cache or no cache), leaving 10 decode steps.

Frozen decision rule (fixed before any benchmark ground truth was consulted):

* slope CI entirely above ``_POS_US_PER_1K`` us/token per 1000 context tokens →
  growing decode state (attention family);
* |slope| < ``_FLAT_US_PER_1K`` with a CI inside +-``_POS_US_PER_1K`` → constant
  state (SSM/linear);
* otherwise inconclusive, no evidence emitted.

Confounds are stated on the evidence: memory effects can grow with L even for
constant-state models, and a heavily paged/quantised KV layout can flatten
attention decode — hence tempered strengths and retained alternatives.
"""
from __future__ import annotations

import numpy as np

from forensics.core.types import GenRequest
from forensics.probes.base import (
    STATISTICAL, WEAK, BudgetExhausted, Estimate, Evidence, Probe, ProbeContext,
    ProbeResult, register,
)
from forensics.stats.regression import theilsen

# Frozen thresholds (us per output token per 1000 context tokens).
_POS_US_PER_1K = 6.0
_FLAT_US_PER_1K = 2.0
_FLOOR = 0.05

_LENGTHS_FULL = (64, 128, 256, 512, 1024, 2048)
_LENGTHS_FAST = (64, 256, 1024)
_N_A, _N_B = 2, 12
_STREAM_TOKENS = 16


def _server_ms(resp) -> float | None:
    ms = resp.timing.server_ms if resp.timing else None
    if ms is not None:
        return float(ms)
    te = resp.timing.t_end if resp.timing else None
    return float(te) * 1000.0 if te is not None else None


def _n_out(resp) -> int:
    u = resp.usage
    if u and u.completion_tokens is not None:
        return max(0, int(u.completion_tokens) - int(u.reasoning_tokens or 0))
    return len(resp.tokens or [])


def _fresh(ctx: ProbeContext, L: int, nonce: int) -> str:
    idx = ctx.rng.integers(0, 29989, size=L)
    return f"n{nonce:05d} " + " ".join(f"w{int(i)}" for i in idx)


def _median_itl_s(resp) -> float | None:
    """Median inter-token time (seconds): server-reported per-chunk times when present
    (Azure latency_checkpoint / wrapper t_server_ms), else client chunk arrival times."""
    raw = getattr(resp, "raw", None) or []
    srv = [c.get("t_server_ms") for c in raw if isinstance(c, dict)
           and c.get("t_server_ms") is not None]
    if len(srv) >= 3:
        ts = np.array(srv, float) / 1000.0
        itl = np.diff(ts)
        itl = itl[itl > 0]
        return float(np.median(itl)) if itl.size else None
    ch = [c for c in (resp.timing.chunks if resp.timing else []) if c.t is not None]
    if len(ch) < 3:
        return None
    ts = np.array([c.t for c in ch], float)          # seconds, adapter-owned clock
    itl = np.diff(ts)
    itl = itl[itl > 0]
    return float(np.median(itl)) if itl.size else None


@register
class DecodeStateGrowthProbe(Probe):
    id = "P23.decode.state_growth"
    title = "Decode-state growth (KV cache vs constant state) from marginal decode time"
    category = "architecture"
    kinds = ("generative",)
    informs = {"arch.family": 0.65, "attn.long_range": 0.45, "attn.kv_sharing": 0.1}
    cost = 40

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ctx.current_probe = self.id
        res = self.result()
        lengths = list(_LENGTHS_FAST if ctx.fast() else _LENGTHS_FULL)
        max_ctx = (ctx.shared.get("context") or {}).get("max_input_tokens")
        if max_ctx:
            lengths = [L for L in lengths if L <= int(max_ctx) * 0.9] or lengths[:2]
        reps = ctx.scale(6, minimum=3)
        use_stream = bool(ctx.caps.streaming)

        rows: list[dict] = []
        try:
            if use_stream:
                trials = [(L, rep) for rep in range(reps) for L in lengths]
                for oi in ctx.rng.permutation(len(trials)):
                    L, rep = trials[int(oi)]
                    r = await ctx.gen(GenRequest.chat(_fresh(ctx, L, rep), temperature=0.0,
                                                      max_tokens=_STREAM_TOKENS, stream=True))
                    if not r.ok:
                        continue
                    itl = _median_itl_s(r)
                    pt = r.usage.prompt_tokens if r.usage else None
                    if itl is not None and pt:
                        rows.append({"L": int(pt), "rep": rep,
                                     "ms_per_token": itl * 1000.0})
            else:
                trials = [(L, rep) for rep in range(reps) for L in lengths]
                for oi in ctx.rng.permutation(len(trials)):
                    L, rep = trials[int(oi)]
                    body = _fresh(ctx, L, rep)          # SAME body, different suffix:
                    ra = await ctx.gen(GenRequest.chat(body + " #a", max_tokens=_N_A,
                                                       temperature=0.0))
                    rb = await ctx.gen(GenRequest.chat(body + " #b", max_tokens=_N_B,
                                                       temperature=0.0))
                    if not (ra.ok and rb.ok):
                        continue
                    ta, tb = _server_ms(ra), _server_ms(rb)
                    na, nb = _n_out(ra), _n_out(rb)
                    if ta is None or tb is None or nb <= na:
                        continue
                    pt = ra.usage.prompt_tokens if ra.usage else L
                    rows.append({"L": int(pt or L), "rep": rep,
                                 "ms_per_token": (tb - ta) / (nb - na)})
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("budget exhausted during decode-state sweep")

        res.n_requests = ctx.n_requests
        res.tables["decode_growth"] = rows
        channel = "streaming inter-token latency" if use_stream else \
            "same-body suffix pairs (prefill-cancelling)"
        if len(rows) < 6:
            res.status = res.status if res.status != "ok" else "partial"
            res.notes.append(f"only {len(rows)} usable (L, rep) cells — no slope fitted")
            return res

        xs = np.array([r["L"] for r in rows], float)
        ys = np.array([r["ms_per_token"] for r in rows], float)
        # contention-robust: per-length minimum across reps (co-tenant CPU steals inflate
        # single trials upward; the min tracks the true decode cost), then Theil-Sen
        xs_f, ys_f = xs, ys
        by_len: dict[float, list[float]] = {}
        for r in rows:
            by_len.setdefault(float(r["L"]), []).append(float(r["ms_per_token"]))
        if len(by_len) >= 3 and all(len(v) >= 2 for v in by_len.values()):
            xs_f = np.array(sorted(by_len), float)
            ys_f = np.array([min(by_len[L]) for L in xs_f], float)
        slope_ms, _intercept, (lo_ms, hi_ms) = theilsen(xs_f, ys_f, level=0.9)
        # ms per token per context token -> us per token per 1000 context tokens
        # (ms->us is x1000; per-ctx-token -> per-1000-ctx-tokens is x1000)
        slope_us = slope_ms * 1_000_000.0
        lo_us, hi_us = lo_ms * 1_000_000.0, hi_ms * 1_000_000.0
        med = float(np.median(ys))
        res.series["decode_growth"] = {
            "x": xs.tolist(), "y": ys.tolist(),
            "xlabel": "context tokens", "ylabel": "marginal ms / output token",
            "kind": "scatter"}
        res.add_estimate(Estimate(
            quantity="timing.kv_growth_us_per_token",
            value=round(slope_us / 1000.0, 4),
            ci=(round(lo_us / 1000.0, 4), round(hi_us / 1000.0, 4)),
            level="estimated", method=f"Theil-Sen on {channel}",
            identifiability=STATISTICAL if len(rows) >= 12 else WEAK,
            unit="us/token/ctx-token"))

        growing = lo_us > _POS_US_PER_1K
        flat = (abs(slope_us) < _FLAT_US_PER_1K and -_POS_US_PER_1K < lo_us
                and hi_us < _POS_US_PER_1K)
        # Likelihood contrasts calibrated on the same-hardware control (dense full
        # attention +18.6 us/token/1k, sliding-window +1.8, SSM +0.5; see paper):
        # a decisive flat/growing read is ~8-12x more likely under the matching family.
        if growing:
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"decoder_only": 1.0, "encoder_decoder": 0.6,
                             "hybrid_attn_ssm": 0.5, "ssm_or_linear": 0.08,
                             "diffusion": 0.25, "encoder_only": _FLOOR, "non_neural": _FLOOR},
                summary=f"marginal decode time grows {slope_us:.1f} us/token per 1k context "
                        f"(CI [{lo_us:.1f}, {hi_us:.1f}], {channel}) → a growing KV decode state",
                level="inferred", strength=0.7,
                stats={"slope_us_per_1k": slope_us, "ci": [lo_us, hi_us], "n": len(rows),
                       "channel": channel},
                alternatives=["memory-bandwidth effects growing with context even for "
                              "constant-state models", "un-cached recompute decoding"],
                experiment_group="decode_growth"))
            res.add_evidence(Evidence(
                property="attn.long_range",
                likelihoods={"full_quadratic": 0.9, "sliding_or_local": 0.8,
                             "hybrid": 0.7, "linear_recurrent": 0.15},
                summary="decode cost grows with context → state that scales with L",
                level="inferred", strength=0.45,
                stats={"slope_us_per_1k": slope_us},
                alternatives=["sliding-window KV also grows until the window fills"],
                experiment_group="decode_growth"))
        elif flat:
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"ssm_or_linear": 1.0, "hybrid_attn_ssm": 0.6,
                             "decoder_only": 0.12, "encoder_decoder": 0.2,
                             "diffusion": 0.25, "encoder_only": _FLOOR, "non_neural": _FLOOR},
                summary=f"marginal decode time flat in context ({slope_us:.1f} us/token/1k, "
                        f"CI [{lo_us:.1f}, {hi_us:.1f}], {channel}) → constant-size decode state",
                level="inferred", strength=0.7,
                stats={"slope_us_per_1k": slope_us, "ci": [lo_us, hi_us], "n": len(rows),
                       "channel": channel},
                alternatives=["an aggressively paged/quantised KV cache can approach flat "
                              "decode over short spans",
                              "a sliding-window decoder reads flat beyond its window"],
                experiment_group="decode_growth"))
            res.add_evidence(Evidence(
                property="attn.long_range",
                likelihoods={"linear_recurrent": 1.0, "hybrid": 0.6,
                             "sliding_or_local": 0.4, "full_quadratic": 0.25},
                summary="decode cost independent of context → constant state",
                level="inferred", strength=0.45,
                stats={"slope_us_per_1k": slope_us},
                alternatives=["short test span can hide a large-but-finite window"],
                experiment_group="decode_growth"))
        else:
            res.notes.append(f"decode-growth slope inconclusive: {slope_us:.1f} us/token/1k "
                             f"(CI [{lo_us:.1f}, {hi_us:.1f}]); median {med:.2f} ms/token; "
                             f"channel={channel}")
        return res
