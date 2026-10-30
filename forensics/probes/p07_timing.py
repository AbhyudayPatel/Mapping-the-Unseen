"""P07 — Timing & scaling forensics.

Turn the endpoint's server-reported latencies into architecture and hardware evidence.
Everything uses server-side time (``timing.server_ms`` in MILLISECONDS → seconds via
``/1000``) or streaming arrival times; the probe never times requests locally, runs at
``concurrency=1``, randomises condition order, and takes medians over repeats.

Observables and what they buy us:

* **RTT baseline** — min-over-k of ``t_end - server_ms/1000`` on tiny requests isolates the
  network round trip (``timing.rtt_ms``).
* **prefill(prompt_len)** — fit server time vs prompt length as degree-1 vs degree-2
  (``poly_model_select``).  A significant positive quadratic term ⇒ full O(n^2) self
  attention; a linear law ⇒ sliding/linear/SSM.  ⇒ ``attn.long_range``.  The linear slope
  gives ``timing.prefill_tokens_per_s``.
* **inter-token latency(context)** — robust (Theil-Sen) slope of median ITL vs context is
  the KV-cache growth per context token (``timing.kv_growth_us_per_token``, weakly informs
  ``attn.kv_sharing``); the intercept is the weight-only decode time
  (``timing.decode_tokens_per_s``).
* **active parameters** — invert the memory-bound decode time
  ``t_decode ≈ active_bytes / (bw · TP)`` with Monte-Carlo priors over bandwidth, tensor
  parallelism and dtype.  Deliberately a WIDE interval under stated assumptions
  (``dims.active_params``).
"""
from __future__ import annotations

import warnings

import numpy as np

from forensics.probes.base import (
    ASSUMPTION, STATISTICAL, WEAK, BudgetExhausted, Estimate, Evidence, Probe,
    ProbeContext, ProbeResult, register,
)
from forensics.core.types import GenRequest
from forensics.stats import poly_model_select, theilsen


def _server_s(resp) -> float | None:
    ms = resp.timing.server_ms
    return None if ms is None else float(ms) / 1000.0


def _floor(v: float) -> float:
    return max(0.02, float(v))


def _server_tbt_s(resp) -> float | None:
    """Server-reported time-between-tokens (seconds), e.g. Azure usage.latency_checkpoint
    .engine_tbt_ms. Far cleaner than client chunk spacing, which proxies/CDNs often buffer."""
    raw = getattr(resp.usage, "raw", None) or {}
    lc = raw.get("latency_checkpoint") if isinstance(raw, dict) else None
    if isinstance(lc, dict):
        for k in ("engine_tbt_ms", "service_tbt_ms"):
            v = lc.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v) / 1000.0
    return None


# Below this, client-side inter-chunk gaps are not decode steps but a buffered stream
# flushing many tokens at once (single-stream decode of any served LLM is > ~1 ms/token).
_MIN_PLAUSIBLE_ITL_S = 0.001


@register
class TimingScalingProbe(Probe):
    id = "P07.timing.scaling"
    title = "Prefill/decode timing, KV growth and active-parameter inversion"
    category = "timing"
    kinds = ("generative",)
    informs = {"attn.long_range": 0.55, "attn.kv_sharing": 0.15}
    estimates = ("timing.rtt_ms", "timing.prefill_tokens_per_s", "timing.decode_tokens_per_s",
                 "timing.itl_ms", "timing.kv_growth_us_per_token", "dims.active_params")
    cost = 30

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        res = self.result()
        if not (ctx.caps.server_timing_header or ctx.caps.streaming):
            res.status = "skipped"
            res.notes.append("no server timing header and no streaming: cannot time server-side")
            return res

        reps = ctx.scale(5, minimum=2)
        # Prefill curvature is only detectable once attention FLOPs are a sizeable share of the
        # total (~ctx / (6·d_model)): at <=4k tokens every transformer looks linear. Full mode
        # therefore extends the sweep toward 32k (bounded by the known context window).
        if ctx.fast():
            p_lens = [128, 256, 512, 1024, 2048]
        else:
            max_ctx = (ctx.shared.get("context") or {}).get("max_input_tokens") or 32_768
            cap_units = int(min(16_000, 0.35 * max_ctx))    # ~2 tokens per unit below
            p_lens = [u for u in (128, 512, 1024, 2048, 4096, 8192, 16_000) if u <= cap_units]
        c_lens = [64, 256, 512, 1024, 2048] if ctx.fast() else [64, 256, 512, 1024, 2048, 3072]

        try:
            rtt_s = await self._rtt(ctx, reps, res)
            prefill = await self._prefill_sweep(ctx, p_lens, reps)
            itl = await self._itl_sweep(ctx, c_lens, max(1, ctx.scale(3, minimum=1)))
        except BudgetExhausted as e:
            res.notes.append(f"budget exhausted: {e}")
            res.status = "partial"
            res.n_requests = ctx.n_requests
            return res

        # ---- RTT -------------------------------------------------------------
        if rtt_s is not None:
            rtt_all = res.shared.get("_rtt_all") or [rtt_s]
            res.add_estimate(Estimate(
                quantity="timing.rtt_ms", value=round(rtt_s * 1000.0, 3),
                ci=(min(rtt_all) * 1000.0, float(np.median(rtt_all)) * 1000.0), level="inferred",
                method="min-of-k of (t_end - server_ms/1000) on tiny requests",
                identifiability=STATISTICAL, unit="ms"))
            res.shared["rtt_s"] = rtt_s

        # ---- prefill scaling / attention family -----------------------------
        px = np.array(prefill["x"], float)
        py = np.array(prefill["y_s"], float)
        res.tables["prefill"] = prefill["rows"]
        res.series["prefill"] = {"x": prefill["x"], "y": [v * 1000 for v in prefill["y_s"]],
                                 "xlabel": "prompt tokens", "ylabel": "server ms", "kind": "line"}
        if px.size >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sel = poly_model_select(px / 1000.0, py, degrees=(1, 2), criterion="bic",
                                        n_boot=300, rng=int(ctx.rng.integers(1 << 30)))
            quad = bool(sel["quadratic_significant"]) and (sel["curvature"] or 0.0) > 0
            max_len = float(px.max())
            if quad:
                like = {"full_quadratic": 1.0, "sliding_or_local": _floor(0.2),
                        "linear_recurrent": _floor(0.12), "hybrid": _floor(0.5)}
                summ = f"prefill time is quadratic in prompt length (up to {max_len:.0f} tokens)"
                strength = 0.7
            elif max_len >= 16_000:
                like = {"full_quadratic": _floor(0.3), "sliding_or_local": 0.85,
                        "linear_recurrent": 1.0, "hybrid": 0.7}
                summ = f"prefill stays ~linear up to {max_len:.0f} tokens"
                strength = 0.5
            else:
                # linear at short lengths is what EVERY architecture shows: near-neutral
                like = {"full_quadratic": 0.8, "sliding_or_local": 0.9,
                        "linear_recurrent": 1.0, "hybrid": 0.9}
                summ = (f"prefill ~linear up to only {max_len:.0f} tokens — expected for any "
                        f"architecture at this length (uninformative about attention type)")
                strength = 0.15
            res.add_evidence(Evidence(
                property="attn.long_range", likelihoods=like, summary=summ, level="inferred",
                strength=strength,
                stats={"test": "poly_model_select(bic)", "degree": sel["degree"],
                       "curvature": sel["curvature"], "curvature_ci": sel["curvature_ci"],
                       "quadratic_significant": sel["quadratic_significant"],
                       "max_prompt_tokens": max_len},
                alternatives=["chunked/paged prefill can linearise a quadratic core",
                              "queueing/batching jitter can inflate curvature"]))
            # prefill throughput from the linear slope (small-length regime)
            slope, _, _ = theilsen(px, py)
            if slope > 0:
                res.add_estimate(Estimate(
                    quantity="timing.prefill_tokens_per_s", value=round(1.0 / slope, 1),
                    level="estimated", method="1 / robust slope of server-time vs prompt length",
                    identifiability=WEAK, unit="tok/s",
                    assumptions=["linear-regime slope approximates marginal prefill cost"]))

        # ---- ITL / KV growth / decode throughput ----------------------------
        cx = np.array(itl["x"], float)
        cy = np.array(itl["itl_s"], float)
        res.tables["itl"] = itl["rows"]
        src = itl.get("source", {})
        res.tables["itl_source"] = [src]
        if src.get("buffered") and not (src.get("server_tbt") or src.get("client_stream")):
            res.notes.append("stream is buffered (chunks arrive in bursts) and the API reports no "
                             "server time-between-tokens: decode timing, KV growth and "
                             "active-parameter inversion are NOT identifiable from this interface")
        itl_method = ("server-reported time-between-tokens (latency_checkpoint)"
                      if src.get("server_tbt") else "median client inter-chunk latency (streaming)")
        res.series["itl"] = {"x": itl["x"], "y": [v * 1000 for v in itl["itl_s"]],
                             "xlabel": "context tokens", "ylabel": "median ITL ms", "kind": "line"}
        decode0 = None
        if cx.size >= 2:
            slope, intercept, slope_ci = theilsen(cx, cy)
            decode0 = max(intercept, min(cy)) if cy.size else intercept
            kv_us = slope * 1e6
            kv_ci = (slope_ci[0] * 1e6, slope_ci[1] * 1e6) if slope_ci else None
            kv_detected = bool(kv_ci and kv_ci[0] > 0)
            res.add_estimate(Estimate(
                quantity="timing.kv_growth_us_per_token",
                value=round(kv_us, 4) if kv_detected else 0.0, ci=kv_ci,
                level="estimated",
                method=("Theil-Sen slope of median ITL vs context" if kv_detected else
                        "no detectable ITL growth over the tested context range (slope CI includes 0)"),
                identifiability=STATISTICAL if kv_detected else WEAK, unit="us/token",
                alternatives=["paged-cache steps", "load-correlated ITL drift",
                              "range too short: KV reads are a small share of decode time at a few k tokens"]))
            res.add_estimate(Estimate(
                quantity="timing.itl_ms", value=round(float(np.median(cy)) * 1000, 4),
                ci=(float(np.min(cy)) * 1000, float(np.max(cy)) * 1000),
                level="inferred", method=itl_method,
                identifiability=STATISTICAL, unit="ms"))
            if decode0 and decode0 > 0:
                res.add_estimate(Estimate(
                    quantity="timing.decode_tokens_per_s", value=round(1.0 / decode0, 1),
                    level="estimated", method="1 / weight-only decode time (ITL intercept)",
                    identifiability=WEAK, unit="tok/s"))
            if kv_detected:
                # weak KV-sharing hint: a very small KV slope => aggressive sharing
                share = "gqa_or_mqa" if kv_us < 0.1 else "mha"
                res.add_evidence(Evidence(
                    property="attn.kv_sharing",
                    likelihoods={"mha": _floor(0.5 if share == "mha" else 0.3),
                                 "gqa_or_mqa": _floor(0.5 if share == "gqa_or_mqa" else 0.35),
                                 "mla": _floor(0.35)},
                    summary=f"KV growth {kv_us:.3f} us/context-token", level="speculative",
                    strength=0.2,
                    stats={"kv_growth_us_per_token": kv_us, "ci": kv_ci},
                    alternatives=["absolute KV bytes depend on layers/heads/dtype (non-identifiable alone)"]))

        # ---- active params via Monte-Carlo inversion of decode time ----------
        if decode0 and decode0 > 0:
            samples = self._active_params_mc(ctx, decode0)
            lo, hi = np.quantile(samples, [0.05, 0.95])
            res.add_estimate(Estimate(
                quantity="dims.active_params", value=round(float(np.median(samples)), 2),
                ci=(float(lo), float(hi)), ci_level=0.9, level="estimated",
                method="invert memory-bound decode: active = t_decode·bw·TP/(1e9·bytes), MC priors",
                identifiability=ASSUMPTION, unit="B", samples=[float(s) for s in samples[:512]],
                assumptions=["decode is memory-bandwidth bound",
                             "priors: bw~lognormal(3e12,0.45), TP∈{1,2,4,8}, bytes∈{2,1}"],
                alternatives=["MoE: active≪total params", "compute-bound or batched decode breaks the model"]))

        res.n_requests = ctx.n_requests
        return res

    # ------------------------------------------------------------------ RTT
    async def _rtt(self, ctx: ProbeContext, reps: int, res: ProbeResult | None = None) -> float | None:
        vals = []
        for i in range(reps):
            resp = await ctx.gen(GenRequest.chat(f"hi {i}", max_tokens=1, tag=self.id))
            s = _server_s(resp)
            if resp.ok and resp.timing.t_end is not None and s is not None:
                vals.append(resp.timing.t_end - s)
        if res is not None and vals:
            res.shared["_rtt_all"] = vals
        return min(vals) if vals else None

    # -------------------------------------------------------------- prefill
    async def _prefill_sweep(self, ctx: ProbeContext, lengths, reps) -> dict:
        conds = [(L, r) for L in lengths for r in range(reps)]
        conds = [conds[i] for i in ctx.rng.permutation(len(conds))]
        acc: dict[int, list] = {L: [] for L in lengths}
        toks: dict[int, list] = {L: [] for L in lengths}
        for L, r in conds:
            # distinct content per (length, rep) defeats prompt caching
            prompt = f"n{L}r{r} " + ("lorem ipsum " * L)
            resp = await ctx.gen(GenRequest.chat(prompt, max_tokens=1, tag=self.id))
            s = _server_s(resp)
            if resp.ok and s is not None:
                acc[L].append(s)
                toks[L].append(resp.usage.prompt_tokens or L)
        x, ys, rows = [], [], []
        for L in lengths:
            if acc[L]:
                med = float(np.median(acc[L]))
                pt = float(np.median(toks[L]))
                x.append(pt)
                ys.append(med)
                rows.append({"target_len": L, "prompt_tokens": pt, "server_ms": med * 1000})
        return {"x": x, "y_s": ys, "rows": rows}

    # ----------------------------------------------------------------- ITL
    async def _itl_sweep(self, ctx: ProbeContext, lengths, reps) -> dict:
        conds = [(L, r) for L in lengths for r in range(reps)]
        conds = [conds[i] for i in ctx.rng.permutation(len(conds))]
        acc: dict[int, list] = {L: [] for L in lengths}
        toks: dict[int, list] = {L: [] for L in lengths}
        source = {"server_tbt": 0, "client_stream": 0, "buffered": 0}
        for L, r in conds:
            prompt = f"c{L}r{r} " + ("lorem ipsum " * L)
            resp = await ctx.gen(GenRequest.chat(prompt, max_tokens=16, stream=True, tag=self.id))
            if not resp.ok:
                continue
            tbt = _server_tbt_s(resp)
            if tbt is not None:
                acc[L].append(tbt)
                toks[L].append(resp.usage.prompt_tokens or L)
                source["server_tbt"] += 1
                continue
            itls = resp.timing.inter_token_latencies()
            if itls:
                med = float(np.median(itls))
                if med < _MIN_PLAUSIBLE_ITL_S:
                    source["buffered"] += 1     # chunks flushed in bursts: not decode steps
                    continue
                acc[L].append(med)
                toks[L].append(resp.usage.prompt_tokens or L)
                source["client_stream"] += 1
        x, ys, rows = [], [], []
        for L in lengths:
            if acc[L]:
                med = float(np.median(acc[L]))
                pt = float(np.median(toks[L]))
                x.append(pt)
                ys.append(med)
                rows.append({"context_tokens": pt, "itl_ms": med * 1000})
        return {"x": x, "itl_s": ys, "rows": rows, "source": source}

    # ------------------------------------------------------- active params MC
    @staticmethod
    def _active_params_mc(ctx: ProbeContext, decode0_s: float, n: int = 20000) -> np.ndarray:
        rng = np.random.default_rng(int(ctx.rng.integers(1 << 30)))
        bw = rng.lognormal(mean=np.log(3.0e12), sigma=0.45, size=n)       # HBM bytes/s
        tp = rng.choice([1, 2, 4, 8], size=n, p=[0.55, 0.25, 0.12, 0.08])
        byts = rng.choice([2, 1], size=n, p=[0.75, 0.25])                  # bf16 vs fp8
        active_b = decode0_s * bw * tp / (1e9 * byts)
        return active_b
