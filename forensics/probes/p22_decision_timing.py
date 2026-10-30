"""P22 — decision-model timing (server-time scaling for Jev-type endpoints).

Decision models expose no streaming and no generation, but many expose a server-side
processing time (Jev: ``x-envoy-upstream-service-time`` -> ``timing.server_ms``, in
MILLISECONDS).  That single number, measured under controlled conditions, gives:

* **server_ms vs state length** — fresh random text on every request (defeats prefix
  caching), lengths spanning ~0.3k -> 16k tokens (full) / 0.3k -> 12k (fast), randomised
  order, concurrency 1, medians over ``ctx.scale`` repeats.  Degree-1 vs degree-2 fit
  (:func:`forensics.stats.poly_model_select`): a significant positive curvature is the
  quadratic self-attention term (``attn.long_range=full_quadratic``); a linear law at these
  lengths is only weak/ambiguous evidence (attention is a small share of FLOPs below ~16k
  even for full attention), so its strength is kept modest.  The slope gives
  ``timing.prefill_tokens_per_s``.
* **server_ms vs #questions at a fixed state** (1, 4, 16, 64) — if an extra question costs
  about as much as a handful of its own tokens and nowhere near re-reading the state, the
  questions are cheap branches over one shared, prefilled state prefix
  (``decision.question_coupling=independent`` weakly, ``decision.scoring_mechanism=
  single_pass_heads``: no per-question decode loop).
* **prefix caching** — the same long state repeated: a clear latency drop on the repeats
  (vs fresh text of the same length) => ``serving.prompt_caching=yes``.
* **active parameters** — prefill is compute-bound: ``2 * N_active`` FLOP per token, so
  ``N_active = peak * MFU * n_gpu / (2 * tokens_per_s)``.  Monte-Carlo over hardware priors
  (peak FLOP/s, MFU, GPUs per replica) gives a deliberately WIDE interval (speculative).

Everything goes through ``ctx.decide``; failed responses are skipped and reported.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from forensics.core.types import DecisionQuestion, DecisionRequest
from forensics.probes.base import (
    STATISTICAL, WEAK, BudgetExhausted, Estimate, Evidence, Probe, ProbeContext,
    ProbeResult, register,
)
from forensics.stats import poly_model_select, theilsen

# Common short English words (~1 token each with a leading space in byte-level BPEs).
_WORDS = (
    "time person year way day thing man world life hand part child eye woman place work "
    "week case point company number group problem fact water room mother area money story "
    "month lot right study book job word business issue side kind head house service friend "
    "father power hour game line end member law car city community name team minute idea "
    "kid body information back parent face others level office door health art war history "
    "party result change morning reason research girl guy moment air teacher force education "
    "foot boy age policy process music market sense nation plan college interest death "
    "experience effect use class control care field development role effort rate heart drug "
    "show leader light voice wife police mind price report decision son view relationship "
    "town road arm difference value building action model season society tax director position "
    "player record paper space ground form event official matter center couple site project "
    "activity star table need court oil situation cost industry figure street image phone data "
    "picture practice piece land product doctor wall patient worker news test movie north love "
    "support technology step baby computer type attention film tree source organization hair "
    "window evidence population site bank"
).split()

_NOUL = "Is this text written in English?"


def _floor(v: float, f: float = 0.03) -> float:
    return max(f, float(v))


def _server_ms(resp) -> float | None:
    ms = resp.timing.server_ms
    return None if ms is None else float(ms)


@register
class DecisionTimingProbe(Probe):
    id = "P22.decision.timing"
    title = "Decision-model server-time scaling (state length, #questions, prefix cache)"
    category = "timing"
    kinds = ("decision",)
    informs = {"attn.long_range": 0.35, "decision.question_coupling": 0.2,
               "decision.scoring_mechanism": 0.3, "serving.prompt_caching": 0.5}
    estimates = ("timing.prefill_tokens_per_s", "dims.active_params")
    cost = 56

    # ------------------------------------------------------------------ helpers
    def _text(self, ctx: ProbeContext, n_words: int) -> str:
        idx = ctx.rng.integers(0, len(_WORDS), size=max(1, n_words))
        # random leading nonce so even the first cache block differs between requests
        nonce = "".join(chr(97 + int(c)) for c in ctx.rng.integers(0, 26, size=8))
        return nonce + " " + " ".join(_WORDS[int(i)] for i in idx) + "."

    @staticmethod
    def _questions(nq: int) -> dict[str, DecisionQuestion]:
        return {f"q{i}": DecisionQuestion(type="noul",
                                          instructions=f"Is statement {i} about urgency true?")
                for i in range(nq)}

    async def _one(self, ctx: ProbeContext, state: str, nq: int
                   ) -> tuple[float | None, int | None, str | None]:
        r = await ctx.decide(DecisionRequest(state=state, questions=self._questions(nq)))
        if not r.ok:
            return None, None, str(r.error)[:120]
        return _server_ms(r), r.usage.prompt_tokens, None

    # ------------------------------------------------------------------ run
    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        res = self.result()
        fast = ctx.fast()
        lengths = list(ctx.config.get("p22_lengths") or
                       ([300, 2000, 6000, 12000] if fast else [300, 1000, 2000, 4000, 8000, 16000]))
        nqs = list(ctx.config.get("p22_questions") or [1, 4, 16, 64])
        reps = int(ctx.config.get("p22_repeats") or ctx.scale(5, minimum=2))
        q_state_len = 3000 if fast else 4000
        cache_len = max(lengths)
        cache_reps = 2 if fast else 4

        # conditions: ("len", L, rep) and ("nq", nq, rep), jointly randomised
        conds: list[tuple[str, int, int]] = ([("len", L, r) for L in lengths for r in range(reps)]
                                             + [("nq", q, r) for q in nqs for r in range(reps)])
        conds = [conds[i] for i in ctx.rng.permutation(len(conds))]
        len_obs: list[tuple[int, float, int]] = []    # (target, ms, input_tokens)
        nq_obs: list[tuple[int, float, int]] = []
        errors: list[str] = []
        cache_rows: list[dict[str, Any]] = []
        try:
            for kind, v, _r in conds:
                if kind == "len":
                    ms, it, err = await self._one(ctx, self._text(ctx, v), 1)
                    if ms is not None and it is not None:
                        len_obs.append((v, ms, int(it)))
                else:
                    ms, it, err = await self._one(ctx, self._text(ctx, q_state_len), v)
                    if ms is not None and it is not None:
                        nq_obs.append((v, ms, int(it)))
                if err:
                    errors.append(f"{kind}={v}: {err}")
            # prefix-cache test: one identical long state, cold then repeated
            same = self._text(ctx, cache_len)
            for i in range(1 + cache_reps):
                ms, it, err = await self._one(ctx, same, 1)
                if ms is not None:
                    cache_rows.append({"call": i, "role": "cold" if i == 0 else "repeat",
                                       "server_ms": round(ms, 2), "input_tokens": it})
                elif err:
                    errors.append(f"cache#{i}: {err}")
        except BudgetExhausted as e:
            res.status = "partial"
            res.notes.append(f"budget exhausted: {e}")
        res.n_requests = ctx.n_requests
        if errors:
            res.notes.append(f"{len(errors)} failed timing requests (e.g. {errors[0]})")

        if len(len_obs) < 3 or len({o[0] for o in len_obs}) < 3:
            res.status = "partial" if len_obs else "skipped"
            res.notes.append("decision timing: too few successful state-length measurements "
                             "(no server time exposed?)")
            return res

        # ---- (a) server_ms vs state length --------------------------------------------
        x_tok = np.array([o[2] for o in len_obs], float)
        y_s = np.array([o[1] for o in len_obs], float) / 1000.0
        rows = []
        med_x: list[float] = []
        med_y: list[float] = []
        for L in lengths:
            pts = [o for o in len_obs if o[0] == L]
            if not pts:
                continue
            mx = float(np.median([p[2] for p in pts]))
            my = float(np.median([p[1] for p in pts]))
            med_x.append(mx)
            med_y.append(my)
            rows.append({"target_words": L, "input_tokens": round(mx), "n": len(pts),
                         "median_server_ms": round(my, 2),
                         "min_ms": round(min(p[1] for p in pts), 2),
                         "max_ms": round(max(p[1] for p in pts), 2)})
        res.tables["server_ms_vs_state"] = rows
        res.series["server_ms_vs_state"] = {"x": med_x, "y": med_y, "xlabel": "input tokens",
                                            "ylabel": "median server ms", "kind": "line"}

        slope, intercept, slope_ci = theilsen(x_tok, y_s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = poly_model_select(x_tok / 1000.0, y_s, degrees=(1, 2), criterion="bic",
                                    n_boot=400, rng=int(ctx.rng.integers(1 << 30)))
        c2 = float(sel.get("curvature") or 0.0)            # s per (ktok)^2
        xmax_k = float(x_tok.max()) / 1000.0
        lin_part = max(1e-9, slope * float(x_tok.max()))
        quad_share = (c2 * xmax_k ** 2) / (lin_part + max(0.0, c2) * xmax_k ** 2) if c2 > 0 else 0.0
        quad = bool(sel.get("quadratic_significant")) and c2 > 0 and sel.get("degree") == 2
        resid = y_s - (slope * x_tok + intercept)
        noise_ms = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) * 1000.0
        ms_per_k = slope * 1e6
        fit_stats = {"test": "poly_model_select(bic) on per-request server time",
                     "degree": sel.get("degree"), "curvature_s_per_ktok2": c2,
                     "curvature_ci": sel.get("curvature_ci"),
                     "quadratic_significant": sel.get("quadratic_significant"),
                     "quadratic_share_at_max_len": round(quad_share, 3),
                     "slope_ms_per_ktok": round(ms_per_k, 3),
                     "slope_ci_ms_per_ktok": (round(slope_ci[0] * 1e6, 3),
                                              round(slope_ci[1] * 1e6, 3)),
                     "intercept_ms": round(intercept * 1000.0, 2), "noise_ms": round(noise_ms, 2),
                     "n": int(x_tok.size), "max_tokens": int(x_tok.max())}
        if quad:
            like = {"full_quadratic": 1.0, "sliding_or_local": _floor(0.35),
                    "linear_recurrent": _floor(0.25), "hybrid": _floor(0.6)}
            strength = 0.45
            summ = (f"server time is super-linear in state length (curvature significant, "
                    f"~{quad_share:.0%} of time at {int(x_tok.max())} tokens) -> quadratic attention")
        else:
            like = {"full_quadratic": 0.6, "sliding_or_local": 0.85, "linear_recurrent": 1.0,
                    "hybrid": 0.85}
            strength = 0.2
            summ = (f"server time ~linear in state length up to {int(x_tok.max())} tokens "
                    f"({ms_per_k:.2f} ms/ktok); weak/ambiguous: attention is a small share of "
                    f"prefill at these lengths even for full attention")
        res.add_evidence(Evidence(
            property="attn.long_range", likelihoods=like, summary=summ, level="inferred",
            strength=strength, stats=fit_stats,
            alternatives=["queueing/batching noise can hide or fake curvature",
                          "chunked prefill / FlashAttention make quadratic terms small below ~16k",
                          "server time may include gateway work that scales with bytes"]))

        # ---- throughput + active params ------------------------------------------------
        tps = None
        if slope > 0:
            tps = 1.0 / slope
            lo_s, hi_s = (max(slope_ci[0], 1e-12), max(slope_ci[1], 1e-12))
            res.add_estimate(Estimate(
                quantity="timing.prefill_tokens_per_s", value=round(tps, 1),
                ci=(round(1.0 / hi_s, 1), round(1.0 / lo_s, 1)), level="estimated",
                method="1 / Theil-Sen slope of server time vs billed state tokens (fresh text)",
                identifiability=STATISTICAL, unit="tok/s",
                assumptions=["server_ms measures prefill of the state (no generation)",
                             "billed input_tokens ~ tokens actually processed"],
                alternatives=["per-request throughput, not the replica's peak (batching)"]))
            samples = self._active_params_mc(ctx, tps)
            q05, q50, q95 = np.quantile(samples, [0.05, 0.5, 0.95])
            res.add_estimate(Estimate(
                quantity="dims.active_params", value=round(float(q50), 2),
                ci=(round(float(q05), 2), round(float(q95), 2)), ci_level=0.9,
                level="speculative",
                method=("compute-bound prefill: N_active = peak*MFU*n_gpu / (2*tokens_per_s); "
                        "Monte-Carlo over hardware priors"),
                identifiability=WEAK, unit="B", samples=[float(v) for v in samples[:512]],
                assumptions=["prefill is compute-bound at 2*N_active FLOP/token",
                             "priors: peak~lognormal(0.99 PFLOP/s, 0.35), MFU~U(0.2,0.65), "
                             "GPUs/replica in {1,2,4,8}"],
                alternatives=["tensor-parallel degree unknown (x8 swing)",
                              "MoE: active << total; timing cannot tell dense from MoE",
                              "attention FLOPs inflate the slope at long states"]))

        # ---- (b) server_ms vs #questions ------------------------------------------------
        if len({o[0] for o in nq_obs}) >= 2:
            qx = np.array([o[0] for o in nq_obs], float)
            qy = np.array([o[1] for o in nq_obs], float)
            q_slope, q_int, q_ci = theilsen(qx, qy)          # ms per question
            q_rows = []
            for q in nqs:
                pts = [o for o in nq_obs if o[0] == q]
                if pts:
                    q_rows.append({"n_questions": q, "n": len(pts),
                                   "median_server_ms": round(float(np.median([p[1] for p in pts])), 2),
                                   "input_tokens": int(np.median([p[2] for p in pts]))})
            res.tables["server_ms_vs_questions"] = q_rows
            res.series["server_ms_vs_questions"] = {
                "x": [r["n_questions"] for r in q_rows],
                "y": [r["median_server_ms"] for r in q_rows],
                "xlabel": "# questions (fixed state)", "ylabel": "median server ms", "kind": "line"}
            state_tok = float(np.median([o[2] for o in nq_obs if o[0] == min(nqs)] or [q_state_len]))
            ms_per_tok = slope * 1000.0 if slope > 0 else None
            tok_equiv = (q_slope / ms_per_tok) if ms_per_tok else None
            ratio = (tok_equiv / state_tok) if tok_equiv is not None and state_tok > 0 else None
            q_stats = {"ms_per_question": round(float(q_slope), 4),
                       "ms_per_question_ci": (round(float(q_ci[0]), 4), round(float(q_ci[1]), 4)),
                       "token_equivalents_per_question": (round(tok_equiv, 1)
                                                          if tok_equiv is not None else None),
                       "state_tokens": state_tok,
                       "question_cost_over_state_cost": (round(ratio, 4)
                                                         if ratio is not None else None),
                       "n": int(qx.size)}
            if ratio is not None and ratio < 0.1:
                res.add_evidence(Evidence(
                    property="decision.question_coupling",
                    likelihoods={"independent": 1.0, "coupled": 0.6},
                    summary=(f"+1 question costs {q_slope:.2f} ms (~{tok_equiv:.0f} state-token "
                             f"equivalents, {ratio:.1%} of re-reading the {state_tok:.0f}-token "
                             f"state) -> short branches over one shared prefilled state"),
                    level="inferred", strength=0.3, stats=q_stats,
                    alternatives=["one concatenated prompt would also add only the question's "
                                  "own tokens (timing alone cannot see isolation)"]))
                res.add_evidence(Evidence(
                    property="decision.scoring_mechanism",
                    likelihoods={"single_pass_heads": 1.0, "per_option_scoring": 0.6,
                                 "generative_then_parse": 0.15},
                    summary=(f"latency nearly flat in #questions ({q_slope:.2f} ms/question) -> "
                             f"no per-question decode loop; single-pass readout over "
                             f"prefix-shared branches"),
                    level="inferred", strength=0.4, stats=q_stats,
                    alternatives=["fully parallel batched generation could also look flat",
                                  "questions here are nouls; many-option choices may cost more"]))
            elif ratio is not None:
                res.notes.append(
                    f"each extra question costs ~{tok_equiv:.0f} token-equivalents "
                    f"({ratio:.0%} of the state) -> the state may be re-processed per question.")
                res.add_evidence(Evidence(
                    property="decision.scoring_mechanism",
                    likelihoods={"single_pass_heads": 0.5, "per_option_scoring": 1.0,
                                 "generative_then_parse": 0.8},
                    summary=(f"per-question cost ({q_slope:.2f} ms) is a large fraction of the "
                             f"state cost -> little prefix sharing across questions"),
                    level="inferred", strength=0.25, stats=q_stats,
                    alternatives=["noisy server times at small sizes",
                                  "queueing proportional to request size"]))

        # ---- (c) prefix caching --------------------------------------------------------
        if len(cache_rows) >= 2:
            cold_fresh = [o[1] for o in len_obs if o[0] == cache_len]
            first = [r["server_ms"] for r in cache_rows if r["role"] == "cold"]
            reps_ms = [r["server_ms"] for r in cache_rows if r["role"] == "repeat"]
            cold = float(np.median(cold_fresh + first))
            warm = float(np.median(reps_ms)) if reps_ms else cold
            overhead = max(0.0, intercept * 1000.0)
            prefill_cold = max(1e-6, cold - overhead)
            drop_ms = cold - warm
            drop_frac = drop_ms / prefill_cold
            thresh = max(5.0, 3.0 * noise_ms)
            cached = drop_ms > thresh and drop_frac > 0.4
            c_stats = {"cold_ms": round(cold, 2), "warm_ms": round(warm, 2),
                       "drop_ms": round(drop_ms, 2), "drop_fraction_of_prefill": round(drop_frac, 3),
                       "threshold_ms": round(thresh, 2), "state_words": cache_len,
                       "n_repeats": len(reps_ms)}
            res.tables["prefix_cache"] = cache_rows + [{"call": "fresh-text median",
                                                        "role": "cold reference",
                                                        "server_ms": round(float(np.median(cold_fresh)), 2)
                                                        if cold_fresh else None,
                                                        "input_tokens": None}]
            if cached:
                res.add_evidence(Evidence(
                    property="serving.prompt_caching", likelihoods={"yes": 1.0, "no": 0.08},
                    summary=(f"repeating an identical {cache_len}-word state cut server time "
                             f"{cold:.0f} -> {warm:.0f} ms ({drop_frac:.0%} of prefill) -> "
                             f"prefix/prompt caching"),
                    level="inferred", strength=0.8, stats=c_stats,
                    alternatives=["warm-up / autoscaling effects coinciding with the repeats",
                                  "load dropped during the repeats"]))
            else:
                res.add_evidence(Evidence(
                    property="serving.prompt_caching", likelihoods={"yes": 0.55, "no": 1.0},
                    summary=(f"no latency drop on an identical repeated {cache_len}-word state "
                             f"({cold:.0f} vs {warm:.0f} ms) -> no prefix cache hit observed"),
                    level="inferred", strength=0.35, stats=c_stats,
                    alternatives=["requests landed on different replicas (per-replica caches)",
                                  "cache needs a minimum prefix or more repeats to warm",
                                  "noise at this length hides a small saving"]))
            res.notes.append(f"prefix cache: cold {cold:.1f} ms vs repeat {warm:.1f} ms "
                             f"(drop {drop_ms:.1f} ms, threshold {thresh:.1f}) -> "
                             f"{'caching' if cached else 'no caching seen'}")

        res.shared["decision_timing"] = {"ms_per_ktok": ms_per_k,
                                         "intercept_ms": intercept * 1000.0,
                                         "prefill_tokens_per_s": tps,
                                         "quadratic": quad}
        res.notes.append(
            f"server_ms = {intercept * 1000:.1f} ms + {ms_per_k:.2f} ms/ktok "
            f"(n={x_tok.size}, up to {int(x_tok.max())} tokens; noise ~{noise_ms:.1f} ms); "
            f"fit degree {sel.get('degree')}, curvature {'significant' if quad else 'not significant'}.")
        return res

    # ------------------------------------------------------- active params MC
    @staticmethod
    def _active_params_mc(ctx: ProbeContext, tokens_per_s: float, n: int = 20000) -> np.ndarray:
        rng = np.random.default_rng(int(ctx.rng.integers(1 << 30)))
        peak = rng.lognormal(mean=np.log(0.99e15), sigma=0.35, size=n)     # FLOP/s per GPU
        mfu = rng.uniform(0.2, 0.65, size=n)
        gpus = rng.choice([1, 2, 4, 8], size=n, p=[0.35, 0.25, 0.15, 0.25])
        return peak * mfu * gpus / (2.0 * tokens_per_s) / 1e9
