"""P03 — hidden size (d_model) from the logit low-rank geometry. FLAGSHIP.

For hidden state ``h in R^d`` and a fixed unembedding ``E in R^{Vxd}`` the logits are
``z = E h (+ bias)``. Stacked over many prompts and restricted to a fixed token column
set ``C`` (``|C| > d``), the matrix ``Z_C = H E_C^T`` has rank exactly ``d`` (up to the
per-row log-softmax offset, killed by row-centering). The number of large singular values
*is* the hidden size (Finlayson et al. 2024; Carlini et al. 2024). See
``research/techniques_logits_tokenizer.md`` §1-§5 and the DEVGUIDE "Hidden-size lessons".

CORRECT ACQUISITION (position-0 only). The row of a logit matrix must be a function of the
PROMPT alone, otherwise rows cannot be aligned across the requests used to reconstruct them.
The only decode position that qualifies is **position 0** (the first generated token): its
logits depend on the prompt, not on which token was sampled, so they are identical across
every ``logit_bias`` group and across repeated calls. Reconstructing rows from *multiple*
decode positions is wrong on a real API: a positive ``logit_bias`` changes which tokens are
sampled, so continuations diverge across bias groups and positions ``>0`` no longer refer to
the same hidden state. We therefore take exactly ONE row per prompt = its position-0 logit
vector, and use many DISTINCT prompts (``> ~1.6 d``) for the rows.

Two acquisition regimes (chosen from ``caps.logit_bias_affects_logprobs``):

* **post-bias stacks** (bias affects reported logprobs): to observe a fixed column set
  ``C`` (``|C| = m > d``) wider than ``top_logprobs`` (``k``), query the SAME prompt in
  ``n_groups = ceil((m-1)/(k-1))`` groups. Each group applies a large positive bias ``B`` to
  ``k-1`` fresh columns plus one shared reference token ``r`` (present in every group); at
  position 0 the reported ``lp(c) - lp(r) = z_c - z_r`` cancels both the log-softmax offset
  and the (equal) bias ``B``, recovering the offset-free logit row over ``C``. Requests per
  prompt = ``n_groups``; total ``= n_prompts * n_groups``.
* **top-k only** (bias does not affect logprobs, e.g. gpt-4o post-mitigation, or logit_bias
  unsupported): union the natural top-k tokens across prompts, keep the densest common column
  block, estimate the rank as a lower bound (usually ``|C| <= d`` so the result is
  indeterminate).

Row-center, then :func:`estimate_rank` with ``method='gap'`` (primary) plus ``mp`` and
``bootstrap`` as cross-checks. Recovery is exact on fp32 logits and within +-1-2 on bf16
(quantisation buries the smallest signal singular values).

Budget: the required request count (``n_prompts * n_groups``) is computed up front. If the
budget / fast-mode cannot reach ``rows > d`` AND ``cols > d``, the probe does not silently
fail — it gathers what it can and returns a lower bound or ``None`` with an honest
identifiability level and a recommendation to raise the budget / give a size hint.

``requires=('logprobs',)`` — reasoning-class endpoints that reject logprobs are skipped
(hidden_size is non-identifiable there, identifiability level 5).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from forensics.core.types import GenRequest
from forensics.probes.base import (
    ASSUMPTION, NON_IDENTIFIABLE, STATISTICAL, WEAK, BudgetExhausted, Evidence, Estimate,
    Probe, ProbeContext, ProbeResult, register,
)
from forensics.stats.rank import estimate_rank

# Absolute safety ceiling on requests when there is neither a budget nor fast mode set
# (avoids a runaway 10^6-request plan on the default d hint with no explicit budget).
_HARD_REQUEST_CEILING = 60_000
_FAST_REQUEST_CAP = 28
_ESCALATION_ATTEMPTS = 3


@register
class HiddenSizeProbe(Probe):
    id = "P03.logits.hidden_size"
    title = "Hidden size (d_model) via centered SVD of the position-0 logit matrix"
    category = "logits"
    kinds = ("generative",)
    informs = {"output.head": 0.3, "arch.family": 0.15}
    estimates = ("dims.hidden_size",)
    requires = ("logprobs",)
    cost = 24

    # ------------------------------------------------------------------ helpers
    def _expected_d(self, ctx: ProbeContext) -> int:
        d = ctx.config.get("p03_expected_d")
        if d is None:
            d = ctx.shared.get("expected_hidden_size")
        return int(d) if d else 4096

    def _request_headroom(self, ctx: ProbeContext) -> int:
        """How many more requests P03 may spend (fast-cap / budget / hard ceiling)."""
        caps: list[int] = [_HARD_REQUEST_CEILING]
        if ctx.fast():
            caps.append(_FAST_REQUEST_CAP - ctx.n_requests)
        b = ctx.budget
        if b is not None and getattr(b, "max_requests", None) is not None:
            caps.append(int(b.max_requests) - int(getattr(b, "requests", 0)))
        return max(0, min(caps))

    def _dtype_margin(self, ctx: ProbeContext) -> int:
        """CI half-width contribution from the logit dtype (bf16/fp16 -> ±2, fp8 -> ±4)."""
        dt = ctx.shared.get("logit_dtype")
        if dt == "fp32":
            return 0
        if dt in ("bf16", "fp16"):
            return 2
        if dt == "fp8_or_lower":
            return 4
        return 2  # unknown -> assume some quantisation noise floor

    # ------------------------------------------------------------------ bias acquisition
    async def _acquire_bias(self, ctx: ProbeContext, k: int, m: int, n_prompts: int,
                            B: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Reconstruct offset-free position-0 logit rows over a fixed column set.

        One row per prompt (position 0 only). Columns ``C`` are covered in groups of
        ``k`` biased token ids (one shared reference ``r`` + ``k-1`` fresh columns);
        ``lp(c) - lp(r)`` at position 0 cancels the log-softmax offset and the bias.
        """
        C = list(range(2, 2 + m))
        r = C[0]
        col_index = {c: i for i, c in enumerate(C)}
        groups: list[list[int]] = []
        i = 1
        while i < m:
            groups.append([r] + C[i:i + (k - 1)])
            i += (k - 1)
        n_groups = len(groups)
        rows: list[np.ndarray] = []
        n_fail = 0
        n_used = 0
        exhausted = False
        for p in range(n_prompts):
            prompt = (f"hidden-geometry position-0 probe row {p} :: "
                      f"alpha beta gamma delta {p * 7 + 3} epsilon zeta")
            row = np.full(m, np.nan)
            seen: set[int] = set()
            for g in groups:
                bias = {t: B for t in g}
                try:
                    resp = await ctx.gen(GenRequest.chat(
                        prompt, logprobs=True, top_logprobs=k, max_tokens=1,
                        temperature=0.0, seed=7, logit_bias=bias))
                except BudgetExhausted:
                    exhausted = True
                    break
                if not resp.ok or not resp.tokens:
                    n_fail += 1
                    continue
                tok = resp.tokens[0]  # POSITION 0 ONLY (prompt-only, bias-invariant)
                lp = {tt.token_id: tt.logprob for tt in (tok.top or [])
                      if tt.token_id is not None}
                if r not in lp:
                    continue
                lp_r = lp[r]
                for c in g:
                    if c in lp:
                        row[col_index[c]] = lp[c] - lp_r
                        seen.add(c)
            n_used += 1
            if len(seen) >= m and not np.isnan(row).any():
                rows.append(row)
            if exhausted:
                break
        M = np.array(rows) if rows else np.zeros((0, m))
        meta = {"path": "logit_bias_pos0", "n_groups": n_groups,
                "n_prompts": n_used, "n_fail": n_fail, "m": m, "k": k, "B": B,
                "exhausted": exhausted}
        return M, meta

    async def _acquire_bias_escalating(self, ctx: ProbeContext, k: int, B: float,
                                       exp_d: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Size the matrix from ``exp_d``; if the hint is too small (no cliff) and budget
        remains, grow the column/row targets and retry."""
        attempt_d = max(8, int(exp_d))
        best_M: np.ndarray = np.zeros((0, 0))
        best_meta: dict[str, Any] = {"path": "logit_bias_pos0", "k": k, "B": B,
                                     "n_groups": 0}
        for _attempt in range(_ESCALATION_ATTEMPTS):
            m = int(1.3 * attempt_d) + k
            rows_target = int(1.6 * attempt_d) + k
            n_groups = max(1, math.ceil((m - 1) / max(1, k - 1)))
            required = rows_target * n_groups
            head = self._request_headroom(ctx)
            best_meta["required"] = required
            best_meta.setdefault("n_groups", n_groups)
            if head < n_groups * 3:  # cannot even reconstruct 3 full rows
                best_meta["budget_insufficient"] = True
                break
            n_prompts = min(rows_target, head // n_groups)
            M, meta = await self._acquire_bias(ctx, k, m, n_prompts, B)
            meta["required"] = required
            meta["attempt_d"] = attempt_d
            if M.size > best_M.size:
                best_M, best_meta = M, meta
            if M.ndim == 2 and M.shape[0] >= 3 and M.shape[1] >= 3:
                g = estimate_rank(M, method="gap", center=True)
                if g["has_cliff"] and not g["indeterminate"]:
                    return M, meta            # clean cliff -> done
                if g["indeterminate"] and not ctx.fast() and not meta["exhausted"]:
                    attempt_d = int(attempt_d * 2)  # hint too small -> grow and retry
                    continue
            break
        return best_M, best_meta

    # ------------------------------------------------------------------ top-k acquisition
    async def _acquire_topk(self, ctx: ProbeContext, k: int, n_prompts: int,
                            max_tokens: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Top-k-only fallback: union natural top-k, keep the densest common column block."""
        per_row: list[dict[int, float]] = []
        n_fail = 0
        for p in range(n_prompts):
            prompt = (f"entropy source {p}: 你好 Привет "
                      "✈ ☀ the of a to in 本 مر")
            try:
                resp = await ctx.gen(GenRequest.chat(
                    prompt, logprobs=True, top_logprobs=k, max_tokens=max_tokens,
                    temperature=1.0, seed=p))
            except BudgetExhausted:
                break
            if not resp.ok or not resp.tokens:
                n_fail += 1
                continue
            for tok in resp.tokens:      # each position is an independent valid row
                d = {tt.token_id: tt.logprob for tt in (tok.top or [])
                     if tt.token_id is not None}
                if d:
                    per_row.append(d)
        # column frequency
        freq: dict[int, int] = {}
        for d in per_row:
            for c in d:
                freq[c] = freq.get(c, 0) + 1
        cols = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])]
        best_M = np.zeros((0, 0))
        for m in range(len(cols), 1, -1):
            Cset = cols[:m]
            rows = [[d[c] for c in Cset] for d in per_row if all(c in d for c in Cset)]
            if len(rows) >= m + 1:
                best_M = np.array(rows, dtype=float)
                break
        return best_M, {"path": "topk_union", "n_prompts": n_prompts, "n_fail": n_fail,
                        "m": int(best_M.shape[1]) if best_M.size else 0,
                        "n_groups": 1, "k": k}

    # ------------------------------------------------------------------ reporting helpers
    def _report_insufficient(self, ctx: ProbeContext, res: ProbeResult,
                             meta: dict[str, Any], M: np.ndarray) -> ProbeResult:
        """Budget / interface could not build a matrix exceeding d in both dimensions."""
        res.status = "partial"
        required = meta.get("required")
        path = meta.get("path", "?")
        shape = tuple(int(x) for x in M.shape) if getattr(M, "ndim", 0) == 2 else ()
        req_txt = (f"~{required} requests (n_prompts*n_groups)" if required
                   else "more requests")
        res.notes.append(
            f"hidden-size extraction needs {req_txt}; budget/interface insufficient "
            f"(matrix {shape}, requests spent {ctx.n_requests})")
        res.notes.append(
            "recommendation: raise max_requests to the required count, disable fast mode, "
            "increase top_logprobs (k) to cut n_groups, or set config['p03_expected_d'] to "
            "size the matrix")
        res.add_estimate(Estimate(
            quantity="dims.hidden_size", value=None, level="speculative",
            method=f"centered-SVD ({path}); budget-limited",
            identifiability=NON_IDENTIFIABLE,
            assumptions=["hidden-size extraction needs ~n_prompts*n_groups requests; "
                         "budget/interface insufficient"],
            alternatives=["raise the request budget or disable fast mode",
                          "use a larger top_logprobs (k) to reduce n_groups",
                          "provide a p03_expected_d hint to size the matrix"]))
        res.shared["hidden_size"] = None
        res.n_requests = ctx.n_requests
        return res

    # ------------------------------------------------------------------ run
    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        res = self.result()
        if not getattr(ctx.caps, "logprobs", None):
            res.status = "skipped"
            res.notes.append("endpoint does not expose logprobs; hidden_size not identifiable")
            res.add_estimate(Estimate(
                quantity="dims.hidden_size", value=None, level="speculative",
                method="unavailable (no logprobs)", identifiability=NON_IDENTIFIABLE,
                assumptions=["logprobs required for logit-geometry rank recovery"]))
            res.shared["hidden_size"] = None
            return res

        exp_d = self._expected_d(ctx)
        k = max(2, min(int(ctx.caps.max_top_logprobs or 5), 20))
        B = float(ctx.caps.logit_bias_max or 100.0)
        bias_ok = bool(ctx.caps.logit_bias and ctx.caps.logit_bias_affects_logprobs)

        if bias_ok:
            M, meta = await self._acquire_bias_escalating(ctx, k, B, exp_d)
        else:
            # top-k union fallback: many prompts (each position is an independent row)
            head = self._request_headroom(ctx)
            n_prompts = min(head, 8 if ctx.fast() else 40)
            M, meta = await self._acquire_topk(
                ctx, k, max(3, n_prompts), max_tokens=12 if ctx.fast() else 24)
            meta["required"] = None

        res.tables["acquisition"] = [{
            "path": meta.get("path", "?"),
            "rows": int(M.shape[0]) if M.ndim == 2 else 0,
            "cols": int(M.shape[1]) if M.ndim == 2 else 0,
            "requests": int(ctx.n_requests),
            "n_groups": meta.get("n_groups"),
            "required": meta.get("required"),
            "k": meta.get("k"), "B": meta.get("B"),
        }]
        res.n_requests = ctx.n_requests

        if M.ndim != 2 or M.shape[0] < 3 or M.shape[1] < 3:
            return self._report_insufficient(ctx, res, meta, M)

        # ---- centered-SVD rank: gap primary, mp + bootstrap cross-check --------
        g = estimate_rank(M, method="gap", center=True)
        mp = estimate_rank(M, method="mp", center=True)
        boot = estimate_rank(M, method="bootstrap", center=True, rng=ctx.rng,
                             n_boot=20 if ctx.fast() else 40)
        r_gap = g["rank"]
        has_cliff = g["has_cliff"]
        indeterminate = g["indeterminate"]
        agree = (boot["rank"] == r_gap)

        sv = g["singular_values"]
        res.series["singular_values"] = {
            "x": list(range(1, len(sv) + 1)), "y": [float(v) for v in sv],
            "xlabel": "index", "ylabel": "singular value", "kind": "line"}
        res.tables["rank_methods"] = [
            {"method": "gap", "rank": r_gap, "has_cliff": has_cliff,
             "confidence": round(g["confidence"], 3)},
            {"method": "mp", "rank": mp["rank"], "threshold": round(mp["mp_threshold"], 4)},
            {"method": "bootstrap", "rank": boot["rank"],
             "confidence": round(boot["confidence"], 3)},
        ]

        if indeterminate:
            # full-rank over the sampled matrix -> observed rank is a LOWER BOUND on d
            lower = int(r_gap)
            res.notes.append(
                f"no signal/noise cliff: hidden_size >= {lower} "
                f"(matrix {M.shape}; collect more rows AND columns than d)")
            res.add_estimate(Estimate(
                quantity="dims.hidden_size", value=lower, ci=(float(lower), float("inf")),
                level="inferred", method=f"centered-SVD ({meta.get('path')}), lower bound",
                identifiability=ASSUMPTION,
                assumptions=["observed matrix rank-limited by rows/columns < true d",
                             "no lm_head bias (else recovered rank is d+1)"],
                alternatives=["true d larger; increase distinct prompts and token columns"]))
            res.add_evidence(Evidence(
                property="output.head",
                likelihoods={"generative_lm_head": 1.0, "classification_heads": 0.6},
                summary=f"logit cloud spans >= {lower} dims (no cliff yet)",
                level="inferred", strength=0.2,
                stats={"lower_bound": lower, "cols": M.shape[1], "rows": M.shape[0]},
                alternatives=["not enough rows/columns to see the bottleneck"]))
            res.shared["hidden_size"] = None
            res.shared["hidden_size_lower_bound"] = lower
            res.n_requests = ctx.n_requests
            return res

        # ---- clean cliff: report d = rank ------------------------------------
        margin = self._dtype_margin(ctx) + (0 if (has_cliff and agree) else 2)
        margin = max(1, margin)
        lo, hi = max(1, r_gap - margin), r_gap + margin
        res.add_estimate(Estimate(
            quantity="dims.hidden_size", value=int(r_gap), ci=(float(lo), float(hi)),
            level="inferred", method=f"centered-SVD gap ({meta.get('path')})",
            identifiability=STATISTICAL,
            assumptions=["no lm_head bias (else recovered rank is d+1)",
                         "logit_bias reaches reported position-0 logprobs"
                         if meta.get("path") == "logit_bias_pos0"
                         else "shared top-k support exists across prompts"],
            alternatives=["bf16/fp8 noise floor can shift the gap by +-1-2",
                          "MoE per-expert adapters could inflate apparent rank"],
        ))

        # ---- evidence: output.head + arch.family (weak) -----------------------
        vocab_hint = int((ctx.shared.get("tokenizer") or {}).get("vocab_size") or 0)
        wide = vocab_hint == 0 or r_gap < 0.5 * vocab_hint
        res.add_evidence(Evidence(
            property="output.head",
            likelihoods={"generative_lm_head": 1.0 if wide else 0.6,
                         "classification_heads": 0.3 if wide else 0.9},
            summary=f"sharp rank cliff at d={r_gap} (softmax bottleneck)",
            level="inferred", strength=0.3,
            stats={"rank": r_gap, "gap_confidence": round(g["confidence"], 3),
                   "cols": M.shape[1], "rows": M.shape[0], "vocab_hint": vocab_hint},
            alternatives=["classification head with many classes could also be low rank"],
        ))
        res.add_evidence(Evidence(
            property="arch.family",
            likelihoods={"decoder_only": 1.0, "encoder_only": 0.9, "encoder_decoder": 0.9,
                         "ssm_or_linear": 0.8, "non_neural": 0.1},
            summary="low-rank logit geometry consistent with a neural LM (rules out non-neural)",
            level="inferred", strength=0.15,
            stats={"rank": r_gap}, alternatives=["any neural LM shows this bottleneck"],
        ))

        res.shared["hidden_size"] = int(r_gap)
        res.notes.append(
            f"d_hat={r_gap} (gap), mp={mp['rank']}, bootstrap={boot['rank']} "
            f"conf={boot['confidence']:.2f}; matrix {M.shape}; requests {ctx.n_requests}")
        res.n_requests = ctx.n_requests
        return res
