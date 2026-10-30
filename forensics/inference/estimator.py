"""Dimension estimator: measured quantities + the reference DB -> a full architecture
table with honest confidence intervals.

The probes measure a few dimensions directly or statistically:

* ``dims.hidden_size`` (P03, logit-matrix rank),
* ``dims.vocab_size`` + ``tokenizer.*`` (P02),
* ``context.*`` (P05), ``out.logit_dtype`` (P04),
* ``timing.*`` and a memory-bound ``dims.active_params`` (P07),
* ``dims.total_params`` (P12, knowledge-capacity), MoE counts (P11).

Everything else (layers, heads, kv-heads, head-dim, FFN width, and the parameter
counts when not otherwise measured) is *derived* from the empirical scaling laws in
:class:`forensics.knowledge.archdb.ArchDB` conditioned on the measured ``d_model``
(and ``vocab``).  Because a black box does not expose these, they are produced by a
Monte-Carlo that propagates (a) the uncertainty on the measured inputs, (b) the
regression residual spread, and (c) coefficient uncertainty, into honest percentile
CIs.  Every derived quantity is labelled ``estimated``/``speculative`` with a weak or
assumption-level identifiability, a ``method`` that names the regression (n, resid),
and a list of alternatives.

A measured (observed/inferred/estimated) estimate is **never** overwritten by a weaker
regressed one: when both exist we keep the measurement and record agreement/conflict.

Finally the estimator builds a measured-constraint dict and asks the DB for the nearest
reference families (:meth:`ArchDB.nearest`), turning distances into 0..1 scores.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from forensics.knowledge.archdb import ArchDB
from forensics.probes.base import (
    ASSUMPTION, DIRECT, NON_IDENTIFIABLE, STATISTICAL, WEAK, Estimate,
)
from forensics.inference.family_match import match_families as _match_families, score_ranked
from forensics.inference.run_types import FamilyMatch

# --------------------------------------------------------------------------- constants
_MC_DRAWS = 6000
_FLOOR = 1e-9

# ordering of the report's "Estimated architecture table"
_TABLE_ORDER: tuple[str, ...] = (
    "dims.hidden_size", "dims.vocab_size", "dims.n_layers", "dims.n_heads",
    "dims.n_kv_heads", "dims.head_dim", "dims.ffn_dim", "dims.total_params",
    "dims.active_params", "dims.n_experts", "dims.active_experts", "arch.sparsity",
    "context.max_input_tokens", "context.max_output_tokens",
    "context.effective_recall_tokens", "context.sliding_window",
    "out.logit_dtype", "tokenizer.family",
)

_LEVEL_RANK = {"observed": 0, "inferred": 1, "estimated": 2, "speculative": 3}
_IDENT_LABEL = {
    DIRECT: "direct", STATISTICAL: "statistical", WEAK: "weak",
    ASSUMPTION: "assumption", NON_IDENTIFIABLE: "non-identifiable",
}

# arch.family (hypothesis value) -> DB arch_type
_FAMILY_TO_ARCH: dict[str, str] = {
    "decoder_only": "decoder", "encoder_only": "encoder",
    "encoder_decoder": "encoder-decoder", "ssm_or_linear": "ssm",
    "hybrid_attn_ssm": "hybrid", "diffusion": "diffusion",
}


# --------------------------------------------------------------------------- belief access
def _belief_top(belief: Any, prop: str) -> tuple[Any, float]:
    """(value, prob) of the top hypothesis for ``prop``, robust to the Belief shape.

    Works with an agent-A ``Belief`` (``.top`` / ``.posterior`` / ``.as_dict``) or a plain
    ``{prop: {value: p}}`` dict, or ``None``.
    """
    if belief is None:
        return (None, 0.0)
    top = getattr(belief, "top", None)
    if callable(top):
        try:
            r = top(prop)
            if r and r[0] is not None:
                return (r[0], float(r[1]))
        except Exception:
            pass
    post: Any = None
    if hasattr(belief, "posterior"):
        post = belief.posterior
    elif hasattr(belief, "as_dict"):
        try:
            post = belief.as_dict()
        except Exception:
            post = None
    elif isinstance(belief, dict):
        post = belief
    if isinstance(post, dict) and isinstance(post.get(prop), dict) and post[prop]:
        d = post[prop]
        k = max(d, key=d.get)
        return (k, float(d[k]))
    return (None, 0.0)


# --------------------------------------------------------------------------- estimate index
def _est_rank(e: Estimate) -> tuple[int, int, float]:
    """Sort key: measured (value not None) first, then by level, then by CI tightness."""
    has_val = 0 if e.value is not None else 1
    lvl = _LEVEL_RANK.get(e.level, 4)
    width = math.inf
    if e.ci and all(math.isfinite(x) for x in e.ci) and e.value:
        try:
            width = abs(e.ci[1] - e.ci[0]) / max(abs(float(e.value)), 1e-9)
        except (TypeError, ValueError):
            width = math.inf
    return (has_val, lvl, width)


def _best_by_quantity(estimates: list[Estimate]) -> dict[str, Estimate]:
    """The single strongest estimate per quantity (measured > inferred > ... )."""
    best: dict[str, Estimate] = {}
    for e in estimates:
        cur = best.get(e.quantity)
        if cur is None or _est_rank(e) < _est_rank(cur):
            best[e.quantity] = e
    return best


def _num(e: Estimate | None) -> float | None:
    if e is None or e.value is None:
        return None
    try:
        return float(e.value)
    except (TypeError, ValueError):
        return None


def _draws_from_estimate(e: Estimate | None, rng: np.random.Generator,
                         n: int) -> np.ndarray | None:
    """Monte-Carlo draws consistent with a measured estimate's value + CI.

    Uses the estimate's own MC ``samples`` when present; else a normal within its CI
    (architecture constants are near-exact once measured, so a point value -> tight draws).
    """
    v = _num(e)
    if v is None:
        return None
    if e is not None and e.samples:
        s = np.asarray([float(x) for x in e.samples], dtype=float)
        s = s[np.isfinite(s) & (s > 0)]
        if s.size >= 8:
            return rng.choice(s, size=n, replace=True)
    sd = 0.0
    if e is not None and e.ci and all(math.isfinite(x) for x in e.ci) and e.ci[1] > e.ci[0]:
        z = 1.645 if abs(e.ci_level - 0.9) < 0.15 else 1.96
        sd = max((float(e.ci[1]) - float(e.ci[0])) / (2.0 * z), 0.0)
    if sd <= 0:
        return np.full(n, v, dtype=float)
    return np.clip(rng.normal(v, sd, n), 1.0, None)


# --------------------------------------------------------------------------- percentile CI
def _ci(draws: np.ndarray, level: float = 0.9) -> tuple[float, float]:
    lo = (1.0 - level) / 2.0
    a, b = np.quantile(draws, [lo, 1.0 - lo])
    return (float(a), float(b))


def _round_int(x: float) -> int:
    return int(max(1, round(x)))


# --------------------------------------------------------------------------- main entry
def estimate_dimensions(belief: Any, estimates: list[Estimate], caps: Any,
                        shared: dict | None = None) -> dict[str, Any]:
    """Turn measured quantities + the reference DB into a full architecture table.

    Returns ``{'estimates', 'family_matches', 'derived_table'}``:
      * ``estimates`` – the input estimates plus the newly derived ones (measurements
        are kept, never overwritten by a weaker regression),
      * ``family_matches`` – DB nearest-neighbour :class:`FamilyMatch` list,
      * ``derived_table`` – ordered report rows for the architecture table.
    """
    shared = shared or {}
    rng = np.random.default_rng(int(shared.get("seed", 0)))
    db = ArchDB.load()
    reg = db.regressions()

    measured = _best_by_quantity(estimates)

    # ---- belief-derived structural facts --------------------------------------
    fam_val, _ = _belief_top(belief, "arch.family")
    arch_type = _FAMILY_TO_ARCH.get(str(fam_val), None) if fam_val else None
    spars_val, _ = _belief_top(belief, "arch.sparsity")
    is_moe = str(spars_val) == "moe"
    kv_val, _ = _belief_top(belief, "attn.kv_sharing")

    era = _pick_era(measured, shared)

    # ---- measured numeric inputs ----------------------------------------------
    d_est = measured.get("dims.hidden_size")
    v_est = measured.get("dims.vocab_size")
    d0 = _num(d_est)
    v0 = _num(v_est)

    d_draws = _draws_from_estimate(d_est, rng, _MC_DRAWS)
    v_draws = _draws_from_estimate(v_est, rng, _MC_DRAWS)
    if v_draws is None and v0 is not None:
        v_draws = np.full(_MC_DRAWS, v0, dtype=float)

    out_estimates: list[Estimate] = list(estimates)
    table: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    # measured rows straight into the table (kept verbatim) ----------------------
    for q in ("dims.hidden_size", "dims.vocab_size", "context.max_input_tokens",
              "context.max_output_tokens", "context.effective_recall_tokens",
              "context.sliding_window", "out.logit_dtype", "tokenizer.family"):
        if q in measured and measured[q].value is not None:
            table[q] = _row_from_estimate(q, measured[q])

    derived_new: list[Estimate] = []

    # ---- derive the width-conditioned dimensions -------------------------------
    if d_draws is not None and d0 and d0 > 0:
        # n_layers ~ f(d_model)
        xbar, sxx = _dense_lnd_stats(db)
        L_draws = _mc_layers(reg["layers_vs_dmodel"], d_draws, xbar, sxx, rng)
        _place(table, derived_new, measured, "dims.n_layers", L_draws, unit="",
               level="estimated", ident=WEAK, reg=reg["layers_vs_dmodel"],
               method_extra="ln(n_layers)=a+b*ln(d_model), dense pop",
               alternatives=["depth decoupled from width (Gemma/Qwen3 fix head_dim, vary L)",
                             "MoE depth differs from dense scaling",
                             "confirm with per-token latency scaling (P07) or a family match"],
               integerize=True)

        # head_dim (empirical distribution) and n_heads = round(d/head_dim)
        hd_draws = _mc_head_dim(reg["head_dim"], kv_val, rng, _MC_DRAWS)
        _place(table, derived_new, measured, "dims.head_dim", hd_draws, unit="",
               level="estimated", ident=WEAK, reg=reg["head_dim"],
               method_extra="empirical head_dim distribution (mode=%d)" % reg["head_dim"]["coef"]["mode"],
               alternatives=["head_dim decoupled from width (Gemma 256, Qwen3 128, gpt-oss 64)",
                             "MLA uses q/k head_dim 192 (128 nope + 64 rope)"],
               integerize=True)
        nh_draws = np.maximum(1.0, np.round(d_draws / np.maximum(hd_draws, 1.0)))
        _place(table, derived_new, measured, "dims.n_heads", nh_draws, unit="",
               level="estimated", ident=WEAK, reg=reg["head_dim"],
               method_extra="n_heads=round(d_model/head_dim)",
               alternatives=["head_dim != d/H when width and heads decouple",
                             "MQA/MLA change the KV-head count, not query heads"],
               integerize=True)

        # n_kv_heads from the era GQA distribution (or MHA/MLA rule)
        kv_draws, kv_ident, kv_alts, kv_method = _mc_kv_heads(
            reg["gqa_ratio_by_era"], era, kv_val, nh_draws, rng)
        _place(table, derived_new, measured, "dims.n_kv_heads", kv_draws, unit="",
               level="speculative", ident=kv_ident, reg=reg["gqa_ratio_by_era"],
               method_extra=kv_method, alternatives=kv_alts, integerize=True)

        # ffn_dim = ratio * d_model (ratio by dense/moe + gated)
        ffn_draws, ffn_method, ffn_alts = _mc_ffn(reg["ffn_ratio"], is_moe, d_draws, rng)
        _place(table, derived_new, measured, "dims.ffn_dim", ffn_draws, unit="",
               level="estimated", ident=WEAK, reg=reg["ffn_ratio"],
               method_extra=ffn_method, alternatives=ffn_alts, integerize=True)

        # ---- parameters --------------------------------------------------------
        _handle_params(db, reg, measured, table, derived_new, notes, d_draws, v_draws,
                       v0, L_draws, d0, is_moe, rng)
    else:
        notes.append("hidden_size not identified within budget → layer/head/param "
                     "estimates unavailable")
        # still surface a measured/derived param picture if possible
        _handle_params(db, reg, measured, table, derived_new, notes, None, v_draws,
                       v0, None, d0, is_moe, rng)

    # ---- MoE / sparsity row + active<total flag --------------------------------
    _sparsity_row(table, measured, is_moe, notes)

    # MoE expert counts (measured only; from P11)
    for q in ("dims.n_experts", "dims.active_experts"):
        if q in measured and measured[q].value is not None:
            table[q] = _row_from_estimate(q, measured[q])

    out_estimates.extend(derived_new)

    # ---- family match ----------------------------------------------------------
    # Only evidence-backed belief categories + MEASURED quantities become constraints, and
    # scores reflect constraint coverage (see forensics.inference.family_match).
    family_matches, fm_notes = _match_families(belief, estimates, db)
    notes.extend(fm_notes)

    derived_table = [table[q] for q in _TABLE_ORDER if q in table]
    for r in derived_table:
        r.setdefault("notes", "")
    return {
        "estimates": out_estimates,
        "family_matches": family_matches,
        "derived_table": derived_table,
        "notes": notes,
    }


# --------------------------------------------------------------------------- era
def _pick_era(measured: dict[str, Estimate], shared: dict) -> str:
    """GQA-ratio era bucket from a measured knowledge cutoff, else the modern default."""
    year = None
    ke = measured.get("knowledge.cutoff")
    if ke is not None and ke.value:
        try:
            year = int(str(ke.value).split("-", 1)[0])
        except (ValueError, TypeError):
            year = None
    if year is None:
        year = int(shared.get("release_year", 2025))
    if year <= 2022:
        return "<=2022"
    if year == 2023:
        return "2023"
    if year == 2024:
        return "2024"
    return ">=2025"


# --------------------------------------------------------------------------- MC pieces
def _dense_lnd_stats(db: ArchDB) -> tuple[float, float]:
    """(mean, Sxx) of ln(d_model) over the dense decoder population (for centered CIs)."""
    lnd = [math.log(r["d_model"]) for r in db.unique()
           if r.get("arch_type") in ("decoder", "diffusion") and not r.get("moe")
           and r.get("d_model") and r.get("n_layers")]
    if len(lnd) < 3:
        return (math.log(4096.0), 1.0)
    arr = np.asarray(lnd, dtype=float)
    xbar = float(arr.mean())
    sxx = float(((arr - xbar) ** 2).sum()) or 1.0
    return (xbar, sxx)


def _mc_layers(reg: dict[str, Any], d_draws: np.ndarray, xbar: float, sxx: float,
               rng: np.random.Generator) -> np.ndarray:
    a = reg["coef"]["intercept"]
    b = reg["coef"]["slope"]
    s = float(reg["resid_std"])
    n = max(int(reg["n"]), 2)
    lnd = np.log(d_draws)
    # textbook predictive draw: residual + centered coefficient uncertainty
    # var(coef contribution) = s^2 * (1/n + (lnd - xbar)^2 / Sxx)  -> ~0 at the centroid
    coef_sd = s * np.sqrt(1.0 / n + (lnd - xbar) ** 2 / sxx)
    ln_l = a + b * lnd + rng.normal(0.0, s, d_draws.size) \
        + rng.normal(0.0, 1.0, d_draws.size) * coef_sd
    return np.clip(np.exp(ln_l), 1.0, None)


def _mc_head_dim(reg: dict[str, Any], kv_val: Any, rng: np.random.Generator,
                 n: int) -> np.ndarray:
    if str(kv_val) == "mla":
        # MLA q/k head_dim is 192 (128 nope + 64 rope) in the DB.
        return np.full(n, 192.0)
    probs = reg["coef"]["probs"]
    vals = np.array(sorted(probs, key=lambda k: -probs[k]), dtype=float)
    p = np.array([probs[int(v)] for v in vals], dtype=float)
    p = p / p.sum()
    return rng.choice(vals, size=n, p=p)


def _mc_kv_heads(reg: dict[str, Any], era: str, kv_val: Any, nh_draws: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, int, list[str], str]:
    n = nh_draws.size
    if str(kv_val) == "mha":
        return (nh_draws.copy(), WEAK,
                ["MHA: n_kv_heads == n_heads (no sharing)"],
                "MHA => n_kv_heads = n_heads")
    if str(kv_val) == "mla":
        return (np.full(n, 1.0), ASSUMPTION,
                ["MLA caches a single latent (512+64), not per-head KV"],
                "MLA => compressed latent (reported as 1 effective KV group)")
    coef = reg["coef"].get(era) or {}
    counts = coef.get("n_kv_head_counts") or {}
    if counts:
        vals = np.array(list(counts.keys()), dtype=float)
        p = np.array(list(counts.values()), dtype=float)
        p = p / p.sum()
        kv = rng.choice(vals, size=n, p=p)
        # never exceed the query-head count
        kv = np.minimum(kv, nh_draws)
        method = ("era %s GQA distribution (modal n_kv_heads=%d)"
                  % (era, int(vals[int(np.argmax(p))])))
        return (np.maximum(1.0, kv), ASSUMPTION,
                ["GQA ratio varies by vendor; KV=8 is the modal TP=8 serving width",
                 "MQA (KV=1) and MHA (KV=H) are the extremes"],
                method)
    # fallback: median H/KV ratio
    ratio = float(coef.get("median_h_over_kv", 4.0)) or 4.0
    kv = np.maximum(1.0, np.round(nh_draws / ratio))
    return (kv, ASSUMPTION, ["GQA ratio from era median H/KV"],
            "era %s median H/KV=%.1f" % (era, ratio))


def _mc_ffn(reg: dict[str, Any], is_moe: bool, d_draws: np.ndarray,
            rng: np.random.Generator) -> tuple[np.ndarray, str, list[str]]:
    coef = reg["coef"]
    if is_moe:
        st = coef["moe_expert"]
        med = float(st.get("median", 0.68)) or 0.68
        lsd = float(st.get("log_sd", 0.84)) or 0.84
        ratios = med * np.exp(rng.normal(0.0, lsd, d_draws.size))
        method = "moe_intermediate/d_model (per-expert FFN width)"
        alts = ["expert width spans fine-grained (0.29) to coarse (1.75+) experts",
                "shared experts add width beyond the routed experts"]
    else:
        st = coef["dense_gated"]
        med = float(st.get("median", 3.3)) or 3.3
        lsd = float(st.get("log_sd", 0.35)) or 0.35
        ratios = med * np.exp(rng.normal(0.0, lsd, d_draws.size))
        method = "d_ffn/d_model (dense gated FFN)"
        alts = ["non-gated FFN uses ratio 4.0 exactly",
                "SwiGLU 8/3 (~2.67) and Gemma (8.0) clusters exist"]
    return (np.maximum(1.0, ratios * d_draws), method, alts)


def _handle_params(db: ArchDB, reg: dict[str, Any], measured: dict[str, Estimate],
                   table: dict[str, dict[str, Any]], derived_new: list[Estimate],
                   notes: list[str], d_draws: np.ndarray | None,
                   v_draws: np.ndarray | None, v0: float | None,
                   L_draws: np.ndarray | None, d0: float | None,
                   is_moe: bool, rng: np.random.Generator) -> None:
    """Total & active parameter counts, reconciling DB formula with measurements."""
    tot_meas = measured.get("dims.total_params")
    act_meas = measured.get("dims.active_params")

    # DB dense param formula draws (a lower bound for MoE)
    formula_draws: np.ndarray | None = None
    if d_draws is not None and L_draws is not None and v_draws is not None:
        formula_draws = _mc_params(reg["params_vs_ldv"], L_draws, d_draws, v_draws, rng)

    # ---- total_params ----------------------------------------------------------
    tot_med: float | None = None
    tot_ci: tuple[float, float] | None = None
    if is_moe:
        # dense formula understates MoE totals; prefer the measured knowledge total.
        if tot_meas is not None and tot_meas.value is not None:
            table["dims.total_params"] = _row_from_estimate("dims.total_params", tot_meas)
            tot_med = _num(tot_meas)
            tot_ci = tot_meas.ci
            if formula_draws is not None:
                lb = float(np.median(formula_draws))
                table["dims.total_params"]["method"] += (
                    " | dense-formula lower bound %.1fB (experts add the rest)" % lb)
        elif formula_draws is not None:
            tot_med = float(np.median(formula_draws))
            tot_ci = _ci(formula_draws)
            est = _mk_estimate("dims.total_params", tot_med, tot_ci, "speculative",
                               ASSUMPTION, reg["params_vs_ldv"],
                               "dense-formula LOWER BOUND for MoE (expert count unknown)",
                               ["MoE total >> dense formula; scale by n_experts",
                                "use P11 expert counts or P12 knowledge capacity"], unit="B")
            derived_new.append(est)
            table["dims.total_params"] = _row_from_estimate("dims.total_params", est)
    else:
        if formula_draws is not None:
            tot_med = float(np.median(formula_draws))
            tot_ci = _ci(formula_draws)
            est = _mk_estimate("dims.total_params", round(tot_med, 2), tot_ci,
                               "speculative", ASSUMPTION, reg["params_vs_ldv"],
                               "P=exp(a+b*lnL+c*lnd)+k*V*d over MC (L,d,V,tie)",
                               ["depends on the regressed n_layers",
                                "tie_word_embeddings unknown (sampled by size prior)",
                                "MoE would break the dense formula"], unit="B")
            if tot_meas is not None and tot_meas.value is not None:
                _note_agreement(est, tot_meas, notes, "dims.total_params")
                # keep both: measured (speculative from knowledge) noted; formula primary
            derived_new.append(est)
            table["dims.total_params"] = _row_from_estimate("dims.total_params", est)
        elif tot_meas is not None and tot_meas.value is not None:
            table["dims.total_params"] = _row_from_estimate("dims.total_params", tot_meas)
            tot_med = _num(tot_meas)
            tot_ci = tot_meas.ci

    # ---- active_params ---------------------------------------------------------
    if act_meas is not None and act_meas.value is not None:
        # measured (P07 timing) wins; never overwritten.
        table["dims.active_params"] = _row_from_estimate("dims.active_params", act_meas)
        if tot_med and _num(act_meas) is not None:
            _note_agreement_active(table["dims.active_params"], _num(act_meas),
                                   tot_med, is_moe, notes)
    elif not is_moe and tot_med is not None:
        # dense: active == total
        est = _mk_estimate("dims.active_params", round(tot_med, 2), tot_ci,
                           "speculative", ASSUMPTION, reg["params_vs_ldv"],
                           "dense model: active_params = total_params",
                           ["MoE would make active << total"], unit="B")
        derived_new.append(est)
        row = _row_from_estimate("dims.active_params", est)
        row["method"] += " (dense: active=total)"
        table["dims.active_params"] = row
    elif is_moe and tot_med is not None:
        # MoE with no timing measurement: apply the DB active/total ratio.
        act_med, act_ci = _mc_active_from_ratio(reg["moe_active_ratio"], measured,
                                                tot_med, tot_ci, rng)
        est = _mk_estimate("dims.active_params", round(act_med, 2), act_ci,
                           "speculative", ASSUMPTION, reg["moe_active_ratio"],
                           "active=total*exp(a+b*ln(top_k/E)) (MoE ratio law)",
                           ["needs top_k / n_experts (P11) for a sharp estimate",
                            "timing-based active (P07) is the direct signal"], unit="B")
        derived_new.append(est)
        row = _row_from_estimate("dims.active_params", est)
        row["method"] += " | MoE: active << total"
        table["dims.active_params"] = row


def _mc_params(reg: dict[str, Any], L_draws: np.ndarray, d_draws: np.ndarray,
               v_draws: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = reg["coef"]
    s = float(reg["resid_std"])
    n = max(int(reg["n"]), 3)
    a = rng.normal(c["intercept"], s / math.sqrt(n), L_draws.size)
    mu = a + c["ln_L"] * np.log(L_draws) + c["ln_d"] * np.log(d_draws) \
        + rng.normal(0.0, s, L_draws.size)
    p_nonemb = np.exp(mu)
    # tie_word_embeddings prior by width (small models tie more often)
    d_med = float(np.median(d_draws))
    p_tie = 0.6 if d_med < 2048 else (0.35 if d_med < 4096 else 0.2)
    k = np.where(rng.random(L_draws.size) < p_tie, 1.0, 2.0)
    total = (p_nonemb + k * v_draws * d_draws) / 1e9
    return np.clip(total, _FLOOR, None)


def _mc_active_from_ratio(reg: dict[str, Any], measured: dict[str, Estimate],
                          tot_med: float, tot_ci: tuple[float, float] | None,
                          rng: np.random.Generator) -> tuple[float, tuple[float, float]]:
    c = reg["coef"]
    s = float(reg["resid_std"] or 0.23)
    n_exp = _num(measured.get("dims.n_experts"))
    act_exp = _num(measured.get("dims.active_experts"))
    if n_exp and act_exp and n_exp > 0:
        frac = act_exp / n_exp
    else:
        frac = float(np.clip(rng.normal(0.12, 0.08, _MC_DRAWS), 0.02, 0.6))  # broad prior
    x = np.log(np.clip(frac, 1e-4, 1.0))
    ln_ratio = c["intercept"] + c["slope"] * x + rng.normal(0.0, s, _MC_DRAWS)
    ratio = np.clip(np.exp(ln_ratio), 1e-3, 1.0)
    tot_draws = (np.full(_MC_DRAWS, tot_med) if tot_ci is None
                 else rng.uniform(tot_ci[0], tot_ci[1], _MC_DRAWS))
    active = ratio * tot_draws
    return (float(np.median(active)), _ci(active))


# --------------------------------------------------------------------------- placement
def _place(table: dict[str, dict[str, Any]], derived_new: list[Estimate],
           measured: dict[str, Estimate], quantity: str, draws: np.ndarray,
           unit: str, level: str, ident: int, reg: dict[str, Any],
           method_extra: str, alternatives: list[str], integerize: bool) -> None:
    """Place a derived quantity, keeping any measured (>=estimated) value instead."""
    med = float(np.median(draws))
    ci = _ci(draws)
    if integerize:
        value: Any = _round_int(med)
        ci = (float(round(ci[0])), float(round(ci[1])))
    else:
        value = round(med, 3)
    method = "DB regression (n=%s, resid=%s): %s" % (
        reg.get("n"), _fmt_resid(reg.get("resid_std")), method_extra)

    meas = measured.get(quantity)
    if meas is not None and meas.value is not None and _LEVEL_RANK.get(meas.level, 4) <= 2:
        # measured (observed/inferred/estimated) -> keep it, note agreement/conflict
        row = _row_from_estimate(quantity, meas)
        mv = _num(meas)
        if mv is not None and med > 0:
            rel = abs(mv - med) / med
            verdict = "agrees" if rel <= 0.15 else "CONFLICT"
            row["method"] += " | regressed ~%s (%s, %.0f%%)" % (
                _round_int(med) if integerize else round(med, 2), verdict, rel * 100)
        table[quantity] = row
        return

    est = _mk_estimate(quantity, value, ci, level, ident, reg, method_extra,
                       alternatives, unit=unit, samples=draws)
    est.method = method
    derived_new.append(est)
    table[quantity] = _row_from_estimate(quantity, est)


def _mk_estimate(quantity: str, value: Any, ci: tuple[float, float] | None,
                 level: str, ident: int, reg: dict[str, Any], method_extra: str,
                 alternatives: list[str], unit: str = "",
                 samples: np.ndarray | None = None) -> Estimate:
    method = "DB regression (n=%s, resid=%s): %s" % (
        reg.get("n"), _fmt_resid(reg.get("resid_std")), method_extra)
    smp = None
    if samples is not None:
        smp = [float(x) for x in np.asarray(samples)[:512]]
    return Estimate(
        quantity=quantity, value=value, ci=ci, ci_level=0.9, level=level,  # type: ignore[arg-type]
        method=method, identifiability=ident, unit=unit, samples=smp,
        assumptions=["measured d_model (and vocab) are correct",
                     "target follows the reference-DB scaling laws"],
        alternatives=alternatives, probe_id="estimator",
    )


def _fmt_resid(r: Any) -> str:
    if r is None:
        return "n/a"
    try:
        return "%.3f" % float(r)
    except (TypeError, ValueError):
        return str(r)


# --------------------------------------------------------------------------- rows
def _row_from_estimate(quantity: str, e: Estimate) -> dict[str, Any]:
    ci_str = _fmt_ci(e.ci, e.value)
    return {
        "quantity": quantity,
        "value": e.value,
        "ci": ci_str,
        "level": e.level,
        "identifiability": _IDENT_LABEL.get(e.identifiability, str(e.identifiability)),
        "method": e.method or "",
        "evidence": e.probe_id or "",
    }


def _fmt_ci(ci: tuple[float, float] | None, value: Any) -> str:
    if not ci:
        return ""
    lo, hi = ci
    if not (math.isfinite(lo) and math.isfinite(hi)):
        if math.isfinite(lo):
            return ">= %s" % _fmt_num(lo)
        return ""
    return "%s – %s" % (_fmt_num(lo), _fmt_num(hi))


def _fmt_num(x: float) -> str:
    ax = abs(x)
    if ax >= 1000:
        return format(int(round(x)), ",d")
    if ax >= 10:
        return "%d" % int(round(x))
    return "%.2f" % x


# --------------------------------------------------------------------------- sparsity
def _sparsity_row(table: dict[str, dict[str, Any]], measured: dict[str, Estimate],
                  is_moe: bool, notes: list[str]) -> None:
    value = "moe" if is_moe else "dense"
    method = "belief arch.sparsity (top hypothesis)"
    tot = table.get("dims.total_params", {}).get("value")
    act = table.get("dims.active_params", {}).get("value")
    flagged = False
    try:
        if tot is not None and act is not None and float(act) < float(tot) * 0.85:
            flagged = True
    except (TypeError, ValueError):
        pass
    if is_moe:
        if flagged:
            method += "; active (%s) << total (%s) => MoE confirmed" % (act, tot)
            notes.append("MoE signal: active_params < total_params")
        else:
            method += "; MoE (active/total not both resolved)"
    table["arch.sparsity"] = {
        "quantity": "arch.sparsity", "value": value, "ci": "",
        "level": "inferred", "identifiability": "weak",
        "method": method, "evidence": "belief",
    }


def _note_agreement(est: Estimate, meas: Estimate, notes: list[str], q: str) -> None:
    a, b = _num(est), _num(meas)
    if a and b and a > 0:
        rel = abs(a - b) / a
        verdict = "agrees" if rel <= 0.2 else "CONFLICT"
        notes.append("%s: formula %.2f vs measured %.2f (%s, %.0f%%)"
                     % (q, a, b, verdict, rel * 100))


def _note_agreement_active(row: dict[str, Any], active: float, total: float,
                           is_moe: bool, notes: list[str]) -> None:
    if total <= 0:
        return
    ratio = active / total
    if ratio < 0.85:
        row["method"] += " | active/total=%.2f (MoE: active << total)" % ratio
        notes.append("active_params (%.2f) << total_params (%.2f): MoE signal (ratio %.2f)"
                     % (active, total, ratio))
    elif not is_moe:
        row["method"] += " | active/total=%.2f (consistent with dense)" % ratio


# --------------------------------------------------------------------------- family match
def _family_matches(db: ArchDB, measured: dict[str, Estimate], arch_type: str | None,
                    kv_val: Any, is_moe: bool, top_k: int = 6) -> list[FamilyMatch]:
    """Legacy helper (explicit constraints). Requires at least one measured quantity and
    uses the coverage-aware score of :mod:`forensics.inference.family_match`; callers are
    responsible for passing only evidence-backed categorical constraints."""
    constraints: dict[str, Any] = {}
    d0 = _num(measured.get("dims.hidden_size"))
    v0 = _num(measured.get("dims.vocab_size"))
    ctx = _num(measured.get("context.max_input_tokens"))
    if d0:
        constraints["hidden_size"] = d0
    if v0:
        constraints["vocab_size"] = v0
    if ctx:
        constraints["context"] = ctx
    if not constraints:
        return []
    if arch_type:
        constraints["arch_type"] = arch_type
    if kv_val in ("mha", "gqa_or_mqa", "mla"):
        constraints["kv_sharing"] = kv_val
    if is_moe:
        constraints["moe"] = True
    ranked = db.nearest(constraints, top_k=None)
    return score_ranked(ranked, top_k=top_k) if ranked else []
