"""Behavioural fingerprint vector: turn a :class:`RunResult` into a stable, comparable
signature of *how an endpoint behaves*.

The fingerprint is a flat ``dict`` mixing

* **numeric features** (floats/ints): tokenizer token-delta vector, context limits and the
  recall knee, position-sensitivity, latency-curve coefficients (prefill degree-1/2, ITL
  slope), output/logprob distribution stats, sampling fidelity, refusal/safety rates,
  formatting/verbosity, numeric/tool behaviour, calibration, and every numeric quantity the
  probes/estimator produced; and
* **categorical tags** (strings): tokenizer family, logit dtype, truncation policy, the
  most-likely value of each hypothesis property, self-identified brand, serving signals, ...

Meta keys (prefixed ``_``) carry provenance and are ignored by :func:`fingerprint_distance`.

The vector is assembled *only* from what the probes published: ``run.probe_results[*].shared``,
``.tables`` / ``.series``, ``run.estimates`` and ``run.posterior``.  Every extractor is
defensive: a missing probe simply omits its features, so a partial run still fingerprints.

:func:`fingerprint_distance` compares two fingerprints with a scale-aware blend of a
symmetric-relative L1 and a cosine distance over the shared numeric keys (plus a numeric
key-coverage term), and a Jaccard distance over the categorical tags.  Identical behaviour
gives distance ``0``; unrelated models give a distance approaching ``1``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["fingerprint", "fingerprint_distance", "numeric_features", "tag_set"]

_EPS = 1e-12


# ============================================================================ helpers
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _finite(v: float) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _num(fp: dict[str, Any], key: str, val: Any) -> None:
    """Store a numeric feature (skipping ``None``/non-finite values)."""
    if val is None:
        return
    if isinstance(val, bool):
        fp[key] = 1.0 if val else 0.0
        return
    if not _is_number(val) or not _finite(val):
        return
    fp[key] = float(val)


def _tag(fp: dict[str, Any], key: str, val: Any) -> None:
    """Store a categorical tag (skipping empty values)."""
    if val is None:
        return
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan"):
        return
    fp[key] = s


def _merge_shared(run: Any) -> dict[str, Any]:
    """Fold every probe's ``shared`` scratchpad into one dict (later probes win)."""
    shared: dict[str, Any] = {}
    for pr in getattr(run, "probe_results", []) or []:
        if getattr(pr, "shared", None):
            shared.update(pr.shared)
    return shared


def _by_probe(run: Any) -> dict[str, Any]:
    return {pr.probe_id: pr for pr in (getattr(run, "probe_results", []) or [])}


def _estimates_map(run: Any) -> dict[str, Any]:
    """quantity -> most-recent non-None value across all estimates."""
    out: dict[str, Any] = {}
    for e in getattr(run, "estimates", []) or []:
        if getattr(e, "value", None) is not None:
            out[e.quantity] = e.value
    return out


def _first_table(pr: Any, name: str) -> list[dict[str, Any]]:
    if pr is None:
        return []
    return list((pr.tables or {}).get(name, []) or [])


def _rows_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [r[field] for r in rows if isinstance(r.get(field), (int, float, bool))]
    if not vals:
        return None
    return float(np.mean([float(v) for v in vals]))


# ============================================================================ extractors
def _add_tokenizer(fp: dict[str, Any], shared: dict, by_probe: dict, est: dict) -> None:
    tok = shared.get("tokenizer") or {}
    _tag(fp, "tokenizer.family", tok.get("family") or est.get("tokenizer.family"))
    _tag(fp, "tokenizer.type", tok.get("type") or est.get("tokenizer.type"))
    _num(fp, "tokenizer.vocab_size", tok.get("vocab_size") or est.get("dims.vocab_size"))
    _num(fp, "tokenizer.chat_template_overhead",
         tok.get("chat_template_overhead") or est.get("tokenizer.chat_template_overhead"))
    # the token-delta vector from P02 is the sharpest tokenizer signature
    for row in _first_table(by_probe.get("P02.tokenizer.fingerprint"), "deltas"):
        name, delta = row.get("probe"), row.get("delta")
        if name is not None and _is_number(delta):
            _num(fp, f"tok.delta.{name}", delta)


def _add_context(fp: dict[str, Any], shared: dict, est: dict) -> None:
    ctx = shared.get("context") or {}
    _num(fp, "context.max_input_tokens",
         ctx.get("max_input_tokens") or est.get("context.max_input_tokens"))
    _num(fp, "context.max_output_tokens",
         ctx.get("max_output_tokens") or est.get("context.max_output_tokens"))
    _tag(fp, "context.truncation_policy", ctx.get("truncation_policy"))
    rec = shared.get("context_recall") or {}
    _num(fp, "context.effective_recall_tokens",
         rec.get("effective_recall_tokens") or est.get("context.effective_recall_tokens"))
    if rec.get("sliding_window") is not None:
        _num(fp, "context.sliding_window", rec.get("sliding_window"))
    _num(fp, "context.recall_cliff", bool(rec.get("cliff")))


def _add_position(fp: dict[str, Any], by_probe: dict) -> None:
    grid = _first_table(by_probe.get("P06.positional.needle_recall"), "grid")
    if not grid:
        return
    overall = _rows_mean(grid, "recall")
    _num(fp, "pos.recall_overall", overall)
    for frac in (0.0, 0.5, 0.9):
        sub = [r for r in grid if r.get("position") == frac]
        _num(fp, f"pos.recall_at_{frac:g}", _rows_mean(sub, "recall"))


def _polyfit(x: list[float], y: list[float], deg: int) -> list[float] | None:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if xa.size <= deg or np.ptp(xa) <= 0:
        return None
    try:
        return [float(c) for c in np.polyfit(xa, ya, deg)]
    except Exception:  # noqa: BLE001
        return None


def _add_latency(fp: dict[str, Any], by_probe: dict, shared: dict, est: dict) -> None:
    _num(fp, "timing.rtt_s", shared.get("rtt_s"))
    p7 = by_probe.get("P07.timing.scaling")
    ser = (p7.series or {}) if p7 else {}
    pre = ser.get("prefill") or {}
    if pre.get("x") and pre.get("y"):
        # scale prompt length to kilotokens so coefficients are O(1) and comparable.
        x = [v / 1000.0 for v in pre["x"]]
        y = list(pre["y"])
        q = _polyfit(x, y, 2)
        if q:
            _num(fp, "lat.prefill_quad_ms_per_ktok2", q[0])
            _num(fp, "lat.prefill_lin_ms_per_ktok", q[1])
        lin = _polyfit(x, y, 1)
        if lin:
            _num(fp, "lat.prefill_slope_ms_per_ktok", lin[0])
        if min(y) > 0:
            _num(fp, "lat.prefill_range_ratio", max(y) / min(y))
    itl = ser.get("itl") or {}
    if itl.get("x") and itl.get("y"):
        x = [v / 1000.0 for v in itl["x"]]
        y = list(itl["y"])
        lin = _polyfit(x, y, 1)
        if lin:
            _num(fp, "lat.itl_slope_ms_per_ktok", lin[0])
            _num(fp, "lat.itl_intercept_ms", lin[1])
        _num(fp, "lat.itl_median_ms", float(np.median(y)))
    # canonical throughput estimates (if measured)
    for q in ("timing.prefill_tokens_per_s", "timing.decode_tokens_per_s",
              "timing.itl_ms", "timing.kv_growth_us_per_token"):
        _num(fp, q, est.get(q))


def _add_output_logprob(fp: dict[str, Any], by_probe: dict, shared: dict) -> None:
    summ = _first_table(by_probe.get("P14.alignment.base_vs_instruct"), "summary")
    if summ:
        _num(fp, "out.mean_topk_entropy_nats", summ[0].get("mean_topk_entropy_nats"))
    det = shared.get("determinism") or {}
    _num(fp, "out.logprob_jitter_std", det.get("logprob_jitter_std"))
    _num(fp, "out.nondeterministic", bool(det.get("nondeterministic")))
    _tag(fp, "out.determinism_flavour", det.get("flavour"))
    fid = _first_table(by_probe.get("P15.sampling.temperature_fidelity"), "summary")
    if fid:
        _num(fp, "out.sampler_tv_distance", fid[0].get("tv_distance"))
        _num(fp, "out.temperature_fidelity", fid[0].get("temperature_fidelity"))
        if fid[0].get("batch_invariant") is not None:
            _num(fp, "out.batch_invariant", bool(fid[0]["batch_invariant"]))


def _add_sampling(fp: dict[str, Any], by_probe: dict, shared: dict) -> None:
    _num(fp, "sampling.temperature_fidelity", shared.get("temperature_fidelity"))
    for row in _first_table(by_probe.get("P15.sampling.penalties"), "sampling_checks"):
        chk = row.get("check")
        if chk == "temperature_0_determinism" and row.get("result") is not None:
            _num(fp, "sampling.t0_deterministic", bool(row["result"]))
        elif chk == "top_p_truncation":
            _tag(fp, "sampling.top_p_truncation", row.get("result"))
            _num(fp, "sampling.distinct_sampled", row.get("distinct_sampled"))
        elif chk == "repetition_penalty":
            if row.get("result") is not None:
                _num(fp, "sampling.repetition_penalty", bool(row["result"]))
            _num(fp, "sampling.recurrence_logprob_slope", row.get("recurrence_logprob_slope"))
    spec = shared.get("speculative") or {}
    _num(fp, "sampling.spec_speedup", spec.get("speedup"))
    _num(fp, "sampling.spec_detected", bool(spec.get("detected")))
    cache = shared.get("prompt_cache") or {}
    _num(fp, "serving.cache_speedup", cache.get("speedup"))
    _num(fp, "serving.cache_detected", bool(cache.get("detected")))
    if cache.get("cache_block_tokens"):
        _num(fp, "serving.cache_block_tokens", cache.get("cache_block_tokens"))
    _num(fp, "serving.latency_modes", shared.get("latency_modes"))


def _add_refusal_safety(fp: dict[str, Any], by_probe: dict, shared: dict) -> None:
    ref = _first_table(by_probe.get("P14.alignment.refusal"), "refusal")
    rates = {r.get("class"): r for r in ref}
    if "unsafe" in rates:
        _num(fp, "safety.unsafe_refusal_rate", rates["unsafe"].get("rate"))
    if "benign" in rates:
        _num(fp, "safety.benign_refusal_rate", rates["benign"].get("rate"))
    if "unsafe" in rates and "benign" in rates:
        gap = float(rates["unsafe"].get("rate") or 0.0) - float(rates["benign"].get("rate") or 0.0)
        _num(fp, "safety.refusal_gap", max(0.0, gap))
    _num(fp, "safety.refusal_rate_borderline", shared.get("refusal_rate_borderline"))
    saf = _first_table(by_probe.get("P16.topology.safety_middleware"), "safety")
    for row in saf:
        if row.get("class") == "borderline":
            _num(fp, "safety.borderline_refused_rate", row.get("rate"))
    # entropy-collapse ("pick a random number")
    rn = _first_table(by_probe.get("P14.alignment.entropy_collapse"), "random_number")
    if rn:
        _num(fp, "safety.random_entropy_norm", rn[0].get("entropy_normalised"))
        _num(fp, "safety.random_modal_share", rn[0].get("modal_share"))


def _add_formatting(fp: dict[str, Any], by_probe: dict) -> None:
    summ = _first_table(by_probe.get("P14.alignment.base_vs_instruct"), "summary")
    if summ:
        _num(fp, "fmt.adherence_rate", summ[0].get("adherence_rate"))
        _num(fp, "fmt.median_answer_words", summ[0].get("median_answer_words"))
    # metamorphic robustness / per-transform drift vector
    labels = None
    vec = None
    p17 = by_probe.get("P17.metamorphic.invariance")
    if p17 and p17.shared:
        vec = p17.shared.get("fingerprint_metamorphic")
        labels = p17.shared.get("fingerprint_metamorphic_labels")
        _num(fp, "fmt.metamorphic_robustness", p17.shared.get("metamorphic_robustness"))
    if isinstance(vec, (list, tuple)):
        for i, v in enumerate(vec):
            name = labels[i] if labels and i < len(labels) else str(i)
            _num(fp, f"fmt.metamorphic.{name}", v)


def _add_behavior(fp: dict[str, Any], by_probe: dict, shared: dict) -> None:
    # numeric / tool-use behaviour: exact hard-arithmetic rate
    arith = _first_table(by_probe.get("P16.topology.tools"), "arithmetic")
    if arith:
        exact = [1.0 if r.get("exact") else 0.0 for r in arith if r.get("ok", True) is not False]
        if exact:
            _num(fp, "behav.arithmetic_exact_rate", float(np.mean(exact)))
    # retrieval / fresh-fact answering
    retr = _first_table(by_probe.get("P16.topology.retrieval"), "retrieval")
    if retr:
        spec = [1.0 if r.get("specific_answer") else 0.0 for r in retr if r.get("ok", True) is not False]
        if spec:
            _num(fp, "behav.fresh_fact_rate", float(np.mean(spec)))
    # reasoning / hidden compute
    reas = shared.get("reasoning") or {}
    _num(fp, "behav.hidden_reasoning", bool(reas.get("hidden")))
    _num(fp, "behav.hidden_tokens_ratio", reas.get("hidden_tokens_ratio"))
    _num(fp, "behav.max_hop_depth", reas.get("max_hop_depth"))
    # knowledge scale / cutoff
    kn = shared.get("knowledge") or {}
    _num(fp, "behav.knowledge_rarity_50", kn.get("rarity_50"))
    _num(fp, "behav.knowledge_total_params_b", kn.get("total_params_b"))
    if kn.get("cutoff") is not None:
        _num(fp, "behav.knowledge_cutoff_year", kn.get("cutoff"))


def _add_errors(fp: dict[str, Any], shared: dict, est: dict) -> None:
    ctx = shared.get("context") or {}
    _tag(fp, "error.truncation_policy", ctx.get("truncation_policy"))
    _num(fp, "error.system_fingerprint_drift", bool(shared.get("system_fingerprint_drift")))
    sig = shared.get("serving_signals")
    if isinstance(sig, (list, tuple)):
        for s in sig:
            _tag(fp, f"serving.signal.{s}", s)


def _add_calibration(fp: dict[str, Any], by_probe: dict, shared: dict, est: dict) -> None:
    # decision models: probability quantization grid
    if est.get("decision.probability_resolution") is not None:
        _num(fp, "calib.probability_resolution", est.get("decision.probability_resolution"))
    # generative: sampler fidelity is a coarse calibration proxy
    if shared.get("temperature_fidelity") is not None:
        _num(fp, "calib.temperature_fidelity", shared.get("temperature_fidelity"))
    # decision choice-set mass redistribution / option-processing tables
    cs = _first_table(by_probe.get("P18.decision.choice_set"), "choice_set")
    if len(cs) >= 2:
        try:
            _num(fp, "calib.choice_mass_shift",
                 float(cs[0].get("mass_on_pair", 0.0)) - float(cs[-1].get("mass_on_pair", 0.0)))
        except (TypeError, ValueError):
            pass


def _add_estimates(fp: dict[str, Any], run: Any) -> None:
    """Fold every canonical quantity: numeric -> feature, categorical -> tag."""
    for e in getattr(run, "estimates", []) or []:
        v = getattr(e, "value", None)
        if v is None:
            continue
        key = f"est.{e.quantity}"
        if _is_number(v):
            _num(fp, key, v)
        elif isinstance(v, bool):
            _num(fp, key, v)
        else:
            _tag(fp, key, v)


# properties whose most-likely value is a stable behavioural tag
_POSTERIOR_PROPS = (
    "arch.family", "arch.sparsity", "attn.long_range", "attn.kv_sharing",
    "pos.encoding", "pos.long_context_ext", "out.logit_precision", "output.head",
    "modal.input", "system.topology", "system.safety_middleware", "system.retrieval",
    "system.tool_orchestration", "post.instruction_tuning", "post.preference_optimization",
    "post.distillation_signal", "decision.question_coupling", "decision.option_processing",
    "decision.scoring_mechanism", "tokenizer.family",
)


def _add_posterior(fp: dict[str, Any], run: Any) -> None:
    post = getattr(run, "posterior", {}) or {}
    for prop in _POSTERIOR_PROPS:
        dist = post.get(prop)
        if not isinstance(dist, dict) or not dist:
            continue
        top = max(dist, key=dist.get)
        conf = float(dist[top])
        # store the confidence as a numeric feature and, when reasonably sharp, a tag
        _num(fp, f"belief.{prop}.conf", conf)
        if conf >= 0.4:
            _tag(fp, f"belief.{prop}", top)


def _add_identity(fp: dict[str, Any], shared: dict) -> None:
    ident = shared.get("fingerprint_identity") or {}
    _num(fp, "identity.brand_rate", ident.get("brand_rate"))
    _tag(fp, "identity.family", ident.get("family") or shared.get("self_identified_family"))
    _tag(fp, "identity.model_reported", shared.get("model_reported"))
    _num(fp, "identity.can_generate", bool(shared.get("can_generate"))
         if shared.get("can_generate") is not None else None)
    _tag(fp, "identity.modal_input", shared.get("modal_input"))
    _tag(fp, "identity.logit_dtype", shared.get("logit_dtype"))


# ============================================================================ public API
def fingerprint(run: Any) -> dict[str, Any]:
    """Assemble a behavioural fingerprint ``dict`` from a :class:`RunResult`.

    Returns a flat mapping of feature name -> value where numeric features are floats and
    categorical features are strings (tags).  Keys prefixed with ``_`` are provenance meta
    and are ignored by :func:`fingerprint_distance`.
    """
    fp: dict[str, Any] = {}
    shared = _merge_shared(run)
    by_probe = _by_probe(run)
    est = _estimates_map(run)

    # provenance (ignored by distance)
    fp["_run_id"] = getattr(run, "run_id", "")
    caps = getattr(run, "caps", None)
    fp["_kind"] = getattr(caps, "kind", None) if caps is not None else None
    fp["_target"] = (getattr(run, "target", {}) or {}).get("name")
    fp["_n_probes"] = len(getattr(run, "probe_results", []) or [])

    _add_tokenizer(fp, shared, by_probe, est)
    _add_context(fp, shared, est)
    _add_position(fp, by_probe)
    _add_latency(fp, by_probe, shared, est)
    _add_output_logprob(fp, by_probe, shared)
    _add_sampling(fp, by_probe, shared)
    _add_refusal_safety(fp, by_probe, shared)
    _add_formatting(fp, by_probe)
    _add_behavior(fp, by_probe, shared)
    _add_errors(fp, shared, est)
    _add_calibration(fp, by_probe, shared, est)
    _add_estimates(fp, run)
    _add_posterior(fp, run)
    _add_identity(fp, shared)
    return fp


def numeric_features(fp: dict[str, Any]) -> dict[str, float]:
    """The numeric (float) sub-vector, excluding meta (``_``-prefixed) keys."""
    out: dict[str, float] = {}
    for k, v in fp.items():
        if k.startswith("_"):
            continue
        if _is_number(v) and _finite(v):
            out[k] = float(v)
    return out


def tag_set(fp: dict[str, Any]) -> set[str]:
    """The categorical tags as ``key=value`` strings, excluding meta keys."""
    out: set[str] = set()
    for k, v in fp.items():
        if k.startswith("_"):
            continue
        if isinstance(v, str):
            out.add(f"{k}={v}")
    return out


def _signed_log(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def fingerprint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Scale-aware distance in ``[0, 1]`` between two fingerprints.

    Numeric part: over the numeric keys present in *both* fingerprints, blend
    (i) a symmetric relative L1 ``mean |x-y| / (|x|+|y|)`` (scale-free per key) and
    (ii) a cosine distance on signed-log-scaled values (captures overall shape); then add a
    key-coverage term (Jaccard over which numeric keys each side has).  Categorical part:
    Jaccard distance over the tag sets.  The two parts are averaged with weights 0.6/0.4,
    dropping whichever part has no data.  Identical fingerprints score ``0.0``.
    """
    na, nb = numeric_features(a), numeric_features(b)
    ta, tb = tag_set(a), tag_set(b)

    components: list[tuple[float, float]] = []   # (value, weight)

    # ---- numeric ---------------------------------------------------------
    shared_keys = sorted(set(na) & set(nb))
    all_keys = set(na) | set(nb)
    if all_keys:
        if shared_keys:
            rels, ua, ub = [], [], []
            for k in shared_keys:
                x, y = na[k], nb[k]
                denom = abs(x) + abs(y)
                rels.append(abs(x - y) / denom if denom > _EPS else 0.0)
                ua.append(_signed_log(x))
                ub.append(_signed_log(y))
            rel_l1 = float(np.mean(rels))
            va, vb = np.asarray(ua), np.asarray(ub)
            nrm = float(np.linalg.norm(va) * np.linalg.norm(vb))
            if nrm > _EPS:
                cos = float(np.dot(va, vb) / nrm)
                cos_dist = (1.0 - max(-1.0, min(1.0, cos))) / 2.0
            else:
                cos_dist = 0.0
        else:
            rel_l1, cos_dist = 1.0, 1.0
        coverage = 1.0 - len(shared_keys) / len(all_keys)
        numeric_dist = 0.55 * rel_l1 + 0.25 * cos_dist + 0.20 * coverage
        components.append((numeric_dist, 0.6))

    # ---- categorical -----------------------------------------------------
    if ta or tb:
        inter = len(ta & tb)
        union = len(ta | tb)
        tag_dist = 1.0 - (inter / union if union else 1.0)
        components.append((tag_dist, 0.4))

    if not components:
        return 0.0
    wsum = sum(w for _, w in components)
    return float(sum(v * w for v, w in components) / wsum)
