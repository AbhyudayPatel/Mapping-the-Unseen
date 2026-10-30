"""Read-only accessors and formatters over a :class:`RunResult`.

Everything the report shows is derived here from run data (posterior, estimates,
evidence, identifiability). Nothing is hard-coded about any specific architecture:
these helpers just look up what a given run measured, with safe fallbacks for the
many fields that can legitimately be missing (e.g. ``hidden_size`` never recovered).
"""
from __future__ import annotations

import math
from typing import Any

from forensics.inference.hypotheses import PROPERTIES, QUANTITIES
from forensics.inference.run_types import RunResult
from forensics.probes.base import Estimate

# a sentinel used across the report when a quantity was not identified within budget
NOT_IDENTIFIED = "not identified within budget"


# --------------------------------------------------------------------------- posterior
def posterior_sorted(run: RunResult, prop: str) -> list[tuple[str, float]]:
    dist = run.posterior.get(prop) or {}
    return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)


def posterior_top(run: RunResult, prop: str) -> tuple[str | None, float]:
    s = posterior_sorted(run, prop)
    return (s[0][0], s[0][1]) if s else (None, 0.0)


def prior_prob(run: RunResult, prop: str, value: str) -> float:
    return float((run.prior.get(prop) or {}).get(value, 0.0))


def has_property(run: RunResult, prop: str) -> bool:
    return prop in run.posterior


# --------------------------------------------------------------------------- estimates
def estimates_for(run: RunResult, quantity: str) -> list[Estimate]:
    return [e for e in run.estimates if e.quantity == quantity]


def _est_rank(e: Estimate) -> tuple[int, int, int]:
    """Prefer a resolved value, then better identifiability, then a stronger level."""
    lvl = {"observed": 0, "inferred": 1, "estimated": 2, "speculative": 3}.get(e.level, 4)
    return (0 if e.value is not None else 1,
            int(getattr(e, "identifiability", 5) or 5), lvl)


def best_estimate(run: RunResult, quantity: str) -> Estimate | None:
    cands = estimates_for(run, quantity)
    if not cands:
        return None
    return sorted(cands, key=_est_rank)[0]


def est_value(run: RunResult, quantity: str) -> Any:
    e = best_estimate(run, quantity)
    return e.value if e is not None else None


# --------------------------------------------------------------------------- human labels
def value_label(prop: str, value: str | None) -> str:
    """Human-readable label for a property value, using hypotheses ``value_docs``."""
    if value is None:
        return NOT_IDENTIFIED
    p = PROPERTIES.get(prop)
    if p and value in p.value_docs:
        return f"{_titleish(value)} — {p.value_docs[value]}"
    return _titleish(value)


def value_short(value: str | None) -> str:
    return NOT_IDENTIFIED if value is None else _titleish(value)


def property_desc(prop: str) -> str:
    p = PROPERTIES.get(prop)
    return p.description if p else prop


def quantity_desc(quantity: str) -> str:
    q = QUANTITIES.get(quantity)
    return q[1] if q else quantity


def quantity_unit(quantity: str) -> str:
    q = QUANTITIES.get(quantity)
    return q[0] if q else ""


def _titleish(value: str) -> str:
    return str(value).replace("_", " ")


# --------------------------------------------------------------------------- formatting
def fmt_num(x: Any) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "∞" if xf > 0 else "-∞"
    ax = abs(xf)
    if ax != 0 and (ax >= 1e6 or ax < 1e-3):
        return f"{xf:.3g}"
    if ax >= 1000:
        return f"{int(round(xf)):,d}"
    if xf == int(xf):
        return f"{int(xf):d}"
    if ax >= 10:
        return f"{xf:.1f}"
    return f"{xf:.2f}"


def fmt_value(v: Any, unit: str = "") -> str:
    if v is None:
        return NOT_IDENTIFIED
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        s = fmt_num(v)
    else:
        s = str(v)
    return f"{s} {unit}".strip() if unit else s


def fmt_ci(ci: Any, value: Any = None) -> str:
    if not ci or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return ""
    lo, hi = ci
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return ""
    if math.isfinite(lo) and math.isfinite(hi):
        if lo == hi:
            return ""
        return f"{fmt_num(lo)} – {fmt_num(hi)}"
    if math.isfinite(lo):
        return f"≥ {fmt_num(lo)}"
    if math.isfinite(hi):
        return f"≤ {fmt_num(hi)}"
    return ""


def fmt_pct(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"


# --------------------------------------------------------------------------- multimodality
def detect_multimodal(run: RunResult) -> dict[str, Any]:
    """Decide, from the posterior, whether to draw vision/audio lanes.

    Returns a dict with the winning input/output modality, the vision-encoder style,
    and boolean flags ``image``/``audio``/``multimodal_out``. Text-only stays False.
    """
    in_val, in_p = posterior_top(run, "modal.input")
    out_val, out_p = posterior_top(run, "modal.output")
    ve_val, ve_p = posterior_top(run, "modal.vision_encoder")

    in_val = in_val or "text_only"
    out_val = out_val or "text_only"
    image = "image" in in_val
    audio = "audio" in in_val
    multimodal_out = out_val not in ("text_only", None)
    return {
        "input": in_val, "input_p": in_p,
        "output": out_val, "output_p": out_p,
        "vision_encoder": ve_val, "vision_encoder_p": ve_p,
        "image": image, "audio": audio,
        "multimodal_out": multimodal_out,
        "any": bool(image or audio or multimodal_out),
    }


# --------------------------------------------------------------------------- probe index
def ran_probe_ids(run: RunResult) -> list[str]:
    return [pr.probe_id for pr in run.probe_results]


def probe_result(run: RunResult, probe_id: str):
    for pr in run.probe_results:
        if pr.probe_id == probe_id:
            return pr
    return None


def evidence_for_property(run: RunResult, prop: str) -> list:
    return [ev for ev in run.evidence if ev.property == prop]


def evidence_for_probe(run: RunResult, probe_id: str) -> list:
    return [ev for ev in run.evidence if ev.probe_id == probe_id]


def identifiability_for(run: RunResult, key: str):
    for ie in run.identifiability:
        if ie.key == key:
            return ie
    return None


def experiment_for_probe(run: RunResult, probe_id: str):
    for st in run.experiment_log:
        if st.probe_id == probe_id:
            return st
    return None
