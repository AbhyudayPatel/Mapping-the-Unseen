"""Reference-family matching with honest, coverage-aware scores.

Shared by :mod:`forensics.inference.estimator` and the engine's fallback matcher.

Two failure modes this module exists to prevent (both seen on a live decision-model run,
where many families scored 1.000 "matched: arch_type, kv_sharing" although nothing had been
measured and ``attn.kv_sharing`` was still sitting at its prior):

1. **Prior-as-constraint.** A belief property is only used as a constraint when its top
   value is *evidence-backed*: its posterior moved away from the prior by >= 0.15 absolute
   (with at least one evidence item in the ledger), or its top posterior is >= 0.8 with at
   least one evidence item.  Coupling nudges alone never count.
2. **Score without coverage.** ``1/(1+distance)`` is 1.0 for *any* family that agrees on
   the one or two weak fields supplied.  The score here is
   ``match_quality * coverage`` where ``match_quality = 1/(1+distance)`` and ``coverage``
   grows with the summed weight of the constraints the family actually *matched*, saturating
   at about four strong constraints (``COVERAGE_SATURATION``).  Two weak categorical matches
   therefore score ~0.2, never 1.0.

At least one MEASURED quantity (hidden size, vocab size, context length or a matched
tokenizer family) is required; otherwise no family match is returned and the caller gets
the note ``insufficient measured constraints for family matching``.
"""
from __future__ import annotations

import math
from typing import Any

from forensics.inference.run_types import FamilyMatch

INSUFFICIENT_NOTE = "insufficient measured constraints for family matching"

#: evidence-backing thresholds for belief-derived constraints
MIN_PRIOR_SHIFT = 0.15
MIN_TOP_POSTERIOR = 0.8

#: per-constraint coverage weight (strong = directly measured architectural constants)
CONSTRAINT_WEIGHTS: dict[str, float] = {
    "hidden_size": 1.0,
    "vocab_size": 1.0,
    "tokenizer_family": 0.9,
    "context": 0.5,
    "moe": 0.35,
    "arch_type": 0.35,
    "kv_sharing": 0.35,
}
#: summed matched weight at which coverage saturates (~ four strong constraints)
COVERAGE_SATURATION = 3.5
#: constraint keys that count as MEASURED quantities (vs belief-derived categories)
MEASURED_KEYS = ("hidden_size", "vocab_size", "context", "tokenizer_family")

# arch.family (hypothesis value) -> DB arch_type
FAMILY_TO_ARCH: dict[str, str] = {
    "decoder_only": "decoder", "encoder_only": "encoder",
    "encoder_decoder": "encoder-decoder", "ssm_or_linear": "ssm",
    "hybrid_attn_ssm": "hybrid", "diffusion": "diffusion",
}


# --------------------------------------------------------------------------- belief checks
def _posterior(belief: Any, prop: str) -> dict[str, float] | None:
    post = getattr(belief, "posterior", None)
    if post is None and hasattr(belief, "as_dict"):
        try:
            post = belief.as_dict()
        except Exception:  # noqa: BLE001
            post = None
    if post is None and isinstance(belief, dict):
        post = belief
    d = post.get(prop) if isinstance(post, dict) else None
    return d if isinstance(d, dict) and d else None


def _prior(belief: Any, prop: str) -> dict[str, float] | None:
    prior = getattr(belief, "prior", None)
    if prior is None and hasattr(belief, "prior_dict"):
        try:
            prior = belief.prior_dict()
        except Exception:  # noqa: BLE001
            prior = None
    d = prior.get(prop) if isinstance(prior, dict) else None
    return d if isinstance(d, dict) and d else None


def _n_evidence(belief: Any, prop: str) -> int | None:
    """Number of real evidence updates in the ledger (None if the belief has no ledger)."""
    ledger = getattr(belief, "ledger", None)
    if not isinstance(ledger, dict):
        return None
    entries = ledger.get(prop) or []
    return sum(1 for e in entries if getattr(e, "kind", "evidence") == "evidence")


def evidence_backed(belief: Any, prop: str, min_shift: float = MIN_PRIOR_SHIFT,
                    min_top: float = MIN_TOP_POSTERIOR) -> tuple[str | None, float, str]:
    """Return ``(value, posterior, reason)`` if ``prop``'s top value is evidence-backed,
    else ``(None, posterior, reason)``.

    Rule: moved from its prior by >= ``min_shift`` absolute (and >= 1 evidence item), or
    top posterior >= ``min_top`` with >= 1 evidence item.  A belief that exposes neither a
    prior nor a ledger (an opaque posterior dict / stub) cannot be audited; there only the
    concentration rule (top >= ``min_top``) applies.
    """
    if belief is None:
        return None, 0.0, "no belief"
    post = _posterior(belief, prop)
    if not post:
        return None, 0.0, "property absent"
    top = max(post, key=lambda k: post[k])
    p = float(post[top])
    prior = _prior(belief, prop)
    n_ev = _n_evidence(belief, prop)
    if prior is None and n_ev is None:
        ok = p >= min_top
        return (top if ok else None), p, ("opaque belief, concentrated" if ok
                                          else "opaque belief, not concentrated")
    if not n_ev:
        return None, p, "no evidence in ledger (prior only)"
    shift = p - float((prior or {}).get(top, 0.0))
    if shift >= min_shift:
        return top, p, f"moved {shift:+.2f} from prior with {n_ev} evidence item(s)"
    if p >= min_top:
        return top, p, f"posterior {p:.2f} with {n_ev} evidence item(s)"
    return None, p, f"not evidence-backed (posterior {p:.2f}, shift {shift:+.2f})"


# --------------------------------------------------------------------------- measured input
def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _best_value(estimates: list[Any], quantity: str) -> Any:
    """Most recent non-speculative value for ``quantity`` (speculative values are not
    measurements and never become constraints)."""
    for e in reversed(list(estimates or [])):
        if getattr(e, "quantity", None) != quantity or getattr(e, "value", None) is None:
            continue
        if getattr(e, "level", "") == "speculative":
            continue
        return e.value
    return None


def resolve_tokenizer_family(name: Any, db: Any) -> str | None:
    """Map a registry tokenizer family (e.g. ``tiktoken:o200k_base``) onto a DB
    ``tokenizer_family`` value, or None if it is unknown/custom/not in the DB."""
    if not name:
        return None
    s = str(name).strip()
    low = s.lower()
    if low.startswith("unknown") or low in ("none", "custom", "n/a"):
        return None
    vals: list[str] = []
    for e in getattr(db, "_entries", []) or []:
        v = e.get("tokenizer_family")
        if v and v not in vals:
            vals.append(str(v))
    if s in vals:
        return s
    stem = low.split(":", 1)[-1]
    for suf in ("_base",):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    exact = [v for v in vals if v.lower() == stem]
    if exact:
        return exact[0]
    pref = [v for v in vals if v.lower().startswith(stem)]
    return pref[0] if pref else None


def measured_constraints(estimates: list[Any], db: Any) -> dict[str, Any]:
    """Constraint dict of MEASURED quantities only."""
    out: dict[str, Any] = {}
    d = _num(_best_value(estimates, "dims.hidden_size"))
    v = _num(_best_value(estimates, "dims.vocab_size"))
    c = _num(_best_value(estimates, "context.max_input_tokens"))
    if d:
        out["hidden_size"] = d
    if v:
        out["vocab_size"] = v
    if c:
        out["context"] = c
    tf = resolve_tokenizer_family(_best_value(estimates, "tokenizer.family"), db)
    if tf:
        out["tokenizer_family"] = tf
    return out


def belief_constraints(belief: Any) -> tuple[dict[str, Any], list[str]]:
    """Evidence-backed categorical constraints from the belief (+ an audit trail)."""
    out: dict[str, Any] = {}
    trail: list[str] = []
    fam, p, why = evidence_backed(belief, "arch.family")
    if fam and FAMILY_TO_ARCH.get(str(fam)):
        out["arch_type"] = FAMILY_TO_ARCH[str(fam)]
    trail.append(f"arch.family: {'used' if 'arch_type' in out else 'ignored'} ({why})")
    kv, p, why = evidence_backed(belief, "attn.kv_sharing")
    if kv in ("mha", "gqa_or_mqa", "mla"):
        out["kv_sharing"] = kv
    trail.append(f"attn.kv_sharing: {'used' if 'kv_sharing' in out else 'ignored'} ({why})")
    sp, p, why = evidence_backed(belief, "arch.sparsity")
    if sp in ("moe", "dense"):
        out["moe"] = sp == "moe"
    trail.append(f"arch.sparsity: {'used' if 'moe' in out else 'ignored'} ({why})")
    return out, trail


# --------------------------------------------------------------------------- scoring
def coverage(matched: list[str]) -> float:
    w = sum(CONSTRAINT_WEIGHTS.get(k, 0.35) for k in set(matched))
    return min(1.0, w / COVERAGE_SATURATION)


def score_ranked(ranked: list[tuple[dict[str, Any], float, list[str], list[str]]],
                 top_k: int = 6) -> list[FamilyMatch]:
    """Turn ``ArchDB.nearest`` output into coverage-aware FamilyMatch scores."""
    scored: list[tuple[float, float, FamilyMatch]] = []
    for entry, dist, matched, conflicts in ranked:
        quality = 1.0 / (1.0 + max(0.0, float(dist)))
        sc = quality * coverage(list(matched))
        scored.append((sc, float(dist), FamilyMatch(
            name=str(entry.get("name") or entry.get("family") or "?"),
            score=round(sc, 4), matched=list(matched), conflicts=list(conflicts))))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [fm for _, _, fm in scored[:top_k]]


def match_families(belief: Any, estimates: list[Any], db: Any = None, top_k: int = 6,
                   extra_constraints: dict[str, Any] | None = None
                   ) -> tuple[list[FamilyMatch], list[str]]:
    """Coverage-aware nearest reference families. Returns ``(matches, notes)``."""
    if db is None:
        from forensics.knowledge.archdb import ArchDB
        db = ArchDB.load()
    notes: list[str] = []
    measured = measured_constraints(estimates, db)
    if extra_constraints:
        measured.update({k: v for k, v in extra_constraints.items()
                         if k in MEASURED_KEYS and v})
    if not any(k in measured for k in MEASURED_KEYS):
        notes.append(INSUFFICIENT_NOTE + " (need hidden_size, vocab_size, context or a "
                     "matched tokenizer family; belief-only categories are not enough)")
        return [], notes
    cats, trail = belief_constraints(belief)
    constraints = {**cats, **measured}
    ranked = db.nearest(constraints, top_k=None)
    if not ranked:
        return [], notes
    out = score_ranked(ranked, top_k=top_k)
    notes.append("family match constraints: " + ", ".join(sorted(constraints)) +
                 " | " + "; ".join(trail))
    return out, notes
