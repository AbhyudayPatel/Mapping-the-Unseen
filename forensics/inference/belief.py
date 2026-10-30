"""Belief: the running posterior over the discrete hypothesis space.

A :class:`Belief` is a bag of independent categorical distributions, one per
hypothesis property in :mod:`forensics.inference.hypotheses`.  It starts at the
priors and is refined by :class:`~forensics.probes.base.Evidence` — each piece of
evidence is a likelihood ``P(observation | value)`` over one property that is
applied as a *tempered* Bayesian update ``p *= L**strength`` followed by
renormalisation (see ``research/techniques_behavioral.md`` §6.2).

Design points
-------------
* **Factorised.** Properties are updated independently; logical links between them
  are handled *softly* by :func:`apply_coupling`, applied by the engine after each
  update round (never strong enough to override real evidence).
* **Floors.** Every value keeps a tiny floor (:data:`MIN_PROB`) so one noisy probe
  can never zero a hypothesis — a later probe can always resurrect it.
* **Dynamic properties.** Evidence may name a property that is not in the static
  table (e.g. ``tokenizer.family``, whose values are discovered at runtime).  Such
  a property is created on first sight from the evidence's own value set with a
  uniform prior; new values seen later are added at the floor.
* **Ledger.** Every applied update is recorded per property (the evidence summary
  plus the resulting distribution) so the report can show the inference trail.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from forensics.inference.hypotheses import (
    DYNAMIC_PROPERTIES, PROPERTIES, Property, normalized_priors,
)
from forensics.probes.base import Evidence

# Numeric floors: no posterior value or likelihood is ever exactly zero.
MIN_PROB = 1e-6
LIK_FLOOR = 1e-6

# Soft-coupling default strength (bounded multiplicative nudge, see apply_coupling).
DEFAULT_COUPLING_STRENGTH = 0.12


@dataclass
class LedgerEntry:
    """One applied update, kept per property for the report / audit trail."""
    probe_id: str
    summary: str
    level: str
    strength: float
    likelihoods: dict[str, float]
    posterior_after: dict[str, float]
    kind: str = "evidence"          # "evidence" | "coupling"


def _entropy_bits(dist: dict[str, float]) -> float:
    """Shannon entropy (bits) of a categorical distribution."""
    h = 0.0
    for p in dist.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def _normalize_floor(dist: dict[str, float]) -> None:
    """In-place: floor every value at MIN_PROB, then renormalise to sum 1."""
    for k, v in dist.items():
        if not math.isfinite(v) or v < MIN_PROB:
            dist[k] = MIN_PROB
    s = sum(dist.values())
    if s <= 0:
        u = 1.0 / len(dist)
        for k in dist:
            dist[k] = u
        return
    for k in dist:
        dist[k] /= s


class Belief:
    """Posterior over the hypothesis properties, updated by tempered evidence."""

    def __init__(self, properties: dict[str, Property] | None = None,
                 kind: str | None = None) -> None:
        """Build a belief from ``properties`` (defaults to the global table).

        ``kind`` (``"generative"`` / ``"decision"`` / ``"embedding"``) keeps only
        the properties that apply to that endpoint kind.  ``None`` or
        ``"unknown"`` keeps everything.
        """
        props = properties or PROPERTIES
        self.kind = kind
        self.prior: dict[str, dict[str, float]] = {}
        self.posterior: dict[str, dict[str, float]] = {}
        self.ledger: dict[str, list[LedgerEntry]] = {}
        self.coupling_log: list[str] = []
        self._dynamic: set[str] = set()

        for key, prop in props.items():
            if kind and kind != "unknown" and kind not in prop.kinds:
                continue
            dist = normalized_priors(prop)
            self.prior[key] = dict(dist)
            self.posterior[key] = dict(dist)
            self.ledger[key] = []

    # ------------------------------------------------------------------ dynamic
    def _ensure_property(self, key: str, values: list[str]) -> None:
        """Create a dynamic property (uniform prior over ``values``) on first sight."""
        if key in self.posterior:
            # add any brand-new values at the floor so later evidence can lift them
            dist = self.posterior[key]
            added = False
            for v in values:
                if v not in dist:
                    dist[v] = MIN_PROB
                    self.prior[key][v] = MIN_PROB
                    added = True
            if added:
                _normalize_floor(dist)
                _normalize_floor(self.prior[key])
            return
        vals = list(dict.fromkeys(values)) or ["unknown"]
        u = 1.0 / len(vals)
        self.prior[key] = {v: u for v in vals}
        self.posterior[key] = {v: u for v in vals}
        self.ledger[key] = []
        self._dynamic.add(key)

    # ------------------------------------------------------------------ update
    def update(self, evidence: Evidence) -> None:
        """Apply one tempered Bayesian update from ``evidence`` to its property.

        For values the evidence does not name, the likelihood is the mean of the
        listed likelihoods (a neutral, non-informative factor), matching the
        :class:`~forensics.probes.base.Evidence` contract.
        """
        prop = evidence.property
        lk_raw = {str(k): float(v) for k, v in (evidence.likelihoods or {}).items()}
        if not lk_raw:
            return
        # create / grow the property as needed
        if prop not in self.posterior:
            self._ensure_property(prop, list(lk_raw))
        else:
            self._ensure_property(prop, list(lk_raw))

        lk = {k: max(v, LIK_FLOOR) for k, v in lk_raw.items()}
        neutral = sum(lk.values()) / len(lk)
        strength = float(evidence.strength)
        strength = 0.0 if strength < 0 else strength

        dist = self.posterior[prop]
        for val in dist:
            L = lk.get(val, neutral)
            L = max(L, LIK_FLOOR)
            dist[val] *= L ** strength
        _normalize_floor(dist)

        self.ledger.setdefault(prop, []).append(LedgerEntry(
            probe_id=evidence.probe_id or "",
            summary=evidence.summary,
            level=str(evidence.level),
            strength=strength,
            likelihoods=lk_raw,
            posterior_after=dict(dist),
            kind="evidence",
        ))

    def apply_all(self, evidence: list[Evidence]) -> None:
        """Apply a batch of evidence in order."""
        for ev in evidence:
            self.update(ev)

    def apply_all_grouped(self, evidence: list[Evidence]) -> None:
        """Group-aware batch update: evidence sharing an ``experiment_group`` (or probe
        series) is combined ONCE per group instead of multiplying every factor
        independently.  Within a group, log-likelihoods are averaged (equivalent to
        assigning the group a single tempered update of unit order), so a long probe
        series cannot dominate the posterior through sheer repetition.  This keeps the
        ranker conservative and measurably better calibrated on ground truth; groups
        remain annotated on the ledger for audit."""
        groups: dict[tuple[str, str], list[Evidence]] = {}
        for ev in evidence:
            gkey = (ev.property, ev.experiment_group or ev.probe_id or "misc")
            groups.setdefault(gkey, []).append(ev)
        for (prop, gname), evs in groups.items():
            if len(evs) == 1:
                self.update(evs[0])
                continue
            # average log-likelihoods across the group, keep max strength
            keys: set[str] = set()
            for ev in evs:
                keys.update(ev.likelihoods)
            if not keys:
                continue
            logacc = {k: 0.0 for k in keys}
            for ev in evs:
                lk = ev.likelihoods or {}
                neutral = sum(lk.values()) / max(1, len(lk))
                for k in keys:
                    logacc[k] += math.log(max(lk.get(k, neutral), LIK_FLOOR))
            n = float(len(evs))
            merged = {k: math.exp(v / n) for k, v in logacc.items()}
            strength = max(float(e.strength) for e in evs)
            self.update(Evidence(
                property=prop, likelihoods=merged,
                summary=f"grouped update over {len(evs)} linked observations ({gname})",
                level=evs[0].level, strength=min(1.0, strength),
                probe_id=gname, experiment_group=gname))

    # ------------------------------------------------------------------ queries
    def entropy(self, prop: str) -> float:
        """Entropy (bits) of a property's current posterior (0 if unknown)."""
        dist = self.posterior.get(prop)
        return _entropy_bits(dist) if dist else 0.0

    def total_entropy(self) -> float:
        """Sum of per-property posterior entropies (bits)."""
        return sum(_entropy_bits(d) for d in self.posterior.values())

    def top(self, prop: str) -> tuple[str, float] | None:
        """The most probable ``(value, probability)`` for a property, or None."""
        dist = self.posterior.get(prop)
        if not dist:
            return None
        v = max(dist, key=lambda k: dist[k])
        return v, dist[v]

    def as_dict(self) -> dict[str, dict[str, float]]:
        """A deep copy of the current posterior."""
        return {p: dict(d) for p, d in self.posterior.items()}

    def prior_dict(self) -> dict[str, dict[str, float]]:
        """A deep copy of the prior."""
        return {p: dict(d) for p, d in self.prior.items()}

    def properties(self) -> list[str]:
        return list(self.posterior.keys())


# --------------------------------------------------------------------------- coupling
#: Undirected soft-consistency links between (property, value) pairs.  Each link
#: is applied in both directions by :func:`apply_coupling`.  These encode weak
#: architectural logic — believing one end gently raises the other, and doubting
#: it gently lowers the other — without ever asserting a hard constraint.
COUPLINGS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # a constant-state family goes hand in hand with linear-recurrent long-range cost
    (("arch.family", "ssm_or_linear"), ("attn.long_range", "linear_recurrent")),
    # a causal-LM objective goes hand in hand with a decoder-only backbone
    (("train.objective", "causal_lm"), ("arch.family", "decoder_only")),
    # classification output heads go hand in hand with a contrastive/classifier objective
    (("output.head", "classification_heads"),
     ("train.objective", "contrastive_or_classifier")),
]


def _nudge(belief: Belief, src_prop: str, src_val: str,
           dst_prop: str, dst_val: str, strength: float) -> bool:
    """Gently push ``dst_val`` toward/away from consistency with the source belief.

    The multiplicative factor is bounded to ``[1-strength, 1+strength]`` and is
    centred at 1 when the source is undecided (``P(src_val)=0.5``); so a single
    coupling pass can never flip a confident posterior — it only breaks ties and
    nudges weakly-held values.
    """
    src = belief.posterior.get(src_prop)
    dst = belief.posterior.get(dst_prop)
    if not src or not dst or src_val not in src or dst_val not in dst:
        return False
    s = src[src_val]
    factor = 1.0 + strength * (2.0 * s - 1.0)   # s in [0,1] -> factor in [1-k, 1+k]
    if factor <= 0:
        factor = MIN_PROB
    before = dst[dst_val]
    dst[dst_val] = before * factor
    _normalize_floor(dst)
    return True


def apply_coupling(belief: Belief, strength: float = DEFAULT_COUPLING_STRENGTH) -> None:
    """Apply all soft-consistency couplings once (both directions per link).

    Intended to be called by the engine after each evidence round.  It mutates
    ``belief.posterior`` in place with small, bounded nudges and appends a short
    note to ``belief.coupling_log`` for the report.
    """
    strength = max(0.0, min(0.5, float(strength)))
    if strength == 0.0:
        return
    for (pa, va), (pb, vb) in COUPLINGS:
        moved_ab = _nudge(belief, pa, va, pb, vb, strength)
        moved_ba = _nudge(belief, pb, vb, pa, va, strength)
        if moved_ab or moved_ba:
            belief.coupling_log.append(
                f"coupled ({pa}={va}) <-> ({pb}={vb}) @strength={strength:.2f}")
