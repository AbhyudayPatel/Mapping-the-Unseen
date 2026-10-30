"""Experiment selection by expected SURROGATE information gain per request.

Given the current :class:`~forensics.inference.belief.Belief` (the prior over the
hypothesis space) and a set of candidate probes, this module scores each probe by
the information it is expected to reveal and ranks them by information-per-cost.
See ``research/techniques_behavioral.md`` §6.3.

For a probe :math:`e` that informs properties :math:`\\{p\\}` with per-value outcome
likelihoods :math:`P(y\\mid v)`, the expected information gain is

.. math::
    \\mathrm{EIG}(e) = \\sum_p \\big[ H(p) - \\mathbb{E}_y\\, H(p\\mid y) \\big]

where the weights are normalized surrogate support (the legacy ``posterior`` field),
not calibrated production-endpoint probabilities, and

.. math::
    p(y) = \\sum_v \\pi(v) P(y\\mid v), \\quad
    \\pi(v\\mid y) = \\pi(v) P(y\\mid v) / p(y).

Each term is the mutual information :math:`I(p; Y_e)\\ge 0` between the property and
the discretised outcome under the declared SURROGATE model, not the production
endpoint's true observation law. When a probe supplies no explicit
:meth:`~forensics.probes.base.Probe.outcome_model`, a generic two-outcome
(``signal`` / ``nosignal``) model is synthesised from the probe's declared
``informs`` diagnosticities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from forensics.inference.belief import Belief

# Below this EIG (bits) a probe is treated as no longer worth running.
DEFAULT_MIN_GAIN_BITS = 0.02
# Floor for synthesised outcome probabilities (keeps entropies finite).
_P_FLOOR = 1e-3


@dataclass
class RankedProbe:
    """One scored candidate: EIG (bits), estimated cost, and EIG/cost score."""
    probe: Any
    eig_bits: float
    cost: int
    score: float           # eig_bits / cost
    applicable: bool = True
    reason: str = ""


def _entropy_bits(dist: dict[str, float]) -> float:
    h = 0.0
    for p in dist.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def _normalized(dist: dict[str, float]) -> dict[str, float]:
    s = sum(dist.values())
    if s <= 0:
        n = len(dist) or 1
        return {k: 1.0 / n for k in dist}
    return {k: v / s for k, v in dist.items()}


def _generic_outcome_model(prior: dict[str, float], d: float
                           ) -> dict[str, dict[str, float]]:
    """Two-outcome model from a diagnosticity ``d`` in [0,1].

    The value the probe would most likely *favour* (taken here as the current
    leading hypothesis, since a myopic experiment tests the front-runner) emits
    ``signal`` with probability ``0.5 + 0.5 d``; every other value emits it with
    ``0.5 - 0.5 d``.  As the leader's mass concentrates the property's entropy
    (and hence the EIG) shrinks, giving natural diminishing returns.
    """
    d = max(0.0, min(1.0, float(d)))
    if not prior:
        return {}
    favored = max(prior, key=lambda k: prior[k])
    p_fav = min(1.0 - _P_FLOOR, max(_P_FLOOR, 0.5 + 0.5 * d))
    p_oth = min(1.0 - _P_FLOOR, max(_P_FLOOR, 0.5 - 0.5 * d))
    out: dict[str, dict[str, float]] = {}
    for v in prior:
        ps = p_fav if v == favored else p_oth
        out[v] = {"signal": ps, "nosignal": 1.0 - ps}
    return out


def _property_eig(prior: dict[str, float],
                  model: dict[str, dict[str, float]]) -> float:
    """Mutual information (bits) between one property and a probe outcome."""
    if not prior or not model:
        return 0.0
    pi = _normalized(prior)
    outcomes: list[str] = []
    seen: set[str] = set()
    for cond in model.values():
        for y in cond:
            if y not in seen:
                seen.add(y)
                outcomes.append(y)
    if len(outcomes) < 2 or len(pi) < 2:
        return 0.0
    neutral = {y: 1.0 / len(outcomes) for y in outcomes}

    def cond_for(v: str) -> dict[str, float]:
        c = model.get(v)
        if not c:
            return neutral
        cc = {y: max(_P_FLOOR, float(c.get(y, 0.0))) for y in outcomes}
        return _normalized(cc)

    # marginal p(y) and prior property entropy
    py = {y: 0.0 for y in outcomes}
    for v, pv in pi.items():
        c = cond_for(v)
        for y in outcomes:
            py[y] += pv * c[y]
    h_prior = _entropy_bits(pi)

    # expected posterior entropy after seeing the outcome
    exp_post = 0.0
    for y in outcomes:
        if py[y] <= 0.0:
            continue
        post = {v: pi[v] * cond_for(v)[y] / py[y] for v in pi}
        exp_post += py[y] * _entropy_bits(post)

    return max(0.0, h_prior - exp_post)


def expected_information_gain(probe: Any, belief: Belief, ctx: Any) -> float:
    """Total expected information gain (bits) of running ``probe`` given ``belief``.

    Sums the mutual information over every property the probe informs, using the
    probe's :meth:`~forensics.probes.base.Probe.outcome_model` when it provides
    one and a synthesised two-outcome model from ``informs`` otherwise.
    """
    informs: dict[str, float] = dict(getattr(probe, "informs", {}) or {})
    try:
        supplied = probe.outcome_model(ctx)
    except Exception:
        supplied = None
    supplied = supplied or {}

    keys = set(informs) | set(supplied)
    total = 0.0
    for prop in keys:
        prior = belief.posterior.get(prop)
        if not prior:
            continue  # property not (yet) tracked -> no measurable gain
        model = supplied.get(prop)
        if not model:
            model = _generic_outcome_model(prior, informs.get(prop, 0.0))
        total += _property_eig(prior, model)
    return total


def rank_probes(candidates: list[Any], belief: Belief, ctx: Any,
                include_inapplicable: bool = False) -> list[RankedProbe]:
    """Rank probes by EIG per estimated cost (descending).

    Inapplicable probes (per :meth:`~forensics.probes.base.Probe.applicable`) are
    skipped unless ``include_inapplicable`` is set (in which case they are returned
    with ``applicable=False`` and a zero score, for reporting/recommendations).
    """
    ranked: list[RankedProbe] = []
    for probe in candidates:
        try:
            ok, reason = probe.applicable(ctx)
        except Exception as e:  # noqa: BLE001
            ok, reason = False, f"applicable() error: {e!r}"
        if not ok and not include_inapplicable:
            continue
        cost = max(1, int(probe.estimated_cost(ctx)))
        eig = expected_information_gain(probe, belief, ctx) if ok else 0.0
        ranked.append(RankedProbe(
            probe=probe, eig_bits=eig, cost=cost,
            score=(eig / cost) if ok else 0.0,
            applicable=ok, reason=reason,
        ))
    ranked.sort(key=lambda r: (r.score, r.eig_bits), reverse=True)
    return ranked


def is_informative(eig_bits: float, threshold: float = DEFAULT_MIN_GAIN_BITS) -> bool:
    """True while a probe's EIG still clears the diminishing-returns threshold."""
    return eig_bits > threshold


def diminishing_returns(ranked: list[RankedProbe],
                        threshold: float = DEFAULT_MIN_GAIN_BITS) -> bool:
    """True when no remaining probe is expected to gain more than ``threshold`` bits."""
    return not any(is_informative(r.eig_bits, threshold) for r in ranked)
