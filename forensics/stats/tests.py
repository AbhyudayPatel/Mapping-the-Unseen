"""Hypothesis tests and effect sizes for black-box model forensics.

Everything here is pure ``numpy``/``scipy``, deterministic given a seed, and free of
global state: any function that draws random numbers takes an ``rng`` (a
``numpy.random.Generator``, an integer seed, or ``None``) and never touches the
global RNG.

Where these are used in the pipeline (see ``research/techniques_*.md``):

- ``bootstrap_ci`` — non-parametric CIs for every reported estimate (throughput,
  slopes, curvature) so the report can honour the "never a bare point estimate" rule.
- ``permutation_test`` — paired/unpaired A/B latency comparisons (e.g. spec-decoding
  content A/B, cache hit vs miss) without distributional assumptions.
- ``mannwhitney`` / ``ks_2samp`` — two-sample timing-distribution shifts; the one-sided
  KS test is the Gu et al. prompt-cache audit workhorse.
- ``welch_t`` — quick unequal-variance mean comparison of latency samples.
- ``benjamini_hochberg`` — FDR control across the many probes/conditions we test.
- ``cohen_d`` / ``cliffs_delta`` — standardized effect sizes to fill Evidence ``stats``.
- ``proportion_ci`` — Wilson interval for agreement/accuracy rates (MoE determinism,
  needle recall) where k successes out of n is the observable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as sps

__all__ = [
    "bootstrap_ci",
    "permutation_test",
    "mannwhitney",
    "ks_2samp",
    "welch_t",
    "benjamini_hochberg",
    "cohen_d",
    "cliffs_delta",
    "proportion_ci",
]

_EPS = 1e-300


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce ``rng`` (Generator | int seed | None) into a Generator (no global state)."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def bootstrap_ci(
    data,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    level: float = 0.9,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for an arbitrary statistic of a 1-D sample.

    Forensics use: attach an honest interval to any point estimate (median ITL,
    prefill throughput, regression slope) computed from a modest number of noisy
    timing/logit samples.

    Returns ``(lo, hi, point)`` where ``point = stat(data)`` and ``[lo, hi]`` is the
    two-sided ``level`` percentile interval of the bootstrap distribution.
    """
    rng = _as_rng(rng)
    data = np.asarray(data, dtype=float)
    n = data.size
    point = float(stat(data)) if n else float("nan")
    if n == 0:
        return (float("nan"), float("nan"), point)
    if n == 1:
        return (point, point, point)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = data[idx]
    try:
        boots = np.asarray(stat(resamples, axis=1), dtype=float)  # fast path (np.mean/median)
    except TypeError:
        boots = np.array([float(stat(resamples[b])) for b in range(n_boot)])
    alpha = 1.0 - level
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return (lo, hi, point)


def permutation_test(
    a,
    b,
    stat: str | Callable[[np.ndarray, np.ndarray], float] = "mean_diff",
    n: int = 10000,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Two-sided permutation (randomization) test for a difference between two samples.

    Forensics use: distribution-free significance for A/B timing contrasts where the
    latency distribution is heavy-tailed and non-normal (spec-decoding predictable vs
    random content, cache hit vs miss). The label-shuffling null needs no parametric
    assumption.

    ``stat`` is either ``"mean_diff"`` (mean(a) - mean(b), vectorised) or a callable
    ``stat(a, b) -> float``. Returns the two-sided p-value with the standard
    ``(count + 1) / (n + 1)`` add-one correction (never exactly 0).
    """
    rng = _as_rng(rng)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = a.size
    pooled = np.concatenate([a, b])
    total = pooled.size

    if stat == "mean_diff":
        obs = float(np.mean(a) - np.mean(b))
        # Vectorised: argsort of uniform noise gives independent permutations.
        perms = np.argsort(rng.random((n, total)), axis=1)
        shuffled = pooled[perms]
        diffs = shuffled[:, :na].mean(axis=1) - shuffled[:, na:].mean(axis=1)
        count = int(np.sum(np.abs(diffs) >= abs(obs) - 1e-12))
    elif callable(stat):
        obs = float(stat(a, b))
        count = 0
        for _ in range(n):
            perm = pooled[rng.permutation(total)]
            s = float(stat(perm[:na], perm[na:]))
            if abs(s) >= abs(obs) - 1e-12:
                count += 1
    else:
        raise ValueError("stat must be 'mean_diff' or a callable(a, b) -> float")

    return (count + 1) / (n + 1)


def mannwhitney(a, b) -> tuple[float, float]:
    """Mann-Whitney U (two-sided) rank-sum test.

    Forensics use: robust two-sample test on lower-quantile latencies where medians
    shift but variances differ; insensitive to the heavy right tail of network jitter.
    Returns ``(U, p)``.
    """
    res = sps.mannwhitneyu(np.asarray(a, float), np.asarray(b, float), alternative="two-sided")
    return (float(res.statistic), float(res.pvalue))


def ks_2samp(a, b) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test (two-sided).

    Forensics use: the prompt-cache audit (Gu et al.) compares hit vs miss TTFT
    distributions with a KS test; more generally detects any shape change between two
    timing/logit samples. Returns ``(D, p)`` (D = max CDF distance).
    """
    res = sps.ks_2samp(np.asarray(a, float), np.asarray(b, float))
    return (float(res.statistic), float(res.pvalue))


def welch_t(a, b) -> tuple[float, float]:
    """Welch's unequal-variance t-test (two-sided).

    Forensics use: fast parametric mean comparison of two latency conditions when
    sample sizes/variances differ; complements the non-parametric tests. Returns
    ``(t, p)``.
    """
    t, p = sps.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False)
    return (float(t), float(p))


def benjamini_hochberg(pvals, q: float = 0.05) -> tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR control over a family of p-values.

    Forensics use: we run many probes/conditions; BH keeps the expected false-discovery
    rate at ``q`` so a few spurious "significant" timing/logit effects don't pollute the
    report. Returns ``(rejected_mask, crit)`` where ``rejected_mask`` is a boolean array
    aligned to ``pvals`` and ``crit`` is the largest p-value declared significant (0.0 if
    none).
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), 0.0
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool), 0.0
    kmax = int(np.max(np.nonzero(below)[0]))  # largest rank satisfying BH condition
    crit = float(ranked[kmax])
    rejected = p <= crit
    return rejected, crit


def cohen_d(a, b) -> float:
    """Cohen's d: standardized mean difference with a pooled SD.

    Forensics use: magnitude of a latency/logit difference in SD units, reported
    alongside a p-value in Evidence ``stats`` so small-but-significant effects are
    visible as small. Positive when mean(a) > mean(b).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled <= _EPS:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def cliffs_delta(a, b) -> float:
    """Cliff's delta: non-parametric effect size in [-1, 1].

    Forensics use: ordinal dominance of one latency sample over another
    (``P(a>b) - P(a<b)``); robust to outliers and the natural companion to the
    Mann-Whitney test. +1 means every a exceeds every b; 0 means full overlap.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return float((gt - lt) / (a.size * b.size))


def proportion_ci(k: int, n: int, level: float = 0.9) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Forensics use: interval for an agreement/accuracy/recall rate (k successes in n
    trials) — MoE temp-0 determinism, needle-in-haystack recall, sampler tests. The
    Wilson interval stays inside [0, 1] and is well-behaved for k near 0 or n, unlike
    the normal (Wald) interval. Returns ``(lo, hi)``.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = float(sps.norm.ppf(1.0 - (1.0 - level) / 2.0))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))
