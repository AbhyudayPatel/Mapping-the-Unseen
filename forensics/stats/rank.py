"""Numerical-rank estimation of a logit matrix — the core of hidden-size inference.

Pure ``numpy``/``scipy``; the bootstrap method takes an ``rng`` and holds no global state.

Science (see ``research/techniques_logits_tokenizer.md`` §1, §5). For hidden state
``h in R^d`` and a fixed unembedding ``W in R^{Vxd}``, the logits are ``z = W h`` so a
matrix of logit vectors from many prompts has rank <= d. The vocabulary V (~10^5) far
exceeds d (~10^3-10^4), so the logit cloud lives in a thin d-dimensional subspace: the
number of large singular values *is* the hidden size d.

WHY ROW-CENTERING MATTERS (the single most common implementation bug).
The API never returns raw logits z. It returns log-probabilities
``l = z - logZ * 1``, where ``logZ = log sum_j exp(z_j)`` is one unknown scalar PER
PROMPT (per row). Stacking the observed rows therefore adds a spurious rank-1 term
``-logZ (x) 1`` (the all-ones column direction weighted by each row's offset) to the
true rank-<=d matrix, so the observed rank is <= d+1. That extra "+1" direction is
exactly the all-ones vector 1. Subtracting each row's mean over the fixed column set
(row-centering) removes the 1-component, restoring rank d and eliminating the spurious
gap at d+1. (Equivalently: work with logit differences l_i - l_j, which cancel the
offset pairwise, or with an unembedding bias absorbed as a constant column.) Without
centering you systematically over-estimate the hidden size by 1.

Methods:
- 'gap'      : rank = index of the largest relative gap in the singular values.
- 'mp'       : Marchenko-Pastur / Gavish-Donoho optimal hard threshold for the noise
               floor given the matrix shape; rank = # singular values above it.
- 'bootstrap': resample rows, recompute the gap-based rank each time, take the modal
               rank and report the fraction of resamples that agree as confidence.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

__all__ = ["estimate_rank"]

_EPS = 1e-12


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _gap_rank(s: np.ndarray, rtol: float = 1e-9,
              min_cliff_ratio: float = 3.0) -> tuple[int, int, float, bool]:
    """Rank from the largest relative singular-value gap.

    Numerically-zero singular values are dropped first (``s_i <= rtol * s_0``): row-
    centering removes exactly one column dimension, so the centered matrix is rank
    deficient and its smallest singular value is ~0. Left in, that value would create an
    astronomically large ratio in the noise tail and mask the true signal/noise gap.

    A *cliff* only counts as a real signal/noise boundary if the drop ``s_i / s_{i+1}`` is
    at least ``min_cliff_ratio``. A smooth power-law spectrum with no floor (which happens
    when the matrix has fewer columns than the true rank, so it is full-rank with no gap)
    has only mild ratios (~1-2): we then report ``has_cliff=False`` and return the number of
    retained singular values as a *lower bound*, NOT a spurious tiny rank. Callers must treat
    ``has_cliff=False`` as "hidden size >= n_cols; collect more distinct token columns".

    Returns ``(rank, gap_index, confidence, has_cliff, n_retained)`` where ``n_retained`` is
    the number of singular values above the numeric-zero floor (used by the caller to tell a
    clean rank-d matrix, whose tail is exactly zero, from a full-rank matrix with no gap).
    """
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return int(s.size), max(0, s.size - 1), 0.0, bool(s.size), int(s.size)
    # Drop the numeric-zero tail: the exact zeros of a clean rank-d matrix PLUS the one ~0
    # value that row-centering always introduces. What remains is either pure signal (clean)
    # or signal followed by a quantisation noise floor (real bf16/fp16/fp8 logits) — and the
    # signal/noise cliff we want lives inside that retained block, not at the drop-to-zero edge
    # (which would otherwise win with an astronomical ratio and give a wildly inflated rank).
    floor = rtol * float(s[0]) if s[0] > 0 else 0.0
    kept = s[s > floor]
    n_retained = int(kept.size)
    if kept.size <= 1:
        return int(kept.size), max(0, kept.size - 1), 0.0, bool(kept.size), n_retained
    ratios = kept[:-1] / (kept[1:] + _EPS)
    gap_index = int(np.argmax(ratios))
    top = float(ratios[gap_index])
    has_cliff = top >= min_cliff_ratio
    if not has_cliff:
        # No signal/noise cliff among the retained values. The rank is the retained count;
        # whether that is the true rank (clean, tail was exact zeros) or only a lower bound
        # (full-rank over too-few columns) is decided by the caller from n_retained vs n_cols.
        return n_retained, gap_index, 0.0, False, n_retained
    rank = gap_index + 1
    others = np.delete(ratios, gap_index)
    second = float(np.max(others)) if others.size else 1.0
    confidence = float(np.clip(1.0 - second / (top + _EPS), 0.0, 1.0))
    return rank, gap_index, confidence, True, n_retained


def _marpas_density(x: float, beta: float) -> float:
    """Marchenko-Pastur density at eigenvalue x for aspect ratio beta = min/max <= 1."""
    top = (1.0 + np.sqrt(beta)) ** 2
    bot = (1.0 - np.sqrt(beta)) ** 2
    if bot <= x <= top:
        return float(np.sqrt((top - x) * (x - bot)) / (2.0 * np.pi * beta * x))
    return 0.0


def _median_marcenko_pastur(beta: float) -> float:
    """Median of the Marchenko-Pastur distribution for aspect ratio ``beta in (0, 1]``.

    Computed from a fine numeric grid + cumulative trapezoid of the density. This is robust
    to the integrable ``1/x`` singularity at the lower edge (which appears as ``beta -> 1``,
    where the lower edge -> 0) that made an adaptive ``quad``-per-node bisection return ~0.
    """
    beta = float(min(max(beta, 1e-6), 1.0))
    lo = (1.0 - np.sqrt(beta)) ** 2
    hi = (1.0 + np.sqrt(beta)) ** 2
    # Offset the lower bound slightly to avoid the 1/x edge singularity, then normalise.
    xs = np.linspace(lo + 1e-6 * (hi - lo), hi, 200000)
    dens = np.array([_marpas_density(float(x), beta) for x in xs])
    cdf = np.concatenate([[0.0], np.cumsum((dens[:-1] + dens[1:]) / 2.0) * np.diff(xs)])
    total = cdf[-1]
    if total <= 0:
        return (hi + lo) / 2.0
    cdf /= total
    med = float(np.interp(0.5, cdf, xs))
    return med if med > 0 else (hi + lo) / 2.0


def _mp_threshold(s: np.ndarray, n_rows: int, n_cols: int) -> tuple[float, int]:
    """Gavish-Donoho optimal hard threshold for unknown noise given the matrix shape.

    Uses the median-singular-value estimator so no explicit noise level is required.
    Returns ``(threshold, rank)`` where rank = # singular values strictly above it.
    """
    beta = min(n_rows, n_cols) / max(n_rows, n_cols)
    lam = np.sqrt(2.0 * (beta + 1.0) + 8.0 * beta / ((beta + 1.0) + np.sqrt(beta ** 2 + 14.0 * beta + 1.0)))
    mu = _median_marcenko_pastur(beta)
    # Median of the *noise* singular values. Drop the numeric-zero tail first (exact zeros of a
    # rank-deficient / clean matrix + the centering dimension), else their count drags the
    # median to ~0 and everything clears the threshold. The estimator is scale-correct when the
    # noise floor dominates the retained spectrum by count, which holds for real logits (V >> d).
    s = np.asarray(s, dtype=float)
    kept = s[s > 1e-9 * float(s[0])] if s.size and s[0] > 0 else s
    med = float(np.median(kept)) if kept.size else 0.0
    if mu <= 0 or med <= 0:
        return 0.0, int(kept.size)
    omega = lam / np.sqrt(mu)
    thresh = float(omega * med)
    rank = int(np.sum(s > thresh))
    return thresh, rank


def estimate_rank(
    matrix,
    method: str = "gap",
    center: bool = True,
    rng: np.random.Generator | int | None = None,
    n_boot: int = 50,
) -> dict:
    """Estimate the numerical rank of a (prompts x tokens) logit/logprob matrix.

    This is the core of hidden-size (d_model) estimation: the recovered rank equals the
    model's hidden dimension. ``center=True`` (default, strongly recommended) subtracts
    each row's mean over the columns to remove the per-row log-softmax offset — see the
    module docstring for why this fixes an otherwise systematic +1 over-estimate.

    Args:
        matrix: 2-D array, rows = prompts, columns = a fixed shared token set (need
            more columns than the expected rank).
        method: 'gap' | 'mp' | 'bootstrap' (see module docstring).
        center: row-center to kill the log-softmax rank-1 offset before the SVD.
        rng: seed/Generator for the 'bootstrap' method.
        n_boot: number of row resamples for 'bootstrap'.

    Returns::

        {
          'rank': int estimate from the selected method,
          'singular_values': [...] of the (centered) matrix, descending,
          'gap_index': index of the largest relative singular-value gap,
          'mp_threshold': Marchenko-Pastur/Gavish-Donoho noise threshold,
          'confidence': method-appropriate confidence in [0, 1],
          'method': the method used,
          'centered': whether row-centering was applied,
        }
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("matrix must be 2-D (prompts x tokens)")
    if center:
        # Row-center: subtract each row's mean over the fixed column set. Removes the
        # log-softmax per-row offset (the spurious all-ones rank-1 direction).
        M = M - M.mean(axis=1, keepdims=True)

    n_rows, n_cols = M.shape
    s = np.linalg.svd(M, compute_uv=False)

    gap_rank, gap_index, gap_conf, has_cliff, n_retained = _gap_rank(s)
    mp_thresh, mp_rank = _mp_threshold(s, n_rows, n_cols)
    # The estimate is a lower bound (indeterminate) only when there is NO signal/noise cliff
    # AND no numeric-zero tail was dropped — i.e. the matrix is full-rank over the columns we
    # sampled, so the true hidden size exceeds n_cols and we must collect more distinct tokens.
    # A clean rank-d matrix has no cliff either, but its exact-zero tail (n_retained << columns)
    # proves the rank is real, not a lower bound.
    full_rank_no_gap = (not has_cliff) and n_retained >= min(n_rows, n_cols) - 1
    indeterminate = bool(full_rank_no_gap)

    if method == "gap":
        rank, confidence = gap_rank, gap_conf
    elif method == "mp":
        rank = mp_rank
        # Confidence: relative margin between the last kept sv and the threshold.
        if 0 < mp_rank <= s.size:
            last_kept = float(s[mp_rank - 1])
            confidence = float(np.clip((last_kept - mp_thresh) / (last_kept + _EPS), 0.0, 1.0))
        else:
            confidence = 0.0
    elif method == "bootstrap":
        rng = _as_rng(rng)
        ranks = np.empty(n_boot, dtype=int)
        for i in range(n_boot):
            idx = rng.integers(0, n_rows, n_rows)
            Mb = M[idx]  # rows already centered; row-resampling preserves centering
            sb = np.linalg.svd(Mb, compute_uv=False)
            ranks[i], _, _, _, _ = _gap_rank(sb)
        vals, counts = np.unique(ranks, return_counts=True)
        best = int(np.argmax(counts))
        rank = int(vals[best])
        confidence = float(counts[best] / n_boot)
    else:
        raise ValueError("method must be 'gap', 'mp', or 'bootstrap'")

    return {
        "rank": int(rank),
        "singular_values": [float(v) for v in s],
        "gap_index": int(gap_index),
        "gap_rank": int(gap_rank),
        "mp_rank": int(mp_rank),
        "mp_threshold": float(mp_thresh),
        "confidence": float(confidence),
        "has_cliff": bool(has_cliff),
        "indeterminate": bool(indeterminate),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "method": method,
        "centered": bool(center),
    }
