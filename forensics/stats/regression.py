"""Robust regression and model selection for timing/scaling forensics.

Pure ``numpy``/``scipy``; the one randomized routine (``poly_model_select``'s bootstrap)
takes an ``rng`` and holds no global state.

Where these are used (see ``research/techniques_timing_serving.md`` §3-§4):

- ``theilsen`` — robust slope of ITL vs context length (KV-cache bytes/token) or any
  latency-vs-size line where a handful of queueing outliers would wreck OLS.
- ``huber_linear`` — Huber M-estimator line fit, the paper's recommended robust
  regressor for TTFT/ITL sweeps (tracks the compute floor under jitter).
- ``poly_model_select`` — the core attention-family discriminator: fit TTFT vs prompt
  length as degree-1 (linear ⇒ SSM/linear-attention) vs degree-2 (quadratic ⇒ full
  self-attention), pick by BIC, and decide whether the quadratic curvature term is
  real via its bootstrap CI excluding 0.
- ``loglog_slope`` — power-law exponent (slope in log-log space) for scaling
  relationships (e.g. throughput vs size).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

__all__ = ["theilsen", "huber_linear", "poly_model_select", "loglog_slope"]


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def theilsen(x, y, level: float = 0.9) -> tuple[float, float, tuple[float, float]]:
    """Theil-Sen robust linear fit (median of pairwise slopes).

    Forensics use: recover a latency-vs-size slope that ignores up to ~29% outliers
    (cold starts, co-tenant contention). Returns ``(slope, intercept, (slope_lo,
    slope_hi))`` where the slope CI is the distribution-free Theil-Sen interval at the
    requested confidence ``level``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = sps.theilslopes(y, x, alpha=level)
    slope, intercept, lo, hi = float(res[0]), float(res[1]), float(res[2]), float(res[3])
    return slope, intercept, (lo, hi)


def huber_linear(x, y, c: float = 1.345, max_iter: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Huber robust linear regression via iteratively reweighted least squares (IRLS).

    Forensics use: the recommended robust fit for TTFT/ITL vs length sweeps — it
    down-weights (but does not fully discard) queueing spikes, tracking the compute
    floor. The tuning constant ``c=1.345`` gives ~95% efficiency vs OLS under normal
    noise; the residual scale is a robust MAD estimate.

    Returns ``coeffs = np.array([slope, intercept])``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # OLS start
    for _ in range(max_iter):
        r = y - X @ beta
        # MAD-based robust scale (1.4826 makes it consistent for normal noise).
        mad = np.median(np.abs(r - np.median(r)))
        s = 1.4826 * mad
        if s <= 1e-12:
            s = float(np.std(r))
            if s <= 1e-12:
                break  # perfect fit
        u = r / (c * s)
        w = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-12))
        sw = np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(sw[:, None] * X, sw * y, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return np.asarray(beta, dtype=float)


def _bic(n: int, rss: float, n_params: int) -> float:
    """Gaussian-likelihood BIC. ``n_params`` counts coefficients; +1 added for sigma."""
    k = n_params + 1
    if rss <= 0:
        return -np.inf
    return float(n * np.log(rss / n) + k * np.log(n))


def _aic(n: int, rss: float, n_params: int) -> float:
    k = n_params + 1
    if rss <= 0:
        return -np.inf
    return float(n * np.log(rss / n) + 2 * k)


def poly_model_select(
    x,
    y,
    degrees: tuple[int, ...] = (1, 2),
    criterion: str = "bic",
    n_boot: int = 1000,
    level: float = 0.9,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Fit competing polynomials, pick one by information criterion, test the curvature.

    Forensics use: the attention-family discriminator. Fit TTFT vs prompt length as a
    line (degree 1 ⇒ linear/SSM/linear-attention or chunked prefill) vs a parabola
    (degree 2 ⇒ genuine O(len^2) self-attention). Selection is by BIC (or AIC). The
    quadratic (curvature) coefficient c2 is only trusted if its percentile-bootstrap CI
    excludes 0 — a significant positive c2 is the signature of full quadratic attention.

    Returns a dict::

        {
          'degree': best degree by the chosen criterion,
          'coeffs': np.polyfit coeffs of the best model (highest power first),
          'bic': {degree: bic_value, ...},
          'aic': {degree: aic_value, ...},
          'curvature': c2 (x^2 coefficient of the degree-2 fit, or None),
          'curvature_ci': (lo, hi) bootstrap CI for c2 (or None),
          'quadratic_significant': bool (CI excludes 0) or None,
          'resid_std': residual std of the best model,
        }
    """
    rng = _as_rng(rng)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    degrees = tuple(int(d) for d in degrees)

    bic: dict[int, float] = {}
    aic: dict[int, float] = {}
    fits: dict[int, np.ndarray] = {}
    for d in degrees:
        coeffs = np.polyfit(x, y, d)
        resid = y - np.polyval(coeffs, x)
        rss = float(np.sum(resid ** 2))
        fits[d] = coeffs
        bic[d] = _bic(n, rss, d + 1)
        aic[d] = _aic(n, rss, d + 1)

    scores = bic if criterion == "bic" else aic if criterion == "aic" else None
    if scores is None:
        raise ValueError("criterion must be 'bic' or 'aic'")
    best = int(min(scores, key=scores.get))
    best_coeffs = fits[best]

    # Residual std of the selected model (degrees-of-freedom corrected when possible).
    resid = y - np.polyval(best_coeffs, x)
    dof = max(1, n - (best + 1))
    resid_std = float(np.sqrt(np.sum(resid ** 2) / dof))

    # Curvature: c2 from a degree-2 fit + bootstrap significance test.
    curvature = None
    curvature_ci = None
    quad_sig = None
    if n >= 3:
        coeffs2 = fits.get(2)
        if coeffs2 is None:
            coeffs2 = np.polyfit(x, y, 2)
        curvature = float(coeffs2[0])  # x^2 coefficient (polyfit: highest power first)
        boot = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            # Guard against a degenerate resample (all x identical).
            if np.ptp(x[idx]) == 0:
                boot[i] = np.nan
                continue
            boot[i] = np.polyfit(x[idx], y[idx], 2)[0]
        boot = boot[~np.isnan(boot)]
        if boot.size:
            alpha = 1.0 - level
            lo = float(np.quantile(boot, alpha / 2.0))
            hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
            curvature_ci = (lo, hi)
            quad_sig = not (lo <= 0.0 <= hi)

    return {
        "degree": best,
        "coeffs": best_coeffs,
        "bic": bic,
        "aic": aic,
        "curvature": curvature,
        "curvature_ci": curvature_ci,
        "quadratic_significant": quad_sig,
        "resid_std": resid_std,
    }


def loglog_slope(x, y) -> dict:
    """Power-law exponent via a straight-line fit in log-log space.

    Forensics use: quantify power-law scaling (e.g. cost vs size, y = A * x**b) — the
    fitted slope is the exponent b, the intercept is log(A). Non-positive x/y values
    are dropped (log undefined). Returns
    ``{'slope', 'intercept', 'r2', 'stderr', 'n'}``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    if lx.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "stderr": float("nan"), "n": int(lx.size)}
    res = sps.linregress(lx, ly)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue ** 2),
        "stderr": float(res.stderr),
        "n": int(lx.size),
    }
