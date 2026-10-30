"""1-D Gaussian mixture modelling and burstiness metrics for serving forensics.

Pure ``numpy``/``scipy``; the EM fit takes an ``rng`` (for multi-start init) and holds
no global state.

Where these are used (see ``research/techniques_timing_serving.md`` §6, §9):

- ``gmm1d`` — EM for a 1-D Gaussian mixture with BIC-based order selection. Fit to
  log-latency to detect routers/cascades/hardware tiers (K>1 well-separated modes) and
  to inter-token latency to detect the speculative-decoding bimodal ITL (a tight
  near-zero cluster of batched tokens plus a slow verification cluster).
- ``burstiness`` — coefficient of variation and Sarle's bimodality coefficient of an
  interval sequence, a cheap first screen for the spec-decoding ITL burst pattern
  before committing to a full GMM.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

__all__ = ["gmm1d", "burstiness"]

_LOG2PI = float(np.log(2.0 * np.pi))


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _log_gauss(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Log N(x; mean, var), computed in log-space for numerical stability."""
    var = max(var, 1e-12)
    return -0.5 * (_LOG2PI + np.log(var) + (x - mean) ** 2 / var)


def _em_gmm(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 8,
    max_iter: int = 300,
    tol: float = 1e-7,
    reg: float = 1e-6,
) -> dict:
    """EM for a k-component 1-D GMM with multiple random restarts; best log-likelihood wins."""
    n = data.size
    data_var = float(np.var(data)) + reg
    best: dict | None = None

    for _ in range(n_init):
        # Initialise means at distinct random data points; equal weights; global variance.
        if n >= k:
            means = np.array(rng.choice(data, size=k, replace=False), dtype=float)
        else:
            means = np.array(rng.choice(data, size=k, replace=True), dtype=float)
        variances = np.full(k, data_var, dtype=float)
        weights = np.full(k, 1.0 / k, dtype=float)
        ll_old = -np.inf

        for _ in range(max_iter):
            # E-step in log-space: log responsibilities via log-sum-exp.
            log_comp = np.empty((n, k))
            for j in range(k):
                log_comp[:, j] = np.log(max(weights[j], 1e-300)) + _log_gauss(data, means[j], variances[j])
            log_norm = np.logaddexp.reduce(log_comp, axis=1)
            ll = float(np.sum(log_norm))
            resp = np.exp(log_comp - log_norm[:, None])

            # M-step.
            Nk = resp.sum(axis=0)
            weights = np.clip(Nk / n, 1e-12, None)
            weights = weights / weights.sum()
            means = (resp * data[:, None]).sum(axis=0) / np.maximum(Nk, 1e-12)
            variances = (resp * (data[:, None] - means) ** 2).sum(axis=0) / np.maximum(Nk, 1e-12) + reg

            if np.abs(ll - ll_old) < tol * (1.0 + abs(ll_old)):
                break
            ll_old = ll

        if best is None or ll > best["ll"]:
            best = {"means": means.copy(), "variances": variances.copy(),
                    "weights": weights.copy(), "ll": ll}
    return best  # type: ignore[return-value]


def gmm1d(
    data,
    k_range: tuple[int, ...] = (1, 2, 3),
    rng: np.random.Generator | int | None = None,
    n_init: int = 8,
) -> dict:
    """Fit 1-D Gaussian mixtures for each k in ``k_range`` and select the best by BIC.

    Forensics use: detect latency regimes / routing. Fit to log-TTFT — a single mode is
    one model on one hardware tier; two or more well-separated modes suggest a
    router/cascade or a mixed fleet. Fit to inter-token latency to spot the
    speculative-decoding bimodal pattern.

    Returns::

        {
          'k': best k by BIC,
          'means': [...] (sorted ascending),
          'weights': [...],
          'covs': [...] (1-D variances, same order as means),
          'bic': {k: bic_value, ...},
          'loglik': best-model log-likelihood,
        }
    """
    rng = _as_rng(rng)
    data = np.asarray(data, dtype=float).ravel()
    n = data.size
    ks = [int(k) for k in k_range if 1 <= int(k) <= max(1, n)]
    if not ks:
        ks = [1]

    bic: dict[int, float] = {}
    models: dict[int, dict] = {}
    for k in ks:
        m = _em_gmm(data, k, rng, n_init=n_init)
        n_params = 3 * k - 1  # k means + k variances + (k-1) free weights
        bic[k] = float(-2.0 * m["ll"] + n_params * np.log(max(n, 1)))
        models[k] = m

    best_k = int(min(bic, key=bic.get))
    m = models[best_k]

    # Sort components by ascending mean for a stable, readable output.
    order = np.argsort(m["means"])
    return {
        "k": best_k,
        "means": [float(v) for v in m["means"][order]],
        "weights": [float(v) for v in m["weights"][order]],
        "covs": [float(v) for v in m["variances"][order]],
        "bic": bic,
        "loglik": float(m["ll"]),
    }


def burstiness(intervals) -> dict:
    """Burstiness metrics for an interval sequence (e.g. inter-token latencies).

    Forensics use: a cheap screen for speculative decoding. Plain autoregression yields
    near-unimodal, low-CV ITLs; speculative decoding interleaves near-zero intervals
    (batched accepted tokens) with large verification stalls, inflating both the CV and
    the bimodality coefficient.

    Returns::

        {
          'cv': coefficient of variation (std/mean),
          'bimodality_coef': Sarle's bimodality coefficient (>~0.555 hints bimodal),
          'burstiness': Goh-Barabasi B = (cv-1)/(cv+1) in [-1, 1],
          'mean': ..., 'std': ..., 'n': ...,
        }
    """
    x = np.asarray(intervals, dtype=float).ravel()
    n = x.size
    mean = float(np.mean(x)) if n else float("nan")
    std = float(np.std(x)) if n else float("nan")
    cv = std / mean if (n and mean != 0) else float("nan")

    if n >= 4:
        g = float(sps.skew(x))            # sample skewness
        kurt = float(sps.kurtosis(x, fisher=True))  # excess kurtosis
        denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        bc = (g * g + 1.0) / denom if denom != 0 else float("nan")
    else:
        bc = float("nan")

    b = (cv - 1.0) / (cv + 1.0) if (cv == cv and (cv + 1.0) != 0) else float("nan")

    return {
        "cv": float(cv),
        "bimodality_coef": float(bc),
        "burstiness": float(b),
        "mean": mean,
        "std": std,
        "n": int(n),
    }
