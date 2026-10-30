"""Change-point and knee detection for timing/recall forensics.

Pure ``numpy``; deterministic, no randomness, no global state.

Where these are used (see ``research/techniques_timing_serving.md`` §3-§4 and the
context/recall probes):

- ``segmented1d`` — piecewise-linear segmentation by least-squares grid search. Detects
  the knee where sliding-window attention turns TTFT from quadratic to linear (the
  window size W), paged-attention block steps, or a KV-cache sawtooth in ITL(ctx).
- ``knee_point`` — Kneedle-style single-knee detector (max distance from the chord).
  Detects the onset of context degradation / a sliding-window recall cliff, i.e. the
  length past which needle recall or accuracy falls off.
"""
from __future__ import annotations

import numpy as np

__all__ = ["segmented1d", "knee_point"]


def _fit_segment(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """OLS line for one segment. Returns ``(slope, intercept, rss)``."""
    if xs.size == 1 or np.ptp(xs) == 0:
        # A single point (or a vertical column) -> flat fit at the mean.
        intercept = float(np.mean(ys))
        resid = ys - intercept
        return 0.0, intercept, float(np.sum(resid ** 2))
    A = np.column_stack([xs, np.ones_like(xs)])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ coef
    return float(coef[0]), float(coef[1]), float(np.sum(resid ** 2))


def segmented1d(x, y, max_segments: int = 2, min_size: int = 2) -> dict:
    """Piecewise-linear segmentation via least-squares over candidate breakpoints.

    Uses dynamic programming over precomputed segment RSS values (an exact grid search
    over all breakpoint placements), then selects the number of segments (1..
    ``max_segments``) by BIC so noise alone does not manufacture a breakpoint.

    Forensics use: locate the sliding-window knee W in TTFT(prompt_len) (quadratic then
    linear), paged-block steps, or KV sawtooth resets — anything where the latency-vs-
    size law changes slope at a threshold.

    Returns::

        {
          'breakpoints': [x-values where the slope regime changes],
          'segments': [{'start_x','end_x','slope','intercept','n','rss'}, ...],
          'n_segments': chosen segment count,
          'bic': {k: bic_value, ...},
        }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = x.size
    max_segments = max(1, min(int(max_segments), n))
    min_size = max(1, int(min_size))

    # Precompute the least-squares cost of every contiguous segment [i, j].
    cost = np.full((n, n), np.inf)
    fit: dict[tuple[int, int], tuple[float, float, float]] = {}
    for i in range(n):
        for j in range(i, n):
            slope, intercept, rss = _fit_segment(x[i:j + 1], y[i:j + 1])
            cost[i, j] = rss
            fit[(i, j)] = (slope, intercept, rss)

    INF = float("inf")
    # dp[k][e] = min RSS covering points 0..e using exactly k segments.
    dp = [[INF] * n for _ in range(max_segments + 1)]
    back = [[-1] * n for _ in range(max_segments + 1)]
    for e in range(n):
        if (e + 1) >= min_size or e == n - 1:
            dp[1][e] = cost[0, e]
            back[1][e] = 0
    for k in range(2, max_segments + 1):
        for e in range(n):
            best_c, best_s = INF, -1
            for s in range(1, e + 1):
                if (e - s + 1) < min_size:
                    continue
                if dp[k - 1][s - 1] == INF:
                    continue
                c = dp[k - 1][s - 1] + cost[s, e]
                if c < best_c:
                    best_c, best_s = c, s
            dp[k][e] = best_c
            back[k][e] = best_s

    # Select the number of segments by BIC.
    bic: dict[int, float] = {}
    for k in range(1, max_segments + 1):
        rss = dp[k][n - 1]
        if rss == INF:
            continue
        n_params = 2 * k + (k - 1) + 1  # slopes+intercepts, breakpoints, sigma
        bic[k] = (n * np.log(max(rss, 1e-300) / n) + n_params * np.log(n)
                  if rss > 0 else -INF)
    best_k = int(min(bic, key=bic.get)) if bic else 1

    # Reconstruct the optimal partition for best_k.
    seg_bounds: list[tuple[int, int]] = []
    e = n - 1
    k = best_k
    while k >= 1 and e >= 0:
        s = back[k][e]
        if s < 0:
            s = 0
        seg_bounds.append((s, e))
        e = s - 1
        k -= 1
    seg_bounds.reverse()

    segments = []
    breakpoints = []
    for idx, (i, j) in enumerate(seg_bounds):
        slope, intercept, rss = fit[(i, j)]
        segments.append({
            "start_x": float(x[i]),
            "end_x": float(x[j]),
            "slope": slope,
            "intercept": intercept,
            "n": int(j - i + 1),
            "rss": float(rss),
        })
        if idx > 0:  # breakpoint sits between the previous segment and this one
            prev_j = seg_bounds[idx - 1][1]
            breakpoints.append(float((x[prev_j] + x[i]) / 2.0))

    return {
        "breakpoints": breakpoints,
        "segments": segments,
        "n_segments": best_k,
        "bic": bic,
    }


def knee_point(x, y) -> dict:
    """Kneedle-style knee/elbow detection (maximum distance from the endpoint chord).

    Normalizes x and y to [0, 1], draws the straight chord between the first and last
    points, and reports the point of maximum perpendicular deviation from that chord.
    This is the classic distance-based Kneedle heuristic and finds both convex elbows
    and concave knees (including a sharp cliff).

    Forensics use: detect the onset of a sliding-window recall cliff or context
    degradation — the length at which accuracy/recall drops sharply — from a curve of
    metric vs context length.

    Returns ``{'knee_x','knee_y','knee_index','distance','x_norm'}`` (all None/0 if
    fewer than 3 points).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = x.size
    if n < 3:
        return {"knee_x": None, "knee_y": None, "knee_index": None,
                "distance": 0.0, "x_norm": None}

    xr = np.ptp(x) or 1.0
    yr = np.ptp(y) or 1.0
    xn = (x - x.min()) / xr
    yn = (y - y.min()) / yr

    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    d = p2 - p1
    norm = np.linalg.norm(d)
    if norm == 0:
        return {"knee_x": None, "knee_y": None, "knee_index": None,
                "distance": 0.0, "x_norm": None}
    dn = d / norm

    pts = np.column_stack([xn, yn]) - p1
    proj = pts @ dn
    perp = pts - np.outer(proj, dn)
    dist = np.linalg.norm(perp, axis=1)
    idx = int(np.argmax(dist))
    return {
        "knee_x": float(x[idx]),
        "knee_y": float(y[idx]),
        "knee_index": idx,
        "distance": float(dist[idx]),
        "x_norm": float(xn[idx]),
    }
