"""Tests for forensics.stats against synthetic data with known ground truth."""
from __future__ import annotations

import numpy as np
import pytest

from forensics.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    burstiness,
    cliffs_delta,
    cohen_d,
    estimate_rank,
    gmm1d,
    knee_point,
    ks_2samp,
    loglog_slope,
    mannwhitney,
    permutation_test,
    poly_model_select,
    proportion_ci,
    segmented1d,
    theilsen,
    welch_t,
)


# --------------------------------------------------------------------------- regression
def test_theilsen_recovers_slope_under_noise_and_outliers():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 10.0, 80)
    true_slope, true_intercept = 2.5, 1.0
    y = true_slope * x + true_intercept + rng.normal(0.0, 0.5, x.size)
    y[::10] += 40.0  # 10% gross outliers that would wreck OLS

    # OLS is wrecked by the outliers; Theil-Sen should still nail the slope.
    ols_slope = np.polyfit(x, y, 1)[0]
    slope, intercept, (lo, hi) = theilsen(x, y)
    assert abs(slope - true_slope) < 0.3
    assert lo <= true_slope <= hi
    assert abs(slope - true_slope) < abs(ols_slope - true_slope)  # more robust than OLS
    assert intercept > 0.0  # intercept sane (all-positive outliers bias it upward)


def test_poly_model_select_picks_degree2_on_quadratic():
    rng = np.random.default_rng(2)
    x = np.linspace(-5.0, 5.0, 90)
    y = 0.8 * x**2 - 1.2 * x + 3.0 + rng.normal(0.0, 1.0, x.size)

    res = poly_model_select(x, y, degrees=(1, 2), criterion="bic", n_boot=600, rng=rng)
    assert res["degree"] == 2
    assert res["quadratic_significant"] is True
    lo, hi = res["curvature_ci"]
    assert lo > 0.0  # CI for the positive curvature excludes 0
    assert abs(res["curvature"] - 0.8) < 0.3


def test_poly_model_select_picks_degree1_on_linear():
    rng = np.random.default_rng(3)
    x = np.linspace(-5.0, 5.0, 90)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 1.0, x.size)

    res = poly_model_select(x, y, degrees=(1, 2), criterion="bic", n_boot=600, rng=rng)
    assert res["degree"] == 1
    assert res["quadratic_significant"] is False
    lo, hi = res["curvature_ci"]
    assert lo <= 0.0 <= hi  # curvature CI straddles 0


def test_loglog_slope_recovers_power_law_exponent():
    x = np.linspace(1.0, 100.0, 50)
    y = 3.0 * x**1.5
    res = loglog_slope(x, y)
    assert abs(res["slope"] - 1.5) < 1e-6
    assert res["r2"] > 0.999


# --------------------------------------------------------------------------- mixture
def test_gmm1d_finds_two_components_on_bimodal():
    rng = np.random.default_rng(4)
    data = np.concatenate([
        rng.normal(-3.0, 0.5, 150),
        rng.normal(3.0, 0.5, 150),
    ])
    res = gmm1d(data, k_range=(1, 2, 3), rng=rng)
    assert res["k"] == 2
    means = sorted(res["means"])
    assert means[0] < -1.5 and means[1] > 1.5
    assert res["bic"][2] < res["bic"][1]  # 2 components beat 1


def test_gmm1d_prefers_one_component_on_unimodal():
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, 400)
    res = gmm1d(data, k_range=(1, 2, 3), rng=rng)
    assert res["k"] == 1


def test_burstiness_flags_bimodal_intervals():
    rng = np.random.default_rng(6)
    intervals = np.concatenate([
        rng.normal(0.01, 0.002, 120),  # batched (accepted) tokens
        rng.normal(1.0, 0.05, 120),    # verification stalls
    ])
    res = burstiness(intervals)
    assert res["bimodality_coef"] > 0.555
    assert res["cv"] > 0.5

    uni = rng.normal(1.0, 0.05, 240)
    res2 = burstiness(uni)
    assert res2["bimodality_coef"] < 0.555


# --------------------------------------------------------------------------- rank (core)
def _rank_r_matrix(rng, n, m, r):
    A = rng.standard_normal((n, r))
    B = rng.standard_normal((r, m))
    return A @ B


def test_estimate_rank_recovers_rank_no_offset():
    rng = np.random.default_rng(7)
    n, m, r = 200, 64, 5
    M = _rank_r_matrix(rng, n, m, r) + 0.01 * rng.standard_normal((n, m))

    for method in ("gap", "mp"):
        res = estimate_rank(M, method=method, center=True)
        assert res["rank"] == r, (method, res["rank"])
    # bootstrap agrees too
    res_bs = estimate_rank(M, method="bootstrap", center=True, rng=rng, n_boot=40)
    assert res_bs["rank"] == r
    assert res_bs["confidence"] > 0.8


def test_estimate_rank_centering_fixes_the_plus_one():
    rng = np.random.default_rng(8)
    n, m, r = 200, 64, 5
    base = _rank_r_matrix(rng, n, m, r) + 0.01 * rng.standard_normal((n, m))

    # Add a per-row constant offset (the log-softmax logZ term): rank-1 all-ones column.
    offset = rng.standard_normal(n)
    M = base + offset[:, None]

    # Without centering, the spurious all-ones direction inflates the rank to r+1.
    uncentered = estimate_rank(M, method="gap", center=False)
    assert uncentered["rank"] == r + 1

    # Row-centering removes the offset and restores the true rank r.
    centered = estimate_rank(M, method="gap", center=True)
    assert centered["rank"] == r

    # MP threshold method agrees after centering.
    assert estimate_rank(M, method="mp", center=True)["rank"] == r


# --------------------------------------------------------------------------- changepoint
def test_segmented1d_finds_one_breakpoint():
    rng = np.random.default_rng(9)
    x = np.arange(0.0, 40.0)
    y = np.where(x < 20.0, 2.0 * x, 40.0 + 0.2 * (x - 20.0)) + rng.normal(0.0, 0.4, x.size)

    res = segmented1d(x, y, max_segments=2)
    assert res["n_segments"] == 2
    assert len(res["breakpoints"]) == 1
    assert 16.0 < res["breakpoints"][0] < 24.0


def test_knee_point_detects_cliff():
    x = np.arange(0.0, 30.0)
    y = np.where(x < 18.0, 1.0, 0.2)  # recall cliff at 18
    res = knee_point(x, y)
    assert res["knee_x"] is not None
    assert 10.0 <= res["knee_x"] <= 20.0


# --------------------------------------------------------------------------- tests.py
def test_bootstrap_ci_brackets_point_and_matches_normal_width():
    rng = np.random.default_rng(10)
    data = rng.normal(5.0, 2.0, 500)
    lo, hi, point = bootstrap_ci(data, np.mean, n_boot=2000, level=0.9, rng=rng)
    se = data.std(ddof=1) / np.sqrt(data.size)
    assert abs(point - np.mean(data)) < 1e-9  # point is the sample statistic
    assert lo < point < hi                     # percentile CI brackets the point
    assert abs(point - 5.0) < 4.0 * se         # sample mean near the truth
    half = (hi - lo) / 2.0                      # ~= z_0.95 * SE for the mean
    assert 1.3 * se < half < 2.0 * se


def test_permutation_and_parametric_tests_detect_difference():
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 1.0, 120)
    b = rng.normal(1.0, 1.0, 120)

    p_perm = permutation_test(a, b, stat="mean_diff", n=2000, rng=rng)
    assert p_perm < 0.01
    _, p_mw = mannwhitney(a, b)
    assert p_mw < 0.01
    _, p_ks = ks_2samp(a, b)
    assert p_ks < 0.01
    _, p_t = welch_t(a, b)
    assert p_t < 0.01

    # No difference -> not significant.
    c = rng.normal(0.0, 1.0, 120)
    assert permutation_test(a, c, n=2000, rng=rng) > 0.05


def test_effect_sizes():
    rng = np.random.default_rng(12)
    a = rng.normal(2.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 200)
    assert cohen_d(a, b) > 1.0
    assert cliffs_delta(a, b) > 0.5
    assert abs(cliffs_delta(a, a)) < 1e-9


def test_benjamini_hochberg_controls_fdr():
    pvals = [0.001, 0.008, 0.02, 0.2, 0.5, 0.9]
    rejected, crit = benjamini_hochberg(pvals, q=0.05)
    assert rejected.tolist() == [True, True, True, False, False, False]
    assert abs(crit - 0.02) < 1e-12

    none_rej, crit0 = benjamini_hochberg([0.4, 0.6, 0.9], q=0.05)
    assert not none_rej.any()
    assert crit0 == 0.0


def test_proportion_ci_wilson():
    lo, hi = proportion_ci(8, 10, level=0.9)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    # Degenerate n=0 returns the trivial interval.
    assert proportion_ci(0, 0) == (0.0, 1.0)


# --------------------------------------------------------------------------- determinism
def test_determinism_given_seed():
    data = np.random.default_rng(99).normal(0.0, 1.0, 100)
    r1 = bootstrap_ci(data, np.mean, n_boot=500, rng=np.random.default_rng(0))
    r2 = bootstrap_ci(data, np.mean, n_boot=500, rng=np.random.default_rng(0))
    assert r1 == r2

    p1 = permutation_test(data[:50], data[50:], n=1000, rng=np.random.default_rng(0))
    p2 = permutation_test(data[:50], data[50:], n=1000, rng=np.random.default_rng(0))
    assert p1 == p2


# --- regression tests for hidden-size rank recovery (added after simulator validation) ---
import numpy as _np
from forensics.stats.rank import estimate_rank as _estimate_rank


def _lowrank(n_rows, n_cols, r, noise=0.0, seed=0):
    g = _np.random.default_rng(seed)
    A = g.standard_normal((n_rows, r))
    B = g.standard_normal((r, n_cols))
    M = A @ B
    if noise:
        M = M + g.standard_normal((n_rows, n_cols)) * noise
    return M


def test_rank_gap_recovers_clean():
    M = _lowrank(600, 500, 64, noise=0.0, seed=1)
    r = _estimate_rank(M, method="gap", center=True)
    assert r["rank"] == 64, r["rank"]
    assert not r["indeterminate"]


def test_rank_recovers_with_noise_floor():
    # realistic case: low-rank signal + a quantisation-like noise floor -> gap & mp both find r
    M = _lowrank(600, 500, 64, noise=0.02, seed=2)
    assert _estimate_rank(M, method="gap", center=True)["rank"] == 64
    assert abs(_estimate_rank(M, method="mp", center=True)["rank"] - 64) <= 2


def test_rank_indeterminate_when_cols_below_rank():
    # only 40 columns but true rank 64 -> no cliff, must flag indeterminate (lower bound), not 1
    M = _lowrank(600, 40, 64, noise=0.01, seed=3)
    r = _estimate_rank(M, method="gap", center=True)
    assert r["indeterminate"] is True
    assert r["rank"] >= 39  # a lower bound near n_cols, never a spurious small value


def test_rank_centering_fixes_plus_one():
    # add a per-row constant (log-softmax-like offset): centering must still give r, not r+1
    M = _lowrank(600, 500, 32, noise=0.01, seed=4)
    M = M + _np.random.default_rng(5).standard_normal((600, 1)) * 50.0  # per-row offset
    assert _estimate_rank(M, method="gap", center=True)["rank"] == 32
