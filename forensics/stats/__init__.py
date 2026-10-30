"""Statistical toolbox for black-box model forensics.

Pure ``numpy``/``scipy``, deterministic given a seed, and free of global state — every
randomized routine takes an explicit ``rng`` (a ``numpy.random.Generator``, an integer
seed, or ``None``).

Submodules:
- ``tests``      : bootstrap CIs, permutation/Mann-Whitney/KS/Welch tests, BH-FDR,
                   effect sizes (Cohen's d, Cliff's delta), Wilson proportion CIs.
- ``regression`` : Theil-Sen and Huber robust lines, polynomial model selection
                   (linear vs quadratic attention), log-log power-law slope.
- ``changepoint``: piecewise-linear segmentation and Kneedle knee detection.
- ``mixture``    : 1-D Gaussian mixtures (EM + BIC) and burstiness metrics.
- ``rank``       : centered-SVD rank estimation (hidden-size / d_model inference).
"""
from __future__ import annotations

from forensics.stats.changepoint import knee_point, segmented1d
from forensics.stats.mixture import burstiness, gmm1d
from forensics.stats.rank import estimate_rank
from forensics.stats.regression import (
    huber_linear,
    loglog_slope,
    poly_model_select,
    theilsen,
)
from forensics.stats.tests import (
    benjamini_hochberg,
    bootstrap_ci,
    cliffs_delta,
    cohen_d,
    ks_2samp,
    mannwhitney,
    permutation_test,
    proportion_ci,
    welch_t,
)

__all__ = [
    # tests
    "bootstrap_ci",
    "permutation_test",
    "mannwhitney",
    "ks_2samp",
    "welch_t",
    "benjamini_hochberg",
    "cohen_d",
    "cliffs_delta",
    "proportion_ci",
    # regression
    "theilsen",
    "huber_linear",
    "poly_model_select",
    "loglog_slope",
    # changepoint
    "segmented1d",
    "knee_point",
    # mixture
    "gmm1d",
    "burstiness",
    # rank
    "estimate_rank",
]
