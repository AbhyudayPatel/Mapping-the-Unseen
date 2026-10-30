"""Behavioural fingerprinting: stable per-model signatures, an equality test and a
reference library.

* :func:`fingerprint` / :func:`fingerprint_distance` — build a comparable behavioural
  fingerprint vector from a :class:`~forensics.inference.run_types.RunResult` and measure
  the distance between two.
* :func:`equality_test` — "are these two endpoints the same model?" (Gao-et-al. style).
* :mod:`reference` — save/load fingerprints to the knowledge base and match new ones.
"""
from __future__ import annotations

from forensics.fingerprint.equality import equality_test
from forensics.fingerprint.reference import (
    REFERENCE_PATH,
    load_references,
    match_fingerprint,
    match_run,
    save_reference,
    save_reference_from_run,
)
from forensics.fingerprint.vector import (
    fingerprint,
    fingerprint_distance,
    numeric_features,
    tag_set,
)

__all__ = [
    "fingerprint",
    "fingerprint_distance",
    "numeric_features",
    "tag_set",
    "equality_test",
    "REFERENCE_PATH",
    "load_references",
    "save_reference",
    "save_reference_from_run",
    "match_fingerprint",
    "match_run",
]
