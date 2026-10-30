"""Tests for the behavioural fingerprint module (offline, simulator-only).

Validates that
* :func:`fingerprint` on two RunResults from the SAME sim config is close, and from
  DIFFERENT configs (dense vs ssm) is far;
* :func:`equality_test` says "same" for two identical simulated adapters and not-"same"
  for two different configs, for both generative and decision endpoints, and never raises
  on API errors or an exhausted budget;
* the reference library round-trips and matches a fresh fingerprint to its nearest ref.

All runs use ``config={'fast': True}`` so the suite stays quick.
"""
from __future__ import annotations

import asyncio

import pytest

from forensics.core.adapters.base import Adapter
from forensics.core.adapters.simulated import (
    SimConfig, SimulatedAdapter, sim_decoder_dense, sim_decoder_moe, sim_jev, sim_ssm,
)
from forensics.core.budget import Budget
from forensics.core.types import Capabilities, GenResponse
from forensics.fingerprint import (
    equality_test, fingerprint, fingerprint_distance, load_references,
    match_run, numeric_features, save_reference_from_run, tag_set,
)
from forensics.inference.engine import run_forensics


def _run(coro):
    return asyncio.run(coro)


def _forensics(cfg: SimConfig, seed: int = 0):
    return _run(run_forensics(SimulatedAdapter(cfg), config={"fast": True}, seed=seed))


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def runs():
    """Two dense runs (different seeds) + one ssm run, computed once for the module."""
    return {
        "dense1": _forensics(sim_decoder_dense(), seed=1),
        "dense2": _forensics(sim_decoder_dense(), seed=2),
        "ssm": _forensics(sim_ssm(), seed=1),
    }


# =========================================================================== fingerprint
def test_fingerprint_structure(runs):
    fp = fingerprint(runs["dense1"])
    nums = numeric_features(fp)
    tags = tag_set(fp)
    assert len(nums) >= 20, "expected a rich numeric feature vector"
    assert len(tags) >= 3, "expected several categorical tags"
    # every numeric feature is a finite float; meta keys are excluded
    assert all(isinstance(v, float) for v in nums.values())
    assert not any(k.startswith("_") for k in nums)
    # the tokenizer token-delta signature is present
    assert any(k.startswith("tok.delta.") for k in nums)
    # a categorical tag names the tokenizer family
    assert any(t.startswith("tokenizer.family=") for t in tags)


def test_fingerprint_self_distance_zero(runs):
    fp = fingerprint(runs["dense1"])
    assert fingerprint_distance(fp, fp) == pytest.approx(0.0, abs=1e-9)


def test_fingerprint_distance_symmetric(runs):
    a = fingerprint(runs["dense1"])
    b = fingerprint(runs["ssm"])
    assert fingerprint_distance(a, b) == pytest.approx(fingerprint_distance(b, a), abs=1e-9)


def test_fingerprint_same_config_close(runs):
    a = fingerprint(runs["dense1"])
    b = fingerprint(runs["dense2"])
    d = fingerprint_distance(a, b)
    assert d < 0.05, f"same-config fingerprints should be close, got {d}"


def test_fingerprint_different_config_far(runs):
    same = fingerprint_distance(fingerprint(runs["dense1"]), fingerprint(runs["dense2"]))
    diff = fingerprint_distance(fingerprint(runs["dense1"]), fingerprint(runs["ssm"]))
    assert diff > 0.12, f"dense-vs-ssm should be far, got {diff}"
    assert diff > same + 0.1, f"cross-config ({diff}) must exceed same-config ({same})"


def test_fingerprint_distance_bounds(runs):
    for x in ("dense1", "dense2", "ssm"):
        for y in ("dense1", "dense2", "ssm"):
            d = fingerprint_distance(fingerprint(runs[x]), fingerprint(runs[y]))
            assert 0.0 <= d <= 1.0


# =========================================================================== equality: gen
def test_equality_same_config_generative():
    res = _run(equality_test(SimulatedAdapter(sim_decoder_dense()),
                             SimulatedAdapter(sim_decoder_dense()),
                             n_prompts=8, samples=3, rng=0))
    assert res["kind"] == "generative"
    assert res["verdict"] == "same", res
    assert res["token_match_rate"] == pytest.approx(1.0)
    assert res["p_value"] is not None and res["p_value"] > 0.05
    assert res["statistic"] == pytest.approx(0.0, abs=1e-6)


def test_equality_different_config_generative():
    res = _run(equality_test(SimulatedAdapter(sim_decoder_dense()),
                             SimulatedAdapter(sim_ssm()),
                             n_prompts=8, samples=3, rng=0))
    assert res["verdict"] == "different", res
    # different tokenizer + logit geometry => no token agreement and a significant MMD
    assert res["token_match_rate"] < 0.5
    assert res["p_value"] is not None and res["p_value"] < 0.05


def test_equality_different_config_not_same_moe():
    # dense vs MoE share the low-rank core but differ in batch-invariance/serving:
    # the verdict must not be "same".
    res = _run(equality_test(SimulatedAdapter(sim_decoder_dense()),
                             SimulatedAdapter(sim_decoder_moe()),
                             n_prompts=10, samples=2, rng=0))
    assert res["verdict"] != "same", res


# =========================================================================== equality: decision
def test_equality_same_config_decision():
    res = _run(equality_test(SimulatedAdapter(sim_jev()), SimulatedAdapter(sim_jev()),
                             n_prompts=6, rng=0))
    assert res["kind"] == "decision"
    assert res["verdict"] == "same", res
    assert res["token_match_rate"] == pytest.approx(1.0)


def test_equality_kind_mismatch():
    res = _run(equality_test(SimulatedAdapter(sim_decoder_dense()),
                             SimulatedAdapter(sim_jev()), n_prompts=4, rng=0))
    assert res["kind"] == "mismatch"
    assert res["verdict"] == "different"


# =========================================================================== equality: robustness
class _BrokenAdapter(Adapter):
    """An adapter that raises on every request (to test error handling)."""

    kind = "generative"
    name = "broken"

    def static_capabilities(self) -> Capabilities:
        return Capabilities(kind="generative", logprobs=True, max_top_logprobs=5,
                            temperature=True, usage_prompt_tokens=True)

    async def generate(self, req):  # noqa: ANN001
        raise RuntimeError("simulated transport failure")


class _ErrorAdapter(Adapter):
    """An adapter that returns error responses (never raises) for every request."""

    kind = "generative"
    name = "erroring"

    def static_capabilities(self) -> Capabilities:
        return Capabilities(kind="generative", logprobs=False, temperature=True,
                            usage_prompt_tokens=True)

    async def generate(self, req):  # noqa: ANN001
        return GenResponse(status=500, error="internal error")


def test_equality_never_raises_on_transport_errors():
    res = _run(equality_test(_BrokenAdapter(), SimulatedAdapter(sim_decoder_dense()),
                             n_prompts=5, samples=2, rng=0))
    assert res["verdict"] in ("inconclusive", "different")
    assert isinstance(res["notes"], list)


def test_equality_never_raises_on_api_errors():
    res = _run(equality_test(_ErrorAdapter(), _ErrorAdapter(), n_prompts=5, samples=2, rng=0))
    # no usable data on either side -> inconclusive, but no exception
    assert res["verdict"] == "inconclusive"


def test_equality_budget_aware():
    budget = Budget(max_requests=3)
    res = _run(equality_test(SimulatedAdapter(sim_decoder_dense()),
                             SimulatedAdapter(sim_decoder_dense()),
                             n_prompts=8, samples=3, rng=0, budget=budget))
    # only a few requests were allowed; must not raise and must report it
    assert res["n_requests"] <= 3
    assert any("budget" in n.lower() for n in res["notes"])


# =========================================================================== reference library
def test_reference_roundtrip_and_match(tmp_path, runs):
    path = tmp_path / "reference_fingerprints.json"
    save_reference_from_run("dense-ref", runs["dense1"], meta={"family": "dense"}, path=path)
    save_reference_from_run("ssm-ref", runs["ssm"], meta={"family": "ssm"}, path=path)

    lib = load_references(path)
    assert set(lib) == {"dense-ref", "ssm-ref"}
    assert lib["dense-ref"]["meta"]["family"] == "dense"

    # a fresh dense run should match the dense reference first
    fresh = _forensics(sim_decoder_dense(), seed=9)
    ranked = match_run(fresh, path=path)
    assert ranked[0][0] == "dense-ref", ranked
    assert ranked[0][1] < ranked[1][1]


def test_load_references_missing_file(tmp_path):
    assert load_references(tmp_path / "does_not_exist.json") == {}
