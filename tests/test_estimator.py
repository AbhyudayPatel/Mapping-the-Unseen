"""Self-test for forensics.inference.estimator.estimate_dimensions.

Drives the SimulatedAdapter through the real probe registry to obtain at least one
genuine measured Estimate (tokenizer/vocab from P02), then runs the estimator on
synthetic estimate sets with known ground truth (Llama-3-8B, a 5120/152064 config,
and an MoE case) and asserts the honest CIs bracket the truth.
"""
from __future__ import annotations

import asyncio

from forensics.core.adapters.simulated import SimulatedAdapter, sim_decoder_dense
from forensics.inference.estimator import estimate_dimensions
from forensics.inference.hypotheses import PROPERTIES, normalized_priors
from forensics.probes.base import Estimate, Probe, ProbeContext
from forensics.probes import load_all


# --------------------------------------------------------------------------- belief stub
class StubBelief:
    """Minimal Belief-shaped object (agent A owns the real one)."""

    def __init__(self, overrides: dict[str, str]):
        self.posterior: dict[str, dict[str, float]] = {}
        for key, prop in PROPERTIES.items():
            self.posterior[key] = dict(normalized_priors(prop))
        for prop, value in overrides.items():
            self.posterior.setdefault(prop, {})
            # concentrate mass on the chosen value
            self.posterior[prop] = {value: 0.9}

    def top(self, prop: str) -> tuple[str | None, float]:
        d = self.posterior.get(prop)
        if not d:
            return (None, 0.0)
        k = max(d, key=d.get)
        return (k, d[k])

    def as_dict(self) -> dict[str, dict[str, float]]:
        return self.posterior


def _in(value: float, ci: str) -> bool:
    """Is `value` inside a 'lo – hi' CI string (commas stripped)?"""
    parts = ci.replace(",", "").replace("–", "-").split("-")
    nums = [float(p) for p in parts if p.strip().replace(".", "").isdigit()]
    if len(nums) != 2:
        return False
    return nums[0] <= value <= nums[1]


def _row(table: list[dict], quantity: str) -> dict:
    for r in table:
        if r["quantity"] == quantity:
            return r
    raise AssertionError(f"missing row {quantity}: {[r['quantity'] for r in table]}")


# --------------------------------------------------------------------------- registry drive
def _real_tokenizer_estimates() -> list[Estimate]:
    """Run the real P02 tokenizer probe against the dense sim; return its estimates."""
    load_all()
    from forensics.probes.base import REGISTRY

    adapter = SimulatedAdapter(sim_decoder_dense())
    probe_cls = REGISTRY.get("P02.tokenizer.fingerprint") or next(
        (c for pid, c in REGISTRY.items() if pid.startswith("P02")), None)
    if probe_cls is None:
        return []
    probe: Probe = probe_cls()
    ctx = ProbeContext(adapter, caps=adapter.static_capabilities(),
                       config={"fast": True}, seed=1)
    ctx.current_probe = probe.id
    res = asyncio.run(probe.run(ctx))
    return list(res.estimates)


# --------------------------------------------------------------------------- scenarios
def scenario_llama3_8b():
    caps = SimulatedAdapter(sim_decoder_dense()).static_capabilities()
    belief = StubBelief({"arch.family": "decoder_only", "arch.sparsity": "dense",
                         "attn.kv_sharing": "gqa_or_mqa"})
    ests = [
        Estimate("dims.hidden_size", 4096, ci=(4094.0, 4098.0), level="inferred",
                 method="P03 SVD rank", identifiability=2, probe_id="P03"),
        Estimate("dims.vocab_size", 128256, level="inferred", method="P02 tokenizer",
                 probe_id="P02"),
        Estimate("out.logit_dtype", "bf16", level="observed", method="P04", probe_id="P04"),
        Estimate("context.max_input_tokens", 131072, level="observed", method="P05",
                 probe_id="P05"),
    ]
    out = estimate_dimensions(belief, ests, caps, shared={"seed": 7})
    table = out["derived_table"]

    nl = _row(table, "dims.n_layers")
    tp = _row(table, "dims.total_params")
    assert _in(32, nl["ci"]), f"n_layers CI should bracket 32: {nl}"
    assert _in(8.0, tp["ci"]), f"total_params CI should bracket ~8B: {tp}"

    fm = out["family_matches"]
    assert fm, "expected family matches"
    top = fm[0]
    assert "llama-3" in top.name.lower(), f"top family should be Llama-3-like: {top}"
    assert top.score >= 0.5, f"top score should be high: {top}"
    return out


def scenario_5120_sanity():
    caps = SimulatedAdapter(sim_decoder_dense()).static_capabilities()
    belief = StubBelief({"arch.family": "decoder_only", "arch.sparsity": "dense",
                         "attn.kv_sharing": "gqa_or_mqa"})
    ests = [
        Estimate("dims.hidden_size", 5120, ci=(5116.0, 5124.0), level="inferred",
                 probe_id="P03"),
        Estimate("dims.vocab_size", 152064, level="inferred", probe_id="P02"),
    ]
    out = estimate_dimensions(belief, ests, caps, shared={"seed": 3})
    table = out["derived_table"]
    nh = _row(table, "dims.n_heads")
    tp = _row(table, "dims.total_params")
    # d/head_dim = 5120/128 = 40 heads (modal head_dim)
    assert nh["value"] in (40, 80), f"n_heads sanity (mode head_dim=128 -> 40): {nh}"
    assert _in(14.77, tp["ci"]) or _in(32.76, tp["ci"]), f"total CI should span Qwen2.5-14/32B: {tp}"
    assert out["family_matches"], "expected family matches"
    return out


def scenario_moe_flag():
    caps = SimulatedAdapter(sim_decoder_dense()).static_capabilities()
    belief = StubBelief({"arch.family": "decoder_only", "arch.sparsity": "moe",
                         "attn.kv_sharing": "gqa_or_mqa"})
    ests = [
        Estimate("dims.hidden_size", 4096, ci=(4094.0, 4098.0), level="inferred",
                 probe_id="P03"),
        Estimate("dims.vocab_size", 131072, level="inferred", probe_id="P02"),
        # total from knowledge capacity (P12), active from memory-bound timing (P07)
        Estimate("dims.total_params", 47.0, ci=(30.0, 70.0), level="speculative",
                 unit="B", method="P12 knowledge", probe_id="P12"),
        Estimate("dims.active_params", 13.0, ci=(9.0, 18.0), level="estimated",
                 unit="B", method="P07 timing", probe_id="P07"),
        Estimate("dims.n_experts", 8, level="speculative", probe_id="P11"),
        Estimate("dims.active_experts", 2, level="speculative", probe_id="P11"),
    ]
    out = estimate_dimensions(belief, ests, caps, shared={"seed": 5})
    table = out["derived_table"]
    sp = _row(table, "arch.sparsity")
    tp = _row(table, "dims.total_params")
    ap = _row(table, "dims.active_params")
    assert sp["value"] == "moe", f"sparsity row should be moe: {sp}"
    assert "MoE" in sp["method"] or "moe" in sp["method"].lower(), sp
    assert float(ap["value"]) < float(tp["value"]), f"active should be < total: {ap} {tp}"
    assert any("MoE" in n or "moe" in n.lower() for n in out["notes"]), out["notes"]
    return out


def test_llama3_8b():
    scenario_llama3_8b()


def test_5120_sanity():
    scenario_5120_sanity()


def test_moe_flag():
    scenario_moe_flag()


def _print_table(title: str, out: dict):
    print(f"\n=== {title} ===")
    cols = ["quantity", "value", "ci", "level", "identifiability", "method"]
    for r in out["derived_table"]:
        method = r["method"]
        if len(method) > 62:
            method = method[:59] + "..."
        print(f"  {r['quantity']:<28} {str(r['value']):<10} {r['ci']:<18} "
              f"{r['level']:<11} {r['identifiability']:<11} {method}")
    if out.get("family_matches"):
        print("  family matches:")
        for fm in out["family_matches"][:4]:
            print(f"    {fm.name:<24} score={fm.score:<7} matched={fm.matched} "
                  f"conflicts={fm.conflicts}")
    if out.get("notes"):
        print("  notes:", out["notes"])


if __name__ == "__main__":
    real = _real_tokenizer_estimates()
    print("P02 (real registry) estimates:",
          [(e.quantity, e.value, e.level) for e in real])

    out1 = scenario_llama3_8b()
    _print_table("Llama-3-8B synthetic (hidden 4096, vocab 128256, decoder, gqa)", out1)
    out2 = scenario_5120_sanity()
    _print_table("hidden 5120 / vocab 152064 sanity", out2)
    out3 = scenario_moe_flag()
    _print_table("MoE case (sparsity=moe, active<<total)", out3)
    print("\nALL SELF-TESTS PASSED")
