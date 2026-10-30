"""Self-test for the identifiability honesty layer.

Drives the SimulatedAdapter through the REAL probe registry, collects the estimates and
the set of probe ids that actually ran, and checks the identifiability levels behave the
way the contract requires (level depends on caps + what ran).
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

from forensics.core.adapters.simulated import SimulatedAdapter, sim_decoder_dense
from forensics.probes import load_all
from forensics.probes.base import ProbeContext
from forensics.inference import identifiability as ident


# --------------------------------------------------------------------------- a tiny Belief stub
class _BeliefStub:
    """Duck-typed stand-in for the real Belief (owned by another module)."""
    def __init__(self, posterior):
        self.posterior = posterior

    def as_dict(self):
        return self.posterior


# --------------------------------------------------------------------------- probe driver
async def _run_probes(adapter, caps, probe_ids, config):
    reg = load_all()
    ran: set[str] = set()
    estimates = []
    for pid in probe_ids:
        cls = reg[pid]
        probe = cls()
        ctx = ProbeContext(adapter, caps=caps, config=config)
        ctx.current_probe = pid
        ok, why = probe.applicable(ctx)
        if not ok:
            print(f"  skip {pid}: {why}")
            continue
        res = await probe.run(ctx)
        ran.add(pid)
        estimates.extend(res.estimates)
        print(f"  ran  {pid}: status={res.status} estimates={[e.quantity for e in res.estimates]}")
    return ran, estimates


def main() -> None:
    cfg = sim_decoder_dense()
    adapter = SimulatedAdapter(cfg)
    caps = adapter.static_capabilities()
    config = {"fast": True}

    # Cheap-but-representative probes that exercise DIRECT / STATISTICAL / timing paths,
    # plus P03 (logits/hidden_size) so the logprobs+P03 path is real.
    probe_ids = [
        "P01.interface.stack",
        "P02.tokenizer.fingerprint",
        "P03.logits.hidden_size",
        "P04.precision.logit_dtype",
        "P05.context.limits",
        "P07.timing.scaling",
    ]

    print("Running probes against sim_decoder_dense (logprobs=True) ...")
    ran, estimates = asyncio.run(_run_probes(adapter, caps, probe_ids, config))
    print(f"\nran probes: {sorted(ran)}")

    belief = _BeliefStub({"tokenizer.family": {"o200k_base": 0.9, "cl100k_base": 0.1},
                          "custom.dynamic": {"a": 0.5, "b": 0.5}})

    entries = ident.analyze(belief, estimates, caps, ran)
    by_key = {e.key: e for e in entries}

    print("\n" + ident.format_by_level(entries))

    # ------------------------------------------------------------------ assertions
    def lvl(k):
        return by_key[k].level

    # 1) logprobs=True + P03 ran  ->  hidden_size level <= 2
    assert "P03.logits.hidden_size" in ran, "P03 should have run"
    assert lvl("dims.hidden_size") <= 2, f"hidden_size should be <=2, got {lvl('dims.hidden_size')}"

    # 2) n_layers always >= 4 (never directly identifiable from black-box I/O)
    assert lvl("dims.n_layers") >= 4, f"n_layers should be >=4, got {lvl('dims.n_layers')}"

    # 3) context.max_input_tokens level 1 when P05 read the limit off the error
    assert "P05.context.limits" in ran, "P05 should have run"
    assert lvl("context.max_input_tokens") == 1, \
        f"max_input_tokens should be DIRECT (1), got {lvl('context.max_input_tokens')}"

    # dynamic property discovered from the belief is present
    assert "custom.dynamic" in by_key, "dynamic belief property should surface as an entry"

    # every level>=4 entry carries an equivalence argument; every level<4 does not
    for e in entries:
        if e.level >= 4:
            assert e.equivalence, f"{e.key} (level {e.level}) must carry an equivalence argument"
        else:
            assert not e.equivalence, f"{e.key} (level {e.level}) must NOT carry an equivalence argument"

    # ------------------------------------------------------------------ logprobs=False case
    caps_nolp = replace(caps, logprobs=False, max_top_logprobs=None, logprob_bytes=False,
                        logit_bias_affects_logprobs=None)
    ran_nolp = ran - {"P03.logits.hidden_size", "P04.precision.logit_dtype"}
    entries2 = ident.analyze(belief, [e for e in estimates if e.quantity != "dims.hidden_size"],
                             caps_nolp, ran_nolp)
    b2 = {e.key: e for e in entries2}
    assert b2["dims.hidden_size"].level == 5, \
        f"hidden_size should be NON_IDENTIFIABLE (5) without logprobs, got {b2['dims.hidden_size'].level}"
    assert b2["dims.hidden_size"].equivalence, "hidden_size level-5 must carry an equivalence note"
    print("\n[logprobs=False] dims.hidden_size ->", b2["dims.hidden_size"].level,
          "|", b2["dims.hidden_size"].equivalence[:70], "...")

    # ------------------------------------------------------------------ no logprobs & no timing case
    caps_bare = replace(caps_nolp, server_timing_header=None)
    ran_bare: set[str] = {"P01.interface.stack"}
    entries3 = ident.analyze(None, [], caps_bare, ran_bare)
    b3 = {e.key: e for e in entries3}
    assert b3["dims.n_heads"].level == 5, \
        f"n_heads should be 5 with no logprobs & no timing, got {b3['dims.n_heads'].level}"
    assert b3["dims.n_heads"].equivalence
    print("[no logprobs & no timing] dims.n_heads ->", b3["dims.n_heads"].level)

    # summary sanity
    summ = ident.summarize(entries)
    assert sum(summ.by_level.values()) == len(entries)
    assert summ.arch_verdict.startswith("YES") or "YES" in summ.arch_verdict
    print("\nby_level:", summ.by_level)
    print("arch verdict:", summ.arch_verdict[:120], "...")

    print("\nPASS: identifiability self-test green "
          f"({len(entries)} entries; hidden_size<=2 w/ logprobs, =5 w/o; n_layers>=4; ctx_in=1)")


def test_identifiability() -> None:
    """Pytest entry point (drives the real probe registry against the simulator)."""
    main()


if __name__ == "__main__":
    main()
