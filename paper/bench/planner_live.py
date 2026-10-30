"""Live (non-replay) planner comparison against the simulator.

Unlike v2's replay of fixed outcomes, every run here is a fresh live scan: probes
execute against a :class:`SimulatedAdapter` with the planner actively choosing the
next probe.  Planners: ``eig`` (surrogate EIG/cost), ``static`` (cheapest-first),
``uniform`` (random).  Eight seeds x four simulator subjects x three planners.

Recovery target (frozen): the correct ``arch.family`` value reaches support >= 0.7
(and for the MoE subject, ``arch.sparsity`` = moe >= 0.6 as well).  Cost is requests
to first crossing, censored at the 300-request budget; tokens are recorded too.

Usage: PYTHONIOENCODING=utf-8 python paper/bench/planner_live.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forensics.core.adapters.simulated import (  # noqa: E402
    SimulatedAdapter, sim_decoder_dense, sim_decoder_moe, sim_encoder, sim_ssm)
from forensics.core.budget import Budget  # noqa: E402
from forensics.core.recorder import Recorder  # noqa: E402
from forensics.inference.engine import run_forensics  # noqa: E402

OUT = ROOT / "paper" / "bench" / "out" / "planner_live.json"
JOURNALS = ROOT / "runs" / "planner-live"

SUBJECTS = {
    "dense": (sim_decoder_dense, "decoder_only", {}),
    "moe": (sim_decoder_moe, "decoder_only", {"arch.sparsity": ("moe", 0.5)}),
    "ssm": (sim_ssm, "ssm_or_linear", {}),
    "encoder": (sim_encoder, "encoder_only", {}),
}
PLANNERS = ["eig", "static", "uniform"]
SEEDS = list(range(8))
BUDGET = 300
TARGET_P = 0.7


async def one_run(subj_key: str, planner: str, seed: int) -> dict:
    factory, fam_true, extra = SUBJECTS[subj_key]
    cfg = factory()
    # per-seed serving noise so seeds are statistically distinct
    cfg.timing_jitter_s = 0.004 * (1 + seed % 4)
    adapter = SimulatedAdapter(cfg)
    run_dir = JOURNALS / f"{subj_key}-{planner}-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(max_requests=BUDGET, max_output_tokens=400_000)
    rec = Recorder(str(run_dir))

    trajectory = []
    if subj_key == "encoder":
        DECISIVE = {"P19.arch.generation"}      # encoder vs generative split
    else:
        DECISIVE = {"P23.decode.state_growth", "P19.arch.prefill_scaling"}

    def hook(step, belief, probe_id, n_requests, probe_requests=0):
        fam = belief.posterior.get("arch.family", {})
        top = max(fam, key=fam.get) if fam else None
        hit = top == fam_true
        extra_ok = all(belief.posterior.get(p, {}).get(v, 0) >= t
                       for p, (v, t) in extra.items())
        # evidence-arrival: has a decisive family discriminator integrated evidence?
        led = belief.ledger.get("arch.family", [])
        decisive = any(e.probe_id in DECISIVE and e.kind == "evidence" for e in led)
        trajectory.append({"step": step, "requests": int(n_requests),
                           "family_top": top, "family_p": round(fam.get(top, 0), 4),
                           "decisive_evidence": decisive,
                           "target_reached": bool(hit and extra_ok)})

    config = {"planner": planner, "seed": seed, "_on_step": hook,
              "max_probes": 60}
    t0 = time.time()
    result = await run_forensics(adapter, config=config, budget=budget, recorder=rec,
                                 llm=None)
    reached_at = next((t["requests"] for t in trajectory if t["target_reached"]), None)
    decisive_at = next((t["requests"] for t in trajectory
                        if t["decisive_evidence"]), None)
    final = trajectory[-1] if trajectory else {}
    return {"subject": subj_key, "planner": planner, "seed": seed,
            "requests_to_target": reached_at, "censored": reached_at is None,
            "requests_to_decisive": decisive_at,
            "decisive_censored": decisive_at is None,
            "final_family_top": final.get("family_top"),
            "final_family_p": final.get("family_p"),
            "total_requests": budget.requests,
            "wall_s": round(time.time() - t0, 1),
            "n_probes_ran": len(result.probe_results)}


async def main():
    JOURNALS.mkdir(parents=True, exist_ok=True)
    rows = []
    for subj in SUBJECTS:
        for planner in PLANNERS:
            for seed in SEEDS:
                r = await one_run(subj, planner, seed)
                rows.append(r)
                print(f"{subj:8s} {planner:7s} s{seed}: "
                      f"target@{r['requests_to_target']} "
                      f"(censored={r['censored']}, probes={r['n_probes_ran']})",
                      flush=True)
    # aggregate with bootstrap CIs (both endpoints: top-1 target and decisive evidence)
    agg = {}
    rng = np.random.default_rng(0)
    for subj in SUBJECTS:
        for planner in PLANNERS:
            sel = [r for r in rows if r["subject"] == subj and r["planner"] == planner]
            vals = np.array([r["requests_to_target"] if r["requests_to_target"]
                             is not None else BUDGET for r in sel], float)
            dec = np.array([r["requests_to_decisive"] if r["requests_to_decisive"]
                            is not None else BUDGET for r in sel], float)
            cens = float(np.mean([r["censored"] for r in sel]))
            dcens = float(np.mean([r["decisive_censored"] for r in sel]))
            boots = [np.mean(rng.choice(vals, size=len(vals), replace=True))
                     for _ in range(2000)]
            dboots = [np.mean(rng.choice(dec, size=len(dec), replace=True))
                      for _ in range(2000)]
            agg[f"{subj}|{planner}"] = {
                "mean_requests": float(np.mean(vals)),
                "ci": [float(np.percentile(boots, 2.5)),
                       float(np.percentile(boots, 97.5))],
                "censor_rate": cens,
                "decisive_mean": float(np.mean(dec)),
                "decisive_ci": [float(np.percentile(dboots, 2.5)),
                                float(np.percentile(dboots, 97.5))],
                "decisive_censor_rate": dcens, "n": len(sel)}
    out = {"rows": rows, "aggregate": agg,
           "protocol": {"budget": BUDGET, "target_p": TARGET_P, "seeds": len(SEEDS)}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
