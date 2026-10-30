"""Run one bench subject locally (CPU), same code path as the Modal job."""
import asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper" / "bench"))
import torch
torch.set_num_threads(4)
from loader import load_subject
from wrapper import ArchitectureWrapper
from battery import run_battery
import subjects as bench_subjects

async def go(key):
    roster = {s["key"]: s for s in bench_subjects.roster()}
    subj = roster[key]
    a, b, gt = load_subject(subj, cache_dir=str(ROOT / ".cache" / "openweight-pilot" / "models"))
    w = ArchitectureWrapper(a, b, subj.get("opts") or {})
    res = await run_battery(subj, w, str(ROOT / "runs" / "forensics-bench" / f"journal-{key}"),
                            request_budget=3400)
    res["ground_truth"] = gt
    out = ROOT / "runs" / "forensics-bench" / f"{key}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print("done", key, res["budget"]["requests"], "protocol", res["protocol"])

if __name__ == "__main__":
    asyncio.run(go(sys.argv[1]))
