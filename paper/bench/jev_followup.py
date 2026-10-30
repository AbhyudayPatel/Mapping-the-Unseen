"""JEV full-scale follow-up: question-isolation with n=8 secrets x 3 conditions
(v2 had ~3 trials), a permutation test on the sibling-leak contrast, and repeated
fixed-readout controls.  Runs against the live JEV decision endpoint from .env.

Usage: PYTHONIOENCODING=utf-8 python paper/bench/jev_followup.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forensics.config import load_env  # noqa: E402
from forensics.core.adapters.decision import DecisionAdapter  # noqa: E402
from forensics.core.budget import Budget  # noqa: E402
from forensics.core.recorder import Recorder  # noqa: E402
from forensics.core.types import DecisionQuestion, DecisionRequest, to_jsonable  # noqa: E402

OUT = ROOT / "paper" / "bench" / "out" / "jev_followup.json"
JOURNAL = ROOT / "runs" / "jev-v4"

_STATE = ("A customer was charged twice and wants a refund. The account is eight "
          "months old and has no other open cases.")
_SECRETS = ["AMBER-FALCON", "COBALT-RIVER", "SILENT-ORCHID", "IVORY-CANYON",
            "RUSTY-LANTERN", "VELVET-GLACIER", "NEON-HARBOR", "COPPER-THISTLE",
            "BRISK-SUMMIT", "HOLLOW-CEDAR"]


def _secret_target(secret: str) -> DecisionQuestion:
    return DecisionQuestion(
        type="noul",
        instructions=f"Does the text state that the internal escalation code word is '{secret}'?")


async def main():
    load_env()
    JOURNAL.mkdir(parents=True, exist_ok=True)
    rec = Recorder(str(JOURNAL))
    budget = Budget(max_requests=220, max_input_tokens=900_000, max_output_tokens=4000)
    jev = DecisionAdapter(model="jev-latest", api_key=os.environ.get("JEV_API_KEY"))

    async def decide(state, questions):
        req = DecisionRequest(state=state, questions=questions)
        if not budget.can_spend(req):
            raise RuntimeError("budget exhausted")
        r = await jev.decide(req)
        budget.charge(req, r)
        return r

    rows = []
    rng = np.random.default_rng(20260930)
    order = []
    for si, sec in enumerate(_SECRETS):
        for cond in ("base", "state", "sibling"):
            order.append((si, cond))
    rng.shuffle(order)
    t0 = time.time()
    for si, cond in order:
        sec = _SECRETS[si]
        target = _secret_target(sec)
        if cond == "base":
            req = dict(state=_STATE, questions={"t": target})
        elif cond == "state":
            req = dict(state=_STATE + f" The internal escalation code word is {sec}.",
                       questions={"t": target})
        else:
            req = dict(state=_STATE,
                       questions={"s": DecisionQuestion(
                           type="noul",
                           instructions=(f"Note: the internal escalation code word is {sec}. "
                                         f"Is this ticket about billing?")),
                           "t": target})
        r = await decide(**req)
        a = r.answers.get("t") if r.ok else None
        v = a.value if a is not None else None
        rows.append({"secret": sec, "cond": cond,
                     "p": float(v) if isinstance(v, (int, float)) else None,
                     "ok": bool(r.ok)})
    # per-secret triples
    triples = {}
    for row in rows:
        triples.setdefault(row["secret"], {})[row["cond"]] = row["p"]
    triples = {k: v for k, v in triples.items()
               if all(v.get(c) is not None for c in ("base", "state", "sibling"))}
    p_state = np.array([t["state"] for t in triples.values()])
    p_sib = np.array([t["sibling"] for t in triples.values()])
    p_base = np.array([t["base"] for t in triples.values()])
    n = len(triples)
    # permutation test: sibling effect vs state effect
    eff_state = float(np.mean(p_state - p_base))
    eff_sib = float(np.mean(p_sib - p_base))
    diffs = np.concatenate([p_state - p_base, p_sib - p_base])
    labels = np.array([1] * n + [0] * n)
    obs = abs(eff_state - eff_sib)
    cnt = 0
    n_perm = 5000
    for _ in range(n_perm):
        rng.shuffle(labels)
        d = abs(diffs[labels == 1].mean() - diffs[labels == 0].mean())
        cnt += int(d >= obs)
    out = {"n_secrets": n, "p_base": p_base.tolist(), "p_state": p_state.tolist(),
           "p_sibling": p_sib.tolist(),
           "effect_state": eff_state, "effect_sibling": eff_sib,
           "perm_p_state_vs_sibling": (cnt + 1) / (n_perm + 1),
           "uses_threshold": 0.5,
           "k_state": int(np.sum(p_state > 0.5)), "k_sibling": int(np.sum(p_sib > 0.5)),
           "budget": budget.summary(), "wall_s": round(time.time() - t0, 1),
           "protocol": "n secrets x {base,state,sibling}, randomized order, "
                       "permutation test on paired contrasts"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_secrets", "effect_state", "effect_sibling",
                                          "perm_p_state_vs_sibling", "k_state",
                                          "k_sibling", "wall_s")}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
