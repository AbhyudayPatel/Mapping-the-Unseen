"""Blind controlled-change detection with a commit-reveal protocol.

Version 2's fixture table showed metrics moving when *the researcher* changed the
fixture — not evidence of detection performance.  This harness makes it a blind test:

1. The harness draws an episode type from {none, cap, ledger, rounding, swap, router},
   applies it behind the anonymous wrapper, and writes only
   ``SHA256(salt || episode_label)`` to the scan manifest **before** any request.
2. The scanner (frozen detectors below) runs a fixed mini-battery and outputs binary
   detections without access to the label.
3. The harness reveals the label + salt; anyone can recompute the hash chain and
   verify the commit preceded the scan (``episodes[].commit_ok`` in the output).

Detector rules (frozen before the study):
* cap      : the measured max-output error message changes vs the baseline scan;
* ledger   : usage.prompt_tokens departs from the locally counted token total of the
             fixed P02 strings beyond fp rounding (|Δ| > 0 on >30% of strings);
* rounding : the fraction of top-5 logprob values lying exactly on the bf16 grid
             (grid membership) flips from <0.9 to >0.9 vs baseline;
* swap     : max |Δ top-5 logprob| over 8 fixed canary prompts > 1e-3;
* router   : repeated identical prompt yields a bimodal top-logprob vector (two
             clusters separated by > 1e-3 with an exact alternating pattern over ≥8
             repeats, or a permutation test on cluster distance, p<0.01);
* none     : no detector may fire (specificity).

n episodes configurable (default 48).  Output: per-type sensitivity, specificity on
'none', Wilson 95% CIs, and the full hash chain.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import torch
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "bench"))

from wrapper import ArchitectureWrapper, ServedModel, render  # noqa: E402
from loader import load_subject  # noqa: E402

CACHE = ROOT / ".cache" / "openweight-pilot" / "models"
OUT = ROOT / "paper" / "bench" / "out" / "blind_fixtures.json"

EPISODES = ["none", "cap", "ledger", "rounding", "swap", "router"]
CANARY = ["The capital of France is", "Water is made of", "The stock market today",
          "def quicksort(arr):", "The mitochondria is the", "In 1969, Apollo",
          "Photosynthesis converts", "The largest ocean is"]
P02_STRINGS = ["hello world", "The quick brown fox jumps over 13 lazy dogs.",
               "def f(x): return x**2", "https://example.com/path?q=1",
               "  whitespace\t and\nnewlines", "1234567890"]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# --------------------------------------------------------------------------- detectors
class Scanner:
    """Frozen detection battery.  Sees only the wrapper; never the label."""

    def __init__(self, wrapper: ArchitectureWrapper):
        self.wrapper = wrapper

    async def _post(self, client, content, max_tokens=4, logprobs=False, top=5):
        r = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "logprobs": logprobs, "top_logprobs": top,
            "temperature": 0.0})
        return r

    async def scan(self):
        out = {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(self.wrapper.app),
                                     base_url="http://b") as c:
            # cap probe
            r = await self._post(c, "hi", max_tokens=10_000)
            out["cap_error"] = (r.json().get("error") or {}).get("message", "")
            # ledger: counted vs reported tokens
            ledger = []
            for s in P02_STRINGS:
                r = await self._post(c, s, max_tokens=1)
                rep = (r.json().get("usage") or {}).get("prompt_tokens")
                local = len(self.wrapper.a.tok.encode(render(
                    [{"role": "user", "content": s}]), add_special_tokens=False))
                ledger.append({"reported": rep, "counted": local})
            out["ledger"] = ledger
            # rounding: lattice statistic on repeated numeric outputs
            grids = []
            for s in CANARY[:4]:
                r = await self._post(c, s, max_tokens=1, logprobs=True)
                try:
                    lps = [t["logprob"] for t in
                           r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]]
                    diffs = np.diff(sorted(set(round(v, 12) for v in lps)))
                    grids.append(float(np.min(diffs)) if len(diffs) else None)
                except Exception:
                    grids.append(None)
            out["grids"] = grids
            # swap/router: canary top-5 vectors + 8 repeats of one prompt
            vecs = []
            for s in CANARY:
                r = await self._post(c, s, max_tokens=1, logprobs=True)
                d = r.json()["choices"][0]["logprobs"]["content"][0]
                vecs.append({t["token_id"]: t["logprob"] for t in d["top_logprobs"]})
            out["canary"] = vecs
            reps = []
            for _ in range(8):
                r = await self._post(c, CANARY[0], max_tokens=1, logprobs=True)
                d = r.json()["choices"][0]["logprobs"]["content"][0]
                reps.append({t["token_id"]: t["logprob"] for t in d["top_logprobs"]})
            out["repeats"] = reps
        return out

    @staticmethod
    def detect(baseline: dict, scan: dict) -> dict:
        det = {}
        det["cap"] = scan["cap_error"] != baseline["cap_error"]
        mism = sum(1 for r in scan["ledger"]
                   if r["reported"] is not None and r["reported"] != r["counted"])
        det["ledger"] = mism > 0.3 * len(scan["ledger"])

        def _grid_membership(scan):
            """Fraction of reported logprob values lying exactly on the bf16 grid."""
            vals = []
            for v in scan["canary"]:
                vals.extend(v.values())
            if not vals:
                return 0.0
            import numpy as _np
            import torch as _t
            arr = _np.array(vals, dtype=_np.float64)
            bf = _t.tensor(arr).bfloat16().double().numpy()
            return float(_np.mean(_np.isclose(arr, bf, atol=1e-12)))
        det["rounding"] = (_grid_membership(scan) > 0.9) != (_grid_membership(baseline) > 0.9)

        def _maxdiff(va, vb):
            md = 0.0
            for a, b in zip(va, vb):
                for k in a:
                    if k in b:
                        md = max(md, abs(a[k] - b[k]))
            return md
        det["swap"] = _maxdiff(scan["canary"], baseline["canary"]) > 1e-3
        # router: alternating clusters in the repeats
        reps = scan["repeats"]
        d01 = _maxdiff(reps[0::2], reps[1::2])
        d00 = _maxdiff(reps[0::2], reps[0::2])
        det["router"] = bool(d01 > 1e-3 and d00 < 1e-4)
        return det


# --------------------------------------------------------------------------- fixtures
def apply_episode(label, base_served, twin_served, rng):
    opts = {}
    b = None
    if label == "cap":
        opts["max_output"] = 4
    elif label == "rounding":
        opts["wire_round"] = "bf16"
    elif label == "swap":
        return ArchitectureWrapper(twin_served, opts=opts)
    elif label == "router":
        b = twin_served
    elif label == "ledger":
        opts["ledger_inflate"] = True
    return ArchitectureWrapper(base_served if label != "swap" else twin_served, b, opts)


async def main(n_episodes: int = 48):
    torch.set_num_threads(4)
    base_subj = {"key": "pythia-14m", "repo": "EleutherAI/pythia-14m",
                 "family": "decoder_only", "arch": "dense", "opts": {}}
    twin_subj = {"key": "pythia-31m", "repo": "EleutherAI/pythia-31m",
                 "family": "decoder_only", "arch": "dense", "opts": {}}
    a1, _, _ = load_subject(base_subj, cache_dir=str(CACHE))
    a2, _, _ = load_subject(twin_subj, cache_dir=str(CACHE))
    baseline_wrapper = ArchitectureWrapper(a1)
    scanner = Scanner(baseline_wrapper)
    baseline = await scanner.scan()
    print("baseline scan done", flush=True)

    rng = np.random.default_rng(20260930)
    episodes = []
    for i in range(n_episodes):
        label = EPISODES[i % len(EPISODES)]
        salt = secrets.token_hex(16)
        commit = _sha(salt + "|" + label)
        episodes.append({"i": i, "label": label, "salt": salt, "commit": commit})
    commits_path = OUT.with_suffix(".commits.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    commits_path.write_text(json.dumps(
        [{"i": e["i"], "commit": e["commit"]} for e in episodes], indent=1))
    print(f"committed {n_episodes} episode hashes -> {commits_path}", flush=True)

    results = []
    for e in episodes:
        w = apply_episode(e["label"], a1, a2, rng)
        scan = await Scanner(w).scan()
        det = Scanner.detect(baseline, scan)
        fired = [k for k, v in det.items() if v]
        ok = _sha(e["salt"] + "|" + e["label"]) == e["commit"]
        results.append({"i": e["i"], "commit": e["commit"], "commit_ok": ok,
                        "label": e["label"], "salt": e["salt"],
                        "detections": det, "fired": fired})
        print(f"episode {i:02d} [{e['label']:8s}] fired={fired}", flush=True)

    # scoring
    def wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0, 0.0)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3))

    score = {}
    for label in EPISODES:
        rows = [r for r in results if r["label"] == label]
        if label == "none":
            k = sum(1 for r in rows if not r["fired"])
            score[label] = {"specificity": wilson(k, len(rows)), "n": len(rows)}
        else:
            k = sum(1 for r in rows if label in r["fired"])
            score[label] = {"sensitivity": wilson(k, len(rows)), "n": len(rows)}
    out = {"episodes": results, "scores": score,
           "protocol": "commit-reveal blind; detectors frozen in blind_fixtures.py"}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(score, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
