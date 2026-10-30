"""Head-to-head baselines on the same ForensicsBench subjects.

Two published-method families, re-implemented in "lite" form and run on the SAME
recorded data as our pipeline (nothing cherry-picked):

* **NightVision-lite** (Ellis et al., 2026): hidden width from the rank of the
  natural top-5 logprob matrix (their restricted-access regime), and depth from
  prefill-time-per-token inverted through a per-hardware constant calibrated on
  the other subjects (leave-one-out).
* **LLMmap-lite** (Pasquini et al., 2025): a fixed 24-prompt fingerprint
  (top-logprob values, top-5 entropy, latency); nearest-neighbour identification
  and nearest-centroid family classification, both leave-one-out.

The comparison isolates the *task difference*: fingerprinting identifies a known
model; it has no mechanism for an architecture outside its reference set.

Usage: PYTHONIOENCODING=utf-8 python paper/bench/baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "runs" / "forensics-bench"
OUT = ROOT / "paper" / "bench" / "out" / "baselines.json"


def nightvision_lite(subjects: dict) -> dict:
    rows = {}
    # width: rank of the natural top-5 matrix (recorded by the battery)
    for key, r in subjects.items():
        gt = r.get("ground_truth", {})
        w = gt.get("width")
        t5 = (r.get("modules") or {}).get("geometry_top5") or {}
        est = t5.get("rank_estimate") if t5.get("status") == "estimated" else None
        lb = t5.get("observed_rank_lower_bound")
        rows[key] = {"true_width": w, "nv_width_estimate": est,
                     "nv_width_lower_bound": lb, "resolved": est is not None}
    # depth: prefill ms/token from P07 table; constant calibrated leave-one-out
    slopes, widths, layers = {}, {}, {}
    for key, r in subjects.items():
        gt = r.get("ground_truth", {})
        if gt.get("model_kind") != "causal":
            continue
        tab = []
        for pr in r.get("probe_results", []):
            if (pr.get("probe_id") or "").startswith("P07"):
                tab = (pr.get("tables") or {}).get("prefill") or []
        pts = []
        for row in tab:
            pt, sv = row.get("prompt_tokens"), row.get("server_s", row.get("server_ms"))
            if pt and sv:
                pts.append((float(pt), float(sv) / (1000.0 if row.get("server_ms") else 1.0)))
        if len(pts) >= 3 and gt.get("width") and gt.get("layers"):
            xs = np.array([p[0] for p in pts], float)
            ys = np.array([p[1] for p in pts], float)
            A = np.vstack([xs, np.ones_like(xs)]).T
            slope = float(np.linalg.lstsq(A, ys, rcond=None)[0][0])
            if slope > 0:
                slopes[key] = slope          # s per prompt token
                widths[key] = gt["width"]
                layers[key] = gt["layers"]
    depth_rows = {}
    for key in slopes:
        others = [k for k in slopes if k != key]
        # model: slope ≈ c * layers * width^2  (attention+FFN FLOPs per token)
        cvals = [slopes[o] / (layers[o] * widths[o] ** 2) for o in others]
        c = float(np.median(cvals)) if cvals else np.nan
        pred = slopes[key] / (c * widths[key] ** 2) if c and np.isfinite(c) else None
        depth_rows[key] = {"true_layers": layers[key],
                           "nv_depth_estimate": round(pred, 1) if pred else None,
                           "rel_error": (round(abs(pred - layers[key]) / layers[key], 3)
                                         if pred else None)}
    return {"width": rows, "depth": depth_rows}


def llmmap_lite(fps: dict) -> dict:
    feats, labels, fams = {}, {}, {}
    for key, r in fps.items():
        gt = r.get("ground_truth", {})
        vec = []
        ok = True
        for row in r.get("rows", []):
            tops = row.get("top") or []
            if not tops:
                ok = False
                break
            lps = np.array([t["lp"] for t in tops], float)
            p = np.exp(lps - lps.max())
            p = p / p.sum()
            ent = float(-(p * np.log(p + 1e-12)).sum())
            vec.extend([float(lps[0]), ent, float(row.get("server_ms") or 0.0)])
        if ok and len(vec) == 72:
            feats[key] = np.array(vec, float)
            labels[key] = key
            fams[key] = gt.get("family") or gt.get("arch")
    keys = sorted(feats)
    if len(keys) < 3:
        return {"error": "insufficient fingerprints", "n": len(keys)}
    X = np.stack([feats[k] for k in keys])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    def base_checkpoint(k):
        """The weights behind the name (dtype/batch variants share GPT-2 weights)."""
        if k.startswith("gpt2-"):
            return "gpt2"
        if k == "router-gpt2-pythia":
            return "MIXED"                    # two checkpoints; no single identity
        return k

    base = {k: base_checkpoint(k) for k in keys}
    # leave-one-out nearest neighbour (identification) and family centroid
    id_hits = id_eval = fam_hits = 0
    per_subj = {}
    for i, k in enumerate(keys):
        d = np.linalg.norm(X - X[i], axis=1)
        d[i] = np.inf
        nn = int(np.argmin(d))
        if base[k] == "MIXED":
            id_hit = False                 # unidentifiable by construction
        elif sum(1 for j in range(len(keys)) if base[keys[j]] == base[k] and j != i) == 0:
            id_hit = None                  # no twin in the reference set
        else:
            id_hit = base[keys[nn]] == base[k]
        cents: dict[str, list] = {}
        for j, kj in enumerate(keys):
            if j != i:
                cents.setdefault(fams[kj], []).append(X[j])
        cf = min(cents, key=lambda f: np.linalg.norm(np.mean(cents[f], axis=0) - X[i]))
        fam_hit = cf == fams[k]
        fam_hits += int(fam_hit)
        if id_hit is not None:
            id_hits += int(id_hit)
            id_eval += 1
        per_subj[k] = {"nn": keys[nn], "family_pred": cf, "family_true": fams[k],
                       "family_hit": fam_hit,
                       "identity_hit": id_hit}
    return {"n": len(keys),
            "identification_top1": (id_hits / id_eval) if id_eval else None,
            "identification_evaluated": id_eval,
            "family_centroid_top1": fam_hits / len(keys), "per_subject": per_subj}


def main():
    subjects = {}
    for f in sorted(BENCH.glob("*.json")):
        if f.name.startswith("fp-"):
            continue
        try:
            subjects[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    fps = {}
    for f in sorted(BENCH.glob("fp-*.json")):
        try:
            fps[f.stem[3:]] = json.loads(f.read_text())
        except Exception:
            pass
    out = {"n_subjects": len(subjects), "n_fingerprints": len(fps),
           "nightvision_lite": nightvision_lite(subjects) if subjects else None,
           "llmmap_lite": llmmap_lite(fps) if fps else None}
    OUT.write_text(json.dumps(out, indent=1))
    nv = out["nightvision_lite"] or {}
    wr = [v for v in (nv.get("width") or {}).values()]
    res = [v for v in wr if v["resolved"]]
    print(f"NightVision-lite width: resolved {len(res)}/{len(wr)} subjects under "
          f"top-5-only access")
    dr = [v for v in (nv.get("depth") or {}).values() if v.get("rel_error") is not None]
    if dr:
        errs = [v["rel_error"] for v in dr]
        print(f"NightVision-lite depth: median rel error {np.median(errs):.3f} "
              f"over {len(dr)} subjects")
    lm = out["llmmap_lite"] or {}
    if lm and "family_centroid_top1" in lm:
        print(f"LLMmap-lite: identification top-1 {lm['identification_top1']:.2f}, "
              f"family top-1 {lm['family_centroid_top1']:.2f} over {lm['n']} subjects")


if __name__ == "__main__":
    sys.exit(0 if main() is None else 1)
