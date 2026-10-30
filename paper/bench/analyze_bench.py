"""Aggregate ForensicsBench subject scans into benchmark tables.

For every subject we know the ground truth (recorded from the loaded config) and the
scanner's posterior.  This script computes, per property:

* top-1 accuracy and the full confusion structure;
* calibration of the support scores actually shown in the paper: Brier score,
  expected calibration error (ECE), and a reliability table;
* geometry coverage by width (resolved estimate vs lower bound) and rank error;
* the transformer-vs-SSM discriminator confusion (P23).

Outputs ``paper/bench/out/bench_metrics.json`` and LaTeX tables under
``paper/generated/`` (bench_*.tex), plus a reliability figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "runs" / "forensics-bench"
OUT = Path(__file__).resolve().parent / "out"
GEN = ROOT / "paper" / "generated"

# ground-truth value per property, derived from the recorded gt record
def gt_value(subj: dict, prop: str):
    gt = subj.get("ground_truth", {})
    arch = gt.get("arch")
    if prop == "arch.family":
        if gt.get("router_twin"):
            return None                      # family is per-model; router is a topology subject
        fam = gt.get("family")
        return fam
    if prop == "arch.sparsity":
        return "moe" if arch == "moe" else "dense"
    if prop == "system.topology":
        return "router_or_cascade" if gt.get("router_twin") else "single_model"
    if prop == "out.logit_precision":
        opts = gt.get("opts") or {}
        if opts.get("wire_round"):
            return opts["wire_round"]
        if opts.get("quant"):
            return "fp32"                    # int8 weights -> near-continuous logit grid
        return opts.get("compute_dtype", "fp32")
    if prop == "attn.long_range":
        if arch == "masked":
            return None
        if gt.get("family") == "ssm_or_linear":
            return "linear_recurrent"
        return "full_quadratic"
    if prop == "attn.kv_sharing":
        if arch in ("masked", "seq2seq") or gt.get("family") == "ssm_or_linear":
            return None
        if arch == "mla":
            return "mla"
        kv, h = gt.get("kv_heads", -1), gt.get("heads", -1)
        if kv <= 0 or h <= 0:
            return None
        if kv == h:
            return "mha"
        return "gqa_or_mqa"
    if prop == "train.objective":
        return {"masked": "masked_lm", "seq2seq": "prefix_lm_or_span"}.get(arch, "causal_lm")
    return None


def _ece_brier(rows: list[tuple[dict, str]], n_bins: int = 5):
    """(posterior, true_value) rows -> brier, ece, reliability bins."""
    briers, bins = [], [[] for _ in range(n_bins)]
    for dist, true in rows:
        vals = sorted(dist)
        y = {v: float(v == true) for v in vals}
        briers.append(sum((dist[v] - y[v]) ** 2 for v in vals))
        top = max(dist, key=dist.get)
        conf, hit = dist[top], float(top == true)
        b = min(n_bins - 1, int(conf * n_bins))
        bins[b].append((conf, hit))
    rel = []
    for i, b in enumerate(bins):
        if b:
            rel.append({"bin": i, "n": len(b),
                        "mean_conf": float(np.mean([c for c, _ in b])),
                        "empirical": float(np.mean([h for _, h in b]))})
    ece = float(sum(r["n"] * abs(r["mean_conf"] - r["empirical"]) for r in rel)
                / max(1, sum(r["n"] for r in rel)))
    return float(np.mean(briers)), ece, rel


PROPS = ["arch.family", "arch.sparsity", "system.topology", "out.logit_precision",
         "attn.long_range", "attn.kv_sharing", "train.objective"]


def cluster_of(key: str) -> str:
    """Checkpoint-level cluster: serving variants of one checkpoint are one cluster."""
    if key.startswith("gpt2"):
        return "gpt2"
    return key


def _cluster_bootstrap_ci(keyed_rows, B: int = 5000, seed: int = 42):
    """Percentile CI for top-1 accuracy, resampling checkpoint clusters."""
    clusters: dict[str, list] = {}
    for key, dist, true in keyed_rows:
        clusters.setdefault(cluster_of(key), []).append((dist, true))
    names = sorted(clusters)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(B):
        draw = rng.choice(names, size=len(names), replace=True)
        hits = [max(d, key=d.get) == t for c in draw for d, t in clusters[c]]
        accs.append(np.mean(hits))
    return [round(float(np.quantile(accs, 0.025)), 3),
            round(float(np.quantile(accs, 0.975)), 3)]


def _macro_bootstrap(view: dict, B: int = 5000, seed: int = 7):
    """Bootstrap CI over properties for the macro-averaged ECE/Brier."""
    props = sorted(view)
    rng = np.random.default_rng(seed)
    eces, briers = [], []
    for _ in range(B):
        draw = rng.choice(props, size=len(props), replace=True)
        eces.append(np.mean([view[p]["ece"] for p in draw]))
        briers.append(np.mean([view[p]["brier"] for p in draw]))
    q = lambda a: [round(float(np.quantile(a, 0.025)), 3),
                   round(float(np.quantile(a, 0.975)), 3)]
    return {"ece": round(float(np.mean([view[p]["ece"] for p in props])), 3),
            "ece_ci": q(eces),
            "brier": round(float(np.mean([view[p]["brier"] for p in props])), 3),
            "brier_ci": q(briers)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from battery import battery_protocol_hash
    proto = battery_protocol_hash()
    subs = sorted(BENCH.glob("*.json"))
    results = []
    skipped = []
    for f in subs:
        if f.name.startswith("fp-"):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("protocol") != proto:
            skipped.append((f.name, d.get("protocol", "")[:8]))
            continue                       # only the frozen current protocol is scored
        results.append(d)
    if skipped:
        print(f"protocol filter: skipped {len(skipped)} stale files {skipped}")
    per_prop_keyed: dict[str, list] = {p: [] for p in PROPS}
    per_prop: dict[str, list] = {p: [] for p in PROPS}
    per_prop_legacy: dict[str, list] = {p: [] for p in PROPS}
    per_prop_flat: dict[str, list] = {p: [] for p in PROPS}
    rows_table = []
    for r in results:
        scan = r.get("scan") or {}
        post = scan.get("posterior") or {}
        post_leg = scan.get("posterior_legacy") or {}
        post_flat = scan.get("posterior_flatprior") or {}
        gt = r.get("ground_truth", {})
        row = {"key": r["subject"], "family": gt.get("family"), "arch": gt.get("arch"),
               "width": gt.get("width")}
        for p in PROPS:
            true = gt_value(r, p)
            dist = post.get(p)
            if true is None or not dist:
                row[p] = None
                continue
            top = max(dist, key=dist.get)
            row[p] = {"top": top, "conf": round(dist[top], 3), "true": true,
                      "hit": top == true}
            per_prop[p].append((dist, true))
            per_prop_keyed[p].append((r["subject"], dist, true))
            dl = post_leg.get(p)
            if dl:
                per_prop_legacy[p].append((dl, true))
            df = post_flat.get(p)
            if df:
                per_prop_flat[p].append((df, true))
        # geometry
        geo = (r.get("modules") or {}).get("geometry_columns") or {}
        fin = geo.get("final") or {}
        row["geometry"] = {"status": fin.get("status"), "rank": fin.get("rank"),
                           "lower_bound": fin.get("lower_bound"),
                           "rows": fin.get("rows")}
        t5 = (r.get("modules") or {}).get("geometry_top5") or {}
        row["geometry_top5"] = {"status": t5.get("status"),
                                "rank": t5.get("rank_estimate"),
                                "lower_bound": t5.get("observed_rank_lower_bound"),
                                "dense_columns": t5.get("dense_columns")}
        results_row = row
        rows_table.append(results_row)

    metrics = {"n_subjects": len(results), "properties": {}, "properties_legacy": {}}
    for p, rows in per_prop.items():
        if not rows:
            continue
        acc = float(np.mean([max(d, key=d.get) == t for d, t in rows]))
        brier, ece, rel = _ece_brier(rows)
        confs = {}
        for d, t in rows:
            top = max(d, key=d.get)
            confs.setdefault(f"{top}<-{t}", 0)
            confs[f"{top}<-{t}"] += 1
        metrics["properties"][p] = {"n": len(rows), "top1_accuracy": round(acc, 3),
                                    "top1_ci_cluster": _cluster_bootstrap_ci(
                                        per_prop_keyed[p]),
                                    "brier": round(brier, 4), "ece": round(ece, 4),
                                    "reliability": rel, "confusion": confs}
    for p, rows in per_prop_legacy.items():
        if not rows:
            continue
        acc = float(np.mean([max(d, key=d.get) == t for d, t in rows]))
        brier, ece, rel = _ece_brier(rows)
        metrics["properties_legacy"][p] = {"n": len(rows), "top1_accuracy": round(acc, 3),
                                           "brier": round(brier, 4), "ece": round(ece, 4),
                                           "reliability": rel}
    metrics["clusters"] = {cluster_of(r["subject"]): None for r in results}
    metrics["n_clusters"] = len(metrics["clusters"])
    metrics["macro"] = {"primary": _macro_bootstrap(metrics["properties"]),
                        "legacy": _macro_bootstrap(metrics["properties_legacy"])}
    metrics["properties_flatprior"] = {}
    for p, rows in per_prop_flat.items():
        if not rows:
            continue
        acc = float(np.mean([max(d, key=d.get) == t for d, t in rows]))
        brier, ece, rel = _ece_brier(rows)
        metrics["properties_flatprior"][p] = {"n": len(rows), "top1_accuracy": round(acc, 3),
                                              "brier": round(brier, 4), "ece": round(ece, 4)}

    # geometry coverage by width
    geo_rows = [(r["width"], r["geometry"]) for r in rows_table
                if r.get("width") and r.get("geometry") and r["geometry"].get("status")]
    cov = {}
    for w, g in sorted(geo_rows, key=lambda t: t[0]):
        cov.setdefault(str(w), {"n": 0, "resolved": 0, "errors": []})
        cov[str(w)]["n"] += 1
        if g["status"] == "estimated" and g["rank"]:
            cov[str(w)]["resolved"] += 1
            cov[str(w)]["errors"].append(abs(g["rank"] - w) / w)
    metrics["geometry_coverage"] = cov
    metrics["subjects"] = rows_table
    metrics["macro"]["flat"] = _macro_bootstrap(metrics["properties_flatprior"])

    OUT.mkdir(exist_ok=True)
    (OUT / "bench_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({p: {"n": m["n"], "acc": m["top1_accuracy"],
                          "brier": m["brier"], "ece": m["ece"]}
                      for p, m in metrics["properties"].items()}, indent=1))
    print("geometry coverage:", json.dumps(cov))
    return metrics


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
