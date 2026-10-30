"""All v4 figures in one clean journal style (no hand-drawn fonts).

Usage: PYTHONIOENCODING=utf-8 python paper/bench/make_v4_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "runs" / "forensics-bench"
BOUT = ROOT / "paper" / "bench" / "out"
FIG = ROOT / "paper" / "figures"

INK = "#25231F"
SOFT = "#777167"
PALE = "#F7F5F1"
ACCENT = "#E8604C"
OK = "#527F65"
MISS = "#B66355"
BLUE = "#4C72B0"
AMBER = "#C49A3A"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "axes.edgecolor": SOFT, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#E3DFD8", "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 8,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
    })


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- accuracy
def fig_accuracy():
    m = json.loads((BOUT / "bench_metrics.json").read_text())
    props = m["properties"]
    flat = m.get("properties_flatprior", {})
    order = [p for p in ("arch.family", "arch.sparsity", "system.topology",
                         "out.logit_precision", "attn.long_range", "attn.kv_sharing",
                         "train.objective") if p in props]
    labels = {"arch.family": "Core family", "arch.sparsity": "FFN sparsity",
              "system.topology": "Topology", "out.logit_precision": "Wire precision",
              "attn.long_range": "Attn/state", "attn.kv_sharing": "KV sharing",
              "train.objective": "Objective sig."}
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(len(order))[::-1]
    ref = [props[p]["top1_accuracy"] for p in order]
    fl = [flat.get(p, {}).get("top1_accuracy", np.nan) for p in order]
    ax.barh(y + 0.19, ref, height=0.36, color=BLUE, label="reference prior")
    ax.barh(y - 0.19, fl, height=0.36, color=ACCENT, label="evidence-only")
    for yi, p in zip(y, order):
        ax.text(1.02, yi, f"n={props[p]['n']}", va="center", fontsize=7.5, color=SOFT)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[p] for p in order])
    ax.set_xlim(0, 1.09)
    ax.set_xlabel("top-1 accuracy vs ground truth")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax.set_title("Reconstruction accuracy on ForensicsBench, two prior views")
    fig.subplots_adjust(bottom=0.26)
    save(fig, "fig_v4_accuracy")


# --------------------------------------------------------------------------- reliability
def fig_reliability():
    m = json.loads((BOUT / "bench_metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, (key, title) in zip(axes, [("properties", "group-aware (primary)"),
                                       ("properties_legacy", "legacy independent")]):
        d = m.get(key)
        if not d:
            continue
        conf, emp = [], []
        for p, v in d.items():
            for b in v.get("reliability", []):
                conf += [b["mean_conf"]] * b["n"]
                emp += [b["empirical"]] * b["n"]
        bins = np.linspace(0, 1, 6)
        bx, by = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [e for c, e in zip(conf, emp) if lo <= c < hi]
            cc = [c for c, e in zip(conf, emp) if lo <= c < hi]
            if sel:
                bx.append(np.mean(cc))
                by.append(np.mean(sel))
        ax.plot([0, 1], [0, 1], color=SOFT, lw=1, ls="--")
        ax.plot(bx, by, "o-", color=BLUE, ms=4)
        ece = np.mean([v["ece"] for v in d.values()])
        ax.set_title(f"{title} (ECE {ece:.2f})")
        ax.set_xlabel("mean confidence")
        ax.set_ylabel("empirical accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Reliability of support scores on ground truth", y=1.03)
    save(fig, "fig_v4_reliability")


# --------------------------------------------------------------------------- geometry
def fig_geometry():
    m = json.loads((BOUT / "bench_metrics.json").read_text())
    rows = [s for s in m["subjects"] if s.get("geometry") and s["geometry"].get("status")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    ws, ests, lbs = [], [], []
    for s in rows:
        w = s["width"]
        g = s["geometry"]
        ws.append(w)
        if g["status"] == "estimated" and g["rank"]:
            ests.append(g["rank"])
            lbs.append(np.nan)
        else:
            ests.append(np.nan)
            lbs.append(g.get("lower_bound") or 0)
    ax.plot([64, 4096], [64, 4096], color=SOFT, ls="--", lw=1, label="perfect")
    ax.scatter(ws, ests, color=OK, zorder=3, label="resolved estimate")
    ax.scatter(ws, lbs, color=MISS, marker="v", zorder=3, label="lower bound (saturated)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true width")
    ax.set_ylabel("recovered rank")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("rich access", fontsize=9)
    ax = axes[1]
    t5 = [((s.get("geometry_top5") or {}).get("dense_columns") or 0, s["width"])
          for s in rows]
    cols = [c for c, _ in t5]
    ws5 = [w for _, w in t5]
    ax.scatter(ws5, cols, color=BLUE)
    ax.axhline(5, color=MISS, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("true width")
    ax.set_ylabel("usable common columns")
    ax.set_title("top-5-only access", fontsize=9)
    fig.subplots_adjust(wspace=0.42)
    fig.suptitle("Output geometry: measured coverage, and the top-k wall", y=1.04)
    save(fig, "fig_v4_geometry")


# --------------------------------------------------------------------------- attn control
def fig_attncontrol():
    d = json.loads((BOUT / "attention_control.json").read_text())
    names = [("full_attention", "full attention (GPT-2)"),
             ("sliding_window_256", "sliding window 256 (Mistral cfg)"),
             ("ssm_mamba", "constant state (Mamba cfg)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
    ax = axes[0]
    for i, (k, lab) in enumerate(names):
        a = d[k]["analysis"]["decode_growth_us_per_1k"]
        ax.errorbar([i], [a["slope"]],
                    yerr=[[a["slope"] - a["ci"][0]], [a["ci"][1] - a["slope"]]],
                    fmt="o", color=[BLUE, AMBER, OK][i], capsize=4, ms=6)
    ax.axhspan(-6, 6, color=PALE)
    ax.axhline(6, color=SOFT, ls=":", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["full", "window", "SSM"], rotation=15)
    ax.set_ylabel("decode growth\n(us/token/1k ctx)")
    ax.set_title("decode-state growth")
    ax = axes[1]
    for i, (k, lab) in enumerate(names):
        pf = d[k]["analysis"]["prefill_fresh"]
        ax.errorbar([i], [pf["exponent"]], yerr=pf["exponent_se"] * 1.96,
                    fmt="s", color=[BLUE, AMBER, OK][i], capsize=4, ms=6)
    ax.axhline(2.0, color=MISS, ls="--", lw=1, label="quadratic")
    ax.axhline(1.0, color=SOFT, ls=":", lw=1, label="linear")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["full", "window", "SSM"], rotation=15)
    ax.set_ylabel("prefill exponent")
    ax.set_ylim(0, 2.3)
    ax.legend(fontsize=7, loc="center left")
    ax.set_title("prefill exponent at <=1k ctx")
    ax = axes[2]
    for i, (k, lab) in enumerate(names):
        pc = d[k]["analysis"]["prefix_cache"]
        ax.bar([i], [pc["ratio"]], color=[BLUE, AMBER, OK][i], width=0.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["full", "window", "SSM"], rotation=15)
    ax.set_ylabel("warm / cold ratio")
    ax.set_title("prefix-cache contrast")
    fig.subplots_adjust(wspace=0.55)
    fig.suptitle("Same-hardware control: what each timing channel can and cannot show", y=1.04)
    save(fig, "fig_v4_attncontrol")


# --------------------------------------------------------------------------- production scaling
def fig_prodscaling():
    d = json.loads((BOUT / "production_followup.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=False)
    fig.subplots_adjust(wspace=0.5)
    for ax, (model, title) in zip(axes, [("DeepSeek-V3.2", "DeepSeek-V3.2"),
                                         ("gpt-4o-mini", "GPT-4o-mini")]):
        rows = d[model]["prefill"]["rows"]
        for kind, color, marker in (("fresh", ACCENT, "o"), ("repeat", BLUE, "s")):
            xs = [r["prompt_tokens"] for r in rows
                  if r.get("kind") == kind and r.get("latency_ms") and r.get("prompt_tokens")]
            ys = [r["latency_ms"] for r in rows
                  if r.get("kind") == kind and r.get("latency_ms") and r.get("prompt_tokens")]
            ax.scatter(xs, ys, s=9, alpha=0.55, color=color, marker=marker,
                       label=f"{kind} prefix")
            fit = d[model]["prefill"][f"{kind}_fit"]
            if fit:
                xs_ = np.linspace(min(xs), max(xs), 64)
                med = np.exp(np.log(xs_) * fit["exponent"]
                             + np.log(np.median(ys) / (np.median(xs) ** fit["exponent"])))
                ax.plot(xs_, med, color=color, lw=1.4,
                        label=f"{kind} fit: exp {fit['exponent']:.2f} "
                              f"[{fit['ci'][0]:.2f},{fit['ci'][1]:.2f}] (n={fit['n']})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("prompt tokens (usage-reported)")
        ax.set_ylabel("latency, ms")
        ax.set_title(title)
        ax.legend(fontsize=6.6, loc="lower right")
        ax.margins(x=0.04)
    fig.suptitle("Production prefill scaling: fresh vs repeated prefix, both fitted", y=1.03)
    save(fig, "fig_v4_prodscaling")


# --------------------------------------------------------------------------- JEV power
def fig_jevpower():
    d = json.loads((BOUT / "jev_followup.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    n = len(d["p_base"])
    xs = np.arange(n)
    ax.scatter(xs - 0.18, d["p_state"], color=OK, label="secret in shared state", s=22)
    ax.scatter(xs, d["p_sibling"], color=MISS, label="secret in sibling question", s=22)
    ax.scatter(xs + 0.18, d["p_base"], color=SOFT, label="no secret (baseline)", s=22)
    ax.axhline(0.5, color=SOFT, ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i+1}" for i in xs], fontsize=7)
    ax.set_ylabel("P(target answers with the secret)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="center left", fontsize=7.5)
    ax.set_title(f"JEV branch isolation, n={n} secrets, randomized order "
                 f"(permutation p={d['perm_p_state_vs_sibling']:.1e})")
    save(fig, "fig_v4_jevpower")


# --------------------------------------------------------------------------- planner
def fig_planner():
    d = json.loads((BOUT / "planner_live.json").read_text())
    agg = d["aggregate"]
    subjects = ["dense", "moe", "ssm", "encoder"]
    planners = ["eig", "static", "uniform"]
    colors = {"eig": ACCENT, "static": BLUE, "uniform": SOFT}
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    w = 0.26
    for si, s in enumerate(subjects):
        for pi, p in enumerate(planners):
            v = agg.get(f"{s}|{p}")
            if not v:
                continue
            x = si + (pi - 1) * w
            val = v["decisive_mean"]
            lo, hi = v["decisive_ci"]
            ax.bar(x, val, width=w * 0.9, color=colors[p],
                   label=p if si == 0 else None)
            ax.errorbar(x, val, yerr=[[val - lo], [hi - val]], color=INK, lw=1,
                        capsize=2)
            if v["decisive_censor_rate"] > 0:
                ax.text(x, val + 6, "cens.", ha="center", fontsize=6.5, color=MISS)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects)
    ax.set_ylabel("requests to decisive family evidence")
    ax.legend(title="planner")
    ax.set_title("Live planner comparison (8 seeds; bars = requests until the decisive\n"
                 "family discriminator's evidence lands; censored at budget 300)")
    save(fig, "fig_v4_planner")


# --------------------------------------------------------------------------- blind fixtures
def fig_blind():
    d = json.loads((BOUT / "blind_fixtures.json").read_text())
    sc = d["scores"]
    order = [("none", "no change\n(specificity)"), ("cap", "cap"), ("ledger", "ledger"),
             ("rounding", "rounding"), ("swap", "swap"), ("router", "router")]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, (k, lab) in enumerate(order):
        stat = sc[k]["specificity"] if k == "none" else sc[k]["sensitivity"]
        ax.bar([i], [stat[0] * 100], color=OK if k != "none" else BLUE, width=0.6)
        ax.errorbar([i], [stat[0] * 100],
                    yerr=[[(stat[0] - stat[1]) * 100], [(stat[2] - stat[0]) * 100]],
                    color=INK, capsize=3, lw=1)
        ax.text(i, stat[0] * 100 + 6, f"n={sc[k]['n']}", ha="center", fontsize=7,
                color=SOFT)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([l for _, l in order], fontsize=7.5)
    ax.set_ylabel("%")
    ax.set_ylim(0, 118)
    ax.set_title("Blind commit-reveal change detection (Wilson 95% CI)")
    save(fig, "fig_v4_blind")


# --------------------------------------------------------------------------- witness
def fig_witness():
    d = json.loads((BOUT / "width_witness.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    ax.bar([0, 1], [d["base_width"], d["twin_declared_width"]],
           color=[BLUE, ACCENT], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["base model", "widened twin"])
    ax.set_ylabel("declared residual width")
    ax.set_title("two different architectures")
    ax = axes[1]
    ax.bar([0, 1], [d["measured_rank_base"], d["measured_rank_twin"]],
           color=[BLUE, ACCENT], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["base model", "widened twin"])
    ax.set_ylabel("scanner's measured rank")
    ax.set_title("identical observations "
                 f"($\\max|\\Delta\\ell|={d['max_abs_logprob_diff']:.0e}$)")
    fig.suptitle("The width witness (Prop. 2), demonstrated", y=1.05)
    save(fig, "fig_v4_witness")


if __name__ == "__main__":
    style()
    import argparse
    only = None
    for fn in (fig_accuracy, fig_reliability, fig_geometry, fig_attncontrol,
               fig_prodscaling, fig_jevpower, fig_planner, fig_blind, fig_witness):
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"skip {fn.__name__}: {exc.filename} missing")
    print("v4 figures pass complete")
