"""Compact, readable production architecture cards for v4 (Fig. cards).

One column per endpoint; rows = components.  Bars show support for the leading
candidate; colour = evidence grade; hatched grey = prior-only (no updating evidence).
Data: the recorded production scan JSONs (unchanged) plus benchmark reliability.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
from make_v4_figures import style, INK, SOFT, PALE, ACCENT, OK, MISS, BLUE, AMBER  # noqa

RUNS = {
    "GPT-4o-mini": ROOT / "runs" / "20260926-221747-gpt-4o-mini" / "run_result.json",
    "DeepSeek-V3.2": ROOT / "runs" / "final" / "20260926-210641-deepseek" / "run_result.json",
    "JEV (decision)": ROOT / "runs" / "20260926-221526-jev" / "run_result.json",
}

ROWS = [
    ("tokenizer.family", "tokenizer", "measured"),
    ("arch.family", "core family", None),
    ("arch.sparsity", "FFN sparsity", None),
    ("attn.long_range", "recall/cost regime", None),
    ("out.logit_precision", "wire precision", None),
    ("attn.kv_sharing", "KV sharing", "prior"),
    ("pos.encoding", "position scheme", "prior"),
    ("dims.hidden_size", "width", "U"),
    ("dims.n_layers", "depth", "U"),
]

GRADE_COLOR = {"measured": OK, "prior": "#BBBBBB", "U": MISS}


def main():
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.4), sharey=True)
    for ax, (name, path) in zip(axes, RUNS.items()):
        run = json.loads(path.read_text(encoding="utf-8"))
        post = run.get("posterior", {})
        prior = run.get("prior", {})
        ylabels, ys, cols, tops = [], [], [], []
        # per-endpoint annotations for the attention row (from the powered follow-up;
        # recall-based support says full-range access, the kernel stays unseparated)
        attn_note = {"GPT-4o-mini": "full-range recall\n(kernel open)",
                     "DeepSeek-V3.2": "long-range;\nsub-quad-compatible",
                     "JEV (decision)": None}
        for key, label, kind in ROWS:
            dist = post.get(key) or {}
            if not dist:
                ylabels.append(label)
                ys.append(0.0)
                cols.append(GRADE_COLOR["U"])
                tops.append("U")
                continue
            top_key = max(dist, key=dist.get)
            top = top_key
            if key == "attn.long_range" and attn_note.get(name):
                top = attn_note[name]
            pr = prior.get(key, {})
            changed = any(abs(dist[v] - pr.get(v, 0)) > 1e-5 for v in dist)
            grade = kind or ("measured" if changed else "prior")
            ylabels.append(label)
            ys.append(dist[top_key] if grade != "U" else 0.0)
            cols.append(GRADE_COLOR[grade])
            tops.append(top.replace("_", " ") if grade != "U" else "U")
        y = np.arange(len(ROWS))[::-1]
        ax.barh(y, ys, color=cols, height=0.62, alpha=0.9)
        for yi, t, v in zip(y, tops, ys):
            ax.text(min(v + 0.02, 0.98), yi, t, va="center", fontsize=6.0, color=INK,
                    linespacing=1.15)
        ax.set_xlim(0, 1.02)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlim(0, 1.25)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel("support", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRADE_COLOR[g]) for g in
               ("measured", "prior", "U")]
    fig.legend(handles, ["updated by measurement", "prior-only", "unresolved"],
               loc="lower center", ncol=3, fontsize=7.5)
    fig.suptitle("Production architecture cards: support by component and evidence status",
                 y=1.0, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(FIG / "fig_v4_cards.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_v4_cards.png", dpi=200, bbox_inches="tight")
    print("wrote fig_v4_cards")


if __name__ == "__main__":
    main()
