"""The v4 master figure: endpoint -> channels -> constraints -> identified set ->
four-way boundary (dominant) -> hypothesis graph.  One page, the whole paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
from make_v4_figures import style, INK, SOFT, PALE, ACCENT, OK, MISS, BLUE, AMBER  # noqa


def _lighten(hexcolor, f=0.82):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8, bold=False, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p1, p2, color=SOFT, lw=1.1):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


def main():
    style()
    fig, ax = plt.subplots(figsize=(7.3, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)

    # top: endpoint; forensic layers line sits inside the tall box
    ax.add_patch(FancyBboxPatch((2.55, 8.98), 4.9, 0.9, boxstyle="round,pad=0.012",
                                facecolor=PALE, edgecolor=INK, lw=1.0))
    ax.text(5.0, 9.56, "OPAQUE AI ENDPOINT   $\\omega=(M,S,I,P)$", ha="center",
            va="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(5.0, 9.24, "model · serving · interface · preprocessing", ha="center",
            va="center", fontsize=6.9, color=SOFT)

    # channels
    chans = [("usage\n(accounting)", 0.35), ("logprobs\n(values)", 2.65),
             ("timing\n(streaming)", 4.95), ("behaviour\n(text)", 7.25)]
    for label, x in chans:
        box(ax, x, 7.45, 2.0, 0.75, label, fc="white", fs=7.6)
        arrow(ax, (5.0, 8.98), (x + 1.0, 8.24))
    # constraints row
    cons = [("tokenizer family\ncount match set", 0.35), ("output rank\n(= bottleneck)", 2.65),
            ("decode-state slope\nKV vs constant", 4.95), ("AR path / contracts\n& branch semantics", 7.25)]
    for label, x in cons:
        box(ax, x, 6.15, 2.0, 0.75, label, fc="white", ec=BLUE, fs=7.4)
        arrow(ax, (x + 1.0, 7.45), (x + 1.0, 6.95), color=BLUE)
    # constraints box (wide enough for the label)
    box(ax, 1.15, 5.05, 7.7, 0.62,
        "HARD CONSTRAINTS  $\\mathcal{K}_i$   +   SOFT EVIDENCE (calibrated support)",
        fc=PALE, bold=True, fs=8.4)
    for _, x in cons:
        arrow(ax, (x + 1.0, 6.15), (5.0, 5.7), color=BLUE)
    # identified set
    box(ax, 2.85, 4.12, 4.3, 0.62, "IDENTIFIED SET    $[\\omega]_{\\mathcal{C},\\mathcal{X}}$",
        fc="white", ec=INK, bold=True, fs=9.5)
    arrow(ax, (5.0, 5.05), (5.0, 4.76), color=INK)

    # four-way boundary — the paper's answer, visually dominant
    cats = [("IDENTIFIED", "L1–L3 · recovered now", "contract, tokenizer,\nAR path, width†", 0.35, OK),
            ("COST-LIMITED", "U-cost · priced via $\\hat\\kappa$", "timing channels,\nnoisy grids", 2.8, AMBER),
            ("ACCESS-LIMITED", "U-access · channel hidden", "top-5 width, depth,\nbuffered streams", 5.25, ACCENT),
            ("NON-IDENTIFIABLE", "L5 · proved equivalence", "o200k variants, width\nbeyond logit rank", 7.7, MISS)]
    for title, sub, ex, x, c in cats:
        # dominant boxes: tinted fill, thick colored border, title bold
        ax.add_patch(FancyBboxPatch((x, 1.95), 2.3, 1.75, boxstyle="round,pad=0.014",
                                    facecolor=_lighten(c, 0.86), edgecolor=c, lw=2.0))
        ax.text(x + 1.15, 3.38, title, ha="center", va="center", fontsize=7.9,
                fontweight="bold", color=c)
        ax.text(x + 1.15, 3.05, sub, ha="center", va="center", fontsize=6.6, color=INK)
        ax.text(x + 1.15, 2.5, ex, ha="center", va="center", fontsize=6.6, color=INK,
                linespacing=1.3)
        arrow(ax, (5.0, 4.12), (x + 1.15, 3.72), color=c, lw=1.5)
    # hypothesis graph
    box(ax, 2.0, 0.62, 6.0, 0.8,
        "ARCHITECTURE HYPOTHESIS GRAPH  $G_A$\ncomponents · candidates · grades · next experiment",
        fc=PALE, bold=True, fs=8.4)
    for _, _, _, x, c in cats:
        arrow(ax, (x + 1.15, 1.95), (min(max(x + 1.15, 2.3), 7.7), 1.44), color=SOFT)
    ax.text(5.0, 0.24, "API surface determines the observable architecture",
            ha="center", fontsize=8.8, color=INK, style="italic")
    fig.savefig(FIG / "fig_v4_masterflow.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_v4_masterflow.png", dpi=220, bbox_inches="tight")
    print("wrote fig_v4_masterflow")


if __name__ == "__main__":
    main()
