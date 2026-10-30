"""Matplotlib chart helpers -> base64 PNG data URIs, all driven by run data.

Each helper returns a ``data:image/png;base64,...`` string, or ``None`` when the run
does not contain the series it needs (so the report skips it gracefully). PNGs render
on a transparent background with neutral-ink axes so a single image reads on both a
light and a dark page; series colours come from the shared accessible palette.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from forensics.inference.run_types import RunResult

from . import extract as X
from .style import CATEGORICAL_LIGHT, NEUTRAL_INK, GRID_INK, SEQUENTIAL_BLUE

_SEQ_CMAP = LinearSegmentedColormap.from_list("forensic_blue", list(SEQUENTIAL_BLUE))

# Consistent, quieter matplotlib chrome for every chart.
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 130, "savefig.dpi": 130,
    "axes.linewidth": 0.8, "lines.solid_capstyle": "round",
})

# Brand accent for "the measured/updated thing" (posterior, observed series).
ACCENT = "#F04B0D"


# --------------------------------------------------------------------------- infra
def _finish(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _style_axes(ax, grid_axis: str = "both") -> None:
    ax.tick_params(colors=NEUTRAL_INK, labelsize=8, length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_INK)
        ax.spines[spine].set_alpha(0.6)
    ax.xaxis.label.set_color(NEUTRAL_INK)
    ax.yaxis.label.set_color(NEUTRAL_INK)
    ax.title.set_color(NEUTRAL_INK)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_INK, alpha=0.28, linewidth=0.7,
                linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)


def _first_series(run: RunResult, probe_prefix: str, name_contains: tuple[str, ...] = (),
                  ylabel_contains: tuple[str, ...] = ()) -> dict[str, Any] | None:
    for pr in run.probe_results:
        if not pr.probe_id.startswith(probe_prefix):
            continue
        for name, s in (pr.series or {}).items():
            lname = name.lower()
            lyl = str(s.get("ylabel", "")).lower()
            if name_contains and not any(t in lname for t in name_contains):
                if not (ylabel_contains and any(t in lyl for t in ylabel_contains)):
                    continue
            if s.get("x") and s.get("y"):
                return s
    return None


# --------------------------------------------------------------------------- belief bars
_KEY_PROPS = (
    "arch.family", "arch.sparsity", "attn.kv_sharing", "attn.long_range",
    "pos.encoding", "out.logit_precision", "output.head", "modal.input",
    "system.topology", "system.hidden_reasoning",
    "decision.scoring_mechanism", "decision.question_coupling",
    "decision.option_processing",
)


def belief_bars(run: RunResult) -> str | None:
    """Prior vs posterior probability of the winning value, per key property.

    Rendered as a *dumbbell*: a grey connector from the prior (hollow marker) to the
    posterior (filled accent marker) — much easier to scan than paired bars, and the
    direction of the belief update is visible at a glance.
    """
    rows: list[tuple[str, float, float]] = []
    for prop in _KEY_PROPS:
        if not X.has_property(run, prop):
            continue
        val, post = X.posterior_top(run, prop)
        if val is None:
            continue
        prior = X.prior_prob(run, prop, val)
        label = f"{prop.split('.', 1)[-1]} = {X.value_short(val)}"
        rows.append((label, prior, post))
    if not rows:
        return None
    rows = rows[:12]
    labels = [r[0] for r in rows]
    priors = [r[1] for r in rows]
    posts = [r[2] for r in rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.5 * n + 0.9)))
    y = list(range(n))
    for i, (pr, po) in enumerate(zip(priors, posts)):
        ax.plot([pr, po], [i, i], "-", color=GRID_INK, linewidth=2.0, alpha=0.55,
                zorder=3, solid_capstyle="round")
    ax.plot(priors, y, "o", color=GRID_INK, markersize=7, markerfacecolor="white",
            markeredgewidth=1.6, label="prior", zorder=4)
    ax.plot(posts, y, "o", color=ACCENT, markersize=8, label="posterior", zorder=5)
    for i, (pr, po) in enumerate(zip(priors, posts)):
        ax.text(min(po + 0.02, 0.98), i, f"{po*100:.0f}%", va="center", ha="left",
                fontsize=8, fontweight="bold", color=NEUTRAL_INK)
        if pr > 0.02 and abs(po - pr) > 0.03:
            ax.text(max(pr - 0.02, 0.02), i, f"{pr*100:.0f}%", va="center", ha="right",
                    fontsize=7, color=NEUTRAL_INK, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("probability of winning value")
    ax.set_title("Belief update: prior → posterior (winning value per property)",
                 loc="left", pad=12)
    _style_axes(ax, grid_axis="x")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8,
                    framealpha=0.0, ncols=2)
    for t in leg.get_texts():
        t.set_color(NEUTRAL_INK)
    return _finish(fig)


# --------------------------------------------------------------------------- P03 spectrum
def singular_spectrum(run: RunResult) -> str | None:
    """P03 logit singular-value spectrum, if the probe produced one."""
    s = _first_series(run, "P03", name_contains=("sing", "spectr", "sv", "rank"),
                      ylabel_contains=("singular", "sigma"))
    if s is None:
        return None
    y = [float(v) for v in s["y"] if v is not None]
    if not y:
        return None
    y = sorted(y, reverse=True)
    x = list(range(1, len(y) + 1))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(x, y, "-", color=CATEGORICAL_LIGHT[0], linewidth=1.6, zorder=3)
    ax.plot(x, y, "o", color=CATEGORICAL_LIGHT[0], markersize=4, zorder=4)
    # largest multiplicative gap = candidate rank cliff
    if len(y) > 2:
        gaps = [(y[i] / max(y[i + 1], 1e-12), i) for i in range(len(y) - 1)]
        _, ci = max(gaps)
        ax.axvline(ci + 1.5, color=CATEGORICAL_LIGHT[7], linestyle="--", linewidth=1.2,
                   label=f"rank cliff ~ {ci + 1}")
        leg = ax.legend(fontsize=8, framealpha=0.0)
        for t in leg.get_texts():
            t.set_color(NEUTRAL_INK)
    if min(y) > 0:
        ax.set_yscale("log")
    ax.set_xlabel(s.get("xlabel", "index"))
    ax.set_ylabel(s.get("ylabel", "singular value"))
    ax.set_title("P03 logit singular-value spectrum", loc="left")
    _style_axes(ax, grid_axis="y")
    return _finish(fig)


# --------------------------------------------------------------------------- needle heatmap
def needle_heatmap(run: RunResult) -> str | None:
    """P06 needle recall vs (position, length) heatmap from recall_pos_* series."""
    pr = X.probe_result(run, "P06.positional.needle_recall")
    if pr is None:
        for p in run.probe_results:
            if p.probe_id.startswith("P06"):
                pr = p
                break
    if pr is None:
        return None
    series = {k: v for k, v in (pr.series or {}).items() if k.startswith("recall_pos")}
    if not series:
        return None
    positions = sorted(series.keys(), key=lambda k: _pos_of(k))
    # common x grid (union, sorted)
    xset: list[float] = []
    for k in positions:
        for xv in series[k]["x"]:
            if xv not in xset:
                xset.append(float(xv))
    xset = sorted(xset)
    if not xset:
        return None
    mat = []
    for k in positions:
        xmap = {float(a): float(b) for a, b in zip(series[k]["x"], series[k]["y"])}
        mat.append([xmap.get(xv, float("nan")) for xv in xset])
    fig, ax = plt.subplots(figsize=(7.0, max(1.8, 0.5 * len(positions) + 1.2)))
    im = ax.imshow(mat, aspect="auto", cmap=_SEQ_CMAP, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(xset)))
    ax.set_xticklabels([X.fmt_num(v) for v in xset], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels([f"depth {_pos_of(k):.0%}" for k in positions], fontsize=8)
    ax.set_xlabel("prompt tokens")
    ax.set_title("P06 needle-in-haystack recall", loc="left")
    ax.tick_params(colors=NEUTRAL_INK)
    ax.title.set_color(NEUTRAL_INK)
    ax.xaxis.label.set_color(NEUTRAL_INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("recall", color=NEUTRAL_INK, fontsize=8)
    cb.ax.tick_params(colors=NEUTRAL_INK, labelsize=7)
    return _finish(fig)


def _pos_of(name: str) -> float:
    try:
        return float(name.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0.0


# --------------------------------------------------------------------------- latency
def latency_vs_length(run: RunResult) -> str | None:
    """P07 prefill latency vs prompt length (server ms)."""
    s = _first_series(run, "P07", name_contains=("prefill",))
    if s is None:
        s = _first_series(run, "P19", name_contains=("prefill",))
    if s is None:
        return None
    x = [float(v) for v in s["x"]]
    y = [float(v) for v in s["y"]]
    if len(x) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(x, y, "-", color=CATEGORICAL_LIGHT[1], linewidth=1.8, zorder=3)
    ax.plot(x, y, "o", color=CATEGORICAL_LIGHT[1], markersize=5, zorder=4)
    ax.set_xlabel(s.get("xlabel", "prompt tokens"))
    ax.set_ylabel(s.get("ylabel", "server ms"))
    ax.set_title("P07 prefill latency vs prompt length", loc="left")
    _style_axes(ax, grid_axis="y")
    return _finish(fig)


# --------------------------------------------------------------------------- calibration
def calibration(run: RunResult) -> str | None:
    """Calibration / fidelity scatter if a probe produced one (predicted vs actual)."""
    s = None
    for pr in run.probe_results:
        for name, ser in (pr.series or {}).items():
            if any(t in name.lower() for t in ("calib", "fidelity")) and ser.get("x") \
                    and ser.get("y"):
                s = ser
                break
        if s:
            break
    if s is None:
        return None
    x = [float(v) for v in s["x"]]
    y = [float(v) for v in s["y"]]
    if not x:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lo = min(min(x), min(y))
    hi = max(max(x), max(y))
    ax.plot([lo, hi], [lo, hi], "--", color=GRID_INK, linewidth=1.0, zorder=2)
    ax.plot(x, y, "o", color=CATEGORICAL_LIGHT[0], markersize=5, alpha=0.8, zorder=3)
    ax.set_xlabel(s.get("xlabel", "requested"))
    ax.set_ylabel(s.get("ylabel", "observed"))
    ax.set_title("Calibration", loc="left")
    _style_axes(ax)
    return _finish(fig)


# --------------------------------------------------------------------------- all
_CHART_TITLES = {
    "belief_bars": "Prior → posterior beliefs",
    "singular_spectrum": "P03 singular-value spectrum",
    "needle_heatmap": "P06 needle recall heatmap",
    "latency_vs_length": "P07 latency vs length",
    "calibration": "Calibration",
}


def all_charts(run: RunResult) -> dict[str, str]:
    """Build every chart the run supports; skip absent ones. Returns {name: data_uri}."""
    out: dict[str, str] = {}
    for name, fn in (("belief_bars", belief_bars),
                     ("singular_spectrum", singular_spectrum),
                     ("needle_heatmap", needle_heatmap),
                     ("latency_vs_length", latency_vs_length),
                     ("calibration", calibration)):
        try:
            uri = fn(run)
        except Exception:  # noqa: BLE001
            uri = None
        if uri:
            out[name] = uri
    return out


def chart_title(name: str) -> str:
    return _CHART_TITLES.get(name, name)
