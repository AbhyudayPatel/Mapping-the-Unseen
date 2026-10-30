"""Hand-rendered SVG figures — the flagship architecture diagram and the serving
topology pipeline.

Why not Mermaid/Graphviz here? The architecture figure needs precise, deterministic
layout: an expanded transformer block with residual bypass arcs, tensor annotations on
the arrows, per-component *reasons* ("which experiment supports this box") and explicit
"not identifiable from this API" placeholders. That is a drawing, not a graph layout.

Everything is still a pure function of the :class:`RunResult` — no hard-coded
architectures. If a quantity was not measured, the figure says so, in grey dashed
strokes, with the *reason it cannot be known* from the identifiability analysis.

The SVG is dependency-free (no CDN, works offline), theme-aware (it references the
host page's CSS custom properties with light fallbacks, so it renders correctly in the
report's light and dark themes and acceptably as a standalone file), and interactive:
groups carry ``data-node`` attributes matching :class:`DiagramNode` ids so the report
inspector binds clicks exactly like it does for Mermaid nodes.
"""
from __future__ import annotations

import html as _h
from typing import Any

from forensics.inference.run_types import RunResult

from . import extract as X
from .style import canonical_level, confidence_marker, ident_confidence, level_style

# --------------------------------------------------------------------------- theme
# Level colours via CSS custom properties (defined by the report/web page); the hex
# fallback after the comma is the light-theme value used for standalone .svg files.
_LV_HEX = {"observed": "#008300", "inferred": "#2a78d6", "estimated": "#c98a00",
           "speculative": "#e34948", "unknown": "#9a968c"}

_SVG_CSS = """
.fg-t{font:650 12.5px system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;fill:var(--ink,#14120e)}
.fg-tt{font:700 15px system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;fill:var(--ink,#14120e)}
.fg-sub{font:11px system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;fill:var(--ink2,#55524a)}
.fg-v{font:11px ui-monospace,'SF Mono',Consolas,monospace;fill:var(--ink2,#55524a)}
.fg-w{font:9.5px ui-monospace,'SF Mono',Consolas,monospace;fill:var(--muted,#8c877c)}
.fg-g{font:600 8.5px ui-monospace,'SF Mono',Consolas,monospace;letter-spacing:.1em}
.fg-e{stroke:var(--ink2,#55524a);stroke-width:1.6;fill:none}
.fg-ea{stroke:var(--muted,#8c877c);stroke-width:1.4;fill:none;stroke-dasharray:5 4}
.fg-el{font:9px ui-monospace,'SF Mono',Consolas,monospace;fill:var(--muted,#8c877c)}
.fg-box{stroke-width:2;fill:color-mix(in srgb,var(--fg-lc) 9%,var(--fg-surface,#ffffff))}
.fg-node{cursor:pointer}
.fg-node:hover .fg-box{stroke-width:3}
.fg-rail{fill:var(--surface2,#f4f2ed);stroke:var(--border,#e8e5dd);stroke-width:1}
.fg-res{stroke:var(--muted,#8c877c);stroke-width:1.5;fill:none;stroke-dasharray:2 3}
.lv-observed{--fg-lc:var(--observed,#008300)}
.lv-inferred{--fg-lc:var(--inferred,#2a78d6)}
.lv-estimated{--fg-lc:var(--estimated,#c98a00)}
.lv-speculative{--fg-lc:var(--speculative,#e34948)}
.lv-unknown{--fg-lc:var(--unknown,#9a968c)}
.ln-dashed{stroke-dasharray:7 4}
.ln-dotted{stroke-dasharray:2 4}
"""

_CHAR_W = {"fg-t": 6.9, "fg-v": 6.62, "fg-w": 5.72, "fg-sub": 5.6, "fg-g": 5.6}


# --------------------------------------------------------------------------- builder
class _Svg:
    """Tiny imperative SVG builder with a y-cursor."""

    def __init__(self, width: int) -> None:
        self.w = width
        self.parts: list[str] = []
        self.defs: list[str] = []

    def raw(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x: float, y: float, w: float, h: float, rx: float = 10,
             cls: str = "", style: str = "", fill: str = "", stroke: str = "",
             dash: str = "", sw: float = 0) -> None:
        a = f'x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"'
        if cls:
            a += f' class="{cls}"'
        if style:
            a += f' style="{style}"'
        if fill:
            a += f' fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if sw:
            a += f' stroke-width="{sw}"'
        self.parts.append(f"<rect {a}/>")

    def text(self, x: float, y: float, s: str, cls: str = "fg-v",
             anchor: str = "middle", style: str = "") -> None:
        st = f' style="{style}"' if style else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{st}>'
            f'{_h.escape(str(s))}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str = "fg-e",
             marker: bool = True, style: str = "") -> None:
        m = ' marker-end="url(#fg-arr)"' if marker else ""
        st = f' style="{style}"' if style else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="{cls}"{m}{st}/>')

    def path(self, d: str, cls: str = "fg-e", marker: bool = True,
             style: str = "") -> None:
        m = ' marker-end="url(#fg-arr)"' if marker else ""
        st = f' style="{style}"' if style else ""
        self.parts.append(f'<path d="{d}" class="{cls}"{m}{st}/>')

    def circle(self, cx: float, cy: float, r: float, style: str = "",
               fill: str = "") -> None:
        st = f' style="{style}"' if style else ""
        f = f' fill="{fill}"' if fill else ""
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"{st}{f}/>')

    def render(self, height: float) -> str:
        defs = ('<defs><marker id="fg-arr" viewBox="0 0 10 10" refX="8.5" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted,#8c877c)"/>'
                "</marker></defs>")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} '
                f'{height:.0f}" width="{self.w}" role="img" aria-label="architecture '
                f'figure"><style>{_SVG_CSS}</style>{defs}' + "".join(self.parts) + "</svg>")


def _wrap(text: str, cls: str, max_w: float) -> list[str]:
    """Greedy word wrap using a per-class average character width."""
    text = " ".join(str(text).split())
    if not text:
        return []
    cw = _CHAR_W.get(cls, 6.2)
    limit = max(8, int(max_w / cw))
    out, cur = [], ""
    for word in text.split(" "):
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= limit:
            cur += " " + word
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- "why" text
def _pid_short(pid: str) -> str:
    return (pid or "").split(".", 1)[0]


def why_for_property(run: RunResult, prop: str, max_chars: int = 240) -> str | None:
    """Strongest evidence for a property -> 'P16 · refusal boundary invariant to …'."""
    evs = X.evidence_for_property(run, prop)
    if not evs:
        return None
    ev = max(evs, key=lambda e: e.strength or 0.0)
    s = f"{_pid_short(ev.probe_id)} · {ev.summary}"
    return s[:max_chars - 1] + "…" if len(s) > max_chars else s


def why_for_quantity(run: RunResult, q: str, max_chars: int = 240) -> str | None:
    """Best estimate's provenance -> 'P03 · centered SVD of position-0 logits'."""
    e = X.best_estimate(run, q)
    if e is None:
        return None
    s = f"{_pid_short(e.probe_id)} · {e.method}" if e.method else _pid_short(e.probe_id)
    return s[:max_chars - 1] + "…" if len(s) > max_chars else s


def why_unknown(run: RunResult, key: str, max_chars: int = 110) -> str:
    """The reason a quantity is *not* knowable, from the identifiability analysis."""
    ie = X.identifiability_for(run, key)
    if ie and ie.reason:
        s = ie.reason
        return s[:max_chars - 1] + "…" if len(s) > max_chars else s
    return "no probe in budget produced a measurable signature"


# --------------------------------------------------------------------------- pieces
def _node_group(svg: _Svg, node_id: str | None, known_ids: set[str]) -> str:
    """Opening <g> for a component; carries data-node when the inspector knows it."""
    if node_id and node_id in known_ids:
        return f'<g class="fg-node" data-node="{_h.escape(node_id)}">'
    return "<g>"


def _level_tag(level: str, conf: float | None) -> str:
    lv = canonical_level(level)
    if lv == "unknown":
        return "UNRESOLVED (U)"
    mark = confidence_marker(conf)
    pct = f" · {conf * 100:.0f}%" if conf is not None else ""
    return f"{mark} {lv.upper()}{pct}"


def _component(svg: _Svg, cx: float, y: float, w: float, *, node_id: str | None,
               known: set[str], title: str, values: list[str], why: str | None,
               level: str, conf: float | None, rx: float = 10, why_max: int = 2) -> float:
    """Draw one component card centred at cx; return its bottom y."""
    pad = 12
    line_h = {"title": 18, "value": 15.5, "why": 13.5, "tag": 16}
    v_lines: list[str] = []
    for v in values:
        v_lines.extend(_wrap(v, "fg-v", w - 2 * pad))
    w_lines = _wrap(why, "fg-w", w - 2 * pad) if why else []
    if len(w_lines) > why_max:                      # truncate with an explicit ellipsis
        w_lines = w_lines[:why_max]
        w_lines[-1] = w_lines[-1].rstrip(" .;,|") + " …"
    h = (pad + line_h["title"] + 2 + len(v_lines) * line_h["value"]
         + (len(w_lines) * line_h["why"] + 4 if w_lines else 0)
         + line_h["tag"] + pad - 6)
    lv = canonical_level(level)
    ls = level_style(lv)
    dash = {"solid": "", "dashed": "ln-dashed", "dotted": "ln-dotted"}[ls.line]
    svg.raw(_node_group(svg, node_id, known))
    svg.rect(cx - w / 2, y, w, h, rx=rx,
             cls=f"fg-box lv-{lv}" + (f" {dash}" if dash else ""))
    ty = y + pad + 12
    svg.text(cx, ty, title, "fg-t"); ty += 4
    for v in v_lines:
        ty += line_h["value"] - 2
        svg.text(cx, ty, v, "fg-v")
    if w_lines:
        ty += 5
        for ln in w_lines:
            ty += line_h["why"] - 3
            svg.text(cx, ty, ln, "fg-w")
    ty += line_h["tag"]
    svg.text(cx, ty, _level_tag(level, conf), "fg-g",
             style=f"fill:var(--{lv},{_LV_HEX[lv]})")
    svg.raw("</g>")
    return y + h


def _arrow(svg: _Svg, x: float, y0: float, y1: float, label: str = "") -> None:
    svg.line(x, y0 + 2, x, y1 - 2)
    if label:
        svg.text(x + 8, (y0 + y1) / 2 + 3, label, "fg-el", anchor="start")


def _residual(svg: _Svg, x_box_left: float, y_from: float, y_to: float) -> None:
    """Curved bypass arc on the left of a sub-block, with a ⊕ junction."""
    x_arc = x_box_left - 24
    mid = (y_from + y_to) / 2
    svg.path(f"M {x_box_left:.1f} {y_from:.1f} C {x_arc:.1f} {y_from:.1f}, "
             f"{x_arc:.1f} {y_to:.1f}, {x_box_left:.1f} {y_to:.1f}",
             cls="fg-res", marker=False)
    svg.circle(x_arc - 1, mid, 7, style="fill:var(--surface,#fff);stroke:var(--muted,#8c877c);"
                                        "stroke-width:1.3")
    svg.text(x_arc - 1, mid + 3.5, "+", "fg-el")


def _qval(run: RunResult, q: str) -> tuple[str, str, float | None, str]:
    """(formatted value+CI, level, confidence, probe-method-why) for a quantity."""
    e = X.best_estimate(run, q)
    if e is None or e.value is None:
        return X.NOT_IDENTIFIED, "unknown", None, why_unknown(run, q)
    s = X.fmt_value(e.value, X.quantity_unit(q))
    ci = X.fmt_ci(e.ci, e.value)
    if ci:
        s += f"  (90% CI {ci})"
    return s, e.level, ident_confidence(getattr(e, "identifiability", None)), \
        why_for_quantity(run, q)


# =========================================================================== ARCHITECTURE
def architecture_figure(run: RunResult, known_ids: set[str] | None = None) -> str:
    """The Raschka-style measured-architecture figure, built from the run's numbers."""
    known = known_ids or set()
    kind = run.caps.kind
    mm = X.detect_multimodal(run)
    W = 920
    CX = 400.0              # main column centre
    BW = 340.0              # main box width
    RAIL_X = 762.0          # right rail centre
    MM_CX = 130.0           # multimodal lane centre
    svg = _Svg(W)

    n_req = (run.budget_summary or {}).get("requests")
    name = str(run.target.get("name") or run.run_id)

    # ---- header ---------------------------------------------------------------
    y = 26
    svg.text(20, y, f"Measured architecture — {name}", "fg-tt", anchor="start"); y += 18
    svg.text(20, y, f"Reverse-engineered from {n_req or '?'} API requests. Every box: "
                    "the inferred value · the experiment behind it · how sure we are.",
             "fg-sub", anchor="start"); y += 15
    svg.text(20, y, "Grey dashed = not identifiable from this interface (with the reason).",
             "fg-sub", anchor="start")
    y += 20

    # ---- right rail: wire-level measurements ----------------------------------
    rail_items = _rail_items(run)
    rail_top = y
    rail_h = 0
    if rail_items:
        rail_h = 30 + len(rail_items[:8]) * 46 + 14
        svg.rect(RAIL_X - 116, rail_top, 232, rail_h, rx=12, cls="fg-rail")
        svg.text(RAIL_X, rail_top + 20, "ALSO MEASURED FROM THE WIRE", "fg-g",
                 style="fill:var(--muted,#8c877c)")
        ry = rail_top + 30
        for label, val, pid, lvl in rail_items[:8]:
            ry += 8
            svg.circle(RAIL_X - 100, ry + 7, 3.5, fill=_LV_HEX[lvl])
            svg.text(RAIL_X - 90, ry + 4, label, "fg-g", anchor="start",
                     style="fill:var(--muted,#8c877c)")
            svg.text(RAIL_X - 90, ry + 18, val, "fg-v", anchor="start")
            svg.text(RAIL_X + 106, ry + 4, pid, "fg-w", anchor="end")
            ry += 38

    # ---- input ----------------------------------------------------------------
    in_label = "Structured question" if kind == "decision" else "Input text"
    in_note = ("a prompt with labelled options" if kind == "decision"
               else "what we send to the endpoint")
    y = _component(svg, CX, y, 240, node_id="b_in", known=known, title=in_label,
                   values=[in_note], why=None, level="observed", conf=1.0, rx=20)
    _arrow(svg, CX, y, y + 30)
    y += 30

    # ---- multimodal lane (left) ------------------------------------------------
    mm_join_y = None
    if mm["image"] or mm["audio"]:
        lc = _prop_lc(run, "modal.input")
        mods = "+".join(m for m in ("image", "audio") if mm[m])
        my = rail_top if rail_items else y - 30
        my = _component(svg, MM_CX, my + 30, 200, node_id="b_mm_in", known=known,
                        title=f"{mods.title()} input", values=[X.value_short(mm["input"])],
                        why=why_for_property(run, "modal.input"),
                        level=lc[0], conf=lc[1], rx=20)
        if mm["image"]:
            velc = _prop_lc(run, "modal.vision_encoder")
            ve = {"vit_patch_tokens": "ViT → patch tokens",
                  "native_resolution_vit": "native-res tiling → tokens",
                  "cross_attn_adapter": "cross-attention adapter"}.get(str(mm["vision_encoder"]),
                                                                       "vision pathway")
            _arrow(svg, MM_CX, my, my + 26)
            my = _component(svg, MM_CX, my + 26, 200, node_id="b_vision", known=known,
                            title="Vision encoder", values=[ve],
                            why=why_for_property(run, "modal.vision_encoder"),
                            level=velc[0], conf=velc[1])
        mm_join_y = my

    # ---- tokenizer --------------------------------------------------------------
    tf = X.best_estimate(run, "tokenizer.family")
    tt = X.best_estimate(run, "tokenizer.type")
    tlc = ("unknown", None)
    tok_val = X.NOT_IDENTIFIED
    tok_why = why_unknown(run, "tokenizer.family")
    if tf and tf.value is not None:
        tlc = (tf.level, ident_confidence(getattr(tf, "identifiability", None)))
        tok_val = {"unknown_custom": "custom tokenizer (no public match)"}.get(
            str(tf.value), X.fmt_value(tf.value))
        if tt and tt.value:
            tok_val += f"  ·  {X.fmt_value(tt.value)}"
        tok_why = why_for_quantity(run, "tokenizer.family")
    oh = X.best_estimate(run, "tokenizer.chat_template_overhead")
    tok_extra = []
    if oh and oh.value is not None:
        tok_extra.append(f"chat overhead: {X.fmt_value(oh.value)} tokens")
    tok_bot = _component(svg, CX, y, BW, node_id="b_tok", known=known, title="Tokenizer",
                         values=[tok_val] + tok_extra, why=tok_why,
                         level=tlc[0], conf=tlc[1])
    _arrow(svg, CX, tok_bot, tok_bot + 30, label="token ids  ·  n counted server-side")
    y = tok_bot + 30

    # ---- embedding + positional -------------------------------------------------
    v_val, v_lvl, v_conf, _ = _qval(run, "dims.vocab_size")
    d_val, d_lvl, d_conf, d_why = _qval(run, "dims.hidden_size")
    tied = X.est_value(run, "dims.embedding_tied")
    emb_vals = [f"vocab: {v_val.split('  ')[0]}"]
    if d_val != X.NOT_IDENTIFIED:
        emb_vals.append(f"d_model: {d_val}")
    else:
        emb_vals.append("d_model: not identifiable")
    if tied is not None:
        emb_vals.append(f"tied with head: {X.fmt_value(tied)}")
    emb_why_parts = [w for w in (why_for_quantity(run, "dims.vocab_size"),) if w]
    if d_val == X.NOT_IDENTIFIED and d_why:
        emb_why_parts.append("d_model: " + d_why)
    emb_why = "  |  ".join(emb_why_parts) or None
    emb_lvl = v_lvl if d_val == X.NOT_IDENTIFIED else d_lvl
    emb_top = y
    y = _component(svg, CX, y, BW, node_id="b_embed", known=known,
                   title="Token embedding", values=emb_vals, why=emb_why,
                   level=emb_lvl, conf=min(c for c in (v_conf, d_conf) if c is not None)
                   if (v_conf or d_conf) else None, why_max=3)
    if mm_join_y is not None:
        # the fused multimodal token lane joins the sequence at the embedding
        svg.path(f"M {MM_CX + 100:.1f} {mm_join_y - 20:.1f} C {CX - BW / 2 - 70:.1f} "
                 f"{mm_join_y - 20:.1f}, {CX - BW / 2 - 50:.1f} {emb_top + 22:.1f}, "
                 f"{CX - BW / 2:.1f} {emb_top + 22:.1f}", cls="fg-ea")
        svg.text((MM_CX + 100 + CX - BW / 2) / 2 + 30, min(mm_join_y - 30, emb_top + 10),
                 "fused tokens", "fg-el", anchor="middle")
    _arrow(svg, CX, y, y + 30)
    y += 30

    # ---- transformer block container --------------------------------------------
    n_layers = X.best_estimate(run, "dims.n_layers")
    if n_layers and n_layers.value is not None:
        n_lbl = X.fmt_value(n_layers.value)
        n_ci = X.fmt_ci(n_layers.ci, n_layers.value)
        block_title = f"Transformer block  ×  {n_lbl}"
        block_sub = f"depth {n_lbl}{f' (90% CI {n_ci})' if n_ci else ''} · " \
                    f"{why_for_quantity(run, 'dims.n_layers') or ''}"
    else:
        block_title = "Transformer block  ×  N"
        block_sub = "N not identifiable: per-layer work never surfaces on the wire"

    inner_top = y + 34
    iy = inner_top
    box_w = BW - 36
    # norm 1 (assumed)
    n1_bot = _component(svg, CX, iy, box_w, node_id="b_norm", known=known,
                        title="Norm", values=["RMSNorm · pre-norm"],
                        why="assumed: norm type has no black-box I/O signature",
                        level="speculative", conf=0.25)
    # attention
    kv_val, _ = X.posterior_top(run, "attn.kv_sharing")
    attn_name = {"mha": "Multi-head attention (MHA)",
                 "gqa_or_mqa": "Grouped-query attention (GQA/MQA)",
                 "mla": "Multi-head latent attention (MLA)"}.get(str(kv_val), "Attention")
    klc = _prop_lc(run, "attn.kv_sharing")
    head_bits = []
    for q, lbl in (("dims.n_heads", "heads"), ("dims.n_kv_heads", "KV heads"),
                   ("dims.head_dim", "head_dim")):
        e = X.best_estimate(run, q)
        if e and e.value is not None:
            head_bits.append(f"{X.fmt_value(e.value)} {lbl}")
    lr_val, _ = X.posterior_top(run, "attn.long_range")
    pos_val, _ = X.posterior_top(run, "pos.encoding")
    attn_vals = [", ".join(head_bits) if head_bits else "head split: not identifiable"]
    if pos_val:
        attn_vals.append(f"positions: {X.value_short(pos_val)} applied to Q/K")
    if lr_val:
        attn_vals.append(f"long-range: {X.value_short(lr_val)}")
    attn_why_parts = []
    for w in (why_for_property(run, "attn.kv_sharing"),
              why_for_property(run, "attn.long_range"),
              why_for_property(run, "pos.encoding")):
        if w and all(w.split("·")[0] != p.split("·")[0] for p in attn_why_parts):
            attn_why_parts.append(w)   # one reason per probe, no duplicates
    attn_why = "  |  ".join(attn_why_parts) or None
    attn_bot = _component(svg, CX, n1_bot + 14, box_w, node_id="b_attn", known=known,
                          title=attn_name, values=attn_vals, why=attn_why,
                          level=klc[0], conf=klc[1], why_max=3)
    _residual(svg, CX - box_w / 2, iy + 14, attn_bot - 12)
    # norm 2 + FFN
    n2_bot = _component(svg, CX, attn_bot + 14, box_w, node_id=None, known=known,
                        title="Norm", values=["RMSNorm · pre-norm"],
                        why="assumed, as above", level="speculative", conf=0.25)
    spars_val, _ = X.posterior_top(run, "arch.sparsity")
    slc = _prop_lc(run, "arch.sparsity")
    if str(spars_val) == "moe":
        ne, ae = X.best_estimate(run, "dims.n_experts"), X.best_estimate(run, "dims.active_experts")
        ffn_title = "MoE feed-forward"
        exp = (f"{X.fmt_value(ne.value)} experts" if ne and ne.value is not None
               else "expert count: not identifiable")
        if ae and ae.value is not None:
            exp += f", top-{X.fmt_value(ae.value)} active"
        ffn_vals = [exp, _qval(run, "dims.ffn_dim")[0].join(["d_ffn/expert: ", ""])]
    else:
        ffn_title = "Feed-forward (dense)"
        ffn_vals = [("d_ffn: " + _qval(run, "dims.ffn_dim")[0])]
    ffn_why = why_for_property(run, "arch.sparsity")
    ffn_bot = _component(svg, CX, n2_bot + 14, box_w, node_id="b_ffn", known=known,
                         title=ffn_title, values=ffn_vals, why=ffn_why,
                         level=slc[0], conf=slc[1])
    _residual(svg, CX - box_w / 2, n2_bot + 12, ffn_bot - 12)

    # container rect around the block
    cont_bot = ffn_bot + 16
    svg.raw("<g>")
    svg.rect(CX - BW / 2 - 34, y, BW + 68, cont_bot - y, rx=14,
             style="fill:color-mix(in srgb,var(--ink,#14120e) 3%,transparent);"
                   "stroke:var(--border2,#ddd9cf);stroke-width:1.2;stroke-dasharray:3 4")
    svg.text(CX, y + 20, block_title, "fg-t")
    svg.raw("</g>")
    svg.text(CX + BW / 2 + 46, y + 20, "", "fg-w")
    for i, ln in enumerate(_wrap(block_sub, "fg-w", 250)[:2]):
        svg.text(CX - BW / 2 - 30, cont_bot + 14 + i * 12, ln, "fg-w", anchor="start")
    y = cont_bot + (30 if block_sub else 16)

    _arrow(svg, CX, y - 2, y + 26)
    y += 28

    # ---- final norm --------------------------------------------------------------
    y = _component(svg, CX, y, 220, node_id="b_fnorm", known=known, title="Final norm",
                   values=["before the readout"], why="assumed: near-universal convention",
                   level="speculative", conf=0.25)
    _arrow(svg, CX, y, y + 28)
    y += 28

    # ---- head ---------------------------------------------------------------------
    if kind == "decision":
        sc_val, _ = X.posterior_top(run, "decision.scoring_mechanism")
        sclc = _prop_lc(run, "decision.scoring_mechanism")
        y = _component(svg, CX, y, BW, node_id="b_head", known=known,
                       title="Decision readout heads",
                       values=[X.value_short(sc_val) if sc_val else "readout heads",
                               "choice · score · yes/no"],
                       why=why_for_property(run, "decision.scoring_mechanism"),
                       level=sclc[0], conf=sclc[1])
        _arrow(svg, CX, y, y + 26)
        y += 26
        pr = X.best_estimate(run, "decision.probability_resolution")
        if pr and pr.value is not None:
            y = _component(svg, CX, y, BW, node_id="b_probgrid", known=known,
                           title="Reported probabilities",
                           values=[f"on a {X.fmt_value(pr.value)} grid"],
                           why=why_for_quantity(run, "decision.probability_resolution"),
                           level=pr.level,
                           conf=ident_confidence(pr.identifiability), rx=20)
            _arrow(svg, CX, y, y + 26)
            y += 26
        y = _component(svg, CX, y, 240, node_id=None, known=known,
                       title="Structured decision", values=["what the API returns"],
                       why=None, level="observed", conf=1.0, rx=20)
    else:
        dtype = X.best_estimate(run, "out.logit_dtype")
        head_val, _ = X.posterior_top(run, "output.head")
        hlc = _prop_lc(run, "output.head")
        head_vals = [f"logits over {v_val.split('  ')[0]}",
                     (f"logit dtype: {X.fmt_value(dtype.value)}"
                      if dtype and dtype.value else "logit dtype: not identified")]
        head_why_parts = [w for w in (why_for_property(run, "output.head"),) if w]
        if dtype and dtype.value:
            dw = why_for_quantity(run, "out.logit_dtype")
            if dw:
                head_why_parts.append(dw)
        head_why = "  |  ".join(head_why_parts) or None
        y = _component(svg, CX, y, BW, node_id="b_head", known=known, title="LM head",
                       values=head_vals, why=head_why, level=hlc[0], conf=hlc[1])
        _arrow(svg, CX, y, y + 26)
        y += 26
        # sampling
        samp = X.best_estimate(run, "sampling.temperature_fidelity")
        if samp and samp.value is not None:
            y = _component(svg, CX, y, BW, node_id=None, known=known,
                           title="Softmax → sampling",
                           values=[f"temperature fidelity: {X.fmt_value(samp.value)}"],
                           why=why_for_quantity(run, "sampling.temperature_fidelity"),
                           level=samp.level, conf=ident_confidence(samp.identifiability))
            _arrow(svg, CX, y, y + 26)
            y += 26
        y = _component(svg, CX, y, 240, node_id="b_out", known=known,
                       title="Output text", values=["sampled autoregressively"],
                       why=None, level="observed", conf=1.0, rx=20)

    bottom = max(y, rail_top + rail_h if rail_items else 0) + 24

    # ---- legend -------------------------------------------------------------------
    bottom = _legend(svg, 20, bottom)
    return svg.render(bottom + 8)


def _rail_items(run: RunResult) -> list[tuple[str, str, str, str]]:
    """Small 'measured from the wire' stats for the side rail: (label, value, probe, level)."""
    out: list[tuple[str, str, str, str]] = []

    def add(q: str, label: str) -> None:
        e = X.best_estimate(run, q)
        if e is None or e.value is None:
            return
        val = X.fmt_value(e.value, X.quantity_unit(q))
        out.append((label.upper(), val, _pid_short(e.probe_id), canonical_level(e.level)))

    add("context.max_input_tokens", "context window ≥")
    add("context.effective_recall_tokens", "effective recall ≥")
    add("context.max_output_tokens", "max output")
    add("out.logit_dtype", "logit precision")
    add("timing.decode_tokens_per_s", "decode speed")
    add("serving.spec_decoding_speedup", "speculative speedup")
    add("serving.cache_block_tokens", "cache block")
    # a couple of serving properties as yes/no chips
    for prop, label in (("serving.prompt_caching", "prompt caching"),
                        ("serving.batch_invariant", "batch invariant"),
                        ("system.hidden_reasoning", "hidden reasoning")):
        v, _ = X.posterior_top(run, prop)
        if v is None:
            continue
        lvl = _prop_lc(run, prop)[0]
        out.append((label.upper(), X.value_short(v),
                    _pid_short((max(X.evidence_for_property(run, prop),
                                    key=lambda e: e.strength or 0).probe_id)
                               if X.evidence_for_property(run, prop) else ""),
                    canonical_level(lvl)))
    return out


def _legend(svg: _Svg, x: float, y: float) -> float:
    """Epistemic legend strip; returns the new bottom y."""
    items = [
        ("observed", "read off the wire"),
        ("inferred", "strong testable argument"),
        ("estimated", "statistical estimate + CI"),
        ("speculative", "under assumptions only"),
        ("unknown", "no signature on this interface"),
    ]
    y += 14
    svg.text(x, y, "HOW TO READ IT", "fg-g", anchor="start",
             style="fill:var(--muted,#8c877c)")
    y += 8
    cx = x
    for key, meaning in items:
        st = level_style(key)
        dash = {"solid": "", "dashed": "ln-dashed", "dotted": "ln-dotted"}[st.line]
        svg.rect(cx, y, 13, 13, rx=4, cls=f"fg-box lv-{key}" + (f" {dash}" if dash else ""))
        svg.text(cx + 19, y + 10, f"{key} — {meaning}", "fg-el", anchor="start")
        cx += 19 + len(f"{key} — {meaning}") * 5.6 + 26
        if cx > svg.w - 190:  # wrap legend row
            cx = x
            y += 22
    return y + 26


def _prop_lc(run: RunResult, prop: str) -> tuple[str, float | None]:
    ie = X.identifiability_for(run, prop)
    level = {1: "observed", 2: "inferred", 3: "estimated",
             4: "speculative", 5: "unknown"}.get(ie.level, "unknown") if ie else "unknown"
    _, p = X.posterior_top(run, prop)
    return level, (p if p else None)


# =========================================================================== TOPOLOGY
def topology_figure(run: RunResult, known_ids: set[str] | None = None) -> str:
    """Serving path as a horizontal evidence-annotated pipeline."""
    known = known_ids or set()
    svg = _Svg(980)
    BW_, BH_MIN, GAP = 196.0, 96.0, 34.0

    # collect the chain in order
    chain: list[dict[str, Any]] = []

    def node(nid, title, values, why, level, conf, rx=10):
        chain.append({"id": nid, "title": title, "values": values, "why": why,
                      "level": level, "conf": conf, "rx": rx})

    node("t_user", "User / client", ["this tool"], None, "observed", 1.0, rx=20)
    node("t_gw", "API gateway", ["TLS · auth · rate-limit"],
         "P01 · response headers observed", "observed", 0.9)

    topo_val, _ = X.posterior_top(run, "system.topology")
    safety_val, _ = X.posterior_top(run, "system.safety_middleware")
    retr_val, _ = X.posterior_top(run, "system.retrieval")
    if topo_val in ("router_or_cascade", "ensemble"):
        lc = _prop_lc(run, "system.topology")
        node("t_router", "Router / cascade", [X.value_short(topo_val)],
             why_for_property(run, "system.topology"), lc[0], lc[1])
    if safety_val in ("pre_filter", "both"):
        lc = _prop_lc(run, "system.safety_middleware")
        node("t_safe_pre", "Safety pre-filter", ["input classifier"],
             why_for_property(run, "system.safety_middleware"), lc[0], lc[1])
    if retr_val == "retrieval_augmented":
        lc = _prop_lc(run, "system.retrieval")
        node("t_retr", "Retrieval", ["context augmentation"],
             why_for_property(run, "system.retrieval"), lc[0], lc[1], rx=8)
    fam_val, _ = X.posterior_top(run, "arch.family")
    mlc = _prop_lc(run, "arch.family")
    node("t_model", "Model", [str(run.target.get("name", "model")),
                              X.value_short(fam_val)],
         why_for_property(run, "arch.family"), mlc[0], mlc[1])
    if safety_val in ("post_filter", "both"):
        lc = _prop_lc(run, "system.safety_middleware")
        node("t_safe_post", "Safety post-filter", ["output classifier"],
             why_for_property(run, "system.safety_middleware"), lc[0], lc[1])
    node("t_resp", "Response", ["what we receive"], None, "observed", 1.0, rx=20)

    # layout: two rows if the chain is long
    per_row = 4 if len(chain) > 4 else len(chain)
    rows = [chain[i:i + per_row] for i in range(0, len(chain), per_row)]
    y = 30
    model_pos: tuple[float, float, float] | None = None   # cx, y, bottom
    arrows: list[tuple[float, float, float, float]] = []
    for r, row in enumerate(rows):
        total_w = len(row) * BW_ + (len(row) - 1) * GAP
        x = (svg.w - total_w) / 2
        row_bot = y
        for it in row:
            cx = x + BW_ / 2
            bot = _component(svg, cx, y, BW_, node_id=it["id"], known=known,
                             title=it["title"], values=it["values"], why=it["why"],
                             level=it["level"], conf=it["conf"], rx=it["rx"])
            row_bot = max(row_bot, bot)
            if it["id"] == "t_model":
                model_pos = (cx, y, bot)
            x += BW_ + GAP
        # arrows within row
        x = (svg.w - total_w) / 2
        for i in range(len(row) - 1):
            x0 = x + BW_ + 3
            x1 = x + BW_ + GAP - 3
            arrows.append((x0, y + 40, x1, y + 40))
            x += BW_ + GAP
        if r < len(rows) - 1:
            # wrap arrow from end of row to start of next row
            x_end = (svg.w - total_w) / 2 + len(row) * (BW_ + GAP) - GAP - BW_ / 2
            next_row = rows[r + 1]
            next_total = len(next_row) * BW_ + (len(next_row) - 1) * GAP
            x_next = (svg.w - next_total) / 2 + BW_ / 2
            svg.path(f"M {x_end:.1f} {row_bot + 6:.1f} C {x_end:.1f} {row_bot + 34:.1f}, "
                     f"{x_next:.1f} {row_bot + 34:.1f}, {x_next:.1f} {row_bot + 52:.1f}")
            y = row_bot + 60
        else:
            y = row_bot
    for x0, yy, x1, _ in arrows:
        svg.line(x0, yy, x1, yy)

    # loops: hidden reasoning / tool orchestration under the model
    if model_pos is not None:
        cx, my, mbot = model_pos
        extras = []
        if X.posterior_top(run, "system.hidden_reasoning")[0] == "yes":
            lc = _prop_lc(run, "system.hidden_reasoning")
            extras.append(("t_reason", "Hidden reasoning loop",
                           "unshown tokens before the answer",
                           why_for_property(run, "system.hidden_reasoning"), lc))
        if X.posterior_top(run, "system.tool_orchestration")[0] == "yes":
            lc = _prop_lc(run, "system.tool_orchestration")
            extras.append(("t_tools", "Tool / agent loop", "calls tools mid-answer",
                           why_for_property(run, "system.tool_orchestration"), lc))
        ey = mbot + 44
        for nid, title, val, why, lc in extras:
            bot = _component(svg, cx, ey, BW_, node_id=nid, known=known, title=title,
                             values=[val], why=why, level=lc[0], conf=lc[1], rx=10)
            svg.path(f"M {cx - BW_ / 2 + 20:.1f} {mbot + 2:.1f} "
                     f"C {cx - BW_ / 2 - 20:.1f} {ey + 10:.1f}, "
                     f"{cx - BW_ / 2 - 20:.1f} {ey + 20:.1f}, "
                     f"{cx - BW_ / 2 + 20:.1f} {ey + 24:.1f}", cls="fg-ea")
            svg.path(f"M {cx + BW_ / 2 - 20:.1f} {ey + 24:.1f} "
                     f"C {cx + BW_ / 2 + 20:.1f} {ey + 20:.1f}, "
                     f"{cx + BW_ / 2 + 20:.1f} {ey + 10:.1f}, "
                     f"{cx + BW_ / 2 - 20:.1f} {mbot + 2:.1f}", cls="fg-ea")
            ey = bot + 34
        y = max(y, ey)

    bottom = _legend(svg, 20, y + 26)
    return svg.render(bottom + 8)
