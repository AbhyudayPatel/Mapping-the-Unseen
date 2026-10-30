"""Diagram sources (Mermaid + Graphviz DOT) built entirely from a :class:`RunResult`.

Three diagrams, each driven by the numbers — never a hard-coded architecture:

* :func:`block_architecture` — the network block diagram (tokenizer -> embedding ->
  N transformer blocks -> head / decision heads), with a vision/audio lane when the
  posterior says the model is multimodal, and explicit "not identified" nodes for
  quantities the interface cannot pin down.
* :func:`system_topology` — the serving path (user -> gateway -> router -> safety ->
  retrieval -> model -> verifier -> response), drawing only components with evidence.
* :func:`evidence_flow` — experiment (probe) -> conclusion (estimate/property) graph
  for traceability.

Each builder returns a :class:`Diagram` carrying its ``nodes``/``edges`` (so the HTML
can build clickable fallbacks and an inspector index), a ``mermaid`` string, a ``dot``
string, and — when the ``dot`` binary is present — a rendered ``svg``.

The colour + line-style of every node comes from :func:`forensics.report.style.style_for`
applied to that node's epistemic level and confidence, so the whole report shares one
visual language.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

from forensics.inference.run_types import RunResult

from . import extract as X
from . import figure as FIG
from .style import (
    canonical_level, confidence_marker, confidence_word, ident_confidence,
    ident_to_level, level_style, style_for,
)


def _short_why(s: str | None, n: int = 66) -> str:
    """Compact one-line provenance for export (mermaid/dot) node labels."""
    if not s:
        return ""
    s = " ".join(str(s).split())
    return "↳ " + (s if len(s) <= n else s[: n - 1] + "…")

# re-export the shared pure mapping so callers can do diagrams.style_for(...)
__all__ = [
    "Diagram", "DiagramNode", "DiagramEdge", "style_for",
    "block_architecture", "system_topology", "evidence_flow",
    "render_svg", "legend", "all_diagrams",
]


# --------------------------------------------------------------------------- data model
@dataclass
class DiagramNode:
    id: str                       # diagram-unique, mermaid-safe (e.g. "b0")
    lines: list[str]              # label lines (rendered as <br/> / \n)
    key: str = ""                 # inspector data-key ("prop:..","quant:..","probe:..","note:..")
    level: str = "unknown"        # epistemic level -> colour
    confidence: float | None = None
    shape: str = "box"            # box|round|stadium|subroutine|cylinder|hexagon
    detail: dict[str, Any] = field(default_factory=dict)   # for note: nodes / extra context

    @property
    def color(self) -> str:
        return style_for(self.level, self.confidence)[0]

    @property
    def line_style(self) -> str:
        return style_for(self.level, self.confidence)[1]


@dataclass
class DiagramEdge:
    src: str
    dst: str
    label: str = ""
    style: str = "solid"          # solid|dashed|dotted


@dataclass
class Diagram:
    title: str
    kind: str                     # block|topology|evidence
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    mermaid: str = ""
    dot: str = ""
    svg: str | None = None
    subgraphs: list[dict[str, Any]] = field(default_factory=list)  # {"title":.., "ids":[..]}
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- small helpers
def _status_line(level: str, confidence: float | None) -> str:
    lv = canonical_level(level)
    mark = confidence_marker(confidence)
    if confidence is None:
        return f"{mark} {level_style(lv).key}"
    return f"{mark} {level_style(lv).key} · {X.fmt_pct(confidence)}"


def _prop_lc(run: RunResult, prop: str) -> tuple[str, float | None]:
    """(level, confidence) for a property node: level from identifiability, confidence
    from posterior mass on the winning value."""
    ie = X.identifiability_for(run, prop)
    level = ident_to_level(ie.level) if ie is not None else "unknown"
    _, p = X.posterior_top(run, prop)
    return level, (p if p else None)


def _quant_lc(run: RunResult, quantity: str) -> tuple[str, float | None]:
    e = X.best_estimate(run, quantity)
    if e is None or e.value is None:
        # not identified within budget -> render as an explicit unknown (grey, dashed)
        return "unknown", None
    return e.level, ident_confidence(getattr(e, "identifiability", None))


def _q_value_line(run: RunResult, quantity: str, prefix: str = "") -> str:
    e = X.best_estimate(run, quantity)
    unit = X.quantity_unit(quantity)
    if e is None or e.value is None:
        return f"{prefix}{X.NOT_IDENTIFIED}"
    s = X.fmt_value(e.value, unit)
    ci = X.fmt_ci(e.ci, e.value)
    return f"{prefix}{s}" + (f" (CI {ci})" if ci else "")


# =========================================================================== BLOCK DIAGRAM
def block_architecture(run: RunResult) -> Diagram:
    nodes: list[DiagramNode] = []
    edges: list[DiagramEdge] = []
    subgraphs: list[dict[str, Any]] = []
    notes: list[str] = []
    kind = run.caps.kind
    mm = X.detect_multimodal(run)

    def add(nid, lines, **kw) -> DiagramNode:
        n = DiagramNode(id=nid, lines=[l for l in lines if l], **kw)
        nodes.append(n)
        return n

    # ---- input ------------------------------------------------------------------
    add("b_in", ["Text prompt"], key="", level="observed", confidence=1.0, shape="stadium",
        detail={"title": "Text input",
                "body": "The endpoint accepts a text prompt (the baseline modality)."})
    prev = "b_in"

    # ---- multimodal input lane --------------------------------------------------
    if mm["image"] or mm["audio"]:
        lc = _prop_lc(run, "modal.input")
        modalities = []
        if mm["image"]:
            modalities.append("image")
        if mm["audio"]:
            modalities.append("audio")
        add("b_mm_in", [f"{' + '.join(modalities).title()} input",
                        X.value_short(mm['input']), _status_line(*lc)],
            key="prop:modal.input", level=lc[0], confidence=lc[1], shape="stadium")
        if mm["image"]:
            ve = mm["vision_encoder"]
            velc = _prop_lc(run, "modal.vision_encoder")
            ve_lbl = {"vit_patch_tokens": "ViT → patch tokens",
                      "native_resolution_vit": "native-resolution tiling → tokens",
                      "cross_attn_adapter": "cross-attention adapter",
                      "none": "vision pathway"}.get(str(ve), "vision encoder")
            add("b_vision", ["Vision encoder", ve_lbl, _status_line(*velc)],
                key="prop:modal.vision_encoder", level=velc[0], confidence=velc[1], shape="box")
            add("b_fuse", ["Modality fusion", "image tokens → sequence"],
                key="prop:modal.vision_encoder", level=velc[0], confidence=velc[1], shape="hexagon")
            edges.append(DiagramEdge("b_mm_in", "b_vision"))
            edges.append(DiagramEdge("b_vision", "b_fuse"))
        if mm["audio"]:
            add("b_audio", ["Audio encoder", "audio → tokens"],
                key="prop:modal.input", level=lc[0], confidence=lc[1], shape="box")
            edges.append(DiagramEdge("b_mm_in", "b_audio"))
            edges.append(DiagramEdge("b_audio", "b_fuse" if mm["image"] else "b_embed"))

    # ---- tokenizer --------------------------------------------------------------
    tf = X.best_estimate(run, "tokenizer.family")
    tt = X.best_estimate(run, "tokenizer.type")
    tlc = _quant_lc(run, "tokenizer.family")
    tok_lines = ["Tokenizer",
                 X.fmt_value(tf.value) if tf and tf.value else X.NOT_IDENTIFIED]
    if tt and tt.value:
        tok_lines.append(f"type: {X.fmt_value(tt.value)}")
    tok_lines.append(_status_line(*tlc))
    add("b_tok", tok_lines, key="quant:tokenizer.family", level=tlc[0], confidence=tlc[1],
        shape="box")
    edges.append(DiagramEdge(prev, "b_tok"))

    # ---- embedding --------------------------------------------------------------
    vlc = _quant_lc(run, "dims.vocab_size")
    d_est = X.best_estimate(run, "dims.hidden_size")
    tied = X.est_value(run, "dims.embedding_tied")
    emb_lines = ["Token embedding",
                 _q_value_line(run, "dims.vocab_size", "vocab: "),
                 (_q_value_line(run, "dims.hidden_size", "d_model: ")),
                 (f"tied: {X.fmt_value(tied)}" if tied is not None else "tied: unknown"),
                 _short_why(FIG.why_for_quantity(run, "dims.vocab_size")),
                 _status_line(*vlc)]
    add("b_embed", emb_lines, key="quant:dims.vocab_size", level=vlc[0], confidence=vlc[1],
        shape="box",
        detail={"related": ["quant:dims.hidden_size", "quant:dims.embedding_tied"]})
    if mm["image"] or mm["audio"]:
        edges.append(DiagramEdge("b_fuse" if mm["image"] else "b_audio", "b_embed",
                                 label="fused tokens"))
    edges.append(DiagramEdge("b_tok", "b_embed"))

    # ---- transformer block (× N) -----------------------------------------------
    n_layers_e = X.best_estimate(run, "dims.n_layers")
    n_lbl = X.fmt_value(n_layers_e.value) if n_layers_e and n_layers_e.value is not None \
        else "N"
    fam_val, fam_p = X.posterior_top(run, "arch.family")
    flc = _prop_lc(run, "arch.family")

    block_ids: list[str] = []
    # attention node
    kv_val, kv_p = X.posterior_top(run, "attn.kv_sharing")
    attn_name = {"mha": "Multi-head attention (MHA)",
                 "gqa_or_mqa": "Grouped / multi-query attention (GQA/MQA)",
                 "mla": "Multi-head latent attention (MLA)"}.get(str(kv_val),
                                                                 "attention")
    klc = _prop_lc(run, "attn.kv_sharing")
    heads = X.best_estimate(run, "dims.n_heads")
    kvh = X.best_estimate(run, "dims.n_kv_heads")
    hd = X.best_estimate(run, "dims.head_dim")
    head_bits = []
    if heads and heads.value is not None:
        head_bits.append(f"{X.fmt_value(heads.value)} heads")
    if kvh and kvh.value is not None:
        head_bits.append(f"{X.fmt_value(kvh.value)} KV")
    if hd and hd.value is not None:
        head_bits.append(f"head_dim {X.fmt_value(hd.value)}")
    pos_val, pos_p = X.posterior_top(run, "pos.encoding")
    plc = _prop_lc(run, "pos.encoding")
    attn_lines = [attn_name,
                  ", ".join(head_bits) if head_bits else "head split: not identified",
                  f"pos: {X.value_short(pos_val)}",
                  _status_line(*klc)]
    add("b_attn", attn_lines, key="prop:attn.kv_sharing", level=klc[0], confidence=klc[1],
        shape="box", detail={"related": ["prop:pos.encoding", "quant:dims.n_heads",
                                          "quant:dims.n_kv_heads", "quant:dims.head_dim"]})
    block_ids.append("b_attn")

    # FFN vs MoE node
    spars_val, spars_p = X.posterior_top(run, "arch.sparsity")
    slc = _prop_lc(run, "arch.sparsity")
    ffn = X.best_estimate(run, "dims.ffn_dim")
    if str(spars_val) == "moe":
        ne = X.best_estimate(run, "dims.n_experts")
        ae = X.best_estimate(run, "dims.active_experts")
        ffn_lines = ["MoE feed-forward",
                     (f"{X.fmt_value(ne.value)} experts" if ne and ne.value is not None
                      else "experts: not identified") +
                     (f", top-{X.fmt_value(ae.value)} active" if ae and ae.value is not None
                      else ""),
                     _q_value_line(run, "dims.ffn_dim", "d_ffn/expert: "),
                     _short_why(FIG.why_for_property(run, "arch.sparsity")),
                     _status_line(*slc)]
    else:
        ffn_lines = ["Feed-forward (dense)",
                     _q_value_line(run, "dims.ffn_dim", "d_ffn: "),
                     _short_why(FIG.why_for_property(run, "arch.sparsity")),
                     _status_line(*slc)]
    add("b_ffn", ffn_lines, key="prop:arch.sparsity", level=slc[0], confidence=slc[1],
        shape="box")
    block_ids.append("b_ffn")

    # norm (assumed — no black-box signature)
    add("b_norm", ["Norm + residual", "RMSNorm, pre-norm (assumed)"],
        key="note:b_norm", level="speculative", confidence=0.25, shape="box",
        detail={"title": "Normalisation & residual stream",
                "body": "Norm type (RMSNorm vs LayerNorm) and placement (pre- vs post-norm) "
                        "have no black-box I/O signature; shown as the modern default under "
                        "assumption.",
                "identifiability": "non-identifiable from this interface"})
    block_ids.append("b_norm")
    edges.append(DiagramEdge("b_attn", "b_ffn"))
    edges.append(DiagramEdge("b_ffn", "b_norm"))

    block_title = f"Transformer block × {n_lbl}"
    if fam_val and fam_val not in ("decoder_only",):
        block_title = f"{X.value_short(fam_val)} block × {n_lbl}"
    subgraphs.append({"id": "blk", "title": block_title, "ids": list(block_ids)})
    edges.append(DiagramEdge("b_embed", "b_attn", label=f"×{n_lbl} layers"))

    # ---- final norm -------------------------------------------------------------
    add("b_fnorm", ["Final norm"], key="note:b_fnorm", level="speculative", confidence=0.25,
        shape="box",
        detail={"title": "Final normalisation",
                "body": "A final norm before the head is the near-universal convention; not "
                        "separately observable.",
                "identifiability": "assumed"})
    edges.append(DiagramEdge("b_norm", "b_fnorm", label="after N blocks"))

    # ---- head(s) ----------------------------------------------------------------
    head_val, head_p = X.posterior_top(run, "output.head")
    hlc = _prop_lc(run, "output.head")
    if kind == "decision" or head_val == "classification_heads":
        # decision heads + question-type branches
        sclc = _prop_lc(run, "decision.scoring_mechanism")
        sc_val, _ = X.posterior_top(run, "decision.scoring_mechanism")
        add("b_head", ["Decision readout heads",
                       X.value_short(sc_val) if sc_val else "readout",
                       _short_why(FIG.why_for_property(run, "decision.scoring_mechanism")),
                       _status_line(*sclc)],
            key="prop:decision.scoring_mechanism", level=sclc[0], confidence=sclc[1],
            shape="subroutine",
            detail={"related": ["prop:output.head", "prop:decision.question_coupling",
                                 "prop:decision.option_processing",
                                 "quant:decision.probability_resolution"]})
        edges.append(DiagramEdge("b_fnorm", "b_head"))
        for qid, qlabel, qdesc in (
                ("b_q_choice", "choice head", "labelled-option probabilities"),
                ("b_q_score", "score head", "ordinal level score"),
                ("b_q_noul", "noul head", "yes/no probability")):
            add(qid, [qlabel, qdesc], key="prop:decision.scoring_mechanism",
                level=sclc[0], confidence=sclc[1], shape="round")
            edges.append(DiagramEdge("b_head", qid))
        pr_e = X.best_estimate(run, "decision.probability_resolution")
        if pr_e and pr_e.value is not None:
            add("b_probgrid", ["Reported probabilities",
                               f"grid: {X.fmt_value(pr_e.value)}", _status_line(*_quant_lc(run, 'decision.probability_resolution'))],
                key="quant:decision.probability_resolution",
                level=pr_e.level, confidence=ident_confidence(pr_e.identifiability),
                shape="stadium")
            edges.append(DiagramEdge("b_q_choice", "b_probgrid", style="dashed"))
    else:
        dtype = X.best_estimate(run, "out.logit_dtype")
        prec = f"logits: {X.fmt_value(dtype.value)}" if dtype and dtype.value else \
            "logits: precision not identified"
        add("b_head", ["LM head",
                       _q_value_line(run, "dims.vocab_size", "→ logits over "),
                       prec,
                       _short_why(FIG.why_for_property(run, "output.head")
                                  or FIG.why_for_quantity(run, "out.logit_dtype")),
                       _status_line(*hlc)],
            key="prop:output.head", level=hlc[0], confidence=hlc[1], shape="subroutine",
            detail={"related": ["quant:out.logit_dtype", "quant:dims.vocab_size"]})
        edges.append(DiagramEdge("b_fnorm", "b_head"))
        add("b_out", ["Sampled text output"], key="", level="observed", confidence=1.0,
            shape="stadium",
            detail={"title": "Output", "body": "Autoregressively sampled tokens."})
        edges.append(DiagramEdge("b_head", "b_out"))
        # multimodal output lane
        if mm["multimodal_out"]:
            olc = _prop_lc(run, "modal.output")
            add("b_mm_out", [f"{X.value_short(mm['output'])} output", _status_line(*olc)],
                key="prop:modal.output", level=olc[0], confidence=olc[1], shape="stadium")
            edges.append(DiagramEdge("b_head", "b_mm_out", style="dashed"))

    title = f"Inferred architecture — {run.target.get('name', run.run_id)}"
    d = Diagram(title=title, kind="block", nodes=nodes, edges=edges, subgraphs=subgraphs,
                notes=notes)
    _finalize(d)
    try:
        svg = FIG.architecture_figure(run, known_ids={n.id for n in d.nodes})
        if svg:
            d.svg = svg
    except Exception:  # noqa: BLE001 — keep the graphviz/mermaid rendering
        pass
    return d


# =========================================================================== TOPOLOGY
def system_topology(run: RunResult) -> Diagram:
    """Serving path, drawing only components with evidence (system.* posterior + caps)."""
    nodes: list[DiagramNode] = []
    edges: list[DiagramEdge] = []

    def add(nid, lines, **kw) -> DiagramNode:
        nodes.append(DiagramNode(id=nid, lines=[l for l in lines if l], **kw))
        return nodes[-1]

    add("t_user", ["User / client"], key="", level="observed", confidence=1.0,
        shape="stadium",
        detail={"title": "Client", "body": "The caller of the API."})
    order = ["t_user"]

    # gateway is always present (we reached an HTTP endpoint)
    add("t_gw", ["API gateway", "TLS / auth / rate-limit"], key="", level="observed",
        confidence=0.9, shape="box",
        detail={"title": "Gateway", "body": "Every hosted endpoint sits behind a gateway; "
                                            "observed via response headers."})
    edges.append(DiagramEdge("t_user", "t_gw"))
    order.append("t_gw")
    tail = "t_gw"

    # router / cascade
    topo_val, topo_p = X.posterior_top(run, "system.topology")
    if topo_val in ("router_or_cascade", "ensemble"):
        lc = _prop_lc(run, "system.topology")
        add("t_router", ["Router / classifier", X.value_short(topo_val),
                         _short_why(FIG.why_for_property(run, "system.topology")),
                         _status_line(*lc)],
            key="prop:system.topology", level=lc[0], confidence=lc[1], shape="hexagon")
        edges.append(DiagramEdge(tail, "t_router"))
        order.append("t_router")
        tail = "t_router"

    # safety pre-filter
    safety_val, safety_p = X.posterior_top(run, "system.safety_middleware")
    slc = _prop_lc(run, "system.safety_middleware")
    if safety_val in ("pre_filter", "both"):
        add("t_safe_pre", ["Safety pre-filter", "input classifier",
                           _short_why(FIG.why_for_property(run, "system.safety_middleware")),
                           _status_line(*slc)],
            key="prop:system.safety_middleware", level=slc[0], confidence=slc[1], shape="box")
        edges.append(DiagramEdge(tail, "t_safe_pre"))
        order.append("t_safe_pre")
        tail = "t_safe_pre"

    # retrieval
    retr_val, _ = X.posterior_top(run, "system.retrieval")
    if retr_val == "retrieval_augmented":
        rlc = _prop_lc(run, "system.retrieval")
        add("t_retr", ["Retrieval", "context augmentation",
                       _short_why(FIG.why_for_property(run, "system.retrieval")),
                       _status_line(*rlc)],
            key="prop:system.retrieval", level=rlc[0], confidence=rlc[1], shape="cylinder")
        edges.append(DiagramEdge(tail, "t_retr"))
        order.append("t_retr")
        tail = "t_retr"

    # the model itself
    fam_val, _ = X.posterior_top(run, "arch.family")
    mlc = _prop_lc(run, "arch.family")
    model_name = run.target.get("name", "model")
    add("t_model", ["Model", f"{model_name}", X.value_short(fam_val),
                    _short_why(FIG.why_for_property(run, "arch.family")),
                    _status_line(*mlc)],
        key="prop:arch.family", level=mlc[0], confidence=mlc[1], shape="subroutine",
        detail={"related": ["prop:arch.sparsity", "prop:attn.long_range"]})
    edges.append(DiagramEdge(tail, "t_model"))
    order.append("t_model")

    # hidden reasoning
    if X.has_property(run, "system.hidden_reasoning"):
        hr_val, _ = X.posterior_top(run, "system.hidden_reasoning")
        if hr_val == "yes":
            hlc = _prop_lc(run, "system.hidden_reasoning")
            add("t_reason", ["Hidden reasoning", "unshown tokens", _status_line(*hlc)],
                key="prop:system.hidden_reasoning", level=hlc[0], confidence=hlc[1],
                shape="box")
            edges.append(DiagramEdge("t_model", "t_reason"))
            edges.append(DiagramEdge("t_reason", "t_model", label="answer", style="dashed"))

    # tool orchestration
    if X.has_property(run, "system.tool_orchestration"):
        tv, _ = X.posterior_top(run, "system.tool_orchestration")
        if tv == "yes":
            tlc = _prop_lc(run, "system.tool_orchestration")
            add("t_tools", ["Tool / agent loop", _status_line(*tlc)],
                key="prop:system.tool_orchestration", level=tlc[0], confidence=tlc[1],
                shape="hexagon")
            edges.append(DiagramEdge("t_model", "t_tools"))
            edges.append(DiagramEdge("t_tools", "t_model", style="dashed"))

    # safety post-filter
    if safety_val in ("post_filter", "both"):
        add("t_safe_post", ["Safety post-filter", "output classifier",
                            _short_why(FIG.why_for_property(run, "system.safety_middleware")),
                            _status_line(*slc)],
            key="prop:system.safety_middleware", level=slc[0], confidence=slc[1], shape="box")
        edges.append(DiagramEdge("t_model", "t_safe_post"))
        last = "t_safe_post"
    else:
        last = "t_model"

    add("t_resp", ["Response"], key="", level="observed", confidence=1.0, shape="stadium",
        detail={"title": "Response", "body": "What the client receives."})
    edges.append(DiagramEdge(last, "t_resp"))

    d = Diagram(title="Serving topology (evidenced components only)", kind="topology",
                nodes=nodes, edges=edges)
    _finalize(d, orient="LR")
    try:
        svg = FIG.topology_figure(run, known_ids={n.id for n in d.nodes})
        if svg:
            d.svg = svg
    except Exception:  # noqa: BLE001
        pass
    return d


# =========================================================================== EVIDENCE FLOW
def evidence_flow(run: RunResult) -> Diagram:
    """Experiment (probe) -> conclusion (estimate / property) traceability graph."""
    nodes: list[DiagramNode] = []
    edges: list[DiagramEdge] = []
    seen: set[str] = set()

    def ensure(nid, lines, **kw):
        if nid in seen:
            return
        seen.add(nid)
        nodes.append(DiagramNode(id=nid, lines=[l for l in lines if l], **kw))

    # collect probe -> conclusions
    probes = sorted({pr.probe_id for pr in run.probe_results} |
                    {st.probe_id for st in run.experiment_log})
    # index estimates & evidence by probe
    est_by_probe: dict[str, list] = {}
    for e in run.estimates:
        est_by_probe.setdefault(e.probe_id or "?", []).append(e)
    ev_by_probe: dict[str, list] = {}
    for ev in run.evidence:
        ev_by_probe.setdefault(ev.probe_id or "?", []).append(ev)

    idx = 0
    for pid in probes:
        ests = est_by_probe.get(pid, [])
        evs = ev_by_probe.get(pid, [])
        if not ests and not evs:
            continue
        pnid = f"e_p{idx}"
        idx += 1
        st = X.experiment_for_probe(run, pid)
        plines = [pid.split(".", 1)[0], pid.split(".", 1)[1] if "." in pid else ""]
        if st and st.eig_bits is not None:
            plines.append(f"EIG {X.fmt_num(st.eig_bits)} bits")
        ensure(pnid, plines, key=f"probe:{pid}", level="observed", confidence=0.9,
               shape="box")
        # estimate conclusions
        for e in ests:
            if e.value is None:
                continue
            cid = f"eq_{_slug(e.quantity)}"
            lc = _quant_lc(run, e.quantity)
            ensure(cid, [e.quantity, X.fmt_value(e.value, X.quantity_unit(e.quantity)),
                         _short_why(e.method, 56)],
                   key=f"quant:{e.quantity}", level=lc[0], confidence=lc[1], shape="round")
            edges.append(DiagramEdge(pnid, cid))
        # property conclusions
        for ev in evs:
            cid = f"ep_{_slug(ev.property)}"
            lc = _prop_lc(run, ev.property)
            top_v, _ = X.posterior_top(run, ev.property)
            ensure(cid, [ev.property, X.value_short(top_v), _short_why(ev.summary, 56)],
                   key=f"prop:{ev.property}",
                   level=lc[0], confidence=lc[1], shape="round")
            edges.append(DiagramEdge(pnid, cid, style="dashed"))

    d = Diagram(title="Evidence flow — experiments to conclusions", kind="evidence",
                nodes=nodes, edges=edges)
    _finalize(d, orient="LR")
    return d


# =========================================================================== rendering
_SHAPE_MM = {
    "box": ('["', '"]'), "round": ('("', '")'), "stadium": ('(["', '"])'),
    "subroutine": ('[["', '"]]'), "cylinder": ('[("', '")]'), "hexagon": ('{{"', '"}}'),
}
_SHAPE_DOT = {
    "box": "box", "round": "box", "stadium": "box", "subroutine": "component",
    "cylinder": "cylinder", "hexagon": "hexagon",
}
_DASH = {"solid": "", "dashed": "stroke-dasharray: 6 4", "dotted": "stroke-dasharray: 2 4"}


def _mm_text(lines: list[str]) -> str:
    out = []
    for ln in lines:
        s = str(ln).replace('"', "'").replace("<", "‹").replace(">", "›")
        out.append(s)
    return "<br/>".join(out) or " "


def _dot_text(lines: list[str]) -> str:
    out = [str(ln).replace("\\", " ").replace('"', "'") for ln in lines]
    return "\\n".join(out) or " "


def _finalize(d: Diagram, orient: str = "TB") -> None:
    d.mermaid = _to_mermaid(d, orient)
    d.dot = _to_dot(d, orient)
    d.svg = render_svg(d.dot)


def _to_mermaid(d: Diagram, orient: str) -> str:
    lines = [f"flowchart {orient}"]
    sub_ids: set[str] = set()
    for sg in d.subgraphs:
        lines.append(f'  subgraph {sg["id"]}["{_mm_text([sg["title"]])}"]')
        for nid in sg["ids"]:
            node = _find(d, nid)
            if node:
                op, cl = _SHAPE_MM.get(node.shape, _SHAPE_MM["box"])
                lines.append(f"    {nid}{op}{_mm_text(node.lines)}{cl}")
                sub_ids.add(nid)
        lines.append("  end")
    for n in d.nodes:
        if n.id in sub_ids:
            continue
        op, cl = _SHAPE_MM.get(n.shape, _SHAPE_MM["box"])
        lines.append(f"  {n.id}{op}{_mm_text(n.lines)}{cl}")
    for e in d.edges:
        arrow = {"solid": "-->", "dashed": "-.->", "dotted": "-.->"}.get(e.style, "-->")
        if e.label:
            lines.append(f'  {e.src} {arrow}|{_mm_text([e.label])}| {e.dst}')
        else:
            lines.append(f"  {e.src} {arrow} {e.dst}")
    # class definitions per epistemic level
    for key in ("observed", "inferred", "estimated", "speculative", "unknown"):
        st = level_style(key)
        dash = _DASH[st.line]
        extra = f",{dash}" if dash else ""
        lines.append(f"  classDef {key} fill:{_tint(st.color)},stroke:{st.color},"
                     f"stroke-width:2px,color:#111{extra};")
    for n in d.nodes:
        lines.append(f"  class {n.id} {canonical_level(n.level)};")
    # NOTE: click wiring is done by the HTML via robust post-render DOM binding
    # (node ids are globally unique across the three diagrams), not mermaid `click`
    # directives, so interactivity works regardless of the mermaid securityLevel.
    return "\n".join(lines)


def _to_dot(d: Diagram, orient: str) -> str:
    rankdir = "LR" if orient == "LR" else "TB"
    lines = ["digraph forensics {",
             f"  rankdir={rankdir};",
             '  bgcolor="transparent";',
             '  node [fontname="Helvetica", fontsize=10, style="filled,rounded"];',
             '  edge [fontname="Helvetica", fontsize=9, color="#888888"];']
    for i, sg in enumerate(d.subgraphs):
        lines.append(f'  subgraph cluster_{i} {{')
        lines.append(f'    label="{_dot_text([sg["title"]])}"; color="#bbbbbb";')
        for nid in sg["ids"]:
            node = _find(d, nid)
            if node:
                lines.append("  " + _dot_node(node))
        lines.append("  }")
    sub_ids = {nid for sg in d.subgraphs for nid in sg["ids"]}
    for n in d.nodes:
        if n.id in sub_ids:
            continue
        lines.append(_dot_node(n))
    for e in d.edges:
        style = e.style if e.style in ("dashed", "dotted") else "solid"
        lbl = f', label="{_dot_text([e.label])}"' if e.label else ""
        lines.append(f'  {e.src} -> {e.dst} [style={style}{lbl}];')
    lines.append("}")
    return "\n".join(lines)


def _dot_node(n: DiagramNode) -> str:
    st = level_style(n.level)
    color, line = style_for(n.level, n.confidence)
    shape = _SHAPE_DOT.get(n.shape, "box")
    style = "filled,rounded"
    if line == "dashed":
        style = "filled,rounded,dashed"
    elif line == "dotted":
        style = "filled,rounded,dotted"
    return (f'  {n.id} [label="{_dot_text(n.lines)}", shape={shape}, '
            f'fillcolor="{_tint(color)}", color="{color}", style="{style}"];')


def _find(d: Diagram, nid: str) -> DiagramNode | None:
    for n in d.nodes:
        if n.id == nid:
            return n
    return None


def _tint(hex_color: str) -> str:
    """A very light tint of a hue for node fill (keeps text readable in light mode)."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, IndexError):
        return "#f4f4f2"
    mix = 0.85
    r = int(r + (255 - r) * mix)
    g = int(g + (255 - g) * mix)
    b = int(b + (255 - b) * mix)
    return f"#{r:02x}{g:02x}{b:02x}"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


# --------------------------------------------------------------------------- dot binary
def render_svg(dot_source: str) -> str | None:
    """Render DOT to inline SVG using the ``dot`` binary if present, else ``None``."""
    exe = shutil.which("dot")
    if not exe or not dot_source:
        return None
    try:
        proc = subprocess.run([exe, "-Tsvg"], input=dot_source.encode("utf-8"),
                              capture_output=True, timeout=20)
        if proc.returncode != 0:
            return None
        svg = proc.stdout.decode("utf-8", "replace")
        i = svg.find("<svg")
        return svg[i:] if i >= 0 else svg
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- legend
def legend() -> list[dict[str, str]]:
    """Colour/line-style -> confidence + epistemic-level legend (shared by md/html)."""
    from .style import legend_rows
    return legend_rows()


def all_diagrams(run: RunResult) -> dict[str, Diagram]:
    return {
        "block": block_architecture(run),
        "topology": system_topology(run),
        "evidence": evidence_flow(run),
    }
