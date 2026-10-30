"""Tests for the reporting package (:mod:`forensics.report`).

All offline: a :class:`RunResult` is generated once per architecture from the
``SimulatedAdapter`` (``config={"fast": True}``) and reused across tests. The checks
verify the public contract of :func:`forensics.report.build_report` and the three
renderers:

* three files are written and their paths returned;
* the Markdown has a fenced ```mermaid``` block and every section A-L;
* the HTML embeds ``window.__FORENSICS__`` + the Inspector, parses without error, and its
  data blob is valid JSON;
* an unrecovered ``hidden_size`` (``None``) degrades to "not identified within budget";
* the decision model's block diagram shows decision heads + question branches (choice /
  score / noul) instead of an LM head;
* multimodality is drawn when the posterior says the model is vision/audio capable;
* the shared ``style_for`` mapping and ``render_svg`` (dot present or not) behave.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from functools import lru_cache
from html.parser import HTMLParser

import pytest

from forensics.core.adapters.simulated import (
    SimulatedAdapter, sim_decoder_dense, sim_jev,
)
from forensics.inference.engine import run_forensics
from forensics.report import build_report
from forensics.report import charts as CH
from forensics.report import diagrams as DG
from forensics.report import html as HT
from forensics.report import markdown as MD
from forensics.report import style as ST


# --------------------------------------------------------------------------- fixtures
@lru_cache(maxsize=None)
def _run(kind: str):
    cfg = sim_jev() if kind == "jev" else sim_decoder_dense()
    return asyncio.run(run_forensics(SimulatedAdapter(cfg), config={"fast": True}))


@pytest.fixture(scope="module")
def dense():
    return _run("dense")


@pytest.fixture(scope="module")
def jev():
    return _run("jev")


class _StrictParser(HTMLParser):
    def error(self, message):  # pragma: no cover - only fires on malformed markup
        raise ValueError(message)


def _assert_html_parses(text: str) -> None:
    p = _StrictParser()
    p.feed(text)
    p.close()


def _extract_blob(html_text: str) -> dict:
    m = re.search(r"window\.__FORENSICS__ = (\{.*?\});</script>", html_text, re.S)
    assert m, "window.__FORENSICS__ assignment not found"
    return json.loads(m.group(1).replace("<\\/", "</"))


# =========================================================================== build_report
def test_build_report_writes_three_files(dense, tmp_path):
    out = build_report(dense, str(tmp_path))
    assert set(out) == {"md", "json", "html"}
    for fmt, path in out.items():
        assert path.endswith(_expected_name(fmt))
        with open(path, encoding="utf-8") as fh:
            assert fh.read().strip(), f"{fmt} file is empty"


def test_build_report_respects_formats_subset(dense, tmp_path):
    out = build_report(dense, str(tmp_path), formats=("json",))
    assert set(out) == {"json"}
    with open(out["json"], encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["run_id"] == dense.run_id


def _expected_name(fmt: str) -> str:
    return {"md": "report.md", "json": "report.json", "html": "report.html"}[fmt]


# =========================================================================== markdown
def test_markdown_has_all_sections_and_mermaid(dense):
    md = MD.build_markdown(dense)
    assert "```mermaid" in md
    for letter in "ABCDEFGHIJKL":
        assert f"## {letter}." in md, f"section {letter} missing"


def test_markdown_json_matches_run(dense, tmp_path):
    out = build_report(dense, str(tmp_path))
    data = json.load(open(out["json"], encoding="utf-8"))
    assert data["caps"]["kind"] == "generative"
    assert len(data["probe_results"]) == len(dense.probe_results)


# =========================================================================== html
def test_html_is_interactive_and_parses(dense):
    html_text = HT.build_html(dense)
    assert "window.__FORENSICS__" in html_text
    assert "Inspector" in html_text
    assert "not identified within budget" in html_text     # hidden_size None handled
    _assert_html_parses(html_text)


def test_html_blob_is_valid_json(dense):
    blob = _extract_blob(HT.build_html(dense))
    for key in ("properties", "quantities", "probes", "nodes", "identifiability",
                "family_matches", "styleMap", "arch"):
        assert key in blob
    # hidden_size is present but unresolved (value None) -> best has no value
    hs = blob["quantities"].get("dims.hidden_size")
    assert hs is not None
    if hs["best"] is not None:
        assert hs["best"]["value"] is None


def test_html_nodes_reference_real_inspector_keys(dense):
    blob = _extract_blob(HT.build_html(dense))
    props, quants, probes, notes = (blob["properties"], blob["quantities"],
                                    blob["probes"], blob["notes"])
    for nid, info in blob["nodes"].items():
        key = info["key"]
        if not key:
            continue
        kind, _, name = key.partition(":")
        if kind == "prop":
            assert name in props, f"node {nid} -> missing property {name}"
        elif kind == "quant":
            assert name in quants, f"node {nid} -> missing quantity {name}"
        elif kind == "probe":
            assert name in probes, f"node {nid} -> missing probe {name}"
        elif kind == "note":
            assert key in notes, f"node {nid} -> missing note {key}"


# =========================================================================== diagrams
def test_dense_block_shows_lm_head(dense):
    d = DG.block_architecture(dense)
    text = " ".join(" ".join(n.lines) for n in d.nodes)
    assert "LM head" in text
    assert "choice head" not in text
    # hidden_size unresolved -> an explicit unknown (grey) node in the embedding
    emb = next(n for n in d.nodes if n.id == "b_embed")
    assert any("not identified" in ln for ln in emb.lines)


def test_jev_block_shows_decision_heads_and_branches(jev):
    d = DG.block_architecture(jev)
    ids = {n.id for n in d.nodes}
    assert {"b_q_choice", "b_q_score", "b_q_noul"} <= ids
    text = " ".join(" ".join(n.lines) for n in d.nodes)
    assert "Decision readout heads" in text
    assert "LM head" not in text


def test_topology_only_draws_evidenced_components(dense):
    d = DG.system_topology(dense)
    ids = {n.id for n in d.nodes}
    assert {"t_user", "t_gw", "t_model", "t_resp"} <= ids
    # single_model posterior -> no router/ensemble node drawn
    assert "t_router" not in ids


def test_evidence_flow_links_probes_to_conclusions(dense):
    d = DG.evidence_flow(dense)
    assert d.nodes and d.edges
    node_ids = {n.id for n in d.nodes}
    for e in d.edges:
        assert e.src in node_ids and e.dst in node_ids


def test_every_diagram_emits_mermaid_and_dot(dense):
    for d in DG.all_diagrams(dense).values():
        assert d.mermaid.startswith("flowchart")
        assert d.dot.startswith("digraph")
        assert d.svg is None or isinstance(d.svg, str)   # dot may or may not be installed


def test_multimodal_lanes_drawn_when_detected(dense):
    run = copy.deepcopy(dense)
    run.posterior["modal.input"] = {"text_only": 0.05, "text_image": 0.85,
                                    "text_image_audio": 0.05, "text_audio": 0.05}
    run.posterior["modal.vision_encoder"] = {"none": 0.1, "vit_patch_tokens": 0.7,
                                             "native_resolution_vit": 0.15,
                                             "cross_attn_adapter": 0.05}
    d = DG.block_architecture(run)
    ids = {n.id for n in d.nodes}
    assert {"b_mm_in", "b_vision", "b_fuse"} <= ids

    run.posterior["modal.input"] = {"text_only": 0.05, "text_audio": 0.85,
                                    "text_image": 0.05, "text_image_audio": 0.05}
    d2 = DG.block_architecture(run)
    assert "b_audio" in {n.id for n in d2.nodes}


# =========================================================================== style / svg
def test_style_for_is_pure_and_covers_levels():
    for lvl in ("observed", "inferred", "estimated", "speculative", "unknown", None, ""):
        color, line = ST.style_for(lvl, 0.9)
        assert color.startswith("#") and line in ("solid", "dashed", "dotted")
    # estimated/speculative are dashed/dotted; low confidence downgrades solid to dashed
    assert ST.style_for("estimated", 0.9)[1] == "dashed"
    assert ST.style_for("speculative", 0.9)[1] == "dotted"
    assert ST.style_for("observed", 0.1)[1] == "dashed"
    assert ST.style_for("observed", 0.99)[1] == "solid"


def test_render_svg_degrades_gracefully():
    svg = DG.render_svg("digraph { a -> b }")
    assert svg is None or (isinstance(svg, str) and "<svg" in svg)
    assert DG.render_svg("") is None


def test_charts_skip_absent_series(jev):
    # the decision model has no timing/needle series -> those charts are skipped, no crash
    ch = CH.all_charts(jev)
    assert "belief_bars" in ch
    for uri in ch.values():
        assert uri.startswith("data:image/png;base64,")
