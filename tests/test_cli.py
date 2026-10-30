"""CLI tests — all offline via the simulator presets (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forensics.cli import (
    PRESETS, app, build_dot, build_mermaid, fallback_build_report, resolve_target,
    run_result_from_dict,
)

runner = CliRunner()


def _run(*args: str):
    result = runner.invoke(app, list(args))
    if result.exit_code != 0:  # surface the traceback on failure
        print(result.output)
        if result.exception:
            import traceback
            traceback.print_exception(type(result.exception), result.exception,
                                      result.exception.__traceback__)
    return result


def _only_subdir(parent: Path) -> Path:
    subs = [p for p in parent.iterdir() if p.is_dir()]
    assert len(subs) == 1, f"expected one run dir, got {subs}"
    return subs[0]


# --------------------------------------------------------------------------- static
def test_probes_lists_at_least_30():
    result = _run("probes")
    assert result.exit_code == 0
    # the table title reports the count, e.g. "probe registry (35 probes)"
    from forensics.probes import load_all
    from forensics.probes.base import REGISTRY
    load_all()
    assert len(REGISTRY) >= 30
    assert "probe registry" in result.output


def test_presets_include_simulators():
    assert "sim-dense" in PRESETS and PRESETS["sim-dense"]["provider"] == "sim"
    assert "sim-jev" in PRESETS and PRESETS["sim-jev"]["kind"] == "decision"
    result = _run("presets")
    assert result.exit_code == 0
    assert "sim-dense" in result.output


def test_presets_reference_env_names_not_secrets():
    # every non-sim preset references env var NAMES, never inline URLs/keys
    for name, p in PRESETS.items():
        if p["provider"] == "sim":
            continue
        assert p.get("base_url_env"), name
        assert p.get("api_key_env"), name
        # no value that looks like a real URL or key literal
        assert "http" not in json.dumps(p).lower()


def test_resolve_target_preset_and_provider():
    t = resolve_target("sim-dense", None, None, None, None, None)
    assert t.provider == "sim" and t.sim == "sim_decoder_dense"
    t2 = resolve_target(None, "azure", "https://x/openai/v1", "gpt-4o-mini",
                        "AZURE_OPENAI_API_KEY", "api-key")
    assert t2.provider == "azure" and t2.model == "gpt-4o-mini"


# --------------------------------------------------------------------------- run
@pytest.fixture(scope="module")
def dense_run(tmp_path_factory) -> Path:
    runs = tmp_path_factory.mktemp("runs")
    result = _run("run", "--preset", "sim-dense", "--fast", "--runs-dir", str(runs))
    assert result.exit_code == 0, result.output
    return _only_subdir(runs)


def test_run_writes_report_html_md_json(dense_run: Path):
    for fname in ("report.html", "report.md", "report.json", "run_result.json",
                  "observations.jsonl", "architecture.mmd", "architecture.dot"):
        assert (dense_run / fname).exists(), f"missing {fname}"
    doc = (dense_run / "report.html").read_text(encoding="utf-8")
    assert "<html" in doc.lower() and "forensics" in doc.lower()
    rr = json.loads((dense_run / "report.json").read_text(encoding="utf-8"))
    assert rr["run_id"] and rr["estimates"] and rr["probe_results"]
    # top architecture hypothesis for a dense decoder should be decoder_only
    fam = rr["posterior"]["arch.family"]
    assert max(fam, key=fam.get) == "decoder_only"


def test_run_result_roundtrips(dense_run: Path):
    d = json.loads((dense_run / "run_result.json").read_text(encoding="utf-8"))
    rr = run_result_from_dict(d)
    assert rr.run_id == d["run_id"]
    assert len(rr.estimates) == len(d["estimates"])
    assert len(rr.probe_results) == len(d["probe_results"])
    # estimate CI is coerced back to a tuple where present
    for e in rr.estimates:
        assert e.ci is None or isinstance(e.ci, tuple)


# --------------------------------------------------------------------------- replay
def test_replay_reanalyses_offline(dense_run: Path):
    result = _run("replay", str(dense_run), "--fast")
    assert result.exit_code == 0, result.output
    out = dense_run / "replay"
    assert (out / "report.json").exists()
    assert (out / "run_result.json").exists()


# --------------------------------------------------------------------------- report
def test_report_from_saved_result(dense_run: Path, tmp_path: Path):
    result = _run("report", str(dense_run), "--formats", "md,json", "--out", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()


def test_report_via_replay_reanalysis(dense_run: Path, tmp_path: Path):
    # --replay ignores the saved RunResult and rebuilds it from observations
    result = _run("report", str(dense_run), "--replay", "--formats", "md",
                  "--out", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.md").exists()


# --------------------------------------------------------------------------- diagram
def test_diagram_emits_sources(dense_run: Path, tmp_path: Path):
    result = _run("diagram", str(dense_run), "--out", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "architecture.mmd").exists()
    assert (tmp_path / "architecture.dot").exists()
    mmd = (tmp_path / "architecture.mmd").read_text(encoding="utf-8")
    assert "flowchart" in mmd


def test_diagram_helpers_build_from_result(dense_run: Path):
    rr = run_result_from_dict(
        json.loads((dense_run / "run_result.json").read_text(encoding="utf-8")))
    assert "flowchart" in build_mermaid(rr)
    assert "digraph" in build_dot(rr)


# --------------------------------------------------------------------------- compare
def test_compare_from_runs(dense_run: Path, tmp_path_factory):
    # produce a second run (moe) then compare the two saved dirs offline
    runs = tmp_path_factory.mktemp("cmp")
    r = _run("run", "--preset", "sim-moe", "--fast", "--runs-dir", str(runs))
    assert r.exit_code == 0, r.output
    moe_run = _only_subdir(runs)
    result = _run("compare", str(dense_run), str(moe_run), "--from-runs")
    assert result.exit_code == 0, result.output
    assert "verdict" in result.output


def test_compare_two_presets_offline():
    result = _run("compare", "sim-dense", "sim-dense", "--fast")
    assert result.exit_code == 0, result.output
    # identical targets should agree strongly
    assert "verdict: same" in result.output or "agreement 100%" in result.output


# --------------------------------------------------------------------------- fallback report
def test_fallback_report_writes_all_formats(dense_run: Path, tmp_path: Path):
    rr = run_result_from_dict(
        json.loads((dense_run / "run_result.json").read_text(encoding="utf-8")))
    written = fallback_build_report(rr, tmp_path, ["html", "md", "json"])
    assert any(p.endswith("report.html") for p in written)
    assert any(p.endswith("report.md") for p in written)
    assert any(p.endswith("report.json") for p in written)
