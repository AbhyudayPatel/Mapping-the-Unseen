"""Auditable claim certificates, separate from legacy capability-level scores.

A missing probe/capability never proves non-identifiability. Positive entries point to
measurements; unresolved entries retain candidates and a next experiment. Formal
non-identifiability is only accepted with an explicit scoped witness. Integrity grades
source control, not statistical confidence. Empty/unknown metadata is preserved as null.
"""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forensics.core.types import to_jsonable

SCHEMA_VERSION = "1.0"
PROVENANCE = {
    "tokenizer.family": "preprocessing", "dims.vocab_size": "preprocessing",
    "tokenizer.chat_template_overhead": "preprocessing",
    "context.max_input_tokens": "interface", "context.max_output_tokens": "interface",
    "decision.probability_resolution": "interface", "modal.input": "interface",
    "modal.output": "interface", "out.logit_precision": "interface",
    "out.logit_dtype": "interface", "output.head": "model",
    "decision.question_coupling": "interface", "decision.option_processing": "interface",
    "decision.scoring_mechanism": "interface",
}
FALSIFIERS = {
    "tokenizer.family": "query a reference-count discriminator for every surviving pair; test preprocessing/accounting alternatives",
    "dims.vocab_size": "validate exposed token IDs and reserved-token behavior; otherwise retain a reference-derived set only",
    "arch.family": "hold the wrapper constant; compare generation and long-context timing across independently known families",
    "arch.sparsity": "length-matched, interleaved domain prompts with repeated measurements and a permutation test of residual timing",
    "pos.encoding": "extend the length grid and predeclare scheme-specific extrapolation tests",
    "out.logit_precision": "repeat lattice tests under transport rounding and temperature controls",
    "out.logit_dtype": "repeat lattice tests under transport rounding and temperature controls",
    "serving.prompt_caching": "randomized repeated-prefix versus nonce-control pairs; report the paired confidence interval",
    "dims.hidden_size": "expand the common-logit set or use an independently validated restricted-API estimator",
    "dims.n_layers": "fit a validated compute/queueing model, controlling width, hardware, batching, and cache",
    "decision.scoring_mechanism": "hold option count and labels fixed while changing state length; compare physical readout models",
}


def provenance(key: str) -> str:
    if key in PROVENANCE: return PROVENANCE[key]
    if key.startswith(("serving.", "system.", "timing.")): return "serving"
    if key.startswith("tokenizer."): return "preprocessing"
    return "model"


def measurement_integrity(probe_id: str) -> dict[str, Any]:
    # A single probe can mix sources; keep the conservative source and list client
    # clocks separately. These grades make no allegation of provider manipulation.
    prefix = probe_id.split(".", 1)[0]
    if prefix in {"P02", "P05"}:
        return {"grade": "I0", "source": "provider accounting or stated constraint",
                "provider_controlled": True, "attested": False}
    if prefix in {"P07", "P08", "P09", "P11", "P22"}:
        return {"grade": "mixed", "source": "I0 server timing; I2 client clock if available",
                "provider_controlled": True, "attested": False}
    return {"grade": "I1", "source": "payload observable at the endpoint boundary",
            "provider_controlled": True, "attested": False}


def valid_witness(witness: dict[str, Any] | None) -> bool:
    return bool(witness and all(witness.get(k) for k in
                ("omega_1", "omega_2", "channels", "probe_class", "different_property_values", "argument")))


def certify_nonidentifiability(key: str, witness: dict[str, Any]) -> dict[str, Any]:
    if not valid_witness(witness):
        raise ValueError("non-identifiability requires two configurations, a channel/probe scope, unequal property values, and an equivalence argument")
    return {"schema_version": SCHEMA_VERSION, "property": key,
            "status": "non-identifiable", "level": "L5", "witness": witness,
            "provenance": provenance(key)}


def build_certificates(run) -> list[dict[str, Any]]:
    evidence = {}
    estimates = {}
    for e in run.evidence: evidence.setdefault(e.property, []).append(e)
    for e in run.estimates:
        if e.value is not None: estimates[e.quantity] = e
    legacy = {e.key: e for e in run.identifiability}
    results = {p.probe_id: p for p in run.probe_results}
    keys = sorted(set(run.posterior) | set(estimates) | set(legacy))
    out = []
    for key in keys:
        evs = evidence.get(key, [])
        dist = run.posterior.get(key, {})
        est = estimates.get(key)
        ids = list(dict.fromkeys([e.probe_id for e in evs] + ([est.probe_id] if est else [])))
        # Scores never eliminate candidates: an empirical compatibility set requires a
        # test model. Return the declared domain unless a finite reference comparison exists.
        candidates = list(dist)
        reference_set = None
        status, level = "unresolved", "U"
        if key == "tokenizer.family":
            p02 = next((p for p in run.probe_results if p.probe_id.startswith("P02")), None)
            rows = p02.tables.get("tokenizer_scores", []) if p02 else []
            matched = [r["family"] for r in rows if r.get("n") and r.get("exact", 0) / r["n"] >= 0.85]
            exact = [r["family"] for r in rows if r.get("n") and r.get("exact", 0) == r["n"]]
            reference_set = exact or matched
            if reference_set:
                candidates = reference_set
                status, level = ("identified", "L2") if exact else ("suggestive", "L4")
            else:
                candidates = ["unrepresented tokenizer", "preprocessing", "accounting mismatch", "custom tokenizer"]
        elif key == "dims.vocab_size" and est:
            status, level = "suggestive", "L4"
            candidates = list(est.ci) if est.ci else [est.value]
        elif key in {"decision.question_coupling", "decision.option_processing"} and evs:
            status, level = "identified", "L2"
        elif key.startswith("timing.") and est:
            status, level = "estimated", "L3"
        elif est and est.level == "observed":
            status, level = "observed", "L1"
        elif key in {"arch.family", "arch.sparsity", "pos.encoding", "attn.kv_sharing",
                     "output.head", "decision.scoring_mechanism"}:
            if evs: status, level = "suggestive", "L4"
        elif key.startswith("serving.") or key.startswith("system."):
            # A negative signature never establishes absence globally.
            if evs: status, level = "suggestive", "L4"
        elif evs:
            best = max(evs, key=lambda e: e.strength)
            status = "observed" if best.level == "observed" else "suggestive"
            level = "L1" if status == "observed" else "L4"
        if key in {"dims.hidden_size", "dims.n_layers", "dims.n_experts", "dims.n_heads"}:
            if not est or est.identifiability >= 4:
                status, level = "unresolved", "U"
        winner = max(dist, key=dist.get) if dist else None
        groups = sorted({pid.split(".", 1)[0] for pid in ids})
        out.append({
            "schema_version": SCHEMA_VERSION, "property": key,
            "status": status, "level": level, "provenance": provenance(key),
            "candidate_set": candidates, "set_kind": "finite-reference-match" if reference_set else "declared-domain; not a confidence set",
            "legacy_leading_value": winner, "legacy_support_score": dist.get(winner) if winner else None,
            "observation_references": ids,
            "observations": [{"probe_id": e.probe_id, "summary": e.summary,
                              "test": e.stats.get("test"), "statistics": e.stats,
                              "factor_kind": e.factor_kind,
                              "integrity": e.measurement_integrity or measurement_integrity(e.probe_id)} for e in evs],
            "alternatives": list(dict.fromkeys(a for e in evs for a in e.alternatives)),
            "cheapest_known_falsifier": FALSIFIERS.get(key, "predeclare a controlled differential experiment for the retained alternatives"),
            "falsifier_cost_verified": False,
            "dependence_groups": groups,
            "dependence_warning": "legacy scores multiply tempered factors; shared observations are not proven independent",
            "structural_status": "not_assessed", "nonidentifiability_witness": None,
            "unresolved_reason": legacy[key].reason if key in legacy else "available evidence does not isolate an implementation",
        })
    return out


def canonical_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def run_manifest(run, root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    def git(*args):
        try: return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError): return None
    source_paths = [p for d in ("forensics/probes", "forensics/inference") for p in (root/d).glob("*.py")]
    sources = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(source_paths)}
    target = run.target or {}
    payload = {"schema_version": SCHEMA_VERSION, "run_id": run.run_id,
        "endpoint": target.get("name"), "api_route": target.get("base_url"),
        "provider": target.get("provider"), "deployment": target.get("model"),
        "region": None, "started_at": run.started_at, "finished_at": run.finished_at,
        "manifest_created_at": datetime.now(timezone.utc).isoformat(),
        "export_python_version": platform.python_version(),
        "client_version": run.config.get("client_version"),
        "probe_source_hash": canonical_hash(sources), "source_files": sources,
        "source_hash_scope": "export-time source snapshot; not a retroactive record of the generating version",
        "original_source_revision": run.config.get("source_revision"),
        "planner_version": "surrogate-v1", "planner": run.config.get("planner", "eig"),
        "seed": run.config.get("seed"), "git_commit": git("rev-parse", "HEAD"),
        "git_snapshot_scope": "export-time checkout; original commit may be unknown",
        "git_dirty": bool(git("status", "--porcelain")),
        "channel_capabilities": to_jsonable(run.caps),
        "unknown_metadata_policy": "null means not recorded; hashes attest bytes, not provider honesty"}
    return {"manifest": payload, "sha256": canonical_hash(payload)}


def export_certificates(run, directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    certificates = build_certificates(run)
    manifest = run_manifest(run)
    from forensics.inference.architecture import architecture_hypothesis_graph
    graph = architecture_hypothesis_graph(run)
    (directory / "architecture_hypothesis.json").write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    (directory / "forensic_certificates.json").write_text(json.dumps(certificates, indent=2, ensure_ascii=False),
                                                       encoding="utf-8", newline="\n")
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                              encoding="utf-8", newline="\n")
    return {"certificates": len(certificates), "manifest_sha256": manifest["sha256"]}
