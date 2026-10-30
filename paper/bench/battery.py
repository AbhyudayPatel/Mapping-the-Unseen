"""The ForensicsBench battery: one fixed, frozen protocol per subject.

Runs the production probe classes (P02/P04/P06/P07/P10/P11/P19) plus benchmark
modules (contract checks, KV-growth decode test, three-regime output geometry)
through the anonymous wrapper, then assembles the same belief/certificates/
identifiability artefacts a production scan produces.  Ground truth is recorded
alongside, but never shown to the probes/estimators.

The protocol (probe set, config, budgets, rank gates) is FROZEN here before any
labels are consulted; ``battery_protocol_hash()`` pins it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time

import numpy as np
import httpx

from forensics.core.types import Capabilities, GenRequest, to_jsonable
from forensics.core.budget import Budget
from forensics.core.recorder import Recorder
from forensics.core.adapters.openai_compat import OpenAICompatAdapter
from forensics.probes.base import ProbeContext
from forensics.probes.p02_tokenizer import TokenizerFingerprintProbe
from forensics.probes.p04_precision import LogitPrecisionProbe
from forensics.probes.p06_positional import NeedleRecallProbe
from forensics.probes.p07_timing import TimingScalingProbe
from forensics.probes.p10_determinism import DeterminismProbe
from forensics.probes.p11_moe import MoEProbe
from forensics.probes.p19_archfamily import GenerationCapabilityProbe, PrefillScalingProbe
from forensics.probes.p23_decodestate import DecodeStateGrowthProbe
from forensics.inference.belief import Belief, apply_coupling
from forensics.inference.estimator import estimate_dimensions
from forensics.inference.identifiability import analyze as ident_analyze
from forensics.inference.certificates import build_certificates
from forensics.inference.architecture import architecture_hypothesis_graph
from forensics.inference.run_types import RunResult

from wrapper import ArchitectureWrapper


# --------------------------------------------------------------------------- protocol
PROTOCOL = {
    # bench-2.0: repaired detectors (P19 material-curvature gate + cap-rejection
    # encoder evidence; P23 calibrated unit-fixed slope, 6 reps, per-length min),
    # single-pass coupling, 24576-token caps where the model allows.
    "version": "bench-2.0",
    "kv_growth": {"lengths": [64, 128, 256, 512, 1024, 2048], "out_tokens": 8, "reps": 2},
    "geometry_columns": {"cols": 2560, "row_ladder": [384, 768, 1536, 2560],
                         "rel_threshold": 1e-6},
    "geometry_top5": {"prompts": 768, "k": 5},
    "needle": {"target": 1536, "positions": [0.1, 0.5, 0.9]},
}


def battery_protocol_hash() -> str:
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- modules
async def contract_checks(ctx: ProbeContext) -> dict:
    """P01/P05-style interface and limits reads."""
    out = {"checks": []}
    async def _rec(name, req):
        r = await ctx.gen(req)
        out["checks"].append({"name": name, "ok": bool(r.ok), "error": r.error,
                              "usage": to_jsonable(r.usage)})
        return r
    await _rec("minimal", GenRequest.chat("hi", max_tokens=1))
    await _rec("empty", GenRequest.chat("", max_tokens=1))
    await _rec("over_output", GenRequest.chat("hi", max_tokens=10_000))
    long_prompt = "word " * 4100
    await _rec("over_input", GenRequest.chat(long_prompt, max_tokens=1))
    r = await _rec("bias_probe", GenRequest.chat("hi", max_tokens=1,
                                                 extra={"logit_bias": {"16": 5.0}}))
    out["logit_bias_accepted"] = bool(r.ok)
    return out


def _spectral(matrix: np.ndarray, rel: float):
    a = np.asarray(matrix, float)
    if a.ndim != 2 or a.shape[0] < 3:
        return {"status": "unresolved", "reason": "insufficient rows"}
    a = a - a.mean(1, keepdims=True)
    sv = np.linalg.svd(a, compute_uv=False)
    tol = rel * sv[0]
    rank = int(np.sum(sv > tol))
    point = rank < min(a.shape) - 4
    return {"status": "estimated" if point else "unresolved",
            "rank_estimate": rank if point else None,
            "observed_rank_lower_bound": rank,
            "singular_values_head": sv[:16].tolist(),
            "rows": int(a.shape[0]), "columns": int(a.shape[1])}


async def geometry_columns(ctx: ProbeContext, vocab: int) -> dict:
    """Rich selected-column acquisition with an adaptive row ladder (>2×d)."""
    spec = PROTOCOL["geometry_columns"]
    rng = np.random.default_rng(11)
    cols = np.sort(rng.choice(vocab, min(spec["cols"], vocab - 2), replace=False) + 1)
    words = ["engine", "river", "blue", "silent", "paper", "code", "vector", "memory",
             "city", "cloud", "summer", "design", "history", "token", "calculus",
             "market", "copper", "moon", "ocean", "tree"]
    matrix, stage_log = [], []
    for rows_target in spec["row_ladder"]:
        while len(matrix) < rows_target:
            row_i = len(matrix)
            length = int(rng.integers(8, 45))
            prompt = " ".join(rng.choice(words, size=length)) + f" Case {row_i}: {row_i*7919%100003}."
            r = await ctx.gen(GenRequest.chat(
                prompt, max_tokens=1, logprobs=True, top_logprobs=5,
                extra={"measurement_columns": [int(c) for c in cols]}))
            vals = {t.token_id: t.logprob for t in r.tokens[0].top} if r.ok and r.tokens else {}
            if all(int(c) in vals for c in cols):
                matrix.append([vals[int(c)] for c in cols])
            if ctx.budget and ctx.n_requests >= ctx.budget.max_requests - 5:
                break
        est = _spectral(np.array(matrix), spec["rel_threshold"])
        stage_log.append({"rows": len(matrix), "status": est["status"],
                          "rank": est.get("rank_estimate"),
                          "lower_bound": est["observed_rank_lower_bound"]})
        if est["status"] == "estimated":
            break
        if ctx.budget and ctx.n_requests >= ctx.budget.max_requests - 5:
            break
    return {"final": stage_log[-1] if stage_log else {}, "ladder": stage_log,
            "columns": len(cols), "access": "selected-output-column diagnostic (rich)"}


async def geometry_top5(ctx: ProbeContext) -> dict:
    """Production-realistic top-5-only acquisition (union of natural top-k)."""
    spec = PROTOCOL["geometry_top5"]
    rng = np.random.default_rng(13)
    words = ["engine", "river", "blue", "silent", "paper", "code", "vector", "memory",
             "city", "cloud", "summer", "design", "history", "token", "calculus",
             "market", "copper", "moon", "ocean", "tree"]
    cells: dict[int, dict[int, float]] = {}
    for row_i in range(spec["prompts"]):
        length = int(rng.integers(8, 45))
        prompt = " ".join(rng.choice(words, size=length)) + f" Case {row_i}."
        r = await ctx.gen(GenRequest.chat(prompt, max_tokens=1, logprobs=True,
                                          top_logprobs=spec["k"]))
        if r.ok and r.tokens:
            cells[row_i] = {int(t.token_id): float(t.logprob) for t in r.tokens[0].top}
        if ctx.budget and ctx.n_requests >= ctx.budget.max_requests - 5:
            break
    counts: dict[int, int] = {}
    for row in cells.values():
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    keep = [c for c, n in counts.items() if n >= max(8, len(cells) // 4)]
    keep = sorted(keep, key=lambda c: -counts[c])[:320]
    matrix = [[row[c] for c in keep] for row in cells.values()
              if all(c in row for c in keep)]
    est = _spectral(np.array(matrix), 1e-6) if len(matrix) >= 8 and keep else {
        "status": "unresolved", "reason": "top-5 union too sparse"}
    est["union_columns"] = len(counts)
    est["dense_columns"] = len(keep)
    est["dense_rows"] = len(matrix)
    est["access"] = "top-5 only (production-realistic)"
    return est


# --------------------------------------------------------------------------- battery
async def run_battery(subj: dict, wrapper: ArchitectureWrapper, out_dir: str,
                      request_budget: int = 3200) -> dict:
    """Run the frozen protocol against one served subject."""
    try:
        vocab = int(wrapper.a.model.get_output_embeddings().weight.shape[0]) \
            if wrapper.a.kind != "masked" else len(wrapper.a.tok)
    except Exception:  # quantised heads expose no plain .weight
        vocab = int(getattr(wrapper.a.model.config, "vocab_size", 0) or len(wrapper.a.tok))
    rec = Recorder(out_dir)
    budget = Budget(max_requests=request_budget, max_output_tokens=200_000)
    caps = Capabilities(
        kind="generative" if wrapper.a.kind != "masked" else "generative",
        chat=True, logprobs=True, max_top_logprobs=20, logit_bias=True,
        logit_bias_affects_logprobs=True, usage_prompt_tokens=True,
        token_ids_known=True, streaming=(wrapper.a.kind != "masked"),
        server_timing_header="x-processing-time")
    t0 = time.time()
    out: dict = {"subject": subj["key"], "protocol": battery_protocol_hash(),
                 "probe_results": [], "modules": {}}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapper.app),
                                 base_url="http://bench") as client:
        ad = OpenAICompatAdapter(base_url="http://bench/v1",
                                 model="opaque-architecture", api_key="")
        ad._client = client
        ctx = ProbeContext(ad, caps=caps, budget=budget, recorder=rec, seed=23,
                           concurrency=1, config={"sample_scale": 1.0})

        ctx.current_probe = "B.contract"
        out["modules"]["contract"] = await contract_checks(ctx)

        probes = [TokenizerFingerprintProbe(), GenerationCapabilityProbe(),
                  PrefillScalingProbe(), LogitPrecisionProbe(), TimingScalingProbe(),
                  DecodeStateGrowthProbe(), DeterminismProbe(), MoEProbe()]
        trained = not subj.get("tiny_random_init", False)
        if trained and wrapper.a.kind != "masked":
            probes.append(NeedleRecallProbe())
        for probe in probes:
            ctx.current_probe = probe.id
            try:
                res = await probe.run(ctx)
                out["probe_results"].append(to_jsonable(res))
            except Exception as exc:  # keep going; a failed probe is data
                out["probe_results"].append({"probe_id": probe.id, "status": "error",
                                             "notes": [repr(exc)]})
            if budget.requests >= budget.max_requests - 5:
                break

        if wrapper.a.kind == "causal" and budget.requests < budget.max_requests - 200:
            ctx.current_probe = "B.geometry_columns"
            out["modules"]["geometry_columns"] = await geometry_columns(ctx, vocab)
            ctx.current_probe = "B.geometry_top5"
            out["modules"]["geometry_top5"] = await geometry_top5(ctx)

    out["budget"] = budget.summary()
    out["wall_s"] = round(time.time() - t0, 1)
    out["scan"] = finalize(out, subj, caps)
    return out


# --------------------------------------------------------------------------- finalize
def finalize(out: dict, subj: dict, caps: Capabilities) -> dict:
    """Assemble belief / estimates / certificates from the frozen battery exactly as a
    production scan would (same ranker, same coupling, same identifiability analyser)."""
    from forensics.probes.base import Estimate, Evidence, ProbeResult

    probe_results: list[ProbeResult] = []
    evidence: list[Evidence] = []
    estimates: list[Estimate] = []
    for pr in out.get("probe_results", []):
        if not isinstance(pr, dict):
            continue
        pres = ProbeResult(probe_id=pr.get("probe_id", ""), status=pr.get("status", "ok"))
        for e in pr.get("evidence", []) or []:
            ev = Evidence(property=e["property"], likelihoods=e.get("likelihoods") or {},
                          summary=e.get("summary", ""), level=e.get("level", "inferred"),
                          strength=e.get("strength", 1.0), stats=e.get("stats") or {},
                          alternatives=e.get("alternatives") or [], probe_id=e.get("probe_id", ""),
                          experiment_group=e.get("experiment_group", ""))
            pres.evidence.append(ev)
            evidence.append(ev)
        for q in pr.get("estimates", []) or []:
            est = Estimate(quantity=q["quantity"], value=q.get("value"),
                           ci=tuple(q["ci"]) if q.get("ci") else None,
                           level=q.get("level", "estimated"), method=q.get("method", ""),
                           identifiability=q.get("identifiability", 2),
                           unit=q.get("unit", ""), probe_id=q.get("probe_id", ""))
            pres.estimates.append(est)
            estimates.append(est)
        probe_results.append(pres)

    geo = (out.get("modules") or {}).get("geometry_columns") or {}
    fin = geo.get("final") or {}
    if fin.get("status") == "estimated" and fin.get("rank"):
        estimates.append(Estimate(
            quantity="dims.hidden_size", value=int(fin["rank"]),
            level="estimated", method="selected-column common-output rank (bench)",
            assumptions=["full-rank output projection", "row-centered log-softmax offsets"],
            identifiability=2, probe_id="B.geometry_columns"))
    elif fin.get("lower_bound"):
        estimates.append(Estimate(
            quantity="dims.hidden_size", value=None,
            level="estimated", method="selected-column common-output rank (bench)",
            assumptions=[f"acquisition saturated: lower bound {fin['lower_bound']}"],
            identifiability=2, probe_id="B.geometry_columns"))

    kind = "generative"
    belief = Belief(kind=kind)
    belief.apply_all_grouped(evidence)          # primary: group-aware combination
    apply_coupling(belief)
    belief_legacy = Belief(kind=kind)           # sensitivity: legacy independent factors
    belief_legacy.apply_all(evidence)
    apply_coupling(belief_legacy)
    belief_flat = Belief(kind=kind)             # evidence-only view: uniform priors
    for k in belief_flat.posterior:
        u = 1.0 / len(belief_flat.posterior[k])
        belief_flat.prior[k] = {v: u for v in belief_flat.posterior[k]}
        belief_flat.posterior[k] = {v: u for v in belief_flat.posterior[k]}
    belief_flat.apply_all_grouped(evidence)
    apply_coupling(belief_flat)

    ran = {p.probe_id for p in probe_results}
    notes: list[str] = []
    try:
        est2, fams = estimate_dimensions(belief, estimates, caps, {})
        estimates = estimates + list(est2 or [])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"estimator error: {exc!r}")
        fams = []
    ident = ident_analyze(belief, estimates, caps, ran)

    run = RunResult(run_id=f"bench-{subj['key']}", target={"kind": "bench"}, caps=caps,
                    probe_results=probe_results, prior=belief.prior_dict(),
                    posterior=belief.as_dict(), evidence=evidence, estimates=estimates,
                    identifiability=ident, notes=notes)
    certs = build_certificates(run)
    try:
        graph = architecture_hypothesis_graph(run)
    except Exception as exc:  # noqa: BLE001
        graph = {"error": repr(exc)}
    return {"posterior": belief.as_dict(), "prior": belief.prior_dict(),
            "posterior_legacy": belief_legacy.as_dict(),
            "posterior_flatprior": belief_flat.as_dict(),
            "n_evidence": len(evidence), "estimates": to_jsonable(estimates),
            "identifiability": to_jsonable(ident), "certificates": to_jsonable(certs),
            "family_matches": to_jsonable(fams), "graph": graph, "notes": notes}
