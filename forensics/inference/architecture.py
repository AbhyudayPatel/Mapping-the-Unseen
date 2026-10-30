"""Functional architecture hypotheses assembled from probe evidence.

User-facing output is a partial computational hypothesis graph, not a model-identity
classifier. Hard measured constraints and heuristic support are separate: scores rank
candidates, never discard them automatically. Template dataflow edges are labeled proposed;
only tested dependency edges carry an empirical claim. Prior-only scores are explicit.
"""
from __future__ import annotations
from typing import Any
from forensics.core.types import to_jsonable
from forensics.inference.certificates import build_certificates, provenance

SCHEMA_VERSION = "1.0"
LABELS = {
    "arch.family": "Core family", "arch.sparsity": "Feed-forward topology",
    "attn.long_range": "Attention / state regime", "attn.kv_sharing": "KV sharing",
    "pos.encoding": "Position scheme", "out.logit_precision": "Wire numeric regime",
    "output.head": "Readout family", "dims.hidden_size": "Hidden width",
    "dims.n_layers": "Depth", "tokenizer.family": "Tokenizer",
    "context.max_input_tokens": "Input boundary", "context.max_output_tokens": "Output cap",
    "serving.prompt_caching": "Prefix cache", "serving.speculative_decoding": "Speculative decode",
    "serving.batch_invariant": "Repeat / batch stability", "system.topology": "Serving topology",
    "modal.input": "Accepted modalities", "decision.question_coupling": "Question branches",
    "decision.option_processing": "Option processing", "decision.scoring_mechanism": "Decision readout",
    "decision.probability_resolution": "Probability grid",
    "tokenizer.chat_template_overhead": "Accounting scaffold",
}
NEXT = {
    "arch.family": "Test output-length scaling and common-set geometry under a fixed wrapper; separate decoder, encoder-decoder, and SSM alternatives.",
    "arch.sparsity": "Interleave length-matched domains under controlled load; test repeat jitter and residual timing, retaining scheduler confounds.",
    "attn.kv_sharing": "Measure context-dependent decode/KV growth with known dtype, hardware, and depth controls.",
    "pos.encoding": "Choose length/extrapolation prompts on which candidate positional schemes predict different behavior.",
    "dims.hidden_size": "Expand fixed common logit columns and distinct prompt rows until an output-rank cliff is resolvable.",
    "dims.n_layers": "Combine a measured width with a calibrated compute/queueing model or independent structural telemetry.",
}


def _node(run, key, cert):
    dist = run.posterior.get(key, {})
    prior = run.prior.get(key, {})
    evidence = [e for e in run.evidence if e.property == key]
    estimates = [e for e in run.estimates if e.quantity == key]
    estimate = next((e for e in reversed(estimates) if e.value is not None), None)
    changed = bool(dist) and any(abs(v-prior.get(k, 0)) > 1e-5 for k,v in dist.items())
    backed = bool(evidence or estimate)
    status = cert.get("status", "unresolved")
    level = cert.get("level", "U")
    if not backed or (dist and not changed and not estimate):
        status, level = "unresolved", "U"
    if estimate and estimate.level == "estimated" and key in {
        "context.max_input_tokens", "context.max_output_tokens", "dims.hidden_size"}:
        status, level = "estimated", "L3"
    best = max(dist,key=dist.get) if dist else (estimate.value if estimate else None)
    candidates = cert.get("candidate_set") or list(dist)
    if key == "arch.family" and run.caps.kind == "decision" and backed:
        # Knowledge and serialization support a core hypothesis, not an identified family.
        status, level = "suggestive", "L4"
    return {
        "id": key, "title": LABELS.get(key, key), "provenance": provenance(key),
        "best_supported": best, "status": status, "evidence_grade": level,
        "candidate_values": candidates, "ranked_candidates": [
            {"value": k,"support_score": float(v),"prior_score": prior.get(k),
             "changed_from_prior": abs(v-prior.get(k,0)) > 1e-5}
            for k,v in sorted(dist.items(),key=lambda kv:-kv[1])],
        "score_semantics": "surrogate support, not calibrated architectural probability",
        "prior_only": not backed or (bool(dist) and not changed and estimate is None),
        "estimate": to_jsonable(estimate) if estimate else None,
        "evidence": [{"probe_id":e.probe_id,"summary":e.summary,"statistics":e.stats,
                      "grade":e.level,"alternatives":e.alternatives} for e in evidence],
        "alternatives": list(dict.fromkeys(a for e in evidence for a in e.alternatives)),
        "next_probe": NEXT.get(key,cert.get("cheapest_known_falsifier")),
        "constraint_kind": "finite-reference-match" if key=="tokenizer.family" and cert.get("set_kind")=="finite-reference-match"
                           else "measured-value" if estimate and estimate.level=="observed"
                           else "soft-architectural-evidence" if backed else "unmeasured-candidate-domain",
    }


def compare_architecture_candidates(run_a, run_b) -> list[dict[str, Any]]:
    """Align full candidate domains without confusing a score change with a label change.

    Missing distributions remain null, and unchanged priors remain explicitly flagged.
    Values are relative support, not posterior model-identity probabilities.
    """
    graphs = [architecture_hypothesis_graph(r) for r in (run_a, run_b)]
    nodes = [{n["id"]: n for n in g["nodes"]} for g in graphs]
    keys = list(dict.fromkeys([n["id"] for g in graphs for n in g["nodes"]]))
    rows = []
    for key in keys:
        a, b = nodes[0].get(key), nodes[1].get(key)
        if not ((a and a["ranked_candidates"]) or (b and b["ranked_candidates"])):
            continue
        domain = list(dict.fromkeys([
            c["value"] for n in (a, b) if n for c in n["ranked_candidates"]]))
        def scores(n):
            return {c["value"]: c for c in n["ranked_candidates"]} if n else {}
        sa, sb = scores(a), scores(b)
        def side(n):
            if n is None or not n["ranked_candidates"]:
                return None
            return {"leading": n["best_supported"], "prior_only": n["prior_only"],
                    "grade": n["evidence_grade"], "status": n["status"],
                    "candidates": n["ranked_candidates"],
                    "probes": list(dict.fromkeys(e["probe_id"] for e in n["evidence"]))}
        rows.append({"key": key, "label": LABELS.get(key, key),
            "a": side(a), "b": side(b),
            "same_leading": bool(a and b and a["best_supported"] == b["best_supported"]),
            "candidates": [{"value": v, "a_score": sa.get(v, {}).get("support_score"),
                            "b_score": sb.get(v, {}).get("support_score"),
                            "a_prior": sa.get(v, {}).get("prior_score"),
                            "b_prior": sb.get(v, {}).get("prior_score")} for v in domain]})
    return rows


def architecture_hypothesis_graph(run) -> dict[str, Any]:
    certificates = {c["property"]:c for c in build_certificates(run)}
    common = ["tokenizer.family","tokenizer.chat_template_overhead","arch.family",
              "output.head","dims.hidden_size","dims.n_layers","out.logit_precision",
              "serving.prompt_caching","serving.batch_invariant","system.topology","arch.sparsity"]
    generative = ["attn.long_range","attn.kv_sharing","pos.encoding",
                  "context.max_input_tokens","context.max_output_tokens",
                  "serving.speculative_decoding","modal.input"]
    decision = ["decision.question_coupling","decision.option_processing",
                "decision.scoring_mechanism","decision.probability_resolution"]
    keys = common + (decision if run.caps.kind=="decision" else generative)
    nodes = [_node(run,key,certificates.get(key,{})) for key in keys]
    edges=[]
    def edge(source,target,label,relation="candidate_dataflow",probe_ids=None):
        edges.append({"source":source,"target":target,"label":label,"relation":relation,
                      "probe_ids":probe_ids or [],
                      "status":"observed-on-tested-inputs" if relation!="candidate_dataflow" and probe_ids else "proposed"})
    edge("tokenizer.family","arch.family","tokenized input to candidate core")
    edge("arch.family","output.head","candidate core to readout")
    if run.caps.kind=="decision":
        edge("arch.family","decision.scoring_mechanism","core to candidate decision readout")
        edge("decision.question_coupling","decision.option_processing","branch-local question/options",
             "tested_information_dependency",[e.probe_id for e in run.evidence if e.property=="decision.question_coupling"])
        edge("decision.scoring_mechanism","decision.probability_resolution","readout to exposed probability grid")
    else:
        for key in ["attn.long_range","attn.kv_sharing","pos.encoding","arch.sparsity"]:
            edge("arch.family",key,"candidate internal component; topology conditional on core")
        edge("modal.input","arch.family","accepted modality to unknown fusion path")
    summary={}
    for n in nodes:summary[n['status']]=summary.get(n['status'],0)+1
    return {"schema_version":SCHEMA_VERSION,"target":run.target,"endpoint_kind":run.caps.kind,
            "objective":"partial functional architecture reconstruction from adaptive API experiments",
            "nodes":nodes,"edges":edges,"summary":summary,
            "recovery_scope":"component-level; no full graph-isomorphism or model-identity claim",
            "hard_constraint_policy":"intersect only valid measured constraints within their assumptions",
            "soft_constraint_policy":"rank remaining architecture candidates; no automatic elimination by support threshold",
            "graph_semantics":"candidate functional dataflow with empirical component constraints; proposed edges are not recovered physical connections"}
