"""Self-contained interactive single-file HTML report.

No server and no network required beyond a Mermaid CDN script (with a graceful
fallback: if Mermaid does not load, each diagram degrades to clickable node "chips"
and its source, and the tables/inspector keep working). Every diagram node and every
estimate/property/experiment row is clickable and opens an **Inspector** side panel
that shows *how that fact was inferred*: the probe(s), the raw observation, the
statistical test / CI, the likelihoods, the prior→posterior delta, alternatives, and
the epistemic + identifiability level.

The whole interactive dataset is emitted as ``window.__FORENSICS__`` and wired up by
inline vanilla JS. Colours follow the shared confidence/epistemic mapping.
"""
from __future__ import annotations

import html as _h
import json
from typing import Any

from forensics.core.types import to_jsonable
from forensics.inference import identifiability as IDF
from forensics.inference.hypotheses import PROPERTIES, QUANTITIES
from forensics.inference.run_types import RunResult
from forensics.inference.certificates import build_certificates
from forensics.inference.architecture import architecture_hypothesis_graph

from . import charts as CH
from . import diagrams as DG
from . import extract as X
from .style import (
    IDENT_LABELS, canonical_level, confidence_word, ident_confidence,
    ident_to_level, legend_rows, level_style,
)


# --------------------------------------------------------------------------- public
def build_html(run: RunResult, diags: dict[str, DG.Diagram] | None = None,
               charts: dict[str, str] | None = None) -> str:
    diags = diags or DG.all_diagrams(run)
    charts = charts if charts is not None else CH.all_charts(run)
    blob = _blob(run, diags, charts)
    blob_json = json.dumps(blob, ensure_ascii=False, default=_json_default)
    blob_json = blob_json.replace("</", "<\\/")  # keep it safe inside <script>

    body = _body(run, diags, charts)
    return _PAGE.format(
        title=_h.escape(str(run.target.get("name", run.run_id))),
        css=_CSS, body=body, blob=blob_json, js=_JS,
    )


# --------------------------------------------------------------------------- data blob
def _key_for(run: RunResult, raw: str) -> str:
    if raw in QUANTITIES or X.best_estimate(run, raw) is not None:
        return f"quant:{raw}"
    if raw in PROPERTIES or raw in run.posterior:
        return f"prop:{raw}"
    return f"note:{raw}"


def _blob(run: RunResult, diags: dict[str, DG.Diagram], charts: dict[str, str]
          ) -> dict[str, Any]:
    # ---- properties ---------------------------------------------------------
    properties: dict[str, Any] = {}
    for prop in run.posterior:
        dist = run.posterior.get(prop, {})
        prior = run.prior.get(prop, {})
        values = [{"value": v, "prior": float(prior.get(v, 0.0)), "post": float(p)}
                  for v, p in sorted(dist.items(), key=lambda kv: -kv[1])]
        ie = X.identifiability_for(run, prop)
        evs = [_ev_dict(ev) for ev in X.evidence_for_property(run, prop)]
        top_v, top_p = X.posterior_top(run, prop)
        properties[prop] = {
            "desc": X.property_desc(prop),
            "top": {"value": top_v, "post": top_p},
            "level": ident_to_level(ie.level) if ie else "unknown",
            "values": values,
            "evidence": evs,
            "identifiability": _ie_dict(ie),
        }

    # ---- quantities ---------------------------------------------------------
    quantities: dict[str, Any] = {}
    qkeys = set(QUANTITIES) | {e.quantity for e in run.estimates} | \
        {ie.key for ie in run.identifiability if ie.key in QUANTITIES}
    for q in qkeys:
        ests = [_est_dict(e) for e in X.estimates_for(run, q)]
        best = X.best_estimate(run, q)
        ie = X.identifiability_for(run, q)
        quantities[q] = {
            "desc": X.quantity_desc(q),
            "unit": X.quantity_unit(q),
            "best": _est_dict(best) if best else None,
            "estimates": ests,
            "identifiability": _ie_dict(ie),
        }

    # ---- probes -------------------------------------------------------------
    probes: dict[str, Any] = {}
    for pr in run.probe_results:
        st = X.experiment_for_probe(run, pr.probe_id)
        probes[pr.probe_id] = {
            "status": pr.status,
            "n_requests": pr.n_requests,
            "error": pr.error,
            "tables": list((pr.tables or {}).keys()),
            "series": list((pr.series or {}).keys()),
            "notes": list(pr.notes or []),
            "evidence": [_ev_dict(ev) for ev in pr.evidence],
            "estimates": [_est_dict(e) for e in pr.estimates],
            "experiment": _step_dict(st) if st else None,
        }

    # ---- diagram nodes + synthetic notes ------------------------------------
    nodes: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    for diag in diags.values():
        for n in diag.nodes:
            key = n.key or (f"note:{n.id}" if n.detail else "")
            nodes[n.id] = {
                "key": key, "level": canonical_level(n.level), "conf": n.confidence,
                "label": " · ".join(n.lines), "diagram": diag.kind,
                "detail": n.detail or {},
            }
            if key.startswith("note:"):
                d = dict(n.detail or {})
                d.setdefault("title", n.lines[0] if n.lines else n.id)
                d.setdefault("level", canonical_level(n.level))
                notes[key] = d

    # ---- identifiability summary -------------------------------------------
    try:
        summ = IDF.summarize(run.identifiability)
        arch = {"question": summ.arch_question, "verdict": summ.arch_verdict,
                "by_level": {str(k): v for k, v in summ.by_level.items()}}
    except Exception:  # noqa: BLE001
        arch = {"question": "", "verdict": "", "by_level": {}}

    return {
        "meta": {
            "run_id": run.run_id, "kind": run.caps.kind, "target": run.target,
            "started": run.started_at, "finished": run.finished_at,
            "n_probes": len(run.probe_results), "n_evidence": len(run.evidence),
            "n_estimates": len(run.estimates), "budget": run.budget_summary,
        },
        "properties": properties,
        "certificates": build_certificates(run),
        "architecture_hypothesis": architecture_hypothesis_graph(run),
        "quantities": quantities,
        "probes": probes,
        "nodes": nodes,
        "notes": notes,
        "identifiability": {ie.key: _ie_dict(ie) for ie in run.identifiability},
        "family_matches": [{"name": f.name, "score": f.score, "matched": f.matched,
                            "conflicts": f.conflicts} for f in run.family_matches],
        "recommendations": [{"what": r.what, "why": r.why,
                             "gain": r.expected_gain_bits, "needs": r.needs}
                            for r in run.recommendations],
        "fingerprint": to_jsonable(run.fingerprint),
        "experiment_log": [_step_dict(s) for s in run.experiment_log],
        "run_notes": list(run.notes),
        "styleMap": _style_map(),
        "arch": arch,
    }


def _style_map() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key in ("observed", "inferred", "estimated", "speculative", "unknown"):
        st = level_style(key)
        out[key] = {"label": st.label, "color": st.color, "dark": st.dark, "line": st.line}
    return out


def _ev_dict(ev) -> dict[str, Any]:
    return {
        "property": ev.property, "probe_id": ev.probe_id, "level": ev.level,
        "strength": ev.strength, "summary": ev.summary,
        "likelihoods": {str(k): float(v) for k, v in (ev.likelihoods or {}).items()},
        "stats": to_jsonable(ev.stats or {}),
        "alternatives": list(ev.alternatives or []),
    }


def _est_dict(e) -> dict[str, Any] | None:
    if e is None:
        return None
    ci = None
    if e.ci and isinstance(e.ci, (list, tuple)) and len(e.ci) == 2:
        ci = X.fmt_ci(e.ci, e.value)
    return {
        "quantity": e.quantity, "value": to_jsonable(e.value), "value_fmt":
        X.fmt_value(e.value, X.quantity_unit(e.quantity)),
        "ci": ci, "level": e.level, "method": e.method,
        "assumptions": list(e.assumptions or []),
        "alternatives": list(e.alternatives or []),
        "identifiability": int(getattr(e, "identifiability", 5) or 5),
        "identifiability_label": IDENT_LABELS.get(
            int(getattr(e, "identifiability", 5) or 5), ""),
        "unit": e.unit, "probe_id": e.probe_id,
    }


def _ie_dict(ie) -> dict[str, Any] | None:
    if ie is None:
        return None
    return {"key": ie.key, "level": ie.level, "label": ie.label,
            "reason": ie.reason, "equivalence": ie.equivalence}


def _step_dict(s) -> dict[str, Any] | None:
    if s is None:
        return None
    return {"step": s.step, "probe_id": s.probe_id, "reason": s.reason,
            "eig_bits": s.eig_bits, "cost": s.cost,
            "entropy_before_bits": s.entropy_before_bits,
            "entropy_after_bits": s.entropy_after_bits,
            "status": s.status, "n_requests": s.n_requests}


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "__dataclass_fields__"):
        return to_jsonable(o)
    return str(o)


# --------------------------------------------------------------------------- body HTML
_SECTIONS = [
    ("A", "Architecture"), ("B", "Quantities"), ("C", "Evidence"),
    ("D", "Experiments"), ("E", "Raw data"), ("F", "Statistics"),
    ("G", "Competing"), ("H", "Confidence"), ("I", "Unknowns"),
    ("J", "Fingerprint"), ("K", "Families"), ("L", "Next steps"),
]


def _body(run: RunResult, diags: dict[str, DG.Diagram], charts: dict[str, str]) -> str:
    p: list[str] = []
    w = p.append
    name = _h.escape(str(run.target.get("name", run.run_id)))

    # header
    w('<header class="topbar">')
    w(f'<div class="brand"><span class="logo">M</span>Model Forensics '
      f'<span class="muted">/ {name}</span></div>')
    w('<nav class="tabs">')
    for sid, label in _SECTIONS:
        w(f'<a href="#sec-{sid}">{sid} · {_h.escape(label)}</a>')
    w('</nav>')
    w('<button id="theme-toggle" class="tbtn" title="Toggle light/dark">◐</button>')
    w('</header>')

    w('<main class="content">')

    # intro / meta
    w('<section class="card">')
    w(f'<h1>Forensic report — {name}</h1>')
    w('<p class="lede">Partial functional architecture reconstruction from controlled API probes. '
      'The graph retains component candidates, evidence, alternatives, and the next discriminator. '
      'Scores rank hypotheses; they are not calibrated probabilities. '
      'Click a node or row to inspect its support and unresolved differences.</p>')
    w('<div class="metagrid">')
    for k, v in (("run id", run.run_id), ("endpoint", run.caps.kind),
                 ("probes", len(run.probe_results)), ("evidence", len(run.evidence)),
                 ("estimates", len(run.estimates)),
                 ("started", run.started_at or "?")):
        w(f'<div class="metacell"><span class="mk">{_h.escape(str(k))}</span>'
          f'<span class="mv">{_h.escape(str(v))}</span></div>')
    w('</div>')
    _legend_html(w)
    w('</section>')

    # A. architecture + diagrams
    w(_sec_open("A", "Architecture hypothesis (prior → posterior)"))
    w('<p class="muted">Illustrative mechanism hypothesis, not a recovered implementation. '
      'Dashed = estimated/speculative; grey = unresolved by this scan. '
      'Legacy level numbers are capability assessments, not formal impossibility proofs.</p>')
    w(_diagram_html("block", diags["block"]))
    w('<h3>Surrogate support over hypothesis properties (legacy posterior field)</h3>')
    w('<table class="grid"><thead><tr><th>Property</th><th>Winning value</th>'
      '<th>Prior</th><th>Support score</th><th>Δ</th><th>Level</th></tr></thead><tbody>')
    for prop in run.posterior:
        val, post = X.posterior_top(run, prop)
        if val is None:
            continue
        prior = X.prior_prob(run, prop, val)
        ie = X.identifiability_for(run, prop)
        lvl = ident_to_level(ie.level) if ie else "unknown"
        w(_row(f"prop:{prop}", [
            _code(prop), _h.escape(X.value_short(val)),
            f"{prior*100:.0f}%",
            _barcell(post), _delta(post - prior), _chip(lvl)]))
    w('</tbody></table>')
    w('<h3>Serving topology</h3>')
    w('<p class="muted">Only components with evidence are drawn.</p>')
    w(_diagram_html("topology", diags["topology"]))
    if charts.get("belief_bars"):
        w(_chart_html("belief_bars", charts["belief_bars"]))
    w(_sec_close())

    # B. quantities
    w(_sec_open("B", "Estimated architecture (quantities)"))
    w('<table class="grid"><thead><tr><th>Quantity</th><th>Value</th><th>90% CI</th>'
      '<th>Level</th><th>Identifiability</th><th>Method</th><th>Probe</th></tr></thead><tbody>')
    def _resolved(q: str) -> bool:
        e = X.best_estimate(run, q)
        return e is not None and e.value is not None

    ordered = [q for q in QUANTITIES if _resolved(q)]
    extra = sorted({e.quantity for e in run.estimates
                    if e.quantity not in QUANTITIES and e.value is not None})
    for q in ordered + extra:
        e = X.best_estimate(run, q)
        if e is None:
            continue
        ident = IDENT_LABELS.get(int(getattr(e, "identifiability", 5) or 5), "")
        w(_row(f"quant:{q}", [
            _code(q), _h.escape(X.fmt_value(e.value, X.quantity_unit(q))),
            _h.escape(X.fmt_ci(e.ci, e.value) or "—"), _chip(e.level),
            _h.escape(ident), _h.escape(_short(e.method, 52)), _code(e.probe_id)]))
    w('</tbody></table>')
    # explicit not-identified quantities
    unresolved = [q for q in QUANTITIES if not _resolved(q)
                  and (X.best_estimate(run, q) is not None
                       or X.identifiability_for(run, q) is not None)]
    if unresolved:
        w('<h3>Not identified within budget</h3>')
        w('<div class="chips">')
        for q in unresolved:
            w(f'<button class="chip" data-key="{_h.escape(_key_for(run, q))}">'
              f'{_h.escape(q)}</button>')
        w('</div>')
    w(_sec_close())

    # C. evidence
    w(_sec_open("C", "Evidence for every inference"))
    w('<p class="muted">Experiment → conclusion traceability. Click a node to inspect.</p>')
    w(_diagram_html("evidence", diags["evidence"]))
    w('<h3>Likelihood evidence (per property)</h3>')
    props_seen = _order_first([ev.property for ev in run.evidence])
    for prop in props_seen:
        evs = X.evidence_for_property(run, prop)
        if not evs:
            continue
        w(f'<h4 class="clickable-h" data-key="prop:{_h.escape(prop)}">{_code(prop)} '
          f'<span class="muted">{_h.escape(X.property_desc(prop))}</span></h4>')
        w('<ul class="ev">')
        for ev in evs:
            w(f'<li data-key="probe:{_h.escape(ev.probe_id)}"><b>{_code(ev.probe_id)}</b> '
              f'{_chip(ev.level)} — {_h.escape(ev.summary)}</li>')
        w('</ul>')
    w(_sec_close())

    # D. experiments
    w(_sec_open("D", "Experiments performed"))
    w('<table class="grid"><thead><tr><th>#</th><th>Probe</th><th>Reason</th>'
      '<th>EIG bits</th><th>Entropy →</th><th>Cost</th><th>Reqs</th><th>Status</th>'
      '</tr></thead><tbody>')
    for st in run.experiment_log:
        eb = "—" if st.entropy_before_bits is None else f"{st.entropy_before_bits:.2f}"
        ea = "—" if st.entropy_after_bits is None else f"{st.entropy_after_bits:.2f}"
        eig = "—" if st.eig_bits is None else f"{st.eig_bits:.3f}"
        w(_row(f"probe:{st.probe_id}", [
            str(st.step), _code(st.probe_id), _h.escape(st.reason), eig,
            f"{eb} → {ea}", str(st.cost if st.cost is not None else "—"),
            str(st.n_requests), _chip_status(st.status)]))
    w('</tbody></table>')
    w(_sec_close())

    # E. raw
    w(_sec_open("E", "Raw observations"))
    if run.run_dir:
        w('<p>Every request/response was recorded for offline replay:</p>')
        w(f'<pre class="path">{_h.escape(run.run_dir)}/observations.jsonl</pre>')
        w('<p class="muted">Re-run offline with <code>config["replay_dir"]</code> pointing '
          'here — no network, no cost.</p>')
    else:
        w('<p class="muted">Recording was not enabled for this run, so there is no '
          '<code>observations.jsonl</code>. Pass a <code>Recorder(run_dir=…)</code> to '
          '<code>run_forensics</code> to capture raw observations for replay.</p>')
    w(_sec_close())

    # F. statistics
    w(_sec_open("F", "Statistical analysis"))
    w('<table class="grid"><thead><tr><th>Property</th><th>Probe</th><th>Test / obs</th>'
      '<th>p</th><th>Effect / stat</th><th>n</th></tr></thead><tbody>')
    any_stat = False
    for ev in run.evidence:
        stt = ev.stats or {}
        test = stt.get("test") or _guess_test(stt)
        pv = stt.get("p_value", stt.get("p"))
        eff = stt.get("effect", stt.get("effect_size"))
        n = stt.get("n", stt.get("samples"))
        if not (test or pv is not None or eff is not None):
            continue
        any_stat = True
        w(_row(f"probe:{ev.probe_id}", [
            _code(ev.property), _code(ev.probe_id),
            _h.escape(_short(str(test or "—"), 40)), _fmt_p(pv),
            _h.escape(_short(str(eff) if eff is not None else "—", 24)),
            str(n if n is not None else "—")]))
    if not any_stat:
        w('<tr><td colspan="6" class="muted">Evidence carried descriptive statistics; '
          'see section C for per-probe detail.</td></tr>')
    w('</tbody></table>')
    for cname in ("singular_spectrum", "needle_heatmap", "latency_vs_length",
                  "calibration"):
        if charts.get(cname):
            w(_chart_html(cname, charts[cname]))
    w(_sec_close())

    # G. competing
    w(_sec_open("G", "Competing hypotheses"))
    w('<p class="muted">A posterior far from 100% means the interface did not decide '
      'between these.</p>')
    w('<table class="grid"><thead><tr><th>Property</th><th>Top candidates</th>'
      '</tr></thead><tbody>')
    for prop in run.posterior:
        s = X.posterior_sorted(run, prop)[:3]
        if not s:
            continue
        cands = " ".join(
            f'<span class="pill">{_h.escape(X.value_short(v))} {pp*100:.0f}%</span>'
            for v, pp in s)
        w(_row(f"prop:{prop}", [_code(prop), cands]))
    w('</tbody></table>')
    w(_sec_close())

    # H. confidence
    w(_sec_open("H", "Confidence levels"))
    try:
        summ = IDF.summarize(run.identifiability)
        w('<table class="grid"><thead><tr><th>Level</th><th>Meaning</th><th>Count</th>'
          '</tr></thead><tbody>')
        for lvl in range(1, 6):
            w(f'<tr><td>{lvl}</td><td>{_h.escape(IDENT_LABELS[lvl])}</td>'
              f'<td>{summ.by_level.get(lvl, 0)}</td></tr>')
        w('</tbody></table>')
    except Exception:  # noqa: BLE001
        pass
    w('<h3>Headline quantities</h3><ul class="ev">')
    for q in ("dims.hidden_size", "dims.vocab_size", "dims.n_layers", "arch.family",
              "arch.sparsity", "context.max_input_tokens"):
        e = X.best_estimate(run, q)
        if e is not None and e.value is not None:
            cw = confidence_word(ident_confidence(getattr(e, "identifiability", None)))
            w(_row2(_key_for(run, q),
                    f'{_code(q)}: <b>{_h.escape(X.fmt_value(e.value))}</b> {_chip(e.level)} '
                    f'<span class="muted">{cw} confidence</span>'))
        else:
            ie = X.identifiability_for(run, q)
            lbl = ie.label if ie else "not identified"
            w(_row2(_key_for(run, q),
                    f'{_code(q)}: <span class="muted">{X.NOT_IDENTIFIED}</span> '
                    f'{_chip("unknown")} {_h.escape(lbl)}'))
    w('</ul>')
    w(_sec_close())

    # I. unknowns
    w(_sec_open("I", "Unresolved properties and legacy capability assessments"))
    w('<p><a href="architecture_hypothesis.json">Architecture hypothesis graph (JSON)</a> · '
      '<a href="forensic_certificates.json">Evidence certificates</a> · '
      '<a href="run_manifest.json">Hashed run manifest</a></p>')
    w('<p class="muted">Missing capabilities and finite-budget estimator limits are unresolved (U), '
      'not impossibility proofs. Legacy equivalence notes below are candidate arguments, not certified '
      'witnesses. See forensic_certificates.json for scoped findings, integrity sources, and falsifiers.</p>')
    try:
        summ = IDF.summarize(run.identifiability)
        w(f'<div class="verdict"><div class="vq">{_h.escape(summ.arch_question)}</div>'
          f'<div class="va">{_h.escape(summ.arch_verdict)}</div></div>')
    except Exception:  # noqa: BLE001
        pass
    by_level: dict[int, list] = {}
    for ie in run.identifiability:
        by_level.setdefault(ie.level, []).append(ie)
    for lvl in (5, 4, 3):
        group = by_level.get(lvl, [])
        if not group:
            continue
        w(f'<h3>Level {lvl} — {_h.escape(IDENT_LABELS[lvl])} '
          f'<span class="muted">({len(group)})</span></h3>')
        w('<ul class="ev">')
        for ie in sorted(group, key=lambda x: x.key):
            eq = (f'<div class="eq"><i>equivalence:</i> {_h.escape(ie.equivalence)}</div>'
                  if ie.equivalence else "")
            w(f'<li data-key="{_h.escape(_key_for(run, ie.key))}">{_code(ie.key)} '
              f'{_chip(ident_to_level(lvl))} {_h.escape(ie.reason)}{eq}</li>')
        w('</ul>')
    w(_sec_close())

    # J. fingerprint
    w(_sec_open("J", "Behavioural fingerprint"))
    if run.fingerprint:
        w('<table class="grid"><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>')
        for k, v in run.fingerprint.items():
            w(f'<tr><td>{_code(k)}</td><td>{_h.escape(_short(str(v), 120))}</td></tr>')
        w('</tbody></table>')
    else:
        w('<p class="muted">No behavioural fingerprint was recorded on this run '
          '(<code>run.fingerprint</code> is empty).</p>')
    w(_sec_close())

    # K. families
    w(_sec_open("K", "Possible model-family matches"))
    if run.family_matches:
        w('<table class="grid"><thead><tr><th>#</th><th>Family / model</th>'
          '<th>Consistency</th><th>Matched</th><th>Conflicts</th></tr></thead><tbody>')
        for i, fm in enumerate(run.family_matches, 1):
            w(f'<tr><td>{i}</td><td>{_h.escape(fm.name)}</td>'
              f'<td>{_barcell(fm.score)}</td>'
              f'<td>{_h.escape(", ".join(fm.matched) or "—")}</td>'
              f'<td>{_h.escape(", ".join(fm.conflicts) or "—")}</td></tr>')
        w('</tbody></table>')
        w('<p class="muted">A high score is necessary, not sufficient: families sharing a '
          'tokenizer/context/KV-scheme are behaviourally close even with different weights.</p>')
    else:
        w('<p class="muted">No family matches were produced (insufficient measured '
          'constraints).</p>')
    w(_sec_close())

    # L. next steps
    w(_sec_open("L", "Recommended next experiments"))
    if run.recommendations:
        w('<table class="grid"><thead><tr><th>Experiment</th><th>Why</th>'
          '<th>Gain bits</th><th>Needs</th></tr></thead><tbody>')
        for r in run.recommendations:
            g = "—" if r.expected_gain_bits is None else f"{r.expected_gain_bits:.3f}"
            w(f'<tr><td>{_code(r.what)}</td><td>{_h.escape(r.why)}</td>'
              f'<td>{g}</td><td>{_h.escape(r.needs or "—")}</td></tr>')
        w('</tbody></table>')
    else:
        w('<p class="muted">No further experiments were recommended: the adaptive loop '
          'reached its surrogate-gain or capability limit. Remaining properties are unresolved; '
          'a new estimator, larger budget, or additional channel may resolve them (section I).</p>')
    w(_sec_close())

    w('</main>')

    # inspector panel
    w('<aside id="inspector" class="inspector" aria-hidden="true">')
    w('<div class="insp-head"><span id="insp-title">Inspector</span>'
      '<button id="insp-close" class="tbtn">✕</button></div>')
    w('<div id="insp-body" class="insp-body">'
      '<p class="muted">Click any diagram node or table row to see how it was inferred.</p>'
      '</div>')
    w('</aside>')
    w('<div id="insp-backdrop" class="backdrop" hidden></div>')
    return "\n".join(p)


# --------------------------------------------------------------------------- html bits
def _diagram_html(kind: str, d: DG.Diagram) -> str:
    p: list[str] = []
    p.append(f'<div class="diagram" data-kind="{kind}">')
    if d.svg:
        p.append(f'<div class="svgwrap">{d.svg}</div>')
    else:
        p.append(f'<pre class="mermaid">{_h.escape(d.mermaid)}</pre>')
    # always-present clickable node chips (revealed if the diagram fails to render)
    p.append('<div class="chips diag-chips" hidden>')
    for n in d.nodes:
        key = n.key or (f"note:{n.id}" if n.detail else "")
        if not key:
            continue
        label = _h.escape(n.lines[0] if n.lines else n.id)
        p.append(f'<button class="chip" data-node="{_h.escape(n.id)}">{label}</button>')
    p.append('</div>')
    p.append(f'<details class="src"><summary>diagram source (Mermaid)</summary>'
             f'<pre>{_h.escape(d.mermaid)}</pre></details>')
    p.append('</div>')
    return "\n".join(p)


def _chart_html(name: str, uri: str) -> str:
    return (f'<figure class="chart"><img alt="{_h.escape(CH.chart_title(name))}" '
            f'src="{uri}"><figcaption>{_h.escape(CH.chart_title(name))}</figcaption></figure>')


def _legend_html(w) -> None:
    w('<div class="legend"><span class="lgt">Legend:</span>')
    for r in legend_rows():
        w(f'<span class="lg" title="{_h.escape(r["meaning"])}">'
          f'<span class="sw sw-{r["level"].split(" ")[0]}"></span>{_h.escape(r["level"])} '
          f'<span class="muted">({r["line"]})</span></span>')
    w('</div>')


def _sec_open(sid: str, title: str) -> str:
    return (f'<section class="card" id="sec-{sid}"><h2><span class="secid">{sid}</span> '
            f'{_h.escape(title)}</h2>')


def _sec_close() -> str:
    return '</section>'


def _row(key: str, cells: list[str]) -> str:
    tds = "".join(f"<td>{c}</td>" for c in cells)
    return f'<tr class="clickable" data-key="{_h.escape(key)}">{tds}</tr>'


def _row2(key: str, inner: str) -> str:
    return f'<li class="clickable" data-key="{_h.escape(key)}">{inner}</li>'


def _code(s: Any) -> str:
    return f'<code>{_h.escape(str(s))}</code>'


def _chip(level: str) -> str:
    lv = canonical_level(level)
    st = level_style(lv)
    return (f'<span class="chip-lvl" style="--lvl:{st.color};--lvld:{st.dark}">'
            f'{_h.escape(st.key)}</span>')


def _chip_status(status: str) -> str:
    cls = {"ok": "st-ok", "partial": "st-warn", "failed": "st-bad",
           "skipped": "st-muted"}.get(status, "st-muted")
    return f'<span class="stbadge {cls}">{_h.escape(status)}</span>'


def _barcell(p: float) -> str:
    pct = max(0.0, min(1.0, float(p))) * 100
    return (f'<div class="bar"><span style="width:{pct:.0f}%"></span>'
            f'<em>{pct:.0f}%</em></div>')


def _delta(d: float) -> str:
    if d > 0.005:
        return f'<span class="up">▲ {d*100:+.0f}%</span>'
    if d < -0.005:
        return f'<span class="down">▼ {d*100:+.0f}%</span>'
    return '<span class="muted">•</span>'


def _fmt_p(p: Any) -> str:
    if p is None:
        return "—"
    try:
        pf = float(p)
    except (TypeError, ValueError):
        return _h.escape(str(p))
    return f"{pf:.1e}" if pf < 1e-4 else f"{pf:.4f}"


def _guess_test(stats: dict[str, Any]) -> str:
    for k in ("test", "method", "statistic"):
        if k in stats:
            return str(stats[k])
    return ""


def _short(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _order_first(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        seen.setdefault(it, None)
    return list(seen.keys())


# --------------------------------------------------------------------------- templates
_CSS = r"""
:root{
  --bg:#faf9f6; --surface:#ffffff; --surface2:#f4f2ed; --ink:#14120e; --ink2:#55524a;
  --muted:#8c877c; --border:#e8e5dd; --border2:#ddd9cf; --accent:#f04b0d;
  --shadow:0 1px 2px rgba(20,18,14,.04),0 8px 28px rgba(20,18,14,.05);
  --observed:#008300; --inferred:#2a78d6; --estimated:#c98a00; --speculative:#e34948;
  --unknown:#9a968c;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,monospace;
}
:root[data-theme="dark"]{
  --bg:#121110; --surface:#1b1916; --surface2:#23201b; --ink:#f0ede6; --ink2:#b8b3a7;
  --muted:#7e7a6f; --border:#2c2924; --border2:#3a362e; --accent:#ff5b1a;
  --shadow:0 1px 2px rgba(0,0,0,.35);
  --observed:#2db55a; --inferred:#5aa0ff; --estimated:#e0a84a; --speculative:#ff6b63;
  --unknown:#8c877c;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:72px}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 "Inter",system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
code{font:12.5px/1.4 var(--mono);background:var(--surface2);
  padding:1px 5px;border-radius:5px;border:1px solid var(--border);color:var(--ink2)}
.muted{color:var(--muted)}
.topbar{position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:14px;
  padding:8px 18px;background:color-mix(in srgb,var(--bg) 85%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
.brand{font-weight:650;letter-spacing:-.01em;white-space:nowrap;display:flex;align-items:center;gap:9px}
.brand .logo{width:26px;height:26px;border-radius:6px;background:var(--accent);display:inline-grid;
  place-items:center;color:#fff;font-weight:800;font-size:12px}
.tabs{display:flex;gap:2px;overflow-x:auto;flex:1;scrollbar-width:thin}
.tabs a{padding:6px 11px;border-radius:999px;font:500 11px var(--mono);letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink2);white-space:nowrap}
.tabs a:hover{background:var(--surface2);color:var(--ink)}
.tbtn{background:var(--surface2);border:1px solid var(--border);color:var(--ink);
  border-radius:999px;width:34px;height:32px;cursor:pointer;font-size:15px}
.tbtn:hover{border-color:var(--accent)}
.content{max-width:1060px;margin:0 auto;padding:22px 18px 120px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:22px 24px;margin:18px 0;box-shadow:var(--shadow)}
h1{font-size:30px;margin:.1em 0 .3em;letter-spacing:-.03em;font-weight:680}
h2{font-size:20px;margin:.2em 0 .7em;display:flex;align-items:center;gap:11px;
  letter-spacing:-.02em;font-weight:680}
h3{font-size:15.5px;margin:1.3em 0 .5em;color:var(--ink);font-weight:650}
h4{font-size:14px;margin:1em 0 .3em}
.secid{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:7px;
  background:var(--accent);color:#fff;font:700 13px var(--mono)}
.lede{color:var(--ink2);max-width:75ch}
.metagrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:0;margin:16px 0;
  border:1px solid var(--border);border-radius:10px;overflow:hidden}
.metacell{background:var(--surface2);padding:10px 12px;display:flex;flex-direction:column;
  border-left:1px solid var(--border)}
.metacell:first-child{border-left:0}
.mk{font:500 10px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.12em}
.mv{font-size:14px;font-weight:650;margin-top:2px;word-break:break-word}
.legend{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-top:14px;
  padding-top:14px;border-top:1px solid var(--border);font-size:12.5px}
.lgt{font:500 10.5px var(--mono);text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
.lg{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:13px;border-radius:4px;display:inline-block;border:2px solid}
.sw-observed{border-color:var(--observed);background:color-mix(in srgb,var(--observed) 18%,transparent)}
.sw-inferred{border-color:var(--inferred);background:color-mix(in srgb,var(--inferred) 18%,transparent)}
.sw-estimated{border-color:var(--estimated);background:color-mix(in srgb,var(--estimated) 18%,transparent);border-style:dashed}
.sw-speculative{border-color:var(--speculative);background:color-mix(in srgb,var(--speculative) 18%,transparent);border-style:dotted}
.sw-unknown{border-color:var(--unknown);background:color-mix(in srgb,var(--unknown) 18%,transparent);border-style:dashed}
table.grid{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0}
.grid th,.grid td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--border);
  vertical-align:top}
.grid th{font:500 10.5px var(--mono);text-transform:uppercase;letter-spacing:.12em;
  color:var(--muted);border-bottom:1px solid var(--border2);position:sticky}
.grid tbody tr.clickable{cursor:pointer}
.grid tbody tr.clickable:hover{background:var(--surface2)}
.grid tbody tr.clickable:hover td:first-child{box-shadow:inset 3px 0 var(--accent)}
.bar{position:relative;background:var(--surface2);border-radius:999px;height:16px;min-width:70px;
  border:1px solid var(--border);overflow:hidden}
.bar>span{position:absolute;left:0;top:0;bottom:0;background:var(--accent);opacity:.9;border-radius:999px}
.bar>em{position:relative;font-style:normal;font-size:11px;padding-left:7px;line-height:16px;
  font-family:var(--mono);color:var(--ink)}
.chip-lvl{display:inline-block;font:600 10px var(--mono);letter-spacing:.06em;text-transform:uppercase;
  padding:2px 8px;border-radius:999px;border:1.5px solid var(--lvl);color:var(--lvl);white-space:nowrap}
:root[data-theme="dark"] .chip-lvl{border-color:var(--lvld);color:var(--lvld)}
.up{color:var(--observed);font-weight:650}.down{color:var(--speculative);font-weight:650}
.pill{display:inline-block;background:var(--surface2);border:1px solid var(--border);
  border-radius:999px;padding:2px 10px;font-size:12px;margin:1px}
.stbadge{font:600 10px var(--mono);letter-spacing:.06em;text-transform:uppercase;
  padding:2px 8px;border-radius:999px}
.st-ok{background:color-mix(in srgb,var(--observed) 16%,transparent);color:var(--observed)}
.st-warn{background:color-mix(in srgb,var(--estimated) 20%,transparent);color:var(--estimated)}
.st-bad{background:color-mix(in srgb,var(--speculative) 18%,transparent);color:var(--speculative)}
.st-muted{background:var(--surface2);color:var(--muted)}
ul.ev{margin:.3em 0;padding-left:18px}
ul.ev li{margin:3px 0}
ul.ev li.clickable,ul.ev li[data-key]{cursor:pointer;list-style:none;padding:7px 10px;
  border:1px solid var(--border);border-radius:9px;margin:5px 0;background:var(--surface2)}
ul.ev li[data-key]:hover{border-color:var(--accent)}
.clickable-h{cursor:pointer}
.clickable-h:hover code{border-color:var(--accent)}
.eq{font-size:12px;color:var(--muted);margin-top:3px}
.diagram{background:var(--surface2);border:1px solid var(--border);border-radius:12px;
  padding:16px;margin:10px 0;overflow-x:auto}
.diagram .mermaid{text-align:center;margin:0}
.svgwrap svg,.mermaid svg{max-width:100%;height:auto}
.mermaid svg g.node,.svgwrap svg g.node{cursor:pointer}
.mermaid svg g.node:hover,.svgwrap svg g.node:hover{filter:brightness(1.05) drop-shadow(0 0 3px var(--accent))}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.chip{cursor:pointer;font:500 12px var(--mono);padding:4px 11px;border-radius:999px;
  border:1px solid var(--border2);background:var(--surface);color:var(--ink2)}
.chip:hover{border-color:var(--accent);color:var(--ink)}
.src{margin-top:10px;font-size:12px}
.src summary{cursor:pointer;color:var(--muted);font:500 11px var(--mono);text-transform:uppercase;
  letter-spacing:.1em}
.src pre,pre.path{background:var(--surface);border:1px solid var(--border);border-radius:9px;
  padding:12px;overflow-x:auto;font:12px/1.6 var(--mono)}
.chart{margin:16px 0;text-align:center}
.chart img{max-width:100%;border:1px solid var(--border);border-radius:12px;background:var(--surface);
  padding:10px}
.chart figcaption{font:500 11px var(--mono);color:var(--muted);margin-top:7px;text-transform:uppercase;
  letter-spacing:.1em}
.verdict{border-left:4px solid var(--speculative);background:var(--surface2);padding:12px 16px;
  border-radius:0 10px 10px 0;margin:10px 0}
.vq{font-weight:650;margin-bottom:5px}
.va{color:var(--ink2);font-size:13.5px}
.mmerr{color:var(--speculative);font-size:12px;margin:6px 0}
/* inspector */
.inspector{position:fixed;top:0;right:0;height:100vh;width:min(470px,92vw);
  background:var(--surface);border-left:1px solid var(--border);box-shadow:-8px 0 32px rgba(20,18,14,.16);
  transform:translateX(100%);transition:transform .22s ease;z-index:100;display:flex;flex-direction:column}
.inspector.open{transform:none}
.insp-head{display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:13px 18px;border-bottom:1px solid var(--border);font-weight:650}
#insp-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.insp-body{padding:16px 18px;overflow-y:auto;flex:1}
.backdrop{position:fixed;inset:0;background:rgba(20,18,14,.3);z-index:90}
.insp-sec{margin:0 0 18px}
.insp-sec h5{margin:0 0 7px;font:500 10.5px var(--mono);text-transform:uppercase;
  letter-spacing:.12em;color:var(--muted)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px}
.kv dt{color:var(--muted)}.kv dd{margin:0;word-break:break-word}
.ins-bar{position:relative;height:14px;background:var(--surface2);border-radius:999px;margin:2px 0;
  border:1px solid var(--border);overflow:hidden}
.ins-bar>i{position:absolute;left:0;top:0;bottom:0;border-radius:999px}
.ins-bar .prior{background:var(--muted);opacity:.45}
.ins-row{display:grid;grid-template-columns:1fr 46px;gap:8px;align-items:center;font-size:12px;margin:4px 0}
.ins-row .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--mono);font-size:11.5px}
.evbox{border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin:8px 0;
  background:var(--surface2)}
.evbox .top{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:5px}
.tag{font:500 10.5px var(--mono);padding:2px 8px;border-radius:999px;background:var(--surface);
  border:1px solid var(--border2)}
.lk{font-size:11.5px;color:var(--ink2);margin:3px 0}
.alts{font-size:12px;color:var(--muted);margin-top:4px}
.relbtns{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.relbtn{cursor:pointer;font:500 11.5px var(--mono);padding:3px 10px;border-radius:999px;
  border:1px solid var(--border2);background:var(--surface);color:var(--accent)}
.relbtn:hover{border-color:var(--accent)}
@media(max-width:640px){.content{padding:14px 10px 120px}.card{padding:16px}}
"""

_PAGE = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Forensics — {title}</title>
<style>{css}</style>
</head>
<body>
{body}
<script>window.__FORENSICS__ = {blob};</script>
<script>{js}</script>
<script type="module">
(async () => {{
  try {{
    const m = await import("https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs");
    const mermaid = m.default;
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    mermaid.initialize({{startOnLoad:false, securityLevel:"loose",
      theme: dark ? "dark" : "neutral", flowchart:{{htmlLabels:true, useMaxWidth:true}}}});
    window.__mermaid = mermaid;
    await window.renderMermaid();
  }} catch (e) {{
    window.onMermaidFail && window.onMermaidFail(e);
  }}
}})();
</script>
</body>
</html>
"""

_JS = r"""
(function(){
  "use strict";
  var F = window.__FORENSICS__ || {};
  var esc = function(s){ return String(s==null?"":s)
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
      .replace(/"/g,"&quot;"); };
  var STYLE = F.styleMap || {};
  function levelColor(lv){ var s = STYLE[lv]||STYLE.unknown||{};
    var dark = document.documentElement.getAttribute("data-theme")==="dark";
    return (dark ? (s.dark||s.color) : s.color) || "#888"; }
  function chip(lv){ lv = lv||"unknown"; var c = levelColor(lv);
    return '<span class="chip-lvl" style="--lvl:'+c+';--lvld:'+c+'">'+esc(lv)+'</span>'; }
  function pct(x){ return (Math.max(0,Math.min(1,+x))*100).toFixed(0)+"%"; }

  /* ---------------- theme ---------------- */
  function applyTheme(t){ document.documentElement.setAttribute("data-theme", t);
    try{ localStorage.setItem("forensics-theme", t); }catch(e){}
  }
  (function(){ var t="light"; try{ t=localStorage.getItem("forensics-theme")||
     (window.matchMedia && matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");}catch(e){}
     applyTheme(t); })();
  document.addEventListener("click", function(e){
    var tg = e.target.closest && e.target.closest("#theme-toggle");
    if(tg){ var cur=document.documentElement.getAttribute("data-theme");
      applyTheme(cur==="dark"?"light":"dark"); }
  });

  /* ---------------- inspector ---------------- */
  var panel = document.getElementById("inspector");
  var body = document.getElementById("insp-body");
  var titleEl = document.getElementById("insp-title");
  var backdrop = document.getElementById("insp-backdrop");
  function openPanel(){ panel.classList.add("open"); panel.setAttribute("aria-hidden","false");
    if(window.innerWidth<820){ backdrop.hidden=false; } }
  function closePanel(){ panel.classList.remove("open"); panel.setAttribute("aria-hidden","true");
    backdrop.hidden=true; }
  document.getElementById("insp-close").addEventListener("click", closePanel);
  backdrop.addEventListener("click", closePanel);
  document.addEventListener("keydown", function(e){ if(e.key==="Escape") closePanel(); });

  function sec(title, inner){ return '<div class="insp-sec"><h5>'+esc(title)+'</h5>'+inner+'</div>'; }
  function relbtns(keys){ if(!keys||!keys.length) return "";
    return '<div class="relbtns">'+keys.map(function(k){
      return '<button class="relbtn" data-key="'+esc(k)+'">'+esc(prettyKey(k))+'</button>';
    }).join("")+'</div>'; }
  function prettyKey(k){ var i=k.indexOf(":"); return i<0?k:k.slice(i+1); }

  function priorPostBars(values){
    if(!values||!values.length) return "";
    return values.slice(0,7).map(function(v){
      return '<div class="ins-row"><div class="nm" title="'+esc(v.value)+'">'+esc(v.value)+'</div>'
        +'<div>'+pct(v.post)+'</div></div>'
        +'<div class="ins-bar"><i class="prior" style="width:'+pct(v.prior)+'"></i>'
        +'<i style="width:'+pct(v.post)+';background:'+levelColor("inferred")+';opacity:.8"></i></div>';
    }).join("");
  }
  function evBox(ev){
    var lk = Object.keys(ev.likelihoods||{}).sort(function(a,b){
        return ev.likelihoods[b]-ev.likelihoods[a];}).slice(0,6)
        .map(function(k){ return esc(k)+"="+(+ev.likelihoods[k]).toPrecision(2); }).join(", ");
    var stats = ev.stats||{}; var sk = Object.keys(stats).slice(0,8).map(function(k){
        var v=stats[k]; if(typeof v==="number") v=(+v).toPrecision(4);
        else if(v&&typeof v==="object") v="["+(v.length!=null?v.length:"…")+"]";
        return esc(k)+"="+esc(v); }).join(", ");
    var alts = (ev.alternatives||[]).map(esc).join("; ");
    return '<div class="evbox"><div class="top"><span class="tag">'+esc(ev.probe_id)+'</span>'
      +chip(ev.level)+'<span class="muted">strength '+esc(ev.strength)+'</span></div>'
      +'<div>'+esc(ev.summary)+'</div>'
      +(lk?'<div class="lk"><b>likelihoods</b> P(obs|value): '+lk+'</div>':'')
      +(sk?'<div class="lk"><b>stats</b>: '+sk+'</div>':'')
      +(alts?'<div class="alts"><b>alternatives:</b> '+alts+'</div>':'')+'</div>';
  }
  function identBlock(ie){
    if(!ie) return "";
    var inner='<div class="kv"><dt>level</dt><dd>'+esc(ie.level)+' — '+esc(ie.label)+'</dd>'
      +'<dt>why</dt><dd>'+esc(ie.reason)+'</dd>'
      +(ie.equivalence?'<dt>equivalence</dt><dd>'+esc(ie.equivalence)+'</dd>':'')+'</div>';
    return sec("Identifiability", inner);
  }

  function renderProp(key){
    var name = prettyKey(key); var d = (F.properties||{})[name];
    if(!d) return "<p class='muted'>No data for "+esc(name)+".</p>";
    var top = d.top||{};
    var h = sec("Property", '<div class="kv"><dt>key</dt><dd><code>'+esc(name)+'</code></dd>'
      +'<dt>meaning</dt><dd>'+esc(d.desc)+'</dd>'
      +'<dt>winner</dt><dd>'+esc(top.value)+' '+chip(d.level)+' '+pct(top.post||0)+'</dd></div>');
    h += sec("Prior → posterior", priorPostBars(d.values));
    if(d.evidence&&d.evidence.length){
      h += sec("Evidence ("+d.evidence.length+")", d.evidence.map(evBox).join(""));
    } else { h += sec("Evidence","<p class='muted'>No direct evidence; value follows the prior / coupling.</p>"); }
    h += identBlock(d.identifiability);
    return h;
  }
  function renderQuant(key){
    var name = prettyKey(key); var d = (F.quantities||{})[name];
    if(!d) return "<p class='muted'>No data for "+esc(name)+".</p>";
    var best = d.best;
    var head = best ? (esc(best.value_fmt)+' '+chip(best.level)) :
      '<span class="muted">not identified within budget</span> '+chip("unknown");
    var h = sec("Quantity", '<div class="kv"><dt>key</dt><dd><code>'+esc(name)+'</code></dd>'
      +'<dt>meaning</dt><dd>'+esc(d.desc)+(d.unit?' ('+esc(d.unit)+')':'')+'</dd>'
      +'<dt>value</dt><dd>'+head+'</dd>'
      +(best&&best.ci?'<dt>90% CI</dt><dd>'+esc(best.ci)+'</dd>':'')+'</div>');
    var withVal = (d.estimates||[]).filter(function(e){return e && e.value!=null;});
    var list = withVal.length?withVal:(d.estimates||[]);
    if(list.length){
      h += sec("How it was estimated", list.map(function(e){
        return '<div class="evbox"><div class="top"><span class="tag">'+esc(e.probe_id)+'</span>'
          +chip(e.level)+'<span class="muted">'+esc(e.identifiability_label)+'</span></div>'
          +'<div class="kv"><dt>value</dt><dd>'+esc(e.value_fmt)+(e.ci?' (CI '+esc(e.ci)+')':'')+'</dd>'
          +(e.method?'<dt>method</dt><dd>'+esc(e.method)+'</dd>':'')
          +(e.assumptions&&e.assumptions.length?'<dt>assumptions</dt><dd>'+e.assumptions.map(esc).join('; ')+'</dd>':'')
          +(e.alternatives&&e.alternatives.length?'<dt>alternatives</dt><dd>'+e.alternatives.map(esc).join('; ')+'</dd>':'')
          +'</div></div>';
      }).join(""));
    }
    h += identBlock(d.identifiability);
    return h;
  }
  function renderProbe(key){
    var name = prettyKey(key); var d = (F.probes||{})[name];
    if(!d) return "<p class='muted'>Probe "+esc(name)+" did not run.</p>";
    var ex = d.experiment||{};
    var h = sec("Probe", '<div class="kv"><dt>id</dt><dd><code>'+esc(name)+'</code></dd>'
      +'<dt>status</dt><dd>'+esc(d.status)+'</dd>'
      +'<dt>requests</dt><dd>'+esc(d.n_requests)+'</dd>'
      +(d.error?'<dt>error</dt><dd>'+esc(d.error)+'</dd>':'')+'</div>');
    if(ex&&ex.step!=null){
      h += sec("Selection", '<div class="kv">'
        +'<dt>reason</dt><dd>'+esc(ex.reason)+'</dd>'
        +'<dt>EIG</dt><dd>'+esc(ex.eig_bits)+' bits</dd>'
        +'<dt>cost</dt><dd>'+esc(ex.cost)+'</dd>'
        +'<dt>entropy</dt><dd>'+esc(ex.entropy_before_bits)+' → '+esc(ex.entropy_after_bits)+' bits</dd>'
        +'</div>');
    }
    if(d.estimates&&d.estimates.length){
      h += sec("Quantities produced", d.estimates.filter(function(e){return e;}).map(function(e){
        return '<button class="relbtn" data-key="quant:'+esc(e.quantity)+'">'+esc(e.quantity)
          +' = '+esc(e.value_fmt)+'</button>'; }).join(" "));
    }
    if(d.evidence&&d.evidence.length){
      h += sec("Evidence produced", d.evidence.map(evBox).join(""));
    }
    var meta=[]; if(d.tables&&d.tables.length) meta.push("tables: "+d.tables.map(esc).join(", "));
    if(d.series&&d.series.length) meta.push("series: "+d.series.map(esc).join(", "));
    if(d.notes&&d.notes.length) meta.push(d.notes.map(esc).join(" · "));
    if(meta.length) h += sec("Outputs", '<div class="lk">'+meta.join("<br>")+'</div>');
    return h;
  }
  function renderNote(key){
    var d = (F.notes||{})[key];
    if(!d) return "<p class='muted'>No detail.</p>";
    return sec(d.title||"Detail", '<div>'+esc(d.body||"")+'</div>'
      +(d.identifiability?'<div class="lk" style="margin-top:6px"><b>identifiability:</b> '
        +esc(d.identifiability)+'</div>':''));
  }
  function renderKey(key){
    if(!key) return "<p class='muted'>Nothing to inspect.</p>";
    if(key.indexOf("prop:")===0) return renderProp(key);
    if(key.indexOf("quant:")===0) return renderQuant(key);
    if(key.indexOf("probe:")===0) return renderProbe(key);
    if(key.indexOf("note:")===0) return renderNote(key);
    return "<p class='muted'>"+esc(key)+"</p>";
  }

  function inspectKey(key, headerLabel){
    titleEl.textContent = headerLabel || prettyKey(key||"") || "Inspector";
    body.innerHTML = renderKey(key);
    openPanel();
  }
  function inspectNode(nodeId){
    var info = (F.nodes||{})[nodeId];
    if(!info){ inspectKey("", nodeId); return; }
    var head = '<div class="insp-sec"><div>'+esc(info.label)+'</div>'
      +'<div style="margin-top:6px">'+chip(info.level)+'</div></div>';
    var related = (info.detail&&info.detail.related)||[];
    body.innerHTML = head + renderKey(info.key) + (related.length?sec("Related", relbtns(related)):"");
    titleEl.textContent = (info.label||nodeId).split(" · ")[0];
    openPanel();
  }
  window.inspectKey = inspectKey;
  window.inspectNode = inspectNode;

  /* delegated clicks: rows/items with data-key, chips with data-node, relbtns */
  document.addEventListener("click", function(e){
    var rel = e.target.closest && e.target.closest(".relbtn[data-key]");
    if(rel){ inspectKey(rel.getAttribute("data-key")); e.stopPropagation(); return; }
    var chipN = e.target.closest && e.target.closest(".chip[data-node]");
    if(chipN){ inspectNode(chipN.getAttribute("data-node")); return; }
    var el = e.target.closest && e.target.closest("[data-key]");
    if(el && !el.classList.contains("relbtn")){ inspectKey(el.getAttribute("data-key")); }
  });

  /* ---------------- mermaid render + node binding ---------------- */
  function bindSvgNodes(){
    document.querySelectorAll(".diagram").forEach(function(dg){
      var svg = dg.querySelector("svg");
      var chips = dg.querySelector(".diag-chips");
      if(!svg){ if(chips) chips.hidden=false; return; }
      if(chips) chips.hidden=true;
      svg.querySelectorAll("g.node").forEach(function(g){
        var id=g.id||""; // e.g. flowchart-b_attn-3
        var m=id.match(/^flowchart-(.+)-\d+$/);
        var nodeId = m?m[1]:null;
        if(!nodeId && id.indexOf("flowchart-")===0){ nodeId=id.slice(10).replace(/-\d+$/,""); }
        if(nodeId && (F.nodes||{})[nodeId]){
          g.style.cursor="pointer";
          g.addEventListener("click", function(ev){ ev.stopPropagation(); inspectNode(nodeId); });
        }
      });
      /* hand-rendered figures: groups carry data-node directly */
      svg.querySelectorAll("[data-node]").forEach(function(g){
        var nid = g.getAttribute("data-node");
        if(nid && (F.nodes||{})[nid]){
          g.style.cursor="pointer";
          g.addEventListener("click", function(ev){ ev.stopPropagation(); inspectNode(nid); });
        }
      });
    });
  }
  window.renderMermaid = async function(){
    var nodes=document.querySelectorAll(".mermaid");
    if(window.__mermaid && nodes.length){
      try{ await window.__mermaid.run({nodes:nodes}); }catch(e){ window.onMermaidFail(e); return; }
    }
    bindSvgNodes();
  };
  window.onMermaidFail = function(err){
    document.querySelectorAll(".diagram").forEach(function(dg){
      var chips=dg.querySelector(".diag-chips"); if(chips) chips.hidden=false;
      var pre=dg.querySelector("pre.mermaid");
      if(pre && !dg.querySelector(".mmerr")){
        var n=document.createElement("div"); n.className="mmerr";
        n.textContent="Diagram viewer offline — click a node below to inspect, or expand the source.";
        dg.insertBefore(n, pre);
      }
    });
  };
  /* if no module loads at all within 4s, reveal chips as a fallback */
  setTimeout(function(){
    document.querySelectorAll(".diagram").forEach(function(dg){
      if(!dg.querySelector("svg")){ var c=dg.querySelector(".diag-chips"); if(c) c.hidden=false; }
    });
  }, 4000);
})();
"""
