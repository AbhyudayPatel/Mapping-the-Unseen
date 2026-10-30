"""Full Markdown report (sections A-L) built from a :class:`RunResult`.

Every number is labelled with its epistemic level (observed / inferred / estimated /
speculative) and its identifiability. Mermaid block diagrams are embedded in fenced
```mermaid``` blocks so GitHub renders them inline.
"""
from __future__ import annotations

from typing import Any

from forensics.inference import identifiability as IDF
from forensics.inference.hypotheses import QUANTITIES
from forensics.inference.run_types import RunResult

from . import diagrams as DG
from . import extract as X
from .style import IDENT_LABELS, confidence_word


def build_markdown(run: RunResult, diags: dict[str, DG.Diagram] | None = None) -> str:
    diags = diags or DG.all_diagrams(run)
    P: list[str] = []
    w = P.append

    # ------------------------------------------------------------------ header
    tname = run.target.get("name", run.run_id)
    w(f"# Model forensics report — {_esc(tname)}")
    w("")
    w(f"- **Run id:** `{run.run_id}`")
    w(f"- **Endpoint kind:** {run.caps.kind}")
    w(f"- **Started / finished:** {run.started_at or '?'} → {run.finished_at or '?'}")
    w(f"- **Probes run:** {len(run.probe_results)}  ·  "
      f"**evidence:** {len(run.evidence)}  ·  **estimates:** {len(run.estimates)}")
    if run.budget_summary:
        w(f"- **Budget:** {_kv_inline(run.budget_summary)}")
    w("")
    w("> This is an evidence map, not a proved architecture reconstruction. "
      "Legacy scores are surrogate support, not calibrated probabilities. Missing probes/capabilities "
      "mean **unresolved (U)**; formal non-identifiability requires a scoped equivalence witness. "
      "See `forensic_certificates.json` for candidates, source integrity, alternatives, and falsifiers.")
    w("")
    _legend(w)

    # ------------------------------------------------------------------ A
    w("## A. Architecture hypothesis (prior → posterior)")
    w("")
    w("Block diagram inferred from the numbers (dashed = estimated/speculative, "
      "grey = unresolved by this scan):")
    w("")
    _mermaid(w, diags["block"].mermaid)
    w("")
    w("| Property | Winning value | Prior | Posterior | Δ | Level |")
    w("|---|---|---:|---:|---:|---|")
    for prop in run.posterior:
        val, post = X.posterior_top(run, prop)
        if val is None:
            continue
        prior = X.prior_prob(run, prop, val)
        ie = X.identifiability_for(run, prop)
        lvl = _lvl_from_ident(ie)
        w(f"| {_esc(prop)} | {_esc(X.value_short(val))} | {prior*100:.0f}% | "
          f"{post*100:.0f}% | {_delta(post - prior)} | {lvl} |")
    w("")
    w("### Serving topology")
    w("")
    w("Only components with evidence are drawn:")
    w("")
    _mermaid(w, diags["topology"].mermaid)
    w("")

    # ------------------------------------------------------------------ B
    w("## B. Estimated architecture (quantities)")
    w("")
    w("| Quantity | Value | 90% CI | Level | Identifiability | Method | Evidence |")
    w("|---|---|---|---|---|---|---|")

    def _resolved(q: str) -> bool:
        e = X.best_estimate(run, q)
        return e is not None and e.value is not None

    ordered = [q for q in QUANTITIES if _resolved(q)]
    extra = [e.quantity for e in run.estimates
             if e.quantity not in QUANTITIES and e.value is not None]
    for q in ordered + sorted(set(extra)):
        e = X.best_estimate(run, q)
        if e is None:
            continue
        unit = X.quantity_unit(q)
        val = X.fmt_value(e.value, unit)
        ci = X.fmt_ci(e.ci, e.value) or "—"
        ident = IDENT_LABELS.get(int(getattr(e, "identifiability", 5) or 5),
                                 str(e.identifiability))
        w(f"| {_esc(q)} | {_esc(val)} | {_esc(ci)} | {e.level} | {_esc(ident)} | "
          f"{_esc(_short(e.method, 60))} | `{_esc(e.probe_id)}` |")
    w("")
    unresolved = [q for q in QUANTITIES if not _resolved(q)
                  and (X.best_estimate(run, q) is not None
                       or X.identifiability_for(run, q) is not None)]
    if unresolved:
        w(f"**{X.NOT_IDENTIFIED.capitalize()}** (shown as explicit unknown nodes in the "
          "diagram; see section I for why): " +
          ", ".join(f"`{_esc(q)}`" for q in unresolved))
        w("")

    # ------------------------------------------------------------------ C
    w("## C. Evidence for every inference")
    w("")
    w("Experiment → conclusion traceability:")
    w("")
    _mermaid(w, diags["evidence"].mermaid)
    w("")
    w("### Likelihood evidence (per property)")
    w("")
    props_with_ev = _order_by_first_seen([ev.property for ev in run.evidence])
    for prop in props_with_ev:
        evs = X.evidence_for_property(run, prop)
        if not evs:
            continue
        w(f"#### {_esc(prop)} — {_esc(X.property_desc(prop))}")
        for ev in evs:
            w(f"- **Probe `{_esc(ev.probe_id)}`** ({ev.level}, strength {ev.strength:g}): "
              f"{_esc(ev.summary)}")
            liks = ", ".join(f"{_esc(k)}={v:.2g}" for k, v in
                             sorted(ev.likelihoods.items(), key=lambda kv: -kv[1])[:6])
            if liks:
                w(f"  - likelihoods P(obs\\|value): {liks}")
            st = _fmt_stats(ev.stats)
            if st:
                w(f"  - statistics: {st}")
            if ev.alternatives:
                w(f"  - alternative explanations: {_esc('; '.join(ev.alternatives))}")
        w("")
    w("### Quantity estimates (method + assumptions)")
    w("")
    for e in run.estimates:
        if e.value is None:
            continue
        ci = X.fmt_ci(e.ci, e.value)
        w(f"- **{_esc(e.quantity)} = {_esc(X.fmt_value(e.value, X.quantity_unit(e.quantity)))}**"
          + (f" (CI {_esc(ci)})" if ci else "")
          + f" — {e.level}, via `{_esc(e.probe_id)}`")
        if e.method:
            w(f"  - method: {_esc(e.method)}")
        if e.assumptions:
            w(f"  - assumptions: {_esc('; '.join(e.assumptions))}")
        if e.alternatives:
            w(f"  - alternatives: {_esc('; '.join(e.alternatives))}")
    w("")

    # ------------------------------------------------------------------ D
    w("## D. Experiments performed")
    w("")
    w("| Step | Probe | Reason | EIG (bits) | Entropy before → after | Cost | Requests | Status |")
    w("|---:|---|---|---:|---|---:|---:|---|")
    for st in run.experiment_log:
        eb = "—" if st.entropy_before_bits is None else f"{st.entropy_before_bits:.2f}"
        ea = "—" if st.entropy_after_bits is None else f"{st.entropy_after_bits:.2f}"
        eig = "—" if st.eig_bits is None else f"{st.eig_bits:.3f}"
        w(f"| {st.step} | `{_esc(st.probe_id)}` | {_esc(st.reason)} | {eig} | "
          f"{eb} → {ea} | {st.cost if st.cost is not None else '—'} | "
          f"{st.n_requests} | {st.status} |")
    w("")

    # ------------------------------------------------------------------ E
    w("## E. Raw observations")
    w("")
    if run.run_dir:
        w(f"Every request/response was recorded for replay at:")
        w("")
        w(f"    {run.run_dir}/observations.jsonl")
        w("")
        w("Re-run the whole analysis offline with `config['replay_dir']` pointing here "
          "(no network, no cost).")
    else:
        w("Recording was not enabled for this run (no `Recorder` supplied), so there is no "
          "`observations.jsonl`. Pass a `Recorder(run_dir=...)` to `run_forensics` to capture "
          "raw observations for replay.")
    w("")

    # ------------------------------------------------------------------ F
    w("## F. Statistical analysis")
    w("")
    w("Tests and effect sizes backing the evidence above:")
    w("")
    w("| Property | Probe | Test / observation | p-value | Effect / stat | n |")
    w("|---|---|---|---:|---|---:|")
    any_stat = False
    for ev in run.evidence:
        stt = ev.stats or {}
        test = stt.get("test") or _guess_test(stt)
        p = stt.get("p_value", stt.get("p"))
        eff = stt.get("effect", stt.get("effect_size"))
        n = stt.get("n", stt.get("samples"))
        if not (test or p is not None or eff is not None):
            continue
        any_stat = True
        w(f"| {_esc(ev.property)} | `{_esc(ev.probe_id)}` | {_esc(_short(str(test or '—'), 40))} | "
          f"{_fmt_p(p)} | {_esc(_short(str(eff) if eff is not None else '—', 24))} | "
          f"{n if n is not None else '—'} |")
    if not any_stat:
        w("| _(evidence carried descriptive statistics; see section C for per-probe detail)_ ||||||")
    w("")

    # ------------------------------------------------------------------ G
    w("## G. Competing hypotheses")
    w("")
    w("For each property, the top competing values (a posterior far from 1.0 means the "
      "interface did not decide between these):")
    w("")
    w("| Property | Top candidates (posterior) |")
    w("|---|---|")
    for prop in run.posterior:
        s = X.posterior_sorted(run, prop)[:3]
        if not s:
            continue
        cands = ", ".join(f"{_esc(X.value_short(v))} {p*100:.0f}%" for v, p in s)
        w(f"| {_esc(prop)} | {cands} |")
    w("")

    # ------------------------------------------------------------------ H
    w("## H. Confidence levels")
    w("")
    summ = _summary(run)
    if summ is not None:
        w("Legacy capability roll-up (1 directly observable … 5 unresolved; not a proof scale):")
        w("")
        w("| Level | Meaning | Count |")
        w("|---:|---|---:|")
        for lvl in range(1, 6):
            w(f"| {lvl} | {_esc(IDENT_LABELS[lvl])} | {summ.by_level.get(lvl, 0)} |")
        w("")
    # per-estimate confidence words
    w("Confidence of the headline quantities:")
    w("")
    for q in ("dims.hidden_size", "dims.vocab_size", "dims.n_layers", "arch.family",
              "arch.sparsity", "context.max_input_tokens"):
        e = X.best_estimate(run, q)
        if e is not None:
            from .style import ident_confidence
            cw = confidence_word(ident_confidence(getattr(e, "identifiability", None))
                                 if e.value is not None else None)
            w(f"- `{_esc(q)}`: **{_esc(X.fmt_value(e.value))}** — {e.level}, {cw} confidence")
        elif prop_exists := X.identifiability_for(run, q):
            w(f"- `{_esc(q)}`: {X.NOT_IDENTIFIED} — {_esc(prop_exists.label)}")
    w("")

    # ------------------------------------------------------------------ I
    w("## I. Unresolved properties and legacy capability assessments")
    w("")
    w("Missing probes or capabilities do not prove non-identifiability. Equivalence notes below "
      "are unverified legacy arguments; a certified L5 result needs an explicit scoped witness.")
    w("")
    if summ is not None:
        w(f"**Q: {_esc(summ.arch_question)}**")
        w("")
        w(f"**A: {_esc(summ.arch_verdict)}**")
        w("")
    by_level: dict[int, list] = {}
    for ie in run.identifiability:
        by_level.setdefault(ie.level, []).append(ie)
    for lvl in (5, 4, 3):
        group = by_level.get(lvl, [])
        if not group:
            continue
        w(f"### Level {lvl} — {_esc(IDENT_LABELS[lvl])} ({len(group)})")
        for ie in sorted(group, key=lambda x: x.key):
            w(f"- **{_esc(ie.key)}** — {_esc(ie.reason)}")
            if ie.equivalence:
                w(f"  - *equivalence:* {_esc(ie.equivalence)}")
        w("")

    # ------------------------------------------------------------------ J
    w("## J. Behavioural fingerprint")
    w("")
    if run.fingerprint:
        w("| Feature | Value |")
        w("|---|---|")
        for k, v in run.fingerprint.items():
            w(f"| {_esc(k)} | {_esc(_short(str(v), 80))} |")
    else:
        w("No behavioural fingerprint was recorded on this run "
          "(`run.fingerprint` is empty).")
    w("")

    # ------------------------------------------------------------------ K
    w("## K. Possible model-family matches")
    w("")
    if run.family_matches:
        w("| Rank | Family / model | Consistency | Matched constraints | Conflicts |")
        w("|---:|---|---:|---|---|")
        for i, fm in enumerate(run.family_matches, 1):
            w(f"| {i} | {_esc(fm.name)} | {fm.score*100:.0f}% | "
              f"{_esc(', '.join(fm.matched) or '—')} | {_esc(', '.join(fm.conflicts) or '—')} |")
        w("")
        w("> A high consistency score is necessary, not sufficient: families sharing a "
          "tokenizer/context/KV-scheme are behaviourally close even when the weights differ.")
    else:
        w("No family matches were produced (insufficient measured constraints).")
    w("")

    # ------------------------------------------------------------------ L
    w("## L. Recommended next experiments")
    w("")
    if run.recommendations:
        w("| Experiment | Why | Expected gain (bits) | Needs |")
        w("|---|---|---:|---|")
        for r in run.recommendations:
            g = "—" if r.expected_gain_bits is None else f"{r.expected_gain_bits:.3f}"
            w(f"| `{_esc(r.what)}` | {_esc(r.why)} | {g} | {_esc(r.needs or '—')} |")
    else:
        w("No further experiments were recommended: the adaptive loop reached diminishing "
          "returns (no unrun probe exceeds the minimum expected-information-gain threshold), "
          "and the remaining unknowns are non-identifiable from this interface (section I).")
    w("")

    if run.notes:
        w("---")
        w("### Run notes")
        for nt in run.notes:
            w(f"- {_esc(nt)}")
        w("")

    return "\n".join(P) + "\n"


# --------------------------------------------------------------------------- helpers
def _legend(w) -> None:
    w("### Legend")
    w("")
    w("| Level | Colour | Line | Meaning |")
    w("|---|---|---|---|")
    for r in DG.legend():
        w(f"| {r['level']} | `{r['color']}` | {r['line']} | {_esc(r['meaning'])} |")
    w("")


def _mermaid(w, src: str) -> None:
    w("```mermaid")
    for line in src.splitlines():
        w(line)
    w("```")


def _summary(run: RunResult):
    try:
        return IDF.summarize(run.identifiability)
    except Exception:  # noqa: BLE001
        return None


def _lvl_from_ident(ie) -> str:
    from .style import ident_to_level
    return ident_to_level(ie.level) if ie is not None else "unknown"


def _delta(d: float) -> str:
    arrow = "▲" if d > 0.005 else ("▼" if d < -0.005 else "•")
    return f"{arrow} {d*100:+.0f}%"


def _fmt_stats(stats: dict[str, Any]) -> str:
    if not stats:
        return ""
    parts = []
    for k, v in stats.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        elif isinstance(v, (list, dict)):
            parts.append(f"{k}=<{type(v).__name__}[{len(v)}]>")
        else:
            parts.append(f"{k}={v}")
    return _esc(", ".join(parts[:8]))


def _guess_test(stats: dict[str, Any]) -> str:
    for k in ("test", "method", "statistic"):
        if k in stats:
            return str(stats[k])
    return ""


def _fmt_p(p: Any) -> str:
    if p is None:
        return "—"
    try:
        pf = float(p)
    except (TypeError, ValueError):
        return str(p)
    if pf < 1e-4:
        return f"{pf:.1e}"
    return f"{pf:.4f}"


def _short(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _order_by_first_seen(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        seen.setdefault(it, None)
    return list(seen.keys())


def _kv_inline(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")
