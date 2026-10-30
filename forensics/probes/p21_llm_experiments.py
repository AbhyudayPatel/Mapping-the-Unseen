"""P21 - LLM-assisted automated experiment generator.

A *meta* probe: instead of measuring the target, it asks the optional LLM advisor
(:class:`forensics.llm.advisor.LLMAdvisor`, exposed as ``ctx.llm``) to INVENT differential
experiments that would best separate the current leading hypotheses, given the running
belief, the interface capabilities and the most uncertain properties.  The proposals are
*recorded* (as notes, a table, and ``shared``) as recommendations for future runs; this
probe does not execute them.

Fully optional and safe:

* when ``ctx.llm is None`` the probe is a no-op with ``status="skipped"`` (core inference is
  unaffected);
* it makes **no** requests to the target adapter (``n_requests == 0``);
* the advisor never raises, but the probe defends against it anyway.

When it does use the advisor it flags ``result.shared["llm_assisted"] = True`` so the engine
marks the run ``llm_assisted``.
"""
from __future__ import annotations

from typing import Any

from forensics.probes.base import Probe, ProbeContext, ProbeResult, register

_FIELDS = ("name", "hypothesis", "method", "observable", "distinguishes")


def _summarize_belief(belief: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(belief_rows, uncertainty_rows)`` from a Belief (or ``([], [])``).

    ``belief_rows``: leading value + probability per property.
    ``uncertainty_rows``: properties ranked by posterior entropy (most uncertain first).
    """
    posterior = getattr(belief, "posterior", None)
    if not isinstance(posterior, dict) or not posterior:
        return [], []
    belief_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    for prop, dist in posterior.items():
        if not isinstance(dist, dict) or not dist:
            continue
        top_val = max(dist, key=lambda k: dist[k])
        top_p = float(dist[top_val])
        try:
            ent = float(belief.entropy(prop))
        except Exception:  # noqa: BLE001
            ent = 0.0
        belief_rows.append({"property": prop, "leading": top_val, "p": round(top_p, 3)})
        uncertainty_rows.append({"property": prop, "entropy_bits": round(ent, 3),
                                 "leading": top_val, "p": round(top_p, 3)})
    uncertainty_rows.sort(key=lambda r: r["entropy_bits"], reverse=True)
    return belief_rows, uncertainty_rows[:8]


def _summarize_caps(caps: Any) -> dict[str, Any]:
    """A compact, salient slice of the interface capabilities for the advisor prompt."""
    keys = ("kind", "logprobs", "max_top_logprobs", "logit_bias",
            "logit_bias_affects_logprobs", "streaming", "seed", "n",
            "reasoning_effort", "usage_reasoning_tokens", "max_tokens_param")
    out: dict[str, Any] = {}
    for k in keys:
        v = getattr(caps, k, None)
        if v is not None:
            out[k] = v
    return out


@register
class LLMExperimentGenerator(Probe):
    id = "P21.llm.experiment_generator"
    title = "LLM-proposed differential experiments (automated experiment generation)"
    category = "meta"
    kinds = ("generative", "decision", "embedding")
    informs: dict[str, float] = {}          # meta-probe: informs no property directly
    estimates: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    cost = 1                                 # a single (optional) LLM call; no target requests
    baseline = True                          # run in the baseline sweep so the engine uses the advisor

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ctx.current_probe = self.id
        res = self.result()

        advisor = getattr(ctx, "llm", None)
        if advisor is None:
            res.status = "skipped"
            res.notes.append(
                "P21 no-op: no LLM advisor configured (ctx.llm is None); "
                "automated experiment generation is optional and was skipped.")
            return res

        belief_rows, uncertainty_rows = _summarize_belief(getattr(ctx, "belief", None))
        caps_summary = _summarize_caps(ctx.caps)
        uncertainty = [r["property"] for r in uncertainty_rows] or ["(unknown)"]

        try:
            proposals = await advisor.propose_experiments(
                belief_summary=belief_rows or "(belief at prior / unavailable)",
                caps_summary=caps_summary,
                uncertainty=uncertainty_rows or uncertainty)
        except Exception as exc:  # noqa: BLE001 - advisor is defensive, but never trust it
            res.status = "partial"
            res.shared["llm_assisted"] = True
            res.notes.append(f"P21: advisor.propose_experiments raised {exc!r}; no proposals.")
            return res

        proposals = proposals or []
        rows: list[dict[str, Any]] = []
        for i, p in enumerate(proposals, 1):
            if not isinstance(p, dict):
                continue
            row = {"rank": i}
            row.update({f: str(p.get(f, "")).strip() for f in _FIELDS})
            rows.append(row)

        res.tables["llm_proposed_experiments"] = rows
        res.shared["llm_proposed_experiments"] = rows
        res.shared["llm_assisted"] = True

        for r in rows:
            res.notes.append(
                f"[LLM-proposed experiment #{r['rank']}] {r['name'] or 'unnamed'}: "
                f"{r['method'] or 'method n/a'} -> observe {r['observable'] or 'n/a'} "
                f"(distinguishes {r['distinguishes'] or 'n/a'}).")

        # surface any degradation notes from the advisor for the audit trail
        for n in list(getattr(advisor, "notes", []))[-3:]:
            res.notes.append(f"advisor: {n}")

        if rows:
            res.notes.append(
                f"P21 recorded {len(rows)} LLM-proposed experiment(s) as recommendations "
                "(llm_assisted); they were NOT executed.")
            res.status = "ok"
        else:
            res.notes.append("P21: advisor returned no usable proposals.")
            res.status = "partial"
        return res
