"""P18 — decision-model (Jev-type) forensics.

Probes for a schema-constrained *decision* model (TypeSafe-Jev family): one shared
``state`` plus a map of typed questions (noul / choice / score) evaluated in parallel,
returning per-question answers with calibrated probabilities. There are no sampling,
logprob or streaming controls, so every side-channel here comes from the two things the
API does expose: the ``usage`` token counts and the reported answer probabilities.

Techniques distilled in ``research/jev.md`` §7:

* (a) **additivity / prefix sharing** — regress ``input_tokens`` on #questions and state
  length. The intercept is the fixed instruction template; a constant per-question slope
  plus a state cost that is billed *once* (independent of #questions) is prefix sharing.
  Estimates :data:`tokenizer.chat_template_overhead`.
* (b) **question isolation** — a secret placed in a *sibling* question is invisible to the
  target answer, while the same secret in the shared *state* moves it → questions are
  independent branches over a shared state prefix (``decision.question_coupling``).
* (c) **choice-set / IIA** — probabilities over options sum to 1 and adding a competitive
  option redistributes mass away from the others → a joint softmax over the in-context
  option set, not independent per-option scoring (``decision.scoring_mechanism``). A
  pure joint softmax preserves the *pairwise* log-odds when a low-mass option is added
  (no extra IIA violation); the real Jev shows a small residual log-odds shift, so that
  secondary test is reported with low strength.
* (d) **option position** — permuting the option order leaves each label's probability
  unchanged → options are read by meaning, not position (``decision.option_processing``).
* (e) **probability quantization** — reported probabilities lie on a 2-decimal grid
  (estimates :data:`decision.probability_resolution`).
* (f) **output-token readout** — ``output_tokens`` grows with label/key length but is flat
  in state length (compute) → a non-autoregressive, post-hoc-serialized readout
  (``output.head`` = classification heads; ``decision.scoring_mechanism``).

All probes go through ``ctx.decide`` only, handle ``resp.ok == False`` gracefully, shrink
under ``ctx.fast()`` and stay well under ~30 requests.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from forensics.core.types import DecisionQuestion, DecisionRequest
from forensics.probes.base import (
    DIRECT, STATISTICAL,
    BudgetExhausted, Estimate, Evidence, Probe, ProbeContext, ProbeResult, register,
)

# A neutral state used across probes (English, plausible support-ticket text).
_STATE = ("The customer was billed twice for the same monthly subscription and is "
          "asking for a refund of the duplicate charge.")

_FLOOR = 0.03


def _prob_l1(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    """L1 distance between two probability maps over the union of their keys."""
    a = a or {}
    b = b or {}
    keys = set(a) | set(b)
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _unrounded(ans) -> dict[str, float] | None:
    """Prefer the model's unrounded probabilities when the sim/adapter exposes them."""
    if ans is None:
        return None
    raw = getattr(ans, "raw", None)
    if isinstance(raw, dict) and isinstance(raw.get("unrounded"), dict):
        return {str(k): float(v) for k, v in raw["unrounded"].items()}
    if ans.probabilities:
        return {str(k): float(v) for k, v in ans.probabilities.items()}
    return None


def _noul_questions(nq: int, instr: str) -> dict[str, DecisionQuestion]:
    return {f"q{i}": DecisionQuestion(type="noul", instructions=instr) for i in range(nq)}


def _choice(labels: list[str], instr: str = "Which team should handle this ticket?"
            ) -> DecisionQuestion:
    return DecisionQuestion(type="choice", instructions=instr,
                            criteria={l: "" for l in labels})


# =========================================================================== (a)
@register
class DecisionAdditivityProbe(Probe):
    """Regress input_tokens on #questions & state length → template + prefix sharing."""

    id = "P18.decision.additivity"
    title = "Decision template additivity & state prefix sharing"
    category = "decision"
    kinds = ("decision",)
    informs = {"decision.question_coupling": 0.3}
    estimates = ("tokenizer.chat_template_overhead",)
    cost = 10
    baseline = True

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        instr = "Is this statement affirmative?"
        nqs = [1, 2, 3] if ctx.fast() else [1, 2, 3, 4, 5]
        try:
            # (1) input vs #questions at empty state.
            xs: list[int] = []
            ins: list[int] = []
            for nq in nqs:
                r = await ctx.decide(DecisionRequest(state="", questions=_noul_questions(nq, instr)))
                if not r.ok or r.usage.prompt_tokens is None:
                    continue
                xs.append(nq)
                ins.append(int(r.usage.prompt_tokens))
            # (2) state-sharing: same state at nq=1 and nq=max.
            state = _STATE
            share_rows: list[dict[str, Any]] = []
            deltas: list[float] = []
            for nq in (1, max(nqs)):
                r0 = await ctx.decide(DecisionRequest(state="", questions=_noul_questions(nq, instr)))
                rS = await ctx.decide(DecisionRequest(state=state, questions=_noul_questions(nq, instr)))
                if r0.ok and rS.ok and r0.usage.prompt_tokens and rS.usage.prompt_tokens:
                    d = int(rS.usage.prompt_tokens) - int(r0.usage.prompt_tokens)
                    deltas.append(float(d))
                    share_rows.append({"n_questions": nq, "state_token_cost": d})
        except BudgetExhausted:
            res.status = "partial"
        res.n_requests = ctx.n_requests

        if len(xs) < 2:
            res.status = "skipped"
            res.notes.append("additivity: too few successful decisions")
            return res

        slope, intercept = np.polyfit(np.asarray(xs, float), np.asarray(ins, float), 1)
        template = float(intercept)
        resid = np.asarray(ins, float) - (slope * np.asarray(xs, float) + intercept)
        r2 = 1.0 - float(np.sum(resid ** 2)) / max(1e-9, float(np.var(ins) * len(ins)))

        # State billed once ⇔ the per-state cost is the same at nq=1 and nq=max.
        billed_once = len(deltas) == 2 and abs(deltas[0] - deltas[1]) <= 1.0
        share_ratio = (abs(deltas[0] - deltas[1]) / max(1.0, deltas[0])) if len(deltas) == 2 else None

        res.tables["input_vs_questions"] = [
            {"n_questions": x, "input_tokens": y} for x, y in zip(xs, ins)
        ]
        if share_rows:
            res.tables["state_sharing"] = share_rows
        res.series["input_vs_questions"] = {
            "x": xs, "y": ins, "xlabel": "# questions", "ylabel": "input_tokens", "kind": "scatter",
        }

        res.add_estimate(Estimate(
            quantity="tokenizer.chat_template_overhead",
            value=int(round(template)),
            ci=(template - 2.0, template + 2.0),
            level="inferred",
            method="linear regression of input_tokens on #questions (intercept)",
            assumptions=["question wrapper cost is constant across questions",
                         "state is billed once (checked separately)"],
            identifiability=STATISTICAL,
            unit="tokens",
        ))

        if billed_once:
            res.add_evidence(Evidence(
                property="decision.question_coupling",
                likelihoods={"independent": 1.0, "coupled": 0.5},
                summary=(f"input_tokens = {template:.0f} + state + {slope:.1f}/question; state "
                         f"billed once (Δ={deltas} tokens) → shared state prefix, per-question branches"),
                level="inferred", strength=0.5,
                stats={"template_tokens": template, "per_question_tokens": float(slope),
                       "r2": r2, "state_cost_delta": deltas, "share_ratio": share_ratio},
                alternatives=["billing tokenizer differs from the model tokenizer",
                              "fixed template merges at question boundaries"],
            ))
            res.notes.append(
                f"Additive accounting: template≈{template:.0f} tok, +{slope:.1f} tok/question, "
                f"state billed once → prefix sharing (independent branches).")
        else:
            res.notes.append("State cost varied with #questions — additivity/sharing inconclusive.")
        return res


# =========================================================================== (b)
# Distinct secrets for the sibling-secret design (research/jev.md section 1, archerhume).
_SECRETS = ["AMBER-FALCON", "COBALT-RIVER", "SILENT-ORCHID", "IVORY-CANYON",
            "RUSTY-LANTERN", "VELVET-GLACIER", "NEON-HARBOR", "COPPER-THISTLE"]
_USES = 0.5          # P(true) above this = the answer used the secret
_EPS_ISOLATED = 0.03  # P(uses secret | secret only in a sibling, isolated branches)
_COUPLED_VIS = 0.7    # fraction of state-sensitivity a coupled prompt keeps for sibling text


def _binom_lik(k: int, n: int, p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def _secret_target(secret: str) -> DecisionQuestion:
    return DecisionQuestion(
        type="noul",
        instructions=f"Does the text state that the internal escalation code word is '{secret}'?")


@register
class DecisionIsolationProbe(Probe):
    """Secret in a sibling question is invisible; in the shared state it is visible.

    For each of K distinct secrets (K = ctx.scale(8), >= 3) three requests:
      * baseline — state without the secret, target question only;
      * secret in state — the state says "the code word is X", target asks about X;
      * secret in sibling — the state is clean, a SIBLING question carries "the code word is
        X", the target asks about X.
    Isolated branches (state + own question only) give P(uses | state) high and
    P(uses | sibling) ~ 0; one concatenated prompt lets the sibling text leak in.

    Likelihood from both contrasts: the state contrast measures the probe's sensitivity s
    (k_state/K, Laplace-smoothed); the sibling contrast is then scored as
    Binom(k_sib; K, eps) under ``independent`` vs Binom(k_sib; K, 0.7*s) under ``coupled``.
    An insensitive probe (s small) automatically yields a likelihood ratio near 1.
    """

    id = "P18.decision.question_isolation"
    title = "Question isolation (independent branches over a shared state)"
    category = "decision"
    kinds = ("decision",)
    informs = {"decision.question_coupling": 0.9}
    cost = 24
    baseline = True

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        res = self.result()
        k_secrets = min(len(_SECRETS), max(3, ctx.scale(8)))
        secrets = _SECRETS[:k_secrets]
        rows: list[dict[str, Any]] = []
        p_base: list[float] = []
        p_state: list[float] = []
        p_sib: list[float] = []
        try:
            for sec in secrets:
                target = _secret_target(sec)
                base = await ctx.decide(DecisionRequest(state=_STATE, questions={"t": target}))
                in_state = await ctx.decide(DecisionRequest(
                    state=_STATE + f" The internal escalation code word is {sec}.",
                    questions={"t": target}))
                sibling = await ctx.decide(DecisionRequest(
                    state=_STATE,
                    questions={
                        "s": DecisionQuestion(
                            type="noul",
                            instructions=(f"Note: the internal escalation code word is {sec}. "
                                          f"Is this ticket about billing?")),
                        "t": target}))
                vals = []
                for r in (base, in_state, sibling):
                    a = r.answers.get("t") if r.ok else None
                    v = a.value if a is not None else None
                    vals.append(float(v) if isinstance(v, (int, float)) else None)
                if any(v is None for v in vals):
                    res.notes.append(f"isolation: secret {sec}: a decision failed / no noul value")
                    continue
                b, st, sb = vals
                p_base.append(b)
                p_state.append(st)
                p_sib.append(sb)
                rows.append({"secret": sec, "p_baseline": round(b, 3),
                             "p_secret_in_state": round(st, 3),
                             "p_secret_in_sibling": round(sb, 3)})
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("isolation: budget exhausted")
        res.n_requests = ctx.n_requests

        n = len(rows)
        if n == 0:
            res.status = "partial" if res.status == "partial" else "skipped"
            res.notes.append("isolation: no complete secret triples")
            return res
        res.tables["isolation"] = rows

        k_state = sum(1 for v in p_state if v > _USES)
        k_sib = sum(1 for v in p_sib if v > _USES)
        k_base = sum(1 for v in p_base if v > _USES)
        d_state = float(np.mean(np.asarray(p_state) - np.asarray(p_base)))
        d_sibling = float(np.mean(np.asarray(p_sib) - np.asarray(p_base)))
        sens = (k_state + 0.5) / (n + 1.0)                   # smoothed sensitivity
        p_coupled = max(_EPS_ISOLATED * 2, _COUPLED_VIS * sens)
        l_ind = _binom_lik(k_sib, n, _EPS_ISOLATED)
        l_coup = _binom_lik(k_sib, n, p_coupled)
        m = max(l_ind, l_coup)
        like = {"independent": max(_FLOOR, l_ind / m), "coupled": max(_FLOOR, l_coup / m)}
        lr = l_ind / max(l_coup, 1e-300)
        sensitive = k_state >= max(2, math.ceil(0.6 * n)) and d_state > 0.2
        strength = 0.95 if sensitive else 0.3
        if sensitive and k_sib == 0:
            summ = (f"secret used when in the shared state ({k_state}/{n}, mean dP={d_state:+.2f}) "
                    f"but never when only in a sibling question ({k_sib}/{n}, "
                    f"dP={d_sibling:+.3f}) -> questions are isolated branches")
        elif sensitive:
            summ = (f"sibling-question secret leaked into the target answer in {k_sib}/{n} "
                    f"cases (dP={d_sibling:+.2f}) -> questions see each other (coupled)")
        else:
            summ = (f"isolation probe weakly sensitive: state secret used only {k_state}/{n} "
                    f"times; evidence tempered")

        res.series["isolation"] = {
            "x": ["baseline", "secret in state", "secret in sibling"],
            "y": [round(float(np.mean(p_base)), 3), round(float(np.mean(p_state)), 3),
                  round(float(np.mean(p_sib)), 3)],
            "xlabel": "condition", "ylabel": "mean P(target true)", "kind": "bar"}
        res.add_evidence(Evidence(
            property="decision.question_coupling", likelihoods=like, summary=summ,
            level="inferred", strength=strength,
            stats={"test": "binomial LR, sibling contrast conditioned on state sensitivity",
                   "n": n, "k_state": k_state, "k_sibling": k_sib, "k_baseline": k_base,
                   "d_state": d_state, "d_sibling": d_sibling, "sensitivity": sens,
                   "likelihood_ratio_independent": lr},
            alternatives=["the model ignores instructions-embedded facts even when coupled "
                          "(sibling text visible but not trusted)",
                          "literal-reading failures make the state contrast insensitive"],
        ))
        res.notes.append(
            f"Question isolation over {n} secrets: P(uses|state)={k_state}/{n}, "
            f"P(uses|sibling)={k_sib}/{n}, baseline {k_base}/{n}; LR(independent)={lr:.3g}.")
        return res


# =========================================================================== (c)
@register
class DecisionChoiceSetProbe(Probe):
    """Adding options redistributes mass (joint softmax) vs independent per-option scores."""

    id = "P18.decision.choice_set"
    title = "Choice-set / IIA test (joint softmax vs independent scoring)"
    category = "decision"
    kinds = ("decision",)
    informs = {"decision.scoring_mechanism": 0.8, "decision.option_processing": 0.2}
    cost = 4

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        pair = ["alpha", "bravo"]
        extras = ["charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
        n_extra = 4 if ctx.fast() else 6
        try:
            r2 = await ctx.decide(DecisionRequest(state=_STATE, questions={"q": _choice(pair)}))
            rN = await ctx.decide(DecisionRequest(
                state=_STATE, questions={"q": _choice(pair + extras[:n_extra])}))
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("choice_set: budget exhausted")
            return res
        res.n_requests = ctx.n_requests
        if not (r2.ok and rN.ok):
            res.status = "partial"
            res.notes.append("choice_set: a decision failed")
            return res

        a2 = r2.answers.get("q")
        aN = rN.answers.get("q")
        p2 = _unrounded(a2)
        pN = _unrounded(aN)
        if not p2 or not pN:
            res.status = "partial"
            res.notes.append("choice_set: no probabilities returned")
            return res

        sum2 = sum(p2.values())
        sumN = sum(pN.values())
        sums_to_one = abs(sum2 - 1.0) < 0.02 and abs(sumN - 1.0) < 0.02
        mass_pair_before = p2.get("alpha", 0.0) + p2.get("bravo", 0.0)
        mass_pair_after = pN.get("alpha", 0.0) + pN.get("bravo", 0.0)
        mass_taken = mass_pair_before - mass_pair_after     # >0 ⇒ added options grabbed mass

        # Secondary: does the pairwise log-odds(alpha:bravo) shift when options are added?
        def _logodds(p: dict[str, float]) -> float:
            a = max(p.get("alpha", 0.0), 1e-9)
            b = max(p.get("bravo", 0.0), 1e-9)
            return math.log(a / b)
        logodds_shift = abs(_logodds(pN) - _logodds(p2))

        res.tables["choice_set"] = [
            {"option_set": f"{len(pair)} options", "sum": round(sum2, 3),
             "mass_on_pair": round(mass_pair_before, 4)},
            {"option_set": f"{len(pair) + n_extra} options", "sum": round(sumN, 3),
             "mass_on_pair": round(mass_pair_after, 4)},
        ]

        # Joint softmax ⇔ probabilities normalise over the set and mass is redistributed.
        joint = sums_to_one and mass_taken > 0.05
        if joint:
            res.add_evidence(Evidence(
                property="decision.scoring_mechanism",
                likelihoods={"single_pass_heads": 1.0, "per_option_scoring": _FLOOR,
                             "generative_then_parse": 0.3},
                summary=(f"probabilities sum to 1 and adding {n_extra} options moved "
                         f"{mass_taken:.2f} of mass off the original pair → a joint softmax over "
                         f"the in-context option set (one forward pass reads out all options)"),
                level="inferred", strength=0.85,
                stats={"sum_small": sum2, "sum_large": sumN, "mass_taken": mass_taken,
                       "logodds_shift": logodds_shift},
                alternatives=["renormalisation applied post-hoc to independent scores",
                              "the added options happened to be low-probability"],
            ))
        else:
            res.notes.append(
                f"choice_set: joint-softmax signature weak (sum_to_one={sums_to_one}, "
                f"mass_taken={mass_taken:.3f}); keeping strength low.")
            res.add_evidence(Evidence(
                property="decision.scoring_mechanism",
                likelihoods={"single_pass_heads": 0.7, "per_option_scoring": 0.5,
                             "generative_then_parse": 0.5},
                summary="options renormalise but the redistribution was small on this item",
                level="inferred", strength=0.2,
                stats={"sum_small": sum2, "sum_large": sumN, "mass_taken": mass_taken},
                alternatives=["added options were irrelevant / near-zero mass"],
            ))

        # Secondary IIA log-odds test — a *pure* joint softmax leaves pairwise log-odds
        # unchanged; a residual shift (as the real Jev shows) is extra in-context interaction.
        res.add_evidence(Evidence(
            property="decision.option_processing",
            likelihoods={"semantic": 0.6, "positional": 0.5},
            summary=(f"pairwise log-odds shift on adding options = {logodds_shift:.3f} "
                     f"({'no extra IIA violation (clean joint softmax)' if logodds_shift < 0.05 else 'residual IIA violation'})"),
            level="inferred", strength=0.2,
            stats={"logodds_shift": logodds_shift},
            alternatives=["numeric jitter in a large-model softmax"],
        ))
        res.notes.append(
            f"Choice-set: sum→1 ({sum2:.2f}/{sumN:.2f}), mass off pair={mass_taken:.2f}, "
            f"log-odds shift={logodds_shift:.3f} (clean joint softmax in this sim; DOCUMENTED: the "
            f"real Jev shows a small residual log-odds shift the sim does not model).")
        return res


# =========================================================================== (d)
@register
class DecisionOptionPositionProbe(Probe):
    """Permuting the option order: label-invariant ⇒ semantic; position-tracking ⇒ positional."""

    id = "P18.decision.option_position"
    title = "Option position vs meaning"
    category = "decision"
    kinds = ("decision",)
    informs = {"decision.option_processing": 0.8}
    cost = 3

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        labels = ["refund_billing", "upsell_sales", "bug_technical"]
        try:
            fwd = await ctx.decide(DecisionRequest(state=_STATE, questions={"q": _choice(labels)}))
            rev = await ctx.decide(DecisionRequest(
                state=_STATE, questions={"q": _choice(list(reversed(labels)))}))
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("option_position: budget exhausted")
            return res
        res.n_requests = ctx.n_requests
        if not (fwd.ok and rev.ok):
            res.status = "partial"
            res.notes.append("option_position: a decision failed")
            return res

        pf = _unrounded(fwd.answers.get("q")) or {}
        pr = _unrounded(rev.answers.get("q")) or {}
        # Per-label drift when the option is moved to a different position.
        by_label = max((abs(pf.get(l, 0.0) - pr.get(l, 0.0)) for l in labels), default=1.0)
        # Positional drift: probability attached to position i regardless of label.
        seq_f = [pf.get(l, 0.0) for l in labels]
        seq_r = [pr.get(l, 0.0) for l in reversed(labels)]  # same label order → compare positions
        by_position = max((abs(a - b) for a, b in zip(seq_f, seq_r)), default=1.0)

        res.tables["option_position"] = [
            {"label": l, "p_forward": round(pf.get(l, 0.0), 4),
             "p_reversed": round(pr.get(l, 0.0), 4)} for l in labels
        ]

        semantic = by_label < 0.02
        if semantic:
            like = {"semantic": 1.0, "positional": _FLOOR}
            strength = 0.9
            summ = (f"each label kept its probability under permutation (max drift {by_label:.4f}) "
                    f"→ options scored by meaning, not position")
        elif by_position < by_label:
            like = {"semantic": _FLOOR, "positional": 1.0}
            strength = 0.7
            summ = (f"probability tracked option position (pos drift {by_position:.3f} < label drift "
                    f"{by_label:.3f}) → positional scoring")
        else:
            like = {"semantic": 0.6, "positional": 0.6}
            strength = 0.2
            summ = "option-position signal ambiguous"

        res.add_evidence(Evidence(
            property="decision.option_processing", likelihoods=like, summary=summ,
            level="inferred", strength=strength,
            stats={"drift_by_label": by_label, "drift_by_position": by_position},
            alternatives=["easy item saturates the max option so order cannot show",
                          "a small last-option (causal) advantage below the noise floor"],
        ))
        res.notes.append(f"Option position: label-drift={by_label:.4f}, position-drift={by_position:.3f}.")
        return res


# =========================================================================== (e)
@register
class DecisionProbabilityGridProbe(Probe):
    """Detect the discrete grid the reported probabilities live on (2-decimal for Jev)."""

    id = "P18.decision.probability_grid"
    title = "Probability quantization grid"
    category = "decision"
    kinds = ("decision",)
    informs = {}
    estimates = ("decision.probability_resolution",)
    cost = 6

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        states = [
            _STATE,
            "The server returned a 500 error during checkout and the order was lost.",
            "Great product, fast shipping, five stars, would buy again!",
            "I have been waiting three weeks for a reply and I am extremely frustrated.",
        ]
        if ctx.fast():
            states = states[:2]
        probs: list[float] = []
        try:
            for st in states:
                r = await ctx.decide(DecisionRequest(state=st, questions={
                    "route": _choice(["billing", "sales", "technical", "other"]),
                    "sev": DecisionQuestion(type="score", instructions="How severe is this?",
                                            criteria=["low", "medium", "high", "critical"]),
                }))
                if not r.ok:
                    continue
                for ans in r.answers.values():
                    if ans.probabilities:
                        probs.extend(float(v) for v in ans.probabilities.values())
        except BudgetExhausted:
            res.status = "partial"
        res.n_requests = ctx.n_requests

        probs = [p for p in probs if 0.0 < p < 1.0]     # 0/1 land on every grid; uninformative
        if len(probs) < 4:
            res.status = "skipped"
            res.notes.append("probability_grid: too few interior probabilities")
            return res

        arr = np.asarray(probs, float)

        def _grid_residual(step: float) -> float:
            return float(np.max(np.abs(arr / step - np.round(arr / step))))

        r01 = _grid_residual(0.01)
        r1 = _grid_residual(0.1)
        on_2dec = r01 < 1e-6
        off_1dec = r1 > 1e-6                              # not all multiples of 0.1

        # Choose the coarsest grid that explains every value.
        step = 0.01 if on_2dec else (0.1 if r1 < 1e-6 else float("nan"))
        decimals = 2 if step == 0.01 else (1 if step == 0.1 else None)

        res.tables["probability_grid"] = [
            {"grid_step": 0.1, "max_residual": round(r1, 6)},
            {"grid_step": 0.01, "max_residual": round(r01, 8)},
        ]
        res.series["probabilities"] = {
            "x": list(range(len(probs))), "y": [round(p, 4) for p in probs],
            "xlabel": "sample", "ylabel": "reported probability", "kind": "scatter",
        }

        if step == step:  # not nan
            res.add_estimate(Estimate(
                quantity="decision.probability_resolution", value=step,
                ci=(step, step), level="observed" if on_2dec and off_1dec else "inferred",
                method="grid-residual test over reported interior probabilities",
                assumptions=["probabilities are rounded to a fixed number of decimals"],
                identifiability=DIRECT,
                unit="",
            ))
            res.notes.append(
                f"Probabilities lie on a {decimals}-decimal grid (step={step}); "
                f"residual@0.01={r01:.1e}, residual@0.1={r1:.1e}.")
        else:
            res.notes.append("probability_grid: values not on a 1- or 2-decimal grid.")
        return res


# =========================================================================== (f)
@register
class DecisionReadoutProbe(Probe):
    """output_tokens grows with label length but not with compute ⇒ non-AR serialized readout."""

    id = "P18.decision.output_readout"
    title = "Output-token side channel (non-autoregressive readout)"
    category = "decision"
    kinds = ("decision",)
    informs = {"output.head": 0.6, "decision.scoring_mechanism": 0.3, "arch.family": 0.15}
    cost = 4

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        short = _choice(["a", "b", "c"])
        long = _choice(["refund_and_billing_department", "upsell_and_sales_department",
                        "engineering_bug_technical_department"])
        big_state = (_STATE + " ") * 40
        try:
            r_short = await ctx.decide(DecisionRequest(state=_STATE, questions={"q": short}))
            r_long = await ctx.decide(DecisionRequest(state=_STATE, questions={"q": long}))
            r_big = await ctx.decide(DecisionRequest(state=big_state, questions={"q": short}))
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("output_readout: budget exhausted")
            return res
        res.n_requests = ctx.n_requests
        if not (r_short.ok and r_long.ok and r_big.ok):
            res.status = "partial"
            res.notes.append("output_readout: a decision failed")
            return res

        out_short = int(r_short.usage.completion_tokens or 0)
        out_long = int(r_long.usage.completion_tokens or 0)
        out_big = int(r_big.usage.completion_tokens or 0)
        in_short = int(r_short.usage.prompt_tokens or 0)
        in_big = int(r_big.usage.prompt_tokens or 0)

        grows_with_labels = out_long > out_short
        flat_in_compute = out_big == out_short and in_big > in_short * 1.5

        res.tables["output_readout"] = [
            {"condition": "short labels", "input_tokens": in_short, "output_tokens": out_short},
            {"condition": "long labels", "input_tokens": in_short, "output_tokens": out_long},
            {"condition": "40x state (more compute)", "input_tokens": in_big, "output_tokens": out_big},
        ]

        non_ar = grows_with_labels and flat_in_compute
        if non_ar:
            res.add_evidence(Evidence(
                property="output.head",
                likelihoods={"classification_heads": 1.0, "generative_lm_head": 0.15},
                summary=(f"output_tokens grew with label length ({out_short}→{out_long}) but was flat "
                         f"when the state (compute) grew {in_short}→{in_big} tokens ({out_big}) → the "
                         f"answer is a post-hoc serialization, not autoregressive generation"),
                level="inferred", strength=0.8,
                stats={"out_short": out_short, "out_long": out_long, "out_big": out_big,
                       "in_short": in_short, "in_big": in_big},
                alternatives=["output_tokens is a billing counter unrelated to the head",
                              "generation is capped at 1 token and the count is metadata"],
            ))
            res.add_evidence(Evidence(
                property="decision.scoring_mechanism",
                likelihoods={"single_pass_heads": 1.0, "per_option_scoring": 0.6,
                             "generative_then_parse": 0.1},
                summary="no compute-dependent output → a single-pass head readout, not generate-then-parse",
                level="inferred", strength=0.5,
                stats={"out_short": out_short, "out_big": out_big},
                alternatives=["a very short fixed-length generation that is then parsed"],
            ))
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"decoder_only": 1.0, "encoder_only": 0.6, "encoder_decoder": 0.6,
                             "ssm_or_linear": 0.6, "hybrid_attn_ssm": 0.6, "diffusion": 0.5,
                             "non_neural": 0.2},
                summary="broad-knowledge decision model with a non-AR readout → most likely a "
                        "pretrained decoder LM behind decision heads (weak)",
                level="speculative", strength=0.2,
                stats={},
                alternatives=["an encoder classifier with enough pretraining",
                              "a bespoke non-transformer scorer"],
            ))
            res.notes.append(
                f"Readout: output {out_short}→{out_long} with label length, flat ({out_big}) under "
                f"{in_short}→{in_big} compute → non-autoregressive.")
        else:
            res.notes.append(
                f"Readout inconclusive (labels {out_short}->{out_long}, compute {out_short}->{out_big}).")
        return res
