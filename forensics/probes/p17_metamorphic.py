"""P17 - metamorphic invariance battery (behavioural robustness fingerprint).

Apply transformations that should NOT change the answer - paraphrase, whitespace/case
changes, reordering irrelevant facts, adding a distractor, reformatting - and measure how
much the answer (and, when available, the next-token distribution) drifts.  The per-
transformation drift vector is a behavioural robustness fingerprint that separates a
well-aligned instruction model (stable) from a brittle/base model (sensitive to surface
form).  The vector is published as ``shared['fingerprint_metamorphic']``.

Evidence weakly informs ``post.instruction_tuning`` and ``arch.family``: a robust model is
(weakly) more consistent with a preference-optimised chat model.  Ground truth in the
simulator comes from a responder that answers consistently (low drift) or inconsistently
(high drift); see ``tests/test_grp_D.py``.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from forensics.core.types import GenRequest
from forensics.probes.base import (
    Evidence, Probe, ProbeContext, ProbeResult, register,
)
from forensics.stats.tests import bootstrap_ci

_FLOOR = 0.05


def _norm_answer(text: str) -> str:
    """Canonical short answer: first alphanumeric token, lowercased."""
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return toks[0] if toks else ""


def _first_token_dist(resp: Any) -> dict[str, float]:
    import math
    if not resp.ok or not resp.tokens or not resp.tokens[0].top:
        return {}
    d: dict[str, float] = {}
    for t in resp.tokens[0].top:
        d[t.token] = d.get(t.token, 0.0) + math.exp(t.logprob)
    return d


def _l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


# --------------------------------------------------------------------------- transformations
def _render(facts: list[str], question: str) -> str:
    return "Facts: " + " ".join(facts) + " Question: " + question


def _t_identity(facts, question, rng):
    return _render(facts, question)


def _t_whitespace(facts, question, rng):
    return ("Facts:\n\n  " + "   ".join(facts) + "\n\n   Question:   " + question + "   ")


def _t_case(facts, question, rng):
    return _render(facts, question).upper()


def _t_paraphrase(facts, question, rng):
    return ("Please read the notes and then answer. Notes: " + " ".join(facts)
            + " Now, " + question.rstrip("?") + "?")


def _t_reorder(facts, question, rng):
    f = list(facts)
    rng.shuffle(f)
    return _render(f, question)


def _t_distractor(facts, question, rng):
    f = list(facts) + ["Also, the number seven is an odd integer.",
                       "By the way, Tuesday comes after Monday."]
    return _render(f, question)


def _t_format(facts, question, rng):
    bullets = "\n".join(f"- {x}" for x in facts)
    return f"### Context\n{bullets}\n\n### Task\n**{question}**"


_TRANSFORMS: dict[str, Callable] = {
    "whitespace": _t_whitespace,
    "case": _t_case,
    "paraphrase": _t_paraphrase,
    "reorder": _t_reorder,
    "distractor": _t_distractor,
    "format": _t_format,
}


# --------------------------------------------------------------------------- probe
@register
class MetamorphicInvariance(Probe):
    id = "P17.metamorphic.invariance"
    title = "Metamorphic invariance battery (robustness fingerprint)"
    category = "metamorphic"
    kinds = ("generative",)
    informs = {"post.instruction_tuning": 0.3, "arch.family": 0.15}
    cost = 20
    baseline = True

    # (facts, question) - question is answerable from one fact; the rest are irrelevant.
    _ITEMS = (
        (["The sky is blue.", "Grass is green.", "Fresh snow is white."],
         "What color is grass? Answer with one word."),
        (["Paris is in France.", "Rome is in Italy.", "Cairo is in Egypt."],
         "In which country is Rome? Answer with one word."),
        (["A triangle has 3 sides.", "A square has 4 sides.", "A pentagon has 5 sides."],
         "How many sides does a square have? Answer with one number."),
        (["Cats say meow.", "Dogs say woof.", "Cows say moo."],
         "What sound does a dog make? Answer with one word."),
    )

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ctx.current_probe = self.id
        res = self.result()
        start = ctx.n_requests
        items = list(self._ITEMS)
        n_items = max(2, ctx.scale(len(items)))
        items = items[:n_items]
        transforms = list(_TRANSFORMS.items())
        use_lp = bool(ctx.caps.logprobs)
        k = min(10, ctx.caps.max_top_logprobs or 5) if use_lp else 0

        def mk(prompt: str) -> GenRequest:
            return GenRequest.chat(prompt, max_tokens=16, temperature=0.0,
                                   logprobs=use_lp, top_logprobs=(k or None))

        # base (untransformed) answers
        base_prompts = [_render(f, q) for f, q in items]
        try:
            base_resps = await ctx.gen_many([mk(p) for p in base_prompts])
        except Exception as exc:
            res.status = "failed"
            res.error = str(exc)
            res.n_requests = ctx.n_requests - start
            return res
        base_ans = [_norm_answer(r.text) if r.ok else None for r in base_resps]
        base_dist = [_first_token_dist(r) for r in base_resps]

        fingerprint: dict[str, float] = {}
        lp_fingerprint: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        per_item_all: list[float] = []

        for tname, tfn in transforms:
            # deterministic per-transform rng (stable shuffles)
            rng = np.random.default_rng(ctx.rng.integers(0, 2**31 - 1))
            prompts = [tfn(f, q, rng) for f, q in items]
            try:
                resps = await ctx.gen_many([mk(p) for p in prompts])
            except Exception as exc:
                res.status = "partial"
                res.error = str(exc)
                break
            disagree: list[float] = []
            lp_dists: list[float] = []
            for i, r in enumerate(resps):
                if base_ans[i] is None or not r.ok:
                    continue
                a = _norm_answer(r.text)
                disagree.append(0.0 if a == base_ans[i] else 1.0)
                if use_lp and base_dist[i]:
                    lp_dists.append(_l1(_first_token_dist(r), base_dist[i]))
            d = float(np.mean(disagree)) if disagree else 0.0
            fingerprint[tname] = round(d, 4)
            per_item_all.extend(disagree)
            if lp_dists:
                lp_fingerprint[tname] = round(float(np.mean(lp_dists)), 4)
            rows.append({"transform": tname, "answer_drift": round(d, 4),
                         "logprob_l1": lp_fingerprint.get(tname),
                         "n": len(disagree)})

        vec = [fingerprint.get(name, 0.0) for name, _ in transforms]
        mean_instability = float(np.mean(vec)) if vec else 0.0
        robustness = 1.0 - mean_instability
        if per_item_all:
            lo, hi, _ = bootstrap_ci(np.array(per_item_all, dtype=float),
                                     stat=np.mean, level=0.9, rng=ctx.rng)
        else:
            lo, hi = 0.0, 0.0

        res.tables["metamorphic"] = rows
        res.tables["summary"] = [{
            "mean_answer_instability": round(mean_instability, 4),
            "robustness": round(robustness, 4),
            "instability_ci": [round(lo, 4), round(hi, 4)],
            "transforms": len(vec), "items": len(items),
        }]
        res.series["fingerprint"] = {
            "x": [name for name, _ in transforms], "y": vec,
            "xlabel": "transformation", "ylabel": "answer drift", "kind": "bar"}
        # published fingerprint vector (answer-drift per transform, stable order)
        res.shared["fingerprint_metamorphic"] = vec
        res.shared["fingerprint_metamorphic_labels"] = [name for name, _ in transforms]
        res.shared["metamorphic_robustness"] = round(robustness, 4)

        res.add_evidence(Evidence(
            property="post.instruction_tuning",
            likelihoods={"instruct": max(_FLOOR, robustness),
                         "base": max(_FLOOR, 1.0 - robustness)},
            summary=(f"answer stable under {sum(1 for v in vec if v < 0.25)}/{len(vec)} "
                     f"surface transforms (mean drift {mean_instability:.2f}); "
                     f"{'robust -> aligned chat model' if robustness >= 0.6 else 'brittle to surface form'}"),
            level="inferred",
            strength=0.3,
            stats={"test": "metamorphic_answer_drift", "effect": round(mean_instability, 3),
                   "ci": [round(lo, 3), round(hi, 3)], "n": len(per_item_all)},
            alternatives=["a base model with strong in-context format can also be stable",
                          "trivial questions may be answered correctly regardless of tuning",
                          "temperature-0 decoding removes sampling-driven drift"],
        ))
        # very weak architecture hint: robust surface-form handling is mildly more typical
        # of large decoder-only chat models than of tiny/encoder classifiers.
        res.add_evidence(Evidence(
            property="arch.family",
            likelihoods={"decoder_only": 0.55 if robustness >= 0.6 else 0.5,
                         "encoder_only": 0.4, "ssm_or_linear": 0.45},
            summary=f"metamorphic robustness {robustness:.2f} (weak family hint)",
            level="speculative",
            strength=0.15,
            stats={"effect": round(robustness, 3)},
            alternatives=["surface-form robustness is only weakly tied to architecture family"],
        ))
        if res.status != "partial":
            res.status = "ok"
        res.n_requests = ctx.n_requests - start
        return res
