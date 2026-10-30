"""P15 - sampling / decoding-stack fidelity probes.

Black-box audits of the *sampler* sitting between the logits and the returned text:

* ``P15.sampling.temperature_fidelity`` - draw many completions at T=1 on a low-entropy
  next token and compare the empirical first-token frequencies to the exposed top-logprob
  distribution.  A faithful sampler reproduces ``softmax(logits/T)``; a large gap means the
  reported logprobs are not the sampling distribution (hidden temperature / top-p / reranking).
  The probe also checks batch-invariance: are the reported logprobs bit-identical across two
  identical calls?  (Non-determinism => batching-sensitive serving.)
* ``P15.sampling.penalties`` - detect (a) deterministic decoding at T=0, (b) top-p / nucleus
  truncation (tail tokens with real probability mass never appear in samples), and
  (c) a repetition penalty (declining logprob of a token as it recurs).

Requires logprobs (to read the exposed distribution) or ``n`` (to draw many samples).  No
estimate is produced; the result carries a sampling-fidelity table.  Evidence informs
``serving.batch_invariant`` (weakly).  In the simulator, register a responder that samples
from ``softmax`` to give the sampler ground truth (see ``tests/test_grp_D.py``).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from forensics.core.types import GenRequest
from forensics.probes.base import (
    Evidence, Probe, ProbeContext, ProbeResult, register,
)

_FLOOR = 0.05


def _floor_pair(favored: str, other: str, p: float, floor: float = _FLOOR) -> dict[str, float]:
    p = float(min(1.0, max(0.0, p)))
    return {favored: max(floor, p), other: max(floor, 1.0 - p)}


def _exposed_first_token_dist(resp: Any) -> dict[str, float]:
    """{token_string: probability} from the first generated token's top-logprobs."""
    if not resp.ok or not resp.tokens:
        return {}
    top = resp.tokens[0].top
    if not top:
        return {}
    dist: dict[str, float] = {}
    for t in top:
        dist[t.token] = dist.get(t.token, 0.0) + math.exp(t.logprob)
    return dist


def _tv_distance(emp: dict[str, float], exp: dict[str, float]) -> float:
    """Total-variation distance over the exposed support plus a lumped tail bucket."""
    keys = set(exp)
    exp_mass = min(1.0, sum(exp.values()))
    e = dict(exp)
    e["__tail__"] = max(0.0, 1.0 - exp_mass)
    m: dict[str, float] = {"__tail__": 0.0}
    for k, v in emp.items():
        if k in keys:
            m[k] = m.get(k, 0.0) + v
        else:
            m["__tail__"] += v
    allk = set(e) | set(m)
    return 0.5 * sum(abs(m.get(k, 0.0) - e.get(k, 0.0)) for k in allk)


# --------------------------------------------------------------------------- temperature fidelity
@register
class TemperatureFidelity(Probe):
    id = "P15.sampling.temperature_fidelity"
    title = "Sampler fidelity: empirical T=1 frequencies vs exposed logprobs"
    category = "sampling"
    kinds = ("generative",)
    informs = {"serving.batch_invariant": 0.4}
    cost = 24

    _PROMPT = ("Continue this text with a single word: The quick brown fox jumps over the lazy")

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ctx.current_probe = self.id
        res = self.result()
        start = ctx.n_requests
        if not ctx.caps.logprobs:
            res.status = "skipped"
            res.error = "temperature fidelity needs logprobs to read the exposed distribution"
            return res

        k = min(20, ctx.caps.max_top_logprobs or 5)
        # 1) reference: the exposed next-token distribution
        try:
            ref = await ctx.gen(GenRequest.chat(self._PROMPT, max_tokens=1, temperature=1.0,
                                                logprobs=True, top_logprobs=k))
        except Exception as exc:
            res.status = "failed"
            res.error = str(exc)
            res.n_requests = ctx.n_requests - start
            return res
        exposed = _exposed_first_token_dist(ref)
        if not exposed:
            res.status = "partial"
            res.error = "no top-logprobs returned"
            res.n_requests = ctx.n_requests - start
            return res

        # 2) draw many completions at T=1 and tabulate the (verbatim) first token
        n = ctx.scale(80, minimum=12)
        reqs = [GenRequest.chat(self._PROMPT, max_tokens=1, temperature=1.0, seed=i)
                for i in range(n)]
        try:
            samples = await ctx.gen_many(reqs)
        except Exception as exc:
            res.status = "partial"
            res.error = str(exc)
            res.n_requests = ctx.n_requests - start
            return res
        texts = [s.text for s in samples if s.ok and s.text]
        if len(texts) < 4:
            res.status = "partial"
            res.error = "too few samples"
            res.n_requests = ctx.n_requests - start
            return res
        counts = Counter(texts)
        total = sum(counts.values())
        emp = {kk: v / total for kk, v in counts.items()}

        tv = _tv_distance(emp, exposed)
        fidelity = max(0.0, 1.0 - tv)
        exp_mode = max(exposed, key=exposed.get)
        emp_mode = max(emp, key=emp.get)
        mode_agree = exp_mode == emp_mode

        # 3) batch invariance: are the exposed logprobs identical across identical calls?
        batch_inv = None
        try:
            a = await ctx.gen(GenRequest.chat(self._PROMPT, max_tokens=1, temperature=0.0,
                                              logprobs=True, top_logprobs=k))
            b = await ctx.gen(GenRequest.chat(self._PROMPT, max_tokens=1, temperature=0.0,
                                              logprobs=True, top_logprobs=k))
            da, db = _exposed_first_token_dist(a), _exposed_first_token_dist(b)
            if da and db:
                keys = set(da) | set(db)
                max_dev = max(abs(da.get(kk, 0.0) - db.get(kk, 0.0)) for kk in keys)
                batch_inv = max_dev < 1e-6
        except Exception:
            pass

        # fidelity comparison table
        rows: list[dict[str, Any]] = []
        for tok in sorted(exposed, key=exposed.get, reverse=True)[:k]:
            rows.append({"token": repr(tok)[:14], "exposed_p": round(exposed[tok], 4),
                         "empirical_p": round(emp.get(tok, 0.0), 4)})
        res.tables["sampling_fidelity"] = rows
        res.tables["summary"] = [{
            "n_samples": total, "tv_distance": round(tv, 4),
            "temperature_fidelity": round(fidelity, 4), "mode_agreement": mode_agree,
            "batch_invariant": batch_inv,
        }]
        res.series["fidelity_scatter"] = {
            "x": [round(exposed[t], 4) for t in exposed],
            "y": [round(emp.get(t, 0.0), 4) for t in exposed],
            "xlabel": "exposed probability", "ylabel": "empirical frequency", "kind": "scatter",
        }
        res.shared["temperature_fidelity"] = round(fidelity, 4)

        if batch_inv is not None:
            res.add_evidence(Evidence(
                property="serving.batch_invariant",
                likelihoods=_floor_pair("yes", "no", 1.0 if batch_inv else 0.0),
                summary=("reported logprobs are bit-identical across identical calls"
                         if batch_inv else
                         "reported logprobs vary across identical calls -> batching-sensitive"),
                level="observed",
                strength=0.6,
                stats={"test": "logprob_repeatability", "n": 2},
                alternatives=["temperature/seed defaults could add sampling noise",
                              "logprobs may be rounded, hiding tiny batch jitter"],
            ))
        res.notes.append(
            f"Sampler fidelity {fidelity:.2f} (TV={tv:.2f}) over {total} T=1 draws; "
            f"mode {'matches' if mode_agree else 'differs from'} the exposed argmax.")
        res.n_requests = ctx.n_requests - start
        return res


# --------------------------------------------------------------------------- penalties / truncation
@register
class SamplingPenalties(Probe):
    id = "P15.sampling.penalties"
    title = "Determinism at T=0, top-p truncation and repetition penalty"
    category = "sampling"
    kinds = ("generative",)
    informs = {"serving.batch_invariant": 0.2}
    cost = 12

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ctx.current_probe = self.id
        res = self.result()
        start = ctx.n_requests
        rows: list[dict[str, Any]] = []

        # ---- (a) deterministic decoding at temperature 0 -----------------------
        deterministic = None
        prompt = "Write one short sentence about the sea."
        try:
            r1 = await ctx.gen(GenRequest.chat(prompt, max_tokens=24, temperature=0.0, seed=0))
            r2 = await ctx.gen(GenRequest.chat(prompt, max_tokens=24, temperature=0.0, seed=0))
            if r1.ok and r2.ok:
                deterministic = r1.text.strip() == r2.text.strip()
        except Exception as exc:
            res.status = "partial"
            res.error = str(exc)

        rows.append({"check": "temperature_0_determinism", "result": deterministic})

        # ---- (b) top-p / nucleus truncation ------------------------------------
        # If sampling produces variation, are only nucleus tokens ever seen?
        truncation = "indeterminate"
        nucleus_only = None
        distinct = None
        if ctx.caps.logprobs:
            k = min(20, ctx.caps.max_top_logprobs or 5)
            try:
                ref = await ctx.gen(GenRequest.chat(
                    "Name a random common English noun. One word only.",
                    max_tokens=1, temperature=1.0, logprobs=True, top_logprobs=k))
                exposed = _exposed_first_token_dist(ref)
                m = ctx.scale(24, minimum=6)
                sreqs = [GenRequest.chat(
                    "Name a random common English noun. One word only.",
                    max_tokens=1, temperature=1.0, top_p=0.5, seed=i) for i in range(m)]
                sres = await ctx.gen_many(sreqs)
                stexts = [s.text for s in sres if s.ok and s.text]
                distinct = len(set(stexts))
                if exposed and distinct is not None:
                    if distinct <= 1:
                        truncation = "indeterminate"   # deterministic serving; cannot assess
                    else:
                        # nucleus = smallest prefix of exposed mass reaching 0.5
                        order = sorted(exposed, key=exposed.get, reverse=True)
                        cum, nucleus = 0.0, set()
                        for t in order:
                            nucleus.add(t)
                            cum += exposed[t]
                            if cum >= 0.5:
                                break
                        outside = [t for t in stexts if t not in nucleus]
                        nucleus_only = len(outside) == 0
                        truncation = "active" if nucleus_only else "none"
            except Exception:
                pass
        rows.append({"check": "top_p_truncation", "result": truncation,
                     "distinct_sampled": distinct, "nucleus_only": nucleus_only})

        # ---- (c) repetition penalty --------------------------------------------
        rep_penalty = None
        rep_slope = None
        if ctx.caps.logprobs:
            try:
                rr = await ctx.gen(GenRequest.chat(
                    "Repeat the word 'na' many times: na na na na na na",
                    max_tokens=24, temperature=0.0, logprobs=True, top_logprobs=1))
                if rr.ok and rr.tokens:
                    toks = [(i, t.token, t.logprob) for i, t in enumerate(rr.tokens)]
                    # most frequent output token, look at its logprob vs occurrence order
                    freq = Counter(t for _, t, _ in toks)
                    common, cnt = freq.most_common(1)[0]
                    if cnt >= 3:
                        idx = [i for i, t, _ in toks if t == common]
                        lps = [lp for _, t, lp in toks if t == common]
                        slope = float(np.polyfit(np.arange(len(lps)), lps, 1)[0])
                        rep_slope = slope
                        # a real penalty pushes recurring-token logprob down over occurrences
                        rep_penalty = slope < -0.2
            except Exception:
                pass
        rows.append({"check": "repetition_penalty", "result": rep_penalty,
                     "recurrence_logprob_slope": round(rep_slope, 4) if rep_slope is not None else None})

        res.tables["sampling_checks"] = rows

        if deterministic is not None:
            res.add_evidence(Evidence(
                property="serving.batch_invariant",
                likelihoods=_floor_pair("yes", "no", 0.7 if deterministic else 0.3),
                summary=("temperature-0 decoding is reproducible"
                         if deterministic else "temperature-0 decoding is not reproducible"),
                level="observed",
                strength=0.3,
                stats={"test": "t0_determinism", "n": 2},
                alternatives=["temperature-0 determinism does not guarantee batch invariance",
                              "a cached response could fake determinism"],
            ))
        res.notes.append(
            f"determinism={deterministic}, top-p truncation={truncation}, "
            f"repetition penalty={rep_penalty}.")
        if res.status != "partial":
            res.status = "ok"
        res.n_requests = ctx.n_requests - start
        return res
