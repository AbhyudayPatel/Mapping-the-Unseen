"""P10 - Determinism / batch-invariance probe.

Observable: the *same* temperature-0 request issued N times.  A deterministic serving
stack returns byte-identical text and identical reported logprobs every time.  Any
variation at temperature 0 is a serving/numeric artefact, not sampling:

* **Continuous, small logprob jitter** (Gaussian-looking spread of the reported
  logprob of a fixed position) is the signature of batch/numeric non-invariance -
  reductions whose result depends on how the request was batched with others / on the
  kernel chosen under load (Anthropic/`thinking-machines` "defeating nondeterminism").
  => ``serving.batch_invariant = no``.
* **Lattice-quantised jitter** (the outputs snap to a small set of discrete values, i.e.
  the top-token identity flips but each logprob is drawn from a tiny discrete set) points
  at a *sampler* being engaged despite temperature 0 rather than kernel non-invariance.

We also compare jitter between many rapid (concurrent, "batched") repeats and slow
sequential ones - a load-correlated jitter magnitude is extra evidence of batch
non-invariance.  MoE routers are more prone to non-invariance, so jitter weakly informs
``arch.sparsity`` too.

No estimate is required from this probe; it produces Evidence only.
"""
from __future__ import annotations

import numpy as np

from forensics.probes.base import (
    Evidence, Probe, ProbeContext, ProbeResult, register,
)
from forensics.core.types import GenRequest

_FLOOR = 0.02
_JITTER_EPS = 1e-6          # below this the stream is treated as deterministic


def _first_logprob(resp) -> float | None:
    """Reported logprob of the first generated position (chosen token)."""
    if not getattr(resp, "tokens", None):
        return None
    t0 = resp.tokens[0]
    return float(t0.logprob) if t0.logprob is not None else None


def _first_top1(resp) -> float | None:
    """Top-1 logprob at the first position (robust to a top-token identity flip)."""
    if not getattr(resp, "tokens", None):
        return None
    t0 = resp.tokens[0]
    if t0.top:
        return float(max(tt.logprob for tt in t0.top))
    return float(t0.logprob) if t0.logprob is not None else None


def _lattice_score(vals: np.ndarray) -> float:
    """Fraction of *distinct* values among the samples (1.0 = all different, continuous;
    small = snaps to a few discrete levels, lattice-like)."""
    v = np.round(vals.astype(float), 9)
    if v.size == 0:
        return float("nan")
    return len(np.unique(v)) / v.size


@register
class DeterminismProbe(Probe):
    id = "P10.determinism.repeat_jitter"
    title = "Temperature-0 repeat determinism / batch invariance"
    category = "serving"
    kinds = ("generative",)
    informs = {"serving.batch_invariant": 0.85, "arch.sparsity": 0.15}
    estimates = ()
    requires = ()
    cost = 14
    baseline = True

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        n_rep = 12 if not ctx.fast() else max(6, ctx.scale(12, minimum=6))
        prompt = ("Write one plain sentence about the ocean. "
                  "Do not add anything else.")
        want_lp = bool(getattr(ctx.caps, "logprobs", False))
        top_k = min(5, int(getattr(ctx.caps, "max_top_logprobs", 5) or 5)) if want_lp else None

        def _mk() -> GenRequest:
            return GenRequest.chat(prompt, max_tokens=16, temperature=0.0, seed=0,
                                   logprobs=want_lp, top_logprobs=top_k)

        # Sequential repeats (concurrency=1: never contaminate the observation with load
        # we create ourselves), then a concurrent burst to look for load-correlated jitter.
        try:
            seq = await ctx.gen_many([_mk() for _ in range(n_rep)], concurrency=1)
            burst = await ctx.gen_many([_mk() for _ in range(n_rep)],
                                       concurrency=max(2, ctx.concurrency))
        except Exception as exc:  # pragma: no cover - budget/adapter failure
            res.status = "partial"
            res.notes.append(f"determinism probe aborted: {exc!r}")
            seq, burst = locals().get("seq", []), []
        res.n_requests = ctx.n_requests

        ok_seq = [r for r in seq if r.ok]
        ok_burst = [r for r in burst if r.ok]
        if len(ok_seq) < 3:
            res.status = "partial"
            res.notes.append("too few successful responses to judge determinism")
            return res

        texts = [r.text for r in ok_seq]
        n_distinct_text = len(set(texts))
        text_divergence = 1.0 - (texts.count(max(set(texts), key=texts.count)) / len(texts))

        lp_seq = np.array([v for v in (_first_logprob(r) for r in ok_seq) if v is not None])
        top_seq = np.array([v for v in (_first_top1(r) for r in ok_seq) if v is not None])
        lp_burst = np.array([v for v in (_first_logprob(r) for r in ok_burst) if v is not None])

        # Jitter = spread of a fixed observable across identical requests.
        jitter_std = float(np.std(top_seq)) if top_seq.size >= 2 else 0.0
        chosen_std = float(np.std(lp_seq)) if lp_seq.size >= 2 else 0.0
        jitter = max(jitter_std, chosen_std)
        lattice = _lattice_score(top_seq) if top_seq.size else float("nan")
        burst_std = float(np.std(lp_burst)) if lp_burst.size >= 2 else 0.0
        load_ratio = (burst_std / jitter) if jitter > _JITTER_EPS else float("nan")

        detected = (jitter > _JITTER_EPS) or (n_distinct_text > 1)
        # Distinguish continuous (numeric non-invariance) from a discrete sampler lattice.
        continuous = detected and (lattice != lattice or lattice >= 0.6 or top_seq.size < 4)

        res.tables["repeat_summary"] = [{
            "n_repeats": len(ok_seq),
            "distinct_texts": n_distinct_text,
            "text_divergence": round(text_divergence, 3),
            "logprob_jitter_std": round(jitter, 6),
            "top1_lattice_fraction": None if lattice != lattice else round(lattice, 3),
            "burst_jitter_std": round(burst_std, 6),
            "load_ratio_burst_over_seq": None if load_ratio != load_ratio else round(load_ratio, 3),
            "detected_nondeterminism": detected,
            "flavour": ("continuous_numeric" if (detected and continuous)
                        else "lattice_sampler" if detected else "deterministic"),
        }]
        if top_seq.size:
            res.series["top1_logprob_by_repeat"] = {
                "x": list(range(top_seq.size)), "y": [float(v) for v in top_seq],
                "xlabel": "repeat #", "ylabel": "top-1 logprob (pos 0)", "kind": "scatter",
            }

        # ---- Evidence: serving.batch_invariant --------------------------------
        if detected and continuous:
            like_bi = {"no": 0.95, "yes": _FLOOR}
            strength_bi = 0.9
            summ = (f"temperature-0 output not reproducible: continuous logprob jitter "
                    f"std={jitter:.4g} over {len(ok_seq)} identical requests")
        elif detected:  # discrete flips -> more likely a sampler than kernel non-invariance
            like_bi = {"no": 0.7, "yes": 0.35}
            strength_bi = 0.5
            summ = (f"temperature-0 output flips between a few discrete values "
                    f"(lattice frac={lattice:.2g}); sampler-like rather than continuous")
        else:
            like_bi = {"yes": 0.9, "no": 0.25}
            strength_bi = 0.85
            summ = (f"temperature-0 output byte-identical and logprobs constant over "
                    f"{len(ok_seq)} repeats: batch/numeric invariant")

        res.add_evidence(Evidence(
            property="serving.batch_invariant",
            likelihoods=like_bi,
            summary=summ,
            level="observed",
            strength=strength_bi,
            stats={"test": "temp0-repeat-jitter", "effect": jitter, "n": len(ok_seq),
                   "distinct_texts": n_distinct_text, "lattice_fraction": lattice,
                   "burst_std": burst_std, "load_ratio": load_ratio},
            alternatives=["temperature not actually 0 server-side",
                          "silent A/B routing between checkpoints",
                          "streaming re-tokenisation differences",
                          "prompt-cache warmup changing the first response only"],
        ))

        # ---- Evidence: arch.sparsity (weak) -----------------------------------
        if detected and continuous:
            like_sp, strength_sp = {"moe": 0.6, "dense": 0.45}, 0.3
            sp_summ = "non-invariant temp-0 logits: MoE routing is a common (not sole) cause"
        else:
            like_sp, strength_sp = {"dense": 0.55, "moe": 0.48}, 0.2
            sp_summ = "reproducible temp-0 logits: mildly favours a dense (or well-pinned) stack"
        res.add_evidence(Evidence(
            property="arch.sparsity",
            likelihoods=like_sp,
            summary=sp_summ,
            level="inferred",
            strength=strength_sp,
            stats={"jitter": jitter, "n": len(ok_seq)},
            alternatives=["non-invariance from all-reduce order, not sparsity",
                          "dense model on a non-deterministic kernel"],
        ))

        res.shared["determinism"] = {
            "nondeterministic": bool(detected),
            "logprob_jitter_std": jitter,
            "flavour": res.tables["repeat_summary"][0]["flavour"],
        }
        res.notes.append(
            f"P10: jitter={jitter:.4g}, distinct_texts={n_distinct_text}, "
            f"flavour={res.tables['repeat_summary'][0]['flavour']}")
        return res
