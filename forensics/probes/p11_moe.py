"""P11 - Mixture-of-Experts detection by domain-conditioned probing.

Idea (see research/techniques_timing_serving.md, arch notes): a sparse MoE routes each
token to a small subset of experts chosen by content, so *different input domains* can
exercise *different computational paths*.  We therefore send length-matched prompts drawn
from distinct regimes -

    english prose | source code | mathematics | multilingual | random tokens

- and measure, per regime, three side-channels:

  * latency (server-side seconds),
  * temperature-0 nondeterminism (logprob jitter across repeats), and
  * first-token logprob entropy.

If those features form **clusters that separate by domain but are tight within a domain**
(``mixture.gmm1d`` finds K>1 well-separated modes whose between-regime variance dwarfs the
within-regime variance) that is a fingerprint of *conditional computation* -> MoE.

Two corroborating signals are combined because pure domain-clustering is easy to spoof and
is often invisible on a lightly-loaded endpoint:

  * **Non-batch-invariance** (temp-0 jitter, shared with P10): MoE routers are a common
    source of load-dependent nondeterminism.
  * **capacity vs speed mismatch**: if a knowledge probe (P12) says the model *stores* far
    more parameters than a timing probe (P07) says are *active* per token, the only way to
    reconcile "big brain, cheap forward pass" is sparsity.  Read from ``ctx.shared``.

Estimates of ``dims.n_experts`` / ``dims.active_experts`` are only weakly identifiable from
a black-box interface (many (E, top_k) split the same active/total ratio); we emit them as
*speculative* with a wide CI and identifiability = non-identifiable, and say so.
"""
from __future__ import annotations

import math

import numpy as np

from forensics.probes.base import (
    Estimate, Evidence, NON_IDENTIFIABLE, Probe, ProbeContext, ProbeResult, register,
)
from forensics.core.types import GenRequest
from forensics.stats import gmm1d

_FLOOR = 0.02
_JITTER_EPS = 1e-6

# Length-matched (~40-word) prompts, five distinct computational regimes.
_REGIMES: dict[str, list[str]] = {
    "english": [
        "The old lighthouse keeper watched the grey waves roll in from the north as gulls "
        "circled the cliffs and the evening fog began to settle quietly over the small "
        "fishing village below the winding coastal road near the harbour wall.",
        "She opened the wooden drawer and found a bundle of letters tied with faded ribbon, "
        "each envelope carrying a distant postmark from a summer that everyone in the family "
        "had long since agreed never to mention again at the dinner table.",
    ],
    "code": [
        "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < "
        "len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
        "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]",
        "class RingBuffer:\n    def __init__(self, n):\n        self.n = n\n        self.buf = "
        "[None]*n\n        self.head = 0\n    def push(self, x):\n        self.buf[self.head] = x\n"
        "        self.head = (self.head + 1) % self.n\n        return self.buf",
    ],
    "math": [
        "Let f(x) = integral from 0 to x of e^{-t^2} dt. Compute the second-order Taylor "
        "expansion of f about x = 0, then evaluate the limit of f(x)/x as x approaches zero "
        "and justify each step using the fundamental theorem of calculus.",
        "Prove that for every prime p greater than three, p^2 minus one is divisible by "
        "twenty four, and then determine the number of integer solutions to the congruence "
        "x^2 is congruent to one modulo twenty four in the range one to one hundred.",
    ],
    "multilingual": [
        "Le vieux gardien du phare regardait les vagues grises rouler depuis le nord tandis "
        "que les mouettes tournoyaient et que la brume du soir descendait lentement sur le "
        "petit village de pêcheurs niché en contrebas de la route côtière sinueuse.",
        "El anciano farero observaba las olas grises llegar desde el norte mientras las "
        "gaviotas giraban sobre los acantilados y la niebla de la tarde se posaba en silencio "
        "sobre el pequeño pueblo pesquero junto al muro del puerto y la carretera.",
    ],
    "random": [
        "qux zib frotlike glemp vorn azzle brindt oqua feln jarvo umbly ketз wrunt phlex "
        "azzo bendiq crumf toqle vint wamp glorb usk teln vopra mizt harnul quep dref onzil "
        "brav tuxen lazp mirrо gwenд floop askt vrand melpo",
        "xar dwelt jonko frimb aztle wound crip zelva morn utby lantz quorf peld ithra "
        "sovm blint arkuz meppo drivn ozzle fant wemб qualt sindo vremp okla brint zulp "
        "harnо flend vusk teppa glomz arb wynct luzu",
    ],
}


def _first_logprob(resp) -> float | None:
    if not getattr(resp, "tokens", None):
        return None
    t0 = resp.tokens[0]
    return float(t0.logprob) if t0.logprob is not None else None


def _first_entropy(resp) -> float | None:
    """Entropy (nats) of the first-position top-k distribution (renormalised over top-k)."""
    if not getattr(resp, "tokens", None) or not resp.tokens[0].top:
        return None
    lps = np.array([tt.logprob for tt in resp.tokens[0].top], dtype=float)
    p = np.exp(lps - lps.max())
    p = p / p.sum()
    return float(-np.sum(p * np.log(np.maximum(p, 1e-12))))


def _server_s(resp) -> float | None:
    t = resp.timing
    if t is None:
        return None
    if t.server_ms is not None:
        return float(t.server_ms) / 1000.0        # server_ms is MILLISECONDS
    return float(t.t_end) if t.t_end is not None else None


def _cluster_separation(labels: list[str], values: list[float], rng) -> dict:
    """Between-regime vs within-regime dispersion of a feature, plus a GMM order check."""
    vals = np.asarray(values, dtype=float)
    if vals.size < 4 or not np.isfinite(vals).all():
        return {"eta_sq": float("nan"), "gmm_k": 1, "separated": False}
    grand = float(vals.mean())
    ss_tot = float(np.sum((vals - grand) ** 2)) or 1e-12
    ss_between = 0.0
    for lab in set(labels):
        g = vals[[i for i, l in enumerate(labels) if l == lab]]
        if g.size:
            ss_between += g.size * (float(g.mean()) - grand) ** 2
    eta_sq = ss_between / ss_tot                    # fraction of variance explained by domain
    gm = gmm1d(vals, k_range=(1, 2, 3), rng=rng)
    means = np.array(gm["means"])
    spread = float(np.std(vals)) or 1e-12
    gap = float(np.min(np.diff(means))) / spread if means.size > 1 else 0.0
    separated = (gm["k"] > 1 and gap > 1.0 and eta_sq > 0.5)
    return {"eta_sq": eta_sq, "gmm_k": int(gm["k"]), "gmm_gap": gap, "separated": separated}


@register
class MoEProbe(Probe):
    id = "P11.moe.domain_regimes"
    title = "MoE detection via domain-conditioned computation"
    category = "architecture"
    kinds = ("generative",)
    informs = {"arch.sparsity": 0.7}
    estimates = ("dims.n_experts", "dims.active_experts")
    requires = ()
    depends_on = ("P07.timing.prefill", "P12.knowledge.graded_recall")
    cost = 30

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        n_prompts = 1 if ctx.fast() else 2
        n_rep = 3                                    # need >=3 to estimate jitter
        want_lp = bool(getattr(ctx.caps, "logprobs", False))
        top_k = min(5, int(getattr(ctx.caps, "max_top_logprobs", 5) or 5)) if want_lp else None

        rows: list[dict] = []
        feat_labels: list[str] = []
        lat_vals: list[float] = []
        jit_vals: list[float] = []
        ent_vals: list[float] = []

        for regime, prompts in _REGIMES.items():
            for prompt in prompts[:n_prompts]:
                reqs = [GenRequest.chat(prompt, max_tokens=12, temperature=0.0, seed=0,
                                        logprobs=want_lp, top_logprobs=top_k)
                        for _ in range(n_rep)]
                try:
                    reps = await ctx.gen_many(reqs, concurrency=1)
                except Exception as exc:  # pragma: no cover
                    res.status = "partial"
                    res.notes.append(f"regime {regime} aborted: {exc!r}")
                    reps = []
                ok = [r for r in reps if r.ok]
                if not ok:
                    continue
                lats = [s for s in (_server_s(r) for r in ok) if s is not None]
                lps = [v for v in (_first_logprob(r) for r in ok) if v is not None]
                ents = [e for e in (_first_entropy(r) for r in ok) if e is not None]
                lat = float(np.median(lats)) if lats else float("nan")
                jit = float(np.std(lps)) if len(lps) >= 2 else 0.0
                ent = float(np.median(ents)) if ents else float("nan")
                rows.append({"regime": regime, "latency_s": round(lat, 5),
                             "logprob_jitter": round(jit, 6),
                             "entropy_nats": None if ent != ent else round(ent, 4),
                             "n": len(ok)})
                if lat == lat:
                    feat_labels.append(regime); lat_vals.append(lat)
                jit_vals.append(jit)
                if ent == ent:
                    ent_vals.append(ent)
        res.n_requests = ctx.n_requests

        if not rows:
            res.status = "partial"
            res.notes.append("no successful regime responses")
            return res

        res.tables["regime_features"] = rows
        res.series["latency_by_regime"] = {
            "x": [r["regime"] for r in rows], "y": [r["latency_s"] for r in rows],
            "xlabel": "regime", "ylabel": "server latency (s)", "kind": "bar",
        }

        # ---- domain-conditioned clustering ------------------------------------
        clus = _cluster_separation(feat_labels, lat_vals, ctx.rng) if lat_vals else \
            {"eta_sq": float("nan"), "gmm_k": 1, "separated": False}
        overall_jitter = float(np.median([j for j in jit_vals])) if jit_vals else 0.0
        nondet = overall_jitter > _JITTER_EPS

        # ---- capacity vs speed mismatch (from shared, if available) -----------
        total_b = _shared_num(ctx, ["knowledge.total_params_b", "total_params_b",
                                    "dims.total_params"])
        active_b = _shared_num(ctx, ["timing.active_params_b", "active_params_b",
                                     "dims.active_params"])
        mismatch = None
        if total_b and active_b and active_b > 0:
            mismatch = total_b / active_b

        # ---- Evidence 1: domain clustering (weak by design) ----------------------
        # Domain-conditioned latency is confounded by token composition, autoscaling
        # and cache hits; ForensicsBench measured it leaning MoE on dense CPU subjects.
        # It is therefore a weak lean, never a strong update: the interleaved,
        # length-matched control is the strong path.
        if clus["separated"]:
            like, strength = {"moe": 0.62, "dense": 0.42}, 0.3
            summ = (f"latency clusters separate by domain "
                    f"(eta^2={clus['eta_sq']:.2f}, GMM k={clus['gmm_k']}): weak MoE lean; "
                    f"confounded by token composition")
        else:
            like, strength = {"dense": 0.6, "moe": 0.4}, 0.2
            summ = (f"no domain-separated compute regime "
                    f"(eta^2={clus['eta_sq']:.2f}, GMM k={clus['gmm_k']}); "
                    f"expected on a lightly-loaded endpoint even for MoE")
        res.add_evidence(Evidence(
            property="arch.sparsity", likelihoods=like, summary=summ, level="inferred",
            strength=strength,
            stats={"test": "domain-latency-gmm", "eta_sq": clus["eta_sq"],
                   "gmm_k": clus["gmm_k"], "gmm_gap": clus.get("gmm_gap"),
                   "n": len(lat_vals)},
            alternatives=["latency differences from prompt-length mismatch, not routing",
                          "autoscaler / cold-start noise mimicking regimes",
                          "cache hits on some regimes only"],
        ))

        # ---- Evidence 2: nondeterminism (the reliable in-sim discriminator) ----
        if nondet:
            like2, strength2 = {"moe": 0.75, "dense": 0.3}, 0.6
            s2 = (f"temperature-0 logprob jitter (median std={overall_jitter:.4g}) - "
                  f"load-dependent routing is a frequent cause in MoE serving")
        else:
            like2, strength2 = {"dense": 0.7, "moe": 0.35}, 0.5
            s2 = "temperature-0 logprobs reproducible: no routing-induced nondeterminism seen"
        res.add_evidence(Evidence(
            property="arch.sparsity", likelihoods=like2, summary=s2, level="inferred",
            strength=strength2,
            stats={"test": "temp0-jitter", "effect": overall_jitter, "n": len(jit_vals)},
            alternatives=["kernel all-reduce nondeterminism on a dense model",
                          "fully deterministic MoE (pinned routing)"],
        ))

        # ---- Evidence 3: capacity vs speed (only if we have both numbers) ------
        if mismatch is not None:
            if mismatch > 1.8:
                like3, strength3 = {"moe": 0.9, "dense": 0.12}, 0.8
                s3 = (f"stored capacity >> active compute (total/active ~ {mismatch:.1f}x): "
                      f"only sparsity reconciles a big model with a cheap forward pass")
            else:
                like3, strength3 = {"dense": 0.7, "moe": 0.3}, 0.5
                s3 = f"total/active ~ {mismatch:.1f}x: consistent with a dense model"
            res.add_evidence(Evidence(
                property="arch.sparsity", likelihoods=like3, summary=s3, level="estimated",
                strength=strength3,
                stats={"total_params_b": total_b, "active_params_b": active_b,
                       "ratio": mismatch},
                alternatives=["over-estimated knowledge params", "under-estimated active params",
                              "aggressive quantisation shrinking the active-byte estimate"],
            ))

        # ---- speculative expert-count estimate (non-identifiable) --------------
        moe_indicated = clus["separated"] or (mismatch is not None and mismatch > 1.8)
        n_exp_guess = 8
        act_guess = 2
        if mismatch is not None and mismatch > 1.0:
            # top_k/E ~ active/total (very rough); assume top_k=2 by convention.
            act_guess = 2
            n_exp_guess = int(max(2, round(2 * mismatch)))
        res.add_estimate(Estimate(
            quantity="dims.n_experts", value=None,   # non-identifiable: report range only, no point guess
            ci=(2, 256), ci_level=0.9, level="speculative",
            method=(f"NOT identifiable black-box; a top_k=2 convention would imply ~{n_exp_guess} experts" if moe_indicated else "NOT identifiable black-box"),
            assumptions=["fixed top_k=2", "uniform expert sizes",
                         "active/total ratio ~ top_k/n_experts"],
            alternatives=["any (E, top_k) with the same active/total ratio is observationally equivalent"],
            identifiability=NON_IDENTIFIABLE,
        ))
        res.add_estimate(Estimate(
            quantity="dims.active_experts", value=None,   # non-identifiable: report range only
            ci=(1, 16), ci_level=0.9, level="speculative",
            method="NOT identifiable black-box: routing width cannot be separated from expert size",
            assumptions=["fixed top_k convention"],
            identifiability=NON_IDENTIFIABLE,
        ))

        res.shared["moe"] = {"nondeterministic": bool(nondet),
                             "domain_separated": bool(clus["separated"]),
                             "capacity_speed_ratio": mismatch}
        res.notes.append(
            f"P11: eta^2={clus['eta_sq']:.3g}, gmm_k={clus['gmm_k']}, "
            f"jitter={overall_jitter:.4g}, total/active={mismatch}")
        return res


def _shared_num(ctx: ProbeContext, keys: list[str]) -> float | None:
    """Best-effort numeric lookup across a few plausible ctx.shared key spellings."""
    shared = getattr(ctx, "shared", {}) or {}
    for k in keys:
        if k in shared:
            v = shared[k]
            if isinstance(v, dict):
                v = v.get("value")
            try:
                fv = float(v)
                if math.isfinite(fv):
                    return fv
            except (TypeError, ValueError):
                continue
    return None
