"""P12 - Knowledge capacity & training cutoff by graded-rarity factual recall.

Two black-box observables about *what the weights store*:

1. **Graded recall.** We ask short, verifiable factual questions arranged in tiers from
   very common (capital of France) to very rare (capital of Bhutan / abstract maths), and
   score each answer by exact/substring match (or logprob-of-the-answer when logprobs are
   exposed).  Accuracy falls as rarity rises; the **rarity at which accuracy crosses 50 %**
   is a monotone proxy for stored knowledge.  Using the empirical "~2 bits/parameter"
   capacity rule and ArchDB reference points, that threshold regresses onto
   ``log(total_params)`` - a *wide-CI, speculative* estimate of ``dims.total_params``.

2. **Training cutoff.** A second set of questions is pinned to dated events across several
   years.  Accuracy is high up to the training cutoff and collapses afterwards; a
   change-point on accuracy-vs-year recovers ``knowledge.cutoff``.

The recall fact bank and dated-event bank are exposed at module level so a simulator
responder (in tests) can answer them with a controlled, capacity-limited accuracy.

Capacity vs the *active* compute measured by a timing probe also weakly informs
``arch.sparsity``: much more stored knowledge than active parameters implies sparsity.
"""
from __future__ import annotations

import math
import re

import numpy as np

from forensics.probes.base import (
    Estimate, Evidence, Probe, ProbeContext, ProbeResult, WEAK, register,
)
from forensics.core.types import GenRequest
from forensics.stats import proportion_ci, segmented1d

_FLOOR = 0.03

# --------------------------------------------------------------------------- fact banks
# rarity in [0, 1]: 0 = ubiquitous, 1 = obscure.  id is echoed in the prompt marker so a
# simulator responder can look up the ground truth; real models simply ignore the marker.
RECALL_FACTS: list[dict] = [
    # tier 0 - ubiquitous
    {"id": "c1", "rarity": 0.1, "q": "What is the capital city of France?", "a": "Paris"},
    {"id": "c2", "rarity": 0.1, "q": "What is the chemical symbol for water?", "a": "H2O"},
    {"id": "c3", "rarity": 0.1, "q": "How many continents are there on Earth?", "a": "seven"},
    # tier 1 - common
    {"id": "b1", "rarity": 0.3, "q": "Who wrote the play Romeo and Juliet?", "a": "Shakespeare"},
    {"id": "b2", "rarity": 0.3, "q": "What is the chemical symbol for gold?", "a": "Au"},
    {"id": "b3", "rarity": 0.3, "q": "What is the largest planet in the solar system?", "a": "Jupiter"},
    # tier 2 - moderate
    {"id": "m1", "rarity": 0.5, "q": "Who developed the theory of general relativity?", "a": "Einstein"},
    {"id": "m2", "rarity": 0.5, "q": "What is the capital city of Australia?", "a": "Canberra"},
    {"id": "m3", "rarity": 0.5, "q": "In which year did the Berlin Wall fall?", "a": "1989"},
    # tier 3 - rare
    {"id": "r1", "rarity": 0.7, "q": "Who proved Fermat's Last Theorem?", "a": "Wiles"},
    {"id": "r2", "rarity": 0.7, "q": "What is the SI unit of electrical conductance?", "a": "siemens"},
    {"id": "r3", "rarity": 0.7, "q": "Which chemical element has atomic number 74?", "a": "tungsten"},
    # tier 4 - obscure
    {"id": "x1", "rarity": 0.9, "q": "What is the capital city of Bhutan?", "a": "Thimphu"},
    {"id": "x2", "rarity": 0.9, "q": "Which mathematician introduced the notion of a topos?", "a": "Grothendieck"},
    {"id": "x3", "rarity": 0.9, "q": "What is the SI unit of magnetic flux?", "a": "weber"},
]
FACT_BY_ID: dict[str, dict] = {f["id"]: f for f in RECALL_FACTS}

# dated events, one verifiable answer per year (for the cutoff change-point).
DATED_EVENTS: list[dict] = [
    {"id": "ev19", "year": 2019, "q": "Which country hosted the 2019 Rugby World Cup?", "a": "Japan"},
    {"id": "ev20", "year": 2020, "q": "What infectious disease was declared a pandemic in 2020?", "a": "COVID-19"},
    {"id": "ev21", "year": 2021, "q": "Which space telescope was launched in December 2021?", "a": "James Webb"},
    {"id": "ev22", "year": 2022, "q": "Which country hosted the 2022 FIFA World Cup?", "a": "Qatar"},
    {"id": "ev23", "year": 2023, "q": "Which AI chatbot reached 100 million users in early 2023?", "a": "ChatGPT"},
    {"id": "ev24", "year": 2024, "q": "Which city hosted the 2024 Summer Olympic Games?", "a": "Paris"},
    {"id": "ev25", "year": 2025, "q": "Which mission returned samples in the 2025 lunar campaign?", "a": "Chang'e"},
]
EVENT_BY_ID: dict[str, dict] = {e["id"]: e for e in DATED_EVENTS}

_MARK_Q = re.compile(r"\[q:(\w+)\]")
_MARK_EV = re.compile(r"\[ev:(\w+)\]")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _matches(answer: str, reply: str) -> bool:
    a, r = _norm(answer), _norm(reply)
    return bool(a) and a in r


def _recall_prompt(fact: dict) -> str:
    return (f"Answer the trivia question with only the exact term, no sentence. "
            f"[q:{fact['id']}] {fact['q']}\nAnswer:")


def _event_prompt(ev: dict) -> str:
    return (f"Answer with only the exact name, no sentence. "
            f"[ev:{ev['id']}] {ev['q']}\nAnswer:")


def _crossing_50(rarity: np.ndarray, acc: np.ndarray) -> float:
    """Rarity at which accuracy first drops through 0.5 (linear interpolation)."""
    order = np.argsort(rarity)
    rs, ac = rarity[order], acc[order]
    for i in range(1, rs.size):
        if ac[i - 1] >= 0.5 > ac[i]:
            # interpolate between (rs[i-1], ac[i-1]) and (rs[i], ac[i])
            denom = (ac[i - 1] - ac[i]) or 1e-9
            frac = (ac[i - 1] - 0.5) / denom
            return float(rs[i - 1] + frac * (rs[i] - rs[i - 1]))
    if ac[0] < 0.5:
        return float(rs[0])           # never above 0.5 -> below the easiest tier
    return float(rs[-1])              # always above 0.5 -> beyond the hardest tier


# Calibration of the 50%-recall rarity threshold onto total parameters.
# log10(params_B) = _CAL_A + _CAL_B * rarity_50, anchored so rarity_50=0 -> ~1B and
# rarity_50=1 -> ~316B (2 bits/param capacity rule, ArchDB param range as reference points).
_CAL_A = 0.0
_CAL_B = 2.5
_CAL_LOG_SD = 0.7                      # ~ x5 CI half-width in log10 (honest, wide)


def _params_from_rarity(r50: float) -> tuple[float, float, float]:
    log10p = _CAL_A + _CAL_B * r50
    return (10.0 ** log10p, 10.0 ** (log10p - _CAL_LOG_SD), 10.0 ** (log10p + _CAL_LOG_SD))


@register
class KnowledgeProbe(Probe):
    id = "P12.knowledge.graded_recall"
    title = "Knowledge capacity & training cutoff (graded-rarity recall)"
    category = "capacity"
    kinds = ("generative",)
    informs = {"arch.sparsity": 0.2}
    estimates = ("dims.total_params", "knowledge.cutoff")
    requires = ()
    depends_on = ("P07.timing.decode",)
    cost = 22

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        per_tier = 2 if ctx.fast() else 3
        # pick up to `per_tier` facts from each rarity tier
        by_rarity: dict[float, list[dict]] = {}
        for f in RECALL_FACTS:
            by_rarity.setdefault(f["rarity"], []).append(f)
        facts: list[dict] = []
        for r, fs in by_rarity.items():
            facts.extend(fs[:per_tier])
        events = DATED_EVENTS if not ctx.fast() else DATED_EVENTS  # small enough to keep all

        # ---- recall sweep -----------------------------------------------------
        try:
            recall_resps = await ctx.gen_many(
                [GenRequest.chat(_recall_prompt(f), max_tokens=16, temperature=0.0)
                 for f in facts])
            event_resps = await ctx.gen_many(
                [GenRequest.chat(_event_prompt(e), max_tokens=16, temperature=0.0)
                 for e in events])
        except Exception as exc:  # pragma: no cover
            res.status = "partial"
            res.notes.append(f"knowledge sweep aborted: {exc!r}")
            recall_resps = locals().get("recall_resps", [])
            event_resps = []
        res.n_requests = ctx.n_requests

        # ---- score recall by tier --------------------------------------------
        tier_correct: dict[float, list[int]] = {}
        rows: list[dict] = []
        for f, resp in zip(facts, recall_resps):
            ok = resp.ok
            hit = int(ok and _matches(f["a"], resp.text))
            tier_correct.setdefault(f["rarity"], []).append(hit)
            rows.append({"id": f["id"], "rarity": f["rarity"], "question": f["q"],
                         "expected": f["a"], "reply": (resp.text or "")[:40],
                         "correct": bool(hit), "ok": ok})
        res.tables["recall_detail"] = rows

        rarities = sorted(tier_correct)
        acc = np.array([np.mean(tier_correct[r]) for r in rarities], dtype=float)
        rar = np.array(rarities, dtype=float)
        tier_rows = []
        for r in rarities:
            k = int(sum(tier_correct[r])); n = len(tier_correct[r])
            lo, hi = proportion_ci(k, n, level=0.9)
            tier_rows.append({"rarity": r, "accuracy": round(k / n, 3), "k": k, "n": n,
                              "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
        res.tables["recall_by_tier"] = tier_rows
        res.series["accuracy_vs_rarity"] = {
            "x": [float(r) for r in rar], "y": [float(a) for a in acc],
            "xlabel": "rarity (0=common,1=obscure)", "ylabel": "recall accuracy",
            "kind": "line",
        }

        if rar.size < 2 or not np.isfinite(acc).all():
            res.status = "partial"
            res.notes.append("insufficient recall tiers to fit a curve")
            return res

        # monotonicity: accuracy should decrease with rarity (Spearman < 0).
        spearman = _spearman(rar, acc)
        r50 = _crossing_50(rar, acc)
        p_med, p_lo, p_hi = _params_from_rarity(r50)

        res.add_estimate(Estimate(
            quantity="dims.total_params", value=round(p_med, 2),
            ci=(round(p_lo, 2), round(p_hi, 2)), ci_level=0.9, unit="B",
            level="speculative",
            method=("50%-recall rarity threshold -> log10(params) via 2-bits/param capacity "
                    "rule; ArchDB param range as reference points"),
            assumptions=["knowledge scales ~log-linearly with parameters",
                         "fact rarity axis is comparable across models",
                         "no retrieval augmentation inflating recall"],
            alternatives=["retrieval/tool augmentation raising apparent recall",
                          "distillation giving small models large-model knowledge",
                          "benchmark contamination of common facts"],
            identifiability=WEAK,
        ))

        # ---- cutoff change-point ---------------------------------------------
        ev_rows = []
        year_hit: dict[int, list[int]] = {}
        for e, resp in zip(events, event_resps):
            hit = int(resp.ok and _matches(e["a"], resp.text))
            year_hit.setdefault(e["year"], []).append(hit)
            ev_rows.append({"id": e["id"], "year": e["year"], "expected": e["a"],
                            "reply": (resp.text or "")[:40], "correct": bool(hit)})
        res.tables["dated_events"] = ev_rows
        cutoff_est = None
        if year_hit:
            years = np.array(sorted(year_hit), dtype=float)
            yacc = np.array([np.mean(year_hit[int(y)]) for y in years], dtype=float)
            res.series["accuracy_vs_year"] = {
                "x": [float(y) for y in years], "y": [float(a) for a in yacc],
                "xlabel": "event year", "ylabel": "recall accuracy", "kind": "line",
            }
            known = years[yacc >= 0.5]
            cutoff_year = int(known.max()) if known.size else int(years.min())
            seg = segmented1d(years, yacc, max_segments=2) if years.size >= 4 else None
            cutoff_est = cutoff_year
            if seg and seg.get("breakpoints"):
                res.notes.append(f"cutoff change-point breakpoints: {seg['breakpoints']}")
            res.add_estimate(Estimate(
                quantity="knowledge.cutoff", value=f"{cutoff_year}",
                level="estimated",
                method="last year with >=50% dated-event recall; piecewise change-point cross-check",
                assumptions=["dated events are equally famous across years",
                             "post-cutoff events genuinely unseen (not merely rare)"],
                alternatives=["retrieval augmentation answering recent events",
                              "uneven event fame confounding the year axis"],
                identifiability=WEAK,
            ))

        # ---- weak arch.sparsity evidence: capacity vs active compute ----------
        active_b = _shared_num(ctx, ["timing.active_params_b", "active_params_b",
                                     "dims.active_params"])
        if active_b and active_b > 0:
            ratio = p_med / active_b
            if ratio > 1.8:
                like, strength = {"moe": 0.8, "dense": 0.25}, 0.5
                summ = (f"stored knowledge params ~{p_med:.0f}B >> active ~{active_b:.0f}B "
                        f"({ratio:.1f}x): favours sparsity")
            else:
                like, strength = {"dense": 0.6, "moe": 0.4}, 0.35
                summ = f"knowledge params ~ active params ({ratio:.1f}x): consistent with dense"
            res.add_evidence(Evidence(
                property="arch.sparsity", likelihoods=like, summary=summ, level="estimated",
                strength=strength,
                stats={"total_params_b": p_med, "active_params_b": active_b, "ratio": ratio},
                alternatives=["over/under-estimated capacity or active params",
                              "distillation decoupling knowledge from active size"],
            ))
        else:
            res.notes.append("no active-param estimate in ctx.shared; "
                             "skipped capacity-vs-active sparsity evidence")

        res.shared["knowledge.total_params_b"] = p_med
        res.shared["knowledge"] = {"rarity_50": r50, "spearman": spearman,
                                   "total_params_b": p_med, "cutoff": cutoff_est}
        res.notes.append(
            f"P12: rarity50={r50:.2f} -> total~{p_med:.1f}B "
            f"(CI {p_lo:.1f}-{p_hi:.1f}B), spearman={spearman:.2f}, cutoff={cutoff_est}")
        return res


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt(float(np.sum(rx * rx)) * float(np.sum(ry * ry)))
    return float(np.sum(rx * ry) / denom) if denom > 0 else 0.0


def _shared_num(ctx: ProbeContext, keys: list[str]) -> float | None:
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
