"""P13 - Hidden reasoning / test-time compute.

Three questions about whether the endpoint spends *invisible* computation before it
answers:

1. **Are there hidden reasoning tokens?**  Read ``usage.reasoning_tokens`` directly when
   the API reports them; otherwise look for a latency-vs-visible-output *residual* (the
   response takes far longer than its visible token count can explain - time spent
   thinking).  ``reasoning.hidden_tokens_ratio`` = hidden tokens / visible tokens.

2. **Does effort scale with difficulty?**  We pose k-hop compositional tasks of increasing
   depth (chains of "X manages Y" relations, asking for the root).  Latency and/or hidden
   token counts that grow with depth are the signature of adaptive test-time compute; the
   depth at which accuracy falls below 50 % (a change-point) is ``reasoning.max_hop_depth``.

3. **Does ``reasoning_effort`` change computation?**  When the parameter is accepted we
   compare a minimal-effort call against the default and report the change.

Evidence informs ``system.hidden_reasoning`` (strongly) and ``post.reasoning_rl`` (weakly:
adaptive, depth-scaled hidden compute is the behavioural mark of reasoning-focused RL).

The k-hop task builder is exposed at module level so a simulator responder (in tests) can
answer with a controlled depth capability.
"""
from __future__ import annotations

import random
import re

import numpy as np

from forensics.probes.base import (
    DIRECT, Estimate, Evidence, Probe, ProbeContext, ProbeResult, STATISTICAL, WEAK,
    register,
)
from forensics.core.types import GenRequest
from forensics.stats import bootstrap_ci, segmented1d

_FLOOR = 0.03

_NAMES = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Heidi", "Ivan",
          "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Rupert", "Sybil", "Trent",
          "Victor", "Walter", "Yvonne", "Zara", "Quinn"]

_MARK = re.compile(r"\[khop d=(\d+) s=(\d+)\]")


def build_khop(depth: int, s: int) -> tuple[str, str]:
    """Deterministic k-hop composition task.  Returns ``(prompt, answer)``.

    A chain ``root -> ... -> leaf`` of ``depth`` "manages" links; the question gives the
    leaf and asks for the root, so the model must compose ``depth`` hops.  Deterministic in
    ``(depth, s)`` so a test responder can reconstruct the exact answer.
    """
    rng = random.Random(depth * 100_003 + s * 7 + 11)
    k = depth + 1
    chain = rng.sample(_NAMES, k)
    links = [f"{chain[i]} manages {chain[i + 1]}." for i in range(depth)]
    # present links in shuffled order so the chain must actually be composed
    order = list(range(depth))
    rng.shuffle(order)
    body = " ".join(links[i] for i in order)
    leaf = chain[-1]
    root = chain[0]
    prompt = (f"[khop d={depth} s={s}] {body} "
              f"Who is at the very top of the management chain above {leaf}? "
              f"Answer with one name only.\nAnswer:")
    return prompt, root


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]+", " ", (s or "").lower()).strip()


def _hit(answer: str, reply: str) -> bool:
    a = _norm(answer)
    return bool(a) and a in _norm(reply)


def _visible_and_reasoning(resp) -> tuple[int, int]:
    u = resp.usage
    reasoning = int(u.reasoning_tokens or 0) if u and u.reasoning_tokens is not None else 0
    comp = int(u.completion_tokens or 0) if u and u.completion_tokens is not None else 0
    if u and u.raw and isinstance(u.raw, dict) and u.raw.get("visible_tokens") is not None:
        visible = int(u.raw["visible_tokens"])
    else:
        visible = max(0, comp - reasoning)
    return visible, reasoning


def _ttft(resp) -> float | None:
    """Think-time proxy in SECONDS. Prefer server-side time (excludes network RTT and
    client queueing, which otherwise swamp the signal), else client TTFT."""
    t = resp.timing
    if t is None:
        return None
    if t.server_ms is not None:
        return float(t.server_ms) / 1000.0
    return t.ttft


@register
class ReasoningProbe(Probe):
    id = "P13.reasoning.test_time_compute"
    title = "Hidden reasoning tokens & test-time compute scaling"
    category = "serving"
    kinds = ("generative",)
    informs = {"system.hidden_reasoning": 0.85, "post.reasoning_rl": 0.4}
    estimates = ("reasoning.max_hop_depth", "reasoning.hidden_tokens_ratio")
    requires = ()
    cost = 20

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        depths = [1, 2, 3, 4, 5]
        n_inst = 2 if ctx.fast() else 4

        # ---- k-hop sweep ------------------------------------------------------
        specs: list[tuple[int, int, str]] = []   # (depth, seed, answer)
        reqs: list[GenRequest] = []
        for d in depths:
            for s in range(n_inst):
                prompt, ans = build_khop(d, s)
                specs.append((d, s, ans))
                reqs.append(GenRequest.chat(prompt, max_tokens=32, temperature=0.0))
        try:
            resps = await ctx.gen_many(reqs, concurrency=1)
        except Exception as exc:  # pragma: no cover
            res.status = "partial"
            res.notes.append(f"k-hop sweep aborted: {exc!r}")
            resps = locals().get("resps", [])
        res.n_requests = ctx.n_requests

        depth_hits: dict[int, list[int]] = {}
        depth_reason: dict[int, list[int]] = {}
        depth_lat: dict[int, list[float]] = {}
        ratios: list[float] = []
        any_reasoning = False
        reasoning_reported = False   # the API exposes a reasoning-token count at all
        server_timed = False
        detail: list[dict] = []
        for (d, s, ans), resp in zip(specs, resps):
            if not resp.ok:
                continue
            hit = int(_hit(ans, resp.text))
            visible, reasoning = _visible_and_reasoning(resp)
            if resp.usage and resp.usage.reasoning_tokens is not None:
                reasoning_reported = True
                any_reasoning = any_reasoning or reasoning > 0
            if resp.timing is not None and resp.timing.server_ms is not None:
                server_timed = True
            depth_hits.setdefault(d, []).append(hit)
            depth_reason.setdefault(d, []).append(reasoning)
            lat = _ttft(resp)
            if lat is not None:
                depth_lat.setdefault(d, []).append(float(lat))
            if visible > 0 and reasoning > 0:
                ratios.append(reasoning / visible)
            detail.append({"depth": d, "seed": s, "expected": ans,
                           "reply": (resp.text or "")[:24], "correct": bool(hit),
                           "visible_tokens": visible, "reasoning_tokens": reasoning})
        res.tables["khop_detail"] = detail

        if not depth_hits:
            res.status = "partial"
            res.notes.append("no successful k-hop responses")
            return res

        ds = np.array(sorted(depth_hits), dtype=float)
        acc = np.array([np.mean(depth_hits[int(d)]) for d in ds], dtype=float)
        res.series["accuracy_vs_depth"] = {
            "x": [float(d) for d in ds], "y": [float(a) for a in acc],
            "xlabel": "composition depth (hops)", "ylabel": "accuracy", "kind": "line",
        }
        reason_by_depth = [float(np.mean(depth_reason[int(d)])) if depth_reason.get(int(d)) else 0.0
                           for d in ds]
        res.series["reasoning_tokens_vs_depth"] = {
            "x": [float(d) for d in ds], "y": reason_by_depth,
            "xlabel": "composition depth (hops)", "ylabel": "hidden tokens", "kind": "line",
        }
        res.tables["depth_summary"] = [
            {"depth": int(d), "accuracy": round(float(a), 3),
             "mean_reasoning_tokens": round(rb, 2),
             "mean_ttft_s": (round(float(np.mean(depth_lat[int(d)])), 5)
                             if depth_lat.get(int(d)) else None)}
            for d, a, rb in zip(ds, acc, reason_by_depth)]

        # ---- max hop depth (accuracy < 50%) -----------------------------------
        solved = ds[acc >= 0.5]
        max_depth = int(solved.max()) if solved.size else 0
        if ds.size >= 4:
            seg = segmented1d(ds, acc, max_segments=2)
            if seg.get("breakpoints"):
                res.notes.append(f"accuracy-vs-depth change-point near hop "
                                 f"{seg['breakpoints'][0]:.1f}")
        res.add_estimate(Estimate(
            quantity="reasoning.max_hop_depth", value=max_depth, unit="hops",
            ci=(max(0, max_depth - 1), max_depth + 1), ci_level=0.9,
            level="estimated" if solved.size else "inferred",
            method="deepest k-hop composition solved at >=50% accuracy (change-point on acc vs depth)",
            assumptions=["k-hop chains are equally hard at fixed depth",
                         "no answer leakage / lookup shortcut"],
            alternatives=["shallow heuristic that fails past a fixed depth without reasoning"],
            identifiability=STATISTICAL,
        ))

        # ---- hidden reasoning detection ---------------------------------------
        ratio_pt = float(np.median(ratios)) if ratios else 0.0
        ratio_ci = bootstrap_ci(ratios, np.median, n_boot=1000, level=0.9,
                                rng=ctx.rng)[:2] if len(ratios) >= 2 else (ratio_pt, ratio_pt)

        # Decision order, strongest first:
        #  1. usage reports reasoning tokens > 0             -> observed YES
        #  2. usage exposes the field and it is always 0     -> observed NO
        #  3. no usage field: latency must grow SIGNIFICANTLY with depth (server-side time
        #     preferred) -> weak, low-strength evidence only. A raw max/min latency ratio is
        #     dominated by network jitter and by deeper prompts simply being longer.
        residual, trend_lo = _latency_trend(depth_lat)
        detected = any_reasoning or (
            not reasoning_reported and residual is not None and residual > 2.0
            and trend_lo is not None and trend_lo > 0)
        if any_reasoning:
            like_hr, strength_hr, lvl = {"yes": 0.95, "no": _FLOOR}, 0.9, "observed"
            summ = f"hidden reasoning tokens reported by the API (median ratio {ratio_pt:.2f} of visible)"
        elif reasoning_reported:
            like_hr, strength_hr, lvl = {"no": 0.95, "yes": 0.04}, 0.9, "observed"
            summ = "API reports reasoning_tokens = 0 on every request"
        elif residual is not None and residual > 2.0 and trend_lo is not None and trend_lo > 0:
            like_hr = {"yes": 0.7, "no": 0.35}
            strength_hr = 0.5 if server_timed else 0.3
            lvl = "estimated"
            summ = (f"{'server' if server_timed else 'client'} latency grows x{residual:.1f} "
                    f"with problem depth (trend CI > 0) with no reasoning-token field")
        else:
            like_hr, strength_hr, lvl = {"no": 0.6, "yes": 0.4}, 0.4, "estimated"
            summ = ("no reasoning-token field and no significant latency growth with depth"
                    + (f" (x{residual:.1f})" if residual else ""))
        res.add_evidence(Evidence(
            property="system.hidden_reasoning", likelihoods=like_hr, summary=summ,
            level=lvl, strength=strength_hr,
            stats={"test": "reasoning-token-usage", "hidden_ratio": ratio_pt,
                   "latency_growth": residual, "trend_slope_lo": trend_lo, "reasoning_field": reasoning_reported, "n_ratio": len(ratios)},
            alternatives=["long visible chain-of-thought (not hidden)",
                          "server-side tool calls adding latency",
                          "queueing latency mimicking think time"],
        ))

        res.add_estimate(Estimate(
            quantity="reasoning.hidden_tokens_ratio",
            value=round(ratio_pt, 3),
            ci=(round(float(ratio_ci[0]), 3), round(float(ratio_ci[1]), 3)),
            ci_level=0.9, unit="x",
            level="observed" if any_reasoning else "inferred",
            method=("mean usage.reasoning_tokens / visible_tokens" if any_reasoning
                    else "no reasoning tokens reported -> ratio 0 (latency-residual fallback)"),
            assumptions=["reported reasoning tokens are the billed hidden thinking"],
            alternatives=["provider folds reasoning into completion_tokens"],
            identifiability=DIRECT if any_reasoning else WEAK,
        ))

        # ---- post.reasoning_rl (weak) -----------------------------------------
        depth_scaled = _increases(ds, np.array(reason_by_depth)) if any_reasoning else False
        if detected and (depth_scaled or max_depth >= 3):
            like_rl, strength_rl = {"yes": 0.7, "no": 0.35}, 0.4
            s_rl = ("adaptive, depth-scaled hidden compute with real multi-hop capability: "
                    "behavioural mark of reasoning-focused RL")
        elif detected:
            like_rl, strength_rl = {"yes": 0.55, "no": 0.45}, 0.3
            s_rl = "hidden compute present but not clearly difficulty-adaptive"
        else:
            like_rl, strength_rl = {"no": 0.6, "yes": 0.4}, 0.3
            s_rl = "no hidden test-time compute detected"
        res.add_evidence(Evidence(
            property="post.reasoning_rl", likelihoods=like_rl, summary=s_rl,
            level="inferred", strength=strength_rl,
            stats={"depth_scaled": depth_scaled, "max_hop_depth": max_depth},
            alternatives=["strong base model without reasoning RL",
                          "prompted (not trained) step-by-step behaviour"],
        ))

        # ---- reasoning_effort knob (best effort) ------------------------------
        if getattr(ctx.caps, "reasoning_effort", False):
            await self._effort_check(ctx, res)

        res.shared["reasoning"] = {"hidden": bool(detected),
                                   "hidden_tokens_ratio": ratio_pt,
                                   "max_hop_depth": max_depth}
        res.notes.append(
            f"P13: hidden={detected} ratio={ratio_pt:.2f} max_hop_depth={max_depth} "
            f"residual={residual}")
        return res

    async def _effort_check(self, ctx: ProbeContext, res: ProbeResult) -> None:
        prompt, _ = build_khop(3, 99)
        try:
            lo = await ctx.gen(GenRequest.chat(prompt, max_tokens=32, temperature=0.0,
                                               reasoning_effort="minimal"))
            hi = await ctx.gen(GenRequest.chat(prompt, max_tokens=32, temperature=0.0,
                                               reasoning_effort="high"))
        except Exception:  # pragma: no cover
            return
        res.n_requests = ctx.n_requests
        _, r_lo = _visible_and_reasoning(lo)
        _, r_hi = _visible_and_reasoning(hi)
        res.tables["reasoning_effort"] = [
            {"effort": "minimal", "reasoning_tokens": r_lo,
             "server_s": _server_s(lo)},
            {"effort": "high", "reasoning_tokens": r_hi,
             "server_s": _server_s(hi)},
        ]
        if r_hi > r_lo * 1.2 + 1:
            res.notes.append("reasoning_effort changes hidden token budget (high > minimal)")
        else:
            res.notes.append("reasoning_effort accepted but did not change observed computation")


def _server_s(resp) -> float | None:
    t = resp.timing
    if t is None:
        return None
    if t.server_ms is not None:
        return round(float(t.server_ms) / 1000.0, 5)
    return round(float(t.t_end), 5) if t.t_end is not None else None


def _increases(x: np.ndarray, y: np.ndarray) -> bool:
    """True if y trends upward with x (positive slope over a non-trivial range)."""
    if x.size < 3 or np.ptp(y) <= 1e-9:
        return False
    slope = float(np.polyfit(x, y, 1)[0])
    return slope > 0 and (y[-1] - y[0]) > 0.5 * (np.mean(np.abs(np.diff(y))) + 1e-9)


def _latency_trend(depth_lat: dict) -> tuple[float | None, float | None]:
    """(growth ratio, lower CI of the Theil-Sen slope) of MEDIAN latency vs depth.

    Growth ratio = fitted latency at the deepest level / at the shallowest (not max/min of
    noisy means). The slope's lower confidence bound must exceed 0 for the trend to count.
    Returns (None, None) with fewer than 3 depths.
    """
    from forensics.stats.regression import theilsen
    depths = sorted(d for d in depth_lat if depth_lat[d])
    if len(depths) < 3:
        return None, None
    x = np.array(depths, dtype=float)
    y = np.array([np.median(depth_lat[d]) for d in depths], dtype=float)
    try:
        slope, intercept, slope_ci = theilsen(x, y)
    except Exception:  # noqa: BLE001
        return None, None
    lo_fit = intercept + slope * x[0]
    hi_fit = intercept + slope * x[-1]
    ratio = float(hi_fit / lo_fit) if lo_fit > 0 else None
    lo = float(slope_ci[0]) if slope_ci is not None else None
    return ratio, lo
