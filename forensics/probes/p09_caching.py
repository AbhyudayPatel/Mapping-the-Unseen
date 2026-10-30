"""P09 — Prompt / prefix caching (Gu et al.-style timing audit).

Serving stacks that cache KV for shared prefixes answer a *repeated* long prefix far faster
than its first occurrence, and often report ``usage.cached_tokens`` on the hit.  We:

* send a long fixed prefix twice (warm, then repeat) and a fresh **novel** prefix of equal
  length as a control — a repeat that is both faster than its own first call *and* faster
  than the novel control (or that reports cached tokens) is a cache hit, not warm-up;
* vary the shared-prefix length and read the granularity of ``cached_tokens`` (the GCD of
  observed cached counts) to estimate ``serving.cache_block_tokens``.

⇒ ``serving.prompt_caching``; estimate ``serving.cache_block_tokens``.
"""
from __future__ import annotations

import math

import numpy as np

from forensics.probes.base import (
    DIRECT, STATISTICAL, WEAK, BudgetExhausted, Estimate, Evidence, Probe, ProbeContext,
    ProbeResult, register,
)
from forensics.core.types import GenRequest


def _server_s(resp) -> float | None:
    ms = resp.timing.server_ms
    return None if ms is None else float(ms) / 1000.0


def _floor(v: float) -> float:
    return max(0.02, min(1.0, float(v)))


@register
class PromptCacheProbe(Probe):
    id = "P09.serving.prompt_cache"
    title = "Prompt/prefix caching and cache-block granularity"
    category = "serving"
    kinds = ("generative",)
    informs = {"serving.prompt_caching": 0.85}
    estimates = ("serving.cache_block_tokens",)
    cost = 12

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        res = self.result()
        rows = []
        cached_vals: list[int] = []
        detected = False
        speedup = 1.0
        cached_seen = None
        try:
            # ---- main warm/repeat + novel control ---------------------------
            prefix = self._prefix(0, 90)
            first = await ctx.gen(GenRequest.chat(prefix, max_tokens=1, tag=self.id))
            repeat = await ctx.gen(GenRequest.chat(prefix, max_tokens=1, tag=self.id))
            novel = await ctx.gen(GenRequest.chat(self._prefix(999, 90), max_tokens=1, tag=self.id))

            s1, s2, sn = _server_s(first), _server_s(repeat), _server_s(novel)
            c2 = repeat.usage.cached_tokens
            cached_seen = c2
            rows.append({"call": "first", "cached": first.usage.cached_tokens,
                         "server_ms": None if s1 is None else s1 * 1000})
            rows.append({"call": "repeat", "cached": c2,
                         "server_ms": None if s2 is None else s2 * 1000})
            rows.append({"call": "novel", "cached": novel.usage.cached_tokens,
                         "server_ms": None if sn is None else sn * 1000})

            if s1 and s2:
                speedup = s1 / s2 if s2 > 0 else 1.0
            faster_than_self = bool(s1 and s2 and s2 < 0.85 * s1)
            faster_than_novel = bool(sn and s2 and s2 < 0.9 * sn)
            has_cached = bool(c2)
            detected = has_cached or (faster_than_self and faster_than_novel)
            if c2:
                cached_vals.append(int(c2))

            # ---- block-size sweep (distinct prefixes, warm then resend) -----
            if detected:
                n_len = 3 if ctx.fast() else 4
                for i, words in enumerate(np.linspace(45, 130, n_len).astype(int)):
                    pf = self._prefix(i + 1, int(words))
                    await ctx.gen(GenRequest.chat(pf, max_tokens=1, tag=self.id))
                    rr = await ctx.gen(GenRequest.chat(pf, max_tokens=1, tag=self.id))
                    cv = rr.usage.cached_tokens
                    rows.append({"call": f"blk[{words}]", "cached": cv,
                                 "prompt_tokens": rr.usage.prompt_tokens})
                    if cv:
                        cached_vals.append(int(cv))
        except BudgetExhausted as e:
            res.notes.append(f"budget exhausted: {e}")
            res.status = "partial"

        res.tables["cache"] = rows

        # ---- evidence -------------------------------------------------------
        if detected:
            yes_l = 1.0
            no_l = 0.1 if cached_seen else _floor(min(0.8, 1.0 / max(speedup, 1.0)))
            summ = f"repeat prefix {speedup:.2f}x faster" + (
                f", cached_tokens={cached_seen}" if cached_seen else "")
        else:
            yes_l, no_l = 0.3, 1.0
            summ = "no repeat speed-up and no cached_tokens"
        res.add_evidence(Evidence(
            property="serving.prompt_caching", likelihoods={"yes": _floor(yes_l),
                                                            "no": _floor(no_l)},
            summary=summ, level="inferred", strength=0.8,
            stats={"speedup": speedup, "cached_tokens": cached_seen, "detected": bool(detected),
                   "test": "warm/repeat vs novel control"},
            alternatives=["provider-side warm-up unrelated to prefix reuse",
                          "load variation between the two calls"]))

        # ---- block-size estimate -------------------------------------------
        if detected and cached_vals:
            blk = 0
            for v in cached_vals:
                blk = math.gcd(blk, v)
            if blk > 0:
                res.add_estimate(Estimate(
                    quantity="serving.cache_block_tokens", value=int(blk),
                    level="observed" if cached_seen else "inferred",
                    method="GCD of reported cached_tokens across prefix lengths",
                    identifiability=DIRECT if cached_seen else STATISTICAL, unit="tokens",
                    alternatives=["cached counts rounded by the API mask the true block size"]))

        res.shared["prompt_cache"] = {"detected": bool(detected), "speedup": speedup,
                                      "cache_block_tokens": (
                                          math.gcd(*cached_vals) if len(cached_vals) > 1
                                          else (cached_vals[0] if cached_vals else None))}
        res.n_requests = ctx.n_requests
        return res

    @staticmethod
    def _prefix(seed: int, words: int) -> str:
        # deterministic, content-distinct filler so different prefixes never share cache blocks
        return f"DOC{seed}: " + (f"section{seed} clause alpha beta gamma delta epsilon " * words)
