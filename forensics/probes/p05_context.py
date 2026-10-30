"""P05 — Context-window limits.

Recover the maximum accepted prompt length and maximum generation length of a black-box
generative endpoint, plus the endpoint's over-length *policy* (silent truncation vs an
explicit error).

Method (all observables come from ``ctx.gen`` responses — never timed locally):

* **max input tokens** — grow the prompt geometrically until the API rejects it
  (``resp.ok == False``), then binary-search the acceptance boundary.  Over-length errors
  are forensic gold: if the error text / ``error_body`` states the limit we read it off
  directly (``observed``); otherwise the largest accepted prompt is a lower-bound estimate.
* **max output tokens** — request escalating ``max_tokens`` with a prompt that induces a
  long reply.  The largest request that the endpoint fully satisfies, before the completion
  length *saturates* below the request (``finish_reason == "length"`` with
  ``completion_tokens < requested``), is the output cap (or, again, a parsed error limit).
* **truncation policy** — place a needle at the very *start* of an over-long prompt and ask
  about it: an error ⇒ explicit-error policy; a silent answer that recalls the head ⇒ silent
  truncation keeping the head; a silent answer that cannot ⇒ silent truncation dropping it.
"""
from __future__ import annotations

import re

from forensics.probes.base import (
    ASSUMPTION, DIRECT, STATISTICAL, WEAK, BudgetExhausted, Estimate, Evidence, Probe,
    ProbeContext, ProbeResult, register,
)
from forensics.core.types import GenRequest

# unit that tokenises to ~1 token on byte-level BPEs (o200k/cl100k); the x-axis always uses
# the *measured* prompt_tokens, so exact tokens-per-unit does not matter.
_UNIT = "lorem "
_NEEDLE = "The vault keyword is ZEBRA-{code}."
_NEEDLE_RE = re.compile(r"vault keyword is ZEBRA-(\w+)")

_LIMIT_KEYS = ("limit", "max_context", "max_input_tokens", "maximum_context_length",
               "max_tokens", "maximum")
_LIMIT_PATTERNS = [
    re.compile(r"maximum context length is\s*([\d,]+)", re.I),
    re.compile(r"context length of\s*([\d,]+)", re.I),
    re.compile(r"maximum(?:\s+\w+){0,3}\s+length\D{0,6}([\d,]+)\s*tokens", re.I),
    re.compile(r"(?:max(?:imum)?|limit)\D{0,12}([\d,]+)\s*tokens", re.I),
]


def _units(n: int) -> str:
    return _UNIT * max(1, int(n))


def _visible(resp) -> int:
    u = resp.usage
    if u.raw and isinstance(u.raw, dict) and u.raw.get("visible_tokens") is not None:
        return int(u.raw["visible_tokens"])
    comp = u.completion_tokens or 0
    return int(comp - (u.reasoning_tokens or 0))


# Wording that marks a stated limit as the OUTPUT (completion) cap rather than the context.
_OUTPUT_CAP_PATTERNS = [
    re.compile(r"at most\s*([\d,]+)\s*(?:completion|output)\s*tokens", re.I),
    re.compile(r"(?:max_tokens|max_completion_tokens|max_output_tokens)\D{0,40}"
               r"(?:less than or equal to|<=|at most|maximum(?: value)? (?:is|of))\s*([\d,]+)", re.I),
    re.compile(r"(?:completion|output) tokens?\D{0,20}(?:limit|maximum|max)\D{0,12}([\d,]+)", re.I),
]
_CONTEXT_PATTERNS = [
    re.compile(r"maximum context length is\s*([\d,]+)", re.I),
    re.compile(r"context length(?: of)?\s*(?:is\s*)?\(?([\d,]+)", re.I),
    re.compile(r"context window\D{0,20}([\d,]+)", re.I),
]


def _classify_limit(resp) -> tuple[int | None, int | None]:
    """From one error, return (context_limit, output_cap) — whichever the text states."""
    text = resp.error or ""
    ctx_lim = out_cap = None
    for pat in _CONTEXT_PATTERNS:
        m = pat.search(text)
        if m:
            ctx_lim = int(m.group(1).replace(",", ""))
            break
    for pat in _OUTPUT_CAP_PATTERNS:
        m = pat.search(text)
        if m:
            out_cap = int(m.group(1).replace(",", ""))
            break
    return ctx_lim, out_cap


def _parse_limit(resp) -> int | None:
    """Extract a stated numeric limit from an error response (error_body dict or text)."""
    body = resp.error_body
    if isinstance(body, dict):
        for k in _LIMIT_KEYS:
            v = body.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    text = resp.error or ""
    for pat in _LIMIT_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


@register
class ContextLimitsProbe(Probe):
    id = "P05.context.limits"
    title = "Context-window and output-length limits"
    category = "context"
    kinds = ("generative",)
    informs = {"pos.long_context_ext": 0.2}
    estimates = ("context.max_input_tokens", "context.max_output_tokens")
    baseline = True
    cost = 20

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        res = self.result()
        status = "ok"
        try:
            oracle = await self._oracle(ctx, res)
            in_info = await self._max_input(ctx, res, known=oracle["context"])
            out_info = await self._max_output(ctx, res, known=oracle["output"])
            pol = await self._truncation_policy(ctx, res, in_info)
        except BudgetExhausted as e:
            res.notes.append(f"budget exhausted: {e}")
            res.status = "partial"
            res.n_requests = ctx.n_requests
            return res

        # ---- input estimate --------------------------------------------------
        if in_info["limit"] is not None:
            res.add_estimate(Estimate(
                quantity="context.max_input_tokens", value=int(in_info["limit"]),
                ci=(float(in_info["limit"]), float(in_info["limit"])), level="observed",
                method="stated in over-length API error", identifiability=DIRECT,
                unit="tokens", assumptions=["provider error text reports the true limit"],
            ))
        elif in_info["last_ok"] is not None and in_info.get("capped"):
            res.add_estimate(Estimate(
                quantity="context.max_input_tokens", value=int(in_info["last_ok"]),
                ci=(float(in_info["last_ok"]), float("inf")), level="estimated",
                method="LOWER BOUND: largest accepted prompt (search capped / no rejection seen)",
                identifiability=WEAK, unit="tokens",
                assumptions=["the true limit is at least this; raise p05_max_probe_tokens "
                             "or drop --fast to search further"],
            ))
        elif in_info["last_ok"] is not None:
            hi = float(in_info["first_err"] or in_info["last_ok"])
            res.add_estimate(Estimate(
                quantity="context.max_input_tokens", value=int(in_info["last_ok"]),
                ci=(float(in_info["last_ok"]), hi), level="estimated",
                method="binary search on acceptance boundary (no limit in error text)",
                identifiability=WEAK, unit="tokens",
                alternatives=["intermediate proxy truncates before the true model limit"],
            ))

        # ---- output estimate -------------------------------------------------
        if out_info["limit"] is not None:
            res.add_estimate(Estimate(
                quantity="context.max_output_tokens", value=int(out_info["limit"]),
                ci=(float(out_info["limit"]), float(out_info["limit"])), level="observed",
                method="stated in max_tokens API error", identifiability=DIRECT, unit="tokens",
            ))
        elif out_info["cap"] is not None:
            res.add_estimate(Estimate(
                quantity="context.max_output_tokens", value=int(out_info["cap"]),
                ci=(float(out_info["cap_lo"]), float(out_info["cap"])), level="inferred",
                method="completion length saturates below request (finish_reason=length)",
                identifiability=STATISTICAL, unit="tokens",
                alternatives=["the model stopped for content reasons, not a hard cap"],
            ))
        elif out_info["max_req"] is not None:
            res.add_estimate(Estimate(
                quantity="context.max_output_tokens", value=int(out_info["max_req"]),
                ci=(float(out_info["max_seen"]), float("inf")), level="estimated",
                method="largest request satisfied (lower bound; no cap observed)",
                identifiability=WEAK, unit="tokens",
            ))

        # ---- weak long-context-extension hint --------------------------------
        big = in_info["limit"] or in_info["last_ok"]
        if big:
            ext = big > 32_768
            res.add_evidence(Evidence(
                property="pos.long_context_ext",
                likelihoods={"interpolation_or_yarn": 0.62 if ext else 0.4,
                             "none": 0.4 if ext else 0.6},
                summary=f"max input ~{big} tokens {'>' if ext else '<='} 32k",
                level="inferred", strength=0.25,
                stats={"max_input_tokens": int(big)},
                alternatives=["natively long-context pretraining (no post-hoc extension)"],
            ))

        res.notes.append(f"truncation policy: {pol}")
        res.tables["truncation"] = [{"policy": pol}]
        res.shared["context"] = {
            "max_input_tokens": in_info["limit"] or in_info["last_ok"],
            "max_output_tokens": out_info["limit"] or out_info["cap"] or out_info["max_req"],
            "truncation_policy": pol,
        }
        res.status = status
        res.n_requests = ctx.n_requests
        return res

    # ------------------------------------------------------------------ input
    async def _oracle(self, ctx: ProbeContext, res: ProbeResult) -> dict:
        """Zero-cost limit oracle: a tiny prompt with an absurd max_tokens. Most APIs reject
        it and STATE a limit — Azure OpenAI the completion cap ("supports at most 16384
        completion tokens"), vLLM/SGLang/MaaS the context ("maximum context length is
        131072"). Providers that silently clamp simply answer "OK" (a few tokens)."""
        resp = await ctx.gen(GenRequest.chat("Reply with the single word OK.",
                                             max_tokens=10_000_000, tag=self.id))
        ctx_lim = out_cap = None
        if not resp.ok:
            ctx_lim, out_cap = _classify_limit(resp)
        res.tables["limit_oracle"] = [{
            "accepted": resp.ok, "status": resp.status, "error": (resp.error or "")[:160],
            "context_limit": ctx_lim, "output_cap": out_cap,
            "interpretation": ("silently clamped (no limit stated)" if resp.ok else
                               "limit stated in error" if (ctx_lim or out_cap) else
                               "rejected without a parsable limit")}]
        return {"context": ctx_lim, "output": out_cap}

    async def _max_input(self, ctx: ProbeContext, res: ProbeResult,
                         known: int | None = None) -> dict:
        if known:
            res.tables["input_search"] = [{"phase": "oracle", "parsed_limit": known,
                                           "accepted": None}]
            return {"limit": known, "last_ok": None, "first_err": None}
        max_iters = 9 if ctx.fast() else 16
        # Growing past the limit costs real input tokens and can exhaust a deployment's
        # tokens-per-minute quota (HTTP 429 → long Retry-After stalls). Fast mode stops at a
        # lower bound; full mode searches up to ~1.1M tokens.
        token_cap = int(ctx.config.get("p05_max_probe_tokens",
                                       34_000 if ctx.fast() else 1_100_000))
        n = 64
        last_ok = None            # largest accepted prompt_tokens
        last_ok_units = None
        first_err = None          # smallest rejected prompt_tokens
        first_err_units = None
        limit = None
        capped = False
        rows = []
        for _ in range(max_iters):
            if last_ok is not None and last_ok * 4 > token_cap:
                rows.append({"phase": "stop", "units": n, "accepted": None,
                             "note": f"next step would exceed probe cap {token_cap} tokens"})
                res.notes.append(f"input search capped at {token_cap} tokens; "
                                 f"max_input_tokens >= {last_ok} (lower bound)")
                capped = True
                break
            resp = await ctx.gen(GenRequest.chat(_units(n), max_tokens=1, tag=self.id))
            if resp.status == 429:
                rows.append({"phase": "grow", "units": n, "accepted": None,
                             "error": "rate limited (429)"})
                res.notes.append("rate limited during context search; result is a lower bound")
                capped = True
                break
            if resp.ok:
                last_ok = resp.usage.prompt_tokens
                last_ok_units = n
                rows.append({"phase": "grow", "units": n,
                             "prompt_tokens": resp.usage.prompt_tokens, "accepted": True})
                n *= 4
            else:
                first_err = (resp.error_body or {}).get("requested") if isinstance(resp.error_body, dict) else None
                first_err_units = n
                limit = _parse_limit(resp)
                rows.append({"phase": "grow", "units": n, "accepted": False,
                             "error": (resp.error or "")[:80], "parsed_limit": limit})
                break
        # binary search the acceptance boundary in unit-space (tightens last_ok / first_err);
        # unnecessary when the rejection already stated the exact limit.
        if limit is None and last_ok_units is not None and first_err_units is not None:
            lo, hi = last_ok_units, first_err_units
            for _ in range(3 if ctx.fast() else 6):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2
                resp = await ctx.gen(GenRequest.chat(_units(mid), max_tokens=1, tag=self.id))
                if resp.ok:
                    lo = mid
                    last_ok = resp.usage.prompt_tokens
                    rows.append({"phase": "bisect", "units": mid,
                                 "prompt_tokens": resp.usage.prompt_tokens, "accepted": True})
                else:
                    hi = mid
                    limit = limit or _parse_limit(resp)
                    fe = (resp.error_body or {}).get("requested") if isinstance(resp.error_body, dict) else None
                    first_err = fe or first_err
                    rows.append({"phase": "bisect", "units": mid, "accepted": False})
        res.tables["input_search"] = rows
        res.series["input_search"] = {
            "x": [r.get("units") for r in rows],
            "y": [1 if r.get("accepted") else 0 for r in rows],
            "xlabel": "prompt units", "ylabel": "accepted", "kind": "scatter",
        }
        if limit is None and first_err is None and first_err_units is None:
            capped = True   # never observed a rejection: everything we saw is a lower bound
        return {"limit": limit, "last_ok": last_ok, "first_err": first_err, "capped": capped}

    # ----------------------------------------------------------------- output
    async def _max_output(self, ctx: ProbeContext, res: ProbeResult,
                          known: int | None = None) -> dict:
        if known:
            res.tables["output_search"] = [{"requested": None, "parsed_limit": known,
                                            "source": "oracle"}]
            return {"limit": known, "cap": None, "cap_lo": 0, "max_req": None,
                    "max_seen": 0}
        prompt = ("Write an extremely long, continuous response. Do not stop early; "
                  "keep going until you hit your own maximum output length.")
        rows = []
        cap = None
        cap_lo = 0
        limit = None
        max_req = None
        max_seen = 0
        R = 32
        # Generating is slow and billed per token: fast mode only establishes a lower
        # bound; full mode generates up to 16k tokens.
        hard = int(ctx.config.get("p05_max_gen_tokens", 1024 if ctx.fast() else 16_384))
        max_iters = 7 if ctx.fast() else 10
        for _ in range(max_iters):
            resp = await ctx.gen(GenRequest.chat(prompt, max_tokens=R, tag=self.id))
            if not resp.ok:
                ctx_lim, out_cap = _classify_limit(resp)
                # a CONTEXT-limit rejection (prompt + max_tokens too big) says nothing
                # about the output cap: keep what we observed as a lower bound
                limit = out_cap
                rows.append({"requested": R, "accepted": False,
                             "error": (resp.error or "")[:80], "parsed_output_cap": out_cap,
                             "parsed_context": ctx_lim})
                if limit is not None:
                    break
                if ctx_lim is not None:
                    break
                # rejected without a number: cap is between max_seen and R
                cap = cap or max_seen or None
                break
            vis = _visible(resp)
            rows.append({"requested": R, "visible": vis, "finish": resp.finish_reason})
            max_req = R
            max_seen = max(max_seen, vis)
            if resp.finish_reason == "length" and vis < R:
                cap = vis
                cap_lo = 0
                break
            if resp.finish_reason != "length":
                # model stopped for content reasons: not a hard cap, keep probing larger
                pass
            cap_lo = R
            if R >= hard:
                break
            R *= 4
        res.tables["output_search"] = rows
        res.series["output_search"] = {
            "x": [r.get("requested") for r in rows],
            "y": [r.get("visible", 0) for r in rows],
            "xlabel": "max_tokens requested", "ylabel": "completion tokens", "kind": "line",
        }
        return {"limit": limit, "cap": cap, "cap_lo": cap_lo,
                "max_req": max_req if cap is None and limit is None else None,
                "max_seen": max_seen}

    # ------------------------------------------------------------- policy
    async def _truncation_policy(self, ctx: ProbeContext, res: ProbeResult, in_info: dict) -> str:
        target = in_info["first_err"] or in_info["limit"] or in_info["last_ok"] or 4096
        n = int(target * 3) + 256          # comfortably over the limit
        code = "77042"
        head = _NEEDLE.format(code=code) + " "
        body = head + _units(n) + " Question: what is the vault keyword? Reply with it."
        resp = await ctx.gen(GenRequest.chat(body, max_tokens=16, tag=self.id))
        if not resp.ok:
            return "explicit_error"
        if code in (resp.text or ""):
            return "silent_truncation_keeps_head"
        return "silent_truncation_drops_head"
