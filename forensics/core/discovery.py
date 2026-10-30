"""Capability discovery: probe an adapter with a few cheap requests.

``discover(adapter, budget=None, recorder=None, fast=False)`` sends a small,
bounded set of minimal requests to a live endpoint and returns a filled
:class:`forensics.core.types.Capabilities`. It works generically over the
:class:`~forensics.core.adapters.base.Adapter` interface and branches on
``adapter.kind`` ("generative" vs "decision").

Design rules:

* It starts from ``adapter.static_capabilities()`` and *refines* by observation:
  a field only becomes ``True``/``False`` when a request proves it. A field left
  ``None`` means untested/unknown; ``False`` means tested-and-unsupported.
* Every check runs in its own ``try/except``; on failure the field is left
  ``None`` and a note is appended. One broken check never aborts the rest.
* Budget/recorder are honoured directly: before each request we consult
  ``budget.can_spend(req)``; after each we ``budget.charge(req, resp)`` and, if a
  recorder is given, ``recorder.record(...)``. The adapter's own recorder/budget
  hooks are temporarily detached so nothing is double-counted.
* Generative facts are learned by *sending*. The production
  ``OpenAICompatAdapter`` is *adaptive*: on a 400 it silently drops the offending
  parameter (or swaps the token-limit name, or caps ``top_logprobs``) and retries,
  so a plain ``resp.ok`` hides rejections. Discovery therefore also reads the
  adapter's learned per-model constraints (``dropped_params`` etc.) when the
  adapter exposes them, and falls back to raw ``resp.ok``/error text for faithful
  (non-adaptive) adapters such as the simulator.

Total logical probes per call: ~14 (fewer with ``fast=True``).
"""
from __future__ import annotations

from typing import Any

from forensics.core.adapters.base import Adapter
from forensics.core.budget import Budget
from forensics.core.recorder import Recorder
from forensics.core.types import (
    Capabilities, DecisionQuestion, DecisionRequest, GenRequest, GenResponse, Message,
)

# A one-word factual prompt whose natural first token is "Paris" (never " the").
_PROMPT = "Reply with exactly one word: the capital of France."
# o200k id for the token " the"; used only as an *identifiable* token to bias.
_BIAS_TOKEN_ID = 290
_BIAS_VALUE = 100.0
_TOP_LOGPROBS_TRY = 20

# Header names that commonly carry server-side processing time.
_SERVER_TIME_HEADERS: tuple[str, ...] = (
    "openai-processing-ms",
    "x-envoy-upstream-service-time",
    "x-processing-time",
    "x-response-time",
    "x-total-time-ms",
    "x-total-time",
    "server-timing",
)


# --------------------------------------------------------------------------- runner
class _Runner:
    """Sends requests through the adapter while honouring budget/recorder."""

    def __init__(self, adapter: Adapter, budget: Budget | None, recorder: Recorder | None):
        self.adapter = adapter
        self.budget = budget
        self.recorder = recorder
        self.sent = 0

    async def gen(self, req: GenRequest) -> GenResponse | None:
        if self.budget is not None and not self.budget.can_spend(req):
            return None
        resp = await self.adapter.generate(req)
        self.sent += 1
        if self.budget is not None:
            self.budget.charge(req, resp)
        if self.recorder is not None:
            self.recorder.record("generate", req, resp, probe="discovery")
        return resp

    async def decide(self, req: DecisionRequest):
        if self.budget is not None and not self.budget.can_spend(req):
            return None
        resp = await self.adapter.decide(req)
        self.sent += 1
        if self.budget is not None:
            self.budget.charge(req, resp)
        if self.recorder is not None:
            self.recorder.record("decide", req, resp, probe="discovery")
        return resp


# --------------------------------------------------------------------------- constraints
def _con(adapter: Adapter) -> Any | None:
    """The adaptive adapter's learned per-model constraints, if it exposes any."""
    cons = getattr(adapter, "_constraints", None)
    model = getattr(adapter, "model", None)
    if isinstance(cons, dict) and model is not None:
        return cons.get(model)
    return None


def _dropped(adapter: Adapter) -> set[str] | None:
    con = _con(adapter)
    if con is None:
        return None
    d = getattr(con, "dropped_params", None)
    return set(d) if d is not None else None


def _accepts(dropped: set[str] | None, param: str, observed_ok: bool | None) -> bool | None:
    """Was ``param`` accepted? Prefer the adapter's learned drop-set; else observation."""
    if dropped is not None:
        return param not in dropped
    return observed_ok


# --------------------------------------------------------------------------- helpers
def _server_timing_header(headers: dict[str, str] | None) -> str | None:
    low = {str(k).lower() for k in (headers or {})}
    for name in _SERVER_TIME_HEADERS:
        if name in low:
            return name
    return None


def _first_token(resp: GenResponse | None) -> tuple[str | None, float | None]:
    if resp is None or not resp.ok or not resp.tokens:
        return None, None
    t = resp.tokens[0]
    return t.token, t.logprob


def _looks_rejected(err: str | None, *needles: str) -> bool:
    e = (err or "").lower()
    return any(n in e for n in needles)


def _needs_token_param_swap(err: str | None) -> bool:
    """True if a 400 asks for max_completion_tokens instead of max_tokens."""
    e = (err or "").lower()
    if "max_completion_tokens" in e:
        return True
    return "max_tokens" in e and _looks_rejected(e, "unsupported", "not supported", "use")


# --------------------------------------------------------------------------- generative
async def _discover_generative(runner: _Runner, caps: Capabilities, fast: bool) -> None:
    ad = runner.adapter
    requested = str(getattr(ad, "model", getattr(ad, "name", "")) or "")

    # ---- 1. baseline chat: chat works + metadata (+ token-limit param via con) ----
    baseline: GenResponse | None = None
    try:
        req = GenRequest.chat(_PROMPT, max_tokens=8, temperature=None, tag="disc.baseline")
        resp = await runner.gen(req)
        # Faithful adapters need a manual token-param swap; adaptive ones self-heal.
        if (resp is not None and not resp.ok and hasattr(ad, "max_tokens_param")
                and _needs_token_param_swap(resp.error)):
            ad.max_tokens_param = "max_completion_tokens"  # type: ignore[attr-defined]
            resp = await runner.gen(GenRequest.chat(_PROMPT, max_tokens=8, temperature=None,
                                                    tag="disc.baseline.mct"))
        baseline = resp
        if resp is not None and resp.ok:
            caps.chat = True
        elif resp is not None:
            caps.notes.append(f"baseline chat failed: {resp.error}")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"baseline error: {e!r}")

    # token-limit param: learned constraint wins, else adapter attribute
    con = _con(ad)
    if con is not None and getattr(con, "max_tokens_param", None):
        caps.max_tokens_param = con.max_tokens_param
    elif hasattr(ad, "max_tokens_param"):
        caps.max_tokens_param = getattr(ad, "max_tokens_param")

    if baseline is not None and baseline.ok:
        try:
            caps.system_fingerprint = baseline.system_fingerprint is not None
            u = baseline.usage
            caps.usage_prompt_tokens = u.prompt_tokens is not None
            caps.usage_cached_tokens = u.cached_tokens is not None
            caps.usage_reasoning_tokens = u.reasoning_tokens is not None
            caps.server_timing_header = _server_timing_header(baseline.headers)
            if caps.server_timing_header is None and isinstance(u.raw, dict) \
                    and "latency_checkpoint" in u.raw:
                caps.notes.append("server timing in usage.raw['latency_checkpoint'] "
                                  "(no timing header)")
            mr = baseline.model_reported
            if mr:
                caps.notes.append(f"model_reported={mr}")
                if requested and requested.lower() not in mr.lower() \
                        and mr.lower() not in requested.lower():
                    caps.notes.append(f"aliasing: reported '{mr}' != requested '{requested}'")
        except Exception as e:  # noqa: BLE001
            caps.notes.append(f"metadata parse error: {e!r}")

    # ---- 2. logprobs + max top_logprobs + bytes/token_ids ----
    # Generous max_tokens: reasoning-capable models spend output budget before the
    # first visible (logprob-bearing) token, so a tiny cap 400s with "max_tokens
    # reached" and yields no sample.
    unbiased_ref: GenResponse | None = None
    try:
        req = GenRequest.chat(_PROMPT, max_tokens=24, temperature=None, logprobs=True,
                              top_logprobs=_TOP_LOGPROBS_TRY, tag="disc.logprobs")
        resp = await runner.gen(req)
        con = _con(ad)
        dropped = _dropped(ad)
        learned_cap = getattr(con, "max_top_logprobs", None) if con else None
        if resp is not None and resp.ok and resp.tokens:
            caps.logprobs = True
            unbiased_ref = resp
            observed = max((len(t.top) for t in resp.tokens if t.top), default=0)
            caps.max_top_logprobs = learned_cap or observed or _TOP_LOGPROBS_TRY
            caps.logprob_bytes = any(t.bytes for t in resp.tokens) or any(
                tt.bytes for t in resp.tokens for tt in t.top)
            caps.token_ids_known = any(t.token_id is not None for t in resp.tokens) or any(
                tt.token_id is not None for t in resp.tokens for tt in t.top)
        elif learned_cap:
            # The adapter parsed a "top_logprobs must be <= N" cap even though the
            # final attempt failed for an unrelated reason (e.g. output-limit).
            caps.logprobs = True
            caps.max_top_logprobs = learned_cap
            caps.notes.append(
                f"logprobs supported (cap {learned_cap} learned); "
                f"no clean sample: {(resp.error or '')[:100] if resp else 'skipped'}")
        elif dropped is not None and "logprobs" in dropped:
            caps.logprobs = False
            caps.notes.append("logprobs unsupported (adapter dropped logprobs)")
        elif resp is not None and not resp.ok and _looks_rejected(resp.error, "logprob") \
                and _looks_rejected(resp.error, "not supported", "unsupported"):
            caps.logprobs = False
            caps.notes.append(f"logprobs rejected: {(resp.error or '')[:120]}")
        elif resp is not None:
            caps.notes.append(f"logprobs inconclusive (ok={resp.ok}, tokens={bool(resp.tokens)})")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"logprobs error: {e!r}")

    # ---- 3. logit_bias accepted? + does it move the returned logprobs? ----
    try:
        cap = caps.max_top_logprobs or 5
        breq = GenRequest.chat(
            _PROMPT, max_tokens=2, temperature=None,
            logprobs=bool(caps.logprobs), top_logprobs=(cap if caps.logprobs else None),
            logit_bias={_BIAS_TOKEN_ID: _BIAS_VALUE}, tag="disc.logitbias")
        bresp = await runner.gen(breq)
        dropped = _dropped(ad)
        accepted = _accepts(dropped, "logit_bias",
                            bool(bresp.ok) if bresp is not None else None)
        if bresp is None:
            pass
        elif accepted is False:
            caps.logit_bias = False
            caps.notes.append("logit_bias rejected (adapter dropped it)"
                              if dropped is not None
                              else f"logit_bias rejected: {(bresp.error or '')[:120]}")
        elif accepted:
            caps.logit_bias = True
            caps.logit_bias_max = _BIAS_VALUE
            # Affects test: a huge +bias forces one token. A *post-bias* stack reports
            # that forced token at logprob ~0; a *pre-bias* stack reports it far below 0
            # (gpt-4o uses a -9999 sentinel). This decides which extraction attacks work.
            if caps.logprobs and bresp.tokens:
                b_tok, b_lp = _first_token(bresp)
                u_tok, _u_lp = _first_token(unbiased_ref)
                if b_tok is not None and u_tok is not None and b_tok == u_tok:
                    caps.logit_bias_affects_logprobs = False
                    caps.notes.append("logit_bias accepted but emitted token unchanged")
                elif b_lp is not None:
                    affects = b_lp > -0.5
                    caps.logit_bias_affects_logprobs = affects
                    caps.notes.append(
                        f"logit_bias affects_logprobs={affects} "
                        f"(forced-token logprob={b_lp:.3f})")
                else:
                    caps.notes.append("logit_bias affects test inconclusive (no logprob)")
            else:
                caps.notes.append("logit_bias accepted; affects untested (no logprobs)")
        else:
            caps.notes.append("logit_bias acceptance inconclusive")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"logit_bias error: {e!r}")

    # ---- 4. sampling params: temperature / top_p / seed / n ----
    try:
        req = GenRequest.chat(_PROMPT, max_tokens=2, temperature=0.7, top_p=0.9,
                              seed=7, n=2, tag="disc.sampling")
        resp = await runner.gen(req)
        dropped = _dropped(ad)
        ok = bool(resp.ok) if resp is not None else None
        if resp is None:
            pass
        elif dropped is not None:
            caps.temperature = "temperature" not in dropped
            caps.top_p = "top_p" not in dropped
            caps.seed = "seed" not in dropped
            n_ok = "n" not in dropped
            caps.n = n_ok and (len(resp.choices_text) >= 2 if resp.ok else True)
        elif ok:  # faithful adapter accepted the whole bundle
            caps.temperature = caps.top_p = caps.seed = True
            caps.n = len(resp.choices_text) >= 2
        else:  # faithful adapter rejected: attribute from the error text
            for p in ("temperature", "top_p", "seed", "n"):
                if p in (resp.error or "").lower():
                    setattr(caps, p, False)
            caps.notes.append(f"sampling bundle rejected: {(resp.error or '')[:100]}")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"sampling error: {e!r}")

    # ---- 5. system role ----
    try:
        resp = await runner.gen(GenRequest.chat(
            _PROMPT, system="You are a helpful assistant.", max_tokens=5,
            temperature=None, tag="disc.system"))
        if resp is not None:
            caps.system_role = bool(resp.ok)
            if not resp.ok:
                caps.notes.append(f"system role rejected: {(resp.error or '')[:100]}")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"system role error: {e!r}")

    # ---- 6. streaming + stream usage ----
    try:
        resp = await runner.gen(GenRequest.chat(
            _PROMPT, max_tokens=8, temperature=None, stream=True, tag="disc.stream"))
        if resp is not None:
            got_chunks = bool(resp.timing and resp.timing.chunks)
            caps.streaming = bool(resp.ok and (got_chunks or resp.text))
            caps.stream_usage = (resp.usage.prompt_tokens is not None) if resp.ok else None
            if not resp.ok:
                caps.notes.append(f"streaming failed: {(resp.error or '')[:100]}")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"streaming error: {e!r}")

    # ---- 7. reasoning_effort (+ supported values) ----
    try:
        resp = await runner.gen(GenRequest.chat(
            _PROMPT, max_tokens=16, temperature=None, reasoning_effort="low",
            tag="disc.reffort"))
        con = _con(ad)
        dropped = _dropped(ad)
        allowed = getattr(con, "allowed_reasoning_efforts", None) if con else None
        if resp is None:
            pass
        elif dropped is not None and "reasoning_effort" in dropped:
            caps.reasoning_effort = False
        elif allowed:
            caps.reasoning_effort = True
            caps.notes.append(f"reasoning_effort values: {allowed}")
        elif resp.ok:
            caps.reasoning_effort = True
        elif _looks_rejected(resp.error, "unrecognized", "unsupported", "unknown",
                             "not supported", "not permitted"):
            caps.reasoning_effort = False
        else:
            caps.notes.append(f"reasoning_effort inconclusive: {(resp.error or '')[:100]}")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"reasoning_effort error: {e!r}")

    if fast:
        return

    # ---- 8. assistant prefill (heuristic: trailing assistant message) ----
    try:
        msgs = [Message("user", "Complete the sentence."),
                Message("assistant", "The capital of France is")]
        resp = await runner.gen(GenRequest(messages=msgs, max_tokens=5, temperature=None,
                                           tag="disc.prefill"))
        if resp is not None:
            caps.assistant_prefill = bool(resp.ok and resp.text)
            caps.notes.append("assistant_prefill is heuristic (trailing assistant message)")
            if not resp.ok:
                caps.notes.append(f"assistant prefill rejected: {(resp.error or '')[:80]}")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"assistant prefill error: {e!r}")

    caps.notes.append("completion endpoint not separately probed "
                      "(adapter routes raw prompts through /chat/completions)")


# --------------------------------------------------------------------------- decision
async def _discover_decision(runner: _Runner, caps: Capabilities, fast: bool) -> None:
    caps.kind = "decision"
    for f in ("chat", "completion", "logprobs", "logit_bias", "streaming",
              "temperature", "top_p", "seed", "n", "system_role"):
        setattr(caps, f, False)
    caps.notes.append("decision model: no sampling / logprob / streaming controls")

    try:
        q = DecisionQuestion(type="choice",
                             instructions="Which team should handle this ticket?",
                             criteria={"billing": "Payments, charges, refunds.",
                                       "technical": "Login, bugs, product not working.",
                                       "sales": "Upgrades, pricing, new purchases."})
        req = DecisionRequest(state="You charged my card twice and I can't log in.",
                              questions={"route": q}, tag="disc.decide")
        resp = await runner.decide(req)
        if resp is None:
            caps.notes.append("decision probe skipped (budget)")
            return
        if resp.ok:
            caps.usage_prompt_tokens = resp.usage.prompt_tokens is not None
            caps.usage_cached_tokens = resp.usage.cached_tokens is not None
            caps.usage_reasoning_tokens = resp.usage.reasoning_tokens is not None
            hdr = _server_timing_header(resp.headers)
            if hdr:
                caps.server_timing_header = hdr
            elif resp.timing.server_ms is not None:
                # server time surfaced (parsed into timing) even if we didn't match a name
                caps.server_timing_header = caps.server_timing_header \
                    or "x-envoy-upstream-service-time"
            mr = resp.model_reported
            if mr:
                caps.notes.append(f"model_reported={mr}")
            ans = resp.answers.get("route")
            if ans is not None:
                caps.notes.append(
                    f"choice answer='{ans.value}' conf={ans.confidence} "
                    f"probs={ans.probabilities}")
        else:
            caps.notes.append(f"decision probe failed: {(resp.error or '')[:140]}")
    except Exception as e:  # noqa: BLE001
        caps.notes.append(f"decision probe error: {e!r}")


# --------------------------------------------------------------------------- entry
async def discover(
    adapter: Adapter,
    budget: Budget | None = None,
    recorder: Recorder | None = None,
    fast: bool = False,
) -> Capabilities:
    """Probe ``adapter`` with a few cheap live requests; return filled Capabilities.

    Generic over the Adapter interface; branches on ``adapter.kind``. Each check is
    isolated (failures leave the field ``None`` with an explanatory note). Budget
    and recorder are consulted directly, and the adapter's own budget/recorder
    hooks are detached during discovery so nothing is double-counted.
    """
    try:
        caps = adapter.static_capabilities()
    except Exception:  # noqa: BLE001
        caps = Capabilities(kind=getattr(adapter, "kind", "unknown"))  # type: ignore[arg-type]
    caps.kind = getattr(adapter, "kind", caps.kind)  # type: ignore[assignment]

    saved_rec, saved_bud = adapter.recorder, adapter.budget
    adapter.recorder = None
    adapter.budget = None
    runner = _Runner(adapter, budget, recorder)
    try:
        if caps.kind == "decision":
            await _discover_decision(runner, caps, fast)
        else:
            await _discover_generative(runner, caps, fast)
    finally:
        adapter.recorder, adapter.budget = saved_rec, saved_bud

    caps.notes.append(f"discovery sent {runner.sent} logical probes (fast={fast})")
    return caps
