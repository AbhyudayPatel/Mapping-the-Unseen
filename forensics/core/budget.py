"""Budget: request / token / cost accounting and gating for a forensic run.

A :class:`Budget` is attached to the engine and consulted by every probe through
``ProbeContext``:

* :meth:`can_spend` is called *before* a request. Tokens can't be predicted
  exactly, so gating is on hard ceilings (requests, cost, current output tokens)
  plus a *soft* input-token estimate derived from the request.
* :meth:`charge` is called *after* a response, updating counters from the
  response ``usage`` (handling both ``prompt_tokens``/``completion_tokens`` and
  ``input_tokens``/``output_tokens`` naming, plus ``usage.raw`` fallbacks).
* :meth:`summary` reports totals, estimated cost and what remains.

All mutating operations take a lock, so the budget is safe under the bounded
concurrency used by ``ctx.gen_many`` / ``ctx.decide_many``.
"""
from __future__ import annotations

import threading
from typing import Any

# Rough chars-per-token used for the soft pre-request input estimate. Real token
# counts come from the response usage; this only needs to be order-of-magnitude.
_CHARS_PER_TOKEN = 4.0


def _usage_tokens(resp: Any) -> tuple[int | None, int | None]:
    """Extract (input_tokens, output_tokens) from a response's usage.

    Accepts the unified ``Usage`` dataclass (``prompt_tokens`` /
    ``completion_tokens``), alternate ``input_tokens`` / ``output_tokens`` naming
    and a ``usage.raw`` dict fallback, for both GenResponse and DecisionResponse.
    """
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return None, None

    def get(name: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    raw = get("raw")
    raw = raw if isinstance(raw, dict) else {}

    inp = get("prompt_tokens")
    if inp is None:
        inp = get("input_tokens")
    if inp is None:
        inp = raw.get("prompt_tokens", raw.get("input_tokens"))

    out = get("completion_tokens")
    if out is None:
        out = get("output_tokens")
    if out is None:
        out = raw.get("completion_tokens", raw.get("output_tokens"))

    return inp, out


def _estimate_input_tokens(req: Any) -> int:
    """Soft estimate of a request's input token count from its text length."""
    if req is None:
        return 0
    chars = 0
    messages = getattr(req, "messages", None)
    if messages:
        for m in messages:
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            if content:
                chars += len(str(content))
    prompt = getattr(req, "prompt", None)
    if prompt:
        chars += len(str(prompt))
    # Decision requests carry a state (+ questions) instead of chat messages.
    state = getattr(req, "state", None)
    if state:
        chars += len(str(state))
    questions = getattr(req, "questions", None)
    if questions:
        chars += len(str(questions))
    return int(chars / _CHARS_PER_TOKEN)


class Budget:
    """Request/token/cost ceilings with thread/async-safe accounting."""

    def __init__(
        self,
        max_requests: int | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_cost_usd: float | None = None,
        price_in_per_m: float = 0.0,
        price_out_per_m: float = 0.0,
    ) -> None:
        self.max_requests = max_requests
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_cost_usd = max_cost_usd
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m

        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cost
    def _est_cost(self) -> float:
        return (
            self.input_tokens / 1_000_000.0 * self.price_in_per_m
            + self.output_tokens / 1_000_000.0 * self.price_out_per_m
        )

    def est_cost_usd(self) -> float:
        """Estimated spend so far (USD) from token counts and per-million prices."""
        with self._lock:
            return self._est_cost()

    # ------------------------------------------------------------------ gating
    def can_spend(self, req: Any = None) -> bool:
        """Return True if at least one more request fits within the ceilings.

        Gates on the request ceiling, the cost ceiling and the current output
        token count (all hard), plus a *soft* input-token estimate from ``req``.
        """
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                return False
            if self.max_cost_usd is not None and self._est_cost() >= self.max_cost_usd:
                return False
            if self.max_output_tokens is not None and self.output_tokens >= self.max_output_tokens:
                return False
            if self.max_input_tokens is not None:
                projected = self.input_tokens + _estimate_input_tokens(req)
                if projected > self.max_input_tokens:
                    return False
            return True

    # ------------------------------------------------------------------ charge
    def charge(self, req: Any, resp: Any) -> None:
        """Record one completed request, adding token usage from ``resp``."""
        inp, out = _usage_tokens(resp)
        with self._lock:
            self.requests += 1
            self.input_tokens += int(inp or 0)
            self.output_tokens += int(out or 0)

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        """Totals, estimated cost and remaining headroom (None where uncapped)."""
        with self._lock:
            remaining = {
                "requests": (None if self.max_requests is None
                             else max(0, self.max_requests - self.requests)),
                "input_tokens": (None if self.max_input_tokens is None
                                 else max(0, self.max_input_tokens - self.input_tokens)),
                "output_tokens": (None if self.max_output_tokens is None
                                  else max(0, self.max_output_tokens - self.output_tokens)),
                "cost_usd": (None if self.max_cost_usd is None
                             else max(0.0, self.max_cost_usd - self._est_cost())),
            }
            return {
                "requests": self.requests,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "est_cost_usd": self._est_cost(),
                "remaining": remaining,
            }
