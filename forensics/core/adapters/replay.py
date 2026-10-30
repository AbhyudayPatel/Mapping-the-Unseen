"""Replay adapter: re-run a whole forensic run offline, with zero network.

A :class:`~forensics.core.recorder.Recorder` writes one JSON line per request to
``<run_dir>/observations.jsonl``::

    {"ts", "kind", "probe", "request": <jsonable>, "response": <jsonable>}

:class:`ReplayAdapter` loads that log and serves the recorded responses back: an
incoming request is matched to a recorded one by ``(kind, normalized-hash of the
request fields)`` and the stored response is reconstructed into the real
:class:`~forensics.core.types.GenResponse` / :class:`DecisionResponse` dataclasses
(including nested usage / timing / per-token logprobs).  This lets every probe and
the whole inference pipeline be re-analysed deterministically after the fact, with
no live endpoint and no cost.

Matching normalises away the volatile ``tag`` label (a recording annotation, not a
semantic request field); everything else in the request participates in the hash,
so identical requests map to identical responses.  When several identical requests
were recorded, they are replayed back in recorded order (the last is repeated once
exhausted).  A request with no recorded match returns a response whose ``error`` is
``"no recorded response"`` — or, when the adapter is constructed with
``strict=True``, raises :class:`NoRecordedResponse`.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from forensics.core.adapters.base import Adapter
from forensics.core.recorder import Recorder
from forensics.core.types import (
    Capabilities, Chunk, DecisionAnswer, DecisionRequest, DecisionResponse,
    GenRequest, GenResponse, Timing, TokenLogprob, TopToken, Usage, to_jsonable,
)

NO_MATCH = "no recorded response"


class NoRecordedResponse(LookupError):
    """Raised by a ``strict`` :class:`ReplayAdapter` when a request is not in the log."""


# --------------------------------------------------------------------------- hashing

def _normalize_request(req: Any) -> dict[str, Any]:
    """Drop volatile/label fields and return the jsonable request used for matching."""
    d = to_jsonable(req)
    if isinstance(d, dict):
        d = {k: v for k, v in d.items() if k != "tag"}
    return d


def request_hash(kind: str, req: Any) -> str:
    """Stable hash of ``(kind, normalized request)``; identical requests collide."""
    norm = _normalize_request(req)
    payload = json.dumps({"kind": kind, "req": norm},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- coercion

def _num(v: Any) -> Any:
    """Coerce inf/nan sentinel strings (as written by ``to_jsonable``) back to floats."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("inf", "+inf", "infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
        if s == "nan":
            return float("nan")
    return v


def _usage_from_dict(d: Any) -> Usage:
    if not isinstance(d, dict):
        return Usage()
    return Usage(
        prompt_tokens=d.get("prompt_tokens"),
        completion_tokens=d.get("completion_tokens"),
        total_tokens=d.get("total_tokens"),
        cached_tokens=d.get("cached_tokens"),
        reasoning_tokens=d.get("reasoning_tokens"),
        raw=d.get("raw"),
    )


def _timing_from_dict(d: Any) -> Timing:
    if not isinstance(d, dict):
        return Timing()
    chunks = [
        Chunk(t=float(_num(c.get("t", 0.0)) or 0.0), text=c.get("text", ""),
              n_tokens=c.get("n_tokens"))
        for c in d.get("chunks", []) if isinstance(c, dict)
    ]
    return Timing(
        t_first_byte=_num(d.get("t_first_byte")),
        t_first_token=_num(d.get("t_first_token")),
        t_end=_num(d.get("t_end")),
        chunks=chunks,
        server_ms=_num(d.get("server_ms")),
        wall_start=_num(d.get("wall_start")),
    )


def _top_from_dict(d: Any) -> TopToken:
    return TopToken(
        token=d.get("token", ""),
        logprob=float(_num(d.get("logprob", 0.0)) or 0.0),
        bytes=d.get("bytes"),
        token_id=d.get("token_id"),
    )


def _token_from_dict(d: Any) -> TokenLogprob:
    return TokenLogprob(
        token=d.get("token", ""),
        logprob=float(_num(d.get("logprob", 0.0)) or 0.0),
        bytes=d.get("bytes"),
        token_id=d.get("token_id"),
        top=[_top_from_dict(t) for t in d.get("top", []) if isinstance(t, dict)],
    )


def gen_response_from_dict(d: dict[str, Any]) -> GenResponse:
    """Reconstruct a :class:`GenResponse` from its ``to_jsonable`` dict form."""
    tokens = d.get("tokens")
    return GenResponse(
        text=d.get("text", ""),
        choices_text=list(d.get("choices_text", []) or []),
        tokens=([_token_from_dict(t) for t in tokens if isinstance(t, dict)]
                if isinstance(tokens, list) else None),
        finish_reason=d.get("finish_reason"),
        usage=_usage_from_dict(d.get("usage")),
        model_reported=d.get("model_reported"),
        system_fingerprint=d.get("system_fingerprint"),
        headers=dict(d.get("headers", {}) or {}),
        timing=_timing_from_dict(d.get("timing")),
        status=d.get("status"),
        error=d.get("error"),
        error_body=d.get("error_body"),
        raw=d.get("raw"),
    )


def _answer_from_dict(d: Any) -> DecisionAnswer:
    return DecisionAnswer(
        type=d.get("type", ""),
        value=d.get("value"),
        confidence=d.get("confidence"),
        probabilities=d.get("probabilities"),
        legend=d.get("legend"),
        raw=d.get("raw"),
    )


def decision_response_from_dict(d: dict[str, Any]) -> DecisionResponse:
    """Reconstruct a :class:`DecisionResponse` from its ``to_jsonable`` dict form."""
    answers = d.get("answers") or {}
    return DecisionResponse(
        answers={k: _answer_from_dict(v) for k, v in answers.items()
                 if isinstance(v, dict)},
        usage=_usage_from_dict(d.get("usage")),
        model_reported=d.get("model_reported"),
        headers=dict(d.get("headers", {}) or {}),
        timing=_timing_from_dict(d.get("timing")),
        status=d.get("status"),
        error=d.get("error"),
        error_body=d.get("error_body"),
        raw=d.get("raw"),
    )


# --------------------------------------------------------------------------- adapter

class ReplayAdapter(Adapter):
    """Serve recorded responses for a run so it can be re-analysed with no network."""

    def __init__(self, run_dir: str, *, strict: bool = False) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.strict = strict
        self.name = f"replay:{run_dir}"

        # key -> list of response dicts (recorded order); cursor tracks replays.
        self._by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._cursor: dict[str, int] = defaultdict(int)
        self._kinds: set[str] = set()
        self._n_obs = 0
        self._records: list[dict[str, Any]] = []

        for obs in Recorder.iter_observations(run_dir):
            kind = obs.get("kind", "")
            req = obs.get("request")
            resp = obs.get("response")
            if req is None or resp is None:
                continue
            self._records.append(obs)
            self._kinds.add(kind)
            self._n_obs += 1
            self._by_key[request_hash(kind, req)].append(resp)

        self.kind = "decision" if (self._kinds == {"decide"}) else "generative"

    # ---- lookup ----------------------------------------------------------
    def _lookup(self, kind: str, req: Any) -> dict[str, Any] | None:
        key = request_hash(kind, req)
        bucket = self._by_key.get(key)
        if not bucket:
            return None
        i = self._cursor[key]
        self._cursor[key] = i + 1
        return bucket[i] if i < len(bucket) else bucket[-1]

    async def generate(self, req: GenRequest) -> GenResponse:
        hit = self._lookup("generate", req)
        if hit is None:
            if self.strict:
                raise NoRecordedResponse(f"{NO_MATCH} for generate tag={req.tag!r}")
            return GenResponse(error=NO_MATCH, status=None)
        return gen_response_from_dict(hit)

    async def decide(self, req: DecisionRequest) -> DecisionResponse:
        hit = self._lookup("decide", req)
        if hit is None:
            if self.strict:
                raise NoRecordedResponse(f"{NO_MATCH} for decide tag={req.tag!r}")
            return DecisionResponse(error=NO_MATCH, status=None)
        return decision_response_from_dict(hit)

    # ---- capabilities ----------------------------------------------------
    def static_capabilities(self) -> Capabilities:
        """Best-effort capabilities derived from what the recorded log exposes."""
        caps = Capabilities(kind=self.kind)  # type: ignore[arg-type]
        gen_resps = [gen_response_from_dict(r["response"]) for r in self._records
                     if r.get("kind") == "generate"]

        if gen_resps:
            caps.chat = True
            caps.completion = True
            has_tokens = any(r.tokens for r in gen_resps)
            caps.logprobs = has_tokens or None
            if has_tokens:
                caps.max_top_logprobs = max(
                    (len(t.top) for r in gen_resps for t in (r.tokens or [])),
                    default=0) or None
                caps.logprob_bytes = any(
                    t.bytes is not None for r in gen_resps for t in (r.tokens or []))
                caps.token_ids_known = any(
                    t.token_id is not None for r in gen_resps for t in (r.tokens or []))
            caps.streaming = any(r.timing.chunks for r in gen_resps) or None
            caps.system_fingerprint = any(r.system_fingerprint for r in gen_resps) or None
            caps.usage_prompt_tokens = any(
                r.usage.prompt_tokens is not None for r in gen_resps) or None
            caps.usage_cached_tokens = any(
                r.usage.cached_tokens is not None for r in gen_resps) or None
            caps.usage_reasoning_tokens = any(
                r.usage.reasoning_tokens is not None for r in gen_resps) or None

        models = sorted({r.model_reported for r in gen_resps if r.model_reported})
        caps.notes = [f"replay of {self._n_obs} observations from {self.run_dir}"]
        if models:
            caps.notes.append("recorded models: " + ", ".join(models))
        return caps

    def describe(self) -> dict[str, Any]:
        return {"adapter": type(self).__name__, "name": self.name, "kind": self.kind,
                "run_dir": self.run_dir, "observations": self._n_obs}
