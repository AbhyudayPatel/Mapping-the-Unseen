"""Decision-model adapter for TypeSafe Jev (``POST /v1/systemone``).

Jev is a non-generative, schema-constrained decision model: one ``state`` plus a
map of typed questions (noul / choice / score), evaluated in parallel, returning
per-question answers with calibrated probabilities. There are no sampling,
logprob or streaming controls (see ``research/jev.md`` §2). The adapter maps a
:class:`DecisionRequest` onto the JSON body and parses the answers/usage back
into a :class:`DecisionResponse`; it never raises on API errors (error texts are
forensic evidence). Server time comes from the ``x-envoy-upstream-service-time``
header, which :func:`forensics.core.httpclient.request_json` parses into
``timing.server_ms``.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from forensics.core.adapters.base import Adapter
from forensics.core.httpclient import make_client, request_json
from forensics.core.types import (
    Capabilities, DecisionAnswer, DecisionQuestion, DecisionRequest, DecisionResponse,
    Usage,
)


def _lower_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


class DecisionAdapter(Adapter):
    """Adapter for the TypeSafe Jev decision API (Bearer auth)."""

    kind = "decision"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "jev-latest",
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL")
                         or "https://api.typesafe.ai/v1/systemone").rstrip("/")
        self._api_key = api_key or os.environ.get("JEV_API_KEY") or ""
        self.model = model
        self.timeout = timeout
        self.name = name or f"jev:{model}"
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------ plumbing
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = make_client(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self._api_key:
            h["authorization"] = f"Bearer {self._api_key}"
        return h

    @staticmethod
    def _q_body(q: DecisionQuestion) -> dict[str, Any]:
        d: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
        if q.criteria is not None:
            d["criteria"] = q.criteria
        return d

    # ------------------------------------------------------------------ decide
    async def decide(self, req: DecisionRequest) -> DecisionResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "state": req.state,
            "questions": {k: self._q_body(q) for k, q in req.questions.items()},
        }
        if req.extra:
            body.update(req.extra)
        client = await self._get_client()
        status, headers, rbody, timing, error = await request_json(
            client, "POST", self.base_url, headers=self._headers(),
            json_body=body, timeout=self.timeout,
        )
        return self._parse(status, headers, rbody, timing, error)

    def _parse(self, status, headers, body, timing, error) -> DecisionResponse:
        resp = DecisionResponse(status=status, headers=_lower_headers(headers),
                                timing=timing, raw=body)
        if isinstance(body, dict):
            resp.model_reported = body.get("model")
        if error is not None:
            resp.error = error
            resp.error_body = body.get("detail", body) if isinstance(body, dict) else body
            return resp
        if not isinstance(body, dict):
            resp.error = f"unexpected decision response body: {type(body).__name__}"
            resp.error_body = body
            return resp

        answers: dict[str, DecisionAnswer] = {}
        for key, a in (body.get("answers") or {}).items():
            if not isinstance(a, dict):
                continue
            typ = a.get("type")
            if typ == "noul":
                value: Any = a.get("noul")
            elif typ == "choice":
                value = a.get("choice")
            elif typ == "score":
                value = a.get("score")
            else:
                value = a.get(typ) if typ else None
            answers[key] = DecisionAnswer(
                type=typ, value=value, confidence=a.get("confidence"),
                probabilities=a.get("probabilities"), legend=a.get("legend"), raw=a)
        resp.answers = answers

        u = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        resp.usage = Usage(
            prompt_tokens=u.get("input_tokens", u.get("prompt_tokens")),
            completion_tokens=u.get("output_tokens", u.get("completion_tokens")),
            total_tokens=u.get("total_tokens"),
            raw=dict(u) if u else None,
        )
        return resp

    # ------------------------------------------------------------------ capabilities
    def static_capabilities(self) -> Capabilities:
        return Capabilities(
            kind="decision", chat=False, completion=False, logprobs=False,
            logit_bias=False, streaming=False, temperature=False, top_p=False,
            seed=False, n=False, system_role=False,
            server_timing_header="x-envoy-upstream-service-time",
            notes=["decision model (Jev): no sampling/logprob/streaming controls"],
        )

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(base_url=self.base_url, model=self.model)
        return d

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
