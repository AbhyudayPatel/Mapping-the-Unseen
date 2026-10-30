"""P01 — interface / serving-stack fingerprint.

From a handful of requests we read the *envelope* rather than the model: response
headers, error-body shape, ``model_reported`` vs the requested model (aliasing), and
``system_fingerprint`` presence / drift across identical repeats. These are directly
observed facts about what sits between the caller and the weights, so they inform
``system.topology`` (single model vs router/cascade vs ensemble) and, weakly, a couple of
``serving.*`` properties. Nothing here needs logprobs or timing; it is cheap and runs in
the baseline sweep.

Serving-stack signatures recognised (headers + error bodies):
  * FastAPI + Pydantic-v2 422        — list of ``{type, loc, msg, input, ctx}`` entries.
  * vLLM                             — error object ``{"object": "error", ...}``.
  * SGLang                           — ``matched_stop`` field in chunks / raw.
  * Azure OpenAI / Foundry           — ``routing.serving_pipereplica`` (prefill/decode
                                       disaggregation), ``x-ms-served-model``,
                                       ``azureml-model-session``, ``latency_checkpoint``.
  * Cloudflare / Envoy edge          — ``cf-ray`` / ``server: cloudflare`` /
                                       ``x-envoy-upstream-service-time`` / ``istio-envoy``.
  * OpenAI-style error               — ``{"type": "invalid_request_error", ...}``.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from forensics.core.types import DecisionQuestion, DecisionRequest, GenRequest
from forensics.probes.base import (
    BudgetExhausted, Evidence, Probe, ProbeContext, ProbeResult, register,
)


def _lower_headers(h: dict[str, str] | None) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (h or {}).items()}


def _walk_strings(obj: Any, keys: bool = True):
    """Yield every scalar (and optionally key) string found anywhere in a nested body."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if keys:
                yield str(k)
            yield from _walk_strings(v, keys)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v, keys)
    elif obj is not None:
        yield str(obj)


def _has_key(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_key(v, key) for v in obj)
    return False


def _pydantic_v2_shape(body: Any) -> bool:
    """A Pydantic-v2 validation error is a list of dicts with type/loc/msg (+input/ctx)."""
    candidates = []
    if isinstance(body, list):
        candidates = body
    elif isinstance(body, dict):
        for k in ("detail", "errors", "error"):
            v = body.get(k)
            if isinstance(v, list):
                candidates = v
                break
    for item in candidates:
        if isinstance(item, dict) and {"type", "loc", "msg"} <= set(item.keys()):
            return True
    return False


def detect_stack(headers: dict[str, str], error_body: Any, raw: Any) -> dict[str, Any]:
    """Return the serving-stack signals visible in one response.

    ``signals`` is a set of short labels; the remaining keys capture the specific evidence
    (limit echoed by an error, disaggregation role, etc.). Pure function — easy to unit-test.
    """
    h = _lower_headers(headers)
    signals: set[str] = set()
    detail: dict[str, Any] = {}

    # ---- edge / proxy layer -------------------------------------------------
    server = h.get("server", "")
    if "cloudflare" in server or "cf-ray" in h:
        signals.add("cloudflare")
    if ("x-envoy-upstream-service-time" in h or "x-envoy-max-retries" in h
            or "istio-envoy" in server):
        signals.add("envoy")

    # ---- Azure / Foundry ----------------------------------------------------
    if "x-ms-served-model" in h or "azureml-model-session" in h or "x-ms-rai-invoked" in h:
        signals.add("azure")
    for blob in (raw, error_body):
        if _has_key(blob, "serving_pipereplica") or _has_key(blob, "routing"):
            signals.add("azure")
        if _has_key(blob, "latency_checkpoint"):
            signals.add("azure")

    # ---- inference server ---------------------------------------------------
    if _has_key(raw, "matched_stop") or _has_key(error_body, "matched_stop"):
        signals.add("sglang")
    # vLLM style error object
    for blob in (error_body, raw):
        if isinstance(blob, dict) and str(blob.get("object", "")).lower() == "error":
            signals.add("vllm")

    # ---- gateway framework (from the error body) ----------------------------
    if _pydantic_v2_shape(error_body):
        signals.add("fastapi_pydantic_v2")
    # OpenAI-style typed error + echoed context limit
    err_strings = list(_walk_strings(error_body))
    joined = " ".join(err_strings).lower()
    if "invalid_request_error" in joined:
        signals.add("openai_style_error")
    # prefill/decode disaggregation role (search headers, error body and raw)
    for s in list(_walk_strings(raw)) + err_strings + list(h.values()):
        low = s.lower()
        if "prefill" in low or "decode" in low:
            detail["disaggregation"] = s
            break
    # vendor request-id header (e.g. x-typesafe-request-id) names the gateway owner
    for name in sorted(h):
        m = re.match(r"^x-([a-z0-9]+)-request-id$", name)
        if m and m.group(1) not in ("amzn", "ms"):
            signals.add("vendor_request_id")
            detail["request_id_header"] = name
            break
    # sanitized "Invalid request." 400 (extra="forbid" top-level model, message scrubbed)
    if isinstance(error_body, dict) and str(error_body.get("message", "")).strip().lower() \
            in ("invalid request.", "invalid request"):
        signals.add("sanitized_400")
    # echoed context limit (forensic gold: exact number in the error text)
    for k in ("limit", "max_context", "maximum context length"):
        if isinstance(error_body, dict) and k in error_body:
            detail["context_limit"] = error_body[k]
    return {"signals": sorted(signals), **detail}


def _norm_model(name: str | None) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _stem(name: str) -> str:
    tok = next((t for t in re.split(r"[^a-z0-9]+", name.lower().strip()) if t), "")
    m = re.match(r"[a-z]+", tok)
    return m.group(0) if m else tok


def alias_kind(requested: str | None, reported: str | None) -> str | None:
    """Classify requested-vs-reported model names.

    ``same`` (one contains the other), ``version_alias`` (same leading name stem, e.g.
    ``jev-latest`` -> ``jev-1.13.0``: an alias resolved to a pinned version, NOT routing to a
    different model) or ``different_model`` (asked X, served Y). None if either is missing.
    """
    if not requested or not reported:
        return None
    a, b = _norm_model(requested), _norm_model(reported)
    if a and b and (a in b or b in a):
        return "same"
    sa, sb = _stem(requested), _stem(reported)
    if sa and sa == sb:
        return "version_alias"
    return "different_model"


_DEC_STATE = ("The customer was billed twice for the same monthly subscription and is "
              "asking for a refund of the duplicate charge.")
_DEC_LABELS = ["billing", "sales", "technical", "legal", "shipping", "other"]


@register
class InterfaceStackProbe(Probe):
    id = "P01.interface.stack"
    title = "Serving-stack & topology fingerprint from headers/errors"
    category = "interface"
    kinds = ("generative", "decision")
    informs = {"system.topology": 0.5, "serving.prompt_caching": 0.15,
               "serving.speculative_decoding": 0.1}
    estimates = ()
    requires = ()
    cost = 6
    baseline = True

    @staticmethod
    def _is_decision(ctx: ProbeContext) -> bool:
        kind = ctx.caps.kind if ctx.caps.kind != "unknown" else getattr(ctx.adapter, "kind", "")
        return kind == "decision"

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        if self._is_decision(ctx):
            return await self._run_decision(ctx)
        res = self.result()
        stack_rows: list[dict[str, Any]] = []
        signals_all: set[str] = set()
        detail_all: dict[str, Any] = {}

        # ---- repeated identical requests: fingerprint presence + drift --------
        n_rep = max(3, ctx.scale(6))
        probe_msg = "Reply with the single word: ping."
        fingerprints: list[str | None] = []
        models: list[str | None] = []
        first_headers: dict[str, str] = {}
        for i in range(n_rep):
            resp = await ctx.gen(GenRequest.chat(probe_msg, max_tokens=4, temperature=0.0,
                                                 seed=1234))
            if not resp.ok and resp.error_body is None and not resp.headers:
                res.notes.append(f"repeat {i}: request failed ({resp.error})")
                continue
            if i == 0:
                first_headers = dict(resp.headers or {})
            fingerprints.append(resp.system_fingerprint)
            models.append(resp.model_reported)
            sig = detect_stack(resp.headers or {}, resp.error_body, resp.raw)
            signals_all.update(sig.get("signals", []))
            detail_all.update({k: v for k, v in sig.items() if k != "signals"})

        distinct_fp = sorted({f for f in fingerprints if f is not None})
        fp_present = any(f is not None for f in fingerprints)
        fp_drift = len(distinct_fp) > 1
        model_reported = next((m for m in models if m), None)

        # ---- optional error elicitation (best effort) -------------------------
        overflow_words = int(ctx.config.get("p01_overflow_words", 0) or 0)
        if overflow_words > 0:
            big = ("word " * overflow_words).strip()
            eresp = await ctx.gen(GenRequest.chat(big, max_tokens=8))
            esig = detect_stack(eresp.headers or {}, eresp.error_body, eresp.raw)
            signals_all.update(esig.get("signals", []))
            detail_all.update({k: v for k, v in esig.items() if k != "signals"})
            if not eresp.ok:
                res.notes.append(
                    f"error elicitation: status={eresp.status} signals={esig.get('signals')}")
            res.tables.setdefault("errors", []).append({
                "status": eresp.status, "ok": eresp.ok,
                "error_body_keys": sorted(eresp.error_body.keys())
                if isinstance(eresp.error_body, dict) else None,
                "signals": ",".join(esig.get("signals", [])),
                "context_limit": esig.get("context_limit"),
            })

        # ---- model aliasing ---------------------------------------------------
        requested = ctx.config.get("requested_model") or ctx.shared.get("requested_model")
        aliasing = None
        if requested is not None and model_reported is not None:
            aliasing = _norm_model(requested) not in _norm_model(model_reported) \
                and _norm_model(model_reported) not in _norm_model(requested)
            res.notes.append(
                f"requested={requested!r} reported={model_reported!r} aliasing={aliasing}")
        elif model_reported is not None:
            res.notes.append(f"model_reported={model_reported!r} (no requested model to compare)")

        # ---- stack table ------------------------------------------------------
        stack_rows.append({
            "serving_signals": ",".join(sorted(signals_all)) or "none",
            "model_reported": model_reported,
            "system_fingerprint_present": fp_present,
            "system_fingerprint_distinct": len(distinct_fp),
            "system_fingerprint_drift": fp_drift,
            "disaggregation": detail_all.get("disaggregation"),
            "context_limit": detail_all.get("context_limit"),
            "requested_model": requested,
            "model_aliasing": aliasing,
            "n_repeats": len(fingerprints),
        })
        res.tables["stack"] = stack_rows
        header_rows = [{"header": k, "value": v} for k, v in sorted(first_headers.items())]
        if header_rows:
            res.tables["headers"] = header_rows

        # ---- evidence: system.topology ---------------------------------------
        # Fingerprint drift across identical temp-0 seeded requests => heterogeneous
        # backends (router/ensemble). Model aliasing (asked X, got Y) => router/cascade.
        vals = ("single_model", "router_or_cascade", "ensemble")
        lk: dict[str, float] = {v: 1.0 for v in vals}
        strength = 0.5
        summ = []
        if fp_present and not fp_drift and aliasing is not True:
            lk = {"single_model": 1.0, "router_or_cascade": 0.35, "ensemble": 0.2}
            summ.append("stable single fingerprint, no aliasing")
        if fp_drift:
            lk = {"single_model": 0.25, "router_or_cascade": 1.0, "ensemble": 0.9}
            strength = 0.7
            summ.append(f"fingerprint drift across repeats ({len(distinct_fp)} values)")
        if aliasing is True:
            lk = {"single_model": 0.2, "router_or_cascade": 1.0, "ensemble": 0.6}
            strength = 0.7
            summ.append("requested model != reported model (aliasing)")
        # floor everything
        lk = {k: max(0.05, v) for k, v in lk.items()}
        res.add_evidence(Evidence(
            property="system.topology",
            likelihoods=lk,
            summary="; ".join(summ) or "topology inconclusive from envelope",
            level="observed",
            strength=strength,
            stats={"distinct_fingerprints": len(distinct_fp), "fp_drift": fp_drift,
                   "aliasing": aliasing, "signals": sorted(signals_all), "n": len(fingerprints)},
            alternatives=["temp-0 nondeterminism from batching (not topology)",
                          "provider rotates fingerprints on one backend",
                          "single deployment behind a router that never re-routes"],
        ))

        res.shared["serving_signals"] = sorted(signals_all)
        res.shared["system_fingerprint_drift"] = fp_drift
        if model_reported:
            res.shared["model_reported"] = model_reported
        res.n_requests = ctx.n_requests
        return res

    # ================================================================== decision models
    async def _run_decision(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        """Decision-model (Jev-type) stack fingerprint via ``ctx.decide`` only.

        <= 6 requests: 3-4 identical valid requests (headers, model aliasing, probability-map
        key order), one wrong-type request (FastAPI/Pydantic-v2 422 shape) and one unknown
        top-level field (extra="forbid" -> sanitized 400)."""
        res = self.result()
        n_rep = max(3, min(4, ctx.scale(4)))
        q = DecisionQuestion(type="choice", instructions="Which team should handle this ticket?",
                             criteria={lab: "" for lab in _DEC_LABELS})
        signals_all: set[str] = set()
        detail_all: dict[str, Any] = {}
        models: list[str | None] = []
        orders: list[tuple[str, ...]] = []
        ok_ms: list[float] = []
        first_headers: dict[str, str] = {}
        err_rows: list[dict[str, Any]] = []

        def absorb(r) -> list[str]:
            sig = detect_stack(r.headers or {}, r.error_body, r.raw)
            signals_all.update(sig.get("signals", []))
            detail_all.update({k: v for k, v in sig.items() if k != "signals"})
            return list(sig.get("signals", []))

        try:
            for i in range(n_rep):
                r = await ctx.decide(DecisionRequest(state=_DEC_STATE, questions={"route": q}))
                absorb(r)
                if not first_headers and r.headers:
                    first_headers = dict(r.headers)
                if not r.ok:
                    res.notes.append(f"repeat {i}: status={r.status} error={str(r.error)[:120]}")
                    continue
                models.append(r.model_reported)
                if r.timing.server_ms is not None:
                    ok_ms.append(float(r.timing.server_ms))
                ans = r.answers.get("route")
                if ans is not None and ans.probabilities:
                    orders.append(tuple(str(k) for k in ans.probabilities.keys()))

            # deliberately invalid requests: error bodies are gateway fingerprints
            noul = DecisionQuestion(type="noul", instructions="Is this text in English?")
            bad_reqs = [
                ("wrong type for 'state' (number)",
                 DecisionRequest(state=12345, questions={"q": noul})),  # type: ignore[arg-type]
                ("unknown top-level field 'temperature'",
                 DecisionRequest(state="hello", questions={"q": noul},
                                 extra={"temperature": 0.0})),
            ]
            for what, breq in bad_reqs:
                r = await ctx.decide(breq)
                sigs = absorb(r)
                body = r.error_body
                if _pydantic_v2_shape(body) or _pydantic_v2_shape(r.raw):
                    shape = "pydantic_v2_list"
                else:
                    shape = "typed_dict" if isinstance(body, dict) else type(body).__name__
                msg = ""
                if isinstance(body, dict):
                    msg = str(body.get("message") or body.get("error_type") or "")[:120]
                elif isinstance(body, list):
                    msg = "; ".join(f"{b.get('type')}:{b.get('msg')}" for b in body[:3]
                                    if isinstance(b, dict))[:160]
                err_rows.append({"request": what, "status": r.status, "ok": r.ok,
                                 "shape": shape, "message": msg,
                                 "server_ms": r.timing.server_ms, "signals": ",".join(sigs)})
        except BudgetExhausted as e:
            res.status = "partial"
            res.notes.append(f"budget exhausted: {e}")
        res.n_requests = ctx.n_requests

        # ---- model aliasing -----------------------------------------------------------
        requested = (ctx.config.get("requested_model") or ctx.shared.get("requested_model")
                     or getattr(ctx.adapter, "model", None))
        reported_set = sorted({m for m in models if m})
        model_reported = reported_set[0] if reported_set else None
        akind = alias_kind(requested, model_reported)

        # ---- key-order randomisation => several Python worker processes ---------------
        distinct_orders = sorted(set(orders))
        multi_workers = len(distinct_orders) > 1
        if multi_workers:
            order_note = ("key order varies across identical requests -> several Python "
                          "worker processes/replicas (per-process hash seed)")
        elif distinct_orders == [tuple(_DEC_LABELS)]:
            order_note = "key order preserved (request order)"
        elif distinct_orders:
            order_note = "key order stable but not the request order (single hash seed?)"
        else:
            order_note = "no probability maps returned"

        # ---- validation latency: rejected before the model? ----------------------------
        err_ms = [float(r["server_ms"]) for r in err_rows
                  if r.get("server_ms") is not None and not r.get("ok")]
        gateway_validation = None
        if err_ms and ok_ms:
            gateway_validation = float(np.median(err_ms)) < 0.35 * float(np.median(ok_ms))

        res.tables["stack"] = [{
            "serving_signals": ",".join(sorted(signals_all)) or "none",
            "model_reported": model_reported,
            "requested_model": requested,
            "alias_kind": akind,
            "model_aliasing": akind in ("version_alias", "different_model") if akind else None,
            "request_id_header": detail_all.get("request_id_header"),
            "n_repeats": len(models),
            "distinct_key_orders": len(distinct_orders),
            "multiple_worker_processes": multi_workers,
            "median_server_ms_ok": round(float(np.median(ok_ms)), 2) if ok_ms else None,
            "median_server_ms_error": round(float(np.median(err_ms)), 2) if err_ms else None,
            "validation_before_model": gateway_validation,
        }]
        if err_rows:
            res.tables["errors"] = err_rows
        if first_headers:
            res.tables["headers"] = [{"header": k, "value": v}
                                     for k, v in sorted(first_headers.items())
                                     if "authorization" not in k]
        if orders:
            res.tables["key_orders"] = [{"repeat": i, "order": ",".join(o)}
                                        for i, o in enumerate(orders)]

        stack_desc = []
        if "cloudflare" in signals_all:
            stack_desc.append("Cloudflare edge")
        if "envoy" in signals_all:
            stack_desc.append("Envoy proxy")
        if "fastapi_pydantic_v2" in signals_all:
            stack_desc.append("FastAPI + Pydantic-v2 gateway")
        if "sanitized_400" in signals_all:
            stack_desc.append("extra='forbid' request model with sanitized 400s")
        if stack_desc:
            res.notes.append("Serving stack: " + " -> ".join(stack_desc) + ".")
        res.notes.append(f"Probability-map key order: {order_note} "
                         f"({len(distinct_orders)} distinct orders over {len(orders)} repeats).")
        if akind:
            res.notes.append(f"requested={requested!r} reported={model_reported!r} ({akind}).")
        if gateway_validation:
            res.notes.append("invalid requests are rejected in a few ms of upstream time -> "
                             "validation happens in the gateway, before the model/GPU.")

        # ---- evidence: system.topology ---------------------------------------------------
        if len(reported_set) > 1 or akind == "different_model":
            lk = {"single_model": 0.2, "router_or_cascade": 1.0, "ensemble": 0.6}
            strength = 0.6
            summ = (f"reported model varies ({reported_set})" if len(reported_set) > 1
                    else f"requested {requested!r} but served {model_reported!r}")
        elif models:
            lk = {"single_model": 1.0, "router_or_cascade": 0.5, "ensemble": 0.4}
            strength = 0.3
            summ = (f"stable reported model {model_reported!r}"
                    + (" (version alias of the requested name)" if akind == "version_alias"
                       else "")
                    + ("; multiple worker replicas behind one model name" if multi_workers
                       else ""))
        else:
            lk = {"single_model": 1.0, "router_or_cascade": 1.0, "ensemble": 1.0}
            strength = 0.05
            summ = "no successful decisions; topology inconclusive"
        res.add_evidence(Evidence(
            property="system.topology",
            likelihoods={k: max(0.05, v) for k, v in lk.items()},
            summary=summ, level="observed", strength=strength,
            stats={"alias_kind": akind, "reported_models": reported_set,
                   "distinct_key_orders": len(distinct_orders), "n": len(models),
                   "signals": sorted(signals_all)},
            alternatives=["replicas of one model behind a load balancer look like a single model",
                          "an alias may be re-pointed to a new version at any time",
                          "a router that never re-routes is invisible from the envelope"],
        ))

        res.shared["serving_signals"] = sorted(signals_all)
        res.shared["decision_multiple_workers"] = multi_workers
        if model_reported:
            res.shared["model_reported"] = model_reported
        return res
