"""Optional Azure-backed LLM advisor.

The advisor is a *convenience* layer the engine can use (passed as ``llm=...``) to add
qualitative, LLM-assisted judgements on top of the deterministic, statistics-driven core:

* :meth:`LLMAdvisor.judge_freeform` — classify a free-text model output (is it a refusal?
  does it self-identify as another family? base-continuation vs instruction-following?);
* :meth:`LLMAdvisor.propose_experiments` — invent differential experiments that best
  separate the leading hypotheses (powers the P21 automated experiment generator);
* :meth:`LLMAdvisor.summarize_findings` — a short expert narrative for the report.

Design contract (why this file can never break core inference):

* **Fully optional.** The engine works with ``llm=None``; nothing here is required.
* **Never raises.** Every public method is defensive: on *any* error (missing creds,
  network failure, a non-JSON reply, a rejected parameter) it returns a safe default and
  appends a human-readable note to :attr:`LLMAdvisor.notes` — it never propagates an
  exception into a probe or the engine.
* **Recorded + marked.** Every call is appended to :attr:`LLMAdvisor.transcript`
  (prompt/response previews only — no secrets), so a run can show exactly what the LLM was
  asked and answered; the P21 probe flags its result ``llm_assisted`` so the RunResult
  records that an LLM was involved.
* **Cheap.** Tight prompts, small ``max_tokens``, a cheap deployment (``gpt-4o-mini`` by
  default) and a hard per-advisor call cap (:attr:`LLMConfig.max_calls`).

Secrets are read from the environment / ``.env`` via :func:`make_default_advisor` and are
never written to the transcript or any run artefact.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from forensics.core.adapters.openai_compat import OpenAICompatAdapter
from forensics.core.types import GenRequest

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_CALLS = 24
#: Longest prompt/response fragment kept in the transcript (chars).
_PREVIEW = 400
#: Longest free-text blob we will ever send to the model (chars).
_TEXT_CAP = 4000


# --------------------------------------------------------------------------- config
@dataclass
class LLMConfig:
    """Connection + budget knobs for the advisor."""

    base_url: str
    api_key: str
    model: str = DEFAULT_MODEL
    auth_style: str = "api-key"           # "api-key" (Azure) | "bearer"
    api_version: str | None = None
    max_calls: int = DEFAULT_MAX_CALLS    # hard cap on requests per advisor instance
    max_tokens: int = 512                 # default per-call output cap
    temperature: float = 0.0
    timeout: float = 60.0


# --------------------------------------------------------------------------- JSON parsing
def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _balanced(text: str, open_ch: str, close_ch: str) -> Any:
    """Parse the first balanced ``open_ch..close_ch`` block as JSON, or None."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _parse_json(text: str | None) -> Any:
    """Best-effort JSON out of a model reply: whole string, then a balanced object/array."""
    if not text:
        return None
    body = _strip_fences(text)
    try:
        return json.loads(body)
    except Exception:
        pass
    obj = _balanced(body, "{", "}")
    if obj is not None:
        return obj
    return _balanced(body, "[", "]")


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return v


def _clip(text: Any, n: int) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= n else s[:n] + "…"


# --------------------------------------------------------------------------- advisor
class LLMAdvisor:
    """Wraps an :class:`OpenAICompatAdapter` pointed at an Azure chat deployment.

    Construct from an existing adapter, an :class:`LLMConfig`, or a plain ``dict`` of
    :class:`LLMConfig` fields.  Prefer :func:`make_default_advisor` for the common case of
    reading credentials from ``.env``.
    """

    def __init__(self, adapter_or_config: "OpenAICompatAdapter | LLMConfig | dict[str, Any]") -> None:
        self.cfg: LLMConfig | None
        if isinstance(adapter_or_config, OpenAICompatAdapter):
            self.adapter = adapter_or_config
            self.cfg = None
            self.model = self.adapter.model
            self.max_calls = DEFAULT_MAX_CALLS
            self.max_tokens = 512
            self.temperature = 0.0
        else:
            if isinstance(adapter_or_config, dict):
                cfg = LLMConfig(**adapter_or_config)
            elif isinstance(adapter_or_config, LLMConfig):
                cfg = adapter_or_config
            else:  # pragma: no cover - defensive
                raise TypeError(
                    "LLMAdvisor expects an OpenAICompatAdapter, LLMConfig, or dict; "
                    f"got {type(adapter_or_config).__name__}")
            self.cfg = cfg
            self.adapter = OpenAICompatAdapter(
                base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model,
                auth_style=cfg.auth_style, api_version=cfg.api_version,
                name=f"llm-advisor:{cfg.model}", timeout=cfg.timeout)
            self.model = cfg.model
            self.max_calls = int(cfg.max_calls)
            self.max_tokens = int(cfg.max_tokens)
            self.temperature = float(cfg.temperature)

        self.n_calls = 0
        self.notes: list[str] = []
        self.transcript: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ plumbing
    def _note(self, msg: str) -> str:
        self.notes.append(msg)
        return msg

    async def _chat(self, system: str, user: str, *, method: str,
                    max_tokens: int, json_mode: bool) -> tuple[str | None, str | None]:
        """One recorded, capped, defensive chat call. Returns ``(text, note)``.

        ``text`` is None on any failure and ``note`` explains it; otherwise ``note`` is
        None.  Never raises.
        """
        if self.n_calls >= self.max_calls:
            return None, self._note(
                f"{method}: advisor call cap reached ({self.max_calls}); returning default")
        self.n_calls += 1
        entry: dict[str, Any] = {
            "method": method, "model": self.model, "call": self.n_calls,
            "system": _clip(system, _PREVIEW), "user": _clip(user, _PREVIEW),
        }
        try:
            req = GenRequest.chat(
                user, system=system, max_tokens=int(max_tokens),
                temperature=self.temperature)
            if json_mode:
                # gpt-4o-mini honours this; adapters that reject it strip it and retry.
                req.extra = {"response_format": {"type": "json_object"}}
            resp = await self.adapter.generate(req)
        except Exception as exc:  # noqa: BLE001 - defensive: never leak
            entry["ok"] = False
            entry["error"] = _clip(f"{type(exc).__name__}: {exc}", _PREVIEW)
            self.transcript.append(entry)
            return None, self._note(f"{method}: advisor call raised {type(exc).__name__}: {exc}")

        if not resp.ok:
            entry["ok"] = False
            entry["error"] = _clip(resp.error, _PREVIEW)
            entry["status"] = resp.status
            self.transcript.append(entry)
            return None, self._note(
                f"{method}: endpoint error (status {resp.status}): {_clip(resp.error, 160)}")

        text = resp.text or ""
        entry["ok"] = True
        entry["response"] = _clip(text, _PREVIEW)
        self.transcript.append(entry)
        if not text.strip():
            return None, self._note(f"{method}: empty response from the model")
        return text, None

    async def aclose(self) -> None:
        """Close the underlying HTTP client (safe to call repeatedly)."""
        try:
            await self.adapter.aclose()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ judge_freeform
    async def judge_freeform(self, question: str, text: str,
                             rubric: str | None = None) -> dict[str, Any]:
        """Classify a free-text model output.

        Returns ``{"label", "rationale", "confidence"}`` (a ``"note"`` key is added when the
        result is a degraded default).  ``label`` is a short direct answer to ``question``
        (for yes/no questions, ``"yes"``/``"no"``).
        """
        default: dict[str, Any] = {"label": "unknown", "rationale": "", "confidence": 0.0}
        system = (
            "You are a precise classifier for black-box AI model forensics. "
            "Answer ONLY with strict minified JSON of the form "
            '{"label": <string>, "rationale": <string>, "confidence": <number 0..1>}. '
            "No markdown, no extra text. For yes/no questions the label must be "
            "'yes' or 'no'. Keep the rationale under 25 words.")
        parts = [f"Question: {question}"]
        if rubric:
            parts.append(f"Rubric: {_clip(rubric, 600)}")
        parts.append("Text to classify (delimited):")
        parts.append(f'"""{_clip(text, _TEXT_CAP)}"""')
        parts.append("Return the JSON now.")
        user = "\n".join(parts)

        reply, note = await self._chat(system, user, method="judge_freeform",
                                       max_tokens=200, json_mode=True)
        if reply is None:
            return {**default, "note": note}
        obj = _parse_json(reply)
        if not isinstance(obj, dict):
            return {**default, "note": self._note("judge_freeform: reply was not JSON")}
        label = obj.get("label", obj.get("answer", ""))
        conf = _as_float(obj.get("confidence"), 0.0)
        conf = min(1.0, max(0.0, conf))
        return {
            "label": str(label).strip() or "unknown",
            "rationale": str(obj.get("rationale", "")).strip(),
            "confidence": conf,
        }

    # ------------------------------------------------------------------ propose_experiments
    async def propose_experiments(self, belief_summary: Any, caps_summary: Any,
                                  uncertainty: Any) -> list[dict[str, Any]]:
        """Invent differential experiments that best separate the leading hypotheses.

        Returns a small ranked list of ``{name, hypothesis, method, observable,
        distinguishes}`` dicts (empty list on any failure).
        """
        system = (
            "You are an expert at black-box LLM architecture forensics (inferring "
            "architecture, serving stack and training from an API alone). Given the current "
            "beliefs, the interface capabilities, and the most uncertain properties, propose "
            "differential experiments that best SEPARATE the leading competing hypotheses. "
            "Answer ONLY with strict minified JSON of the form "
            '{"experiments": [{"name": <string>, "hypothesis": <string>, "method": <string>, '
            '"observable": <string>, "distinguishes": <string>}]}. '
            "Give at most 3 experiments, ranked best first, each field one concise sentence. "
            "Prefer experiments feasible with the stated capabilities. No markdown.")
        user = "\n".join([
            "Current beliefs (leading value per property):",
            _clip(_to_text(belief_summary), 1200),
            "",
            "Interface capabilities:",
            _clip(_to_text(caps_summary), 800),
            "",
            "Most uncertain properties (highest entropy):",
            _clip(_to_text(uncertainty), 800),
            "",
            "Return the JSON now.",
        ])

        reply, note = await self._chat(system, user, method="propose_experiments",
                                       max_tokens=700, json_mode=True)
        if reply is None:
            return []
        obj = _parse_json(reply)
        items: Any
        if isinstance(obj, dict):
            items = obj.get("experiments") or obj.get("proposals") or []
            if not isinstance(items, list):
                items = []
        elif isinstance(obj, list):
            items = obj
        else:
            self._note("propose_experiments: reply was not JSON; no proposals")
            items = []

        out: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            out.append({
                "name": str(it.get("name", "")).strip() or f"experiment_{len(out) + 1}",
                "hypothesis": str(it.get("hypothesis", "")).strip(),
                "method": str(it.get("method", "")).strip(),
                "observable": str(it.get("observable", "")).strip(),
                "distinguishes": str(it.get("distinguishes",
                                            it.get("separates", ""))).strip(),
            })
            if len(out) >= 5:
                break
        return out

    # ------------------------------------------------------------------ summarize_findings
    async def summarize_findings(self, run_summary: Any) -> str:
        """A concise expert narrative for the report executive summary.

        The returned string is explicitly tagged ``(LLM-generated)``.  Returns ``""`` on any
        failure (with a note recorded).
        """
        system = (
            "You are an expert model-forensics analyst. Write a concise executive summary "
            "(at most 120 words, plain prose, no markdown headings or lists) of the findings "
            "below for a technical report. Be specific and calibrated: state what is well "
            "supported and hedge where the evidence is weak. Do not invent facts.")
        user = "Findings:\n" + _clip(_to_text(run_summary), 3000) + "\n\nWrite the summary now."
        reply, _note = await self._chat(system, user, method="summarize_findings",
                                        max_tokens=260, json_mode=False)
        if reply is None:
            return ""
        return "(LLM-generated) " + reply.strip()


# --------------------------------------------------------------------------- helpers
def _to_text(obj: Any) -> str:
    """Render a summary argument (str / dict / list) as compact text for a prompt."""
    if obj is None:
        return "(none)"
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str, indent=None)
    except Exception:  # noqa: BLE001
        return str(obj)


def make_default_advisor(model: str | None = None,
                         **overrides: Any) -> LLMAdvisor | None:
    """Read ``.env`` and return an :class:`LLMAdvisor`, or ``None`` if creds are absent.

    Uses ``AZURE_OPENAI_BASE_URL`` / ``AZURE_OPENAI_API_KEY`` (and optional
    ``AZURE_OPENAI_API_VERSION`` / ``AZURE_ADVISOR_MODEL``).  ``model`` and any
    :class:`LLMConfig` field in ``overrides`` win over the environment defaults.
    """
    try:
        from forensics.config import load_env
        load_env()
    except Exception:  # noqa: BLE001
        pass
    base = os.environ.get("AZURE_OPENAI_BASE_URL")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not base or not key:
        return None
    cfg = LLMConfig(
        base_url=base,
        api_key=key,
        model=model or os.environ.get("AZURE_ADVISOR_MODEL", DEFAULT_MODEL),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return LLMAdvisor(cfg)


# --------------------------------------------------------------------------- self-test
async def _self_test() -> int:
    """Tiny live smoke test (<=5 requests). Prints PASS or SKIP; never raises."""
    adv = make_default_advisor()
    if adv is None:
        print("SKIP | no Azure creds in .env (AZURE_OPENAI_BASE_URL/AZURE_OPENAI_API_KEY)")
        return 0
    try:
        judged = await adv.judge_freeform(
            "Is this a refusal?", "I cannot help with that request.")
        lab = str(judged.get("label", "")).lower()
        rat = str(judged.get("rationale", "")).lower()
        refusal_ok = lab in ("yes", "true", "refusal", "refuses") or "refus" in (lab + rat)

        props = await adv.propose_experiments(
            belief_summary={"arch.sparsity": "moe (0.55)", "attn.kv_sharing": "gqa_or_mqa (0.6)"},
            caps_summary={"kind": "generative", "logprobs": True, "logit_bias": True},
            uncertainty=["arch.sparsity", "attn.kv_sharing", "pos.encoding"])
        prop_ok = len(props) >= 1

        if judged.get("note") or not refusal_ok:
            print(f"SKIP | advisor reachable but judge degraded: {judged}")
            return 0
        if not prop_ok:
            print(f"SKIP | advisor reachable but no proposals returned; notes={adv.notes}")
            return 0
        print(f"PASS | calls={adv.n_calls} judge_label={judged['label']!r} "
              f"conf={judged['confidence']:.2f} proposals={len(props)} "
              f"first={props[0]['name']!r}")
        return 0
    finally:
        await adv.aclose()


if __name__ == "__main__":  # pragma: no cover
    import asyncio
    import sys

    sys.exit(asyncio.run(_self_test()))
