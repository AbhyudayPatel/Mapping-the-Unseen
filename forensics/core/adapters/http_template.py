"""Template-driven adapter for arbitrary generative HTTP JSON endpoints.

Many providers expose a chat/completion API that is *shaped* like OpenAI's but is
not byte-compatible: a different URL, a different auth header, a body with extra
required fields, or a response whose text/usage/logprobs live at different JSON
paths.  :class:`HTTPTemplateAdapter` turns such an endpoint into a first-class
forensic :class:`~forensics.core.adapters.base.Adapter` using two declarative
pieces of configuration:

* a **request body template** — a JSON-able dict whose string values may contain
  placeholders that are filled from a :class:`~forensics.core.types.GenRequest`:

    ``{prompt}`` ``{messages_json}`` ``{max_tokens}`` ``{temperature}``
    ``{system}`` ``{stop_json}`` (plus ``{top_p}`` ``{seed}`` ``{n}``).

  When a value is *exactly* one placeholder (``"{messages_json}"``) it is replaced
  by the typed object (a list / int / float); when a placeholder is embedded in a
  larger string it is replaced by its string form.  Header values may contain
  ``{api_key}`` which is substituted at request time from the environment (the key
  itself is never stored in :meth:`to_dict` output).

* **extraction rules** — JSONPath-ish dotted paths (``a.b[0].c``, negative indices,
  and the helpers ``first()`` / ``last()`` / ``len()`` / ``join()``) that say where
  to read the reply text, finish reason, token usage, model id and (optionally)
  per-token logprobs from the response body.

Timing and retries come from :mod:`forensics.core.httpclient`, so the adapter
inherits monotonic timing, exponential backoff on 429/5xx and the golden rule that
HTTP/transport errors are *returned* (as ``GenResponse.error``), never raised.

``from_dict`` / ``to_dict`` round-trip the whole configuration through JSON so a
target can be saved as a preset (see :data:`PRESETS` and the ``openai_chat`` entry).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from forensics.core.adapters.base import Adapter
from forensics.core.httpclient import RetryPolicy, make_client, request_json
from forensics.core.types import (
    Capabilities, GenRequest, GenResponse, Message, TokenLogprob, TopToken, Usage,
)

# --------------------------------------------------------------------------- paths

_SEG_RE = re.compile(r"([^\[\].]*)((?:\[-?\d+\])+)?")
_BRACKET_RE = re.compile(r"\[(-?\d+)\]")
_HELPER_RE = re.compile(r"\s*(\w+)\((.*)\)\s*", re.DOTALL)


def _split_path(path: str) -> list[str | int]:
    """Parse ``"choices[0].message.content"`` into ``["choices", 0, "message", ...]``.

    Supports dotted keys, ``[i]`` / ``[-i]`` bracket indices and bare integer
    segments (``choices.0.text``).
    """
    parts: list[str | int] = []
    for chunk in path.split("."):
        if chunk == "":
            continue
        m = _SEG_RE.fullmatch(chunk)
        name = m.group(1) if m else chunk
        brackets = (m.group(2) if m else None) or ""
        if name != "":
            parts.append(int(name) if re.fullmatch(r"-?\d+", name) else name)
        for b in _BRACKET_RE.findall(brackets):
            parts.append(int(b))
    return parts


def _step(cur: Any, seg: str | int) -> Any:
    """One resolution step: index a list or look up a dict key. Missing -> None."""
    if cur is None:
        return None
    if isinstance(seg, int):
        if isinstance(cur, (list, tuple)):
            return cur[seg] if -len(cur) <= seg < len(cur) else None
        return None
    if isinstance(cur, dict):
        return cur.get(seg)
    return None


def _helper_join(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return "".join("" if x is None else str(x) for x in v)
    return v


_HELPERS = {
    "first": lambda v: v[0] if isinstance(v, (list, tuple)) and v else None,
    "last": lambda v: v[-1] if isinstance(v, (list, tuple)) and v else None,
    "len": lambda v: len(v) if isinstance(v, (list, tuple, str, dict)) else None,
    "join": _helper_join,
}


def resolve_path(obj: Any, path: str | None) -> Any:
    """Resolve a dotted/bracketed path (with optional ``helper(...)`` wrapper).

    Returns ``None`` for a missing key/out-of-range index or a ``None`` path, so
    extraction never raises on a shape mismatch.
    """
    if not path:
        return None
    m = _HELPER_RE.fullmatch(path)
    if m and m.group(1) in _HELPERS:
        return _HELPERS[m.group(1)](resolve_path(obj, m.group(2)))
    cur = obj
    for seg in _split_path(path):
        cur = _step(cur, seg)
        if cur is None:
            return None
    return cur


# --------------------------------------------------------------------------- body fill

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _stringify_sub(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _placeholder_values(req: GenRequest) -> dict[str, Any]:
    """Build the substitution table from a :class:`GenRequest`."""
    if req.messages:
        msgs = list(req.messages)
    elif req.prompt is not None:
        msgs = [Message("user", req.prompt)]
    else:
        msgs = []
    messages_json = [{"role": m.role, "content": m.content} for m in msgs]

    system = ""
    for m in msgs:
        if m.role == "system" and m.content:
            system = m.content
            break
    user_prompt = req.prompt if req.prompt is not None else ""
    if not user_prompt:
        for m in reversed(msgs):
            if m.role == "user":
                user_prompt = m.content
                break

    return {
        "prompt": user_prompt,
        "messages_json": messages_json,
        "max_tokens": int(req.max_tokens),
        "temperature": req.temperature,
        "top_p": req.top_p,
        "seed": req.seed,
        "n": int(req.n),
        "system": system,
        "stop_json": req.stop,
    }


def fill_template(node: Any, subs: dict[str, Any]) -> Any:
    """Recursively substitute ``{placeholder}`` tokens in a body template.

    A string that is *exactly* one known placeholder becomes the typed value; an
    embedded placeholder becomes its string form. Unknown placeholders are left
    untouched (so literal braces in a template survive).
    """
    if isinstance(node, dict):
        return {k: fill_template(v, subs) for k, v in node.items()}
    if isinstance(node, list):
        return [fill_template(v, subs) for v in node]
    if isinstance(node, str):
        exact = _PLACEHOLDER_RE.fullmatch(node.strip())
        if exact and exact.group(1) in subs:
            return subs[exact.group(1)]

        def _repl(m: re.Match[str]) -> str:
            key = m.group(1)
            return _stringify_sub(subs[key]) if key in subs else m.group(0)

        return _PLACEHOLDER_RE.sub(_repl, node)
    return node


# --------------------------------------------------------------------------- config

@dataclass
class LogprobsMap:
    """Where per-token logprobs live and how each entry maps to a ``TokenLogprob``."""

    list_path: str
    token_key: str = "token"
    logprob_key: str = "logprob"
    bytes_key: str | None = "bytes"
    token_id_key: str | None = "token_id"
    top_key: str | None = "top_logprobs"
    top_token_key: str = "token"
    top_logprob_key: str = "logprob"
    top_bytes_key: str | None = "bytes"
    top_token_id_key: str | None = "token_id"

    def to_dict(self) -> dict[str, Any]:
        return {
            "list_path": self.list_path,
            "token_key": self.token_key,
            "logprob_key": self.logprob_key,
            "bytes_key": self.bytes_key,
            "token_id_key": self.token_id_key,
            "top_key": self.top_key,
            "top_token_key": self.top_token_key,
            "top_logprob_key": self.top_logprob_key,
            "top_bytes_key": self.top_bytes_key,
            "top_token_id_key": self.top_token_id_key,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogprobsMap":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Extraction:
    """JSONPath-ish rules for reading a response body into a ``GenResponse``."""

    text_path: str = "choices[0].message.content"
    finish_path: str | None = "choices[0].finish_reason"
    usage_path: str | None = "usage"
    prompt_path: str | None = "usage.prompt_tokens"
    completion_path: str | None = "usage.completion_tokens"
    total_path: str | None = "usage.total_tokens"
    cached_path: str | None = None
    reasoning_path: str | None = None
    model_path: str | None = "model"
    system_fingerprint_path: str | None = "system_fingerprint"
    choices_path: str | None = None
    choice_text_subpath: str | None = None
    logprobs: LogprobsMap | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_path": self.text_path,
            "finish_path": self.finish_path,
            "usage": {
                "path": self.usage_path,
                "prompt_path": self.prompt_path,
                "completion_path": self.completion_path,
                "total_path": self.total_path,
                "cached_path": self.cached_path,
                "reasoning_path": self.reasoning_path,
            },
            "model_path": self.model_path,
            "system_fingerprint_path": self.system_fingerprint_path,
            "choices_path": self.choices_path,
            "choice_text_subpath": self.choice_text_subpath,
            "logprobs": self.logprobs.to_dict() if self.logprobs else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Extraction":
        usage = d.get("usage") or {}
        lp = d.get("logprobs")
        return cls(
            text_path=d.get("text_path", cls.text_path),
            finish_path=d.get("finish_path", cls.finish_path),
            usage_path=usage.get("path", cls.usage_path),
            prompt_path=usage.get("prompt_path", cls.prompt_path),
            completion_path=usage.get("completion_path", cls.completion_path),
            total_path=usage.get("total_path", cls.total_path),
            cached_path=usage.get("cached_path", cls.cached_path),
            reasoning_path=usage.get("reasoning_path", cls.reasoning_path),
            model_path=d.get("model_path", cls.model_path),
            system_fingerprint_path=d.get("system_fingerprint_path",
                                          cls.system_fingerprint_path),
            choices_path=d.get("choices_path", cls.choices_path),
            choice_text_subpath=d.get("choice_text_subpath", cls.choice_text_subpath),
            logprobs=LogprobsMap.from_dict(lp) if lp else None,
        )


@dataclass
class TemplateConfig:
    """Full, JSON-serialisable description of an HTTP generative endpoint."""

    url: str
    body_template: dict[str, Any]
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    api_key_env: str | None = None
    extraction: Extraction = field(default_factory=Extraction)
    timeout: float = 60.0
    max_retries: int = 4
    name: str = "http-template"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "url": self.url,
            "headers": dict(self.headers),
            "api_key_env": self.api_key_env,
            "body_template": json.loads(json.dumps(self.body_template)),
            "extraction": self.extraction.to_dict(),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TemplateConfig":
        return cls(
            url=d["url"],
            body_template=d.get("body_template", {}),
            method=d.get("method", cls.method),
            headers=dict(d.get("headers", {})),
            api_key_env=d.get("api_key_env"),
            extraction=Extraction.from_dict(d.get("extraction", {})),
            timeout=d.get("timeout", cls.timeout),
            max_retries=d.get("max_retries", cls.max_retries),
            name=d.get("name", cls.name),
        )


# --------------------------------------------------------------------------- adapter

class HTTPTemplateAdapter(Adapter):
    """Generative adapter for an arbitrary JSON HTTP endpoint, driven by templates."""

    kind = "generative"

    def __init__(
        self,
        config: TemplateConfig,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.name = config.name
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._policy = RetryPolicy(max_retries=config.max_retries)

    # ---- construction / serialisation ------------------------------------
    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, api_key: str | None = None,
                  client: httpx.AsyncClient | None = None) -> "HTTPTemplateAdapter":
        return cls(TemplateConfig.from_dict(cfg), api_key=api_key, client=client)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable config (never includes the API key)."""
        return self.cfg.to_dict()

    def describe(self) -> dict[str, Any]:
        return {"adapter": type(self).__name__, "name": self.name, "kind": self.kind,
                "url": self.cfg.url, "method": self.cfg.method}

    # ---- plumbing --------------------------------------------------------
    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        if self.cfg.api_key_env:
            return os.environ.get(self.cfg.api_key_env, "")
        return ""

    def _headers(self) -> dict[str, str]:
        key = self._resolve_api_key()
        return {k: v.replace("{api_key}", key) for k, v in self.cfg.headers.items()}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = make_client(timeout=self.cfg.timeout)
            self._owns_client = True
        return self._client

    def _build_body(self, req: GenRequest) -> dict[str, Any]:
        body = fill_template(self.cfg.body_template, _placeholder_values(req))
        if not isinstance(body, dict):
            body = {"body": body}
        if req.extra:
            body.update(req.extra)
        return body

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ---- capabilities ----------------------------------------------------
    def static_capabilities(self) -> Capabilities:
        ex = self.cfg.extraction
        return Capabilities(
            kind="generative",
            chat=True,
            logprobs=ex.logprobs is not None,
            usage_prompt_tokens=ex.prompt_path is not None,
            system_fingerprint=ex.system_fingerprint_path is not None,
            notes=[f"http-template {self.cfg.method} {self.cfg.url}"],
        )

    # ---- generate --------------------------------------------------------
    async def generate(self, req: GenRequest) -> GenResponse:
        body = self._build_body(req)
        status, headers, resp_body, timing, error = await request_json(
            self._get_client(), self.cfg.method, self.cfg.url,
            headers=self._headers(), json_body=body, timeout=self.cfg.timeout,
            policy=self._policy,
        )

        if error is not None:
            return GenResponse(status=status, headers=headers, timing=timing,
                               error=error, error_body=resp_body, raw=resp_body)

        return self._extract(resp_body, status, headers, timing)

    def _extract(self, body: Any, status: int | None, headers: dict[str, str],
                 timing: Any) -> GenResponse:
        ex = self.cfg.extraction
        text = resolve_path(body, ex.text_path)
        text = "" if text is None else str(text)

        choices_text: list[str] = [text] if text else []
        if ex.choices_path:
            raw_choices = resolve_path(body, ex.choices_path)
            if isinstance(raw_choices, list):
                collected: list[str] = []
                for ch in raw_choices:
                    val = (resolve_path(ch, ex.choice_text_subpath)
                           if ex.choice_text_subpath else ch)
                    collected.append("" if val is None else str(val))
                if collected:
                    choices_text = collected
                    text = collected[0]

        usage = Usage(
            prompt_tokens=_as_int(resolve_path(body, ex.prompt_path)),
            completion_tokens=_as_int(resolve_path(body, ex.completion_path)),
            total_tokens=_as_int(resolve_path(body, ex.total_path)),
            cached_tokens=_as_int(resolve_path(body, ex.cached_path)),
            reasoning_tokens=_as_int(resolve_path(body, ex.reasoning_path)),
            raw=resolve_path(body, ex.usage_path) if ex.usage_path else None,
        )

        finish = resolve_path(body, ex.finish_path)
        model = resolve_path(body, ex.model_path)
        fp = resolve_path(body, ex.system_fingerprint_path)

        return GenResponse(
            text=text,
            choices_text=choices_text,
            tokens=self._extract_logprobs(body) if ex.logprobs else None,
            finish_reason=None if finish is None else str(finish),
            usage=usage,
            model_reported=None if model is None else str(model),
            system_fingerprint=None if fp is None else str(fp),
            headers=headers,
            timing=timing,
            status=status,
            raw=body,
        )

    def _extract_logprobs(self, body: Any) -> list[TokenLogprob] | None:
        lp = self.cfg.extraction.logprobs
        assert lp is not None
        entries = resolve_path(body, lp.list_path)
        if not isinstance(entries, list):
            return None
        out: list[TokenLogprob] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            top: list[TopToken] = []
            if lp.top_key:
                raw_top = e.get(lp.top_key)
                if isinstance(raw_top, list):
                    for t in raw_top:
                        if not isinstance(t, dict):
                            continue
                        top.append(TopToken(
                            token=str(t.get(lp.top_token_key, "")),
                            logprob=_as_float(t.get(lp.top_logprob_key)) or 0.0,
                            bytes=t.get(lp.top_bytes_key) if lp.top_bytes_key else None,
                            token_id=_as_int(t.get(lp.top_token_id_key))
                            if lp.top_token_id_key else None,
                        ))
            out.append(TokenLogprob(
                token=str(e.get(lp.token_key, "")),
                logprob=_as_float(e.get(lp.logprob_key)) or 0.0,
                bytes=e.get(lp.bytes_key) if lp.bytes_key else None,
                token_id=_as_int(e.get(lp.token_id_key)) if lp.token_id_key else None,
                top=top,
            ))
        return out


# --------------------------------------------------------------------------- coercion

def _as_int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("inf", "+inf", "infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
        if s == "nan":
            return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- presets

#: A ready-to-edit template for a stock OpenAI ``/v1/chat/completions`` endpoint.
#: Copy it, point ``url`` at your target, set ``api_key_env`` and adjust the model.
#: For Azure OpenAI, swap the auth header to ``{"api-key": "{api_key}"}`` and use
#: the ``.../openai/v1/chat/completions`` URL (the body/extraction are identical).
OPENAI_CHAT_PRESET: dict[str, Any] = {
    "name": "openai_chat",
    "method": "POST",
    "url": "https://api.openai.com/v1/chat/completions",
    "headers": {
        "Authorization": "Bearer {api_key}",
        "content-type": "application/json",
    },
    "api_key_env": "OPENAI_API_KEY",
    "body_template": {
        "model": "gpt-4o-mini",
        "messages": "{messages_json}",
        "max_tokens": "{max_tokens}",
        "temperature": "{temperature}",
    },
    "extraction": {
        "text_path": "choices[0].message.content",
        "finish_path": "choices[0].finish_reason",
        "usage": {
            "path": "usage",
            "prompt_path": "usage.prompt_tokens",
            "completion_path": "usage.completion_tokens",
            "total_path": "usage.total_tokens",
        },
        "model_path": "model",
        "system_fingerprint_path": "system_fingerprint",
        "logprobs": {
            "list_path": "choices[0].logprobs.content",
            "token_key": "token",
            "logprob_key": "logprob",
            "bytes_key": "bytes",
            "top_key": "top_logprobs",
        },
    },
    "timeout": 60.0,
    "max_retries": 4,
}

PRESETS: dict[str, dict[str, Any]] = {"openai_chat": OPENAI_CHAT_PRESET}
