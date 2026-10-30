"""Core data types shared by adapters, probes, inference and reporting.

Rules:
- Adapters fill every field they can observe; unknown fields stay None (never guessed).
- All timing lives in `Timing` and is produced by the adapter (virtual time in the
  simulator, wall-clock in real adapters). Probes must never time requests themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# --------------------------------------------------------------------------- requests
@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenRequest:
    """A generative request. Either `messages` (chat) or `prompt` (raw completion)."""
    messages: list[Message] | None = None
    prompt: str | None = None
    max_tokens: int = 16
    temperature: float | None = 0.0
    top_p: float | None = None
    seed: int | None = None
    n: int = 1
    stop: list[str] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    logit_bias: dict[int, float] | None = None
    stream: bool = False
    reasoning_effort: str | None = None
    # Provider-specific passthrough (merged into the body verbatim).
    extra: dict[str, Any] = field(default_factory=dict)
    # Free-form label for recording (probe id, variant, ...).
    tag: str = ""

    @staticmethod
    def chat(user: str, system: str | None = None, **kw: Any) -> "GenRequest":
        msgs = ([Message("system", system)] if system else []) + [Message("user", user)]
        return GenRequest(messages=msgs, **kw)


@dataclass
class DecisionQuestion:
    """Typed question for decision models (Jev-like). type: choice | score | noul."""
    type: Literal["choice", "score", "noul"]
    instructions: str
    # choice: {label: description}; score: [level descriptions, low→high]; noul: None
    criteria: dict[str, str] | list[str] | None = None


@dataclass
class DecisionRequest:
    state: str | dict[str, Any]
    questions: dict[str, DecisionQuestion]
    extra: dict[str, Any] = field(default_factory=dict)
    tag: str = ""


# --------------------------------------------------------------------------- responses
@dataclass
class TopToken:
    token: str
    logprob: float
    bytes: list[int] | None = None
    token_id: int | None = None


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    bytes: list[int] | None = None
    token_id: int | None = None
    top: list[TopToken] = field(default_factory=list)


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw: dict[str, Any] | None = None


@dataclass
class Chunk:
    """One streamed chunk: arrival time (seconds, relative to t_start) and its text."""
    t: float
    text: str
    n_tokens: int | None = None  # if the provider reports per-chunk token counts / logprobs


@dataclass
class Timing:
    """All times in seconds relative to request start (t_start == 0)."""
    t_first_byte: float | None = None      # SECONDS: response headers received
    t_first_token: float | None = None     # SECONDS: first non-empty content chunk (streaming)
    t_end: float | None = None             # SECONDS: full response received
    chunks: list[Chunk] = field(default_factory=list)  # Chunk.t in SECONDS
    # NOTE: server_ms is in MILLISECONDS (from parse_server_timing / x-envoy-upstream-service-time
    # / Azure latency_checkpoint), unlike every other field here which is in seconds. The
    # simulator, httpclient and all live adapters agree on this. Timing probes MUST convert
    # (server_ms / 1000.0) before combining with t_* fields.
    server_ms: float | None = None         # MILLISECONDS: server-reported processing time
    wall_start: float | None = None        # absolute epoch seconds (for drift/load analysis)

    @property
    def ttft(self) -> float | None:
        return self.t_first_token if self.t_first_token is not None else self.t_first_byte

    def inter_token_latencies(self) -> list[float]:
        ts = [c.t for c in self.chunks if c.text]
        return [b - a for a, b in zip(ts, ts[1:])]


@dataclass
class GenResponse:
    text: str = ""
    choices_text: list[str] = field(default_factory=list)   # all n choices (text[0] == text)
    tokens: list[TokenLogprob] | None = None                # logprobs of choice 0, if exposed
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model_reported: str | None = None
    system_fingerprint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timing: Timing = field(default_factory=Timing)
    status: int | None = None
    error: str | None = None          # error message text if the request failed
    error_body: Any = None
    raw: Any = None                   # raw JSON (non-streaming) or list of chunk JSONs

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class DecisionAnswer:
    type: str
    value: Any = None                          # chosen label / score float / noul prob
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    legend: dict[str, str] | None = None
    raw: dict[str, Any] | None = None


@dataclass
class DecisionResponse:
    answers: dict[str, DecisionAnswer] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    model_reported: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timing: Timing = field(default_factory=Timing)
    status: int | None = None
    error: str | None = None
    error_body: Any = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------- capabilities
@dataclass
class Capabilities:
    """What the interface exposes. Filled by capability discovery; drives probe applicability
    and identifiability levels. None = unknown/untested, False = tested & unsupported."""
    kind: Literal["generative", "decision", "embedding", "unknown"] = "unknown"
    chat: bool | None = None
    completion: bool | None = None
    logprobs: bool | None = None
    max_top_logprobs: int | None = None
    logprob_bytes: bool | None = None
    logit_bias: bool | None = None              # accepted by the API
    logit_bias_affects_logprobs: bool | None = None  # returned logprobs reflect the bias
    logit_bias_max: float | None = None
    streaming: bool | None = None
    stream_usage: bool | None = None
    temperature: bool | None = None
    top_p: bool | None = None
    seed: bool | None = None
    n: bool | None = None
    stop: bool | None = None
    system_role: bool | None = None
    assistant_prefill: bool | None = None       # continues a trailing assistant message
    usage_prompt_tokens: bool | None = None
    usage_cached_tokens: bool | None = None
    usage_reasoning_tokens: bool | None = None
    reasoning_effort: bool | None = None
    system_fingerprint: bool | None = None
    server_timing_header: str | None = None     # header name carrying server time
    max_tokens_param: str | None = None         # "max_tokens" | "max_completion_tokens"
    token_ids_known: bool | None = None         # we can map token strings to ids (tokenizer matched)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses to JSON-serialisable structures."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)
    return obj
