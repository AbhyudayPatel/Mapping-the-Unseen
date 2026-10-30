"""Ground-truth simulator adapter.

This is the reference "device under test" for the whole forensics stack: it has a
*known* hidden architecture (a :class:`SimConfig`) and synthesises API responses whose
observable side-channels are physically faithful to that architecture. Every probe is
validated against it, so its fidelity is what makes the probe test-suite meaningful.

What is modelled (and why a probe can recover it):

* **Logit geometry / softmax bottleneck** — logits are produced by a genuine low-rank
  map ``z = E @ h (+ bias)`` with a fixed unembedding ``E in R^{V x d}`` and a per-prompt
  hidden vector ``h in R^d``.  The matrix of logit vectors over many prompts therefore has
  rank exactly ``d`` (up to the log-softmax / bias offset), which is what P03 recovers by
  SVD (Finlayson et al. 2024, Carlini et al. 2024).
* **Numeric precision** — logit *values* are rounded onto the configured dtype lattice
  (real bf16 / fp16 / fp8-E4M3 rounding), so the ULP staircase P04 measures is genuine.
* **Timing** — a roofline model: compute-bound prefill (linear in prompt length plus a
  quadratic attention term that vanishes for SSM/linear or saturates at a sliding window),
  and memory-bandwidth-bound decode whose per-token latency grows with the KV-cache read
  (so inter-token latency rises with context).  Speculative decoding, prompt caching,
  hidden reasoning and load-correlated batch non-invariance are all layered on top.
* **Tokenizer / usage** — real ``tiktoken`` token counts (or a custom single-digit-split
  tokenizer), additive chat-template and per-message overhead, cached-prefix accounting.
* **Decision (Jev) mode** — a joint softmax over supplied options with 2-decimal rounding
  and the documented confidence formula.

The adapter never raises on "API" errors: over-length prompts come back as a ``status=400``
response whose ``error`` text carries the limit (error texts are forensic evidence).

Nothing here is secret and nothing is written to disk; it is pure, deterministic
computation seeded by :attr:`SimConfig.determinism_seed`.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np

from forensics.core.adapters.base import Adapter
from forensics.core.types import (
    Capabilities, Chunk, DecisionAnswer, DecisionQuestion, DecisionRequest,
    DecisionResponse, GenRequest, GenResponse, Message, Timing, TokenLogprob,
    TopToken, Usage,
)

ArchFamily = Literal[
    "decoder_only", "encoder_only", "encoder_decoder", "ssm", "hybrid",
    "diffusion", "decision",
]
Sparsity = Literal["dense", "moe"]
PosEncoding = Literal["rope", "alibi", "learned", "nope"]
LogitDtype = Literal["fp32", "bf16", "fp16", "fp8"]

# --------------------------------------------------------------------------- constants
# Arithmetic intensity (FLOP per byte of HBM traffic) used to turn a memory-bandwidth
# number into a compute-throughput number, so that timing magnitudes land in a realistic
# tens-of-milliseconds range.  Loosely the roofline ridge point of a modern accelerator.
_ARITH_INTENSITY = 300.0
# Scale of the per-prompt hidden vector; sets the spread (std) of the synthesised logits.
_H_SCALE = 6.0
# Bytes per element on the various logit / KV dtypes.
_DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}
# Default number of vocabulary columns over which top-logprobs / logZ are computed.
_POOL_CAP = 2048


# --------------------------------------------------------------------------- dtype rounding
def to_bf16(x: np.ndarray | float) -> np.ndarray:
    """Round IEEE-754 values onto the bfloat16 grid (8-bit mantissa), round-to-nearest-even.

    Returned as float64 for convenient downstream maths; only the *value* is quantised.
    """
    f32 = np.asarray(x, dtype=np.float32)
    u = f32.view(np.uint32).astype(np.uint64)
    # round-to-nearest-even on the 16 low bits we are about to drop
    u = u + (0x7FFF + ((u >> 16) & 1))
    u = (u & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32).astype(np.float64)


def to_fp16(x: np.ndarray | float) -> np.ndarray:
    """Round values onto the IEEE float16 grid (10-bit mantissa)."""
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def to_fp8_e4m3(x: np.ndarray | float) -> np.ndarray:
    """Round values onto the fp8 E4M3 grid (3-bit mantissa, exp bias 7, max magnitude 448).

    A deliberately simple (but monotone and correctly-coarse) quantiser: within each binade
    the step is ``2**(exp-3)``.  Enough to reproduce the very coarse fp8 staircase / heavy
    tie-ing that P04 keys on; not a bit-exact hardware model of subnormals.
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.minimum(np.abs(x), 448.0)
    nz = ax > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log2(np.where(nz, ax, 1.0)))
    e = np.clip(e, -6, 8)                 # normal exponent range for E4M3
    step = np.power(2.0, e - 3)           # 3 mantissa bits
    q = np.round(ax / step) * step
    return sign * np.where(nz, q, 0.0)


def quantize_logits(x: np.ndarray | float, dtype: LogitDtype) -> np.ndarray:
    if dtype == "fp32":
        return np.asarray(x, dtype=np.float32).astype(np.float64)
    if dtype == "bf16":
        return to_bf16(x)
    if dtype == "fp16":
        return to_fp16(x)
    if dtype == "fp8":
        return to_fp8_e4m3(x)
    raise ValueError(f"unknown logit dtype {dtype!r}")


# --------------------------------------------------------------------------- tokenizer
class SimTokenizer:
    """Thin wrapper giving token *counts*, token strings and byte lists.

    Wraps a ``tiktoken`` encoding, or emulates a Qwen/Jev-style single-digit-split
    tokenizer (``tokenizer_name == "digit-split"``): digit runs become one token per digit,
    everything else is tokenised with the o200k base merges.
    """

    _DIGIT_BASE = 1_000_000  # synthetic ids for single-digit tokens (well above any base id)

    def __init__(self, name: str) -> None:
        import tiktoken

        self.name = name
        self.digit_split = name == "digit-split"
        base_name = "o200k_base" if self.digit_split else name
        self._enc = tiktoken.get_encoding(base_name)
        self.n_vocab = self._enc.n_vocab + (10 if self.digit_split else 0)

    # -- encoding -----------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.digit_split:
            return self._enc.encode(text, disallowed_special=())
        ids: list[int] = []
        for chunk in re.split(r"(\d)", text):   # capture group => each digit isolated
            if not chunk:
                continue
            if chunk.isdigit():                 # single digit
                ids.append(self._DIGIT_BASE + int(chunk))
            else:
                ids.extend(self._enc.encode(chunk, disallowed_special=()))
        return ids

    def count(self, text: str) -> int:
        return len(self.encode(text))

    # -- decoding / surface forms ------------------------------------------
    def token_bytes(self, tid: int) -> list[int]:
        if tid >= self._DIGIT_BASE:
            return [ord(str(tid - self._DIGIT_BASE))]
        try:
            return list(self._enc.decode_single_token_bytes(tid))
        except Exception:
            return []

    def token_str(self, tid: int) -> str:
        return bytes(self.token_bytes(tid)).decode("utf-8", errors="replace")

    def decode_ids(self, ids: Sequence[int]) -> str:
        return "".join(self.token_str(t) for t in ids)


# --------------------------------------------------------------------------- config
@dataclass
class SimConfig:
    """Known hidden architecture the probes must recover.

    Every field is either a directly-observable capability or a physical parameter that
    drives a synthesised side-channel.  Presets are provided by the ``sim_*`` factories.
    """

    # ---- core architecture ------------------------------------------------
    arch_family: ArchFamily = "decoder_only"
    hidden_size: int = 4096                  # d_model
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn_dim: int = 14336
    vocab_size: int = 128_256
    max_context: int = 131_072
    max_output: int = 4096
    # How an over-large max_tokens is handled. "reject_context" (vLLM / SGLang / most
    # serverless APIs): prompt + max_tokens > max_context -> 400 stating the context limit;
    # "clamp": silently cap generation at max_output (e.g. Fireworks).
    max_tokens_policy: str = "reject_context"
    sliding_window: int | None = None
    pos_encoding: PosEncoding = "rope"
    rope_theta: float = 500_000.0
    # ---- sparsity ---------------------------------------------------------
    sparsity: Sparsity = "dense"
    n_experts: int = 0
    active_experts: int = 0
    active_params_b: float = 8.0
    total_params_b: float = 8.0
    # ---- numerics / output ------------------------------------------------
    logit_dtype: LogitDtype = "bf16"
    # Real modern LMs (Llama, Qwen, Mistral, GPT-NeoX, DeepSeek, ...) use lm_head.bias=None.
    # With no bias the stacked logit matrix has rank exactly d (only the log-softmax per-row
    # offset remains, which row-centering removes). Set True to test P03 robustness to a
    # non-zero output bias (which adds one extra column dimension → recovered rank d+1).
    lm_head_bias: bool = False
    tokenizer_name: str = "o200k_base"
    chat_template_overhead: int = 7          # tokens per request from the template scaffold
    per_message_overhead: int = 4            # tokens per message wrapper
    # ---- interface capabilities ------------------------------------------
    expose_logprobs: bool = True
    expose_usage: bool = True               # False = provider hides the billing ledger
    max_top_logprobs: int = 20
    logit_bias_affects_logprobs: bool = True
    supports_streaming: bool = True
    supports_seed: bool = True
    supports_n: bool = True
    supports_temperature: bool = True
    supports_logit_bias: bool = True
    # ---- serving stack ----------------------------------------------------
    hidden_reasoning: bool = False
    reasoning_token_multiplier: float = 0.0
    speculative_decoding: bool = False
    prompt_caching: bool = False
    cache_block_tokens: int = 256
    batch_invariant: bool = True             # if False, add load-correlated logit jitter
    # ---- behaviour --------------------------------------------------------
    instruction_tuned: bool = True
    refusal_topics: list[str] = field(default_factory=list)
    knowledge_scale: float = 1.0             # larger => recalls rarer facts
    # ---- hardware / timing ------------------------------------------------
    server_gpu_bandwidth_gbps: float = 3350.0  # per-GPU HBM bandwidth (H100 ~ 3.35 TB/s)
    tensor_parallel: int = 1
    mfu: float = 0.5
    base_overhead_s: float = 0.02
    network_rtt_s: float = 0.03
    timing_jitter_s: float = 0.0             # gaussian std added to reported times
    determinism_seed: int = 12345
    # ---- decision (Jev) serving model ------------------------------------
    #: questions are isolated branches over the shared state (each question sees only the
    #: state + its own text). False = one concatenated prompt (siblings visible).
    decision_isolated: bool = True
    #: number of Python worker processes behind the gateway; >1 randomises the key order of
    #: probability maps per request (per-process PYTHONHASHSEED), like the real Jev.
    decision_workers: int = 1
    #: emit a Cloudflare -> Envoy -> gateway header set (server, cf-ray,
    #: x-envoy-upstream-service-time, x-typesafe-request-id) on decision responses.
    edge_headers: bool = False
    # ---- pluggable responders --------------------------------------------
    #: callables (rendered_prompt) -> reply text | None; first non-None wins
    responders: list[Callable[[str], str | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sparsity == "moe" and self.n_experts and not self.active_experts:
            self.active_experts = max(1, self.n_experts // 8)

    @property
    def head_dim(self) -> int:
        return max(1, self.hidden_size // max(1, self.n_heads))


# --------------------------------------------------------------------------- adapter
class SimulatedAdapter(Adapter):
    """Deterministic simulator whose observables encode :class:`SimConfig`."""

    def __init__(self, config: SimConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or SimConfig()
        self.kind = "decision" if self.cfg.arch_family == "decision" else "generative"
        self.name = f"simulated-{self.cfg.arch_family}"
        self.tok = SimTokenizer(self.cfg.tokenizer_name)
        self._emb_scale = 1.0 / math.sqrt(self.cfg.hidden_size)
        self._row_cache: dict[int, np.ndarray] = {}   # tid -> [embedding(d), bias]
        self._pool_ids: list[int] = list(range(min(self.cfg.vocab_size, _POOL_CAP)))
        self._pool_E: np.ndarray | None = None        # lazily built (|pool| x d)
        self._pool_b: np.ndarray | None = None
        # prompt-cache: set of block-boundary prefix hashes already "seen"
        self._cache_prefixes: set[str] = set()
        # per-adapter RNG for load jitter / non-determinism (advances every call)
        self._jitter_rng = np.random.default_rng(self.cfg.determinism_seed ^ 0xB16B00B5)
        # separate stream for decision-gateway effects (worker choice / request ids) so the
        # timing/logit jitter sequence is unchanged by them
        self._gw_rng = np.random.default_rng(self.cfg.determinism_seed ^ 0x5EED6A7E)
        self._call_count = 0

    # ==================================================================== low-rank core
    def _row(self, tid: int) -> np.ndarray:
        """Deterministic ``[embedding(d) , bias]`` for a token id (cached)."""
        row = self._row_cache.get(tid)
        if row is None:
            seed = ((self.cfg.determinism_seed & 0xFFFFFFFF) * 2654435761
                    ^ ((int(tid) + 1) * 40503)) & 0xFFFFFFFFFFFF
            r = np.random.default_rng(seed)
            emb = r.standard_normal(self.cfg.hidden_size) * self._emb_scale
            # No lm_head bias by default (matches real LMs); optional for robustness testing.
            bias = float(r.standard_normal()) * 0.5 if self.cfg.lm_head_bias else 0.0
            row = np.concatenate([emb, [bias]])
            self._row_cache[tid] = row
        return row

    def _embed(self, ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(E, bias)`` for the requested ids: ``E`` is ``len x d``, ``bias`` is ``len``."""
        rows = np.stack([self._row(int(t)) for t in ids])
        return rows[:, :-1], rows[:, -1]

    def _hidden(self, prompt: str) -> np.ndarray:
        """Per-prompt hidden vector ``h in R^d`` (stable across processes/runs)."""
        digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little")
        return np.random.default_rng(seed).standard_normal(self.cfg.hidden_size) * _H_SCALE

    def synthesize_logits(
        self,
        prompt: str,
        token_ids: Sequence[int] | None = None,
        apply_dtype: bool = True,
    ) -> np.ndarray:
        """Ground-truth logit vector ``z = E@h + bias`` for a prompt over ``token_ids``.

        This is the *clean* physical quantity (no log-softmax offset, no load jitter): it is
        what the internal state actually is, and what P03/P04 tests validate against.  The
        matrix stacked over many prompts has rank exactly :attr:`SimConfig.hidden_size`.
        """
        if token_ids is None:
            token_ids = self._pool_ids
        h = self._hidden(prompt)
        E, bias = self._embed(list(token_ids))
        z = E @ h + bias
        return quantize_logits(z, self.cfg.logit_dtype) if apply_dtype else z

    def _pool_logits(self, h: np.ndarray, extra_ids: Sequence[int] = ()) -> tuple[list[int], np.ndarray]:
        """Quantised logits over the shared pool (plus any extra ids), with optional jitter."""
        if self._pool_E is None:
            self._pool_E, self._pool_b = self._embed(self._pool_ids)
        ids = list(self._pool_ids)
        z = self._pool_E @ h + self._pool_b
        extra = [t for t in extra_ids if t not in self._pool_ids]
        if extra:
            E2, b2 = self._embed(extra)
            z = np.concatenate([z, E2 @ h + b2])
            ids = ids + extra
        z = quantize_logits(z, self.cfg.logit_dtype)
        if not self.cfg.batch_invariant:
            # load-correlated jitter: magnitude tracks a drifting synthetic load level
            load = 0.5 + 0.5 * self._jitter_rng.random()
            z = z + self._jitter_rng.standard_normal(z.shape) * 0.02 * load
        return ids, z

    # ==================================================================== capabilities
    def static_capabilities(self) -> Capabilities:
        c = self.cfg
        gen = self.kind == "generative"
        return Capabilities(
            kind=self.kind,                                   # type: ignore[arg-type]
            chat=True,
            completion=gen,
            logprobs=c.expose_logprobs if gen else False,
            max_top_logprobs=c.max_top_logprobs if (gen and c.expose_logprobs) else None,
            logprob_bytes=c.expose_logprobs if gen else False,
            logit_bias=c.supports_logit_bias if gen else False,
            logit_bias_affects_logprobs=c.logit_bias_affects_logprobs if gen else None,
            logit_bias_max=100.0 if (gen and c.supports_logit_bias) else None,
            streaming=c.supports_streaming if gen else False,
            stream_usage=(c.supports_streaming and c.expose_usage) if gen else False,
            temperature=c.supports_temperature if gen else False,
            top_p=c.supports_temperature if gen else False,
            seed=c.supports_seed,
            n=c.supports_n if gen else False,
            stop=gen,
            system_role=True,
            assistant_prefill=gen,
            usage_prompt_tokens=c.expose_usage,
            usage_cached_tokens=c.prompt_caching and c.expose_usage,
            usage_reasoning_tokens=c.hidden_reasoning and c.expose_usage,
            reasoning_effort=c.hidden_reasoning,
            system_fingerprint=True,
            server_timing_header="x-sim-upstream-ms",
            max_tokens_param="max_tokens",
            token_ids_known=True,
            notes=[f"simulated {c.arch_family} d={c.hidden_size} L={c.n_layers}"],
        )

    # ==================================================================== timing model
    def _compute_flops(self) -> float:
        c = self.cfg
        bw = c.server_gpu_bandwidth_gbps * 1e9 * c.tensor_parallel     # bytes/s
        return bw * c.mfu * _ARITH_INTENSITY                            # ~ FLOP/s

    def _bandwidth(self) -> float:
        return self.cfg.server_gpu_bandwidth_gbps * 1e9 * self.cfg.tensor_parallel

    def _prefill_s(self, p: int) -> float:
        """Prefill time for ``p`` prompt tokens: linear (params) + quadratic (attention)."""
        c = self.cfg
        comp = self._compute_flops()
        active = c.active_params_b * 1e9
        t = (2.0 * active / comp) * p                                   # dense matmuls
        # attention (QK^T + AV): quadratic in p, ~ 4 * L * d FLOP per token-pair
        p_eff = p
        quad = True
        if c.arch_family in ("ssm",):
            quad = False                                               # constant-state models
        elif c.sliding_window is not None:
            p_eff = min(p, c.sliding_window)                            # cost saturates at window
        elif c.arch_family == "hybrid":
            p_eff = int(p * 0.5)                                       # only some layers are full-attn
        if quad:
            t += (4.0 * c.n_layers * c.hidden_size / comp) * (p_eff ** 2)
        return t

    def _kv_bytes_per_token(self) -> float:
        c = self.cfg
        if c.arch_family == "ssm":
            return 0.0                                                 # constant-size recurrent state
        b = _DTYPE_BYTES.get(c.logit_dtype, 2)
        return 2.0 * c.n_kv_heads * c.head_dim * c.n_layers * b        # K and V

    def _decode_token_s(self, ctx: int) -> float:
        """Per-token decode time at context length ``ctx`` (weights + KV read)."""
        c = self.cfg
        bw = self._bandwidth()
        wbytes = c.active_params_b * 1e9 * _DTYPE_BYTES.get(c.logit_dtype, 2)
        return (wbytes + self._kv_bytes_per_token() * ctx) / bw

    # ==================================================================== prompt cache
    def _cache_lookup_and_fill(self, ids: list[int]) -> int:
        """Return #cached prompt tokens (matched leading blocks) and record this prefix."""
        if not self.cfg.prompt_caching:
            return 0
        blk = max(1, self.cfg.cache_block_tokens)
        n_blocks = len(ids) // blk
        cached_blocks = 0
        for k in range(1, n_blocks + 1):
            sig = hashlib.blake2b(
                np.asarray(ids[: k * blk], dtype=np.int64).tobytes(), digest_size=8
            ).hexdigest()
            if sig in self._cache_prefixes:
                cached_blocks = k
            else:
                self._cache_prefixes.add(sig)
        return cached_blocks * blk

    # ==================================================================== rendering
    @staticmethod
    def _render(req: GenRequest) -> tuple[str, list[Message]]:
        if req.messages:
            msgs = req.messages
        elif req.prompt is not None:
            msgs = [Message("user", req.prompt)]
        else:
            msgs = [Message("user", "")]
        text = "\n".join(f"{m.role}: {m.content}" for m in msgs)
        return text, list(msgs)

    def _prompt_tokens(self, req: GenRequest, msgs: list[Message]) -> int:
        body = sum(self.tok.count(m.content) for m in msgs)
        if req.messages is not None:
            return self.cfg.chat_template_overhead + self.cfg.per_message_overhead * len(msgs) + body
        return body

    # ==================================================================== responders
    def register_responder(self, fn: Callable[[str], str | None]) -> None:
        """Register a callable ``(rendered_prompt) -> reply | None`` used by :meth:`generate`."""
        self.cfg.responders.append(fn)

    def _responder_text(self, rendered: str) -> str | None:
        for fn in self.cfg.responders:
            try:
                out = fn(rendered)
            except Exception:
                out = None
            if out is not None:
                return out
        return None

    def _refuses(self, rendered: str) -> bool:
        low = rendered.lower()
        return any(t.lower() in low for t in self.cfg.refusal_topics)

    def _synth_ids(self, rendered: str, max_new: int,
                   logit_bias: dict[int, float] | None = None) -> list[int]:
        """Fallback sampling: the greedy (biased) argmax token ids over the pool.

        ``logit_bias`` steers the *sampled* tokens regardless of whether the config exposes
        the bias in reported logprobs (faithful to APIs where bias affects sampling only):
        a large positive bias on a token id makes that token the argmax, so it is emitted at
        position 0. These ids are used verbatim as the completion (no decode/strip/re-encode
        round trip), so a dominant bias is not lost — which is what a real post-bias API does.
        """
        h = self._hidden(rendered)
        k = min(max_new, 12)
        if k <= 0:
            return []
        ids, z = self._pool_logits(h, extra_ids=list(logit_bias or ()))
        z = np.asarray(z, dtype=np.float64)
        if logit_bias:
            for j, tid in enumerate(ids):
                if tid in logit_bias:
                    z[j] += float(logit_bias[tid])
        order = np.argsort(z)[::-1][:k]
        return [int(ids[i]) for i in order]

    def _synth_text(self, rendered: str, max_new: int,
                    logit_bias: dict[int, float] | None = None) -> str:
        """Fallback reply text (deterministic, real token strings) from :meth:`_synth_ids`."""
        ids = self._synth_ids(rendered, max_new, logit_bias)
        if not ids:
            return ""
        return self.tok.decode_ids(ids).strip() or "ok"

    # ==================================================================== generate
    async def generate(self, req: GenRequest) -> GenResponse:  # noqa: C901 (faithful model)
        self._call_count += 1
        c = self.cfg
        rendered, msgs = self._render(req)
        prompt_tokens = self._prompt_tokens(req, msgs)

        # ---- context-length enforcement (error text carries the limit) ----
        if prompt_tokens > c.max_context:
            return GenResponse(
                status=400,
                error=(f"This model's maximum context length is {c.max_context} tokens, "
                       f"however your request used {prompt_tokens} tokens."),
                error_body={"type": "invalid_request_error", "param": "messages",
                            "limit": c.max_context, "requested": prompt_tokens},
                model_reported=f"sim-{c.arch_family}",
                headers=self._headers(server_ms=2.0),
                timing=Timing(t_first_byte=c.network_rtt_s, t_end=c.network_rtt_s,
                              server_ms=2.0),
            )

        if c.max_tokens_policy == "reject_context" and prompt_tokens + req.max_tokens > c.max_context:
            return GenResponse(
                status=400,
                error=(f"This model's maximum context length is {c.max_context} tokens. "
                       f"However, you requested {prompt_tokens + req.max_tokens} tokens "
                       f"({prompt_tokens} in the messages, {req.max_tokens} in the completion)."),
                error_body={"type": "invalid_request_error", "param": "max_tokens",
                            "limit": c.max_context,
                            "requested": prompt_tokens + req.max_tokens},
                model_reported=f"sim-{c.arch_family}",
                headers=self._headers(server_ms=2.0),
                timing=Timing(t_first_byte=c.network_rtt_s, t_end=c.network_rtt_s,
                              server_ms=2.0),
            )

        # ---- decide the reply text ----------------------------------------
        max_tokens = max(1, min(req.max_tokens, c.max_output))
        if self._refuses(rendered):
            text = "I'm sorry, but I can't help with that."
            out_ids = self.tok.encode(text)[:max_tokens]
        else:
            rtext = self._responder_text(rendered)
            if rtext is not None:
                text = rtext
                out_ids = self.tok.encode(text)[:max_tokens]
            else:
                # Faithful fallback sampling: emit the exact (biased) argmax ids so that a
                # dominant logit_bias is genuinely SAMPLED at position 0 (as a real post-bias
                # API does) rather than lost to a decode/strip/re-encode round trip.
                out_ids = self._synth_ids(rendered, max_tokens, req.logit_bias)[:max_tokens]
                text = self.tok.decode_ids(out_ids).strip() or "ok"

        finish = "stop" if len(out_ids) < max_tokens else "length"
        if req.stop:
            for s in req.stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    out_ids = self.tok.encode(text)[:max_tokens]
                    finish = "stop"
                    break
        visible = len(out_ids)
        reasoning = int(round(c.reasoning_token_multiplier * visible)) if c.hidden_reasoning else 0

        # ---- per-token logprobs (only when requested / exposed) -----------
        tokens: list[TokenLogprob] | None = None
        if req.logprobs and c.expose_logprobs:
            tokens = self._logprobs_for(rendered, out_ids, req)

        # ---- usage --------------------------------------------------------
        cached = self._cache_lookup_and_fill(self.tok.encode(rendered))
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=visible + reasoning,
            total_tokens=prompt_tokens + visible + reasoning,
            cached_tokens=cached if c.prompt_caching else None,
            reasoning_tokens=reasoning if c.hidden_reasoning else None,
            raw={"visible_tokens": visible},
        ) if c.expose_usage else Usage()

        # ---- timing -------------------------------------------------------
        timing = self._gen_timing(prompt_tokens, cached, reasoning, out_ids,
                                   stream=bool(req.stream and c.supports_streaming))

        # ---- choices (n>1) ------------------------------------------------
        choices = [text]
        if req.n and req.n > 1 and c.supports_n:
            choices = [text] + [self._synth_text(rendered + f"\x1f{i}", max_tokens)
                                for i in range(1, req.n)]

        return GenResponse(
            text=text,
            choices_text=choices,
            tokens=tokens,
            finish_reason=finish,
            usage=usage,
            model_reported=f"sim-{c.arch_family}",
            system_fingerprint=f"sim_fp_{c.determinism_seed:x}",
            headers=self._headers(server_ms=(timing.server_ms or 0.0)),
            timing=timing,
            status=200,
        )

    def _logprobs_for(self, rendered: str, out_ids: list[int], req: GenRequest) -> list[TokenLogprob]:
        c = self.cfg
        k = min(req.top_logprobs or 0, c.max_top_logprobs) if req.top_logprobs else 0
        bias = req.logit_bias or {}
        toks: list[TokenLogprob] = []
        prefix_ids: list[int] = []
        for i, tid in enumerate(out_ids):
            h = self._hidden(rendered + "\x1f" + self.tok.decode_ids(prefix_ids) + f"#{i}")
            extra = [tid] + [t for t in bias]
            ids, z = self._pool_logits(h, extra_ids=extra)
            z = np.asarray(z, dtype=np.float64)
            biased = z.copy()
            if bias:
                for j, tk in enumerate(ids):
                    if tk in bias:
                        biased[j] += float(bias[tk])
            # reported distribution: biased iff the config says bias affects logprobs
            rep = biased if (bias and c.logit_bias_affects_logprobs) else z
            logZ = float(_logsumexp(rep))
            idpos = ids.index(tid)
            lp = float(rep[idpos] - logZ)
            top: list[TopToken] = []
            if k:
                order = np.argsort(rep)[::-1][:k]
                top = [TopToken(token=self.tok.token_str(ids[o]),
                                logprob=float(rep[o] - logZ),
                                bytes=self.tok.token_bytes(ids[o]),
                                token_id=int(ids[o])) for o in order]
            toks.append(TokenLogprob(token=self.tok.token_str(tid), logprob=lp,
                                     bytes=self.tok.token_bytes(tid), token_id=int(tid),
                                     top=top))
            prefix_ids.append(tid)
        return toks

    def _gen_timing(self, prompt_tokens: int, cached: int, reasoning: int,
                    out_ids: list[int], stream: bool) -> Timing:
        c = self.cfg
        billable_prefill = max(1, prompt_tokens - cached)
        prefill = self._prefill_s(billable_prefill)
        if cached:
            prefill += 0.02 * self._prefill_s(cached)                  # cache lookup is ~free

        # speculative decoding: predictable (low-entropy / repetitive) text decodes faster
        speedup = 1.0
        if c.speculative_decoding and out_ids:
            uniq = len(set(out_ids)) / len(out_ids)
            if uniq < 0.5:
                speedup = 3.0                                          # most drafts accepted

        # reasoning tokens are generated (silently) before the first visible token
        ctx = prompt_tokens
        reason_s = 0.0
        for _ in range(reasoning):
            reason_s += self._decode_token_s(ctx) / speedup
            ctx += 1

        # visible decode
        chunks: list[Chunk] = []
        base = c.base_overhead_s + prefill + reason_s
        t_cursor = base
        decode_total = 0.0
        i = 0
        n = len(out_ids)
        while i < n:
            burst = 3 if (speedup > 1.0 and i + 3 <= n) else 1        # spec-decode bursts
            dt = 0.0
            for _j in range(burst):
                dt += self._decode_token_s(ctx) / speedup
                ctx += 1
            decode_total += dt
            t_cursor += dt
            i += burst
            if stream:
                text_piece = self.tok.decode_ids(out_ids[i - burst:i])
                chunks.append(Chunk(t=self._net(t_cursor), text=text_piece, n_tokens=burst))

        server_s = base + decode_total
        # first visible token arrives after prefill + reasoning + its own decode step
        first_tok_server = base + (self._decode_token_s(prompt_tokens + reasoning) / speedup)

        t_first_byte = self._net(c.base_overhead_s + prefill * 0.1)
        t_first_token = self._net(first_tok_server)
        t_end = self._net(server_s)
        server_ms = (server_s + self._jit()) * 1000.0
        return Timing(
            t_first_byte=max(0.0, t_first_byte),
            t_first_token=max(t_first_byte, t_first_token) if out_ids else None,
            t_end=max(t_first_token, t_end),
            chunks=chunks if stream else [],
            server_ms=max(0.0, server_ms),
            wall_start=None,
        )

    def _net(self, server_time: float) -> float:
        return self.cfg.network_rtt_s + server_time + self._jit()

    def _jit(self) -> float:
        if self.cfg.timing_jitter_s <= 0:
            return 0.0
        return abs(float(self._jitter_rng.normal(0.0, self.cfg.timing_jitter_s)))

    def _headers(self, server_ms: float) -> dict[str, str]:
        return {
            "x-sim-arch": self.cfg.arch_family,
            "x-sim-upstream-ms": f"{server_ms:.2f}",
            "x-sim-tensor-parallel": str(self.cfg.tensor_parallel),
            "content-type": "application/json",
        }

    # ==================================================================== decide (Jev)
    async def decide(self, req: DecisionRequest) -> DecisionResponse:  # noqa: C901
        c = self.cfg
        if self.kind != "decision":
            return await super().decide(req)  # raises NotImplementedError for non-decision sims
        self._call_count += 1

        # ---- gateway validation (FastAPI + Pydantic-v2 behind Envoy, like Jev) ----------
        # Validation never reaches the "GPU": answered in a few ms of upstream time.
        extra_keys = sorted(k for k in (req.extra or {}) if k != "model")
        if extra_keys:
            # extra="forbid" on the top-level model; message sanitized (real Jev: 400)
            return self._gw_error(400, {"error_type": "api_usage_error",
                                        "message": "Invalid request."})
        if req.state is None or not isinstance(req.state, (str, dict, list)):
            if req.state is None:
                detail = [{"type": "missing", "loc": ["body", "state"],
                           "msg": "Field required", "input": None}]
            else:
                detail = [
                    {"type": "string_type", "loc": ["body", "state", "str"],
                     "msg": "Input should be a valid string", "input": req.state},
                    {"type": "dict_type", "loc": ["body", "state", "dict[any,any]"],
                     "msg": "Input should be a valid dictionary", "input": req.state},
                    {"type": "list_type", "loc": ["body", "state", "list[any]"],
                     "msg": "Input should be a valid list", "input": req.state},
                ]
            return self._gw_error(422, detail)
        if not req.questions:
            return self._gw_error(422, [{
                "type": "too_short", "loc": ["body", "questions"],
                "msg": "Dictionary should have at least 1 item after validation, not 0",
                "input": {}, "ctx": {"field_type": "Dictionary", "min_length": 1,
                                     "actual_length": 0}}])

        state_text = req.state if isinstance(req.state, str) else _stringify(req.state)
        state_ids = self.tok.encode(state_text)
        state_tokens = len(state_ids)
        template = c.chat_template_overhead

        if state_tokens + template > c.max_context:
            return DecisionResponse(
                status=400,
                error=(f"max_tokens_exceeded: state uses {state_tokens} tokens, "
                       f"maximum context length is {c.max_context} tokens."),
                model_reported=f"sim-{c.arch_family}",
                headers=self._headers(server_ms=3.0),
                timing=Timing(t_first_byte=c.network_rtt_s, t_end=c.network_rtt_s, server_ms=3.0),
            )

        # which worker process serves this request (per-process hash seed => key order)
        worker = int(self._gw_rng.integers(max(1, c.decision_workers)))

        answers: dict[str, DecisionAnswer] = {}
        input_tokens = template + state_tokens
        output_tokens = 2
        q_texts = {k: _stringify(q.instructions) for k, q in req.questions.items()}
        for key, q in req.questions.items():
            input_tokens += c.per_message_overhead + self.tok.count(q_texts[key])
            if c.decision_isolated:
                visible = state_text          # branch sees the shared state + its own question
            else:                             # one concatenated prompt: siblings are visible
                visible = state_text + "\n" + "\n".join(
                    t for k2, t in q_texts.items() if k2 != key)
            ans, in_add, out_add = self._answer_question(visible, key, q)
            if c.decision_workers > 1 and ans.probabilities:
                ans.probabilities = _worker_order(ans.probabilities, worker)
            input_tokens += in_add
            output_tokens += out_add
            answers[key] = ans

        # timing: single shared-state prefill (minus any cached prefix blocks) + cheap
        # per-question branches attending to that prefix
        cached = self._cache_lookup_and_fill(state_ids) if c.prompt_caching else 0
        prefill = self._prefill_s(state_tokens + template)
        if cached:
            prefill = max(0.0, prefill - self._prefill_s(cached))
        branch = 0.0006 * len(req.questions)
        server_s = c.base_overhead_s + prefill + branch
        timing = Timing(
            t_first_byte=self._net(c.base_overhead_s),
            t_end=self._net(server_s),
            server_ms=(server_s + self._jit()) * 1000.0,
        )
        return DecisionResponse(
            answers=answers,
            usage=Usage(prompt_tokens=input_tokens, completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        raw={"input_tokens": input_tokens, "output_tokens": output_tokens}),
            model_reported=f"sim-{c.arch_family}",
            headers=self._decision_headers(server_ms=(timing.server_ms or 0.0)),
            timing=timing,
            status=200,
        )

    def _decision_headers(self, server_ms: float) -> dict[str, str]:
        h = self._headers(server_ms)
        if self.cfg.edge_headers:
            rid = "".join(f"{int(x):02x}" for x in self._gw_rng.integers(0, 256, 16))
            h.update({
                "server": "cloudflare",
                "cf-ray": f"{rid[:16]}-SIM",
                "x-envoy-upstream-service-time": str(int(round(server_ms))),
                "x-typesafe-request-id": f"req_{rid}",
            })
        return h

    def _gw_error(self, status: int, detail: Any) -> DecisionResponse:
        """A gateway validation error (FastAPI/Pydantic-v2 422 or sanitized 400)."""
        c = self.cfg
        ms = 3.0
        msg = (detail.get("message") if isinstance(detail, dict)
               else "; ".join(str(d.get("msg")) for d in detail if isinstance(d, dict)))
        return DecisionResponse(
            status=status, error=f"HTTP {status}: {msg}", error_body=detail,
            raw={"detail": detail}, model_reported=None,
            headers=self._decision_headers(server_ms=ms),
            timing=Timing(t_first_byte=c.network_rtt_s, t_end=c.network_rtt_s, server_ms=ms),
        )

    def _answer_question(self, state_text: str, key: str,
                         q: DecisionQuestion) -> tuple[DecisionAnswer, int, int]:
        """Return (answer, input_token_add, output_token_add) for one question."""
        h = self._hidden(state_text + "\x1f" + _stringify(q.instructions))
        if q.type == "choice":
            crit = q.criteria if isinstance(q.criteria, dict) else {}
            labels = list(crit.keys()) or ["true"]
            in_add = sum(self.tok.count(l) + self.tok.count(_stringify(crit.get(l))) + 2
                         for l in labels)
            probs = self._option_softmax(h, labels)
            rounded = _round2_normalize(probs)
            best = max(rounded, key=rounded.get)
            n = len(labels)
            pmax = max(probs.values())
            conf = (n * pmax - 1.0) / (n - 1.0) if n > 1 else 1.0
            conf = float(min(1.0, max(0.0, conf)))
            out_add = 18 + sum(self.tok.count(l) + self.tok.count(f"{rounded[l]:.2f}") + 2
                               for l in labels)
            return (DecisionAnswer(type="choice", value=best, confidence=conf,
                                   probabilities=rounded, raw={"unrounded": probs}),
                    in_add, out_add)
        if q.type == "score":
            levels = q.criteria if isinstance(q.criteria, list) else ["0"]
            names = [str(x) for x in levels]
            in_add = sum(self.tok.count(_stringify(x)) + 2 for x in levels)
            probs_list = self._level_softmax(h, names)
            probs = {str(i): p for i, p in enumerate(probs_list)}
            rounded = _round2_normalize(probs)
            score = float(sum(i * probs_list[i] for i in range(len(probs_list))))
            conf = _ordinal_confidence(probs_list)
            legend = {str(i): names[i] for i in range(len(names))}
            return (DecisionAnswer(type="score", value=round(score, 2), confidence=conf,
                                   probabilities=rounded, legend=legend,
                                   raw={"unrounded": probs}),
                    in_add, 17)
        # noul
        crit = q.criteria if isinstance(q.criteria, dict) else {}
        in_add = (self.tok.count(_stringify(crit.get("true"))) +
                  self.tok.count(_stringify(crit.get("false")))) if crit else 0
        p_true = self._option_softmax(h, ["true", "false"])["true"]
        # Minimal literal grounding: a noul that asks about a quoted string ('X' / "X") is
        # answered by whether X is visible to this question's branch (the state, plus the
        # sibling questions only when branches are NOT isolated). The quote inside the
        # question's own text never counts - that is exactly the sibling-secret design.
        quoted = _quoted_terms(_stringify(q.instructions))
        if quoted:
            seen = all(t.lower() in state_text.lower() for t in quoted)
            wobble = 0.03 * math.tanh(float(h[0]) / _H_SCALE)
            p_true = (0.93 + wobble) if seen else (0.04 + abs(wobble))
        return (DecisionAnswer(type="noul", value=round(float(p_true), 2),
                               raw={"unrounded": p_true}),
                in_add, 18)

    def _option_softmax(self, h: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
        """Joint softmax over option scores; options interact (choice-set / IIA violation)."""
        raw = np.array([float(self._embed([_str_id(l)])[0][0] @ h) for l in labels])
        coupled = raw - 0.15 * raw.mean()                 # scores depend on the whole set
        p = _softmax(coupled)
        return {l: float(p[i]) for i, l in enumerate(labels)}

    def _level_softmax(self, h: np.ndarray, names: Sequence[str]) -> list[float]:
        raw = np.array([float(self._embed([_str_id(n)])[0][0] @ h) for n in names])
        return list(map(float, _softmax(raw)))


# --------------------------------------------------------------------------- helpers
def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _round2_normalize(probs: dict[str, float]) -> dict[str, float]:
    """Round to 2 decimals (Jev style); nudge the max so the reported map still sums to 1.00."""
    r = {k: round(float(v), 2) for k, v in probs.items()}
    diff = round(1.0 - sum(r.values()), 2)
    if r and abs(diff) >= 0.01:
        kmax = max(r, key=r.get)
        r[kmax] = round(r[kmax] + diff, 2)
    return r


def _ordinal_confidence(probs: list[float]) -> float:
    """Score confidence ~ 1 - SD/SD_uniform (an ordinal-dispersion statistic)."""
    n = len(probs)
    if n <= 1:
        return 1.0
    idx = np.arange(n)
    p = np.asarray(probs, dtype=np.float64)
    mean = float((idx * p).sum())
    var = float((p * (idx - mean) ** 2).sum())
    sd_uniform = math.sqrt((n * n - 1) / 12.0)
    return float(min(1.0, max(0.0, 1.0 - math.sqrt(var) / sd_uniform))) if sd_uniform else 1.0


def _quoted_terms(text: str) -> list[str]:
    """Quoted literals ('X' or "X", >= 3 chars) inside a question's instructions."""
    pat = r"(?<!\w)'([^'\n]{3,80})'(?!\w)|(?<!\w)\"([^\"\n]{3,80})\"(?!\w)"
    return [a or b for a, b in re.findall(pat, text or "")]


def _worker_order(probs: dict[str, float], worker: int) -> dict[str, float]:
    """Re-key a probability map in a per-worker hash order (per-process PYTHONHASHSEED)."""
    def key(k: str) -> bytes:
        return hashlib.blake2b(f"{worker}|{k}".encode("utf-8"), digest_size=8).digest()
    return {k: probs[k] for k in sorted(probs, key=key)}


def _str_id(s: str) -> int:
    """Stable synthetic token id for an option/level surface form."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


def _stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    import json
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- presets
def sim_decoder_dense() -> SimConfig:
    """Dense decoder-only transformer, Llama-3-8B-ish, pre-mitigation logprobs."""
    return SimConfig(
        arch_family="decoder_only", hidden_size=4096, n_layers=32, n_heads=32,
        n_kv_heads=8, ffn_dim=14336, vocab_size=128_256, max_context=131_072,
        max_output=4096, sliding_window=None, pos_encoding="rope", rope_theta=500_000.0,
        sparsity="dense", active_params_b=8.0, total_params_b=8.0, logit_dtype="bf16",
        tokenizer_name="o200k_base", chat_template_overhead=7, per_message_overhead=4,
        expose_logprobs=True, max_top_logprobs=20, logit_bias_affects_logprobs=True,
        prompt_caching=True, cache_block_tokens=256, batch_invariant=True,
        server_gpu_bandwidth_gbps=3350.0, tensor_parallel=1, mfu=0.5,
    )


def sim_decoder_moe() -> SimConfig:
    """Sparse mixture-of-experts decoder (Mixtral/DeepSeek-ish), not batch-invariant."""
    return SimConfig(
        arch_family="decoder_only", hidden_size=4096, n_layers=32, n_heads=32,
        n_kv_heads=8, ffn_dim=14336, vocab_size=131_072, max_context=65_536,
        max_output=4096, pos_encoding="rope", rope_theta=1_000_000.0,
        sparsity="moe", n_experts=8, active_experts=2, active_params_b=13.0,
        total_params_b=47.0, logit_dtype="bf16", tokenizer_name="o200k_base",
        chat_template_overhead=7, per_message_overhead=4, expose_logprobs=True,
        max_top_logprobs=20, logit_bias_affects_logprobs=False,  # post-mitigation
        speculative_decoding=True, prompt_caching=True, cache_block_tokens=256,
        batch_invariant=False, server_gpu_bandwidth_gbps=3350.0, tensor_parallel=8, mfu=0.45,
    )


def sim_ssm() -> SimConfig:
    """State-space / linear model (Mamba-ish): linear prefill, flat ITL, no RoPE."""
    return SimConfig(
        arch_family="ssm", hidden_size=2560, n_layers=64, n_heads=1, n_kv_heads=1,
        ffn_dim=0, vocab_size=50_280, max_context=1_000_000, max_output=4096,
        sliding_window=None, pos_encoding="nope", rope_theta=0.0, sparsity="dense",
        active_params_b=2.8, total_params_b=2.8, logit_dtype="bf16",
        tokenizer_name="cl100k_base", chat_template_overhead=5, per_message_overhead=3,
        expose_logprobs=True, max_top_logprobs=10, logit_bias_affects_logprobs=True,
        prompt_caching=False, batch_invariant=True, server_gpu_bandwidth_gbps=3350.0,
        tensor_parallel=1, mfu=0.5,
    )


def sim_encoder() -> SimConfig:
    """Bidirectional encoder-only classifier (BERT-ish): short context, learned positions."""
    return SimConfig(
        arch_family="encoder_only", hidden_size=1024, n_layers=24, n_heads=16,
        n_kv_heads=16, ffn_dim=4096, vocab_size=30_522, max_context=512, max_output=1,
        sliding_window=None, pos_encoding="learned", rope_theta=0.0, sparsity="dense",
        active_params_b=0.34, total_params_b=0.34, logit_dtype="fp32",
        tokenizer_name="cl100k_base", chat_template_overhead=2, per_message_overhead=1,
        expose_logprobs=True, max_top_logprobs=5, logit_bias_affects_logprobs=True,
        instruction_tuned=False, supports_streaming=False, supports_n=False,
        prompt_caching=False, batch_invariant=True, server_gpu_bandwidth_gbps=2000.0,
        tensor_parallel=1, mfu=0.4,
    )


def sim_jev() -> SimConfig:
    """TypeSafe-Jev-like decision model: non-autoregressive readout, digit-split tokenizer."""
    return SimConfig(
        arch_family="decision", hidden_size=3584, n_layers=28, n_heads=28, n_kv_heads=4,
        ffn_dim=18944, vocab_size=152_064, max_context=65_536, max_output=1,
        sliding_window=None, pos_encoding="rope", rope_theta=1_000_000.0, sparsity="moe",
        n_experts=64, active_experts=8, active_params_b=10.0, total_params_b=70.0,
        logit_dtype="bf16", tokenizer_name="digit-split", chat_template_overhead=260,
        per_message_overhead=7, expose_logprobs=False, max_top_logprobs=0,
        logit_bias_affects_logprobs=False, supports_streaming=False, supports_seed=False,
        supports_n=False, supports_temperature=False, supports_logit_bias=False,
        prompt_caching=True, cache_block_tokens=256, batch_invariant=False,
        instruction_tuned=True, server_gpu_bandwidth_gbps=3350.0, tensor_parallel=8, mfu=0.5,
        decision_isolated=True, decision_workers=4, edge_headers=True,
    )
