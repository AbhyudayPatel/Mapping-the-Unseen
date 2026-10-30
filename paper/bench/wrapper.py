"""Anonymous OpenAI-compatible serving wrapper for ForensicsBench subjects.

One HTTP contract hides every subject: dense/SSM/MoE/MLA decoders, masked
encoders, encoder-decoders, dtype/quantisation variants, batched serving, and
router swaps.  The scanner only ever sees ``/v1/chat/completions`` responses.

Options (subject ``opts``)
--------------------------
compute_dtype : "fp32" (default) | "bf16" | "fp16"   -- model compute precision
quant         : "int8_dynamic"                          -- dynamic int8 Linear quant
wire_round    : "bf16" | "fp16"                         -- round reported logprobs
batch         : int > 1                                 -- batch concurrent requests
router        : True                                    -- alternate two checkpoints
logit_bias    : default True                            -- bias applies to position-0
                                                          logits (reported + sampled)
measurement_columns : always accepted (explicit diagnostic extension; the analysis
                keeps this richer-than-production regime separate)

The wrapper reports honest ``usage`` token counts and an ``x-processing-time``
header with measured server-side compute milliseconds.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def render(messages):
    """Adapter-owned rendering, identical for every subject (wrapper fact)."""
    return "\n".join(str(m.get("content", "")) for m in messages)


def _round_grid(x: torch.Tensor, mode: str | None) -> torch.Tensor:
    if mode == "bf16":
        return x.to(torch.bfloat16).float()
    if mode == "fp16":
        return x.half().float()
    return x


class ServedModel:
    """One loaded checkpoint with a uniform forward-logits interface."""

    def __init__(self, model, tok, kind: str, opts: dict | None = None):
        self.model, self.tok, self.kind = model, tok, kind
        self.opts = opts or {}
        mname = type(model).__name__.lower()
        self.ssm = "mamba" in mname or "rwkv" in mname

    # ------------------------------------------------------------------ forward
    def _masked_logits(self, ids):
        mask_id = getattr(self.tok, "mask_token_id", None) or 103
        with torch.inference_mode():
            return self.model(torch.tensor([ids + [mask_id]])).logits[0, -1, :].float()

    def first_token(self):
        t = (getattr(self.tok, "bos_token_id", None)
             or getattr(self.tok, "eos_token_id", None)
             or getattr(self.tok, "cls_token_id", None) or 0)
        return int(t)

    def new_session(self, ids):
        """A decode session that caches state across steps, exactly like real
        serving stacks (KV cache for attention decoders, conv/scan state for SSMs,
        encoder output + decoder KV for encoder-decoders)."""
        return DecodeSession(self, ids)


class DecodeSession:
    """Per-request decode state; ``next_logits`` feeds only NEW tokens."""

    def __init__(self, served: ServedModel, ids):
        self.served = served
        self.cache = None
        self.enc = None
        self.pending = list(ids) if served.kind != "encdec" else [0]
        if served.kind == "encdec":
            with torch.inference_mode():
                self.enc = served.model.encoder(torch.tensor([ids])).last_hidden_state

    def next_logits(self) -> torch.Tensor:
        m = self.served.model
        inp = torch.tensor([self.pending])
        self.pending = []
        with torch.inference_mode():
            if self.served.kind == "encdec":
                out = m(decoder_input_ids=inp, encoder_outputs=(self.enc,),
                        past_key_values=self.cache, use_cache=True)
                self.cache = out.past_key_values
            elif self.served.ssm:
                out = m(input_ids=inp, cache_params=self.cache, use_cache=True)
                self.cache = out.cache_params
            else:
                out = m(input_ids=inp, past_key_values=self.cache, use_cache=True)
                self.cache = out.past_key_values
        return out.logits[0, -1, :].float()

    def feed(self, tid: int) -> None:
        self.pending.append(int(tid))


class ArchitectureWrapper:
    """The anonymous contract.  One instance per served endpoint."""

    def __init__(self, served: ServedModel, served_b: ServedModel | None = None,
                 opts: dict | None = None):
        self.a = served
        self.b = served_b                      # router twin (optional)
        self.opts = opts or {}
        if served.kind == "masked":
            default_cap = 256
        elif served.kind == "encdec":
            default_cap = 2048
        else:
            # SSM prefill is O(L) and cheap enough to sweep past 16k tokens (the strong
            # linear-evidence gate); transformer prefill is O(L^2)-heavy, capped at 16k.
            default_cap = 24576 if served.ssm else 16384
        model_max = int(getattr(served.model.config, "max_position_embeddings",
                        getattr(served.model.config, "n_positions", 0)) or 0)
        if served.ssm or model_max <= 0:
            model_max = 1 << 30                      # constant-state: no positional cap
        if "mistral" in type(served.model).__name__.lower():
            model_max = 1 << 30                      # sliding-window: RoPE, no hard cap here
        self.cap = int(self.opts.get("cap", min(default_cap, model_max)))
        self.max_output = int(self.opts.get("max_output", 24 if served.kind != "masked" else 1))
        self.wire_round = self.opts.get("wire_round")
        self.support_bias = bool(self.opts.get("logit_bias", True))
        self.batch = int(self.opts.get("batch", 1))
        self.router = served_b is not None
        self.prefix_cache = bool(self.opts.get("prefix_cache", False))
        self.model_max_position = model_max
        self._cache_store: dict = {}
        self._rr = 0
        self._batch_q: asyncio.Queue | None = None
        self.app = FastAPI()
        self.app.post("/v1/chat/completions")(self.complete)

    # ------------------------------------------------------------------ routing
    def _pick(self) -> ServedModel:
        if not self.router:
            return self.a
        self._rr += 1
        return (self.a, self.b)[self._rr % 2]

    # ------------------------------------------------------------------ streaming
    async def _stream(self, b, m, seq, wanted):
        """SSE token stream with realistic per-token decode timing (cached state)."""
        import json as _json
        from fastapi.responses import StreamingResponse
        n = min(wanted, self.max_output)
        session = m.new_session(seq)

        t0 = time.perf_counter()

        async def gen():
            for _i in range(n):
                logits = session.next_logits()
                lp = torch.log_softmax(logits.double(), -1)
                tid = int(lp.argmax())
                piece = m.tok.decode([tid], clean_up_tokenization_spaces=False)
                session.feed(tid)
                # server-reported cumulative time at this token (Azure-style
                # latency_checkpoint equivalent), robust to client-side buffering
                yield "data: " + _json.dumps({
                    "choices": [{"index": 0, "delta": {"content": piece}}],
                    "t_server_ms": (time.perf_counter() - t0) * 1000.0}) + "\n\n"
                await asyncio.sleep(0)
            yield "data: " + _json.dumps({
                "choices": [],
                "usage": {"prompt_tokens": len(seq), "completion_tokens": n,
                          "total_tokens": len(seq) + n}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    # ------------------------------------------------------------------ endpoint
    async def complete(self, req: Request):
        b = await req.json()
        m = self._pick()
        t0 = time.perf_counter()
        ids = m.tok.encode(render(b.get("messages", [])), add_special_tokens=False)
        wanted = int(b.get("max_tokens", 1) or 1)
        if wanted > self.max_output:
            return JSONResponse({"error": {"message":
                f"max_tokens must be <= {self.max_output} output tokens"}}, 400)
        if len(ids) > self.cap:
            return JSONResponse({"error": {"message":
                f"maximum context length is {self.cap} tokens"}}, 400)
        if b.get("stream") and m.kind != "masked":
            return await self._stream(b, m, list(ids) or [m.first_token()], wanted)
        bias = b.get("logit_bias")
        if bias and not self.support_bias:
            return JSONResponse({"error": {"message": "logit_bias not supported"}}, 400)

        n = min(wanted, 1) if m.kind == "masked" else wanted
        k = min(int(b.get("top_logprobs", 5) or 5), 20)
        columns = b.get("measurement_columns")
        seq = list(ids) or [m.first_token()]
        pieces, rows = [], []
        bias_t = None
        if bias:
            try:
                nv = (m.model.get_output_embeddings().weight.shape[0]
                      if m.kind != "masked" else len(m.tok))
            except Exception:  # quantised head: fall back to config vocab
                nv = int(getattr(m.model.config, "vocab_size", 0) or len(m.tok))
            bias_t = torch.zeros(nv)
            for tid_, val in bias.items():
                bias_t[int(tid_)] = float(val)
        # full-prefix cache: a hit skips prefill entirely (serving condition)
        cache_key = tuple(seq) if (self.prefix_cache and m.kind == "causal") else None
        cached = self._cache_store.get(cache_key) if cache_key is not None else None
        if cached is not None:
            import copy as _copy
            session = m.new_session([])
            session.cache = _copy.deepcopy(cached[0])
            first_logits = cached[1].clone()
        else:
            session = m.new_session(seq)
            first_logits = None
        for _step in range(n):
            if m.kind == "masked":
                logits = m._masked_logits(seq)
            elif first_logits is not None:
                logits = first_logits
                first_logits = None
            else:
                logits = session.next_logits()
            if (cache_key is not None and cached is None and _step == 0
                    and len(seq) >= 32):
                import copy as _copy
                self._cache_store[cache_key] = (_copy.deepcopy(session.cache),
                                                logits.clone())
                if len(self._cache_store) > 64:
                    self._cache_store.pop(next(iter(self._cache_store)))
            if bias_t is not None:
                logits = logits + bias_t[: logits.shape[0]]
            lp = torch.log_softmax(logits.double(), -1)
            if self.wire_round:
                lp = _round_grid(lp.float(), self.wire_round).double()
            tid = int(lp.argmax())
            piece = m.tok.decode([tid], clean_up_tokenization_spaces=False)
            pieces.append(piece)
            if m.kind != "masked":
                session.feed(tid)
            if b.get("logprobs"):
                if columns is not None:
                    idx = torch.tensor([int(i) for i in columns], dtype=torch.long)
                    val = lp[idx]
                else:
                    val, idx = torch.topk(lp, k)
                rows.append({"token": piece, "token_id": tid,
                             "logprob": float(lp[tid]),
                             "top_logprobs": [
                                 {"token": m.tok.decode([int(i)],
                                     clean_up_tokenization_spaces=False),
                                  "token_id": int(i), "logprob": float(v)}
                                 for i, v in zip(idx, val)]})
        dt = (time.perf_counter() - t0) * 1000
        p_report = len(ids)
        if self.opts.get("ledger_inflate"):
            p_report = int(round(p_report * 1.3)) + 7
        return JSONResponse(
            {"model": "opaque-architecture",
             "choices": [{"index": 0,
                          "message": {"role": "assistant", "content": "".join(pieces)},
                          "finish_reason": "length",
                          "logprobs": {"content": rows} if rows else None}],
             "usage": {"prompt_tokens": p_report, "completion_tokens": len(pieces),
                       "total_tokens": p_report + len(pieces)}},
            headers={"x-processing-time": str(dt)})
