"""P19 — architecture-family discriminator (generative endpoints).

Separates the core network family behind a generative API — decoder-only vs
encoder-decoder vs encoder-only vs (masked/discrete) diffusion — using only black-box
generation behaviour and the server-time side channel:

* **Generation capability** (``P19.arch.generation``): can the endpoint emit more than one
  token of a free continuation at all?  An encoder classifier caps output at ~1 token and
  never continues a prompt; a decoder / SSM / encoder-decoder / diffusion LM generates.
  This is the cleanest encoder-vs-generative split and also pins ``output.head`` and
  ``train.objective``.  Two further reads sharpen it: whether latency scales with the
  number of *output* tokens (autoregressive decode) or is flat while still producing many
  tokens (diffusion / parallel refinement), and the streaming granularity.
* **Prefill curvature** (``P19.arch.prefill_scaling``): server time vs prompt length is
  linear for a constant-state SSM / linear-attention model and genuinely quadratic for
  full self-attention.  This is the decoder-vs-SSM split.  It only fires when the sweep
  actually reaches lengths where an O(len^2) term would be visible (otherwise a short
  context can masquerade as linear), so a short-context encoder contributes no spurious
  "linear" evidence here.

Evidence informs ``arch.family``, ``train.objective``, ``output.head`` and
``serving.streaming_granularity``.  Both probes go through ``ctx.gen`` only, convert
``timing.server_ms`` (milliseconds) to seconds, handle ``resp.ok == False`` (an
over-length reply is itself a short-context signal), randomise prompt content to defeat
prompt caching, and stay under ~30 requests in ``ctx.fast()``.
"""
from __future__ import annotations

import warnings

import numpy as np

from forensics.core.types import GenRequest
from forensics.probes.base import (
    BudgetExhausted, Evidence, Probe, ProbeContext, ProbeResult, register,
)
from forensics.stats.regression import poly_model_select

_ARCH = ("decoder_only", "encoder_only", "encoder_decoder", "ssm_or_linear",
         "hybrid_attn_ssm", "diffusion", "non_neural")

_FLOOR = 0.03
_WORDS = [f"lorem{i:04d}" for i in range(30000)]


def _server_s(resp) -> float | None:
    ms = resp.timing.server_ms if resp.timing else None
    if ms is None:
        # fall back to end-to-end wall time if no server header
        te = resp.timing.t_end if resp.timing else None
        return float(te) if te is not None else None
    return float(ms) / 1000.0


def _n_generated(resp) -> int:
    u = resp.usage
    if u and u.completion_tokens is not None:
        return max(0, int(u.completion_tokens) - int(u.reasoning_tokens or 0))
    if resp.tokens:
        return len(resp.tokens)
    return len((resp.text or "").split())


def _rand_prompt(ctx: ProbeContext, n_words: int, nonce: int) -> str:
    idx = ctx.rng.integers(0, len(_WORDS), size=n_words)
    body = " ".join(_WORDS[int(i)] for i in idx)
    return f"seed{nonce:06d} " + body


# =========================================================================== generation
@register
class GenerationCapabilityProbe(Probe):
    """Can it generate a multi-token continuation? (encoder vs generative + head/objective)."""

    id = "P19.arch.generation"
    title = "Generation capability, output-length latency & streaming granularity"
    category = "architecture"
    kinds = ("generative",)
    informs = {"arch.family": 0.7, "output.head": 0.7, "train.objective": 0.6,
               "serving.streaming_granularity": 0.5}
    cost = 5
    baseline = True

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        prompt = ("Continue this story with several sentences: Once upon a time in a distant "
                  "kingdom there lived a curious inventor who")
        try:
            r_gen = await ctx.gen(GenRequest.chat(prompt, max_tokens=24, temperature=0.0))
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("generation: budget exhausted")
            return res
        res.n_requests = ctx.n_requests
        if not r_gen.ok:
            # An output-cap rejection is itself the encoder contract: the endpoint
            # administratively refuses multi-token generation.  Retry at the cap.
            err = r_gen.error or ""
            if "max_tokens" in err or "output token" in err.lower():
                try:
                    r_cap = await ctx.gen(GenRequest.chat(prompt, max_tokens=1,
                                                          temperature=0.0))
                except BudgetExhausted:
                    r_cap = None
                n_cap = _n_generated(r_cap) if (r_cap and r_cap.ok) else 0
                res.add_evidence(Evidence(
                    property="arch.family",
                    likelihoods={"encoder_only": 1.0, "encoder_decoder": 0.4,
                                 "non_neural": 0.6, "decoder_only": _FLOOR,
                                 "ssm_or_linear": _FLOOR, "hybrid_attn_ssm": _FLOOR,
                                 "diffusion": 0.2},
                    summary=f"endpoint refuses multi-token generation ({err[:60]}); "
                            f"at the cap it produced {n_cap} token(s) -> a classifier / "
                            f"encoder-style contract, not a generative LM",
                    level="observed", strength=0.9,
                    stats={"error": err[:120], "tokens_at_cap": n_cap},
                    alternatives=["a generative model administratively capped at 1 token"],
                ))
                res.add_evidence(Evidence(
                    property="output.head",
                    likelihoods={"classification_heads": 1.0, "generative_lm_head": _FLOOR},
                    summary="output capped at a single token -> classification-style head",
                    level="observed", strength=0.9,
                    stats={"error": err[:120]},
                    alternatives=["max output tokens administratively set to 1"],
                ))
                res.add_evidence(Evidence(
                    property="train.objective",
                    likelihoods={"masked_lm": 1.0, "contrastive_or_classifier": 0.8,
                                 "causal_lm": 0.15, "prefix_lm_or_span": 0.4,
                                 "diffusion": 0.3},
                    summary="no left-to-right generation permitted -> masked/contrastive "
                            "objective, not causal LM",
                    level="inferred", strength=0.7,
                    stats={"error": err[:120]},
                    alternatives=["a causal LM behind a 1-token output cap"],
                ))
                res.shared["can_generate"] = False
                res.notes.append("generation refused by the output cap -> encoder-family "
                                 "evidence (cap could also administratively hide a decoder)")
                return res
            res.status = "partial"
            res.notes.append(f"generation: request failed: {r_gen.error}")
            return res

        n_gen = _n_generated(r_gen)
        can_generate = n_gen > 1

        # --- output-length vs latency (AR decode vs flat/diffusion) --------------
        ar_ratio = None
        latency_scales = None
        if can_generate:
            try:
                a = await ctx.gen(GenRequest.chat(_rand_prompt(ctx, 20, 1), max_tokens=2, temperature=0.0))
                b = await ctx.gen(GenRequest.chat(_rand_prompt(ctx, 20, 2), max_tokens=16, temperature=0.0))
            except BudgetExhausted:
                a = b = None
            if a is not None and b is not None and a.ok and b.ok:
                sa, sb = _server_s(a), _server_s(b)
                na, nb = _n_generated(a), _n_generated(b)
                if sa and sb and nb > na:
                    ar_ratio = sb / max(sa, 1e-9)
                    latency_scales = sb > sa * 1.15
                    res.tables["output_vs_latency"] = [
                        {"output_tokens": na, "server_s": round(sa, 4)},
                        {"output_tokens": nb, "server_s": round(sb, 4)},
                    ]

        # --- streaming granularity ----------------------------------------------
        tokens_per_chunk = None
        if ctx.caps.streaming:
            try:
                s = await ctx.gen(GenRequest.chat(_rand_prompt(ctx, 12, 3), max_tokens=12,
                                                  temperature=0.0, stream=True))
            except BudgetExhausted:
                s = None
            if s is not None and s.ok and s.timing.chunks:
                nt = [c.n_tokens for c in s.timing.chunks if c.n_tokens]
                if nt:
                    tokens_per_chunk = float(np.median(nt))
                else:
                    tokens_per_chunk = 1.0  # one chunk per emitted piece
        res.n_requests = ctx.n_requests

        res.tables["generation"] = [{"max_tokens_requested": 24, "tokens_generated": n_gen,
                                     "can_generate": can_generate}]

        if can_generate:
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"decoder_only": 1.0, "ssm_or_linear": 1.0, "hybrid_attn_ssm": 1.0,
                             "encoder_decoder": 0.9, "diffusion": 0.9, "encoder_only": _FLOOR,
                             "non_neural": 0.3},
                summary=f"endpoint generated {n_gen} tokens of free continuation → a generative "
                        f"family, not a pure encoder classifier",
                level="observed", strength=0.85,
                stats={"tokens_generated": n_gen},
                alternatives=["an encoder-decoder whose decoder does the generating",
                              "a retrieval system that emits stored text"],
            ))
            res.add_evidence(Evidence(
                property="output.head",
                likelihoods={"generative_lm_head": 1.0, "classification_heads": 0.1},
                summary="produces a token stream → a generative LM head",
                level="observed", strength=0.85, stats={"tokens_generated": n_gen},
                alternatives=["classification head plus a templated verbaliser"],
            ))
            res.add_evidence(Evidence(
                property="train.objective",
                likelihoods={"causal_lm": 1.0, "prefix_lm_or_span": 0.7, "diffusion": 0.5,
                             "masked_lm": 0.1, "contrastive_or_classifier": 0.1},
                summary="free-form left-to-right continuation → a language-modelling objective",
                level="inferred", strength=0.6, stats={"tokens_generated": n_gen},
                alternatives=["prefix-LM / span-corruption pretraining also continues text"],
            ))
        else:
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"encoder_only": 1.0, "encoder_decoder": 0.4, "non_neural": 0.6,
                             "decoder_only": _FLOOR, "ssm_or_linear": _FLOOR,
                             "hybrid_attn_ssm": _FLOOR, "diffusion": 0.2},
                summary=f"endpoint emitted only {n_gen} token(s) and did not continue the prompt → "
                        f"a classifier / encoder head, not a generative LM",
                level="observed", strength=0.9, stats={"tokens_generated": n_gen},
                alternatives=["a generative model artificially capped to 1 output token"],
            ))
            res.add_evidence(Evidence(
                property="output.head",
                likelihoods={"classification_heads": 1.0, "generative_lm_head": _FLOOR},
                summary="single-token / fixed output → classification (non-generative) head",
                level="observed", strength=0.9, stats={"tokens_generated": n_gen},
                alternatives=["max output tokens administratively set to 1"],
            ))
            res.add_evidence(Evidence(
                property="train.objective",
                likelihoods={"masked_lm": 1.0, "contrastive_or_classifier": 1.0, "causal_lm": _FLOOR,
                             "prefix_lm_or_span": 0.3, "diffusion": 0.3},
                summary="no left-to-right generation → a masked / contrastive (encoder) objective",
                level="inferred", strength=0.7, stats={"tokens_generated": n_gen},
                alternatives=["a causal LM whose generation endpoint is disabled"],
            ))

        # AR vs diffusion refinement (only meaningful when it does generate).
        if latency_scales is True:
            res.notes.append(
                f"Latency scales with output length (ratio≈{ar_ratio:.2f}) → autoregressive decode.")
        elif latency_scales is False:
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"diffusion": 1.0, "decoder_only": 0.4, "ssm_or_linear": 0.4,
                             "encoder_decoder": 0.4, "hybrid_attn_ssm": 0.4, "encoder_only": 0.2,
                             "non_neural": 0.3},
                summary=f"generates many tokens but latency is flat in output length "
                        f"(ratio≈{ar_ratio:.2f}) → parallel/iterative-refinement decode (diffusion-like)",
                level="inferred", strength=0.5,
                stats={"ar_ratio": ar_ratio},
                alternatives=["speculative decoding accepting whole drafts",
                              "output-length measurement dominated by fixed overhead"],
            ))
            res.notes.append("DOCUMENTED: flat output-length latency is the diffusion code path; "
                             "no diffusion preset exists in the simulator to exercise it end-to-end.")

        if tokens_per_chunk is not None:
            gran = "token" if tokens_per_chunk <= 1.5 else "multi_token_chunks"
            res.add_evidence(Evidence(
                property="serving.streaming_granularity",
                likelihoods=({"token": 1.0, "multi_token_chunks": 0.3} if gran == "token"
                             else {"token": 0.3, "multi_token_chunks": 1.0}),
                summary=f"streamed ~{tokens_per_chunk:.1f} token(s) per chunk → {gran}",
                level="observed", strength=0.6, stats={"tokens_per_chunk": tokens_per_chunk},
                alternatives=["chunk coalescing by the gateway/proxy"],
            ))

        res.shared["can_generate"] = can_generate
        res.notes.append(f"Generation: {n_gen} tokens for a 24-token request (can_generate={can_generate}).")
        return res


# =========================================================================== prefill
@register
class PrefillScalingProbe(Probe):
    """Prefill server-time vs prompt length: linear (SSM) vs quadratic (full attention)."""

    id = "P19.arch.prefill_scaling"
    title = "Prefill curvature (linear SSM vs quadratic attention)"
    category = "architecture"
    kinds = ("generative",)
    informs = {"arch.family": 0.6}
    cost = 8
    #: minimum max prompt length (tokens) at which an O(len^2) term would be visible;
    #: below this we cannot separate linear from quadratic and emit no family evidence.
    _MIN_QUAD_TOKENS = 1000

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        # Curvature only shows once attention is a sizeable FLOP share (~ctx/(6·d)), so the
        # full sweep reaches ~32k tokens, bounded by the known context window.
        # Fast mode stays short and therefore emits only weak evidence.
        if ctx.config.get("p19_word_counts"):
            word_counts = list(ctx.config["p19_word_counts"])
        elif ctx.fast():
            word_counts = [80, 200, 380, 620, 900]
        else:
            max_ctx = (ctx.shared.get("context") or {}).get("max_input_tokens") or 32_768
            cap_words = int(min(24_000, 0.6 * max_ctx))      # ~1.35 tokens per word
            word_counts = [w for w in (200, 1_000, 3_000, 6_000, 12_000, 24_000) if w <= cap_words]
        ps: list[float] = []
        ts: list[float] = []
        over_length = False
        try:
            for k, nw in enumerate(word_counts):
                r = await ctx.gen(GenRequest(prompt=_rand_prompt(ctx, nw, k), max_tokens=1,
                                             temperature=0.0))
                if not r.ok:
                    over_length = True
                    break  # context limit reached; stop lengthening
                s = _server_s(r)
                pt = r.usage.prompt_tokens if r.usage else None
                if s is None or pt is None:
                    continue
                ps.append(float(pt))
                ts.append(float(s))
        except BudgetExhausted:
            res.status = "partial"
        res.n_requests = ctx.n_requests

        res.tables["prefill_sweep"] = [
            {"prompt_tokens": int(p), "server_s": round(t, 5)} for p, t in zip(ps, ts)
        ]
        if ps:
            res.series["prefill"] = {"x": [int(p) for p in ps], "y": [round(t, 5) for t in ts],
                                     "xlabel": "prompt tokens", "ylabel": "server_s", "kind": "scatter"}

        if len(ps) < 4:
            res.status = res.status if res.status != "ok" else "partial"
            res.notes.append(
                f"prefill: only {len(ps)} usable points"
                + (" (hit context limit early → short context)" if over_length else "")
                + " — cannot separate linear from quadratic.")
            return res

        max_tok = max(ps)
        x = np.asarray(ps, float) / 1000.0        # scale for numerical conditioning
        y = np.asarray(ts, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # bootstrap resamples can be ill-conditioned
            sel = poly_model_select(x, y, degrees=(1, 2), n_boot=400,
                                    rng=int(ctx.rng.integers(1 << 30)))
        curv = sel.get("curvature")
        sig = bool(sel.get("quadratic_significant"))
        degree = sel.get("degree")

        if max_tok < self._MIN_QUAD_TOKENS:
            res.notes.append(
                f"prefill: sweep reached only {max_tok:.0f} tokens (< {self._MIN_QUAD_TOKENS}); a "
                f"short context cannot show O(len^2) — no family evidence emitted "
                f"(short-context / encoder-like signal).")
            return res

        # BIC degree selection is the robust primary signal (deterministic); the bootstrap
        # significance is a secondary confirmation reported in stats.  A numerical-noise
        # curvature (~1e-20) must NOT count: the quadratic term has to contribute a
        # material share of the predicted time at the longest prompt (>=15%).
        quad_share = 0.0
        if curv is not None and curv > 0 and len(ps) >= 4:
            try:
                c2, c1, c0 = np.polyfit(x, y, 2)
                y_hat = np.polyval((c2, c1, c0), x)
                quad_share = float(c2 * x.max() ** 2 / max(float(y_hat.max()), 1e-12))
            except Exception:
                quad_share = 0.0
        quadratic = (curv is not None and curv > 0 and quad_share >= 0.15
                     and (degree == 2 or sig))
        if quadratic:
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"ssm_or_linear": _FLOOR, "hybrid_attn_ssm": 0.4, "decoder_only": 1.0,
                             "encoder_only": 1.0, "encoder_decoder": 1.0, "diffusion": 0.6,
                             "non_neural": 0.5},
                summary=f"prefill server-time is quadratic in prompt length (curvature {curv:.3e}, "
                        f"bootstrap CI excludes 0) → full self-attention, not a constant-state model",
                level="inferred", strength=0.8,
                stats={"curvature": curv, "curvature_ci": sel.get("curvature_ci"),
                       "degree": degree, "max_prompt_tokens": max_tok,
                       "quad_share_at_max": quad_share,
                       "bic": sel.get("bic")},
                alternatives=["queueing/batching noise faking curvature",
                              "chunked prefill that is piecewise-linear"],
            ))
            res.notes.append(f"Prefill quadratic (curv={curv:.3e}, deg={degree}) → full attention.")
        else:
            # Absence of curvature is weak evidence: chunked prefill, sliding windows, and the
            # small attention FLOP share at moderate lengths all linearise a full-attention
            # model. Never floor the attention families; scale strength with the length reached.
            # P(linear timing | full attention) shrinks with length: at >=32k tokens attention
            # FLOPs rival or exceed the MLP's, so only sliding-window decoders stay linear.
            if max_tok >= 32_000:
                lin_strength, p_attn = 0.8, 0.12
            elif max_tok >= 16_000:
                lin_strength, p_attn = 0.4, 0.3
            else:
                lin_strength, p_attn = 0.15, 0.6
            res.add_evidence(Evidence(
                property="arch.family",
                likelihoods={"ssm_or_linear": 1.0, "hybrid_attn_ssm": 0.7, "decoder_only": p_attn,
                             "encoder_only": p_attn, "encoder_decoder": p_attn, "diffusion": 0.4,
                             "non_neural": 0.4},
                summary=f"prefill server-time shows no significant curvature up to {max_tok:.0f} "
                        f"tokens (degree {degree}) — mildly favours constant-state SSM / linear "
                        f"attention; full attention often looks linear at these lengths",
                level="inferred", strength=lin_strength,
                stats={"curvature": curv, "curvature_ci": sel.get("curvature_ci"),
                       "degree": degree, "max_prompt_tokens": max_tok, "bic": sel.get("bic"),
                       "quad_share_at_max": quad_share},
                alternatives=["a sliding-window / chunked-prefill full-attention model",
                              "the quadratic term is below the timing noise floor at these lengths"],
            ))
            res.notes.append(f"Prefill linear (curv={curv}, deg={degree}) up to {max_tok:.0f} tokens "
                             f"(weak evidence, strength {lin_strength}).")
        return res
