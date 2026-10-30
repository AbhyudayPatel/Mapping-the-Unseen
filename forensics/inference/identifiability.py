"""Identifiability analysis -- the honesty layer.

Every other component tries to *pin down* the architecture; this one states, for each
hypothesis property and each measurable quantity, **how much the interface actually lets
us know** and, where the answer is "not much", *why* -- the concrete way in which two
different architectures could have produced the exact same observations.

The scale (see :data:`forensics.probes.base` and ``docs/DEVGUIDE.md``):

======  ================  ============================================================
level   label             meaning
======  ================  ============================================================
1       DIRECT            read straight off the interface (error text, usage counts,
                          a reported field) -- no modelling.
2       STATISTICAL       recoverable from the available side channels with enough
                          samples and a testable statistical argument.
3       WEAK              only a noisy / confounded signal exists; a weak indication.
4       ASSUMPTION        obtainable only by importing outside modelling assumptions
                          (DB scaling laws, hardware priors, extrapolation shape).
5       NON_IDENTIFIABLE  not identifiable from this interface at all -- distinct
                          architectures can produce identical observations.
======  ================  ============================================================

The level is **not** fixed per key: it is a function of what the interface exposes
(:class:`~forensics.core.types.Capabilities`) and which probes actually ran.  ``d_model``
is level 2 when ``caps.logprobs`` is true and P03 recovered a logit-rank cliff, and level
5 when ``caps.logprobs`` is false.  For every level >= 4 entry we attach an ``equivalence``
string: the core non-identifiability argument naming architectures that would look the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from forensics.core.types import Capabilities
from forensics.inference.hypotheses import PROPERTIES, QUANTITIES
from forensics.inference.run_types import IdentifiabilityEntry
from forensics.probes.base import (
    ASSUMPTION, DIRECT, NON_IDENTIFIABLE, STATISTICAL, WEAK, Estimate,
)

if TYPE_CHECKING:  # belief.py is owned by another module; import only for type checking
    from forensics.inference.belief import Belief

__all__ = [
    "analyze", "summarize", "format_by_level", "IdentifiabilitySummary",
    "LEVEL_LABELS", "LEVEL_NOTES",
]

# --------------------------------------------------------------------------- labels
LEVEL_LABELS: dict[int, str] = {
    DIRECT: "directly observable",
    STATISTICAL: "statistically identifiable",
    WEAK: "weakly identifiable (confounded)",
    ASSUMPTION: "identifiable only under assumptions",
    NON_IDENTIFIABLE: "non-identifiable from this interface",
}

LEVEL_NOTES: dict[int, str] = {
    DIRECT: "read straight off the API surface (error text, usage counts, reported fields).",
    STATISTICAL: "recoverable from available side channels with enough samples and a testable argument.",
    WEAK: "only a noisy/confounded signal exists -- a weak indication, not a measurement.",
    ASSUMPTION: "obtainable only by importing outside assumptions (DB scaling laws, HW priors, extrapolation shape).",
    NON_IDENTIFIABLE: "not identifiable from this interface: distinct architectures can produce identical observations.",
}

# --------------------------------------------------------------------------- equivalence arguments
_EQ_ROTATION = (
    "The stacked logit matrix identifies the unembedding only up to a d x d invertible map, "
    "so it pins the residual-stream *rank*, not the internal width or layer structure; a "
    "width-d model and any invertibly reparameterised twin are I/O-identical."
)
_EQ_NO_LOGPROBS = (
    "With no logprobs the API exposes only sampled text; infinitely many (width, unembedding, "
    "temperature) combinations reproduce the same token distribution, so d_model has no observable trace."
)
_EQ_DEPTH = (
    "A 32-layer and a 48-layer model distilled to match each other's outputs are behaviorally "
    "indistinguishable from I/O alone; depth trades against width/precision at fixed loss, and only "
    "the compute-per-token *product* (~layers x width) leaks through timing."
)
_EQ_HEADS = (
    "GQA/MQA make many (n_heads, n_kv_heads, head_dim) factorizations give identical attention "
    "behavior and KV growth at fixed d_model = n_heads x head_dim; the query-head count is not separable."
)
_EQ_KV = (
    "Only the product n_kv_heads x head_dim x n_layers x dtype_bytes is observable as per-token KV "
    "growth; infinitely many factorizations match one measured growth rate."
)
_EQ_FFN = (
    "FFN width is invisible to I/O: SwiGLU/GeGLU variants with different intermediate sizes and gating "
    "produce the same next-token distribution."
)
_EQ_MOE = (
    "Dense-N-active and MoE-with-N-active-params are throughput- and behavior-equivalent; an 8x7B and a "
    "16x3.5B MoE with equal active params emit identical per-token side channels, so total/expert count "
    "is non-identifiable -- only the active-compute footprint leaks."
)
_EQ_TIED = (
    "Tied and untied input/output embeddings can be fit to the same output distribution; tying is an "
    "internal storage choice with no I/O signature."
)
_EQ_POS = (
    "Different positional schemes (RoPE-theta, YaRN, ALiBi, learned, NoPE) can be tuned to identical "
    "length-generalization/recall curves; the family is inferred only from extrapolation *shape*, under assumption."
)
_EQ_DISTILL = (
    "A distilled model and one independently trained to the same behavior are indistinguishable; identity "
    "banners and self-names are trainable and system-promptable, so distillation can be suggested, never proven."
)
_EQ_RETRIEVAL = (
    "Knowing a post-cutoff fact is equally explained by retrieval augmentation or by later continued "
    "pretraining; the two are non-identifiable without a citation/latency side channel."
)
_EQ_SCORING = (
    "Single-pass readout heads, per-option scoring passes, and generate-then-parse can all emit the "
    "identical reported probability vector."
)
_EQ_VISION = (
    "Different vision pathways (patch-token ViT, native-resolution tiling, cross-attention adapter) can map "
    "an image to the same downstream effect and token accounting."
)
_EQ_ARCH = (
    "YES -- two fundamentally different architectures can produce identical observable behavior: any model "
    "distilled to match another's output distribution, or a dense model tuned to a MoE's active-compute "
    "profile, is indistinguishable from black-box I/O. Black-box forensics identifies a *behavioral "
    "equivalence class*, never the weights or the exact internal computation graph."
)

_LEVEL_ORDER = {"observed": 0, "inferred": 1, "estimated": 2, "speculative": 3}


# --------------------------------------------------------------------------- analysis context
@dataclass
class _Ctx:
    """Bundles everything a per-key rule inspects, with terse predicates."""
    caps: Capabilities
    ran: set[str]
    est_by_q: dict[str, Estimate]

    def cap(self, name: str) -> bool:
        """True only if the capability is *known* truthy (None/False -> False)."""
        return getattr(self.caps, name, None) is True

    def capval(self, name: str):
        return getattr(self.caps, name, None)

    def ranp(self, *prefixes: str) -> bool:
        """Did any probe whose id starts with one of ``prefixes`` run?"""
        return any(pid.startswith(pref) for pid in self.ran for pref in prefixes)

    def est(self, quantity: str) -> Estimate | None:
        return self.est_by_q.get(quantity)


Rule = Callable[[_Ctx], "tuple[int, str, str]"]


def _pick_estimate(a: Estimate, b: Estimate) -> Estimate:
    """Prefer the more identifiable / more directly observed estimate for a quantity."""
    def key(e: Estimate):
        return (int(getattr(e, "identifiability", STATISTICAL)),
                _LEVEL_ORDER.get(getattr(e, "level", "estimated"), 9),
                0 if e.value is not None else 1)
    return a if key(a) <= key(b) else b


# =========================================================================== QUANTITY rules
def _r_hidden_size(I: _Ctx):
    if not I.cap("logprobs"):
        return (NON_IDENTIFIABLE,
                "caps.logprobs is False: P03's centered-SVD of the logit matrix cannot run, and no other "
                "channel exposes the residual dimension.", _EQ_NO_LOGPROBS)
    if I.ranp("P03"):
        e = I.est("dims.hidden_size")
        if e is not None and e.value is not None and int(getattr(e, "identifiability", 9)) <= STATISTICAL:
            return (STATISTICAL,
                    f"caps.logprobs=True and P03 recovered a logit-rank cliff (d~{e.value}) via centered SVD "
                    "of the stacked logit matrix.", "")
        return (STATISTICAL,
                "caps.logprobs=True and P03 ran: d_model is the rank of the stacked logit matrix and is "
                "statistically identifiable, though this run under-sampled columns (lower bound only; sweep "
                "more token columns/rows via logit-bias to sharpen the cliff).", "")
    return (STATISTICAL,
            "caps.logprobs=True: d_model is recoverable as the rank of the stacked logit matrix, but P03 "
            "has not been run yet.", "")


def _r_vocab_size(I: _Ctx):
    tf = I.est("tokenizer.family")
    if I.ranp("P02") and tf is not None and tf.value not in (None, "unknown") \
            and getattr(tf, "level", "") in ("observed", "inferred"):
        return (DIRECT,
                f"P02 matched a public tokenizer ({tf.value}); its vocabulary size is read directly from "
                "that tokenizer.", "")
    if I.cap("logprobs") and I.cap("logit_bias"):
        return (STATISTICAL,
                "No public-tokenizer match, but caps.logprobs + logit_bias let a full-vocab sweep bound the "
                "vocabulary statistically.", "")
    if I.cap("usage_prompt_tokens") and I.ranp("P02"):
        return (WEAK,
                "P02 has only usage token-count deltas and no tokenizer match; vocabulary is a weak inference "
                "from merge behavior.", "")
    return (ASSUMPTION,
            "No tokenizer match and no logit-bias sweep; vocabulary is only assumable from the family DB.",
            "Two families sharing a tokenizer (e.g. many o200k-based models) have identical vocab yet different weights.")


def _r_n_layers(I: _Ctx):
    basis = []
    if I.cap("logprobs") and I.ranp("P03"):
        basis.append("d_model from P03")
    if I.ranp("P07"):
        basis.append("prefill/decode timing from P07")
    if basis:
        return (ASSUMPTION,
                "Never directly observable from black-box I/O; only estimable via archdb scaling regressions "
                f"({archdb_hint()}) on {', '.join(basis)}, under strong architecture-family priors.", _EQ_DEPTH)
    return (NON_IDENTIFIABLE,
            "No d_model (needs caps.logprobs + P03) and no timing (P07): nothing here constrains the layer count.",
            _EQ_DEPTH)


def archdb_hint() -> str:
    return "archdb.predict_n_layers / regressions"


def _r_n_heads(I: _Ctx):
    if I.ranp("P07") or (I.cap("logprobs") and I.ranp("P03")):
        return (ASSUMPTION,
                "Estimable only via head_dim conventions (64/128) and DB regressions on d_model (P03) or "
                "KV-growth timing (P07); GQA/MQA hide the true query-head count.", _EQ_HEADS)
    return (NON_IDENTIFIABLE,
            "No logprobs+P03 (no d_model) and no timing (P07): the query-head count leaves no black-box trace.",
            _EQ_HEADS)


def _r_n_kv_heads(I: _Ctx):
    if I.ranp("P07"):
        return (ASSUMPTION,
                "P07 measures per-token KV growth, but that is the product n_kv_heads x head_dim x n_layers x "
                "dtype_bytes; factoring out n_kv_heads needs head_dim/n_layers/dtype assumptions.", _EQ_KV)
    return (NON_IDENTIFIABLE,
            "No timing (P07): KV-head count has no observable channel.", _EQ_KV)


def _r_head_dim(I: _Ctx):
    if (I.cap("logprobs") and I.ranp("P03")) or I.ranp("P07"):
        return (ASSUMPTION,
                "Assumed from the near-universal 64/128 head_dim convention and d_model (P03) / KV-growth (P07); "
                "not separately measured.", _EQ_HEADS)
    return (NON_IDENTIFIABLE, "No d_model and no timing: head dimension is unconstrained.", _EQ_HEADS)


def _r_ffn_dim(I: _Ctx):
    if I.cap("logprobs") and I.ranp("P03"):
        return (ASSUMPTION,
                "Only assumable from the DB FFN-ratio regression given d_model (P03); the FFN width has no "
                "direct I/O signature.", _EQ_FFN)
    return (NON_IDENTIFIABLE, "No d_model (needs logprobs+P03): FFN width is unconstrained.", _EQ_FFN)


def _r_total_params(I: _Ctx):
    if I.ranp("P12"):
        return (WEAK,
                "P12 grades factual recall vs rarity as a capacity proxy, but this confounds parameter count "
                "with training-data quality, dedup and distillation.", "")
    if I.ranp("P07"):
        return (ASSUMPTION,
                "Only via active-params timing (P07) plus a DB total/active ratio assumption.", _EQ_MOE)
    return (NON_IDENTIFIABLE, "No capacity probe (P12) and no timing (P07): total parameters are unconstrained.", _EQ_MOE)


def _r_active_params(I: _Ctx):
    if I.ranp("P07"):
        return (WEAK,
                "P07 inverts decode throughput to active parameters, but only under hardware priors "
                "(HBM bandwidth, MFU, tensor-parallelism, weight dtype) -- all confounders.", "")
    return (NON_IDENTIFIABLE, "No timing (P07): active parameters cannot be inverted.", _EQ_MOE)


def _r_n_experts(I: _Ctx):
    reason = ("P11 can weakly *detect* sparsity, but the exact routed-expert count is non-identifiable: "
              "idle experts never touch a given token's forward pass." if I.ranp("P11")
              else "No MoE probe ran, and the exact expert count is non-identifiable from I/O regardless.")
    return (NON_IDENTIFIABLE, reason, _EQ_MOE)


def _r_active_experts(I: _Ctx):
    return (NON_IDENTIFIABLE,
            "Active experts per token change only the active-params footprint, which is itself throughput-"
            "equivalent to a dense model; the count is non-identifiable.", _EQ_MOE)


def _r_embedding_tied(I: _Ctx):
    return (NON_IDENTIFIABLE,
            "Input/output embedding tying is an internal storage choice with no black-box signature.", _EQ_TIED)


def _r_ctx_in(I: _Ctx):
    e = I.est("context.max_input_tokens")
    if e is not None and getattr(e, "level", "") == "observed":
        return (DIRECT, f"P05 read the context limit ({e.value}) directly from the over-length API error text.", "")
    if e is not None:
        if getattr(e, "level", "") == "estimated" and int(getattr(e, "identifiability", 9)) >= WEAK:
            return (WEAK,
                    "P05 found no limit in the error text; the value is a binary-search lower bound on the "
                    "acceptance boundary.", "")
        return (STATISTICAL,
                "P05 bounded the window by probing the acceptance boundary (no explicit limit in the error).", "")
    if I.ranp("P05"):
        return (STATISTICAL, "P05 ran but returned no limit; the window is identifiable by pushing prompt length.", "")
    return (STATISTICAL,
            "Directly probeable: an over-length prompt usually returns the limit in its error (P05 not yet run).", "")


def _r_ctx_out(I: _Ctx):
    e = I.est("context.max_output_tokens")
    if e is not None and getattr(e, "level", "") == "observed":
        return (DIRECT, f"P05 read the max-output limit ({e.value}) directly from the max_tokens API error.", "")
    if e is not None:
        if getattr(e, "level", "") == "estimated":
            return (WEAK, "P05 saw no cap error; the value is the largest satisfied request (a lower bound).", "")
        return (STATISTICAL, "P05 inferred the cap from completion length saturating (finish_reason=length).", "")
    if I.ranp("P05"):
        return (STATISTICAL, "P05 ran; output cap is identifiable from generation-length saturation.", "")
    return (STATISTICAL, "Identifiable from a max_tokens error or length saturation (P05 not yet run).", "")


def _r_recall_tokens(I: _Ctx):
    if I.ranp("P06"):
        return (STATISTICAL,
                "P06 needle-in-haystack recall vs length gives the length at which recall collapses.", "")
    return (WEAK, "No needle-recall probe (P06) ran; effective recall is only a weak guess.", "")


def _r_sliding_window(I: _Ctx):
    if I.ranp("P06", "P07"):
        return (STATISTICAL,
                "A recall cliff (P06) or a prefill-cost saturation (P07) at a fixed length reveals a sliding "
                "attention window.", "")
    return (ASSUMPTION,
            "No recall (P06) or timing (P07) probe ran; a window is only assumable from the family DB.",
            "Full attention and a large sliding window are indistinguishable until the prompt exceeds the window.")


def _r_tok_family(I: _Ctx):
    if I.cap("usage_prompt_tokens") and I.ranp("P02"):
        e = I.est("tokenizer.family")
        if e is not None and e.value not in (None, "unknown"):
            return (DIRECT,
                    f"usage.prompt_tokens deltas across crafted strings fingerprint the tokenizer exactly "
                    f"(matched: {e.value}).", "")
        return (STATISTICAL,
                "usage.prompt_tokens deltas (P02) constrain the tokenizer, but no public encoder matched cleanly.", "")
    if I.cap("usage_prompt_tokens"):
        return (STATISTICAL, "usage.prompt_tokens is exposed; P02 can fingerprint the tokenizer (not yet run).", "")
    return (WEAK,
            "caps.usage_prompt_tokens is absent; the tokenizer can only be guessed from surface token strings.", "")


def _r_tok_type(I: _Ctx):
    if I.ranp("P02") and I.cap("usage_prompt_tokens"):
        e = I.est("tokenizer.family")
        if e is not None and e.value not in (None, "unknown"):
            return (DIRECT, "Read from the matched public tokenizer's algorithm (BPE/unigram/wordpiece/byte-level).", "")
        return (STATISTICAL, "Merge/split behavior in usage deltas (P02) distinguishes BPE vs unigram vs byte-level.", "")
    return (WEAK, "No tokenizer probe with usage counts; the tokenizer type is a weak guess.", "")


def _r_template_overhead(I: _Ctx):
    if I.cap("usage_prompt_tokens") and I.ranp("P02", "P18"):
        return (DIRECT,
                "usage.prompt_tokens for a bare vs wrapped message reads the fixed chat-template + per-message "
                "overhead directly.", "")
    if I.cap("usage_prompt_tokens"):
        return (STATISTICAL, "usage.prompt_tokens is exposed; template overhead is a direct subtraction (probe not run).", "")
    return (NON_IDENTIFIABLE,
            "caps.usage_prompt_tokens is absent, so the template's hidden token overhead is unobservable.",
            "Any per-request scaffold size is consistent with the observed text when token counts are hidden.")


def _r_timing_direct(I: _Ctx):
    if I.ranp("P07"):
        return (DIRECT, "Measured directly from response timing (server_timing header / wall clock) by P07.", "")
    return (STATISTICAL, "Directly measurable by timing requests; P07 has not run.", "")


def _r_timing_stat(I: _Ctx):
    if I.ranp("P07"):
        return (STATISTICAL,
                "Measured by P07, but throughput/ITL are confounded by batching, load and network jitter, so "
                "the estimate is statistical.", "")
    return (WEAK, "No timing probe (P07) ran; throughput is only a weak guess without measurement.", "")


def _r_kv_growth(I: _Ctx):
    if I.ranp("P07"):
        return (STATISTICAL,
                "P07 measures the rise in per-token decode latency with context -- a genuine KV-cache-read "
                "signal, but load-confounded.", "")
    return (WEAK, "No timing probe (P07): KV growth is unmeasured.", "")


def _r_cache_block(I: _Ctx):
    if I.cap("usage_cached_tokens") and I.ranp("P09"):
        return (STATISTICAL, "P09 reads usage.cached_tokens jumps to locate the prompt-cache block boundary.", "")
    if I.ranp("P09"):
        return (STATISTICAL, "P09 detects cache-hit latency steps to size the cache block.", "")
    return (WEAK, "No prompt-cache probe (P09) ran; block granularity is unknown.", "")


def _r_spec_speedup(I: _Ctx):
    if I.cap("streaming") and I.ranp("P08"):
        return (STATISTICAL,
                "P08 compares inter-token latency on predictable vs high-entropy text over the stream to size "
                "the speculative-decoding speedup.", "")
    if not I.cap("streaming"):
        return (NON_IDENTIFIABLE,
                "caps.streaming is False: without per-token arrival times the acceptance-rate speedup is invisible.",
                "A model with speculative decoding and a plain model that is simply faster are indistinguishable "
                "without inter-token latency structure.")
    return (STATISTICAL, "caps.streaming=True; P08 can measure the speedup (not yet run).", "")


def _r_logit_dtype(I: _Ctx):
    if I.cap("logprobs") and I.ranp("P04"):
        return (STATISTICAL,
                "P04 reads the ULP staircase of the reported logprobs to infer the numeric dtype (fp32/bf16/fp16/fp8).",
                "")
    if not I.cap("logprobs"):
        return (NON_IDENTIFIABLE,
                "caps.logprobs is False: with no numeric logit values the quantization lattice is invisible.",
                "The internal compute dtype is unobservable when only sampled text is returned.")
    return (STATISTICAL, "caps.logprobs=True; P04 can read the logprob quantization lattice (not yet run).", "")


def _r_cutoff(I: _Ctx):
    if I.ranp("P12", "P16.topology.retrieval"):
        return (WEAK,
                "Dateable-fact recall (P12) suggests a cutoff, but continued pretraining and retrieval (P16) "
                "confound the boundary.", "")
    return (ASSUMPTION, "No knowledge/retrieval probe ran; the cutoff is only assumable.", _EQ_RETRIEVAL)


def _r_hop_depth(I: _Ctx):
    if I.ranp("P13"):
        return (WEAK,
                "P13 finds the compositional depth at which accuracy falls below chance, but this is heavily "
                "task- and prompt-dependent.", "")
    return (ASSUMPTION, "No reasoning-depth probe (P13) ran; hop depth is only assumable.",
            "Reasoning depth reflects post-training and prompt scaffolding, not a fixed architectural quantity.")


def _r_hidden_ratio(I: _Ctx):
    if I.cap("usage_reasoning_tokens"):
        return (DIRECT, "usage.reasoning_tokens is reported, so the hidden/visible token ratio is read directly.", "")
    if I.ranp("P13"):
        return (WEAK,
                "No billed reasoning-token field; P13 infers hidden tokens from the prefill/first-token timing "
                "gap, which is load-confounded.", "")
    return (NON_IDENTIFIABLE,
            "No usage.reasoning_tokens and no reasoning probe (P13): silent hidden reasoning leaves no channel.",
            "Unbilled hidden reasoning is indistinguishable from a slower first token or a longer prompt.")


def _r_prob_resolution(I: _Ctx):
    if I.caps.kind != "decision":
        return (NON_IDENTIFIABLE, "Not a decision endpoint: no reported probability grid exists.",
                "A generative endpoint exposes no calibrated option probabilities to quantize.")
    if I.ranp("P18.decision.probability_grid"):
        return (DIRECT, "P18 reads the granularity straight off the reported (rounded) option probabilities.", "")
    return (STATISTICAL, "Decision endpoint reports probabilities; their quantization grid is directly readable (probe not run).", "")


# =========================================================================== PROPERTY rules
def _r_arch_family(I: _Ctx):
    if I.ranp("P19", "P07"):
        return (STATISTICAL,
                "Generation behavior and prefill curvature (P19/P07) separate the broad class "
                "(decoder vs encoder vs SSM/linear vs diffusion); exact internal structure remains hidden.", "")
    if I.ranp("P02", "P17"):
        return (WEAK, "Only tokenizer/metamorphic hints (P02/P17) available; the family is a weak inference.", "")
    return (ASSUMPTION, "No generation or timing probe ran; family is only assumable from the tokenizer/DB.", _EQ_ARCH)


def _r_sparsity(I: _Ctx):
    if I.ranp("P11", "P10"):
        return (WEAK,
                "P11 (domain-conditioned compute) and P10 (nondeterminism) *hint* at MoE, but batch "
                "non-invariance has many causes and dense/MoE are throughput-equivalent.", "")
    return (ASSUMPTION, "No MoE/determinism probe ran; sparsity is only assumable.", _EQ_MOE)


def _r_long_range(I: _Ctx):
    if I.ranp("P07", "P19.arch.prefill_scaling", "P06"):
        note = "" if I.capval("server_timing_header") else " (client-side timing only, noisier)"
        return (STATISTICAL,
                f"Prefill-cost curvature (P07/P19) and recall vs length (P06) classify the attention regime "
                f"(quadratic vs local/window vs linear-recurrent){note}.", "")
    return (ASSUMPTION, "No timing or recall probe ran; the long-range regime is only assumable.",
            "Full attention and a large sliding window look identical below the window length.")


def _r_kv_sharing(I: _Ctx):
    if I.ranp("P07"):
        return (WEAK,
                "P07's KV-growth slope separates MHA from GQA/MQA coarsely, but MLA and specific head counts "
                "are confounded by dtype/layers.", "")
    return (ASSUMPTION, "No timing probe (P07) ran; KV-sharing scheme is only assumable.", _EQ_KV)


def _r_pos_encoding(I: _Ctx):
    if I.ranp("P06"):
        return (ASSUMPTION,
                "P06's length-extrapolation and recall *shape* suggest a positional family, but only under "
                "modelling assumptions about how each scheme degrades.", _EQ_POS)
    return (NON_IDENTIFIABLE, "No positional/recall probe (P06) ran; the encoding family is unconstrained.", _EQ_POS)


def _r_pos_ext(I: _Ctx):
    if I.ranp("P06", "P05"):
        return (ASSUMPTION,
                "A long advertised window with recall degrading past a base length (P05/P06) suggests "
                "interpolation/YaRN, under extrapolation-shape assumptions.", _EQ_POS)
    return (NON_IDENTIFIABLE, "No context-extension evidence (P05/P06); the method is unconstrained.", _EQ_POS)


def _r_logit_precision(I: _Ctx):
    if I.cap("logprobs") and I.ranp("P04"):
        return (STATISTICAL, "P04 infers logit precision from the ULP lattice of reported logprobs.", "")
    if not I.cap("logprobs"):
        return (NON_IDENTIFIABLE, "caps.logprobs is False: no numeric logits, so precision is invisible.",
                "The serving precision is unobservable without logit values.")
    return (STATISTICAL, "caps.logprobs=True; P04 can read the precision lattice (not yet run).", "")


def _r_output_head(I: _Ctx):
    if I.ranp("P19", "P03", "P18.decision.output_readout"):
        return (STATISTICAL,
                "Presence of open-ended generation vs fixed heads (P19/P18) and logit geometry (P03) "
                "distinguish an LM head from classification heads.", "")
    return (WEAK, "No generation/logit probe ran; head type is a weak inference.", "")


def _r_modal_input(I: _Ctx):
    if I.ranp("P20.modal.input"):
        return (DIRECT, "P20 sends image/audio inputs and reads acceptance/rejection straight off the API.", "")
    return (STATISTICAL, "Accepted modalities are directly testable by submitting each (P20 not yet run).", "")


def _r_modal_output(I: _Ctx):
    if I.ranp("P20.modal.output"):
        return (DIRECT, "P20 observes which output modalities the endpoint actually returns.", "")
    return (STATISTICAL, "Output modalities are directly testable (P20 not yet run).", "")


def _r_vision_encoder(I: _Ctx):
    if I.ranp("P20.modal.input"):
        return (ASSUMPTION,
                "If images are accepted (P20), the fusion style is only assumable from image-token accounting "
                "and timing.", _EQ_VISION)
    return (NON_IDENTIFIABLE, "No multimodal probe (P20) ran; the vision pathway is unobservable.", _EQ_VISION)


def _r_spec_decoding(I: _Ctx):
    if I.cap("streaming") and I.ranp("P08"):
        return (STATISTICAL, "P08 detects speculative decoding from ITL bimodality on predictable text.", "")
    if not I.cap("streaming"):
        return (NON_IDENTIFIABLE, "caps.streaming is False: no per-token timing to reveal drafting.",
                "Speculative decoding vs a plain faster model are indistinguishable without ITL structure.")
    return (STATISTICAL, "caps.streaming=True; P08 can test for speculative decoding (not yet run).", "")


def _r_prompt_caching(I: _Ctx):
    if I.cap("usage_cached_tokens"):
        return (DIRECT, "usage.cached_tokens is reported, so prompt caching is directly observable.", "")
    if I.ranp("P09"):
        return (STATISTICAL, "P09 detects a latency drop on a repeated prefix (timing side channel).", "")
    return (WEAK, "No cached-token field and no cache probe (P09); caching is a weak inference.", "")


def _r_batch_invariant(I: _Ctx):
    if I.ranp("P10", "P15"):
        return (STATISTICAL, "P10/P15 measure temperature-0 repeat determinism across load -- a direct test.", "")
    return (WEAK, "No determinism probe (P10/P15) ran; batch invariance is a weak inference.", "")


def _r_stream_granularity(I: _Ctx):
    if I.cap("streaming") and I.ranp("P08", "P19"):
        return (DIRECT, "With streaming, P08/P19 read the chunk sizes (single-token vs multi-token) directly.", "")
    if I.cap("streaming"):
        return (STATISTICAL, "caps.streaming=True; chunk granularity is directly observable (probe not run).", "")
    return (NON_IDENTIFIABLE, "caps.streaming is False: no chunk boundaries to observe.",
            "Without streaming, token-level vs chunked emission is indistinguishable.")


def _r_topology(I: _Ctx):
    if I.ranp("P16.topology.latency_mixture"):
        return (WEAK,
                "P16 finds multimodal latency across prompts (router/cascade hint), but a single model with "
                "high variance can mimic it.", "")
    return (ASSUMPTION, "No topology probe (P16) ran; topology is only assumable.",
            "A single high-variance model, a router, and an ensemble can share the same latency/text distribution.")


def _r_safety_mw(I: _Ctx):
    if I.ranp("P16.topology.safety_middleware"):
        return (STATISTICAL,
                "P16 separates pre-filter (instant refusal, no generation timing) from post-filter (generate "
                "then blank) from timing and partial output.", "")
    return (WEAK, "No safety-middleware probe (P16) ran; external filters are a weak inference.", "")


def _r_retrieval(I: _Ctx):
    if I.ranp("P16.topology.retrieval"):
        return (WEAK, "P16 checks post-cutoff fact recall, but training recency confounds retrieval detection.", "")
    return (ASSUMPTION, "No retrieval probe (P16) ran; retrieval augmentation is only assumable.", _EQ_RETRIEVAL)


def _r_hidden_reasoning(I: _Ctx):
    if I.cap("usage_reasoning_tokens"):
        return (DIRECT, "usage.reasoning_tokens is reported, so hidden reasoning is directly observable.", "")
    if I.ranp("P13"):
        return (WEAK, "No billed reasoning field; P13 infers it from the first-token timing gap (confounded).", "")
    return (ASSUMPTION, "No reasoning-token field and no P13; hidden reasoning is only assumable.",
            "Unbilled hidden reasoning is indistinguishable from a slow first token.")


def _r_tools(I: _Ctx):
    if I.ranp("P16.topology.tools"):
        return (STATISTICAL,
                "P16 checks exact answers to computations beyond model capability (a tool signature).", "")
    return (WEAK, "No tool-use probe (P16) ran; server-side tools are a weak inference.", "")


def _r_train_objective(I: _Ctx):
    if I.ranp("P19", "P14.alignment.base_vs_instruct"):
        return (WEAK,
                "Generation/continuation behavior (P19/P14) hints at the pretraining objective, but post-"
                "training masks it heavily.", "")
    return (ASSUMPTION, "No generation/alignment probe ran; the objective is only assumable.",
            "Causal-LM and prefix-LM/span models can be post-trained to identical instruction behavior.")


def _r_instruction_tuning(I: _Ctx):
    if I.ranp("P14.alignment.base_vs_instruct"):
        return (STATISTICAL,
                "P14 measures format/answer compliance vs raw continuation -- a clear base-vs-instruct signal.", "")
    return (WEAK, "No base-vs-instruct probe (P14) ran; tuning status is a weak inference.", "")


def _r_pref_opt(I: _Ctx):
    if I.ranp("P14.alignment.refusal", "P14.alignment.entropy_collapse"):
        return (WEAK,
                "P14 refusal gaps and 'pick a random number' mode collapse suggest preference optimization, "
                "but a safety system prompt can mimic both.", "")
    return (ASSUMPTION, "No refusal/entropy probe (P14) ran; preference optimization is only assumable.",
            "RLHF-driven refusals and system-prompt-driven refusals are indistinguishable from I/O.")


def _r_reasoning_rl(I: _Ctx):
    if I.ranp("P13"):
        return (WEAK, "P13's test-time-compute scaling hints at reasoning RL, but prompting can elicit CoT too.", "")
    return (ASSUMPTION, "No reasoning probe (P13) ran; reasoning-RL is only assumable.",
            "Reasoning-RL and few-shot-elicited chain-of-thought produce similar visible behavior.")


def _r_distillation(I: _Ctx):
    if I.ranp("P14.alignment.identity"):
        return (ASSUMPTION,
                "P14 banner-grabs identity/self-name leaks, but these are trainable and system-promptable, so "
                "distillation can only be assumed, not shown.", _EQ_DISTILL)
    return (NON_IDENTIFIABLE, "No identity probe (P14) ran; distillation origin is unobservable.", _EQ_DISTILL)


def _r_q_coupling(I: _Ctx):
    if I.ranp("P18.decision.question_isolation", "P18.decision.additivity"):
        return (STATISTICAL,
                "P18 tests whether questions share a state prefix / answer independently via token accounting "
                "and answer stability.", "")
    return (WEAK, "No question-isolation probe (P18) ran; coupling is a weak inference.", "")


def _r_option_processing(I: _Ctx):
    if I.ranp("P18.decision.option_position"):
        return (STATISTICAL, "P18 permutes option order to separate positional from semantic scoring.", "")
    return (WEAK, "No option-position probe (P18) ran; processing style is a weak inference.", "")


def _r_scoring(I: _Ctx):
    if I.ranp("P18.decision.choice_set", "P18.decision.output_readout"):
        return (WEAK,
                "P18's IIA/choice-set test detects joint softmax vs independent scoring, but the exact "
                "readout mechanism stays hidden.", _EQ_SCORING)
    return (ASSUMPTION, "No choice-set/readout probe (P18) ran; scoring mechanism is only assumable.", _EQ_SCORING)


# --------------------------------------------------------------------------- rule tables
_QUANTITY_RULES: dict[str, Rule] = {
    "dims.hidden_size": _r_hidden_size,
    "dims.vocab_size": _r_vocab_size,
    "dims.n_layers": _r_n_layers,
    "dims.n_heads": _r_n_heads,
    "dims.n_kv_heads": _r_n_kv_heads,
    "dims.head_dim": _r_head_dim,
    "dims.ffn_dim": _r_ffn_dim,
    "dims.total_params": _r_total_params,
    "dims.active_params": _r_active_params,
    "dims.n_experts": _r_n_experts,
    "dims.active_experts": _r_active_experts,
    "dims.embedding_tied": _r_embedding_tied,
    "context.max_input_tokens": _r_ctx_in,
    "context.max_output_tokens": _r_ctx_out,
    "context.effective_recall_tokens": _r_recall_tokens,
    "context.sliding_window": _r_sliding_window,
    "tokenizer.family": _r_tok_family,
    "tokenizer.type": _r_tok_type,
    "tokenizer.chat_template_overhead": _r_template_overhead,
    "timing.rtt_ms": _r_timing_direct,
    "timing.prefill_tokens_per_s": _r_timing_stat,
    "timing.decode_tokens_per_s": _r_timing_stat,
    "timing.itl_ms": _r_timing_stat,
    "timing.kv_growth_us_per_token": _r_kv_growth,
    "serving.cache_block_tokens": _r_cache_block,
    "serving.spec_decoding_speedup": _r_spec_speedup,
    "out.logit_dtype": _r_logit_dtype,
    "knowledge.cutoff": _r_cutoff,
    "reasoning.max_hop_depth": _r_hop_depth,
    "reasoning.hidden_tokens_ratio": _r_hidden_ratio,
    "decision.probability_resolution": _r_prob_resolution,
}

_PROPERTY_RULES: dict[str, Rule] = {
    "arch.family": _r_arch_family,
    "arch.sparsity": _r_sparsity,
    "attn.long_range": _r_long_range,
    "attn.kv_sharing": _r_kv_sharing,
    "pos.encoding": _r_pos_encoding,
    "pos.long_context_ext": _r_pos_ext,
    "out.logit_precision": _r_logit_precision,
    "output.head": _r_output_head,
    "modal.input": _r_modal_input,
    "modal.output": _r_modal_output,
    "modal.vision_encoder": _r_vision_encoder,
    "serving.speculative_decoding": _r_spec_decoding,
    "serving.prompt_caching": _r_prompt_caching,
    "serving.batch_invariant": _r_batch_invariant,
    "serving.streaming_granularity": _r_stream_granularity,
    "system.topology": _r_topology,
    "system.safety_middleware": _r_safety_mw,
    "system.retrieval": _r_retrieval,
    "system.hidden_reasoning": _r_hidden_reasoning,
    "system.tool_orchestration": _r_tools,
    "train.objective": _r_train_objective,
    "post.instruction_tuning": _r_instruction_tuning,
    "post.preference_optimization": _r_pref_opt,
    "post.reasoning_rl": _r_reasoning_rl,
    "post.distillation_signal": _r_distillation,
    "decision.question_coupling": _r_q_coupling,
    "decision.option_processing": _r_option_processing,
    "decision.scoring_mechanism": _r_scoring,
}


def _default_rule(key: str) -> Rule:
    def rule(I: _Ctx):
        return (ASSUMPTION,
                f"No dedicated observable channel for '{key}' on this interface; assumable only from priors.",
                _EQ_ARCH)
    return rule


# --------------------------------------------------------------------------- public API
def analyze(
    belief: "Belief | None",
    estimates: list[Estimate],
    caps: Capabilities,
    ran_probe_ids: set[str],
) -> list[IdentifiabilityEntry]:
    """Assign an identifiability level (1-5) to every property and quantity the run cares about.

    The level depends on ``caps`` and which probes are in ``ran_probe_ids`` (and, for a few
    quantities, on what the produced ``estimates`` actually observed -- e.g. a context limit
    read off an API error is DIRECT). For every level >= 4 entry we attach the concrete
    non-identifiability argument (``equivalence``). ``belief`` is optional and used only to
    surface dynamically-discovered properties (e.g. tokenizer.family variants).
    """
    ran = set(ran_probe_ids or ())

    # best estimate per quantity (most identifiable / most directly observed wins)
    est_by_q: dict[str, Estimate] = {}
    for e in estimates or ():
        q = getattr(e, "quantity", None)
        if not q:
            continue
        est_by_q[q] = e if q not in est_by_q else _pick_estimate(est_by_q[q], e)

    I = _Ctx(caps=caps, ran=ran, est_by_q=est_by_q)
    kind = caps.kind

    entries: list[IdentifiabilityEntry] = []

    # ---- properties (filtered by endpoint kind) ------------------------------
    for key, prop in PROPERTIES.items():
        if kind not in prop.kinds and kind != "unknown":
            continue
        rule = _PROPERTY_RULES.get(key, _default_rule(key))
        level, reason, equiv = rule(I)
        entries.append(_entry(key, level, reason, equiv))

    # ---- dynamically-discovered properties from the belief (e.g. tokenizer.family variants)
    if belief is not None:
        post = getattr(belief, "posterior", None) or {}
        for key in post:
            if key in PROPERTIES or key in QUANTITIES or key in _QUANTITY_RULES:
                continue
            rule = _PROPERTY_RULES.get(key, _default_rule(key))
            level, reason, equiv = rule(I)
            entries.append(_entry(key, level, reason, equiv))

    # ---- quantities ----------------------------------------------------------
    for key in QUANTITIES:
        if key.startswith("decision.") and kind not in ("decision", "unknown"):
            continue
        rule = _QUANTITY_RULES.get(key, _default_rule(key))
        level, reason, equiv = rule(I)
        entries.append(_entry(key, level, reason, equiv))

    entries.sort(key=lambda e: (e.level, e.key))
    return entries


def _entry(key: str, level: int, reason: str, equiv: str) -> IdentifiabilityEntry:
    if level < ASSUMPTION:
        equiv = ""   # equivalence argument only carried for level >= 4 (per contract)
    return IdentifiabilityEntry(
        key=key, level=level, label=LEVEL_LABELS[level], reason=reason, equivalence=equiv,
    )


# --------------------------------------------------------------------------- summary
@dataclass
class IdentifiabilitySummary:
    """Per-level roll-up plus the top-line non-identifiability verdict."""
    by_level: dict[int, int]
    level_notes: dict[int, str]
    arch_question: str
    arch_verdict: str
    arch_basis: list[str] = field(default_factory=list)
    non_identifiable: list[str] = field(default_factory=list)
    assumption_only: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "by_level": {str(k): v for k, v in self.by_level.items()},
            "level_notes": {str(k): v for k, v in self.level_notes.items()},
            "arch_question": self.arch_question,
            "arch_verdict": self.arch_verdict,
            "arch_basis": self.arch_basis,
            "non_identifiable": self.non_identifiable,
            "assumption_only": self.assumption_only,
        }


# the keys that make up a top-line architecture conclusion
_ARCH_KEYS = (
    "arch.family", "arch.sparsity", "dims.hidden_size", "dims.n_layers",
    "dims.n_heads", "dims.n_experts", "attn.long_range",
)

_ARCH_QUESTION = "Could two fundamentally different architectures produce the exact same observed behavior?"


def summarize(entries: list[IdentifiabilityEntry]) -> IdentifiabilitySummary:
    """Roll up ``entries`` per level and answer the top-line non-identifiability question."""
    by_level = {lvl: 0 for lvl in range(DIRECT, NON_IDENTIFIABLE + 1)}
    for e in entries:
        by_level[e.level] = by_level.get(e.level, 0) + 1

    by_key = {e.key: e for e in entries}
    hard = [e.key for e in entries if e.level >= ASSUMPTION]
    non_id = [e.key for e in entries if e.level == NON_IDENTIFIABLE]

    # which top-line arch conclusions are only equivalence-classes, and which are pinned down
    hard_arch = [k for k in _ARCH_KEYS if k in by_key and by_key[k].level >= ASSUMPTION]
    pinned_arch = [k for k in _ARCH_KEYS if k in by_key and by_key[k].level <= STATISTICAL]

    if hard_arch:
        verdict = (
            "YES. " + _EQ_ARCH.split(" -- ", 1)[1] + " In this run the following top-line arch "
            f"conclusions are not pinned to a single architecture: {', '.join(hard_arch)}. "
        )
        if pinned_arch:
            verdict += (f"The interface does constrain {', '.join(pinned_arch)} to a behavioral "
                        "equivalence class, but never below it.")
        else:
            verdict += "No arch dimension was constrained below its equivalence class by this interface."
    else:
        verdict = (
            "Even where every measurable arch dimension is statistically identifiable "
            f"({', '.join(pinned_arch) or 'none measured'}), the answer remains YES in principle: a "
            "model distilled to match the output distribution, or a dense/MoE pair with equal active "
            "compute, would reproduce identical observations. Black-box forensics fixes a behavioral "
            "equivalence class, not the weights."
        )

    return IdentifiabilitySummary(
        by_level=by_level,
        level_notes=dict(LEVEL_NOTES),
        arch_question=_ARCH_QUESTION,
        arch_verdict=verdict,
        arch_basis=hard_arch,
        non_identifiable=non_id,
        assumption_only=[k for k in hard if k not in non_id],
    )


# --------------------------------------------------------------------------- rendering
def format_by_level(entries: list[IdentifiabilityEntry], width: int = 30) -> str:
    """Human-readable table grouped by identifiability level (1 -> 5)."""
    summ = summarize(entries)
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("IDENTIFIABILITY  (1 direct .. 5 non-identifiable)")
    lines.append("=" * 78)
    by_level: dict[int, list[IdentifiabilityEntry]] = {}
    for e in entries:
        by_level.setdefault(e.level, []).append(e)
    for lvl in range(DIRECT, NON_IDENTIFIABLE + 1):
        group = by_level.get(lvl, [])
        header = f"[{lvl}] {LEVEL_LABELS[lvl].upper()}  ({len(group)})"
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(f"    {LEVEL_NOTES[lvl]}")
        for e in sorted(group, key=lambda x: x.key):
            lines.append(f"    {e.key:<{width}} {e.reason}")
            if e.equivalence:
                lines.append(f"    {'':<{width}} equivalence: {e.equivalence}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("Q: " + summ.arch_question)
    lines.append("A: " + summ.arch_verdict)
    lines.append("=" * 78)
    return "\n".join(lines)
