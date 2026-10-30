"""Hypothesis space: discrete properties with values and priors, plus the canonical
quantity names estimates use. Probes MUST use these names (the engine rejects unknown
properties; unknown quantities are allowed but flagged in the report).

Priors are deliberately mild and reflect the 2024-2026 model landscape (most served
LLM APIs are decoder-only transformers). The report always shows prior → posterior so the
reader can see how much the evidence, not the prior, did.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Property:
    key: str
    description: str
    values: dict[str, float]            # value -> prior (normalised on load)
    kinds: tuple[str, ...] = ("generative", "decision", "embedding")
    value_docs: dict[str, str] = field(default_factory=dict)


def _p(key, desc, values, kinds=("generative", "decision", "embedding"), docs=None):
    return Property(key, desc, values, kinds, docs or {})


PROPERTIES: dict[str, Property] = {p.key: p for p in [
    # ---------------------------------------------------------------- architecture
    _p("arch.family", "Core network family",
       {"decoder_only": 0.62, "encoder_only": 0.07, "encoder_decoder": 0.07,
        "ssm_or_linear": 0.05, "hybrid_attn_ssm": 0.08, "diffusion": 0.06, "non_neural": 0.05},
       docs={"decoder_only": "causal transformer (GPT/Llama-style)",
             "encoder_only": "bidirectional encoder + heads (BERT-style)",
             "encoder_decoder": "encoder + cross-attending decoder (T5-style)",
             "ssm_or_linear": "state-space / linear-attention / RNN (Mamba, RWKV)",
             "hybrid_attn_ssm": "interleaved attention + SSM/linear layers (Jamba, MiniMax, Qwen3-Next)",
             "diffusion": "masked/discrete diffusion LM (LLaDA, Mercury)",
             "non_neural": "rule system / retrieval / lookup, or not an ML model"}),
    _p("arch.sparsity", "Dense vs sparse mixture-of-experts FFN",
       {"dense": 0.5, "moe": 0.5}),
    _p("attn.long_range", "How cost/recall scales with context",
       {"full_quadratic": 0.6, "sliding_or_local": 0.12, "linear_recurrent": 0.1, "hybrid": 0.18},
       docs={"full_quadratic": "every token attends to all previous tokens",
             "sliding_or_local": "attention limited to a window (possibly with a few global layers)",
             "linear_recurrent": "constant-size state (SSM/linear attention)",
             "hybrid": "mix of full attention and local/linear layers"}),
    _p("attn.kv_sharing", "KV-head sharing",
       {"mha": 0.2, "gqa_or_mqa": 0.6, "mla": 0.2}),
    _p("pos.encoding", "Positional encoding family",
       {"rope": 0.7, "alibi": 0.06, "learned_absolute": 0.1, "relative_bias": 0.06, "nope": 0.08}),
    _p("pos.long_context_ext", "Context-extension method",
       {"none": 0.4, "interpolation_or_yarn": 0.6}),
    # ---------------------------------------------------------------- output / numerics
    _p("out.logit_precision", "Numeric precision of the logits the API reports",
       {"fp32": 0.35, "bf16": 0.4, "fp16": 0.1, "fp8_or_lower": 0.15}),
    _p("output.head", "Output head type",
       {"generative_lm_head": 0.7, "classification_heads": 0.3}),
    # ---------------------------------------------------------------- multimodality
    _p("modal.input", "Accepted input modalities",
       {"text_only": 0.6, "text_image": 0.25, "text_image_audio": 0.1, "text_audio": 0.05},
       docs={"text_only": "only text prompts accepted",
             "text_image": "vision-language (image inputs accepted)",
             "text_image_audio": "omni (image + audio inputs)",
             "text_audio": "speech/audio inputs (no image)"}),
    _p("modal.output", "Produced output modalities",
       {"text_only": 0.7, "text_image": 0.15, "text_audio": 0.1, "text_image_audio": 0.05}),
    _p("modal.vision_encoder", "Vision pathway type when image input is accepted",
       {"none": 0.6, "vit_patch_tokens": 0.25, "native_resolution_vit": 0.1, "cross_attn_adapter": 0.05},
       docs={"vit_patch_tokens": "image → ViT → patch tokens fused into the sequence (LLaVA-style)",
             "native_resolution_vit": "dynamic/native-resolution tiling (Qwen-VL/InternVL-style)",
             "cross_attn_adapter": "image features injected via cross-attention (Flamingo/Llama-4-style)"}),
    # ---------------------------------------------------------------- serving stack
    _p("serving.speculative_decoding", "Speculative / multi-token decoding in serving",
       {"yes": 0.4, "no": 0.6}, kinds=("generative",)),
    _p("serving.prompt_caching", "Prefix / prompt caching",
       {"yes": 0.5, "no": 0.5}),
    _p("serving.batch_invariant", "Outputs invariant to batching/load at temperature 0",
       {"yes": 0.3, "no": 0.7}),
    _p("serving.streaming_granularity", "Streaming unit",
       {"token": 0.6, "multi_token_chunks": 0.4}, kinds=("generative",)),
    # ---------------------------------------------------------------- system topology
    _p("system.topology", "What sits between the user and the weights",
       {"single_model": 0.6, "router_or_cascade": 0.3, "ensemble": 0.1}),
    _p("system.safety_middleware", "External safety classifier(s)",
       {"none": 0.4, "pre_filter": 0.2, "post_filter": 0.15, "both": 0.25}),
    _p("system.retrieval", "Retrieval augmentation",
       {"none": 0.85, "retrieval_augmented": 0.15}),
    _p("system.hidden_reasoning", "Hidden (billed or unbilled) reasoning before the visible answer",
       {"yes": 0.4, "no": 0.6}, kinds=("generative",)),
    _p("system.tool_orchestration", "Server-side tool use / agent loop",
       {"yes": 0.1, "no": 0.9}, kinds=("generative",)),
    # ---------------------------------------------------------------- training
    _p("train.objective", "Pre-training objective",
       {"causal_lm": 0.7, "masked_lm": 0.08, "prefix_lm_or_span": 0.08, "diffusion": 0.06,
        "contrastive_or_classifier": 0.08}),
    _p("post.instruction_tuning", "Base vs instruction-tuned",
       {"base": 0.15, "instruct": 0.85}, kinds=("generative",)),
    _p("post.preference_optimization", "RLHF/DPO-style preference optimisation",
       {"none": 0.25, "rlhf_or_dpo": 0.75}, kinds=("generative",)),
    _p("post.reasoning_rl", "Reasoning-focused RL (o1/R1-style)",
       {"yes": 0.4, "no": 0.6}, kinds=("generative",)),
    _p("post.distillation_signal", "Behavioural signs of distillation from another model family",
       {"none": 0.75, "distilled_from_other": 0.25}, kinds=("generative",)),
    # ---------------------------------------------------------------- decision models
    _p("decision.question_coupling", "Multiple questions processed independently vs jointly",
       {"independent": 0.5, "coupled": 0.5}, kinds=("decision",)),
    _p("decision.option_processing", "Options scored by meaning vs position/label",
       {"semantic": 0.6, "positional": 0.4}, kinds=("decision",)),
    _p("decision.scoring_mechanism", "How option probabilities are produced",
       {"single_pass_heads": 0.4, "per_option_scoring": 0.3, "generative_then_parse": 0.3},
       kinds=("decision",),
       docs={"single_pass_heads": "one forward pass over state+questions, heads read out all options",
             "per_option_scoring": "each option scored by (state, question, option) passes",
             "generative_then_parse": "an LM generates an answer that is parsed/normalised"}),
]}

# Dynamic properties (values added at runtime, e.g. tokenizer families found in registry).
DYNAMIC_PROPERTIES = {"tokenizer.family"}


# Canonical quantity names. `unit` and a short description are used in reports.
QUANTITIES: dict[str, tuple[str, str]] = {
    "dims.hidden_size": ("", "residual stream / final hidden dimension (d_model)"),
    "dims.vocab_size": ("tokens", "vocabulary size"),
    "dims.n_layers": ("", "number of transformer blocks"),
    "dims.n_heads": ("", "attention query heads"),
    "dims.n_kv_heads": ("", "key/value heads"),
    "dims.head_dim": ("", "per-head dimension"),
    "dims.ffn_dim": ("", "FFN intermediate size (dense) or per-expert size (MoE)"),
    "dims.total_params": ("B", "total parameters (billions)"),
    "dims.active_params": ("B", "active parameters per token (billions)"),
    "dims.n_experts": ("", "routed experts per MoE layer"),
    "dims.active_experts": ("", "experts active per token"),
    "dims.embedding_tied": ("", "input/output embedding tying"),
    "context.max_input_tokens": ("tokens", "maximum prompt length accepted"),
    "context.max_output_tokens": ("tokens", "maximum generation length"),
    "context.effective_recall_tokens": ("tokens", "length up to which needle recall stays high"),
    "context.sliding_window": ("tokens", "attention window if local attention detected"),
    "tokenizer.family": ("", "best-matching tokenizer"),
    "tokenizer.type": ("", "BPE / unigram / wordpiece / byte-level"),
    "tokenizer.chat_template_overhead": ("tokens", "tokens added per request/message by the template"),
    "timing.rtt_ms": ("ms", "network round trip baseline"),
    "timing.prefill_tokens_per_s": ("tok/s", "prefill throughput"),
    "timing.decode_tokens_per_s": ("tok/s", "single-stream decode throughput"),
    "timing.itl_ms": ("ms", "median inter-token latency"),
    "timing.kv_growth_us_per_token": ("us/token", "increase of per-token decode time per context token"),
    "serving.cache_block_tokens": ("tokens", "prompt-cache granularity"),
    "serving.spec_decoding_speedup": ("x", "predictable vs high-entropy decode speed ratio"),
    "out.logit_dtype": ("", "inferred numeric dtype of reported logits"),
    "knowledge.cutoff": ("", "estimated training-data cutoff (YYYY-MM)"),
    "reasoning.max_hop_depth": ("hops", "compositional depth at which accuracy falls below 50%"),
    "reasoning.hidden_tokens_ratio": ("x", "hidden reasoning tokens / visible tokens"),
    "decision.probability_resolution": ("", "granularity of reported probabilities"),
}


def normalized_priors(prop: Property) -> dict[str, float]:
    s = sum(prop.values.values())
    return {k: v / s for k, v in prop.values.items()}
