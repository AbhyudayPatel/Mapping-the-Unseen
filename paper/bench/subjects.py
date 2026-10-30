"""ForensicsBench subject roster — single source of truth for ground truth.

Every subject is served behind the same anonymous OpenAI-compatible wrapper
(:mod:`paper.bench.wrapper`).  Ground-truth fields are recorded from the checkpoint
config at load time (never hard-coded here except the family labels, which come from
the model cards / config classes).

Axes covered
------------
family   : dense decoder, SSM (Mamba), MoE (OLMoE + tiny Mixtral), MLA (tiny
           DeepSeek-V3), encoder (BERT family), encoder-decoder (T5)
width    : 128 … 2048 residual widths
dtype    : fp32 / bf16 / fp16 serving of the same checkpoint, int8 dynamic quant
serving  : single-stream vs batched-8, router swap behind one name

Tiny random-init subjects are *mechanics* subjects: they carry real architecture
code paths (MoE routing, MLA attention) but untrained weights, so behavioural
probes (needle, knowledge) are not meaningful for them and are skipped.
"""
from __future__ import annotations

# (key, repo, family_label, notes).  width/depth/heads are read from the config at
# load time and stored into the ground-truth record by the loader.
DENSE_DECODERS = [
    ("pythia-14m",  "EleutherAI/pythia-14m",  "decoder_only", "dense"),
    ("pythia-31m",  "EleutherAI/pythia-31m",  "decoder_only", "dense"),
    ("pythia-70m",  "EleutherAI/pythia-70m",  "decoder_only", "dense"),
    ("pythia-160m", "EleutherAI/pythia-160m", "decoder_only", "dense"),
    ("opt-125m",    "facebook/opt-125m",      "decoder_only", "dense"),
    ("distilgpt2",  "distilbert/distilgpt2",  "decoder_only", "dense"),
    ("gpt2",        "openai-community/gpt2",  "decoder_only", "dense"),
    ("smollm-135m", "HuggingFaceTB/SmolLM-135M", "decoder_only", "dense"),
    ("qwen25-05b",  "Qwen/Qwen2.5-0.5B",      "decoder_only", "dense"),
    ("tinyllama-1b","TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "decoder_only", "dense"),
]

SSMS = [
    ("mamba-130m",  "state-spaces/mamba-130m-hf", "ssm_or_linear", "mamba-1 SSM"),
    # 370m's pure-PyTorch scan is slow at 24k-token sweeps on CPU; capped at 12k (a
    # serving condition, recorded in ground truth) so the battery completes.
    ("mamba-370m",  "state-spaces/mamba-370m-hf", "ssm_or_linear", "mamba-1 SSM",
     {"cap": 12288}),
]

MOES = [
    # tiny random-init MoE (mechanics subject; real Mixtral code path)
    ("tiny-mixtral-moe", "TINY:mixtral", "decoder_only", "moe"),
    # real trained MoE (large; bf16 on CPU)
    ("olmoe-1b7b", "allenai/OLMoE-1B-7B-0924", "decoder_only", "moe"),
]

MLAS = [
    # tiny random-init MLA (mechanics subject; real DeepSeek-V3 attention code path)
    ("tiny-dsv3-mla", "TINY:deepseek_v3", "decoder_only", "mla"),
]

ENCODERS = [
    ("bert-tiny", "prajjwal1/bert-tiny", "encoder_only", "masked"),
    ("bert-mini", "prajjwal1/bert-mini", "encoder_only", "masked"),
    ("bert-small","prajjwal1/bert-small","encoder_only", "masked"),
    ("distilbert","distilbert/distilbert-base-uncased", "encoder_only", "masked"),
]

ENC_DEC = [
    ("t5-small", "google-t5/t5-small", "encoder_decoder", "seq2seq"),
]

# dtype axis: same checkpoint, different serving precision
DTYPE_AXIS = [
    ("gpt2-bf16",  "openai-community/gpt2", "decoder_only", "dense", {"compute_dtype": "bf16"}),
    ("gpt2-fp16",  "openai-community/gpt2", "decoder_only", "dense", {"compute_dtype": "fp16"}),
    ("gpt2-int8",  "openai-community/gpt2", "decoder_only", "dense", {"quant": "int8_dynamic"}),
    ("gpt2-wirebf16", "openai-community/gpt2", "decoder_only", "dense", {"wire_round": "bf16"}),
]

# serving axis
SERVING_AXIS = [
    ("gpt2-batch8", "openai-community/gpt2", "decoder_only", "dense", {"batch": 8}),
    ("router-gpt2-pythia", "ROUTE:openai-community/gpt2|EleutherAI/pythia-160m",
     "router_or_cascade", "router", {"router": True}),
]

ALL = (DENSE_DECODERS + SSMS + MOES + MLAS + ENCODERS + ENC_DEC + DTYPE_AXIS + SERVING_AXIS)

#: subjects that need big RAM containers
BIG = {"olmoe-1b7b": dict(cpu=8.0, memory=32768, timeout=7200),
       "tinyllama-1b": dict(cpu=4.0, memory=12288, timeout=5400),
       "qwen25-05b": dict(cpu=4.0, memory=8192, timeout=3600)}

DEFAULT_RESOURCES = dict(cpu=2.0, memory=6144, timeout=3600)

#: HF repos to pre-download into the volume (tiny subjects are constructed locally)
DOWNLOADS = sorted({spec[1] for spec in ALL
                    if not spec[1].startswith(("TINY:", "ROUTE:"))})


def roster():
    """Return the full roster as a list of dicts."""
    out = []
    for spec in ALL:
        key, repo, family, arch = spec[0], spec[1], spec[2], spec[3]
        opts = spec[4] if len(spec) > 4 else {}
        out.append({"key": key, "repo": repo, "family": family, "arch": arch,
                    "opts": dict(opts)})
    return out
