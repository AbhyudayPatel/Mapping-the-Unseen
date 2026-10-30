"""Subject loader: real checkpoints, tiny random-init mechanics subjects,
dtype/quantisation variants, and router pairs.  Records ground truth from the
loaded config classes (never asserted by hand).
"""
from __future__ import annotations

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer)

from wrapper import ServedModel


# --------------------------------------------------------------------------- tiny configs
def _tiny_mixtral():
    from transformers import MixtralConfig, MixtralForCausalLM
    cfg = MixtralConfig(vocab_size=50257, hidden_size=256, intermediate_size=512,
                        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                        num_local_experts=8, num_experts_per_tok=2, max_position_embeddings=2048)
    return MixtralForCausalLM(cfg), cfg


def _tiny_dsv3():
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
    cfg = DeepseekV3Config(vocab_size=50257, hidden_size=256, intermediate_size=512,
                           moe_intermediate_size=256, num_hidden_layers=4,
                           num_attention_heads=8,
                           kv_lora_rank=64, q_lora_rank=128, qk_rope_head_dim=16,
                           v_head_dim=32, qk_nope_head_dim=16,
                           n_routed_experts=8, num_experts_per_tok=2, n_group=2,
                           topk_group=2, first_k_dense_replace=1,
                           moe_layer_freq=1, max_position_embeddings=2048)
    return DeepseekV3ForCausalLM(cfg), cfg


TINY_BUILDERS = {"TINY:mixtral": _tiny_mixtral, "TINY:deepseek_v3": _tiny_dsv3}

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _ground_truth(model, cfg, kind: str, family: str, arch: str) -> dict:
    gt = {"model_kind": kind, "family": family, "arch": arch,
          "config_class": type(cfg).__name__,
          "width": int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", -1))),
          "layers": int(getattr(cfg, "num_hidden_layers",
                        getattr(cfg, "n_layer", getattr(cfg, "num_layers", -1)))),
          "heads": int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", -1))),
          "kv_heads": int(getattr(cfg, "num_key_value_heads", -1)),
          "vocab": int(getattr(cfg, "vocab_size", -1)),
          "max_position": int(getattr(cfg, "max_position_embeddings",
                              getattr(cfg, "n_positions", -1)))}
    for extra in ("num_local_experts", "n_routed_experts", "num_experts_per_tok",
                  "kv_lora_rank", "q_lora_rank", "sliding_window",
                  "intermediate_size", "moe_intermediate_size"):
        v = getattr(cfg, extra, None)
        if v is not None:
            gt[extra] = int(v) if isinstance(v, (int, float)) else str(v)
    try:
        gt["output_projection_width"] = int(model.get_output_embeddings().weight.shape[1])
        gt["tied_embeddings"] = bool(model.get_output_embeddings().weight.data_ptr()
                                     == model.get_input_embeddings().weight.data_ptr())
    except Exception:
        pass
    try:
        gt["total_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    except Exception:
        pass
    return gt


def load_subject(subj: dict, cache_dir: str | None = None):
    """Load one roster entry.  Returns (ServedModel a, ServedModel b|None, gt dict)."""
    repo, kind_hint, opts = subj["repo"], subj["arch"], subj.get("opts") or {}
    dt = _DTYPES.get(opts.get("compute_dtype", "fp32"), torch.float32)
    common = {"cache_dir": cache_dir} if cache_dir else {}

    def _one(repo_spec: str):
        if repo_spec in TINY_BUILDERS:
            model, cfg = TINY_BUILDERS[repo_spec]()
            model = model.to(dt).eval()
            tok = AutoTokenizer.from_pretrained("openai-community/gpt2", **common)
            return model, tok, cfg, True, "causal"
        if repo_spec.startswith("prajjwal1/"):
            # these checkpoints lack a model_type key; they are BERT masked-LM models
            from transformers import BertConfig, BertForMaskedLM, BertTokenizer
            cfg = BertConfig.from_pretrained(repo_spec, **common)
            tok = BertTokenizer.from_pretrained(repo_spec, **common)
            model = BertForMaskedLM.from_pretrained(repo_spec, dtype=dt, **common)
            return model.eval(), tok, cfg, False, "masked"
        cfg = AutoConfig.from_pretrained(repo_spec, **common)
        tok = AutoTokenizer.from_pretrained(repo_spec, **common)
        mt = getattr(cfg, "model_type", "").lower()
        if mt in ("bert", "distilbert", "roberta"):
            kind = "masked"
        elif getattr(cfg, "is_encoder_decoder", False):
            kind = "encdec"
        else:
            kind = "causal"
        if kind == "masked":
            model = AutoModelForMaskedLM.from_pretrained(repo_spec, dtype=dt, **common)
        elif kind == "encdec":
            model = AutoModelForSeq2SeqLM.from_pretrained(repo_spec, dtype=dt, **common)
        else:
            model = AutoModelForCausalLM.from_pretrained(repo_spec, dtype=dt, **common)
        return model.eval(), tok, cfg, False, kind

    primary_repo = repo
    if repo.startswith("ROUTE:"):
        primary_repo = repo[len("ROUTE:"):].split("|", 1)[0]
    elif repo.startswith("TINY:") and "|" in repo:
        primary_repo = repo.split("|", 1)[0]
    model, tok, cfg, tiny, kind = _one(primary_repo)
    repo = primary_repo
    if opts.get("quant") == "int8_dynamic":
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear},
                                                       dtype=torch.qint8)
    if repo in TINY_BUILDERS:
        kind = "causal"
    a = ServedModel(model, tok, kind, opts)
    gt = _ground_truth(model, cfg, kind, subj["family"], subj["arch"])
    gt["tiny_random_init"] = bool(tiny)
    gt["opts"] = dict(opts)

    b_model = None
    if opts.get("router") and "|" in subj["repo"]:
        repo_b = subj["repo"].split("|", 1)[1]
        m2, t2, c2, _, _k2 = _one(repo_b)
        b_model = ServedModel(m2, t2, "causal", opts)
        gt["router_twin"] = repo_b
        gt["router_twin_width"] = int(getattr(c2, "hidden_size", -1))
    return a, b_model, gt
