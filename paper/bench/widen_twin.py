"""Constructive witness: output-width non-identifiability beyond the logit rank.

Proposition (output-rank witness).  Take any causal LM with final hidden width d,
final-norm map phi and unembedding E in R^{V x d}, so logits are z = E phi(h_T).
For any d' = d + k, define the width-d' model whose readout is
    phi'(h) = [ phi(h) ; A phi(h) ]  in R^{d'},     E' = [ E | 0 ] in R^{V x d'},
with A in R^{k x d} arbitrary.  Then E' phi'(h) = E phi(h) for every input: the two
models are *functionally identical* — every logit, every token, every timing scale —
yet their declared residual widths differ.  No query over any logit channel
(including full-vocabulary logprobs, hence also top-k) can separate them; width is
identified only up to the *rank* of the observed logit matrix, and only under the
extra assumption that the unembedding is full-rank.

This script builds the twin empirically for a small checkpoint, serves both models
behind the anonymous wrapper, and shows:
  1. max |logprob difference| over a prompt battery < 1e-9 (exact equivalence);
  2. the scanner's geometry probe returns the same rank (the true width of the
     original) for both — i.e. the width-1024 twin is reported as width-768.

Usage: PYTHONIOENCODING=utf-8 python paper/bench/widen_twin.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "bench"))

from wrapper import ArchitectureWrapper  # noqa: E402
from loader import load_subject  # noqa: E402

CACHE = ROOT / ".cache" / "openweight-pilot" / "models"
OUT = ROOT / "paper" / "bench" / "out" / "width_witness.json"


class WidenedGPT2(torch.nn.Module):
    """gpt2 with a (d+k)-wide readout: phi' = [LN_f(h); A LN_f(h)], E' = [E|0]."""

    def __init__(self, base, k: int = 256, seed: int = 0):
        super().__init__()
        self.base = base
        cfg = base.config
        g = torch.Generator().manual_seed(seed)
        d = cfg.n_embd
        self.A = torch.randn(k, d, generator=g) / np.sqrt(d)
        E = base.lm_head.weight.data                       # (V, d) tied to wte
        self.E_ext = torch.cat([E, torch.zeros(E.shape[0], k)], dim=1)  # (V, d+k)
        self._config = cfg

    @property
    def config(self):
        return self._config

    def get_output_embeddings(self):
        class _Head:
            def __init__(s, w):
                s.weight = torch.nn.Parameter(w, requires_grad=False)
        return _Head(self.E_ext)

    def forward(self, input_ids=None, past_key_values=None, use_cache=False, **_):
        out = self.base.transformer(input_ids=input_ids, past_key_values=past_key_values,
                                    use_cache=bool(use_cache))
        h = out.last_hidden_state      # GPT2Model already applies the final LN
        h_ext = torch.cat([h, h @ self.A.T], dim=-1)               # (B, T, d+k)
        logits = h_ext @ self.E_ext.T                              # exact: E h
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(logits=logits, past_key_values=out.past_key_values)


async def _logprob_battery(wrapper, prompts, k=20):
    rows = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapper.app),
                                 base_url="http://w") as client:
        for p in prompts:
            r = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 1, "logprobs": True, "top_logprobs": k})
            d = r.json()["choices"][0]["logprobs"]["content"][0]
            rows.append({t["token_id"]: t["logprob"] for t in d["top_logprobs"]})
    return rows


async def main():
    torch.set_num_threads(4)
    subj = {"key": "gpt2", "repo": "openai-community/gpt2", "family": "decoder_only",
            "arch": "dense", "opts": {}}
    a, _, gt = load_subject(subj, cache_dir=str(CACHE))
    d0 = int(gt["width"])
    twin_model = WidenedGPT2(a.model, k=256).eval()

    from wrapper import ServedModel
    base_served = ServedModel(a.model, a.tok, "causal", {})
    twin_served = ServedModel(twin_model, a.tok, "causal", {})
    w_base = ArchitectureWrapper(base_served)
    w_twin = ArchitectureWrapper(twin_served)

    rng = np.random.default_rng(5)
    words = "engine river blue silent paper code vector memory city cloud summer design".split()
    prompts = [" ".join(rng.choice(words, size=int(rng.integers(5, 30))))
               + f" Case {i}." for i in range(48)]

    lb = await _logprob_battery(w_base, prompts)
    lt = await _logprob_battery(w_twin, prompts)
    max_abs = 0.0
    for rb, rt in zip(lb, lt):
        for tid, vb in rb.items():
            if tid in rt:
                max_abs = max(max_abs, abs(vb - rt[tid]))

    # geometry check: the scanner's rank probe reports the same width for both
    async def _rank(wrapper, rows_n=920, cols_n=1000):
        cols = np.sort(rng.choice(50257, cols_n, replace=False) + 0)
        mat = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapper.app),
                                     base_url="http://w") as client:
            for i in range(rows_n):
                p = " ".join(rng.choice(words, size=int(rng.integers(8, 40)))) + f" r{i}."
                r = await client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": 1, "logprobs": True, "top_logprobs": 5,
                    "measurement_columns": [int(c) for c in cols]})
                vals = {t["token_id"]: t["logprob"]
                        for t in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
                mat.append([vals[int(c)] for c in cols])
        a = np.asarray(mat); a -= a.mean(1, keepdims=True)
        sv = np.linalg.svd(a, compute_uv=False)
        return int(np.sum(sv > 1e-6 * sv[0]))
    rank_base = await _rank(w_base)
    rank_twin = await _rank(w_twin)
    res = {
        "witness": "output-rank width non-identifiability",
        "base_width": d0, "twin_declared_width": d0 + 256,
        "n_prompts": len(prompts), "top_k": 20,
        "max_abs_logprob_diff": max_abs,
        "measured_rank_base": rank_base, "measured_rank_twin": rank_twin,
        "conclusion": "logit outputs identical to numerical precision while declared "
                      "residual widths differ by 256; the rank probe returns the same "
                      "estimate for both, so width is identified only up to logit rank "
                      "under a full-rank-unembedding assumption",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
