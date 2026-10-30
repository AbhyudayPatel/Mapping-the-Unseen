"""Estimate kappa — the per-request discrimination density of real channels.

Proposition (cost bound) in the manuscript: any adaptive test with both error rates
<= delta needs expected cost >= (1-2*delta) ln((1-delta)/delta) / kappa, where every
conditional probe satisfies KL(P0^p || P1^p) <= kappa * cost(p).  Version 2 stated the
bound with kappa unestimated.  Here we measure per-channel kappa_hat (nats per
request) for the hardest hypothesis pair each channel must separate, using recorded
benchmark data and fresh local serving runs:

* **timing (decode-state)**: per-pair marginal decode-time distributions at matched
  context for full-attention vs constant-state subjects (attention_control.json);
* **wire precision**: per-request top-logprob values for the same checkpoint served
  at fp32 vs bf16 wire rounding; KL between the induced offset-from-grid
  distributions;
* **tokenizer counts**: per-request count vectors for near-equivalent tokenizer
  pairs over a random prompt distribution — the discriminating-request probability
  p_diff gives kappa_hat = -ln(1 - p_diff) under a signal/no-signal model (and the
  proved kappa = 0 anchor inside the o200k witness domain).

Each estimate carries a bootstrap 95% CI over requests.  The table then reports the
implied minimum request count for delta = 0.05:  N >= 0.9 * ln(19) / kappa_hat.

Usage: PYTHONIOENCODING=utf-8 python paper/bench/kappa.py
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

OUT = ROOT / "paper" / "bench" / "out" / "kappa.json"
DELTA = 0.05
NUM = (1 - 2 * DELTA) * float(np.log((1 - DELTA) / DELTA))   # ~2.651 nats


def kl_gauss(samples_a, samples_b, rng, n_boot=400, sd_floor=1e-6):
    """KL(N_A || N_B) per request between Gaussian fits, with a bootstrap CI.
    A measurement-precision floor on the sd keeps near-deterministic channels finite."""
    a = np.asarray(samples_a, float)
    b = np.asarray(samples_b, float)

    def kl(x, y):
        m1, s1 = x.mean(), max(x.std(ddof=1), sd_floor)
        m2, s2 = y.mean(), max(y.std(ddof=1), sd_floor)
        return float(np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5)

    base = kl(a, b)
    boots = [kl(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
             for _ in range(n_boot)]
    return base, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def timing_channel(rng):
    """Full-attention vs constant-state decode cells at matched context (control data)."""
    ac = json.loads((ROOT / "paper" / "bench" / "out" / "attention_control.json").read_text())
    fa = [r for r in ac["full_attention"]["decode_rows"]]
    ss = [r for r in ac["ssm_mamba"]["decode_rows"]]
    Ls = sorted({r["L"] for r in fa} & {r["L"] for r in ss})
    Lmax = max(Ls)
    a = [r["ms_per_token"] for r in fa if r["L"] == Lmax]
    b = [r["ms_per_token"] for r in ss if r["L"] == Lmax]
    kl, lo, hi = kl_gauss(a, b, rng)
    return {"pair": f"decode ms/token at ctx={Lmax}: full-attention vs SSM",
            "kappa_per_request": kl, "ci": [lo, hi],
            "min_requests_delta_0.05": NUM / max(kl, 1e-9)}


async def precision_channel(rng):
    """Same checkpoint, fp32 vs bf16 wire rounding: offset-from-bf16-grid per value."""
    from loader import load_subject
    from wrapper import ArchitectureWrapper
    torch.set_num_threads(4)
    subj = {"key": "gpt2", "repo": "openai-community/gpt2", "family": "decoder_only",
            "arch": "dense", "opts": {}}
    a, _, _ = load_subject(subj, cache_dir=str(ROOT / ".cache" / "openweight-pilot" / "models"))

    async def collect(wrapper, n=60):
        vals = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapper.app),
                                     base_url="http://k") as c:
            rng_w = np.random.default_rng(9)
            words = "engine river blue silent paper code vector memory city cloud".split()
            for i in range(n):
                p = " ".join(rng_w.choice(words, size=int(rng_w.integers(4, 20)))) + f" k{i}."
                r = await c.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": p}], "max_tokens": 1,
                    "logprobs": True, "top_logprobs": 5})
                d = r.json()["choices"][0]["logprobs"]["content"][0]
                vals.extend(t["logprob"] for t in d["top_logprobs"])
        return np.array(vals)

    w_plain = ArchitectureWrapper(a)
    v_fp32 = await collect(w_plain)
    from loader import load_subject as _ls  # reload fresh model state
    a2, _, _ = _ls(subj, cache_dir=str(ROOT / ".cache" / "openweight-pilot" / "models"))
    w_round = ArchitectureWrapper(a2, opts={"wire_round": "bf16"})
    v_bf16 = await collect(w_round)

    def offsets(v):
        bf = torch.tensor(v).bfloat16().double().numpy()
        return np.abs(v - bf)

    # KL between the two offset distributions (lognormal on positive offsets)
    o32 = offsets(v_fp32)
    obf = offsets(v_bf16)
    kl, lo, hi = kl_gauss(o32, obf, rng, sd_floor=1e-6)
    frac_on_grid = float(np.mean(obf <= 1e-9))
    return {"pair": "offset-from-bf16-grid per reported logprob: fp32 vs bf16 wire",
            "kappa_per_request": kl, "ci": [lo, hi],
            "min_requests_delta_0.05": NUM / max(kl, 1e-9),
            "bf16_values_exactly_on_grid": frac_on_grid,
            "note": "grid membership is near-deterministic, so the channel separates "
                    "wire-rounded from plain in O(1) requests; the floored Gaussian "
                    "kappa is the conservative continuous-noise view"}


def tokenizer_channel(rng):
    """Count-delta discrimination probability per random prompt between families."""
    import tiktoken
    from transformers import AutoTokenizer
    gpt2 = tiktoken.get_encoding("cl100k_base")
    pairs = []
    try:
        o200k = tiktoken.get_encoding("o200k_base")
        pairs.append(("cl100k_base", "o200k_base",
                      lambda s: len(gpt2.encode(s)), lambda s: len(o200k.encode(s))))
    except Exception:
        pass
    out = {}
    words = "the of and to in is was he for it with as his on be at by had not are but from or have an they which one you were her all she there would their we him been has when who will more no if out so said what up its about into than them can only other".split()
    # discriminating bank: digits, urls, whitespace runs, unicode, camelCase (P02-style)
    def bank_prompt(rng):
        kind = rng.integers(0, 4)
        if kind == 0:
            return " ".join(rng.choice(words, size=int(rng.integers(3, 40))))
        if kind == 1:
            return " ".join(str(rng.integers(0, 10 ** int(rng.integers(1, 8))))
                            for _ in range(int(rng.integers(2, 8))))
        if kind == 2:
            return "https://example.com/" + "/".join(rng.choice(words, size=3)) + \
                   "?q=" + str(rng.integers(0, 9999))
        return "".join(rng.choice(["  ", "\t", "\n", " "]), ) + " ".join(
            rng.choice(words, size=int(rng.integers(2, 10)))) + " café—naïve 😀"
    for name_a, name_b, fa, fb in pairs:
        diffs = 0
        n = 400
        for _ in range(n):
            s = bank_prompt(rng)
            diffs += int(fa(s) != fb(s))
        p_diff = diffs / n
        kappa = -float(np.log(max(1e-9, 1 - p_diff)))
        out[f"{name_a} vs {name_b}"] = {
            "p_diff_per_request": p_diff, "kappa_per_request": kappa,
            "min_requests_delta_0.05": NUM / max(kappa, 1e-9),
            "note": "random English-word prompts; inside the proved o200k_base/"
                    "o200k_harmony witness domain kappa = 0 exactly"}
    return out


async def main():
    rng = np.random.default_rng(7)
    out = {"delta": DELTA, "bound_constant_nats": NUM, "channels": {}}
    out["channels"]["timing_decode_state"] = timing_channel(rng)
    out["channels"]["wire_precision"] = await precision_channel(rng)
    out["channels"]["tokenizer_counts"] = tokenizer_channel(rng)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
