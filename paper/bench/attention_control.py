"""Same-hardware attention/sparse/SSM control for the timing channel.

Three served subjects on identical hardware behind the same wrapper:

* ``full``    — gpt2 (trained, full causal attention);
* ``window``  — tiny random-init Mistral with a 256-token sliding window
                (mechanics subject: timing code paths are weight-independent);
* ``ssm``     — tiny random-init Mamba (constant-size recurrent state).

Protocol per subject: lengths L in a fixed grid; at each L, fresh-nonce prompt pairs
decoded for n_a / n_b tokens (prefill-cancelling marginal decode cost, the P23
design) plus a cold/warm prefix pair under a full-prefix cache (repeated-prefix
curve, separating cache hits from length effects — the confound that contaminated
the production fresh-nonce comparison).

Analysis, identical for every subject and both curves: Theil-Sen decode-growth slope
with CI; BIC degree-1-vs-2 model selection and a log-log exponent with CI for the
prefill curves; and an explicit power analysis (n needed to separate full from
window decode growth at 80% power given the measured residual variance).

Frozen thresholds: decode growth > 6 us/token/1k ctx = growing KV state; < 2 = flat.

Usage: PYTHONIOENCODING=utf-8 python paper/bench/attention_control.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "bench"))

from wrapper import ArchitectureWrapper, ServedModel  # noqa: E402
from loader import load_subject  # noqa: E402
from forensics.stats.regression import theilsen, poly_model_select  # noqa: E402

CACHE = ROOT / ".cache" / "openweight-pilot" / "models"
OUT = ROOT / "paper" / "bench" / "out" / "attention_control.json"

LENGTHS = [96, 192, 384, 768, 960]          # token targets (all inside gpt2's 1024 cap)
REPS = 6
N_A, N_B = 2, 12
_TOKEN_POOL: list[str] = []


def _pool(tok) -> list[str]:
    """Words that tokenize to exactly one token (so word count == token count)."""
    global _TOKEN_POOL
    if not _TOKEN_POOL:
        base = ("alpha beta gamma delta omega sigma theta kappa lambda vector matrix "
                "tensor engine river stone cloud paper code memory city tree ocean "
                "market copper silver gold iron steel glass plastic winter summer "
                "spring autumn monday friday sunday jacket window garden bridge "
                "castle tunnel rocket planet galaxy comet orbit atom proton photon").split()
        pool = []
        for w in base:
            for form in (w, w.capitalize(), w + "s"):
                if len(tok.encode(" " + form, add_special_tokens=False)) == 1:
                    pool.append(" " + form)
        _TOKEN_POOL = pool
    return _TOKEN_POOL


def _mistral_tiny(tok):
    from transformers import MistralConfig, MistralForCausalLM
    cfg = MistralConfig(vocab_size=50257, hidden_size=256, intermediate_size=512,
                        num_hidden_layers=4, num_attention_heads=8,
                        num_key_value_heads=2, sliding_window=256,
                        max_position_embeddings=4096)
    return MistralForCausalLM(cfg).eval()


def _mamba_tiny(tok):
    from transformers import MambaConfig, MambaForCausalLM
    cfg = MambaConfig(vocab_size=50257, hidden_size=256, num_hidden_layers=8,
                      state_size=16, conv_kernel=4, expand=2)
    return MambaForCausalLM(cfg).eval()


def _fresh(rng, L, nonce):
    pool = _pool_global[0]
    idx = rng.integers(0, len(pool), size=L)
    return f"n{nonce:05d}" + "".join(pool[int(i)] for i in idx)


_pool_global = [None]


async def _post(client, prompt, n):
    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
        "temperature": 0.0})
    if r.status_code != 200 or "x-processing-time" not in r.headers:
        return None
    d = r.json()
    u = d.get("usage") or {}
    return (float(r.headers["x-processing-time"]),
            int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))


async def _post_stream_itl(client, prompt, n=16):
    """Stream n tokens; return (median server-side per-token ms, prompt_tokens)."""
    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
        "temperature": 0.0, "stream": True})
    if r.status_code != 200:
        return None
    ts, ptoks = [], None
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        import json as _json
        try:
            obj = _json.loads(line[6:])
        except Exception:
            continue
        if obj.get("t_server_ms") is not None:
            ts.append(float(obj["t_server_ms"]))
        u = obj.get("usage")
        if isinstance(u, dict):
            ptoks = u.get("prompt_tokens")
    if len(ts) < 4 or ptoks is None:
        return None
    itl = np.diff(np.array(ts))
    itl = itl[itl > 0]
    return (float(np.median(itl)), int(ptoks))


def _analyse(rows_decode, rows_ttft):
    out = {}
    xs = np.array([r["L"] for r in rows_decode], float)
    ys = np.array([r["ms_per_token"] for r in rows_decode], float)
    slope, _, (lo, hi) = theilsen(xs, ys, level=0.9)
    out["decode_growth_us_per_1k"] = {"slope": slope * 1000, "ci": [lo * 1000, hi * 1000],
                                      "n": len(rows_decode)}
    # cold-prefill curve: exponent + BIC degree selection
    cx = np.array([r["L"] for r in rows_ttft], float)
    cy = np.array([r["cold_ms"] for r in rows_ttft], float) / 1000.0
    sel = poly_model_select(cx / 1000.0, cy, degrees=(1, 2), criterion="bic",
                            n_boot=200, rng=0)
    A = np.vstack([np.log(cx), np.ones_like(cx)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(cy), rcond=None)
    resid = np.log(cy) - A @ coef
    out["prefill_fresh"] = {"exponent": float(coef[0]),
                            "exponent_se": float(resid.std() / max(np.ptp(np.log(cx)), 1e-9)),
                            "quadratic_significant": bool(sel["quadratic_significant"]),
                            "curvature": sel.get("curvature")}
    warm = np.array([r["warm_ms"] for r in rows_ttft], float)
    out["prefix_cache"] = {"warm_minus_cold_ms": float(np.median(warm - cy * 1000.0)),
                           "ratio": float(np.median(warm / np.maximum(cy * 1000.0, 1e-9)))}
    # residual variance of decode cells (per length) for the power analysis
    by_L: dict[float, list[float]] = {}
    for r in rows_decode:
        by_L.setdefault(r["L"], []).append(r["ms_per_token"])
    resids = []
    for L, v in by_L.items():
        v = np.array(v)
        if len(v) > 1:
            resids.extend((v - v.mean()).tolist())
    out["decode_resid_sd_ms"] = float(np.std(resids)) if resids else None
    return out


async def main():
    torch.set_num_threads(4)
    subj = {"key": "gpt2", "repo": "openai-community/gpt2", "family": "decoder_only",
            "arch": "dense", "opts": {}}
    a, _, gt = load_subject(subj, cache_dir=str(CACHE))
    tok = a.tok
    subjects = {
        "full_attention": ServedModel(a.model, tok, "causal", {}),
        "sliding_window_256": ServedModel(_mistral_tiny(tok), tok, "causal", {}),
        "ssm_mamba": ServedModel(_mamba_tiny(tok), tok, "causal", {}),
    }
    rng = np.random.default_rng(3)
    _pool_global[0] = _pool(tok)
    results = {}
    for name, served in subjects.items():
        wrapper = ArchitectureWrapper(served, opts={"prefix_cache": True})
        t0 = time.time()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapper.app),
                                     base_url="http://c") as client:
            global _client
            import builtins
            builtins._client = client
            rows_decode, rows_ttft = [], []
            trials = [(L, rep) for rep in range(REPS) for L in LENGTHS]
            order = rng.permutation(len(trials))
            for oi in order:
                L, rep = trials[int(oi)]
                ra = await _post_stream_itl(client, _fresh(rng, L, 1000 + rep))
                rb = await _post_stream_itl(client, _fresh(rng, L, 2000 + rep))
                pc = _fresh(rng, L, 3000 + rep)
                rc0 = await _post(client, pc, 2)   # cold
                rc1 = await _post(client, pc, 2)   # warm (prefix-cache hit)
                for rr in (ra, rb):
                    if rr:
                        rows_decode.append({"L": rr[1], "ms_per_token": rr[0]})
                if rc0 and rc1:
                    rows_ttft.append({"L": rc0[1], "cold_ms": rc0[0],
                                      "warm_ms": rc1[0]})
        results[name] = {"decode_rows": rows_decode, "ttft_rows": rows_ttft,
                         "analysis": _analyse(rows_decode, rows_ttft),
                         "wall_s": round(time.time() - t0, 1)}
        print(name, json.dumps(results[name]["analysis"], indent=1), flush=True)

    # power analysis: n per length needed to separate full vs window decode growth
    fa = results["full_attention"]["analysis"]
    sw = results["sliding_window_256"]["analysis"]
    if fa.get("decode_resid_sd_ms") and sw.get("decode_resid_sd_ms"):
        from scipy import stats as sps
        sd = max(fa["decode_resid_sd_ms"], sw["decode_resid_sd_ms"])
        delta = abs(fa["decode_growth_us_per_1k"]["slope"]
                    - sw["decode_growth_us_per_1k"]["slope"]) / 1000.0  # ms/token per ctx tok
        span = float(max(LENGTHS) - min(LENGTHS))
        delta_range = delta * span                    # contrast in ms/token across the grid
        effect = delta_range / (np.sqrt(2) * sd) if sd > 0 else 0.0
        n_needed = (2 * (sps.norm.ppf(0.975) + sps.norm.ppf(0.8)) ** 2
                    / max(effect ** 2, 1e-12))
        results["power"] = {"resid_sd_ms": sd, "contrast_us_per_1k": delta * 1000,
                            "n_per_arm_for_80pct": float(np.ceil(n_needed)),
                            "achieved_n_per_arm": REPS * len(LENGTHS)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    asyncio.run(main())
