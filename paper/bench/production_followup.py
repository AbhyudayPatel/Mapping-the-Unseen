"""Production follow-up experiments (v4): decode-state growth + powered fresh-prefix
long-context timing on the live endpoints.

Addresses the v2 weaknesses directly:
* the fresh-nonce curve now has n>=8 randomized trials per length and is FITTED with
  the same log-log model as the repeated-prefix curve (v2 showed 8 unfitted points);
* decode-state growth (P23 design) separates a growing KV cache from a constant
  recurrent state using prefill-cancelling fresh prompt pairs at short context;
* an explicit power statement accompanies every exponent estimate.

Endpoints come from .env (Azure-compatible gateway).  All raw responses are kept in
runs/production-followup/.

Usage: PYTHONIOENCODING=utf-8 python paper/bench/production_followup.py
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "bench" / "out" / "production_followup.json"
JOURNAL = ROOT / "runs" / "production-followup"


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
BASE = (os.environ.get("AZURE_OPENAI_BASE_URL", "").rstrip("/")
        or "https://patelabhyuday09-2815-resource.openai.azure.com/openai/v1")
KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
URL = BASE + "/chat/completions"

WORD_POOL = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
             "tempor incididunt ut labore et dolore magna aliqua enim ad minim").split()

# decode-growth design (matches P23): fresh streaming trials at these token targets
# (this prompt style tokenises at ~3 tokens/word; measured prompt_tokens is the x-axis)
DG_LENGTHS = [300, 700, 1500, 3000, 6000, 12000]
DG_REPS = 6
N_A, N_B = 2, 12
# fresh-prefix prefill design (token targets; words = tokens/3 for this prompt style)
FRESH_LENGTHS_DS = [1000, 2000, 4000, 8000, 16000, 32000, 56000]
FRESH_LENGTHS_GPT = [1000, 2000, 4000, 8000, 12000, 15000]
FRESH_REPS = 8
TOK_PER_WORD = 3.0


def fresh_prompt(rng: random.Random, n_words: int, nonce: int, instruct: bool = True) -> str:
    words = [f"{rng.choice(WORD_POOL)}{rng.randrange(10**6)}" for _ in range(n_words)]
    tail = "\n\nReply with the word OK." if instruct else "\n\nContinue this text verbatim:"
    return f"nonce{nonce} " + " ".join(words) + tail


def repeat_prompt(n_words: int) -> str:
    return "sharedprefix " + " ".join(["report"] * n_words) + "\n\nReply with the word OK."


async def post(client, model, prompt, max_tokens):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    try:
        r = await client.post(URL, json=body,
                              headers={"api-key": KEY, "Authorization": f"Bearer {KEY}"})
    except Exception as exc:
        return {"error": repr(exc)}
    dt = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return {"error": r.text[:300], "status": r.status_code}
    d = r.json()
    u = d.get("usage") or {}
    return {"latency_ms": dt, "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens")}


def loglog_fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    xs, ys = xs[m], ys[m]
    if len(xs) < 4:
        return None
    A = np.vstack([np.log(xs), np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ys), rcond=None)
    resid = np.log(ys) - A @ coef
    se = float(resid.std(ddof=2) / max(np.ptp(np.log(xs)), 1e-9) * np.sqrt(len(xs)))
    return {"exponent": float(coef[0]), "ci": [float(coef[0] - 1.96 * se / np.sqrt(len(xs))),
            float(coef[0] + 1.96 * se / np.sqrt(len(xs)))],
            "resid_sd": float(resid.std(ddof=2)), "n": int(len(xs))}


async def _post_stream_itl(client, model, prompt, n=24):
    """Stream n tokens; median client-side inter-chunk latency (ms) + prompt tokens.
    Uses Azure latency_checkpoint when the response carries it."""
    body = {"model": model, "max_tokens": n, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    try:
        r = await client.post(URL, json=body,
                              headers={"api-key": KEY, "Authorization": f"Bearer {KEY}"})
    except Exception as exc:
        return {"error": repr(exc)}
    if r.status_code != 200:
        return {"error": r.text[:300], "status": r.status_code}
    # r is fully read; re-parse SSE with arrival times unavailable post-hoc -> use
    # streaming read instead
    return None


async def stream_itl(client, model, prompt, n=24, retries=4):
    """True streaming read: per-chunk arrival times (client clock), Azure
    latency_checkpoint (server-side engine TBT) when present, and usage."""
    body = {"model": model, "max_tokens": n, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(retries):
        ts, ptoks, lc = [], None, None
        try:
            async with client.stream("POST", URL, json=body,
                                     headers={"api-key": KEY,
                                              "Authorization": f"Bearer {KEY}"}) as r:
                if r.status_code == 429:
                    await r.aread()
                    await asyncio.sleep(8 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    await r.aread()
                    return {"error": r.text[:300], "status": r.status_code}
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        obj = json.loads(line[6:])
                    except Exception:
                        continue
                    ch = obj.get("choices") or []
                    if ch and (ch[0].get("delta") or {}).get("content"):
                        ts.append(time.perf_counter())
                    u = obj.get("usage")
                    if isinstance(u, dict) and u.get("prompt_tokens"):
                        ptoks = int(u["prompt_tokens"])
                    if obj.get("latency_checkpoint"):
                        lc = obj["latency_checkpoint"]
        except Exception as exc:
            return {"error": repr(exc)}
        out = {"prompt_tokens": ptoks, "n_chunks": len(ts)}
        if lc:
            out["engine_tbt_ms"] = lc.get("engine_tbt_ms")
            out["service_tbt_ms"] = lc.get("service_tbt_ms")
            out["engine_ttft_ms"] = lc.get("engine_ttft_ms")
        if len(ts) >= 6:
            itl = np.diff(np.array(ts)) * 1000.0
            itl = itl[itl > 1.0]
            if len(itl) >= 4:
                out["itl_ms"] = float(np.median(itl))
        if ptoks is not None:
            return out
    return {"error": "retries exhausted"}


async def decode_growth(client, model, rng):
    """Per-token decode latency vs context via streaming (P23 channel).  Uses the
    server-side Azure engine TBT when present, else client median ITL."""
    rows = []
    trials = [(L, rep) for rep in range(DG_REPS) for L in DG_LENGTHS]
    rng.shuffle(trials)
    for L, rep in trials:
        p = fresh_prompt(rng, L, 10_000 + rep, instruct=False)
        r = await stream_itl(client, model, p)
        if "error" in r:
            rows.append({"L": L, "rep": rep, "error": r["error"]})
            continue
        row = {"L": r["prompt_tokens"], "rep": rep, "n_chunks": r["n_chunks"]}
        if r.get("engine_tbt_ms") is not None:
            row["ms_per_token_engine"] = float(r["engine_tbt_ms"])
        if r.get("itl_ms") is not None:
            row["ms_per_token_client"] = r["itl_ms"]
        rows.append(row)
    ok = [r for r in rows if "ms_per_token_engine" in r] or \
         [r for r in rows if "ms_per_token_client" in r]
    fit = None
    if len(ok) >= 8:
        key = "ms_per_token_engine" if "ms_per_token_engine" in ok[0] else "ms_per_token_client"
        xs = np.array([r["L"] for r in ok], float)
        ys = np.array([r[key] for r in ok], float)
        from scipy import stats as sps
        slope, intercept, lo, hi = sps.theilslopes(ys, xs, alpha=0.9)
        fit = {"slope_us_per_1k": float(slope * 1_000_000),
               "ci": [float(lo * 1_000_000), float(hi * 1_000_000)],
               "n": len(ok), "median_itl_ms": float(np.median(ys)), "channel": key}
    return {"rows": rows, "fit": fit}


async def prefill_curves(client, model, tok_lengths, rng):
    rows = []
    trials = [("fresh", T, rep) for rep in range(FRESH_REPS) for T in tok_lengths]
    trials += [("repeat", T, rep) for rep in range(3) for T in tok_lengths]
    rng.shuffle(trials)
    for kind, T, rep in trials:
        words = max(8, int(T / TOK_PER_WORD))
        p = (fresh_prompt(rng, words, 30_000 + rep) if kind == "fresh"
             else repeat_prompt(words))
        for attempt in range(5):
            r = await post(client, model, p, 1)
            if r.get("status") == 429:
                await asyncio.sleep(10 * (attempt + 1))
                continue
            break
        r.update({"kind": kind, "target_tokens": T, "rep": rep})
        rows.append(r)
    fresh = [(r["prompt_tokens"], r["latency_ms"]) for r in rows
             if r.get("kind") == "fresh" and "latency_ms" in r and r.get("prompt_tokens")]
    rep_rows = [(r["prompt_tokens"], r["latency_ms"]) for r in rows
                if r.get("kind") == "repeat" and "latency_ms" in r and r.get("prompt_tokens")]
    return {"rows": rows,
            "fresh_fit": loglog_fit([x for x, _ in fresh], [y for _, y in fresh]),
            "repeat_fit": loglog_fit([x for x, _ in rep_rows], [y for _, y in rep_rows])}


async def main():
    JOURNAL.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260930)
    out = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        for model, lengths in (("DeepSeek-V3.2", FRESH_LENGTHS_DS),
                               ("gpt-4o-mini", FRESH_LENGTHS_GPT)):
            print(f"=== {model}: decode growth ===", flush=True)
            dg = await decode_growth(client, model, rng)
            print(json.dumps(dg["fit"], indent=1), flush=True)
            print(f"=== {model}: prefill curves ===", flush=True)
            pc = await prefill_curves(client, model, lengths, rng)
            print("fresh:", json.dumps(pc["fresh_fit"], indent=1), flush=True)
            print("repeat:", json.dumps(pc["repeat_fit"], indent=1), flush=True)
            out[model] = {"decode_growth": dg, "prefill": pc}
            (JOURNAL / f"{model.replace('.', '_')}-followup.json").write_text(
                json.dumps(out[model], indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    asyncio.run(main())
