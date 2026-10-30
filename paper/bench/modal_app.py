"""Modal app: ForensicsBench serving + battery on CPU containers.

Usage (from repo root):
    PYTHONIOENCODING=utf-8 python -m modal run paper/bench/modal_app.py --download-all
    PYTHONIOENCODING=utf-8 python -m modal run paper/bench/modal_app.py --run gpt2
    PYTHONIOENCODING=utf-8 python -m modal run paper/bench/modal_app.py --run-all

Weights are cached in the ``model-forensics-checkpoints`` volume (HF hub layout).
Per-subject results are written back to the volume under ``/results/<key>.json``
and pulled locally into ``runs/forensics-bench/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

try:
    ROOT = Path(__file__).resolve().parents[2]
except IndexError:  # inside the Modal container the module lives at /root
    ROOT = Path("/root")

app = modal.App("model-forensics-bench")
vol = modal.Volume.from_name("model-forensics-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1",
                 extra_options="--index-url https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers==5.17.0", "tokenizers>=0.22", "tiktoken>=0.7",
        "numpy>=1.26", "scipy>=1.11", "fastapi>=0.110", "httpx>=0.27",
        "safetensors>=0.4", "accelerate>=1.0", "sentencepiece>=0.2",
        "protobuf>=5",
    )
    .env({"HF_HUB_CACHE": "/ckpt", "TOKENIZERS_PARALLELISM": "false",
          "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
          "FORENSICS_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS": "1"})
    .add_local_dir(ROOT / "forensics", "/app/forensics")
    .add_local_dir(ROOT / "paper" / "bench", "/app/bench")
)

with image.imports():
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/bench")
    import subjects as bench_subjects  # noqa: E402


# --------------------------------------------------------------------------- downloads
@app.function(image=image, volumes={"/ckpt": vol}, timeout=7200, cpu=2.0, memory=8192)
def download(repo: str) -> str:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo, cache_dir="/ckpt",
                             allow_patterns=["*.json", "*.safetensors", "*.model",
                                             "*.txt", "*.py", "*.tiktoken", "*.jinja"])
    vol.commit()
    return path


# --------------------------------------------------------------------------- battery
def _run_one(key: str) -> dict:
    """Load a subject, serve it, run the frozen battery, persist results."""
    import asyncio
    import os
    import torch
    from loader import load_subject
    from wrapper import ArchitectureWrapper
    from battery import run_battery

    torch.set_num_threads(max(2, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    roster = {s["key"]: s for s in bench_subjects.roster()}
    subj = roster[key]
    a, b, gt = load_subject(subj, cache_dir="/ckpt")
    wrapper = ArchitectureWrapper(a, b, subj.get("opts") or {})
    budget = 2900 if key == "olmoe-1b7b" else 3400
    result = asyncio.run(run_battery(subj, wrapper, "/tmp/journal",
                                     request_budget=budget))
    result["ground_truth"] = gt
    return result


@app.function(image=image, volumes={"/ckpt": vol}, timeout=7200, cpu=4.0, memory=16384)
def run_subject(key: str) -> dict:
    result = _run_one(key)
    Path(f"/ckpt/results/{key}.json").parent.mkdir(parents=True, exist_ok=True)
    Path(f"/ckpt/results/{key}.json").write_text(json.dumps(result))
    vol.commit()
    return result


@app.function(image=image, volumes={"/ckpt": vol}, timeout=10800, cpu=8.0, memory=40960)
def run_subject_big(key: str) -> dict:
    result = _run_one(key)
    Path(f"/ckpt/results/{key}.json").parent.mkdir(parents=True, exist_ok=True)
    Path(f"/ckpt/results/{key}.json").write_text(json.dumps(result))
    vol.commit()
    return result


# --------------------------------------------------------------------------- fingerprints
LLMMAP_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is the capital of Japan?",
    "Write a haiku about autumn leaves.",
    "def fib(n):",
    "Translate 'good morning' into French.",
    "What is 17 times 23?",
    "List three primary colors.",
    "Summarize the plot of Romeo and Juliet in one line.",
    "What causes rain?",
    "Give a one-line Python function to reverse a string.",
    "Who wrote the novel 1984?",
    "What is the boiling point of water at sea level?",
    "Name a mammal that can fly.",
    "What is the derivative of x^2?",
    "How many days are in a leap year?",
    "What is the largest planet in the solar system?",
    "Spell the word 'necessary'.",
    "What does HTTP stand for?",
    "Give one synonym for 'happy'.",
    "What is the speed of light in km/s?",
    "Name the chemical symbol for gold.",
    "What is the square root of 144?",
    "In which continent is Brazil?",
    "Write a SQL query to select all rows from a table users.",
]


@app.function(image=image, volumes={"/ckpt": vol}, timeout=7200, cpu=4.0, memory=16384)
def collect_fingerprint(key: str) -> dict:
    """LLMmap-style fixed-prompt fingerprint: top-5 logprob vectors + latency per
    prompt, collected through the same anonymous wrapper."""
    import asyncio
    import os
    import time
    import numpy as np
    import httpx
    import torch
    from loader import load_subject
    from wrapper import ArchitectureWrapper

    torch.set_num_threads(max(2, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    roster = {s["key"]: s for s in bench_subjects.roster()}
    subj = roster[key]
    a, b, gt = load_subject(subj, cache_dir="/ckpt")
    wrapper = ArchitectureWrapper(a, b, subj.get("opts") or {})

    async def go():
        rows = []
        max_tok = 1 if a.kind == "masked" else 4
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapper.app),
                                     base_url="http://f") as c:
            for p in LLMMAP_PROMPTS:
                r = await c.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": max_tok, "logprobs": True, "top_logprobs": 5,
                    "temperature": 0.0})
                d = r.json()
                if "choices" not in d:
                    rows.append({"prompt": p, "error": str(d)[:120]})
                    continue
                lp = ((d["choices"][0].get("logprobs") or {}).get("content") or [])
                rows.append({
                    "prompt": p,
                    "server_ms": float(r.headers.get("x-processing-time", "nan")),
                    "top": [{"id": t["token_id"], "lp": round(t["logprob"], 6)}
                            for t in (lp[0]["top_logprobs"] if lp else [])],
                    "usage": d.get("usage"),
                })
        return rows

    rows = asyncio.run(go())
    out = {"subject": key, "prompts": LLMMAP_PROMPTS, "rows": rows,
           "ground_truth": gt}
    Path(f"/ckpt/results/fp-{key}.json").parent.mkdir(parents=True, exist_ok=True)
    Path(f"/ckpt/results/fp-{key}.json").write_text(json.dumps(out))
    vol.commit()
    return out


# --------------------------------------------------------------------------- pull
@app.function(image=image, volumes={"/ckpt": vol}, timeout=600)
def pull_results() -> dict:
    """Read back every /results/*.json from the volume (survives local runner death)."""
    import pathlib
    out = {}
    for f in sorted(pathlib.Path("/ckpt/results").glob("*.json")):
        try:
            out[f.name] = f.read_text()
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- entrypoint
@app.local_entrypoint()
def main(download_all: bool = False, run: str = "", run_all: bool = False,
         keys: str = "", fingerprints: bool = False, pull: bool = False):
    out_dir = ROOT / "runs" / "forensics-bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    if pull:
        got = pull_results.remote()
        for name, text in got.items():
            (out_dir / name).write_text(text)
        print(f"pulled {len(got)} result files from volume")
        return
    if fingerprints:
        allow = set(keys.split(",")) if keys else None
        todo = [s["key"] for s in bench_subjects.roster()
                if allow is None or s["key"] in allow]
        big = {"olmoe-1b7b", "tinyllama-1b", "qwen25-05b"}
        handles = [(k, collect_fingerprint.spawn(k)) for k in todo]
        for key, h in handles:
            try:
                out = h.get()
            except Exception as exc:
                print(f"[fp {key}] FAILED: {exc!r}")
                continue
            (out_dir / f"fp-{key}.json").write_text(json.dumps(out, indent=1))
            print(f"[fp {key}] ok")
        return
    if download_all:
        repos = [r for r in bench_subjects.DOWNLOADS]
        print(f"downloading {len(repos)} repos into volume…")
        for path in download.starmap([(r,) for r in repos]):
            print("  ok:", path)
        return
    todo = []
    if run:
        todo = [run]
    elif keys:
        todo = [k for k in keys.split(",") if k]
    elif run_all:
        todo = [s["key"] for s in bench_subjects.roster()]
    if not todo:
        print("nothing to do; use --download-all, --run KEY, --keys a,b,c, --run-all")
        return
    big = {"olmoe-1b7b", "tinyllama-1b", "qwen25-05b"}
    handles = []
    for key in todo:
        fn = run_subject_big if key in big else run_subject
        handles.append((key, fn.spawn(key)))
    for key, h in handles:
        try:
            result = h.get()
        except Exception as exc:
            print(f"[{key}] FAILED: {exc!r}")
            continue
        (out_dir / f"{key}.json").write_text(json.dumps(result, indent=1))
        gt = result.get("ground_truth", {})
        print(f"[{key}] done: width={gt.get('width')} layers={gt.get('layers')} "
              f"requests={result.get('budget', {}).get('requests')} "
              f"wall={result.get('wall_s')}s")
