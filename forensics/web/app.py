"""Local web UI: start scans, watch them live, browse every report and diagram.

    python -m forensics web            # http://127.0.0.1:8765

Binds to localhost only. API keys typed into the form live in this process's memory for
the duration of the scan and are never written to disk (run artefacts store env-var names
only, and the recorder redacts headers).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from forensics.config import ROOT, load_env

RUNS_DIR = ROOT / "runs"
WEB_CACHE = ROOT / ".cache" / "web"
PAGE = Path(__file__).with_name("index.html")


# =============================================================================== jobs
@dataclass
class Job:
    id: str
    label: str
    mode: str
    status: str = "starting"          # starting | running | rendering | done | error
    run_dir: str | None = None
    log: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    budget: Any = None
    max_requests: int | None = None

    def add(self, msg: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del self.log[:-400]

    def view(self) -> dict[str, Any]:
        req = None
        if self.budget is not None:
            try:
                req = self.budget.summary().get("requests")
            except Exception:  # noqa: BLE001
                req = None
        return {"id": self.id, "label": self.label, "mode": self.mode, "status": self.status,
                "run_id": _run_id(Path(self.run_dir)) if self.run_dir else None,
                "log": self.log[-120:], "requests": req, "max_requests": self.max_requests,
                "elapsed_s": round((self.finished or time.time()) - self.started, 1),
                "error": self.error}


JOBS: dict[str, Job] = {}


class ScanRequest(BaseModel):
    preset: str | None = None
    provider: str | None = None           # openai | azure | jev | http
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None            # typed into the form: memory only
    api_key_env: str | None = None        # or the NAME of an env var / .env entry
    auth_style: str | None = None         # api-key | bearer
    template: str | None = None           # provider=http: TemplateConfig JSON text
    mode: str = "quick"                   # quick | thorough
    max_requests: int | None = None
    name: str | None = None


def _run_scan(job: Job, req: ScanRequest) -> None:
    """Thread body: its own event loop, so CPU-heavy steps never block the web server."""
    from forensics import cli
    from forensics.core.recorder import Recorder

    key_env = None
    try:
        load_env()
        api_key_env = req.api_key_env
        if req.api_key:
            key_env = f"FORENSICS_WEB_KEY_{job.id}"
            os.environ[key_env] = req.api_key
            api_key_env = key_env
        model = req.model
        if req.template and not model:
            model = "custom-http"
        target = cli.resolve_target(req.preset or None, None if req.preset else req.provider,
                                    req.base_url or None, model or None, api_key_env,
                                    req.auth_style or None)
        if req.preset and req.api_key:
            target.api_key_env = key_env
        if req.template:
            WEB_CACHE.mkdir(parents=True, exist_ok=True)
            tpath = WEB_CACHE / f"template-{job.id}.json"
            json.loads(req.template)                      # validate early
            tpath.write_text(req.template, encoding="utf-8")
            target.template = str(tpath)
        adapter = cli.build_adapter(target)

        fast = req.mode != "thorough"
        max_requests = req.max_requests or (400 if fast else None)
        job.max_requests = max_requests
        run_dir = cli.new_run_dir(req.name or job.label)
        job.run_dir = str(run_dir)
        recorder = Recorder(str(run_dir))
        budget = cli.build_budget(max_requests, None, None, None, 0.0, 0.0)
        job.budget = budget
        config: dict[str, Any] = {"fast": fast, "run_id": run_dir.name}
        recorder.meta({"target": target.describe(), "config": config, "seed": 0,
                       "created": datetime.now(timezone.utc).isoformat(), "source": "web"})

        job.status = "running"
        job.add(f"scanning {job.label} ({'quick' if fast else 'thorough'} mode)")
        result = asyncio.run(cli._run_and_close(
            adapter, config=config, budget=budget, recorder=recorder, llm=None, seed=0,
            progress=job.add))
        result.run_dir = str(run_dir)

        job.status = "rendering"
        job.add("writing report and diagrams")
        cli.dump_json(result, run_dir / "run_result.json")
        cli.render_report(result, run_dir, ["html", "md", "json"])
        _SUMMARY_CACHE.pop(str(run_dir), None)
        job.status = "done"
        job.add("done")
    except Exception as e:  # noqa: BLE001
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.add(f"ERROR {job.error}")
    finally:
        job.finished = time.time()
        if key_env:
            os.environ.pop(key_env, None)


# =============================================================================== runs
_SUMMARY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _run_id(d: Path) -> str:
    return d.resolve().relative_to(RUNS_DIR.resolve()).as_posix()


def _run_dir(run_id: str) -> Path:
    d = (RUNS_DIR / run_id).resolve()
    if RUNS_DIR.resolve() not in d.parents or not (d / "run_result.json").exists():
        raise HTTPException(404, "run not found")
    return d


def _top(post: dict, key: str) -> tuple[str, float] | None:
    v = post.get(key)
    if not v:
        return None
    k = max(v, key=v.get)
    return k, round(float(v[k]), 3)


def _summarize(d: Path) -> dict[str, Any]:
    f = d / "run_result.json"
    mt = f.stat().st_mtime
    hit = _SUMMARY_CACHE.get(str(d))
    if hit and hit[0] == mt:
        return hit[1]
    r = json.loads(f.read_text(encoding="utf-8"))
    post = r.get("posterior") or {}
    est: dict[str, Any] = {}
    for e in r.get("estimates") or []:
        if e.get("value") is not None and e["quantity"] not in est:
            est[e["quantity"]] = {"value": e["value"], "ci": e.get("ci"), "level": e.get("level")}
    fams = [{"name": m["name"], "score": round(m["score"], 2)}
            for m in (r.get("family_matches") or [])[:3]]
    tgt = r.get("target") or {}
    bs = r.get("budget_summary") or {}
    s = {
        "id": _run_id(d),
        "label": tgt.get("name") or tgt.get("model") or d.name,
        "model": tgt.get("model"),
        "kind": (r.get("caps") or {}).get("kind"),
        "created": r.get("started_at") or datetime.fromtimestamp(mt).isoformat(),
        "requests": bs.get("requests"),
        "aborted": bool(bs.get("aborted_unreachable")),
        "arch": _top(post, "arch.family"),
        "sparsity": _top(post, "arch.sparsity"),
        "modal": _top(post, "modal.input"),
        "tokenizer": (est.get("tokenizer.family") or {}).get("value"),
        "context": (est.get("context.max_input_tokens") or {}).get("value"),
        "families": fams,
        "files": sorted(p.name for p in d.iterdir() if p.is_file()),
        "notes": [n for n in (r.get("notes") or []) if str(n).startswith("ABORTED")],
    }
    _SUMMARY_CACHE[str(d)] = (mt, s)
    return s


def _all_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for f in RUNS_DIR.rglob("run_result.json"):
        parts = f.relative_to(RUNS_DIR).parts
        if len(parts) > 4 or any(x.startswith("_") for x in parts[:-1]):
            continue          # folders starting with "_" are archived / hidden
        try:
            out.append(_summarize(f.parent))
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda s: s["created"] or "", reverse=True)
    return out


_DETAIL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _detail(d: Path) -> dict[str, Any]:
    """Chart-ready extraction of one run for the dashboard: posterior distributions,
    estimates with CIs, family-match scores, the entropy decline over experiments,
    identifiability level counts and budget usage. Cached on file mtime."""
    f = d / "run_result.json"
    mt = f.stat().st_mtime
    hit = _DETAIL_CACHE.get(str(d))
    if hit and hit[0] == mt:
        return hit[1]
    r = json.loads(f.read_text(encoding="utf-8"))
    post = r.get("posterior") or {}
    prior = r.get("prior") or {}
    props = []
    for k, dist in post.items():
        if not dist:
            continue
        vals = [{"value": v, "prior": round(float((prior.get(k) or {}).get(v, 0.0)), 4),
                 "post": round(float(p), 4)}
                for v, p in sorted(dist.items(), key=lambda kv: -kv[1])[:5]]
        props.append({"key": k, "values": vals})
    props.sort(key=lambda p: p["key"])

    quants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in r.get("estimates") or []:
        q = e.get("quantity")
        if not q or q in seen or e.get("value") is None:
            continue
        seen.add(q)
        quants.append({"key": q, "value": e.get("value"), "ci": e.get("ci"),
                       "level": e.get("level"), "method": e.get("method"),
                       "probe": e.get("probe_id"), "unit": e.get("unit") or "",
                       "ident": e.get("identifiability")})
    quants.sort(key=lambda q: q["key"])

    fams = [{"name": m.get("name"), "score": round(float(m.get("score") or 0.0), 3),
             "matched": list(m.get("matched") or []),
             "conflicts": list(m.get("conflicts") or [])}
            for m in (r.get("family_matches") or [])[:6]]

    steps = [{"step": s.get("step"), "probe": s.get("probe_id"), "status": s.get("status"),
              "req": s.get("n_requests"), "eig": s.get("eig_bits"),
              "h0": s.get("entropy_before_bits"), "h1": s.get("entropy_after_bits")}
             for s in (r.get("experiment_log") or [])]

    ident_levels: dict[str, int] = {}
    for ie in r.get("identifiability") or []:
        lv = str(ie.get("level") or 5)
        ident_levels[lv] = ident_levels.get(lv, 0) + 1
    ev_levels: dict[str, int] = {}
    for ev in r.get("evidence") or []:
        lv = str(ev.get("level") or "unknown")
        ev_levels[lv] = ev_levels.get(lv, 0) + 1

    tgt = r.get("target") or {}
    from forensics.cli import run_result_from_dict
    from forensics.inference.architecture import architecture_hypothesis_graph
    architecture = architecture_hypothesis_graph(run_result_from_dict(r))
    out = {
        "id": _run_id(d),
        "label": tgt.get("name") or tgt.get("model") or d.name,
        "model": tgt.get("model"), "kind": (r.get("caps") or {}).get("kind"),
        "architecture": architecture,
        "score_semantics": "relative hypothesis support; not calibrated model probabilities",
        "created": r.get("started_at"), "finished": r.get("finished_at"),
        "budget": r.get("budget_summary") or {},
        "counts": {"probes": len(r.get("probe_results") or []),
                   "evidence": len(r.get("evidence") or []),
                   "estimates": len(r.get("estimates") or []),
                   "experiments": len(steps)},
        "properties": props, "quantities": quants, "families": fams,
        "experiments": steps, "ident_levels": ident_levels, "evidence_levels": ev_levels,
        "notes": [str(n) for n in (r.get("notes") or [])][:12],
    }
    _DETAIL_CACHE[str(d)] = (mt, out)
    return out


# =============================================================================== app
app = FastAPI(title="Model Forensics", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/presets")
def presets() -> list[dict[str, Any]]:
    from forensics.cli import PRESETS
    load_env()
    out = []
    for name, p in PRESETS.items():
        kenv = p.get("api_key_env")
        out.append({"name": name, "provider": p.get("provider"), "model": p.get("model"),
                    "kind": p.get("kind"),
                    "ready": p.get("provider") == "sim" or bool(kenv and os.environ.get(kenv))})
    return out


@app.get("/api/probes")
def probes() -> list[dict[str, Any]]:
    from forensics.probes import load_all
    return [{"id": c.id, "title": c.title, "kinds": list(c.kinds),
             "requires": list(c.requires), "estimates": list(c.estimates)}
            for _, c in sorted(load_all().items())]


@app.post("/api/scan")
def start_scan(req: ScanRequest) -> dict[str, Any]:
    if not (req.preset or req.provider):
        raise HTTPException(400, "choose a preset or a provider")
    if req.provider and not req.preset and req.provider != "jev" and not (
            req.model or req.template):
        raise HTTPException(400, "model name is required")
    label = req.name or req.preset or req.model or req.provider or "scan"
    job = Job(id=uuid.uuid4().hex[:10], label=label, mode=req.mode)
    JOBS[job.id] = job
    threading.Thread(target=_run_scan, args=(job, req), daemon=True).start()
    return job.view()


@app.get("/api/jobs")
def jobs() -> list[dict[str, Any]]:
    return [j.view() for j in sorted(JOBS.values(), key=lambda j: -j.started)]


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[job_id].view()


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    return _all_runs()


@app.get("/api/runs/{run_id:path}/summary")
def run_summary(run_id: str) -> dict[str, Any]:
    return _summarize(_run_dir(run_id))


@app.get("/api/runs/{run_id:path}/detail")
def run_detail(run_id: str) -> dict[str, Any]:
    return _detail(_run_dir(run_id))


@app.get("/files/{run_id:path}/{name}")
def run_file(run_id: str, name: str):
    d = _run_dir(run_id)
    f = (d / name).resolve()
    if f.parent != d or not f.is_file():
        raise HTTPException(404, "file not found")
    media = {".html": "text/html", ".md": "text/markdown; charset=utf-8",
             ".mmd": "text/plain; charset=utf-8", ".dot": "text/plain; charset=utf-8",
             ".json": "application/json", ".jsonl": "text/plain; charset=utf-8",
             ".svg": "image/svg+xml"}.get(f.suffix, "application/octet-stream")
    return FileResponse(f, media_type=media)


@app.delete("/api/runs/{run_id:path}")
def delete_run(run_id: str) -> dict[str, Any]:
    d = _run_dir(run_id)
    shutil.rmtree(d)
    _SUMMARY_CACHE.pop(str(d), None)
    _DETAIL_CACHE.pop(str(d), None)
    return {"deleted": run_id}


@app.get("/api/compare")
def compare(a: str, b: str) -> JSONResponse:
    from forensics.cli import load_run_result
    from forensics.fingerprint import fingerprint, fingerprint_distance
    ra, rb = load_run_result(_run_dir(a)), load_run_result(_run_dir(b))
    if ra is None or rb is None:
        raise HTTPException(400, "could not load one of the runs")
    fa, fb = fingerprint(ra), fingerprint(rb)
    dist = float(fingerprint_distance(fa, fb))
    sa, sb = _summarize(_run_dir(a)), _summarize(_run_dir(b))
    rows = []
    for key in ("arch", "sparsity", "modal", "tokenizer", "context"):
        va, vb = sa.get(key), sb.get(key)
        same = va[0] == vb[0] if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)) else va == vb
        rows.append({"key": key, "a": va, "b": vb, "same": same,
                     "scores_equal": va == vb})
    from forensics.inference.architecture import compare_architecture_candidates
    candidates = compare_architecture_candidates(ra, rb)
    verdict = ("similar measured signatures" if dist < 0.08 else
               "partly similar measured signatures" if dist < 0.18 else "distinct measured signatures")
    return JSONResponse({"distance": round(dist, 4), "verdict": verdict, "rows": rows,
                         "candidate_comparison": candidates,
                         "score_semantics": "normalized surrogate support; not calibrated probabilities",
                         "identity_caveat": "Fingerprint distance does not prove model identity or replacement.",
                         "a": sa["label"], "b": sb["label"]})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    RUNS_DIR.mkdir(exist_ok=True)
    url = f"http://{host}:{port}"
    if open_browser:
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Model Forensics UI -> {url}   (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
