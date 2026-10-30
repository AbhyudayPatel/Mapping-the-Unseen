"""``forensics`` command-line interface (Typer app; also ``python -m forensics``).

The CLI wires the phase-1 building blocks (adapters, :class:`~forensics.core.budget.Budget`,
:class:`~forensics.core.recorder.Recorder`) to the inference engine
(:func:`forensics.inference.engine.run_forensics`) and the reporting layer, and prints a
rich summary of what the run inferred about a target model.

Commands
--------
* ``run``      — probe a target (``--preset`` or ``--provider ...``), write a run dir with
  ``report.{html,md,json}`` and print a summary.
* ``compare``  — compare two targets (or two saved run dirs) and print an equality verdict.
* ``report``   — regenerate the report from a saved run dir (from the recorded RunResult
  json, or by re-analysing the recorded observations via replay).
* ``diagram``  — emit just the architecture diagrams (mermaid / dot / svg) for a run dir.
* ``probes``   — list the probe registry as a table.
* ``replay``   — re-analyse a recorded run dir offline (``ReplayAdapter``, no network).

Secrets are never printed or written to run artefacts: presets reference environment
*variable names* (resolved at run time via :func:`forensics.config.load_env`), and the
recorder redacts secret-looking header/config values.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forensics.config import ROOT, load_env
from forensics.core.budget import Budget
from forensics.core.recorder import Recorder
from forensics.core.types import Capabilities, to_jsonable
from forensics.inference.engine import run_forensics
from forensics.inference.hypotheses import PROPERTIES, QUANTITIES
from forensics.inference.run_types import (
    ExperimentStep, FamilyMatch, IdentifiabilityEntry, Recommendation, RunResult,
)
from forensics.probes.base import Estimate, Evidence, ProbeResult

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Black-box model-forensics: infer architecture, serving stack and training "
         "of a model API through designed experiments.",
)


# =============================================================================== presets
# Each preset references env-var NAMES only (never secrets). ``provider`` selects the
# adapter; ``sim`` names a SimConfig factory for the offline simulator presets.
PRESETS: dict[str, dict[str, Any]] = {
    # ---- Azure OpenAI-compatible deployments (AZURE_OPENAI_BASE_URL + _API_KEY) -------
    "gpt-4o-mini": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
                    "model": "gpt-4o-mini", "api_key_env": "AZURE_OPENAI_API_KEY",
                    "auth_style": "api-key", "kind": "generative"},
    "gpt-4o": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
               "model": "gpt-4o", "api_key_env": "AZURE_OPENAI_API_KEY",
               "auth_style": "api-key", "kind": "generative"},
    "gpt-5.4-nano": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
                     "model": "gpt-5.4-nano", "api_key_env": "AZURE_OPENAI_API_KEY",
                     "auth_style": "api-key", "kind": "generative"},
    "gpt-5.4": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
                "model": "gpt-5.4", "api_key_env": "AZURE_OPENAI_API_KEY",
                "auth_style": "api-key", "kind": "generative"},
    "DeepSeek-V3.2": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
                      "model": "DeepSeek-V3.2", "api_key_env": "AZURE_OPENAI_API_KEY",
                      "auth_style": "api-key", "kind": "generative"},
    "DeepSeek-V4-Flash": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
                          "model": "DeepSeek-V4-Flash",
                          "api_key_env": "AZURE_OPENAI_API_KEY",
                          "auth_style": "api-key", "kind": "generative"},
    "Kimi-K2.6": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
                  "model": "Kimi-K2.6", "api_key_env": "AZURE_OPENAI_API_KEY",
                  "auth_style": "api-key", "kind": "generative"},
    "FW-MiniMax-M2.5": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
                        "model": "FW-MiniMax-M2.5", "api_key_env": "AZURE_OPENAI_API_KEY",
                        "auth_style": "api-key", "kind": "generative"},
    # ---- TypeSafe Jev decision model (JEV_BASE_URL + JEV_API_KEY) ---------------------
    "jev": {"provider": "jev", "base_url_env": "JEV_BASE_URL", "model": "jev-latest",
            "api_key_env": "JEV_API_KEY", "auth_style": "bearer", "kind": "decision"},
    # ---- offline simulator presets (no network; for tests / demos) --------------------
    "sim-dense": {"provider": "sim", "sim": "sim_decoder_dense",
                  "model": "sim-decoder-dense", "kind": "generative"},
    "sim-moe": {"provider": "sim", "sim": "sim_decoder_moe",
                "model": "sim-decoder-moe", "kind": "generative"},
    "sim-jev": {"provider": "sim", "sim": "sim_jev", "model": "sim-jev",
                "kind": "decision"},
    "sim-ssm": {"provider": "sim", "sim": "sim_ssm", "model": "sim-ssm",
                "kind": "generative"},
    "sim-encoder": {"provider": "sim", "sim": "sim_encoder", "model": "sim-encoder",
                    "kind": "generative"},
    # ---- more Azure deployments (aliases/duplicates are useful for `compare`) -------
    "gpt-5-nano": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5-nano", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.4-mini": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.4-mini", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.5": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.5", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.3-chat": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.3-chat", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-chat-latest": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-chat-latest", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.6-sol": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.6-sol", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.6-sol-2": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.6-sol-2", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.6-terra": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.6-terra", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-6-astra": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-6-astra", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.4-pro": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.4-pro", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.3-codex": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.3-codex", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "gpt-5.2-codex": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "gpt-5.2-codex", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "FW-Kimi-K3": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "FW-Kimi-K3", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},
    "FW-Kimi-K3-2": {"provider": "azure", "base_url_env": "AZURE_OPENAI_BASE_URL",
        "model": "FW-Kimi-K3-2", "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_style": "api-key", "kind": "generative"},

}

_SIM_FACTORIES = {
    "sim_decoder_dense", "sim_decoder_moe", "sim_ssm", "sim_encoder", "sim_jev",
}


# =============================================================================== targets
@dataclass
class Target:
    """A resolved probe target (never holds a secret; keys live in the environment)."""
    provider: str                       # openai | azure | jev | http | sim
    model: str
    kind: str = "generative"
    base_url: str | None = None         # direct URL (overrides base_url_env)
    base_url_env: str | None = None
    api_key_env: str | None = None
    auth_style: str = "api-key"
    sim: str | None = None              # SimConfig factory name for provider == "sim"
    name: str = ""
    template: str | None = None         # provider == "http": path to a TemplateConfig JSON

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "kind": self.kind,
                "base_url_env": self.base_url_env, "api_key_env": self.api_key_env,
                "auth_style": self.auth_style, "sim": self.sim, "name": self.name}


def resolve_target(preset: str | None, provider: str | None, base_url: str | None,
                   model: str | None, api_key_env: str | None,
                   auth_style: str | None) -> Target:
    """Build a :class:`Target` from a preset name or explicit provider parameters."""
    if preset:
        if preset not in PRESETS:
            raise typer.BadParameter(
                f"unknown preset {preset!r}; known: {', '.join(sorted(PRESETS))}")
        p = PRESETS[preset]
        return Target(
            provider=p["provider"], model=p.get("model", preset),
            kind=p.get("kind", "generative"), base_url_env=p.get("base_url_env"),
            api_key_env=p.get("api_key_env"),
            auth_style=p.get("auth_style", "api-key"), sim=p.get("sim"),
            name=preset,
        )
    if not provider:
        raise typer.BadParameter("give either --preset or --provider")
    provider = provider.lower()
    if provider == "sim":
        sim = model or "sim_decoder_dense"
        return Target(provider="sim", model=sim, sim=sim, kind="generative",
                      name=sim)
    if provider in ("openai", "azure", "http"):
        if not model:
            raise typer.BadParameter(f"--model is required for provider {provider!r}")
        kind = "generative"
    elif provider == "jev":
        model = model or "jev-latest"
        kind = "decision"
    else:
        raise typer.BadParameter(
            f"unknown provider {provider!r}; use openai|azure|jev|http")
    return Target(
        provider=provider, model=model, kind=kind, base_url=base_url,
        base_url_env=None, api_key_env=api_key_env,
        auth_style=(auth_style or ("bearer" if provider in ("openai", "jev") else "api-key")),
        name=model,
    )


def build_adapter(target: Target, *, api_version: str | None = None):
    """Instantiate the concrete adapter for ``target`` (reads env for URL/key)."""
    prov = target.provider
    if prov == "sim":
        from forensics.core.adapters import simulated as _sim
        factory_name = target.sim or target.model
        if factory_name not in _SIM_FACTORIES:
            raise typer.BadParameter(
                f"unknown simulator {factory_name!r}; known: {', '.join(sorted(_SIM_FACTORIES))}")
        factory = getattr(_sim, factory_name)
        return _sim.SimulatedAdapter(factory())

    base_url = target.base_url
    if not base_url and target.base_url_env:
        base_url = os.environ.get(target.base_url_env)
    api_key = os.environ.get(target.api_key_env, "") if target.api_key_env else ""

    if prov in ("openai", "azure"):
        from forensics.core.adapters.openai_compat import OpenAICompatAdapter
        if not base_url:
            raise typer.BadParameter(
                "no base URL: set --base-url or the preset's base_url_env "
                f"({target.base_url_env})")
        return OpenAICompatAdapter(
            base_url=base_url, api_key=api_key, model=target.model,
            auth_style=target.auth_style, api_version=api_version, name=target.name)
    if prov == "jev":
        from forensics.core.adapters.decision import DecisionAdapter
        return DecisionAdapter(base_url=base_url, api_key=api_key, model=target.model,
                               name=target.name)
    if prov == "http":
        from forensics.core.adapters.http_template import HTTPTemplateAdapter, TemplateConfig
        if target.template:
            # Fully custom endpoint: request body template + JSONPath extraction rules.
            spec = json.loads(Path(target.template).read_text(encoding="utf-8"))
            if base_url:
                spec["url"] = base_url
            if target.api_key_env and not spec.get("api_key_env"):
                spec["api_key_env"] = target.api_key_env
            cfg = TemplateConfig.from_dict(spec)
            key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else api_key
            return HTTPTemplateAdapter(cfg, api_key=key or None)
        if not base_url:
            raise typer.BadParameter("--base-url (or --template) is required for provider 'http'")
        cfg = TemplateConfig(
            url=base_url,
            body_template={"model": target.model, "messages": "{messages_json}",
                           "max_tokens": "{max_tokens}", "temperature": "{temperature}"},
            headers={"content-type": "application/json",
                     ("Authorization" if target.auth_style == "bearer" else "api-key"):
                     ("Bearer {api_key}" if target.auth_style == "bearer" else "{api_key}")},
            api_key_env=target.api_key_env, name=target.name)
        return HTTPTemplateAdapter(cfg, api_key=api_key or None)
    raise typer.BadParameter(f"cannot build adapter for provider {prov!r}")


# =============================================================================== budget
def build_budget(max_requests: int | None, max_cost: float | None,
                 max_input_tokens: int | None, max_output_tokens: int | None,
                 price_in: float, price_out: float) -> Budget:
    """Always return a Budget (uncapped ceilings stay None) so we get accounting."""
    return Budget(
        max_requests=max_requests, max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens, max_cost_usd=max_cost,
        price_in_per_m=price_in, price_out_per_m=price_out)


# =============================================================================== llm advisor
def make_advisor(enabled: bool) -> Any:
    """Best-effort Azure advisor (may be unavailable / None); never raises."""
    if not enabled:
        return None
    try:
        from forensics.llm.advisor import make_default_advisor  # type: ignore
    except Exception as e:  # noqa: BLE001
        err_console.print(f"[yellow]--llm requested but advisor unavailable ({e!r})[/]")
        return None
    try:
        adv = make_default_advisor()
    except Exception as e:  # noqa: BLE001
        err_console.print(f"[yellow]advisor init failed ({e!r}); continuing without it[/]")
        return None
    if adv is None:
        err_console.print("[yellow]advisor is None (no Azure config); continuing[/]")
    return adv


# =============================================================================== run dir
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "run"


def new_run_dir(name: str, runs_dir: Path | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = runs_dir or (ROOT / "runs")
    d = base / f"{ts}-{_slug(name)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================== (de)serialize
def _json_default(o: Any) -> Any:
    """Fallback JSON encoder: numpy scalars/arrays, sets, dataclasses, etc."""
    if hasattr(o, "item"):          # numpy scalar
        try:
            return o.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(o, "tolist"):        # numpy array
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if hasattr(o, "__dataclass_fields__"):
        return to_jsonable(o)
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.write_text(
        json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")


def _mk(cls: type, d: dict[str, Any]) -> Any:
    fields = getattr(cls, "__dataclass_fields__", {})
    return cls(**{k: v for k, v in d.items() if k in fields})


def _estimate_from_dict(d: dict[str, Any]) -> Estimate:
    e = _mk(Estimate, d)
    if isinstance(e.ci, list):
        e.ci = tuple(e.ci)  # type: ignore[assignment]
    return e


def _probe_result_from_dict(d: dict[str, Any]) -> ProbeResult:
    pr = _mk(ProbeResult, {k: v for k, v in d.items()
                           if k not in ("evidence", "estimates")})
    pr.evidence = [_mk(Evidence, x) for x in d.get("evidence", []) if isinstance(x, dict)]
    pr.estimates = [_estimate_from_dict(x) for x in d.get("estimates", [])
                    if isinstance(x, dict)]
    return pr


def run_result_from_dict(d: dict[str, Any]) -> RunResult:
    """Reconstruct a :class:`RunResult` from its ``to_jsonable`` dict form."""
    caps_d = d.get("caps") or {}
    caps = _mk(Capabilities, caps_d) if isinstance(caps_d, dict) else Capabilities()
    rr = RunResult(
        run_id=d.get("run_id", ""),
        target=d.get("target") or {},
        caps=caps,
        started_at=d.get("started_at", ""),
        finished_at=d.get("finished_at", ""),
        config=d.get("config") or {},
        prior=d.get("prior") or {},
        posterior=d.get("posterior") or {},
        fingerprint=d.get("fingerprint") or {},
        budget_summary=d.get("budget_summary") or {},
        notes=list(d.get("notes") or []),
        llm_assisted=list(d.get("llm_assisted") or []),
        run_dir=d.get("run_dir", ""),
    )
    rr.probe_results = [_probe_result_from_dict(x) for x in d.get("probe_results", [])
                        if isinstance(x, dict)]
    rr.evidence = [_mk(Evidence, x) for x in d.get("evidence", []) if isinstance(x, dict)]
    rr.estimates = [_estimate_from_dict(x) for x in d.get("estimates", [])
                    if isinstance(x, dict)]
    rr.identifiability = [_mk(IdentifiabilityEntry, x)
                          for x in d.get("identifiability", []) if isinstance(x, dict)]
    rr.family_matches = [_mk(FamilyMatch, x) for x in d.get("family_matches", [])
                         if isinstance(x, dict)]
    rr.experiment_log = [_mk(ExperimentStep, x) for x in d.get("experiment_log", [])
                         if isinstance(x, dict)]
    rr.recommendations = [_mk(Recommendation, x) for x in d.get("recommendations", [])
                          if isinstance(x, dict)]
    return rr


def load_run_result(run_dir: Path) -> RunResult | None:
    for fname in ("run_result.json", "report.json"):
        p = run_dir / fname
        if p.exists():
            try:
                return run_result_from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
    return None


# =============================================================================== formatting helpers
def _top(posterior: dict[str, dict[str, float]], prop: str) -> tuple[str, float] | None:
    dist = posterior.get(prop) or {}
    if not dist:
        return None
    k = max(dist, key=lambda x: dist[x])
    return k, float(dist[k])


def _value_label(prop: str, value: str) -> str:
    p = PROPERTIES.get(prop)
    if p and value in p.value_docs:
        return f"{value} — {p.value_docs[value]}"
    return str(value)


def _est_map(estimates: list[Estimate]) -> dict[str, Estimate]:
    """Last estimate wins per quantity (estimator refinements come after probe outputs)."""
    out: dict[str, Estimate] = {}
    for e in estimates:
        if e.value is not None:
            out[e.quantity] = e
    return out


def _fmt_num(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if abs(x) >= 1000 or (x == int(x) and abs(x) >= 1):
            return f"{x:,.0f}"
        return f"{x:g}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def _fmt_ci(ci: Any) -> str:
    if not ci or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return ""
    return f"[{_fmt_num(ci[0])}, {_fmt_num(ci[1])}]"


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return _fmt_num(v)
    return str(v)


# =============================================================================== diagrams
#: Quantities surfaced in the architecture diagram / summary, in display order.
_KEY_QUANTITIES = [
    "dims.hidden_size", "dims.n_layers", "dims.n_heads", "dims.n_kv_heads",
    "dims.ffn_dim", "dims.vocab_size", "dims.total_params", "dims.active_params",
    "dims.n_experts", "context.max_input_tokens", "context.max_output_tokens",
    "tokenizer.family", "out.logit_dtype",
]


def _diagram_facts(run: RunResult) -> dict[str, str]:
    facts: dict[str, str] = {}
    for prop in ("arch.family", "arch.sparsity", "attn.kv_sharing", "attn.long_range",
                 "pos.encoding", "system.topology", "system.safety_middleware"):
        t = _top(run.posterior, prop)
        if t:
            facts[prop] = f"{t[0]} ({t[1] * 100:.0f}%)"
    ests = _est_map(run.estimates)
    for q in _KEY_QUANTITIES:
        if q in ests:
            unit, _ = QUANTITIES.get(q, ("", ""))
            v = _fmt_value(ests[q].value)
            facts[q] = f"{v} {unit}".strip()
    return facts


def build_mermaid(run: RunResult) -> str:
    facts = _diagram_facts(run)
    fam = facts.get("arch.family", "?")
    spars = facts.get("arch.sparsity", "")
    topo = _top(run.posterior, "system.topology")
    safety = _top(run.posterior, "system.safety_middleware")
    ests = _est_map(run.estimates)
    dim_lines = []
    for q in _KEY_QUANTITIES:
        if q in ests:
            unit, _ = QUANTITIES.get(q, ("", ""))
            short = q.split(".", 1)[1]
            dim_lines.append(f"{short}={_fmt_value(ests[q].value)}{(' ' + unit) if unit else ''}")
    dims = "<br/>".join(dim_lines) or "no measured dimensions"
    target_name = run.target.get("name") or run.target.get("model") or run.run_id

    def esc(s: str) -> str:
        return s.replace('"', "'")

    lines = ["flowchart LR",
             f'  U([Client])'
             ]
    pre = safety and safety[0] in ("pre_filter", "both")
    post = safety and safety[0] in ("post_filter", "both")
    if pre:
        lines.append('  PRE["safety pre-filter"]')
        lines.append("  U --> PRE --> M")
    else:
        lines.append("  U --> M")
    lines.append(f'  M["{esc(target_name)}<br/>family: {esc(fam)}'
                 f'{("<br/>" + esc(spars)) if spars else ""}"]')
    lines.append(f'  D["dimensions<br/>{esc(dims)}"]')
    lines.append("  M --- D")
    if post:
        lines.append('  POST["safety post-filter"]')
        lines.append("  M --> POST --> U2([Client])")
    else:
        lines.append("  M --> U2([Client])")
    if topo:
        lines.append(f'  T["topology: {esc(topo[0])}"]')
        lines.append("  M --- T")
    return "\n".join(lines) + "\n"


def build_dot(run: RunResult) -> str:
    facts = _diagram_facts(run)
    fam = facts.get("arch.family", "?")
    spars = facts.get("arch.sparsity", "")
    ests = _est_map(run.estimates)
    dim_lines = []
    for q in _KEY_QUANTITIES:
        if q in ests:
            unit, _ = QUANTITIES.get(q, ("", ""))
            short = q.split(".", 1)[1]
            dim_lines.append(f"{short}={_fmt_value(ests[q].value)}{(' ' + unit) if unit else ''}")
    dims = "\\n".join(dim_lines) or "no measured dimensions"
    target_name = run.target.get("name") or run.target.get("model") or run.run_id

    def esc(s: str) -> str:
        return s.replace('"', "'").replace("\n", " ")

    model_label = f"{esc(target_name)}\\nfamily: {esc(fam)}"
    if spars:
        model_label += f"\\n{esc(spars)}"
    return (
        "digraph forensics {\n"
        "  rankdir=LR;\n"
        '  node [shape=box, style=rounded, fontname="Helvetica"];\n'
        '  U [label="Client", shape=ellipse];\n'
        f'  M [label="{model_label}"];\n'
        f'  D [label="dimensions\\n{dims}"];\n'
        "  U -> M;\n"
        "  M -> D [dir=none, style=dashed];\n"
        "  M -> U [label=response];\n"
        "}\n"
    )


def build_diagrams(run: RunResult, out_dir: Path) -> dict[str, str]:
    """Write diagram sources for the architecture, system topology and evidence flow
    (.mmd + .dot each, plus .svg when graphviz ``dot`` exists). Uses the full
    ``forensics.report.diagrams`` generators; falls back to the simple built-in writer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from forensics.report import diagrams as _dg
        written: dict[str, str] = {}
        for stem, fn in (("architecture", _dg.block_architecture),
                         ("topology", _dg.system_topology),
                         ("evidence", _dg.evidence_flow)):
            d = fn(run)
            (out_dir / f"{stem}.mmd").write_text(d.mermaid, encoding="utf-8")
            (out_dir / f"{stem}.dot").write_text(d.dot, encoding="utf-8")
            written[f"{stem}_mermaid"] = str(out_dir / f"{stem}.mmd")
            written[f"{stem}_dot"] = str(out_dir / f"{stem}.dot")
            svg = d.svg or _dg.render_svg(d.dot)   # hand-rendered figure beats graphviz
            if svg:
                (out_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")
                written[f"{stem}_svg"] = str(out_dir / f"{stem}.svg")
        if not shutil.which("dot"):
            written["svg_error"] = "graphviz 'dot' not installed; emitted .mmd/.dot only"
        return written
    except Exception as e:  # noqa: BLE001
        err_console.print(f"[yellow]full diagram generator failed ({e!r}); using simple diagrams[/]")
    written = {}
    # The simple writer must never raise: a run should ALWAYS leave architecture.mmd/.dot on
    # disk (downstream commands and the report reader depend on them), even if fact extraction
    # trips over an unusual estimate. Fall back to a minimal placeholder diagram on error.
    mmd = out_dir / "architecture.mmd"
    try:
        mmd_text = build_mermaid(run)
    except Exception as e:  # noqa: BLE001
        err_console.print(f"[yellow]mermaid generation failed ({e!r}); minimal diagram[/]")
        mmd_text = f'flowchart LR\n  M["{run.run_id}<br/>diagram unavailable"]\n'
    mmd.write_text(mmd_text, encoding="utf-8")
    written["mermaid"] = str(mmd)
    dot = out_dir / "architecture.dot"
    try:
        dot_text = build_dot(run)
    except Exception as e:  # noqa: BLE001
        dot_text = 'digraph forensics {\n  M [label="diagram unavailable"];\n}\n'
    dot.write_text(dot_text, encoding="utf-8")
    written["dot"] = str(dot)
    dot_bin = shutil.which("dot")
    if dot_bin:
        svg = out_dir / "architecture.svg"
        try:
            subprocess.run([dot_bin, "-Tsvg", str(dot), "-o", str(svg)],
                           check=True, capture_output=True, timeout=30)
            written["svg"] = str(svg)
        except Exception as e:  # noqa: BLE001
            written["svg_error"] = f"graphviz failed: {e!r}"
    else:
        written["svg_error"] = "graphviz 'dot' not installed; emitted .mmd/.dot only"
    return written


# =============================================================================== fallback report
def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out) + "\n"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    from html import escape
    if not rows:
        return "<p><em>none</em></p>"
    h = "".join(f"<th>{escape(str(c))}</th>" for c in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _report_rows(run: RunResult) -> dict[str, tuple[list[str], list[list[str]]]]:
    # architecture properties
    arch_rows: list[list[str]] = []
    for prop in sorted(run.posterior):
        top = _top(run.posterior, prop)
        if not top:
            continue
        prior = run.prior.get(prop, {})
        prior_p = prior.get(top[0])
        arch_rows.append([prop, top[0], f"{top[1] * 100:.1f}%",
                          f"{prior_p * 100:.1f}%" if prior_p is not None else ""])
    # estimates
    est_rows: list[list[str]] = []
    for q, e in _est_map(run.estimates).items():
        unit, _desc = QUANTITIES.get(q, ("", ""))
        est_rows.append([q, f"{_fmt_value(e.value)}{(' ' + unit) if unit else ''}",
                         _fmt_ci(e.ci), e.level, str(e.identifiability),
                         e.method or "", e.probe_id or ""])
    # families
    fam_rows = [[f.name, f"{f.score:.3f}", ", ".join(f.matched[:4]),
                 ", ".join(f.conflicts[:4])] for f in run.family_matches]
    # identifiability
    idf_rows = [[i.key, str(i.level), i.label, i.reason] for i in run.identifiability]
    # experiment log
    log_rows = [[str(s.step), s.probe_id, s.reason, s.status,
                 "" if s.eig_bits is None else f"{s.eig_bits:.3f}", str(s.n_requests)]
                for s in run.experiment_log]
    # recommendations
    rec_rows = [[r.what, r.why,
                 "" if r.expected_gain_bits is None else f"{r.expected_gain_bits:.3f}",
                 r.needs] for r in run.recommendations]
    return {
        "arch": (["property", "top value", "posterior", "prior"], arch_rows),
        "est": (["quantity", "value", "CI", "level", "ident.", "method", "probe"], est_rows),
        "fam": (["family", "score", "matched", "conflicts"], fam_rows),
        "idf": (["key", "level", "label", "reason"], idf_rows),
        "log": (["#", "probe", "reason", "status", "EIG bits", "requests"], log_rows),
        "rec": (["what", "why", "gain bits", "needs"], rec_rows),
    }


def fallback_build_report(run: RunResult, out_dir: Path, formats: list[str]) -> list[str]:
    """Built-in report writer used until ``forensics.report.build_report`` lands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    sections = _report_rows(run)
    fam = _top(run.posterior, "arch.family")
    fam_line = (f"{_value_label('arch.family', fam[0])} ({fam[1] * 100:.1f}%)"
                if fam else "unknown")
    title = run.target.get("name") or run.target.get("model") or run.run_id
    diagrams = build_diagrams(run, out_dir)
    mermaid_src = (out_dir / "architecture.mmd").read_text(encoding="utf-8")

    if "json" in formats:
        p = out_dir / "report.json"
        dump_json(run, p)
        written.append(str(p))

    if "md" in formats:
        lines = [f"# Model forensics report — {title}", "",
                 f"- run id: `{run.run_id}`",
                 f"- target: `{json.dumps(run.target, default=str)}`",
                 f"- interface kind: `{run.caps.kind}`",
                 f"- started: {run.started_at}  finished: {run.finished_at}",
                 f"- top architecture hypothesis: **{fam_line}**", ""]
        if run.llm_assisted:
            lines.append(f"- LLM-assisted parts: {', '.join(run.llm_assisted)}\n")
        lines += ["## Architecture properties (posterior)", "",
                  _md_table(*sections["arch"]), "",
                  "## Measured quantities", "", _md_table(*sections["est"]), "",
                  "## Family matches", "", _md_table(*sections["fam"]), "",
                  "## Identifiability", "", _md_table(*sections["idf"]), "",
                  "## Experiment log", "", _md_table(*sections["log"]), "",
                  "## Recommendations", "", _md_table(*sections["rec"]), "",
                  "## Architecture diagram (mermaid)", "",
                  "```mermaid", mermaid_src.rstrip(), "```", "",
                  "## Budget", "",
                  f"```json\n{json.dumps(run.budget_summary, indent=2, default=str)}\n```",
                  "", "## Notes", ""]
        lines += [f"- {n}" for n in run.notes] or ["_none_"]
        p = out_dir / "report.md"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(str(p))

    if "html" in formats:
        from html import escape
        css = ("body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;"
               "margin:2rem auto;max-width:1000px;padding:0 1rem;color:#1a1a1a;"
               "line-height:1.5}"
               "table{border-collapse:collapse;width:100%;margin:0.5rem 0 1.5rem}"
               "th,td{border:1px solid #d0d0d0;padding:4px 8px;text-align:left;"
               "font-size:14px;vertical-align:top}"
               "th{background:#f3f4f6}h1{margin-bottom:0.2rem}"
               "code,pre{background:#f6f8fa;border-radius:4px}"
               "pre{padding:10px;overflow:auto}"
               ".kv{color:#444;font-size:14px}")
        body = [f"<h1>Model forensics — {escape(str(title))}</h1>",
                f"<p class=kv>run id <code>{escape(run.run_id)}</code> · "
                f"kind <code>{escape(str(run.caps.kind))}</code> · "
                f"started {escape(run.started_at)} · finished {escape(run.finished_at)}</p>",
                f"<p><strong>Top architecture hypothesis:</strong> {escape(fam_line)}</p>"]
        if run.llm_assisted:
            body.append(f"<p class=kv>LLM-assisted: {escape(', '.join(run.llm_assisted))}</p>")
        body += ["<h2>Architecture properties (posterior)</h2>", _html_table(*sections["arch"]),
                 "<h2>Measured quantities</h2>", _html_table(*sections["est"]),
                 "<h2>Family matches</h2>", _html_table(*sections["fam"]),
                 "<h2>Identifiability</h2>", _html_table(*sections["idf"]),
                 "<h2>Experiment log</h2>", _html_table(*sections["log"]),
                 "<h2>Recommendations</h2>", _html_table(*sections["rec"]),
                 "<h2>Architecture diagram</h2>",
                 f'<pre class="mermaid">{escape(mermaid_src)}</pre>',
                 "<h2>Budget</h2>",
                 f"<pre>{escape(json.dumps(run.budget_summary, indent=2, default=str))}</pre>",
                 "<h2>Notes</h2><ul>"]
        body += [f"<li>{escape(str(n))}</li>" for n in run.notes] or ["<li>none</li>"]
        body.append("</ul>")
        mermaid_js = ('<script type="module">import mermaid from '
                      '"https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
                      "mermaid.initialize({startOnLoad:true});</script>")
        html = (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
                f"<meta name=viewport content='width=device-width,initial-scale=1'>"
                f"<title>Model forensics — {escape(str(title))}</title>"
                f"<style>{css}</style></head><body>"
                + "\n".join(body) + mermaid_js + "</body></html>")
        p = out_dir / "report.html"
        p.write_text(html, encoding="utf-8")
        written.append(str(p))

    for key in ("mermaid", "dot", "svg"):
        if key in diagrams:
            written.append(diagrams[key])
    return written


def render_report(run: RunResult, out_dir: Path, formats: list[str]) -> tuple[list[str], str]:
    """Use ``forensics.report.build_report`` if present, else the built-in fallback."""
    from forensics.inference.certificates import export_certificates
    export_certificates(run, out_dir)
    try:
        import forensics.report as _rep  # type: ignore
        bp = getattr(_rep, "build_report", None)
    except Exception:  # noqa: BLE001
        bp = None
    if bp is not None:
        try:
            out = bp(run, str(out_dir), formats)
            paths = list(out) if isinstance(out, (list, tuple)) else \
                [str(v) for v in out.values()] if isinstance(out, dict) else []
            # ensure the canonical json exists for downstream commands
            if not (out_dir / "report.json").exists():
                dump_json(run, out_dir / "report.json")
            # standalone diagram sources (architecture/topology/evidence .mmd/.dot/.svg)
            try:
                paths += [v for k, v in build_diagrams(run, out_dir).items()
                          if not k.endswith("_error")]
            except Exception as e:  # noqa: BLE001
                err_console.print(f"[yellow]diagram export failed: {e!r}[/]")
            return paths, "forensics.report.build_report"
        except Exception as e:  # noqa: BLE001
            err_console.print(
                f"[yellow]forensics.report.build_report failed ({e!r}); "
                f"using built-in report[/]")
    return fallback_build_report(run, out_dir, formats), "builtin"


# =============================================================================== summary
def print_summary(run: RunResult, report_paths: list[str], run_dir: Path) -> None:
    title = run.target.get("name") or run.target.get("model") or run.run_id
    fam = _top(run.posterior, "arch.family")
    header = f"[bold]{title}[/]  ·  run [cyan]{run.run_id}[/]  ·  kind [magenta]{run.caps.kind}[/]"
    if fam:
        header += (f"\n[bold green]top architecture:[/] "
                   f"{_value_label('arch.family', fam[0])}  "
                   f"([bold]{fam[1] * 100:.0f}%[/])")
    console.print(Panel(header, title="model forensics", expand=False))
    for n in run.notes:
        if str(n).startswith("ABORTED"):
            console.print(Panel(f"[bold red]{n}[/]", title="run incomplete", expand=False))
    terr = (run.budget_summary or {}).get("transport_errors")
    if terr:
        err_console.print(f"[yellow]{terr} request(s) failed at the transport level "
                          f"(network/DNS); their probes' evidence was discarded[/]")

    # architecture posterior highlights
    arch = Table(title="architecture posterior (top values)", show_lines=False)
    for c in ("property", "value", "posterior", "prior"):
        arch.add_column(c)
    for prop in ("arch.family", "arch.sparsity", "attn.long_range", "attn.kv_sharing",
                 "pos.encoding", "system.topology", "system.hidden_reasoning",
                 "out.logit_precision"):
        t = _top(run.posterior, prop)
        if not t:
            continue
        prior = (run.prior.get(prop) or {}).get(t[0])
        arch.add_row(prop, t[0], f"{t[1] * 100:.1f}%",
                     f"{prior * 100:.1f}%" if prior is not None else "-")
    console.print(arch)

    # key estimates with CIs
    ests = _est_map(run.estimates)
    if ests:
        et = Table(title="key measured quantities", show_lines=False)
        for c in ("quantity", "value", "90% CI", "level", "ident."):
            et.add_column(c)
        shown = [q for q in _KEY_QUANTITIES if q in ests]
        shown += [q for q in ests if q not in shown]
        for q in shown[:16]:
            e = ests[q]
            unit, _ = QUANTITIES.get(q, ("", ""))
            et.add_row(q, f"{_fmt_value(e.value)}{(' ' + unit) if unit else ''}",
                       _fmt_ci(e.ci) or "-", e.level, str(e.identifiability))
        console.print(et)

    # identifiability highlights (hardest to pin down)
    if run.identifiability:
        hard = sorted(run.identifiability, key=lambda i: -i.level)[:6]
        it = Table(title="identifiability highlights", show_lines=False)
        for c in ("key", "level", "label", "reason"):
            it.add_column(c, overflow="fold")
        for i in hard:
            it.add_row(i.key, str(i.level), i.label, i.reason)
        console.print(it)

    # family matches
    if run.family_matches:
        ft = Table(title="closest reference families", show_lines=False)
        for c in ("family", "score", "matched", "conflicts"):
            ft.add_column(c, overflow="fold")
        for f in run.family_matches[:6]:
            ft.add_row(f.name, f"{f.score:.3f}", ", ".join(f.matched[:4]),
                       ", ".join(f.conflicts[:3]))
        console.print(ft)

    bs = run.budget_summary or {}
    if bs:
        console.print(f"[dim]budget: {bs.get('requests', '?')} requests, "
                      f"{bs.get('input_tokens', 0)} in / {bs.get('output_tokens', 0)} out tokens, "
                      f"est ${bs.get('est_cost_usd', 0.0):.4f}[/]")
    if run.llm_assisted:
        console.print(f"[dim]LLM-assisted: {', '.join(run.llm_assisted)}[/]")

    html = next((p for p in report_paths if p.endswith(".html")), None)
    md = next((p for p in report_paths if p.endswith(".md")), None)
    console.print(Panel(
        f"run dir: [cyan]{run_dir}[/]\n"
        f"html:    {html or '(not generated)'}\n"
        f"md:      {md or '(not generated)'}\n"
        f"json:    {run_dir / 'report.json'}",
        title="artefacts", expand=False))


# =============================================================================== engine call
async def _run_and_close(adapter, *, config, budget, recorder, llm, seed,
                         progress) -> RunResult:
    try:
        return await run_forensics(adapter, config=config, budget=budget,
                                   recorder=recorder, llm=llm, seed=seed,
                                   progress=progress)
    finally:
        try:
            await adapter.aclose()
        except Exception:  # noqa: BLE001
            pass


def _progress_cb(verbose: bool) -> Callable[[str], None] | None:
    if not verbose:
        return None
    return lambda m: err_console.print(f"[dim]· {m}[/]")


# =============================================================================== commands
@app.command()
def run(
    preset: Optional[str] = typer.Option(None, "--preset", help="named target from PRESETS"),
    provider: Optional[str] = typer.Option(None, "--provider",
                                           help="openai|azure|jev|http|sim"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    model: Optional[str] = typer.Option(None, "--model"),
    api_key_env: Optional[str] = typer.Option(None, "--api-key-env",
                                              help="env var NAME holding the API key"),
    auth_style: Optional[str] = typer.Option(None, "--auth-style", help="api-key|bearer"),
    api_version: Optional[str] = typer.Option(None, "--api-version"),
    template: Optional[str] = typer.Option(
        None, "--template",
        help="provider=http: JSON TemplateConfig (body template + JSONPath extraction) "
             "for arbitrary non-OpenAI endpoints; see examples/http_template_example.json"),
    fast: bool = typer.Option(True, "--fast/--thorough",
                              help="default: quick scan (~300 requests, a few minutes). "
                                   "--thorough runs full-size experiments"),
    seed: int = typer.Option(0, "--seed"),
    replay: Optional[str] = typer.Option(None, "--replay",
                                         help="re-analyse a recorded run dir offline"),
    llm: bool = typer.Option(False, "--llm", help="enable the Azure LLM advisor"),
    max_requests: Optional[int] = typer.Option(None, "--max-requests"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="USD ceiling"),
    max_input_tokens: Optional[int] = typer.Option(None, "--max-input-tokens"),
    max_output_tokens: Optional[int] = typer.Option(None, "--max-output-tokens"),
    price_in: float = typer.Option(0.0, "--price-in", help="USD per 1M input tokens"),
    price_out: float = typer.Option(0.0, "--price-out", help="USD per 1M output tokens"),
    name: Optional[str] = typer.Option(None, "--name", help="run label (run-dir suffix)"),
    runs_dir: Optional[str] = typer.Option(None, "--runs-dir"),
    formats: str = typer.Option("html,md,json", "--formats"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Probe a target model, write a run dir with a report and print a summary."""
    load_env()
    fmts = [f.strip() for f in formats.split(",") if f.strip()]

    if replay:
        from forensics.core.adapters.replay import ReplayAdapter
        adapter = ReplayAdapter(replay)
        label = name or f"replay-{Path(replay).name}"
        target_desc = {"name": label, "mode": "replay", "source": replay}
    else:
        if template and not model:
            model = "custom-http"
        target = resolve_target(preset, provider, base_url, model, api_key_env, auth_style)
        if template:
            target.template = template
        adapter = build_adapter(target, api_version=api_version)
        label = name or target.name
        target_desc = target.describe()

    run_dir = new_run_dir(label, Path(runs_dir) if runs_dir else None)
    recorder = Recorder(str(run_dir))
    if max_requests is None and fast:
        max_requests = 400          # quick scan default ceiling; override with --max-requests
    budget = build_budget(max_requests, max_cost, max_input_tokens, max_output_tokens,
                          price_in, price_out)
    advisor = make_advisor(llm)
    config: dict[str, Any] = {"fast": fast, "run_id": run_dir.name}

    recorder.meta({"target": target_desc, "config": config, "seed": seed,
                   "created": datetime.now(timezone.utc).isoformat()})

    err_console.print(f"[bold]probing[/] {label} -> {run_dir}")
    result = asyncio.run(_run_and_close(
        adapter, config=config, budget=budget, recorder=recorder, llm=advisor,
        seed=seed, progress=_progress_cb(verbose)))
    result.run_dir = str(run_dir)

    dump_json(result, run_dir / "run_result.json")
    paths, engine_used = render_report(result, run_dir, fmts)
    err_console.print(f"[dim]report via {engine_used}[/]")
    print_summary(result, paths, run_dir)


@app.command()
def replay(
    run_dir: str = typer.Argument(..., help="recorded run dir to re-analyse"),
    fast: bool = typer.Option(False, "--fast"),
    seed: int = typer.Option(0, "--seed"),
    out: Optional[str] = typer.Option(None, "--out", help="output dir (default: <dir>/replay)"),
    formats: str = typer.Option("html,md,json", "--formats"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-analyse a recorded run offline with ReplayAdapter (no network)."""
    from forensics.core.adapters.replay import ReplayAdapter
    src = Path(run_dir)
    if not (src / "observations.jsonl").exists():
        raise typer.BadParameter(f"no observations.jsonl in {run_dir}")
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    out_dir = Path(out) if out else (src / "replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = ReplayAdapter(str(src))
    recorder = Recorder(str(out_dir))
    config: dict[str, Any] = {"fast": fast, "run_id": out_dir.name}
    err_console.print(f"[bold]replaying[/] {src} -> {out_dir}")
    result = asyncio.run(_run_and_close(
        adapter, config=config, budget=build_budget(None, None, None, None, 0.0, 0.0),
        recorder=recorder, llm=None, seed=seed, progress=_progress_cb(verbose)))
    result.run_dir = str(out_dir)

    dump_json(result, out_dir / "run_result.json")
    paths, _ = render_report(result, out_dir, fmts)
    print_summary(result, paths, out_dir)


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="run dir with run_result.json / observations"),
    formats: str = typer.Option("html,md,json", "--formats"),
    replay: bool = typer.Option(False, "--replay",
                                help="ignore saved RunResult; re-analyse observations"),
    out: Optional[str] = typer.Option(None, "--out", help="output dir (default: run dir)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Regenerate the report for a saved run dir."""
    src = Path(run_dir)
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    out_dir = Path(out) if out else src
    out_dir.mkdir(parents=True, exist_ok=True)

    result: RunResult | None = None
    if not replay:
        result = load_run_result(src)
    if result is None:
        from forensics.core.adapters.replay import ReplayAdapter
        if not (src / "observations.jsonl").exists():
            raise typer.BadParameter(
                f"{run_dir}: no run_result.json and no observations.jsonl to replay")
        err_console.print(f"[dim]re-analysing observations from {src}[/]")
        adapter = ReplayAdapter(str(src))
        result = asyncio.run(_run_and_close(
            adapter, config={"fast": False, "run_id": src.name},
            budget=build_budget(None, None, None, None, 0.0, 0.0),
            recorder=None, llm=None, seed=0, progress=_progress_cb(verbose)))
        result.run_dir = str(src)
        dump_json(result, src / "run_result.json")

    paths, engine_used = render_report(result, out_dir, fmts)
    err_console.print(f"[dim]report via {engine_used}[/]")
    print_summary(result, paths, out_dir)


@app.command()
def diagram(
    run_dir: str = typer.Argument(..., help="run dir with run_result.json / observations"),
    out: Optional[str] = typer.Option(None, "--out", help="output dir (default: run dir)"),
) -> None:
    """Emit just the architecture diagrams (mermaid / dot / svg) for a run dir."""
    src = Path(run_dir)
    result = load_run_result(src)
    if result is None:
        raise typer.BadParameter(
            f"{run_dir}: no run_result.json (run `forensics report {run_dir}` first)")
    out_dir = Path(out) if out else src
    written = build_diagrams(result, out_dir)
    t = Table(title="diagrams", show_header=True)
    t.add_column("kind")
    t.add_column("path / note")
    for k in ("mermaid", "dot", "svg", "svg_error"):
        if k in written:
            t.add_row(k, written[k])
    console.print(t)


@app.command()
def compare(
    a: str = typer.Argument(..., help="preset name (or run dir with --from-runs)"),
    b: str = typer.Argument(..., help="preset name (or run dir with --from-runs)"),
    from_runs: bool = typer.Option(False, "--from-runs",
                                   help="treat A and B as saved run dirs"),
    fast: bool = typer.Option(True, "--fast/--no-fast"),
    seed: int = typer.Option(0, "--seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compare two targets (or two saved runs) and print an equality verdict."""
    load_env()
    if from_runs:
        ra = load_run_result(Path(a))
        rb = load_run_result(Path(b))
        if ra is None or rb is None:
            raise typer.BadParameter("both run dirs need a run_result.json (or observations)")
        name_a, name_b = a, b
    else:
        ra = _quick_run(a, fast=fast, seed=seed, verbose=verbose)
        rb = _quick_run(b, fast=fast, seed=seed, verbose=verbose)
        name_a, name_b = a, b

    verdict, score, details, source = _equality(ra, rb)
    t = Table(title=f"compare: {name_a}  vs  {name_b}", show_lines=False)
    for c in ("property", name_a, name_b, "agree"):
        t.add_column(c, overflow="fold")
    for prop, va, vb, same in details:
        t.add_row(prop, va, vb, "[green]yes[/]" if same else "[red]no[/]")
    console.print(t)
    colour = "green" if verdict == "same" else "red" if verdict == "different" else "yellow"
    console.print(Panel(
        f"[bold {colour}]verdict: {verdict}[/]  (agreement {score * 100:.0f}%)\n"
        f"[dim]method: {source}[/]",
        title="equality test", expand=False))


def _quick_run(preset: str, *, fast: bool, seed: int, verbose: bool) -> RunResult:
    target = resolve_target(preset, None, None, None, None, None)
    adapter = build_adapter(target)
    err_console.print(f"[dim]probing {preset}...[/]")
    return asyncio.run(_run_and_close(
        adapter, config={"fast": fast, "run_id": f"cmp-{_slug(preset)}"},
        budget=build_budget(None, None, None, None, 0.0, 0.0), recorder=None,
        llm=None, seed=seed, progress=_progress_cb(verbose)))


def _equality(a: RunResult, b: RunResult) -> tuple[str, float, list, str]:
    """Behavioural-fingerprint distance between two completed runs; posterior fallback.

    ``forensics.fingerprint.equality_test`` needs two *live adapters*; here we only have two
    :class:`RunResult`s, so we compare their behavioural fingerprints directly with
    :func:`~forensics.fingerprint.fingerprint_distance` and turn the distance into a verdict.
    """
    try:
        import forensics.fingerprint as _fp  # type: ignore
        fa, fb = _fp.fingerprint(a), _fp.fingerprint(b)
        dist = float(_fp.fingerprint_distance(fa, fb))
        verdict = "same" if dist < 0.08 else "different" if dist > 0.15 else "uncertain"
        score = max(0.0, 1.0 - dist)
        details = _fingerprint_details(_fp.tag_set(fa), _fp.tag_set(fb))
        return verdict, score, details, "forensics.fingerprint (behavioural distance)"
    except Exception:  # noqa: BLE001
        return _fallback_equality(a, b)


def _fingerprint_details(tags_a: set, tags_b: set) -> list[tuple[str, str, str, bool]]:
    """Row-per-tag comparison over a curated set of interpretable fingerprint tags."""
    def _as_map(tags: set) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in tags:
            k, _, v = str(t).partition("=")
            out[k] = v
        return out

    ma, mb = _as_map(tags_a), _as_map(tags_b)
    curated = ["tokenizer.family", "identity.logit_dtype", "error.truncation_policy",
               "belief.arch.family", "belief.arch.sparsity", "belief.attn.long_range",
               "belief.attn.kv_sharing", "belief.pos.encoding", "belief.system.topology",
               "identity.family"]
    details: list[tuple[str, str, str, bool]] = []
    for key in curated:
        if key not in ma and key not in mb:
            continue
        va, vb = ma.get(key, "-"), mb.get(key, "-")
        details.append((key, va, vb, va == vb and va != "-"))
    return details


def _fallback_equality(a: RunResult, b: RunResult) -> tuple[str, float, list, str]:
    props = ["arch.family", "arch.sparsity", "attn.kv_sharing", "attn.long_range",
             "pos.encoding", "out.logit_precision", "system.topology"]
    details: list[tuple[str, str, str, bool]] = []
    agree = 0
    total = 0
    for p in props:
        ta, tb = _top(a.posterior, p), _top(b.posterior, p)
        if not ta or not tb:
            continue
        total += 1
        same = ta[0] == tb[0]
        agree += 1 if same else 0
        details.append((p, ta[0], tb[0], same))
    ea, eb = _est_map(a.estimates), _est_map(b.estimates)
    for q in ("dims.hidden_size", "dims.vocab_size", "context.max_input_tokens",
              "tokenizer.family"):
        if q not in ea or q not in eb:
            continue
        va, vb = ea[q].value, eb[q].value
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and max(va, vb):
            same = abs(va - vb) / max(abs(va), abs(vb), 1) < 0.1
        else:
            same = str(va) == str(vb)
        total += 1
        agree += 1 if same else 0
        details.append((q, _fmt_value(va), _fmt_value(vb), same))
    score = agree / total if total else 0.0
    verdict = "same" if score >= 0.75 else "different" if score < 0.4 else "uncertain"
    return verdict, score, details, "fallback (posterior + estimate agreement)"



@app.command()
def web(
    port: int = typer.Option(8765, "--port"),
    host: str = typer.Option("127.0.0.1", "--host", help="keep 127.0.0.1 unless you know why"),
    no_browser: bool = typer.Option(False, "--no-browser"),
) -> None:
    """Open the web UI: start scans, watch them live, view every report and diagram."""
    from forensics.web.app import serve
    serve(host=host, port=port, open_browser=not no_browser)

@app.command()
def probes(
    kind: Optional[str] = typer.Option(None, "--kind", help="filter: generative|decision|embedding"),
    category: Optional[str] = typer.Option(None, "--category"),
) -> None:
    """List the probe registry (id, title, category, kinds, informs, estimates, ...)."""
    from forensics.probes import load_all
    from forensics.probes.base import REGISTRY
    load_all()
    rows = sorted(REGISTRY.values(), key=lambda c: c.id)
    t = Table(title=f"probe registry ({len(rows)} probes)", show_lines=False)
    for c in ("id", "title", "category", "kinds", "informs", "estimates", "requires",
              "cost", "base"):
        t.add_column(c, overflow="fold")
    shown = 0
    for c in rows:
        if kind and kind not in c.kinds:
            continue
        if category and c.category != category:
            continue
        shown += 1
        t.add_row(
            c.id, c.title, c.category, ", ".join(c.kinds),
            ", ".join(sorted(c.informs)), ", ".join(c.estimates) or "-",
            ", ".join(c.requires) or "-", str(c.cost),
            "yes" if c.baseline else "")
    console.print(t)
    console.print(f"[dim]{shown} shown / {len(rows)} total[/]")


@app.command("presets")
def list_presets() -> None:
    """List the built-in target presets (env var NAMES only; never secrets)."""
    t = Table(title="presets", show_lines=False)
    for c in ("name", "provider", "model", "kind", "base_url_env", "api_key_env"):
        t.add_column(c, overflow="fold")
    for name, p in PRESETS.items():
        t.add_row(name, p["provider"], p.get("model", ""), p.get("kind", ""),
                  p.get("base_url_env", "") or ("(sim)" if p["provider"] == "sim" else ""),
                  p.get("api_key_env", "") or "")
    console.print(t)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
