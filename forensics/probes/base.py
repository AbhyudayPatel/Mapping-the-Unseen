"""Probe framework: the contract every forensic experiment implements.

A probe is a self-contained experiment:
  - declares which hypothesis properties it informs and how diagnostic it is (for
    expected-information-gain selection), which quantities it estimates, what interface
    capabilities it requires and roughly how many requests it costs;
  - runs requests ONLY through `ctx.gen` / `ctx.gen_many` / `ctx.decide` (budget-aware,
    recorded, timed by the adapter);
  - returns a ProbeResult with Evidence (likelihoods over hypothesis values), Estimates
    (quantities with CIs and an epistemic level), tables/series for the report, and notes.

Epistemic levels (always label honestly):
  observed    – read directly off the interface (e.g. error text says "context 128000")
  inferred    – follows from observations by a strong, testable argument (e.g. SVD rank gap)
  estimated   – statistical estimate with a model + CI (e.g. active params from throughput)
  speculative – plausible under stated assumptions only (e.g. layers from d_model regression)
"""
from __future__ import annotations

import abc
import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from forensics.core.types import (
    Capabilities, DecisionRequest, DecisionResponse, GenRequest, GenResponse,
)

Level = Literal["observed", "inferred", "estimated", "speculative"]
LEVELS: tuple[str, ...] = ("observed", "inferred", "estimated", "speculative")

# Identifiability scale (see inference/identifiability.py)
DIRECT, STATISTICAL, WEAK, ASSUMPTION, NON_IDENTIFIABLE = 1, 2, 3, 4, 5


class BudgetExhausted(Exception):
    """Raised by ctx.gen*/decide when the run's request/token/cost budget is spent."""


class EndpointUnreachable(BudgetExhausted):
    """Raised when the endpoint stops answering at the transport level (DNS, connection,
    TLS) several times in a row. Transport failures say nothing about the model, so the
    run stops instead of letting probes interpret them as API behaviour. Subclasses
    BudgetExhausted so every probe's existing partial-result handling applies."""


def is_transport_error(resp) -> bool:
    """True when the request never produced an HTTP response (no status code)."""
    return getattr(resp, "status", None) is None and bool(getattr(resp, "error", None))


#: consecutive transport failures (after the HTTP client's own retries) that abort a run
MAX_CONSECUTIVE_TRANSPORT_ERRORS = 5


@dataclass
class Evidence:
    """Likelihood evidence about ONE hypothesis property.

    likelihoods: P(observed outcome | property == value), for the values this evidence
    speaks to. Values not listed are treated as neutral (the mean of listed likelihoods).
    Do not return 0.0 — use small floors (e.g. 0.02) so one noisy probe can't zero a
    hypothesis. `strength` in (0, 1] tempers the update (likelihood ** strength); use < 1
    when the probe's statistical model is approximate or confounders are plausible.
    """
    property: str
    likelihoods: dict[str, float]
    summary: str
    level: Level = "inferred"
    strength: float = 1.0
    stats: dict[str, Any] = field(default_factory=dict)   # p_value, effect, ci, n, test...
    alternatives: list[str] = field(default_factory=list)  # alternative explanations
    probe_id: str = ""
    # Certificate metadata. Legacy probes may leave these unset; absence is recorded,
    # never treated as a calibrated likelihood model or an independence guarantee.
    falsifier: str = ""
    measurement_integrity: dict[str, Any] = field(default_factory=dict)
    experiment_group: str = ""
    factor_kind: Literal["heuristic", "calibrated_likelihood_ratio"] = "heuristic"


@dataclass
class Estimate:
    """A (possibly observed) quantity. `value` may be numeric or categorical (str)."""
    quantity: str                     # e.g. "dims.hidden_size", "context.max_input_tokens"
    value: Any
    ci: tuple[float, float] | None = None
    ci_level: float = 0.9
    level: Level = "estimated"
    method: str = ""
    assumptions: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    identifiability: int = STATISTICAL
    samples: list[float] | None = None   # optional draws for Monte-Carlo combination
    unit: str = ""
    probe_id: str = ""


@dataclass
class ProbeResult:
    probe_id: str
    status: Literal["ok", "partial", "skipped", "failed"] = "ok"
    evidence: list[Evidence] = field(default_factory=list)
    estimates: list[Estimate] = field(default_factory=list)
    # report material: tables = list of row dicts; series = {name: {"x": [...], "y": [...], ...}}
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    series: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)   # published into ctx.shared
    n_requests: int = 0
    error: str | None = None

    def add_evidence(self, ev: Evidence) -> None:
        ev.probe_id = ev.probe_id or self.probe_id
        self.evidence.append(ev)

    def add_estimate(self, est: Estimate) -> None:
        est.probe_id = est.probe_id or self.probe_id
        self.estimates.append(est)


class ProbeContext:
    """Everything a probe may touch. Created by the engine (or tests)."""

    def __init__(self, adapter, caps: Capabilities | None = None, budget=None, recorder=None,
                 belief=None, config: dict[str, Any] | None = None, llm=None,
                 seed: int = 0, logger: Callable[[str], None] | None = None,
                 concurrency: int = 4) -> None:
        self.adapter = adapter
        self.caps = caps or adapter.static_capabilities()
        self.budget = budget
        self.recorder = recorder
        self.belief = belief
        self.config = config or {}
        self.llm = llm
        self.rng = np.random.default_rng(seed)
        self.shared: dict[str, Any] = {}
        self._log = logger or (lambda m: None)
        self.concurrency = concurrency
        self.current_probe: str = ""
        self.n_requests = 0
        self.transport_errors = 0            # total
        self._consecutive_transport_errors = 0

    # ------------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        self._log(f"[{self.current_probe}] {msg}" if self.current_probe else msg)

    def fast(self) -> bool:
        """Probes should shrink sample sizes when config['fast'] is set (tests, quick scans)."""
        return bool(self.config.get("fast"))

    def scale(self, n: int, minimum: int = 1) -> int:
        """Scale a sample size by config['sample_scale'] (default 1.0; fast → 0.25)."""
        s = float(self.config.get("sample_scale", 0.25 if self.fast() else 1.0))
        return max(minimum, int(math.ceil(n * s)))

    def _check_budget(self, req: Any) -> None:
        if self._consecutive_transport_errors >= MAX_CONSECUTIVE_TRANSPORT_ERRORS:
            raise EndpointUnreachable(
                f"endpoint unreachable: {self._consecutive_transport_errors} consecutive "
                f"transport failures")
        if self.budget is not None and not self.budget.can_spend(req):
            raise BudgetExhausted(f"budget exhausted before request tag={getattr(req, 'tag', '')}")

    def _track_transport(self, resp: Any) -> None:
        if is_transport_error(resp):
            self.transport_errors += 1
            self._consecutive_transport_errors += 1
            if self._consecutive_transport_errors >= MAX_CONSECUTIVE_TRANSPORT_ERRORS:
                raise EndpointUnreachable(
                    f"endpoint unreachable: {self._consecutive_transport_errors} consecutive "
                    f"transport failures (last: {str(getattr(resp, 'error', ''))[:120]})")
        else:
            self._consecutive_transport_errors = 0

    async def gen(self, req: GenRequest) -> GenResponse:
        self._check_budget(req)
        if not req.tag:
            req.tag = self.current_probe
        resp = await self.adapter.generate(req)
        self.n_requests += 1
        if self.budget is not None:
            self.budget.charge(req, resp)
        if self.recorder is not None:
            self.recorder.record("generate", req, resp, probe=self.current_probe)
        self._track_transport(resp)
        return resp

    async def gen_many(self, reqs: list[GenRequest], concurrency: int | None = None) -> list[GenResponse]:
        """Run requests with bounded concurrency; preserves order. concurrency=1 → sequential
        (use for timing probes where parallel requests would contaminate latency)."""
        sem = asyncio.Semaphore(concurrency or self.concurrency)

        async def one(r: GenRequest) -> GenResponse:
            async with sem:
                return await self.gen(r)

        return list(await asyncio.gather(*(one(r) for r in reqs)))

    async def decide(self, req: DecisionRequest) -> DecisionResponse:
        self._check_budget(req)
        if not req.tag:
            req.tag = self.current_probe
        resp = await self.adapter.decide(req)
        self.n_requests += 1
        if self.budget is not None:
            self.budget.charge(req, resp)
        if self.recorder is not None:
            self.recorder.record("decide", req, resp, probe=self.current_probe)
        self._track_transport(resp)
        return resp

    async def decide_many(self, reqs: list[DecisionRequest], concurrency: int | None = None) -> list[DecisionResponse]:
        sem = asyncio.Semaphore(concurrency or self.concurrency)

        async def one(r: DecisionRequest) -> DecisionResponse:
            async with sem:
                return await self.decide(r)

        return list(await asyncio.gather(*(one(r) for r in reqs)))


class Probe(abc.ABC):
    id: str = ""                         # e.g. "P07.timing.prefill_scaling"
    title: str = ""
    category: str = ""                   # interface|tokenizer|logits|precision|context|...
    kinds: tuple[str, ...] = ("generative",)
    #: property -> diagnosticity in [0,1] (how sharply this probe separates the values)
    informs: dict[str, float] = {}
    #: quantity names this probe can produce
    estimates: tuple[str, ...] = ()
    #: Capabilities attributes that must be truthy
    requires: tuple[str, ...] = ()
    #: probe ids that should run first if available (their `shared` outputs are used)
    depends_on: tuple[str, ...] = ()
    #: rough number of requests at sample_scale=1
    cost: int = 10
    #: run in the baseline sweep regardless of EIG (cheap and broadly useful)
    baseline: bool = False

    def applicable(self, ctx: ProbeContext) -> tuple[bool, str]:
        if ctx.caps.kind not in self.kinds and ctx.caps.kind != "unknown":
            return False, f"probe needs kind in {self.kinds}, endpoint is {ctx.caps.kind}"
        for r in self.requires:
            if not getattr(ctx.caps, r, None):
                return False, f"requires capability '{r}'"
        return True, ""

    def estimated_cost(self, ctx: ProbeContext) -> int:
        return ctx.scale(self.cost)

    def outcome_model(self, ctx: ProbeContext) -> dict[str, dict[str, dict[str, float]]] | None:
        """Optional: {property: {value: {outcome: P(outcome|value)}}} for EIG. If None, the
        engine synthesises one from `informs` diagnosticities."""
        return None

    @abc.abstractmethod
    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ...

    def result(self) -> ProbeResult:
        return ProbeResult(probe_id=self.id)


# --------------------------------------------------------------------------- registry
REGISTRY: dict[str, type[Probe]] = {}


def register(cls: type[Probe]) -> type[Probe]:
    if not cls.id:
        raise ValueError(f"probe {cls.__name__} has no id")
    if cls.id in REGISTRY and REGISTRY[cls.id] is not cls:
        raise ValueError(f"duplicate probe id {cls.id}")
    REGISTRY[cls.id] = cls
    return cls


def likelihood_from_pvalue(p: float, favored: str, other: str, floor: float = 0.05) -> dict[str, float]:
    """Helper: turn a test's p-value into a two-value likelihood with a floor.
    Small p → evidence for `favored` (the alternative). Calibrated conservatively:
    the Bayes factor is bounded by the Sellke–Bayarri–Berger bound -e·p·ln(p) (p < 1/e)."""
    p = min(max(p, 1e-12), 1.0)
    if p < 1 / math.e:
        bf_null = max(-math.e * p * math.log(p), floor)   # minimum BF in favor of null
    else:
        bf_null = 1.0
    return {favored: 1.0, other: min(1.0, bf_null)}
