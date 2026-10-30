"""Result types shared by engine, estimator, fingerprint, report and CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forensics.core.types import Capabilities
from forensics.probes.base import Estimate, Evidence, ProbeResult


@dataclass
class ExperimentStep:
    step: int
    probe_id: str
    reason: str                       # "baseline" | "max EIG/cost" | "user-requested" | ...
    eig_bits: float | None = None     # expected information gain at selection time
    cost: int | None = None
    entropy_before_bits: float | None = None
    entropy_after_bits: float | None = None
    status: str = ""
    n_requests: int = 0


@dataclass
class IdentifiabilityEntry:
    key: str                          # property or quantity name
    level: int                        # 1 direct … 5 non-identifiable
    label: str                        # "directly observable" …
    reason: str
    equivalence: str = ""             # why distinct architectures could produce the same behaviour


@dataclass
class FamilyMatch:
    name: str                         # reference model / family name from the arch DB
    score: float                      # 0..1 consistency with measured constraints
    matched: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    what: str                         # probe id or free-text experiment description
    why: str
    expected_gain_bits: float | None = None
    needs: str = ""                   # e.g. "logprobs", "more budget", "a second API key"


@dataclass
class RunResult:
    run_id: str
    target: dict[str, Any]
    caps: Capabilities
    started_at: str = ""
    finished_at: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    probe_results: list[ProbeResult] = field(default_factory=list)
    prior: dict[str, dict[str, float]] = field(default_factory=dict)
    posterior: dict[str, dict[str, float]] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    estimates: list[Estimate] = field(default_factory=list)          # probe + estimator outputs
    identifiability: list[IdentifiabilityEntry] = field(default_factory=list)
    family_matches: list[FamilyMatch] = field(default_factory=list)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    experiment_log: list[ExperimentStep] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    budget_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    llm_assisted: list[str] = field(default_factory=list)            # which parts used the LLM advisor
    run_dir: str = ""
