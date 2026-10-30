"""The forensic inference loop: discover -> baseline -> adaptive EIG selection.

:func:`run_forensics` drives an :class:`~forensics.core.adapters.base.Adapter`
through the probe registry, maintaining a :class:`~forensics.inference.belief.Belief`
that every probe's :class:`~forensics.probes.base.Evidence` updates.  Probes are
chosen greedily by expected information gain per estimated cost
(:mod:`forensics.inference.select`); baseline probes run first, then the loop keeps
picking the most cost-effective informative probe until the budget is spent, the
probes run out, or no probe is expected to gain more than ``min_gain_bits``.

The run is fully **resumable**: with ``config['replay_dir']`` set, a
:class:`~forensics.core.adapters.replay.ReplayAdapter` serves recorded responses so
the whole analysis re-runs offline with no network and no cost.

The dimension estimator (agent B) and identifiability analyzer (agent C) are called
by name and imported lazily, so the loop still completes (with a note) if they land
later.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from forensics.core.discovery import discover
from forensics.core.types import Capabilities
from forensics.inference.belief import Belief, apply_coupling
from forensics.inference.hypotheses import DYNAMIC_PROPERTIES, PROPERTIES
from forensics.inference.run_types import (
    ExperimentStep, FamilyMatch, IdentifiabilityEntry, Recommendation, RunResult,
)
from forensics.inference.select import (
    DEFAULT_MIN_GAIN_BITS, expected_information_gain, rank_probes,
)
from forensics.probes import load_all
from forensics.probes.base import (
    EndpointUnreachable, MAX_CONSECUTIVE_TRANSPORT_ERRORS,
    BudgetExhausted, Estimate, Evidence, Probe, ProbeContext, ProbeResult,
)

_KNOWN_PROPERTIES = set(PROPERTIES) | set(DYNAMIC_PROPERTIES)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quant_value(estimates: list[Estimate], quantity: str) -> Any:
    """Most recent value estimated for ``quantity`` (None if not estimated)."""
    for est in reversed(estimates):
        if est.quantity == quantity and est.value is not None:
            return est.value
    return None


class _Engine:
    def __init__(self, adapter, config: dict[str, Any], budget, recorder, llm,
                 seed: int, progress: Callable[[str], None] | None):
        self.config = dict(config or {})
        self.budget = budget
        self.recorder = recorder
        self.llm = llm
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self._progress = progress
        self.notes: list[str] = []
        self.llm_assisted: list[str] = []
        # persistent cross-probe scratchpad (so e.g. P03 sees P02's tokenizer)
        self.shared: dict[str, Any] = {}
        self.belief: Belief | None = None
        self._ctx_counter = 0

        # resumability: swap in the replay adapter when a recording dir is given
        replay_dir = self.config.get("replay_dir")
        if replay_dir:
            from forensics.core.adapters.replay import ReplayAdapter
            self.adapter = ReplayAdapter(replay_dir)
            self.notes.append(f"replay mode: analysing recording at {replay_dir}")
        else:
            self.adapter = adapter

        self.caps: Capabilities = self.adapter.static_capabilities()
        self.min_gain = float(self.config.get("min_gain_bits", DEFAULT_MIN_GAIN_BITS))
        # selection policy: "eig" (default adaptive), "static" (cheapest-first fixed
        # order), "uniform" (seeded random) -- the latter two are evaluation baselines
        self.planner = str(self.config.get("planner", "eig"))
        if self.planner not in {"eig", "static", "uniform"}:
            raise ValueError("planner must be 'eig', 'static', or 'uniform'")
        import random as _random
        self._planner_rng = _random.Random(self.seed ^ 0x9E3779B9)

    # ------------------------------------------------------------------ helpers
    def progress(self, msg: str) -> None:
        if self._progress is not None:
            try:
                self._progress(msg)
            except Exception:  # noqa: BLE001
                pass

    def _new_ctx(self) -> ProbeContext:
        self._ctx_counter += 1
        ctx = ProbeContext(
            self.adapter, caps=self.caps, budget=self.budget, recorder=self.recorder,
            belief=self.belief, config=self.config, llm=self.llm,
            seed=self.seed + self._ctx_counter,
            concurrency=int(self.config.get("concurrency", 4)),
        )
        ctx.shared = self.shared          # persist scratchpad across probes
        return ctx

    _CORE_PROPS = ("arch.family", "arch.sparsity", "attn.long_range")

    def _coverage_challenger(self, remaining, belief):
        """Cheapest unrated probe with high diagnosticity for a core property whose
        leading value is largely prior/coupling-driven (few supporting measurements).
        Returns None when every core property is measurement-backed or no probe remains."""
        candidates = []
        for p in remaining:
            informs = getattr(p, "informs", {}) or {}
            for prop in self._CORE_PROPS:
                d = informs.get(prop)
                if not d or d < 0.6:
                    continue
                dist = belief.posterior.get(prop) or {}
                if not dist:
                    continue
                n_ev = len(belief.ledger.get(prop, []))
                top_p = max(dist.values())
                # lead is confident on thin measurement: prior did the work
                if top_p >= 0.75 and n_ev <= 2:
                    candidates.append((p.estimated_cost(self._new_ctx()), p))
                    break
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1] if candidates else None

    def _budget_ok(self) -> bool:
        if getattr(self, "unreachable", False):
            return False
        return self.budget is None or self.budget.can_spend()

    # ------------------------------------------------------------------ discovery
    async def discover_caps(self) -> None:
        self.progress("discovering interface capabilities")
        fast = bool(self.config.get("fast"))
        try:
            self.caps = await discover(self.adapter, budget=self.budget,
                                       recorder=self.recorder, fast=fast)
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"discovery failed ({e!r}); using static capabilities")
            self.caps = self.adapter.static_capabilities()

    # ------------------------------------------------------------------ probe run
    async def run_probe(self, probe: Probe) -> tuple[ProbeResult, bool]:
        """Run one probe; return (result, budget_exhausted_or_unreachable)."""
        ctx = self._new_ctx()
        ctx.current_probe = probe.id
        # transport-failure streaks span probes: an outage mid-probe must stop the run
        ctx._consecutive_transport_errors = getattr(self, "_consec_transport", 0)

        def _sync() -> None:
            self._consec_transport = ctx._consecutive_transport_errors
            self.transport_errors = getattr(self, "transport_errors", 0) + ctx.transport_errors

        try:
            res = await probe.run(ctx)
            if res is None:
                res = ProbeResult(probe_id=probe.id, status="failed",
                                  error="probe returned None")
        except EndpointUnreachable as e:
            _sync()
            self._mark_unreachable(str(e))
            return (ProbeResult(probe_id=probe.id, status="failed", error=str(e),
                                n_requests=ctx.n_requests), True)
        except BudgetExhausted as e:
            _sync()
            return (ProbeResult(probe_id=probe.id, status="partial",
                                error=str(e), n_requests=ctx.n_requests), True)
        except Exception as e:  # noqa: BLE001
            _sync()
            return (ProbeResult(probe_id=probe.id, status="failed",
                                error=repr(e), n_requests=ctx.n_requests), False)
        _sync()
        if not res.n_requests:
            res.n_requests = ctx.n_requests
        # A probe that caught its own transport errors may still have "concluded" something
        # from them. If most of its requests never reached the server, its evidence is about
        # the network, not the model: drop it.
        if ctx.n_requests and ctx.transport_errors / ctx.n_requests > 0.3:
            res.notes.append(
                f"discarded {len(res.evidence)} evidence / {len(res.estimates)} estimates: "
                f"{ctx.transport_errors}/{ctx.n_requests} requests failed at the transport level")
            res.evidence, res.estimates, res.shared = [], [], {}
            res.status = "failed"
        if self._consec_transport >= MAX_CONSECUTIVE_TRANSPORT_ERRORS:
            self._mark_unreachable("endpoint stopped responding (transport errors)")
            return res, True
        # publish anything the probe shared for downstream probes
        if res.shared:
            self.shared.update(res.shared)
        return res, False

    def _mark_unreachable(self, why: str) -> None:
        if not getattr(self, "unreachable", False):
            self.unreachable = True
            self.notes.insert(0, f"ABORTED: {why}. Results cover only the probes that ran "
                                 f"before the outage; re-run when the endpoint is reachable.")

    def integrate(self, belief: Belief, res: ProbeResult) -> None:
        """Fold a probe's evidence into the belief (rejecting unknown properties)."""
        applied: list[Evidence] = []
        for ev in res.evidence:
            if ev.property in _KNOWN_PROPERTIES:
                applied.append(ev)
            else:
                self.notes.append(
                    f"{res.probe_id}: rejected evidence for unknown property "
                    f"{ev.property!r}")
        if applied:
            belief.apply_all(applied)
        # NOTE: soft coupling is applied ONCE after the adaptive loop (below), not per
        # probe.  Per-round application compounds the nudge multiplicatively (~x1.1 per
        # probe) and silently manufactures confidence no measurement supplied.

    # ------------------------------------------------------------------ estimator (B)
    def run_estimator(self, belief: Belief, estimates: list[Estimate]
                      ) -> tuple[list[Estimate], list[FamilyMatch]]:
        extra_est: list[Estimate] = []
        families: list[FamilyMatch] = []
        try:
            from forensics.inference import estimator as _est  # type: ignore
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"dimension estimator unavailable ({e!r}); "
                              "quantities limited to probe outputs")
            return extra_est, families

        fn = getattr(_est, "estimate_dimensions", None)
        if fn is None:
            self.notes.append("estimator.estimate_dimensions missing")
        else:
            try:
                out = fn(belief, estimates, self.caps, self.shared)
                e2, f2 = _split_estimator_output(out)
                if isinstance(out, dict):
                    for n in out.get("notes") or []:
                        s = str(n)
                        surface = ("family matching" in s or "hidden_size" in s
                                   or "d_model" in s)
                        if surface and s not in self.notes:
                            self.notes.append(s)
                extra_est.extend(e2)
                families.extend(f2)
                if self.llm:
                    self.llm_assisted.append("estimator")
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"dimension estimator errored: {e!r}")

        if not families:
            for name in ("match_families", "family_matches", "match"):
                mf = getattr(_est, name, None)
                if callable(mf):
                    try:
                        families = _as_family_matches(
                            mf(belief, estimates + extra_est, self.caps, self.shared))
                        break
                    except Exception:  # noqa: BLE001
                        continue
        return extra_est, families

    def fallback_family_matches(self, belief: Belief, estimates: list[Estimate]
                                ) -> list[FamilyMatch]:
        """Best-effort family match straight from the reference DB (no agent B).

        Same rules as the estimator: belief categories only when evidence-backed (not the
        prior), at least one MEASURED quantity, coverage-aware scores."""
        try:
            from forensics.inference.family_match import INSUFFICIENT_NOTE, match_families
            out, notes = match_families(belief, estimates, top_k=5)
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"fallback family matching failed ({e!r})")
            return []
        for n in notes:
            if n.startswith(INSUFFICIENT_NOTE):
                if not any(x.startswith(INSUFFICIENT_NOTE) for x in self.notes):
                    self.notes.append(n)
        if out:
            self.notes.append("family matches from reference-DB nearest-neighbour "
                              "(no dimension estimator)")
        return out

    # ------------------------------------------------------------------ identifiability (C)
    def run_identifiability(self, belief: Belief, estimates: list[Estimate],
                            ran: list[str]) -> list[IdentifiabilityEntry]:
        try:
            from forensics.inference import identifiability as _idf  # type: ignore
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"identifiability analyzer unavailable ({e!r})")
            return []
        fn = getattr(_idf, "analyze", None)
        if fn is None:
            self.notes.append("identifiability.analyze missing")
            return []
        try:
            out = fn(belief, estimates, self.caps, ran)
            return list(out) if out else []
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"identifiability analyzer errored: {e!r}")
            return []

    # ------------------------------------------------------------------ recommendations
    def recommendations(self, belief: Belief, all_probes: list[Probe],
                        ran: set[str]) -> list[Recommendation]:
        sel_ctx = self._new_ctx()
        recs: list[Recommendation] = []

        # 1) most informative probes not yet run (or worth re-running)
        unrun = [p for p in all_probes if p.id not in ran]
        for r in rank_probes(unrun, belief, sel_ctx)[:5]:
            if r.eig_bits <= self.min_gain and self.planner == "eig":
                continue
            recs.append(Recommendation(
                what=r.probe.id, why=f"unrun; expected to cut uncertainty on "
                f"{', '.join(sorted(r.probe.informs))}",
                expected_gain_bits=round(r.eig_bits, 4),
                needs="more budget" if not self._budget_ok() else ""))

        # 2) capability-gated probes: what a richer interface would unlock
        for probe in all_probes:
            if probe.id in ran:
                continue
            ok, reason = probe.applicable(sel_ctx)
            if ok or "requires capability" not in reason:
                continue
            cap = reason.split("'")[1] if "'" in reason else reason
            recs.append(Recommendation(
                what=probe.id, why=f"blocked: {reason}", needs=cap))
        return recs

    # ------------------------------------------------------------------ main
    async def run(self) -> RunResult:
        started = _now()
        t0 = time.time()

        await self.discover_caps()
        kind = self.caps.kind if self.caps.kind != "unknown" else \
            getattr(self.adapter, "kind", "unknown")
        belief = Belief(PROPERTIES, kind=kind)
        self.belief = belief
        self.progress(f"kind={kind}; tracking {len(belief.posterior)} properties")

        load_all()
        sel_ctx = self._new_ctx()
        all_probes: list[Probe] = []
        for cls in list(_registry().values()):
            probe = cls()
            ok, _reason = probe.applicable(sel_ctx)
            if ok:
                all_probes.append(probe)
        self.progress(f"{len(all_probes)} applicable probes for kind={kind}")

        probe_results: list[ProbeResult] = []
        all_evidence: list[Evidence] = []
        all_estimates: list[Estimate] = []
        experiment_log: list[ExperimentStep] = []
        ran: set[str] = set()
        step = 0
        stop = False

        def _record(probe: Probe, res: ProbeResult, reason: str, eig: float,
                    h_before: float) -> None:
            nonlocal step
            self.integrate(belief, res)
            all_evidence.extend(res.evidence)
            all_estimates.extend(res.estimates)
            probe_results.append(res)
            # a probe that used the optional LLM advisor flags it -> mark the run
            if res.shared.get("llm_assisted") and probe.id not in self.llm_assisted:
                self.llm_assisted.append(probe.id)
            ran.add(probe.id)
            step += 1
            experiment_log.append(ExperimentStep(
                step=step, probe_id=probe.id, reason=reason,
                eig_bits=round(eig, 4), cost=probe.estimated_cost(sel_ctx),
                entropy_before_bits=round(h_before, 4),
                entropy_after_bits=round(belief.total_entropy(), 4),
                status=res.status, n_requests=res.n_requests))
            # optional evaluation hook (planner comparisons etc.): snapshot the belief
            hook = self.config.get("_on_step")
            if hook is not None:
                hook(step=step, belief=belief, probe_id=probe.id,
                     n_requests=int(getattr(self.budget, "requests", 0) or res.n_requests),
                     probe_requests=res.n_requests)

        # ---- (d) baseline sweep ------------------------------------------------
        baselines = sorted([p for p in all_probes if p.baseline],
                           key=lambda p: p.estimated_cost(sel_ctx))
        for probe in baselines:
            if not self._budget_ok():
                self.notes.append("budget exhausted before baseline sweep completed")
                stop = True
                break
            eig = expected_information_gain(probe, belief, sel_ctx)
            h_before = belief.total_entropy()
            self.progress(f"baseline {probe.id}")
            res, exhausted = await self.run_probe(probe)
            _record(probe, res, "baseline", eig, h_before)
            if exhausted:
                self.notes.append(f"budget exhausted during baseline {probe.id}")
                stop = True
                break

        # ---- (e) adaptive EIG loop (single bounded coupling pass after the loop) ----
        max_probes = int(self.config.get("max_probes", 10_000))
        while not stop and len(ran) < max_probes:
            if not self._budget_ok():
                self.notes.append("budget exhausted; stopping adaptive loop")
                break
            remaining = [p for p in all_probes if p.id not in ran]
            if not remaining:
                break
            if self.planner == "eig":
                ranked = rank_probes(remaining, belief, sel_ctx)
                informative = [r for r in ranked if r.eig_bits > self.min_gain]
                if not informative:
                    # Core-coverage floor: a confident front-runner built mostly on prior
                    # mass must not starve the one probe that could defeat it.  Before
                    # stopping on diminishing returns, run any unrated probe that is
                    # highly diagnostic for a core architecture property whose current
                    # lead is prior- or coupling-driven rather than measurement-driven.
                    challenger = self._coverage_challenger(remaining, belief)
                    if challenger is not None and self._budget_ok():
                        probe = challenger
                        eig_sel = expected_information_gain(probe, belief, sel_ctx)
                        reason = "core coverage (challenger)"
                        self.progress(f"select {probe.id} (core coverage)")
                    else:
                        self.notes.append(
                            f"diminishing returns: no probe exceeds {self.min_gain} bits EIG")
                        break
                else:
                    best = informative[0]
                    probe = best.probe
                    reason, eig_sel = "max EIG/cost", best.eig_bits
                    self.progress(f"select {probe.id} (EIG={best.eig_bits:.3f} bits, "
                                  f"score={best.score:.4f})")
            elif self.planner == "static":
                probe = min(remaining, key=lambda p: p.estimated_cost(sel_ctx))
                eig_sel = expected_information_gain(probe, belief, sel_ctx)
                reason = "static order"
                self.progress(f"select {probe.id} (static)")
            else:  # uniform random baseline
                probe = self._planner_rng.choice(list(remaining))
                eig_sel = expected_information_gain(probe, belief, sel_ctx)
                reason = "uniform random"
                self.progress(f"select {probe.id} (uniform)")
            h_before = belief.total_entropy()
            res, exhausted = await self.run_probe(probe)
            _record(probe, res, reason, eig_sel, h_before)
            if exhausted:
                self.notes.append(f"budget exhausted during {probe.id}")
                break

        # ---- (e2) measurement / completeness pass ------------------------------
        # Estimate-producing probes (P03 hidden-size, P05 context, P07 timing, ...) carry
        # almost no belief-EIG — their value is the quantity they MEASURE, which feeds the
        # architecture table and diagram. The EIG loop above stops on diminishing belief
        # returns, so those probes would never run. Run every remaining applicable probe
        # that yields an estimate (and, under config['thorough_all'], all remaining probes),
        # ordered estimate-first then cheapest-first, until budget/max_probes is reached.
        if self.config.get("thorough", True):
            thorough_all = bool(self.config.get("thorough_all", False))
            remaining = [p for p in all_probes if p.id not in ran
                         and (thorough_all or p.estimates)]
            remaining.sort(key=lambda p: (not bool(p.estimates), p.estimated_cost(sel_ctx)))
            for probe in remaining:
                if len(ran) >= max_probes:
                    break
                if not self._budget_ok():
                    self.notes.append("budget exhausted; stopping measurement pass")
                    break
                h_before = belief.total_entropy()
                eig = expected_information_gain(probe, belief, sel_ctx)
                self.progress(f"measure {probe.id} (estimates: {', '.join(probe.estimates) or 'n/a'})")
                res, exhausted = await self.run_probe(probe)
                _record(probe, res, "measurement", eig, h_before)
                if exhausted:
                    self.notes.append(f"budget exhausted during measurement {probe.id}")
                    break

        # ---- (f2) single bounded soft-coupling pass over the assembled evidence ------
        apply_coupling(belief, float(self.config.get("coupling_strength", 0.12)))

        # ---- (g) dimension estimator + family matcher --------------------------
        self.progress("estimating quantities and matching families")
        extra_est, families = self.run_estimator(belief, all_estimates)
        # the estimator may echo the inputs back (same objects); append only new ones
        seen_ids = {id(e) for e in all_estimates}
        for e in extra_est:
            if id(e) not in seen_ids:
                all_estimates.append(e)
                seen_ids.add(id(e))
        if not families:
            families = self.fallback_family_matches(belief, all_estimates)

        # ---- (h) identifiability ----------------------------------------------
        identifiability = self.run_identifiability(belief, all_estimates, sorted(ran))

        # ---- (i) recommendations ----------------------------------------------
        recs = self.recommendations(belief, all_probes, ran)

        # ---- (j) assemble ------------------------------------------------------
        finished = _now()
        run_id = str(self.config.get("run_id") or f"run-{int(t0)}")
        result = RunResult(
            run_id=run_id,
            target=self.adapter.describe(),
            caps=self.caps,
            started_at=started,
            finished_at=finished,
            config=self.config,
            probe_results=probe_results,
            prior=belief.prior_dict(),
            posterior=belief.as_dict(),
            evidence=all_evidence,
            estimates=all_estimates,
            identifiability=identifiability,
            family_matches=families,
            experiment_log=experiment_log,
            recommendations=recs,
            budget_summary={**(self.budget.summary() if self.budget else {}),
                            "transport_errors": getattr(self, "transport_errors", 0),
                            "aborted_unreachable": bool(getattr(self, "unreachable", False))},
            notes=self.notes + belief.coupling_log[-3:],
            llm_assisted=self.llm_assisted,
            run_dir=str(getattr(self.recorder, "run_dir", "") or ""),
        )
        self.progress(f"done in {time.time() - t0:.2f}s; "
                      f"{len(ran)} probes, total entropy {belief.total_entropy():.2f} bits")
        return result


def _registry():
    from forensics.probes.base import REGISTRY
    return REGISTRY


def _as_family_matches(out: Any) -> list[FamilyMatch]:
    if not out:
        return []
    res: list[FamilyMatch] = []
    for item in out:
        if isinstance(item, FamilyMatch):
            res.append(item)
        elif isinstance(item, dict):
            res.append(FamilyMatch(
                name=str(item.get("name", "?")), score=float(item.get("score", 0.0)),
                matched=list(item.get("matched", [])),
                conflicts=list(item.get("conflicts", []))))
    return res


def _split_estimator_output(out: Any) -> tuple[list[Estimate], list[FamilyMatch]]:
    """Accept the several shapes agent B's estimator might return."""
    estimates: list[Estimate] = []
    families: list[FamilyMatch] = []
    if out is None:
        return estimates, families
    if isinstance(out, dict):
        estimates = [e for e in out.get("estimates", []) if isinstance(e, Estimate)]
        families = _as_family_matches(out.get("family_matches"))
        return estimates, families
    if hasattr(out, "estimates") or hasattr(out, "family_matches"):
        estimates = [e for e in getattr(out, "estimates", []) if isinstance(e, Estimate)]
        families = _as_family_matches(getattr(out, "family_matches", None))
        return estimates, families
    if isinstance(out, tuple) and len(out) == 2:
        a, b = out
        estimates = [e for e in a if isinstance(e, Estimate)] if a else []
        families = _as_family_matches(b)
        return estimates, families
    if isinstance(out, list):
        estimates = [e for e in out if isinstance(e, Estimate)]
        families = _as_family_matches([o for o in out if not isinstance(o, Estimate)])
        return estimates, families
    return estimates, families


async def run_forensics(adapter, config: dict[str, Any] | None = None, budget=None,
                        recorder=None, llm=None, seed: int = 0,
                        progress: Callable[[str], None] | None = None) -> RunResult:
    """Run the full adaptive forensic loop against ``adapter`` and return a RunResult.

    Parameters mirror the phase-1 contracts: ``budget`` (a
    :class:`~forensics.core.budget.Budget`) gates requests, ``recorder`` captures
    them for replay, ``llm`` is an optional advisor passed to probes/estimator,
    ``seed`` makes probe sampling deterministic, and ``progress`` is an optional
    ``(msg)`` status callback.  Set ``config['replay_dir']`` to re-analyse a
    recorded run offline.
    """
    engine = _Engine(adapter, config or {}, budget, recorder, llm, seed, progress)
    return await engine.run()
