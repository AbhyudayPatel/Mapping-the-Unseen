"""Reference architecture database: loader, indexer and empirical-prior engine.

This module loads the *frozen* data file ``architectures.json`` (198 records, 83
families) and turns it into

* a queryable index (:meth:`ArchDB.all`, :meth:`ArchDB.by_family`,
  :meth:`ArchDB.get`, :meth:`ArchDB.filter`),
* a nearest-neighbour matcher over a *measured* constraint set
  (:meth:`ArchDB.nearest`), and
* the empirical scaling laws (:meth:`ArchDB.regressions`) that the dimension
  estimator uses as a prior to go from a measured ``d_model``/``vocab`` to
  layers/heads/params.

The data file is a frozen contract and is never modified here: we only load and
index it.  All regularities are computed with numpy over the DB and reproduce
the reference numbers documented in ``research/arch_db.md`` §2.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------- paths
_HERE = Path(__file__).resolve().parent
DEFAULT_PATH = _HERE / "architectures.json"

# Map the *measured* constraint names used by probes/estimator to DB field names.
# (Unknown keys fall through to the raw DB field name, so raw fields also work.)
MEASURED_TO_DB: dict[str, str] = {
    "hidden_size": "d_model",
    "d_model": "d_model",
    "vocab_size": "vocab_size",
    "arch_type": "arch_type",
    "context": "max_position_embeddings",
    "max_position_embeddings": "max_position_embeddings",
    "n_layers": "n_layers",
    "n_heads": "n_heads",
    "n_kv_heads": "n_kv_heads",
    "head_dim": "head_dim",
    "ffn_dim": "d_ffn",
    "d_ffn": "d_ffn",
    "family": "family",
}

# Fields treated as positive numeric quantities in :meth:`ArchDB.nearest`.
_NUMERIC_FIELDS = {"hidden_size", "d_model", "vocab_size", "context",
                   "max_position_embeddings", "n_layers", "n_heads",
                   "n_kv_heads", "head_dim", "ffn_dim", "d_ffn"}

# Default per-field weights for the nearest-neighbour distance.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "hidden_size": 1.0, "d_model": 1.0,
    "vocab_size": 1.2, "context": 0.5,
    "arch_type": 1.0, "kv_sharing": 0.8, "moe": 0.8,
    "n_layers": 0.6, "n_heads": 0.6, "n_kv_heads": 0.5,
    "head_dim": 0.5, "ffn_dim": 0.5, "d_ffn": 0.5,
}

# match / conflict thresholds for numeric fields (relative difference)
_MATCH_REL = 0.01      # within 1 %  -> "matched"
_CONFLICT_REL = 0.05   # beyond 5 %  -> "conflict" (architectural constants are discrete)
_CAT_PENALTY = 1.5     # distance added for a categorical mismatch


def _norm_kv(v: Any) -> str | None:
    """Normalise a KV-sharing / attention-variant label to a coarse class."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("mha", "multi_head", "multihead"):
        return "mha"
    if s in ("gqa", "mqa", "gqa_or_mqa", "grouped", "multi_query"):
        return "gqa_or_mqa"
    if s in ("mla", "latent"):
        return "mla"
    return s


def _to_year(release: Any) -> int | None:
    """Parse a release string like ``'2024-11'`` into an int year."""
    if not release:
        return None
    try:
        return int(str(release).split("-", 1)[0])
    except (ValueError, TypeError):
        return None


class Encoder:  # pragma: no cover - trivial container, kept for symmetry
    """Placeholder so archdb has no hard dependency on the tokenizer module."""


class ArchDB:
    """In-memory index over the reference architecture database.

    Construct via :meth:`load` (cached) rather than directly, so the (fairly
    large) JSON file is parsed only once per process.
    """

    _cache: dict[str, "ArchDB"] = {}

    def __init__(self, entries: list[dict[str, Any]]):
        self._entries: list[dict[str, Any]] = list(entries)
        # name -> entry (case-insensitive), first-wins
        self._by_name: dict[str, dict[str, Any]] = {}
        for e in self._entries:
            key = str(e.get("name", "")).lower()
            self._by_name.setdefault(key, e)
        self._regressions_cache: dict[str, dict[str, Any]] | None = None
        self._unique_cache: list[dict[str, Any]] | None = None
        # robust log-scales for numeric fields (for scaled distances), lazy
        self._log_scale: dict[str, float] = {}

    # --------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "ArchDB":
        """Load (and cache) the architecture DB.  Repeated calls are free."""
        p = Path(path) if path else DEFAULT_PATH
        key = str(p.resolve())
        if key not in cls._cache:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(f"expected a list of records in {p}, got {type(data)}")
            cls._cache[key] = cls(data)
        return cls._cache[key]

    # --------------------------------------------------------------- access
    def all(self) -> list[dict[str, Any]]:
        """Every record (a shallow copy of the list; entries are shared)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def families(self) -> list[str]:
        return sorted({str(e.get("family")) for e in self._entries if e.get("family")})

    def by_family(self, family: str | None = None):
        """Records for ``family``; with no argument, a ``{family: [records]}`` dict."""
        if family is None:
            out: dict[str, list[dict[str, Any]]] = {}
            for e in self._entries:
                out.setdefault(str(e.get("family")), []).append(e)
            return out
        fam = str(family).lower()
        return [e for e in self._entries if str(e.get("family", "")).lower() == fam]

    def get(self, name: str) -> dict[str, Any] | None:
        """Exact record by ``name`` (case-insensitive), or ``None``."""
        return self._by_name.get(str(name).lower())

    def filter(self, **constraints: Any) -> list[dict[str, Any]]:
        """Filter records by exact values or ``(lo, hi)`` inclusive ranges.

        Keys may be measured names (``hidden_size``) or raw DB fields
        (``d_model``).  A tuple/list of length 2 is treated as an inclusive
        range; either bound may be ``None`` for an open end.  A scalar requires
        an exact (case-insensitive for strings) match.  Records missing the
        field are excluded.
        """
        out = self._entries
        for key, want in constraints.items():
            field = MEASURED_TO_DB.get(key, key)
            out = [e for e in out if _matches(e.get(field), want)]
        return list(out)

    # --------------------------------------------------------------- nearest
    def _field_log_scale(self, field: str) -> float:
        """Robust log-scale (MAD of ln values) for a numeric DB field."""
        if field not in self._log_scale:
            vals = [e.get(field) for e in self._entries]
            logs = np.array([math.log(v) for v in vals
                             if isinstance(v, (int, float)) and v and v > 0])
            if logs.size >= 2:
                mad = float(np.median(np.abs(logs - np.median(logs))))
                self._log_scale[field] = mad if mad > 1e-6 else float(np.std(logs)) or 1.0
            else:
                self._log_scale[field] = 1.0
        return self._log_scale[field]

    def nearest(self, measured: dict[str, Any],
                weights: dict[str, float] | None = None,
                top_k: int | None = None,
                ) -> list[tuple[dict[str, Any], float, list[str], list[str]]]:
        """Rank DB records by consistency with a *measured* constraint set.

        ``measured`` may contain ``hidden_size``, ``vocab_size``, ``context``,
        ``arch_type``, ``kv_sharing`` and ``moe`` (plus any other aliased
        field).  Fields whose value is ``None`` are **ignored** entirely.

        Numeric fields use a robust, scale-normalised log-distance
        (``|ln m - ln v| / MAD``); categorical fields (``arch_type``,
        ``kv_sharing``, ``moe``) contribute a fixed penalty on mismatch.

        Returns a list of ``(entry, distance, matched_fields, conflicts)``
        sorted by ascending distance (best first).
        """
        w = dict(_DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        # keep only usable measured fields
        active = {k: v for k, v in measured.items() if v is not None}

        ranked: list[tuple[dict[str, Any], float, list[str], list[str]]] = []
        for e in self._entries:
            dist = 0.0
            matched: list[str] = []
            conflicts: list[str] = []
            for key, mval in active.items():
                wf = float(w.get(key, 0.5))
                if key == "moe":
                    same = bool(e.get("moe")) == bool(mval)
                    if same:
                        matched.append(key)
                    else:
                        conflicts.append(key)
                        dist += wf * _CAT_PENALTY
                elif key == "kv_sharing":
                    ev = _norm_kv(e.get("attention_variant"))
                    mv = _norm_kv(mval)
                    if ev is None:
                        continue
                    if ev == mv:
                        matched.append(key)
                    else:
                        conflicts.append(key)
                        dist += wf * _CAT_PENALTY
                elif key in _NUMERIC_FIELDS:
                    field = MEASURED_TO_DB.get(key, key)
                    ev = e.get(field)
                    if not isinstance(ev, (int, float)) or ev is None or ev <= 0 \
                            or not isinstance(mval, (int, float)) or mval <= 0:
                        continue
                    ld = abs(math.log(float(mval)) - math.log(float(ev)))
                    rel = abs(float(mval) - float(ev)) / max(float(ev), 1e-9)
                    dist += wf * ld / self._field_log_scale(field)
                    if rel <= _MATCH_REL:
                        matched.append(key)
                    elif rel >= _CONFLICT_REL:
                        conflicts.append(key)
                else:  # generic categorical (e.g. arch_type, family)
                    field = MEASURED_TO_DB.get(key, key)
                    ev = e.get(field)
                    if ev is None:
                        continue
                    if str(ev).lower() == str(mval).lower():
                        matched.append(key)
                    else:
                        conflicts.append(key)
                        dist += wf * _CAT_PENALTY
            ranked.append((e, dist, matched, conflicts))

        ranked.sort(key=lambda t: (t[1], -len(t[2])))
        return ranked[:top_k] if top_k else ranked

    # --------------------------------------------------------------- regressions
    def unique(self) -> list[dict[str, Any]]:
        """Records deduplicated by structural shape (see research/arch_db.md §5)."""
        if self._unique_cache is None:
            seen: set = set()
            out: list[dict[str, Any]] = []
            for r in self._entries:
                k = (r.get("arch_type"), r.get("n_layers"), r.get("d_model"),
                     r.get("n_heads"), r.get("n_kv_heads"), r.get("d_ffn"),
                     r.get("vocab_size"), json.dumps(r.get("moe"), sort_keys=True))
                if k not in seen:
                    seen.add(k)
                    out.append(r)
            self._unique_cache = out
        return self._unique_cache

    def _dense(self) -> list[dict[str, Any]]:
        """The 'dense' population used for depth/param scaling (n≈91)."""
        return [r for r in self.unique()
                if r.get("arch_type") in ("decoder", "diffusion")
                and r.get("family") != "gemma3n"
                and not r.get("moe")
                and r.get("d_ffn")
                and r.get("attention_variant") in ("mha", "gqa", "mqa")]

    def regressions(self) -> dict[str, dict[str, Any]]:
        """Empirical scaling laws over the DB (cached).

        Returns ``{relation: {'coef': ..., 'resid_std': ..., 'n': ...,
        'method': ...}}`` for:

        * ``layers_vs_dmodel``  – ``ln L = a + b ln d`` (dense population)
        * ``head_dim``          – mode + range + probability distribution
        * ``ffn_ratio``         – d_ffn/d by {dense gated, dense non-gated, moe}
        * ``params_vs_ldv``     – ``ln(P − kVd) = a + b lnL + c lnd`` (dense)
        * ``moe_active_ratio``  – ``ln(active/total) = a + b ln(top_k/E)``
        * ``gqa_ratio_by_era``  – n_kv_heads/n_heads distribution by release year

        Every numeric coefficient is finite.  These are the priors the
        dimension estimator uses.
        """
        if self._regressions_cache is not None:
            return self._regressions_cache
        out: dict[str, dict[str, Any]] = {}
        out["layers_vs_dmodel"] = self._reg_layers()
        out["head_dim"] = self._reg_head_dim()
        out["ffn_ratio"] = self._reg_ffn_ratio()
        out["params_vs_ldv"] = self._reg_params()
        out["moe_active_ratio"] = self._reg_moe_ratio()
        out["gqa_ratio_by_era"] = self._reg_gqa_by_era()
        self._regressions_cache = out
        return out

    # ---- individual fits ------------------------------------------------
    def _reg_layers(self) -> dict[str, Any]:
        D = self._dense()
        L = np.log([r["n_layers"] for r in D])
        d = np.log([r["d_model"] for r in D])
        slope, intercept = np.polyfit(d, L, 1)
        resid = L - (slope * d + intercept)
        return {
            "coef": {"intercept": float(intercept), "slope": float(slope)},
            "resid_std": float(np.std(resid, ddof=2)),
            "n": int(len(D)),
            "method": "OLS log-log: ln(n_layers) = intercept + slope*ln(d_model), dense population",
        }

    def _reg_head_dim(self) -> dict[str, Any]:
        vals = [r.get("head_dim") for r in self.unique()
                if isinstance(r.get("head_dim"), (int, float)) and r.get("head_dim")]
        arr = np.array(vals, dtype=float)
        uniq, counts = np.unique(arr, return_counts=True)
        order = np.argsort(-counts)
        dist = {int(uniq[i]): int(counts[i]) for i in order}
        probs = {k: v / arr.size for k, v in dist.items()}
        mode = int(uniq[int(np.argmax(counts))])
        return {
            "coef": {"mode": mode, "range": [int(arr.min()), int(arr.max())],
                     "counts": dist, "probs": probs},
            "resid_std": float(np.std(arr)),
            "n": int(arr.size),
            "method": "empirical distribution of head_dim over unique configs",
        }

    def _reg_ffn_ratio(self) -> dict[str, Any]:
        U = self.unique()

        def stats(ratios: list[float]) -> dict[str, Any]:
            a = np.array(ratios, dtype=float)
            if a.size == 0:
                return {"n": 0}
            logsd = float(np.std(np.log(a), ddof=1)) if a.size > 1 else 0.0
            return {"median": float(np.median(a)), "mean": float(np.mean(a)),
                    "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
                    "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                    "log_sd": logsd, "n": int(a.size)}

        nongated = [r["d_ffn"] / r["d_model"] for r in U
                    if not r.get("moe") and r.get("gated_ffn") is False
                    and r.get("d_ffn") and r.get("d_model")]
        gated = [r["d_ffn"] / r["d_model"] for r in U
                 if not r.get("moe") and r.get("gated_ffn") is True
                 and r.get("d_ffn") and r.get("d_model")]
        moe = [r["moe_intermediate"] / r["d_model"] for r in U
               if r.get("moe") and r.get("moe_intermediate") and r.get("d_model")]
        coef = {"dense_nongated": stats(nongated),
                "dense_gated": stats(gated),
                "moe_expert": stats(moe)}
        return {
            "coef": coef,
            "resid_std": coef["dense_gated"].get("log_sd", 0.0),
            "n": len(nongated) + len(gated) + len(moe),
            "method": "d_ffn/d_model (dense, split by gated_ffn) and moe_intermediate/d_model (MoE)",
        }

    def _reg_params(self) -> dict[str, Any]:
        rows_x: list[list[float]] = []
        rows_y: list[float] = []
        for r in self._dense():
            V, dm, Lr = r.get("vocab_size"), r.get("d_model"), r.get("n_layers")
            P = r.get("total_params_b")
            if not (V and dm and Lr and P):
                continue
            k = 1 if r.get("tie_word_embeddings") else 2
            p_nonemb = P * 1e9 - k * V * dm
            if p_nonemb <= 0:
                continue
            rows_x.append([1.0, math.log(Lr), math.log(dm)])
            rows_y.append(math.log(p_nonemb))
        X = np.array(rows_x)
        y = np.array(rows_y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        return {
            "coef": {"intercept": float(coef[0]), "ln_L": float(coef[1]),
                     "ln_d": float(coef[2]), "k_tied": 1, "k_untied": 2},
            "resid_std": float(np.std(resid, ddof=3)),
            "n": int(len(y)),
            "method": ("OLS: ln(total_params - k*V*d) = intercept + ln_L*ln(L) + ln_d*ln(d); "
                       "k=1 if embeddings tied else 2; params in raw units"),
        }

    def _reg_moe_ratio(self) -> dict[str, Any]:
        M = [r for r in self.unique()
             if r.get("moe") and r.get("active_params_b") and r.get("total_params_b")
             and isinstance(r["moe"], dict)
             and r["moe"].get("top_k") and r["moe"].get("n_experts")]
        x = np.log([r["moe"]["top_k"] / r["moe"]["n_experts"] for r in M])
        yv = np.log([r["active_params_b"] / r["total_params_b"] for r in M])
        slope, intercept = np.polyfit(x, yv, 1)
        resid = yv - (slope * x + intercept)
        return {
            "coef": {"intercept": float(intercept), "slope": float(slope)},
            "resid_std": float(np.std(resid, ddof=2)),
            "n": int(len(M)),
            "method": "OLS: ln(active/total) = intercept + slope*ln(top_k/n_experts)",
        }

    def _reg_gqa_by_era(self) -> dict[str, Any]:
        pop = [r for r in self.unique()
               if r.get("arch_type") == "decoder"
               and r.get("attention_variant") in ("mha", "gqa", "mqa")
               and r.get("n_heads") and r.get("n_kv_heads")]
        buckets: dict[str, list[dict[str, Any]]] = {
            "<=2022": [], "2023": [], "2024": [], ">=2025": []}
        for r in pop:
            y = _to_year(r.get("release"))
            if y is None:
                continue
            if y <= 2022:
                buckets["<=2022"].append(r)
            elif y == 2023:
                buckets["2023"].append(r)
            elif y == 2024:
                buckets["2024"].append(r)
            else:
                buckets[">=2025"].append(r)
        coef: dict[str, Any] = {}
        total = 0
        for era, rs in buckets.items():
            if not rs:
                coef[era] = {"n": 0}
                continue
            total += len(rs)
            kv_over_h = np.array([r["n_kv_heads"] / r["n_heads"] for r in rs])
            h_over_kv = np.array([r["n_heads"] / r["n_kv_heads"] for r in rs])
            share_mha = float(np.mean([1.0 if r.get("attention_variant") == "mha"
                                       else 0.0 for r in rs]))
            kvcnt: dict[int, int] = {}
            for r in rs:
                kvcnt[int(r["n_kv_heads"])] = kvcnt.get(int(r["n_kv_heads"]), 0) + 1
            coef[era] = {
                "median_kv_over_h": float(np.median(kv_over_h)),
                "median_h_over_kv": float(np.median(h_over_kv)),
                "share_mha": share_mha,
                "n_kv_head_counts": dict(sorted(kvcnt.items(), key=lambda kv: -kv[1])),
                "n": int(len(rs)),
            }
        return {
            "coef": coef,
            "resid_std": None,
            "n": total,
            "method": "empirical n_kv_heads/n_heads distribution by release-year bucket",
        }

    # --------------------------------------------------------------- predictors
    def predict_n_layers(self, d_model: float) -> tuple[float, float, float]:
        """Predict (median, ~1σ low, ~1σ high) n_layers from d_model."""
        reg = self.regressions()["layers_vs_dmodel"]
        a, b = reg["coef"]["intercept"], reg["coef"]["slope"]
        s = reg["resid_std"]
        mu = a + b * math.log(d_model)
        return (math.exp(mu), math.exp(mu - s), math.exp(mu + s))

    def predict_total_params_b(self, n_layers: float, d_model: float,
                               vocab_size: float, tied: bool = False,
                               ) -> tuple[float, float, float]:
        """Predict (median, 1σ low, 1σ high) total params in billions."""
        reg = self.regressions()["params_vs_ldv"]
        c = reg["coef"]
        s = reg["resid_std"]
        k = 1 if tied else 2
        mu = c["intercept"] + c["ln_L"] * math.log(n_layers) + c["ln_d"] * math.log(d_model)
        emb = k * vocab_size * d_model
        med = (math.exp(mu) + emb) / 1e9
        lo = (math.exp(mu - s) + emb) / 1e9
        hi = (math.exp(mu + s) + emb) / 1e9
        return (med, lo, hi)


# --------------------------------------------------------------------------- helpers
def _matches(value: Any, want: Any) -> bool:
    """Match a single field value against a constraint (scalar or range)."""
    if value is None:
        return False
    if isinstance(want, (tuple, list)) and len(want) == 2 \
            and not isinstance(value, (list, tuple)):
        lo, hi = want
        try:
            if lo is not None and value < lo:
                return False
            if hi is not None and value > hi:
                return False
            return True
        except TypeError:
            return False
    if isinstance(want, str) or isinstance(value, str):
        return str(value).lower() == str(want).lower()
    return value == want


def load(path: str | Path | None = None) -> ArchDB:
    """Convenience module-level loader (see :meth:`ArchDB.load`)."""
    return ArchDB.load(path)


def _iter_records(entries: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    return entries
