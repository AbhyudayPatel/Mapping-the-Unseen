"""Reference fingerprint library: save, load and match behavioural fingerprints.

Fingerprints (from :func:`forensics.fingerprint.vector.fingerprint`) can be persisted to a
JSON library — by default ``forensics/knowledge/reference_fingerprints.json`` — and a freshly
measured fingerprint can be matched against the library by nearest
:func:`~forensics.fingerprint.vector.fingerprint_distance`.

The file format is::

    {
      "<name>": {"fingerprint": {<feature>: <value>, ...},
                 "meta": {<free-form provenance>}},
      ...
    }

Everything is defensive: a missing/corrupt file loads as an empty library, and non-JSON
values (numpy scalars) are coerced on save.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from forensics.fingerprint.vector import fingerprint, fingerprint_distance

__all__ = [
    "REFERENCE_PATH", "save_reference", "save_reference_from_run",
    "load_references", "match_fingerprint", "match_run",
]

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "reference_fingerprints.json"


# --------------------------------------------------------------------------- json coercion
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    return str(obj)


def _resolve(path: str | Path | None) -> Path:
    return Path(path) if path else REFERENCE_PATH


# --------------------------------------------------------------------------- load / save
def load_references(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the reference library (``{name: {"fingerprint":..., "meta":...}}``).

    A missing or unparseable file returns an empty dict rather than raising.
    """
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, rec in data.items():
        if isinstance(rec, dict) and isinstance(rec.get("fingerprint"), dict):
            out[str(name)] = {"fingerprint": rec["fingerprint"],
                              "meta": rec.get("meta", {}) or {}}
        elif isinstance(rec, dict):
            # tolerate a bare fingerprint stored directly under the name
            out[str(name)] = {"fingerprint": rec, "meta": {}}
    return out


def save_reference(name: str, fp: dict[str, Any], meta: dict[str, Any] | None = None,
                   path: str | Path | None = None) -> Path:
    """Add/replace fingerprint ``name`` in the library and write it back to disk.

    Returns the path written.  Creates the file (and parent dir) if missing.
    """
    p = _resolve(path)
    lib = load_references(p)
    lib[str(name)] = {"fingerprint": _jsonable(fp), "meta": _jsonable(meta or {})}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(lib), fh, ensure_ascii=False, indent=2, sort_keys=True)
    return p


def save_reference_from_run(name: str, run: Any, meta: dict[str, Any] | None = None,
                            path: str | Path | None = None) -> Path:
    """Fingerprint ``run`` and persist it under ``name`` (convenience wrapper)."""
    m = dict(meta or {})
    m.setdefault("run_id", getattr(run, "run_id", ""))
    m.setdefault("target", getattr(run, "target", {}))
    return save_reference(name, fingerprint(run), meta=m, path=path)


# --------------------------------------------------------------------------- matching
def match_fingerprint(fp: dict[str, Any], references: dict[str, dict[str, Any]] | None = None,
                      top_k: int = 5, path: str | Path | None = None,
                      ) -> list[tuple[str, float, dict[str, Any]]]:
    """Rank library entries by ascending distance from ``fp``.

    Returns ``[(name, distance, meta), ...]`` sorted nearest-first (length ``<= top_k``).
    Loads the on-disk library when ``references`` is not supplied.
    """
    lib = references if references is not None else load_references(path)
    ranked: list[tuple[str, float, dict[str, Any]]] = []
    for name, rec in lib.items():
        ref_fp = rec.get("fingerprint") if isinstance(rec, dict) else None
        if not isinstance(ref_fp, dict):
            continue
        try:
            dist = fingerprint_distance(fp, ref_fp)
        except Exception:  # noqa: BLE001
            continue
        ranked.append((name, float(dist), rec.get("meta", {}) if isinstance(rec, dict) else {}))
    ranked.sort(key=lambda t: t[1])
    return ranked[:top_k] if top_k else ranked


def match_run(run: Any, references: dict[str, dict[str, Any]] | None = None,
              top_k: int = 5, path: str | Path | None = None,
              ) -> list[tuple[str, float, dict[str, Any]]]:
    """Fingerprint ``run`` and match it against the library (convenience wrapper)."""
    return match_fingerprint(fingerprint(run), references=references, top_k=top_k, path=path)
