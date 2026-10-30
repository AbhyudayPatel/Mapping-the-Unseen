"""Recorder: append-only capture of every request/response for replay and audit.

The :class:`Recorder` writes one JSON line per observation to
``<run_dir>/observations.jsonl`` and a single ``<run_dir>/run_meta.json``. It is
safe to call concurrently (an internal lock serialises writes). Secret-looking
header values (Authorization / api-key / token / ...) are redacted via
:func:`forensics.config.redact` before anything touches disk; forensic payload
fields such as per-token logprobs are preserved verbatim.

The recorded lines are the input to ``ReplayAdapter`` (see the dev guide), so the
schema is intentionally simple and stable:

    {"ts": <epoch>, "kind": <str>, "probe": <str>,
     "request": <jsonable>, "response": <jsonable>}
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from forensics.config import SECRET_MARKERS, redact
from forensics.core.types import to_jsonable


def _redact_headers_only(obj: Any) -> Any:
    """Recursively redact only dicts stored under a ``*header*`` key.

    This deliberately leaves every other field untouched so that forensic data
    (e.g. a ``token`` string inside a logprob entry) survives, while API keys in
    request/response headers do not.
    """
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and "header" in k.lower() and isinstance(v, dict):
                out[k] = redact(v)
            else:
                out[k] = _redact_headers_only(v)
        return out
    if isinstance(obj, list):
        return [_redact_headers_only(x) for x in obj]
    return obj


def _redact_deep(obj: Any) -> Any:
    """Recursively redact secret-looking keys and header dicts (for run metadata)."""
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            ks = str(k).lower()
            if any(m in ks for m in SECRET_MARKERS):
                out[k] = "***"
            elif "header" in ks and isinstance(v, dict):
                out[k] = redact(v)
            else:
                out[k] = _redact_deep(v)
        return out
    if isinstance(obj, list):
        return [_redact_deep(x) for x in obj]
    return obj


class Recorder:
    """Append-only, thread/async-safe recorder of requests and responses."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.obs_path = self.run_dir / "observations.jsonl"
        self.meta_path = self.run_dir / "run_meta.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ writing
    def record(self, kind: str, req: Any, resp: Any, probe: str = "") -> None:
        """Append one observation line. ``req``/``resp`` may be dataclasses or dicts."""
        rec = {
            "ts": time.time(),
            "kind": kind,
            "probe": probe,
            "request": _redact_headers_only(to_jsonable(req)),
            "response": _redact_headers_only(to_jsonable(resp)),
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with self.obs_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def meta(self, data: dict[str, Any]) -> None:
        """Write ``run_meta.json`` (secret-looking keys/headers redacted)."""
        payload = _redact_deep(to_jsonable(data))
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        with self._lock:
            self.meta_path.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ reading
    @staticmethod
    def iter_observations(run_dir: str) -> Iterator[dict[str, Any]]:
        """Yield recorded observations (as dicts) from ``run_dir``."""
        path = Path(run_dir) / "observations.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def read_meta(run_dir: str) -> dict[str, Any]:
        """Read back ``run_meta.json`` (empty dict if absent)."""
        path = Path(run_dir) / "run_meta.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
