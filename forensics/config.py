"""Environment/config helpers. Secrets are read from the environment or a local .env file
and are never written to run artefacts."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Load KEY=VALUE lines from .env into os.environ (without overriding existing vars)."""
    p = Path(path) if path else ROOT / ".env"
    loaded: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            loaded[k] = v
            os.environ.setdefault(k, v)
    return loaded


SECRET_MARKERS = ("key", "token", "secret", "authorization", "password")


def redact(d: dict) -> dict:
    """Redact secret-looking header/config values."""
    return {k: ("***" if any(m in k.lower() for m in SECRET_MARKERS) else v) for k, v in d.items()}
