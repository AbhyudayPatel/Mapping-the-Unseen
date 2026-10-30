"""Probe modules. `load_all()` imports every pNN_*.py so their @register decorators run."""
from __future__ import annotations

import importlib
import pkgutil

from forensics.probes.base import REGISTRY, Probe  # noqa: F401


def load_all() -> dict[str, type[Probe]]:
    import forensics.probes as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("p") and m.name[1:3].isdigit():
            importlib.import_module(f"forensics.probes.{m.name}")
    return REGISTRY
