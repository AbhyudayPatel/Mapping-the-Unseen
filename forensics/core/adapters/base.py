"""Adapter interface. An adapter turns our neutral request types into provider calls.

Every adapter:
- measures timing itself and fills `Timing` (relative seconds, chunk arrival times);
- never raises on HTTP/API errors: it returns a response with `error`/`status` set
  (error texts are forensic evidence). It only raises for programming errors;
- reports what it knows statically via `static_capabilities()` (discovery refines it);
- passes every call through the `Recorder` when one is attached (see core/recorder.py).
"""
from __future__ import annotations

import abc
from typing import Any

from forensics.core.types import (
    Capabilities, DecisionRequest, DecisionResponse, GenRequest, GenResponse,
)


class Adapter(abc.ABC):
    #: "generative" | "decision" | "embedding"
    kind: str = "generative"
    #: human-readable identity of the target (for the report); never used as evidence
    name: str = "unknown"

    def __init__(self) -> None:
        self.recorder = None   # set by the engine: forensics.core.recorder.Recorder
        self.budget = None     # set by the engine: forensics.core.budget.Budget

    def static_capabilities(self) -> Capabilities:
        return Capabilities(kind=self.kind)  # type: ignore[arg-type]

    async def generate(self, req: GenRequest) -> GenResponse:
        raise NotImplementedError(f"{type(self).__name__} does not support generate()")

    async def decide(self, req: DecisionRequest) -> DecisionResponse:
        raise NotImplementedError(f"{type(self).__name__} does not support decide()")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(f"{type(self).__name__} does not support embed()")

    async def aclose(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        """Non-secret description of the target for the report (no API keys!)."""
        return {"adapter": type(self).__name__, "name": self.name, "kind": self.kind}
