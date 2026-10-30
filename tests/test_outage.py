"""A network outage must abort the run, not become 'evidence' about the model."""
import asyncio

from forensics.core.adapters.simulated import SimulatedAdapter, sim_decoder_dense
from forensics.core.types import GenResponse
from forensics.inference.engine import run_forensics


class FlakyAdapter(SimulatedAdapter):
    """Works for the first `ok_calls` requests, then DNS fails forever."""

    def __init__(self, cfg, ok_calls: int):
        super().__init__(cfg)
        self.ok_calls = ok_calls
        self.calls = 0

    async def generate(self, req):
        self.calls += 1
        if self.calls > self.ok_calls:
            return GenResponse(status=None, error="ConnectError: [Errno 11001] getaddrinfo failed")
        return await super().generate(req)


def test_outage_aborts_run_and_discards_error_evidence():
    ad = FlakyAdapter(sim_decoder_dense(), ok_calls=40)
    rr = asyncio.run(run_forensics(ad, config={"fast": True}))
    assert rr.budget_summary["aborted_unreachable"] is True
    assert any(str(n).startswith("ABORTED") for n in rr.notes)
    # the run stops soon after the outage instead of burning through every probe
    assert ad.calls < 40 + 30
    # no probe that ran during the outage contributed evidence
    for pr in rr.probe_results:
        if pr.status == "failed" and "transport" in " ".join(pr.notes + [pr.error or ""]):
            assert not pr.evidence
