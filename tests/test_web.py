"""Web UI end-to-end: page loads, a simulated scan runs to completion from the API, and
its report, diagrams and raw files are served; compare works on two finished runs."""
import time

import pytest
from fastapi.testclient import TestClient

import forensics.web.app as web


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    import forensics.cli as cli
    orig = cli.new_run_dir
    monkeypatch.setattr(cli, "new_run_dir", lambda name, runs_dir=None: orig(name, tmp_path))
    web.JOBS.clear()
    web._SUMMARY_CACHE.clear()
    return TestClient(web.app)


def _wait(client, job_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.5)
    raise AssertionError("scan did not finish")


def test_page_and_catalogues(client):
    r = client.get("/")
    assert r.status_code == 200 and "Model Forensics" in r.text and "Start scan" in r.text
    presets = client.get("/api/presets").json()
    assert any(p["name"] == "sim-dense" and p["ready"] for p in presets)
    assert len(client.get("/api/probes").json()) >= 30
    assert client.get("/api/runs").json() == []


def test_scan_runs_and_serves_everything(client):
    j = client.post("/api/scan", json={"preset": "sim-dense", "mode": "quick", "max_requests": 250}).json()
    done = _wait(client, j["id"])
    assert done["status"] == "done", done
    assert done["log"] and done["run_id"]

    runs = client.get("/api/runs").json()
    assert len(runs) == 1
    run = runs[0]
    assert run["arch"] and run["arch"][0] == "decoder_only"
    for name in ("report.html", "report.md", "report.json", "architecture.mmd", "topology.mmd",
                 "evidence.mmd", "observations.jsonl"):
        assert name in run["files"], name
        r = client.get(f"/files/{run['id']}/{name}")
        assert r.status_code == 200 and r.content, name
    assert "window.__FORENSICS__" in client.get(f"/files/{run['id']}/report.html").text
    # path traversal is refused
    assert client.get(f"/files/{run['id']}/..%2F..%2Fpyproject.toml").status_code == 404

    # second run -> compare, then delete
    j2 = client.post("/api/scan", json={"preset": "sim-moe", "mode": "quick", "max_requests": 250}).json()
    assert _wait(client, j2["id"])["status"] == "done"
    ids = [r["id"] for r in client.get("/api/runs").json()]
    cmp = client.get("/api/compare", params={"a": ids[0], "b": ids[1]}).json()
    assert "verdict" in cmp and 0 <= cmp["distance"] <= 1
    assert client.delete(f"/api/runs/{ids[0]}").status_code == 200
    assert len(client.get("/api/runs").json()) == 1


def test_bad_requests(client):
    assert client.post("/api/scan", json={"mode": "quick"}).status_code == 400
    assert client.post("/api/scan", json={"provider": "openai", "base_url": "http://x"}).status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/files/nope/report.html").status_code == 404
