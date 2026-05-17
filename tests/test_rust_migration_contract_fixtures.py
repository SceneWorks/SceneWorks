from __future__ import annotations

import json
from pathlib import Path

from scene_worker.runtime import worker_capabilities
from sceneworks_api.jobs import JobType
from sceneworks_api.jobs_store import ACTIVE_STATUSES, JOB_STATUSES, NON_GPU_JOB_TYPES, TERMINAL_STATUSES
from sceneworks_api.main import create_app
from sceneworks_api.projects import PROJECT_FOLDERS
from sceneworks_api.settings import Settings


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rust_migration_contracts"


def load_fixture(name: str) -> dict:
    with (FIXTURE_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def openapi_surface(tmp_path, monkeypatch) -> list[dict[str, str]]:
    monkeypatch.setenv("SCENEWORKS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCENEWORKS_JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    app = create_app(Settings())
    surface = []
    for path, operations in app.openapi()["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        for method in operations:
            if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                surface.append({"path": path, "method": method.upper()})
    return sorted(surface, key=lambda item: (item["path"], item["method"]))


def fixture_surface() -> list[dict[str, str]]:
    fixture = load_fixture("api_surface.json")
    surface = []
    for endpoint in fixture["endpoints"]:
        for method in endpoint["methods"]:
            surface.append({"path": endpoint["path"], "method": method})
    return sorted(surface, key=lambda item: (item["path"], item["method"]))


def test_api_surface_fixture_matches_python_openapi(tmp_path, monkeypatch):
    assert fixture_surface() == openapi_surface(tmp_path, monkeypatch)


def test_job_protocol_fixture_matches_python_constants(monkeypatch):
    fixture = load_fixture("job_protocol.json")

    assert fixture["jobTypes"] == list(JobType.__args__)
    assert fixture["statuses"] == list(JOB_STATUSES)
    assert fixture["activeStatuses"] == list(ACTIVE_STATUSES)
    assert fixture["terminalStatuses"] == list(TERMINAL_STATUSES)
    assert fixture["nonGpuJobTypes"] == list(NON_GPU_JOB_TYPES)

    monkeypatch.setenv("SCENEWORKS_UTILITY_JOBS", "1")
    assert fixture["workerCapabilityProfiles"]["cpu"] == worker_capabilities(
        {"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]}
    )

    monkeypatch.setenv("SCENEWORKS_UTILITY_JOBS", "0")
    assert fixture["workerCapabilityProfiles"]["gpuChild"] == worker_capabilities(
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]}
    )


def test_resource_sidecar_fixtures_are_loadable_and_cover_project_folders():
    fixture = load_fixture("resource_sidecars.json")

    assert fixture["projectFolders"] == PROJECT_FOLDERS
    for sidecar in fixture["fixtures"]:
        payload_path = FIXTURE_DIR / sidecar["path"]
        assert payload_path.exists(), f"Missing fixture: {payload_path}"
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert set(sidecar["requiredTopLevelKeys"]).issubset(payload.keys()), sidecar["name"]

