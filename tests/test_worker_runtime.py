from __future__ import annotations

from scene_worker.image_adapters import resolve_seed
from scene_worker.runtime import download_progress_payload, format_bytes, heartbeat, loaded_models_from_adapters, worker_capabilities


def test_cpu_worker_does_not_advertise_gpu_generation_capabilities():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert "image_generate" not in capabilities
    assert "video_generate" not in capabilities
    assert "model_download" in capabilities
    assert "timeline_export" in capabilities


def test_gpu_worker_advertises_generation_capabilities():
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})

    assert "image_generate" in capabilities
    assert "video_generate" in capabilities


def test_auto_gpu_worker_can_disable_utility_capabilities(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_UTILITY_JOBS", "0")

    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})

    assert "image_generate" in capabilities
    assert "model_download" not in capabilities
    assert "lora_import" not in capabilities


def test_loaded_models_are_collected_from_adapter_cache():
    class Adapter:
        def loaded_models(self):
            return ["Tongyi-MAI/Z-Image-Turbo"]

    assert loaded_models_from_adapters({"z": Adapter()}) == ["Tongyi-MAI/Z-Image-Turbo"]


def test_heartbeat_loaded_models_are_not_sent_as_current_job():
    class Api:
        def __init__(self):
            self.path = None
            self.payload = None

        def post(self, path, payload):
            self.path = path
            self.payload = payload
            return {}

    class Settings:
        worker_id = "worker-1"

    api = Api()
    heartbeat(api, Settings(), "idle", loaded_models=["model-a"])

    assert api.path == "/api/v1/workers/worker-1/heartbeat"
    assert api.payload == {"status": "idle", "currentJobId": None, "loadedModels": ["model-a"]}


def test_random_batch_seeds_are_used_per_image():
    assert resolve_seed(None, "city at night", 2, [101, 202, 303, 404]) == 303


def test_explicit_seed_uses_reproducible_ladder():
    assert resolve_seed(1234, "city at night", 2, [101, 202, 303, 404]) == 1236


def test_download_progress_payload_reports_remaining_bytes(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.time.monotonic", lambda: 20.0)

    payload = download_progress_payload(
        "owner/model",
        downloaded_bytes=512 * 1024 * 1024,
        total_bytes=1024 * 1024 * 1024,
        started_bytes=0,
        started_at=10.0,
    )

    assert payload["status"] == "downloading"
    assert payload["stage"] == "downloading"
    assert payload["progress"] == 0.525
    assert payload["message"] == "Downloading owner/model: 512.0 MB of 1.0 GB (512.0 MB left)."
    assert payload["etaSeconds"] == 10.0


def test_download_progress_payload_handles_unknown_total(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.time.monotonic", lambda: 20.0)

    payload = download_progress_payload(
        "owner/model",
        downloaded_bytes=128 * 1024 * 1024,
        total_bytes=None,
        started_bytes=0,
        started_at=10.0,
    )

    assert payload["progress"] == 0.1
    assert payload["message"] == "Downloading owner/model: 128.0 MB written."
    assert payload["etaSeconds"] is None


def test_format_bytes_uses_readable_units():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024 * 1024) == "1.0 MB"
