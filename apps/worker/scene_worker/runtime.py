from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any

import httpx

from .gpu import discover_gpu
from .image_adapters import create_image_adapter
from .settings import WorkerSettings
from .timeline_exporter import run_timeline_export
from .video_adapters import ProceduralVideoAdapter


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


class ApiClient:
    def __init__(self, settings: WorkerSettings) -> None:
        headers = {}
        if settings.access_token:
            headers["X-SceneWorks-Token"] = settings.access_token
        self.client = httpx.Client(base_url=settings.api_url, headers=headers, timeout=20)

    def post(self, path: str, payload: dict) -> dict:
        response = self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def get(self, path: str) -> dict:
        response = self.client.get(path)
        response.raise_for_status()
        return response.json()


def worker_capabilities(gpu: dict) -> list[str]:
    gpu_capabilities = set(gpu["capabilities"])
    capabilities = set(gpu["capabilities"]) | {"timeline_export", "model_download", "lora_import"}
    if "cpu" not in gpu_capabilities and "gpu" in gpu_capabilities:
        capabilities |= {"image_generate", "image_edit", "video_generate", "video_extend", "video_bridge"}
    return sorted(capabilities)


def register_worker(api: ApiClient, settings: WorkerSettings, gpu: dict) -> None:
    payload = {
        "workerId": settings.worker_id,
        "gpuId": gpu["id"],
        "gpuName": gpu["name"],
        "capabilities": worker_capabilities(gpu),
        "loadedModels": [],
    }
    worker = api.post("/api/v1/workers/register", payload)
    emit({"event": "registered", "worker": worker, "reportedAt": now()})


def heartbeat(
    api: ApiClient,
    settings: WorkerSettings,
    status: str,
    current_job_id: str | None = None,
) -> None:
    api.post(
        f"/api/v1/workers/{settings.worker_id}/heartbeat",
        {"status": status, "currentJobId": current_job_id, "loadedModels": []},
    )


def update_job(api: ApiClient, job_id: str, payload: dict[str, Any]) -> dict:
    job = api.post(f"/api/v1/jobs/{job_id}/progress", payload)
    emit({"event": "job_progress", "jobId": job_id, "status": job["status"], "stage": job["stage"]})
    return job


def job_cancel_requested(api: ApiClient, job_id: str) -> bool:
    return bool(api.get(f"/api/v1/jobs/{job_id}")["cancelRequested"])


def safe_download_dir(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "._-" else "__" for char in value)
    return normalized.strip("_") or "download"


def snapshot_huggingface_repo(repo: str, target_dir: Path, files: list[str] | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for model and LoRA downloads") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=target_dir,
        allow_patterns=files or None,
        local_dir_use_symlinks=False,
    )
    return target_dir


def run_model_download_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    payload = job["payload"]
    repo = payload.get("repo")
    if not repo:
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Model download is missing a repository.",
                "error": "Missing payload.repo",
            },
        )
        heartbeat(api, settings, "idle")
        return

    target_dir = Path(payload.get("targetDir") or settings.data_dir / "models" / safe_download_dir(repo))
    try:
        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": "downloading",
                "stage": "downloading",
                "progress": 0.1,
                "message": f"Downloading {repo}.",
            },
        )
        if job_cancel_requested(api, job_id):
            raise InterruptedError("Model download canceled before transfer started.")
        snapshot_huggingface_repo(repo, target_dir, payload.get("files") or [])
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Model download completed.",
                "result": {
                    "modelId": payload.get("modelId"),
                    "repo": repo,
                    "path": str(target_dir),
                    "completedAt": now(),
                },
            },
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Model download failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_lora_import_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    payload = job["payload"]
    repo = payload.get("repo")
    source_path = payload.get("sourcePath")
    target_name = safe_download_dir(payload.get("loraId") or payload.get("name") or repo or Path(source_path or "lora").stem)
    target_dir = settings.data_dir / "loras" / target_name

    try:
        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": "downloading",
                "stage": "importing",
                "progress": 0.1,
                "message": "Importing LoRA.",
            },
        )
        if job_cancel_requested(api, job_id):
            raise InterruptedError("LoRA import canceled before transfer started.")
        if repo:
            snapshot_huggingface_repo(repo, target_dir, payload.get("files") or [])
        elif source_path:
            source = Path(source_path).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(f"LoRA source not found: {source}")
            target_dir.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target_dir, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target_dir / source.name)
        else:
            raise ValueError("Provide repo or sourcePath for LoRA import")
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "LoRA import completed.",
                "result": {"repo": repo, "path": str(target_dir), "completedAt": now()},
            },
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "LoRA import failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_placeholder_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    stages = [
        ("preparing", "preparing", 0.1, "Preparing placeholder job."),
        ("running", "running", 0.35, "Running placeholder step 1."),
        ("running", "running", 0.65, "Running placeholder step 2."),
        ("saving", "saving", 0.9, "Saving placeholder result."),
    ]

    for status, stage, progress, message in stages:
        if job_cancel_requested(api, job_id):
            update_job(
                api,
                job_id,
                {
                    "status": "canceled",
                    "stage": "canceled",
                    "progress": progress,
                    "message": "Worker canceled the job before completion.",
                },
            )
            heartbeat(api, settings, "idle")
            return

        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
            },
        )
        time.sleep(1.5)

    update_job(
        api,
        job_id,
        {
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Placeholder job completed.",
            "result": {"completedAt": now(), "output": "placeholder"},
        },
    )
    heartbeat(api, settings, "idle")


def run_image_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    adapter = create_image_adapter(job)

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": status,
                "stage": stage,
                "progress": value,
                "message": message,
            },
        )

    try:
        progress("preparing", "preparing", 0.08, "Preparing Image Studio request.")
        progress("loading_model", "loading_model", 0.16, "Resolving image adapter target.")
        result = adapter.generate(
            settings=settings,
            job=job,
            progress=progress,
            cancel_requested=lambda: job_cancel_requested(api, job_id),
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Image generation assets saved.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Image generation failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_video_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    adapter = ProceduralVideoAdapter()

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": status,
                "stage": stage,
                "progress": value,
                "message": message,
            },
        )

    try:
        progress("preparing", "preparing", 0.06, "Preparing Video Studio request.")
        request = adapter.prepare(settings=settings, job=job)
        progress("loading_model", "loading_model", 0.14, "Resolving video adapter target.")
        adapter.ensure_models(request)
        requirements = adapter.estimate_requirements(request)
        progress(
            "running",
            "estimating",
            0.18,
            f"Estimated {requirements['previewFrames']} preview frames for this clip.",
        )
        result = adapter.run(
            settings=settings,
            job=job,
            request=request,
            progress=progress,
            cancel_requested=lambda: job_cancel_requested(api, job_id),
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Video generation asset saved.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        adapter.cancel(job_id)
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        adapter.cleanup(job_id)
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Video generation failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_timeline_export_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": status,
                "stage": stage,
                "progress": value,
                "message": message,
            },
        )

    try:
        progress("preparing", "preparing", 0.06, "Preparing timeline export.")
        result = run_timeline_export(
            settings=settings,
            job=job,
            progress=progress,
            cancel_requested=lambda: job_cancel_requested(api, job_id),
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Timeline MP4 export saved.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Timeline export failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def main() -> None:
    settings = WorkerSettings()
    gpu = discover_gpu(settings.gpu_id)
    api = ApiClient(settings)
    max_registration_attempts = 20

    for attempt in range(1, max_registration_attempts + 1):
        try:
            register_worker(api, settings, gpu)
            break
        except httpx.HTTPError as exc:
            delay = min(30, settings.poll_seconds * (2 ** (attempt - 1)))
            emit(
                {
                    "event": "register_failed",
                    "attempt": attempt,
                    "maxAttempts": max_registration_attempts,
                    "retryInSeconds": delay,
                    "error": str(exc),
                    "reportedAt": now(),
                }
            )
            if attempt == max_registration_attempts:
                raise RuntimeError(f"Worker registration failed after {max_registration_attempts} attempts.") from exc
            time.sleep(delay)

    while True:
        try:
            heartbeat(api, settings, "idle")
            claimed = api.post("/api/v1/jobs/claim", {"workerId": settings.worker_id})
            job = claimed.get("job")
            if job is None:
                time.sleep(settings.poll_seconds)
                continue

            emit({"event": "claimed", "jobId": job["id"], "gpuId": job["assignedGpu"], "reportedAt": now()})
            if job["type"] == "placeholder":
                run_placeholder_job(api, settings, job)
            elif job["type"] in ("image_generate", "image_edit"):
                run_image_job(api, settings, job)
            elif job["type"] in ("video_generate", "video_extend", "video_bridge"):
                run_video_job(api, settings, job)
            elif job["type"] == "timeline_export":
                run_timeline_export_job(api, settings, job)
            elif job["type"] == "model_download":
                run_model_download_job(api, settings, job)
            elif job["type"] == "lora_import":
                run_lora_import_job(api, settings, job)
            else:
                update_job(
                    api,
                    job["id"],
                    {
                        "status": "failed",
                        "stage": "failed",
                        "progress": 1,
                        "message": "No adapter exists for this job type yet.",
                        "error": f"Unsupported job type: {job['type']}",
                    },
                )
        except httpx.HTTPError as exc:
            emit({"event": "api_error", "error": str(exc), "reportedAt": now()})
            time.sleep(settings.poll_seconds)
