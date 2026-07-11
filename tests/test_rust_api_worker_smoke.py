from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

import httpx
import pytest

# In CI a missing cargo means the workflow toolchain regressed, so `require_tool`
# fails the e2e gate instead of letting it skip to a silent green (sc-8935 / F-133).
from conftest import require_tool

# Shared, corruption-free fixtures (sc-8934 / F-132): PNG + safetensors builders
# and the API-spawn helpers, previously copy-pasted (and drifted) between this
# file and test_rust_api_contract_snapshots.py.
from rust_api_harness import (
    PNG_1X1,
    free_port,
    minimal_safetensors as _minimal_safetensors,
    wait_for_health,
)

# sc-8861 (F-059): these e2e tests used to drive the live Rust API via
# `scene_worker.runtime.ApiClient` and run the image pipeline with the in-process
# Python worker (`run_image_job`). Both couple the ONLY live coverage of the Rust
# API's worker protocol / procedural-pipeline / sc-2226 LoRA boundary to the
# retired `apps/worker` package, so an epic-8283 deletion would silently take that
# coverage down. They are now PURE-HTTP: the test itself plays the worker role over
# the same endpoints (register → claim → heartbeat → cancel → progress), and for the
# procedural / LoRA jobs it reports the same flat asset facts the worker posts (the
# Rust API is the single owner of the fact→asset transform, story 1656), writing the
# PNG bytes to the shared project dir exactly as the worker's `ImageAssetWriter` did.
# No `scene_worker` import remains. The `rust_api` fixture spawns
# `cargo run -p sceneworks-rust-api`, so the whole module is e2e: it must run in the
# CI step that follows the Rust build, not the lightweight worker-suite step (sc-4180).
pytestmark = pytest.mark.e2e


def register_worker(base_url: str, worker_id: str, gpu: dict) -> dict:
    """POST /api/v1/workers/register. The image queue only offers an
    image_generate job to a worker advertising that capability, so the tests
    register with it explicitly (the retired Python `worker_capabilities` derived
    a richer set; image routing only needs `image_generate`)."""
    response = httpx.post(
        f"{base_url}/api/v1/workers/register",
        json={
            "workerId": worker_id,
            "gpuId": gpu["id"],
            "gpuName": gpu["name"],
            "capabilities": gpu["capabilities"],
            "loadedModels": gpu.get("loadedModels", []),
        },
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def claim_job(base_url: str, worker_id: str) -> dict:
    """POST /api/v1/jobs/claim — returns the claimed job envelope's `job`."""
    response = httpx.post(
        f"{base_url}/api/v1/jobs/claim", json={"workerId": worker_id}, timeout=5
    )
    response.raise_for_status()
    return response.json()["job"]


def heartbeat(base_url: str, worker_id: str, status: str, current_job_id: str | None, loaded_models: list[str]) -> None:
    """POST /api/v1/workers/{id}/heartbeat."""
    response = httpx.post(
        f"{base_url}/api/v1/workers/{worker_id}/heartbeat",
        json={"status": status, "currentJobId": current_job_id, "loadedModels": loaded_models},
        timeout=5,
    )
    response.raise_for_status()


def job_cancel_requested(base_url: str, job_id: str) -> bool:
    """GET /api/v1/jobs/{id} → cancelRequested."""
    response = httpx.get(f"{base_url}/api/v1/jobs/{job_id}", timeout=5)
    response.raise_for_status()
    return bool(response.json()["cancelRequested"])


def report_progress(base_url: str, worker_id: str, job_id: str, payload: dict) -> dict:
    """POST /api/v1/jobs/{id}/progress, stamping the reporting worker id like the
    worker's `update_job` did (the server 409s if this worker no longer owns the
    job — sc-4172)."""
    body = {"workerId": worker_id, **payload}
    response = httpx.post(f"{base_url}/api/v1/jobs/{job_id}/progress", json=body, timeout=5)
    response.raise_for_status()
    return response.json()


def build_procedural_asset_writes(
    *,
    project_path: Path,
    generation_set_id: str,
    request: dict,
    loras: list[dict] | None = None,
) -> list[dict]:
    """Write a weightless PNG per image under `assets/images/<genset>/` and return
    the flat `assetWrites` facts, mirroring the shape the Python worker's
    `ImageAssetWriter.write_incremental_outputs` posted (the Rust API's
    `build_image_sidecar_parts` turns each fact into the served `result.assets`
    entry, so `file`/`recipe` fields below drive the completed job's assets)."""
    model = request["model"]
    prompt = request["prompt"]
    count = request.get("count", 1)
    width = request.get("width", 512)
    height = request.get("height", 512)
    date_slug = time.strftime("%Y-%m-%d")
    images_dir = project_path / "assets" / "images" / generation_set_id
    images_dir.mkdir(parents=True, exist_ok=True)
    writes: list[dict] = []
    for index in range(count):
        filename = f"{date_slug}_{model}_image_{index + 1:04d}.png"
        (images_dir / filename).write_bytes(PNG_1X1)
        writes.append(
            {
                "assetId": f"asset_{uuid4().hex}",
                "mediaPath": f"assets/images/{generation_set_id}/{filename}",
                "mimeType": "image/png",
                "width": width,
                "height": height,
                "normalizedWidth": width,
                "normalizedHeight": height,
                "count": count,
                "family": "z-image",
                "seed": index,
                "index": index,
                "displayName": f"{prompt[:56] or 'Generated image'} #{index + 1}",
                "createdAt": date_slug,
                "mode": request.get("mode", "text_to_image"),
                "model": model,
                "adapter": "procedural_preview",
                "prompt": prompt,
                "negativePrompt": request.get("negativePrompt"),
                "loras": loras or [],
                "sourceAssetId": request.get("referenceAssetId"),
                "rawAdapterSettings": dict(request.get("advanced", {})),
            }
        )
    return writes


def complete_image_job(
    base_url: str,
    worker_id: str,
    job: dict,
    *,
    project_path: Path,
    loras: list[dict] | None = None,
) -> dict:
    """Play the worker's procedural image path over HTTP: emit the flat asset
    facts + PNG bytes, then POST the terminal `completed` progress so the Rust API
    builds + serves `result.assets`. Returns the API's progress response."""
    request = dict(job["payload"])
    generation_set_id = f"genset_{uuid4().hex}"
    asset_writes = build_procedural_asset_writes(
        project_path=project_path,
        generation_set_id=generation_set_id,
        request=request,
        loras=loras,
    )
    result = {
        "generationSetId": generation_set_id,
        "expectedCount": len(asset_writes),
        "adapter": "procedural_preview",
        "model": request["model"],
        "generationSet": {
            "id": generation_set_id,
            "mode": request.get("mode", "text_to_image"),
            "model": request["model"],
            "prompt": request["prompt"],
            "count": len(asset_writes),
        },
        "assetWrites": asset_writes,
    }
    return report_progress(
        base_url,
        worker_id,
        job["id"],
        {"status": "completed", "stage": "completed", "progress": 1, "message": "Image generation assets saved.", "result": result},
    )


ROOT = Path(__file__).resolve().parents[1]


def _launch_command(binary_env: str, package: str, purpose: str) -> list[str]:
    """Command to spawn a workspace binary for the e2e smoke.

    Prefer a prebuilt binary exported by CI (SCENEWORKS_RUST_API_BINARY /
    SCENEWORKS_RUST_WORKER_BINARY) so the ~30 API+worker spawns across the e2e
    and parity steps skip `cargo run`'s per-launch freshness check over the whole
    workspace graph (sc-10823). Local ad-hoc runs with no export fall back to
    `cargo run`. A binary path that is set but missing means the earlier build
    step regressed, so fail loudly rather than silently reverting to cargo —
    mirroring the require_tool honesty guard (sc-8935 / F-133)."""
    binary = os.getenv(binary_env)
    if binary:
        if not Path(binary).is_file():
            raise AssertionError(f"{binary_env}={binary} does not exist")
        return [binary]
    require_tool("cargo", purpose)
    return ["cargo", "run", "-q", "-p", package]


def wait_for_job_status(base_url: str, job_id: str, status: str, process: subprocess.Popen) -> dict:
    deadline = time.monotonic() + 30
    last_job: dict | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"Rust worker exited early with code {process.returncode}: {stderr}")
        response = httpx.get(f"{base_url}/api/v1/jobs/{job_id}", timeout=5)
        response.raise_for_status()
        last_job = response.json()
        if last_job["status"] == status:
            return last_job
        if last_job["status"] in {"failed", "canceled", "interrupted"}:
            raise AssertionError(f"Job reached terminal status {last_job['status']}: {last_job}")
        time.sleep(0.25)
    raise AssertionError(f"Job did not reach {status}: {last_job}")


@pytest.fixture()
def rust_api(tmp_path):
    command = _launch_command(
        "SCENEWORKS_RUST_API_BINARY", "sceneworks-rust-api", "the Rust API smoke test"
    )

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "SCENEWORKS_API_HOST": "127.0.0.1",
            "SCENEWORKS_API_PORT": str(port),
            "SCENEWORKS_DATA_DIR": str(tmp_path / "data"),
            "SCENEWORKS_CONFIG_DIR": str(tmp_path / "config"),
            "SCENEWORKS_JOBS_DB_PATH": str(tmp_path / "cache" / "jobs.db"),
            "SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE": "1",
        }
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_health(base_url, process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_python_worker_protocol_round_trips_against_rust_api_binary(rust_api):
    # sc-8861: the live claim → heartbeat → cancel → terminal-progress contract,
    # driven entirely over HTTP (was `scene_worker.runtime.ApiClient`).
    worker_id = "live-test-worker"

    # An image_generate job is only offered to workers advertising the
    # image_generate capability, so register with it or the claim below returns no job.
    register_worker(
        rust_api,
        worker_id,
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["image_generate"]},
    )
    created = httpx.post(
        f"{rust_api}/api/v1/image/jobs",
        json={
            "projectId": "project-1",
            "prompt": "mist over hills",
            "model": "z_image_turbo",
            "requestedGpu": "gpu-0",
        },
        timeout=5,
    )
    created.raise_for_status()
    job = created.json()

    claimed = claim_job(rust_api, worker_id)
    assert claimed["id"] == job["id"]
    assert claimed["workerId"] == worker_id
    assert claimed["assignedGpu"] == "gpu-0"

    heartbeat(rust_api, worker_id, "busy", claimed["id"], loaded_models=["Tongyi-MAI/Z-Image-Turbo"])
    workers = httpx.get(f"{rust_api}/api/v1/workers", timeout=5).json()
    worker = next(worker for worker in workers if worker["id"] == worker_id)
    assert worker["loadedModels"] == ["Tongyi-MAI/Z-Image-Turbo"]

    canceled = httpx.post(f"{rust_api}/api/v1/jobs/{claimed['id']}/cancel", timeout=5)
    canceled.raise_for_status()
    assert job_cancel_requested(rust_api, claimed["id"]) is True

    completed = report_progress(
        rust_api,
        worker_id,
        claimed["id"],
        {
            "status": "canceled",
            "stage": "canceled",
            "progress": 1,
            "message": "Worker canceled the job before completion.",
        },
    )
    assert completed["status"] == "canceled"
    assert completed["cancelRequested"] is True


def test_python_worker_completes_procedural_image_job_against_rust_api_binary(rust_api):
    # sc-8861: the procedural image pipeline's API contract — create → claim →
    # asset-writing (flat facts + PNG bytes) → completion → served `result.assets`.
    # The test plays the worker's procedural path over HTTP (was in-process
    # `run_image_job` -> ProceduralImageAdapter -> ImageAssetWriter). The Rust API
    # owns the fact→asset transform (story 1656), so this exercises the same
    # asset-writing contract the served assertions below cover.
    worker_id = "image-e2e-worker"

    # Create the project through the API so its store resolves the same path the
    # worker (here, the test) writes PNG bytes into.
    created_project = httpx.post(
        f"{rust_api}/api/v1/projects", json={"name": "Procedural E2E"}, timeout=5
    )
    created_project.raise_for_status()
    project = created_project.json()
    project_id = project["id"]
    project_path = Path(project["path"])

    register_worker(
        rust_api,
        worker_id,
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["image_generate"]},
    )

    created = httpx.post(
        f"{rust_api}/api/v1/image/jobs",
        json={
            "projectId": project_id,
            "prompt": "mist over rolling hills",
            "model": "z_image_turbo",
            "count": 1,
            "requestedGpu": "gpu-0",
        },
        timeout=5,
    )
    created.raise_for_status()
    job = created.json()
    assert job["type"] == "image_generate"

    claimed = claim_job(rust_api, worker_id)
    assert claimed["id"] == job["id"]
    assert claimed["type"] == "image_generate"
    assert claimed["payload"]["projectId"] == project_id

    complete_image_job(rust_api, worker_id, claimed, project_path=project_path)

    completed = httpx.get(f"{rust_api}/api/v1/jobs/{job['id']}", timeout=5).json()
    assert completed["status"] == "completed"
    assert completed["workerId"] == worker_id
    assets = completed["result"]["assets"]
    assert len(assets) == 1
    asset = assets[0]
    assert asset["type"] == "image"
    assert asset["file"]["mimeType"] == "image/png"
    assert asset["recipe"]["adapter"] == "procedural_preview"
    written = project_path / asset["file"]["path"]
    assert written.exists()
    assert written.suffix == ".png"


def test_character_image_angle_and_pose_sets_carry_loras_through_worker(rust_api, tmp_path):
    """sc-2226: a character_image angle set AND pose set, each carrying a `loras` array,
    submitted through the Rust API binary, must have those loras hydrated + normalized by
    the API and delivered to the worker on the claimed payload (the
    payload -> catalog-normalized -> worker ImageRequest.loras boundary), then survive the
    completion round-trip onto the served asset recipe.

    sc-8861: driven entirely over HTTP — the test plays the worker, so the boundary is
    proven at the two live seams the Rust API owns: (1) `claimed["payload"]["loras"]` is the
    API-normalized lora spec the worker would read into `request.loras`, and (2) the API
    rebuilds `result.assets[].recipe.loras` from the completion facts. z_image_turbo stands
    in as a weightless backbone; the angle/pose LoRA path itself is unit-covered in
    sc-2224/2225. (The Python `_RecordingImageAdapter` that captured `request.loras`
    in-process was retired with apps/worker; the payload assertion below is the same
    boundary the recorder observed.)"""
    # The Rust API resolves a job's project through its own project store (for project
    # LoRAs), so create the project via the API and write asset bytes into its path.
    created_project = httpx.post(f"{rust_api}/api/v1/projects", json={"name": "Lora E2E"}, timeout=5)
    created_project.raise_for_status()
    project = created_project.json()
    project_id = project["id"]
    project_path = Path(project["path"])

    # The rust_api fixture points SCENEWORKS_DATA_DIR + SCENEWORKS_CONFIG_DIR at tmp_path.
    # The Rust API rejects loras not present in the catalog (it hydrates + normalizes
    # submitted specs against installed LoRAs). Seed one user LoRA whose family matches
    # the z_image_turbo backbone so the submission validates and reaches the worker.
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    lora_id = "kelsie-zit"
    lora_dir = data_dir / "loras" / lora_id
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "kelsie.safetensors").write_bytes(_minimal_safetensors())
    manifests = tmp_path / "config" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    # The lora compatibility check resolves the model + builtin loras from the catalog,
    # which reads these manifests; copy the real ones so z_image_turbo (family z-image)
    # is known. Size estimation is disabled by the rust_api fixture (no network).
    for manifest_name in ("builtin.models.jsonc", "builtin.loras.jsonc"):
        shutil.copy(ROOT / "config" / "manifests" / manifest_name, manifests / manifest_name)
    (manifests / "user.loras.jsonc").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "loras": [
                    {
                        "id": lora_id,
                        "name": "Kelsie ZIT",
                        "family": "z-image",
                        "scope": "global",
                        "files": ["kelsie.safetensors"],
                        "source": {"path": f"loras/{lora_id}", "provider": "local", "repo": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    worker_id = "lora-e2e-worker"
    register_worker(
        rust_api,
        worker_id,
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["image_generate"]},
    )

    loras = [{"id": lora_id, "weight": 0.8}]
    pose_keypoints = [[0.5, index / 18] for index in range(18)]
    base = {
        "projectId": project_id,
        "mode": "character_image",
        "model": "z_image_turbo",
        "prompt": "the character",
        "referenceAssetId": "ref-1",
        "count": 1,
        "width": 256,
        "height": 256,
        "requestedGpu": "gpu-0",
        "loras": loras,
    }
    jobs = {
        "angle": {**base, "advanced": {"angleSet": True, "ipAdapterScale": 0.8}},
        "pose": {
            **base,
            "advanced": {"poses": [{"id": "standing_01", "keypoints": pose_keypoints}], "ipAdapterScale": 0.8},
        },
    }

    for kind, body in jobs.items():
        created = httpx.post(f"{rust_api}/api/v1/image/jobs", json=body, timeout=5)
        assert created.status_code == 201, (kind, created.status_code, created.text)
        job = created.json()

        claimed = claim_job(rust_api, worker_id)
        assert claimed["id"] == job["id"], kind
        # The Rust API persisted + served the (normalized) loras on the claimed payload —
        # exactly the spec the worker reads into request.loras (the sc-2226 boundary).
        claimed_loras = claimed["payload"]["loras"]
        assert [lora["id"] for lora in claimed_loras] == [lora_id], kind
        assert claimed_loras[0]["weight"] == 0.8, kind

        # Complete over HTTP, echoing the claimed loras into the asset facts as the worker
        # would, so the API rebuilds recipe.loras and we verify the round-trip.
        complete_image_job(rust_api, worker_id, claimed, project_path=project_path, loras=claimed_loras)

        completed = httpx.get(f"{rust_api}/api/v1/jobs/{job['id']}", timeout=5).json()
        assert completed["status"] == "completed", (kind, completed)
        recipe_loras = completed["result"]["assets"][0]["recipe"]["loras"]
        assert [lora["id"] for lora in recipe_loras] == [lora_id], kind
        assert recipe_loras[0]["weight"] == 0.8, kind


def test_rust_worker_claims_and_completes_lora_import_against_rust_api_binary(rust_api, tmp_path):
    worker_command = _launch_command(
        "SCENEWORKS_RUST_WORKER_BINARY", "sceneworks-rust-worker", "the Rust worker smoke test"
    )

    # The Rust API only imports a sourcePath from app-managed roots (data/loras,
    # project loras, or staged uploads) for path safety; stage the source inside
    # the worker's data/loras dir so the import isn't rejected.
    source = tmp_path / "data" / "loras" / "tiny.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(_minimal_safetensors())
    env = os.environ.copy()
    env.update(
        {
            "SCENEWORKS_API_URL": rust_api,
            "SCENEWORKS_DATA_DIR": str(tmp_path / "data"),
            "SCENEWORKS_CONFIG_DIR": str(tmp_path / "config"),
            "SCENEWORKS_WORKER_ID": "rust-worker-smoke",
            "SCENEWORKS_GPU_ID": "cpu",
            "SCENEWORKS_POLL_SECONDS": "1",
            "SCENEWORKS_HEARTBEAT_SECONDS": "5",
        }
    )
    worker = subprocess.Popen(
        worker_command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        created = httpx.post(
            f"{rust_api}/api/v1/loras/import",
            json={"sourcePath": str(source), "name": "Smoke LoRA"},
            timeout=5,
        )
        created.raise_for_status()
        job = created.json()

        completed = wait_for_job_status(rust_api, job["id"], "completed", worker)

        # The supervisor spawns per-device child workers (e.g. <id>-cpu-2), so the
        # configured WORKER_ID is a prefix of the claiming worker's id.
        assert completed["workerId"].startswith("rust-worker-smoke")
        assert completed["result"]["repo"] is None
        assert completed["result"]["path"].endswith("smoke_lora")
        assert (
            tmp_path / "data" / "loras" / "smoke_lora" / "tiny.safetensors"
        ).read_bytes() == _minimal_safetensors()
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)


def test_rust_worker_completes_ffmpeg_frame_and_timeline_jobs_against_rust_api_binary(rust_api, tmp_path):
    # ffmpeg is intentionally NOT provisioned in the `check.yml` CI (only in the
    # desktop/release packaging workflows), so this stays a plain skip: it must not
    # red the e2e gate. The all-skipped guard in conftest still fires if cargo (and
    # thus every other e2e test) also went missing (sc-8935 / F-133).
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the FFmpeg worker smoke test")

    worker_command = _launch_command(
        "SCENEWORKS_RUST_WORKER_BINARY", "sceneworks-rust-worker", "the Rust worker smoke test"
    )

    env = os.environ.copy()
    env.update(
        {
            "SCENEWORKS_API_URL": rust_api,
            "SCENEWORKS_DATA_DIR": str(tmp_path / "data"),
            "SCENEWORKS_CONFIG_DIR": str(tmp_path / "config"),
            "SCENEWORKS_WORKER_ID": "rust-ffmpeg-smoke",
            "SCENEWORKS_GPU_ID": "cpu",
            "SCENEWORKS_POLL_SECONDS": "1",
            "SCENEWORKS_HEARTBEAT_SECONDS": "5",
        }
    )
    worker = subprocess.Popen(
        worker_command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        created_project = httpx.post(f"{rust_api}/api/v1/projects", json={"name": "FFmpeg Smoke"}, timeout=5)
        created_project.raise_for_status()
        project_id = created_project.json()["id"]

        uploaded = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/assets",
            files={"file": ("source.png", PNG_1X1, "image/png")},
            timeout=5,
        )
        uploaded.raise_for_status()
        asset = uploaded.json()
        asset_id = asset["id"]
        detection_jobs = []
        for index in range(5):
            detection_job = httpx.post(
                f"{rust_api}/api/v1/projects/{project_id}/person-tracks/detections",
                json={"sourceAssetId": asset_id, "sourceTimestamp": index * 0.1},
                timeout=5,
            )
            detection_job.raise_for_status()
            detection_jobs.append(detection_job.json())

        created_timeline = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/timelines",
            json={"name": "Main timeline", "aspectRatio": "16:9", "fps": 24},
            timeout=5,
        )
        created_timeline.raise_for_status()
        timeline = created_timeline.json()
        timeline_id = timeline["id"]
        timeline["tracks"][0]["items"] = [
            {
                "id": "item-1",
                "trackId": "track_main",
                "assetId": asset_id,
                "type": "image",
                "displayName": "Still",
                "sourceIn": 0,
                "sourceOut": 1,
                "timelineStart": 0,
                "timelineEnd": 1,
                "speed": 1,
                "fit": "fit",
                "volume": 1,
            }
        ]
        saved_timeline = httpx.put(
            f"{rust_api}/api/v1/projects/{project_id}/timelines/{timeline_id}",
            json={"timeline": timeline},
            timeout=5,
        )
        saved_timeline.raise_for_status()

        frame_job = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/timelines/{timeline_id}/items/item-1/frames",
            json={"playheadSeconds": 0.5, "intendedUse": "reuse"},
            timeout=5,
        )
        frame_job.raise_for_status()
        export_job = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/timelines/{timeline_id}/exports",
            json={"resolution": 240, "fps": 24, "requestedGpu": "auto"},
            timeout=5,
        )
        export_job.raise_for_status()

        frame_completed = wait_for_job_status(rust_api, frame_job.json()["id"], "completed", worker)
        export_completed = wait_for_job_status(rust_api, export_job.json()["id"], "completed", worker)
        detection_completed = [
            wait_for_job_status(rust_api, job["id"], "completed", worker) for job in detection_jobs
        ]
        first_detection = detection_completed[0]["result"]["detections"][0]
        track_job = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/person-tracks/jobs",
            json={
                "sourceAssetId": asset_id,
                "representativeFrameAssetId": detection_completed[0]["result"]["frameAssetId"],
                "detection": first_detection,
                "trackName": "Hero",
            },
            timeout=5,
        )
        track_job.raise_for_status()
        track_completed = wait_for_job_status(rust_api, track_job.json()["id"], "completed", worker)

        assert frame_completed["workerId"] == "rust-ffmpeg-smoke"
        assert frame_completed["result"]["assets"][0]["type"] == "frame"
        assert frame_completed["result"]["assets"][0]["recipe"]["mode"] == "frame_extract"
        assert export_completed["workerId"] == "rust-ffmpeg-smoke"
        assert export_completed["result"]["assets"][0]["type"] == "render"
        assert export_completed["result"]["assets"][0]["file"]["mimeType"] == "video/mp4"
        assert {job["workerId"] for job in detection_completed} == {"rust-ffmpeg-smoke"}
        assert all(job["result"]["detections"] for job in detection_completed)
        assert track_completed["workerId"] == "rust-ffmpeg-smoke"
        assert track_completed["result"]["track"]["recipe"]["mode"] == "person_track"
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
