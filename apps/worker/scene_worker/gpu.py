from __future__ import annotations

import os
import re
import subprocess


def gpu_worker_id(base_worker_id: str, gpu_id: str) -> str:
    safe_gpu_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", gpu_id).strip("-") or "gpu"
    if safe_gpu_id == "0" and base_worker_id.endswith("-0"):
        return base_worker_id
    if base_worker_id.endswith("-0") and safe_gpu_id.isdigit():
        return f"{base_worker_id[:-1]}{safe_gpu_id}"
    return f"{base_worker_id}-gpu-{safe_gpu_id}"


def cpu_worker_id(base_worker_id: str) -> str:
    base = base_worker_id[:-2] if base_worker_id.endswith("-0") else base_worker_id
    return f"{base}-cpu"


def parse_nvidia_smi_gpus(output: str) -> list[dict]:
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) != 3:
            continue
        index, name, memory_mb = parts
        gpus.append(
            {
                "id": index,
                "name": f"{name} ({memory_mb} MB)",
                "capabilities": ["gpu", "nvidia"],
            }
        )
    return gpus


def query_nvidia_gpus() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return parse_nvidia_smi_gpus(result.stdout)
    except (OSError, subprocess.SubprocessError):
        return []


def visible_gpu_ids() -> list[str] | None:
    visible_devices = os.getenv("NVIDIA_VISIBLE_DEVICES", "").strip()
    if not visible_devices or visible_devices == "all":
        return None
    if visible_devices in ("void", "none"):
        return []
    return [device.strip() for device in visible_devices.split(",") if device.strip()]


def discover_gpus() -> list[dict]:
    ids = visible_gpu_ids()
    if ids == []:
        return []

    gpus = query_nvidia_gpus()
    if ids is not None:
        by_id = {gpu["id"]: gpu for gpu in gpus}
        return [
            by_id.get(gpu_id, {"id": gpu_id, "name": f"GPU {gpu_id}", "capabilities": ["gpu"]})
            for gpu_id in ids
        ]
    return gpus


def discover_gpu(requested_gpu_id: str) -> dict:
    if requested_gpu_id == "cpu":
        return {
            "id": "cpu",
            "name": "CPU inference worker",
            "capabilities": ["cpu"],
        }

    gpus = discover_gpus()
    if requested_gpu_id and requested_gpu_id != "auto":
        for gpu in gpus:
            if gpu["id"] == requested_gpu_id:
                return gpu
        return {
            "id": requested_gpu_id,
            "name": f"GPU {requested_gpu_id}",
            "capabilities": ["gpu"],
        }

    if gpus:
        return gpus[0]
    return {
        "id": "cpu",
        "name": "CPU inference worker",
        "capabilities": ["cpu"],
    }
