from __future__ import annotations

import os
import re
import subprocess
import sys


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
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index, name, memory_mb = parts[:3]
        gpu = {
            "id": index,
            "name": f"{name} ({memory_mb} MB)",
            "capabilities": ["gpu", "nvidia"],
        }
        if len(parts) >= 6:
            used_mb, free_mb, load_percent = parts[3:6]
            gpu["utilization"] = {
                "memoryTotalMb": parse_int(memory_mb),
                "memoryUsedMb": parse_int(used_mb),
                "memoryFreeMb": parse_int(free_mb),
                "gpuLoadPercent": parse_float(load_percent),
            }
        gpus.append(gpu)
    return gpus


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def query_nvidia_gpus() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
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


def query_mps_gpus() -> list[dict]:
    """Detect an Apple Silicon MPS device, mirroring the nvidia-smi probe shape.

    Returns a single-entry list when torch reports MPS available, otherwise [].
    macOS-only — a no-op (empty) on every other platform (sc-1334).
    """
    if sys.platform != "darwin":
        return []
    try:
        import torch
    except Exception:
        return []
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        available = bool(mps and mps.is_available())
    except Exception:
        available = False
    if not available:
        return []
    gpu = {"id": "mps", "name": "Apple GPU (unified)", "capabilities": ["gpu", "mps"]}
    utilization = query_mps_utilization()
    if utilization:
        gpu["utilization"] = utilization
    return [gpu]


def _sysctl_memsize_mb() -> int | None:
    """Total unified memory (MB) on macOS via `sysctl hw.memsize`."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = parse_int(result.stdout.strip())
    return value // (1024 * 1024) if value is not None else None


def _ioreg_accelerator_stats() -> dict[str, int]:
    """Numeric PerformanceStatistics from the IOAccelerator IOKit node.

    `ioreg` exposes the integrated GPU's live stats (e.g. "Device Utilization %",
    "In use system memory") without elevated privileges, unlike powermetrics.
    Returns the last-seen value per integer key (a Mac has one such accelerator);
    empty when ioreg is unavailable.
    """
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    stats: dict[str, int] = {}
    for key, value in re.findall(r'"([^"]+)"\s*=\s*(\d+)', result.stdout):
        stats[key] = int(value)
    return stats


def query_mps_utilization() -> dict | None:
    """Apple Silicon unified-memory + GPU-load snapshot, shaped like the nvidia path.

    Memory total is the machine's unified RAM; GPU-resident memory and load come
    from IOKit's IOAccelerator node (unprivileged). Returns None when neither
    probe yields anything (non-Apple-Silicon, sandboxed, etc.).
    """
    if sys.platform != "darwin":
        return None
    stats = _ioreg_accelerator_stats()
    total_mb = _sysctl_memsize_mb()
    snapshot: dict = {}
    if total_mb is not None:
        snapshot["memoryTotalMb"] = total_mb
    used_bytes = stats.get("In use system memory") or stats.get("Alloc system memory")
    if used_bytes is not None:
        used_mb = int(used_bytes) // (1024 * 1024)
        snapshot["memoryUsedMb"] = used_mb
        if total_mb is not None:
            snapshot["memoryFreeMb"] = max(0, total_mb - used_mb)
    load = stats.get("Device Utilization %")
    if load is None:
        load = stats.get("GPU Activity(%)")
    if load is not None:
        snapshot["gpuLoadPercent"] = float(load)
    return snapshot or None


def gpu_utilization(gpu_id: str) -> dict | None:
    if gpu_id == "cpu":
        return None
    if gpu_id == "mps":
        return query_mps_utilization()
    for gpu in query_nvidia_gpus():
        if gpu["id"] == gpu_id:
            return gpu.get("utilization")
    return None


def visible_gpu_ids() -> list[str] | None:
    visible_devices = os.getenv("NVIDIA_VISIBLE_DEVICES", "").strip()
    if not visible_devices or visible_devices == "all":
        return None
    if visible_devices in ("void", "none"):
        return []
    return [device.strip() for device in visible_devices.split(",") if device.strip()]


def discover_gpus() -> list[dict]:
    # NVIDIA_VISIBLE_DEVICES is a CUDA/Linux concept; ignore it on macOS, where
    # the only accelerator is the unified-memory MPS device (sc-1334/sc-1335).
    ids = None if sys.platform == "darwin" else visible_gpu_ids()
    if ids == []:
        return []

    gpus = query_nvidia_gpus()
    if ids is not None:
        by_id = {gpu["id"]: gpu for gpu in gpus}
        return [
            by_id.get(gpu_id, {"id": gpu_id, "name": f"GPU {gpu_id}", "capabilities": ["gpu"]})
            for gpu_id in ids
        ]
    if gpus:
        return gpus
    # No NVIDIA GPUs → fall back to MPS on Apple Silicon (empty elsewhere → CPU).
    return query_mps_gpus()


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
