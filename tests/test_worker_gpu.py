from __future__ import annotations

from scene_worker.gpu import cpu_worker_id, gpu_worker_id, parse_nvidia_smi_gpus


def test_parse_nvidia_smi_gpus_returns_all_devices():
    gpus = parse_nvidia_smi_gpus(
        "0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887\n"
        "1, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887\n"
    )

    assert [gpu["id"] for gpu in gpus] == ["0", "1"]
    assert gpus[0]["name"] == "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (97887 MB)"
    assert "nvidia" in gpus[1]["capabilities"]
    assert "placeholder" not in gpus[1]["capabilities"]


def test_gpu_worker_id_preserves_existing_first_worker_id():
    assert gpu_worker_id("worker-gpu-auto-0", "0") == "worker-gpu-auto-0"
    assert gpu_worker_id("worker-gpu-auto-0", "1") == "worker-gpu-auto-1"


def test_cpu_worker_id_uses_same_worker_family():
    assert cpu_worker_id("worker-gpu-auto-0") == "worker-gpu-auto-cpu"
