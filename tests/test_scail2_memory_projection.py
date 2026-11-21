"""Weights-free tests for importing already verified SCAIL measurement artifacts."""

import json
import runpy
from pathlib import Path

import pytest

tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/project-scail2-memory.py"))


def windows_guard():
    adapter = {
        "cuda_uuid": "GPU-e4b79931-7be6-f216-460a-f5405cfafffe",
        "counter": r"\GPU Process Memory(*)\Dedicated Usage",
        "luid": "luid_0x00000000_0x0001e266",
        "physical_node": 0,
        "node_mask": 1,
    }
    dedicated = {
        "adapter": adapter,
        "pid": 41,
        "created_filetime": 900,
        "error": None,
        "missing_active_polls": 0,
        "samples": [[901, 60], [902, 93]],
    }
    return adapter, {
        "pid": 41,
        "peak_gpu_bytes": 80,
        "admission": {"gpuGlobalUsedBaselineBytes": 20, "cudaDeviceUuid": adapter["cuda_uuid"]},
        "cuda_attribution": {
            "wddm_dedicated": dedicated,
            "pid_peak_bytes": 77,
            "unknown_pid_polls": 1,
            "global_upper_bytes": 120,
            "pool_fallback_peak_bytes": 80,
        },
        "fit_observation": {
            "wddm_shared": dict(dedicated, samples=[[901, 0], [902, 2]]),
            "wddm_shared_counter": "GPU Process Memory/Shared Usage",
            "card_samples_monotonic_ns_free_owned_or_fallback_bytes": [
                [1, 190, 0],
                [2, 95, 80],
                [3, 140, 40],
            ],
        },
    }


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_review_projects_only_the_measured_process_and_accepted_domains(tmp_path, backend):
    mlx = backend == "mlx"
    os_name = "macos" if mlx else "windows"
    axes = {"backend": backend, "references": 4, "classifier_free_guidance": True}
    revision = "a" * 40
    guard = {
        "returncode": 0,
        "pid": 41,
        "lifetime_peak_host_bytes": 90,
        "peak_host_bytes": 95,
        "admission": {
            "childGpuCapBytes": 160,
            "childFootprintCapBytes": 200,
            "policySha256": "b" * 64,
        },
    }
    if not mlx:
        adapter, cuda = windows_guard()
        guard["admission"].update(cuda.pop("admission"))
        guard.update(cuda)
        axes["floor_device_bytes"] = 70
    guard_name = "count-4.supervisor.json"
    guard_file = tmp_path / guard_name
    guard_file.write_text(json.dumps(guard))
    witness = {
        "axes": axes,
        "text_tokens": 512,
        "gross_host_bytes": 145,
        "gross_unified_bytes": 140,
        "gross_device_bytes": 120,
        "mlx_decode_budget_bytes": 16 if mlx else None,
    }
    native = {
        "pid": 41,
        "axes": axes,
        "prompt_tokens": [512, 512],
        "allocator_peak": 80,
    }
    row = {
        "platform": os_name,
        "device": "board",
        "runtime": "version",
        "inference_revision": revision,
        "witness": witness,
    }
    receipt = {
        "accepted": True,
        "run_id": "123",
        "run_attempt": "2",
        "contract": {"protocol": "scail2-plural-fit-v1", "head": revision},
        "counts": {
            "4": {
                "accepted": True,
                "claim": {"post_execution_verified": True},
                "native": native,
                "candidate_row_pending_physical_review": row,
            }
        },
        "adapter": None if mlx else adapter,
        "hardware": {
            "chip": "board",
            "name": "board",
            "os_build": "version",
            "driver_version": "version",
        },
        "logs": {guard_name: {"sha256": tool["sha256"](guard_file.read_bytes())}},
    }
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(receipt))
    empirical = {
        "basis": "measured_aggregate",
        "co_live_extra_bytes": 5,
        "uncertainty_allowance_bytes": 8,
    }
    review = {
        "artifact_id": 314,
        "run_id": 123,
        "run_attempt": 2,
        "references": 4,
        "rationale": "fixture",
        "receipt_sha256": tool["sha256"](receipt_file.read_bytes()),
        "domains": {"unified": empirical}
        if mlx
        else {
            "host": empirical,
            "device": {
                "basis": "guarded_ceiling",
                "unsampled_overshoot_upper_bytes": 15,
            },
        },
    }
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps(review))

    def project():
        return tool["project"](receipt_file, review_file)

    result = project()
    pool = result["bounds"]["unified" if mlx else "host"]
    assert (pool["observed_peak_bytes"], pool["classified_peak_bytes"]) == (
        (90, 140) if mlx else (95, 145)
    )
    assert pool["review_sha256"] == tool["sha256"](review_file.read_bytes())
    assert result["axes"] == axes
    if not mlx:
        assert result["bounds"]["device"]["guard_ceiling_bytes"] == 160
        review["domains"]["device"] = empirical
        review_file.write_text(json.dumps(review))
        measured = project()
        assert measured["bounds"]["device"]["observed_peak_bytes"] == 80
        assert measured["bounds"]["device"]["classified_peak_bytes"] == 100
        assert measured["bounds"]["host"]["classified_peak_bytes"] == 145
    guard_file.write_text("{}")
    with pytest.raises(ValueError, match="guard record changed"):
        project()
    receipt_file.write_text("{}")
    with pytest.raises(ValueError, match="different receipt"):
        project()


def test_windows_device_peak_deducts_each_card_baseline_only_from_its_own_series():
    peak = tool["windows_device_peak"]
    adapter, guard = windows_guard()
    native = {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}}
    assert peak(guard, native, adapter, 120) == 100
    guard["admission"]["gpuGlobalUsedBaselineBytes"] = 60
    assert peak(guard, native, adapter, 120) == 95
    free = guard["fit_observation"]["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
    free[1][1] = 145
    assert peak(guard, native, adapter, 120) == 93
    # A mixture of owned and unknown ticks may still have a higher attributable PID peak.
    attrs = guard["cuda_attribution"]
    attrs["pid_peak_bytes"] = guard["peak_gpu_bytes"] = 103
    assert peak(guard, native, adapter, 120) == 103
    attrs.update(unknown_pid_polls=0, global_upper_bytes=0, pool_fallback_peak_bytes=0)
    assert peak(guard, native, adapter, 103) == 103
    native["axes"]["floor_device_bytes"] = 110
    assert peak(guard, native, adapter, 110) == 110
    native["allocator_peak"] = 115
    assert peak(guard, native, adapter, 115) == 115


CARD = ("fit_observation", "card_samples_monotonic_ns_free_owned_or_fallback_bytes")
DEDICATED = ("cuda_attribution", "wddm_dedicated")
BASELINE = ("admission", "gpuGlobalUsedBaselineBytes")
MISSING = object()


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (BASELINE, MISSING),
        (BASELINE, -1),
        ((*CARD, 1, 0), 1),
        (CARD, []),
        ((*CARD, 2, 1), None),
        ((*DEDICATED, "pid"), 42),
        ((*DEDICATED, "adapter"), {}),
        ((*DEDICATED, "missing_active_polls"), 1),
        ((*DEDICATED, "error"), "owned process identity is unavailable"),
        ((*DEDICATED, "samples", 1, 0), 901),
        (("fit_observation", "wddm_shared", "created_filetime"), 901),
        ((*DEDICATED, "samples", 0, 1), 0),
        (("cuda_attribution", "global_upper_bytes"), 0),
    ],
)
def test_windows_device_peak_refuses_incomplete_or_crossed_attribution(path, invalid):
    adapter, guard = windows_guard()
    parent = guard
    for key in path[:-1]:
        parent = parent[key]
    if invalid is MISSING:
        del parent[path[-1]]
    else:
        parent[path[-1]] = invalid
    with pytest.raises(ValueError):
        tool["windows_device_peak"](
            guard, {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}}, adapter, 120
        )
