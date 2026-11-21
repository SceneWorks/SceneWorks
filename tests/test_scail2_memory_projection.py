"""Weights-free tests for importing already verified SCAIL measurement artifacts."""

import json
import runpy
from pathlib import Path

import pytest

tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/project-scail2-memory.py"))


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
    guard_file.write_text("{}")
    with pytest.raises(ValueError, match="guard record changed"):
        project()
    receipt_file.write_text("{}")
    with pytest.raises(ValueError, match="different receipt"):
        project()
