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
        "samples": [
            {
                "phase": "cuda-global-used-upper-bound-sample",
                "cudaDeviceUuid": adapter["cuda_uuid"],
                "globalUsedUpperBoundBytes": 110,
                "childGlobalUsedGrowthBytes": 90,
                "gpuFreeBytes": 95,
                "at_ns": 105,
            }
        ],
        "admission": {
            "gpuGlobalUsedBaselineBytes": 20,
            "cudaDeviceUuid": adapter["cuda_uuid"],
        },
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


def render_scope_guard(guard):
    observation = guard["fit_observation"]
    dedicated = guard["cuda_attribution"]["wddm_dedicated"]
    identity = {
        "pid": guard["pid"],
        "created_filetime": dedicated["created_filetime"],
        "cuda_uuid": dedicated["adapter"]["cuda_uuid"],
    }
    guard["admission"]["childGpuCapBytes"] = 160
    guard["cuda_attribution"]["unknown_pid_polls"] = 2
    guard["samples"][0]["childGlobalUsedGrowthBytes"] = 110
    dedicated["missing_active_polls"] = observation["wddm_shared"]["missing_active_polls"] = 1
    observation["card_samples_monotonic_ns_free_owned_or_fallback_bytes"] = [
        [100, 190, 0],
        [200, 95, 80],
        [300, 145, 0],
    ]
    observation["wddm_retry"] = {
        key: {
            "attempts": 1,
            "events": [
                {
                    "start_ns": 270,
                    "end_ns": 290,
                    "retried": True,
                    "local_filetime_before": 904,
                    "local_filetime_after": 905,
                    "previous": series["samples"][-1],
                    "retry_sample": None,
                    "recovered": False,
                }
            ],
        }
        for key, series in (
            ("dedicated", dedicated),
            ("shared", observation["wddm_shared"]),
        )
    }
    observation["cuda_render_end"] = {
        "pool_ticks": 1,
        "pool_growth": [
            {
                "card_index": 1,
                "read_ns": 185,
                "at_ns": 200,
                "cuda_uuid": identity["cuda_uuid"],
                "free_bytes": 95,
                "global_used_upper_bytes": 120,
                "child_growth_bytes": 103,
            }
        ],
        "pool": dict(
            identity,
            basis="verified-output-pool",
            card_index=1,
            at_ns=200,
            pool_reserved_high_bytes=80,
        ),
        "zero": dict(
            identity,
            basis="owned-process-attributed",
            card_index=2,
            at_ns=300,
            before="live",
            after="exited_empty",
        ),
    }
    return dict(
        identity,
        pool_card_index=1,
        zero_card_index=2,
        dedicated_missing_after_output=1,
        shared_missing_after_output=1,
    )


def with_rejected_reads(guard):
    scope = render_scope_guard(guard)
    end = guard["fit_observation"]["cuda_render_end"]
    uuid = guard["admission"]["cudaDeviceUuid"]
    guard["admission"]["gpuFreeReserveBytes"] = 50
    guard["cuda_attribution"]["unknown_pid_polls"] += 2
    pool = {
        "pid": guard["pid"],
        "cudaDeviceUuid": uuid,
        "poolUsedBytes": 50,
        "poolReservedBytes": 70,
        "poolUsedHighBytes": 75,
        "poolReservedHighBytes": 80,
    }
    retired = {
        "next_card_index": 2,
        "read_ns": 210,
        "at_ns": 220,
        "pid": scope["pid"],
        "created_filetime": scope["created_filetime"],
        "cuda_uuid": uuid,
        "before": "live",
        "after": "live",
        "reason": "probe-failure",
        "detail": "CUDA pool proof exceeds concurrent global use",
        "pool": pool.copy(),
        "free_bytes": 80,
        "global_used_upper_bytes": 60,
        "child_growth_bytes": 40,
    }
    end["rejected_growth"] = [
        retired,
        dict(
            retired,
            read_ns=230,
            at_ns=245,
            after="exited_empty",
            detail="CUDA pool proof has no valid owned PID/UUID allocation",
            pool=pool.copy(),
            global_used_upper_bytes=120,
            child_growth_bytes=117,
        ),
    ]
    end["zero"]["before"] = "exited_empty"
    return scope, pool


@pytest.mark.parametrize("case", ["mlx", "cuda", "cuda_render_scope", "cuda_rejected"])
def test_review_projects_only_the_measured_process_and_accepted_domains(tmp_path, case):
    mlx = case == "mlx"
    backend = "mlx" if mlx else "cuda"
    scope = native_pool = None
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
        if case == "cuda_render_scope":
            scope = render_scope_guard(guard)
        elif case == "cuda_rejected":
            scope, native_pool = with_rejected_reads(guard)
            policy = {
                k: guard["admission"][k]
                for k in ("cudaDeviceUuid", "childGpuCapBytes", "gpuFreeReserveBytes")
            }
            guard["admission"]["policySha256"] = tool["sha256"](
                (json.dumps(policy, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
    guard_name = "count-4.supervisor.json"
    guard_file = tmp_path / guard_name
    guard_file.write_text(json.dumps(guard))
    witness = {
        "axes": axes,
        "text_tokens": 512,
        "gross_host_bytes": 145,
        "gross_unified_bytes": 140,
        "gross_device_bytes": 117 if native_pool else 110 if scope else 120,
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
        **({"cuda_render_scope": scope} if scope else {}),
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
    if native_pool:
        receipt["request"] = {"policy": policy}
        receipt["contract"]["request"] = {"policy": policy}
        pool_file = tmp_path / "count-4.cuda-pool.json"
        pool_file.write_text(json.dumps(native_pool))
        receipt["logs"][pool_file.name] = {"sha256": tool["sha256"](pool_file.read_bytes())}
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
        assert measured["bounds"]["device"]["classified_peak_bytes"] == (
            117 if native_pool else 110 if scope else 100
        )
        assert measured["bounds"]["host"]["classified_peak_bytes"] == 145
    if native_pool:
        original = guard_file.read_bytes()
        guard["admission"]["gpuFreeReserveBytes"] -= 1
        guard_file.write_text(json.dumps(guard))
        receipt["logs"][guard_name]["sha256"] = tool["sha256"](guard_file.read_bytes())
        receipt_file.write_text(json.dumps(receipt))
        review["receipt_sha256"] = tool["sha256"](receipt_file.read_bytes())
        review_file.write_text(json.dumps(review))
        with pytest.raises(ValueError, match="differs from approved capture policy"):
            project()
        guard_file.write_bytes(original)
        receipt["logs"][guard_name]["sha256"] = tool["sha256"](original)
        receipt_file.write_text(json.dumps(receipt))
        review["receipt_sha256"] = tool["sha256"](receipt_file.read_bytes())
        review_file.write_text(json.dumps(review))
        pool_file.write_text("{}")
        with pytest.raises(ValueError, match="native CUDA pool record changed"):
            project()
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
    guard["samples"] = []
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
            guard,
            {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}},
            adapter,
            120,
        )


def test_new_gapless_capture_binds_supervisor_charge_even_when_old_card_baseline_was_larger():
    adapter, guard = windows_guard()
    render_scope_guard(guard)
    guard["cuda_attribution"]["wddm_dedicated"]["missing_active_polls"] = 0
    guard["fit_observation"]["wddm_shared"]["missing_active_polls"] = 0
    native = {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}}
    # Launcher tenants have released their allocation. Static card growth is 100 but
    # the supervisor measured a child charge of 110 on the selected CUDA device.
    assert tool["windows_device_peak"](guard, native, adapter, 110) == 110
    with pytest.raises(ValueError):
        tool["windows_device_peak"](guard, native, adapter, 100)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scope", "cuda_uuid"), "different card"),
        (("scope", "dedicated_missing_after_output"), 0),
        (("scope", "pool_card_index"), True),
        (("scope", "pid"), 999),
        (("end", "pool", "created_filetime"), 999),
        (("end", "zero", "basis"), "global-used-upper-bound"),
        (("end", "zero", "card_index"), 1),
        (("end", "zero", "at_ns"), 201),
        (("end", "zero", "before"), "exited_empty"),
        (("end", "pool", "pool_reserved_high_bytes"), 79),
        (("end", "pool_ticks"), 2),
        (("end", "pool_growth"), []),
        (("end", "pool_growth", 0, "read_ns"), 100),
        (("end", "pool_growth", 0, "read_ns"), 201),
        (("end", "pool_growth", 0, "cuda_uuid"), "another GPU"),
        (("end", "pool_growth", 0, "free_bytes"), 94),
        (("end", "pool_growth", 0, "child_growth_bytes"), 99),
        (("end", "pool_growth", 0, "global_used_upper_bytes"), 119),
        (("guard", "fit_observation", "wddm_shared", "error"), "stat failure"),
        (("guard", "samples", 0, "cudaDeviceUuid"), "different card"),
        (("guard", "samples", 0, "childGlobalUsedGrowthBytes"), 89),
        (("guard", "samples", 0, "childGlobalUsedGrowthBytes"), 111),
        (("guard", "admission", "childGpuCapBytes"), 109),
    ],
)
def test_new_cuda_capture_refuses_changed_evidence(path, value):
    adapter, guard = windows_guard()
    scope = render_scope_guard(guard)
    end = guard["fit_observation"]["cuda_render_end"]
    if path == ("end", "zero", "before"):
        end["zero"]["after"] = "live"
    item = {"scope": scope, "guard": guard, "end": end}
    for k in path[:-1]:
        item = item[k]
    item[path[-1]] = value
    with pytest.raises(ValueError):
        tool["windows_device_peak"](
            guard,
            {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}},
            adapter,
            110,
            scope,
        )


def test_raw_teardown_gaps_without_verified_summary_still_refuse():
    adapter, guard = windows_guard()
    render_scope_guard(guard)
    with pytest.raises(ValueError, match="without verified render scope"):
        tool["windows_device_peak"](
            guard,
            {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}},
            adapter,
            110,
        )


def test_original_system_read_can_dominate_pool_returned_by_the_same_poll():
    # Same physical byte values as the inference producer's CUDA render-end fixture.
    adapter, guard = windows_guard()
    scope = render_scope_guard(guard)
    native = {"allocator_peak": 60, "axes": {"floor_device_bytes": 10}}
    guard["admission"].update(gpuGlobalUsedBaselineBytes=50, childGpuCapBytes=78)
    guard["samples"][0].update(globalUsedUpperBoundBytes=125, childGlobalUsedGrowthBytes=75)
    guard["cuda_attribution"].update(
        global_upper_bytes=125, pid_peak_bytes=0, pool_fallback_peak_bytes=60
    )
    guard["peak_gpu_bytes"] = 60
    guard["cuda_attribution"]["wddm_dedicated"]["samples"] = [[901, 50], [902, 12]]
    raw = guard["fit_observation"]
    raw["wddm_retry"]["dedicated"]["events"][0]["previous"] = [902, 12]
    raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"][1][2] = 60
    end = raw["cuda_render_end"]
    end["pool"]["pool_reserved_high_bytes"] = 60
    end["pool_growth"][0].update(global_used_upper_bytes=125, child_growth_bytes=78)
    peak = tool["windows_device_peak"]
    assert peak(guard, native, adapter, 78, scope) == 78
    with pytest.raises(ValueError, match="differs from verified native quote"):
        peak(guard, native, adapter, 75, scope)
    guard["admission"]["childGpuCapBytes"] = 77
    with pytest.raises(ValueError, match="exceeds guard"):
        peak(guard, native, adapter, 78, scope)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("events", 1, "next_card_index"), 1),
        (("events", 1, "read_ns"), 219),
        (("events", 1, "at_ns"), 301),
        (("events", 1, "pid"), 42),
        (("events", 1, "created_filetime"), 901),
        (("events", 1, "cuda_uuid"), "foreign"),
        (("events", 0, "detail"), "CUDA pool proof is unreadable"),
        (("events", 0, "free_bytes"), 49),
        (("events", 0, "child_growth_bytes"), 70),
        (("events", 0, "reason"), "device-memory"),
        (("events", 1, "pool", "poolUsedBytes"), 51),
        (("events", 1, "after"), "live"),
        (("end", "zero", "before"), "live"),
        (("end", "rejected_growth"), []),
        (("admission", "childGpuCapBytes"), 116),
        (("card", 0, 1), 49),
        (("samples", 0, "gpuFreeBytes"), 49),
        (("attr", "unknown_pid_polls"), 5),
        (("retry", "dedicated", "events", 0, "start_ns"), 228),
    ],
)
def test_refuses_unverified_rejected_pool_attempt(path, value):
    adapter, guard = windows_guard()
    scope, pool = with_rejected_reads(guard)
    obs = guard["fit_observation"]
    end = obs["cuda_render_end"]
    root = {
        "events": end["rejected_growth"],
        "end": end,
        "attr": guard["cuda_attribution"],
        "admission": guard["admission"],
        "retry": obs["wddm_retry"],
        "card": obs["card_samples_monotonic_ns_free_owned_or_fallback_bytes"],
        "samples": guard["samples"],
    }
    parent = root
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        tool["windows_device_peak"](
            guard,
            {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}},
            adapter,
            117,
            scope,
            pool,
        )


def test_gapless_pdh_still_requires_the_native_pool_and_output_markers_for_failed_attempts():
    adapter, guard = windows_guard()
    _, pool = with_rejected_reads(guard)
    obs = guard["fit_observation"]
    for name, series in (
        ("dedicated", guard["cuda_attribution"]["wddm_dedicated"]),
        ("shared", obs["wddm_shared"]),
    ):
        series["missing_active_polls"] = 0
        obs["wddm_retry"][name] = {"attempts": 0, "events": []}
    quote = tool["windows_device_peak"]
    args = (guard, {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}}, adapter)
    assert quote(*args, 117, None, pool) == 117
    with pytest.raises(ValueError):
        quote(*args, 110, None, pool)
    with pytest.raises(ValueError):
        quote(*args, 117, None, None)
    obs["cuda_render_end"]["zero"] = None
    with pytest.raises(ValueError):
        quote(*args, 117, None, pool)
