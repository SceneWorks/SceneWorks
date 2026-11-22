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


@pytest.mark.parametrize("case", ["mlx", "cuda", "cuda_gpu0", "cuda_render_scope", "cuda_rejected"])
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
        if case == "cuda_gpu0":
            uuid = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
            adapter["cuda_uuid"] = uuid
            cuda["admission"]["cudaDeviceUuid"] = uuid
            cuda["samples"][0]["cudaDeviceUuid"] = uuid
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
            "uuid": None if mlx else adapter["cuda_uuid"],
        },
        "logs": {guard_name: {"sha256": tool["sha256"](guard_file.read_bytes())}},
    }
    if case == "cuda_gpu0":
        receipt["hardware"].update(index="0", **{"pci.bus_id": "00000000:21:00.0"})
        native["cuda_device"] = {
            "ordinal": 0,
            "visible_count": 1,
            "uuid": adapter["cuda_uuid"],
            "pci_bus_id": "00000000:21:00.0",
            "luid": adapter["luid"],
            "node_mask": 1,
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
    if case == "cuda_gpu0":
        for key, wrong in (
            ("ordinal", 1),
            ("visible_count", 2),
            ("luid", "luid_0x00000000_0x00000001"),
            ("uuid", "other"),
            ("pci_bus_id", "00000000:C1:00.0"),
            ("node_mask", 2),
        ):
            actual = native["cuda_device"][key]
            native["cuda_device"][key] = wrong
            receipt_file.write_text(json.dumps(receipt))
            review["receipt_sha256"] = tool["sha256"](receipt_file.read_bytes())
            review_file.write_text(json.dumps(review))
            with pytest.raises(ValueError, match="GPU0 capture lacks"):
                project()
            native["cuda_device"][key] = actual
        receipt_file.write_text(json.dumps(receipt))
        review["receipt_sha256"] = tool["sha256"](receipt_file.read_bytes())
        review_file.write_text(json.dumps(review))
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


def completed_job_fixture(*, tied=False):
    adapter, guard = windows_guard()
    uuid = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
    adapter["cuda_uuid"] = guard["admission"]["cudaDeviceUuid"] = guard["samples"][0][
        "cudaDeviceUuid"
    ] = uuid
    scope, pool = with_rejected_reads(guard)
    observation = guard["fit_observation"]
    end = observation["cuda_render_end"]
    guard["returncode"] = 0
    guard["admission"]["pollMillis"] = 250
    completion = {
        "scope": "supervisor-owned-windows-job",
        "root_pid": 41,
        "root_returncode": 0,
        "waited_returncode": 0,
        "job_member_pids": [],
        "owned_pids": [],
    }
    guard["owned_completion"] = completion.copy()
    claim = {
        "physical_index": 0,
        "device_uuid": uuid,
        "lease_released": True,
        "post_execution_verified": True,
        "release_metadata_verification": {"status": "verified"},
        "owned_execution_completion": completion.copy(),
        "release_background_activity": {
            "samples": [
                {
                    "selected_index_matches_admission": True,
                    "post_owned_exit_primary_metadata_valid": True,
                    "selected_card": {"index": 0, "used_mib": 0, "utilization_percent": 0},
                }
                for _ in range(5)
            ]
        },
    }
    if tied:
        set_tied_read(guard, scope)
        return guard, claim, scope, pool
    del scope["zero_card_index"]
    scope.update(basis="owned-job-completion", terminal_rejected_attempts=4)
    end["zero"] = None
    observation["card_samples_monotonic_ns_free_owned_or_fallback_bytes"].pop()
    end["rejected_growth"] = [
        dict(
            end["rejected_growth"][0],
            read_ns=250 + i * 10,
            at_ns=259 + i * 10,
            child_growth_bytes=40 - i,
            global_used_upper_bytes=60 - i,
        )
        for i in range(4)
    ]
    guard["cuda_attribution"]["unknown_pid_polls"] = 6
    for i, (name, series) in enumerate(
        (
            ("dedicated", guard["cuda_attribution"]["wddm_dedicated"]),
            ("shared", observation["wddm_shared"]),
        )
    ):
        series["samples"][:0] = [[898 + j, 0] for j in range(3)]
        event = observation["wddm_retry"][name]["events"][0]
        event.update(
            start_ns=281 + 3 * i,
            end_ns=283 + 3 * i,
            local_filetime_before=904 + 2 * i,
            local_filetime_after=905 + 2 * i,
        )
    return (guard, claim, scope, pool)


def completed_peak(guard, claim, scope, pool):
    native = {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}}
    return tool["windows_device_peak"](
        guard,
        native,
        guard["cuda_attribution"]["wddm_dedicated"]["adapter"],
        110,
        native_pool=pool,
        completion=(claim, scope, scope["basis"]),
    )


def test_completed_job_proves_output_interval_with_all_raw_tail_readings():
    guard, claim, scope, pool = completed_job_fixture()
    assert completed_peak(guard, claim, scope, pool) == 110
    assert guard["fit_observation"]["cuda_render_end"]["zero"] is None


@pytest.mark.parametrize(
    "change",
    [
        "release",
        "owned",
        "cap",
        "reserve",
        "retry",
        "read",
        "reentry",
        "gap",
        "marker",
        "missing-counter",
        "gauge-order",
        "reversed-filetime",
        "overlong-filetime",
        "untyped-clock",
    ],
)
def test_completed_job_fails_on_unproven_terminal_evidence(change):
    guard, claim, scope, pool = completed_job_fixture()
    end = guard["fit_observation"]["cuda_render_end"]
    if change == "release":
        claim["release_background_activity"]["samples"][0]["selected_card"]["used_mib"] = 1
    elif change == "owned":
        claim["owned_execution_completion"]["job_member_pids"] = [41]
    elif change == "cap":
        guard["samples"][0]["childGlobalUsedGrowthBytes"] = 161
    elif change == "reserve":
        end["pool_growth"][0]["free_bytes"] = 1
    elif change == "retry":
        end["rejected_growth"].pop()
    elif change == "read":
        end["rejected_growth"][2]["child_growth_bytes"] += 2
    elif change == "reentry":
        end["rejected_growth"][0]["after"] = "exited_empty"
    elif change == "gap":
        guard["fit_observation"]["wddm_retry"]["dedicated"]["events"][0]["start_ns"] = 235
    elif change == "missing-counter":
        guard["fit_observation"]["wddm_shared"]["samples"].pop(0)
    elif change in {"reversed-filetime", "overlong-filetime", "untyped-clock"}:
        gap = guard["fit_observation"]["wddm_retry"]["shared"]["events"][0]
        gap["local_filetime_after"] = {
            "reversed-filetime": 0,
            "overlong-filetime": 3_000_000,
            "untyped-clock": 907.0,
        }[change]
    elif change == "gauge-order":
        guard["fit_observation"]["wddm_retry"]["shared"]["events"][0]["start_ns"] = 281
    else:
        scope["pool_card_index"] += 1
    with pytest.raises(ValueError):
        completed_peak(guard, claim, scope, pool)


def completed_capture_fixture(tmp_path, *, tied=False):
    guard, claim, summary, pool = completed_job_fixture(tied=tied)
    base = tool["RECOVERED_SIX"].copy()

    def write(name, data):
        path = tmp_path / name
        path.write_bytes((json.dumps(data, sort_keys=True) + "\n").encode())
        return path

    guard["peak_host_bytes"] = 95
    guard["fit_observation"]["windows_peak"] = {
        "peak_job_commit_bytes": 110,
        "peak_private_commit_bytes": 90,
        "peak_working_set_bytes": 100,
    }
    policy = {
        k: guard["admission"][k]
        for k in ("cudaDeviceUuid", "childGpuCapBytes", "gpuFreeReserveBytes", "pollMillis")
    }
    guard["admission"].update(
        childFootprintCapBytes=200,
        policySha256=tool["sha256"](
            (json.dumps(policy, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ),
    )
    adapter = guard["cuda_attribution"]["wddm_dedicated"]["adapter"]
    axes = {
        "backend": "cuda",
        "references": 6,
        "classifier_free_guidance": True,
        "floor_device_bytes": 70,
        "floor_host_bytes": 60,
    }
    native = {
        "axes": axes,
        "pid": 41,
        "allocator_peak": 80,
        "prompt_tokens": [512, 512],
        "cuda_device": {
            "ordinal": 0,
            "visible_count": 1,
            "uuid": adapter["cuda_uuid"],
            "pci_bus_id": "00000000:21:00.0",
            "luid": adapter["luid"],
            "node_mask": 1,
        },
    }
    row = {
        "platform": "windows",
        "device": "board",
        "runtime": "runtime",
        "inference_revision": base["inference_revision"],
        ("cuda_render_scope" if tied else "cuda_completed_job"): summary,
        "witness": {
            "axes": axes,
            "text_tokens": 512,
            "gross_device_bytes": 110,
            "gross_host_bytes": 112,
            "gross_unified_bytes": 0,
            "mlx_decode_budget_bytes": None,
        },
    }
    record = {
        "accepted": True,
        "native": native,
        "claim": claim,
        "candidate_row_pending_physical_review": row,
    }
    capture = {
        "accepted": True,
        "run_id": base["run_id"],
        "run_attempt": "1",
        "contract": {
            "protocol": "scail2-plural-fit-v1",
            "head": base["inference_revision"],
            "request": {"policy": policy},
        },
        "counts": {"6": record},
        "request": {"policy": policy},
        "adapter": adapter,
        "hardware": {
            "index": "0",
            "pci.bus_id": "00000000:21:00.0",
            "uuid": adapter["cuda_uuid"],
            "name": "board",
            "driver_version": "runtime",
        },
        "logs": {},
    }
    for name, data in (("count-6.cuda-pool.json", pool), ("count-6.supervisor.json", guard)):
        capture["logs"][name] = {"sha256": tool["sha256"](write(name, data).read_bytes())}
    receipt = write("receipt.json", capture)
    empirical = {
        "basis": "measured_aggregate",
        "co_live_extra_bytes": 0,
        "uncertainty_allowance_bytes": 8,
    }
    review = {
        "artifact_id": 333,
        "run_id": int(base["run_id"]),
        "run_attempt": 1,
        "receipt_sha256": tool["sha256"](receipt.read_bytes()),
        "references": 6,
        "rationale": "test",
        "domains": {"host": empirical, "device": empirical},
    }
    return write, guard, claim, summary, base, row, record, capture, receipt, review


def test_fresh_job_and_independently_recovered_six_keep_their_original_receipts(
    tmp_path, monkeypatch
):
    write, guard, claim, summary, base, row, record, capture, receipt, review = (
        completed_capture_fixture(tmp_path)
    )
    result = tool["project"](receipt, write("review.json", review))
    assert result["accepted"] is True and "recovery_certificate_sha256" not in result
    assert result["bounds"]["device_interval"] == "load_through_completed_rgb"

    del record["candidate_row_pending_physical_review"]
    capture["accepted"] = record["accepted"] = False
    del claim["owned_execution_completion"]
    del guard["owned_completion"]
    capture["logs"]["count-6.supervisor.json"]["sha256"] = tool["sha256"](
        write("count-6.supervisor.json", guard).read_bytes()
    )
    receipt = write("receipt.json", capture)
    review["receipt_sha256"] = base["receipt_sha256"] = tool["sha256"](receipt.read_bytes())
    review["artifact_id"] = int(base["artifact_id"])
    summary["basis"] = "archived-supervisor-return"
    archive = write("synthetic.archive", "sealed test input")
    base["archive_sha256"] = tool["sha256"](archive.read_bytes())
    monkeypatch.setitem(tool["recovered_row"].__globals__, "RECOVERED_SIX", base)
    cert = write(
        "certificate.json",
        {
            "schema": "scail2-quiet-gpu0-six-recovery-v1",
            "capture": base,
            "references": 6,
            "candidate_row_pending_physical_review": row,
        },
    )
    review["recovery_certificate_sha256"] = tool["sha256"](cert.read_bytes())
    monkeypatch.setitem(
        tool["recovered_row"].__globals__,
        "RECOVERY_CERTIFICATE_SHA256",
        review["recovery_certificate_sha256"],
    )
    reviewed = write("review.json", review)
    with pytest.raises(ValueError, match="incomplete"):
        tool["project"](receipt, reviewed)

    def recover():
        return tool["project"](receipt, reviewed, certificate_path=cert, archive_path=archive)

    recovered = recover()
    assert recovered["accepted"] is False
    assert recovered["receipt_sha256"] == base["receipt_sha256"]
    assert recovered["recovery_certificate_sha256"] == review["recovery_certificate_sha256"]
    assert recovered["bounds"]["device"]["classified_peak_bytes"] == 110
    original = archive.read_bytes()
    archive.write_bytes(b"another archive")
    with pytest.raises(ValueError, match="certificate"):
        recover()
    archive.write_bytes(original)
    cert.write_bytes(cert.read_bytes() + b" ")
    with pytest.raises(ValueError, match="certificate"):
        recover()


def set_tied_read(guard, scope):
    o = guard["fit_observation"]
    end = o["cuda_render_end"]
    guard["admission"]["pollMillis"] = 250
    last = end["rejected_growth"][-1]
    last.update(
        detail="CUDA pool proof exceeds concurrent global use",
        global_used_upper_bytes=30,
        child_growth_bytes=18,
    )
    scope.update(terminal_rejected_read_ns=230, terminal_rejected_at_ns=245)
    d, s = [o["wddm_retry"][g]["events"][0] for g in ("dedicated", "shared")]
    d.update(start_ns=230, end_ns=235)
    s.update(start_ns=236, end_ns=245, local_filetime_before=905)
    return end, last, d, s


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "strict_old_scope",
        "certificate_time",
        "early_dedicated",
        "late_dedicated",
        "early_shared_end",
        "crossed_filetime",
        "reversed_filetime",
        "earlier_reader",
        "missing_retry",
        "wrong_zero",
        "missing_pool_sample",
        "growing_rejected_charge",
    ],
)
def test_owned_zero_requires_both_gauge_gaps_in_the_same_final_cuda_read(changed):
    adapter, guard = windows_guard()
    scope, pool = with_rejected_reads(guard)
    end, last, d, s = set_tied_read(guard, scope)
    if changed == "certificate_time":
        scope["terminal_rejected_read_ns"] += 1
    elif changed == "early_dedicated":
        d["start_ns"] -= 1
    elif changed == "late_dedicated":
        d["start_ns"] += 1
    elif changed == "early_shared_end":
        s["end_ns"] -= 1
    elif changed == "crossed_filetime":
        s["local_filetime_before"] -= 1
    elif changed == "reversed_filetime":
        d["local_filetime_after"] = 903
    elif changed == "earlier_reader":
        end["rejected_growth"].pop(0)
        guard["cuda_attribution"]["unknown_pid_polls"] -= 1
    elif changed == "missing_retry":
        d["retried"] = False
    elif changed == "wrong_zero":
        end["zero"]["before"] = "live"
    elif changed == "missing_pool_sample":
        end["pool_growth"].clear()
    elif changed == "growing_rejected_charge":
        last["child_growth_bytes"] = last["global_used_upper_bytes"] = 41
    elif changed == "strict_old_scope":
        del scope["terminal_rejected_at_ns"], scope["terminal_rejected_read_ns"]

    def peak():
        return tool["windows_device_peak"](
            guard,
            {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}},
            adapter,
            110,
            scope,
            pool,
            tied_read=changed != "strict_old_scope",
        )

    if changed is None:
        assert peak() == 110
    else:
        with pytest.raises(ValueError):
            peak()


def test_partial_archive_keeps_original_verdict_and_binds_each_certificate(tmp_path, monkeypatch):
    receipt = {"accepted": False, "contract": {"head": "a" * 40}, "counts": {}}
    archive, proofs = tmp_path / "archive.zip", tmp_path / "certificates.json"
    archive.write_bytes(b"original archive")

    def compact(v):
        return json.dumps(v, sort_keys=True, separators=(",", ":")).encode()

    identity = dict(
        tool["FOUR_COUNT_CAPTURE"],
        inference_revision=receipt["contract"]["head"],
        plan_sha256=tool["sha256"](compact(receipt["contract"])),
        archive_sha256=tool["sha256"](archive.read_bytes()),
    )
    namespace = tool["partial_count_row"].__globals__
    monkeypatch.setitem(namespace, "FOUR_COUNT_CAPTURE", identity)
    certificates, hashes = {}, {}
    for count in range(2, 6):
        row = {"measured": count}
        record = {"accepted": count in (2, 4)}
        if record["accepted"]:
            record["candidate_row_pending_physical_review"] = row
        else:
            record["telemetry_failure"] = "incomplete_wddm_samples"
            cert = {
                "schema": "scail2-quiet-gpu0-owned-zero-recovery-v1",
                "capture": identity,
                "references": count,
                "candidate_row_pending_physical_review": row,
            }
            certificates[str(count)] = cert
            hashes[count] = tool["sha256"](compact(cert) + b"\n")
        receipt["counts"][str(count)] = record
    monkeypatch.setitem(namespace, "FOUR_COUNT_CERTIFICATES", hashes)
    proofs.write_bytes(compact(certificates))
    for count in range(2, 6):
        review = {
            k: identity[k] for k in ("artifact_id", "receipt_sha256", "run_id", "run_attempt")
        }
        review["references"] = count
        source = None
        if count in hashes:
            review["recovery_certificate_sha256"] = hashes[count]
            source = proofs
        read = tool["partial_count_row"]
        assert read(receipt, review, archive, source) == {"measured": count}
        for name in ("artifact_id", "receipt_sha256", "run_id", "run_attempt"):
            with pytest.raises(ValueError):
                read(receipt, dict(review, **{name: "other"}), archive, source)
        if source:
            with pytest.raises(ValueError):
                read(receipt, review, archive, None)
            with pytest.raises(ValueError):
                read(
                    receipt,
                    dict(review, recovery_certificate_sha256=hashes[8 - count]),
                    archive,
                    proofs,
                )
        else:
            with pytest.raises(ValueError):
                read(receipt, review, archive, proofs)
        record = receipt["counts"][str(count)]
        record["accepted"] = not record["accepted"]
        with pytest.raises(ValueError):
            read(receipt, review, archive, source)
        record["accepted"] = not record["accepted"]


@pytest.mark.parametrize(
    "tied", [True, False], ids=["sampled-owned-zero", "no-zero-owned-completion"]
)
def test_ordinary_full_gpu0_captures_bind_the_hosted_plan_and_real_job(tmp_path, tied):
    write, guard, claim, _, _, _, record, capture, _, review = completed_capture_fixture(
        tmp_path, tied=tied
    )
    digest = tool["sha256"]

    def plan_digest(data):
        return digest(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())

    policy = {
        "selection": "scail2-plural-fit",
        "policy": {
            "schemaVersion": 1,
            "deadlineSeconds": 2400,
            "pollMillis": 250,
            "termGraceMillis": 2000,
            "stdoutCapBytes": 4194304,
            "stderrCapBytes": 4194304,
            "eventCapBytes": 16777216,
            "backend": "windows-cuda",
            "childFootprintCapBytes": 64 << 30,
            "hostFreeReserveBytes": 8 << 30,
            "cudaDeviceUuid": claim["device_uuid"],
            "childGpuCapBytes": 84 << 30,
            "gpuFreeReserveBytes": 6 << 30,
        },
    }
    contract = capture["contract"]
    capture["request"] = contract["request"] = policy
    contract.update(
        backend="candle-cuda",
        counts=[6, 2, 3, 4, 5],
        cuda_completion="supervisor-owned-windows-job",
    )
    review["hosted_plan_sha256"] = plan_digest(contract)
    for n in (2, 3, 4, 5):
        capture["counts"][str(n)] = {"accepted": True, "candidate_row_pending_physical_review": {}}
    admission = guard["admission"]
    admission.update(
        policy["policy"],
        originalClaimDeadlineMonotonic=500,
        policySha256=digest(
            (json.dumps(policy["policy"], sort_keys=True, separators=(",", ":")) + "\n").encode()
        ),
    )
    free_reserve = 6 << 30
    obs = guard["fit_observation"]
    for value in obs["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]:
        value[1] += free_reserve
    for event in guard["samples"]:
        event["gpuFreeBytes"] += free_reserve
    end = obs["cuda_render_end"]
    for event in end["pool_growth"] + end["rejected_growth"]:
        event["free_bytes"] += free_reserve
    claim["claim_deadline_monotonic"] = 500
    record["native"].update(
        count=6,
        backend="candle-cuda",
        output={"width": 832, "height": 480, "frames": 45, "fps": 16, "sha256": "1" * 64},
    )
    record["body"] = {
        "actual_devices": ["candle-cuda"],
        "body_completed": 1,
        "successful_libtest_summaries": 1,
    }

    def project():
        capture["logs"]["count-6.supervisor.json"]["sha256"] = digest(
            write("count-6.supervisor.json", guard).read_bytes()
        )
        receipt = write("receipt.json", capture)
        review["receipt_sha256"] = digest(receipt.read_bytes())
        return tool["project"](receipt, write("review.json", review))

    result = project()
    assert result["accepted"] is True and "recovery_certificate_sha256" not in result
    assert result["bounds"]["device_interval"] == "load_through_completed_rgb"
    assert result["bounds"]["device"]["classified_peak_bytes"] == 110
    for target, field, invalid in [
        (review, "hosted_plan_sha256", "0" * 64),
        (capture["counts"]["4"], "accepted", False),
        (claim, "claim_deadline_monotonic", 501),
        (claim["owned_execution_completion"], "owned_pids", [41]),
        (admission, "childFootprintCapBytes", 1),
        (guard["samples"][0], "gpuFreeBytes", free_reserve - 1),
        (record["native"]["output"], "frames", 44),
        (review["domains"]["device"], "basis", "guarded_ceiling"),
    ]:
        original = target[field]
        target[field] = invalid
        with pytest.raises(ValueError):
            project()
        target[field] = original
    contract["counts"] = [2, 3, 4, 5]
    review["hosted_plan_sha256"] = plan_digest(contract)
    with pytest.raises(ValueError, match="complete accepted hosted"):
        project()


@pytest.mark.parametrize("with_rejections", [False, True])
def test_fresh_strict_post_output_gaps_bound_both_clocks(with_rejections):
    adapter, guard = windows_guard()
    if with_rejections:
        scope, pool = with_rejected_reads(guard)
        gross = 117
    else:
        scope, pool = render_scope_guard(guard), None
        gross = 110
    guard["admission"]["pollMillis"] = 250

    def quote():
        return tool["windows_device_peak"](
            guard,
            {"allocator_peak": 80, "axes": {"floor_device_bytes": 70}},
            adapter,
            gross,
            scope,
            pool,
            fresh_gpu0=True,
        )

    assert quote() == gross
    for gauge in ("dedicated", "shared"):
        event = guard["fit_observation"]["wddm_retry"][gauge]["events"][0]
        for field, wrong in [
            ("start_ns", 0),
            ("end_ns", event["start_ns"] - 1),
            ("local_filetime_after", 903),
            ("local_filetime_after", 3_000_000),
            ("local_filetime_after", True),
        ]:
            old = event[field]
            event[field] = wrong
            with pytest.raises(ValueError):
                quote()
            event[field] = old
