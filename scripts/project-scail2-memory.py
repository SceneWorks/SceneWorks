"""Print one compact worker row from a verified SCAIL physical capture and its review.

Run after the inference producer's --verify-artifacts has passed. The review JSON
names the immutable artifact_id, run_id, run_attempt, receipt_sha256, references, rationale, and
one entry per domain in "domains". Each entry supplies a "basis" and only its
reviewed extra/uncertainty or overshoot bytes. Capture facts come from the files.

    python scripts/project-scail2-memory.py path/to/receipt.json physical-review.json

For the archived GPU0 six, supply --recovery-archive and --recovery-certificate.
The independent review must name the hash of the native verifier's certificate.
The original campaign acceptance remains false. Device observations end at the
completed RGB output. The separate Windows Job counters cover host lifetime.
For the four-count GPU0 continuation, supply the original --recovery-archive for
each count. Counts 3 and 5 also require the native owned-zero certificate map.
The review names the SHA-256 of the count's sorted compact JSON certificate plus
one newline. Each projection retains the original campaign and count verdicts.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def byte_count(value):
    require(type(value) is int and 0 <= value < 2**64, "invalid byte count")
    return value


def envelope(review, observed, classified, guard_ceiling, review_digest):
    observed = byte_count(observed)
    classified = byte_count(classified)
    require(
        0 < observed <= classified,
        "physical observation exceeds classified measurement",
    )
    if review["basis"] == "measured_aggregate":
        require(
            set(review) == {"basis", "co_live_extra_bytes", "uncertainty_allowance_bytes"},
            "unexpected empirical price field",
        )
        byte_count(
            classified
            + byte_count(review["co_live_extra_bytes"])
            + byte_count(review["uncertainty_allowance_bytes"])
        )
        return dict(
            review,
            observed_peak_bytes=observed,
            classified_peak_bytes=classified,
            review_sha256=review_digest,
        )
    require(
        review["basis"] == "guarded_ceiling"
        and set(review) == {"basis", "unsampled_overshoot_upper_bytes"},
        "unknown physical pricing basis",
    )
    ceiling = byte_count(guard_ceiling)
    upper = byte_count(ceiling + byte_count(review["unsampled_overshoot_upper_bytes"]))
    require(
        observed <= ceiling and upper >= classified,
        "guard cannot cover the verified measurement",
    )
    return dict(review, observed_peak_bytes=observed, guard_ceiling_bytes=ceiling)


def sample_values(samples, width, label):
    require(isinstance(samples, list) and len(samples) >= 2, f"incomplete {label} timeline")
    last = 0
    values = []
    for sample in samples:
        require(
            isinstance(sample, list)
            and len(sample) == width
            and type(sample[0]) is int
            and last < sample[0],
            f"invalid {label} timeline",
        )
        last = sample[0]
        value, *_ = (byte_count(cell) for cell in sample[1:])
        values.append(value)
    return values


def tenant_growth(guard, uuid, absolute, unknown, baseline, card, pool):
    """Read the child charge from the same supervisor sample as its whole-card upper bound."""
    samples = guard.get("samples")
    require(isinstance(samples, list), "missing supervisor CUDA samples")
    events = [
        event for event in samples if event.get("phase") == "cuda-global-used-upper-bound-sample"
    ]
    require(len(events) <= unknown, "unexpected supervisor CUDA sample count")

    def charge_values(charge, upper):
        charge, upper = byte_count(charge), byte_count(upper)
        require(
            max(0, upper - baseline) <= charge <= upper <= absolute,
            "inconsistent supervisor CUDA charge",
        )
        return charge, upper

    peak = absolute_peak = last = 0
    for event in events:
        charge, upper = charge_values(
            event.get("childGlobalUsedGrowthBytes"), event.get("globalUsedUpperBoundBytes")
        )
        require(
            event.get("cudaDeviceUuid") == uuid
            and type(event.get("at_ns")) is int
            and event["at_ns"] > last,
            "inconsistent supervisor CUDA charge or device",
        )
        byte_count(event.get("gpuFreeBytes"))
        peak = max(peak, charge)
        absolute_peak = max(absolute_peak, upper)
        last = event["at_ns"]

    observation = guard["fit_observation"]
    if "cuda_render_end" in observation:
        end = observation["cuda_render_end"]
        require(isinstance(end, dict), "missing CUDA render-end measurements")
        entries = end.get("pool_growth")
        require(
            isinstance(entries, list) and len(entries) == byte_count(end.get("pool_ticks")),
            "missing original charge at CUDA pool substitution",
        )
        previous = -1
        for entry in entries:
            index = byte_count(entry["card_index"])
            require(
                previous < index < len(card)
                and entry["cuda_uuid"] == uuid
                and (card[index - 1][0] if index else 0)
                < byte_count(entry["read_ns"])
                <= byte_count(entry["at_ns"])
                == card[index][0]
                and byte_count(entry["free_bytes"]) == card[index][1]
                and card[index][2] == pool,
                "crossed original CUDA charge or pool sample",
            )
            charge, upper = charge_values(
                entry["child_growth_bytes"], entry["global_used_upper_bytes"]
            )
            peak = max(peak, charge)
            absolute_peak = max(absolute_peak, upper)
            previous = index
        if "rejected_growth" in end:
            reserve = byte_count(guard["admission"]["gpuFreeReserveBytes"])
            cap = byte_count(guard["admission"]["childGpuCapBytes"])
            require(
                all(free >= reserve and used <= cap for _, free, used in card)
                and all(event["gpuFreeBytes"] >= reserve for event in events),
                "completed CUDA reading breached the capture guard",
            )
        rejected = end.get("rejected_growth", [])
        require(isinstance(rejected, list), "invalid rejected CUDA readings")
        for entry in rejected:
            require(entry["cuda_uuid"] == uuid, "rejected CUDA reading belongs to another device")
            charge, upper = charge_values(
                entry["child_growth_bytes"], entry["global_used_upper_bytes"]
            )
            peak = max(peak, charge)
            absolute_peak = max(absolute_peak, upper)
        require(absolute_peak == absolute, "original CUDA charges omit the card high-water mark")
        if entries:
            mark = end.get("pool")
            require(
                isinstance(mark, dict)
                and entries[0]["card_index"] == byte_count(mark.get("card_index"))
                and pool == byte_count(mark.get("pool_reserved_high_bytes")),
                "CUDA pool substitutions differ from the output marker",
            )
    return peak, len(events)


def validate_render_scope(scope, end, dedicated, misses, pool, card):
    """Bind the capture verifier's post-output exception to its unchanged raw Job/card records."""
    if any(misses):
        require(
            isinstance(scope, dict), "incomplete WDDM measurement without verified render scope"
        )
    else:
        require(scope is None, "unexpected WDDM render-scope exception")
    identity = {
        "pid": dedicated["pid"],
        "created_filetime": dedicated["created_filetime"],
        "cuda_uuid": dedicated["adapter"]["cuda_uuid"],
    }
    pool_mark, zero = end["pool"], end["zero"]
    require(
        isinstance(pool_mark, dict) and isinstance(zero, dict) and end["pool_ticks"] > 0,
        "missing WDDM render markers",
    )
    for mark, value in ((pool_mark, pool), (zero, 0)):
        index = byte_count(mark["card_index"])
        require(
            all(type(mark.get(k)) is type(v) and mark[k] == v for k, v in identity.items())
            and index < len(card)
            and byte_count(mark["at_ns"]) == card[index][0]
            and value == card[index][2],
            "crossed WDDM render marker or card sample",
        )
    require(
        pool_mark["basis"] == "verified-output-pool"
        and byte_count(pool_mark["pool_reserved_high_bytes"]) == pool
        and zero["basis"] == "owned-process-attributed"
        and pool_mark["card_index"] < zero["card_index"] == len(card) - 1
        and (zero["before"], zero["after"])
        in {
            ("live", "live"),
            ("live", "exited_empty"),
            ("exited_empty", "exited_empty"),
        },
        "invalid WDDM output or release marker",
    )
    if not any(misses):
        return
    expected = dict(
        identity,
        pool_card_index=pool_mark["card_index"],
        zero_card_index=zero["card_index"],
        dedicated_missing_after_output=misses[0],
        shared_missing_after_output=misses[1],
    )
    require(
        scope == expected and all(type(scope[k]) is type(v) for k, v in expected.items()),
        "verified render scope differs from original WDDM observations",
    )


def validate_rejected_growth(
    guard, end, native_pool, card, pool_high, *, completed=False, tied_read=False
):
    """A failed read is chargeable only when this Job's verified output pool has retired."""
    mark, zero = end["pool"], end["zero"]
    require(
        isinstance(native_pool, dict)
        and set(native_pool)
        == {
            "pid",
            "cudaDeviceUuid",
            "poolUsedBytes",
            "poolUsedHighBytes",
            "poolReservedBytes",
            "poolReservedHighBytes",
        },
        "missing original native CUDA pool",
    )
    used = byte_count(native_pool["poolUsedBytes"])
    reserved = byte_count(native_pool["poolReservedBytes"])
    require(
        byte_count(native_pool["pid"]) == mark["pid"]
        and native_pool["cudaDeviceUuid"] == mark["cuda_uuid"]
        and 0 < used <= reserved <= pool_high == byte_count(native_pool["poolReservedHighBytes"])
        and used <= byte_count(native_pool["poolUsedHighBytes"]) <= pool_high,
        "rejected observation differs from native CUDA pool or process",
    )
    previous = 0
    last_positive_read = 0
    exited_at = None
    reserve = byte_count(guard["admission"]["gpuFreeReserveBytes"])
    for event in end["rejected_growth"]:
        index = byte_count(event["next_card_index"])
        read, finished = byte_count(event["read_ns"]), byte_count(event["at_ns"])
        original = event["pool"]
        require(
            mark["card_index"] < index <= (len(card) if completed else zero["card_index"])
            and max(previous, card[index - 1][0]) < read <= finished
            and (index == len(card) or finished <= card[index][0])
            and byte_count(event["pid"]) == mark["pid"]
            and byte_count(event["created_filetime"]) == mark["created_filetime"]
            and event["before"] == "live"
            and event["after"] in ("live", "exited_empty")
            and exited_at is None
            and event["reason"] == "probe-failure"
            and original == native_pool
            and all(type(original[k]) is type(v) for k, v in native_pool.items())
            and byte_count(event["free_bytes"]) >= reserve
            and byte_count(event["child_growth_bytes"])
            <= byte_count(guard["admission"]["childGpuCapBytes"]),
            "unverified rejected CUDA reading or physical guard violation",
        )
        retired = (
            event["detail"] == "CUDA pool proof exceeds concurrent global use"
            and event["child_growth_bytes"] < reserved
        )
        exited = (
            event["detail"] == "CUDA pool proof has no valid owned PID/UUID allocation"
            and event["after"] == "exited_empty"
        )
        require(retired or exited, "unverified CUDA pool rejection reason")
        if event["after"] == "exited_empty":
            exited_at = index
        previous = finished
        if event["child_growth_bytes"]:
            last_positive_read = read

    if exited_at is not None:
        require(
            (zero is None or zero["before"] == "exited_empty")
            and not any(value for _, _, value in card[exited_at:]),
            "original CUDA Job resumed after exit",
        )

    if completed:
        # Job completion has a different terminal interval, validated by validate_completed_job.
        return
    # A rejected sample may extend live GPU usage beyond a previously plausible terminal PDH gap.
    observation = guard["fit_observation"]
    last_used = max(last_positive_read, max(t for t, _, value in card if value > 0))
    if tied_read:
        events = end["rejected_growth"]
        final = events[-1]
        require(
            len(events) >= 2
            and [v["card_index"] for v in end["pool_growth"]]
            == list(range(mark["card_index"], zero["card_index"]))
            and all(v["next_card_index"] == zero["card_index"] for v in events)
            and all(
                v["child_growth_bytes"] <= events[i]["child_growth_bytes"]
                for i, v in enumerate(events[1:])
            )
            and mark["at_ns"] < events[-2]["at_ns"] < final["read_ns"] == last_used
            and 0 < final["child_growth_bytes"] < reserved
            and final["before"] == "live"
            and final["after"] == "exited_empty"
            and final["detail"] == "CUDA pool proof exceeds concurrent global use"
            and exited_at == final["next_card_index"] == zero["card_index"]
            and final["read_ns"] <= final["at_ns"] < zero["at_ns"]
            and zero["before"] == zero["after"] == "exited_empty",
            "terminal CUDA read does not precede this Job's measured zero",
        )
    gaps = []
    for name, series in (
        ("dedicated", guard["cuda_attribution"]["wddm_dedicated"]),
        ("shared", observation["wddm_shared"]),
    ):
        missed = [
            item for item in observation["wddm_retry"][name]["events"] if item["recovered"] is False
        ]
        require(
            len(missed) == series["missing_active_polls"],
            "unlocated WDDM gap during CUDA pool retry",
        )
        if tied_read:
            require(
                len(missed) == 1
                and len(observation["wddm_retry"][name]["events"])
                == observation["wddm_retry"][name]["attempts"]
                == 1,
                "terminal read lacks both original isolated gauge gaps",
            )
        for item in missed:
            start = byte_count(item["start_ns"])
            if tied_read:
                gaps.append(item)
                tick_ns = byte_count(guard["admission"]["pollMillis"]) * 1_000_000
                finish = byte_count(item["end_ns"])
                lower, upper = (
                    byte_count(item["local_filetime_before"]),
                    byte_count(item["local_filetime_after"]),
                )
                require(
                    0 < lower <= upper
                    and (upper - lower) * 100 <= tick_ns
                    and 0 <= finish - start <= tick_ns
                    and final["read_ns"] <= start <= finish <= final["at_ns"],
                    "terminal gauge retry falls outside its source attempt",
                )
            require(
                (start >= last_used if tied_read else start > max(mark["at_ns"], last_used))
                and start <= byte_count(item["end_ns"]) <= zero["at_ns"]
                and item["retried"] is True
                and item["retry_sample"] is None
                and item["previous"] == series["samples"][-1]
                and byte_count(item["local_filetime_before"]) >= series["samples"][-1][0],
                "WDDM gap overlaps a rejected CUDA reading",
            )
    if tied_read:
        dedicated, shared = gaps
        require(
            dedicated["start_ns"] == final["read_ns"]
            and dedicated["end_ns"] <= shared["start_ns"] <= shared["end_ns"] == final["at_ns"]
            and dedicated["local_filetime_after"] <= shared["local_filetime_before"],
            "synchronous terminal gauge order differs from the original CUDA read",
        )


def validate_job_release(guard, claim, basis):
    pid = guard["pid"]
    if basis == "owned-job-completion":
        expected = {
            "scope": "supervisor-owned-windows-job",
            "root_pid": pid,
            "root_returncode": 0,
            "waited_returncode": 0,
            "job_member_pids": [],
            "owned_pids": [],
        }
        for observed in (guard["owned_completion"], claim["owned_execution_completion"]):
            require(
                json.dumps(observed, sort_keys=True) == json.dumps(expected, sort_keys=True),
                "missing owned Job completion",
            )
    else:
        require(basis == "archived-supervisor-return", "unknown CUDA completion basis")
    release = claim["release_background_activity"]["samples"]
    require(
        len(release) == 5
        and all(
            v["selected_index_matches_admission"] is True
            and v["post_owned_exit_primary_metadata_valid"] is True
            and all(
                byte_count(v["selected_card"][key]) == 0
                for key in ("index", "used_mib", "utilization_percent")
            )
            for v in release
        ),
        "missing five quiet selected-card release samples",
    )


def validate_cuda_guard_samples(guard, end, card):
    cap, reserve = guard["admission"]["childGpuCapBytes"], guard["admission"]["gpuFreeReserveBytes"]
    values = [(free, used) for _, free, used in card]
    values.extend(
        (v["gpuFreeBytes"], v["childGlobalUsedGrowthBytes"])
        for v in guard["samples"]
        if v.get("phase") == "cuda-global-used-upper-bound-sample"
    )
    values.extend(
        (v["free_bytes"], v["child_growth_bytes"])
        for v in end["pool_growth"] + end["rejected_growth"]
    )
    require(
        all(free >= reserve and used <= cap for free, used in values),
        "completed CUDA Job exceeded guard",
    )


def validate_completed_job(guard, claim, summary, basis, pool, card, native_pool):
    """Verify observed load through output. The final Job release supplies no PID memory sample."""
    end = guard["fit_observation"]["cuda_render_end"]
    mark, events = end["pool"], end["rejected_growth"]
    pid, uuid = guard["pid"], guard["admission"]["cudaDeviceUuid"]
    require(
        end["zero"] is None
        and uuid == "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
        and mark["basis"] == "verified-output-pool"
        and mark["pid"] == pid
        and mark["cuda_uuid"] == uuid
        and byte_count(mark["card_index"]) < len(card)
        and (mark["at_ns"], mark["pool_reserved_high_bytes"]) == (card[mark["card_index"]][0], pool)
        and card[mark["card_index"]][2] == pool
        and claim["physical_index"] == 0
        and claim["device_uuid"] == uuid
        and claim["lease_released"] is True
        and claim["post_execution_verified"] is True
        and claim["release_metadata_verification"] == {"status": "verified"},
        "CUDA completion does not identify the released output and selected device",
    )
    validate_job_release(guard, claim, basis)
    final = [v for v in events if v["next_card_index"] == len(card)]
    require(
        len(final) == 4
        and events[-4:] == final
        and [v["card_index"] for v in end["pool_growth"]]
        == list(range(mark["card_index"], len(card)))
        and all(
            v["child_growth_bytes"] <= final[i]["child_growth_bytes"]
            for i, v in enumerate(final[1:])
        ),
        "incomplete CUDA post-output attempt sequence",
    )
    validate_rejected_growth(guard, end, native_pool, card, pool, completed=True)
    validate_cuda_guard_samples(guard, end, card)
    expected = {
        "basis": basis,
        "pid": pid,
        "cuda_uuid": uuid,
        "created_filetime": mark["created_filetime"],
        "pool_card_index": mark["card_index"],
        "terminal_rejected_attempts": len(final),
    }
    observation = guard["fit_observation"]
    tick_ns = byte_count(guard["admission"]["pollMillis"]) * 1_000_000
    terminal_gaps = []
    for name, series in (
        ("dedicated", guard["cuda_attribution"]["wddm_dedicated"]),
        ("shared", observation["wddm_shared"]),
    ):
        gaps = [v for v in observation["wddm_retry"][name]["events"] if v["recovered"] is False]
        require(
            mark["created_filetime"] == series["created_filetime"]
            and len(gaps) == series["missing_active_polls"] <= 1
            and len(series["samples"]) + len(gaps) == len(card) + len(events),
            "unlocated measurements in completed CUDA Job",
        )
        for gap in gaps:
            terminal_gaps.append(gap)
            start, finish = gap["start_ns"], gap["end_ns"]
            lower, upper = gap["local_filetime_before"], gap["local_filetime_after"]
            require(
                all(type(v) is int and v > 0 for v in (start, finish, lower, upper))
                and 0 <= finish - start <= tick_ns
                and 0 <= (upper - lower) * 100 <= tick_ns
                and final[-2]["at_ns"]
                < final[-1]["read_ns"]
                <= gap["start_ns"]
                <= gap["end_ns"]
                <= final[-1]["at_ns"]
                and gap["retried"] is True
                and gap["retry_sample"] is None
                and gap["previous"] == series["samples"][-1]
                and gap["local_filetime_before"] >= series["samples"][-1][0],
                "WDDM gap overlaps measured render",
            )
        expected[name + "_missing_after_output"] = len(gaps)
    if len(terminal_gaps) == 2:
        dedicated, shared = terminal_gaps
        require(
            dedicated["end_ns"] <= shared["start_ns"]
            and dedicated["local_filetime_after"] <= shared["local_filetime_before"],
            "terminal WDDM observations contradict gauge order",
        )
    require(
        json.dumps(summary, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "CUDA completion evidence differs from raw record",
    )


def windows_device_peak(
    guard,
    native,
    adapter,
    raw_gross,
    scope=None,
    native_pool=None,
    *,
    completion=None,
    tied_read=False,
):
    admission = guard["admission"]
    baseline = byte_count(admission.get("gpuGlobalUsedBaselineBytes"))
    attribution = guard["cuda_attribution"]
    observation = guard["fit_observation"]
    dedicated = attribution["wddm_dedicated"]
    shared = observation["wddm_shared"]
    require(
        isinstance(adapter, dict)
        and isinstance(adapter.get("cuda_uuid"), str)
        and adapter["cuda_uuid"]
        and adapter["cuda_uuid"] == admission["cudaDeviceUuid"]
        and adapter.get("counter") == r"\GPU Process Memory(*)\Dedicated Usage"
        and observation["wddm_shared_counter"] == "GPU Process Memory/Shared Usage"
        and type(guard["pid"]) is int
        and guard["pid"] > 0
        and type(dedicated["created_filetime"]) is int
        and dedicated["created_filetime"] > 0,
        "invalid WDDM adapter or process identity",
    )
    misses = [byte_count(rows["missing_active_polls"]) for rows in (dedicated, shared)]
    for rows in (dedicated, shared):
        require(
            rows["adapter"] == adapter
            and type(rows["pid"]) is int
            and rows["pid"] == guard["pid"]
            and type(rows["created_filetime"]) is int
            and rows["created_filetime"] == dedicated["created_filetime"]
            and rows["error"] is None,
            "incomplete or crossed WDDM process measurements",
        )
    wddm = sample_values(dedicated["samples"], 2, "WDDM dedicated")
    require(sum(value > 0 for value in wddm) >= 2, "incomplete WDDM dedicated peak")
    # Host pricing retains the producer's separately classified Job/working-set plus WDDM
    # shared peak. Its samples must belong to the same verified process as the device series.
    sample_values(shared["samples"], 2, "WDDM shared")

    pid = byte_count(attribution["pid_peak_bytes"])
    pool = byte_count(native["allocator_peak"])
    fallback = byte_count(attribution["pool_fallback_peak_bytes"])
    absolute = byte_count(attribution["global_upper_bytes"])
    unknown = byte_count(attribution["unknown_pid_polls"])
    require(
        pool > 0
        and fallback <= pool
        and guard["peak_gpu_bytes"] == (max(pid, fallback) or None)
        and (
            (unknown > 0 and absolute > 0)
            or (unknown == 0 and pid > 0 and absolute == fallback == 0)
        ),
        "inconsistent owned-PID or whole-card CUDA measurement",
    )
    owned_or_floor = max(pool, pid, max(wddm), byte_count(native["axes"]["floor_device_bytes"]))
    card = observation["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
    free = sample_values(card, 3, "free-card")
    growth, growth_count = tenant_growth(
        guard, adapter["cuda_uuid"], absolute, unknown, baseline, card, pool
    )
    if "cuda_render_end" in observation:
        end = observation["cuda_render_end"]
        rejected = end.get("rejected_growth", [])
        require(
            byte_count(end.get("pool_ticks")) + growth_count + len(rejected) == unknown,
            "unaccounted supervisor CUDA fallback tick",
        )
        if completion is not None:
            require(
                type(guard["returncode"]) is int and guard["returncode"] == 0,
                "CUDA Job did not return successfully",
            )
            claim, summary, basis = completion
            require(
                scope is None and rejected and fallback == pool,
                "conflicting CUDA end-of-render evidence",
            )
            validate_completed_job(guard, claim, summary, basis, pool, card, native_pool)
        elif any(misses) or rejected:
            require(fallback == pool, "output marker differs from native pool fallback")
            if tied_read:
                scope = scope.copy()
                final = rejected[-1]
                for field, raw in (
                    ("terminal_rejected_read_ns", "read_ns"),
                    ("terminal_rejected_at_ns", "at_ns"),
                ):
                    require(
                        byte_count(scope.pop(field)) == final[raw],
                        "certificate terminal read timestamp differs",
                    )
            validate_render_scope(scope, end, dedicated, misses, pool, card)
            if rejected:
                validate_rejected_growth(guard, end, native_pool, card, pool, tied_read=tied_read)
        else:
            require(scope is None, "unexpected WDDM render-scope exception")
        classified = max(owned_or_floor, growth)
        require(
            classified == raw_gross and classified <= byte_count(admission["childGpuCapBytes"]),
            "device measurement differs from verified native quote or exceeds guard",
        )
        return classified
    # Older producers quote the absolute card peak and have no output marker. Keep their
    # strict gap-free requirement and their conservative checks against both card baselines.
    require(
        not any(misses) and scope is None and completion is None,
        "incomplete WDDM measurement without render evidence",
    )
    require(
        max(absolute, owned_or_floor) == raw_gross,
        "device measurement differs from verified native quote",
    )
    return max(owned_or_floor, growth, max(0, absolute - baseline), free[0] - min(free))


RECOVERY_CERTIFICATE_SHA256 = "dcc7c1eb186b001c75f4a9df2d49dd06501e8064be7d1818adb2ea59031c093f"
RECOVERED_SIX = {
    "run_id": "44935671433",
    "run_attempt": "1",
    "artifact_id": "14719969057",
    "inference_revision": "7f6c4cca3fc053030bd71f0bfce6886adb762eb6",
    "plan_sha256": "0bcc7e1047f960234f93ba23c04a2abfe99c2581287c4163224defe74e4751b7",
    "archive_sha256": "83985597d8d0cf2399b2311c1f1d737058cc1c1530b1a53fce6c79faee123969",
    "receipt_sha256": "750313a3cdb89dd4a0e4e49b31935949301b309fe9410f2bf0ce47cb4628575f",
}


FOUR_COUNT_CAPTURE = {
    "run_id": "44944284405",
    "run_attempt": "1",
    "artifact_id": "14725116132",
    "inference_revision": "f5c9ff80acd00334fd668e4ecc8d56be3347affd",
    "plan_sha256": "0f4b541c702bf45d607e138b56044fa8407a788b3b041d6e5f600b040097cf35",
    "archive_sha256": "39366924a8a5e09d4e1b0186615b6d3d5de7995ac18c87cad2f48ca918feb8e8",
    "receipt_sha256": "d423f72d816ac6145c7553623a26e99f67fd59db8e83ab228a89ede8698ce8af",
}
FOUR_COUNT_CERTIFICATES = {
    3: "c18632894d2d40c7d2b0eea519657714d873a611ca9e6f0845b0809dab4a024d",
    5: "bd8567c64d120f24e9821747d94b63faec3e5919930ce65dccb5a79f7df86fe7",
}


def partial_count_row(receipt, review, archive, certificates):
    identity = FOUR_COUNT_CAPTURE
    count = review["references"]
    require(
        archive
        and sha256(archive.read_bytes()) == identity["archive_sha256"]
        and all(
            str(review[k]) == identity[k]
            for k in ("artifact_id", "receipt_sha256", "run_id", "run_attempt")
        )
        and receipt["contract"]["head"] == identity["inference_revision"]
        and sha256(json.dumps(receipt["contract"], sort_keys=True, separators=(",", ":")).encode())
        == identity["plan_sha256"]
        and receipt["accepted"] is False
        and count in (2, 3, 4, 5),
        "different four-count archive or review",
    )
    record = receipt["counts"][str(count)]
    if count in (2, 4):
        require(
            record["accepted"] is True
            and certificates is None
            and "recovery_certificate_sha256" not in review,
            "original accepted count cannot use recovery",
        )
        return record["candidate_row_pending_physical_review"]
    require(
        certificates
        and record["accepted"] is False
        and record["telemetry_failure"] == "incomplete_wddm_samples"
        and "candidate_row_pending_physical_review" not in record,
        "missing original rejected count or separate certificate",
    )
    cert = json.loads(certificates.read_bytes())[str(count)]
    digest = sha256((json.dumps(cert, sort_keys=True, separators=(",", ":")) + "\n").encode())
    require(
        cert["schema"] == "scail2-quiet-gpu0-owned-zero-recovery-v1"
        and cert["references"] == count
        and cert["capture"] == identity
        and digest == FOUR_COUNT_CERTIFICATES[count] == review["recovery_certificate_sha256"],
        "different owned-zero recovery certificate",
    )
    return cert["candidate_row_pending_physical_review"]


def recovered_row(receipt, review, certificate_path, archive_path):
    require(
        certificate_path and archive_path,
        "recovery requires both the original archive and native certificate",
    )
    certificate_bytes = certificate_path.read_bytes()
    certificate = json.loads(certificate_bytes)
    require(
        certificate.get("schema") == "scail2-quiet-gpu0-six-recovery-v1"
        and certificate["capture"] == RECOVERED_SIX
        and certificate["references"] == review["references"] == 6
        and all(
            str(review[k]) == RECOVERED_SIX[k]
            for k in ("artifact_id", "receipt_sha256", "run_id", "run_attempt")
        )
        and receipt["contract"]["head"] == RECOVERED_SIX["inference_revision"]
        and receipt["accepted"] is False
        and receipt["counts"]["6"]["accepted"] is False
        and "candidate_row_pending_physical_review" not in receipt["counts"]["6"]
        and sha256(archive_path.read_bytes()) == RECOVERED_SIX["archive_sha256"]
        and review.get("recovery_certificate_sha256")
        == sha256(certificate_bytes)
        == RECOVERY_CERTIFICATE_SHA256,
        "recovery certificate, original capture and independent review must agree",
    )
    return certificate["candidate_row_pending_physical_review"]


def project(receipt_path, review_path, *, certificate_path=None, archive_path=None):
    receipt_bytes, review_bytes = receipt_path.read_bytes(), review_path.read_bytes()
    review = json.loads(review_bytes)
    require(
        sha256(receipt_bytes) == review["receipt_sha256"],
        "review belongs to a different receipt",
    )
    receipt = json.loads(receipt_bytes)
    partial = (
        str(review["artifact_id"]) == FOUR_COUNT_CAPTURE["artifact_id"]
        or review["receipt_sha256"] == FOUR_COUNT_CAPTURE["receipt_sha256"]
    )
    recovery = not partial and (certificate_path is not None or archive_path is not None)
    require(
        (receipt["accepted"] is True or recovery or partial)
        and receipt["contract"]["protocol"] == "scail2-plural-fit-v1",
        "incomplete or unrelated campaign",
    )
    require(
        str(review["run_id"]) == receipt["run_id"]
        and str(review["run_attempt"]) == receipt["run_attempt"],
        "review belongs to another workflow attempt",
    )
    if (
        str(review["artifact_id"]) == RECOVERED_SIX["artifact_id"]
        or review["receipt_sha256"] == RECOVERED_SIX["receipt_sha256"]
    ):
        require(recovery, "archived six can qualify only through the recovered count")
    count = review["references"]
    require(
        type(count) is int
        and 2 <= count <= 6
        and type(review["artifact_id"]) is int
        and review["artifact_id"] > 0,
        "missing artifact or character identity",
    )
    require(
        isinstance(review["rationale"], str) and review["rationale"].strip(),
        "missing physical review rationale",
    )
    record = receipt["counts"][str(count)]
    require(
        (record["accepted"] is True or recovery or partial)
        and record["claim"]["post_execution_verified"] is True,
        "capture did not complete",
    )
    guard_name = f"count-{count}.supervisor.json"
    guard_bytes = (receipt_path.parent / guard_name).read_bytes()
    require(
        sha256(guard_bytes) == receipt["logs"][guard_name]["sha256"],
        "physical guard record changed",
    )
    guard = json.loads(guard_bytes)
    require(
        guard["returncode"] == 0 and guard["pid"] == record["native"]["pid"],
        "capture guard did not complete for this process",
    )
    if partial:
        row = partial_count_row(receipt, review, archive_path, certificate_path)
    elif recovery:
        row = recovered_row(receipt, review, certificate_path, archive_path)
    else:
        row = record["candidate_row_pending_physical_review"]
        require(
            "recovery_certificate_sha256" not in review,
            "unexpected recovery review for ordinary capture",
        )
    native, witness = record["native"], row["witness"]
    require(
        witness["axes"] == native["axes"] and native["axes"]["references"] == count,
        "crossed native axes",
    )
    tokens = native["prompt_tokens"]
    require(
        tokens == [512] * (2 if native["axes"]["classifier_free_guidance"] else 1)
        and witness["text_tokens"] == min(tokens),
        "incomplete text coverage",
    )
    require(
        row["inference_revision"] == receipt["contract"]["head"]
        and re.fullmatch(r"[0-9a-f]{40}", row["inference_revision"]),
        "crossed inference revision",
    )
    admission, hw = guard["admission"], receipt["hardware"]
    if row["platform"] == "macos":
        require(
            (row["device"], row["runtime"]) == (hw["chip"], hw["os_build"])
            and native["axes"]["backend"] == "mlx",
            "crossed Metal identity",
        )
        pools = {
            "unified": (
                guard["lifetime_peak_host_bytes"],
                witness["gross_unified_bytes"],
                admission["childFootprintCapBytes"],
            )
        }
        bounds = {
            "measurement": "darwin_lifetime_footprint",
            "mlx_decode_budget_bytes": witness["mlx_decode_budget_bytes"],
        }
        require(
            byte_count(bounds["mlx_decode_budget_bytes"]) > 0,
            "missing decoder measurement cap",
        )
    else:
        require(
            row["platform"] == "windows"
            and (row["device"], row["runtime"]) == (hw["name"], hw["driver_version"])
            and native["axes"]["backend"] == "cuda"
            and hw["uuid"] == receipt["adapter"]["cuda_uuid"]
            and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", hw["uuid"])
            and re.fullmatch(r"luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}", receipt["adapter"]["luid"])
            and receipt["adapter"]["luid"] != "luid_0x00000000_0x00000000"
            and type(receipt["adapter"]["physical_node"]) is int
            and 0 <= receipt["adapter"]["physical_node"] < 32
            and type(receipt["adapter"]["node_mask"]) is int
            and receipt["adapter"]["node_mask"] == 1 << receipt["adapter"]["physical_node"],
            "crossed CUDA identity",
        )
        if str(hw.get("index")) == "0" or hw["uuid"] == "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0":
            adapter = receipt["adapter"]
            require(
                str(hw["index"]) == "0"
                and hw["uuid"] == "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
                and isinstance(native.get("cuda_device"), dict)
                and all(
                    type(native["cuda_device"].get(k)) is int
                    for k in ("ordinal", "visible_count", "node_mask")
                )
                and hw["pci.bus_id"].upper() == "00000000:21:00.0"
                and native.get("cuda_device")
                == {
                    "ordinal": 0,
                    "visible_count": 1,
                    "uuid": hw["uuid"],
                    "pci_bus_id": "00000000:21:00.0",
                    "luid": adapter["luid"],
                    "node_mask": adapter["node_mask"],
                },
                "GPU0 capture lacks the matching native execution adapter",
            )
        native_pool = None
        end = guard["fit_observation"].get("cuda_render_end", {})
        if "rejected_growth" in end:
            policy = receipt["contract"]["request"]["policy"]
            policy_bytes = (
                json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
            require(
                receipt["request"]["policy"] == policy
                and admission["policySha256"] == sha256(policy_bytes)
                and all(
                    admission[k] == policy[k] and type(admission[k]) is type(policy[k])
                    for k in ("cudaDeviceUuid", "childGpuCapBytes", "gpuFreeReserveBytes")
                ),
                "CUDA rejection guard differs from approved capture policy",
            )
        if end.get("rejected_growth"):
            pool_name = f"count-{count}.cuda-pool.json"
            pool_bytes = (receipt_path.parent / pool_name).read_bytes()
            require(
                sha256(pool_bytes) == receipt["logs"][pool_name]["sha256"],
                "native CUDA pool record changed",
            )
            native_pool = json.loads(pool_bytes)
        completion = None
        if recovery or "cuda_completed_job" in row:
            summary = row["cuda_completed_job"]
            basis = "archived-supervisor-return" if recovery else "owned-job-completion"
            require(
                summary["basis"] == basis
                and hw["uuid"] == "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0",
                "unverified CUDA completion source",
            )
            require(
                review["domains"]["device"]["basis"] == "measured_aggregate",
                "the completed RGB device interval requires an empirical review",
            )
            require(
                type(guard["admission"]["pollMillis"]) is int
                and guard["admission"]["pollMillis"]
                == receipt["request"]["policy"]["pollMillis"]
                == 250,
                "completed Job polling interval differs from the signed policy",
            )
            completion = (record["claim"], summary, basis)
        if partial:
            require(
                guard["admission"]["pollMillis"]
                == receipt["request"]["policy"]["pollMillis"]
                == 250
                and record["claim"]["device_uuid"] == hw["uuid"]
                and record["claim"]["physical_index"] == 0
                and record["claim"]["lease_released"] is True
                and record["claim"]["release_metadata_verification"] == {"status": "verified"}
                and review["domains"]["device"]["basis"] == "measured_aggregate",
                "partial CUDA count lacks original signed guard or release",
            )
            validate_job_release(guard, record["claim"], "owned-job-completion")
            validate_cuda_guard_samples(
                guard,
                end,
                guard["fit_observation"]["card_samples_monotonic_ns_free_owned_or_fallback_bytes"],
            )
        if completion or partial:
            counters = guard["fit_observation"]["windows_peak"]
            host = max(
                byte_count(guard["peak_host_bytes"]),
                *(
                    byte_count(counters[k])
                    for k in (
                        "peak_job_commit_bytes",
                        "peak_private_commit_bytes",
                        "peak_working_set_bytes",
                    )
                ),
            )
            shared = sample_values(
                guard["fit_observation"]["wddm_shared"]["samples"], 2, "WDDM shared"
            )
            require(
                witness["gross_host_bytes"]
                == max(host + max(shared), native["axes"]["floor_host_bytes"])
                and witness["gross_unified_bytes"] == 0
                and witness["mlx_decode_budget_bytes"] is None,
                "completed CUDA host bounds differ from original measurements",
            )
        pools = {
            "host": (
                guard["peak_host_bytes"],
                witness["gross_host_bytes"],
                admission["childFootprintCapBytes"],
            ),
            "device": (
                native["allocator_peak"],
                windows_device_peak(
                    guard,
                    native,
                    receipt["adapter"],
                    witness["gross_device_bytes"],
                    row.get("cuda_render_scope"),
                    native_pool,
                    completion=completion,
                    tied_read=partial and not record["accepted"],
                ),
                admission["childGpuCapBytes"],
            ),
        }
        bounds = {
            "measurement": "windows_job_and_wddm",
            **(
                {"device_interval": "load_through_completed_rgb"}
                if completion or (partial and not record["accepted"])
                else {}
            ),
            "capture_adapter": {
                k: receipt["adapter"][k] for k in ("cuda_uuid", "luid", "node_mask")
            },
        }
    require(set(review["domains"]) == set(pools), "review is missing a physical domain")
    for domain, (observed, classified, ceiling) in pools.items():
        bounds[domain] = envelope(
            review["domains"][domain],
            observed,
            classified,
            ceiling,
            sha256(review_bytes),
        )
    require(
        re.fullmatch(r"[0-9a-f]{64}", admission["policySha256"]),
        "missing guard policy digest",
    )
    return {
        **{k: row[k] for k in ("platform", "device", "runtime", "inference_revision")},
        "artifact_id": review["artifact_id"],
        "receipt_sha256": review["receipt_sha256"],
        "guard_policy_sha256": admission["policySha256"],
        "accepted": receipt["accepted"],
        **({"count_accepted": record["accepted"]} if partial else {}),
        **(
            {"recovery_certificate_sha256": review["recovery_certificate_sha256"]}
            if recovery or (partial and not record["accepted"])
            else {}
        ),
        "post_execution_verified": record["claim"]["post_execution_verified"],
        "guard_exit_code": guard["returncode"],
        "axes": native["axes"],
        "prompt_tokens": tokens,
        "bounds": bounds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--recovery-certificate", type=Path)
    parser.add_argument("--recovery-archive", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            project(
                args.receipt,
                args.review,
                certificate_path=args.recovery_certificate,
                archive_path=args.recovery_archive,
            ),
            sort_keys=True,
            indent=2,
        )
    )
