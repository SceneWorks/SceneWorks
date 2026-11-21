"""Print one compact worker row from a verified SCAIL physical capture and its review.

Run after the inference producer's --verify-artifacts has passed. The review JSON
names the immutable artifact_id, run_id, run_attempt, receipt_sha256, references, rationale, and
one entry per domain in "domains". Each entry supplies a "basis" and only its
reviewed extra/uncertainty or overshoot bytes. Capture facts come from the files.

    python scripts/project-scail2-memory.py path/to/receipt.json physical-review.json
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


def tenant_growth(guard, uuid, absolute, unknown, baseline):
    """Read the child charge from the same supervisor sample as its whole-card upper bound."""
    samples = guard.get("samples")
    require(isinstance(samples, list), "missing supervisor CUDA samples")
    events = [
        event for event in samples if event.get("phase") == "cuda-global-used-upper-bound-sample"
    ]
    require(len(events) <= unknown, "unexpected supervisor CUDA sample count")
    peak = last = 0
    for event in events:
        upper = byte_count(event.get("globalUsedUpperBoundBytes"))
        charge = byte_count(event.get("childGlobalUsedGrowthBytes"))
        require(
            event.get("cudaDeviceUuid") == uuid
            and max(0, upper - baseline) <= charge <= upper <= absolute
            and type(event.get("at_ns")) is int
            and event["at_ns"] > last,
            "inconsistent supervisor CUDA charge or device",
        )
        byte_count(event.get("gpuFreeBytes"))
        peak = max(peak, charge)
        last = event["at_ns"]
    return peak, len(events)


def validate_render_scope(scope, end, dedicated, misses, pool, card):
    """Bind the capture verifier's post-output exception to its unchanged raw Job/card records."""
    require(
        isinstance(scope, dict),
        "incomplete WDDM measurement without verified render scope",
    )
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


def windows_device_peak(guard, native, adapter, raw_gross, scope=None):
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
    growth, growth_count = tenant_growth(guard, adapter["cuda_uuid"], absolute, unknown, baseline)
    if "cuda_render_end" in observation:
        end = observation["cuda_render_end"]
        require(
            isinstance(end, dict) and byte_count(end.get("pool_ticks")) + growth_count == unknown,
            "unaccounted supervisor CUDA fallback tick",
        )
        if any(misses):
            require(fallback == pool, "output marker differs from native pool fallback")
            validate_render_scope(scope, end, dedicated, misses, pool, card)
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
        not any(misses) and scope is None,
        "incomplete WDDM measurement without render evidence",
    )
    require(
        max(absolute, owned_or_floor) == raw_gross,
        "device measurement differs from verified native quote",
    )
    return max(owned_or_floor, growth, max(0, absolute - baseline), free[0] - min(free))


def project(receipt_path, review_path):
    receipt_bytes, review_bytes = receipt_path.read_bytes(), review_path.read_bytes()
    review = json.loads(review_bytes)
    require(
        sha256(receipt_bytes) == review["receipt_sha256"],
        "review belongs to a different receipt",
    )
    receipt = json.loads(receipt_bytes)
    require(
        receipt["accepted"] is True and receipt["contract"]["protocol"] == "scail2-plural-fit-v1",
        "incomplete or unrelated campaign",
    )
    require(
        str(review["run_id"]) == receipt["run_id"]
        and str(review["run_attempt"]) == receipt["run_attempt"],
        "review belongs to another workflow attempt",
    )
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
        record["accepted"] is True and record["claim"]["post_execution_verified"] is True,
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
    row = record["candidate_row_pending_physical_review"]
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
            and native["axes"]["backend"] == "cuda",
            "crossed CUDA identity",
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
                ),
                admission["childGpuCapBytes"],
            ),
        }
        bounds = {"measurement": "windows_job_and_wddm"}
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
    args = parser.parse_args()
    print(json.dumps(project(args.receipt, args.review), sort_keys=True, indent=2))
