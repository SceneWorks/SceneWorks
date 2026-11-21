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
            "host": (guard["peak_host_bytes"], admission["childFootprintCapBytes"]),
            "device": (native["allocator_peak"], admission["childGpuCapBytes"]),
        }
        bounds = {"measurement": "windows_job_and_wddm"}
    require(set(review["domains"]) == set(pools), "review is missing a physical domain")
    for domain, (observed, ceiling) in pools.items():
        bounds[domain] = envelope(
            review["domains"][domain],
            observed,
            witness[f"gross_{domain}_bytes"],
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
