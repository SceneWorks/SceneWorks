#!/usr/bin/env python3
"""Dispatch-only artifact verifier. Never loads a model or executes native code.

Schema validity is necessary, not sufficient. This module independently checks
loss arithmetic, actual adapter payloads, prepared-input reuse, source/artifact
bindings, recovery, resource telemetry and the two independent phase clocks.
SCHEMA TEST ONLY records can never become accepted campaign evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MANIFEST_SHA256 = "c02b05e3c40e6c9666554580ff0636d5853691ffe5f5db5d586d12468c82a0fd"
CELLS = tuple(f"{backend}/{parent}/{method}" for backend in ("mlx", "cuda")
              for parent in ("base", "sft") for method in ("lora", "lokr"))
CHUNK_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024


# Explicit trusted sibling loading works under Python -I without adding the
# downloaded evidence directory (or cwd) to sys.path.
_normalizer_spec = importlib.util.spec_from_file_location(
    "moss_acceptance_native_normalizer", Path(__file__).with_name("normalize_native.py"))
_normalizer = importlib.util.module_from_spec(_normalizer_spec)
_normalizer_spec.loader.exec_module(_normalizer)


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _invalid_constant(value):
    raise EvidenceError(f"non-JSON/nonfinite numeric constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(file):
    with Path(file).open(encoding="utf-8") as source:
        return json.load(source, parse_constant=_invalid_constant,
                         object_pairs_hook=_unique_object)


def digest(file):
    result = hashlib.sha256()
    with Path(file).open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            result.update(block)
    return result.hexdigest()


def member_path(root, relative):
    require(isinstance(relative, str) and relative, "artifact path missing")
    member = PurePosixPath(relative)
    require(not member.is_absolute() and ".." not in member.parts
            and "\\" not in relative and ":" not in relative,
            f"unsafe artifact path: {relative}")
    base = Path(root).resolve()
    file = base
    for part in member.parts:
        file = file / part
        require(not file.is_symlink(), f"artifact symlink forbidden: {relative}")
    require(file.is_file() and file.resolve().is_relative_to(base),
            f"missing/nonregular artifact: {relative}")
    return file


def checked_artifact(root, reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            "artifact must bind path and SHA256")
    file = member_path(root, reference["path"])
    require(digest(file) == reference["sha256"], f"artifact hash mismatch: {reference['path']}")
    return file


def finite(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/missing {label}")
    if positive:
        require(value > 0, f"nonpositive {label}")
    return value


def timestamp(value):
    require(isinstance(value, str), "timestamp missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"invalid timestamp: {value}") from error
    require(parsed.tzinfo is not None, "timestamp must include timezone")
    return parsed.timestamp()


def read_safetensors(file):
    try:
        tensors, start, _ = _normalizer._safetensors_header(file)
        return tensors, start
    except _normalizer.NormalizationError as error:
        raise EvidenceError(str(error)) from error

def _values(data, dtype):
    if dtype == "BF16":
        return (struct.unpack("<f", struct.pack("<I", word[0] << 16))[0]
                for word in struct.iter_unpack("<H", data))
    return (value[0] for value in struct.iter_unpack({"F32": "<f", "F16": "<e", "F64": "<d"}[dtype], data))


def tensor_payloads_equal(left_file, right_file):
    try:
        _normalizer.tensor_payloads_equal(left_file, right_file)
    except _normalizer.NormalizationError as error:
        raise EvidenceError(str(error)) from error


def recompute_factor_delta(left_file, right_file):
    left, left_start = read_safetensors(left_file)
    right, right_start = read_safetensors(right_file)
    require(set(left) == set(right), "factor export tensor keys differ")
    elements = changed = 0
    maximum = squared = 0.0
    sizes = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}
    with Path(left_file).open("rb") as a, Path(right_file).open("rb") as b:
        for key in sorted(left):
            x, y = left[key], right[key]
            require(x["shape"] == y["shape"] and x["dtype"] == y["dtype"] == "F32",
                    f"actual portable factor layout/dtype changed: {key}")
            item_size = sizes[x["dtype"]]
            remaining = x["data_offsets"][1] - x["data_offsets"][0]
            a.seek(left_start + x["data_offsets"][0])
            b.seek(right_start + y["data_offsets"][0])
            while remaining:
                count = min(remaining, CHUNK_BYTES - CHUNK_BYTES % item_size)
                ax, bx = a.read(count), b.read(count)
                require(len(ax) == count and len(bx) == count, "truncated actual factor payload")
                for av, bv in zip(_values(ax, x["dtype"]), _values(bx, y["dtype"])):
                    finite(av, "start factor")
                    finite(bv, "end factor")
                    if key.endswith(".alpha"):
                        require(struct.pack("<f", av) == struct.pack("<f", bv),
                                "portable alpha scalar changed")
                        continue
                    delta = abs(av - bv)
                    elements += 1
                    changed += av != bv
                    maximum = max(maximum, delta)
                    squared += delta * delta
                remaining -= count
    finite(squared, "recomputed squared factor delta", positive=True)
    require(changed > 0 and maximum > 0, "actual adapter factors did not update")
    return {"tensorCount": sum(not name.endswith(".alpha") for name in left), "elementCount": elements, "changedElements": changed,
            "maxAbsDelta": maximum, "squaredL2Delta": squared}


def verify_optional_prepared_evaluation(record, manifest):
    """Pure completed-subrun policy against the verified merged source contract.

    Expected canceled-run partial diagnostics are a separate context and are not
    treated as a completed learning pair by this policy.

    Merged reference: b939f3e883b75d3ec5e12263bde914ae5f5b2830,
    moss_training_probe.rs SHA256
    1f0148ffa9414ef1c15c8875928d1c2b9c0ffc5888a5da22107d36f911e2d5ec.
    Merged-byte reverification confirms identity with PR #1344 head 160fc5de.
    This function never labels a record accepted or executes a model.
    """
    required = manifest["evaluation"]["withinNativeRun"]["recordBeforeAndAfterTensorShapesDtypesLayoutsAndHashes"]
    diagnostic = record.get("prepared_evaluation")
    if diagnostic is None:
        require(not required, "required prepared_evaluation diagnostic is absent; fail-closed refusal")
        return None
    require(isinstance(diagnostic, dict), "prepared_evaluation must be an object")
    for phase in ("before", "after"):
        point = diagnostic.get(phase)
        require(isinstance(point, dict) and {"loss", "sigma", "tensors"} <= set(point),
                f"prepared_evaluation {phase} has incomplete observation keys")
        finite(point["loss"], f"prepared {phase} loss")
        finite(point["sigma"], f"prepared {phase} sigma")
        exact_native_f32(point["sigma"], manifest["evaluation"]["sigma"], f"prepared {phase} fixed sigma")
        tensors = point["tensors"]
        require(isinstance(tensors, dict)
                and {"latent", "context", "noise"} <= set(tensors)
                and set(tensors) <= {"latent", "context", "noise", "mask"},
                f"prepared_evaluation {phase} has an incomplete/unexpected tensor key set")
        for name, descriptor in tensors.items():
            require(isinstance(descriptor, dict)
                    and {"shape", "strides", "dtype", "content_encoding", "content_sha256"} <= set(descriptor),
                    f"prepared {phase}/{name} descriptor keys incomplete")
            shape, strides = descriptor["shape"], descriptor["strides"]
            require(isinstance(shape, list) and shape and all(type(x) is int and x > 0 for x in shape)
                    and isinstance(strides, list) and len(strides) == len(shape)
                    and all(type(x) is int and x >= 0 for x in strides),
                    f"prepared {phase}/{name} shape/strides invalid")
            require(isinstance(descriptor["dtype"], str) and descriptor["dtype"].strip(),
                    f"prepared {phase}/{name} dtype missing")
            require(descriptor["content_encoding"] == "logical-row-major-f32-le-v1",
                    f"prepared {phase}/{name} content encoding unrecognized")
            content_hash = descriptor["content_sha256"]
            require(isinstance(content_hash, str) and len(content_hash) == 64
                    and all(c in "0123456789abcdef" for c in content_hash),
                    f"prepared {phase}/{name} content hash missing/invalid")
    before, after = diagnostic["before"], diagnostic["after"]
    require(set(before["tensors"]) == set(after["tensors"]),
            "prepared evaluation tensor key sets changed between before and after")
    keys = ("shape", "strides", "dtype", "content_encoding", "content_sha256")
    for name in before["tensors"]:
        require(all(before["tensors"][name][key] == after["tensors"][name][key] for key in keys),
                f"actual prepared tensor changed before/after: {name}")
    require(after["loss"] < before["loss"], "fixed-evaluation loss did not strictly decrease")
    return diagnostic


def exact_native_f32(left, right, label):
    finite(left, label)
    finite(right, label)
    try:
        same = struct.pack("<f", left) == struct.pack("<f", right)
    except OverflowError as error:
        raise EvidenceError(f"{label} is not a finite native f32") from error
    require(same, f"normalized/native {label} bits differ")


def verify_native_run(root, run, evaluation_hash, manifest, canceled=False):
    # Each reference is a selected, directly serialized native RunReceipt, not a
    # hand-authored summary or a replacement for the retained full probe artifact.
    raw = read_json(checked_artifact(root, run["nativeReceipt"]))
    require(raw.get("cpu_reference") is False and raw.get("run_scope_returned") is True,
            "native receipt is CPU/fixture evidence or its scope did not return")
    require(raw.get("evaluation_manifest") == evaluation_hash, "raw native evaluation identity differs")
    gradients = run["gradients"]
    require(type(raw.get("gradient_tensors")) is int and raw.get("gradient_tensors") == gradients["tensorCount"] > 0
            and raw.get("gradients_finite") is gradients["allFiniteObserved"] is True
            and raw.get("nonzero_gradient") is gradients["nonzeroObserved"] is True,
            "gradient observations are not bound to actual native receipt")
    exact_native_f32(raw.get("evaluation_before"), run["learning"]["before"]["loss"], "evaluation-before")
    # Verified merged b939f3e88 diagnostic mapping (byte-identical to 160fc5de),
    # not the old nonexistent prepared_evaluations vector. Campaign source/pin
    # proof still requires the final selected revision to be inference/main-reachable.
    prepared = raw.get("prepared_evaluation")
    require(isinstance(prepared, dict) and isinstance(prepared.get("before"), dict),
            "native prepared-tensor observations unavailable; request/config echoes are not measurements")
    if not canceled:
        verify_optional_prepared_evaluation(raw, manifest)
    before = _normalizer._point(prepared["before"])
    require(before == run["learning"]["before"], "raw prepared-before observation differs")
    if canceled:
        require(raw.get("complete") is False and raw.get("evaluation_after") is None
                and prepared.get("after") is None,
                "canceled raw receipt was mislabeled completed")
        checkpoints = [item for item in raw.get("checkpoints", []) if item.get("state", {}).get("step") == 20]
        require(len(checkpoints) == 1, "actual native durable checkpoint 20 missing")
        native_delta = checkpoints[0]["factor_delta"]
    else:
        require(raw.get("complete") is True and raw.get("error") is None, "native completion/error mismatch")
        exact_native_f32(raw.get("evaluation_after"), run["learning"]["after"]["loss"], "evaluation-after")
        require(_normalizer._point(prepared["after"]) == run["learning"]["after"], "raw prepared-after observation differs")
        native_delta = raw.get("final_parameter_delta")
    delta = run["factorDelta"]
    require(isinstance(native_delta, dict), "actual native numeric factor delta is missing")
    mapping = {"tensorCount": "tensors", "elementCount": "elements", "changedElements": "changed_elements",
               "maxAbsDelta": "max_abs_delta", "squaredL2Delta": "squared_l2_delta",
               "layoutIdentity": "layout_sha256"}
    require(all(delta[key] == native_delta[native_key] for key, native_key in mapping.items()),
            "normalized factor metrics/layout differ from native observations")
    require(native_delta["initial_sha256"] == delta["startExport"]["sha256"]
            and native_delta["final_sha256"] == delta["finalExport"]["sha256"], "native factor exports unbound")
    start = raw.get("run_start_adapter")
    require(isinstance(start, dict) and start["sha256"] == delta["startExport"]["sha256"]
            and start["layout_sha256"] == delta["layoutIdentity"], "native start snapshot identity differs")
    frozen = raw.get("frozen", {})
    if not canceled:
        require(all(key in frozen and isinstance(frozen[key].get("before"), str)
                    and len(frozen[key]["before"]) == 64
                    and all(c in "0123456789abcdef" for c in frozen[key]["before"])
                    and frozen[key]["before"] == frozen[key]["after"]
                    for key in ("backbone", "text_encoder", "vae")), "actual native frozen tensor witnesses missing/different")
    else:
        require(all(item["before"] == item["after"] for item in frozen.values()),
                "observed canceled-run frozen component changed")


def verify_learning(learning, evaluation_hash):
    require(learning["manifestHash"] == evaluation_hash, "learning evaluation identity mismatch")
    before = finite(learning["before"]["loss"], "evaluation-before")
    after = finite(learning["after"]["loss"], "evaluation-after")
    require(after < before, "fixed-evaluation loss did not strictly decrease")
    require(learning["improvement"] == before - after, "reported loss improvement differs from actual pair")
    require(learning["before"]["prepared"] == learning["after"]["prepared"],
            "before/after prepared tensors or package identity differ")
    require(learning["sameActualPreparedPackageVerified"] is True
            and learning["reencodingBetweenMeasurements"] is False
            and learning["noiseRegenerationBetweenMeasurements"] is False,
            "prepared evaluation inputs were regenerated or not observed")
    return before, after


def verify_device(device, backend, manifest):
    fraction = manifest["resources"]["backends"][backend]["gpuMemoryLimitFraction"]
    require(device["gpuMemoryLimitFraction"] == fraction
            and device["capacityBasis"] == manifest["resources"]["backends"][backend]["capacityBasis"],
            "wrong frozen product fraction or actual-capacity basis")
    # Product MLX registration selects the default GPU using literal selector
    # "mlx"; the native probe additionally measures the default-GPU stream.
    # Preserve both observations instead of inventing a UUID or overwriting the
    # product selector with a supposedly identical measurement.
    same_device = device["claimedIdentity"] == device["observedIdentity"]
    if backend == "mlx" and device["claimedIdentity"] == "mlx":
        stream = device["observedIdentity"].removeprefix("mlx-default-gpu/stream-")
        same_device = device["observedIdentity"].startswith("mlx-default-gpu/stream-") and stream.isdecimal()
    require(same_device, "claimed/actual device identity differs")
    require(device["cpuReference"] is False and device["resolvedBeforeWeightsLoad"] is True,
            "CPU or post-load device observation cannot admit this case")
    expected_class = "apple-silicon-unified" if backend == "mlx" else "nvidia-cuda"
    require(device["class"] == expected_class, "actual device class differs from execution backend")
    total = finite(device["totalCapacityBytes"], "actual selected-device total", positive=True)
    # Preserve the product's f32 transport; this is representation, not a tolerance.
    product_fraction = struct.unpack("<f", struct.pack("<f", fraction))[0]
    ceiling = finite(device["resolvedCeilingBytes"], "resolved device ceiling", positive=True)
    require(ceiling <= math.floor(total * product_fraction), "resolved ceiling exceeds the actual product fraction")
    require(0 < device["projectedDemandBytes"] <= ceiling, "header-derived demand was not admitted")
    return ceiling


def verify_cross_load_clock(clock, limit):
    require(clock["clockSource"] == "cross-load-case-start" and clock["wallLimitSeconds"] == limit,
            "cross-load clock/bound must belong to its own import case")
    started, ended = timestamp(clock["phaseStartedAt"]), timestamp(clock["completedAt"])
    require(0 < ended - started <= limit
            and 0 < finite(clock["wallSeconds"], "cross-load wall time") <= limit,
            "cross-load independent import deadline violated")
    return started, ended


def verify_rss(guard, ceiling, started, ended):
    require(guard["positiveInitialMeasurement"] is True and guard["telemetryComplete"] is True
            and guard["missingMeasurementCount"] == 0, "missing native root RSS telemetry")
    rows = guard["rootRssSamples"]
    require(len(rows) >= 2, "native root RSS observations missing")
    previous = started
    for row in rows:
        at = timestamp(row["at"])
        require(started <= at <= ended and at >= previous, "RSS timestamp outside active phase or out of order")
        previous = at
        require(0 < row["bytes"] <= ceiling,
                "missing/nonpositive/over-cap native root RSS")
    process = guard["rootProcess"]
    require(guard["nativeRootReleased"] is True and process["releaseObserved"] is True
            and process["scope"] == "native-api-worker-root" and process["pid"] > 0
            and process["executable"] and re.fullmatch(r"[a-f0-9]{64}", process["binarySha256"])
            and process["metadataHelperAbsenceClaim"] is False,
            "native root identity/exit proof missing")


def decoded_png(file):
    from PIL import Image
    # Decode actual pixels with the standard image library. Metadata/job IDs may
    # legitimately differ; ordinary/disable controls compare exact RGBA pixels.
    with Image.open(file) as image:
        require(image.format == "PNG" and image.size == (256, 256), "not an actual fixed-canvas PNG")
        image.verify()
    with Image.open(file) as image:
        image.load()
        return image.size, image.convert("RGBA").tobytes()


def verify_png(root, png, strength, manifest):
    file = checked_artifact(root, png["artifact"])
    expected = manifest["previews"]
    require(decoded_png(file)[0] == (png["width"], png["height"])
            == (expected["width"], expected["height"]), "actual PNG canvas mismatch")
    require(png["adapterStrength"] == strength and png["seed"] == expected["seed"]
            and png["prompt"] == expected["prompt"], "preview strength/seed/prompt changed")
    checked_artifact(root, png["executionReceipt"])


def _artifact_references(value, location=()):
    if isinstance(value, dict):
        # Weight-source refs point to retained JSON file-identity proofs, never
        # copied multi-GB model payloads. They are still hash-verified artifacts.
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for key, item in value.items():
                yield from _artifact_references(item, location + (key,))
    elif isinstance(value, list):
        for item in value:
            yield from _artifact_references(item, location)


def verify_inspection_clock(completed_at, verification_time):
    """An inspection must already have happened, not merely fit a future case window."""
    at = timestamp(completed_at)
    require(at <= finite(verification_time, "verification clock"),
            "inspection completedAt is in the verification future")
    return at


def verify_record(record, root, manifest, expected_source, verification_time=None):
    if verification_time is None:
        verification_time = datetime.now(timezone.utc).timestamp()
    from jsonschema import Draft202012Validator, FormatChecker
    schema = read_json(Path(__file__).with_name("evidence.schema.json"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    require(record["recordKind"] == "campaign-acceptance-evidence"
            and record["isCampaignEvidence"] is True
            and record["acceptanceDisposition"] == "accepted",
            "schema-test/missing/unproven records are not acceptance evidence")
    require(not record["missing"] and not record["unproven"] and record["failureReason"] is None,
            "explicit missing/unproven/failure marker")
    require(record["manifestSha256"] == MANIFEST_SHA256, "campaign configuration changed")
    backend, parent, method = record["caseId"].split("/")
    identity = record["identity"]
    require((identity["backend"], identity["parent"], identity["method"]) == (backend, parent, method),
            "cell identity mismatch")
    source = identity["source"]
    require(source["sceneWorksSourceSha"] == expected_source["workflowSource"]
            and source["inferencePin"] == expected_source["inferencePin"], "source/pin binding mismatch")
    require(identity["workflow"]["fileSha256"] == expected_source["workflowSha256"], "workflow bytes mismatch")
    ceiling = verify_device(identity["device"], backend, manifest)
    for reference in _artifact_references(record):
        checked_artifact(root, reference)
    evaluation_hash = record["evaluationIdentity"]["manifestHash"]
    dataset_hash = record["datasetIdentity"]["manifestSha256"]
    config_hash = record["datasetIdentity"]["configSha256"]
    uploaded = record["datasetIdentity"]["uploadedItems"]
    wanted = {item["image"]: item["caption"] for item in manifest["dataset"]["items"]}
    require(len(uploaded) == len(wanted) == record["datasetIdentity"]["uploadedCount"], "uploaded dataset count mismatch")
    require({item["image"]: item["caption"] for item in uploaded} == wanted
            and len({item["uploadId"] for item in uploaded}) == len(wanted), "uploaded source captions/assets differ")
    require(next(item["imageSha256"] for item in uploaded if item["image"] == manifest["evaluation"]["image"])
            == record["evaluationIdentity"]["imageSha256"], "evaluation bytes differ from uploaded dataset")
    factor_audits = {}
    for name, run in record["subruns"].items():
        require(run["originalHorizon"] == 60 and run["seed"] == 1789
                and run["datasetManifestSha256"] == dataset_hash
                and run["configSha256"] == config_hash
                and run["evaluationManifestHash"] == evaluation_hash, "subrun recovery/evaluation identity changed")
        verify_native_run(root, run, evaluation_hash, manifest, canceled=(name == "canceled"))
        for loss in run.get("trainingLosses", []):
            finite(loss, "native training loss")
        gradients = run["gradients"]
        require(gradients["tensorCount"] > 0 and gradients["allFiniteObserved"] is True
                and gradients["nonzeroObserved"] is True, "finite/nonzero actual gradients missing")
        delta = run["factorDelta"]
        require((delta["parent"], delta["method"], delta["alpha"], delta["rank"])
                == (parent, method, manifest["training"]["config"]["alpha"], manifest["training"]["config"]["rank"])
                and delta["sourceSha"] == source["sceneWorksSourceSha"], "factor layout/alpha/source binding differs")
        actual = recompute_factor_delta(checked_artifact(root, delta["startExport"]),
                                        checked_artifact(root, delta["finalExport"]))
        # The hash-verified native snapshot streams F32 entries in sorted-name
        # order and excludes invariant .alpha scalars. Match that exact arithmetic;
        # do not invent an epsilon or relabel a metadata-only change as an update.
        for key in ("tensorCount", "elementCount", "changedElements", "maxAbsDelta", "squaredL2Delta"):
            require(actual[key] == delta[key], f"actual exported factor delta differs: {name}/{key}")
        factor_audits[name] = actual
        if name == "canceled":
            require(run["completedStep"] == 20 and run["checkpointDurable"] is True
                    and run["learning"]["after"] is None and run["learning"]["learningClaimed"] is False,
                    "expected cancellation was relabeled completion/learning")
        else:
            verify_learning(run["learning"], evaluation_hash)
            require(len(run["trainingLosses"]) == run["optimizerUpdatesInThisRun"],
                    "native update/loss observation coverage incomplete")
    reference, resumed = record["subruns"]["uninterrupted"], record["subruns"]["resumed"]
    tensor_payloads_equal(checked_artifact(root, reference["factorDelta"]["finalExport"]),
                          checked_artifact(root, resumed["factorDelta"]["finalExport"]))
    recovery = record["recovery"]
    require(recovery["originalCancelToken"] != recovery["resumedCancelToken"]
            and recovery["originalCancelToken"] == record["subruns"]["canceled"]["originalCancelToken"],
            "original/resumed cancellation token binding missing")
    stall = recovery["checkpointStall"]
    require(stall["waitedForTokenId"] == recovery["originalCancelToken"]
            and 0 <= timestamp(stall["exitedAt"]) - timestamp(stall["enteredAt"]) <= stall["timeoutSeconds"]
            and stall["updatesSkippedOrReplayed"] == 0 and stall["timedOut"] is False,
            "checkpoint stall deadlocked, timed out or changed updates")
    reentry = recovery["backendReentryWitness"]
    require(reentry["anchorCaseId"] == f"{backend}/base/lora"
            and reentry["originalCancelToken"] != reentry["freshRunCancelToken"], "backend re-entry token/anchor mismatch")
    # The producer's recovery witness contains actual complete reference/resumed state
    # snapshots in a normalized artifact, not a process-status assertion.
    state = read_json(checked_artifact(root, recovery["optimizerSchedulerRngState"]))
    require(set(state) == {"reference", "resumed"}, "complete recovery pair missing")
    state_root = checked_artifact(root, recovery["optimizerSchedulerRngState"]).parent
    for key in ("optimizer", "scheduler", "rng", "cursor", "partialAccumulation"):
        require(key in state["reference"] and key in state["resumed"], "complete recovery state observations missing")
        left, right = state["reference"][key], state["resumed"][key]
        if key == "optimizer":
            tensor_payloads_equal(checked_artifact(state_root, left), checked_artifact(state_root, right))
        else:
            require(left == right, f"actual recovered {key} differs")
    saved = record["saveReloadGenerate"]
    require(saved["adapterSha256"] == saved["resolvedSha256"] == reference["factorDelta"]["finalExport"]["sha256"],
            "library resolution did not select final exported adapter")
    for role, strength in (("before", 1.0), ("after", 1.0), ("disableControl", 0.0)):
        verify_png(root, saved[role], strength, manifest)
    require(decoded_png(checked_artifact(root, saved["disableControl"]["artifact"]))
            == decoded_png(checked_artifact(root, saved["noAdapterBaseline"])),
            "disable control did not reproduce ordinary baseline pixels")
    cross = record["crossBackendLoad"]
    require(cross["sourceCaseId"] == record["caseId"] and cross["sourceBackend"] == backend
            and cross["importedBackend"] == ("cuda" if backend == "mlx" else "mlx")
            and cross["parent"] == parent and cross["method"] == method
            and cross["receivedSha256"] == cross["transferredAdapter"]["sha256"] == saved["adapterSha256"],
            "cross-backend original cell/artifact binding mismatch")
    cross_clock = cross["phaseTiming"]
    cs, ce = verify_cross_load_clock(cross_clock, manifest["limits"]["crossLoadWallSeconds"])
    cross_backend = cross["importedBackend"]
    cross_ceiling = verify_device(cross["resourceGuard"]["device"], cross_backend, manifest)
    verify_rss(cross["resourceGuard"], cross_ceiling, cs, ce)
    if cross["outcome"] == "generated":
        verify_png(root, cross["generation"], 1.0, manifest)
    else:
        # The currently frozen native portable LoRA/LoKR exports permit both backends.
        # Retain a truthful refusal record, but never turn it into campaign acceptance.
        raise EvidenceError(f"cross-backend loading remains unproven: {cross['unsupportedFormatRefusal']}")
    refusal = record["memoryRefusal"]
    require(0 < refusal["resolvedControlCapBytes"] < refusal["projectedHeaderDemandBytes"]
            and refusal["publishedOutput"] is False and not refusal["publishedOutputInventory"],
            "infeasible low-cap/no-publication witness missing")
    cleanup = record["cleanup"]
    started, ended = timestamp(cleanup["caseStartedAt"]), timestamp(cleanup["caseEndedAt"])
    require(0 < ended - started <= manifest["limits"]["caseWallSeconds"]
            and 0 < finite(cleanup["caseWallSeconds"], "training-case wall time") <= manifest["limits"]["caseWallSeconds"],
            "training/round-trip case deadline violated")
    require(cleanup["resolvedCeilingBytes"] == ceiling, "RSS/native ceiling binding mismatch")
    verify_rss(cleanup, ceiling, started, ended)
    inspected = set()
    for inspection in record["inspections"]:
        at = verify_inspection_clock(inspection["completedAt"], verification_time)
        require(started <= at and inspection["beforeFinalAcceptance"] is True, "inspection precedes capture or final acceptance requirement is missing")
        checked_artifact(root, inspection["reviewedArtifact"])
        inspected.add((inspection["reviewedArtifact"]["path"], inspection["reviewedArtifact"]["sha256"]))
    require(all((saved[role]["artifact"]["path"], saved[role]["artifact"]["sha256"]) in inspected
                for role in ("before", "after", "disableControl")), "named preview inspection missing")
    regression = record["regression"]
    require(decoded_png(checked_artifact(root, regression["ordinary"]["beforeNoAdapter"]))
            == decoded_png(checked_artifact(root, regression["ordinary"]["afterUnload"])),
            "ordinary generation pixels changed after unload")
    for execution in regression["dmdExecutions"]:
        require(execution["selectedMode"] == execution["executedMode"], "DMD selected/executed mode differs")
        verify_png(root, execution["generation"], 1.0, manifest)
    return factor_audits


def verify_bundle(root, phase, backend, expected_source=None):
    bundle = Path(root) / f"{phase}-{backend}"
    receipt = read_json(bundle / "driver.json")
    require(receipt["ok"] is True and receipt["phase"] == phase and receipt["backend"] == backend,
            f"incomplete {phase}/{backend} bundle")
    source = receipt["source"]
    require(source["manifestSha256"] == MANIFEST_SHA256 and source.get("finalMergedSource") is True,
            "bundle manifest/final merged source proof mismatch")
    if expected_source is not None:
        require(source == expected_source, "source binding differs across backend/phase bundles")
    source_backend = ("cuda" if backend == "mlx" else "mlx") if phase == "cross-load" else backend
    expected = [case for case in CELLS if case.startswith(source_backend + "/")]
    require(isinstance(receipt.get("cases"), list)
            and [case.get("caseId") for case in receipt["cases"]] == expected
            and all(case.get("executionPassed") is True for case in receipt["cases"]),
            "all four exact completed bundle cases required")
    listed = receipt["members"]
    require(isinstance(listed, list) and len({x["path"] for x in listed}) == len(listed), "duplicate/missing bundle members")
    actual = set()
    for file in bundle.rglob("*"):
        require(not file.is_symlink(), "bundle contains symlink")
        if file.is_file() and file != bundle / "driver.json":
            actual.add(file.relative_to(bundle).as_posix())
    require(actual == {x["path"] for x in listed}, "bundle member list differs from actual files")
    for reference in listed:
        checked_artifact(bundle, reference)
    return bundle, receipt


def _rebase_artifacts(value, prefix):
    # Normalization can reuse one reference in several fields. Rebuild each
    # occurrence so aliases receive one prefix and producer observations stay local.
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return {"path": f"{prefix}/{value['path']}", "sha256": value["sha256"]}
        return {key: _rebase_artifacts(item, prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_artifacts(item, prefix) for item in value]
    return value


def controller_observations(bundle, receipt, entry, phase):
    """Bind native root RSS to the parent's exact launched binary/exit/stdio receipt."""
    _, parent, method = entry["caseId"].split("/")
    log = f"case-{parent}-{method}.log"
    commands = [command for command in receipt["commands"] if command["log"] == log]
    require(len(commands) == 1, "exact cell command supervision missing/duplicated")
    command = commands[0]
    state = command["supervision"]
    require(command["exitCode"] == 0 and state.get("cleanup", {}).get("released") is True
            and not state.get("unsafe") and not state.get("cancelled"),
            "parent-owned native root release/cancellation proof failed")
    require(entry["counts"] == {"passed": 1, "failed": 0, "ignored": 0, "exit_code": 0},
            "exact native test did not execute once positively")
    case_file = member_path(bundle, entry["rawEvidence"])
    preflight_file = case_file.parent / "devicepreflight.json"
    preflight = read_json(preflight_file)
    require(preflight["caseId"] == entry["caseId"] and preflight["backend"] == receipt["backend"]
            and preflight["phase"] == phase
            and preflight["verdict"] == "admitted" and preflight["refusalReason"] is None
            and preflight["offline"] is True and preflight["weightsLoaded"] is False,
            "exact per-cell preweight admission proof missing")
    require(preflight["modelRevisions"] and preflight["resolvedCapProvenance"]
            and preflight["headerDemandProvenance"] and preflight["weights"]
            and all(weight["present"] is True for weight in preflight["weights"]),
            "required model revisions, present weights or native demand provenance missing")
    for key in ("workflowSource", "inferencePin", "manifestSha256"):
        require(preflight[key] == receipt["source"][key], "preweight source binding mismatch")
    native = read_json(case_file)
    raw = native.get("raw", {})
    measured = raw.get("rootProcess", {})
    process = state.get("rootProcess", {})
    require(process.get("scope") == "native-api-worker-root"
            and measured.get("scope") == "native-api-worker-root"
            and isinstance(measured.get("pid"), int) and measured["pid"] > 0
            and measured["pid"] == process.get("pid")
            and process.get("executable") == receipt["executable"]["path"]
            and process.get("binarySha256") == receipt["executable"]["sha256"]
            and raw.get("serverJoined") is True
            and isinstance(raw.get("nativeBodyReturnedAtUnixMs"), int)
            and state["cleanup"].get("scope") == "native-api-worker-root"
            and state["cleanup"].get("stdioClosed") is True,
            "native root PID/binary/joins/stdio binding missing")
    require(measured.get("measurement") == "native-current-process-rss-sampled-maximum"
            and measured.get("metadataHelperAbsenceClaim") is False
            and state["cleanup"].get("metadataHelperAbsenceClaim") is False
            and measured.get("firstFailure") is None
            and measured.get("missingMeasurementCount") == 0
            and measured.get("resolvedCeilingBytes") == preflight["resolvedCapBytes"],
            "native root RSS scope/cap/completeness differs")
    rows = measured.get("samples")
    require(isinstance(rows, list) and len(rows) >= 2
            and all(type(row.get("bytes")) is int and row["bytes"] > 0 for row in rows)
            and measured.get("sampledPeakBytes") == max(row["bytes"] for row in rows),
            "native root sampled maximum missing or differs")
    process = {"scope":"native-api-worker-root", "pid":process["pid"],
               "executable":process["executable"], "binarySha256":process["binarySha256"],
               "releaseObserved":True,"metadataHelperAbsenceClaim":False}
    driver_reference = {"path": f"{phase}-{receipt['backend']}/driver.json",
                        "sha256": digest(Path(bundle) / "driver.json")}
    cleanup = {"rootProcess": process, "nativeRootReleased": True,
               "releaseReceipt": driver_reference, "rootRssSamples": rows,
               "positiveInitialMeasurement": True, "telemetryComplete": True,
               "missingMeasurementCount": 0, "resolvedCeilingBytes": preflight["resolvedCapBytes"],
               "caseStartedAt": entry["caseStartedAt"], "caseEndedAt": command["finishedAt"],
               "caseWallSeconds": timestamp(command["finishedAt"]) - timestamp(entry["caseStartedAt"]),
               "caseDeadlineSeconds": 3600,
               "buildPreparationWallSeconds": timestamp(receipt["preparationFinishedAt"])
                   - timestamp(receipt["preparationStartedAt"]),
               "buildPreparationDeadlineSeconds": 7200}
    return cleanup, preflight, command


def verify_workflow_provenance(receipt, capture_run):
    """Same-run retries preserve each producing bundle's actual attempt/job."""
    workflow = receipt.get("workflow")
    require(isinstance(workflow, dict), "actual driver workflow provenance missing")
    require(workflow.get("runId") == capture_run, "bundle belongs to a different capture workflow run")
    attempt = workflow.get("attempt")
    require(type(attempt) is int and attempt > 0, "actual workflow attempt missing/invalid")
    require(workflow.get("sha") == receipt["source"]["workflowSource"]
            and workflow.get("fileSha256") == receipt["source"]["workflowSha256"]
            and isinstance(workflow.get("ref"), str) and workflow["ref"]
            and isinstance(workflow.get("jobId"), str) and workflow["jobId"],
            "workflow source/ref/file/job identity mismatch")
    return attempt


def retrieve_workflow_jobs(capture_run, request_json=None):
    """Actions-read metadata only: all attempts, at most ten pages per resource."""
    require(isinstance(capture_run, str) and capture_run.isdecimal() and int(capture_run) > 0,
            "invalid capture run for job retrieval")
    if request_json is None:
        token = os.environ.get("MOSS_ACTIONS_READ_TOKEN")
        require(token, "Actions-read token required to authenticate capture job attempts")

        def request_json(url):
            request = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28"})
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read(16 * 1024 * 1024 + 1)
            require(len(body) <= 16 * 1024 * 1024, "GitHub jobs response exceeds metadata bound")
            return json.loads(body, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)

    started = time.monotonic()
    result = {"repository": "SceneWorks/SceneWorks", "runId": capture_run,
              "retrievedAt": datetime.now(timezone.utc).isoformat(), "attempts": []}
    def pages(resource, query):
        rows = []
        for page in range(1, 11):
            require(time.monotonic() - started < 120, "GitHub provenance authentication deadline exceeded")
            url = (f"https://api.github.com/repos/SceneWorks/SceneWorks/actions/runs/{capture_run}"
                   f"/{resource}?{query}per_page=100&page={page}")
            response = request_json(url)
            require(time.monotonic() - started < 120, "GitHub provenance authentication deadline exceeded")
            current = response.get(resource)
            require(isinstance(current, list) and len(current) <= 100, "invalid GitHub provenance page")
            rows.extend(current)
            if len(current) < 100:
                require(response.get("total_count") == len(rows), "incomplete GitHub provenance inventory")
                break
        else:
            raise EvidenceError("GitHub provenance inventory exceeds ten-page bound")
        require(len({row["id"] for row in rows}) == len(rows), "duplicate GitHub provenance identities")
        return rows
    jobs = pages("jobs", "filter=all&")
    attempts = {job.get("run_attempt") for job in jobs}
    require(all(type(attempt) is int and attempt > 0 for attempt in attempts), "GitHub job attempt missing")
    result["attempts"] = [{"attempt": attempt, "jobs": [job for job in jobs if job["run_attempt"] == attempt]}
                          for attempt in sorted(attempts)]
    result["artifacts"] = pages("artifacts", "")
    return result


def authenticate_bundle_jobs(bundles, capture_run, inventory):
    require(inventory.get("repository") == "SceneWorks/SceneWorks" and inventory.get("runId") == capture_run,
            "GitHub job inventory belongs to another repository/run")
    authenticated = []
    for (phase, backend), (_, receipt) in bundles.items():
        attempt = verify_workflow_provenance(receipt, capture_run)
        require(receipt["workflow"]["jobId"] == phase, "bundle logical job identity differs from its phase")
        selected = [item for item in inventory["attempts"] if item["attempt"] == attempt]
        require(len(selected) == 1, "actual bundle attempt missing/duplicated in GitHub jobs")
        suffix = f"MOSS {phase} {backend}"
        executions = [job for item in inventory["attempts"] for job in item["jobs"]
                      if (job.get("name") == suffix or job.get("name", "").endswith(" / " + suffix))
                      and job.get("conclusion") != "skipped"]
        require(executions and attempt == max(job["run_attempt"] for job in executions),
                "newer actual backend execution invalidates this older artifact")
        jobs = [job for job in selected[0]["jobs"]
                if job.get("name") == suffix or job.get("name", "").endswith(" / " + suffix)]
        require(len(jobs) == 1, "exact backend job name missing/ambiguous in GitHub attempt")
        job = jobs[0]
        require(type(job.get("id")) is int and job["id"] > 0
                and str(job.get("run_id")) == capture_run and job.get("run_attempt") == attempt
                and job.get("head_sha") == receipt["source"]["workflowSource"]
                and job.get("status") == "completed" and job.get("conclusion") == "success",
                "GitHub job run/attempt/head/completed-success identity mismatch")
        if receipt.get("runnerName"):
            require(job.get("runner_name") == receipt["runnerName"], "GitHub job runner identity differs")
        authenticated.append({"bundle": f"{phase}-{backend}", "attempt": attempt, "jobId": job["id"]})
    return authenticated


def authenticate_bundle_artifacts(bundles, capture_run, inventory, selection):
    require(selection.get("runId") == capture_run, "artifact selection belongs to another run")
    listed = selection.get("bundles", [])
    require(len(listed) == len(bundles) and len({item["bundle"] for item in listed}) == len(bundles),
            "missing/duplicate selected artifact bindings")
    authenticated = []
    for (phase, backend), (directory, receipt) in bundles.items():
        name = f"{phase}-{backend}-attempt-{receipt['workflow']['attempt']}"
        selected = [item for item in listed if item["bundle"] == f"{phase}-{backend}"]
        require(len(selected) == 1 and selected[0]["artifactName"] == name
                and selected[0]["attempt"] == receipt["workflow"]["attempt"]
                and selected[0]["driverSha256"] == digest(directory / "driver.json"),
                "selected artifact name/attempt/driver hash differs from consumed bundle")
        rows = [row for row in inventory["artifacts"] if row.get("name") == name]
        require(len(rows) == 1, "selected GitHub artifact missing/ambiguous")
        artifact = rows[0]
        require(type(artifact.get("id")) is int and artifact["id"] > 0 and artifact.get("expired") is False
                and str(artifact.get("workflow_run", {}).get("id")) == capture_run
                and artifact["workflow_run"].get("head_sha") == receipt["source"]["workflowSource"],
                "GitHub artifact run/head/availability mismatch")
        authenticated.append({"bundle": f"{phase}-{backend}", "artifactId": artifact["id"],
                              "artifactName": name, "attempt": receipt["workflow"]["attempt"],
                              "driverSha256": selected[0]["driverSha256"], "digest": artifact.get("digest")})
    return authenticated


def verify_capture_dependencies(bundles):
    for (phase, backend), (_, receipt) in bundles.items():
        if phase != "cross-load":
            continue
        source_backend = "cuda" if backend == "mlx" else "mlx"
        source_bundle, _ = bundles[("capture", source_backend)]
        require(receipt.get("sourceCapture") == {"backend": source_backend,
                "driverSha256": digest(source_bundle / "driver.json")},
                "cross-load consumed a different/replaced capture bundle")


def verify_inspection_capture_binding(review, capture):
    require(review.get("captureWorkflow") == capture["workflow"],
            "inspection belongs to a different capture attempt/job/source")


def inspection_set(root, verification_time=None):
    if verification_time is None:
        verification_time = datetime.now(timezone.utc).timestamp()
    encoded = os.environ.get("MOSS_INSPECTION_JSON")
    if encoded:
        document = json.loads(encoded, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
    else:
        file = Path(root) / "inspections.json"
        require(file.is_file(), "named preview inspection document is missing")
        document = read_json(file)
    require(document.get("manifestSha256") == MANIFEST_SHA256, "inspection manifest seal mismatch")
    capture_run = document.get("captureRunId")
    require(isinstance(capture_run, str) and capture_run.isdecimal() and int(capture_run) > 0,
            "inspection must name the original capture run")
    selected_run = os.environ.get("MOSS_CAPTURE_RUN_ID")
    require(not selected_run or selected_run == capture_run, "inspection/capture run selection mismatch")
    cases = document.get("cases")
    require(isinstance(cases, list) and len(cases) == 8, "all eight case inspections required")
    by_id = {}
    for case in cases:
        require(case.get("caseId") in CELLS and case["caseId"] not in by_id,
                "unexpected/duplicate inspection cell")
        require(isinstance(case.get("inspections"), list) and len(case["inspections"]) >= 3,
                "named per-case preview inspections missing")
        for inspection in case["inspections"]:
            verify_inspection_clock(inspection["completedAt"], verification_time)
        by_id[case["caseId"]] = case
    return capture_run, by_id


def finalize(root, manifest_file, output_file, selection_file):
    from jsonschema import Draft202012Validator
    # One actual clock for the whole verification, before copying/hash work can
    # let a future-dated declaration become current during a slow finalization.
    verification_time = datetime.now(timezone.utc).timestamp()
    root, manifest_file, output_file = Path(root), Path(manifest_file), Path(output_file)
    require(digest(manifest_file) == MANIFEST_SHA256, "sealed manifest changed")
    manifest = read_json(manifest_file)
    Draft202012Validator(read_json(manifest_file.with_name("campaign.schema.json"))).validate(manifest)
    require(not output_file.exists(), "never overwrite/carry a prior verification")
    destination = output_file.parent
    require(not destination.resolve().is_relative_to(root.resolve())
            and not root.resolve().is_relative_to(destination.resolve()),
            "verification output and downloaded artifact roots must be separate")
    records_dir = destination / "accepted-records"
    require(not records_dir.exists(), "accepted-record directory must be fresh")
    records_dir.mkdir(parents=True)
    source = None
    bundles = {}
    for phase in ("capture", "cross-load"):
        for backend in ("mlx", "cuda"):
            bundle, receipt = verify_bundle(root, phase, backend, source)
            source = receipt["source"] if source is None else source
            bundles[(phase, backend)] = (bundle, receipt)
    # Work only in fresh output. Preserve downloaded input bundles byte-for-byte;
    # controller-derived post-exit observations must never mutate sealed raw input.
    for phase in ("capture", "cross-load"):
        for backend in ("mlx", "cuda"):
            shutil.copytree(root / f"{phase}-{backend}", destination / f"{phase}-{backend}")
            verify_bundle(destination, phase, backend, source)
    capture_run, reviews = inspection_set(root, verification_time)
    for _, receipt in bundles.values():
        verify_workflow_provenance(receipt, capture_run)
    inventory = retrieve_workflow_jobs(capture_run)
    jobs_file = destination / "github-jobs.json"
    require(not jobs_file.exists(), "never replace job authentication provenance")
    jobs_file.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    authenticated_jobs = authenticate_bundle_jobs(bundles, capture_run, inventory)
    authenticated_artifacts = authenticate_bundle_artifacts(bundles, capture_run, inventory, read_json(selection_file))
    verify_capture_dependencies(bundles)
    records, summaries, accepted_payloads = [], [], []
    common_evaluation = common_dataset = None
    reentries = {}
    for case_id in CELLS:
        backend, parent, method = case_id.split("/")
        _, capture = bundles[("capture", backend)]
        verify_inspection_capture_binding(reviews[case_id], capture)
        captured_bundle = destination / f"capture-{backend}"
        imported_backend = "cuda" if backend == "mlx" else "mlx"
        _, cross_receipt = bundles[("cross-load", imported_backend)]
        cross_bundle = destination / f"cross-load-{imported_backend}"
        capture_entry = next((x for x in capture["cases"] if x["caseId"] == case_id), None)
        cross_entry = next((x for x in cross_receipt["cases"] if x["caseId"] == case_id), None)
        require(capture_entry and cross_entry and capture_entry["executionPassed"] and cross_entry["executionPassed"],
                f"missing exact capture/cross-load cell: {case_id}")
        case_file = member_path(captured_bundle, capture_entry["rawEvidence"])
        raw = read_json(case_file)
        cross_file = member_path(cross_bundle, cross_entry["rawEvidence"])
        cross_raw = read_json(cross_file)
        require(raw["caseId"] == cross_raw["caseId"] == case_id, "raw source-cell identity mismatch")
        # Normalize only trusted producer format into the fresh output mirror.
        # Derivatives never modify the original, hash-inventoried input bundles.
        record = _normalizer.normalize_capture(raw, case_file.parent, manifest, capture, capture_entry)
        cross_record = _normalizer.normalize_cross(cross_raw, cross_file.parent, manifest, cross_receipt, cross_entry)
        require(cross_record.get("failureReason") is None and cross_record.get("crossBackendLoad") is not None
                and not cross_record.get("unproven")
                and set(cross_record.get("missing", [])) <= {"resourceGuard", "cleanup", "inspections"},
                "cross-load retains missing native observations")
        record["evaluationIdentity"]["sharedAcrossAllCellsAndSubruns"] = True
        # This is a provisional in-memory global claim. It is published only
        # after the all-eight equality comparison below passes for every case.
        record["unproven"] = [marker for marker in record["unproven"]
                              if marker != "evaluationIdentity.sharedAcrossAllCellsAndSubruns"]
        external_cleanup, capture_preflight, capture_command = controller_observations(captured_bundle, capture, capture_entry, "capture")
        cross_cleanup, cross_preflight, cross_command = controller_observations(cross_bundle, cross_receipt, cross_entry, "cross-load")
        require(record["identity"]["toolchain"]["binarySha256"] == capture["executable"]["sha256"],
                "native case is not bound to the actual compiled test executable")
        require(record["identity"]["device"]["observedIdentity"] == capture_preflight["deviceIdentity"]
                and record["identity"]["device"]["resolvedCeilingBytes"] == capture_preflight["resolvedCapBytes"],
                "actual execution/preflight device or ceiling differs")
        record["crossBackendLoad"] = cross_record["crossBackendLoad"]
        record["crossBackendLoad"]["resourceGuard"] = {"device": cross_record["device"]}
        capture_prefix = case_file.parent.relative_to(destination).as_posix()
        cross_prefix = cross_file.parent.relative_to(destination).as_posix()
        # Artifact paths are relative to the producing case directory. The peer
        # cross-load piece is rebased separately, preserving original cell identity.
        cross_piece = record.pop("crossBackendLoad")
        record = _rebase_artifacts(record, capture_prefix)
        cross_piece = _rebase_artifacts(cross_piece, cross_prefix)
        record["crossBackendLoad"] = cross_piece
        record["cleanup"] = external_cleanup
        guard = cross_piece["resourceGuard"]
        require(guard["device"]["observedIdentity"] == cross_preflight["deviceIdentity"]
                and guard["device"]["resolvedCeilingBytes"] == cross_preflight["resolvedCapBytes"],
                "cross-load actual/preflight device or ceiling differs")
        for key in ("rootProcess", "nativeRootReleased", "releaseReceipt", "rootRssSamples",
                    "positiveInitialMeasurement", "telemetryComplete", "missingMeasurementCount"):
            guard[key] = cross_cleanup[key]
        started = cross_entry["caseStartedAt"]
        ended = cross_command["finishedAt"]
        cross_piece["phaseTiming"] = {"phaseStartedAt": started, "completedAt": ended,
                                    "wallSeconds": timestamp(ended) - timestamp(started),
                                    "wallLimitSeconds": manifest["limits"]["crossLoadWallSeconds"],
                                    "clockSource": "cross-load-case-start"}
        require(record["identity"]["workflow"]["runId"] == capture_run,
                "capture run identity differs from reviewed artifact run")
        record["inspections"] = reviews[case_id]["inspections"]
        # Resolve only actual external post-exit cleanup and deliberately later
        # cross-load/inspection observations. Never erase native measurement gaps.
        require(isinstance(record.get("missing"), list) and isinstance(record.get("unproven"), list),
                "raw missing/unproven marker lists are required")
        resolved_markers = set(record["missing"] + record["unproven"])
        require(resolved_markers <= {"crossBackendLoad", "inspections", "cleanup"}
                and record.get("failureReason") is None and record.get("acceptanceDisposition") != "failed",
                f"raw evidence retains unresolved native observations/failure: {case_id}")
        record["missing"] = []
        record["unproven"] = []
        require(record.get("recordKind") == "campaign-acceptance-evidence"
                and record.get("isCampaignEvidence") is True, "fixture/foreign raw record rejected")
        record["acceptanceDisposition"] = "accepted"
        audits = verify_record(record, destination, manifest, source, verification_time)
        evaluation = record["evaluationIdentity"]
        dataset = record["datasetIdentity"]["manifestSha256"]
        if common_evaluation is None:
            common_evaluation, common_dataset = evaluation, dataset
        require(evaluation == common_evaluation and dataset == common_dataset,
                "evaluation source identity/dataset differs across cells")
        witness = copy.deepcopy(record["recovery"]["backendReentryWitness"])
        # One witness may be archived in multiple case bundles. Its content hash
        # and observed token/run identity, not the copy's relative path, bind it.
        witness["witness"] = {"sha256": witness["witness"]["sha256"]}
        if backend in reentries:
            require(witness == reentries[backend], "more than one/different backend re-entry witness")
        reentries[backend] = witness
        file = records_dir / f"{backend}-{parent}-{method}.json"
        accepted_payloads.append((file, record))
        records.append(file.relative_to(destination).as_posix())
        summaries.append({"caseId": case_id, "independentFactorChecks": audits,
                          "resolvedLaterPhaseMarkers": sorted(resolved_markers)})
    require(set(reentries) == {"mlx", "cuda"}, "both backend re-entry witnesses required")
    # Publish accepted records only after the entire eight-cell set passed.
    for file, record in accepted_payloads:
        file.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    result = {"accepted": True, "acceptedCaseCount": len(records), "manifestSha256": MANIFEST_SHA256,
              "records": records, "source": source, "semanticChecks": summaries,
              "authenticatedBundleJobs": authenticated_jobs, "authenticatedBundleArtifacts": authenticated_artifacts,
              "githubJobs": {"path": jobs_file.name, "sha256": digest(jobs_file)}}
    output_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", required=True)
    options = parser.parse_args()
    output = Path(options.output)
    try:
        finalize(options.root, options.manifest, output, options.selection)
    except Exception as error:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            output.write_text(json.dumps({"accepted": False, "acceptedCaseCount": 0,
                                          "records": [], "error": str(error)}, indent=2) + "\n")
        raise SystemExit(f"MOSS acceptance verification failed closed: {error}") from error


if __name__ == "__main__":
    main()
