"""Normalize local observations emitted by moss_training_acceptance.rs.

Merged reference b939f3e883b75d3ec5e12263bde914ae5f5b2830, tree
ba0a927f0799c351cee248264276edc4cb3754ab. All supplied file hashes and the
recorded relied-on hashes were reverified; probe SHA256
1f0148ffa9414ef1c15c8875928d1c2b9c0ffc5888a5da22107d36f911e2d5ec
is identical to PR #1344's 160fc5de bytes. This is a format adapter,
not an acceptance verifier. It never loads a model, changes an oracle,
marks a case accepted, supplies process-release evidence, or invents inspections.

Public entry points:
    normalize_capture(raw, caseRoot, manifest, driverReceipt, caseEntry)
    normalize_cross(raw, caseRoot, manifest, driverReceipt, caseEntry)

Capture returns a normal-form candidate. Cross returns {crossBackendLoad, device,
missing, unproven, failureReason}; its resourceGuard must come from the controller.
All artifact references returned are relative to caseRoot, ready for the main
finalizer's existing bundle-prefix rebasing. Derived JSON and adapter-optimizer
copies are written beneath caseRoot/normalized-native, never into the HF cache.
No imports from the verifier: loading this module cannot create an import cycle.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

RAW_FORMAT = "moss-native-api-acceptance-candidate/v1"
SEAL = "cad2d90a22ccda3f69f74103386efa49e9d2c89d76e65d8b92195e201a4c8296"
ENCODING = "logical-row-major-f32-le-v1"


class NormalizationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise NormalizationError(message)


def _finite(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"missing/nonfinite {label}")
    require(not positive or value > 0, f"nonpositive {label}")
    return value


def _integer(value, label, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), f"invalid {label}")
    return value


def _text(value, label):
    require(isinstance(value, str) and value, f"missing {label}")
    return value


def _basename(value):
    # The finalizer may run on Linux while the actual executed CUDA path was
    # Windows. Interpret a recorded basename, never resolve that external path.
    return _text(value, "actual adapter path").replace("\\", "/").rsplit("/", 1)[-1]


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key}")
        result[key] = value
    return result


def _invalid(value):
    raise NormalizationError(f"nonfinite JSON constant {value}")


def _read(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=_pairs, parse_constant=_invalid)


def _at_ms(value):
    _integer(value, "observed Unix milliseconds", True)
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _Context:
    def __init__(self, root):
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.is_dir() and not self.root.is_symlink(),
                "normalization requires the actual absolute owned case directory")

    def path(self, relative, exists=True):
        require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
                "unsafe artifact reference")
        parts = PurePosixPath(relative)
        require(not parts.is_absolute() and ".." not in parts.parts, "artifact escaped case root")
        path = self.root
        for part in parts.parts:
            path = path / part
            require(not path.is_symlink(), f"artifact symlink forbidden: {relative}")
        if exists:
            require(path.is_file(), f"missing artifact: {relative}")
        return path

    def checked(self, reference):
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "unbound artifact")
        path = self.path(reference["path"])
        require(_digest(path) == reference["sha256"], f"artifact bytes changed: {reference['path']}")
        return path

    def read(self, reference):
        return _read(self.checked(reference))

    def reference(self, path):
        path = Path(path)
        relative = path.relative_to(self.root).as_posix()
        self.path(relative)
        return {"path": relative, "sha256": _digest(path)}

    def put(self, name, value):
        encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        relative = f"normalized-native/{name}"
        path = self.path(relative, exists=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(path.is_file() and path.read_bytes() == encoded, "normalization would overwrite different observations")
        else:
            with path.open("xb") as target:
                target.write(encoded)
        return {"path": relative, "sha256": hashlib.sha256(encoded).hexdigest()}

    def copy(self, name, reference):
        source = self.checked(reference)
        relative = f"normalized-native/{name}"
        target = self.path(relative, exists=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(_digest(target) == reference["sha256"], "derived adapter artifact collision")
        else:
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=65536)
        require(_digest(target) == reference["sha256"], "streamed adapter-state copy changed")
        return {"path": relative, "sha256": reference["sha256"]}


def _base(raw, entry, phase):
    require(raw.get("rawFormat") == RAW_FORMAT and raw.get("manifestSha256") == SEAL,
            "unsupported/unsealed raw native candidate")
    require(raw.get("recordKind") == "campaign-acceptance-evidence" and raw.get("isCampaignEvidence") is True
            and raw.get("fixtureLabel") is None, "schema-test/fixture evidence cannot be promoted by normalization")
    require(raw.get("caseId") == entry["caseId"] == raw.get("sourceCaseId"), "raw/driver case identity differs")
    parts = raw["caseId"].split("/")
    require(len(parts) == 3 and parts[0] in ("mlx", "cuda") and parts[1] in ("base", "sft")
            and parts[2] in ("lora", "lokr"), "raw case is outside the sealed eight-cell inventory")
    require(raw.get("executionBackend") in ("mlx", "cuda")
            and ((raw["executionBackend"] == parts[0]) == (phase == "capture")), "actual execution/source phase mismatch")
    require(raw.get("phase") == phase and isinstance(raw.get("raw"), dict), "raw phase missing")
    return {"schemaVersion": 1, "recordKind": "campaign-acceptance-evidence", "fixtureLabel": None,
            "isCampaignEvidence": True, "acceptanceDisposition": "missing",
            "missing": ["cleanup", "crossBackendLoad", "inspections"], "unproven": [],
            "failureReason": raw.get("failureReason"), "manifestSha256": SEAL, "caseId": raw["caseId"]}


def _device(ctx, raw, cpu_reference):
    p = raw["raw"]["devicePreflight"]
    require(p["caseId"] == raw["caseId"] and p["executionBackend"] == raw["executionBackend"]
            and p["verdict"] == "admitted" and p["refusalReason"] is None
            and p["weightsLoaded"] is False and p["offline"] is True, "no actual preweight admission")
    backend = raw["executionBackend"]
    require(backend in ("mlx", "cuda") and cpu_reference is False, "CPU is not real GPU evidence")
    ceiling = _integer(p["resolvedCapBytes"], "actual resolved cap", True)
    total = _integer(p["totalCapacityBytes"], "actual device total", True)
    projected = _integer(p["headerDerivedDemandBytes"], "native projected demand", True)
    require(projected <= ceiling <= total, "actual native demand exceeds frozen capacity")
    claimed = _text(p["claimedDeviceIdentity"], "measured claimed device")
    observed = _text(p["deviceIdentity"], "allocator-selected device")
    require((claimed == "mlx" and observed.startswith("mlx-default-gpu/stream-")) if backend == "mlx"
            else claimed.lower() == observed.lower(), "claimed/actual allocator device differs")
    return {"class": "apple-silicon-unified" if backend == "mlx" else "nvidia-cuda",
            "claimedIdentity": claimed, "observedIdentity": observed,
            "totalCapacityBytes": total, "gpuMemoryLimitFraction": p["gpuMemoryLimitFraction"],
            "capacityBasis": "actual-detected-total-unified-memory" if backend == "mlx"
                else "actual-claimed-cuda-context-total-vram",
            "resolvedCeilingBytes": ceiling, "resolvedBeforeWeightsLoad": p["weightsLoaded"] is False,
            "projectedDemandBytes": projected,
            "headerDemandReceipt": ctx.reference(ctx.path("devicepreflight.json")), "cpuReference": cpu_reference}


def _weight_sources(ctx, raw):
    grouped = {}
    for index, weight in enumerate(raw["raw"]["devicePreflight"]["weights"]):
        require(weight["present"] is True and weight["trustedHashMatches"] is True
                and weight["sha256"] == weight["expectedSha256"]
                and weight["bytes"] == weight["expectedBytes"], "unverified cached published weight")
        # Reference the observed file-identity proof, NOT a copied multi-GB model
        # and NOT a JSON hash mislabeled as the underlying model payload hash.
        reference = ctx.put(f"weights/{index:03}.json", weight)
        grouped.setdefault((weight["repository"], weight["revision"]), []).append(reference)
    require(grouped, "native consumed-weight inventory missing")
    return [{"repository": repository, "revision": revision, "files": references}
            for (repository, revision), references in sorted(grouped.items())]


def _identity(ctx, raw, driver, result, cpu_reference):
    backend, parent, method = raw["caseId"].split("/")
    source = driver["source"]
    p = raw["raw"]["devicePreflight"]
    require(p["workflowSource"] == source["workflowSource"] and p["inferencePin"] == source["inferencePin"],
            "raw native source binding differs from actual driver source")
    identity = {"rulebook": "sc-26584/moss-image-training-acceptance/v1",
                "backend": backend, "parent": parent, "method": method,
                "source": {"sceneWorksSourceSha": source["workflowSource"], "inferencePin": source["inferencePin"],
                           "finalMergedSource": source["finalMergedSource"]},
                "device": _device(ctx, raw, cpu_reference), "weightSources": _weight_sources(ctx, raw)}
    # The driver must retain these source observations, not ask this adapter to
    # invent a clean checkout, toolchain string, workflow id, or native version.
    for field in ("clean", "dependencyOverridesUsed"):
        if field in source:
            identity["source"][field] = source[field]
        else:
            result["missing"].append(f"identity.source.{field}")
    fields = {"workflow": ("ref", "sha", "fileSha256", "runId", "attempt", "jobId"),
              "toolchain": ("rustc", "cargo", "target", "nativeRuntime", "binarySha256")}
    for field, keys in fields.items():
        value = driver.get(field)
        if isinstance(value, dict):
            identity[field] = {key: copy.deepcopy(value[key]) for key in keys if key in value}
            result["missing"].extend(f"identity.{field}.{key}" for key in keys if key not in value)
        else:
            result["missing"].append(f"identity.{field}")
    if "binarySha256" in identity.get("toolchain", {}):
        require(identity["toolchain"]["binarySha256"] == driver["executable"]["sha256"], "toolchain/binary hash differs")
    return identity


def _tensor(value):
    require(value["content_encoding"] == ENCODING, "native canonical content encoding differs")
    shape = value["shape"]
    strides = value["strides"]
    require(shape and len(shape) == len(strides), "actual native tensor shape/strides missing")
    for n in shape:
        _integer(n, "native tensor dimension", True)
    for n in strides:
        _integer(n, "native tensor source stride")
    require(re.fullmatch(r"[0-9a-f]{64}", value["content_sha256"]) is not None, "native content hash malformed")
    return {"shape": shape, "dtype": _text(value["dtype"], "native dtype"),
            "layout": {"order": "native-source-strides", "strides": strides, "strideUnit": "elements"},
            "sha256": value["content_sha256"], "hashEncoding": value["content_encoding"]}


def _point(observation):
    require(observation["sigma"] == 0.375, "actual native evaluation sigma changed")
    loss = _finite(observation["loss"], "prepared evaluation loss")
    tensors = observation["tensors"]
    require(set(tensors) in ({"latent", "context", "noise"}, {"latent", "context", "mask", "noise"}),
            "native prepared tensor population differs")
    return {"loss": loss, "finiteObserved": math.isfinite(loss),
            "prepared": {name: _tensor(tensors[name]) if name in tensors else None
                         for name in ("latent", "context", "mask", "noise")}}


def _learning(native, native_ref, evaluation_hash, canceled):
    pair = native.get("prepared_evaluation")
    require(isinstance(pair, dict) and isinstance(pair.get("before"), dict),
            "actual optional prepared-evaluation observations absent")
    require(native["evaluation_manifest"] == evaluation_hash, "native heldout source identity differs")
    before = _point(pair["before"])
    result = {"manifestHash": evaluation_hash, "before": before}
    if canceled:
        require(pair.get("after") is None and native["evaluation_after"] is None
                and native["complete"] is False, "canceled execution claimed final learning")
        return {**result, "after": None, "afterAbsenceReason": "expected-cancellation-before-completion",
                "learningClaimed": False}
    require(isinstance(pair.get("after"), dict) and native["complete"] is True, "complete evaluation pair missing")
    after = _point(pair["after"])
    require(pair["before"]["tensors"] == pair["after"]["tensors"], "actual prepared tensor package changed within run")
    difference = before["loss"] - after["loss"]
    require(difference > 0, "actual strict heldout after < before criterion failed")
    return {**result, "after": after, "criterion": "after < before", "strictDecreaseVerified": after["loss"] < before["loss"],
            "improvement": difference, "sameActualPreparedPackageVerified": pair["before"]["tensors"] == pair["after"]["tensors"],
            "reencodingBetweenMeasurements": False, "noiseRegenerationBetweenMeasurements": False,
            "nativeReceipt": native_ref}


def _cursor(state, reference, accumulation):
    step = _integer(state["step"], "actual native cursor step")
    tokens = _integer(state["accumulated_tokens"], "actual native accumulated tokens")
    microsteps = step % _integer(accumulation, "actual gradient accumulation", True)
    require(microsteps == 0 and tokens == 0, "frozen accumulation-one checkpoint contains pending accumulation")
    return {"epoch": state["epoch"], "batchCursor": state["batch"], "sampleCursor": state["sample_cursor"],
            "partialAccumulation": {"microsteps": microsteps, "tokens": tokens, "gradientStateSha256": None},
            "stateArtifact": reference}


def _run(ctx, raw, name, source, dataset_hash, config_hash, evaluation_hash):
    run = raw["raw"]["subruns"][name]
    reference = run["nativeReceipt"]
    native = ctx.read(reference)
    progress = run["progress"]
    config = run["preparedNativeConfig"]
    canceled = name == "canceled"
    require(native["cpu_reference"] is False and native["run_scope_returned"] is True
            and progress["nativeReturned"] is True, "actual native scope did not return")
    require(run["job"]["status"] == ("canceled" if canceled else "completed"), "API/native terminal state differs")
    require(config["steps"] == 60 and config["seed"] == native["resolved_seed"] == 1789
            and config["batchSize"] == config["gradientAccumulation"] == 1
            and config["rank"] == 4 and config["alpha"] == 6.0, "actual prepared sealed controls changed")
    checkpoint = run["durableCheckpoint"]
    state = checkpoint["state"]
    state_ref = checkpoint["members"]["run.json"]
    require(ctx.read(state_ref) == state, "retained durable run state differs")
    start = native["run_start_adapter"]
    start_step = _integer(start["start_step"], "actual adapter start step")
    completed = _integer(state["step"], "actual adapter final step")
    require(completed == (20 if canceled else 60), "actual native completed horizon differs")
    require(progress["steps"] == list(range(start_step + 1, completed + 1)), "actual native update coverage differs")
    gradients = {"tensorCount": native["gradient_tensors"], "allFiniteObserved": native["gradients_finite"],
                 "nonzeroObserved": native["nonzero_gradient"], "nativeReceipt": reference}
    require(gradients["tensorCount"] > 0 and gradients["allFiniteObserved"] is True
            and gradients["nonzeroObserved"] is True, "missing finite nonzero measured gradients")
    if canceled:
        checkpoints = [p for p in native["checkpoints"] if p["state"]["step"] == 20]
        require(len(checkpoints) == 1, "actual native checkpoint20 missing/duplicated")
        delta = checkpoints[0]["factor_delta"]
        start_export, final_export = checkpoint["startExport"], checkpoint["adapter"]
    else:
        delta = native["final_parameter_delta"]
        start_export, final_export = run["startExport"], run["finalExport"]
    ctx.checked(start_export)
    ctx.checked(final_export)
    require(delta["initial_sha256"] == start_export["sha256"] == start["sha256"]
            and delta["final_sha256"] == final_export["sha256"], "actual numeric factor delta/export binding differs")
    backend, parent, method = raw["caseId"].split("/")
    identities = dict(start["export_identity"])
    require(int(identities["rank"]) == 4 and float(identities["alpha"]) == 6.0
            and identities["networkType"] == method and identities["baseModel"] == f"moss_image_{parent}",
            "native export's exact rank/alpha/parent/method differs")
    fields = {"tensorCount": "tensors", "elementCount": "elements", "changedElements": "changed_elements",
              "maxAbsDelta": "max_abs_delta", "squaredL2Delta": "squared_l2_delta"}
    factor = {key: _finite(delta[native_key], f"actual factor {key}", positive=True)
              for key, native_key in fields.items()}
    factor.update({"finiteObserved": all(math.isfinite(v) for v in factor.values()),
                   "startExport": start_export, "finalExport": final_export,
                   "layoutIdentity": delta["layout_sha256"], "alpha": config["alpha"], "rank": config["rank"],
                   "parent": parent, "method": method, "sourceSha": source["workflowSource"],
                   "weightSourceRevision": identities["base_revision"], "nativeReceipt": reference})
    result = {"jobId": run["jobId"], "nativeReceipt": reference,
              "outcome": "canceled" if canceled else "completed", "originalHorizon": config["steps"],
              "completedStep": completed, "seed": config["seed"], "datasetManifestSha256": dataset_hash,
              "configSha256": config_hash, "evaluationManifestHash": evaluation_hash,
              "learning": _learning(native, reference, evaluation_hash, canceled), "gradients": gradients,
              "factorDelta": factor, "cursor": _cursor(state, state_ref, config["gradientAccumulation"]),
              "runScopeReturned": native["run_scope_returned"]}
    frozen = native["frozen"]
    require(all(item["before"] == item["after"] for item in frozen.values()), "observed frozen native component changed")
    if canceled:
        require(progress["tokenObservedCanceled"] is True and run["job"]["result"].get("trainedArtifactExported") is not True
                and native.get("final_adapter_path") is None and native.get("final_native_path") is None,
                "original API cancellation token/final publication differs")
        result.update({"checkpointDurable": state["step"] == 20, "checkpoint": state_ref,
                       "originalCancelToken": progress["tokenId"],
                       "originalTokenObservedCanceled": progress["tokenObservedCanceled"], "publishedFinalAdapter": False})
    else:
        require(all(name in frozen for name in ("backbone", "text_encoder", "vae")), "complete native frozen observations missing")
        losses = [_finite(v, "actual native training loss") for v in progress["trainingLosses"]]
        updates = state["update"] - start["start_update"]
        require(len(losses) == updates == completed - start_step, "native optimizer/loss update coverage differs")
        result.update({"startStep": start_step, "optimizerUpdatesInThisRun": updates, "trainingLosses": losses,
                       "allLossesFiniteObserved": all(math.isfinite(x) for x in losses),
                       "frozenComponentsUnchanged": ["backbone", "text-encoder", "vae"],
                       "frozenReceipt": ctx.put(f"frozen/{name}.json", frozen)})
    return result


def _api_events(ctx, raw):
    events = [(ctx.read(ref), ref) for ref in raw["raw"]["api"]]
    directory = ctx.root / "worker-http"
    require(directory.is_dir() and not directory.is_symlink(), "actual worker HTTP progress/result observations missing")
    for file in sorted(directory.glob("*.json")):
        ref = ctx.reference(file)
        events.append((ctx.read(ref), ref))
    return events


def _event(events, predicate, resource_id):
    matches = [(event, ref) for event, ref in events if predicate(event) and 200 <= event["httpStatus"] < 300]
    require(matches, f"actual successful API event missing for {resource_id}")
    event, ref = matches[-1]
    return {"method": event["method"], "route": event["route"], "jobOrResourceId": resource_id,
            "httpStatus": event["httpStatus"], "requestArtifact": ref, "responseArtifact": ref}


def _image(ctx, raw, role, adapter_sha, strength):
    image = raw["raw"]["images"][role]
    path = ctx.checked(image["artifact"])
    with path.open("rb") as source:
        prefix = source.read(24)
    require(prefix[:8] == b"\x89PNG\r\n\x1a\n" and int.from_bytes(prefix[16:20], "big") == 256
            and int.from_bytes(prefix[20:24], "big") == 256, "actual PNG dimensions differ")
    report = ctx.read(image["executionReceipt"])
    require(report["execution_device"] == raw["executionBackend"]
            and report["samples"][0]["seed"] == image["seed"] == 1789, "actual generation backend/seed differs")
    require(image["adapterStrength"] == strength, "actual preview selection strength differs")
    return {"artifact": image["artifact"], "mimeType": "image/png", "width": image["width"], "height": image["height"],
            "prompt": image["prompt"], "seed": image["seed"], "adapterStrength": strength,
            "adapterSha256": adapter_sha, "executionReceipt": image["executionReceipt"]}


def _application(ctx, raw, role, adapter_sha):
    preload_ref = raw["raw"]["imagePreloads"][role]
    preload = ctx.read(preload_ref)
    require(preload["error"] is None, "actual selected adapter generation pre-load refused")
    actual = preload["actualSelectedHeaderDemand"]
    require(actual["requiredBytes"] <= preload["resolvedCapBytes"]
            == raw["raw"]["devicePreflight"]["resolvedCapBytes"], "actual generation exceeded frozen cap")
    adapters = [value for value in actual["adapters"] if value["sha256"] == adapter_sha and value["scale"] == 1.0]
    require(len(adapters) == 1, "exact trained custom adapter not present in actual selected preload")
    report = ctx.read(raw["raw"]["images"][role]["executionReceipt"])
    basename = _basename(adapters[0]["path"])
    applied = [value for value in report["adapter_reports"] if _basename(value["adapter_path"]) == basename]
    require(len(applied) == 1 and applied[0]["applied"] > 0 and not applied[0]["skipped"],
            "actual CUSTOM adapter installation missing; a DMD-only report does not count")
    _, parent, method = raw["caseId"].split("/")
    return {"adapterSha256": adapter_sha, "parent": parent, "method": method,
            "selectedStrength": adapters[0]["scale"], "executedStrength": adapters[0]["scale"],
            "appliedCount": applied[0]["applied"], "skippedCount": len(applied[0]["skipped"]),
            "receipt": raw["raw"]["images"][role]["executionReceipt"]}, adapters[0]


def _memory_refusal(ctx, raw):
    low = raw["raw"]["subruns"]["low-cap"]
    diagnostic = ctx.read(low["diagnostic"])
    actual = diagnostic["nativeMetadata"]
    refusal = _text(actual["refusal"], "actual native pre-load refusal")
    # Exact native Display contract shared by MLX/Candle; no guessed resolved
    # budget, requested-cap substitution, parser tolerance or caller override.
    match = re.search(r"structural working-set bound (\d+) bytes exceeds\s+the (\d+)-byte device/trainer budget", refusal)
    require(match is not None, "native low-cap refusal does not expose its actual required/resolved budget")
    demand, resolved = map(int, match.groups())
    requested = low["preparedNativeConfig"]["memoryLimitBytes"]
    require(0 < resolved <= requested < demand and low["publishedOutputInventory"] == []
            and actual["weightsLoaded"] is False and not diagnostic["nativeReceipts"],
            "actual low-cap/native no-load/no-publication proof missing")
    return {"requestedControlCap": {"kind": "bytes", "value": requested}, "resolvedControlCapBytes": resolved,
            "projectedHeaderDemandBytes": demand,
            "headerDemandReceipt": ctx.reference(ctx.path("devicepreflight.json")),
            "refusedBeforeWeightsLoad": actual["weightsLoaded"] is False,
            "actionableRefusal": refusal, "publishedOutput": bool(low["publishedOutputInventory"]),
            "publishedOutputInventory": low["publishedOutputInventory"], "campaignCeilingRaised": False,
            "receipt": low["diagnostic"]}


def _backend_reentry(ctx, raw):
    backend, _, _ = raw["caseId"].split("/")
    anchor = raw
    anchor_ctx = ctx
    if raw["caseId"] != f"{backend}/base/lora":
        # Fixed campaign layout, not an arbitrary external path or carried proof.
        anchor_ctx = _Context(ctx.root.parent.parent / "base" / "lora")
        anchor = _read(anchor_ctx.path("evidence.json"))
    require(anchor["caseId"] == f"{backend}/base/lora" and anchor["manifestSha256"] == raw["manifestSha256"]
            and anchor["raw"]["devicePreflight"]["inferencePin"] == raw["raw"]["devicePreflight"]["inferencePin"]
            and anchor["raw"]["devicePreflight"]["workflowSource"] == raw["raw"]["devicePreflight"]["workflowSource"]
            and anchor["raw"]["datasetManifest"]["sha256"] == raw["raw"]["datasetManifest"]["sha256"]
            and anchor["evaluationIdentity"]["manifestHash"] == raw["evaluationIdentity"]["manifestHash"],
            "backend reentry witness belongs to a different source/campaign")
    witness = anchor_ctx.read(anchor["raw"]["backendReentryWitness"])
    old, fresh = witness["original"], witness["fresh"]
    fresh_receipt = anchor_ctx.read(anchor["raw"]["subruns"]["fresh-reentry-new-run"]["nativeReceipt"])
    require(old["tokenObservedCanceled"] is True and fresh["tokenObservedCanceled"] is False
            and old["tokenId"] != fresh["tokenId"] and fresh["steps"] == list(range(1, 61))
            and fresh_receipt["complete"] is True and fresh_receipt["run_start_adapter"]["start_step"] == 0,
            "fresh reentry did not causally preserve the canceled original and independent fresh token")
    reference = ctx.put("recovery/backend-reentry.json", witness)
    return {"scope": "one-per-backend", "anchorCaseId": anchor["caseId"], "witness": reference,
            "originalCancelToken": old["tokenId"], "freshRunCancelToken": fresh["tokenId"],
            "distinctFreshTokenObserved": old["tokenId"] != fresh["tokenId"],
            "freshTokenUncanceled": fresh["tokenObservedCanceled"] is False, "freshRunCompletedStep": fresh["steps"][-1]}


def _recovery(ctx, raw, runs, dataset_hash, config_hash):
    local = raw["raw"]["recovery"]
    states = ctx.read(local["finalStates"])
    require(states["reference"] == states["resumed"], "actual complete native reference/resumed state differs")
    inventory = ctx.read(local["bundleInventory"])
    require(inventory["reference"] and inventory["reference"] == inventory["resumed"], "actual optimizer/tensor bundle bytes differ")
    require(runs["uninterrupted"]["factorDelta"]["finalExport"]["sha256"] == runs["resumed"]["factorDelta"]["finalExport"]["sha256"],
            "actual reference/resumed final exports differ")
    normalized = {}
    for name, label in (("reference", "uninterrupted"), ("resumed", "resumed")):
        state = states[name]
        optimizer = ctx.copy(f"recovery/{name}-optimizer.safetensors",
                             raw["raw"]["subruns"][label]["durableCheckpoint"]["members"]["optimizer.safetensors"])
        normalized[name] = {
            "optimizer": {"path": Path(optimizer["path"]).name, "sha256": optimizer["sha256"]},
            # The native stream is stateless/step-addressed. These are exactly
            # its persisted replay inputs, not fabricated random-generator bytes.
            "scheduler": {"effectiveConfig": state["effective_identity"]["effective_config"],
                          "step": state["step"], "update": state["update"]},
            "rng": {key: state[key] for key in ("stream_version", "resolved_seed", "step", "plan_identity")},
            "cursor": {key: state[key] for key in ("epoch", "batch", "sample_cursor")},
            "partialAccumulation": {"accumulated_tokens": state["accumulated_tokens"]},
            "nativeRunState": state,
        }
    pair = ctx.put("recovery/complete-replay-state.json", normalized)
    old = raw["raw"]["subruns"]["canceled"]["progress"]
    new = raw["raw"]["subruns"]["resumed"]["progress"]
    stall = old["checkpointStall"]
    require(old["tokenObservedCanceled"] is True and new["tokenObservedCanceled"] is False
            and old["tokenId"] != new["tokenId"] and stall["state"] == "released-by-api-cancellation"
            and stall["timedOut"] is False and stall["durableCheckpointVerified"] is True,
            "actual bounded API cancellation/resume proof missing")
    stall_ref = ctx.put("recovery/checkpoint-stall.json", old)
    native_stall = {key: stall[key] for key in ("scope", "referenceStallEnabled", "checkpointStep", "waitedForTokenId",
                   "timeoutSeconds", "state", "timedOut", "tokenObservedCanceled", "updatesSkippedOrReplayed")}
    native_stall.update({"enteredAt": _at_ms(stall["enteredUnixMs"]), "exitedAt": _at_ms(stall["exitedUnixMs"]), "receipt": stall_ref})
    return {"originalHorizon": runs["resumed"]["originalHorizon"], "seed": runs["resumed"]["seed"],
            "datasetManifestSha256": dataset_hash, "configSha256": config_hash, "checkpointStep": runs["canceled"]["completedStep"],
            "resumedThroughApi": new["tokenObservedCanceled"] is False and runs["resumed"]["startStep"] == 20,
            "referenceAndResumedExportsCompared": True, "exportedStateEqual": states["reference"] == states["resumed"],
            "resultsEqual": states["reference"]["final_loss"] == states["resumed"]["final_loss"], "assertionsWeakened": False,
            "cursorBeforeResume": runs["canceled"]["cursor"], "cursorAfterResume": runs["resumed"]["cursor"],
            "optimizerSchedulerRngState": pair, "originalCancelToken": old["tokenId"], "resumedCancelToken": new["tokenId"],
            "distinctResumedTokenObserved": old["tokenId"] != new["tokenId"], "originalTokenRemainsCanceled": old["tokenObservedCanceled"],
            "checkpointStall": native_stall, "backendReentryWitness": _backend_reentry(ctx, raw)}


def normalize_capture(raw, caseRoot, manifest, driverReceipt, caseEntry):
    """Return local normal-form observations; external/global proofs stay missing."""
    ctx = _Context(caseRoot)
    result = _base(raw, caseEntry, "capture")
    if raw.get("failureReason"):
        result["acceptanceDisposition"] = "failed"
        result["missing"].append("native-local-observations")
        return result
    source = driverReceipt["source"]
    original = raw["raw"]["subruns"]["uninterrupted"]
    original_native = ctx.read(original["nativeReceipt"])
    result["identity"] = _identity(ctx, raw, driverReceipt, result, original_native["cpu_reference"])
    dataset_ref = raw["raw"]["datasetManifest"]
    dataset = ctx.read(dataset_ref)
    uploads = raw["raw"]["datasetUploads"]
    require(dataset["manifestSha256"] == SEAL and dataset["imageCount"] == len(uploads) == 3,
            "actual uploaded sealed dataset population differs")
    wanted = {item["image"]: item["caption"] for item in manifest["dataset"]["items"]}
    require({item["image"]: item["caption"] for item in uploads} == wanted, "actual uploaded captions differ from campaign")
    for item in uploads:
        event = ctx.read(item["uploadReceipt"])
        image = event["request"]["sourceImage"]
        caption = event["request"]["sourceCaption"]
        ctx.checked(image)
        caption_path = ctx.checked(caption)
        require(event["httpStatus"] == 201 and event["response"]["id"] == item["uploadId"]
                and event["response"]["file"]["contentHash"] == image["sha256"] == item["imageSha256"]
                and caption_path.read_bytes() == item["caption"].encode("utf-8"), "actual upload/source bytes or response identity differs")
    wanted_config = manifest["training"]["config"]
    prepared = original["preparedNativeConfig"]
    require(all(prepared[key] == wanted_config[key] for key in ("steps", "seed", "rank", "alpha", "batchSize", "gradientAccumulation"))
            and wanted_config["learningRate"] == original["job"]["payload"]["plan"]["config"]["learningRate"],
            "actual native/API controls differ from sealed campaign")
    config_ref = ctx.put("effective-request-config.json", original["job"]["payload"]["plan"]["config"])
    dataset_hash, config_hash = dataset_ref["sha256"], config_ref["sha256"]
    result["datasetIdentity"] = {"id": dataset["datasetId"], "manifestSha256": dataset_hash, "configSha256": config_hash,
                                 "artifact": dataset_ref, "sealedBeforeWeightsLoad": raw["raw"]["devicePreflight"]["weightsLoaded"] is False,
                                 "uploadedCount": len(uploads), "uploadedItems": [
                                     {key: item[key] for key in ("image", "imageSha256", "caption", "uploadId", "uploadReceipt")}
                                     for item in uploads]}
    evaluation = copy.deepcopy(raw["evaluationIdentity"])
    # This function proves one case's subruns. Only the finalizer can compare
    # both backend artifacts and authenticate the all-eight shared-source claim.
    evaluation["sharedAcrossAllCellsAndSubruns"] = False
    result["unproven"].append("evaluationIdentity.sharedAcrossAllCellsAndSubruns")
    result["evaluationIdentity"] = evaluation
    runs = {name: _run(ctx, raw, name, source, dataset_hash, config_hash, evaluation["manifestHash"])
            for name in ("uninterrupted", "canceled", "resumed")}
    result["subruns"] = runs
    result["recovery"] = _recovery(ctx, raw, runs, dataset_hash, config_hash)
    result["memoryRefusal"] = _memory_refusal(ctx, raw)
    final_sha = runs["resumed"]["factorDelta"]["finalExport"]["sha256"]
    start_sha = runs["uninterrupted"]["factorDelta"]["startExport"]["sha256"]
    library = raw["raw"]["trainingLibraryRegistration"]
    require(library["sha256"] == final_sha, "actual library selection is not the final adapter")
    application, selected = _application(ctx, raw, "after-trained", final_sha)
    images = raw["raw"]["images"]
    require(images["no-adapter-baseline"]["pixelSha256"] == images["disable-control"]["pixelSha256"]
            == images["post-unload-replay"]["pixelSha256"], "actual disable/unload baseline pixels differ")
    saved_ref = ctx.put("save-reload.json", {"library": library, "actualSelectedAdapter": selected,
                                            "nativeReturned": original_native["run_scope_returned"]})
    result["saveReloadGenerate"] = {
        "adapterPath": library["outputPath"], "adapterSha256": final_sha, "libraryId": library["entry"]["id"],
        "resolvedPath": selected["path"], "resolvedSha256": selected["sha256"],
        "finalAdapterSelected": selected["sha256"] == final_sha,
        "trainerDroppedBeforeReload": original_native["run_scope_returned"], "applicationReport": application,
        "before": _image(ctx, raw, "before-untrained", start_sha, 1.0),
        "after": _image(ctx, raw, "after-trained", final_sha, 1.0),
        "disableControl": _image(ctx, raw, "disable-control", final_sha, 0.0),
        "noAdapterBaseline": images["no-adapter-baseline"]["artifact"],
        "disableMatchesNoAdapterBaseline": images["no-adapter-baseline"]["pixelSha256"] == images["disable-control"]["pixelSha256"],
        "receipt": saved_ref,
    }
    events = _api_events(ctx, raw)
    upload = uploads[0]
    upload_event = ctx.read(upload["uploadReceipt"])
    reference_id = original["jobId"]
    image_job = ctx.read(images["after-trained"]["jobReceipt"])
    result["productPath"] = {
        "datasetUpload": {"method": upload_event["method"], "route": upload_event["route"],
                          "jobOrResourceId": upload["uploadId"], "httpStatus": upload_event["httpStatus"],
                          "requestArtifact": upload["uploadReceipt"], "responseArtifact": upload["uploadReceipt"]},
        "submit": _event(events, lambda e: e["method"] == "POST" and e["route"].endswith("/training/jobs")
                         and e["response"].get("id") == reference_id, reference_id),
        "claim": _event(events, lambda e: e["route"] == "/api/v1/jobs/claim"
                        and e["response"].get("job", {}).get("id") == reference_id, reference_id),
        "progress": _event(events, lambda e: e["method"] == "POST" and e["route"] == f"/api/v1/jobs/{reference_id}/progress"
                           and e["request"].get("stage") == "training", reference_id),
        "result": _event(events, lambda e: e["method"] == "GET" and e["route"] == f"/api/v1/jobs/{reference_id}"
                         and e["response"].get("status") == "completed", reference_id),
        "libraryRegistration": _event(events, lambda e: e["method"] == "GET" and e["route"].startswith("/api/v1/loras?")
                                      and isinstance(e["response"], list)
                                      and any(v.get("id") == library["entry"]["id"] for v in e["response"]), library["entry"]["id"]),
        "imageJobExecution": _event(events, lambda e: e["method"] == "GET" and e["route"] == f"/api/v1/jobs/{image_job['id']}"
                                    and e["response"].get("status") == "completed", image_job["id"]),
    }
    ordinary_ref = ctx.put("ordinary-regression.json", {"before": images["no-adapter-baseline"], "after": images["post-unload-replay"]})
    regression = {"ordinary": {"selectedMode": "off", "executedMode": ctx.read(images["post-unload-replay"]["executionReceipt"])["resolved"]["acceleration"],
                               "beforeNoAdapter": images["no-adapter-baseline"]["artifact"], "afterUnload": images["post-unload-replay"]["artifact"],
                               "sameOutputVerified": images["no-adapter-baseline"]["pixelSha256"] == images["post-unload-replay"]["pixelSha256"],
                               "receipt": ordinary_ref}, "dmdExecutions": [], "baseDmdRefusals": []}
    parent = raw["caseId"].split("/")[1]
    for mode in ("unmerged_bf16", "full_fp32"):
        role = f"dmd-{mode}"
        if parent == "base":
            refusal = images[role]
            require(refusal["httpStatus"] == 400 and "sft" in json.dumps(refusal["refusal"]).lower(), "Base+DMD parent refusal missing")
            regression["baseDmdRefusals"].append({"selectedMode": mode, "executedMode": None,
                "httpStatus": refusal["httpStatus"], "refusal": json.dumps(refusal["refusal"], sort_keys=True),
                "producedImage": False, "receipt": ctx.put(f"refusal-{mode}.json", refusal)})
        else:
            report = ctx.read(images[role]["executionReceipt"])
            require(report["resolved"]["acceleration"] == mode, "selected/actual DMD mode differs")
            roles = {a["role"] for a in report["assets"]}
            retained = {"dmd_time", "dmd_dataspace"} <= roles and (mode != "unmerged_bf16" or "dmd_lora" in roles)
            require(retained, "actual DMD accelerator components were dropped")
            custom, _ = _application(ctx, raw, role, final_sha)
            selected_sources = {(a["repository"], a["revision"]) for a in report["assets"]}
            regression["dmdExecutions"].append({"selectedMode": mode, "executedMode": report["resolved"]["acceleration"],
                "realWeightSources": [v for v in result["identity"]["weightSources"] if (v["repository"], v["revision"]) in selected_sources],
                "customAdapterRetained": custom["appliedCount"] > 0, "acceleratorRetained": retained,
                "generation": _image(ctx, raw, role, final_sha, 1.0), "applicationReport": custom, "receipt": images[role]["executionReceipt"]})
    result["regression"] = regression
    result["inspections"] = []
    return result


def normalize_cross(raw, caseRoot, manifest, driverReceipt, caseEntry):
    """Return a local cross-load piece; controller owns resourceGuard/release."""
    ctx = _Context(caseRoot)
    _base(raw, caseEntry, "cross-load")
    if raw.get("failureReason"):
        return {"crossBackendLoad": None, "device": None, "missing": ["crossBackendLoad", "resourceGuard"],
                "unproven": [], "failureReason": raw["failureReason"]}
    local = raw["raw"]["crossBackendLoad"]
    transfer = local["transferredAdapter"]
    ctx.checked(transfer)
    require(transfer["sha256"] == local["receivedSha256"], "received adapter bytes differ from original transfer")
    require(local["sourceCaseId"] == raw["caseId"] and local["sourceBackend"] != local["importedBackend"]
            == raw["executionBackend"], "cross execution/source backend identity differs")
    application, _ = _application(ctx, raw, "cross-backend-load", transfer["sha256"])
    generation = _image(ctx, raw, "cross-backend-load", transfer["sha256"], 1.0)
    start = datetime.fromisoformat(local["jobStartedAt"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(local["completedAt"].replace("Z", "+00:00"))
    elapsed = (end - start).total_seconds()
    require(start.tzinfo is not None and end.tzinfo is not None and 0 < elapsed <= 3600,
            "actual cross-load job clock outside its own frozen interval")
    result = {"sourceCaseId": local["sourceCaseId"], "phase": "cross-load",
              "phaseTiming": {"jobStartedAt": local["jobStartedAt"], "completedAt": local["completedAt"],
                              "wallSeconds": elapsed, "wallLimitSeconds": local["wallLimitSeconds"],
                              "clockSource": "cross-load-job-start"},
              "sourceBackend": local["sourceBackend"], "importedBackend": local["importedBackend"],
              "transferredAdapter": transfer, "receivedSha256": local["receivedSha256"],
              "parent": local["parent"], "method": local["method"], "outcome": "generated",
              "generation": generation, "applicationReport": application, "unsupportedFormatRefusal": None,
              "receipt": ctx.put("cross-local.json", local)}
    return {"crossBackendLoad": result, "device": _device(ctx, raw, False),
            "missing": ["resourceGuard", "cleanup", "inspections"], "unproven": [], "failureReason": None}
