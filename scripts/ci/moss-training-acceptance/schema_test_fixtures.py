"""Construct JSON Schema test data only. NEVER campaign evidence or model execution.

All identities, artifacts, tensors, measurements and observations below are
synthetic placeholders. No artifact files, native processes, weights or devices
are opened. The root discriminator prohibits campaign-evidence classification.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA = "0" * 64
REV = "0" * 40
DATASET = "3" * 64
CONFIG = "4" * 64
EVALUATION = "5" * 64
CAPTION = "sksmoss red circle centered on a pale background"


def artifact(name, digest=SHA):
    return {"path": "SCHEMA_TEST_ONLY/" + name, "sha256": digest}


def tensors(package):
    descriptor = {
        "shape": [1, 1], "dtype": "f32",
        "layout": {"order": "row-major", "strides": [1, 1], "strideUnit": "elements"},
        "sha256": "7" * 64,
        "hashEncoding": "logical-row-major-f32-le-v1",
    }
    return {
        "latent": copy.deepcopy(descriptor), "context": copy.deepcopy(descriptor),
        "mask": None, "noise": copy.deepcopy(descriptor),
    }


def evaluation_point(package, loss):
    return {"loss": loss, "finiteObserved": True, "prepared": tensors(package)}


def learning(package):
    return {
        "manifestHash": EVALUATION,
        "before": evaluation_point(package, 0.75),
        "after": evaluation_point(package, 0.50),
        "criterion": "after < before", "strictDecreaseVerified": True,
        "improvement": 0.25, "sameActualPreparedPackageVerified": True,
        "reencodingBetweenMeasurements": False,
        "noiseRegenerationBetweenMeasurements": False,
        "nativeReceipt": artifact(package + "-native.json"),
    }


def gradients(name):
    return {
        "tensorCount": 1, "allFiniteObserved": True, "nonzeroObserved": True,
        "nativeReceipt": artifact(name + "-native.json"),
    }


def delta(name):
    return {
        "tensorCount": 1, "elementCount": 1, "changedElements": 1,
        "maxAbsDelta": 0.125, "squaredL2Delta": 0.015625,
        "finiteObserved": True,
        "startExport": artifact(name + "-start.safetensors", "1" * 64),
        "finalExport": artifact(name + "-end.safetensors", "2" * 64),
        "layoutIdentity": "8" * 64, "alpha": 6.0, "rank": 4,
        "parent": "base", "method": "lora", "sourceSha": REV,
        "weightSourceRevision": REV,
        "nativeReceipt": artifact(name + "-native.json"),
    }


def cursor(name, samples):
    return {
        "epoch": 0, "batchCursor": samples, "sampleCursor": samples,
        "partialAccumulation": {"microsteps": 0, "tokens": 0, "gradientStateSha256": None},
        "stateArtifact": artifact(name + "-state.json"),
    }


def complete(name, start, updates):
    return {
        "jobId": "SCHEMA_TEST_ONLY_" + name,
        "nativeReceipt": artifact(name + "-native.json"),
        "outcome": "completed", "originalHorizon": 60, "startStep": start,
        "completedStep": 60, "optimizerUpdatesInThisRun": updates, "seed": 1789,
        "datasetManifestSha256": DATASET, "configSha256": CONFIG,
        "evaluationManifestHash": EVALUATION,
        "trainingLosses": [0.75, 0.50], "allLossesFiniteObserved": True,
        "gradients": gradients(name), "learning": learning(name),
        "factorDelta": delta(name), "cursor": cursor(name, 60),
        "frozenComponentsUnchanged": ["backbone", "text-encoder", "vae"],
        "frozenReceipt": artifact(name + "-frozen.json"), "runScopeReturned": True,
    }


def application():
    return {
        "adapterSha256": "2" * 64, "parent": "base", "method": "lora",
        "selectedStrength": 1.0, "executedStrength": 1.0,
        "appliedCount": 1, "skippedCount": 0,
        "receipt": artifact("application.json"),
    }


def png(name, strength, digest):
    return {
        "artifact": artifact(name + ".png", digest), "mimeType": "image/png",
        "width": 256, "height": 256, "prompt": CAPTION, "seed": 1789,
        "adapterStrength": strength,
        "adapterSha256": ("1" if name == "before" else "2") * 64,
        "executionReceipt": artifact(name + "-execution.json"),
    }


def api_event(name, route):
    return {
        "method": "POST", "route": route,
        "jobOrResourceId": "SCHEMA_TEST_ONLY_" + name, "httpStatus": 200,
        "requestArtifact": artifact(name + "-request.json"),
        "responseArtifact": artifact(name + "-response.json"),
    }


before = png("before", 1.0, "b" * 64)
after = png("after", 1.0, "a" * 64)
disable = png("disable", 0.0, "c" * 64)
record = {
    "schemaVersion": 1,
    "recordKind": "schema-test-fixture",
    "fixtureLabel": "SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE",
    "isCampaignEvidence": False,
    "acceptanceDisposition": "not-campaign-evidence",
    "missing": [], "unproven": [], "failureReason": None,
    "manifestSha256": "cad2d90a22ccda3f69f74103386efa49e9d2c89d76e65d8b92195e201a4c8296",
    "caseId": "mlx/base/lora",
    "identity": {
        "rulebook": "sc-26584/moss-image-training-acceptance/v1",
        "backend": "mlx", "parent": "base", "method": "lora",
        "source": {
            "sceneWorksSourceSha": REV, "inferencePin": REV, "clean": True,
            "finalMergedSource": True, "dependencyOverridesUsed": False,
        },
        "workflow": {
            "ref": "SCHEMA_TEST_ONLY_NOT_A_WORKFLOW_REF", "sha": REV,
            "fileSha256": SHA, "runId": "SCHEMA_TEST_ONLY_NOT_A_RUN",
            "attempt": 1, "jobId": "SCHEMA_TEST_ONLY_NOT_A_JOB",
        },
        "toolchain": {
            "rustc": "SCHEMA_TEST_ONLY_NO_RUST_EXECUTED",
            "cargo": "SCHEMA_TEST_ONLY_NO_CARGO_EXECUTED",
            "target": "SCHEMA_TEST_ONLY_NOT_A_TARGET",
            "nativeRuntime": "SCHEMA_TEST_ONLY_NO_NATIVE_RUNTIME",
            "binarySha256": SHA,
        },
        "device": {
            "class": "apple-silicon-unified",
            "claimedIdentity": "SCHEMA_TEST_ONLY_NOT_A_DEVICE",
            "observedIdentity": "SCHEMA_TEST_ONLY_NOT_A_DEVICE",
            "totalCapacityBytes": 1000, "gpuMemoryLimitFraction": 0.60,
            "capacityBasis": "actual-detected-total-unified-memory",
            "resolvedCeilingBytes": 600, "resolvedBeforeWeightsLoad": True,
            "projectedDemandBytes": 500,
            "headerDemandReceipt": artifact("synthetic-header.json"),
            "cpuReference": False,
        },
        "weightSources": [
            {"repository": "SCHEMA_TEST_ONLY/NO_BASE_WEIGHTS", "revision": REV, "files": [artifact("not-a-weight-file")]},
            {"repository": "SCHEMA_TEST_ONLY/NO_SHARED_WEIGHTS", "revision": REV, "files": [artifact("not-a-shared-weight-file")]},
        ],
    },
    "datasetIdentity": {
        "id": "synthetic-moss-v1", "manifestSha256": DATASET,
        "configSha256": CONFIG, "artifact": artifact("not-a-dataset.json"),
        "sealedBeforeWeightsLoad": True,
        "uploadedCount": 3,
        "uploadedItems": [
            {"image": "shape-0.png", "imageSha256": "6" * 64, "caption": CAPTION, "uploadId": "SCHEMA_TEST_ONLY_dataset_upload-0", "uploadReceipt": artifact("upload-0.json")},
            {"image": "shape-1.png", "imageSha256": "6" * 64, "caption": "sksmoss blue square centered on a pale background", "uploadId": "SCHEMA_TEST_ONLY_dataset_upload-1", "uploadReceipt": artifact("upload-1.json")},
            {"image": "shape-2.png", "imageSha256": "6" * 64, "caption": "sksmoss green triangle centered on a pale background", "uploadId": "SCHEMA_TEST_ONLY_dataset_upload-2", "uploadReceipt": artifact("upload-2.json")},
        ],
    },
    "evaluationIdentity": {
        "manifestHash": EVALUATION, "computedAtRunTimeBeforeWeightsLoad": True,
        "imageSha256": "6" * 64, "caption": CAPTION, "maskSha256": None,
        "width": 256, "height": 256, "noiseSeedHex": "0x48454c444f555401",
        "sigma": 0.375, "sharedAcrossAllCellsAndSubruns": True,
        "crossBackendTensorValuesMayDiffer": True,
    },
    "subruns": {
        "uninterrupted": complete("reference", 0, 60),
        "canceled": {
            "jobId": "SCHEMA_TEST_ONLY_canceled", "nativeReceipt": artifact("cancel-native.json"),
            "outcome": "canceled", "originalHorizon": 60, "completedStep": 20,
            "seed": 1789, "datasetManifestSha256": DATASET, "configSha256": CONFIG,
            "evaluationManifestHash": EVALUATION,
            "learning": {
                "manifestHash": EVALUATION, "before": evaluation_point("canceled", 0.75),
                "after": None, "afterAbsenceReason": "expected-cancellation-before-completion",
                "learningClaimed": False,
            },
            "gradients": gradients("canceled"), "factorDelta": delta("checkpoint-20"),
            "checkpointDurable": True, "checkpoint": artifact("step-20-complete.json"),
            "cursor": cursor("checkpoint-20", 20),
            "originalCancelToken": "SCHEMA_TEST_ONLY_original-token",
            "originalTokenObservedCanceled": True,
            "publishedFinalAdapter": False, "runScopeReturned": True,
        },
        "resumed": complete("resumed", 20, 40),
    },
    "saveReloadGenerate": {
        "adapterPath": "SCHEMA_TEST_ONLY/adapter.safetensors", "adapterSha256": "2" * 64,
        "libraryId": "SCHEMA_TEST_ONLY_library-entry", "resolvedPath": "SCHEMA_TEST_ONLY/adapter.safetensors",
        "resolvedSha256": "2" * 64, "finalAdapterSelected": True,
        "trainerDroppedBeforeReload": True, "applicationReport": application(),
        "before": before, "after": after, "disableControl": disable,
        "noAdapterBaseline": artifact("baseline.png", "c" * 64),
        "disableMatchesNoAdapterBaseline": True, "receipt": artifact("reload.json"),
    },
    "crossBackendLoad": {
        "sourceCaseId": "mlx/base/lora", "phase": "cross-load",
        "phaseTiming": {"jobStartedAt": "2000-01-02T00:00:00Z", "completedAt": "2000-01-02T00:00:02Z", "wallSeconds": 2, "wallLimitSeconds": 3600, "clockSource": "cross-load-job-start"},
        "resourceGuard": {
            "device": {"class": "nvidia-cuda", "claimedIdentity": "SCHEMA_TEST_ONLY_NOT_A_CUDA_DEVICE", "observedIdentity": "SCHEMA_TEST_ONLY_NOT_A_CUDA_DEVICE", "totalCapacityBytes": 1000, "gpuMemoryLimitFraction": 0.85, "capacityBasis": "actual-claimed-cuda-context-total-vram", "resolvedCeilingBytes": 850, "resolvedBeforeWeightsLoad": True, "projectedDemandBytes": 500, "headerDemandReceipt": artifact("cross-header.json"), "cpuReference": False},
            "ownedProcesses": [{"pid": 1, "birthIdentity": "SCHEMA_TEST_ONLY_NOT_A_CROSS_PROCESS", "executable": "SCHEMA_TEST_ONLY_NOT_AN_EXECUTABLE", "role": "api-worker", "releaseObserved": True}],
            "aggregateRssSamples": [{"at": "2000-01-02T00:00:00Z", "bytes": 200, "ownedProcessCount": 1}, {"at": "2000-01-02T00:00:01Z", "bytes": 250, "ownedProcessCount": 1}],
            "positiveInitialMeasurement": True, "telemetryComplete": True, "missingMeasurementCount": 0,
            "allChildrenReleased": True, "releaseReceipt": artifact("cross-release.json")
        },
        "sourceBackend": "mlx", "importedBackend": "cuda",
        "transferredAdapter": artifact("transferred.safetensors", "2" * 64),
        "receivedSha256": "2" * 64, "parent": "base", "method": "lora",
        "outcome": "generated", "generation": after, "applicationReport": application(),
        "unsupportedFormatRefusal": None, "receipt": artifact("cross-load.json"),
    },
    "recovery": {
        "originalHorizon": 60, "seed": 1789, "datasetManifestSha256": DATASET,
        "configSha256": CONFIG, "checkpointStep": 20, "resumedThroughApi": True,
        "referenceAndResumedExportsCompared": True, "exportedStateEqual": True,
        "resultsEqual": True, "assertionsWeakened": False,
        "cursorBeforeResume": cursor("before-resume", 20),
        "cursorAfterResume": cursor("after-resume", 60),
        "optimizerSchedulerRngState": artifact("recovery-state.json"),
        "originalCancelToken": "SCHEMA_TEST_ONLY_original-token",
        "resumedCancelToken": "SCHEMA_TEST_ONLY_resumed-token",
        "distinctResumedTokenObserved": True, "originalTokenRemainsCanceled": True,
        "checkpointStall": {"scope": "test-support-only", "referenceStallEnabled": False, "checkpointStep": 20, "enteredAt": "2000-01-01T00:00:00Z", "exitedAt": "2000-01-01T00:00:01Z", "waitedForTokenId": "SCHEMA_TEST_ONLY_original-token", "timeoutSeconds": 3600, "state": "released-by-api-cancellation", "timedOut": False, "tokenObservedCanceled": True, "updatesSkippedOrReplayed": 0, "receipt": artifact("stall.json")},
        "backendReentryWitness": {
            "scope": "one-per-backend", "anchorCaseId": "mlx/base/lora",
            "witness": artifact("mlx-reentry.json"),
            "originalCancelToken": "SCHEMA_TEST_ONLY_reentry-canceled-token",
            "freshRunCancelToken": "SCHEMA_TEST_ONLY_fresh-token",
            "distinctFreshTokenObserved": True, "freshTokenUncanceled": True,
            "freshRunCompletedStep": 60,
        },
    },
    "memoryRefusal": {
        "requestedControlCap": {"kind": "bytes", "value": 1},
        "resolvedControlCapBytes": 1, "projectedHeaderDemandBytes": 500,
        "headerDemandReceipt": artifact("low-cap-header.json"),
        "refusedBeforeWeightsLoad": True,
        "actionableRefusal": "SCHEMA TEST ONLY: illustrative refusal, not a native result",
        "publishedOutput": False, "publishedOutputInventory": [],
        "campaignCeilingRaised": False, "receipt": artifact("refusal.json"),
    },
    "cleanup": {
        "ownedProcesses": [{"pid": 1, "birthIdentity": "SCHEMA_TEST_ONLY_NOT_A_PROCESS", "executable": "SCHEMA_TEST_ONLY_NOT_AN_EXECUTABLE", "role": "api-worker", "releaseObserved": True}],
        "allChildrenReleased": True, "releaseReceipt": artifact("release.json"),
        "aggregateRssSamples": [
            {"at": "2000-01-01T00:00:00Z", "bytes": 200, "ownedProcessCount": 1},
            {"at": "2000-01-01T00:00:01Z", "bytes": 250, "ownedProcessCount": 1},
        ],
        "positiveInitialMeasurement": True, "telemetryComplete": True,
        "missingMeasurementCount": 0, "resolvedCeilingBytes": 600,
        "caseStartedAt": "2000-01-01T00:00:00Z", "caseEndedAt": "2000-01-01T00:00:02Z",
        "caseWallSeconds": 2, "caseDeadlineSeconds": 3600,
        "buildPreparationWallSeconds": 0, "buildPreparationDeadlineSeconds": 7200,
    },
    "productPath": {
        "datasetUpload": api_event("upload", "/api/v1/projects/SCHEMA_TEST_ONLY/training/uploads"),
        "submit": api_event("submit", "/api/v1/projects/SCHEMA_TEST_ONLY/training/jobs"),
        "claim": api_event("claim", "/api/v1/jobs/claim"),
        "progress": api_event("progress", "/api/v1/jobs/SCHEMA_TEST_ONLY/progress"),
        "result": api_event("result", "/api/v1/jobs/SCHEMA_TEST_ONLY"),
        "libraryRegistration": api_event("library", "/api/v1/loras"),
        "imageJobExecution": api_event("image", "/api/v1/image/jobs"),
    },
    "regression": {
        "ordinary": {
            "selectedMode": "off", "executedMode": "off",
            "beforeNoAdapter": artifact("baseline.png", "c" * 64),
            "afterUnload": artifact("after-unload.png", "c" * 64),
            "sameOutputVerified": True, "receipt": artifact("ordinary.json"),
        },
        "dmdExecutions": [],
        "baseDmdRefusals": [
            {"selectedMode": mode, "executedMode": None, "httpStatus": 400,
             "refusal": "SCHEMA TEST ONLY: illustrative Base+DMD refusal",
             "producedImage": False, "receipt": artifact("refuse-" + mode + ".json")}
            for mode in ["unmerged_bf16", "full_fp32"]
        ],
    },
    "inspections": [
        {"reviewer": "SCHEMA_TEST_ONLY_NOT_AN_INSPECTION", "reviewedArtifact": item["artifact"],
         "findings": ["SCHEMA TEST ONLY: no image exists or was inspected"],
         "completedAt": "2000-01-01T00:00:01Z", "withinCaseDeadline": True}
        for item in [before, after, disable]
    ],
}

if __name__ == "__main__":
    valid_path = ROOT / "valid.schema-test-fixture.json"
    missing_path = ROOT / "missing-learning.schema-test-fixture.json"
    valid_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    missing = copy.deepcopy(record)
    del missing["subruns"]["uninterrupted"]["learning"]["after"]
    missing_path.write_text(json.dumps(missing, indent=2, allow_nan=False) + "\n")
    print("Created two SCHEMA TEST ONLY fixtures; no campaign, model or GPU executed.")
