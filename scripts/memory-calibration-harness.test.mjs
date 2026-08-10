import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

import {
  HARNESS_VERSION, RUNG_REUSE_TOLERANCE, assessProviderReuse, atomicWrite, canonicalJson,
  compareRungReuse, evidenceSemantics, expandPlan, logicalCaseId, mergeBundles, recordId,
  runProviderPlan, validateBundle, validateRecord, validateSourceSessionFiles,
} from "./memory-calibration-harness.mjs";
import { calibrationBinding } from "./generate-memory-matrix.mjs";

// sc-17774: the runner stamps each record with the provider's inference compile-closure digest.
// These tests drive synthetic repositories with no inference crate layout, so the derivation is
// injected here. `inference-closure-digest.test.mjs` covers the real derivation.
const stubClosureDigest = async (provider) =>
  createHash("sha256").update(`closure:${provider ?? "none"}`).digest("hex");


const execFileAsync = promisify(execFile);

async function cleanFixtureRepo() {
  const root = await mkdtemp(path.join(tmpdir(), "memory-harness-repo-"));
  await mkdir(path.join(root, "docs/generated"), { recursive: true });
  await writeFile(
    path.join(root, "docs/generated/memory-matrix.json"),
    JSON.stringify({ generatedFrom: { sceneWorksRevision: `source-tree:${"1".repeat(64)}` } }),
  );
  await execFileAsync("git", ["init", root]);
  await execFileAsync("git", ["-C", root, "config", "user.email", "fixture@example.invalid"]);
  await execFileAsync("git", ["-C", root, "config", "user.name", "Fixture"]);
  await execFileAsync("git", ["-C", root, "add", "docs/generated/memory-matrix.json"]);
  await execFileAsync("git", ["-C", root, "commit", "-m", "fixture"]);
  return root;
}

const phase = (value) => ({
  activeBytes: value,
  allocatorBytes: value + 10,
  deviceBytes: value + 20,
  wiredBytes: value + 30,
  reclaimableBytes: 0,
});

function complete(overrides = {}) {
  const record = {
    logicalCaseId: "",
    status: "complete",
    evidenceScope: "fixture",
    backend: "candle",
    loadShape: "eager_materialization",
    repositories: {
      sceneWorks: {
        revision: "a".repeat(40),
        dirty: false,
        matrixSourceRevision: `source-tree:${"1".repeat(64)}`,
      },
      inference: { revision: "b".repeat(40), dirty: false, closureDigest: "b".repeat(64) },
    },
    hardware: {
      probe: "fixture executable probe",
      memoryBytes: 48 * 1024 ** 3,
      deviceId: "0",
      name: "Fixture CUDA",
      computeCapability: "9.0",
      driverVersion: "999.1",
      runtimeVersion: "12.8",
    },
    artifact: { repository: "SceneWorks/fixture", resolvedRevision: "c".repeat(40), variant: "q4" },
    target: {
      modelId: "krea_2_turbo", provider: "krea_2_turbo", tier: "q4",
      mode: "text_to_image", overlay: "none",
      geometry: { width: 1024, height: 1024, batch: 1, frames: 1 },
    },
    fixture: "fixture-seed42",
    strategy: {
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      parameters: { decodeTileEdge: 512, decodeOverlap: 128 },
    },
    sweep: {
      axes: [{ parameter: "decodeTileEdge", testedValues: [384, 512] }],
      cases: [
        { parameters: { decodeTileEdge: 384, decodeOverlap: 128 }, result: "passed" },
        { parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, result: "passed" },
        { parameters: { decodeTileEdge: 256, decodeOverlap: 32 }, result: "failed" },
      ],
      rangeVerified: true,
    },
    scenarios: [
      { name: "exact_fit", result: "passed", predictedBytes: 200, effectiveBudgetBytes: 200 },
      { name: "unknown_budget", result: "passed" },
      { name: "stale_evidence", result: "passed" },
      { name: "warm_repeat", result: "passed" },
      { name: "cancel", result: "passed", cleanupVerified: true, warmFollowUpPassed: true },
      { name: "error", result: "passed", cleanupVerified: true, warmFollowUpPassed: true },
      { name: "loadability", result: "passed" },
      { name: "overlay", result: "not_applicable", reason: "provider has no overlay" },
    ],
    predictedPeakBytes: { conditioning: 100, denoise: 200, decode: 150, overall: 200 },
    observedMemory: {
      conditioning: phase(100), denoise: phase(200), decode: phase(150), overall: phase(200),
    },
    quality: {
      contract: "tolerance", identicalLatents: true, result: "passed",
      maximumError: 0.01, meanError: 0.001,
      maximumErrorThreshold: 0.08, meanErrorThreshold: 0.01,
    },
    negativeMutation: {
      parameters: { decodeTileEdge: 256, decodeOverlap: 32 }, measured: true,
      result: "failed_as_expected", maximumError: 0.09, meanError: 0.02,
    },
    loadability: { result: "passed", resolvedPathFingerprint: "fixture@resolved:q4" },
    calibrationFingerprint: "fixture-formula-v2",
    capturedAt: "2026-07-28T12:00:00Z",
    harnessVersion: HARNESS_VERSION,
    ...overrides,
  };
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  return record;
}

function runtimeComplete() {
  const record = complete({
    status: "runtime_complete",
    sweep: {
      axes: [{ parameter: "decodeTileEdge", testedValues: [512] }],
      cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, result: "passed" }],
      rangeVerified: true,
    },
    scenarios: [
      { name: "exact_fit", result: "passed", predictedBytes: 200, effectiveBudgetBytes: 200 },
      { name: "unknown_budget", result: "passed" },
      { name: "stale_evidence", result: "passed" },
      { name: "warm_repeat", result: "not_run", reason: "not exercised by the physical runtime campaign" },
      { name: "cancel", result: "not_run", reason: "deferred to exhaustive lifecycle certification" },
      { name: "error", result: "not_run", reason: "deferred to exhaustive lifecycle certification" },
      { name: "loadability", result: "passed" },
      { name: "overlay", result: "not_applicable", reason: "base-only runtime record" },
    ],
    predictedPeakBytes: { overall: 200 },
    observedMemory: { overall: { activeBytes: 200 } },
    quality: {
      contract: "physical final-output parity", identicalInputs: true, result: "passed",
      maximumError: 0.01, meanError: 0.001, rootMeanSquareError: 0.002,
      maximumErrorThreshold: 0.08, meanErrorThreshold: 0.01,
      rootMeanSquareErrorThreshold: 0.02,
    },
    negativeMutation: null,
  });
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  return record;
}

function qwenPositiveComplete() {
  const record = complete({
    evidenceScope: "authoritative",
    backend: "mlx",
    hardware: {
      probe: "fixture Apple hardware probe",
      memoryBytes: 128 * 1024 ** 3,
      model: "Mac16,5",
      chip: "Apple M4 Max",
      osVersion: "15.7",
      metalDevice: "Apple M4 Max",
      mlxMemoryLimitBytes: 96 * 1024 ** 3,
      wiredLimitBytes: 80 * 1024 ** 3,
    },
    target: {
      modelId: "qwen_image", provider: "qwen_image", tier: "bf16",
      mode: "text_to_image", overlay: "none",
      geometry: { width: 1024, height: 1024, batch: 1, frames: 1 },
    },
    loadShape: "deferred_materialization",
    fixture: "qwen-image-bf16-seed15511-step2",
    strategy: {
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      parameters: { decodeTileEdge: 512, decodeOverlap: 64 },
    },
    calibrationFingerprint: "qwen-image-mlx-shared-ladder-2026-08-01-v1",
  });
  record.sweep = {
    axes: [
      { parameter: "decodeTileEdge", testedValues: [512] },
      { parameter: "decodeOverlap", testedValues: [64] },
    ],
    cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 64 }, result: "passed" }],
    rangeVerified: true,
  };
  record.negativeMutation.parameters = { decodeTileEdge: 512, decodeOverlap: 64 };
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  return record;
}

test("complete record validates and identity includes evidence scope plus resolved provenance", () => {
  const record = complete();
  assert.equal(validateRecord(record), record);
  for (const mutate of [
    (r) => (r.evidenceScope = "authoritative"),
    (r) => (r.repositories.inference.revision = "d".repeat(40)),
    (r) => (r.hardware.driverVersion = "different"),
    (r) => (r.artifact.resolvedRevision = "different"),
    (r) => (r.strategy.engagedRungs = ["resident", "staged_residency", "bounded_decode"]),
  ]) {
    const changed = structuredClone(record);
    mutate(changed);
    assert.notEqual(recordId(changed), record.id);
  }
});

test("runtime-complete accepts an honest overall CUDA high-water mark without fabricated phases", () => {
  const record = runtimeComplete();
  assert.equal(validateRecord(record), record);
  validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [record] });

  const malformed = runtimeComplete();
  malformed.observedMemory.overall.deviceBytes = malformed.observedMemory.overall.activeBytes;
  malformed.id = recordId(malformed);
  assert.throws(() => validateRecord(malformed), /only the measured activeBytes/);

  const overCapacity = runtimeComplete();
  overCapacity.hardware.memoryBytes = 100;
  overCapacity.id = recordId(overCapacity);
  assert.throws(() => validateRecord(overCapacity), /exceed probed hardware/);
});

test("complete status still rejects overall-only runtime telemetry", () => {
  const record = runtimeComplete();
  record.status = "complete";
  record.id = recordId(record);
  assert.throws(() => validateRecord(record), /conditioning/);
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [record] }),
    /schema validation failed/,
  );
});

test("complete final-output quality may attest identical inputs without claiming identical latents", () => {
  const record = complete();
  record.quality.identicalInputs = true;
  record.quality.identicalLatents = false;
  assert.equal(validateRecord(record), record);

  delete record.quality.identicalInputs;
  assert.throws(() => validateRecord(record), /identical latents or identical inputs/);
});

test("the Krea adapter's complete fragment shape passes the real harness with singleton axes", async () => {
  const adapter = await readFile(
    fileURLToPath(new URL("../crates/sceneworks-memory-adapter/src/bin/mlx.rs", import.meta.url)),
    "utf8",
  );
  const kreaArm = adapter.slice(
    adapter.indexOf("fn run_krea_base(request:"),
    adapter.indexOf("fn run_krea_control", adapter.indexOf("fn run_krea_base(request:")),
  );
  assert.match(kreaArm, /"status": "complete"/);
  assert.match(kreaArm, /"sweep": krea_base_complete_sweep\(request\)\?/);
  assert.match(kreaArm, /"negativeMutation": \{/);

  const record = qwenPositiveComplete();
  record.target = {
    modelId: "krea_2_turbo", provider: "krea_2_turbo", tier: "q4",
    mode: "text_to_image", overlay: "none",
    geometry: { width: 768, height: 768, batch: 1, frames: 1 },
  };
  record.fixture = "krea-base-mlx-q4-768-seed18377-step2";
  record.calibrationFingerprint =
    "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3";
  record.strategy = {
    rung: "bounded_transformer_residency",
    engagedRungs: [
      "resident", "staged_residency", "bounded_decode", "bounded_attention",
      "bounded_transformer_residency",
    ],
    parameters: {
      decodeTileEdge: 512,
      decodeOverlap: 64,
      attentionChunkSize: 67_108_864,
      transformerWindowSize: 1,
    },
  };
  record.sweep = {
    axes: Object.entries(record.strategy.parameters).map(([parameter, value]) => ({
      parameter,
      testedValues: [value],
    })),
    cases: [{ parameters: structuredClone(record.strategy.parameters), result: "passed" }],
    rangeVerified: true,
  };
  record.negativeMutation.parameters = structuredClone(record.strategy.parameters);
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  assert.equal(validateRecord(record), record);

  const oldEmptySweep = structuredClone(record);
  oldEmptySweep.sweep.axes = [];
  assert.throws(
    () => validateRecord(oldEmptySweep),
    /parameterized complete strategy must sweep at least one axis/,
  );
});

test("the SDXL adapter's runtime-complete fragment passes the real harness without inventing lifecycle coverage", async () => {
  const adapter = await readFile(
    fileURLToPath(new URL("../crates/sceneworks-memory-adapter/src/bin/mlx.rs", import.meta.url)),
    "utf8",
  );
  const start = adapter.indexOf("fn run_sdxl(request:");
  const sdxlArm = adapter.slice(start, adapter.indexOf("fn run_krea_control", start));
  assert.match(sdxlArm, /"status": "runtime_complete"/);
  assert.match(sdxlArm, /"sweep": sdxl_runtime_complete_sweep\(request\)\?/);
  assert.match(sdxlArm, /"warm_repeat", "result": "not_run"/);
  assert.match(sdxlArm, /"cancel", "result": "not_run"/);
  assert.match(sdxlArm, /"error", "result": "not_run"/);
  assert.match(sdxlArm, /"negativeMutation": null/);
  assert.match(sdxlArm, /"rootMeanSquareError": rms_error/);
  assert.match(sdxlArm, /negativeMutationRootMeanSquareErrorPer255/);

  const record = runtimeComplete();
  record.backend = "mlx";
  record.loadShape = "deferred_materialization";
  record.hardware = {
    probe: "fixture sysctl and MLX allocator probe",
    memoryBytes: 128 * 1024 ** 3,
    model: "Mac16,5",
    chip: "Apple M4 Max",
    osVersion: "macOS 26.0",
    metalDevice: "Apple M4 Max",
    mlxMemoryLimitBytes: 96 * 1024 ** 3,
    wiredLimitBytes: 64 * 1024 ** 3,
  };
  record.artifact = {
    repository: "SceneWorks/sdxl-base-mlx",
    resolvedRevision: "d".repeat(40),
    variant: "q4",
  };
  record.target = {
    modelId: "sdxl", provider: "sdxl", tier: "q4",
    mode: "text_to_image", overlay: "none",
    geometry: { width: 768, height: 768, batch: 1, frames: 1 },
  };
  record.fixture = "sdxl-base-mlx-q4-768-seed18379-step2";
  record.calibrationFingerprint = "sdxl-mlx-unet-shared-ladder-v3";
  record.strategy = {
    rung: "bounded_transformer_residency",
    engagedRungs: ["resident", "staged_residency", "bounded_transformer_residency"],
    parameters: { transformerWindowSize: 1, transformerWindowComponent: "dit" },
  };
  record.sweep = {
    axes: [{ parameter: "transformerWindowSize", testedValues: [1] }],
    cases: [{ parameters: structuredClone(record.strategy.parameters), result: "passed" }],
    rangeVerified: true,
  };
  record.predictedPeakBytes = { conditioning: 100, denoise: 200, decode: 150, overall: 200 };
  record.observedMemory = {
    conditioning: phase(100), denoise: phase(200), decode: phase(150), overall: phase(200),
  };
  record.loadability = {
    result: "passed",
    resolvedPathFingerprint: `SceneWorks/sdxl-base-mlx@${"d".repeat(40)}:q4`,
  };
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  assert.equal(validateRecord(record), record);
  validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [record] });

  const stringAxis = structuredClone(record);
  stringAxis.sweep.axes.push({ parameter: "transformerWindowComponent", testedValues: ["dit"] });
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [stringAxis] }),
    /schema validation failed/,
  );
});

test("complete status fails closed on scenario, quality, mutation, memory and loadability mutations", () => {
  const mutations = [
    [(r) => (r.scenarios.find((x) => x.name === "warm_repeat").result = "not_run"), /warm_repeat/],
    [(r) => (r.scenarios.push(structuredClone(r.scenarios[0]))), /unique/],
    [(r) => (r.scenarios.find((x) => x.name === "exact_fit").effectiveBudgetBytes = 201), /equality/],
    [(r) => (r.scenarios.find((x) => x.name === "cancel").cleanupVerified = false), /clean up/],
    [(r) => (r.quality.result = "not_run"), /quality/],
    [(r) => (r.negativeMutation.measured = false), /measured/],
    [(r) => { r.negativeMutation.maximumError = 0.01; r.negativeMutation.meanError = 0.001; }, /breach/],
    [(r) => (r.observedMemory.overall.deviceBytes = 1), /cover/],
    [(r) => (r.observedMemory.decode.allocatorBytes = 1), /cover active/],
    [(r) => (r.observedMemory.decode.reclaimableBytes = 999), /reclaimable/],
    [(r) => (r.hardware.memoryBytes = 100), /exceed probed hardware/],
    [(r) => (r.loadability.resolvedPathFingerprint = ""), /non-empty/],
    [(r) => (r.quality.contract = ""), /non-empty/],
    [(r) => (r.repositories.sceneWorks.dirty = true), /dirty/],
    [(r) => (r.sweep.axes = []), /parameterized complete strategy must sweep/],
    [(r) => r.sweep.axes.push(structuredClone(r.sweep.axes[0])), /axes must be unique/],
    [(r) => r.sweep.cases.push(structuredClone(r.sweep.cases[0])), /cases must be unique/],
  ];
  for (const [mutate, pattern] of mutations) {
    const record = complete();
    mutate(record);
    record.id = recordId(record);
    assert.throws(() => validateRecord(record), pattern);
  }
});

// sc-18100: rehomed from `sc-15823-flux1-evidence.test.mjs`, which was the ONLY coverage of these
// three `validateRuntimeComplete` rejection paths. They gate the SURVIVING harness, not the deleted
// one-shot, so deleting that file without this test would have dropped the gates silently.
test("runtime-complete fails closed on promoted lifecycle scenarios, overlay targets and extra cases", () => {
  // A runtime-complete record attests only that the rung ran. Promoting an unexercised lifecycle
  // scenario to `passed` is the overclaim the status exists to prevent.
  for (const name of ["warm_repeat", "cancel", "error"]) {
    const record = runtimeComplete();
    record.scenarios.find((scenario) => scenario.name === name).result = "passed";
    record.id = recordId(record);
    assert.throws(() => validateRecord(record), new RegExp(`${name} must remain explicitly not_run`));
  }

  // Runtime-complete evidence is base-only: an overlay coordinate may never be attested this way.
  const overlay = runtimeComplete();
  overlay.target.overlay = "identity";
  overlay.logicalCaseId = logicalCaseId(overlay);
  overlay.id = recordId(overlay);
  assert.throws(() => validateRecord(overlay), /base-only none overlay/);

  // Exactly one passed case, and it must be the strategy's own parameters — a second case would
  // silently widen the attested domain, a mismatched one would attest a different configuration.
  const secondCase = runtimeComplete();
  secondCase.sweep.cases.push({ parameters: { unexpected: 1 }, result: "passed" });
  assert.throws(
    () => validateRecord(secondCase),
    /exactly one passed case matching its strategy parameters/,
  );
  assert.throws(
    () => validateBundle({
      schemaVersion: 4, harnessVersion: HARNESS_VERSION, sourceSessions: [], records: [secondCase],
    }),
    /schema validation failed: .*sweep\.cases: array has too many items/,
  );

  const mismatch = runtimeComplete();
  mismatch.sweep.cases[0].parameters = { unexpected: 1 };
  assert.throws(
    () => validateRecord(mismatch),
    /exactly one passed case matching its strategy parameters/,
  );
  assert.throws(
    () => validateBundle({
      schemaVersion: 4, harnessVersion: HARNESS_VERSION, sourceSessions: [], records: [mismatch],
    }),
    /exactly one passed case matching its strategy parameters/,
  );
});

test("range axes derive exactly from passed cases and contain exact strategy parameters", () => {
  const unrun = complete();
  unrun.sweep.axes[0].testedValues.push(640);
  assert.throws(() => validateRecord(unrun), /derived from passed/);
  const absent = complete();
  absent.sweep.cases = absent.sweep.cases.filter((item) => item.parameters.decodeTileEdge !== 512);
  assert.throws(() => validateRecord(absent), /exact strategy parameters/);
});

test("a singleton production parameter domain is valid complete evidence", () => {
  const record = complete();
  record.sweep = {
    axes: [
      { parameter: "decodeTileEdge", testedValues: [512] },
      { parameter: "decodeOverlap", testedValues: [128] },
    ],
    cases: [{
      parameters: { decodeTileEdge: 512, decodeOverlap: 128 },
      result: "passed",
    }],
    rangeVerified: true,
  };
  record.id = recordId(record);
  assert.equal(validateBundle({
    schemaVersion: 4,
    harnessVersion: HARNESS_VERSION,
    records: [record],
  }).records[0], record);
});

test("merge is commutative and rejects conflicting exact-identity captures", () => {
  const first = complete();
  const second = complete({
    fixture: "fixture-seed43",
    capturedAt: "2026-07-28T13:00:00Z",
  });
  second.logicalCaseId = logicalCaseId(second);
  second.id = recordId(second);
  const a = { schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [first] };
  const b = { schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [second] };
  assert.equal(canonicalJson(mergeBundles(a, b)), canonicalJson(mergeBundles(b, a)));
  const conflict = structuredClone(first);
  conflict.capturedAt = "2026-07-28T14:00:00Z";
  assert.throws(
    () => mergeBundles(a, { schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [conflict] }),
    /conflicting record/,
  );
});

test("fresh and reused rung captures use a committed absolute-or-relative tolerance", () => {
  const freshRecord = complete();
  const reusedRecord = structuredClone(freshRecord);
  const withinTolerance = RUNG_REUSE_TOLERANCE.absoluteBytes;
  for (const phaseName of ["conditioning", "denoise", "decode", "overall"]) {
    for (const metric of ["activeBytes", "allocatorBytes", "deviceBytes", "wiredBytes"]) {
      reusedRecord.observedMemory[phaseName][metric] += withinTolerance;
    }
  }
  const fresh = { schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [freshRecord] };
  const reused = { schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [reusedRecord] };
  assert.equal(compareRungReuse(fresh, reused).verdict, "amortizable");

  reusedRecord.observedMemory.conditioning.activeBytes += 1;
  reusedRecord.observedMemory.conditioning.allocatorBytes += 1;
  reusedRecord.observedMemory.conditioning.deviceBytes += 1;
  reusedRecord.observedMemory.conditioning.wiredBytes += 1;
  assert.equal(compareRungReuse(fresh, reused).verdict, "unable_to_amortize");

  const differentHardware = structuredClone(reusedRecord);
  differentHardware.hardware.driverVersion = "different";
  differentHardware.id = recordId(differentHardware);
  assert.throws(
    () => compareRungReuse(fresh, { ...reused, records: [differentHardware] }),
    /comparison domain differs/,
  );
});

test("gated and fixture semantics can never become current", () => {
  const fixture = complete();
  assert.equal(evidenceSemantics(fixture, {
    sceneWorks: fixture.repositories.sceneWorks.revision,
    inference: fixture.repositories.inference.revision,
  }), "fixture");
  const gated = complete({ status: "gated", evidenceScope: "authoritative" });
  gated.logicalCaseId = logicalCaseId(gated);
  gated.id = recordId(gated);
  assert.equal(evidenceSemantics(gated, {
    sceneWorks: gated.repositories.sceneWorks.revision,
    inference: gated.repositories.inference.revision,
  }), "gated");
});

test("candidate evidence is permanently non-promotable", () => {
  const candidate = complete({ status: "gated", evidenceScope: "candidate" });
  candidate.logicalCaseId = logicalCaseId(candidate);
  candidate.id = recordId(candidate);
  assert.equal(evidenceSemantics(candidate, {
    sceneWorks: candidate.repositories.sceneWorks.matrixSourceRevision,
    inference: candidate.repositories.inference.revision,
  }), "candidate");
});

test("currency follows the provider's own compile closure, never the inference pin (sc-17774)", () => {
  const record = complete({ evidenceScope: "authoritative" });
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  // Keyed `<backend>:<provider>` — a provider id is not unique across backends.
  const provider = `${record.backend}:${record.target.provider}`;
  const captured = record.repositories.inference.closureDigest;
  const revisions = {
    sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
    inference: record.repositories.inference.revision,
    inferenceClosureDigests: { [provider]: captured },
  };
  assert.equal(evidenceSemantics(record, revisions), "current");

  // The whole point: the pin may move arbitrarily far and the measurement stays in force, as long as
  // THIS provider's closure did not move. Previously this exact case returned "historical", and it
  // is why every calibration was demoted ~1.5 times a day.
  assert.equal(
    evidenceSemantics(record, { ...revisions, inference: "c".repeat(40) }),
    "current",
    "an unrelated inference commit must not demote a measurement",
  );
  assert.equal(
    evidenceSemantics(record, { ...revisions, sceneWorks: "source-tree:different" }),
    "current",
    "matrixSourceRevision is exact provenance; calibrationBinding separately enforces the SceneWorks ABI fingerprint",
  );

  // ...and the unit is not blind: this provider's closure moving DOES demote it.
  assert.equal(
    evidenceSemantics(record, {
      ...revisions,
      inferenceClosureDigests: { [provider]: "d".repeat(64) },
    }),
    "historical",
  );

  // Another provider's closure moving is not this provider's business at all.
  assert.equal(
    evidenceSemantics(record, {
      ...revisions,
      inferenceClosureDigests: { [provider]: captured, some_other_model: "e".repeat(64) },
    }),
    "current",
  );
});

test("current Qwen q4 and bf16 evidence cannot omit physical MLX provenance", async () => {
  for (const tier of ["q4", "bf16"]) {
    const record = qwenPositiveComplete();
    record.target.tier = tier;
    record.artifact.variant = tier;
    record.logicalCaseId = logicalCaseId(record);
    record.id = recordId(record);
    const provider = `${record.backend}:${record.target.provider}`;
    const live = record.repositories.inference.closureDigest;
    const revisions = { inferenceClosureDigests: { [provider]: live } };

    assert.throws(
      () => evidenceSemantics(record, revisions),
      /current authoritative Qwen MLX .* requires sourceProvenance=physical_mlx_v1/,
    );
    await assert.rejects(
      validateSourceSessionFiles(
        { schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [record] },
        null,
        revisions.inferenceClosureDigests,
      ),
      /current authoritative Qwen MLX .* requires sourceProvenance=physical_mlx_v1/,
    );

    assert.equal(
      evidenceSemantics(record, {
        inferenceClosureDigests: { [provider]: "f".repeat(64) },
      }),
      "historical",
      "pre-provenance captures remain valid history",
    );
  }

  const q8 = qwenPositiveComplete();
  q8.target.tier = "q8";
  q8.artifact.variant = "q8";
  q8.logicalCaseId = logicalCaseId(q8);
  q8.id = recordId(q8);
  const q8Provider = `${q8.backend}:${q8.target.provider}`;
  assert.equal(
    evidenceSemantics(q8, {
      inferenceClosureDigests: {
        [q8Provider]: q8.repositories.inference.closureDigest,
      },
    }),
    "current",
    "the retained q8 campaign is outside the q4/bf16 recapture requirement",
  );
});

test("a record with no closure digest fails loudly instead of falling back to pin equality", () => {
  // The fallback would be invisible in a green run and would silently restore the old policy.
  const record = complete({ evidenceScope: "authoritative" });
  delete record.repositories.inference.closureDigest;
  assert.throws(
    () =>
      evidenceSemantics(record, {
        sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
        inference: record.repositories.inference.revision,
        inferenceClosureDigests: {
          [`${record.backend}:${record.target.provider}`]: "f".repeat(64),
        },
      }),
    /closureDigest/,
  );
});

test("an undeclared provider fails loudly rather than being treated as current", () => {
  const record = complete({ evidenceScope: "authoritative" });
  assert.throws(
    () =>
      evidenceSemantics(record, {
        sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
        inference: record.repositories.inference.revision,
        inferenceClosureDigests: {},
      }),
    /inference-provider-closures\.json/,
  );
});

test("matrix binding rejects batch and frame mismatches even when width and height match", () => {
  const record = complete({ evidenceScope: "authoritative" });
  const cell = {
    calibrationFingerprint: record.calibrationFingerprint,
    engagedRungs: record.strategy.engagedRungs,
    strategyParameters: record.strategy.parameters,
    geometryEnvelope: { resolutions: ["1024x1024"] },
    evidence: {
      loadability: [{
        repository: record.artifact.repository,
        revision: record.artifact.resolvedRevision,
        variant: record.artifact.variant,
      }],
    },
  };
  assert.equal(calibrationBinding(record, cell).eligible, true);
  const batch = structuredClone(record);
  batch.target.geometry.batch = 2;
  assert.ok(calibrationBinding(batch, cell).reasons.includes("batch-out-of-envelope"));
  const frames = structuredClone(record);
  frames.target.geometry.frames = 2;
  assert.ok(calibrationBinding(frames, cell).reasons.includes("frames-out-of-envelope"));
  const composition = structuredClone(record);
  composition.strategy.engagedRungs = [
    "resident",
    "staged_residency",
    "bounded_decode",
  ];
  assert.ok(calibrationBinding(composition, cell).reasons.includes("composition-mismatch"));
});

test("Qwen plan covers the BF16 ladder plus Q4/Q8 rung-3-versus-rung-4 pairs", async () => {
  const config = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const cases = expandPlan(config);
  const qwen = cases.filter(
    (item) => item.target.provider === "qwen_image" && item.backend === "mlx",
  );
  assert.equal(qwen.length, 15);
  assert.ok(qwen.every((item) => item.expectedResult === "passed" && !item.negative));
  assert.deepEqual(
    [...new Set(qwen.map((item) => item.strategy.rung))].sort(),
    ["resident", "staged_residency", "bounded_decode", "bounded_attention", "bounded_transformer_residency"].sort(),
  );
  assert.ok(
    qwen.every(
      (item) => item.calibrationFingerprint === "qwen-image-mlx-shared-ladder-2026-08-01-v1",
    ),
    "load shape is a separate evidence-key axis and must not be encoded in the provider fingerprint",
  );
  assert.ok(
    qwen.every(
      (item) => item.loadShape === "deferred_materialization"
    ),
    "every Qwen capture case must use the production deferred shape",
  );
  assert.ok(
    qwen.every((item) =>
      item.target.tier === "q8"
        ? item.sourceProvenance === undefined
        : item.sourceProvenance === "physical_mlx_v1"),
    "new q4/bf16 captures must require physical MLX provenance without rewriting retained q8 evidence",
  );
  assert.deepEqual(
    qwen
      .filter((item) => item.strategy.rung === "bounded_decode")
      .map((item) => item.strategy.parameters.decodeTileEdge)
      .sort((left, right) => right - left),
    [768, 640, 512, 448, 384, 320, 256],
  );
  for (const tier of ["q4", "q8"]) {
    const packed = qwen.filter((item) => item.target.tier === tier);
    assert.equal(packed.length, 2);
    assert.deepEqual(
      packed.map((item) => item.strategy.rung).sort(),
      ["bounded_attention", "bounded_transformer_residency"],
    );
    assert.ok(packed.every((item) => item.fixture === `qwen-image-${tier}-seed16353-step2`));
  }
});

test("FLUX.2-dev MLX plan is the reference-free T2I q4/q8 resident matrix", async () => {
  const config = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)),
  );
  const flux2 = expandPlan(config).filter(
    (item) => item.backend === "mlx" && item.target.provider === "flux2_dev",
  );

  assert.equal(flux2.length, 4);
  assert.deepEqual(
    flux2.map((item) => [
      item.target.tier,
      item.target.geometry.width,
      item.target.geometry.height,
      item.strategy.rung,
    ]).sort(),
    [
      ["q4", 768, 768, "resident"],
      ["q4", 1024, 1024, "resident"],
      ["q8", 768, 768, "resident"],
      ["q8", 1024, 1024, "resident"],
    ].sort(),
  );
  assert.ok(flux2.every((item) =>
    item.evidenceScope === "authoritative" &&
    item.loadShape === "eager_materialization" &&
    item.target.modelId === "flux2_dev" &&
    item.target.mode === "text_to_image" &&
    item.target.overlay === "none" &&
    item.target.geometry.batch === 1 &&
    item.target.geometry.frames === 1 &&
    item.strategy.engagedRungs.length === 1 &&
    item.strategy.engagedRungs[0] === "resident" &&
    Object.keys(item.strategy.parameters).length === 0 &&
    item.calibrationFingerprint === "sc-18218-flux2-dev-t2i-resident-evidence-v1"
  ));
  assert.ok(flux2.every((item) =>
    item.fixture ===
      `flux2-dev-mlx-${item.target.tier}-${item.target.geometry.width}-seed18218-step2`
  ));
});

test("plain MLX Krea plan covers q4, q8, and bf16 across the exact five-rung contract", async () => {
  const config = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)),
  );
  const expectedRungs = [
    "resident",
    "staged_residency",
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
  ];
  const expectedCompositions = {
    resident: ["resident"],
    staged_residency: ["resident", "staged_residency"],
    bounded_decode: ["resident", "bounded_decode"],
    bounded_attention: ["resident", "bounded_decode", "bounded_attention"],
    bounded_transformer_residency: [
      "resident",
      "staged_residency",
      "bounded_decode",
      "bounded_attention",
      "bounded_transformer_residency",
    ],
  };
  const assertPlan = (candidate) => {
    const krea = expandPlan(candidate).filter(
      (item) => item.backend === "mlx" && item.target.provider === "krea_2_turbo",
    );
    assert.equal(krea.length, 15, "three tiers must each publish exactly five planned rungs");
    for (const [tier, edge] of [["q4", 768], ["q8", 1024], ["bf16", 1024]]) {
      const cases = krea.filter((item) => item.target.tier === tier);
      assert.deepEqual(cases.map((item) => item.strategy.rung).sort(), [...expectedRungs].sort());
      assert.ok(cases.every((item) =>
        item.evidenceScope === "authoritative" &&
        item.loadShape === "deferred_materialization" &&
        item.target.modelId === "krea_2_turbo" &&
        item.target.mode === "text_to_image" &&
        item.target.overlay === "none" &&
        item.target.geometry.width === edge &&
        item.target.geometry.height === edge &&
        item.target.geometry.batch === 1 &&
        item.target.geometry.frames === 1 &&
        item.calibrationFingerprint ===
          "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3" &&
        item.fixture === `krea-base-mlx-${tier}-${edge}-seed18377-step2` &&
        JSON.stringify(item.strategy.engagedRungs) ===
          JSON.stringify(expectedCompositions[item.strategy.rung])
      ), `${tier} plain Krea entries must preserve the exact capture tuple and composition`);
    }
  };
  assertPlan(config);

  const missingTierRung = structuredClone(config);
  missingTierRung.providers = missingTierRung.providers.filter(
    (provider) => provider.name !== "mlx-krea-base-q8-bounded-transformer-1024",
  );
  assert.throws(() => assertPlan(missingTierRung), /exactly five/);

  const wrongSurface = structuredClone(config);
  wrongSurface.providers.find(
    (provider) => provider.name === "mlx-krea-base-q4-resident-768",
  ).target.overlay = "control:1";
  assert.throws(() => assertPlan(wrongSurface), /plain Krea entries/);
});

test("plain MLX SDXL plan covers every shipped tier without inventing measured-Missing rungs", async () => {
  const config = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)),
  );
  const expectedRungs = ["resident", "staged_residency", "bounded_transformer_residency"];
  const expectedCompositions = {
    resident: ["resident"],
    staged_residency: ["resident", "staged_residency"],
    bounded_transformer_residency: [
      "resident", "staged_residency", "bounded_transformer_residency",
    ],
  };
  const assertPlan = (candidate) => {
    const sdxl = expandPlan(candidate).filter(
      (item) => item.backend === "mlx" && item.target.provider === "sdxl",
    );
    assert.equal(sdxl.length, 18, "three tiers must each publish two base rungs plus four window cases");
    for (const [tier, edge] of [["q4", 768], ["q8", 1024], ["bf16", 1024]]) {
      const cases = sdxl.filter((item) => item.target.tier === tier);
      assert.equal(cases.length, 6, `${tier} must carry Resident, Staged, and four window cases`);
      assert.deepEqual(
        [...new Set(cases.map((item) => item.strategy.rung))].sort(),
        [...expectedRungs].sort(),
      );
      assert.ok(cases.every((item) =>
        item.evidenceScope === "authoritative" &&
        item.loadShape === "deferred_materialization" &&
        item.target.modelId === "sdxl" &&
        item.target.mode === "text_to_image" &&
        item.target.overlay === "none" &&
        item.target.geometry.width === edge &&
        item.target.geometry.height === edge &&
        item.target.geometry.batch === 1 &&
        item.target.geometry.frames === 1 &&
        item.calibrationFingerprint === "sdxl-mlx-unet-shared-ladder-v3" &&
        item.fixture === `sdxl-base-mlx-${tier}-${edge}-seed18379-step2` &&
        JSON.stringify(item.strategy.engagedRungs) ===
          JSON.stringify(expectedCompositions[item.strategy.rung]) &&
        (item.strategy.rung !== "bounded_transformer_residency" ||
          ([1, 2, 5, 10].includes(item.strategy.parameters.transformerWindowSize) &&
            item.strategy.parameters.transformerWindowComponent === "dit"))
      ), `${tier} SDXL entries must preserve the exact base T2I capture tuple`);
      assert.deepEqual(
        cases
          .filter((item) => item.strategy.rung === "bounded_transformer_residency")
          .map((item) => item.strategy.parameters.transformerWindowSize)
          .sort((left, right) => left - right),
        [1, 2, 5, 10],
        `${tier} must schedule every provider-implemented SDXL cadence`,
      );
      assert.ok(
        cases.every((item) => !["bounded_decode", "bounded_attention"].includes(item.strategy.rung)),
        `${tier} must not plan measured-Missing SDXL rungs`,
      );
    }
  };
  assertPlan(config);

  const missingRung = structuredClone(config);
  missingRung.providers = missingRung.providers.filter(
    (provider) => provider.name !== "mlx-sdxl-base-q8-bounded-transformer-window5-1024",
  );
  assert.throws(() => assertPlan(missingRung), /four window cases|every provider-implemented/);

  const inventedRung = structuredClone(config);
  inventedRung.providers.find(
    (provider) => provider.name === "mlx-sdxl-base-q4-resident-768",
  ).rung = "bounded_decode";
  assert.throws(() => assertPlan(inventedRung));
});

test("shipped five-rung oracles stay fresh after backend reuse verdicts", async () => {
  const config = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const expectedRungs = [
    "resident",
    "staged_residency",
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
  ];
  const assertLadder = (cases, backend, expectedTarget, expectedCompositions) => {
    assert.equal(cases.length, 5, `${backend} must declare exactly five fresh-reference cases`);
    assert.deepEqual(
      cases.map((item) => item.strategy.rung).sort(),
      [...expectedRungs].sort(),
      `${backend} must declare every rung exactly once`,
    );
    assert.equal(
      new Set(cases.map((item) => JSON.stringify(item.target))).size,
      1,
      `${backend} cases must keep one exact target tuple`,
    );
    assert.deepEqual(cases[0].target, expectedTarget);
    for (const item of cases) {
      assert.deepEqual(
        item.strategy.engagedRungs,
        expectedCompositions[item.strategy.rung],
        `${backend} ${item.strategy.rung} composition`,
      );
    }
  };
  const assertPlan = (candidate) => {
    const cases = expandPlan(candidate);
    assertLadder(
      cases.filter((item) => item.fixture === "fresh-five-rung-z-image-q4-768-seed16402-step2"),
      "mlx",
      {
        provider: "z_image_turbo", modelId: "z_image_turbo", tier: "q4",
        mode: "text_to_image", overlay: "none",
        geometry: { width: 768, height: 768, batch: 1, frames: 1 },
      },
      {
        resident: ["resident"],
        staged_residency: ["resident", "staged_residency"],
        bounded_decode: ["resident", "bounded_decode"],
        bounded_attention: ["resident", "bounded_decode", "bounded_attention"],
        bounded_transformer_residency: [
          "resident", "bounded_decode", "bounded_attention", "bounded_transformer_residency",
        ],
      },
    );
    const mlx = cases.filter(
      (item) => item.fixture === "fresh-five-rung-z-image-q4-768-seed16402-step2",
    );
    assert.ok(mlx.every((item) => item.modelLoadPolicy === "fresh_per_case"));
    assert.deepEqual(
      [...new Set(mlx.map((item) => item.calibrationFingerprint))],
      ["z-image-mlx-independent-materialization-v3"],
      "load shape is a typed receipt axis, not part of the provider content fingerprint",
    );
    assert.equal(mlx.filter((item) => item.loadShape === "eager_materialization").length, 4);
    assert.equal(mlx.filter((item) => item.loadShape === "deferred_materialization").length, 1);
    const candle = cases.filter(
      (item) => item.fixture === "fresh-five-rung-krea-q4-1024-seed16402-step2",
    );
    assertLadder(
      candle,
      "candle",
      {
        provider: "krea_2_turbo", modelId: "krea_2_turbo", tier: "q4",
        mode: "text_to_image", overlay: "none",
        geometry: { width: 1024, height: 1024, batch: 1, frames: 1 },
      },
      {
        resident: ["resident"],
        staged_residency: ["resident", "staged_residency"],
        bounded_decode: ["resident", "staged_residency", "bounded_decode"],
        bounded_attention: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
        bounded_transformer_residency: [
          "resident", "staged_residency", "bounded_decode", "bounded_attention",
          "bounded_transformer_residency",
        ],
      },
    );
    assert.ok(candle.every((item) => item.modelLoadPolicy === "fresh_per_case"));
    assert.ok(candle.every((item) => item.modelLoadGroup === null));
  };
  assertPlan(config);

  const missingRung = structuredClone(config);
  missingRung.providers = missingRung.providers.filter(
    (provider) => provider.name !== "mlx-z-image-q4-fresh-reference-bounded-attention",
  );
  assert.throws(() => assertPlan(missingRung), /exactly five/);

  const wrongComposition = structuredClone(config);
  wrongComposition.providers.find(
    (provider) => provider.name === "candle-krea-q4-fresh-reference-bounded-transformer",
  ).engagedRungs = ["resident", "bounded_decode", "bounded_attention", "bounded_transformer_residency"];
  assert.throws(() => assertPlan(wrongComposition), /composition/);
});

test("Krea current v1 production truth is separate from non-promotable v2 candidates", async () => {
  const config = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const current = config.providers.find((provider) => provider.name === "candle-krea-production-current-v1");
  assert.equal(current.evidenceScope, "authoritative");
  assert.equal(current.calibrationFingerprint, "krea-turbo-cuda-phase-curves-v1");
  assert.deepEqual(current.cases, [{
    parameters: {
      decodeTileEdge: 512,
      decodeOverlap: 128,
      attentionChunkSize: 134217728,
      transformerWindowSize: 1,
    },
    expectedResult: "passed",
  }]);
  const candidates = config.providers.find((provider) => provider.name === "candle-krea-v2-candidates");
  assert.equal(candidates.evidenceScope, "candidate");
  assert.equal(candidates.calibrationFingerprint, "krea-turbo-cuda-phase-curves-v2");
  assert.equal(candidates.cases.length, 2);
});

test("provider execution requires one backend-specific hardware probe", async () => {
  const config = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  await assert.rejects(
    runProviderPlan({
    closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [process.execPath, "must-not-start.mjs"],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    }),
    /select exactly one backend/,
  );
});

test("a legacy Qwen completion cannot suppress a provenance-required recapture", async () => {
  const config = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const record = qwenPositiveComplete();
  validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [record] });
  const provenanceRequired = structuredClone(record);
  provenanceRequired.sourceProvenance = "physical_mlx_v1";
  provenanceRequired.logicalCaseId = logicalCaseId(provenanceRequired);
  provenanceRequired.id = recordId(provenanceRequired);
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [provenanceRequired] }),
    /exact artifact inventory|missing source-session derivation/,
  );
  const qwenRemaining = expandPlan(config, [record]).filter(
    (item) => item.target.provider === "qwen_image" && item.backend === "mlx",
  );
  assert.equal(qwenRemaining.length, 15);
  assert.equal(
    qwenRemaining.some(
      (item) =>
        item.strategy.rung === "bounded_decode" &&
        item.strategy.parameters.decodeTileEdge === 512 &&
        item.strategy.parameters.decodeOverlap === 64,
    ),
    true,
  );
  assert.ok(qwenRemaining.some((item) => item.strategy.rung === "bounded_attention"));
  assert.ok(qwenRemaining.some((item) => item.strategy.rung === "bounded_transformer_residency"));
});

test("runtime bundle validation matches schema closure for malformed gated and nested values", () => {
  const valid = { schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [complete()] };
  validateBundle(valid);
  const mutations = [
    (bundle) => (bundle.unexpected = true),
    (bundle) => (bundle.records[0].hardware.unexpected = true),
    (bundle) => (bundle.records[0].sweep.axes[0].testedValues = ["wrong"]),
    (bundle) => (bundle.records[0].sweep.cases[0].parameters = "wrong"),
    (bundle) => { bundle.records[0].status = "gated"; delete bundle.records[0].artifact; },
  ];
  for (const mutate of mutations) {
    const invalid = structuredClone(valid);
    mutate(invalid);
    assert.throws(() => validateBundle(invalid), /schema validation failed/);
  }
});

test("authoritative Z-Image evidence requires dimension-specific source sessions", () => {
  const source = {
    id: `ims-${"1".repeat(20)}`,
    kind: "physical_cuda",
    command: "fixture CUDA probe",
    sourcePath: "docs/calibration/fixture.log",
    capturedAt: "2026-08-01T12:00:00Z",
    repositories: {
      sceneWorks: { revision: "a".repeat(40), dirty: true },
      inference: { revision: "b".repeat(40), dirty: false, closureDigest: "b".repeat(64) },
    },
    hardware: { probe: "nvidia-smi", memoryBytes: 1024 },
    target: {
      tier: "q4", mode: "text_to_image", overlay: "control", rung: "bounded_decode",
    },
    stdoutSha256: "2".repeat(64),
    inputs: [{
      role: "base", path: "fixture/q4", bytes: 1024, sha256: "4".repeat(64),
      repository: "SceneWorks/fixture", resolvedRevision: "c".repeat(40), variant: "q4",
    }, {
      role: "control", path: "fixture/control.safetensors", bytes: 512, sha256: "5".repeat(64),
      repository: "SceneWorks/control", resolvedRevision: "d".repeat(40), variant: "union",
    }],
    outputs: [{ path: "fixture.png", sha256: "3".repeat(64) }],
    claims: ["memory", "quality", "negative_mutation", "lifecycle", "loadability", "overlay"],
    result: "passed",
  };
  const inventory = structuredClone(source);
  inventory.id = `ims-${"6".repeat(20)}`;
  inventory.kind = "static_analysis";
  inventory.command = "fixture exact artifact inventory";
  inventory.sourcePath = "docs/calibration/fixture-inventory.log";
  delete inventory.target;
  inventory.outputs = [];
  inventory.claims = ["loadability", "overlay"];
  const reference = (kind) => ({ kind, sourceSessionIds: [source.id] });
  const record = complete({
    evidenceScope: "authoritative",
    target: {
      modelId: "z_image", provider: "z_image", tier: "q4", mode: "text_to_image",
      overlay: "control", geometry: { width: 1024, height: 1024, batch: 1, frames: 1 },
    },
    loadability: {
      result: "passed",
      resolvedPathFingerprint: `SceneWorks/fixture@${"c".repeat(40)}:q4+SceneWorks/control@${"d".repeat(40)}`,
    },
    derivation: {
      memory: reference("direct"), quality: reference("direct"),
      negativeMutation: reference("shared_implementation"), lifecycle: reference("shared_implementation"),
      loadability: { kind: "direct", sourceSessionIds: [source.id, inventory.id] },
      overlay: { kind: "direct", sourceSessionIds: [source.id, inventory.id] }, justification: "fixture",
    },
  });
  const bundleWith = (candidate) => ({
    schemaVersion: 4, harnessVersion: HARNESS_VERSION,
    sourceSessions: [candidate, inventory], records: [record],
  });
  validateBundle(bundleWith(source));

  const wrongArtifactIdentity = structuredClone(source);
  wrongArtifactIdentity.inputs[0] = {
    ...wrongArtifactIdentity.inputs[0], repository: "attacker/wrong-model",
    resolvedRevision: "deadbeef", sha256: "0".repeat(64), bytes: 1,
  };
  assert.throws(
    () => validateBundle(bundleWith(wrongArtifactIdentity)),
    /input differs from its exact inventory identity/,
  );

  const wrongQualityBytes = structuredClone(source);
  wrongQualityBytes.inputs[0] = {
    ...wrongQualityBytes.inputs[0], path: "D:\\attacker\\different-q4",
    sha256: "0".repeat(64), bytes: 1,
  };
  assert.throws(
    () => validateBundle(bundleWith(wrongQualityBytes)),
    /input differs from its exact inventory identity/,
  );

  const missingInputs = structuredClone(source);
  missingInputs.inputs = [];
  assert.throws(
    () => validateBundle(bundleWith(missingInputs)),
    /artifact claims require exact inputs|schema validation failed/,
  );

  const wrongOverlayInput = structuredClone(source);
  wrongOverlayInput.inputs = wrongOverlayInput.inputs.filter((input) => input.role !== "control");
  assert.throws(
    () => validateBundle(bundleWith(wrongOverlayInput)),
    /artifact claim is missing its exact tier\/overlay inputs/,
  );

  const qualityWithoutInputs = structuredClone(source);
  qualityWithoutInputs.claims = ["quality"];
  qualityWithoutInputs.inputs = [];
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, sourceSessions: [qualityWithoutInputs], records: [] }),
    /artifact claims require exact inputs|schema validation failed/,
  );

  const negativeWithoutControl = structuredClone(source);
  negativeWithoutControl.claims = ["negative_mutation"];
  negativeWithoutControl.inputs = negativeWithoutControl.inputs.filter((input) => input.role !== "control");
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, sourceSessions: [negativeWithoutControl], records: [] }),
    /artifact claim is missing its exact tier\/overlay inputs/,
  );

  const missing = structuredClone(record);
  delete missing.derivation;
  missing.logicalCaseId = logicalCaseId(missing);
  missing.id = recordId(missing);
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, sourceSessions: [source], records: [missing] }),
    /missing source-session derivation/,
  );

  const crossTier = structuredClone(source);
  crossTier.target.tier = "q8";
  crossTier.inputs.find((input) => input.role === "base").variant = "q8";
  assert.throws(
    () => validateBundle(bundleWith(crossTier)),
    /cannot cross precision tiers|has no canonical base\/q8 inventory/,
  );
});

test("gated real-adapter diagnostics are closed, typed, and never promote evidence", () => {
  const record = complete({
    status: "gated",
    evidenceScope: "authoritative",
    diagnostics: {
      adapter: "memory-candle-adapter",
      execution: "gated_before_execution",
      blockers: ["plan/provider calibration fingerprint mismatch"],
      measurements: [{ name: "contractFingerprintMismatch", unit: "count", value: 1 }],
    },
  });
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  assert.equal(validateRecord(record), record);
  assert.equal(evidenceSemantics(record, {
    sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
    inference: record.repositories.inference.revision,
  }), "gated");

  for (const mutate of [
    (value) => (value.diagnostics.execution = "pretend"),
    (value) => (value.diagnostics.measurements[0].value = -1),
    (value) => (value.diagnostics.measurements[0].unexpected = true),
  ]) {
    const invalid = structuredClone(record);
    mutate(invalid);
    assert.throws(
      () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [invalid] }),
      /schema validation failed/,
    );
  }
});

test("parameterless strategies use a truthful degenerate sweep instead of a fabricated field", () => {
  const record = complete({
    status: "gated",
    sweep: {
      axes: [],
      cases: [{ parameters: {}, result: "passed" }],
      rangeVerified: false,
    },
  });
  record.strategy.parameters = {};
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  assert.equal(validateRecord(record), record);

  const promoted = structuredClone(record);
  promoted.status = "complete";
  promoted.sweep.rangeVerified = true;
  promoted.logicalCaseId = logicalCaseId(promoted);
  promoted.id = recordId(promoted);
  assert.equal(validateRecord(promoted), promoted);

  const invalid = structuredClone(promoted);
  invalid.strategy.parameters = { decodeTileEdge: 512 };
  invalid.logicalCaseId = logicalCaseId(invalid);
  invalid.id = recordId(invalid);
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [invalid] }),
    /parameterized complete strategy must sweep at least one axis/,
  );
});

test("executable runner handles fragmented responses across provider processes", async () => {
  const config = {
    providers: [{
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-seed42",
      cases: [
        { parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" },
        { parameters: { decodeTileEdge: 384, decodeOverlap: 128 }, expectedResult: "passed" },
      ],
    }],
  };
  const result = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fragmented-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
    inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
  });
  const clean = result.records.every((record) =>
    !record.repositories.sceneWorks.dirty && !record.repositories.inference.dirty
  );
  assert.equal(result.records.length, clean ? 1 : 2);
  assert.equal(expandPlan(config, result.records).length, clean ? 0 : 2);
  assert.equal(result.records[0].hardware.deviceId, "fixture:0");
  assert.match(result.records[0].repositories.sceneWorks.revision, /^[0-9a-f]{40}$/);
});

function physicalMlxConfig() {
  const target = {
    ...complete().target,
    provider: "fixture_mlx",
    modelId: "fixture_mlx",
  };
  return {
    providers: [{
      evidenceScope: "fixture",
      backend: "mlx",
      loadShape: "eager_materialization",
      target,
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "physical-mlx-fixture",
      cases: [{
        parameters: { decodeTileEdge: 512, decodeOverlap: 128 },
        expectedResult: "passed",
      }],
    }],
  };
}

test("physical MLX capture binds raw provider stdout, exact inventory, and persisted outputs", async () => {
  const config = physicalMlxConfig();
  const cleanRepo = await cleanFixtureRepo();
  const rawLogDir = await mkdtemp(path.join(tmpdir(), "physical-mlx-receipts-"));
  const sourcePathPrefix = "docs/calibration/sc-test";
  const result = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
      rawLogDir,
      sourcePathPrefix,
    ],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    rawLogDir,
    sourcePathPrefix,
  });
  assert.equal(result.records.length, 1);
  assert.equal(result.sourceSessions.length, 1);
  const record = result.records[0];
  const session = result.sourceSessions[0];
  assert.equal(session.kind, "physical_mlx");
  assert.deepEqual(session.inputs, [{
    role: "base",
    path: "/fixture/q4",
    bytes: 1234,
    sha256: "d".repeat(64),
    repository: "SceneWorks/fixture",
    resolvedRevision: "c".repeat(40),
    variant: "q4",
  }]);
  for (const claim of ["memory", "quality", "negativeMutation", "lifecycle", "loadability", "overlay"]) {
    assert.deepEqual(record.derivation[claim].sourceSessionIds, [session.id]);
  }
  const raw = await readFile(path.join(rawLogDir, session.sourcePath));
  assert.equal(createHash("sha256").update(raw).digest("hex"), session.stdoutSha256);
  assert.equal(session.outputs.length, 3);
  assert.deepEqual(
    session.outputs.map((output) => output.role).sort(),
    ["reference_rgb", "request", "selected_rgb"],
  );
  for (const output of session.outputs) {
    const local = path.join(rawLogDir, output.path);
    const bytes = await readFile(local);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), output.sha256);
    assert.equal(bytes.length, output.bytes);
  }
  assert.equal(validateBundle(result), result);
  assert.equal(await validateSourceSessionFiles(result, rawLogDir), result);

  const authoritative = structuredClone(result);
  const authoritativeRecord = authoritative.records[0];
  authoritativeRecord.evidenceScope = "authoritative";
  authoritativeRecord.sourceProvenance = "physical_mlx_v1";
  authoritativeRecord.target.modelId = "qwen_image";
  authoritativeRecord.target.provider = "qwen_image";
  authoritativeRecord.loadability.resolvedPathFingerprint = `SceneWorks/fixture@${"c".repeat(40)}:q4`;
  const priorLogicalCaseId = authoritativeRecord.logicalCaseId;
  authoritativeRecord.logicalCaseId = logicalCaseId(authoritativeRecord);
  authoritativeRecord.id = recordId(authoritativeRecord);
  for (const output of authoritative.sourceSessions[0].outputs.filter(({ role }) => role !== "request")) {
    output.path = output.path.replace(priorLogicalCaseId, authoritativeRecord.logicalCaseId);
  }
  assert.equal(validateBundle(authoritative), authoritative);

  const missingDerivation = structuredClone(authoritative);
  delete missingDerivation.records[0].derivation;
  assert.throws(() => validateBundle(missingDerivation), /missing source-session derivation/);
  const wrongInventory = structuredClone(authoritative);
  wrongInventory.records[0].artifact.inventorySha256 = "0".repeat(64);
  wrongInventory.records[0].id = recordId(wrongInventory.records[0]);
  assert.throws(() => validateBundle(wrongInventory), /does not match the record artifact inventory/);

  const missingOutput = structuredClone(result);
  missingOutput.sourceSessions[0].outputs[1].sha256 = "0".repeat(64);
  missingOutput.sourceSessions[0].outputs[1].path =
    `${sourcePathPrefix}/${record.logicalCaseId}-selected_rgb-1024x1024-${"0".repeat(64)}.rgb`;
  await assert.rejects(
    validateSourceSessionFiles(missingOutput, rawLogDir),
    /missing immutable source receipt/,
  );

  const noOutputs = structuredClone(result);
  noOutputs.sourceSessions[0].outputs = [];
  await assert.rejects(
    validateSourceSessionFiles(noOutputs, rawLogDir),
    /sourceSessions.*outputs|typed output receipts/,
  );

  const duplicateRole = structuredClone(result);
  duplicateRole.sourceSessions[0].outputs[2].role = "selected_rgb";
  assert.throws(
    () => validateBundle(duplicateRole),
    /sourceSessions.*outputs|repeats output role|selected_rgb.*reference_rgb/,
  );

  const duplicatePath = structuredClone(result);
  duplicatePath.sourceSessions[0].outputs[2].path = duplicatePath.sourceSessions[0].outputs[1].path;
  assert.throws(() => validateBundle(duplicatePath), /schema validation|wrong role|repeats output path/);

  const swappedRoles = structuredClone(result);
  swappedRoles.sourceSessions[0].outputs[0].role = "selected_rgb";
  swappedRoles.sourceSessions[0].outputs[1].role = "request";
  assert.throws(
    () => validateBundle(swappedRoles),
    /schema validation|request receipt|wrong role|repeats output role/,
  );

  const requestOutput = session.outputs.find(({ role }) => role === "request");
  const requestPath = path.join(rawLogDir, requestOutput.path);
  const originalRequest = await readFile(requestPath);
  const forgedRequest = canonicalJson({ action: "run", planned: {} });
  await writeFile(requestPath, forgedRequest);
  const forgedRequestBundle = structuredClone(result);
  const forgedRequestOutput = forgedRequestBundle.sourceSessions[0].outputs
    .find(({ role }) => role === "request");
  forgedRequestOutput.sha256 = createHash("sha256").update(forgedRequest).digest("hex");
  forgedRequestOutput.bytes = Buffer.byteLength(forgedRequest);
  await assert.rejects(
    validateSourceSessionFiles(forgedRequestBundle, rawLogDir),
    /request receipt logicalCaseId does not match/,
  );
  await writeFile(requestPath, originalRequest);

  const tamperedCaptureDir = await mkdtemp(path.join(tmpdir(), "physical-mlx-tamper-race-"));
  await assert.rejects(
    runProviderPlan({
      closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [
        process.execPath,
        fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
        tamperedCaptureDir,
        sourcePathPrefix,
        tamperedCaptureDir,
        "tamper-after-attest",
      ],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      rawLogDir: tamperedCaptureDir,
      sourcePathPrefix,
    }),
    /provider output differs from its provider attestation/,
  );

  await writeFile(path.join(rawLogDir, session.sourcePath), "tampered provider output");
  await assert.rejects(
    runProviderPlan({
      closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [
        process.execPath,
        fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
        rawLogDir,
        sourcePathPrefix,
      ],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      rawLogDir,
      sourcePathPrefix,
    }),
    /immutable source receipt already exists with different bytes/,
  );
});

test("physical MLX receipts reject traversal prefixes and outputs outside the raw directory", async () => {
  const config = physicalMlxConfig();
  const cleanRepo = await cleanFixtureRepo();
  const rawLogDir = await mkdtemp(path.join(tmpdir(), "physical-mlx-receipts-"));
  const outsideDir = await mkdtemp(path.join(tmpdir(), "physical-mlx-outside-"));
  const fixtureCommand = [
    process.execPath,
    fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    rawLogDir,
  ];
  await assert.rejects(
    runProviderPlan({
      closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [...fixtureCommand, "docs/calibration/../escape"],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      rawLogDir,
      sourcePathPrefix: "docs/calibration/../escape",
    }),
    /source path prefix must be a normalized path under docs\/calibration/,
  );
  await assert.rejects(
    runProviderPlan({
      closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [...fixtureCommand, "docs/calibration/sc-test", outsideDir],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      rawLogDir,
      sourcePathPrefix: "docs/calibration/sc-test",
    }),
    /physical MLX local output must stay under the raw log directory/,
  );
  await assert.rejects(
    runProviderPlan({
      closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [
        process.execPath,
        fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
      ],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      rawLogDir,
      sourcePathPrefix: "docs/calibration/sc-test",
    }),
    /configured raw-log provenance requires provider sourceCapture/,
  );
});

test("runner batches one target's five rungs into one attested model load", async () => {
  const rungs = [
    ["resident", ["resident"]],
    ["staged_residency", ["resident", "staged_residency"]],
    ["bounded_decode", ["resident", "bounded_decode"]],
    ["bounded_attention", ["resident", "bounded_decode", "bounded_attention"]],
    [
      "bounded_transformer_residency",
      ["resident", "bounded_decode", "bounded_attention", "bounded_transformer_residency"],
    ],
  ];
  const config = {
    providers: rungs.map(([rung, engagedRungs]) => ({
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung,
      engagedRungs,
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-five-rungs",
      modelLoadPolicy: "batch_rungs",
      modelLoadGroup: "fixture-target",
      cases: [
        { parameters: { decodeTileEdge: 384, decodeOverlap: 128 }, expectedResult: "passed" },
        { parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" },
      ],
    })),
  };
  const invocations = [];
  const cleanRepo = await cleanFixtureRepo();
  const result = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    onProviderInvocation: (invocation) => invocations.push(invocation),
  });
  assert.equal(result.records.length, 5);
  assert.deepEqual(invocations.map(({ action, cases }) => [action, cases.length]), [["run_batch", 5]]);
  assert.equal(expandPlan(config, result.records).length, 0);
});

function qwenGatedBatchConfig() {
  const target = {
    ...complete().target,
    modelId: "qwen_image",
    provider: "qwen_image",
  };
  const shared = {
    evidenceScope: "candidate",
    backend: "candle",
    loadShape: "deferred_materialization",
    target,
    calibrationFingerprint:
      "qwen-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v1",
    fixture: "qwen-image-candle-q4-seed15817-step2",
  };
  const decode = (decodeTileEdge) => ({ decodeTileEdge, decodeOverlap: 64 });
  const attention = {
    ...decode(512),
    attentionChunkSize: 67_108_864,
  };
  return {
    providers: [
      {
        ...shared,
        rung: "resident",
        engagedRungs: ["resident"],
        cases: [{ parameters: {}, expectedResult: "passed" }],
      },
      {
        ...shared,
        rung: "staged_residency",
        engagedRungs: ["resident", "staged_residency"],
        cases: [{ parameters: {}, expectedResult: "passed" }],
      },
      {
        ...shared,
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        cases: [768, 640, 512, 448, 384, 320, 256].map((edge) => ({
          parameters: decode(edge),
          expectedResult: "passed",
        })),
      },
      {
        ...shared,
        rung: "bounded_attention",
        engagedRungs: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
        cases: [{ parameters: attention, expectedResult: "passed" }],
      },
      {
        ...shared,
        rung: "bounded_transformer_residency",
        engagedRungs: [
          "resident", "staged_residency", "bounded_decode", "bounded_attention",
          "bounded_transformer_residency",
        ],
        cases: [1, 2, 4, 8, 15, 30].map((transformerWindowSize) => ({
          parameters: { ...attention, transformerWindowSize },
          expectedResult: "passed",
        })),
      },
    ],
  };
}

test("gated Qwen batch persists the canonical five rungs then serializes all remaining sweep points", async () => {
  const config = qwenGatedBatchConfig();
  assert.equal(expandPlan(config).length, 16);

  const invocations = [];
  const checkpointSizes = [];
  const cleanRepo = await cleanFixtureRepo();
  const outputDir = await mkdtemp(path.join(tmpdir(), "memory-harness-output-"));
  const output = path.join(outputDir, "evidence.json");
  const result = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL(
        "./fixtures/memory-provider-gated-canonical-batch-fixture.mjs",
        import.meta.url,
      )),
    ],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    forceBatchRungs: true,
    onProviderInvocation: (invocation) => invocations.push(invocation),
    onProviderCheckpoint: async (checkpoint) => {
      checkpointSizes.push(checkpoint.records.length);
      await atomicWrite(output, checkpoint);
      validateBundle(JSON.parse(await readFile(output, "utf8")));
    },
  });

  assert.deepEqual(
    invocations.map(({ action, cases }) => [action, cases.length]),
    [["run_batch", 5], ...Array.from({ length: 11 }, () => ["run", 1])],
  );
  assert.deepEqual(checkpointSizes, Array.from({ length: 12 }, (_, index) => index + 5));
  assert.equal(result.records.length, 16);
  assert.ok(result.records.every((record) =>
    record.status === "gated" && record.evidenceScope === "candidate"
  ));
  assert.deepEqual(
    result.records.map((record) => record.logicalCaseId).sort(),
    expandPlan(config).map((planned) => planned.logicalCaseId).sort(),
  );
  assert.equal(
    expandPlan(config, result.records).length,
    16,
    "gated evidence must not falsely retire or promote planned calibration points",
  );
  assert.equal(validateBundle(JSON.parse(await readFile(output, "utf8"))).records.length, 16);
  assert.deepEqual(await readdir(outputDir), ["evidence.json"]);
});

test("gated Qwen resume continues after failure without repeating provenance-matched attempts", async () => {
  const config = qwenGatedBatchConfig();
  const cleanRepo = await cleanFixtureRepo();
  const outputDir = await mkdtemp(path.join(tmpdir(), "memory-harness-resume-"));
  const output = path.join(outputDir, "evidence.json");
  const state = path.join(outputDir, "provider-state.json");
  const fixture = fileURLToPath(new URL(
    "./fixtures/memory-provider-gated-canonical-batch-fixture.mjs",
    import.meta.url,
  ));
  const firstInvocations = [];
  await assert.rejects(
    runProviderPlan({
    closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [process.execPath, fixture, state, "4"],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      forceBatchRungs: true,
      onProviderInvocation: (invocation) => firstInvocations.push(invocation),
      onProviderCheckpoint: (checkpoint) => atomicWrite(output, checkpoint),
    }),
    /fixture fails after 4 successful invocations/,
  );
  assert.deepEqual(
    firstInvocations.map(({ action, cases }) => [action, cases.length]),
    [["run_batch", 5], ...Array.from({ length: 4 }, () => ["run", 1])],
  );

  const checkpoint = validateBundle(JSON.parse(await readFile(output, "utf8")));
  assert.equal(checkpoint.records.length, 8);
  const checkpointLogicalIds = new Set(checkpoint.records.map((record) => record.logicalCaseId));
  const resumedInvocations = [];
  const resumed = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [process.execPath, fixture, state],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    resume: checkpoint,
    forceBatchRungs: true,
    onProviderInvocation: (invocation) => resumedInvocations.push(invocation),
    onProviderCheckpoint: (nextCheckpoint) => atomicWrite(output, nextCheckpoint),
  });

  assert.deepEqual(
    resumedInvocations.map(({ action, cases }) => [action, cases.length]),
    Array.from({ length: 8 }, () => ["run", 1]),
  );
  const resumedLogicalIds = resumedInvocations.flatMap(({ cases }) =>
    cases.map((planned) => planned.logicalCaseId)
  );
  assert.ok(resumedLogicalIds.every((logicalId) => !checkpointLogicalIds.has(logicalId)));
  assert.equal(new Set([...checkpointLogicalIds, ...resumedLogicalIds]).size, 16);
  assert.equal(resumed.records.length, 16);
  assert.equal(new Set(resumed.records.map((record) => record.id)).size, 16);
  assert.ok(resumed.records.every((record) => record.id === recordId(record)));
  assert.equal(new Set(resumed.records.map((record) => record.capturedAt)).size, 12);
  for (const prior of checkpoint.records) {
    assert.deepEqual(resumed.records.find((record) => record.id === prior.id), prior);
  }
  assert.equal(
    expandPlan(config, resumed.records).length,
    16,
    "operational attempt resume must not turn gated receipts into completion evidence",
  );
  assert.equal(validateBundle(JSON.parse(await readFile(output, "utf8"))).records.length, 16);
  assert.ok((await readdir(outputDir)).every((name) => !name.includes(".tmp-")));
});

test("operational resume does not suppress an attempt from mismatched hardware provenance", async () => {
  const config = { providers: [qwenGatedBatchConfig().providers[0]] };
  const cleanRepo = await cleanFixtureRepo();
  const fixtureCommand = [
    process.execPath,
    fileURLToPath(new URL(
      "./fixtures/memory-provider-gated-canonical-batch-fixture.mjs",
      import.meta.url,
    )),
  ];
  const first = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: fixtureCommand,
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
  });
  const stale = structuredClone(first);
  stale.records[0].hardware.driverVersion = "stale-driver";
  stale.records[0].id = recordId(stale.records[0]);
  validateBundle(stale);

  const invocations = [];
  const resumed = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: fixtureCommand,
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    resume: stale,
    onProviderInvocation: (invocation) => invocations.push(invocation),
  });
  assert.deepEqual(invocations.map(({ action, cases }) => [action, cases.length]), [["run", 1]]);
  assert.equal(resumed.records.length, 2);
  assert.equal(new Set(resumed.records.map((record) => record.logicalCaseId)).size, 1);
  assert.equal(new Set(resumed.records.map((record) => record.id)).size, 2);
});

test("runner flags provide explicit fresh and experimental batch controls", async () => {
  const config = {
    providers: ["resident", "bounded_decode"].map((rung) => ({
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung,
      engagedRungs: rung === "resident" ? ["resident"] : ["resident", "bounded_decode"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-forced-rungs",
      cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" }],
    })),
  };
  const fixtureCommand = [
    process.execPath,
    fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
  ];
  const freshInvocations = [];
  await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: fixtureCommand,
    sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
    inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    forceFreshPerCase: true,
    onProviderInvocation: (invocation) => freshInvocations.push(invocation),
  });
  assert.deepEqual(freshInvocations.map(({ action }) => action), ["run", "run"]);

  const batchInvocations = [];
  await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: fixtureCommand,
    sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
    inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    forceBatchRungs: true,
    onProviderInvocation: (invocation) => batchInvocations.push(invocation),
  });
  assert.deepEqual(batchInvocations.map(({ action, cases }) => [action, cases.length]), [["run_batch", 2]]);
});

test("provider reuse assessment records whether a batch can be measured", async () => {
  const config = {
    providers: [{
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung: "resident",
      engagedRungs: ["resident"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-assessment",
      cases: [{ parameters: {}, expectedResult: "passed" }],
    }],
  };
  const assessment = await assessProviderReuse({
    config,
    backend: "candle",
    fixture: "fixture-assessment",
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    ],
  });
  assert.equal(assessment.verdict, "eligible_for_measurement");
  assert.deepEqual(assessment.tolerance, RUNG_REUSE_TOLERANCE);
});

test("a completed sweep retires its other parameter points before another spawn", async () => {
  const config = {
    providers: [{
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-seed42",
      cases: [
        { parameters: { decodeTileEdge: 384, decodeOverlap: 128 }, expectedResult: "passed" },
        { parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" },
      ],
    }],
  };
  const invocations = [];
  const cleanRepo = await cleanFixtureRepo();
  const result = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    onProviderInvocation: (invocation) => invocations.push(invocation),
  });
  assert.equal(result.records.length, 1);
  assert.deepEqual(invocations.map(({ action, cases }) => [action, cases.length]), [["run", 1]]);
  assert.equal(expandPlan(config, result.records).length, 0);
});

test("provider execution can select every rung sharing one reproducible fixture", async () => {
  const provider = {
    evidenceScope: "fixture",
    backend: "candle",
    loadShape: "eager_materialization",
    target: complete().target,
    calibrationFingerprint: "fixture-formula-v2",
    fixture: "fresh-five-rung-fixture",
    cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" }],
  };
  const config = {
    providers: [
      {
        ...provider,
        name: "selected",
        rung: "bounded_decode",
        engagedRungs: ["resident", "bounded_decode"],
      },
      {
        ...provider,
        name: "unrelated",
        fixture: "different-fixture",
        rung: "bounded_decode",
        engagedRungs: ["resident", "bounded_decode"],
      },
    ],
  };
  const result = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    fixture: "fresh-five-rung-fixture",
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fragmented-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
    inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
  });
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].fixture, "fresh-five-rung-fixture");
});

test("provider early exit is rejected without an unhandled stdin EPIPE", async () => {
  const config = {
    providers: [{
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-seed42",
      cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" }],
    }],
  };
  await assert.rejects(
    runProviderPlan({
    closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [
        process.execPath,
        fileURLToPath(new URL("./fixtures/memory-provider-early-exit-fixture.mjs", import.meta.url)),
      ],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    }),
    /closed its provider-protocol stdin|Unexpected end of JSON input/,
  );
});

test("expected-failure plan case produces a resumable negative record, never a complete positive", async () => {
  const config = {
    providers: [{
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-seed42",
      cases: [{
        parameters: { decodeTileEdge: 256, decodeOverlap: 32 },
        expectedResult: "failed",
        negative: true,
      }],
    }],
  };
  const result = await runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fragmented-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
    inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
  });
  assert.equal(result.records[0].status, "negative_complete");
  assert.equal(expandPlan(config, result.records).length, 0);
});

test("provider execution rejects an adapter-attested composition that differs from the plan", async () => {
  const config = {
    providers: [{
      evidenceScope: "fixture",
      backend: "candle",
      loadShape: "eager_materialization",
      target: complete().target,
      rung: "bounded_decode",
      engagedRungs: ["resident", "bounded_decode"],
      calibrationFingerprint: "fixture-formula-v2",
      fixture: "fixture-seed42",
      cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" }],
    }],
  };
  await assert.rejects(
    runProviderPlan({
    closureDigestFor: stubClosureDigest,
      config,
      providerCommand: [
        process.execPath,
        fileURLToPath(new URL(
          "./fixtures/memory-provider-composition-mismatch-fixture.mjs",
          import.meta.url,
        )),
      ],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    }),
    /adapter measured strategy does not match planned strategy/,
  );
});

test("the measured load shape is a receipt field, never copied from the plan", async () => {
  // Before this, the runner wrote `loadShape: planned.loadShape` onto every record: the adapter was
  // never asked, so the field recorded the plan's CLAIM rather than what the run did, and no
  // divergence was detectable. That is the same backfill sc-16482 forbids for historical receipts,
  // applied silently to new ones. These two properties pin the fix.
  const provider = {
    evidenceScope: "fixture",
    backend: "candle",
    target: complete().target,
    rung: "bounded_decode",
    engagedRungs: ["resident", "bounded_decode"],
    calibrationFingerprint: "fixture-formula-v2",
    loadShape: "eager_materialization",
    fixture: "fixture-seed42",
    cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" }],
  };
  const run = (fixture) =>
    runProviderPlan({
    closureDigestFor: stubClosureDigest,
      config: { providers: [provider] },
      providerCommand: [process.execPath, fileURLToPath(new URL(fixture, import.meta.url))],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    });

  // 1. An adapter that measured a different shape than the plan declared is rejected outright.
  await assert.rejects(
    run("./fixtures/memory-provider-load-shape-mismatch-fixture.mjs"),
    /adapter measured loadShape deferred_materialization but the plan declared eager_materialization/,
  );

  // 2. An adapter that attests nothing is rejected too — otherwise "no opinion" would silently
  //    become "whatever the plan said", which is the behaviour being removed.
  await assert.rejects(
    run("./fixtures/memory-provider-composition-mismatch-fixture.mjs"),
    /adapter measured strategy does not match planned strategy|must attest a loadShape/,
  );

  // 3. The agreeing case records the ADAPTER's value. Identical to the plan's here by construction,
  //    so this alone proves little — properties 1 and 2 are what make it meaningful.
  const result = await run("./fixtures/memory-provider-fixture.mjs");
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].loadShape, "eager_materialization");
});

// SC-16211. The `harnessVersion` const is asserted twice in the schema and embedded in every emitted
// record. Adding the required engaged-composition identity bumps it and MUST invalidate every v3
// record rather than treating a missing load-shape identity as shape-agnostic.
const PRIOR_HARNESS_VERSION = "sceneworks-memory-v4";

test("pre-composition evidence is rejected as stale by the schema and harness gates", () => {
  assert.equal(HARNESS_VERSION, "sceneworks-memory-v5");
  assert.notEqual(HARNESS_VERSION, PRIOR_HARNESS_VERSION);

  // A genuine prior-vintage record: every field populated, and its deterministic id recomputed over
  // the old version exactly as the v2 harness would have emitted it, so identity is NOT what fails.
  const stale = complete();
  stale.harnessVersion = PRIOR_HARNESS_VERSION;
  stale.id = recordId(stale);
  assert.equal(stale.id, recordId(stale));

  // 1. the runtime record gate rejects it, and names the version rather than an incidental field.
  assert.throws(() => validateRecord(stale), /invalid harnessVersion/);

  // 2. the schema const rejects both the envelope and the record it carries.
  assert.throws(
    () => validateBundle({ schemaVersion: 2, harnessVersion: PRIOR_HARNESS_VERSION, records: [stale] }),
    (error) =>
      /schema validation failed/.test(error.message) &&
      error.message.includes("$.harnessVersion: value does not equal const") &&
      error.message.includes("$.records[0].harnessVersion: value does not equal const"),
  );

  // 3. a stale record cannot be smuggled in under a current envelope either.
  assert.throws(
    () => validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [stale] }),
    /schema validation failed/,
  );

  // 4. control: the identical record at the current version passes both gates, so the rejections
  //    above are caused by the version bump and by nothing else.
  const current = complete();
  assert.equal(validateRecord(current), current);
  validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [current] });
});

test("the promoted evidence bundle carries the current harnessVersion", async () => {
  const bundle = JSON.parse(
    await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url)),
  );
  assert.equal(bundle.harnessVersion, HARNESS_VERSION);
  validateBundle(bundle);
});

// sc-18100: rehomed from `sc-15823-flux1-evidence.test.mjs`, which bound only ITS ten FLUX.1 logs to
// their recorded hashes. The bundle's whole point is that a record is traceable to immutable captured
// output, so the gate belongs to the bundle, not to one campaign's script. Generalised: a missing or
// edited log now fails for every campaign, and no per-campaign census has to be hand-maintained.
test("every committed source session binds immutable files whose bytes match their recorded hashes", async () => {
  const root = new URL("../", import.meta.url);
  const bundle = JSON.parse(
    await readFile(new URL("docs/generated/memory-calibration-evidence.json", root)),
  );
  const sessions = bundle.sourceSessions ?? [];
  assert.ok(sessions.length > 0, "the shipped bundle must carry source sessions to bind");
  assert.equal(await validateSourceSessionFiles(bundle), bundle);
});

// -----------------------------------------------------------------------------------------------
// sc-17935 item 3: a capture run must be able to derive its own closure digest.
//
// The digest was keyed on `providerName` — the `--provider` PLAN-ENTRY name — while the declarations
// table is keyed by LANE (`<backend>:<provider>`). Every checked-in capture workflow selects with
// `--fixture`, so `providerName` was `undefined` and the run died with `provider "undefined" has no
// entry` before touching the GPU; passing `--provider` failed on the other spelling. macOS and
// Windows capture jobs therefore could not produce replacement evidence — the one remedy a narrowed
// currency term leaves you.
// -----------------------------------------------------------------------------------------------

/** An authoritative one-rung plan provider on a given lane, selected by its fixture. */
function laneProvider({ backend, provider, fixture, modelId = provider }) {
  return {
    evidenceScope: "authoritative",
    backend,
    loadShape: "eager_materialization",
    target: { ...complete().target, modelId, provider },
    rung: "resident",
    engagedRungs: ["resident"],
    calibrationFingerprint: `${provider}-formula-v1`,
    fixture,
    cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 128 }, expectedResult: "passed" }],
  };
}

async function runFixtureSelection({ config, fixture, providerName, closureDigestFor }) {
  const cleanRepo = await cleanFixtureRepo();
  return runProviderPlan({
    closureDigestFor,
    config,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    fixture,
    providerName,
  });
}

test("a --fixture-only capture run keys its closure digest by lane, not by plan-entry name", async () => {
  // This is the checked-in workflow invocation shape verbatim: `--backend … --fixture … ` and no
  // `--provider`, so `providerName` is undefined.
  const asked = [];
  const result = await runFixtureSelection({
    config: { providers: [laneProvider({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-krea" })] },
    fixture: "cap-krea",
    closureDigestFor: async (lane) => {
      asked.push(lane);
      return createHash("sha256").update(`closure:${lane}`).digest("hex");
    },
  });
  assert.deepEqual(asked, ["candle:krea_2_turbo"], "the table key is <backend>:<provider>");
  assert.equal(result.records.length, 1);
  assert.equal(
    result.records[0].repositories.inference.closureDigest,
    createHash("sha256").update("closure:candle:krea_2_turbo").digest("hex"),
  );
});

test("the stamped digest is the one evidenceSemantics compares, so the capture reads current", async () => {
  // The end-to-end property: a fresh capture of a lane whose declared digest is live must come back
  // `current`. If the runner keyed on anything but the lane, this is where it would read historical.
  const result = await runFixtureSelection({
    config: { providers: [laneProvider({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-krea" })] },
    fixture: "cap-krea",
    closureDigestFor: stubClosureDigest,
  });
  const record = result.records[0];
  const live = { "candle:krea_2_turbo": await stubClosureDigest("candle:krea_2_turbo") };
  assert.equal(
    evidenceSemantics(record, {
      sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
      inference: record.repositories.inference.revision,
      inferenceClosureDigests: live,
    }),
    "current",
  );
  // Mutation check: the same record against ANOTHER lane's digest must not read current, so the
  // assertion above is testing the key and not merely that both sides ran the same stub.
  assert.equal(
    evidenceSemantics(record, {
      sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
      inference: record.repositories.inference.revision,
      inferenceClosureDigests: { "candle:krea_2_turbo": await stubClosureDigest("mlx:qwen_image") },
    }),
    "historical",
  );
});

test("a multi-lane capture stamps each record with ITS lane's digest", async () => {
  // `--backend candle` with no `--fixture` selects every candle lane. One run-level digest would
  // stamp both records with whichever lane happened to be first.
  const config = {
    providers: [
      laneProvider({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-multi" }),
      laneProvider({ backend: "candle", provider: "flux2_dev", fixture: "cap-multi" }),
    ],
  };
  const result = await runFixtureSelection({ config, fixture: "cap-multi", closureDigestFor: stubClosureDigest });
  assert.equal(result.records.length, 2);
  const byLane = new Map(result.records.map((record) =>
    [`${record.backend}:${record.target.provider}`, record.repositories.inference.closureDigest]));
  assert.equal(byLane.get("candle:krea_2_turbo"), await stubClosureDigest("candle:krea_2_turbo"));
  assert.equal(byLane.get("candle:flux2_dev"), await stubClosureDigest("candle:flux2_dev"));
  assert.notEqual(byLane.get("candle:krea_2_turbo"), byLane.get("candle:flux2_dev"));
});

test("a selection with no authoritative entry derives no digest at all", async () => {
  // A fixture/candidate capture can never be `current`, so it must not need an inference crate
  // layout or a declarations file. The injected deriver throws to prove it is never reached.
  const config = {
    providers: [{
      ...laneProvider({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-fixture" }),
      evidenceScope: "fixture",
    }],
  };
  const result = await runFixtureSelection({
    config,
    fixture: "cap-fixture",
    closureDigestFor: async () => assert.fail("a fixture-scope capture must not derive a closure digest"),
  });
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].repositories.inference.closureDigest, undefined);
});

test("an undeclared lane fails BEFORE the first capture, not after it", async () => {
  // Eager derivation is the point: the hardware probe is cheap, but a 26 GB `run`/`run_batch` must
  // not burn before the runner discovers it cannot stamp a currency term. This drives the REAL
  // deriver (no `closureDigestFor`) against the real declarations file, and `onProviderInvocation`
  // fires only for capture actions — so an empty list proves nothing was measured.
  const cleanRepo = await cleanFixtureRepo();
  const invocations = [];
  await assert.rejects(
    runProviderPlan({
      config: {
        providers: [laneProvider({
          backend: "candle", provider: "never_declared_provider", fixture: "cap-undeclared",
        })],
      },
      providerCommand: [
        process.execPath,
        fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
      ],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: cleanRepo,
      fixture: "cap-undeclared",
      onProviderInvocation: (invocation) => invocations.push(invocation),
    }),
    /lane "candle:never_declared_provider" has no entry in config\/inference-provider-closures\.json/,
  );
  assert.deepEqual(invocations, [], "no capture may run once the lane is known to be underivable");
});

test("resume is not defeated by the per-lane digest a prior record carries", async () => {
  // A GATED authoritative record is the shape that actually reaches the operational-attempt check:
  // `completedLogicalIds` returns nothing for it, so `expandPlan` does not suppress it and only
  // `operationallyAttemptedLogicalIds` can stop it being measured again. It still carries a lane
  // digest — stamping keys on evidenceScope, not status — while the run-level `repositories` now
  // carries none. Comparing those two raw makes every prior record look foreign, and the cost of
  // that is repeating a multi-hour, tens-of-GB GPU capture.
  const cleanRepo = await cleanFixtureRepo();
  const config = {
    providers: [laneProvider({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-resume" })],
  };
  const gated = fileURLToPath(
    new URL("./fixtures/memory-provider-gated-canonical-batch-fixture.mjs", import.meta.url),
  );
  const run = (resume, onProviderInvocation) => runProviderPlan({
    closureDigestFor: stubClosureDigest,
    config,
    providerCommand: [process.execPath, gated],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    fixture: "cap-resume",
    resume,
    onProviderInvocation,
  });

  const first = await run(undefined);
  assert.equal(first.records.length, 1);
  assert.equal(first.records[0].status, "gated");
  assert.equal(
    first.records[0].repositories.inference.closureDigest,
    await stubClosureDigest("candle:krea_2_turbo"),
    "an authoritative capture stamps its lane digest even when the result is gated",
  );

  const invocations = [];
  const resumed = await run(first, (invocation) => invocations.push(invocation));
  assert.deepEqual(invocations, [], "a resumed capture must not repeat work it has already paid for");
  assert.equal(resumed.records.length, 1);
});

test("every checked-in capture invocation selects a plan fixture on a DECLARED lane", async () => {
  // The story's regression gate, driven off the real workflow files. Three ways this drifts: a
  // workflow names a fixture the plan no longer has, its `--backend`/`--fixture` pair selects
  // nothing, or the plan gains an authoritative lane nobody declared a crate for. Each one kills a
  // capture job on a runner instead of here.
  const root = new URL("../", import.meta.url);
  const plan = JSON.parse(await readFile(new URL("config/memory-calibration-plan.json", root)));
  const closures = JSON.parse(await readFile(new URL("config/inference-provider-closures.json", root)));
  const workflows = await readdir(new URL(".github/workflows/", root));

  const selected = [];
  for (const file of workflows.filter((name) => name.endsWith(".yml"))) {
    const body = await readFile(new URL(`.github/workflows/${file}`, root), "utf8");
    // `harness.mjs run` invocations only; `assess-reuse` and `check` never derive a digest. Windows
    // packs the whole invocation onto one line, macOS uses backslash continuations.
    for (const invocation of body.match(/memory-calibration-harness\.mjs\s+run\b[\s\S]*?--output\s+\S+/g) ?? []) {
      const backend = invocation.match(/--backend\s+(\S+)/)?.[1];
      const fixture = invocation.match(/--fixture\s+"?([^"\s\\]+)"?/)?.[1];
      assert.ok(backend && fixture, `${file}: a capture invocation selects no --backend/--fixture`);
      if (!fixture.includes("$")) {
        selected.push([`${file} --fixture ${fixture}`, backend, (name) => name === fixture]);
        continue;
      }
      // The Qwen invocation templates its fixture from `inputs.qwen_tier` and a seed chosen in the
      // same shell block. Both come out of the workflow itself rather than being restated here, so a
      // drift on either side is caught: every DECLARED tier option must reach a plan fixture, and
      // that fixture's seed must be one the workflow can actually set.
      const tiers = body.match(/qwen_tier:[\s\S]*?options:\s*((?:\s*-\s*\w+\n)+)/)?.[1];
      // The seed is assigned upstream of the invocation in the same shell block, so it is read off
      // the workflow rather than the sliced invocation.
      const seeds = [...body.matchAll(/QWEN_SEED=(\d+)/g)].map((match) => match[1]);
      assert.ok(tiers, `${file}: the templated fixture's tier input declares no options`);
      assert.ok(seeds.length, `${file}: the templated fixture's seed is set nowhere in the workflow`);
      for (const tier of tiers.match(/-\s*(\w+)/g).map((line) => line.replace(/-\s*/, ""))) {
        const shape = new RegExp(`^${fixture.replace("${QWEN_TIER}", tier).replace("${QWEN_SEED}", `(?:${seeds.join("|")})`)}$`);
        selected.push([`${file} --fixture ${fixture} (tier ${tier})`, backend, (name) => shape.test(name)]);
      }
    }
  }
  assert.ok(selected.length >= 6, `expected the checked-in capture selections, found ${selected.length}`);

  for (const [label, backend, matches] of selected) {
    // Exactly what `runProviderPlan` does: filter the plan, then key every authoritative survivor.
    const entries = plan.providers.filter((provider) => provider.backend === backend && matches(provider.fixture));
    assert.ok(entries.length, `${label}: selects no plan provider on backend ${backend}`);
    for (const entry of entries.filter((provider) => provider.evidenceScope === "authoritative")) {
      const lane = `${entry.backend}:${entry.target.provider}`;
      assert.ok(closures.providers[lane], `${label}: lane ${lane} has no closure declaration`);
    }
  }

  // And the converse, so a plan lane that no workflow captures yet still cannot be added undeclared.
  const undeclared = [...new Set(plan.providers
    .filter((provider) => provider.evidenceScope === "authoritative")
    .map((provider) => `${provider.backend}:${provider.target.provider}`))]
    .filter((lane) => !closures.providers[lane]);
  assert.deepEqual(undeclared, [], "authoritative plan lanes with no closure declaration cannot be captured");
});
