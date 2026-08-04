import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import {
  HARNESS_VERSION, RUNG_REUSE_TOLERANCE, assessProviderReuse, atomicWrite, canonicalJson,
  compareRungReuse, evidenceSemantics, expandPlan, logicalCaseId, mergeBundles, recordId,
  runProviderPlan, validateBundle, validateRecord,
} from "./memory-calibration-harness.mjs";
import { calibrationBinding } from "./generate-memory-matrix.mjs";

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
      inference: { revision: "b".repeat(40), dirty: false },
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

test("calibration ABI binding owns SceneWorks invalidation while inference SHA remains currentness input", () => {
  const record = complete({ evidenceScope: "authoritative" });
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  const revisions = {
    sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
    inference: record.repositories.inference.revision,
  };
  assert.equal(evidenceSemantics(record, revisions), "current");
  assert.equal(
    evidenceSemantics(record, { ...revisions, sceneWorks: "source-tree:different" }),
    "current",
    "matrixSourceRevision is exact provenance; calibrationBinding separately enforces the SceneWorks ABI fingerprint",
  );
  assert.equal(evidenceSemantics(record, { ...revisions, inference: "c".repeat(40) }), "historical");
  assert.equal(
    evidenceSemantics(record, {
      ...revisions,
      inference: "c".repeat(40),
      compatibleCapturedInferenceRevisions: [record.repositories.inference.revision],
    }),
    "current",
    "an explicit provider-closure audit may admit captured provenance at one later runtime pin",
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
      (item) => item.loadShape === (
        item.strategy.rung === "bounded_transformer_residency"
          ? "deferred_materialization"
          : "eager_materialization"
      )),
    "every Qwen plan case must retain its exact typed load shape",
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
      config,
      providerCommand: [process.execPath, "must-not-start.mjs"],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    }),
    /select exactly one backend/,
  );
});

test("one exact Qwen completion suppresses only its own ladder case", async () => {
  const config = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const record = qwenPositiveComplete();
  validateBundle({ schemaVersion: 4, harnessVersion: HARNESS_VERSION, records: [record] });
  const qwenRemaining = expandPlan(config, [record]).filter(
    (item) => item.target.provider === "qwen_image" && item.backend === "mlx",
  );
  assert.equal(qwenRemaining.length, 14);
  assert.equal(
    qwenRemaining.some(
      (item) =>
        item.strategy.rung === "bounded_decode" &&
        item.strategy.parameters.decodeTileEdge === 512 &&
        item.strategy.parameters.decodeOverlap === 64,
    ),
    false,
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
      inference: { revision: "b".repeat(40), dirty: false },
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
