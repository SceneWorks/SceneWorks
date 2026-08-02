import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import {
  HARNESS_VERSION, RUNG_REUSE_TOLERANCE, assessProviderReuse, canonicalJson,
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
    calibrationFingerprint: "qwen-image-mlx-shared-ladder-2026-08-01-v1-eager",
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

test("Qwen plan covers every ladder rung and all seven production decode edges", async () => {
  const config = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const cases = expandPlan(config);
  const qwen = cases.filter(
    (item) => item.target.provider === "qwen_image" && item.backend === "mlx",
  );
  assert.equal(qwen.length, 11);
  assert.ok(qwen.every((item) => item.expectedResult === "passed" && !item.negative));
  assert.deepEqual(
    [...new Set(qwen.map((item) => item.strategy.rung))].sort(),
    ["resident", "staged_residency", "bounded_decode", "bounded_attention", "bounded_transformer_residency"].sort(),
  );
  assert.deepEqual(
    qwen
      .filter((item) => item.strategy.rung === "bounded_decode")
      .map((item) => item.strategy.parameters.decodeTileEdge)
      .sort((left, right) => right - left),
    [768, 640, 512, 448, 384, 320, 256],
  );
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
  assert.equal(qwenRemaining.length, 10);
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
