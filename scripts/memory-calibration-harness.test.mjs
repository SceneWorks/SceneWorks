import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, realpath, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

import {
  ANCHOR_PLAN_SCHEMA, ANCHOR_STRATEGY, HARNESS_VERSION, LTX25_CAPTURE_REPOSITORY,
  LTX25_CAPTURE_REVISION, SCHEMA_VERSION, atomicWrite, canonicalJson, captureAnchor,
  projectPhaseMetricsToSchemaV5,
  evidenceSemantics, logicalCaseId, parseAnchorKey, planAnchor, planAnchors, recordId,
  ltx25ProviderEnvironment, parsePhysicalMlxAvContent, physicalMlxSessionId,
  prepareLtx25CaptureArtifacts, validateBundle, validateRecord, validatePlan,
  validatePhysicalMlxAvContentsAgainstRecord,
  validateSourceSessionFiles,
} from "./memory-calibration-harness.mjs";

/**
 * A one-anchor plan in the collapsed format (sc-22514). Every runner test drives exactly one
 * anchor, because the format cannot express anything else.
 */
function anchorPlanFixture(key, overrides = {}) {
  const { modelId } = parseAnchorKey(key);
  return {
    schemaVersion: 1,
    anchors: {
      [key]: {
        provider: modelId,
        mode: "text_to_image",
        overlay: "none",
        geometry: { width: 1024, height: 1024, batch: 1, frames: 1 },
        evidenceScope: "fixture",
        loadShape: "eager_materialization",
        calibrationFingerprint: "fixture-formula-v2",
        fixture: "fixture-seed42",
        ...overrides,
      },
    },
  };
}

// sc-17774: the runner stamps each record with the provider's inference compile-closure digest.
// These tests drive synthetic repositories with no inference crate layout, so the derivation is
// injected here. `inference-closure-digest.test.mjs` covers the real derivation.
const stubClosureDigest = async (provider) =>
  createHash("sha256").update(`closure:${provider ?? "none"}`).digest("hex");


const execFileAsync = promisify(execFile);

test("diagnostic LTX safety canary output is structurally non-ingestible", () => {
  for (const [id, status] of [
    ["ltx-safety-canary", "diagnostic_canary_complete"],
    ["ltx-product-envelope-canary", "diagnostic_product_envelope_canary_complete"],
  ]) {
    assert.throws(
      () => validateRecord({ id, logicalCaseId: id, status }),
      /invalid status/,
    );
  }
});

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

async function ltx25FixtureSnapshot({
  omitRoot,
  symlinkedEnhancer = false,
  escapedEnhancer = false,
  symlinkedDevAdapter = false,
} = {}) {
  const cache = await mkdtemp(path.join(tmpdir(), "memory-ltx25-cache-"));
  const snapshot = path.join(
    cache,
    "models--SceneWorks--ltx-2.5-mlx",
    "snapshots",
    LTX25_CAPTURE_REVISION,
  );
  await mkdir(path.join(snapshot, "enhancer"), { recursive: true });
  const enhancerFile = path.join(snapshot, "enhancer", "model.safetensors");
  if (symlinkedEnhancer || escapedEnhancer) {
    const target = escapedEnhancer
      ? path.join(cache, "escaped-enhancer.safetensors")
      : path.join(cache, "models--SceneWorks--ltx-2.5-mlx", "blobs", "enhancer-blob");
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, "enhancer");
    await symlink(path.relative(path.dirname(enhancerFile), target), enhancerFile);
  } else {
    await writeFile(enhancerFile, "enhancer");
  }
  await mkdir(path.join(snapshot, "distilled_lora"), { recursive: true });
  const adapterFile = path.join(
    snapshot,
    "distilled_lora",
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
  );
  if (symlinkedDevAdapter) {
    const adapterBlob = path.join(
      cache,
      "models--SceneWorks--ltx-2.5-mlx",
      "blobs",
      "adapter-blob",
    );
    await mkdir(path.dirname(adapterBlob), { recursive: true });
    await writeFile(adapterBlob, "adapter");
    await symlink(path.relative(path.dirname(adapterFile), adapterBlob), adapterFile);
  } else {
    await writeFile(adapterFile, "adapter");
  }
  for (const variant of ["distilled", "dev"]) {
    for (const tier of ["q4", "q8", "bf16"]) {
      if (`${variant}/${tier}` === omitRoot) continue;
      const root = path.join(snapshot, variant, tier);
      await mkdir(root, { recursive: true });
      await writeFile(path.join(root, "split_model.json"), JSON.stringify({ variant, tier }));
      await writeFile(path.join(root, "transformer.safetensors"), `${variant}:${tier}:transformer`);
      await writeFile(path.join(root, "vae_decoder.safetensors"), `${variant}:${tier}:conv`);
      await writeFile(path.join(root, "vae_diffusion_decoder.safetensors"), `${variant}:${tier}:diffvae`);
    }
  }
  return realpath(snapshot);
}

const phase = (value) => ({
  activeBytes: value,
  allocatorBytes: value + 10,
  reclaimableBytes: 10,
});

const audioQuality = (overrides = {}) => ({
  result: "passed",
  sampleRateHz: 24000,
  channels: 2,
  sampleCount: 48000,
  selectedPcmSha256: "a".repeat(64),
  referencePcmSha256: "b".repeat(64),
  maximumAbsoluteError: 0.001,
  meanAbsoluteError: 0.0001,
  rootMeanSquareError: 0.0002,
  maximumAbsoluteErrorThreshold: 0.01,
  meanAbsoluteErrorThreshold: 0.01,
  rootMeanSquareErrorThreshold: 0.01,
  ...overrides,
});

function canonicalAvFixture() {
  const magic = Buffer.from("SCENEWORKS_AV1\0", "ascii");
  const frame = Buffer.from([1, 2, 3, 4, 5, 6]);
  const pcm = Buffer.alloc(16);
  [0.25, -0.25, 0.5, -0.5].forEach((value, index) => pcm.writeFloatLE(value, index * 4));
  const bytes = Buffer.alloc(magic.length + 4 * 5 + 2 + 8 + 4 + 4 + 8 + frame.length + pcm.length);
  let offset = 0;
  magic.copy(bytes, offset); offset += magic.length;
  for (const value of [2, 1, 1, 24, 24000]) {
    bytes.writeUInt32LE(value, offset); offset += 4;
  }
  bytes.writeUInt16LE(2, offset); offset += 2;
  bytes.writeBigUInt64LE(4n, offset); offset += 8;
  bytes.writeUInt32LE(2, offset); offset += 4;
  bytes.writeUInt32LE(1, offset); offset += 4;
  bytes.writeBigUInt64LE(BigInt(frame.length), offset); offset += 8;
  frame.copy(bytes, offset); offset += frame.length;
  pcm.copy(bytes, offset);
  return { bytes, pcm };
}

test("canonical A/V parser binds the complete header, frame payload, and PCM bytes", () => {
  const { bytes, pcm } = canonicalAvFixture();
  const content = parsePhysicalMlxAvContent(bytes);
  assert.deepEqual(content, {
    width: 2,
    height: 1,
    frames: 1,
    fps: 24,
    sampleRateHz: 24000,
    channels: 2,
    sampleCount: 4,
    pcmSha256: createHash("sha256").update(pcm).digest("hex"),
  });
  const wrongMagic = Buffer.from(bytes);
  wrongMagic[0] ^= 0xff;
  assert.throws(() => parsePhysicalMlxAvContent(wrongMagic), /canonical SCENEWORKS_AV1 header/);
  const wrongFrameLength = Buffer.from(bytes);
  wrongFrameLength.writeBigUInt64LE(5n, Buffer.from("SCENEWORKS_AV1\0").length + 4 * 5 + 2 + 8 + 8);
  assert.throws(() => parsePhysicalMlxAvContent(wrongFrameLength), /canonical RGB geometry/);
  assert.throws(() => parsePhysicalMlxAvContent(bytes.subarray(0, -1)), /PCM payload length/);

  const record = {
    id: "imc-audio-content-binding",
    target: { geometry: { width: 2, height: 1, frames: 1 } },
    diagnostics: { measurements: [{ name: "outputFps", value: 24 }] },
    quality: {
      audio: audioQuality({
        sampleCount: 4,
        selectedPcmSha256: content.pcmSha256,
        referencePcmSha256: content.pcmSha256,
      }),
    },
  };
  const avContents = new Map([["selected_av", content], ["reference_av", content]]);
  validatePhysicalMlxAvContentsAgainstRecord(record, avContents, record.id);
  const mismatchedMetadata = structuredClone(record);
  mismatchedMetadata.quality.audio.sampleRateHz = 48000;
  assert.throws(
    () => validatePhysicalMlxAvContentsAgainstRecord(mismatchedMetadata, avContents, record.id),
    /A\/V header differs from measured video\/audio identity/,
  );
  const mismatchedPcm = structuredClone(record);
  mismatchedPcm.quality.audio.selectedPcmSha256 = "0".repeat(64);
  assert.throws(
    () => validatePhysicalMlxAvContentsAgainstRecord(mismatchedPcm, avContents, record.id),
    /A\/V PCM hashes differ from quality.audio/,
  );
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
  validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [record] });

  const malformed = runtimeComplete();
  malformed.observedMemory.overall.allocatorBytes = malformed.observedMemory.overall.activeBytes;
  malformed.id = recordId(malformed);
  assert.throws(() => validateRecord(malformed), /only the measured activeBytes/);

  const overCapacity = runtimeComplete();
  overCapacity.hardware.memoryBytes = 100;
  overCapacity.id = recordId(overCapacity);
  assert.throws(() => validateRecord(overCapacity), /exceed probed hardware/);
});

// sc-18864 GUARD: the MLX wired ceiling is now checked on `runtime_complete`, not only on
// `complete`. It was checked only on `complete` before, which is exactly how three
// `mlx:flux2_dev` runtime-complete records shipped claiming up to 26.0 GB more wired residency
// than the probed limit. Exercised at the boundary in both directions.
test("runtime-complete MLX telemetry is held to the probed wired ceiling", () => {
  const wiredLimit = 80 * 1024 ** 3;

  const atLimit = qwenPositiveComplete();
  atLimit.status = "runtime_complete";
  atLimit.sweep = {
    axes: [{ parameter: "decodeTileEdge", testedValues: [512] }],
    cases: [{ parameters: { decodeTileEdge: 512, decodeOverlap: 64 }, result: "passed" }],
    rangeVerified: true,
  };
  atLimit.scenarios = runtimeComplete().scenarios;
  atLimit.negativeMutation = null;
  atLimit.quality = runtimeComplete().quality;
  atLimit.predictedPeakBytes = { overall: wiredLimit };
  atLimit.observedMemory = {
    conditioning: { activeBytes: 1, allocatorBytes: 1, reclaimableBytes: 0 },
    denoise: { activeBytes: 1, allocatorBytes: 1, reclaimableBytes: 0 },
    decode: { activeBytes: wiredLimit, allocatorBytes: wiredLimit, reclaimableBytes: 0 },
    overall: { activeBytes: wiredLimit, allocatorBytes: wiredLimit, reclaimableBytes: 0 },
  };
  atLimit.logicalCaseId = logicalCaseId(atLimit);
  atLimit.id = recordId(atLimit);
  assert.equal(validateRecord(atLimit), atLimit, "resident bytes exactly at the ceiling are admissible");

  const overLimit = structuredClone(atLimit);
  for (const name of ["decode", "overall"]) {
    overLimit.observedMemory[name] = {
      activeBytes: wiredLimit + 1, allocatorBytes: wiredLimit + 1, reclaimableBytes: 0,
    };
  }
  overLimit.id = recordId(overLimit);
  assert.throws(() => validateRecord(overLimit), /overall wired bytes exceed the probed wired ceiling/);

  // And the co-existence BOUND may sit far above the host without disqualifying the capture: it is
  // a peak-over-window summed with an instantaneous cache reading, which the allocator releases
  // under pressure. This is the shape every committed LTX record carries, and these two numbers are
  // `imc-2c064567893ea869006e`'s `observedMemory.overall` pair verbatim (their sum is its
  // `allocatorBytes`, 142_648_318_860, against a 137_438_953_472-byte host).
  const elasticCache = structuredClone(atLimit);
  const resident = 37_931_479_408;
  const reclaimable = 104_716_839_452;
  assert.ok(resident + reclaimable > elasticCache.hardware.memoryBytes, "fixture must discriminate");
  for (const name of ["conditioning", "denoise", "decode", "overall"]) {
    elasticCache.observedMemory[name] = {
      activeBytes: resident, allocatorBytes: resident + reclaimable, reclaimableBytes: reclaimable,
    };
  }
  elasticCache.id = recordId(elasticCache);
  assert.equal(validateRecord(elasticCache), elasticCache);
});

// sc-18864 GUARD: the immutable v4 provider receipts are projected, never rewritten — and a
// receipt whose aliases are NOT copies of `allocatorBytes` is refused rather than normalised,
// because that would mean discarding a reading the field names claimed.
test("the v4 provider-receipt projection drops copies and refuses non-copies", () => {
  const v4 = {
    overall: { activeBytes: 100, allocatorBytes: 150, deviceBytes: 150, wiredBytes: 150, reclaimableBytes: 50 },
  };
  assert.deepEqual(projectPhaseMetricsToSchemaV5(structuredClone(v4), "receipt"), {
    overall: { activeBytes: 100, allocatorBytes: 150, reclaimableBytes: 50 },
  });

  for (const alias of ["deviceBytes", "wiredBytes"]) {
    const tampered = structuredClone(v4);
    tampered.overall[alias] = 151;
    assert.throws(
      () => projectPhaseMetricsToSchemaV5(tampered, "receipt"),
      new RegExp(`${alias} is 151, not a copy of allocatorBytes 150`),
      `${alias} must be refused individually`,
    );
  }

  // A v5 receipt passes through untouched, so the projection cannot rewrite current captures.
  const v5 = { overall: { activeBytes: 100, allocatorBytes: 150, reclaimableBytes: 50 } };
  assert.deepEqual(projectPhaseMetricsToSchemaV5(structuredClone(v5), "receipt"), v5);
});

test("complete status still rejects overall-only runtime telemetry", () => {
  const record = runtimeComplete();
  record.status = "complete";
  record.id = recordId(record);
  assert.throws(() => validateRecord(record), /conditioning/);
  assert.throws(
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [record] }),
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
  validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [record] });

  const stringAxis = structuredClone(record);
  stringAxis.sweep.axes.push({ parameter: "transformerWindowComponent", testedValues: ["dit"] });
  assert.throws(
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [stringAxis] }),
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
    // sc-18864: keep the derived identity intact so this row exercises the overall-covers-phases
    // rule and not the identity rule two rows down.
    [(r) => (r.observedMemory.overall = { activeBytes: 1, allocatorBytes: 1, reclaimableBytes: 0 }), /cover/],
    [(r) => (r.observedMemory.decode.allocatorBytes = 1), /allocator bytes must equal/],
    [(r) => (r.observedMemory.decode.reclaimableBytes = 999), /allocator bytes must equal/],
    // sc-18864 review: the two rows above both drive `allocatorBytes` BELOW `active + reclaimable`,
    // so a regression of the identity to the old one-sided `allocator >= active + reclaimable`
    // shipped green in JS while the Rust mirror caught it. This row drives it ABOVE the sum — the
    // direction that lets a DERIVED field inflate past its own derivation, which is precisely the
    // drift this story removes. `overall` is raised in step so the failure is the identity rule and
    // not the overall-covers-phases rule that would otherwise fire first.
    [(r) => {
      r.observedMemory.decode.allocatorBytes = r.observedMemory.decode.allocatorBytes + 5_000;
      r.observedMemory.overall.allocatorBytes = r.observedMemory.decode.allocatorBytes;
    }, /allocator bytes must equal/],
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
test("runtime-complete requires lifecycle scenarios to be wholly deferred or fully proven", () => {
  const proven = runtimeComplete();
  for (const name of ["warm_repeat", "cancel", "error"]) {
    const scenario = proven.scenarios.find((entry) => entry.name === name);
    scenario.result = "passed";
    scenario.reason = `${name} executed under the selected request scope`;
    if (name !== "warm_repeat") {
      scenario.cleanupVerified = true;
      scenario.warmFollowUpPassed = true;
    }
  }
  proven.id = recordId(proven);
  assert.equal(validateRecord(proven), proven);
  validateBundle({
    schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION,
    sourceSessions: [], records: [proven],
  });

  for (const mutate of [
    (record) => { record.scenarios.find((entry) => entry.name === "warm_repeat").result = "not_run"; },
    (record) => { record.scenarios.find((entry) => entry.name === "cancel").cleanupVerified = false; },
    (record) => { delete record.scenarios.find((entry) => entry.name === "error").warmFollowUpPassed; },
    (record) => { record.scenarios.find((entry) => entry.name === "warm_repeat").reason = ""; },
  ]) {
    const record = structuredClone(proven);
    mutate(record);
    record.id = recordId(record);
    assert.throws(() => validateRecord(record), /lifecycle|non-empty/);
    assert.throws(() => validateBundle({
      schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION,
      sourceSessions: [], records: [record],
    }), /schema validation failed|non-empty/);
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
      schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, sourceSessions: [], records: [secondCase],
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
      schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, sourceSessions: [], records: [mismatch],
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
    schemaVersion: SCHEMA_VERSION,
    harnessVersion: HARNESS_VERSION,
    records: [record],
  }).records[0], record);
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
        { schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [record] },
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

test("a record with no closure digest reads historical, never current and never a pin fallback", () => {
  // sc-22512: this used to assert a THROW. Absence of a currency term is not a defect — under E8 it
  // withholds an improvement instead of blocking. `historical` is the conservative answer, and the
  // property that actually matters is preserved and asserted directly: it is not the pin-equality
  // fallback this replaced. The lane digest below is deliberately UNEQUAL to anything the record
  // carries, and the record's inference revision matches the run's, so a pin-equality regression
  // would read `current` here and still fail this test.
  const record = complete({ evidenceScope: "authoritative" });
  delete record.repositories.inference.closureDigest;
  assert.equal(
    evidenceSemantics(record, {
      sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
      inference: record.repositories.inference.revision,
      inferenceClosureDigests: {
        [`${record.backend}:${record.target.provider}`]: "f".repeat(64),
      },
    }),
    "historical",
  );
});

test("an undeclared provider reads historical rather than being treated as current", () => {
  // sc-22512: the lane-declaration REFUSAL is gone; the "never current" half is what protected
  // anything, and it is asserted here without making an undeclared lane red the suite.
  const record = complete({ evidenceScope: "authoritative" });
  assert.equal(
    evidenceSemantics(record, {
      sceneWorks: record.repositories.sceneWorks.matrixSourceRevision,
      inference: record.repositories.inference.revision,
      inferenceClosureDigests: {},
    }),
    "historical",
  );
});

test("LTX-2.5 evidence identity requires and hashes transformer and decoder axes", () => {
  const base = complete({
    target: {
      modelId: "ltx_2_5",
      provider: "ltx_2_5_distilled",
      tier: "q4",
      mode: "text_to_video",
      overlay: "none",
      transformerVariant: "distilled",
      decoder: "conv",
      geometry: { width: 768, height: 512, batch: 1, frames: 145 },
    },
  });
  assert.equal(validateRecord(base), base);
  validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [base] });

  const identities = new Set();
  for (const transformerVariant of ["distilled", "dev"]) {
    for (const decoder of ["conv", "diffvae"]) {
      identities.add(logicalCaseId({
        ...base,
        target: { ...base.target, transformerVariant, decoder },
      }));
    }
  }
  assert.equal(identities.size, 4, "each transformer/decoder pair must have a distinct logical id");

  for (const field of ["transformerVariant", "decoder"]) {
    const missing = structuredClone(base);
    delete missing.target[field];
    missing.logicalCaseId = logicalCaseId(missing);
    missing.id = recordId(missing);
    assert.throws(
      () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [missing] }),
      /schema validation failed/,
      `${field} is required by the persisted schema`,
    );
    assert.throws(() => validateRecord(missing), /must identify|must identify transformerVariant and decoder/);
  }

  const sessionId = `ims-${"7".repeat(20)}`;
  const direct = { kind: "direct", sourceSessionIds: [sessionId] };
  const derived = structuredClone(base);
  derived.derivation = {
    memory: direct, quality: direct, negativeMutation: direct,
    lifecycle: direct, loadability: direct, overlay: direct,
    justification: "exact LTX pipeline capture",
  };
  const source = {
    id: sessionId,
    kind: "unit_test",
    command: "exact LTX fixture",
    sourcePath: "docs/calibration/sc-test/ltx-exact.log",
    capturedAt: derived.capturedAt,
    repositories: structuredClone(derived.repositories),
    hardware: { probe: "fixture", memoryBytes: derived.hardware.memoryBytes },
    target: {
      tier: derived.target.tier,
      mode: derived.target.mode,
      overlay: derived.target.overlay,
      transformerVariant: derived.target.transformerVariant,
      decoder: derived.target.decoder,
      rung: derived.strategy.rung,
    },
    stdoutSha256: "8".repeat(64),
    inputs: [{
      role: "base", path: "/fixture/q4", bytes: 1, sha256: "9".repeat(64),
      repository: derived.artifact.repository,
      resolvedRevision: derived.artifact.resolvedRevision,
      variant: derived.target.tier,
    }],
    outputs: [{ path: "fixture.json", sha256: "a".repeat(64) }],
    claims: ["memory", "quality", "negative_mutation", "lifecycle", "loadability", "overlay"],
    result: "passed",
  };
  const derivedBundle = {
    schemaVersion: SCHEMA_VERSION,
    harnessVersion: HARNESS_VERSION,
    sourceSessions: [source],
    records: [derived],
  };
  validateBundle(derivedBundle);
  const nonPhysicalAudio = structuredClone(derivedBundle);
  nonPhysicalAudio.records[0].quality.audio = audioQuality();
  assert.throws(
    () => validateBundle(nonPhysicalAudio),
    /schema validation|physical.*A\/V source session|source must be physical_mlx/,
    "typed audio cannot be sourced from a non-physical derived record",
  );
  const failedAudio = structuredClone(derived);
  failedAudio.quality.audio = audioQuality({ result: "failed" });
  assert.throws(() => validateRecord(failedAudio), /audio quality did not pass/);
  const overThresholdAudio = structuredClone(derived);
  overThresholdAudio.quality.audio = audioQuality({ maximumAbsoluteError: 0.02 });
  assert.throws(() => validateRecord(overThresholdAudio), /audio quality threshold exceeded/);
  for (const field of ["tier", "mode", "overlay", "rung", "transformerVariant", "decoder"]) {
    const crossed = structuredClone(derivedBundle);
    crossed.sourceSessions[0].target[field] = "wrong";
    if (field === "tier") crossed.sourceSessions[0].inputs[0].variant = "wrong";
    assert.throws(
      () => validateBundle(crossed),
      /wrong LTX|schema validation failed/,
      `${field} cannot cross LTX derivation identities`,
    );
  }
  const missingSourceTarget = structuredClone(derivedBundle);
  delete missingSourceTarget.sourceSessions[0].target;
  assert.throws(
    () => validateBundle(missingSourceTarget),
    /schema validation failed|without a target identity/,
  );
  // One LTX derivation record must not retro-type the non-LTX sessions the bundle also carries:
  // a non-LTX session can never declare transformerVariant/decoder, so a per-session requirement
  // would red terminal ingestion of the existing corpus.
  const mixedCorpus = structuredClone(derivedBundle);
  const legacySession = structuredClone(mixedCorpus.sourceSessions[0]);
  legacySession.id = `ims-${"c".repeat(20)}`;
  delete legacySession.target.transformerVariant;
  delete legacySession.target.decoder;
  mixedCorpus.sourceSessions.push(legacySession);
  validateBundle(mixedCorpus);
  // The LTX session itself still has to be typed.
  const untypedLtxSource = structuredClone(derivedBundle);
  delete untypedLtxSource.sourceSessions[0].target.transformerVariant;
  delete untypedLtxSource.sourceSessions[0].target.decoder;
  assert.throws(
    () => validateBundle(untypedLtxSource),
    /schema validation failed|wrong LTX/,
  );
});

// -----------------------------------------------------------------------------------------------
// sc-22514 / epic acceptance test 1: the PLAN FORMAT cannot express a second measurement of one
// (model, tier, lane) cell, nor any sweep. This is a structural claim about
// packages/memory-anchor-plan.schema.json, not a review convention, so it is asserted against the
// schema itself and against the checked-in plan the capture commands read.
// -----------------------------------------------------------------------------------------------
// sc-22734. The per-anchor `strategy` override: a plan row may name its own single composition,
// for a provider whose contract classifies the lane default as StructurallyNotApplicable (SenseNova
// on candle). It stays a single composition — a rung GRID is still unwritable — and a row without
// one still takes `ANCHOR_STRATEGY[backend]`.
test("a plan row's strategy override replaces the lane default, and is still one composition", () => {
  const key = "fixture_model:q4:candle";
  const defaulted = planAnchor(anchorPlanFixture(key), key);
  assert.deepEqual(defaulted.strategy, {
    rung: ANCHOR_STRATEGY.candle.rung,
    engagedRungs: [...ANCHOR_STRATEGY.candle.engagedRungs],
    parameters: {},
  });

  const override = { rung: "resident", engagedRungs: ["resident"] };
  const overridden = anchorPlanFixture(key, { strategy: override });
  assert.equal(validatePlan(overridden), overridden);
  assert.deepEqual(planAnchor(overridden, key).strategy, { ...override, parameters: {} });
  // …and the override changes the case identity, so an overridden capture can never be mistaken
  // for a default-composition one.
  assert.notEqual(planAnchor(overridden, key).logicalCaseId, defaulted.logicalCaseId);

  // The MLX row of the same shape is unchanged by an override that names the lane default.
  const mlxKey = "fixture_model:q4:mlx";
  assert.deepEqual(
    planAnchor(anchorPlanFixture(mlxKey, { strategy: override }), mlxKey).strategy,
    planAnchor(anchorPlanFixture(mlxKey), mlxKey).strategy,
  );

  // A grid, an unknown rung, a missing member and a stray parameter block are all unwritable.
  for (const strategy of [
    { rung: "resident" },
    { engagedRungs: ["resident"] },
    { rung: "sequential", engagedRungs: ["resident"] },
    { rung: "resident", engagedRungs: ["resident", "resident"] },
    { rung: "resident", engagedRungs: [] },
    { rung: ["resident", "staged_residency"], engagedRungs: ["resident"] },
    { rung: "resident", engagedRungs: ["resident"], parameters: { decodeTileEdge: 512 } },
  ]) {
    assert.throws(
      () => validatePlan(anchorPlanFixture(key, { strategy })),
      /anchor plan is invalid/,
      JSON.stringify(strategy),
    );
  }
});

test("the anchor plan schema cannot express a duplicate cell or any sweep", async () => {
  const plan = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  assert.equal(validatePlan(plan), plan);

  // 1. ONE anchor per (model, tier, lane), enforced by the anchor set being a JSON OBJECT keyed on
  //    that triple. A duplicate key is not a rejected document; it is not a document at all.
  assert.equal(ANCHOR_PLAN_SCHEMA.properties.anchors.type, "object");
  assert.equal(
    ANCHOR_PLAN_SCHEMA.properties.anchors.propertyNames.pattern,
    "^[a-z][a-z0-9_]*:(q4|q8|bf16):(mlx|candle)$",
  );
  const keys = Object.keys(plan.anchors);
  assert.equal(new Set(keys).size, keys.length);
  for (const key of keys) {
    assert.match(key, new RegExp(ANCHOR_PLAN_SCHEMA.properties.anchors.propertyNames.pattern));
  }
  assert.deepEqual(planAnchors(plan).map((planned) => planned.logicalCaseId).sort(),
    [...new Set(planAnchors(plan).map((planned) => planned.logicalCaseId))].sort(),
    "one anchor per cell means one logical case per cell");

  // 2. A key outside the (model, tier, lane) vocabulary — the only way to write a second entry for
  //    one cell — is rejected.
  for (const key of ["krea_2_turbo:q4:mlx:second", "krea_2_turbo:q4", "krea_2_turbo:q4:cuda", "krea_2_turbo:fp8:mlx"]) {
    const duplicate = structuredClone(plan);
    duplicate.anchors[key] = structuredClone(plan.anchors["krea_2_turbo:q4:mlx"]);
    assert.throws(() => validatePlan(duplicate), /anchor plan is invalid/, key);
  }

  // 2b. The key PATTERN only constrains the SHAPE of a model id, so a well-shaped INVENTED id was
  //     the remaining escape hatch: `krea_2_turbo_b:q4:mlx`, a byte-for-byte copy of
  //     `krea_2_turbo:q4:mlx`, is schema-valid and is a second measurement of one physical cell
  //     wearing a new name. validatePlan closes it by requiring the model id to be a
  //     docs/generated/memory-matrix.json `models[].id` or one of that file's own
  //     `summary.outOfMatrixEntries` — a coherence check between two artifacts that are both
  //     PRESENT in the tree, which never asks whether a cell has been measured.
  for (const invented of ["krea_2_turbo_b:q4:mlx", "krea_2_turbo_2:q4:mlx", "not_a_model:q4:candle"]) {
    const renamed = structuredClone(plan);
    renamed.anchors[invented] = structuredClone(plan.anchors["krea_2_turbo:q4:mlx"]);
    assert.throws(
      () => validatePlan(renamed),
      /is not a docs\/generated\/memory-matrix\.json model/,
      invented,
    );
  }
  // The allowlist half: an out-of-matrix model the matrix itself names is accepted, and today that
  // is exactly the MiniMax-H3 pair — which is why the shipped plan's three `minimax_h3` anchors
  // validate above even though `minimax_h3` is in no `models[]` row.
  const matrix = JSON.parse(
    await readFile(new URL("../docs/generated/memory-matrix.json", import.meta.url)),
  );
  const outOfMatrix = matrix.summary.outOfMatrixEntries.map((entry) => entry.id);
  assert.ok(outOfMatrix.includes("minimax_h3"), "minimax_h3 is the out-of-matrix allowlist entry");
  assert.ok(
    Object.keys(plan.anchors).some((key) => key.startsWith("minimax_h3:")),
    "the shipped plan exercises the out-of-matrix allowlist",
  );
  assert.equal(
    matrix.models.some((model) => model.id === "minimax_h3"),
    false,
    "…and does so because minimax_h3 is genuinely absent from models[]",
  );

  // 3. Every sweep axis the grid plan used to carry is now unwritable.
  for (const [field, value] of [
    ["cases", [{ parameters: {}, expectedResult: "passed" }]],
    ["parameters", { decodeTileEdge: 512 }],
    ["rung", "bounded_decode"],
    ["engagedRungs", ["resident", "bounded_decode"]],
    ["expectedResult", "failed"],
    ["negative", true],
    ["warmRepeats", 3],
    ["modelLoadGroup", "batch-1"],
    ["temporal", [97, 121, 145]],
    ["geometries", [{ width: 768, height: 768, batch: 1, frames: 1 }]],
  ]) {
    const swept = structuredClone(plan);
    swept.anchors["krea_2_turbo:q4:mlx"][field] = value;
    assert.throws(() => validatePlan(swept), /anchor plan is invalid/, field);
  }

  // 4. A geometry axis is one integer, so a list or a range is a type error rather than a sweep.
  for (const value of [[768, 1024], { min: 768, max: 1024 }, "768"]) {
    const swept = structuredClone(plan);
    swept.anchors["krea_2_turbo:q4:mlx"].geometry.width = value;
    assert.throws(() => validatePlan(swept), /anchor plan is invalid/);
  }
  const batched = structuredClone(plan);
  batched.anchors["krea_2_turbo:q4:mlx"].geometry.batch = 2;
  assert.throws(() => validatePlan(batched), /anchor plan is invalid/);
});

// sc-22514 / epic acceptance test 1, second half: ONE command captures ONE anchor and writes ONE
// record, with no campaign, resume or currency ceremony — and the record it writes is one the
// anchor extractor can consume.
test("one capture command writes exactly one record in the extractor's bundle shape", async () => {
  const cleanRepo = await cleanFixtureRepo();
  const output = path.join(await mkdtemp(path.join(tmpdir(), "anchor-capture-")), "anchor.json");
  const planPath = path.join(await mkdtemp(path.join(tmpdir(), "anchor-plan-")), "plan.json");
  const key = "fixture_model:q4:candle";
  await writeFile(planPath, JSON.stringify(anchorPlanFixture(key)));
  await execFileAsync(process.execPath, [
    fileURLToPath(new URL("./memory-calibration-harness.mjs", import.meta.url)),
    "capture",
    "--plan", planPath,
    "--anchor", key,
    "--provider-command", JSON.stringify([
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    ]),
    "--sceneworks-repo", cleanRepo,
    "--inference-repo", cleanRepo,
    "--output", output,
  ]);
  const bundle = JSON.parse(await readFile(output, "utf8"));
  assert.equal(bundle.schemaVersion, SCHEMA_VERSION);
  assert.equal(bundle.harnessVersion, HARNESS_VERSION);
  // EXACTLY one record. A second record in this file would mean the command captured more than the
  // one anchor it was told to.
  assert.equal(bundle.records.length, 1);
  assert.equal(bundle.records[0].logicalCaseId, planAnchor(anchorPlanFixture(key), key).logicalCaseId);
  assert.equal(bundle.records[0].target.modelId, "fixture_model");
  assert.equal(bundle.records[0].target.tier, "q4");
  assert.equal(bundle.records[0].backend, "candle");
  assert.equal(bundle.records[0].strategy.rung, ANCHOR_STRATEGY.candle.rung);
  assert.deepEqual(bundle.records[0].strategy.parameters, {});
  // The `{records: [...]}` shape scripts/extract-memory-anchors.mjs walks, validated by the same
  // bundle validator every retained corpus is held to.
  assert.equal(validateBundle(bundle), bundle);

  // Every WRITING arm names its destination, and omitting it is a usage error rather than the raw
  // `TypeError` `path.resolve(undefined)` used to throw from inside `atomicWrite`. `capture` is
  // additionally proven to refuse BEFORE the provider runs.
  for (const argv of [
    ["plan"],
    ["ingest", "--input", output],
    ["capture", "--plan", planPath, "--anchor", key,
      "--provider-command", JSON.stringify(["node", "must-not-start.mjs"]),
      "--sceneworks-repo", cleanRepo, "--inference-repo", cleanRepo],
  ]) {
    const failure = await execFileAsync(process.execPath, [
      fileURLToPath(new URL("./memory-calibration-harness.mjs", import.meta.url)),
      ...argv,
    ]).then(() => null, (error) => error);
    assert.ok(failure, `${argv[0]} must refuse without --output`);
    assert.match(failure.stderr, /--output is required/, argv[0]);
    assert.doesNotMatch(failure.stderr, /TypeError/, argv[0]);
  }
});

test("a capture names one anchor the plan declares, before starting the provider", async () => {
  const plan = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: "qwen_image:q4",
      providerCommand: [process.execPath, "must-not-start.mjs"],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    }),
    /must be <modelId>:<tier>:<backend>/,
  );
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: "qwen_image:q4:cuda",
      providerCommand: [process.execPath, "must-not-start.mjs"],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    }),
    /must be <modelId>:<tier>:<backend>/,
  );
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: "not_a_model:q4:mlx",
      providerCommand: [process.execPath, "must-not-start.mjs"],
      sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
      inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    }),
    /unknown anchor "not_a_model:q4:mlx"/,
  );
});

test("a legacy Qwen completion cannot suppress a provenance-required recapture", async () => {
  const record = qwenPositiveComplete();
  validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [record] });
  const provenanceRequired = structuredClone(record);
  provenanceRequired.sourceProvenance = "physical_mlx_v1";
  provenanceRequired.logicalCaseId = logicalCaseId(provenanceRequired);
  provenanceRequired.id = recordId(provenanceRequired);
  assert.throws(
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [provenanceRequired] }),
    /exact artifact inventory|missing source-session derivation/,
  );
});

test("runtime bundle validation matches schema closure for malformed gated and nested values", () => {
  const valid = { schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [complete()] };
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
    schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION,
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
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, sourceSessions: [qualityWithoutInputs], records: [] }),
    /artifact claims require exact inputs|schema validation failed/,
  );

  const negativeWithoutControl = structuredClone(source);
  negativeWithoutControl.claims = ["negative_mutation"];
  negativeWithoutControl.inputs = negativeWithoutControl.inputs.filter((input) => input.role !== "control");
  assert.throws(
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, sourceSessions: [negativeWithoutControl], records: [] }),
    /artifact claim is missing its exact tier\/overlay inputs/,
  );

  const missing = structuredClone(record);
  delete missing.derivation;
  missing.logicalCaseId = logicalCaseId(missing);
  missing.id = recordId(missing);
  assert.throws(
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, sourceSessions: [source], records: [missing] }),
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
      () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [invalid] }),
      /schema validation failed/,
    );
  }
});

// sc-18864 review GUARD: `predictedOverallCeiling` is a diagnostic COPY of the typed
// `predictedPeakBytes.overall`, and the LTX arm's own comment says they agree by construction.
// Nothing compared them, so all 14 committed LTX records shipped a diagnostic still computed over
// the `allocatorBytes` co-existence bound while the typed field beside it carried the resident
// peak. Both directions are rejected: an ordering in either direction is what let one quantity
// carry two values.
test("a diagnostic ceiling that disagrees with the typed predicted peak is refused in both directions", () => {
  // The numbers are `imc-2c064567893ea869006e` verbatim: its migrated diagnostic and typed field,
  // and the pre-migration value that this guard would have caught.
  const agreeing = 39_862_665_216;
  const preMigrationOverBound = 149_786_984_448;
  const record = complete({
    status: "gated",
    evidenceScope: "authoritative",
    diagnostics: {
      adapter: "memory-mlx-adapter:ltx-2-3-staged-video",
      execution: "executed",
      blockers: ["the pinned mlx-gen-ltx crate registers no MemoryStrategyContract at all"],
      measurements: [{ name: "predictedOverallCeiling", unit: "bytes", value: agreeing }],
    },
  });
  record.predictedPeakBytes = { conditioning: 100, denoise: 200, decode: 150, overall: agreeing };
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  assert.equal(validateRecord(record), record);

  // ABOVE the typed field — the real committed defect, a ceiling over `allocatorBytes`.
  const above = structuredClone(record);
  above.diagnostics.measurements[0].value = preMigrationOverBound;
  assert.throws(() => validateRecord(above), /predictedOverallCeiling .* must equal predictedPeakBytes\.overall/);

  // BELOW the typed field — the mirror error, and the direction a one-sided rule would miss.
  const below = structuredClone(record);
  below.diagnostics.measurements[0].value = agreeing - 64 * 1024 * 1024;
  assert.throws(() => validateRecord(below), /predictedOverallCeiling .* must equal predictedPeakBytes\.overall/);

  // A record carrying only one of the two is untouched: the candle arm emits a null
  // `predictedPeakBytes`, and most arms emit no such diagnostic at all.
  const typedOnly = structuredClone(record);
  typedOnly.diagnostics.measurements = [{ name: "preRungActiveAfterClear", unit: "bytes", value: 1 }];
  assert.equal(validateRecord(typedOnly), typedOnly);
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
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [invalid] }),
    /parameterized complete strategy must sweep at least one axis/,
  );
});

test("executable runner handles a fragmented response and writes exactly one record", async () => {
  const plan = anchorPlanFixture("fixture_model:q4:candle");
  const actions = [];
  const executeProvider = async (command, args, input) => {
    const request = JSON.parse(input);
    actions.push(request.action);
    if (request.action === "run") {
      assert.equal(typeof request.planned.logicalCaseId, "string");
      assert.equal(Object.hasOwn(request.planned, "_campaignEntry"), false);
      assert.deepEqual(request.planned.strategy, {
        rung: ANCHOR_STRATEGY.candle.rung,
        engagedRungs: [...ANCHOR_STRATEGY.candle.engagedRungs],
        parameters: {},
      });
    }
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => { stdout += chunk; });
      child.stderr.on("data", (chunk) => { stderr += chunk; });
      child.on("error", reject);
      child.on("close", (code) => code === 0
        ? resolve(stdout)
        : reject(new Error(`provider exited ${code}: ${stderr}`)));
      child.stdin.end(input);
    });
  };
  const result = await captureAnchor({
    closureDigestFor: stubClosureDigest,
    plan,
    anchorKey: "fixture_model:q4:candle",
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fragmented-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
    inferenceRepo: fileURLToPath(new URL("..", import.meta.url)),
    executeProvider,
  });
  // ONE command, ONE render, ONE record — the whole point of the collapse. A second record cannot
  // appear because there is no second case to schedule and no batch to fragment.
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].hardware.deviceId, "fixture:0");
  assert.match(result.records[0].repositories.sceneWorks.revision, /^[0-9a-f]{40}$/);
  assert.deepEqual(actions, ["probe", "run"]);
});

const PHYSICAL_MLX_ANCHOR = "fixture_mlx:q4:mlx";

function physicalMlxPlan() {
  const { geometry } = complete().target;
  return anchorPlanFixture(PHYSICAL_MLX_ANCHOR, {
    provider: "fixture_mlx",
    mode: complete().target.mode,
    overlay: complete().target.overlay,
    geometry,
    fixture: "physical-mlx-fixture",
  });
}

test("physical MLX capture binds raw provider stdout, exact inventory, and persisted outputs", async () => {
  const plan = physicalMlxPlan();
  const cleanRepo = await cleanFixtureRepo();
  const rawLogDir = await mkdtemp(path.join(tmpdir(), "physical-mlx-receipts-"));
  const sourcePathPrefix = "docs/calibration/sc-test";
  const result = await captureAnchor({
    closureDigestFor: stubClosureDigest,
    plan,
    anchorKey: PHYSICAL_MLX_ANCHOR,
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

  const semanticTamper = structuredClone(result);
  const semanticSession = semanticTamper.sourceSessions[0];
  const semanticRecord = semanticTamper.records[0];
  const tamperedProviderResponse = JSON.parse(raw.toString("utf8"));
  tamperedProviderResponse.observedMemory.overall.activeBytes = 987654321;
  tamperedProviderResponse.quality.maximumError = 0.077;
  const tamperedProviderBytes = Buffer.from(JSON.stringify(tamperedProviderResponse));
  const tamperedStdoutSha256 = createHash("sha256").update(tamperedProviderBytes).digest("hex");
  const tamperedSessionId = physicalMlxSessionId({
    kind: semanticSession.kind,
    logicalCaseId: semanticRecord.logicalCaseId,
    capturedAt: tamperedProviderResponse.capturedAt,
    repositories: semanticSession.repositories,
    hardware: semanticSession.hardware,
    stdoutSha256: tamperedStdoutSha256,
  });
  const oldSessionId = semanticSession.id;
  semanticSession.id = tamperedSessionId;
  semanticSession.stdoutSha256 = tamperedStdoutSha256;
  semanticSession.sourcePath = semanticSession.sourcePath.replace(oldSessionId, tamperedSessionId);
  const semanticRequestOutput = semanticSession.outputs.find(({ role }) => role === "request");
  semanticRequestOutput.path = semanticRequestOutput.path.replace(oldSessionId, tamperedSessionId);
  for (const [claim, reference] of Object.entries(semanticRecord.derivation)) {
    if (claim !== "justification") reference.sourceSessionIds = [tamperedSessionId];
  }
  await writeFile(path.join(rawLogDir, semanticSession.sourcePath), tamperedProviderBytes);
  await writeFile(
    path.join(rawLogDir, semanticRequestOutput.path),
    await readFile(path.join(rawLogDir, session.outputs.find(({ role }) => role === "request").path)),
  );
  await assert.rejects(
    validateSourceSessionFiles(semanticTamper, rawLogDir),
    /provider response measurements do not match the evidence record/,
  );

  const repositoryTamper = structuredClone(result);
  const repositorySession = repositoryTamper.sourceSessions[0];
  const repositoryRecord = repositoryTamper.records[0];
  // Capture construction reuses the same in-memory repository object for session and record; a
  // serialized bundle does not preserve that alias, so split it before simulating an on-disk edit.
  repositorySession.repositories = structuredClone(repositorySession.repositories);
  repositorySession.repositories.inference.revision = "e".repeat(40);
  repositorySession.repositories.inference.closureDigest = "f".repeat(64);
  const repositorySessionId = physicalMlxSessionId({
    kind: repositorySession.kind,
    logicalCaseId: repositoryRecord.logicalCaseId,
    capturedAt: repositorySession.capturedAt,
    repositories: repositorySession.repositories,
    hardware: repositorySession.hardware,
    stdoutSha256: repositorySession.stdoutSha256,
  });
  const priorRepositorySessionId = repositorySession.id;
  repositorySession.id = repositorySessionId;
  repositorySession.sourcePath = repositorySession.sourcePath
    .replace(priorRepositorySessionId, repositorySessionId);
  const repositoryRequestOutput = repositorySession.outputs.find(({ role }) => role === "request");
  repositoryRequestOutput.path = repositoryRequestOutput.path
    .replace(priorRepositorySessionId, repositorySessionId);
  for (const [claim, reference] of Object.entries(repositoryRecord.derivation)) {
    if (claim !== "justification") reference.sourceSessionIds = [repositorySessionId];
  }
  await writeFile(path.join(rawLogDir, repositorySession.sourcePath), raw);
  await writeFile(
    path.join(rawLogDir, repositoryRequestOutput.path),
    await readFile(path.join(rawLogDir, session.outputs.find(({ role }) => role === "request").path)),
  );
  await assert.rejects(
    validateSourceSessionFiles(repositoryTamper, rawLogDir),
    /request receipt provenance does not match its evidence record/,
  );

  const captureTimeTamper = structuredClone(result);
  captureTimeTamper.sourceSessions[0].capturedAt = "2026-07-28T12:00:01Z";
  await assert.rejects(
    validateSourceSessionFiles(captureTimeTamper, rawLogDir),
    /request receipt provenance does not match its evidence record/,
  );
  const targetTamper = structuredClone(result);
  targetTamper.sourceSessions[0].target.mode = "image_to_image";
  await assert.rejects(
    validateSourceSessionFiles(targetTamper, rawLogDir),
    /request receipt provenance does not match its evidence record/,
  );

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

  const av = structuredClone(authoritative);
  const avRecord = av.records[0];
  avRecord.quality.audio = audioQuality();
  for (const output of av.sourceSessions[0].outputs.filter(({ role }) => role !== "request")) {
    output.role = output.role === "selected_rgb" ? "selected_av" : "reference_av";
    output.path = `${sourcePathPrefix}/${avRecord.logicalCaseId}-${output.role}-1024x1024-f1-${output.sha256}.avbin`;
  }
  assert.equal(validateBundle(av), av, "typed A/V receipts round-trip through schema and JS validation");
  const ltxAv = structuredClone(av);
  const ltxAvRecord = ltxAv.records[0];
  delete ltxAvRecord.sourceProvenance;
  ltxAvRecord.target.modelId = "ltx_2_5";
  ltxAvRecord.target.provider = "ltx_2_5";
  ltxAvRecord.target.transformerVariant = "distilled";
  ltxAvRecord.target.decoder = "conv";
  const priorAvLogicalCaseId = ltxAvRecord.logicalCaseId;
  ltxAvRecord.logicalCaseId = logicalCaseId(ltxAvRecord);
  ltxAvRecord.id = recordId(ltxAvRecord);
  ltxAv.sourceSessions[0].target.transformerVariant = "distilled";
  ltxAv.sourceSessions[0].target.decoder = "conv";
  for (const output of ltxAv.sourceSessions[0].outputs.filter(({ role }) => role !== "request")) {
    output.path = output.path.replace(priorAvLogicalCaseId, ltxAvRecord.logicalCaseId);
  }
  assert.equal(validateBundle(ltxAv), ltxAv, "LTX A/V provenance binds the complete typed source target");
  const mismatchedAudioSource = structuredClone(ltxAv);
  const nonPhysicalSession = structuredClone(mismatchedAudioSource.sourceSessions[0]);
  nonPhysicalSession.id = `ims-${"f".repeat(20)}`;
  nonPhysicalSession.kind = "unit_test";
  nonPhysicalSession.sourcePath = "docs/calibration/sc-test/mismatched-audio-source.log";
  mismatchedAudioSource.sourceSessions.push(nonPhysicalSession);
  mismatchedAudioSource.records[0].derivation.quality.sourceSessionIds = [nonPhysicalSession.id];
  assert.throws(
    () => validateBundle(mismatchedAudioSource),
    /typed audio quality source must be physical_mlx/,
    "an unrelated physical A/V session cannot cover a non-physical quality derivation",
  );
  const crossedAv = structuredClone(av);
  crossedAv.sourceSessions[0].outputs[2].role = "reference_rgb";
  assert.throws(() => validateBundle(crossedAv), /schema validation|selected\/reference RGB or A\/V pair/);
  const badAudioHash = structuredClone(av);
  badAudioHash.records[0].quality.audio.selectedPcmSha256 = "not-a-digest";
  assert.throws(() => validateBundle(badAudioHash), /schema validation|lowercase SHA-256/);
  const failedAudio = structuredClone(av);
  failedAudio.records[0].quality.audio.result = "failed";
  assert.throws(() => validateBundle(failedAudio), /schema validation|audio quality did not pass/);
  const overThresholdAudio = structuredClone(av);
  overThresholdAudio.records[0].quality.audio.maximumAbsoluteError = 0.02;
  assert.throws(() => validateBundle(overThresholdAudio), /audio quality threshold exceeded/);

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
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: PHYSICAL_MLX_ANCHOR,
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
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: PHYSICAL_MLX_ANCHOR,
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
  const plan = physicalMlxPlan();
  const cleanRepo = await cleanFixtureRepo();
  const rawLogDir = await mkdtemp(path.join(tmpdir(), "physical-mlx-receipts-"));
  const outsideDir = await mkdtemp(path.join(tmpdir(), "physical-mlx-outside-"));
  const fixtureCommand = [
    process.execPath,
    fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    rawLogDir,
  ];
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: PHYSICAL_MLX_ANCHOR,
      providerCommand: [...fixtureCommand, "docs/calibration/../escape"],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      rawLogDir,
      sourcePathPrefix: "docs/calibration/../escape",
    }),
    /source path prefix must be a normalized path under docs\/calibration/,
  );
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: PHYSICAL_MLX_ANCHOR,
      providerCommand: [...fixtureCommand, "docs/calibration/sc-test", outsideDir],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      rawLogDir,
      sourcePathPrefix: "docs/calibration/sc-test",
    }),
    /physical MLX local output must stay under the raw log directory/,
  );
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: PHYSICAL_MLX_ANCHOR,
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

test("provider early exit is rejected without an unhandled stdin EPIPE", async () => {
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan: anchorPlanFixture("fixture_model:q4:candle"),
      anchorKey: "fixture_model:q4:candle",
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

test("provider execution rejects an adapter-attested composition that differs from the anchor", async () => {
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan: anchorPlanFixture("fixture_model:q4:candle"),
      anchorKey: "fixture_model:q4:candle",
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
    /adapter measured strategy does not match the anchor composition/,
  );
});

test("the measured load shape is a receipt field, never copied from the plan", async () => {
  // Before this, the runner wrote `loadShape: planned.loadShape` onto every record: the adapter was
  // never asked, so the field recorded the plan's CLAIM rather than what the run did, and no
  // divergence was detectable. That is the same backfill sc-16482 forbids for historical receipts,
  // applied silently to new ones. These two properties pin the fix.
  const run = (fixture) =>
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan: anchorPlanFixture("fixture_model:q4:candle"),
      anchorKey: "fixture_model:q4:candle",
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
    /adapter measured strategy does not match the anchor composition|must attest a loadShape/,
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
    () => validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [stale] }),
    /schema validation failed/,
  );

  // 4. control: the identical record at the current version passes both gates, so the rejections
  //    above are caused by the version bump and by nothing else.
  const current = complete();
  assert.equal(validateRecord(current), current);
  validateBundle({ schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, records: [current] });
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

/** An authoritative anchor plan on a given lane. */
function lanePlan({ backend, provider, fixture, modelId = provider, tier = "q4" }) {
  const key = `${modelId}:${tier}:${backend}`;
  return {
    key,
    plan: {
      schemaVersion: 1,
      anchors: {
        [key]: {
          provider,
          mode: complete().target.mode,
          overlay: complete().target.overlay,
          geometry: complete().target.geometry,
          evidenceScope: "authoritative",
          loadShape: "eager_materialization",
          calibrationFingerprint: `${provider}-formula-v1`,
          fixture,
        },
      },
    },
  };
}

async function runLaneCapture({ plan, key, closureDigestFor, onProviderInvocation }) {
  const cleanRepo = await cleanFixtureRepo();
  return captureAnchor({
    closureDigestFor,
    plan,
    anchorKey: key,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: cleanRepo,
    inferenceRepo: cleanRepo,
    onProviderInvocation,
  });
}

test("the LTX-2.5 snapshot driver binds the anchor's nested root to an exact inventory", async () => {
  const downloadEvidence = JSON.parse(
    await readFile(new URL("../config/download-pattern-evidence.json", import.meta.url)),
  );
  const authority = downloadEvidence.repos.find((entry) => entry.repo === LTX25_CAPTURE_REPOSITORY);
  assert.equal(authority.revision, LTX25_CAPTURE_REVISION);
  assert.equal(authority.resolvedSha, LTX25_CAPTURE_REVISION);
  assert.equal(authority.gated, false);
  const plan = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)),
  );
  // sc-22514: the plan carries ONE ltx_2_5 anchor per tier and lane, so the sealed inventory is
  // per-capture rather than a six-root campaign. Each MLX tier is prepared on its own here, which
  // is exactly how a capture command prepares it.
  const ltx25Keys = Object.keys(plan.anchors)
    .filter((key) => key.startsWith("ltx_2_5:") && key.endsWith(":mlx"))
    .sort();
  assert.ok(ltx25Keys.length, "the plan must declare LTX-2.5 MLX anchors");
  const snapshot = await ltx25FixtureSnapshot();
  for (const key of ltx25Keys) {
    const planned = planAnchor(plan, key);
    const prepared = await prepareLtx25CaptureArtifacts(snapshot, [planned]);
    assert.equal(prepared.repository, LTX25_CAPTURE_REPOSITORY);
    assert.equal(prepared.revision, LTX25_CAPTURE_REVISION);
    assert.equal(prepared.enhancer.root, path.join(snapshot, "enhancer"));
    assert.ok(prepared.enhancer.bytes > 0);
    assert.match(prepared.enhancer.sha256, /^[0-9a-f]{64}$/);
    const artifactKey = `${planned.target.transformerVariant}/${planned.target.tier}`;
    assert.deepEqual([...prepared.artifacts.keys()], [artifactKey]);
    const artifact = prepared.artifacts.get(artifactKey);
    const environment = ltx25ProviderEnvironment(prepared, planned, {
      SCENEWORKS_LTX25_ROOT: "/stale/root",
      SCENEWORKS_MEMORY_MODEL_BYTES: "1",
      SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256: "0".repeat(64),
      SCENEWORKS_MEMORY_CAPTURE_DIR: "/capture/raw",
      SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX: "docs/calibration/sc-18783-terminal",
    });
    assert.equal(environment.SCENEWORKS_LTX25_REPOSITORY, LTX25_CAPTURE_REPOSITORY);
    assert.equal(environment.SCENEWORKS_LTX25_REVISION, LTX25_CAPTURE_REVISION);
    assert.equal(environment.SCENEWORKS_LTX25_ROOT, artifact.root);
    assert.equal(environment.SCENEWORKS_MEMORY_MODEL_BYTES, String(artifact.bytes));
    assert.equal(environment.SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256, artifact.sha256);
    assert.equal(environment.SCENEWORKS_LTX25_ENHANCER_BYTES, String(prepared.enhancer.bytes));
    assert.equal(
      environment.SCENEWORKS_LTX25_ENHANCER_INVENTORY_SHA256,
      prepared.enhancer.sha256,
    );
    if (planned.target.transformerVariant === "dev") {
      assert.equal(environment.SCENEWORKS_LTX25_DEV_ADAPTER_BYTES, String(prepared.devAdapter.bytes));
      assert.equal(environment.SCENEWORKS_LTX25_DEV_ADAPTER_SHA256, prepared.devAdapter.sha256);
    } else {
      assert.equal(environment.SCENEWORKS_LTX25_DEV_ADAPTER_BYTES, undefined);
      assert.equal(environment.SCENEWORKS_LTX25_DEV_ADAPTER_SHA256, undefined);
    }
    assert.equal(environment.SCENEWORKS_MEMORY_CAPTURE_DIR, "/capture/raw");
    assert.equal(
      environment.SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX,
      "docs/calibration/sc-18783-terminal",
    );
  }

  const selected = ltx25Keys.map((key) => planAnchor(plan, key));
  const prepared = await prepareLtx25CaptureArtifacts(snapshot, selected);
  await writeFile(path.join(snapshot, "enhancer", "model.safetensors"), "enhancer-mutated");
  await writeFile(prepared.devAdapter.path, "adapter-mutated");
  const mutated = await prepareLtx25CaptureArtifacts(snapshot, selected);
  assert.notEqual(
    mutated.enhancer.sha256,
    prepared.enhancer.sha256,
    "mutating shared enhancer bytes must change the sealed receipt identity",
  );
  assert.notEqual(
    mutated.devAdapter.sha256,
    prepared.devAdapter.sha256,
    "mutating the dev refinement file must change the sealed receipt identity",
  );
  assert.deepEqual(
    [...mutated.artifacts].map(([key, artifact]) => [key, artifact.sha256]),
    [...prepared.artifacts].map(([key, artifact]) => [key, artifact.sha256]),
    "shared-artifact mutation must not be hidden inside a tier-root inventory",
  );

  const wrongRevision = path.join(path.dirname(snapshot), "a".repeat(40));
  await mkdir(wrongRevision, { recursive: true });
  await assert.rejects(
    prepareLtx25CaptureArtifacts(wrongRevision, selected),
    new RegExp(
      `must be models--SceneWorks--ltx-2\\.5-mlx/snapshots/${LTX25_CAPTURE_REVISION}`,
    ),
  );
  const missingLayout = await ltx25FixtureSnapshot({
    omitRoot: `${selected[0].target.transformerVariant}/${selected[0].target.tier}`,
  });
  await assert.rejects(
    prepareLtx25CaptureArtifacts(missingLayout, selected),
    /nested artifact root [a-z0-9_]+\/[a-z0-9]+ is missing/,
  );
  const normalHfSymlinks = await ltx25FixtureSnapshot({
    symlinkedEnhancer: true,
    symlinkedDevAdapter: true,
  });
  assert.match(
    (await prepareLtx25CaptureArtifacts(normalHfSymlinks, selected)).enhancer.sha256,
    /^[0-9a-f]{64}$/,
  );
  const escapedHfSymlink = await ltx25FixtureSnapshot({ escapedEnhancer: true });
  await assert.rejects(
    prepareLtx25CaptureArtifacts(escapedHfSymlink, selected),
    /escaped its trusted root/,
  );
});

test("the LTX-2.5 anchor capture injects the selected root only after the hardware probe", async () => {
  const plan = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)),
  );
  const snapshot = await ltx25FixtureSnapshot();
  const cleanRepo = await cleanFixtureRepo();
  let capturedEnvironment;
  let capturedPlanned;
  await assert.rejects(
    captureAnchor({
      closureDigestFor: stubClosureDigest,
      plan,
      anchorKey: "ltx_2_5:q4:mlx",
      providerCommand: ["fixture-ltx25-provider"],
      sceneWorksRepo: cleanRepo,
      inferenceRepo: cleanRepo,
      ltx25SnapshotRoot: snapshot,
      executeProvider: async (_command, _args, input, options) => {
        const request = JSON.parse(input);
        if (request.action === "probe") {
          assert.equal(options.env.SCENEWORKS_LTX25_REPOSITORY, LTX25_CAPTURE_REPOSITORY);
          assert.equal(options.env.SCENEWORKS_LTX25_REVISION, LTX25_CAPTURE_REVISION);
          assert.equal(options.env.SCENEWORKS_LTX25_ROOT, undefined);
          assert.equal(options.env.SCENEWORKS_MEMORY_MODEL_BYTES, undefined);
          assert.equal(options.env.SCENEWORKS_LTX25_ENHANCER_BYTES, undefined);
          assert.equal(options.env.SCENEWORKS_LTX25_DEV_ADAPTER_BYTES, undefined);
          return JSON.stringify({
            hardware: {
              probe: "fixture MLX probe",
              memoryBytes: 137438953472,
              model: "Mac17,6",
              chip: "Apple M5 Max",
              osVersion: "26.5.2",
              metalDevice: "Apple M5 Max",
              mlxMemoryLimitBytes: 130567005798,
              wiredLimitBytes: 87044670532,
            },
          });
        }
        capturedPlanned = request.planned;
        capturedEnvironment = options.env;
        throw new Error("stop after LTX-2.5 invocation environment capture");
      },
    }),
    /stop after LTX-2\.5 invocation environment capture/,
  );
  const key = `${capturedPlanned.target.transformerVariant}/${capturedPlanned.target.tier}`;
  assert.equal(capturedEnvironment.SCENEWORKS_LTX25_ROOT, path.join(snapshot, ...key.split("/")));
  assert.ok(Number(capturedEnvironment.SCENEWORKS_MEMORY_MODEL_BYTES) > 0);
  assert.match(capturedEnvironment.SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256, /^[0-9a-f]{64}$/);
  assert.ok(Number(capturedEnvironment.SCENEWORKS_LTX25_ENHANCER_BYTES) > 0);
  assert.match(capturedEnvironment.SCENEWORKS_LTX25_ENHANCER_INVENTORY_SHA256, /^[0-9a-f]{64}$/);
  if (capturedPlanned.target.transformerVariant === "dev") {
    assert.ok(Number(capturedEnvironment.SCENEWORKS_LTX25_DEV_ADAPTER_BYTES) > 0);
    assert.match(capturedEnvironment.SCENEWORKS_LTX25_DEV_ADAPTER_SHA256, /^[0-9a-f]{64}$/);
    // The Candle arm resolves the same adapter from the snapshot ROOT rather than by digest, so a
    // dev anchor binds both spellings (sc-22725).
    assert.equal(capturedEnvironment.SCENEWORKS_LTX25_DISTILL_LORA_ROOT, snapshot);
  } else {
    assert.equal(capturedEnvironment.SCENEWORKS_LTX25_DEV_ADAPTER_BYTES, undefined);
    assert.equal(capturedEnvironment.SCENEWORKS_LTX25_DEV_ADAPTER_SHA256, undefined);
    assert.equal(capturedEnvironment.SCENEWORKS_LTX25_DISTILL_LORA_ROOT, undefined);
  }
});

// -----------------------------------------------------------------------------------------------
// sc-22725: the SAME snapshot binding serves the CANDLE lane. LTX-2.5 reaches Candle under a
// different engine id (`ltx_2_5_distilled`, candle.rs `LTX25_ID`), which is the only reason
// `prepareLtx25CaptureArtifacts` used to refuse it — the artifacts, the layout and the env family
// are identical. These cases prove the candle plan rows reach the adapter's `run` action with the
// LTX-2.5 environment bound, and that the widened refusal is still a refusal.
// -----------------------------------------------------------------------------------------------
test("the LTX-2.5 candle anchors bind the same prepared snapshot and reach the adapter's run action", async () => {
  const plan = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)),
  );
  const candleKeys = Object.keys(plan.anchors)
    .filter((key) => key.startsWith("ltx_2_5:") && key.endsWith(":candle"))
    .sort();
  assert.deepEqual(candleKeys, ["ltx_2_5:bf16:candle", "ltx_2_5:q4:candle", "ltx_2_5:q8:candle"],
    "every shipped tier of the candle lane is planned");
  const snapshot = await ltx25FixtureSnapshot();
  const cleanRepo = await cleanFixtureRepo();
  for (const anchorKey of candleKeys) {
    let runEnvironment;
    let runPlanned;
    await assert.rejects(
      captureAnchor({
        closureDigestFor: stubClosureDigest,
        plan,
        anchorKey,
        providerCommand: ["fixture-ltx25-candle-provider"],
        sceneWorksRepo: cleanRepo,
        inferenceRepo: cleanRepo,
        ltx25SnapshotRoot: snapshot,
        executeProvider: async (_command, _args, input, options) => {
          const request = JSON.parse(input);
          if (request.action === "probe") {
            // The snapshot is prepared BEFORE the hardware probe, but the per-anchor root is
            // injected only for the run — the same ordering the MLX lane keeps.
            assert.equal(options.env.SCENEWORKS_LTX25_REPOSITORY, LTX25_CAPTURE_REPOSITORY);
            assert.equal(options.env.SCENEWORKS_LTX25_ROOT, undefined);
            return JSON.stringify({
              hardware: {
                probe: "fixture CUDA probe",
                memoryBytes: 96 * 1024 ** 3,
                deviceId: "0",
                name: "Fixture CUDA",
                computeCapability: "9.0",
                driverVersion: "999.1",
                runtimeVersion: "12.8",
              },
            });
          }
          assert.equal(request.action, "run", "the capture reaches the adapter's run action");
          runPlanned = request.planned;
          runEnvironment = options.env;
          throw new Error("stop after LTX-2.5 candle invocation environment capture");
        },
      }),
      /stop after LTX-2\.5 candle invocation environment capture/,
    );
    assert.equal(runPlanned.backend, "candle");
    assert.equal(runPlanned.target.provider, "ltx_2_5_distilled");
    assert.equal(runPlanned.target.transformerVariant, "distilled");
    assert.equal(
      runEnvironment.SCENEWORKS_LTX25_ROOT,
      path.join(snapshot, runPlanned.target.transformerVariant, runPlanned.target.tier),
      "the candle arm canonicalizes SCENEWORKS_LTX25_ROOT as <snapshot>/<variant>/<tier>",
    );
    assert.equal(runEnvironment.SCENEWORKS_LTX25_REPOSITORY, LTX25_CAPTURE_REPOSITORY);
    assert.equal(runEnvironment.SCENEWORKS_LTX25_REVISION, LTX25_CAPTURE_REVISION);
    // candle.rs `ltx25_load_spec` reads exactly these; a missing one is a hard `required_env` error.
    assert.match(runEnvironment.SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256, /^[0-9a-f]{64}$/);
    assert.ok(Number(runEnvironment.SCENEWORKS_MEMORY_MODEL_BYTES) > 0);
    assert.ok(Number(runEnvironment.SCENEWORKS_LTX25_ENHANCER_BYTES) > 0);
    // The distilled variant needs no official refinement LoRA, so its root is deliberately unbound.
    assert.equal(runEnvironment.SCENEWORKS_LTX25_DISTILL_LORA_ROOT, undefined);
  }
});

test("the widened LTX-2.5 snapshot binding still refuses a lane that does not load LTX-2.5", async () => {
  const snapshot = await ltx25FixtureSnapshot();
  const planned = planAnchor(
    JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url))),
    "ltx_2_5:q4:candle",
  );
  for (const wrong of [
    // The candle engine id on the MLX lane, and the MLX engine id on the candle lane: each is a
    // real provider, and each would prepare this snapshot for a loader that never asked for it.
    { ...planned, backend: "mlx" },
    { ...planned, target: { ...planned.target, provider: "ltx_2_5" } },
    { ...planned, target: { ...planned.target, modelId: "ltx_2_3" } },
    { ...planned, backend: "cuda" },
  ]) {
    await assert.rejects(
      prepareLtx25CaptureArtifacts(snapshot, [wrong]),
      /--ltx25-snapshot-root is valid only for the ltx_2_5 plan/,
    );
  }
});

test("every LTX-2.5 provider invocation rehashes the nested and shared artifacts", async () => {
  const plan = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)),
  );
  const cleanRepo = await cleanFixtureRepo();
  for (const [mutation, expected] of [
    ["base", /artifact inventory changed during provider execution/],
    ["enhancer", /enhancer changed after campaign preparation/],
    ["devAdapter", /devAdapter changed after campaign preparation/],
  ]) {
    const snapshot = await ltx25FixtureSnapshot();
    await assert.rejects(
      captureAnchor({
        closureDigestFor: stubClosureDigest,
        plan,
        anchorKey: "ltx_2_5:q4:mlx",
        providerCommand: ["fixture-ltx25-provider"],
        sceneWorksRepo: cleanRepo,
        inferenceRepo: cleanRepo,
        ltx25SnapshotRoot: snapshot,
        executeProvider: async (_command, _args, input, options) => {
          const request = JSON.parse(input);
          if (request.action === "probe") {
            return JSON.stringify({
              hardware: {
                probe: "fixture MLX probe",
                memoryBytes: 137438953472,
                model: "Mac17,6",
                chip: "Apple M5 Max",
                osVersion: "26.5.2",
                metalDevice: "Apple M5 Max",
                mlxMemoryLimitBytes: 130567005798,
                wiredLimitBytes: 87044670532,
              },
            });
          }
          const file = mutation === "base"
            ? path.join(options.env.SCENEWORKS_LTX25_ROOT, "transformer.safetensors")
            : mutation === "enhancer"
              ? path.join(snapshot, "enhancer", "model.safetensors")
              : path.join(
                  snapshot,
                  "distilled_lora",
                  "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
                );
          await writeFile(file, `${mutation}-mutated`);
          throw new Error("provider failed after mutating an input");
        },
      }),
      expected,
    );
  }
});

test("an anchor capture keys its closure digest by lane, not by model", async () => {
  const asked = [];
  const { plan, key } = lanePlan({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-krea" });
  const result = await runLaneCapture({
    plan,
    key,
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
  const { plan, key } = lanePlan({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-krea" });
  const result = await runLaneCapture({ plan, key, closureDigestFor: stubClosureDigest });
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

test("each lane's anchor is stamped with ITS OWN lane digest", async () => {
  // Two separate captures, because a capture is one anchor. One shared run-level digest would stamp
  // both records with whichever lane happened to be captured first.
  const digests = [];
  for (const provider of ["krea_2_turbo", "flux2_dev"]) {
    const { plan, key } = lanePlan({ backend: "candle", provider, fixture: "cap-multi" });
    const result = await runLaneCapture({ plan, key, closureDigestFor: stubClosureDigest });
    assert.equal(result.records.length, 1);
    digests.push([
      `${result.records[0].backend}:${result.records[0].target.provider}`,
      result.records[0].repositories.inference.closureDigest,
    ]);
  }
  const byLane = new Map(digests);
  assert.equal(byLane.get("candle:krea_2_turbo"), await stubClosureDigest("candle:krea_2_turbo"));
  assert.equal(byLane.get("candle:flux2_dev"), await stubClosureDigest("candle:flux2_dev"));
  assert.notEqual(byLane.get("candle:krea_2_turbo"), byLane.get("candle:flux2_dev"));
});

test("a non-authoritative anchor derives no digest at all", async () => {
  // A fixture/candidate capture can never be `current`, so it must not need an inference crate
  // layout or a declarations file. The injected deriver throws to prove it is never reached.
  const { plan, key } = lanePlan({ backend: "candle", provider: "krea_2_turbo", fixture: "cap-fixture" });
  plan.anchors[key].evidenceScope = "fixture";
  const result = await runLaneCapture({
    plan,
    key,
    closureDigestFor: async () => assert.fail("a fixture-scope capture must not derive a closure digest"),
  });
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].repositories.inference.closureDigest, undefined);
});

test("an undeclared lane is CAPTURED without a currency term, never refused", async () => {
  // sc-22512 inverted this test. It used to assert that an undeclared lane `rejects` BEFORE the
  // first capture — which meant a lane nobody had declared could not be measured at all, so the
  // absence of bookkeeping blocked the measurement that would have relieved it. That is the exact
  // shape E8 retires.
  //
  // The capture now runs and the record carries NO closure digest, which is the conservative
  // answer: `evidenceSemantics` reads such a record `historical`, so it can never certify a cell.
  // This still drives the REAL deriver (no `closureDigestFor`) against the real declarations file,
  // so it is the production path and not a stub.
  const cleanRepo = await cleanFixtureRepo();
  const invocations = [];
  // The MODEL id is real (validatePlan requires one the memory matrix names); the LANE
  // `candle:never_declared_provider` is what is undeclared, and the lane is what the digest is
  // keyed on.
  const { plan, key } = lanePlan({
    backend: "candle", provider: "never_declared_provider", modelId: "krea_2_turbo",
    fixture: "cap-undeclared",
  });
  const result = await captureAnchor({
    plan,
    anchorKey: key,
    providerCommand: [
      process.execPath,
      fileURLToPath(new URL("./fixtures/memory-provider-fixture.mjs", import.meta.url)),
    ],
    sceneWorksRepo: fileURLToPath(new URL("..", import.meta.url)),
    inferenceRepo: cleanRepo,
    onProviderInvocation: (invocation) => invocations.push(invocation),
  });
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].repositories.inference.closureDigest, undefined);
  assert.ok(invocations.length, "the undeclared lane must actually be measured, not skipped");
});

// sc-22512 removed the DECLARED-LANE half of the test below: it required every authoritative
// survivor's lane to carry a config/inference-provider-closures.json declaration, plus a converse
// sweep that reddened `npm run check` for any authoritative plan lane no workflow had captured yet.
// That is a lane-declaration gate in the strict sense E8 forbids — adding a model to the
// calibration plan reddened CI until somebody hand-maintained a second file, and the failure was
// about bookkeeping rather than about anything a capture would find. The harness no longer refuses
// an undeclared lane either (see "an undeclared lane is CAPTURED without a currency term, never
// refused" above): the capture runs and simply carries no currency term.
//
// What is kept is PRESENT-DATA agreement between two checked-in artifacts: the workflows say which
// anchors the runners will capture, and config/memory-calibration-plan.json says which anchors
// exist. A workflow naming an anchor the plan no longer declares kills a capture job on a runner
// rather than here — and nothing about that defect is a missing measurement, so it stays gated.
test("every checked-in capture invocation names a declared anchor", async () => {
  const root = new URL("../", import.meta.url);
  const plan = validatePlan(JSON.parse(await readFile(new URL("config/memory-calibration-plan.json", root))));
  const workflows = await readdir(new URL(".github/workflows/", root));

  const named = [];
  for (const file of workflows.filter((name) => name.endsWith(".yml"))) {
    const body = await readFile(new URL(`.github/workflows/${file}`, root), "utf8");
    // `harness.mjs capture` invocations only; `check` names no anchor. Windows packs the whole
    // invocation onto one line, macOS uses backslash continuations.
    for (const invocation of body.match(/memory-calibration-harness\.mjs\s+capture\b[\s\S]*?--output\s+\S+/g) ?? []) {
      const anchor = invocation.match(/--anchor\s+"?([^"\s\\]+)"?/)?.[1];
      assert.ok(anchor, `${file}: a capture invocation names no --anchor`);
      if (!anchor.includes("$")) {
        named.push([`${file} --anchor ${anchor}`, (key) => key === anchor]);
        continue;
      }
      // The Qwen invocation templates its tier from `inputs.qwen_tier`. The options come out of the
      // workflow itself rather than being restated here, so a drift on either side is caught:
      // every DECLARED tier option must reach a declared anchor.
      const tiers = body.match(/qwen_tier:[\s\S]*?options:\s*((?:\s*-\s*\w+\n)+)/)?.[1];
      assert.ok(tiers, `${file}: the templated anchor's tier input declares no options`);
      for (const tier of tiers.match(/-\s*(\w+)/g).map((line) => line.replace(/-\s*/, ""))) {
        const exact = anchor.replace("${QWEN_TIER}", tier);
        named.push([`${file} --anchor ${anchor} (tier ${tier})`, (key) => key === exact]);
      }
    }
  }
  // No population pin: retiring a capture workflow is absence, not a defect. The loop is
  // universally quantified over whatever invocations exist and goes vacuous rather than red.
  for (const [label, matches] of named) {
    assert.ok(
      Object.keys(plan.anchors).some(matches),
      `${label}: names no anchor the plan declares`,
    );
  }
});
