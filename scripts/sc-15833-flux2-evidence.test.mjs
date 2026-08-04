import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { deflateSync } from "node:zlib";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { buildMatrix, validatedInferenceCompatibility } from "./generate-memory-matrix.mjs";
import { AUDIT_CLOSURE_PATHS, AUDIT_METHOD } from "./inference-artifact-audit.mjs";
import { validateRecord } from "./memory-calibration-harness.mjs";
import {
  flux2CalibrationPlans,
  ingestFlux2Capture as ingestFlux2CaptureStrict,
  updateBundle,
  updatePlan,
} from "./sc-15833-flux2-evidence.mjs";

const SCENEWORKS_REVISION = "1".repeat(40);
const INFERENCE_REVISION = "5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d";
const LIVE_INFERENCE_REVISION = "277f423822bf1899340ed3d867c3d6a773473d7b";
const MODEL_REVISION = "2868b1461b2b6e6e05d84e52534df3632b4c7d5d";
const MODEL_INVENTORY = "896f227194e48c6e4df10cec20733f4ed1a357affc021bc131e6851b787da98b";
const CONTROL_REVISION = "b3dcd7836a0e926248dac3ccba8fc0853495764b";
const CONTROL_SHA = "516532a885d12ae84bb3c6b24ef4816ac05ffa1c9c7b93476f74652eb0a7a794";
const MATRIX_SOURCE = `source-tree:${"7".repeat(64)}`;
const FINGERPRINT = "flux2-dev-cuda-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v2";

// These tests replay immutable 5ffd capture fixtures. Live-pin authorization is tested separately;
// replay must continue validating the captured bytes after a later runtime pin makes them historical.
const ingestFlux2Capture = (capture, options = {}) => ingestFlux2CaptureStrict(capture, {
  ...options,
  enforceLivePin: false,
});
const OUTPUT = Buffer.alloc(1024 * 1024 * 3, 17);
const OUTPUT_SHA = createHash("sha256").update(OUTPUT).digest("hex");
const REPOSITORIES = {
  sceneWorks: { revision: SCENEWORKS_REVISION, dirty: false, matrixSourceRevision: MATRIX_SOURCE },
  inference: { revision: INFERENCE_REVISION, dirty: false },
};
const HARDWARE = {
  probe: "CUDA_VISIBLE_DEVICES=0; nvidia-smi + nvcc --version",
  memoryBytes: 102641958912,
  deviceId: "0",
  name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
  computeCapability: "12.0",
  driverVersion: "596.36",
  runtimeVersion: "CUDA 12.9 (nvcc V12.9.41)",
};
const BASE_INPUT = {
  role: "base",
  path: "E:/huggingface/sceneworks-staging/sc15833-flux2-dev-q4-2868b146",
  bytes: 33582076196,
  sha256: MODEL_INVENTORY,
  repository: "SceneWorks/flux2-dev-mlx",
  resolvedRevision: MODEL_REVISION,
  variant: "q4",
};
const CONTROL_INPUT = {
  role: "control",
  path: "E:/huggingface/sceneworks-staging/sc15833-flux2-control-b3dcd783/FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors",
  bytes: 8232506680,
  sha256: CONTROL_SHA,
  repository: "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
  resolvedRevision: CONTROL_REVISION,
  variant: "union-2602",
};

const RUNGS = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];

const COMPOSITIONS = {
  resident: ["resident"],
  staged_residency: ["resident", "staged_residency"],
  bounded_decode: ["resident", "staged_residency", "bounded_decode"],
  bounded_attention: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
  bounded_transformer_residency: [
    "resident", "staged_residency", "bounded_decode", "bounded_attention",
    "bounded_transformer_residency",
  ],
};

const PARAMS = {
  resident: [null, null, null, null, null],
  staged_residency: [null, null, null, null, null],
  bounded_decode: [1024, 1, null, null, null],
  bounded_attention: [1024, 1, 67108864, null, null],
  bounded_transformer_residency: [1024, 1, 67108864, 1, "dit"],
};

const STRATEGY_DEBUG = {
  staged_residency: "StagedResidency",
  bounded_decode: "BoundedDecode",
  bounded_attention: "BoundedAttention",
  bounded_transformer_residency: "BoundedTransformerResidency",
};
const RUNG_ENV = {
  resident: "resident",
  staged_residency: "staged",
  bounded_decode: "decode",
  bounded_attention: "attention",
  bounded_transformer_residency: "blocks",
};
const DEVICE_PEAK_GB = {
  resident: "21.0",
  staged_residency: "20.0",
  bounded_decode: "19.0",
  bounded_attention: "18.0",
  bounded_transformer_residency: "17.0",
};

function parametersFor(rung) {
  const [decodeTileEdge, decodeOverlap, attentionChunkSize, transformerWindowSize, transformerWindowComponent] = PARAMS[rung];
  return Object.fromEntries(Object.entries({
    decodeTileEdge, decodeOverlap, attentionChunkSize, transformerWindowSize, transformerWindowComponent,
  }).filter(([, value]) => value !== null));
}

function baseCommand(rung) {
  const outputPath = `docs/calibration/sc-15833/base-q4-${rung}.rgb`;
  const parity = rung === "resident"
    ? ""
    : " FLUX2_PARITY_REFERENCE=docs/calibration/sc-15833/base-q4-resident.rgb";
  return `CUDA_VISIBLE_DEVICES=0 FLUX2_DEV_DIR=${BASE_INPUT.path} FLUX2_QUANT=q4 FLUX2_MEMORY_RUNG=${RUNG_ENV[rung]} FLUX2_OUT=${outputPath}${parity} MEMORY_EXPECTED_ABI=3 MEMORY_EXPECTED_FINGERPRINT=${FINGERPRINT} INFERENCE_REVISION=${INFERENCE_REVISION} SCENEWORKS_REVISION=${SCENEWORKS_REVISION} MEMORY_MODEL_REVISION=${MODEL_REVISION} MEMORY_MODEL_INVENTORY_SHA256=${MODEL_INVENTORY} cargo test -p candle-gen-flux2 --release --features cuda tests::flux2_dev_probed_generate_for_offload_ab -- --ignored --nocapture --test-threads=1`;
}

function routeCommand(route) {
  const control = route === "control" ? ` FLUX2_CONTROL=${CONTROL_INPUT.path}` : "";
  return `CUDA_VISIBLE_DEVICES=0 FLUX2_DEV_DIR=${BASE_INPUT.path}${control} FLUX2_DEV_OUT_DIR=docs/calibration/sc-15833 FLUX2_DEV_W=512 FLUX2_DEV_H=512 cargo test -p sceneworks-worker --release --no-default-features --features backend-candle flux2_dev_gpu_smoke::flux2_dev_${route}_candle_gpu_smoke -- --ignored --nocapture --test-threads=1`;
}

function outputForRung(rung) {
  const output = Buffer.from(OUTPUT);
  if (rung.startsWith("bounded_")) output[0] += 2;
  return output;
}

function parityMetrics(reference, candidate) {
  let changed = 0;
  let maximum = 0;
  let absolute = 0;
  let squared = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const error = Math.abs(reference[index] - candidate[index]);
    if (error !== 0) changed += 1;
    maximum = Math.max(maximum, error);
    absolute += error;
    squared += error * error;
  }
  const rmse = Math.sqrt(squared / reference.length);
  return {
    changed: changed / reference.length,
    maximum,
    mean: absolute / reference.length,
    rmse,
    psnr: rmse === 0 ? "inf" : 20 * Math.log10(255 / rmse),
  };
}

function memoryRecord(rung) {
  const [decodeTile, overlap, attention, window, component] = PARAMS[rung];
  const bounded = rung.startsWith("bounded_");
  return {
    schema_version: 1,
    key: {
      resolved_route: "flux2_dev",
      backend: "candle",
      tier: { precision: "bf16", quant: "q4", component_precision_floors: [] },
      load_shape: "deferred_materialization",
      mode: "text_to_image",
      overlay: null,
      geometry: { width: 1024, height: 1024, batch: 1, frames: 1, reference_count: 0 },
      strategy: rung,
      engaged_composition: COMPOSITIONS[rung],
      parameters: {
        decode_tile_edge: decodeTile,
        decode_overlap: overlap,
        attention_chunk_size: attention,
        transformer_window_size: window,
        transformer_window_component: component,
      },
    },
    declared_calibration: { abi: 3, fingerprint: FINGERPRINT, load_shape: "deferred_materialization" },
    observed_calibration: { abi: 3, fingerprint: FINGERPRINT, load_shape: "deferred_materialization" },
    predicted_peak_bytes: 20_000_000_000 - RUNGS.indexOf(rung) * 1_000_000_000,
    observed_peak_bytes: 20_000_000_000 - RUNGS.indexOf(rung) * 1_000_000_000,
    inference_revision: INFERENCE_REVISION,
    sceneworks_revision: SCENEWORKS_REVISION,
    model_revision: MODEL_REVISION,
    model_inventory_sha256: MODEL_INVENTORY,
    harness_version: "candle-flux2-memory-ladder-v1",
    output_sha256: OUTPUT_SHA,
    parity: bounded
      ? { kind: "tolerance", metric: "rgb8_max_abs_error", maximum_error: 2 }
      : { kind: "exact" },
    parity_result: { kind: rung === "resident" ? "not_run" : "passed" },
  };
}

function wrappedJson(value) {
  const json = JSON.stringify(value);
  return json.match(/.{1,71}/g).join("\n");
}

function sourceHeader(capturedAt, inputs) {
  return `SC15833_SOURCE_V1 ${wrappedJson({
    schema_version: 1,
    captured_at: capturedAt,
    repositories: REPOSITORIES,
    hardware: HARDWARE,
    inputs,
  })}\n`;
}

function baseLog(rung, capturedAt, overrides = {}) {
  const record = structuredClone(memoryRecord(rung));
  Object.assign(record, overrides.record ?? {});
  let diagnostic = "";
  if (rung !== "resident") {
    const measured = parityMetrics(OUTPUT, outputForRung(rung));
    const maximum = overrides.maximumError ?? measured.maximum;
    const changed = overrides.maximumError === undefined ? measured.changed : (maximum === 0 ? 0 : measured.changed);
    const mean = overrides.maximumError === undefined ? measured.mean : (maximum === 0 ? 0 : measured.mean);
    const rmse = overrides.maximumError === undefined ? measured.rmse : (maximum === 0 ? 0 : measured.rmse);
    const psnr = overrides.maximumError === undefined ? measured.psnr : (maximum === 0 ? "inf" : measured.psnr);
    diagnostic = `MEMORY_PARITY_DIAGNOSTIC strategy=${STRATEGY_DEBUG[rung]} reference=docs/calibration/sc-15833/base-q4-resident.rgb changed_fraction=${changed} max_abs=${maximum} mean_abs=${mean} rmse=${rmse} psnr_db=${psnr}\n`;
  }
  return Buffer.from(
    `${sourceHeader(capturedAt, [BASE_INPUT])}running 1 test\nMEMORY_EVIDENCE_V1 ${wrappedJson(record)}\n${diagnostic}` +
    `MEMORY_EVIDENCE_DIAGNOSTIC gpu=0 load-peak 0.5 GB | steady 0.5 GB | overall-peak ${DEVICE_PEAK_GB[rung]} GB (baseline 0.0 GB) bytes=${OUTPUT.length} 1024x1024 out=docs/calibration/sc-15833/base-q4-${rung}.rgb\n` +
    "test tests::flux2_dev_probed_generate_for_offload_ab ... ok\n" +
    "test result: ok. 1 passed; 0 failed\n",
  );
}

const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, value) => {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) crc = (crc & 1) ? (0xedb88320 ^ (crc >>> 1)) : (crc >>> 1);
  return crc >>> 0;
});

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const chunk = Buffer.alloc(data.length + 12);
  chunk.writeUInt32BE(data.length, 0);
  typeBytes.copy(chunk, 4);
  data.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])), data.length + 8);
  return chunk;
}

function routePng(route, width = 512, height = 512, permuted = false) {
  const pixels = Buffer.alloc(width * height * 3);
  let sum = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const at = (y * width + x) * 3;
      pixels[at] = x & 0xff;
      pixels[at + 1] = y & 0xff;
      pixels[at + 2] = route === "edit" ? ((x + y) & 0xff) : ((x ^ y) & 0xff);
      sum += pixels[at] + pixels[at + 1] + pixels[at + 2];
    }
  }
  if (permuted) pixels.reverse();
  const mean = sum / pixels.length;
  let squared = 0;
  for (const value of pixels) squared += (value - mean) ** 2;
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const row = y * (width * 3 + 1);
    raw[row] = 0;
    pixels.copy(raw, row + 1, y * width * 3, (y + 1) * width * 3);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const bytes = Buffer.concat([
    PNG_SIGNATURE,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
  return { bytes, stdDev: Math.sqrt(squared / pixels.length) };
}

function routeLog(route, capturedAt, stdDev, outputBytes) {
  const testName = route === "edit"
    ? "flux2_dev_edit_candle_gpu_smoke"
    : "flux2_dev_control_candle_gpu_smoke";
  const outputPath = `docs/calibration/sc-15833/flux2_dev_${route}_candle.png`;
  const outputSha256 = createHash("sha256").update(outputBytes).digest("hex");
  const loading = route === "edit"
    ? `[smoke] loading flux2_dev EDIT (Q4) from ${BASE_INPUT.path} ...`
    : `[smoke] loading flux2_dev CONTROL (Q4) from ${BASE_INPUT.path} + ${CONTROL_INPUT.path} ...`;
  return Buffer.from(
    `${sourceHeader(capturedAt, route === "control" ? [BASE_INPUT, CONTROL_INPUT] : [BASE_INPUT])}running 1 test\n` +
    `test flux2_dev_gpu_smoke::${testName} ... ${loading}\n` +
    `[smoke] dev ${route} 512x512 std ${stdDev.toFixed(2)} -> ${outputPath} sha256=${outputSha256}\n` +
    `[smoke] DONE: flux2_dev ${route} (candle) coherent\n` +
    "ok\n" +
    "test result: ok. 1 passed; 0 failed\n",
  );
}

function fixture({ promotionIntent = "runtime_complete" } = {}) {
  const files = new Map();
  const baseSessions = RUNGS.map((rung, index) => {
    const sourcePath = `docs/calibration/sc-15833/base-q4-${rung}.log`;
    const outputPath = `docs/calibration/sc-15833/base-q4-${rung}.rgb`;
    const capturedAt = `2026-08-04T04:0${index}:00Z`;
    const output = outputForRung(rung);
    files.set(sourcePath, baseLog(rung, capturedAt, {
      record: { output_sha256: createHash("sha256").update(output).digest("hex") },
    }));
    files.set(outputPath, output);
    return {
      rung,
      command: baseCommand(rung),
      sourcePath,
      capturedAt,
      outputPath,
    };
  });
  const routeSessions = ["edit", "control"].map((route, index) => {
    const sourcePath = `docs/calibration/sc-15833/${route}-q4-resident.log`;
    const outputPath = `docs/calibration/sc-15833/flux2_dev_${route}_candle.png`;
    const capturedAt = `2026-08-04T04:1${index}:00Z`;
    const output = routePng(route);
    files.set(sourcePath, routeLog(route, capturedAt, output.stdDev, output.bytes));
    files.set(outputPath, output.bytes);
    return {
      route,
      command: routeCommand(route),
      sourcePath,
      capturedAt,
      outputPath,
    };
  });
  const excludedPath = "target/sc15833-evidence/dirty_experimental_decode_1024_1.log";
  files.set(excludedPath, Buffer.from("diagnostic only; repositories intentionally dirty"));
  const capture = {
    schemaVersion: 1,
    story: "SC-15833",
    promotionIntent,
    repositories: structuredClone(REPOSITORIES),
    hardware: structuredClone(HARDWARE),
    calibrationAbi: 3,
    calibrationFingerprint: FINGERPRINT,
    baseInput: structuredClone(BASE_INPUT),
    controlInput: structuredClone(CONTROL_INPUT),
    baseSessions,
    routeSessions,
    excludedAttempts: [{
      sourcePath: excludedPath,
      reason: "diagnostic sweep ran from a dirty experimental inference tree and is never promotion evidence",
    }],
  };
  const reader = async (relative) => {
    const value = files.get(relative);
    if (!value) throw new Error(`missing fixture ${relative}`);
    return value;
  };
  return { capture, files, reader };
}

test("SC-15833 ingests five strict physical base rungs plus real edit/control PNGs", async () => {
  const { capture, reader } = fixture();
  const ingestion = await ingestFlux2Capture(capture, { reader });
  assert.equal(ingestion.report.authorization, "historical_replay");
  assert.equal(ingestion.report.promotionEligible, false);
  assert.deepEqual(ingestion.report.blockers, []);
  assert.equal(ingestion.records.length, 5);
  assert.equal(ingestion.sourceSessions.length, 7);
  assert.equal(ingestion.report.acceptedSessions.length, 7);
  assert.equal(ingestion.report.excludedAttempts.length, 1);
  assert.match(ingestion.report.excludedAttempts[0].reason, /dirty experimental/);
  for (const record of ingestion.records) {
    validateRecord(record);
    assert.equal(record.status, "runtime_complete");
    assert.deepEqual(Object.keys(record.predictedPeakBytes), ["overall"]);
    assert.deepEqual(Object.keys(record.observedMemory), ["overall"]);
    assert.deepEqual(Object.keys(record.observedMemory.overall), ["activeBytes"]);
    const displayed = Number(DEVICE_PEAK_GB[record.strategy.rung]);
    assert.equal(record.predictedPeakBytes.overall, displayed * 1_000_000_000 + 100_000_000);
    assert.ok(record.predictedPeakBytes.overall > record.observedMemory.overall.activeBytes);
    assert.equal(
      record.diagnostics.measurements.find(({ name }) => name === "requestDevicePeakUpperBound").value,
      record.predictedPeakBytes.overall,
    );
    assert.equal(record.negativeMutation, null);
    for (const name of ["warm_repeat", "cancel", "error"]) {
      assert.equal(record.scenarios.find((scenario) => scenario.name === name).result, "not_run");
    }
    assert.equal(record.repositories.inference.revision, INFERENCE_REVISION);
    assert.equal(record.repositories.sceneWorks.revision, SCENEWORKS_REVISION);
    assert.equal(record.target.provider, "flux2_dev");
  }
  assert.deepEqual(
    ingestion.report.baseRungs.map(({ rung }) => rung),
    RUNGS,
  );
  for (const rung of ingestion.report.baseRungs) {
    assert.equal(rung.devicePeakReportedGb, Number(DEVICE_PEAK_GB[rung.rung]));
    assert.equal(rung.devicePeakUpperBoundBytes, rung.selectorPredictedPeakBytes);
    assert.ok(rung.devicePeakUpperBoundBytes > rung.activeAllocationPeakBytes);
  }
  assert.equal(ingestion.report.baseRungs[1].parity.kind, "exact");
  assert.equal(ingestion.report.baseRungs[1].parityResult.kind, "passed");
  for (const rung of ingestion.report.baseRungs.slice(2)) {
    assert.deepEqual(rung.parity, {
      kind: "tolerance", metric: "rgb8_max_abs_error", maximum_error: 2,
    });
    assert.equal(rung.parityResult.kind, "passed");
    assert.equal(rung.diagnostics.maximumError, 2);
    assert.ok(rung.diagnostics.meanError > 0);
    assert.ok(rung.diagnostics.rootMeanSquareError > 0);
    assert.ok(rung.diagnostics.psnrDb > 60);
  }
  assert.deepEqual(
    ingestion.report.routes.map(({ route }) => route).sort(),
    ["control", "edit"],
  );
  assert.ok(ingestion.report.routes.every((route) =>
    route.outputFormat === "png" && route.outputWidth === 512 && route.outputHeight === 512
    && route.outputStdDev > 0));
});

test("SC-15833 requires the device-level peak and conservatively covers display rounding", async () => {
  const missing = fixture();
  const path = "docs/calibration/sc-15833/base-q4-staged_residency.log";
  missing.files.set(
    path,
    Buffer.from(missing.files.get(path).toString("utf8").replace(/^MEMORY_EVIDENCE_DIAGNOSTIC gpu=.*\n/m, "")),
  );
  await assert.rejects(
    ingestFlux2Capture(missing.capture, { reader: missing.reader }),
    /exactly one complete device-level MEMORY_EVIDENCE_DIAGNOSTIC/,
  );

  const understated = fixture();
  const residentPath = "docs/calibration/sc-15833/base-q4-resident.log";
  understated.files.set(
    residentPath,
    Buffer.from(understated.files.get(residentPath).toString("utf8").replace("overall-peak 21.0 GB", "overall-peak 20.0 GB")),
  );
  const ingestion = await ingestFlux2Capture(understated.capture, { reader: understated.reader });
  const resident = ingestion.records.find(({ strategy }) => strategy.rung === "resident");
  assert.equal(resident.predictedPeakBytes.overall, 20_100_000_000);
  assert.ok(resident.predictedPeakBytes.overall >= resident.observedMemory.overall.activeBytes);
});

test("SC-15833 requires a unique ordered Cargo nocapture route result", async () => {
  const missingOk = fixture();
  const editPath = "docs/calibration/sc-15833/edit-q4-resident.log";
  missingOk.files.set(
    editPath,
    Buffer.from(missingOk.files.get(editPath).toString("utf8").replace("\nok\ntest result:", "\ntest result:")),
  );
  await assert.rejects(
    ingestFlux2Capture(missingOk.capture, { reader: missingOk.reader }),
    /exact approved nocapture test/,
  );

  const duplicateIdentity = fixture();
  const controlPath = "docs/calibration/sc-15833/control-q4-resident.log";
  const controlText = duplicateIdentity.files.get(controlPath).toString("utf8");
  const identity = controlText.split("\n").find((line) =>
    line.startsWith("test flux2_dev_gpu_smoke::flux2_dev_control_candle_gpu_smoke ... [smoke] "));
  duplicateIdentity.files.set(controlPath, Buffer.from(controlText.replace(identity, `${identity}\n${identity}`)));
  await assert.rejects(
    ingestFlux2Capture(duplicateIdentity.capture, { reader: duplicateIdentity.reader }),
    /exact approved nocapture test/,
  );
});

test("SC-15833 admits five Q4 base cells only through the exact audited 5ffd-to-277f compatibility", async () => {
  const manifestSource = await readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8");
  const manifest = JSON.parse(stripJsoncComments(manifestSource));
  const model = manifest.models.find(({ id }) => id === "flux2_dev");
  const bindings = model.candle.calibrations;
  assert.equal(bindings.length, 5);
  assert.deepEqual(
    bindings.map(({ rung }) => rung).sort(),
    [...RUNGS].sort(),
  );
  for (const binding of bindings) {
    assert.equal(binding.abi, 3);
    assert.equal(binding.loadShape, "deferred_materialization");
    assert.equal(binding.fingerprint, FINGERPRINT);
    assert.equal(binding.inferenceRevision, INFERENCE_REVISION);
    assert.equal(binding.compatibleInferenceRevision, LIVE_INFERENCE_REVISION);
    assert.equal(binding.provider, "flux2_dev");
    assert.equal(binding.tier, "q4");
    assert.equal(binding.mode, "text_to_image");
    assert.equal(binding.overlay, "none");
    assert.deepEqual(binding.geometry, { width: 1024, height: 1024, batch: 1, frames: 1 });
    assert.equal(binding.artifactRepository, BASE_INPUT.repository);
    assert.equal(binding.artifactResolvedRevision, BASE_INPUT.resolvedRevision);
    assert.equal(binding.artifactVariant, BASE_INPUT.variant);
  }

  const matrix = JSON.parse(await readFile(new URL("../docs/generated/memory-matrix.json", import.meta.url)));
  assert.equal(matrix.generatedFrom.inferenceRevision, LIVE_INFERENCE_REVISION);
  assert.notEqual(matrix.generatedFrom.inferenceRevision, INFERENCE_REVISION);
  const runs = matrix.calibrationRuns.filter(({ record }) => record.target.provider === "flux2_dev");
  assert.equal(runs.length, 5);
  assert.ok(runs.every(({ binding, semantics, record }) =>
    binding.eligible && binding.reasons.length === 0 && semantics === "current" &&
    record.status === "runtime_complete"));
  const cells = matrix.cells.filter((cell) =>
    cell.modelId === "flux2_dev" && cell.backend === "candle" && cell.tier === "q4" &&
    cell.mode === "text_to_image" && cell.overlay === "none",
  );
  assert.equal(cells.length, 5);
  assert.ok(cells.every((cell) => cell.state === "Runtime verified"));
  assert.ok(cells.every((cell) =>
    cell.evidence.currentEnvironmentVerification.length === 1 &&
    cell.evidence.currentEnvironmentVerification[0].recordStatus === "runtime_complete" &&
    cell.evidence.historicalVerification.length === 0,
  ));

  const proof = JSON.parse(await readFile(
    new URL("../docs/calibration/sc-15833/inference-compatibility-277f.json", import.meta.url),
  ));
  assert.equal(proof.capturedInferenceRevision, INFERENCE_REVISION);
  assert.equal(proof.compatibleInferenceRevision, LIVE_INFERENCE_REVISION);
  assert.deepEqual(
    proof.auditedObjects.map(({ path }) => path).sort(),
    [
      "Cargo.toml",
      "crates/bundles/runtime-cuda",
      "crates/contracts/gen-core",
      "crates/media/candle-gen/candle-gen",
      "crates/media/candle-gen/candle-gen-flux2",
      "crates/media/candle-gen/candle-gen-pid",
      "crates/media/candle-gen/vendor/candle-kernels",
    ].sort(),
  );
  assert.ok(proof.auditedObjects.every(({ capturedObject, compatibleObject }) =>
    /^[0-9a-f]{40}$/.test(capturedObject) && capturedObject === compatibleObject));
  assert.equal(
    matrix.generatedFrom.sources.inferenceCompatibility.path,
    "docs/calibration/sc-15833/inference-compatibility-277f.json",
  );
  assert.match(proof.changedPathDisposition["candle-gen-catalog"], /SenseNova preview registration/);
  assert.match(proof.changedPathDisposition["mlx-flux2"], /outside the Candle\/CUDA dependency closure/);
});

test("SC-15833 compatibility admission rejects a missing or fabricated closure audit", async () => {
  await assert.rejects(
    buildMatrix({ sourceOverrides: { inferenceCompatibility: "" } }),
    /JSON|compatibility audit/,
  );
  const proof = JSON.parse(await readFile(
    new URL("../docs/calibration/sc-15833/inference-compatibility-277f.json", import.meta.url),
  ));
  proof.auditedObjects[0].capturedObject = "a".repeat(40);
  proof.auditedObjects[0].compatibleObject = "a".repeat(40);
  await assert.rejects(
    buildMatrix({ sourceOverrides: { inferenceCompatibility: JSON.stringify(proof) } }),
    /closure is incomplete or unrecognized/,
  );
});

test("SC-15833 refuses to authorize capture promotion against a newer live inference pin", async () => {
  const { capture, reader } = fixture();
  await assert.rejects(
    ingestFlux2CaptureStrict(capture, { reader }),
    new RegExp(`Cargo\\.toml contains a SceneWorks/inference pin other than ${INFERENCE_REVISION}`),
  );
  const replay = await ingestFlux2Capture(capture, { reader });
  assert.equal(replay.records.length, 5, "historical replay still validates the immutable records");
  assert.equal(replay.report.authorization, "historical_replay");
  assert.equal(replay.report.promotionEligible, false);
  const existing = JSON.parse(await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url)));
  const plan = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  assert.throws(() => updateBundle(existing, replay), /historical replay is not promotion-authorized/);
  assert.throws(() => updatePlan(plan, replay), /historical replay is not promotion-authorized/);
});

test("SC-15833 report-only validation can never update promotion artifacts", async () => {
  const reportOnly = fixture({ promotionIntent: "report_only" });
  const reported = await ingestFlux2Capture(reportOnly.capture, { reader: reportOnly.reader });
  assert.equal(reported.report.promotionEligible, false);
  assert.equal(reported.records.length, 0);
  assert.match(reported.report.blockers.join("\n"), /report_only/);

  const existing = JSON.parse(await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url)));
  assert.throws(() => updateBundle(existing, reported), /refusing evidence update/);

  const overclaim = fixture();
  overclaim.capture.promotionIntent = "complete";
  await assert.rejects(
    ingestFlux2Capture(overclaim.capture, { reader: overclaim.reader }),
    /exhaustive complete certification belongs to SC-15922/,
  );
});

test("SC-15833 rejects staged drift and bounded diagnostics above provider maxAbs", async () => {
  const staged = fixture();
  staged.files.set(
    "docs/calibration/sc-15833/base-q4-staged_residency.log",
    baseLog("staged_residency", "2026-08-04T04:01:00Z", { maximumError: 1 }),
  );
  await assert.rejects(
    ingestFlux2Capture(staged.capture, { reader: staged.reader }),
    /staged output is not byte-exact/,
  );

  const bounded = fixture();
  bounded.files.set(
    "docs/calibration/sc-15833/base-q4-bounded_decode.log",
    baseLog("bounded_decode", "2026-08-04T04:02:00Z", { maximumError: 3 }),
  );
  await assert.rejects(
    ingestFlux2Capture(bounded.capture, { reader: bounded.reader }),
    /exceeds maxAbs 2/,
  );

  const stagedDigest = fixture();
  const differentOutput = Buffer.alloc(1024 * 1024 * 3, 18);
  const differentSha = createHash("sha256").update(differentOutput).digest("hex");
  stagedDigest.files.set(
    "docs/calibration/sc-15833/base-q4-staged_residency.log",
    baseLog("staged_residency", "2026-08-04T04:01:00Z", {
      record: { output_sha256: differentSha },
    }),
  );
  stagedDigest.files.set(
    "docs/calibration/sc-15833/base-q4-staged_residency.rgb",
    differentOutput,
  );
  await assert.rejects(
    ingestFlux2Capture(stagedDigest.capture, { reader: stagedDigest.reader }),
    /does not match parity recomputed from the accepted resident RGB8 bytes|not byte-exact/,
  );

  const boundedBytes = fixture();
  const mutatedOutput = Buffer.alloc(1024 * 1024 * 3, 255);
  const mutatedSha = createHash("sha256").update(mutatedOutput).digest("hex");
  boundedBytes.files.set(
    "docs/calibration/sc-15833/base-q4-bounded_decode.log",
    baseLog("bounded_decode", "2026-08-04T04:02:00Z", {
      record: { output_sha256: mutatedSha },
    }),
  );
  boundedBytes.files.set(
    "docs/calibration/sc-15833/base-q4-bounded_decode.rgb",
    mutatedOutput,
  );
  await assert.rejects(
    ingestFlux2Capture(boundedBytes.capture, { reader: boundedBytes.reader }),
    /does not match parity recomputed from the accepted resident RGB8 bytes|recomputed bounded parity exceeds/,
  );
});

test("SC-15833 rejects arbitrary route bytes and PNG diagnostics not derived from pixels", async () => {
  const arbitrary = fixture();
  arbitrary.files.set(
    "docs/calibration/sc-15833/flux2_dev_edit_candle.png",
    Buffer.from("not-a-png"),
  );
  await assert.rejects(
    ingestFlux2Capture(arbitrary.capture, { reader: arbitrary.reader }),
    /is not a PNG image/,
  );

  const mismatched = fixture();
  const output = routePng("edit");
  mismatched.files.set(
    "docs/calibration/sc-15833/edit-q4-resident.log",
    routeLog("edit", "2026-08-04T04:10:00Z", output.stdDev + 1, output.bytes),
  );
  await assert.rejects(
    ingestFlux2Capture(mismatched.capture, { reader: mismatched.reader }),
    /output diagnostic does not match the accepted PNG pixels/,
  );

  const corrupt = fixture();
  const bytes = Buffer.from(corrupt.files.get("docs/calibration/sc-15833/flux2_dev_control_candle.png"));
  bytes[bytes.length - 20] ^= 1;
  corrupt.files.set("docs/calibration/sc-15833/flux2_dev_control_candle.png", bytes);
  await assert.rejects(
    ingestFlux2Capture(corrupt.capture, { reader: corrupt.reader }),
    /invalid .* CRC|invalid compressed image data/,
  );

  const wrongDimensions = fixture();
  const narrow = routePng("edit", 511, 512);
  wrongDimensions.files.set("docs/calibration/sc-15833/flux2_dev_edit_candle.png", narrow.bytes);
  wrongDimensions.files.set(
    "docs/calibration/sc-15833/edit-q4-resident.log",
    routeLog("edit", "2026-08-04T04:10:00Z", narrow.stdDev, narrow.bytes),
  );
  await assert.rejects(
    ingestFlux2Capture(wrongDimensions.capture, { reader: wrongDimensions.reader }),
    /must be a non-interlaced 512x512/,
  );

  const permutedPixels = fixture();
  const permuted = routePng("edit", 512, 512, true);
  permutedPixels.files.set("docs/calibration/sc-15833/flux2_dev_edit_candle.png", permuted.bytes);
  await assert.rejects(
    ingestFlux2Capture(permutedPixels.capture, { reader: permutedPixels.reader }),
    /output digest does not match the accepted PNG bytes/,
  );
});

test("SC-15833 rejects dirty naming, dirty repositories, malformed provenance, and accepted/excluded overlap", async () => {
  const named = fixture();
  const session = named.capture.baseSessions[0];
  const oldPath = session.sourcePath;
  session.sourcePath = "docs/calibration/sc-15833/dirty-experimental-resident.log";
  named.files.set(session.sourcePath, named.files.get(oldPath));
  await assert.rejects(
    ingestFlux2Capture(named.capture, { reader: named.reader }),
    /resident\.sourcePath must equal/,
  );

  const dirty = fixture();
  dirty.capture.repositories.inference.dirty = true;
  await assert.rejects(ingestFlux2Capture(dirty.capture, { reader: dirty.reader }), /must be clean/);

  const zero = fixture();
  zero.capture.repositories.inference.revision = "0".repeat(40);
  await assert.rejects(ingestFlux2Capture(zero.capture, { reader: zero.reader }), /nonzero lowercase Git SHA/);

  const stalePin = fixture();
  stalePin.capture.repositories.inference.revision = "a".repeat(40);
  await assert.rejects(ingestFlux2Capture(stalePin.capture, { reader: stalePin.reader }), /reviewed merge/);

  const arbitraryCommand = fixture();
  arbitraryCommand.capture.baseSessions[0].command += " --arbitrary-passing-test";
  await assert.rejects(
    ingestFlux2Capture(arbitraryCommand.capture, { reader: arbitraryCommand.reader }),
    /exact approved command/,
  );

  const alternateBasePaths = fixture();
  const boundedBase = alternateBasePaths.capture.baseSessions.find(
    (item) => item.rung === "bounded_decode",
  );
  const originalBaseSource = boundedBase.sourcePath;
  const originalBaseOutput = boundedBase.outputPath;
  boundedBase.sourcePath = "docs/calibration/sc-15833/clean-copied-bounded-decode.log";
  boundedBase.outputPath = "docs/calibration/sc-15833/clean-copied-bounded-decode.rgb";
  boundedBase.command = boundedBase.command.replace(originalBaseOutput, boundedBase.outputPath);
  alternateBasePaths.files.set(boundedBase.sourcePath, alternateBasePaths.files.get(originalBaseSource));
  alternateBasePaths.files.set(boundedBase.outputPath, alternateBasePaths.files.get(originalBaseOutput));
  await assert.rejects(
    ingestFlux2Capture(alternateBasePaths.capture, { reader: alternateBasePaths.reader }),
    /bounded_decode\.sourcePath must equal/,
  );

  const alternateRoutePath = fixture();
  const editRoute = alternateRoutePath.capture.routeSessions.find((item) => item.route === "edit");
  const originalRouteSource = editRoute.sourcePath;
  editRoute.sourcePath = "docs/calibration/sc-15833/clean-copied-edit.log";
  alternateRoutePath.files.set(editRoute.sourcePath, alternateRoutePath.files.get(originalRouteSource));
  await assert.rejects(
    ingestFlux2Capture(alternateRoutePath.capture, { reader: alternateRoutePath.reader }),
    /edit\.sourcePath must equal/,
  );

  const mutableModel = fixture();
  mutableModel.capture.baseInput.resolvedRevision = "main";
  await assert.rejects(
    ingestFlux2Capture(mutableModel.capture, { reader: mutableModel.reader }),
    /exact immutable SC-15833 artifact identity/,
  );

  const mutableControl = fixture();
  mutableControl.capture.controlInput.resolvedRevision = "latest";
  await assert.rejects(
    ingestFlux2Capture(mutableControl.capture, { reader: mutableControl.reader }),
    /exact immutable SC-15833 artifact identity/,
  );

  for (const mutate of [
    (capture) => { capture.baseInput.path += "-copy"; },
    (capture) => { capture.baseInput.bytes += 1; },
    (capture) => { capture.controlInput.sha256 = "0".repeat(64); },
    (capture) => { capture.controlInput.repository = "mutable/control"; },
  ]) {
    const artifact = fixture();
    mutate(artifact.capture);
    await assert.rejects(
      ingestFlux2Capture(artifact.capture, { reader: artifact.reader }),
      /exact immutable SC-15833 artifact identity/,
    );
  }

  const wrongAbi = fixture();
  wrongAbi.capture.calibrationAbi += 1;
  await assert.rejects(ingestFlux2Capture(wrongAbi.capture, { reader: wrongAbi.reader }), /provider ABI 3/);

  const vagueFingerprint = fixture();
  vagueFingerprint.capture.calibrationFingerprint = "candidate-full-edge-decode";
  await assert.rejects(
    ingestFlux2Capture(vagueFingerprint.capture, { reader: vagueFingerprint.reader }),
    /exact reviewed FLUX\.2 dev provider fingerprint/,
  );

  const hardware = fixture();
  hardware.capture.hardware.driverVersion = "unexpected-driver";
  await assert.rejects(
    ingestFlux2Capture(hardware.capture, { reader: hardware.reader }),
    /hardware provenance mismatch/,
  );

  const absolute = fixture();
  absolute.capture.baseSessions[0].sourcePath = "C:/temp/resident.log";
  await assert.rejects(
    ingestFlux2Capture(absolute.capture, { reader: absolute.reader }),
    /must stay inside the repository/,
  );

  const overlap = fixture();
  overlap.capture.excludedAttempts[0].sourcePath = overlap.capture.baseSessions[0].sourcePath;
  await assert.rejects(ingestFlux2Capture(overlap.capture, { reader: overlap.reader }), /both accepted and excluded/);
});

test("SC-15833 committed bundle and plan preserve all unrelated evidence structurally", async () => {
  const { capture, reader } = fixture();
  const ingestion = await ingestFlux2Capture(capture, { reader });
  const existing = JSON.parse(await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url)));
  const unrelatedRecords = existing.records.filter((record) =>
    !(record.backend === "candle" && record.target.provider === "flux2_dev"),
  );
  for (const provider of ["qwen_image", "z_image_turbo"]) {
    assert.ok(
      unrelatedRecords.some((record) => record.target.provider === provider),
      `fixture bundle must exercise structural preservation for ${provider}`,
    );
  }
  const unrelatedSessions = (existing.sourceSessions ?? []).filter(
    (session) => !session.sourcePath.startsWith("docs/calibration/sc-15833/"),
  );
  assert.equal(existing.records.filter((record) => record.target.provider === "flux2_dev").length, 5);
  assert.equal(
    existing.sourceSessions.filter((session) => session.sourcePath.startsWith("docs/calibration/sc-15833/")).length,
    7,
  );

  const plan = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const plans = flux2CalibrationPlans(ingestion.records);
  const unrelatedPlanItems = plan.providers.filter((item) => item.target.provider !== "flux2_dev");
  assert.deepEqual(
    plan.providers.filter((item) => item.target.provider !== "flux2_dev"),
    unrelatedPlanItems,
  );
  for (const provider of ["qwen_image", "qwen_image_edit", "z_image", "z_image_turbo"]) {
    assert.ok(
      unrelatedPlanItems.some((item) => item.target.provider === provider),
      `fixture plan must exercise structural preservation for ${provider}`,
    );
  }
  assert.equal(plans.length, 5);
  assert.equal(plan.providers.filter((item) => item.target.provider === "flux2_dev").length, 5);
  assert.deepEqual(plans.map(({ rung }) => rung), RUNGS);
});

// ---------------------------------------------------------------------------------------------
// sc-17497: the compiled-artifact layer.
//
// The v1 audit's unit was a crate tree, so 42 lines of `//!` doc comment in inference `35251a88`
// invalidated a proof it could not possibly have affected. These tests pin the layered replacement:
// object identity still carries the common case for free, and when a path DOES move the record must
// produce a compiled-artifact digest that matches the one frozen in source.
// ---------------------------------------------------------------------------------------------

const V2_METHOD =
  "compiled artifact identity for changed paths, git object identity for unchanged paths, across " +
  "the complete Candle FLUX.2 runtime dependency closure";
const V1_METHOD = "git object identity across the complete Candle FLUX.2 runtime dependency closure";
const ARTIFACT_DIGEST = `sha256:${"c".repeat(64)}`;
const CANDLE_GEN = "crates/media/candle-gen/candle-gen";
const RUNTIME_CUDA = "crates/bundles/runtime-cuda";
// What the `candle-gen-flux2` lib test binary links. `runtime-cuda` is deliberately absent: it
// depends on the provider, not the other way round.
const ADJUDICATES = [
  "crates/contracts/gen-core",
  "crates/media/candle-gen/candle-gen",
  "crates/media/candle-gen/candle-gen-pid",
  "crates/media/candle-gen/candle-gen-flux2",
  "crates/media/candle-gen/vendor/candle-kernels",
];
const PROOF = { digest: ARTIFACT_DIGEST, adjudicates: ADJUDICATES };

async function shippedAudit() {
  return JSON.parse(await readFile(
    new URL("../docs/calibration/sc-15833/inference-compatibility-277f.json", import.meta.url),
  ));
}

/** A v2 record in which `moved` closure paths carry a different compatible object. */
function v2Record(audit, { moved = [], artifact = undefined } = {}) {
  const record = {
    schemaVersion: 2,
    story: "SC-15833",
    capturedInferenceRevision: INFERENCE_REVISION,
    compatibleInferenceRevision: LIVE_INFERENCE_REVISION,
    method: V2_METHOD,
    command: "node scripts/inference-artifact-audit.mjs --repo PATH --captured SHA40 --compatible SHA40",
    changedClosurePaths: [...moved],
    auditedObjects: audit.auditedObjects.map(({ path, capturedObject }) => ({
      path,
      capturedObject,
      compatibleObject: moved.includes(path) ? "e".repeat(40) : capturedObject,
    })),
  };
  if (artifact !== undefined) record.auditedArtifact = artifact;
  return record;
}

function cudaArtifact(overrides = {}) {
  return {
    package: "candle-gen-flux2",
    kind: "lib test binary",
    test: "tests::flux2_dev_probed_generate_for_offload_ab",
    profile: "release",
    lane: "cuda",
    features: ["cuda"],
    rustc: "rustc 1.96.0",
    adjudicates: [...ADJUDICATES],
    capturedDigest: ARTIFACT_DIGEST,
    compatibleDigest: ARTIFACT_DIGEST,
    ...overrides,
  };
}

const accepts = (record, proof) => validatedInferenceCompatibility(JSON.stringify(record), proof);
const rejects = (record, proof, pattern, label) =>
  assert.throws(() => validatedInferenceCompatibility(JSON.stringify(record), proof), pattern, label);

test("SC-17497 the audit script and both validators agree on the strings they all hardcode", async () => {
  // The method string is duplicated across the emitting script, the JS validator and the Rust
  // validator; the closure set across all three. Only the script<->JS pair was pinned, so Rust could
  // drift into refusing records the matrix had already published as `Runtime verified`.
  assert.equal(AUDIT_METHOD, V2_METHOD, "a drift here would reject every record the script emits");
  const rust = await readFile(
    new URL("../crates/sceneworks-worker/src/inference_compatibility_audit.rs", import.meta.url),
    "utf8",
  );
  const rustMethod = /FLUX2_V2_AUDIT_METHOD: &str = concat!\(\s*"([^"]*)",\s*"([^"]*)"/.exec(rust);
  assert.ok(rustMethod, "the Rust v2 method constant must stay machine-readable from this test");
  assert.equal(`${rustMethod[1]}${rustMethod[2]}`, V2_METHOD);
  assert.match(rust, new RegExp(`FLUX2_V1_AUDIT_METHOD: &str =\\s*\n?\\s*"${V1_METHOD}"`));
  assert.deepEqual(
    [...rust.matchAll(/\(\s*"((?:Cargo\.toml|crates\/[^"]+))",\s*\n?\s*"[0-9a-f]{40}"/g)]
      .map(([, objectPath]) => objectPath)
      .sort(),
    [...AUDIT_CLOSURE_PATHS].sort(),
    "the Rust closure set must match the one the script audits",
  );
});

test("SC-17497 the shipped v1 record still validates, and still may not carry a moved object", async () => {
  const audit = await shippedAudit();
  assert.ok(accepts(audit, null));

  const moved = structuredClone(audit);
  moved.auditedObjects.find(({ path }) => path === CANDLE_GEN).compatibleObject = "e".repeat(40);
  rejects(moved, null, /object pair is invalid/, "v1 asserts object identity and nothing else");

  // Both validators require v1's exact `command`; only JS used to.
  rejects({ ...audit, command: "git rev-parse" }, null, /identity is invalid/, "v1 command is pinned");
  rejects({ ...audit, command: 7 }, null, /identity is invalid/, "a non-string command is not a command");
});

test("SC-17497 a v2 record with an unmoved closure needs no build and no artifact block", async () => {
  const audit = await shippedAudit();
  assert.ok(accepts(v2Record(audit), null));
  rejects(
    v2Record(audit),
    PROOF,
    /missing its expected artifact proof/,
    "a proof left frozen after the closure went quiet demands a build that is not due",
  );
});

test("SC-17497 a doc-comment move is authorized by a matching compiled-artifact digest", async () => {
  // The sc-16961 shape exactly: one crate tree moved, identical compiled code.
  const audit = await shippedAudit();
  assert.ok(accepts(v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact() }), PROOF));

  // Mutation check: the digest is the whole proof, so one character must be fatal.
  const off = `sha256:d${"c".repeat(63)}`;
  rejects(
    v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact({ capturedDigest: off, compatibleDigest: off }) }),
    PROOF,
    /compiled-artifact proof is invalid/,
  );
});

test("SC-17497 every way of faking the artifact proof is refused", async () => {
  const audit = await shippedAudit();
  const bad = (label, overrides, pattern = /compiled-artifact proof is invalid/) =>
    rejects(v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact(overrides) }), PROOF, pattern, label);

  bad("a Metal build is not proof of the CUDA artifact the capture ran", { lane: "metal", features: ["metal"] });
  bad("digests that disagree are a re-capture, not an authorization", {
    compatibleDigest: `sha256:${"d".repeat(64)}`,
  });
  bad("a captured digest that does not match the frozen one", { capturedDigest: `sha256:${"d".repeat(64)}` });
  bad("auditing some other crate's binary", { package: "candle-gen" });
  bad("auditing a debug build the capture never used", { profile: "debug" });
  bad("auditing some other test than the one that produced the measurements", {
    test: "tests::flux2_dev_smoke",
  });
  bad("a record whose own adjudicable set is not a list of paths", { adjudicates: "everything" });

  rejects(
    v2Record(audit, { moved: [CANDLE_GEN] }),
    PROOF,
    /compiled-artifact proof is invalid/,
    "a moved path with no artifact block at all",
  );
  rejects(
    { ...v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact() }), changedClosurePaths: [] },
    PROOF,
    /misdeclares which closure paths changed/,
    "understating which paths moved hides a second, unproven change",
  );
  const tamperedCapture = v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact() });
  tamperedCapture.auditedObjects[0].capturedObject = "a".repeat(40);
  rejects(
    tamperedCapture,
    PROOF,
    /closure is incomplete or unrecognized/,
    "a captured object that does not match the code the measurements ran on",
  );
  rejects(
    { ...v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact() }), schemaVersion: 3 },
    PROOF,
    /identity is invalid/,
    "a v3 nobody has defined",
  );
  // v2 does not pin `command` to a literal, so this is the only thing standing between the record
  // and a non-string in the field both languages read.
  rejects(
    { ...v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact() }), command: 7 },
    PROOF,
    /identity is invalid/,
    "a v2 command that is not a string",
  );
  rejects(
    v2Record(audit, { moved: [CANDLE_GEN], artifact: cudaArtifact() }),
    null,
    /compiled-artifact proof is invalid/,
    "the record may not authorize itself: with no proof frozen in source there is no proof",
  );
});

test("SC-17497 a digest cannot speak for a closure path the audited binary never links", async () => {
  // `runtime-cuda` depends on `candle-gen-flux2`, so a commit into the CUDA bundle leaves the
  // measurement binary byte-identical. Reading that unchanged digest as proof would be a false
  // green — strictly worse than the false positive this story removes.
  const audit = await shippedAudit();
  rejects(
    v2Record(audit, { moved: [RUNTIME_CUDA], artifact: cudaArtifact() }),
    PROOF,
    /cannot adjudicate crates\/bundles\/runtime-cuda/,
  );
  rejects(
    v2Record(audit, { moved: [CANDLE_GEN, RUNTIME_CUDA], artifact: cudaArtifact() }),
    PROOF,
    /cannot adjudicate crates\/bundles\/runtime-cuda/,
    "riding along with an adjudicable path does not launder it",
  );

  // The frozen set is a HUMAN TRANSCRIPTION, and unlike the digest an over-wide one fails OPEN.
  // Intersecting it with the record's own set means both halves must be wrong the same way.
  rejects(
    v2Record(audit, { moved: [RUNTIME_CUDA], artifact: cudaArtifact() }),
    { digest: ARTIFACT_DIGEST, adjudicates: [...ADJUDICATES, RUNTIME_CUDA] },
    /cannot adjudicate crates\/bundles\/runtime-cuda/,
    "an over-wide frozen set is still checked against what the build reported linking",
  );
  assert.ok(
    accepts(
      v2Record(audit, {
        moved: [RUNTIME_CUDA],
        artifact: cudaArtifact({ adjudicates: [...ADJUDICATES, RUNTIME_CUDA] }),
      }),
      { digest: ARTIFACT_DIGEST, adjudicates: [...ADJUDICATES, RUNTIME_CUDA] },
    ),
    "an artifact that DOES link it may adjudicate it — the rule is coverage, not a blocklist",
  );
});
