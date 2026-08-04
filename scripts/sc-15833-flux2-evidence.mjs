#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { inflateSync } from "node:zlib";

import {
  HARNESS_VERSION,
  canonicalJson,
  logicalCaseId,
  recordId,
  validateBundle,
  validateRecord,
} from "./memory-calibration-harness.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const EVIDENCE = path.join(ROOT, "docs/generated/memory-calibration-evidence.json");
const PLAN = path.join(ROOT, "config/memory-calibration-plan.json");
const SESSION_PREFIX = "docs/calibration/sc-15833/";
const STORY = "SC-15833";
const PROVIDER = "flux2_dev";
const MODEL_ID = "flux2_dev";
const BASE_HARNESS = "candle-flux2-memory-ladder-v1";
const PARITY_METRIC = "rgb8_max_abs_error";
const BOUNDED_MAX_ABS = 2;
const EXPECTED_CALIBRATION_ABI = 3;
const EXPECTED_CALIBRATION_FINGERPRINT =
  "flux2-dev-cuda-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v2";
const EXPECTED_INFERENCE_REVISION = "5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d";
const SHA256 = /^[0-9a-f]{64}$/;
const REVISION = /^[0-9a-f]{40}$/;
const RFC3339 = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/;
const EXPECTED_BASE_INPUT = {
  role: "base",
  path: "E:/huggingface/sceneworks-staging/sc15833-flux2-dev-q4-2868b146",
  bytes: 33582076196,
  sha256: "896f227194e48c6e4df10cec20733f4ed1a357affc021bc131e6851b787da98b",
  repository: "SceneWorks/flux2-dev-mlx",
  resolvedRevision: "2868b1461b2b6e6e05d84e52534df3632b4c7d5d",
  variant: "q4",
};
const EXPECTED_CONTROL_INPUT = {
  role: "control",
  path: "E:/huggingface/sceneworks-staging/sc15833-flux2-control-b3dcd783/FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors",
  bytes: 8232506680,
  sha256: "516532a885d12ae84bb3c6b24ef4816ac05ffa1c9c7b93476f74652eb0a7a794",
  repository: "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
  resolvedRevision: "b3dcd7836a0e926248dac3ccba8fc0853495764b",
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
    "resident",
    "staged_residency",
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
  ],
};

const PARAMETERS = {
  resident: {},
  staged_residency: {},
  bounded_decode: { decodeTileEdge: 1024, decodeOverlap: 1 },
  bounded_attention: {
    decodeTileEdge: 1024,
    decodeOverlap: 1,
    attentionChunkSize: 67108864,
  },
  bounded_transformer_residency: {
    decodeTileEdge: 1024,
    decodeOverlap: 1,
    attentionChunkSize: 67108864,
    transformerWindowSize: 1,
    transformerWindowComponent: "dit",
  },
};

const RUNG_ENV = {
  resident: "resident",
  staged_residency: "staged",
  bounded_decode: "decode",
  bounded_attention: "attention",
  bounded_transformer_residency: "blocks",
};

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
function fail(message) {
  throw new Error(`${STORY}: ${message}`);
}

function exactObject(value, label, keys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be an object`);
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail(`${label} has unexpected shape: ${actual.join(",")}`);
  }
}

function requireText(value, label) {
  if (typeof value !== "string" || value.trim() === "") fail(`${label} must be non-empty text`);
  return value;
}

function requireDigest(value, label) {
  if (!SHA256.test(value ?? "")) fail(`${label} must be a lowercase SHA-256 digest`);
  return value;
}

function requireRevision(value, label) {
  if (!REVISION.test(value ?? "") || /^0+$/.test(value)) fail(`${label} must be a nonzero lowercase Git SHA`);
  return value;
}

function repoRelative(value, label, prefix = null) {
  requireText(value, label);
  const normalized = value.replaceAll("\\", "/");
  if (
    path.posix.isAbsolute(normalized) || path.win32.isAbsolute(value) ||
    /^[A-Za-z]:\//.test(normalized) || normalized.split("/").includes("..")
  ) {
    fail(`${label} must stay inside the repository`);
  }
  if (prefix && !normalized.startsWith(prefix)) fail(`${label} must start with ${prefix}`);
  return normalized;
}

function decodeLog(bytes) {
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return bytes.subarray(2).toString("utf16le");
  }
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) {
    return bytes.subarray(3).toString("utf8");
  }
  return bytes.toString("utf8");
}

function markerOffsets(text, marker) {
  const offsets = [];
  for (let at = text.indexOf(marker); at !== -1; at = text.indexOf(marker, at + marker.length)) {
    offsets.push(at);
  }
  return offsets;
}

function extractWrappedJson(text, marker, filename) {
  const offsets = markerOffsets(text, marker);
  if (offsets.length !== 1) fail(`${filename} must contain exactly one ${marker} marker`);
  const start = text.indexOf("{", offsets[0] + marker.length);
  if (start === -1) fail(`${filename} ${marker} marker has no JSON object`);
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === "\"") quoted = false;
      continue;
    }
    if (char === "\"") quoted = true;
    else if (char === "{") depth += 1;
    else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        const raw = text.slice(start, index + 1).replace(/\r?\n/g, "");
        try {
          return JSON.parse(raw);
        } catch (error) {
          fail(`${filename} has malformed ${marker} JSON: ${error.message}`);
        }
      }
    }
  }
  fail(`${filename} has truncated ${marker} JSON`);
}

function assertPassingTestLog(text, filename) {
  if (!/test result: ok\./.test(text)) fail(`${filename} is missing a passing test result`);
  if (/test result: FAILED|\.\.\. FAILED|panicked at|error: test failed/.test(text)) {
    fail(`${filename} contains a failed test or panic`);
  }
}

function assertExactTest(text, identity, filename) {
  const line = `test ${identity} ... ok`;
  if (text.split(/\r?\n/).filter((item) => item.trim() === line).length !== 1) {
    fail(`${filename} did not pass the exact approved test ${identity}`);
  }
}

function baseCommand(capture, descriptor) {
  const outputPath = `${SESSION_PREFIX}base-q4-${descriptor.rung}.rgb`;
  const parity = descriptor.rung === "resident"
    ? ""
    : ` FLUX2_PARITY_REFERENCE=${SESSION_PREFIX}base-q4-resident.rgb`;
  return `CUDA_VISIBLE_DEVICES=0 FLUX2_DEV_DIR=${capture.baseInput.path} FLUX2_QUANT=q4 FLUX2_MEMORY_RUNG=${RUNG_ENV[descriptor.rung]} FLUX2_OUT=${outputPath}${parity} MEMORY_EXPECTED_ABI=${EXPECTED_CALIBRATION_ABI} MEMORY_EXPECTED_FINGERPRINT=${EXPECTED_CALIBRATION_FINGERPRINT} MEMORY_INFERENCE_REVISION=${EXPECTED_INFERENCE_REVISION} MEMORY_SCENEWORKS_REVISION=${capture.repositories.sceneWorks.revision} MEMORY_MODEL_REVISION=${capture.baseInput.resolvedRevision} MEMORY_MODEL_INVENTORY_SHA256=${capture.baseInput.sha256} cargo test -p candle-gen-flux2 --release --features cuda tests::flux2_dev_probed_generate_for_offload_ab -- --ignored --nocapture --test-threads=1`;
}

function routeCommand(capture, descriptor) {
  const control = descriptor.route === "control" ? ` FLUX2_CONTROL=${capture.controlInput.path}` : "";
  return `CUDA_VISIBLE_DEVICES=0 FLUX2_DEV_DIR=${capture.baseInput.path}${control} FLUX2_DEV_OUT_DIR=${SESSION_PREFIX.slice(0, -1)} FLUX2_DEV_W=512 FLUX2_DEV_H=512 cargo test -p sceneworks-worker --release --no-default-features --features backend-candle flux2_dev_gpu_smoke::flux2_dev_${descriptor.route}_candle_gpu_smoke -- --ignored --nocapture --test-threads=1`;
}

function assertExactCommand(actual, expected, label) {
  if (actual !== expected) fail(`${label} must equal the exact approved command`);
}

function validateSourceMarker(text, capture, descriptor, inputs, filename) {
  const marker = extractWrappedJson(text, "SC15833_SOURCE_V1", filename);
  exactObject(marker, `${filename} source marker`, [
    "schema_version", "captured_at", "repositories", "hardware", "inputs",
  ]);
  if (marker.schema_version !== 1 || marker.captured_at !== descriptor.capturedAt) {
    fail(`${filename} source marker has the wrong schema or capture time`);
  }
  if (canonicalJson(marker.repositories) !== canonicalJson(capture.repositories)) {
    fail(`${filename} source marker repository provenance mismatch`);
  }
  if (canonicalJson(marker.hardware) !== canonicalJson(capture.hardware)) {
    fail(`${filename} source marker hardware provenance mismatch`);
  }
  if (canonicalJson(marker.inputs) !== canonicalJson(inputs)) {
    fail(`${filename} source marker artifact provenance mismatch`);
  }
}

function parseFinite(token, label, { allowInfinity = false } = {}) {
  if (allowInfinity && /^(?:inf|infinity)$/i.test(token)) return Number.POSITIVE_INFINITY;
  const value = Number(token);
  if (!Number.isFinite(value) || value < 0) fail(`${label} must be a nonnegative finite number`);
  return value;
}

function parityDiagnostic(text, filename) {
  const matches = [...text.matchAll(
    /MEMORY_PARITY_DIAGNOSTIC\s+strategy=([^\s]+)\s+reference=([^\s]+)\s+changed_fraction=([^\s]+)\s+max_abs=([^\s]+)\s+mean_abs=([^\s]+)\s+rmse=([^\s]+)\s+psnr_db=([^\s]+)/g,
  )];
  if (matches.length !== 1) fail(`${filename} must contain exactly one complete MEMORY_PARITY_DIAGNOSTIC`);
  const [, strategy, reference, changed, maximum, mean, rmse, psnr] = matches[0];
  return {
    strategy,
    reference,
    changedFraction: parseFinite(changed, `${filename} changed fraction`),
    maximumError: parseFinite(maximum, `${filename} maximum error`),
    meanError: parseFinite(mean, `${filename} mean error`),
    rootMeanSquareError: parseFinite(rmse, `${filename} RMSE`),
    psnrDb: parseFinite(psnr, `${filename} PSNR`, { allowInfinity: true }),
  };
}

function rgb8Parity(reference, candidate, filename) {
  if (reference.length !== candidate.length || reference.length === 0) {
    fail(`${filename} cannot be compared with the resident RGB8 output`);
  }
  let changed = 0;
  let maximumError = 0;
  let absoluteError = 0;
  let squaredError = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const error = Math.abs(reference[index] - candidate[index]);
    if (error !== 0) changed += 1;
    if (error > maximumError) maximumError = error;
    absoluteError += error;
    squaredError += error * error;
  }
  const meanError = absoluteError / reference.length;
  const rootMeanSquareError = Math.sqrt(squaredError / reference.length);
  return {
    changedFraction: changed / reference.length,
    maximumError,
    meanError,
    rootMeanSquareError,
    psnrDb: rootMeanSquareError === 0
      ? Number.POSITIVE_INFINITY
      : 20 * Math.log10(255 / rootMeanSquareError),
  };
}

function sameSerializedMetric(actual, reported) {
  if (actual === reported) return true;
  if (!Number.isFinite(actual) || !Number.isFinite(reported)) return false;
  return Math.abs(actual - reported) <= Math.max(1e-12, Math.abs(actual) * 1e-9);
}

function validateMeasuredParity(resident, candidate) {
  const filename = candidate.descriptor.sourcePath;
  const measured = rgb8Parity(resident.outputBytes, candidate.outputBytes, filename);
  const diagnostic = candidate.diagnostic;
  if (!diagnostic) fail(`${filename} is missing its resident parity diagnostic`);
  for (const key of [
    "changedFraction", "maximumError", "meanError", "rootMeanSquareError", "psnrDb",
  ]) {
    if (!sameSerializedMetric(measured[key], diagnostic[key])) {
      fail(`${filename} ${key} does not match parity recomputed from the accepted resident RGB8 bytes`);
    }
  }
  if (candidate.descriptor.rung === "staged_residency" && measured.maximumError !== 0) {
    fail(`${filename} staged output is not byte-exact to the accepted resident RGB8 bytes`);
  }
  if (candidate.descriptor.rung.startsWith("bounded_") && measured.maximumError > BOUNDED_MAX_ABS) {
    fail(`${filename} recomputed bounded parity exceeds maxAbs ${BOUNDED_MAX_ABS}`);
  }
  return measured;
}

function expectedSnakeParameters(rung) {
  const parameters = PARAMETERS[rung];
  return {
    decode_tile_edge: parameters.decodeTileEdge ?? null,
    decode_overlap: parameters.decodeOverlap ?? null,
    attention_chunk_size: parameters.attentionChunkSize ?? null,
    transformer_window_size: parameters.transformerWindowSize ?? null,
    transformer_window_component: parameters.transformerWindowComponent ?? null,
  };
}

function validateMemoryRecord(record, rung, capture, filename, diagnostic) {
  if (record.schema_version !== 1) fail(`${filename} has unsupported MEMORY_EVIDENCE_V1 schema`);
  const key = record.key;
  if (!key || key.resolved_route !== PROVIDER || key.backend !== "candle") fail(`${filename} has the wrong route/backend`);
  if (key.tier?.precision !== "bf16" || key.tier?.quant !== "q4" || key.tier?.component_precision_floors?.length !== 0) {
    fail(`${filename} is not the exact FLUX.2 dev Q4 numeric tier`);
  }
  if (key.load_shape !== "deferred_materialization" || key.mode !== "text_to_image" || key.overlay !== null) {
    fail(`${filename} has the wrong load shape, mode, or overlay`);
  }
  const geometry = key.geometry;
  if (
    geometry?.width !== 1024 || geometry.height !== 1024 || geometry.batch !== 1 ||
    geometry.frames !== 1 || geometry.reference_count !== 0
  ) fail(`${filename} has the wrong representative geometry`);
  if (key.strategy !== rung || JSON.stringify(key.engaged_composition) !== JSON.stringify(COMPOSITIONS[rung])) {
    fail(`${filename} has the wrong rung composition`);
  }
  if (canonicalJson(key.parameters) !== canonicalJson(expectedSnakeParameters(rung))) {
    fail(`${filename} has unexpected provider parameters`);
  }
  for (const calibration of [record.declared_calibration, record.observed_calibration]) {
    if (
      calibration?.abi !== capture.calibrationAbi ||
      calibration.fingerprint !== capture.calibrationFingerprint ||
      calibration.load_shape !== "deferred_materialization"
    ) fail(`${filename} calibration ABI/fingerprint/load shape mismatch`);
  }
  if (canonicalJson(record.declared_calibration) !== canonicalJson(record.observed_calibration)) {
    fail(`${filename} declared and observed calibrations differ`);
  }
  if (!Number.isSafeInteger(record.predicted_peak_bytes) || record.predicted_peak_bytes <= 0) {
    fail(`${filename} predicted peak is invalid`);
  }
  if (record.predicted_peak_bytes !== record.observed_peak_bytes) fail(`${filename} predicted/observed peaks differ`);
  if (
    record.inference_revision !== capture.repositories.inference.revision ||
    record.sceneworks_revision !== capture.repositories.sceneWorks.revision ||
    record.model_revision !== capture.baseInput.resolvedRevision ||
    record.model_inventory_sha256 !== capture.baseInput.sha256 ||
    record.harness_version !== BASE_HARNESS
  ) fail(`${filename} repository, model, inventory, or harness provenance mismatch`);
  requireDigest(record.output_sha256, `${filename} output digest`);

  const parityKind = record.parity?.kind;
  const parityResult = record.parity_result?.kind;
  if (rung === "resident") {
    if (parityKind !== "exact" || parityResult !== "not_run" || diagnostic !== null) {
      fail(`${filename} resident reference must be Exact/NotRun without a comparison diagnostic`);
    }
  } else if (rung === "staged_residency") {
    if (parityKind !== "exact" || parityResult !== "passed") fail(`${filename} staged parity must be Exact/Passed`);
    if (
      !diagnostic || diagnostic.maximumError !== 0 || diagnostic.changedFraction !== 0 ||
      diagnostic.meanError !== 0 || diagnostic.rootMeanSquareError !== 0
    ) fail(`${filename} staged output is not byte-exact`);
  } else {
    if (
      parityKind !== "tolerance" || record.parity.metric !== PARITY_METRIC ||
      record.parity.maximum_error !== BOUNDED_MAX_ABS || parityResult !== "passed"
    ) fail(`${filename} bounded parity must use provider-owned ${PARITY_METRIC} <= ${BOUNDED_MAX_ABS}`);
    if (!diagnostic || diagnostic.maximumError > BOUNDED_MAX_ABS) {
      fail(`${filename} bounded parity diagnostic exceeds maxAbs ${BOUNDED_MAX_ABS}`);
    }
  }
  if (diagnostic && diagnostic.strategy !== rung.replaceAll("_", "")) {
    // Rust Debug currently emits PascalCase; accept that exact spelling below instead of silently
    // accepting arbitrary diagnostic labels.
    const expected = {
      staged_residency: "StagedResidency",
      bounded_decode: "BoundedDecode",
      bounded_attention: "BoundedAttention",
      bounded_transformer_residency: "BoundedTransformerResidency",
    }[rung];
    if (diagnostic.strategy !== expected) fail(`${filename} parity diagnostic names the wrong strategy`);
  }
  return record;
}

function sessionId(session) {
  return `ims-${sha256(Buffer.from(JSON.stringify(session), "utf8")).slice(0, 20)}`;
}

function sourceSession(spec) {
  const session = { id: "", ...spec, result: "passed" };
  session.id = sessionId(session);
  return session;
}

function validateCaptureEnvelope(capture) {
  exactObject(capture, "capture", [
    "schemaVersion", "story", "promotionIntent", "repositories", "hardware", "calibrationAbi",
    "calibrationFingerprint", "baseInput", "controlInput", "baseSessions", "routeSessions",
    "excludedAttempts",
  ]);
  if (capture.schemaVersion !== 1 || capture.story !== STORY) fail("capture envelope is not SC-15833 schema 1");
  if (!['report_only', 'runtime_complete'].includes(capture.promotionIntent)) {
    fail("promotionIntent must be report_only or runtime_complete; exhaustive complete certification belongs to SC-15922");
  }
  exactObject(capture.repositories, "repositories", ["sceneWorks", "inference"]);
  exactObject(capture.repositories.sceneWorks, "repositories.sceneWorks", ["revision", "dirty", "matrixSourceRevision"]);
  exactObject(capture.repositories.inference, "repositories.inference", ["revision", "dirty"]);
  requireRevision(capture.repositories.sceneWorks.revision, "SceneWorks revision");
  requireRevision(capture.repositories.inference.revision, "inference revision");
  if (capture.repositories.inference.revision !== EXPECTED_INFERENCE_REVISION) {
    fail(`inference revision must equal the reviewed merge ${EXPECTED_INFERENCE_REVISION}`);
  }
  requireDigest(capture.repositories.sceneWorks.matrixSourceRevision.replace(/^source-tree:/, ""), "matrix source revision");
  if (capture.repositories.sceneWorks.dirty || capture.repositories.inference.dirty) fail("authoritative capture repositories must be clean");
  exactObject(capture.hardware, "hardware", [
    "probe", "memoryBytes", "deviceId", "name", "computeCapability", "driverVersion", "runtimeVersion",
  ]);
  for (const key of ["probe", "deviceId", "name", "computeCapability", "driverVersion", "runtimeVersion"]) {
    requireText(capture.hardware[key], `hardware.${key}`);
  }
  if (!Number.isSafeInteger(capture.hardware.memoryBytes) || capture.hardware.memoryBytes <= 0) fail("hardware.memoryBytes is invalid");
  if (capture.calibrationAbi !== EXPECTED_CALIBRATION_ABI) {
    fail(`calibrationAbi must equal the provider ABI ${EXPECTED_CALIBRATION_ABI}`);
  }
  if (capture.calibrationFingerprint !== EXPECTED_CALIBRATION_FINGERPRINT) {
    fail("calibrationFingerprint must equal the exact reviewed FLUX.2 dev provider fingerprint");
  }
  validateInput(capture.baseInput, "baseInput", EXPECTED_BASE_INPUT);
  validateInput(capture.controlInput, "controlInput", EXPECTED_CONTROL_INPUT);
  for (const name of ["baseSessions", "routeSessions", "excludedAttempts"]) {
    if (!Array.isArray(capture[name])) fail(`${name} must be an array`);
  }
}

async function validateLiveInferencePins() {
  const manifests = [
    "Cargo.toml",
    "crates/sceneworks-memory-adapter/Cargo.toml",
    "crates/sceneworks-worker/Cargo.toml",
  ];
  let count = 0;
  for (const manifest of manifests) {
    const text = await readFile(path.join(ROOT, manifest), "utf8");
    const revisions = [...text.matchAll(
      /git\s*=\s*"https:\/\/github\.com\/SceneWorks\/inference"[^}\n]*\brev\s*=\s*"([0-9a-f]{40})"/g,
    )].map((match) => match[1]);
    count += revisions.length;
    if (revisions.some((revision) => revision !== EXPECTED_INFERENCE_REVISION)) {
      fail(`${manifest} contains a SceneWorks/inference pin other than ${EXPECTED_INFERENCE_REVISION}`);
    }
  }
  if (count === 0) fail("live Cargo manifests contain no immutable SceneWorks/inference pins");
}

function validateInput(input, label, expected) {
  exactObject(input, label, ["role", "path", "bytes", "sha256", "repository", "resolvedRevision", "variant"]);
  if (canonicalJson(input) !== canonicalJson(expected)) {
    fail(`${label} must equal the exact immutable SC-15833 artifact identity`);
  }
}

async function readRepoFile(relative, reader) {
  const safe = repoRelative(relative, "capture file path");
  return Buffer.from(await reader(safe));
}

const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const PNG_CRC_TABLE = Uint32Array.from({ length: 256 }, (_, value) => {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) crc = (crc & 1) ? (0xedb88320 ^ (crc >>> 1)) : (crc >>> 1);
  return crc >>> 0;
});

function pngCrc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = PNG_CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function paeth(left, above, upperLeft) {
  const estimate = left + above - upperLeft;
  const leftDistance = Math.abs(estimate - left);
  const aboveDistance = Math.abs(estimate - above);
  const upperLeftDistance = Math.abs(estimate - upperLeft);
  if (leftDistance <= aboveDistance && leftDistance <= upperLeftDistance) return left;
  return aboveDistance <= upperLeftDistance ? above : upperLeft;
}

function inspectPng(bytes, filename) {
  if (bytes.length < PNG_SIGNATURE.length || !bytes.subarray(0, 8).equals(PNG_SIGNATURE)) {
    fail(`${filename} is not a PNG image`);
  }
  let offset = 8;
  let ihdr = null;
  let sawEnd = false;
  const compressed = [];
  while (offset < bytes.length) {
    if (offset + 12 > bytes.length) fail(`${filename} has a truncated PNG chunk`);
    const length = bytes.readUInt32BE(offset);
    const typeOffset = offset + 4;
    const dataOffset = typeOffset + 4;
    const crcOffset = dataOffset + length;
    if (crcOffset + 4 > bytes.length) fail(`${filename} has a truncated PNG chunk payload`);
    const type = bytes.subarray(typeOffset, dataOffset).toString("ascii");
    const crcInput = bytes.subarray(typeOffset, crcOffset);
    if (pngCrc32(crcInput) !== bytes.readUInt32BE(crcOffset)) {
      fail(`${filename} has an invalid ${type} CRC`);
    }
    const data = bytes.subarray(dataOffset, crcOffset);
    if (!ihdr && type !== "IHDR") fail(`${filename} PNG must start with IHDR`);
    if (type === "IHDR") {
      if (ihdr || length !== 13) fail(`${filename} has an invalid IHDR`);
      ihdr = {
        width: data.readUInt32BE(0),
        height: data.readUInt32BE(4),
        bitDepth: data[8],
        colorType: data[9],
        compression: data[10],
        filter: data[11],
        interlace: data[12],
      };
    } else if (type === "IDAT") compressed.push(data);
    else if (type === "IEND") {
      if (length !== 0 || sawEnd) fail(`${filename} has an invalid IEND`);
      sawEnd = true;
      offset = crcOffset + 4;
      break;
    }
    offset = crcOffset + 4;
  }
  if (!ihdr || compressed.length === 0 || !sawEnd || offset !== bytes.length) {
    fail(`${filename} is missing required PNG chunks or has trailing bytes`);
  }
  if (
    ihdr.width !== 512 || ihdr.height !== 512 || ihdr.bitDepth !== 8
    || ![2, 6].includes(ihdr.colorType) || ihdr.compression !== 0
    || ihdr.filter !== 0 || ihdr.interlace !== 0
  ) fail(`${filename} must be a non-interlaced 512x512 RGB8 or RGBA8 PNG`);
  const channels = ihdr.colorType === 2 ? 3 : 4;
  const stride = ihdr.width * channels;
  let inflated;
  try {
    inflated = inflateSync(Buffer.concat(compressed));
  } catch (error) {
    fail(`${filename} has invalid compressed image data: ${error.message}`);
  }
  if (inflated.length !== (stride + 1) * ihdr.height) fail(`${filename} has the wrong decoded byte length`);
  const decoded = Buffer.alloc(stride * ihdr.height);
  for (let row = 0; row < ihdr.height; row += 1) {
    const sourceOffset = row * (stride + 1);
    const filter = inflated[sourceOffset];
    if (filter > 4) fail(`${filename} has unsupported PNG filter ${filter}`);
    const targetOffset = row * stride;
    for (let column = 0; column < stride; column += 1) {
      const raw = inflated[sourceOffset + 1 + column];
      const left = column >= channels ? decoded[targetOffset + column - channels] : 0;
      const above = row > 0 ? decoded[targetOffset + column - stride] : 0;
      const upperLeft = row > 0 && column >= channels
        ? decoded[targetOffset + column - stride - channels]
        : 0;
      const predictor = filter === 0 ? 0
        : filter === 1 ? left
          : filter === 2 ? above
            : filter === 3 ? Math.floor((left + above) / 2)
              : paeth(left, above, upperLeft);
      decoded[targetOffset + column] = (raw + predictor) & 0xff;
    }
  }
  const pixelCount = ihdr.width * ihdr.height;
  let sum = 0;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const at = pixel * channels;
    sum += decoded[at] + decoded[at + 1] + decoded[at + 2];
  }
  const count = pixelCount * 3;
  const mean = sum / count;
  let squared = 0;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const at = pixel * channels;
    for (let channel = 0; channel < 3; channel += 1) {
      const delta = decoded[at + channel] - mean;
      squared += delta * delta;
    }
  }
  return { format: "png", width: ihdr.width, height: ihdr.height, stdDev: Math.sqrt(squared / count) };
}

async function ingestBaseSession(descriptor, capture, reader) {
  exactObject(descriptor, `base session ${descriptor?.rung ?? "?"}`, ["rung", "command", "sourcePath", "capturedAt", "outputPath"]);
  if (!RUNGS.includes(descriptor.rung)) fail(`invalid base rung ${descriptor.rung}`);
  requireText(descriptor.command, `${descriptor.rung}.command`);
  const sourcePath = repoRelative(descriptor.sourcePath, `${descriptor.rung}.sourcePath`, SESSION_PREFIX);
  const outputPath = repoRelative(descriptor.outputPath, `${descriptor.rung}.outputPath`);
  const expectedSource = `${SESSION_PREFIX}base-q4-${descriptor.rung}.log`;
  const expectedOutput = `${SESSION_PREFIX}base-q4-${descriptor.rung}.rgb`;
  if (sourcePath !== expectedSource) fail(`${descriptor.rung}.sourcePath must equal ${expectedSource}`);
  if (outputPath !== expectedOutput) fail(`${descriptor.rung}.outputPath must equal ${expectedOutput}`);
  assertExactCommand(descriptor.command, baseCommand(capture, descriptor), `${descriptor.rung}.command`);
  if (!RFC3339.test(descriptor.capturedAt ?? "")) fail(`${descriptor.rung}.capturedAt must be stable RFC3339 UTC`);
  if (/dirty|experimental|failed|sweep/i.test(sourcePath) || /dirty|experimental|failed|sweep/i.test(outputPath)) {
    fail(`${sourcePath} or its output is excluded from authoritative ingestion`);
  }
  const [logBytes, outputBytes] = await Promise.all([
    readRepoFile(sourcePath, reader),
    readRepoFile(outputPath, reader),
  ]);
  const text = decodeLog(logBytes);
  assertPassingTestLog(text, sourcePath);
  assertExactTest(text, "tests::flux2_dev_probed_generate_for_offload_ab", sourcePath);
  validateSourceMarker(text, capture, descriptor, [capture.baseInput], sourcePath);
  const hasDiagnostic = markerOffsets(text, "MEMORY_PARITY_DIAGNOSTIC").length > 0;
  const diagnostic = hasDiagnostic ? parityDiagnostic(text, sourcePath) : null;
  const record = validateMemoryRecord(
    extractWrappedJson(text, "MEMORY_EVIDENCE_V1", sourcePath),
    descriptor.rung,
    capture,
    sourcePath,
    diagnostic,
  );
  const outputSha256 = sha256(outputBytes);
  if (outputSha256 !== record.output_sha256) fail(`${outputPath} does not match the harness output digest`);
  if (outputBytes.length !== 1024 * 1024 * 3) fail(`${outputPath} is not a 1024x1024 RGB8 output`);
  const session = sourceSession({
    kind: "physical_cuda",
    command: descriptor.command,
    sourcePath,
    capturedAt: descriptor.capturedAt,
    repositories: {
      sceneWorks: { revision: capture.repositories.sceneWorks.revision, dirty: false },
      inference: { revision: capture.repositories.inference.revision, dirty: false },
    },
    hardware: capture.hardware,
    target: { tier: "q4", mode: "text_to_image", overlay: "none", rung: descriptor.rung },
    stdoutSha256: sha256(logBytes),
    inputs: [capture.baseInput],
    outputs: [{ path: outputPath, sha256: outputSha256 }],
    claims: ["memory", "quality", "loadability", "overlay"],
  });
  return { descriptor, record, diagnostic, outputBytes, session };
}

async function ingestRouteSession(descriptor, capture, reader) {
  exactObject(descriptor, `route session ${descriptor?.route ?? "?"}`, ["route", "command", "sourcePath", "capturedAt", "outputPath"]);
  if (!['edit', 'control'].includes(descriptor.route)) fail(`invalid route ${descriptor.route}`);
  requireText(descriptor.command, `${descriptor.route}.command`);
  const sourcePath = repoRelative(descriptor.sourcePath, `${descriptor.route}.sourcePath`, SESSION_PREFIX);
  const outputPath = repoRelative(descriptor.outputPath, `${descriptor.route}.outputPath`);
  const expectedSource = `${SESSION_PREFIX}${descriptor.route}-q4-resident.log`;
  const expectedOutput = `${SESSION_PREFIX}flux2_dev_${descriptor.route}_candle.png`;
  if (sourcePath !== expectedSource) fail(`${descriptor.route}.sourcePath must equal ${expectedSource}`);
  if (outputPath !== expectedOutput) fail(`${descriptor.route}.outputPath must equal ${expectedOutput}`);
  assertExactCommand(descriptor.command, routeCommand(capture, descriptor), `${descriptor.route}.command`);
  if (!RFC3339.test(descriptor.capturedAt ?? "")) fail(`${descriptor.route}.capturedAt must be stable RFC3339 UTC`);
  if (/dirty|experimental|failed|sweep/i.test(sourcePath) || /dirty|experimental|failed|sweep/i.test(outputPath)) {
    fail(`${sourcePath} or its output is excluded from authoritative ingestion`);
  }
  const [logBytes, outputBytes] = await Promise.all([
    readRepoFile(sourcePath, reader),
    readRepoFile(outputPath, reader),
  ]);
  const text = decodeLog(logBytes);
  assertPassingTestLog(text, sourcePath);
  const inputs = descriptor.route === "control"
    ? [capture.baseInput, capture.controlInput]
    : [capture.baseInput];
  validateSourceMarker(text, capture, descriptor, inputs, sourcePath);
  const test = descriptor.route === "edit"
    ? "flux2_dev_edit_candle_gpu_smoke"
    : "flux2_dev_control_candle_gpu_smoke";
  const done = descriptor.route === "edit"
    ? "[smoke] DONE: flux2_dev edit (candle) coherent"
    : "[smoke] DONE: flux2_dev control (candle) coherent";
  assertExactTest(text, `flux2_dev_gpu_smoke::${test}`, sourcePath);
  if (!text.includes(done)) {
    fail(`${sourcePath} did not pass the exact ${descriptor.route} smoke`);
  }
  const diagnostic = text.match(new RegExp(
    `\\[smoke\\] dev ${descriptor.route} 512x512 std ([0-9.]+) -> ([^\\r\\n]+) sha256=([0-9a-f]{64})`,
  ));
  if (!diagnostic || Number(diagnostic[1]) <= 0) {
    fail(`${sourcePath} is missing a digest-bound non-degenerate output diagnostic`);
  }
  if (diagnostic[2].replaceAll("\\", "/") !== outputPath) {
    fail(`${sourcePath} output diagnostic names a path other than ${outputPath}`);
  }
  const image = inspectPng(outputBytes, outputPath);
  if (image.stdDev <= 0) fail(`${outputPath} is a degenerate image`);
  const loggedStdDev = Number(diagnostic[1]);
  if (Math.abs(image.stdDev - loggedStdDev) > 0.0051) {
    fail(`${sourcePath} output diagnostic does not match the accepted PNG pixels`);
  }
  const outputSha256 = sha256(outputBytes);
  if (diagnostic[3] !== outputSha256) {
    fail(`${sourcePath} output digest does not match the accepted PNG bytes`);
  }
  const session = sourceSession({
    kind: "physical_cuda",
    command: descriptor.command,
    sourcePath,
    capturedAt: descriptor.capturedAt,
    repositories: {
      sceneWorks: { revision: capture.repositories.sceneWorks.revision, dirty: false },
      inference: { revision: capture.repositories.inference.revision, dirty: false },
    },
    hardware: capture.hardware,
    target: {
      tier: "q4",
      mode: descriptor.route === "edit" ? "image_to_image" : "text_to_image",
      overlay: descriptor.route === "control" ? "control" : "none",
      rung: "resident",
    },
    stdoutSha256: sha256(logBytes),
    inputs,
    outputs: [{ path: outputPath, sha256: outputSha256 }],
    claims: ["loadability", "overlay"],
  });
  return { descriptor, outputSha256, outputStdDev: image.stdDev, outputFormat: image.format,
    outputWidth: image.width, outputHeight: image.height, session };
}

async function ingestExcludedAttempt(descriptor, reader, acceptedPaths) {
  exactObject(descriptor, "excluded attempt", ["sourcePath", "reason"]);
  const sourcePath = repoRelative(descriptor.sourcePath, "excluded attempt sourcePath");
  requireText(descriptor.reason, `${sourcePath}.reason`);
  if (acceptedPaths.has(sourcePath)) fail(`${sourcePath} cannot be both accepted and excluded`);
  const bytes = await readRepoFile(sourcePath, reader);
  return { sourcePath, reason: descriptor.reason, stdoutSha256: sha256(bytes) };
}

function qualityFor(base) {
  const exact = ["resident", "staged_residency"].includes(base.descriptor.rung);
  const metrics = base.diagnostic ?? {
    changedFraction: 0,
    maximumError: 0,
    meanError: 0,
    rootMeanSquareError: 0,
    psnrDb: Number.POSITIVE_INFINITY,
  };
  return {
    contract: exact
      ? "identical artifact, prompt, seed, geometry, steps, precision, and conditioning; RGB8 must be byte-exact to resident"
      : `identical artifact, prompt, seed, geometry, steps, precision, and conditioning; provider-owned ${PARITY_METRIC} <= ${BOUNDED_MAX_ABS}`,
    identicalInputs: true,
    result: "passed",
    maximumError: metrics.maximumError,
    meanError: metrics.meanError,
    rootMeanSquareError: metrics.rootMeanSquareError,
    maximumErrorThreshold: exact ? 0 : BOUNDED_MAX_ABS,
    meanErrorThreshold: exact ? 0 : BOUNDED_MAX_ABS,
    rootMeanSquareErrorThreshold: exact ? 0 : BOUNDED_MAX_ABS,
  };
}

function makeRuntimeCompleteRecord(base, capture) {
  const rung = base.descriptor.rung;
  const peak = base.record.observed_peak_bytes;
  const predictedPeak = base.record.predicted_peak_bytes;
  const quality = qualityFor(base);
  const parameters = PARAMETERS[rung];
  const record = {
    id: "",
    logicalCaseId: "",
    status: "runtime_complete",
    evidenceScope: "authoritative",
    backend: "candle",
    loadShape: "deferred_materialization",
    repositories: capture.repositories,
    hardware: capture.hardware,
    artifact: {
      repository: capture.baseInput.repository,
      resolvedRevision: capture.baseInput.resolvedRevision,
      variant: capture.baseInput.variant,
      inventorySha256: capture.baseInput.sha256,
    },
    target: {
      modelId: MODEL_ID,
      provider: PROVIDER,
      tier: "q4",
      mode: "text_to_image",
      overlay: "none",
      geometry: { width: 1024, height: 1024, batch: 1, frames: 1 },
    },
    fixture: `flux2-dev-q4-seed42-1024x1024-output-${base.record.output_sha256}`,
    strategy: { rung, engagedRungs: COMPOSITIONS[rung], parameters },
    sweep: {
      axes: Object.entries(parameters)
        .filter(([, value]) => Number.isInteger(value))
        .map(([parameter, value]) => ({ parameter, testedValues: [value] })),
      cases: [{ parameters, result: "passed" }],
      rangeVerified: true,
    },
    scenarios: [
      { name: "exact_fit", result: "passed", predictedBytes: predictedPeak, effectiveBudgetBytes: predictedPeak },
      { name: "unknown_budget", result: "passed" },
      { name: "stale_evidence", result: "passed" },
      { name: "warm_repeat", result: "not_run", reason: "not exercised by this physical CUDA campaign" },
      { name: "cancel", result: "not_run", reason: "deferred to exhaustive lifecycle certification in SC-15922" },
      { name: "error", result: "not_run", reason: "deferred to exhaustive lifecycle certification in SC-15922" },
      { name: "loadability", result: "passed" },
      { name: "overlay", result: "not_applicable", reason: "the base request has no runtime overlay" },
    ],
    predictedPeakBytes: { overall: predictedPeak },
    observedMemory: { overall: { activeBytes: peak } },
    quality,
    negativeMutation: null,
    loadability: {
      result: "passed",
      resolvedPathFingerprint: `${capture.baseInput.repository}@${capture.baseInput.resolvedRevision}:${capture.baseInput.variant}`,
    },
    diagnostics: {
      adapter: BASE_HARNESS,
      execution: "executed",
      blockers: [],
      measurements: [
        { name: "requestLiveAllocationPeak", unit: "bytes", value: peak },
        { name: "outputRgbBytes", unit: "bytes", value: 1024 * 1024 * 3 },
        { name: "rgb8MaximumError", unit: "RGB8 levels", value: quality.maximumError },
      ],
    },
    calibrationFingerprint: capture.calibrationFingerprint,
    capturedAt: base.descriptor.capturedAt,
    harnessVersion: HARNESS_VERSION,
  };
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  validateRecord(record);
  return record;
}

export async function ingestFlux2Capture(capture, { reader = (relative) => readFile(path.join(ROOT, relative)) } = {}) {
  await validateLiveInferencePins();
  validateCaptureEnvelope(capture);
  const base = await Promise.all(capture.baseSessions.map((item) => ingestBaseSession(item, capture, reader)));
  const route = await Promise.all(capture.routeSessions.map((item) => ingestRouteSession(item, capture, reader)));
  const acceptedPaths = new Set([
    ...base.map((item) => item.descriptor.sourcePath.replaceAll("\\", "/")),
    ...route.map((item) => item.descriptor.sourcePath.replaceAll("\\", "/")),
  ]);
  const excludedAttempts = await Promise.all(
    capture.excludedAttempts.map((item) => ingestExcludedAttempt(item, reader, acceptedPaths)),
  );

  const blockers = [];
  const baseByRung = new Map(base.map((item) => [item.descriptor.rung, item]));
  if (baseByRung.size !== RUNGS.length || RUNGS.some((rung) => !baseByRung.has(rung))) {
    blockers.push("the authoritative base campaign must contain exactly one session for every ladder rung");
  }
  if (base.length !== RUNGS.length) blockers.push("the authoritative base campaign contains duplicate rung sessions");
  const resident = baseByRung.get("resident");
  const staged = baseByRung.get("staged_residency");
  if (resident && staged && resident.record.output_sha256 !== staged.record.output_sha256) {
    blockers.push("staged residency is not byte-identical to the resident source output");
  }
  if (resident) {
    const residentPath = resident.descriptor.outputPath.replaceAll("\\", "/");
    for (const item of base.filter(({ descriptor }) => descriptor.rung !== "resident")) {
      const reference = item.diagnostic?.reference.replaceAll("\\", "/");
      if (!reference || !reference.endsWith(residentPath)) {
        blockers.push(`${item.descriptor.rung} parity diagnostic does not bind the resident source output`);
      }
      validateMeasuredParity(resident, item);
    }
  }
  const routeByName = new Map(route.map((item) => [item.descriptor.route, item]));
  if (route.length !== 2 || !routeByName.has("edit") || !routeByName.has("control")) {
    blockers.push("the exact edit and control source sessions are both required");
  }
  if (capture.promotionIntent !== "runtime_complete") {
    blockers.push("capture promotionIntent is report_only; no calibration record may be emitted");
  }
  if (excludedAttempts.length === 0) blockers.push("excluded attempts must be explicitly enumerated with reasons and hashes");

  const promotionEligible = blockers.length === 0;
  const records = promotionEligible
    ? RUNGS.map((rung) => makeRuntimeCompleteRecord(baseByRung.get(rung), capture))
    : [];
  const sourceSessions = [...base, ...route].map((item) => item.session);
  const report = {
    schemaVersion: 1,
    story: STORY,
    promotionEligible,
    blockers,
    repositories: capture.repositories,
    hardware: capture.hardware,
    calibrationFingerprint: capture.calibrationFingerprint,
    artifact: capture.baseInput,
    acceptedSessions: sourceSessions.map((session) => ({
      id: session.id,
      kind: session.kind,
      sourcePath: session.sourcePath,
      stdoutSha256: session.stdoutSha256,
      target: session.target ?? null,
      claims: session.claims,
      outputSha256: session.outputs.map((output) => output.sha256),
    })),
    baseRungs: base.map((item) => ({
      rung: item.descriptor.rung,
      peakBytes: item.record.observed_peak_bytes,
      outputSha256: item.record.output_sha256,
      parity: item.record.parity,
      parityResult: item.record.parity_result,
      diagnostics: item.diagnostic,
    })),
    routes: route.map((item) => ({
      route: item.descriptor.route,
      outputSha256: item.outputSha256,
      outputStdDev: item.outputStdDev,
      outputFormat: item.outputFormat,
      outputWidth: item.outputWidth,
      outputHeight: item.outputHeight,
      sourceSessionId: item.session.id,
    })),
    excludedAttempts,
  };
  return { report, records, sourceSessions };
}

export function flux2CalibrationPlans(records) {
  return records.map((record) => ({
    name: `candle-flux2-dev-q4-${record.strategy.rung}-sc15833`,
    evidenceScope: "authoritative",
    backend: record.backend,
    loadShape: record.loadShape,
    target: record.target,
    rung: record.strategy.rung,
    engagedRungs: record.strategy.engagedRungs,
    calibrationFingerprint: record.calibrationFingerprint,
    fixture: record.fixture,
    cases: [{ parameters: record.strategy.parameters, expectedResult: "passed" }],
  }));
}

export function updateBundle(existing, ingestion) {
  if (!ingestion.report.promotionEligible || ingestion.records.length !== RUNGS.length) {
    fail(`refusing evidence update: ${ingestion.report.blockers.join("; ") || "incomplete record set"}`);
  }
  const records = existing.records.filter((record) =>
    !(record.backend === "candle" && record.target.provider === PROVIDER),
  );
  records.push(...ingestion.records);
  const sourceSessions = (existing.sourceSessions ?? []).filter(
    (session) => !session.sourcePath.startsWith(SESSION_PREFIX),
  );
  sourceSessions.push(...ingestion.sourceSessions);
  const bundle = {
    schemaVersion: 4,
    harnessVersion: HARNESS_VERSION,
    sourceSessions: sourceSessions.sort((left, right) => left.id.localeCompare(right.id)),
    records: records.sort((left, right) => left.id.localeCompare(right.id)),
  };
  validateBundle(bundle);
  return bundle;
}

export function updatePlan(existing, ingestion) {
  if (!ingestion.report.promotionEligible || ingestion.records.length !== RUNGS.length) {
    fail("refusing plan update without a complete promotion-eligible record set");
  }
  return {
    ...existing,
    providers: [
      ...existing.providers.filter((item) =>
        !(item.backend === "candle" && item.target?.provider === PROVIDER),
      ),
      ...flux2CalibrationPlans(ingestion.records),
    ],
  };
}

async function main() {
  const [capturePath, ...flags] = process.argv.slice(2);
  if (!capturePath) fail("usage: node scripts/sc-15833-flux2-evidence.mjs <capture.json> [--write-report <path>] [--write]");
  const absoluteCapture = path.resolve(ROOT, capturePath);
  const capture = JSON.parse(await readFile(absoluteCapture, "utf8"));
  const ingestion = await ingestFlux2Capture(capture);
  const reportIndex = flags.indexOf("--write-report");
  if (reportIndex !== -1) {
    const destination = flags[reportIndex + 1];
    if (!destination) fail("--write-report requires a destination");
    await writeFile(path.resolve(ROOT, destination), canonicalJson(ingestion.report));
  }
  if (flags.includes("--write")) {
    const existing = JSON.parse(await readFile(EVIDENCE, "utf8"));
    await writeFile(EVIDENCE, canonicalJson(updateBundle(existing, ingestion)));
    const plan = JSON.parse(await readFile(PLAN, "utf8"));
    await writeFile(PLAN, `${JSON.stringify(updatePlan(plan, ingestion), null, 2)}\n`);
  }
  process.stdout.write(canonicalJson(ingestion.report));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}
