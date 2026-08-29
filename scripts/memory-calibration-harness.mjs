#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, realpath, rename, writeFile } from "node:fs/promises";
import { createReadStream, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { providerClosureDigest } from "./inference-closure-digest.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CALIBRATION_SCHEMA = JSON.parse(
  readFileSync(path.join(ROOT, "packages/schemas/memory-calibration.schema.json"), "utf8"),
);
export const HARNESS_VERSION = "sceneworks-memory-v5";
// sc-18864 bumped the RECORD SHAPE (per-phase `deviceBytes`/`wiredBytes` removed) without changing
// the measuring instrument, so the bundle schema version moves and the harness version does not.
export const SCHEMA_VERSION = 6;
export const REQUIRED_SCENARIOS = [
  "exact_fit", "unknown_budget", "stale_evidence", "warm_repeat",
  "cancel", "error", "loadability", "overlay",
];
const RUNGS = [
  "resident", "staged_residency", "bounded_decode",
  "bounded_attention", "bounded_transformer_residency",
];
const RUNG_SET = new Set(RUNGS);
/// Persisted spellings of `gen_core::LoadShape`. Eager and deferred measurements are not
/// interchangeable, so this is a receipt axis rather than a fingerprint naming convention.
export const LOAD_SHAPES = ["eager_materialization", "deferred_materialization"];
export const LTX25_TRANSFORMER_VARIANTS = ["distilled", "dev"];
export const LTX25_DECODERS = ["conv", "diffvae"];
export const RUNG_REUSE_TOLERANCE = Object.freeze({
  absoluteBytes: 256 * 1024 * 1024,
  relative: 0.05,
});
const PHYSICAL_MLX_SESSION_OUTPUT_ROLES = Object.freeze([
  "request", "selected_rgb", "reference_rgb",
]);
const PHYSICAL_MLX_AV_SESSION_OUTPUT_ROLES = Object.freeze([
  "request", "selected_av", "reference_av",
]);
const PHYSICAL_MLX_PROVIDER_OUTPUT_ROLES = Object.freeze([
  "selected_rgb", "reference_rgb",
]);
const PHYSICAL_MLX_AV_PROVIDER_OUTPUT_ROLES = Object.freeze([
  "selected_av", "reference_av",
]);
const PHYSICAL_MLX_RGB_BASENAME = /^(implan-[0-9a-f]{20})-(selected_rgb|reference_rgb)-([1-9][0-9]*)x([1-9][0-9]*)-([0-9a-f]{64})\.rgb$/;
const PHYSICAL_MLX_AV_BASENAME = /^(implan-[0-9a-f]{20})-(selected_av|reference_av)-([1-9][0-9]*)x([1-9][0-9]*)-f([1-9][0-9]*)-([0-9a-f]{64})\.avbin$/;

function physicalMlxExpectedRoles(outputs, includeRequest) {
  const hasAv = outputs?.some((output) => output?.role === "selected_av" || output?.role === "reference_av");
  return hasAv
    ? (includeRequest ? PHYSICAL_MLX_AV_SESSION_OUTPUT_ROLES : PHYSICAL_MLX_AV_PROVIDER_OUTPUT_ROLES)
    : (includeRequest ? PHYSICAL_MLX_SESSION_OUTPUT_ROLES : PHYSICAL_MLX_PROVIDER_OUTPUT_ROLES);
}

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stable(value[key])]));
  }
  return value;
}

function validateExactOutputReceipts(outputs, expectedRoles, label) {
  if (!Array.isArray(outputs) || outputs.length !== expectedRoles.length) {
    fail(`${label} requires exactly ${expectedRoles.length} typed output receipts`);
  }
  const roles = new Set();
  const paths = new Set();
  for (const output of outputs) {
    object(output, `${label}[]`);
    text(output.role, `${label}[].role`);
    text(output.path, `${label}[].path`);
    if (!/^[0-9a-f]{64}$/.test(output.sha256)) {
      fail(`${label}[].sha256 must be a lowercase SHA-256 digest`);
    }
    if (!Number.isSafeInteger(output.bytes) || output.bytes <= 0) {
      fail(`${label}[].bytes must be a positive safe integer`);
    }
    if (roles.has(output.role)) fail(`${label} repeats output role ${output.role}`);
    if (paths.has(output.path)) fail(`${label} repeats output path ${output.path}`);
    roles.add(output.role);
    paths.add(output.path);
  }
  const expected = new Set(expectedRoles);
  if (roles.size !== expected.size || [...roles].some((role) => !expected.has(role))) {
    fail(`${label} must contain exactly ${expectedRoles.join(", ")}`);
  }
}

function physicalMlxRgbMetadata(output, label) {
  const match = path.posix.basename(output.path).match(PHYSICAL_MLX_RGB_BASENAME);
  if (!match || match[2] !== output.role) {
    fail(`${label} path must bind its logical case, RGB role, dimensions, and content digest`);
  }
  const width = Number(match[3]);
  const height = Number(match[4]);
  const expectedBytes = width * height * 3;
  if (!Number.isSafeInteger(expectedBytes) || output.bytes !== expectedBytes) {
    fail(`${label} byte count must equal its encoded RGB dimensions`);
  }
  if (output.sha256 !== match[5]) {
    fail(`${label} SHA-256 must match its content-addressed filename`);
  }
  return { logicalCaseId: match[1], width, height };
}

function physicalMlxAvMetadata(output, label) {
  const match = path.posix.basename(output.path).match(PHYSICAL_MLX_AV_BASENAME);
  if (!match || match[2] !== output.role) {
    fail(`${label} path must bind its logical case, A/V role, geometry, frames, and content digest`);
  }
  if (output.sha256 !== match[6]) {
    fail(`${label} SHA-256 must match its content-addressed filename`);
  }
  return {
    logicalCaseId: match[1],
    width: Number(match[3]),
    height: Number(match[4]),
    frames: Number(match[5]),
  };
}

export function parsePhysicalMlxAvContent(bytes, label = "physical MLX A/V output") {
  const magic = Buffer.from("SCENEWORKS_AV1\0", "ascii");
  if (bytes.length < 45 || !bytes.subarray(0, magic.length).equals(magic)) {
    fail(`${label} must begin with the canonical SCENEWORKS_AV1 header`);
  }
  let offset = magic.length;
  const width = bytes.readUInt32LE(offset); offset += 4;
  const height = bytes.readUInt32LE(offset); offset += 4;
  const frames = bytes.readUInt32LE(offset); offset += 4;
  const fps = bytes.readUInt32LE(offset); offset += 4;
  const sampleRateHz = bytes.readUInt32LE(offset); offset += 4;
  const channels = bytes.readUInt16LE(offset); offset += 2;
  const sampleCount = Number(bytes.readBigUInt64LE(offset)); offset += 8;
  if (![width, height, frames, fps, sampleRateHz, channels, sampleCount]
      .every((value) => Number.isSafeInteger(value) && value > 0)) {
    fail(`${label} canonical A/V header contains a zero or unsafe dimension`);
  }
  for (let frame = 0; frame < frames; frame += 1) {
    if (offset + 16 > bytes.length) fail(`${label} frame ${frame} header is truncated`);
    const frameWidth = bytes.readUInt32LE(offset); offset += 4;
    const frameHeight = bytes.readUInt32LE(offset); offset += 4;
    const pixelLength = Number(bytes.readBigUInt64LE(offset)); offset += 8;
    const expected = frameWidth * frameHeight * 3;
    if (frameWidth !== width || frameHeight !== height || !Number.isSafeInteger(expected)
        || pixelLength !== expected || offset + pixelLength > bytes.length) {
      fail(`${label} frame ${frame} does not match the canonical RGB geometry`);
    }
    offset += pixelLength;
  }
  const pcmBytes = sampleCount * 4;
  if (!Number.isSafeInteger(pcmBytes) || offset + pcmBytes !== bytes.length) {
    fail(`${label} PCM payload length does not match sampleCount`);
  }
  return {
    width, height, frames, fps, sampleRateHz, channels, sampleCount,
    pcmSha256: createHash("sha256").update(bytes.subarray(offset)).digest("hex"),
  };
}

function physicalMlxOutputMetadata(output, label) {
  return output.role === "selected_av" || output.role === "reference_av"
    ? physicalMlxAvMetadata(output, label)
    : physicalMlxRgbMetadata(output, label);
}

function validatePhysicalMlxSessionReceipts(session) {
  const sourceDirectory = path.posix.dirname(session.sourcePath);
  if (session.sourcePath !== `${sourceDirectory}/${session.id}.log`) {
    fail(`${session.id}: physical MLX sourcePath must be named from the session id`);
  }
  const request = session.outputs.find((output) => output.role === "request");
  if (request.path !== `${sourceDirectory}/${session.id}.request.json`) {
    fail(`${session.id}: request receipt must share the source directory and session id`);
  }
  for (const output of session.outputs.filter((candidate) => candidate.role !== "request")) {
    if (path.posix.dirname(output.path) !== sourceDirectory) {
      fail(`${session.id}: physical MLX RGB receipts must share the source directory`);
    }
    physicalMlxOutputMetadata(output, `${session.id}.${output.role}`);
  }
}

function validatePhysicalMlxOutputsAgainstRecord(record, session) {
  for (const output of session.outputs.filter((candidate) => candidate.role !== "request")) {
    const metadata = physicalMlxOutputMetadata(output, `${session.id}.${output.role}`);
    if (metadata.logicalCaseId !== record.logicalCaseId
        || metadata.width !== record.target.geometry.width
        || metadata.height !== record.target.geometry.height
        || (metadata.frames !== undefined && metadata.frames !== record.target.geometry.frames)) {
      fail(`${record.id}: physical MLX output receipt does not match the measured logical case geometry`);
    }
  }
  const hasAv = session.outputs.some((output) => output.role === "selected_av");
  if (hasAv !== (record.quality.audio !== undefined)) {
    fail(`${record.id}: physical MLX A/V receipts and typed audio quality must be present together`);
  }
  if (hasAv) validateAudioQuality(record);
}

function validateAudioQuality(record) {
  const audio = record.quality.audio;
  object(audio, `${record.id}.quality.audio`);
  if (audio.result !== "passed") fail(`${record.id}: A/V audio quality did not pass`);
  for (const field of ["sampleRateHz", "channels", "sampleCount"]) {
    if (!Number.isSafeInteger(audio[field]) || audio[field] <= 0) {
      fail(`${record.id}.quality.audio.${field} must be a positive safe integer`);
    }
  }
  for (const field of ["selectedPcmSha256", "referencePcmSha256"]) {
    if (!/^[0-9a-f]{64}$/.test(audio[field])) {
      fail(`${record.id}.quality.audio.${field} must be a lowercase SHA-256 digest`);
    }
  }
  for (const metric of [
    "maximumAbsoluteError", "meanAbsoluteError", "rootMeanSquareError",
    "maximumAbsoluteErrorThreshold", "meanAbsoluteErrorThreshold",
    "rootMeanSquareErrorThreshold",
  ]) number(audio[metric], `${record.id}.quality.audio.${metric}`);
  if (audio.maximumAbsoluteError > audio.maximumAbsoluteErrorThreshold
      || audio.meanAbsoluteError > audio.meanAbsoluteErrorThreshold
      || audio.rootMeanSquareError > audio.rootMeanSquareErrorThreshold) {
    fail(`${record.id}: audio quality threshold exceeded`);
  }
}

function validatePhysicalMlxAvContentsAgainstRecord(record, avContents, label) {
  if (avContents.size === 0) return;
  validateAudioQuality(record);
  const selected = avContents.get("selected_av");
  const reference = avContents.get("reference_av");
  const audio = record.quality.audio;
  const outputFps = record.diagnostics?.measurements?.find(
    (measurement) => measurement.name === "outputFps",
  )?.value;
  for (const content of [selected, reference]) {
    if (!content
        || content.width !== record.target.geometry.width
        || content.height !== record.target.geometry.height
        || content.frames !== record.target.geometry.frames
        || content.fps !== outputFps
        || content.sampleRateHz !== audio.sampleRateHz
        || content.channels !== audio.channels
        || content.sampleCount !== audio.sampleCount) {
      fail(`${label}: canonical A/V header differs from measured video/audio identity`);
    }
  }
  if (selected.pcmSha256 !== audio.selectedPcmSha256
      || reference.pcmSha256 !== audio.referencePcmSha256) {
    fail(`${label}: canonical A/V PCM hashes differ from quality.audio`);
  }
}

export function physicalMlxSessionId({
  kind, logicalCaseId, capturedAt, repositories, hardware, stdoutSha256,
}) {
  return `ims-${digest({
    kind,
    logicalCaseId,
    capturedAt,
    repositories,
    hardware,
    stdoutSha256,
  }).slice(0, 20)}`;
}

function physicalMlxDerivation(sessionId) {
  const direct = { kind: "direct", sourceSessionIds: [sessionId] };
  return {
    memory: direct,
    quality: direct,
    negativeMutation: direct,
    lifecycle: direct,
    loadability: direct,
    overlay: direct,
    justification: "Exact physical MLX provider response, artifact inventory, and selected/reference outputs are preserved as immutable capture receipts.",
  };
}

// sc-18864. The immutable provider-stdout receipts under `docs/calibration/` were emitted by the
// schema-v4 adapter, which wrote `deviceBytes` and `wiredBytes` as verbatim copies of
// `allocatorBytes`. The receipts are byte-immutable provenance and must NOT be rewritten, so the
// v4 -> v5 projection happens here, during reconstruction, and only for a receipt that actually
// carries the aliasing the adapter is known to have produced. A receipt whose aliases are NOT
// copies is refused outright rather than normalised: it would mean the adapter measured something
// the field names claimed, and this projection would be discarding a real reading.
export function projectPhaseMetricsToSchemaV5(observedMemory, label) {
  if (!observedMemory || typeof observedMemory !== "object") return observedMemory;
  const projected = {};
  for (const [phaseName, values] of Object.entries(observedMemory)) {
    if (!values || typeof values !== "object" || !("deviceBytes" in values || "wiredBytes" in values)) {
      projected[phaseName] = values;
      continue;
    }
    const { deviceBytes, wiredBytes, ...rest } = values;
    for (const [alias, aliasValue] of [["deviceBytes", deviceBytes], ["wiredBytes", wiredBytes]]) {
      if (aliasValue !== rest.allocatorBytes) {
        fail(`${label}.${phaseName}.${alias} is ${aliasValue}, not a copy of allocatorBytes ${rest.allocatorBytes}`);
      }
    }
    projected[phaseName] = rest;
  }
  return projected;
}

function recordFromPhysicalMlxResponse(providerResponse, request, session) {
  const { sourceCapture, ...fragment } = providerResponse;
  if (fragment.observedMemory) {
    fragment.observedMemory = projectPhaseMetricsToSchemaV5(
      fragment.observedMemory,
      `${session.id}.providerResponse.observedMemory`,
    );
  }
  const planned = request.planned;
  const baseInput = sourceCapture.inputs.find((input) => input?.role === "base");
  const record = {
    ...fragment,
    artifact: { ...fragment.artifact, inventorySha256: baseInput.sha256 },
    logicalCaseId: planned.logicalCaseId,
    evidenceScope: planned.evidenceScope,
    backend: planned.backend,
    loadShape: fragment.loadShape,
    repositories: request.repositories,
    hardware: request.hardware,
    target: planned.target,
    strategy: fragment.strategy,
    ...(planned.sourceProvenance ? { sourceProvenance: planned.sourceProvenance } : {}),
    calibrationFingerprint: planned.calibrationFingerprint,
    fixture: planned.fixture,
    harnessVersion: HARNESS_VERSION,
    derivation: physicalMlxDerivation(session.id),
  };
  record.id = recordId(record);
  return record;
}

async function writeImmutableReceipt(file, contents) {
  const bytes = Buffer.isBuffer(contents) ? contents : Buffer.from(contents, "utf8");
  try {
    await writeFile(file, bytes, { flag: "wx" });
  } catch (error) {
    if (error?.code !== "EEXIST") throw error;
    const existing = await readFile(file);
    if (!existing.equals(bytes)) fail(`immutable source receipt already exists with different bytes: ${file}`);
  }
}
export function canonicalJson(value) {
  return `${JSON.stringify(stable(value), null, 2)}\n`;
}
function digest(value) {
  return createHash("sha256").update(JSON.stringify(stable(value))).digest("hex");
}
function fail(message) {
  throw new Error(`memory-strategy calibration: ${message}`);
}
function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be an object`);
}
function text(value, label) {
  if (typeof value !== "string" || !value.trim()) fail(`${label} must be a non-empty string`);
}
function number(value, label, positive = false) {
  if (!Number.isFinite(value) || (positive ? value <= 0 : value < 0)) {
    fail(`${label} must be a ${positive ? "positive" : "nonnegative"} finite number`);
  }
}
function equal(left, right) {
  return JSON.stringify(stable(left)) === JSON.stringify(stable(right));
}

function requiresPhysicalMlxProvenanceForCurrency(record) {
  return record.backend === "mlx"
    && record.target.modelId === "qwen_image"
    && ["q4", "bf16"].includes(record.target.tier)
    && record.evidenceScope === "authoritative"
    && ["complete", "runtime_complete"].includes(record.status);
}

function validateCurrentPhysicalMlxProvenance(record, inferenceClosureDigests) {
  if (!requiresPhysicalMlxProvenanceForCurrency(record)) return;
  const provider = `${record.backend}:${record.target.provider}`;
  const live = inferenceClosureDigests?.[provider];
  if (
    live
    && record.repositories.inference.closureDigest === live
    && record.sourceProvenance !== "physical_mlx_v1"
  ) {
    fail(
      `${record.id}: current authoritative Qwen MLX ${record.target.tier} evidence requires ` +
        "sourceProvenance=physical_mlx_v1 and its validated physical source-session derivation",
    );
  }
}

function isNormalizedCalibrationPath(value) {
  if (typeof value !== "string" || !value.startsWith("docs/calibration/")) return false;
  const parts = value.split("/");
  return path.posix.normalize(value) === value
    && parts.every((part) => part && part !== "." && part !== "..");
}

function schemaTypeMatches(value, type) {
  if (type === "null") return value === null;
  if (type === "array") return Array.isArray(value);
  if (type === "object") return value !== null && typeof value === "object" && !Array.isArray(value);
  if (type === "integer") return Number.isInteger(value);
  if (type === "number") return typeof value === "number" && Number.isFinite(value);
  return typeof value === type;
}

function resolveLocalRef(root, reference) {
  if (!reference.startsWith("#/")) fail(`runtime schema validator supports only local refs, got ${reference}`);
  return reference
    .slice(2)
    .split("/")
    .reduce((value, token) => value[token.replaceAll("~1", "/").replaceAll("~0", "~")], root);
}

function schemaErrors(value, schema, root = schema, location = "$") {
  if (schema.$ref) return schemaErrors(value, resolveLocalRef(root, schema.$ref), root, location);
  const errors = [];
  if (schema.allOf) {
    for (const part of schema.allOf) errors.push(...schemaErrors(value, part, root, location));
  }
  if (schema.oneOf) {
    const matching = schema.oneOf.filter((part) => schemaErrors(value, part, root, location).length === 0);
    if (matching.length !== 1) errors.push(`${location}: expected exactly one schema branch, matched ${matching.length}`);
  }
  if (schema.if && schemaErrors(value, schema.if, root, location).length === 0 && schema.then) {
    errors.push(...schemaErrors(value, schema.then, root, location));
  }
  if (schema.const !== undefined && !equal(value, schema.const)) errors.push(`${location}: value does not equal const`);
  if (schema.enum && !schema.enum.some((candidate) => equal(value, candidate))) errors.push(`${location}: value is outside enum`);
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((type) => schemaTypeMatches(value, type))) {
      errors.push(`${location}: expected type ${types.join("|")}`);
      return errors;
    }
  }
  if (typeof value === "string") {
    if (schema.minLength !== undefined && value.length < schema.minLength) errors.push(`${location}: string is too short`);
    if (schema.pattern && !new RegExp(schema.pattern).test(value)) errors.push(`${location}: string does not match pattern`);
  }
  if (typeof value === "number") {
    if (schema.minimum !== undefined && value < schema.minimum) errors.push(`${location}: number is below minimum`);
    if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) {
      errors.push(`${location}: number is below exclusive minimum`);
    }
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${location}: array has too few items`);
    if (schema.maxItems !== undefined && value.length > schema.maxItems) errors.push(`${location}: array has too many items`);
    if (schema.uniqueItems && new Set(value.map((item) => JSON.stringify(stable(item)))).size !== value.length) {
      errors.push(`${location}: array items must be unique`);
    }
    if (schema.items) value.forEach((item, index) => errors.push(...schemaErrors(item, schema.items, root, `${location}[${index}]`)));
    if (schema.contains && !value.some((item, index) => schemaErrors(item, schema.contains, root, `${location}[${index}]`).length === 0)) {
      errors.push(`${location}: array does not contain required item`);
    }
  }
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const required of schema.required ?? []) {
      if (!Object.hasOwn(value, required)) errors.push(`${location}.${required}: required property is missing`);
    }
    const properties = schema.properties ?? {};
    for (const [key, child] of Object.entries(value)) {
      if (properties[key]) errors.push(...schemaErrors(child, properties[key], root, `${location}.${key}`));
      else if (schema.additionalProperties === false) errors.push(`${location}.${key}: unexpected property`);
    }
  }
  return errors;
}

function validateSchema(value) {
  const errors = schemaErrors(value, CALIBRATION_SCHEMA);
  if (errors.length) fail(`schema validation failed: ${errors.slice(0, 8).join("; ")}`);
}

export function logicalCaseId(spec) {
  return `implan-${digest({
    harnessVersion: HARNESS_VERSION,
    evidenceScope: spec.evidenceScope,
    backend: spec.backend,
    loadShape: spec.loadShape,
    target: spec.target,
    strategy: spec.strategy,
    sourceProvenance: spec.sourceProvenance,
    calibrationFingerprint: spec.calibrationFingerprint,
    fixture: spec.fixture,
    negative: spec.negative === true || spec.status === "negative_complete",
  }).slice(0, 20)}`;
}

/** `repositories` with the derived closure digest stripped, for identity hashing only. */
function repositoriesIdentity(repositories) {
  const { inference, ...rest } = repositories ?? {};
  if (!inference) return repositories;
  const { closureDigest, ...inferenceIdentity } = inference;
  return { ...rest, inference: inferenceIdentity };
}

export function recordId(record) {
  return `imc-${digest({
    harnessVersion: record.harnessVersion,
    evidenceScope: record.evidenceScope,
    // sc-17774: `inference.closureDigest` is DERIVED provenance — it is a pure function of
    // (provider, inference revision), both of which are already inside this identity — so it is
    // excluded. Including it would rotate all 65 record ids on a field that carries no new identity,
    // and every `evidenceRecords` reference in the manifest and matrix would have to be rewritten
    // for nothing.
    repositories: repositoriesIdentity(record.repositories),
    backend: record.backend,
    loadShape: record.loadShape,
    hardware: record.hardware,
    artifact: record.artifact,
    target: record.target,
    strategy: record.strategy,
    sourceProvenance: record.sourceProvenance,
    calibrationFingerprint: record.calibrationFingerprint,
    fixture: record.fixture,
  }).slice(0, 20)}`;
}

function validateEngagedRungs(strategy, label) {
  const engaged = strategy.engagedRungs;
  if (!Array.isArray(engaged) || engaged.length === 0) {
    fail(`${label}.engagedRungs must be a non-empty canonical rung set`);
  }
  if (engaged.some((rung) => !RUNG_SET.has(rung))) {
    fail(`${label}.engagedRungs contains an invalid rung`);
  }
  const canonical = RUNGS.filter((rung) => engaged.includes(rung));
  if (!equal(engaged, canonical)) {
    fail(`${label}.engagedRungs must be unique and in canonical rung order`);
  }
  if (!engaged.includes("resident") || !engaged.includes(strategy.rung)) {
    fail(`${label}.engagedRungs must include resident and the selected rung`);
  }
}

function validateRepositories(record) {
  object(record.repositories, `${record.id}.repositories`);
  for (const name of ["sceneWorks", "inference"]) {
    const repo = record.repositories[name];
    object(repo, `${record.id}.repositories.${name}`);
    if (!/^[0-9a-f]{7,40}$/.test(repo.revision)) fail(`${record.id}: ${name} revision must be a git SHA`);
    if (typeof repo.dirty !== "boolean") fail(`${record.id}: ${name}.dirty must be boolean`);
  }
  text(record.repositories.sceneWorks.matrixSourceRevision, `${record.id}.repositories.sceneWorks.matrixSourceRevision`);
}

function validateHardware(record) {
  const hardware = record.hardware;
  object(hardware, `${record.id}.hardware`);
  text(hardware.probe, `${record.id}.hardware.probe`);
  number(hardware.memoryBytes, `${record.id}.hardware.memoryBytes`, true);
  if (record.backend === "candle") {
    for (const key of ["deviceId", "name", "computeCapability", "driverVersion", "runtimeVersion"]) {
      text(hardware[key], `${record.id}.hardware.${key}`);
    }
  } else {
    for (const key of ["model", "chip", "osVersion", "metalDevice"]) {
      text(hardware[key], `${record.id}.hardware.${key}`);
    }
    number(hardware.mlxMemoryLimitBytes, `${record.id}.hardware.mlxMemoryLimitBytes`, true);
    number(hardware.wiredLimitBytes, `${record.id}.hardware.wiredLimitBytes`, true);
  }
}

// sc-18864: schema v5 carries the two counters MLX actually exposes plus their documented sum.
// `deviceBytes`/`wiredBytes` are gone — both adapters emitted them as copies of `allocatorBytes`,
// which is how every committed MLX record asserted wired residency above its own probed ceiling.
// `allocatorBytes` is DERIVED, so the rule here is an IDENTITY, not an ordering: an ordering is
// what let one number wear three names and drift from its own definition.
export const PHASE_METRICS = Object.freeze(["activeBytes", "allocatorBytes", "reclaimableBytes"]);

function validatePhaseMetrics(metrics, label) {
  object(metrics, label);
  for (const phase of ["conditioning", "denoise", "decode", "overall"]) {
    const values = metrics[phase];
    object(values, `${label}.${phase}`);
    for (const metric of PHASE_METRICS) {
      number(values[metric], `${label}.${phase}.${metric}`);
    }
    if (values.allocatorBytes !== values.activeBytes + values.reclaimableBytes) {
      fail(`${label}.${phase}: allocator bytes must equal active plus reclaimable bytes`);
    }
  }
  for (const metric of PHASE_METRICS) {
    const phaseMax = Math.max(metrics.conditioning[metric], metrics.denoise[metric], metrics.decode[metric]);
    if (metrics.overall[metric] < phaseMax) fail(`${label}.overall.${metric} must cover phase peaks`);
  }
}

function validatePredicted(predicted, label) {
  object(predicted, label);
  for (const phase of ["conditioning", "denoise", "decode", "overall"]) number(predicted[phase], `${label}.${phase}`);
  if (predicted.overall < Math.max(predicted.conditioning, predicted.denoise, predicted.decode)) {
    fail(`${label}.overall must cover phase peaks`);
  }
}

function validateRuntimePredicted(predicted, label) {
  object(predicted, label);
  const keys = Object.keys(predicted);
  if (keys.length === 1 && keys[0] === "overall") {
    number(predicted.overall, `${label}.overall`);
    return;
  }
  validatePredicted(predicted, label);
}

function validateRuntimeObserved(observed, label) {
  object(observed, label);
  const keys = Object.keys(observed);
  if (keys.length === 1 && keys[0] === "overall") {
    object(observed.overall, `${label}.overall`);
    const overallKeys = Object.keys(observed.overall);
    if (overallKeys.length !== 1 || overallKeys[0] !== "activeBytes") {
      fail(`${label}.overall must contain only the measured activeBytes high-water mark`);
    }
    number(observed.overall.activeBytes, `${label}.overall.activeBytes`);
    return;
  }
  validatePhaseMetrics(observed, label);
}

function validateComplete(record) {
  if (record.repositories.sceneWorks.dirty || record.repositories.inference.dirty) {
    fail(`${record.id}: complete evidence cannot come from a dirty repository`);
  }
  validatePredicted(record.predictedPeakBytes, `${record.id}.predictedPeakBytes`);
  validatePhaseMetrics(record.observedMemory, `${record.id}.observedMemory`);
  object(record.sweep, `${record.id}.sweep`);
  if (!Array.isArray(record.sweep.axes)) {
    fail(`${record.id}: sweep axes must be an array`);
  }
  if (record.sweep.axes.length === 0 && Object.keys(record.strategy.parameters).length > 0) {
    fail(`${record.id}: a parameterized complete strategy must sweep at least one axis`);
  }
  const axisNames = record.sweep.axes.map((axis) => axis.parameter);
  if (new Set(axisNames).size !== axisNames.length) fail(`${record.id}: sweep axes must be unique`);
  for (const axis of record.sweep.axes) {
    text(axis.parameter, `${record.id}.sweep.axis.parameter`);
    if (
      !Array.isArray(axis.testedValues) ||
      axis.testedValues.length < 1 ||
      new Set(axis.testedValues).size !== axis.testedValues.length
    ) fail(`${record.id}: ${axis.parameter} tested values must be nonempty and unique`);
  }
  if (record.sweep.rangeVerified !== true) fail(`${record.id}: complete evidence must verify its range`);
  if (!Array.isArray(record.sweep.cases) || record.sweep.cases.length < 1) {
    fail(`${record.id}: complete evidence needs at least one executed case`);
  }
  const passedCases = record.sweep.cases.filter((item) => item.result === "passed");
  const caseKeys = record.sweep.cases.map((item) => JSON.stringify(stable(item.parameters)));
  if (new Set(caseKeys).size !== caseKeys.length) fail(`${record.id}: executed sweep cases must be unique`);
  if (!passedCases.some((item) => equal(item.parameters, record.strategy.parameters))) {
    fail(`${record.id}: exact strategy parameters were not a passed executed case`);
  }
  for (const axis of record.sweep.axes) {
    const actual = [...new Set(passedCases.map((item) => item.parameters[axis.parameter]))].sort();
    const declared = [...axis.testedValues].sort();
    if (!equal(actual, declared) || actual.length < 1) {
      fail(`${record.id}: ${axis.parameter} range is not derived from passed executed cases`);
    }
  }
  const scenarios = new Map(record.scenarios.map((item) => [item.name, item]));
  if (record.scenarios.length !== REQUIRED_SCENARIOS.length || scenarios.size !== REQUIRED_SCENARIOS.length) {
    fail(`${record.id}: scenarios must be unique`);
  }
  for (const name of REQUIRED_SCENARIOS) {
    const scenario = scenarios.get(name);
    if (!scenario) fail(`${record.id}: missing ${name} scenario`);
    if (name === "overlay") {
      if (!["passed", "not_applicable"].includes(scenario.result)) fail(`${record.id}: overlay did not pass`);
      if (scenario.result === "not_applicable") text(scenario.reason, `${record.id}.overlay.reason`);
    } else if (scenario.result !== "passed") {
      fail(`${record.id}: ${name} scenario did not pass`);
    }
  }
  const exact = scenarios.get("exact_fit");
  number(exact.predictedBytes, `${record.id}.exact_fit.predictedBytes`);
  number(exact.effectiveBudgetBytes, `${record.id}.exact_fit.effectiveBudgetBytes`);
  if (exact.predictedBytes !== exact.effectiveBudgetBytes) fail(`${record.id}: exact_fit must exercise equality`);
  for (const name of ["cancel", "error"]) {
    if (scenarios.get(name).cleanupVerified !== true || scenarios.get(name).warmFollowUpPassed !== true) {
      fail(`${record.id}: ${name} must clean up and pass a warm follow-up`);
    }
  }
  if (
    record.quality.result !== "passed" ||
    (record.quality.identicalLatents !== true && record.quality.identicalInputs !== true)
  ) {
    fail(`${record.id}: complete quality evidence must pass with identical latents or identical inputs`);
  }
  text(record.quality.contract, `${record.id}.quality.contract`);
  for (const metric of ["maximumError", "meanError", "maximumErrorThreshold", "meanErrorThreshold"]) {
    number(record.quality[metric], `${record.id}.quality.${metric}`);
  }
  if (
    record.quality.maximumError > record.quality.maximumErrorThreshold ||
    record.quality.meanError > record.quality.meanErrorThreshold
  ) fail(`${record.id}: quality threshold exceeded`);
  const mutation = record.negativeMutation;
  object(mutation, `${record.id}.negativeMutation`);
  if (mutation.result !== "failed_as_expected" || !mutation.measured) {
    fail(`${record.id}: negative mutation must be measured and fail as expected`);
  }
  number(mutation.maximumError, `${record.id}.negativeMutation.maximumError`);
  number(mutation.meanError, `${record.id}.negativeMutation.meanError`);
  if (
    mutation.maximumError <= record.quality.maximumErrorThreshold &&
    mutation.meanError <= record.quality.meanErrorThreshold
  ) fail(`${record.id}: negative mutation did not breach a threshold`);
  if (record.loadability.result !== "passed") fail(`${record.id}: loadability did not pass`);
  text(record.loadability.resolvedPathFingerprint, `${record.id}.loadability.resolvedPathFingerprint`);
  assertResidencyFitsHardware(record);
}

// sc-18864. The quantity that must physically fit is the NON-RECLAIMABLE residency — the live-array
// peak. `allocatorBytes` adds an instantaneous end-of-phase cache reading to a peak-over-window, so
// it is an upper bound across two instants and legitimately exceeds physical memory on a capture
// that completed (a real LTX render co-existed 7.46 GiB above recommendedMaxWorkingSetSize). This
// used to read `deviceBytes`, which was that bound under another name.
function assertResidencyFitsHardware(record) {
  const resident = record.observedMemory.overall.activeBytes;
  if (resident > record.hardware.memoryBytes) {
    fail(`${record.id}: overall resident bytes exceed probed hardware memory`);
  }
  if (record.backend === "mlx" && resident > record.hardware.wiredLimitBytes) {
    fail(`${record.id}: overall wired bytes exceed the probed wired ceiling`);
  }
}

function validateNegative(record) {
  const mutation = record.negativeMutation;
  object(mutation, `${record.id}.negativeMutation`);
  if (mutation.result !== "failed_as_expected" || mutation.measured !== true) {
    fail(`${record.id}: negative case was not a measured expected failure`);
  }
  if (!equal(mutation.parameters, record.strategy.parameters)) {
    fail(`${record.id}: negative mutation parameters do not match the planned strategy parameters`);
  }
  text(record.quality.contract, `${record.id}.quality.contract`);
  for (const metric of ["maximumErrorThreshold", "meanErrorThreshold"]) {
    number(record.quality[metric], `${record.id}.quality.${metric}`);
  }
  number(mutation.maximumError, `${record.id}.negativeMutation.maximumError`);
  number(mutation.meanError, `${record.id}.negativeMutation.meanError`);
  if (
    mutation.maximumError <= record.quality.maximumErrorThreshold &&
    mutation.meanError <= record.quality.meanErrorThreshold
  ) fail(`${record.id}: negative case did not breach a threshold`);
}

function validateRuntimeComplete(record) {
  if (record.target.overlay !== "none") {
    fail(`${record.id}: runtime-complete evidence must target the base-only none overlay`);
  }
  if (record.repositories.sceneWorks.dirty || record.repositories.inference.dirty) {
    fail(`${record.id}: runtime-complete evidence cannot come from a dirty repository`);
  }
  validateRuntimePredicted(record.predictedPeakBytes, `${record.id}.predictedPeakBytes`);
  validateRuntimeObserved(record.observedMemory, `${record.id}.observedMemory`);
  const [soleCase] = record.sweep.cases;
  if (
    record.sweep.rangeVerified !== true ||
    record.sweep.cases.length !== 1 ||
    soleCase?.result !== "passed" ||
    !equal(soleCase.parameters, record.strategy.parameters)
  ) fail(`${record.id}: runtime-complete evidence needs exactly one passed case matching its strategy parameters`);
  const scenarios = new Map(record.scenarios.map((item) => [item.name, item]));
  if (record.scenarios.length !== REQUIRED_SCENARIOS.length || scenarios.size !== REQUIRED_SCENARIOS.length) {
    fail(`${record.id}: runtime-complete scenarios must be unique and exhaustive`);
  }
  for (const name of ["exact_fit", "unknown_budget", "stale_evidence", "loadability"]) {
    if (scenarios.get(name)?.result !== "passed") fail(`${record.id}: ${name} must pass for runtime activation`);
  }
  const lifecycle = ["warm_repeat", "cancel", "error"].map((name) => scenarios.get(name));
  const lifecycleNotRun = lifecycle.every((scenario) => scenario?.result === "not_run");
  const parityOnlyLifecycle = lifecycle[0]?.result === "passed"
    && lifecycle.slice(1).every((scenario) => scenario?.result === "not_run");
  const lifecyclePassed = lifecycle.every((scenario) => scenario?.result === "passed")
    && lifecycle.slice(1).every((scenario) =>
      scenario.cleanupVerified === true && scenario.warmFollowUpPassed === true);
  if (!lifecycleNotRun && !parityOnlyLifecycle && !lifecyclePassed) {
    fail(`${record.id}: runtime lifecycle must be entirely not_run, parity-only, or fully passed with cleanup and recovery`);
  }
  for (const [index, name] of ["warm_repeat", "cancel", "error"].entries()) {
    text(lifecycle[index].reason, `${record.id}.${name}.reason`);
  }
  const exact = scenarios.get("exact_fit");
  number(exact.predictedBytes, `${record.id}.exact_fit.predictedBytes`);
  number(exact.effectiveBudgetBytes, `${record.id}.exact_fit.effectiveBudgetBytes`);
  if (exact.predictedBytes !== exact.effectiveBudgetBytes) fail(`${record.id}: exact_fit must exercise equality`);
  const overlay = scenarios.get("overlay");
  if (overlay?.result !== "not_applicable") fail(`${record.id}: runtime-complete evidence must be base-only`);
  text(overlay.reason, `${record.id}.overlay.reason`);
  if (record.negativeMutation !== null) fail(`${record.id}: unexecuted negative mutation must remain null`);
  if (
    record.quality.result !== "passed" ||
    record.quality.identicalInputs !== true
  ) fail(`${record.id}: runtime-complete quality evidence must pass with identical inputs`);
  text(record.quality.contract, `${record.id}.quality.contract`);
  for (const metric of [
    "maximumError", "meanError", "rootMeanSquareError",
    "maximumErrorThreshold", "meanErrorThreshold", "rootMeanSquareErrorThreshold",
  ]) {
    number(record.quality[metric], `${record.id}.quality.${metric}`);
  }
  if (
    record.quality.maximumError > record.quality.maximumErrorThreshold ||
    record.quality.meanError > record.quality.meanErrorThreshold ||
    record.quality.rootMeanSquareError > record.quality.rootMeanSquareErrorThreshold
  ) fail(`${record.id}: runtime-complete quality threshold exceeded`);
  if (record.loadability.result !== "passed") fail(`${record.id}: runtime-complete loadability did not pass`);
  text(record.loadability.resolvedPathFingerprint, `${record.id}.loadability.resolvedPathFingerprint`);
  // The wired ceiling was checked only on `complete` before sc-18864, which is exactly how three
  // runtime-complete `mlx:flux2_dev` records shipped claiming up to 26.0 GB more wired residency
  // than the probed limit allows. Both statuses now run the same check.
  assertResidencyFitsHardware(record);
}

// sc-18864 review: `diagnostics.measurements.predictedOverallCeiling` is a DIAGNOSTIC COPY of the
// typed `predictedPeakBytes.overall`. The LTX arm derives both from `predicted_ceiling` over the
// same `overall.active` peak and says so in its own comment ("Agrees with the emitted
// predictedPeakBytes.overall by construction"). Nothing compared them, so when this story moved
// `predictedPeakBytes` onto the resident peak, all 14 committed LTX records kept a diagnostic still
// computed over the `allocatorBytes` co-existence bound: 2.5-4x the typed field sitting beside it,
// and `imc-2c064567893ea869006e` publishing 149.79 GB of predicted demand on a 130.57 GB host — the
// exact impossible figure this story exists to remove. An unchecked copy is a SECOND, UNVERSIONED
// DEFINITION of one quantity, which is the drift this story closes. The rule is EQUALITY, not an
// ordering: an ordering is what let the two spellings diverge in the first place.
function validateDiagnosticCeilingAgreesWithTypedField(record) {
  const measurements = record.diagnostics?.measurements;
  if (!Array.isArray(measurements)) return;
  const declared = measurements.find((entry) => entry?.name === "predictedOverallCeiling");
  const typed = record.predictedPeakBytes?.overall;
  if (declared === undefined || typeof typed !== "number") return;
  if (declared.value !== typed) {
    fail(
      `${record.id}: diagnostics predictedOverallCeiling ${declared.value} must equal ` +
        `predictedPeakBytes.overall ${typed}`,
    );
  }
}

export function validateRecord(record) {
  object(record, "record");
  text(record.id, "record.id");
  text(record.logicalCaseId, `${record.id}.logicalCaseId`);
  if (!["complete", "runtime_complete", "gated", "negative_complete"].includes(record.status)) {
    fail(`${record.id}: invalid status`);
  }
  if (!["authoritative", "candidate", "fixture"].includes(record.evidenceScope)) {
    fail(`${record.id}: invalid evidenceScope`);
  }
  if (!["mlx", "candle"].includes(record.backend)) fail(`${record.id}: invalid backend`);
  if (!["eager_materialization", "deferred_materialization"].includes(record.loadShape)) {
    fail(`${record.id}: invalid loadShape`);
  }
  validateRepositories(record);
  validateHardware(record);
  object(record.artifact, `${record.id}.artifact`);
  for (const key of ["repository", "resolvedRevision", "variant"]) text(record.artifact[key], `${record.id}.artifact.${key}`);
  if (record.artifact.inventorySha256 !== undefined && !/^[0-9a-f]{64}$/.test(record.artifact.inventorySha256)) {
    fail(`${record.id}.artifact.inventorySha256 must be a lowercase SHA-256 digest`);
  }
  object(record.target, `${record.id}.target`);
  for (const key of ["modelId", "provider", "tier", "mode", "overlay"]) text(record.target[key], `${record.id}.target.${key}`);
  if (record.target.modelId === "ltx_2_5") {
    if (!LTX25_TRANSFORMER_VARIANTS.includes(record.target.transformerVariant)) {
      fail(
        `${record.id}.target.transformerVariant must identify an LTX-2.5 transformer ` +
          `(${LTX25_TRANSFORMER_VARIANTS.join("|")})`,
      );
    }
    if (!LTX25_DECODERS.includes(record.target.decoder)) {
      fail(
        `${record.id}.target.decoder must identify an LTX-2.5 decoder ` +
          `(${LTX25_DECODERS.join("|")})`,
      );
    }
  }
  for (const key of ["width", "height", "batch", "frames"]) number(record.target.geometry[key], `${record.id}.target.geometry.${key}`, true);
  object(record.strategy, `${record.id}.strategy`);
  if (!RUNG_SET.has(record.strategy.rung)) fail(`${record.id}: invalid rung`);
  validateEngagedRungs(record.strategy, `${record.id}.strategy`);
  object(record.strategy.parameters, `${record.id}.strategy.parameters`);
  text(record.fixture, `${record.id}.fixture`);
  text(record.calibrationFingerprint, `${record.id}.calibrationFingerprint`);
  text(record.capturedAt, `${record.id}.capturedAt`);
  if (Number.isNaN(Date.parse(record.capturedAt))) fail(`${record.id}: invalid capturedAt`);
  if (record.harnessVersion !== HARNESS_VERSION) fail(`${record.id}: invalid harnessVersion`);
  if (record.logicalCaseId !== logicalCaseId(record)) fail(`${record.id}: logical identity mismatch`);
  if (record.id !== recordId(record)) fail(`${record.id}: deterministic identity mismatch`);
  object(record.loadability, `${record.id}.loadability`);
  object(record.quality, `${record.id}.quality`);
  if (!Array.isArray(record.scenarios)) fail(`${record.id}: scenarios must be an array`);
  validateDiagnosticCeilingAgreesWithTypedField(record);
  if (record.status === "complete") validateComplete(record);
  if (record.status === "runtime_complete") validateRuntimeComplete(record);
  if (record.status === "negative_complete") validateNegative(record);
  return record;
}

export function validateBundle(bundle) {
  validateSchema(bundle);
  object(bundle, "bundle");
  if (bundle.schemaVersion !== SCHEMA_VERSION || bundle.harnessVersion !== HARNESS_VERSION || !Array.isArray(bundle.records)) {
    fail("invalid bundle envelope");
  }
  const sessions = new Map();
  const inventoryInputs = new Map();
  for (const session of bundle.sourceSessions ?? []) {
    if (sessions.has(session.id)) fail(`duplicate source session ${session.id}`);
    if (!isNormalizedCalibrationPath(session.sourcePath)) {
      fail(`${session.id}: sourcePath must be a normalized path under docs/calibration`);
    }
    if (session.kind === "physical_mlx") {
      validateExactOutputReceipts(
        session.outputs,
        physicalMlxExpectedRoles(session.outputs, true),
        `${session.id}.outputs`,
      );
      validatePhysicalMlxSessionReceipts(session);
      for (const output of session.outputs) {
        if (!isNormalizedCalibrationPath(output.path)) {
          fail(`${session.id}: physical MLX output must be a normalized path under docs/calibration`);
        }
      }
    }
    const requiresExactInputs = session.claims.some((claim) =>
      ["loadability", "quality", "negative_mutation"].includes(claim));
    if (requiresExactInputs && session.inputs.length === 0) {
      fail(`${session.id}: artifact claims require exact inputs`);
    }
    if (session.target && requiresExactInputs) {
      const hasBase = session.inputs.some((input) => input.role === "base" && input.variant === session.target.tier);
      const hasOverlay = session.target.overlay === "control"
        ? session.inputs.some((input) => input.role === "control")
        : session.target.overlay === "lora"
          ? session.inputs.some((input) => input.role === "adapter")
          : true;
      if (!hasBase || !hasOverlay) fail(`${session.id}: artifact claim is missing its exact tier/overlay inputs`);
    }
    sessions.set(session.id, session);
    if (session.kind === "static_analysis" && !session.target) {
      for (const input of session.inputs) {
        const key = `${input.role}\0${input.variant}`;
        const existing = inventoryInputs.get(key);
        if (existing && !equal(existing, input)) fail(`inventory sessions disagree on exact ${input.role}/${input.variant} input identity`);
        inventoryInputs.set(key, input);
      }
    }
  }
  const ids = new Set();
  for (const record of bundle.records) {
    validateRecord(record);
    if (ids.has(record.id)) fail(`duplicate record ${record.id}`);
    ids.add(record.id);
    const requiresZImageDerivation = record.backend === "candle"
      && record.target.modelId === "z_image"
      && record.evidenceScope === "authoritative"
      && record.status === "complete";
    const isAuthoritativeQwenMlx = record.backend === "mlx"
      && record.target.modelId === "qwen_image"
      && record.evidenceScope === "authoritative"
      && record.status === "complete";
    if (record.sourceProvenance !== undefined
        && (record.sourceProvenance !== "physical_mlx_v1" || !isAuthoritativeQwenMlx)) {
      fail(`${record.id}: sourceProvenance is valid only for complete authoritative Qwen MLX evidence`);
    }
    const requiresQwenMlxDerivation = isAuthoritativeQwenMlx
      && record.sourceProvenance === "physical_mlx_v1";
    const provenancePolicy = requiresZImageDerivation
      ? "z_image"
      : requiresQwenMlxDerivation ? "qwen_image" : null;
    const requiresDerivation = provenancePolicy !== null;
    if (requiresQwenMlxDerivation && !record.artifact.inventorySha256) {
      fail(`${record.id}: authoritative Qwen MLX evidence requires an exact artifact inventory`);
    }
    if (requiresDerivation && !record.derivation) fail(`${record.id}: missing source-session derivation`);
    if (record.derivation) {
      const derivationSessionIds = new Set();
      for (const [claim, reference] of Object.entries(record.derivation).filter(([key]) => key !== "justification")) {
        const sourceClaim = claim === "negativeMutation" ? "negative_mutation" : claim;
        for (const sessionId of reference.sourceSessionIds) {
          derivationSessionIds.add(sessionId);
          const session = sessions.get(sessionId);
          if (!session) fail(`${record.id}: missing source session ${sessionId}`);
          if (!session.claims.includes(sourceClaim)) fail(`${record.id}: ${sessionId} does not claim ${sourceClaim}`);
          if (requiresQwenMlxDerivation && session.kind !== "physical_mlx") {
            fail(`${record.id}: authoritative Qwen MLX derivation requires a physical_mlx source session`);
          }
          if (requiresDerivation) {
            validateSourceInputsAgainstRecord(record, session, sourceClaim, inventoryInputs, provenancePolicy);
          }
          if (record.target.modelId.startsWith("ltx_")) {
            if (!session.target) {
              fail(`${record.id}: ${sessionId} is an LTX derivation source without a target identity`);
            }
            for (const [sourceField, recordField] of [
              ["tier", record.target.tier],
              ["mode", record.target.mode],
              ["overlay", record.target.overlay],
              ["rung", record.strategy.rung],
              ["transformerVariant", record.target.transformerVariant],
              ["decoder", record.target.decoder],
            ]) {
              if (session.target[sourceField] !== recordField) {
                fail(`${record.id}: ${sessionId} has the wrong LTX ${sourceField} identity`);
              }
            }
          }
          if (["memory", "quality", "overlay"].includes(claim)
              && session.target && session.target.tier !== record.target.tier) {
            fail(`${record.id}: ${claim} cannot cross precision tiers`);
          }
          if (claim === "memory" && session.target && session.target.rung !== record.strategy.rung) {
            fail(`${record.id}: memory cannot cross ladder rungs`);
          }
          if (["quality", "overlay"].includes(claim) && reference.kind === "direct"
              && session.target && session.target.overlay !== record.target.overlay) {
            fail(`${record.id}: direct ${claim} cannot cross overlays`);
          }
        }
      }
      if (!["direct", "conservative_upper_bound"].includes(record.derivation.memory.kind)) {
        fail(`${record.id}: invalid memory derivation kind`);
      }
      if (record.derivation.loadability.kind !== "direct") {
        fail(`${record.id}: loadability must be direct`);
      }
      if (requiresDerivation) {
        const exactInputs = new Map();
        for (const sessionId of record.derivation.loadability.sourceSessionIds) {
          for (const input of sessions.get(sessionId).inputs) {
            const existing = exactInputs.get(input.role);
            if (existing && !equal(existing, input)) {
              fail(`${record.id}: loadability sources disagree on exact ${input.role} input identity`);
            }
            exactInputs.set(input.role, input);
          }
        }
      }
      if (requiresQwenMlxDerivation && derivationSessionIds.size !== 1) {
        fail(`${record.id}: authoritative Qwen MLX claims must share one physical capture session`);
      }
      if (requiresQwenMlxDerivation) {
        const [sessionId] = derivationSessionIds;
        validatePhysicalMlxOutputsAgainstRecord(record, sessions.get(sessionId));
      }
      if (record.target.modelId.startsWith("ltx_")) {
        for (const sessionId of derivationSessionIds) {
          const session = sessions.get(sessionId);
          if (session.kind === "physical_mlx") {
            validatePhysicalMlxOutputsAgainstRecord(record, session);
          }
        }
      }
    }
  }
  return bundle;
}

function validateSourceInputsAgainstRecord(record, session, sourceClaim, inventoryInputs, provenancePolicy) {
  const fingerprint = record.loadability.resolvedPathFingerprint ?? "";
  for (const input of session.inputs) {
    const expectedInput = inventoryInputs.get(`${input.role}\0${input.variant}`);
    if (expectedInput) {
      if (!equal(expectedInput, input)) fail(`${record.id}: ${session.id} input differs from its exact inventory identity`);
    } else if (provenancePolicy !== "qwen_image" || session.kind !== "physical_mlx") {
      fail(`${record.id}: ${session.id} has no canonical ${input.role}/${input.variant} inventory`);
    }
    if (input.role === "base") {
      if (input.repository !== record.artifact.repository
          || input.resolvedRevision !== record.artifact.resolvedRevision) {
        fail(`${record.id}: ${session.id} base input does not match record artifact identity`);
      }
      const expectedTier = session.target?.tier ?? record.target.tier;
      if (input.variant !== expectedTier) fail(`${record.id}: ${session.id} base input has the wrong tier variant`);
      if (provenancePolicy === "qwen_image" && input.sha256 !== record.artifact.inventorySha256) {
        fail(`${record.id}: ${session.id} base input does not match the record artifact inventory`);
      }
    }
    const exactOverlaySource = (!session.target || session.target.overlay === record.target.overlay)
      && ["quality", "loadability", "overlay"].includes(sourceClaim);
    if (!exactOverlaySource) continue;
    const token = input.role === "base"
      ? `${input.repository}@${input.resolvedRevision}:${input.variant}`
      : input.role === "control"
        ? `+${input.repository}@${input.resolvedRevision}`
        : `+lora@${input.resolvedRevision}`;
    if (!fingerprint.includes(token)) {
      fail(`${record.id}: ${session.id} input is absent from the record artifact fingerprint`);
    }
  }
}

async function sha256File(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

async function resolveReceiptPath(relativePath, roots) {
  for (const root of roots) {
    const physicalRoot = await realpath(root);
    const candidate = path.resolve(physicalRoot, relativePath);
    const relative = path.relative(physicalRoot, candidate);
    if (!relative || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) continue;
    try {
      const physical = await realpath(candidate);
      const physicalRelative = path.relative(physicalRoot, physical);
      if (physicalRelative && !physicalRelative.startsWith(`..${path.sep}`) && !path.isAbsolute(physicalRelative)) {
        return physical;
      }
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
  fail(`missing immutable source receipt ${relativePath}`);
}

export async function validateSourceSessionFiles(
  bundle,
  extraRoot = null,
  inferenceClosureDigests = null,
) {
  validateBundle(bundle);
  if (inferenceClosureDigests) {
    for (const record of bundle.records) {
      validateCurrentPhysicalMlxProvenance(record, inferenceClosureDigests);
    }
  }
  const roots = [...new Set([extraRoot, ROOT].filter(Boolean).map((root) => path.resolve(root)))];
  for (const session of bundle.sourceSessions ?? []) {
    const sourceFile = await resolveReceiptPath(session.sourcePath, roots);
    const sourceBytes = await readFile(sourceFile);
    if (createHash("sha256").update(sourceBytes).digest("hex") !== session.stdoutSha256) {
      fail(`${session.id}: sourcePath no longer matches stdoutSha256`);
    }
    if (session.kind !== "physical_mlx") continue;
    const boundRecords = bundle.records.filter((record) => Object.entries(record.derivation ?? {})
      .filter(([key]) => key !== "justification")
      .some(([, reference]) => reference.sourceSessionIds.includes(session.id)));
    if (boundRecords.length !== 1) {
      fail(`${session.id}: physical MLX session must bind exactly one evidence record`);
    }
    const record = boundRecords[0];
    const requestOutput = session.outputs.find((output) => output.role === "request");
    const avContents = new Map();
    for (const output of session.outputs) {
      const outputFile = await resolveReceiptPath(output.path, roots);
      const outputBytes = await readFile(outputFile);
      if (createHash("sha256").update(outputBytes).digest("hex") !== output.sha256
          || outputBytes.length !== output.bytes) {
        fail(`${session.id}: output ${output.path} no longer matches its SHA-256 receipt`);
      }
      if (output.role === "selected_av" || output.role === "reference_av") {
        avContents.set(
          output.role,
          parsePhysicalMlxAvContent(outputBytes, `${session.id}.${output.role}`),
        );
      }
    }
    validatePhysicalMlxAvContentsAgainstRecord(record, avContents, session.id);
    const requestBytes = await readFile(await resolveReceiptPath(requestOutput.path, roots));
    let request;
    try {
      request = JSON.parse(requestBytes.toString("utf8"));
    } catch {
      fail(`${session.id}: request receipt must contain JSON`);
    }
    if (canonicalJson(request) !== requestBytes.toString("utf8")) {
      fail(`${session.id}: request receipt must contain canonical JSON`);
    }
    if (request.action !== "run" || !request.planned) {
      fail(`${session.id}: request receipt must describe one fresh planned case`);
    }
    for (const key of [
      "logicalCaseId", "evidenceScope", "backend", "loadShape", "target", "strategy",
      "sourceProvenance", "calibrationFingerprint", "fixture",
    ]) {
      if (!equal(request.planned[key], record[key])) {
        fail(`${session.id}: request receipt ${key} does not match its evidence record`);
      }
    }
    const expectedSessionTarget = {
      tier: request.planned.target.tier,
      mode: request.planned.target.mode,
      overlay: request.planned.target.overlay,
      ...(request.planned.target.transformerVariant
        ? { transformerVariant: request.planned.target.transformerVariant }
        : {}),
      ...(request.planned.target.decoder ? { decoder: request.planned.target.decoder } : {}),
      rung: request.planned.strategy.rung,
    };
    if (!equal(request.repositories, record.repositories)
        || !equal(session.repositories, record.repositories)
        || !equal(request.hardware, record.hardware)
        || !equal(session.hardware, record.hardware)
        || session.capturedAt !== record.capturedAt
        || !equal(session.target, expectedSessionTarget)) {
      fail(`${session.id}: request receipt provenance does not match its evidence record`);
    }
    let providerResponse;
    try {
      providerResponse = JSON.parse(sourceBytes.toString("utf8"));
    } catch {
      fail(`${session.id}: provider response receipt must contain JSON`);
    }
    object(providerResponse, `${session.id}.providerResponse`);
    object(providerResponse.sourceCapture, `${session.id}.providerResponse.sourceCapture`);
    if (providerResponse.sourceCapture.kind !== session.kind
        || !equal(providerResponse.sourceCapture.inputs, session.inputs)
        || !equal(providerResponse.sourceCapture.claims, session.claims)
        || providerResponse.capturedAt !== session.capturedAt) {
      fail(`${session.id}: provider response source inputs, claims, or capture time do not match the session receipt`);
    }
    validateExactOutputReceipts(
      providerResponse?.sourceCapture?.outputs,
      physicalMlxExpectedRoles(providerResponse?.sourceCapture?.outputs, false),
      `${session.id}.providerResponse.sourceCapture.outputs`,
    );
    for (const providerOutput of providerResponse.sourceCapture.outputs) {
      physicalMlxOutputMetadata(
        providerOutput,
        `${session.id}.providerResponse.sourceCapture.outputs[${providerOutput.role}]`,
      );
      const sessionOutput = session.outputs.find((output) => output.role === providerOutput.role);
      if (!equal(
        {
          role: providerOutput.role,
          path: providerOutput.path,
          sha256: providerOutput.sha256,
          bytes: providerOutput.bytes,
        },
        sessionOutput,
      )) {
        fail(`${session.id}: provider response output attestation does not match the session receipt`);
      }
    }
    const expectedSessionId = physicalMlxSessionId({
      kind: providerResponse.sourceCapture.kind,
      logicalCaseId: request.planned.logicalCaseId,
      capturedAt: providerResponse.capturedAt,
      repositories: session.repositories,
      hardware: session.hardware,
      stdoutSha256: session.stdoutSha256,
    });
    if (session.id !== expectedSessionId) {
      fail(`${session.id}: physical MLX session id does not match its provider response digest`);
    }
    const reconstructedRecord = recordFromPhysicalMlxResponse(providerResponse, request, session);
    if (!equal(reconstructedRecord, record)) {
      fail(`${session.id}: provider response measurements do not match the evidence record`);
    }
  }
  return bundle;
}

export function evidenceSemantics(record, revisions) {
  validateRecord(record);
  if (record.evidenceScope === "fixture") return "fixture";
  if (record.evidenceScope === "candidate") return "candidate";
  if (record.status === "negative_complete") return "negative";
  if (!["complete", "runtime_complete"].includes(record.status)) return "gated";

  // sc-17774: currency is decided by the PROVIDER'S OWN compile closure, never by the inference pin.
  //
  // This used to read `record.repositories.inference.revision === revisions.inference`. The unit of
  // invalidation was therefore the whole inference repository at commit granularity: a commit to
  // `mlx-gen-z-image` demoted `flux2_dev`, and a documentation-only commit demoted all six
  // calibrated providers. Measured over the 90 days to `fbb00d6b`, all 2812 non-merge commits
  // demoted everything. The comment that used to sit here already stated the intended policy —
  // "invalidation is owned by calibrationBinding's provider ABI fingerprint" — and then the line
  // below it did the opposite for the inference side. That gap is what this change closes.
  //
  // The closure digest is derived in `scripts/inference-closure-digest.mjs` and lives in
  // `config/inference-provider-closures.json`; the record carries the digest it was captured under.
  const captured = record.repositories.inference.closureDigest;
  // Keyed `<backend>:<provider>` — a provider id alone is not unique. `krea_2_turbo_control` exists
  // on both mlx and candle with different crates, so the bare id would compare one backend's
  // measurements against the other backend's code.
  const provider = `${record.backend}:${record.target.provider}`;
  const live = revisions.inferenceClosureDigests?.[provider];
  // Fail closed and LOUDLY. Falling back to pin equality when a digest is missing would silently
  // restore the policy this replaces, and the fallback would be invisible in a green run.
  if (!captured) {
    fail(
      `${record.id}: no repositories.inference.closureDigest. Every complete record must carry the ` +
        "provider closure digest it was captured under (sc-17774); re-run the backfill in " +
        "scripts/backfill-closure-digests.mjs against an inference clone.",
    );
  }
  if (!live) {
    fail(
      `${record.id}: provider "${provider}" has no entry in config/inference-provider-closures.json. ` +
        "Declare its inference crate and regenerate: node scripts/inference-closure-digest.mjs " +
        "--repo <inference> --write.",
    );
  }
  validateCurrentPhysicalMlxProvenance(record, revisions.inferenceClosureDigests);
  return captured === live ? "current" : "historical";
}

export function mergeBundles(left, right) {
  validateBundle(left);
  validateBundle(right);
  const records = new Map(left.records.map((record) => [record.id, record]));
  for (const record of right.records) {
    const existing = records.get(record.id);
    if (existing && !equal(existing, record)) fail(`conflicting record with exact identity ${record.id}`);
    records.set(record.id, record);
  }
  const sourceSessions = new Map((left.sourceSessions ?? []).map((session) => [session.id, session]));
  for (const session of right.sourceSessions ?? []) {
    const existing = sourceSessions.get(session.id);
    if (existing && !equal(existing, session)) fail(`conflicting source session ${session.id}`);
    sourceSessions.set(session.id, session);
  }
  return {
    schemaVersion: SCHEMA_VERSION,
    harnessVersion: HARNESS_VERSION,
    sourceSessions: [...sourceSessions.values()].sort((a, b) => a.id.localeCompare(b.id)),
    records: [...records.values()].sort((a, b) => a.id.localeCompare(b.id)),
  };
}

export function compareRungReuse(fresh, reused, tolerance = RUNG_REUSE_TOLERANCE) {
  validateBundle(fresh);
  validateBundle(reused);
  const freshByLogicalId = new Map(fresh.records.map((record) => [record.logicalCaseId, record]));
  const reusedByLogicalId = new Map(reused.records.map((record) => [record.logicalCaseId, record]));
  if (freshByLogicalId.size !== fresh.records.length || reusedByLogicalId.size !== reused.records.length) {
    fail("fresh/reused comparison cannot contain duplicate logical cases");
  }
  if (freshByLogicalId.size !== reusedByLogicalId.size) {
    fail(`fresh/reused comparison cardinality differs: ${freshByLogicalId.size} != ${reusedByLogicalId.size}`);
  }
  const comparisons = [];
  for (const [logicalCaseId, freshRecord] of freshByLogicalId) {
    const reusedRecord = reusedByLogicalId.get(logicalCaseId);
    if (!reusedRecord) fail(`reused capture is missing ${logicalCaseId}`);
    if (freshRecord.id !== reusedRecord.id) {
      fail(`${logicalCaseId}: fresh/reused comparison domain differs in repository, hardware, or artifact provenance`);
    }
    if (!freshRecord.observedMemory || !reusedRecord.observedMemory) {
      fail(`${logicalCaseId}: fresh/reused comparison requires observedMemory on both records`);
    }
    const metrics = [];
    const overallOnly = Object.keys(freshRecord.observedMemory).length === 1
      || Object.keys(reusedRecord.observedMemory).length === 1;
    if (overallOnly && (
      Object.keys(freshRecord.observedMemory).length !== 1
      || Object.keys(reusedRecord.observedMemory).length !== 1
    )) fail(`${logicalCaseId}: fresh/reused observedMemory shapes differ`);
    const phases = overallOnly ? ["overall"] : ["conditioning", "denoise", "decode", "overall"];
    const metricNames = overallOnly ? ["activeBytes"] : PHASE_METRICS;
    for (const phase of phases) {
      for (const metric of metricNames) {
        const freshBytes = freshRecord.observedMemory[phase][metric];
        const reusedBytes = reusedRecord.observedMemory[phase][metric];
        const differenceBytes = Math.abs(reusedBytes - freshBytes);
        const allowedBytes = Math.max(tolerance.absoluteBytes, Math.ceil(freshBytes * tolerance.relative));
        metrics.push({ phase, metric, freshBytes, reusedBytes, differenceBytes, allowedBytes,
          passed: differenceBytes <= allowedBytes });
      }
    }
    comparisons.push({
      logicalCaseId,
      rung: freshRecord.strategy.rung,
      passed: metrics.every((metric) => metric.passed),
      metrics,
    });
  }
  const backend = fresh.records[0]?.backend;
  if (
    !backend ||
    fresh.records.some((record) => record.backend !== backend) ||
    reused.records.some((record) => record.backend !== backend)
  ) {
    fail("fresh/reused comparison must contain exactly one backend");
  }
  return {
    schemaVersion: 1,
    backend,
    tolerance,
    verdict: comparisons.every((comparison) => comparison.passed)
      ? "amortizable"
      : "unable_to_amortize",
    comparisons,
  };
}

function completedLogicalIds(record) {
  if (record.status === "negative_complete") return [record.logicalCaseId];
  if (!["complete", "runtime_complete"].includes(record.status)) return [];
  return record.sweep.cases.filter((item) => item.result === "passed").map((item) =>
    logicalCaseId({
      evidenceScope: record.evidenceScope,
      backend: record.backend,
      loadShape: record.loadShape,
      target: record.target,
      strategy: {
        rung: record.strategy.rung,
        engagedRungs: record.strategy.engagedRungs,
        parameters: item.parameters,
      },
      sourceProvenance: record.sourceProvenance,
      calibrationFingerprint: record.calibrationFingerprint,
      fixture: record.fixture,
      negative: item.result === "failed",
    }),
  );
}

function operationallyAttemptedLogicalIds(records, repositories, hardware) {
  // sc-17935: compared on repository IDENTITY, with the derived closure digest stripped from both
  // sides — the same argument `recordId` makes. The digest is a pure function of (lane, inference
  // revision); the revision is compared here, and the lane is fixed by `logicalCaseId`, so it adds
  // nothing. Comparing it raw would make resume lane-sensitive: the run-level `repositories` no
  // longer carries any one lane's digest, so every prior record would look foreign and a resumed
  // capture would repeat GPU work it had already paid for.
  //
  // The converse is deliberate and worth stating: resume is now blind to the digest in BOTH
  // directions, so a record stamped under a wrong declaration stays "already attempted" until the
  // bundle is discarded. That is the correct trade — this check decides whether to re-run a
  // multi-hour capture, not whether the evidence is current. Currency is `evidenceSemantics`, which
  // reads the digest and fails closed.
  const wanted = repositoriesIdentity(repositories);
  return new Set(records
    .filter((record) =>
      record.harnessVersion === HARNESS_VERSION &&
      equal(repositoriesIdentity(record.repositories), wanted) &&
      equal(record.hardware, hardware)
    )
    .map((record) => record.logicalCaseId));
}

export function expandPlan(config, completed = []) {
  object(config, "plan config");
  const completedLogical = new Set(
    completed.flatMap(completedLogicalIds),
  );
  const cases = [];
  for (const provider of config.providers) {
    if (provider.target?.modelId === "ltx_2_5") {
      if (!LTX25_TRANSFORMER_VARIANTS.includes(provider.target.transformerVariant)) {
        fail(
          `${provider.name ?? provider.target.provider}: LTX-2.5 plan target requires ` +
            `transformerVariant=${LTX25_TRANSFORMER_VARIANTS.join("|")}`,
        );
      }
      if (!LTX25_DECODERS.includes(provider.target.decoder)) {
        fail(
          `${provider.name ?? provider.target.provider}: LTX-2.5 plan target requires ` +
            `decoder=${LTX25_DECODERS.join("|")}`,
        );
      }
    }
    if (!["eager_materialization", "deferred_materialization"].includes(provider.loadShape)) {
      fail(`${provider.name ?? provider.target.provider}: plan provider requires an explicit loadShape`);
    }
    for (const candidate of provider.cases) {
      if (!["passed", "failed"].includes(candidate.expectedResult)) {
        fail(`${provider.name ?? provider.target.provider}: plan case requires expectedResult`);
      }
      if ((candidate.expectedResult === "failed") !== (candidate.negative === true)) {
        fail(`${provider.name ?? provider.target.provider}: failed cases must be explicitly negative`);
      }
      const spec = {
        evidenceScope: provider.evidenceScope,
        backend: provider.backend,
        loadShape: provider.loadShape,
        target: provider.target,
        strategy: {
          rung: provider.rung,
          engagedRungs: provider.engagedRungs,
          parameters: candidate.parameters,
        },
        ...(provider.sourceProvenance ? { sourceProvenance: provider.sourceProvenance } : {}),
        calibrationFingerprint: provider.calibrationFingerprint,
        fixture: provider.fixture,
        negative: candidate.negative === true,
      };
      const id = logicalCaseId(spec);
      if (!completedLogical.has(id)) cases.push({
        logicalCaseId: id,
        ...spec,
        expectedResult: candidate.expectedResult,
        modelLoadPolicy: provider.modelLoadPolicy ?? "fresh_per_case",
        modelLoadGroup: provider.modelLoadGroup ?? null,
      });
    }
  }
  return cases.sort((a, b) => a.logicalCaseId.localeCompare(b.logicalCaseId));
}

export async function assessProviderReuse({ config, providerCommand, backend, fixture }) {
  if (!Array.isArray(providerCommand) || !providerCommand.length) fail("provider command must be a JSON argv array");
  const planned = expandPlan(config).filter(
    (candidate) => candidate.backend === backend && (!fixture || candidate.fixture === fixture),
  ).sort((left, right) => RUNGS.indexOf(left.strategy.rung) - RUNGS.indexOf(right.strategy.rung));
  if (planned.length === 0) fail(`reuse assessment selected no ${backend} cases`);
  const response = JSON.parse(await execute(
    providerCommand[0],
    providerCommand.slice(1),
    canonicalJson({ action: "assess_batch", planned }),
  ));
  if (!["eligible_for_measurement", "unable_to_amortize"].includes(response.verdict)) {
    fail(`provider returned invalid reuse-assessment verdict ${JSON.stringify(response.verdict)}`);
  }
  text(response.reason, "reuse assessment reason");
  return {
    schemaVersion: 1,
    backend,
    fixture: fixture ?? null,
    tolerance: RUNG_REUSE_TOLERANCE,
    ...response,
  };
}

function execute(command, args, input) {
  return new Promise((resolve, reject) => {
    const hasInput = input !== undefined;
    const child = spawn(command, args, {
      stdio: [hasInput ? "pipe" : "ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    let spawnError;
    let stdinError;
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", (error) => (spawnError = error));
    child.stdin?.on("error", (error) => (stdinError = error));
    child.on("close", (code) => {
      if (spawnError) return reject(new Error(`could not start ${command}: ${spawnError.message}`));
      if (stdinError) {
        return reject(
          new Error(
            `${command} closed its provider-protocol stdin before consuming the JSON request: ${stdinError.message}`,
          ),
        );
      }
      return code === 0
        ? resolve(stdout)
        : reject(new Error(`${command} exited ${code}: ${stderr.trim() || "no stderr"}`));
    });
    if (hasInput) child.stdin.end(input);
  });
}

export async function runProviderPlan({
  config, providerCommand, sceneWorksRepo, inferenceRepo, resume, backend, providerName, fixture,
  onProviderInvocation, onProviderCheckpoint, forceFreshPerCase = false, forceBatchRungs = false,
  rawLogDir = null, sourcePathPrefix = null,
  // sc-17774: injectable so the runner's own tests can drive synthetic repositories, which have no
  // inference crate layout to derive a real closure from. Production always uses the default.
  closureDigestFor = null,
  // SC-20191 keeps the canonical request/record assembly here while allowing the one contained
  // campaign entry to transport the provider request through its sealed watchdog wrapper.
  executeProvider = execute,
}) {
  if (!Array.isArray(providerCommand) || !providerCommand.length) fail("provider command must be a JSON argv array");
  if (Boolean(rawLogDir) !== Boolean(sourcePathPrefix)) {
    fail("--raw-log-dir and --source-path-prefix must be supplied together");
  }
  if (sourcePathPrefix) {
    const parts = sourcePathPrefix.split("/");
    if (!/^docs\/calibration\/[A-Za-z0-9._/-]+$/.test(sourcePathPrefix)
        || path.posix.normalize(sourcePathPrefix) !== sourcePathPrefix
        || parts.some((part) => !part || part === "." || part === "..")) {
      fail("source path prefix must be a normalized path under docs/calibration");
    }
  }
  if (rawLogDir) {
    await mkdir(rawLogDir, { recursive: true });
    rawLogDir = await realpath(rawLogDir);
    for (const [name, repo] of [["SceneWorks", sceneWorksRepo], ["inference", inferenceRepo]]) {
      const relative = path.relative(await realpath(repo), rawLogDir);
      if (!relative.startsWith("..") && !path.isAbsolute(relative)) {
        fail(`raw log directory must be outside the ${name} checkout so capture provenance cannot dirty it`);
      }
    }
  }
  const gitState = async (repo, sceneWorks = false) => ({
    revision: (await execute("git", ["-C", repo, "rev-parse", "HEAD"])).trim(),
    dirty: Boolean((await execute("git", ["-C", repo, "status", "--porcelain"])).trim()),
    ...(sceneWorks
      ? {
          matrixSourceRevision: JSON.parse(
            await readFile(path.join(repo, "docs/generated/memory-matrix.json"), "utf8"),
          ).generatedFrom.sceneWorksRevision,
        }
      : {}),
  });
  // sc-17774: stamp the provider's compile-closure digest AT CAPTURE TIME. The runner already has a
  // live inference checkout, so the captured half of the currency comparison is derived here rather
  // than backfilled later.
  //
  // sc-17935: the LANE decides which closure is measured, and a lane is `<backend>:<provider>` —
  // exactly the key `config/inference-provider-closures.json` and `evidenceSemantics` use. This used
  // to pass `providerName`, the `--provider` PLAN-ENTRY name (`candle-krea-q4-fresh-reference-
  // resident`), which is not a table key and is `undefined` under the `--fixture` selection every
  // checked-in capture workflow actually uses. Both spellings failed with `provider "…" has no
  // entry`, so no macOS or Windows capture job could produce replacement evidence — precisely when
  // a narrowed currency term makes replacement evidence the remedy.
  const closureDigest =
    closureDigestFor ??
    (async (lane, revision) => {
      const declarations = JSON.parse(
        await readFile(path.join(sceneWorksRepo, "config/inference-provider-closures.json"), "utf8"),
      );
      const crateDir = declarations.providers?.[lane]?.crate;
      if (!crateDir) {
        fail(
          `lane "${lane}" has no entry in config/inference-provider-closures.json. Declare its ` +
            "inference crate before capturing evidence, or the record cannot carry a currency term.",
        );
      }
      return providerClosureDigest({
        repo: inferenceRepo,
        revision,
        provider: lane,
        crateDir,
      }).digest;
    });
  const laneOf = (planned) => `${planned.backend}:${planned.target.provider}`;
  const probeRepositories = async () => ({
    sceneWorks: await gitState(sceneWorksRepo, true),
    inference: await gitState(inferenceRepo),
  });
  // The stability probe deliberately carries NO closure digest: the digest is a pure function of
  // (lane, inference revision), and the revision is compared here, so hashing it again would only
  // re-derive a value that cannot move while `revision` holds still.
  const repositories = await probeRepositories();
  const assertRepositoriesStable = async () => {
    const after = await probeRepositories();
    if (!equal(repositories, after)) fail("repository HEAD or dirty state changed during provider execution");
  };
  const existing = resume
    ? validateBundle(resume)
    : { schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION, sourceSessions: [], records: [] };
  const selectedConfig = {
    ...config,
    providers: config.providers.filter(
      (provider) => (!providerName || provider.name === providerName) && (!fixture || provider.fixture === fixture),
    ),
  };
  if (providerName && selectedConfig.providers.length === 0) {
    fail(`provider run selected no plan provider named ${providerName}`);
  }
  if (fixture && selectedConfig.providers.length === 0) {
    fail(`provider run selected no plan provider with fixture ${fixture}`);
  }
  if (forceFreshPerCase && forceBatchRungs) fail("cannot force both fresh and batched provider execution");
  const applyExecutionPolicy = (plannedCases) => plannedCases.map((planned) => {
    if (forceFreshPerCase) {
      return { ...planned, modelLoadPolicy: "fresh_per_case", modelLoadGroup: null };
    }
    if (forceBatchRungs) return {
      ...planned,
      modelLoadPolicy: "batch_rungs",
      modelLoadGroup: `forced-${digest({ backend: planned.backend, target: planned.target, fixture: planned.fixture })}`,
    };
    return planned;
  });
  const allExpanded = applyExecutionPolicy(expandPlan(selectedConfig));
  const expanded = applyExecutionPolicy(expandPlan(selectedConfig, existing.records));
  const selectedCases = backend ? expanded.filter((planned) => planned.backend === backend) : expanded;
  if (selectedCases.length === 0) fail(`provider run selected no ${backend ?? "remaining"} cases`);
  const backends = new Set(selectedCases.map((planned) => planned.backend));
  if (backends.size !== 1) {
    fail(`provider run must select exactly one backend; pass --backend mlx|candle (selected: ${[...backends].join(", ")})`);
  }
  const probe = JSON.parse(await executeProvider(
    providerCommand[0],
    providerCommand.slice(1),
    canonicalJson({ action: "probe", repositories }),
  ));
  await assertRepositoriesStable();
  // Completion remains an evidence-semantic decision: candidate and gated receipts cannot retire
  // plan cases or promote matrix cells. Resume has a narrower operational concern. A prior receipt
  // proves that its exact logical case was already attempted only when the harness, both repository
  // receipts (including matrix source identity/dirty state), and hardware probe all match this run.
  // Stale or foreign receipts therefore remain scheduled, while a failed multi-invocation capture
  // can continue without repeating expensive GPU work or colliding on a fresh capturedAt value.
  const attempted = operationallyAttemptedLogicalIds(existing.records, repositories, probe.hardware);
  const cases = selectedCases.filter((planned) => !attempted.has(planned.logicalCaseId));
  if (cases.length === 0) return existing;

  // Derive one digest per AUTHORITATIVE lane actually being captured, before the first GPU-bound
  // invocation. Only an authoritative capture can ever be `current` — `evidenceSemantics`
  // short-circuits fixture and candidate scopes before the comparison is reached — so a selection
  // that is entirely fixture/candidate derives nothing and needs neither an inference crate layout
  // nor a declarations file. That is what lets the schema-mutation suite drive this runner against a
  // synthetic repo. Deriving EAGERLY is the point of doing it here rather than lazily at stamping
  // time: an undeclared lane must fail before a 26 GB capture burns, not after.
  const digestByLane = new Map();
  for (const lane of new Set(cases.filter((planned) => planned.evidenceScope === "authoritative").map(laneOf))) {
    digestByLane.set(lane, await closureDigest(lane, repositories.inference.revision));
  }
  const repositoriesFor = (planned) => {
    const digest = digestByLane.get(laneOf(planned));
    if (!digest) return repositories;
    return { ...repositories, inference: { ...repositories.inference, closureDigest: digest } };
  };
  const incoming = [];
  const incomingSessions = [];
  let remaining = cases;
  const sameBatch = (left, right) =>
    left.modelLoadPolicy === "batch_rungs" &&
    right.modelLoadPolicy === "batch_rungs" &&
    left.modelLoadGroup &&
    left.modelLoadGroup === right.modelLoadGroup &&
    left.backend === right.backend &&
    equal(left.target, right.target) &&
    left.fixture === right.fixture;
  const requiredBatchRungs = (planned) => {
    const rungs = new Set(
      allExpanded
        .filter((candidate) => sameBatch(planned, candidate))
        .map((candidate) => candidate.strategy.rung),
    );
    return RUNGS.filter((rung) => rungs.has(rung));
  };
  while (remaining.length > 0) {
    const first = remaining[0];
    const batchRungs = new Set();
    const pendingBatch = first.modelLoadPolicy === "batch_rungs"
      ? remaining
          .filter((planned) => sameBatch(first, planned))
          .sort((left, right) => RUNGS.indexOf(left.strategy.rung) - RUNGS.indexOf(right.strategy.rung))
          .filter((planned) => {
            if (batchRungs.has(planned.strategy.rung)) return false;
            batchRungs.add(planned.strategy.rung);
            return true;
          })
      : [first];
    const pendingRungs = pendingBatch.map((planned) => planned.strategy.rung);
    // A provider's batch protocol is defined by the complete rung cohort, not by whichever
    // parameter cases happen to remain. Candidate/gated evidence intentionally cannot retire its
    // sibling sweep points, so a canonical five-rung Qwen batch leaves decode/window alternatives
    // pending. Sending those two rungs as a second `run_batch` violates the adapter contract and
    // used to discard the first successful batch. Measure an incomplete remainder serially instead;
    // this preserves gated semantics while ensuring every parameter point is still executed.
    const invocation = first.modelLoadPolicy === "batch_rungs" &&
        !equal(pendingRungs, requiredBatchRungs(first))
      ? [first]
      : pendingBatch;
    const uniqueRungs = new Set(invocation.map((planned) => planned.strategy.rung));
    if (first.modelLoadPolicy === "batch_rungs" && uniqueRungs.size !== invocation.length) {
      fail(`${first.modelLoadGroup}: a rung batch may contain only one pending case per rung`);
    }
    const action = invocation.length > 1 ? "run_batch" : "run";
    onProviderInvocation?.({ action, cases: invocation });
    const providerRequest = canonicalJson({
      action,
      ...(action === "run" ? { planned: first } : { planned: invocation }),
      repositories: repositoriesFor(first),
      repositoryPaths: { sceneWorks: sceneWorksRepo, inference: inferenceRepo },
      hardware: probe.hardware,
    });
    const providerOutput = await executeProvider(
      providerCommand[0],
      providerCommand.slice(1),
      providerRequest,
    );
    await assertRepositoriesStable();
    const response = JSON.parse(providerOutput);
    const fragments = action === "run_batch" ? response.fragments : [response];
    if (!Array.isArray(fragments) || fragments.length !== invocation.length) {
      fail(`${first.modelLoadGroup ?? first.logicalCaseId}: provider returned ${
        Array.isArray(fragments) ? fragments.length : "a non-array"
      } fragments for ${invocation.length} planned cases`);
    }
    if (action === "run_batch" && response.modelLoads !== 1) {
      fail(`${first.modelLoadGroup}: batched provider must attest exactly one model load`);
    }
    for (const [index, fragment] of fragments.entries()) {
      const planned = invocation[index];
      const sourceCapture = fragment.sourceCapture;
      delete fragment.sourceCapture;
      if (rawLogDir && !sourceCapture) {
        fail(`${planned.logicalCaseId}: configured raw-log provenance requires provider sourceCapture`);
      }
      if (!fragment.strategy || typeof fragment.strategy !== "object" || Array.isArray(fragment.strategy)) {
        fail(`${planned.logicalCaseId}: provider fragment.strategy must attest the executed strategy`);
      }
      validateEngagedRungs(fragment.strategy, `${planned.logicalCaseId}.provider strategy`);
      if (!equal(fragment.strategy, planned.strategy)) {
        fail(`${planned.logicalCaseId}: adapter measured strategy does not match planned strategy`);
      }
      // The materialization shape is a RECEIPT field: the adapter attests what its run actually
      // loaded under, and the plan only declares what that rung is expected to select. Taking
      // `planned.loadShape` here would stamp every record with the plan's claim and make the field
      // unfalsifiable — the same backfill sc-16482 forbids for historical receipts, applied
      // silently to new ones. Cross-check instead, and fail closed on divergence.
      if (!LOAD_SHAPES.includes(fragment.loadShape)) {
        fail(
          `${planned.logicalCaseId}: provider fragment must attest a loadShape (${LOAD_SHAPES.join("|")})`,
        );
      }
      if (fragment.loadShape !== planned.loadShape) {
        fail(
          `${planned.logicalCaseId}: adapter measured loadShape ${fragment.loadShape} but the plan ` +
            `declared ${planned.loadShape}`,
        );
      }
      const record = {
        ...fragment,
        logicalCaseId: planned.logicalCaseId,
        evidenceScope: planned.evidenceScope,
        backend: planned.backend,
        loadShape: fragment.loadShape,
        // Per-lane, not per-run: a `--backend candle` run with no `--fixture` selects several
        // providers, and stamping them all with one lane's digest would compare each against the
        // wrong code path forever.
        repositories: repositoriesFor(planned),
        hardware: probe.hardware,
        target: planned.target,
        strategy: fragment.strategy,
        ...(planned.sourceProvenance ? { sourceProvenance: planned.sourceProvenance } : {}),
        calibrationFingerprint: planned.calibrationFingerprint,
        fixture: planned.fixture,
        harnessVersion: HARNESS_VERSION,
      };
      if (sourceCapture) {
        if (!rawLogDir || !sourcePathPrefix) {
          fail(`${planned.logicalCaseId}: provider returned sourceCapture without configured raw-log provenance`);
        }
        if (action !== "run") {
          fail(`${planned.logicalCaseId}: physical source capture must run fresh per case`);
        }
        object(sourceCapture, `${planned.logicalCaseId}.sourceCapture`);
        if (sourceCapture.kind !== "physical_mlx") {
          fail(`${planned.logicalCaseId}: unsupported source capture kind ${JSON.stringify(sourceCapture.kind)}`);
        }
        if (!Array.isArray(sourceCapture.inputs) || sourceCapture.inputs.length === 0) {
          fail(`${planned.logicalCaseId}: physical MLX source capture requires exact inputs`);
        }
        validateExactOutputReceipts(
          sourceCapture.outputs,
          physicalMlxExpectedRoles(sourceCapture.outputs, false),
          `${planned.logicalCaseId}.sourceCapture.outputs`,
        );
        if (!Array.isArray(sourceCapture.claims) || sourceCapture.claims.length === 0) {
          fail(`${planned.logicalCaseId}: physical MLX source capture requires explicit claims`);
        }
        const baseInput = sourceCapture.inputs.find((input) => input?.role === "base");
        if (!baseInput || baseInput.repository !== fragment.artifact?.repository
            || baseInput.resolvedRevision !== fragment.artifact?.resolvedRevision
            || baseInput.variant !== fragment.artifact?.variant) {
          fail(`${planned.logicalCaseId}: physical MLX source input must match the measured artifact exactly`);
        }
        if (record.artifact.inventorySha256 !== undefined
            && record.artifact.inventorySha256 !== baseInput.sha256) {
          fail(`${planned.logicalCaseId}: provider artifact inventory disagrees with sourceCapture`);
        }
        record.artifact.inventorySha256 = baseInput.sha256;
        const stdoutSha256 = createHash("sha256").update(providerOutput).digest("hex");
        const sessionId = physicalMlxSessionId({
          kind: sourceCapture.kind,
          logicalCaseId: planned.logicalCaseId,
          capturedAt: fragment.capturedAt,
          repositories: repositoriesFor(planned),
          hardware: probe.hardware,
          stdoutSha256,
        });
        const sourcePath = `${sourcePathPrefix}/${sessionId}.log`;
        const receiptDir = path.join(rawLogDir, ...sourcePathPrefix.split("/"));
        await mkdir(receiptDir, { recursive: true });
        await writeImmutableReceipt(path.join(receiptDir, `${sessionId}.log`), providerOutput);
        const requestFileName = `${sessionId}.request.json`;
        await writeImmutableReceipt(path.join(receiptDir, requestFileName), providerRequest);
        const outputs = [{
          role: "request",
          path: `${sourcePathPrefix}/${requestFileName}`,
          sha256: createHash("sha256").update(providerRequest).digest("hex"),
          bytes: Buffer.byteLength(providerRequest),
        }];
        const avContents = new Map();
        for (const output of sourceCapture.outputs) {
          object(output, `${planned.logicalCaseId}.sourceCapture.outputs[]`);
          text(output.role, `${planned.logicalCaseId}.sourceCapture.outputs[].role`);
          text(output.path, `${planned.logicalCaseId}.sourceCapture.outputs[].path`);
          text(output.localPath, `${planned.logicalCaseId}.sourceCapture.outputs[].localPath`);
          const metadata = physicalMlxOutputMetadata(
            output,
            `${planned.logicalCaseId}.sourceCapture.outputs[${output.role}]`,
          );
          if (metadata.logicalCaseId !== planned.logicalCaseId
              || metadata.width !== planned.target.geometry.width
              || metadata.height !== planned.target.geometry.height
              || (metadata.frames !== undefined && metadata.frames !== planned.target.geometry.frames)) {
            fail(`${planned.logicalCaseId}: physical MLX provider output has the wrong logical case geometry`);
          }
          const outputRelative = path.posix.relative(sourcePathPrefix, output.path);
          if (!outputRelative || outputRelative.startsWith("../") || path.posix.isAbsolute(outputRelative)) {
            fail(`${planned.logicalCaseId}: physical MLX output path must stay under ${sourcePathPrefix}`);
          }
          if (!path.isAbsolute(output.localPath)) {
            fail(`${planned.logicalCaseId}: physical MLX local output path must be absolute`);
          }
          const physicalOutputPath = await realpath(output.localPath);
          const localRelative = path.relative(rawLogDir, physicalOutputPath);
          if (!localRelative || localRelative.startsWith(`..${path.sep}`) || path.isAbsolute(localRelative)) {
            fail(`${planned.logicalCaseId}: physical MLX local output must stay under the raw log directory`);
          }
          const bytes = await readFile(physicalOutputPath);
          const actualSha256 = createHash("sha256").update(bytes).digest("hex");
          if (actualSha256 !== output.sha256 || bytes.length !== output.bytes) {
            fail(`${planned.logicalCaseId}: physical MLX provider output differs from its provider attestation`);
          }
          if (output.role === "selected_av" || output.role === "reference_av") {
            const content = parsePhysicalMlxAvContent(
              bytes,
              `${planned.logicalCaseId}.sourceCapture.outputs[${output.role}]`,
            );
            const outputFps = record.diagnostics?.measurements?.find(
              (measurement) => measurement.name === "outputFps",
            )?.value;
            if (content.width !== planned.target.geometry.width
                || content.height !== planned.target.geometry.height
                || content.frames !== planned.target.geometry.frames
                || content.fps !== outputFps) {
              fail(`${planned.logicalCaseId}: canonical A/V header differs from measured geometry/FPS`);
            }
            avContents.set(output.role, content);
          }
          outputs.push({
            role: output.role,
            path: output.path,
            sha256: output.sha256,
            bytes: output.bytes,
          });
        }
        validatePhysicalMlxAvContentsAgainstRecord(record, avContents, planned.logicalCaseId);
        validateExactOutputReceipts(
          outputs,
          physicalMlxExpectedRoles(outputs, true),
          `${sessionId}.outputs`,
        );
        record.derivation = physicalMlxDerivation(sessionId);
        incomingSessions.push({
          id: sessionId,
          kind: sourceCapture.kind,
          command: JSON.stringify(providerCommand),
          sourcePath,
          capturedAt: fragment.capturedAt,
          repositories: repositoriesFor(planned),
          hardware: probe.hardware,
          target: {
            tier: planned.target.tier,
            mode: planned.target.mode,
            overlay: planned.target.overlay,
            ...(planned.target.transformerVariant
              ? { transformerVariant: planned.target.transformerVariant }
              : {}),
            ...(planned.target.decoder ? { decoder: planned.target.decoder } : {}),
            rung: planned.strategy.rung,
          },
          stdoutSha256,
          inputs: sourceCapture.inputs,
          outputs,
          claims: sourceCapture.claims,
          result: "passed",
        });
      }
      if (planned.expectedResult === "failed" && record.status !== "negative_complete") {
        fail(`${planned.logicalCaseId}: negative plan case must return status=negative_complete`);
      }
      if (planned.expectedResult === "passed" && record.status === "negative_complete") {
        fail(`${planned.logicalCaseId}: passing plan case returned a negative result`);
      }
      record.id = recordId(record);
      validateRecord(record);
      incoming.push(record);
    }
    const completed = new Set([...existing.records, ...incoming].flatMap(completedLogicalIds));
    const invoked = new Set(invocation.map((planned) => planned.logicalCaseId));
    remaining = remaining.filter(
      (planned) => !invoked.has(planned.logicalCaseId) && !completed.has(planned.logicalCaseId),
    );
    if (onProviderCheckpoint) {
      await onProviderCheckpoint(mergeBundles(existing, {
        schemaVersion: SCHEMA_VERSION,
        harnessVersion: HARNESS_VERSION,
        sourceSessions: incomingSessions,
        records: incoming,
      }));
    }
  }
  return mergeBundles(existing, {
    schemaVersion: SCHEMA_VERSION,
    harnessVersion: HARNESS_VERSION,
    sourceSessions: incomingSessions,
    records: incoming,
  });
}

async function readJson(file) {
  return JSON.parse(await readFile(path.resolve(ROOT, file), "utf8"));
}
async function liveInferenceClosureDigests() {
  const declarations = await readJson("config/inference-provider-closures.json");
  return Object.fromEntries(
    Object.entries(declarations.providers ?? {}).map(([provider, declaration]) => [
      provider,
      declaration.digest,
    ]),
  );
}
export async function atomicWrite(file, value) {
  const destination = path.resolve(ROOT, file);
  const temporary = `${destination}.tmp-${process.pid}`;
  await writeFile(temporary, canonicalJson(value));
  await rename(temporary, destination);
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  const value = (flag) => {
    const index = args.indexOf(flag);
    return index < 0 ? undefined : args[index + 1];
  };
  if (command === "check") {
    const closureDigests = await liveInferenceClosureDigests();
    return void await validateSourceSessionFiles(
      await readJson(value("--input")),
      value("--source-root") ? path.resolve(value("--source-root")) : null,
      closureDigests,
    );
  }
  if (command === "plan") {
    const resume = value("--resume") ? validateBundle(await readJson(value("--resume"))).records : [];
    return void await atomicWrite(value("--output"), { harnessVersion: HARNESS_VERSION, cases: expandPlan(await readJson(value("--config")), resume) });
  }
  if (command === "ingest") {
    const sourceRoot = value("--source-root") ? path.resolve(value("--source-root")) : null;
    const closureDigests = await liveInferenceClosureDigests();
    const incoming = await validateSourceSessionFiles(
      await readJson(value("--input")),
      sourceRoot,
      closureDigests,
    );
    const output = value("--resume")
      ? mergeBundles(
          await validateSourceSessionFiles(await readJson(value("--resume")), null, closureDigests),
          incoming,
        )
      : incoming;
    return void await atomicWrite(value("--output"), output);
  }
  if (command === "compare-reuse") {
    return void await atomicWrite(
      value("--output"),
      compareRungReuse(await readJson(value("--fresh")), await readJson(value("--reused"))),
    );
  }
  if (command === "assess-reuse") {
    return void await atomicWrite(value("--output"), await assessProviderReuse({
      config: await readJson(value("--config")),
      providerCommand: JSON.parse(value("--provider-command")),
      backend: value("--backend"),
      fixture: value("--fixture"),
    }));
  }
  if (command === "run") {
    const outputPath = value("--output");
    const output = await runProviderPlan({
      config: await readJson(value("--config")),
      providerCommand: JSON.parse(value("--provider-command")),
      sceneWorksRepo: path.resolve(value("--sceneworks-repo")),
      inferenceRepo: path.resolve(value("--inference-repo")),
      resume: value("--resume") ? await readJson(value("--resume")) : undefined,
      backend: value("--backend"),
      providerName: value("--provider"),
      fixture: value("--fixture"),
      forceFreshPerCase: args.includes("--fresh-per-case"),
      forceBatchRungs: args.includes("--batch-rungs"),
      rawLogDir: value("--raw-log-dir") ? path.resolve(value("--raw-log-dir")) : null,
      sourcePathPrefix: value("--source-path-prefix"),
      onProviderCheckpoint: (checkpoint) => atomicWrite(outputPath, checkpoint),
    });
    return void await atomicWrite(outputPath, output);
  }
  fail("usage: check|plan|ingest|assess-reuse|compare-reuse|run (see docs/memory-calibration-harness.md)");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
