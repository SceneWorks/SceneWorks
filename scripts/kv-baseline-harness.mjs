#!/usr/bin/env node

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { cp, mkdir, readFile, readdir, rename, rm, stat, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const SCHEMA_VERSION = 3;
export const HARNESS_VERSION = "sc-20671-kv-baseline-v3";
export const CONTRACT_PATH = "config/kv-baseline-quality-contract.json";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONTRACT_FILE = path.join(ROOT, CONTRACT_PATH);
const SCHEMA_FILE = path.join(ROOT, "packages/schemas/kv-baseline-receipt.schema.json");
const CONTRACT_RAW = readFileSync(CONTRACT_FILE);
const CONTRACT = JSON.parse(CONTRACT_RAW.toString("utf8"));
const PHASES = [
  "process-start",
  "weights-loaded",
  "prefill-peak",
  "first-token",
  "decode-steady",
  "prompt-cache-reuse",
  "cancellation-cleanup",
  "post-run-release",
];
const LIFECYCLE = [
  "append",
  "chunkedPrefill",
  "singleShotPrefill",
  "promptCacheReuse",
  "trim",
  "rollback",
  "clear",
  "cancel",
  "clone",
  "batchSplit",
  "batchMerge",
  "prefixCopyOnWrite",
  "pageImport",
  "pageExport",
  "serialization",
  "restore",
  "denseFallback",
  "postRunRelease",
];
const CONTEXT_BANDS = ["short", "medium", "memory-material", "fit-boundary"];
export const MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS = 1_000;
export const FIT_BOUNDARY_MIN_CONTEXT_BPS = 9_000;
const SCENEWORKS_REPOSITORY = "github.com/SceneWorks/SceneWorks";
const INFERENCE_REPOSITORY = "github.com/SceneWorks/inference";
const PMETAL_MLX_REPOSITORY = "https://github.com/michaeltrefry/mlx-rs";
const TIMING_FIELDS = [
  "loadMs",
  "prefillMs",
  "ttftMs",
  "firstTokenMs",
  "decodeTokensPerSecond",
  "coldCompileMs",
  "warmCompileMs",
];
const ERROR_QUALITY_FIELDS = ["parityMaxError", "perplexityDelta"];
const AGREEMENT_QUALITY_FIELDS = [
  "greedyTokenAgreement",
  "structuredToolAgreement",
  "needleRetrieval",
  "multiTurnPromptCache",
];
const FIXTURES = [
  "kernel-fp32-reference",
  "structured-tool-call",
  "long-context-needle",
  "multi-turn-prompt-cache",
];
const FIXTURE_SOURCES = Symbol("fixtureSources");

function fail(message) {
  throw new Error(`KV baseline receipt: ${message}`);
}

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, stable(value[key])]),
    );
  }
  return value;
}

export function canonicalJson(value) {
  return JSON.stringify(stable(value), null, 2);
}

export function sha256(value) {
  const input = typeof value === "string" || Buffer.isBuffer(value)
    ? value
    : canonicalJson(value);
  return createHash("sha256").update(input).digest("hex");
}

function object(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${name} must be an object`);
  }
  return value;
}

function exactKeys(value, allowed, name) {
  object(value, name);
  const unexpected = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unexpected.length) fail(`${name} has unexpected fields: ${unexpected.join(", ")}`);
}

function text(value, name) {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail(`${name} must be non-empty text`);
  }
  return value;
}

function finite(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${name} must be finite`);
  }
  return value;
}

function nonnegativeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 0) {
    fail(`${name} must be a non-negative safe integer`);
  }
  return value;
}

function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    fail(`${name} must be a positive safe integer`);
  }
  return value;
}

function checkedSum(values, name) {
  const total = values.reduce((sum, value) => sum + BigInt(value), 0n);
  if (total > BigInt(Number.MAX_SAFE_INTEGER)) fail(`${name} overflows safe integer accounting`);
  return Number(total);
}

function checkedProduct(values, name) {
  const total = values.reduce((product, value) => product * BigInt(value), 1n);
  if (total > BigInt(Number.MAX_SAFE_INTEGER)) fail(`${name} overflows safe integer accounting`);
  return Number(total);
}

function positiveNumber(value, name) {
  if (finite(value, name) <= 0) fail(`${name} must be positive`);
  return value;
}

function isoTimestamp(value, name) {
  text(value, name);
  if (!/^\d{4}-\d{2}-\d{2}T/.test(value) || Number.isNaN(Date.parse(value))) {
    fail(`${name} must be an ISO-8601 timestamp`);
  }
  return value;
}

function gitRevision(value, name) {
  if (typeof value !== "string" || !/^[0-9a-f]{40}$/.test(value)) {
    fail(`${name} must be an immutable 40-character git revision`);
  }
}

function digest(value, name) {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    fail(`${name} must be SHA-256`);
  }
}

function parseSidecar(sidecar, expectedName) {
  const fields = sidecar.trim().split(/\s+/);
  if (fields.length !== 2 || fields[1] !== expectedName) {
    fail(`invalid sidecar identity for ${expectedName}`);
  }
  digest(fields[0], `${expectedName} sidecar`);
  return fields[0];
}

function checkContract(contract) {
  exactKeys(contract, ["version", "thresholds", "statistics", "fixtures"], "contract");
  if (contract.version !== 2) fail("unsupported quality contract version");
  exactKeys(
    contract.thresholds,
    [...ERROR_QUALITY_FIELDS, ...AGREEMENT_QUALITY_FIELDS],
    "contract.thresholds",
  );
  for (const field of ERROR_QUALITY_FIELDS) {
    finite(contract.thresholds[field], `contract.thresholds.${field}`);
  }
  for (const field of AGREEMENT_QUALITY_FIELDS) {
    const value = finite(contract.thresholds[field], `contract.thresholds.${field}`);
    if (value < 0 || value > 1) fail(`contract.thresholds.${field} must be in [0,1]`);
  }
  exactKeys(
    contract.statistics,
    [
      "repeats",
      "warmups",
      "confidenceInterval",
      "outlierPolicy",
      "variancePolicy",
      "maxCoefficientOfVariation",
    ],
    "contract.statistics",
  );
  positiveInteger(contract.statistics.repeats, "contract.statistics.repeats");
  nonnegativeInteger(contract.statistics.warmups, "contract.statistics.warmups");
  for (const field of ["confidenceInterval", "outlierPolicy", "variancePolicy"]) {
    text(contract.statistics[field], `contract.statistics.${field}`);
  }
  const maxCv = finite(
    contract.statistics.maxCoefficientOfVariation,
    "contract.statistics.maxCoefficientOfVariation",
  );
  if (maxCv <= 0 || maxCv >= 1) fail("contract maxCoefficientOfVariation must be in (0,1)");
  if (canonicalJson(contract.fixtures) !== canonicalJson(FIXTURES)) {
    fail("quality contract fixture surface mismatch");
  }
}

const CONTRACT_RAW_HASH = sha256(CONTRACT_RAW);
const CONTRACT_SIDECAR_HASH = parseSidecar(
  readFileSync(`${CONTRACT_FILE}.sha256`, "utf8"),
  path.basename(CONTRACT_FILE),
);
if (CONTRACT_RAW_HASH !== CONTRACT_SIDECAR_HASH) {
  throw new Error("KV baseline receipt: quality contract sidecar does not match raw bytes");
}
checkContract(CONTRACT);
export const QUALITY_CONTRACT_HASH = CONTRACT_RAW_HASH;

const schemaAjv = new Ajv2020({ allErrors: true, strict: true, strictTypes: false });
addFormats(schemaAjv);
const validateReceiptSchema = schemaAjv.compile(
  JSON.parse(readFileSync(SCHEMA_FILE, "utf8")),
);

function footprintBytes(value) {
  const match = String(value).trim().match(/^([0-9]+(?:[.][0-9]+)?)[ \t]*(B|KB|MB|GB)$/i);
  if (!match) fail(`invalid phys_footprint value ${value}`);
  const bytes = Number(match[1]) * ({ B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3 })[
    match[2].toUpperCase()
  ];
  if (!Number.isSafeInteger(bytes)) fail("phys_footprint is not an exact safe integer");
  return bytes;
}

export function readDarwinMemory(
  pid = process.pid,
  runner = execFileSync,
  phase = "process-start",
  timestamp = new Date().toISOString(),
) {
  if (process.platform !== "darwin") {
    return { supported: false, reason: "Darwin phys_footprint is unavailable" };
  }
  positiveInteger(pid, "pid");
  text(phase, "phase");
  isoTimestamp(timestamp, "timestamp");
  let output;
  try {
    output = runner("footprint", ["-p", String(pid)], { encoding: "utf8" });
  } catch (error) {
    fail(`cannot read Darwin phys_footprint: ${error.message}`);
  }
  const currentMatch = String(output).match(
    /phys_footprint[ \t]*:[ \t]*([0-9]+(?:[.][0-9]+)?[ \t]*(?:KB|MB|GB|B)(?![A-Za-z]))/i,
  );
  const peakMatch = String(output).match(
    /phys_footprint_peak[ \t]*:[ \t]*([0-9]+(?:[.][0-9]+)?[ \t]*(?:KB|MB|GB|B)(?![A-Za-z]))/i,
  );
  if (!currentMatch || !peakMatch) {
    fail("footprint output omitted unit-bearing phys_footprint or phys_footprint_peak");
  }
  const physFootprintBytes = footprintBytes(currentMatch[1]);
  const physFootprintPeakBytes = footprintBytes(peakMatch[1]);
  if (physFootprintPeakBytes < physFootprintBytes) {
    fail("phys_footprint_peak is below phys_footprint");
  }
  return {
    phase,
    pid,
    source: "footprint -p",
    timestamp,
    physFootprintBytes,
    physFootprintPeakBytes,
  };
}

function validateAllocationEvent(event, index) {
  const name = `memory.allocationEvents[${index}]`;
  exactKeys(event, ["kind", "role", "lifetime", "phase", "timestamp", "bytes"], name);
  text(event.kind, `${name}.kind`);
  if (!["cache", "attention-workspace", "weights", "output"].includes(event.role)) {
    fail(`${name}.role is unsupported`);
  }
  if (!["persistent", "transient", "released"].includes(event.lifetime)) {
    fail(`${name}.lifetime is unsupported`);
  }
  if (event.lifetime === "released"
    && (event.role !== "cache" || event.kind !== "product-cache_release")) {
    fail(`${name} is not a cache release lifecycle event`);
  }
  if (!PHASES.includes(event.phase)) fail(`${name}.phase is unsupported`);
  isoTimestamp(event.timestamp, `${name}.timestamp`);
  positiveInteger(event.bytes, `${name}.bytes`);
}

export function detectFullCacheTemporary(events, denseBytes) {
  if (!Array.isArray(events) || events.length === 0) fail("typed allocation events are required");
  positiveInteger(denseBytes, "denseTheoreticalKvBytes");
  const thresholdBytes = Number((BigInt(denseBytes) * 9n + 9n) / 10n);
  const witnesses = [];
  events.forEach((event, index) => {
    validateAllocationEvent(event, index);
    const relevantRole = event.role === "cache" || event.role === "attention-workspace" || event.role === "output";
    const explicit = event.kind === "dense_cache_temporary"
      || event.kind === "full_cache_materialization";
    if (event.lifetime === "transient" && explicit) {
      witnesses.push(event);
    }
    if (event.lifetime === "transient" && relevantRole) {
      if (event.bytes >= thresholdBytes) witnesses.push(event);
    }
  });
  return { detected: witnesses.length > 0, thresholdBytes, witnesses };
}

function maxRoleBytes(events, role, lifetime) {
  return Math.max(0, ...events
    .filter((event) => event.role === role && event.lifetime === lifetime)
    .map((event) => event.bytes));
}

function maxClassifiedTransientBytes(events) {
  return Math.max(0, ...events
    .filter((event) => event.lifetime === "transient"
      && ["cache", "attention-workspace", "output"].includes(event.role))
    .map((event) => event.bytes));
}

function validateCacheRelease(events) {
  let liveCache;
  let releases = 0;
  for (const event of events) {
    if (event.role === "cache" && event.lifetime === "persistent") {
      liveCache = event.bytes;
    } else if (event.lifetime === "released") {
      if (liveCache !== event.bytes) fail("cache release bytes do not match retained KV ownership");
      liveCache = undefined;
      releases += 1;
      if (!Number.isSafeInteger(releases)) fail("cache release event count overflow");
    }
  }
  if (releases === 0 || liveCache !== undefined) {
    fail("persistent KV ownership was not explicitly released");
  }
}

function validatePhaseSample(sample, index, expectedPid) {
  const name = `memory.phaseSamples[${index}]`;
  exactKeys(
    sample,
    ["phase", "pid", "source", "timestamp", "physFootprintBytes", "physFootprintPeakBytes", "mlx"],
    name,
  );
  if (sample.phase !== PHASES[index]) fail(`${name}.phase must be ${PHASES[index]}`);
  positiveInteger(sample.pid, `${name}.pid`);
  if (expectedPid !== undefined && sample.pid !== expectedPid) fail("phase samples span multiple PIDs");
  if (sample.source !== "footprint -p") fail(`${name}.source must be footprint -p`);
  isoTimestamp(sample.timestamp, `${name}.timestamp`);
  nonnegativeInteger(sample.physFootprintBytes, `${name}.physFootprintBytes`);
  nonnegativeInteger(sample.physFootprintPeakBytes, `${name}.physFootprintPeakBytes`);
  if (sample.physFootprintPeakBytes < sample.physFootprintBytes) {
    fail(`${name}.physFootprintPeakBytes is below physFootprintBytes`);
  }
  exactKeys(sample.mlx, ["source", "activeBytes", "cacheBytes", "peakBytes"], `${name}.mlx`);
  if (sample.mlx.source !== "mlx_rs::memory") fail(`${name}.mlx.source must be mlx_rs::memory`);
  for (const field of ["activeBytes", "cacheBytes", "peakBytes"]) {
    nonnegativeInteger(sample.mlx[field], `${name}.mlx.${field}`);
  }
  if (sample.mlx.peakBytes < sample.mlx.activeBytes) fail(`${name}.mlx.peakBytes is below activeBytes`);
  return sample.pid;
}

function timingMean(samples, field) {
  return samples.reduce((sum, sample) => sum + sample[field], 0) / samples.length;
}

function timingVariance(samples, field, mean) {
  return samples.reduce((sum, sample) => sum + (sample[field] - mean) ** 2, 0) / samples.length;
}

function nearlyEqual(actual, expected) {
  return Math.abs(actual - expected) <= Math.max(1e-9, Math.abs(expected) * 1e-9);
}

function validateTimingSample(sample, index) {
  const name = `timings.samples[${index}]`;
  exactKeys(sample, TIMING_FIELDS, name);
  for (const field of TIMING_FIELDS) positiveNumber(sample[field], `${name}.${field}`);
}

function validateFixtureEvidence(evidence) {
  exactKeys(evidence, FIXTURES, "quality.fixtureEvidence");
  for (const fixture of FIXTURES) {
    const row = evidence[fixture];
    exactKeys(row, ["passed", "artifactName", "artifactSha256", "artifactSidecarSha256", "independentReference"], `quality.fixtureEvidence.${fixture}`);
    if (row.passed !== true) fail(`quality fixture ${fixture} did not pass`);
    if (row.artifactName !== `fixtures/${fixture}.json`) fail(`quality fixture ${fixture} artifact name mismatch`);
    digest(row.artifactSha256, `quality.fixtureEvidence.${fixture}.artifactSha256`);
    digest(row.artifactSidecarSha256, `quality.fixtureEvidence.${fixture}.artifactSidecarSha256`);
    if (row.artifactSidecarSha256 !== sha256(`${row.artifactSha256}  ${row.artifactName}\n`)) {
      fail(`quality fixture ${fixture} artifact sidecar binding mismatch`);
    }
    text(row.independentReference, `quality.fixtureEvidence.${fixture}.independentReference`);
  }
}

function contextBandTarget(contextWindowTokens, contextBand) {
  positiveInteger(contextWindowTokens, "geometry.contextWindowTokens");
  if (contextWindowTokens < 1_024) fail("context window is below the frozen minimum");
  const medium = Math.min(1_024, Math.max(128, Math.floor(contextWindowTokens / 16)));
  const memoryMaterial = Math.floor(contextWindowTokens / 4);
  const fitBoundary = Math.max(
    contextWindowTokens - 512,
    Math.ceil(contextWindowTokens * FIT_BOUNDARY_MIN_CONTEXT_BPS / 10_000),
  );
  const targets = { short: 32, medium, "memory-material": memoryMaterial, "fit-boundary": fitBoundary };
  if (!(32 < medium && medium < memoryMaterial && memoryMaterial < fitBoundary)) {
    fail("context window cannot represent four distinct frozen bands");
  }
  return targets[contextBand];
}

function receiptCore(receipt) {
  return Object.fromEntries(Object.entries(receipt).filter(([key]) => key !== "receiptSha256"));
}

export function validateReceipt(receipt, { verifyHash = true } = {}) {
  if (!validateReceiptSchema(receipt)) {
    fail(`schema validation failed: ${schemaAjv.errorsText(validateReceiptSchema.errors)}`);
  }
  exactKeys(
    receipt,
    [
      "schemaVersion", "harnessVersion", "runId", "capturedAt", "mode", "status",
      "contractHash", "receiptSha256", "provenance", "matrix", "geometry", "memory",
      "timings", "quality", "lifecycle", "cancellation", "warmup",
    ],
    "receipt",
  );
  if (receipt.schemaVersion !== SCHEMA_VERSION || receipt.harnessVersion !== HARNESS_VERSION) {
    fail("unsupported schema or harness version");
  }
  text(receipt.runId, "receipt.runId");
  isoTimestamp(receipt.capturedAt, "receipt.capturedAt");
  if (!["dense", "compressed"].includes(receipt.mode) || receipt.status !== "complete") {
    fail("receipt must be complete and have dense or compressed mode");
  }
  if (receipt.contractHash !== QUALITY_CONTRACT_HASH) fail("quality contract hash mismatch");
  if (verifyHash) {
    digest(receipt.receiptSha256, "receipt.receiptSha256");
    if (receipt.receiptSha256 !== sha256(receiptCore(receipt))) fail("receiptSha256 mismatch");
  }

  exactKeys(
    receipt.provenance,
    [
      "sceneWorksRepository", "inferenceRepository", "sceneWorksRevision", "inferenceRevision", "mlxVersion", "mlxSource", "mlxRevision", "dependencyLockSha256",
      "os", "xcode", "hardware", "modelId", "modelFileSha256", "modelFileBytes", "powerMode",
      "referenceModelId", "referenceModelSha256", "referenceModelBytes",
      "thermalState", "commandTemplate", "command", "campaignSessionId", "campaignCacheStateVersion", "coordinateOperationSha256",
    ],
    "provenance",
  );
  gitRevision(receipt.provenance.sceneWorksRevision, "provenance.sceneWorksRevision");
  gitRevision(receipt.provenance.inferenceRevision, "provenance.inferenceRevision");
  if (receipt.provenance.sceneWorksRepository !== SCENEWORKS_REPOSITORY
    || receipt.provenance.inferenceRepository !== INFERENCE_REPOSITORY) {
    fail("provenance repository identity is not the paired SceneWorks repositories");
  }
  for (const field of [
    "mlxVersion", "mlxSource", "mlxRevision", "os", "xcode", "hardware", "modelId",
    "referenceModelId", "powerMode", "thermalState",
    "commandTemplate", "command",
  ]) {
    text(receipt.provenance[field], `provenance.${field}`);
  }
  digest(receipt.provenance.dependencyLockSha256, "provenance.dependencyLockSha256");
  gitRevision(receipt.provenance.mlxRevision, "provenance.mlxRevision");
  if (receipt.provenance.mlxSource
    !== `git+${PMETAL_MLX_REPOSITORY}?rev=${receipt.provenance.mlxRevision}#${receipt.provenance.mlxRevision}`) {
    fail("MLX dependency source does not bind its exact Git revision");
  }
  digest(receipt.provenance.modelFileSha256, "provenance.modelFileSha256");
  digest(receipt.provenance.referenceModelSha256, "provenance.referenceModelSha256");
  digest(receipt.provenance.campaignSessionId, "provenance.campaignSessionId");
  digest(receipt.provenance.coordinateOperationSha256, "provenance.coordinateOperationSha256");
  positiveInteger(receipt.provenance.modelFileBytes, "provenance.modelFileBytes");
  positiveInteger(receipt.provenance.referenceModelBytes, "provenance.referenceModelBytes");
  positiveInteger(receipt.provenance.campaignCacheStateVersion, "provenance.campaignCacheStateVersion");
  if (receipt.provenance.thermalState !== "nominal") {
    fail("thermal state is not nominal");
  }
  if (!receipt.provenance.commandTemplate.includes("{mode}")
    || receipt.provenance.command
      !== receipt.provenance.commandTemplate.replaceAll("{mode}", receipt.mode)) {
    fail("command must be the mode-substitution of commandTemplate");
  }

  exactKeys(
    receipt.matrix,
    ["family", "contextBand", "requestMode", "prefillMode", "processTemperature"],
    "matrix",
  );
  if (!["llama", "qwen"].includes(receipt.matrix.family)
    || !CONTEXT_BANDS.includes(receipt.matrix.contextBand)
    || !["single", "supported-batch"].includes(receipt.matrix.requestMode)
    || !["chunked", "single-shot"].includes(receipt.matrix.prefillMode)
    || !["cold", "warm"].includes(receipt.matrix.processTemperature)) {
    fail("receipt matrix coordinate is incomplete");
  }

  exactKeys(
    receipt.geometry,
    ["batch", "queryHeads", "kvHeads", "headDimension", "queryLength", "kvLength", "layers", "elementBytes", "capacity", "contextWindowTokens", "contextTargetTokens", "contextPayloadTokens"],
    "geometry",
  );
  for (const field of Object.keys(receipt.geometry)) positiveInteger(receipt.geometry[field], `geometry.${field}`);
  if (receipt.geometry.contextTargetTokens
      !== contextBandTarget(receipt.geometry.contextWindowTokens, receipt.matrix.contextBand)
    || receipt.geometry.contextPayloadTokens > receipt.geometry.contextTargetTokens
    || receipt.geometry.contextPayloadTokens < Math.floor(receipt.geometry.contextTargetTokens / 2)) {
    fail("context payload token count is outside its producer-measured band");
  }
  if (receipt.geometry.queryHeads % receipt.geometry.kvHeads !== 0) {
    fail("geometry queryHeads must be divisible by kvHeads");
  }
  if (receipt.geometry.capacity < receipt.geometry.kvLength) fail("geometry capacity is below kvLength");
  if ((receipt.matrix.requestMode === "single" && receipt.geometry.batch !== 1)
    || (receipt.matrix.requestMode === "supported-batch" && receipt.geometry.batch <= 1)) {
    fail("matrix requestMode disagrees with geometry.batch");
  }

  exactKeys(
    receipt.memory,
    [
      "modelWeightsBytes", "persistentKvBytes", "transientWorkspaceBytes",
      "denseTheoreticalKvBytes", "phaseSamples", "allocationEvents", "reconciliation", "release",
    ],
    "memory",
  );
  positiveInteger(receipt.memory.modelWeightsBytes, "memory.modelWeightsBytes");
  positiveInteger(receipt.memory.persistentKvBytes, "memory.persistentKvBytes");
  nonnegativeInteger(receipt.memory.transientWorkspaceBytes, "memory.transientWorkspaceBytes");
  positiveInteger(receipt.memory.denseTheoreticalKvBytes, "memory.denseTheoreticalKvBytes");
  if (!Array.isArray(receipt.memory.phaseSamples) || receipt.memory.phaseSamples.length !== PHASES.length) {
    fail(`exactly ${PHASES.length} phase samples are required`);
  }
  let workerPid;
  let priorTimestamp = -Infinity;
  let priorFootprintPeak = -Infinity;
  let priorMlxPeak = -Infinity;
  receipt.memory.phaseSamples.forEach((sample, index) => {
    workerPid = validatePhaseSample(sample, index, workerPid);
    const timestamp = Date.parse(sample.timestamp);
    if (timestamp <= priorTimestamp) fail("phase sample timestamps are not strictly increasing");
    if (sample.physFootprintPeakBytes < priorFootprintPeak) {
      fail("phys_footprint_peak decreased within one worker process");
    }
    if (sample.mlx.peakBytes < priorMlxPeak) {
      fail("MLX peak memory decreased without a declared reset boundary");
    }
    if (sample.physFootprintBytes < sample.mlx.activeBytes) {
      fail("process physical footprint is below MLX live tensor bytes");
    }
    priorTimestamp = timestamp;
    priorFootprintPeak = sample.physFootprintPeakBytes;
    priorMlxPeak = sample.mlx.peakBytes;
  });
  if (!Array.isArray(receipt.memory.allocationEvents) || receipt.memory.allocationEvents.length === 0) {
    fail("memory allocation events are required");
  }
  receipt.memory.allocationEvents.forEach(validateAllocationEvent);
  validateCacheRelease(receipt.memory.allocationEvents);
  if (receipt.mode === "compressed" && detectFullCacheTemporary(
    receipt.memory.allocationEvents,
    receipt.memory.denseTheoreticalKvBytes,
  ).detected) fail("full-cache temporary detected");
  if (maxRoleBytes(receipt.memory.allocationEvents, "cache", "persistent")
      !== receipt.memory.persistentKvBytes
    || maxRoleBytes(receipt.memory.allocationEvents, "weights", "persistent")
      !== receipt.memory.modelWeightsBytes
    || maxClassifiedTransientBytes(receipt.memory.allocationEvents)
      !== receipt.memory.transientWorkspaceBytes) {
    fail("typed allocation events do not reconcile with memory attribution totals");
  }
  const weightsLoaded = receipt.memory.phaseSamples[1];
  const prefillPeak = receipt.memory.phaseSamples[2];
  const decodeSteady = receipt.memory.phaseSamples[4];
  const prefillAttributedBytes = checkedSum([
    receipt.memory.modelWeightsBytes,
    receipt.memory.persistentKvBytes,
    receipt.memory.transientWorkspaceBytes,
  ], "prefill attributed memory");
  const decodeAttributedBytes = checkedSum([
    receipt.memory.modelWeightsBytes,
    receipt.memory.persistentKvBytes,
  ], "decode attributed memory");
  if (weightsLoaded.mlx.activeBytes < receipt.memory.modelWeightsBytes
    || prefillPeak.mlx.activeBytes < prefillAttributedBytes
    || prefillPeak.mlx.peakBytes < prefillAttributedBytes
    || decodeSteady.mlx.activeBytes < decodeAttributedBytes) {
    fail("MLX live/peak memory does not contain the attributed weights, KV, and workspace bytes");
  }

  exactKeys(
    receipt.memory.reconciliation,
    ["expectedDenseKvBytes", "observedPersistentKvBytes", "toleranceBytes"],
    "memory.reconciliation",
  );
  for (const field of ["expectedDenseKvBytes", "observedPersistentKvBytes", "toleranceBytes"]) {
    nonnegativeInteger(receipt.memory.reconciliation[field], `memory.reconciliation.${field}`);
  }
  const expectedDenseKvBytes = checkedProduct([
    receipt.geometry.batch,
    receipt.geometry.layers,
    receipt.geometry.kvHeads,
    receipt.geometry.capacity,
    receipt.geometry.headDimension,
    receipt.geometry.elementBytes,
    2,
  ], "dense KV byte attribution");
  if (receipt.memory.denseTheoreticalKvBytes !== expectedDenseKvBytes
    || receipt.memory.reconciliation.expectedDenseKvBytes !== expectedDenseKvBytes
    || receipt.memory.reconciliation.observedPersistentKvBytes !== receipt.memory.persistentKvBytes) {
    fail("dense KV byte attribution does not reconcile with geometry");
  }
  if (receipt.matrix.contextBand === "memory-material"
    && expectedDenseKvBytes * 10_000
      < prefillPeak.physFootprintBytes * MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS) {
    fail("memory-material dense KV is below the frozen process-footprint share");
  }
  if (receipt.matrix.contextBand === "fit-boundary"
    && (receipt.geometry.capacity > receipt.geometry.contextWindowTokens
      || receipt.geometry.capacity * 10_000
        < receipt.geometry.contextWindowTokens * FIT_BOUNDARY_MIN_CONTEXT_BPS)) {
    fail("fit-boundary cache occupancy is below the frozen admission ratio");
  }
  if (receipt.mode === "dense"
    && Math.abs(receipt.memory.persistentKvBytes - expectedDenseKvBytes)
      > receipt.memory.reconciliation.toleranceBytes) {
    fail("dense persistent KV bytes are outside the reconciliation tolerance");
  }

  exactKeys(
    receipt.memory.release,
    ["verified", "physFootprintToleranceBytes", "mlxActiveToleranceBytes", "mlxCacheToleranceBytes"],
    "memory.release",
  );
  if (receipt.memory.release.verified !== true) fail("post-run release is not verified");
  for (const field of ["physFootprintToleranceBytes", "mlxActiveToleranceBytes", "mlxCacheToleranceBytes"]) {
    nonnegativeInteger(receipt.memory.release[field], `memory.release.${field}`);
  }
  const start = receipt.memory.phaseSamples[0];
  const released = receipt.memory.phaseSamples.at(-1);
  if (released.physFootprintBytes > start.physFootprintBytes + receipt.memory.release.physFootprintToleranceBytes
    || released.mlx.activeBytes > start.mlx.activeBytes + receipt.memory.release.mlxActiveToleranceBytes
    || released.mlx.cacheBytes > start.mlx.cacheBytes + receipt.memory.release.mlxCacheToleranceBytes) {
    fail("post-run footprint or MLX allocator state did not return within tolerance");
  }

  exactKeys(receipt.timings, [...TIMING_FIELDS, "samples", "summary"], "timings");
  for (const field of TIMING_FIELDS) positiveNumber(receipt.timings[field], `timings.${field}`);
  if (!Array.isArray(receipt.timings.samples)
    || receipt.timings.samples.length !== CONTRACT.statistics.repeats) {
    fail("raw timing sample count differs from the frozen repeat policy");
  }
  receipt.timings.samples.forEach(validateTimingSample);
  for (const field of TIMING_FIELDS) {
    const mean = timingMean(receipt.timings.samples, field);
    if (!nearlyEqual(receipt.timings[field], mean)) fail(`timings.${field} does not derive from raw samples`);
  }
  exactKeys(
    receipt.timings.summary,
    [
      "decodeTokensPerSecondMean", "decodeTokensPerSecondP95",
      "decodeTokensPerSecondVariance", "decodeTokensPerSecondCoefficientOfVariation",
      "confidenceIntervalLow", "confidenceIntervalHigh",
    ],
    "timings.summary",
  );
  for (const field of Object.keys(receipt.timings.summary)) {
    finite(receipt.timings.summary[field], `timings.summary.${field}`);
  }
  const decodeMean = timingMean(receipt.timings.samples, "decodeTokensPerSecond");
  const decodeVariance = timingVariance(receipt.timings.samples, "decodeTokensPerSecond", decodeMean);
  const sortedDecode = receipt.timings.samples
    .map((sample) => sample.decodeTokensPerSecond)
    .sort((left, right) => left - right);
  const decodeP95 = sortedDecode[Math.ceil(sortedDecode.length * 0.95) - 1];
  const coefficient = Math.sqrt(decodeVariance) / decodeMean;
  if (!nearlyEqual(receipt.timings.summary.decodeTokensPerSecondMean, decodeMean)
    || !nearlyEqual(receipt.timings.summary.decodeTokensPerSecondP95, decodeP95)
    || !nearlyEqual(receipt.timings.summary.decodeTokensPerSecondVariance, decodeVariance)
    || !nearlyEqual(receipt.timings.summary.decodeTokensPerSecondCoefficientOfVariation, coefficient)) {
    fail("timing summary does not derive from raw samples");
  }
  if (receipt.timings.summary.confidenceIntervalLow > decodeMean
    || receipt.timings.summary.confidenceIntervalHigh < decodeMean
    || receipt.timings.summary.confidenceIntervalLow > receipt.timings.summary.confidenceIntervalHigh) {
    fail("timing confidence interval does not contain the decode mean");
  }
  if (coefficient > CONTRACT.statistics.maxCoefficientOfVariation) {
    fail("dense baseline variance exceeds the frozen band");
  }

  exactKeys(
    receipt.quality,
    [...ERROR_QUALITY_FIELDS, ...AGREEMENT_QUALITY_FIELDS, "statistics", "fixtureEvidence"],
    "quality",
  );
  for (const field of ERROR_QUALITY_FIELDS) finite(receipt.quality[field], `quality.${field}`);
  if (receipt.quality.parityMaxError < 0) fail("quality.parityMaxError must be non-negative");
  for (const field of AGREEMENT_QUALITY_FIELDS) {
    const value = finite(receipt.quality[field], `quality.${field}`);
    if (value < 0 || value > 1) fail(`quality.${field} must be in [0,1]`);
  }
  exactKeys(
    receipt.quality.statistics,
    [
      "repeats", "warmups", "confidenceInterval", "outlierPolicy", "variancePolicy",
      "maxCoefficientOfVariation",
    ],
    "quality.statistics",
  );
  if (canonicalJson(receipt.quality.statistics) !== canonicalJson(CONTRACT.statistics)) {
    fail("quality statistics policy differs from the frozen contract");
  }
  if (receipt.quality.parityMaxError > CONTRACT.thresholds.parityMaxError
    || receipt.quality.perplexityDelta > CONTRACT.thresholds.perplexityDelta) {
    fail("quality error exceeds the frozen maximum");
  }
  for (const field of AGREEMENT_QUALITY_FIELDS) {
    if (receipt.quality[field] < CONTRACT.thresholds[field]) {
      fail(`quality.${field} is below the frozen minimum`);
    }
  }
  validateFixtureEvidence(receipt.quality.fixtureEvidence);

  const allowedLifecycleKeys = [...LIFECYCLE, ...LIFECYCLE.map((field) => `${field}FallbackReason`)];
  exactKeys(receipt.lifecycle, allowedLifecycleKeys, "lifecycle");
  for (const field of LIFECYCLE) {
    if (typeof receipt.lifecycle[field] !== "boolean") fail(`lifecycle.${field} must be boolean`);
    if (!receipt.lifecycle[field]) {
      text(receipt.lifecycle[`${field}FallbackReason`], `lifecycle.${field}FallbackReason`);
    } else if (`${field}FallbackReason` in receipt.lifecycle) {
      fail(`lifecycle.${field}FallbackReason is present for a passing operation`);
    }
  }
  exactKeys(receipt.cancellation, ["cleanupVerified"], "cancellation");
  if (receipt.cancellation.cleanupVerified !== true) fail("cancellation cleanup is not verified");
  exactKeys(
    receipt.warmup,
    ["required", "completed", "workerPid", "suiteSha256", "sessionId", "cacheStateVersion"],
    "warmup",
  );
  positiveInteger(receipt.warmup.workerPid, "warmup.workerPid");
  nonnegativeInteger(receipt.warmup.cacheStateVersion, "warmup.cacheStateVersion");
  const warmRequired = receipt.matrix.processTemperature === "warm";
  if (receipt.warmup.required !== warmRequired
    || (warmRequired && (!receipt.warmup.completed
      || !/^[0-9a-f]{64}$/.test(receipt.warmup.suiteSha256)
      || receipt.warmup.sessionId !== receipt.provenance.campaignSessionId
      || receipt.warmup.cacheStateVersion === 0
      || receipt.warmup.cacheStateVersion > receipt.provenance.campaignCacheStateVersion))
    || (!warmRequired && (receipt.warmup.completed || receipt.warmup.suiteSha256 !== ""
      || receipt.warmup.sessionId !== "" || receipt.warmup.cacheStateVersion !== 0))) {
    fail("warmup session/cache-state evidence is inconsistent with the coordinate");
  }
  return receipt;
}

export function buildReceipt(input) {
  const receipt = {
    ...input,
    schemaVersion: SCHEMA_VERSION,
    harnessVersion: HARNESS_VERSION,
    contractHash: QUALITY_CONTRACT_HASH,
  };
  delete receipt.receiptSha256;
  receipt.receiptSha256 = sha256(receipt);
  return validateReceipt(receipt);
}

async function sha256File(file) {
  const hash = createHash("sha256");
  await new Promise((resolve, reject) => {
    const stream = createReadStream(file);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolve);
  });
  return hash.digest("hex");
}

export async function inventoryModelArtifact(artifactPath) {
  let metadata;
  try {
    metadata = await stat(artifactPath);
  } catch (error) {
    fail(`model artifact is unavailable: ${error.message}`);
  }
  if (metadata.isFile()) {
    if (metadata.size <= 0) fail("model artifact must be non-empty");
    return { bytes: metadata.size, sha256: await sha256File(artifactPath), files: 1 };
  }
  if (!metadata.isDirectory()) fail("model artifact must be a file or snapshot directory");

  const root = path.resolve(artifactPath);
  const files = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const absolute = path.join(directory, entry.name);
      const resolved = await stat(absolute);
      if (resolved.isDirectory()) {
        if (entry.isSymbolicLink()) fail(`model snapshot contains a directory symlink: ${absolute}`);
        await visit(absolute);
        continue;
      }
      if (!resolved.isFile() || resolved.size <= 0) {
        fail(`model snapshot contains an empty or unsupported entry: ${absolute}`);
      }
      files.push({
        path: path.relative(root, absolute).split(path.sep).join("/"),
        bytes: resolved.size,
        sha256: await sha256File(absolute),
      });
    }
  }
  await visit(root);
  if (files.length === 0) fail("model snapshot is empty");
  files.sort((left, right) => left.path.localeCompare(right.path));
  const bytes = files.reduce((total, file) => total + file.bytes, 0);
  if (!Number.isSafeInteger(bytes)) fail("model snapshot byte count exceeds safe integer range");
  const hash = createHash("sha256");
  for (const file of files) {
    hash.update(file.path);
    hash.update("\0");
    hash.update(String(file.bytes));
    hash.update("\0");
    hash.update(file.sha256);
    hash.update("\n");
  }
  return { bytes, sha256: hash.digest("hex"), files: files.length };
}

export async function buildVerifiedReceipt(input) {
  object(input?.provenance, "provenance");
  const modelFilePath = text(input.provenance.modelFilePath, "provenance.modelFilePath");
  const inventory = await inventoryModelArtifact(modelFilePath);
  if (input.provenance.modelFileBytes !== inventory.bytes) fail("model artifact byte count mismatch");
  if (inventory.sha256 !== input.provenance.modelFileSha256) {
    fail("model artifact SHA-256 mismatch");
  }
  const verifiedInput = structuredClone(input);
  delete verifiedInput.provenance.modelFilePath;
  object(verifiedInput.quality?.fixtureEvidence, "quality.fixtureEvidence");
  for (const fixture of FIXTURES) {
    const sourceRow = object(input.quality?.fixtureEvidence?.[fixture], `quality.fixtureEvidence.${fixture}`);
    const artifactPath = text(
      sourceRow.artifactPath,
      `quality.fixtureEvidence.${fixture}.artifactPath`,
    );
    let artifactMetadata;
    try {
      artifactMetadata = await stat(artifactPath);
    } catch (error) {
      fail(`quality fixture ${fixture} artifact is unavailable: ${error.message}`);
    }
    if (!artifactMetadata.isFile() || artifactMetadata.size <= 0) {
      fail(`quality fixture ${fixture} artifact must be a non-empty file`);
    }
    const artifactSha256 = await sha256File(artifactPath);
    if (artifactSha256 !== sourceRow.artifactSha256) {
      fail(`quality fixture ${fixture} artifact SHA-256 mismatch`);
    }
    let artifact;
    try {
      artifact = JSON.parse(await readFile(artifactPath, "utf8"));
    } catch (error) {
      fail(`quality fixture ${fixture} artifact is not valid JSON: ${error.message}`);
    }
    validateFixtureArtifact(artifact, fixture, sourceRow);
    const artifactName = `fixtures/${fixture}.json`;
    verifiedInput.quality.fixtureEvidence[fixture] = {
      passed: sourceRow.passed,
      artifactName,
      artifactSha256,
      artifactSidecarSha256: sha256(`${artifactSha256}  ${artifactName}\n`),
      independentReference: sourceRow.independentReference,
    };
  }
  const receipt = buildReceipt(verifiedInput);
  Object.defineProperty(receipt, FIXTURE_SOURCES, {
    value: Object.fromEntries(FIXTURES.map((fixture) => [fixture, input.quality.fixtureEvidence[fixture].artifactPath])),
    enumerable: false,
  });
  return receipt;
}

export function validateFixtureArtifact(artifact, fixture, sourceRow) {
  const keys = ["fixture", "independentReference", "evidence", "metrics"];
  if (Object.hasOwn(artifact, "binding")) keys.push("binding");
  exactKeys(artifact, keys, `fixture artifact ${fixture}`);
  if (artifact.fixture !== fixture) fail(`fixture artifact name mismatch for ${fixture}`);
  if (artifact.independentReference !== sourceRow.independentReference) {
    fail(`fixture artifact reference mismatch for ${fixture}`);
  }
  object(artifact.evidence, `fixture artifact ${fixture}.evidence`);
  object(artifact.metrics, `fixture artifact ${fixture}.metrics`);
  if (Object.hasOwn(artifact, "binding")) object(artifact.binding, `fixture artifact ${fixture}.binding`);
  for (const field of ["parityMaxError", "perplexityDelta", "greedyTokenAgreement", "structuredToolAgreement", "needleRetrieval", "multiTurnPromptCache"]) {
    if (typeof artifact.metrics[field] !== "number" || !Number.isFinite(artifact.metrics[field])) {
      fail(`fixture artifact ${fixture}.metrics.${field} must be finite`);
    }
  }
  const requiredEvidence = {
    "kernel-fp32-reference": ["candidatePerplexity", "referencePerplexity", "parityErrors", "greedyMatches", "greedyTotal"],
    "structured-tool-call": ["matches", "total"],
    "long-context-needle": ["matches", "total"],
    "multi-turn-prompt-cache": ["matches", "total"],
  }[fixture];
  exactKeys(artifact.evidence, requiredEvidence, `fixture artifact ${fixture}.evidence`);
  if (fixture === "kernel-fp32-reference") {
    for (const field of ["candidatePerplexity", "referencePerplexity", "greedyMatches", "greedyTotal"]) {
      if (typeof artifact.evidence[field] !== "number" || !Number.isFinite(artifact.evidence[field])) fail(`fixture artifact ${fixture}.${field} must be finite`);
    }
    if (!Array.isArray(artifact.evidence.parityErrors) || artifact.evidence.parityErrors.length === 0
      || artifact.evidence.parityErrors.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
      fail(`fixture artifact ${fixture}.parityErrors must contain finite values`);
    }
  } else {
    for (const field of ["matches", "total"]) positiveInteger(artifact.evidence[field], `fixture artifact ${fixture}.${field}`);
    if (artifact.evidence.matches > artifact.evidence.total) fail(`fixture artifact ${fixture} matches exceed total`);
  }
}

async function removeIfPresent(file) {
  try {
    await unlink(file);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

async function writeSealedText(file, bytes) {
  const hash = sha256(bytes);
  const nonce = `${process.pid}-${randomUUID()}`;
  const temporaryFile = `${file}.tmp-${nonce}`;
  const sidecarFile = `${file}.sha256`;
  const temporarySidecar = `${sidecarFile}.tmp-${nonce}`;
  try {
    await writeFile(temporaryFile, bytes, { flag: "wx" });
    await writeFile(temporarySidecar, `${hash}  ${path.basename(file)}\n`, { flag: "wx" });
    await rename(temporaryFile, file);
    await rename(temporarySidecar, sidecarFile);
  } finally {
    await removeIfPresent(temporaryFile);
    await removeIfPresent(temporarySidecar);
  }
  return hash;
}

export async function writeSealedJson(file, value) {
  return writeSealedText(file, `${canonicalJson(value)}\n`);
}

function humanPath(jsonPath) {
  return jsonPath.endsWith(".json") ? `${jsonPath.slice(0, -5)}.md` : `${jsonPath}.md`;
}

export function renderReceiptMarkdown(receipt) {
  validateReceipt(receipt);
  const peak = Math.max(...receipt.memory.phaseSamples.map((sample) => sample.physFootprintPeakBytes));
  const supportedLifecycle = LIFECYCLE.filter((field) => receipt.lifecycle[field]).length;
  const fallbackLifecycle = LIFECYCLE
    .filter((field) => !receipt.lifecycle[field])
    .map((field) => `${field}: ${receipt.lifecycle[`${field}FallbackReason`]}`);
  const releasedCacheBytes = maxRoleBytes(receipt.memory.allocationEvents, "cache", "released");
  return `# ${receipt.mode === "dense" ? "Dense" : "Compressed"} KV receipt\n\n`
    + `- Run: ${receipt.runId}\n`
    + `- Mode: ${receipt.mode}\n`
    + `- Captured: ${receipt.capturedAt}\n`
    + `- Model: ${receipt.provenance.modelId} (${receipt.provenance.modelFileSha256})\n`
    + `- SceneWorks: ${receipt.provenance.sceneWorksRevision}\n`
    + `- Inference: ${receipt.provenance.inferenceRevision}\n`
    + `- Matrix: ${Object.values(receipt.matrix).join(" / ")}\n`
    + `- Persistent KV bytes: ${receipt.memory.persistentKvBytes}\n`
    + `- Released cache ownership bytes: ${releasedCacheBytes}\n`
    + `- Theoretical dense KV bytes: ${receipt.memory.denseTheoreticalKvBytes}\n`
    + `- Process footprint peak bytes: ${peak}\n`
    + `- Decode throughput: ${receipt.timings.decodeTokensPerSecond} tok/s\n`
    + `- TTFT: ${receipt.timings.ttftMs} ms\n`
    + `- Quality contract: ${receipt.contractHash}\n`
    + `- Receipt hash: ${receipt.receiptSha256}\n`
    + `- Lifecycle checks: ${supportedLifecycle}/${LIFECYCLE.length} supported\n`
    + `- Cancellation cleanup: verified\n`
    + (fallbackLifecycle.length ? `- Explicit dense fallbacks: ${fallbackLifecycle.join("; ")}\n` : "");
}

function assertMarkdownBound(markdown, receipt) {
  if (!markdown.includes(`- Receipt hash: ${receipt.receiptSha256}\n`)) {
    fail("human receipt is not bound to the sealed JSON receipt");
  }
}

export function renderComparisonMarkdown(comparison) {
  return `# Dense/compressed KV comparison\n\n`
    + `- Dense run: ${comparison.denseRunId}\n`
    + `- Compressed run: ${comparison.compressedRunId}\n`
    + `- Persistent KV reduction: ${(comparison.persistentKvReduction * 100).toFixed(2)}%\n`
    + `- Decode steady footprint delta: ${comparison.decodeSteadyPhysFootprintDeltaBytes} bytes\n`
    + `- Process footprint peak delta: ${comparison.peakPhysFootprintDeltaBytes} bytes\n`
    + `- Decode MLX live/cache deltas: ${comparison.decodeSteadyMlxActiveDeltaBytes} / ${comparison.decodeSteadyMlxCacheDeltaBytes} bytes\n`
    + `- MLX peak delta: ${comparison.mlxPeakDeltaBytes} bytes\n`
    + `- Decode throughput ratio: ${comparison.decodeThroughputRatio.toFixed(6)}\n`
    + `- Quality contract: ${comparison.contractHash}\n`;
}

export async function readSealedJson(file) {
  const bytes = await readFile(file, "utf8");
  const expected = parseSidecar(await readFile(`${file}.sha256`, "utf8"), path.basename(file));
  if (sha256(bytes) !== expected) fail(`sidecar hash mismatch for ${file}`);
  return JSON.parse(bytes);
}

async function readSealedText(file) {
  const bytes = await readFile(file, "utf8");
  const expected = parseSidecar(await readFile(`${file}.sha256`, "utf8"), path.basename(file));
  if (sha256(bytes) !== expected) fail(`sidecar hash mismatch for ${file}`);
  return bytes;
}

/** Publish JSON, human receipt, and both sidecars as one directory rename. */
export async function writeReceiptSet(directory, receipt) {
  validateReceipt(receipt);
  if (!receipt[FIXTURE_SOURCES]) fail("published receipt requires exact fixture source bytes");
  const parent = path.dirname(directory);
  const base = path.basename(directory);
  const staging = path.join(parent, `.${base}.staging-${process.pid}-${randomUUID()}`);
  try {
    await mkdir(staging, { recursive: false });
    const json = `${canonicalJson(receipt)}\n`;
    const markdown = `${renderReceiptMarkdown(receipt)}\n`;
    assertMarkdownBound(markdown, receipt);
    await Promise.all([
      writeFile(path.join(staging, "receipt.json"), json, { flag: "wx" }),
      writeFile(path.join(staging, "receipt.md"), markdown, { flag: "wx" }),
      writeFile(path.join(staging, "receipt.json.sha256"), `${sha256(json)}  receipt.json\n`, { flag: "wx" }),
      writeFile(path.join(staging, "receipt.md.sha256"), `${sha256(markdown)}  receipt.md\n`, { flag: "wx" }),
    ]);
    await mkdir(path.join(staging, "fixtures"), { recursive: false });
    for (const fixture of FIXTURES) {
      const source = receipt[FIXTURE_SOURCES][fixture];
      const bytes = await readFile(source);
      const name = `${fixture}.json`;
      const artifactName = `fixtures/${name}`;
      if (sha256(bytes) !== receipt.quality.fixtureEvidence[fixture].artifactSha256) {
        fail(`fixture ${fixture} changed before publication`);
      }
      const sidecar = `${sha256(bytes)}  ${artifactName}\n`;
      if (sha256(sidecar) !== receipt.quality.fixtureEvidence[fixture].artifactSidecarSha256) {
        fail(`fixture ${fixture} sidecar changed before publication`);
      }
      await writeFile(path.join(staging, "fixtures", name), bytes, { flag: "wx" });
      await writeFile(path.join(staging, "fixtures", `${name}.sha256`), sidecar, { flag: "wx" });
    }
    await rename(staging, directory);
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

export async function readReceiptSet(directory) {
  const receipt = await readSealedJson(path.join(directory, "receipt.json"));
  const markdown = await readSealedText(path.join(directory, "receipt.md"));
  assertMarkdownBound(markdown, receipt);
  const fixtureDirectory = path.join(directory, "fixtures");
  for (const fixture of FIXTURES) {
    const file = path.join(fixtureDirectory, `${fixture}.json`);
    const bytes = await readFile(file);
    const sidecar = await readFile(`${file}.sha256`, "utf8");
    const row = receipt.quality.fixtureEvidence[fixture];
    if (sha256(bytes) !== row.artifactSha256
      || parseSidecar(sidecar, row.artifactName) !== row.artifactSha256
      || sha256(sidecar) !== row.artifactSidecarSha256) {
      fail(`published fixture ${fixture} is not bound to the receipt`);
    }
    let artifact;
    try {
      artifact = JSON.parse(bytes.toString("utf8"));
    } catch (error) {
      fail(`published fixture ${fixture} is not valid JSON: ${error.message}`);
    }
    validateFixtureArtifact(artifact, fixture, row);
  }
  return receipt;
}

function phaseByName(receipt, phase) {
  return receipt.memory.phaseSamples.find((sample) => sample.phase === phase);
}

export function compareReceipts(dense, compressed) {
  validateReceipt(dense);
  validateReceipt(compressed);
  if (dense.mode !== "dense" || compressed.mode !== "compressed") {
    fail("comparison requires dense then compressed receipts");
  }
  for (const field of [
    "sceneWorksRepository", "inferenceRepository", "sceneWorksRevision", "inferenceRevision",
    "mlxVersion", "mlxSource", "mlxRevision", "dependencyLockSha256",
    "os", "xcode", "hardware", "modelId", "modelFileSha256", "modelFileBytes", "powerMode",
    "referenceModelId", "referenceModelSha256", "referenceModelBytes", "thermalState", "commandTemplate",
  ]) {
    if (dense.provenance[field] !== compressed.provenance[field]) {
      fail(`comparison identity differs at provenance.${field}`);
    }
  }
  if (dense.contractHash !== compressed.contractHash
    || canonicalJson(dense.matrix) !== canonicalJson(compressed.matrix)
    || canonicalJson(dense.geometry) !== canonicalJson(compressed.geometry)) {
    fail("comparison matrix, geometry, or contract differs");
  }
  const denseDecode = phaseByName(dense, "decode-steady");
  const compressedDecode = phaseByName(compressed, "decode-steady");
  const densePeak = Math.max(...dense.memory.phaseSamples.map((sample) => sample.physFootprintPeakBytes));
  const compressedPeak = Math.max(...compressed.memory.phaseSamples.map((sample) => sample.physFootprintPeakBytes));
  return {
    schemaVersion: SCHEMA_VERSION,
    complete: true,
    denseRunId: dense.runId,
    compressedRunId: compressed.runId,
    contractHash: dense.contractHash,
    persistentKvReduction:
      (dense.memory.persistentKvBytes - compressed.memory.persistentKvBytes)
      / dense.memory.persistentKvBytes,
    decodeSteadyPhysFootprintDeltaBytes:
      compressedDecode.physFootprintBytes - denseDecode.physFootprintBytes,
    peakPhysFootprintDeltaBytes: compressedPeak - densePeak,
    decodeSteadyMlxActiveDeltaBytes:
      compressedDecode.mlx.activeBytes - denseDecode.mlx.activeBytes,
    decodeSteadyMlxCacheDeltaBytes:
      compressedDecode.mlx.cacheBytes - denseDecode.mlx.cacheBytes,
    mlxPeakDeltaBytes:
      Math.max(...compressed.memory.phaseSamples.map((sample) => sample.mlx.peakBytes))
      - Math.max(...dense.memory.phaseSamples.map((sample) => sample.mlx.peakBytes)),
    decodeThroughputRatio:
      compressed.timings.decodeTokensPerSecond / dense.timings.decodeTokensPerSecond,
    quality: Object.fromEntries(
      [...ERROR_QUALITY_FIELDS, ...AGREEMENT_QUALITY_FIELDS]
        .map((field) => [field, compressed.quality[field]]),
    ),
  };
}

export function validateCampaign(receipts) {
  if (!Array.isArray(receipts) || receipts.length === 0) fail("campaign has no sealed receipts");
  const coordinates = new Set();
  const familyModels = new Map();
  let campaignIdentity;
  for (const receipt of receipts) {
    validateReceipt(receipt);
    if (receipt.mode !== "dense") fail("dense baseline campaign contains a non-dense receipt");
    const coordinate = [
      receipt.matrix.family, receipt.matrix.contextBand, receipt.matrix.requestMode,
      receipt.matrix.prefillMode, receipt.matrix.processTemperature,
    ].join("/");
    if (coordinates.has(coordinate)) fail(`duplicate campaign coordinate ${coordinate}`);
    coordinates.add(coordinate);
    const globalIdentity = canonicalJson(Object.fromEntries([
      "sceneWorksRepository", "inferenceRepository", "sceneWorksRevision", "inferenceRevision",
      "mlxVersion", "mlxSource", "mlxRevision", "dependencyLockSha256", "os", "xcode",
      "hardware", "powerMode", "thermalState", "commandTemplate",
    ].map((field) => [field, receipt.provenance[field]])));
    if (campaignIdentity && campaignIdentity !== globalIdentity) {
      fail("campaign source, dependency, toolchain, hardware, or power identity drift");
    }
    campaignIdentity ??= globalIdentity;
    const identity = canonicalJson({
      modelId: receipt.provenance.modelId,
      modelFileSha256: receipt.provenance.modelFileSha256,
      modelFileBytes: receipt.provenance.modelFileBytes,
      referenceModelId: receipt.provenance.referenceModelId,
      referenceModelSha256: receipt.provenance.referenceModelSha256,
      referenceModelBytes: receipt.provenance.referenceModelBytes,
      queryHeads: receipt.geometry.queryHeads,
      kvHeads: receipt.geometry.kvHeads,
      headDimension: receipt.geometry.headDimension,
      layers: receipt.geometry.layers,
      elementBytes: receipt.geometry.elementBytes,
      contextWindowTokens: receipt.geometry.contextWindowTokens,
    });
    const prior = familyModels.get(receipt.matrix.family);
    if (prior && prior !== identity) fail(`model identity drift within ${receipt.matrix.family}`);
    familyModels.set(receipt.matrix.family, identity);
  }
  const missing = [];
  for (const family of ["llama", "qwen"]) {
    for (const band of CONTEXT_BANDS) {
      for (const request of ["single", "supported-batch"]) {
        for (const prefill of ["chunked", "single-shot"]) {
          for (const temperature of ["cold", "warm"]) {
            const coordinate = [family, band, request, prefill, temperature].join("/");
            if (!coordinates.has(coordinate)) missing.push(coordinate);
          }
        }
      }
    }
  }
  if (missing.length) fail(`campaign incomplete; missing ${missing.length} coordinates`);
  for (const family of ["llama", "qwen"]) {
    const bands = new Map(receipts
      .filter((receipt) => receipt.matrix.family === family)
      .map((receipt) => [receipt.matrix.contextBand, receipt.geometry.contextPayloadTokens]));
    if (bands.size !== CONTEXT_BANDS.length
      || CONTEXT_BANDS.some((band, index) => index > 0
        && bands.get(band) <= bands.get(CONTEXT_BANDS[index - 1]))) {
      fail(`context-band tokenizer measurements are not strictly increasing for ${family}`);
    }
  }
  return {
    schemaVersion: SCHEMA_VERSION,
    complete: true,
    receipts: receipts.length,
    coordinates: coordinates.size,
    contractHash: QUALITY_CONTRACT_HASH,
  };
}

/**
 * Publish a complete matrix as one directory transaction.  Child receipt sets are read and
 * verified before copying; the public destination remains absent if a worker crashes, a sidecar
 * drifts, or any of the required 64 coordinates is missing.  This deliberately supersedes a
 * loose manifest file, which could otherwise describe a mixed generation of receipt directories.
 */
export async function writeCampaignSet(directory, receiptSetDirectories) {
  if (!Array.isArray(receiptSetDirectories) || receiptSetDirectories.length !== 64) {
    fail("complete campaign publication requires exactly 64 receipt-set directories");
  }
  const receipts = await Promise.all(receiptSetDirectories.map((source) => readReceiptSet(source)));
  const summary = validateCampaign(receipts);
  const coordinates = receipts.map((receipt) => [
    receipt.matrix.family, receipt.matrix.contextBand, receipt.matrix.requestMode,
    receipt.matrix.prefillMode, receipt.matrix.processTemperature,
  ].join("-"));
  if (new Set(coordinates).size !== 64) fail("complete campaign has duplicate coordinate directories");
  const parent = path.dirname(directory);
  const base = path.basename(directory);
  const staging = path.join(parent, `.${base}.campaign-staging-${process.pid}-${randomUUID()}`);
  try {
    await mkdir(staging, { recursive: false });
    for (let index = 0; index < receipts.length; index += 1) {
      await cp(receiptSetDirectories[index], path.join(staging, coordinates[index]), {
        recursive: true,
        errorOnExist: true,
        force: false,
      });
    }
    const manifest = {
      ...summary,
      kind: "sc-20671-complete-coordinate-set",
      coordinateReceipts: receipts.map((receipt, index) => ({
        coordinate: coordinates[index], receiptSha256: receipt.receiptSha256,
        workerPid: receipt.memory.phaseSamples[0].pid,
      })),
    };
    await Promise.all([
      writeFile(path.join(staging, "campaign.json"), `${canonicalJson(manifest)}\n`, { flag: "wx" }),
      writeFile(path.join(staging, "campaign.json.sha256"), `${sha256(`${canonicalJson(manifest)}\n`)}  campaign.json\n`, { flag: "wx" }),
    ]);
    await rename(staging, directory);
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

export async function cancellationSafe(work, cleanup, signal) {
  try {
    if (signal?.aborted) fail("cancelled before start");
    return await work(signal);
  } finally {
    await cleanup();
  }
}

function usage() {
  console.error(
    "usage: kv-baseline-harness.mjs record <input> <receipt-set-directory> | "
      + "compare <dense-set> <compressed-set> <comparison> | campaign <receipt-set-directory> <manifest>",
  );
}

async function main() {
  const [command, first, second, third, ...extra] = process.argv.slice(2);
  if (extra.length) fail("too many CLI arguments");
  if (command === "record" && first && second && !third) {
    const input = JSON.parse(await readFile(first, "utf8"));
    const receipt = await buildVerifiedReceipt(input);
    await writeReceiptSet(second, receipt);
    return;
  }
  if (command === "compare" && first && second && third) {
    const dense = await readReceiptSet(first);
    const compressed = await readReceiptSet(second);
    const comparison = compareReceipts(dense, compressed);
    await writeSealedText(humanPath(third), `${renderComparisonMarkdown(comparison)}\n`);
    await writeSealedJson(third, comparison);
    return;
  }
  if (command === "campaign" && first && second && !third) {
    const entries = await readdir(first, { withFileTypes: true });
    const sets = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort();
    const receipts = await Promise.all(sets.map((name) => readReceiptSet(path.join(first, name))));
    await writeSealedJson(second, validateCampaign(receipts));
    return;
  }
  usage();
  process.exitCode = 2;
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
