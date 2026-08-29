#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import os from "node:os";

export const SCHEMA_VERSION = 1;
export const HARNESS_VERSION = "sc-20671-kv-baseline-v1";

const fail = (message) => { throw new Error(`KV baseline receipt: ${message}`); };
const object = (value, name) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${name} must be an object`);
  return value;
};
const positive = (value, name) => {
  if (!Number.isSafeInteger(value) || value < 0) fail(`${name} must be a non-negative safe integer`);
  return value;
};
const text = (value, name) => {
  if (typeof value !== "string" || value.length === 0) fail(`${name} must be non-empty text`);
  return value;
};

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stable(value[key])]));
  }
  return value;
}
export function canonicalJson(value) { return JSON.stringify(stable(value), null, 2); }
export function sha256(value) { return createHash("sha256").update(typeof value === "string" ? value : canonicalJson(value)).digest("hex"); }

export function readDarwinMemory(pid = process.pid, runner = execFileSync) {
  if (process.platform !== "darwin") return { supported: false, reason: "Darwin phys_footprint is unavailable on this platform" };
  try {
    const output = runner("footprint", ["-p", String(pid)], { encoding: "utf8" });
    const match = output.match(/phys_footprint\s*:\s*([0-9]+)/i);
    if (!match) fail("footprint output omitted phys_footprint");
    return { supported: true, bytes: Number(match[1]), source: "footprint -p" };
  } catch (error) { fail(`cannot read Darwin phys_footprint: ${error.message}`); }
}

export function detectFullCacheTemporary(events, denseCacheBytes) {
  if (!Array.isArray(events)) fail("memoryEvents must be an array");
  positive(denseCacheBytes, "denseCacheBytes");
  const threshold = Math.max(1, Math.floor(denseCacheBytes * 0.9));
  const witnesses = events.filter((event) => event?.kind === "dense_cache_temporary"
    || event?.materializedFullCache === true
    || (event?.kind === "allocation" && Number.isSafeInteger(event.bytes) && event.bytes >= threshold));
  return { detected: witnesses.length > 0, thresholdBytes: threshold, witnesses };
}

export function validateReceipt(receipt) {
  object(receipt, "receipt");
  if (receipt.schemaVersion !== SCHEMA_VERSION) fail("unsupported schemaVersion");
  if (receipt.harnessVersion !== HARNESS_VERSION) fail("unsupported harnessVersion");
  for (const key of ["runId", "capturedAt", "mode", "status"]) text(receipt[key], `receipt.${key}`);
  if (!["dense", "compressed"].includes(receipt.mode)) fail("mode must be dense or compressed");
  if (receipt.status !== "complete") fail("incomplete runs are not ingestible");
  object(receipt.provenance, "provenance");
  for (const key of ["sceneWorksRevision", "inferenceRevision", "mlxRevision", "os", "xcode", "hardware", "modelId", "modelFileSha256", "powerMode", "thermalState", "command"]) text(receipt.provenance[key], `provenance.${key}`);
  if (!/^[0-9a-f]{64}$/.test(receipt.provenance.modelFileSha256)) fail("provenance.modelFileSha256 must be SHA-256");
  object(receipt.geometry, "geometry");
  for (const key of ["batch", "queryHeads", "kvHeads", "headDimension", "queryLength", "kvLength"]) positive(receipt.geometry[key], `geometry.${key}`);
  object(receipt.memory, "memory");
  for (const key of ["modelWeightsBytes", "persistentKvBytes", "transientWorkspaceBytes", "denseTheoreticalKvBytes"]) positive(receipt.memory[key], `memory.${key}`);
  object(receipt.memory.physFootprint, "memory.physFootprint");
  positive(receipt.memory.physFootprint.beforeBytes, "physFootprint.beforeBytes");
  positive(receipt.memory.physFootprint.afterBytes, "physFootprint.afterBytes");
  object(receipt.memory.mlxAllocator, "memory.mlxAllocator");
  for (const key of ["activeBytes", "cacheBytes", "peakBytes"]) positive(receipt.memory.mlxAllocator[key], `mlxAllocator.${key}`);
  object(receipt.timings, "timings");
  for (const key of ["loadMs", "prefillMs", "firstTokenMs", "decodeTokensPerSecond", "coldCompileMs", "warmCompileMs"]) positive(receipt.timings[key], `timings.${key}`);
  object(receipt.quality, "quality");
  for (const key of ["parityMaxError", "perplexityDelta", "greedyAgreement"])
    if (typeof receipt.quality[key] !== "number" || receipt.quality[key] < 0) fail(`quality.${key} must be non-negative number`);
  object(receipt.quality.contract, "quality.contract");
  for (const key of ["parityMaxErrorThreshold", "perplexityDeltaThreshold", "greedyAgreementThreshold", "repeats", "warmups"])
    positive(receipt.quality.contract[key], `quality.contract.${key}`);
  if (!Array.isArray(receipt.lifecycle) || receipt.lifecycle.length === 0) fail("lifecycle checks are required");
  if (receipt.fullCacheTemporary?.detected !== false) fail("full-cache temporary detection must be explicitly false");
  if (receipt.cancellation?.cleanupVerified !== true) fail("cancellation cleanup must be verified");
  return receipt;
}

export function buildReceipt(input) {
  object(input, "input");
  const receipt = { ...input, schemaVersion: SCHEMA_VERSION, harnessVersion: HARNESS_VERSION };
  receipt.fullCacheTemporary ??= detectFullCacheTemporary(input.memoryEvents ?? [], input.memory?.denseTheoreticalKvBytes ?? 0);
  validateReceipt(receipt);
  return { ...receipt, receiptSha256: sha256(receipt) };
}

export function compareReceipts(dense, compressed) {
  validateReceipt(dense); validateReceipt(compressed);
  if (dense.mode !== "dense" || compressed.mode !== "compressed") fail("comparison requires dense then compressed receipts");
  if (dense.provenance.sceneWorksRevision !== compressed.provenance.sceneWorksRevision
    || dense.provenance.inferenceRevision !== compressed.provenance.inferenceRevision
    || JSON.stringify(dense.geometry) !== JSON.stringify(compressed.geometry)) fail("inputs or geometry differ");
  const reduction = dense.memory.persistentKvBytes === 0 ? 0
    : (dense.memory.persistentKvBytes - compressed.memory.persistentKvBytes) / dense.memory.persistentKvBytes;
  return { schemaVersion: SCHEMA_VERSION, denseRunId: dense.runId, compressedRunId: compressed.runId,
    persistentKvReduction: reduction, physFootprintDeltaBytes: compressed.memory.physFootprint.afterBytes - dense.memory.physFootprint.afterBytes,
    decodeThroughputRatio: dense.timings.decodeTokensPerSecond === 0 ? 0 : compressed.timings.decodeTokensPerSecond / dense.timings.decodeTokensPerSecond,
    quality: { parityMaxError: compressed.quality.parityMaxError, perplexityDelta: compressed.quality.perplexityDelta,
      greedyAgreement: compressed.quality.greedyAgreement },
    comparable: true, comparisonSha256: sha256({ denseRunId: dense.runId, compressedRunId: compressed.runId, reduction }) };
}

export async function cancellationSafe(work, cleanup, signal) {
  if (signal?.aborted) fail("cancelled before start");
  try { return await work(signal); }
  finally { await cleanup(); }
}

function usage() { console.error("usage: kv-baseline-harness.mjs record <input.json> <receipt.json> | compare <dense.json> <compressed.json> <comparison.json>"); }
if (import.meta.url === `file://${process.argv[1]}`) {
  const [command, first, second, third] = process.argv.slice(2);
  if (!command) { usage(); process.exitCode = 2; }
  else if (command === "record") writeFile(second, `${canonicalJson(buildReceipt(JSON.parse(await readFile(first, "utf8"))))}\n`);
  else if (command === "compare") writeFile(third, `${canonicalJson(compareReceipts(JSON.parse(await readFile(first, "utf8")), JSON.parse(await readFile(second, "utf8"))))}\n`);
  else { usage(); process.exitCode = 2; }
}
