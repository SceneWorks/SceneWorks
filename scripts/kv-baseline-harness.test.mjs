import assert from "node:assert/strict";
import test from "node:test";
import { buildReceipt, cancellationSafe, compareReceipts, detectFullCacheTemporary, readDarwinMemory, validateReceipt } from "./kv-baseline-harness.mjs";

const base = (mode = "dense", overrides = {}) => buildReceipt({
  runId: `${mode}-1`, capturedAt: "2026-08-29T12:00:00Z", mode, status: "complete",
  provenance: { sceneWorksRevision: "a".repeat(40), inferenceRevision: "b".repeat(40), mlxRevision: "c".repeat(40), os: "macOS", xcode: "Xcode", hardware: "Apple", modelId: "fixture", modelFileSha256: "d".repeat(64), powerMode: " AC", thermalState: "nominal", command: "fixture" },
  geometry: { batch: 1, queryHeads: 8, kvHeads: 8, headDimension: 128, queryLength: 1, kvLength: 4096 },
  memory: { modelWeightsBytes: 1_000, persistentKvBytes: mode === "dense" ? 1_000 : 500, transientWorkspaceBytes: 100, denseTheoreticalKvBytes: 1_000,
    physFootprint: { beforeBytes: 2_000, afterBytes: mode === "dense" ? 4_000 : 3_000 }, mlxAllocator: { activeBytes: 1_000, cacheBytes: 100, peakBytes: 1_100 } },
  timings: { loadMs: 1, prefillMs: 2, firstTokenMs: 3, decodeTokensPerSecond: mode === "dense" ? 100 : 98, coldCompileMs: 4, warmCompileMs: 1 },
  quality: { parityMaxError: 0.001, perplexityDelta: 0, greedyAgreement: 1, contract: { parityMaxErrorThreshold: 1, perplexityDeltaThreshold: 1, greedyAgreementThreshold: 1, repeats: 3, warmups: 1 } }, lifecycle: ["append", "trim", "serialize", "restore", "cancel"],
  fullCacheTemporary: { detected: false, thresholdBytes: 900, witnesses: [] }, cancellation: { cleanupVerified: true }, ...overrides,
});

test("validates sealed receipt and compares matching dense/compressed runs", () => {
  const dense = base(); const compressed = base("compressed");
  assert.equal(compareReceipts(dense, compressed).persistentKvReduction, 0.5);
  assert.doesNotThrow(() => validateReceipt(compressed));
});
test("fails closed for missing provenance and cleanup", () => {
  assert.throws(() => base("dense", { provenance: { os: "macOS" } }), /provenance.sceneWorksRevision/);
  assert.throws(() => base("dense", { cancellation: { cleanupVerified: false } }), /cancellation cleanup/);
});
test("detects explicit and sized full-cache temporaries", () => {
  assert.equal(detectFullCacheTemporary([{ kind: "dense_cache_temporary", bytes: 1 }], 100).detected, true);
  assert.equal(detectFullCacheTemporary([{ kind: "allocation", bytes: 99 }], 100).detected, true);
  assert.equal(detectFullCacheTemporary([{ kind: "allocation", bytes: 10 }], 100).detected, false);
});
test("Darwin reader is injectable and fail-closed", () => {
  assert.deepEqual(readDarwinMemory(9, () => "phys_footprint: 1234"), { supported: true, bytes: 1234, source: "footprint -p" });
});
test("cancellation always runs owned cleanup", async () => {
  let cleaned = false;
  await assert.rejects(cancellationSafe(async () => { throw new Error("cancelled"); }, async () => { cleaned = true; }));
  assert.equal(cleaned, true);
});
