// CI_IMPACT: pure request/receipt validation only, <1 second; no model, CUDA context or build.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { ENDPOINTS, endpointCase, validateEndpoint, compareRgb } from "./measure-sdxl-cuda-endpoints.mjs";

const plan = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8"));
const uuid = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe";
function receipt() {
  return { endpoint: "resident", deviceUuid: uuid, cudaContextUuid: uuid, logicalCudaDevice: 0,
    loadPeakBytes: 1024, wholeRunPeakBytes: 2048, baselineBytes: 512, wholeDevicePeakBytes: 2560,
    output: { width: 1024, height: 1024, channels: 3, rgbBytes: 1024 * 1024 * 3,
      spatiallyNonFlat: true, rgbSha256: "a".repeat(64) },
    cleanup: { beforeBytes: 512, afterBytes: 512, toleranceBytes: 64 * 1024 ** 2, adapterProcessExited: true } };
}

test("endpoint pair keeps one q4 artifact and renders only maximum and minimum", () => {
  const original = structuredClone(plan);
  const cases = ENDPOINTS.map((endpoint) => endpointCase(plan, endpoint));
  assert.equal(cases.length, 2);
  assert.deepEqual(cases.map((item) => item.strategy.rung), ["resident", "bounded_attention"]);
  assert.deepEqual(cases.map((item) => item.target), [cases[0].target, cases[0].target]);
  assert.deepEqual(cases[0].target.geometry, { width: 1024, height: 1024, batch: 1, frames: 1 });
  assert.equal(cases[0].target.tier, "q4");
  assert.notEqual(cases[0].logicalCaseId, cases[1].logicalCaseId);
  assert.deepEqual(cases[1].strategy.parameters, { stageResidency: true, decodeTileEdge: 512, decodeOverlap: 64, attentionChunkSize: 64 * 1024 ** 2 });
  assert.deepEqual(plan, original);
});

test("endpoint receipt accepts whole-run load-inclusive peak and natural cleanup", () => {
  assert.doesNotThrow(() => validateEndpoint(receipt(), ENDPOINTS[0]));
});

test("endpoint receipt rejects missing load inclusion, context mismatch, flat output and leaked residency", () => {
  for (const mutate of [
    (value) => { value.wholeRunPeakBytes = value.loadPeakBytes - 1; },
    (value) => { value.cudaContextUuid = "GPU-foreign"; },
    (value) => { value.output.spatiallyNonFlat = false; },
    (value) => { value.cleanup.afterBytes += value.cleanup.toleranceBytes + 1; },
  ]) {
    const value = receipt(); mutate(value);
    assert.throws(() => validateEndpoint(value, ENDPOINTS[0]));
  }
});

// One changed channel distinguishes a true RGB comparison from a digest-only attestation.
test("saved RGB quality comparison measures every channel and rejects a shape mismatch", () => {
  const comparison = compareRgb(Buffer.from([0, 100, 255]), Buffer.from([0, 101, 255]));
  assert.equal(comparison.maximumAbsoluteRgbDelta255, 1);
  assert.equal(comparison.meanAbsoluteRgbDelta255, 1 / 3);
  assert.equal(comparison.changedChannels, 1);
  assert.equal(comparison.identical, false);
  assert.throws(() => compareRgb(Buffer.from([0]), Buffer.from([0, 1])));
});
