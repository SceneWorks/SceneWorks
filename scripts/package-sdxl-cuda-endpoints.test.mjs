import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { logicalCaseId, recordId } from "./memory-calibration-harness.mjs";
import { buildEndpointConfig, compiledSources, verifyOutputBytes, EVIDENCE_DIR } from "./package-sdxl-cuda-endpoints.mjs";

const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const seed = JSON.parse(await readFile(new URL("../docs/calibration/sc-22738/sdxl-q4-candle-evidence.json", import.meta.url)));

// Unit-only data derived from a valid historical gated shape. Never packaged as a measurement.
function fixture() {
  const bundles = {};
  const endpoints = [];
  for (const [index, name] of ["resident", "sequential"].entries()) {
    const bundle = structuredClone(seed);
    const record = bundle.records[0];
    record.id = `imc-${String(index + 1).padStart(20, "0")}`;
    record.loadShape = "eager_materialization";
    record.artifact.inventorySha256 = "a".repeat(64);
    record.target.geometry = { width: 1024, height: 1024, frames: 1, batch: 1 };
    record.strategy = name === "resident"
      ? { rung: "resident", engagedRungs: ["resident"], parameters: {} }
      : { rung: "bounded_attention", engagedRungs: ["resident", "staged_residency", "bounded_decode", "bounded_attention"], parameters: { stageResidency: true, decodeTileEdge: 512, decodeOverlap: 64, attentionChunkSize: 67108864 } };
    record.sweep = { axes: [], cases: [{ parameters: record.strategy.parameters, result: "passed" }], rangeVerified: false };
    record.logicalCaseId = logicalCaseId(record);
    record.id = recordId(record);
    bundles[name] = JSON.stringify(bundle);
    endpoints.push({ endpoint: name, inferencePin: record.repositories.inference.revision,
      evidenceSha256: hash(bundles[name]), recordId: record.id,
      wholeRunPeakBytes: record.observedMemory.overall.allocatorBytes });
  }
  return { bundles, summary: { schemaVersion: 1, profile: "sdxl-cuda-endpoints",
    inferencePin: seed.records[0].repositories.inference.revision,
    artifactInventory: { sha256: "a".repeat(64) }, endpoints } };
}

const build = (f) => buildEndpointConfig(JSON.stringify(f.summary), f.bundles, "b".repeat(64));

test("packaging keeps both observed endpoints and source bytes without certification", () => {
  const f = fixture();
  const before = JSON.stringify(f);
  const config = build(f);
  assert.equal(JSON.stringify(f), before);
  for (const name of ["resident", "sequential"]) {
    const record = JSON.parse(f.bundles[name]).records[0];
    const anchor = config.pair[name];
    assert.equal(anchor.source.sha256, hash(f.bundles[name]));
    assert.equal(anchor.source.recordId, record.id);
    assert.equal(anchor.source.path, `${EVIDENCE_DIR}/${name}-evidence.json`);
    assert.equal(anchor.overallAllocatorEnvelopeBytes, record.observedMemory.overall.allocatorBytes);
    assert.equal(anchor.phaseActivePeakBytes.denoise, record.observedMemory.denoise.activeBytes);
    assert.equal(anchor.source.loaderClosureDigest, "b".repeat(64));
    assert.equal(record.status, "gated");
    assert.equal(record.predictedPeakBytes, null);
    assert.equal(record.quality.result, "not_run");
  }
  assert.equal(config.pair.resident.measuredRegime.staged, false);
  assert.equal(config.pair.sequential.measuredRegime.staged, true);
  assert.equal(config.pair.sequential.measuredRegime.attentionChunked, true);
  assert.equal(config.pair.summary.sha256, hash(JSON.stringify(f.summary)));
  const entries = compiledSources(config);
  assert.equal((entries.match(/include_str!/g) ?? []).length, 3);
  assert.ok(entries.includes(`../../../${EVIDENCE_DIR}/sdxl-cuda-endpoints.json`));
  assert.ok(!entries.includes("memory-anchors.json"));
});

test("packaging refuses changed or incomplete capture bindings", () => {
  for (const mutate of [
    (f) => { f.bundles.resident += " "; },
    (f) => { f.summary.endpoints.pop(); },
    (f) => { f.summary.endpoints[1].endpoint = "resident"; },
    (f) => { f.summary.endpoints[1].recordId = f.summary.endpoints[0].recordId; },
    (f) => { f.summary.endpoints[1].inferencePin = "f".repeat(40); },
    (f) => { f.summary.artifactInventory.sha256 = "f".repeat(64); },
    (f) => { f.summary.endpoints[1].wholeRunPeakBytes++; },
  ]) {
    const f = fixture(); mutate(f); assert.throws(() => build(f));
  }
  const f = fixture();
  const bundle = JSON.parse(f.bundles.resident);
  bundle.records[0].loadShape = "deferred_materialization";
  bundle.records[0].logicalCaseId = logicalCaseId(bundle.records[0]);
  bundle.records[0].id = recordId(bundle.records[0]);
  f.bundles.resident = JSON.stringify(bundle);
  f.summary.endpoints[0].recordId = bundle.records[0].id;
  f.summary.endpoints[0].evidenceSha256 = hash(f.bundles.resident);
  assert.throws(() => build(f), /stock eager/);
});

function outputFixture() {
  const rgb = Buffer.alloc(1024 * 1024 * 3);
  rgb[3] = 255;
  const outputs = { "resident.rgb": rgb, "sequential.rgb": Buffer.from(rgb),
    "resident.png": Buffer.from("unit-test-png-receipt"), "sequential.png": Buffer.from("unit-test-png-receipt") };
  const summary = { endpoints: ["resident", "sequential"].map((name) => ({ endpoint: name, output: {
    width: 1024, height: 1024, channels: 3, rgbBytes: rgb.length, rgbSha256: hash(rgb),
    pngBytes: outputs[`${name}.png`].length, pngSha256: hash(outputs[`${name}.png`]),
    spatiallyNonFlat: true, minimumChannel: 0, maximumChannel: 255,
  } })), qualityComparison: { result: "measured", reference: "resident", selected: "sequential",
    channels: rgb.length, maximumAbsoluteRgbDelta255: 0, meanAbsoluteRgbDelta255: 0,
    maximumAbsoluteError: 0, meanAbsoluteError: 0, changedChannels: 0, identical: true } };
  return { summary, outputs };
}

test("packaging hashes retained PNG/RGB and independently checks full RGB metrics", () => {
  const f = outputFixture();
  verifyOutputBytes(f.summary, f.outputs);
  f.outputs["sequential.rgb"][4] = 1;
  assert.throws(() => verifyOutputBytes(f.summary, f.outputs), /bytes do not match/);
  f.summary.endpoints[1].output.rgbSha256 = hash(f.outputs["sequential.rgb"]);
  assert.throws(() => verifyOutputBytes(f.summary, f.outputs), /comparison/);
  f.summary.qualityComparison = { ...f.summary.qualityComparison, maximumAbsoluteRgbDelta255: 1,
    meanAbsoluteRgbDelta255: 1 / (1024 * 1024 * 3), maximumAbsoluteError: 1 / 255,
    meanAbsoluteError: 1 / (1024 * 1024 * 3) / 255, changedChannels: 1, identical: false };
  verifyOutputBytes(f.summary, f.outputs);
  f.outputs["resident.png"] = Buffer.from("changed PNG");
  assert.throws(() => verifyOutputBytes(f.summary, f.outputs), /resident.png/);
});
