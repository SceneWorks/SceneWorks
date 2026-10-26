// CI_IMPACT: pure request/receipt validation only, <1 second; no model, CUDA context or build.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";
import { ENDPOINTS, endpointCase, validateEndpoint, compareRgb, endpointEvidenceReference, sdxlAssemblyInventory } from "./measure-sdxl-cuda-endpoints.mjs";

const plan = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8"));
const uuid = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe";
function receipt() {
  return { inferencePin: "b".repeat(40), endpoint: "resident", deviceUuid: uuid, cudaContextUuid: uuid, logicalCudaDevice: 0,
    loadPeakBytes: 1024, wholeRunPeakBytes: 2048, baselineBytes: 512, wholeDevicePeakBytes: 2560,
    output: { width: 1024, height: 1024, channels: 3, rgbBytes: 1024 * 1024 * 3,
      spatiallyNonFlat: true, rgbSha256: "a".repeat(64) },
    settings: { flashAttn: true, count: 1, cfgInterval: null, guidanceMethod: null, schedulerShift: null },
    engineBuild: { adapterCandleFeature: true, sdxlFlashAttentionKernelCompiled: false },
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
  assert.deepEqual(cases.map((item) => item.loadShape), ["eager_materialization", "eager_materialization"]);
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
    (value) => { value.inferencePin = "missing"; },
    (value) => { value.output.spatiallyNonFlat = false; },
    (value) => { value.settings.flashAttn = false; },
    (value) => { value.settings.cfgInterval = [0.1, 0.9]; },
    (value) => { value.settings.guidanceMethod = "apg"; },
    (value) => { value.settings.schedulerShift = 3; },
    (value) => { value.settings.count = 2; },
    (value) => { value.engineBuild.sdxlFlashAttentionKernelCompiled = true; },
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

// CI_IMPACT: pure bytes/identity checks extend the existing endpoint receipt tests; no inference.
test("endpoint summary binds exact evidence bytes, record, artifact inventory, compiled pin and device", () => {
  const inventorySha256 = "a".repeat(64); const inferencePin = "b".repeat(40);
  const record = { id: "imc-" + "c".repeat(20), artifact: { inventorySha256 },
    repositories: { inference: { revision: inferencePin } }, hardware: { deviceId: uuid } };
  const bundle = { records: [record] }; const bytes = JSON.stringify(bundle) + "\n";
  const expected = { inventorySha256, inferencePin, deviceUuid: uuid };
  const reference = endpointEvidenceReference(bundle, bytes, expected);
  assert.equal(reference.recordId, record.id);
  assert.match(reference.evidenceSha256, /^[a-f0-9]{64}$/);
  assert.notEqual(reference.evidenceSha256, endpointEvidenceReference(bundle, bytes + " ", expected).evidenceSha256);
  for (const patch of [{ inventorySha256: "d".repeat(64) }, { inferencePin: "d".repeat(40) }, { deviceUuid: "GPU-foreign" }]) {
    assert.throws(() => endpointEvidenceReference(bundle, bytes, { ...expected, ...patch }));
  }
  assert.throws(() => endpointEvidenceReference({ records: [record, record] }, bytes, expected));
});

// CI_IMPACT: executes only the shell revision-list guard with a synthetic node function; no Git,
// network, model, CUDA context or source preparation is executed.
test("only the manual SDXL profile omits historical inference anchor queries", async () => {
  const source = await readFile(new URL("./ci/memory-catalog/inference.sh", import.meta.url), "utf8");
  const block = source.slice(source.indexOf('WANTED=("$PIN")'), source.indexOf("# A space-separated string"));
  assert.ok(block.includes("--anchor-revisions"));
  const pin = "a".repeat(40); const historical = "b".repeat(40);
  const script = `set -euo pipefail\nPIN=${pin}\nnode() { printf '%s\\n' ${historical}; printf 'historical-query\\n' >&2; }\n${block}\nprintf 'wanted:%s\\n' "\${WANTED[@]}"\n`;
  const wanted = (profile) => {
    const result = spawnSync("bash", ["-c", script], {
      env: { ...process.env, MEMORY_CAMPAIGN_PROFILE: profile }, encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    return { revisions: result.stdout.trim().split("\n").filter((line) => line.startsWith("wanted:")), queried: result.stderr.includes("historical-query") };
  };
  assert.deepEqual(wanted("sdxl-cuda-endpoints"), { revisions: [`wanted:${pin}`], queried: false });
  assert.deepEqual(wanted("catalog"), { revisions: [`wanted:${pin}`, `wanted:${historical}`], queried: true });
  assert.deepEqual(wanted(""), { revisions: [`wanted:${pin}`, `wanted:${historical}`], queried: true });
  const workflow = await readFile(new URL("../.github/workflows/memory-catalog-campaign.yml", import.meta.url), "utf8");
  assert.match(workflow, /MEMORY_CAMPAIGN_PROFILE: \$\{\{ inputs\.profile \|\| 'catalog' \}\}/);
});

// CI_IMPACT: four tiny source-directory fixtures prove all actual SDXL assembly roles are bound.
test("SDXL assembly inventory binds shared VAE and both tokenizer roots beside the base tier", async () => {
  const hub = await mkdtemp(path.join(tmpdir(), "sdxl-assembly-"));
  const roots = {};
  for (const role of ["base", "tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"]) {
    const root = path.join(hub, `models--fixture--${role}`, "snapshots", "a".repeat(40));
    roots[role] = role === "base" ? path.join(root, "q4") : root;
    await mkdir(roots[role], { recursive: true });
    await writeFile(path.join(roots[role], "weights.bin"), role);
  }
  const row = { tierRoot: roots.base, env: {
    SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_L: roots.tokenizer_clip_l,
    SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_BIGG: roots.tokenizer_clip_bigg,
    SCENEWORKS_SDXL_COMPONENT_VAE_FP16_FIX: roots.vae_fp16_fix,
  } };
  const first = await sdxlAssemblyInventory(row);
  assert.deepEqual(first.entries.map((entry) => entry.path), ["base/weights.bin", "components/tokenizer_clip_bigg/weights.bin", "components/tokenizer_clip_l/weights.bin", "components/vae_fp16_fix/weights.bin"]);
  await writeFile(path.join(roots.vae_fp16_fix, "weights.bin"), "changed-shared-vae");
  assert.notEqual((await sdxlAssemblyInventory(row)).sha256, first.sha256);
  await assert.rejects(() => sdxlAssemblyInventory({ ...row, env: {} }), /source missing/);
});
