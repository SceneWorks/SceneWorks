// CI_IMPACT: pure request/receipt validation only, <1 second; no model, CUDA context or build.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";
import { ENDPOINTS, endpointCase, validateEndpoint, compareRgb, endpointEvidenceReference, sdxlAssemblyInventory, admittedReplayRequest, admittedReplayEnv, assertAdmittedReplaySelection, admittedReplaySelection, persistedReplayAsset, parseReplayGpuSample } from "./measure-sdxl-cuda-endpoints.mjs";

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
  assert.deepEqual(wanted("sdxl-cuda-admitted-replay"), { revisions: [`wanted:${pin}`], queried: false });
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
    const file = role === "base" ? "weights.bin" : role === "vae_fp16_fix" ? "diffusion_pytorch_model.safetensors" : "tokenizer.json";
    await writeFile(path.join(roots[role], file), role);
  }
  const row = { tierRoot: roots.base, env: {
    SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_L: roots.tokenizer_clip_l,
    SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_BIGG: roots.tokenizer_clip_bigg,
    SCENEWORKS_SDXL_COMPONENT_VAE_FP16_FIX: roots.vae_fp16_fix,
  } };
  const first = await sdxlAssemblyInventory(row);
  assert.deepEqual(first.entries.map((entry) => entry.path), ["base/weights.bin", "components/tokenizer_clip_bigg/tokenizer.json", "components/tokenizer_clip_l/tokenizer.json", "components/vae_fp16_fix/diffusion_pytorch_model.safetensors"]);
  await writeFile(path.join(roots.tokenizer_clip_l, "model.safetensors"), "unused cached CLIP weights");
  assert.equal((await sdxlAssemblyInventory(row)).sha256, first.sha256);
  await writeFile(path.join(roots.tokenizer_clip_l, "tokenizer.json"), "changed consumed tokenizer");
  const changedTokenizer = await sdxlAssemblyInventory(row);
  assert.notEqual(changedTokenizer.sha256, first.sha256);
  await writeFile(path.join(roots.vae_fp16_fix, "diffusion_pytorch_model.safetensors"), "changed-shared-vae");
  assert.notEqual((await sdxlAssemblyInventory(row)).sha256, changedTokenizer.sha256);
  await assert.rejects(() => sdxlAssemblyInventory({ ...row, env: {} }), /source missing/);
});

// CI_IMPACT: pure production-request/environment/receipt contracts; never starts API or worker.
test("admitted replay submits ordinary explicit-q4 defaults without a strategy override", () => {
  const request = admittedReplayRequest("owned-project");
  assert.deepEqual(request.advanced, { mlxQuantize: 4 });
  assert.equal(request.width, 1024); assert.equal(request.height, 1024); assert.equal(request.count, 1);
  assert.equal(request.model, "sdxl"); assert.deepEqual(request.loras, []);
  const env = admittedReplayEnv({ PATH: "/tools", CUDA_VISIBLE_DEVICES: "0", SCENEWORKS_GPU_ID: "0", SCENEWORKS_DATA_DIR: "/user", HF_TOKEN: "secret" },
    { state: "/owned/state", hub: "/cache/hub", workerId: "owned-worker", port: 17931, parentPid: 1234 });
  assert.equal(env.CUDA_VISIBLE_DEVICES, "1"); assert.equal(env.SCENEWORKS_GPU_ID, "1");
  assert.equal(env.SCENEWORKS_CUDA_VRAM_CAP_GB, "6"); assert.equal(env.SCENEWORKS_PARENT_PID, "1234");
  assert.equal(env.SCENEWORKS_DATA_DIR, path.join("/owned/state", "data")); assert.equal(env.HF_TOKEN, undefined);
  assert.equal(env.SCENEWORKS_WORKER_CHILD, "1"); assert.equal(env.SCENEWORKS_BACKEND_CANDLE_ENABLED, "true");
});

test("admitted replay requires actual measured optimized worker selection and budget telemetry", () => {
  const selection = { backend: "candle", route: "sdxl", actualTier: "q4", mode: "text_to_image", strategy: "bounded_attention",
    basis: "measured_endpoint", geometry: { width: 1024, height: 1024, batch: 1 },
    parameters: { stageResidency: true, decodeTileEdge: 512, decodeOverlap: 64, attentionChunkSize: 67_108_864 },
    predictedPeakBytes: 3_000_000_000, reserveGb: 1.5, availableGb: 4.5 };
  assert.doesNotThrow(() => assertAdmittedReplaySelection(selection));
  for (const patch of [{ strategy: "resident" }, { actualTier: "q8" }, { basis: "estimated" }, { predictedPeakBytes: null }, { availableGb: 95 }]) {
    assert.throws(() => assertAdmittedReplaySelection({ ...selection, ...patch }));
  }
  assert.throws(() => assertAdmittedReplaySelection({ ...selection, parameters: { ...selection.parameters, stageResidency: false } }));
  const emitted = { event: "image_memory_strategy_selected", ...selection };
  const flat = { event: emitted.event, route: "sdxl", actual_tier: "q4", predicted_peak_bytes: selection.predictedPeakBytes };
  const workerLog = ["non-JSON startup diagnostic", JSON.stringify(flat), JSON.stringify(emitted)].join("\n");
  assert.deepEqual(admittedReplaySelection(workerLog), emitted, "retain the complete emitted payload, ignoring its adjacent flat tracing summary");
  assert.throws(() => admittedReplaySelection(JSON.stringify(flat)), /saw 0/);
  assert.throws(() => admittedReplaySelection(`${workerLog}\n${JSON.stringify(emitted)}`), /saw 2/);
  assert.throws(() => admittedReplaySelection(JSON.stringify({ ...emitted, parameters: { ...selection.parameters, stageResidency: false } })), /did not admit/);
});

test("admitted replay waits for API persisted assets and validates the actual sidecar shape", () => {
  const image = { id: "asset_replay", projectId: "project_replay", type: "image", lineage: { jobId: "job_replay" },
    file: { path: "assets/images/replay.png", mimeType: "image/png", width: 1024, height: 1024 } };
  const job = { status: "completed", result: { assets: [image], assetIds: [image.id] } };
  assert.equal(persistedReplayAsset({ status: "completed", result: { assetWrites: [{ mediaPath: image.file.path }] } }, "project_replay", "job_replay"), null,
    "completed worker facts do not prove durable API asset persistence");
  assert.equal(persistedReplayAsset({ ...job, result: { ...job.result, assetWrites: [] } }, "project_replay", "job_replay"), null);
  assert.deepEqual(persistedReplayAsset(job, "project_replay", "job_replay"), image);
  for (const result of [{ ...job.result, assetIds: ["other"] }, { ...job.result, assets: [image, image] },
    { ...job.result, assets: [{ ...image, projectId: "other" }] },
    { ...job.result, assets: [{ ...image, lineage: { jobId: "other" } }] },
    { ...job.result, assets: [{ ...image, file: { ...image.file, width: 768 } }] },
    { ...job.result, assets: [{ ...image, file: { ...image.file, path: "../replay.png" } }] },
    { ...job.result, assets: [{ ...image, file: { ...image.file, path: "C:\\replay.png" } }] },
    { ...job.result, assets: [{ ...image, file: { ...image.file, mimeType: "video/mp4" } }] }]) {
    assert.throws(() => persistedReplayAsset({ ...job, result }, "project_replay", "job_replay"), /sidecar/);
  }
});

test("admitted replay rejects malformed or incomplete physical GPU1 telemetry", () => {
  assert.deepEqual(parseReplayGpuSample(`${uuid}, 2048, 90000, 0, 3`, "sample-time"), {
    at: "sample-time", uuid, usedBytes: 2048 * 1024 ** 2, freeBytes: 90000 * 1024 ** 2, gpuUtilization: 0, memoryUtilization: 3,
  });
  for (const csv of [`${uuid},2048,90000,0`, `${uuid},2048,90000,0,`, `${uuid},2048,90000,0,NaN`,
    `${uuid},2048,90000,0,N/A`, `${uuid},2048,90000,0,101`, `${uuid},2048,90000,101,0`,
    `${uuid},-1,90000,0,0`, `${uuid},2048,Infinity,0,0`, `${uuid},2048,90000,0,0,extra`,
    `${uuid},2048,90000,0,0\n${uuid},2048,90000,0,0`, "GPU-wrong,2048,90000,0,0"]) {
    assert.throws(() => parseReplayGpuSample(csv), /telemetry identity or counters unavailable/);
  }
});

test("admitted replay workflow remains manual and runs native pure gates on GPU1 before one app job", async () => {
  const source = await readFile(new URL("../.github/workflows/memory-catalog-campaign.yml", import.meta.url), "utf8");
  assert.match(source, /- sdxl-cuda-admitted-replay/);
  assert.match(source, /CUDA_VISIBLE_DEVICES: "1"/); assert.match(source, /SCENEWORKS_GPU_ID: "1"/);
  assert.match(source, /cargo test --release --locked -p sceneworks-core --lib sdxl_endpoint_/);
  assert.match(source, /cargo test --release --locked -p sceneworks-worker --features backend-candle --lib candle_memory_strategy::tests::sdxl_/);
  assert.match(source, /cargo clippy --release --locked -p sceneworks-worker --features backend-candle --lib -- -D warnings/);
  assert.ok(source.indexOf("Validate the SDXL gate") < source.indexOf("Replay one ordinary admitted SDXL"));
  const script = await readFile(new URL("./measure-sdxl-cuda-endpoints.mjs", import.meta.url), "utf8");
  assert.match(script, /POST", "\/api\/v1\/image\/jobs"/);
  assert.match(script, /child\.stdin\.end\(\)/, "owned worker uses existing natural stdin-EOF shutdown");
  assert.match(script, /SCENEWORKS_PARENT_PID/, "owned API uses existing parent-death natural shutdown");
  assert.match(script, /receipt\.selection = admittedReplaySelection\(await readFile/);
  assert.match(script, /persistedReplayAsset\(job, project\.id, jobId\)/);
  assert.match(script, /\/assets\/\$\{encodeURIComponent\(persisted\.image\.id\)\}/, "read the persisted sidecar through the ordinary asset API");
});
