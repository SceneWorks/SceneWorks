import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { inflateSync } from "node:zlib";
import { assertCompleted, assertPins, assertPlan, assertResume, assertWorker, BASE_REVISION, LICENSE_URL, runLifecycle, syntheticPng, trainingBody } from "./qwen-image-2-1-app-lifecycle.mjs";

const sha = "a".repeat(40), hash = "b".repeat(64), workerId = "owned-qwen-worker";
const read = (file) => readFileSync(new URL(`../${file}`, import.meta.url), "utf8");
const target = (mode) => ({ id: mode === "edit" ? "qwen_image_2_1_edit_lora" : "qwen_image_2_1_lora", ui: { requiresLicenseAcknowledgment: true, licenseUrl: LICENSE_URL }, defaults: { advanced: { mixedPrecision: "bf16" }, learningRate: 0.0001, optimizer: "adamw8bit" } });
function fixture(mode = "t2i") {
  const body = trainingBody(target(mode), "dataset", mode);
  const outputDir = "/owned/project/loras/adapter";
  const plan = { jobId: "original", target: { targetId: body.targetId, kernel: body.targetId, baseModel: "qwen_image_2_1", baseModelPath: "/owned/snapshot" }, config: body.config, dataset: { datasetId: "dataset", items: [{ imagePath: "/owned/image.png", ...(mode === "edit" ? { referenceImagePaths: ["/owned/source.png"] } : {}) }] }, output: { loraId: "adapter", outputDir, fileName: "adapter.safetensors" } };
  const job = { id: "original", type: "lora_train", status: "completed", stage: "completed", backend: "mlx", workerId, error: null, payload: { dryRun: false, plan }, result: { backend: "mlx", targetId: body.targetId, kernel: body.targetId, networkType: body.config.advanced.networkType, steps: 200, stepsCompleted: 200, finalLoss: 0.1, checkpoints: [{ step: 50 }], trainingSamples: [{ step: 100, sampleSource: "live_adapter", relativePath: "loras/adapter/samples/image.png" }], loraId: "adapter", fileName: "adapter.safetensors", outputPath: `${outputDir}/adapter.safetensors`, datasetId: "dataset" } };
  const library = { id: "adapter", scope: "project", installState: "installed", installedPath: outputDir, source: { provider: "training", path: "loras/adapter" }, files: ["adapter.safetensors"], family: "qwen-image-2-1", baseModel: "qwen_image_2_1", networkType: body.config.advanced.networkType, license: "Qwen RESEARCH LICENSE AGREEMENT", licenseUrl: LICENSE_URL, provenance: { trainingJobId: "original", targetId: body.targetId, datasetId: "dataset" }, ...(mode === "edit" ? { trainingMode: "edit" } : {}) };
  const metadata = { family: "qwen-image-2-1", baseModel: "qwen_image_2_1", ss_base_model_version: "qwen_image_2_1", license: "Qwen Research License Agreement (research/evaluation only)", "modelspec.license": "Qwen Research License Agreement (research/evaluation only)", licenseNotice: "Qwen RESEARCH LICENSE AGREEMENT applies", networkType: body.config.advanced.networkType, rank: "4", alpha: "4", ...(mode === "edit" ? { trainingMode: "edit" } : {}) };
  return { body, job, library, metadata };
}

test("lifecycle uses valid product floors and moderate previews for both actual targets", () => {
  for (const mode of ["t2i", "edit"]) {
    const { body, job, library, metadata } = fixture(mode);
    assert.equal(body.config.steps, 200); assert.equal(body.config.resolution, 768);
    assert.equal(body.config.saveEvery, 50); assert.equal(body.config.advanced.sampleEvery, 100);
    assert.equal(body.config.advanced.sampleCount, 1); assert.equal(body.dryRun, false);
    assertPlan(job, body, "/owned/snapshot"); assertCompleted(job, library, metadata, mode, workerId);
  }
  assert.throws(() => trainingBody({ ...target("t2i"), id: "qwen_image_lora" }, "dataset", "t2i"));
  assert.throws(() => trainingBody({ ...target("t2i"), ui: {} }, "dataset", "t2i"));
});

test("canonical route rejects legacy aliases, wrong frozen base, and lost edit references", () => {
  const mutants = [
    (f) => { f.job.payload.plan.target.kernel = "qwen_image_lora"; },
    (f) => { f.job.payload.plan.target.baseModelPath = "/external/cache/main"; },
    (f) => { f.job.payload.plan.dataset.items[0].referenceImagePaths = []; },
    (f) => { f.job.payload.plan.config.steps = 2; },
    (f) => { f.job.payload.dryRun = true; },
  ];
  for (const mutate of mutants) { const f = fixture("edit"); mutate(f); assert.throws(() => assertPlan(f.job, f.body, "/owned/snapshot")); }
});

test("completion cannot substitute checkpoints, dry probes, or an imported library adapter", () => {
  const mutants = [
    (f) => { f.job.status = "canceled"; },
    (f) => { f.job.result.stepsCompleted = 2; },
    (f) => { f.job.backend = "cpu"; },
    (f) => { f.job.workerId = "unrelated-worker"; },
    (f) => { f.job.result.checkpoints = []; },
    (f) => { f.job.result.trainingSamples = []; },
    (f) => { f.library.source.provider = "import"; },
    (f) => { f.library.files = ["adapter-step000050.safetensors"]; },
    (f) => { f.job.result.outputPath = "/owned/checkpoint.safetensors"; },
    (f) => { f.library.provenance.trainingJobId = "other-job"; },
    (f) => { f.library.licenseUrl = "https://example.com/license"; },
    (f) => { delete f.metadata.licenseNotice; },
    (f) => { f.metadata.baseModel = "qwen_image_2512"; },
    (f) => { f.metadata.networkType = "lora"; },
    (f) => { delete f.metadata.trainingMode; },
    (f) => { delete f.library.trainingMode; },
  ];
  for (const mutate of mutants) { const f = fixture("edit"); mutate(f); assert.throws(() => assertCompleted(f.job, f.library, f.metadata, "edit", workerId)); }
});

test("resume preserves the entire plan and continues from a real fingerprinted optimizer bundle", () => {
  const f = fixture(); const canceled = { ...f.job, status: "canceled", canceledAt: "2026-10-04T00:00:00Z" };
  const resumed = structuredClone(f.job); resumed.id = "retry"; resumed.sourceJobId = f.job.id; resumed.payload.plan.config.advanced.resume = true;
  const observation = { first_training_step: 51, worker_id: workerId };
  const checkpoint = { step: 50, metadata: { step: "50", training_config: "steps=200;accum=1;scheduler=Constant", request_fingerprint: hash }, optimizer_bytes: 1234 };
  assertResume(f.job, canceled, resumed, observation, checkpoint);
  const mutants = [
    (parts) => { parts[1].status = "canceling"; },
    (parts) => { parts[2].payload.plan.config.steps = 2; },
    (parts) => { parts[2].payload.plan.dataset.items[0].imagePath = "/changed.png"; },
    (parts) => { parts[2].payload.plan.output.outputDir = "/different/output"; },
    (parts) => { parts[3].first_training_step = 1; },
    (parts) => { parts[3].first_training_step = 100; },
    (parts) => { parts[4].metadata.request_fingerprint = ""; },
    (parts) => { parts[4].optimizer_bytes = 0; },
  ];
  for (const mutate of mutants) { const parts = structuredClone([f.job, canceled, resumed, observation, checkpoint]); mutate(parts); assert.throws(() => assertResume(...parts)); }
});

test("native worker rejects CPU fallback and unrelated idle workers", () => {
  const worker = { id: workerId, status: "idle", currentJobId: null, gpuId: "mlx", gpuName: "Apple Silicon (MLX)", capabilities: ["gpu", "lora_train_execute"] };
  assertWorker(worker, workerId);
  for (const changed of [{ id: "other" }, { status: "busy" }, { gpuId: "cpu" }, { capabilities: ["gpu", "cpu", "lora_train_execute"] }, { capabilities: ["gpu"] }]) assert.throws(() => assertWorker({ ...worker, ...changed }, workerId));
});

test("every manifest and lock dependency must resolve the exact final inference revision", () => {
  const manifest = `dependency = { git = "https://github.com/SceneWorks/inference", rev = "${sha}" }`, lock = `source = "git+https://github.com/SceneWorks/inference?rev=${sha}#${sha}"`;
  assertPins([manifest, manifest], lock, sha);
  assert.throws(() => assertPins([manifest, manifest.replace(sha, "c".repeat(40))], lock, sha));
  assert.throws(() => assertPins([manifest], lock.replace(`#${sha}`, `#${"d".repeat(40)}`), sha));
  assert.throws(() => assertPins([""], lock, sha));
  // Exercises the actual repository spelling, beyond synthetic regex fixtures.
  const manifests = ["Cargo.toml", "crates/sceneworks-worker/Cargo.toml", "crates/sceneworks-memory-adapter/Cargo.toml"].map(read);
  const actual = /SceneWorks\/inference", rev = "([0-9a-f]{40})"/.exec(manifests[0])[1]; assertPins(manifests, read("Cargo.lock"), actual);
});

test("public synthetic image pairs have valid PNG payloads and distinct source/target pixels", () => {
  const source = syntheticPng(), target = syntheticPng(true);
  assert.equal(source.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
  assert.equal(source.readUInt32BE(16), 768); assert.equal(source.readUInt32BE(20), 768);
  const chunks = (png) => { let offset = 8, compressed = []; while (offset < png.length) { const length = png.readUInt32BE(offset), type = png.toString("ascii", offset + 4, offset + 8); if (type === "IDAT") compressed.push(png.subarray(offset + 8, offset + 8 + length)); offset += length + 12; } return inflateSync(Buffer.concat(compressed)); };
  assert.equal(chunks(source).length, (768 * 3 + 1) * 768);
  assert.notDeepEqual(chunks(source), chunks(target));
});

function assertWorkflowScope(workflow) {
  const trigger = workflow.slice(workflow.indexOf("on:"), workflow.indexOf("permissions:"));
  assert.match(trigger, /workflow_dispatch:/); assert.doesNotMatch(trigger, /\b(push|pull_request|schedule):/);
  assert.match(workflow, /runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/);
  assert.match(workflow, /test "\$RUNNER_NAME" = nax-macos\s*$/m);
  assert.match(workflow, /ref: \$\{\{ inputs.source_sha \}\}/);
  assert.match(workflow, /QWEN_APP_SOURCE_SHA: \$\{\{ inputs.source_sha \}\}/);
  assert.match(workflow, /cancel-in-progress: false/); assert.match(workflow, /if: always\(\)/);
  assert.match(workflow, /--build-type Release --github-env/);
  assert.doesNotMatch(workflow, /measure-memory|download-missing|gh pr|git push|bump-inference/);
}
test("manual app workflow preserves exclusive primary-runner and immutable evidence scope", () => {
  const workflow = read(".github/workflows/qwen-image-2-1-app-lifecycle.yml"); assertWorkflowScope(workflow);
  for (const mutant of [workflow.replace("workflow_dispatch:", "push:"), workflow.replace("ARM64, rw-starvector", "ARM64, nax"), workflow.replace("= nax-macos", "= nax-macos-2"), workflow.replace("inputs.source_sha", "github.ref"), workflow.replace("if: always()", "if: success()"), workflow + "\nrun: node scripts/measure-memory-catalog.mjs\n"]) assert.throws(() => assertWorkflowScope(mutant));
  const harness = read("scripts/qwen-image-2-1-app-lifecycle.mjs");
  assert.match(harness, /SCENEWORKS_CREDENTIALS_DIR: path.join\(state, "credentials"\)/);
  assert.match(harness, /HF_HUB_OFFLINE: "1"/); assert.match(harness, /SCENEWORKS_WORKER_ONLY: "1"/);
  assert.doesNotMatch(harness, /\/loras\/import|\/models\/download|\.cache[\\/]huggingface/);
  assert.match(harness, /process.kill\(-child.pid, "SIGTERM"\)/);
  assert.equal(BASE_REVISION, "790c92633540aa0cb11d9abf19eb46d861714758");
});

test("wrong runtime writes a failed receipt before any native build or service starts", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "qwen-app-guard-")), output = path.join(temporary, "attempt");
  try {
    await assert.rejects(runLifecycle({ RUNNER_NAME: "unrelated-runner", RUNNER_TEMP: temporary, QWEN_APP_OUTPUT: output }), /primary Apple Silicon Mac required/);
    const receipt = JSON.parse(await readFile(path.join(output, "evidence", "receipt.json"), "utf8"));
    assert.equal(receipt.status, "failed"); assert.deepEqual(receipt.cases, []); assert.deepEqual(receipt.cleanup, []); assert.ok(receipt.finished_at);
  } finally {
    // Exact mkdtemp path remains inside the OS temporary directory before recursive cleanup.
    assert.ok(path.resolve(temporary).startsWith(path.resolve(tmpdir()) + path.sep));
    await rm(temporary, { recursive: true, force: true });
  }
});
