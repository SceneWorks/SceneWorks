import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { inflateSync } from "node:zlib";
import { assertCompleted, assertDenseSnapshotLayout, assertHardwareIdentity, assertImmutableRevisions, assertOwnedCacheEnv, assertPins, assertPlan, assertResume, assertRunnerIdentity, assertWorker, BASE_REVISION, LICENSE_URL, buildChildEnv, runLifecycle, syntheticPng, trainingBody } from "./qwen-image-2-1-app-lifecycle.mjs";

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
  assert.match(trigger, /workflow_call:/); assert.doesNotMatch(trigger, /\b(workflow_dispatch|push|pull_request|schedule):/);
  assert.match(workflow, /runner_target:[\s\S]*?default: primary/);
  assert.match(workflow, /inputs\.runner_target == 'secondary'.*\["self-hosted","macOS","ARM64","rw-mage"\].*\["self-hosted","macOS","ARM64","rw-starvector"\]/);
  assert.match(workflow, /QWEN_APP_RUNNER_TARGET: \$\{\{ inputs.runner_target \}\}/);
  assert.match(workflow, /ref: \$\{\{ inputs.source_sha \}\}/);
  assert.match(workflow, /QWEN_APP_SOURCE_SHA: \$\{\{ inputs.source_sha \}\}/);
  assert.match(workflow, /- name: Record and require the selected Mac before build\n        run: node scripts\/qwen-image-2-1-app-lifecycle\.mjs --preflight-only\n        env:\n          QWEN_APP_OUTPUT: \$\{\{ runner\.temp \}\}\/qwen-image-2-1-app-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}/);
  const preflight = workflow.indexOf("--preflight-only"), toolchain = workflow.indexOf("dtolnay/rust-toolchain"), fetch = workflow.indexOf("Fetch Release libmlx"), lifecycle = workflow.indexOf("Build and exercise the owned API");
  assert.ok(preflight > workflow.indexOf("actions/setup-node") && preflight < toolchain && toolchain < fetch && fetch < lifecycle, "receipt-producing route/hardware preflight must run before build setup and services");
  assert.doesNotMatch(workflow, /case "\$QWEN_APP_RUNNER_TARGET:\$RUNNER_NAME"/, "workflow shell must not bypass the receipt-producing harness");
  assert.match(workflow, /- name: Build and exercise the owned API and MLX worker\n        run: node scripts\/qwen-image-2-1-app-lifecycle\.mjs\n        env:\n          QWEN_APP_OUTPUT: \$\{\{ runner\.temp \}\}\/qwen-image-2-1-app-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}/);
  assert.match(workflow, /path: \$\{\{ runner\.temp \}\}\/qwen-image-2-1-app-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}\/evidence\//);
  assert.doesNotMatch(workflow, /^      QWEN_APP_OUTPUT: \$\{\{ runner\.temp \}\}/m, "runner.temp is unavailable in job-level env");
  assert.equal((workflow.match(/^    concurrency:$/gm) ?? []).length, 1); assert.match(workflow, /group: qwen-image-2-1-app-lifecycle-primary-mac/);
  assert.match(workflow, /cancel-in-progress: false/); assert.match(workflow, /if: always\(\)/);
  assert.match(workflow, /--build-type Release --github-env/);
  assert.doesNotMatch(workflow, /measure-memory|download-missing|gh pr|git push|bump-inference/);
}
test("reusable app workflow selects one exact app runner while preserving immutable caller scope", () => {
  const workflow = read(".github/workflows/qwen-image-2-1-app-lifecycle.yml"); assertWorkflowScope(workflow);
  const runnerTempAtJobScope = workflow.replace("      QWEN_APP_SOURCE_SHA: ${{ inputs.source_sha }}\n      QWEN_APP_INFERENCE_SHA: ${{ inputs.inference_sha }}", "      QWEN_APP_SOURCE_SHA: ${{ inputs.source_sha }}\n      QWEN_APP_INFERENCE_SHA: ${{ inputs.inference_sha }}\n      QWEN_APP_OUTPUT: ${{ runner.temp }}/qwen-image-2-1-app-${{ github.run_id }}-${{ github.run_attempt }}");
  assert.throws(() => assertWorkflowScope(runnerTempAtJobScope), /runner\.temp is unavailable in job-level env/);
  for (const mutant of [workflow.replace("workflow_call:", "workflow_dispatch:"), workflow.replace('"rw-mage"', '"nax"'), workflow.replace('"rw-starvector"', '"nax"'), workflow.replace("--preflight-only", "--route-check-only"), workflow.replace("inputs.source_sha", "github.ref"), workflow.replace("if: always()", "if: success()"), workflow + "\nrun: node scripts/measure-memory-catalog.mjs\n"]) assert.throws(() => assertWorkflowScope(mutant));
  const harness = read("scripts/qwen-image-2-1-app-lifecycle.mjs");
  assert.match(harness, /SCENEWORKS_CREDENTIALS_DIR: path.join\(state, "credentials"\)/);
  assert.match(harness, /HF_HUB_OFFLINE: "1"/); assert.match(harness, /SCENEWORKS_WORKER_ONLY: "1"/);
  assert.doesNotMatch(harness, /\/loras\/import|\/models\/download|\.cache[\\/]huggingface/);
  assert.match(harness, /process.kill\(-child.pid, "SIGTERM"\)/);
  assert.equal(BASE_REVISION, "790c92633540aa0cb11d9abf19eb46d861714758");
});

function assertDispatcherIsolation(workflow) {
  assert.match(workflow, /profile:[\s\S]*?default: catalog/);
  assert.match(workflow, /uses: \.\/\.github\/workflows\/qwen-image-2-1-app-lifecycle.yml/);
  assert.match(workflow, /source_sha: \$\{\{ inputs.ref \}\}/);
  assert.match(workflow, /inference_sha: \$\{\{ inputs.inference_sha \}\}/);
  assert.match(workflow, /qwen_app_runner_target:[\s\S]*?options:\s*\n\s*- primary\s*\n\s*- secondary[\s\S]*?default: primary/);
  assert.match(workflow, /runner_target: \$\{\{ inputs.qwen_app_runner_target \}\}/);
  const jobs = [...workflow.slice(workflow.indexOf("\njobs:\n")).matchAll(/^  ([a-z][a-z0-9-]*):\n/gm)].map((match) => match[1]);
  assert.deepEqual(jobs, ["qwen-app-lifecycle", "mlx", "candle"], "all dispatcher jobs must have an explicit profile gate");
  const runs = (profile, backend) => jobs.filter((job) => {
    const condition = new RegExp(`^  ${job}:\\n    if: \\$\\{\\{ (.+) \\}\\}$`, "m").exec(workflow)?.[1];
    assert.ok(condition, `missing ${job} dispatch predicate`);
    // Evaluate only the tiny predicate grammar, never a workflow step or command.
    assert.match(condition, /^[a-z0-9_. '=()&|!-]+$/);
    return Function("inputs", `return (${condition})`)({ profile, backend });
  });
  for (const backend of ["mlx", "candle"]) {
    assert.deepEqual(runs("qwen-image-2-1-app-lifecycle", backend), ["qwen-app-lifecycle"], "app proof must skip all catalog mutation steps");
    assert.deepEqual(runs("catalog", backend), [backend], "default catalog contract must remain unchanged");
    assert.deepEqual(runs("", backend), [backend], "omitted profile preserves catalog behavior");
    assert.deepEqual(runs("unknown", backend), [], "unknown profiles must fail closed");
  }
}
test("registered dispatcher isolates app proof from both catalog lanes and their publication", () => {
  const workflow = read(".github/workflows/memory-catalog-campaign.yml").replace(/\r\n/g, "\n");
  assertDispatcherIsolation(workflow);
  const guard = "(inputs.profile == '' || inputs.profile == 'catalog') && ";
  const mutants = [workflow.replace(guard, ""), workflow.replaceAll(guard, ""), workflow.replace("inputs.profile == 'qwen-image-2-1-app-lifecycle'", "inputs.backend == 'mlx'"), workflow.replace("default: catalog", "default: qwen-image-2-1-app-lifecycle"), workflow.replace("source_sha: ${{ inputs.ref }}", "source_sha: ${{ github.ref }}"), workflow.replace("runner_target: ${{ inputs.qwen_app_runner_target }}", "runner_target: primary"), workflow + "\n  publish:\n    if: ${{ inputs.backend == 'mlx' }}\n    runs-on: ubuntu-latest\n", workflow + "\n  unguarded-publish:\n    runs-on: ubuntu-latest\n"];
  for (const mutant of mutants) assert.throws(() => assertDispatcherIsolation(mutant));
});

test("frozen dense snapshot uses processor tokenizer files and rejects the older tokenizer layout", async () => {
  const base = await mkdtemp(path.join(tmpdir(), "qwen-21-layout-"));
  try {
    for (const file of ["model_index.json", "transformer/config.json", "text_encoder/config.json", "vae/config.json", "processor/tokenizer_config.json", "processor/tokenizer.json"]) {
      await mkdir(path.dirname(path.join(base, file)), { recursive: true }); await writeFile(path.join(base, file), "{}");
    }
    await assertDenseSnapshotLayout(base);
    await mkdir(path.join(base, "tokenizer"));
    for (const file of ["tokenizer_config.json", "tokenizer.json"]) { await writeFile(path.join(base, "tokenizer", file), "{}"); await rm(path.join(base, "processor", file)); }
    await assert.rejects(assertDenseSnapshotLayout(base), /processor/);
    await writeFile(path.join(base, "processor", "tokenizer_config.json"), "{}");
    await assert.rejects(assertDenseSnapshotLayout(base), /processor.*tokenizer.json/);
    await writeFile(path.join(base, "processor", "tokenizer.json"), "");
    await assert.rejects(assertDenseSnapshotLayout(base), /incomplete.*processor/);
  } finally {
    assert.ok(path.resolve(base).startsWith(path.resolve(tmpdir()) + path.sep)); await rm(base, { recursive: true, force: true });
  }
});

test("runner route admits only the selected exact target/name pair", () => {
  assertRunnerIdentity("primary", "nax-macos");
  assertRunnerIdentity("secondary", "nax-macos-2");
  for (const pair of [["unknown", "nax-macos"], ["primary", "nax-macos-2"], ["secondary", "nax-macos"], ["secondary", "unrelated-runner"]]) assert.throws(() => assertRunnerIdentity(...pair), /target\/name mismatch/);
  assertImmutableRevisions(sha, "b".repeat(40));
  assert.throws(() => assertImmutableRevisions("main", "b".repeat(40)), /immutable source/);
  assertHardwareIdentity([{ chip_type: "Apple M5 Max", physical_memory: "128 GB", machine_model: "Mac" }]);
  for (const hardware of [[], [{ chip_type: "Apple M4 Max", physical_memory: "128 GB" }], [{ chip_type: "Apple M5 Max", physical_memory: "64 GB" }], [{ chip_type: "Apple M5 Max", physical_memory: "128 GB" }, { chip_type: "Apple M5 Max", physical_memory: "128 GB" }]]) assert.throws(() => assertHardwareIdentity(hardware), /128 GB M5 Max/);
});

const profiler = (devices) => async (command, args) => {
  assert.equal(command, "system_profiler"); assert.deepEqual(args, ["SPHardwareDataType", "-json"]);
  return { stdout: JSON.stringify({ SPHardwareDataType: devices }) };
};
const m5 = [{ chip_type: "Apple M5 Max", physical_memory: "128 GB", machine_model: "Mac" }];

test("route and hardware failures record actual identity before build", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "qwen-app-guard-")), output = path.join(temporary, "attempt");
  try {
    const cases = [
      { name: "wrong target", target: "unknown", runner: "nax-macos", devices: m5, error: /target\/name mismatch/ },
      { name: "wrong name", target: "secondary", runner: "nax-macos", devices: m5, error: /target\/name mismatch/ },
      { name: "wrong chip", target: "secondary", runner: "nax-macos-2", devices: [{ chip_type: "Apple M4 Max", physical_memory: "128 GB", machine_model: "Mac" }], error: /128 GB M5 Max/ },
      { name: "wrong memory", target: "secondary", runner: "nax-macos-2", devices: [{ chip_type: "Apple M5 Max", physical_memory: "64 GB", machine_model: "Mac" }], error: /128 GB M5 Max/ },
    ];
    for (const [index, item] of cases.entries()) {
      const attempt = path.join(output, String(index));
      await assert.rejects(runLifecycle({ QWEN_APP_RUNNER_TARGET: item.target, RUNNER_NAME: item.runner, RUNNER_TEMP: temporary, QWEN_APP_OUTPUT: attempt, QWEN_APP_SOURCE_SHA: sha, QWEN_APP_INFERENCE_SHA: "b".repeat(40), GITHUB_RUN_ID: "42", GITHUB_RUN_ATTEMPT: "3" }, { platform: "darwin", arch: "arm64", execSystemProfiler: profiler(item.devices) }), item.error, item.name);
      const receipt = JSON.parse(await readFile(path.join(attempt, "evidence", "receipt.json"), "utf8"));
      assert.equal(receipt.status, "failed"); assert.deepEqual(receipt.cases, []); assert.deepEqual(receipt.cleanup, []); assert.ok(receipt.finished_at);
      assert.equal(receipt.identity.requested_runner_target, item.target); assert.equal(receipt.identity.runner_name, item.runner);
      assert.deepEqual(receipt.identity.platform, { os: "darwin", arch: "arm64" });
      assert.equal(receipt.identity.sceneworks_revision, sha); assert.equal(receipt.identity.inference_revision, "b".repeat(40));
      assert.equal(receipt.identity.run_id, "42"); assert.equal(receipt.identity.run_attempt, "3");
      assert.equal(receipt.identity.hardware.status, "observed"); assert.deepEqual(receipt.identity.hardware.devices, item.devices);
      assert.match(receipt.identity.hardware.stdout_sha256, /^[0-9a-f]{64}$/);
    }
  } finally {
    // Exact mkdtemp path remains inside the OS temporary directory before recursive cleanup.
    assert.ok(path.resolve(temporary).startsWith(path.resolve(tmpdir()) + path.sep));
    await rm(temporary, { recursive: true, force: true });
  }
});

test("system profiler failure records its concrete query error before build", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "qwen-app-profiler-")), output = path.join(temporary, "attempt");
  try {
    const unavailable = Object.assign(new Error("system_profiler unavailable"), { code: "ENOENT" });
    await assert.rejects(runLifecycle({ QWEN_APP_RUNNER_TARGET: "secondary", RUNNER_NAME: "nax-macos-2", RUNNER_TEMP: temporary, QWEN_APP_OUTPUT: output, QWEN_APP_SOURCE_SHA: sha, QWEN_APP_INFERENCE_SHA: "b".repeat(40), GITHUB_RUN_ID: "43", GITHUB_RUN_ATTEMPT: "1" }, { platform: "darwin", arch: "arm64", execSystemProfiler: async () => { throw unavailable; } }), /hardware inventory query failed: system_profiler unavailable/);
    const receipt = JSON.parse(await readFile(path.join(output, "evidence", "receipt.json"), "utf8"));
    assert.equal(receipt.status, "failed"); assert.deepEqual(receipt.identity.platform, { os: "darwin", arch: "arm64" });
    assert.deepEqual(receipt.identity.hardware, { status: "query_failed", query: ["system_profiler", "SPHardwareDataType", "-json"], error: { name: "Error", message: "system_profiler unavailable", code: "ENOENT", signal: null } });
  } finally {
    assert.ok(path.resolve(temporary).startsWith(path.resolve(tmpdir()) + path.sep));
    await rm(temporary, { recursive: true, force: true });
  }
});

test("owned child environment replaces a poisoned ambient Xet cache", () => {
  const owned = { state: "/runner/temp/state", weights: "/Users/MTrefry/sceneworks-rw-weights", hub: "/Users/MTrefry/sceneworks-rw-weights/hub", xet: "/Users/MTrefry/sceneworks-rw-weights/xet", workerId: "worker", url: "http://127.0.0.1:17921" };
  const child = buildChildEnv({ HOME: "/Users/MTrefry", HF_XET_CACHE: "/Volumes/shared-hf/xet", HF_HUB_CACHE: "/Volumes/shared-hf/hub", SCENEWORKS_DATA_DIR: "/poisoned" }, owned);
  assertOwnedCacheEnv(child, owned);
  assert.equal(child.HF_XET_CACHE, owned.xet);
  assert.equal(child.HF_HUB_CACHE, owned.hub);
  assert.equal(child.HF_HOME, owned.weights);
  assert.equal(child.SCENEWORKS_DATA_DIR, path.join(owned.state, "data"));
});

test("removing the explicit Xet assignment kills the owned-cache guard", () => {
  const owned = { state: "/runner/temp/state", weights: "/Users/MTrefry/sceneworks-rw-weights", hub: "/Users/MTrefry/sceneworks-rw-weights/hub", xet: "/Users/MTrefry/sceneworks-rw-weights/xet", workerId: "worker", url: "http://127.0.0.1:17921" };
  const mutant = buildChildEnv({ HF_XET_CACHE: "/Volumes/shared-hf/xet" }, owned);
  delete mutant.HF_XET_CACHE;
  assert.throws(() => assertOwnedCacheEnv(mutant, owned), /task-owned Hugging Face hub and Xet caches/);
});
