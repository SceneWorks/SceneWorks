// sc-24163 E11. No imports, synthetic completion, downloads, or memory campaign.
// The production API submits/resumes both jobs and registers their final adapters.
import assert from "node:assert/strict";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { cp, lstat, mkdir, open, readFile, readdir, realpath, rename, stat, writeFile, copyFile } from "node:fs/promises";
import { createServer } from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { deflateSync } from "node:zlib";

const execFile = promisify(execFileCallback);
export const BASE_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758";
export const LICENSE_URL = `https://huggingface.co/Qwen/Qwen-Image-2.1/blob/${BASE_REVISION}/LICENSE`;
const SHA = /^[0-9a-f]{40}$/;
const HASH = /^[0-9a-f]{64}$/;
const requireFact = (value, message) => assert.ok(value, message);

export function assertRunnerIdentity(target, runnerName) {
  const expected = { primary: "nax-macos", secondary: "nax-macos-2" }[target];
  requireFact(expected && runnerName === expected, "selected app runner target/name mismatch");
}

export function assertHardwareIdentity(hardware) {
  requireFact(hardware.length === 1 && hardware[0].chip_type?.includes("M5 Max") && hardware[0].physical_memory === "128 GB", "expected 128 GB M5 Max hardware required");
}

export function assertImmutableRevisions(source, inference) {
  requireFact(SHA.test(source) && SHA.test(inference), "immutable source and inference revisions required");
}

export function assertPins(manifests, lock, expected) {
  requireFact(SHA.test(expected), "exact inference SHA required");
  for (const manifest of manifests) {
    const pins = [...manifest.matchAll(/git\s*=\s*"https:\/\/github.com\/SceneWorks\/inference"\s*,\s*rev\s*=\s*"([^"]+)"/g)].map((match) => match[1]);
    requireFact(pins.length > 0 && pins.every((pin) => pin === expected), "every inference dependency must use the dispatched final pin");
  }
  const pins = [...lock.matchAll(/source = "git\+https:\/\/github.com\/SceneWorks\/inference\?rev=([^#]+)#([^"]+)"/g)];
  requireFact(pins.length > 0 && pins.every((match) => match[1] === expected && match[2] === expected), "Cargo.lock must resolve the exact final inference pin");
}

export function trainingBody(target, datasetId, mode) {
  requireFact(target?.id === (mode === "edit" ? "qwen_image_2_1_edit_lora" : "qwen_image_2_1_lora"), "exact Qwen 2.1 target required");
  requireFact(target.ui?.requiresLicenseAcknowledgment === true && target.ui.licenseUrl === LICENSE_URL, "target must expose the frozen research license");
  const config = structuredClone(target.defaults);
  Object.assign(config, { steps: 200, resolution: 768, rank: 4, alpha: 4, batchSize: 1, gradientAccumulation: 1, saveEvery: 50, seed: 24163 });
  Object.assign(config.advanced, { networkType: mode === "edit" ? "lokr" : "lora", outputScope: "project", gradientCheckpointing: true, sampleEvery: 100, sampleSteps: 8, sampleCount: 1, samplePrompts: [mode === "edit" ? "Turn the teal circle orange while preserving the background" : "A teal circle on a pale yellow background"] });
  return { targetId: target.id, datasetId, config, outputName: `sc24163-mlx-app-${mode}`, licenseAcknowledged: true, dryRun: false };
}

export function assertWorker(worker, id) {
  requireFact(worker?.id === id && worker.status === "idle" && worker.currentJobId == null, "owned worker must be registered and idle");
  requireFact(worker.gpuId === "mlx" && worker.gpuName === "Apple Silicon (MLX)", "native MLX device required");
  requireFact(Array.isArray(worker.capabilities) && worker.capabilities.includes("gpu") && worker.capabilities.includes("lora_train_execute") && !worker.capabilities.some((capability) => ["cpu", "candle", "nvidia"].includes(capability)), "native training worker capabilities required");
}

export function assertPlan(job, body, basePath) {
  const plan = job.payload?.plan;
  requireFact(job.type === "lora_train" && job.payload.dryRun === false && plan?.target?.targetId === body.targetId && plan.target.kernel === body.targetId, "typed API must route to the exact real trainer");
  requireFact(plan.target.baseModel === "qwen_image_2_1" && path.resolve(plan.target.baseModelPath) === path.resolve(basePath), "plan must resolve the immutable dense snapshot");
  requireFact(plan.config.steps === 200 && plan.config.resolution === 768 && plan.config.saveEvery === 50 && plan.config.advanced.sampleEvery === 100 && plan.config.advanced.networkType === body.config.advanced.networkType, "canonical plan must preserve valid lifecycle configuration");
  requireFact(plan.dataset.datasetId === body.datasetId && plan.dataset.items.length > 0, "canonical dataset required");
  for (const item of plan.dataset.items) requireFact(body.targetId.endsWith("edit_lora") ? item.referenceImagePaths?.length === 1 : !item.referenceImagePaths?.length, "edit plan must preserve ordered reference images");
  return plan;
}

export function assertCompleted(job, library, metadata, mode, workerId) {
  const result = job.result, plan = job.payload?.plan;
  const network = mode === "edit" ? "lokr" : "lora";
  requireFact(job.status === "completed" && job.stage === "completed" && job.backend === "mlx" && job.workerId === workerId && job.error == null, "owned MLX job must complete successfully");
  requireFact(result.backend === "mlx" && result.targetId === plan.target.targetId && result.kernel === plan.target.kernel && result.networkType === network && result.steps === 200 && result.stepsCompleted === 200 && Number.isFinite(result.finalLoss), "real trainer must finish all 200 steps");
  requireFact(result.checkpoints?.some((checkpoint) => checkpoint.step >= 50), "completed run must expose real checkpoints");
  requireFact(result.trainingSamples?.some((sample) => sample.step >= 100 && sample.sampleSource === "live_adapter" && sample.relativePath?.endsWith(".png")), "real live-adapter previews required");
  requireFact(library?.id === result.loraId && library.scope === "project" && library.installState === "installed" && library.source?.provider === "training" && library.source.path === `loras/${result.loraId}` && library.files?.includes(result.fileName), "completed final adapter must be automatically registered in the project library");
  requireFact(path.resolve(result.outputPath) === path.resolve(plan.output.outputDir, plan.output.fileName) && result.fileName === plan.output.fileName && path.resolve(library.installedPath) === path.resolve(plan.output.outputDir), "library must select the declared final adapter, not a checkpoint");
  requireFact(library.family === "qwen-image-2-1" && library.baseModel === "qwen_image_2_1" && library.networkType === network && library.license?.includes("Qwen") && library.licenseUrl === LICENSE_URL, "library derivative identity and license required");
  // Retry keeps the original manifest's provenance; sourceJobId binds that origin.
  requireFact(library.provenance?.trainingJobId === (job.sourceJobId ?? job.id) && library.provenance.targetId === result.targetId && library.provenance.datasetId === result.datasetId, "automatic library provenance must identify the accepted training job");
  requireFact(metadata.family === "qwen-image-2-1" && metadata.baseModel === "qwen_image_2_1" && metadata.ss_base_model_version === "qwen_image_2_1" && metadata.license?.includes("Qwen Research") && metadata["modelspec.license"] === metadata.license && metadata.licenseNotice?.includes("Qwen RESEARCH LICENSE AGREEMENT"), "on-disk adapter derivative metadata required");
  requireFact(metadata.networkType === network && Number(metadata.rank) === 4 && Number(metadata.alpha) === 4, "on-disk adapter must preserve the trained network and rank");
  requireFact(mode === "edit" ? metadata.trainingMode === "edit" : metadata.trainingMode == null, "adapter must preserve training mode");
  requireFact(mode === "edit" ? library.trainingMode === "edit" : library.trainingMode == null, "library must preserve training mode");
  return { job_id: job.id, target: result.targetId, network_type: network, steps_completed: result.stepsCompleted, final_loss: result.finalLoss, backend: result.backend, library_id: library.id };
}

export function assertResume(before, canceled, resumed, observation, checkpoint) {
  requireFact(canceled.id === before.id && canceled.status === "canceled" && canceled.canceledAt && canceled.workerId === observation.worker_id, "cancellation must be acknowledged by the owned running worker");
  requireFact(checkpoint.step >= 50 && checkpoint.step < 200 && checkpoint.metadata.training_config?.startsWith("steps=200;accum=1;") && HASH.test(checkpoint.metadata.request_fingerprint) && Number(checkpoint.metadata.step) === checkpoint.step && checkpoint.optimizer_bytes > 0, "real fingerprinted optimizer checkpoint required before cancellation");
  requireFact(resumed.id !== before.id && resumed.sourceJobId === before.id && resumed.payload.plan.config.advanced.resume === true, "retry must resume the original accepted job");
  const stripResume = (plan) => { const copy = structuredClone(plan); delete copy.config.advanced.resume; return copy; };
  assert.deepEqual(stripResume(resumed.payload.plan), stripResume(before.payload.plan), "resume must preserve the complete original training plan");
  requireFact(observation.first_training_step > checkpoint.step && observation.first_training_step <= checkpoint.step + 5, "resumed execution must continue near the saved step, without restarting");
}

// Deterministic, public synthetic inputs: no operator photographs or private datasets.
export function syntheticPng(edit = false) {
  const width = 768, height = 768, rows = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const circle = (x - 384) ** 2 + (y - 384) ** 2 < 180 ** 2;
    const rgb = circle ? (edit ? [240, 120, 40] : [30, 170, 175]) : [245, 235, 180];
    rows.set(rgb, y * (width * 3 + 1) + 1 + x * 3);
  }
  const chunk = (type, data) => {
    const tag = Buffer.from(type), bytes = Buffer.concat([tag, data]); let crc = 0xffffffff;
    for (const byte of bytes) { crc ^= byte; for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0); }
    const length = Buffer.alloc(4), checksum = Buffer.alloc(4); length.writeUInt32BE(data.length); checksum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
    return Buffer.concat([length, bytes, checksum]);
  };
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 2;
  return Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"), chunk("IHDR", header), chunk("IDAT", deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]);
}

async function hash(file) { const digest = createHash("sha256"); for await (const bytes of createReadStream(file)) digest.update(bytes); return digest.digest("hex"); }
async function safetensorsMetadata(file) {
  const handle = await open(file, "r");
  try {
    const length = Buffer.alloc(8); requireFact((await handle.read(length, 0, 8, 0)).bytesRead === 8, "safetensors header missing");
    const size = Number(length.readBigUInt64LE()); requireFact(size > 0 && size < 16 * 1024 * 1024, "safetensors header length invalid");
    const header = Buffer.alloc(size); requireFact((await handle.read(header, 0, size, 8)).bytesRead === size, "safetensors header truncated");
    const value = JSON.parse(header); requireFact(Object.keys(value).some((key) => key !== "__metadata__") && (await handle.stat()).size > size + 8, "adapter tensors missing");
    return value.__metadata__ ?? {};
  } finally { await handle.close(); }
}
async function physicalDirectory(directory) {
  const entry = await lstat(directory); requireFact(entry.isDirectory() && !entry.isSymbolicLink() && await realpath(directory) === directory, "physical task-owned directory required");
}
export async function assertDenseSnapshotLayout(base) {
  // Frozen Qwen 2.1 uses Qwen3-VL's processor tree, not the older tokenizer/ layout.
  for (const file of ["model_index.json", "transformer/config.json", "text_encoder/config.json", "vae/config.json", "processor/tokenizer_config.json", "processor/tokenizer.json"]) {
    const entry = await stat(path.join(base, file)); requireFact(entry.isFile() && entry.size > 0, `staged dense snapshot incomplete: ${file}`);
  }
}
async function snapshotInventory(base, hub) {
  const files = [];
  const walk = async (directory) => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name), resolved = await realpath(file);
      requireFact(resolved.startsWith(hub + path.sep), "snapshot file must stay inside the physical task hub");
      const info = await stat(resolved);
      if (info.isDirectory()) { requireFact(!entry.isSymbolicLink(), "snapshot component cannot redirect to another tree"); await walk(file); }
      else { requireFact(info.isFile() && info.size > 0 && !entry.name.endsWith(".incomplete"), "complete nonempty model file required"); files.push({ file: path.relative(base, file), bytes: info.size }); }
    }
  };
  await walk(base);
  for (const component of ["transformer", "text_encoder", "vae"]) requireFact(files.some((entry) => entry.file.startsWith(component + path.sep) && entry.file.endsWith(".safetensors")), "dense component weights missing");
  return files.sort((a, b) => a.file.localeCompare(b.file));
}
async function assertOwnedFile(file, state) {
  const resolved = await realpath(file), relative = path.relative(state, resolved);
  requireFact(relative && !relative.startsWith("..") && !path.isAbsolute(relative) && (await stat(resolved)).isFile() && (await stat(resolved)).size > 0, "nonempty artifact must stay inside owned state");
  return resolved;
}

export async function runLifecycle(env = process.env, runtime = {}) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), ".."), output = path.resolve(env.QWEN_APP_OUTPUT ?? "");
  requireFact(env.QWEN_APP_OUTPUT && env.RUNNER_TEMP && output.startsWith(path.resolve(env.RUNNER_TEMP) + path.sep), "output must be a unique directory under RUNNER_TEMP");
  await mkdir(output, { recursive: true }); const evidence = path.join(output, "evidence"); await mkdir(evidence, { recursive: true });
  const platform = { os: runtime.platform ?? process.platform, arch: runtime.arch ?? process.arch };
  const receipt = { schema_version: 1, kind: "qwen_image_2_1_mlx_app_lifecycle", status: "running", phase: runtime.preflightOnly ? "preflight" : "lifecycle", started_at: new Date().toISOString(), run_id: env.GITHUB_RUN_ID, run_attempt: env.GITHUB_RUN_ATTEMPT, identity: { requested_runner_target: env.QWEN_APP_RUNNER_TARGET, runner_name: env.RUNNER_NAME, platform, sceneworks_revision: env.QWEN_APP_SOURCE_SHA, inference_revision: env.QWEN_APP_INFERENCE_SHA, run_id: env.GITHUB_RUN_ID, run_attempt: env.GITHUB_RUN_ATTEMPT, hardware: { status: "pending", query: ["system_profiler", "SPHardwareDataType", "-json"] } }, cases: [] };
  const save = async () => { const temporary = path.join(evidence, "receipt.tmp"); await writeFile(temporary, JSON.stringify(receipt, null, 2) + "\n"); await rename(temporary, path.join(evidence, "receipt.json")); };
  await save();
  const children = [], handles = [], controller = new AbortController();
  const abort = () => controller.abort(new Error("app lifecycle interrupted"));
  process.once("SIGTERM", abort); process.once("SIGINT", abort);
  const deadline = setTimeout(() => controller.abort(new Error("app lifecycle exceeded ten hours")), 10 * 60 * 60 * 1000);
  const delay = (ms) => new Promise((resolve, reject) => { if (controller.signal.aborted) return reject(controller.signal.reason); const done = () => { clearTimeout(timer); controller.signal.removeEventListener("abort", interrupted); }; const interrupted = () => { done(); reject(controller.signal.reason); }; const timer = setTimeout(() => { done(); resolve(); }, ms); controller.signal.addEventListener("abort", interrupted, { once: true }); });
  const alive = () => { if (controller.signal.aborted) throw controller.signal.reason; for (const child of children.filter((child) => child.service)) requireFact(child.exitCode === null && child.signalCode === null, "owned API or worker exited"); };
  const launch = async (command, args, childEnv, name, service = false) => {
    const log = await open(path.join(evidence, `${name}.log`), "wx"); handles.push(log);
    const child = spawn(command, args, { cwd: root, env: childEnv, detached: true, stdio: ["ignore", log.fd, log.fd] }); child.service = service; children.push(child);
    // Install listeners immediately; spawn failures never become unhandled events.
    child.done = new Promise((resolve, reject) => { child.once("error", reject); child.once("exit", (code, signal) => resolve({ code, signal })); });
    if (service) child.done.catch((error) => controller.abort(error));
    return child;
  };
  const checkedCommand = async (command, args, childEnv, name) => {
    const child = await launch(command, args, childEnv, name);
    let rejectInterrupted;
    const interruption = new Promise((resolve, reject) => { rejectInterrupted = reject; });
    const interrupted = () => rejectInterrupted(controller.signal.reason);
    controller.signal.addEventListener("abort", interrupted, { once: true });
    try { alive(); const result = await Promise.race([child.done, interruption]); alive(); requireFact(result.code === 0, `${name} failed; see owned log`); } finally { controller.signal.removeEventListener("abort", interrupted); }
  };
  try {
    try {
      const result = await (runtime.execSystemProfiler ?? execFile)("system_profiler", ["SPHardwareDataType", "-json"], { timeout: 30_000 });
      const raw = result.stdout, parsed = JSON.parse(raw), devices = parsed.SPHardwareDataType;
      requireFact(Array.isArray(devices), "system_profiler output omitted SPHardwareDataType");
      receipt.identity.hardware = {
        status: "observed",
        query: ["system_profiler", "SPHardwareDataType", "-json"],
        stdout_sha256: createHash("sha256").update(raw).digest("hex"),
        devices: devices.map(({ chip_type, physical_memory, machine_model }) => ({ chip_type, physical_memory, machine_model })),
      };
    } catch (error) {
      receipt.identity.hardware = {
        status: "query_failed",
        query: ["system_profiler", "SPHardwareDataType", "-json"],
        error: { name: error.name ?? "Error", message: String(error.message ?? error), code: error.code ?? null, signal: error.signal ?? null },
      };
      await save();
      throw new Error(`hardware inventory query failed: ${receipt.identity.hardware.error.message}`);
    }
    await save();
    assertRunnerIdentity(env.QWEN_APP_RUNNER_TARGET, env.RUNNER_NAME);
    assertImmutableRevisions(env.QWEN_APP_SOURCE_SHA, env.QWEN_APP_INFERENCE_SHA);
    requireFact(platform.os === "darwin" && platform.arch === "arm64", "Darwin arm64 runner required");
    assertHardwareIdentity(receipt.identity.hardware.devices);
    const git = async (...args) => (await execFile("git", args, { cwd: root })).stdout.trim();
    requireFact(await git("rev-parse", "HEAD") === env.QWEN_APP_SOURCE_SHA && await git("status", "--porcelain") === "", "exact clean feature checkout required");
    const manifests = await Promise.all(["Cargo.toml", "crates/sceneworks-worker/Cargo.toml", "crates/sceneworks-memory-adapter/Cargo.toml"].map((file) => readFile(path.join(root, file), "utf8")));
    assertPins(manifests, await readFile(path.join(root, "Cargo.lock"), "utf8"), env.QWEN_APP_INFERENCE_SHA);
    if (runtime.preflightOnly) { receipt.status = "passed"; return receipt; }
    const home = await realpath(env.HOME), weights = path.join(home, "sceneworks-rw-weights"), hub = path.join(weights, "hub"), repo = path.join(hub, "models--Qwen--Qwen-Image-2.1"), snapshots = path.join(repo, "snapshots"), base = path.join(snapshots, BASE_REVISION);
    for (const directory of [weights, hub, repo, snapshots, base]) await physicalDirectory(directory);
    requireFact((await readdir(snapshots)).every((name) => name === BASE_REVISION), "dense repo must contain only the frozen snapshot");
    await assertDenseSnapshotLayout(base);
    receipt.snapshot_files = await snapshotInventory(base, hub);
    const state = path.join(output, "state"); await mkdir(state); await mkdir(path.join(state, "data")); await mkdir(path.join(state, "credentials")); await cp(path.join(root, "config"), path.join(state, "config"), { recursive: true });
    const workerId = `qwen-app-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`, url = "http://127.0.0.1:17921";
    // Exclusive bind fails if another app already owns this port; never attach to it.
    const socket = createServer(); await new Promise((resolve, reject) => { socket.once("error", reject); socket.listen({ host: "127.0.0.1", port: 17921, exclusive: true }, resolve); }); await new Promise((resolve) => socket.close(resolve));
    const childEnv = { ...env };
    for (const key of Object.keys(childEnv)) if (key.startsWith("SCENEWORKS_") || /TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(key) || ["HF_ENDPOINT", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "CARGO_TARGET_DIR"].includes(key)) delete childEnv[key];
    Object.assign(childEnv, { SCENEWORKS_DATA_DIR: path.join(state, "data"), SCENEWORKS_CONFIG_DIR: path.join(state, "config"), SCENEWORKS_JOBS_DB_PATH: path.join(state, "data", "cache", "jobs.db"), SCENEWORKS_CREDENTIALS_DIR: path.join(state, "credentials"), SCENEWORKS_API_HOST: "127.0.0.1", SCENEWORKS_API_PORT: "17921", SCENEWORKS_API_URL: url, SCENEWORKS_WORKER_ID: workerId, SCENEWORKS_WORKER_CHILD: "1", SCENEWORKS_GPU_ID: "mlx", SCENEWORKS_POLL_SECONDS: "1", SCENEWORKS_HEARTBEAT_SECONDS: "2", SCENEWORKS_BACKEND_MLX_ENABLED: "true", SCENEWORKS_BACKEND_CANDLE_ENABLED: "false", SCENEWORKS_MLX_REQUIRED: "1", SCENEWORKS_MLX_UNSUPPORTED_MODE: "enforce", HF_HOME: weights, HF_HUB_CACHE: hub, HUGGINGFACE_HUB_CACHE: hub, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" });
    Object.assign(receipt.identity, { base_revision: BASE_REVISION, base_snapshot: base, worker_id: workerId });
    await save();
    await checkedCommand("cargo", ["build", "--release", "--locked", "-p", "sceneworks-rust-api"], { ...env, CARGO_TARGET_DIR: path.join(root, "target") }, "build");
    requireFact(await git("status", "--porcelain") === "", "build changed the pinned checkout");
    const binary = path.join(root, "target", "release", "sceneworks-rust-api"); receipt.identity.binary_sha256 = await hash(binary);
    await checkedCommand(binary, [], { ...childEnv, SCENEWORKS_GPU_CHECK: "1" }, "metal-preflight");
    await launch(binary, [], childEnv, "api", true);
    await launch(binary, [], { ...childEnv, SCENEWORKS_WORKER_ONLY: "1" }, "worker", true);
    const call = async (method, route, body) => { alive(); const response = await fetch(url + route, { method, signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]), headers: body instanceof FormData ? {} : { "content-type": "application/json" }, body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body) }); const value = await response.json(); return { status: response.status, value }; };
    const must = async (...args) => { const response = await call(...args); requireFact(response.status < 300, `API ${args[0]} ${args[1]} refused (${response.status}): ${JSON.stringify(response.value)}`); return response.value; };
    const wait = async (fn, seconds, label) => { const end = Date.now() + seconds * 1000; let last; while (Date.now() < end) { alive(); last = await fn(); if (last) return last; await delay(1000); } throw new Error(`${label} timed out`); };
    receipt.health = await wait(async () => { try { const health = await must("GET", "/api/v1/health"); return health.status === "ok" && health.readiness?.status === "ready" && health; } catch (error) { alive(); return false; } }, 180, "API readiness");
    receipt.worker = await wait(async () => { const workers = await must("GET", "/api/v1/workers"); const worker = workers.find((item) => item.id === workerId); if (!worker || worker.status !== "idle") return false; assertWorker(worker, workerId); return worker; }, 180, "MLX worker registration");
    // Capture the production volume binding in the isolated config, then read it back.
    const relocation = await must("POST", "/api/v1/model-library/relocate", { path: weights }), libraryProbe = await must("GET", "/api/v1/model-library");
    requireFact(relocation.adopted === true && relocation.hfHome === weights && relocation.libraryRoot === hub && libraryProbe.available === true && libraryProbe.probeStatus === "available" && libraryProbe.configuredLibraryPath === hub && libraryProbe.expectedLibrary?.configuredPath === hub, "product must bind exactly the persistent task-owned model library");
    receipt.model_library = { hf_home: weights, library_root: hub, adopted: relocation.adopted, probe_status: libraryProbe.probeStatus };
    const project = await must("POST", "/api/v1/projects", { name: "sc-24163 public synthetic MLX lifecycle" }); receipt.project_id = project.id;
    const upload = async (name, data) => { await writeFile(path.join(evidence, name), data); const form = new FormData(); form.append("file", new Blob([data], { type: "image/png" }), name); return (await must("POST", `/api/v1/projects/${project.id}/assets`, form)).id; };
    const source = await upload("source.png", syntheticPng()), target = await upload("target.png", syntheticPng(true));
    const datasets = {};
    for (const mode of ["t2i", "edit"]) datasets[mode] = await must("POST", `/api/v1/projects/${project.id}/training/datasets`, { name: `public-synthetic-${mode}`, items: [{ assetId: mode === "edit" ? target : source, caption: { text: mode === "edit" ? "Turn the teal circle orange while preserving the background" : "A teal circle on a pale yellow background" }, ...(mode === "edit" ? { references: [{ assetId: source }] } : {}) }] });
    const targets = await must("GET", "/api/v1/training/targets"), allTargets = targets.targets ?? targets;
    const getJob = (id) => must("GET", `/api/v1/jobs/${id}`);
    for (const mode of ["t2i", "edit"]) {
      const body = trainingBody(allTargets.find((item) => item.id === (mode === "edit" ? "qwen_image_2_1_edit_lora" : "qwen_image_2_1_lora")), datasets[mode].id, mode), route = `/api/v1/projects/${project.id}/training/jobs`;
      const denied = await call("POST", route, { ...body, licenseAcknowledged: false }); requireFact(denied.status === 403 && /licen[cs]e/i.test(JSON.stringify(denied.value)), "unacknowledged negative twin must return license 403");
      const accepted = await must("POST", route, body); assertPlan(accepted, body, base);
      const caseRecord = { mode, accepted_job: accepted, license_negative_twin: denied, events: [] }; receipt.cases.push(caseRecord); await save();
      const observe = async (id) => { const job = await getJob(id); if (caseRecord.events.at(-1)?.message !== job.message || caseRecord.events.at(-1)?.status !== job.status) { caseRecord.events.push({ id, status: job.status, stage: job.stage, message: job.message, worker_id: job.workerId, backend: job.backend, observed_at: new Date().toISOString() }); await save(); } requireFact(!["failed", "canceled"].includes(job.status), `job ${id} unexpectedly ${job.status}: ${job.error}`); return job; };
      let running = accepted;
      if (mode === "t2i") {
        const checkpoint = await wait(async () => { const job = await observe(accepted.id); requireFact(job.status !== "completed", "T2I completed before cancellation proof"); const step = /^Training step (\d+) of 200\b/.exec(job.message); if (!(job.message === "Saved checkpoint at step 50." || (step && Number(step[1]) > 50))) return false; const dir = path.resolve(accepted.payload.plan.output.outputDir); const names = await readdir(dir).catch((error) => error.code === "ENOENT" ? [] : Promise.reject(error)); const name = names.find((item) => /-step000050\.resume\.safetensors$/.test(item)); if (!name) return false; const file = await assertOwnedFile(path.join(dir, name), state), metadata = await safetensorsMetadata(file), optim = await assertOwnedFile(file.replace(".resume.safetensors", ".optim.safetensors"), state); return { step: 50, file: path.relative(output, file), sha256: await hash(file), metadata, optimizer_bytes: (await stat(optim)).size, optimizer_sha256: await hash(optim) }; }, 3 * 3600, "first fingerprinted checkpoint");
        requireFact(checkpoint.metadata.training_config?.startsWith("steps=200;accum=1;") && HASH.test(checkpoint.metadata.request_fingerprint), "resume identity must be present before cancel");
        await must("POST", `/api/v1/jobs/${accepted.id}/cancel`, {});
        const canceled = await wait(async () => { const job = await getJob(accepted.id); requireFact(job.status !== "completed" && job.status !== "failed", "cancel must stop the running trainer"); return job.status === "canceled" && job; }, 300, "worker cancellation");
        await wait(async () => { const workers = await must("GET", "/api/v1/workers"); const worker = workers.find((item) => item.id === workerId); return worker?.status === "idle" && worker.currentJobId == null; }, 180, "worker drain after cancellation");
        const plan = structuredClone(canceled.payload.plan); plan.config.advanced.resume = true;
        running = await must("POST", `/api/v1/jobs/${canceled.id}/retry`, { payloadChanges: { plan } });
        const resumedStep = await wait(async () => { const job = await observe(running.id); const match = /^Training step (\d+) of 200\b/.exec(job.message); return match && { first_training_step: Number(match[1]), worker_id: job.workerId }; }, 3600, "first resumed training step");
        assertResume(accepted, canceled, running, resumedStep, checkpoint); caseRecord.resume = { canceled_job: canceled, retry_job: running, checkpoint, observation: resumedStep }; await save();
      }
      const completed = await wait(async () => { const job = await observe(running.id); return job.status === "completed" && job; }, 4 * 3600, `${mode} completion`);
      const library = await wait(async () => (await must("GET", `/api/v1/loras?projectId=${project.id}`)).find((entry) => entry.id === completed.result.loraId), 60, "automatic training registration");
      const adapter = await assertOwnedFile(completed.result.outputPath, state), metadata = await safetensorsMetadata(adapter);
      caseRecord.completed_job = completed; caseRecord.library_entry = library; caseRecord.adapter = { file: path.relative(output, adapter), bytes: (await stat(adapter)).size, sha256: await hash(adapter), metadata };
      caseRecord.summary = assertCompleted(completed, library, metadata, mode, workerId); caseRecord.previews = [];
      for (const [index, sample] of completed.result.trainingSamples.entries()) {
        const file = await assertOwnedFile(sample.path, state), response = await fetch(`${url}/api/v1/projects/${project.id}/files/${sample.relativePath}`, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]) }); requireFact(response.ok, "preview must be served by the product file route");
        const bytes = Buffer.from(await response.arrayBuffer()); requireFact(bytes.subarray(0, 8).toString("hex") === "89504e470d0a1a0a" && bytes.readUInt32BE(16) > 0 && bytes.readUInt32BE(20) > 0 && createHash("sha256").update(bytes).digest("hex") === await hash(file), "served preview must match the real saved PNG");
        const name = `${mode}-preview-${index}.png`; await copyFile(file, path.join(evidence, name)); caseRecord.previews.push({ file: name, step: sample.step, sha256: await hash(file), width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20), api_status: response.status });
      }
      await save();
    }
    requireFact(await git("status", "--porcelain") === "", "runtime changed the immutable checkout"); receipt.status = "passed";
  } catch (error) { receipt.status = "failed"; receipt.error = error.message; throw error; }
  finally {
    clearTimeout(deadline); process.removeListener("SIGTERM", abort); process.removeListener("SIGINT", abort);
    receipt.cleanup = [];
    for (const child of children.reverse()) {
      if (!child.pid) continue;
      try {
        if (child.exitCode === null && child.signalCode === null) {
          process.kill(-child.pid, "SIGTERM");
          await Promise.race([child.done.catch(() => {}), new Promise((resolve) => setTimeout(resolve, 15_000))]);
          if (child.exitCode === null && child.signalCode === null) { process.kill(-child.pid, "SIGKILL"); await child.done.catch(() => {}); }
        }
        receipt.cleanup.push({ pid: child.pid, exited: child.exitCode !== null || child.signalCode !== null });
      } catch (error) { receipt.status = "failed"; receipt.cleanup.push({ pid: child.pid, exited: false, error: error.message }); process.exitCode = 1; }
    }
    await Promise.all(handles.map((handle) => handle.close()));
    receipt.finished_at = new Date().toISOString(); await save();
  }
  requireFact(receipt.status === "passed", "lifecycle cleanup failed"); return receipt;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  runLifecycle(process.env, { preflightOnly: args.length === 1 && args[0] === "--preflight-only" }).then((receipt) => console.log(`${receipt.status}: ${receipt.cases.length} completed app jobs`)).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
