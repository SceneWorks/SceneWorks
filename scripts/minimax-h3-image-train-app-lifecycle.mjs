// SW-7103 AC1 (feature SW-F363): ONE ordinary SceneWorks app lifecycle per backend, real weights.
// The owned API installs the MiniMax-H3-Image tier through the Model Manager download route, trains
// a text-to-image LoRA and an edit-pair LoRA through the Training Studio training-job route, checks
// that each final adapter is registered in the project LoRA library, and then renders through the
// Image Studio image-job route with that LoRA applied (plus the same request without it, so the
// receipt states how much the LoRA changed the pixels). Dispatch-only: reached from the manual
// memory-catalog dispatcher, never from push, pull_request or schedule.
//
// Reuses the Qwen-Image 2.1 (sc-24163) and FLUX3 (sc-26234) app-lifecycle helpers unchanged.
import assert from "node:assert/strict";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { appendFile, copyFile, cp, mkdir, open, readFile, readdir, realpath, rename, stat, statfs, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { deflateSync, inflateSync } from "node:zlib";
import { fileSha256 as hash, sha256 } from "./lib/file-sha256.mjs";
import { enterMacMetalExecution, metalChildOptions } from "./lib/apple-metal-execution-lock.mjs";
import { bindMetallib } from "./ci/memory-catalog/bind-metallib.mjs";
import { assertPins } from "./qwen-image-2-1-app-lifecycle.mjs";
import { attemptIdentity, assertDownloadWorker, assertServedPng, bindOwnedEnv, boundApiUrl, ownedApiController, redactSensitiveText } from "./flux3-image-klein-app-lifecycle.mjs";

const execFile = promisify(execFileCallback);
const fact = (value, message) => assert.ok(value, message);
export const MODEL_ID = "minimax_h3_image";
export const FAMILY = "minimax-h3-image";
export const LICENSE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/LICENSE";
export const TIERS = ["bf16", "q8", "q4"];
export const TARGETS = { t2i: "minimax_h3_image_lora", edit: "minimax_h3_image_edit_lora" };
// The smallest run the targets' limits admit: 200 steps at the 1024 training canvas.
export const TRAINING = { steps: 200, resolution: 1024, rank: 8, alpha: 8, saveEvery: 100, seed: 7103 };
export const RENDER = { width: 1024, height: 1024, seed: 7103 };
export const CAPTIONS = {
  t2i: "A teal circle on a pale yellow background",
  edit: "Turn the teal circle in <Picture 1> orange while preserving the background",
};
const GIB = 1024 ** 3;
// Safety margins, not measurements: the release build target and the attempt state.
export const BUILD_HEADROOM_BYTES = 40 * GIB;
export const STATE_HEADROOM_BYTES = 10 * GIB;
// Each backend's runner: the Mac by name (nax-macos-2 carries rw-mage; nax-macos is never used),
// the Windows CUDA real-weights host by its listener names.
export const RUNNERS = { mlx: ["nax-macos-2"], cuda: ["cuda-windows", "cuda-windows-2"] };

export function assertSelection(env, platform = process.platform) {
  fact(["mlx", "cuda"].includes(env.H3I_APP_BACKEND), "H3I_APP_BACKEND must be mlx or cuda");
  fact(TIERS.includes(env.H3I_APP_TIER), "H3I_APP_TIER must be bf16, q8 or q4");
  fact(/^[a-f0-9]{40}$/.test(env.H3I_APP_SOURCE_SHA ?? "") && /^[a-f0-9]{40}$/.test(env.H3I_APP_INFERENCE_SHA ?? ""), "exact 40-hex source_sha and inference_sha required");
  fact(env.H3I_APP_SOURCE_SHA === env.GITHUB_SHA, "the dispatched commit must be the selected source_sha");
  fact(platform === (env.H3I_APP_BACKEND === "mlx" ? "darwin" : "win32"), `${env.H3I_APP_BACKEND} needs ${env.H3I_APP_BACKEND === "mlx" ? "macOS" : "Windows"}`);
  fact(RUNNERS[env.H3I_APP_BACKEND].includes(env.RUNNER_NAME), `runner ${env.RUNNER_NAME} is not an admitted ${env.H3I_APP_BACKEND} host`);
  return { backend: env.H3I_APP_BACKEND, tier: env.H3I_APP_TIER };
}

// The download rows a tier install fetches: the tier's primary row plus every co-requisite that is
// tier-agnostic or scoped to that tier, narrowed to this OS. Used only for the disk budget.
export function tierRows(entry, tier, os) {
  const platform = { darwin: "macos", win32: "windows", linux: "linux" }[os] ?? os;
  return (entry.downloads ?? []).filter((row) =>
    (!row.platforms || row.platforms.includes(platform)) &&
    (row.coRequisite ? !row.variant || row.variant === tier : row.variant === tier));
}

export function assertDiskBudget({ freeBytes, missingBytes, includeBuild }) {
  const need = missingBytes + STATE_HEADROOM_BYTES + (includeBuild ? BUILD_HEADROOM_BYTES : 0);
  const gb = (bytes) => `${(bytes / 1e9).toFixed(1)} GB`;
  fact(Number.isFinite(freeBytes) && freeBytes >= need, `insufficient free disk: ${gb(freeBytes)} free, ${gb(need)} required (${gb(missingBytes)} of the tier not yet installed${includeBuild ? ` + ${gb(BUILD_HEADROOM_BYTES)} build` : ""} + ${gb(STATE_HEADROOM_BYTES)} state). Free space on this runner before re-dispatch.`);
  return { free_bytes: freeBytes, required_bytes: need, missing_tier_bytes: missingBytes };
}

// Bytes of the tier's rows whose pinned snapshot subdirectory is not on disk yet (an estimate: a
// present subdirectory counts as installed, so a partial earlier download is under-counted).
export async function missingTierBytes(hub, rows) {
  let missing = 0;
  for (const row of rows) {
    const subdir = row.subdir ?? row.files?.[0]?.split("/").slice(0, -1).join("/") ?? "";
    const dir = path.join(hub, `models--${row.repo.replaceAll("/", "--")}`, "snapshots", row.revision, subdir);
    const present = await stat(dir).then((info) => info.isDirectory(), () => false);
    if (!present) missing += row.estimatedSizeBytes ?? 0;
  }
  return missing;
}

export function trainingBody(target, datasetId, mode, tier) {
  fact(target?.id === TARGETS[mode], `exact ${TARGETS[mode]} target required`);
  fact(target.family === FAMILY && target.baseModel === MODEL_ID, "target must train the MiniMax-H3-Image base");
  fact(target.ui?.requiresLicenseAcknowledgment === true && target.ui.licenseUrl === LICENSE_URL, "target must expose the MiniMax-H3-Image licence gate");
  fact(target.limits?.baseTiers?.includes(tier), `target must offer the ${tier} base tier`);
  const config = structuredClone(target.defaults);
  Object.assign(config, { steps: TRAINING.steps, resolution: TRAINING.resolution, rank: TRAINING.rank, alpha: TRAINING.alpha, batchSize: 1, gradientAccumulation: 1, saveEvery: TRAINING.saveEvery, seed: TRAINING.seed });
  Object.assign(config.advanced, { networkType: "lora", baseTier: tier, outputScope: "project", gradientCheckpointing: true, sampleEvery: 0 });
  return { targetId: target.id, datasetId, config, outputName: `sw7103-${mode}-${tier}`, licenseAcknowledged: true, dryRun: false };
}

export function assertPlan(job, body, mode) {
  const plan = job?.payload?.plan;
  fact(job.type === "lora_train" && job.payload.dryRun === false && plan?.target?.targetId === body.targetId && plan.target.kernel === body.targetId && plan.target.baseModel === MODEL_ID, "typed API must route to the real H3-Image trainer");
  fact(plan.config.steps === TRAINING.steps && plan.config.resolution === TRAINING.resolution && plan.config.rank === TRAINING.rank && plan.config.advanced.baseTier === body.config.advanced.baseTier, "canonical plan must keep the requested steps, resolution, rank and tier");
  fact(plan.dataset?.datasetId === body.datasetId && plan.dataset.items?.length > 0, "canonical dataset required");
  for (const item of plan.dataset.items) fact(mode === "edit" ? item.referenceImagePaths?.length === 1 : !item.referenceImagePaths?.length, "edit plan must keep its ordered reference; t2i plan carries none");
  return plan;
}

export function assertTrained(job, library, metadata, mode, backend, workerId) {
  const result = job.result, plan = job.payload?.plan;
  fact(job.status === "completed" && job.workerId === workerId && job.error == null, "owned worker must complete the training job");
  fact(result?.backend === backend && result.targetId === plan.target.targetId && result.networkType === "lora" && result.stepsCompleted === TRAINING.steps && Number.isFinite(result.finalLoss), `real ${backend} trainer must finish all ${TRAINING.steps} steps`);
  fact(library?.id === result.loraId && library.scope === "project" && library.installState === "installed" && library.source?.provider === "training" && library.files?.includes(result.fileName), "the final adapter must be registered in the project LoRA library");
  fact(path.resolve(result.outputPath) === path.resolve(plan.output.outputDir, plan.output.fileName) && path.resolve(library.installedPath) === path.resolve(plan.output.outputDir), "the library must select the declared final adapter, not a checkpoint");
  fact(library.family === FAMILY && library.baseModel === MODEL_ID && library.networkType === "lora" && library.licenseUrl === LICENSE_URL, "library entry must carry the H3-Image family, base and licence");
  fact(library.provenance?.trainingJobId === (job.sourceJobId ?? job.id) && library.provenance.targetId === result.targetId, "library provenance must name the training job");
  fact(mode === "edit" ? library.trainingMode === "edit" && metadata.trainingMode === "edit" : library.trainingMode == null && metadata.trainingMode == null, "adapter and library must keep the training mode");
  fact(metadata.family === FAMILY && Number(metadata.rank) === TRAINING.rank, "on-disk adapter must carry the H3-Image family and the trained rank");
  return { job_id: job.id, lora_id: library.id, target: result.targetId, backend: result.backend, steps_completed: result.stepsCompleted, final_loss: result.finalLoss, file: result.fileName };
}

export function imageBody(projectId, mode, tier, loraId, referenceAssetId) {
  const bits = { bf16: 0, q8: 8, q4: 4 }[tier];
  return { projectId, mode: mode === "edit" ? "edit_image" : "text_to_image", model: MODEL_ID, prompt: CAPTIONS[mode], negativePrompt: "", seed: RENDER.seed, width: RENDER.width, height: RENDER.height, count: 1, advanced: { mlxQuantize: bits }, loras: loraId ? [{ id: loraId, weight: 1 }] : [], ...(mode === "edit" ? { referenceAssetIds: [referenceAssetId] } : {}) };
}

export function assertRendered(job, asset, body, backend, workerId) {
  fact(job.status === "completed" && job.workerId === workerId && job.error == null, "owned worker must complete the image job");
  fact(job.result?.model === MODEL_ID && job.result.assetIds?.length === 1 && asset?.id === job.result.assetIds[0] && asset.type === "image", "the image job must save one project-library image");
  const loras = (asset.recipe?.loras ?? []).map((lora) => lora.id);
  assert.deepEqual(loras, body.loras.map((lora) => lora.id), "the saved recipe must record exactly the applied LoRAs");
  fact(asset.recipe?.model === MODEL_ID && asset.recipe.mode === body.mode && asset.recipe.seed === body.seed && asset.recipe.prompt === body.prompt, "saved recipe must keep the request");
  return { job_id: job.id, asset_id: asset.id, backend, mode: body.mode, loras, media_path: asset.file?.path };
}

// Deterministic public synthetic inputs (no operator photographs), at the training canvas.
export function syntheticPng(edit = false, size = TRAINING.resolution) {
  const rows = Buffer.alloc((size * 3 + 1) * size), center = size / 2, radius = (size * 15) / 64;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const rgb = (x - center) ** 2 + (y - center) ** 2 < radius ** 2 ? (edit ? [240, 120, 40] : [30, 170, 175]) : [245, 235, 180];
    rows.set(rgb, y * (size * 3 + 1) + 1 + x * 3);
  }
  const chunk = (type, data) => {
    const bytes = Buffer.concat([Buffer.from(type), data]); let crc = 0xffffffff;
    for (const byte of bytes) { crc ^= byte; for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0); }
    const length = Buffer.alloc(4), checksum = Buffer.alloc(4); length.writeUInt32BE(data.length); checksum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
    return Buffer.concat([length, bytes, checksum]);
  };
  const header = Buffer.alloc(13); header.writeUInt32BE(size); header.writeUInt32BE(size, 4); header[8] = 8; header[9] = 2;
  return Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"), chunk("IHDR", header), chunk("IDAT", deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]);
}

// Decode an 8-bit RGB/RGBA non-interlaced PNG to its raw samples (the generator's output format).
export function decodePng(bytes) {
  fact(bytes.subarray(0, 8).toString("hex") === "89504e470d0a1a0a", "PNG signature required");
  let offset = 8, width = 0, height = 0, channels = 0; const data = [];
  while (offset < bytes.length) {
    const length = bytes.readUInt32BE(offset), type = bytes.toString("ascii", offset + 4, offset + 8), body = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") { width = body.readUInt32BE(0); height = body.readUInt32BE(4); fact(body[8] === 8 && [2, 6].includes(body[9]) && body[12] === 0, "8-bit non-interlaced RGB/RGBA PNG required"); channels = body[9] === 6 ? 4 : 3; }
    if (type === "IDAT") data.push(body);
    offset += 12 + length;
  }
  const raw = inflateSync(Buffer.concat(data)), stride = width * channels, out = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)], line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? out[y * stride + x - channels] : 0, b = y ? out[(y - 1) * stride + x] : 0, c = x >= channels && y ? out[(y - 1) * stride + x - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const predictor = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
      fact(predictor !== undefined, `unknown PNG filter ${filter}`);
      out[y * stride + x] = (line[x] + predictor) & 0xff;
    }
  }
  return { width, height, channels, pixels: out };
}

// Mean absolute difference over RGB samples (0-255) between two same-size renders.
export function meanAbsoluteDifference(first, second) {
  const a = decodePng(first), b = decodePng(second);
  fact(a.width === b.width && a.height === b.height, "renders must share a size");
  let total = 0, count = 0;
  for (let i = 0; i < a.width * a.height; i++) for (let c = 0; c < 3; c++) { total += Math.abs(a.pixels[i * a.channels + c] - b.pixels[i * b.channels + c]); count++; }
  return total / count;
}

async function safetensorsMetadata(file) {
  const handle = await open(file, "r");
  try {
    const length = Buffer.alloc(8); fact((await handle.read(length, 0, 8, 0)).bytesRead === 8, "safetensors header missing");
    const size = Number(length.readBigUInt64LE()); fact(size > 0 && size < 16 * 1024 * 1024, "safetensors header length invalid");
    const header = Buffer.alloc(size); fact((await handle.read(header, 0, size, 8)).bytesRead === size, "safetensors header truncated");
    const value = JSON.parse(header); fact(Object.keys(value).some((key) => key !== "__metadata__"), "adapter tensors missing");
    return value.__metadata__ ?? {};
  } finally { await handle.close(); }
}

async function ownedFile(file, root) {
  const resolved = await realpath(file), relative = path.relative(await realpath(root), resolved);
  fact(relative && !relative.startsWith("..") && !path.isAbsolute(relative) && (await stat(resolved)).size > 0, "evidence file must be nonempty and inside owned state");
  return resolved;
}

// The env var scripts/ci/windows_cuda_execution.py sets for the child it launches under its lease.
export const CUDA_LEASE_KEY = "SW_CI_CUDA_EXECUTION_ACTIVE";

export function parseCudaCards(stdout) {
  const cards = stdout.trim().split(/\r?\n/).map((line) => { const [index, uuid, name, total] = line.split(",").map((value) => value.trim()); return { index, uuid, name, total_mib: Number(total) }; });
  fact(cards.length > 0 && cards.every((card) => /^[0-9]+$/.test(card.index) && /^GPU-[A-Za-z0-9-]+$/.test(card.uuid) && Number.isFinite(card.total_mib)), "nvidia-smi reported no usable CUDA card");
  return cards;
}
// The card the lease is taken on: the largest total memory, lowest PCI index on a tie. Chosen from
// fixed inventory (never current free memory), so the strict-idle lease waits for that card.
export function chooseCudaCard(cards) {
  return cards.reduce((best, card) => (card.total_mib > best.total_mib ? card : best));
}
async function cudaCards() {
  const { stdout } = await execFile("nvidia-smi", ["--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"], { timeout: 30_000, windowsHide: true });
  return parseCudaCards(stdout);
}

function childEnvironment(env, { backend, weights, card }) {
  const child = {};
  for (const [key, value] of Object.entries(env)) {
    if (key.startsWith("SCENEWORKS_") && key !== "SCENEWORKS_METAL_EXECUTION_LOCK_FD") continue;
    if (/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(key) || ["HF_ENDPOINT", "HF_XET_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "CARGO_TARGET_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"].includes(key)) continue;
    child[key] = value;
  }
  const hub = path.join(weights, "hub");
  Object.assign(child, { HF_HOME: weights, HF_HUB_CACHE: hub, HUGGINGFACE_HUB_CACHE: hub, HF_XET_CACHE: path.join(weights, "xet"), SCENEWORKS_WORKER_CHILD: "1", SCENEWORKS_POLL_SECONDS: "1", SCENEWORKS_HEARTBEAT_SECONDS: "2" });
  if (backend === "mlx") Object.assign(child, { SCENEWORKS_GPU_ID: "mlx", SCENEWORKS_BACKEND_MLX_ENABLED: "true", SCENEWORKS_BACKEND_CANDLE_ENABLED: "false", SCENEWORKS_MLX_REQUIRED: "1", SCENEWORKS_MLX_UNSUPPORTED_MODE: "enforce" });
  else Object.assign(child, { SCENEWORKS_GPU_ID: card.index, CUDA_DEVICE_ORDER: "PCI_BUS_ID", CUDA_VISIBLE_DEVICES: card.index, NVIDIA_VISIBLE_DEVICES: card.index, SCENEWORKS_BACKEND_CANDLE_ENABLED: "true", SCENEWORKS_BACKEND_MLX_ENABLED: "false", SCENEWORKS_CANDLE_REQUIRED: "1", SCENEWORKS_RESOLVED_CACHE_ENABLED: "false" });
  return child;
}

// phase "build" compiles the API/worker binary into the attempt's output (no GPU, no lease);
// phase "run" (on Windows, only as the child of the CUDA lease) runs the whole app lifecycle.
export async function runLifecycle(env = process.env, { phase = "run" } = {}) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { backend, tier } = assertSelection(env);
  fact(env.H3I_APP_OUTPUT && env.H3I_APP_WEIGHTS && env.RUNNER_TEMP, "H3I_APP_OUTPUT, H3I_APP_WEIGHTS and RUNNER_TEMP required");
  const output = path.resolve(env.H3I_APP_OUTPUT), relative = path.relative(path.resolve(env.RUNNER_TEMP), output);
  fact(relative && !relative.startsWith("..") && !path.isAbsolute(relative), "output must stay under RUNNER_TEMP");
  const evidence = path.join(output, "evidence"), logs = path.join(output, "logs");
  await mkdir(evidence, { recursive: true }); await mkdir(logs, { recursive: true });
  const receipt = { schema_version: 1, kind: "minimax_h3_image_train_app_lifecycle", status: "running", started_at: new Date().toISOString(), identity: { backend, tier, source_sha: env.H3I_APP_SOURCE_SHA, inference_sha: env.H3I_APP_INFERENCE_SHA, runner_name: env.RUNNER_NAME, run_id: env.GITHUB_RUN_ID, run_attempt: env.GITHUB_RUN_ATTEMPT }, settings: { training: TRAINING, render: RENDER, captions: CAPTIONS, dataset: "one public synthetic 1024x1024 image per mode (edit: one reference + target)" }, events: [], cases: [] };
  const secrets = new Set(Object.entries(env).filter(([key, value]) => /TOKEN|SECRET|PASSWORD|API_KEY/i.test(key) && value?.length >= 8).map(([, value]) => value));
  const receiptName = phase === "build" ? "build-receipt.json" : "receipt.json";
  receipt.phase = phase;
  const save = async () => { await writeFile(path.join(evidence, "receipt.tmp"), redactSensitiveText(JSON.stringify(receipt, null, 2) + "\n", secrets)); await rename(path.join(evidence, "receipt.tmp"), path.join(evidence, receiptName)); };
  await save();
  const children = [], handles = [], controller = new AbortController();
  let endpoint;
  const abort = () => controller.abort(new Error("lifecycle interrupted"));
  process.once("SIGTERM", abort); process.once("SIGINT", abort);
  const deadline = setTimeout(() => controller.abort(new Error("lifecycle exceeded eleven hours")), 11 * 60 * 60_000);
  const alive = () => { if (controller.signal.aborted) throw controller.signal.reason; for (const child of children.filter((c) => c.service)) fact(child.exitCode === null && child.signalCode === null, `owned ${child.logName} exited; see its log`); };
  const delay = async (ms = 2000) => { alive(); await new Promise((resolve) => setTimeout(resolve, ms)); alive(); };
  const launch = async (command, args, childEnv, name, service = false) => {
    const log = await open(path.join(logs, `${name}.log`), "wx"); handles.push(log);
    const stdin = process.platform === "win32" && service ? "pipe" : "ignore";
    const options = { cwd: root, env: childEnv, detached: process.platform !== "win32", windowsHide: true, stdio: [stdin, log.fd, log.fd] };
    const child = spawn(command, args, process.platform === "darwin" ? metalChildOptions(options) : options);
    Object.assign(child, { service, logName: name });
    child.done = new Promise((resolve, reject) => { child.once("error", reject); child.once("exit", (code, signal) => resolve({ code, signal })); });
    child.done.catch(() => controller.abort(new Error(`${name} failed to launch`)));
    if (service) child.once("exit", () => controller.abort(new Error(`owned ${name} exited; see its log`)));
    children.push(child); return child;
  };
  const checked = async (command, args, childEnv, name) => {
    const child = await launch(command, args, childEnv, name);
    while (child.exitCode === null && child.signalCode === null) await delay(1000);
    fact((await child.done).code === 0, `${name} failed; see its log`);
  };
  const wait = async (fn, seconds, label) => { const end = Date.now() + seconds * 1000; while (Date.now() < end) { alive(); const value = await fn(); if (value) return value; await delay(); } throw new Error(`${label} timed out after ${seconds}s`); };
  try {
    const git = async (...args) => (await execFile("git", args, { cwd: root, timeout: 30_000 })).stdout.trim();
    fact(await git("rev-parse", "HEAD") === env.H3I_APP_SOURCE_SHA && await git("status", "--porcelain") === "", "exact clean source checkout required");
    const manifests = await Promise.all(["Cargo.toml", "crates/sceneworks-worker/Cargo.toml", "crates/sceneworks-memory-adapter/Cargo.toml"].map((file) => readFile(path.join(root, file), "utf8")));
    assertPins(manifests, await readFile(path.join(root, "Cargo.lock"), "utf8"), env.H3I_APP_INFERENCE_SHA);
    let card;
    if (backend === "mlx") {
      const hardware = JSON.parse((await execFile("system_profiler", ["SPHardwareDataType", "-json"], { timeout: 30_000 })).stdout).SPHardwareDataType;
      receipt.identity.hardware = hardware.map(({ chip_type, physical_memory, machine_model }) => ({ chip_type, physical_memory, machine_model }));
    } else {
      // The workflow chose the card before the build and leased exactly that UUID for this run.
      const cards = await cudaCards();
      card = cards.find((item) => item.index === env.H3I_APP_CUDA_INDEX);
      fact(card, `H3I_APP_CUDA_INDEX ${env.H3I_APP_CUDA_INDEX} names no CUDA card`);
      if (phase === "run") fact(env[CUDA_LEASE_KEY], "the CUDA lifecycle must run as the child of scripts/ci/windows_cuda_execution.py");
      receipt.identity.cuda = { selected: card, cards, lease_holder_pid: env[CUDA_LEASE_KEY] ?? null };
    }
    // Disk: the persistent task-owned weights root plus this attempt's build and state.
    const weights = path.resolve(env.H3I_APP_WEIGHTS), hub = path.join(weights, "hub");
    await mkdir(hub, { recursive: true }); await mkdir(path.join(weights, "xet"), { recursive: true });
    const catalog = (await import("./lib/jsonc.mjs")).stripJsoncComments(await readFile(path.join(root, "config/manifests/builtin.models.jsonc"), "utf8"));
    const entry = JSON.parse(catalog).models.find((model) => model.id === MODEL_ID); fact(entry, "shipped catalog entry missing");
    const rows = tierRows(entry, tier, process.platform);
    const fs = await statfs(weights, { bigint: true });
    receipt.disk = assertDiskBudget({ freeBytes: Number(fs.bavail * fs.bsize), missingBytes: await missingTierBytes(hub, rows), includeBuild: phase === "build" });
    receipt.identity.weights_root = weights; await save();
    // The ordinary API/worker binary at this commit, built by the build phase of this job.
    const target = path.join(output, "target");
    const binary = path.join(target, "release", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api");
    if (phase === "build") {
      const features = backend === "cuda" ? ["--features", "backend-candle"] : [];
      await checked("cargo", ["build", "--release", "--locked", "-p", "sceneworks-rust-api", ...features, "--message-format=json-render-diagnostics"], { ...env, CARGO_TARGET_DIR: target }, "build");
      fact(await git("status", "--porcelain") === "", "build changed the checkout");
      receipt.identity.binary_sha256 = await hash(binary); receipt.status = "built";
      return receipt;
    }
    receipt.identity.binary_sha256 = await hash(binary);
    const identity = attemptIdentity(output), { state, workerId } = identity;
    await mkdir(path.join(state, "data", "projects"), { recursive: true }); await mkdir(path.join(state, "credentials")); await cp(path.join(root, "config"), path.join(state, "config"), { recursive: true });
    let childEnv = bindOwnedEnv(childEnvironment(env, { backend, weights, card }), identity);
    if (backend === "mlx") {
      const native = await bindMetallib({ messages: path.join(logs, "build.log"), lockFile: path.join(root, "Cargo.lock"), targetRoot: target, temporaryRoot: env.RUNNER_TEMP, consumer: "app", explicitOverride: env.PMETAL_METALLIB_PATH ?? "", explicitNativeDirectory: env.PMETAL_MLX_PREBUILT_DIR ?? "" });
      await copyFile(native.receipt, path.join(evidence, "native-metallib-binding.json"));
      childEnv.PMETAL_METALLIB_PATH = native.path;
      await checked(binary, [], { ...childEnv, SCENEWORKS_GPU_CHECK: "1" }, "metal-preflight");
    }
    const api = await launch(binary, [], childEnv, "api", true);
    const url = await wait(async () => boundApiUrl(await readFile(path.join(logs, "api.log"), "utf8")), 180, "API bound-port announcement");
    childEnv = bindOwnedEnv(childEnv, identity, url);
    endpoint = ownedApiController({ url, state, api, controller });
    receipt.health = await wait(async () => { try { return await endpoint.verify(); } catch { alive(); return false; } }, 180, "API readiness");
    await endpoint.startWorker(() => launch(binary, [], { ...childEnv, SCENEWORKS_WORKER_ONLY: "1" }, "worker", true));
    const must = (...args) => endpoint.call(...args);
    // A refused POST names the API's own reason (an install, licence or memory refusal).
    const post = async (route, body) => { const response = await endpoint.exchange("POST", route, body); fact(response.status < 300, `POST ${route} refused (${response.status}): ${JSON.stringify(response.body)}`); return response.body; };
    receipt.worker = await wait(async () => {
      const worker = (await must("GET", "/api/v1/workers")).find((item) => item.id === workerId);
      if (!worker || worker.status !== "idle") return false;
      fact(worker.capabilities?.includes("gpu") && worker.capabilities.includes("lora_train_execute"), "owned GPU worker must advertise training");
      fact(backend === "mlx" ? worker.gpuId === "mlx" && !worker.capabilities.includes("candle") : worker.gpuId === card.index && worker.capabilities.includes("candle"), `owned worker must be the ${backend} device`);
      return worker;
    }, 300, "GPU worker registration");
    receipt.download_worker = await wait(async () => { try { return assertDownloadWorker(await must("GET", "/api/v1/workers"), workerId); } catch (error) { alive(); if (/owned API/.test(error.message)) throw error; return false; } }, 180, "utility worker able to claim model_download");
    const relocation = await post("/api/v1/model-library/relocate", { path: weights });
    fact(relocation.adopted === true && path.resolve(relocation.hfHome) === weights && path.resolve(relocation.libraryRoot) === hub, "API must adopt the task-owned model library");
    const getJob = (id) => must("GET", `/api/v1/jobs/${id}`);
    const terminal = async (id, record) => {
      const job = await getJob(id), last = receipt.events.at(-1);
      if (last?.id !== id || last.status !== job.status || last.message !== job.message) { receipt.events.push({ id, type: job.type, status: job.status, stage: job.stage, message: job.message, worker_id: job.workerId, at: new Date().toISOString() }); if (record) await save(); }
      fact(!["failed", "canceled"].includes(job.status), `job ${id} ${job.status}: ${job.error ?? job.message}`);
      return job.status === "completed" && job;
    };
    // 1. Install the tier through the Model Manager route (the licence gate refuses first).
    const denied = await endpoint.exchange("POST", `/api/v1/models/${MODEL_ID}/download`, { variant: tier, licenseAcknowledged: false });
    fact(denied.status >= 400 && denied.status < 500 && /licen[cs]e/i.test(JSON.stringify(denied.body)), "an unacknowledged install must be refused for the licence");
    const download = await post(`/api/v1/models/${MODEL_ID}/download`, { variant: tier, licenseAcknowledged: true });
    receipt.install = { license_refusal: denied, job_id: download.id, tier }; await save();
    receipt.install.job = await wait(() => terminal(download.id, true), 4 * 3600, `${tier} install`);
    receipt.install.model = (await must("GET", "/api/v1/models")).find((model) => model.id === MODEL_ID);
    receipt.install.license_acknowledgment = await must("GET", `/api/v1/models/${MODEL_ID}/license-acknowledgment`);
    fact(receipt.install.license_acknowledgment.acknowledged === true, "the acknowledged install must record the licence acceptance renders require"); await save();
    // 2. Project, synthetic assets and one dataset per mode.
    const project = await post("/api/v1/projects", { name: `SW-7103 H3-Image ${backend} training lifecycle` }); receipt.project_id = project.id;
    const upload = async (name, data) => { await writeFile(path.join(evidence, name), data); const form = new FormData(); form.append("file", new Blob([data], { type: "image/png" }), name); const response = await endpoint.exchange("POST", `/api/v1/projects/${project.id}/assets`, form, { multipart: true }); fact(response.status < 300, `asset upload refused (${response.status})`); return response.body.id; };
    const source = await upload("source.png", syntheticPng()), editTarget = await upload("target.png", syntheticPng(true));
    const targets = await must("GET", "/api/v1/training/targets"), allTargets = targets.targets ?? targets;
    for (const mode of ["t2i", "edit"]) {
      const dataset = await post(`/api/v1/projects/${project.id}/training/datasets`, { name: `sw7103-synthetic-${mode}`, items: [{ assetId: mode === "edit" ? editTarget : source, caption: { text: CAPTIONS[mode] }, ...(mode === "edit" ? { references: [{ assetId: source }] } : {}) }] });
      const body = trainingBody(allTargets.find((item) => item.id === TARGETS[mode]), dataset.id, mode, tier), route = `/api/v1/projects/${project.id}/training/jobs`;
      const refused = await endpoint.exchange("POST", route, { ...body, licenseAcknowledged: false });
      fact(refused.status === 403 && /licen[cs]e/i.test(JSON.stringify(refused.body)), "an unacknowledged training run must be refused for the licence");
      const accepted = await post(route, body); assertPlan(accepted, body, mode);
      const record = { mode, dataset_id: dataset.id, accepted_job: accepted, license_refusal: refused }; receipt.cases.push(record); await save();
      // 3. Train to completion (the worker's training-memory admission runs before any weight read).
      const completed = await wait(() => terminal(accepted.id, true), 5 * 3600, `${mode} training`);
      const library = await wait(async () => (await must("GET", `/api/v1/loras?projectId=${project.id}`)).find((item) => item.id === completed.result?.loraId), 120, "LoRA library registration");
      const adapter = await ownedFile(completed.result.outputPath, state), metadata = await safetensorsMetadata(adapter);
      Object.assign(record, { completed_job: completed, library_entry: library, adapter: { file: path.basename(adapter), bytes: (await stat(adapter)).size, sha256: await hash(adapter), metadata } });
      // The worker labels results with its device id (`backend_label(gpu_id)`): "mlx", or the
      // CUDA card index on Candle.
      record.training = assertTrained(completed, library, metadata, mode, receipt.worker.gpuId, workerId); await save();
      // 4. Render through the Image Studio route with the LoRA, and the same request without it.
      record.renders = [];
      const pngs = [];
      for (const loraId of [library.id, null]) {
        const image = imageBody(project.id, mode, tier, loraId, source), job = await post("/api/v1/image/jobs", image);
        const done = await wait(() => terminal(job.id, true), 3 * 3600, `${mode} render${loraId ? " with LoRA" : ""}`);
        const asset = (await must("GET", `/api/v1/projects/${project.id}/assets`)).find((item) => item.id === done.result?.assetIds?.[0]);
        const rendered = assertRendered(done, asset, image, backend, workerId);
        const media = await ownedFile(path.join(project.path, asset.file.path), state), saved = await readFile(media);
        assertServedPng((await endpoint.served(`/api/v1/projects/${project.id}/files/${asset.file.path}`)).bytes, saved, RENDER.width, RENDER.height);
        const name = `${mode}-${loraId ? "with-lora" : "without-lora"}.png`; await copyFile(media, path.join(evidence, name));
        record.renders.push({ ...rendered, file: name, sha256: sha256(saved) }); pngs.push(saved); await save();
      }
      record.lora_effect = { mean_absolute_rgb_difference: meanAbsoluteDifference(pngs[0], pngs[1]) };
      fact(record.lora_effect.mean_absolute_rgb_difference > 0, "the LoRA must change the render at the same seed");
      await save();
    }
    fact(await git("status", "--porcelain") === "", "runtime changed the checkout");
    receipt.status = "passed";
  } catch (error) { receipt.status = "failed"; receipt.error = error.message; throw error; }
  finally {
    clearTimeout(deadline); process.removeListener("SIGTERM", abort); process.removeListener("SIGINT", abort); receipt.cleanup = [];
    for (const child of children.reverse()) {
      if (!child.pid || child.exitCode !== null || child.signalCode !== null) continue;
      try {
        if (process.platform === "win32") child.kill(); else process.kill(-child.pid, "SIGTERM");
        await Promise.race([child.done.catch(() => {}), new Promise((resolve) => setTimeout(resolve, 15_000))]);
        if (child.exitCode === null && child.signalCode === null) {
          if (process.platform === "win32") await execFile("taskkill", ["/F", "/PID", String(child.pid)]).catch(() => {}); else process.kill(-child.pid, "SIGKILL");
          await child.done.catch(() => {});
        }
        receipt.cleanup.push({ name: child.logName, pid: child.pid, exited: true });
      } catch (error) { receipt.status = "failed"; receipt.cleanup.push({ name: child.logName, pid: child.pid, exited: false, error: error.message }); process.exitCode = 1; }
    }
    await Promise.all(handles.map((handle) => handle.close()));
    endpoint?.close();
    receipt.finished_at = new Date().toISOString(); await save();
    // Keep the owned service logs with the evidence (the receipt redacts tokens; logs never see one).
    for (const name of await readdir(logs).catch(() => [])) if (name !== "build.log") await copyFile(path.join(logs, name), path.join(evidence, name)).catch(() => {});
  }
  fact(receipt.status === (phase === "build" ? "built" : "passed"), "owned process cleanup failed");
  return receipt;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  fact(args.length === 0 || (args.length === 1 && ["--preflight-only", "--select-cuda-card", "--build-only"].includes(args[0])), "only --preflight-only, --select-cuda-card or --build-only is supported");
  const run = async () => {
    if (args[0] === "--preflight-only") { assertSelection(process.env); console.log("preflight passed"); return; }
    if (args[0] === "--select-cuda-card") {
      fact(process.env.GITHUB_ENV, "GITHUB_ENV required");
      const card = chooseCudaCard(await cudaCards());
      await appendFile(process.env.GITHUB_ENV, `H3I_APP_CUDA_INDEX=${card.index}\n`);
      console.log(`selected CUDA card ${card.index} ${card.uuid} ${card.name} ${card.total_mib} MiB`); return;
    }
    if (args[0] === "--build-only") { const receipt = await runLifecycle(process.env, { phase: "build" }); console.log(`${receipt.status}: ${receipt.identity.binary_sha256}`); return; }
    if (process.platform === "darwin" && await enterMacMetalExecution(11 * 60 * 60)) return;
    const receipt = await runLifecycle(process.env);
    console.log(`${receipt.status}: ${receipt.cases.length} trained LoRAs, ${receipt.cases.reduce((sum, item) => sum + (item.renders?.length ?? 0), 0)} renders`);
  };
  run().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
