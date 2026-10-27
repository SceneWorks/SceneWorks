#!/usr/bin/env node
// Issue 3197. Exactly two cold SDXL q4 renders, using the maintained catalog artifact resolver
// and calibration capture protocol. Dispatch-only; never imported by a normal-CI inference test.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFile, writeFile, mkdir, realpath, open } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { performance } from "node:perf_hooks";
import { ROOT, parseArgs, planRun, providerCommand, run } from "./measure-memory-catalog.mjs";
import { capturePlannedCase, logicalCaseId, planAnchor } from "./memory-calibration-harness.mjs";
import { hashArtifactInventory, artifactInventoryReceipt } from "./hash-artifact-inventory.mjs";

export const ENDPOINTS = Object.freeze([
  { name: "resident", strategy: { rung: "resident", engagedRungs: ["resident"], parameters: {} } },
  { name: "sequential", strategy: {
    rung: "bounded_attention",
    engagedRungs: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
    parameters: { stageResidency: true, decodeTileEdge: 256, decodeOverlap: 64, attentionChunkSize: 16_777_216 },
  } },
]);

const GPU_UUID = /^GPU-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const LEGACY_GPU = Object.freeze({ physicalIndex: 1, uuid: "GPU-e4b79931-7be6-f216-460a-f5405cfafffe" });
const GPU_QUERY = "index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory";

function validateGpu(gpu) {
  if (![0, 1].includes(gpu?.physicalIndex) || !GPU_UUID.test(gpu?.uuid ?? "")) {
    throw new Error("SDXL capture requires one physical GPU0/1 index and exact NVIDIA UUID");
  }
  return gpu;
}

export function endpointGpuEnv(env, gpu) {
  validateGpu(gpu);
  return { ...env, CUDA_VISIBLE_DEVICES: String(gpu.physicalIndex), SCENEWORKS_GPU_ID: String(gpu.physicalIndex),
    SCENEWORKS_SDXL_EXPECTED_GPU_UUID: gpu.uuid };
}

export function endpointCase(plan, endpoint) {
  const planned = planAnchor(plan, "sdxl:q4:candle");
  planned.loadShape = "eager_materialization";
  planned.target.geometry = { width: 1024, height: 1024, batch: 1, frames: 1 };
  planned.strategy = structuredClone(endpoint.strategy);
  planned.calibrationFingerprint = "sdxl-candle-sdxl-staged-decode-attention-v3";
  // Existing SDXL fixture validation accepts the five-rung prefix; the settings are separately
  // recorded from the actual GenerationRequest, with provider-default sampler and step count.
  planned.fixture = "fresh-five-rung-sdxl-q4-1024-seed16402-default-steps";
  planned.logicalCaseId = logicalCaseId(planned);
  return planned;
}

export function validateEndpoint(measurement, endpoint, expectedInferencePin, expectedUuid = LEGACY_GPU.uuid) {
  if (!GPU_UUID.test(expectedUuid) || measurement?.endpoint !== endpoint.name || measurement.deviceUuid !== expectedUuid
      || measurement.cudaContextUuid !== measurement.deviceUuid || measurement.logicalCudaDevice !== 0) {
    throw new Error("endpoint identity or selected physical GPU UUID differs from the requested capture");
  }
  if (!/^[a-f0-9]{40}$/.test(measurement.inferencePin ?? "")
      || (expectedInferencePin && measurement.inferencePin !== expectedInferencePin)) {
    throw new Error("endpoint adapter was compiled against a different inference pin");
  }
  for (const key of ["loadPeakBytes", "wholeRunPeakBytes", "baselineBytes", "wholeDevicePeakBytes"]) {
    if (!Number.isSafeInteger(measurement[key]) || measurement[key] < 0) throw new Error(`missing ${key}`);
  }
  if (measurement.wholeRunPeakBytes <= 0 || measurement.wholeRunPeakBytes < measurement.loadPeakBytes) {
    throw new Error("whole-run peak must include the load peak");
  }
  const output = measurement.output;
  if (output?.width !== 1024 || output.height !== 1024 || output.channels !== 3 || output.rgbBytes !== 1024 * 1024 * 3
      || output.spatiallyNonFlat !== true || !/^[a-f0-9]{64}$/.test(output.rgbSha256 ?? "")) {
    throw new Error("endpoint did not return a non-flat 1024-square RGB image receipt");
  }
  const settings = measurement.settings;
  if (settings?.flashAttn !== true || settings.count !== 1 || settings.cfgInterval !== null
      || settings.guidanceMethod !== null || settings.schedulerShift !== null) {
    throw new Error("endpoint kernel/request settings differ from the ordinary SDXL worker defaults");
  }
  if (measurement.engineBuild?.adapterCandleFeature !== true || measurement.engineBuild.sdxlFlashAttentionKernelCompiled !== false) {
    throw new Error("endpoint did not attest the pinned CUDA build and SDXL attention kernel policy");
  }
  const cleanup = measurement.cleanup;
  if (!cleanup || cleanup.adapterProcessExited !== true
      || [cleanup.beforeBytes, cleanup.afterBytes, cleanup.toleranceBytes].some((value) => !Number.isSafeInteger(value) || value < 0)
      || cleanup.afterBytes > cleanup.beforeBytes + cleanup.toleranceBytes) throw new Error("endpoint leaked device residency or lacks exit telemetry");
}

/** Full base plus the exact shared files the SDXL pipeline reads, under portable role paths. */
export async function sdxlAssemblyInventory(row) {
  const sources = [
    { role: "base", root: row.tierRoot, tier: true, includeFiles: row.inventoryFiles },
    ...[
      ["tokenizer_clip_l", "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_L", "tokenizer.json"],
      ["tokenizer_clip_bigg", "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_BIGG", "tokenizer.json"],
      ["vae_fp16_fix", "SCENEWORKS_SDXL_COMPONENT_VAE_FP16_FIX", "diffusion_pytorch_model.safetensors"],
    ].map(([role, key, file]) => ({ role: `components/${role}`, root: row.env?.[key], includeFiles: [file] })),
  ];
  const entries = [];
  for (const source of sources) {
    if (!source.root) throw new Error(`SDXL assembly source missing: ${source.role}`);
    // The resolver supplies hub snapshot roots; files may point into their own repo's blobs/.
    const trustedRoot = source.tier
      ? path.dirname(path.dirname(path.dirname(source.root))) : path.dirname(path.dirname(source.root));
    const inventory = await hashArtifactInventory(source.root, {
      includeEntries: true, trustedRoot, ...(source.includeFiles ? { includeFiles: source.includeFiles } : {}),
    });
    entries.push(...inventory.entries.map((entry) => ({ ...entry, path: `${source.role}/${entry.path}` })));
  }
  entries.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  return { root: path.resolve(row.tierRoot), ...artifactInventoryReceipt(entries) };
}

export function endpointEvidenceReference(bundle, bundleBytes, { inventorySha256, inferencePin, deviceUuid }) {
  const [record] = bundle.records ?? [];
  if (bundle.records?.length !== 1 || !/^imc-[a-f0-9]{20}$/.test(record?.id ?? "")) {
    throw new Error("endpoint evidence must contain exactly one identified source record");
  }
  if (record.artifact?.inventorySha256 !== inventorySha256
      || record.repositories?.inference?.revision !== inferencePin
      || record.hardware?.deviceId !== deviceUuid) {
    throw new Error("endpoint source record differs from the measured inventory, compiled pin or physical device");
  }
  return { recordId: record.id, evidenceSha256: createHash("sha256").update(bundleBytes).digest("hex") };
}

export function compareRgb(reference, selected) {
  if (!reference.length || reference.length !== selected.length) throw new Error("RGB comparison requires equal nonempty images");
  let maximum = 0; let total = 0; let changed = 0;
  for (let index = 0; index < reference.length; index++) {
    const delta = Math.abs(reference[index] - selected[index]);
    maximum = Math.max(maximum, delta); total += delta; changed += Number(delta !== 0);
  }
  return { result: "measured", reference: "resident", selected: "sequential", channels: reference.length,
    maximumAbsoluteRgbDelta255: maximum, meanAbsoluteRgbDelta255: total / reference.length,
    maximumAbsoluteError: maximum / 255, meanAbsoluteError: total / reference.length / 255,
    changedChannels: changed, identical: changed === 0 };
}

export function admittedReplayRequest(projectId) {
  // Only the installed precision is explicit. API/worker resolve their ordinary sampler/settings;
  // the request contains no strategy, memory recipe, model path or manifest override.
  return { projectId, model: "sdxl", mode: "text_to_image", prompt: "a photorealistic red apple on a wooden table, studio lighting",
    width: 1024, height: 1024, count: 1, seed: 16402, loras: [], advanced: { mlxQuantize: 4 } };
}

export function validatePackagedEndpointSummarySource(source) {
  if (source?.path !== "config/sdxl-cuda-memory-endpoint-evidence/sdxl-cuda-endpoints.json"
      || !/^[a-f0-9]{64}$/.test(source?.sha256 ?? "")) {
    throw new Error("replay requires the dedicated packaged measured endpoint summary");
  }
  return source;
}

// Replay may advance only commit metadata: complete Git trees must remain byte-identical.
// The capture keeps its historical pin; this comparison is neither a recapture nor certification.
export async function replayInferenceSourceIdentity(capturedPin, activePin, inferenceRepo, execute = run) {
  if (![capturedPin, activePin].every((pin) => typeof pin === "string" && /^[a-f0-9]{40}$/.test(pin))) {
    throw new Error("replay requires canonical captured and active inference pins");
  }
  if (capturedPin === activePin) {
    return { capturedPin, activePin, match: "same_pin", capturedTree: null, activeTree: null, completeGitTreesEqual: null };
  }
  const readTree = async (pin) => {
    const { stdout } = await execute("git", ["rev-parse", "--verify", `${pin}^{tree}`], {
      cwd: inferenceRepo, env: { ...process.env, GIT_NO_LAZY_FETCH: "1" }, timeoutMs: 10_000,
    });
    const tree = stdout.trim();
    if (!/^[a-f0-9]{40}$/.test(tree)) throw new Error("inference Git tree identity is unavailable or malformed");
    return tree;
  };
  const [capturedTree, activeTree] = await Promise.all([readTree(capturedPin), readTree(activePin)]);
  if (capturedTree !== activeTree) throw new Error("active inference complete Git tree differs from the packaged capture");
  return { capturedPin, activePin, match: "identical_complete_git_tree", capturedTree, activeTree, completeGitTreesEqual: true };
}

export function assertAdmittedReplaySelection(selection) {
  const parameters = selection?.parameters;
  const noDecode = parameters?.decodeTileEdge == null && parameters?.decodeOverlap == null;
  const boundedDecode = [256, 512].includes(parameters?.decodeTileEdge) && parameters?.decodeOverlap === 64;
  const noAttention = parameters?.attentionChunkSize == null;
  const recipe = typeof parameters?.stageResidency === "boolean" && parameters.transformerWindowSize == null
    && Object.keys(parameters).every((key) => ["stageResidency", "decodeTileEdge", "decodeOverlap", "attentionChunkSize", "transformerWindowSize"].includes(key))
    && (selection.strategy === "resident" && !parameters.stageResidency && noDecode && noAttention
      || selection.strategy === "staged_residency" && parameters.stageResidency && noDecode && noAttention
      || selection.strategy === "bounded_decode" && boundedDecode && noAttention
      || selection.strategy === "bounded_attention" && boundedDecode && [16_777_216, 67_108_864].includes(parameters.attentionChunkSize));
  const minimum = ENDPOINTS[1].strategy.parameters;
  const measuredMinimum = selection?.strategy === "bounded_attention" && parameters?.stageResidency === true
    && parameters.decodeTileEdge === minimum.decodeTileEdge && parameters.decodeOverlap === minimum.decodeOverlap
    && parameters.attentionChunkSize === minimum.attentionChunkSize;
  const coverage = selection?.strategy === "resident" ? "measured_resident"
    : measuredMinimum ? "measured_minimum" : "estimated_intermediate";
  // availableGb is the selector's pool AFTER subtracting its reserve. Compare the
  // widened admission to that pool, and account for the reserve once against the
  // six-GiB guard. The selector canonicalizes decimal GiB ceilings to whole bytes.
  const roundingGiB = 1 / 1024 ** 3;
  if (selection?.backend !== "candle" || selection.route !== "sdxl" || selection.actualTier !== "q4"
      || selection.mode !== "text_to_image" || !recipe
      || selection.basis !== (coverage === "estimated_intermediate" ? "anchor_derived" : "measured_endpoint") || selection.geometry?.width !== 1024
      || selection.geometry?.height !== 1024 || selection.geometry?.batch !== 1
      || !Number.isSafeInteger(selection.predictedPeakBytes) || selection.predictedPeakBytes <= 0
      || !Number.isFinite(selection.reserveGb) || selection.reserveGb <= 0
      || !Number.isFinite(selection.availableGb) || selection.availableGb <= 0
      || !Number.isFinite(selection.admittedPeakGb) || selection.admittedPeakGb <= 0
      || selection.predictedPeakBytes / 1024 ** 3 > selection.admittedPeakGb + roundingGiB
      || selection.admittedPeakGb > selection.availableGb + roundingGiB
      || selection.availableGb + selection.reserveGb > 6 + roundingGiB) {
    throw new Error("ordinary worker did not admit a supported endpoint-priced SDXL q4 recipe within the guarded six-GiB budget");
  }
  return coverage;
}

export function admittedReplaySelection(workerLog) {
  // emit_event's complete payload is flattened by SceneworksJsonFormat. The nearby
  // tracing::info! has the same event name, but only snake_case summary fields.
  const selections = workerLog.split(/\r?\n/).flatMap((line) => {
    try {
      const event = JSON.parse(line);
      return event?.event === "image_memory_strategy_selected" && event.route === "sdxl"
        && typeof event.actualTier === "string" && event.geometry && event.parameters ? [event] : [];
    } catch { return []; }
  });
  if (selections.length !== 1) throw new Error(`expected one complete emitted worker memory selection, saw ${selections.length}`);
  assertAdmittedReplaySelection(selections[0]);
  return selections[0];
}

export function persistedReplayAsset(job, projectId, jobId) {
  const result = job?.result;
  // Terminal status can precede the API's durable side effects. Never accept the
  // worker's assetWrites facts as proof that the API persisted the asset.
  if (!Array.isArray(result?.assets) || !Array.isArray(result.assetIds) || result.assetWrites !== undefined) return null;
  const image = result.assets[0];
  const mediaPath = image?.file?.path;
  if (result.assets.length !== 1 || result.assetIds.length !== 1 || typeof image?.id !== "string"
      || !image.id || result.assetIds[0] !== image.id || image.projectId !== projectId
      || image.lineage?.jobId !== jobId || image.type !== "image" || image.file?.mimeType !== "image/png"
      || image.file.width !== 1024 || image.file.height !== 1024 || typeof mediaPath !== "string"
      || !mediaPath.endsWith(".png") || mediaPath.split("/").some((part) => !part || part === "." || part === "..")
      || /[\\:\u0000-\u001f\u007f]/.test(mediaPath)) {
    throw new Error("ordinary job did not persist exactly one matching project PNG asset sidecar");
  }
  return image;
}

function parseGpuSample(stdout, at) {
  const fields = stdout.trim().split(",").map((value) => value.trim());
  const numbers = [fields[2], fields[3], fields[4], fields[5]].map(Number);
  if (fields.length !== 6 || !["0", "1"].includes(fields[0]) || !GPU_UUID.test(fields[1] ?? "")
      || fields.slice(2).some((value) => !/^\d+$/.test(value))
      || numbers.some((value) => !Number.isSafeInteger(value) || value < 0)
      || numbers[2] > 100 || numbers[3] > 100
      || !numbers.slice(0, 2).every((value) => Number.isSafeInteger(value * 1024 ** 2))) {
    throw new Error("physical GPU telemetry identity or counters unavailable");
  }
  return { at, physicalIndex: Number(fields[0]), uuid: fields[1], usedBytes: numbers[0] * 1024 ** 2,
    freeBytes: numbers[1] * 1024 ** 2, gpuUtilization: numbers[2], memoryUtilization: numbers[3] };
}

export function parseReplayGpuSample(stdout, at = new Date().toISOString(), gpu = LEGACY_GPU) {
  validateGpu(gpu);
  const sample = parseGpuSample(stdout, at);
  if (sample.physicalIndex !== gpu.physicalIndex || sample.uuid !== gpu.uuid) {
    throw new Error("selected physical GPU telemetry identity or counters unavailable");
  }
  return sample;
}

export function parseGpuInventory(stdout, at = new Date().toISOString()) {
  const samples = stdout.trim().split(/\r?\n/).map((line) => parseGpuSample(line, at));
  if (samples.length !== 2 || new Set(samples.map((sample) => sample.physicalIndex)).size !== 2
      || new Set(samples.map((sample) => sample.uuid)).size !== 2) {
    throw new Error("GPU0/1 inventory must contain both distinct physical indices and UUIDs");
  }
  return samples;
}

function idleWindow(window, sample) {
  if (sample.gpuUtilization || sample.memoryUtilization || sample.freeBytes < 16 * 1024 ** 3) return [];
  const next = [...window, sample].slice(-5);
  return Math.max(...next.map((item) => item.usedBytes)) - Math.min(...next.map((item) => item.usedBytes)) > 64 * 1024 ** 2 ? [sample] : next;
}

// One physical index/UUID is selected after the build and frozen for the complete endpoint pair.
export async function selectCaptureGpu(probe, { timeoutMs = 30_000, pollMs = 200, now = () => performance.now(),
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)) } = {}) {
  const started = now(), samples = [], windows = [[], []], identities = new Map();
  try {
    while (now() - started < timeoutMs) {
      const raw = await probe(timeoutMs - (now() - started));
      let inventory;
      try { inventory = parseGpuInventory(raw); }
      catch (error) { samples.push({ elapsedMs: now() - started, raw, error: error.message }); throw error; }
      samples.push({ elapsedMs: now() - started, inventory });
      for (const sample of inventory) {
        if (identities.has(sample.physicalIndex) && identities.get(sample.physicalIndex) !== sample.uuid) {
          throw new Error("physical GPU UUID changed during selection");
        }
        identities.set(sample.physicalIndex, sample.uuid);
        windows[sample.physicalIndex] = idleWindow(windows[sample.physicalIndex], sample);
      }
      const physicalIndex = [1, 0].find((index) => windows[index].length === 5);
      if (physicalIndex !== undefined && now() - started <= timeoutMs) {
        return { selected: Object.freeze({ physicalIndex, uuid: identities.get(physicalIndex) }),
          inventory: samples[0].inventory, samples, stableSamples: 5, elapsedMs: now() - started };
      }
      const remaining = timeoutMs - (now() - started);
      if (remaining > 0) await sleep(Math.min(pollMs, remaining));
    }
    throw new Error(`neither physical GPU0 nor GPU1 settled within ${timeoutMs}ms`);
  } catch (error) { error.samples = samples; throw error; }
}

// Capture/replay may settle only its frozen physical card; it never switches or disturbs processes.
export async function settlePostExitGpu(probe, { gpu = LEGACY_GPU, timeoutMs = 30_000, pollMs = 200, now = () => performance.now(),
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)) } = {}) {
  validateGpu(gpu);
  const started = now(); const samples = []; let stable = [];
  const fail = (reason) => {
    const error = new Error(`${reason}; selected GPU${gpu.physicalIndex} samples: ${JSON.stringify(samples)}`);
    error.samples = samples;
    return error;
  };
  while (now() - started < timeoutMs) {
    let raw;
    try {
      raw = await probe(timeoutMs - (now() - started));
      const sample = { ...parseReplayGpuSample(raw, new Date().toISOString(), gpu), elapsedMs: now() - started };
      samples.push(sample);
      stable = idleWindow(stable, sample);
      if (stable.length === 5 && now() - started <= timeoutMs) {
        return { afterBytes: Math.max(...stable.map((item) => item.usedBytes)), samples, stableWindow: stable,
          stableSamples: 5, elapsedMs: now() - started };
      }
    } catch (error) {
      samples.push({ elapsedMs: now() - started, raw, error: error.message });
      throw fail(error.message);
    }
    const remaining = timeoutMs - (now() - started);
    if (remaining > 0) await sleep(Math.min(pollMs, remaining));
  }
  throw fail(`physical GPU${gpu.physicalIndex} did not settle to five consecutive idle stable samples within ${timeoutMs}ms`);
}

export function admittedReplayEnv(env, { state, hub, workerId, port, parentPid, gpu }) {
  const child = Object.fromEntries(Object.entries(env).filter(([key]) => !key.startsWith("SCENEWORKS_")
    && !/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/.test(key)));
  return { ...endpointGpuEnv(child, gpu), SCENEWORKS_WORKER_CHILD: "1",
    SCENEWORKS_PARENT_PID: String(parentPid), SCENEWORKS_WORKER_ID: workerId,
    SCENEWORKS_DATA_DIR: path.join(state, "data"), SCENEWORKS_CONFIG_DIR: path.join(state, "config"),
    SCENEWORKS_JOBS_DB_PATH: path.join(state, "data", "cache", "jobs.db"),
    SCENEWORKS_CREDENTIALS_DIR: path.join(state, "credentials"), SCENEWORKS_API_HOST: "127.0.0.1",
    SCENEWORKS_API_PORT: String(port), SCENEWORKS_API_URL: `http://127.0.0.1:${port}`,
    SCENEWORKS_TRUST_LOOPBACK: "1", SCENEWORKS_POLL_SECONDS: "1", SCENEWORKS_HEARTBEAT_SECONDS: "1",
    SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS: "30", SCENEWORKS_BACKEND_MLX_ENABLED: "false",
    SCENEWORKS_BACKEND_CANDLE_ENABLED: "true", SCENEWORKS_LOG_FORMAT: "json",
    SCENEWORKS_CUDA_VRAM_CAP_GB: "6", HF_HOME: path.dirname(hub), HF_HUB_CACHE: hub,
    HUGGINGFACE_HUB_CACHE: hub, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" };
}

async function runAdmittedReplay({ args, workDir, env, pin, command, inventory, row, gpu, gpuPreflight }) {
  if (process.platform !== "win32") throw new Error("admitted replay requires the native Windows CUDA lane");
  const apiBinary = process.env.SDXL_REPLAY_API;
  if (!apiBinary || !path.isAbsolute(apiBinary)) throw new Error("SDXL_REPLAY_API must name the built native API/worker executable");
  const configBytes = await readFile(path.join(ROOT, "config/sdxl-cuda-memory-endpoints.json"));
  const packageConfig = JSON.parse(configBytes);
  const summarySource = validatePackagedEndpointSummarySource(packageConfig.pair?.summary);
  const summaryBytes = await readFile(path.join(ROOT, summarySource.path));
  if (createHash("sha256").update(summaryBytes).digest("hex") !== summarySource.sha256) throw new Error("packaged endpoint summary digest mismatch");
  const captured = JSON.parse(summaryBytes);
  if (captured.artifactInventory?.sha256 !== inventory.sha256) throw new Error("replay actual assembly differs from the packaged measurement");
  const inferenceSource = await replayInferenceSourceIdentity(captured.inferencePin, pin, path.resolve(args.inferenceRepo));
  const hub = path.dirname(path.dirname(path.dirname(path.dirname(row.tierRoot))));
  for (const key of ["SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_L", "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_BIGG", "SCENEWORKS_SDXL_COMPONENT_VAE_FP16_FIX"]) {
    if (path.dirname(path.dirname(path.dirname(row.env[key]))) !== hub) throw new Error("ordinary worker replay requires all staged inputs in the same existing HF hub");
  }
  const state = path.join(workDir, "owned-app-state"); await mkdir(state, { recursive: true });
  const port = await new Promise((resolve, reject) => { const server = createServer(); server.once("error", reject); server.listen(0, "127.0.0.1", () => { const value = server.address().port; server.close((error) => error ? reject(error) : resolve(value)); }); });
  const workerId = `sdxl-replay-${process.env.GITHUB_RUN_ID ?? process.pid}-${process.env.GITHUB_RUN_ATTEMPT ?? "1"}`;
  const hashFile = async (file) => { const digest = createHash("sha256"); for await (const bytes of createReadStream(file)) digest.update(bytes); return digest.digest("hex"); };
  const receipt = { schemaVersion: 1, profile: "sdxl-cuda-admitted-replay", status: "running", inferencePin: pin,
    inferenceSource,
    apiBinary: { path: apiBinary, sha256: await hashFile(apiBinary) },
    adapterBinary: { path: command[0], sha256: await hashFile(command[0]) },
    sourceRevision: (await run("git", ["rev-parse", "HEAD"])).stdout.trim(),
    packagedEndpointSummary: summarySource, artifactInventory: inventory, workerId, selectedGpu: gpu, gpuPreflight,
    guardedAvailableGiB: 6, physicalEightGiBDeviceValidated: false,
    budgetSemantics: "SCENEWORKS_CUDA_VRAM_CAP_GB=6 caps available and total to six GiB on a larger physical GPU; this is not exact eight-total/six-free emulation",
    deviceMapping: "production SDXL uses default_device/new_cuda(0); the child CUDA_VISIBLE_DEVICES maps that logical device to selectedGpu (source contract, not worker UUID telemetry)",
    telemetry: "sampled selected physical-GPU memory.used growth above a stable pre-request baseline; not a per-process allocator peak", cleanup: [] };
  const save = () => writeFile(path.join(workDir, "sdxl-cuda-admitted-replay.json"), JSON.stringify(receipt, null, 2) + "\n");
  const handles = [], children = []; let api, worker, sentinel, jobId, sampling = false, sampler;
  let interrupted = null; const abort = () => { interrupted = new Error("owned admitted replay interrupted"); };
  process.once("SIGTERM", abort); process.once("SIGINT", abort);
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const exitWithin = async (child, ms) => {
    let timer;
    try { return await Promise.race([child.done, new Promise((resolve) => { timer = setTimeout(() => resolve(null), ms); })]); }
    finally { clearTimeout(timer); }
  };
  const launch = async (program, programArgs, childEnv, name) => {
    const handle = await open(path.join(workDir, `${name}.log`), "wx"); handles.push(handle);
    const child = spawn(program, programArgs, { cwd: ROOT, env: childEnv, stdio: ["pipe", handle.fd, handle.fd] });
    child.done = new Promise((resolve) => { child.once("error", (error) => { child.spawnError = error; resolve({ error: error.message }); }); child.once("exit", (code, signal) => resolve({ code, signal })); });
    children.push(child); return child;
  };
  const alive = () => { if (interrupted) throw interrupted; for (const child of [api, worker].filter(Boolean)) if (child.spawnError || child.exitCode !== null || child.signalCode !== null) throw new Error("owned API/worker exited before replay completion"); };
  const apiCall = async (method, route, body) => {
    const response = await fetch(`http://127.0.0.1:${port}${route}`, { method, ...(body ? { headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(10_000) });
    if (!response.ok) throw new Error(`${method} ${route} returned ${response.status}: ${(await response.text()).slice(0, 1500)}`);
    return response.json();
  };
  const wait = async (probe, seconds, label) => { const deadline = Date.now() + seconds * 1000; while (Date.now() < deadline) { alive(); const value = await probe(); if (value) return value; await delay(1000); } throw new Error(`${label} exceeded ${seconds}s`); };
  const smi = path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "nvidia-smi.exe");
  const gpuRaw = async (remainingMs = 10_000) => (await run(smi,
    ["-i", String(gpu.physicalIndex), `--query-gpu=${GPU_QUERY}`, "--format=csv,noheader,nounits"],
    { env, timeoutMs: Math.min(5_000, remainingMs) })).stdout;
  const gpuSample = async () => parseReplayGpuSample(await gpuRaw(), new Date().toISOString(), gpu);
  const stableBaseline = async (label) => {
    try {
      const settled = await settlePostExitGpu(gpuRaw, { gpu });
      receipt[`${label}Settling`] = settled;
      return settled.stableWindow;
    } catch (error) {
      receipt[`${label}Settling`] = { status: "failed", error: error.message, samples: error.samples };
      throw error;
    }
  };
  try {
    receipt.preflightBaseline = await stableBaseline("preflight"); await save();
    // This task-owned sentinel supplies the API's existing desktop parent-death graceful shutdown.
    sentinel = await launch(process.execPath, ["-e", 'process.stdin.resume();process.stdin.on("end",()=>process.exit(0));'], env, "api-parent");
    const childEnv = admittedReplayEnv(env, { state, hub, workerId, port, parentPid: sentinel.pid, gpu });
    api = await launch(apiBinary, [], { ...childEnv, SCENEWORKS_RUN_UTILITY_INPROCESS: "1", SCENEWORKS_RUST_WORKER_GPU_ID: "cpu" }, "api");
    receipt.health = await wait(async () => { try { const health = await apiCall("GET", "/api/v1/health"); return health.status === "ok" && health.readiness?.status === "ready" && health; } catch { return false; } }, 180, "API readiness");
    worker = await launch(apiBinary, [], { ...childEnv, SCENEWORKS_WORKER_ONLY: "1" }, "worker");
    receipt.worker = await wait(async () => { const value = (await apiCall("GET", "/api/v1/workers")).find((item) => item.id === workerId); return value?.status === "idle" && value; }, 180, "owned CUDA worker registration");
    if (receipt.worker.gpuId !== String(gpu.physicalIndex) || !receipt.worker.capabilities?.includes("image_generate")) throw new Error("owned worker did not register selected GPU image generation");
    receipt.modelLibrary = await apiCall("GET", "/api/v1/model-library");
    const project = await apiCall("POST", "/api/v1/projects", { name: `SDXL issue 3197 admitted replay ${workerId}` }); receipt.project = project;
    const projectPath = await realpath(project.path); const relativeProject = path.relative(await realpath(state), projectPath);
    if (!relativeProject || relativeProject.startsWith("..") || path.isAbsolute(relativeProject)) throw new Error("API project escaped owned replay state");
    receipt.request = admittedReplayRequest(project.id);
    receipt.requestBaseline = await stableBaseline("request");
    const baseline = Math.max(...receipt.requestBaseline.map((item) => item.usedBytes));
    receipt.samples = []; let samplerError;
    sampling = true; sampler = (async () => { try { while (sampling) { receipt.samples.push(await gpuSample()); await delay(200); } } catch (error) { samplerError = error; } })();
    const accepted = await apiCall("POST", "/api/v1/image/jobs", receipt.request); jobId = accepted.id; receipt.acceptedJob = accepted; await save();
    receipt.completedJob = await wait(async () => { const job = await apiCall("GET", `/api/v1/jobs/${jobId}`); if (["failed", "canceled"].includes(job.status)) throw new Error(`ordinary worker job ${job.status}: ${job.error ?? job.message}`); return job.status === "completed" && job; }, 3600, "ordinary admitted image render");
    sampling = false; await sampler; if (samplerError) throw samplerError;
    if (receipt.completedJob.workerId !== workerId || receipt.completedJob.backend !== "candle") throw new Error("completed image was not rendered by the owned CUDA worker");
    receipt.selection = admittedReplaySelection(await readFile(path.join(workDir, "worker.log"), "utf8"));
    receipt.selectionCoverage = assertAdmittedReplaySelection(receipt.selection);
    receipt.selectionSource = "complete emit_event payload flattened into worker.log by SceneworksJsonFormat";
    if (!receipt.samples.some((sample) => sample.gpuUtilization > 0)) throw new Error("no selected physical GPU activity was sampled during the actual render");
    const peak = Math.max(...receipt.samples.map((sample) => sample.usedBytes));
    const observed = Math.max(0, peak - baseline);
    receipt.prediction = { predictedPeakBytes: receipt.selection.predictedPeakBytes, observedSampledGrowthBytes: observed,
      rawDifferenceBytes: observed - receipt.selection.predictedPeakBytes, reserveBytes: Math.round(receipt.selection.reserveGb * 1024 ** 3),
      baselineBytes: baseline, absoluteSampledPeakBytes: peak };
    if (observed <= 0 || observed > receipt.prediction.predictedPeakBytes + receipt.prediction.reserveBytes) throw new Error("observed worker GPU growth exceeded the selected prediction plus its explicit reserve");
    const persisted = await wait(async () => {
      const job = await apiCall("GET", `/api/v1/jobs/${jobId}`);
      const image = persistedReplayAsset(job, project.id, jobId);
      return image && { job, image };
    }, 60, "API persisted image asset");
    receipt.completedJob = persisted.job;
    const image = await apiCall("GET", `/api/v1/projects/${project.id}/assets/${encodeURIComponent(persisted.image.id)}`);
    persistedReplayAsset({ result: { assets: [image], assetIds: [persisted.image.id] } }, project.id, jobId);
    if (image.file.path !== persisted.image.file.path) throw new Error("asset API sidecar differs from the completed job asset");
    const sourcePath = path.resolve(projectPath, image.file.path); const relative = path.relative(projectPath, sourcePath);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) throw new Error("served image escaped owned project");
    const response = await fetch(`http://127.0.0.1:${port}/api/v1/projects/${project.id}/files/${relative.split(path.sep).map(encodeURIComponent).join("/")}`, { signal: AbortSignal.timeout(30_000) });
    if (!response.ok) throw new Error(`product PNG serving failed: ${response.status}`);
    const served = Buffer.from(await response.arrayBuffer()); const outputPath = path.join(workDir, "replay.png"); await writeFile(outputPath, served);
    const inspected = JSON.parse((await run(command[0], command.slice(1), { env, input: JSON.stringify({ action: "sdxl_inspect_output", inputPath: outputPath, outputPath,
      planned: { target: { geometry: { width: 1024, height: 1024 } } } }), timeoutMs: 30_000 })).stdout);
    receipt.output = { ...inspected.output, servedPngSha256: createHash("sha256").update(served).digest("hex"), servedPngBytes: served.length, asset: image, apiStatus: response.status };
    await wait(async () => { const value = (await apiCall("GET", "/api/v1/workers")).find((item) => item.id === workerId); return value?.status === "idle" && value.currentJobId == null; }, 60, "worker drain");
    receipt.status = "passed";
  } catch (error) { receipt.status = "failed"; receipt.error = error.message; throw error; }
  finally {
    sampling = false; if (sampler) await sampler;
    if (jobId && receipt.status !== "passed") await apiCall("POST", `/api/v1/jobs/${jobId}/cancel`, {}).catch(() => {});
    for (const [name, child] of [["worker", worker], ["api-parent", sentinel], ["api", api]]) {
      if (!child?.pid) continue;
      child.stdin.end();
      let result = await exitWithin(child, 40_000);
      const natural = result !== null && !result.error && result.code === 0;
      if (!natural) {
        receipt.status = "failed"; receipt.error ??= `${name} failed natural cleanup`;
        await run("taskkill", ["/PID", String(child.pid), "/T", "/F"], { timeoutMs: 10_000 }).catch(() => {});
        result = await exitWithin(child, 10_000);
      }
      receipt.cleanup.push({ name, pid: child.pid, natural, result });
    }
    await Promise.all(handles.map((handle) => handle.close()));
    try {
      receipt.postExitBaseline = await stableBaseline("postExit");
      const before = Math.max(...(receipt.preflightBaseline ?? []).map((sample) => sample.usedBytes));
      const after = Math.max(...receipt.postExitBaseline.map((sample) => sample.usedBytes));
      if (Number.isFinite(before) && after > before + 64 * 1024 ** 2) throw new Error("owned replay processes retained selected physical GPU residency after exit");
    } catch (error) { receipt.status = "failed"; receipt.error ??= error.message; }
    if ((await run("git", ["status", "--porcelain"])).stdout.trim()) { receipt.status = "failed"; receipt.error ??= "replay modified the source checkout"; }
    receipt.finishedAt = new Date().toISOString(); await save();
    process.removeListener("SIGTERM", abort); process.removeListener("SIGINT", abort);
  }
  if (receipt.status !== "passed") throw new Error(receipt.error ?? "replay cleanup failed");
  process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
}


export async function main(argv = process.argv.slice(2)) {
  const admittedReplay = argv.includes("--admitted-replay");
  const args = parseArgs(argv.filter((item) => item !== "--admitted-replay"));
  if (args.dryRun || args.list) throw new Error("endpoint capture is explicit execution only; use measure-memory-catalog.mjs --dry-run or --list for metadata");
  if (!args.adapter || !args.inferenceRepo || !args.workDir) throw new Error("--adapter, --inference-repo and --work-dir are required");
  if (args.backend && args.backend !== "candle") throw new Error("SDXL endpoints are CUDA only");
  const workDir = path.resolve(args.workDir);
  await mkdir(workDir, { recursive: true });
  for (const repo of [ROOT, path.resolve(args.inferenceRepo)]) {
    const relative = path.relative(await realpath(repo), await realpath(workDir));
    if (!relative.startsWith("..") && !path.isAbsolute(relative)) throw new Error("endpoint output must be outside both clean checkouts");
    if ((await run("git", ["status", "--porcelain"], { cwd: repo })).stdout.trim()) throw new Error(`dirty source checkout: ${repo}`);
  }
  const pin = /INFERENCE_PIN: &str = "([a-f0-9]{40})"/.exec(await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/lib.rs"), "utf8"))?.[1];
  const inferenceHead = (await run("git", ["rev-parse", "HEAD"], { cwd: args.inferenceRepo })).stdout.trim();
  if (pin !== inferenceHead) throw new Error(`inference checkout ${inferenceHead} differs from compiled adapter pin ${pin}`);
  const { plan, rows } = await planRun({ ...args, ...(admittedReplay ? { campaign: `${args.campaign ?? "gi-3197"}-replay` } : {}), backend: "candle", anchors: ["sdxl:q4:candle"], models: ["sdxl"], skipCurrent: false });
  const row = rows[0];
  if (row?.status !== "runnable") throw new Error(`SDXL q4 cannot be captured: ${row?.status}: ${row?.reason}`);
  const inventory = await sdxlAssemblyInventory(row);
  const baseEnv = { ...process.env, ...row.env,
    SCENEWORKS_MEMORY_MODEL_BYTES: String(inventory.bytes), SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256: inventory.sha256 };
  const command = providerCommand(args.adapter);
  const sdxlCrate = path.join(args.inferenceRepo, "crates/media/candle-gen/candle-gen-sdxl");
  const providerSource = await readFile(path.join(sdxlCrate, "src/lib.rs"), "utf8");
  const providerCargo = await readFile(path.join(sdxlCrate, "Cargo.toml"), "utf8");
  if (!/let flash = false;/.test(providerSource) || /^flash-attn\s*=/m.test(providerCargo)) {
    throw new Error("SDXL flash-attention build policy changed; refresh the endpoint build receipt before capture");
  }
  const pipelineSource = await readFile(path.join(sdxlCrate, "src/pipeline.rs"), "utf8");
  const resolvedDefaults = {
    steps: Number(/const DEFAULT_STEPS: usize = (\d+);/.exec(pipelineSource)?.[1]),
    guidance: Number(/const DEFAULT_GUIDANCE: f64 = ([0-9.]+);/.exec(pipelineSource)?.[1]),
    sampler: /const DEFAULT_SAMPLER: &str = "([^"]+)";/.exec(pipelineSource)?.[1],
  };
  if (resolvedDefaults.steps !== 30 || resolvedDefaults.guidance !== 7 || !resolvedDefaults.sampler) throw new Error("cannot bind SDXL provider defaults at the pin");
  const smi = process.platform === "win32"
    ? path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "nvidia-smi.exe") : "/usr/bin/nvidia-smi";
  let gpuPreflight;
  try {
    gpuPreflight = await selectCaptureGpu(async (remainingMs) => (await run(smi,
      ["-i", "0,1", `--query-gpu=${GPU_QUERY}`, "--format=csv,noheader,nounits"],
      { env: baseEnv, timeoutMs: Math.min(5_000, remainingMs) })).stdout);
    await writeFile(path.join(workDir, "gpu-selection.json"), JSON.stringify({ status: "selected", ...gpuPreflight }, null, 2) + "\n");
  } catch (error) {
    await writeFile(path.join(workDir, "gpu-selection.json"), JSON.stringify({ status: "failed", error: error.message, samples: error.samples }, null, 2) + "\n");
    throw error;
  }
  const gpu = gpuPreflight.selected;
  const env = endpointGpuEnv(baseEnv, gpu);
  if (admittedReplay) {
    await runAdmittedReplay({ args, workDir, env, pin, command, inventory, row, gpu, gpuPreflight });
    return;
  }
  const summary = { schemaVersion: 1, profile: "sdxl-cuda-endpoints", inferencePin: pin,
    artifactInventory: inventory, stagedArtifacts: row.env, resolvedDefaults, selectedGpu: gpu, gpuPreflight, physicalEightGiBDeviceValidated: false,
    reserveBytes: 2 * 1024 ** 3, reserveIncludedInMeasuredPeaks: false,
    telemetry: "measured sampled physical-device growth under stable pre/post baseline; not a per-process allocator peak or mathematical upper bound",
    endpoints: [] };
  const gpuRaw = async (remainingMs = 10_000) => (await run(smi,
    ["-i", String(gpu.physicalIndex), `--query-gpu=${GPU_QUERY}`, "--format=csv,noheader,nounits"],
    { env, timeoutMs: Math.min(5_000, remainingMs) })).stdout;
  const usedBytes = async () => parseReplayGpuSample(await gpuRaw(), new Date().toISOString(), gpu).usedBytes;
  for (const endpoint of ENDPOINTS) {
    let measurement;
    const outputPath = path.join(workDir, `${endpoint.name}.png`);
    const planned = endpointCase(plan, endpoint);
    const bundle = await capturePlannedCase({ planned, providerCommand: command,
      sceneWorksRepo: ROOT, inferenceRepo: path.resolve(args.inferenceRepo),
      executeProvider: async (program, programArgs, input) => {
        const request = JSON.parse(input);
        if (request.action === "run") Object.assign(request, { action: "sdxl_endpoint", sdxlEndpoint: endpoint.name, outputPath });
        else if (request.action === "probe") request.action = "sdxl_endpoint_probe";
        const requestFile = path.join(workDir, `${endpoint.name}-${request.action}-request.json`);
        await writeFile(requestFile, JSON.stringify(request, null, 2) + "\n");
        const beforeBytes = request.action === "sdxl_endpoint" ? await usedBytes() : null;
        let result;
        try {
          // Each invocation is a fresh process. The maintained runner owns timeout/tree cleanup.
          result = await run(program, programArgs, { cwd: ROOT, env, input: JSON.stringify(request), timeoutMs: 60 * 60 * 1000 });
        } catch (error) {
          await writeFile(path.join(workDir, `${endpoint.name}-failure.log`), `${error.stdout ?? ""}\n${error.stderr ?? ""}\n${error.message}\n`);
          throw error;
        }
        await writeFile(path.join(workDir, `${endpoint.name}-${request.action}.log`), result.stderr);
        const fragment = JSON.parse(result.stdout);
        if (request.action === "sdxl_endpoint") {
          // Retain the exact adapter response even if post-exit validation fails.
          // This diagnostic is never packaged as validated endpoint evidence.
          await writeFile(path.join(workDir, `${endpoint.name}-response.json`), JSON.stringify(fragment, null, 2) + "\n");
          measurement = fragment.endpointMeasurement;
          delete fragment.endpointMeasurement;
          let settled;
          try {
            settled = await settlePostExitGpu(gpuRaw, { gpu });
          } catch (error) {
            await writeFile(path.join(workDir, `${endpoint.name}-settle.json`), JSON.stringify({ status: "failed", error: error.message, samples: error.samples }, null, 2) + "\n");
            throw error;
          }
          await writeFile(path.join(workDir, `${endpoint.name}-settle.json`), JSON.stringify({ status: "settled", ...settled }, null, 2) + "\n");
          measurement.cleanup = { beforeBytes, afterBytes: settled.afterBytes, toleranceBytes: 64 * 1024 ** 2, adapterProcessExited: true };
          validateEndpoint(measurement, endpoint, pin, gpu.uuid);
        }
        return JSON.stringify(fragment);
      },
    });
    const png = await readFile(outputPath);
    const rgb = await readFile(measurement.output.rgbPath);
    if (createHash("sha256").update(rgb).digest("hex") !== measurement.output.rgbSha256) throw new Error("retained RGB does not match adapter output receipt");
    measurement.output.pngSha256 = createHash("sha256").update(png).digest("hex");
    measurement.output.pngBytes = png.length;
    const bundlePath = path.join(workDir, `${endpoint.name}-evidence.json`);
    const bundleBytes = JSON.stringify(bundle, null, 2) + "\n";
    const evidenceReference = endpointEvidenceReference(bundle, bundleBytes, {
      inventorySha256: inventory.sha256, inferencePin: measurement.inferencePin, deviceUuid: gpu.uuid,
    });
    await writeFile(bundlePath, bundleBytes);
    summary.endpoints.push({ ...measurement, ...evidenceReference, planned, evidencePath: bundlePath });
    await writeFile(path.join(workDir, "sdxl-cuda-endpoints.json"), JSON.stringify(summary, null, 2) + "\n");
  }
  summary.qualityComparison = compareRgb(
    await readFile(summary.endpoints[0].output.rgbPath), await readFile(summary.endpoints[1].output.rgbPath));
  await writeFile(path.join(workDir, "sdxl-cuda-endpoints.json"), JSON.stringify(summary, null, 2) + "\n");
  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
}
