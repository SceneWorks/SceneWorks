#!/usr/bin/env node
// Issue 3197. Exactly two cold SDXL q4 renders, using the maintained catalog artifact resolver
// and calibration capture protocol. Dispatch-only; never imported by a normal-CI inference test.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFile, writeFile, mkdir, realpath } from "node:fs/promises";
import { createHash } from "node:crypto";
import { ROOT, parseArgs, planRun, providerCommand, run } from "./measure-memory-catalog.mjs";
import { capturePlannedCase, logicalCaseId, planAnchor } from "./memory-calibration-harness.mjs";
import { hashArtifactInventory, artifactInventoryReceipt } from "./hash-artifact-inventory.mjs";

export const ENDPOINTS = Object.freeze([
  { name: "resident", strategy: { rung: "resident", engagedRungs: ["resident"], parameters: {} } },
  { name: "sequential", strategy: {
    rung: "bounded_attention",
    engagedRungs: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
    parameters: { stageResidency: true, decodeTileEdge: 512, decodeOverlap: 64, attentionChunkSize: 67_108_864 },
  } },
]);

export function endpointCase(plan, endpoint) {
  const planned = planAnchor(plan, "sdxl:q4:candle");
  planned.loadShape = "eager_materialization";
  planned.target.geometry = { width: 1024, height: 1024, batch: 1, frames: 1 };
  planned.strategy = structuredClone(endpoint.strategy);
  planned.calibrationFingerprint = "sdxl-candle-sdxl-staged-decode-attention-v2";
  // Existing SDXL fixture validation accepts the five-rung prefix; the settings are separately
  // recorded from the actual GenerationRequest, with provider-default sampler and step count.
  planned.fixture = "fresh-five-rung-sdxl-q4-1024-seed16402-default-steps";
  planned.logicalCaseId = logicalCaseId(planned);
  return planned;
}

export function validateEndpoint(measurement, endpoint, expectedInferencePin) {
  if (measurement?.endpoint !== endpoint.name || measurement.deviceUuid !== "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
      || measurement.cudaContextUuid !== measurement.deviceUuid || measurement.logicalCudaDevice !== 0) {
    throw new Error("endpoint identity or physical GPU1 UUID differs from the requested capture");
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

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.dryRun || args.list) throw new Error("endpoint capture is explicit execution only; use measure-memory-catalog.mjs --dry-run or --list for metadata");
  if (!args.adapter || !args.inferenceRepo || !args.workDir) throw new Error("--adapter, --inference-repo and --work-dir are required");
  if (args.backend && args.backend !== "candle") throw new Error("SDXL endpoints are CUDA only");
  if (process.env.CUDA_VISIBLE_DEVICES !== "1" || process.env.SCENEWORKS_GPU_ID !== "1") {
    throw new Error("both CUDA_VISIBLE_DEVICES=1 and SCENEWORKS_GPU_ID=1 are required");
  }
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
  const { plan, rows } = await planRun({ ...args, backend: "candle", anchors: ["sdxl:q4:candle"], models: ["sdxl"], skipCurrent: false });
  const row = rows[0];
  if (row?.status !== "runnable") throw new Error(`SDXL q4 cannot be captured: ${row?.status}: ${row?.reason}`);
  const inventory = await sdxlAssemblyInventory(row);
  const env = { ...process.env, ...row.env,
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
  const summary = { schemaVersion: 1, profile: "sdxl-cuda-endpoints", inferencePin: pin,
    artifactInventory: inventory, stagedArtifacts: row.env, resolvedDefaults, physicalEightGiBDeviceValidated: false,
    reserveBytes: 2 * 1024 ** 3, reserveIncludedInMeasuredPeaks: false,
    telemetry: "measured sampled physical-device growth under stable pre/post baseline; not a per-process allocator peak or mathematical upper bound",
    endpoints: [] };
  const smi = process.platform === "win32"
    ? path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "nvidia-smi.exe") : "/usr/bin/nvidia-smi";
  const usedBytes = async () => {
    const value = Number((await run(smi, ["-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], { env, timeoutMs: 30_000 })).stdout.trim());
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("invalid physical GPU1 cleanup telemetry");
    return value * 1024 ** 2;
  };
  const settledUsedBytes = async () => {
    const samples = [];
    for (let index = 0; index < 5; index++) {
      const values = (await run(smi, ["-i", "1", "--query-gpu=memory.used,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"], { env, timeoutMs: 30_000 })).stdout.trim().split(",").map((value) => Number(value.trim()));
      if (values.length !== 3 || values.some((value) => !Number.isSafeInteger(value) || value < 0)
          || values[1] !== 0 || values[2] !== 0) throw new Error("post-exit physical GPU1 is not idle or observable");
      samples.push(values[0] * 1024 ** 2);
      if (index < 4) await new Promise((resolve) => setTimeout(resolve, 200));
    }
    if (Math.max(...samples) - Math.min(...samples) > 64 * 1024 ** 2) throw new Error("post-exit physical GPU1 baseline is unstable");
    return Math.max(...samples);
  };
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
          measurement = fragment.endpointMeasurement;
          delete fragment.endpointMeasurement;
          measurement.cleanup = { beforeBytes, afterBytes: await settledUsedBytes(), toleranceBytes: 64 * 1024 ** 2, adapterProcessExited: true };
          validateEndpoint(measurement, endpoint, pin);
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
      inventorySha256: inventory.sha256, inferencePin: measurement.inferencePin, deviceUuid: measurement.deviceUuid,
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
