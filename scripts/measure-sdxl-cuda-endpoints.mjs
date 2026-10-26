#!/usr/bin/env node
// Issue 3197. Exactly two cold SDXL q4 renders, using the maintained catalog artifact resolver
// and calibration capture protocol. Dispatch-only; never imported by a normal-CI inference test.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFile, writeFile, mkdir, realpath } from "node:fs/promises";
import { createHash } from "node:crypto";
import { ROOT, parseArgs, planRun, providerCommand, run } from "./measure-memory-catalog.mjs";
import { capturePlannedCase, logicalCaseId, planAnchor } from "./memory-calibration-harness.mjs";
import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";

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
  planned.target.geometry = { width: 1024, height: 1024, batch: 1, frames: 1 };
  planned.strategy = structuredClone(endpoint.strategy);
  planned.calibrationFingerprint = "sdxl-candle-sdxl-staged-decode-attention-v2";
  // Existing SDXL fixture validation accepts the five-rung prefix; the settings are separately
  // recorded from the actual GenerationRequest, with provider-default sampler and step count.
  planned.fixture = "fresh-five-rung-sdxl-q4-1024-seed16402-default-steps";
  planned.logicalCaseId = logicalCaseId(planned);
  return planned;
}

export function validateEndpoint(measurement, endpoint) {
  if (measurement?.endpoint !== endpoint.name || measurement.deviceUuid !== "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
      || measurement.cudaContextUuid !== measurement.deviceUuid || measurement.logicalCudaDevice !== 0) {
    throw new Error("endpoint identity or physical GPU1 UUID differs from the requested capture");
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
  const cleanup = measurement.cleanup;
  if (!cleanup || cleanup.adapterProcessExited !== true
      || [cleanup.beforeBytes, cleanup.afterBytes, cleanup.toleranceBytes].some((value) => !Number.isSafeInteger(value) || value < 0)
      || cleanup.afterBytes > cleanup.beforeBytes + cleanup.toleranceBytes) throw new Error("endpoint leaked device residency or lacks exit telemetry");
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
  const inventory = await hashArtifactInventory(row.tierRoot, row.inventoryFiles
    ? { includeFiles: row.inventoryFiles, trustedRoot: path.dirname(path.dirname(row.tierRoot)) } : {});
  const env = { ...process.env, ...row.env,
    SCENEWORKS_MEMORY_MODEL_BYTES: String(inventory.bytes), SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256: inventory.sha256 };
  const command = providerCommand(args.adapter);
  const pipelineSource = await readFile(path.join(args.inferenceRepo, "crates/media/candle-gen/candle-gen-sdxl/src/pipeline.rs"), "utf8");
  const resolvedDefaults = {
    steps: Number(/const DEFAULT_STEPS: usize = (\d+);/.exec(pipelineSource)?.[1]),
    guidance: Number(/const DEFAULT_GUIDANCE: f64 = ([0-9.]+);/.exec(pipelineSource)?.[1]),
    sampler: /const DEFAULT_SAMPLER: &str = "([^"]+)";/.exec(pipelineSource)?.[1],
  };
  if (resolvedDefaults.steps !== 30 || resolvedDefaults.guidance !== 7 || !resolvedDefaults.sampler) throw new Error("cannot bind SDXL provider defaults at the pin");
  const summary = { schemaVersion: 1, profile: "sdxl-cuda-endpoints", inferencePin: pin,
    artifactInventory: inventory, stagedArtifacts: row.env, resolvedDefaults, physicalEightGiBDeviceValidated: false,
    reserveBytes: 2 * 1024 ** 3, reserveIncludedInMeasuredPeaks: false,
    telemetry: "stable-baseline-subtracted physical-GPU1 device-level upper bound; not a per-process allocator peak",
    endpoints: [] };
  const smi = process.platform === "win32"
    ? path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "nvidia-smi.exe") : "/usr/bin/nvidia-smi";
  const usedBytes = async () => {
    const value = Number((await run(smi, ["-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], { env, timeoutMs: 30_000 })).stdout.trim());
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("invalid physical GPU1 cleanup telemetry");
    return value * 1024 ** 2;
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
          measurement.cleanup = { beforeBytes, afterBytes: await usedBytes(), toleranceBytes: 64 * 1024 ** 2, adapterProcessExited: true };
          validateEndpoint(measurement, endpoint);
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
    await writeFile(bundlePath, JSON.stringify(bundle, null, 2) + "\n");
    summary.endpoints.push({ ...measurement, planned, evidencePath: bundlePath });
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
