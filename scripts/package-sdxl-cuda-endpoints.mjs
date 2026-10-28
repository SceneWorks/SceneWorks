#!/usr/bin/env node
// Package two completed physical captures; never execute a model or promote runtime certification.
import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { anchorCandidate } from "./extract-memory-anchors.mjs";
import { anchorLoaderDigests } from "./anchor-loader-closure.mjs";
import { validateBundle } from "./memory-calibration-harness.mjs";
import { artifactInventoryReceipt } from "./hash-artifact-inventory.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const EVIDENCE_DIR = "config/sdxl-cuda-memory-endpoint-evidence";
const CONFIG = "config/sdxl-cuda-memory-endpoints.json";
const CORE = "crates/sceneworks-core/src/sdxl_cuda_memory_endpoints.rs";
const SUMMARY = "sdxl-cuda-endpoints.json";
const NAMES = ["resident", "sequential"];
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const require = (ok, reason) => { if (!ok) throw new Error(`SDXL packaging: ${reason}`); };
const relative = (name) => `${EVIDENCE_DIR}/${name}`;

/** Generate rows directly from source observations. The existing extractor's representative
 * selection is intentionally not used: the explicit pair retains both measured compositions. */
export function buildEndpointConfig(summaryRaw, bundleRaws, loaderClosureDigest) {
  require(/^[a-f0-9]{64}$/.test(loaderClosureDigest), "missing captured loader-closure digest");
  const summary = JSON.parse(summaryRaw);
  require(summary.profile === "sdxl-cuda-endpoints" && summary.schemaVersion === 1, "wrong capture profile");
  require(summary.endpoints?.length === 2, "requires exactly two endpoints");
  const inventory = artifactInventoryReceipt(summary.artifactInventory?.entries);
  for (const field of ["files", "bytes", "sha256"]) {
    require(inventory[field] === summary.artifactInventory[field], `ordered artifact inventory ${field} differs`);
  }
  const sharedFiles = ["components/tokenizer_clip_l/tokenizer.json", "components/tokenizer_clip_bigg/tokenizer.json", "components/vae_fp16_fix/diffusion_pytorch_model.safetensors"];
  require(inventory.entries.some((entry) => entry.path.startsWith("base/"))
    && sharedFiles.every((file) => inventory.entries.some((entry) => entry.path === file))
    && inventory.entries.every((entry) => entry.path.startsWith("base/") || sharedFiles.includes(entry.path)),
  "inventory must cover the full base tier and exactly the consumed shared files");
  const pair = {};
  for (const name of NAMES) {
    const matches = summary.endpoints.filter((entry) => entry.endpoint === name);
    require(matches.length === 1, `duplicate or missing ${name} receipt`);
    const receipt = matches[0];
    const raw = bundleRaws[name];
    require(typeof raw === "string", `missing ${name} bundle`);
    const bundle = JSON.parse(raw);
    validateBundle(bundle);
    require(bundle.records.length === 1, `${name} bundle must contain exactly one record`);
    const record = bundle.records[0];
    require(receipt.evidenceSha256 === sha256(raw) && receipt.recordId === record.id, `${name} summary/bundle binding`);
    require(receipt.inferencePin === summary.inferencePin && record.repositories.inference.revision === summary.inferencePin, `${name} capture pin binding`);
    require(record.artifact.inventorySha256 === summary.artifactInventory?.sha256, `${name} inventory binding`);
    require(record.loadShape === "eager_materialization", `${name} must measure stock eager materialization`);
    const sourcePath = relative(`${name}-evidence.json`);
    const candidate = anchorCandidate(record, { path: sourcePath, sha256: sha256(raw) });
    require(candidate !== null, `${name} lacks observed anchor phases`);
    require(candidate.modelId === "sdxl" && candidate.backend === "candle" && candidate.tier === "q4", `${name} cell differs`);
    require(candidate.overallAllocatorEnvelopeBytes === receipt.wholeRunPeakBytes, `${name} whole-run peak differs`);
    const fields = ["modelId", "route", "provider", "backend", "tier", "transformerVariant", "decoder", "mode", "overlay", "referenceCount", "loadShape", "measuredRegime", "geometry", "phaseActivePeakBytes", "overallAllocatorEnvelopeBytes"];
    pair[name] = {
      id: `sdxl:candle:q4:${name}:${record.id}`,
      modelFamily: "sdxl",
      ...Object.fromEntries(fields.map((field) => [field, candidate[field]])),
      source: { path: sourcePath, sha256: sha256(raw), recordId: record.id,
        calibrationFingerprint: candidate.calibrationFingerprint, loaderClosureDigest },
      ...(candidate.phaseAllocatorEnvelopeBytes === null ? {} : {
        phaseAllocatorEnvelopeBytes: candidate.phaseAllocatorEnvelopeBytes,
      }),
    };
  }
  pair.summary = { path: relative(SUMMARY), sha256: sha256(summaryRaw) };
  return { schemaVersion: 1, pair };
}

/** Check retained bytes before making their receipt available to admission. Comparison metrics
 * are measurements only; this performs no quality thresholding or parity promotion. */
export function verifyOutputBytes(summary, outputs) {
  for (const name of NAMES) {
    const output = summary.endpoints.find((entry) => entry.endpoint === name)?.output;
    require(output, `missing ${name} output receipt`);
    for (const [extension, digestKey, sizeKey] of [["png", "pngSha256", "pngBytes"], ["rgb", "rgbSha256", "rgbBytes"]]) {
      const bytes = outputs[`${name}.${extension}`];
      require(Buffer.isBuffer(bytes) && bytes.length > 0 && bytes.length === output[sizeKey] && sha256(bytes) === output[digestKey], `${name}.${extension} bytes do not match capture`);
    }
    require(output.width === 1024 && output.height === 1024 && output.channels === 3 && output.rgbBytes === 1024 * 1024 * 3, `${name} RGB geometry differs`);
    const rgb = outputs[`${name}.rgb`];
    let nonFlat = false;
    let minimum = 255;
    let maximum = 0;
    for (let index = 0; index < rgb.length; index++) {
      minimum = Math.min(minimum, rgb[index]); maximum = Math.max(maximum, rgb[index]);
      nonFlat ||= rgb[index] !== rgb[index % 3];
    }
    require(nonFlat && output.spatiallyNonFlat === true && minimum === output.minimumChannel && maximum === output.maximumChannel, `${name} RGB receipt differs`);
  }
  const resident = outputs["resident.rgb"];
  const sequential = outputs["sequential.rgb"];
  let total = 0; let maximum = 0; let changed = 0;
  for (let index = 0; index < resident.length; index++) {
    const delta = Math.abs(resident[index] - sequential[index]);
    total += delta; maximum = Math.max(maximum, delta); changed += Number(delta !== 0);
  }
  const measured = { result: "measured", reference: "resident", selected: "sequential", channels: resident.length,
    maximumAbsoluteRgbDelta255: maximum, meanAbsoluteRgbDelta255: total / resident.length,
    maximumAbsoluteError: maximum / 255, meanAbsoluteError: total / resident.length / 255,
    changedChannels: changed, identical: changed === 0 };
  for (const [key, value] of Object.entries(measured)) {
    require(summary.qualityComparison?.[key] === value, `RGB comparison ${key} differs from retained bytes`);
  }
}

export function compiledSources(config) {
  return [config.pair.resident.source.path, config.pair.sequential.source.path, config.pair.summary.path]
    .map((source) => `    (\n        "${source}",\n        include_str!("../../../${source}"),\n    ),`)
    .join("\n");
}

async function readExisting(file) {
  try { return await readFile(file); } catch (error) { if (error.code === "ENOENT") return null; throw error; }
}

export async function packageEndpoints({ captureDir, inferenceRepo, write = false, root = ROOT }) {
  const summaryRaw = await readFile(path.join(captureDir, SUMMARY), "utf8");
  const summary = JSON.parse(summaryRaw);
  require(/^[a-f0-9]{40}$/.test(summary.inferencePin), "invalid captured inference revision");
  const bundleRaws = Object.fromEntries(await Promise.all(NAMES.map(async (name) => [name,
    await readFile(path.join(captureDir, `${name}-evidence.json`), "utf8")])));
  const outputs = Object.fromEntries(await Promise.all(NAMES.flatMap((name) => ["png", "rgb"].map(async (extension) => {
    const file = `${name}.${extension}`;
    return [file, await readFile(path.join(captureDir, file))];
  }))));
  verifyOutputBytes(summary, outputs);
  const declared = JSON.parse(await readFile(path.join(root, "config/anchor-loader-closures.json"), "utf8")).models;
  // Read the exact captured Git tree; never stamp with the active pin or the clone's HEAD.
  const entry = anchorLoaderDigests({ repo: inferenceRepo, revision: summary.inferencePin,
    declared: { "sdxl:candle": declared["sdxl:candle"] } }).get("sdxl:candle");
  const config = buildEndpointConfig(summaryRaw, bundleRaws, entry.digest);
  const configRaw = `${JSON.stringify(config, null, 2)}\n`;
  const core = await readFile(path.join(root, CORE), "utf8");
  const table = `const PACKAGED_ENDPOINT_SOURCES: &[(&str, &str)] = &[\n${compiledSources(config)}\n];`;
  const pattern = /const PACKAGED_ENDPOINT_SOURCES: &\[\(&str, &str\)\] = &\[[\s\S]*?\];/g;
  require([...core.matchAll(pattern)].length === 1, "compiled source table not uniquely found");
  const nextCore = core.replace(pattern, () => table);
  const oldConfig = await readExisting(path.join(root, CONFIG));
  require(oldConfig && (JSON.parse(oldConfig).pair === null || oldConfig.toString() === configRaw), "refusing to replace a different packaged pair");
  const files = new Map([[relative(SUMMARY), Buffer.from(summaryRaw)], ...NAMES.map((name) =>
    [relative(`${name}-evidence.json`), Buffer.from(bundleRaws[name])]),
    ...Object.entries(outputs).filter(([name]) => name.endsWith(".png")).map(([name, bytes]) => [relative(name), bytes])]);
  // Complete every verification and collision check before the first write.
  for (const [file, bytes] of files) {
    const existing = await readExisting(path.join(root, file));
    require(existing === null || existing.equals(bytes), `refusing to replace different evidence ${file}`);
  }
  if (write) {
    await mkdir(path.join(root, EVIDENCE_DIR), { recursive: true });
    for (const [file, bytes] of files) await writeFile(path.join(root, file), bytes);
    await writeFile(path.join(root, CONFIG), configRaw);
    await writeFile(path.join(root, CORE), nextCore);
  }
  return { wrote: write, capturedInferenceRevision: summary.inferencePin, loaderClosureDigest: entry.digest,
    config, compiledSources: table, retainedFiles: [...files.keys()],
    rgbRetention: "RGB verified byte-for-byte and bound by immutable summary hashes/metrics; raw RGB remains in the campaign artifact, not the runtime package.",
    nextCheck: "Run the native core sdxl_endpoint_ tests, inspect both retained PNGs, then validate worker admission/replay. No runtime certification was created." };
}

export async function main(argv = process.argv.slice(2)) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    if (argv[index] === "--write") options.write = true;
    else if (["--capture-dir", "--inference-repo"].includes(argv[index])) {
      require(argv[index + 1] && !argv[index + 1].startsWith("--"), `missing value for ${argv[index]}`);
      options[argv[index] === "--capture-dir" ? "captureDir" : "inferenceRepo"] = path.resolve(argv[++index]);
    } else throw new Error(`Unknown argument ${argv[index]}`);
  }
  require(options.captureDir && options.inferenceRepo, "use --capture-dir <downloaded capture> --inference-repo <clone containing captured SHA> [--write]");
  process.stdout.write(`${JSON.stringify(await packageEndpoints(options), null, 2)}\n`);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
}
