#!/usr/bin/env node
// Source-owned, idempotent host provisioning for the terminal campaign. It
// acquires immutable inputs and assembles manifests; it never starts a product
// service, invokes a model, writes an install receipt, or claims a campaign.
import { createHash } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import { copyFile, lstat, mkdir, readFile, readdir, realpath, rename, rm, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import { inventory } from "./starvector-terminal-producer.mjs";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";

const execFile = promisify(execFileCallback);
const SHA256 = /^[a-f0-9]{64}$/;
const REVISION = /^[a-f0-9]{40}$/;
const sha = (value) => createHash("sha256").update(value).digest("hex");
const die = (message) => { throw new Error(`starvector terminal provision: ${message}`); };
const json = async (file) => JSON.parse(await readFile(file, "utf8"));

async function sourceFile(root, rootReal, file) {
  const info = await lstat(file);
  if (info.isFile()) return info;
  if (!info.isSymbolicLink()) die(`source trees reject non-regular entry ${file}`);
  const target = await realpath(file), relative = path.relative(rootReal, target);
  if (!relative || relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) die(`source tree symlink escapes ${root}: ${file}`);
  const targetInfo = await lstat(target);
  if (!targetInfo.isFile()) die(`source tree symlink is not a regular file: ${file}`);
  return targetInfo;
}

async function tree(root, symlinkBoundary = root) {
  const entries = [];
  const rootInfo = await lstat(root).catch(() => null);
  if (!rootInfo?.isDirectory() || rootInfo.isSymbolicLink()) die(`regular source directory required: ${root}`);
  const boundaryReal = await realpath(symlinkBoundary);
  for (const name of await readdir(root, { recursive: true })) {
    const file = path.join(root, name), info = await lstat(file);
    if (info.isDirectory()) continue;
    const regular = await sourceFile(symlinkBoundary, boundaryReal, file);
    entries.push({ path: name.split(path.sep).join("/"), byte_size: regular.size, sha256: sha(await readFile(file)) });
  }
  entries.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  if (entries.length === 0) die(`source tree is empty: ${root}`);
  return { entries, aggregate_sha256: sha(JSON.stringify(entries)) };
}

async function copyTree(source, destination, symlinkBoundary = source) {
  const sourceIdentity = await tree(source, symlinkBoundary), existing = await lstat(destination).catch(() => null);
  if (existing) {
    if (!existing.isDirectory() || existing.isSymbolicLink()) die(`destination is not a regular directory: ${destination}`);
    const destinationIdentity = await tree(destination);
    if (JSON.stringify(destinationIdentity) !== JSON.stringify(sourceIdentity)) die(`destination already exists with different bytes: ${destination}`);
    return sourceIdentity;
  }
  const staging = `${destination}.staging-${process.pid}`;
  await rm(staging, { recursive: true, force: true });
  await mkdir(staging, { recursive: true });
  try {
    for (const entry of sourceIdentity.entries) {
      const output = path.join(staging, ...entry.path.split("/"));
      await mkdir(path.dirname(output), { recursive: true }); await copyFile(path.join(source, ...entry.path.split("/")), output);
    }
    if (JSON.stringify(await tree(staging)) !== JSON.stringify(sourceIdentity)) die(`staged tree copy drifted: ${destination}`);
    await mkdir(path.dirname(destination), { recursive: true }); await rename(staging, destination);
  } catch (error) { await rm(staging, { recursive: true, force: true }); throw error; }
  return sourceIdentity;
}

function huggingFaceSnapshot(hfHome, repo, revision) {
  const parts = repo.split("/");
  if (parts.length !== 2 || parts.some((part) => !/^[A-Za-z0-9._-]+$/.test(part) || part === "." || part === "..") || !REVISION.test(revision)) die(`unsafe Hugging Face receipt identity ${repo}@${revision}`);
  return path.join(hfHome, "hub", `models--${parts.join("--")}`, "snapshots", revision);
}

export async function validateServiceReceipts(appDataRoot, hfHomeRoot, identities) {
  const receipts = [];
  for (const name of await readdir(appDataRoot, { recursive: true })) {
    if (path.basename(name) !== ".sceneworks-download-complete.json") continue;
    const file = path.join(appDataRoot, name), info = await lstat(file);
    if (!info.isFile() || info.isSymbolicLink()) die(`service install receipt is not a regular file: ${file}`);
    let value; try { value = JSON.parse(await readFile(file, "utf8")); } catch { die(`service install receipt is not valid JSON: ${file}`); }
    receipts.push(...[value, ...(Array.isArray(value?.receipts) ? value.receipts : [])]);
  }
  const snapshots = new Map(), hfHomeReal = await realpath(hfHomeRoot);
  for (const identity of identities) {
    const { repo, revision, modelId, variant } = identity;
    const matches = receipts.filter((receipt) => receipt?.schemaVersion === 2 && receipt.repo === repo && receipt.snapshotRevision === revision && (!modelId || receipt.modelId === modelId) && (!variant || receipt.variant === variant) && Array.isArray(receipt.resolvedFiles) && receipt.resolvedFiles.length > 0);
    if (matches.length === 0) die(`service app-data closure lacks a source-produced receipt for ${repo}@${revision}`);
    const snapshot = huggingFaceSnapshot(hfHomeRoot, repo, revision);
    for (const receipt of matches) for (const relative of receipt.resolvedFiles) {
      if (typeof relative !== "string" || !relative || path.isAbsolute(relative) || path.win32.isAbsolute(relative) || relative.split(/[\\/]/).includes("..")) die(`service install receipt contains unsafe resolved file for ${repo}@${revision}`);
      const file = path.join(snapshot, ...relative.split(/[\\/]/)), info = await lstat(file).catch(() => null);
      if (!info) die(`service HF closure lacks resolved file ${relative} for ${repo}@${revision}`);
      const regular = await sourceFile(hfHomeRoot, hfHomeReal, file);
      if (regular.size === 0) die(`service HF closure has empty resolved file ${relative} for ${repo}@${revision}`);
    }
    snapshots.set(`${repo}@${revision}`, snapshot);
  }
  return snapshots;
}

async function writeExact(file, bytes) {
  const existing = await lstat(file).catch(() => null);
  if (existing) {
    if (!existing.isFile() || existing.isSymbolicLink() || sha(await readFile(file)) !== sha(bytes)) die(`existing generated identity differs: ${file}`);
    return;
  }
  await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, bytes, { flag: "wx" });
}

export async function downloadExact(url, destination, expected) {
  if (!SHA256.test(expected)) die("download requires an exact SHA-256");
  const existing = await lstat(destination).catch(() => null);
  if (existing) {
    if (!existing.isFile() || existing.isSymbolicLink() || sha(await readFile(destination)) !== expected) die(`existing download differs: ${destination}`);
    return destination;
  }
  const parsed = new URL(url); if (parsed.protocol !== "https:") die("only HTTPS download sources are allowed");
  const response = await fetch(parsed, { redirect: "follow" }); if (!response.ok) die(`download failed ${response.status}: ${url}`);
  const bytes = Buffer.from(await response.arrayBuffer()); if (sha(bytes) !== expected) die(`download digest mismatch: ${url}`);
  await mkdir(path.dirname(destination), { recursive: true }); await writeFile(destination, bytes, { flag: "wx" }); return destination;
}

export async function installCheckout(source, destination, revision) {
  if (!REVISION.test(revision) || !path.isAbsolute(source) || !path.isAbsolute(destination)) die("checkout requires absolute roots and an exact SHA");
  const existing = await lstat(destination).catch(() => null);
  if (!existing) {
    await mkdir(path.dirname(destination), { recursive: true });
    await execFile("git", ["clone", "--no-hardlinks", "--no-checkout", source, destination]);
    await execFile("git", ["-C", destination, "checkout", "--detach", revision]);
  }
  const head = (await execFile("git", ["-C", destination, "rev-parse", "HEAD"])).stdout.trim();
  const dirty = (await execFile("git", ["-C", destination, "status", "--porcelain"])).stdout.trim();
  if (head !== revision || dirty) die("published inference checkout is not exact and clean");
  for (const relative of ["release/starvector-terminal-receipt-v1.schema.json", "release/starvector-terminal-corpus-v1.json", "scripts/release/starvector_terminal_evidence.mjs"]) {
    const info = await lstat(path.join(destination, relative)).catch(() => null); if (!info?.isFile() || info.isSymbolicLink()) die(`inference checkout lacks ${relative}`);
  }
  return head;
}

export async function assemblePreflight(source, destination, revision) {
  const index = await json(path.join(source, "starvector-terminal-preflight.json"));
  if (index.head_sha !== revision || typeof index.workflow_run_id !== "string" || !index.workflow_run_id || !Number.isInteger(index.workflow_run_attempt) || index.workflow_run_attempt < 1 || !Array.isArray(index.inventory_artifacts) || index.inventory_artifacts.length !== 2 || !Array.isArray(index.hook_logs) || index.hook_logs.length !== 4) die("preflight bundle identity/cardinality mismatches dispatch revision");
  const inventories = index.inventory_artifacts.map((entry) => entry?.tier).sort();
  const hooks = index.hook_logs.map((entry) => `${entry?.backend}:${entry?.tier}`).sort();
  if (JSON.stringify(inventories) !== JSON.stringify(["1b", "8b"]) || JSON.stringify(hooks) !== JSON.stringify(["candle-cuda:1b", "candle-cuda:8b", "mlx:1b", "mlx:8b"])) die("preflight bundle inventory or hook identities are incomplete");
  const required = [...index.inventory_artifacts, ...index.hook_logs];
  for (const entry of required) {
    if (!entry.path || path.isAbsolute(entry.path) || entry.path.split(/[\\/]/).includes("..") || !SHA256.test(entry.sha256 ?? "")) die("preflight bundle contains an unsafe entry");
    const file = path.join(source, ...entry.path.split(/[\\/]/)), info = await lstat(file).catch(() => null);
    if (!info?.isFile() || info.isSymbolicLink() || info.size === 0 || sha(await readFile(file)) !== entry.sha256) die(`preflight artifact is missing or drifted: ${entry.path}`);
  }
  await copyTree(source, destination); return index;
}

export async function assembleWeights({ hostRoot, serviceAppData, serviceHfHome, promptProvider, promptModel, promptRevision }) {
  if (promptProvider !== "candle_flux" || promptModel !== "flux_schnell") die("terminal prompt raster must use the Windows Candle FLUX schnell route");
  const weightsRoot = path.join(hostRoot, "weights");
  const models = {};
  for (const [key, relativePath, revision] of [
    ["starvector-1b", "models/starvector-1b", "380ab95d25a8e9ab1dc825debe238b4953ae13b9"],
    ["starvector-8b", "models/starvector-8b", "518beea8dcb5f7a37c5911e92d1d62a76beee7f9"],
  ]) {
    const identity = await inventory(path.join(weightsRoot, ...relativePath.split("/")));
    if (identity.entries.length === 0) die(`${key} snapshot is empty`);
    models[key] = { relative_path: relativePath, revision, inventory_sha256: identity.aggregate_sha256 };
  }
  const serviceSnapshots = await validateServiceReceipts(serviceAppData, serviceHfHome, [
    { repo: "starvector/starvector-1b-im2svg", revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", modelId: "starvector_1b" },
    { repo: "starvector/starvector-8b-im2svg", revision: "518beea8dcb5f7a37c5911e92d1d62a76beee7f9", modelId: "starvector_8b" },
    { repo: "SceneWorks/flux1-schnell-mlx", revision: promptRevision, modelId: "flux_schnell", variant: "q4" },
  ]);
  const promptSnapshot = serviceSnapshots.get(`SceneWorks/flux1-schnell-mlx@${promptRevision}`), promptRuntime = path.join(promptSnapshot, "q4");
  const promptDestination = path.join(weightsRoot, "models", "prompt-raster");
  // Inventory and copy the exact q4 directory the product loader will resolve
  // from the receipt-backed offline HF closure. A second operator-supplied copy
  // could drift while still carrying the right revision label.
  const promptIdentity = await copyTree(promptRuntime, promptDestination, serviceHfHome);
  const promptInventory = Buffer.from(`${JSON.stringify(promptIdentity, null, 2)}\n`);
  const promptInventoryPath = "inventories/prompt-raster-inventory-v1.json";
  await writeExact(path.join(weightsRoot, ...promptInventoryPath.split("/")), promptInventory);
  const appDestination = path.join(weightsRoot, "service-closure", "app-data"), hfDestination = path.join(weightsRoot, "service-closure", "hf-home");
  const appIdentity = await copyTree(serviceAppData, appDestination), hfIdentity = await copyTree(serviceHfHome, hfDestination);
  const manifest = { schema_version: 1, models, prompt_raster: { provider_id: promptProvider, model: promptModel, revision: promptRevision, relative_path: "models/prompt-raster", inventory_path: promptInventoryPath, inventory_sha256: sha(promptInventory) }, terminal_service_closure: { app_data_relative_path: "service-closure/app-data", app_data_sha256: appIdentity.aggregate_sha256, hf_home_relative_path: "service-closure/hf-home", hf_home_sha256: hfIdentity.aggregate_sha256 } };
  await writeExact(path.join(weightsRoot, "starvector-terminal-weights-v1.json"), Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`));
  return manifest;
}

export async function assembleMetrics({ sceneWorksRoot, metricsRoot, python, clipRevision }) {
  const lockBytes = await readFile(path.join(sceneWorksRoot, "release", "starvector-terminal-metrics-lock-v1.json")), lock = JSON.parse(lockBytes);
  const packages = lock.required_packages;
  const probe = `import importlib.metadata as m,json\nprint(json.dumps({name:m.version(name) for name in ${JSON.stringify(packages.map(({ name }) => name))}}))`;
  const installed = JSON.parse((await execFile(python, ["-c", probe])).stdout);
  for (const entry of packages) if (installed[entry.name] !== entry.version) die(`metric package ${entry.name} is not exactly ${entry.version}`);
  const manifest = { schema_version: 1, metrics_lock_sha256: sha(lockBytes), packages, weights: { lpips_linear: { path: "checkpoints/lpips-v0.1-alex.pth", sha256: lock.lpips.linear_weights_sha256 }, alexnet: { path: "checkpoints/alexnet-owt-7be5be79.pth", sha256: lock.lpips.alexnet_weights_sha256 } }, clip: { provider_id: "open-clip-torch", model: lock.clip.model, revision: clipRevision, checkpoint: { path: "checkpoints/open_clip_pytorch_model.bin", sha256: "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad" } } };
  for (const entry of [manifest.weights.lpips_linear, manifest.weights.alexnet, manifest.clip.checkpoint]) {
    const info = await lstat(path.join(metricsRoot, ...entry.path.split("/"))).catch(() => null); if (!info?.isFile() || info.isSymbolicLink() || sha(await readFile(path.join(metricsRoot, ...entry.path.split("/")))) !== entry.sha256) die(`metric checkpoint missing or drifted: ${entry.path}`);
  }
  await writeExact(path.join(metricsRoot, "starvector-terminal-metrics-environment-v1.json"), Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`)); return manifest;
}

async function main(argv) {
  const [command, ...args] = argv;
  if (command === "download" && args.length === 3) return downloadExact(args[0], args[1], args[2]);
  if (command === "checkout" && args.length === 3) return installCheckout(args[0], args[1], args[2]);
  if (command === "preflight" && args.length === 3) return assemblePreflight(args[0], args[1], args[2]);
  if (command === "weights" && args.length === 6) return assembleWeights({ hostRoot: args[0], serviceAppData: args[1], serviceHfHome: args[2], promptProvider: args[3], promptModel: args[4], promptRevision: args[5] });
  if (command === "metrics" && args.length === 4) return assembleMetrics({ sceneWorksRoot: args[0], metricsRoot: args[1], python: args[2], clipRevision: args[3] });
  die("usage: download <url> <path> <sha256> | checkout <source> <destination> <sha> | preflight <source> <destination> <sha> | weights <host-root> <app-data-source> <hf-source> <provider> <model> <revision> | metrics <sceneworks-root> <metrics-root> <python> <clip-revision>");
}

if (isExecutedModule(import.meta.url)) main(process.argv.slice(2)).catch((error) => { console.error(error.message); process.exitCode = 1; });
