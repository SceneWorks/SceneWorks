#!/usr/bin/env node
// Source-owned, idempotent host provisioning for the terminal campaign. It
// acquires immutable inputs and assembles manifests; it never starts a product
// service, invokes a model, writes an install receipt, or claims a campaign.
import { createHash, randomUUID } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import { copyFile, lstat, mkdir, readFile, readdir, realpath, rename, rm, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import path from "node:path";
import { inventory } from "./starvector-terminal-producer.mjs";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
import { readPlanAndLock } from "./starvector-terminal-campaign.mjs";
import { fileSha256 } from "./lib/file-sha256.mjs";
import { assertTerminalPhysicalContainment, assertTerminalPinPhysicalContainment, ensureTerminalPhysicalDirectory } from "./lib/starvector-terminal-pin-paths.mjs";
import { sortTerminalTreeEntries, terminalTreeEntry, terminalTreeSha256 } from "./lib/terminal-tree-identity.mjs";

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

export async function tree(root, symlinkBoundary = root, digestFile = fileSha256) {
  const entries = [];
  const rootInfo = await lstat(root).catch(() => null);
  if (!rootInfo?.isDirectory() || rootInfo.isSymbolicLink()) die(`regular source directory required: ${root}`);
  const boundaryReal = await realpath(symlinkBoundary);
  for (const name of await readdir(root, { recursive: true })) {
    const file = path.join(root, name), info = await lstat(file);
    if (info.isDirectory()) continue;
    const regular = await sourceFile(symlinkBoundary, boundaryReal, file);
    entries.push(terminalTreeEntry(name.split(path.sep).join("/"), regular.size, await digestFile(file)));
  }
  const canonicalEntries = sortTerminalTreeEntries(entries);
  if (canonicalEntries.length === 0) die(`source tree is empty: ${root}`);
  return { entries: canonicalEntries, aggregate_sha256: terminalTreeSha256(canonicalEntries) };
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

async function validatePublishedCheckout(destination, revision) {
  const existing = await lstat(destination).catch((error) => error?.code === "ENOENT" ? null : Promise.reject(error));
  if (!existing) return null;
  if (!existing.isDirectory() || existing.isSymbolicLink()) die("published inference checkout is not an ordinary directory");
  let head;
  let dirty;
  try {
    head = (await execFile("git", ["-C", destination, "rev-parse", "HEAD"])).stdout.trim();
    dirty = (await execFile("git", ["-C", destination, "status", "--porcelain"])).stdout.trim();
  } catch {
    die("published inference checkout is not exact and clean");
  }
  if (head !== revision || dirty) die("published inference checkout is not exact and clean");
  for (const relative of ["release/starvector-terminal-receipt-v1.schema.json", "release/starvector-terminal-corpus-v1.json", "scripts/release/starvector_terminal_evidence.mjs"]) {
    const info = await lstat(path.join(destination, relative)).catch(() => null); if (!info?.isFile() || info.isSymbolicLink()) die(`inference checkout lacks ${relative}`);
  }
  return head;
}

export async function installCheckout(source, destination, revision) {
  if (!REVISION.test(revision) || !path.isAbsolute(source) || !path.isAbsolute(destination)) die("checkout requires absolute roots and an exact SHA");
  const existing = await lstat(destination).catch(() => null);
  if (existing) return validatePublishedCheckout(destination, revision);
  await mkdir(path.dirname(destination), { recursive: true });
  const staging = `${destination}.staging-${process.pid}-${randomUUID()}`;
  try {
    await execFile("git", ["clone", "--no-hardlinks", "--no-checkout", source, staging]);
    await execFile("git", ["-C", staging, "checkout", "--detach", revision]);
    await validatePublishedCheckout(staging, revision);
    try {
      await rename(staging, destination);
    } catch (error) {
      if (!["EEXIST", "ENOTEMPTY", "EPERM"].includes(error?.code)) throw error;
      await validatePublishedCheckout(destination, revision);
    }
    return validatePublishedCheckout(destination, revision);
  } finally {
    // The UUID makes this process the sole owner of the staging path. Never
    // sweep siblings: another direct provision may be cloning there.
    await rm(staging, { recursive: true, force: true });
  }
}

export function pinnedCheckoutLockPath(hostRoot, revision) {
  if (!REVISION.test(revision) || !path.isAbsolute(hostRoot)) die("checkout lock requires an absolute host root and exact SHA");
  return path.join(hostRoot, ".locks", `inference-checkout-${revision}.lock`);
}

function processIsAlive(pid) {
  if (!Number.isSafeInteger(pid) || pid < 1) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return error?.code === "EPERM"; }
}

async function removeOwnedLock(lockRoot, token) {
  const info = await lstat(lockRoot).catch((error) => error?.code === "ENOENT" ? null : Promise.reject(error));
  if (!info) return;
  if (!info.isDirectory() || info.isSymbolicLink()) die(`checkout lock is not an ordinary directory: ${lockRoot}`);
  const owner = await json(path.join(lockRoot, "owner.json")).catch(() => null);
  if (owner?.token !== token) die(`checkout lock ownership changed before release: ${lockRoot}`);
  await rm(lockRoot, { recursive: true, force: true });
}

async function withPinnedCheckoutLock(hostRoot, revision, callback, { timeoutMs = 30_000 } = {}) {
  const lockParent = path.join(hostRoot, ".locks");
  await ensureTerminalPhysicalDirectory(hostRoot, lockParent);
  const lockRoot = pinnedCheckoutLockPath(hostRoot, revision);
  const token = randomUUID();
  const deadline = Date.now() + timeoutMs;
  while (true) {
    try {
      await mkdir(lockRoot);
      await writeFile(path.join(lockRoot, "owner.json"), `${JSON.stringify({ token, pid: process.pid, created_at: new Date().toISOString() })}\n`, { flag: "wx" });
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") {
        await rm(lockRoot, { recursive: true, force: true }).catch(() => {});
        throw error;
      }
      const lockInfo = await lstat(lockRoot).catch((error) => error?.code === "ENOENT" ? null : Promise.reject(error));
      // An owner may release after our mkdir reports contention, before inspection.
      if (!lockInfo) {
        if (Date.now() >= deadline) die(`timed out waiting for exact-pin checkout lock: ${lockRoot}`);
        continue;
      }
      if (!lockInfo.isDirectory() || lockInfo.isSymbolicLink()) die(`checkout lock is not an ordinary directory: ${lockRoot}`);
      const owner = await json(path.join(lockRoot, "owner.json")).catch(() => null);
      const ownerAgeMs = Date.now() - lockInfo.mtimeMs;
      if ((owner && !processIsAlive(owner.pid)) || (!owner && ownerAgeMs > 5_000)) {
        await rm(lockRoot, { recursive: true, force: true });
        continue;
      }
      if (Date.now() >= deadline) die(`timed out waiting for exact-pin checkout lock: ${lockRoot}`);
      await delay(50);
    }
  }
  try {
    return await callback();
  } finally {
    await removeOwnedLock(lockRoot, token);
  }
}

export async function installPinnedCheckout(source, hostRoot, revision) {
  const roots = await assertTerminalPinPhysicalContainment(hostRoot, revision);
  await ensureTerminalPhysicalDirectory(roots.hostRoot, roots.pinRoot);
  return withPinnedCheckoutLock(roots.hostRoot, revision, async () => {
    await assertTerminalPhysicalContainment(roots.hostRoot, roots.inferenceRoot);
    try {
      const existing = await validatePublishedCheckout(roots.inferenceRoot, revision);
      if (existing) return existing;
    } catch {
      const info = await lstat(roots.inferenceRoot).catch((error) => error?.code === "ENOENT" ? null : Promise.reject(error));
      if (info) {
        if (!info.isDirectory() || info.isSymbolicLink()) die("stale inference checkout root is not an ordinary exact destination");
        await rm(roots.inferenceRoot, { recursive: true, force: true });
      }
    }
    const published = await installCheckout(source, roots.inferenceRoot, revision);
    await assertTerminalPhysicalContainment(roots.hostRoot, roots.inferenceRoot);
    return published;
  });
}

export async function validatePreflightTransport(planPath, transport) {
  const { plan } = await readPlanAndLock(planPath);
  const expected = plan.inference_preflight;
  const observed = {
    revision: transport.revision,
    workflow_run_id: transport.workflowRunId,
    artifact_name: transport.artifactName,
  };
  const accepted = {
    revision: plan.inference_contract.revision,
    workflow_run_id: expected.workflow_run_id,
    artifact_name: expected.artifact.name,
  };
  if (JSON.stringify(observed) !== JSON.stringify(accepted)) die("preflight transport does not equal the sealed terminal plan");
  return { ...accepted, workflow_run_attempt: expected.workflow_run_attempt, artifact_id: expected.artifact.id, artifact_digest: expected.artifact.digest };
}

export async function validatePreflightMetadata(planPath, artifact, run) {
  const { plan } = await readPlanAndLock(planPath);
  const expected = plan.inference_preflight;
  const expectedRepository = expected.repository;
  const observedArtifact = {
    id: artifact?.id,
    name: artifact?.name,
    size_in_bytes: artifact?.size_in_bytes,
    digest: artifact?.digest,
    expired: artifact?.expired,
    workflow_run_id: String(artifact?.workflow_run?.id ?? ""),
    workflow_run_head_sha: artifact?.workflow_run?.head_sha,
    workflow_run_repository_id: artifact?.workflow_run?.repository_id,
    workflow_run_head_repository_id: artifact?.workflow_run?.head_repository_id,
  };
  const acceptedArtifact = {
    id: expected.artifact.id,
    name: expected.artifact.name,
    size_in_bytes: expected.artifact.size_in_bytes,
    digest: expected.artifact.digest,
    expired: false,
    workflow_run_id: expected.workflow_run_id,
    workflow_run_head_sha: expected.head_sha,
    workflow_run_repository_id: run?.repository?.id,
    workflow_run_head_repository_id: run?.head_repository?.id,
  };
  const observedRun = {
    id: String(run?.id ?? ""),
    run_attempt: run?.run_attempt,
    head_sha: run?.head_sha,
    workflow_id: run?.workflow_id,
    name: run?.name,
    path: run?.path,
    event: run?.event,
    status: run?.status,
    conclusion: run?.conclusion,
    repository: run?.repository?.full_name,
    head_repository: run?.head_repository?.full_name,
  };
  const acceptedRun = {
    id: expected.workflow_run_id,
    run_attempt: expected.workflow_run_attempt,
    head_sha: expected.head_sha,
    workflow_id: expected.workflow.id,
    name: expected.workflow.name,
    path: expected.workflow.path,
    event: expected.workflow.event,
    status: "completed",
    conclusion: "success",
    repository: expectedRepository,
    head_repository: expectedRepository,
  };
  if (expectedRepository !== plan.inference_contract.repository || JSON.stringify(observedArtifact) !== JSON.stringify(acceptedArtifact) || JSON.stringify(observedRun) !== JSON.stringify(acceptedRun)) {
    die("live preflight artifact and workflow metadata do not equal the sealed terminal plan");
  }
  return { artifact: acceptedArtifact, run: acceptedRun };
}

export function validateSealedPreflightIndex(observed, expected) {
  const sealedIndex = expected && {
    workflow_run_id: expected.workflow_run_id,
    workflow_run_attempt: expected.workflow_run_attempt,
    head_sha: expected.head_sha,
    inventory_artifacts: expected.inventory_artifacts,
    hook_logs: expected.hook_logs,
  };
  if (!sealedIndex || JSON.stringify(observed) !== JSON.stringify(sealedIndex)) die("preflight index does not equal the sealed terminal plan provenance");
  return observed;
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

export async function assemblePinnedPreflight(source, hostRoot, planPath) {
  const { plan } = await readPlanAndLock(planPath);
  const expected = plan.inference_preflight;
  const observed = await json(path.join(source, "starvector-terminal-preflight.json"));
  validateSealedPreflightIndex(observed, expected);
  const roots = await assertTerminalPinPhysicalContainment(hostRoot, plan.inference_contract.revision);
  await ensureTerminalPhysicalDirectory(roots.hostRoot, roots.pinRoot);
  await assertTerminalPhysicalContainment(roots.hostRoot, roots.preflightRoot);
  return assemblePreflight(source, roots.preflightRoot, plan.inference_contract.revision);
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

export async function prepareUpstreamSource(source, lock) {
  if (!await lstat(source).catch(() => null)) {
    await execFile("git", ["clone", "--config", "core.autocrlf=false", "--config", "core.eol=lf", "--no-checkout", `${lock.implementation_repository}.git`, source]);
    await execFile("git", ["-C", source, "checkout", "--detach", lock.implementation_revision]);
  }
  await assertTerminalPhysicalContainment(source, source);
  const head = (await execFile("git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();
  const dirty = (await execFile("git", ["-C", source, "status", "--porcelain"])).stdout.trim();
  if (head !== lock.implementation_revision || dirty) die("upstream source must be exact and clean");
  // Host attributes and smudge filters can override checkout newline settings.
  // Read the audited blobs directly, without checkout conversion, and validate
  // the complete identity before replacing any bytes in this owned checkout.
  const listing = (await execFile("git", ["-C", source, "ls-tree", "-r", "-z", head, "--", "starvector"])).stdout;
  const files = [];
  for (const record of listing.split("\0").filter(Boolean)) {
    const match = /^(\d{6}) (\w+) ([a-f0-9]{40})\t([\s\S]+)$/.exec(record);
    if (!match) die("invalid upstream Git tree entry");
    const [, mode, kind, oid, relative] = match;
    if (!relative.startsWith("starvector/") || !relative.endsWith(".py")) continue;
    if (!/^100(644|755)$/.test(mode) || kind !== "blob" || /[\\:\x00-\x1f\x7f]/.test(relative)
        || relative.split("/").some(part => !part || part === "." || part === "..")) die("upstream Python source must be an ordinary contained Git blob");
    const file = path.join(source, ...relative.split("/"));
    await assertTerminalPhysicalContainment(source, path.dirname(file));
    const info = await lstat(file);
    if (!info.isFile() || info.isSymbolicLink() || info.nlink !== 1) die("upstream Python source must be an ordinary unlinked file");
    const bytes = (await execFile("git", ["-C", source, "cat-file", "blob", oid], { encoding: "buffer", maxBuffer: 4 * 1024 * 1024 })).stdout;
    files.push({ path: relative, file, bytes, mode, oid, sha256: sha(bytes) });
  }
  files.sort((left, right) => Buffer.compare(Buffer.from(left.path), Buffer.from(right.path)));
  const identity = entries => sha(JSON.stringify(entries.map(({ path, sha256 }) => ({ path, sha256 }))));
  const observed = identity(files);
  if (!files.length || observed !== lock.python_source_sha256) die(`upstream Git blob identity mismatch: expected=${lock.python_source_sha256} actual=${observed} files=${files.length}`);
  await execFile("git", ["-C", source, "config", "core.autocrlf", "false"]);
  await execFile("git", ["-C", source, "config", "core.eol", "lf"]);
  // Keep later cleanliness checks meaningful under host conversion policies too.
  // info/attributes is clone-local and takes precedence over global/repo rules;
  // retain existing rules and override only the audited Python paths.
  const infoRoot = path.join(source, ".git", "info");
  await assertTerminalPhysicalContainment(source, infoRoot);
  await mkdir(infoRoot, { recursive: true });
  const attributesPath = path.join(infoRoot, "attributes");
  const attributesInfo = await lstat(attributesPath).catch(error => { if (error.code === "ENOENT") return null; throw error; });
  if (attributesInfo && (!attributesInfo.isFile() || attributesInfo.isSymbolicLink() || attributesInfo.nlink !== 1)) die("upstream local attributes must be an ordinary unlinked file");
  const previous = attributesInfo ? await readFile(attributesPath, "utf8") : "";
  const rules = files.map(entry => `${JSON.stringify(`/${entry.path}`)} -text -filter -ident`).join("\n") + "\n";
  if (!previous.endsWith(rules)) await writeFile(attributesPath, `${previous}${previous.endsWith("\n") || !previous ? "" : "\n"}${rules}`);
  for (const { file, bytes } of files) await writeFile(file, bytes);
  const materialized = await Promise.all(files.map(async entry => ({ path: entry.path, sha256: sha(await readFile(entry.file)) })));
  if (identity(materialized) !== observed) die("materialized upstream source differs from audited Git blobs");
  // Discard conversion-dependent cached stat data without changing index blobs.
  for (const { mode, oid, path: relative } of files) await execFile("git", ["-C", source, "update-index", "--cacheinfo", `${mode},${oid},${relative}`]);
  if ((await execFile("git", ["-C", source, "status", "--porcelain"])).stdout.trim()) die("materialized upstream source must remain clean");
  return source;
}

// Never forward pip output verbatim: index/proxy configuration may contain
// credentials. The raw renderer supplies byte counts without URLs or paths.
export function upstreamPipProgress(line) {
  const progress = /^Progress ([0-9]{1,16}) of ([0-9]{1,16})$/.exec(line.trim());
  if (progress) {
    const current = Number(progress[1]), total = Number(progress[2]);
    if (Number.isSafeInteger(current) && Number.isSafeInteger(total) && (!total || current <= total)) return { event: "download", bytes: current, total_bytes: total };
  }
  if (/^Installing collected packages: /.test(line)) return { event: "installing" };
  if (/ReadTimeoutError|The read operation timed out/.test(line)) return { event: "socket_timeout" };
  if (/^ERROR: No matching distribution found for /.test(line)) return { event: "no_matching_distribution" };
  if (/^ERROR: .*No space left on device/.test(line)) return { event: "disk_full" };
  const wheel = /^\s*(Downloading|Using cached) (\S+)/.exec(line);
  if (wheel) {
    try {
      const url = new URL(wheel[2]);
      if (url.protocol !== "https:" || url.username || url.password || !["download.pytorch.org", "download-r2.pytorch.org", "files.pythonhosted.org"].includes(url.hostname)) return null;
      // Only identify our two locked CUDA wheels; other dependencies still report
      // numeric progress without reflecting arbitrary package names or paths.
      const name = decodeURIComponent(url.pathname.split("/").at(-1));
      const packageName = /^torch-2\.7\.1\+cu128-/.test(name) ? "torch" : /^torchvision-0\.22\.1\+cu128-/.test(name) ? "torchvision" : null;
      return { event: wheel[1] === "Downloading" ? "download_started" : "cached_download", ...(packageName ? { package: packageName } : {}) };
    } catch { return null; }
  }
  return null;
}

export async function runUpstreamPip(python, args, bounds, { emit = console.log, heartbeatMs = 30_000, noProgressMs = 300_000 } = {}) {
  const start = Date.now();
  let latest = null, transfer = null, lastProgressAt = start, installing = false, stopReason = null;
  const report = event => emit(JSON.stringify({ kind: "upstream-package-install", elapsed_seconds: Math.floor((Date.now() - start) / 1000), ...event }));
  report({ event: "started", timeout_seconds: bounds.timeout / 1000, no_progress_timeout_seconds: noProgressMs / 1000 });
  return new Promise((resolve, reject) => {
    const partial = { stdout: "", stderr: "" };
    const child = execFileCallback(python, args, bounds, (error) => {
      clearInterval(heartbeat);
      for (const line of Object.values(partial)) if (line) accept(line);
      const failed = error || stopReason;
      const outcome = { event: failed ? "failed" : "completed", ...(latest ? { last_progress: latest } : {}), ...(transfer ? { last_transfer: transfer } : {}) };
      if (failed) {
        outcome.killed = error?.killed === true;
        outcome.signal = ["SIGTERM", "SIGKILL"].includes(error?.signal) ? error.signal : null;
        outcome.exit_code = Number.isInteger(error?.code) ? error.code : null;
        outcome.failure = stopReason ?? (error.killed && Date.now() - start >= bounds.timeout ? "process_deadline" : error.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" ? "output_limit" : "process_exit");
      }
      report(outcome);
      if (failed) reject(new Error(`upstream package acquisition failed: ${JSON.stringify(outcome)}`));
      else resolve();
    });
    const accept = line => {
      const event = upstreamPipProgress(line);
      if (!event) return;
      latest = event;
      if (event.event === "installing") installing = true;
      if (event.event === "download") {
        // Repeated counts do not prove progress. A completed transfer followed by
        // a new file starts a new bounded observation window.
        if (!transfer || event.bytes > transfer.bytes || (transfer.total_bytes && transfer.bytes === transfer.total_bytes && event.bytes < transfer.bytes)) {
          lastProgressAt = Date.now();
          transfer = event;
        }
      } else if (event.event === "download_started" || event.event === "cached_download") {
        lastProgressAt = Date.now();
        transfer = null;
      }
      if (event.event !== "download") report(event);
    };
    const consume = stream => chunk => {
      partial[stream] += chunk.toString();
      const lines = partial[stream].split(/\r?\n/); partial[stream] = lines.pop();
      for (const line of lines) accept(line);
      // Discard oversized/unrecognized lines rather than accumulating secrets.
      if (partial[stream].length > 8192) partial[stream] = "";
    };
    child.stdout.on("data", consume("stdout"));
    child.stderr.on("data", consume("stderr"));
    const heartbeat = setInterval(() => {
      const silence = Date.now() - lastProgressAt;
      report({ event: "waiting", phase: installing ? "installing" : "acquiring", seconds_since_progress: Math.floor(silence / 1000), ...(latest ? { last_progress: latest } : {}) });
      if (!installing && silence >= noProgressMs && !stopReason) {
        stopReason = transfer ? "download_stalled" : "no_observable_download_progress";
        child.kill();
      }
    }, Math.min(heartbeatMs, noProgressMs));
  });
}

export async function installUpstreamPackages(python, lock, execute = runUpstreamPip) {
  if (lock.torch_index_url !== "https://download.pytorch.org/whl/cu128") die("oracle requires the locked CUDA 12.8 wheel index");
  // pip's timeout is a socket timeout, independent of the bounded process time.
  // Do not restart a failed multi-gigabyte install or discard its reusable cache.
  const base = ["-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "raw", "--timeout", "120", "--retries", "3"];
  const bounds = { timeout: 60 * 60 * 1000, maxBuffer: 4 * 1024 * 1024 };
  await execute(python, [...base, "--index-url", lock.torch_index_url, ...["torch", "torchvision"].map(name => `${name}==${lock.required_packages[name]}`)], bounds);
  await execute(python, [...base, ...Object.entries(lock.required_packages).map(([name, version]) => `${name}==${version}`)], bounds);
}

// Provisioning may acquire dependencies; the campaign validation/execution is offline.
export async function provisionUpstream({ sceneWorksRoot, hostRoot, python, assetsRoot, sanitizer }) {
  const lock = await json(path.join(sceneWorksRoot, "release/starvector-terminal-upstream-lock-v1.json"));
  const source = path.join(hostRoot, "upstream-source"), environment = path.join(hostRoot, "upstream-env");
  await prepareUpstreamSource(source, lock);
  const oraclePython = path.join(environment, process.platform === "win32" ? "Scripts" : "bin", process.platform === "win32" ? "python.exe" : "python");
  if (!await lstat(oraclePython).catch(() => null)) await execFile(python, ["-m", "venv", environment]);
  const pipVersion = (await execFile(oraclePython, ["-c", "import importlib.metadata; print(importlib.metadata.version('pip'))"], { timeout: 30_000 })).stdout.trim();
  if (!/^[0-9]+\.[0-9]+(?:\.[0-9]+)?$/.test(pipVersion)) die("upstream pip version probe returned an invalid version");
  console.log(JSON.stringify({ kind: "upstream-package-installer", pip_version: pipVersion, resume_configured: false }));
  await installUpstreamPackages(oraclePython, lock);
  const { validateUpstreamInputs } = await import("./starvector-terminal-upstream.mjs");
  // Authenticated component configs are provisioned separately with immutable
  // repository/revision/hash metadata; never synthesize missing backbone defaults.
  return validateUpstreamInputs({ sceneWorksRoot, python: oraclePython, upstreamRoot: source, weightsRoot: path.join(hostRoot, "weights"), assetsRoot, componentsRoot: path.join(hostRoot, "upstream-components"), sanitizer }, path.join(hostRoot, "upstream-validation"));
}

async function main(argv) {
  const [command, ...args] = argv;
  if (command === "upstream" && args.length === 5) return provisionUpstream({ sceneWorksRoot: args[0], hostRoot: args[1], python: args[2], assetsRoot: args[3], sanitizer: args[4] });
  if (command === "download" && args.length === 3) return downloadExact(args[0], args[1], args[2]);
  if (command === "checkout" && args.length === 3) return installPinnedCheckout(args[0], args[1], args[2]);
  if (command === "preflight" && args.length === 3) return assemblePinnedPreflight(args[0], args[1], args[2]);
  if (command === "preflight-transport" && args.length === 4) {
    const accepted = await validatePreflightTransport(args[0], { revision: args[1], workflowRunId: args[2], artifactName: args[3] });
    console.log(accepted.artifact_id);
    return accepted;
  }
  if (command === "preflight-metadata" && args.length === 3) {
    return validatePreflightMetadata(args[0], await json(args[1]), await json(args[2]));
  }
  if (command === "weights" && args.length === 6) return assembleWeights({ hostRoot: args[0], serviceAppData: args[1], serviceHfHome: args[2], promptProvider: args[3], promptModel: args[4], promptRevision: args[5] });
  if (command === "metrics" && args.length === 4) return assembleMetrics({ sceneWorksRoot: args[0], metricsRoot: args[1], python: args[2], clipRevision: args[3] });
  die("usage: download <url> <path> <sha256> | checkout <source> <host-root> <sha> | preflight <source> <host-root> <plan> | preflight-transport <plan> <sha> <run-id> <artifact-name> | preflight-metadata <plan> <artifact-json> <run-json> | weights <host-root> <app-data-source> <hf-source> <provider> <model> <revision> | metrics <sceneworks-root> <metrics-root> <python> <clip-revision>");
}

if (isExecutedModule(import.meta.url)) main(process.argv.slice(2)).catch((error) => { console.error(error.message); process.exitCode = 1; });
