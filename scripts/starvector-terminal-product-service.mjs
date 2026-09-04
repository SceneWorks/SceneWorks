#!/usr/bin/env node
// Start the actual current-tree Rust API and worker for the terminal campaign.
// This is intentionally not an arbitrary service command: its source revision,
// Cargo inference pin, binaries and health response are all recorded together.
import { createHash, randomBytes } from "node:crypto";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { copyFile, lstat, mkdir, open, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { promisify } from "node:util";
import path from "node:path";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
import { fileSha256 } from "./lib/file-sha256.mjs";
import { sortTerminalTreeEntries, terminalTreeEntry, terminalTreeSha256 } from "./lib/terminal-tree-identity.mjs";

const execFile = promisify(execFileCallback);
const sha = (value) => createHash("sha256").update(value).digest("hex");
const die = (value) => { throw new Error(`starvector terminal service: ${value}`); };
export function productServiceBuildArgs(platform = process.platform) {
  const args = ["build", "--locked", "-p", "sceneworks-rust-api"];
  if (platform === "win32") args.push("--features", "backend-candle");
  return args;
}
export function productServiceBackendEnv(platform = process.platform) {
  return platform === "win32" ? { SCENEWORKS_BACKEND_CANDLE_ENABLED: "true" } : {};
}
export function productServiceStateRoot(output) {
  return path.join(path.dirname(output), `${path.basename(output)}-product-service-state`);
}
export function productServiceActiveStatePath(output) {
  return path.join(productServiceStateRoot(output), "product-service-active.json");
}
const PRODUCT_SERVICE_LOG_FILES = Object.freeze({
  api_stdout: "product-service-api.stdout.log",
  api_stderr: "product-service-api.stderr.log",
  worker_stdout: "product-service-worker.stdout.log",
  worker_stderr: "product-service-worker.stderr.log",
});
export function productServiceLogPaths(output) {
  return Object.fromEntries(Object.entries(PRODUCT_SERVICE_LOG_FILES).map(([name, file]) => [name, path.join(output, file)]));
}
async function openProductServiceLogs(output) {
  const handles = {};
  try {
    for (const [name, file] of Object.entries(productServiceLogPaths(output))) handles[name] = await open(file, "wx", 0o600);
    return handles;
  } catch (error) {
    await Promise.allSettled(Object.values(handles).map((handle) => handle.close()));
    throw error;
  }
}
async function fixedPrefixSha256(file, byteSize) {
  if (byteSize === 0) return sha("");
  return fileSha256(file, { openReadStream: (input, options) => createReadStream(input, { ...options, start: 0, end: byteSize - 1 }) });
}
export async function productServiceLogsIdentity(output, { requireAll = true } = {}) {
  const entries = [];
  for (const file of Object.values(productServiceLogPaths(output))) {
    const info = await lstat(file).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error));
    if (!info) {
      if (requireAll) die(`product service log is missing: ${file}`);
      continue;
    }
    if (info.isSymbolicLink() || !info.isFile()) {
      if (requireAll) die(`product service log is not a regular file: ${file}`);
      continue;
    }
    const relative = path.relative(output, file).split(path.sep).join("/");
    entries.push(terminalTreeEntry(relative, info.size, await fixedPrefixSha256(file, info.size)));
  }
  const canonicalEntries = sortTerminalTreeEntries(entries);
  return { schema_version: 1, entries: canonicalEntries, sha256: terminalTreeSha256(canonicalEntries) };
}
async function writeProductServiceLogsIdentity(output, options) {
  const identity = await productServiceLogsIdentity(output, options);
  await writeFile(path.join(output, "product-service-logs.json"), JSON.stringify(identity, null, 2) + "\n");
  await writeFile(path.join(output, "product-service-logs.sha256"), `${identity.sha256}\n`);
  return identity;
}
async function git(root, args) { return (await execFile("git", ["-C", root, ...args])).stdout.trim(); }
async function serviceIdentity(root, permanentPin) {
  if (await git(root, ["status", "--porcelain"])) die("current-tree product service checkout must be clean");
  const revision = await git(root, ["rev-parse", "HEAD"]), cargo = await readFile(path.join(root, "Cargo.toml"), "utf8");
  const pin = cargo.match(/SceneWorks\/inference",\s*rev\s*=\s*"([a-f0-9]{40})"/)?.[1];
  if (!pin || pin !== permanentPin) die("current-tree product service Cargo inference pin mismatches permanent pin");
  return { sceneworks_revision: revision, inference_revision: pin };
}
async function waitHealthy(url, assertRunning = () => {}) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    assertRunning();
    try { const response = await fetch(new URL("/api/v1/health", url), { signal: AbortSignal.timeout(1000) }); const body = await response.json(); if (response.ok && body?.status === "ok" && body?.readiness?.status === "ready") { assertRunning(); return body; } } catch { /* booting */ }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  die("source-built API did not become ready");
}
export async function relocateProductServiceLibrary(url, hfHome, {
  fetchImpl = fetch,
  timeoutMs = 120_000,
} = {}) {
  if (typeof hfHome !== "string" || !path.isAbsolute(hfHome)) die("offline HF home must be an absolute path");
  const expectedHfHome = path.resolve(hfHome), expectedLibraryRoot = path.join(expectedHfHome, "hub");
  let response;
  let body;
  let probeResponse;
  let probe;
  try {
    const signal = AbortSignal.timeout(timeoutMs);
    response = await fetchImpl(new URL("/api/v1/model-library/relocate", url), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path: expectedHfHome }),
      signal,
    });
    body = await response.json();
    if (response.ok) {
      probeResponse = await fetchImpl(new URL("/api/v1/model-library", url), { signal });
      probe = await probeResponse.json();
    }
  } catch (error) {
    die(`offline model library relocation request failed: ${error.message}`);
  }
  if (!response.ok) die(`offline model library relocation HTTP ${response.status}: ${JSON.stringify(body)}`);
  if (body?.adopted !== true || typeof body.hfHome !== "string" || typeof body.libraryRoot !== "string" || !path.isAbsolute(body.hfHome) || !path.isAbsolute(body.libraryRoot) || path.relative(expectedHfHome, path.resolve(body.hfHome)) !== "" || path.relative(expectedLibraryRoot, path.resolve(body.libraryRoot)) !== "") {
    die("offline model library relocation returned an inexact binding");
  }
  if (!probeResponse.ok || probe?.available !== true || probe.probeStatus !== "available" || typeof probe.configuredLibraryPath !== "string" || path.relative(expectedLibraryRoot, path.resolve(probe.configuredLibraryPath)) !== "" || typeof probe.expectedLibrary?.configuredPath !== "string" || path.relative(expectedLibraryRoot, path.resolve(probe.expectedLibrary.configuredPath)) !== "") {
    die("offline model library relocation did not read back as the exact available binding");
  }
  return { adopted: true, hf_home: expectedHfHome, library_root: expectedLibraryRoot, probe_status: probe.probeStatus };
}
export async function assertProductServicePortFree(url, { serverFactory = createServer } = {}) {
  const parsed = new URL(url), port = Number(parsed.port);
  if (parsed.hostname !== "127.0.0.1" || !Number.isInteger(port) || port < 1 || port > 65535) die("product service must bind an explicit loopback host/port");
  await new Promise((resolve, reject) => {
    const server = serverFactory();
    server.unref();
    server.once("error", (error) => reject(new Error(`starvector terminal service: API port is already occupied: ${error.message}`)));
    server.listen({ host: parsed.hostname, port, exclusive: true }, () => server.close((error) => error ? reject(error) : resolve()));
  });
  return { host: parsed.hostname, port };
}
export function productServiceTaskkillArguments(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) die("invalid product service PID");
  return ["/PID", String(pid), "/T", "/F"];
}
async function runProductServiceTaskkill(pid, timeoutMs) {
  await execFile("taskkill.exe", productServiceTaskkillArguments(pid), { timeout: timeoutMs, windowsHide: true });
}
async function pidIsRunning(pid, kill) {
  try { kill(pid, 0); return true; } catch (error) { if (error.code === "ESRCH") return false; if (error.code === "EPERM") return true; throw error; }
}
export async function terminateProductServicePids(pids, {
  requireAll = true,
  platform = process.platform,
  kill = process.kill,
  taskkill = runProductServiceTaskkill,
  wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
  timeoutMs = 5000,
  pollMs = 100,
} = {}) {
  const validPids = pids.filter((pid) => Number.isInteger(pid) && pid > 0);
  if (requireAll && validPids.length !== pids.length) die("invalid product service PID");
  const live = async () => {
    const result = [];
    for (const pid of validPids) if (await pidIsRunning(pid, kill)) result.push(pid);
    return result;
  };
  let remaining = await live();
  if (platform === "win32") {
    for (const pid of remaining) {
      try { await taskkill(pid, timeoutMs); } catch (error) { if (await pidIsRunning(pid, kill)) throw error; }
    }
  } else {
    for (const pid of remaining) { try { kill(pid, "SIGTERM"); } catch (error) { if (error.code !== "ESRCH") throw error; } }
  }
  const attempts = Math.max(1, Math.ceil(timeoutMs / pollMs));
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    remaining = await live();
    if (remaining.length === 0) return;
    await wait(pollMs);
  }
  if (platform !== "win32") {
    for (const pid of remaining) { try { kill(pid, "SIGKILL"); } catch (error) { if (error.code !== "ESRCH") throw error; } }
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      remaining = await live();
      if (remaining.length === 0) return;
      await wait(pollMs);
    }
  }
  die(`product service did not stop (pids: ${remaining.join(", ")})`);
}
export async function copyRegularTree(source, destination, digestFile = fileSha256) {
  const info = await lstat(source);
  if (info.isSymbolicLink()) die(`weights closure rejects symlink ${source}`);
  if (info.isDirectory()) {
    await mkdir(destination, { recursive: true });
    for (const name of await readdir(source)) await copyRegularTree(path.join(source, name), path.join(destination, name), digestFile);
    return;
  }
  if (!info.isFile()) die(`weights closure rejects non-regular file ${source}`);
  await mkdir(path.dirname(destination), { recursive: true });
  await copyFile(source, destination);
  if (await digestFile(source) !== await digestFile(destination)) die(`weights closure copy hash drifted: ${source}`);
}

export async function closureTreeHash(root, digestFile = fileSha256) {
  const rows = [];
  for (const name of await readdir(root, { recursive: true })) {
    const file = path.join(root, name), info = await lstat(file);
    if (info.isSymbolicLink()) die(`weights closure copy contains symlink ${name}`);
    if (info.isFile()) rows.push(terminalTreeEntry(name.split(path.sep).join("/"), info.size, await digestFile(file)));
  }
  const canonicalRows = sortTerminalTreeEntries(rows);
  return terminalTreeSha256(canonicalRows);
}

async function materializeOfflineWeights(weightsRoot, stateRoot) {
  const manifest = JSON.parse(await readFile(path.join(weightsRoot, "starvector-terminal-weights-v1.json"), "utf8"));
  const closure = manifest.terminal_service_closure;
  if (!closure || typeof closure.app_data_relative_path !== "string" || typeof closure.hf_home_relative_path !== "string" || !/^[a-f0-9]{64}$/.test(closure.app_data_sha256) || !/^[a-f0-9]{64}$/.test(closure.hf_home_sha256)) die("weights manifest lacks exact app receipt/HF offline closure");
  for (const relative of [closure.app_data_relative_path, closure.hf_home_relative_path]) if (path.isAbsolute(relative) || relative.split(/[\\/]/).includes("..")) die("weights closure path is unsafe");
  const appSource = path.join(weightsRoot, ...closure.app_data_relative_path.split(/[\\/]/));
  const hfSource = path.join(weightsRoot, ...closure.hf_home_relative_path.split(/[\\/]/));
  await copyRegularTree(appSource, path.join(stateRoot, "data"));
  await copyRegularTree(hfSource, path.join(stateRoot, "hf"));
  if (await closureTreeHash(path.join(stateRoot, "data")) !== closure.app_data_sha256 || await closureTreeHash(path.join(stateRoot, "hf")) !== closure.hf_home_sha256) die("materialized receipt/HF closure hash mismatch");
  const models = {};
  for (const [key, model] of Object.entries(manifest.models ?? {})) {
    if (!model?.relative_path || !model?.inventory_sha256) die("weights manifest model inventory is incomplete");
    // This is the same manifest validated by the controller before execution;
    // retain only the measured snapshot identity, never a corpus-supplied run
    // identity, in the source-built service record.
    models[key] = { revision: model.revision, inventory_sha256: model.inventory_sha256 };
  }
  if (Object.keys(models).length !== 2) die("weights manifest must close both StarVector snapshots");
  return { app_data_sha256: closure.app_data_sha256, hf_home_sha256: closure.hf_home_sha256, models };
}

const ACTIVE_STATE_KEYS = [
  "api_binary_sha256",
  "api_pid",
  "api_url",
  "inference_revision",
  "instance_token",
  "kind",
  "schema_version",
  "sceneworks_revision",
  "started_at",
  "state_root",
  "worker_pid",
].sort();
const ACTIVE_BINDING_KEYS = ACTIVE_STATE_KEYS.filter((key) => !["kind", "schema_version"].includes(key));

export function validateProductServiceActiveState(active, record, output) {
  if (!active || typeof active !== "object" || Array.isArray(active) || JSON.stringify(Object.keys(active).sort()) !== JSON.stringify(ACTIVE_STATE_KEYS)) die("active product service state shape is invalid");
  if (active.schema_version !== 1 || active.kind !== "starvector_terminal_product_service_active" || !/^[a-f0-9]{64}$/.test(active.instance_token)) die("active product service sentinel is invalid");
  if (!/^[a-f0-9]{40}$/.test(active.sceneworks_revision) || !/^[a-f0-9]{40}$/.test(active.inference_revision) || !/^[a-f0-9]{64}$/.test(active.api_binary_sha256)) die("active product service identity is invalid");
  if (!Number.isSafeInteger(active.api_pid) || active.api_pid <= 0 || !Number.isSafeInteger(active.worker_pid) || active.worker_pid <= 0 || active.api_pid === active.worker_pid) die("active product service PIDs are invalid");
  if (active.state_root !== path.relative(output, productServiceStateRoot(output)) || !active.api_url?.startsWith("http://127.0.0.1:")) die("active product service location is invalid");
  for (const key of ACTIVE_BINDING_KEYS) if (active[key] !== record?.[key]) die(`active product service state mismatches provenance field ${key}`);
  return active;
}

export async function startProductService({ root, output, permanentPin, url, weightsRoot }) {
  if (!root || !output || !weightsRoot || !permanentPin || !url || !url.startsWith("http://127.0.0.1:")) die("local current-tree root/output/offline weights/pin/API URL required");
  const identity = await serviceIdentity(root, permanentPin);
  const endpoint = await assertProductServicePortFree(url);
  await mkdir(output, { recursive: true });
  // `cargo build` is source ownership: no prebuilt or /opt sidecar can satisfy
  // this contract.  It never downloads model weights; the controller separately
  // rejects any model acquisition at job time.
  await execFile("cargo", productServiceBuildArgs(), { cwd: root });
  const binary = path.join(root, "target", "debug", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api");
  const common = { cwd: root, detached: true }, stateRoot = productServiceStateRoot(output);
  if (await lstat(stateRoot).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error))) die("temporary product service state already exists");
  const logPaths = productServiceLogPaths(output), instanceToken = randomBytes(32).toString("hex");
  let stateCreated = false;
  let logHandles = {};
  let api;
  let worker;
  let activeState;
  const spawnErrors = [];
  try {
    await mkdir(stateRoot);
    stateCreated = true;
    await mkdir(path.join(stateRoot, "data"), { recursive: true }); await mkdir(path.join(stateRoot, "config"), { recursive: true });
    const weights = await materializeOfflineWeights(weightsRoot, stateRoot), hfHome = path.join(stateRoot, "hf"), binarySha256 = await fileSha256(binary);
    const serviceEnv = { ...process.env, ...productServiceBackendEnv(), SCENEWORKS_TERMINAL_CAMPAIGN: "1", SCENEWORKS_TERMINAL_SERVICE_INSTANCE_TOKEN: instanceToken, SCENEWORKS_API_HOST: endpoint.host, SCENEWORKS_API_PORT: String(endpoint.port), SCENEWORKS_API_URL: url, SCENEWORKS_DATA_DIR: path.join(stateRoot, "data"), SCENEWORKS_CONFIG_DIR: path.join(stateRoot, "config"), SCENEWORKS_JOBS_DB_PATH: path.join(stateRoot, "data", "cache", "jobs.db"), SCENEWORKS_GPU_ID: process.env.STARVECTOR_TERMINAL_GPU_ID ?? "auto", HF_HOME: hfHome, HUGGINGFACE_HUB_CACHE: path.join(hfHome, "hub"), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" };
    for (const inherited of ["TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_ENDPOINT"]) delete serviceEnv[inherited];
    // Piped readable streams keep this CLI referenced after child.unref(). Give
    // each service durable files instead; the children retain their inherited
    // descriptors after these parent FileHandles close and after this CLI exits.
    logHandles = await openProductServiceLogs(output);
    try {
      api = spawn(binary, [], { ...common, env: serviceEnv, stdio: ["ignore", logHandles.api_stdout.fd, logHandles.api_stderr.fd] });
      api.once("error", (error) => spawnErrors.push(`API: ${error.message}`));
      worker = spawn(binary, [], { ...common, env: { ...serviceEnv, SCENEWORKS_WORKER_ONLY: "1" }, stdio: ["ignore", logHandles.worker_stdout.fd, logHandles.worker_stderr.fd] });
      worker.once("error", (error) => spawnErrors.push(`worker: ${error.message}`));
    } finally {
      await Promise.allSettled(Object.values(logHandles).map((handle) => handle.close()));
      logHandles = {};
    }
    if (api.pid === undefined || worker.pid === undefined) die("source-built API or worker has no process identity");
    const startedAt = new Date().toISOString();
    activeState = { api_binary_sha256: binarySha256, api_pid: api.pid, api_url: url, inference_revision: identity.inference_revision, instance_token: instanceToken, kind: "starvector_terminal_product_service_active", schema_version: 1, sceneworks_revision: identity.sceneworks_revision, started_at: startedAt, state_root: path.relative(output, stateRoot), worker_pid: worker.pid };
    await writeFile(productServiceActiveStatePath(output), JSON.stringify(activeState, null, 2) + "\n", { flag: "wx", mode: 0o600 });
    for (const child of [api, worker]) child.unref();
    const assertRunning = () => {
      if (spawnErrors.length || api.pid === undefined || worker.pid === undefined || api.exitCode !== null || worker.exitCode !== null || api.signalCode !== null || worker.signalCode !== null) die(`source-built API or worker exited during readiness${spawnErrors.length ? `: ${spawnErrors.join("; ")}` : ""}`);
    };
    const health = await waitHealthy(url, assertRunning);
    // The app-data closure deliberately carries the source machine's physical-library binding.
    // Rebind the already hash-verified offline copy through the product relocation seam: it proves
    // every receipted/validated model at the new root and atomically captures its path + volume
    // identity without rewriting receipts or downloading anything.
    const relocation = await relocateProductServiceLibrary(url, hfHome);
    assertRunning();
    const record = { ...identity, ...weights, instance_token: instanceToken, api_url: url, api_host: endpoint.host, api_port: endpoint.port, state_root: path.relative(output, stateRoot), api_binary: path.relative(root, binary), worker_binary: path.relative(root, binary), api_binary_sha256: binarySha256, api_pid: api.pid, worker_pid: worker.pid, logs: Object.fromEntries(Object.entries(logPaths).map(([name, file]) => [name, path.relative(output, file)])), health, offline: { hf_home: path.relative(output, hfHome), hf_hub_offline: serviceEnv.HF_HUB_OFFLINE, transformers_offline: serviceEnv.TRANSFORMERS_OFFLINE, library_relocation: { adopted: relocation.adopted, hf_home: path.relative(output, relocation.hf_home), library_root: path.relative(output, relocation.library_root), probe_status: relocation.probe_status } }, started_at: startedAt };
    await writeFile(path.join(output, "product-service-provenance.json"), JSON.stringify(record, null, 2) + "\n");
    return record;
  } catch (error) {
    await Promise.allSettled(Object.values(logHandles).map((handle) => handle.close()));
    const knownPids = [worker?.pid, api?.pid].filter((pid) => Number.isSafeInteger(pid) && pid > 0);
    const cleanup = { status: knownPids.length ? "terminated" : "not_needed", error: null, state_retained: false };
    try { await terminateProductServicePids(knownPids, { requireAll: false }); } catch (cleanupError) { cleanup.status = "failed"; cleanup.error = cleanupError.message; cleanup.state_retained = true; error.message += `; service cleanup failed: ${cleanupError.message}`; }
    let recorded = false;
    try {
      const logs = await writeProductServiceLogsIdentity(output, { requireAll: false });
      await writeFile(path.join(output, "product-service-start-failed.json"), JSON.stringify({ schema_version: 1, status: "failed", api_pid: api?.pid, worker_pid: worker?.pid, active_state: activeState ?? null, cleanup, logs, error: error.message, failed_at: new Date().toISOString() }, null, 2) + "\n");
      recorded = true;
    } catch (recordError) {
      cleanup.state_retained = true;
      error.message += `; failure record could not be written: ${recordError.message}`;
    }
    if (stateCreated && cleanup.status !== "failed" && recorded) await rm(stateRoot, { recursive: true, force: true });
    throw error;
  }
}
export async function stopProductService(output, { terminate = terminateProductServicePids } = {}) {
  if (await lstat(path.join(output, "product-service-stopped.json")).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error))) die("product service is already stopped");
  let record;
  try {
    record = JSON.parse(await readFile(path.join(output, "product-service-provenance.json"), "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const failure = JSON.parse(await readFile(path.join(output, "product-service-start-failed.json"), "utf8"));
    if (failure.cleanup?.status !== "failed" || !failure.active_state) die("failed product service start has no active instance to stop");
    record = failure.active_state;
  }
  const stateRoot = path.resolve(output, record.state_root);
  if (stateRoot !== path.resolve(productServiceStateRoot(output))) die("product service state path escaped its tuple temporary root");
  const stateInfo = await lstat(stateRoot);
  if (stateInfo.isSymbolicLink() || !stateInfo.isDirectory()) die("product service state root is not an owned directory");
  const activePath = productServiceActiveStatePath(output), activeInfo = await lstat(activePath);
  if (activeInfo.isSymbolicLink() || !activeInfo.isFile()) die("active product service sentinel is not a regular file");
  const active = validateProductServiceActiveState(JSON.parse(await readFile(activePath, "utf8")), record, output);
  try {
    await terminate([active.worker_pid, active.api_pid]);
    await assertProductServicePortFree(active.api_url);
    const logs = await writeProductServiceLogsIdentity(output);
    await writeFile(path.join(output, "product-service-stopped.json"), JSON.stringify({ schema_version: 1, status: "stopped", instance_token: active.instance_token, api_pid: active.api_pid, worker_pid: active.worker_pid, logs, stopped_at: new Date().toISOString() }, null, 2) + "\n");
    await rm(stateRoot, { recursive: true });
  } catch (error) {
    const logs = await writeProductServiceLogsIdentity(output, { requireAll: false });
    await writeFile(path.join(output, "product-service-stop-failed.json"), JSON.stringify({ schema_version: 1, status: "failed", instance_token: active.instance_token, api_pid: active.api_pid, worker_pid: active.worker_pid, cleanup: { status: "failed", error: error.message, state_retained: true }, logs, error: error.message, failed_at: new Date().toISOString() }, null, 2) + "\n");
    throw error;
  }
}
if (isExecutedModule(import.meta.url)) {
  const [command, ...args] = process.argv.slice(2); const run = command === "start" ? startProductService({ root: args[0], output: args[1], permanentPin: args[2], url: args[3], weightsRoot: args[4] }) : command === "stop" ? stopProductService(args[0]) : Promise.reject(new Error("usage: start <root> <output> <pin> <url> <weights-root> | stop <output>")); run.catch((error) => { console.error(error.message); process.exitCode = 1; });
}
