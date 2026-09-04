#!/usr/bin/env node
// Start the actual current-tree Rust API and worker for the terminal campaign.
// This is intentionally not an arbitrary service command: its source revision,
// Cargo inference pin, binaries and health response are all recorded together.
import { createHash } from "node:crypto";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { copyFile, lstat, mkdir, open, readFile, readdir, rm, writeFile } from "node:fs/promises";
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
export async function productServiceLogsSha256(output) {
  const chunks = await Promise.all(Object.values(productServiceLogPaths(output)).map((file) => readFile(file)));
  return sha(Buffer.concat(chunks));
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
    try { const response = await fetch(new URL("/api/v1/health", url)); const body = await response.json(); if (response.ok && body?.status === "ok" && body?.readiness?.status === "ready") { assertRunning(); return body; } } catch { /* booting */ }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  die("source-built API did not become ready");
}
async function terminateProductServicePids(pids, { requireAll = true } = {}) {
  const validPids = pids.filter((pid) => Number.isInteger(pid) && pid > 0);
  if (requireAll && validPids.length !== pids.length) die("invalid product service PID");
  for (const pid of validPids) { try { process.kill(pid, "SIGTERM"); } catch (error) { if (error.code !== "ESRCH") throw error; } }
  for (let attempt = 0; attempt < 50; attempt += 1) {
    let alive = false;
    for (const pid of validPids) {
      try { process.kill(pid, 0); alive = true; } catch (error) { if (error.code !== "ESRCH") throw error; }
    }
    if (!alive) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  die("product service did not stop after SIGTERM");
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

export async function startProductService({ root, output, permanentPin, url, weightsRoot }) {
  if (!root || !output || !weightsRoot || !permanentPin || !url || !url.startsWith("http://127.0.0.1:")) die("local current-tree root/output/offline weights/pin/API URL required");
  const identity = await serviceIdentity(root, permanentPin); await mkdir(output, { recursive: true });
  // `cargo build` is source ownership: no prebuilt or /opt sidecar can satisfy
  // this contract.  It never downloads model weights; the controller separately
  // rejects any model acquisition at job time.
  await execFile("cargo", productServiceBuildArgs(), { cwd: root });
  const binary = path.join(root, "target", "debug", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api");
  const common = { cwd: root, detached: true }, parsed = new URL(url), apiPort = parsed.port, stateRoot = productServiceStateRoot(output);
  if (parsed.hostname !== "127.0.0.1" || !apiPort) die("product service must bind an explicit loopback host/port");
  if (await lstat(stateRoot).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error))) die("temporary product service state already exists");
  await mkdir(path.join(stateRoot, "data"), { recursive: true }); await mkdir(path.join(stateRoot, "config"), { recursive: true });
  const weights = await materializeOfflineWeights(weightsRoot, stateRoot);
  const hfHome = path.join(stateRoot, "hf");
  const serviceEnv = { ...process.env, ...productServiceBackendEnv(), SCENEWORKS_TERMINAL_CAMPAIGN: "1", SCENEWORKS_API_HOST: "127.0.0.1", SCENEWORKS_API_PORT: apiPort, SCENEWORKS_API_URL: url, SCENEWORKS_DATA_DIR: path.join(stateRoot, "data"), SCENEWORKS_CONFIG_DIR: path.join(stateRoot, "config"), SCENEWORKS_JOBS_DB_PATH: path.join(stateRoot, "data", "cache", "jobs.db"), SCENEWORKS_GPU_ID: process.env.STARVECTOR_TERMINAL_GPU_ID ?? "auto", HF_HOME: hfHome, HUGGINGFACE_HUB_CACHE: path.join(hfHome, "hub"), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" };
  for (const inherited of ["TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_ENDPOINT"]) delete serviceEnv[inherited];
  // Piped readable streams keep this CLI referenced after child.unref(). Give
  // each service durable files instead; the children retain their inherited
  // descriptors after these parent FileHandles close and after this CLI exits.
  const logPaths = productServiceLogPaths(output), logHandles = await openProductServiceLogs(output);
  let api;
  let worker;
  const spawnErrors = [];
  try {
    try {
      api = spawn(binary, [], { ...common, env: serviceEnv, stdio: ["ignore", logHandles.api_stdout.fd, logHandles.api_stderr.fd] });
      api.once("error", (error) => spawnErrors.push(`API: ${error.message}`));
      worker = spawn(binary, [], { ...common, env: { ...serviceEnv, SCENEWORKS_WORKER_ONLY: "1" }, stdio: ["ignore", logHandles.worker_stdout.fd, logHandles.worker_stderr.fd] });
      worker.once("error", (error) => spawnErrors.push(`worker: ${error.message}`));
    } finally {
      await Promise.allSettled(Object.values(logHandles).map((handle) => handle.close()));
    }
    for (const child of [api, worker]) child.unref();
    const assertRunning = () => {
      if (spawnErrors.length || api.pid === undefined || worker.pid === undefined || api.exitCode !== null || worker.exitCode !== null || api.signalCode !== null || worker.signalCode !== null) die(`source-built API or worker exited during readiness${spawnErrors.length ? `: ${spawnErrors.join("; ")}` : ""}`);
    };
    const health = await waitHealthy(url, assertRunning);
    const record = { ...identity, ...weights, api_url: url, api_host: "127.0.0.1", api_port: Number(apiPort), state_root: path.relative(output, stateRoot), api_binary: path.relative(root, binary), worker_binary: path.relative(root, binary), api_binary_sha256: sha(await readFile(binary)), api_pid: api.pid, worker_pid: worker.pid, logs: Object.fromEntries(Object.entries(logPaths).map(([name, file]) => [name, path.relative(output, file)])), health, offline: { hf_home: path.relative(output, hfHome), hf_hub_offline: serviceEnv.HF_HUB_OFFLINE, transformers_offline: serviceEnv.TRANSFORMERS_OFFLINE }, started_at: new Date().toISOString() };
    await writeFile(path.join(output, "product-service-provenance.json"), JSON.stringify(record, null, 2) + "\n");
    await writeFile(path.join(output, "product-service-logs.sha256"), `${await productServiceLogsSha256(output)}\n`);
    return record;
  } catch (error) {
    await terminateProductServicePids([worker?.pid, api?.pid], { requireAll: false }).catch((cleanupError) => { error.message += `; service cleanup failed: ${cleanupError.message}`; });
    await writeFile(path.join(output, "product-service-start-failed.json"), JSON.stringify({ api_pid: api?.pid, worker_pid: worker?.pid, logs: Object.fromEntries(Object.entries(logPaths).map(([name, file]) => [name, path.relative(output, file)])), error: error.message, failed_at: new Date().toISOString() }, null, 2) + "\n");
    await writeFile(path.join(output, "product-service-logs.sha256"), `${await productServiceLogsSha256(output)}\n`);
    await rm(stateRoot, { recursive: true, force: true });
    throw error;
  }
}
export async function stopProductService(output) {
  let record;
  try {
    record = JSON.parse(await readFile(path.join(output, "product-service-provenance.json"), "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const failure = JSON.parse(await readFile(path.join(output, "product-service-start-failed.json"), "utf8"));
    await terminateProductServicePids([failure.worker_pid, failure.api_pid], { requireAll: false });
    const logsSha256 = await productServiceLogsSha256(output);
    await writeFile(path.join(output, "product-service-logs.sha256"), `${logsSha256}\n`);
    await writeFile(path.join(output, "product-service-stopped.json"), JSON.stringify({ api_pid: failure.api_pid, worker_pid: failure.worker_pid, status: "start_failed", logs_sha256: logsSha256, stopped_at: new Date().toISOString() }, null, 2) + "\n");
    return;
  }
  const stateRoot = path.resolve(output, record.state_root);
  if (stateRoot !== path.resolve(productServiceStateRoot(output))) die("product service state path escaped its tuple temporary root");
  await terminateProductServicePids([record.worker_pid, record.api_pid]);
  try {
    const response = await fetch(new URL("/api/v1/health", record.api_url));
    if (response.ok) die("product service port remains occupied after shutdown");
  } catch (error) {
    if (String(error.message).includes("port remains occupied")) throw error;
  }
  const logsSha256 = await productServiceLogsSha256(output);
  await writeFile(path.join(output, "product-service-logs.sha256"), `${logsSha256}\n`);
  await writeFile(path.join(output, "product-service-stopped.json"), JSON.stringify({ api_pid: record.api_pid, worker_pid: record.worker_pid, status: "stopped", logs_sha256: logsSha256, stopped_at: new Date().toISOString() }, null, 2) + "\n");
  await rm(stateRoot, { recursive: true, force: true });
}
if (isExecutedModule(import.meta.url)) {
  const [command, ...args] = process.argv.slice(2); const run = command === "start" ? startProductService({ root: args[0], output: args[1], permanentPin: args[2], url: args[3], weightsRoot: args[4] }) : command === "stop" ? stopProductService(args[0]) : Promise.reject(new Error("usage: start <root> <output> <pin> <url> <weights-root> | stop <output>")); run.catch((error) => { console.error(error.message); process.exitCode = 1; });
}
