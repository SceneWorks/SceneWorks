#!/usr/bin/env node
// Start the actual current-tree Rust API and worker for the terminal campaign.
// This is intentionally not an arbitrary service command: its source revision,
// Cargo inference pin, binaries and health response are all recorded together.
import { createHash } from "node:crypto";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { copyFile, lstat, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";

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
async function git(root, args) { return (await execFile("git", ["-C", root, ...args])).stdout.trim(); }
async function serviceIdentity(root, permanentPin) {
  if (await git(root, ["status", "--porcelain"])) die("current-tree product service checkout must be clean");
  const revision = await git(root, ["rev-parse", "HEAD"]), cargo = await readFile(path.join(root, "Cargo.toml"), "utf8");
  const pin = cargo.match(/SceneWorks\/inference",\s*rev\s*=\s*"([a-f0-9]{40})"/)?.[1];
  if (!pin || pin !== permanentPin) die("current-tree product service Cargo inference pin mismatches permanent pin");
  return { sceneworks_revision: revision, inference_revision: pin };
}
async function waitHealthy(url) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try { const response = await fetch(new URL("/api/v1/health", url)); const body = await response.json(); if (response.ok && body?.status === "ok" && body?.readiness?.status === "ready") return body; } catch { /* booting */ }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  die("source-built API did not become ready");
}
async function copyRegularTree(source, destination) {
  const info = await lstat(source);
  if (info.isSymbolicLink()) die(`weights closure rejects symlink ${source}`);
  if (info.isDirectory()) {
    await mkdir(destination, { recursive: true });
    for (const name of await readdir(source)) await copyRegularTree(path.join(source, name), path.join(destination, name));
    return;
  }
  if (!info.isFile()) die(`weights closure rejects non-regular file ${source}`);
  await mkdir(path.dirname(destination), { recursive: true });
  await copyFile(source, destination);
  if (sha(await readFile(source)) !== sha(await readFile(destination))) die(`weights closure copy hash drifted: ${source}`);
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
  const hashTree = async (root) => {
    const rows = [];
    for (const name of await readdir(root, { recursive: true })) { const file = path.join(root, name), info = await lstat(file); if (info.isSymbolicLink()) die(`weights closure copy contains symlink ${name}`); if (info.isFile()) rows.push([name.split(path.sep).join("/"), info.size, sha(await readFile(file))]); }
    rows.sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0); return sha(JSON.stringify(rows));
  };
  if (await hashTree(path.join(stateRoot, "data")) !== closure.app_data_sha256 || await hashTree(path.join(stateRoot, "hf")) !== closure.hf_home_sha256) die("materialized receipt/HF closure hash mismatch");
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
  const common = { cwd: root, detached: true, stdio: ["ignore", "pipe", "pipe"] }, parsed = new URL(url), apiPort = parsed.port, stateRoot = productServiceStateRoot(output);
  if (parsed.hostname !== "127.0.0.1" || !apiPort) die("product service must bind an explicit loopback host/port");
  if (await lstat(stateRoot).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error))) die("temporary product service state already exists");
  await mkdir(path.join(stateRoot, "data"), { recursive: true }); await mkdir(path.join(stateRoot, "config"), { recursive: true });
  const weights = await materializeOfflineWeights(weightsRoot, stateRoot);
  const hfHome = path.join(stateRoot, "hf");
  const serviceEnv = { ...process.env, ...productServiceBackendEnv(), SCENEWORKS_TERMINAL_CAMPAIGN: "1", SCENEWORKS_API_HOST: "127.0.0.1", SCENEWORKS_API_PORT: apiPort, SCENEWORKS_API_URL: url, SCENEWORKS_DATA_DIR: path.join(stateRoot, "data"), SCENEWORKS_CONFIG_DIR: path.join(stateRoot, "config"), SCENEWORKS_JOBS_DB_PATH: path.join(stateRoot, "data", "cache", "jobs.db"), SCENEWORKS_GPU_ID: process.env.STARVECTOR_TERMINAL_GPU_ID ?? "auto", HF_HOME: hfHome, HUGGINGFACE_HUB_CACHE: path.join(hfHome, "hub"), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" };
  for (const inherited of ["TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_ENDPOINT"]) delete serviceEnv[inherited];
  const api = spawn(binary, [], { ...common, env: serviceEnv });
  const worker = spawn(binary, [], { ...common, env: { ...serviceEnv, SCENEWORKS_WORKER_ONLY: "1" } });
  const stdout = [], stderr = []; for (const child of [api, worker]) { child.stdout.on("data", (chunk) => stdout.push(chunk)); child.stderr.on("data", (chunk) => stderr.push(chunk)); child.unref(); }
  const health = await waitHealthy(url); if (api.exitCode !== null || worker.exitCode !== null || api.pid === undefined || worker.pid === undefined) die("source-built API or worker exited during readiness"); const record = { ...identity, ...weights, api_url: url, api_host: "127.0.0.1", api_port: Number(apiPort), state_root: path.relative(output, stateRoot), api_binary: path.relative(root, binary), worker_binary: path.relative(root, binary), api_binary_sha256: sha(await readFile(binary)), api_pid: api.pid, worker_pid: worker.pid, health, offline: { hf_home: path.relative(output, hfHome), hf_hub_offline: serviceEnv.HF_HUB_OFFLINE, transformers_offline: serviceEnv.TRANSFORMERS_OFFLINE }, started_at: new Date().toISOString() };
  await writeFile(path.join(output, "product-service-provenance.json"), JSON.stringify(record, null, 2) + "\n");
  await writeFile(path.join(output, "product-service-logs.sha256"), sha(Buffer.concat([...stdout, ...stderr])) + "\n");
  return record;
}
export async function stopProductService(output) {
  const record = JSON.parse(await readFile(path.join(output, "product-service-provenance.json"), "utf8"));
  const stateRoot = path.resolve(output, record.state_root);
  if (stateRoot !== path.resolve(productServiceStateRoot(output))) die("product service state path escaped its tuple temporary root");
  for (const pid of [record.worker_pid, record.api_pid]) { if (!Number.isInteger(pid) || pid < 1) die("invalid product service PID"); try { process.kill(pid, "SIGTERM"); } catch (error) { if (error.code !== "ESRCH") throw error; } }
  for (let attempt = 0; attempt < 50; attempt += 1) {
    let alive = false;
    for (const pid of [record.worker_pid, record.api_pid]) {
      try { process.kill(pid, 0); alive = true; } catch (error) { if (error.code !== "ESRCH") throw error; }
    }
    if (!alive) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
    if (attempt === 49) die("product service did not stop after SIGTERM");
  }
  try {
    const response = await fetch(new URL("/api/v1/health", record.api_url));
    if (response.ok) die("product service port remains occupied after shutdown");
  } catch (error) {
    if (String(error.message).includes("port remains occupied")) throw error;
  }
  await writeFile(path.join(output, "product-service-stopped.json"), JSON.stringify({ api_pid: record.api_pid, worker_pid: record.worker_pid, stopped_at: new Date().toISOString() }, null, 2) + "\n");
  await rm(stateRoot, { recursive: true, force: true });
}
if (import.meta.url === `file://${process.argv[1]}`) {
  const [command, ...args] = process.argv.slice(2); const run = command === "start" ? startProductService({ root: args[0], output: args[1], permanentPin: args[2], url: args[3], weightsRoot: args[4] }) : command === "stop" ? stopProductService(args[0]) : Promise.reject(new Error("usage: start <root> <output> <pin> <url> <weights-root> | stop <output>")); run.catch((error) => { console.error(error.message); process.exitCode = 1; });
}
