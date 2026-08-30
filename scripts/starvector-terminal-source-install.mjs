#!/usr/bin/env node
// Build the current product service, install the exact terminal model closure
// through the ordinary typed Model Manager routes, and retain the resulting
// source-produced receipts/Hugging Face cache for offline provisioning. This is
// preparation only: it never invokes a model or emits campaign evidence.
import { spawn } from "node:child_process";
import { lstat, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { validateServiceReceipts } from "./starvector-terminal-provision.mjs";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";

const PROMPT_REVISION = "bba3ae01dfd94089f173c05edd4e1a4c551f2599";
const IDENTITIES = [
  { repo: "starvector/starvector-1b-im2svg", revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", modelId: "starvector_1b" },
  { repo: "starvector/starvector-8b-im2svg", revision: "518beea8dcb5f7a37c5911e92d1d62a76beee7f9", modelId: "starvector_8b" },
  { repo: "SceneWorks/flux1-schnell-mlx", revision: PROMPT_REVISION, modelId: "flux_schnell", variant: "q4" },
];
const TERMINAL = new Set(["completed", "failed", "canceled"]);
const die = (message) => { throw new Error(`starvector terminal source install: ${message}`); };
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export function sourceInstallRequests() {
  return [
    { modelId: "starvector_1b", body: { requestedGpu: "cpu" } },
    { modelId: "starvector_8b", body: { requestedGpu: "cpu" } },
    { modelId: "flux_schnell", body: { requestedGpu: "cpu", variant: "q4" } },
  ];
}

export function sourceInstallEnvironment(stateRoot, url, inherited = process.env) {
  const parsed = new URL(url);
  if (!path.isAbsolute(stateRoot) || parsed.protocol !== "http:" || parsed.hostname !== "127.0.0.1" || !parsed.port) {
    die("absolute source root and explicit loopback API URL are required");
  }
  const dataDir = path.join(stateRoot, "app-data"), hfHome = path.join(stateRoot, "hf-home"), hub = path.join(hfHome, "hub");
  const env = {
    ...inherited,
    SCENEWORKS_API_HOST: "127.0.0.1",
    SCENEWORKS_API_PORT: parsed.port,
    SCENEWORKS_API_URL: url,
    SCENEWORKS_DATA_DIR: dataDir,
    SCENEWORKS_CONFIG_DIR: path.join(stateRoot, "config"),
    SCENEWORKS_JOBS_DB_PATH: path.join(dataDir, "cache", "jobs.db"),
    SCENEWORKS_GPU_ID: "cpu",
    SCENEWORKS_UTILITY_WORKERS: "1",
    SCENEWORKS_BACKEND_MLX_ENABLED: "false",
    SCENEWORKS_BACKEND_CANDLE_ENABLED: "false",
    HF_HOME: hfHome,
    HF_HUB_CACHE: hub,
    HUGGINGFACE_HUB_CACHE: hub,
  };
  for (const name of ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "SCENEWORKS_TERMINAL_CAMPAIGN", "SCENEWORKS_TERMINAL_NO_JOB_DOWNLOADS"]) delete env[name];
  return { env, dataDir, hfHome };
}

async function responseJson(response, context) {
  let body;
  try { body = await response.json(); } catch { die(`${context} returned non-JSON HTTP ${response.status}`); }
  if (!response.ok) die(`${context} returned HTTP ${response.status}: ${body?.detail ?? JSON.stringify(body)}`);
  return body;
}

async function waitHealthy(url, children) {
  for (let attempt = 0; attempt < 180; attempt += 1) {
    if (children.some((child) => child.exitCode !== null)) die("source-built API or worker exited during startup");
    try {
      const response = await fetch(new URL("/api/v1/health", url));
      const body = await response.json();
      if (response.ok && body?.status === "ok" && body?.readiness?.status === "ready") return body;
    } catch { /* service is still starting */ }
    await delay(1000);
  }
  die("source-built API did not become ready");
}

async function installModel(url, request, children) {
  const created = await responseJson(await fetch(new URL(`/api/v1/models/${request.modelId}/download`, url), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request.body),
  }), `Model Manager install ${request.modelId}`);
  if (typeof created?.id !== "string" || !created.id) die(`Model Manager install ${request.modelId} returned no job id`);
  for (let attempt = 0; attempt < 10800; attempt += 1) {
    if (children.some((child) => child.exitCode !== null)) die(`source-built API or worker exited while installing ${request.modelId}`);
    const job = await responseJson(await fetch(new URL(`/api/v1/jobs/${created.id}`, url)), `Model Manager job ${created.id}`);
    if (TERMINAL.has(job.status)) {
      if (job.status !== "completed") die(`${request.modelId} ended ${job.status}: ${job.error ?? job.message ?? "no error detail"}`);
      return { model_id: request.modelId, job_id: created.id, completed_at: job.completedAt };
    }
    await delay(2000);
  }
  die(`timed out waiting for Model Manager install ${request.modelId}`);
}

async function stop(children) {
  for (const child of [...children].reverse()) if (child.exitCode === null) child.kill("SIGTERM");
  for (let attempt = 0; attempt < 100 && children.some((child) => child.exitCode === null); attempt += 1) await delay(100);
  if (children.some((child) => child.exitCode === null)) die("source-built API or worker did not stop");
}

export async function installSourceClosure({ root, stateRoot, url }) {
  if (!path.isAbsolute(root)) die("absolute repository root is required");
  const binary = path.join(root, "target", "debug", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api");
  const binaryInfo = await lstat(binary).catch(() => null);
  if (!binaryInfo?.isFile()) die(`build the current-tree product service first: ${binary}`);
  const { env, dataDir, hfHome } = sourceInstallEnvironment(stateRoot, url);
  await mkdir(dataDir, { recursive: true }); await mkdir(path.join(stateRoot, "config"), { recursive: true }); await mkdir(hfHome, { recursive: true });
  const api = spawn(binary, [], { cwd: root, env, stdio: "inherit" });
  const worker = spawn(binary, [], { cwd: root, env: { ...env, SCENEWORKS_WORKER_ONLY: "1" }, stdio: "inherit" });
  const children = [api, worker];
  try {
    const health = await waitHealthy(url, children), jobs = [];
    for (const request of sourceInstallRequests()) jobs.push(await installModel(url, request, children));
    await validateServiceReceipts(dataDir, hfHome, IDENTITIES);
    const record = { schema_version: 1, preparation_only: true, model_execution: false, campaign_evidence: false, data_dir: dataDir, hf_home: hfHome, prompt_revision: PROMPT_REVISION, jobs, health, completed_at: new Date().toISOString() };
    await writeFile(path.join(stateRoot, "source-closure.json"), `${JSON.stringify(record, null, 2)}\n`);
    return record;
  } finally { await stop(children); }
}

if (isExecutedModule(import.meta.url)) {
  const [root, stateRoot, url] = process.argv.slice(2);
  installSourceClosure({ root, stateRoot, url }).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
