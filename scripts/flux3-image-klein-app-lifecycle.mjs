// sc-26234: ONE ordinary installed-model/API/worker/library smoke, not a provider probe
// or memory campaign. The API alone creates jobs and publishes worker assetWrites.
import assert from "node:assert/strict";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, cp, mkdir, open, readFile, realpath, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { assertPins, assertRunnerIdentity, buildChildEnv } from "./qwen-image-2-1-app-lifecycle.mjs";

const execFile = promisify(execFileCallback);
const fact = (value, message) => assert.ok(value, message);
export const MODEL_REVISION = "8454017f9ebaada30c998fccd9f67a80c7580f68";
export const MODEL_REPO = "black-forest-labs/flux-3-image-klein";
const REQUIRED_FILES = ["config.json", "manifest.json", "model.safetensors", "ae.safetensors", "LICENSE.md", "NOTICE.md", "text_encoder/config.json", "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json", "text_encoder/model.safetensors.index.json", "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors"];
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
async function hash(file) { const h = createHash("sha256"); for await (const chunk of createReadStream(file)) h.update(chunk); return h.digest("hex"); }

export function assertCheckoutIdentity({ source, inference, dispatch, head, dirty }) {
  fact(/^[a-f0-9]{40}$/.test(source ?? "") && /^[a-f0-9]{40}$/.test(inference ?? ""), "exact 40-hex source_sha and inference_sha required; branch names are refused");
  fact(source === dispatch && source === head && dirty === "", "dispatch SHA, clean checkout HEAD and supplied source_sha must agree");
}
export function assertModel(model) {
  fact(model?.id === "flux3_klein" && model.family === "flux3" && model.capabilities?.includes("text_to_image"), "select the native FLUX3 distilled T2I catalog entry");
  fact(model.gated === true && model.credentialHost === "huggingface.co" && /^https:\/\/huggingface\.co\/black-forest-labs\/flux-3-image-klein\/blob\/(main|8454017f9ebaada30c998fccd9f67a80c7580f68)\/LICENSE\.md$/.test(model.licenseUrl ?? ""), "frozen upstream FLUX Kommunity license and gated HF credential declaration required; no rehost");
  fact(Array.isArray(model.downloads), "pinned model downloads required");
  const downloads = model.downloads.filter((download) => download.coRequisite !== true);
  const optional = model.downloads.filter((download) => download.coRequisite === true);
  fact(downloads.length === 1 && downloads[0].provider === "huggingface" && downloads[0].repo === MODEL_REPO && downloads[0].revision === MODEL_REVISION, "one exact upstream pinned primary package required");
  // S5 exposes a separate optional package. Its presence in the catalog must not
  // reject this unchanged plain T2I profile or turn enhancement into a requirement.
  fact(optional.length <= 1 && optional.every((download) => download.provider === "huggingface" && download.repo === "black-forest-labs/flux-3-image-klein-pup" && download.revision === "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed" && download.required === "soft" && download.componentId === "prompt_upsampler" && Array.isArray(download.files) && download.files.length === 0), "only the exact pinned soft PUP component may accompany the primary package");
  // Fetch every file at the pinned authority. Required component/index/SHA validation below
  // remains independent of this download policy; an allow-list must not under-fetch the package.
  fact(Array.isArray(downloads[0].files) && downloads[0].files.length === 0, "whole pinned repository download required (files: []), not a selective allow-list");
}
export function assertDownload(job) {
  fact(job?.type === "model_download" && job.payload?.modelId === "flux3_klein" && job.payload.repo === MODEL_REPO && job.payload.revision === MODEL_REVISION, "ordinary download job must retain exact model/repository/revision identity");
}
export function imageBody(projectId) {
  return { projectId, mode: "text_to_image", model: "flux3_klein", prompt: "A small teal ceramic teapot on a pale yellow table, soft daylight, studio photograph", negativePrompt: "", seed: 26234, width: 1024, height: 1024, count: 1, advanced: { steps: 16, sampler: "euler", guidanceScale: 0 }, loras: [] };
}
export function assertAccepted(job, body) {
  fact(job?.type === "image_generate" && job.projectId === body.projectId && typeof job.id === "string" && job.id.length > 0, "typed image API must create an ordinary project image_generate job");
  for (const key of ["mode", "model", "prompt", "seed", "width", "height", "count"]) assert.deepEqual(job.payload?.[key], body[key], `accepted image job must preserve ${key}`);
  for (const key of ["steps", "sampler", "guidanceScale"]) assert.deepEqual(job.payload?.advanced?.[key], body.advanced[key], `accepted image job must preserve advanced.${key}`);
}
// Backend-neutral seam: Candle can reuse the request + saved-asset checks with its
// own admitted runner/launcher and expected adapter, not impersonate a Mac receipt.
export function assertSavedAsset(job, library, sidecar, { projectId, workerId, backend, adapter }) {
  const body = imageBody(projectId); assertAccepted(job, body);
  fact(job.status === "completed" && job.stage === "completed" && job.backend === backend && job.workerId === workerId && job.error == null, "owned native worker must complete the ordinary job");
  const result = job.result;
  fact(result?.model === body.model && result.adapter === adapter && !Object.hasOwn(result, "assetWrites") && result.assetIds?.length === 1 && result.assets?.length === 1, "API must publish worker facts into final assets/assetIds, not provider-only PNG or unregistered assetWrites");
  fact(library?.id && result.assetIds[0] === library.id && result.assets[0].id === library.id && library.projectId === projectId && library.type === "image", "job result must identify the saved project-library image");
  // Job results and disk sidecars are raw persistence envelopes. The library API adds these
  // three filesystem-derived fields; compare every persisted field, then verify hydration itself.
  const { url, sidecarPath, generationSet, ...persistedLibrary } = library;
  assert.deepEqual(result.assets[0], sidecar, "job asset and raw persisted sidecar must agree");
  assert.deepEqual(persistedLibrary, sidecar, "library persisted fields and raw sidecar must agree");
  fact(library.generationSetId && library.generationSetId === result.generationSetId && library.lineage?.jobId === job.id, "saved asset provenance must bind generation set and actual job");
  const recipe = library.recipe;
  fact(recipe?.model === body.model && recipe.adapter === adapter && recipe.mode === body.mode && recipe.prompt === body.prompt && recipe.seed === body.seed, "saved recipe must preserve native model/adapter/mode/prompt/seed");
  fact(recipe.rawAdapterSettings?.realModelInference === true && recipe.rawAdapterSettings?.numInferenceSteps === 16 && recipe.rawAdapterSettings?.repo === MODEL_REPO, "real flow-map inference settings required; no procedural stub");
  const settings = recipe.normalizedSettings;
  fact(settings?.family === "flux3" && settings.width === body.width && settings.height === body.height && settings.count === 1, "saved FLUX3 settings required");
  const file = library.file;
  fact(file?.mimeType === "image/png" && file.width === body.width && file.height === body.height && /^assets\/images\/[a-zA-Z0-9_./-]+\.png$/.test(file.path ?? "") && !file.path.split("/").includes(".."), "saved PNG must be a project-relative library image of the requested dimensions");
  assert.equal(url, `/api/v1/projects/${projectId}/files/${file.path}`, "hydrated library URL must identify the exact saved project file");
  assert.equal(sidecarPath, file.path.replace(/\.png$/, ".sceneworks.json"), "hydrated sidecar path must identify the exact project-relative sidecar");
  fact(generationSet?.schemaVersion === 1 && generationSet.id === library.generationSetId && generationSet.projectId === projectId && generationSet.jobId === job.id, "hydrated generation set must identify the same persisted set, project and job");
  for (const key of ["mode", "model", "prompt", "negativePrompt", "count"]) assert.deepEqual(generationSet[key], body[key], `hydrated generation set must preserve ${key}`);
  fact(typeof generationSet.createdAt === "string" && generationSet.createdAt.length > 0, "hydrated generation set must retain its creation provenance");
  assert.deepEqual(generationSet.recipe, sidecar.recipe, "hydrated generation set recipe must match the raw persisted recipe");
  return { job_id: job.id, asset_id: library.id, generation_set_id: library.generationSetId, project_id: projectId, backend, adapter, model: body.model, media_path: file.path };
}
export function assertServedPng(served, saved, width, height) {
  fact(served.length > 24 && served.subarray(0, 8).toString("hex") === "89504e470d0a1a0a" && served.toString("ascii", 12, 16) === "IHDR" && served.readUInt32BE(16) === width && served.readUInt32BE(20) === height, "product file route must serve the generated PNG at requested dimensions");
  fact(sha256(served) === sha256(saved), "API-served PNG must exactly match the saved library file");
}

export function redactSensitiveText(text, secrets) {
  for (const secret of secrets) if (secret) for (const value of new Set([secret, encodeURIComponent(secret)])) text = text.split(value).join("[REDACTED]");
  return text;
}

async function ownedFile(file, root) {
  const resolved = await realpath(file), relative = path.relative(await realpath(root), resolved);
  fact(relative && relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative) && (await stat(resolved)).isFile() && (await stat(resolved)).size > 0, "evidence file must be nonempty and remain inside its owned root"); return resolved;
}

// Discover only the actual port-zero listener announced by our exclusively created
// API log. Never probe-close-bind a fixed port or discover an endpoint over HTTP.
export function boundApiUrl(text) {
  for (const line of text.split("\n")) {
    let event; try { event = JSON.parse(line); } catch { continue; }
    if (event.event !== "api_listening") continue;
    const match = /^127\.0\.0\.1:([0-9]{1,5})$/.exec(event.address ?? "");
    fact(match && Number(match[1]) > 0 && Number(match[1]) <= 65535, "owned API bound address invalid");
    return `http://127.0.0.1:${Number(match[1])}`;
  }
  return undefined;
}

export function attemptIdentity(output) {
  const nonce = randomBytes(24).toString("hex");
  return { nonce, state: path.join(output, `state-${nonce}`), workerId: `flux3-app-${nonce}` };
}

export function bindOwnedEnv(childEnv, { state, workerId }, url) {
  // Bind AFTER the shared helper scrubs ambient SCENEWORKS_* values and overwrite
  // its historical fixed port. Keep the runner's HF_TOKEN only in the child env.
  const bound = { ...childEnv, SCENEWORKS_DATA_DIR: path.join(state, "data"), SCENEWORKS_CONFIG_DIR: path.join(state, "config"), SCENEWORKS_JOBS_DB_PATH: path.join(state, "data", "cache", "jobs.db"), SCENEWORKS_CREDENTIALS_DIR: path.join(state, "credentials"), SCENEWORKS_WORKER_ID: workerId, SCENEWORKS_API_HOST: "127.0.0.1", SCENEWORKS_API_PORT: url ? new URL(url).port : "0", SCENEWORKS_LOG_FORMAT: "json", SCENEWORKS_RUN_UTILITY_INPROCESS: "0", RUST_LOG: "info" };
  // The API derives its own internal self-call URL from its actual port-zero
  // listener. Only the separately launched worker gets the discovered URL.
  if (url) bound.SCENEWORKS_API_URL = url; else delete bound.SCENEWORKS_API_URL;
  return bound;
}

// Field names are the API's serialized camelCase HealthResponse/DirectoriesResponse
// (apps/rust-api dto.rs). The JS tests consume scripts/fixtures/sceneworks-api-health-ready.json,
// which a rust-api test asserts is the real ready body, so a rename fails both suites.
export async function assertOwnedHealth(health, state) {
  fact(health?.status === "ok" && health.service === "sceneworks-api" && health.runtime === "rust" && health.readiness?.status === "ready" && health.authRequired === false, "owned API readiness/identity required");
  const expected = { data: path.join(state, "data"), config: path.join(state, "config"), projects: path.join(state, "data", "projects"), jobsDb: path.join(state, "data", "cache", "jobs.db") };
  for (const [key, file] of Object.entries(expected)) {
    const actual = health.directories?.[key];
    fact(typeof actual === "string" && path.isAbsolute(actual), "owned API directory identity missing");
    let same = false;
    try { same = await realpath(actual) === await realpath(file); } catch { /* no foreign paths/body in errors */ }
    fact(same, "owned API canonical directory identity mismatch");
  }
}

// One gate for worker launch and every subsequent request. API exit cancels even
// an in-flight response-body read; no foreign body/path/fetch error is disclosed.
export function ownedApiController({ url, state, api, controller, fetchImpl = fetch }) {
  const abort = () => controller.abort(new Error("owned API exited"));
  api.once("exit", abort); api.once("error", abort);
  const alive = () => {
    if (controller.signal.aborted) throw controller.signal.reason;
    fact(api.exitCode === null && api.signalCode === null, "owned API exited");
  };
  const request = async (method, route, body) => {
    alive();
    try {
      const response = await fetchImpl(url + route, { method, redirect: "error", headers: { "content-type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]) });
      const value = await response.json(); alive();
      fact(response.ok, "owned API request refused"); return value;
    } catch { alive(); throw new Error("owned API request unavailable"); }
  };
  const verify = async () => {
    const health = await request("GET", "/api/v1/health");
    if (health?.readiness?.status !== "ready") return false;
    try { await assertOwnedHealth(health, state); alive(); return health; }
    catch { controller.abort(new Error("owned API endpoint identity refused")); throw controller.signal.reason; }
  };
  const requireOwned = async () => { fact(await verify(), "owned API not ready"); alive(); };
  return {
    verify,
    async startWorker(launch) { await requireOwned(); alive(); return launch(); },
    async call(method, route, body) { await requireOwned(); return request(method, route, body); },
    async served(route) {
      await requireOwned();
      try {
        const response = await fetchImpl(url + route, { redirect: "error", signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]) });
        fact(response.ok, "owned API file refused"); const bytes = Buffer.from(await response.arrayBuffer()); alive(); return { bytes, status: response.status };
      } catch { alive(); throw new Error("owned API file unavailable"); }
    },
    close() { api.removeListener("exit", abort); api.removeListener("error", abort); },
  };
}

export function stopOwnedChildren(children, kill = process.kill) {
  for (const child of children) if (child.pid && child.exitCode === null && child.signalCode === null) {
    try { kill(-child.pid, "SIGTERM"); } catch { /* final cleanup verifies exit */ }
  }
}

export async function runLifecycle(env = process.env, { preflightOnly = false } = {}) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  fact(env.FLUX3_APP_OUTPUT && env.RUNNER_TEMP, "unique FLUX3_APP_OUTPUT under RUNNER_TEMP required");
  const output = path.resolve(env.FLUX3_APP_OUTPUT), relative = path.relative(path.resolve(env.RUNNER_TEMP), output);
  fact(relative && !relative.startsWith("..") && !path.isAbsolute(relative), "output must remain under RUNNER_TEMP");
  await mkdir(output, { recursive: true }); const evidence = path.join(output, "evidence"), logs = path.join(output, "private-logs"); await mkdir(evidence, { recursive: true }); await mkdir(logs, { recursive: true });
  const receipt = { schema_version: 1, kind: "flux3_image_klein_mlx_app_lifecycle", status: "running", phase: preflightOnly ? "preflight" : "lifecycle", started_at: new Date().toISOString(), identity: { source_sha: env.FLUX3_APP_SOURCE_SHA, inference_sha: env.FLUX3_APP_INFERENCE_SHA, dispatch_sha: env.GITHUB_SHA, runner_name: env.RUNNER_NAME, runner_target: env.FLUX3_APP_RUNNER_TARGET, run_id: env.GITHUB_RUN_ID, run_attempt: env.GITHUB_RUN_ATTEMPT, model_repo: MODEL_REPO, model_revision: MODEL_REVISION }, events: [] };
  const secrets = new Set(Object.entries(env).filter(([key, value]) => /TOKEN$|SECRET|PASSWORD|API_KEY/i.test(key) && value && value.length >= 8).map(([, value]) => value));
  const save = async () => { await writeFile(path.join(evidence, "receipt.tmp"), redactSensitiveText(JSON.stringify(receipt, null, 2) + "\n", secrets)); await rename(path.join(evidence, "receipt.tmp"), path.join(evidence, "receipt.json")); };
  await save();
  const children = [], handles = [], controller = new AbortController();
  let endpoint;
  // Abort fetches and stop only our own process groups immediately on API exit.
  const stopOwned = () => stopOwnedChildren(children);
  controller.signal.addEventListener("abort", stopOwned, { once: true });
  const abort = () => controller.abort(new Error("FLUX3 lifecycle interrupted"));
  process.once("SIGTERM", abort); process.once("SIGINT", abort);
  const deadline = setTimeout(() => controller.abort(new Error("FLUX3 app lifecycle exceeded 150 minutes")), 150 * 60_000);
  const alive = () => { if (controller.signal.aborted) throw controller.signal.reason; for (const c of children.filter((c) => c.service)) fact(c.exitCode === null && c.signalCode === null, "owned API/worker exited; inspect owned logs"); };
  const delay = async () => { alive(); await new Promise((resolve) => setTimeout(resolve, 1000)); alive(); };
  const launch = async (command, args, childEnv, name, service = false) => {
    const log = await open(path.join(logs, `${name}.log`), "wx"); handles.push(log);
    alive();
    const child = spawn(command, args, { cwd: root, env: childEnv, detached: true, stdio: ["ignore", log.fd, log.fd] }); children.push(child); child.service = service;
    child.done = new Promise((resolve, reject) => { child.once("error", reject); child.once("exit", (code, signal) => resolve({ code, signal })); });
    child.done.catch(() => controller.abort(new Error(`${name} process failed to launch`)));
    if (service) child.once("exit", () => controller.abort(new Error("owned API/worker exited")));
    return child;
  };
  const checked = async (command, args, childEnv, name) => {
    const child = await launch(command, args, childEnv, name);
    while (child.exitCode === null && child.signalCode === null) await delay();
    fact((await child.done).code === 0, `${name} failed; inspect owned log`);
  };
  try {
    const git = async (...args) => (await execFile("git", args, { cwd: root, timeout: 30_000, maxBuffer: 8 * 1024 * 1024 })).stdout.trim();
    assertCheckoutIdentity({ source: env.FLUX3_APP_SOURCE_SHA, inference: env.FLUX3_APP_INFERENCE_SHA, dispatch: env.GITHUB_SHA, head: await git("rev-parse", "HEAD"), dirty: await git("status", "--porcelain") });
    assertRunnerIdentity(env.FLUX3_APP_RUNNER_TARGET, env.RUNNER_NAME);
    fact(process.platform === "darwin" && process.arch === "arm64", "native Apple Silicon Mac required; Windows cannot compile or certify MLX");
    const manifests = await Promise.all(["Cargo.toml", "crates/sceneworks-worker/Cargo.toml", "crates/sceneworks-memory-adapter/Cargo.toml"].map((file) => readFile(path.join(root, file), "utf8")));
    assertPins(manifests, await readFile(path.join(root, "Cargo.lock"), "utf8"), env.FLUX3_APP_INFERENCE_SHA);
    const hardware = JSON.parse((await execFile("system_profiler", ["SPHardwareDataType", "-json"], { timeout: 30_000 })).stdout).SPHardwareDataType;
    fact(Array.isArray(hardware) && hardware.length === 1 && hardware[0].chip_type?.startsWith("Apple") && Number.parseInt(hardware[0].physical_memory, 10) >= 32, "Apple Silicon with at least 32 GB unified memory required");
    receipt.identity.hardware = hardware.map(({ chip_type, physical_memory, machine_model }) => ({ chip_type, physical_memory, machine_model }));
    // Prefer the selected runner account's OWN HF login, not a dispatcher secret.
    // Read once in memory; no credential paths/content enter the receipt or artifact.
    let token;
    try { token = (await readFile(env.HF_TOKEN_PATH || path.join(env.HF_HOME || path.join(env.HOME, ".cache", "huggingface"), "token"), "utf8")).trim(); } catch { /* actionable refusal below */ }
    fact(token, "Selected Mac has no stored HF credential. Log that runner account into Hugging Face and authorize the FLUX Kommunity gate; do not transfer a Windows token or rehost weights.");
    secrets.add(token);
    const auth = await fetch(`https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/config.json`, { headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]) });
    fact(auth.ok, `Selected Mac cannot fetch the pinned gated config (HTTP ${auth.status}). Authorize this runner account for black-forest-labs/flux-3-image-klein and refresh its stored HF login.`);
    const config = Buffer.from(await auth.arrayBuffer()); fact(config.length > 0, "pinned gated config empty");
    receipt.auth = { status: "passed", mechanism: "runner_stored_hf_credential", pinned_file: "config.json", http_status: auth.status, sha256: sha256(config) };
    if (preflightOnly) { receipt.status = "passed"; return receipt; }
    const identity = attemptIdentity(output), { state, workerId } = identity; receipt.identity.attempt_nonce = identity.nonce; await mkdir(state); await mkdir(path.join(state, "data", "projects"), { recursive: true }); await mkdir(path.join(state, "credentials")); await cp(path.join(root, "config"), path.join(state, "config"), { recursive: true });
    const weights = path.join(env.HOME, "sceneworks-flux3-app-weights"), hub = path.join(weights, "hub"), xet = path.join(weights, "xet");
    await mkdir(hub, { recursive: true }); await mkdir(xet, { recursive: true });
    fact(await realpath(weights) === weights && await realpath(hub) === hub, "physical task-owned model hub required");
    let childEnv = bindOwnedEnv(buildChildEnv(env, { state, weights, hub, xet, workerId, url: "http://127.0.0.1:0" }), identity);
    // Ordinary native downloader receives the runner login only in its environment.
    // Its online fetch installs the manifest-selected pinned package, not a fixture.
    delete childEnv.HF_HUB_OFFLINE; delete childEnv.TRANSFORMERS_OFFLINE; childEnv.HF_TOKEN = token; token = undefined;
    const target = path.join(output, "target");
    await checked("cargo", ["build", "--release", "--locked", "-p", "sceneworks-rust-api"], { ...env, CARGO_TARGET_DIR: target }, "build");
    fact(await git("status", "--porcelain") === "", "build changed the exact pinned checkout");
    const binary = path.join(target, "release", "sceneworks-rust-api"); receipt.identity.binary_sha256 = await hash(binary);
    await checked(binary, [], { ...childEnv, SCENEWORKS_GPU_CHECK: "1" }, "metal-preflight");
    const api = await launch(binary, [], childEnv, "api", true);
    const wait = async (fn, seconds, label) => { const end = Date.now() + seconds * 1000; while (Date.now() < end) { alive(); const value = await fn(); if (value) return value; await delay(); } throw new Error(`${label} timed out`); };
    const url = await wait(async () => { alive(); const text = await readFile(path.join(logs, "api.log"), "utf8"); fact(text.length <= 8 * 1024 * 1024, "owned API startup log exceeded bound"); return boundApiUrl(text); }, 180, "owned API bound-port announcement");
    childEnv = bindOwnedEnv(childEnv, identity, url);
    endpoint = ownedApiController({ url, state, api, controller });
    receipt.health = await wait(async () => { try { return await endpoint.verify(); } catch { alive(); return false; } }, 180, "owned API ready and canonical nonce paths (HTTP 200 alone is not readiness)");
    await endpoint.startWorker(() => launch(binary, [], { ...childEnv, SCENEWORKS_WORKER_ONLY: "1" }, "worker", true));
    const must = (...args) => endpoint.call(...args);
    receipt.worker = await wait(async () => { const w = (await must("GET", "/api/v1/workers")).find((w) => w.id === workerId); if (!w || w.status !== "idle") return false; fact(w.gpuId === "mlx" && w.gpuName === "Apple Silicon (MLX)" && w.capabilities?.includes("gpu") && !w.capabilities.some((c) => ["cpu", "candle", "nvidia"].includes(c)), "owned native MLX worker required"); return w; }, 180, "owned MLX worker registration");
    const relocation = await must("POST", "/api/v1/model-library/relocate", { path: weights });
    fact(relocation.adopted === true && relocation.hfHome === weights && relocation.libraryRoot === hub, "API must adopt the exact task-owned library");
    const getModel = async () => (await must("GET", "/api/v1/models")).find((m) => m.id === "flux3_klein");
    const model = await getModel(); assertModel(model); receipt.catalog_model = model;
    const download = await must("POST", "/api/v1/models/flux3_klein/download", { licenseAcknowledged: true }); assertDownload(download); receipt.download_job_id = download.id; await save();
    const getJob = (id) => must("GET", `/api/v1/jobs/${id}`);
    const completedJob = async (id) => { const j = await getJob(id); fact(!["failed", "canceled"].includes(j.status), `ordinary job ${id} ${j.status}; inspect owned worker log (HF 401/403 requires this Mac account's gate acceptance)`); const event = { job_id: id, status: j.status, stage: j.stage, backend: j.backend, worker_id: j.workerId }; if (JSON.stringify(receipt.events.at(-1)) !== JSON.stringify(event)) { receipt.events.push(event); await save(); } return j.status === "completed" && j; };
    receipt.download = await wait(() => completedJob(download.id), 60 * 60, "ordinary pinned installation"); assertDownload(receipt.download);
    const installed = await getModel(); assertModel(installed);
    const snapshot = path.join(hub, "models--black-forest-labs--flux-3-image-klein", "snapshots", MODEL_REVISION);
    fact(installed.installState === "installed" && installed.installedPath && await realpath(installed.installedPath) === await realpath(snapshot), "selected catalog model must be installed at the exact pinned snapshot");
    receipt.components = [];
    for (const file of REQUIRED_FILES) { alive(); const resolved = await ownedFile(path.join(snapshot, file), hub); receipt.components.push({ file, bytes: (await stat(resolved)).size, sha256: await hash(resolved) }); }
    fact(await hash(await ownedFile(path.join(snapshot, "config.json"), hub)) === receipt.auth.sha256, "installed config must match the actually authorized pinned HF file");
    const license = await readFile(await ownedFile(path.join(snapshot, "LICENSE.md"), hub), "utf8");
    fact(/FLUX.*Kommunity/i.test(license), "installed package must carry FLUX Kommunity terms");
    receipt.installed_model = installed; await save();
    const project = await must("POST", "/api/v1/projects", { name: "sc-26234 FLUX3 ordinary T2I saved-asset smoke" });
    fact(path.relative(state, await realpath(project.path)).startsWith("data" + path.sep), "project must remain inside isolated app state");
    const body = imageBody(project.id), accepted = await must("POST", "/api/v1/image/jobs", body); assertAccepted(accepted, body); receipt.accepted_job = accepted; await save();
    const job = await wait(() => completedJob(accepted.id), 45 * 60, "ordinary FLUX3 T2I completion");
    const assets = await must("GET", `/api/v1/projects/${project.id}/assets`), asset = assets.find((a) => a.id === job.result?.assetIds?.[0]);
    fact(asset, "completed image must be indexed in the project library");
    const media = await ownedFile(path.join(project.path, asset.file.path), state), sidecarFile = await ownedFile(media.replace(/\.png$/, ".sceneworks.json"), state), sidecar = JSON.parse(await readFile(sidecarFile, "utf8"));
    receipt.saved_asset = assertSavedAsset(job, asset, sidecar, { projectId: project.id, workerId, backend: "mlx", adapter: "mlx_flux3" });
    const response = await endpoint.served(`/api/v1/projects/${project.id}/files/${asset.file.path}`);
    const served = response.bytes, saved = await readFile(media); assertServedPng(served, saved, body.width, body.height);
    await copyFile(media, path.join(evidence, "saved-asset.png")); await writeFile(path.join(evidence, "saved-asset.sceneworks.json"), redactSensitiveText(await readFile(sidecarFile, "utf8"), secrets));
    receipt.completed_job = job; receipt.library_asset = asset; receipt.saved_asset.sha256 = sha256(saved); receipt.saved_asset.bytes = saved.length; receipt.saved_asset.api_status = response.status;
    fact(await git("status", "--porcelain") === "", "runtime changed the immutable checkout"); receipt.status = "passed";
  } catch (error) { receipt.status = "failed"; receipt.error = error.message; throw error; }
  finally {
    clearTimeout(deadline); process.removeListener("SIGTERM", abort); process.removeListener("SIGINT", abort); receipt.cleanup = [];
    for (const child of children.reverse()) {
      if (!child.pid) continue;
      try { if (child.exitCode === null && child.signalCode === null) { process.kill(-child.pid, "SIGTERM"); await Promise.race([child.done.catch(() => {}), new Promise((resolve) => setTimeout(resolve, 15_000))]); if (child.exitCode === null && child.signalCode === null) { process.kill(-child.pid, "SIGKILL"); await child.done.catch(() => {}); } } receipt.cleanup.push({ pid: child.pid, exited: true }); }
      catch { receipt.status = "failed"; receipt.cleanup.push({ pid: child.pid, exited: false }); process.exitCode = 1; }
    }
    await Promise.all(handles.map((handle) => handle.close()));
    endpoint?.close(); controller.signal.removeEventListener("abort", stopOwned);
    // Neither raw nor redacted logs enter artifacts, only the redacted receipt,
    // generated PNG and persisted redacted sidecar.
    receipt.finished_at = new Date().toISOString(); await save();
  }
  fact(receipt.status === "passed", "owned process cleanup failed"); return receipt;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  fact(args.length === 0 || (args.length === 1 && args[0] === "--preflight-only"), "only --preflight-only is supported");
  runLifecycle(process.env, { preflightOnly: args.length === 1 }).then((r) => console.log(`${r.status}: ${r.phase}; ${r.saved_asset ? "saved API library asset" : "no generation performed"}`)).catch((e) => { console.error(e.message); process.exitCode = 1; });
}
