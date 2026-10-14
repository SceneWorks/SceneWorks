#!/usr/bin/env node
// Owned, fail-closed Actions entrypoint. GPU execution remains exclusively in
// h3-veda-acceptance.mjs and its physical guardian.
import { spawn, execFileSync } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { createServer } from "node:net";
import { readFile, writeFile, appendFile, mkdir, lstat, copyFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { artifactsForHost, boundFile, catalogs, declaredRows, digest, hashFile, inspectSources, validateManifest } from "./h3-veda-acceptance.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HEAD = /^[a-f0-9]{40}$/;
const HEX = /^[a-f0-9]{64}$/;
const check = (ok, why) => { if (!ok) throw new Error(`H3 S9 Actions: ${why}`); };
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const git = (root, ...args) => execFileSync("git", ["-C", root, ...args], { encoding: "utf8", timeout: 15000 }).trim();
const absolute = (p) => typeof p === "string" && path.isAbsolute(p);
const inside = (root, p) => { const r = path.relative(root, p); return r && r !== ".." && !r.startsWith(`..${path.sep}`) && !path.isAbsolute(r); };

export async function emitOwnedApiToken({ envFile = process.env.GITHUB_ENV, stdout = process.stdout, random = randomBytes } = {}) {
  check(process.env.GITHUB_ACTIONS === "true" && absolute(envFile), "owned Actions environment file required");
  const token = random(32).toString("hex");
  check(HEX.test(token), "private API token generation failed");
  stdout.write(`::add-mask::${token}\n`);
  await appendFile(envFile, `SCENEWORKS_TOKEN=${token}\n`, { encoding: "utf8" });
}

export async function requirePreviousJobClosed(runDir) {
  const control = await lstat(path.join(runDir, "physical-guardian/control.json")).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
  const closed = await lstat(path.join(runDir, "physical-guardian/closed.json")).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
  check(!control || (control.isFile() && !control.isSymbolicLink() && closed?.isFile() && !closed.isSymbolicLink()), "retained guardian uses a previous job API token; reconcile its owned process and physical lock before a new job");
}

export function probeReadAuthority({ backend, runnerName, sceneWorksSha, runId = process.env.GITHUB_RUN_ID, attempt = process.env.GITHUB_RUN_ATTEMPT, exec = execFileSync } = {}) {
  check(process.env.GITHUB_ACTIONS === "true" && process.env.GH_TOKEN, "existing runner-read credential unavailable");
  check(process.env.GITHUB_REPOSITORY === "SceneWorks/SceneWorks" && process.env.GITHUB_EVENT_NAME === "workflow_dispatch", "Actions owner repository/event differs");
  check(process.env.GITHUB_SHA === sceneWorksSha && process.env.GITHUB_WORKFLOW_REF?.startsWith("SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@") && process.env.GITHUB_JOB === (backend === "mlx" ? "mac2" : "windows_cuda") && process.env.RUNNER_NAME === runnerName, "Actions owner workflow/source/job/runner differs");
  check(/^\d+$/.test(runId ?? "") && /^\d+$/.test(attempt ?? ""), "live Actions run/attempt required");
  const read = (endpoint, paginated = false) => {
    try {
      const args = paginated ? ["api", "--paginate", "--slurp", endpoint] : ["api", endpoint];
      return JSON.parse(exec("gh", args, { encoding: "utf8", timeout: 15000, stdio: ["ignore", "pipe", "pipe"] }));
    } catch { throw new Error(`H3 S9 Actions: existing credential cannot read ${endpoint.split("?")[0]}`); }
  };
  const run = read(`repos/SceneWorks/SceneWorks/actions/runs/${runId}`);
  const pages = read(`repos/SceneWorks/SceneWorks/actions/runs/${runId}/jobs?per_page=100`, true);
  const runners = read("orgs/SceneWorks/actions/runners?per_page=100", true);
  const lensRunning = read("repos/michaeltrefry/LensWorks/actions/runs?status=in_progress&per_page=100", true);
  const lensQueued = read("repos/michaeltrefry/LensWorks/actions/runs?status=queued&per_page=100", true);
  check(run?.id === Number(runId) && run.run_attempt === Number(attempt) && run.head_sha === sceneWorksSha && run.status === "in_progress" && run.conclusion === null && run.event === "workflow_dispatch" && run.name === "H3 Veda terminal owned campaign", "live workflow readback differs");
  const matchingJobs = pages.flatMap((page) => page.jobs ?? []).filter((job) => job.run_id === Number(runId) && job.run_attempt === Number(attempt) && job.head_sha === sceneWorksSha && job.name === process.env.GITHUB_JOB && job.runner_name === runnerName && job.status === "in_progress" && job.conclusion === null);
  check(matchingJobs.length === 1 && Number.isSafeInteger(matchingJobs[0].runner_id), "live job/runner readback missing or ambiguous");
  check(runners.flatMap((page) => page.runners ?? []).filter((runner) => runner.id === matchingJobs[0].runner_id && runner.name === runnerName && runner.status === "online" && runner.busy === true).length === 1, "organization runner readback missing or ambiguous");
  check([lensRunning, lensQueued].every((pages) => Array.isArray(pages) && pages.every((page) => Array.isArray(page.workflow_runs))), "LensWorks queue readback malformed");
  return { runId: Number(runId), jobId: matchingJobs[0].id, runnerId: matchingJobs[0].runner_id, runnerName, readable: true };
}

export function assertActionsIdentity({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, manifest, runDir, cacheRoot }) {
  check(["mlx", "candle_cuda"].includes(backend), "explicit backend required");
  check((backend === "mlx" ? expectedRunner === "nax-macos-2" : ["cuda-windows", "cuda-windows-2"].includes(expectedRunner)) && runnerName === expectedRunner, "runner name differs from admitted physical host");
  check(HEAD.test(sceneWorksSha) && HEAD.test(inferenceSha), "exact source heads required");
  check(manifest?.sourceFreeze?.sceneworks?.head === sceneWorksSha && manifest.sourceFreeze.inferencePin === inferenceSha && manifest.sourceFreeze.inference?.head === inferenceSha, "manifest source/pin differs from dispatch");
  check(manifest.sourceFreeze.sceneworks.root === ROOT, "manifest SceneWorks root must be the Actions checkout");
  check(absolute(runDir) && absolute(cacheRoot) && runDir !== ROOT && cacheRoot !== ROOT && !inside(ROOT, runDir) && !inside(ROOT, cacheRoot), "external run/cache roots required");
  const g = manifest.runners?.[backend]?.guardian;
  check(g && inside(runDir, g.dataDir) && inside(runDir, g.configDir), "guardian data/config must be task-owned under exact run directory");
  check(manifest.runners[backend].apiUrl === `http://127.0.0.1:${g.port}`, "private loopback API binding required");
  return g;
}

export async function preflight({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha }) {
  check(absolute(manifestPath), "absolute frozen manifest input required");
  check(process.env.GITHUB_ACTIONS === "true" && process.env.GH_TOKEN, "source-bound Actions execution and scoped live runner token required");
  const m = await json(manifestPath);
  const g = assertActionsIdentity({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, manifest: m, runDir, cacheRoot });
  check(git(ROOT, "rev-parse", "HEAD") === sceneWorksSha && git(ROOT, "status", "--porcelain") === "", "SceneWorks checkout dirty/stale");
  check(git(m.sourceFreeze.inference.root, "rev-parse", "HEAD") === inferenceSha && git(m.sourceFreeze.inference.root, "status", "--porcelain") === "", "inference checkout dirty/stale");
  check(git(m.sourceFreeze.inference.root, "rev-parse", "refs/remotes/origin/main") === inferenceSha, "inference pin is not fetched permanent main");
  const cache = await lstat(cacheRoot); check(cache.isDirectory() && !cache.isSymbolicLink(), "offline cache unavailable/unsafe");
  const sw = await lstat(runDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
  check(!sw || (sw.isDirectory() && !sw.isSymbolicLink()), "run directory unsafe");
  const active = m.runners[backend], built = path.join(ROOT, "target", "release", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api");
  check(active.apiBinary?.path === built && active.workerBinary?.path === built && active.apiBinary?.sha256 === active.workerBinary?.sha256, "API/worker must be the same source-built normal product binary");
  for (const binding of [active.apiBinary, active.workerBinary]) {
    check(absolute(binding?.path) && HEX.test(binding.sha256) && await hashFile(binding.path) === binding.sha256, "normal product binary identity differs");
  }
  check(m.artifacts?.length > 0 && m.artifactInventory && Object.keys(m.turboArtifacts ?? {}).length > 0 && m.styles?.length > 0, "complete model/style closure missing");
  for (const binding of [...artifactsForHost(m.artifacts, backend), ...Object.values(m.turboArtifacts), ...m.styles]) {
    check(inside(cacheRoot, binding.path), "model/style closure outside explicit offline cache");
    await boundFile(binding);
  }
  for (const binding of [m.artifactInventory, ...m.styles.map((s) => s.compatibilityEvidence)]) await boundFile(binding);
  for (const relative of ["scripts/h3-veda-actions.mjs", ".github/workflows/h3-veda-terminal.yml"]) {
    check(HEX.test(m.sourceFreeze.files?.[relative]) && await hashFile(path.join(ROOT, relative)) === m.sourceFreeze.files[relative], `Actions source omitted from freeze: ${relative}`);
  }
  return { manifest: m, guardian: g };
}

export async function sealActiveBinary({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha }) {
  check(absolute(manifestPath), "absolute frozen manifest template required");
  const m = await json(manifestPath);
  assertActionsIdentity({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, manifest: m, runDir, cacheRoot });
  const existing = await lstat(runDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
  check(!existing || (existing.isDirectory() && !existing.isSymbolicLink()), "owned run directory unsafe");
  await inspectSources(m);
  const binary = path.join(ROOT, "target", "release", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api");
  const info = await lstat(binary); check(info.isFile() && !info.isSymbolicLink() && info.size > 0, "source-built product binary unavailable");
  const binding = { path: binary, bytes: info.size, sha256: await hashFile(binary) };
  m.runners[backend].apiBinary = binding; m.runners[backend].workerBinary = { ...binding };
  const attestation = { schemaVersion: 1, method: "actions_source_build_same_binary_api_and_worker", inferencePin: inferenceSha, sceneworksHead: sceneWorksSha, apiBinarySha256: binding.sha256, workerBinarySha256: binding.sha256, workerId: m.runners[backend].workerId };
  await mkdir(runDir, { recursive: true });
  const attestationPath = path.join(runDir, `${backend}-product-binary-attestation.json`);
  await writeFile(attestationPath, JSON.stringify(attestation, null, 2) + "\n", { flag: "wx" });
  const a = await lstat(attestationPath); m.runners[backend].attestation = { path: attestationPath, bytes: a.size, sha256: await hashFile(attestationPath) };
  const sealedPath = path.join(runDir, "manifest.active-sealed.json");
  await writeFile(sealedPath, JSON.stringify(m, null, 2) + "\n", { flag: "wx" });
  return { sealedPath, binarySha256: binding.sha256, attestationSha256: m.runners[backend].attestation.sha256 };
}

function safeCorpusPath(root, relative) {
  check(typeof relative === "string" && relative.length > 0 && !path.isAbsolute(relative) && !relative.split(/[\\/]/).includes(".."), "unsafe corpus relative path");
  const p = path.resolve(root, relative); check(inside(root, p), "corpus escapes staging root"); return p;
}

async function request(url, init = {}) {
  const response = await fetch(url, { ...init, signal: AbortSignal.timeout(30000) });
  const body = await response.json(); check(response.ok, `private product API HTTP ${response.status}`); return body;
}

async function requireFreeLoopbackPort(port) {
  await new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", () => reject(new Error("H3 S9 Actions: private API port already claimed")));
    server.listen(port, "127.0.0.1", () => server.close(resolve));
  });
}

async function stopOwned(child) {
  if (child.exitCode !== null || !child.pid) return;
  const ended = new Promise((resolve) => child.once("exit", resolve));
  child.kill("SIGTERM");
  const stopped = await Promise.race([ended.then(() => true), delay(15000).then(() => false)]);
  check(stopped, "private import API still alive; preserve owned workspace for inspection");
}

export async function importOwnedCorpus({ manifest, plan, planPath, corpusRoot, runDir, cacheRoot, backend, token, spawnApi = spawn }) {
  check(token && !/\s/.test(token), "private product API token absent");
  const g = manifest.runners[backend].guardian;
  check(!await lstat(g.dataDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "owned data directory already exists; no implicit reuse");
  check(!await lstat(g.configDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "owned config directory already exists; no implicit reuse");
  await requireFreeLoopbackPort(g.port);
  await mkdir(g.dataDir, { recursive: true }); await mkdir(path.join(g.configDir, "manifests"), { recursive: true });
  for (const name of ["builtin.models.jsonc", "builtin.loras.jsonc"]) await copyFile(path.join(ROOT, "config/manifests", name), path.join(g.configDir, "manifests", name));
  const env = { ...process.env, SCENEWORKS_DATA_DIR: g.dataDir, SCENEWORKS_JOBS_DB_PATH: path.join(g.dataDir, "cache/jobs.db"), SCENEWORKS_CONFIG_DIR: g.configDir, SCENEWORKS_OWN_MANIFESTS: "1", SCENEWORKS_API_HOST: "127.0.0.1", SCENEWORKS_API_PORT: String(g.port), SCENEWORKS_API_URL: manifest.runners[backend].apiUrl, SCENEWORKS_ACCESS_TOKEN: token, SCENEWORKS_GPU_ID: "cpu", SCENEWORKS_RUST_WORKER_GPU_ID: "cpu", SCENEWORKS_UTILITY_JOBS: "0", HF_HOME: cacheRoot, HUGGINGFACE_HUB_CACHE: path.join(cacheRoot, "hub"), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" };
  for (const key of ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "GH_TOKEN", "GITHUB_TOKEN", "SCENEWORKS_TOKEN"]) delete env[key];
  const child = spawnApi(manifest.runners[backend].apiBinary.path, [], { cwd: ROOT, env, stdio: "ignore" });
  let launchError; child.on("error", (error) => { launchError = error; });
  try {
    const base = manifest.runners[backend].apiUrl, headers = { "X-SceneWorks-Token": token };
    let ready = false;
    for (let i = 0; i < 60; i++) {
      check(!launchError && child.exitCode === null, "private import API exited before readiness");
      try { await request(new URL("/api/v1/workers", base), { headers }); ready = true; break; } catch { await delay(500); }
    }
    check(ready, "private import API readiness deadline exceeded");
    const project = await request(new URL("/api/v1/projects", base), { method: "POST", headers: { ...headers, "content-type": "application/json" }, body: JSON.stringify({ name: `H3 S9 ${manifest.campaignId}` }) });
    check(/^project_[a-f0-9]+$/.test(project.id ?? ""), "owned project ID missing");
    const assets = {};
    check(Object.keys(plan.assets ?? {}).length === 33, "all 33 approved corpus carriers required");
    const normalizedRoot = path.join(runDir, "normalized-carriers"); await mkdir(normalizedRoot, { recursive: true });
    for (const [key, item] of Object.entries(plan.assets)) {
      check(/^[a-z0-9_-]+$/.test(key) && ["image", "video", "audio"].includes(item.type), "corpus key/type invalid");
      const input = safeCorpusPath(corpusRoot, item.file), meta = await lstat(input);
      check(meta.isFile() && !meta.isSymbolicLink() && meta.size === item.bytes && await hashFile(input) === item.sha256, `staged corpus bytes differ: ${key}`);
      const bytes = await readFile(input), form = new FormData();
      const mime = item.type === "image" ? "image/png" : item.type === "video" ? "video/mp4" : "audio/wav";
      form.append("file", new Blob([bytes], { type: mime }), path.basename(input));
      const created = await request(new URL(`/api/v1/projects/${project.id}/assets`, base), { method: "POST", headers, body: form });
      check(/^asset_[a-f0-9]+$/.test(created.id ?? ""), `product import ID missing: ${key}`);
      const asset = await request(new URL(`/api/v1/projects/${project.id}/assets/${created.id}`, base), { headers });
      check(asset.id === created.id && asset.projectId === project.id && asset.type === item.type, `product import readback differs: ${key}`);
      const mediaUrl = new URL(asset.url, base);
      check(mediaUrl.origin === new URL(base).origin && mediaUrl.pathname.startsWith(`/api/v1/projects/${project.id}/files/`), "product media URL outside owned project");
      const mediaResponse = await fetch(mediaUrl, { headers, redirect: "error", signal: AbortSignal.timeout(300000) }); check(mediaResponse.ok && mediaResponse.body, `product media readback failed: ${key}`);
      const chunks = []; let servedBytes = 0;
      for await (const chunk of mediaResponse.body) { servedBytes += chunk.length; check(servedBytes <= item.bytes, `product served bytes exceed approved corpus: ${key}`); chunks.push(chunk); }
      const served = Buffer.concat(chunks);
      check(served.length === item.bytes && sha(served) === item.sha256, `product served bytes differ from approved corpus: ${key}`);
      const normalized = path.join(normalizedRoot, `${key}${path.extname(input)}`); await writeFile(normalized, served);
      assets[key] = { assetId: asset.id, path: input, bytes: item.bytes, sha256: item.sha256, servedBytes: served.length, servedSha256: sha(served), normalized: { path: normalized, bytes: served.length, sha256: sha(served) } };
    }
    const receipt = { projectId: project.id, planSha256: await hashFile(planPath), assets };
    const receiptPath = path.join(runDir, "corpus-import-receipt.json"); await writeFile(receiptPath, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx" });
    return { receipt, receiptPath };
  } finally { await stopOwned(child); }
}

export async function prepare({ manifestPath, planPath, corpusRoot, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha, token }) {
  check(absolute(corpusRoot) && corpusRoot !== ROOT && !inside(ROOT, corpusRoot), "external staged corpus root required");
  const staged = await lstat(corpusRoot); check(staged.isDirectory() && !staged.isSymbolicLink(), "staged corpus directory unavailable/unsafe");
  const sealed = await sealActiveBinary({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha });
  const { manifest } = await preflight({ manifestPath: sealed.sealedPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha });
  const plan = await json(planPath);
  check(path.resolve(planPath) === path.join(ROOT, "config/terminal-evidence/epic-25140-corpus.plan.json"), "corpus plan must be source-owned");
  check(HEX.test(manifest.sourceFreeze.files["config/terminal-evidence/epic-25140-corpus.plan.json"]) && await hashFile(planPath) === manifest.sourceFreeze.files["config/terminal-evidence/epic-25140-corpus.plan.json"], "corpus plan differs from freeze");
  const { receipt, receiptPath } = await importOwnedCorpus({ manifest, plan, planPath, corpusRoot, runDir, cacheRoot, backend, token });
  const { bindCorpusPlan } = await import("./h3-veda-corpus.mjs");
  manifest.corpusPlan = { path: path.resolve(planPath), bytes: (await lstat(planPath)).size, sha256: await hashFile(planPath) };
  manifest.corpusImportReceipt = { path: receiptPath, bytes: (await lstat(receiptPath)).size, sha256: await hashFile(receiptPath) };
  const bound = bindCorpusPlan(manifest, plan, receipt);
  check(bound.projectId === receipt.projectId, "bound project differs from owned import");
  const marker = { campaignId: bound.campaignId, sourceFreezeSha256: digest(bound.sourceFreeze), projectId: bound.projectId };
  await writeFile(path.join(bound.runners[backend].guardian.dataDir, "s9-workspace-owner.json"), JSON.stringify(marker) + "\n", { flag: "wx" });
  const output = path.join(runDir, "manifest.bound.json"); await writeFile(output, JSON.stringify(bound, null, 2) + "\n", { flag: "wx" });
  await validateManifest(bound, { backend });
  return { manifestPath: output, receiptPath, fingerprint: digest(bound), source: { sceneWorksSha, inferenceSha }, backend, runnerName, cpuImportOnly: true };
}

export async function resume({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha }) {
  check(path.resolve(manifestPath) === path.join(runDir, "manifest.bound.json"), "resume must use exact owned bound manifest");
  const { manifest: m, guardian: g } = await preflight({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha });
  const marker = await json(path.join(g.dataDir, "s9-workspace-owner.json"));
  check(digest(marker) === digest({ campaignId: m.campaignId, sourceFreezeSha256: digest(m.sourceFreeze), projectId: m.projectId }), "resume owned workspace marker differs");
  // A new Actions job has a new private API token. Never attach to a retained
  // guardian whose API child still holds the previous job's token.
  await requirePreviousJobClosed(runDir);
  await validateManifest(m, { backend });
  return { manifestPath, fingerprint: digest(m), backend, runnerName, resumed: true };
}

export function assertActionsTimeBudget(run, job, { now = Date.now(), runId, attempt, backend, runnerName, sceneWorksSha } = {}) {
  const expectedJob = backend === "mlx" ? "mac2" : "windows_cuda";
  check(run?.id === Number(runId) && run.run_attempt === Number(attempt) && run.head_sha === sceneWorksSha && run.status === "in_progress" && run.event === "workflow_dispatch", "live workflow admission identity differs");
  check(job?.run_id === run.id && job.run_attempt === run.run_attempt && job.head_sha === sceneWorksSha && job.name === expectedJob && job.runner_name === runnerName && job.status === "in_progress", "live job admission identity differs");
  const queuedAt = Date.parse(run.created_at), startedAt = Date.parse(job.started_at);
  check(Number.isFinite(queuedAt) && Number.isFinite(startedAt) && startedAt >= queuedAt && startedAt - queuedAt <= 30 * 60_000, "Actions queue/admission exceeded 30 minutes");
  // The physical guardian can retain an owned child for one hour; reserve
  // five minutes beyond that inside the 75-minute Actions hard timeout.
  check(now >= startedAt && now - startedAt <= 10 * 60_000, "insufficient owned-child/cleanup time remains in 75-minute job");
}

export async function admitOwnedJob({ backend, runnerName, sceneWorksSha, runId = process.env.GITHUB_RUN_ID, attempt = process.env.GITHUB_RUN_ATTEMPT }) {
  check(process.env.GH_TOKEN, "GH_TOKEN with exact runner/job read scope required");
  check(/^\d+$/.test(runId ?? "") && /^\d+$/.test(attempt ?? ""), "live Actions run/attempt required");
  const repo = "SceneWorks/SceneWorks";
  const run = JSON.parse(execFileSync("gh", ["api", `repos/${repo}/actions/runs/${runId}`], { encoding: "utf8", timeout: 15000 }));
  const pages = JSON.parse(execFileSync("gh", ["api", "--paginate", "--slurp", `repos/${repo}/actions/runs/${runId}/jobs?per_page=100`], { encoding: "utf8", timeout: 15000 }));
  const jobs = pages.flatMap((page) => page.jobs ?? []).filter((job) => job.runner_name === runnerName && job.name === (backend === "mlx" ? "mac2" : "windows_cuda"));
  check(jobs.length === 1, "own Actions job ambiguous/missing");
  assertActionsTimeBudget(run, jobs[0], { runId, attempt, backend, runnerName, sceneWorksSha });
  return { runId: Number(runId), attempt: Number(attempt), jobId: jobs[0].id, runnerName, source: sceneWorksSha, queueMinutes: (Date.parse(jobs[0].started_at) - Date.parse(run.created_at)) / 60000 };
}

const withoutPath = (binding) => { const { path: _path, ...portable } = binding ?? {}; return portable; };
export function approvedCampaignIdentity(m) {
  const freeze = m.sourceFreeze;
  check(HEAD.test(freeze?.sceneworks?.head) && freeze.inference?.head === freeze.inferencePin && HEAD.test(freeze.inferencePin), "approved source heads unavailable");
  check(HEX.test(freeze.files?.["config/manifests/builtin.models.jsonc"]) && HEX.test(freeze.files?.["config/manifests/builtin.loras.jsonc"]) && HEX.test(freeze.files?.["config/terminal-evidence/epic-25140-corpus.plan.json"]), "approved source/catalog/corpus hashes unavailable");
  return digest({
    campaignId: m.campaignId, scope: m.scope,
    source: { stage: freeze.stage, sceneworksHead: freeze.sceneworks.head, inferenceHead: freeze.inference.head, inferencePin: freeze.inferencePin, files: freeze.files, inferenceFiles: freeze.inferenceFiles, artifactInventorySha256: freeze.artifactInventorySha256 },
    corpusPlanSha256: m.corpusPlan?.sha256, artifactInventory: withoutPath(m.artifactInventory),
    artifacts: m.artifacts?.map(withoutPath), turboArtifacts: Object.fromEntries(Object.entries(m.turboArtifacts ?? {}).map(([id, a]) => [id, withoutPath(a)])),
    styles: m.styles?.map((style) => ({ ...withoutPath(style), compatibilityEvidence: withoutPath(style.compatibilityEvidence) })),
    prompts: m.prompts, bindings: m.bindings, boundaryBindings: m.boundaryBindings, proofRoutes: m.proofRoutes,
    productProofCases: m.productProofCases?.map(({ setupReceipt: _hostReceipt, ...proof }) => proof),
  });
}

export function buildHostEvidence(m, campaign, backend, rows, closure, { manifestSha256, campaignSha256 } = {}) {
  check(["mlx", "candle_cuda"].includes(backend) && campaign?.campaignId === m.campaignId && campaign.fingerprint === digest(m), "host campaign/manifest identity differs");
  check(HEX.test(manifestSha256) && HEX.test(campaignSha256), "host manifest/campaign file hashes required");
  const expected = rows.filter((row) => row.backend === backend);
  check(expected.length > 0 && new Set(rows.map((row) => row.id)).size === rows.length, "required row identities missing or duplicate");
  const expectedIds = new Set(expected.map((row) => row.id));
  check(Object.keys(campaign.cells ?? {}).every((id) => expectedIds.has(id)), "host campaign contains foreign or unknown row");
  const runner = m.runners[backend];
  check([runner.apiBinary?.sha256, runner.workerBinary?.sha256, runner.attestation?.sha256, runner.physicalGuardAdapter?.sha256].every((value) => HEX.test(value)), "host binary/attestation/guardian identity incomplete");
  const cases = expected.map((row) => {
    const cell = campaign.cells[row.id];
    return { id: row.id, backend, kind: row.kind, style: row.style ?? null, status: cell?.status ?? "missing", evidenceSha256: cell ? digest(cell) : null };
  });
  return {
    schemaVersion: 1, backend, campaignId: m.campaignId, approvedIdentitySha256: approvedCampaignIdentity(m),
    source: { sceneworksHead: m.sourceFreeze.sceneworks.head, inferencePin: m.sourceFreeze.inferencePin, sourceFilesSha256: digest(m.sourceFreeze.files), catalogModelsSha256: m.sourceFreeze.files["config/manifests/builtin.models.jsonc"], catalogLorasSha256: m.sourceFreeze.files["config/manifests/builtin.loras.jsonc"], corpusPlanSha256: m.corpusPlan.sha256, artifactInventorySha256: m.sourceFreeze.artifactInventorySha256 },
    local: { manifestFingerprint: digest(m), manifestSha256, campaignSha256, projectId: m.projectId, corpusImportReceiptSha256: m.corpusImportReceipt?.sha256, physicalGpuId: runner.physicalGpuId, workerId: runner.workerId, binarySha256: runner.workerBinary.sha256, apiBinarySha256: runner.apiBinary.sha256, attestationSha256: runner.attestation.sha256, physicalGuardSourceSha256: runner.physicalGuardAdapter.sha256 },
    expectedAllRowsSha256: digest(rows.map((row) => ({ id: row.id, kind: row.kind, style: row.style ?? null }))),
    expectedHostRowsSha256: digest(expected.map((row) => ({ id: row.id, kind: row.kind, style: row.style ?? null }))),
    expectedAllRows: rows.length, expectedNamedRows: rows.filter((row) => !row.style && !row.boundary && row.kind !== "product_proof").length,
    cases,
    guardianClosed: closure?.closed === true && closure.servicesDrained === true && closure.physicalLockReleased === true && closure.fingerprint === digest(m),
    qualityAcceptance: "pending Michael", completedAcceptance: false,
  };
}

export function aggregateHostReceipts(left, right) {
  const hosts = [left?.hostEvidence, right?.hostEvidence];
  check(hosts.every((h) => h?.schemaVersion === 1) && new Set(hosts.map((h) => h.backend)).size === 2 && hosts.some((h) => h.backend === "mlx") && hosts.some((h) => h.backend === "candle_cuda"), "exact MLX and CUDA host receipts required");
  const [first, second] = hosts;
  check(first.campaignId === second.campaignId && first.approvedIdentitySha256 === second.approvedIdentitySha256 && HEX.test(first.approvedIdentitySha256) && digest(first.source) === digest(second.source), "approved campaign/recipe/source/catalog identities differ");
  check(first.expectedAllRowsSha256 === second.expectedAllRowsSha256 && first.expectedAllRows === second.expectedAllRows && first.expectedNamedRows === second.expectedNamedRows && first.expectedNamedRows === 5400, "full named/style/boundary/proof row universe differs");
  const cases = hosts.flatMap((h) => {
    check(h.local && [h.local.manifestFingerprint, h.local.manifestSha256, h.local.campaignSha256, h.local.binarySha256, h.local.apiBinarySha256, h.local.attestationSha256, h.local.physicalGuardSourceSha256, h.local.corpusImportReceiptSha256].every((value) => HEX.test(value)), "host source/binary/corpus/campaign receipts incomplete");
    check(h.local.physicalGpuId && h.local.workerId && h.cases?.length && h.cases.every((c) => c.backend === h.backend && c.id && c.kind && typeof c.status === "string" && (c.status === "missing" ? c.evidenceSha256 === null : HEX.test(c.evidenceSha256))), "host row disposition/physical identity incomplete");
    check(HEX.test(h.expectedHostRowsSha256) && digest(h.cases.map((c) => ({ id: c.id, kind: c.kind, style: c.style }))) === h.expectedHostRowsSha256, "host row identity coverage missing/duplicate");
    return h.cases;
  });
  check(cases.length === first.expectedAllRows && new Set(cases.map((c) => c.id)).size === cases.length, "full two-host row identity coverage missing/duplicate");
  check(hosts.every((h) => h.cases.filter((c) => !c.style && !c.id.startsWith("boundary/") && !c.id.startsWith("proof/")).length === 2700), "each host must retain half of the 5400 named rows");
  const missing = cases.filter((c) => c.status === "missing").map((c) => c.id);
  const unresolved = cases.filter((c) => !["expected_refusal_observed", "product_proof_observed", "media_collected_review_pending"].includes(c.status)).map((c) => c.id);
  return {
    schemaVersion: 1, campaignId: first.campaignId, approvedIdentitySha256: first.approvedIdentitySha256,
    hostManifestFingerprints: Object.fromEntries(hosts.map((h) => [h.backend, h.local.manifestFingerprint])),
    hostBinarySha256: Object.fromEntries(hosts.map((h) => [h.backend, h.local.binarySha256])),
    expectedRows: cases.length, observedDispositions: cases.length - missing.length, missing, unresolved,
    bothGuardianClosuresVerified: hosts.every((h) => h.guardianClosed), qualityAcceptance: "pending Michael", completedAcceptance: false,
    requiresRawLocalEvidenceReview: true,
  };
}

export async function redactReceipt({ runDir, output, backend }) {
  const campaign = await json(path.join(runDir, "campaign.json")).catch(() => null);
  const closure = await json(path.join(runDir, "physical-guardian/closed.json")).catch(() => null);
  const summary = campaign && { campaignId: campaign.campaignId, fingerprint: campaign.fingerprint, namedCells: campaign.namedCells, cellStatusCounts: Object.values(campaign.cells ?? {}).reduce((a, c) => { a[c.status] = (a[c.status] ?? 0) + 1; return a; }, {}), qualityAcceptance: campaign.qualityAcceptance, completedAcceptance: campaign.completedAcceptance };
  const guardianClosure = closure && { closed: closure.closed, servicesDrained: closure.servicesDrained, physicalLockReleased: closure.physicalLockReleased, fingerprint: closure.fingerprint, ownerTokenSha256: closure.ownerTokenSha256, pid: closure.pid };
  let hostEvidence = null, hostEvidenceError = null;
  if (campaign && backend) {
    try {
      const manifestPath = path.join(runDir, "manifest.bound.json"), m = await json(manifestPath);
      await inspectSources(m);
      for (const binding of [m.runners[backend].apiBinary, m.runners[backend].workerBinary, m.runners[backend].attestation, m.runners[backend].physicalGuardAdapter]) await boundFile(binding);
      const catalog = await catalogs(m.sourceFreeze.sceneworks.root);
      hostEvidence = buildHostEvidence(m, campaign, backend, declaredRows(m, catalog), closure, { manifestSha256: await hashFile(manifestPath), campaignSha256: await hashFile(path.join(runDir, "campaign.json")) });
    } catch { hostEvidenceError = "host source, selected binary, or campaign receipt unavailable; inspect local raw evidence"; }
  }
  const receipt = { schemaVersion: 1, summary, guardianClosure, hostEvidence, hostEvidenceError, mediaUploaded: false, secretsUploaded: false, requiresRawLocalEvidenceReview: true };
  await mkdir(path.dirname(output), { recursive: true }); await writeFile(output, JSON.stringify(receipt, null, 2) + "\n");
  return receipt;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [command, ...args] = process.argv.slice(2);
  (async () => {
    if (command === "owned-token") {
      check(args.length === 0, "owned-token takes no arguments");
      await emitOwnedApiToken();
    } else if (command === "probe-auth") {
      check(args.length === 1 && ["mlx", "candle_cuda"].includes(args[0]), "probe-auth BACKEND");
      const receipt = probeReadAuthority({ backend: args[0], runnerName: process.env.RUNNER_NAME, sceneWorksSha: process.env.S9_SCENEWORKS_SHA });
      console.log(JSON.stringify(receipt));
    } else if (command === "prepare") {
      const [manifestPath, planPath, corpusRoot, backend, runDir, cacheRoot] = args;
      check(args.length === 6, "prepare MANIFEST PLAN CORPUS_ROOT BACKEND RUN_DIR CACHE_ROOT");
      const receipt = await prepare({ manifestPath, planPath, corpusRoot, backend, runDir, cacheRoot, runnerName: process.env.RUNNER_NAME, expectedRunner: process.env.S9_EXPECTED_RUNNER, sceneWorksSha: process.env.S9_SCENEWORKS_SHA, inferenceSha: process.env.S9_INFERENCE_SHA, token: process.env.SCENEWORKS_TOKEN });
      console.log(JSON.stringify(receipt));
    } else if (command === "resume") {
      const [manifestPath, backend, runDir, cacheRoot] = args;
      check(args.length === 4, "resume BOUND_MANIFEST BACKEND RUN_DIR CACHE_ROOT");
      console.log(JSON.stringify(await resume({ manifestPath, backend, runDir, cacheRoot, runnerName: process.env.RUNNER_NAME, expectedRunner: process.env.S9_EXPECTED_RUNNER, sceneWorksSha: process.env.S9_SCENEWORKS_SHA, inferenceSha: process.env.S9_INFERENCE_SHA })));
    } else if (command === "admit") {
      check(args.length === 1 && ["mlx", "candle_cuda"].includes(args[0]), "admit BACKEND");
      console.log(JSON.stringify(await admitOwnedJob({ backend: args[0], runnerName: process.env.RUNNER_NAME, sceneWorksSha: process.env.S9_SCENEWORKS_SHA })));
    } else if (command === "redact") {
      check(args.length === 3 && ["mlx", "candle_cuda"].includes(args[2]), "redact RUN_DIR OUTPUT BACKEND"); console.log(JSON.stringify(await redactReceipt({ runDir: args[0], output: args[1], backend: args[2] })));
    } else if (command === "aggregate") {
      check(args.length === 3, "aggregate MLX_RECEIPT CUDA_RECEIPT OUTPUT");
      const aggregate = aggregateHostReceipts(await json(args[0]), await json(args[1]));
      await mkdir(path.dirname(args[2]), { recursive: true });
      await writeFile(args[2], JSON.stringify(aggregate, null, 2) + "\n", { flag: "wx" });
      console.log(JSON.stringify({ output: args[2], expectedRows: aggregate.expectedRows, missing: aggregate.missing.length, unresolved: aggregate.unresolved.length, completedAcceptance: false }));
    } else throw new Error("H3 S9 Actions: unknown command");
  })().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
