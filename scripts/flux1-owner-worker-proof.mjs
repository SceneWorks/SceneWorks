// Manual, separately dispatched proof of the FLUX.1 native SceneWorks worker boundary.
// It never runs in ordinary CI. All model execution uses the real typed API and MLX worker.
import assert from "node:assert/strict";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { cp, lstat, mkdir, open, readFile, readdir, realpath, rename, stat, statfs, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { fileSha256, sha256 } from "./lib/file-sha256.mjs";
import { assertPins, assertRunnerIdentity, buildChildEnv } from "./qwen-image-2-1-app-lifecycle.mjs";
import {
  assertCheckoutIdentity, assertClaimedWithin, assertDownloadWorker,
  assertServedPng, attemptIdentity, bindOwnedEnv, boundApiUrl,
  ownedApiController, redactSensitiveText, stopOwnedChildren,
} from "./flux3-image-klein-app-lifecycle.mjs";

const execFile = promisify(execFileCallback);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const fact = (condition, message) => assert.ok(condition, message);
const MODELS = Object.freeze([
  { id: "flux_dev", engine: "flux1_dev", repo: "SceneWorks/flux1-dev-mlx", revision: "323fd12d79f78ad444e882e8d8e871914584f2b9", guidance: 3.5, bytes: 9_617_334_683 },
  { id: "flux_schnell", engine: "flux1_schnell", repo: "SceneWorks/flux1-schnell-mlx", revision: "bba3ae01dfd94089f173c05edd4e1a4c551f2599", guidance: 0, bytes: 9_611_551_284 },
]);
const SEEDS = [33111, 33122];
const TIMEOUT_MS = 5 * 60 * 60_000;
const GIB = 1024 ** 3;

async function presentModelBytes(hub, model) {
  const blobs = path.join(hub, `models--${model.repo.replace("/", "--")}`, "blobs");
  let names; try { names = await readdir(blobs); } catch { return 0; }
  let total = 0;
  for (const name of names) { const info = await lstat(path.join(blobs, name)); if (info.isFile()) total += info.size; }
  return Math.min(total, model.bytes);
}

export function assertDiskBudget({ sameVolume, homeFree, outputFree, remainingWeights }) {
  const buildAndEvidence = 48 * GIB; // release target plus attempt state and PNG evidence
  if (sameVolume) fact(homeFree >= remainingWeights + buildAndEvidence, "insufficient shared volume space for pinned q4 sources, release build and state");
  else {
    fact(homeFree >= remainingWeights, "insufficient model-cache volume space for pinned q4 sources");
    fact(outputFree >= buildAndEvidence, "insufficient output volume space for release build and state");
  }
}

export function imageBody(projectId, model, seed, steps = 4) {
  return {
    projectId, mode: "text_to_image", model: model.id,
    prompt: "A red glass apple beside a blue ceramic cup, studio photograph",
    negativePrompt: "", seed, width: 768, height: 768, count: 1,
    advanced: { steps, sampler: "euler", guidanceScale: model.guidance }, loras: [],
  };
}

export function assertModelCatalog(model, expected) {
  fact(model?.id === expected.id && model.adapter === "flux_diffusers", `wrong ${expected.id} catalog entry`);
  const selected = model.downloads?.find((download) => download.default === true);
  fact(selected?.repo === expected.repo && selected.revision === expected.revision && selected.variant === "q4" && selected.estimatedSizeBytes === expected.bytes, `wrong ${expected.id} pinned q4 package`);
  fact(Array.isArray(selected.files) && selected.files.length === 1 && selected.files[0] === "q4/*", `wrong ${expected.id} pinned file closure`);
}

export function assertSerializedTrajectory(isolated, queued, observations) {
  for (const seed of SEEDS) {
    const alone = isolated.find((entry) => entry.seed === seed);
    const together = queued.find((entry) => entry.seed === seed);
    fact(alone?.sha256 && together?.sha256 === alone.sha256, `seed ${seed} changed across isolated and queued native worker runs`);
    fact(alone.worker_id === together.worker_id && alone.backend === "mlx", `seed ${seed} left the owned MLX worker`);
  }
  const first = queued[0], second = queued[1];
  fact(first.seed === SEEDS[0] && second.seed === SEEDS[1], "queued seed order changed");
  fact(observations.some((sample) => sample[first.job_id] === "running" && sample[second.job_id] === "queued"), "second global-seed trajectory did not remain queued while the first ran");
}

export function assertOwnerTrace(models, trace) {
  fact(Array.isArray(trace) && trace.length >= 40, "native owner trace missing");
  const ownerThreads = new Set(), sequence = new Set();
  for (const event of trace) {
    fact(Number.isSafeInteger(event.sequence) && event.sequence > 0 && !sequence.has(event.sequence), "owner trace sequence invalid or repeated");
    fact(/^ThreadId\(\d+\)$/.test(event.owner_thread), "owner thread identity missing");
    sequence.add(event.sequence); ownerThreads.add(event.owner_thread);
  }
  fact(ownerThreads.size === 1, "FLUX.1 trajectories left their single owning thread");
  for (const model of models) {
    const expectedEngine = MODELS.find((entry) => entry.id === model.id)?.engine;
    fact(expectedEngine, "unknown FLUX.1 model in trace");
    const completed = [...model.isolated, ...model.queued, model.active];
    for (const entry of completed) {
      const events = trace.filter((event) => event.job_id === entry.job_id);
      fact(events.length === 4 && events.every((event) => event.engine_id === expectedEngine), `${model.id} owner event set incomplete`);
      assert.deepEqual(events.map((event) => event.phase), ["prepare_begin", "prepare_end", "drive_begin", "drive_end"], `${model.id} owner phases changed`);
      fact(events.every((event, index) => index === 0 || events[index - 1].sequence < event.sequence), "owner event order changed");
    }
    fact(!trace.some((event) => event.job_id === model.cancel.job_id), `${model.id} canceled queued job reached the owner`);
    const firstEnd = trace.find((event) => event.job_id === model.queued[0].job_id && event.phase === "drive_end");
    const secondBegin = trace.find((event) => event.job_id === model.queued[1].job_id && event.phase === "prepare_begin");
    fact(firstEnd.sequence < secondBegin.sequence, `${model.id} queued trajectories interleaved on the owner`);
  }
  fact(trace.length === models.reduce((count, model) => count + (model.isolated.length + model.queued.length + 1) * 4, 0), "unexpected FLUX.1 owner work in isolated proof process");
}

export function parseOwnerTrace(logText) {
  const events = [];
  for (const line of logText.split("\n")) {
    if (!line.trim()) continue;
    let record; try { record = JSON.parse(line); } catch { continue; }
    const fields = record.fields ?? record;
    if (fields.event !== "flux1_owner_turn") continue;
    events.push({ job_id: fields.job_id, engine_id: fields.engine_id, phase: fields.phase, owner_thread: fields.owner_thread, sequence: fields.sequence });
  }
  return events.sort((a, b) => a.sequence - b.sequence);
}

export async function verifyArtifact(output) {
  const evidence = path.join(output, "evidence");
  const receipt = JSON.parse(await readFile(path.join(evidence, "receipt.json"), "utf8"));
  fact(receipt.schema_version === 1 && receipt.kind === "flux1_native_owner_worker_proof" && receipt.status === "passed", "accepted proof receipt required");
  fact(/^[a-f0-9]{40}$/.test(receipt.identity.source_sha) && /^[a-f0-9]{40}$/.test(receipt.identity.inference_sha), "receipt must bind immutable source and inference commits");
  fact(receipt.identity.dispatch_sha === receipt.identity.source_sha && /^[a-f0-9]{64}$/.test(receipt.identity.binary_sha256), "dispatched source and native binary digest required");
  fact(receipt.worker?.id && receipt.worker.gpuId === "mlx" && receipt.worker.capabilities?.includes("gpu"), "owned native MLX worker registration required");
  fact(receipt.models?.length === 2 && receipt.models.every((entry, index) => entry.id === MODELS[index].id), "both FLUX.1 providers required");
  for (const [index, model] of receipt.models.entries()) {
    fact(model.repo === MODELS[index].repo && model.revision === MODELS[index].revision && model.variant === "q4", "pinned model provenance changed");
    fact(/^[a-f0-9]{64}$/.test(receipt.identity[`${model.id}_model_index_sha256`]), "pinned model index digest missing");
    fact(model.isolated?.length === 2 && model.queued?.length === 2 && model.cancel?.status === "canceled", "complete isolated, queued and cancel cases required");
    assertSerializedTrajectory(model.isolated, model.queued, model.observations);
    for (const entry of [...model.isolated, ...model.queued]) {
      fact(entry.worker_id === receipt.worker.id && entry.backend === "mlx" && entry.asset_id && entry.generation_set_id, "saved native worker provenance missing");
      fact(/^[a-z0-9_-]+\.png$/.test(entry.file), "unsafe proof filename");
      fact((await fileSha256(path.join(evidence, entry.file))) === entry.sha256, "saved proof PNG digest mismatch");
    }
  }
  assertOwnerTrace(receipt.models, receipt.owner_trace);
  fact(receipt.cleanup?.every((entry) => entry.exited) && receipt.cleanup.length >= 2, "owned API and worker teardown required");
  return receipt;
}

async function run(env, preflightOnly) {
  fact(env.FLUX1_OWNER_OUTPUT && env.RUNNER_TEMP, "unique FLUX1_OWNER_OUTPUT below RUNNER_TEMP required");
  const output = path.resolve(env.FLUX1_OWNER_OUTPUT);
  const relative = path.relative(path.resolve(env.RUNNER_TEMP), output);
  fact(relative && !relative.startsWith("..") && !path.isAbsolute(relative), "proof output must remain under RUNNER_TEMP");
  const evidence = path.join(output, "evidence"), logs = path.join(output, "private-logs");
  await mkdir(evidence, { recursive: true }); await mkdir(logs, { recursive: true });
  const receipt = { schema_version: 1, kind: "flux1_native_owner_worker_proof", status: "running", identity: { source_sha: env.FLUX1_OWNER_SOURCE_SHA, inference_sha: env.FLUX1_OWNER_INFERENCE_SHA, dispatch_sha: env.GITHUB_SHA, runner_name: env.RUNNER_NAME, runner_target: env.FLUX1_OWNER_RUNNER_TARGET, run_id: env.GITHUB_RUN_ID, run_attempt: env.GITHUB_RUN_ATTEMPT }, models: [], cleanup: [] };
  const secrets = new Set(Object.entries(env).filter(([key, value]) => /TOKEN|SECRET|PASSWORD|API_KEY/i.test(key) && typeof value === "string" && value.length >= 8).map(([, value]) => value));
  const save = async () => { const destination = path.join(evidence, "receipt.json"); await writeFile(`${destination}.tmp`, redactSensitiveText(JSON.stringify(receipt, null, 2) + "\n", secrets)); await rename(`${destination}.tmp`, destination); };
  await save();
  const children = [], handles = [], controller = new AbortController(); let endpoint;
  const abort = () => controller.abort(new Error("FLUX.1 owner proof interrupted"));
  const deadline = setTimeout(() => controller.abort(new Error("FLUX.1 owner proof deadline exceeded")), TIMEOUT_MS);
  controller.signal.addEventListener("abort", () => stopOwnedChildren(children), { once: true });
  process.once("SIGTERM", abort); process.once("SIGINT", abort);
  const alive = () => { if (controller.signal.aborted) throw controller.signal.reason; for (const child of children.filter((entry) => entry.service)) fact(child.exitCode === null && child.signalCode === null, "owned API/worker exited"); };
  const wait = async (probe, seconds, label) => { const until = Date.now() + seconds * 1000; while (Date.now() < until) { alive(); const value = await probe(); if (value) return value; await new Promise((resolve) => setTimeout(resolve, 1000)); } throw new Error(`${label} timed out`); };
  const launch = async (command, args, childEnv, name, service = false) => { const log = await open(path.join(logs, `${name}.log`), "wx"); handles.push(log); const child = spawn(command, args, { cwd: root, env: childEnv, detached: true, stdio: ["ignore", log.fd, log.fd] }); child.service = service; children.push(child); child.done = new Promise((resolve, reject) => { child.once("error", reject); child.once("exit", (code, signal) => resolve({ code, signal })); }); child.done.catch(() => controller.abort(new Error(`${name} launch failed`))); if (service) child.once("exit", () => controller.abort(new Error("owned API/worker exited"))); return child; };
  const git = async (...args) => (await execFile("git", args, { cwd: root, timeout: 30_000 })).stdout.trim();
  try {
    assertCheckoutIdentity({ source: env.FLUX1_OWNER_SOURCE_SHA, inference: env.FLUX1_OWNER_INFERENCE_SHA, dispatch: env.GITHUB_SHA, head: await git("rev-parse", "HEAD"), dirty: await git("status", "--porcelain") });
    assertRunnerIdentity(env.FLUX1_OWNER_RUNNER_TARGET, env.RUNNER_NAME);
    fact(process.platform === "darwin" && process.arch === "arm64", "native Apple Silicon worker required");
    const manifests = await Promise.all(["Cargo.toml", "crates/sceneworks-worker/Cargo.toml", "crates/sceneworks-memory-adapter/Cargo.toml"].map((file) => readFile(path.join(root, file), "utf8")));
    assertPins(manifests, await readFile(path.join(root, "Cargo.lock"), "utf8"), env.FLUX1_OWNER_INFERENCE_SHA);
    const hubPath = path.join(env.HOME, "sceneworks-flux1-owner-proof-weights", "hub");
    const remainingWeights = (await Promise.all(MODELS.map(async (model) => model.bytes - await presentModelBytes(hubPath, model)))).reduce((sum, bytes) => sum + bytes, 0);
    const free = async (directory) => { const fs = await statfs(directory, { bigint: true }); return Number(fs.bavail * fs.bsize); };
    const sameVolume = (await stat(env.HOME)).dev === (await stat(output)).dev;
    const homeFree = await free(env.HOME), outputFree = await free(output);
    assertDiskBudget({ sameVolume, homeFree, outputFree, remainingWeights });
    receipt.identity.disk_budget = { same_volume: sameVolume, remaining_model_bytes: remainingWeights, home_free_bytes: homeFree, output_free_bytes: outputFree };
    for (const model of MODELS) {
      const response = await fetch(`https://huggingface.co/${model.repo}/resolve/${model.revision}/q4/model_index.json`, { signal: AbortSignal.timeout(30_000), redirect: "follow" });
      fact(response.ok, `${model.id} pinned public q4 source unavailable (HTTP ${response.status})`);
      const bytes = Buffer.from(await response.arrayBuffer()); fact(bytes.length > 20, `${model.id} pinned source is empty`);
      receipt.identity[`${model.id}_model_index_sha256`] = sha256(bytes);
    }
    if (preflightOnly) { receipt.status = "preflight_passed"; return receipt; }
    const identity = attemptIdentity(output), { state, workerId } = identity;
    receipt.identity.attempt_nonce = identity.nonce;
    await mkdir(path.join(state, "data", "projects"), { recursive: true }); await mkdir(path.join(state, "credentials")); await cp(path.join(root, "config"), path.join(state, "config"), { recursive: true });
    const weights = path.join(env.HOME, "sceneworks-flux1-owner-proof-weights"), hub = path.join(weights, "hub"), xet = path.join(weights, "xet");
    await mkdir(hub, { recursive: true }); await mkdir(xet, { recursive: true });
    fact(await realpath(weights) === weights && await realpath(hub) === hub, "physical task-owned model cache required");
    let childEnv = bindOwnedEnv(buildChildEnv(env, { state, weights, hub, xet, workerId, url: "http://127.0.0.1:0" }), identity);
    delete childEnv.HF_HUB_OFFLINE; delete childEnv.TRANSFORMERS_OFFLINE;
    const target = path.join(output, "target");
    const build = await launch("cargo", ["build", "--release", "--locked", "-p", "sceneworks-rust-api"], { ...env, CARGO_TARGET_DIR: target }, "build");
    fact((await build.done).code === 0, "native API/worker build failed");
    fact(await git("status", "--porcelain") === "", "build changed immutable checkout");
    const binary = path.join(target, "release", "sceneworks-rust-api"); receipt.identity.binary_sha256 = await fileSha256(binary);
    const api = await launch(binary, [], childEnv, "api", true);
    const url = await wait(async () => boundApiUrl(await readFile(path.join(logs, "api.log"), "utf8")), 180, "owned API bound port");
    childEnv = bindOwnedEnv(childEnv, identity, url);
    endpoint = ownedApiController({ url, state, api, controller });
    await wait(() => endpoint.verify(), 180, "owned API health and canonical state");
    await endpoint.startWorker(() => launch(binary, [], { ...childEnv, SCENEWORKS_WORKER_ONLY: "1" }, "worker", true));
    const must = (...args) => endpoint.call(...args);
    receipt.worker = await wait(async () => { const worker = (await must("GET", "/api/v1/workers")).find((row) => row.id === workerId); if (worker?.status !== "idle") return false; fact(worker.gpuId === "mlx" && worker.capabilities?.includes("gpu"), "owned MLX worker registration required"); return { id: worker.id, gpuId: worker.gpuId, capabilities: worker.capabilities }; }, 180, "owned worker registration");
    await wait(async () => { try { return assertDownloadWorker(await must("GET", "/api/v1/workers"), workerId); } catch { return false; } }, 180, "owned CPU utility downloader");
    const relocation = await must("POST", "/api/v1/model-library/relocate", { path: weights });
    fact(relocation.adopted === true && relocation.hfHome === weights, "model library relocation failed");
    const project = await must("POST", "/api/v1/projects", { name: "FLUX.1 native owner proof" });
    fact(path.relative(state, await realpath(project.path)).startsWith(`data${path.sep}`), "project escaped owned state");
    const getJob = (id) => must("GET", `/api/v1/jobs/${id}`);
    const firstSeen = new Map();
    const observe = async (id, observations) => { const job = await getJob(id); if (!firstSeen.has(id)) firstSeen.set(id, Date.now()); assertClaimedWithin(job, firstSeen.get(id), Date.now()); observations?.push({ at_ms: Date.now(), id, status: job.status, stage: job.stage, worker_id: job.workerId }); return job; };
    for (const model of MODELS) {
      const selected = (await must("GET", "/api/v1/models")).find((row) => row.id === model.id);
      assertModelCatalog(selected, model);
      const download = await must("POST", `/api/v1/models/${model.id}/download`, { licenseAcknowledged: true });
      fact(download.type === "model_download" && download.payload?.repo === model.repo && download.payload?.revision === model.revision, "ordinary pinned model download required");
      await wait(async () => { const job = await observe(download.id); fact(!["failed", "canceled"].includes(job.status), `${model.id} download failed`); return job.status === "completed" && job; }, 3600, `${model.id} pinned download`);
      const installed = (await must("GET", "/api/v1/models")).find((row) => row.id === model.id);
      fact(installed?.installState === "installed" && installed.installedPath, `${model.id} pinned q4 installation required`);
      const snapshot = path.join(hub, `models--${model.repo.replace("/", "--")}`, "snapshots", model.revision);
      const physicalSnapshot = await realpath(snapshot), physicalInstall = await realpath(installed.installedPath);
      fact(physicalInstall === physicalSnapshot || physicalInstall === path.join(physicalSnapshot, "q4"), `${model.id} selected source escaped the pinned snapshot`);
      fact(await fileSha256(path.join(snapshot, "q4", "model_index.json")) === receipt.identity[`${model.id}_model_index_sha256`], `${model.id} installed model index differs from pinned public source`);
      const record = { id: model.id, repo: model.repo, revision: model.revision, variant: "q4", isolated: [], queued: [], observations: [] };
      receipt.models.push(record); await save();
      const complete = async (accepted, events, label) => wait(async () => { const job = await observe(accepted.id, events); fact(!["failed", "canceled"].includes(job.status), `${label} failed`); return job.status === "completed" && job; }, 3600, label);
      const image = async (job, body, suffix) => {
        fact(job.status === "completed" && job.stage === "completed" && job.error == null && job.backend === "mlx" && job.workerId === workerId && job.result?.model === model.id && job.result?.assetIds?.length === 1 && job.result?.assets?.length === 1, "completed native worker/image result required");
        const asset = (await must("GET", `/api/v1/projects/${project.id}/assets`)).find((row) => row.id === job.result.assetIds[0]);
        fact(asset?.file?.path && asset.file.mimeType === "image/png" && asset.projectId === project.id && asset.generationSetId === job.result.generationSetId, "saved project-library PNG and generation set required");
        fact(asset.recipe?.model === body.model && asset.recipe?.seed === body.seed && asset.recipe?.mode === body.mode, "saved native recipe provenance changed");
        const media = path.resolve(project.path, asset.file.path);
        fact(path.relative(project.path, media) && !path.relative(project.path, media).startsWith(".."), "saved media escaped project");
        const sidecar = JSON.parse(await readFile(media.replace(/\.png$/, ".sceneworks.json"), "utf8"));
        assert.deepEqual(job.result.assets[0], sidecar, "worker result and persisted asset sidecar differ");
        const bytes = await readFile(media), response = await endpoint.served(`/api/v1/projects/${project.id}/files/${asset.file.path}`);
        assertServedPng(response.bytes, bytes, body.width, body.height);
        const file = `${model.id}-${suffix}.png`; await writeFile(path.join(evidence, file), bytes);
        return { job_id: job.id, seed: body.seed, worker_id: job.workerId, backend: job.backend, asset_id: asset.id, generation_set_id: asset.generationSetId, file, sha256: sha256(bytes), completed_at_ms: Date.now() };
      };
      for (const seed of SEEDS) {
        const body = imageBody(project.id, model, seed), accepted = await must("POST", "/api/v1/image/jobs", body);
        fact(accepted.type === "image_generate" && accepted.payload?.seed === seed, "typed image job did not preserve seed");
        const job = await complete(accepted, null, `${model.id} isolated seed ${seed}`);
        record.isolated.push(await image(job, body, `isolated-${seed}`)); await save();
      }
      const bodies = SEEDS.map((seed) => imageBody(project.id, model, seed));
      // Submit in a declared order; concurrent HTTP arrival order is unspecified.
      const queued = [];
      for (const body of bodies) queued.push(await must("POST", "/api/v1/image/jobs", body));
      fact(queued.every((job, index) => job.type === "image_generate" && job.payload?.seed === SEEDS[index]), "typed queued seeds changed");
      const states = new Map(queued.map((job) => [job.id, { completed: null }]));
      await wait(async () => {
        const snapshot = {};
        for (const job of queued) {
          const current = await observe(job.id); fact(!["failed", "canceled"].includes(current.status), "queued native render failed");
          snapshot[job.id] = current.status;
          const state = states.get(job.id);
          if (current.status === "completed" && state.completed === null) state.completed = current;
        }
        record.observations.push(snapshot); await save();
        return [...states.values()].every((state) => state.completed);
      }, 3600, `${model.id} queued trajectories`);
      for (let index = 0; index < queued.length; index++) {
        const state = states.get(queued[index].id);
        const entry = await image(state.completed, bodies[index], `queued-${SEEDS[index]}`);
        record.queued.push(entry);
      }
      assertSerializedTrajectory(record.isolated, record.queued, record.observations); await save();
      const cancelBody = imageBody(project.id, model, 33144, 8);
      const active = await must("POST", "/api/v1/image/jobs", imageBody(project.id, model, 33133, 8));
      await wait(async () => (await observe(active.id))?.status === "running", 300, "active owner render before cancel");
      const cancel = await must("POST", "/api/v1/image/jobs", cancelBody);
      await must("POST", `/api/v1/jobs/${cancel.id}/cancel`, {});
      const canceled = await wait(async () => { const job = await observe(cancel.id); fact(job.status !== "completed" && job.status !== "failed", "queued cancellation did not stop the job"); return job.status === "canceled" && job; }, 300, "queued cancellation");
      await complete(active, null, "active render after queued cancellation");
      record.active = { job_id: active.id };
      record.cancel = { job_id: canceled.id, status: canceled.status, worker_id: canceled.workerId ?? null }; await save();
    }
    receipt.owner_trace = parseOwnerTrace(await readFile(path.join(logs, "worker.log"), "utf8"));
    assertOwnerTrace(receipt.models, receipt.owner_trace);
    fact(await git("status", "--porcelain") === "", "native proof changed immutable checkout");
    receipt.status = "passed";
  } catch (error) { receipt.status = "failed"; receipt.error = error.message; throw error; }
  finally {
    clearTimeout(deadline); process.removeListener("SIGTERM", abort); process.removeListener("SIGINT", abort);
    for (const child of children.reverse()) {
      if (!child.pid) continue;
      try { if (child.exitCode === null && child.signalCode === null) { process.kill(-child.pid, "SIGTERM"); await Promise.race([child.done.catch(() => {}), new Promise((resolve) => setTimeout(resolve, 15_000))]); if (child.exitCode === null && child.signalCode === null) { process.kill(-child.pid, "SIGKILL"); await child.done.catch(() => {}); } } receipt.cleanup.push({ pid: child.pid, exited: true }); }
      catch { receipt.status = "failed"; receipt.cleanup.push({ pid: child.pid, exited: false }); process.exitCode = 1; }
    }
    await Promise.all(handles.map((handle) => handle.close())); endpoint?.close();
    receipt.finished_at = new Date().toISOString(); await save();
  }
  if (!preflightOnly) await verifyArtifact(output);
  return receipt;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const mode = args[0] ?? "run";
  fact(args.length <= 1 && ["run", "--preflight-only", "--verify-artifact"].includes(mode), "supported modes: run, --preflight-only, --verify-artifact");
  const task = mode === "--verify-artifact" ? verifyArtifact(process.env.FLUX1_OWNER_OUTPUT) : run(process.env, mode === "--preflight-only");
  task.then((receipt) => console.log(`${receipt.status}: ${receipt.kind}`)).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
