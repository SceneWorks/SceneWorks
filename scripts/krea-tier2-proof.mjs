import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { link, lstat, mkdir, mkdtemp, open, readFile, realpath, rename, rm, statfs, writeFile } from "node:fs/promises";
import path from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";

import { probeCacheFile } from "./h3-finetune-terminal-stage.mjs";
import { KREA_ALLOCATION_BASE_SIZE, KREA_ALLOCATION_FINAL_SIZE } from "./krea-tier2-allocation.mjs";
import { kreaNativeProgressReceipt } from "./krea-tier2-performance.mjs";
import { fileSha256 } from "./lib/file-sha256.mjs";
import { assertTerminalPhysicalContainment, ensureTerminalPhysicalDirectory } from "./lib/starvector-terminal-pin-paths.mjs";

export const KREA_SUITE = "krea-scoped-acceptance";
export const KREA_BASE_APP = "2aa525dfb2fb1ab889bd6a15f80c0a0cc1df1e2d";
export const KREA_BASE_ENGINE = "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294";
const RAW = { repo: "SceneWorks/krea-2-raw-mlx", revision: "b88090c79762ba34b1de23a892da239eb76962e4" };
export const COMPANION = Object.freeze({ repo: "SceneWorks/krea-2-turbo-mlx", revision: "d009674080cc1bccf2b629d834c34bf5eccdb723" });
const COMFY = { repo: "Comfy-Org/Krea-2", revision: "952f49d49653cb42e7d6cf7cbfad74738073ec7d" };
export const KREA_RAW_ADAPTER = Object.freeze({ ...COMFY, file: "loras/krea2_darkbrush.safetensors", bytes: 469291992, sha256: "f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6" });
export const KREA_RAW_LORA = "krea2_darkbrush_phase_proof";
export const CHECKPOINT = Object.freeze({ ...COMFY, file: "diffusion_models/krea2_turbo_bf16.safetensors", bytes: 26283332608, sha256: "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3" });
export const LINKED_MODEL = "linked_krea2_turbo_bf16_proof";
const PROMPT = "A red glass apple beside a blue ceramic cup, a sharp studio photograph with soft shadows";
const NEGATIVE = "blur, text, watermark";
const PHASES = [{ steps: 3, guidance: 0, loras: [] }, { steps: 4, guidance: 1, loras: [{ index: 0, weight: 0.75 }] }, { steps: 3, guidance: 0, loras: [] }];
const HIRES = { enabled: true, upscaleBy: 1.5, steps: 4, denoisingStrength: 0.5 };
const PROJECTOR = "transformer.text_fusion.projector";
const GIB = 1024 ** 3;
const STAGING_PROGRESS_BYTES = GIB / 4;

export function kreaRoutes(backend, { allocation = false } = {}) {
  const shared = { backend, kind: "image", mode: "text_to_image", group: null, endpoint: "/api/v1/image/jobs", claimMinutes: 5, timeoutMinutes: 20 };
  const body = { mode: shared.mode, prompt: PROMPT, width: 512, height: 512, count: 1, seed: 33602 };
  const routes = [
    { ...shared, id: `${backend}-krea-raw-phases`, model: "krea_2_raw", tier: "q4", proof: "raw", steps: 10, executedSteps: 10, finalSize: 512, body: { ...body, model: "krea_2_raw", negativePrompt: NEGATIVE, loras: [{ id: KREA_RAW_LORA, weight: 1 }], advanced: { mlxQuantize: 4, sampler: "euler", phases: PHASES } } },
    { ...shared, id: `${backend}-krea-imported-bf16-hires`, model: "krea_2_turbo", tier: "bf16", proof: "hires", steps: 12, executedSteps: 10, finalSize: 768, body: { ...body, model: LINKED_MODEL, negativePrompt: "", loras: [], advanced: { mlxQuantize: 0, steps: 8, guidanceScale: 0, sampler: "euler" }, hiresFix: HIRES } },
  ];
  if (!allocation) return routes;
  assert.equal(backend, "cuda", "Krea allocation needs the Candle CUDA worker");
  const hires = routes[1];
  return [{ ...hires, id: "cuda-krea-imported-bf16-hires-allocation", timeoutMinutes: 30, finalSize: KREA_ALLOCATION_FINAL_SIZE, body: { ...hires.body, width: KREA_ALLOCATION_BASE_SIZE, height: KREA_ALLOCATION_BASE_SIZE, hiresFix: { ...HIRES, upscaleBy: 2 } } }];
}

export function assertKreaPins(baseApp, baseEngine, headApp, headEngine) {
  assert.equal(baseApp, KREA_BASE_APP, "Krea baseline app is frozen");
  assert.equal(baseEngine, KREA_BASE_ENGINE, "Krea baseline inference is frozen");
  assert.match(headApp, /^[a-f0-9]{40}$/);
  assert.match(headEngine, /^[a-f0-9]{40}$/);
  assert.notEqual(headApp, baseApp, "candidate app must differ from baseline");
  assert.notEqual(headEngine, baseEngine, "candidate inference must differ from baseline");
}

const keyOf = ({ repo, revision }) => `${repo}@${revision}`;
const treeDirs = ["transformer", "text_encoder", "tokenizer", "scheduler", "vae"];
function tierEntries(evidence, source, tier, sharedOnly) {
  const files = evidence.get(keyOf(source));
  assert.ok(files, `No checked-in file inventory for ${keyOf(source)}`);
  const prefix = `${tier}/`;
  const chosen = files.filter((file) => {
    if (!file.startsWith(prefix)) return false;
    const [, component, name] = file.split("/");
    if (sharedOnly && component === "transformer") return name === "config.json";
    return treeDirs.includes(component) || file === `${tier}/model_index.json`;
  });
  for (const component of treeDirs) assert.ok(chosen.some((file) => file.startsWith(`${prefix}${component}/`)), `missing ${keyOf(source)} ${tier}/${component} listing`);
  return chosen.map((file) => {
    assert.match(file, /^[A-Za-z0-9_-]+\/[A-Za-z0-9_./-]+$/);
    assert.ok(file.split("/").every((part) => part && !part.startsWith(".")), `unsafe inventory path ${file}`);
    return { ...source, file };
  });
}

export function kreaAssetInventory(evidence, routes) {
  return [
    ...(routes.some(({ proof }) => proof === "raw") ? [...tierEntries(evidence, RAW, "q4", false), KREA_RAW_ADAPTER] : []),
    ...(routes.some(({ proof }) => proof === "hires") ? [...tierEntries(evidence, COMPANION, "bf16", true), CHECKPOINT] : []),
  ];
}

function resolveUrl({ repo, revision, file }) {
  return `https://huggingface.co/${repo}/resolve/${revision}/${file.split("/").map(encodeURIComponent).join("/")}`;
}

export async function priceKreaAssets(inventory, signal, fetchImpl = fetch) {
  const trees = new Map();
  const files = [];
  for (const source of inventory) {
    const dir = source.file.split("/")[0];
    const key = `${keyOf(source)}/${dir}`;
    if (!trees.has(key)) {
      const url = `https://huggingface.co/api/models/${source.repo}/tree/${source.revision}/${dir}?recursive=true`;
      const response = await fetchImpl(url, { redirect: "error", signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]) });
      assert.ok(response.ok && !response.headers.get("link"), `incomplete pinned inventory, HTTP ${response.status}: ${key}`);
      trees.set(key, await response.json());
    }
    const row = trees.get(key).find((item) => item.type === "file" && item.path === source.file);
    assert.ok(Number.isSafeInteger(row?.size) && row.size > 0, `missing sized artifact ${keyOf(source)}/${source.file}`);
    if (source.bytes) assert.equal(row.size, source.bytes, `changed length for ${source.file}`);
    if (source.sha256) assert.equal(row.lfs?.oid, source.sha256, `changed LFS digest for ${source.file}`);
    const sha256 = row.lfs?.oid;
    assert.ok(sha256 ? /^[a-f0-9]{64}$/.test(sha256) : /^[a-f0-9]{40}$/.test(row.oid), `missing artifact digest: ${source.file}`);
    files.push({ ...source, bytes: row.size, ...(sha256 ? { sha256 } : { gitBlobSha1: row.oid }) });
  }
  return { bytes: files.reduce((total, item) => total + item.bytes, 0), files };
}

export function kreaSnapshotFile(hub, item) {
  return path.join(hub, `models--${item.repo.replace("/", "--")}`, "snapshots", item.revision, ...item.file.split("/"));
}

async function verifyKreaTransfer(input, output, item, signal, onBytes) {
  const hash = createHash(item.sha256 ? "sha256" : "sha1");
  if (!item.sha256) hash.update(`blob ${item.bytes}\0`);
  let bytes = 0;
  const verify = new Transform({ transform(chunk, _encoding, done) {
    bytes += chunk.length;
    onBytes(bytes);
    if (bytes > item.bytes) return done(new Error(`pinned artifact exceeded its length: ${item.file}`));
    hash.update(chunk);
    done(null, chunk);
  } });
  await pipeline(input, verify, output, { signal });
  assert.equal(bytes, item.bytes, `wrong artifact length: ${item.file}`);
  const digest = hash.digest("hex");
  assert.equal(digest, item.sha256 ?? item.gitBlobSha1, `wrong artifact digest: ${item.file}`);
  return digest;
}

async function publishKreaCacheMiss({ hub, target, item, signal, download, onBytes }) {
  const directory = path.dirname(target);
  await ensureTerminalPhysicalDirectory(hub, directory);
  const partial = path.join(directory, `.krea-${randomUUID()}.part`);
  const file = await open(partial, "wx");
  try {
    const digest = await verifyKreaTransfer(await download(), file.createWriteStream(), item, signal, onBytes);
    signal.throwIfAborted();
    await assertTerminalPhysicalContainment(hub, directory);
    try {
      // The exact snapshot path accepts an ordinary file on Windows. Publish the complete leaf
      // with exclusive creation. A concurrent installer can win without being overwritten.
      await link(partial, target);
      return { digest, cacheDisposition: "published" };
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
      return { digest, cacheDisposition: "existing" };
    }
  } finally {
    try { await file.close(); }
    finally { await rm(partial, { force: true }); }
  }
}

export async function stageKreaAssets(hub, price, signal, { cacheHub, fetchImpl = fetch, progress = () => {} } = {}) {
  await mkdir(hub, { recursive: true });
  if (cacheHub) {
    const [attempt, shared] = await Promise.all([realpath(hub), realpath(cacheHub)]);
    const outside = (root, target) => {
      const relative = path.relative(root, target);
      return relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative);
    };
    assert.ok(outside(shared, attempt) && outside(attempt, shared), "Krea staging requires a separate attempt and canonical hub");
  }
  const fs = await statfs(hub, { bigint: true });
  const freeBytes = Number(fs.bavail * fs.bsize);
  assert.ok(freeBytes >= price.bytes + 20 * GIB, `need ${price.bytes} bytes for pinned Krea assets plus 20 GiB for builds, have ${freeBytes} bytes`);
  const deadline = AbortSignal.any([signal, AbortSignal.timeout((cacheHub ? 180 : 90) * 60_000)]);
  for (const item of price.files) {
    const dest = kreaSnapshotFile(hub, item);
    const partial = `${dest}.part`;
    const identity = `${keyOf(item)}/${item.file}`;
    let source = "cache-probe";
    let bytes = 0;
    let nextProgress = STAGING_PROGRESS_BYTES;
    const report = (status, detail = {}) => progress({ status, identity, source, receivedBytes: bytes, expectedBytes: item.bytes, ...detail });
    const onBytes = (received) => {
      bytes = received;
      if (bytes >= nextProgress) {
        report("transferring");
        nextProgress = bytes + STAGING_PROGRESS_BYTES;
      }
    };
    const download = async () => {
      const response = await fetchImpl(resolveUrl(item), { redirect: "follow", signal: deadline });
      assert.ok(response.ok && response.body, `cannot fetch pinned ${identity} (HTTP ${response.status})`);
      return Readable.fromWeb(response.body);
    };
    try {
      deadline.throwIfAborted();
      const probe = () => probeCacheFile({ ...item, repoType: "model" }, cacheHub, { signal: deadline });
      let cached = cacheHub ? await probe() : null;
      if (cached?.state === "absent") {
        source = "pinned-https";
        report("started");
        report("download-verified", await publishKreaCacheMiss({ hub: cacheHub, target: cached.candidate, item, signal: deadline, download, onBytes }));
        source = "runner-cache";
        bytes = 0;
        nextProgress = STAGING_PROGRESS_BYTES;
        cached = await probe();
        assert.equal(cached.state, "present", `verified canonical snapshot file disappeared: ${item.file}`);
      }
      source = cached ? "runner-cache" : "pinned-https";
      report("started");
      await mkdir(path.dirname(dest), { recursive: true });
      let stream;
      if (cached) {
        assert.equal(cached.size, item.bytes, `wrong cached artifact length: ${item.file}`);
        stream = createReadStream(cached.sourcePath);
      } else {
        stream = await download();
      }
      const digest = await verifyKreaTransfer(stream, createWriteStream(partial, { flags: "wx" }), item, deadline, onBytes);
      await rename(partial, dest);
      report("verified", { digest });
    } catch (error) {
      report("failed", { error: error.message });
      throw new Error(`Krea staging failed for ${identity} after receiving ${bytes} bytes: ${error.message}`, { cause: error });
    } finally {
      await rm(partial, { force: true });
    }
  }
  for (const { repo, revision } of new Map(price.files.map((item) => [item.repo, item])).values()) {
    const refs = path.join(hub, `models--${repo.replace("/", "--")}`, "refs");
    await mkdir(refs, { recursive: true });
    await writeFile(path.join(refs, "main"), revision, { flag: "wx" });
  }
  const receipt = { ...price, freeBytes, stagedFileCount: price.files.length };
  if (price.files.some(isRawAdapter)) receipt.rawAdapter = await readRawAdapterFactors(hub);
  return receipt;
}

function isRawAdapter({ repo, revision, file }) {
  return repo === KREA_RAW_ADAPTER.repo && revision === KREA_RAW_ADAPTER.revision && file === KREA_RAW_ADAPTER.file;
}

export function assertKreaRawAdapterFactors(tensors) {
  const factors = Object.entries(tensors).filter(([name]) => name !== "__metadata__");
  assert.equal(factors.length, 528, "Darkbrush must contain exactly 528 LoRA factors");
  const targets = new Set();
  for (const [name, tensor] of factors) {
    const match = /^(transformer\.[a-zA-Z0-9_.]+)\.lora_([AB])\.weight$/.exec(name);
    assert.ok(match, `Darkbrush has a tensor outside its LoRA factors: ${name}`);
    const [, target, side] = match;
    assert.equal(tensor?.dtype, "F32", `wrong Darkbrush factor dtype: ${name}`);
    assert.ok(Array.isArray(tensor.shape) && tensor.shape.length === 2 && tensor.shape.every((dim) => Number.isSafeInteger(dim) && dim > 0), `wrong Darkbrush factor shape: ${name}`);
    assert.equal(tensor.shape[side === "A" ? 0 : 1], 32, `wrong Darkbrush factor rank: ${name}`);
    const peer = `${target}.lora_${side === "A" ? "B" : "A"}.weight`;
    assert.ok(Object.hasOwn(tensors, peer), `Darkbrush factor without its peer: ${name}`);
    targets.add(target);
  }
  const projectorFactors = [`${PROJECTOR}.lora_A.weight`, `${PROJECTOR}.lora_B.weight`];
  assert.deepEqual(projectorFactors.map((name) => tensors[name]?.shape), [[32, 12], [1, 32]], "Darkbrush projector factors changed");
  return { tensorCount: factors.length, targetCount: targets.size, rank: 32, dtype: "F32", projectorFactors };
}

async function readRawAdapterFactors(hub) {
  const adapter = await open(kreaSnapshotFile(hub, KREA_RAW_ADAPTER), "r");
  try {
    const prefix = Buffer.alloc(8);
    assert.equal((await adapter.read(prefix, 0, prefix.length, 0)).bytesRead, prefix.length);
    const length = Number(prefix.readBigUInt64LE());
    assert.ok(length > 0 && length <= 1024 * 1024, "pinned adapter header length is invalid");
    const header = Buffer.alloc(length);
    assert.equal((await adapter.read(header, 0, length, 8)).bytesRead, length);
    return assertKreaRawAdapterFactors(JSON.parse(header.toString("utf8")));
  } finally {
    await adapter.close();
  }
}

export async function assertStagedKreaAssets(hub, price, inventory) {
  assert.deepEqual(price.files.map(({ repo, revision, file }) => ({ repo, revision, file })), inventory.map(({ repo, revision, file }) => ({ repo, revision, file })), "staged asset receipt does not match the exact proof inventory");
  for (const item of price.files) {
    const file = await lstat(kreaSnapshotFile(hub, item));
    assert.ok(file.isFile() && file.nlink === 1 && file.size === item.bytes, `staged asset changed: ${item.file}`);
    const ref = path.join(hub, `models--${item.repo.replace("/", "--")}`, "refs", "main");
    assert.equal(await readFile(ref, "utf8"), item.revision, `staged snapshot ref changed: ${item.repo}`);
  }
  const adapter = price.files.find(isRawAdapter);
  if (adapter) {
    assert.deepEqual(adapter, KREA_RAW_ADAPTER, "staged Raw adapter pin changed");
    assert.deepEqual(price.rawAdapter, await readRawAdapterFactors(hub), "staged Raw adapter header changed");
  }
}

export function assertKreaCatalogRows(route, rows, evidence) {
  const source = route.proof === "raw" ? RAW : COMPANION;
  assert.equal(rows.required.length, 1);
  assert.equal(keyOf(rows.required[0]), keyOf(source), `public catalog no longer binds the exact ${route.model} snapshot`);
  return { tier: route.tier, optional: [], required: kreaAssetInventory(evidence, [route]).map((row) => ({ ...row, files: [row.file] })) };
}

async function waitKreaImport({ endpoint, wait, ownsWorker }, jobId, label) {
  const job = await wait(async () => {
    const result = await endpoint.call("GET", `/api/v1/jobs/${jobId}`);
    assert.ok(!["failed", "canceled", "interrupted"].includes(result.status), `${label} ${result.status}: ${result.error ?? result.message}`);
    return result.status === "completed" && result;
  }, 300, label);
  assert.ok(ownsWorker(job.workerId), `${label} was not claimed by this attempt's worker`);
  return job;
}

export async function importKreaRawAdapter({ endpoint, hub, project, projectPath, wait, trackJob, ownsWorker, signal }) {
  const file = path.basename(KREA_RAW_ADAPTER.file);
  const loras = path.join(projectPath, "loras");
  await mkdir(loras, { recursive: true });
  const inputDirectory = await mkdtemp(path.join(loras, "proof-input-"));
  const sourcePath = path.join(inputDirectory, file);
  await pipeline(createReadStream(kreaSnapshotFile(hub, KREA_RAW_ADAPTER)), createWriteStream(sourcePath, { flags: "wx" }), { signal });
  const queued = await endpoint.exchange("POST", "/api/v1/loras/import", {
    loraId: KREA_RAW_LORA, name: "Krea Darkbrush phase proof", family: "krea_2", baseModel: "krea_2_raw",
    scope: "project", projectId: project.id, sourcePath, files: [file], expectedSha256: KREA_RAW_ADAPTER.sha256,
  });
  assert.equal(queued.status, 201, `public Darkbrush import refused: ${queued.body?.detail ?? queued.status}`);
  assert.ok(queued.body?.id, "public Darkbrush import returned no job id");
  trackJob(queued.body);
  assert.equal(queued.body.payload?.loraId, KREA_RAW_LORA, "worker import changed the adapter slot");
  const job = await waitKreaImport({ endpoint, wait, ownsWorker }, queued.body.id, "Darkbrush import");
  const catalog = await endpoint.call("GET", `/api/v1/loras?projectId=${encodeURIComponent(project.id)}`);
  const entry = catalog.find(({ id }) => id === KREA_RAW_LORA);
  const target = path.join(loras, "imports", KREA_RAW_LORA);
  assert.equal(entry?.scope, "project");
  assert.equal(entry.family, "krea_2");
  assert.equal(entry.baseModel, "krea_2_raw");
  assert.equal(entry.installState, "installed");
  assert.equal(entry.source?.provider, "local");
  assert.equal(entry.source?.path, `loras/imports/${KREA_RAW_LORA}`);
  assert.deepEqual(entry.files, [file]);
  assert.equal(await realpath(entry.installedPath), target, "catalog selected a different Darkbrush directory");
  assert.equal(await realpath(job.result?.path), target, "worker installed Darkbrush in a different directory");
  const installed = path.join(target, file);
  const stat = await lstat(installed);
  assert.ok(stat.isFile() && stat.nlink === 1 && stat.size === KREA_RAW_ADAPTER.bytes, "installed Darkbrush file changed");
  const digest = await fileSha256(installed, { signal });
  assert.equal(digest, KREA_RAW_ADAPTER.sha256, "installed Darkbrush digest changed");
  await rm(inputDirectory, { recursive: true });
  return { jobId: job.id, workerId: job.workerId, adapterId: KREA_RAW_LORA, scope: entry.scope, source: KREA_RAW_ADAPTER };
}

export async function importLinkedKrea({ endpoint, hub, wait, trackJob, ownsWorker }) {
  const root = path.dirname(kreaSnapshotFile(hub, CHECKPOINT));
  const relative = path.basename(CHECKPOINT.file);
  const approved = await endpoint.call("POST", "/api/v1/models/library-roots", { path: root, label: "Krea Tier 2 proof" });
  assert.ok(approved.rootId, "public linked-library approval returned no root id");
  const queued = await endpoint.exchange("POST", "/api/v1/models/import", { modelId: LINKED_MODEL, name: "Krea Turbo BF16 proof", type: "image", linkedRootId: approved.rootId, linkedRelativePath: relative, expectedSha256: CHECKPOINT.sha256 });
  assert.equal(queued.status, 201, `public linked import refused: ${queued.body?.detail ?? queued.status}`);
  assert.ok(queued.body?.id && queued.body.payload?.linkedRootId === approved.rootId && !queued.body.payload?.targetDir, "worker import must retain linked ownership");
  trackJob(queued.body);
  const job = await waitKreaImport({ endpoint, wait, ownsWorker }, queued.body.id, "linked model import");
  const catalog = await endpoint.call("GET", "/api/v1/models");
  const goal = catalog.find((item) => item.id === LINKED_MODEL);
  return assertKreaImportStamp(job, goal, approved.rootId);
}

// Same linked compiler/manifest invariants as imported_nvfp4_worker_gpu_smoke, through HTTP.
export function assertKreaImportStamp(job, entry, rootId) {
  assert.equal(entry?.family, "krea_2");
  assert.equal(entry.importSourceShape, "transformer_file");
  assert.equal(entry.importQuantFormat, "bf16");
  assert.equal(entry.source?.provider, "linked-library");
  assert.equal(entry.source?.rootId, rootId);
  assert.equal(entry.source?.relativePath, path.basename(CHECKPOINT.file));
  assert.equal(entry.paths?.model, undefined);
  assert.ok(job.result?.checkpointId && job.result.checkpointId === entry.importPlan?.checkpointId, "public catalog must carry the worker's compiled checkpoint id");
  return { jobId: job.id, workerId: job.workerId, model: entry.id, checkpointId: job.result.checkpointId, rootId, sourceShape: entry.importSourceShape, quantFormat: entry.importQuantFormat };
}

// Subscribe before queuing the request. Polling the job row cannot prove that fast step events
// arrived in order. Keep only the safe fields necessary to check this one generation.
export async function openKreaEvents(endpoint, url, signal, fetchImpl = fetch) {
  const { ticket } = await endpoint.call("POST", "/api/v1/jobs/events/ticket", {});
  assert.ok(ticket, "owned API did not issue a job event ticket");
  const controller = new AbortController();
  const streamSignal = AbortSignal.any([signal, controller.signal]);
  let ready, readyTimer;
  let activeReader;
  const stopReader = (reason) => {
    controller.abort(reason);
    void activeReader?.cancel().catch(() => {});
  };
  const started = new Promise((resolve) => { ready = resolve; });
  const readyDeadline = new Promise((_, reject) => {
    readyTimer = setTimeout(() => reject(new Error("job event stream did not become ready")), 30_000);
  });
  const updates = [];
  const finished = (async () => {
    const response = await fetchImpl(`${url}/api/v1/jobs/events?ticket=${encodeURIComponent(ticket)}`, { redirect: "error", signal: streamSignal });
    assert.ok(response.ok && response.body, "owned job event stream refused");
    const decoder = new TextDecoder();
    let pending = "";
    const reader = response.body.getReader();
    activeReader = reader;
    try {
      while (!streamSignal.aborted) {
        const { done, value: chunk } = await reader.read();
        if (done) break;
        pending += decoder.decode(chunk, { stream: true });
        for (let end; (end = pending.search(/\r?\n\r?\n/)) >= 0;) {
          const frame = pending.slice(0, end);
          pending = pending.slice(end).replace(/^\r?\n\r?\n/, "");
          const fields = frame.split(/\r?\n/);
          const event = fields.find((line) => line.startsWith("event: "))?.slice(7);
          if (event === "ready") ready();
          if (event !== "job.updated") continue;
          const value = JSON.parse(fields.filter((line) => line.startsWith("data: ")).map((line) => line.slice(6)).join("\n"));
          const publishedAssets = value.result?.assets?.length ?? 0;
          const assetId = publishedAssets === 1 ? value.result.assets[0]?.id : undefined;
          updates.push({
            id: value.id, stage: value.stage, status: value.status, message: value.message, backend: value.backend,
            workerId: value.workerId, publishedAssets, receivedMonotonicMs: performance.now(),
            publishedAssetId: typeof assetId === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(assetId) ? assetId : undefined,
          });
        }
        assert.ok(pending.length < 16 * 1024 * 1024 && updates.length < 1024, "owned job event stream exceeded the evidence limit");
      }
      if (!streamSignal.aborted) throw new Error("owned job event stream ended early");
    } finally {
      activeReader = null;
      reader.releaseLock();
    }
  })().catch((error) => {
    if (!streamSignal.aborted || (controller.signal.aborted && controller.signal.reason?.message !== "Krea event collection complete")) throw new Error(`owned job event stream failed: ${error.name}`);
  });
  try { await Promise.race([started, finished, readyDeadline]); }
  catch (error) { stopReader(error); throw error; }
  finally { clearTimeout(readyTimer); }
  return {
    async collect(id, wait) {
      await Promise.race([finished, wait(async () => updates.some((event) => event.id === id && event.status === "completed" && event.publishedAssets === 1), 15, "completed event")]);
      return updates.filter((event) => event.id === id);
    },
    async close() {
      stopReader(new Error("Krea event collection complete"));
      let closeTimer;
      try {
        // The owned API teardown also closes the socket if the fetch reader never settles.
        await Promise.race([finished, new Promise((resolve) => { closeTimer = setTimeout(resolve, 2000); })]);
      } finally { clearTimeout(closeTimer); }
    },
  };
}

export function assertKreaEnhancementRefused(response) {
  assert.equal(response.status, 400, "Krea prompt enhancement must be refused before queuing");
  assert.match(response.body?.detail ?? "", /advanced\.enhancePrompt.*FLUX/);
  assert.equal(response.body?.id, undefined);
}

export function kreaExpectedStepEvents(route) {
  return route.proof === "raw"
    ? PHASES.flatMap(({ steps }) => Array.from({ length: steps }, (_, index) => [index + 1, steps]))
    : Array.from({ length: route.executedSteps }, (_, index) => [index + 1, route.steps]);
}

export function assertKreaEvents(events, route, { nativeLog, requireScope, assetId } = {}) {
  const terminal = events.findIndex((event) => event.status === "completed" && event.publishedAssets === 1);
  assert.ok(terminal >= 0, "worker terminal event with one output missing");
  const nativePasses = nativeLog !== undefined
    ? kreaNativeProgressReceipt(nativeLog, route, { requireScope })
    : undefined;
  let lastStep = -1;
  const steps = events.flatMap(({ message, stage, status }, index) => {
    const match = /\bstep (\d+)\/(\d+)\b/i.exec(message ?? "");
    if (!match) return [];
    if (nativePasses) assert.ok(status === "running" && stage === "generating", "public Krea steps must belong to a running render");
    lastStep = index;
    return [[Number(match[1]), Number(match[2])]];
  });
  const unique = steps.filter((pair, index) => index === 0 || pair[0] !== steps[index - 1][0] || pair[1] !== steps[index - 1][1]);
  // Hires.fix keeps the two-pass authored total on the wire while Krea clips the refine
  // schedule at floor(4 * (1 - 0.5)) = 2. Its two emitted ticks are numbered 9 and 10.
  // Raw calls the native solver on separate phase slices. Both pinned engines forward
  // their numbering unchanged, so every phase's first Step starts at 1 on the public wire.
  if (nativePasses) {
    // The worker publishes latest-wins progress labels. Native callbacks establish the exact
    // schedule. Every observed public label must still be part of that schedule, in order.
    const nativeSteps = nativePasses.flatMap(({ coordinates }) => coordinates);
    // Both Hires.fix native passes share the one authored denominator on the public stream.
    const expected = route.proof === "hires" ? nativeSteps.map((_, index) => [index + 1, route.steps]) : nativeSteps;
    let cursor = 0;
    assert.ok(unique.length > 0 && unique.every(([current, total]) => {
      const match = expected.findIndex(([step, count], index) => index >= cursor && step === current && count === total);
      if (match < 0) return false;
      cursor = match + 1;
      return true;
    }), "public Krea steps must form a nonempty ordered subsequence of native steps");
    // Recovery may publish a completed snapshot before persistence and may repeat the same saved asset.
    assert.ok(assetId && events.filter((event) => event.status === "completed").every((event) =>
      event.publishedAssets === 0 || (event.publishedAssets === 1 && event.publishedAssetId === assetId)),
    "Krea terminal events must identify the one saved asset");
  } else {
    assert.deepEqual(unique, kreaExpectedStepEvents(route), "public step events must match the authored phase slices or actual clipped schedule");
  }
  const decoding = events.flatMap((event, index) => event.status === "running" && /\bdecoding\b/i.test(event.message ?? "") ? [index] : []);
  if (nativePasses) assert.ok(decoding.every((index) => events[index].stage === "generating"),
    "public Krea decoding must belong to a running render");
  // The job stream republishes snapshots while the worker remains in the same stage.
  assert.ok(decoding.length && decoding[0] > lastStep && decoding.at(-1) < terminal &&
    decoding.every((index, offset) => index === decoding[0] + offset), "only one final decoding state may be published");
  assert.ok(events.filter((event) => event.status !== "completed").every((event) => event.publishedAssets === 0), "intermediate output reached public job events");
  return { observedSteps: unique, decodingEvents: 1, decodingUpdates: decoding.length,
    authoredStepSlices: route.proof === "raw" ? PHASES.map(({ steps }) => steps) : [8, 4],
    executedStepSlices: nativePasses ? nativePasses.flatMap(({ coordinates }) =>
      coordinates.filter(([current]) => current === 1).map(([, total]) => total)) : route.proof === "raw" ? [3, 4, 3] : [8, 2],
    ...(nativePasses && { nativePasses }) };
}

function publicKreaRawAsset(sidecar, route, adapterImport, dataRoot) {
  assert.equal(adapterImport?.adapterId, KREA_RAW_LORA);
  assert.equal(adapterImport.scope, "project");
  assert.deepEqual(adapterImport.source, KREA_RAW_ADAPTER, "Raw output must use the pinned project adapter import");
  const loras = sidecar.recipe?.loras;
  assert.ok(Array.isArray(loras) && loras.length === 1, "Raw disk recipe must retain the selected adapter");
  const [adapter] = loras;
  assert.deepEqual({ id: adapter.id, weight: adapter.weight }, route.body.loras[0]);
  assert.equal(adapter.scope, adapterImport.scope);
  assert.equal(adapter.source?.provider, "local");
  assert.equal(adapter.source.repo, null);

  const locator = adapter.installedPath;
  assert.ok(typeof dataRoot === "string", "Raw output must identify the owned data directory");
  const paths = path.posix.isAbsolute(dataRoot) ? path.posix : path.win32;
  const separator = paths === path.win32 ? /[\\/]/ : "/";
  const absolute = paths === path.win32
    ? (value) => /^(?:\\\\\?\\)?[a-z]:[\\/]/i.test(value)
    : path.posix.isAbsolute;
  assert.ok(absolute(dataRoot) && typeof locator === "string" && absolute(locator) &&
    !locator.includes("\0") && !locator.split(separator).some((part) => part === ".." || part === "."),
  "Raw disk adapter locator must be an absolute bundle path");
  const member = paths.relative(paths.toNamespacedPath(dataRoot), paths.toNamespacedPath(locator)).split(paths.sep).join("/");
  assert.match(member, /^models\/portable-attempts\/[a-f0-9]{64}\/bundle\/model0\/member1$/, "Raw disk adapter locator must identify the owned portable bundle member");

  // job_redaction.rs removes these five physical locators from the pinned LoRA row.
  // The persisted generation set retains them and every other asset field must agree.
  const publicAsset = structuredClone(sidecar);
  const [publicAdapter] = publicAsset.recipe.loras;
  for (const key of ["installedPath", "modelPath", "path", "sourcePath"]) {
    assert.ok(adapter[key] === locator, `Raw disk adapter ${key} must match its installed locator`);
    delete publicAdapter[key];
  }
  assert.ok(adapter.source.path === locator, "Raw disk adapter source.path must match its installed locator");
  delete publicAdapter.source.path;
  return publicAsset;
}

function hydratedKreaRawRecipe(recipe) {
  const raw = recipe?.rawAdapterSettings;
  assert.equal(raw?.repo, RAW.repo);
  // Asset history upgrades the recipe's options: "repo" is retained and mirrored to "modelRepo".
  assert.ok(!Object.hasOwn(raw, "modelRepo") || raw.modelRepo === raw.repo, "Raw repo aliases must agree");
  return { ...recipe, rawAdapterSettings: { ...raw, modelRepo: raw.repo } };
}

function kreaRawShareLoras(route) {
  return [{ name: path.posix.basename(KREA_RAW_ADAPTER.file, ".safetensors"), weight: route.body.loras[0].weight, hash: KREA_RAW_ADAPTER.sha256 }];
}

function assertKreaRawSampling(raw, route) {
  assert.equal(raw?.realModelInference, true);
  assert.equal(raw.repo, RAW.repo);
  assert.equal(raw.samplingRegime, "krea_2_raw_multiphase");
  assert.equal(raw.multiPhase, true);
  assert.equal(raw.mlxQuantize, 4);
  assert.deepEqual(raw.phases, PHASES);
  assert.equal(raw.numInferenceSteps, route.steps);
}

export function kreaRawOutputDiagnostics({ route, asset, library, sidecar, share, adapterImport, dataRoot }) {
  function passes(check) {
    try {
      check();
      return true;
    } catch {
      return false;
    }
  }
  return {
    libraryCount: Array.isArray(library) ? library.length : null,
    jobAssetMatches: passes(() => assert.deepEqual(asset, publicKreaRawAsset(sidecar, route, adapterImport, dataRoot))),
    generationSetRecipeMatches: passes(() => assert.deepEqual(library?.[0]?.generationSet?.recipe, hydratedKreaRawRecipe(sidecar.recipe))),
    diskSamplingMatches: passes(() => assertKreaRawSampling(sidecar.recipe?.rawAdapterSettings, route)),
    sharePromptMatches: share?.prompt === route.body.prompt && (share?.negativePrompt ?? "") === route.body.negativePrompt,
    shareLorasMatch: passes(() => assert.deepEqual(share?.loras, kreaRawShareLoras(route))),
  };
}

export function assertKreaOutput({ route, job, asset, library, sidecar, share, width, height, workerId, projectId, adapterImport, dataRoot }) {
  assert.equal(job.workerId, workerId);
  assert.equal(job.result?.model, route.body.model);
  assert.equal(job.payload?.prompt, PROMPT);
  assert.equal(job.payload?.negativePrompt ?? "", route.body.negativePrompt);
  assert.notEqual(job.payload?.advanced?.enhancePrompt, true);
  assert.equal(job.result?.assets?.length, 1);
  assert.deepEqual(job.result?.assetIds, [asset.id]);
  assert.equal(job.result?.assetWrites, undefined);
  assert.equal(library.length, 1, "fresh project must have exactly one saved image");
  const [hydrated] = library;
  assert.equal(hydrated.id, asset.id);
  assert.equal(hydrated.projectId, projectId);
  assert.equal(hydrated.lineage?.jobId, job.id);
  assert.equal(hydrated.generationSet?.jobId, job.id);
  assert.equal(hydrated.generationSet?.id, hydrated.generationSetId);
  assert.equal(job.result?.generationSetId, hydrated.generationSetId);
  if (route.proof === "raw") {
    assert.deepEqual(asset, publicKreaRawAsset(sidecar, route, adapterImport, dataRoot), "public Raw job asset must equal its exact redacted disk sidecar");
  } else {
    assert.deepEqual(asset, sidecar, "API asset and exact disk sidecar must agree");
  }
  const generationSetRecipe = route.proof === "raw" ? hydratedKreaRawRecipe(sidecar.recipe) : sidecar.recipe;
  assert.deepEqual(hydrated.generationSet?.recipe, generationSetRecipe, "generation set must preserve persisted recipe");
  assert.equal(hydrated.generationSet.prompt, PROMPT);
  assert.equal(sidecar.recipe?.prompt, PROMPT);
  assert.equal(sidecar.recipe?.seed, route.body.seed);
  assert.equal(sidecar.recipe?.model, route.body.model);
  assert.equal(share?.prompt, PROMPT);
  assert.equal(share?.negativePrompt ?? "", route.body.negativePrompt);
  assert.equal(width, route.finalSize);
  assert.equal(height, route.finalSize);
  assert.equal(sidecar.file?.width, width);
  assert.equal(sidecar.file?.height, height);
  const raw = sidecar.recipe?.rawAdapterSettings;
  if (route.proof === "raw") {
    assert.equal(job.payload?.model, "krea_2_raw");
    assert.deepEqual(job.payload?.loras?.map(({ id, weight }) => ({ id, weight })), route.body.loras);
    assert.deepEqual(share.loras, kreaRawShareLoras(route), "PNG workflow must contain the exact shareable adapter attribution");
    assertKreaRawSampling(raw, route);
  } else {
    assert.equal(raw?.realModelInference, true);
    assert.equal(raw.importedCheckpoint, LINKED_MODEL);
    assert.equal(raw.engine, `${route.backend === "mlx" ? "mlx" : "candle"}_krea_imported`);
    const facts = raw.checkpointWeightFacts;
    assert.equal(facts?.sourceBinding?.stableToken, `${path.basename(CHECKPOINT.file)}@${CHECKPOINT.bytes}`);
    assert.ok(facts.source?.some(({ codecId }) => codecId === "dense-bf16-v1"), "worker output must identify the loaded BF16 checkpoint codec");
    assert.equal(raw.kreaImportedBase, `${COMPANION.repo}#bf16`);
    for (const [key, value] of Object.entries(route.body.hiresFix)) assert.equal(raw.hiresFix?.[key], value, `wrong resolved Hires.fix ${key}`);
  }
  return { libraryCount: library.length, assetId: asset.id, generationSetId: hydrated.generationSetId, width, height, prompt: PROMPT, negativePrompt: route.body.negativePrompt, raw: route.proof === "raw" ? { phasePlan: raw.phases, tier: "q4", repo: raw.repo } : { importedCheckpoint: raw.importedCheckpoint, companion: { ...COMPANION, tier: "bf16" }, hiresFix: raw.hiresFix } };
}

export function kreaPerformance(metrics, route, memory) {
  assert.equal(metrics?.steps, kreaExpectedStepEvents(route)[0][1], "native metrics must match the first observed step denominator");
  assert.equal(metrics.imageCount, 1);
  assert.ok(Number.isFinite(metrics.sampleMs) && metrics.sampleMs > 0, "native sampleMs is missing");
  assert.ok(memory.samples > 0 && memory.peakBytes > 0, "observed memory peak and sample count are required");
  return { secondsPerObservedStep: metrics.sampleMs / (1000 * route.executedSteps), sampleMs: metrics.sampleMs, authoredSteps: route.steps, executedSteps: route.executedSteps, memory, method: "worker sampleMs divided by validated executed steps; Hires.fix includes the interval between its two sampling passes" };
}
