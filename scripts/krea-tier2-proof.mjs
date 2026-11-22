import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { lstat, mkdir, open, readFile, rename, rm, statfs, writeFile } from "node:fs/promises";
import path from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";

import { probeCacheFile } from "./h3-finetune-terminal-stage.mjs";
import { KREA_ALLOCATION_BASE_SIZE, KREA_ALLOCATION_FINAL_SIZE } from "./krea-tier2-allocation.mjs";

export const KREA_SUITE = "krea-scoped-acceptance";
export const KREA_BASE_APP = "2aa525dfb2fb1ab889bd6a15f80c0a0cc1df1e2d";
export const KREA_BASE_ENGINE = "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294";
const RAW = { repo: "SceneWorks/krea-2-raw-mlx", revision: "b88090c79762ba34b1de23a892da239eb76962e4" };
const COMPANION = { repo: "SceneWorks/krea-2-turbo-mlx", revision: "d009674080cc1bccf2b629d834c34bf5eccdb723" };
const COMFY = { repo: "Comfy-Org/Krea-2", revision: "952f49d49653cb42e7d6cf7cbfad74738073ec7d" };
const ADAPTER = { ...COMFY, file: "loras/krea2_turbo_lora_rank_64_bf16.safetensors", bytes: 469423778, sha256: "db8c5bae0a415d448da9d842111d6e51f7d32e47143a3118eb267e5c4773de87" };
export const CHECKPOINT = Object.freeze({ ...COMFY, file: "diffusion_models/krea2_turbo_bf16.safetensors", bytes: 26283332608, sha256: "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3" });
export const LINKED_MODEL = "linked_krea2_turbo_bf16_proof";
const PROMPT = "A red glass apple beside a blue ceramic cup, a sharp studio photograph with soft shadows";
const NEGATIVE = "blur, text, watermark";
const PHASES = [{ steps: 3, guidance: 0, loras: [] }, { steps: 4, guidance: 1, loras: [{ index: 0, weight: 0.75 }] }, { steps: 3, guidance: 0, loras: [] }];
const HIRES = { enabled: true, upscaleBy: 1.5, steps: 4, denoisingStrength: 0.5 };
const PROJECTOR_KEYS = ["diffusion_model.txtfusion.projector.lora_down.weight", "diffusion_model.txtfusion.projector.lora_up.weight"];
const GIB = 1024 ** 3;
const STAGING_PROGRESS_BYTES = GIB / 4;

export function kreaRoutes(backend, { allocation = false } = {}) {
  const shared = { backend, kind: "image", mode: "text_to_image", group: null, endpoint: "/api/v1/image/jobs", claimMinutes: 5, timeoutMinutes: 20 };
  const body = { mode: shared.mode, prompt: PROMPT, width: 512, height: 512, count: 1, seed: 33602 };
  const routes = [
    { ...shared, id: `${backend}-krea-raw-phases`, model: "krea_2_raw", tier: "q4", proof: "raw", steps: 10, executedSteps: 10, finalSize: 512, body: { ...body, model: "krea_2_raw", negativePrompt: NEGATIVE, loras: [{ id: "krea2_turbo_accel", weight: 1 }], advanced: { mlxQuantize: 4, sampler: "euler", phases: PHASES } } },
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
    ...(routes.some(({ proof }) => proof === "raw") ? [...tierEntries(evidence, RAW, "q4", false), ADAPTER] : []),
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

export async function stageKreaAssets(hub, price, signal, { cacheHub, fetchImpl = fetch, progress = () => {} } = {}) {
  await mkdir(hub, { recursive: true });
  const fs = await statfs(hub, { bigint: true });
  const freeBytes = Number(fs.bavail * fs.bsize);
  assert.ok(freeBytes >= price.bytes + 20 * GIB, `need ${price.bytes} bytes for pinned Krea assets plus 20 GiB for builds, have ${freeBytes} bytes`);
  const deadline = AbortSignal.any([signal, AbortSignal.timeout((cacheHub ? 180 : 90) * 60_000)]);
  for (const item of price.files) {
    const dest = kreaSnapshotFile(hub, item);
    const partial = `${dest}.part`;
    const hash = createHash(item.sha256 ? "sha256" : "sha1");
    if (!item.sha256) hash.update(`blob ${item.bytes}\0`);
    const identity = `${keyOf(item)}/${item.file}`;
    let source = "cache-probe";
    let bytes = 0;
    let nextProgress = STAGING_PROGRESS_BYTES;
    const report = (status, detail = {}) => progress({ status, identity, source, receivedBytes: bytes, expectedBytes: item.bytes, ...detail });
    try {
      deadline.throwIfAborted();
      const cached = cacheHub && await probeCacheFile({ ...item, repoType: "model" }, cacheHub, { signal: deadline });
      const cacheHit = cached?.state === "present";
      source = cacheHit ? "runner-cache" : "pinned-https";
      report("started");
      await mkdir(path.dirname(dest), { recursive: true });
      let stream;
      if (cacheHit) {
        assert.equal(cached.size, item.bytes, `wrong cached artifact length: ${item.file}`);
        stream = createReadStream(cached.sourcePath);
      } else {
        const response = await fetchImpl(resolveUrl(item), { redirect: "follow", signal: deadline });
        assert.ok(response.ok && response.body, `cannot fetch pinned ${identity} (HTTP ${response.status})`);
        stream = Readable.fromWeb(response.body);
      }
      const verify = new Transform({ transform(chunk, _encoding, done) {
        bytes += chunk.length;
        if (bytes > item.bytes) return done(new Error(`pinned artifact exceeded its length: ${item.file}`));
        hash.update(chunk);
        if (bytes >= nextProgress) {
          report("transferring");
          nextProgress = bytes + STAGING_PROGRESS_BYTES;
        }
        done(null, chunk);
      } });
      await pipeline(stream, verify, createWriteStream(partial, { flags: "wx" }), { signal: deadline });
      assert.equal(bytes, item.bytes, `wrong artifact length: ${item.file}`);
      const digest = hash.digest("hex");
      assert.equal(digest, item.sha256 ?? item.gitBlobSha1, `wrong artifact digest: ${item.file}`);
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
  if (!price.files.some(({ repo, revision, file }) => repo === ADAPTER.repo && revision === ADAPTER.revision && file === ADAPTER.file)) return receipt;
  const adapter = await open(kreaSnapshotFile(hub, ADAPTER), "r");
  try {
    const prefix = Buffer.alloc(8);
    assert.equal((await adapter.read(prefix, 0, prefix.length, 0)).bytesRead, prefix.length);
    const length = Number(prefix.readBigUInt64LE());
    assert.ok(length > 0 && length <= 1024 * 1024, "pinned adapter header length is invalid");
    const header = Buffer.alloc(length);
    assert.equal((await adapter.read(header, 0, length, 8)).bytesRead, length);
    const tensors = JSON.parse(header.toString("utf8"));
    assert.ok(PROJECTOR_KEYS.every((name) => tensors[name]), "exact Raw adapter must exercise the historical projector path");
    return { ...receipt, adapterProjectorKeys: PROJECTOR_KEYS };
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
}

export function assertKreaLoraManifest(loras) {
  const entry = loras.find(({ id }) => id === "krea2_turbo_accel");
  assert.deepEqual(entry?.source, { provider: "huggingface", repo: ADAPTER.repo, revision: ADAPTER.revision, file: ADAPTER.file }, "public catalog must source the exact projector adapter");
}

export function assertKreaCatalogRows(route, rows, evidence) {
  const source = route.proof === "raw" ? RAW : COMPANION;
  assert.equal(rows.required.length, 1);
  assert.equal(keyOf(rows.required[0]), keyOf(source), `public catalog no longer binds the exact ${route.model} snapshot`);
  return { tier: route.tier, optional: [], required: kreaAssetInventory(evidence, [route]).map((row) => ({ ...row, files: [row.file] })) };
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
  const job = await wait(async () => {
    const result = await endpoint.call("GET", `/api/v1/jobs/${queued.body.id}`);
    assert.ok(!["failed", "canceled", "interrupted"].includes(result.status), `linked import ${result.status}: ${result.error ?? result.message}`);
    return result.status === "completed" && result;
  }, 300, "linked model import");
  assert.ok(ownsWorker(job.workerId), "linked import was not claimed by this attempt's worker");
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
          updates.push({ id: value.id, stage: value.stage, status: value.status, message: value.message, backend: value.backend, workerId: value.workerId, publishedAssets: value.result?.assets?.length ?? 0, receivedMonotonicMs: performance.now() });
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

export function assertKreaEvents(events, route) {
  const terminal = events.findIndex((event) => event.status === "completed" && event.publishedAssets === 1);
  assert.ok(terminal >= 0, "worker terminal event with one output missing");
  let lastStep = -1;
  const steps = events.flatMap(({ message }, index) => {
    const match = /\bstep (\d+)\/(\d+)\b/i.exec(message ?? "");
    if (!match) return [];
    lastStep = index;
    return [[Number(match[1]), Number(match[2])]];
  });
  const unique = steps.filter((pair, index) => index === 0 || pair[0] !== steps[index - 1][0] || pair[1] !== steps[index - 1][1]);
  // Hires.fix keeps the two-pass authored total on the wire while Krea clips the refine
  // schedule at floor(4 * (1 - 0.5)) = 2. Its two emitted ticks are numbered 9 and 10.
  // Raw calls the native solver on separate phase slices. Both pinned engines forward
  // their numbering unchanged, so every phase's first Step starts at 1 on the public wire.
  assert.deepEqual(unique, kreaExpectedStepEvents(route), "public step events must match the authored phase slices or actual clipped schedule");
  const decoding = events.flatMap((event, index) => event.status === "running" && /\bdecoding\b/i.test(event.message ?? "") ? [index] : []);
  // The job stream republishes snapshots while the worker remains in the same stage.
  assert.ok(decoding.length && decoding[0] > lastStep && decoding.at(-1) < terminal &&
    decoding.every((index, offset) => index === decoding[0] + offset), "only one final decoding state may be published");
  assert.ok(events.filter((event) => event.status !== "completed").every((event) => event.publishedAssets === 0), "intermediate output reached public job events");
  return { observedSteps: unique, decodingEvents: 1, decodingUpdates: decoding.length, authoredStepSlices: route.proof === "raw" ? PHASES.map(({ steps }) => steps) : [8, 4], executedStepSlices: route.proof === "raw" ? [3, 4, 3] : [8, 2] };
}

export function assertKreaOutput({ route, job, asset, library, sidecar, share, width, height, workerId, projectId }) {
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
  assert.deepEqual(asset, sidecar, "API asset and exact disk sidecar must agree");
  assert.deepEqual(hydrated.generationSet?.recipe, sidecar.recipe, "generation set must preserve persisted recipe");
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
  assert.equal(raw?.realModelInference, true);
  if (route.proof === "raw") {
    assert.equal(job.payload?.model, "krea_2_raw");
    assert.equal(raw.repo, RAW.repo);
    assert.equal(raw.samplingRegime, "krea_2_raw_multiphase");
    assert.equal(raw.multiPhase, true);
    assert.equal(raw.mlxQuantize, 4);
    assert.deepEqual(raw.phases, PHASES);
    assert.equal(raw.numInferenceSteps, route.steps);
  } else {
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
  return { secondsPerObservedStep: metrics.sampleMs / (1000 * route.executedSteps), sampleMs: metrics.sampleMs, authoredSteps: route.steps, executedSteps: route.executedSteps, memory, method: "worker sampleMs divided by actual step events; Hires.fix includes the interval between its two sampling passes" };
}
