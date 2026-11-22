import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { dispatchOptions, exitCodeFor, planRoutes, readCatalog, readPatternEvidence, resolveConfig } from "./tier2-ab.mjs";
import {
  KREA_LINKED_SOURCE_REPAIR_REVISION, bindKreaLinkedSourceBuild,
  reviewedKreaLinkedSourceCandidate, withKreaLinkedSourceBuild,
} from "./krea-tier2-linked-source.mjs";
import {
  KREA_SUITE, KREA_BASE_APP, KREA_BASE_ENGINE, KREA_RAW_ADAPTER, KREA_RAW_LORA, CHECKPOINT, LINKED_MODEL, assertKreaEvents,
  assertKreaImportStamp, assertKreaEnhancementRefused, assertKreaOutput, assertKreaRawAdapterFactors, importKreaRawAdapter, kreaRoutes,
  kreaAssetInventory, kreaExpectedStepEvents, kreaPerformance, kreaSnapshotFile, openKreaEvents, priceKreaAssets, stageKreaAssets,
} from "./krea-tier2-proof.mjs";

const HEAD = "1".repeat(40);
const PIN = "2".repeat(40);
const [RAW, HIRES] = kreaRoutes("mlx");

test("Krea source bind refuses an otherwise ordinary checkout with different source bytes", async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), "krea-source-bind-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const sides = ["base", "head", "repair"].map((side) => path.join(dir, side));
  for (const side of sides) {
    const parent = path.join(side, "apps", "rust-api", "src");
    await mkdir(parent, { recursive: true });
    await writeFile(path.join(parent, "portable_admission.rs"), "unknown source");
  }
  await assert.rejects(bindKreaLinkedSourceBuild(...sides), /unexpected Krea linked-source bytes for base/);
  await assert.rejects(bindKreaLinkedSourceBuild(sides[0], sides[1], sides[1]), /distinct checkouts/);
});

test("Krea candidate fingerprints must belong to one exact reviewed source set", () => {
  const original = [
    "1f96bafdc500cb9c938d176345389a4938b859efd9e7b4d8309d4593c1e06b52",
    "62405de585a204660319e50130baaf3aa57904131d15c949271c7aeb8e9307d8",
    "77e579b5d86ec1764130bcddb8de2b39b0d7f9b384cca9a0924bd3fa28784908",
    "9a84528d02e654b94cc39ca8c962a8d157398069282aa40d9e78d7364d754bb0",
  ];
  const integrated = [
    "9d9191f521d43c3f0850e4d80f305c187b97a3119f3e0681044670a7e61a1676",
    original[1], original[2],
    "be3b26400a7babc34ce1bd0f6fabf433cf11d3af62d41766b6439e3a72852d5d",
  ];
  assert.equal(reviewedKreaLinkedSourceCandidate(original), KREA_LINKED_SOURCE_REPAIR_REVISION);
  assert.equal(reviewedKreaLinkedSourceCandidate(integrated), "e98dea13f401500998947e4359ebd56fae113bf9");
  assert.throws(() => reviewedKreaLinkedSourceCandidate([...original.slice(0, 3), integrated[3]]), /one reviewed source set/);
  assert.throws(() => reviewedKreaLinkedSourceCandidate(integrated.slice(0, 3)), /one reviewed source set/);
});

test("paired Krea Actions fetch only the four frozen repair paths on both hosts", async () => {
  const workflow = await readFile(new URL("../.github/workflows/tier2-ab.yml", import.meta.url), "utf8");
  const repairs = workflow.split("- name: Check out the four reviewed Krea baseline repair files\n").slice(1);
  assert.equal(repairs.length, 2);
  for (const repair of repairs) {
    const step = repair.slice(0, repair.indexOf("\n      - "));
    assert.match(step, /if: \$\{\{ inputs\.suite == 'krea-scoped-acceptance' && !inputs\.krea_allocation \}\}/);
    assert.match(step, /uses: actions\/checkout@[a-f0-9]{40}/);
    assert.ok(step.includes(`ref: ${KREA_LINKED_SOURCE_REPAIR_REVISION}\n`));
    assert.match(step, /path: tier2-ab-krea-repair\n\s+fetch-depth: 1\n\s+persist-credentials: false\n\s+sparse-checkout-cone-mode: false/);
    assert.deepEqual(step.split("sparse-checkout: |\n")[1]?.split("\n").map((entry) => entry.trim()), [
      "/apps/rust-api/src/portable_admission.rs",
      "/apps/rust-api/src/portable_routes.rs",
      "/crates/sceneworks-core/src/model_artifacts/artifact_selection.rs",
      "/crates/sceneworks-worker/src/external_library_runtime.rs",
    ]);
  }
});

test("Krea baseline restores exact checkout bytes after the borrowed-source build fails", async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), "krea-source-restore-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const base = await realpath(dir);
  const replaced = { file: "guard.rs", original: Buffer.from("original\r\n"), repaired: Buffer.from("repaired\n") };
  const file = path.join(base, replaced.file);
  await writeFile(file, replaced.original);
  await assert.rejects(withKreaLinkedSourceBuild({ base, replacements: [replaced] }, async () => {
    assert.deepEqual(await readFile(file), replaced.repaired);
    throw new Error("fixture build failed");
  }), /fixture build failed/);
  assert.deepEqual(await readFile(file), replaced.original);
});

function inputs(options = {}, over = {}) {
  return { profile: "tier2-ab", backend: "mlx", ref: HEAD, inference_sha: PIN, moss_options_json: JSON.stringify({ suite: KREA_SUITE, head_sha: HEAD, ...options }), ...over };
}

test("Krea dispatch is opt-in, selects two fixed public routes, and requires exact independent pins", () => {
  const selected = dispatchOptions(inputs(), HEAD);
  assert.equal(selected.preflight_only, true);
  assert.equal(selected.base_sha, KREA_BASE_APP);
  assert.equal(selected.head_sha, HEAD);
  assert.throws(() => dispatchOptions(inputs({}, { inference_sha: KREA_BASE_ENGINE }), HEAD), /differ from baseline/);
  assert.throws(() => dispatchOptions(inputs({ head_sha: "main" }), HEAD), /match/);
  assert.throws(() => dispatchOptions(inputs({ hubs: ["/other"] }), HEAD), /Krea selector/);
  assert.throws(() => dispatchOptions(inputs({ preflight_only: false }), HEAD), /requires download_missing/);
  assert.throws(() => dispatchOptions(inputs({}, { download_missing: true }), HEAD), /leaves it off/);
  assert.doesNotThrow(() => dispatchOptions(inputs({ preflight_only: false }, { download_missing: true }), HEAD));
  const env = { TIER2_AB_SUITE: KREA_SUITE, TIER2_AB_BACKEND: "mlx", TIER2_AB_BASE_SHA: KREA_BASE_APP, TIER2_AB_HEAD_SHA: HEAD, TIER2_AB_INFERENCE_SHA: PIN, TIER2_AB_BASE_DIR: "/base", TIER2_AB_HEAD_DIR: "/head", TIER2_AB_WORK: "/work" };
  const cfg = resolveConfig(env, { platform: "darwin" });
  assert.deepEqual(cfg.routes, [RAW, HIRES]);
  assert.deepEqual(cfg.hubs, ["/work/scratch/krea-hf/hub"]);
  assert.equal(cfg.repairDir, "/tier2-ab-krea-repair");
  assert.throws(() => resolveConfig({ ...env, TIER2_AB_BASE_SHA: "3".repeat(40) }, { platform: "darwin" }), /baseline app is frozen/);
  assert.throws(() => resolveConfig({ ...env, TIER2_AB_HUBS: "/other" }, { platform: "darwin" }), /attempt-owned/);
});

test("pinned inventory preserves the Raw q4 tier, exact adapter and linked BF16 with only the companion surface", async () => {
  const evidence = await readPatternEvidence(".");
  const inventory = kreaAssetInventory(evidence, [RAW, HIRES]);
  assert.equal(inventory.length, 23);
  assert.ok(inventory.some((item) => item.file.match(/^q4\/transformer\/.*\.safetensors$/)));
  assert.deepEqual(inventory.find(({ file }) => file.startsWith("loras/")), {
    repo: "Comfy-Org/Krea-2",
    revision: "952f49d49653cb42e7d6cf7cbfad74738073ec7d",
    file: "loras/krea2_darkbrush.safetensors",
    bytes: 469291992,
    sha256: "f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6",
  });
  assert.deepEqual(RAW.body.loras, [{ id: KREA_RAW_LORA, weight: 1 }]);
  assert.ok(inventory.some((item) => item.file === "bf16/text_encoder/model.safetensors"));
  assert.deepEqual(inventory.filter((item) => item.file.startsWith("bf16/transformer/")).map((item) => item.file), ["bf16/transformer/config.json"]);
  assert.deepEqual(inventory.find((item) => item.file.startsWith("diffusion_models/")), CHECKPOINT);
  const catalog = await readCatalog(".");
  const plan = await planRoutes({ routes: [RAW, HIRES], hubs: ["/missing"], baseCatalog: catalog, headCatalog: catalog, platform: "darwin", evidence });
  assert.ok(plan.every((route) => route.status === "skipped" && route.rows.required.length > 0));
  const changed = structuredClone(catalog);
  changed.find((model) => model.id === "krea_2_raw").downloads.find((row) => row.variant === "q4").revision = "1".repeat(40);
  const drifted = await planRoutes({ routes: [RAW], hubs: ["/missing"], baseCatalog: changed, headCatalog: changed, platform: "darwin", evidence });
  assert.match(drifted[0].skipReason, /exact krea_2_raw snapshot/);
});

test("Darkbrush factors must be complete PEFT pairs and cannot include a merge-only bias", () => {
  const projector = "transformer.text_fusion.projector";
  const targets = [projector, ...Array.from({ length: 263 }, (_, index) => `transformer.blocks.${index}.attn.to_q`)];
  const tensors = Object.fromEntries(targets.flatMap((target) => [
    [`${target}.lora_A.weight`, { dtype: "F32", shape: [32, target === projector ? 12 : 512] }],
    [`${target}.lora_B.weight`, { dtype: "F32", shape: [target === projector ? 1 : 512, 32] }],
  ]));
  assert.deepEqual(assertKreaRawAdapterFactors(tensors), { tensorCount: 528, targetCount: 264, rank: 32, dtype: "F32", projectorFactors: [`${projector}.lora_A.weight`, `${projector}.lora_B.weight`] });
  const mixed = { "transformer.first.diff_b": { dtype: "F32", shape: [6144] }, ...tensors };
  delete mixed[`${targets.at(-1)}.lora_B.weight`];
  assert.throws(() => assertKreaRawAdapterFactors(mixed), /outside its LoRA factors.*\.diff_b/);
  const unpaired = { ...tensors, "transformer.extra.lora_B.weight": { dtype: "F32", shape: [512, 32] } };
  delete unpaired[`${targets.at(-1)}.lora_B.weight`];
  assert.throws(() => assertKreaRawAdapterFactors(unpaired), /without its peer/);
});

test("Raw sends the fixed pin to normal project import and refuses a changed installation", async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), "krea-lora-import-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const [hub, projectPath] = ["hub", "project"].map((name) => path.join(dir, name));
  const source = kreaSnapshotFile(hub, KREA_RAW_ADAPTER);
  await mkdir(path.dirname(source), { recursive: true });
  await writeFile(source, "fake adapter input");
  const target = path.join(projectPath, "loras", "imports", KREA_RAW_LORA);
  const file = path.basename(source);
  const exchange = async (method, url, body) => {
    assert.deepEqual([method, url], ["POST", "/api/v1/loras/import"]);
    assert.equal(body.loraId, KREA_RAW_LORA);
    assert.equal(body.expectedSha256, KREA_RAW_ADAPTER.sha256);
    assert.equal(body.projectId, "owned project");
    assert.equal(body.scope, "project");
    assert.equal(body.family, "krea_2");
    assert.equal(body.baseModel, "krea_2_raw");
    assert.equal(await readFile(body.sourcePath, "utf8"), "fake adapter input");
    assert.notEqual(body.sourcePath, source);
    return { status: 201, body: { id: "import-job", payload: { loraId: body.loraId } } };
  };
  const call = async (method, url) => {
    assert.equal(method, "GET");
    if (url === "/api/v1/jobs/import-job") return { status: "completed", workerId: "owned", result: { path: target } };
    assert.equal(url, "/api/v1/loras?projectId=owned%20project");
    await mkdir(target, { recursive: true });
    await writeFile(path.join(target, file), "substituted adapter");
    return [{ id: KREA_RAW_LORA, scope: "project", family: "krea_2", baseModel: "krea_2_raw", installState: "installed", source: { provider: "local", path: `loras/imports/${KREA_RAW_LORA}` }, files: [file], installedPath: target }];
  };
  let tracked;
  await assert.rejects(importKreaRawAdapter({ endpoint: { exchange, call }, hub, project: { id: "owned project" }, projectPath: path.join(await realpath(dir), "project"), wait: (probe) => probe(), trackJob: (job) => { tracked = job.id; }, ownsWorker: (id) => id === "owned", signal: new AbortController().signal }), /installed Darkbrush file changed/);
  assert.equal(tracked, "import-job");
  assert.equal(await readFile(source, "utf8"), "fake adapter input");
});

test("asset metadata pricing refuses digest and length drift before downloading", async () => {
  let row = { type: "file", path: CHECKPOINT.file, size: CHECKPOINT.bytes, lfs: { oid: CHECKPOINT.sha256 } };
  const fetchMetadata = async (url, opts) => {
    assert.match(url, /\/tree\/952f49d49653cb42e7d6cf7cbfad74738073ec7d\/diffusion_models\?recursive=true$/);
    assert.equal(opts.redirect, "error");
    return new Response(JSON.stringify([row]));
  };
  const signal = new AbortController().signal;
  assert.deepEqual(await priceKreaAssets([CHECKPOINT], signal, fetchMetadata), { bytes: CHECKPOINT.bytes, files: [CHECKPOINT] });
  row = { ...row, size: row.size + 10 };
  await assert.rejects(priceKreaAssets([CHECKPOINT], signal, fetchMetadata), /changed length/);
  row = { ...row, size: CHECKPOINT.bytes, lfs: { oid: "f".repeat(64) } };
  await assert.rejects(priceKreaAssets([CHECKPOINT], signal, fetchMetadata), /changed LFS digest/);
});

test("staging copies verified cache hits and downloads only missing exact assets", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "krea-stage-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cacheHub = path.join(root, "cache");
  const hub = path.join(root, "attempt");
  const data = Buffer.from("cached fake weight");
  const config = Buffer.from("{}");
  const cached = { ...CHECKPOINT, bytes: data.length, sha256: createHash("sha256").update(data).digest("hex") };
  const missing = { repo: CHECKPOINT.repo, revision: CHECKPOINT.revision, file: "bf16/config.json", bytes: config.length,
    gitBlobSha1: createHash("sha1").update(`blob ${config.length}\0`).update(config).digest("hex") };
  const cachedFile = kreaSnapshotFile(cacheHub, cached);
  await mkdir(path.dirname(cachedFile), { recursive: true });
  await writeFile(cachedFile, data);
  const events = [];
  const receipt = await stageKreaAssets(hub, { bytes: cached.bytes + missing.bytes, files: [cached, missing] },
    new AbortController().signal, {
      cacheHub, progress: (event) => events.push(event),
      fetchImpl: async (url) => {
        assert.match(url, /\/bf16\/config\.json$/);
        return new Response(config);
      },
    });
  assert.equal(receipt.stagedFileCount, 2);
  assert.deepEqual(await readFile(kreaSnapshotFile(hub, cached)), data);
  assert.deepEqual(await readFile(kreaSnapshotFile(hub, missing)), config);
  assert.equal((await lstat(cachedFile)).nlink, 1);
  assert.equal((await lstat(kreaSnapshotFile(hub, cached))).nlink, 1);
  assert.deepEqual(events.filter(({ status }) => status === "verified").map(({ source, receivedBytes, digest }) => [source, receivedBytes, digest]),
    [["runner-cache", cached.bytes, cached.sha256], ["pinned-https", missing.bytes, missing.gitBlobSha1]]);
});

test("staging rejects corrupt cache bytes and records their attempted transfer", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "krea-stage-corrupt-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cacheHub = path.join(root, "cache");
  const hub = path.join(root, "attempt");
  const cached = { ...CHECKPOINT, bytes: 4 };
  const cachedFile = kreaSnapshotFile(cacheHub, cached);
  await mkdir(path.dirname(cachedFile), { recursive: true });
  await writeFile(cachedFile, "fake");
  const events = [];
  await assert.rejects(stageKreaAssets(hub, { bytes: 4, files: [cached] }, new AbortController().signal, {
    cacheHub, progress: (event) => events.push(event), fetchImpl: () => assert.fail("corrupt cache must refuse")
  }), /after receiving 4 bytes: wrong artifact digest/);
  assert.equal(events.at(-1).status, "failed");
  assert.equal(events.at(-1).source, "runner-cache");
  assert.equal(await readFile(cachedFile, "utf8"), "fake");
  await assert.rejects(readFile(`${kreaSnapshotFile(hub, cached)}.part`), { code: "ENOENT" });
});

function jobEvents(route) {
  return [
    ...kreaExpectedStepEvents(route).map(([current, total]) => ({ status: "running", message: `Image 1/1 · step ${current}/${total}` })),
    { status: "running", message: "decoding" },
    { status: "completed", publishedAssets: 1 },
  ].map((event) => ({ publishedAssets: 0, ...event }));
}

test("queue proof checks observed CFG phase and clipped Hires progress without exposing intermediate output", () => {
  assert.doesNotThrow(() => assertKreaEnhancementRefused({ status: 400, body: { detail: "advanced.enhancePrompt is supported only by FLUX.2-dev" } }));
  assert.throws(() => assertKreaEnhancementRefused({ status: 400, body: { detail: "model missing" } }), /match/);
  const raw = assertKreaEvents(jobEvents(RAW), RAW);
  assert.deepEqual(raw.executedStepSlices, [3, 4, 3]);
  assert.deepEqual(raw.observedSteps, [[1, 3], [2, 3], [3, 3], [1, 4], [2, 4], [3, 4], [4, 4], [1, 3], [2, 3], [3, 3]]);
  assert.equal(kreaPerformance({ steps: 3, imageCount: 1, sampleMs: 5000 }, RAW, { peakBytes: 100, samples: 2 }).secondsPerObservedStep, 0.5);
  assert.deepEqual(assertKreaEvents(jobEvents(HIRES), HIRES).executedStepSlices, [8, 2]);
  assert.throws(() => assertKreaEvents(jobEvents(HIRES).slice(1), HIRES), /clipped schedule/);
  const leaked = jobEvents(HIRES);
  leaked[5].publishedAssets = 1;
  assert.throws(() => assertKreaEvents(leaked, HIRES), /intermediate/);
  const result = kreaPerformance({ steps: 12, imageCount: 1, sampleMs: 5000 }, HIRES, { peakBytes: 100, samples: 2 });
  assert.equal(result.secondsPerObservedStep, 0.5);
  assert.throws(() => kreaPerformance({ steps: 12, imageCount: 1, sampleMs: 5000 }, HIRES, { samples: 0 }), /memory/);
});

test("queue proof accepts repeated snapshots of the final decoding state only", () => {
  const events = jobEvents(HIRES);
  const finalDecode = events.at(-2);
  events.splice(-1, 0, { ...finalDecode }, { ...finalDecode });
  const proof = assertKreaEvents(events, HIRES);
  assert.equal(proof.decodingEvents, 1);
  assert.equal(proof.decodingUpdates, 3);

  const earlyDecode = [...events];
  earlyDecode.splice(8, 0, { ...finalDecode });
  assert.throws(() => assertKreaEvents(earlyDecode, HIRES), /one final decoding state/);
  const interruptedDecode = [...events];
  interruptedDecode.splice(-2, 0, { status: "running", message: "Saving.", publishedAssets: 0 });
  assert.throws(() => assertKreaEvents(interruptedDecode, HIRES), /one final decoding state/);
  assert.throws(() => assertKreaEvents([...events, finalDecode], HIRES), /one final decoding state/);
  events.at(-2).publishedAssets = 1;
  assert.throws(() => assertKreaEvents(events, HIRES), /intermediate output/);
});

test("owned SSE opens before the request and keeps only scoped job events across fragmented frames", async () => {
  const encoder = new TextEncoder();
  let output;
  const stream = new ReadableStream({ start(controller) { output = controller; } });
  const endpoint = { async call(method, route) {
    assert.equal(method, "POST");
    assert.equal(route, "/api/v1/jobs/events/ticket");
    return { ticket: "one-time-secret" };
  } };
  const fetchImpl = async (url, { signal }) => {
    assert.equal(url, "http://127.0.0.1:1234/api/v1/jobs/events?ticket=one-time-secret");
    signal.addEventListener("abort", () => output.close(), { once: true });
    output.enqueue(encoder.encode("event: rea"));
    output.enqueue(encoder.encode("dy\r\ndata: {}\r\n\r\n"));
    return new Response(stream);
  };
  const events = await openKreaEvents(endpoint, "http://127.0.0.1:1234", new AbortController().signal, fetchImpl);
  for (const id of ["unrelated", "requested"]) output.enqueue(encoder.encode(`event: job.updated\ndata: ${JSON.stringify({ id, status: "completed", payload: "unneeded", result: { assets: [{}] } })}\n\n`));
  const rows = await events.collect("requested", async (probe) => { for (let i = 0; i < 10; i++) { if (await probe()) return; await new Promise(setImmediate); } throw new Error("no target update"); });
  assert.deepEqual(rows.map(({ id, status, payload }) => ({ id, status, payload })), [{ id: "requested", status: "completed", payload: undefined }]);
  await events.close();
});

test("owned SSE shutdown stays bounded if the reader never acknowledges abort", { timeout: 6000 }, async () => {
  const stalled = new Promise(() => {});
  const ready = new TextEncoder().encode("event: ready\ndata: {}\n\n");
  let started = false, cancels = 0;
  const reader = {
    read() {
      if (started) return stalled;
      started = true;
      return Promise.resolve({ done: false, value: ready });
    },
    cancel() { cancels++; return stalled; },
    releaseLock() {},
  };
  const endpoint = { async call() { return { ticket: "owned-test-ticket" }; } };
  const fetchImpl = async () => ({ ok: true, body: { getReader: () => reader } });
  const events = await openKreaEvents(endpoint, "http://127.0.0.1:1234", new AbortController().signal, fetchImpl);
  await events.close();
  assert.equal(cancels, 1);
});

test("owned SSE startup rejects at its deadline even if the reader never becomes ready", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const stalled = new Promise(() => {});
  let cancels = 0;
  const reader = { read: () => stalled, cancel() { cancels++; return stalled; }, releaseLock() {} };
  const endpoint = { async call() { return { ticket: "owned-test-ticket" }; } };
  const fetchImpl = async () => ({ ok: true, body: { getReader: () => reader } });
  const events = openKreaEvents(endpoint, "http://127.0.0.1:1234", new AbortController().signal, fetchImpl);
  const rejected = assert.rejects(events, /job event stream did not become ready/);
  await new Promise(setImmediate);
  t.mock.timers.tick(30_000);
  await rejected;
  assert.equal(cancels, 1);
});

test("the linked stamp and sole final output bind the actual authored request and saved recipe", () => {
  const stamp = { id: LINKED_MODEL, family: "krea_2", importSourceShape: "transformer_file", importQuantFormat: "bf16", importPlan: { checkpointId: "checkpoint" }, source: { provider: "linked-library", rootId: "root", relativePath: "krea2_turbo_bf16.safetensors" } };
  const importJob = { id: "import", workerId: "worker", result: { checkpointId: "checkpoint" } };
  assert.equal(assertKreaImportStamp(importJob, stamp, "root").checkpointId, "checkpoint");
  assert.throws(() => assertKreaImportStamp(importJob, { ...stamp, paths: { model: "/managed" } }, "root"));
  const body = HIRES.body;
  const checkpointWeightFacts = { sourceBinding: { stableToken: `krea2_turbo_bf16.safetensors@${CHECKPOINT.bytes}` }, source: [{ codecId: "dense-bf16-v1" }] };
  const raw = { realModelInference: true, checkpointWeightFacts, engine: "mlx_krea_imported", importedCheckpoint: LINKED_MODEL, kreaImportedBase: "SceneWorks/krea-2-turbo-mlx#bf16", hiresFix: body.hiresFix };
  const recipe = { model: body.model, seed: body.seed, prompt: body.prompt, rawAdapterSettings: raw };
  const asset = { id: "asset", file: { width: 768, height: 768 }, recipe };
  const libraryAsset = { ...asset, projectId: "p", lineage: { jobId: "job" }, generationSetId: "set", generationSet: { id: "set", jobId: "job", recipe, prompt: body.prompt } };
  const proof = { route: HIRES, job: { id: "job", workerId: "worker", payload: body, result: { model: body.model, generationSetId: "set", assets: [asset], assetIds: [asset.id] } }, width: 768, height: 768, asset, library: [libraryAsset], sidecar: asset, share: { prompt: body.prompt }, workerId: "worker", projectId: "p" };
  assert.equal(assertKreaOutput(proof).raw.importedCheckpoint, LINKED_MODEL);
  assert.throws(() => assertKreaOutput({ ...proof, width: 512 }), /768/);
  assert.throws(() => assertKreaOutput({ ...proof, library: [libraryAsset, libraryAsset] }), /one saved image/);
  assert.throws(() => assertKreaOutput({ ...proof, share: { prompt: "changed" } }), /prompt|changed/);
});

test("a skipped, refused or incomparable arm cannot pass the Krea product suite", () => {
  const pass = { status: "ran", verdict: "equal" };
  assert.equal(exitCodeFor({ kind: KREA_SUITE, mode: "run", routes: [pass, pass] }), 0);
  for (const other of [{ status: "skipped" }, { status: "ran", verdict: "error" }, { status: "ran", verdict: "head_differs" }, { status: "ran", verdict: "nondeterministic_on_runner" }, { status: "ran", verdict: "product_pass_oracle_required" }]) assert.equal(exitCodeFor({ kind: KREA_SUITE, mode: "run", routes: [pass, other] }), 1);
});
