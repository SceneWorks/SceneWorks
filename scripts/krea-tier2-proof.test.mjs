import assert from "node:assert/strict";
import test from "node:test";
import { dispatchOptions, exitCodeFor, planRoutes, readCatalog, readPatternEvidence, resolveConfig } from "./tier2-ab.mjs";
import {
  KREA_SUITE, KREA_BASE_APP, KREA_BASE_ENGINE, CHECKPOINT, LINKED_MODEL, assertKreaEvents,
  assertKreaImportStamp, assertKreaEnhancementRefused, assertKreaOutput, kreaRoutes,
  kreaAssetInventory, kreaExpectedStepEvents, kreaPerformance, openKreaEvents, priceKreaAssets,
} from "./krea-tier2-proof.mjs";

const HEAD = "1".repeat(40);
const PIN = "2".repeat(40);
const [RAW, HIRES] = kreaRoutes("mlx");

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
  assert.throws(() => resolveConfig({ ...env, TIER2_AB_BASE_SHA: "3".repeat(40) }, { platform: "darwin" }), /baseline app is frozen/);
  assert.throws(() => resolveConfig({ ...env, TIER2_AB_HUBS: "/other" }, { platform: "darwin" }), /attempt-owned/);
});

test("pinned inventory preserves the Raw q4 tier, exact adapter and linked BF16 with only the companion surface", async () => {
  const evidence = await readPatternEvidence(".");
  const inventory = kreaAssetInventory(evidence, [RAW, HIRES]);
  assert.equal(inventory.length, 23);
  assert.ok(inventory.some((item) => item.file.match(/^q4\/transformer\/.*\.safetensors$/)));
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
  for (const id of ["unrelated", "requested"]) output.enqueue(encoder.encode(`event: job.updated\ndata: ${JSON.stringify({ id, status: "completed", payload: "unneeded" })}\n\n`));
  const rows = await events.collect("requested", async (probe) => { for (let i = 0; i < 10; i++) { if (await probe()) return; await new Promise(setImmediate); } throw new Error("no target update"); });
  assert.deepEqual(rows.map(({ id, status, payload }) => ({ id, status, payload })), [{ id: "requested", status: "completed", payload: undefined }]);
  await events.close();
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
  const oracle = { status: "ran", verdict: "product_pass_oracle_required" };
  assert.equal(exitCodeFor({ kind: KREA_SUITE, mode: "run", routes: [pass, oracle] }), 0);
  for (const other of [{ status: "skipped" }, { status: "ran", verdict: "error" }, { status: "ran", verdict: "nondeterministic_on_runner" }]) assert.equal(exitCodeFor({ kind: KREA_SUITE, mode: "run", routes: [pass, other] }), 1);
});
