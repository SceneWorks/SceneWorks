import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, mkdtemp, readFile, readdir, realpath, rm, symlink, writeFile } from "node:fs/promises";
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

test("Windows Actions Krea uses the registered job child and canonical HF cache", () => {
  const jobs = "D:\\dev-host\\t\\abcd1234";
  const hub = "E:\\huggingface\\hub";
  const env = {
    GITHUB_ACTIONS: "true", TIER2_AB_SUITE: KREA_SUITE, TIER2_AB_BACKEND: "cuda",
    TIER2_AB_BASE_SHA: KREA_BASE_APP, TIER2_AB_HEAD_SHA: HEAD, TIER2_AB_INFERENCE_SHA: PIN,
    TIER2_AB_BASE_DIR: "C:\\runner\\base", TIER2_AB_HEAD_DIR: "C:\\runner\\head",
    RUNNER_TEMP: "C:\\runner\\temp", DEV_HOST_ROOT: "D:\\dev-host", DEV_HOST_JOB_ROOT: jobs,
    TIER2_AB_WORK: `${jobs}\\tier2-ab`, HF_HUB_CACHE: hub,
  };
  const config = (over = {}) => resolveConfig({ ...env, ...over }, { platform: "win32" });
  assert.equal(config().managedJobRoot, jobs);
  assert.equal(config().kreaCacheHub, hub);
  assert.deepEqual(config().hubs, [`${jobs}\\tier2-ab\\scratch\\krea-hf\\hub`]);
  assert.throws(() => config({ TIER2_AB_WORK: `${env.RUNNER_TEMP}\\tier2-ab` }), /DEV_HOST_JOB_ROOT/);
  assert.throws(() => config({ DEV_HOST_JOB_ROOT: "D:\\dev-host\\foreign\\abcd1234" }), /registered D:/);
  assert.throws(() => config({ HF_HUB_CACHE: "E:\\huggingface\\other" }), /canonical E:/);
  assert.doesNotThrow(() => config({ TIER2_AB_SUITE: "", TIER2_AB_WORK: `${env.RUNNER_TEMP}\\tier2-ab`, DEV_HOST_JOB_ROOT: "" }));
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
  const missingFile = kreaSnapshotFile(cacheHub, missing);
  await mkdir(path.dirname(cachedFile), { recursive: true });
  await writeFile(cachedFile, data);
  const existing = await lstat(cachedFile);
  const ref = path.join(cacheHub, `models--${CHECKPOINT.repo.replace("/", "--")}`, "refs", "main");
  await mkdir(path.dirname(ref));
  await writeFile(ref, "existing-ref");
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
  assert.deepEqual(await readFile(missingFile), config);
  assert.equal(await readFile(ref, "utf8"), "existing-ref");
  assert.equal((await lstat(cachedFile)).ino, existing.ino);
  assert.deepEqual(await readdir(path.dirname(missingFile)), [path.basename(missingFile)]);
  assert.equal((await lstat(cachedFile)).nlink, 1);
  assert.equal((await lstat(missingFile)).nlink, 1);
  assert.equal((await lstat(kreaSnapshotFile(hub, missing))).nlink, 1);
  assert.deepEqual(events.filter(({ status }) => ["download-verified", "verified"].includes(status)).map(({ status, source, receivedBytes, digest }) => [status, source, receivedBytes, digest]),
    [["verified", "runner-cache", cached.bytes, cached.sha256], ["download-verified", "pinned-https", missing.bytes, missing.gitBlobSha1], ["verified", "runner-cache", missing.bytes, missing.gitBlobSha1]]);
});

test("a concurrent exact snapshot winner is verified and kept without overwrite", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "krea-stage-race-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cacheHub = path.join(root, "cache");
  const hub = path.join(root, "attempt");
  await mkdir(cacheHub);
  const bytes = Buffer.from("valid");
  const item = { ...CHECKPOINT, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
  const candidate = kreaSnapshotFile(cacheHub, item);
  const events = [];
  let original;
  await stageKreaAssets(hub, { bytes: bytes.length, files: [item] }, new AbortController().signal, {
    cacheHub, progress: (event) => events.push(event), fetchImpl: async () => {
      await writeFile(candidate, bytes, { flag: "wx" });
      original = await lstat(candidate);
      return new Response(bytes);
    }
  });
  assert.equal((await lstat(candidate)).ino, original.ino);
  assert.equal(events.find(({ status }) => status === "download-verified").cacheDisposition, "existing");
  assert.deepEqual(await readdir(path.dirname(candidate)), [path.basename(candidate)]);
});

test("canonical staging leaves redirects, bad network bytes and corrupt race winners untouched", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "krea-stage-unsafe-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cacheHub = path.join(root, "cache"), external = path.join(root, "external");
  await mkdir(cacheHub);
  await mkdir(external);
  const bytes = Buffer.from("valid");
  const item = { ...CHECKPOINT, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
  const price = { bytes: bytes.length, files: [item] };
  const candidate = kreaSnapshotFile(cacheHub, item);
  const repoDir = path.join(cacheHub, `models--${item.repo.replace("/", "--")}`);
  await symlink(external, repoDir, process.platform === "win32" ? "junction" : "dir");
  const signal = new AbortController().signal;
  await assert.rejects(stageKreaAssets(path.join(root, "redirect"), price, signal, {
    cacheHub, fetchImpl: () => assert.fail("redirected canonical parent must refuse before HTTPS"),
  }), /rejects symlink, junction/);
  assert.deepEqual(await readdir(external), []);
  await rm(repoDir, { recursive: true });
  await assert.rejects(stageKreaAssets(path.join(root, "bad-download"), price, signal, {
    cacheHub, fetchImpl: async () => new Response("bogus"),
  }), /wrong artifact digest/);
  assert.deepEqual(await readdir(path.dirname(candidate)), []);
  await assert.rejects(stageKreaAssets(path.join(root, "bad-winner"), price, signal, {
    cacheHub, fetchImpl: async () => {
      await writeFile(candidate, "bogus", { flag: "wx" });
      return new Response(bytes);
    },
  }), /wrong artifact digest/);
  assert.equal(await readFile(candidate, "utf8"), "bogus");
  assert.deepEqual(await readdir(path.dirname(candidate)), [path.basename(candidate)]);
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

test("native Hires.fix traces permit the public snapshots observed on both Windows arms", () => {
  const [, route] = kreaRoutes("cuda");
  const nativePasses = (scoped) => [8, 2].map((steps, index) => {
    const size = index === 0 ? 512 : 768;
    return { event: "sgaf336_krea_hires_progress", engine: "krea_2_turbo", width: size, height: size,
      authored_steps: index === 0 ? 8 : 4, reference_count: index, scope_configured: scoped,
      context_present: scoped, context_width: scoped ? size : 0, context_height: scoped ? size : 0,
      context_reference_count: scoped ? index : 0, decode_count: 1, valid: true,
      coordinates: JSON.stringify(Array.from({ length: steps }, (_, step) => [step + 1, steps])) };
  });
  const nativeLog = (passes) => passes.map((fields) => JSON.stringify({ fields })).join("\r\n");
  const publicEvents = (steps, decodes) => [
    ...steps.map((step) => ({ message: `Generating step ${step}/12.`, status: "running", stage: "generating", publishedAssets: 0 })),
    ...Array.from({ length: decodes }, () => ({ message: "Decoding.", status: "running", stage: "generating", publishedAssets: 0 })),
    { message: "Generated 1 output(s).", status: "completed", stage: "completed", publishedAssets: 1, publishedAssetId: "asset_saved" },
  ];
  // Run 45039550044 saw 7/12 and 9/12 followed by 14 decode snapshots on the baseline.
  // Its candidate saw all ten step labels, with duplicates, followed by three decode snapshots.
  const base = publicEvents([7, 9], 14);
  const head = publicEvents([1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10], 3);
  const baseOptions = { nativeLog: nativeLog(nativePasses(false)), assetId: "asset_saved" };
  const headOptions = { nativeLog: nativeLog(nativePasses(true)), requireScope: true, assetId: "asset_saved" };
  assert.deepEqual(assertKreaEvents(base, route, baseOptions).observedSteps, [[7, 12], [9, 12]]);
  const headProof = assertKreaEvents(head, route, headOptions);
  assert.deepEqual(headProof.executedStepSlices, [8, 2]);
  assert.equal(headProof.decodingUpdates, 3);
  assert.deepEqual(headProof.nativePasses[1].coordinates, [[1, 2], [2, 2]]);
  assert.throws(() => assertKreaEvents(base, route), /clipped schedule/);
  assert.throws(() => assertKreaEvents(base, route, { nativeLog: "" }), /two completed native/);
  assert.throws(() => assertKreaEvents(base, route, { ...baseOptions, requireScope: true }), /required scope/);
  for (const steps of [[], [9, 7], [0, 9], [7, 11], [7, 9, 7]]) {
    assert.throws(() => assertKreaEvents(publicEvents(steps, 1), route, baseOptions), /ordered subsequence/);
  }
  const earlyAsset = [{ ...base[0], publishedAssets: 1 }, ...base.slice(1)];
  assert.throws(() => assertKreaEvents(earlyAsset, route, baseOptions), /intermediate output/);
  const recovering = { ...base.at(-1), publishedAssets: 0, publishedAssetId: undefined };
  assert.deepEqual(assertKreaEvents([...base.slice(0, -1), recovering, base.at(-1), base.at(-1)], route, baseOptions).observedSteps,
    [[7, 12], [9, 12]]);
  const wrongAsset = { ...base.at(-1), publishedAssetId: "asset_other" };
  assert.throws(() => assertKreaEvents([...base, wrongAsset], route, baseOptions), /one saved asset/);
  const wrongDecodeStage = base.map((event) => event.message === "Decoding." ? { ...event, stage: "loading_model" } : event);
  assert.throws(() => assertKreaEvents(wrongDecodeStage, route, baseOptions), /decoding must belong to a running render/);

  const validPasses = nativePasses(true);
  const invalidRecords = [
    [{ coordinates: "[[1,2],[1,2]]" }, /exact clipped schedule/],
    [{ coordinates: "[[1,4],[2,4]]" }, /exact clipped schedule/],
    [{ valid: false }, /one decode/],
    [{ decode_count: 2 }, /one decode/],
    [{ width: 512 }, /pass geometry/],
    [{ reference_count: 0 }, /pass geometry/],
    [{ context_reference_count: 0 }, /admission geometry/],
  ];
  for (const [wrong, error] of invalidRecords) {
    const corrupted = [validPasses[0], { ...validPasses[1], ...wrong }];
    assert.throws(() => assertKreaEvents(head, route, { ...headOptions, nativeLog: nativeLog(corrupted) }), error);
  }
  assert.throws(() => assertKreaEvents(head, route, { nativeLog: nativeLog(validPasses.slice(0, 1)) }), /two completed native/);
  assert.throws(() => assertKreaEvents(jobEvents(RAW).slice(1), RAW, headOptions), /authored phase slices/);
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
  for (const id of ["unrelated", "requested"]) output.enqueue(encoder.encode(`event: job.updated\ndata: ${JSON.stringify({ id, status: "completed", payload: "unneeded", result: { assets: [{ id: "asset_saved" }] } })}\n\n`));
  const rows = await events.collect("requested", async (probe) => { for (let i = 0; i < 10; i++) { if (await probe()) return; await new Promise(setImmediate); } throw new Error("no target update"); });
  assert.deepEqual(rows.map(({ id, status, payload }) => ({ id, status, payload })), [{ id: "requested", status: "completed", payload: undefined }]);
  assert.equal(rows[0].publishedAssetId, "asset_saved");
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

function rawOutputProof(dataRoot, locator) {
  const body = RAW.body;
  const publicAdapter = {
    id: KREA_RAW_LORA, name: "Krea Darkbrush phase proof", weight: 1, scope: "project",
    source: { provider: "local", repo: null }, presetManaged: false, triggerWords: [],
  };
  const diskAdapter = {
    ...publicAdapter, installedPath: locator, modelPath: locator, path: locator, sourcePath: locator,
    source: { ...publicAdapter.source, path: locator },
  };
  const raw = {
    repo: "SceneWorks/krea-2-raw-mlx", samplingRegime: "krea_2_raw_multiphase", realModelInference: true,
    multiPhase: true, mlxQuantize: 4, numInferenceSteps: 10, phases: structuredClone(body.advanced.phases),
  };
  const recipe = { model: body.model, seed: body.seed, prompt: body.prompt, rawAdapterSettings: raw, loras: [diskAdapter] };
  const sidecar = { id: "asset", file: { width: 512, height: 512 }, recipe };
  const asset = { ...sidecar, recipe: { ...recipe, loras: [publicAdapter] } };
  const libraryAsset = {
    ...sidecar, projectId: "p", lineage: { jobId: "job" }, generationSetId: "set",
    generationSet: { id: "set", jobId: "job", recipe, prompt: body.prompt },
  };
  const share = {
    prompt: body.prompt, negativePrompt: body.negativePrompt,
    loras: [{ name: "krea2_darkbrush", weight: 1, hash: KREA_RAW_ADAPTER.sha256 }],
  };
  return structuredClone({
    route: RAW, dataRoot, width: 512, height: 512, asset, library: [libraryAsset], sidecar, share,
    workerId: "worker", projectId: "p", adapterImport: { adapterId: KREA_RAW_LORA, scope: "project", source: KREA_RAW_ADAPTER },
    job: { id: "job", workerId: "worker", payload: body, result: { model: body.model, generationSetId: "set", assets: [asset], assetIds: [asset.id] } },
  });
}

test("Raw compares the exact public job projection while keeping the full generation-set recipe", () => {
  const segments = ["models", "portable-attempts", "a".repeat(64), "bundle", "model0", "member1"];
  const dataRoot = "/runner/run/state/data";
  const locator = `${path.posix.join(dataRoot, ...segments)}/`;
  const proof = rawOutputProof(dataRoot, locator);
  assert.deepEqual(assertKreaOutput(proof).raw.phasePlan, RAW.body.advanced.phases);
  assert.deepEqual(proof.sidecar.recipe.loras, proof.library[0].generationSet.recipe.loras);
  assert.equal(proof.sidecar.recipe.loras[0].source.path, locator);

  const windowsRoot = "D:\\runner\\run\\state\\data";
  const windowsLocator = path.win32.toNamespacedPath(path.win32.join(windowsRoot, ...segments));
  assert.equal(assertKreaOutput(rawOutputProof(windowsRoot, windowsLocator)).raw.tier, "q4");
  assert.throws(() => assertKreaOutput(rawOutputProof(windowsRoot, path.win32.join("\\", ...segments))), /absolute bundle path/);

  function rejects(change, message) {
    const altered = rawOutputProof(dataRoot, locator);
    change(altered);
    assert.throws(() => assertKreaOutput(altered), message);
  }
  for (const key of ["installedPath", "modelPath", "path", "sourcePath"]) {
    rejects(({ asset }) => { asset.recipe.loras[0][key] = locator; }, /exact redacted disk sidecar/);
    rejects(({ sidecar }) => { delete sidecar.recipe.loras[0][key]; }, /absolute bundle path|must match its installed locator/);
  }
  rejects(({ asset }) => { asset.recipe.loras[0].source.path = locator; }, /exact redacted disk sidecar/);
  rejects(({ asset }) => { asset.recipe.loras[0].weight = 0.75; }, /exact redacted disk sidecar/);
  rejects(({ asset }) => { delete asset.recipe.loras[0].triggerWords; }, /exact redacted disk sidecar/);
  rejects(({ asset, library }) => { library[0].generationSet.recipe = asset.recipe; }, /generation set must preserve persisted recipe/);
  rejects(({ sidecar }) => { delete sidecar.recipe.loras[0].source.path; }, /source.path must match its installed locator/);
  rejects(({ sidecar }) => { sidecar.recipe.loras[0].sourcePath = `${locator}/other`; }, /sourcePath must match its installed locator/);
  assert.throws(() => assertKreaOutput(rawOutputProof(`${dataRoot}-other`, locator)), /owned portable bundle member/);
  assert.throws(() => assertKreaOutput(rawOutputProof(dataRoot, locator.replace("/bundle/", "/bundle/../bundle/"))), /absolute bundle path/);
  assert.throws(() => assertKreaOutput(rawOutputProof(dataRoot, locator.replace("/member1/", "/member2/"))), /owned portable bundle member/);
  rejects(({ adapterImport }) => { adapterImport.source.sha256 = "b".repeat(64); }, /pinned project adapter import/);
  rejects(({ share }) => { share.loras[0].path = locator; }, /exact shareable adapter attribution/);
  rejects(({ share }) => { share.loras[0].hash = "b".repeat(64); }, /exact shareable adapter attribution/);
  rejects(({ sidecar }) => { sidecar.recipe.rawAdapterSettings.phases[1].loras[0].weight = 0.5; }, /0.75/);
});

test("a skipped, refused or incomparable arm cannot pass the Krea product suite", () => {
  const pass = { status: "ran", verdict: "equal" };
  assert.equal(exitCodeFor({ kind: KREA_SUITE, mode: "run", routes: [pass, pass] }), 0);
  for (const other of [{ status: "skipped" }, { status: "ran", verdict: "error" }, { status: "ran", verdict: "head_differs" }, { status: "ran", verdict: "nondeterministic_on_runner" }, { status: "ran", verdict: "product_pass_oracle_required" }]) assert.equal(exitCodeFor({ kind: KREA_SUITE, mode: "run", routes: [pass, other] }), 1);
});
