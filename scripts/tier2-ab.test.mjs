// SW-7033 Tier 2 A/B: pure-function and static-workflow contracts only. No build, API, worker,
// model, GPU or hub is touched; synthetic PNG/WAV bytes and temp directories only.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, rm } from "node:fs/promises";
import { deflateSync } from "node:zlib";
import os from "node:os";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import { sha256 } from "./lib/file-sha256.mjs";
import {
  DEFAULT_ROUTE_IDS, ROUTES, contentDigest, decodePng, decodedContent, descendants, diffObjects, dispatchOptions,
  cudaLeaseMinFreeMiB, scailCudaMemoryViolation, scailCudaSampler, childEnvironments, childStdio, exitCodeFor, expandSelector, readPatternEvidence, selectorProblems, assertTrustedShas, scrubSecrets, TRUSTED_REF_PATTERNS, trustedRefFor, ffprobePath, gpuInventory, hubFingerprint, memoryCell, streamShape, HUB_MUTATION_LISTING_LIMIT, hubMutationAttribution, processState, identitySubset, jobClaimed, jobDiagnostics, jobProgressDecision, lastLines, numericDifference, progressLine, removeScratch, parseTopMem, parseWav, planRoutes, readCatalog, redactPaths, referenceWav,
  renderMarkdown, reportName, requestBody, requestTemplate, resolveConfig, routeDownloadRows, routePresence, selectRoutes,
  summarize, verdictFor, uploadReference, wanClaimSource, wanObservations, stageWanSnapshots, readWanClaim, verifyWanRecipe, wanExecutionFacts,
} from "./tier2-ab.mjs";

const BASE = "d3f7b05bf22826e44fd43da19cff0985fb17d68b", HEAD = "d2b0c76c9b9b9e00b5174aceec61c11a186dac68";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const env = (changes = {}) => ({ TIER2_AB_BACKEND: "mlx", TIER2_AB_BASE_SHA: BASE, TIER2_AB_HEAD_SHA: HEAD, TIER2_AB_BASE_DIR: "/w/base", TIER2_AB_HEAD_DIR: "/w/head", TIER2_AB_WORK: "/t/work", ...changes });

test("inputs: exact 40-hex SHAs, distinct sides, absolute paths, known backend-matched routes", () => {
  const cfg = resolveConfig(env(), { platform: "darwin", home: "/Users/runner" });
  assert.deepEqual(cfg.routes.map((route) => route.id), DEFAULT_ROUTE_IDS.mlx);
  assert.equal(cfg.hubs[0], "/Volumes/Models/huggingface/hub");
  for (const bad of [{ TIER2_AB_BASE_SHA: "main" }, { TIER2_AB_HEAD_SHA: HEAD.slice(0, 39) }, { TIER2_AB_HEAD_SHA: HEAD.toUpperCase() }, { TIER2_AB_HEAD_SHA: BASE }, { TIER2_AB_BACKEND: "candle" }, { TIER2_AB_BASE_DIR: "relative" }, { TIER2_AB_HEAD_DIR: "/w/base" }, { TIER2_AB_WORK: "/w/base/work" }, { TIER2_AB_INFERENCE_SHA: "abc" }, { TIER2_AB_ROUTES: "cuda-sdxl-q4-t2i" }, { TIER2_AB_ROUTES: "nope" }, { TIER2_AB_ROUTES: "mlx-moss-image-t2i,mlx-moss-image-t2i" }, { TIER2_AB_HUBS: "relative/hub" }]) {
    assert.throws(() => resolveConfig(env(bad), { platform: "darwin" }), undefined, JSON.stringify(bad));
  }
  assert.throws(() => resolveConfig(env({ GITHUB_ACTIONS: "true", RUNNER_TEMP: "/r" }), { platform: "darwin" }), /under RUNNER_TEMP/);
  assert.equal(resolveConfig(env({ GITHUB_ACTIONS: "true", RUNNER_TEMP: "/t" }), { platform: "darwin" }).work, "/t/work");
  const cuda = resolveConfig(env({ TIER2_AB_BACKEND: "cuda", TIER2_AB_BASE_DIR: "D:\\w\\base", TIER2_AB_HEAD_DIR: "D:\\w\\head", TIER2_AB_WORK: "D:\\t\\work", TIER2_AB_ROUTES: '["cuda-moss-image-t2i"]', TIER2_AB_HUBS: "E:\\hub;F:\\hub2" }), { platform: "win32" });
  assert.deepEqual(cuda.routes.map((route) => route.id), ["cuda-moss-image-t2i"]);
  assert.deepEqual(cuda.hubs, ["E:\\hub", "F:\\hub2"]);
  assert.deepEqual(selectRoutes("cuda", "").map((route) => route.id), DEFAULT_ROUTE_IDS.cuda);
});

test("physical GPU0 requires explicit SCAIL or CUDA Wan A14B routes at dispatch and on the runner", () => {
  const select = (backend, routes, cuda_physical_index) => dispatchOptions({ profile: "tier2-ab", backend, ref: BASE,
    moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, routes, cuda_physical_index }) }, BASE);
  const worker = (routes, index) => resolveConfig(env({ TIER2_AB_BACKEND: "cuda", TIER2_AB_HUBS: "/hub",
    TIER2_AB_ROUTES: JSON.stringify(routes), TIER2_AB_PHYSICAL_GPU: index }), { platform: "darwin" });
  for (const id of Object.keys(ROUTES).filter((id) => id.startsWith("cuda-"))) {
    if (ROUTES[id].scailCandidate || ROUTES[id].wanA14b) {
      assert.equal(select("candle", [id], "0").cuda_physical_index, "0");
      assert.equal(worker([id], "0").physicalGpu, "0");
    } else {
      assert.throws(() => select("candle", [id], "0"), /physical GPU0/);
      assert.throws(() => worker([id], "0"), /physical GPU0/);
    }
    assert.equal(select("candle", [id]).cuda_physical_index, "1");
  }
  for (const value of [0, "all", [0], "2", null]) assert.throws(() => select("candle", ["cuda-scail2-q4-animate-2"], value));
  assert.throws(() => select("mlx", ["mlx-scail2-q4-animate-2"], "0"));
  assert.throws(() => select("candle", undefined, "0"));
  const envs = childEnvironments({CUDA_VISIBLE_DEVICES:"0", NVIDIA_VISIBLE_DEVICES:"0"}, {
    backend: "cuda", physicalGpu:"0", state:"/s", hub:"/h", xet:"/x", workerId:"w" });
  for (const child of [envs.api, envs.worker("http://127.0.0.1")])
    for (const key of ["SCENEWORKS_GPU_ID", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"]) assert.equal(child[key], "0");
});

test("CUDA SCAIL leases and requests preserve the child cap and physical free reserve", () => {
  const gib = 1024 ** 3;
  assert.equal(cudaLeaseMinFreeMiB(""), 75776);
  for (const route of Object.values(ROUTES).filter((r) => r.backend === "cuda")) {
    assert.equal(cudaLeaseMinFreeMiB(JSON.stringify([route.id])), route.scailCandidate ? 92160 : 75776);
  }
  assert.equal(cudaLeaseMinFreeMiB("cuda-wan22-a14b-t2v-uni_pc,cuda-scail2-q4-animate-2"), 92160);
  assert.equal(scailCudaMemoryViolation({ usedBytes: 5 * gib, freeBytes: 90 * gib }, true), null);
  assert.match(scailCudaMemoryViolation({ usedBytes: 5 * gib, freeBytes: 90 * gib - 1 }, true), /90 GiB/);
  assert.equal(scailCudaMemoryViolation({ usedBytes: 84 * gib, freeBytes: 6 * gib }), null);
  assert.match(scailCudaMemoryViolation({ usedBytes: 84 * gib + 1, freeBytes: 7 * gib }), /84 GiB/);
  assert.match(scailCudaMemoryViolation({ usedBytes: 50 * gib, freeBytes: 6 * gib - 1 }), /6 GiB/);
  assert.match(scailCudaMemoryViolation({ usedBytes: 50 * gib, freeBytes: null }), /measure free/);
});

test("a final SCAIL CUDA guard breach stops the owned worker before cooperative cleanup", { timeout: 5000 }, async (t) => {
  const child = spawn(process.execPath, ["-e", 'process.on("SIGTERM", () => {}); process.stdin.resume(); process.send("ready"); setInterval(() => {}, 60000);'],
    { stdio: ["pipe", "ignore", "ignore", "ipc"] });
  t.after(() => { if (processState(child).alive) child.kill("SIGKILL"); });
  const exit = once(child, "exit");
  await once(child, "message");
  const finalRead = Promise.withResolvers();
  const controller = new AbortController(), record = { status: "running", watchdog: {} };
  const read = scailCudaSampler(() => finalRead.promise, child, record, controller, "0");
  const pending = read();
  record.status = "refused"; // The typed job result returned while nvidia-smi was still running.
  finalRead.resolve({ usedBytes: 50 * 1024 ** 3, freeBytes: 5 * 1024 ** 3 });
  await assert.rejects(pending, /6 GiB/);
  assert.equal(controller.signal.aborted, true);
  assert.equal(record.status, "error");
  assert.equal(record.watchdog.gpuStop.pid, child.pid);
  assert.equal(record.watchdog.gpuStop.killIssued, true);
  await exit; // No cancellation POST, sentinel exit or normal 60-second teardown grace.
  assert.equal(processState(child).alive, false);
});

test("SCAIL GPU telemetry failure and insufficient final baseline stop before submission", async () => {
  let kills = 0;
  const worker = { label: "worker", pid: 42, exitCode: null, signalCode: null,
    kill(signal) { assert.equal(signal, "SIGKILL"); kills++; return true; } };
  const gib = 1024 ** 3;
  const baseline = { usedBytes: 5 * gib, freeBytes: 90 * gib };
  for (const [read, expected] of [
    [() => Promise.reject(new Error("device query failed")), /telemetry unavailable/],
    [() => ({ ...baseline, freeBytes: baseline.freeBytes - 1 }), /90 GiB/],
    [() => ({ ...baseline, usedBytes: 84 * gib + 1 }), /84 GiB/],
  ]) {
    const record = { watchdog: {}, status: "running" }, controller = new AbortController();
    assert.deepEqual(await scailCudaSampler(() => baseline, worker, record, controller, "0")(true), baseline);
    const before = kills, fault = scailCudaSampler(read, worker, record, controller, "0");
    await assert.rejects(fault(true), expected);
    await assert.rejects(fault(true), expected);
    assert.equal(kills, before + 1);
    assert.equal(record.status, "error");
    assert.equal(controller.signal.aborted, true);
  }
});

test("default routes send one fixed request per route and the same JSON to both sides", () => {
  for (const ids of Object.values(DEFAULT_ROUTE_IDS)) for (const id of ids) {
    const route = ROUTES[id], template = requestTemplate(route);
    assert.equal(typeof route.body.seed, "number", `${id} needs a fixed seed`);
    assert.equal(route.body.model, route.model);
    assert.match(template.sha256, /^[0-9a-f]{64}$/);
    const a = requestBody(route, "p-1", "ref-1"), b = requestBody(route, "p-2", "ref-2");
    const strip = (body) => JSON.stringify({ ...body, projectId: "<project>", ...(route.referenceAudio ? { referenceAudioAssetId: "<reference-audio>" } : {}) });
    assert.equal(strip(a), strip(b)); assert.equal(strip(a), JSON.stringify(template.body));
  }
});

test("dispatch selector maps catalog inputs, defaults to preflight, refuses mismatches", () => {
  const inputs = { profile: "tier2-ab", ref: "a".repeat(40), backend: "candle", inference_sha: "", moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD }) };
  assert.deepEqual(dispatchOptions(inputs, inputs.ref), { base_sha: BASE, head_sha: HEAD, backend: "cuda", cuda_physical_index: "1", routes: "", hubs: "", preflight_only: true, inference_sha: "" });
  const custom = dispatchOptions({ ...inputs, backend: "mlx", moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, routes: ["mlx-moss-image-t2i"], hubs: ["/Volumes/X/hub"], preflight_only: false }) }, inputs.ref);
  assert.equal(custom.preflight_only, false); assert.equal(custom.routes, '["mlx-moss-image-t2i"]'); assert.equal(custom.hubs, '["/Volumes/X/hub"]');
  for (const bad of [{ ref: "b".repeat(40) }, { backend: "cpu" }, { download_missing: true }, { moss_options_json: "[]" }, { moss_options_json: "{" }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, case: "x" }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: BASE }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, routes: ["mlx-moss-image-t2i"] }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, hubs: ["/posix/hub"] }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, preflight_only: "no" }) }, { inference_sha: "x" }]) {
    assert.throws(() => dispatchOptions({ ...inputs, ...bad }, inputs.ref), undefined, JSON.stringify(bad));
  }
});

test("Wan A14B routes freeze the supported product matrix and the I2V input through multipart", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-wan-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const models = await readCatalog(root);
  const routes = Object.values(ROUTES).filter((route) => route.wanA14b);
  assert.equal(routes.length, 6);
  for (const route of routes) {
    const { body } = route;
    assert.equal(DEFAULT_ROUTE_IDS[route.backend].includes(route.id), false);
    const mode = route.referenceImage ? "i2v" : "t2v";
    assert.equal(body.model, `wan_2_2_${mode}_14b`);
    assert.deepEqual([body.width, body.height, body.duration, body.fps, body.seed], [832, 480, 1, 16, 33401]);
    assert.deepEqual(body.advanced, { mlxQuantize: 4, steps: 6, guidanceScale: 3.5,
      sampler: route.id.endsWith("heun") ? "heun" : "uni_pc", lightning: false });
    assert.deepEqual(body.loras, []);
    const platform = route.backend === "mlx" ? "darwin" : "win32";
    const fs = { stat: async () => ({ isFile: () => true, size: 1 }) };
    const [plan] = await planRoutes({ routes: [route], hubs: [dir], baseCatalog: models, headCatalog: models,
      evidence: await readPatternEvidence(root), platform, fs });
    assert.equal(plan.status, "ready", plan.skipReason);
    assert.match(plan.rows.required.find((r) => r.variant)?.repo, new RegExp(`wan2.2-${mode}-a14b-${platform === "darwin" ? "mlx" : "candle"}$`));
    if (!route.referenceImage) { assert.equal(Object.hasOwn(plan.request.body, "sourceAssetId"), false); continue; }
    const endpoint = { async exchange(method, url, form, options) {
      assert.equal(method, "POST"); assert.equal(url, "/api/v1/projects/proj1/assets");
      assert.deepEqual(options, { multipart: true });
      const upload = await new Response(form).formData();
      assert.deepEqual([...upload.keys()], ["file"]);
      const file = upload.get("file");
      assert.equal(file.type, "image/png"); assert.equal(file.name, "source.png");
      const bytes = Buffer.from(await file.arrayBuffer());
      assert.deepEqual(bytes, await readFile(path.join(dir, "source.png")));
      const { width, height } = decodePng(bytes);
      assert.deepEqual([width, height], [832, 480]);
      assert.equal(sha256(bytes), plan.request.sourceImageSha256);
      return { status: 201, body: { id: "the-uploaded-asset", projectId: "proj1", type: "image", file: { path: "assets/images/source image.png" } } };
    }, async served(url) {
      assert.equal(url, "/api/v1/projects/proj1/files/assets/images/source%20image.png");
      return { bytes: await readFile(path.join(dir, "source.png")) };
    } };
    const ref = await uploadReference(route, endpoint, "proj1", dir);
    const bound = requestBody(route, "proj1", ref.id);
    assert.equal(bound.sourceAssetId, "the-uploaded-asset");
    const normalized = { ...bound, projectId: "<project>", sourceAssetId: "<source-image>" };
    assert.equal(sha256(JSON.stringify({ body: normalized, sourceImageSha256: ref.sha256 })), plan.request.sha256);
  }
  assert.throws(() => selectRoutes("cuda", "cuda-wan22-a14b-t2v-heun"), /unknown route/);
});

test("Wan compares the accepted q4 source and worker settings, including a valid decoded stream", async (t) => {
  const route = ROUTES["mlx-wan22-a14b-i2v-heun"];
  const [row] = routeDownloadRows((await readCatalog(root)).find((m) => m.id === route.model),
    { platform: "darwin", tier: "q4" }).required.filter((r) => r.variant);
  const plan = { rows: { required: [row] } };
  const file = { relativePath: "q4/transformer.safetensors", sizeBytes: 123, sha256: `sha256:${"a".repeat(64)}` };
  const selected = { backend: "mlx", platform: "macos", models: [{ modelId: route.model, backend: "mlx", slot: "modelManifestEntry",
    tier: "q4", variant: "q4", members: [{ role: "primary", tier: "q4", files: [file], source: {
      kind: "revision", identity: { repository: row.repo, revision: row.revision, variant: "q4" } } }] }] };
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-claim-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const dbPath = path.join(dir, "jobs.db");
  const { DatabaseSync } = await import("node:sqlite");
  const db = new DatabaseSync(dbPath);
  try {
    db.exec("CREATE TABLE jobs (id TEXT, status TEXT, worker_id TEXT, execution_claim_source_route_json TEXT)");
    db.prepare("INSERT INTO jobs VALUES (?, ?, ?, ?)").run("job", "completed", "owned", JSON.stringify(selected));
  } finally { db.close(); }
  assert.deepEqual(await readWanClaim(dbPath, "job", "owned"), selected);
  await assert.rejects(readWanClaim(dbPath, "job", "other"), /owned completed job/);
  const source = wanClaimSource(route, plan, selected);
  assert.equal(source.revision, row.revision);
  assert.throws(() => wanClaimSource(route, plan, { ...selected, backend: "candle" }), /backend/);
  const member = selected.models[0].members[0];
  member.source.identity.revision = "b".repeat(40);
  assert.throws(() => wanClaimSource(route, plan, selected), /snapshot/);
  member.source.identity.revision = row.revision;
  file.relativePath = "q8/transformer.safetensors";
  assert.throws(() => wanClaimSource(route, plan, selected), /file closure/);
  file.relativePath = "q4/transformer.safetensors";
  file.sha256 = `sha256:${"b".repeat(64)}`;
  assert.notEqual(wanClaimSource(route, plan, selected).boundMembersSha256, source.boundMembersSha256);
  const recipe = { mode: route.mode, model: route.model, adapter: "mlx_wan", prompt: route.body.prompt, seed: route.body.seed, rawAdapterSettings: { realModelInference: true, lightning: false },
    normalizedSettings: { sourceAssetId: "uploaded" }, loras: [] };
  verifyWanRecipe(route, recipe, "uploaded", row.repo);
  for (const changed of [{ adapter: "procedural_video" }, { loras: [{ id: "extra" }] }, { normalizedSettings: {} }])
    assert.throws(() => verifyWanRecipe(route, { ...recipe, ...changed }, "uploaded", row.repo));
  const metrics = { model: route.model, backend: "mlx", sampler: "heun", steps: 6, guidanceScale: 3.5, seed: 33401,
    width: 832, height: 480, quantLabel: "bf16", sampleMs: 999, totalMs: 5999, peakMemoryBytes: 10999 };
  const content = { method: "ffmpeg_rgba_frames+s16le_audio+ffprobe_streams", shape: { video: { width: 832, height: 480, decodedFrames: 16, avgFrameRate: "16/1" } } };
  assert.equal(wanObservations(route, metrics, content).progressSampleIntervalMs, 999);
  for (const bad of [{ backend: "cuda" }, { sampler: "default" }, { steps: 4 }, { sampleMs: null }])
    assert.throws(() => wanObservations(route, { ...metrics, ...bad }, content), /Wan worker/);
  assert.throws(() => wanObservations(route, metrics, { ...content, method: "file_bytes_no_probe" }), /decode/);
});

test("Wan validates each provider's decoded frame count and actual solver from its execution receipt", () => {
  for (const route of Object.values(ROUTES).filter((r) => r.wanA14b)) {
    const mlx = route.backend === "mlx";
    const backend = mlx ? "mlx" : "candle";
    const frames = mlx ? 16 : 13;
    const { model, width, height, fps, advanced, seed } = route.body;
    const metrics = { model, width, height, seed, backend: route.backend, steps: advanced.steps, sampler: advanced.sampler,
      guidanceScale: advanced.guidanceScale, quantLabel: mlx ? "bf16" : "q4", totalMs: 99, sampleMs: 33 };
    const content = { method: "ffmpeg_rgba_frames+s16le_audio+ffprobe_streams", shape: { video: {
      width, height, avgFrameRate: `${fps}/1`, decodedFrames: frames } } };
    const obs = wanObservations(route, metrics, content);
    assert.equal(obs.engineRequestFrames, 13); assert.equal(obs.decodedFrames, frames);
    content.shape.video.decodedFrames = mlx ? 13 : 16;
    assert.throws(() => wanObservations(route, metrics, content), /output must decode/);

    const solver = mlx ? (advanced.sampler === "heun" ? "Heun" : "UniPC") : "wan_flow_unipc_bh2_order2_v1";
    const stage = advanced.sampler === "heun" ? "wan_curated" : "wan";
    const digest = { element: mlx ? "f32_le_bits" : "f64_le_bits", count: 7, sha256: "2".repeat(64) };
    const provider = route.referenceImage ? "wan2_2_i2v_14b" : "wan2_2_t2v_14b";
    const identity = { provider_id: provider, backend, sampling: { state: "resolved", value: [mlx
      ? { stage, solver, estimator: "flow_velocity" } : solver] },
      schedule: { state: "resolved", value: [mlx ? { stage, digest } : { sampler: solver, schedule: digest, estimator: "flow_velocity" }] } };
    const receipt = { jobId: "job", version: 1, acceptedBackend: backend, outputs: [
      { artifactIds: ["video"], execution: { state: "resolved", identity } }] };
    assert.deepEqual(wanExecutionFacts(route, receipt, "job", "video"), { provider, backend, solver, schedule: digest });
    assert.throws(() => wanExecutionFacts(route, receipt, "job", "different-asset"), /bind the saved video/);
    identity.provider_id = "wan2_2_ti2v_5b";
    assert.throws(() => wanExecutionFacts(route, receipt, "job", "video"), /provider substitution/);
    identity.provider_id = provider;
    digest.count = 6;
    assert.throws(() => wanExecutionFacts(route, receipt, "job", "video"), /schedule digest/);
    digest.count = 7;
    identity.sampling.value = [mlx ? { stage, solver: "Euler", estimator: "flow_velocity" } : "wan_flow_euler_v1"];
    assert.throws(() => wanExecutionFacts(route, receipt, "job", "video"), /executed solver/);
    receipt.outputs[0].execution = { state: "unavailable", reason: { detail: "missing runtime schedule" } };
    assert.throws(() => wanExecutionFacts(route, receipt, "job", "video"), /missing runtime schedule/);
  }
});

test("explicit MLX Wan stages pinned rows before comparison and refuses mismatched sources", async () => {
  const route = ROUTES["mlx-wan22-a14b-t2v-uni_pc"];
  const cfg = { backend: "mlx", routes: [route], hubs: ["/owned/hub"] };
  const input = { profile: "tier2-ab", backend: "mlx", ref: BASE, download_missing: true };
  const options = { base_sha: BASE, head_sha: HEAD, routes: [route.id], preflight_only: false };
  assert.equal(dispatchOptions({ ...input, moss_options_json: JSON.stringify(options) }, BASE).backend, "mlx");
  for (const bad of [{ preflight_only: true }, { routes: DEFAULT_ROUTE_IDS.mlx }])
    assert.throws(() => dispatchOptions({ ...input, moss_options_json: JSON.stringify({ ...options, ...bad }) }, BASE));
  const models = await readCatalog(root);
  const paths = { routes: [route], hubs: cfg.hubs, baseCatalog: models, headCatalog: models, platform: "darwin" };
  const fs = { stat: async () => { throw Error("missing"); }, readdir: async () => [] };
  const missing = await planRoutes({ ...paths, fs }); const calls = [];
  await stageWanSnapshots({}, cfg, { routes: [...missing, ...missing] }, null, (args) => calls.push(args));
  assert.equal(calls.length, 3); // q4 and the two required macOS catalog co-requisite files
  for (const args of calls) {
    assert.equal(args[0], "download"); assert.match(args[3], /^[a-f0-9]{40}$/);
    assert.equal(args[4], "--include"); assert.equal(args.at(-1), cfg.hubs[0]);
  }
  const model = models.find((m) => m.id === route.model);
  const drifted = { ...model, downloads: model.downloads.map((d) => ({ ...d, revision: "b".repeat(40) })) };
  const incompatible = await planRoutes({ ...paths, baseCatalog: [drifted], fs });
  await assert.rejects(stageWanSnapshots({}, cfg, { routes: incompatible }, null, () => assert.fail()), /differ between base and head/);
});

test("CUDA Wan staging fills only absent literal q4 members from the selected Windows hub and rechecks both routes", async () => {
  const routes = Object.values(ROUTES).filter((route) => route.wanA14b && route.backend === "cuda");
  const hubs = [String.raw`E:\local models\hub`];
  const input = { profile: "tier2-ab", backend: "candle", ref: BASE, download_missing: true };
  const options = { base_sha: BASE, head_sha: HEAD, routes: routes.map((r) => r.id), hubs, preflight_only: false };
  const select = (changes = {}) => dispatchOptions({ ...input, moss_options_json: JSON.stringify({ ...options, ...changes }) }, BASE);
  assert.equal(select().hubs, JSON.stringify(hubs));
  assert.equal(select().backend, "cuda");
  for (const bad of [{ preflight_only: true }, { routes: undefined }, { routes: DEFAULT_ROUTE_IDS.cuda },
    { routes: ["cuda-wan22-a14b-i2v-heun"] }, { hubs: ["/posix/hub"] }]) assert.throws(() => select(bad));

  const cfg = { backend: "cuda", routes, hubs, baseDir: root, headDir: root };
  const catalog = await readCatalog(root), evidence = await readPatternEvidence(root);
  const absent = new Set(["q4/.gitattributes", "q4/transformer_2/model.safetensors"]);
  const present = new Set();
  const fileKey = (repo, file) => `${repo}/${file}`;
  const fs = { stat: async (file) => {
    const normalized = file.replaceAll("\\", "/");
    const [, repo, member] = normalized.match(/models--SceneWorks--([^/]+)\/snapshots\/[a-f0-9]{40}\/(q4\/.+)$/) ?? [];
    assert.ok(member, file);
    if (absent.has(member) && !present.has(fileKey(repo, member))) throw Object.assign(Error("missing"), { code: "ENOENT" });
    return { isFile: () => true, size: 42 };
  } };
  fs.lstat = fs.stat;
  const plan = () => planRoutes({ routes, hubs, baseCatalog: catalog, headCatalog: catalog, platform: "win32", fs, evidence });
  const before = await plan();
  assert.ok(before.every((row) => row.status === "skipped"));
  const calls = [];
  await stageWanSnapshots({}, cfg, { routes: before }, null, (args) => {
    const [command, repo, ...rest] = args;
    const row = before.find((route) => route.rows.required[0].repo === repo).rows.required[0];
    assert.deepEqual([command, ...rest], ["download", ...absent, "--revision", row.revision, "--cache-dir", hubs[0]]);
    for (const file of absent) present.add(fileKey(repo.split("/")[1], file));
    calls.push(repo);
  }, fs);
  assert.equal(new Set(calls).size, 2);
  const after = await plan();
  assert.ok(after.every((row) => row.status === "ready" && row.hub === hubs[0]));
  await stageWanSnapshots({}, cfg, { routes: after }, null, () => assert.fail("already complete"), fs);
  present.clear();
  await assert.rejects(stageWanSnapshots({}, cfg, { routes: before }, null, () => assert.fail(),
    { ...fs, lstat: async () => ({ isSymbolicLink: () => true }) }), /refusing to overwrite/);
  await assert.rejects(stageWanSnapshots({}, { ...cfg, baseDir: path.join(root, "absent") }, { routes: before }, null,
    () => assert.fail(), fs), /missing or divergent pinned q4 file listings/);
});

const run = (sha, status = "completed") => ({ status, output: { content: { sha256: sha } }, error: status === "completed" ? undefined : "boom" });
test("verdict: all four DEC-7 outcomes plus error, with the bounded stop rule", () => {
  assert.deepEqual(verdictFor({ base1: run("a"), head: run("a") }), { verdict: "equal", needBase2: false, reason: "head decoded content equals base#1" });
  const pending = verdictFor({ base1: run("a"), head: run("b") });
  assert.equal(pending.verdict, null); assert.equal(pending.needBase2, true);
  assert.equal(verdictFor({ base1: run("a"), head: run("b"), base2: run("c") }).verdict, "nondeterministic_on_runner");
  assert.equal(verdictFor({ base1: run("a"), head: run("b"), base2: run("a") }).verdict, "head_differs");
  assert.equal(verdictFor({ base1: run("a"), head: run("b", "error") }).verdict, "error");
  assert.equal(verdictFor({ base1: run("a", "error"), head: run("a") }).verdict, "error");
  assert.equal(verdictFor({ base1: run("a"), head: run("b"), base2: run("a", "error") }).verdict, "error");
  // equal never asks for base#2 even when base#2 would differ
  assert.equal(verdictFor({ base1: run("a"), head: run("a"), base2: run("z") }).verdict, "equal");
});

test("exit policy: head_differs, error, hub mutation and zero ran fail; nondeterminism and skips do not", () => {
  const report = (routes, mode = "run") => ({ mode, routes, summary: summarize(routes) });
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "equal" }, { status: "skipped", skipReason: "absent" }])), 0);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "nondeterministic_on_runner" }])), 0);
  const required = { id: "mlx-wan22-a14b-i2v-heun", status: "ran" };
  const ordinary = { id: "mlx-moss-image-t2i", status: "ran", verdict: "equal" };
  const firstDifference = verdictFor({ base1: run("a"), head: run("b") });
  assert.equal(firstDifference.needBase2, true);
  const diagnosed = verdictFor({ base1: run("a"), head: run("b"), base2: run("c") });
  assert.equal(diagnosed.needBase2, false); assert.equal(diagnosed.verdict, "nondeterministic_on_runner");
  assert.equal(exitCodeFor(report([ordinary, { ...required, verdict: diagnosed.verdict }])), 1);
  assert.equal(exitCodeFor(report([ordinary, { ...required, status: "not_run" }])), 1);
  assert.equal(exitCodeFor(report([ordinary, { ...required, verdict: "equal" }])), 0);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "head_differs" }])), 1);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "error" }])), 1);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "equal", hubMutation: { changed: true } }])), 1, "unattributed mutation fails");
  const runs = (baseChanged, headChanged) => [{ label: "base#1", side: "base", hubMutation: { changed: baseChanged } }, { label: "head", side: "head", hubMutation: { changed: headChanged } }];
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "equal", hubMutation: { changed: true }, runs: runs(true, false) }])), 0, "base-only mutation is a note, not a failure");
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "equal", hubMutation: { changed: true }, runs: runs(false, true) }])), 1, "head mutation fails");
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "equal", hubMutation: { changed: true }, runs: runs(true, true) }])), 1);
  assert.deepEqual([runs(true, false), runs(false, true), runs(false, false)].map((list) => hubMutationAttribution({ hubMutation: { changed: true }, runs: list })), ["base_only", "head", "unattributed"]);
  assert.equal(hubMutationAttribution({ hubMutation: { changed: false } }), "none");
  assert.equal(exitCodeFor(report([{ status: "skipped" }, { status: "skipped" }])), 1, "zero routes ran");
  assert.equal(exitCodeFor(report([{ status: "ready" }], "preflight")), 0);
  assert.equal(exitCodeFor(report([{ status: "skipped" }], "preflight")), 1);
  assert.equal(exitCodeFor({ ...report([{ status: "ran", verdict: "equal" }]), fatal: "x" }), 1);
});

test("trusted-ref rule: only commits reachable from main, release/* or feature/* of this repository", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-refs-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const git = (...args) => { const result = spawnSync("git", ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", ...args], { cwd: dir, encoding: "utf8" }); assert.equal(result.status, 0, result.stderr); return result.stdout.trim(); };
  git("init", "-q"); await writeFile(path.join(dir, "f"), "1"); git("add", "f"); git("commit", "-qm", "a"); const onMain = git("rev-parse", "HEAD");
  await writeFile(path.join(dir, "f"), "2"); git("commit", "-qam", "b"); const mainTip = git("rev-parse", "HEAD");
  git("checkout", "-q", "-b", "work"); await writeFile(path.join(dir, "f"), "3"); git("commit", "-qam", "c"); const featureTip = git("rev-parse", "HEAD");
  await writeFile(path.join(dir, "f"), "4"); git("commit", "-qam", "d"); const storyOnly = git("rev-parse", "HEAD");
  git("checkout", "-q", "--orphan", "loose"); await writeFile(path.join(dir, "f"), "5"); git("add", "f"); git("commit", "-qm", "e"); const unreferenced = git("rev-parse", "HEAD");
  git("update-ref", "refs/remotes/origin/main", mainTip); git("update-ref", "refs/remotes/origin/feature/sw-f1-x", featureTip);
  git("update-ref", "refs/remotes/origin/story/sw-1-x", storyOnly); git("update-ref", "refs/remotes/origin/mainline", storyOnly);
  git("checkout", "-q", "--detach", mainTip); git("branch", "-q", "-D", "loose"); git("branch", "-q", "-D", "work");
  assert.equal(trustedRefFor(onMain, { cwd: dir }), "refs/remotes/origin/main", "an ancestor of main is admitted");
  assert.equal(trustedRefFor(mainTip, { cwd: dir }), "refs/remotes/origin/main");
  assert.equal(trustedRefFor(featureTip, { cwd: dir }), "refs/remotes/origin/feature/sw-f1-x", "the tip of a feature branch is admitted");
  assert.equal(trustedRefFor(storyOnly, { cwd: dir }), null, "story/* and look-alike names are not trusted lines");
  assert.equal(trustedRefFor(unreferenced, { cwd: dir }), null, "a commit on no branch is refused");
  assert.equal(trustedRefFor("0".repeat(40), { cwd: dir }), null, "an absent commit is refused");
  assert.deepEqual(assertTrustedShas({ base_sha: onMain, head_sha: featureTip }, { cwd: dir }), { base_sha: { sha: onMain, trustedBy: "origin/main" }, head_sha: { sha: featureTip, trustedBy: "origin/feature/sw-f1-x" } });
  assert.throws(() => assertTrustedShas({ base_sha: onMain, head_sha: storyOnly }, { cwd: dir }), new RegExp(`head_sha ${storyOnly} is not reachable from origin/main, origin/release/\\* or origin/feature/\\*`));
  assert.deepEqual([...TRUSTED_REF_PATTERNS], ["refs/remotes/origin/main", "refs/remotes/origin/release/", "refs/remotes/origin/feature/"]);
});

test("no credential reaches the compared builds or their processes", async () => {
  const host = { PATH: "/usr/bin", HOME: "/h", HF_TOKEN: "hf_x", HUGGING_FACE_HUB_TOKEN: "hf_y", HF_HUB_TOKEN: "hf_z", GITHUB_TOKEN: "ghs", AWS_SECRET_ACCESS_KEY: "k", OPENAI_API_KEY: "k", SIGNING_KEY: "k", NPM_PASSWORD: "p", GIT_CREDENTIALS: "c", CUDA_VISIBLE_DEVICES: "1", KEYCHAIN_PATH: "/k" };
  const clean = scrubSecrets(host);
  assert.deepEqual(Object.keys(clean).sort(), ["CUDA_VISIBLE_DEVICES", "HOME", "KEYCHAIN_PATH", "PATH"]);
  // The ffmpeg preflight validated reaches API and worker; other ambient SCENEWORKS_* values do not.
  const withFfmpeg = childEnvironments({ ...host, SCENEWORKS_FFMPEG: "C:\\tools\\ffmpeg.exe", SCENEWORKS_FFMPEG_TIMEOUT_SECONDS: "900", SCENEWORKS_PULID_WEIGHTS: "D:\\pulid", SCENEWORKS_INFERENCE_READ_TOKEN: "t" }, { backend: "cuda", state: "/s", hub: "/h/hub", xet: "/x", workerId: "w", metallib: null });
  for (const child of [withFfmpeg.api, withFfmpeg.worker("http://127.0.0.1:1")]) {
    assert.equal(child.SCENEWORKS_FFMPEG, "C:\\tools\\ffmpeg.exe"); assert.equal(child.SCENEWORKS_FFMPEG_TIMEOUT_SECONDS, "900");
    assert.equal(child.SCENEWORKS_PULID_WEIGHTS, undefined); assert.equal(child.SCENEWORKS_INFERENCE_READ_TOKEN, undefined);
  }
  assert.equal(childEnvironments(host, { backend: "mlx", state: "/s", hub: "/h/hub", xet: "/x", workerId: "w", metallib: null }).api.SCENEWORKS_FFMPEG, undefined, "unset stays unset: the children find ffmpeg on PATH like preflight did");
  const envs = childEnvironments(host, { backend: "cuda", state: "/s", hub: "/h/hub", xet: "/x", workerId: "w", metallib: null });
  const secretKeys = Object.keys(host).filter((key) => !(key in clean)), secretValues = secretKeys.map((key) => host[key]);
  for (const child of [envs.api, envs.worker("http://127.0.0.1:1")]) {
    for (const key of secretKeys) assert.equal(child[key], undefined, key);
    for (const value of Object.values(child)) assert.ok(!secretValues.includes(value), "no secret value is passed under another name");
    // The only credential-named key is the app's own (empty) credential store directory path.
    assert.deepEqual(Object.keys(child).filter((key) => /TOKEN|SECRET|PASSWORD|CREDENTIAL|_KEY$/i.test(key)), ["SCENEWORKS_CREDENTIALS_DIR"]);
  }
  const source = await readFile(path.join(root, "scripts", "tier2-ab.mjs"), "utf8");
  assert.match(source, /let cargoEnv = \{ \.\.\.scrubSecrets\(env\), CARGO_TARGET_DIR: target \}/);
  assert.match(source, /if \(label === "base" && report\.base\.kreaAppleRawProjector\) \{[^}]*cargoEnv = await appleRawBaselineCargoEnv\(cargoEnv\)/);
  assert.match(source, /execFile\("cargo", \["metadata", "--locked"[^\n]*\n[^\n]*env: cargoEnv/, "Cargo metadata gets the scrubbed environment");
  assert.match(source, /runLogged\("cargo", args, \{ cwd: dir, env: cargoEnv, log \}\)/, "Cargo build gets the scrubbed environment");
  assert.doesNotMatch(source, /base\.HF_TOKEN\s*=/);
  // A gated repository needs a credential for head's pinned-metadata admission, so it is skipped.
  const models = await readCatalog(root);
  const planned = await planRoutes({ routes: [ROUTES["mlx-flux3-klein-t2i"]], hubs: ["/nonexistent-tier2-hub"], baseCatalog: models, headCatalog: models, platform: "darwin" });
  assert.equal(planned[0].status, "skipped"); assert.match(planned[0].skipReason, /gated repository/);
});

test("win32 launch: API and worker hold a stdin pipe and the worker gets the candle GPU1 environment", () => {
  // The real worker shuts down on stdin EOF when SCENEWORKS_WORKER_CHILD=1 (Windows); stdin "ignore"
  // is immediate EOF from NUL, which is what took every worker offline in run 43427303870.
  assert.deepEqual(childStdio("worker", 7), ["pipe", 7, 7]); assert.deepEqual(childStdio("api", 7), ["pipe", 7, 7]); assert.deepEqual(childStdio("sentinel", 7), ["pipe", 7, 7]);
  const host = { Path: "C:\\Windows", CUDA_VISIBLE_DEVICES: "1", NVIDIA_VISIBLE_DEVICES: "1", CUDA_DEVICE_ORDER: "PCI_BUS_ID", SCENEWORKS_GPU_ID: "1", SCENEWORKS_STALE: "x", HF_TOKEN: "hf_secret_token", GITHUB_TOKEN: "ghs_secret" };
  const envs = childEnvironments(host, { backend: "cuda", state: "D:\\t\\state", hub: "E:\\huggingface\\hub", xet: "D:\\t\\xet", workerId: "w1", metallib: null });
  const worker = envs.worker("http://127.0.0.1:5555"), api = envs.api;
  for (const [key, value] of Object.entries({ SCENEWORKS_WORKER_CHILD: "1", SCENEWORKS_WORKER_ONLY: "1", SCENEWORKS_GPU_ID: "1", SCENEWORKS_BACKEND_CANDLE_ENABLED: "true", SCENEWORKS_BACKEND_MLX_ENABLED: "false", SCENEWORKS_RUN_UTILITY_INPROCESS: "0", SCENEWORKS_WORKER_ID: "w1", SCENEWORKS_API_URL: "http://127.0.0.1:5555", SCENEWORKS_API_PORT: "5555", CUDA_VISIBLE_DEVICES: "1", NVIDIA_VISIBLE_DEVICES: "1", CUDA_DEVICE_ORDER: "PCI_BUS_ID", HF_HUB_OFFLINE: "1", HF_HUB_CACHE: "E:\\huggingface\\hub" })) assert.equal(worker[key], value, key);
  for (const key of ["SCENEWORKS_MLX_REQUIRED", "SCENEWORKS_MLX_UNSUPPORTED_MODE", "SCENEWORKS_STALE", "SCENEWORKS_RUST_WORKER_GPU_ID", "GITHUB_TOKEN", "HF_TOKEN"]) assert.equal(worker[key], undefined, key);
  for (const [key, value] of Object.entries({ SCENEWORKS_RUN_UTILITY_INPROCESS: "1", SCENEWORKS_RUST_WORKER_GPU_ID: "cpu", SCENEWORKS_WORKER_ID: "w1-utility", SCENEWORKS_API_PORT: "0", SCENEWORKS_WORKER_CHILD: "1" })) assert.equal(api[key], value, key);
  assert.equal(api.SCENEWORKS_API_URL, undefined);
  assert.deepEqual(processState({ label: "worker", pid: 9, exitCode: 3, signalCode: null }), { pid: 9, alive: false, exitCode: 3, signal: null, stdin: "pipe" });
});

test("paired Krea Hires pins the Hugging Face origin for both Windows child processes", () => {
  const host = {
    SCENEWORKS_HUGGINGFACE_BASE_URL: "https://alternate.invalid",
    Sceneworks_HuggingFace_Base_Url: "https://alternate.invalid",
    sceneworks_huggingface_base_url: "https://alternate.invalid",
    HF_ENDPOINT: "https://alternate.invalid",
    Hf_Endpoint: "https://alternate.invalid",
    hf_endpoint: "https://alternate.invalid",
  };
  const options = { backend: "cuda", state: "/s", hub: "/h/hub", xet: "/x", workerId: "w", metallib: null };
  const envs = childEnvironments(host, { ...options, pairedKreaHires: true });
  for (const child of [envs.api, envs.worker("http://127.0.0.1:1")]) {
    const origins = Object.entries(child).filter(([key]) =>
      ["SCENEWORKS_HUGGINGFACE_BASE_URL", "HF_ENDPOINT"].includes(key.toUpperCase()));
    assert.deepEqual(origins, [["SCENEWORKS_HUGGINGFACE_BASE_URL", "https://huggingface.co"]]);
  }
  assert.equal(childEnvironments(host, options).api.SCENEWORKS_HUGGINGFACE_BASE_URL, undefined);
});

test("job polling: claim bound before a worker attaches, execution bound after, terminal states stop at once", () => {
  const base = { postedMs: 0, claimMinutes: 5, timeoutMinutes: 20 };
  assert.equal(jobProgressDecision({ ...base, job: { status: "queued", stage: "queued" }, nowMs: 4 * 60_000 }).state, "waiting_claim");
  const unclaimed = jobProgressDecision({ ...base, job: { status: "queued", stage: "queued", message: "no eligible worker" }, nowMs: 5 * 60_000 + 1 });
  assert.equal(unclaimed.state, "stop"); assert.match(unclaimed.reason, /^unclaimed: job stayed queued\/queued \(no eligible worker\) for more than 5 min/);
  assert.equal(jobProgressDecision({ ...base, job: { status: "pending_workflow", stage: "pending_workflow" }, nowMs: 6 * 60_000 }).state, "stop");
  assert.equal(jobProgressDecision({ ...base, job: { status: "queued", workerId: "w" }, nowMs: 6 * 60_000, claimedAtMs: 5.5 * 60_000 }).state, "running", "a worker id is a claim");
  const running = jobProgressDecision({ ...base, job: { status: "running", stage: "generating" }, nowMs: 30 * 60_000, claimedAtMs: 10 * 60_000 });
  assert.equal(running.state, "running");
  const over = jobProgressDecision({ ...base, job: { status: "running", stage: "generating" }, nowMs: 30 * 60_000, claimedAtMs: 10 * 60_000 - 1 });
  assert.equal(over.state, "stop"); assert.match(over.reason, /^execution bound: job still running\/generating 20 min after the claim/);
  assert.deepEqual(jobProgressDecision({ ...base, job: { status: "failed", error: "model_artifact_incomplete" }, nowMs: 1 }), { state: "stop", reason: "job failed: model_artifact_incomplete" });
  assert.equal(jobProgressDecision({ ...base, job: { status: "completed" }, nowMs: 99 * 60_000 }).state, "completed");
  assert.equal(jobClaimed({ status: "queued", workerId: null }), false); assert.equal(jobClaimed({ status: "running" }), true);
  for (const id of Object.keys(ROUTES)) { const route = ROUTES[id]; assert.ok(route.claimMinutes > 0 && route.claimMinutes <= 10 && route.timeoutMinutes <= (route.kind === "video" ? 45 : 20), `${id} bounds fit the route`); }
  assert.equal(resolveConfig(env({ TIER2_AB_CLAIM_MINUTES: "2" }), { platform: "darwin" }).claimMinutesOverride, 2);
  for (const bad of ["0", "-1", "61", "x"]) assert.throws(() => resolveConfig(env({ TIER2_AB_CLAIM_MINUTES: bad }), { platform: "darwin" }), /TIER2_AB_CLAIM_MINUTES/);
});

test("diagnostics keep the served job state and refusal fields; progress lines are one greppable line", () => {
  const job = { id: "j", status: "queued", stage: "queued", message: "waiting", workerId: null, prompt: "x", payload: { model: "m", admissionRefusal: "no claim" }, result: { blockedReason: "b" } };
  const diag = jobDiagnostics(job);
  assert.equal(diag.status, "queued"); assert.equal(diag.prompt, undefined);
  assert.deepEqual(diag.reasons, { "payload.admissionRefusal": "no claim", "result.blockedReason": "b" });
  assert.equal(jobDiagnostics(null), null);
  assert.deepEqual(lastLines(Array.from({ length: 80 }, (_, i) => `l${i}`).join("\r\n") + "\n", 50), Array.from({ length: 50 }, (_, i) => `l${i + 30}`));
  const line = progressLine({ route: "r", run: "head", phase: "claimed", worker: "owned", empty: "", reason: "a\nb  c" }, new Date("2026-11-12T00:00:00Z"));
  assert.equal(line, "[tier2-ab] 2026-11-12T00:00:00.000Z route=r run=head phase=claimed worker=owned reason=a b c");
});

test("GPU inventory joins nvidia-smi index/bus with driver display, memory and typed processes", () => {
  const xml = `<nvidia_smi_log><gpu id="00000000:21:00.0"><product_name>RTX</product_name><display_attached>No</display_attached><display_active>Disabled</display_active><driver_model><current_dm>WDDM</current_dm></driver_model><uuid>GPU-b1</uuid><fb_memory_usage><total>97887 MiB</total><used>50220 MiB</used><free>47219 MiB</free></fb_memory_usage><utilization><gpu_util>99 %</gpu_util></utilization><processes><process_info><pid>40052</pid><type>C</type><process_name>C:\\Apps\\sceneworks-api.exe</process_name><used_memory>N/A</used_memory></process_info></processes></gpu><gpu id="00000000:C1:00.0"><product_name>RTX</product_name><display_attached>Yes</display_attached><display_active>Disabled</display_active><driver_model><current_dm>WDDM</current_dm></driver_model><uuid>GPU-e4</uuid><fb_memory_usage><total>97887 MiB</total><used>5847 MiB</used><free>92040 MiB</free></fb_memory_usage><utilization><gpu_util>12 %</gpu_util></utilization><processes><process_info><pid>768</pid><type>C+G</type><process_name>C:\\x\\explorer.exe</process_name><used_memory>N/A</used_memory></process_info><process_info><pid>40068</pid><type>C</type><process_name>C:\\Apps\\sceneworks-api.exe</process_name><used_memory>N/A</used_memory></process_info></processes></gpu></nvidia_smi_log>`;
  const gpus = gpuInventory("0, GPU-b1, 00000000:21:00.0, RTX\n1, GPU-e4, 00000000:C1:00.0, RTX\n", xml);
  assert.deepEqual(gpus.map((gpu) => [gpu.index, gpu.uuid, gpu.pciBusId, gpu.displayAttached, gpu.memoryUsed]), [[0, "GPU-b1", "00000000:21:00.0", "No", "50220 MiB"], [1, "GPU-e4", "00000000:C1:00.0", "Yes", "5847 MiB"]]);
  assert.deepEqual(gpus[1].processes.map((row) => `${row.type}:${row.name}:${row.usedMemory}`), ["C+G:explorer.exe:N/A", "C:sceneworks-api.exe:N/A"]);
  const markdown = renderMarkdown({ backend: "cuda", mode: "preflight", status: "finished", exitCode: 0, base: { sha: BASE }, head: { sha: HEAD }, runner: { platform: "win32", arch: "x64" }, startedAt: "s", memoryMethod: "m", routes: [], gpus, gpuBinding: { policy: "shared-display" } });
  assert.match(markdown, /physical GPU1 is the host's display GPU/); assert.match(markdown, /\| 1 \| 00000000:C1:00\.0 \| GPU-e4 \| Yes \| 5847 MiB \/ 97887 MiB \| 12 % \| C 1 \(sceneworks-api\.exe:40068\), G 0, C\+G 1 \|/);
});

test("scratch cleanup refuses roots outside scratch or overlapping a hub, and never changes a multiply linked Windows file", async () => {
  const allowedRoots = ["/runner/_temp", "/runner/_work/repo"];
  await assert.rejects(removeScratch("/runner/_temp", { allowedRoots }), /not strictly inside/);
  await assert.rejects(removeScratch("/elsewhere/x", { allowedRoots }), /not strictly inside/);
  await assert.rejects(removeScratch("/runner/_temp/t2ab", { allowedRoots, protectedRoots: ["/runner/_temp/t2ab/hub"] }), /overlaps the model hub/);
  await assert.rejects(removeScratch("/runner/_temp/t2ab/hub/x", { allowedRoots, protectedRoots: ["/runner/_temp/t2ab/hub"] }), /overlaps the model hub/);
  const entries = { "C:\\t\\w": { dir: true, mode: 0o40555, children: ["a", "b", "l"] }, "C:\\t\\w\\a": { mode: 0o100444, nlink: 1 }, "C:\\t\\w\\b": { mode: 0o100444, nlink: 2 }, "C:\\t\\w\\l": { link: true, mode: 0o120777 } };
  const changed = [], removed = [];
  const fakeFs = { lstat: async (p) => { const e = entries[p]; if (!e) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" }); return { mode: e.mode, nlink: e.nlink ?? 1, isDirectory: () => Boolean(e.dir), isSymbolicLink: () => Boolean(e.link) }; }, readdir: async (p) => entries[p].children, chmod: async (p, mode) => changed.push([p, mode]), rm: async (p) => removed.push(p) };
  await assert.rejects(removeScratch("C:\\t\\w", { allowedRoots: ["C:\\t"], platform: "win32", fs: fakeFs }), /multiply linked read-only/);
  assert.deepEqual(removed, [], "nothing removed after a refusal");
  assert.ok(!changed.some(([p]) => p.endsWith("\\b") || p.endsWith("\\l")), "neither the linked file nor the symlink is modified");
  entries["C:\\t\\w\\b"].nlink = 1; changed.length = 0;
  const result = await removeScratch("C:\\t\\w", { allowedRoots: ["C:\\t"], platform: "win32", fs: fakeFs });
  assert.deepEqual({ dirs: result.restoredDirectories, files: result.restoredFiles, removed }, { dirs: 1, files: 2, removed: ["C:\\t\\w"] });
  assert.ok(!changed.some(([p]) => p.endsWith("\\l")));
  assert.equal((await removeScratch("C:\\t\\missing", { allowedRoots: ["C:\\t"], platform: "win32", fs: fakeFs })).absent, true);
});

// --- synthetic PNG encoder with every filter type and arbitrary extra chunks ---
const crc32 = (bytes) => { let c = 0xffffffff; for (const v of bytes) { c ^= v; for (let i = 0; i < 8; i++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0); } return (c ^ 0xffffffff) >>> 0; };
const chunk = (kind, data) => { const type = Buffer.from(kind), out = Buffer.alloc(data.length + 12); out.writeUInt32BE(data.length); type.copy(out, 4); data.copy(out, 8); out.writeUInt32BE(crc32(Buffer.concat([type, data])), out.length - 4); return out; };
function png(pixels, width, height, channels, { filters = [0], extra = [] } = {}) {
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = channels === 4 ? 6 : 2;
  const stride = width * channels, rows = [];
  for (let y = 0; y < height; y++) {
    const filter = filters[y % filters.length], row = pixels.subarray(y * stride, (y + 1) * stride), prior = y ? pixels.subarray((y - 1) * stride, y * stride) : Buffer.alloc(stride), out = Buffer.alloc(stride + 1);
    out[0] = filter;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? row[x - channels] : 0, b = prior[x], c = x >= channels ? prior[x - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const predictor = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
      out[x + 1] = (row[x] - predictor) & 0xff;
    }
    rows.push(out);
  }
  return Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"), chunk("IHDR", header), ...extra, chunk("IDAT", deflateSync(Buffer.concat(rows))), chunk("IEND", Buffer.alloc(0))]);
}
const itxt = (text) => chunk("iTXt", Buffer.concat([Buffer.from("sceneworks:workflow\0\0\0\0\0"), Buffer.from(text)]));
const pixels = (w, h, channels, seed = 7) => { const out = Buffer.alloc(w * h * channels); let s = seed; for (let i = 0; i < out.length; i++) { s = (s * 1103515245 + 12345) >>> 0; out[i] = s >>> 24; } return out; };

test("decoded PNG content ignores metadata chunks and filter choice; file hash does not", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-png-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const rgb = pixels(9, 5, 3);
  const a = png(rgb, 9, 5, 3, { extra: [itxt('{"jobId":"job-1","createdAt":"2026-11-12T00:00:00Z"}')] });
  const b = png(rgb, 9, 5, 3, { filters: [1, 2, 3, 4, 0], extra: [itxt('{"jobId":"job-2","createdAt":"2026-11-12T09:09:09Z"}'), chunk("tEXt", Buffer.from("Comment\0other"))] });
  const changed = Buffer.from(rgb); changed[10] ^= 1;
  const c = png(changed, 9, 5, 3);
  for (const [name, bytes] of [["a.png", a], ["b.png", b], ["c.png", c]]) await writeFile(path.join(dir, name), bytes);
  const [da, db, dc] = await Promise.all(["a.png", "b.png", "c.png"].map((name) => decodedContent(path.join(dir, name), "image")));
  assert.notEqual(a.toString("hex"), b.toString("hex"));
  assert.equal(da.method, "png_rgba8"); assert.deepEqual(da.shape, { width: 9, height: 5 });
  assert.equal(da.sha256, db.sha256, "same pixels must give the same decoded hash");
  assert.notEqual(da.sha256, dc.sha256, "one changed channel must change the decoded hash");
  const decoded = decodePng(a);
  assert.equal(decoded.rgba.length, 9 * 5 * 4); assert.equal(decoded.rgba[3], 255); assert.deepEqual([...decoded.rgba.subarray(0, 3)], [...rgb.subarray(0, 3)]);
  const rgba = pixels(4, 3, 4, 3);
  assert.deepEqual(decodePng(png(rgba, 4, 3, 4, { filters: [4, 3, 1] })).rgba, rgba);
  assert.deepEqual(numericDifference(da.buffer, dc.buffer).changedElements, 1);
  assert.equal(numericDifference(da.buffer, db.buffer).maxAbsoluteDifference, 0);
  assert.equal(numericDifference(da.buffer, Buffer.alloc(3)).comparable, false);
  assert.throws(() => decodePng(Buffer.from("not a png")), /not a PNG/);
});

test("WAV fidelity hashes PCM samples, not header or trailing chunks", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-wav-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const wav = referenceWav(), parsed = parseWav(wav);
  assert.equal(parsed.sampleRate, 24_000); assert.equal(parsed.channels, 1); assert.equal(parsed.data.length, 96_000);
  const list = Buffer.concat([Buffer.from("LIST"), Buffer.from([4, 0, 0, 0]), Buffer.from("INFO")]);
  const tagged = Buffer.concat([wav, list]); tagged.writeUInt32LE(tagged.length - 8, 4);
  await writeFile(path.join(dir, "a.wav"), wav); await writeFile(path.join(dir, "b.wav"), tagged);
  const [a, b] = await Promise.all([decodedContent(path.join(dir, "a.wav"), "audio"), decodedContent(path.join(dir, "b.wav"), "audio")]);
  assert.equal(a.method, "wav_pcm_samples"); assert.equal(a.sha256, b.sha256);
  assert.notEqual(contentDigest({ x: 1 }, Buffer.from([1])), contentDigest({ x: 2 }, Buffer.from([1])));
});

test("video fidelity digest covers stream geometry and timing, not only decoded bytes", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-video-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, "clip.mp4"); await writeFile(file, "container");
  const frames = Buffer.alloc(768 * 512 * 4 * 3, 7), pcm = Buffer.alloc(4800, 3);
  const decode = async (_ffmpeg, _file, kind) => ({ ok: true, data: kind === "video-frames" ? frames : pcm });
  const probe = (video, audio = { codec_type: "audio", sample_rate: "48000", channels: 2, channel_layout: "stereo", time_base: "1/48000", duration: "0.125" }) => async () => ({ ok: true, data: { streams: [{ codec_type: "video", ...video }, audio], format: { duration: "0.125" } } });
  const base = { width: 768, height: 512, nb_frames: "3", r_frame_rate: "24/1", avg_frame_rate: "24/1", time_base: "1/12288", duration: "0.125" };
  const digest = async (video, audio) => (await decodedContent(file, "video", { decode, probe: probe(video, audio) }));
  const reference = await digest(base);
  assert.equal(reference.method, "ffmpeg_rgba_frames+s16le_audio+ffprobe_streams");
  assert.deepEqual(reference.shape.video, { width: 768, height: 512, decodedFrames: 3, nbFrames: "3", rFrameRate: "24/1", avgFrameRate: "24/1", timeBase: "1/12288", duration: "0.125" });
  assert.deepEqual(reference.shape.audio, { sampleRate: 48000, channels: 2, channelLayout: "stereo", timeBase: "1/48000", duration: "0.125" });
  assert.equal((await digest(base)).sha256, reference.sha256, "same stream and bytes stay equal");
  assert.notEqual((await digest({ ...base, r_frame_rate: "25/1", avg_frame_rate: "25/1" })).sha256, reference.sha256, "equal bytes at another frame rate differ");
  assert.notEqual((await digest({ ...base, width: 512, height: 768 })).sha256, reference.sha256, "equal bytes at another geometry of the same area differ");
  assert.notEqual((await digest(base, { codec_type: "audio", sample_rate: "44100", channels: 2, channel_layout: "stereo" })).sha256, reference.sha256, "another audio sample rate differs");
  // An audio stream that fails to decode never digests as silence: container bytes and the reason.
  const failingAudio = async (_ffmpeg, _file, kind) => (kind === "audio" ? { ok: false, error: "unsupported codec" } : { ok: true, data: frames });
  const brokenAudio = await decodedContent(file, "video", { decode: failingAudio, probe: probe(base) });
  assert.equal(brokenAudio.method, "file_bytes_stream_not_decoded"); assert.match(brokenAudio.decoderError, /^audio stream present but decode failed: unsupported codec/);
  assert.equal(brokenAudio.sha256, sha256(Buffer.from("container"))); assert.equal(brokenAudio.shape.audio.sampleRate, 48000);
  // No audio stream (per ffprobe) is a comparable shape, and is never confused with a decode failure.
  const silentProbe = async () => ({ ok: true, data: { streams: [{ codec_type: "video", ...base }], format: { duration: "0.125" } } });
  const silent = await decodedContent(file, "video", { decode: failingAudio, probe: silentProbe });
  assert.equal(silent.method, "ffmpeg_rgba_frames+s16le_audio+ffprobe_streams"); assert.equal(silent.shape.audio, null); assert.equal(silent.shape.decoded.audioBytes, null);
  assert.notEqual(silent.sha256, reference.sha256);
  // A video stream that fails to decode, a missing video stream, or bytes that are not whole frames fall back too.
  const brokenVideo = await decodedContent(file, "video", { decode: async (_f, _x, kind) => (kind === "video-frames" ? { ok: false, error: "corrupt" } : { ok: true, data: pcm }), probe: probe(base) });
  assert.equal(brokenVideo.method, "file_bytes_stream_not_decoded"); assert.match(brokenVideo.decoderError, /^video stream present but decode failed: corrupt/);
  assert.match((await decodedContent(file, "video", { decode, probe: async () => ({ ok: true, data: { streams: [], format: {} } }) })).decoderError, /no video stream/);
  assert.match((await decodedContent(file, "video", { decode: async (_f, _x, kind) => ({ ok: true, data: kind === "video-frames" ? Buffer.alloc(10) : pcm }), probe: probe(base) })).decoderError, /not a whole number of 768x512 RGBA frames/);
  // The non-WAV audio branch: stream absent or undecodable falls back with its reason.
  assert.match((await decodedContent(file, "audio", { decode, probe: async () => ({ ok: true, data: { streams: [] } }) })).decoderError, /no audio stream/);
  assert.match((await decodedContent(file, "audio", { decode: failingAudio, probe: probe(base) })).decoderError, /audio stream present but decode failed/);
  const noProbe = await decodedContent(file, "video", { decode, probe: async () => ({ ok: false, error: "ffprobe missing" }) });
  assert.equal(noProbe.method, "file_bytes_no_probe"); assert.equal(noProbe.sha256, sha256(Buffer.from("container")));
  assert.equal(streamShape({ streams: [] }).video, null);
  assert.equal(ffprobePath("ffmpeg"), "ffprobe"); assert.equal(ffprobePath("/opt/bin/ffmpeg"), "/opt/bin/ffprobe"); assert.equal(ffprobePath("/opt/bin/avconv"), "ffprobe");
});

test("wildcard presence: config-only, dangling or empty sets are not complete; the next hub is chosen", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-partial-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const rev = "d".repeat(40), repo = "O/w";
  const snap = (hub) => path.join(hub, "models--O--w", "snapshots", rev);
  const model = { id: "w_model", downloads: [{ provider: "huggingface", repo, revision: rev, variant: "q8", default: true, files: ["q8/*"] }] };
  const rows = routeDownloadRows(model, { platform: "darwin", tier: "q8" });
  const configOnly = path.join(dir, "config-only"), dangling = path.join(dir, "dangling"), empty = path.join(dir, "empty-weight"), full = path.join(dir, "full");
  await mkdir(path.join(snap(configOnly), "q8"), { recursive: true }); await writeFile(path.join(snap(configOnly), "q8", "config.json"), "{}");
  await mkdir(path.join(snap(dangling), "q8"), { recursive: true }); await writeFile(path.join(snap(dangling), "q8", "model.safetensors"), "w");
  const { symlink } = await import("node:fs/promises"); await symlink("../../../blobs/missing", path.join(snap(dangling), "q8", "shard-2.safetensors"));
  await mkdir(path.join(snap(empty), "q8"), { recursive: true }); await writeFile(path.join(snap(empty), "q8", "config.json"), "{}"); await writeFile(path.join(snap(empty), "q8", "model.safetensors"), "");
  await mkdir(path.join(snap(full), "q8", "transformer"), { recursive: true }); await writeFile(path.join(snap(full), "q8", "config.json"), "{}"); await writeFile(path.join(snap(full), "q8", "transformer", "model.safetensors"), "w");
  // Heuristic (no checked-in listing for this revision).
  const heuristic = async (hub) => routePresence(hub, rows);
  const a = await heuristic(configOnly); assert.equal(a.complete, false); assert.match(a.missing[0], /no non-empty weight file/);
  const b = await heuristic(dangling); assert.equal(b.complete, false); assert.match(b.missing[0], /shard-2\.safetensors \(dangling/);
  const c = await heuristic(empty); assert.equal(c.complete, false); assert.match(c.missing[0], /model\.safetensors \(empty file\)/);
  assert.equal((await heuristic(full)).complete, true);
  const route = { id: "r", backend: "mlx", model: "w_model", kind: "image", mode: "text_to_image", tier: "q8", group: null, endpoint: "/api/v1/image/jobs", body: { model: "w_model", seed: 1 } };
  const planned = (await planRoutes({ routes: [route], hubs: [configOnly, dangling, empty, full], baseCatalog: [model], headCatalog: [model], platform: "darwin" }))[0];
  assert.equal(planned.status, "ready"); assert.equal(planned.hub, full, "a partial hub never stops the search");
  assert.equal(planned.presence.length, 4);
  // With a checked-in listing the exact expected files decide, so a missing listed shard is caught
  // even when another weight file is present.
  const evidence = new Map([[`${repo}@${rev}`, ["README.md", "q8/config.json", "q8/transformer/model.safetensors", "q8/transformer/model-2.safetensors"]]]);
  const listed = await routePresence(full, rows, undefined, evidence);
  assert.equal(listed.complete, false); assert.equal(listed.partial, false, "a single wildcard selector is either present or not"); assert.match(listed.missing[0], /q8\/transformer\/model-2\.safetensors/);
  await writeFile(path.join(snap(full), "q8", "transformer", "model-2.safetensors"), "w");
  assert.equal((await routePresence(full, rows, undefined, evidence)).complete, true);
  assert.deepEqual(await selectorProblems(snap(full), "q8/*", { evidenceFiles: evidence.get(`${repo}@${rev}`) }), []);
  // Partial across selectors is recorded on the ready route.
  const twoRows = { id: "m2", downloads: [{ provider: "huggingface", repo, revision: rev, files: ["q8/config.json", "q8/extra.json"] }] };
  const partialRoute = { ...route, id: "r2", model: "m2", tier: null };
  const plannedPartial = (await planRoutes({ routes: [partialRoute], hubs: [configOnly, full], baseCatalog: [twoRows], headCatalog: [twoRows], platform: "darwin" }))[0];
  assert.equal(plannedPartial.status, "skipped"); assert.match(plannedPartial.skipReason, /q8\/extra\.json/);
  await writeFile(path.join(snap(full), "q8", "extra.json"), "{}");
  const readyPartial = (await planRoutes({ routes: [partialRoute], hubs: [configOnly, full], baseCatalog: [twoRows], headCatalog: [twoRows], platform: "darwin" }))[0];
  assert.equal(readyPartial.status, "ready"); assert.equal(readyPartial.hub, full);
  assert.deepEqual(readyPartial.presenceNotes, [`partial at ${configOnly} [untiered]: O/w@${rev.slice(0, 12)}/q8/extra.json`]);
});

test("the checked-in download listing expands every wildcard of the default routes to weight files", async () => {
  const evidence = await readPatternEvidence(root), models = await readCatalog(root);
  assert.ok(evidence.size > 0);
  for (const id of [...DEFAULT_ROUTE_IDS.mlx, ...DEFAULT_ROUTE_IDS.cuda]) {
    const route = ROUTES[id], model = models.find((entry) => entry.id === route.model);
    if (model.gated) continue;
    for (const tier of route.tierCandidates ?? [route.tier]) {
      const rows = routeDownloadRows(model, { platform: route.backend === "mlx" ? "darwin" : "win32", tier });
      for (const row of rows.required) for (const selector of row.files.filter((item) => item.includes("*"))) {
        const files = evidence.get(`${row.repo}@${row.revision}`);
        assert.ok(files, `${id}: ${row.repo}@${row.revision} has a checked-in listing`);
        assert.ok(expandSelector(selector, files).some((file) => file.endsWith(".safetensors")), `${id} ${tier}: ${selector} expands to weight files`);
      }
    }
  }
  assert.deepEqual(expandSelector("q4/*", ["q4/a/b.safetensors", "q8/c.safetensors", "q4x"]), ["q4/a/b.safetensors"], "`*` crosses `/`, as the product's glob does");
});

test("untiered presence follows the product's preference order and reports the tier found", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-tier-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const models = await readCatalog(root), flux = models.find((entry) => entry.id === "flux_dev"), route = ROUTES["mlx-flux1-dev-t2i"];
  assert.equal(route.tier, null); assert.deepEqual(route.tierCandidates, ["q8", "bf16", "q4"]); assert.equal(route.body.advanced.mlxQuantize, undefined);
  const install = async (hub, tier) => { for (const row of routeDownloadRows(flux, { platform: "darwin", tier }).required) for (const selector of row.files) { const target = path.join(hub, `models--${row.repo.replace("/", "--")}`, "snapshots", row.revision, ...selector.replace(/\/\*$/, "/w.bin").split("/")); await mkdir(path.dirname(target), { recursive: true }); await writeFile(target, "w"); } };
  const q8Only = path.join(dir, "q8"), q4Only = path.join(dir, "q4"), both = path.join(dir, "both"), empty = path.join(dir, "empty");
  await install(q8Only, "q8"); await install(q4Only, "q4"); await install(both, "q4"); await install(both, "q8"); await mkdir(empty);
  const plan = async (hub) => (await planRoutes({ routes: [route], hubs: [hub], baseCatalog: models, headCatalog: models, platform: "darwin" }))[0];
  const a = await plan(q8Only); assert.equal(a.status, "ready"); assert.equal(a.tier, "q8"); assert.match(a.tierSource, /untiered preference order/);
  const b = await plan(q4Only); assert.equal(b.status, "ready"); assert.equal(b.tier, "q4");
  assert.equal((await plan(both)).tier, "q8", "q8 wins when both are installed, as the worker's resolver does");
  const c = await plan(empty); assert.equal(c.status, "skipped"); assert.match(c.skipReason, /weights absent from every hub root for tier q8 or bf16 or q4/);
  assert.equal(requestTemplate(route).body.advanced.mlxQuantize, undefined, "the request stays untiered");
});

test("hub fingerprint sees a same-length blob rewrite behind a snapshot link", async (t) => {
  const hub = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-blob-")); t.after(() => rm(hub, { recursive: true, force: true }));
  const rev = "c".repeat(40), repoDir = path.join(hub, "models--O--m"), snap = path.join(repoDir, "snapshots", rev), blob = path.join(repoDir, "blobs", "abc123");
  await mkdir(snap, { recursive: true }); await mkdir(path.dirname(blob), { recursive: true }); await writeFile(blob, "weights-v1");
  const { symlink } = await import("node:fs/promises"); await symlink(path.relative(snap, blob), path.join(snap, "model.safetensors"));
  const before = await hubFingerprint(hub, [{ repo: "O/m", revision: rev }]);
  const entry = before.repos[`O/m@${rev}`].snapshot["model.safetensors#entry"];
  assert.equal(entry.link, true); assert.equal(entry.target.size, 10);
  await new Promise((resolve) => setTimeout(resolve, 20));
  await writeFile(blob, "weights-v2"); // same length, in place
  const after = await hubFingerprint(hub, [{ repo: "O/m", revision: rev }]);
  const diff = diffObjects(before.repos, after.repos);
  assert.ok(diff.count > 0, "an in-place blob rewrite is a hub change");
  assert.ok(diff.entries.every((item) => item.path.includes("model.safetensors#entry/target/")), JSON.stringify(diff.entries));
});

test("report memory cell: CUDA peak above baseline with the baseline, MLX own peak", () => {
  assert.equal(memoryCell({ peakBytes: 17.45 * 1024 ** 3, peakAboveBaselineBytes: 9.32 * 1024 ** 3, gpuBaselineBytes: 8.13 * 1024 ** 3 }), "9.32 GiB (+ baseline 8.13 GiB)");
  assert.equal(memoryCell({ peakBytes: 42 * 1024 ** 3 }), "42.00 GiB");
  assert.equal(memoryCell(undefined), "–");
});

test("catalog rows: default or requested tier, platform filter, soft rows optional, derived tiers ignored", () => {
  const model = { id: "m", downloads: [
    { provider: "huggingface", repo: "o/m", revision: "1".repeat(40), variant: "q4", default: true, files: ["q4/*"] },
    { provider: "huggingface", repo: "o/m", revision: "1".repeat(40), variant: "q8", files: ["q8/*"] },
    { provider: "huggingface", repo: "o/m", revision: "1".repeat(40), variant: "q8", files: [], localDerivation: { fromVariant: "bf16" } },
    { provider: "huggingface", repo: "o/tok", revision: "2".repeat(40), coRequisite: true, componentId: "tok", files: ["tokenizer.json"], platforms: ["windows"] },
    { provider: "huggingface", repo: "o/ctl", revision: "3".repeat(40), coRequisite: true, required: "soft", files: ["c.safetensors"] },
  ] };
  const mac = routeDownloadRows(model, { platform: "darwin", tier: null });
  assert.equal(mac.tier, "q4"); assert.deepEqual(mac.required.map((row) => row.repo), ["o/m"]); assert.deepEqual(mac.optional.map((row) => row.repo), ["o/ctl"]);
  const win = routeDownloadRows(model, { platform: "win32", tier: "q8" });
  assert.deepEqual(win.required.map((row) => `${row.repo}:${row.files.join(",")}`), ["o/m:q8/*", "o/tok:tokenizer.json"]);
  assert.throws(() => routeDownloadRows(model, { platform: "darwin", tier: "bf16" }), /no bf16/);
});

test("hub presence: metadata only, first complete hub wins, groups and catalog drift skip with reasons", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-hub-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const rev = "a".repeat(40), empty = path.join(dir, "empty"), full = path.join(dir, "full");
  const snap = path.join(full, "models--O--img", "snapshots", rev);
  await mkdir(path.join(snap, "q4"), { recursive: true }); await mkdir(empty);
  await writeFile(path.join(snap, "q4", "model.safetensors"), "x"); await writeFile(path.join(snap, "tokenizer.json"), "y");
  const model = { id: "img_model", downloads: [{ provider: "huggingface", repo: "O/img", revision: rev, variant: "q4", default: true, files: ["q4/*", "tokenizer.json"] }, { provider: "huggingface", repo: "O/soft", revision: rev, required: "soft", coRequisite: true, files: ["s.bin"] }] };
  const rows = routeDownloadRows(model, { platform: "darwin", tier: "q4" });
  assert.equal((await routePresence(empty, rows)).complete, false);
  const present = await routePresence(full, rows);
  assert.equal(present.complete, true); assert.equal(present.optionalMissing.length, 1);
  const route = (id, group, modelId = "img_model") => ({ id, backend: "mlx", model: modelId, kind: "image", mode: "text_to_image", tier: "q4", group, endpoint: "/api/v1/image/jobs", body: { model: modelId, seed: 1 } });
  const planned = await planRoutes({ routes: [route("r1", "g"), route("r2", "g"), route("r3", null, "absent_model")], hubs: [empty, full], baseCatalog: [model], headCatalog: [model], platform: "darwin" });
  assert.deepEqual(planned.map((item) => item.status), ["ready", "skipped", "skipped"]);
  assert.equal(planned[0].hub, full); assert.equal(planned[0].presence.length, 2);
  assert.match(planned[1].skipReason, /alternative r1/); assert.match(planned[2].skipReason, /not in the base catalog/);
  const drifted = { ...model, downloads: [{ ...model.downloads[0], revision: "b".repeat(40) }] };
  assert.match((await planRoutes({ routes: [route("r1", null)], hubs: [full], baseCatalog: [drifted], headCatalog: [model], platform: "darwin" }))[0].skipReason, /differ between base and head/);
  const missing = await planRoutes({ routes: [route("r1", null)], hubs: [empty], baseCatalog: [model], headCatalog: [model], platform: "darwin" });
  assert.match(missing[0].skipReason, /weights absent from every hub root/);
});

test("shipped catalogs at this checkout satisfy every default route's model and tier", async () => {
  const models = await readCatalog(root);
  for (const [backend, ids] of Object.entries(DEFAULT_ROUTE_IDS)) for (const id of ids) {
    const route = ROUTES[id], model = models.find((entry) => entry.id === route.model);
    assert.ok(model, `${id}: ${route.model} missing from catalog`);
    const rows = routeDownloadRows(model, { platform: backend === "mlx" ? "darwin" : "win32", tier: route.tier });
    assert.ok(rows.required.length > 0, `${id} needs pinned rows`);
  }
});

test("CUDA video group: valid short-clip requests at this catalog, first present member runs", async (t) => {
  assert.deepEqual(DEFAULT_ROUTE_IDS.cuda, ["cuda-sdxl-q4-t2i", "cuda-moss-image-t2i", "cuda-ltx23-distilled-t2v", "cuda-wan22-ti2v-5b-t2v"]);
  assert.ok(!DEFAULT_ROUTE_IDS.cuda.includes("cuda-ltx25-distilled-t2v") && ROUTES["cuda-ltx25-distilled-t2v"], "LTX-2.5 stays selectable, not default, until the Candle provider declares a footprint");
  const models = await readCatalog(root);
  for (const [id, expected] of [["cuda-ltx23-distilled-t2v", { model: "ltx_2_3", duration: 4, fps: 24, width: 768, height: 512, steps: 8 }], ["cuda-wan22-ti2v-5b-t2v", { model: "wan_2_2", duration: 4, fps: 16, width: 832, height: 480, steps: 8 }]]) {
    const route = ROUTES[id], body = route.body, model = models.find((entry) => entry.id === route.model);
    assert.equal(route.group, "cuda-video"); assert.equal(route.kind, "video"); assert.equal(route.endpoint, "/api/v1/video/jobs");
    assert.deepEqual(route.tierCandidates ?? [route.tier], id === "cuda-wan22-ti2v-5b-t2v" ? ["q8", "q4"] : ["q4"], "untiered Wan follows Candle's q8-then-q4 order; LTX resolves exactly q4");
    assert.equal(body.advanced.mlxQuantize, undefined, "the request itself names no tier");
    assert.deepEqual({ model: body.model, duration: body.duration, fps: body.fps, width: body.width, height: body.height, steps: body.advanced.steps }, expected);
    assert.equal(body.mode, "text_to_video"); assert.equal(typeof body.seed, "number");
    assert.ok(model.capabilities.includes("text_to_video"), `${id} catalog offers t2v`);
    assert.ok(model.limits.durations.includes(body.duration) && model.limits.fps.includes(body.fps) && model.limits.resolutions.includes(`${body.width}x${body.height}`), `${id} request inside catalog limits`);
    if (model.limits.steps) assert.ok(model.limits.steps.includes(body.advanced.steps));
    assert.ok(routeDownloadRows(model, { platform: "win32", tier: "q4" }).required.length > 0);
    assert.ok(route.claimMinutes === 10 && route.timeoutMinutes === 45);
  }
  // Group selection against fake hubs: LTX-2.3 present wins; absent LTX-2.3 lets Wan run.
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-video-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const place = async (hub, model) => { for (const row of routeDownloadRows(models.find((entry) => entry.id === model), { platform: "win32", tier: "q4" }).required) { for (const selector of row.files) { const target = path.join(hub, `models--${row.repo.replace("/", "--")}`, "snapshots", row.revision, ...selector.replace(/\/\*$/, "/w.bin").split("/")); await mkdir(path.dirname(target), { recursive: true }); await writeFile(target, "w"); } } };
  const both = path.join(dir, "both"), wanOnly = path.join(dir, "wan");
  await place(both, "ltx_2_3"); await place(both, "wan_2_2"); await place(wanOnly, "wan_2_2");
  const video = [ROUTES["cuda-ltx23-distilled-t2v"], ROUTES["cuda-wan22-ti2v-5b-t2v"]];
  const plannedBoth = await planRoutes({ routes: video, hubs: [both], baseCatalog: models, headCatalog: models, platform: "win32" });
  assert.deepEqual(plannedBoth.map((item) => item.status), ["ready", "skipped"]); assert.match(plannedBoth[1].skipReason, /alternative cuda-ltx23-distilled-t2v of group cuda-video/);
  const plannedWan = await planRoutes({ routes: video, hubs: [wanOnly], baseCatalog: models, headCatalog: models, platform: "win32" });
  assert.deepEqual(plannedWan.map((item) => item.status), ["skipped", "ready"]); assert.match(plannedWan[0].skipReason, /weights absent/);
});

test("identity subset, path redaction and diff are stable and hub-relative", () => {
  const payload = { prompt: "secret prompt", seed: 3, model: "flux_dev", modelBinding: { artifact: { path: "E:\\huggingface\\hub\\models--O--m\\snapshots\\x\\q4", sha256: "abc" } }, advanced: { steps: 4, tier: "q4" }, output: { dir: "/tmp/state/data/projects/p" } };
  const subset = identitySubset(payload);
  assert.deepEqual(Object.keys(subset).sort(), ["advanced", "model", "modelBinding"]);
  assert.deepEqual(subset.advanced, { tier: "q4" });
  const redacted = redactPaths(subset, { hub: "E:\\huggingface\\hub", "hf-home": "E:\\huggingface" });
  assert.equal(redacted.modelBinding.artifact.path, "<hub>/models--O--m/snapshots/x/q4");
  assert.equal(redactPaths("/Users/r/work/state/x", { work: "/Users/r/work", state: "/Users/r/work/state" }), "<state>/x");
  const many = Object.fromEntries(Array.from({ length: 300 }, (_, i) => [`gemma/f${i}#entry`, { link: true }]));
  const relinked = Object.fromEntries(Object.keys(many).map((key) => [key, { link: false }]));
  const full = diffObjects(many, relinked, HUB_MUTATION_LISTING_LIMIT);
  assert.equal(full.count, 300); assert.equal(full.entries.length, 300); assert.equal(full.truncated, false, "a 300-entry hub listing is kept whole in report.json");
  const diff = diffObjects({ a: 1, b: { c: [1, 2] } }, { a: 1, b: { c: [1, 3] }, d: true });
  assert.deepEqual(diff.entries.map((entry) => entry.path), ["/b/c/1", "/d"]); assert.equal(diff.count, 2);
});

test("memory sampling helpers parse top MEM units and walk the worker tree", () => {
  assert.equal(parseTopMem("2640K"), 2640 * 1024); assert.equal(parseTopMem("512M+"), 512 * 1024 ** 2); assert.equal(parseTopMem("31G-"), 31 * 1024 ** 3); assert.equal(parseTopMem("12"), 12);
  assert.throws(() => parseTopMem("n/a"));
  assert.deepEqual(descendants([{ pid: 2, ppid: 1 }, { pid: 3, ppid: 2 }, { pid: 4, ppid: 3 }, { pid: 5, ppid: 9 }], 2), [2, 3, 4]);
});

test("report shape: summary counts, markdown table and per-mode file names", () => {
  const routes = [{ id: "a", model: "m", status: "ran", verdict: "equal", request: { sha256: "r" }, hub: "/h", runs: [{ label: "base#1", status: "completed", timing: { enqueueToCompleteSeconds: 12.5 }, memory: { peakBytes: 2 * 1024 ** 3 }, output: { content: { method: "png_rgba8", sha256: "x" }, fileSha256: "f" } }, { label: "head", status: "completed", timing: { enqueueToCompleteSeconds: 13 }, memory: { peakBytes: 2 * 1024 ** 3 }, output: { content: { sha256: "x" } } }], comparison: { payloadIdentityDiff: { count: 1, entries: [{ path: "/modelBinding", base: null, head: "{}" }] } } }, { id: "b", model: "n", status: "skipped", skipReason: "weights absent" }];
  const summary = summarize(routes);
  assert.deepEqual({ ran: summary.ran, skipped: summary.skipped, equal: summary.equal }, { ran: 1, skipped: 1, equal: 1 });
  const markdown = renderMarkdown({ backend: "mlx", mode: "run", status: "finished", exitCode: 0, base: { sha: BASE }, head: { sha: HEAD }, runner: { name: "nax-macos-2", platform: "darwin", arch: "arm64", totalMemoryBytes: 128 * 1024 ** 3 }, startedAt: "s", finishedAt: "f", memoryMethod: "m", routes, summary });
  assert.match(markdown, /\| a \| m \| ran \| equal \| 12\.5 s \| 13\.0 s \| – \| 2\.00 GiB/);
  assert.match(markdown, /\| b \| n \| skipped \| – .*weights absent/);
  assert.match(markdown, /`\/modelBinding`: base null → head \{\}/);
  assert.equal(reportName("run"), "report"); assert.equal(reportName("preflight"), "preflight");
});

test("CLI refuses missing mode and invalid inputs before touching any checkout", () => {
  const cli = path.join(root, "scripts", "tier2-ab.mjs");
  const none = spawnSync(process.execPath, [cli], { encoding: "utf8", env: { ...process.env, ...env() } });
  assert.notEqual(none.status, 0); assert.match(none.stderr, /exactly one of --preflight-only/);
  const bad = spawnSync(process.execPath, [cli, "--preflight-only"], { encoding: "utf8", env: { ...process.env, ...env({ TIER2_AB_BASE_SHA: "main" }) } });
  assert.notEqual(bad.status, 0); assert.match(bad.stderr, /40-hex/);
});

// ---- --run wiring against a fake API/worker binary (plain Node; no build, model or GPU) ----
// The fake speaks only the API surface the driver uses: port-0 announcement, owned health,
// worker registration, project, job lifecycle with stages, saved PNG and the product file route.
// Each job's PNG carries a different iTXt chunk; pixels come from the seed plus mode.json's offset.
const FAKE_APP = String.raw`#!/usr/bin/env node
const http = require("http"), fs = require("fs"), path = require("path"), zlib = require("zlib");
const env = process.env, mode = JSON.parse(fs.readFileSync(path.join(__dirname, "mode.json"), "utf8"));
const parent = Number(env.SCENEWORKS_PARENT_PID);
setInterval(() => { try { process.kill(parent, 0); } catch { process.exit(0); } }, 100);
// Mirror the real worker's Windows rule (lib.rs wait_for_process_shutdown_source): with
// SCENEWORKS_WORKER_CHILD=1, stdin EOF is a graceful-shutdown request, so a child started with
// stdin "ignore" goes offline right after registering. Applied here on every platform.
const childStdinShutdown = (onEnd) => { if (env.SCENEWORKS_WORKER_CHILD === "1") { process.stdin.on("end", onEnd); process.stdin.on("error", onEnd); process.stdin.resume(); } };
if (env.SCENEWORKS_WORKER_ONLY === "1") {
  if (mode.workerExit) process.exit(mode.workerExit);
  if (mode.fail) console.log("fake worker failure detail");
  childStdinShutdown(() => { const req = http.request(env.SCENEWORKS_API_URL + "/fake/offline", { method: "POST" }, (res) => { res.resume(); res.on("end", () => process.exit(0)); }); req.on("error", () => process.exit(0)); req.end(env.SCENEWORKS_WORKER_ID); });
  const register = () => { const req = http.request(env.SCENEWORKS_API_URL + "/fake/register", { method: "POST" }, (res) => res.resume()); req.on("error", () => setTimeout(register, 100)); req.end(env.SCENEWORKS_WORKER_ID); };
  register(); return;
}
childStdinShutdown(() => process.exit(0));
const data = env.SCENEWORKS_DATA_DIR, projects = path.join(data, "projects");
fs.mkdirSync(path.dirname(env.SCENEWORKS_JOBS_DB_PATH), { recursive: true }); fs.writeFileSync(env.SCENEWORKS_JOBS_DB_PATH, "");
const workers = [], jobs = new Map(), owned = new Map();
const crc = (b) => { let c = 0xffffffff; for (const v of b) { c ^= v; for (let i = 0; i < 8; i++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0); } return (c ^ 0xffffffff) >>> 0; };
const chunk = (t, d) => { const o = Buffer.alloc(d.length + 12); o.writeUInt32BE(d.length); o.write(t, 4); d.copy(o, 8); o.writeUInt32BE(crc(Buffer.concat([Buffer.from(t), d])), o.length - 4); return o; };
const png = (value, note) => { const h = Buffer.alloc(13); h.writeUInt32BE(4); h.writeUInt32BE(4, 4); h[8] = 8; h[9] = 2; const rows = Buffer.alloc(4 * 13, value & 255); for (let y = 0; y < 4; y++) rows[y * 13] = 0; return Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"), chunk("IHDR", h), chunk("iTXt", Buffer.from("workflow\0\0\0\0\0" + note)), chunk("IDAT", zlib.deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]); };
const send = (res, status, value) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(value)); };
const server = http.createServer((req, res) => {
  let body = ""; req.on("data", (c) => (body += c)); req.on("end", () => {
    const url = new URL(req.url, "http://x"), p = url.pathname;
    if (p === "/fake/offline") { const row = workers.find((w) => w.id === body); if (row) row.status = "offline"; return send(res, 200, {}); }
    if (p === "/fake/register") { workers.push({ id: body, status: "idle", gpuId: "mlx", gpuName: "Apple Silicon (MLX)", capabilities: ["gpu", "image_generate"] }); return send(res, 200, {}); }
    if (p === "/api/v1/health") return send(res, 200, { status: "ok", service: "sceneworks-api", runtime: "rust", readiness: { status: "ready" }, authRequired: false, directories: { data, config: env.SCENEWORKS_CONFIG_DIR, projects, jobsDb: env.SCENEWORKS_JOBS_DB_PATH } });
    if (p === "/api/v1/workers") return send(res, 200, workers);
    if (p === "/api/v1/projects" && req.method === "POST") { const id = "p" + owned.size; const dir = path.join(projects, id); fs.mkdirSync(dir, { recursive: true }); owned.set(id, fs.realpathSync(dir)); return send(res, 201, { id, path: fs.realpathSync(dir) }); }
    if (p === "/api/v1/image/jobs") {
      // Immutable attempt bundle, as head's worker publishes under data/models (read-only dirs).
      const bundle = path.join(data, "models", "portable-attempts", "r", "bundle", "hf");
      if (!fs.existsSync(bundle)) { fs.mkdirSync(bundle, { recursive: true }); fs.writeFileSync(path.join(bundle, "f.bin"), "x"); if (mode.outside) fs.symlinkSync(mode.outside, path.join(bundle, "link")); fs.chmodSync(path.join(bundle, "f.bin"), 0o444); for (const d of [bundle, path.dirname(bundle), path.dirname(path.dirname(bundle))]) fs.chmodSync(d, 0o555); }
      if (mode.overwrite && !mode.overwritten) { mode.overwritten = true; const size = fs.statSync(mode.overwrite).size; fs.writeFileSync(mode.overwrite, "x".repeat(size)); }
      if (mode.mutate && !mode.mutated) { mode.mutated = true; fs.unlinkSync(mode.mutate); fs.writeFileSync(mode.mutate, "w"); }
      const request = JSON.parse(body); const id = "job-" + Date.now() + "-" + jobs.size; jobs.set(id, { id, request, created: Date.now() }); return send(res, 201, { id, type: "image_generate", payload: request }); }
    const job = /^\/api\/v1\/jobs\/([^/]+)$/.exec(p);
    if (job) {
      const j = jobs.get(job[1]), age = Date.now() - j.created, done = age > 900;
      const stage = age < 150 ? "queued" : age < 400 ? "loading_model" : age < 900 ? "generating" : "completed";
      if (j.canceled) return send(res, 200, { id: j.id, status: "canceled", stage: "canceled", message: "", workerId: null, payload: j.request, result: {} });
      if (mode.neverClaim) return send(res, 200, { id: j.id, status: "queued", stage: "queued", message: "waiting for an eligible worker", blockedReason: "no worker publishes a matching source claim", workerId: null, payload: j.request, result: {} });
      if (mode.fail && age > 300) return send(res, 200, { id: j.id, status: "failed", stage: "failed", message: "", error: "model_artifact_incomplete: fake refusal", workerId: workers[0].id, payload: j.request, result: {} });
      const view = { id: j.id, status: done ? "completed" : age < 150 ? "queued" : "running", stage, message: "", workerId: age < 150 ? null : workers[0].id, backend: "mlx", assignedGpu: "mlx", payload: { ...j.request, modelBinding: { revision: mode.binding, path: path.join(env.HF_HUB_CACHE, "models--x") } }, result: {} };
      if (done) { const rel = "assets/images/" + j.id + ".png"; const file = path.join(owned.get(j.request.projectId), ...rel.split("/")); if (!fs.existsSync(file)) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, png(j.request.seed + mode.offset, j.id)); } view.result = { adapter: "fake", model: j.request.model, assets: [{ id: "a-" + j.id, type: "image", file: { path: rel, mimeType: "image/png" } }], assetIds: ["a-" + j.id] }; }
      return send(res, 200, view);
    }
    const file = /^\/api\/v1\/projects\/([^/]+)\/files\/(.+)$/.exec(p);
    if (file) { res.writeHead(200); return res.end(fs.readFileSync(path.join(owned.get(file[1]), ...decodeURIComponent(file[2]).split("/")))); }
    const cancel = /^\/api\/v1\/jobs\/([^/]+)\/cancel$/.exec(p);
    if (cancel) { const j = jobs.get(cancel[1]); if (j) j.canceled = true; return send(res, 200, {}); }
    send(res, 404, { error: p });
  });
});
server.listen(0, "127.0.0.1", () => console.log(JSON.stringify({ event: "api_listening", address: "127.0.0.1:" + server.address().port })));
`;
const macWithPython = process.platform === "darwin" && process.arch === "arm64" && spawnSync("python3.12", ["--version"]).status === 0;
test("--run drives fake base/head services through the stop rule and writes evidence", { skip: !macWithPython && "needs Apple-silicon macOS with python3.12 (the mlx preflight's own requirements)", timeout: 120_000 }, async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-run-")); t.after(() => { spawnSync("chmod", ["-R", "u+w", dir]); return rm(dir, { recursive: true, force: true }); });
  const pin = "4".repeat(40), revision = "f".repeat(40);
  const lock = `[[package]]\nname = "pmetal-mlx-sys"\nversion = "0.1.0"\nsource = "git+https://example.invalid/mlx-rs?rev=${"9".repeat(40)}#${"9".repeat(40)}"\n\n[[package]]\nname = "gen-core"\nversion = "0.1.0"\nsource = "git+https://github.com/SceneWorks/inference?rev=${pin}#${pin}"\n`;
  const catalog = JSON.stringify({ models: [{ id: "moss_image_sft", downloads: [{ provider: "huggingface", repo: "OpenMOSS-Team/MOSS-Image-SFT", revision, files: ["a.bin"] }] }, { id: "flux_dev", downloads: [{ provider: "huggingface", repo: "SceneWorks/flux1-dev-mlx", revision, variant: "q4", default: true, files: ["q4/*"] }] }] });
  const git = (cwd, ...args) => { const result = spawnSync("git", ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", ...args], { cwd, encoding: "utf8" }); assert.equal(result.status, 0, result.stderr); return result.stdout.trim(); };
  const sides = {};
  for (const [label, note] of [["base", "b"], ["head", "h"]]) {
    const repo = path.join(dir, label); await mkdir(path.join(repo, "config", "manifests"), { recursive: true });
    await writeFile(path.join(repo, "Cargo.lock"), lock); await writeFile(path.join(repo, "config", "manifests", "builtin.models.jsonc"), catalog); await writeFile(path.join(repo, "note"), note);
    git(repo, "init", "-q"); git(repo, "add", "."); git(repo, "commit", "-qm", label); sides[label] = { repo, sha: git(repo, "rev-parse", "HEAD") };
  }
  const hub = path.join(dir, "hub"), snapshot = path.join(hub, "models--OpenMOSS-Team--MOSS-Image-SFT", "snapshots", revision);
  await mkdir(snapshot, { recursive: true }); await writeFile(path.join(snapshot, "a.bin"), "w");
  // A snapshot entry that is a link to a blob, as the Hugging Face cache lays it out.
  const blobFile = path.join(hub, "models--OpenMOSS-Team--MOSS-Image-SFT", "blobs", "b1"); await mkdir(path.dirname(blobFile), { recursive: true }); await writeFile(blobFile, "blob-bytes");
  const { symlink: linkFile } = await import("node:fs/promises"); await linkFile(path.relative(snapshot, blobFile), path.join(snapshot, "linked.bin"));
  const fluxSnapshot = path.join(hub, "models--SceneWorks--flux1-dev-mlx", "snapshots", revision, "q4"); await mkdir(fluxSnapshot, { recursive: true }); await writeFile(path.join(fluxSnapshot, "w.bin"), "w");
  const outside = path.join(dir, "outside-hub-file"); await writeFile(outside, "hub bytes"); const { chmod: chmodFile, stat: statFile } = await import("node:fs/promises"); await chmodFile(outside, 0o444);
  const scenario = async (name, headOffset, headMode = {}, extraEnv = {}, baseMode = {}) => {
    const work = path.join(dir, `work-${name}`), built = {};
    for (const label of ["base", "head"]) {
      const bin = path.join(work, "scratch", "bin", label), binary = path.join(bin, "sceneworks-rust-api");
      await mkdir(bin, { recursive: true }); await writeFile(binary, FAKE_APP, { mode: 0o755 });
      await writeFile(path.join(bin, "mode.json"), JSON.stringify({ offset: label === "head" ? headOffset : 0, binding: label, outside, ...(label === "head" ? headMode : baseMode) }));
      built[label] = { binary, binarySha256: sha256(await readFile(binary)), metallib: null };
    }
    await writeFile(path.join(work, "scratch", "build.json"), JSON.stringify({ baseSha: sides.base.sha, headSha: sides.head.sha, backend: "mlx", built }));
    const childEnv = { PATH: process.env.PATH, HOME: process.env.HOME, TIER2_AB_BACKEND: "mlx", TIER2_AB_BASE_SHA: sides.base.sha, TIER2_AB_HEAD_SHA: sides.head.sha, TIER2_AB_BASE_DIR: sides.base.repo, TIER2_AB_HEAD_DIR: sides.head.repo, TIER2_AB_WORK: work, TIER2_AB_ROUTES: "mlx-moss-image-t2i", TIER2_AB_HUBS: hub, ...extraEnv };
    const startedAt = Date.now();
    const result = spawnSync(process.execPath, [path.join(root, "scripts", "tier2-ab.mjs"), "--run"], { env: childEnv, encoding: "utf8", timeout: 100_000 });
    return { status: result.status, stderr: result.stderr, stdout: result.stdout, seconds: (Date.now() - startedAt) / 1000, report: JSON.parse(await readFile(path.join(work, "evidence", "report.json"), "utf8")), work };
  };
  const equal = await scenario("equal", 0);
  assert.equal(equal.status, 0, equal.stderr + JSON.stringify(equal.report.fatal));
  const route = equal.report.routes[0];
  assert.equal(route.verdict, "equal"); assert.deepEqual(route.runs.map((run) => run.label), ["base#1", "head"], "stop after head == base#1");
  assert.notEqual(route.runs[0].output.fileSha256, route.runs[1].output.fileSha256, "metadata chunk differs per job");
  assert.equal(route.runs[0].output.content.sha256, route.runs[1].output.content.sha256);
  assert.equal(route.runs[0].identity.payload.modelBinding.path, "<hub>/models--x");
  assert.equal(route.comparison.payloadIdentityDiff.count, 1, "binding revision base->head is reported");
  assert.ok(route.runs.every((run) => run.timing.enqueueToCompleteSeconds > 0 && Number.isFinite(run.timing.loadSeconds) && run.memory.peakBytes > 0 && run.cleanup.every((c) => c.exited && !c.forced)));
  assert.equal(route.hubMutation.changed, false);
  assert.ok(route.runs.every((run) => run.hubMutation && run.hubMutation.changed === false), "each run is fingerprinted on its own");
  const differs = await scenario("differs", 1);
  assert.equal(differs.status, 1); assert.match(differs.stderr, /head_differs/);
  assert.equal(differs.report.routes[0].verdict, "head_differs");
  assert.deepEqual(differs.report.routes[0].runs.map((run) => run.label), ["base#1", "head", "base#2"]);
  assert.equal(differs.report.routes[0].comparison.numericDifference.changedElements, 48);
  assert.match(await readFile(path.join(differs.work, "evidence", "report.md"), "utf8"), /head_differs/);
  // Live progress: one line per phase transition, in order, for every run plus the verdict.
  const phases = equal.stdout.split("\n").filter((line) => line.startsWith("[tier2-ab] ")).map((line) => / route=(\S+)(?: run=(\S+))? phase=(\S+)/.exec(line)?.slice(1).filter(Boolean).join(":")).filter(Boolean);
  for (const expected of ["mlx-moss-image-t2i:route_start", "mlx-moss-image-t2i:base#1:services_up", "mlx-moss-image-t2i:base#1:submitted", "mlx-moss-image-t2i:base#1:claimed", "mlx-moss-image-t2i:base#1:completed", "mlx-moss-image-t2i:head:completed", "mlx-moss-image-t2i:verdict"]) assert.ok(phases.includes(expected), `missing progress ${expected} in ${phases.join(", ")}`);
  assert.ok(phases.indexOf("mlx-moss-image-t2i:base#1:submitted") < phases.indexOf("mlx-moss-image-t2i:base#1:claimed") && phases.indexOf("mlx-moss-image-t2i:base#1:claimed") < phases.indexOf("mlx-moss-image-t2i:base#1:completed"));
  assert.match(equal.stdout, /phase=driver_started mode=run/);
  // An unclaimed head job stops at the claim bound with the served reason, not the execution ceiling.
  const unclaimed = await scenario("unclaimed", 0, { neverClaim: true }, { TIER2_AB_CLAIM_MINUTES: "0.05" });
  const stuck = unclaimed.report.routes[0].runs.find((run) => run.label === "head");
  assert.equal(unclaimed.status, 1); assert.equal(unclaimed.report.routes[0].verdict, "error");
  assert.match(stuck.error, /^unclaimed: job stayed queued\/queued \(waiting for an eligible worker\) for more than 0.05 min/);
  assert.equal(stuck.diagnostics.lastJob.status, "queued"); assert.equal(stuck.diagnostics.lastJob.reasons["job.blockedReason"], "no worker publishes a matching source claim");
  assert.ok(stuck.diagnostics.apiLogTail.length > 0 && stuck.diagnostics.apiLogTail.length <= 50 && Array.isArray(stuck.diagnostics.workerLogTail));
  assert.ok(stuck.timing.claimSeconds === undefined || stuck.timing.claimSeconds === null);
  assert.ok(unclaimed.seconds < 60, `claim bound must stop early, took ${unclaimed.seconds}s`);
  assert.match(await readFile(path.join(unclaimed.work, "evidence", "report.md"), "utf8"), /last job queued\/queued "waiting for an eligible worker" reasons/);
  assert.match(unclaimed.stdout, /run=head phase=run_error .*reason=unclaimed/);
  // A worker that exits before registering on base#1 of the first route stops the whole run.
  const noWorker = await scenario("noworker", 0, {}, { TIER2_AB_ROUTES: "mlx-moss-image-t2i,mlx-flux1-dev-t2i" }, { workerExit: 3 });
  assert.equal(noWorker.status, 1); assert.match(noWorker.report.fatal, /owned worker registration failed on base#1 of the first route/);
  const [firstRoute, secondRoute] = noWorker.report.routes;
  assert.equal(firstRoute.verdict, "error"); assert.deepEqual(firstRoute.runs.map((run) => run.label), ["base#1"]);
  assert.equal(secondRoute.status, "not_run"); assert.match(secondRoute.skipReason, /could not register/);
  const dead = firstRoute.runs[0];
  assert.match(dead.error, /^owned worker exited before registering idle \(code 3/); assert.equal(dead.failureClass, "worker_registration");
  assert.deepEqual({ alive: dead.diagnostics.processes.worker.alive, exitCode: dead.diagnostics.processes.worker.exitCode, stdin: dead.diagnostics.processes.worker.stdin }, { alive: false, exitCode: 3, stdin: "pipe" });
  assert.equal(dead.diagnostics.processes.api.alive, true); assert.ok(Array.isArray(dead.diagnostics.workersResponse));
  assert.ok(noWorker.seconds < 30, `stopped after one run, took ${noWorker.seconds}s`);
  assert.match(await readFile(path.join(noWorker.work, "evidence", "report.md"), "utf8"), /worker exited code 3 signal null \(stdin pipe\); \/workers: /);
  // A hub change made only by the BASE worker is a prominent note; one made by HEAD fails the run.
  const mutated = path.join(snapshot, "a.bin");
  const baseMut = await scenario("basemut", 0, {}, {}, { mutate: mutated });
  assert.equal(baseMut.status, 0, baseMut.stderr); const baseRoute = baseMut.report.routes[0];
  assert.equal(baseRoute.verdict, "equal"); assert.equal(baseRoute.hubMutation.changed, true); assert.equal(hubMutationAttribution(baseRoute), "base_only");
  assert.deepEqual(baseRoute.runs.map((run) => [run.label, run.hubMutation.changed]), [["base#1", true], ["head", false]]);
  assert.match(await readFile(path.join(baseMut.work, "evidence", "report.md"), "utf8"), /\*\*Note:\*\* the BASE worker changed the model hub during mlx-moss-image-t2i[\s\S]*hub mutation during the BASE run only \(does not fail this comparison\)/);
  const headMut = await scenario("headmut", 0, { mutate: mutated });
  assert.equal(headMut.status, 1); assert.equal(hubMutationAttribution(headMut.report.routes[0]), "head");
  assert.match(await readFile(path.join(headMut.work, "evidence", "report.md"), "utf8"), /hub mutation \(head; fails this comparison\)/);
  // A same-length rewrite of the blob behind a snapshot link, done by the HEAD worker, is caught and attributed.
  const blobMut = await scenario("blobmut", 0, { overwrite: blobFile });
  assert.equal(blobMut.status, 1); const blobRoute = blobMut.report.routes[0];
  assert.equal(hubMutationAttribution(blobRoute), "head");
  assert.deepEqual(blobRoute.runs.map((run) => [run.label, run.hubMutation.changed]), [["base#1", false], ["head", true]]);
  assert.ok(blobRoute.runs[1].hubMutation.diff.entries.some((item) => item.path.includes("linked.bin#entry/target/")));
  // A failed job stops immediately with the served error.
  const failed = await scenario("failed", 0, { fail: true });
  const failedHead = failed.report.routes[0].runs.find((run) => run.label === "head");
  assert.equal(failed.status, 1);
  assert.equal(failedHead.status, "error");
  assert.match(failedHead.error, /^job failed: model_artifact_incomplete: fake refusal/);
  assert.equal(failedHead.diagnostics.lastJob.status, "failed");
  assert.equal(failedHead.diagnostics.lastJob.error, "model_artifact_incomplete: fake refusal");
  assert.ok(failedHead.diagnostics.workerLogTail.includes("fake worker failure detail"));
  assert.equal(failedHead.cancelRequested, undefined);
  assert.ok(failedHead.cleanup.every((child) => child.exited));
  assert.match(failed.stdout, /run=head phase=run_error_pending_cleanup .*last_status=failed/);
  // Cleanup restores write permission only inside scratch, never through a link, then removes it.
  const cleanupEnv = { PATH: process.env.PATH, HOME: process.env.HOME, TIER2_AB_BACKEND: "mlx", TIER2_AB_BASE_SHA: sides.base.sha, TIER2_AB_HEAD_SHA: sides.head.sha, TIER2_AB_BASE_DIR: sides.base.repo, TIER2_AB_HEAD_DIR: sides.head.repo, TIER2_AB_WORK: equal.work, TIER2_AB_HUBS: hub, RUNNER_TEMP: dir, GITHUB_WORKSPACE: dir };
  const blocked = spawnSync("rm", ["-rf", equal.work], { encoding: "utf8" });
  assert.notEqual(blocked.status, 0, "the read-only bundle reproduces the plain rm failure");
  const cleaned = spawnSync(process.execPath, [path.join(root, "scripts", "tier2-ab.mjs"), "--cleanup"], { env: cleanupEnv, encoding: "utf8" });
  assert.equal(cleaned.status, 0, cleaned.stdout + cleaned.stderr);
  for (const gone of [equal.work, sides.base.repo, sides.head.repo]) await assert.rejects(statFile(gone));
  assert.equal((await statFile(outside)).mode & 0o777, 0o444, "a linked file outside scratch keeps its mode");
  assert.equal(await readFile(path.join(snapshot, "a.bin"), "utf8"), "w", "hub untouched");
});

// ---- static workflow contracts (dispatch-only, runner labels, GPU binding, cleanup) ----
const workflowText = (name) => readFile(path.join(root, ".github", "workflows", name), "utf8");
function job(workflow, name) {
  const start = workflow.indexOf(`\n  ${name}:\n`);
  assert.ok(start >= 0, `${name} job exists`);
  const next = workflow.slice(start + 1).search(/\n  [a-z][a-z0-9-]*:\n/);
  return workflow.slice(start, next < 0 ? undefined : start + 1 + next);
}

test("tier2-ab is reachable only as a dispatched campaign profile and keeps the 25-input ceiling", async () => {
  const dispatcher = await workflowText("memory-catalog-campaign.yml");
  assert.match(dispatcher, /^on:\n  workflow_dispatch:\n/m);
  assert.doesNotMatch(dispatcher, /^  (push|pull_request|pull_request_target|schedule|merge_group|workflow_run):/m);
  assert.match(dispatcher, /          - tier2-ab\n/);
  assert.match(dispatcher, /if \(i\.profile === 'tier2-ab'\) \{ emit\('enabled','false'\); process\.exit\(0\); \}/);
  const inputs = dispatcher.split("    inputs:\n")[1].split("\n# `gh pr create`")[0];
  assert.equal([...inputs.matchAll(/^      ([A-Za-z_][A-Za-z0-9_]*):$/gm)].length, 25);
  const selector = job(dispatcher, "tier2-ab-selector");
  assert.match(selector, /if: \$\{\{ needs\.moss-selector-preflight\.outputs\.enabled != 'true' && inputs\.profile == 'tier2-ab' \}\}/);
  assert.match(selector, /dispatchOptions/); assert.match(selector, /runs-on: ubuntu-latest/);
  const caller = job(dispatcher, "tier2-ab");
  assert.match(caller, /if: \$\{\{ needs\.moss-selector-preflight\.outputs\.enabled != 'true' && inputs\.profile == 'tier2-ab' \}\}/);
  assert.match(caller, /uses: \.\/\.github\/workflows\/tier2-ab\.yml/);
  assert.doesNotMatch(caller, /secrets[.:]/, "the A/B caller passes no secrets");
  assert.match(selector, /fetch-depth: 0\n\s+filter: tree:0/); assert.match(selector, /assertTrustedShas\(\{base_sha:options\.base_sha,head_sha:options\.head_sha\}/);
  for (const catalogJob of ["mlx", "candle"]) assert.doesNotMatch(job(dispatcher, catalogJob).split("\n").find((line) => line.startsWith("    if:")), /tier2-ab/);
  const reusable = await workflowText("tier2-ab.yml");
  const triggers = reusable.slice(reusable.indexOf("\non:"), reusable.indexOf("\npermissions:"));
  assert.match(triggers, /workflow_call:/);
  assert.doesNotMatch(triggers, /\n  (workflow_dispatch|push|pull_request|schedule|merge_group):/);
  assert.match(reusable, /permissions:\n  contents: read/);
});

test("tier2-ab jobs bind the selected physical index through lease and worker masks", async () => {
  const reusable = await workflowText("tier2-ab.yml");
  const mlx = job(reusable, "mlx"), cuda = job(reusable, "cuda");
  assert.match(mlx, /runs-on: \[self-hosted, macOS, ARM64, rw-mage\]/);
  assert.doesNotMatch(mlx, /\b(nax|real-weights|rw-starvector|weights)\]/);
  assert.match(cuda, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  for (const item of [mlx, cuda]) {
    assert.match(item, /timeout-minutes: [0-9]+/);
    assert.match(item, /cancel-in-progress: false/);
    assert.match(item, /--preflight-only/); assert.match(item, /--build-only/); assert.match(item, /--run/);
    assert.match(item, /if: \$\{\{ !inputs\.preflight_only \}\}/);
    assert.match(item, /actions\/upload-artifact@[0-9a-f]{40}/);
  }
  assert.match(mlx, /name: Remove every scratch directory\s+if: \$\{\{ always\(\) \}\}/);
  assert.match(cuda, /name: Remove every scratch directory\s+if: \$\{\{ always\(\) && inputs\.suite != 'krea-scoped-acceptance' \}\}/);
  assert.match(mlx, /metal_execution_entry [0-9]+ node tier2-ab-driver\/scripts\/tier2-ab\.mjs --run/);
  assert.match(mlx, /node tier2-ab-driver\/scripts\/tier2-ab\.mjs --cleanup/); assert.match(cuda, /node tier2-ab-driver\\scripts\\tier2-ab\.mjs --cleanup/);
  assert.doesNotMatch(reusable, /rm -rf -- "\$\{TIER2_AB_WORK/, "plain rm of the work dir fails on read-only bundles");
  assert.match(mlx, /phase=waiting_for_metal_lock/); assert.match(cuda, /phase=waiting_for_gpu_lease/);
  for (const [key, value] of [["CUDA_DEVICE_ORDER", "PCI_BUS_ID"], ["RUSTC_WRAPPER", '""']]) assert.match(cuda, new RegExp(`\\n      ${key}: ${value}\\n`));
  for (const key of ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SCENEWORKS_GPU_ID", "TIER2_AB_PHYSICAL_GPU"]) assert.ok(cuda.includes(`${key}: \${{ inputs.cuda_physical_index }}`));
  assert.match(cuda, /\$minFreeMiB = node .*cudaLeaseMinFreeMiB\(process\.env\.TIER2_AB_ROUTES\)/);
  assert.match(cuda, /python -I scripts\/ci\/windows_cuda_execution\.py --actor tier2-ab-cuda --min-free-mib \$minFreeMiB --budget-minutes [0-9]+ --cuda-ordinals 0 --require-physical-index \$env:TIER2_AB_PHYSICAL_GPU -- node scripts\/tier2-ab\.mjs --run/);
  assert.match(cuda, /TIER2_AB_GPU_POLICY: .*inputs\.cuda_physical_index == '0' && 'shared-compute' \|\| 'shared-display'/);
  // The TIER2_AB_GPU_POLICY label is this profile's alone (its report wording); admission itself is the
  // wrapper's one rule for every actor.
  for (const name of ["windows-candle.yml", "memory-catalog-campaign.yml", "load-progress-acceptance.yml"]) assert.doesNotMatch(await workflowText(name), /shared-display/);
  assert.match(cuda, /Add-Content -Path \$env:GITHUB_ENV -Value 'RUSTC_WRAPPER='/);
  assert.match(cuda, /vcvars64\.bat/);
  assert.match(mlx, /if: \$\{\{ inputs\.download_missing && !inputs\.preflight_only && inputs\.suite != 'krea-scoped-acceptance' \}\}/);
  assert.match(mlx, /--stage-wan/);
  assert.match(cuda, /if: \$\{\{ inputs\.download_missing && !inputs\.preflight_only && inputs\.suite != 'krea-scoped-acceptance' \}\}/);
  assert.match(cuda, /Join-Path \$env:TIER2_AB_WORK 'hf-cli'/);
  assert.match(cuda, /-r \.github\/requirements-hf\.txt/);
  assert.match(cuda, /--stage-wan[\s\S]+--preflight-only[\s\S]+--build-only[\s\S]+--run\b/);
  assert.doesNotMatch(reusable, /secrets[.:]|HF_TOKEN:/, "no step, build or run, receives a secret");
});

function scailCatalog(catalog, enabled) {
  return catalog.map((model) => {
    if (model.id !== "scail2_14b") return model;
    const ui = { ...model.ui };
    if (enabled === undefined) delete ui.scail2MultiReference;
    else ui.scail2MultiReference = enabled;
    return { ...model, ui };
  });
}

test("SCAIL requires an enabled candidate catalog and both pinned product components", async () => {
  const catalog = await readCatalog(root);
  const evidence = await readPatternEvidence(root);
  const head = scailCatalog(catalog, true), closed = scailCatalog(catalog, false);
  const fs = { stat: async () => ({ isFile: () => true, size: 99 }) };
  for (const backend of ["mlx", "cuda"]) {
    const routes = Object.values(ROUTES).filter((r) => r.scailCandidate && r.backend === backend);
    assert.deepEqual(routes.map((r) => r.references), [2, 3, 4, 5, 6, 6]); assert.ok(routes.at(-1).refuse);
    const options = { routes, platform: backend === "mlx" ? "darwin" : "win32", hubs: [backend === "mlx" ? "/hub" : "E:\\hub"], baseCatalog: closed, headCatalog: head, evidence, fs };
    await assert.rejects(planRoutes({ ...options, headCatalog: closed }), /plural character descriptor/);
    await assert.rejects(planRoutes({ ...options, baseCatalog: head }), /requires a base without/);
    const absentFlag = await planRoutes({ ...options, baseCatalog: scailCatalog(catalog, undefined) });
    const plans = await planRoutes(options);
    assert.deepEqual(absentFlag, plans);
    for (const [i, plan] of plans.entries()) {
      assert.deepEqual(plan.pluralCharacterSupport, { base: false, head: true });
      assert.equal(plan.status, "ready");
      assert.equal(plan.request.body.referenceAssetIds.length, routes[i].references);
      assert.deepEqual(plan.rows.required.map((r) => r.repo), ["SceneWorks/scail2-mlx", "SceneWorks/sam3-mlx"]);
      assert.deepEqual(plan.rows.required[1].files, ["model.safetensors", "tokenizer.json"]);
      const report = { mode: "run", routes: [{ ...plan, status: "ran", verdict: "verified" }], summary: { ran: 1 } };
      assert.equal(exitCodeFor(report), 0);
      report.routes[0].verdict = "skipped"; assert.equal(exitCodeFor(report), 1);
    }
    assert.deepEqual(selectRoutes(backend, routes[5].id), [routes[5]]);
    assert.throws(() => selectRoutes(backend, [routes[0].id, routes[5].id].join(",")), /preceding accepted/);
    assert.throws(() => selectRoutes(backend, [routes[5].id, routes[4].id].join(",")), /preceding accepted/);
    assert.equal(selectRoutes(backend, [routes[4].id, routes[5].id].join(",")).length, 2);
    assert.ok(!DEFAULT_ROUTE_IDS[backend].some((id) => ROUTES[id].scailCandidate));
  }
});

test("CUDA grouped and standalone SCAIL refusals preserve the worker reserve above the effective cap", async () => {
  const { scailPackagedRefusalCapBytes, scailRefusalCapBytes, scailRefusalBudget, verifyScailRefusal } = await import("./lib/tier2-scail.mjs");
  const config = async (name) => JSON.parse(await readFile(path.join(root, "config", name), "utf8"));
  const [registry, anchor] = await Promise.all([
    config("manifests/registry/evidence.json"), readFile(path.join(root, "crates/sceneworks-core/src/memory_anchor.rs"), "utf8"),
  ]);
  const gib = 1024 ** 3;
  const reserveBytes = Number(anchor.match(/pub const DEDICATED_VRAM_ALLOCATOR_SLACK_GB: f64 = (\d+(?:\.\d+)?);/)[1]) * gib;
  // Freeze the measured donor that reproduced the emulation bug. Its coverage is
  // specific to that capture's provider closure; future pins can have different prices.
  const price = {grossDeviceBytes:57_048_903_680 + 9_294_577_664, grossHostBytes:58_836_885_504 + 77_594_624, grossUnifiedBytes:0};
  const six = {artifactId:14_719_969_057, sha256:"750313a3cdb89dd4a0e4e49b31935949301b309fe9410f2bf0ce47cb4628575f",
    references:6, nativePolicy:"cuda-q4-resident", price, acceleratorBytes:price.grossDeviceBytes};
  const evidence = [six];
  const cold = {...six.price, stage:"cold_load", admitted:true, backend:"candle", referenceCount:6, providerCreditBytes:0,
    requiredAcceleratorBytes:six.acceleratorBytes, acceleratorReserveBytes:reserveBytes,
    evidenceArtifactId:six.artifactId, evidenceSha256:six.sha256, evidenceReferences:6};
  const grouped = scailRefusalBudget("cuda", scailRefusalCapBytes([cold], evidence));
  const standalone = scailRefusalBudget("cuda", scailPackagedRefusalCapBytes(evidence));
  assert.deepEqual(grouped, standalone);
  assert.deepEqual(grouped, {effectiveCapBytes:66_343_481_343, rawCapBytes:68_490_964_991});
  const fromEnv = Number(String(grouped.rawCapBytes / gib)) * gib;
  const legacyQ4Bytes = registry.scail2_14b.candle.vramGbByTier.q4 * gib + reserveBytes;
  assert.ok(grouped.effectiveCapBytes < legacyQ4Bytes && fromEnv > legacyQ4Bytes);
  assert.equal(fromEnv - reserveBytes, six.acceleratorBytes - 1);
  const denied = {...cold, admitted:false, availableAcceleratorBytes:grouped.effectiveCapBytes};
  const message = (e) => `video load failed: unsupported: SCAIL-2 plural fit requires ${e.requiredAcceleratorBytes} accelerator bytes, only ${e.availableAcceleratorBytes} are available`;
  verifyScailRefusal([denied], grouped.effectiveCapBytes, message(denied));
  assert.throws(() => verifyScailRefusal([{...denied, acceleratorReserveBytes:0}], grouped.effectiveCapBytes, message(denied)), /reserve/);
  const over = {...denied, availableAcceleratorBytes:grouped.effectiveCapBytes + 1};
  assert.throws(() => verifyScailRefusal([over], grouped.effectiveCapBytes, message(over)), /ceiling/);
  assert.deepEqual(scailRefusalBudget("mlx", grouped.effectiveCapBytes), {effectiveCapBytes:grouped.effectiveCapBytes, rawCapBytes:grouped.effectiveCapBytes});
});

test("SCAIL workers must observe the exact file seal, ordered RGB inputs and live cold and execution budgets", async () => {
  const { scailFitEvents, scailArtifactSha256, scailRefusalCapBytes, verifyScailRefusal } = await import("./lib/tier2-scail.mjs");
  for (const backend of ["mlx", "cuda"]) {
    for (const n of [2, 3, 4, 5, 6]) {
      const route = ROUTES[`${backend}-scail2-q4-animate-${n}`];
      const sources = { references: Array.from({ length: n }, (_, i) => ({ rgbSha256: sha256(String(i)) })) };
      const lane = backend === "mlx" ? backend : "candle";
      const pairs = sources.references.map((_, i) => `reference-${i}-1:0:832x480:reference-mask-${i}-1:0:832x480`).join(":");
      const frames = Array.from({ length: 45 }, (_, i) => `${i}:832x480`).join(",");
      const carrier = `provider_video_mode:animation+scail2-carrier-v2:backend-${lane}:task-animation:composite-832x480:characters-${n}:${pairs}:driving-frames-45:${frames}:driving-mask-45:${frames}:strength-3f800000:mode-FaceOnly:frame-0+scail2-carrier-request-seal-v2-${sha256("carrier")}`;
      const price = { grossHostBytes: backend === "mlx" ? 0 : 100, grossDeviceBytes: backend === "mlx" ? 0 : 100,
        grossUnifiedBytes: backend === "mlx" ? 100 : 0 };
      const cold = { ...price, event: "scail2_plural_fit", evidenceSha256: sha256("shared-receipt"), evidenceArtifactId: 781, evidenceReferences: n, jobId: "owned", stage: "cold_load", backend: lane, tier: "q4", artifactTier: "q4",
        width: 832, height: 480, frames: 45, referenceCount: n, artifactSha256: scailArtifactSha256, carrierReceipt: carrier,
        referenceRgbSha256: sources.references.map((r) => r.rgbSha256).join(","), requiredAcceleratorBytes: 100,
        availableAcceleratorBytes: 100, acceleratorReserveBytes: 2 * 1024 ** 3, requiredPhysicalHostBytes: 100, availablePhysicalHostBytes: 100, providerCreditBytes: 0, admitted: true };
      const execution = { ...cold, stage: "execution" };
      const pool = [...new Set([2, n, 6])].map((references) => ({ sha256: cold.evidenceSha256, artifactId: 781,
        device: "board", runtime: "runtime", nativePolicy:"native-policy", references, price, acceleratorBytes: 100 }));
      const verify = (entries, admitted = true) => scailFitEvents(entries.map((e) => JSON.stringify(e)).join("\n"), route, "owned", admitted, sources, pool);
      assert.equal(verify([cold, execution]).length, 2);
      const usingCapture = (references) => [cold, execution].map((event) => ({ ...event, evidenceReferences: references }));
      if (n === 6) {
        assert.throws(() => verify(usingCapture(2)), /covering the requested reference count/);
        assert.equal(scailRefusalCapBytes([cold], pool), 99);
        const six = pool.find((row) => row.references === 6);
        assert.equal(scailRefusalCapBytes([cold], [...pool, { ...six, artifactId:782, acceleratorBytes:75 },
          { ...six, artifactId:783, nativePolicy:"other", acceleratorBytes:5 }]), 74);
        assert.throws(() => scailRefusalCapBytes([cold], [...pool, six]), /one physical capture/);
        assert.throws(() => scailRefusalCapBytes(usingCapture(2), pool), /covering the requested reference count/);
        assert.throws(() => scailRefusalCapBytes([{ ...cold, evidenceArtifactId: 780 }], pool));
      }
      if (backend === "mlx" && n < 6) {
        assert.equal(verify(usingCapture(6)).length, 2);
        assert.throws(() => verify([cold, { ...execution, evidenceReferences: 6 }]), /changed its admitted physical evidence/);
      } else if (backend === "cuda" && n < 6) {
        assert.throws(() => verify(usingCapture(6)), /covering the requested reference count/);
      }
      for (const change of [{ referenceRgbSha256: cold.referenceRgbSha256.split(",").reverse().join(",") },
        { stage: "cold_load" }, { artifactSha256: sha256("wrong") }, { evidenceSha256: sha256("other") }, { evidenceArtifactId: 17 }, { grossHostBytes: 95 }, { requiredAcceleratorBytes: 99 }, { carrierReceipt: carrier.replace("reference-mask-1", "reference-mask-0") },
        { jobId: "another" }, { carrierReceipt: carrier.replace(sha256("carrier"), sha256("another")) },
        { availablePhysicalHostBytes: null }, { availableAcceleratorBytes: 99 }, { tier: "bf16" }]) {
        assert.throws(() => verify([cold, { ...execution, ...change }]));
      }
      for (const malformed of [carrier.replace("provider_video_mode:animation+", ""),
        carrier.replace("provider_video_mode:animation", "provider_video_mode:replacement"),
        carrier.replace("animation+", "animation++"), carrier + "+extra", carrier.replace(":strength-", ":extra:strength-")]) {
        assert.throws(() => verify([cold, execution].map((e) => ({ ...e, carrierReceipt: malformed }))));
      }
      const denied = { ...cold, admitted: false, availableAcceleratorBytes: 99 };
      const refused = verify([denied], false);
      const detail = "SCAIL-2 plural fit requires 100 accelerator bytes, only 99 are available";
      const wrapped = "video load failed: unsupported: " + detail;
      const message = backend === "mlx" ? detail : wrapped;
      verifyScailRefusal(refused, 99, message);
      assert.throws(() => verifyScailRefusal([...refused, { ...denied, requiredAcceleratorBytes: 102 }], 99, message), /observed byte counts/);
      assert.throws(() => verifyScailRefusal(refused, 98, message), /ceiling/);
      for (const wrong of [undefined, backend === "mlx" ? wrapped : detail, "video load failed: " + detail,
        "unsupported: " + detail, message + ".", message.replace("100", "101"), message.replace("99", "98"),
        "video load failed: prepared source or placement changed"]) {
        assert.throws(() => verifyScailRefusal(refused, 99, wrong), /observed byte counts/);
      }
      assert.throws(() => verify([denied, execution], false));
      assert.throws(() => verify([{ ...denied, availableAcceleratorBytes: 150, availablePhysicalHostBytes: 99 }], false));
    }
  }
});

test("SCAIL rendered outputs require the backend decoder length, changing pixels, actual seed and exported source IDs", async () => {
  const { verifyScailOutput } = await import("./lib/tier2-scail.mjs");
  const sources = { references: [{ id: "first" }, { id: "second" }], clip: { id: "drive" } };
  for (const backend of ["mlx", "cuda"]) {
    const route = ROUTES[`${backend}-scail2-q4-animate-2`], lane = backend === "mlx" ? "mlx" : "candle";
    const buffer = Buffer.alloc(832 * 480 * 4 * route.savedFrames);
    buffer[4] = 7;
    const video = { width: 832, height: 480, avgFrameRate: "16/1", decodedFrames: route.savedFrames };
    const content = { method: "ffmpeg_rgba_frames+s16le_audio+ffprobe_streams", shape: { video, audio: null }, buffer };
    const metrics = { model: route.model, backend, steps: 4, guidanceScale: 5, width: 832, height: 480, seed: 33402, totalMs: 75 };
    const recipe = { adapter: lane + "_scail2", loras: [], prompt: route.body.prompt, model: route.model, mode: route.mode, seed: 33402,
      rawAdapterSettings: { realModelInference: true, frameCount: route.savedFrames }, normalizedSettings: { sourceClipAssetId: "drive", referenceAssetIds: ["first", "second"] } };
    const identity = { provider_id: route.model, backend: lane, randomness: { kind: "draws", draws: [{ seed: 33402 }] },
      schedule: { state: "resolved", value: [{ digest: { count: 5, sha256: sha256("schedule") } }] },
      sampling: { state: "resolved", value: [backend === "mlx" ? { solver: "UniPC" } : "wan_flow_unipc_bh2_order2_v1"] } };
    const receipt = { version: 1, acceptedBackend: lane, jobId: "owned", outputs: [{ artifactIds: ["export"], execution: { state: "resolved", identity } }] };
    const verify = () => verifyScailOutput(route, content, receipt, "owned", "export", recipe, sources, metrics);
    assert.equal(verify().decodedFrameSha256.length, route.savedFrames);
    video.decodedFrames = backend === "mlx" ? 45 : 48; assert.throws(verify); video.decodedFrames = route.savedFrames;
    recipe.normalizedSettings.referenceAssetIds.reverse(); assert.throws(verify); recipe.normalizedSettings.referenceAssetIds.reverse();
    identity.randomness.draws[0].seed++; assert.throws(verify); identity.randomness.draws[0].seed--;
    buffer.fill(0); assert.throws(verify, /single color/);
  }
});

test("SCAIL imports ordered files through project multipart and verifies the served bytes", async (t) => {
  const { uploadScailMedia } = await import("./lib/tier2-scail.mjs");
  const dir = await mkdtemp(path.join(os.tmpdir(), "scail-import-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const refs = ["A.png", "B.png"].map((name) => path.join(dir, name));
  const rgb = refs.map((_, i) => Buffer.alloc(832 * 480 * 3, i + 17));
  for (const [i, ref] of refs.entries()) await writeFile(ref, png(rgb[i], 832, 480, 3));
  const clip = path.join(dir, "driving.mp4");
  await writeFile(clip, Buffer.from("video fixture"));
  const uploads = new Map();
  const endpoint = {
    async exchange(method, route, form, options) {
      assert.deepEqual([method, route, options], ["POST", "/api/v1/projects/p/assets", { multipart: true }]);
      assert.deepEqual([...form.keys()], ["file"]);
      const file = form.get("file"), rel = "assets/" + file.name;
      uploads.set(rel, Buffer.from(await file.arrayBuffer()));
      return { status: 201, body: { id: file.name, projectId: "p", type: file.type.split("/")[0], file: { path: rel } } };
    },
    async served(url) { return { bytes: uploads.get(url.replace("/api/v1/projects/p/files/", "")) }; },
  };
  const route = ROUTES["mlx-scail2-q4-animate-2"];
  const uploaded = await uploadScailMedia(route, endpoint, "p", { refs, clip }, decodePng);
  assert.deepEqual(uploaded.references.map((r) => r.rgbSha256), rgb.map(sha256));
  const request = requestBody(route, "p", uploaded);
  assert.deepEqual(request.referenceAssetIds, ["A.png", "B.png"]);
  assert.equal(request.sourceClipAssetId, "driving.mp4");
  endpoint.served = async () => ({ bytes: Buffer.from("altered") });
  await assert.rejects(uploadScailMedia(route, endpoint, "p", { refs, clip }, decodePng), /served input differs/);
});

test("standalone SCAIL refusal authenticates both compiled candidate digests and identifies the leased device", async () => {
  const { scailPackageDigests } = await import("./lib/tier2-scail.mjs");
  const { scailHostIdentity } = await import("./tier2-ab.mjs");
  const [index, coverage, approvals] = await Promise.all([readFile(path.join(root, "config/scail2-plural-memory.json")),
    readFile(path.join(root, "config/scail2-memory-pin-coverage.json")),
    readFile(path.join(root, "crates/sceneworks-worker/src/video_admission/scail2/evidence.rs"), "utf8")]);
  assert.deepEqual(scailPackageDigests(index, coverage, approvals), { indexSha256: sha256(index), coverageSha256: sha256(coverage) });
  assert.throws(() => scailPackageDigests(index + " ", coverage, approvals), /APPROVED_INDEX_SHA256/);
  assert.throws(() => scailPackageDigests(index, coverage + " ", approvals), /APPROVED/);
  assert.throws(() => scailPackageDigests(index, coverage, ""), /missing or ambiguous/);
  for (const backend of ["mlx", "cuda"]) {
    const mlx = backend === "mlx";
    const uuid = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0", luid = "luid_0x00000000_0x00000002";
    const identity = await scailHostIdentity(backend, { TIER2_AB_PHYSICAL_GPU: "0" }, async (cmd, args) => {
      if (mlx) {
        assert.equal(cmd, "/usr/sbin/sysctl");
        assert.deepEqual(args, ["-n", "machdep.cpu.brand_string", "kern.osversion"]);
        return { stdout: "board\nruntime\n" };
      }
      if (cmd === "python") {
        assert.equal(args.at(-1), "--wddm-adapter");
        return { stdout: JSON.stringify({ context_created: false, devices: [{ ordinal: 0, uuid, luid, node_mask: 1 }] }) };
      }
      assert.equal(cmd, "C:\\Windows\\System32\\nvidia-smi.exe");
      assert.deepEqual(args, ["--id=0", "--query-gpu=name,driver_version,memory.total,uuid", "--format=csv,noheader,nounits"]);
      return { stdout: `board, runtime, 97887, ${uuid}` };
    });
    assert.deepEqual(identity, { device: "board", runtime: "runtime",
      ...(!mlx ? { deviceTotalMib: 97887, captureAdapter: { cuda_uuid: uuid, luid, node_mask: 1 } } : {}) });
    if (!mlx) await assert.rejects(scailHostIdentity(backend, {}, async (cmd) => ({ stdout: cmd === "python"
      ? JSON.stringify({ context_created: false, devices: [{ ordinal: 0, uuid: "GPU-e4b79931-7be6-f216-460a-f5405cfafffe", luid, node_mask: 1 }] })
      : `board, runtime, 97887, ${uuid}` })), /logical CUDA 0 and selected physical adapter differ/);
    for (const stdout of ["", "board", "board,runtime\nother,driver", ...(!mlx ? [`board, runtime, ${uuid}`,
      `board, runtime, 0, ${uuid}`, `board, runtime, [N/A], ${uuid}`] : [])]) {
      await assert.rejects(scailHostIdentity(backend, {}, async () => ({ stdout })), /identity unavailable/);
    }
  }
});

test("SCAIL product reads the actual projector output, including shared receipt IDs and reviewed prices", async (t) => {
  const { scailFitEvidence, scailArtifactSha256, scailPackagedRefusalCapBytes } = await import("./lib/tier2-scail.mjs");
  const dir = await mkdtemp(path.join(os.tmpdir(), "scail-project-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const put = async (name, data) => {
    const bytes = JSON.stringify(data); await writeFile(path.join(dir, name), bytes); return sha256(bytes);
  };
  for (const backend of ["mlx", "cuda"]) {
    const mlx = backend === "mlx", counts = mlx ? [2, 6] : [2, 3, 4, 5, 6];
    const revision = "1".repeat(40);
    const contract = { compiledPin: revision, coverage: [], closures: {} };
    const receipt = { accepted: true, run_id: "4321", run_attempt: "1", contract: {protocol: "scail2-plural-fit-v1", head: revision},
      hardware: { name:"board", chip:"board", driver_version:"runtime", os_build:"runtime" }, counts:{}, logs:{} };
    if (!mlx) Object.assign(receipt.hardware, { uuid: "GPU-e4b79931-7be6-f216-460a-f5405cfafffe", "memory.total": "97887" });
    if (!mlx) receipt.adapter = { cuda_uuid: "GPU-e4b79931-7be6-f216-460a-f5405cfafffe", counter: String.raw`\GPU Process Memory(*)\Dedicated Usage`,
      luid: "luid_0x00000000_0x0001e266", physical_node: 0, node_mask: 1 };
    for (const n of counts) {
      // Pinned SCAIL window values by count: packed tokens, final FFN rows, CUDA query rows and scores.
      const [tokens, tail, rows, scores] = [
        [26520, 1944, 942, 999273600], [28080, 3504, 890, 999648000], [29640, 5064, 843, 999460800],
        [31200, 6624, 801, 999648000], [32760, 8184, 763, 999835200],
      ][n - 2];
      const window = { latent_frames: 12, history_frames: 0, decoded_frames: mlx ? 48 : 45, tokens,
        mlx_unsplit_ffn: mlx, mlx_ffn_chunks: 4, mlx_ffn_tail: tail,
        self_attention_query_rows: mlx ? 0 : rows, self_attention_score_elements: mlx ? 0 : scores };
      const axes = { backend, artifact_sha256: scailArtifactSha256, artifact_tier:"q4", effective_tier:"q4", width:832, height:480,
        references:n, history_order:4, classifier_free_guidance:true, mlx_dpmpp:false, windows:[window],
        source_bytes:(2*n+90)*832*480*3, driving_stack_bytes:45*832*480*24, schedule_bytes:320, source_resize_bytes:0,
        source_shape_sha256:sha256("fixture shape "+n), load_policy:(mlx?"None":"Some(Q4)")+"/Bf16/Resident/EagerMaterialization",
        execution_policy:[{name:mlx?"SCAIL2_COMPUTE_BF16":"CANDLE_DEQUANTIZE_ALL",value:null}], floor_host_bytes:0, floor_device_bytes:0, floor_unified_bytes:0 };
      const name = `count-${n}.supervisor.json`;
      const guard = { returncode: 0, pid: n + 100, lifetime_peak_host_bytes: 90, peak_host_bytes: 95,
        admission: { childFootprintCapBytes: 200, childGpuCapBytes: 160, policySha256: sha256("policy") } };
      if (!mlx) {
        // Whole-card growth is 120 - 20 bytes; PID, native pool and WDDM samples already describe the child.
        const wddmProcess = { adapter: receipt.adapter, pid: guard.pid, created_filetime: 900 + n, error: null, missing_active_polls: 0 };
        Object.assign(guard.admission, { cudaDeviceUuid: receipt.adapter.cuda_uuid, gpuGlobalUsedBaselineBytes: 20 });
        guard.peak_gpu_bytes = 80;
        guard.samples = [{ phase: "cuda-global-used-upper-bound-sample", cudaDeviceUuid: receipt.adapter.cuda_uuid,
          globalUsedUpperBoundBytes: 120, childGlobalUsedGrowthBytes: 100, gpuFreeBytes: 95, at_ns: 105 }];
        guard.cuda_attribution = { wddm_dedicated: { ...wddmProcess, samples: [[901, 60], [902, 93]] },
          pid_peak_bytes: 77, unknown_pid_polls: 1, global_upper_bytes: 120, pool_fallback_peak_bytes: 80 };
        guard.fit_observation = { wddm_shared: { ...wddmProcess, samples: [[901, 0], [902, 2]] },
          wddm_shared_counter: "GPU Process Memory/Shared Usage",
          card_samples_monotonic_ns_free_owned_or_fallback_bytes: [[1, 190, 0], [2, 95, 80], [3, 140, 40]] };
      }
      receipt.logs[name] = { sha256: await put(name, guard) };
      receipt.counts[n] = { accepted:true, claim:{post_execution_verified:true}, native: {axes, pid:n+100, prompt_tokens:[512,512], allocator_peak:80},
        candidate_row_pending_physical_review:{ platform:mlx?"macos":"windows", device:"board", runtime:"runtime", inference_revision:revision,
          witness:{ axes, text_tokens:512, gross_host_bytes:145, gross_unified_bytes:140, gross_device_bytes:120, mlx_decode_budget_bytes:16 } } };
    }
    const digest = await put("receipt.json", receipt), projected = [];
    const aggregate = { basis:"measured_aggregate", co_live_extra_bytes:5, uncertainty_allowance_bytes:8 };
    for (const n of counts) {
      await put("review.json", { artifact_id:781, run_id:4321, run_attempt:1, receipt_sha256:digest, references:n, rationale:"metadata fixture",
        domains:mlx ? {unified:aggregate} : {host:aggregate, device:{basis:"guarded_ceiling", unsampled_overshoot_upper_bytes:15}} });
      const produced = spawnSync(process.platform === "win32" ? "python" : "python3", [path.join(root, "scripts/project-scail2-memory.py"),
        path.join(dir, "receipt.json"), path.join(dir, "review.json")], { encoding:"utf8" });
      assert.equal(produced.status, 0, produced.stderr); projected.push(JSON.parse(produced.stdout));
    }
    const read = (pool, compatibility = contract) => scailFitEvidence(pool, backend, compatibility);
    const rows = read(projected);
    assert.deepEqual(rows.map((r) => r.references), counts);
    const host = { device: "board", runtime: "runtime", env: {},
      ...(!mlx ? { deviceTotalMib: 97887, captureAdapter: projected[0].bounds.capture_adapter } : {}) };
    const scoped = (pool, current = host) => read(pool, { ...contract, host: current });
    if (!mlx) {
      assert.ok(projected.every((r) => r.bounds.device_total_mib === 97887));
      const quoted = (evidence) => evidence.map((r) => [r.references, r.nativePolicy, r.artifactId, r.sha256, r.price]);
      // A worker on a second identical card (another UUID, LUID and node) gets the same proof and prices,
      // and so does a capture that ran on that card: the adapter is provenance, not the hardware class.
      const gpu1 = { cuda_uuid: "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0", luid: "luid_0x00000000_0x000000fe", node_mask: 2 };
      assert.deepEqual(scoped(projected, { ...host, captureAdapter: gpu1 }), scoped(projected));
      const moved = structuredClone(projected);
      for (const r of moved) r.bounds.capture_adapter = gpu1;
      assert.deepEqual(quoted(scoped(moved)), quoted(scoped(projected)));
      // The device model, the card's total memory and the driver still refuse.
      for (const change of [{ device: "other" }, { runtime: "other" }, { deviceTotalMib: 48943 }, { deviceTotalMib: undefined }]) {
        assert.throws(() => scoped(projected, { ...host, ...change }), /no packaged native/);
      }
      for (const total of [undefined, 0, "97887"]) {
        const blank = structuredClone(projected);
        blank[0].bounds.device_total_mib = total;
        assert.throws(() => read(blank), /card capacity/);
        assert.throws(() => scoped(blank), /card capacity/);
      }
      // A complete second campaign on the other card runs under its own guard (the executed
      // policy names its card). The worker prices either, so both campaigns stay available,
      // each priced from its own rows; one card still keeps a single guard.
      const second = structuredClone(moved);
      for (const r of second) {
        Object.assign(r, { artifact_id: 900, guard_policy_sha256: sha256("gpu1 guard") });
        r.bounds.device.unsampled_overshoot_upper_bytes -= 1;
      }
      const both = scoped([...projected, ...second]);
      assert.deepEqual(both.map((r) => [r.artifactId, r.references]),
        [...projected.map((r) => [r.artifact_id, r.axes.references]), ...second.map((r) => [900, r.axes.references])]);
      const own = scoped(projected);
      for (const [i, r] of both.entries()) {
        if (r.artifactId === 900) {
          assert.equal(r.captureAdapter.cuda_uuid, gpu1.cuda_uuid);
          assert.equal(r.price.grossDeviceBytes, own[i % counts.length].price.grossDeviceBytes - 1);
        } else assert.deepEqual(r, own[i]);
      }
      assert.notEqual(both[0].campaign, both[counts.length].campaign);
      const sameCard = structuredClone(projected);
      for (const r of sameCard) Object.assign(r, { artifact_id: 901, guard_policy_sha256: sha256("second gpu0 guard") });
      assert.throws(() => scoped([...projected, ...sameCard]), /different capture guards/);
      // Partial campaigns on two cards never combine into one complete campaign, even under one guard.
      const halves = [...projected.filter((r) => r.axes.references <= 4), ...moved.filter((r) => r.axes.references > 4)];
      assert.throws(() => scoped(halves), /no packaged native/);
      assert.throws(() => read(halves), /no packaged native/);
      assert.throws(() => scoped([...projected, ...moved.filter((r) => r.axes.references > 4)]), /different capture guards/);
    }
    const competing = structuredClone(projected.at(-1));
    competing.artifact_id = 782;
    const envelope = mlx ? competing.bounds.unified : competing.bounds.device;
    if (mlx) envelope.uncertainty_allowance_bytes -= 1;
    else envelope.unsampled_overshoot_upper_bytes -= 1;
    const price = rows.at(-1).acceleratorBytes;
    assert.equal(scailPackagedRefusalCapBytes(scoped([...projected, competing])), price - 2);
    for (const change of [r => { r.device = "other"; }, r => { r.runtime = "other"; },
      r => { if (mlx) r.runtime = "another"; else r.bounds.device_total_mib = 48943; },
      r => { r.axes.execution_policy[0].value = "1"; }]) {
      const other = structuredClone(competing); change(other);
      assert.deepEqual(scoped([...projected, other]), rows);
      assert.equal(scailPackagedRefusalCapBytes(scoped([...projected, other])), price - 1);
    }
    assert.throws(() => scoped(projected, { ...host, runtime: "wrong" }), /no packaged native/);
    assert.throws(() => scoped(projected, { ...host, env: { [projected[0].axes.execution_policy[0].name]: "" } }), /no packaged native/);
    assert.throws(() => scailPackagedRefusalCapBytes(rows.filter((r) => r.references !== 6)));
    assert.throws(() => scailPackagedRefusalCapBytes([{ references: 6, acceleratorBytes: 1 }]));
    assert.ok(rows.every((r) => r.artifactId === 781 && r.sha256 === digest));
    assert.deepEqual(rows[0].price, mlx ? {grossHostBytes:0,grossDeviceBytes:0,grossUnifiedBytes:153}
      : {grossHostBytes:158,grossDeviceBytes:175,grossUnifiedBytes:0});
    assert.throws(() => read(projected.slice(1)), /no packaged native/);
    const wrong = (mutate) => { const copy = structuredClone(projected); mutate(copy[0]); assert.throws(() => read(copy)); };
    const short = structuredClone(projected);
    for (const r of short.filter((r) => r.axes.references < 6)) {
      Object.assign(r.axes.windows[0], { latent_frames: 1, decoded_frames: 1 });
    }
    assert.throws(() => read(short), /different native SCAIL window/);
    for (const key of Object.keys(projected[0].axes.windows[0])) {
      wrong((r) => {
        const window = r.axes.windows[0];
        if (typeof window[key] === "boolean") window[key] = !window[key];
        else window[key]++;
      });
    }
    wrong((r) => { r.guard_policy_sha256 = sha256("different policy"); });
    wrong((r) => { r.post_execution_verified = false; });
    wrong((r) => { if (mlx) r.bounds.mlx_decode_budget_bytes=154; else r.axes.load_policy="unpriced policy"; });
    wrong((r) => { (mlx ? r.bounds.unified : r.bounds.host).classified_peak_bytes = 2; });
    wrong((r) => { if (mlx) r.axes.floor_unified_bytes=154; else r.axes.floor_device_bytes=176; });
    wrong((r) => { (mlx ? r.bounds.unified : r.bounds.host).co_live_extra_bytes = Number.MAX_SAFE_INTEGER; });
    const target = "2".repeat(40), closure = {digest:sha256("provider"),sharedContractDigest:sha256("shared")};
    const coverage = {capture_revision:revision,target_revision:target,backend, provider_digest:closure.digest,
      shared_contract_digest:closure.sharedContractDigest, tree_diff_sha256:sha256("diff")};
    const changed = { compiledPin:target, coverage:[coverage], closures:{inferenceRevision:target,providers:{[(mlx?"mlx":"candle")+":scail2_14b"]:closure}} };
    assert.deepEqual(read(projected, changed), rows);
    const split = structuredClone(projected), six = split.at(-1);
    six.inference_revision = "3".repeat(40);
    six.artifact_id = 1562;
    six.receipt_sha256 = sha256("separate quiet six run");
    const both = { ...changed, coverage: [...changed.coverage, { ...coverage, capture_revision: six.inference_revision,
      tree_diff_sha256: sha256("six to compiled target") }] };
    assert.throws(() => read(split, changed), /no packaged native/);
    if (mlx) assert.throws(() => read(split, both), /no packaged native/);
    else {
      const { scailPackageDigests } = await import("./lib/tier2-scail.mjs");
      const exactSix = structuredClone(split); exactSix[4].inference_revision = target;
      const indexBytes = JSON.stringify(exactSix), staleApproval = JSON.stringify([]);
      const approvals = `const APPROVED_INDEX_SHA256: &str = "${sha256(indexBytes)}"; const APPROVED: &str = "${sha256(staleApproval)}";`;
      assert.deepEqual(read(exactSix, changed).map((r) => r.references), counts);
      assert.throws(() => scailPackageDigests(indexBytes, JSON.stringify(changed.coverage), approvals), /APPROVED/);
      const archived = structuredClone(split);
      Object.assign(archived[4], {accepted:false, artifact_id:14719969057,
        inference_revision:"7f6c4cca3fc053030bd71f0bfce6886adb762eb6",
        receipt_sha256:"750313a3cdb89dd4a0e4e49b31935949301b309fe9410f2bf0ce47cb4628575f", recovery_certificate_sha256:"dcc7c1eb186b001c75f4a9df2d49dd06501e8064be7d1818adb2ea59031c093f"});
      archived[4].bounds.device_interval = "load_through_completed_rgb";
      archived[4].bounds.device = {...archived[4].bounds.host};
      const archiveCoverage = {...both, coverage:[coverage, {...both.coverage[1], capture_revision:archived[4].inference_revision}]};
      assert.equal(read(archived, archiveCoverage).at(-1).recoveryCertificateSha256, archived[4].recovery_certificate_sha256);
      for (const mutate of [r => {r.accepted=true;}, r => {r.receipt_sha256=sha256("other");},
        r => {r.artifact_id+=1;}, r => {delete r.recovery_certificate_sha256;},
        r => {r.recovery_certificate_sha256=sha256("other certification");},
        r => {r.accepted=true; delete r.recovery_certificate_sha256; delete r.bounds.device_interval;},
        r => {r.bounds.device_interval="full_job";}]) {
        const wrong = structuredClone(archived); mutate(wrong[4]); assert.throws(() => read(wrong, archiveCoverage));
      }
      const partial = structuredClone(archived);
      const partialPin = "f5c9ff80acd00334fd668e4ecc8d56be3347affd";
      const certificates = {3:"c18632894d2d40c7d2b0eea519657714d873a611ca9e6f0845b0809dab4a024d",
        5:"bd8567c64d120f24e9821747d94b63faec3e5919930ce65dccb5a79f7df86fe7"};
      for (const r of partial) {
        r.bounds.capture_adapter = {cuda_uuid:"GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0", luid:"luid_0x00000000_0x0001f603", node_mask:1};
        if (r.axes.references === 6) continue;
        const cert = certificates[r.axes.references];
        Object.assign(r, {accepted:false, count_accepted:!cert, artifact_id:14725116132, inference_revision:partialPin,
          receipt_sha256:"d423f72d816ac6145c7553623a26e99f67fd59db8e83ab228a89ede8698ce8af"});
        r.bounds.device = {...r.bounds.host};
        if (cert) {
          r.recovery_certificate_sha256 = cert;
          r.bounds.device_interval = "load_through_completed_rgb";
        }
      }
      const partialCoverage = {...archiveCoverage, coverage:[...archiveCoverage.coverage, {...coverage, capture_revision:partialPin}]};
      assert.deepEqual(read(partial, partialCoverage).map(r => r.references), counts);
      for (const [count, change] of [
        [3, r => {delete r.recovery_certificate_sha256;}],
        [3, r => {r.recovery_certificate_sha256=certificates[5];}],
        [5, r => {r.accepted=true; delete r.count_accepted; delete r.recovery_certificate_sha256; delete r.bounds.device_interval;}],
        [5, r => {delete r.bounds.device_interval;}],
        [2, r => {r.recovery_certificate_sha256=certificates[3];}],
        [4, r => {r.count_accepted=false;}],
        [2, r => {r.inference_revision=target;}],
        [2, r => {r.artifact_id+=1;}],
        [4, r => {r.receipt_sha256=sha256("another receipt");}],
        [3, r => {r.bounds.capture_adapter.node_mask=2;}],
      ]) {
        const wrong = structuredClone(partial);
        change(wrong.find(r => r.axes.references === count));
        assert.throws(() => read(wrong, partialCoverage));
      }
      const latestPin = "4fdd13cce7f590be0aed3c52a73a9cea9e6d1b54";
      const latestCoverage = {...both, coverage:[...both.coverage, {...coverage, capture_revision:latestPin}]};
      const latest = structuredClone(partial);
      for (const r of latest) {
        Object.assign(r, {artifact_id:14735122440, receipt_sha256:"d711f34b0c83b54b41b147921b9601fe999b74ccaec8823b6dceadcd6d9fc545",
          inference_revision:latestPin, count_accepted:r.axes.references !== 3, accepted:false});
        if (r.axes.references === 3) r.recovery_certificate_sha256 = "ca4696364a2a4c951b76f58879e0d25b7e48fdd4d6df5d9e7b605c313cbfd02d";
        else delete r.recovery_certificate_sha256;
        if (![3,6].includes(r.axes.references)) delete r.bounds.device_interval;
      }
      assert.deepEqual(read(latest, latestCoverage).map(r => r.references), counts);
      for (const [count, mutate] of [
        [3, r => {delete r.recovery_certificate_sha256;}], [3, r => {r.recovery_certificate_sha256 = certificates[3];}],
        [3, r => {r.count_accepted = true;}], [6, r => {delete r.bounds.device_interval;}],
        [5, r => {r.bounds.device_interval = "load_through_completed_rgb";}], [2, r => {r.accepted = true; delete r.count_accepted;}],
        [4, r => {r.artifact_id = 14725116132;}],
      ]) {
        const wrong = structuredClone(latest);
        mutate(wrong.find(r => r.axes.references === count));
        assert.throws(() => read(wrong, latestCoverage));
      }
      const joined = read(split, both);
      assert.deepEqual(joined.map((r) => [r.references, r.artifactId, r.sha256, r.inferenceRevision]),
        split.map((r) => [r.axes.references, r.artifact_id, r.receipt_sha256, r.inference_revision]));
      for (const key of ["provider_digest", "shared_contract_digest", "target_revision"]) {
        assert.throws(() => read(split, { ...both, coverage:[coverage, {...both.coverage[1], [key]:sha256("crossed")}] }));
      }
      const altered = structuredClone(split);
      altered[4].guard_policy_sha256 = sha256("different capture guard");
      assert.throws(() => read(altered, both), /no packaged native/);
      altered[4].bounds.device.unsampled_overshoot_upper_bytes -= 1;
      assert.throws(() => read([...split, altered[4]], both), /different capture guards/);
      for (const row of altered) row.guard_policy_sha256 = sha256("different capture guard");
      assert.throws(() => read([...split, ...altered], both), /different capture guards/);
    }
    assert.throws(() => read(projected, { ...changed, coverage:[] }), /no packaged native/);
    assert.throws(() => read(projected, { ...changed, coverage:[{ ...coverage, provider_digest:sha256("wrong") }] }), /no packaged native/);
  }
});

test("SCAIL staging downloads only exact absent q4 and SAM3 filenames on both hosts", async () => {
  const { SAM3, SCAIL_REVISION } = await import("./lib/tier2-scail.mjs");
  const catalog = await readCatalog(root), evidence = await readPatternEvidence(root);
  const baseCatalog = scailCatalog(catalog, false), headCatalog = scailCatalog(catalog, true);
  for (const backend of ["mlx", "cuda"]) {
    const routes = [ROUTES[`${backend}-scail2-q4-animate-6`]], hubs = [backend === "mlx" ? "/local models/hub" : "E:\\local models\\hub"];
    const cfg = { routes, hubs, backend, baseDir: root, headDir: root };
    const absent = new Set(["q4/dit.safetensors", "model.safetensors"]);
    const fs = { stat: async (file) => {
      const member = file.replaceAll("\\", "/").split(/\/snapshots\/[a-f0-9]{40}\//)[1];
      if (absent.has(member)) throw Object.assign(Error("missing"), { code: "ENOENT" });
      return { isFile: () => true, size: 29 };
    } };
    fs.lstat = fs.stat;
    const plan = () => planRoutes({ routes, hubs, baseCatalog, headCatalog, platform: backend === "mlx" ? "darwin" : "win32", fs, evidence });
    const before = await plan(); assert.equal(before[0].status, "skipped");
    const calls = [];
    await stageWanSnapshots({}, cfg, { routes: before }, null, (args) => {
      calls.push(args); absent.delete(args[2]);
    }, fs);
    assert.deepEqual(calls, [
      ["download", "SceneWorks/scail2-mlx", "q4/dit.safetensors", "--revision", SCAIL_REVISION, "--cache-dir", hubs[0]],
      ["download", SAM3.repo, "model.safetensors", "--revision", SAM3.revision, "--cache-dir", hubs[0]],
    ]);
    const after = await plan(); assert.equal(after[0].status, "ready");
    await stageWanSnapshots({}, cfg, { routes: after }, null, () => assert.fail("complete rows"), fs);
  }
});

test("Windows Krea keeps runtime until the registered post action verifies published evidence", async () => {
  const reusable = await workflowText("tier2-ab.yml");
  const mlx = job(reusable, "mlx"), cuda = job(reusable, "cuda");
  assert.doesNotMatch(mlx, /dev-host|DEV_HOST_JOB_ROOT/);
  const registration = cuda.indexOf("./tier2-ab-driver/.github/actions/cleanup-runner-storage");
  assert.ok(registration > cuda.indexOf("name: Check out the dispatched driver"));
  assert.ok(registration < cuda.indexOf("name: Prepare the Rust toolchain"));
  assert.match(cuda, /Register disposable Windows Krea storage\s+if: \$\{\{ inputs\.suite == 'krea-scoped-acceptance' \}\}[\s\S]*dev-host: 'true'/);
  assert.match(cuda, /TIER2_AB_WORK=\$env:DEV_HOST_JOB_ROOT\\tier2-ab/);
  assert.match(cuda, /\$env:RUNNER_TEMP = Join-Path \$env:TIER2_AB_WORK 'evidence'/);
  assert.match(cuda, /steps\.krea-evidence\.outcome == 'success'/);
  assert.match(cuda, /steps\.krea-evidence\.outputs\.artifact-id/);
  assert.match(cuda, /steps\.krea-evidence\.outputs\.artifact-digest/);
  assert.match(cuda, /Missing Windows Krea driver evidence[\s\S]*DEV_HOST_WORK_UPLOADS_COMPLETE=true/);
  assert.match(cuda, /DEV_HOST_WORK_UPLOADS_COMPLETE: "false"/);
});
