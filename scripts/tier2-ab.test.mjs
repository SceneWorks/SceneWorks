// SW-7033 Tier 2 A/B: pure-function and static-workflow contracts only. No build, API, worker,
// model, GPU or hub is touched; synthetic PNG/WAV bytes and temp directories only.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, rm } from "node:fs/promises";
import { deflateSync } from "node:zlib";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { sha256 } from "./lib/file-sha256.mjs";
import {
  DEFAULT_ROUTE_IDS, ROUTES, contentDigest, decodePng, decodedContent, descendants, diffObjects, dispatchOptions,
  childEnvironments, childStdio, exitCodeFor, assertTrustedShas, scrubSecrets, TRUSTED_REF_PATTERNS, trustedRefFor, ffprobePath, gpuInventory, hubFingerprint, memoryCell, streamShape, HUB_MUTATION_LISTING_LIMIT, hubMutationAttribution, processState, identitySubset, jobClaimed, jobDiagnostics, jobProgressDecision, lastLines, numericDifference, progressLine, removeScratch, parseTopMem, parseWav, planRoutes, readCatalog, redactPaths, referenceWav,
  renderMarkdown, reportName, requestBody, requestTemplate, resolveConfig, routeDownloadRows, routePresence, selectRoutes,
  summarize, verdictFor,
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
  assert.deepEqual(dispatchOptions(inputs, inputs.ref), { base_sha: BASE, head_sha: HEAD, backend: "cuda", routes: "", hubs: "", preflight_only: true, inference_sha: "" });
  const custom = dispatchOptions({ ...inputs, backend: "mlx", moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, routes: ["mlx-moss-image-t2i"], hubs: ["/Volumes/X/hub"], preflight_only: false }) }, inputs.ref);
  assert.equal(custom.preflight_only, false); assert.equal(custom.routes, '["mlx-moss-image-t2i"]'); assert.equal(custom.hubs, '["/Volumes/X/hub"]');
  for (const bad of [{ ref: "b".repeat(40) }, { backend: "cpu" }, { download_missing: true }, { moss_options_json: "[]" }, { moss_options_json: "{" }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, case: "x" }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: BASE }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, routes: ["mlx-moss-image-t2i"] }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, hubs: ["/posix/hub"] }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, preflight_only: "no" }) }, { inference_sha: "x" }]) {
    assert.throws(() => dispatchOptions({ ...inputs, ...bad }, inputs.ref), undefined, JSON.stringify(bad));
  }
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
  assert.match(source, /env: \{ \.\.\.scrubSecrets\(env\), CARGO_TARGET_DIR: target \}/, "the cargo build gets the scrubbed environment");
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
  assert.match(failed.report.routes[0].runs.find((run) => run.label === "head").error, /^job failed: model_artifact_incomplete: fake refusal/);
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

test("tier2-ab jobs pin the isolated runners, physical GPU1, ceilings, Metal lock and cleanup", async () => {
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
    assert.match(item, /name: Remove every scratch directory[\s\S]*if: \$\{\{ always\(\) \}\}/);
  }
  assert.match(mlx, /metal_execution_entry [0-9]+ node tier2-ab-driver\/scripts\/tier2-ab\.mjs --run/);
  assert.match(mlx, /node tier2-ab-driver\/scripts\/tier2-ab\.mjs --cleanup/); assert.match(cuda, /node tier2-ab-driver\\scripts\\tier2-ab\.mjs --cleanup/);
  assert.doesNotMatch(reusable, /rm -rf -- "\$\{TIER2_AB_WORK/, "plain rm of the work dir fails on read-only bundles");
  assert.match(mlx, /phase=waiting_for_metal_lock/); assert.match(cuda, /phase=waiting_for_gpu1_lease/);
  for (const [key, value] of [["CUDA_VISIBLE_DEVICES", '"1"'], ["NVIDIA_VISIBLE_DEVICES", '"1"'], ["SCENEWORKS_GPU_ID", '"1"'], ["CUDA_DEVICE_ORDER", "PCI_BUS_ID"], ["RUSTC_WRAPPER", '""']]) assert.match(cuda, new RegExp(`\\n      ${key}: ${value}\\n`));
  assert.match(cuda, /python -I scripts\/ci\/windows_cuda_execution\.py --actor tier2-ab-cuda --admission-policy shared-display --min-free-mib 75776 --budget-minutes [0-9]+ --cuda-ordinals 0 -- node scripts\/tier2-ab\.mjs --run/);
  assert.match(cuda, /\n      TIER2_AB_GPU_POLICY: shared-display\n/);
  // shared-display is this profile's alone; every other wrapper caller keeps strict-idle or its own policy.
  for (const name of ["windows-candle.yml", "memory-catalog-campaign.yml", "load-progress-acceptance.yml"]) assert.doesNotMatch(await workflowText(name), /shared-display/);
  assert.match(cuda, /Add-Content -Path \$env:GITHUB_ENV -Value 'RUSTC_WRAPPER='/);
  assert.match(cuda, /vcvars64\.bat/);
  assert.doesNotMatch(reusable, /download_missing|--download-missing|hf download|huggingface-cli/);
  assert.doesNotMatch(reusable, /secrets[.:]|HF_TOKEN:/, "no step, build or run, receives a secret");
});
