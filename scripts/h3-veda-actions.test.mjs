import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "node:net";
import { assertActionsIdentity, assertActionsTimeBudget, assertOwnedImportReadiness, emitOwnedApiToken, probeReadAuthority, probeCpuInventoryAuthority, requirePreviousJobClosed, importOwnedCorpus, redactReceipt, approvedCampaignIdentity, buildHostEvidence, aggregateHostReceipts, exportReviewMedia, cpuImportApiEnvironment, buildArguments, verifyDurableProduct, pinnedMlxSysPackageId, mlxMetallibBuildOutput, mlxBuildEnvironment } from "./h3-veda-actions.mjs";
import { digest } from "./h3-veda-acceptance.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const H = "a".repeat(40);
function m(root, backend = "mlx") {
  return { sourceFreeze: { sceneworks: { root: ROOT, head: H }, inference: { head: H }, inferencePin: H }, runners: { [backend]: { apiUrl: "http://127.0.0.1:13579", guardian: { port: 13579, dataDir: path.join(root, "data"), configDir: path.join(root, "config") } } } };
}
const options = (root, backend = "mlx") => ({ backend, runnerName: backend === "mlx" ? "nax-macos-2" : "cuda-windows-2", expectedRunner: backend === "mlx" ? "nax-macos-2" : "cuda-windows-2", sceneWorksSha: H, inferenceSha: H, manifest: m(root, backend), runDir: root, cacheRoot: path.join(root, "cache") });

test("owned import begins only from source-frozen readiness with product IDs as the sole gap", () => {
  const descriptor = { path: "/host-owned/product", bytes: 12, sha256: "b".repeat(64) };
  const runner = (physicalGpuId, mode) => ({ physicalGpuId, nativeResources: mode === "mlx" ? {schemaVersion:1,mode:"locked_cargo_source_build",sourceOutput:{packageId:`git+https://github.com/michaeltrefry/mlx-rs?rev=${H}#pmetal-mlx-sys@0.2.4`,revision:H,outDir:"/host-owned/cargo-target/release/build/pmetal-mlx-sys/out",path:"/host-owned/cargo-target/release/build/pmetal-mlx-sys/out/build/lib/mlx.metallib",bytes:12,sha256:"d".repeat(64)},mlxMetallib:{path:"/host-owned/product-resources/mlx/mlx.metallib",bytes:12,sha256:"d".repeat(64)}} : {schemaVersion:1,mode:"not_applicable",sourceOutput:null,mlxMetallib:null}, apiBinary: descriptor, workerBinary: { ...descriptor }, attestation: { path: "/host-owned/attestation.json", bytes: 42, sha256: "c".repeat(64) } });
  const ready = { sourceFreeze: { stage: "permanent-inference-main-pin", resourceContract: { mlx: { physicalGpuId: "Mac-physical" }, candle_cuda: { physicalGpuId: "GPU-physical" } } },
    runners: { mlx: runner("Mac-physical","mlx"), candle_cuda: runner("GPU-physical","candle_cuda") },
    preparation: { campaignReady: false, unresolved: ["normal owned corpus imports/readbacks required"], missingBindings: { permanentSourceFreeze: false, binaries: [], counterpartInventory: null, modelPayloadVerification: false, cacheSelection: false, activeDeviceSelection: false, productImports: true, productProofSetup: [], unknownArtifactHashes: [] } } };
  assertOwnedImportReadiness(ready);
  for (const change of [
    (v) => v.sourceFreeze.stage = "preparation-unbound",
    (v) => v.preparation.missingBindings.productImports = false,
    (v) => v.preparation.missingBindings.modelPayloadVerification = true,
    (v) => v.preparation.unresolved.push("counterpart cache unknown"),
    (v) => v.runners.candle_cuda.apiBinary = null,
    (v) => v.runners.candle_cuda.workerBinary.sha256 = "d".repeat(64),
    (v) => v.sourceFreeze.resourceContract.candle_cuda.physicalGpuId = "other GPU",
  ]) { const candidate = structuredClone(ready); change(candidate); assert.throws(() => assertOwnedImportReadiness(candidate)); }
});

test("Actions host/source/run roots refuse unknown or mismatched physical identity", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  assertActionsIdentity(options(root)); assertActionsIdentity(options(root, "candle_cuda"));
  for (const change of [
    (o) => o.runnerName = "nax-macos",
    (o) => o.expectedRunner = "nax-macos",
    (o) => o.inferenceSha = "b".repeat(40),
    (o) => o.manifest.sourceFreeze.sceneworks.root = "/untrusted/source",
    (o) => o.manifest.runners.mlx.guardian.dataDir = "/shared/data",
    (o) => o.manifest.runners.mlx.apiUrl = "http://0.0.0.0:13579",
  ]) { const o = structuredClone(options(root)); change(o); assert.throws(() => assertActionsIdentity(o)); }
  const win = options(root, "candle_cuda"); win.expectedRunner = "cuda-windows-3"; win.runnerName = "cuda-windows-3"; assert.throws(() => assertActionsIdentity(win));
});

test("owned import refuses missing private token before creating data or starting API", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  let starts = 0;
  await assert.rejects(importOwnedCorpus({ manifest: m(root), plan: { assets: {} }, planPath: "plan", corpusRoot: root, runDir: root, cacheRoot: root, backend: "mlx", token: "", spawnApi: () => { starts++; } }), /token absent/);
  assert.equal(starts, 0);
  await assert.rejects(readFile(path.join(root, "data", "s9-workspace-owner.json")), /ENOENT/);
});

test("CPU-only import API environment overrides inherited engine startup switches and credentials", () => {
  const env = cpuImportApiEnvironment({ CUDA_VISIBLE_DEVICES: "0", NVIDIA_VISIBLE_DEVICES: "0", SCENEWORKS_GPU_CHECK: "1", SCENEWORKS_WORKER_ONLY: "1", SCENEWORKS_RUN_UTILITY_INPROCESS: "1", SCENEWORKS_TOKEN: "ambient", GH_TOKEN: "ambient" }, { dataDir: "/run/data", configDir: "/run/config", jobsDbPath: "/run/data/jobs.db", cacheRoot: "/cache", apiUrl: "http://127.0.0.1:12345", token: "owned", backend: "mlx", nativeResources: { mlxMetallib: { path: "/run/product-resources/mlx/mlx.metallib", sha256: "f".repeat(64) } } });
  assert.deepEqual([env.SCENEWORKS_GPU_CHECK, env.SCENEWORKS_WORKER_ONLY, env.SCENEWORKS_RUN_UTILITY_INPROCESS, env.SCENEWORKS_GPU_ID, env.SCENEWORKS_RUST_WORKER_GPU_ID], ["0", "0", "0", "cpu", "cpu"]);
  assert.equal(env.PMETAL_METALLIB_PATH, "/run/product-resources/mlx/mlx.metallib"); assert.equal(Object.hasOwn(env, "PMETAL_MLX_PREBUILT_DIR"), false);
  for (const key of ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SCENEWORKS_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]) assert.equal(Object.hasOwn(env, key), false, `${key} must not reach the import API`);
  assert.equal(env.SCENEWORKS_ACCESS_TOKEN, "owned");
});

test("durable host binary verification binds both API and worker to one source-frozen executable", async (t) => {
  const runDir = await mkdtemp(path.join(tmpdir(), "s9-product-build-")); t.after(() => rm(runDir, { recursive: true, force: true }));
  const backend = "mlx", freeze = { inferencePin: H, sceneworks: { head: H } }, binaryPath = path.join(runDir, "product-binaries", backend, process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api");
  await mkdir(path.dirname(binaryPath), { recursive: true }); await writeFile(binaryPath, "source-built product fixture");
  const binary = await readFile(binaryPath), binarySha256 = await (await import("./h3-veda-acceptance.mjs")).hashFile(binaryPath);
  const targetDir=path.join(runDir,"cargo-target"), {packageId,revision}=pinnedMlxSysPackageId(await readFile(path.join(ROOT,"Cargo.lock"),"utf8"));
  const sourcePath=path.join(targetDir,"release","build","pmetal-mlx-sys-deadbeef","out","build","lib","mlx.metallib"), resourcePath=path.join(runDir,"product-resources","mlx","mlx.metallib");
  await mkdir(path.dirname(sourcePath),{recursive:true}); await mkdir(path.dirname(resourcePath),{recursive:true}); await writeFile(sourcePath,"verified fixture metallib"); await writeFile(resourcePath,"verified fixture metallib");
  const resourceBytes=await readFile(sourcePath), resourceSha=await (await import("./h3-veda-acceptance.mjs")).hashFile(sourcePath);
  const nativeResources={schemaVersion:1,mode:"locked_cargo_source_build",sourceOutput:{packageId,revision,outDir:path.dirname(path.dirname(path.dirname(sourcePath))),path:sourcePath,bytes:resourceBytes.length,sha256:resourceSha},mlxMetallib:{path:resourcePath,bytes:resourceBytes.length,sha256:resourceSha}};
  const identity = { backend, sourceFreezeSha256: digest(freeze), binary: { sha256: binarySha256, bytes: binary.length }, nativeResources, environment:{PMETAL_METALLIB_PATH:null,PMETAL_MLX_PREBUILT_DIR:null} };
  const attestation = { schemaVersion: 3, method: "actions_source_build_durable_same_binary_api_worker_and_native_resources", backend, inferencePin: H, sceneworksHead: H, sourceFreezeSha256: digest(freeze), apiBinarySha256: binarySha256, workerBinarySha256: binarySha256, binaryBytes: binary.length, workerId: "mlx-worker", nativeResources, buildIdentity: identity, buildIdentitySha256: digest(identity) };
  const attestationPath = path.join(runDir, `${backend}-product-binary-attestation.json`); await writeFile(attestationPath, JSON.stringify(attestation));
  const descriptor = { path: binaryPath, bytes: binary.length, sha256: binarySha256 }, m = { sourceFreeze: freeze, runners: { [backend]: { apiBinary: descriptor, workerBinary: { ...descriptor }, workerId: "mlx-worker", nativeResources, attestation: { path: attestationPath, bytes: (await readFile(attestationPath)).length, sha256: await (await import("./h3-veda-acceptance.mjs")).hashFile(attestationPath) } } } };
  assert.equal((await verifyDurableProduct(m, backend, runDir, { sceneWorksSha: H, inferenceSha: H })).binary.sha256, binarySha256);
  await writeFile(resourcePath,"altered shader"); await assert.rejects(verifyDurableProduct(m, backend, runDir, {sceneWorksSha:H,inferenceSha:H}),/MLX metallib missing or changed/); await writeFile(resourcePath,"verified fixture metallib");
  await rm(resourcePath); await assert.rejects(verifyDurableProduct(m, backend, runDir, {sceneWorksSha:H,inferenceSha:H}),/MLX metallib missing or changed/); await writeFile(resourcePath,"verified fixture metallib");
  m.runners[backend].nativeResources.mlxMetallib.sha256="0".repeat(64); await assert.rejects(verifyDurableProduct(m, backend, runDir, {sceneWorksSha:H,inferenceSha:H}),/durable native resource descriptors differ/); m.runners[backend].nativeResources.mlxMetallib.sha256=resourceSha;
  m.runners[backend].workerBinary = { ...descriptor, path: path.join(ROOT, "target", "release", "sceneworks-rust-api") };
  await assert.rejects(verifyDurableProduct(m, backend, runDir, { sceneWorksSha: H, inferenceSha: H }), /same durable run-owned binary/);
  assert.deepEqual(buildArguments("mlx"), ["build", "--release", "--locked", "--message-format=json-render-diagnostics", "-p", "sceneworks-rust-api"]);
  assert.deepEqual(buildArguments("candle_cuda"), ["build", "--release", "--locked", "--message-format=json-render-diagnostics", "-p", "sceneworks-rust-api", "--features", "backend-candle"]);
  assert.throws(() => buildArguments("unknown"), /explicit backend/);
});


test("MLX resource discovery binds one exact locked Cargo build-script output and clears inherited overrides", async()=>{
  const lock=await readFile(path.join(ROOT,"Cargo.lock"),"utf8"), pin=pinnedMlxSysPackageId(lock), target="/owned/run/cargo-target", out=`${target}/release/build/pmetal-mlx-sys-deadbeef/out`;
  const line=JSON.stringify({reason:"build-script-executed",package_id:pin.packageId,out_dir:out});
  assert.equal(mlxMetallibBuildOutput(`${line}\n`,pin.packageId,target).path,`${out}/build/lib/mlx.metallib`);
  assert.throws(()=>mlxMetallibBuildOutput("",pin.packageId,target),/found 0/);
  assert.throws(()=>mlxMetallibBuildOutput(`${line}\n${line}\n`,pin.packageId,target),/found 2/);
  assert.throws(()=>mlxMetallibBuildOutput(JSON.stringify({reason:"build-script-executed",package_id:pin.packageId,out_dir:"/other/out"}),pin.packageId,target),/inside the durable Cargo target/);
  const env=mlxBuildEnvironment({PMETAL_METALLIB_PATH:"/wrong",PMETAL_MLX_PREBUILT_DIR:"/wrong/prebuilt",PATH:"/bin"}); assert.equal(Object.hasOwn(env,"PMETAL_METALLIB_PATH"),false);assert.equal(Object.hasOwn(env,"PMETAL_MLX_PREBUILT_DIR"),false);assert.equal(env.PATH,"/bin");
});
test("job-owned API token is random-length hex, masked before job env write, and never returned", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-token-")); t.after(() => rm(root, { recursive: true, force: true }));
  const prior = process.env.GITHUB_ACTIONS; process.env.GITHUB_ACTIONS = "true";
  try {
    const output = [], envFile = path.join(root, "job-env");
    const result = await emitOwnedApiToken({ envFile, stdout: { write: (value) => { output.push(value); } }, random: (n) => { assert.equal(n, 32); return Buffer.alloc(n, 0xab); } });
    assert.equal(result, undefined);
    assert.deepEqual(output, [`::add-mask::${"ab".repeat(32)}\n`]);
    assert.equal(await readFile(envFile, "utf8"), `SCENEWORKS_TOKEN=${"ab".repeat(32)}\n`);
    await assert.rejects(emitOwnedApiToken({ envFile: "relative", random: () => Buffer.alloc(32) }), /owned Actions environment file/);
  } finally { if (prior === undefined) delete process.env.GITHUB_ACTIONS; else process.env.GITHUB_ACTIONS = prior; }
});

test("existing credential probe requires exact live job, physical runner and cross-repository queue readbacks", () => {
  const before = Object.fromEntries(["GITHUB_ACTIONS", "GH_TOKEN", "GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_SHA", "GITHUB_WORKFLOW_REF", "GITHUB_JOB", "RUNNER_NAME"].map((key) => [key, process.env[key]]));
  Object.assign(process.env, { GITHUB_ACTIONS: "true", GH_TOKEN: "not-logged-test-token", GITHUB_REPOSITORY: "SceneWorks/SceneWorks", GITHUB_EVENT_NAME: "workflow_dispatch", GITHUB_SHA: H, GITHUB_WORKFLOW_REF: `SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@${H}`, GITHUB_JOB: "mac2", RUNNER_NAME: "nax-macos-2" });
  const responses = (endpoint) => {
    if (endpoint.endsWith("/actions/runs/123")) return { id: 123, run_attempt: 1, head_sha: H, status: "in_progress", conclusion: null, event: "workflow_dispatch", name: "H3 Veda terminal owned campaign" };
    if (endpoint.includes("/actions/runs/123/jobs")) return [{ jobs: [{ id: 456, run_id: 123, run_attempt: 1, head_sha: H, name: "mac2", runner_name: "nax-macos-2", runner_id: 5281, status: "in_progress", conclusion: null }] }];
    if (endpoint.startsWith("orgs/")) return [{ runners: [{ id: 5281, name: "nax-macos-2", status: "online", busy: true }] }];
    if (endpoint.startsWith("repos/michaeltrefry/LensWorks/")) return [{ workflow_runs: [] }];
    throw Error("unexpected endpoint");
  };
  const seen = [], exec = (_bin, args) => { const endpoint = args.at(-1); seen.push(endpoint); return JSON.stringify(responses(endpoint)); };
  try {
    const full = probeReadAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec });
    assert.equal(full.runnerId, 5281); assert.equal(full.resourceQueueVerified, true); assert.equal(full.resourceScope, "cross_repository_runner_and_queue_read");
    assert.equal(seen.length, 5);
    assert.throws(() => probeReadAuthority({ backend: "mlx", runnerName: "nax-macos", sceneWorksSha: H, runId: "123", attempt: "1", exec }), /workflow\/source\/job\/runner differs/);
    const failed = (_bin, args) => { if (args.at(-1).startsWith("orgs/")) throw Error("credential value must not surface"); return JSON.stringify(responses(args.at(-1))); };
    assert.throws(() => probeReadAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: failed }), (error) => error.message.includes("cannot read orgs/SceneWorks/actions/runners") && !error.message.includes("credential value"));
    const wrong = (_bin, args) => { const result = responses(args.at(-1)); if (args.at(-1).includes("/jobs")) result[0].jobs[0].runner_id = 999; return JSON.stringify(result); };
    assert.throws(() => probeReadAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: wrong }), /runner readback missing/);
  } finally { for (const [key, value] of Object.entries(before)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; } }
});

test("CPU inventory authority binds only the exact same-repository run, job and physical listener", () => {
  const names = ["GITHUB_ACTIONS", "GH_TOKEN", "GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_SHA", "GITHUB_WORKFLOW_REF", "GITHUB_JOB", "RUNNER_NAME"];
  const before = Object.fromEntries(names.map((key) => [key, process.env[key]]));
  Object.assign(process.env, { GITHUB_ACTIONS: "true", GH_TOKEN: "not-logged-test-token", GITHUB_REPOSITORY: "SceneWorks/SceneWorks", GITHUB_EVENT_NAME: "workflow_dispatch", GITHUB_SHA: H, GITHUB_WORKFLOW_REF: `SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@${H}`, GITHUB_JOB: "mac2", RUNNER_NAME: "nax-macos-2" });
  const run = { id: 123, run_attempt: 1, head_sha: H, status: "in_progress", conclusion: null, event: "workflow_dispatch", name: "H3 Veda terminal owned campaign", path: ".github/workflows/h3-veda-terminal.yml" };
  const job = { id: 456, run_id: 123, run_attempt: 1, head_sha: H, name: "mac2", runner_name: "nax-macos-2", runner_id: 5281, status: "in_progress", conclusion: null };
  const probe = (alter = () => {}) => {
    const [r, j] = [structuredClone(run), structuredClone(job)]; alter(r, j);
    const seen = [], exec = (_bin, args) => { const endpoint = args.at(-1); seen.push(args); if (endpoint === "repos/SceneWorks/SceneWorks/actions/runs/123") return JSON.stringify(r); if (endpoint === "repos/SceneWorks/SceneWorks/actions/runs/123/jobs?per_page=100") return JSON.stringify([{ jobs: [j] }]); throw Error("cross-repository endpoint forbidden"); };
    const result = probeCpuInventoryAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec });
    return { result, seen };
  };
  try {
    const { result, seen } = probe();
    assert.deepEqual(seen.map((args) => args.at(-1)), ["repos/SceneWorks/SceneWorks/actions/runs/123", "repos/SceneWorks/SceneWorks/actions/runs/123/jobs?per_page=100"]);
    assert.deepEqual(seen[1].slice(0, 3), ["api", "--paginate", "--slurp"]);
    assert.equal(result.runnerId, 5281); assert.equal(result.ownRunReadable, true); assert.equal(result.ownRunnerIdentityVerified, true);
    assert.equal(result.resourceQueueVerified, false); assert.equal(result.resourceScope, "same_repository_actions_identity_only"); assert.equal(Object.hasOwn(result, "readable"), false);
    const badPages = (pages) => (_bin, args) => JSON.stringify(args.at(-1).includes("/jobs?") ? pages : run);
    assert.throws(() => probeCpuInventoryAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: badPages({ jobs: [job] }) }), /paginated job readback malformed/);
    assert.throws(() => probeCpuInventoryAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: badPages([{ jobs: [job] }, { jobs: [job] }]) }), /identity missing or ambiguous/);
    for (const mutate of [
      (r) => r.run_attempt = 2, (r) => r.head_sha = "b".repeat(40), (r) => r.path = ".github/workflows/other.yml", (r) => r.event = "push", (r) => r.status = "completed", (r) => r.name = "Other workflow",
      (_r, j) => j.runner_id = 999, (_r, j) => j.runner_name = "foreign", (_r, j) => j.name = "windows_cuda", (_r, j) => j.run_attempt = 2, (_r, j) => j.head_sha = "b".repeat(40),
    ]) assert.throws(() => probe(mutate), /readback differs|identity missing/);
    assert.throws(() => probeCpuInventoryAuthority({ backend: "candle_cuda", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: () => "{}" }), /frozen physical listener/);
    Object.assign(process.env, { GITHUB_JOB: "windows_cuda", RUNNER_NAME: "cuda-windows-2" });
    const windowsExec = (_bin, args) => JSON.stringify(args.at(-1).includes("/jobs?") ? [{ jobs: [{ ...job, name: "windows_cuda", runner_name: "cuda-windows-2", runner_id: 2619 }] }] : run);
    assert.equal(probeCpuInventoryAuthority({ backend: "candle_cuda", runnerName: "cuda-windows-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: windowsExec }).runnerId, 2619);
    Object.assign(process.env, { GITHUB_JOB: "mac2", RUNNER_NAME: "nax-macos-2" });
    process.env.GITHUB_EVENT_NAME = "push"; assert.throws(() => probe(), /repository\/event differs/); process.env.GITHUB_EVENT_NAME = "workflow_dispatch";
    process.env.GH_TOKEN = ""; assert.throws(() => probe(), /credential unavailable/);
  } finally { for (const [key, value] of Object.entries(before)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; } }
});

test("new job token refuses retained guardian before fresh physical admission", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-token-")); t.after(() => rm(root, { recursive: true, force: true }));
  await requirePreviousJobClosed(root);
  await mkdir(path.join(root, "physical-guardian"));
  await writeFile(path.join(root, "physical-guardian/control.json"), "{}");
  await assert.rejects(requirePreviousJobClosed(root), /previous job API token/);
  await writeFile(path.join(root, "physical-guardian/closed.json"), "{}");
  await requirePreviousJobClosed(root);
});

test("owned import refuses a foreign listener before creating data or spawning API", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  const server = createServer(); await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve)); t.after(() => server.close());
  const manifest = m(root); manifest.runners.mlx.guardian.port = server.address().port;
  let starts = 0;
  await assert.rejects(importOwnedCorpus({ manifest, manifestPath: path.join(root, "manifest.json"), plan: { assets: {} }, planPath: "plan", corpusRoot: root, runDir: root, cacheRoot: root, backend: "mlx", token: "owned-token", spawnApi: () => { starts++; } }), /port already claimed/);
  assert.equal(starts, 0);
  await assert.rejects(readFile(path.join(root, "data", "s9-workspace-owner.json")), /ENOENT/);
});

test("live Actions queue and owned-child deadlines are bounded by exact run/job identity", () => {
  const start = Date.parse("2026-10-13T20:00:00Z"), run = { id: 123, run_attempt: 1, head_sha: H, status: "in_progress", event: "workflow_dispatch", created_at: new Date(start).toISOString() };
  const job = { run_id: 123, run_attempt: 1, head_sha: H, name: "mac2", runner_name: "nax-macos-2", status: "in_progress", started_at: new Date(start + 29 * 60_000).toISOString() };
  const options = { now: start + 38 * 60_000, runId: "123", attempt: "1", backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H };
  assertActionsTimeBudget(run, job, options);
  assert.throws(() => assertActionsTimeBudget(run, { ...job, started_at: new Date(start + 31 * 60_000).toISOString() }, options), /30 minutes/);
  assert.throws(() => assertActionsTimeBudget(run, job, { ...options, now: start + 40 * 60_000 }), /insufficient owned-child/);
  assert.throws(() => assertActionsTimeBudget(run, { ...job, runner_name: "nax-macos" }, options), /job admission identity/);
  assert.throws(() => assertActionsTimeBudget({ ...run, status: "completed" }, job, options), /workflow admission identity/);
});

test("artifact transport is allowlisted summary and hashed closure, never raw media, token, or request", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "physical-guardian"));
  await writeFile(path.join(root, "campaign.json"), JSON.stringify({ campaignId: "owned", fingerprint: "f", namedCells: 5400, qualityAcceptance: "pending Michael", completedAcceptance: false, secret: "PRIVATE_TOKEN", cells: { a: { status: "unresolved", request: "PRIVATE_PROMPT" } } }));
  await writeFile(path.join(root, "physical-guardian/closed.json"), JSON.stringify({ closed: true, servicesDrained: true, physicalLockReleased: true, ownerTokenSha256: "h", fingerprint: "f", pid: 123, token: "PRIVATE_TOKEN" }));
  const output = path.join(root, "upload", "receipt.json"); await redactReceipt({ runDir: root, output });
  const text = await readFile(output, "utf8"), receipt = JSON.parse(text);
  assert.equal(receipt.summary.cellStatusCounts.unresolved, 1);
  assert.equal(receipt.guardianClosure.physicalLockReleased, true);
  assert.equal(receipt.mediaUploaded, false);
  assert.doesNotMatch(text, /PRIVATE_TOKEN|PRIVATE_PROMPT|request/);
});

test("closed owned review exports exact decoded comparison and product clips with safe source-bound labels", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-review-")); t.after(() => rm(root, { recursive: true, force: true }));
  const runDir = path.join(root, "run"), outputDir = path.join(root, "upload"), id = "mlx/q4/t2va/base_50/768x768/124/none";
  const m0 = m(runDir); m0.campaignId = "epic-25140-s9"; m0.runners.mlx.leasePath = path.join(root, "lease"); m0.limits = { maxMediaBytes: 1000 };
  const cell = { kind: "media", status: "media_collected_evidence_incomplete", arms: {}, comparison: {}, request: "PRIVATE_PROMPT", token: "PRIVATE_TOKEN" };
  const bind = async (file, bytes) => { await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, bytes); return { path: file, bytes: Buffer.byteLength(bytes), sha256: digest(bytes) }; };
  for (const arm of ["dense", "sparse"]) { const key = digest(arm); cell.arms[arm] = { key, status: "collected", media: await bind(path.join(runDir, key, `${arm}.mp4`), `${arm}-actual-product`) }; }
  const compare = path.join(runDir, digest(id));
  cell.comparison.sideBySide = await bind(path.join(compare, "side-by-side.mp4"), "actual-decoded-comparison");
  cell.comparison.denseAudio = await bind(path.join(compare, "dense.wav"), "actual-dense-audio");
  cell.comparison.sparseAudio = await bind(path.join(compare, "sparse.wav"), "actual-sparse-audio");
  const campaign = { campaignId: m0.campaignId, fingerprint: digest(m0), cells: { [id]: cell } };
  await mkdir(path.join(runDir, "physical-guardian"));
  await writeFile(path.join(runDir, "manifest.bound.json"), JSON.stringify(m0));
  await writeFile(path.join(runDir, "campaign.json"), JSON.stringify(campaign));
  const ownerToken = "a".repeat(48), deadPid = 999999999;
  await writeFile(path.join(runDir, "physical-guardian/control.json"), JSON.stringify({ token: ownerToken, fingerprint: digest(m0), pid: deadPid }));
  await writeFile(path.join(runDir, "physical-guardian/closed.json"), JSON.stringify({ closed: true, servicesDrained: true, physicalLockReleased: true, fingerprint: digest(m0), ownerTokenSha256: digest(ownerToken), pid: deadPid }));
  const rows = [{ id, backend: "mlx", kind: "media", profile: "t2va", model: "minimax_h3", tier: "q4", recipe: "base_50", canvas: "768x768", frames: 124, duration: 5.1667 }];
  const args = { runDir, backend: "mlx", caseIds: [id], sourceCheck: async () => {}, rows };
  const result = await exportReviewMedia({ ...args, outputDir });
  assert.equal(result.files.length, 5); assert.equal(result.selectedCases[0].recipe, "base_50");
  assert.equal(result.completedAcceptance, false); assert.equal(result.qualityAcceptance, "pending Michael");
  assert.equal(await readFile(path.join(outputDir, "case-01/side-by-side.mp4"), "utf8"), "actual-decoded-comparison");
  assert.doesNotMatch(await readFile(path.join(outputDir, "review-media.json"), "utf8"), /PRIVATE_PROMPT|PRIVATE_TOKEN|request/);
  await assert.rejects(exportReviewMedia({ ...args, outputDir: path.join(root, "duplicate"), caseIds: [id, id] }), /unique bounded review/);
  await assert.rejects(exportReviewMedia({ ...args, outputDir: path.join(root, "unknown"), caseIds: ["foreign-case"] }), /outside declared owning-host/);
  await writeFile(m0.runners.mlx.leasePath, "held");
  await assert.rejects(exportReviewMedia({ ...args, outputDir: path.join(root, "leased") }), /lease still held/);
  await rm(m0.runners.mlx.leasePath);
  const closurePath = path.join(runDir, "physical-guardian/closed.json"), validClosure = await readFile(closurePath, "utf8");
  await writeFile(closurePath, JSON.stringify({ ...JSON.parse(validClosure), ownerTokenSha256: "b".repeat(64) }));
  await assert.rejects(exportReviewMedia({ ...args, outputDir: path.join(root, "wrong-owner") }), /control\/closure identity differs/);
  await writeFile(closurePath, validClosure);
  const active = structuredClone(campaign); active.cells[id].arms.dense.status = "ambiguous";
  await writeFile(path.join(runDir, "campaign.json"), JSON.stringify(active));
  await assert.rejects(exportReviewMedia({ ...args, outputDir: path.join(root, "ambiguous") }), /unresolved live or ambiguous/);
  await writeFile(path.join(runDir, "campaign.json"), JSON.stringify(campaign));
  await writeFile(cell.arms.dense.media.path, "tampered-product");
  await assert.rejects(exportReviewMedia({ ...args, outputDir: path.join(root, "tampered") }), /file size\/type mismatch|file digest mismatch/);
  await writeFile(cell.arms.dense.media.path, "dense-actual-product");
  await symlink(path.join(root, "outside.mp4"), path.join(root, "link.mp4"));
  cell.arms.dense.media.path = path.join(root, "link.mp4");
  await writeFile(path.join(runDir, "campaign.json"), JSON.stringify(campaign));
  await assert.rejects(exportReviewMedia({ ...args, outputDir: path.join(root, "escaped") }), /media binding incomplete/);
});

test("cross-host aggregate joins approved scope, exact row dispositions and distinct local imports without inventing acceptance", () => {
  const hex = "a".repeat(64), other = "b".repeat(64);
  const rows = ["mlx", "candle_cuda"].flatMap((backend) => [
    ...Array.from({ length: 2700 }, (_, i) => ({ id: `${backend}/named/${i}`, backend, kind: "media", style: null })),
    { id: `${backend}/style/one`, backend, kind: "media", style: "one" },
    { id: `boundary/${backend}/cap`, backend, kind: "expected_refusal", boundary: "cap", style: null },
    { id: `proof/route/one/${backend}`, backend, kind: "product_proof", style: null },
  ]);
  const manifest = (backend) => {
    const x = m(`/owned/${backend}`, backend);
    x.campaignId = "epic-25140-s9"; x.scope = "named-required"; x.projectId = `project_${backend}`;
    x.sourceFreeze.files = { "config/manifests/builtin.models.jsonc": hex, "config/manifests/builtin.loras.jsonc": hex, "config/terminal-evidence/epic-25140-corpus.plan.json": hex };
    x.sourceFreeze.artifactInventorySha256 = hex;
    x.corpusPlan = { path: `/host/${backend}/plan`, sha256: hex };
    x.corpusImportReceipt = { path: `/host/${backend}/import`, sha256: other };
    x.artifactInventory = { path: `/host/${backend}/inventory`, sha256: hex, bytes: 1 };
    x.artifacts = [{ id: "model", path: `/host/${backend}/weights`, sha256: hex, bytes: 1, coverage: [{ backend: "mlx" }, { backend: "candle_cuda" }] }];
    x.turboArtifacts = { turbo: { path: `/host/${backend}/turbo`, sha256: hex, bytes: 1, modelIds: ["minimax_h3"] } };
    x.styles = [{ id: "one", path: `/host/${backend}/style`, sha256: hex, bytes: 1, compatibilityEvidence: { path: `/host/${backend}/compat`, sha256: hex, bytes: 1 } }];
    x.prompts = { t2va: "same prompt" }; x.bindings = { t2va: { seed: 7 } }; x.boundaryBindings = { cap: { images: ["one"] } }; x.proofRoutes = { route: { route: "product_job" } }; x.productProofCases = [];
    Object.assign(x.runners[backend], { physicalGpuId: `physical-${backend}`, workerId: `worker-${backend}`, apiBinary: { sha256: backend === "mlx" ? hex : other }, workerBinary: { sha256: backend === "mlx" ? hex : other }, attestation: { sha256: hex }, physicalGuardAdapter: { sha256: hex } });
    return x;
  };
  const receipt = (backend) => {
    const x = manifest(backend), cells = Object.fromEntries(rows.filter((r) => r.backend === backend).map((r) => [r.id, { kind: r.kind, status: r.kind === "expected_refusal" ? "expected_refusal_observed" : r.kind === "product_proof" ? "product_proof_observed" : "media_collected_evidence_incomplete", privateRequest: "must never be uploaded" }]));
    const state = { campaignId: x.campaignId, fingerprint: digest(x), cells };
    return { hostEvidence: buildHostEvidence(x, state, backend, rows, { closed: true, servicesDrained: true, physicalLockReleased: true, fingerprint: digest(x) }, { manifestSha256: hex, campaignSha256: other }) };
  };
  const mlx = receipt("mlx"), cuda = receipt("candle_cuda");
  assert.equal(mlx.hostEvidence.approvedIdentitySha256, cuda.hostEvidence.approvedIdentitySha256);
  assert.notEqual(mlx.hostEvidence.local.manifestFingerprint, cuda.hostEvidence.local.manifestFingerprint);
  assert.notEqual(mlx.hostEvidence.local.binarySha256, cuda.hostEvidence.local.binarySha256);
  assert.doesNotMatch(JSON.stringify(mlx), /must never be uploaded/);
  const joined = aggregateHostReceipts(mlx, cuda);
  assert.equal(joined.expectedRows, rows.length);
  assert.equal(joined.observedDispositions, rows.length);
  assert.equal(joined.missing.length, 0);
  assert.equal(joined.unresolved.length, 5402, "real media still lacks native proof and Michael review");
  assert.equal(joined.completedAcceptance, false);
  const changed = structuredClone(cuda); changed.hostEvidence.approvedIdentitySha256 = other;
  assert.throws(() => aggregateHostReceipts(mlx, changed), /approved campaign\/recipe\/source\/catalog/);
  const dropped = structuredClone(cuda); dropped.hostEvidence.cases.pop();
  assert.throws(() => aggregateHostReceipts(mlx, dropped), /row identity coverage missing/);
  const unresolved = structuredClone(cuda); unresolved.hostEvidence.cases[0].status = "missing"; unresolved.hostEvidence.cases[0].evidenceSha256 = null;
  assert.equal(aggregateHostReceipts(mlx, unresolved).missing.length, 1);
  const wrongDisposition = structuredClone(cuda); wrongDisposition.hostEvidence.cases[0].status = "expected_refusal_observed";
  assert.throws(() => aggregateHostReceipts(mlx, wrongDisposition), /disposition differs/);
  const foreign = structuredClone(mlx); foreign.hostEvidence.cases[0].id = "candle_cuda/foreign";
  assert.throws(() => aggregateHostReceipts(foreign, cuda), /row identity coverage missing\/duplicate/);
  const sourceMismatch = structuredClone(cuda); sourceMismatch.hostEvidence.source.catalogModelsSha256 = other;
  assert.throws(() => aggregateHostReceipts(mlx, sourceMismatch), /approved campaign\/recipe\/source\/catalog/);
  assert.equal(approvedCampaignIdentity(manifest("mlx")), approvedCampaignIdentity(manifest("candle_cuda")));
});

test("workflow binds explicit runner, source, owned import and guardian, with no blanket artifact upload", async () => {
  const workflow = await readFile(path.join(ROOT, ".github/workflows/h3-veda-terminal.yml"), "utf8");
  assert.match(workflow, /runs-on: \[self-hosted, macOS, ARM64, rw-mage\]/);
  assert.deepEqual([...workflow.slice(workflow.indexOf("    inputs:"), workflow.indexOf("\npermissions:")).matchAll(/^      ([a-z_]+):$/gm)].map((m) => m[1]),
    ["backend", "mode", "review_case_ids", "sceneworks_sha", "inference_main_sha", "expected_runner", "manifest_path", "corpus_root", "offline_cache_root", "run_dir"]);
  assert.match(workflow, /options: \[inventory, stage, build, prepare, resume, review\]/);
  assert.match(workflow, /review_case_ids:/);
  assert.match(workflow, /h3-veda-actions\.mjs review/);
  assert.match(workflow, /h3-veda-review-media-(?:mlx|cuda)-/);
  for (const step of ["Prepare or verify owned private corpus"]) {
    const escaped = step.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    assert.match(workflow, new RegExp(`- name: ${escaped}\\n\\s+if: inputs.mode == 'prepare' \\|\\| inputs.mode == 'resume'`), `inventory/stage/build/review modes must skip ${step}`);
  }
  assert.match(workflow, /- name: Build durable normal product API and worker from frozen source\n\s+if: inputs\.mode == 'build'/);
  assert.match(workflow, /- name: Build durable normal Candle\/CUDA product API and worker from frozen source\n\s+if: inputs\.mode == 'build'/);
  assert.match(workflow, /- name: Run bounded guardian-owned native campaign\n\s+if: inputs\.mode == 'resume'/);
  assert.match(workflow, /CARGO_TARGET_DIR: \$\{\{ inputs\.run_dir \}\}\/cargo-target/);
  assert.match(workflow, /h3-veda-product-build-(?:mlx|cuda)-/);
  assert.match(workflow, /actions\/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97/);
  assert.match(workflow, /python-version: "3\.12"/);
  assert.match(workflow, /S9_GUARD_PYTHON=/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.match(workflow, /test "\$RUNNER_NAME" = 'nax-macos-2'/);
  assert.match(workflow, /S9_EXPECTED_RUNNER: \$\{\{ inputs.expected_runner \}\}/);
  assert.match(workflow, /h3-veda-actions\.mjs prepare/);
  assert.match(workflow, /vcvars64\.bat/);
  assert.match(workflow, /set NVCC_CCBIN=%VCToolsInstallDir%bin\\Hostx64\\x64/);
  assert.equal((workflow.match(/vcvars64\.bat/g) ?? []).length, 3, "Windows build, owned import, and GPU guardian must inherit VS/NVCC setup");
  assert.match(workflow, /h3-veda-acceptance\.mjs run/);
  assert.match(workflow, /h3-veda-actions\.mjs redact/);
  assert.equal((workflow.match(/h3-veda-actions\.mjs probe-auth/g) ?? []).length, 2);
  assert.equal((workflow.match(/h3-veda-actions\.mjs probe-inventory-auth/g) ?? []).length, 2);
  assert.match(workflow, /permissions:\n  contents: read\n  actions: read/);
  assert.equal((workflow.match(/- name: Verify configured cross-repository read credential\n\s+if: inputs\.mode != 'inventory'/g) ?? []).length, 2);
  assert.equal((workflow.match(/- name: Verify existing read credential against live run, runner, and LensWorks queue\n\s+if: inputs\.mode != 'inventory'/g) ?? []).length, 2);
  assert.equal((workflow.match(/- name: Verify CPU inventory same-repository Actions identity\n\s+if: inputs\.mode == 'inventory'\n\s+env:\n\s+GH_TOKEN: \$\{\{ github\.token \}\}/g) ?? []).length, 2);
  assert.equal((workflow.match(/token: \$\{\{ inputs\.mode == 'inventory' && github\.token \|\| secrets\.INFERENCE_READ_TOKEN \}\}/g) ?? []).length, 2);
  assert.equal((workflow.match(/GH_TOKEN: \$\{\{ inputs\.mode == 'inventory' && github\.token \|\| secrets\.INFERENCE_READ_TOKEN \}\}/g) ?? []).length, 2);
  assert.equal((workflow.match(/h3-veda-actions\.mjs owned-token/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /SCENEWORKS_S9_TOKEN|S9_RUNNER_READ_TOKEN/);
  assert.doesNotMatch(workflow, /path:.*\*\*/);
});
