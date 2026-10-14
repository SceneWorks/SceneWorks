#!/usr/bin/env node
// Owned, fail-closed Actions entrypoint. GPU execution remains exclusively in
// h3-veda-acceptance.mjs and its physical guardian.
import { spawn, execFileSync } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { createServer } from "node:net";
import { constants as fsConstants } from "node:fs";
import { readFile, writeFile, appendFile, mkdir, lstat, copyFile, realpath, rename, unlink } from "node:fs/promises";
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
const productName = () => process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api";
const productPath = (runDir, backend) => path.join(runDir, "product-binaries", backend, productName());
const productAttestationPath = (runDir, backend) => path.join(runDir, `${backend}-product-binary-attestation.json`);
const mlxMetallibPath = (runDir) => path.join(runDir, "product-resources", "mlx", "mlx.metallib");
const MLX_SYS_SOURCE = "https://github.com/michaeltrefry/mlx-rs";

function targetCargoDir(runDir, cargoTargetDir = process.env.CARGO_TARGET_DIR) {
  check(absolute(cargoTargetDir) && path.resolve(cargoTargetDir) === path.join(path.resolve(runDir), "cargo-target") && !inside(ROOT, cargoTargetDir), "durable Cargo target must be the external owned run directory cargo-target");
  return cargoTargetDir;
}

export function cpuImportApiEnvironment(base, { dataDir, configDir, jobsDbPath, cacheRoot, apiUrl, token, backend, nativeResources } = {}) {
  const env = {
    ...base,
    SCENEWORKS_DATA_DIR: dataDir,
    SCENEWORKS_JOBS_DB_PATH: jobsDbPath,
    SCENEWORKS_CONFIG_DIR: configDir,
    SCENEWORKS_OWN_MANIFESTS: "1",
    SCENEWORKS_API_HOST: "127.0.0.1",
    SCENEWORKS_API_PORT: new URL(apiUrl).port,
    SCENEWORKS_API_URL: apiUrl,
    SCENEWORKS_ACCESS_TOKEN: token,
    SCENEWORKS_GPU_ID: "cpu",
    SCENEWORKS_RUST_WORKER_GPU_ID: "cpu",
    SCENEWORKS_RUN_UTILITY_INPROCESS: "0",
    SCENEWORKS_GPU_CHECK: "0",
    SCENEWORKS_WORKER_ONLY: "0",
    HF_HOME: cacheRoot,
    HUGGINGFACE_HUB_CACHE: path.join(cacheRoot, "hub"),
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
  };
  delete env.CUDA_VISIBLE_DEVICES;
  delete env.NVIDIA_VISIBLE_DEVICES;
  delete env.GH_TOKEN;
  delete env.GITHUB_TOKEN;
  delete env.SCENEWORKS_TOKEN;
  delete env.PMETAL_METALLIB_PATH;
  delete env.PMETAL_MLX_PREBUILT_DIR;
  if (backend === "mlx") { check(absolute(nativeResources?.mlxMetallib?.path) && HEX.test(nativeResources.mlxMetallib.sha256), "MLX runtime metallib descriptor missing"); env.PMETAL_METALLIB_PATH = nativeResources.mlxMetallib.path; }
  return env;
}

export function pinnedMlxSysPackageId(cargoLock) {
  const blocks = cargoLock.split(/(?=^\[\[package\]\]$)/m).filter((block) => /^name = "pmetal-mlx-sys"$/m.test(block));
  check(blocks.length === 1, "Cargo.lock must contain exactly one pmetal-mlx-sys package");
  const version = blocks[0].match(/^version = "([^"]+)"$/m)?.[1];
  const source = blocks[0].match(/^source = "git\+https:\/\/github\.com\/michaeltrefry\/mlx-rs\?rev=([a-f0-9]{40})#\1"$/m)?.[1];
  check(version && source, "Cargo.lock MLX sys source/version is not exact and pinned");
  return { revision: source, packageId: `git+${MLX_SYS_SOURCE}?rev=${source}#pmetal-mlx-sys@${version}` };
}

export function mlxMetallibBuildOutput(stdout, expectedPackageId, targetDir) {
  const messages = [];
  for (const line of stdout.split(/\r?\n/)) {
    if (!line) continue;
    let message;
    try { message = JSON.parse(line); } catch { continue; }
    if (message.reason === "build-script-executed" && message.package_id === expectedPackageId) messages.push(message);
  }
  check(messages.length === 1, `expected one exact pinned pmetal-mlx-sys build-script output, found ${messages.length}`);
  const outDir = messages[0].out_dir;
  check(absolute(outDir) && inside(path.resolve(targetDir), path.resolve(outDir)), "pinned MLX build-script OUT_DIR must be inside the durable Cargo target");
  return { packageId: expectedPackageId, outDir: path.resolve(outDir), path: path.join(path.resolve(outDir), "build", "lib", "mlx.metallib") };
}

export function mlxBuildEnvironment(base) {
  const env = { ...base };
  delete env.PMETAL_METALLIB_PATH;
  delete env.PMETAL_MLX_PREBUILT_DIR;
  return env;
}

function safeRunRoot(runDir) {
  check(absolute(runDir) && path.resolve(runDir) === runDir && runDir !== ROOT && !inside(ROOT, runDir), "durable owned run directory must be absolute and external to the checkout");
}

export function buildArguments(backend) {
  check(["mlx", "candle_cuda"].includes(backend), "explicit backend required");
  return ["build", "--release", "--locked", "--message-format=json-render-diagnostics", "-p", "sceneworks-rust-api", ...(backend === "candle_cuda" ? ["--features", "backend-candle"] : [])];
}

function cargoFact(args) {
  return execFileSync(args[0], args.slice(1), { cwd: ROOT, encoding: "utf8", timeout: 15000 }).trim();
}

function selectedPython() {
  const python = process.env.S9_GUARD_PYTHON;
  check(absolute(python), "absolute Actions-selected Python 3.12 interpreter required");
  const version = execFileSync(python, ["-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"], { cwd: ROOT, encoding: "utf8", timeout: 15000 }).trim();
  check(version === "3.12", "H3 owned process supervisor requires Python 3.12");
  return { path: python, version };
}

function buildIdentity({ backend, sourceFreezeSha256, runDir, targetDir, args, binaryBytes, binarySha256, cargoLockSha256, nativeResources }) {
  const envKeys = ["CARGO_BUILD_TARGET", "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "NVCC_CCBIN", "VCToolsInstallDir", "PMETAL_METALLIB_PATH", "PMETAL_MLX_PREBUILT_DIR"];
  const environment = Object.fromEntries(envKeys.map((key) => [key, key.startsWith("PMETAL_") ? null : process.env[key] ?? null]));
  const rustc = cargoFact(["rustc", "-Vv"]), rustcHost = rustc.match(/^host: (.+)$/m)?.[1];
  return {
    schemaVersion: 1, backend, sourceFreezeSha256, runDir, targetDir,
    host: { platform: process.platform, arch: process.arch, targetTriple: process.env.CARGO_BUILD_TARGET ?? rustcHost, runnerName: process.env.RUNNER_NAME ?? null, runnerId: process.env.S9_OBSERVED_RUNNER_ID },
    cargo: { rustc, cargo: cargoFact(["cargo", "-V"]), python: selectedPython(), args, profile: "release", cargoLockSha256 },
    environment, binary: { bytes: binaryBytes, sha256: binarySha256 }, nativeResources,
  };
}

async function noSymlinkPath(root, target) {
  if (!inside(root, target)) return false;
  let current = root;
  const rootInfo = await lstat(current).catch(() => null);
  if (!rootInfo?.isDirectory() || rootInfo.isSymbolicLink()) return false;
  for (const component of path.relative(root, target).split(path.sep)) {
    current = path.join(current, component);
    const info = await lstat(current).catch(() => null);
    if (!info?.isDirectory() || info.isSymbolicLink()) return false;
  }
  return true;
}

export async function verifyNativeResources(nativeResources, backend, runDir) {
  check(nativeResources?.schemaVersion === 1, `native resource inventory missing: ${backend}`);
  if (backend === "candle_cuda") {
    check(nativeResources.mode === "not_applicable" && nativeResources.mlxMetallib === null && nativeResources.sourceOutput === null, "Candle host must explicitly declare no MLX metallib resource");
    return [];
  }
  check(nativeResources.mode === "locked_cargo_source_build" && nativeResources.mlxMetallib && nativeResources.sourceOutput, "MLX host must bind its locked Cargo-built metallib");
  const { revision, packageId } = pinnedMlxSysPackageId(await readFile(path.join(ROOT, "Cargo.lock"), "utf8"));
  const targetDir = path.join(runDir, "cargo-target");
  const resource = nativeResources.mlxMetallib, source = nativeResources.sourceOutput;
  check(resource.path === mlxMetallibPath(runDir) && source.packageId === packageId && source.revision === revision, "MLX resource identity differs from exact locked pmetal-mlx-sys package");
  check(path.resolve(source.outDir) === source.outDir && await noSymlinkPath(targetDir, source.outDir) && source.path === path.join(source.outDir, "build", "lib", "mlx.metallib") && (await lstat(path.dirname(resource.path))).isDirectory() && !(await lstat(path.dirname(resource.path))).isSymbolicLink(), "MLX resource source path differs from exact build-script output");
  const checked = [];
  for (const binding of [source, resource]) {
    const info = await lstat(binding.path).catch((error) => { if (error.code === "ENOENT") return null; throw error; });
    check(info?.isFile() && !info.isSymbolicLink() && Number.isSafeInteger(binding.bytes) && binding.bytes > 0 && info.size === binding.bytes && HEX.test(binding.sha256) && await hashFile(binding.path) === binding.sha256, `MLX metallib missing or changed: ${binding.path}`);
    checked.push({ path: binding.path, bytes: info.size, sha256: binding.sha256 });
  }
  check(source.bytes === resource.bytes && source.sha256 === resource.sha256, "durable MLX metallib copy differs from exact Cargo output");
  return checked;
}

export async function buildDurableProduct({ manifestPath, backend, runDir, runnerName, expectedRunner, sceneWorksSha, inferenceSha, build = execFileSync }) {
  check(absolute(manifestPath), "absolute source-frozen manifest required");
  safeRunRoot(runDir);
  check(process.env.GITHUB_ACTIONS === "true", "owned Actions build-only mode required");
  check(/^\d+$/.test(process.env.S9_OBSERVED_RUNNER_ID ?? ""), "live runner identity readback required before durable build");
  selectedPython();
  const m = await json(manifestPath);
  assertActionsIdentity({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, manifest: m, runDir, cacheRoot: path.join(runDir, "offline-cache") });
  check(git(ROOT, "rev-parse", "HEAD") === sceneWorksSha && git(ROOT, "status", "--porcelain") === "", "SceneWorks checkout dirty/stale");
  check(git(m.sourceFreeze.inference.root, "rev-parse", "HEAD") === inferenceSha && git(m.sourceFreeze.inference.root, "status", "--porcelain") === "", "inference checkout dirty/stale");
  check(git(m.sourceFreeze.inference.root, "rev-parse", "refs/remotes/origin/main") === inferenceSha, "inference pin is not fetched permanent main");
  await inspectSources(m);
  const targetDir = targetCargoDir(runDir);
  const args = buildArguments(backend);
  const runInfo = await lstat(runDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
  check(!runInfo || (runInfo.isDirectory() && !runInfo.isSymbolicLink()), "durable run directory unsafe");
  await mkdir(runDir, { recursive: true });
  const receiptPath = path.join(runDir, `${backend}-product-build-receipt.json`), attestationPath = productAttestationPath(runDir, backend), destination = productPath(runDir, backend), resourceDestination = mlxMetallibPath(runDir);
  const productFiles = backend === "mlx" ? [receiptPath, attestationPath, destination, resourceDestination] : [receiptPath, attestationPath, destination];
  const existing = await Promise.all(productFiles.map((file) => lstat(file).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e))));
  if (existing.some(Boolean)) {
    check(existing.every((item) => item?.isFile() && !item.isSymbolicLink()), "partial or unsafe durable product build exists; preserve it for reconciliation");
    const receipt = await json(receiptPath), attestation = await json(attestationPath), binarySha256 = await hashFile(destination), binaryBytes = existing[2].size;
    check(receipt.schemaVersion === 1 && receipt.backend === backend && receipt.runnerName === runnerName && receipt.runnerId === process.env.S9_OBSERVED_RUNNER_ID && receipt.source?.sceneworksHead === sceneWorksSha && receipt.source?.inferencePin === inferenceSha && receipt.source?.sourceFreezeSha256 === digest(m.sourceFreeze), "existing durable build receipt source/runner differs");
    check(receipt.inputManifestSha256 === await hashFile(manifestPath) && receipt.apiBinary?.path === destination && receipt.workerBinary?.path === destination && receipt.apiBinary.sha256 === binarySha256 && receipt.workerBinary.sha256 === binarySha256 && receipt.apiBinary.bytes === binaryBytes && receipt.workerBinary.bytes === binaryBytes, "existing durable build receipt binary/input differs");
    check(receipt.attestation?.path === attestationPath && receipt.attestation.bytes === existing[1].size && receipt.attestation.sha256 === await hashFile(attestationPath), "existing durable build receipt attestation differs");
    check(attestation.schemaVersion === 3 && attestation.backend === backend && attestation.sceneworksHead === sceneWorksSha && attestation.inferencePin === inferenceSha && attestation.sourceFreezeSha256 === digest(m.sourceFreeze) && attestation.inputManifestSha256 === receipt.inputManifestSha256 && attestation.apiBinarySha256 === binarySha256 && attestation.workerBinarySha256 === binarySha256 && attestation.binaryBytes === binaryBytes && attestation.workerId === m.runners[backend].workerId && attestation.buildIdentitySha256 === digest(attestation.buildIdentity) && receipt.buildIdentitySha256 === attestation.buildIdentitySha256 && digest(attestation.nativeResources) === digest(receipt.nativeResources), "existing durable build attestation differs");
    await verifyNativeResources(receipt.nativeResources, backend, runDir);
    return receipt;
  }
  await mkdir(targetDir, { recursive: true });
  const targetInfo = await lstat(targetDir);
  check(targetInfo.isDirectory() && !targetInfo.isSymbolicLink() && await noSymlinkPath(runDir, targetDir), "durable Cargo target directory unsafe");
  const cargoEnv = mlxBuildEnvironment({ ...process.env, CARGO_TARGET_DIR: targetDir });
  let cargoOutput;
  try { cargoOutput = build("cargo", args, { cwd: ROOT, env: cargoEnv, encoding: "utf8", maxBuffer: 128 * 1024 * 1024, stdio: ["ignore", "pipe", "inherit"], timeout: 60 * 60_000 }); }
  catch (error) { if (error.stdout) process.stdout.write(error.stdout); throw error; }
  if (cargoOutput) process.stdout.write(cargoOutput);
  let nativeResources = { schemaVersion: 1, mode: "not_applicable", sourceOutput: null, mlxMetallib: null };
  if (backend === "mlx") {
    const { revision, packageId } = pinnedMlxSysPackageId(await readFile(path.join(ROOT, "Cargo.lock"), "utf8"));
    const sourceOutput = mlxMetallibBuildOutput(cargoOutput ?? "", packageId, targetDir);
    sourceOutput.revision = revision;
    const sourceInfo = await lstat(sourceOutput.path);
    check(sourceInfo.isFile() && !sourceInfo.isSymbolicLink() && sourceInfo.size > 0, "exact locked MLX build output metallib missing or unsafe");
    sourceOutput.bytes = sourceInfo.size; sourceOutput.sha256 = await hashFile(sourceOutput.path);
    await mkdir(path.dirname(resourceDestination), { recursive: true });
    check(await noSymlinkPath(runDir, path.dirname(resourceDestination)), "durable MLX resource parent is unsafe");
    check(!await lstat(resourceDestination).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "durable MLX metallib already exists; refusing replacement");
    const tempResource = `${resourceDestination}.pending-${process.pid}-${randomBytes(5).toString("hex")}`;
    await copyFile(sourceOutput.path, tempResource, fsConstants.COPYFILE_EXCL);
    try { await rename(tempResource, resourceDestination); } catch (error) { await unlink(tempResource).catch(() => {}); throw error; }
    const resourceInfo = await lstat(resourceDestination);
    nativeResources = { schemaVersion: 1, mode: "locked_cargo_source_build", sourceOutput, mlxMetallib: { path: resourceDestination, bytes: resourceInfo.size, sha256: await hashFile(resourceDestination) } };
    await verifyNativeResources(nativeResources, backend, runDir);
  }
  const built = path.join(targetDir, "release", productName());
  const builtInfo = await lstat(built);
  check(builtInfo.isFile() && !builtInfo.isSymbolicLink() && builtInfo.size > 0, "release product API binary missing after build");
  const destinationRoot = path.dirname(destination);
  await mkdir(destinationRoot, { recursive: true });
  const destinationRootInfo = await lstat(destinationRoot);
  check(destinationRootInfo.isDirectory() && !destinationRootInfo.isSymbolicLink() && await noSymlinkPath(runDir, destinationRoot), "durable product directory unsafe");
  check(!await lstat(destination).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "durable product binary already exists; refusing replacement");
  const tempBinary = `${destination}.pending-${process.pid}-${randomBytes(5).toString("hex")}`;
  await copyFile(built, tempBinary, fsConstants.COPYFILE_EXCL);
  try { await rename(tempBinary, destination); } catch (error) { await unlink(tempBinary).catch(() => {}); throw error; }
  const productInfo = await lstat(destination);
  check(productInfo.isFile() && !productInfo.isSymbolicLink() && productInfo.size === builtInfo.size, "durable product copy differs from build output");
  const binarySha256 = await hashFile(destination);
  const sourceFreezeSha256 = digest(m.sourceFreeze);
  const identity = buildIdentity({ backend, sourceFreezeSha256, runDir, targetDir, args, binaryBytes: productInfo.size, binarySha256, cargoLockSha256: sha(await readFile(path.join(ROOT, "Cargo.lock"))), nativeResources });
  const attestation = {
    schemaVersion: 3, method: "actions_source_build_durable_same_binary_api_worker_and_native_resources", backend,
    inferencePin: inferenceSha, sceneworksHead: sceneWorksSha, sourceFreezeSha256,
    inputManifestSha256: await hashFile(manifestPath), apiBinarySha256: binarySha256,
    workerBinarySha256: binarySha256, binaryBytes: productInfo.size,
    workerId: m.runners[backend].workerId, nativeResources, buildIdentity: identity, buildIdentitySha256: digest(identity),
  };
  await writeFile(attestationPath, JSON.stringify(attestation, null, 2) + "\n", { flag: "wx" });
  const receipt = {
    schemaVersion: 1, backend, runnerName, runnerId: process.env.S9_OBSERVED_RUNNER_ID, nativeResources,
    source: { sceneworksHead: sceneWorksSha, inferencePin: inferenceSha, sourceFreezeSha256 },
    inputManifestSha256: attestation.inputManifestSha256,
    apiBinary: { path: destination, bytes: productInfo.size, sha256: binarySha256 },
    workerBinary: { path: destination, bytes: productInfo.size, sha256: binarySha256 },
    attestation: { path: attestationPath, bytes: (await lstat(attestationPath)).size, sha256: await hashFile(attestationPath) },
    buildIdentity: identity, buildIdentitySha256: digest(identity), campaignReady: false,
  };
  await writeFile(path.join(runDir, `${backend}-product-build-receipt.json`), JSON.stringify(receipt, null, 2) + "\n", { flag: "wx" });
  return receipt;
}

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
  return { runId: Number(runId), jobId: matchingJobs[0].id, runnerId: matchingJobs[0].runner_id, runnerName, readable: true, ownRunReadable: true, ownRunnerIdentityVerified: true, resourceQueueVerified: true, resourceScope: "cross_repository_runner_and_queue_read" };
}

export function probeCpuInventoryAuthority({ backend, runnerName, sceneWorksSha, runId = process.env.GITHUB_RUN_ID, attempt = process.env.GITHUB_RUN_ATTEMPT, exec = execFileSync } = {}) {
  const expectedJob = backend === "mlx" ? "mac2" : backend === "candle_cuda" ? "windows_cuda" : null;
  const runnerIds = { "nax-macos-2": 5281, "cuda-windows": 2313, "cuda-windows-2": 2619 };
  check(expectedJob && (backend === "mlx" ? runnerName === "nax-macos-2" : ["cuda-windows", "cuda-windows-2"].includes(runnerName)) && Number.isSafeInteger(runnerIds[runnerName]), "CPU inventory backend/runner is not the frozen physical listener");
  check(process.env.GITHUB_ACTIONS === "true" && process.env.GH_TOKEN, "same-repository Actions read credential unavailable");
  check(process.env.GITHUB_REPOSITORY === "SceneWorks/SceneWorks" && process.env.GITHUB_EVENT_NAME === "workflow_dispatch", "CPU inventory owner repository/event differs");
  check(process.env.GITHUB_SHA === sceneWorksSha && process.env.GITHUB_WORKFLOW_REF?.startsWith("SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@") && process.env.GITHUB_JOB === expectedJob && process.env.RUNNER_NAME === runnerName, "CPU inventory workflow/source/job/runner differs");
  check(/^\d+$/.test(runId ?? "") && /^\d+$/.test(attempt ?? ""), "CPU inventory run/attempt missing");
  const read = (endpoint, paginated = false) => {
    try { return JSON.parse(exec("gh", paginated ? ["api", "--paginate", "--slurp", endpoint] : ["api", endpoint], { encoding: "utf8", timeout: 15000, stdio: ["ignore", "pipe", "pipe"] })); }
    catch { throw new Error(`H3 S9 Actions: same-repository credential cannot read ${endpoint.split("?")[0]}`); }
  };
  const run = read(`repos/SceneWorks/SceneWorks/actions/runs/${runId}`);
  const pages = read(`repos/SceneWorks/SceneWorks/actions/runs/${runId}/jobs?per_page=100`, true);
  check(run?.id === Number(runId) && run.run_attempt === Number(attempt) && run.head_sha === sceneWorksSha && run.status === "in_progress" && run.conclusion === null && run.event === "workflow_dispatch" && run.name === "H3 Veda terminal owned campaign" && run.path === ".github/workflows/h3-veda-terminal.yml", "CPU inventory exact live workflow readback differs");
  check(Array.isArray(pages) && pages.length > 0 && pages.every((page) => Array.isArray(page.jobs)), "CPU inventory paginated job readback malformed");
  const matches = pages.flatMap((page) => page.jobs).filter((job) => job.run_id === Number(runId) && job.run_attempt === Number(attempt) && job.head_sha === sceneWorksSha && job.name === expectedJob && job.runner_name === runnerName && job.runner_id === runnerIds[runnerName] && job.status === "in_progress" && job.conclusion === null && Number.isSafeInteger(job.id) && job.id > 0);
  check(matches.length === 1, "CPU inventory live job/runner identity missing or ambiguous");
  return { runId: Number(runId), jobId: matches[0].id, runnerId: runnerIds[runnerName], runnerName, ownRunReadable: true, ownRunnerIdentityVerified: true, resourceQueueVerified: false, resourceScope: "same_repository_actions_identity_only" };
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

export async function verifyDurableProduct(m, backend, runDir, { sceneWorksSha, inferenceSha } = {}) {
  safeRunRoot(runDir);
  const runner = m.runners?.[backend], expected = productPath(runDir, backend);
  check(runner?.apiBinary?.path === expected && runner?.workerBinary?.path === expected && runner.apiBinary.sha256 === runner.workerBinary.sha256 && runner.apiBinary.bytes === runner.workerBinary.bytes, "API/worker descriptors must bind the exact same durable run-owned binary");
  const binary = await boundFile(runner.apiBinary);
  check(binary.path === expected && binary.bytes === runner.workerBinary.bytes, "durable product binary descriptor differs");
  const attestationBinding = runner.attestation;
  check(attestationBinding?.path === productAttestationPath(runDir, backend), "product attestation must use the exact durable run-owned path");
  const attestationFile = await boundFile(attestationBinding);
  const attestation = JSON.parse(await readFile(attestationFile.path, "utf8"));
  check(attestation.schemaVersion === 3 && attestation.method === "actions_source_build_durable_same_binary_api_worker_and_native_resources" && attestation.backend === backend, "durable product attestation schema/backend differs");
  check(attestation.sceneworksHead === sceneWorksSha && attestation.inferencePin === inferenceSha && attestation.sourceFreezeSha256 === digest(m.sourceFreeze), "durable product source freeze differs");
  check(attestation.apiBinarySha256 === binary.sha256 && attestation.workerBinarySha256 === binary.sha256 && attestation.binaryBytes === binary.bytes && attestation.workerId === runner.workerId, "durable product binary/worker identity differs");
  check(attestation.nativeResources && digest(attestation.nativeResources) === digest(runner.nativeResources), "durable native resource descriptors differ from bound host manifest");
  const nativeResources = await verifyNativeResources(runner.nativeResources, backend, runDir);
  check(attestation.buildIdentity?.backend === backend && attestation.buildIdentity.sourceFreezeSha256 === digest(m.sourceFreeze) && attestation.buildIdentity.binary?.sha256 === binary.sha256 && attestation.buildIdentity.binary?.bytes === binary.bytes && digest(attestation.buildIdentity.nativeResources) === digest(attestation.nativeResources) && attestation.buildIdentity.environment?.PMETAL_METALLIB_PATH === null && attestation.buildIdentity.environment?.PMETAL_MLX_PREBUILT_DIR === null && attestation.buildIdentitySha256 === digest(attestation.buildIdentity), "durable product build identity differs");
  return { binary, attestation: attestationBinding, buildIdentitySha256: attestation.buildIdentitySha256, nativeResources };
}

// A source-frozen preparation manifest may leave only the product import IDs
// unbound: prepare creates those through the normal owned API and reads them back.
export function assertOwnedImportReadiness(m) {
  const missing = m.preparation?.missingBindings;
  check(m.sourceFreeze?.stage === "permanent-inference-main-pin" && m.preparation?.campaignReady === false && missing?.productImports === true, "source-frozen preparation with actual product imports pending required");
  check(Array.isArray(m.preparation.unresolved) && m.preparation.unresolved.length === 1 && m.preparation.unresolved[0] === "normal owned corpus imports/readbacks required", "other preparation gaps remain unresolved");
  for (const [key, value] of Object.entries(missing)) if (key !== "productImports") {
    check(value === false || value === null || (Array.isArray(value) && value.length === 0), `preparation binding still missing: ${key}`);
  }
  for (const backend of ["mlx", "candle_cuda"]) {
    const runner = m.runners?.[backend], resource = m.sourceFreeze.resourceContract?.[backend];
    check(typeof runner?.physicalGpuId === "string" && runner.physicalGpuId.length > 0 && resource?.physicalGpuId === runner.physicalGpuId, `physical resource identity unbound: ${backend}`);
    for (const name of ["apiBinary", "workerBinary", "attestation"]) {
      const file = runner[name];
      check(typeof file?.path === "string" && file.path.length > 0 && Number.isSafeInteger(file.bytes) && file.bytes > 0 && HEX.test(file.sha256 ?? ""), `source-built binary descriptor unbound: ${backend}/${name}`);
    }
    const native = runner.nativeResources;
    if (backend === "mlx") {
      const source = native?.sourceOutput, resource = native?.mlxMetallib;
      check(native.schemaVersion === 1 && native.mode === "locked_cargo_source_build" && source && resource && absolute(source.outDir) && absolute(source.path) && absolute(resource.path) && source.path === path.join(source.outDir, "build", "lib", "mlx.metallib") && Number.isSafeInteger(source.bytes) && source.bytes > 0 && source.bytes === resource.bytes && HEX.test(source.sha256 ?? "") && source.sha256 === resource.sha256 && source.packageId === `git+${MLX_SYS_SOURCE}?rev=${source.revision}#pmetal-mlx-sys@${source.packageId.split("@").at(-1)}` && HEAD.test(source.revision ?? ""), `native resource provenance unbound: ${backend}`);
    } else check(native?.schemaVersion === 1 && native.mode === "not_applicable" && native.mlxMetallib === null && native.sourceOutput === null, `native resource provenance unbound: ${backend}`);
    check(runner.apiBinary.sha256 === runner.workerBinary.sha256 && runner.apiBinary.bytes === runner.workerBinary.bytes && runner.apiBinary.path === runner.workerBinary.path, `API/worker binary identity differs: ${backend}`);
  }
}

export async function preflight({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha }) {
  check(absolute(manifestPath), "absolute frozen manifest input required");
  check(process.env.GITHUB_ACTIONS === "true" && process.env.GH_TOKEN, "source-bound Actions execution and scoped live runner token required");
  const m = await json(manifestPath);
  assertOwnedImportReadiness(m);
  const g = assertActionsIdentity({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, manifest: m, runDir, cacheRoot });
  check(git(ROOT, "rev-parse", "HEAD") === sceneWorksSha && git(ROOT, "status", "--porcelain") === "", "SceneWorks checkout dirty/stale");
  check(git(m.sourceFreeze.inference.root, "rev-parse", "HEAD") === inferenceSha && git(m.sourceFreeze.inference.root, "status", "--porcelain") === "", "inference checkout dirty/stale");
  check(git(m.sourceFreeze.inference.root, "rev-parse", "refs/remotes/origin/main") === inferenceSha, "inference pin is not fetched permanent main");
  const cache = await lstat(cacheRoot); check(cache.isDirectory() && !cache.isSymbolicLink(), "offline cache unavailable/unsafe");
  const sw = await lstat(runDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
  check(!sw || (sw.isDirectory() && !sw.isSymbolicLink()), "run directory unsafe");
  await verifyDurableProduct(m, backend, runDir, { sceneWorksSha, inferenceSha });
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

async function stopOwnedImportSupervisor(child, { manifest, backend, runDir }) {
  check(child?.pid, "owned CPU import API supervisor did not start");
  if (child.exitCode === null) {
    const ended = new Promise((resolve) => child.once("exit", resolve));
    child.kill("SIGTERM");
    const stopped = await Promise.race([ended.then((code) => ({ stopped: true, code })), delay(15000).then(() => ({ stopped: false }))]);
    check(stopped.stopped, "CPU import API supervisor/tree still alive; preserve owned workspace for inspection");
    check(stopped.code === 0, "CPU import API supervisor failed before recording process-tree closure");
  }
  const closurePath = path.join(runDir, "cpu-import-api", "closure.json"), closure = await json(closurePath);
  check(closure.schemaVersion === 1 && closure.backend === backend && closure.physicalLockClaimed === false && closure.cpuOnlyEnvironment === true && closure.sourceFreezeSha256 === digest(manifest.sourceFreeze), "CPU import API closure source/device identity differs");
  check(closure.api?.reaped === true && closure.api.descendantsDrained === true && Number.isSafeInteger(closure.api.pid) && closure.api.pid > 0 && closure.api.birth && closure.api.apiBinarySha256 === manifest.runners[backend].apiBinary.sha256, "CPU import API process-tree drain proof missing or wrong binary");
  check(!Object.hasOwn(closure.api.apiConfigBinding?.environment ?? {}, "SCENEWORKS_ACCESS_TOKEN") && !Object.hasOwn(closure.api.apiConfigBinding?.environment ?? {}, "SCENEWORKS_PARENT_PID"), "CPU import closure exposes private API token or supervisor PID");
  return { path: closurePath, bytes: (await lstat(closurePath)).size, sha256: await hashFile(closurePath), apiPid: closure.api.pid, apiBirth: closure.api.birth, apiBinarySha256: closure.api.apiBinarySha256, reaped: closure.api.reaped, descendantsDrained: closure.api.descendantsDrained };
}

export async function importOwnedCorpus({ manifest, manifestPath, plan, planPath, corpusRoot, runDir, cacheRoot, backend, token, spawnApi = spawn }) {
  check(token && !/\s/.test(token), "private product API token absent");
  check(absolute(manifestPath), "absolute source-bound combined manifest required for CPU import supervisor");
  const g = manifest.runners[backend].guardian;
  check(!await lstat(g.dataDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "owned data directory already exists; no implicit reuse");
  check(!await lstat(g.configDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "owned config directory already exists; no implicit reuse");
  await requireFreeLoopbackPort(g.port);
  await mkdir(g.dataDir, { recursive: true }); await mkdir(path.join(g.configDir, "manifests"), { recursive: true });
  for (const name of ["builtin.models.jsonc", "builtin.loras.jsonc"]) await copyFile(path.join(ROOT, "config/manifests", name), path.join(g.configDir, "manifests", name));
  const env = cpuImportApiEnvironment(process.env, { dataDir: g.dataDir, configDir: g.configDir, jobsDbPath: path.join(g.dataDir, "cache/jobs.db"), cacheRoot, apiUrl: manifest.runners[backend].apiUrl, token, backend, nativeResources: manifest.runners[backend].nativeResources });
  const python = selectedPython().path;
  const child = spawnApi(python, [path.join(ROOT, "scripts/h3-veda-guardian.py"), "--cpu-import-api", manifestPath, backend, runDir], { cwd: ROOT, env, stdio: "ignore" });
  let supervisorStopped = false, apiProcessClosure;
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
    apiProcessClosure = await stopOwnedImportSupervisor(child, { manifest, backend, runDir }); supervisorStopped = true;
    const receipt = { projectId: project.id, planSha256: await hashFile(planPath), apiProcessClosure, assets };
    const receiptPath = path.join(runDir, "corpus-import-receipt.json"); await writeFile(receiptPath, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx" });
    return { receipt, receiptPath };
  } finally { if (!supervisorStopped) await stopOwnedImportSupervisor(child, { manifest, backend, runDir }); }
}

export async function prepare({ manifestPath, planPath, corpusRoot, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha, token }) {
  check(absolute(corpusRoot) && corpusRoot !== ROOT && !inside(ROOT, corpusRoot), "external staged corpus root required");
  const staged = await lstat(corpusRoot); check(staged.isDirectory() && !staged.isSymbolicLink(), "staged corpus directory unavailable/unsafe");
  const { manifest } = await preflight({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha });
  const plan = await json(planPath);
  check(path.resolve(planPath) === path.join(ROOT, "config/terminal-evidence/epic-25140-corpus.plan.json"), "corpus plan must be source-owned");
  check(HEX.test(manifest.sourceFreeze.files["config/terminal-evidence/epic-25140-corpus.plan.json"]) && await hashFile(planPath) === manifest.sourceFreeze.files["config/terminal-evidence/epic-25140-corpus.plan.json"], "corpus plan differs from freeze");
  const { receipt, receiptPath } = await importOwnedCorpus({ manifest, manifestPath, plan, planPath, corpusRoot, runDir, cacheRoot, backend, token });
  const { bindCorpusPlan } = await import("./h3-veda-corpus.mjs");
  manifest.corpusPlan = { path: path.resolve(planPath), bytes: (await lstat(planPath)).size, sha256: await hashFile(planPath) };
  manifest.corpusImportReceipt = { path: receiptPath, bytes: (await lstat(receiptPath)).size, sha256: await hashFile(receiptPath) };
  manifest.cpuImportApiClosures = { ...(manifest.cpuImportApiClosures ?? {}), [backend]: receipt.apiProcessClosure };
  const bound = bindCorpusPlan(manifest, plan, receipt);
  check(bound.projectId === receipt.projectId, "bound project differs from owned import");
  bound.preparation = { ...bound.preparation, campaignReady: false, productImported: true,
    missingBindings: { ...bound.preparation.missingBindings, productImports: false }, unresolved: [] };
  const marker = { campaignId: bound.campaignId, sourceFreezeSha256: digest(bound.sourceFreeze), projectId: bound.projectId };
  await writeFile(path.join(bound.runners[backend].guardian.dataDir, "s9-workspace-owner.json"), JSON.stringify(marker) + "\n", { flag: "wx" });
  const output = path.join(runDir, "manifest.bound.json"); await writeFile(output, JSON.stringify(bound, null, 2) + "\n", { flag: "wx" });
  await validateManifest(bound, { backend });
  return { manifestPath: output, receiptPath, fingerprint: digest(bound), source: { sceneWorksSha, inferenceSha }, backend, runnerName, cpuImportOnly: true };
}

export async function resume({ manifestPath, backend, runDir, cacheRoot, runnerName, expectedRunner, sceneWorksSha, inferenceSha }) {
  check(path.resolve(manifestPath) === path.join(runDir, "manifest.bound.json"), "resume must use exact owned bound manifest");
  check(process.env.GITHUB_ACTIONS === "true" && process.env.GH_TOKEN, "source-bound Actions execution and scoped live runner token required");
  selectedPython();
  const m = await json(manifestPath);
  const g = assertActionsIdentity({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, manifest: m, runDir, cacheRoot });
  check(git(ROOT, "rev-parse", "HEAD") === sceneWorksSha && git(ROOT, "status", "--porcelain") === "", "SceneWorks checkout dirty/stale");
  check(git(m.sourceFreeze.inference.root, "rev-parse", "HEAD") === inferenceSha && git(m.sourceFreeze.inference.root, "status", "--porcelain") === "" && git(m.sourceFreeze.inference.root, "rev-parse", "refs/remotes/origin/main") === inferenceSha, "inference checkout dirty/stale or not permanent main");
  await verifyDurableProduct(m, backend, runDir, { sceneWorksSha, inferenceSha });
  const marker = await json(path.join(g.dataDir, "s9-workspace-owner.json"));
  check(digest(marker) === digest({ campaignId: m.campaignId, sourceFreezeSha256: digest(m.sourceFreeze), projectId: m.projectId }), "resume owned workspace marker differs");
  // A new Actions job has a new private API token. Never attach to a retained
  // guardian whose API child still holds the previous job's token.
  await requirePreviousJobClosed(runDir);
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
    if (cell) {
      check(cell.kind === row.kind, `host row kind differs: ${row.id}`);
      check(cell.status !== "expected_refusal_observed" || row.kind === "expected_refusal", `non-refusal row claims refusal: ${row.id}`);
      check(!["media_collected_review_pending", "media_collected_evidence_incomplete"].includes(cell.status) || row.kind === "media", `non-media row claims media: ${row.id}`);
      check(cell.status !== "product_proof_observed" || row.kind === "product_proof", `non-proof row claims product proof: ${row.id}`);
    }
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
    check(h.cases.every((c) => (c.status !== "expected_refusal_observed" || c.kind === "expected_refusal") && (!["media_collected_review_pending", "media_collected_evidence_incomplete"].includes(c.status) || c.kind === "media") && (c.status !== "product_proof_observed" || c.kind === "product_proof")), "host row disposition differs from required kind");
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

const MAX_REVIEW_CASES = 4;
const MAX_REVIEW_BYTES = 8 * 1024 ** 3;
export async function exportReviewMedia({ runDir, outputDir, backend, caseIds, sourceCheck = inspectSources, rows: suppliedRows } = {}) {
  check(absolute(runDir) && absolute(outputDir) && runDir !== outputDir && !inside(runDir, outputDir) && !inside(outputDir, runDir) && !inside(ROOT, outputDir) && ["mlx", "candle_cuda"].includes(backend), "review requires external owned run/output roots and physical backend");
  check(Array.isArray(caseIds) && caseIds.length > 0 && caseIds.length <= MAX_REVIEW_CASES && caseIds.every((id) => typeof id === "string" && id.length > 0 && id.length <= 256) && new Set(caseIds).size === caseIds.length, "explicit unique bounded review case identities required");
  const rootInfo = await lstat(runDir); check(rootInfo.isDirectory() && !rootInfo.isSymbolicLink(), "owned review run root unsafe");
  await requirePreviousJobClosed(runDir);
  const manifestPath = path.join(runDir, "manifest.bound.json"), campaignPath = path.join(runDir, "campaign.json"), controlPath = path.join(runDir, "physical-guardian/control.json"), closurePath = path.join(runDir, "physical-guardian/closed.json");
  const m = await json(manifestPath), campaign = await json(campaignPath), control = await json(controlPath), closure = await json(closurePath);
  await sourceCheck(m);
  check(campaign.campaignId === m.campaignId && campaign.fingerprint === digest(m), "review campaign/manifest identity differs");
  check(closure.closed === true && closure.servicesDrained === true && closure.physicalLockReleased === true && closure.fingerprint === digest(m), "review requires proven drained worker and released physical lock");
  check(control.fingerprint === digest(m) && /^[a-f0-9]{48}$/.test(control.token ?? "") && sha(control.token) === closure.ownerTokenSha256 && Number.isSafeInteger(control.pid) && control.pid > 0 && control.pid === closure.pid, "review guardian control/closure identity differs");
  let ownerAlive = true; try { process.kill(control.pid, 0); } catch (error) { check(error.code === "ESRCH", "review guardian process state unknown"); ownerAlive = false; }
  check(!ownerAlive, "review guardian still alive");
  check(Object.values(campaign.cells ?? {}).every((cell) => Object.values(cell.arms ?? {}).every((arm) => !["submitting", "ambiguous", "timeout_owned_job_active"].includes(arm.status) && (!arm.jobId || ["completed", "failed", "canceled", "interrupted"].includes(arm.job?.status)))), "review has unresolved live or ambiguous owned child");
  check(absolute(m.runners?.[backend]?.leasePath), "review requires exact campaign lease identity");
  const lease = await lstat(m.runners[backend].leasePath).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
  check(!lease, "review campaign lease still held");
  const rows = suppliedRows ?? declaredRows(m, await catalogs(m.sourceFreeze.sceneworks.root));
  const allowed = new Map(rows.filter((row) => row.backend === backend && ["media", "product_proof"].includes(row.kind)).map((row) => [row.id, row]));
  check(caseIds.every((id) => allowed.has(id)), "review selection outside declared owning-host output rows");
  const realRoot = await realpath(runDir), sourceHash = await hashFile(campaignPath), manifestHash = await hashFile(manifestPath), controlHash = await hashFile(controlPath), closureHash = await hashFile(closurePath), files = [], selectedCases = [];
  for (let index = 0; index < caseIds.length; index++) {
    const id = caseIds[index], row = allowed.get(id), cell = campaign.cells?.[id];
    check(cell?.kind === row.kind, `review case missing or wrong kind: ${id}`);
    selectedCases.push({ id, kind: row.kind, profile: row.profile, model: row.model, backend, tier: row.tier, recipe: row.recipe, canvas: row.canvas, frames: row.frames, duration: row.duration, style: row.style ?? null, observedStatus: cell.status });
    const entries = [];
    const add = (role, binding, expectedPath) => { check(binding?.path === expectedPath && Number.isSafeInteger(binding.bytes) && binding.bytes > 0 && binding.bytes <= m.limits.maxMediaBytes && HEX.test(binding.sha256), `review media binding incomplete: ${id}/${role}`); entries.push({ id, index, role, binding }); };
    for (const arm of row.kind === "media" ? ["dense", "sparse"] : ["sparse"]) {
      const attempt = cell.arms?.[arm]; check(attempt?.status === "collected" && HEX.test(attempt.key), `review product output not collected: ${id}/${arm}`);
      add(`${arm}-product`, attempt.media, path.join(runDir, attempt.key, `${arm}.mp4`));
    }
    if (row.kind === "media") {
      const dir = path.join(runDir, digest(id));
      add("side-by-side", cell.comparison?.sideBySide, path.join(dir, "side-by-side.mp4"));
      add("dense-audio", cell.comparison?.denseAudio, path.join(dir, "dense.wav"));
      add("sparse-audio", cell.comparison?.sparseAudio, path.join(dir, "sparse.wav"));
    }
    files.push(...entries);
  }
  check(files.length <= MAX_REVIEW_CASES * 5 && files.reduce((sum, file) => sum + file.binding.bytes, 0) <= MAX_REVIEW_BYTES, "selected review transport exceeds bounded file/byte budget");
  await mkdir(path.dirname(outputDir), { recursive: true }); await mkdir(outputDir);
  const exported = [];
  for (const { id, index, role, binding } of files) {
    const realSource = await realpath(binding.path);
    check(inside(realRoot, realSource), "review source escapes owned run directory");
    await boundFile(binding);
    const extension = role.endsWith("audio") ? ".wav" : ".mp4";
    const relative = `case-${String(index + 1).padStart(2, "0")}/${role}${extension}`;
    const target = path.join(outputDir, relative);
    await mkdir(path.dirname(target), { recursive: true }); await copyFile(binding.path, target, fsConstants.COPYFILE_EXCL);
    check(await hashFile(target) === binding.sha256 && (await lstat(target)).size === binding.bytes, "staged review bytes differ from raw product output");
    exported.push({ caseId: id, role, file: relative, bytes: binding.bytes, sha256: binding.sha256 });
  }
  check(await hashFile(campaignPath) === sourceHash && await hashFile(manifestPath) === manifestHash && await hashFile(controlPath) === controlHash && await hashFile(closurePath) === closureHash && !await lstat(m.runners[backend].leasePath).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "review campaign/guardian/lease changed during transport");
  const review = { schemaVersion: 1, kind: "epic25140_selected_real_output_review", backend, campaignId: m.campaignId, manifestFingerprint: digest(m), manifestSha256: manifestHash, campaignSha256: sourceHash, guardianClosureSha256: closureHash, source: { sceneworksHead: m.sourceFreeze.sceneworks.head, inferencePin: m.sourceFreeze.inferencePin }, selectedCases, files: exported, selectedForReviewOnly: true, qualityAcceptance: "pending Michael", completedAcceptance: false, allRequiredRowsStillRequireNativeEvidence: true };
  await writeFile(path.join(outputDir, "review-media.json"), JSON.stringify(review, null, 2) + "\n", { flag: "wx" });
  return review;
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
    if (command === "inventory" || command === "stage") {
      check(args.length === 1 && ["mlx", "candle_cuda"].includes(args[0]), "inventory|stage BACKEND");
      const setup = await import("./h3-veda-host-setup.mjs");
      const observation = await setup.inventory({ backend: args[0], runnerName: process.env.RUNNER_NAME, expectedRunner: process.env.S9_EXPECTED_RUNNER, sceneWorksSha: process.env.S9_SCENEWORKS_SHA, inferenceSha: process.env.S9_INFERENCE_SHA, authority: command === "inventory" ? probeCpuInventoryAuthority : probeReadAuthority });
      const receipt = command === "inventory" ? observation : await setup.stage({ observation, runDir: process.env.S9_RUN_DIR || undefined, corpusRoot: process.env.S9_CORPUS_ROOT || undefined, cacheRoot: process.env.S9_CACHE || undefined });
      check(absolute(process.env.RUNNER_TEMP), "owned Actions temporary directory unavailable");
      const output = path.join(process.env.RUNNER_TEMP, "h3-veda-terminal", "setup-receipt.json");
      await mkdir(path.dirname(output), { recursive: true }); await writeFile(output, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx" });
      console.log(JSON.stringify({ command, output, campaignReady: false, stagedCarriers: receipt.stagedCarriers ?? 0 }));
    } else if (command === "owned-token") {
      check(args.length === 0, "owned-token takes no arguments");
      await emitOwnedApiToken();
    } else if (command === "probe-auth") {
      check(args.length === 1 && ["mlx", "candle_cuda"].includes(args[0]), "probe-auth BACKEND");
      const receipt = probeReadAuthority({ backend: args[0], runnerName: process.env.RUNNER_NAME, sceneWorksSha: process.env.S9_SCENEWORKS_SHA });
      check(absolute(process.env.GITHUB_ENV) && Number.isSafeInteger(receipt.runnerId), "live Actions environment/runner readback unavailable");
      await appendFile(process.env.GITHUB_ENV, `S9_OBSERVED_RUNNER_ID=${receipt.runnerId}\n`, { encoding: "utf8" });
      console.log(JSON.stringify(receipt));
    } else if (command === "probe-inventory-auth") {
      check(args.length === 1 && ["mlx", "candle_cuda"].includes(args[0]), "probe-inventory-auth BACKEND");
      console.log(JSON.stringify(probeCpuInventoryAuthority({ backend: args[0], runnerName: process.env.RUNNER_NAME, sceneWorksSha: process.env.S9_SCENEWORKS_SHA })));
    } else if (command === "build") {
      const [manifestPath, backend, runDir] = args;
      check(args.length === 3, "build SOURCE_FROZEN_MANIFEST BACKEND RUN_DIR");
      console.log(JSON.stringify(await buildDurableProduct({ manifestPath, backend, runDir, runnerName: process.env.RUNNER_NAME, expectedRunner: process.env.S9_EXPECTED_RUNNER, sceneWorksSha: process.env.S9_SCENEWORKS_SHA, inferenceSha: process.env.S9_INFERENCE_SHA })));
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
    } else if (command === "review") {
      check(args.length === 3 && ["mlx", "candle_cuda"].includes(args[2]), "review RUN_DIR OUTPUT_DIR BACKEND (S9_REVIEW_CASE_IDS JSON environment)");
      check(absolute(process.env.RUNNER_TEMP) && inside(process.env.RUNNER_TEMP, args[1]), "review artifact must stay in owned Actions temporary directory");
      const caseIds = JSON.parse(process.env.S9_REVIEW_CASE_IDS ?? "");
      const m = await json(path.join(args[0], "manifest.bound.json"));
      assertActionsIdentity({ backend: args[2], runnerName: process.env.RUNNER_NAME, expectedRunner: process.env.S9_EXPECTED_RUNNER, sceneWorksSha: process.env.S9_SCENEWORKS_SHA, inferenceSha: process.env.S9_INFERENCE_SHA, manifest: m, runDir: args[0], cacheRoot: process.env.S9_CACHE });
      const review = await exportReviewMedia({ runDir: args[0], outputDir: args[1], backend: args[2], caseIds });
      console.log(JSON.stringify({ campaignId: review.campaignId, backend: review.backend, selectedCases: review.selectedCases.length, files: review.files.length, completedAcceptance: false }));
    } else throw new Error("H3 S9 Actions: unknown command");
  })().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
