#!/usr/bin/env node
// Optional secondary-Mac facts and focused assertion evidence. Receipts do not gate pushes or PRs.
// Nothing here grants admission to dispatch a device job.
import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { closeSync, existsSync, lstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, renameSync, rmSync, writeFileSync, writeSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { loadRegistry, parseTestResult, validateAssertionReceipt, REGISTRY_PATH } from "./moss-consumer-assertions.mjs";
import { createUnitHfSandbox } from "./unit-hf-sandbox.mjs";
export const ASSERTION_DRIVER = "scripts/moss-consumer-assertions.mjs";

export const REPOSITORY = "SceneWorks/SceneWorks";
export const WORKFLOW = ".github/workflows/macos-mlx.yml";
export const CONTROLLER = "scripts/remote-macos-verification.mjs";
export const JOB = "manual-verification";
export const PROFILES = ["facts-and-projections", "consumer-assertions"];
export const CACHE_FLAGS = Object.freeze({ deployment_target: "26.2", debug: "0", test_debug: "0", build_override_debug: "0", test_build_override_debug: "0", build_override_strip: "none", test_build_override_strip: "none", incremental: "0", jobs: "2", cmake_jobs: "2", test_threads: "1", rustc_wrapper: "", unit_hf_isolated: true });
export const TARGET_REGISTRY = "tests/fixtures/rust_migration_contracts/training/target-registry.json";
export const TARGET_REGISTRY_WRITER = "regen_target_registry_fixture";
export const OUTPUTS = [
  "config/engine-capabilities/capabilities.mlx.json",
  "config/engine-capabilities/audio/capabilities.candle.json",
  "config/engine-capabilities/runtime/capabilities.mlx.json",
  "config/backend-capabilities/matrix.json",
  "config/manifests/builtin.preview-support.jsonc",
  "apps/web/src/data/previewSupport.json",
  "docs/generated/memory-matrix.json",
  "docs/generated/memory-matrix.md",
  TARGET_REGISTRY,
];
export const UNIT_SANDBOX = "scripts/unit-hf-sandbox.mjs";
export const CONTROL_FILES = Object.freeze([WORKFLOW, CONTROLLER, UNIT_SANDBOX, "scripts/lib/jsonc.mjs", ASSERTION_DRIVER, REGISTRY_PATH]);
const SHA = /^[0-9a-f]{40}$/;
const HASH = /^[0-9a-f]{64}$/;
const MAX_ARCHIVE = 128 * 1024 * 1024;
const fail = message => { throw new Error(`remote verification: ${message}`); };
const requireFact = (value, message) => { if (!value) fail(message); };
export const sha256 = value => createHash("sha256").update(value).digest("hex");
export const ownedRef = value => typeof value === "string" && /^(main|(?:story|feature)\/sc-[0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*)$/.test(value);
export const baseRef = value => value === "main" || (ownedRef(value) && value.startsWith("feature/"));
export const artifactName = (profile, sha, run, attempt) => `macos-verification-${profile}-${sha}-${run}-${attempt}`;

export function validateRequest(request) {
  requireFact(request.repository === REPOSITORY, "same-origin repository required");
  requireFact(PROFILES.includes(request.profile), "unknown manual profile");
  requireFact(SHA.test(request.source_sha) && SHA.test(request.inference_sha), "exact lowercase source/inference SHA required");
  requireFact(ownedRef(request.source_ref) && baseRef(request.base_ref), "owned story/feature source and main/feature base required");
  requireFact(request.legacy_device_inputs === false, "verification cannot share calibration/provisioning inputs");
  requireFact(!Object.hasOwn(request, "allow_nax_unit"), "device execution is not a facts/assertion option");
  return request;
}

export function validateMachine(machine, profile) {
  requireFact(machine.runner_name === "nax-macos-2" && machine.platform === "darwin" && machine.arch === "arm64", "selected physical secondary Mac required");
  const atLeast262 = version => /^\d+\.\d+(?:\.\d+)?$/.test(version ?? "") && (Number(version.split(".")[0]) > 26 || (Number(version.split(".")[0]) === 26 && Number(version.split(".")[1]) >= 2));
  requireFact(atLeast262(machine.runtime) && atLeast262(machine.sdk) && atLeast262(machine.deployment_target), "runtime, SDK and deployment target must be >=26.2");
  requireFact(Number(/^Apple M(\d+)\b/.exec(machine.chip ?? "")?.[1]) >= 5, "M5-or-newer NAX hardware required");
  requireFact(/\/Contents\/Developer\/?$/.test(machine.developer_dir ?? "") && path.isAbsolute(machine.metal_compiler ?? ""), "full Xcode and Metal compiler required");
  requireFact(PROFILES.includes(profile), "unknown machine profile");
  return machine;
}

function command(root, args, options = {}) {
  const result = spawnSync(args[0], args.slice(1), { cwd: root, encoding: "utf8", maxBuffer: MAX_ARCHIVE, ...options });
  if (result.error) throw result.error;
  requireFact(result.status === 0, `command failed (${result.status}): ${args[0]} ${args[1] ?? ""}`);
  return result.stdout.trim();
}
const git = (root, ...args) => command(root, ["git", ...args]);
const clean = root => git(root, "status", "--porcelain") === "";
const canonical = values => [...new Set(values)].sort();
const equal = (a, b) => { try { assert.deepEqual(a, b); return true; } catch { return false; } };

export function inferencePin(root) {
  const manifest = readFileSync(path.join(root, "Cargo.toml"), "utf8");
  const revisions = [...manifest.matchAll(/git\s*=\s*"https:\/\/github\.com\/SceneWorks\/inference"\s*,\s*rev\s*=\s*"([0-9a-f]{40})"/g)].map(match => match[1]);
  requireFact(revisions.length > 0 && new Set(revisions).size === 1, "Cargo inference revisions must agree");
  const lock = readFileSync(path.join(root, "Cargo.lock"), "utf8");
  const locked = [...lock.matchAll(/git\+https:\/\/github\.com\/SceneWorks\/inference\?rev=([0-9a-f]{40})#([0-9a-f]{40})/g)];
  requireFact(locked.length > 0 && locked.every(match => match[1] === revisions[0] && match[2] === revisions[0]), "Cargo lock must resolve the exact inference pin");
  return revisions[0];
}

// Transport spelling is not repository identity. Accept only these exact same-repo forms;
// do not use a permissive URL parser (credentials, escaping, queries and suffixes stay invalid).
export function canonicalRepositoryOrigin(origin) {
  requireFact(["https://github.com/SceneWorks/SceneWorks", "https://github.com/SceneWorks/SceneWorks.git", "git@github.com:SceneWorks/SceneWorks.git", "ssh://git@github.com/SceneWorks/SceneWorks.git"].includes(origin), "same-origin checkout required");
  return `https://github.com/${REPOSITORY}.git`;
}

export function checkoutContext(root, request, { executeGit = git, readPin = inferencePin, platform = process.platform } = {}) {
  validateRequest(request);
  const g = (...args) => executeGit(root, ...args);
  requireFact(g("rev-parse", "HEAD") === request.source_sha && g("status", "--porcelain") === "", "exact clean committed source required");
  canonicalRepositoryOrigin(g("remote", "get-url", "origin"));
  g("fetch", "--no-tags", "origin", `+refs/heads/${request.source_ref}:refs/remotes/origin/${request.source_ref}`, `+refs/heads/${request.base_ref}:refs/remotes/origin/${request.base_ref}`);
  g("merge-base", "--is-ancestor", "HEAD", `origin/${request.source_ref}`);
  requireFact(readPin(root) === request.inference_sha, "requested inference pin differs from checkout");
  const base_sha = g("rev-parse", `origin/${request.base_ref}`);
  const merge_base = g("merge-base", "HEAD", base_sha);
  requireFact(SHA.test(base_sha) && SHA.test(merge_base), "exact fetched base/merge-base required");
  const changed_files = canonical(g("diff", "--name-only", merge_base, "HEAD").split("\n").filter(Boolean));
  return { ...request, base_sha, merge_base, changed_files, cargo_config_sha256: sha256(readFileSync(path.join(root, ".cargo/config.toml"))) };
}

export function registryGenerationRecord(output) {
  const plain = output.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
  return { purpose: "generation-only", generator: TARGET_REGISTRY_WRITER, ...parseTestResult(plain, 0, TARGET_REGISTRY_WRITER, "green") };
}
function validateRegistryGeneration(record) {
  requireFact(equal(record, { purpose: "generation-only", generator: TARGET_REGISTRY_WRITER, passed: 1, failed: 0, ignored: 0, exit_code: 0 }), "target registry writer did not execute exactly once; generation is not validation proof");
}
export function validateTargetRegistry(value) {
  const object = row => row !== null && typeof row === "object" && !Array.isArray(row);
  requireFact(object(value) && value.schemaVersion === 1 && equal(Object.keys(value).sort(), ["schemaVersion", "targets"]) && Array.isArray(value.targets) && value.targets.length > 0, "invalid training registry wire root");
  const ids = new Set();
  for (const row of value.targets) {
    requireFact(object(row) && ["id", "name", "modality", "outputKind", "family", "baseModel", "kernel"].every(key => typeof row[key] === "string" && row[key].length > 0), "invalid training target identity");
    requireFact(!ids.has(row.id), "duplicate training target id"); ids.add(row.id);
    requireFact(["defaults", "limits", "ui"].every(key => object(row[key])) && object(row.defaults.advanced), "invalid training target objects");
    for (const key of ["rank", "alpha", "steps", "batchSize", "gradientAccumulation", "resolution", "saveEvery"]) requireFact(Number.isInteger(row.defaults[key]) && row.defaults[key] >= 0 && row.defaults[key] <= 0xffffffff, `invalid training config ${key}`);
    requireFact(Number.isSafeInteger(row.defaults.seed) && Number.isFinite(row.defaults.learningRate) && row.defaults.learningRate >= 0 && typeof row.defaults.optimizer === "string", "invalid training config scalar");
  }
  for (const base of ["moss_image_base", "moss_image_sft"]) {
    const row = value.targets.find(target => target.id === `${base}_lora`);
    requireFact(row?.baseModel === base && row.family === "moss-image" && row.kernel === "moss_image_lora" && row.modality === "image" && row.outputKind === "lora", "MOSS target missing or checkpoint identity changed");
    requireFact(row.baseModelRepo === `OpenMOSS-Team/MOSS-Image-${base === "moss_image_base" ? "Base" : "SFT"}`, "MOSS target checkpoint repository changed");
    requireFact(equal(row.limits.networkTypes, ["lora", "lokr"]) && equal(row.limits.mixedPrecisions, ["bf16", "f32"]), "MOSS target methods or precision contract changed");
    requireFact(row.defaults.advanced.mixedPrecision === "bf16" && row.defaults.advanced.timestepType === "moss_stage3_v1" && equal(row.defaults.advanced.modelOptions, { vae_posterior: "posterior_mean_v1" }), "MOSS target source/sampler policy changed");
  }
}

export function validateCapabilityFile(value, backend, kind, pin) {
  const revision = value?.generatedFrom?.inferenceRevision;
  requireFact(SHA.test(revision ?? "") && (pin === undefined || revision === pin), "capability file has wrong inference pin");
  if (kind === "runtime") {
    requireFact(value.schemaVersion === 2 && value.snapshot?.backend === backend && value.snapshot?.platform === (backend === "mlx" ? "macos" : "cuda"), "runtime schema/backend/platform mismatch");
    for (const key of ["generator_ids", "trainer_ids"]) requireFact(Array.isArray(value.snapshot[key]) && value.snapshot[key].every(id => typeof id === "string" && id.length > 0) && new Set(value.snapshot[key]).size === value.snapshot[key].length, `invalid runtime identity list: ${key}`);
    for (const key of ["generator_capabilities", "trainer_capabilities"]) requireFact(Array.isArray(value.snapshot[key]) && value.snapshot[key].every(row => typeof row.id === "string" && row.backend === backend), `invalid runtime descriptors: ${key}`);
    for (const key of ["modelMappings", "trainerMappings"]) requireFact(value[key] && typeof value[key] === "object" && !Array.isArray(value[key]), `invalid runtime mapping: ${key}`);
    requireFact(Array.isArray(value.videoModelMappings) && value.videoModelMappings.every(row => typeof row.modelId === "string" && typeof row.mode === "string" && Array.isArray(row.engineIds) && row.engineIds.every(id => typeof id === "string")), "invalid runtime video mapping list");
    requireFact(Array.isArray(value.workerCapabilities) && value.workerCapabilities.every(item => typeof item === "string"), "invalid worker capability list");
  } else {
    requireFact(value.backend === backend && Array.isArray(value.engines) && value.engines.length > 0 && value.engines.every(row => typeof row.id === "string" && row.id.length > 0 && typeof row.modality === "string" && typeof row.supportsPreview === "boolean"), "invalid native capability registry");
    if (kind === "audio") requireFact(value.registry === "audio", "wrong audio registry");
  }
}

export function validateFacts(files, pin, backends = ["mlx", "candle"]) {
  for (const backend of backends) {
    for (const prefix of ["", "audio/", "runtime/"]) {
      const name = `config/engine-capabilities/${prefix}capabilities.${prefix === "audio/" ? "candle" : backend}.json`;
      const value = JSON.parse(files(name));
      validateCapabilityFile(value, prefix === "audio/" ? "candle" : backend, prefix === "runtime/" ? "runtime" : prefix === "audio/" ? "audio" : "media", pin);
    }
  }
}

// Validate against authenticated GitHub responses AND locally recomputed source/scope, not receipt
// claims alone. The API adapter and expected local context are injectable for hermetic policy tests.
export async function authenticateReceipt({ receipt, files, context, api, runId, attempt, artifactId, controls, archiveDigest, assertionRegistry }) {
  validateRequest(context);
  requireFact(!Object.hasOwn(receipt, "allow_nax_unit"), "device permission is not receipt evidence");
  requireFact(receipt.schema_version === 1 && receipt.status === "passed" && receipt.repository === REPOSITORY, "invalid receipt schema/status/repository");
  for (const key of ["profile", "source_sha", "source_ref", "base_ref", "inference_sha", "merge_base", "legacy_device_inputs", "cargo_config_sha256"]) requireFact(receipt[key] === context[key], `receipt ${key} mismatch`);
  requireFact(equal(receipt.changed_files, context.changed_files), "receipt scope differs from committed source");
  requireFact(receipt.clean_before === true && typeof receipt.clean_after === "boolean" && (context.profile === "facts-and-projections" || receipt.clean_after === true), "dirty assertion source cannot pass");
  requireFact(receipt.run_id === runId && receipt.run_attempt === attempt && receipt.job === JOB && receipt.runner_name === "nax-macos-2", "receipt execution identity mismatch");
  validateMachine(receipt.machine, context.profile);
  validateCacheContext(receipt.cache_context, context.profile);
  requireFact(receipt.cache_context.sdk === receipt.machine.sdk && receipt.cache_context.cargo_config_sha256 === context.cargo_config_sha256, "cache SDK/config differs from observed source");
  const run = await api.json(`repos/${REPOSITORY}/actions/runs/${runId}/attempts/${attempt}`);
  requireFact(run.id === runId && run.run_attempt === attempt && run.event === "workflow_dispatch" && run.status === "completed" && run.conclusion === "success", "actual run did not complete successfully");
  requireFact(run.repository?.full_name === REPOSITORY && run.head_repository?.full_name === REPOSITORY && ownedRef(run.head_branch), "fork or external workflow ref rejected");
  requireFact(run.path?.split("@")[0] === WORKFLOW && run.head_sha === receipt.workflow_sha && receipt.workflow_ref === `${REPOSITORY}/${WORKFLOW}@refs/heads/${run.head_branch}`, "actual workflow revision/ref mismatch");
  for (const file of CONTROL_FILES) {
    const body = await api.file(file, run.head_sha);
    requireFact(sha256(body) === controls[file], `untrusted dispatcher implementation: ${file}`);
  }
  const sourceRef = await api.json(`repos/${REPOSITORY}/git/ref/heads/${context.source_ref}`);
  const sourceHistory = await api.json(`repos/${REPOSITORY}/compare/${context.source_sha}...${sourceRef.object.sha}`);
  requireFact(sourceHistory.merge_base_commit?.sha === context.source_sha, "source SHA is not on the owned source ref");
  requireFact(SHA.test(receipt.base_sha), "exact receipt base SHA required");
  const actualBase = await api.json(`repos/${REPOSITORY}/git/ref/heads/${context.base_ref}`);
  const baseHistory = await api.json(`repos/${REPOSITORY}/compare/${receipt.base_sha}...${actualBase.object.sha}`);
  requireFact(SHA.test(receipt.base_sha) && baseHistory.merge_base_commit?.sha === receipt.base_sha, "receipt base is not on the expected target ref");
  const jobs = await api.json(`repos/${REPOSITORY}/actions/runs/${runId}/attempts/${attempt}/jobs?per_page=100`);
  requireFact(jobs.total_count === jobs.jobs?.length && jobs.total_count <= 100, "incomplete job provenance");
  const owned = jobs.jobs.filter(job => job.name === JOB);
  requireFact(owned.length === 1 && owned[0].run_id === runId && owned[0].run_attempt === attempt && owned[0].head_sha === run.head_sha && owned[0].conclusion === "success", "actual verification job mismatch");
  requireFact(owned[0].runner_name === "nax-macos-2" && ["self-hosted", "macOS", "ARM64", "rw-llm"].every(label => owned[0].labels?.includes(label)), "actual secondary runner mismatch");
  const requiredSteps = ["Run exact remote verification", "Upload remote verification artifact"];
  for (const name of requiredSteps) requireFact(owned[0].steps?.some(step => step.name === name && step.conclusion === "success"), `required actual step missing: ${name}`);
  requireFact(jobs.jobs.every(job => job.name === JOB || job.conclusion === "skipped"), "legacy jobs unexpectedly ran for manual verification");
  const artifact = await api.json(`repos/${REPOSITORY}/actions/artifacts/${artifactId}`);
  requireFact(/^sha256:[0-9a-f]{64}$/.test(archiveDigest ?? "") && artifact.digest === archiveDigest, "authenticated artifact archive changed");
  requireFact(artifact.id === artifactId && artifact.expired === false && artifact.name === artifactName(context.profile, context.source_sha, runId, attempt), "artifact identity/expiry mismatch");
  requireFact(artifact.workflow_run?.id === runId && artifact.workflow_run?.head_sha === run.head_sha && artifact.workflow_run?.repository_id === run.repository.id && artifact.workflow_run?.head_repository_id === run.head_repository.id, "artifact belongs to a different run/repository");
  requireFact(receipt.command?.join("\0") === (context.profile === "consumer-assertions" ? ["consumer-assertions", ASSERTION_DRIVER] : ["facts-and-projections"]).join("\0"), "receipt command mismatch");
  const expected = context.profile === "consumer-assertions" ? ["assertions.json", "assertions.log"] : [...OUTPUTS.map(file => `payload/${file}`), "facts.log"];
  requireFact(equal(Object.keys(receipt.files ?? {}).sort(), [...expected].sort()) && equal(Object.keys(files).sort(), ["receipt.json", ...expected].sort()), "unexpected or missing artifact members");
  for (const file of expected) requireFact(HASH.test(receipt.files[file]) && sha256(files[file]) === receipt.files[file], `artifact member digest mismatch: ${file}`);
  if (context.profile === "consumer-assertions") {
    const spec = assertionRegistry ?? loadRegistry();
    requireFact(spec.registry_sha256 === controls[REGISTRY_PATH], "consumer case registry differs from authenticated source");
    validateAssertionReceipt(JSON.parse(files["assertions.json"].toString()), spec, context.source_sha, controls[ASSERTION_DRIVER]);
  } else {
    for (const file of OUTPUTS.filter(file => /\.jsonc?$/.test(file))) JSON.parse(stripJsoncComments(files[`payload/${file}`].toString()));
    for (const [index, kind] of ["media", "audio", "runtime"].entries()) validateCapabilityFile(JSON.parse(files[`payload/${OUTPUTS[index]}`]), kind === "audio" ? "candle" : "mlx", kind, context.inference_sha);
    validateRegistryGeneration(receipt.target_registry_generation);
    validateTargetRegistry(JSON.parse(files[`payload/${TARGET_REGISTRY}`]));
  }
  return artifact;
}

// ZIP is data, never executable code. Read it using isolated Python, reject traversal, symlinks,
// duplicates and expansion bombs BEFORE returning any member. No archive script is ever executed.
export function readArchive(bytes, digest) {
  requireFact(bytes.length <= MAX_ARCHIVE && digest === `sha256:${sha256(bytes)}`, "authenticated artifact archive digest mismatch");
  const scratch = mkdtempSync(path.join(os.tmpdir(), "remote-macos-archive-"));
  try {
    const archive = path.join(scratch, "artifact.zip"); writeFileSync(archive, bytes);
    const script = String.raw`import base64,json,pathlib,stat,sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
 assert len(z.infolist())<=64, "too many artifact members"
 names=set(); total=0; result={}
 for i in z.infolist():
  p=pathlib.PurePosixPath(i.filename)
  assert i.filename and not p.is_absolute() and ".." not in p.parts and "\\" not in i.filename and ":" not in i.filename, "unsafe member"
  assert i.filename not in names, "duplicate member"
  names.add(i.filename)
  mode=i.external_attr>>16
  assert not stat.S_ISLNK(mode), "symlink member"
  if i.is_dir(): continue
  assert not stat.S_IFMT(mode) or stat.S_ISREG(mode), "non-file member"
  total+=i.file_size
  assert total<=134217728 and i.file_size<=67108864, "expanded artifact too large"
  result[i.filename]=base64.b64encode(z.read(i)).decode()
 print(json.dumps(result))`;
    const python = process.platform === "win32" ? "python" : "python3";
    const values = JSON.parse(command(scratch, [python, "-I", "-c", script, archive], { maxBuffer: MAX_ARCHIVE * 2 }));
    return Object.fromEntries(Object.entries(values).map(([name, value]) => [name, Buffer.from(value, "base64")]));
  } finally { rmSync(scratch, { recursive: true, force: true }); }
}

function githubApi(root) {
  return {
    json: async endpoint => JSON.parse(command(root, ["gh", "api", "--hostname", "github.com", endpoint])),
    file: async (name, ref) => {
      const result = JSON.parse(command(root, ["gh", "api", "--hostname", "github.com", `repos/${REPOSITORY}/contents/${name}?ref=${ref}`]));
      requireFact(result.encoding === "base64" && typeof result.content === "string", "GitHub source object missing");
      return Buffer.from(result.content, "base64");
    },
    archive: id => {
      const result = spawnSync("gh", ["api", "--hostname", "github.com", `repos/${REPOSITORY}/actions/artifacts/${id}/zip`], { cwd: root, maxBuffer: MAX_ARCHIVE });
      requireFact(!result.error && result.status === 0, "artifact download failed");
      return result.stdout;
    },
  };
}

function atomicWrite(file, bytes) {
  mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.import-${process.pid}`;
  writeFileSync(temporary, bytes, { flag: "wx" });
  try { renameSync(temporary, file); }
  finally { if (existsSync(temporary)) rmSync(temporary); }
}

export async function importReceipt(root, request, { runId, attempt, artifactId, api = githubApi(root), contextFor = checkoutContext, commonDirectory = directory => path.resolve(directory, git(directory, "rev-parse", "--git-common-dir")) }) {
  const context = contextFor(root, request);
  const controls = Object.fromEntries(CONTROL_FILES.map(file => [file, sha256(readFileSync(path.join(root, file)))]));
  const metadata = await api.json(`repos/${REPOSITORY}/actions/artifacts/${artifactId}`);
  requireFact(Number.isSafeInteger(metadata.size_in_bytes) && metadata.size_in_bytes > 0 && metadata.size_in_bytes <= MAX_ARCHIVE, "invalid artifact size");
  const files = readArchive(await api.archive(artifactId), metadata.digest);
  requireFact(files["receipt.json"], "receipt missing");
  const receipt = JSON.parse(files["receipt.json"]);
  await authenticateReceipt({ receipt, files, context, api, runId, attempt, artifactId, controls, archiveDigest: metadata.digest, assertionRegistry: request.profile === "consumer-assertions" ? loadRegistry(root) : undefined });
  // Authentication can take time; reject local edits or ref/context changes before writing anything.
  const current = contextFor(root, request);
  requireFact(equal(current.changed_files, context.changed_files) && current.merge_base === context.merge_base, "local source/scope changed during import");
  if (request.profile === "consumer-assertions") {
    const common = commonDirectory(root);
    const destination = path.join(common, "sceneworks-consumer-assertions", request.source_sha, `${runId}-${attempt}-${artifactId}.json`);
    atomicWrite(destination, Buffer.from(`${JSON.stringify({ artifact_id: artifactId, archive_digest: metadata.digest, receipt, assertions: JSON.parse(files["assertions.json"].toString()) }, null, 2)}\n`));
  } else {
    for (const file of OUTPUTS) {
      let cursor = root;
      for (const component of file.split("/")) { cursor = path.join(cursor, component); requireFact(!existsSync(cursor) || !lstatSync(cursor).isSymbolicLink(), "refusing a symlink destination"); }
    }
    for (const file of OUTPUTS) atomicWrite(path.join(root, file), files[`payload/${file}`]);
  }
  return { profile: request.profile, source_sha: request.source_sha, artifact_id: artifactId };
}

function requestFromEnvironment(env) {
  return validateRequest({ repository: env.GITHUB_REPOSITORY, profile: env.VERIFICATION_PROFILE, source_sha: env.VERIFICATION_SOURCE_SHA, source_ref: env.VERIFICATION_SOURCE_REF, base_ref: env.VERIFICATION_BASE_REF, inference_sha: env.VERIFICATION_INFERENCE_SHA, legacy_device_inputs: env.VERIFICATION_LEGACY_INPUTS === "true" });
}

export function dispatchIdentity(env) {
  const prefix = `${REPOSITORY}/${WORKFLOW}@refs/heads/`;
  requireFact(SHA.test(env.VERIFICATION_WORKFLOW_SHA) && env.VERIFICATION_WORKFLOW_REF?.startsWith(prefix) && ownedRef(env.VERIFICATION_WORKFLOW_REF.slice(prefix.length)), "owned dispatcher identity required");
  const run_id = Number(env.GITHUB_RUN_ID), run_attempt = Number(env.GITHUB_RUN_ATTEMPT);
  requireFact(Number.isSafeInteger(run_id) && run_id > 0 && Number.isSafeInteger(run_attempt) && run_attempt > 0, "exact positive run/attempt identity required");
  return { workflow_sha: env.VERIFICATION_WORKFLOW_SHA, workflow_ref: env.VERIFICATION_WORKFLOW_REF, run_id, run_attempt };
}

function machineFromHost(env) {
  const hardware = JSON.parse(command(process.cwd(), ["system_profiler", "SPHardwareDataType", "-json"]));
  return { runner_name: env.RUNNER_NAME, platform: process.platform, arch: process.arch, runtime: command(process.cwd(), ["sw_vers", "-productVersion"]), sdk: command(process.cwd(), ["xcrun", "--sdk", "macosx", "--show-sdk-version"]), deployment_target: env.MACOSX_DEPLOYMENT_TARGET, chip: hardware.SPHardwareDataType?.[0]?.chip_type, developer_dir: command(process.cwd(), ["xcode-select", "-p"]), metal_compiler: command(process.cwd(), ["xcrun", "--find", "metal"]) };
}

async function runLogged(root, args, log, env, append = false) {
  const output = openSync(log, append ? "a" : "w");
  try {
    writeSync(output, `${JSON.stringify(args)}\n`);
    const child = spawn(args[0], args.slice(1), { cwd: root, env, stdio: ["ignore", output, output] });
    const code = await new Promise((resolve, reject) => { child.on("error", reject); child.on("exit", resolve); });
    // Evidence is best-effort: a diagnostics problem must never replace or mask the build's own failure.
    if (code !== 0 && args[0] === "cargo") { try { writeSync(output, buildDiagnostics(root, env, undefined, readFileSync(log, "utf8").slice(-8 * 1024 * 1024))); } catch (error) { writeSync(output, `\n(build diagnostics unavailable: ${error.message})\n`); } }
    requireFact(code === 0, `${args[0]} ${args[1] ?? ""} exited ${code}; retained log: ${path.basename(log)}`);
  } finally { closeSync(output); }
}

// Read-only, bounded evidence for a failed native build. Fixed commands only; never changes the
// verdict. Records the compiler/SDK/linker identity and the target's proc-macro dylibs so a
// toolchain/cache defect can be told apart from a source error.
export function buildDiagnostics(root, env, run = (cmd, args) => spawnSync(cmd, args, { cwd: root, env, encoding: "utf8", timeout: 20000, maxBuffer: 4 * 1024 * 1024 }), originalLog = null) {
  const lines = ["", "=== remote verification build diagnostics (read-only) ==="];
  const probe = (cmd, args) => { const r = run(cmd, args); lines.push(`$ ${[cmd, ...args].join(" ")}`, `${(r.stdout ?? "").trim()}${r.stderr ? `\n${r.stderr.trim()}` : ""}`.slice(0, 2000), `exit ${r.status}`); };
  probe("rustc", ["-vV"]);
  probe("cargo", ["-V"]);
  probe("xcrun", ["--show-sdk-path"]);
  probe("xcrun", ["--show-sdk-version"]);
  probe("ld", ["-v"]);
  probe("xcode-select", ["-p"]);
  probe("rustc", ["--print", "sysroot"]);
  for (const key of ["SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN", "TMPDIR"]) lines.push(`${key}=${env[key] ?? ""}`);
  const deps = path.join(env.CARGO_TARGET_DIR ?? path.join(root, "target"), "debug", "deps");
  // Inspect proc-macro dylibs as DATA only (file/otool/stat); nothing in the target is executed.
  // Bounded to 12 libraries; LC_BUILD_VERSION/cputype only from otool, never full output.
  probe("/bin/sh", ["-c", [
    'n=0; for f in "$1"/*.dylib; do [ -e "$f" ] || { echo "no dylibs in $1"; break; }',
    '  n=$((n+1)); [ "$n" -gt 12 ] && { echo "...truncated at 12"; break; }',
    '  stat -f "%z %N" "$f"; file "$f"',
    '  otool -hv "$f" 2>&1 | sed -n "3,4p"',
    '  otool -l "$f" 2>&1 | grep -A4 -E "LC_BUILD_VERSION" | grep -E "platform|minos|sdk" | head -n 3',
    '  otool -L "$f" 2>&1 | sed -n "2,6p"',
    // Where (if anywhere) rustc metadata landed: segment/section of the .rustc payload.
    '  otool -l "$f" 2>&1 | grep -B1 -A2 -E "sectname \\.rustc" | grep -E "sectname|segname|size" | head -n 4 || true',
    '  echo "rustc-metadata-sections: $(otool -l "$f" 2>/dev/null | grep -c "sectname .rustc")"',
    // Header words of the metadata payload (expected "rust" magic + format version) and its size/offset,
    // plus whether the exported rust_metadata_* symbol survived linking/strip. Read-only, 2 rows only.
    '  otool -l "$f" 2>&1 | grep -A4 -E "sectname \\.rustc" | grep -E "^ *(size|offset)" | head -n 2',
    '  otool -s __DATA .rustc "$f" 2>&1 | sed -n "3,4p"',
    '  echo "rust_metadata exports: $(nm -gU "$f" 2>/dev/null | grep -c rust_metadata_)"',
    'done',
  ].join("\n"), "sh", deps]);
  probe("xcrun", ["--find", "ld-classic"]);
  probe("/bin/sh", ["-c", "ls -d /Applications/Xcode*.app 2>&1 | head -n 10"]);
  probe("rustup", ["toolchain", "list"]);
  // Evidence from the ORIGINAL verbose cargo execution only; nothing is re-run or recompiled.
  if (typeof originalLog === "string") lines.push(...invocationEvidence(originalLog));
  return `${lines.join("\n")}\n`;
}

/** The last rustc invocations of the original verbose cargo run, plus the proc-macro extern. */
export function invocationEvidence(log, exists = existsSync) {
  // Cargo colours "Running" in CI logs; strip ANSI before matching. Cargo's own "Caused by:
  // process didn't exit successfully: `rustc …`" line is also an invocation of the failed crate.
  const plain = log.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
  const invocations = plain.split("\n").filter(line => (line.includes("Running `") || line.includes("process didn't exit successfully: `")) && line.includes("--crate-name"));
  const lines = [`original verbose cargo log: ${invocations.length} rustc invocation line(s); showing last ${Math.min(3, invocations.length)}`];
  for (const line of invocations.slice(-3)) lines.push(line.length > 3000 ? `${line.slice(0, 3000)} …[truncated ${line.length - 3000} chars]` : line);
  // Cargo fingerprints strip settings for host units. An actual invocation (not a Fresh line)
  // is required as evidence that the changed build-override recipe rebuilt the derive macro.
  const macros = invocations.filter(line => /--crate-type[ =]proc-macro(?:\s|$)/.test(line));
  for (const line of macros.filter((line, index) => index >= macros.length - 12 || /--crate-name bytemuck_derive(?:\s|$)/.test(line))) {
    const crate = /--crate-name (\S+)/.exec(line)?.[1];
    const strip = [...line.matchAll(/(?:-C\s+|-C)strip=(\S+)/g)].map(match => match[1]);
    lines.push(`original proc-macro invocation: ${crate}; strip=${strip.length ? strip.join(",") : "absent"}`);
  }
  if (!macros.some(line => /--crate-name bytemuck_derive(?:\s|$)/.test(line))) lines.push("bytemuck_derive proc-macro invocation: NOT REBUILT in captured original log (no strip hypothesis verdict)");
  const failing = [...invocations].reverse().find(line => /--crate-name bytemuck(\s|$)/.test(line));
  const extern = failing && /--extern bytemuck_derive=(\S+?)`?(\s|$)/.exec(failing);
  lines.push(!failing ? "bytemuck invocation: not found in original log"
    : !extern ? "bytemuck invocation: NO --extern bytemuck_derive argument"
    : `bytemuck invocation: --extern bytemuck_derive=${extern[1]} (exists: ${exists(extern[1])})`);
  return lines;
}

// This protects ONLY this workflow's dedicated target directory, not the physical host/GPU.
// Managed completion/failure releases it. Abrupt SIGKILL/job loss may leave an unknown owner;
// refuse that state until the operator verifies release. Never auto-steal a stale directory.
export function acquireTargetLease(directory, owner) {
  requireFact(owner.runner_name === "nax-macos-2" && Number.isSafeInteger(owner.run_id) && owner.run_id > 0 && Number.isSafeInteger(owner.run_attempt) && owner.run_attempt > 0, "cache owner identity invalid");
  mkdirSync(directory, { recursive: true });
  requireFact(!lstatSync(directory).isSymbolicLink(), "shared target cannot be a symlink");
  const lease = path.join(directory, ".remote-verification-owner");
  try { mkdirSync(lease); } catch (error) { if (error.code === "EEXIST") fail("dedicated target cache already leased; verify prior run/process release, never steal a lock"); throw error; }
  const token = randomUUID();
  writeFileSync(path.join(lease, "owner.json"), JSON.stringify({ ...owner, token, pid: process.pid }));
  return { release() {
    const current = JSON.parse(readFileSync(path.join(lease, "owner.json")));
    requireFact(current.token === token && current.run_id === owner.run_id && current.run_attempt === owner.run_attempt && current.pid === process.pid, "refusing another run's target lease");
    rmSync(lease, { recursive: true });
  } };
}

// Cargo 1.97.1 (c980f486) gives CARGO_INCREMENTAL=1 precedence over every profile/config.
// Incremental is part of Cargo's Profile fingerprint and unit metadata; the lease is unchanged.
export function cacheFlags(profile = "facts-and-projections") {
  requireFact(PROFILES.includes(profile), "unknown cache recipe profile");
  return { ...CACHE_FLAGS, incremental: profile === "consumer-assertions" ? "1" : "0" };
}

export function validateCacheContext(context, profile = "facts-and-projections") {
  requireFact(context?.serialized === true && context.released === true && context.runner_name === "nax-macos-2", "target cache was not exclusively owned and released");
  requireFact(context.toolchain === "1.97.1" && context.host === "aarch64-apple-darwin" && HASH.test(context.rustc_sha256 ?? "") && HASH.test(context.cargo_config_sha256 ?? "") && HASH.test(context.target_directory_sha256 ?? "") && /^\d+\.\d+(?:\.\d+)?$/.test(context.sdk ?? ""), "compiler/cache context mismatch");
  requireFact(equal(context.build_flags, cacheFlags(profile)), "inconsistent cache build flags");
  const { context_sha256, ...body } = context;
  requireFact(HASH.test(context_sha256 ?? "") && sha256(JSON.stringify(body)) === context_sha256, "cache context fingerprint mismatch");
}

export async function executeRemote(root, output, env = process.env) {
  const request = requestFromEnvironment(env), identity = dispatchIdentity(env);
  const machine = validateMachine(machineFromHost(env), request.profile);
  requireFact(path.isAbsolute(env.GITHUB_WORKSPACE ?? "") && path.resolve(root) === path.join(env.GITHUB_WORKSPACE, `candidate-${identity.run_id}-${identity.run_attempt}`), "fresh per-run/attempt candidate required");
  const target = path.resolve(env.GITHUB_WORKSPACE, "..", "remote-verification-target");
  requireFact(path.resolve(env.CARGO_TARGET_DIR ?? "") === target, "dedicated serialized target cache required");
  requireFact(!env.RUSTFLAGS && !env.CARGO_ENCODED_RUSTFLAGS, "nonstandard compiler flags require a separate validated cache context");
  const rustc = command(root, [path.join(os.homedir(), ".cargo/bin/rustup"), "run", "1.97.1", "rustc", "-vV"]);
  requireFact(/^release: 1\.97\.1$/m.test(rustc) && /^host: aarch64-apple-darwin$/m.test(rustc), "native pinned compiler required");
  const cache = { serialized: true, released: true, runner_name: env.RUNNER_NAME, toolchain: "1.97.1", host: "aarch64-apple-darwin", sdk: machine.sdk, target_directory_sha256: sha256(target), rustc_sha256: sha256(rustc), cargo_config_sha256: sha256(readFileSync(path.join(root, ".cargo/config.toml"))), build_flags: cacheFlags(request.profile) };
  const lease = acquireTargetLease(target, { ...identity, runner_name: env.RUNNER_NAME, source_sha: request.source_sha });
  let receipt;
  try { receipt = await executeRemoteWithinLease(root, output, env); }
  finally { lease.release(); }
  receipt.cache_context = { ...cache, context_sha256: sha256(JSON.stringify(cache)) };
  validateCacheContext(receipt.cache_context, request.profile);
  writeFileSync(path.join(output, "receipt.json"), `${JSON.stringify(receipt, null, 2)}\n`);
  return receipt;
}

// The verifier's build recipe. HYPOTHESIS TEST (run 39553468673): proc macros built with Cargo's
// implicit `-C strip=debuginfo` (debug=0) may be unreadable to rustc 1.97.1 on Xcode 27.
// NOT CONFIRMED. Cargo at c980f486 (Rust 1.97.1) documents BUILD_OVERRIDE_STRIP=none and
// hashes Profile.strip into both fingerprints and unit metadata; old stripped units cannot be reused.
// Disable strip ONLY for build-override units (build scripts, proc macros and their dependencies); the
// regular dev/test units keep Cargo's defaults. Recorded in CACHE_FLAGS so the receipt binds it.
export function remoteBuildEnv(env, profile = "facts-and-projections") {
  const runEnv = { ...env, PATH: `${path.join(os.homedir(), ".cargo", "bin")}${path.delimiter}${env.PATH ?? ""}`, CI: "true", RUSTUP_TOOLCHAIN: "1.97.1", MACOSX_DEPLOYMENT_TARGET: "26.2", CARGO_NET_OFFLINE: "false", CARGO_BUILD_JOBS: "2", CMAKE_BUILD_PARALLEL_LEVEL: "2", CARGO_PROFILE_DEV_DEBUG: "0", CARGO_PROFILE_TEST_DEBUG: "0", CARGO_PROFILE_DEV_BUILD_OVERRIDE_DEBUG: "0", CARGO_PROFILE_TEST_BUILD_OVERRIDE_DEBUG: "0", CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP: "none", CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP: "none", RUST_TEST_THREADS: "1", RUSTC_WRAPPER: "" };
  runEnv.CARGO_INCREMENTAL = cacheFlags(profile).incremental;
  delete runEnv.REGEN_FIXTURE;
  delete runEnv.CARGO_BUILD_TARGET;
  delete runEnv.CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER;
  return runEnv;
}

function validateControllerCheckout(root) {
  const controlRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  for (const file of CONTROL_FILES) requireFact(readFileSync(path.join(root, file)).equals(readFileSync(path.join(controlRoot, file))), `dispatcher and candidate implementation differ: ${file}`);
}

async function executeRemoteWithinLease(root, output, env = process.env) {
  validateControllerCheckout(root);
  const request = requestFromEnvironment(env);
  const machine = validateMachine(machineFromHost(env), request.profile);
  const context = checkoutContext(root, request);
  const identity = dispatchIdentity(env);
  requireFact(!existsSync(output), "output directory must be new"); mkdirSync(output, { recursive: true });
  const receipt = { schema_version: 1, status: "running", ...context, ...identity, repository: REPOSITORY, job: JOB, runner_name: env.RUNNER_NAME, machine, clean_before: true, files: {} };
  const runEnv = { ...remoteBuildEnv(env, request.profile), ...createUnitHfSandbox(root, env).env };
  if (request.profile === "facts-and-projections") {
    // Keep committed Windows provenance as declared; never restamp it or add a pin-currency gate.
    // Changed Candle semantics still require genuine producer facts before the final matrix can pass.
    validateFacts(file => readFileSync(path.join(root, file), "utf8"), undefined, ["candle"]);
    const scratch = mkdtempSync(path.join(os.tmpdir(), "remote-macos-facts-"));
    const log = path.join(output, "facts.log");
    try {
      await runLogged(root, ["cargo", "run", "--locked", "--verbose", "-p", "sceneworks-worker", "--bin", "dump-engine-capabilities", "--", scratch], log, runEnv);
      for (const file of OUTPUTS.slice(0, 3)) {
        const from = path.join(scratch, file.slice("config/engine-capabilities/".length));
        const bytes = readFileSync(from); requireFact(JSON.parse(bytes).generatedFrom?.inferenceRevision === request.inference_sha, "dump pin differs");
        writeFileSync(path.join(root, file), bytes);
      }
      validateFacts(file => readFileSync(path.join(root, file), "utf8"));
      validateFacts(file => readFileSync(path.join(root, file), "utf8"), request.inference_sha, ["mlx"]);
      const matrix = path.join(scratch, "backend-matrix.json");
      await runLogged(root, ["cargo", "run", "--locked", "-p", "sceneworks-core", "--bin", "dump-backend-capability-matrix", "--", matrix], log, runEnv, true);
      writeFileSync(path.join(root, OUTPUTS[3]), readFileSync(matrix));
      await runLogged(root, ["npm", "--prefix", "apps/web", "run", "gen:preview-support"], log, runEnv, true);
      await runLogged(root, ["npm", "run", "generate:memory-matrix"], log, runEnv, true);
      // GENERATION ONLY: the existing ignored writer is not a validation test or mutation proof.
      // Its flag is confined to this one child; all later commands keep the declared profile env.
      const writerLogStart = readFileSync(log).length;
      await runLogged(root, ["cargo", "test", "--locked", "-p", "sceneworks-core", "--test", "training_contracts", TARGET_REGISTRY_WRITER, "--", "--exact", "--ignored"], log, { ...runEnv, REGEN_FIXTURE: "1" }, true);
      receipt.target_registry_generation = registryGenerationRecord(readFileSync(log).subarray(writerLogStart).toString("utf8"));
      validateRegistryGeneration(receipt.target_registry_generation);
      validateTargetRegistry(JSON.parse(readFileSync(path.join(root, TARGET_REGISTRY), "utf8")));
      requireFact(git(root, "diff", "--name-only").split("\n").filter(Boolean).every(file => OUTPUTS.includes(file)), "producer modified non-derived source");
      for (const file of OUTPUTS) atomicWrite(path.join(output, "payload", file), readFileSync(path.join(root, file)));
      receipt.command = ["facts-and-projections"];
    } finally { rmSync(scratch, { recursive: true, force: true }); }
  } else if (request.profile === "consumer-assertions") {
    const scratch = mkdtempSync(path.join(os.tmpdir(), "moss-consumer-assertions-"));
    const run = path.join(scratch, "run");
    try {
      await runLogged(root, [process.execPath, path.join(root, ASSERTION_DRIVER), root, run], path.join(output, "assertions.log"), runEnv);
    } finally {
      const result = path.join(run, "assertions.json");
      if (existsSync(result)) writeFileSync(path.join(output, "assertions.json"), readFileSync(result));
      rmSync(scratch, { recursive: true, force: true });
    }
    requireFact(clean(root) && git(root, "rev-parse", "HEAD") === request.source_sha, "consumer driver left changed source");
    const spec = loadRegistry(root);
    validateAssertionReceipt(JSON.parse(readFileSync(path.join(output, "assertions.json"))), spec, request.source_sha, sha256(readFileSync(path.join(root, ASSERTION_DRIVER))));
    receipt.command = ["consumer-assertions", ASSERTION_DRIVER];
  }
  receipt.clean_after = clean(root);
  receipt.status = "passed";
  const members = request.profile === "consumer-assertions" ? ["assertions.json", "assertions.log"] : [...OUTPUTS.map(file => `payload/${file}`), "facts.log"];
  receipt.files = Object.fromEntries(members.map(file => [file, sha256(readFileSync(path.join(output, file)))]));
  return receipt;
}

async function main() {
  const [mode, ...args] = process.argv.slice(2);
  if (mode === "request") {
    requestFromEnvironment(process.env);
    dispatchIdentity(process.env);
  } else if (mode === "check") {
    const request = requestFromEnvironment(process.env);
    dispatchIdentity(process.env);
    validateMachine(machineFromHost(process.env), request.profile);
    validateControllerCheckout(path.resolve(args[0]));
    checkoutContext(path.resolve(args[0]), request);
  } else if (mode === "run") {
    await executeRemote(path.resolve(args[0]), path.resolve(args[1]));

  } else if (mode === "import") {
    requireFact(args.length === 8, "import requires ROOT PROFILE SOURCE_REF BASE_REF SOURCE_SHA RUN_ID ATTEMPT ARTIFACT_ID");
    const [root, profile, source_ref, base_ref, source_sha, run, attempt, artifact] = args;
    for (const value of [run, attempt, artifact]) requireFact(/^[1-9][0-9]*$/.test(value) && Number.isSafeInteger(Number(value)), "positive exact execution ids required");
    const request = { repository: REPOSITORY, profile, source_ref, base_ref, source_sha, inference_sha: inferencePin(path.resolve(root)), legacy_device_inputs: false };
    console.log(JSON.stringify(await importReceipt(path.resolve(root), request, { runId: Number(run), attempt: Number(attempt), artifactId: Number(artifact) })));
  } else fail("expected request, check, run or import");
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => { console.error(error.message); process.exitCode = 1; });
