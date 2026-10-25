import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, utimesSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const nestedTestArgs = target => ["--test", "--test-reporter=spec", `--test-name-pattern=^${target.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`, fileURLToPath(import.meta.url)];
test("nested Node probes require spec reporter and exact test scope", () => {
  const target = "fixture (+) exact", args = nestedTestArgs(target);
  assert.equal(args[0], "--test"); assert.equal(args.filter(arg => arg === "--test-reporter=spec").length, 1);
  assert.equal(args.length, 4); assert.equal(args[3], fileURLToPath(import.meta.url));
  const pattern = new RegExp(args[2].slice("--test-name-pattern=".length));
  assert.ok(pattern.test(target)); assert.ok(!pattern.test(target + " extra")); assert.ok(!pattern.test("unrelated"));
});
const modulePath = process.env.REMOTE_MACOS_CONTROLLER_UNDER_TEST ?? path.join(ROOT, "scripts/remote-macos-verification.mjs");
const policy = await import(pathToFileURL(modulePath));
const { REPOSITORY, WORKFLOW, CONTROLLER, JOB, OUTPUTS, CACHE_FLAGS, ASSERTION_DRIVER, CONTROL_FILES, UNIT_SANDBOX, TARGET_REGISTRY, TARGET_REGISTRY_WRITER, acquireTargetLease, validateCacheContext, artifactName, authenticateReceipt, checkoutContext, dispatchIdentity, importReceipt, inferencePin, readArchive, sha256, validateCapabilityFile, validateFacts, validateMachine, validateRequest } = policy;
const { loadRegistry, TESTS: ASSERTION_TESTS, MUTABLE_FILES, REGISTRY_PATH } = await import("./moss-consumer-assertions.mjs");
const SOURCE = "a".repeat(40), PIN = "b".repeat(40), BASE = "c".repeat(40), DISPATCHER = "d".repeat(40);
const sourceRef = "story/sc-26585-epic-26584-moss-image-training", base = "feature/sc-26584-moss-image-training";
const machine = () => ({ runner_name: "nax-macos-2", platform: "darwin", arch: "arm64", runtime: "26.5.2", sdk: "26.5", deployment_target: "26.2", chip: "Apple M5 Max", developer_dir: "/Applications/Xcode.app/Contents/Developer", metal_compiler: "/Applications/Xcode.app/Contents/Developer/usr/bin/metal" });
const request = (profile = "consumer-assertions") => ({ repository: REPOSITORY, profile, source_sha: SOURCE, source_ref: sourceRef, inference_sha: PIN, base_ref: base, legacy_device_inputs: false });
function cacheContext(profile = "facts-and-projections") {
  const value = { serialized: true, released: true, runner_name: "nax-macos-2", toolchain: "1.97.1", host: "aarch64-apple-darwin", sdk: "26.5", target_directory_sha256: "1".repeat(64), rustc_sha256: "2".repeat(64), cargo_config_sha256: sha256(readFileSync(path.join(ROOT, ".cargo/config.toml"))), build_flags: policy.cacheFlags(profile) };
  return { ...value, context_sha256: sha256(JSON.stringify(value)) };
}
function syntheticAssertionResult() {
  const spec = loadRegistry(ROOT);
  const green = key => ({ test_id: key, passed: 1, failed: 0, ignored: 0, exit_code: 0 });
  const hashes = Object.fromEntries(MUTABLE_FILES.map(file => [file, "1".repeat(64)]));
  return { schema_version: 2, status: "passed", elapsed_seconds: 1, source_sha: SOURCE, registry_sha256: spec.registry_sha256, driver_sha256: sha256(readFileSync(path.join(ROOT, ASSERTION_DRIVER))), test_ids: Object.keys(ASSERTION_TESTS), baseline: Object.keys(ASSERTION_TESTS).map(green), final_green: Object.keys(ASSERTION_TESTS).map(green), cases: spec.cases.map(item => ({ id: item.id, killed: true, restored: true, red: item.tests.map(test_id => ({ test_id, passed: 0, failed: 1, ignored: 0, exit_code: 101 })), green: item.tests.map(green) })), source_before: hashes, source_after: { ...hashes }, restored: true, clean_after: true, head_after: SOURCE };
}
function targetRegistry() {
  return { schemaVersion: 1, targets: ["moss_image_base", "moss_image_sft"].map(base => ({
    id: `${base}_lora`, name: base, modality: "image", outputKind: "lora", family: "moss-image", baseModel: base,
    baseModelRepo: `OpenMOSS-Team/MOSS-Image-${base === "moss_image_base" ? "Base" : "SFT"}`, kernel: "moss_image_lora",
    defaults: { rank: 16, alpha: 16, learningRate: 0.0001, steps: 1000, batchSize: 1, gradientAccumulation: 1, resolution: 1024, saveEvery: 250, seed: 42, optimizer: "adamw", advanced: { mixedPrecision: "bf16", timestepType: "moss_stage3_v1", modelOptions: { vae_posterior: "posterior_mean_v1" } } },
    limits: { networkTypes: ["lora", "lokr"], mixedPrecisions: ["bf16", "f32"] }, ui: {},
  })) };
}
const generationRecord = () => ({ purpose: "generation-only", generator: TARGET_REGISTRY_WRITER, passed: 1, failed: 0, ignored: 0, exit_code: 0 });
function capability(backend, kind) {
  const generatedFrom = { inferenceRevision: PIN };
  return kind === "runtime"
    ? { generatedFrom, schemaVersion: 2, modelMappings: {}, videoModelMappings: [], trainerMappings: {}, workerCapabilities: [], snapshot: { backend, platform: backend === "mlx" ? "macos" : "cuda", generator_ids: ["fixture"], trainer_ids: ["fixture"], generator_capabilities: [{ id: "fixture", backend }], trainer_capabilities: [{ id: "fixture", backend }] } }
    : { generatedFrom, backend, ...(kind === "audio" ? { registry: "audio" } : {}), engines: [{ id: "fixture", modality: kind === "audio" ? "audio" : "image", supportsPreview: false }] };
}
function fixture(profile = "consumer-assertions") {
  const context = { ...request(profile), cargo_config_sha256: sha256(readFileSync(path.join(ROOT, ".cargo/config.toml"))), base_sha: BASE, merge_base: BASE, changed_files: ["Cargo.toml", "config/manifests/builtin.models.jsonc"] };
  const files = {};
  if (profile === "consumer-assertions") {
    files["assertions.json"] = Buffer.from(JSON.stringify(syntheticAssertionResult()));
    files["assertions.log"] = Buffer.from("synthetic assertion protocol fixture, not remote execution\n");
  } else {
    OUTPUTS.forEach((file, index) => { files[`payload/${file}`] = Buffer.from(index < 3 ? JSON.stringify(capability(index === 1 ? "candle" : "mlx", ["media", "audio", "runtime"][index])) : file.endsWith(".md") ? "# Static generated fixture\n" : file.endsWith(".jsonc") ? "// generated fixture\n{}\n" : "{}\n"); });
  }
  if (profile === "facts-and-projections") {
    files["facts.log"] = Buffer.from("actual static producer output\n");
    files[`payload/${TARGET_REGISTRY}`] = Buffer.from(JSON.stringify(targetRegistry()));
  }
  const receipt = { schema_version: 1, status: "passed", ...context, workflow_sha: DISPATCHER, workflow_ref: `${REPOSITORY}/${WORKFLOW}@refs/heads/${sourceRef}`, run_id: 41, run_attempt: 2, job: JOB, runner_name: "nax-macos-2", machine: machine(), cache_context: cacheContext(profile), ...(profile === "facts-and-projections" ? { target_registry_generation: generationRecord() } : {}), clean_before: true, clean_after: profile !== "facts-and-projections", command: profile === "consumer-assertions" ? ["consumer-assertions", ASSERTION_DRIVER] : ["facts-and-projections"], files: Object.fromEntries(Object.entries(files).map(([name, bytes]) => [name, sha256(bytes)])) };
  const controls = Object.fromEntries(CONTROL_FILES.map(file => [file, sha256(readFileSync(path.join(ROOT, file)))]));
  const run = { id: 41, run_attempt: 2, event: "workflow_dispatch", status: "completed", conclusion: "success", repository: { id: 9, full_name: REPOSITORY }, head_repository: { id: 9, full_name: REPOSITORY }, head_branch: sourceRef, head_sha: DISPATCHER, path: WORKFLOW };
  const ownedJob = { name: JOB, run_id: 41, run_attempt: 2, head_sha: DISPATCHER, conclusion: "success", runner_name: "nax-macos-2", labels: ["self-hosted", "macOS", "ARM64", "rw-llm"], steps: ["Run exact remote verification", "Upload remote verification artifact"].map(name => ({ name, conclusion: "success" })) };
  const jobs = { total_count: 2, jobs: [ownedJob, { name: "nax-worker", conclusion: "skipped" }] };
  const artifact = { id: 83, expired: false, name: artifactName(profile, SOURCE, 41, 2), digest: `sha256:${"e".repeat(64)}`, size_in_bytes: 1, workflow_run: { id: 41, head_sha: DISPATCHER, repository_id: 9, head_repository_id: 9 } };
  const f = { context, receipt, files, controls, run, jobs, artifact, archiveDigest: artifact.digest, sourceHistory: { merge_base_commit: { sha: SOURCE } }, baseHistory: { merge_base_commit: { sha: BASE } }, alteredControl: null };
  f.api = {
    json: async endpoint => {
      if (endpoint.endsWith("/attempts/2")) return structuredClone(f.run);
      if (endpoint.endsWith("/jobs?per_page=100")) return structuredClone(f.jobs);
      if (endpoint.endsWith("/artifacts/83")) return structuredClone(f.artifact);
      if (endpoint.endsWith(`/git/ref/heads/${sourceRef}`)) return { object: { sha: SOURCE } };
      if (endpoint.endsWith(`/git/ref/heads/${base}`)) return { object: { sha: BASE } };
      if (endpoint.includes(`/compare/${SOURCE}...`)) return structuredClone(f.sourceHistory);
      if (endpoint.includes(`/compare/${BASE}...`)) return structuredClone(f.baseHistory);
      throw new Error(`unexpected API endpoint: ${endpoint}`);
    },
    file: async (file, ref) => f.alteredControl === `${ref}:${file}` ? Buffer.from("different code") : readFileSync(path.join(ROOT, file)),
  };
  f.sync = () => { f.files["receipt.json"] = Buffer.from(JSON.stringify(f.receipt)); };
  f.verify = () => { f.sync(); return authenticateReceipt({ ...f, runId: 41, attempt: 2, artifactId: 83 }); };
  f.sync(); return f;
}
function replaceMember(f, name, bytes) { f.files[name] = Buffer.from(bytes); f.receipt.files[name] = sha256(f.files[name]); }

for (const profile of ["facts-and-projections", "consumer-assertions"]) test(`authenticated ${profile} accepts a different valid dispatcher SHA, not a relabeled checkout`, async () => { const f = fixture(profile); await f.verify(); assert.notEqual(f.receipt.workflow_sha, f.context.source_sha); });

const requestFaults = {
  "foreign repository": value => { value.repository = "fork/SceneWorks"; },
  "unknown profile": value => { value.profile = "anything"; },
  "retired canonical profile": value => { value.profile = "canonical-preflight"; },
  "short source SHA": value => { value.source_sha = SOURCE.slice(0, 8); },
  "short inference SHA": value => { value.inference_sha = PIN.slice(0, 8); },
  "pull ref": value => { value.source_ref = "refs/pull/2/head"; },
  "external ref": value => { value.source_ref = "other/repo@main"; },
  "story base": value => { value.base_ref = sourceRef; },
  "legacy GPU mixing": value => { value.legacy_device_inputs = true; },
  "device admission": value => { value.allow_nax_unit = true; },
};
for (const [name, mutate] of Object.entries(requestFaults)) test(`request rejects ${name}`, () => { const value = request(); mutate(value); assert.throws(() => validateRequest(value)); });
test("facts request rejects device admission", () => { const value = request("facts-and-projections"); value.allow_nax_unit = true; assert.throws(() => validateRequest(value)); });
for (const [name, patch] of Object.entries({ "primary runner": { runner_name: "nax-macos" }, "Linux": { platform: "linux" }, "Intel": { arch: "x64" }, "old runtime": { runtime: "26.1" }, "old SDK": { sdk: "26.0" }, "hosted deployment override": { deployment_target: "15.0" }, "pre-M5": { chip: "Apple M4 Max" }, "CLT instead of Xcode": { developer_dir: "/Library/Developer/CommandLineTools" }, "missing Metal": { metal_compiler: "" } })) test(`machine rejects ${name}`, () => assert.throws(() => validateMachine({ ...machine(), ...patch }, "facts-and-projections")));
test("dispatcher rejects missing or foreign execution identity", () => {
  const valid = { VERIFICATION_WORKFLOW_SHA: DISPATCHER, VERIFICATION_WORKFLOW_REF: `${REPOSITORY}/${WORKFLOW}@refs/heads/${sourceRef}`, GITHUB_RUN_ID: "41", GITHUB_RUN_ATTEMPT: "2" };
  assert.equal(dispatchIdentity(valid).run_attempt, 2);
  for (const patch of [{ GITHUB_RUN_ID: "0" }, { GITHUB_RUN_ATTEMPT: "NaN" }, { VERIFICATION_WORKFLOW_SHA: "main" }, { VERIFICATION_WORKFLOW_REF: `${REPOSITORY}/${WORKFLOW}@refs/pull/2/head` }]) assert.throws(() => dispatchIdentity({ ...valid, ...patch }));
});

const faults = {
  "receipt status": f => { f.receipt.status = "running"; },
  "receipt source": f => { f.receipt.source_sha = BASE; },
  "receipt source ref": f => { f.receipt.source_ref = "main"; },
  "receipt base ref": f => { f.receipt.base_ref = "main"; },
  "receipt pin": f => { f.receipt.inference_sha = BASE; },
  "receipt merge base": f => { f.receipt.merge_base = SOURCE; },
  "receipt scope": f => { f.receipt.changed_files = ["unrelated.rs"]; },
  "receipt changed sources": f => { f.receipt.changed_files = []; },
  "dirty initial source": f => { f.receipt.clean_before = false; },
  "dirty final source": f => { f.receipt.clean_after = false; },
  "receipt run": f => { f.receipt.run_id = 40; },
  "receipt attempt": f => { f.receipt.run_attempt = 1; },
  "receipt job": f => { f.receipt.job = "other"; },
  "receipt runner": f => { f.receipt.runner_name = "nax-macos"; },
  "receipt GPU permission": f => { f.receipt.allow_nax_unit = true; },
  "receipt legacy mixing": f => { f.receipt.legacy_device_inputs = true; },
  "run outcome": f => { f.run.conclusion = "failure"; },
  "incomplete run": f => { f.run.status = "in_progress"; },
  "run event": f => { f.run.event = "pull_request"; },
  "run attempt": f => { f.run.run_attempt = 1; },
  "run repository": f => { f.run.repository.full_name = "foreign/repo"; },
  "fork origin": f => { f.run.head_repository.full_name = "fork/SceneWorks"; },
  "workflow path": f => { f.run.path = ".github/workflows/other.yml"; },
  "workflow SHA": f => { f.run.head_sha = BASE; },
  "workflow ref": f => { f.receipt.workflow_ref = `${REPOSITORY}/${WORKFLOW}@refs/heads/main`; },
  "dispatcher code": f => { f.alteredControl = `${DISPATCHER}:${CONTROLLER}`; },
  "workflow code": f => { f.alteredControl = `${DISPATCHER}:${WORKFLOW}`; },
  "parser dependency code": f => { f.alteredControl = `${DISPATCHER}:scripts/lib/jsonc.mjs`; },
  "unit sandbox dependency code": f => { f.alteredControl = `${DISPATCHER}:${UNIT_SANDBOX}`; },
  "assertion driver code": f => { f.alteredControl = `${DISPATCHER}:${ASSERTION_DRIVER}`; },
  "case registry code": f => { f.alteredControl = `${DISPATCHER}:${REGISTRY_PATH}`; },
  "cache not released": f => { f.receipt.cache_context.released = false; },
  "cache compiler": f => { f.receipt.cache_context.toolchain = "other"; },
  "unowned source ancestry": f => { f.sourceHistory.merge_base_commit.sha = BASE; },
  "foreign base ancestry": f => { f.baseHistory.merge_base_commit.sha = SOURCE; },
  "malformed base SHA": f => { f.receipt.base_sha = "../other"; },
  "missing job pages": f => { f.jobs.total_count += 1; },
  "duplicate verification job": f => { f.jobs.jobs.push(structuredClone(f.jobs.jobs[0])); f.jobs.total_count += 1; },
  "job failed": f => { f.jobs.jobs[0].conclusion = "failure"; },
  "job run": f => { f.jobs.jobs[0].run_id = 40; },
  "job attempt": f => { f.jobs.jobs[0].run_attempt = 1; },
  "job head": f => { f.jobs.jobs[0].head_sha = SOURCE; },
  "actual primary runner": f => { f.jobs.jobs[0].runner_name = "nax-macos"; },
  "wrong runner labels": f => { f.jobs.jobs[0].labels = ["nax"]; },
  "missing verification step": f => { f.jobs.jobs[0].steps.shift(); },
  "legacy job ran": f => { f.jobs.jobs[1].conclusion = "success"; },
  "artifact expired": f => { f.artifact.expired = true; },
  "artifact id": f => { f.artifact.id = 84; },
  "artifact name": f => { f.artifact.name = "some-other-artifact"; },
  "artifact run": f => { f.artifact.workflow_run.id = 40; },
  "artifact workflow SHA": f => { f.artifact.workflow_run.head_sha = SOURCE; },
  "artifact source repository": f => { f.artifact.workflow_run.head_repository_id = 10; },
  "artifact repository": f => { f.artifact.workflow_run.repository_id = 10; },
  "artifact digest changed": f => { f.artifact.digest = `sha256:${"f".repeat(64)}`; },
  "command changed": f => { f.receipt.command.push("--all"); },
  "missing artifact member": f => { delete f.files["assertions.json"]; },
  "extra artifact executable": f => { f.files["run-me.py"] = Buffer.from("not executable data"); },
  "member digest": f => { f.files["assertions.json"] = Buffer.from("forged"); },
};
for (const [name, mutate] of Object.entries(faults)) test(`fault rejects ${name}`, async () => { const f = fixture(); mutate(f); await assert.rejects(f.verify); });

for (const [name, mutate] of Object.entries({ "wrong produced pin": value => { value.generatedFrom.inferenceRevision = BASE; }, "CPU runtime": value => { value.snapshot.backend = "cpu"; }, "CUDA on Mac": value => { value.snapshot.platform = "cuda"; }, "unknown runtime schema": value => { value.schemaVersion = 0; }, "duplicate identities": value => { value.snapshot.generator_ids.push("fixture"); }, "foreign descriptor": value => { value.snapshot.trainer_capabilities[0].backend = "candle"; }, "missing mapping": value => { delete value.trainerMappings; } })) test(`facts rejects ${name}`, async () => { const f = fixture("facts-and-projections"), value = capability("mlx", "runtime"); mutate(value); replaceMember(f, `payload/${OUTPUTS[2]}`, JSON.stringify(value)); await assert.rejects(f.verify); });
test("facts rejects malformed generated JSONC", async () => { const f = fixture("facts-and-projections"); replaceMember(f, `payload/${OUTPUTS[4]}`, "invalid JSONC"); await assert.rejects(f.verify); });
test("provenance-only drift does not invalidate capability semantics", () => {
  const older = capability("mlx", "media"); older.generatedFrom.inferenceRevision = BASE;
  assert.doesNotThrow(() => validateCapabilityFile(older, "mlx", "media"));
  assert.throws(() => validateCapabilityFile(older, "mlx", "media", PIN));
});
test("all six input facts must match real native shapes and one pin", () => {
  const read = file => JSON.stringify(capability(file.includes("runtime/") ? (file.endsWith(".mlx.json") ? "mlx" : "candle") : file.includes("audio/") ? "candle" : file.endsWith(".mlx.json") ? "mlx" : "candle", file.includes("runtime/") ? "runtime" : file.includes("audio/") ? "audio" : "media"));
  validateFacts(read, PIN);
  assert.throws(() => validateFacts(read, BASE));
  assert.throws(() => validateCapabilityFile({ ...capability("mlx", "media"), engines: [] }, "mlx", "media", PIN));
});

function zip(entries) {
  const python = String.raw`import base64,io,json,stat,sys,zipfile
b=io.BytesIO()
with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
 for name,value,mode in json.loads(sys.argv[1]):
  i=zipfile.ZipInfo(name);i.external_attr=mode<<16;z.writestr(i,base64.b64decode(value))
sys.stdout.buffer.write(b.getvalue())`;
  const result = spawnSync(process.platform === "win32" ? "python" : "python3", ["-I", "-c", python, JSON.stringify(entries.map(([name, value, mode = 0o100644]) => [name, Buffer.from(value).toString("base64"), mode]))], { maxBuffer: 8 * 1024 * 1024 });
  assert.equal(result.status, 0, result.stderr.toString()); return result.stdout;
}

test("archive digest rejects substituted bytes", () => { const bytes = zip([["receipt.json", "{}"]]); assert.throws(() => readArchive(bytes, `sha256:${"0".repeat(64)}`)); });
test("archive rejects excess members before reading their data", () => {
  const bytes = zip(Array.from({ length: 65 }, (_, index) => [`entry-${index}.json`, "{}"]));
  assert.throws(() => readArchive(bytes, `sha256:${sha256(bytes)}`));
});
test("archive parser accepts data without executing it", () => { const bytes = zip([["receipt.json", "{}"]]); assert.equal(readArchive(bytes, `sha256:${sha256(bytes)}`)["receipt.json"].toString(), "{}"); });
for (const [name, entries] of Object.entries({ traversal: [["../stamp.txt", "bad"]], absolute: [["/stamp.txt", "bad"]], windows: [["C:\\stamp.txt", "bad"]], backslash: [["folder\\stamp.txt", "bad"]], symlink: [["stamp.txt", "/outside", 0o120777]], duplicate: [["stamp.txt", "one"], ["stamp.txt", "two"]] })) test(`archive rejects ${name}`, () => { const bytes = zip(entries); assert.throws(() => readArchive(bytes, `sha256:${sha256(bytes)}`)); });

function localFixture(f) {
  const root = mkdtempSync(path.join(os.tmpdir(), "remote-macos-import-test-"));
  for (const file of Object.keys(f.controls)) { mkdirSync(path.dirname(path.join(root, file)), { recursive: true }); writeFileSync(path.join(root, file), readFileSync(path.join(ROOT, file))); }
  const bytes = zip(Object.entries(f.files));
  f.artifact.digest = `sha256:${sha256(bytes)}`; f.artifact.size_in_bytes = bytes.length;
  f.api.archive = async () => bytes;
  return { root, options: { runId: 41, attempt: 2, artifactId: 83, api: f.api, contextFor: () => f.context, commonDirectory: () => path.join(root, ".git") } };
}

for (const [name, change] of Object.entries({ "failed actual run": (f) => { f.run.conclusion = "failure"; }, "local changed during authentication": (_f, local) => { let calls = 0; local.options.contextFor = () => { if (++calls > 1) throw new Error("dirty local source"); return _f.context; }; }, "unexpected artifact code": (f) => { f.files["evil.py"] = Buffer.from("not executed"); } })) test(`import writes no proof for ${name}`, async () => {
  const f = fixture(); change(f, { options: {} }); const local = localFixture(f); if (name === "local changed during authentication") change(f, local);
  try { await assert.rejects(() => importReceipt(local.root, request(), local.options)); assert.equal(existsSync(path.join(local.root, ".git/sceneworks-consumer-assertions", SOURCE)), false); }
  finally { rmSync(local.root, { recursive: true, force: true }); }
});
test("consumer assertion evidence preserves independently authenticated receipt bytes", async () => {
  const f = fixture("consumer-assertions"), local = localFixture(f);
  try {
    await importReceipt(local.root, request("consumer-assertions"), local.options);
    assert.equal(existsSync(path.join(local.root, ".git/sceneworks-consumer-assertions", SOURCE, "41-2-83.json")), true);
    assert.equal(JSON.parse(readFileSync(path.join(local.root, ".git/sceneworks-consumer-assertions", SOURCE, "41-2-83.json"))).artifact_id, 83);
  } finally { rmSync(local.root, { recursive: true, force: true }); }
});
for (const [name, change] of Object.entries({ registry: value => { value.registry_sha256 = "0".repeat(64); }, implementation: value => { value.driver_sha256 = "0".repeat(64); }, survivor: value => { value.cases[0].killed = false; }, missing: value => { value.cases.pop(); }, counts: value => { value.cases[0].green[0].passed = 0; }, wrongTest: value => { const row = value.cases[0].green[0]; row.test_id = row.test_id === "protocol" ? "target" : "protocol"; }, restore: value => { value.source_after[MUTABLE_FILES[0]] = "wrong"; }, carry: value => { value.carried = { status: "carried-prior-proof-not-executed" }; } })) test(`consumer evidence rejects ${name}`, async () => {
  const f = fixture("consumer-assertions"), value = syntheticAssertionResult(); change(value);
  replaceMember(f, "assertions.json", JSON.stringify(value)); await assert.rejects(f.verify);
});
test("dedicated target cache lease prevents overlap and never steals or releases another owner", () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "remote-macos-cache-"));
  const owner = { runner_name: "nax-macos-2", run_id: 41, run_attempt: 2 };
  try {
    const first = acquireTargetLease(root, owner);
    assert.throws(() => acquireTargetLease(root, { ...owner, run_id: 42 }), /leased/);
    const file = path.join(root, ".remote-verification-owner/owner.json"), bytes = readFileSync(file);
    writeFileSync(file, JSON.stringify({ ...JSON.parse(bytes), token: "different" }));
    assert.throws(() => first.release(), /another run/);
    writeFileSync(file, bytes); first.release();
    acquireTargetLease(root, { ...owner, run_id: 42 }).release();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("cache context rejects changed compiler flags even with a self-consistent data hash", () => {
  const context = cacheContext(); validateCacheContext(context);
  const { context_sha256: _hash, ...changed } = context; changed.build_flags.jobs = "3";
  assert.throws(() => validateCacheContext({ ...changed, context_sha256: sha256(JSON.stringify(changed)) }), /flags/);
});
test("facts importer writes only authenticated derived outputs and no proof", async () => {
  const f = fixture("facts-and-projections"), local = localFixture(f);
  try { await importReceipt(local.root, request("facts-and-projections"), local.options); for (const file of OUTPUTS) assert.deepEqual(readFileSync(path.join(local.root, file)), f.files[`payload/${file}`]); assert.equal(existsSync(path.join(local.root, ".git/sceneworks-consumer-assertions", SOURCE)), false); }
  finally { rmSync(local.root, { recursive: true, force: true }); }
});
test("facts importer refuses a symlink destination before any output writes", async () => {
  const f = fixture("facts-and-projections"), local = localFixture(f);
  try {
    const outside = path.join(local.root, "outside"); mkdirSync(outside); mkdirSync(path.join(local.root, "config")); symlinkSync(outside, path.join(local.root, "config/engine-capabilities"), process.platform === "win32" ? "junction" : "dir");
    await assert.rejects(() => importReceipt(local.root, request("facts-and-projections"), local.options), /symlink/);
    assert.equal(existsSync(path.join(outside, "capabilities.mlx.json")), false);
  } finally { rmSync(local.root, { recursive: true, force: true }); }
});

test("checkout identity uses exact clean owned source, ancestry and changed-file scope", () => {
  const calls = [];
  const executeGit = (_root, ...args) => {
    calls.push(args);
    if (args[0] === "status") return "";
    if (args[0] === "remote") return `https://github.com/${REPOSITORY}.git`;
    if (args[0] === "fetch" || args.includes("--is-ancestor")) return "";
    if (args[0] === "diff") return "Cargo.toml\nconfig/manifests/builtin.models.jsonc";
    return args[0] === "rev-parse" && args[1] === "HEAD" ? SOURCE : BASE;
  };
  const result = checkoutContext(ROOT, request(), { executeGit, readPin: () => PIN, platform: "darwin" });
  assert.deepEqual(result.changed_files, ["Cargo.toml", "config/manifests/builtin.models.jsonc"]);
  assert.ok(calls.some(args => args.join(" ") === `merge-base --is-ancestor HEAD origin/${sourceRef}`));
  for (const fault of ["dirty", "head", "origin", "ancestry", "pin"]) {
    const changed = (root, ...args) => {
      if (fault === "dirty" && args[0] === "status") return " M source.rs";
      if (fault === "head" && args.join(" ") === "rev-parse HEAD") return BASE;
      if (fault === "origin" && args[0] === "remote") return "https://github.com/fork/SceneWorks.git";
      if (fault === "ancestry" && args.includes("--is-ancestor")) throw new Error("not an ancestor");
      return executeGit(root, ...args);
    };
    assert.throws(() => checkoutContext(ROOT, request(), { executeGit: changed, readPin: () => fault === "pin" ? BASE : PIN, platform: "darwin" }), fault);
  }
});
test("same-repository origin accepts exact HTTPS and SSH transports and rejects confusable identities", () => {
  const good = ["https://github.com/SceneWorks/SceneWorks", "https://github.com/SceneWorks/SceneWorks.git", "git@github.com:SceneWorks/SceneWorks.git", "ssh://git@github.com/SceneWorks/SceneWorks.git"];
  for (const origin of good) {
    assert.equal(policy.canonicalRepositoryOrigin(origin), `https://github.com/${REPOSITORY}.git`);
    const executeGit = (_root, ...args) => {
      if (args[0] === "status") return "";
      if (args[0] === "remote") return origin;
      if (args[0] === "fetch" || args.includes("--is-ancestor")) return "";
      if (args[0] === "diff") return "Cargo.toml";
      return args[0] === "rev-parse" && args[1] === "HEAD" ? SOURCE : BASE;
    };
    assert.equal(checkoutContext(ROOT, request("facts-and-projections"), { executeGit, readPin: () => PIN }).source_sha, SOURCE);
  }
  const bad = [undefined, null, "", "https://github.com/fork/SceneWorks.git", "git@github.com:SceneWorks/inference.git", "ssh://git@github.com/SceneWorks/SceneWorks.git.evil", "https://github.com.evil/SceneWorks/SceneWorks.git", "git@github.com.evil:SceneWorks/SceneWorks.git", "ssh://root@github.com/SceneWorks/SceneWorks.git", "https://git@github.com/SceneWorks/SceneWorks.git", "https://token:secret@github.com/SceneWorks/SceneWorks.git", "ssh://git:secret@github.com/SceneWorks/SceneWorks.git", "ssh://git@github.com:22/SceneWorks/SceneWorks.git", "https://github.com/SceneWorks/%53ceneWorks.git", "https://github.com/SceneWorks/SceneWorks.git/../inference", "https://github.com/SceneWorks/SceneWorks.git/", "git@github.com:SceneWorks/SceneWorks.git?x", "https://github.com/SceneWorks/SceneWorks.git#x", "ssh://git@github.com/SceneWorks/SceneWorks.git?x", "git@github.com:SceneWorks/SceneWorks.git\n", " https://github.com/SceneWorks/SceneWorks.git", "https://github.com/sceneworks/SceneWorks.git"];
  for (const origin of bad) assert.throws(() => policy.canonicalRepositoryOrigin(origin), /same-origin checkout required/, String(origin));
  // The checkout path must call the strict canonicalizer, not merely export a dead helper.
  assert.match(readFileSync(modulePath, "utf8"), /canonicalRepositoryOrigin\(g\("remote", "get-url", "origin"\)\);/);
});
test("Cargo pin reader rejects multiple or differently resolved inference sources", () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "remote-macos-pin-"));
  try {
    const manifest = `[dependencies]\ngen-core = { git = "https://github.com/SceneWorks/inference", rev = "${PIN}" }\n`;
    writeFileSync(path.join(root, "Cargo.toml"), manifest); writeFileSync(path.join(root, "Cargo.lock"), `source = "git+https://github.com/SceneWorks/inference?rev=${PIN}#${PIN}"\n`);
    assert.equal(inferencePin(root), PIN);
    writeFileSync(path.join(root, "Cargo.lock"), `source = "git+https://github.com/SceneWorks/inference?rev=${PIN}#${BASE}"\n`); assert.throws(() => inferencePin(root));
    writeFileSync(path.join(root, "Cargo.toml"), manifest + `other = { git = "https://github.com/SceneWorks/inference", rev = "${BASE}" }\n`); assert.throws(() => inferencePin(root));
  } finally { rmSync(root, { recursive: true, force: true }); }
});

const workflowPath = process.env.REMOTE_MACOS_WORKFLOW_UNDER_TEST ?? path.join(ROOT, WORKFLOW);
const workflow = readFileSync(workflowPath, "utf8");
function jobBody(name, body = workflow) { const after = body.split(`\n  ${name}:\n`)[1]; assert.ok(after, name); return after.split(/\n  [a-z][a-z-]+:\n/)[0]; }
function condition(name, event, profile, relevant = "true", result = "success", target = "feature/sc-26584-moss-image-training", repository = REPOSITORY, extraInputs = {}) {
  const expression = /\n    if: \$\{\{ (.+) \}\}/.exec(`\n${jobBody(name)}`)?.[1]; assert.ok(expression, name);
  const github = { event_name: event, repository, event: { pull_request: { base: { ref: target }, head: { repo: { full_name: REPOSITORY } } } } };
  return Boolean(Function("github", "inputs", "needs", "cancelled", `return (${expression});`)(github, { verification_profile: profile, ...extraInputs }, { changes: { result, outputs: { relevant } } }, () => false));
}
test("manual profiles never run legacy jobs", () => {
  for (const profile of ["facts-and-projections", "consumer-assertions", "unknown-unsafe-value"]) {
    for (const name of ["changes", "macos-checks", "nax-worker"]) for (const result of ["success", "skipped", "failure"]) assert.equal(condition(name, "workflow_dispatch", profile, "true", result), false, `${profile}/${name}/${result}`);
    assert.equal(condition(JOB, "workflow_dispatch", profile), profile !== "unknown-unsafe-value", profile);
  }
});
test("browser fixture artifact profile never queues the self-hosted manual verifier", () => {
  assert.equal(condition(JOB, "workflow_dispatch", "browser-fixture-artifact"), false);
  assert.equal(
    condition(
      JOB,
      "workflow_dispatch",
      "browser-fixture-artifact",
      "true",
      "success",
      base,
      REPOSITORY,
      { run_memory_calibration: true },
    ),
    false,
  );
});

test("browser fixture conflicts execute rejection before checkout or compiler setup instead of skipping green", () => {
  const name = "browser-fixture-artifact", body = jobBody(name);
  const step = /      - name: Reject conflicting browser fixture selectors before setup\n[\s\S]*?(?=\n      - )/.exec(body)?.[0];
  assert.ok(step, "selected producer must have an independent early rejection step");
  assert.ok(body.indexOf("Reject conflicting browser fixture selectors") < body.indexOf("actions/checkout"));
  assert.ok(body.indexOf("Reject conflicting browser fixture selectors") < body.indexOf("dtolnay/rust-toolchain"));
  const script = step.split("        run: |\n")[1].split("\n").map(line => line.startsWith("          ") ? line.slice(10) : line).join("\n");
  const selectors = ["RUN_MEMORY_CALIBRATION", "RUN_FIVE_RUNG_REFERENCE", "PROVISION_QWEN_SNAPSHOT", "PROVISION_Z_IMAGE_SNAPSHOT", "RUN_MOSS_NATIVE_VALIDATION"];
  const clean = Object.fromEntries(selectors.map(key => [key, "false"]));
  assert.equal(condition(name, "workflow_dispatch", name), true);
  assert.equal(spawnSync("bash", ["-c", script], { env: { ...process.env, ...clean }, encoding: "utf8" }).status, 0);
  for (const selector of selectors) {
    const inputs = { [selector.toLowerCase()]: true };
    assert.equal(condition(name, "workflow_dispatch", name, "true", "success", base, REPOSITORY, inputs), true,
      "a conflict must start rejection, not skip the only selected producer");
    for (const legacy of ["changes", "macos-checks", "nax-worker"]) {
      assert.equal(condition(legacy, "workflow_dispatch", name, "true", "success", base, REPOSITORY, inputs), false);
    }
    const result = spawnSync("bash", ["-c", script], { env: { ...process.env, ...clean, [selector]: "true" }, encoding: "utf8" });
    assert.equal(result.status, 1, selector);
    assert.match(result.stdout, new RegExp(`cannot be combined with ${selector}`));
  }
});

test("fork manual verification never queues a self-hosted runner", () => {
  assert.equal(condition(JOB, "workflow_dispatch", "consumer-assertions", "true", "success", base, "fork/SceneWorks"), false);
});
test("legacy PR main dispatch and fail-open decisions remain unchanged", () => {
  assert.equal(condition("changes", "workflow_dispatch", "legacy"), true);
  assert.equal(condition("nax-worker", "workflow_dispatch", "legacy"), true);
  assert.equal(condition("macos-checks", "pull_request", undefined, "false", "failure"), true);
  assert.equal(condition("macos-checks", "pull_request", undefined, "false", "success"), false);
  assert.equal(condition("nax-worker", "pull_request", undefined, "true", "success", "main"), true);
  assert.equal(condition("nax-worker", "pull_request", undefined), false);
  assert.equal(condition("nax-worker", "merge_group", undefined), false);
  assert.equal(condition(JOB, "pull_request", undefined), false);
});
test("manual job pins read-only control and exact candidate before any build", () => {
  const body = jobBody(JOB);
  assert.match(body, /runs-on: \[self-hosted, macOS, ARM64, rw-llm\]/);
  assert.match(body, /permissions:\n      contents: read/);
  assert.match(body, /MACOSX_DEPLOYMENT_TARGET: "26\.2"/);
  assert.equal((body.match(/persist-credentials: false/g) ?? []).length, 2);
  assert.ok(body.indexOf("Require actual secondary Mac") < body.indexOf("dtolnay/rust-toolchain"));
  assert.ok(body.indexOf("Require exact source and NAX runtime SDK hardware") < body.indexOf("dtolnay/rust-toolchain"));
  assert.match(body, /ref: \$\{\{ github\.workflow_sha \}\}/);
  assert.match(body, /ref: \$\{\{ inputs\.source_sha \}\}/);
  assert.doesNotMatch(body, /secrets\.|--skip|--ignored|no-verify|SKIP_/);
  assert.match(body, /remote-macos-verification\.mjs" run/);
});

function fixtureNode(root, name) {
  const executable = path.join(root, "deps", name + (process.platform === "win32" ? ".exe" : ""));
  // Alias this trusted Node runtime, not a downloaded executable. A cross-volume fixture copies it.
  try { linkSync(process.execPath, executable); }
  catch (error) { if (!["EXDEV", "EPERM", "EACCES"].includes(error.code)) throw error; copyFileSync(process.execPath, executable); }
  return executable;
}
test("manual verifier overrides only host build strip and binds both profiles in the cache recipe", () => {
  const original = { PATH: "/host/bin", SDKROOT: "/host/sdk", TMPDIR: "/host/tmp", CARGO_TARGET_DIR: "/dedicated/target", CARGO_INCREMENTAL: "1", CARGO_BUILD_TARGET: "foreign", CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER: "foreign-runner" };
  const env = policy.remoteBuildEnv(original);
  const expected = {
    CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP: "none", CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP: "none",
    CARGO_PROFILE_DEV_DEBUG: "0", CARGO_PROFILE_TEST_DEBUG: "0", CARGO_PROFILE_DEV_BUILD_OVERRIDE_DEBUG: "0", CARGO_PROFILE_TEST_BUILD_OVERRIDE_DEBUG: "0",
    CARGO_BUILD_JOBS: "2", CMAKE_BUILD_PARALLEL_LEVEL: "2", RUST_TEST_THREADS: "1", RUSTC_WRAPPER: "", RUSTUP_TOOLCHAIN: "1.97.1", MACOSX_DEPLOYMENT_TARGET: "26.2",
  };
  for (const [key, value] of Object.entries(expected)) assert.equal(env[key], value, key);
  assert.deepEqual(Object.keys(env).filter(key => key.endsWith("_STRIP")).sort(), ["CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP", "CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP"]);
  for (const key of ["SDKROOT", "TMPDIR", "CARGO_TARGET_DIR"]) assert.equal(env[key], original[key]);
  assert.equal(env.CARGO_INCREMENTAL, "0");
  assert.equal(original.CARGO_BUILD_TARGET, "foreign"); assert.equal(env.CARGO_BUILD_TARGET, undefined); assert.equal(env.CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER, undefined);
  assert.equal(CACHE_FLAGS.build_override_strip, "none"); assert.equal(CACHE_FLAGS.test_build_override_strip, "none");
  for (const key of ["build_override_strip", "test_build_override_strip"]) {
    const stale = cacheContext(); delete stale.build_flags[key];
    const { context_sha256, ...data } = stale; stale.context_sha256 = sha256(JSON.stringify(data));
    assert.throws(() => validateCacheContext(stale), /build flags/);
  }
  const controller = readFileSync(modulePath, "utf8");
  assert.match(controller, /const runEnv = \{ \.\.\.remoteBuildEnv\(env, request.profile\), \.\.\.createUnitHfSandbox\(root, env\)\.env \};/);
  assert.ok(controller.indexOf("const runEnv = { ...remoteBuildEnv(env, request.profile)") < controller.indexOf('if (request.profile === "facts-and-projections")'));
  assert.equal([...controller.matchAll(/\bremoteBuildEnv\(/g)].length, 2); // definition + single common profile entry
  // The hosted artifact producer owns an isolated profile; no override may leak into
  // workflow-global settings or the existing manual/legacy lanes.
  const stripOverride = /BUILD_OVERRIDE_STRIP|CARGO_PROFILE_(?:DEV|TEST)_STRIP/;
  assert.doesNotMatch(workflow.split("\njobs:\n")[0], stripOverride);
  const jobNames = [...workflow.split("\njobs:\n")[1].matchAll(/^  ([a-z][a-z-]+):$/gm)].map((match) => match[1]);
  assert.ok(jobNames.includes("browser-fixture-artifact"));
  for (const name of jobNames.filter((name) => name !== "browser-fixture-artifact")) {
    assert.doesNotMatch(jobBody(name), stripOverride, name);
  }
  const artifactProducer = jobBody("browser-fixture-artifact");
  for (const profile of ["DEV", "TEST"]) {
    assert.match(artifactProducer, new RegExp(`CARGO_PROFILE_${profile}_STRIP: "false"`));
    assert.match(artifactProducer, new RegExp(`CARGO_PROFILE_${profile}_BUILD_OVERRIDE_STRIP: "none"`));
  }
});
test("incremental compilation is scoped to the declared consumer profile and authenticated cache recipe", async () => {
  for (const profile of ["facts-and-projections", "consumer-assertions"]) {
    const expected = profile === "consumer-assertions" ? "1" : "0";
    for (const ambient of [undefined, "0", "1", "foreign-value"]) {
      const env = policy.remoteBuildEnv({ CARGO_INCREMENTAL: ambient, VERIFICATION_PROFILE: "consumer-assertions", TMPDIR: "/tmp/unchanged", SDKROOT: "/sdk/unchanged", CARGO_TARGET_DIR: "/leased/target" }, profile);
      assert.equal(env.CARGO_INCREMENTAL, expected, `${profile}/${ambient}`);
      assert.equal(env.TMPDIR, "/tmp/unchanged"); assert.equal(env.SDKROOT, "/sdk/unchanged"); assert.equal(env.CARGO_TARGET_DIR, "/leased/target");
      assert.equal(env.CARGO_BUILD_JOBS, "2"); assert.equal(env.CARGO_PROFILE_TEST_DEBUG, "0"); assert.equal(env.CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP, "none");
    }
    assert.equal(policy.cacheFlags(profile).incremental, expected);
    const valid = cacheContext(profile); validateCacheContext(valid, profile);
    for (const incorrect of [undefined, expected === "1" ? "0" : "1"]) {
      const stale = structuredClone(valid);
      if (incorrect === undefined) delete stale.build_flags.incremental; else stale.build_flags.incremental = incorrect;
      const { context_sha256, ...body } = stale; stale.context_sha256 = sha256(JSON.stringify(body));
      assert.throws(() => validateCacheContext(stale, profile), /cache build flags/);
    }
  }
  assert.notEqual(cacheContext("consumer-assertions").context_sha256, cacheContext("facts-and-projections").context_sha256);
  assert.throws(() => policy.remoteBuildEnv({}, "foreign-profile"), /unknown cache recipe profile/);
  const controller = readFileSync(modulePath, "utf8");
  assert.match(controller, /build_flags: cacheFlags\(request.profile\)/);
  assert.match(controller, /validateCacheContext\(receipt.cache_context, request.profile\)/);
  assert.match(controller, /validateCacheContext\(receipt.cache_context, context.profile\)/);
  assert.match(controller, /remoteBuildEnv\(env, request.profile\)/);
  const valid = fixture("consumer-assertions"); await valid.verify();
  const stale = fixture("consumer-assertions");
  stale.receipt.cache_context = cacheContext("facts-and-projections");
  await assert.rejects(stale.verify, /cache build flags/);
});
test("facts transport binds the exact authoritative target writer as generation only", async () => {
  assert.equal(OUTPUTS.length, 9); assert.equal(OUTPUTS.at(-1), TARGET_REGISTRY);
  assert.deepEqual(OUTPUTS.slice(0, 8), ["config/engine-capabilities/capabilities.mlx.json", "config/engine-capabilities/audio/capabilities.candle.json", "config/engine-capabilities/runtime/capabilities.mlx.json", "config/backend-capabilities/matrix.json", "config/manifests/builtin.preview-support.jsonc", "apps/web/src/data/previewSupport.json", "docs/generated/memory-matrix.json", "docs/generated/memory-matrix.md"]);
  assert.equal(TARGET_REGISTRY, "tests/fixtures/rust_migration_contracts/training/target-registry.json");
  const writer = `running 1 test\ntest ${TARGET_REGISTRY_WRITER} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 99 filtered out; finished in 0.01s\n`;
  assert.deepEqual(policy.registryGenerationRecord(writer), generationRecord());
  for (const bad of [writer.replace("1 passed", "0 passed"), writer.replace("0 ignored", "1 ignored"), writer.replaceAll(TARGET_REGISTRY_WRITER, "other_writer"), "error[E0308]: cannot compile"]) assert.throws(() => policy.registryGenerationRecord(bad));
  for (const profile of ["facts-and-projections", "consumer-assertions"]) assert.equal(policy.remoteBuildEnv({ REGEN_FIXTURE: "ambient" }, profile).REGEN_FIXTURE, undefined);
  const registry = targetRegistry(); policy.validateTargetRegistry(registry);
  const expanded = structuredClone(registry); expanded.targets.push({ ...expanded.targets[0], id: "extra_target" }); policy.validateTargetRegistry(expanded); // no pinned total population
  for (const change of [
    value => { value.targets = []; }, value => { value.targets.pop(); }, value => { value.targets.push(value.targets[0]); },
    value => { value.targets[0].baseModel = "moss_image_sft"; }, value => { value.targets[0].baseModelRepo = "foreign/checkpoint"; },
    value => { value.targets[0].limits.networkTypes = ["lora"]; }, value => { value.targets[0].defaults.rank = 1.5; },
    value => { value.targets[0].defaults.advanced.modelOptions.vae_posterior = "sample"; }, value => { value.machine = "host-dependent"; },
  ]) { const bad = structuredClone(registry); change(bad); assert.throws(() => policy.validateTargetRegistry(bad)); }
  const controller = readFileSync(modulePath, "utf8");
  assert.ok(controller.includes('["cargo", "test", "--locked", "-p", "sceneworks-core", "--test", "training_contracts", TARGET_REGISTRY_WRITER, "--", "--exact", "--ignored"]'));
  assert.ok(controller.includes('log, { ...runEnv, REGEN_FIXTURE: "1" }, true)'));
  assert.doesNotMatch(controller, /runEnv\.REGEN_FIXTURE\s*=|process\.env\.REGEN_FIXTURE\s*=/);
  const valid = fixture("facts-and-projections"); await valid.verify();
  for (const change of [f => { delete f.receipt.target_registry_generation; }, f => { f.receipt.target_registry_generation.passed = 0; }, f => { f.receipt.target_registry_generation.purpose = "validation-proof"; }]) {
    const f = fixture("facts-and-projections"); change(f); await assert.rejects(f.verify);
  }
  const badWire = fixture("facts-and-projections"); replaceMember(badWire, `payload/${TARGET_REGISTRY}`, JSON.stringify({ schemaVersion: 1, targets: [] })); await assert.rejects(badWire.verify);
});

test("original proc macro invocation distinguishes rebuilt no-strip from stale or stripped evidence", () => {
  const command = "   Running `rustc --crate-name bytemuck_derive src/lib.rs --crate-type proc-macro -C prefer-dynamic";
  const clean = policy.invocationEvidence(command + " --out-dir /target/debug/deps`", () => true).join("\n");
  assert.match(clean, /original proc-macro invocation: bytemuck_derive; strip=absent/); assert.doesNotMatch(clean, /NOT REBUILT/);
  for (const flag of ["-C strip=debuginfo", "-Cstrip=symbols"]) assert.match(policy.invocationEvidence(command + " " + flag + "`", () => true).join("\n"), /strip=(?:debuginfo|symbols)/);
  assert.match(policy.invocationEvidence("Fresh bytemuck_derive v1", () => true).join("\n"), /NOT REBUILT.*no strip hypothesis verdict/);
});
test("failed native build records fixed read-only toolchain diagnostics without changing the verdict", () => {
  const calls = [];
  const text = policy.buildDiagnostics(ROOT, { SDKROOT: "/sdk", CARGO_TARGET_DIR: "/target" }, (cmd, args) => { calls.push([cmd, ...args].join(" ")); return { status: 0, stdout: `${cmd} ok`, stderr: "" }; });
  for (const expected of ["rustc -vV", "cargo -V", "xcrun --show-sdk-path", "xcrun --show-sdk-version", "ld -v", "xcode-select -p", "rustc --print sysroot"]) assert.ok(calls.includes(expected), expected);
  const dylibProbe = calls.find(call => call.endsWith(path.join("/target", "debug", "deps")));
  assert.ok(dylibProbe);
  for (const expected of ["otool -hv", "LC_BUILD_VERSION", "otool -L", "-gt 12", "file \"$f\""]) assert.ok(dylibProbe.includes(expected), expected);
  // The library path is only ever an ARGUMENT to inspection tools, never a command to execute.
  assert.doesNotMatch(dylibProbe, /(?:^|\n|;|&&|\||\bthen|\bdo)\s*"\$f"|\bexec\b|(?:^|\s)\.\s+"\$f"|dlopen/);
  assert.ok(dylibProbe.includes("sectname \\.rustc") && dylibProbe.includes("rustc-metadata-sections"));
  for (const expected of ["xcrun --find ld-classic", "rustup toolchain list"]) assert.ok(calls.includes(expected), expected);
  assert.ok(calls.some(call => call.includes("ls -d /Applications/Xcode*.app") && call.includes("head -n 10")));
  // Diagnostics NEVER compile: no cargo build/run/test and no rustc beyond version/sysroot queries.
  const original = "   Running `rustc --crate-name syn x`\n   Running `rustc --crate-name bytemuck --edition=2018 --extern bytemuck_derive=/t/libbytemuck_derive-1.dylib -L dependency=/t`\nnoise";
  const evidence = []; const replayed = policy.buildDiagnostics(ROOT, {}, (cmd, args) => { evidence.push([cmd, ...args]); return { status: 0, stdout: "", stderr: "" }; }, original);
  assert.ok(evidence.every(([cmd, ...args]) => cmd !== "cargo" ? cmd !== "rustc" || ["-vV", "--print"].includes(args[0]) : args[0] === "-V"), JSON.stringify(evidence));
  assert.match(replayed, /2 rustc invocation line\(s\); showing last 2/);
  assert.match(replayed, /--extern bytemuck_derive=\/t\/libbytemuck_derive-1\.dylib \(exists: false\)/);
  assert.doesNotMatch(replayed, /noise/);
  assert.match(policy.invocationEvidence("   Running `rustc --crate-name bytemuck -L x`").join("\n"), /NO --extern bytemuck_derive argument/);
  assert.match(policy.invocationEvidence("   Running `rustc --crate-name bytemuck " + "a".repeat(4000) + "`").join("\n"), /truncated/);
  assert.match(policy.invocationEvidence("nothing").join("\n"), /not found in original log/);
  // Real CI shape (run 39553468673): coloured "Running" and Cargo's "Caused by" invocation line.
  const coloured = "\x1b[1m\x1b[92m     Running\x1b[0m `/r/rustc --crate-name zerocopy_derive x`\n  process didn't exit successfully: `/r/rustc --crate-name bytemuck --edition=2018 --extern bytemuck_derive=/t/../d/libbytemuck_derive-8.dylib --cap-lints allow` (exit status: 1)";
  const parsed = policy.invocationEvidence(coloured, () => true).join("\n");
  assert.match(parsed, /2 rustc invocation line\(s\)/); assert.match(parsed, /bytemuck_derive=\/t\/\.\.\/d\/libbytemuck_derive-8\.dylib \(exists: true\)/);
  assert.ok(dylibProbe.includes('otool -s __DATA .rustc') && dylibProbe.includes("rust_metadata exports"));
  // Exactly the original facts dump is verbose (not -vv, no second cargo execution).
  const controller = readFileSync(path.join(ROOT, CONTROLLER), "utf8");
  assert.equal(controller.split('"--verbose"').length - 1, 1); assert.doesNotMatch(controller, /"-vv"|"build", "-v"/);
  // A probe failure is evidence, not a new verdict.
  const failing = policy.buildDiagnostics(ROOT, {}, () => ({ status: 1, stdout: "", stderr: "unavailable" }));
  assert.match(failing, /exit 1/);
  assert.match(text, /SDKROOT=\/sdk/); assert.match(text, /read-only/);
  assert.ok(calls.every(call => !/\b(rm|mv|cp|chmod|install|sudo)\b/.test(call.split(" ").slice(0, 2).join(" "))));
});
test("self-hosted manual job uses provisioned Python 3.12, not the hosted toolcache installer", () => {
  const body = jobBody(JOB);
  assert.doesNotMatch(body, /actions\/setup-python@/);
  assert.match(body, /command -v python3\.12/);
  assert.match(body, /--version \| grep -E '\^Python 3\\\.12\\\.'/);
  assert.match(body, /echo "\$shim" >> "\$GITHUB_PATH"/);
  assert.ok(body.indexOf("Require exact host Python 3.12") < body.indexOf("Run exact remote verification"));
});




if (!process.env.REMOTE_MACOS_MUTATION_CHILD) test("named source bypass mutations make their original tests red", async t => {
  const controller = readFileSync(path.join(ROOT, CONTROLLER), "utf8");
  const cases = [
    ["RV-registry-writer-env", "facts transport binds the exact authoritative target writer as generation only", '{ ...runEnv, REGEN_FIXTURE: "1" }', 'runEnv'],
    ["RV-registry-env-leak", "facts transport binds the exact authoritative target writer as generation only", 'delete runEnv.REGEN_FIXTURE;', 'void runEnv.REGEN_FIXTURE;'],
    ["RV-registry-generation-proof", "facts transport binds the exact authoritative target writer as generation only", 'equal(record, { purpose: "generation-only", generator: TARGET_REGISTRY_WRITER, passed: 1, failed: 0, ignored: 0, exit_code: 0 })', 'true'],
    ["RV-registry-wire", "facts transport binds the exact authoritative target writer as generation only", 'export function validateTargetRegistry(value) {', 'export function validateTargetRegistry(value) { return;'],
    ["RV-incremental-runtime", "incremental compilation is scoped to the declared consumer profile and authenticated cache recipe", 'runEnv.CARGO_INCREMENTAL = cacheFlags(profile).incremental;', 'void profile;'],
    ["RV-incremental-scope", "incremental compilation is scoped to the declared consumer profile and authenticated cache recipe", 'incremental: profile === "consumer-assertions" ? "1" : "0"', 'incremental: "1"'],
    ["RV-incremental-recipe", "incremental compilation is scoped to the declared consumer profile and authenticated cache recipe", 'equal(context.build_flags, cacheFlags(profile))', 'true'],
    ["RV-incremental-auth-profile", "incremental compilation is scoped to the declared consumer profile and authenticated cache recipe", 'validateCacheContext(receipt.cache_context, context.profile);', 'validateCacheContext(receipt.cache_context);'],
    ["RV-origin-identity", "same-repository origin accepts exact HTTPS and SSH transports and rejects confusable identities", '].includes(origin), "same-origin checkout required"', '].length > 0, "same-origin checkout required"'],
    ["RV-origin-call", "same-repository origin accepts exact HTTPS and SSH transports and rejects confusable identities", 'canonicalRepositoryOrigin(g("remote", "get-url", "origin"));', 'g("remote", "get-url", "origin");'],
    ["RV-host-strip-dev", "manual verifier overrides only host build strip and binds both profiles in the cache recipe", 'CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP: "none", ', ''],
    ["RV-host-strip-test", "manual verifier overrides only host build strip and binds both profiles in the cache recipe", 'CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP: "none", ', ''],
    ["RV-host-strip-cache-dev", "manual verifier overrides only host build strip and binds both profiles in the cache recipe", ' build_override_strip: "none", ', ''],
    ["RV-host-strip-cache-test", "manual verifier overrides only host build strip and binds both profiles in the cache recipe", 'test_build_override_strip: "none", ', ''],
    ["RV-cache-flags", "cache context rejects changed compiler flags even with a self-consistent data hash", 'equal(context.build_flags, cacheFlags(profile))', 'true'],
    ["RV-cache-exclusion", "dedicated target cache lease prevents overlap and never steals or releases another owner", 'mkdirSync(lease);', 'mkdirSync(lease, { recursive: true });'],
    ["RV-consumer-proof", "consumer evidence rejects survivor", 'validateAssertionReceipt(JSON.parse(files["assertions.json"].toString()), spec, context.source_sha, controls[ASSERTION_DRIVER]);', 'void spec;'],
    ["RV-no-currency-gate", "provenance-only drift does not invalidate capability semantics", 'pin === undefined || revision === pin', 'revision === pin'],
    ["RV-run-authority", "fault rejects run outcome", 'run.conclusion === "success"', 'true'],
    ["RV-scope", "fault rejects receipt scope", 'equal(receipt.changed_files, context.changed_files)', 'true'],
    ["RV-archive-digest", "archive digest rejects substituted bytes", 'digest === `sha256:${sha256(bytes)}`', 'true'],
    ["RV-archive-members", "archive rejects excess members before reading their data", 'assert len(z.infolist())<=64, "too many artifact members"', 'pass'],
    ["RV-primary-machine", "machine rejects primary runner", 'machine.runner_name === "nax-macos-2"', 'true'],
  ];
  for (const [name, target, old, replacement] of cases) await t.test(name, () => {
    const scratch = mkdtempSync(path.join(os.tmpdir(), "remote-macos-source-mutation-"));
    try {
      assert.equal(controller.split(old).length - 1, 1, name);
      mkdirSync(path.join(scratch, "scripts/lib"), { recursive: true });
      const mutated = path.join(scratch, CONTROLLER);
      writeFileSync(mutated, controller.replace(old, replacement)); utimesSync(mutated, new Date(), new Date());
      for (const file of [UNIT_SANDBOX, "scripts/lib/jsonc.mjs", ASSERTION_DRIVER, REGISTRY_PATH]) writeFileSync(path.join(scratch, file), readFileSync(path.join(ROOT, file)));
      const childEnv = { ...process.env, REMOTE_MACOS_MUTATION_CHILD: "1", REMOTE_MACOS_CONTROLLER_UNDER_TEST: mutated };
      // A new test-runner process must not inherit the outer worker's private IPC protocol.
      delete childEnv.NODE_TEST_CONTEXT;
      const result = spawnSync(process.execPath, nestedTestArgs(target), { cwd: ROOT, env: childEnv, encoding: "utf8", timeout: 10000 });
      assert.notEqual(result.status, 0, `${name}\n${result.stdout}\n${result.stderr}`); assert.ok(result.stdout.includes(`✖ ${target}`), result.stdout); assert.match(result.stdout, /(?:ℹ|#) fail 1/); assert.doesNotMatch(result.stderr, /SyntaxError|ERR_MODULE_NOT_FOUND/);
      writeFileSync(mutated, controller); utimesSync(mutated, new Date(), new Date());
      if (name.startsWith("RV-cross-")) {
        const green = spawnSync(process.execPath, nestedTestArgs(target), { cwd: ROOT, env: childEnv, encoding: "utf8", timeout: 10000 });
        assert.equal(green.status, 0, green.stdout + green.stderr);
      }
    } finally { rmSync(scratch, { recursive: true, force: true }); }
  });

  await t.test("RV-hosted-python-installer", () => {
    const scratch = mkdtempSync(path.join(os.tmpdir(), "remote-macos-workflow-mutation-"));
    try {
      const old = "      - name: Require exact host Python 3.12\n";
      assert.equal(workflow.split(old).length - 1, 1);
      const mutated = path.join(scratch, "macos-mlx.yml");
      writeFileSync(mutated, workflow.replace(old, '      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0\n        with:\n          python-version: "3.12"\n' + old)); utimesSync(mutated, new Date(), new Date());
      const childEnv = { ...process.env, REMOTE_MACOS_MUTATION_CHILD: "1", REMOTE_MACOS_WORKFLOW_UNDER_TEST: mutated }; delete childEnv.NODE_TEST_CONTEXT;
      const target = "self-hosted manual job uses provisioned Python 3.12, not the hosted toolcache installer";
      const result = spawnSync(process.execPath, nestedTestArgs(target), { cwd: ROOT, env: childEnv, encoding: "utf8", timeout: 10000 });
      assert.notEqual(result.status, 0, result.stdout); assert.ok(result.stdout.includes(`✖ ${target}`), result.stdout); assert.match(result.stdout, /(?:ℹ|#) fail 1/);
    } finally { rmSync(scratch, { recursive: true, force: true }); }
  });
  for (const name of ["changes", "macos-checks", "nax-worker", JOB]) await t.test(name === JOB ? "RV-manual-origin" : `RV-legacy-${name}`, () => {
    const scratch = mkdtempSync(path.join(os.tmpdir(), "remote-macos-workflow-mutation-"));
    try {
      const body = jobBody(name), line = /    if: \$\{\{ .+ \}\}/.exec(body)?.[0]; assert.ok(line);
      const mutated = path.join(scratch, "macos-mlx.yml");
      const changed = name === JOB ? line.replace(" && github.repository == 'SceneWorks/SceneWorks'", "") : "    if: ${{ true }}";
      assert.notEqual(changed, line);
      writeFileSync(mutated, workflow.replace(body, body.replace(line, changed))); utimesSync(mutated, new Date(), new Date());
      const childEnv = { ...process.env, REMOTE_MACOS_MUTATION_CHILD: "1", REMOTE_MACOS_WORKFLOW_UNDER_TEST: mutated }; delete childEnv.NODE_TEST_CONTEXT;
      const target = name === JOB ? "fork manual verification never queues a self-hosted runner" : "manual profiles never run legacy jobs";
      const result = spawnSync(process.execPath, nestedTestArgs(target), { cwd: ROOT, env: childEnv, encoding: "utf8", timeout: 10000 });
      assert.notEqual(result.status, 0, `${name}\n${result.stdout}\n${result.stderr}`); assert.ok(result.stdout.includes(`✖ ${target}`), result.stdout); assert.match(result.stdout, /(?:ℹ|#) fail 1/);
      writeFileSync(mutated, workflow); utimesSync(mutated, new Date(), new Date());
    } finally { rmSync(scratch, { recursive: true, force: true }); }
  });
});
