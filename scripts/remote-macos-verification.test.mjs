import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, utimesSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { selectSteps } from "./preflight.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modulePath = process.env.REMOTE_MACOS_CONTROLLER_UNDER_TEST ?? path.join(ROOT, "scripts/remote-macos-verification.mjs");
const policy = await import(pathToFileURL(modulePath));
const { REPOSITORY, WORKFLOW, CONTROLLER, JOB, OUTPUTS, CACHE_FLAGS, ASSERTION_DRIVER, acquireTargetLease, validateCacheContext, artifactName, authenticateReceipt, checkoutContext, dispatchIdentity, importReceipt, inferencePin, readArchive, requireFreshStamp, sha256, unitKind, unitRunner, validateCapabilityFile, validateFacts, validateMachine, validateRequest, validateStamp, validateUnits } = policy;
const { loadRegistry, TESTS: ASSERTION_TESTS, MUTABLE_FILES, REGISTRY_PATH } = await import("./moss-consumer-assertions.mjs");
const SOURCE = "a".repeat(40), PIN = "b".repeat(40), BASE = "c".repeat(40), DISPATCHER = "d".repeat(40);
const sourceRef = "story/sc-26585-epic-26584-moss-image-training", base = "feature/sc-26584-moss-image-training";
const machine = () => ({ runner_name: "nax-macos-2", platform: "darwin", arch: "arm64", runtime: "26.5.2", sdk: "26.5", deployment_target: "26.2", chip: "Apple M5 Max", developer_dir: "/Applications/Xcode.app/Contents/Developer", metal_compiler: "/Applications/Xcode.app/Contents/Developer/usr/bin/metal" });
const request = (profile = "canonical-preflight") => ({ repository: REPOSITORY, profile, source_sha: SOURCE, source_ref: sourceRef, inference_sha: PIN, base_ref: base, allow_nax_unit: profile === "canonical-preflight", legacy_device_inputs: false });
const unitLedger = () => ({ limit_seconds: 300, rss_limit_bytes: 8 * 1024 ** 3, used_seconds: 3, runs: ["adapter-library", "adapter-bin", "nax-gpu"].map((kind, index) => ({ kind, pid: 100 + index, exit_code: 0, peak_rss_bytes: 1024, rss_samples: 1, seconds: 1, violation: null })) });
function cacheContext() {
  const value = { serialized: true, released: true, runner_name: "nax-macos-2", toolchain: "1.97.1", host: "aarch64-apple-darwin", sdk: "26.5", target_directory_sha256: "1".repeat(64), rustc_sha256: "2".repeat(64), cargo_config_sha256: sha256(readFileSync(path.join(ROOT, ".cargo/config.toml"))), build_flags: { ...CACHE_FLAGS } };
  return { ...value, context_sha256: sha256(JSON.stringify(value)) };
}
function syntheticAssertionResult() {
  const spec = loadRegistry(ROOT);
  const green = key => ({ test_id: key, passed: 1, failed: 0, ignored: 0, exit_code: 0 });
  const hashes = Object.fromEntries(MUTABLE_FILES.map(file => [file, "1".repeat(64)]));
  return { schema_version: 2, status: "passed", elapsed_seconds: 1, source_sha: SOURCE, registry_sha256: spec.registry_sha256, driver_sha256: sha256(readFileSync(path.join(ROOT, ASSERTION_DRIVER))), test_ids: Object.keys(ASSERTION_TESTS), baseline: Object.keys(ASSERTION_TESTS).map(green), final_green: Object.keys(ASSERTION_TESTS).map(green), cases: spec.cases.map(item => ({ id: item.id, killed: true, restored: true, red: item.tests.map(test_id => ({ test_id, passed: 0, failed: 1, ignored: 0, exit_code: 101 })), green: item.tests.map(green) })), source_before: hashes, source_after: { ...hashes }, restored: true, clean_after: true, head_after: SOURCE };
}
function capability(backend, kind) {
  const generatedFrom = { inferenceRevision: PIN };
  return kind === "runtime"
    ? { generatedFrom, schemaVersion: 2, modelMappings: {}, videoModelMappings: [], trainerMappings: {}, workerCapabilities: [], snapshot: { backend, platform: backend === "mlx" ? "macos" : "cuda", generator_ids: ["fixture"], trainer_ids: ["fixture"], generator_capabilities: [{ id: "fixture", backend }], trainer_capabilities: [{ id: "fixture", backend }] } }
    : { generatedFrom, backend, ...(kind === "audio" ? { registry: "audio" } : {}), engines: [{ id: "fixture", modality: kind === "audio" ? "audio" : "image", supportsPreview: false }] };
}
function fixture(profile = "canonical-preflight") {
  const context = { ...request(profile), cargo_config_sha256: sha256(readFileSync(path.join(ROOT, ".cargo/config.toml"))), base_sha: BASE, merge_base: BASE, changed_files: ["Cargo.toml", "config/manifests/builtin.models.jsonc"], selection_platform: profile === "canonical-preflight" ? "darwin" : null, selected_steps: profile === "canonical-preflight" ? ["rust:check (tier docs, fmt, clippy, test, build)", "candle build (not-macOS + candle)"] : [] };
  const files = {};
  if (profile === "canonical-preflight") {
    files["stamp.txt"] = Buffer.from(`preflight summary @ ${SOURCE.slice(0, 9)}, base origin/${base}:\n${context.selected_steps.map(step => `  PASS  ${step} (1s)`).join("\n")}\n`);
    files["native-units.json"] = Buffer.from(JSON.stringify(unitLedger()));
    files["preflight.log"] = Buffer.from("actual canonical command output\n");
  } else if (profile === "consumer-assertions") {
    files["assertions.json"] = Buffer.from(JSON.stringify(syntheticAssertionResult()));
    files["assertions.log"] = Buffer.from("synthetic assertion protocol fixture, not remote execution\n");
  } else {
    OUTPUTS.forEach((file, index) => { files[`payload/${file}`] = Buffer.from(index < 3 ? JSON.stringify(capability(index === 1 ? "candle" : "mlx", ["media", "audio", "runtime"][index])) : file.endsWith(".md") ? "# Static generated fixture\n" : file.endsWith(".jsonc") ? "// generated fixture\n{}\n" : "{}\n"); });
  }
  if (profile === "facts-and-projections") files["facts.log"] = Buffer.from("actual static producer output\n");
  const receipt = { schema_version: 1, status: "passed", ...context, workflow_sha: DISPATCHER, workflow_ref: `${REPOSITORY}/${WORKFLOW}@refs/heads/${sourceRef}`, run_id: 41, run_attempt: 2, job: JOB, runner_name: "nax-macos-2", machine: machine(), cache_context: cacheContext(), clean_before: true, clean_after: profile !== "facts-and-projections", stamp_created_this_run: profile === "canonical-preflight", command: profile === "canonical-preflight" ? ["npm", "run", "preflight", "--", "--base", `origin/${base}`] : profile === "consumer-assertions" ? ["consumer-assertions", ASSERTION_DRIVER] : ["facts-and-projections"], files: Object.fromEntries(Object.entries(files).map(([name, bytes]) => [name, sha256(bytes)])) };
  const controls = Object.fromEntries([WORKFLOW, CONTROLLER, "scripts/preflight.mjs", "scripts/lib/jsonc.mjs", ASSERTION_DRIVER, REGISTRY_PATH].map(file => [file, sha256(readFileSync(path.join(ROOT, file)))]));
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

for (const profile of ["facts-and-projections", "consumer-assertions", "canonical-preflight"]) test(`authenticated ${profile} accepts a different valid dispatcher SHA, not a relabeled checkout`, async () => { const f = fixture(profile); await f.verify(); assert.notEqual(f.receipt.workflow_sha, f.context.source_sha); });

const requestFaults = {
  "foreign repository": value => { value.repository = "fork/SceneWorks"; },
  "unknown profile": value => { value.profile = "anything"; },
  "short source SHA": value => { value.source_sha = SOURCE.slice(0, 8); },
  "short inference SHA": value => { value.inference_sha = PIN.slice(0, 8); },
  "pull ref": value => { value.source_ref = "refs/pull/2/head"; },
  "external ref": value => { value.source_ref = "other/repo@main"; },
  "story base": value => { value.base_ref = sourceRef; },
  "legacy GPU mixing": value => { value.legacy_device_inputs = true; },
  "missing NAX admission": value => { value.allow_nax_unit = false; },
};
for (const [name, mutate] of Object.entries(requestFaults)) test(`request rejects ${name}`, () => { const value = request(); mutate(value); assert.throws(() => validateRequest(value)); });
test("facts request rejects device admission", () => { const value = request("facts-and-projections"); value.allow_nax_unit = true; assert.throws(() => validateRequest(value)); });
for (const [name, patch] of Object.entries({ "primary runner": { runner_name: "nax-macos" }, "Linux": { platform: "linux" }, "Intel": { arch: "x64" }, "old runtime": { runtime: "26.1" }, "old SDK": { sdk: "26.0" }, "hosted deployment override": { deployment_target: "15.0" }, "pre-M5": { chip: "Apple M4 Max" }, "CLT instead of Xcode": { developer_dir: "/Library/Developer/CommandLineTools" }, "missing Metal": { metal_compiler: "" } })) test(`machine rejects ${name}`, () => assert.throws(() => validateMachine({ ...machine(), ...patch }, "canonical-preflight")));
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
  "receipt scope": f => { f.receipt.selected_steps = []; },
  "Linux consumer scope": f => { f.context.selection_platform = "linux"; },
  "receipt changed sources": f => { f.receipt.changed_files = []; },
  "dirty initial source": f => { f.receipt.clean_before = false; },
  "dirty final source": f => { f.receipt.clean_after = false; },
  "older reused stamp": f => { f.receipt.stamp_created_this_run = false; },
  "receipt run": f => { f.receipt.run_id = 40; },
  "receipt attempt": f => { f.receipt.run_attempt = 1; },
  "receipt job": f => { f.receipt.job = "other"; },
  "receipt runner": f => { f.receipt.runner_name = "nax-macos"; },
  "receipt GPU permission": f => { f.receipt.allow_nax_unit = false; },
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
  "assertion driver code": f => { f.alteredControl = `${DISPATCHER}:${ASSERTION_DRIVER}`; },
  "case registry code": f => { f.alteredControl = `${DISPATCHER}:${REGISTRY_PATH}`; },
  "cache not released": f => { f.receipt.cache_context.released = false; },
  "cache compiler": f => { f.receipt.cache_context.toolchain = "other"; },
  "candidate selector code": f => { f.alteredControl = `${SOURCE}:scripts/preflight.mjs`; },
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
  "missing canonical step": f => { f.jobs.jobs[0].steps.shift(); },
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
  "missing artifact member": f => { delete f.files["stamp.txt"]; },
  "extra artifact executable": f => { f.files["run-me.py"] = Buffer.from("not executable data"); },
  "member digest": f => { f.files["stamp.txt"] = Buffer.from("forged"); },
  "failed stamp row": f => { replaceMember(f, "stamp.txt", f.files["stamp.txt"].toString().replace("PASS", "FAIL")); },
  "wrong stamp base": f => { replaceMember(f, "stamp.txt", f.files["stamp.txt"].toString().replace(`origin/${base}`, "origin/main")); },
  "missing unit target": f => { const value = unitLedger(); value.runs.pop(); value.used_seconds = 2; replaceMember(f, "native-units.json", JSON.stringify(value)); },
  "late unit": f => { const value = unitLedger(); value.used_seconds = 301; replaceMember(f, "native-units.json", JSON.stringify(value)); },
  "unit RSS": f => { const value = unitLedger(); value.runs[0].peak_rss_bytes = 9 * 1024 ** 3; replaceMember(f, "native-units.json", JSON.stringify(value)); },
  "unit negative telemetry": f => { const value = unitLedger(); value.runs[0].peak_rss_bytes = -1; replaceMember(f, "native-units.json", JSON.stringify(value)); },
  "unobserved unit": f => { const value = unitLedger(); value.runs[0].rss_samples = 0; replaceMember(f, "native-units.json", JSON.stringify(value)); },
  "reset cumulative ledger": f => { const value = unitLedger(); value.used_seconds = 1; replaceMember(f, "native-units.json", JSON.stringify(value)); },
};
for (const [name, mutate] of Object.entries(faults)) test(`fault rejects ${name}`, async () => { const f = fixture(); mutate(f); await assert.rejects(f.verify); });

for (const [name, mutate] of Object.entries({ "wrong produced pin": value => { value.generatedFrom.inferenceRevision = BASE; }, "CPU runtime": value => { value.snapshot.backend = "cpu"; }, "CUDA on Mac": value => { value.snapshot.platform = "cuda"; }, "unknown runtime schema": value => { value.schemaVersion = 0; }, "duplicate identities": value => { value.snapshot.generator_ids.push("fixture"); }, "foreign descriptor": value => { value.snapshot.trainer_capabilities[0].backend = "candle"; }, "missing mapping": value => { delete value.trainerMappings; } })) test(`facts rejects ${name}`, async () => { const f = fixture("facts-and-projections"), value = capability("mlx", "runtime"); mutate(value); replaceMember(f, `payload/${OUTPUTS[2]}`, JSON.stringify(value)); await assert.rejects(f.verify); });
test("facts rejects malformed generated JSONC", async () => { const f = fixture("facts-and-projections"); replaceMember(f, `payload/${OUTPUTS[4]}`, "invalid JSONC"); await assert.rejects(f.verify); });
test("provenance-only drift is not a canonical currency gate", () => {
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
test("old stamp cannot stand in for a new canonical execution", () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "remote-macos-freshness-"));
  try {
    requireFreshStamp(root, SOURCE);
    mkdirSync(path.join(root, "sceneworks-preflight")); writeFileSync(path.join(root, "sceneworks-preflight", SOURCE), "old synthetic proof");
    assert.throws(() => requireFreshStamp(root, SOURCE), /fresh checkout/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
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
test("authentic importer preserves actual stamp bytes and independent transport evidence", async () => {
  const f = fixture(), local = localFixture(f);
  try {
    await importReceipt(local.root, request(), local.options);
    const imported = path.join(local.root, ".git/sceneworks-preflight", SOURCE);
    assert.deepEqual(readFileSync(imported), f.files["stamp.txt"]);
    assert.equal(JSON.parse(readFileSync(path.join(local.root, ".git/sceneworks-preflight-receipts", SOURCE, "41-2-83.json"))).artifact_id, 83);
  } finally { rmSync(local.root, { recursive: true, force: true }); }
});
for (const [name, change] of Object.entries({ "failed actual run": (f) => { f.run.conclusion = "failure"; }, "local changed during authentication": (_f, local) => { let calls = 0; local.options.contextFor = () => { if (++calls > 1) throw new Error("dirty local source"); return _f.context; }; }, "unexpected artifact code": (f) => { f.files["evil.py"] = Buffer.from("not executed"); } })) test(`import writes no stamp for ${name}`, async () => {
  const f = fixture(); change(f, { options: {} }); const local = localFixture(f); if (name === "local changed during authentication") change(f, local);
  try { await assert.rejects(() => importReceipt(local.root, request(), local.options)); assert.equal(existsSync(path.join(local.root, ".git/sceneworks-preflight", SOURCE)), false); }
  finally { rmSync(local.root, { recursive: true, force: true }); }
});
test("consumer assertion evidence is authenticated but can never create a canonical stamp", async () => {
  const f = fixture("consumer-assertions"), local = localFixture(f);
  try {
    await importReceipt(local.root, request("consumer-assertions"), local.options);
    assert.equal(existsSync(path.join(local.root, ".git/sceneworks-consumer-assertions", SOURCE, "41-2-83.json")), true);
    assert.equal(existsSync(path.join(local.root, ".git/sceneworks-preflight", SOURCE)), false);
  } finally { rmSync(local.root, { recursive: true, force: true }); }
});
for (const [name, change] of Object.entries({ registry: value => { value.registry_sha256 = "0".repeat(64); }, implementation: value => { value.driver_sha256 = "0".repeat(64); }, survivor: value => { value.cases[0].killed = false; }, missing: value => { value.cases.pop(); }, counts: value => { value.cases[0].green[0].passed = 0; }, wrongTest: value => { value.cases[0].green[0].test_id = "protocol"; }, restore: value => { value.source_after[MUTABLE_FILES[0]] = "wrong"; }, carry: value => { value.carried = { status: "carried-prior-proof-not-executed" }; } })) test(`consumer evidence rejects ${name}`, async () => {
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
test("facts importer writes only authenticated derived outputs and no stamp", async () => {
  const f = fixture("facts-and-projections"), local = localFixture(f);
  try { await importReceipt(local.root, request("facts-and-projections"), local.options); for (const file of OUTPUTS) assert.deepEqual(readFileSync(path.join(local.root, file)), f.files[`payload/${file}`]); assert.equal(existsSync(path.join(local.root, ".git/sceneworks-preflight", SOURCE)), false); }
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

test("checkout identity uses exact clean owned source, ancestry and recomputed Mac scope", () => {
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
  assert.deepEqual(result.selected_steps, selectSteps(result.changed_files, false, ROOT, "darwin").map(step => step.name));
  assert.ok(calls.some(args => args.join(" ") === `merge-base --is-ancestor HEAD origin/${sourceRef}`));
  for (const fault of ["dirty", "head", "origin", "ancestry", "pin", "empty scope"]) {
    const changed = (root, ...args) => {
      if (fault === "dirty" && args[0] === "status") return " M source.rs";
      if (fault === "head" && args.join(" ") === "rev-parse HEAD") return BASE;
      if (fault === "origin" && args[0] === "remote") return "https://github.com/fork/SceneWorks.git";
      if (fault === "ancestry" && args.includes("--is-ancestor")) throw new Error("not an ancestor");
      if (fault === "empty scope" && args[0] === "diff") return "";
      return executeGit(root, ...args);
    };
    assert.throws(() => checkoutContext(ROOT, request(), { executeGit: changed, readPin: () => fault === "pin" ? BASE : PIN, platform: "darwin" }), fault);
  }
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
function condition(name, event, profile, relevant = "true", result = "success", target = "feature/sc-26584-moss-image-training", repository = REPOSITORY) {
  const expression = /\n    if: \$\{\{ (.+) \}\}/.exec(`\n${jobBody(name)}`)?.[1]; assert.ok(expression, name);
  const github = { event_name: event, repository, event: { pull_request: { base: { ref: target }, head: { repo: { full_name: REPOSITORY } } } } };
  return Boolean(Function("github", "inputs", "needs", "cancelled", `return (${expression});`)(github, { verification_profile: profile }, { changes: { result, outputs: { relevant } } }, () => false));
}
test("manual profiles never run legacy jobs", () => {
  for (const profile of ["facts-and-projections", "consumer-assertions", "canonical-preflight", "unknown-unsafe-value"]) {
    for (const name of ["changes", "macos-checks", "nax-worker"]) for (const result of ["success", "skipped", "failure"]) assert.equal(condition(name, "workflow_dispatch", profile, "true", result), false, `${profile}/${name}/${result}`);
    assert.equal(condition(JOB, "workflow_dispatch", profile), true);
  }
});
test("fork manual verification never queues a self-hosted runner", () => {
  assert.equal(condition(JOB, "workflow_dispatch", "canonical-preflight", "true", "success", base, "fork/SceneWorks"), false);
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
  assert.match(body, /command -v x86_64-unknown-linux-gnu-gcc/);
  assert.match(body, /remote-macos-verification\.mjs" run/);
});

function fixtureNode(root, name) {
  const executable = path.join(root, "deps", name + (process.platform === "win32" ? ".exe" : ""));
  // Alias this trusted Node runtime, not a downloaded executable. A cross-volume fixture copies it.
  try { linkSync(process.execPath, executable); }
  catch (error) { if (!["EXDEV", "EPERM", "EACCES"].includes(error.code)) throw error; copyFileSync(process.execPath, executable); }
  return executable;
}
test("native runner classifies all selected adapter and NAX units without running a device", () => {
  assert.equal(unitKind(path.join(os.tmpdir(), "deps", "memory_mlx_adapter-abc")), "adapter-bin");
  assert.equal(unitKind(path.join(os.tmpdir(), "deps", "sceneworks_memory_adapter-abc")), "adapter-library");
  assert.equal(unitKind(path.join(os.tmpdir(), "deps", "nax_guard-abc")), "nax-gpu");
  assert.equal(unitKind(path.join(os.tmpdir(), "deps", "sceneworks_worker-abc")), null);
  assert.equal(unitKind(path.join(os.tmpdir(), "memory_mlx_adapter-abc")), null);
  validateUnits(unitLedger());
});
test("synthetic CPU-only runner forwards exits and accumulates one ledger", async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "remote-macos-unit-synthetic-"));
  try {
    mkdirSync(path.join(root, "deps")); const ledger = path.join(root, "ledger.json");
    writeFileSync(ledger, JSON.stringify({ limit_seconds: 300, rss_limit_bytes: 8 * 1024 ** 3, used_seconds: 0, runs: [] }));
    const exe = fixtureNode(root, "memory_mlx_adapter-fixture");
    const env = { ...process.env, REMOTE_MACOS_UNIT_LEDGER: ledger, REMOTE_MACOS_ALLOW_NAX_UNIT: "false" }; delete env.NODE_TEST_CONTEXT;
    const argumentsFor = code => [exe, "-e", `setTimeout(() => process.exit(${code}), 300)`];
    assert.equal(await unitRunner(argumentsFor(7), env), 7); const first = JSON.parse(readFileSync(ledger));
    assert.equal(await unitRunner(argumentsFor(0), env), 0); const second = JSON.parse(readFileSync(ledger));
    assert.ok(second.used_seconds > first.used_seconds && second.runs.length === 2);
    const exhausted = { ...second, used_seconds: 300 }; writeFileSync(ledger, JSON.stringify(exhausted)); await assert.rejects(() => unitRunner(argumentsFor(0), env), /budget/);
    await assert.rejects(() => unitRunner([path.join(root, "deps", "nax_guard-fixture")], env), /not admitted/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("synthetic NAX-class CPU fixture drains safely on a budget violation and cannot pass", async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "remote-macos-unit-drain-synthetic-"));
  try {
    mkdirSync(path.join(root, "deps")); const ledger = path.join(root, "ledger.json"), marker = path.join(root, "drained");
    writeFileSync(ledger, JSON.stringify({ limit_seconds: 300, rss_limit_bytes: 8 * 1024 ** 3, used_seconds: 299.99, runs: [] }));
    const exe = fixtureNode(root, "nax_guard-synthetic-not-metal");
    const env = { ...process.env, REMOTE_MACOS_UNIT_LEDGER: ledger, REMOTE_MACOS_ALLOW_NAX_UNIT: "true" }; delete env.NODE_TEST_CONTEXT;
    assert.equal(await unitRunner([exe, "-e", `setTimeout(() => require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'drained'), 300)`], env), 124);
    assert.equal(readFileSync(marker, "utf8"), "drained");
    assert.match(JSON.parse(readFileSync(ledger)).runs[0].violation, /time ceiling/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

if (!process.env.REMOTE_MACOS_MUTATION_CHILD) test("named source bypass mutations make their original tests red", async t => {
  const controller = readFileSync(path.join(ROOT, CONTROLLER), "utf8");
  const cases = [
    ["RV-cache-flags", "cache context rejects changed compiler flags even with a self-consistent data hash", 'equal(context.build_flags, CACHE_FLAGS)', 'true'],
    ["RV-cache-exclusion", "dedicated target cache lease prevents overlap and never steals or releases another owner", 'mkdirSync(lease);', 'mkdirSync(lease, { recursive: true });'],
    ["RV-consumer-proof", "consumer evidence rejects survivor", 'validateAssertionReceipt(JSON.parse(files["assertions.json"].toString()), spec, context.source_sha, controls[ASSERTION_DRIVER]);', 'void spec;'],
    ["RV-no-currency-gate", "provenance-only drift is not a canonical currency gate", 'pin === undefined || revision === pin', 'revision === pin'],
    ["RV-run-authority", "fault rejects run outcome", 'run.conclusion === "success"', 'true'],
    ["RV-scope", "fault rejects receipt scope", 'equal(receipt.changed_files, context.changed_files) && equal(receipt.selected_steps, context.selected_steps)', 'true'],
    ["RV-archive-digest", "archive digest rejects substituted bytes", 'digest === `sha256:${sha256(bytes)}`', 'true'],
    ["RV-archive-members", "archive rejects excess members before reading their data", 'assert len(z.infolist())<=64, "too many artifact members"', 'pass'],
    ["RV-primary-machine", "machine rejects primary runner", 'machine.runner_name === "nax-macos-2"', 'true'],
    ["RV-host-scope", "fault rejects Linux consumer scope", 'receipt.selection_platform === context.selection_platform && (context.profile !== "canonical-preflight" || context.selection_platform === "darwin")', 'true'],
    ["RV-fresh-stamp", "old stamp cannot stand in for a new canonical execution", '!existsSync(path.join(common, "sceneworks-preflight", sha))', 'true'],
    ["RV-native-budget", "synthetic CPU-only runner forwards exits and accumulates one ledger", 'ledger.used_seconds < 300', 'true'],
    ["RV-native-safe-drain", "synthetic NAX-class CPU fixture drains safely on a budget violation and cannot pass", 'entry.violation && kind !== "nax-gpu" && !stopping', 'entry.violation && !stopping'],
  ];
  for (const [name, target, old, replacement] of cases) await t.test(name, () => {
    const scratch = mkdtempSync(path.join(os.tmpdir(), "remote-macos-source-mutation-"));
    try {
      assert.equal(controller.split(old).length - 1, 1, name);
      mkdirSync(path.join(scratch, "scripts/lib"), { recursive: true });
      const mutated = path.join(scratch, CONTROLLER);
      writeFileSync(mutated, controller.replace(old, replacement)); utimesSync(mutated, new Date(), new Date());
      for (const file of ["scripts/preflight.mjs", "scripts/lib/jsonc.mjs", ASSERTION_DRIVER, REGISTRY_PATH]) writeFileSync(path.join(scratch, file), readFileSync(path.join(ROOT, file)));
      const childEnv = { ...process.env, REMOTE_MACOS_MUTATION_CHILD: "1", REMOTE_MACOS_CONTROLLER_UNDER_TEST: mutated };
      // A new test-runner process must not inherit the outer worker's private IPC protocol.
      delete childEnv.NODE_TEST_CONTEXT;
      const result = spawnSync(process.execPath, ["--test", `--test-name-pattern=^${target}$`, fileURLToPath(import.meta.url)], { cwd: ROOT, env: childEnv, encoding: "utf8", timeout: 10000 });
      assert.notEqual(result.status, 0, `${name}\n${result.stdout}\n${result.stderr}`); assert.ok(result.stdout.includes(`✖ ${target}`), result.stdout); assert.match(result.stdout, /(?:ℹ|#) fail 1/); assert.doesNotMatch(result.stderr, /SyntaxError|ERR_MODULE_NOT_FOUND/);
      writeFileSync(mutated, controller); utimesSync(mutated, new Date(), new Date());
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
      const result = spawnSync(process.execPath, ["--test", `--test-name-pattern=^${target}$`, fileURLToPath(import.meta.url)], { cwd: ROOT, env: childEnv, encoding: "utf8", timeout: 10000 });
      assert.notEqual(result.status, 0, `${name}\n${result.stdout}\n${result.stderr}`); assert.ok(result.stdout.includes(`✖ ${target}`), result.stdout); assert.match(result.stdout, /(?:ℹ|#) fail 1/);
      writeFileSync(mutated, workflow); utimesSync(mutated, new Date(), new Date());
    } finally { rmSync(scratch, { recursive: true, force: true }); }
  });
});
