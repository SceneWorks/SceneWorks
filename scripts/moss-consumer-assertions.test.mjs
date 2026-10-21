import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
const modulePath = process.env.MOSS_ASSERTIONS_UNDER_TEST ?? fileURLToPath(new URL("./moss-consumer-assertions.mjs", import.meta.url));
const { bodySpan, digest, executeCpuCommand, loadRegistry, MUTABLE_FILES, ownedCpuTree, parseTestResult, patchText, REGISTRY_PATH, runAssertions, TESTS, validateAssertionReceipt } = await import(pathToFileURL(modulePath));

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHA = "a".repeat(40);
const driverHash = digest(readFileSync(modulePath));
function output(exact, red = false, count = 1) {
  return `${red ? `thread '${exact}' panicked at fixture.rs:1:1:\nassertion failed: fixture\n` : ""}test ${exact} ... ${red ? "FAILED" : "ok"}\n\ntest result: ${red ? "FAILED" : "ok"}. ${red ? 0 : count} passed; ${red ? count : 0} failed; 0 ignored; 0 measured; 999 filtered out; finished in 0.01s\n`;
}
function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-assertions-synthetic-"));
  for (const file of MUTABLE_FILES) { mkdirSync(path.dirname(path.join(root, file)), { recursive: true }); writeFileSync(path.join(root, file), "fn fixture() { let value = 1; }\n"); }
  const original = Object.fromEntries(MUTABLE_FILES.map(file => [file, readFileSync(path.join(root, file))]));
  const spec = { schema_version: 2, registry_sha256: "1".repeat(64), cases: [{ id: "SW-synthetic", tests: ["target"], patches: [{ file: MUTABLE_FILES[0], op: "replace", old: "value = 1", new: "value = 2" }] }] };
  const state = { calls: [], fault: null, clock: 0 };
  const execute = async (_root, args, _log, _limit, control) => {
    state.calls.push(args); state.clock += 1;
    assert.equal(args[0], "cargo"); assert.ok(args.includes("--lib"));
    assert.ok(!args.includes("--ignored") && !args.includes("--all-targets"));
    if (args.includes("--no-run")) return { exit: 0, output: "compiled without executing tests" };
    const exact = args[args.indexOf("--lib") + 1];
    const red = MUTABLE_FILES.some(file => !readFileSync(path.join(root, file)).equals(original[file]));
    if (red && state.fault === "timeout") throw new Error("managed deadline");
    if (red && state.fault === "cancel") { control.cancelled = true; throw new Error("cancelled"); }
    if (red && state.fault === "compile") return { exit: 101, output: "error[E0308]: incompatible fixture" };
    return { exit: red && state.fault !== "survive" ? 101 : 0, output: output(exact, red && state.fault !== "survive", state.fault === "zero" ? 0 : 1) };
  };
  const inspect = () => ({ head: SHA, clean: true });
  const options = { spec, execute, inspect, now: () => state.clock, signals: false, budgetMs: 1000 };
  return { root, original, spec, state, options, output: path.join(root, "evidence") };
}
function restored(f) { return MUTABLE_FILES.every(file => readFileSync(path.join(f.root, file)).equals(f.original[file])); }

test("closed committed inventory executes all 74 cases across seven exact tests", () => {
  const spec = loadRegistry(ROOT);
  assert.equal(spec.cases.length, 74); assert.equal(Object.keys(TESTS).length, 7);
  assert.deepEqual(spec.cases.slice(0, 3).map(item => item.id), ["SW-load-text-encoder", "SW-thin-explicit-options", "SW-result-backend"]);
  // Three independent fail-fast cases follow baseline; the remaining 71 retain their order.
  let workerSeen = false;
  for (const item of spec.cases.slice(3)) {
    const worker = item.tests.some(key => TESTS[key][0] === "sceneworks-worker");
    if (worker) workerSeen = true;
    assert.ok(worker || !workerSeen, "core cases must precede worker cases");
  }
  for (const id of ["SW-Base-official-source", "SW-SFT-official-source", "SW-compute-demoted", "SW-thin-blank-sampler", "SW-validate-modules-array", "SW-validate-modules-members", "SW-validate-decomp-integer", "SW-validate-decomp-positive", "SW-validate-decomp-overflow", "SW-validate-valid-decomp"]) assert.ok(spec.cases.some(item => item.id === id), id);
  // This mutation must compile before the protocol assertion can kill it. LoadSpec's field is
  // Option<WeightsSource>, not Option<PathBuf>; a bare string's .into() is not a valid source.
  assert.equal(spec.cases.find(item => item.id === "SW-load-text-encoder").patches[0].new,
    'spec.text_encoder = Some(gen_core::WeightsSource::Dir(std::path::PathBuf::from("wrong")));');
  assert.equal(Object.hasOwn(spec, "carried"), false);
  assert.equal(spec.registry_sha256, digest(readFileSync(path.join(ROOT, REGISTRY_PATH))));
  for (const item of spec.cases) {
    const sources = new Map();
    for (const patch of item.patches) {
      const before = sources.get(patch.file) ?? readFileSync(path.join(ROOT, patch.file), "utf8");
      sources.set(patch.file, patchText(before, patch));
    }
  }
  for (const id of ["SW-generation-base-Candle-adapter", "SW-generation-sft-Candle-adapter", "SW-generation-conditioning-carrier-refusal", "SW-generation-quant-tier-refusal", "SW-generation-edit-mode-refusal"]) assert.ok(spec.cases.some(item => item.id === id), id);
});
test("byte patching handles Rust strings, raw strings, characters, nested comments and lifetimes", () => {
  const source = `fn sample<'a>() { let a = "}"; let b = r#"{ raw }"#; let c = '}'; /* outer { /* inner } */ } */ let value = 1; }`;
  const [a, b] = bodySpan(source, "fn sample"); assert.equal(source[b], "}"); assert.ok(source.slice(a, b).includes("value = 1"));
  assert.match(patchText(source, { op: "replace", anchor: "fn sample", old: "value = 1", new: "value = 2" }), /value = 2/);
  for (const text of ["fn absent() {}", source + source]) assert.throws(() => patchText(text, { op: "replace", anchor: "fn sample", old: "value = 1", new: "value = 2" }));
});
test("positive and assertion-red parser rejects compilation, crash, no selection and wrong named test", () => {
  const exact = TESTS.target[1];
  assert.equal(parseTestResult(output(exact), 0, exact, "green").passed, 1);
  assert.equal(parseTestResult(output(exact, true), 101, exact, "red").failed, 1);
  const interleaved = `test ${exact} ... \nthread '${exact}' panicked at fixture.rs:1:1:\nassertion failed: fixture\nFAILED\nfailures:\n    ${exact}\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 999 filtered out; finished in 0.01s\n`;
  assert.equal(parseTestResult(interleaved, 101, exact, "red").failed, 1);
  for (const [text, exit, expected] of [[output(exact, false, 0), 0, "green"], ["error[E0308]", 101, "red"], ["process failed with SIGSEGV", 1, "red"], [output("other", true), 101, "red"], [output(exact), 0, "red"], [output(exact, true, 2), 101, "red"]]) assert.throws(() => parseTestResult(text, exit, exact, expected));
});
test("synthetic executor proves compile-once, exact positive counts, mutation red, touch restore and final green", async () => {
  const f = fixture();
  try {
    const result = await runAssertions(f.root, f.output, f.options);
    assert.equal(result.status, "passed"); assert.equal(result.baseline.length, 7); assert.equal(result.final_green.length, 7);
    assert.deepEqual(result.cases[0].red.map(row => row.failed), [1]); assert.deepEqual(result.cases[0].green.map(row => row.passed), [1]);
    assert.equal(f.state.calls.filter(args => args.includes("--no-run")).length, 1);
    assert.equal(f.state.calls.find(args => args.includes("--no-run")).filter(arg => arg === "--verbose").length, 1);
    assert.ok(f.state.calls.filter(args => !args.includes("--no-run")).every(args => !args.includes("--verbose") && !args.includes("-vv")));
    assert.equal(restored(f), true); assert.equal(Object.hasOwn(result, "carried"), false);
    validateAssertionReceipt(result, f.spec, SHA, driverHash);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});
for (const fault of ["compile", "timeout", "cancel", "survive", "zero"]) test(`synthetic ${fault} cannot become RED proof and restores all source`, async () => {
  const f = fixture(); f.state.fault = fault;
  try { const result = await runAssertions(f.root, f.output, f.options); assert.equal(result.status, "failed"); assert.equal(restored(f), true); assert.equal(result.cases.every(row => !row.killed), true); }
  finally { rmSync(f.root, { recursive: true, force: true }); }
});
test("receipt validates case IDs, exact selected test counts, implementation hashes and complete restoration", async () => {
  const f = fixture();
  try {
    const valid = await runAssertions(f.root, f.output, f.options);
    for (const change of [value => { value.registry_sha256 = "0".repeat(64); }, value => { value.driver_sha256 = "0".repeat(64); }, value => { value.cases[0].id = "foreign"; }, value => { value.cases[0].killed = false; }, value => { value.cases[0].red[0].failed = 0; }, value => { value.cases[0].green[0].test_id = "protocol"; }, value => { value.baseline[0].passed = 0; }, value => { value.final_green[0].test_id = "protocol"; }, value => { value.source_after[MUTABLE_FILES[0]] = "wrong"; }, value => { value.clean_after = false; }, value => { value.head_after = "b".repeat(40); }, value => { value.carried = { status: "carried-prior-proof-not-executed" }; }]) {
      const bad = structuredClone(valid); change(bad); assert.throws(() => validateAssertionReceipt(bad, f.spec, SHA, driverHash));
    }
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});

test("restore rewrites and touches only changed files while auditing every mutable source", async () => {
  const f = fixture();
  try {
    for (const file of MUTABLE_FILES) utimesSync(path.join(f.root, file), new Date(1000), new Date(1000));
    const before = Object.fromEntries(MUTABLE_FILES.map(file => [file, statSync(path.join(f.root, file), { bigint: true })]));
    const execute = f.options.execute;
    let corrupted = false;
    f.options.execute = async (...args) => {
      const red = !readFileSync(path.join(f.root, MUTABLE_FILES[0])).equals(f.original[MUTABLE_FILES[0]]);
      if (red && !corrupted) { writeFileSync(path.join(f.root, MUTABLE_FILES[1]), "unexpected mutable-file change"); corrupted = true; }
      return execute(...args);
    };
    const result = await runAssertions(f.root, f.output, f.options);
    assert.equal(result.status, "passed"); assert.equal(restored(f), true);
    for (const [index, file] of MUTABLE_FILES.entries()) {
      const after = statSync(path.join(f.root, file), { bigint: true });
      assert.equal(after.ino, before[file].ino, file);
      if (index < 2) assert.ok(after.mtimeNs > before[file].mtimeNs, `changed source must get a fresh restore timestamp: ${file}`);
      else assert.equal(after.mtimeNs, before[file].mtimeNs, `untouched dependency was rewritten: ${file}`);
    }
    assert.deepEqual(result.source_before, result.source_after);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});
for (const fault of ["partial write", "failed apply", "cancel", "timeout"]) test(`selective restore repairs all changed files after ${fault}`, async () => {
  const f = fixture();
  try {
    const execute = f.options.execute;
    if (fault === "failed apply") f.spec.cases[0].patches.push({ file: MUTABLE_FILES[1], op: "replace", old: "value = 1", new: "value = 3" });
    f.options.execute = async (...args) => {
      const command = args[1], control = args[4];
      if (fault === "failed apply" && command.includes(TESTS.protocol[1])) {
        const result = await execute(...args);
        writeFileSync(path.join(f.root, MUTABLE_FILES[1]), "anchor changed after validation");
        return result;
      }
      if (!readFileSync(path.join(f.root, MUTABLE_FILES[0])).equals(f.original[MUTABLE_FILES[0]])) {
        writeFileSync(path.join(f.root, MUTABLE_FILES[2]), "unexpected partial write");
        if (fault === "cancel") control.cancelled = true;
        throw new Error(fault);
      }
      return execute(...args);
    };
    const result = await runAssertions(f.root, f.output, f.options);
    assert.equal(result.status, "failed"); assert.equal(restored(f), true);
    assert.deepEqual(result.source_before, result.source_after);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});

test("unreaped compiler fails closed without racing source restoration", async () => {
  const f = fixture();
  try {
    const execute = f.options.execute;
    f.options.execute = async (...args) => {
      if (!readFileSync(path.join(f.root, MUTABLE_FILES[0])).equals(f.original[MUTABLE_FILES[0]])) {
        args[4].unsafeToRestore = true;
        throw new Error("managed deadline; cleanup: EPERM, owned compiler still live");
      }
      return execute(...args);
    };
    const result = await runAssertions(f.root, f.output, f.options);
    assert.equal(result.status, "failed"); assert.equal(result.restored, false);
    assert.equal(restored(f), false); assert.notDeepEqual(result.source_before, result.source_after);
    assert.match(result.failure, /deadline.*EPERM/);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});
test("selective restore handles successful multiple-file mutations", async () => {
  const f = fixture();
  try {
    f.spec.cases[0].patches.push({ file: MUTABLE_FILES[1], op: "replace", old: "value = 1", new: "value = 3" });
    const result = await runAssertions(f.root, f.output, f.options);
    assert.equal(result.status, "passed"); assert.equal(restored(f), true);
    assert.deepEqual(result.source_before, result.source_after);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});

const posixOnly = { skip: process.platform === "win32", timeout: 8000 };
const tinyNode = code => [process.execPath, "--max-old-space-size=24", "-e", `setInterval(()=>{if(process.memoryUsage().rss>128*1024**2)process.exit(91)},25).unref();${code}`];
function gone(pid) { const r = spawnSync("ps", ["-o", "stat=", "-p", String(pid)], { encoding: "utf8" }); return r.status === 1 || /^\s*Z/.test(r.stdout); }
test("CPU command normal close never sends a post-close signal", posixOnly, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-cpu-normal-")), state = { cancelled: false }, signals = [];
  try {
    const result = await executeCpuCommand(root, tinyNode("setTimeout(()=>{},200)"), path.join(root, "log"), 2000, state, { send: (pid, signal) => { signals.push([pid, signal]); process.kill(pid, signal); }, graceMs: 100 });
    assert.equal(result.exit, 0); assert.deepEqual(signals, []); assert.equal(state.child, null);
    assert.deepEqual(state.cleanup, { errors: [], remaining: [], unproven: [] });
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("CPU child inherits the selected incremental recipe and logs only its whitelist", posixOnly, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-cpu-recipe-")), state = { cancelled: false };
  const oldIncremental = process.env.CARGO_INCREMENTAL, oldSecret = process.env.MOSS_FIXTURE_SECRET, originalLog = console.log, messages = [];
  process.env.CARGO_INCREMENTAL = "1"; process.env.MOSS_FIXTURE_SECRET = "not-a-recipe-field";
  console.log = text => messages.push(text);
  try {
    const result = await executeCpuCommand(root, tinyNode("console.log('incremental='+process.env.CARGO_INCREMENTAL);setTimeout(()=>{},200)"), path.join(root, "log"), 2000, state, { graceMs: 100 });
    assert.equal(result.exit, 0); assert.match(result.output, /incremental=1/);
    const records = messages.map(text => JSON.parse(text));
    assert.equal(records[0].child_recipe.CARGO_INCREMENTAL, "1");
    assert.deepEqual(Object.keys(records[0].child_recipe), ["CARGO_INCREMENTAL", "CARGO_BUILD_JOBS", "CARGO_PROFILE_TEST_DEBUG", "CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP", "RUSTUP_TOOLCHAIN"]);
    assert.ok(Number.isFinite(records[1].elapsed_ms)); assert.equal(records[1].exit_code, 0);
    assert.ok(messages.every(text => !text.includes("not-a-recipe-field")));
  } finally {
    console.log = originalLog;
    if (oldIncremental === undefined) delete process.env.CARGO_INCREMENTAL; else process.env.CARGO_INCREMENTAL = oldIncremental;
    if (oldSecret === undefined) delete process.env.MOSS_FIXTURE_SECRET; else process.env.MOSS_FIXTURE_SECRET = oldSecret;
    rmSync(root, { recursive: true, force: true });
  }
});
test("CPU cancellation terminates and reaps before returning", posixOnly, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-cpu-cancel-")), state = { cancelled: false };
  const timer = setTimeout(() => { state.cancelled = true; }, 350);
  try {
    await assert.rejects(() => executeCpuCommand(root, tinyNode("setInterval(()=>{},1000)"), path.join(root, "log"), 2000, state, { graceMs: 100 }), /managed deadline\/cancellation/);
    assert.equal(state.child, null); assert.equal(state.unsafeToRestore, false);
  } finally { clearTimeout(timer); rmSync(root, { recursive: true, force: true }); }
});
test("CPU spawn errors are failures and receive no speculative process signal", posixOnly, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-cpu-spawn-")), state = { cancelled: false }, sent = [];
  try {
    await assert.rejects(() => executeCpuCommand(root, [path.join(root, "missing-executable")], path.join(root, "log"), 1000, state, { send: (...args) => sent.push(args), graceMs: 100 }), /ENOENT/);
    assert.equal(state.child, null); assert.deepEqual(sent, []);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("CPU deadline reaps TERM-resistant descendants even after the root exits", posixOnly, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-cpu-tree-")), state = { cancelled: false };
  try {
    const grandchild = "process.on('SIGTERM',()=>{});setInterval(()=>{},1000)";
    const child = `const c=require('node:child_process').spawn(process.execPath,${JSON.stringify(tinyNode(grandchild).slice(1))},{stdio:'inherit'});require('node:fs').writeFileSync(${JSON.stringify(path.join(root, "pids"))},JSON.stringify([process.pid,c.pid]));process.on('SIGTERM',()=>{});setInterval(()=>{},1000)`;
    const code = `require('node:child_process').spawn(process.execPath,${JSON.stringify(tinyNode(child).slice(1))},{stdio:'inherit'});process.on('SIGTERM',()=>process.exit(0));setInterval(()=>{},1000)`;
    await assert.rejects(() => executeCpuCommand(root, tinyNode(code), path.join(root, "log"), 700, state, { graceMs: 100 }), /managed deadline\/cancellation/);
    for (const pid of JSON.parse(readFileSync(path.join(root, "pids")))) assert.ok(gone(pid), `owned descendant ${pid} survived`);
    assert.equal(state.child, null); assert.equal(state.unsafeToRestore, false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("CPU root close still cleans an observed live grandchild", posixOnly, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-cpu-orphan-")), state = { cancelled: false };
  try {
    const child = "process.on('SIGTERM',()=>{});setInterval(()=>{},1000)";
    const code = `const c=require('node:child_process').spawn(process.execPath,${JSON.stringify(tinyNode(child).slice(1))},{stdio:'ignore'});require('node:fs').writeFileSync(${JSON.stringify(path.join(root, "pid"))},String(c.pid));setTimeout(()=>process.exit(0),500)`;
    await assert.rejects(() => executeCpuCommand(root, tinyNode(code), path.join(root, "log"), 2000, state, { graceMs: 100 }), /descendants outlived command/);
    assert.ok(gone(Number(readFileSync(path.join(root, "pid"), "utf8")))); assert.equal(state.child, null);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("CPU signal EPERM stays explicit and fail-closed without replacing the deadline", posixOnly, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-cpu-eperm-")), state = { cancelled: false };
  try {
    await assert.rejects(() => executeCpuCommand(root, tinyNode("process.on('SIGTERM',()=>{});setInterval(()=>{},1000)"), path.join(root, "log"), 300, state, {
      graceMs: 100, send: (pid, signal) => { if (signal === "SIGTERM") throw Object.assign(new Error("injected permission failure"), { code: "EPERM" }); process.kill(pid, signal); },
    }), /managed deadline\/cancellation.*cleanup:.*EPERM/);
    assert.equal(state.child, null); assert.equal(state.unsafeToRestore, false); assert.ok(state.cleanup.errors.some(error => error.includes("EPERM")));
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("owned CPU identity rejects reused and foreign processes and tolerates known ESRCH races", posixOnly, async () => {
  const uid = process.getuid(), pid = 910001;
  const row = (id, ppid, birth = "first") => ({ pid: id, ppid, pgid: pid, uid, birth, command: `fixture-${id}`, status: "S" });
  const rows = new Map([[pid, row(pid, process.pid)], [pid + 1, row(pid + 1, pid)]]), sent = [];
  const tree = ownedCpuTree(pid, { sample: () => new Map(rows), graceMs: 1, send: (id, signal) => { sent.push([id, signal]); rows.delete(id); } });
  assert.equal(tree.observe().live.length, 2);
  rows.set(pid, { ...rows.get(pid), command: "exec changed the executable, not process identity" });
  assert.equal(tree.observe().live.length, 2);
  rows.set(pid + 1, row(pid + 1, 1, "reused"));
  rows.set(pid + 2, { ...row(pid + 2, 1), uid: uid + 1 });
  const cleanup = await tree.cleanup();
  assert.ok(sent.every(([id]) => id === pid)); assert.deepEqual(cleanup.unproven, [pid + 1, pid + 2]);
  const raceRows = new Map([[pid, row(pid, process.pid)]]);
  const race = ownedCpuTree(pid, { sample: () => new Map(raceRows), graceMs: 1, send: id => { raceRows.delete(id); throw Object.assign(new Error("already gone"), { code: "ESRCH" }); } });
  assert.deepEqual(await race.cleanup(), { errors: [], remaining: [], unproven: [] });
});

if (!process.env.MOSS_ASSERTIONS_MUTATION_CHILD) test("named driver-policy bypasses are killed by original tests without Cargo", async t => {
  const source = readFileSync(path.join(ROOT, "scripts/moss-consumer-assertions.mjs"), "utf8");
  for (const [id, target, old, replacement] of [
    ["DRV-child-recipe", "CPU child inherits the selected incremental recipe and logs only its whitelist", '["CARGO_INCREMENTAL", "CARGO_BUILD_JOBS"', '["CARGO_BUILD_JOBS"'],
    ["DRV-initial-verbose", "synthetic executor proves compile-once, exact positive counts, mutation red, touch restore and final green", '["cargo", "test", "--locked", "--verbose", "-p", "sceneworks-core"', '["cargo", "test", "--locked", "-p", "sceneworks-core"'],
    ["DRV-restore-before-reap", "unreaped compiler fails closed without racing source restoration", "} finally { if (!state.unsafeToRestore) restore();", "} finally { restore();"],
    ["DRV-restore-unchanged", "restore rewrites and touches only changed files while auditing every mutable source", "if (!current?.equals(originals[file]))", "if (true)"],
    ["DRV-restore-touch", "restore rewrites and touches only changed files while auditing every mutable source", "utimesSync(path.join(root, file), new Date(), new Date());", "utimesSync(path.join(root, file), new Date(1000), new Date(1000));"],
    ["DRV-source-hash", "receipt validates case IDs, exact selected test counts, implementation hashes and complete restoration", "JSON.stringify(result.source_before) === JSON.stringify(result.source_after)", "true"],
    ["DRV-owned-identity", "owned CPU identity rejects reused and foreign processes and tolerates known ESRCH races", "row && row.uid === identity.uid && row.birth === identity.birth", "row"],
    ["DRV-signal-EPERM", "CPU signal EPERM stays explicit and fail-closed without replacing the deadline", 'if (error.code !== "ESRCH") errors.push', 'if (false) errors.push'],
    ["DRV-positive-count", "positive and assertion-red parser rejects compilation, crash, no selection and wrong named test", "record.passed === 1", "record.passed >= 0"],
    ["DRV-restore", "synthetic executor proves compile-once, exact positive counts, mutation red, touch restore and final green", "writeFileSync(path.join(root, file), originals[file]);", "void originals[file];"],
    ["DRV-no-stale-carry", "receipt validates case IDs, exact selected test counts, implementation hashes and complete restoration", 'requireFact(!Object.hasOwn(result, "carried"), "stale carried evidence is not accepted; every registry case must execute");', ""],
    ["DRV-case-binding", "receipt validates case IDs, exact selected test counts, implementation hashes and complete restoration", "row.id === spec.cases[index].id", "true"],
  ]) await t.test(id, () => {
    const scratch = mkdtempSync(path.join(os.tmpdir(), "moss-driver-policy-mutation-"));
    try {
      assert.equal(source.split(old).length - 1, 1, id);
      const mutated = path.join(scratch, "driver.mjs"); writeFileSync(mutated, source.replace(old, replacement)); utimesSync(mutated, new Date(), new Date());
      const env = { ...process.env, MOSS_ASSERTIONS_MUTATION_CHILD: "1", MOSS_ASSERTIONS_UNDER_TEST: mutated }; delete env.NODE_TEST_CONTEXT;
      const result = spawnSync(process.execPath, ["--test", `--test-name-pattern=^${target}$`, fileURLToPath(import.meta.url)], { cwd: ROOT, env, encoding: "utf8", timeout: 10000 });
      assert.notEqual(result.status, 0, `${id}\n${result.stdout}\n${result.stderr}`);
      assert.ok(result.stdout.includes(`✖ ${target}`), result.stdout); assert.match(result.stdout, /(?:ℹ|#) fail 1/); assert.doesNotMatch(result.stderr, /SyntaxError|ERR_MODULE_NOT_FOUND/);
      writeFileSync(mutated, source); utimesSync(mutated, new Date(), new Date());
    } finally { rmSync(scratch, { recursive: true, force: true }); }
  });
});
