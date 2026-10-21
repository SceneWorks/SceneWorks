import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
const modulePath = process.env.MOSS_ASSERTIONS_UNDER_TEST ?? fileURLToPath(new URL("./moss-consumer-assertions.mjs", import.meta.url));
const { bodySpan, digest, loadRegistry, MUTABLE_FILES, parseTestResult, patchText, REGISTRY_PATH, runAssertions, TESTS, validateAssertionReceipt } = await import(pathToFileURL(modulePath));

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

test("closed committed inventory executes all 69 cases across seven exact tests", () => {
  const spec = loadRegistry(ROOT);
  assert.equal(spec.cases.length, 69); assert.equal(Object.keys(TESTS).length, 7);
  let workerSeen = false;
  for (const item of spec.cases) {
    const worker = item.tests.some(key => TESTS[key][0] === "sceneworks-worker");
    if (worker) workerSeen = true;
    assert.ok(worker || !workerSeen, "core cases must precede worker cases");
  }
  for (const id of ["SW-Base-official-source", "SW-SFT-official-source", "SW-compute-demoted", "SW-thin-blank-sampler", "SW-validate-modules-array", "SW-validate-modules-members", "SW-validate-decomp-integer", "SW-validate-decomp-positive", "SW-validate-decomp-overflow", "SW-validate-valid-decomp"]) assert.ok(spec.cases.some(item => item.id === id), id);
  assert.equal(Object.hasOwn(spec, "carried"), false);
  assert.equal(spec.registry_sha256, digest(readFileSync(path.join(ROOT, REGISTRY_PATH))));
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

if (!process.env.MOSS_ASSERTIONS_MUTATION_CHILD) test("named driver-policy bypasses are killed by original tests without Cargo", async t => {
  const source = readFileSync(path.join(ROOT, "scripts/moss-consumer-assertions.mjs"), "utf8");
  for (const [id, target, old, replacement] of [
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
