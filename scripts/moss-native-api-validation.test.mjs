import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  BOUNDS, checkDiscovery, ENVIRONMENT_WHITELIST, EXPECTED_TESTS, ignoredSourceFacts,
  MOSS_PREFIX, ownedWindowsTree, parseArgs, parseTestList, planAuthenticatedCleanup,
  requireManualContext, RSS_CAP_GB, runValidation, startRssGuard,
  statedRssCapBytes, sumOwnedRss, sumWindowsOwnedRss,
} from "./moss-native-api-validation.mjs";
import { TESTS as CLOSED_SELECTOR_TESTS, digest as sha256 } from "./moss-consumer-assertions.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHA = "a".repeat(40);
const EXE = "/vault/moss/sceneworks_rust_api-fixture";
const RUNNER_SOURCE = readFileSync(path.join(ROOT, "scripts/moss-native-api-validation.mjs"), "utf8");

// ---- workflow-shape helpers (semantic evaluation, mirroring platform-review-contracts) ----

function jobBlock(workflow, name) {
  const marker = `  ${name}:\n`;
  const at = workflow.indexOf(marker);
  assert.ok(at >= 0, `workflow must keep a ${name} job`);
  const remainder = workflow.slice(at + marker.length);
  const next = remainder.search(/^  [A-Za-z0-9_-]+:\s*$/m);
  return workflow.slice(at, next === -1 ? undefined : at + marker.length + next);
}
const jobIf = (block) => {
  const match = block.match(/^ {4}if: \$\{\{ (.+) \}\}$/m);
  assert.ok(match, "job must expose one evaluable if");
  return match[1];
};
const evaluate = (expression, event, inputs = {}) =>
  Boolean(Function("github", "inputs", "needs", "steps", "always", "cancelled", `return (${expression});`)(
    { event_name: event, repository: "SceneWorks/SceneWorks", ref: "refs/heads/main", event: { pull_request: { base: { ref: "main" }, head: { repo: { full_name: "SceneWorks/SceneWorks" } } } } },
    inputs, { changes: { result: "success", outputs: { relevant: "true" } } }, {}, () => true, () => false,
  ));

const MLX_JOBS = ["changes", "macos-checks", "browser-fixture-artifact", "nax-worker", "manual-verification", "moss-native-api-mlx"];
const CANDLE_JOBS = ["candle-worker", "imported-nvfp4-worker-smoke", "yue2-terminal-cuda", "iris-worker-smoke", "moss-native-api-candle"];
const MLX_SELECTORS = { verification_profile: ["legacy", "facts-and-projections", "consumer-assertions", "browser-fixture-artifact"], run_memory_calibration: true, run_five_rung_reference: true, provision_qwen_snapshot: true, provision_z_image_snapshot: true };
const CANDLE_SELECTORS = { capability_facts_only: true, run_five_rung_reference: true, provision_snapshot: true, run_ltx_eros_acceptance: true, run_epic_20738_terminal_cuda: true, run_yue2_terminal_cuda: true, run_iris_worker_smoke: true };
const Moss = (value) => ({ run_moss_native_validation: value });

test("a moss dispatch runs exactly its own job in each registered workflow and no other", async () => {
  for (const [file, jobs, selectors, mossJob] of [
    [".github/workflows/macos-mlx.yml", MLX_JOBS, MLX_SELECTORS, "moss-native-api-mlx"],
    [".github/workflows/windows-candle.yml", CANDLE_JOBS, CANDLE_SELECTORS, "moss-native-api-candle"],
  ]) {
    const workflow = readFileSync(path.join(ROOT, file), "utf8");
    const expressions = Object.fromEntries(jobs.map((job) => [job, jobIf(jobBlock(workflow, job))]));
    // Pure moss shape: only the dedicated validation job runs.
    const profiles = file.includes("macos-mlx") ? ["legacy", "facts-and-projections", "consumer-assertions", "browser-fixture-artifact", "unknown"] : [undefined];
    for (const [job, expression] of Object.entries(expressions)) {
      for (const profile of profiles) {
        const inputs = { ...Moss(true), ...(file.includes("macos-mlx") ? { verification_profile: profile } : {}) };
        const expected = job === mossJob;
        assert.equal(evaluate(expression, "workflow_dispatch", inputs), expected,
          `${file} ${job} must run only for the pure moss dispatch`);
      }
    }
    // Mixed shapes start ONLY the MOSS job, whose early input guard fails
    // actionably before compiler setup; every unrelated job stays off.
    const mixed = [];
    for (const [selector, value] of Object.entries(selectors)) {
      // `legacy` with nothing else set IS the pure shape, so it is not a mixed selector.
      for (const item of Array.isArray(value) ? value.filter((option) => option !== "legacy") : [value]) {
        mixed.push({ ...Moss(true), [selector]: item, ...(selector === "verification_profile" ? {} : { verification_profile: "legacy" }) });
      }
    }
    mixed.push({ ...Moss(true), ...Object.fromEntries(Object.entries(selectors).flatMap(([key, value]) => [[key, Array.isArray(value) ? "consumer-assertions" : true]])) });
    for (const inputs of mixed) {
      for (const [job, expression] of Object.entries(expressions))
        assert.equal(evaluate(expression, "workflow_dispatch", inputs), job === mossJob, `${file} mixed dispatch must reach only the early-failing MOSS job: ${job} ${JSON.stringify(inputs)}`);
    }
    // Ordinary triggers never reach the validation job.
    for (const event of ["push", "pull_request", "merge_group", "schedule"]) {
      assert.equal(evaluate(expressions[mossJob], event, Moss(true)), false, `${file} ${mossJob} must be dispatch-only`);
    }
  }
});

test("mixed inputs fail the actual early guards before any compiler setup", () => {
  for (const [file, name, selectors, compiler] of [
    [".github/workflows/macos-mlx.yml", "moss-native-api-mlx", MLX_SELECTORS, "- uses: dtolnay/rust-toolchain@"],
    [".github/workflows/windows-candle.yml", "moss-native-api-candle", CANDLE_SELECTORS, "- uses: ./.github/actions/prepare-rust-runner"],
  ]) {
    const block = jobBlock(readFileSync(path.join(ROOT, file), "utf8"), name);
    const start = block.indexOf("- name: Refuse conflicting MOSS validation inputs before compiler setup");
    assert.ok(start >= 0 && start < block.indexOf(compiler));
    const guard = block.slice(start, block.indexOf(compiler));
    const variables = Object.fromEntries([...guard.matchAll(/^\s+([A-Z_0-9]+): \$\{\{ inputs\.([a-z_0-9]+) \}\}/gm)].map((match) => [match[1], match[2]]));
    const conditions = name.endsWith("mlx")
      ? [...guard.matchAll(/\[ "\$([A-Z_0-9]+)" (!=|=) (legacy|true) \]/g)].map((match) => ({ variable: match[1], unequal: match[2] === "!=", value: match[3] }))
      : [...guard.matchAll(/\$env:([A-Z_0-9]+) -eq 'true'/g)].map((match) => ({ variable: match[1], unequal: false, value: "true" }));
    assert.deepEqual(conditions.map((condition) => variables[condition.variable]).sort(), Object.keys(selectors).sort());
    const refuses = (inputs) => conditions.some(({ variable, unequal, value }) => {
      const actual = String(inputs[variables[variable]] ?? (variables[variable] === "verification_profile" ? "legacy" : false));
      return unequal ? actual !== value : actual === value;
    });
    assert.equal(refuses({ verification_profile: "legacy" }), false);
    for (const [key, values] of Object.entries(selectors)) {
      for (const value of Array.isArray(values) ? values.filter((entry) => entry !== "legacy") : [values]) assert.equal(refuses({ [key]: value }), true, `${file}: ${key}=${value}`);
    }
    assert.match(guard, /exit 1|throw 'MOSS native validation must be dispatched alone/);
  }
});

test("every pre-existing dispatch shape keeps its exact behaviour when the moss flag is false", async () => {
  const mlx = readFileSync(path.join(ROOT, ".github/workflows/macos-mlx.yml"), "utf8");
  const candle = readFileSync(path.join(ROOT, ".github/workflows/windows-candle.yml"), "utf8");
  const mlxIf = (job) => jobIf(jobBlock(mlx, job));
  const candleIf = (job) => jobIf(jobBlock(candle, job));
  for (const [expression, event, inputs, expected] of [
    [mlxIf("changes"), "workflow_dispatch", { ...Moss(false), verification_profile: "legacy" }, true],
    [mlxIf("macos-checks"), "workflow_dispatch", { ...Moss(false), verification_profile: "legacy" }, true],
    [mlxIf("nax-worker"), "workflow_dispatch", { ...Moss(false), verification_profile: "legacy" }, true],
    [mlxIf("browser-fixture-artifact"), "workflow_dispatch", { ...Moss(false), verification_profile: "browser-fixture-artifact", run_memory_calibration: false, run_five_rung_reference: false, provision_qwen_snapshot: false, provision_z_image_snapshot: false }, true],
    [mlxIf("manual-verification"), "workflow_dispatch", { ...Moss(false), verification_profile: "consumer-assertions" }, true],
    [mlxIf("manual-verification"), "workflow_dispatch", { ...Moss(true), verification_profile: "consumer-assertions" }, false],
    [mlxIf("browser-fixture-artifact"), "workflow_dispatch", { ...Moss(true), verification_profile: "browser-fixture-artifact", run_memory_calibration: false, run_five_rung_reference: false, provision_qwen_snapshot: false, provision_z_image_snapshot: false }, false],
    [candleIf("candle-worker"), "workflow_dispatch", { ...Moss(false), capability_facts_only: true }, true],
    [candleIf("candle-worker"), "workflow_dispatch", { ...Moss(true), capability_facts_only: true }, false],
    [candleIf("yue2-terminal-cuda"), "workflow_dispatch", { ...Moss(false), run_yue2_terminal_cuda: true, capability_facts_only: false }, true],
    [candleIf("yue2-terminal-cuda"), "workflow_dispatch", { ...Moss(true), run_yue2_terminal_cuda: true, capability_facts_only: false }, false],
    [candleIf("iris-worker-smoke"), "workflow_dispatch", { ...Moss(false), run_iris_worker_smoke: true, capability_facts_only: false }, true],
    [candleIf("iris-worker-smoke"), "workflow_dispatch", { ...Moss(true), run_iris_worker_smoke: true, capability_facts_only: false }, false],
    [mlxIf("macos-checks"), "push", Moss(false), true],
    [candleIf("candle-worker"), "pull_request", Moss(false), true],
  ]) assert.equal(evaluate(expression, event, inputs), expected, `${event} ${JSON.stringify(inputs)}`);
});

test("the four native tests stay ignored in source and unreachable from ordinary CI", async () => {
  const source = readFileSync(path.join(ROOT, "apps/rust-api/src/tests/moss_training_native.rs"), "utf8");
  assert.deepEqual(ignoredSourceFacts(source).ignored.sort(), [...EXPECTED_TESTS].sort());
  assert.match(readFileSync(path.join(ROOT, "apps/rust-api/src/tests/mod.rs"), "utf8"), /mod moss_training_native;/);
  assert.equal(Object.values(CLOSED_SELECTOR_TESTS).some(([_pkg, fqn]) => fqn.includes("moss_training_native")), false,
    "the closed consumer-assertion registry must not execute any native test in ordinary CI");
  const shortNames = EXPECTED_TESTS.map((fqn) => fqn.slice(MOSS_PREFIX.length));
  const workflowFiles = readdirSync(path.join(ROOT, ".github/workflows")).filter((name) => /\.ya?ml$/.test(name));
  for (const name of workflowFiles) {
    const body = readFileSync(path.join(ROOT, ".github/workflows", name), "utf8");
    for (const shortName of shortNames) assert.ok(!body.includes(shortName), `${name} must not name ${shortName}`);
    assert.ok(!body.includes("--include-ignored"), `${name} must never run ignored tests wholesale`);
  }
  for (const name of readdirSync(path.join(ROOT, "scripts")).filter((name) => name.endsWith(".mjs"))) {
    if (name === "moss-native-api-validation.mjs" || name === "moss-native-api-validation.test.mjs") continue;
    const body = readFileSync(path.join(ROOT, "scripts", name), "utf8");
    for (const shortName of shortNames) assert.ok(!body.includes(shortName), `scripts/${name} must not name ${shortName}`);
  }
  // The runner's execution surface is exactly: --ignored --exact per case, and
  // Windows cleanup is per authenticated pid — never a bare /T tree.
  for (const fragment of ['test, "--ignored", "--exact", "--nocapture", "--test-threads=1"', '"/F", "/PID"']) assert.ok(RUNNER_SOURCE.includes(fragment), fragment);
  assert.ok(!RUNNER_SOURCE.includes("--include-ignored"));
  assert.ok(!RUNNER_SOURCE.includes('"/T"'));
  // The contract bounds are fixed, not knobbed: both jobs restate the existing
  // 2 GiB aggregate owned-RSS cap, no dispatch input can widen it, and the guard
  // is required on BOTH darwin and win32 — never Mac-only.
  for (const file of [".github/workflows/macos-mlx.yml", ".github/workflows/windows-candle.yml"]) {
    const body = readFileSync(path.join(ROOT, file), "utf8");
    const block = jobBlock(body, file.includes("macos-mlx") ? "moss-native-api-mlx" : "moss-native-api-candle");
    assert.match(block, /MOSS_NATIVE_RSS_CAP_GB: "2"/);
    assert.match(block, /name: Upload MOSS native validation evidence\n\s+if: \$\{\{ always\(\) \}\}/);
    assert.ok(!body.includes("moss_native_rss_cap_gb"), `${file} must not expose a cap dispatch input`);
  }
  assert.equal(RSS_CAP_GB, 2);
  assert.ok(RUNNER_SOURCE.includes('if (platform === "darwin" || platform === "win32") {\n        const stated = statedRssCapBytes(env);'), "the cap must be required on both darwin and win32");
  assert.ok(RUNNER_SOURCE.includes("const defaultGuardSample = (state) =>\n  process.platform === \"win32\" ? sumWindowsOwnedRss(queryWindowsProcesses(), state.identity)"), "the guard must sample aggregate owned RSS on Windows too");
});

test("the source guard rejects any additional test, including an unrelated name", () => {
  const source = readFileSync(path.join(ROOT, "apps/rust-api/src/tests/moss_training_native.rs"), "utf8");
  assert.throws(() => ignoredSourceFacts(`${source}\n#[test]\nfn unexpected_model_execution() {}\n`), /exactly the four/);
  assert.throws(() => ignoredSourceFacts(`${source}\n#[tokio::test]\n#[ignore]\nasync fn unexpected_model_execution() {}\n`), /exactly the four/);
  assert.throws(() => ignoredSourceFacts(source.replace(/#\[ignore = [^\n]+\]\n/, "")), /missing #\[ignore\]/);
});

test("discovery requires exactly the four expected, fully ignored tests", () => {
  const list = [...EXPECTED_TESTS, "tests::browser_fixture::serve_owned_manual_browser_fixture", "tests::other::unit"].map((name) => `${name}: test`).join("\n");
  const ignored = [...EXPECTED_TESTS, "tests::browser_fixture::serve_owned_manual_browser_fixture"].map((name) => `${name}: test`).join("\n");
  const parse = (text) => parseTestList(`${text}\n`);
  const extraTest = "tests::moss_training_native::moss_real_dataset_extra_round_trip_cpu";
  assert.deepEqual(checkDiscovery(parse(list), parse(ignored)).discovered, [...EXPECTED_TESTS].sort());
  assert.throws(() => checkDiscovery(parse(list.replace(`${EXPECTED_TESTS[0]}: test`, `${extraTest}: test`)), parse(ignored)), /discovery mismatch/);
  assert.throws(() => checkDiscovery(parse(`${list}\n${extraTest}: test`), parse(ignored)), /discovery mismatch/);
  assert.throws(() => checkDiscovery(parse(list), parse(ignored.replace(`${EXPECTED_TESTS[1]}: test`, ""))), /ignored mismatch/);
  assert.throws(() => checkDiscovery(parse(list), parse(`${ignored}\n${extraTest}: test`)), /ignored mismatch/);
  assert.deepEqual(parseTestList("tests::a::b: test\n\ntests::c::d: benchmark\n"), [{ name: "tests::a::b", kind: "test" }, { name: "tests::c::d", kind: "benchmark" }]);
});

// ---- mocked end-to-end driver ----

const GREEN = (name) => `running 1 test\ntest ${name} ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s\n`;
const ARTIFACT = `${JSON.stringify({ reason: "compiler-artifact", executable: EXE, target: { name: "sceneworks_rust_api", kind: ["lib"] } })}\n`;
const LIST = `${[...EXPECTED_TESTS, "tests::browser_fixture::serve_owned_manual_browser_fixture", "tests::other::unit"].map((name) => `${name}: test`).join("\n")}\n`;
const IGNORED_LIST = `${[...EXPECTED_TESTS, "tests::browser_fixture::serve_owned_manual_browser_fixture"].map((name) => `${name}: test`).join("\n")}\n`;

function fixtureRoot() {
  const root = mkdtempSync(path.join(os.tmpdir(), "moss-native-source-"));
  for (const [file, body] of [
    ["Cargo.lock", `[[package]]\nname = "inference"\nsource = "git+https://github.com/SceneWorks/inference?rev=${SHA}#${SHA}"\n`],
    ["rust-toolchain.toml", 'channel = "1.97.1"\n'],
    [".github/workflows/macos-mlx.yml", "# fixture workflow source\n"],
    ["apps/rust-api/src/tests/moss_training_native.rs",
      [...EXPECTED_TESTS].map((fqn) => `#[tokio::test]\n#[ignore = "full native model execution; explicit moss-native-api manual validation only"]\nasync fn ${fqn.slice(MOSS_PREFIX.length)}() { let value = 1; }\n`).join("\n")],
  ]) { mkdirSync(path.dirname(path.join(root, file)), { recursive: true }); writeFileSync(path.join(root, file), body); }
  return root;
}

function mockDriver({ compile = { output: ARTIFACT, exit: 0 }, cases = {}, lists = [{ output: LIST, exit: 0 }, { output: IGNORED_LIST, exit: 0 }] } = {}) {
  const calls = [];
  const execute = async (root, args, log, limitMs, state) => {
    calls.push(args);
    const emit = (output, exit = 0) => { writeFileSync(log, output); return { exit, output }; };
    if (args[0] === "cargo") return emit(compile.output, compile.exit ?? 0);
    if (args.includes("--ignored") && args.includes("--list")) return emit(lists[1].output, lists[1].exit ?? 0);
    if (args.includes("--list")) return emit(lists[0].output, lists[0].exit ?? 0);
    const name = args[1];
    const behavior = cases[name] ?? {};
    if (behavior.signal) behavior.signal();
    if (behavior.unsafe) state.unsafe = true;
    if (behavior.throw) { writeFileSync(log, behavior.output ?? ""); throw new Error(behavior.throw); }
    return emit(behavior.output ?? GREEN(name), behavior.exit ?? 0);
  };
  return { calls, execute };
}

async function pipeline(overrides = {}) {
  const root = fixtureRoot();
  const output = mkdtempSync(path.join(os.tmpdir(), "moss-native-output-"));
  const driver = overrides.driver ?? mockDriver(overrides.driverOptions ?? {});
  const result = await runValidation({
    root, output, backend: overrides.backend ?? "mlx",
    env: { GITHUB_EVENT_NAME: "workflow_dispatch", GITHUB_SHA: SHA, GITHUB_TOKEN: "secret-token", AZURE_SIGNING_SECRET: "secret", MOSS_NATIVE_RSS_CAP_GB: "2", MOSS_NATIVE_WORKFLOW_FILE: ".github/workflows/macos-mlx.yml", ...overrides.env },
    inspect: () => ({ head: SHA, clean: true }),
    statExecutable: () => true,
    toolchain: "rustc 1.97.1",
    rssGuard: (state) => { state.rssFirstBytes = 1024; state.rssPeakBytes = 2048; return () => undefined; },
    installSignals: () => () => undefined,
    execute: driver.execute,
    ...overrides.options,
  });
  return { result, receipt: JSON.parse(readFileSync(path.join(output, "receipt.json"), "utf8")), calls: driver.calls, output, root };
}

test("green test text without a positive owned-RSS sample cannot pass", async () => {
  const { result, receipt } = await pipeline({ options: { rssGuard: () => () => undefined } });
  assert.equal(result.ok, false);
  assert.equal(receipt.cases.length, 4);
  assert.ok(receipt.cases.every((entry) => !entry.ok && /owned-RSS observation/.test(entry.error)));
});

test("a green validation records exact evidence and passes", async () => {
  const { result, receipt, calls, root } = await pipeline();
  assert.equal(result.ok, true);
  assert.equal(receipt.status, "passed");
  assert.equal(receipt.source.head, SHA);
  assert.deepEqual(receipt.source.ignoredTests.sort(), [...EXPECTED_TESTS].sort());
  assert.equal(receipt.source.inferencePin, SHA);
  assert.equal(receipt.source.workflowFile, ".github/workflows/macos-mlx.yml");
  assert.equal(receipt.source.workflowSha256, sha256(readFileSync(path.join(root, ".github/workflows/macos-mlx.yml"))));
  assert.equal(receipt.compile.executable, EXE);
  assert.equal(receipt.lists.discovered.length, 4);
  assert.deepEqual(receipt.cases.map((entry) => entry.ok), [true, true, true, true]);
  for (const entry of receipt.cases) {
    assert.equal(entry.counts.passed, 1); assert.equal(entry.counts.failed, 0); assert.equal(entry.counts.ignored, 0); assert.equal(entry.exitCode, 0);
    assert.deepEqual(entry.command.slice(1, 6), [entry.test, "--ignored", "--exact", "--nocapture", "--test-threads=1"]);
    assert.ok(Number.isFinite(entry.durationMs));
    assert.equal(entry.cleanup, null);
  }
  assert.equal(calls.length, 7); // compile + two lists + four independent cases
  assert.equal(receipt.environment.CUDA_VISIBLE_DEVICES, "-1");
  assert.equal(ENVIRONMENT_WHITELIST.includes("GITHUB_TOKEN"), false);
  assert.ok(!JSON.stringify(receipt).includes("secret-token"));
  assert.ok(!JSON.stringify(receipt).includes("AZURE_SIGNING_SECRET"));
  assert.equal(receipt.environment.MOSS_NATIVE_RSS_CAP_GB, "2");
  assert.equal(BOUNDS.caseMs, 180_000, "the fixed 180s external per-case bound");
  assert.equal(receipt.supervisor.caseBoundMs, BOUNDS.caseMs);
  assert.equal(receipt.supervisor.rssGuardBytes, 2 * 1024 ** 3);
});

test("compile failure and duplicate executables fail before any case runs, receipt first", async () => {
  const compile = await pipeline({ driverOptions: { compile: { output: "error[E0308]: fixture\n", exit: 101 } } });
  assert.equal(compile.result.ok, false);
  assert.equal(compile.calls.length, 1);
  assert.match(compile.receipt.compile.error, /exited 101, not 0/);
  assert.match(compile.receipt.compile.rawOutput, /error\[E0308\]/);
  assert.deepEqual(compile.receipt.cases, []);
  const duplicate = await pipeline({ driverOptions: { compile: { output: ARTIFACT + ARTIFACT, exit: 0 } } });
  assert.equal(duplicate.result.ok, false);
  assert.equal(duplicate.calls.length, 1);
  assert.match(duplicate.receipt.failures.join(" "), /got .*sceneworks_rust_api/);
  assert.ok(readFileSync(path.join(duplicate.output, "receipt.json"), "utf8").length > 0, "receipt is written before the failure exit");
});

test("a nonzero compile or list exit is a failure even when artifact or list text is valid", async () => {
  const compile = await pipeline({ driverOptions: { compile: { output: ARTIFACT, exit: 101 } } });
  assert.equal(compile.result.ok, false);
  assert.equal(compile.calls.length, 1);
  assert.match(compile.receipt.compile.error, /exited 101, not 0/);
  assert.deepEqual(compile.receipt.cases, []);
  const listed = await pipeline({ driverOptions: { lists: [{ output: LIST, exit: 0 }, { output: IGNORED_LIST, exit: 3 }] } });
  assert.equal(listed.result.ok, false);
  assert.equal(listed.calls.length, 3);
  assert.match(listed.receipt.lists.error, /--ignored --list exited 3, not 0/);
  assert.deepEqual(listed.receipt.cases, []);
});

test("an unproven cleanup blocks the remaining cases instead of attempting them", async () => {
  const { result, receipt, calls } = await pipeline({ driverOptions: { cases: { [EXPECTED_TESTS[0]]: { throw: "owned child cleanup failed; unproven cleanup", unsafe: true } } } });
  assert.equal(result.ok, false);
  assert.equal(calls.length, 4, "compile + two lists + exactly the one attempted case");
  assert.equal(receipt.cases.length, 4);
  assert.equal(receipt.cases[0].ok, false);
  for (const entry of receipt.cases.slice(1)) {
    assert.equal(entry.attempted, false);
    assert.equal(entry.blocked, true);
    assert.equal(entry.ok, false, "a blocked case is never a pass");
    assert.match(entry.reason, /refusing to start further cases/);
  }
});

test("signals propagate to the active case, halt further attempts, and are recorded", async () => {
  const handlers = [];
  const { result, receipt, calls } = await pipeline({
    options: { installSignals: (handler) => { handlers.push(handler); return () => {}; } },
    driverOptions: { cases: { [EXPECTED_TESTS[0]]: { throw: "managed deadline/cancellation", signal: () => handlers[0]?.() } } },
  });
  assert.equal(handlers.length, 1);
  assert.equal(result.ok, false);
  assert.equal(calls.length, 4);
  assert.equal(receipt.cases[0].ok, false);
  assert.equal(receipt.cases[0].cancelled, true);
  assert.equal(receipt.cases[0].cancellation.source, "signal");
  assert.deepEqual(receipt.cases.slice(1).map((entry) => entry.blocked), [true, true, true]);
  assert.deepEqual(receipt.cases.slice(1).every((entry) => entry.attempted === false), true);
  assert.equal(receipt.cancellation.requested, true);
  assert.equal(receipt.cancellation.source, "signal");
});

test("each case executes independently and degenerate results are never credited", async () => {
  const [first, second, third] = EXPECTED_TESTS;
  const { result, receipt, calls } = await pipeline({ driverOptions: { cases: {
    [first]: { throw: "managed deadline/cancellation", output: "partial fixture output\n" },
    [second]: { exit: 0, output: `test ${second} ... ignored\n\ntest result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 3 filtered out; finished in 0.01s\n` },
    [third]: { exit: 0, output: `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out;\ntest ${third} ... ok\n` },
  } } });
  assert.equal(result.ok, false);
  assert.equal(receipt.status, "failed");
  assert.equal(calls.length, 7, "one failing case never suppresses the other three");
  assert.equal(receipt.cases.length, 4);
  assert.match(receipt.cases[0].error, /managed deadline/);
  assert.match(receipt.cases[0].rawOutput, /partial fixture output/);
  assert.equal(receipt.cases[0].ok, false);
  assert.equal(receipt.cases[1].ok, false);
  assert.match(receipt.cases[1].error, /restore\/baseline must execute exactly one positive test|selected exact test marker/);
  assert.equal(receipt.cases[2].ok, false);
  assert.equal(receipt.cases[3].ok, true);
  assert.equal(receipt.failures.length, 3);
  const zero = await pipeline({ driverOptions: { cases: { [first]: { exit: 0, output: "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 4 filtered out;\n" } } } });
  assert.equal(zero.receipt.cases[0].ok, false);
  const compileText = await pipeline({ driverOptions: { cases: { [first]: { exit: 101, output: "error: could not compile fixture\n" } } } });
  assert.equal(compileText.receipt.cases[0].ok, false);
});

test("the driver refuses non-dispatch contexts, dirty trees, wrong SHAs and unstated RSS caps", async () => {
  const push = await pipeline({ env: { GITHUB_EVENT_NAME: "push" } });
  assert.equal(push.result.ok, false);
  assert.equal(push.calls.length, 0);
  assert.match(push.receipt.failures[0], /requires an explicit workflow_dispatch/);
  const dirty = await pipeline({ options: { inspect: () => ({ head: SHA, clean: false }) } });
  assert.equal(dirty.result.ok, false);
  assert.match(dirty.receipt.failures[0], /working tree is not clean/);
  const wrongSha = await pipeline({ options: { inspect: () => ({ head: "b".repeat(40), clean: true }) } });
  assert.equal(wrongSha.result.ok, false);
  assert.match(wrongSha.receipt.failures[0], /is not the dispatched SHA/);
  const missingCap = await pipeline({ env: { MOSS_NATIVE_RSS_CAP_GB: undefined } });
  assert.equal(missingCap.result.ok, false);
  assert.equal(missingCap.calls.length, 0);
  assert.match(missingCap.receipt.failures[0], /aggregate owned-RSS cap/);
  const widenedCap = await pipeline({ env: { MOSS_NATIVE_RSS_CAP_GB: "4" } });
  assert.equal(widenedCap.result.ok, false);
  assert.equal(widenedCap.calls.length, 0);
  assert.match(widenedCap.receipt.failures[0], /bounded CPU validation contract cap \(2 GiB, an admitted cap\)/);
  assert.equal(statedRssCapBytes({ MOSS_NATIVE_RSS_CAP_GB: "2" }), RSS_CAP_GB * 1024 ** 3);
  assert.throws(() => statedRssCapBytes({ MOSS_NATIVE_RSS_CAP_GB: "4" }), /bounded CPU validation contract cap/);
  assert.throws(() => statedRssCapBytes({ MOSS_NATIVE_RSS_CAP_GB: "0" }), /bounded CPU validation contract cap/);
  assert.throws(() => statedRssCapBytes({ MOSS_NATIVE_RSS_CAP_GB: "not-a-number" }), /bounded CPU validation contract cap/);
  assert.equal(statedRssCapBytes({}), null);
  assert.throws(() => requireManualContext({ GITHUB_EVENT_NAME: "pull_request" }));
});

test("the backend must match the actual runner platform of the real caller", async () => {
  const wrongPlatform = await pipeline({ backend: "candle", options: { platform: "linux" } });
  assert.equal(wrongPlatform.result.ok, false);
  assert.equal(wrongPlatform.calls.length, 0);
  assert.match(wrongPlatform.receipt.failures[0], /backend candle does not match the actual runner platform linux/);
  const windowsCandle = await pipeline({ backend: "candle", options: { platform: "win32" } });
  assert.equal(windowsCandle.result.ok, true);
  const missingWorkflowFile = await pipeline({ env: { MOSS_NATIVE_WORKFLOW_FILE: undefined } });
  assert.equal(missingWorkflowFile.result.ok, false);
  assert.equal(missingWorkflowFile.calls.length, 0);
  assert.match(missingWorkflowFile.receipt.failures[0], /MOSS_NATIVE_WORKFLOW_FILE/);
});

test("parseArgs validates the backend and keeps output outside the checkout", () => {
  const base = ["--root", "/repo", "--output", "/tmp/out", "--backend", "mlx"];
  assert.deepEqual(parseArgs(base).backend, "mlx");
  assert.deepEqual(parseArgs([...base.slice(0, -1), "candle"]).backend, "candle");
  assert.throws(() => parseArgs(["--root", "/repo", "--backend", "mlx"]));
  assert.throws(() => parseArgs(["--root", "/repo", "--output", "/repo/out", "--backend", "mlx"]), /outside the checkout/);
  assert.throws(() => parseArgs(["--root", "/repo", "--output", "/tmp/out", "--backend", "gpu"]), /mlx or candle/);
});

test("the external RSS guard bounds only the owned descendant tree", async () => {
  const rows = new Map([[10, { pid: 10, ppid: 1, rssBytes: 1024 }], [11, { pid: 11, ppid: 10, rssBytes: 2048 }], [12, { pid: 12, ppid: 11, rssBytes: 4096 }], [99, { pid: 99, ppid: 1, rssBytes: 1 << 30 }]]);
  assert.equal(sumOwnedRss(rows, 10), 7168);
  assert.equal(sumOwnedRss(rows, 777), null);
  assert.equal(sumOwnedRss(null, 10), null);
  const child = { pid: 10, exitCode: null };
  const state = { child, cancelled: false };
  const stop = startRssGuard(state, { sample: () => sumOwnedRss(rows, 10), capBytes: 4096, intervalMs: 1 });
  await new Promise((resolve) => setTimeout(resolve, 20));
  stop();
  assert.equal(state.cancelled, true);
  assert.equal(state.rssExceededBytes, 7168);
  assert.equal(state.rssFirstBytes, 7168, "a bound claim is backed by real samples");
  assert.equal(state.rssPeakBytes, 7168);
  // A sampler failure or a missing sample while the case is LIVE fails closed.
  const throwing = { child, cancelled: false };
  const stopThrowing = startRssGuard(throwing, { sample: () => { throw new Error("ps unavailable"); }, capBytes: 1 << 30, intervalMs: 1 });
  await new Promise((resolve) => setTimeout(resolve, 10));
  stopThrowing();
  assert.equal(throwing.cancelled, true);
  assert.match(throwing.rssSamplingError, /sampler threw: ps unavailable/);
  const missing = { child, cancelled: false };
  const stopMissing = startRssGuard(missing, { sample: () => null, capBytes: 1 << 30, intervalMs: 1 });
  await new Promise((resolve) => setTimeout(resolve, 10));
  stopMissing();
  assert.equal(missing.cancelled, true);
  assert.match(missing.rssSamplingError, /returned no sample while the case was live/);
  // First and peak aggregates are recorded, not just the over-cap verdict.
  const observed = { child, cancelled: false };
  const samples = [1, 5, 3];
  const stopObserved = startRssGuard(observed, { sample: () => samples.shift() ?? 3, capBytes: 1 << 30, intervalMs: 1 });
  await new Promise((resolve) => setTimeout(resolve, 10));
  stopObserved();
  assert.equal(observed.cancelled, false);
  assert.equal(observed.rssFirstBytes, 1);
  assert.equal(observed.rssPeakBytes, 5);
  const calm = { child: { pid: 10, exitCode: 0 }, cancelled: false };
  const stopCalm = startRssGuard(calm, { sample: () => { throw new Error("must not sample an exited child"); }, capBytes: 1, intervalMs: 1 });
  await new Promise((resolve) => setTimeout(resolve, 10));
  stopCalm();
  assert.equal(calm.cancelled, false);
  assert.throws(() => startRssGuard({ child, cancelled: false }, { sample: () => 1 }), /aggregate cap/);
});

test("Windows cleanup signals only re-authenticated owned identities", () => {
  const identity = { pid: 100, birthMs: 10_000, path: "c:\\cargo\\cargo.exe" };
  const rows = (extra = []) => new Map([
    [100, { pid: 100, ppid: 1, birthMs: 10_000, path: "c:\\cargo\\cargo.exe" }],
    [101, { pid: 101, ppid: 100, birthMs: 11_000, path: "c:\\rustc\\rustc.exe" }],
    [102, { pid: 102, ppid: 101, birthMs: 12_000, path: "c:\\link\\link.exe" }],
    [201, { pid: 201, ppid: 100, birthMs: 400, path: "c:\\old\\predates.exe" }],
    [99, { pid: 99, ppid: 1, birthMs: 9_900, path: "c:\\stranger\\unrelated.exe" }],
    ...extra,
  ]);
  const owned = ownedWindowsTree(rows(), identity);
  assert.equal(owned.authenticated, true);
  assert.deepEqual(owned.owned.map((entry) => entry.pid).sort(), [100, 101, 102]);
  assert.equal(ownedWindowsTree(rows(), { ...identity, path: "c:\\other\\reused.exe" }).authenticated, false);
  assert.equal(ownedWindowsTree(null, identity).authenticated, false);
  // Cleanup plan: only live descendants whose birth still matches; PID reuse and
  // exited pids are dropped; the root joins only while its full identity holds.
  const known = new Map([[101, 11_000], [102, 12_000], [103, 13_000]]);
  const fresh = rows([{ 103: { pid: 103, ppid: 1, birthMs: 99_000, path: "c:\\reused\\reused.exe" } }]);
  assert.deepEqual(planAuthenticatedCleanup(fresh, identity, known, true).targets, [101, 102, 100]);
  assert.deepEqual(planAuthenticatedCleanup(new Map(), identity, known, true).targets, []);
  assert.deepEqual(planAuthenticatedCleanup(fresh, identity, known, false).targets, [101, 102]);
  assert.deepEqual(planAuthenticatedCleanup(fresh, { ...identity, birthMs: 20_000 }, known, true).targets, [101, 102], "a reused root pid is never a target");
  // Aggregate owned RSS on Windows sums exactly the authenticated tree, so the
  // 2 GiB bound means the same quantity as the POSIX guard.
  const rssRows = new Map([
    [100, { pid: 100, ppid: 1, birthMs: 10_000, path: "c:\\cargo\\cargo.exe", rssBytes: 500 * 1024 * 1024 }],
    [101, { pid: 101, ppid: 100, birthMs: 11_000, path: "c:\\rustc\\rustc.exe", rssBytes: 300 * 1024 * 1024 }],
    [102, { pid: 102, ppid: 101, birthMs: 12_000, path: "c:\\link\\link.exe", rssBytes: 200 * 1024 * 1024 }],
    [99, { pid: 99, ppid: 1, birthMs: 9_900, path: "c:\\stranger\\unrelated.exe", rssBytes: 64 * 1024 ** 3 }],
  ]);
  assert.equal(sumWindowsOwnedRss(rssRows, identity), 1000 * 1024 * 1024);
  assert.equal(sumWindowsOwnedRss(rssRows, { ...identity, path: "c:\\reused\\other.exe" }), null);
  assert.equal(sumWindowsOwnedRss(null, identity), null);
});
