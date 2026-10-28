import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  BOUNDS, checkDiscovery, ENVIRONMENT_WHITELIST, executeOwnedChild, EXPECTED_TESTS, ignoredSourceFacts,
  MOSS_PREFIX, ownedWindowsTree, parseArgs, parseTestList, planAuthenticatedCleanup,
  executePreparation, isPreparationCommand, requireManualContext, RSS_CAP_GB, runValidation, startRssGuard,
  statedRssCapBytes, sumOwnedRss, sumWindowsOwnedRss, queryWindowsProcesses, disposeWindowsProcessReader, PRE_MODEL_CPU_TEST,
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
const MLX_SELECTORS = { hosted_only: true, verification_profile: ["legacy", "facts-and-projections", "consumer-assertions", "browser-fixture-artifact"], run_memory_calibration: true, run_five_rung_reference: true, provision_qwen_snapshot: true, provision_z_image_snapshot: true };
const CANDLE_SELECTORS = { capability_facts_only: true, run_five_rung_reference: true, provision_snapshot: true, run_ltx_eros_acceptance: true, run_epic_20738_terminal_cuda: true, run_yue2_terminal_cuda: true, run_iris_worker_smoke: true, run_supra2_worker_smoke: true };
const Moss = (value) => ({ run_moss_native_validation: value });

function assertBrowserConflictRejector(workflow) {
  const block = jobBlock(workflow, "browser-fixture-artifact");
  const guard = block.indexOf("- name: Reject conflicting browser fixture selectors before setup");
  assert.ok(guard >= 0 && guard < block.indexOf("actions/checkout"));
  assert.ok(guard < block.indexOf("dtolnay/rust-toolchain"));
  const step = block.slice(guard).split(/\n      - /)[0];
  assert.match(step, /RUN_MOSS_NATIVE_VALIDATION: \$\{\{ inputs\.run_moss_native_validation \}\}/);
  assert.match(step, /\$\{!selector\}/);
  assert.match(step, /echo "::error::browser-fixture-artifact cannot be combined with \$selector"\n\s+exit 1/);
}

test("a moss dispatch reaches only its guard and an explicit hosted browser conflict rejector", async () => {
  for (const [file, jobs, selectors, mossJob] of [
    [".github/workflows/macos-mlx.yml", MLX_JOBS, MLX_SELECTORS, "moss-native-api-mlx"],
    [".github/workflows/windows-candle.yml", CANDLE_JOBS, CANDLE_SELECTORS, "moss-native-api-candle"],
  ]) {
    const workflow = readFileSync(path.join(ROOT, file), "utf8");
    if (file.includes("macos-mlx")) {
      assertBrowserConflictRejector(workflow);
      assert.throws(() => assertBrowserConflictRejector(workflow.replace("Reject conflicting browser fixture selectors before setup", "removed guard")));
    }
    const expressions = Object.fromEntries(jobs.map((job) => [job, jobIf(jobBlock(workflow, job))]));
    // Pure moss shape: only the dedicated validation job runs.
    const profiles = file.includes("macos-mlx") ? ["legacy", "facts-and-projections", "consumer-assertions", "browser-fixture-artifact", "unknown"] : [undefined];
    for (const [job, expression] of Object.entries(expressions)) {
      for (const profile of profiles) {
        const inputs = { ...Moss(true), ...(file.includes("macos-mlx") ? { verification_profile: profile } : {}) };
        const expected = job === mossJob || (job === "browser-fixture-artifact" && profile === "browser-fixture-artifact");
        assert.equal(evaluate(expression, "workflow_dispatch", inputs), expected,
          `${file} ${job} must be the MOSS guard or an authenticated hosted conflict rejector`);
      }
    }
    // Mixed shapes reach early input rejection, never model work. The browser
    // profile's hosted rejector also starts so conflicts cannot skip green.
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
        assert.equal(evaluate(expression, "workflow_dispatch", inputs),
          job === mossJob || (job === "browser-fixture-artifact" && inputs.verification_profile === "browser-fixture-artifact"),
          `${file} mixed dispatch must reach only early-failing guards: ${job} ${JSON.stringify(inputs)}`);
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

test("valid dispatch shapes stay unchanged while browser conflicts reach explicit early rejection", async () => {
  const mlx = readFileSync(path.join(ROOT, ".github/workflows/macos-mlx.yml"), "utf8");
  assertBrowserConflictRejector(mlx);
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
    [mlxIf("browser-fixture-artifact"), "workflow_dispatch", { ...Moss(true), verification_profile: "browser-fixture-artifact", run_memory_calibration: false, run_five_rung_reference: false, provision_qwen_snapshot: false, provision_z_image_snapshot: false }, true],
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
  const backend = overrides.backend ?? "mlx";
  const result = await runValidation({
    root, output, backend,
    // Default the injected platform to the backend's own supported host, so the
    // mocked suite proves the same contracts on every host it runs on; the
    // mismatch case below overrides it explicitly.
    platform: backend === "candle" ? "win32" : "darwin",
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
  // Pin the platform to the backend's own supported host, so the workflow-file
  // guard is what must fail here regardless of the machine running the suite.
  const missingWorkflowFile = await pipeline({ backend: "mlx", options: { platform: "darwin" }, env: { MOSS_NATIVE_WORKFLOW_FILE: undefined } });
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
  const fresh = rows([[103, { pid: 103, ppid: 1, birthMs: 99_000, path: "c:\\reused\\reused.exe" }]]);
  // 103 must be a REAL live row with a replacement birth identity, otherwise the
  // exclusion below would only prove that an absent pid is skipped.
  assert.equal(fresh.has(103), true);
  assert.equal(fresh.get(103).birthMs, 99_000);
  assert.deepEqual(planAuthenticatedCleanup(fresh, identity, known, true).targets, [101, 102, 100]);
  // Both directions in one sample: a live descendant whose birth still matches is
  // selected, while the live reused pid above is excluded.
  const mixed = rows([
    [103, { pid: 103, ppid: 1, birthMs: 99_000, path: "c:\\reused\\reused.exe" }],
    [104, { pid: 104, ppid: 100, birthMs: 13_000, path: "c:\\cargo\\child.exe" }],
  ]);
  assert.equal(mixed.has(104), true);
  assert.deepEqual(
    planAuthenticatedCleanup(mixed, identity, new Map([...known, [104, 13_000]]), true).targets,
    [101, 102, 104, 100],
    "a live matching-birth descendant is selected while the live reused pid is excluded",
  );
  assert.deepEqual(planAuthenticatedCleanup(new Map(), identity, known, true).targets, []);
  for (const unavailable of [null, undefined, {}]) {
    assert.throws(() => planAuthenticatedCleanup(unavailable, identity, known, true), /cleanup planning process telemetry unavailable/);
  }
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

// Simulated Windows rows and EventEmitter children only: no host queries or kills.
const WINDOWS_ROOT = { pid: 100, ppid: 1, birthMs: 10_000, path: "cargo.exe", rssBytes: 100 };
const WINDOWS_PARENT = { pid: 101, ppid: 100, birthMs: 20_000, path: "rustc.exe", rssBytes: 200 };
const WINDOWS_STALE = { pid: 102, ppid: 101, birthMs: 15_000, path: "unrelated-survivor.exe", rssBytes: 10_000 };
const WINDOWS_CHILD = { pid: 103, ppid: 101, birthMs: 21_000, path: "link.exe", rssBytes: 300 };
const windowsRows = (...rows) => new Map(rows.map((row) => [row.pid, { ...row }]));

function windowsSupervisor({ planningMissing = false, verificationMissing = false, reused = false, output = "" } = {}) {
  const live = windowsRows(WINDOWS_ROOT, WINDOWS_PARENT, WINDOWS_STALE, WINDOWS_CHILD);
  const survivors = windowsRows(WINDOWS_PARENT, WINDOWS_STALE, WINDOWS_CHILD);
  if (reused) survivors.set(103, { ...WINDOWS_CHILD, birthMs: 30_000 });
  const killed = [], ownHandleKills = [];
  let child, queries = 0;
  return {
    killed, ownHandleKills,
    supervision: {
      sampleMs: 0,
      spawnChild: () => {
        child = new EventEmitter();
        Object.assign(child, { pid: 100, stdout: new PassThrough(), stderr: new PassThrough(), kill: (signal) => { ownHandleKills.push(signal); return true; } });
        return child;
      },
      query: () => {
        queries += 1;
        if (queries === 1) return live;
        if (queries === 2) {
          child.stdout.emit("data", Buffer.from(output));
          child.emit("close", 0, null);
          return live;
        }
        if (queries === 3) return planningMissing ? null : survivors;
        assert.equal(queries, 4, "exactly one fresh post-termination sample");
        return verificationMissing ? null : survivors;
      },
      kill: (pid) => {
        killed.push(pid);
        survivors.delete(pid);
        return 0;
      },
    },
  };
}

test("Windows tree authenticates every immediate parent birth, not just the root birth", () => {
  const rows = windowsRows(WINDOWS_ROOT, WINDOWS_PARENT, WINDOWS_STALE, WINDOWS_CHILD,
    { pid: 104, ppid: 102, birthMs: 25_000, path: "unrelated-descendant", rssBytes: 20_000 });
  const tree = ownedWindowsTree(rows, WINDOWS_ROOT);
  assert.equal(tree.authenticated, true);
  assert.deepEqual(tree.owned.map(({ pid }) => pid), [100, 101, 103]);
  assert.equal(sumWindowsOwnedRss(rows, WINDOWS_ROOT), 600, "stale edge and its subtree are excluded from RSS too");
  const known = new Map(tree.owned.filter(({ pid }) => pid !== 100).map(({ pid, birthMs }) => [pid, birthMs]));
  assert.deepEqual(planAuthenticatedCleanup(rows, WINDOWS_ROOT, known, false).targets, [101, 103]);
  // No clock-skew allowance: match the terminal supervisor's strict edge order.
  rows.set(103, { ...WINDOWS_CHILD, birthMs: WINDOWS_PARENT.birthMs - 1 });
  assert.deepEqual(ownedWindowsTree(rows, WINDOWS_ROOT).owned.map(({ pid }) => pid), [100, 101]);
  rows.set(101, { ...WINDOWS_PARENT, birthMs: NaN });
  assert.equal(ownedWindowsTree(rows, WINDOWS_ROOT).authenticated,false,"unknown child birth is not a stale edge");
  assert.equal(ownedWindowsTree(rows, { ...WINDOWS_ROOT, birthMs: 9000 }).authenticated, false);
});

test('Windows generic supervision waits for queued close after a fresh root absence and still requires final telemetry',async t=>{
 for(const missingFinal of [false,true]){
  let child,queries=0,closed=false;const state={};
  const dir=mkdtempSync(path.join(os.tmpdir(),'moss-close-query-'));t.after(()=>rmSync(dir,{recursive:true,force:true}));
  const supervision={sampleMs:0,graceMs:50,spawnChild:()=>{
    child=Object.assign(new EventEmitter(),{pid:100,stdout:new PassThrough(),stderr:new PassThrough(),kill:()=>assert.fail('naturally completed own handle must not be killed')});
    return child;
  },query:()=>{
    queries++;
    if(queries===1)return windowsRows(WINDOWS_ROOT);
    if(queries===2){assert.equal(closed,false);queueMicrotask(()=>{closed=true;child.emit('close',0,null);});return new Map();}
    if(queries===3)return new Map();
    assert.equal(queries,4);return missingFinal?null:new Map();
  },kill:()=>assert.fail('no foreign PID may be signalled')};
  const execution=executeOwnedChild(ROOT,['mock-only'],path.join(dir,'command.log'),3000,state,supervision);
  if(missingFinal){await assert.rejects(execution,/post-termination process telemetry unavailable/);assert.equal(state.cleanup.released,false);assert.equal(state.unsafe,true);}
  else{assert.equal((await execution).exit,0);assert.equal(state.cleanup.released,true);assert.notEqual(state.unsafe,true);}
  assert.equal(queries,4);assert.equal(closed,true);
 }
});

test('Windows queued root close cannot erase unknown facts of a previously known child',async t=>{
 for(const missing of ['birthMs','path','rssBytes']){
  let child,queries=0;const state={},killed=[];
  const dir=mkdtempSync(path.join(os.tmpdir(),'moss-known-close-'));t.after(()=>rmSync(dir,{recursive:true,force:true}));
  const supervision={sampleMs:0,graceMs:50,spawnChild:()=>{
    child=Object.assign(new EventEmitter(),{pid:100,stdout:new PassThrough(),stderr:new PassThrough(),kill:()=>true});return child;
  },query:()=>{
    queries++;
    if(queries<=2)return windowsRows(WINDOWS_ROOT,WINDOWS_PARENT);
    if(queries===3){queueMicrotask(()=>child.emit('close',0,null));return windowsRows({...WINDOWS_PARENT,ppid:1,[missing]:missing==='path'?'':null},WINDOWS_CHILD);}
    assert.ok(queries<=5);return new Map(); // Both disappear before final cleanup.
  },kill:pid=>{killed.push(pid);return 0;}};
  await assert.rejects(executeOwnedChild(ROOT,['mock-only'],path.join(dir,'command.log'),3000,state,supervision),/owned Windows identity\/RSS unavailable/);
  assert.equal(state.unsafe,true);assert.equal(state.cleanup.released,false);assert.deepEqual(killed,[]);assert.equal(queries,5);
 }
});

test('Windows generic supervision keeps unknown reachable identities unsafe after root close',async t=>{
  const mock=windowsSupervisor(),baseQuery=mock.supervision.query;let calls=0;
  mock.supervision.query=()=>{const rows=baseQuery();if(++calls===2)rows.set(101,{...rows.get(101),birthMs:null,path:'',rssBytes:null});return rows;};
  const dir=mkdtempSync(path.join(os.tmpdir(),'moss-unknown-owned-'));t.after(()=>rmSync(dir,{recursive:true,force:true}));
  const state={};
  await assert.rejects(executeOwnedChild(ROOT,['mock-only'],path.join(dir,'command.log'),BOUNDS.caseMs,state,mock.supervision),/owned Windows identity\/RSS unavailable/);
  assert.equal(state.unsafe,true);assert.equal(state.cleanup.released,false);assert.deepEqual(mock.killed,[]);
});

test("Windows supervisor never adopts or kills stale-edge survivors or reused descendant PIDs", async () => {
  for (const reused of [false, true]) {
    const mock = windowsSupervisor({ reused });
    const state = {};
    const log = path.join(mkdtempSync(path.join(os.tmpdir(), "moss-owned-mock-")), "command.log");
    const result = await executeOwnedChild(ROOT, ["mock-only"], log, BOUNDS.caseMs, state, mock.supervision);
    assert.equal(result.exit, 0);
    assert.deepEqual(mock.killed, reused ? [101] : [101, 103]);
    assert.deepEqual(mock.ownHandleKills, []);
    assert.equal(state.cleanup.released, true);
    assert.deepEqual(state.cleanup.unproven, []);
    assert.notEqual(state.unsafe, true);
  }
});

test("preparation allowlist excludes actual native execution and learned Cargo tests", () => {
  for (const command of [["rustc","-Vv"],["cargo","-V"],["cargo","test","--locked","--no-run"],["/compiled/api","--list"],["/compiled/api","--ignored","--list"],
    ["python3","-I","-m","jsonschema","-i","/record",path.join(ROOT,"scripts/ci/moss-training-acceptance/evidence.schema.json")],
    ["python3","-I",path.join(ROOT,"scripts/ci/moss-training-acceptance/verify_evidence.py"),"--root","/peers","--manifest","/manifest","--selection","/selection","--output","/out"]]) assert.equal(isPreparationCommand(ROOT,command),true,command.join(" "));
  for (const command of [["cargo","test","--ignored"],["cargo","run"],["rustc","model.rs"],["/compiled/api",EXPECTED_TESTS[0],"--ignored","--exact"],["/compiled/api","--list",EXPECTED_TESTS[0]],["python3","arbitrary.py"],["python3","-I","-m","jsonschema","-i","/record","/another.schema.json"]]) assert.equal(isPreparationCommand(ROOT,command),false,command.join(" "));
});

test("preparation permits only the exact pre-model CPU entry and fixed unignored flags", () => {
  assert.equal(PRE_MODEL_CPU_TEST, "tests::moss_training_acceptance_contracts::pre_model_library_and_dispatch_contracts");
  const command = [EXE, PRE_MODEL_CPU_TEST, "--exact", "--nocapture", "--test-threads=1"];
  assert.equal(isPreparationCommand(ROOT, command), true);
  for (const mutant of [
    command.map(value => value === PRE_MODEL_CPU_TEST ? `${value}_other` : value),
    command.map(value => value === PRE_MODEL_CPU_TEST ? EXPECTED_TESTS[0] : value),
    command.map(value => value === PRE_MODEL_CPU_TEST ? "tests::moss_training_acceptance_contracts::training_library_resolves_declared_final_file_by_filesystem_identity" : value),
    command.filter(value => value !== "--exact"),
    command.map(value => value === "--exact" ? "--ignored" : value),
    command.map(value => value === "--test-threads=1" ? "--test-threads=2" : value),
    [...command, "--ignored"], [...command, "--include-ignored"], [...command, "--nocapture"],
  ]) assert.equal(isPreparationCommand(ROOT, mutant), false, mutant.join(" "));
});

test("CPU preparation captures real fast zero/nonzero Node exits with root-and-stdio scope only", async t => {
  const root=mkdtempSync(path.join(os.tmpdir(),"moss-cpu-preparation-"));t.after(()=>rmSync(root,{recursive:true,force:true}));
  for(const code of [0,7]) {
    const state={};
    const result=await executePreparation(ROOT,[process.execPath,"-e",`console.log('metadata only');process.exit(${code})`],path.join(root,`${code}.log`),5000,state);
    assert.equal(result.exit,code);assert.match(result.output,/metadata only/);
    assert.deepEqual(state.completion,{scope:"root-and-stdio",completed:true,rootExited:true,stdioClosed:true,exitCode:code,signal:null});
    assert.equal(state.cleanup,undefined,"CPU preparation must never claim whole-tree release");
  }
});

test("CPU preparation spawn failure aborts without a completion claim",async t=>{
  const root=mkdtempSync(path.join(os.tmpdir(),"moss-cpu-spawn-"));t.after(()=>rmSync(root,{recursive:true,force:true}));const state={};
  await assert.rejects(executePreparation(ROOT,[path.join(root,"missing-executable")],path.join(root,"spawn.log"),1000,state),/spawn failed/);
  assert.equal(state.completion.completed,false);assert.equal(state.cleanup,undefined);
});

for(const mode of ["timeout","cancel","read","write"])test(`CPU preparation ${mode} aborts and reaps only its real owned handle`,async t=>{
  const root=mkdtempSync(path.join(os.tmpdir(),"moss-cpu-abort-"));t.after(()=>rmSync(root,{recursive:true,force:true}));let child;const state={};const kills=[];
  const result=executePreparation(ROOT,[process.execPath,"-e","console.log('ready');setInterval(()=>{},1000)"],path.join(root,"command.log"),mode==="timeout"?150:5000,state,{
    spawnChild:(...args)=>{child=spawn(...args);const kill=child.kill.bind(child);child.kill=signal=>{kills.push(signal);return kill(signal);};
      if(mode==="cancel")child.stdout.once("data",()=>{state.cancelled=true;});
      if(mode==="read")child.stdout.once("data",()=>{child.stdout.emit("error",new Error("simulated read failure"));});return child;},
    ...(mode==="write"?{writeLog:()=>{throw new Error("simulated full log device");}}:{}),
  });
  await assert.rejects(result,/deadline\/cancellation|read failed|log write failed/);
  assert.deepEqual(kills,["SIGKILL"]);assert.ok(child.exitCode!==null||child.signalCode!==null);
  assert.equal(state.completion.scope,"root-and-stdio");assert.equal(state.completion.stdioClosed,true);assert.equal(state.cleanup,undefined);
});

test("CPU preparation cannot claim completion when stdio remains unclosed after its reap bound",async t=>{
  const root=mkdtempSync(path.join(os.tmpdir(),"moss-cpu-unclosed-"));t.after(()=>rmSync(root,{recursive:true,force:true}));const state={};let child;
  await assert.rejects(executePreparation(ROOT,["metadata-only"],path.join(root,"command.log"),10,state,{graceMs:10,spawnChild:()=>{
    child=new EventEmitter();Object.assign(child,{pid:100,exitCode:0,stdout:new PassThrough(),stderr:new PassThrough(),kill:()=>assert.fail("an exited own handle is never signalled by PID")});
    setImmediate(()=>child.emit("exit",0,null));return child;
  }}),/deadline\/cancellation/);
  assert.equal(state.unsafe,true);assert.equal(state.completion.completed,false);assert.equal(state.completion.stdioClosed,false);
  assert.equal(child.stdout.destroyed,true);assert.equal(child.stderr.destroyed,true);assert.equal(state.cleanup,undefined);
});

test("CPU preparation refuses an observed close after its deadline before the next poll",async t=>{
  const root=mkdtempSync(path.join(os.tmpdir(),"moss-cpu-late-close-"));t.after(()=>rmSync(root,{recursive:true,force:true}));const state={};let clock=0,kills=0;
  await assert.rejects(executePreparation(ROOT,["metadata-only"],path.join(root,"command.log"),5,state,{now:()=>clock,spawnChild:()=>{
    const child=new EventEmitter();Object.assign(child,{pid:100,exitCode:null,stdout:new PassThrough(),stderr:new PassThrough(),kill:()=>{kills++;}});
    setImmediate(()=>{clock=6;child.exitCode=0;child.emit("exit",0,null);child.emit("close",0,null);});
    return child;
  }}),/deadline\/cancellation/);
  assert.equal(state.completion.stdioClosed,true);assert.equal(state.completion.completed,false);assert.equal(kills,0);
});

// Literal subprocess seam: materialize labelled fake assembly bytes so tests
// exercise the real private-file SHA binding; no C#/Windows API executes here.
function simulatedWindowsReaderResult(script,result){
  if(result.error||result.status!==0)return result;
  const compiled=/^'@ -OutputAssembly '((?:[^']|'')+)'$/m.exec(script);
  const assembly=/\$readerBytes = \[IO.File\]::ReadAllBytes\('((?:[^']|'')+)'\)/.exec(script)?.[1].replaceAll("''","'");
  if(compiled)writeFileSync(compiled[1].replaceAll("''","'"),'literal simulated native reader assembly');
  const source=/type_ready source_sha256=([a-f0-9]{64})/.exec(script)[1];
  const marker=`MOSS_PROCESS type_ready source_sha256=${source} assembly_sha256=${sha256(readFileSync(assembly))} elapsed_ms=1\n`;
  return {...result,stderr:marker+(result.stderr??'')};
}

// CI_IMPACT: tiny labelled files and injected subprocess results, never compiled/executed.
test('Windows native reader compiles once then loads the same sealed bytes for full and selected snapshots',t=>{
  const row={pid:42,ppid:1,birthMs:100,path:'owned.exe',rssBytes:10};let calls=0,assembly,source,assemblySha;
  const run=(program,args,options)=>{
    calls++;const script=args.at(-1);assert.equal(program,'powershell.exe');
    assert.equal(options.timeout,calls===1?20000:1234);assert.equal(options.maxBuffer,calls===1?32*1024*1024:1024*1024);
    if(calls===1){assertNativeSnapshotScript(script,'$null');assembly=/^'@ -OutputAssembly '((?:[^']|'')+)'$/m.exec(script)[1].replaceAll("''","'");}
    else{assert.doesNotMatch(script,/Add-Type|Process32FirstW/);assert.match(script,/Read\(\(\[long\[\]\]@\(42\)\)\)/);assert.ok(script.includes(`$readerDigest -cne '${assemblySha}'`));}
    assert.match(script,/\[Reflection.Assembly\]::Load\(\$readerBytes\)/);
    const result=simulatedWindowsReaderResult(script,{status:0,stdout:JSON.stringify([row])});
    const marker=/source_sha256=([a-f0-9]{64}) assembly_sha256=([a-f0-9]{64})/.exec(result.stderr);
    if(calls===1){source=marker[1];assemblySha=marker[2];}else assert.deepEqual(marker.slice(1),[source,assemblySha]);
    return result;
  };
  t.after(()=>disposeWindowsProcessReader(run));
  assert.equal(queryWindowsProcesses({run}).get(42).rssBytes,10);
  assert.equal(queryWindowsProcesses({run,pids:[42],timeoutMs:1234}).get(42).birthMs,100);
  assert.equal(calls,2);assert.ok(existsSync(assembly));
  disposeWindowsProcessReader(run);assert.equal(existsSync(path.dirname(assembly)),false);
  assert.equal(queryWindowsProcesses({run}),null);assert.equal(calls,2,'disposed reader must not recompile');
});

test('Windows native reader refuses changed or missing sealed assembly before launching another process',t=>{
  for(const change of ['bytes','missing']){
    let calls=0,assembly;
    const run=(_p,args)=>{
      calls++;const script=args.at(-1);assembly=/^'@ -OutputAssembly '((?:[^']|'')+)'$/m.exec(script)[1].replaceAll("''","'");
      return simulatedWindowsReaderResult(script,{status:0,stdout:'[]'});
    };t.after(()=>disposeWindowsProcessReader(run));
    assert.equal(queryWindowsProcesses({run}).size,0);
    if(change==='bytes')writeFileSync(assembly,'changed assembly');else rmSync(assembly);
    const failures=[];assert.equal(queryWindowsProcesses({run,onFailure:fact=>failures.push(fact)}),null);
    assert.equal(failures[0].category,'reader-integrity');assert.equal(calls,1);
    assert.equal(queryWindowsProcesses({run}),null);assert.equal(calls,1,'integrity failure never re-prepares');
  }
});

test('Windows reader preparation failure is latched without retrying compilation',t=>{
  let calls=0;const timeout=Object.assign(new Error('literal compiler timeout'),{code:'ETIMEDOUT'});
  const run=()=>{calls++;return {error:timeout,status:null,signal:'SIGTERM',stdout:'',stderr:'MOSS_PROCESS script_entry utc_ms=1 elapsed_ms=0\n'};};
  t.after(()=>disposeWindowsProcessReader(run));const failures=[];
  assert.equal(queryWindowsProcesses({run,onFailure:fact=>failures.push(fact)}),null);
  assert.equal(queryWindowsProcesses({run,onFailure:fact=>failures.push(fact)}),null);
  assert.equal(calls,1);assert.deepEqual(failures.map(fact=>fact.category),['command-error','reader-preparation-unavailable']);
  assert.ok(failures.every(fact=>fact.timedOut&&fact.error.code==='ETIMEDOUT'));
});

test('Windows reader independently authenticates source and produced assembly digests',t=>{
  for(const field of ['source_sha256','assembly_sha256']){
    let calls=0;const run=(_p,args)=>{
      calls++;const result=simulatedWindowsReaderResult(args.at(-1),{status:0,stdout:'[]'});
      return {...result,stderr:result.stderr.replace(new RegExp(`${field}=[a-f0-9]{64}`),`${field}=${'0'.repeat(64)}`)};
    };t.after(()=>disposeWindowsProcessReader(run));const failures=[];
    assert.equal(queryWindowsProcesses({run,onFailure:fact=>failures.push(fact)}),null);assert.equal(failures[0].category,'reader-preparation-unavailable');
    assert.equal(queryWindowsProcesses({run}),null);assert.equal(calls,1);
  }
});

test('Windows cleanup can reuse authenticated compilation after a failed snapshot without re-preparing it',t=>{
  let calls=0;const run=(_p,args)=>{
    calls++;const script=args.at(-1);
    if(calls===1){const result=simulatedWindowsReaderResult(script,{status:0,stdout:'[]'});return {...result,status:1,stdout:'',stderr:result.stderr+'literal native snapshot failure'};}
    assert.doesNotMatch(script,/Add-Type/);return simulatedWindowsReaderResult(script,{status:0,stdout:'[]'});
  };t.after(()=>disposeWindowsProcessReader(run));const failures=[];
  assert.equal(queryWindowsProcesses({run,onFailure:fact=>failures.push(fact)}),null);assert.equal(failures[0].category,'nonzero-status');
  assert.equal(queryWindowsProcesses({run}).size,0);assert.equal(calls,2);
});

function assertNativeSnapshotScript(script,selection) {
  assert.doesNotMatch(script,/Get-CimInstance|Get-WmiObject|Get-Process|HashSet/);
  assert.match(script,/Array.IndexOf\(selected, \(long\)entry.pid\)/);
  const stages=['MOSS_PROCESS script_entry','Add-Type -TypeDefinition', 'MOSS_PROCESS type_ready',`$records = @([MossProcessSnapshot]::Read(${selection}))`,'MOSS_PROCESS snapshot_returned','ConvertTo-Json -InputObject $records -Compress','MOSS_PROCESS json_completed'];
  for(let i=1;i<stages.length;i++)assert.ok(script.indexOf(stages[i-1])<script.indexOf(stages[i]),stages[i]);
  assert.equal((script.match(/\[Console\]::Error\.WriteLine\('MOSS_PROCESS /g)??[]).length,4);
  assert.equal(script.split('\n').filter(line=>line==='ConvertTo-Json -InputObject $records -Compress').length,1);
  for(const api of ['Process32FirstW','Process32NextW','GetProcessTimes','QueryFullProcessImageNameW','GetProcessMemoryInfo','GetSystemTimePreciseAsFileTime'])assert.ok(script.includes(api));
  assert.match(script,/CreateToolhelp32Snapshot\(0x00000002, 0\)/);
  assert.match(script,/OpenProcess\(0x00001000, false, entry.pid\)/);
  assert.match(script,/public UIntPtr heap/);assert.match(script,/SizeConst=260/);
  assert.match(script,/entry.size = \(uint\)Marshal.SizeOf\(typeof\(ProcessEntry\)\)/);
  assert.match(script,/birth <= boundary/);assert.match(script,/row.rssBytes = checked\(\(long\)memory.workingSet.ToUInt64\(\)\)/);
  assert.match(script,/new StringBuilder\(32768\)/);assert.match(script,/QueryFullProcessImageName\(process, 0,/);
  assert.match(script,/finally \{ Close\(process\); \}/);assert.match(script,/finally \{ Close\(snapshot\); \}/);
  assert.match(script,/GetLastWin32Error\(\) != 18/);
}

// CI_IMPACT: literal rows only; actual native API execution remains the existing Windows CPU prefix.
test('Windows snapshot preserves inaccessible graph rows and refuses malformed census records',()=>{
  const row={pid:57240,ppid:44816,birthMs:null,path:'',rssBytes:null};
  for(const pids of [null,[57240]]){
    const rows=queryWindowsProcesses({pids,run:(_p,args)=>simulatedWindowsReaderResult(args.at(-1),{status:0,stdout:JSON.stringify([row])})});
    assert.deepEqual(rows.get(row.pid),row,'protected/vanished detail failure remains present and unknown');
    for(const malformed of [{...row,pid:0},{...row,ppid:null},{...row,birthMs:undefined},{...row,rssBytes:-1},{...row,path:null}]){
      assert.equal(queryWindowsProcesses({pids,run:(_p,args)=>simulatedWindowsReaderResult(args.at(-1),{status:0,stdout:JSON.stringify([malformed])})}),null);
    }
    assert.equal(queryWindowsProcesses({pids,run:(_p,args)=>simulatedWindowsReaderResult(args.at(-1),{status:0,stdout:JSON.stringify([row,row])})}),null);
  }
});

test('Windows unknown owned facts never become zero RSS or cleanup absence',()=>{
  const unrelated={pid:999,ppid:1,birthMs:null,path:'',rssBytes:null};
  const rows=windowsRows(WINDOWS_ROOT,WINDOWS_PARENT,unrelated);
  assert.equal(ownedWindowsTree(rows,WINDOWS_ROOT).authenticated,true);
  assert.equal(sumWindowsOwnedRss(rows,WINDOWS_ROOT),300);
  for(const missing of ['birthMs','path','rssBytes']){
    const unknown={...WINDOWS_PARENT,[missing]:missing==='path'?'':null};rows.set(unknown.pid,unknown);
    if(missing!=='rssBytes')assert.equal(ownedWindowsTree(rows,WINDOWS_ROOT).authenticated,false);
    assert.equal(sumWindowsOwnedRss(rows,WINDOWS_ROOT),null,missing);
  }
  rows.set(WINDOWS_PARENT.pid,{...WINDOWS_PARENT,birthMs:null});
  assert.throws(()=>planAuthenticatedCleanup(rows,WINDOWS_ROOT,new Map([[101,20000]]),true),/birth unavailable/);
  rows.set(101,{...WINDOWS_PARENT,path:''});
  assert.throws(()=>planAuthenticatedCleanup(rows,WINDOWS_ROOT,new Map([[101,20000]]),true),/path unavailable/);
  rows.delete(101);rows.set(100,{...WINDOWS_ROOT,birthMs:null});
  assert.throws(()=>planAuthenticatedCleanup(rows,WINDOWS_ROOT,new Map(),false),/birth unavailable/);
});

test("selected Windows census batches exact PIDs and distinguishes empty JSON from unavailable metadata",()=>{
  const row={pid:57240,ppid:44816,birthMs:1793073218066,path:"C:\\Windows\\System32\\tool.exe",rssBytes:28229632};
  for(const stdout of [JSON.stringify([row]),"[]"]){
    const rows=queryWindowsProcesses({pids:[57240,7708],timeoutMs:1234,run:(program,args,options)=>{
      assert.equal(program,"powershell.exe");assertNativeSnapshotScript(args.at(-1),"([long[]]@(57240,7708))");
      assert.match(args.at(-1),/ConvertTo-Json -InputObject \$records/);assert.equal(options.timeout,1234);assert.equal(options.maxBuffer,1024*1024);
      return simulatedWindowsReaderResult(args.at(-1),{status:0,stdout});
    }});
    assert.ok(rows instanceof Map);assert.equal(rows.size,stdout==="[]"?0:1);
  }
  for(const result of [{status:1,stdout:"[]"},{status:0,stdout:""},{status:0,stdout:"null"},{status:0,stdout:"not-json"},
    {status:0,stdout:JSON.stringify([{...row,path:null}])},
    {status:0,stdout:JSON.stringify([{...row,pid:99}])},{status:0,stdout:JSON.stringify([row,row])},{error:new Error("denied"),stdout:"[]"}]){
    assert.equal(queryWindowsProcesses({pids:[57240],run:(_p,args)=>simulatedWindowsReaderResult(args.at(-1),result)}),null,"denial/malformed rows are not absence");
  }
  for(const pids of [[],[1,1],["1 OR 1=1"],Array.from({length:65},(_,i)=>i+1)])assert.equal(queryWindowsProcesses({pids,run:()=>assert.fail("invalid selection launched a query")}),null);
});

test("Windows real CPU preparation handles fast metadata before compilation",{skip:process.platform!=="win32",timeout:120_000},async t=>{
  const root=mkdtempSync(path.join(os.tmpdir(),"moss-real-preparation-"));t.after(()=>rmSync(root,{recursive:true,force:true}));
  for(const [name,command,code,output] of [
    ["rustc",["rustc","-Vv"],0,/^host: \S+$/m],
    ["cargo",["cargo","-V"],0,/^cargo \S+/m],
    ["list",[process.execPath,"-e","console.log('metadata_only: test')"],0,/metadata_only: test/],
    ["nonzero",[process.execPath,"-e","process.exit(7)"],7,/^$/],
  ]) {
    const state={};let result;
    try{result=await executePreparation(ROOT,command,path.join(root,`${name}.log`),30_000,state);}
    catch(error){t.diagnostic(JSON.stringify({kind:"preparation-failure",name,completion:state.completion??null,unsafe:state.unsafe??null}));throw error;}
    assert.equal(result.exit,code);assert.match(result.output,output);assert.equal(state.completion.completed,true);assert.equal(state.cleanup,undefined);
    t.diagnostic(JSON.stringify({name,completion:state.completion}));
  }

});

test("manual Windows capture checks actual preparation supervision before its first compiler command", () => {
  const capture = readFileSync(path.join(ROOT, ".github/workflows/moss-training-acceptance.yml"), "utf8").split("  cross-load:")[0];
  const check = capture.indexOf("- name: Check Windows bounded CPU metadata before compilation");
  assert.ok(check > 0 && check < capture.indexOf("- name: Capture and publish base/lora before continuing"));
  const step = capture.slice(check, capture.indexOf("      - ", check + 1));
  assert.match(step, /if: \$\{\{ matrix\.backend == 'cuda' \}\}/);
  assert.match(step, /node --test --test-name-pattern="\^Windows real CPU preparation" scripts\/moss-native-api-validation\.test\.mjs/);
  const normal=readFileSync(path.join(ROOT,".github/workflows/windows-candle.yml"),"utf8").split("  imported-nvfp4-worker-smoke:")[0];
  for(const family of ["scripts/moss-native-api-validation*.mjs","scripts/moss-training-acceptance*.mjs"])assert.ok(normal.includes(`      - "${family}"`),`Windows probe must be selected when ${family} changes`);
  const early=normal.indexOf("- name: Check Windows bounded CPU metadata before compilation");
  assert.ok(early>normal.indexOf("- name: Set up Node for Cargo cache selection")&&early<normal.indexOf("- name: Isolate Cargo dependency checkout")&&early<normal.indexOf("- name: Fetch the pinned inference release"));
  const normalStep=normal.slice(early,normal.indexOf("      - ",early+1));
  assert.match(normalStep,/if: \$\{\{ github\.event_name != 'workflow_dispatch' \}\}/);assert.match(normalStep,/timeout-minutes: 3/);
  assert.match(normalStep,/node --test --test-name-pattern="\^Windows real CPU preparation" scripts\/moss-native-api-validation\.test\.mjs/);
});

for (const phase of ["planning", "post-termination"]) {
  test(`Windows ${phase} null telemetry is unsafe, never released, and blocks the next case`, async () => {
    const mock = windowsSupervisor({ planningMissing: phase === "planning", verificationMissing: phase === "post-termination", output: GREEN(EXPECTED_TESTS[0]) });
    const driver = mockDriver();
    const mockedExecute = driver.execute;
    let state;
    driver.execute = async (root, args, log, limitMs, commandState) => {
      if (args[0] === "cargo" || args.includes("--list")) return mockedExecute(root, args, log, limitMs, commandState);
      driver.calls.push(args);
      state = commandState;
      return executeOwnedChild(root, args, log, limitMs, commandState, mock.supervision);
    };
    const { result, receipt, calls } = await pipeline({ backend: "candle", driver });
    assert.equal(result.ok, false);
    assert.equal(calls.length, 4, "compile + two lists + one attempted case, never a next command");
    assert.equal(state.unsafe, true);
    assert.equal(state.cleanup.released, false);
    assert.match(state.cleanup.unproven.join(" "), /process telemetry unavailable/);
    assert.deepEqual(mock.killed, phase === "planning" ? [] : [101, 103]);
    assert.deepEqual(mock.ownHandleKills, []);
    assert.equal(receipt.cases[0].ok, false);
    assert.equal(receipt.cases[0].cleanup.released, false);
    assert.match(receipt.cases[0].error, /unproven cleanup/);
    assert.match(receipt.cases[0].rawOutput, /1 passed; 0 failed; 0 ignored/);
    assert.deepEqual(receipt.cases.slice(1).map((entry) => [entry.attempted, entry.blocked, entry.ok]),
      [[false, true, false], [false, true, false], [false, true, false]]);
  });
}

test('Candle MOSS output uses runner context only at step scope and uploads the same path',()=>{
 const block=jobBlock(readFileSync(path.join(ROOT,'.github/workflows/windows-candle.yml'),'utf8'),'moss-native-api-candle');
 const jobConfiguration=block.split('    steps:\n')[0];
 assert.doesNotMatch(jobConfiguration,/runner\.temp/);
 const run=block.split('      - name: Run the MOSS native API validation (candle)\n')[1].split('\n      - ')[0];
 assert.match(run,/env:\n          MOSS_NATIVE_OUTPUT: \$\{\{ runner\.temp \}\}/);
 const expected=/MOSS_NATIVE_OUTPUT: (.+)/.exec(run)[1];
 const upload=block.split('      - name: Upload MOSS native validation evidence\n')[1];
 assert.ok(upload.includes(`path: ${expected}\n`));
 assert.match(run,/--output "%MOSS_NATIVE_OUTPUT%"/);
});

// CI_IMPACT: injected query results only; platform test above remains the real native Windows check.
test('Windows query failures preserve bounded subtype facts without process stdout',()=>{
 const secret='DO_NOT_LOG_PROCESS_ROWS';
 const timeout=Object.assign(new Error('spawnSync powershell.exe ETIMEDOUT'),{code:'ETIMEDOUT'});
 const markers='MOSS_PROCESS script_entry utc_ms=1793188800000 elapsed_ms=0\nMOSS_PROCESS type_ready elapsed_ms=9\nMOSS_PROCESS snapshot_returned elapsed_ms=12\nMOSS_PROCESS json_completed elapsed_ms=14\n';
 const variants=[
  [{error:timeout,status:null,signal:'SIGTERM',stdout:secret,stderr:'x'.repeat(700)},'command-error'],
  [{error:timeout,status:0,signal:null,stdout:'[{"pid":1,"ppid":0,"birthMs":1,"path":"owned.exe","rssBytes":1}]',stderr:markers},'command-error'],
  [{status:1,stdout:'[]',stderr:'CIM access denied'},'nonzero-status'],
  [{status:0,stdout:' '},'empty-stdout'],[{status:0,stdout:secret},'invalid-json'],[{status:0,stdout:'null'},'invalid-shape'],
 ];
 for(const [result,category] of variants){
  const facts=[];
  assert.equal(queryWindowsProcesses({run:(_p,args,options)=>{assertNativeSnapshotScript(args.at(-1),'$null');assert.equal(options.timeout,20000);assert.equal(options.maxBuffer,32*1024*1024);return simulatedWindowsReaderResult(args.at(-1),result);},onFailure:fact=>facts.push(fact)}),null);
  assert.equal(facts.length,1);const fact=facts[0];assert.equal(fact.category,category);assert.equal(fact.status,result.status);assert.equal(fact.timeoutMs,20000);assert.ok(fact.elapsedMs>=0);assert.equal(fact.timedOut,category==='command-error');
  assert.ok((fact.stderr?.length??0)<=512);assert.equal(JSON.stringify(facts).includes(secret),false);
  if(result.stderr===markers)assert.equal(fact.stderr,markers,'bounded stage markers must survive timeout/status0 refusal unchanged');
 }
 const facts=[];assert.equal(queryWindowsProcesses({run:(_p,args)=>simulatedWindowsReaderResult(args.at(-1),{status:0,stdout:'[]'}),onFailure:f=>facts.push(f)}).size,0);assert.deepEqual(facts,[]);
});
