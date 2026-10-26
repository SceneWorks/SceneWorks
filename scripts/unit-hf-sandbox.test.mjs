import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { lstatSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createUnitHfSandbox } from "./unit-hf-sandbox.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const launcher = path.join(root, "scripts/unit-hf-sandbox.mjs");
function fixture() {
  const base = realpathSync(mkdtempSync(path.join(os.tmpdir(), "unit-hf-fixture-")));
  const checkout = path.join(base, "checkout"), temporary = path.join(base, "runner-temp");
  mkdirSync(checkout); mkdirSync(temporary);
  const env = { ...process.env, RUNNER_TEMP: temporary, GITHUB_RUN_ID: "41", GITHUB_RUN_ATTEMPT: "2", GITHUB_JOB: "unit_test", HF_HOME: "/not-a-real-cache", HF_HUB_CACHE: "/not-a-real-cache/hub", HUGGINGFACE_HUB_CACHE: "/another-not-real-cache" };
  return { base, checkout, env, close: () => rmSync(base, { recursive: true, force: true }) };
}
test("unit sandbox binds all three aliases to one fresh run/job-owned home without mutating the parent", () => {
  const f = fixture();
  try {
    const before = { ...f.env }, parent = { ...process.env };
    const one = createUnitHfSandbox(f.checkout, f.env), two = createUnitHfSandbox(f.checkout, f.env);
    assert.notEqual(one.directory, two.directory);
    assert.equal(path.dirname(one.directory), f.env.RUNNER_TEMP);
    assert.match(path.basename(one.directory), /^unit-hf-41-2-unit_test-/);
    assert.equal(one.env.HF_HUB_CACHE, path.join(one.env.HF_HOME, "hub"));
    assert.equal(one.env.HUGGINGFACE_HUB_CACHE, one.env.HF_HUB_CACHE);
    for (const p of [one.directory, one.env.HF_HOME, one.env.HF_HUB_CACHE]) { assert.equal(lstatSync(p).isSymbolicLink(), false); assert.equal(realpathSync(p), p); }
    for (const [actual, expected] of [[f.env, before], [{ ...process.env }, parent]]) {
      assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort());
      for (const key of Object.keys(expected)) assert.ok(actual[key] === expected[key], `parent environment changed: ${key}`);
    }
  } finally { f.close(); }
});
for (const key of ["GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB"]) test(`unit sandbox rejects missing or path-shaped ${key}`, () => {
  const f = fixture(); try { for (const value of [undefined, "../other", ""]) assert.throws(() => createUnitHfSandbox(f.checkout, { ...f.env, [key]: value })); } finally { f.close(); }
});
test("unit sandbox rejects relative, in-checkout and symlink temporary roots", () => {
  const f = fixture();
  try {
    assert.throws(() => createUnitHfSandbox(f.checkout, { ...f.env, RUNNER_TEMP: "relative" }));
    assert.throws(() => createUnitHfSandbox(f.checkout, { ...f.env, RUNNER_TEMP: f.checkout }));
    const link = path.join(f.base, "linked-temp"); symlinkSync(f.env.RUNNER_TEMP, link, process.platform === "win32" ? "junction" : "dir");
    assert.throws(() => createUnitHfSandbox(f.checkout, { ...f.env, RUNNER_TEMP: link }), /symlink/);
  } finally { f.close(); }
});
test("launcher forwards CPU-child output and nonzero exit while only its descendants inherit owned HF aliases", () => {
  const f = fixture();
  try {
    const result = spawnSync(process.execPath, [launcher, "--", process.execPath, "-e", 'console.log(JSON.stringify([process.env.HF_HOME,process.env.HF_HUB_CACHE,process.env.HUGGINGFACE_HUB_CACHE]));console.error("child stderr");process.exit(7)'], { cwd: f.checkout, env: f.env, encoding: "utf8", timeout: 5000 });
    assert.equal(result.status, 7, result.stderr); assert.match(result.stderr, /child stderr/);
    const [home, hub, alias] = JSON.parse(result.stdout); assert.equal(hub, path.join(home, "hub")); assert.equal(alias, hub); assert.equal(path.dirname(path.dirname(home)), f.env.RUNNER_TEMP);
    assert.equal(f.env.HF_HOME, "/not-a-real-cache");
    if (process.platform !== "win32") {
      const signaled = spawnSync(process.execPath, [launcher, "--", process.execPath, "-e", 'process.kill(process.pid,"SIGTERM")'], { cwd: f.checkout, env: f.env, encoding: "utf8", timeout: 5000 });
      assert.equal(signaled.signal, "SIGTERM");
    }
  } finally { f.close(); }
});

const text = name => readFileSync(path.join(root, name), "utf8");
const selected = {
  ".github/workflows/check.yml": ["python -m pytest -q tests/ -m \"not e2e and not parity\" --strict-markers", "npm run rust:check", "node scripts/scaffold-test-selection.mjs --portable", "node scripts/scaffold-test-selection.mjs --history"],
  ".github/workflows/macos-mlx.yml": ["cargo test -p sceneworks-memory-adapter --features mlx", "cargo test -- --skip nax_16bit_sdpa_is_correct", "cargo test -p sceneworks-worker --test nax_guard"],
  ".github/workflows/windows-candle.yml": ["cargo test -p sceneworks-worker --features backend-candle %capability_facts_test_args%", "cargo test -p sceneworks-memory-adapter --features candle %capability_facts_test_args%"],
};
test("only declared ordinary unit steps use the child sandbox, never ignored real-weight or calibration commands", () => {
  for (const [file, commands] of Object.entries(selected)) {
    const source = text(file);
    assert.equal(source.split("node scripts/unit-hf-sandbox.mjs -- ").length - 1, commands.length, file);
    for (const command of commands) assert.ok(source.includes(`node scripts/unit-hf-sandbox.mjs -- ${command}\n`), command);
    for (const step of source.split(/^      - /m).slice(1)) if (step.includes("node scripts/unit-hf-sandbox.mjs -- ")) assert.doesNotMatch(step, /--ignored|--include-ignored|cargo (?:run|build) --release/);
    assert.doesNotMatch(source, /HF_(?:HOME|HUB_CACHE):.*unit-hf|HUGGINGFACE_HUB_CACHE:.*unit-hf/); // no job-global aliases
  }
  const controller = text("scripts/remote-macos-verification.mjs");
  assert.match(controller, /UNIT_SANDBOX = "scripts\/unit-hf-sandbox\.mjs"/);
  assert.match(controller, /CONTROL_FILES = Object\.freeze\(\[WORKFLOW, CONTROLLER, UNIT_SANDBOX,/);
  assert.match(controller, /const runEnv = \{ \.\.\.remoteBuildEnv\(env, request.profile\), \.\.\.createUnitHfSandbox\(root, env\)\.env \};/);
  assert.doesNotMatch(controller, /canonical-preflight|scripts\/preflight\.mjs|unit-runner|stamp\.txt/);
  assert.deepEqual(JSON.parse(text("package.json")).scripts.preflight, undefined);
});
test("existing API cache and Iris environment guards clear every inherited HF alias", () => {
  const support = text("apps/rust-api/src/tests/support.rs");
  const isolate = support.slice(support.indexOf("pub(crate) fn isolate_hf_cache()"), support.indexOf("impl Drop for HfCacheEnvGuard"));
  assert.match(isolate, /\["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"\]/); assert.match(isolate, /std::env::remove_var\(key\)/);
  const pin = support.slice(support.indexOf("pub(crate) fn pin_hf_hub_cache("), support.indexOf("pub(crate) fn readiness_worker("));
  assert.match(pin, /\["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"\]/); assert.match(pin, /std::env::remove_var\(key\)/);
  for (const file of ["apps/rust-api/src/tests/iris_training.rs", "apps/rust-api/src/tests/iris_task_training.rs"]) { const body = text(file); assert.match(body, /let hf = isolate_hf_cache\(\);/); assert.match(body, /_hf: hf/); }
});
