#!/usr/bin/env node
// Closed S1/S2 consumer assertion proof. This driver never runs an integration/default/ignored
// test, never writes a preflight stamp, and accepts no operator-supplied command or mutation.
import { createHash } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, utimesSync, mkdirSync, renameSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const REGISTRY_PATH = "scripts/moss-consumer-assertions.cases.json";
export const MUTABLE_FILES = [
  "crates/sceneworks-core/src/training.rs",
  "crates/sceneworks-core/src/lora_family.rs",
  "crates/sceneworks-core/src/jobs_store/routing/catalog.rs",
  "crates/sceneworks-core/src/jobs_store/routing/candle.rs",
  "crates/sceneworks-core/src/jobs_store/routing/mlx.rs",
  "crates/sceneworks-core/src/jobs_store/routing/matrix.rs",
  "crates/sceneworks-worker/src/training_jobs.rs",
];
export const TESTS = {
  target: ["sceneworks-core", "training::moss_training_s1_tests::moss_targets_keep_explicit_checkpoint_and_method_choices"],
  library: ["sceneworks-core", "lora_family::moss_training_adapter_tests::native_moss_import_keeps_exact_base_and_refuses_cross_checkpoint_use"],
  scheduler: ["sceneworks-core", "jobs_store::routing::catalog::moss_native_training_routing_tests::moss_exact_checkpoint_adapters_route_to_both_native_lanes_not_generic"],
  catalog: ["sceneworks-core", "jobs_store::routing::catalog::tests::routed_model_lists_match_snapshot"],
  matrix: ["sceneworks-core", "jobs_store::routing::matrix::tests::checked_in_matrix_matches_all_authoritative_sources"],
  thin: ["sceneworks-worker", "training_jobs::tests::moss_shared_finalization_defaults_strict_options_and_preserves_explicit_sampler"],
  protocol: ["sceneworks-worker", "training_jobs::tests::moss_native_requests_keep_official_bf16_source_and_requested_compute_for_both_methods"],
};
const fail = message => { throw new Error(`MOSS consumer assertions: ${message}`); };
const requireFact = (value, message) => { if (!value) fail(message); };
export const digest = value => createHash("sha256").update(value).digest("hex");
const normalizedDigest = value => digest(value.replace(/\s+/g, ""));
const SHA = /^[0-9a-f]{40}$/;

// Byte-scoped Rust patching, not regex replacement across a crate. Ignore braces in comments,
// strings, raw strings and character literals; ambiguous anchors are failures, never skips.
export function bodySpan(text, anchor, start = 0, requireUnique = true) {
  const first = text.indexOf(anchor, start);
  requireFact(first >= 0 && (!requireUnique || text.indexOf(anchor, first + anchor.length) === -1), `missing/ambiguous anchor: ${anchor}`);
  const opening = text.indexOf("{", first + anchor.length - 1);
  requireFact(opening >= 0, "body opening missing");
  let depth = 1;
  for (let i = opening + 1; i < text.length; i += 1) {
    if (text.startsWith("//", i)) { const end = text.indexOf("\n", i); i = end < 0 ? text.length : end; continue; }
    if (text.startsWith("/*", i)) {
      let nesting = 1; i += 2;
      while (i < text.length && nesting) { if (text.startsWith("/*", i)) { nesting += 1; i += 2; } else if (text.startsWith("*/", i)) { nesting -= 1; i += 2; } else i += 1; }
      requireFact(nesting === 0, "unclosed comment"); i -= 1; continue;
    }
    const raw = /^r(#+)?"/.exec(text.slice(i));
    if (raw) { const closing = `"${raw[1] ?? ""}`; const end = text.indexOf(closing, i + raw[0].length); requireFact(end >= 0, "unclosed raw string"); i = end + closing.length - 1; continue; }
    if (text[i] === '"') { for (i += 1; i < text.length; i += 1) { if (text[i] === "\\") i += 1; else if (text[i] === '"') break; } continue; }
    if (text[i] === "'") { const character = /^'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^'\\])'/u.exec(text.slice(i)); if (character) { i += character[0].length - 1; continue; } }
    if (text[i] === "{") depth += 1;
    else if (text[i] === "}" && --depth === 0) return [opening + 1, i];
  }
  fail(`unclosed body: ${anchor}`);
}

export function patchText(text, patch) {
  let left = 0, right = text.length;
  if (patch.anchor) [left, right] = bodySpan(text, patch.anchor);
  const section = text.slice(left, right);
  if (patch.op === "body") return text.slice(0, left) + `\n${patch.new}\n` + text.slice(right);
  requireFact(patch.op === "replace" && patch.old && patch.old !== patch.new, "invalid fixed patch");
  requireFact(section.split(patch.old).length - 1 === 1, `patch must match exactly once: ${patch.old}`);
  return text.slice(0, left) + section.replace(patch.old, patch.new) + text.slice(right);
}

export function loadRegistry(root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")) {
  const bytes = readFileSync(path.join(root, REGISTRY_PATH));
  const spec = JSON.parse(bytes);
  requireFact(spec.schema_version === 2 && Array.isArray(spec.cases) && spec.cases.length === 74 && !Object.hasOwn(spec, "carried"), "closed 74-case executed registry required");
  const ids = new Set();
  for (const item of spec.cases) {
    requireFact(/^SW-[A-Za-z0-9-]+$/.test(item.id) && !ids.has(item.id), "duplicate/invalid case id"); ids.add(item.id);
    requireFact(Array.isArray(item.tests) && item.tests.length > 0 && item.tests.every(key => Object.hasOwn(TESTS, key)), "unknown exact test selector");
    requireFact(Array.isArray(item.patches) && item.patches.length > 0 && item.patches.every(patch => MUTABLE_FILES.includes(patch.file) && ["replace", "body"].includes(patch.op)), "patch outside closed source scope");
  }
  return { ...spec, registry_sha256: digest(bytes) };
}

export function parseTestResult(output, exit, exact, expected) {
  const summaries = [...output.matchAll(/test result: (ok|FAILED)\. ([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored;/g)];
  requireFact(summaries.length === 1 && !/error\[E[0-9]+\]|error: could not compile|could not execute process/.test(output), "compile/error/zero-summary is not an assertion proof");
  const [, state, passed, failed, ignored] = summaries[0];
  const record = { passed: Number(passed), failed: Number(failed), ignored: Number(ignored), exit_code: exit };
  const markers = output.split("\n").filter(value => value.startsWith(`test ${exact} ... `));
  requireFact(markers.length === 1, "selected exact test marker missing or duplicated");
  const line = markers[0];
  const failedMarker = line.includes("FAILED") || output.split("\n").some(value => value === "FAILED");
  const passedMarker = line.endsWith("ok") || output.split("\n").some(value => value === "ok");
  if (expected === "red") requireFact(Number.isInteger(exit) && exit > 0 && state === "FAILED" && record.passed === 0 && record.failed === 1 && record.ignored === 0 && failedMarker && /panicked at|assertion .*failed/.test(output), "mutant survived, selected no test, or failed outside its assertion");
  else requireFact(exit === 0 && state === "ok" && record.passed === 1 && record.failed === 0 && record.ignored === 0 && passedMarker, "restore/baseline must execute exactly one positive test");
  return record;
}

export function validateAssertionReceipt(result, spec, expectedSha, expectedDriverHash) {
  requireFact(result.schema_version === 2 && result.status === "passed" && result.source_sha === expectedSha, "assertion result source/status mismatch");
  requireFact(Number.isFinite(result.elapsed_seconds) && result.elapsed_seconds >= 0 && result.elapsed_seconds <= 80 * 60, "assertion driver exceeded its original bound");
  requireFact(result.registry_sha256 === spec.registry_sha256 && /^[0-9a-f]{64}$/.test(expectedDriverHash ?? "") && result.driver_sha256 === expectedDriverHash, "assertion implementation/registry mismatch");
  requireFact(JSON.stringify(result.test_ids) === JSON.stringify(Object.keys(TESTS)), "incomplete focused test scope");
  requireFact(result.baseline.length === Object.keys(TESTS).length && result.final_green.length === Object.keys(TESTS).length, "missing baseline/final positive tests");
  const positive = records => records.every(row => row.passed === 1 && row.failed === 0 && row.ignored === 0 && row.exit_code === 0);
  requireFact(positive(result.baseline) && positive(result.final_green) && JSON.stringify(result.baseline.map(row => row.test_id)) === JSON.stringify(Object.keys(TESTS)) && JSON.stringify(result.final_green.map(row => row.test_id)) === JSON.stringify(Object.keys(TESTS)), "baseline/final test not positive or exact");
  requireFact(result.cases.length === spec.cases.length && result.cases.every((row, index) => row.id === spec.cases[index].id && row.killed === true && row.restored === true && row.red.length === spec.cases[index].tests.length && row.green.length === row.red.length && JSON.stringify(row.red.map(test => test.test_id)) === JSON.stringify(spec.cases[index].tests) && JSON.stringify(row.green.map(test => test.test_id)) === JSON.stringify(spec.cases[index].tests) && row.red.every(test => test.passed === 0 && test.failed === 1 && test.ignored === 0 && Number.isInteger(test.exit_code) && test.exit_code > 0) && positive(row.green)), "incomplete, surviving or unrestored mutation proof");
  requireFact(result.restored === true && result.clean_after === true && result.head_after === expectedSha && Object.values(result.source_before).every(value => /^[0-9a-f]{64}$/.test(value)) && JSON.stringify(Object.keys(result.source_before).sort()) === JSON.stringify([...MUTABLE_FILES].sort()) && JSON.stringify(result.source_before) === JSON.stringify(result.source_after), "source restoration mismatch");
  requireFact(!Object.hasOwn(result, "carried"), "stale carried evidence is not accepted; every registry case must execute");
}

function git(root, ...args) {
  const p = spawnSync("git", args, { cwd: root, encoding: "utf8" });
  requireFact(!p.error && p.status === 0, `git ${args[0]} failed`); return p.stdout.trim();
}
function snapshot(root) { return { head: git(root, "rev-parse", "HEAD"), clean: git(root, "status", "--porcelain") === "" }; }
function store(file, value) { mkdirSync(path.dirname(file), { recursive: true }); const tmp = `${file}.${process.pid}.tmp`; writeFileSync(tmp, `${JSON.stringify(value, null, 2)}\n`); renameSync(tmp, file); }

export async function executeCpuCommand(root, args, log, limitMs, state) {
  const { openSync, closeSync, writeSync } = await import("node:fs");
  const fd = openSync(log, "w");
  console.log(JSON.stringify({ command: args, log: path.basename(log), bound_ms: limitMs }));
  const child = spawn(args[0], args.slice(1), { cwd: root, env: process.env, stdio: ["ignore", "pipe", "pipe"], detached: true });
  for (const stream of [child.stdout, child.stderr]) stream.on("data", bytes => { writeSync(fd, bytes); process.stdout.write(bytes); });
  state.child = Number.isInteger(child.pid) ? child : null;
  const stop = signal => { try { process.kill(-child.pid, signal); } catch (error) { if (error.code !== "ESRCH") throw error; } };
  let exceeded = false, escalation;
  const timer = setTimeout(() => { exceeded = true; stop("SIGTERM"); escalation = setTimeout(() => stop("SIGKILL"), 2000); }, limitMs);
  try {
    const exit = await new Promise((resolve, reject) => { child.on("error", reject); child.on("close", (code, signal) => resolve(code ?? (signal ? 1 : 0))); });
    requireFact(!exceeded && !state.cancelled, "managed deadline/cancellation");
    return { exit, output: readFileSync(log, "utf8") };
  } finally { clearTimeout(timer); if (escalation) clearTimeout(escalation); if (state.cancelEscalation) { clearTimeout(state.cancelEscalation); state.cancelEscalation = null; } if ((exceeded || state.cancelled) && Number.isInteger(child.pid)) stop("SIGKILL"); state.child = null; closeSync(fd); }
}

export async function runAssertions(root, output, { execute = executeCpuCommand, inspect = snapshot, now = () => performance.now(), budgetMs = 80 * 60 * 1000, spec = loadRegistry(), signals = true } = {}) {
  const start = now(), deadline = start + budgetMs;
  requireFact(budgetMs > 0 && budgetMs <= 80 * 60 * 1000, "invalid driver bound");
  const initial = inspect(root); requireFact(SHA.test(initial.head) && initial.clean, "exact clean source required before assertion driver");
  const originals = Object.fromEntries(MUTABLE_FILES.map(file => [file, readFileSync(path.join(root, file))]));
  const hashes = values => Object.fromEntries(MUTABLE_FILES.map(file => [file, digest(values[file])]));
  const restore = () => { for (const file of MUTABLE_FILES) { writeFileSync(path.join(root, file), originals[file]); utimesSync(path.join(root, file), new Date(), new Date()); } };
  const verifyRestored = () => MUTABLE_FILES.every(file => readFileSync(path.join(root, file)).equals(originals[file]));
  // Validate every planned patch against baseline BEFORE spending a compile or changing a byte.
  for (const item of spec.cases) {
    const changed = {};
    for (const patch of item.patches) changed[patch.file] = patchText(changed[patch.file] ?? originals[patch.file].toString(), patch);
  }
  mkdirSync(output, { recursive: false });
  const resultPath = path.join(output, "assertions.json");
  const result = { schema_version: 2, status: "running", source_sha: initial.head, registry_sha256: spec.registry_sha256, driver_sha256: digest(readFileSync(fileURLToPath(import.meta.url))), test_ids: Object.keys(TESTS), baseline: [], cases: [], final_green: [], source_before: hashes(originals), source_after: {}, restored: false };
  const state = { child: null, cancelled: false, cancelEscalation: null };
  const cancel = () => {
    state.cancelled = true;
    if (state.child) {
      const pid = state.child.pid;
      const stop = signal => { try { process.kill(-pid, signal); } catch (error) { if (error.code !== "ESRCH") throw error; } };
      stop("SIGTERM");
      if (!state.cancelEscalation) state.cancelEscalation = setTimeout(() => stop("SIGKILL"), 2000);
    } else restore();
    // An active compiler is reaped before finally restores source; it cannot race a restore write.
  };
  if (signals) { process.on("SIGTERM", cancel); process.on("SIGINT", cancel); }
  const invoke = async (args, label, maxMs) => {
    requireFact(!state.cancelled && deadline > now(), "original driver deadline exhausted");
    const log = path.join(output, `${label}.log`);
    return execute(root, args, log, Math.min(maxMs, deadline - now()), state);
  };
  const selected = async (key, label, expected) => {
    const [pkg, exact] = TESTS[key];
    const run = await invoke(["cargo", "test", "--locked", "-p", pkg, "--lib", exact, "--", "--exact", "--nocapture"], label, 5 * 60 * 1000);
    return { test_id: key, ...parseTestResult(run.output, run.exit, exact, expected) };
  };
  try {
    const build = await invoke(["cargo", "test", "--locked", "-p", "sceneworks-core", "-p", "sceneworks-worker", "--lib", "--no-run"], "initial-compile", 35 * 60 * 1000);
    requireFact(build.exit === 0, "initial focused-library compilation failed");
    for (const key of Object.keys(TESTS)) result.baseline.push(await selected(key, `baseline-${key}`, "green"));
    store(resultPath, result);
    // Registry order is mechanism-grouped: core mutations finish before worker mutations, so a
    // core patch never asks Cargo to rebuild a dependent worker test binary.
    for (const item of spec.cases) {
      const record = { id: item.id, red: [], green: [], killed: false, restored: false };
      result.cases.push(record);
      try {
        for (const patch of item.patches) { const file = path.join(root, patch.file); writeFileSync(file, patchText(readFileSync(file, "utf8"), patch)); utimesSync(file, new Date(), new Date()); }
        for (const key of item.tests) record.red.push(await selected(key, `${item.id}-red-${key}`, "red"));
        record.killed = true;
      } finally { restore(); record.restored = verifyRestored(); store(resultPath, result); }
      requireFact(record.restored, "mutation source did not restore");
      for (const key of item.tests) record.green.push(await selected(key, `${item.id}-green-${key}`, "green"));
      store(resultPath, result);
    }
    for (const key of Object.keys(TESTS)) result.final_green.push(await selected(key, `final-${key}`, "green"));
    result.status = "passed";
  } catch (error) { result.status = "failed"; result.failure = error.message; }
  finally {
    restore(); result.restored = verifyRestored();
    result.source_after = hashes(Object.fromEntries(MUTABLE_FILES.map(file => [file, readFileSync(path.join(root, file))])));
    const after = inspect(root); result.clean_after = after.clean; result.head_after = after.head;
    result.elapsed_seconds = (now() - start) / 1000;
    if (signals) { process.off("SIGTERM", cancel); process.off("SIGINT", cancel); }
    if (!result.restored || !after.clean || after.head !== initial.head || now() > deadline) result.status = "failed";
    if (result.status === "passed") {
      try { validateAssertionReceipt(result, spec, initial.head, digest(readFileSync(fileURLToPath(import.meta.url)))); }
      catch (error) { result.status = "failed"; result.failure = error.message; }
    }
    store(resultPath, result);
  }
  return result;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [root, output] = process.argv.slice(2);
  if (!root || !output || process.argv.length !== 4) { console.error("usage: moss-consumer-assertions.mjs ROOT NEW_OUTPUT"); process.exitCode = 1; }
  else runAssertions(path.resolve(root), path.resolve(output)).then(result => { console.log(JSON.stringify({ status: result.status, cases: result.cases.length, source_sha: result.source_sha })); process.exitCode = result.status === "passed" ? 0 : 1; }).catch(error => { console.error(error.message); process.exitCode = 1; });
}
