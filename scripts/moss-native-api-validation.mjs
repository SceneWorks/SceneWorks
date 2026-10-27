#!/usr/bin/env node
// Manual-only validation driver for the four #[ignore]d MOSS native API tests
// (tests::moss_training_native::*_round_trip_cpu — real native model execution,
// excluded from ordinary PR/main CI by their #[ignore] attributes). Its only
// callers are the two explicit workflow_dispatch jobs: macos-mlx.yml
// `moss-native-api-mlx` and windows-candle.yml `moss-native-api-candle`.
//
// Compilation and discovery never count as an execution pass: every case must
// report exactly 1 passed / 0 failed / 0 ignored through the proven
// parseTestResult green contract. Receipts (commands, exits, durations, raw
// output, whitelisted environment — never credentials) are written before the
// exit status so evidence uploads even when the run fails. Supervision reuses
// the closed moss-consumer-assertions driver's owned-CPU tree on POSIX (own
// reauthenticated pids only — never process groups or broad pkill). On Windows
// the root is only ever terminated through Node's own child handle and any
// descendant only after its pid+birth+path identity is re-authenticated against
// a fresh sample — never a bare `taskkill /T` tree, which PID reuse could aim
// at a stranger.
//
// BOUNDS ARE THE EXISTING BOUNDED CPU VALIDATION CONTRACT, NOT MEASURED PEAKS
// OR FIT PROOFS: 180 seconds external per case (the case's own internal 120s
// run + 30s join + 15s server teardown stay untouched inside the test), and a
// 2 GiB AGGREGATE OWNED-RSS cap stated by the job environment
// (MOSS_NATIVE_RSS_CAP_GB) on BOTH macOS and Windows — never a Mac-only guard.
// The 2 GiB is an ADMITTED CAP — it matches the native fixture's HOST_CAP_BYTES
// memory-budget override in the inference checkout's moss-image test support —
// not a measured peak, and no claim is made here that the API resident set
// fits under it. The cap is fixed in the workflows and pinned by source tests;
// widening it is a parent decision that follows a real recorded overrun, never
// a silent expansion. An unstated or invalid cap fails the run before
// compilation. Compile time has its own separate workflow timeout and is not
// part of the per-case bound.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { digest, executeCpuCommand, parseTestResult } from "./moss-consumer-assertions.mjs";
import { lockedInferenceRevision } from "./select-windows-cargo-cache.mjs";

export const MOSS_PREFIX = "tests::moss_training_native::";
export const TEST_SOURCE = "apps/rust-api/src/tests/moss_training_native.rs";
export const EXPECTED_TESTS = Object.freeze([
  "tests::moss_training_native::moss_real_dataset_base_lora_round_trip_cpu",
  "tests::moss_training_native::moss_real_dataset_base_lokr_round_trip_cpu",
  "tests::moss_training_native::moss_real_dataset_sft_lora_round_trip_cpu",
  "tests::moss_training_native::moss_real_dataset_sft_lokr_round_trip_cpu",
]);
export const BOUNDS = Object.freeze({ compileMs: 60 * 60_000, listMs: 5 * 60_000, caseMs: 180_000 });
// The existing bounded CPU validation contract cap: 2 GiB AGGREGATE owned RSS
// per case — the native fixture's admitted HOST_CAP_BYTES, not a measured
// peak. Widening this is a parent decision that follows a real recorded
// overrun on the remote validation runs — never a silent expansion.
export const RSS_CAP_GB = 2;
export const ENVIRONMENT_WHITELIST = Object.freeze([
  "CI", "GITHUB_EVENT_NAME", "GITHUB_SHA", "GITHUB_REPOSITORY", "GITHUB_WORKFLOW_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB",
  "RUNNER_OS", "RUNNER_ARCH", "MACOSX_DEPLOYMENT_TARGET", "MOSS_NATIVE_RSS_CAP_GB",
  "CARGO_BUILD_JOBS", "CARGO_INCREMENTAL", "CARGO_TARGET_DIR",
  "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG", "CARGO_PROFILE_DEV_STRIP", "CARGO_PROFILE_TEST_STRIP",
  "CARGO_PROFILE_DEV_BUILD_OVERRIDE_DEBUG", "CARGO_PROFILE_TEST_BUILD_OVERRIDE_DEBUG",
  "CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP", "CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP",
  "RUSTUP_TOOLCHAIN", "RUSTC_WRAPPER", "NVCC_CCBIN", "PMETAL_MLX_PREBUILT_DIR", "PMETAL_METALLIB_PATH", "CUDA_VISIBLE_DEVICES",
]);

const fail = (message) => { throw new Error(`MOSS native validation: ${message}`); };
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sortedTests = () => [...EXPECTED_TESTS].sort();

export function parseArgs(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i]?.startsWith("--") || argv[i + 1] === undefined) fail(`expected --option value pairs, got ${argv.slice(i).join(" ")}`);
    options[argv[i].slice(2)] = argv[i + 1];
  }
  for (const key of ["root", "output", "backend"]) if (!options[key]) fail(`--${key} is required`);
  if (options.backend !== "mlx" && options.backend !== "candle") fail(`--backend must be mlx or candle, got ${options.backend}`);
  options.root = path.resolve(options.root);
  options.output = path.resolve(options.output);
  const relative = path.relative(options.root, options.output);
  if (relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative)))
    fail("output must live outside the checkout so the clean-tree proof stays meaningful");
  return options;
}

export function requireManualContext(env) {
  if (env.GITHUB_EVENT_NAME !== "workflow_dispatch") fail(`manual validation requires an explicit workflow_dispatch, got ${env.GITHUB_EVENT_NAME ?? "none"}`);
}

// The job environment must restate the external aggregate owned-RSS cap, and it
// must be EXACTLY the existing bounded CPU validation contract cap — a wider
// value is refused here, so no environment edit alone can loosen the guard.
// Unstated caps fail the run before compilation. The stated value is recorded
// in every receipt.
export function statedRssCapBytes(env) {
  const raw = env.MOSS_NATIVE_RSS_CAP_GB;
  if (raw === undefined || raw === "") return null;
  if (Number(raw) !== RSS_CAP_GB) fail(`MOSS_NATIVE_RSS_CAP_GB must state the existing bounded CPU validation contract cap (${RSS_CAP_GB} GiB, an admitted cap), got ${raw}`);
  return RSS_CAP_GB * 1024 ** 3;
}

export function defaultInspect(root) {
  const run = (...args) => {
    const result = spawnSync("git", args, { cwd: root, encoding: "utf8" });
    if (result.error || result.status !== 0) fail(`git ${args[0]} failed`);
    return result.stdout.trim();
  };
  return { head: run("rev-parse", "HEAD"), clean: run("status", "--porcelain") === "" };
}

// Structural guard against the four tests entering ordinary CI: the source must
// still declare exactly them, each directly preceded by its #[ignore].
export function ignoredSourceFacts(text) {
  const lines = text.split("\n");
  const facts = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (!/^\s*#\[(?:tokio::)?test(?:\(|\])/.test(lines[i])) continue;
    const header = lines.slice(i, i + 8).join("\n");
    const match = /^\s*(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m.exec(header);
    if (!match) fail("native test attribute has no inspectable function signature");
    facts.push({ test: MOSS_PREFIX + match[1], ignored: /#\[ignore(?:\s|=|\])/.test(header.slice(0, match.index)) });
  }
  const found = facts.map((fact) => fact.test).sort();
  if (JSON.stringify(found) !== JSON.stringify(sortedTests())) fail(`source must declare exactly the four expected native tests, found ${JSON.stringify(found)}`);
  const unignored = facts.filter((fact) => !fact.ignored).map((fact) => fact.test);
  if (unignored.length) fail(`native tests missing #[ignore] in source: ${JSON.stringify(unignored)}`);
  return { ignored: facts.map((fact) => fact.test) };
}

export function parseTestList(text) {
  const entries = [];
  for (const line of text.split("\n")) {
    const match = /^(\S.*): (test|benchmark)$/.exec(line.trimEnd());
    if (match) entries.push({ name: match[1], kind: match[2] });
  }
  return entries;
}

// Exact discovery against the UNFILTERED --list plus the --ignored --list: the
// moss prefix must yield exactly the four expected FQNs, and all four must be
// ignored. Missing, extra or not-ignored are failures, never skips.
export function checkDiscovery(listEntries, ignoredEntries) {
  const testsOf = (entries) => entries.filter((entry) => entry.kind === "test" && entry.name.startsWith(MOSS_PREFIX)).map((entry) => entry.name).sort();
  const discovered = testsOf(listEntries);
  const ignored = testsOf(ignoredEntries);
  const expected = sortedTests();
  const missing = expected.filter((test) => !discovered.includes(test));
  const extras = discovered.filter((test) => !expected.includes(test));
  if (missing.length || extras.length) fail(`discovery mismatch — missing: ${JSON.stringify(missing)} extras: ${JSON.stringify(extras)}`);
  const notIgnored = expected.filter((test) => !ignored.includes(test));
  const wrongIgnored = ignored.filter((test) => !expected.includes(test));
  if (notIgnored.length || wrongIgnored.length) fail(`ignored mismatch — not ignored: ${JSON.stringify(notIgnored)} unexpected ignored: ${JSON.stringify(wrongIgnored)}`);
  return { discovered, ignored };
}

export function defaultStatExecutable(file) {
  if (!existsSync(file)) return false;
  if (process.platform === "win32") return true;
  return (statSync(file).mode & 0o111) !== 0;
}

// Exactly one compiler-artifact executable for the sceneworks_rust_api lib
// target, resolved from the current compile's messages — never a glob, never a
// stale binary from an earlier build.
export function selectLibTestExecutable(jsonl, statExecutable = defaultStatExecutable) {
  const candidates = [];
  for (const line of jsonl.split("\n")) {
    let event;
    try { event = JSON.parse(line); } catch { continue; }
    if (event.reason !== "compiler-artifact" || !event.executable) continue;
    const target = event.target ?? {};
    if (target.name !== "sceneworks_rust_api" || !(target.kind ?? []).includes("lib")) continue;
    candidates.push(event.executable);
  }
  if (candidates.length !== 1) fail(`expected one sceneworks_rust_api libtest executable, got ${JSON.stringify(candidates)}`);
  if (!statExecutable(candidates[0])) fail(`resolved executable is not runnable: ${candidates[0]}`);
  return candidates[0];
}

export const PRE_MODEL_CPU_TEST = "tests::moss_training_acceptance_contracts::pre_model_library_and_dispatch_contracts";

// Only source-owned compile, metadata, schema and the one pre-model CPU
// contract entry may use preparation completion. Model tests retain their supervisor.
export function isPreparationCommand(root, args) {
  if (!Array.isArray(args) || args.some(arg => typeof arg !== "string")) return false;
  if (args.length === 2 && ((args[0] === "rustc" && ["-Vv", "-vV"].includes(args[1])) || (args[0] === "cargo" && args[1] === "-V"))) return true;
  if (args[0] === "cargo" && args[1] === "test" && args.includes("--no-run") && !args.includes("--")) return true;
  if ((args.length === 2 && args[1] === "--list") || (args.length === 3 && args[1] === "--ignored" && args[2] === "--list")) return true;
  if (args.length === 5 && args[1] === PRE_MODEL_CPU_TEST && args[2] === "--exact"
      && args[3] === "--nocapture" && args[4] === "--test-threads=1") return true;
  if (args.length === 7 && args.slice(0, 5).join(" ") === "python3 -I -m jsonschema -i"
      && path.resolve(args[6]) === path.join(root, "scripts/ci/moss-training-acceptance/evidence.schema.json")) return true;
  return args.length === 11 && args[0] === "python3" && args[1] === "-I"
    && path.resolve(args[2]) === path.join(root, "scripts/ci/moss-training-acceptance/verify_evidence.py")
    && [args[3], args[5], args[7], args[9]].join(" ") === "--root --manifest --selection --output";
}

// CPU preparation proves completion of the child handle and captured stdio.
// It neither censuses nor claims release of descendants. Cancellation/failure
// aborts preparation; only this invocation's own handle may be terminated.
export async function executePreparation(root, args, log, limitMs, state, { env = process.env, spawnChild = spawn, graceMs = 10_000, writeLog = null, now = () => performance.now() } = {}) {
  const { openSync, closeSync, writeSync } = await import("node:fs");
  if (state.cancelled) fail("preparation cancelled before launch");
  if (!Number.isFinite(limitMs) || limitMs <= 0) fail("invalid preparation deadline");
  const fd = openSync(log, "w"), started = now();
  let child, closed = false, rootExited = false, exit = null, signal = null, failure = null, logOpen = true;
  const wait = ms => new Promise(resolve => {
    const done = () => { clearTimeout(timer); child?.off("close", done); resolve(); };
    const timer = setTimeout(done, ms);
    if (closed) done(); else child?.once("close", done);
  });
  try {
    child = spawnChild(args[0], args.slice(1), { cwd: root, env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
    child.on("error", error => { failure ??= `preparation spawn failed: ${error.message}`; });
    child.on("exit", () => { rootExited = true; });
    child.on("close", (code, childSignal) => { closed = true; exit = code; signal = childSignal; });
    for (const [name, stream] of [["stdout", child.stdout], ["stderr", child.stderr]]) {
      if (!stream) { failure ??= `preparation ${name} unavailable`; continue; }
      stream.on("error", () => { failure ??= `preparation ${name} read failed`; });
      stream.on("data", bytes => {
        if (!logOpen) return;
        try { (writeLog ?? writeSync)(fd, bytes); process.stdout.write(bytes); } catch { failure ??= "preparation log write failed"; }
      });
    }
    while (!closed && !failure) {
      if (state.cancelled || now() - started >= limitMs) { failure = "preparation deadline/cancellation"; break; }
      await wait(25);
    }
    if (state.cancelled || now() - started >= limitMs) failure ??= "preparation deadline/cancellation";
  } catch (error) { failure ??= error.message; }
  finally {
    if (child && !closed) {
      if (child.exitCode === null) { try { child.kill("SIGKILL"); } catch { /* bounded close below decides completion */ } }
      await wait(graceMs);
      if (!closed) {
        state.unsafe = true;
        child.stdout?.destroy(); child.stderr?.destroy();
        failure ??= "preparation root/stdio did not close within the reap bound";
      }
    }
    state.completion = { scope: "root-and-stdio", completed: closed && rootExited && !state.unsafe && !failure && !state.cancelled, rootExited, stdioClosed: closed, exitCode: exit, signal };
    logOpen = false; closeSync(fd);
  }
  if (failure) fail(failure);
  if (!closed || !rootExited || signal !== null || !Number.isInteger(exit)) fail("preparation completion unavailable");
  return { exit, output: readFileSync(log, "utf8") };
}

// ---- Windows owned-process supervision -------------------------------------------------------
// Identities come from Win32_Process (the same CreationDate/ExecutablePath
// technique scripts/lib/yue2-windows-owned-gpu.mjs validates GPU captures
// with). A process is signalled only while its pid, birth and — for the root —
// executable path still match what we observed. There is no POSIX ps and no
// getuid here; a bare `taskkill /T` tree is never used.

export function queryWindowsProcesses({pids = null, timeoutMs = 20_000, run = spawnSync} = {}) {
  if (pids !== null && (!Array.isArray(pids) || !pids.length || pids.length > 64 || new Set(pids).size !== pids.length || !pids.every(pid=>Number.isSafeInteger(pid)&&pid>0))) return null;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return null;
  const filter=pids===null?"":` -Filter '${pids.map(pid=>`ProcessId = ${pid}`).join(" OR ")}'`;
  const script = [
    "$ErrorActionPreference = 'Stop'",
    `$rows = @(Get-CimInstance Win32_Process${filter} | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,WorkingSetSize)`,
    "$records = @($rows | ForEach-Object { @{ pid = [int]$_.ProcessId; ppid = [int]$_.ParentProcessId;",
    "  birthMs = [long](([DateTimeOffset]$_.CreationDate).ToUnixTimeMilliseconds()); path = [string]$_.ExecutablePath;",
    "  rssBytes = [long]$_.WorkingSetSize } })",
    "ConvertTo-Json -InputObject $records -Compress",
  ].join("\n");
  const result = run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], { encoding: "utf8", timeout: Math.min(20_000,timeoutMs), maxBuffer: pids===null?32*1024*1024:1024*1024 });
  if (result.error || result.status !== 0 || !result.stdout.trim()) return null;
  let parsed;
  try { parsed=JSON.parse(result.stdout.trim()); } catch { return null; }
  if (!Array.isArray(parsed) && (!parsed || typeof parsed!=="object")) return null;
  const rows = new Map();
  for (const row of Array.isArray(parsed) ? parsed : [parsed]) {
    if (![row?.pid, row?.ppid, row?.birthMs, row?.rssBytes].every(Number.isSafeInteger) || row.pid<=0 || row.ppid<0 || row.birthMs<0 || row.rssBytes<0 || typeof row.path !== "string" || rows.has(row.pid) || (pids!==null&&!pids.includes(row.pid))) {
      if(pids!==null)return null;
      continue;
    }
    rows.set(row.pid, { pid: row.pid, ppid: row.ppid, birthMs: row.birthMs, path: row.path.toLowerCase(), rssBytes: row.rssBytes });
  }
  return rows;
}

// Aggregate RSS of the authenticated owned tree on Windows — same quantity the
// POSIX guard sums, so the 2 GiB bound means the same thing on both platforms.
export function sumWindowsOwnedRss(rows, identity) {
  const tree = ownedWindowsTree(rows, identity);
  if (!tree.authenticated) return null;
  return tree.owned.reduce((total, entry) => total + (rows.get(entry.pid).rssBytes ?? 0), 0);
}

export function killWindowsPid(pid) {
  const killed = spawnSync("taskkill", ["/F", "/PID", String(pid)], { timeout: 10_000 });
  return killed.error ? `error: ${killed.error.message}` : killed.status;
}

// Authenticated descendants of a live, identity-matching root. As in the
// terminal supervisor's observeOwned, every edge must use the CURRENT immediate
// parent's birth: CIM can retain an old parent's PID after that PID is reused.
export function ownedWindowsTree(rows, identity) {
  if (!(rows instanceof Map) || !identity) return { authenticated: false, owned: [] };
  const root = rows.get(identity.pid);
  if (!root || !Number.isSafeInteger(root.birthMs) || root.birthMs !== identity.birthMs || root.path !== identity.path) return { authenticated: false, owned: [] };
  const owned = new Map([[identity.pid, 0]]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of rows.values()) {
      if (owned.has(row.pid) || !owned.has(row.ppid)) continue;
      const parent = rows.get(row.ppid);
      if (!Number.isSafeInteger(row.birthMs) || row.birthMs < parent.birthMs) continue;
      owned.set(row.pid, owned.get(row.ppid) + 1);
      changed = true;
    }
  }
  return { authenticated: true, owned: [...owned].map(([pid, depth]) => ({ pid, depth, birthMs: rows.get(pid).birthMs })) };
}

// The cleanup decision, pure so tests can prove it: with the root gone, only
// pids whose CURRENT row still matches the birth we observed while the root was
// alive may be signalled. Stale entries (exited or PID-reused) are dropped.
export function planAuthenticatedCleanup(rows, identity, knownDescendants, rootAlive) {
  if (!(rows instanceof Map)) throw new Error("cleanup planning process telemetry unavailable");
  const targets = [];
  for (const [pid, birthMs] of knownDescendants) {
    const row = rows.get(pid);
    if (row && row.birthMs === birthMs) targets.push(pid);
  }
  const rootRow = rows.get(identity.pid);
  if (rootAlive && rootRow && rootRow.birthMs === identity.birthMs && rootRow.path === identity.path) targets.push(identity.pid);
  return { targets: [...new Set(targets)] };
}

export async function executeOwnedChild(root, args, log, limitMs, state, supervision = {}) {
  const { query = queryWindowsProcesses, kill = killWindowsPid, spawnChild = spawn, sampleMs = 1000, graceMs = 10_000 } = supervision;
  const { openSync, closeSync, writeSync } = await import("node:fs");
  const fd = openSync(log, "w");
  const started = performance.now();
  const child = spawnChild(args[0], args.slice(1), { cwd: root, env: process.env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  let closed = false, exit = null, failure = null, logOpen = true;
  const completion = new Promise((resolve) => {
    child.on("error", (error) => { failure ??= error.message; });
    child.on("close", (code, signal) => { closed = true; exit = code ?? (signal ? 1 : 0); resolve(); });
  });
  for (const stream of [child.stdout, child.stderr]) stream.on("data", (bytes) => { if (logOpen) { writeSync(fd, bytes); process.stdout.write(bytes); } });
  state.child = Number.isInteger(child.pid) ? child : null;
  let identity = null;
  try {
    const rows = query(), row = rows?.get(child.pid);
    if (!row) throw new Error("spawned child is absent from the process table");
    identity = { pid: row.pid, birthMs: row.birthMs, path: row.path };
  } catch (error) {
    failure ??= `owned child identity unavailable: ${error.message}`;
  }
  state.identity = identity;
  const known = new Map();
  let sampled = 0;
  const observe = () => {
    const rows = query(), tree = ownedWindowsTree(rows, identity);
    if (tree.authenticated) for (const entry of tree.owned) if (entry.pid !== identity.pid) known.set(entry.pid, entry.birthMs);

  };
  try {
    if (!identity) throw new Error(failure);
    if (state.cancelled || performance.now() >= started + limitMs) throw new Error("managed deadline/cancellation before preparation release");
    while (!closed) {
      if (failure) break;
      if (performance.now() - sampled >= sampleMs) { sampled = performance.now(); observe(); }
      if (state.cancelled || performance.now() >= started + limitMs) { failure ??= "managed deadline/cancellation"; break; }
      await Promise.race([completion, delay(100)]);
    }
  } catch (error) {
    failure ??= error.message;
  } finally {
    // Root only through Node's own child handle; descendants only by re-authenticated pid+birth.
    const killed = [], unproven = [];
    try {
      // Node's own handle remains ours even if CIM cannot authenticate a birth.
      // Kill/reap that handle before reporting the still-unproven tree cleanup.
      if (!identity && !closed) {
        try { child.kill("SIGKILL"); killed.push(`${child.pid} (own handle)`); } catch { unproven.push(child.pid); }
        await Promise.race([completion, delay(graceMs)]);
      }
      if (!identity) throw new Error("owned child identity unavailable for cleanup");
      if (!closed) { try { child.kill("SIGKILL"); killed.push(`${child.pid} (own handle)`); } catch { unproven.push(child.pid); } }
      const plan = planAuthenticatedCleanup(query(), identity, known, !closed);
      for (const pid of plan.targets) {
        if (pid === identity.pid) continue; // The root is terminated only through its own child handle.
        const status = kill(pid);
        if (status === 0) killed.push(pid); else unproven.push(pid);
      }
      const remaining = query();
      if (!(remaining instanceof Map)) throw new Error("post-termination process telemetry unavailable");
      // A missing sample is NOT an empty table. Prove release of every known
      // identity, not merely targets that happened to appear in the kill plan.
      for (const [pid, birth] of new Map([[identity.pid, identity.birthMs], ...known])) {
        const row = remaining.get(pid);
        if (row && row.birthMs === birth) unproven.push(pid);
      }
      if (unproven.length) { state.unsafe = true; failure = `${failure ?? "owned child cleanup failed"}; unproven cleanup: ${JSON.stringify(unproven)}`; }
    } catch (error) {
      state.unsafe = true;
      unproven.push(error.message);
      failure = `${failure ?? "owned child cleanup failed"}; unproven cleanup: ${error.message}`;
    }
    if (!closed) await Promise.race([completion, delay(graceMs)]);
    if (!closed) { state.unsafe = true; failure = `${failure ?? "owned child cleanup failed"}; owned child outlived the managed bound`; }
    state.cleanup = { supervisor: "authenticated per-pid termination (own-handle root, pid+birth re-authenticated descendants; never bare taskkill /T)", killed, unproven, released: !state.unsafe };
    logOpen = false;
    closeSync(fd);
  }
  if (failure) fail(failure);
  return { exit, output: readFileSync(log, "utf8") };
}

// ---- external RSS guard (macOS and Windows) ----------------------------------------------------

export function sampleRssRows() {
  const result = spawnSync("ps", ["-axo", "pid=,ppid=,uid=,rss="], { encoding: "utf8", env: { ...process.env, LC_ALL: "C" }, timeout: 3000, maxBuffer: 8 * 1024 * 1024 });
  if (result.error || result.status !== 0) return null;
  const rows = new Map();
  for (const line of result.stdout.split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts.length !== 4) continue;
    const [pid, ppid, uid, rss] = parts.map(Number);
    if (![pid, ppid, uid, rss].every(Number.isSafeInteger)) continue;
    rows.set(pid, { pid, ppid, uid, rssBytes: rss * 1024 });
  }
  return rows;
}

export function sumOwnedRss(rows, rootPid) {
  if (!rows || !Number.isSafeInteger(rootPid) || !rows.has(rootPid)) return null;
  const owned = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of rows.values()) if (!owned.has(row.pid) && owned.has(row.ppid)) { owned.add(row.pid); changed = true; }
  }
  return [...owned].reduce((total, pid) => total + rows.get(pid).rssBytes, 0);
}

// Fail-closed guard: while the owned child is LIVE, a sampler exception or a
// missing sample CANCELS the case (never silently passes), and the first and
// peak observed aggregates are recorded so a receipt never claims a bound was
// held without samples actually taken.
export function startRssGuard(state, { sample = defaultGuardSample, capBytes, intervalMs = 1000 } = {}) {
  if (!Number.isSafeInteger(capBytes) || capBytes <= 0) fail("the external RSS guard requires the job-stated aggregate cap");
  const timer = setInterval(() => {
    const child = state.child;
    if (!child || !Number.isSafeInteger(child.pid) || child.exitCode !== null) return;
    let total = null;
    try { total = sample(state); } catch (error) { state.rssSamplingError ??= `sampler threw: ${error.message}`; }
    if (!Number.isSafeInteger(total) || total < 0) {
      if (!state.rssSamplingError) state.rssSamplingError = "aggregate owned-RSS sampler returned no sample while the case was live";
      state.cancelled = true;
      return;
    }
    if (state.rssFirstBytes === undefined) state.rssFirstBytes = total;
    state.rssPeakBytes = Math.max(state.rssPeakBytes ?? 0, total);
    if (total > capBytes && !state.cancelled) {
      state.cancelled = true;
      state.rssExceededBytes = total;
      state.rssCapBytes = capBytes;
    }
  }, intervalMs);
  return () => clearInterval(timer);
}

// Signals propagate to the ACTIVE command's state so the supervisor's cleanup
// reaps the owned tree before the receipt is written — same pattern as the
// closed assertion driver, and the handler never signals anything itself.
const defaultInstallSignals = (cancel) => {
  process.on("SIGINT", cancel);
  process.on("SIGTERM", cancel);
  return () => { process.off("SIGINT", cancel); process.off("SIGTERM", cancel); };
};

// Platform-split aggregate owned-RSS sampling: POSIX via ps ppid chains from the
// live child, Windows via the authenticated Win32_Process identity tree.
const defaultGuardSample = (state) =>
  process.platform === "win32" ? sumWindowsOwnedRss(queryWindowsProcesses(), state.identity) : sumOwnedRss(sampleRssRows(), state.child?.pid);

const noGuard = () => () => undefined;
const defaultExecute = (root, args, log, limitMs, state) =>
  isPreparationCommand(root, args) ? executePreparation(root, args, log, limitMs, state)
    : process.platform === "win32" ? executeOwnedChild(root, args, log, limitMs, state) : executeCpuCommand(root, args, log, limitMs, state);

function store(file, value) {
  mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`);
  renameSync(temporary, file);
}

export async function runValidation(options) {
  const {
    root, output, backend,
    env = process.env,
    platform = process.platform,
    inspect = defaultInspect,
    execute = defaultExecute,
    statExecutable = defaultStatExecutable,
    bounds = BOUNDS,
    toolchain = null,
    rssCapBytes = null,
    rssGuard = platform === "darwin" || platform === "win32" ? startRssGuard : noGuard,
    installSignals = defaultInstallSignals,
  } = options;
  let capBytes = rssCapBytes;
  const receipt = {
    schemaVersion: 1,
    backend,
    cpuOnly: true,
    node: process.version,
    platform,
    runnerOsArch: `${env.RUNNER_OS ?? platform}/${env.RUNNER_ARCH ?? process.arch}`,
    toolchain,
    supervisor: {
      mode: platform === "win32" ? "authenticated owned-pid termination" : "ownedCpuTree (reauthenticated owned pids; never process groups or broad pkill)",
      preparationScope: "root-and-stdio completion only; no descendant release claim",
      ownership: "only processes this driver spawned and their observed descendants",
      caseBoundMs: bounds.caseMs,
      rssGuardBytes: null,
    },
    environment: Object.fromEntries(ENVIRONMENT_WHITELIST.filter((key) => env[key] !== undefined).map((key) => [key, env[key]])),
    cancellation: null,
    failures: [],
  };
  const record = (failure) => { receipt.failures.push(failure); console.error(`moss-native validation: ${failure}`); };
  mkdirSync(output, { recursive: true });
  // Signals mark the ACTIVE command cancelled so the supervisor reaps its owned
  // tree before the receipt is written; they never signal anything directly.
  const control = { interrupted: false };
  const active = { state: null };
  const removeSignals = installSignals(() => { control.interrupted = true; if (active.state) active.state.cancelled = true; });
  let executable = null;
  try {
    receipt.source = null;
    try {
      requireManualContext(env);
      if ((backend === "mlx" && platform !== "darwin") || (backend === "candle" && platform !== "win32"))
        fail(`backend ${backend} does not match the actual runner platform ${platform}`);
      if (platform === "darwin" || platform === "win32") {
        const stated = statedRssCapBytes(env);
        if (!stated) fail("the job must state the external aggregate owned-RSS cap (MOSS_NATIVE_RSS_CAP_GB) before any case runs");
        capBytes = stated;
      }
      if (!env.MOSS_NATIVE_WORKFLOW_FILE) fail("the dispatching job must state its workflow file (MOSS_NATIVE_WORKFLOW_FILE)");
      const snapshot = inspect(root);
      receipt.source = {
        head: snapshot.head,
        githubSha: env.GITHUB_SHA ?? null,
        clean: snapshot.clean,
        cargoLockSha256: digest(readFileSync(path.join(root, "Cargo.lock"))),
        rustToolchainSha256: digest(readFileSync(path.join(root, "rust-toolchain.toml"))),
        testSourceSha256: digest(readFileSync(path.join(root, TEST_SOURCE))),
        inferencePin: lockedInferenceRevision(readFileSync(path.join(root, "Cargo.lock"), "utf8")),
        workflowFile: env.MOSS_NATIVE_WORKFLOW_FILE,
        workflowSha256: digest(readFileSync(path.join(root, env.MOSS_NATIVE_WORKFLOW_FILE))),
      };
      if (snapshot.head !== env.GITHUB_SHA) record(`source HEAD ${snapshot.head} is not the dispatched SHA ${env.GITHUB_SHA}`);
      if (!snapshot.clean) record("working tree is not clean");
      receipt.source.ignoredTests = ignoredSourceFacts(readFileSync(path.join(root, TEST_SOURCE), "utf8")).ignored;
    } catch (error) {
      record(error.message);
    }
    receipt.supervisor.rssGuardBytes = platform === "darwin" || platform === "win32" ? capBytes : null;

    if (!receipt.failures.length) {
      const compileLog = path.join(output, "compile-cargo.jsonl");
      const compileArgs = ["cargo", "test", "--locked", "-p", "sceneworks-rust-api", "--lib",
        ...(backend === "candle" ? ["--features", "backend-candle"] : []), "--no-run", "--message-format=json"];
      receipt.compile = { command: compileArgs, log: path.basename(compileLog), boundMs: bounds.compileMs };
      const state = { cancelled: false };
      active.state = state;
      const started = performance.now();
      try {
        const result = await execute(root, compileArgs, compileLog, bounds.compileMs, state);
        receipt.compile.exitCode = result.exit;
        receipt.compile.durationMs = performance.now() - started;
        // A nonzero compile exit is a failure even when the messages contain a
        // valid artifact line; compilation is never credited from a red build.
        if (result.exit !== 0) fail(`compilation exited ${result.exit}, not 0`);
        executable = selectLibTestExecutable(result.output, statExecutable);
        receipt.compile.executable = executable;
      } catch (error) {
        receipt.compile.exitCode ??= null;
        receipt.compile.durationMs = performance.now() - started;
        receipt.compile.error = error.message;
        receipt.compile.rawOutput = existsSync(compileLog) ? readFileSync(compileLog, "utf8").slice(0, 200_000) : null;
        record(`compilation failed: ${error.message}`);
      } finally {
        receipt.compile.completion = state.completion ?? null;
        active.state = null;
      }
    }

    if (executable && !receipt.failures.length) {
      // Fixture runtime only: hidden CUDA (inert where CUDA is absent), set AFTER
      // compilation so the toolchain compile is never affected.
      process.env.CUDA_VISIBLE_DEVICES = "-1";
      receipt.environment.CUDA_VISIBLE_DEVICES = process.env.CUDA_VISIBLE_DEVICES;
      const listLog = path.join(output, "test-list.txt");
      const ignoredListLog = path.join(output, "ignored-list.txt");
      receipt.lists = { unfiltered: path.basename(listLog), ignoredOnly: path.basename(ignoredListLog), boundMs: bounds.listMs };
      const unfilteredState = { cancelled: false }, ignoredOnlyState = { cancelled: false };
      try {
        active.state = unfilteredState;
        const unfiltered = await execute(root, [executable, "--list"], listLog, bounds.listMs, unfilteredState);
        if (unfiltered.exit !== 0) fail(`unfiltered --list exited ${unfiltered.exit}, not 0`);
        active.state = ignoredOnlyState;
        const ignoredOnly = await execute(root, [executable, "--ignored", "--list"], ignoredListLog, bounds.listMs, ignoredOnlyState);
        if (ignoredOnly.exit !== 0) fail(`--ignored --list exited ${ignoredOnly.exit}, not 0`);
        const discovery = checkDiscovery(parseTestList(unfiltered.output), parseTestList(ignoredOnly.output));
        receipt.lists.discovered = discovery.discovered;
        receipt.lists.ignored = discovery.ignored;
      } catch (error) {
        receipt.lists.error = error.message;
        for (const [key, log] of [["rawUnfiltered", listLog], ["rawIgnored", ignoredListLog]])
          if (existsSync(log)) receipt.lists[key] = readFileSync(log, "utf8").slice(0, 200_000);
        record(`discovery failed: ${error.message}`);
      } finally {
        receipt.lists.unfilteredCompletion = unfilteredState.completion ?? null;
        receipt.lists.ignoredOnlyCompletion = ignoredOnlyState.completion ?? null;
        active.state = null;
      }
    }

    receipt.cases = [];
    if (executable && !receipt.failures.length) {
      let halted = null;
      for (const test of EXPECTED_TESTS) {
        if (halted) {
          // An unsafe cleanup or a signal stops further attempts: the remaining
          // cases are recorded BLOCKED, never as passes.
          receipt.cases.push({ test, attempted: false, blocked: true, ok: false, reason: halted });
          continue;
        }
        const caseLog = path.join(output, `case-${test.slice(MOSS_PREFIX.length)}.log`);
        const command = [executable, test, "--ignored", "--exact", "--nocapture", "--test-threads=1"];
        const state = { cancelled: false };
        const entry = { test, command, log: path.basename(caseLog), boundMs: bounds.caseMs };
        const stopGuard = rssGuard(state, { capBytes });
        const started = performance.now();
        active.state = state;
        try {
          const result = await execute(root, command, caseLog, bounds.caseMs, state);
          entry.durationMs = performance.now() - started;
          entry.exitCode = result.exit;
          if (state.cancelled || state.unsafe || state.unsafeToRestore) fail("case cancellation or unproven resource release cannot count as a pass");
          if (!Number.isSafeInteger(state.rssFirstBytes) || state.rssFirstBytes <= 0 || !Number.isSafeInteger(state.rssPeakBytes) || state.rssPeakBytes > capBytes) fail("case has no positive bounded aggregate owned-RSS observation");
          entry.counts = parseTestResult(result.output, result.exit, test, "green");
          entry.ok = true;
        } catch (error) {
          entry.durationMs = performance.now() - started;
          entry.ok = false;
          entry.error = error.message;
          if (existsSync(caseLog)) entry.rawOutput = readFileSync(caseLog, "utf8").slice(0, 200_000);
          record(`case ${test} failed: ${error.message}`);
        } finally {
          active.state = null;
          stopGuard();
          entry.cancelled = Boolean(state.cancelled);
          if (state.rssFirstBytes !== undefined) { entry.rssFirstBytes = state.rssFirstBytes; entry.rssPeakBytes = state.rssPeakBytes; }
          if (state.rssSamplingError) entry.rssSamplingError = state.rssSamplingError;
          if (state.rssExceededBytes) { entry.rssExceededBytes = state.rssExceededBytes; entry.rssCapBytes = state.rssCapBytes; }
          entry.cleanup = state.cleanup ?? null;
          if (state.cancelled) entry.cancellation = {
            requested: true,
            released: !state.unsafe && !state.unsafeToRestore,
            source: state.rssExceededBytes ? "rss over cap" : state.rssSamplingError ? "rss sampler failure" : control.interrupted ? "signal" : "deadline",
          };
          if (state.unsafe || state.unsafeToRestore) {
            halted = `case ${test} left cleanup unproven; refusing to start further cases`;
            record(halted);
          } else if (control.interrupted) {
            halted = "interrupted by signal; refusing to start further cases";
            record(halted);
          }
        }
        receipt.cases.push(entry);
      }
    }
  } catch (error) {
    record(error.message);
  } finally {
    removeSignals();
    if (control.interrupted) receipt.cancellation = { requested: true, source: "signal" };
    receipt.ok = !receipt.failures.length && (receipt.cases?.length ?? 0) === EXPECTED_TESTS.length && receipt.cases.every((entry) => entry.ok);
    receipt.status = receipt.ok ? "passed" : "failed";
    store(path.join(output, "receipt.json"), receipt);
  }
  return { ok: receipt.ok, receipt };
}

const runningAsScript = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (runningAsScript) {
  try {
    const options = parseArgs(process.argv.slice(2));
    const probe = spawnSync("rustc", ["-vV"], { encoding: "utf8" });
    const result = await runValidation({ ...options, toolchain: probe.status === 0 ? probe.stdout.trim() : "rustc -vV unavailable" });
    process.exitCode = result.ok ? 0 : 1;
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
