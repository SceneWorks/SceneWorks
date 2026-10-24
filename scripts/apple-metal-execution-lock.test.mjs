// CPU-only metadata, subprocess and POSIX-flock tests. No model, GPU or production lock file.
import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createHash } from "node:crypto";
import { readFile, mkdtemp, writeFile, rm, access, realpath } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import { createMetalExecutionEntry, metalExecutionWaitSeconds, METAL_LOCK_FD, METAL_LOCK_PATH, METAL_LOCK_PROTOCOL } from "./lib/apple-metal-execution-lock.mjs";

const helper = fileURLToPath(new URL("./ci/metal_execution_lock.py", import.meta.url));
const python = process.env.PYTHON3_12 ?? "python3.12";
const sourceHash = "9e1803828a6b1af1c67f96bb644b612e81d44079501e795cfeed3708d3d74334";
const inode = 9007199254740993n;
const info = (ino = inode) => ({ dev: 7n, ino, nlink: 1n, isFile: () => true });
const record = `METAL_EXECUTION_LEASE {"protocol":"${METAL_LOCK_PROTOCOL}","path":"${METAL_LOCK_PATH}","borrowed":true,"device":7,"inode":${inode}}\n`;

test("both repositories use the identical frozen fixed-namespace helper", async () => {
  const bytes = await readFile(helper);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), sourceHash);
  assert.match(bytes.toString(), /CANONICAL_PATH = Path\('\/Users\/Shared\/SceneWorks\/gpu-locks\/apple-metal\.lock'\)/);
  assert.doesNotMatch(bytes.toString(), /add_argument\(['"]--(?:path|lock-file)/);
});

test("wait is finite, explicit at the helper, and never a queue-size/model-runtime cap", () => {
  assert.equal(metalExecutionWaitSeconds({}, 73), 73);
  assert.equal(metalExecutionWaitSeconds({ SCENEWORKS_METAL_EXECUTION_WAIT_SECONDS: "0.25" }, 73), 0.25);
  for (const value of ["", "0", "-1", "NaN", "Infinity"]) assert.throws(() => metalExecutionWaitSeconds({ SCENEWORKS_METAL_EXECUTION_WAIT_SECONDS: value }), /finite and positive/);
});

test("non-Mac paths and original spawn options are untouched", async () => {
  const entry = createMetalExecutionEntry({ platform: "win32", environment: { [METAL_LOCK_FD]: "forged" }, spawnImpl: () => assert.fail("no Python on Windows"), fstat: () => assert.fail("no lock query") });
  assert.equal(await entry.enter(2), false);
  const options = { stdio: ["ignore", "pipe", "pipe"], env: { MODEL_OPTION: "unchanged" } };
  assert.equal(entry.childOptions(options), options);
});

test("top-level launcher preserves all argv/env/status and defers only through its owned supervisor", async () => {
  const processObject = new EventEmitter();
  const child = new EventEmitter(); const signals = []; child.kill = (signal) => signals.push(signal);
  const environment = { MODEL: "same model", TIER: "q4" }, argv = ["/node", "/driver.mjs", "--prompt", "a b;not-a-shell", "--steps", "37"];
  let called;
  const entry = createMetalExecutionEntry({ platform: "darwin", environment, argv, processObject, spawnImpl: (...args) => { called = args; return child; } });
  const done = entry.enter(47);
  assert.deepEqual(called[1], ["-I", helper, "--wait-seconds", "47", "--", ...argv]);
  assert.equal(called[2].env, environment); assert.equal(called[2].stdio, "inherit");
  processObject.emit("SIGTERM"); assert.deepEqual(signals, ["SIGTERM"]);
  let finished = false; done.then(() => { finished = true; }); await Promise.resolve(); assert.equal(finished, false);
  child.emit("close", 143, null); assert.equal(await done, true); assert.equal(processObject.exitCode, 143);
  assert.equal(processObject.listenerCount("SIGTERM"), 0); assert.equal(processObject.listenerCount("SIGINT"), 0);
});

test("reentry requires actual helper flock proof and preserves exact 64-bit inode identity", async () => {
  const environment = { [METAL_LOCK_FD]: "7", PRIVATE_TOKEN: "do-not-revive" }; const calls = [];
  const entry = createMetalExecutionEntry({ platform: "darwin", environment, fstat: () => info(), stat: () => info(), spawnSyncImpl: (...args) => { calls.push(args); return { status: 0, stderr: record }; }, spawnImpl: () => assert.fail("never acquire a second lease") });
  assert.equal(await entry.enter(19), false); assert.equal(calls.length, 1);
  assert.equal(calls[0][2].stdio[3], 7); assert.equal(calls[0][2].env[METAL_LOCK_FD], "3");
  const child = entry.childOptions({ stdio: ["ignore", 11, 12, "pipe"], env: { MODEL: "same", ORIGINAL_CONTROL: "kept" } });
  assert.deepEqual(child.stdio, ["ignore", 11, 12, "pipe", 7]);
  assert.deepEqual(child.env, { MODEL: "same", ORIGINAL_CONTROL: "kept", [METAL_LOCK_FD]: "4" });
  assert.equal(calls.length, 2, "each handoff revalidates the same borrowed kernel lease without creating another owner");
  delete environment[METAL_LOCK_FD]; assert.throws(() => entry.childOptions({}), /descriptor was lost/);
});

test("forged/lost/inode-swapped lease and helper failures refuse without fallback", async () => {
  for (const raw of ["token", "2", "99999999999999999999999"]) {
    await assert.rejects(createMetalExecutionEntry({ platform: "darwin", environment: { [METAL_LOCK_FD]: raw } }).enter(1), /invalid inherited/);
  }
  await assert.rejects(createMetalExecutionEntry({ platform: "darwin", environment: { [METAL_LOCK_FD]: "7" }, fstat: () => info(), stat: () => info(inode + 1n), spawnImpl: () => assert.fail("no fallback") }).enter(1), /canonical inode/);
  await assert.rejects(createMetalExecutionEntry({ platform: "darwin", environment: { [METAL_LOCK_FD]: "7" }, fstat: () => info(), stat: () => info(), spawnSyncImpl: () => ({ status: 1, stderr: "held by another process" }), spawnImpl: () => assert.fail("no fallback") }).enter(1), /lease rejected/);
  await assert.rejects(createMetalExecutionEntry({ platform: "darwin", environment: { [METAL_LOCK_FD]: "7" }, fstat: () => info(), stat: () => info(), spawnSyncImpl: () => ({ status: 0, stderr: record.replace(String(inode), String(inode + 1n)) }) }).enter(1), /no matching kernel ownership/);
});

test("frozen canonical CPU contention/security/descendant suite", { skip: process.platform === "win32" }, () => {
  const suite = fileURLToPath(new URL("./tests/test_metal_execution_lock.py", import.meta.url));
  const result = spawnSync(python, ["-I", suite], { encoding: "utf8", timeout: 15000 });
  assert.equal(result.status, 0, `${result.error ?? ""}\n${result.stdout}\n${result.stderr}`);
  assert.match(result.stderr, /Ran [1-9][0-9]* tests/); assert.match(result.stderr, /\bOK\b/);
});

test("real CPU supervisor defers TERM and retains the same lease until its owned grandchild exits", { skip: process.platform === "win32" }, async () => {
  const temporary = await realpath(await mkdtemp(path.join(tmpdir(), "sceneworks-metal-cpu-")));
  const lock = path.join(temporary, "lease.lock"), ready = path.join(temporary, "ready"), release = path.join(temporary, "release");
  const grandchild = "import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text('ready');\nwhile not pathlib.Path(sys.argv[2]).exists():time.sleep(0.01)";
  const root = "import subprocess,sys;subprocess.Popen([sys.executable,'-I','-c',sys.argv[1],sys.argv[2],sys.argv[3]]);sys.exit(7)";
  const load = "import importlib.util,sys;from pathlib import Path;spec=importlib.util.spec_from_file_location('lease_cpu',sys.argv[1]);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)\n";
  const wrapper = load + "with m._acquire(Path(sys.argv[2]),2) as lease:\n print('HELD',flush=True)\n code=m.run_owned([sys.executable,'-I','-c',sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]],lease)\n raise SystemExit(code)\n";
  const child = spawn(python, ["-I", "-c", wrapper, helper, lock, root, grandchild, ready, release], { stdio: ["ignore", "pipe", "pipe"] });
  let stderr = ""; child.stderr.on("data", (chunk) => { stderr += chunk; });
  const done = new Promise((resolve, reject) => { child.once("error", reject); child.once("close", (code) => resolve(code)); });
  const waitFor = async (file) => { const end = Date.now() + 5000; while (Date.now() < end) { if (await access(file).then(() => true, () => false)) return; await new Promise((resolve) => setTimeout(resolve, 10)); } assert.fail(`CPU fixture did not create ${file}: ${stderr}`); };
  try {
    await waitFor(ready); child.kill("SIGTERM");
    const competitor = load + "try:\n with m._acquire(Path(sys.argv[2]),0.08):pass\nexcept TimeoutError:raise SystemExit(17)\n";
    const blocked = spawnSync(python, ["-I", "-c", competitor, helper, lock], { encoding: "utf8", timeout: 5000 });
    assert.equal(blocked.status, 17, blocked.stderr); assert.equal(child.exitCode, null);
    await writeFile(release, "done"); assert.equal(await done, 143, stderr);
    const acquired = spawnSync(python, ["-I", "-c", load + "with m._acquire(Path(sys.argv[2]),1):pass\n", helper, lock], { encoding: "utf8", timeout: 5000 });
    assert.equal(acquired.status, 0, acquired.stderr);
  } finally {
    await writeFile(release, "done").catch(() => {}); await done;
    await rm(temporary, { recursive: true, force: true });
  }
});
