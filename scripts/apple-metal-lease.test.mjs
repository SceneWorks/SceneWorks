// CPU-only real OFD ownership/handoff tests. No production namespace, model or GPU.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, writeFile, rm, access, realpath } from "node:fs/promises";
import { openSync, closeSync, fstatSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import { METAL_LOCK_FD, createMetalExecutionEntry } from "./lib/apple-metal-execution-lock.mjs";
import { createMacMetalLeaseFactory } from "./lib/apple-metal-lease.mjs";
import { execFileWithFds } from "./lib/exec-file-with-fds.mjs";
import { ownedTupleCommands, runOwnedTuple } from "./starvector-terminal-owned-tuple.mjs";
const helper = fileURLToPath(new URL("./ci/metal_execution_lock.py", import.meta.url));
const python = process.env.PYTHON3_12 ?? "python3.12";

async function fdLeaseFixture() {
  const directory = await realpath(await mkdtemp(path.join(tmpdir(), "sceneworks-node-metal-ofd-")));
  const canonicalPath = path.join(directory, "lease.lock");
  await writeFile(canonicalPath, "");
  const productionBridge = fileURLToPath(new URL("./ci/metal_execution_node_bridge.py", import.meta.url));
  const bridge = path.join(directory, "private-test-bridge.py");
  // Test-only namespace seam: actual bridge/protocol bodies unchanged, real kernel flock.
  // The production bridge offers no path override or fallback.
  await writeFile(bridge, `import importlib.util,sys\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location('actual_node_bridge',${JSON.stringify(productionBridge)});m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)\nm.protocol.CANONICAL_PATH=Path(${JSON.stringify(canonicalPath)})\ntry:raise SystemExit(m.main())\nexcept (ValueError,OSError,TimeoutError) as e:print(str(e),file=sys.stderr);raise SystemExit(1)\n`);
  const executable = spawnSync(python, ["-I", "-c", "import sys;print(sys.executable)"], { encoding: "utf8" });
  assert.equal(executable.status, 0, executable.stderr);
  const pythonPath = executable.stdout.trim();
  const environment = { ...process.env }; delete environment[METAL_LOCK_FD];
  const acquire = (extra = {}) => createMacMetalLeaseFactory({ platform: "darwin", environment, python: pythonPath, canonicalPath, bridge, ...extra });
  const competitor = (seconds = 0.08) => spawnSync(pythonPath, ["-I", "-c", "import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('cpu_compete',sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)\ntry:\n with m._acquire(Path(sys.argv[2]),float(sys.argv[3])):pass\nexcept TimeoutError:raise SystemExit(17)\n", helper, canonicalPath, String(seconds)], { encoding: "utf8", timeout: 5000 });
  return { directory, canonicalPath, bridge, pythonPath, acquire, competitor };
}
async function fileReady(file) {
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    if (await access(file).then(() => true, () => false)) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.fail(`owned CPU child did not create ${file}`);
}

for (const kind of ["entry", "factory"]) {
  test(`borrowed ${kind} rejects same-FD same-inode unlocked OFD replacement before handoff`, { skip: process.platform === "win32" }, async () => {
    const f = await fdLeaseFixture(); let owner, borrowed, fd, replacement, done;
    const ready = path.join(f.directory, "borrowed-ready"), release = path.join(f.directory, "borrowed-release");
    try {
      owner = await f.acquire({ marker: () => 42, open: (...args) => { fd = openSync(...args); return fd; } })(2);
      const environment = { ...process.env, [METAL_LOCK_FD]: String(fd) };
      if (kind === "entry") {
        const privateHelper = path.join(f.directory, "private-entry.py");
        await writeFile(privateHelper, `import importlib.util,sys\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location('actual_entry_protocol',${JSON.stringify(helper)});m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)\nm.CANONICAL_PATH=Path(${JSON.stringify(f.canonicalPath)})\nraise SystemExit(m.main())\n`);
        borrowed = createMetalExecutionEntry({ platform: "darwin", environment, python: f.pythonPath, helperPath: privateHelper, canonicalPath: f.canonicalPath });
        assert.equal(await borrowed.enter(2), false);
      } else borrowed = await f.acquire({ environment })(2);
      borrowed.childOptions({ stdio: "ignore", env: process.env });
      const code = "import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text('held');\nwhile not pathlib.Path(sys.argv[2]).exists():time.sleep(0.01)";
      const child = spawn(f.pythonPath, ["-I", "-c", code, ready, release], owner.childOptions({ stdio: "ignore", env: process.env }));
      done = new Promise((resolve, reject) => { child.once("error", reject); child.once("close", resolve); });
      await fileReady(ready);
      const before = fstatSync(fd);
      closeSync(fd);
      replacement = openSync(f.canonicalPath, "r+");
      assert.equal(replacement, fd, "fault reuses the exact descriptor number");
      assert.equal(fstatSync(replacement).ino, before.ino, "fault retains the canonical inode, not the original locked OFD");
      assert.equal(f.competitor().status, 17, "original OFD is still leased by the cooperating CPU child");
      assert.throws(() => borrowed.childOptions({ stdio: "ignore", env: process.env }), /kernel ownership|execution lease rejected/);
      if (kind === "factory") await assert.rejects(borrowed.close(), /kernel ownership/);
      assert.equal(fstatSync(replacement).ino, before.ino, "borrower never closes a foreign replacement");
      await assert.rejects(owner.close(), /lost\/reused/);
      assert.equal(fstatSync(replacement).ino, before.ino, "owned-cookie refusal also leaves the foreign descriptor open");
    } finally {
      if (replacement !== undefined) closeSync(replacement);
      else if (owner) await owner.close().catch(() => {});
      await writeFile(release, "CPU child drain").catch(() => {});
      if (done) await done;
      await rm(f.directory, { recursive: true, force: true });
    }
  });
}

test("default OFD cookie stays within a portable seek ceiling while real inode/flock proof remains mandatory", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let lease;
  const markers = [], ceiling = 0x7fffffff;
  try {
    lease = await f.acquire({ spawnSyncImpl: (program, args, options) => {
      const at = args.indexOf("--marker-offset");
      if (at >= 0) {
        const offset = Number(args[at + 1]); markers.push(offset);
        // Model a filesystem's real SEEK_SET ceiling, not a platform skip or fabricated proof.
        if (offset > ceiling) return { status: 1, signal: null, stdout: "", stderr: "OSError: [Errno 22] Invalid argument" };
      }
      return spawnSync(program, args, options);
    } })(2);
    assert.equal(markers.length, 1); assert.ok(Number.isSafeInteger(markers[0]) && markers[0] > 0 && markers[0] <= ceiling);
    assert.equal(f.competitor().status, 17, "the real private kernel flock must still exclude another OFD");
    assert.equal((await readFile(f.canonicalPath)).length, 0, "marking never writes or allocates lock-file contents");
    await lease.close(); lease = null; assert.equal(f.competitor().status, 0);
  } finally { if (lease) await lease.close(); await rm(f.directory, { recursive: true, force: true }); }
});
for (const offset of [1, 0x7fffffff]) test(`portable OFD boundary ${offset} retains real kernel ownership`, { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let lease;
  try {
    lease = await f.acquire({ marker: () => offset })(2); assert.equal(f.competitor().status, 17);
    await lease.close(); lease = null; assert.equal(f.competitor().status, 0);
    assert.equal((await readFile(f.canonicalPath)).length, 0);
  } finally { if (lease) await lease.close(); await rm(f.directory, { recursive: true, force: true }); }
});
test("old 48-bit/nonpositive cookies refuse before seek in both Node and bridge, with no fallback", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture();
  try {
    for (const offset of [0, 2 ** 48 - 1]) {
      let fd, marked = false;
      try {
        await assert.rejects(f.acquire({ marker: () => offset, open: (...args) => { fd = openSync(...args); return fd; },
          spawnSyncImpl: () => { marked = true; assert.fail("invalid cookie must not reach a seek"); } })(2), /invalid portable owned-description marker/);
        assert.equal(marked, false);
        const result = spawnSync(f.pythonPath, ["-I", f.bridge, "--mode", "inspect", "--wait-seconds", "1", "--marker-offset", String(offset)], {
          encoding: "utf8", stdio: ["ignore", "pipe", "pipe", fd], env: { ...process.env, [METAL_LOCK_FD]: "3" },
        });
        assert.equal(result.status, 1); assert.match(result.stderr, /invalid portable owned-description marker/);
      } finally { if (fd !== undefined) closeSync(fd); }
    }
  } finally { await rm(f.directory, { recursive: true, force: true }); }
});
test("failed seek retains bounded phase/status/errno diagnostics and never grants a lease", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let fd;
  try {
    await assert.rejects(f.acquire({ marker: () => 42, open: (...args) => { fd = openSync(...args); return fd; },
      spawnSyncImpl: () => ({ status: 1, signal: null, stdout: "", stderr: "OSError: [Errno 22] Invalid argument" }),
    })(2), (error) => /description was lost\/reused/.test(error.message)
      && error.message.includes('"phase":"mark"') && error.message.includes('"expectedOffset":42')
      && error.message.includes('"status":1') && error.message.includes("Errno 22"));
    assert.equal(f.competitor().status, 0, "a failed mark did not grant kernel ownership");
  } finally { if (fd !== undefined) closeSync(fd); await rm(f.directory, { recursive: true, force: true }); }
});

test("real Node lease survives bridge exit, start returns with actual worker PID, and stop/reload release correctly", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let done;
  const ready = path.join(f.directory, "ready"), release = path.join(f.directory, "release");
  try {
    const lease = await f.acquire()(2);
    assert.equal(f.competitor().status, 17, "Node holds the real kernel flock after metadata bridge exits");
    const args = ["-I", "-c", "import os,pathlib,sys,time;os.fstat(int(os.environ['SCENEWORKS_METAL_EXECUTION_LOCK_FD']));pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));\nwhile not pathlib.Path(sys.argv[2]).exists():time.sleep(0.01)", ready, release];
    const worker = spawn(f.pythonPath, args, lease.childOptions({ stdio: ["ignore", "pipe", "pipe"], env: process.env }));
    done = new Promise((resolve, reject) => { worker.once("error", reject); worker.once("close", (code) => resolve(code)); });
    const state = { schema_version: 1, worker_pid: worker.pid, started_at: new Date().toISOString() };
    await fileReady(ready);
    assert.equal(Number(await readFile(ready, "utf8")), state.worker_pid, "actual worker PID, not metadata supervisor");
    await lease.close(); await lease.close();
    assert.equal(worker.pid, state.worker_pid); assert.equal(worker.exitCode, null, "start returned while worker remains running");
    assert.equal(f.competitor().status, 17, "parent close must not LOCK_UN the actual worker");
    await writeFile(release, "stop/drain"); assert.equal(await done, 0);
    assert.equal(f.competitor().status, 0);
    const reload = await f.acquire()(2); assert.equal(f.competitor().status, 17); await reload.close();
    assert.equal(f.competitor().status, 0, "reload uses same namespace, releases on final close");
    assert.deepEqual(args.slice(-2), [ready, release]);
  } finally {
    await writeFile(release, "stop/drain").catch(() => {}); if (done) await done;
    await rm(f.directory, { recursive: true, force: true });
  }
});

test("cooperating grandchild retains lease after Node closes and actual worker root exits", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let done;
  const ready = path.join(f.directory, "grandchild-ready"), release = path.join(f.directory, "grandchild-release");
  try {
    const lease = await f.acquire()(2);
    const grand = "import os,pathlib,sys,time;os.fstat(int(os.environ['SCENEWORKS_METAL_EXECUTION_LOCK_FD']));pathlib.Path(sys.argv[1]).write_text('ready');\nwhile not pathlib.Path(sys.argv[2]).exists():time.sleep(0.01)";
    const root = "import os,subprocess,sys;fd=int(os.environ['SCENEWORKS_METAL_EXECUTION_LOCK_FD']);subprocess.Popen([sys.executable,'-I','-c',sys.argv[1],sys.argv[2],sys.argv[3]],pass_fds=(fd,));raise SystemExit(9)";
    const worker = spawn(f.pythonPath, ["-I", "-c", root, grand, ready, release], lease.childOptions({ stdio: "ignore", env: process.env }));
    done = new Promise((resolve, reject) => { worker.once("error", reject); worker.once("close", (code) => resolve(code)); });
    await fileReady(ready); assert.equal(await done, 9); await lease.close();
    assert.equal(f.competitor().status, 17, "actual cooperating grandchild retains kernel ownership");
    await writeFile(release, "drain");
    for (let n = 0; n < 30; n += 1) { if (f.competitor(0.02).status === 0) break; await new Promise((resolve) => setTimeout(resolve, 10)); }
    assert.equal(f.competitor().status, 0);
  } finally {
    await writeFile(release, "drain").catch(() => {}); if (done) await done;
    await rm(f.directory, { recursive: true, force: true });
  }
});

test("startup failure releases own reference; nested inherited lease never unlocks or closes owner", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let fd;
  try {
    const own = await f.acquire({ open: (...args) => { fd = openSync(...args); return fd; } })(2);
    const nested = await f.acquire({ environment: { ...process.env, [METAL_LOCK_FD]: String(fd) } })(2);
    await nested.close(); assert.doesNotThrow(() => fstatSync(fd)); assert.equal(f.competitor().status, 17);
    // Same cleanup as a failed native spawn/start/readiness before any model was launched.
    await own.close(); assert.equal(f.competitor().status, 0);
  } finally { await rm(f.directory, { recursive: true, force: true }); }
});

test("acquisition failure never closes a foreign same-inode descriptor reused while the bridge runs", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let fd, foreign;
  try {
    const acquire = f.acquire({
      open: (...args) => { fd = openSync(...args); return fd; },
      spawnImpl: (command, args, options) => {
        const changed = [...args]; changed[changed.indexOf("--mode") + 1] = "invalid-cpu-fixture-mode";
        const child = spawn(command, changed, options);
        closeSync(fd); foreign = openSync(f.canonicalPath, "r+"); assert.equal(foreign, fd);
        return child;
      },
    });
    await assert.rejects(acquire(2), /bridge failed/);
    assert.doesNotThrow(() => fstatSync(foreign), "failure cleanup cannot close another same-inode open description");
    assert.equal(f.competitor().status, 0);
  } finally { if (foreign !== undefined) closeSync(foreign); await rm(f.directory, { recursive: true, force: true }); }
});

test("release is idempotent and refuses a foreign reused FD even on SAME inode; no PID signals", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let fd, foreign;
  try {
    const lease = await f.acquire({ open: (...args) => { fd = openSync(...args); return fd; } })(2);
    closeSync(fd); foreign = openSync(f.canonicalPath, "r+"); assert.equal(foreign, fd);
    await assert.rejects(lease.close(), /description was lost\/reused/);
    assert.doesNotThrow(() => fstatSync(foreign), "must not close another description");
    await lease.close(); assert.doesNotThrow(() => fstatSync(foreign)); closeSync(foreign); foreign = undefined;
    const source = await readFile(new URL("./lib/apple-metal-lease.mjs", import.meta.url), "utf8");
    assert.doesNotMatch(source, /process\.kill\(|\.kill\(|taskkill|SIGKILL/);
  } finally { if (foreign !== undefined) closeSync(foreign); await rm(f.directory, { recursive: true, force: true }); }
});

test("actual start CLI exits while the same real worker PID retains the descriptor lease", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture();
  const statePath = path.join(f.directory, "state.json"), ready = path.join(f.directory, "ready-cli"), release = path.join(f.directory, "release-cli");
  const moduleUrl = new URL("./lib/apple-metal-lease.mjs", import.meta.url).href;
  const workerCode = "import os,pathlib,sys,time;os.fstat(int(os.environ['SCENEWORKS_METAL_EXECUTION_LOCK_FD']));pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));\nwhile not pathlib.Path(sys.argv[2]).exists():time.sleep(0.01)";
  const cli = `import {spawn} from 'node:child_process';import {writeFile} from 'node:fs/promises';import {createMacMetalLeaseFactory} from ${JSON.stringify(moduleUrl)};const env={...process.env};delete env.SCENEWORKS_METAL_EXECUTION_LOCK_FD;const lease=await createMacMetalLeaseFactory({platform:'darwin',environment:env,python:${JSON.stringify(f.pythonPath)},canonicalPath:${JSON.stringify(f.canonicalPath)},bridge:${JSON.stringify(f.bridge)}})(2);const child=spawn(${JSON.stringify(f.pythonPath)},['-I','-c',${JSON.stringify(workerCode)},${JSON.stringify(ready)},${JSON.stringify(release)}],lease.childOptions({stdio:'ignore',detached:true,env:process.env}));child.unref();await writeFile(${JSON.stringify(statePath)},JSON.stringify({schema_version:1,worker_pid:child.pid,started_at:new Date().toISOString()}));await lease.close();`;
  try {
    const result = await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, ["--input-type=module", "-e", cli], { stdio: ["ignore", "pipe", "pipe"] }); let error = "";
      child.stderr.on("data", (chunk) => { error += chunk; }); child.once("error", reject);
      child.once("close", (code) => code === 0 ? resolve(code) : reject(new Error(error)));
    });
    assert.equal(result, 0); await fileReady(ready);
    assert.equal(JSON.parse(await readFile(statePath, "utf8")).worker_pid, Number(await readFile(ready, "utf8")));
    assert.equal(f.competitor().status, 17, "lease survives actual Node CLI process exit");
  } finally {
    await writeFile(release, "stop/drain").catch(() => {});
    for (let n = 0; n < 50; n += 1) { if (f.competitor(0.02).status === 0) break; await new Promise((resolve) => setTimeout(resolve, 10)); }
    await rm(f.directory, { recursive: true, force: true });
  }
});

test("FD-aware exec preserves lease/argv/stdout/stderr; unleased calls retain original execFile path", { skip: process.platform === "win32" }, async () => {
  const f = await fdLeaseFixture(); let lease;
  try {
    lease = await f.acquire()(2);
    const result = await execFileWithFds(f.pythonPath, ["-I", "-c", "import os,sys;os.fstat(int(os.environ['SCENEWORKS_METAL_EXECUTION_LOCK_FD']));print(sys.argv[1]);print('diagnostic',file=sys.stderr)", "exact model option;not shell"], lease.childOptions({ env: process.env, maxBuffer: 65536 }));
    assert.equal(result.stdout, "exact model option;not shell\n"); assert.equal(result.stderr, "diagnostic\n");
    let called; const options = { env: { CUDA_VISIBLE_DEVICES: "same" }, timeout: 17 };
    assert.deepEqual(await execFileWithFds("unchanged.exe", ["--original"], options, async (...args) => { called = args; return { stdout: "old", stderr: "" }; }), { stdout: "old", stderr: "" });
    assert.deepEqual(called, ["unchanged.exe", ["--original"], options]);
  } finally { if (lease) await lease.close(); await rm(f.directory, { recursive: true, force: true }); }
});

function tupleFixture() {
  return { root: "/source", output: "/output", permanentPin: "original-pin", url: "http://127.0.0.1:17821", weightsRoot: "/weights", tuple: "candle-cuda:1b", campaignRunId: "original-run", env: { STARVECTOR_TERMINAL_INFERENCE_ROOT: "/inference", STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT: "/assets", STARVECTOR_TERMINAL_CASE_BUNDLE: "/output/case-bundle.json", STARVECTOR_TERMINAL_METRICS_ROOT: "/metrics", STARVECTOR_TERMINAL_LEASE_ROOT: "/logical-leases", STARVECTOR_TERMINAL_LEASE_HELPER: "/logical-helper", STARVECTOR_TERMINAL_PRODUCT_ROUTE_COMMAND: "/source/scripts/starvector-terminal-route.mjs" } };
}
test("closed tuple preserves original start/assets/bundle/producer/stop ordering and parameters", async () => {
  const options = tupleFixture(), planned = ownedTupleCommands(options), calls = [];
  await runOwnedTuple(options, { start: async (value) => calls.push(["start", value]), execute: async (file, args, env) => calls.push([path.basename(file), args, env]), stop: async (value) => calls.push(["stop", value]) });
  assert.equal(calls[0][0], "start"); assert.deepEqual(calls[0][1], planned.start);
  assert.deepEqual(calls.slice(1, -1).map(([name]) => name), ["starvector-terminal-assets.mjs", "starvector-terminal-case-bundle.mjs", "starvector-terminal-producer.mjs"]);
  assert.deepEqual(calls.slice(1, -1).map(([, args]) => args), planned.commands.map(([, args]) => args));
  assert.equal(calls.at(-1)[0], "stop"); assert.equal(calls.at(-1)[1], options.output);
});
test("closed tuple stops exactly once on start/producer failures and preserves primary failure", async () => {
  for (const failStart of [true, false]) {
    const primary = new Error(failStart ? "original start failure" : "original producer failure"); let stopped = 0;
    await assert.rejects(runOwnedTuple(tupleFixture(), { start: async () => { if (failStart) throw primary; }, execute: async () => { throw primary; }, stop: async () => { stopped += 1; throw new Error("cleanup failed"); } }), (error) => error === primary && error.message.includes("cleanup failed"));
    assert.equal(stopped, 1, "old stop operation is non-idempotent");
  }
});
