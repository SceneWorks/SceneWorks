import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const WATCHDOG = path.join(ROOT, "scripts/memory-calibration-watchdog.py");

async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), "sc19642-watchdog-"));
  const program = path.join(root, "tree.py");
  await writeFile(program, String.raw`import os, signal, subprocess, sys, time
mode, pid_file, telemetry_file, event_file = sys.argv[1:]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_DFL); time.sleep(60)" if mode == "complete" else "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
child = subprocess.Popen([sys.executable, "-c", child_code])
with open(pid_file, "w") as output:
    output.write(f"{os.getpid()}\n{child.pid}\n")
    output.flush()
if mode == "high":
    time.sleep(0.15)
    open(telemetry_file, "w").write("100\n")
elif mode == "lost":
    time.sleep(0.15)
    os.unlink(telemetry_file)
elif mode == "complete":
    time.sleep(0.3)
    child.terminate()
    child.wait()
    raise SystemExit(0)
elif mode == "root-exit":
    raise SystemExit(7)
elif mode == "event-failure":
    time.sleep(0.15)
    os.unlink(event_file)
    os.mkdir(event_file)
time.sleep(60)
`);
  return {
    program,
    pids: path.join(root, "pids"),
    telemetry: path.join(root, "telemetry"),
    events: path.join(root, "events"),
  };
}

async function run(mode, ceiling, telemetry = 1) {
  const files = await fixture();
  await writeFile(files.telemetry, `${telemetry}\n`);
  const started = Date.now();
  let status = 0;
  try {
    await execFileAsync("python3", [
      WATCHDOG,
      "--max-footprint-bytes", `${ceiling}`,
      "--sample-interval", "0.02",
      "--telemetry-timeout", "0.2",
      "--term-grace", "0.1",
      "--event-file", files.events,
      "--telemetry-file", files.telemetry,
      "--allow-synthetic-telemetry",
      "--", "python3", files.program, mode, files.pids, files.telemetry, files.events,
    ], { timeout: 10_000 });
  } catch (error) {
    status = error.code;
  }
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  return { status, events, pids, elapsed: Date.now() - started };
}

async function waitForJsonEvent(file, predicate) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const bytes = await readFile(file, "utf8").catch(() => "");
    const events = bytes.trim() ? bytes.trim().split("\n").map(JSON.parse) : [];
    const found = events.find(predicate);
    if (found) return found;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`event never appeared in ${file}`);
}

async function waitForFile(file) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await readFile(file, "utf8").catch(() => "")) return;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`file never appeared: ${file}`);
}

async function controlled(mode, action) {
  const files = await fixture();
  await writeFile(files.telemetry, "1\n");
  const child = spawn("python3", [
    WATCHDOG,
    "--max-footprint-bytes", "100",
    "--sample-interval", "0.02",
    "--telemetry-timeout", "0.2",
    "--term-grace", "0.1",
    "--event-file", files.events,
    "--telemetry-file", files.telemetry,
    "--allow-synthetic-telemetry",
    "--", "python3", files.program, mode, files.pids, files.telemetry, files.events,
  ], { stdio: "ignore" });
  const started = await waitForJsonEvent(files.events, (event) => event.event === "started");
  await waitForJsonEvent(files.events, (event) => event.event === "sample");
  await waitForFile(files.pids);
  await new Promise((resolve) => setTimeout(resolve, 60));
  await action(child, started);
  const status = await new Promise((resolve) => child.once("close", resolve));
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  return { status, pids, events };
}

function assertGone(pid) {
  assert.throws(() => process.kill(pid, 0), (error) => error.code === "ESRCH", `pid ${pid} survived`);
}

test("physical-footprint hard stop terminates the responsive owned group with no residue", async () => {
  const result = await run("high", 100);
  assert.equal(result.status, 97);
  assert.ok(result.elapsed < 5_000, `termination took ${result.elapsed}ms`);
  assert.ok(result.events.some((event) =>
    event.event === "hard_stop" && event.reason.includes("physical_footprint")));
  assert.equal(result.events.at(-1).event, "terminated");
  result.pids.forEach(assertGone);
});

test("loss of telemetry fails closed and terminates the owned group", async () => {
  const result = await run("lost", 100);
  assert.equal(result.status, 97);
  assert.ok(result.events.some((event) =>
    event.event === "hard_stop" && event.reason.includes("telemetry_lost")));
  result.pids.forEach(assertGone);
});

test("the ceiling comparison is mutation-sensitive at the exact boundary", async () => {
  const below = await run("complete", 100, 99);
  assert.equal(below.status, 0);
  const at = await run("high", 99, 98);
  assert.equal(at.status, 97);
  assert.ok(at.events.some((event) => event.reason?.includes("at_or_above_99")));
  at.pids.forEach(assertGone);
});

test("SIGINT and SIGTERM preserve shell status while cleaning the exact owned group", async () => {
  for (const [signalName, expected] of [["SIGINT", 130], ["SIGTERM", 143]]) {
    const result = await controlled("hold", async (watchdog) => watchdog.kill(signalName));
    assert.equal(result.status, expected, signalName);
    result.pids.forEach(assertGone);
  }
});

test("a signal delivered in the blocked spawn window is cleaned after the sentinel is anchored", async () => {
  const files = await fixture();
  const launchReady = `${files.pids}.launch-ready`;
  await writeFile(files.telemetry, "1\n");
  const watchdog = spawn("python3", [
    WATCHDOG, "--max-footprint-bytes", "100", "--sample-interval", "0.02",
    "--telemetry-timeout", "0.2", "--term-grace", "0.1",
    "--telemetry-file", files.telemetry, "--allow-synthetic-telemetry",
    "--synthetic-launch-ready-file", launchReady, "--synthetic-spawn-delay", "0.2",
    "--", "python3", files.program, "hold",
    files.pids, files.telemetry, files.events,
  ], { stdio: "ignore" });
  await waitForFile(launchReady);
  watchdog.kill("SIGTERM");
  const status = await new Promise((resolve) => watchdog.once("close", resolve));
  assert.equal(status, 143);
  const pgrep = await execFileAsync("pgrep", ["-f", files.program]).catch((error) => error);
  assert.equal((pgrep.stdout ?? "").trim(), "", `spawn-window signal leaked: ${pgrep.stdout}`);
});

test("an early-exiting command root cannot leak its TERM-resistant descendant", async () => {
  const result = await run("root-exit", 100);
  assert.equal(result.status, 7);
  result.pids.forEach(assertGone);
});

test("loss of the stable launch sentinel fails closed and removes retained descendants", async () => {
  const result = await controlled("hold", async (_watchdog, started) => {
    process.kill(started.pid, "SIGKILL");
  });
  assert.equal(result.status, 97);
  assert.ok(result.events.some((event) => event.reason?.includes("launch_sentinel_lost")));
  result.pids.forEach(assertGone);
});

test("event-log failure is a monitor failure and still leaves no owned residue", async () => {
  const files = await fixture();
  await writeFile(files.telemetry, "1\n");
  let status = 0;
  try {
    await execFileAsync("python3", [
      WATCHDOG, "--max-footprint-bytes", "100", "--sample-interval", "0.02",
      "--telemetry-timeout", "0.2", "--term-grace", "0.1",
      "--event-file", files.events, "--telemetry-file", files.telemetry,
      "--allow-synthetic-telemetry", "--", "python3", files.program, "event-failure",
      files.pids, files.telemetry, files.events,
    ], { timeout: 10_000 });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});

test("stale start identity is never treated as the live PID", async () => {
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
live = module.process_identity(os.getpid())
stale = module.Identity(live.pid, live.pgid, live.state, "Thu Jan  1 00:00:00 1970")
assert not module.identity_is_live(stale)
print("stale identity refused")
`]);
  assert.match(probe.stdout, /stale identity refused/);
});
