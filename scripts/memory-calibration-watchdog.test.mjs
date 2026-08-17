import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { validateWatchdogEventChain } from "./run-ltx-safety-canary.mjs";

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
child = None if mode == "delayed-child" else subprocess.Popen([sys.executable, "-c", child_code])
with open(pid_file, "w") as output:
    output.write(f"{os.getpid()}\n")
    if child:
        output.write(f"{child.pid}\n")
    output.flush()
if mode == "delayed-child":
    time.sleep(0.05)
    child = subprocess.Popen([sys.executable, "-c", child_code])
    with open(pid_file, "a") as output:
        output.write(f"{child.pid}\n")
        output.flush()
elif mode == "high":
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

async function run(
  mode, ceiling, telemetry = 1, maxRuntimeSeconds = null, sampleInterval = "0.02",
) {
  const files = await fixture();
  await writeFile(files.telemetry, `${telemetry}\n`);
  const started = Date.now();
  let status = 0;
  try {
    const args = [
      WATCHDOG,
      "--max-footprint-bytes", `${ceiling}`,
      "--sample-interval", sampleInterval,
      "--telemetry-timeout", "0.2",
      "--term-grace", "0.1",
      "--event-file", files.events,
      "--telemetry-file", files.telemetry,
      "--allow-synthetic-telemetry",
      "--", "python3", files.program, mode, files.pids, files.telemetry, files.events,
    ];
    if (maxRuntimeSeconds !== null) {
      args.splice(3, 0, "--max-runtime-seconds", `${maxRuntimeSeconds}`);
    }
    await execFileAsync("python3", args, { timeout: 10_000 });
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

async function runWithMockedProductionTelemetry(files, childCommand, options = {}) {
  const {
    telemetryTimeout = 0.5, actualHostMemory = 1000, requestedHostMemory = 1000,
    childAttestationTimeout = 1,
    requireProviderPhases = false,
    memoryFreePercent = 90, memoryFreeBytes = 900, swapFreeBytes = 900,
    pressureSamples = [[memoryFreePercent, memoryFreeBytes, swapFreeBytes]],
    pressureFailureAt = null,
    environment = process.env,
  } = options;
  const pressureFailureAtPython = pressureFailureAt === null ? "None" : String(pressureFailureAt);
  const launcher = `${files.program}.production-watchdog.py`;
  await writeFile(launcher, String.raw`import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
class Footprint:
    def sample(self, pids, timeout): return 1
class Pressure:
    def __init__(self, host_memory_bytes): self.index = 0
    def sample(self, timeout):
        if self.index == ${pressureFailureAtPython}:
            raise RuntimeError("injected startup pressure failure")
        samples = ${JSON.stringify(pressureSamples)}
        values = samples[min(self.index, len(samples) - 1)]
        self.index += 1
        return module.HostPressure(*values)
module.DarwinFootprintSampler = Footprint
module.DarwinHostPressureSampler = Pressure
module.DarwinHostPressureSampler.actual_host_memory_bytes = staticmethod(
    lambda timeout: ${actualHostMemory})
sys.argv = [${JSON.stringify(WATCHDOG)},
    "--max-footprint-bytes", "100", "--max-runtime-seconds", "2",
    "--host-memory-bytes", ${JSON.stringify(String(requestedHostMemory))},
    "--min-memory-free-bytes", "100",
    "--sample-interval", "0.02",
    "--telemetry-timeout", ${JSON.stringify(String(telemetryTimeout))},
    "--child-attestation-timeout", ${JSON.stringify(String(childAttestationTimeout))},
    "--term-grace", "0.1", "--event-file", ${JSON.stringify(files.events)},
    "--require-child-attestation", ${requireProviderPhases ? '"--require-provider-phases",' : ""}
    "--", *${JSON.stringify(childCommand)}]
raise SystemExit(module.guard(module.parse_args()))
`);
  return execFileAsync("python3", [launcher], { timeout: 10_000, env: environment });
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

test("a wall-time ceiling fails closed and removes the owned group", async () => {
  const result = await run("hold", 100, 1, 0.2, "2");
  assert.equal(result.status, 97);
  const started = result.events.find((event) => event.event === "started");
  const stopped = result.events.find((event) =>
    event.event === "hard_stop" && event.reason.includes("runtime_at_or_above_0.2s"));
  assert.ok(stopped);
  assert.ok(stopped.at - started.at < 1, "the uncapped two-second sample sleep won");
  result.pids.forEach(assertGone);
});

test("cleanup crossing the deadline cannot relabel an earlier telemetry failure", async () => {
  const files = await fixture();
  const launcher = `${files.program}.predeadline-failure.py`;
  await writeFile(launcher, String.raw`import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
state = {"samples": 0, "failed": False, "delayed": False}
class Footprint:
    def sample(self, pids, timeout):
        state["samples"] += 1
        if state["samples"] == 1: return 1
        state["failed"] = True
        raise RuntimeError("predeadline_source_failure")
original_refresh = module.OwnedGroup.refresh
def delayed_refresh(self):
    if state["failed"] and not state["delayed"]:
        state["delayed"] = True
        time.sleep(1.1)
    return original_refresh(self)
module.DarwinFootprintSampler = Footprint
module.OwnedGroup.refresh = delayed_refresh
sys.argv = [${JSON.stringify(WATCHDOG)},
    "--max-footprint-bytes", "100", "--max-runtime-seconds", "1",
    "--sample-interval", "0.02", "--telemetry-timeout", "0.2",
    "--term-grace", "0.1", "--event-file", ${JSON.stringify(files.events)},
    "--", "python3", ${JSON.stringify(files.program)}, "hold",
    ${JSON.stringify(files.pids)}, ${JSON.stringify(files.telemetry)},
    ${JSON.stringify(files.events)}]
raise SystemExit(module.guard(module.parse_args()))
`);
  let status = 0;
  try {
    await execFileAsync("python3", [launcher], { timeout: 10_000 });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  const stopped = events.find((event) => event.event === "hard_stop");
  assert.match(stopped.reason, /^telemetry_lost:RuntimeError:predeadline_source_failure$/);
  assert.doesNotMatch(stopped.reason, /runtime_at_or_above/);
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});

test("host memory and swap floors are enforced inside the owned-group watchdog", async () => {
  const files = await fixture();
  const pressure = `${files.pids}.pressure`;
  await writeFile(files.telemetry, "1\n");
  await writeFile(pressure, JSON.stringify({
    memoryFreePercent: 49,
    memoryFreeBytes: 99,
    swapFreeBytes: 100,
  }));
  let status = 0;
  try {
    await execFileAsync("python3", [
      WATCHDOG, "--max-footprint-bytes", "1000",
      "--host-memory-bytes", "1000", "--min-memory-free-bytes", "100",
      "--min-swap-free-bytes", "100", "--sample-interval", "0.02",
      "--telemetry-timeout", "0.2", "--term-grace", "0.1",
      "--event-file", files.events, "--telemetry-file", files.telemetry,
      "--host-pressure-file", pressure, "--allow-synthetic-telemetry",
      "--", "python3", files.program, "hold", files.pids, files.telemetry, files.events,
    ], { timeout: 10_000 });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) =>
    event.event === "hard_stop" && event.reason.includes("host_memory_free_below_100")));
  assert.equal(
    await readFile(files.pids, "utf8").catch(() => ""), "",
    "unsafe initial host pressure must refuse before the guarded command starts",
  );
});

test("the monitor samples before child release and requires ACK before allocation release", async () => {
  const files = await fixture();
  const attester = `${files.program}.attest.py`;
  const launched = `${files.pids}.launched`;
  await writeFile(attester, String.raw`import json, os, socket, sys
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line())
assert "minSwapFreeBytes" not in payload
assert "minInitialMemoryFreePercent" not in payload
nonce = payload["nonce"]
sock.sendall(f"ACK {nonce}\n".encode())
release = line()
assert release == f"GO {nonce}"
open(sys.argv[1], "w").write("released\n")
sock.sendall(f"DONE {nonce}\n".encode())
while True:
    message = line()
    if message == f"BYE {nonce}": break
    assert message == f"PING {nonce}"
`);
  await runWithMockedProductionTelemetry(files, ["python3", attester, launched], {
    swapFreeBytes: 0,
  });
  assert.equal(await readFile(launched, "utf8"), "released\n");
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  const attested = events.findIndex((event) => event.event === "child_attested");
  assert.ok(attested > 0);
  assert.ok(events.slice(0, attested).some((event) =>
    event.event === "sample" && event.phase === "before_child_release"));
  assert.ok(events.slice(0, attested).some((event) =>
    event.event === "sample" && event.phase === "child_attested_before_allocation"));
  assert.ok(events.some((event) => event.event === "child_completed"));
  assert.ok(events.some((event) => event.event === "sample" && event.swapFreeBytes === 0),
    "swap telemetry remains present without imposing an arbitrary free-capacity floor");
});

test("authenticated provider phases are exact, monotonic and bound to terminal evidence", async () => {
  const files = await fixture();
  const attester = `${files.program}.phases.py`;
  const phases = [
    "common_load", "primary_conditioning", "primary_denoise", "primary_decode",
    "lifecycle_warm_repeat", "lifecycle_cancel", "lifecycle_cancel_recovery",
    "lifecycle_error", "lifecycle_error_recovery", "cleanup",
  ];
  await writeFile(attester, String.raw`import json, os, socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
assert payload["providerPhaseProtocol"] == "sceneworks-provider-phase-v1"
assert payload["providerPhases"] == ${JSON.stringify(phases)}
sock.sendall(f"ACK {nonce}\n".encode()); assert line() == f"GO {nonce}"
for sequence, name in enumerate(payload["providerPhases"], 1):
    sock.sendall(f"PHASE {nonce} {sequence} {name}\n".encode())
    while True:
        acknowledgement = line()
        if acknowledgement == f"PING {nonce}": continue
        assert acknowledgement == f"PHASE_ACK {nonce} {sequence} {name}"
        break
sock.sendall(f"DONE {nonce}\n".encode())
while True:
    message = line()
    if message == f"BYE {nonce}": break
    assert message == f"PING {nonce}"
`);
  await runWithMockedProductionTelemetry(files, ["python3", attester], {
    requireProviderPhases: true,
  });
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  const markers = events.filter((event) => event.event === "provider_phase");
  assert.deepEqual(markers.map((event) => event.providerPhase.name), phases);
  assert.ok(markers.every((event) => event.authenticated === true));
  const completed = events.find((event) => event.event === "child_completed");
  assert.deepEqual(completed.providerPhase, { sequence: 10, name: "cleanup" });
  for (const event of events.filter((event) => event.event === "sample"
      && event.providerPhase !== null)) {
    const preceding = markers.filter((marker) => marker.at <= event.at).at(-1);
    assert.deepEqual(event.providerPhase, preceding.providerPhase);
  }
  assert.deepEqual(events.map((event) => event.eventSequence),
    events.map((_, index) => index + 1));
  assert.equal(events[0].previousEventHash, "0".repeat(64));
  for (let index = 1; index < events.length; index += 1) {
    assert.equal(events[index].previousEventHash, events[index - 1].eventHash);
  }
  assert.ok(events.every((event) => /^[0-9a-f]{64}$/.test(event.eventHash)));
  validateWatchdogEventChain(events);
});

test("reordered or foreign provider phases hard-stop the owned group", async () => {
  for (const [statement, expectedReason] of [
    ['sock.sendall(f"PHASE {nonce} 2 primary_conditioning\\n".encode())', /reordered_provider_phase/],
    ['sock.sendall(b"PHASE foreign 1 common_load\\n")', /foreign_provider_phase_nonce/],
  ]) {
    const files = await fixture();
    const attester = `${files.program}.bad-phase.py`;
    await writeFile(attester, String.raw`import json, os, socket, time
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
sock.sendall(f"ACK {nonce}\n".encode()); assert line() == f"GO {nonce}"
${statement}
time.sleep(60)
`);
    let status = 0;
    try {
      await runWithMockedProductionTelemetry(files, ["python3", attester], {
        requireProviderPhases: true,
      });
    } catch (error) {
      status = error.code;
    }
    assert.equal(status, 97);
    const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
    assert.match(events.find((event) => event.event === "hard_stop").reason, expectedReason);
    assert.equal(events.at(-1).event, "terminated");
  }
});

test("child attestation uses a short socket path when TMPDIR is a long external path", async (t) => {
  if (process.platform !== "darwin") {
    t.skip("the production safety canary is categorically Darwin-only");
    return;
  }
  const files = await fixture();
  const longTmpRoot = await mkdtemp(path.join(tmpdir(), "sc19741-long-external-tmp-"));
  t.after(() => rm(longTmpRoot, { recursive: true, force: true }));
  const longTmp = path.join(
    longTmpRoot,
    "nested-external-campaign-directory".repeat(4),
  );
  await mkdir(longTmp, { recursive: true });
  const attester = `${files.program}.long-tmp-attest.py`;
  const socketRecord = `${files.pids}.socket`;
  await writeFile(attester, String.raw`import json, os, socket, sys
socket_path = os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"]
open(sys.argv[1], "w").write(socket_path + "\n")
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(socket_path)
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
sock.sendall(f"ACK {nonce}\n".encode())
assert line() == f"GO {nonce}"
sock.sendall(f"DONE {nonce}\n".encode())
while True:
    message = line()
    if message == f"BYE {nonce}": break
    assert message == f"PING {nonce}"
`);
  await runWithMockedProductionTelemetry(files, ["python3", attester, socketRecord], {
    environment: { ...process.env, TMPDIR: longTmp },
  });
  const socketPath = (await readFile(socketRecord, "utf8")).trim();
  assert.equal(path.dirname(path.dirname(socketPath)), "/tmp");
  assert.ok(socketPath.length < 104, `socket path is ${socketPath.length} bytes`);
  assert.equal(socketPath.startsWith(longTmp), false);
});

test("a cold child gets a separately bounded startup window while strict telemetry continues", async () => {
  const files = await fixture();
  const attester = `${files.program}.delayed-attest.py`;
  await writeFile(attester, String.raw`import json, os, socket, time
time.sleep(0.35)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
sock.sendall(f"ACK {nonce}\n".encode())
assert line() == f"GO {nonce}"
sock.sendall(f"DONE {nonce}\n".encode())
while True:
    message = line()
    if message == f"BYE {nonce}": break
    assert message == f"PING {nonce}"
`);
  await runWithMockedProductionTelemetry(files, ["python3", attester], {
    telemetryTimeout: 0.1,
    childAttestationTimeout: 3,
  });
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  const waiting = events.filter((event) =>
    event.event === "sample" && event.phase === "awaiting_child_attestation");
  assert.ok(waiting.length >= 2, `expected repeated startup samples, saw ${waiting.length}`);
  assert.ok(events.some((event) => event.event === "child_attested"));
  assert.ok(events.some((event) => event.event === "child_completed"));
});

test("child startup never weakens the strict pre-allocation pressure floor", async () => {
  const files = await fixture();
  const attester = `${files.program}.pressured-delayed-attest.py`;
  await writeFile(attester, String.raw`import os, socket, time
time.sleep(0.35)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
time.sleep(60)
`);
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(files, ["python3", attester], {
      telemetryTimeout: 0.1,
      childAttestationTimeout: 1,
      pressureSamples: [[90, 900, 900], [69, 150, 900]],
    });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) =>
    event.reason?.includes("initial_host_memory_free_below_210")));
  assert.equal(events.some((event) => event.event === "child_attested"), false);
});

test("a child that cannot attest is terminated with no owned residue", async () => {
  const files = await fixture();
  const started = Date.now();
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(files, [
      "python3", files.program, "hold", files.pids, files.telemetry, files.events,
    ], { telemetryTimeout: 0.2, childAttestationTimeout: 0.25 });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) =>
    event.reason === "child_attestation_timeout_at_or_above_0.25s"));
  assert.ok(Date.now() - started < 1_500, "startup timeout drifted toward the runtime deadline");
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});

test("a connected child that stalls before ACK remains monitored until the hard startup deadline", async () => {
  const files = await fixture();
  const staller = `${files.program}.stall-before-ack.py`;
  await writeFile(staller, String.raw`import os, signal, socket, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], "w").write(f"{os.getpid()}\n")
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
while not sock.recv(4096).endswith(b"\n"): pass
time.sleep(60)
`);
  const started = Date.now();
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(files, ["python3", staller, files.pids], {
      telemetryTimeout: 0.1,
      childAttestationTimeout: 1.5,
    });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) => event.phase === "awaiting_child_ack"));
  assert.ok(events.some((event) =>
    event.reason === "child_attestation_timeout_at_or_above_1.5s"));
  assert.ok(Date.now() - started < 2_800, "ACK stall exceeded the startup bound");
  assertGone(Number((await readFile(files.pids, "utf8")).trim()));
});

test("telemetry loss during child startup preserves its exact forensic category", async () => {
  const files = await fixture();
  const delayed = `${files.program}.telemetry-loss-startup.py`;
  await writeFile(delayed, "import time; time.sleep(60)\n");
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(files, ["python3", delayed], {
      telemetryTimeout: 0.1,
      childAttestationTimeout: 1,
      pressureFailureAt: 1,
    });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) => event.reason?.startsWith(
    "child_attestation_telemetry_lost:RuntimeError:injected startup pressure failure")));
});

test("sentinel loss during child startup preserves the launch-sentinel category", async () => {
  const files = await fixture();
  const killer = `${files.program}.kill-sentinel-during-startup.py`;
  await writeFile(killer, String.raw`import os, signal, sys, time
open(sys.argv[1], "w").write(f"{os.getpid()}\n")
time.sleep(0.08)
os.kill(os.getppid(), signal.SIGKILL)
time.sleep(60)
`);
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(files, ["python3", killer, files.pids], {
      telemetryTimeout: 0.1,
      childAttestationTimeout: 1,
    });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) => event.reason?.startsWith("launch_sentinel_lost:status_-")));
  assertGone(Number((await readFile(files.pids, "utf8")).trim()));
});

test("loss of the held attestation lease after GO hard-stops the owned group", async () => {
  const files = await fixture();
  const disconnect = `${files.program}.disconnect.py`;
  await writeFile(disconnect, String.raw`import json, os, signal, socket, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
sock.sendall(f"ACK {nonce}\n".encode())
assert line() == f"GO {nonce}"
open(sys.argv[1], "w").write(f"{os.getpid()}\n")
sock.close()
time.sleep(60)
`);
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(files, ["python3", disconnect, files.pids]);
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) => event.reason?.includes("attestation_channel_lost")));
  const pid = Number((await readFile(files.pids, "utf8")).trim());
  assertGone(pid);
});

test("child attestation categorically rejects every synthetic telemetry surface", async () => {
  const files = await fixture();
  await writeFile(files.telemetry, "1\n");
  let status = 0;
  try {
    await execFileAsync("python3", [
      WATCHDOG, "--max-footprint-bytes", "1000", "--max-runtime-seconds", "2",
      "--host-memory-bytes", "1000", "--min-memory-free-bytes", "100",
      "--min-swap-free-bytes", "100", "--telemetry-file", files.telemetry,
      "--allow-synthetic-telemetry", "--require-child-attestation", "--",
      "python3", files.program, "hold", files.pids, files.telemetry, files.events,
    ], { timeout: 10_000 });
  } catch (error) {
    status = error.code;
    assert.match(error.stderr, /requires production Darwin telemetry and launch controls/);
  }
  assert.equal(status, 2);
  assert.equal(await readFile(files.pids, "utf8").catch(() => ""), "");
});

test("child attestation binds actual RAM and the two-boundary initial release floor", async () => {
  const mismatch = await fixture();
  await assert.rejects(
    () => runWithMockedProductionTelemetry(mismatch, [
      "python3", mismatch.program, "hold", mismatch.pids, mismatch.telemetry, mismatch.events,
    ], { actualHostMemory: 999 }),
    /does not match hw\.memsize 999/,
  );
  assert.equal(await readFile(mismatch.pids, "utf8").catch(() => ""), "");

  const pressured = await fixture();
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(pressured, [
      "python3", pressured.program, "hold", pressured.pids,
      pressured.telemetry, pressured.events,
    ], { memoryFreePercent: 69, memoryFreeBytes: 209 });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  const events = (await readFile(pressured.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) =>
    event.reason?.includes("initial_host_memory_free_below_210")));
  assert.equal(await readFile(pressured.pids, "utf8").catch(() => ""), "");
});

test("the second pre-allocation sample cannot fall back to the runtime-only floor", async () => {
  const files = await fixture();
  const attester = `${files.program}.second-sample.py`;
  const allocated = `${files.pids}.allocated`;
  await writeFile(attester, String.raw`import json, os, socket, sys
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk: raise SystemExit(7)
        data += chunk
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
sock.sendall(f"ACK {nonce}\n".encode())
assert line() == f"GO {nonce}"
open(sys.argv[1], "w").write("allocation-released\n")
`);
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(files, ["python3", attester, allocated], {
      pressureSamples: [[90, 900, 900], [60, 150, 900]],
    });
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 97);
  assert.equal(await readFile(allocated, "utf8").catch(() => ""), "");
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.ok(events.some((event) => event.reason?.includes("initial_host_memory_free_below_210")));
});

test("Darwin host-pressure parsers fail closed on malformed or partial telemetry", async () => {
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
sampler = module.DarwinHostPressureSampler
assert sampler.parse_memory_free_percent("System-wide memory free percentage: 92%\n") == 92
assert sampler.parse_swap_free_bytes("vm.swapusage: total = 4.00G used = 2.00G free = 2.00G") == 2 * 1024 ** 3
for malformed in ["", "System-wide memory free percentage: 101%", "System-wide memory free percentage: 92"]:
    try: sampler.parse_memory_free_percent(malformed)
    except (RuntimeError, ValueError): pass
    else: raise AssertionError("malformed memory pressure accepted")
try: sampler.parse_swap_free_bytes("vm.swapusage unavailable")
except RuntimeError: pass
else: raise AssertionError("missing swap telemetry accepted")
print("host pressure parsers fail closed")
`]);
  assert.match(probe.stdout, /host pressure parsers fail closed/);
});

test("footprint plus host pressure share one aggregate telemetry deadline", async () => {
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, os, sys, time
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
identity = module.process_identity(os.getpid())
class Group:
    def refresh(self): return [identity]
class Footprint:
    def sample(self, pids, timeout): time.sleep(0.06); return 1
class Pressure:
    def sample(self, timeout): time.sleep(0.06); return module.HostPressure(90, 900, 900)
try: module.observe_group(Group(), Footprint(), Pressure(), 0.1)
except TimeoutError: pass
else: raise AssertionError("sequential samplers received independent deadlines")
print("aggregate telemetry deadline enforced")
`]);
  assert.match(probe.stdout, /aggregate telemetry deadline enforced/);
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

test("immediate sentinel loss cannot hide a descendant spawned after the last census", async () => {
  const files = await fixture();
  await writeFile(files.telemetry, "1\n");
  const watchdog = spawn("python3", [
    WATCHDOG, "--max-footprint-bytes", "100", "--sample-interval", "0.2",
    "--telemetry-timeout", "0.2", "--term-grace", "0.1",
    "--event-file", files.events, "--telemetry-file", files.telemetry,
    "--allow-synthetic-telemetry", "--", "python3", files.program, "delayed-child",
    files.pids, files.telemetry, files.events,
  ], { stdio: "ignore" });
  const started = await waitForJsonEvent(files.events, (event) => event.event === "started");
  await waitForFile(files.pids);
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const pids = (await readFile(files.pids, "utf8")).trim().split("\n");
    if (pids.length === 2) break;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  process.kill(started.pid, "SIGKILL");
  const status = await new Promise((resolve) => watchdog.once("close", resolve));
  assert.equal(status, 97);
  await waitForFile(files.pids);
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  assert.equal(pids.length, 2, "delayed descendant never exercised the post-census race");
  pids.forEach(assertGone);
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
group = module.OwnedGroup.__new__(module.OwnedGroup)
group.pgid = live.pgid
group.leader = stale
group.anchors = (stale,)
group.retained = {stale}
class Finished:
    def wait(self, timeout=None): return 0
group.child = Finished()
module.group_identities = lambda pgid: (_ for _ in ()).throw(AssertionError("blind PGID census"))
module.os.killpg = lambda pgid, sig: (_ for _ in ()).throw(AssertionError("blind PGID signal"))
assert group.refresh() == []
group.terminate(0.01)
print("stale identity refused")
`]);
  assert.match(probe.stdout, /stale identity refused/);
});

test("footprint parser requires exact PID parity and sums a complete multi-PID sample", async () => {
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
payload = {"processes": [
    {"pid": 11, "auxiliary": {"phys_footprint": 40}},
    {"pid": 22, "auxiliary": {"phys_footprint": 60}},
]}
assert module.DarwinFootprintSampler.parse_processes([11, 22], payload) == 100
for mutated in [
    {"processes": payload["processes"][:1]},
    {"processes": [payload["processes"][0], payload["processes"][0]]},
    {"processes": [*payload["processes"], {"pid": 33, "auxiliary": {"phys_footprint": 1}}]},
]:
    try:
        module.DarwinFootprintSampler.parse_processes([11, 22], mutated)
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial, duplicate, or extra PID telemetry was accepted")
print("exact footprint PID set required")
`]);
  assert.match(probe.stdout, /exact footprint PID set required/);
});
