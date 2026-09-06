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

// The wall-clock ceiling is a backstop behind the serialized attestation barriers, never
// the property a test asserts. A flat budget quietly turns into a real deadline as the
// parallel suite grows, so derive it the way the provider-phase test already does: the
// startup window plus one aggregate telemetry window per barrier (payload, ACK, GO, DONE,
// BYE and the spawn itself) and per advertised provider phase. The historical flat value
// stays as a floor so runs configured with a tight telemetry timeout keep their slack.
const ATTESTATION_HANDSHAKE_BARRIERS = 6;
const MIN_MAX_RUNTIME_SECONDS = 2;
// The production tolerance window is 60 s of wall clock (sc-22738). A suite cannot wait that out,
// so every harness that drives a fault to its escalation passes an explicit short window; the
// property under test is the WINDOW, never its production length.
const TEST_TELEMETRY_FAULT_WINDOW = "0.25";

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
      "--telemetry-fault-window", TEST_TELEMETRY_FAULT_WINDOW,
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
    "--telemetry-fault-window", TEST_TELEMETRY_FAULT_WINDOW,
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
    childAttestationTimeout = 1, maxRuntimeSeconds = null,
    telemetryFaultWindow = TEST_TELEMETRY_FAULT_WINDOW,
    requireProviderPhases = false,
    providerPhaseProfile = "campaign-entry",
    memoryFreePercent = 90, memoryFreeBytes = 900, swapFreeBytes = 900,
    pressureSamples = [[memoryFreePercent, memoryFreeBytes, swapFreeBytes]],
    pressureFailureAt = null,
    environment = process.env,
  } = options;
  const pressureFailureAtPython = pressureFailureAt === null ? "None" : String(pressureFailureAt);
  // The phase count comes from the watchdog's own profile table, so the derived deadline
  // cannot drift from the protocol it bounds.
  const providerPhaseProfilePython = requireProviderPhases
    ? JSON.stringify(providerPhaseProfile) : "None";
  const providerPhaseCountPython =
    `len(module.PROVIDER_PHASE_PROFILES.get(${providerPhaseProfilePython}, ()))`;
  const maxRuntimePython = maxRuntimeSeconds === null
    ? `max(${MIN_MAX_RUNTIME_SECONDS}, ${childAttestationTimeout}`
      + ` + (${providerPhaseCountPython} + ${ATTESTATION_HANDSHAKE_BARRIERS})`
      + ` * ${telemetryTimeout})`
    : String(maxRuntimeSeconds);
  // Harness backstop only: it must stay clear of the watchdog's own derived deadline so a
  // real hard stop, not this timeout, is always what a test observes. The longest phase
  // profile today is ten phases; the bound is deliberately looser than that.
  const harnessTimeoutMs = Math.max(10_000, Math.round(1_000 * (6 + (maxRuntimeSeconds
    ?? Math.max(MIN_MAX_RUNTIME_SECONDS, childAttestationTimeout
      + (16 + ATTESTATION_HANDSHAKE_BARRIERS) * telemetryTimeout)))));
  const launcher = `${files.program}.production-watchdog.py`;
  await writeFile(launcher, String.raw`import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
class Footprint:
    def sample(self, pids, timeout, required=()): return 1
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
    "--max-footprint-bytes", "100", "--max-runtime-seconds",
    str(${maxRuntimePython}),
    "--host-memory-bytes", ${JSON.stringify(String(requestedHostMemory))},
    "--min-memory-free-bytes", "100",
    "--sample-interval", "0.02",
    "--telemetry-timeout", ${JSON.stringify(String(telemetryTimeout))},
    "--telemetry-fault-window", ${JSON.stringify(String(telemetryFaultWindow))},
    "--child-attestation-timeout", ${JSON.stringify(String(childAttestationTimeout))},
    "--term-grace", "0.1", "--event-file", ${JSON.stringify(files.events)},
    "--require-child-attestation", ${requireProviderPhases
      ? `"--require-provider-phases", "--provider-phase-profile", ${JSON.stringify(providerPhaseProfile)},`
      : ""}
    "--", *${JSON.stringify(childCommand)}]
raise SystemExit(module.guard(module.parse_args()))
`);
  return execFileAsync("python3", [launcher], { timeout: harnessTimeoutMs, env: environment });
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
    def sample(self, pids, timeout, required=()):
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
  const telemetryTimeoutSeconds = 0.5;
  // Bound startup, each serialized phase/ACK barrier, and completion independently.
  const maxRuntimeSeconds = (phases.length + 2) * telemetryTimeoutSeconds;
  await writeFile(attester, String.raw`import json, os, socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
assert payload["providerPhaseProtocol"] == "sceneworks-provider-phase-v1"
assert payload["providerPhaseProfile"] == "campaign-entry"
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
  try {
    await runWithMockedProductionTelemetry(files, ["python3", attester], {
      requireProviderPhases: true,
      maxRuntimeSeconds,
    });
  } catch (error) {
    const events = (await readFile(files.events, "utf8").catch(() => "")).trim()
      .split("\n").filter(Boolean).map(JSON.parse);
    const hardStops = events.filter((event) => event.event === "hard_stop")
      .map((event) => event.reason);
    assert.fail(
      `provider-phase watchdog unexpectedly exited ${error.code}; hard_stop=${JSON.stringify(hardStops)}; events=${JSON.stringify(events)}`,
    );
  }
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
  assert.equal(events.some((event) => event.event === "hard_stop"), false);
  validateWatchdogEventChain(events);
});

test("bounded carrier profile advertises and acknowledges exactly five phases", async () => {
  const files = await fixture();
  const attester = `${files.program}.bounded-phases.py`;
  const phases = [
    "common_load", "primary_conditioning", "primary_denoise", "primary_decode", "cleanup",
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
assert payload["providerPhaseProfile"] == "bounded-carrier"
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
    providerPhaseProfile: "bounded-carrier",
  });
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.deepEqual(
    events.filter((event) => event.event === "provider_phase")
      .map((event) => event.providerPhase.name),
    phases,
  );
  assert.deepEqual(
    events.find((event) => event.event === "child_completed").providerPhase,
    { sequence: 5, name: "cleanup" },
  );
});

test("bounded campaign entry profile advertises and acknowledges exactly five phases", async () => {
  const files = await fixture();
  const attester = `${files.program}.bounded-campaign-phases.py`;
  const phases = [
    "common_load", "primary_conditioning", "primary_denoise", "primary_decode", "cleanup",
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
assert payload["providerPhaseProfile"] == "bounded-campaign-entry"
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
    providerPhaseProfile: "bounded-campaign-entry",
  });
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  assert.deepEqual(
    events.filter((event) => event.event === "provider_phase")
      .map((event) => event.providerPhase.name),
    phases,
  );
  assert.deepEqual(
    events.find((event) => event.event === "child_completed").providerPhase,
    { sequence: 5, name: "cleanup" },
  );
  assert.equal(events.some((event) => event.event === "hard_stop"), false);
  validateWatchdogEventChain(events);
});

test("bounded campaign entry phase omissions and reordering hard-stop before completion", async () => {
  const phases = [
    "common_load", "primary_conditioning", "primary_denoise", "primary_decode", "cleanup",
  ];
  for (const [mode, expectedReason] of [
    ["missing", /child_completed_before_provider_phase_sequence:observed_4/],
    ["reordered", /child_returned_reordered_provider_phase:expected_1:observed_2/],
  ]) {
    const files = await fixture();
    const attester = `${files.program}.bad-bounded-campaign-phase.py`;
    await writeFile(attester, String.raw`import json, os, socket, sys, time
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"])
def line():
    data = b""
    while not data.endswith(b"\n"): data += sock.recv(1)
    return data.decode().strip()
payload = json.loads(line()); nonce = payload["nonce"]
sock.sendall(f"ACK {nonce}\n".encode()); assert line() == f"GO {nonce}"
if sys.argv[1] == "missing":
    for sequence, name in enumerate(payload["providerPhases"][:-1], 1):
        sock.sendall(f"PHASE {nonce} {sequence} {name}\n".encode())
        while True:
            acknowledgement = line()
            if acknowledgement == f"PING {nonce}": continue
            assert acknowledgement == f"PHASE_ACK {nonce} {sequence} {name}"
            break
    sock.sendall(f"DONE {nonce}\n".encode())
else:
    sock.sendall(f"PHASE {nonce} 2 ${phases[1]}\n".encode())
time.sleep(60)
`);
    let status = 0;
    try {
      await runWithMockedProductionTelemetry(files, ["python3", attester, mode], {
        requireProviderPhases: true,
        providerPhaseProfile: "bounded-campaign-entry",
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

test("an unknown provider phase profile is rejected before the guarded command starts", async () => {
  const files = await fixture();
  let status = 0;
  try {
    await runWithMockedProductionTelemetry(
      files,
      ["python3", files.program, "complete", files.pids, files.telemetry, files.events],
      { requireProviderPhases: true, providerPhaseProfile: "foreign-campaign-profile" },
    );
  } catch (error) {
    status = error.code;
  }
  assert.equal(status, 2);
  assert.equal(await readFile(files.pids, "utf8").catch(() => ""), "");
  assert.equal(await readFile(files.events, "utf8").catch(() => ""), "");
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
  const telemetryTimeoutSeconds = 0.5;
  const childConnectDelaySeconds = 1.1;
  await writeFile(attester, String.raw`import json, os, socket, time
time.sleep(${childConnectDelaySeconds})
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
  try {
    await runWithMockedProductionTelemetry(files, ["python3", attester], {
      telemetryTimeout: telemetryTimeoutSeconds,
      childAttestationTimeout: 3,
      maxRuntimeSeconds: 5,
    });
  } catch (error) {
    const events = (await readFile(files.events, "utf8").catch(() => "")).trim()
      .split("\n").filter(Boolean).map(JSON.parse);
    const hardStops = events.filter((event) => event.event === "hard_stop")
      .map((event) => event.reason);
    assert.fail(
      `cold child watchdog unexpectedly exited ${error.code}; hard_stop=${JSON.stringify(hardStops)}; events=${JSON.stringify(events)}`,
    );
  }
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  const waiting = events.filter((event) =>
    event.event === "sample" && event.phase === "awaiting_child_attestation");
  assert.ok(waiting.length >= 2, `expected repeated startup samples, saw ${waiting.length}`);
  assert.ok(
    waiting.at(-1).at - waiting[0].at > telemetryTimeoutSeconds,
    "startup samples did not span more than one aggregate telemetry window",
  );
  assert.ok(events.some((event) => event.event === "child_attested"));
  assert.ok(events.some((event) => event.event === "child_completed"));
  assert.equal(events.some((event) => event.event === "hard_stop"), false);
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

test("every telemetry probe gets its own full budget under an aggregate staleness deadline", async () => {
  // sc-22738, measured 2026-09-06: the census, the footprint sample and the host-pressure sample
  // drew from ONE `--telemetry-timeout`, so a slow census handed `/usr/bin/footprint` an arbitrary
  // residue of it — 0.31 s of a 1 s budget on a 38 GB process — and starved the host-pressure
  // probe to zero, manufacturing both fault kinds that then hard-stopped a 57-minute render.
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, os, sys, time
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
identity = module.process_identity(os.getpid())
budgets = []
class SlowCensusGroup:
    def refresh(self): time.sleep(0.06); return [identity]
    def root_pids(self, live): return [identity.pid]
class Footprint:
    def sample(self, pids, timeout, required=()):
        budgets.append(timeout); time.sleep(0.06); return 1
class Pressure:
    def sample(self, timeout):
        budgets.append(timeout); time.sleep(0.06); return module.HostPressure(90, 900, 900)
# Three probes at 0.06s each cross a 0.1s per-probe budget in aggregate, and each still received
# the WHOLE budget: a slow census never shortens the probes that follow it.
live, footprint, pressure, elapsed = module.observe_group(
    SlowCensusGroup(), Footprint(), Pressure(), 0.1)
assert footprint == 1, footprint
assert budgets == [0.1, 0.1], budgets
assert elapsed > 0.1, elapsed
# The aggregate remains a real staleness deadline: it is exactly TELEMETRY_PROBE_BUDGETS full
# budgets, so it can never be shorter than one full sample.
assert module.TELEMETRY_PROBE_BUDGETS == 3
class Stalled:
    def sample(self, pids, timeout, required=()): time.sleep(0.2); return 1
class StalledPressure:
    def sample(self, timeout): time.sleep(0.2); return module.HostPressure(90, 900, 900)
try: module.observe_group(SlowCensusGroup(), Stalled(), StalledPressure(), 0.1)
except TimeoutError: pass
else: raise AssertionError("the aggregate staleness deadline was not enforced")
print("independent probe budgets under one staleness deadline")
`]);
  assert.match(probe.stdout, /independent probe budgets under one staleness deadline/);
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

test("footprint parser sums the survivors and refuses only foreign, duplicate or empty telemetry", async () => {
  // sc-22738, measured 2026-09-06: a real MLX capture spawns and reaps compiler/`xcrun` helpers
  // constantly, so PIDs enumerated by the group census routinely exit before `footprint` reads
  // them. Treating that as a PID-set mismatch SIGKILLed live renders (`krea_2_raw:bf16` at 274 s
  // with missing=[16262, 39495], `krea_2_raw:q4` at 167 s with missing=[40001]).
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
parse = module.DarwinFootprintSampler.parse_processes
payload = {"processes": [
    {"pid": 11, "auxiliary": {"phys_footprint": 40}},
    {"pid": 22, "auxiliary": {"phys_footprint": 60}},
]}
assert parse([11, 22], payload) == 100
assert parse([11, 22], payload, [11, 22]) == 100
# A tracked PID that vanished between the census and the sample is DROPPED, and the survivors are
# summed: the sample proceeds, the next tick re-enumerates the group.
assert parse([11, 22], {"processes": payload["processes"][:1]}) == 40
assert parse([11, 22, 33], payload, [22]) == 100
for mutated, required in [
    ({"processes": [payload["processes"][0], payload["processes"][0]]}, ()),
    ({"processes": [*payload["processes"], {"pid": 33, "auxiliary": {"phys_footprint": 1}}]}, ()),
    ({"processes": []}, ()),
]:
    try:
        parse([11, 22], mutated, required)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate, foreign, or empty PID telemetry was accepted")
# Losing the guarded ROOT is telemetry loss, and it is its own category so the tolerance cannot
# swallow it.
try:
    parse([11, 22], {"processes": payload["processes"][1:]}, [11])
except module.RootTelemetryLost:
    pass
else:
    raise AssertionError("the guarded root vanishing from its own sample was accepted")
assert issubclass(module.RootTelemetryLost, RuntimeError)
print("survivor sum with root parity required")
`]);
  assert.match(probe.stdout, /survivor sum with root parity required/);
});

test("a transient child exiting mid-sample is tolerated; the guarded root vanishing is not", async () => {
  // The end-to-end shape of the same defect, through `observe_group` and the guard loop: the
  // census enumerates a helper that exits before the sampler reads it. `required` carries only the
  // guarded root, so the tick proceeds on the survivors.
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
root = module.Identity(101, 100, "S", "root")
helper = module.Identity(202, 100, "S", "helper")
class Group:
    def refresh(self): return [root, helper]
    def root_pids(self, live): return [root.pid]
class Footprint:
    """The transient helper exits between the census and the sample."""
    def sample(self, pids, timeout, required=()):
        assert list(required) == [root.pid], required
        payload = {"processes": [{"pid": root.pid, "auxiliary": {"phys_footprint": 7}}]}
        return module.DarwinFootprintSampler.parse_processes(pids, payload, required)
live, footprint, pressure, _ = module.observe_group(Group(), Footprint(), None, 1.0)
assert footprint == 7, footprint
assert [item.pid for item in live] == [root.pid, helper.pid]
class RootGone:
    def sample(self, pids, timeout, required=()):
        payload = {"processes": [{"pid": helper.pid, "auxiliary": {"phys_footprint": 7}}]}
        return module.DarwinFootprintSampler.parse_processes(pids, payload, required)
try:
    module.observe_group(Group(), RootGone(), None, 1.0)
except module.RootTelemetryLost:
    pass
else:
    raise AssertionError("the guarded root vanishing from its sample did not stop the guard")
print("transient child tolerated, root loss stops")
`]);
  assert.match(probe.stdout, /transient child tolerated, root loss stops/);
});

/**
 * Drive the guard with a scripted footprint sampler. `outcome` is the body of a Python
 * `def outcome(n)` called with the 1-based sample number: it returns a footprint or raises.
 * Everything else is production code — the real group, the real loop, the real escalation rule.
 */
async function runWithScriptedFootprint(files, name, outcome, options = {}) {
  const {
    mode = "hold", ceiling = 100, maxRuntimeSeconds = "30",
    telemetryFaultWindow = null, timeoutMs = 20_000,
  } = options;
  const launcher = `${files.program}.${name}.py`;
  const windowArgument = telemetryFaultWindow === null
    ? "" : `"--telemetry-fault-window", ${JSON.stringify(String(telemetryFaultWindow))},`;
  await writeFile(launcher, String.raw`import importlib.util, subprocess, sys, time
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
def footprint_timeout():
    return subprocess.TimeoutExpired(cmd=["/usr/bin/footprint", "-p", "1"], timeout=10.0)
def aggregate_deadline():
    return TimeoutError("aggregate host-pressure telemetry deadline expired")
${outcome}
state = {"samples": 0}
class Footprint:
    def sample(self, pids, timeout, required=()):
        state["samples"] += 1
        return outcome(state["samples"])
module.DarwinFootprintSampler = Footprint
sys.argv = [${JSON.stringify(WATCHDOG)},
    "--max-footprint-bytes", ${JSON.stringify(String(ceiling))},
    "--max-runtime-seconds", ${JSON.stringify(String(maxRuntimeSeconds))},
    "--sample-interval", "0.02", "--telemetry-timeout", "0.2",
    ${windowArgument}
    "--term-grace", "0.1", "--event-file", ${JSON.stringify(files.events)},
    "--", "python3", ${JSON.stringify(files.program)}, ${JSON.stringify(mode)},
    ${JSON.stringify(files.pids)}, ${JSON.stringify(files.telemetry)},
    ${JSON.stringify(files.events)}]
raise SystemExit(module.guard(module.parse_args()))
`);
  let status = 0;
  try {
    await execFileAsync("python3", [launcher], { timeout: timeoutMs });
  } catch (error) {
    status = error.code;
  }
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  return {
    status,
    events,
    faults: events.filter((event) => event.event === "telemetry_fault"),
    stopped: events.find((event) => event.event === "hard_stop") ?? null,
  };
}

test("a footprint timeout followed by a good sample is never telemetry loss", async () => {
  // sc-22738, measured 2026-09-06: `bernini:q4:mlx` rendered 56.8 minutes at a steady 38 GB
  // against a 94.8 GB ceiling and was SIGKILLed by a `/usr/bin/footprint` timeout. One failed
  // probe is not a reading, and a false process-group SIGKILL through a live Metal command buffer
  // is strictly worse for this host than a late stop.
  const files = await fixture();
  const result = await runWithScriptedFootprint(files, "single-footprint-timeout", String.raw`
def outcome(n):
    if n == 2: raise footprint_timeout()
    return 1
`, { maxRuntimeSeconds: "1.5" });
  // The guard runs to its own wall-time ceiling: the tolerated fault left no mark on the stop.
  assert.equal(result.status, 97);
  assert.equal(result.stopped.reason, "runtime_at_or_above_1.5s");
  assert.equal(result.faults.length, 1);
  assert.equal(result.faults[0].consecutiveFaults, 1);
  assert.match(result.faults[0].reason, /^TimeoutExpired:/);
  const samplesAfter = result.events.filter((event) =>
    event.event === "sample" && event.eventSequence > result.faults[0].eventSequence);
  assert.ok(samplesAfter.length > 0, "the guard stopped sampling after the tolerated fault");
});

test("heterogeneous sampler faults in one run share the window; none escalates on its own", async () => {
  // The measured escalation: ONE `/usr/bin/footprint` timeout plus the SEPARATE aggregate
  // host-pressure deadline inside the same fault run exhausted a three-TICK tolerance in 1.2 s.
  // Five alternating faults well inside the window must all be tolerated, whatever their kind.
  const files = await fixture();
  const result = await runWithScriptedFootprint(files, "mixed-fault-run", String.raw`
def outcome(n):
    if n in (2, 4, 6): raise footprint_timeout()
    if n in (3, 5): raise aggregate_deadline()
    return 1
`, { maxRuntimeSeconds: "1.5" });
  assert.equal(result.status, 97);
  assert.equal(result.stopped.reason, "runtime_at_or_above_1.5s",
    "a tolerated fault run must not become the stop reason");
  assert.deepEqual(result.faults.map((event) => event.consecutiveFaults), [1, 2, 3, 4, 5],
    "one unbroken fault run of five, tolerated on wall clock rather than tick count");
  assert.deepEqual(
    [...new Set(result.faults.map((event) => event.reason.split(":")[0]))],
    ["TimeoutExpired", "TimeoutError"],
    "both measured fault kinds went through the one tolerance",
  );
  assert.ok(result.faults.every((event) => event.faultElapsedSeconds < event.faultWindowSeconds));
});

test("a transient sampling fault is re-enumerated; one past the wall-clock window is telemetry loss", async () => {
  const files = await fixture();
  // A short run then a recovery that resets the window, followed by a permanent failure: the
  // guard rides out the first run and stops only once a run occupies the whole window.
  const result = await runWithScriptedFootprint(files, "transient-fault", String.raw`
def outcome(n):
    if n in (2, 3) or n >= 5: raise RuntimeError("transient_source_failure")
    return 1
`, { telemetryFaultWindow: 1 });
  assert.equal(result.status, 97);
  const runs = result.faults.map((event) => event.consecutiveFaults);
  assert.deepEqual(runs.slice(0, 3), [1, 2, 1],
    "two tolerated ticks, a recovery that resets the run, then a fresh run");
  // A tick COUNT cannot be what stopped it: the surviving run had to occupy a full second of wall
  // clock, which is many more ticks than any fixed tolerance would have allowed.
  assert.ok(runs.at(-1) > 3, `the final fault run was only ${runs.at(-1)} ticks`);
  assert.ok(result.faults.every((event) =>
    event.reason === "RuntimeError:transient_source_failure"));
  assert.ok(result.faults.every((event) => event.lastGoodPhysicalFootprintBytes === 1),
    "a tolerated fault must keep the previous good sample as the current reading");
  assert.match(result.stopped.reason, /^telemetry_lost:RuntimeError:transient_source_failure$/);
  assert.equal(result.stopped.telemetryFaultHistory.windowSeconds, 1);
  assert.ok(result.stopped.telemetryFaultHistory.elapsedSeconds >= 1,
    `stopped after ${result.stopped.telemetryFaultHistory.elapsedSeconds}s`);
  assert.equal(result.stopped.telemetryFaultHistory.faults, runs.at(-1) + 1);
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});

test("a good sample re-anchors the window, so a later fault gets the whole window again", async () => {
  // The window is measured from the first fault of the CURRENT run, never from an ancient one: a
  // capture that sampled cleanly for minutes must not be one unlucky probe away from a SIGKILL.
  const files = await fixture();
  const result = await runWithScriptedFootprint(files, "window-reanchored", String.raw`
start = time.monotonic()
def outcome(n):
    elapsed = time.monotonic() - start
    if 0.05 < elapsed < 0.20: raise footprint_timeout()
    if 1.00 < elapsed < 1.15: raise footprint_timeout()
    return 1
`, { telemetryFaultWindow: 0.3, maxRuntimeSeconds: "2.0" });
  assert.equal(result.status, 97);
  assert.equal(result.stopped.reason, "runtime_at_or_above_2.0s",
    "the second fault burst inherited the first burst's window anchor");
  assert.equal(result.faults.filter((event) => event.consecutiveFaults === 1).length, 2,
    "the healthy stretch between the bursts did not close the first fault run");
});

test("the pre-release observation takes the same tolerance as every other sampler path", async () => {
  const files = await fixture();
  const tolerated = await runWithScriptedFootprint(files, "initial-fault-tolerated", String.raw`
def outcome(n):
    if n == 1: raise footprint_timeout()
    return 1
`, { maxRuntimeSeconds: "1.5" });
  assert.equal(tolerated.stopped.reason, "runtime_at_or_above_1.5s",
    "a single failed pre-release probe must not refuse the capture");
  assert.deepEqual(tolerated.faults.map((event) => event.phase), ["before_child_release"]);

  const lost = await fixture();
  const stopped = await runWithScriptedFootprint(lost, "initial-fault-persistent", String.raw`
def outcome(n): raise footprint_timeout()
`, { telemetryFaultWindow: 0.3 });
  assert.equal(stopped.status, 97);
  assert.match(stopped.stopped.reason, /^initial_telemetry_lost:TimeoutExpired:/);
  assert.ok(stopped.faults.length >= 1, "the tolerated pre-release faults were never recorded");
  assert.ok(stopped.stopped.telemetryFaultHistory.elapsedSeconds >= 0.3);
});

test("the guard's production cadence and telemetry budgets are the documented ones", async () => {
  // sc-22738: the measured false positive ran `/usr/bin/footprint` over a 38 GB process ~2.5x a
  // second on a 1 s budget it did not even get to keep. These constants are the fix's other half.
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
assert module.SAMPLE_INTERVAL_SECONDS == 2.0, module.SAMPLE_INTERVAL_SECONDS
assert module.TELEMETRY_TIMEOUT_SECONDS >= 10.0, module.TELEMETRY_TIMEOUT_SECONDS
assert module.TELEMETRY_FAULT_WINDOW_SECONDS == 60.0, module.TELEMETRY_FAULT_WINDOW_SECONDS
# The aggregate deadline is derived from the per-probe budget and is never shorter than one sample.
assert module.TELEMETRY_PROBE_BUDGETS >= 1, module.TELEMETRY_PROBE_BUDGETS
# A cold child must be allowed to import its framework, and its window must clear a whole tick.
assert module.CHILD_ATTESTATION_TIMEOUT_SECONDS > (
    module.TELEMETRY_TIMEOUT_SECONDS * module.TELEMETRY_PROBE_BUDGETS)
sys.argv = [${JSON.stringify(WATCHDOG)}, "--max-footprint-bytes", "1", "--", "true"]
args = module.parse_args()
assert args.sample_interval == module.SAMPLE_INTERVAL_SECONDS
assert args.telemetry_timeout == module.TELEMETRY_TIMEOUT_SECONDS
assert args.telemetry_fault_window == module.TELEMETRY_FAULT_WINDOW_SECONDS
assert args.child_attestation_timeout == module.CHILD_ATTESTATION_TIMEOUT_SECONDS
print("production cadence and budgets asserted")
`]);
  assert.match(probe.stdout, /production cadence and budgets asserted/);
});

test("losing the guarded root stops immediately, even inside a tolerated fault run", async () => {
  const files = await fixture();
  const result = await runWithScriptedFootprint(files, "root-loss-mid-run", String.raw`
def outcome(n):
    if n == 2: raise footprint_timeout()
    if n == 3: raise module.RootTelemetryLost("footprint lost the guarded root PIDs: missing=[1]")
    return 1
`);
  assert.equal(result.status, 97);
  assert.equal(result.faults.length, 1, "root loss must not be absorbed by the open fault run");
  assert.match(result.stopped.reason, /^telemetry_lost:RootTelemetryLost:/);
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});

/**
 * Drive the guard with a scripted `/bin/ps` process-group census. `censusOutcome` is the body of a
 * Python `def census_outcome(n, state)` called with the 1-based census number: it returns to let the
 * REAL census run, or raises. `footprintOutcome` is the same contract as `runWithScriptedFootprint`
 * plus the shared `state`, so a test can make the census fail only at a chosen point in the tick.
 * Everything else is production code — the real group, the real loop, the real escalation rule.
 */
async function runWithScriptedCensus(files, name, censusOutcome, options = {}) {
  const {
    mode = "hold", ceiling = 100, maxRuntimeSeconds = "1.5", timeoutMs = 20_000,
    footprintOutcome = "def footprint_outcome(n, state): return 1",
  } = options;
  const launcher = `${files.program}.${name}.py`;
  await writeFile(launcher, String.raw`import importlib.util, subprocess, sys, time
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
def census_timeout():
    return subprocess.TimeoutExpired(
        cmd=["/bin/ps", "-ww", "-axo", "pid=,pgid=,state=,lstart="], timeout=10.0)
def footprint_timeout():
    return subprocess.TimeoutExpired(cmd=["/usr/bin/footprint", "-p", "1"], timeout=10.0)
state = {"census": 0, "samples": 0, "censusFaults": 0, "inFault": False}
${censusOutcome}
${footprintOutcome}
real_group_identities = module.group_identities
real_process_identity = module.process_identity
# A slow or hung /bin/ps fails BOTH census probes: the whole-system group walk and the per-PID
# identity check that OwnedGroup.refresh runs over its anchors first.
def scripted_census(pgid, timeout=None):
    state["census"] += 1
    census_outcome(state["census"], state)
    return real_group_identities(pgid, timeout)
def scripted_identity(pid, timeout=None):
    state["census"] += 1
    census_outcome(state["census"], state)
    return real_process_identity(pid, timeout)
module.group_identities = scripted_census
module.process_identity = scripted_identity
class Footprint:
    def sample(self, pids, timeout, required=()):
        state["samples"] += 1
        return footprint_outcome(state["samples"], state)
module.DarwinFootprintSampler = Footprint
sys.argv = [${JSON.stringify(WATCHDOG)},
    "--max-footprint-bytes", ${JSON.stringify(String(ceiling))},
    "--max-runtime-seconds", ${JSON.stringify(String(maxRuntimeSeconds))},
    "--sample-interval", "0.02", "--telemetry-timeout", "0.2",
    "--term-grace", "0.1", "--event-file", ${JSON.stringify(files.events)},
    "--", "python3", ${JSON.stringify(files.program)}, ${JSON.stringify(mode)},
    ${JSON.stringify(files.pids)}, ${JSON.stringify(files.telemetry)},
    ${JSON.stringify(files.events)}]
raise SystemExit(module.guard(module.parse_args()))
`);
  let status = 0;
  try {
    await execFileAsync("python3", [launcher], { timeout: timeoutMs });
  } catch (error) {
    status = error.code;
  }
  const events = (await readFile(files.events, "utf8")).trim().split("\n").map(JSON.parse);
  return {
    status,
    events,
    faults: events.filter((event) => event.event === "telemetry_fault"),
    stopped: events.find((event) => event.event === "hard_stop") ?? null,
  };
}

test("a census timeout during the runtime tick is a tolerated telemetry fault", async () => {
  // sc-22738, measured 2026-09-06: `bernini:q8:mlx` rendered 85 minutes at a steady 52 GB against a
  // 94.8 GB ceiling and was SIGKILLed by `monitor_failure:TimeoutExpired` from the whole-system
  // `/bin/ps` census on a hard-coded 1 s budget. `ps` is a telemetry SOURCE like `/usr/bin/
  // footprint`: a census that fails is an unknown view, never a monitor bug and never an empty
  // group.
  const files = await fixture();
  const result = await runWithScriptedCensus(files, "census-timeout-tolerated", String.raw`
def census_outcome(n, state):
    # Well inside the runtime loop: the pre-release observation takes only the first sample.
    if state["samples"] >= 3 and state["censusFaults"] < 3:
        state["censusFaults"] += 1
        raise census_timeout()
`);
  assert.equal(result.status, 97);
  // The guard rode the census outage out to its own wall-time ceiling — it neither failed as a
  // monitor bug nor read the unknown view as a finished render.
  assert.equal(result.stopped.reason, "runtime_at_or_above_1.5s");
  assert.equal(result.faults.length, 3);
  assert.ok(result.faults.every((event) => event.phase === "runtime"));
  assert.ok(result.faults.every((event) => event.reason.startsWith("TimeoutExpired:/bin/ps")
    || event.reason.startsWith("TimeoutExpired:Command '['/bin/ps'")),
  `census fault reason was ${result.faults[0].reason}`);
  assert.ok(result.faults.every((event) => event.lastGoodPhysicalFootprintBytes === 1),
    "a tolerated census must keep the previous good sample as the current reading");
  const samplesAfter = result.events.filter((event) =>
    event.event === "sample" && event.eventSequence > result.faults.at(-1).eventSequence);
  assert.ok(samplesAfter.length > 0, "the guard stopped sampling after the tolerated census fault");
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});

test("a census timeout raised from the post-tick refresh paths is tolerated", async () => {
  // The census inside `observe_group` was already covered by the tolerance; the `refresh()` calls
  // around it were not — and the one inside the sampler-fault handler raised THROUGH that handler
  // straight to `monitor_failure`. Fail the census only while a footprint fault is being handled,
  // which is exactly that call.
  const files = await fixture();
  const result = await runWithScriptedCensus(files, "census-timeout-post-tick", String.raw`
def census_outcome(n, state):
    if state["inFault"]:
        state["inFault"] = False
        state["censusFaults"] += 1
        raise census_timeout()
`, {
    footprintOutcome: String.raw`
def footprint_outcome(n, state):
    if n == 4:
        state["inFault"] = True
        raise footprint_timeout()
    return 1
`,
  });
  assert.equal(result.status, 97);
  assert.equal(result.stopped.reason, "runtime_at_or_above_1.5s");
  assert.equal(result.faults.length, 2, "one census fault and one footprint fault, both tolerated");
  // The census fault is recorded FIRST even though the footprint failed first: it can only have
  // come from the census the sampler-fault handler itself runs.
  assert.deepEqual(result.faults.map((event) => event.consecutiveFaults), [1, 2]);
  assert.match(result.faults[0].reason, /^TimeoutExpired:.*\/bin\/ps/);
  assert.match(result.faults[1].reason, /^TimeoutExpired:.*footprint/);
});

test("a successful census that lacks the root stops at once, after tolerated census faults", async () => {
  // Root-loss detection must survive the tolerance: a census that FAILS is unknown, but the first
  // SUCCESSFUL one that no longer enumerates the group ends the guard immediately with the child's
  // own status rather than riding the wall-time ceiling out.
  const files = await fixture();
  const result = await runWithScriptedCensus(files, "census-fault-then-root-gone", String.raw`
def census_outcome(n, state):
    if state["samples"] >= 2 and state["censusFaults"] < 2:
        state["censusFaults"] += 1
        raise census_timeout()
`, { mode: "complete", maxRuntimeSeconds: "10" });
  assert.equal(result.status, 0, "the completed group was reported through the child's own status");
  assert.equal(result.stopped, null, "a completed group is not a hard stop");
  assert.equal(result.faults.length, 2, "the census faults were tolerated, not escalated");
});

test("a census outage across the sentinel's exit defers instead of declaring a live group", async () => {
  // The other half of "a failed census is unknown": once the sentinel reports a status, the guard
  // re-censuses to tell normal cleanup from a failure. Reading a FAILED census there as the
  // previous view would call the group live and hard-stop a run that simply finished — so the
  // guard defers until a census succeeds. The outage here spans the sentinel's exit and then
  // clears.
  const files = await fixture();
  const result = await runWithScriptedCensus(files, "census-outage-over-exit", String.raw`
start = time.monotonic()
def census_outcome(n, state):
    if 0.15 < time.monotonic() - start < 1.2:
        state["censusFaults"] += 1
        raise census_timeout()
`, { mode: "complete", maxRuntimeSeconds: "10" });
  assert.equal(result.status, 0, "the completed group was reported through the child's own status");
  assert.equal(result.stopped, null,
    "an unknown census while the sentinel exits must not become launch_sentinel_failed_with_live_group");
  assert.ok(result.faults.length > 0, "the census outage was never recorded");
});

test("an exception that is not a telemetry source stays a monitor failure", async () => {
  // The tolerance absorbs telemetry SOURCES, enumerated in TELEMETRY_SOURCE_ERRORS. A bug in the
  // monitor itself must still fail closed as `monitor_failure`, never be ridden out as a fault.
  const files = await fixture();
  const result = await runWithScriptedFootprint(files, "monitor-bug", String.raw`
def outcome(n):
    if n == 2: raise AttributeError("monitor bug")
    return 1
`, { maxRuntimeSeconds: "1.5" });
  assert.equal(result.status, 97);
  assert.equal(result.stopped.reason, "monitor_failure:AttributeError:monitor bug");
  assert.equal(result.faults.length, 0, "a monitor bug must not be recorded as a telemetry fault");
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});

test("the process-group census carries the per-probe telemetry budget, not a hard-coded second", async () => {
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, subprocess, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
# The census is a telemetry probe: it takes the telemetry budget, never the 1 s that SIGKILLed a
# healthy 85-minute render.
assert module.CENSUS_TIMEOUT_SECONDS == module.TELEMETRY_TIMEOUT_SECONDS, module.CENSUS_TIMEOUT_SECONDS
assert module.CENSUS_TIMEOUT_SECONDS >= 10.0, module.CENSUS_TIMEOUT_SECONDS
budgets = []
class Recorder:
    TimeoutExpired = subprocess.TimeoutExpired
    SubprocessError = subprocess.SubprocessError
    PIPE = subprocess.PIPE
    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""
    @staticmethod
    def run(command, **kwargs):
        budgets.append(kwargs["timeout"])
        return Recorder.Completed()
    class Popen:
        pid = -1
        returncode = 0
        def __init__(self, command, **kwargs): pass
        def communicate(self, timeout=None):
            budgets.append(timeout)
            return ("", "")
module.subprocess = Recorder
def census_budgets():
    budgets.clear()
    module.process_identity(1)
    module.group_identities(1)
    module.parent_pids()
    return list(budgets)
assert census_budgets() == [10.0, 10.0, 10.0], budgets
# Every census reads the run's adopted budget, so --telemetry-timeout reaches identity_is_live,
# OwnedGroup.refresh and root_pids without a per-call argument.
module.set_census_timeout(0.75)
assert census_budgets() == [0.75, 0.75, 0.75], budgets
# The enumerated telemetry sources are the ones the probes actually raise.
for error in [subprocess.TimeoutExpired(cmd=["/bin/ps"], timeout=1.0), OSError("ps"),
              TimeoutError("aggregate"), RuntimeError("ps group census failed"),
              ValueError("malformed")]:
    assert isinstance(error, module.TELEMETRY_SOURCE_ERRORS), error
for error in [AttributeError("bug"), TypeError("bug"), NameError("bug"), KeyError("bug")]:
    assert not isinstance(error, module.TELEMETRY_SOURCE_ERRORS), error
print("census budget is the telemetry budget")
`]);
  assert.match(probe.stdout, /census budget is the telemetry budget/);
});

test("a guard run adopts its --telemetry-timeout as the census budget", async () => {
  const files = await fixture();
  await writeFile(files.telemetry, "1\n");
  const probe = await execFileAsync("python3", ["-c", String.raw`
import importlib.util, sys
spec = importlib.util.spec_from_file_location("watchdog", ${JSON.stringify(WATCHDOG)})
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
assert module.CENSUS_TIMEOUT_SECONDS == 10.0, module.CENSUS_TIMEOUT_SECONDS
sys.argv = [${JSON.stringify(WATCHDOG)},
    "--max-footprint-bytes", "100", "--max-runtime-seconds", "0.5",
    "--sample-interval", "0.02", "--telemetry-timeout", "0.37",
    "--term-grace", "0.1", "--event-file", ${JSON.stringify(files.events)},
    "--telemetry-file", ${JSON.stringify(files.telemetry)}, "--allow-synthetic-telemetry",
    "--", "python3", ${JSON.stringify(files.program)}, "hold",
    ${JSON.stringify(files.pids)}, ${JSON.stringify(files.telemetry)},
    ${JSON.stringify(files.events)}]
module.guard(module.parse_args())
assert module.CENSUS_TIMEOUT_SECONDS == 0.37, module.CENSUS_TIMEOUT_SECONDS
print("census budget adopted from the run")
`], { timeout: 20_000 });
  assert.match(probe.stdout, /census budget adopted from the run/);
});

test("the ceiling still stops on the first good sample that breaches it after tolerated faults", async () => {
  const files = await fixture();
  const result = await runWithScriptedFootprint(files, "breach-after-faults", String.raw`
def outcome(n):
    if n in (2, 3): raise footprint_timeout()
    if n >= 4: return 150
    return 1
`);
  assert.equal(result.status, 97);
  assert.equal(result.faults.length, 2);
  assert.equal(result.stopped.reason,
    "physical_footprint_at_or_above_100:observed_150",
    "the stop must carry the good sample's value, never a tolerated fault");
  const pids = (await readFile(files.pids, "utf8")).trim().split("\n").map(Number);
  pids.forEach(assertGone);
});
