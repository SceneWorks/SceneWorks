import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { win32 } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

export const KREA_WDDM_SOURCE = "wddm-pdh-dedicated-sampled";
const COUNTER = String.raw`\GPU Process Memory(*)\Dedicated Usage`;
const INTERVAL_MS = 200;
const MAX_GAP_MS = 1_500;
const MAX_READ_MS = 250;
const FILETIME_TICKS_PER_MS = 10_000n;
// Allow Windows clock quantization within a poll while independently checking
// the monotonic read duration and the interval between fresh provider readings.
const FILETIME_CLOCK_ALLOWANCE = 50n * FILETIME_TICKS_PER_MS;
const MAX_ROWS = 20_000;
const UUID = /^GPU-[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$/i;
const LUID = /^luid_0x[\da-f]{8}_0x[\da-f]{8}$/;
const COLLECTOR = fileURLToPath(new URL("../ci/krea_windows_worker_memory.py", import.meta.url));

const samePath = (left, right) => win32.normalize(left).toLowerCase() === win32.normalize(right).toLowerCase();
const bytes = (value) => Number.isSafeInteger(value) && value >= 0;
function filetime(value) {
  assert.ok(typeof value === "string" && /^[1-9]\d{14,19}$/.test(value), "invalid WDDM FILETIME");
  return BigInt(value);
}

function validateReady(ready, { pid, executable, gpuUuid }) {
  const { identity, adapter } = ready ?? {};
  assert.ok(ready?.event === "ready" && ready.interval_ms === INTERVAL_MS
    && identity?.pid === pid && Number.isSafeInteger(pid) && pid > 0 && typeof identity.executable === "string"
    && win32.isAbsolute(executable) && samePath(identity.executable, executable), "WDDM observer did not identify the launched worker");
  filetime(identity.created_filetime);
  assert.ok(UUID.test(gpuUuid) && adapter?.cuda_uuid === gpuUuid && adapter.counter === COUNTER
    && LUID.test(adapter.luid) && adapter.luid !== "luid_0x00000000_0x00000000"
    && Number.isInteger(adapter.physical_node) && adapter.physical_node >= 0 && adapter.physical_node < 32
    && adapter.node_mask === 2 ** adapter.physical_node, "WDDM observer adapter does not match the leased CUDA UUID and node");
  return { identity, adapter };
}

// Recompute the attributed high-water from retained observations. The compared
// value is sampled Windows dedicated process accounting for this Krea proof.
export function kreaWindowsDedicatedPeak(memory) {
  const { gpuUuid, wddmDedicated: capture, ownedGpuIdentityError } = memory;
  assert.ok(capture?.source === KREA_WDDM_SOURCE && capture.complete === true && !capture.error && !ownedGpuIdentityError
    && capture.exit?.code === 0 && capture.exit.signal === null && capture.intervalMs === INTERVAL_MS
    && capture.maxAllowedGapMs === MAX_GAP_MS && capture.physicalResidencyClaim === false && capture.losslessPeakClaim === false
    && Array.isArray(capture.rows) && capture.rows.length <= MAX_ROWS,
  "Krea Windows dedicated-process capture is incomplete");
  const [ready, ...rows] = capture.rows;
  const { identity, adapter } = validateReady(ready, { pid: capture.workerPid, executable: capture.workerExecutable, gpuUuid });
  let peakBytes = 0, samples = 0, lastPoll = 0, lastEvent = 0, lastSample = null, stamp = 0n;
  let previousReadEnd = 0n, active = false, terminalRequest = null, terminal = false;
  for (const row of rows) {
    const at = row.elapsed_ms;
    assert.ok(!terminal && bytes(at) && at >= lastEvent && at - lastPoll <= MAX_GAP_MS,
      "Krea Windows dedicated-process poll missing or out of order");
    lastEvent = at;
    if (row.event === "active") {
      assert.ok(!active && terminalRequest === null, "unexpected Krea Windows worker attachment marker");
      active = true;
      continue;
    }
    if (row.event === "terminal_requested") {
      assert.ok(terminalRequest === null, "repeated Krea Windows terminal observation request");
      terminalRequest = { at, filetime: filetime(row.not_before_filetime) };
      continue;
    }
    lastPoll = at;
    assert.ok(bytes(row.read_started_ms) && row.read_started_ms <= at && at - row.read_started_ms <= MAX_READ_MS,
      "Krea Windows dedicated-process read exceeded its sampling skew");
    const before = filetime(row.read_started_filetime), after = filetime(row.read_ended_filetime);
    const readTicks = BigInt(at - row.read_started_ms) * FILETIME_TICKS_PER_MS;
    assert.ok(before >= previousReadEnd && after >= before && after - before <= readTicks + FILETIME_CLOCK_ALLOWANCE,
      "Krea Windows local FILETIME clock moved outside the measured read window");
    previousReadEnd = after;
    assert.ok(lastSample === null || at - lastSample <= MAX_GAP_MS, "Krea Windows dedicated-process fresh sample gap");
    if (row.event === "missing") {
      assert.ok(!active && lastSample === null && terminalRequest === null, "Krea Windows dedicated-process counter disappeared while active");
      continue;
    }
    const next = filetime(row.filetime);
    if (row.event === "cached") {
      assert.equal(next, stamp, "Krea Windows cached read does not match the previous observation");
      continue;
    }
    assert.ok((row.event === "sample" || row.event === "terminal") && next > stamp && bytes(row.dedicated_bytes),
      "invalid Krea Windows dedicated-process reading");
    assert.ok(next + FILETIME_CLOCK_ALLOWANCE >= before && next <= after + FILETIME_CLOCK_ALLOWANCE,
      "Krea Windows PDH provider timestamp is outside this poll's local FILETIME window");
    assert.ok(lastSample === null || next - stamp <= BigInt(MAX_GAP_MS) * FILETIME_TICKS_PER_MS,
      "Krea Windows dedicated-process provider timestamp gap");
    stamp = next;
    lastSample = at;
    peakBytes = Math.max(peakBytes, row.dedicated_bytes);
    if (row.dedicated_bytes > 0) samples++;
    if (row.event === "terminal") {
      assert.ok(terminalRequest && row.read_started_ms >= terminalRequest.at && stamp >= terminalRequest.filetime, "Krea Windows terminal reading predates its request");
      terminal = true;
    }
  }
  assert.ok(terminal && samples >= 3 && peakBytes > 0, "Krea Windows dedicated-process terminal or positive samples are incomplete");
  return { peakBytes, samples, identity, adapter, source: KREA_WDDM_SOURCE };
}

export function startKreaWindowsWorkerMemory({ pid, executable, gpuUuid, env }) {
  const expected = { pid, executable, gpuUuid };
  const record = { source: KREA_WDDM_SOURCE, physicalResidencyClaim: false, losslessPeakClaim: false,
    workerPid: pid, workerExecutable: executable, intervalMs: INTERVAL_MS, maxAllowedGapMs: MAX_GAP_MS, rows: [], complete: false };
  const child = spawn("python", ["-I", COLLECTOR, "--pid", String(pid), "--executable", executable, "--cuda-uuid", gpuUuid],
    { env, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
  const started = Promise.withResolvers();
  const closed = Promise.withResolvers();
  let active = false, terminating = false, stderr = "";
  const fault = (error) => {
    record.error ??= error.message;
    started.reject(error);
  };
  child.on("error", fault);
  child.stdin.on("error", fault);
  child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr = (stderr + chunk).slice(-2048); });
  const lines = createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    if (record.error || record.rows.length >= MAX_ROWS) {
      fault(new Error("Krea Windows observer reached its audit row bound or lost its protocol"));
      child.kill();
      return;
    }
    try {
      const row = JSON.parse(line);
      const first = record.rows.length === 0;
      record.rows.push(row);
      if (row.event === "error") throw new Error(row.message);
      if (first) {
        validateReady(row, expected);
        started.resolve();
      }
    } catch (error) {
      fault(error);
      child.kill();
    }
  });
  child.once("close", (code, signal) => {
    if (stderr) record.stderr = stderr;
    record.exit = { code, signal };
    if (code !== 0 || record.rows.at(-1)?.event !== "terminal") fault(new Error("Krea Windows observer closed without a terminal reading"));
    record.complete = !record.error;
    closed.resolve();
  });
  const bounded = async (promise, message) => {
    let timeout;
    try {
      await Promise.race([promise, new Promise((_, reject) => { timeout = setTimeout(() => reject(new Error(message)), 10_000); })]);
    } catch (error) {
      fault(error);
      child.kill();
    } finally {
      clearTimeout(timeout);
    }
  };
  return {
    record,
    async ready() {
      await bounded(started.promise, "Krea Windows observer did not arm");
      if (record.error) throw new Error(record.error);
    },
    markActive() {
      if (!active && !terminating && !record.error && child.exitCode === null) {
        active = true;
        child.stdin.write("active\n");
      }
    },
    async finish() {
      terminating = true;
      if (!record.error && child.exitCode === null) child.stdin.write("terminal\n");
      await bounded(closed.promise, "Krea Windows observer missed the terminal capture deadline");
    },
    async close() {
      if (child.exitCode === null && !child.killed) child.stdin.end();
      await bounded(closed.promise, "Krea Windows observer did not exit");
    },
  };
}
