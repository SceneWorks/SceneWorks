// Synthetic observations for portable proof-controller tests. No device or model.
import { KREA_WDDM_SOURCE } from "./krea-windows-worker-memory.mjs";

export function kreaWindowsMemoryFixture({ gpuUuid = "GPU-2160ecaf-a2b7-7321-87a2-1234567890ab", peakBytes = 80, hostPeakBytes = 100 } = {}) {
  const workerExecutable = String.raw`C:\proof\owned\sceneworks-rust-api.exe`;
  const workerPid = 37;
  const baseTime = 134397427954231587n;
  const stamp = (ms) => String(baseTime + BigInt(ms) * 10_000n);
  const poll = (event, elapsed_ms, extra) => ({ event, elapsed_ms, read_started_ms: elapsed_ms - 1,
    read_started_filetime: stamp(elapsed_ms - 1), read_ended_filetime: stamp(elapsed_ms), ...extra });
  return { gpuUuid, hostLifetimePeakBytes: hostPeakBytes, wddmDedicated: { source: KREA_WDDM_SOURCE, complete: true, exit: { code: 0, signal: null },
    intervalMs: 200, maxAllowedGapMs: 1500, physicalResidencyClaim: false, losslessPeakClaim: false, workerPid, workerExecutable, rows: [
      { event: "ready", interval_ms: 200, identity: { pid: workerPid, created_filetime: "134397607950102030", executable: workerExecutable },
        adapter: { cuda_uuid: gpuUuid, luid: "luid_0x00000000_0x0001e266", node_mask: 1, physical_node: 0, counter: String.raw`\GPU Process Memory(*)\Dedicated Usage` } },
      poll("missing", 10),
      { event: "active", elapsed_ms: 180 },
      poll("sample", 200, { filetime: stamp(200), dedicated_bytes: peakBytes - 20 }),
      poll("cached", 400, { filetime: stamp(200) }),
      poll("sample", 600, { filetime: stamp(600), dedicated_bytes: peakBytes }),
      { event: "terminal_requested", elapsed_ms: 750, not_before_filetime: stamp(750) },
      poll("terminal", 800, { filetime: stamp(800), dedicated_bytes: peakBytes - 10 }),
    ] } };
}
