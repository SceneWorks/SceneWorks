import assert from "node:assert/strict";
import test from "node:test";
import { parseTerminalCudaProbe, probeTerminalCuda, terminalGpuBinding, terminalGpuEnvironment } from "./starvector-terminal-gpu.mjs";
const row = "0, GPU-1234-abcd, NVIDIA RTX test, 600.1, 24000, 22000, 2000\n";
test("CUDA probe selects one stable device on hosts with other devices", async () => {
  const calls = [];
  const options = { execFileImpl: async (command, args) => { calls.push([command, args]); return { stdout: args.includes("--id=0") ? row : row + row.replace("0,", "1,") }; } };
  const binding = await terminalGpuBinding("win32", options);
  assert.equal(binding.uuid, "GPU-1234-abcd");
  assert.equal(calls[0][1][0], "--id=0");
  assert.deepEqual(terminalGpuEnvironment(binding), { SCENEWORKS_GPU_ID: "0", CUDA_VISIBLE_DEVICES: "GPU-1234-abcd", NVIDIA_VISIBLE_DEVICES: "0" });
  await assert.rejects(probeTerminalCuda(binding.uuid, { execFileImpl: async () => ({ stdout: row.replace("1234", "9999") }), expectedUuid: binding.uuid }), /UUID drifted/);
});
test("malformed, ambiguous, and mismatched GPU observations fail before execution", () => {
  assert.throws(() => parseTerminalCudaProbe(row + row), /exactly one/);
  assert.throws(() => parseTerminalCudaProbe(row.replace("24000", "N\/A")), /memory/);
  assert.throws(() => parseTerminalCudaProbe(row, "GPU-other"), /UUID drifted/);
  assert.throws(() => parseTerminalCudaProbe(""), /exactly one/);
});
