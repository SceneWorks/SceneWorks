import { execFile as callback } from "node:child_process";
import { promisify } from "node:util";
const execFile = promisify(callback);
const fail = (message) => { throw new Error(`terminal GPU: ${message}`); };

export function parseTerminalCudaProbe(stdout, expectedUuid) {
  const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length !== 1) fail("selected probe must return exactly one GPU");
  const [index, uuid, name, driver, total, free, used, ...extra] = lines[0].split(",").map((item) => item.trim());
  if (extra.length || !/^[0-9]+$/.test(index) || !/^GPU-[a-f0-9-]+$/i.test(uuid ?? "") || !name || !driver) fail("malformed selected GPU identity");
  if (expectedUuid && uuid !== expectedUuid) fail("selected GPU UUID drifted");
  const quantities = [total, free, used].map((value) => Number(value) * 1024 * 1024);
  if (quantities.some((value) => !Number.isSafeInteger(value) || value < 0) || quantities[0] === 0 || quantities[1] > quantities[0] || quantities[2] > quantities[0]) fail("invalid selected GPU memory");
  return { index, uuid, name, driver, total_bytes: quantities[0], free_bytes: quantities[1], used_bytes: quantities[2] };
}

export async function probeTerminalCuda(selector = "0", { execFileImpl = execFile, expectedUuid } = {}) {
  if (!/^(?:[0-9]+|GPU-[a-f0-9-]+)$/i.test(selector)) fail("explicit CUDA index or UUID required");
  const { stdout } = await execFileImpl("nvidia-smi", [`--id=${selector}`, "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"], { timeout: 10_000, maxBuffer: 64 * 1024, windowsHide: true });
  return parseTerminalCudaProbe(stdout, expectedUuid);
}

export async function terminalGpuBinding(platform = process.platform, options = {}) {
  if (platform === "darwin") return { gpu_id: "mlx", backend: "mlx" };
  if (platform !== "win32") fail(`unsupported campaign platform ${platform}`);
  // This campaign's advertised Windows tier uses ordinal zero. Resolve its
  // physical UUID, then expose exactly that device as CUDA ordinal zero to the
  // native provider; nvidia-smi keeps its physical index for worker discovery.
  const gpu = await probeTerminalCuda("0", options);
  if (gpu.index !== "0") fail("CUDA ordinal zero probe returned a different physical index");
  return { gpu_id: "0", backend: "candle", uuid: gpu.uuid, name: gpu.name, driver: gpu.driver, total_bytes: gpu.total_bytes };
}

export function terminalGpuEnvironment(binding) {
  return binding.backend === "candle" ? { SCENEWORKS_GPU_ID: "0", CUDA_VISIBLE_DEVICES: binding.uuid, NVIDIA_VISIBLE_DEVICES: "0" } : { SCENEWORKS_GPU_ID: "mlx" };
}
