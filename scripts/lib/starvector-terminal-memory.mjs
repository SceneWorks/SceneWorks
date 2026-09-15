// OS observations are separate from the provider allocator high-water mark.
import { execFile as callback } from "node:child_process";
import { promisify } from "node:util";
import { appendFile } from "node:fs/promises";
import { probeTerminalCuda } from "./starvector-terminal-gpu.mjs";
const execFile = promisify(callback);
const fail = (message) => { throw new Error(`terminal memory: ${message}`); };
const positive = (value, name) => { if (!Number.isSafeInteger(value) || value <= 0) fail(`invalid ${name}`); return value; };

export function validateMemorySample(sample, pids) {
  positive(sample.total_bytes, "host total"); positive(sample.available_bytes, "host available");
  if (sample.available_bytes > sample.total_bytes || !Number.isFinite(Date.parse(sample.observed_at))) fail("invalid host sample");
  if (!pids.length || new Set(pids).size !== pids.length || !Array.isArray(sample.processes) || sample.processes.length !== pids.length || pids.some((pid) => sample.processes.filter((item) => item.pid === pid).length !== 1)) fail("an owned process is unobservable");
  for (const item of sample.processes) positive(item.rss_bytes, "OS process RSS");
  return sample;
}

export async function observeTerminalMemory(service, { platform = process.platform, execute = execFile, cuda = probeTerminalCuda } = {}) {
  const pids = [service.api_pid, service.worker_pid].filter((pid) => pid !== null);
  if (!pids.length || pids.some((pid) => !Number.isSafeInteger(pid) || pid <= 0)) fail("invalid owned process identity");
  const shell = async (command, args) => (await execute(command, args, { timeout: 10_000, maxBuffer: 1024 * 1024, windowsHide: true })).stdout.trim();
  const started_at = new Date().toISOString();
  let sample;
  if (platform === "darwin") {
    const total = Number(await shell("sysctl", ["-n", "hw.memsize"]));
    const vm = await shell("vm_stat", []), page = Number(vm.match(/page size of (\d+) bytes/)?.[1]);
    const names = ["free", "inactive", "speculative"];
    const pages = Object.fromEntries(names.map((name) => [name, Number(vm.match(new RegExp(`Pages ${name}:\\s+(\\d+)\\.`))?.[1])]));
    const available = names.reduce((sum, name) => sum + pages[name], 0) * page;
    const rows = await shell("ps", ["-o", "pid=,rss=", "-p", pids.join(",")]);
    sample = { total_bytes: total, available_bytes: available, processes: rows.split(/\r?\n/).filter(Boolean).map((row) => { const [pid, rss] = row.trim().split(/\s+/).map(Number); return { pid, rss_bytes: rss * 1024 }; }), method: "ps RSS KiB; vm_stat (free+inactive+speculative)*page_size; inactive pages treated as reclaimable", raw_host: { page_size: page, pages } };
  } else if (platform === "win32") {
    // IDs are validated integers; emit only memory fields, never command lines or environment.
    const script = `$ErrorActionPreference='Stop'; $m=Get-CimInstance Win32_OperatingSystem; $p=@(Get-Process -Id ${pids.join(",")} | ForEach-Object { @{pid=$_.Id; rss_bytes=$_.WorkingSet64} }); @{total_bytes=[long]$m.TotalVisibleMemorySize*1KB; available_bytes=[long]$m.FreePhysicalMemory*1KB; processes=$p} | ConvertTo-Json -Depth 4 -Compress`;
    sample = { ...JSON.parse(await shell("powershell", ["-NoProfile", "-NonInteractive", "-Command", script])), method: "Get-Process WorkingSet64 (sampled, not lifetime peak); Win32_OperatingSystem physical memory" };
    const binding = service.gpu_binding;
    if (!binding?.uuid || binding.gpu_id !== service.worker?.gpu_id) fail("CUDA binding is missing");
    const gpu = await cuda(binding.uuid, { expectedUuid: binding.uuid });
    if (gpu.uuid !== binding.uuid || gpu.index !== binding.gpu_id || gpu.name !== binding.name) fail("CUDA physical identity drifted");
    sample.accelerator = gpu;
  } else fail(`unsupported platform ${platform}`);
  return validateMemorySample({ ...sample, started_at, observed_at: new Date().toISOString(), worker_pid: service.worker_pid }, pids);
}

// Every observation is durable immediately. Stop/transition await any in-flight
// probe before final capture; callers may hash the file only after stop returns.
export async function startTerminalMemorySampler({ observe, file, intervalMs = 1000 }) {
  const samples = [];
  let timer, pending = Promise.resolve(), error, stopped = false;
  const capture = async (phase) => { const sample = { ...await observe(), phase }; await appendFile(file, `${JSON.stringify(sample)}\n`); samples.push(sample); };
  await capture("before_first_provider_load");
  const schedule = () => { if (!stopped && !error) timer = setTimeout(() => { pending = capture("running").catch((failure) => { error = failure; }).finally(schedule); }, intervalMs); };
  schedule();
  return {
    samples,
    async transition(action) {
      clearTimeout(timer); stopped = true; await pending;
      if (error) throw error;
      await capture("before_worker_transition");
      await action();
      await capture("after_worker_transition");
      stopped = false; schedule();
    },
    async stop() {
      stopped = true; clearTimeout(timer); await pending;
      if (error) throw error;
      await capture("after_native_suites");
      return { interval_ms: intervalMs, peak_kind: "sampled OS RSS and host/device pressure; provider allocator high-water recorded separately", samples };
    },
  };
}

export function terminalHardwareFromSamples({ samples, allocatorPeaks, platform, arch, runnerName }) {
  if (!Array.isArray(samples) || samples.length < 2 || samples[0].phase !== "before_first_provider_load" || samples.at(-1).phase !== "after_native_suites") fail("baseline or final observation missing");
  if (!allocatorPeaks.length || allocatorPeaks.some((value) => !Number.isSafeInteger(value) || value <= 0)) fail("provider allocator high-water missing");
  const baseline = samples[0];
  let observedAt = -Infinity;
  for (const sample of samples) {
    validateMemorySample(sample, sample.processes.map((item) => item.pid));
    if (sample.total_bytes !== baseline.total_bytes) fail("host total changed");
    if (Date.parse(sample.observed_at) < observedAt) fail("observation timestamps moved backwards");
    observedAt = Date.parse(sample.observed_at);
  }
  const rss = Math.max(...samples.map((sample) => sample.processes.reduce((sum, item) => sum + item.rss_bytes, 0)));
  const accelerator = platform === "darwin"
    ? { name: "Apple unified memory", uuid: null, driver_runtime: "MLX", total_bytes: baseline.total_bytes, baseline_free_bytes: baseline.available_bytes, peak_used_bytes: Math.max(...samples.map((sample) => sample.total_bytes - sample.available_bytes)) }
    : (() => { const first = baseline.accelerator; if (!first) fail("CUDA baseline missing"); for (const sample of samples) if (sample.accelerator?.uuid !== first.uuid || sample.accelerator.total_bytes !== first.total_bytes) fail("CUDA sample identity drifted"); return { name: first.name, uuid: first.uuid, driver_runtime: `CUDA/${first.driver}`, total_bytes: first.total_bytes, baseline_free_bytes: first.free_bytes, peak_used_bytes: Math.max(...samples.map((sample) => sample.accelerator.used_bytes), ...allocatorPeaks) }; })();
  return { runner_name: runnerName, os: platform === "darwin" ? "macOS" : "Windows", arch, system_memory_total_bytes: baseline.total_bytes, baseline_available_bytes: baseline.available_bytes, peak_process_rss_bytes: rss, accelerator };
}
