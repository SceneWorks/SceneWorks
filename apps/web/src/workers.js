// Shared worker-state helpers (sc-2082 — Worker Progress Redesign epic sc-2080).
//
// The web client already maintains a live `workers` array via the SSE channel
// (App.jsx subscribes to worker.updated + queue.updated). Each WorkerProgressCard
// needs to look up the live worker assigned to a job and pull its hardware
// description + utilization. Centralizing that here means screens don't each
// rebuild the workersById map and the WorkerProgressCard component doesn't
// reach into AppContext directly.

/**
 * Derive hardware descriptors from a worker snapshot.
 *
 * Returns { device, vendor, architecture, gpuLabel } where:
 *   device:       "CPU" | "GPU" | null
 *   vendor:       "NVIDIA" | "Apple" | null
 *   architecture: "cuda" | "mps" | "mlx" | null
 *   gpuLabel:     the human-readable GPU/device label (e.g. "Apple M2 Ultra")
 *
 * Vendor + architecture are best-effort heuristics over `worker.gpuName`. The
 * worker contract does not carry these as separate fields today; if it grows
 * them, switch to using them directly.
 */
export function deriveWorkerHardware(worker) {
  if (!worker) {
    return { device: null, vendor: null, architecture: null, gpuLabel: null };
  }
  const capabilities = Array.isArray(worker.capabilities) ? worker.capabilities : [];
  const isGpu = capabilities.includes("gpu");
  const isCpu = capabilities.includes("cpu") && !isGpu;
  const device = isGpu ? "GPU" : isCpu ? "CPU" : null;
  const gpuLabel = worker.gpuName ?? worker.gpu_name ?? null;

  if (!isGpu) {
    return { device, vendor: null, architecture: null, gpuLabel };
  }

  const name = (gpuLabel ?? "").toLowerCase();
  let vendor = null;
  let architecture = null;
  if (/(nvidia|rtx|geforce|tesla|titan|quadro|^a\d{2,4}\b|^h\d{2,4}\b|^l\d\b)/i.test(name)) {
    vendor = "NVIDIA";
    architecture = "cuda";
  } else if (/(apple|^m\d\b|m\d\s*(pro|max|ultra)?)/i.test(name)) {
    vendor = "Apple";
    architecture = "mps";
  }
  return { device, vendor, architecture, gpuLabel };
}

/**
 * Find the live worker snapshot associated with a job.
 *
 * Preference order:
 *   1. exact `worker.id === job.workerId`
 *   2. exact `worker.gpuId === job.assignedGpu`
 *   3. null
 */
export function findWorkerForJob(job, workers) {
  if (!job || !Array.isArray(workers) || workers.length === 0) {
    return null;
  }
  if (job.workerId) {
    const byId = workers.find((worker) => worker.id === job.workerId);
    if (byId) return byId;
  }
  if (job.assignedGpu && job.assignedGpu !== "auto") {
    const byGpu = workers.find((worker) => worker.gpuId === job.assignedGpu);
    if (byGpu) return byGpu;
  }
  return null;
}

/**
 * Pull live percentage meters off a worker's utilization snapshot.
 *
 * Returns { memUsedPct, loadPct } in 0–100 range, or null for either field
 * when the worker has not reported it yet. This is the source for the
 * WorkerProgressCard hardware row while a job is queued or running.
 */
export function liveMeters(worker) {
  const utilization = worker?.utilization;
  if (!utilization) {
    return { memUsedPct: null, loadPct: null };
  }
  const usedMb = Number(utilization.memoryUsedMb ?? utilization.memory_used_mb);
  const totalMb = Number(utilization.memoryTotalMb ?? utilization.memory_total_mb);
  const memUsedPct =
    Number.isFinite(usedMb) && Number.isFinite(totalMb) && totalMb > 0
      ? Math.max(0, Math.min(100, (usedMb / totalMb) * 100))
      : null;
  const loadRaw = Number(utilization.gpuLoadPercent ?? utilization.gpu_load_percent);
  const loadPct = Number.isFinite(loadRaw) ? Math.max(0, Math.min(100, loadRaw)) : null;
  return { memUsedPct, loadPct };
}

/**
 * Build a Map keyed by worker.id for O(1) lookup. Cheap to call from useMemo
 * at the app level so every consumer shares one instance.
 */
export function buildWorkersById(workers) {
  if (!Array.isArray(workers) || workers.length === 0) {
    return new Map();
  }
  return new Map(workers.map((worker) => [worker.id, worker]));
}
