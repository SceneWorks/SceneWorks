// Engine + GPU summary from the live worker registry — the one signal available in every
// deployment (desktop, Docker, remote browser). Shared by the advanced Settings screen's
// "This machine" group and Simple Settings' Device group so both read the identical
// derivation (Settings-page redesign handoff); it used to live in simple/SimpleSettings.jsx.
//
// An MLX worker means Apple Silicon unified memory; anything else is the discrete-GPU
// (candle) lane. Unknown values read "Detecting…" rather than a fabricated placeholder.
export function describeDevice(workers, macCapabilities) {
  const gpuWorkers = (workers ?? []).filter((worker) => worker?.gpuId && worker.gpuId !== "cpu");
  const mlx = gpuWorkers.find((worker) => worker.gpuId === "mlx");
  const primary = mlx ?? gpuWorkers[0] ?? null;
  const engine = mlx
    ? "MLX · Apple Silicon"
    : primary
      ? "Candle · discrete GPU"
      : macCapabilities?.platform === "macos"
        ? "MLX · Apple Silicon"
        : "Detecting…";
  if (!primary) {
    return { engine, gpu: "No worker connected" };
  }
  const totalMb = Number(primary.utilization?.memoryTotalMb);
  const memory = Number.isFinite(totalMb) && totalMb > 0 ? ` · ${(totalMb / 1024).toFixed(0)} GiB` : "";
  return { engine, gpu: `${primary.gpuName ?? `GPU ${primary.gpuId}`}${memory}` };
}
