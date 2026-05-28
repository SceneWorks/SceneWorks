import { describe, expect, it } from "vitest";
import {
  buildWorkersById,
  deriveWorkerHardware,
  findWorkerForJob,
  liveMeters,
} from "./workers.js";

const nvidiaWorker = {
  id: "worker-cuda-1",
  gpuId: "gpu-0",
  gpuName: "NVIDIA GeForce RTX 4090",
  capabilities: ["gpu", "image_generate"],
  utilization: { memoryUsedMb: 12000, memoryTotalMb: 24000, gpuLoadPercent: 73.4 },
};

const appleWorker = {
  id: "worker-mps-1",
  gpuId: "mps-0",
  gpuName: "Apple M2 Ultra",
  capabilities: ["gpu", "image_generate"],
  utilization: { memoryUsedMb: 30000, memoryTotalMb: 64000, gpuLoadPercent: 41 },
};

const cpuWorker = {
  id: "worker-cpu-1",
  gpuId: "cpu-0",
  gpuName: null,
  capabilities: ["cpu", "prompt_refine"],
  utilization: null,
};

describe("deriveWorkerHardware", () => {
  it("identifies NVIDIA GPUs as CUDA", () => {
    expect(deriveWorkerHardware(nvidiaWorker)).toMatchObject({
      device: "GPU",
      vendor: "NVIDIA",
      architecture: "cuda",
    });
  });

  it("identifies Apple Silicon GPUs as MPS", () => {
    expect(deriveWorkerHardware(appleWorker)).toMatchObject({
      device: "GPU",
      vendor: "Apple",
      architecture: "mps",
    });
  });

  it("identifies CPU workers without a vendor/architecture", () => {
    expect(deriveWorkerHardware(cpuWorker)).toEqual({
      device: "CPU",
      vendor: null,
      architecture: null,
      gpuLabel: null,
    });
  });

  it("returns blanks for unknown vendor labels", () => {
    const worker = {
      gpuName: "Some Other Accelerator",
      capabilities: ["gpu"],
    };
    expect(deriveWorkerHardware(worker)).toMatchObject({
      device: "GPU",
      vendor: null,
      architecture: null,
    });
  });

  it("returns all-null for a missing worker", () => {
    expect(deriveWorkerHardware(null)).toEqual({
      device: null,
      vendor: null,
      architecture: null,
      gpuLabel: null,
    });
  });
});

describe("findWorkerForJob", () => {
  const workers = [nvidiaWorker, appleWorker, cpuWorker];

  it("matches by workerId first", () => {
    const job = { workerId: "worker-mps-1", assignedGpu: "gpu-0" };
    expect(findWorkerForJob(job, workers)).toBe(appleWorker);
  });

  it("falls back to assignedGpu when workerId is absent", () => {
    const job = { workerId: null, assignedGpu: "gpu-0" };
    expect(findWorkerForJob(job, workers)).toBe(nvidiaWorker);
  });

  it("ignores assignedGpu=auto", () => {
    const job = { workerId: null, assignedGpu: "auto" };
    expect(findWorkerForJob(job, workers)).toBeNull();
  });

  it("returns null when no worker matches", () => {
    const job = { workerId: "missing", assignedGpu: "missing" };
    expect(findWorkerForJob(job, workers)).toBeNull();
  });

  it("returns null for empty workers", () => {
    expect(findWorkerForJob({ workerId: "x" }, [])).toBeNull();
    expect(findWorkerForJob({ workerId: "x" }, null)).toBeNull();
  });
});

describe("liveMeters", () => {
  it("computes a memory percentage from used/total", () => {
    expect(liveMeters(nvidiaWorker)).toEqual({ memUsedPct: 50, loadPct: 73.4 });
  });

  it("clamps load percent to 0..100", () => {
    const worker = { utilization: { memoryUsedMb: 0, memoryTotalMb: 0, gpuLoadPercent: 150 } };
    expect(liveMeters(worker)).toEqual({ memUsedPct: null, loadPct: 100 });
  });

  it("returns nulls when utilization is missing", () => {
    expect(liveMeters(cpuWorker)).toEqual({ memUsedPct: null, loadPct: null });
    expect(liveMeters(null)).toEqual({ memUsedPct: null, loadPct: null });
  });

  it("accepts snake_case utilization keys", () => {
    const worker = { utilization: { memory_used_mb: 1000, memory_total_mb: 2000, gpu_load_percent: 25 } };
    expect(liveMeters(worker)).toEqual({ memUsedPct: 50, loadPct: 25 });
  });
});

describe("buildWorkersById", () => {
  it("indexes workers by id", () => {
    const index = buildWorkersById([nvidiaWorker, appleWorker]);
    expect(index.get("worker-cuda-1")).toBe(nvidiaWorker);
    expect(index.get("worker-mps-1")).toBe(appleWorker);
  });

  it("returns an empty map for missing input", () => {
    expect(buildWorkersById(null).size).toBe(0);
    expect(buildWorkersById([]).size).toBe(0);
  });
});
