import { describe, expect, it, vi } from "vitest";
import { createJobsStore } from "./jobStore.js";

describe("jobs store", () => {
  it("derives queue counts, worker selectors, and local generation stacks", () => {
    const store = createJobsStore({
      activeProjectId: "project-1",
      jobs: [
        {
          id: "job-image",
          type: "image_generate",
          status: "running",
          projectId: "project-1",
          createdAt: "2026-01-02T00:00:00.000Z",
        },
        {
          id: "job-other-project",
          type: "image_generate",
          status: "running",
          projectId: "project-2",
          createdAt: "2026-01-01T00:00:00.000Z",
        },
      ],
      workers: [
        { id: "worker-live", status: "idle", gpuId: "gpu-0", capabilities: ["gpu", "image_generate"] },
        { id: "worker-placeholder", status: "idle", gpuId: "gpu-1", capabilities: ["placeholder", "gpu", "nvidia"] },
        { id: "worker-offline", status: "offline", gpuId: "gpu-2", capabilities: ["gpu"] },
      ],
    });

    store.actions.rememberLocalGenerationJob("image", { id: "job-image" });
    store.actions.setQueueSummary({ counts: { queued: 10, active: 10 }, activeJobs: [{ id: "job-image" }] });
    store.actions.setProjectFilter("project-1");

    const snapshot = store.getSnapshot();
    expect(snapshot.queueCounts.active).toBe(1);
    expect(snapshot.queueCounts.queued).toBe(10);
    expect(snapshot.filteredJobs.map((job) => job.id)).toEqual(["job-image"]);
    expect(snapshot.visibleWorkers.map((worker) => worker.id)).toEqual(["worker-live"]);
    expect(snapshot.gpuOptions).toEqual(["auto", "gpu-0"]);
    expect(snapshot.workersById.get("worker-live").gpuId).toBe("gpu-0");
    expect(snapshot.imageLocalJobs.map((job) => job.id)).toEqual(["job-image"]);
  });

  it("notifies subscribers when jobs change", () => {
    const store = createJobsStore();
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);

    store.actions.upsertJob({
      id: "job-1",
      type: "image_generate",
      status: "queued",
      createdAt: "2026-01-01T00:00:00.000Z",
    });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().jobs.map((job) => job.id)).toEqual(["job-1"]);
    unsubscribe();
    store.actions.upsertJob({
      id: "job-2",
      type: "image_generate",
      status: "queued",
      createdAt: "2026-01-02T00:00:00.000Z",
    });
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
