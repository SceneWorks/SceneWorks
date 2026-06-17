import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { JobsProvider } from "../context/JobsContext.jsx";
import { createJobsStore } from "../jobs/jobStore.js";
import { QueueScreen } from "./QueueScreen.jsx";

describe("QueueScreen jobs provider boundary", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    if (root) {
      act(() => root.unmount());
    }
    document.body.removeChild(container);
    container = null;
    root = null;
  });

  it("renders jobs and workers from JobsProvider without job state in AppContext", () => {
    const store = createJobsStore({
      jobs: [
        {
          id: "job-queued",
          type: "image_generate",
          status: "queued",
          requestedGpu: "gpu-0",
          payload: { prompt: "selector job" },
          createdAt: "2026-01-01T00:00:00.000Z",
        },
      ],
      workers: [
        {
          id: "worker-gpu",
          status: "idle",
          gpuId: "gpu-0",
          gpuName: "NVIDIA RTX",
          capabilities: ["gpu", "image_generate"],
          utilization: { memoryFreeMb: 12000, memoryUsedMb: 4000, memoryTotalMb: 16000, gpuLoadPercent: 12 },
        },
      ],
    });
    const jobAction = vi.fn();
    root = createRoot(container);

    act(() => {
      root.render(
        <JobsProvider actions={{ createPlaceholderJob: (event) => event.preventDefault(), jobAction }} store={store}>
          <AppContext.Provider
            value={{
              activeProject: { id: "project-1", name: "Project" },
              assets: [],
              projects: [{ id: "project-1", name: "Project" }],
              requestedGpu: "auto",
              setPreviewAsset: vi.fn(),
              setRequestedGpu: vi.fn(),
            }}
          >
            <QueueScreen />
          </AppContext.Provider>
        </JobsProvider>,
      );
    });

    expect(container.textContent).toContain("Generate Image");
    expect(container.textContent).toContain("selector job");
    expect(container.textContent).toContain("NVIDIA RTX");
    expect(container.querySelector("#queue-gpu").textContent).toContain("gpu-0");
  });
});
