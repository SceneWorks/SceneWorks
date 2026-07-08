import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { qualityChoices, GPU_REQUIRED_JOB_TYPES, errorStatuses } from "./jobTypes.js";
import { withAppContext, FakeEventSource, response, settle } from "./main.testSupport.jsx";

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("uses one worker-canonical quality enum across studios (no draft/final drift)", () => {
    // sc-1657: PresetManager previously used draft/balanced/final while VideoStudio
    // and the worker (video_adapters.py step maps) use fast/balanced/best, so saved
    // presets didn't match the studio control. Pin the shared values.
    expect(qualityChoices.map(([value]) => value)).toEqual(["fast", "balanced", "best"]);
    expect(qualityChoices.map(([, label]) => label)).toEqual(["Draft", "Balanced", "Final"]);
  });

  it("keeps the centralized job-type/status enums consistent", () => {
    // GPU-required job types must stay aligned with the Rust dispatch gate
    // (jobs_store.rs::job_requires_gpu); errorStatuses is terminal minus completed.
    expect(GPU_REQUIRED_JOB_TYPES.has("video_generate")).toBe(true);
    expect(GPU_REQUIRED_JOB_TYPES.has("model_download")).toBe(false);
    expect([...errorStatuses].sort()).toEqual(["canceled", "failed", "interrupted"]);
    expect(errorStatuses.has("completed")).toBe(false);
  });

  it("filters stale and placeholder-only GPU workers from the queue view", async () => {
    global.fetch.mockImplementation((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/workers")) {
        return Promise.resolve(
          response([
            {
              id: "python-gpu-0",
              gpuId: "0",
              gpuName: "Fixture GPU 0",
              status: "idle",
              capabilities: ["placeholder", "gpu", "image_generate"],
              utilization: { memoryTotalMb: 24576, memoryUsedMb: 4096, memoryFreeMb: 20480, gpuLoadPercent: 12 },
            },
            {
              id: "rust-gpu-1",
              gpuId: "1",
              gpuName: "Rust placeholder GPU",
              status: "idle",
              capabilities: ["placeholder", "gpu", "nvidia"],
            },
            {
              id: "stale-gpu-2",
              gpuId: "2",
              gpuName: "Stale GPU",
              status: "offline",
              capabilities: ["placeholder", "gpu", "image_generate"],
            },
            {
              id: "rust-cpu",
              gpuId: "cpu",
              gpuName: "Rust CPU utility worker",
              status: "idle",
              capabilities: ["placeholder", "cpu", "model_download"],
            },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue").click();
    });
    await settle();

    expect(container.textContent).toContain("Fixture GPU 0");
    expect(container.textContent).toContain("20.0 GB");
    expect(container.textContent).toContain("4.0 GB / 24.0 GB");
    expect(container.textContent).toContain("12%");
    expect(container.textContent).not.toContain("Rust CPU utility worker");
    expect(container.textContent).not.toContain("Rust placeholder GPU");
    expect(container.textContent).not.toContain("Stale GPU");
    expect([...document.body.querySelector("#queue-gpu").options].map((option) => option.value)).toEqual(["auto", "0"]);
  });

  it("shows queued job cancellation from the action response even when the list refresh is stale", async () => {
    const queuedJob = {
      id: "job-queued",
      type: "image_generate",
      status: "queued",
      stage: "queued",
      progress: 0,
      projectId: "project-1",
      projectName: "Project 1",
      requestedGpu: "auto",
      payload: { prompt: "mist" },
      attempts: 1,
      cancelRequested: false,
      createdAt: "2026-05-19T09:00:00Z",
      updatedAt: "2026-05-19T09:00:00Z",
    };
    const canceledJob = {
      ...queuedJob,
      status: "canceled",
      stage: "canceled",
      progress: 1,
      cancelRequested: true,
      message: "Canceled before a worker started.",
      updatedAt: "2026-05-19T09:01:00Z",
    };
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/job-queued/cancel") && options.method === "POST") {
        return Promise.resolve(response(canceledJob));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project 1" }]));
      }
      if (path.endsWith("/workers")) {
        return Promise.resolve(response([]));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response([queuedJob]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue").click();
    });
    await settle();

    expect(container.textContent).toContain("Queued");

    await act(async () => {
      [...document.body.querySelectorAll(".worker-progress-card__actions button")].find((button) => button.textContent === "Cancel").click();
    });
    await settle();

    expect(container.textContent).toContain("Cancelled");
    expect(container.textContent).toContain("Canceled before a worker started.");
    expect(container.textContent).not.toContain("Queued");
    expect(container.textContent).not.toContain("Waiting for an available GPU worker.");
  });

  it("keeps fresher SSE job state when a post-action refresh returns stale data", async () => {
    const failedJob = {
      id: "job-failed",
      type: "image_generate",
      status: "failed",
      stage: "failed",
      progress: 1,
      projectId: "project-1",
      projectName: "Project 1",
      requestedGpu: "auto",
      payload: { prompt: "mist" },
      attempts: 1,
      cancelRequested: false,
      createdAt: "2026-05-19T09:00:00Z",
      updatedAt: "2026-05-19T09:00:00Z",
    };
    const retryJob = {
      ...failedJob,
      id: "job-retry",
      status: "running",
      stage: "generating",
      progress: 0.1,
      attempts: 2,
      updatedAt: "2026-05-19T09:01:00Z",
    };
    const fresherRetryJob = {
      ...retryJob,
      progress: 0.4,
      message: "Worker advanced during refresh.",
      updatedAt: "2026-05-19T09:02:00Z",
    };
    let jobsRequestCount = 0;
    let resolvePostRetryJobs;
    const postRetryJobs = new Promise((resolve) => {
      resolvePostRetryJobs = resolve;
    });
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/job-failed/retry") && options.method === "POST") {
        return Promise.resolve(response(retryJob));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project 1" }]));
      }
      if (path.endsWith("/workers")) {
        return Promise.resolve(response([]));
      }
      if (path.endsWith("/jobs")) {
        jobsRequestCount += 1;
        return jobsRequestCount === 1 ? Promise.resolve(response([failedJob])) : postRetryJobs;
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue").click();
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll(".worker-progress-card__actions button")].find((button) => button.textContent === "Retry").click();
      await Promise.resolve();
    });
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({ data: JSON.stringify(fresherRetryJob) });
    });
    await act(async () => {
      resolvePostRetryJobs(response([retryJob]));
    });
    await settle();

    expect(document.body.querySelector(".progress-track")?.getAttribute("aria-label")).toBe("40% complete");
    expect(container.textContent).toContain("Worker advanced during refresh.");
  });

  it("explains queued GPU jobs that are waiting on capability or busy workers", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Project 1" },
          createPlaceholderJob: (event) => event.preventDefault(),
          filteredJobs: [
            {
              id: "job-waiting",
              type: "image_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "mist", model: "z_image_turbo" },
              attempts: 1,
            },
            {
              id: "job-blocked",
              type: "video_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "clip" },
              attempts: 1,
            },
            {
              id: "job-download",
              type: "model_download",
              status: "downloading",
              stage: "downloading",
              progress: 0.4,
              projectId: null,
              projectName: null,
              requestedGpu: "auto",
              assignedGpu: "cpu",
              payload: {
                modelId: "qwen_image_edit",
                modelName: "Qwen Image Edit",
                repo: "Qwen/Qwen-Image-Edit",
              },
              attempts: 1,
            },
            {
              id: "job-waiting-download",
              type: "image_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "edit", model: "qwen_image_edit" },
              attempts: 1,
            },
            {
              id: "job-dependency",
              type: "image_generate",
              status: "running",
              stage: "generating",
              progress: 0.5,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "0",
              assignedGpu: "0",
              payload: { prompt: "source" },
              attempts: 1,
            },
            {
              id: "job-waiting-dependency",
              type: "image_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "dependent", dependsOnJobId: "job-dependency" },
              attempts: 1,
            },
          ],
          gpuOptions: ["auto", "0"],
          jobAction: () => {},
          jobPrompt: "Placeholder generation",
          projectFilter: "all",
          projects: [{ id: "project-1", name: "Project 1" }],
          requestedGpu: "auto",
          setJobPrompt: () => {},
          setProjectFilter: () => {},
          setRequestedGpu: () => {},
          visibleWorkers: [
            {
              id: "misregistered-cpu",
              gpuId: "cpu",
              gpuName: "CPU worker",
              status: "idle",
              currentJobId: null,
              capabilities: ["placeholder", "cpu", "video_generate"],
              loadedModels: [],
            },
            {
              id: "python-gpu-0",
              gpuId: "0",
              gpuName: "Fixture GPU 0",
              status: "busy",
              currentJobId: "job-active",
              capabilities: ["placeholder", "gpu", "image_generate"],
              loadedModels: ["z_image_turbo"],
            },
          ],
          },
          <QueueScreen />,
        ),
      );
    });

    expect(container.textContent).toContain("Waiting: an eligible worker is busy.");
    expect(container.textContent).toContain("Blocked: no active worker supports video generate.");
    expect(container.textContent).toContain("Waiting for model download Qwen Image Edit to finish.");
    expect(container.textContent).toContain("Waiting for dependency job-dependency to finish.");
    expect(container.textContent).toContain("Warm: z_image_turbo");
  });

  it("updates Queue GPU utilization when worker props change", async () => {
    const queueProps = {
      activeProject: { id: "project-1", name: "Project 1" },
      createPlaceholderJob: (event) => event.preventDefault(),
      filteredJobs: [],
      gpuOptions: ["auto", "0"],
      jobAction: () => {},
      jobPrompt: "Placeholder generation",
      projectFilter: "all",
      projects: [{ id: "project-1", name: "Project 1" }],
      requestedGpu: "auto",
      setJobPrompt: () => {},
      setProjectFilter: () => {},
      setRequestedGpu: () => {},
    };
    const worker = {
      id: "python-gpu-0",
      gpuId: "0",
      gpuName: "Fixture GPU 0",
      status: "idle",
      capabilities: ["placeholder", "gpu", "image_generate"],
      loadedModels: [],
      utilization: { memoryTotalMb: 24576, memoryUsedMb: 4096, memoryFreeMb: 20480, gpuLoadPercent: 12 },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext({ ...queueProps, visibleWorkers: [worker] }, <QueueScreen />));
    });

    expect(container.textContent).toContain("20.0 GB");
    expect(container.textContent).toContain("12%");

    await act(async () => {
      root.render(
        withAppContext(
          {
            ...queueProps,
            visibleWorkers: [
              {
                ...worker,
                utilization: { memoryTotalMb: 24576, memoryUsedMb: 12288, memoryFreeMb: 12288, gpuLoadPercent: 67 },
              },
            ],
          },
          <QueueScreen />,
        ),
      );
    });

    expect(container.textContent).toContain("12.0 GB / 24.0 GB");
    expect(container.textContent).toContain("67%");
    expect(container.textContent).not.toContain("20.0 GB");
  });

});
