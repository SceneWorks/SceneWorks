import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "./api.js";
import { CREATE_JOB_DEFINITIONS, makeCreateJob } from "./createJob.js";
import {
  createModelLibraryGate,
  setModelLibraryHandler,
  MODEL_LIBRARY_UNAVAILABLE_CODE,
} from "./modelLibrary.js";

vi.mock("./api.js", () => ({
  apiFetch: vi.fn(),
}));

const project = { id: "project-7", name: "Project Seven" };

const cases = [
  {
    key: "video",
    args: [{ prompt: "video" }, { navigateToQueue: true }],
    path: "/api/v1/video/jobs",
    body: {
      prompt: "video",
      projectId: "project-7",
      projectName: "Project Seven",
      requestedGpu: "gpu-2",
    },
  },
  {
    key: "audio",
    args: [{ prompt: "audio" }],
    path: "/api/v1/audio/jobs",
    body: {
      prompt: "audio",
      projectId: "project-7",
      projectName: "Project Seven",
      requestedGpu: "gpu-2",
    },
  },
  {
    key: "image",
    args: [{ prompt: "image" }],
    path: "/api/v1/image/jobs",
    body: {
      prompt: "image",
      projectId: "project-7",
      projectName: "Project Seven",
      requestedGpu: "gpu-2",
    },
  },
  {
    key: "videoUpscale",
    args: [{ sourceAssetId: "asset-1", factor: 2 }],
    path: "/api/v1/jobs",
    body: {
      type: "video_upscale",
      projectId: "project-7",
      projectName: "Project Seven",
      requestedGpu: "gpu-2",
      payload: { sourceAssetId: "asset-1", factor: 2, projectId: "project-7" },
    },
  },
  {
    key: "vqa",
    args: [{ id: "asset-2" }, "What is here?", 384],
    path: "/api/v1/image/vqa/jobs",
    body: {
      projectId: "project-7",
      projectName: "Project Seven",
      sourceAssetId: "asset-2",
      question: "What is here?",
      maxNewTokens: 384,
      requestedGpu: "gpu-2",
    },
  },
  {
    key: "interleave",
    args: [{ prompt: "sequence", assetIds: ["a", "b"] }],
    path: "/api/v1/image/interleave/jobs",
    body: {
      prompt: "sequence",
      assetIds: ["a", "b"],
      projectId: "project-7",
      projectName: "Project Seven",
      requestedGpu: "gpu-2",
    },
  },
];

describe("makeCreateJob", () => {
  beforeEach(() => {
    apiFetch.mockReset();
  });

  it.each(cases)("preserves the $key route and request body", async ({ key, args, path, body }) => {
    const job = { id: `job-${key}`, status: "queued", createdAt: "2026-07-25T12:00:00Z" };
    apiFetch.mockResolvedValue(job);
    const setError = vi.fn();
    let jobs = [];
    const creator = makeCreateJob({
      definition: CREATE_JOB_DEFINITIONS[key],
      token: "token-1",
      project,
      requestedGpu: "gpu-2",
      setJobs: (update) => {
        jobs = update(jobs);
      },
      setError,
    });

    await expect(creator(...args)).resolves.toBe(job);
    expect(apiFetch).toHaveBeenCalledWith(path, "token-1", {
      method: "POST",
      body: JSON.stringify(body),
    });
    expect(jobs).toEqual([job]);
    expect(setError).toHaveBeenLastCalledWith("");
  });

  it("keeps per-creator post-success behavior", async () => {
    apiFetch.mockResolvedValue({
      id: "job-video",
      status: "queued",
      createdAt: "2026-07-25T12:00:00Z",
    });
    const afterCreate = vi.fn();
    const creator = makeCreateJob({
      definition: CREATE_JOB_DEFINITIONS.video,
      token: "",
      project,
      requestedGpu: "auto",
      setJobs: vi.fn(),
      setError: vi.fn(),
      afterCreate,
    });
    const options = { navigateToQueue: true };

    await creator({ prompt: "go" }, options);

    expect(afterCreate).toHaveBeenCalledWith(
      expect.objectContaining({ id: "job-video" }),
      { prompt: "go" },
      options,
    );
  });

  it("can hold or cancel a request before the POST", async () => {
    const beforeCreate = vi.fn(async () => false);
    const creator = makeCreateJob({
      definition: CREATE_JOB_DEFINITIONS.image,
      token: "",
      project,
      requestedGpu: "auto",
      setJobs: vi.fn(),
      setError: vi.fn(),
      beforeCreate,
    });
    const payload = { prompt: "wait for the decision" };

    await expect(creator(payload)).resolves.toBeNull();

    expect(beforeCreate).toHaveBeenCalledWith(payload);
    expect(apiFetch).not.toHaveBeenCalled();
  });

  it("retains the missing-project and request-error contracts", async () => {
    const setError = vi.fn();
    const missingProjectCreator = makeCreateJob({
      definition: CREATE_JOB_DEFINITIONS.image,
      token: "",
      project: null,
      requestedGpu: "auto",
      setJobs: vi.fn(),
      setError,
    });
    await expect(missingProjectCreator({ prompt: "x" })).resolves.toBeNull();
    expect(setError).toHaveBeenLastCalledWith("Create or open a project first.");
    expect(apiFetch).not.toHaveBeenCalled();

    apiFetch.mockRejectedValue(new Error("network down"));
    const failingCreator = makeCreateJob({
      definition: CREATE_JOB_DEFINITIONS.image,
      token: "",
      project,
      requestedGpu: "auto",
      setJobs: vi.fn(),
      setError,
    });
    await expect(failingCreator({ prompt: "x" })).resolves.toBeNull();
    expect(setError).toHaveBeenLastCalledWith("network down");
  });

  // sc-19709. The end-to-end shape of the recovery, at the one choke point every generation
  // submission passes through: a typed unavailable-library rejection becomes a prompt (not an
  // error banner), and resuming it re-POSTs the SAME submission exactly once.
  describe("unavailable model library", () => {
    function unavailableRejection() {
      return {
        code: MODEL_LIBRARY_UNAVAILABLE_CODE,
        message: "Model 'z_image' is installed on an external model library…",
        context: {
          schemaVersion: 1,
          availability: "installed_external_unavailable",
          modelId: "z_image",
          modelName: "Z-Image",
          configuredLibraryPath: "/Volumes/Models/hf/hub",
          expectedLibraryPath: "/Volumes/Models/hf/hub",
          expectedVolumeId: "macos-volume:abc",
        },
      };
    }

    it("hands the blocked submission to the prompt and resumes it exactly once", async () => {
      const job = { id: "job-image", status: "queued", createdAt: "2026-07-25T12:00:00Z" };
      apiFetch.mockRejectedValueOnce(unavailableRejection()).mockResolvedValue(job);
      const probe = vi.fn(async () => ({ available: true }));
      const gate = createModelLibraryGate({ probe });
      const unregister = setModelLibraryHandler(gate.block);
      const setError = vi.fn();
      let jobs = [];
      const creator = makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.image,
        token: "token-1",
        project,
        requestedGpu: "gpu-2",
        setJobs: (update) => {
          jobs = update(jobs);
        },
        setError,
      });

      await expect(creator({ prompt: "a lighthouse" })).resolves.toBeNull();
      // No raw error surface: the prompt owns it, and it names the model from the typed context.
      expect(setError).not.toHaveBeenCalledWith(
        expect.stringContaining("external model library"),
      );
      expect(gate.getState()).toMatchObject({ status: "blocked" });
      expect(gate.getState().context.modelName).toBe("Z-Image");
      expect(apiFetch).toHaveBeenCalledTimes(1);

      // Reconnect: the queued submission resumes, and the double-click plus a concurrent
      // reconnect event produce ONE additional POST, not three.
      const resumed = await Promise.all([gate.retry(), gate.retry(), gate.retry({ auto: true })]);
      expect(resumed.filter(Boolean)).toEqual([job]);
      expect(apiFetch).toHaveBeenCalledTimes(2);
      expect(apiFetch).toHaveBeenLastCalledWith("/api/v1/image/jobs", "token-1", {
        method: "POST",
        body: JSON.stringify({
          prompt: "a lighthouse",
          projectId: "project-7",
          projectName: "Project Seven",
          requestedGpu: "gpu-2",
        }),
      });
      expect(jobs).toEqual([job]);
      unregister();
    });

    it("leaves no queued submission behind when the prompt is cancelled", async () => {
      apiFetch.mockRejectedValue(unavailableRejection());
      const gate = createModelLibraryGate({ probe: async () => ({ available: true }) });
      const unregister = setModelLibraryHandler(gate.block);
      const creator = makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.image,
        token: "",
        project,
        requestedGpu: "auto",
        setJobs: vi.fn(),
        setError: vi.fn(),
      });

      await creator({ prompt: "x" });
      gate.cancel();
      expect(gate.getState()).toMatchObject({ status: "idle", context: null });
      apiFetch.mockClear();
      await gate.retry();
      expect(apiFetch).not.toHaveBeenCalled();
      unregister();
    });

    it("still reports ordinary failures when no prompt is registered", async () => {
      apiFetch.mockRejectedValue(unavailableRejection());
      const setError = vi.fn();
      const creator = makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.image,
        token: "",
        project,
        requestedGpu: "auto",
        setJobs: vi.fn(),
        setError,
      });
      await expect(creator({ prompt: "x" })).resolves.toBeNull();
      expect(setError).toHaveBeenCalledWith(
        "Model 'z_image' is installed on an external model library…",
      );
    });
  });
});
