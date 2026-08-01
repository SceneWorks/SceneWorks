import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "./api.js";
import { CREATE_JOB_DEFINITIONS, makeCreateJob } from "./createJob.js";

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
});
