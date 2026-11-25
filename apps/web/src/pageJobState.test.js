import { expect, it } from "vitest";
import { retainPageJobs } from "./pageJobState.js";
import { isEditorJob } from "./jobTypes.js";

function jobs(count) {
  return Array.from({ length: count }, (_, index) => ({
    id: `job-${index}`, type: "image_generate", projectId: "project", status: "completed",
    stage: "completed", progress: 1, revision: 3,
    payload: { prompt: `prompt ${index}` },
    result: { assets: [{ id: `asset-${index}` }], previewFrame: { dataUrl: "data:image/png;base64,large" } },
  }));
}

it("releases every Queue payload/output/preview on Assets while retaining progress and revisions", () => {
  const source = jobs(1000);
  const queue = retainPageJobs(source, { route: "Queue", projectId: "project" });
  expect(queue).toBe(source);
  const assets = retainPageJobs(queue, { route: "Library", projectId: "project" });
  expect(assets.every((job) => Object.keys(job.result).length === 0 && Object.keys(job.payload).length === 0)).toBe(true);
  expect(assets.map(({ id, status, stage, progress, revision }) => ({ id, status, stage, progress, revision })))
    .toEqual(source.map(({ id, status, stage, progress, revision }) => ({ id, status, stage, progress, revision })));
  expect(retainPageJobs(assets, { route: "Library", projectId: "project" })).toBe(assets);
});

it("bounds Studio history independently of batch length, preserving every active job", () => {
  const source = jobs(1000);
  source.unshift({ ...source[0], id: "active", status: "running", progress: 0.63 });
  const retained = retainPageJobs(source, { route: "Image", projectId: "project" });
  expect(retained.filter((job) => job.result.assets)).toHaveLength(26);
  expect(retained[0]).toBe(source[0]);
  const settings = retainPageJobs(retained, { route: "Settings", projectId: "project" });
  expect(settings[0]).toMatchObject({ id: "active", status: "running", progress: 0.63, payload: {}, result: {} });
});

it("keeps orphaned editor-op results only until their cleanup owner releases them", () => {
  const source = jobs(2);
  const policy = { route: "Library", projectId: "project", ownsScratch: (id) => id === "job-1" };
  const retained = retainPageJobs(source, policy);
  expect(retained[0].result).toEqual({});
  expect(retained[1]).toBe(source[1]);
  expect(retainPageJobs(retained, { ...policy, ownsScratch: () => false })[1].result).toEqual({});
});

it("keeps global keypoint extraction results for capture while a workspace is selected", () => {
  const extraction = { id: "capture", projectId: "project_global_keypoints", type: "kps_extract",
    status: "completed", payload: { sourcePath: "staged-face.png" }, result: { detected: true, kps: [[10, 20]] } };
  expect(retainPageJobs([extraction], { route: "Keypoints", projectId: "workspace" })[0]).toBe(extraction);
  expect(retainPageJobs([extraction], { route: "Library", projectId: "workspace" })[0].result).toEqual({});
});

it("preserves Editor activity attribution after releasing a timeline generation payload", () => {
  const job = { ...jobs(1)[0], type: "video_generate", status: "running",
    payload: { prompt: "private timeline prompt", advanced: { timelineAction: "extend", timelineContext: { timelineId: "timeline" } } } };
  const policy = { route: "Library", projectId: "project" };
  const [summary] = retainPageJobs([job], policy);
  expect(isEditorJob(summary)).toBe(true);
  expect(summary.payload).toEqual({ advanced: { timelineAction: "extend" } });
  expect(summary.result).toEqual({});
  expect(retainPageJobs([summary], policy)[0]).toBe(summary);
});
