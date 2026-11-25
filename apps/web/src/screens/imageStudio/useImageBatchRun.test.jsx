import React, { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { useImageBatchController, useImageBatchRun } from "./useImageBatchRun.js";
import { validatePortableRequest } from "../../portableValidation.js";
vi.mock("../../portableValidation.js", () => ({
  validatePortableRequest: vi.fn(async () => ({ state: "valid" })),
}));
let root, container;
beforeEach(() => { globalThis.IS_REACT_ACT_ENVIRONMENT = true; ({ root, container } = mountRoot()); });
afterEach(() => unmountRoot(root, container));

it("continues a bounded batch after Image Studio unmounts and restores progress on return", async () => {
  vi.mocked(validatePortableRequest).mockResolvedValue({ state: "valid" });
  const createImageJob = vi.fn(async (request) => ({ id: request.prompt, status: "queued" }));
  const jobAction = vi.fn();
  let navigate, updateJobs, start, controller, running;
  function Page({ backgroundImageBatch }) {
    ({ runBatch: start } = useImageBatchRun({
      backgroundImageBatch, activeProject: { id: "project" }, jobs: [], jobAction,
      preflightError: () => "", setBatchError: vi.fn(), setBatchConfirmPending: vi.fn(),
      batchTotal: 2, BATCH_RENDER_CAP: 100, batchJobCount: 2, BATCH_ACTIVE_JOB_LIMIT: 1,
      batchPrompts: ["first", "second"], batchVariables: {}, resolutionOptions: [],
      structuredPromptModel: false, selectedModel: null,
      buildBatchJobRequest: (prompt) => ({ prompt, count: 1 }), createImageJob,
    }));
    return <span data-page="image">{backgroundImageBatch.batchRunProgress?.done ?? 0}</span>;
  }
  function Shell() {
    const [visible, setVisible] = React.useState(true);
    const [jobs, setJobs] = React.useState([]);
    navigate = setVisible;
    updateJobs = setJobs;
    controller = useImageBatchController({ activeProject: { id: "project" }, jobs, jobAction });
    return visible ? <Page backgroundImageBatch={controller} /> : <span data-page="queue" />;
  }
  await act(async () => root.render(<Shell />));
  await act(async () => { running = start(); });
  expect(createImageJob).toHaveBeenCalledTimes(1);
  await act(async () => navigate(false));
  expect(container.querySelector('[data-page="image"]')).toBeNull();
  await act(async () => updateJobs([{ id: "first", status: "completed" }]));
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(2));
  await act(async () => { await running; updateJobs([{ id: "first", status: "completed" }, { id: "second", status: "completed" }]); });
  expect(controller.batchRunProgress).toMatchObject({ total: 2, completed: 2, done: 2 });
  expect(controller.batchRun.sizes).toEqual({});
  expect(jobAction).not.toHaveBeenCalled();
  await act(async () => navigate(true));
  expect(container.querySelector('[data-page="image"]').textContent).toBe("2");
});

it("continues the committed project's batch while a different project render suspends", async () => {
  vi.mocked(validatePortableRequest).mockClear();
  let changeProject, resolveFirst, release, running;
  let ready = false;
  const pending = new Promise((resolve) => { release = () => { ready = true; resolve(); }; });
  const createImageJob = vi.fn()
    .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
    .mockResolvedValue({ id: "second", status: "queued" });
  const cancelA = vi.fn(), cancelB = vi.fn();
  function Owner({ project }) {
    const [batchRun, setBatchRun] = React.useState(null);
    const { runBatch } = useImageBatchRun({ batchRun, setBatchRun, activeProject: { id: project },
      jobs: [], jobAction: project === "A" ? cancelA : cancelB, BATCH_ACTIVE_JOB_LIMIT: 10,
      preflightError: () => "", setBatchError: () => {}, batchTotal: 2, BATCH_RENDER_CAP: 100,
      setBatchConfirmPending: () => {}, batchPrompts: ["one", "two"], batchVariables: {},
      resolutionOptions: [], structuredPromptModel: false, batchJobCount: 2, selectedModel: null,
      buildBatchJobRequest: (prompt) => ({ prompt, count: 1 }), onMagicExpand: () => {}, createImageJob });
    if (project === "B" && !ready) throw pending;
    return <button onClick={() => { running = runBatch(); }}>Run</button>;
  }
  function Harness() {
    const [project, setProject] = React.useState("A");
    changeProject = setProject;
    return <React.Suspense fallback="loading"><Owner project={project} /></React.Suspense>;
  }
  await act(async () => root.render(<Harness />));
  await act(async () => container.querySelector("button").click());
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(1));
  expect(validatePortableRequest).toHaveBeenCalledWith(expect.objectContaining({
    kind: "image", payload: { prompt: "one", count: 1, projectId: "A" }, token: "", signal: expect.any(AbortSignal),
  }));
  await act(async () => React.startTransition(() => changeProject("B")));
  await act(async () => { resolveFirst({ id: "first", status: "queued" }); await running; });
  expect(createImageJob).toHaveBeenCalledTimes(2);
  expect(vi.mocked(validatePortableRequest).mock.calls.map(([request]) => request.payload.projectId))
    .toEqual(["A", "A"]);
  expect(cancelA).not.toHaveBeenCalled();
  expect(cancelB).not.toHaveBeenCalled();
  await act(async () => release());
  expect(cancelB.mock.calls.map(([job]) => job.id)).toEqual(["first", "second"]);
});

it("stops before enqueue when request 257 receives a typed refusal", async () => {
  vi.mocked(validatePortableRequest).mockClear();
  const refusal = { code: "invalid_generation_option", field: "width", detail: "Invalid request 257" };
  vi.mocked(validatePortableRequest).mockImplementation(async ({ payload }) => payload.prompt === "256"
    ? { state: "invalid", refusal } : { state: "valid" });
  const createImageJob = vi.fn(async (request) => ({ id: `job-${request.prompt}`, status: "queued" }));
  const prompts = Array.from({ length: 260 }, (_, index) => String(index));
  let run;
  function Owner() {
    const [batchRun, setBatchRun] = React.useState(null);
    ({ runBatch: run } = useImageBatchRun({
      batchRun, setBatchRun, activeProject: { id: "project" }, jobs: [], jobAction: vi.fn(),
      BATCH_ACTIVE_JOB_LIMIT: 300, preflightError: () => "", setBatchError: vi.fn(),
      batchTotal: 260, BATCH_RENDER_CAP: 1000, setBatchConfirmPending: vi.fn(),
      batchPrompts: prompts, batchVariables: {}, resolutionOptions: [],
      structuredPromptModel: false, batchJobCount: 260, selectedModel: null,
      buildBatchJobRequest: (prompt) => ({ prompt, count: 1 }),
      onMagicExpand: vi.fn(), createImageJob,
    }));
    return null;
  }
  await act(async () => root.render(<Owner />));
  await act(async () => run(true));
  expect(validatePortableRequest).toHaveBeenCalledTimes(257);
  expect(createImageJob).toHaveBeenCalledTimes(256);
  expect(createImageJob.mock.calls.every(([request]) => request.prompt !== "256")).toBe(true);
  vi.mocked(validatePortableRequest).mockReset();
});
