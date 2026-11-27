import React, { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { createImageBatchRuntime, useImageBatchController, useImageBatchRun } from "./useImageBatchRun.js";
import { validatePortableBatch } from "../../portableValidation.js";
import { emptyCaption, serializeCaption } from "../../ideogramCaption.js";
import { CREATE_JOB_DEFINITIONS, makeCreateJob } from "../../createJob.js";
import { submitJobRequest } from "../../api/jobs.js";
vi.mock("../../api/jobs.js", async (original) => ({ ...await original(), submitJobRequest: vi.fn() }));
vi.mock("../../portableValidation.js", () => ({
  MAX_VALIDATION_BATCH: 32,
  validatePortableBatch: vi.fn(),
}));
let root, container;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
  vi.mocked(validatePortableBatch).mockReset();
  vi.mocked(validatePortableBatch).mockImplementation(async ({ requests }) => requests.map(() => ({ state: "valid" })));
  vi.mocked(submitJobRequest).mockReset();
});
afterEach(() => unmountRoot(root, container));

it("continues a bounded batch after Image Studio unmounts and restores progress on return", async () => {
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
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(2));
  expect(validatePortableBatch).toHaveBeenCalledWith(expect.objectContaining({
    kind: "image", requests: [
      { prompt: "one", count: 1, projectId: "A" },
      { prompt: "two", count: 1, projectId: "A" },
    ], token: "", signal: expect.any(AbortSignal),
  }));
  await act(async () => React.startTransition(() => changeProject("B")));
  await act(async () => { resolveFirst({ id: "first", status: "queued" }); await running; });
  expect(createImageJob).toHaveBeenCalledTimes(2);
  expect(vi.mocked(validatePortableBatch).mock.calls.flatMap(([request]) => request.requests.map((payload) => payload.projectId)))
    .toEqual(["A", "A"]);
  expect(cancelA).not.toHaveBeenCalled();
  expect(cancelB).not.toHaveBeenCalled();
  await act(async () => release());
  expect(cancelB.mock.calls.map(([job]) => job.id).sort()).toEqual(["first", "second"]);
});

it("stops before enqueue when request 257 receives a typed refusal", async () => {
  const refusal = { code: "invalid_generation_option", field: "width", detail: "Invalid request 257" };
  vi.mocked(validatePortableBatch).mockImplementation(async ({ requests }) => requests.map((payload) => payload.prompt === "256"
    ? { state: "invalid", refusal } : { state: "valid" }));
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
  expect(validatePortableBatch).toHaveBeenCalledTimes(9);
  expect(createImageJob).toHaveBeenCalledTimes(256);
  expect(createImageJob.mock.calls.every(([request]) => request.prompt !== "256")).toBe(true);
});

function runtimeBatch(overrides = {}) {
  let state;
  const runtime = createImageBatchRuntime((next) => { state = next; });
  runtime.commit("project", [], vi.fn());
  runtime.attachProject();
  const plan = {
    BATCH_ACTIVE_JOB_LIMIT: 300, batchPrompts: ["first", "second", "third"],
    batchVariables: {}, resolutionOptions: [], structuredPromptModel: false,
    batchJobCount: 3, selectedModel: null, token: "batch-token",
    buildBatchJobRequest: (prompt, { resolution, promptToSend } = {}) => ({
      prompt: promptToSend ?? prompt, count: 1, ...resolution,
    }),
    onMagicExpand: vi.fn(),
    createImageJob: vi.fn(async (request) => ({ id: `job-${request.prompt}`, status: "queued" })),
    ...overrides,
  };
  return { runtime, plan, state: () => state };
}

it.each(["krea_2", "flux_dev", "sdxl"])("validates 148 direct prompts in five calls for %s", async (model) => {
  const prompts = Array.from({ length: 148 }, (_, index) => `prompt ${index}`);
  const { runtime, plan, state } = runtimeBatch({
    batchPrompts: prompts, batchJobCount: prompts.length, selectedModel: { id: model },
    buildBatchJobRequest: (prompt) => ({ model, prompt: `Styled ${prompt}`, count: 1, negativePrompt: "blur" }),
  });
  await runtime.runBatch(plan);
  const calls = vi.mocked(validatePortableBatch).mock.calls.map(([call]) => call);
  expect(calls.map((call) => call.requests.length)).toEqual([32, 32, 32, 32, 20]);
  expect(calls.flatMap((call) => call.requests)).toEqual(prompts.map((prompt) => ({
    model, prompt: `Styled ${prompt}`, count: 1, negativePrompt: "blur", projectId: "project",
  })));
  expect(calls.every((call) => call.kind === "image" && call.token === "batch-token" && call.signal instanceof AbortSignal)).toBe(true);
  expect(plan.createImageJob.mock.calls.map(([request]) => request.prompt)).toEqual(prompts.map((prompt) => `Styled ${prompt}`));
  expect(plan.onMagicExpand).not.toHaveBeenCalled();
  expect(state()).toMatchObject({ total: 148, submitted: 148, failed: 0, submitting: false });
});

it.each(["krea_2", "flux_dev", "sdxl"])("pipelines 15 submissions before the first response for %s", async (model) => {
  const prompts = Array.from({ length: 40 }, (_, index) => `prompt ${index}`);
  const pending = new Map();
  let active = 0, peak = 0;
  const createImageJob = vi.fn((request) => {
    active += 1;
    peak = Math.max(peak, active);
    return new Promise((resolve) => pending.set(request.prompt, () => {
      active -= 1;
      pending.delete(request.prompt);
      resolve({ id: `job-${request.prompt}`, status: "queued" });
    }));
  });
  const { runtime, plan, state } = runtimeBatch({
    selectedModel: { id: model }, batchPrompts: prompts, batchJobCount: 40, createImageJob,
  });
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(15));
  expect(state()).toMatchObject({ submitted: 0, submitting: true });
  // Refill a single freed slot while all 14 earlier requests remain unresolved.
  pending.get("prompt 7")();
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(16));
  expect(pending.has("prompt 0")).toBe(true);
  while (createImageJob.mock.calls.length < prompts.length) {
    const before = createImageJob.mock.calls.length;
    [...pending.values()].forEach((finish) => finish());
    await vi.waitFor(() => expect(createImageJob.mock.calls.length).toBeGreaterThan(before));
  }
  [...pending.values()].forEach((finish) => finish());
  await running;
  expect(peak).toBe(15);
  expect(createImageJob.mock.calls.map(([request]) => request.prompt)).toEqual(prompts);
  expect(state()).toMatchObject({ submitted: 40, failed: 0, submitting: false });
});

it("sends 15 actual job POSTs through the shared creator after one confirmation", async () => {
  let decide;
  const beforeCreate = vi.fn(() => new Promise((resolve) => { decide = resolve; }));
  const pending = [];
  vi.mocked(submitJobRequest).mockImplementation((_path, _token, { body }) => new Promise((resolve) => {
    pending.push(() => resolve({ id: JSON.parse(body).prompt, status: "queued" }));
  }));
  const createImageJob = makeCreateJob({ definition: CREATE_JOB_DEFINITIONS.image, token: "token",
    project: { id: "project", name: "Project" }, requestedGpu: "auto", beforeCreate,
    setJobs: vi.fn(), setError: vi.fn() });
  const { runtime, plan, state } = runtimeBatch({
    batchPrompts: Array.from({ length: 20 }, (_, index) => String(index)), batchJobCount: 20, createImageJob,
  });
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(beforeCreate).toHaveBeenCalledTimes(1));
  expect(submitJobRequest).not.toHaveBeenCalled();
  decide(true);
  await vi.waitFor(() => expect(submitJobRequest).toHaveBeenCalledTimes(15));
  expect(beforeCreate).toHaveBeenCalledTimes(1);
  expect(state()).toMatchObject({ submitted: 0, submitting: true });
  for (const [path, token, { body }] of vi.mocked(submitJobRequest).mock.calls) {
    expect(path).toBe("/api/v1/image/jobs");
    expect(token).toBe("token");
    expect(JSON.parse(body)).toMatchObject({ projectId: "project", requestedGpu: "auto" });
  }
  runtime.cancelBatchRun();
  pending.forEach((finish) => finish());
  await running;
  expect(submitJobRequest).toHaveBeenCalledTimes(15);
  expect(state()).toMatchObject({ submitted: 15, failed: 0, submitting: false });
});

it("reserves queue capacity for POSTs still awaiting a job id", async () => {
  const pending = [];
  const createImageJob = vi.fn((request) => new Promise((resolve) => {
    pending.push(() => resolve({ id: `job-${request.prompt}`, status: "queued" }));
  }));
  const { runtime, plan, state } = runtimeBatch({
    BATCH_ACTIVE_JOB_LIMIT: 3, batchPrompts: ["a", "b", "c", "d", "e"], batchJobCount: 5, createImageJob,
  });
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(3));
  pending.splice(0).forEach((finish) => finish());
  await vi.waitFor(() => expect(state()?.submitted).toBe(3));
  expect(createImageJob).toHaveBeenCalledTimes(3);
  runtime.observeJobs([{ id: "job-a", status: "completed" }, { id: "job-b", status: "queued" }, { id: "job-c", status: "queued" }]);
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(4));
  expect(state().activeJobIds).toHaveLength(2);
  runtime.cancelBatchRun();
  pending.splice(0).forEach((finish) => finish());
  await running;
  expect(createImageJob).toHaveBeenCalledTimes(4);
  expect(state()).toMatchObject({ submitted: 4, completed: 1, failed: 0, submitting: false });
});

it("drains and cancels every late-created job after stopping 15 pending submissions", async () => {
  const pending = [];
  const jobAction = vi.fn();
  const createImageJob = vi.fn((request) => new Promise((resolve) => {
    pending.push(() => resolve({ id: `job-${request.prompt}`, status: "queued" }));
  }));
  const { runtime, plan, state } = runtimeBatch({
    batchPrompts: Array.from({ length: 20 }, (_, index) => String(index)), batchJobCount: 20, createImageJob,
  });
  runtime.commit("project", [], jobAction);
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(15));
  runtime.cancelBatchRun();
  expect(state().submitting).toBe(true);
  pending.reverse().forEach((finish) => finish());
  await running;
  expect(createImageJob).toHaveBeenCalledTimes(15);
  expect(jobAction).toHaveBeenCalledTimes(15);
  expect(new Set(jobAction.mock.calls.map(([job]) => job.id)).size).toBe(15);
  expect(jobAction.mock.calls.every(([, action]) => action === "cancel")).toBe(true);
  expect(state()).toMatchObject({ submitted: 15, failed: 0, submitting: false });
});

it("keeps late responses from a prior project's 15 submissions out of the new run", async () => {
  const pending = [];
  const jobAction = vi.fn();
  const createImageJob = vi.fn((request) => request.prompt === "new prompt"
    ? Promise.resolve({ id: "new-job", status: "queued" })
    : new Promise((resolve) => pending.push(() => resolve({ id: `old-${request.prompt}`, status: "queued" }))));
  const { runtime, plan, state } = runtimeBatch({
    batchPrompts: Array.from({ length: 20 }, (_, index) => String(index)), batchJobCount: 20, createImageJob,
  });
  const previous = runtime.runBatch(plan);
  await vi.waitFor(() => expect(createImageJob).toHaveBeenCalledTimes(15));
  runtime.commit("other-project", [], jobAction);
  runtime.attachProject();
  await runtime.runBatch({ ...plan, batchPrompts: ["new prompt"], batchJobCount: 1 });
  pending.forEach((finish) => finish());
  await previous;
  expect(createImageJob).toHaveBeenCalledTimes(16);
  expect(jobAction).toHaveBeenCalledTimes(15);
  expect(state()).toMatchObject({ total: 1, submitted: 1, failed: 0, activeJobIds: ["new-job"], submitting: false });
});

it("stops the valid prefix at a refusal and still drains submissions already started", async () => {
  let finish;
  vi.mocked(validatePortableBatch).mockResolvedValue([
    { state: "valid" }, { state: "invalid", refusal: { detail: "Invalid second prompt" } }, { state: "valid" },
  ]);
  const { runtime, plan, state } = runtimeBatch({
    createImageJob: vi.fn(() => new Promise((resolve) => { finish = resolve; })),
  });
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(state()?.failed).toBe(1));
  expect(plan.createImageJob).toHaveBeenCalledTimes(1);
  expect(state().submitting).toBe(true);
  finish({ id: "first", status: "queued" });
  await running;
  expect(state()).toMatchObject({ submitted: 2, failed: 1, submitting: false });
});

it("keeps structured rewriting serial while a submitted job awaits its response", async () => {
  let finish;
  const { runtime, plan } = runtimeBatch({
    structuredPromptModel: true,
    onMagicExpand: vi.fn(async () => ({ ...emptyCaption(), high_level_description: "A fox in a forest" })),
    createImageJob: vi.fn().mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }))
      .mockResolvedValue({ id: "later", status: "queued" }),
  });
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(plan.createImageJob).toHaveBeenCalledTimes(1));
  expect(plan.onMagicExpand).toHaveBeenCalledTimes(1);
  finish({ id: "first", status: "queued" });
  await running;
  expect(plan.onMagicExpand).toHaveBeenCalledTimes(3);
});

it("finishes submitting an exact validation window even when every queue slot is occupied", async () => {
  const { runtime, plan, state } = runtimeBatch({
    BATCH_ACTIVE_JOB_LIMIT: 32,
    batchPrompts: Array.from({ length: 32 }, (_, index) => `prompt ${index}`), batchJobCount: 32,
  });
  await runtime.runBatch(plan);
  expect(validatePortableBatch).toHaveBeenCalledTimes(1);
  expect(state()).toMatchObject({ submitted: 32, submitting: false });
  expect(state().activeJobIds).toHaveLength(32);
});

it("preserves native prompt groups, per-prompt sizes and progress accounting", async () => {
  const { runtime, plan, state } = runtimeBatch({
    selectedModel: { image: { maxPromptBatchImages: 8 } },
    batchPrompts: ["[512x512] one", "[512x512] two", "[768x512] three", "[768x512] four", "[768x512] five"],
    batchJobCount: 5,
    buildBatchJobRequest: (prompt, { resolution } = {}) => ({ prompt: `Styled ${prompt}`, count: 2, ...resolution }),
  });
  await runtime.runBatch(plan);
  expect(validatePortableBatch).toHaveBeenCalledTimes(1);
  expect(vi.mocked(validatePortableBatch).mock.calls[0][0].requests).toEqual([
    { prompt: "Styled one", prompts: ["Styled one", "Styled two"], count: 2, width: 512, height: 512, projectId: "project" },
    { prompt: "Styled three", prompts: ["Styled three", "Styled four", "Styled five"], count: 2, width: 768, height: 512, projectId: "project" },
  ]);
  expect(state()).toMatchObject({ submitted: 5, failed: 0, sizes: { "job-Styled one": 2, "job-Styled three": 3 } });
  runtime.observeJobs([{ id: "job-Styled one", status: "completed" }, { id: "job-Styled three", status: "failed" }]);
  expect(state()).toMatchObject({ completed: 2, failed: 3, activeJobIds: [], sizes: {} });
});

it("submits only the valid prefix and counts a refused native group by its prompt count", async () => {
  const refusal = { detail: "Unsupported dimensions" };
  vi.mocked(validatePortableBatch).mockImplementation(async ({ requests }) => requests.map((_request, index) =>
    index === 1 ? { state: "invalid", refusal } : { state: "valid" }));
  const { runtime, plan, state } = runtimeBatch({
    selectedModel: { image: { maxPromptBatchImages: 2 } },
    batchPrompts: ["one", "two", "three", "four", "five"], batchJobCount: 5,
  });
  await runtime.runBatch(plan);
  expect(plan.createImageJob.mock.calls.map(([request]) => request.prompts)).toEqual([["one", "two"]]);
  expect(state()).toMatchObject({ submitted: 4, failed: 2, firstError: refusal.detail, submitting: false });
  expect(runtime.progress([])).toMatchObject({ queued: 2, failed: 2, pending: 1 });
});

it.each(["server-required", "throw"])("stops cleanly when validation is %s", async (failure) => {
  vi.mocked(validatePortableBatch).mockImplementation(async ({ requests }) => {
    if (failure === "throw") throw new Error("Validation unavailable");
    return requests.map(() => ({ state: "server-required" }));
  });
  const { runtime, plan, state } = runtimeBatch();
  await runtime.runBatch(plan);
  expect(plan.createImageJob).not.toHaveBeenCalled();
  expect(state()).toMatchObject({ submitted: 1, failed: 1, submitting: false });
  expect(runtime.progress([]).pending).toBe(2);
});

it("aborts pending batch validation without submitting or recording failures", async () => {
  let signal;
  vi.mocked(validatePortableBatch).mockImplementation(({ signal: pendingSignal }) => {
    signal = pendingSignal;
    return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(new DOMException("Stopped", "AbortError"))));
  });
  const { runtime, plan, state } = runtimeBatch();
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(validatePortableBatch).toHaveBeenCalledTimes(1));
  runtime.cancelBatchRun();
  await running;
  expect(signal.aborted).toBe(true);
  expect(plan.createImageJob).not.toHaveBeenCalled();
  expect(state()).toMatchObject({ submitted: 0, failed: 0, submitting: false });
});

it("ignores a late validation response after switching projects and starting another batch", async () => {
  let finishValidation;
  vi.mocked(validatePortableBatch).mockImplementationOnce(({ requests }) => new Promise((resolve) => {
    finishValidation = () => resolve(requests.map(() => ({ state: "valid" })));
  }));
  const { runtime, plan, state } = runtimeBatch();
  const previous = runtime.runBatch(plan);
  await vi.waitFor(() => expect(validatePortableBatch).toHaveBeenCalledTimes(1));
  runtime.commit("other-project", [], vi.fn());
  runtime.attachProject();
  await runtime.runBatch({ ...plan, batchPrompts: ["new prompt"], batchJobCount: 1 });
  finishValidation();
  await previous;
  expect(plan.createImageJob.mock.calls.map(([request]) => request.prompt)).toEqual(["new prompt"]);
  expect(vi.mocked(validatePortableBatch).mock.calls[1][0].requests[0].projectId).toBe("other-project");
  expect(state()).toMatchObject({ total: 1, submitted: 1, failed: 0, submitting: false });
});

it("bounds request preparation while waiting for queue capacity", async () => {
  const buildBatchJobRequest = vi.fn((prompt) => ({ prompt, count: 1 }));
  const { runtime, plan, state } = runtimeBatch({
    BATCH_ACTIVE_JOB_LIMIT: 1, batchJobCount: 148,
    batchPrompts: Array.from({ length: 148 }, (_, index) => `prompt ${index}`),
    buildBatchJobRequest,
  });
  const running = runtime.runBatch(plan);
  await vi.waitFor(() => expect(state()?.submitted).toBe(1));
  expect(validatePortableBatch).toHaveBeenCalledTimes(1);
  // One count probe plus at most one validation window of composed requests.
  expect(buildBatchJobRequest.mock.calls.length).toBeLessThanOrEqual(33);
  expect(plan.createImageJob).toHaveBeenCalledTimes(1);
  runtime.cancelBatchRun();
  await running;
  expect(state()).toMatchObject({ submitted: 1, failed: 0, submitting: false });
});

it("discards a prepared validation window after stopping during submission", async () => {
  const { runtime, plan, state } = runtimeBatch();
  plan.createImageJob.mockImplementationOnce(async () => {
    runtime.cancelBatchRun();
    return { id: "just-created", status: "queued" };
  });
  await runtime.runBatch(plan);
  expect(plan.createImageJob).toHaveBeenCalledTimes(1);
  expect(state()).toMatchObject({ submitted: 1, failed: 0, submitting: false });
});

it("keeps request-building failures aligned with their batch items", async () => {
  const { runtime, plan, state } = runtimeBatch({
    buildBatchJobRequest: (prompt) => {
      if (prompt === "second") throw new Error("Could not compose second prompt");
      return { prompt, count: 1 };
    },
  });
  await runtime.runBatch(plan);
  expect(vi.mocked(validatePortableBatch).mock.calls[0][0].requests.map((request) => request.prompt)).toEqual(["first", "third"]);
  expect(plan.createImageJob.mock.calls.map(([request]) => request.prompt)).toEqual(["first", "third"]);
  expect(state()).toMatchObject({ submitted: 3, failed: 1, firstError: "Could not compose second prompt", submitting: false });
});

it("keeps falsy request-building exceptions out of validation and submission", async () => {
  const { runtime, plan, state } = runtimeBatch({
    buildBatchJobRequest: (prompt) => {
      if (prompt === "second") throw "";
      return { prompt, count: 1 };
    },
  });
  await runtime.runBatch(plan);
  expect(vi.mocked(validatePortableBatch).mock.calls[0][0].requests.map((request) => request.prompt)).toEqual(["first", "third"]);
  expect(plan.createImageJob.mock.calls.map(([request]) => request.prompt)).toEqual(["first", "third"]);
  expect(state()).toMatchObject({ submitted: 3, failed: 1, submitting: false });
});

it("rewrites and validates structured captions just before their own submission", async () => {
  const events = [];
  const caption = { ...emptyCaption(), high_level_description: "A fox in a forest" };
  vi.mocked(validatePortableBatch).mockImplementation(async ({ requests }) => {
    events.push("validate");
    expect(requests).toHaveLength(1);
    expect(requests[0].prompt).toBe(serializeCaption(caption));
    return [{ state: "valid" }];
  });
  const { runtime, plan, state } = runtimeBatch({
    structuredPromptModel: true,
    onMagicExpand: vi.fn(async (prompt) => {
      events.push(`rewrite:${prompt}`);
      if (prompt === "second") throw new Error("Rewrite failed");
      return caption;
    }),
    createImageJob: vi.fn(async () => {
      events.push("submit");
      return { id: `job-${events.length}`, status: "queued" };
    }),
  });
  await runtime.runBatch(plan);
  expect(events).toEqual(["rewrite:first", "validate", "submit", "rewrite:second", "rewrite:third", "validate", "submit"]);
  expect(state()).toMatchObject({ submitted: 3, failed: 1, firstError: "Rewrite failed", submitting: false });
});

it("does not rewrite later structured prompts after a validation refusal", async () => {
  const refusal = { detail: "Caption refused" };
  vi.mocked(validatePortableBatch).mockResolvedValue([{ state: "invalid", refusal }]);
  const { runtime, plan, state } = runtimeBatch({
    structuredPromptModel: true,
    onMagicExpand: vi.fn(async () => ({ ...emptyCaption(), high_level_description: "A fox in a forest" })),
  });
  await runtime.runBatch(plan);
  expect(plan.onMagicExpand).toHaveBeenCalledTimes(1);
  expect(plan.createImageJob).not.toHaveBeenCalled();
  expect(state()).toMatchObject({ submitted: 1, failed: 1, firstError: refusal.detail, submitting: false });
});
