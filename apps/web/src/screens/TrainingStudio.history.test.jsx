import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { settle } from "../testUtils/dom.js";
import { AppContext } from "../context/AppContext.js";
import { trainingApi } from "../api/trainingResources.js";
import { TrainingStudio } from "./TrainingStudio.jsx";
import { irisTarget } from "../training/iris.testSupport.js";

vi.mock("../api/trainingResources.js", () => ({ trainingApi: { savedRuns: vi.fn(), removeSavedRun: vi.fn() } }));
const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("../appConfirm.jsx", async (importOriginal) => ({ ...(await importOriginal()), appConfirm: appConfirmMock }));

function run(id, name = "Saved LoRA", projectId = "project", overrides = {}) {
  return { id, type: "lora_train", status: "completed", projectId, progress: 1,
    payload: { outputName: name, dryRun: false,
      plan: { target: { outputKind: "lora" }, config: { steps: 3000 } } }, result: {}, ...overrides };
}
function context(overrides = {}) {
  return { activeProject: { id: "project" }, token: "token", authenticated: true,
    jobs: [], models: [], assets: [], characters: [], loras: [],
    trainingDatasets: [], trainingDatasetsProjectId: "project",
    trainingTargets: { targets: [] }, trainingPresets: { presets: [] },
    setActiveView: vi.fn(), refreshTrainingDatasets: vi.fn(),
    jobAction: vi.fn().mockResolvedValue({ id: "continued" }), ...overrides };
}
let root;
let container;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  trainingApi.savedRuns.mockReset();
  trainingApi.removeSavedRun.mockReset();
  appConfirmMock.mockReset();
  appConfirmMock.mockResolvedValue(true);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); });
async function render(value) {
  await act(async () => root.render(<AppContext.Provider value={value}><TrainingStudio /></AppContext.Provider>));
  await settle({ microtasks: 10 });
}
async function click(label, scope = container) {
  const button = [...scope.querySelectorAll("button")].find((item) => item.textContent === label);
  expect(button, label).toBeTruthy();
  await act(async () => button.click());
  await settle({ microtasks: 10 });
}
function history() { return container.querySelector('[aria-label="Saved training runs"]'); }

it("offers saved-run continuation with an empty Queue, no dataset and no installed base", async () => {
  const saved = run("saved");
  const ctx = context({
    models: [{ id: "sdxl", name: "SDXL", installState: "missing" }],
    trainingTargets: { targets: [{ ...irisTarget(), id: "sdxl_lora", kernel: "sdxl_lora", baseModel: "sdxl" }] },
  });
  trainingApi.savedRuns.mockResolvedValue({ jobs: [saved], nextOffset: null });
  await render(ctx);
  expect(container.textContent).toContain("Training needs a trainable base model");
  await click("Continue training");
  expect(history().textContent).toContain("Saved LoRA");
  expect(trainingApi.savedRuns).toHaveBeenCalledWith("project", "token", { offset: 0 }, expect.objectContaining({ signal: expect.any(AbortSignal) }));
  await click("Continue training", history());
  expect(history().textContent).toContain("3,000 → 4,000 total steps");
  await act(async () => history().querySelector("form").dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(ctx.jobAction).toHaveBeenCalledExactlyOnceWith(saved, "continue-training", { body: { additionalSteps: 1000 } });
  await click("New training");
  expect(history()).toBeNull();
  expect(container.textContent).toContain("Training needs a trainable base model");
});

it("pages older saved runs and keeps them discoverable when the Queue is cleared", async () => {
  const ctx = context({ jobs: [run("new")] });
  trainingApi.savedRuns
    .mockResolvedValueOnce({ jobs: [run("new", "Newest LoRA")], nextOffset: 1 })
    .mockResolvedValueOnce({ jobs: [run("old", "Older LoRA")], nextOffset: null })
    .mockResolvedValueOnce({ jobs: [run("new", "Newest LoRA")], nextOffset: 1 });
  await render(ctx);
  await click("Continue training");
  await click("Load more runs", history());
  expect(history().textContent).toContain("Newest LoRA");
  expect(history().textContent).toContain("Older LoRA");
  expect(trainingApi.savedRuns.mock.calls[1][2]).toEqual({ offset: 1 });
  await render({ ...ctx, jobs: [] });
  expect(history().textContent).toContain("Newest LoRA");
  expect(history().textContent).not.toContain("No completed, canceled or interrupted LoRA runs");
});

it("ignores a late response from the previous workspace", async () => {
  let finishOld;
  trainingApi.savedRuns
    .mockImplementationOnce(() => new Promise((resolve) => { finishOld = resolve; }))
    .mockResolvedValueOnce({ jobs: [run("other", "Other workspace LoRA", "other")], nextOffset: null });
  const ctx = context();
  await render(ctx);
  await click("Continue training");
  const oldSignal = trainingApi.savedRuns.mock.calls[0][3].signal;
  await render({ ...ctx, activeProject: { id: "other" } });
  expect(oldSignal.aborted).toBe(true);
  await act(async () => finishOld({ jobs: [run("stale", "Stale workspace LoRA")], nextOffset: null }));
  expect(history().textContent).toContain("Other workspace LoRA");
  expect(history().textContent).not.toContain("Stale workspace LoRA");
});

it("shows a load error and allows retry without treating it as an empty history", async () => {
  trainingApi.savedRuns.mockRejectedValueOnce(new Error("History unavailable"))
    .mockResolvedValueOnce({ jobs: [], nextOffset: null });
  await render(context());
  await click("Continue training");
  expect(history().querySelector('[role="alert"]').textContent).toBe("History unavailable");
  expect(history().textContent).not.toContain("No completed, canceled or interrupted LoRA runs");
  await click("Refresh runs", history());
  expect(history().textContent).toContain("No completed, canceled or interrupted LoRA runs in this workspace yet.");
});

it("offers canceled and interrupted runs a resume from their last checkpoint", async () => {
  const canceled = run("canceled", "Canceled LoRA", "project",
    { status: "canceled", progress: 0.4, result: { resumeCheckpointStep: 1200 } });
  const interrupted = run("interrupted", "Interrupted LoRA", "project",
    { status: "interrupted", progress: 0.1, result: {} });
  const ctx = context();
  trainingApi.savedRuns.mockResolvedValue({ jobs: [canceled, interrupted], nextOffset: null });
  await render(ctx);
  await click("Continue training");
  expect(history().textContent).toContain("Canceled LoRA");
  expect(history().textContent).toContain("Interrupted LoRA");
  // A stopped run resumes to its planned total; adding steps is for completed runs only.
  expect([...history().querySelectorAll("button")].filter((button) => button.textContent === "Continue training")).toHaveLength(0);
  await click("Resume from step 1200", history());
  expect(ctx.jobAction).toHaveBeenCalledExactlyOnceWith(canceled, "retry", { body: { resume: true } });
  // Without a recorded checkpoint the only way forward is a restart.
  expect([...history().querySelectorAll("button")].filter((button) => button.textContent.startsWith("Resume from step"))).toHaveLength(1);
  expect([...history().querySelectorAll("button")].filter((button) => button.textContent === "Restart from step 0")).toHaveLength(2);
});

it("reloads saved runs when a resume queues a new run", async () => {
  const canceled = run("canceled", "Canceled LoRA", "project", { status: "canceled", result: { resumeCheckpointStep: 1200 } });
  const ctx = context({ jobs: [canceled] });
  trainingApi.savedRuns.mockResolvedValue({ jobs: [canceled], nextOffset: null });
  await render(ctx);
  await click("Continue training");
  expect(trainingApi.savedRuns).toHaveBeenCalledTimes(1);
  trainingApi.savedRuns.mockResolvedValue({ jobs: [], nextOffset: null });
  await render({ ...ctx, jobs: [canceled, run("resumed", "Canceled LoRA", "project", { status: "queued", sourceJobId: "canceled" })] });
  expect(trainingApi.savedRuns).toHaveBeenCalledTimes(2);
  expect(history().textContent).not.toContain("Canceled LoRA");
});

it("removes a saved run permanently only after confirmation", async () => {
  const kept = run("kept", "Kept LoRA");
  const stopped = run("stopped", "Stopped LoRA", "project", { status: "canceled" });
  trainingApi.savedRuns.mockResolvedValue({ jobs: [kept, stopped], nextOffset: null });
  trainingApi.removeSavedRun.mockResolvedValue(null);
  await render(context());
  await click("Continue training");
  const removeButtons = () => [...history().querySelectorAll("button")].filter((button) => button.textContent === "Remove");
  expect(removeButtons()).toHaveLength(2);
  appConfirmMock.mockResolvedValueOnce(false);
  await act(async () => removeButtons()[1].click());
  await settle({ microtasks: 10 });
  expect(trainingApi.removeSavedRun).not.toHaveBeenCalled();
  expect(history().textContent).toContain("Stopped LoRA");
  await act(async () => removeButtons()[1].click());
  await settle({ microtasks: 10 });
  expect(appConfirmMock).toHaveBeenLastCalledWith(expect.objectContaining({ tone: "danger", confirmLabel: "Remove run" }));
  expect(trainingApi.removeSavedRun).toHaveBeenCalledExactlyOnceWith("project", "stopped", "token");
  expect(history().textContent).not.toContain("Stopped LoRA");
  expect(history().textContent).toContain("Kept LoRA");
});

it("keeps a saved run listed and reports the error when removal fails", async () => {
  trainingApi.savedRuns.mockResolvedValue({ jobs: [run("saved")], nextOffset: null });
  trainingApi.removeSavedRun.mockRejectedValue(new Error("Record not found: saved"));
  await render(context());
  await click("Continue training");
  await click("Remove", history());
  expect(history().querySelector('[role="alert"]').textContent).toBe("Record not found: saved");
  expect(history().textContent).toContain("Saved LoRA");
});

it("loads the next page from the shifted offset after a removal", async () => {
  trainingApi.savedRuns
    .mockResolvedValueOnce({ jobs: [run("first", "First LoRA"), run("second", "Second LoRA")], nextOffset: 2 })
    .mockResolvedValueOnce({ jobs: [run("third", "Third LoRA")], nextOffset: null });
  trainingApi.removeSavedRun.mockResolvedValue(null);
  // The removed run was already cleared from the Queue, so no live job change triggers a reload.
  await render(context());
  await click("Continue training");
  const removeFirst = [...history().querySelectorAll("button")].find((button) => button.textContent === "Remove");
  await act(async () => removeFirst.click());
  await settle({ microtasks: 10 });
  expect(history().textContent).not.toContain("First LoRA");
  await click("Load more runs", history());
  expect(trainingApi.savedRuns.mock.calls[1][2]).toEqual({ offset: 1 });
  expect(history().textContent).toContain("Second LoRA");
  expect(history().textContent).toContain("Third LoRA");
});

it("does not page while a removal is pending, nor remove while a page loads", async () => {
  let finishRemoval;
  let finishPage;
  trainingApi.savedRuns
    .mockResolvedValueOnce({ jobs: [run("first", "First LoRA"), run("second", "Second LoRA")], nextOffset: 2 })
    .mockImplementationOnce(() => new Promise((resolve) => { finishPage = resolve; }));
  trainingApi.removeSavedRun.mockImplementationOnce(() => new Promise((resolve) => { finishRemoval = resolve; }));
  await render(context());
  await click("Continue training");
  const button = (label) => [...history().querySelectorAll("button")].find((item) => item.textContent === label);
  await act(async () => button("Remove").click());
  await settle({ microtasks: 10 });
  expect(button("Removing…").disabled).toBe(true);
  expect(button("Load more runs").disabled).toBe(true);
  expect(button("Refresh runs").disabled).toBe(true);
  await act(async () => finishRemoval(null));
  await settle({ microtasks: 10 });
  expect(button("Load more runs").disabled).toBe(false);
  await click("Load more runs", history());
  expect(trainingApi.savedRuns.mock.calls[1][2]).toEqual({ offset: 1 });
  expect(button("Remove").disabled).toBe(true);
  await act(async () => finishPage({ jobs: [run("third", "Third LoRA")], nextOffset: null }));
  await settle({ microtasks: 10 });
  expect(button("Remove").disabled).toBe(false);
  expect(history().textContent).toContain("Third LoRA");
});
