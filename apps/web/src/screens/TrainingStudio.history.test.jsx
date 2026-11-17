import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { settle } from "../testUtils/dom.js";
import { AppContext } from "../context/AppContext.js";
import { trainingApi } from "../api/trainingResources.js";
import { TrainingStudio } from "./TrainingStudio.jsx";
import { irisTarget } from "../training/iris.testSupport.js";

vi.mock("../api/trainingResources.js", () => ({ trainingApi: { completedRuns: vi.fn() } }));

function run(id, name = "Saved LoRA", projectId = "project") {
  return { id, type: "lora_train", status: "completed", projectId, progress: 1,
    payload: { outputName: name, dryRun: false,
      plan: { target: { outputKind: "lora" }, config: { steps: 3000 } } }, result: {} };
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
  trainingApi.completedRuns.mockReset();
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
function history() { return container.querySelector('[aria-label="Completed training runs"]'); }

it("offers saved-run continuation with an empty Queue, no dataset and no installed base", async () => {
  const saved = run("saved");
  const ctx = context({
    models: [{ id: "sdxl", name: "SDXL", installState: "missing" }],
    trainingTargets: { targets: [{ ...irisTarget(), id: "sdxl_lora", kernel: "sdxl_lora", baseModel: "sdxl" }] },
  });
  trainingApi.completedRuns.mockResolvedValue({ jobs: [saved], nextOffset: null });
  await render(ctx);
  expect(container.textContent).toContain("Training needs a trainable base model");
  await click("Continue training");
  expect(history().textContent).toContain("Saved LoRA");
  expect(trainingApi.completedRuns).toHaveBeenCalledWith("project", "token", { offset: 0 }, expect.objectContaining({ signal: expect.any(AbortSignal) }));
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
  trainingApi.completedRuns
    .mockResolvedValueOnce({ jobs: [run("new", "Newest LoRA")], nextOffset: 1 })
    .mockResolvedValueOnce({ jobs: [run("old", "Older LoRA")], nextOffset: null })
    .mockResolvedValueOnce({ jobs: [run("new", "Newest LoRA")], nextOffset: 1 });
  await render(ctx);
  await click("Continue training");
  await click("Load more runs", history());
  expect(history().textContent).toContain("Newest LoRA");
  expect(history().textContent).toContain("Older LoRA");
  expect(trainingApi.completedRuns.mock.calls[1][2]).toEqual({ offset: 1 });
  await render({ ...ctx, jobs: [] });
  expect(history().textContent).toContain("Newest LoRA");
  expect(history().textContent).not.toContain("No completed LoRA runs");
});

it("ignores a late response from the previous workspace", async () => {
  let finishOld;
  trainingApi.completedRuns
    .mockImplementationOnce(() => new Promise((resolve) => { finishOld = resolve; }))
    .mockResolvedValueOnce({ jobs: [run("other", "Other workspace LoRA", "other")], nextOffset: null });
  const ctx = context();
  await render(ctx);
  await click("Continue training");
  const oldSignal = trainingApi.completedRuns.mock.calls[0][3].signal;
  await render({ ...ctx, activeProject: { id: "other" } });
  expect(oldSignal.aborted).toBe(true);
  await act(async () => finishOld({ jobs: [run("stale", "Stale workspace LoRA")], nextOffset: null }));
  expect(history().textContent).toContain("Other workspace LoRA");
  expect(history().textContent).not.toContain("Stale workspace LoRA");
});

it("shows a load error and allows retry without treating it as an empty history", async () => {
  trainingApi.completedRuns.mockRejectedValueOnce(new Error("History unavailable"))
    .mockResolvedValueOnce({ jobs: [], nextOffset: null });
  await render(context());
  await click("Continue training");
  expect(history().querySelector('[role="alert"]').textContent).toBe("History unavailable");
  expect(history().textContent).not.toContain("No completed LoRA runs");
  await click("Refresh runs", history());
  expect(history().textContent).toContain("No completed LoRA runs in this workspace yet.");
});
