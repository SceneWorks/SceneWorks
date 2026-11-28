import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const feedbackState = vi.hoisted(() => ({ value: { state: "valid" } }));
vi.mock("../hooks/usePortableFeedback.js", () => ({ usePortableFeedback: (_kind, _payload, _entry, enabled = true) => enabled ? feedbackState.value : null }));
import { settle } from "../testUtils/dom.js";
import React, { act } from "react";
import { createRoot } from "react-dom/client";

// sc-25685/sc-25686 — the Training Studio wiring for the Iris-3B target end to end: picking a
// dataset fires the data-walk preflight and shows its warning before launch, the Iris options
// reach the panel (previous runs, caption fields), and Start training sends `advanced.iris`
// without any API-owned path key.

import { AppContext } from "../context/AppContext.js";
import { TrainingStudio } from "./TrainingStudio.jsx";
import { irisJob, irisTarget } from "../training/iris.testSupport.js";

const WARNING = "Only 640 of 700 items will be used: Iris walks whole 640-item chunks.";

const irisDataset = {
  id: "dataset-1",
  name: "Mira Set",
  version: 2,
  characterId: "",
  items: Array.from({ length: 700 }, (_, index) => ({
    id: `item_${index}`,
    path: `images/${index}.png`,
    caption: { text: `caption ${index}` },
    ...(index === 0 ? { captions: { short: "mira" } } : {}),
  })),
};

function context(overrides = {}) {
  return {
    activeProject: { id: "p1", name: "Project" },
    token: "tok",
    authenticated: true,
    assets: [],
    characters: [],
    jobs: [irisJob({ id: "job_prev", outputName: "Mira v1" })],
    models: [],
    setPreviewAsset: () => {},
    trainingDatasets: [{ id: "dataset-1", name: "Mira Set", version: 2 }],
    trainingDatasetsProjectId: "p1",
    loadingTrainingDatasets: false,
    refreshTrainingDatasets: () => {},
    loadTrainingDataset: vi.fn(async () => irisDataset),
    createTrainingJob: vi.fn(async () => ({ id: "job_new" })),
    trainingPresets: { presets: [] },
    trainingTargets: { targets: [irisTarget()] },
    setActiveView: () => {},
    createModelDownloadJob: () => {},
    ...overrides,
  };
}



describe("TrainingStudio — Iris", () => {
  let container;
  let root;
  let fetchMock;

  beforeEach(() => {
    feedbackState.value = { state: "valid" };
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    fetchMock = vi.fn(async (url) => ({
      ok: true,
      status: 200,
      json: async () =>
        String(url).includes("/training/data-walk")
          ? { applies: true, items: 700, covered: 640, unusedItems: 60, chunks: 640, warning: WARNING }
          : null,
    }));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.unstubAllGlobals();
  });

  async function renderAndPickDataset(ctx) {
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={ctx}>{<TrainingStudio />}</AppContext.Provider>);
    });
    await settle({ microtasks: 10 });
    const select = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith("Dataset"))
      .querySelector("select");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(select), "value").set;
      setter.call(select, "dataset-1");
      select.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    await settle({ microtasks: 10 });
  }

  // A stopped Iris run cannot take the generic retry (the Iris rerun guard refuses a checkpointed
  // run), so Continue training routes it to New training's Iris resume picker — including a run
  // already cleared from the Queue, which the live `jobs` no longer carry.
  it("resumes a queue-cleared stopped Iris run through New training's resume picker", async () => {
    const stopped = { ...irisJob({ id: "job_stopped", status: "canceled", outputName: "Mira stopped",
      result: { stepsCompleted: 500 } }), irisResumable: true };
    stopped.payload.plan.dataset = { datasetId: "dataset-1" };
    fetchMock.mockImplementation(async (url) => ({
      ok: true,
      status: 200,
      json: async () => (String(url).includes("/training/runs") ? { jobs: [stopped], nextOffset: null } : null),
    }));
    const ctx = context({ jobs: [] });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={ctx}>{<TrainingStudio />}</AppContext.Provider>);
    });
    await settle({ microtasks: 10 });
    const buttons = () => [...container.querySelectorAll("button")];
    await act(async () => buttons().find((button) => button.textContent === "Continue training").click());
    await settle({ microtasks: 10 });
    const history = container.querySelector('[aria-label="Saved training runs"]');
    expect(history.textContent).toContain("Mira stopped");
    const labels = [...history.querySelectorAll("button")].map((button) => button.textContent);
    expect(labels).toContain("Resume in New training");
    expect(labels.some((label) => label.startsWith("Resume from step") || label === "Restart from step 0")).toBe(false);
    await act(async () => buttons().find((button) => button.textContent === "Resume in New training").click());
    await settle({ microtasks: 10 });
    expect(container.querySelector('[aria-label="Saved training runs"]')).toBeNull();
    expect(ctx.loadTrainingDataset).toHaveBeenCalled();
    const dataset = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith("Dataset"))
      .querySelector("select");
    expect(dataset.value).toBe("dataset-1");
    await act(async () => {
      container.querySelector(".advanced-section-toggle").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    const resume = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith("Resume a previous run"))
      ?.querySelector("select");
    expect([...resume.options].map((option) => option.value)).toEqual(["", "job_stopped"]);
    expect(resume.value).toBe("job_stopped");
  });

  // A resume continues the same adapter: the API refuses another network type, and the
  // checkpoint fixes the factor shapes. A LoKr run resumed from the default (LoRA) draft must
  // submit as LoKr with its own rank, alpha and decomposition factor.
  it("resumes a queue-cleared Iris LoKr run with its own adapter shape", async () => {
    const lokr = { ...irisJob({ id: "job_lokr", status: "canceled", outputName: "Mira LoKr", networkType: "lokr" }), irisResumable: true };
    lokr.payload.plan.config = { ...lokr.payload.plan.config, rank: 8, alpha: 4,
      advanced: { ...lokr.payload.plan.config.advanced, decomposeFactor: 4 } };
    lokr.payload.plan.dataset = { datasetId: "dataset-1" };
    fetchMock.mockImplementation(async (url) => ({
      ok: true,
      status: 200,
      json: async () => (String(url).includes("/training/runs") ? { jobs: [lokr], nextOffset: null } : null),
    }));
    const ctx = context({ jobs: [] });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={ctx}>{<TrainingStudio />}</AppContext.Provider>);
    });
    await settle({ microtasks: 10 });
    const button = (label) => [...container.querySelectorAll("button")].find((item) => item.textContent === label);
    await act(async () => button("Continue training").click());
    await settle({ microtasks: 10 });
    await act(async () => button("Resume in New training").click());
    await settle({ microtasks: 10 });
    expect(button("Start training").disabled).toBe(false);
    await act(async () => button("Start training").dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    await settle({ microtasks: 10 });
    expect(ctx.createTrainingJob).toHaveBeenCalledTimes(1);
    const { config } = ctx.createTrainingJob.mock.calls[0][0];
    expect(config.advanced.resumeFromJobId).toBe("job_lokr");
    expect(config.advanced.networkType).toBe("lokr");
    expect(config.advanced.decomposeFactor).toBe(4);
    expect(config.rank).toBe(8);
    expect(config.alpha).toBe(4);
  });

  // Stopped before its first checkpoint, an Iris run can only restart: the Iris resume would be
  // refused (no checkpoint), while the rerun guard allows a retry.
  it("keeps restart for a queue-cleared Iris run stopped before its first checkpoint", async () => {
    const stopped = { ...irisJob({ id: "job_early", status: "interrupted", outputName: "Mira early" }), irisResumable: false };
    fetchMock.mockImplementation(async (url) => ({
      ok: true,
      status: 200,
      json: async () => (String(url).includes("/training/runs") ? { jobs: [stopped], nextOffset: null } : { id: "job_retry" }),
    }));
    const ctx = context({ jobs: [], jobAction: vi.fn(async () => ({ id: "job_retry" })) });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={ctx}>{<TrainingStudio />}</AppContext.Provider>);
    });
    await settle({ microtasks: 10 });
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Continue training").click());
    await settle({ microtasks: 10 });
    const history = container.querySelector('[aria-label="Saved training runs"]');
    expect(history.textContent).toContain("Mira early");
    const labels = [...history.querySelectorAll("button")].map((button) => button.textContent);
    expect(labels).not.toContain("Resume in New training");
    await act(async () => [...history.querySelectorAll("button")].find((button) => button.textContent === "Restart from step 0").click());
    expect(ctx.jobAction).toHaveBeenCalledExactlyOnceWith(stopped, "retry", { body: { resume: false } });
  });

  it("shows the data-walk warning for the picked dataset and submits advanced.iris", async () => {
    const ctx = context();
    await renderAndPickDataset(ctx);

    const walkCall = fetchMock.mock.calls.find(([url]) => String(url).includes("/training/data-walk"));
    expect(walkCall?.[0]).toContain("targetId=iris_3b_generation&items=700&batchSize=1");
    expect(container.textContent).toContain(WARNING);

    // The panel received the Iris context: the previous run and the dataset's named field.
    await act(async () => {
      container.querySelector(".advanced-section-toggle").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    const resume = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith("Resume a previous run"))
      ?.querySelector("select");
    expect([...resume.options].map((option) => option.value)).toEqual(["", "job_prev"]);
    await act(async () => {
      Object.getOwnPropertyDescriptor(Object.getPrototypeOf(resume), "value").set.call(resume, "job_prev");
      resume.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    expect(resume.value).toBe("job_prev");
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "New training").click());
    expect(resume.value).toBe("");
    const captionField = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith("Caption field"))
      ?.querySelector("select");
    expect([...captionField.options].map((option) => option.value)).toEqual(["caption", "short"]);

    const start = [...container.querySelectorAll("button")].find((button) => button.textContent === "Start training");
    expect(start.disabled).toBe(false);
    await act(async () => {
      start.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    await settle({ microtasks: 10 });
    expect(ctx.createTrainingJob).toHaveBeenCalledTimes(1);
    const { config } = ctx.createTrainingJob.mock.calls[0][0];
    expect(config.advanced.iris).toMatchObject({ text_conditioning: "cached", ema_enabled: true, export_dtype: "fp32" });
    expect(config.advanced.iris).not.toHaveProperty("resume_from");
    expect(config.advanced.iris).not.toHaveProperty("load_from");
    expect(config.advanced).not.toHaveProperty("resumeFromJobId");
    expect(config.advanced.gradientCheckpointing).toBe(false);
  });

  it.each(["pending", "server-required", "invalid"])("blocks the full training form while feedback is %s", async (state) => {
    feedbackState.value = { state, refusal: { field: "config.advanced.iris", detail: "Fix the Iris options" } };
    const ctx = context();
    await renderAndPickDataset(ctx);
    const start = () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Start training");
    expect(start().disabled).toBe(true);
    if (state === "pending") expect(container.textContent).not.toContain("Checking options with the server");
    else expect(container.textContent).toContain(state === "server-required" ? "Reconnect to continue" : "Fix the Iris options");
    await act(async () => start().dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(ctx.createTrainingJob).not.toHaveBeenCalled();
    feedbackState.value = { state: "valid" };
    await act(async () => root.render(<AppContext.Provider value={ctx}><TrainingStudio /></AppContext.Provider>));
    expect(start().disabled).toBe(false);
    await act(async () => start().click());
    expect(ctx.createTrainingJob).toHaveBeenCalledTimes(1);
  });

  it("does not run the preflight for a non-Iris target", async () => {
    const legacy = { ...irisTarget(), id: "sdxl_lora", kernel: "sdxl_lora", memoryFloors: undefined };
    await renderAndPickDataset(context({ trainingTargets: { targets: [legacy] } }));
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/training/data-walk"))).toBe(false);
    expect(container.textContent).not.toContain(WARNING);
    expect(container.querySelector("details.training-iris-options")).toBeNull();
  });
});
