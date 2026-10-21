import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

async function settle() {
  await act(async () => {
    for (let index = 0; index < 10; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("TrainingStudio — Iris", () => {
  let container;
  let root;
  let fetchMock;

  beforeEach(() => {
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
    await settle();
    const select = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith("Dataset"))
      .querySelector("select");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(select), "value").set;
      setter.call(select, "dataset-1");
      select.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    await settle();
  }

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
    const captionField = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.startsWith("Caption field"))
      ?.querySelector("select");
    expect([...captionField.options].map((option) => option.value)).toEqual(["caption", "short"]);

    const start = [...container.querySelectorAll("button")].find((button) => button.textContent === "Start training");
    expect(start.disabled).toBe(false);
    await act(async () => {
      start.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    await settle();
    expect(ctx.createTrainingJob).toHaveBeenCalledTimes(1);
    const { config } = ctx.createTrainingJob.mock.calls[0][0];
    expect(config.advanced.iris).toMatchObject({ text_conditioning: "cached", ema_enabled: true, export_dtype: "fp32" });
    expect(config.advanced.iris).not.toHaveProperty("resume_from");
    expect(config.advanced.iris).not.toHaveProperty("load_from");
    expect(config.advanced.gradientCheckpointing).toBe(false);
  });

  it("does not run the preflight for a non-Iris target", async () => {
    const legacy = { ...irisTarget(), id: "sdxl_lora", kernel: "sdxl_lora", memoryFloors: undefined };
    await renderAndPickDataset(context({ trainingTargets: { targets: [legacy] } }));
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/training/data-walk"))).toBe(false);
    expect(container.textContent).not.toContain(WARNING);
    expect(container.querySelector("details.training-iris-options")).toBeNull();
  });
});
