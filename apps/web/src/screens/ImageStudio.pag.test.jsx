import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, settle, unmountRoot } from "../testUtils/dom.js";
import { recipeFromWorkflowShare } from "../workflowShare.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path) =>
      path === "/api/v1/host-capabilities"
        ? { memoryGb: 128, memoryKind: "unified", platform: "macos" }
        : {},
    ),
  };
});

import { AppContext } from "../context/AppContext.js";
import { ImageStudio } from "./ImageStudio.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const CATALOG_PATH = resolve(HERE, "../testUtils/catalogBaseline/candidate-catalog.json");
const catalogModels = JSON.parse(readFileSync(CATALOG_PATH, "utf8")).models;

const installed = (id, extra = {}) => ({
  ...catalogModels.find((model) => model.id === id),
  installState: "installed",
  usable: true,
  ...extra,
});

const MINIMAX_H3 = installed("minimax_h3_image");
const MOSS_SFT = installed("moss_image_sft", {
  conditionalPurposes: {
    moss_dmd_unmerged_bf16: { installState: "installed" },
    moss_image_edit: { installState: "installed" },
  },
});
const FLUX2_KLEIN_KV = installed("flux2_klein_9b_kv");
const Z_IMAGE_EDIT = installed("z_image_edit");
const KREA_2_RAW = installed("krea_2_raw");
const QWEN_2_1_TURBO = installed("qwen_image_2_1_turbo");
const REALVISXL = installed("realvisxl");
const IRIS_3B = installed("iris_3b");

const SOURCE_IMAGE = {
  id: "asset_ref_1",
  type: "image",
  projectId: "project_1",
  displayName: "Reference",
  width: 1024,
  height: 1024,
  status: {},
  file: { path: "assets/images/reference.png", mimeType: "image/png" },
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [SOURCE_IMAGE],
    characters: [],
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createPreset: vi.fn(async (payload) => ({ id: payload.id, ...payload })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [
      KREA_2_RAW,
      MINIMAX_H3,
      MOSS_SFT,
      FLUX2_KLEIN_KV,
      Z_IMAGE_EDIT,
      QWEN_2_1_TURBO,
      REALVISXL,
      IRIS_3B,
    ],
    models: [
      KREA_2_RAW,
      MINIMAX_H3,
      MOSS_SFT,
      FLUX2_KLEIN_KV,
      Z_IMAGE_EDIT,
      QWEN_2_1_TURBO,
      REALVISXL,
      IRIS_3B,
    ],
    importAsset: vi.fn(),
    latestImageAssets: [],
    recentImageAssets: [],
    studioLaunch: null,
    imageLocalJobs: [],
    loras: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setPreviewAsset: vi.fn(),
    presets: [],
    requestedGpu: "",
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    ...overrides,
  };
}

const labelled = (container, text) =>
  [...container.querySelectorAll("label")].find(
    (node) =>
      node.querySelector("input, select, textarea") &&
      (node.firstChild?.textContent?.trim() === text || node.textContent.trim().startsWith(text)),
  );
const field = (container, text) => labelled(container, text)?.querySelector("input, select, textarea");
const generateButton = () =>
  [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate");
const modeTab = (container, label) =>
  [...container.querySelectorAll('button[role="tab"]')].find((tab) => tab.textContent.trim() === label);

async function submitGenerate() {
  await settle({ microtasks: 12 });
  await click(generateButton());
}

function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

describe("ImageStudio — Perturbed-Attention Guidance (PAG, SW-7181)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context, { openAdvanced = true } = {}) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    if (openAdvanced) {
      const toggle = document.body.querySelector(".advanced-section-toggle");
      if (toggle && toggle.getAttribute("aria-expanded") !== "true") {
        await click(toggle);
        await act(async () => {});
      }
    }
  }

  it.each([
    ["minimax_h3_image", "text_to_image", MINIMAX_H3],
    ["moss_image_sft", "text_to_image", MOSS_SFT],
    ["flux2_klein_9b_kv", "text_to_image", FLUX2_KLEIN_KV],
    ["krea_2_raw", "text_to_image", KREA_2_RAW],
    ["qwen_image_2_1_turbo", "text_to_image", QWEN_2_1_TURBO],
    ["z_image_edit", "edit_image", Z_IMAGE_EDIT],
  ])("exposes capability-driven PAG controls and model hints on %s (%s)", async (_id, mode, model) => {
    await render(baseContext({ imageModels: [model], models: [model] }));
    if (mode === "edit_image" && modeTab(container, "Edit")) {
      await click(modeTab(container, "Edit"));
      await act(async () => {});
    }
    const scaleInput = field(container, "PAG scale");
    expect(scaleInput).toBeTruthy();
    expect(scaleInput.value).toBe("");
    expect(scaleInput.getAttribute("placeholder")).toBe("0 (Off)");
    expect(scaleInput.getAttribute("step")).toBe("any");

    const hintText = labelled(container, "PAG scale").querySelector(".field-hint")?.textContent;
    expect(hintText).toContain(`Recommended: ${model.pag.recommendedScale}`);
    expect(hintText).toContain(`default blocks [${model.pag.defaultLayers.join(", ")}]`);
    expect(hintText).toContain(`window [${model.pag.defaultStart}, ${model.pag.defaultStop})`);

    const details = container.querySelector("details.pag-advanced");
    expect(details).toBeTruthy();
    expect(details.open).toBe(false);
    expect(field(container, "PAG blocks").getAttribute("placeholder")).toBe(
      `Default (${model.pag.defaultLayers.join(", ")})`,
    );
    expect(field(container, "PAG window start").getAttribute("placeholder")).toBe(String(model.pag.defaultStart));
    expect(field(container, "PAG window stop").getAttribute("placeholder")).toBe(String(model.pag.defaultStop));
    expect(field(container, "PAG adaptive scale").getAttribute("placeholder")).toBe("0");
  });

  it.each([
    ["realvisxl", REALVISXL],
    ["iris_3b", IRIS_3B],
  ])("hides PAG controls on non-PAG model %s", async (_id, model) => {
    await render(baseContext({ imageModels: [model], models: [model] }));
    expect(field(container, "PAG scale")).toBeUndefined();
    expect(container.querySelector("details.pag-advanced")).toBeNull();
  });

  it("shows PAG controls on MOSS T2I (ordinary + DMD) and MOSS Edit and emits advanced.pag in MOSS Edit", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-moss-pag" }));
    await render(baseContext({ createImageJob, imageModels: [MOSS_SFT], models: [MOSS_SFT] }));

    // Enable 8-step acceleration in MOSS T2I: PAG remains visible.
    const dmdCheckbox = labelled(container, "8-step acceleration").querySelector("input");
    await click(dmdCheckbox);
    await act(async () => {});
    expect(field(container, "PAG scale")).toBeTruthy();

    // Switch to Edit mode (mossEdit === true): PAG stays visible and emits advanced.pag.
    await render(
      baseContext({
        createImageJob,
        imageModels: [MOSS_SFT],
        models: [MOSS_SFT],
        studioLaunch: {
          id: "moss-edit-pag",
          view: "Image",
          recipe: {
            mode: "edit_image",
            model: MOSS_SFT.id,
            prompt: "make it snow",
            loras: [],
            normalizedSettings: {},
            rawAdapterSettings: {
              referenceAssetIds: [SOURCE_IMAGE.id],
              mossImageEdit: { reference_max_edge: 1024, use_kv_cache: true },
            },
          },
        },
      }),
    );
    expect(field(container, "PAG scale")).toBeTruthy();

    await act(async () => setInput(field(container, "PAG scale"), "2.5"));
    await submitGenerate();
    expect(createImageJob).toHaveBeenCalledTimes(1);
    expect(createImageJob.mock.calls[0][0].mode).toBe("edit_image");
    expect(createImageJob.mock.calls[0][0].advanced.pag).toEqual({ scale: 2.5 });
  });

  it("omits advanced.pag when blank or 0, sends clean payloads with optional overrides, and blocks invalid inputs", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-pag" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a neon alley"));

    // Blank scale -> no advanced.pag
    await submitGenerate();
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("pag");

    // Explicit 0 -> no advanced.pag
    await act(async () => setInput(field(container, "PAG scale"), "0"));
    await submitGenerate();
    expect(createImageJob.mock.calls[1][0].advanced).not.toHaveProperty("pag");

    // Active scale with blank overrides -> clean { scale }
    await act(async () => setInput(field(container, "PAG scale"), "1.5"));
    await submitGenerate();
    expect(createImageJob.mock.calls[2][0].advanced.pag).toEqual({ scale: 1.5 });

    // Open collapsible overrides and set layers, start/stop, adaptiveScale
    await click(container.querySelector("details.pag-advanced summary"));
    await act(async () => setInput(field(container, "PAG blocks"), "2, 5"));
    await act(async () => setInput(field(container, "PAG window start"), "0.1"));
    await act(async () => setInput(field(container, "PAG window stop"), "0.9"));
    await act(async () => setInput(field(container, "PAG adaptive scale"), "0.25"));
    await submitGenerate();
    expect(createImageJob.mock.calls[3][0].advanced.pag).toEqual({
      scale: 1.5,
      layers: [2, 5],
      start: 0.1,
      stop: 0.9,
      adaptiveScale: 0.25,
    });

    // Out-of-range block index (>= 28 on krea_2_raw) shows inline alert and blocks submit.
    await act(async () => setInput(field(container, "PAG blocks"), "28"));
    const alert = labelled(container, "PAG scale").querySelector('[role="alert"]');
    expect(alert?.textContent).toContain("0 and 27");
    await submitGenerate();
    expect(createImageJob).toHaveBeenCalledTimes(4);

    // Inverted window (start > stop) shows inline alert and blocks submit.
    await act(async () => setInput(field(container, "PAG blocks"), "2, 5"));
    await act(async () => setInput(field(container, "PAG window start"), "0.95"));
    expect(labelled(container, "PAG scale").querySelector('[role="alert"]')?.textContent).toContain("≤ stop");
    await submitGenerate();
    expect(createImageJob).toHaveBeenCalledTimes(4);
  });

  it("resets PAG and never leaks advanced.pag when switching to a non-PAG model", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-switch" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setInput(field(container, "PAG scale"), "1.5"));
    await act(async () => setSelect(field(container, "Model"), REALVISXL.id));
    await act(async () => {});
    expect(field(container, "PAG scale")).toBeUndefined();

    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a harbour"));
    await submitGenerate();
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("pag");
  });

  it("saves and restores PAG in local settings, presets, GenerationSet recipes, and imported workflow PNGs", async () => {
    const buttonWithText = (text) => [...container.querySelectorAll("button")].find((b) => b.textContent.includes(text));
    const context = baseContext({ createImageJob: vi.fn(async () => ({ id: "job-roundtrip" })) });
    await render(context);

    await act(async () => setInput(field(container, "PAG scale"), "1.8"));
    await act(async () => setInput(field(container, "PAG blocks"), "4, 14"));
    await act(async () => setInput(field(container, "PAG window start"), "0.15"));
    await act(async () => setInput(field(container, "PAG window stop"), "0.85"));
    await act(async () => setInput(field(container, "PAG adaptive scale"), "0.3"));

    // 1. Studio local storage persistence across remount.
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await render(context);
    expect(field(container, "PAG scale").value).toBe("1.8");
    expect(field(container, "PAG blocks").value).toBe("4, 14");
    expect(field(container, "PAG window start").value).toBe("0.15");
    expect(field(container, "PAG window stop").value).toBe("0.85");
    expect(field(container, "PAG adaptive scale").value).toBe("0.3");

    // 2. Preset save & apply round-trip.
    await act(async () => setInput(container.querySelector('input[aria-label="Preset name"]'), "Krea PAG"));
    await click(buttonWithText("Save as preset"));
    await click(
      [...container.querySelectorAll(".save-preset-scope-picker .compact-selector-item")].find((b) =>
        b.textContent.includes("This project"),
      ),
    );
    const savedPreset = context.createPreset.mock.calls[0][0];
    expect(savedPreset.defaults.pag).toEqual({
      scale: 1.8,
      layers: [4, 14],
      start: 0.15,
      stop: 0.85,
      adaptiveScale: 0.3,
    });

    // Clear PAG manually, then apply the saved preset -> PAG controls restore intact,
    // and clicking "None" restores the prior Off state without manual clearing.
    await render({ ...context, presets: [savedPreset] }, { openAdvanced: true });
    await click([...container.querySelectorAll(".preset-chips button")].find((b) => b.textContent === "None"));
    await act(async () => {
      setInput(field(container, "PAG scale"), "");
      setInput(field(container, "PAG blocks"), "");
      setInput(field(container, "PAG window start"), "");
      setInput(field(container, "PAG window stop"), "");
      setInput(field(container, "PAG adaptive scale"), "");
    });
    await click(buttonWithText("Krea PAG"));
    expect(field(container, "PAG scale").value).toBe("1.8");
    expect(field(container, "PAG blocks").value).toBe("4, 14");
    expect(field(container, "PAG window start").value).toBe("0.15");
    expect(field(container, "PAG window stop").value).toBe("0.85");
    expect(field(container, "PAG adaptive scale").value).toBe("0.3");
    await click([...container.querySelectorAll(".preset-chips button")].find((b) => b.textContent === "None"));
    expect(field(container, "PAG scale").value).toBe("");
    expect(field(container, "PAG blocks").value).toBe("");
    expect(field(container, "PAG window start").value).toBe("");
    expect(field(container, "PAG window stop").value).toBe("");
    expect(field(container, "PAG adaptive scale").value).toBe("");

    // Fine-grained decimals (e.g. 1.25 / 0.125) pass native HTML5 form validity (`step="any"`).
    await act(async () => {
      setInput(field(container, "PAG scale"), "1.25");
      setInput(field(container, "PAG window start"), "0.125");
    });
    expect(container.querySelector("form.studio-shell").checkValidity()).toBe(true);

    // 3. GenerationSet recipe 'Use Settings' replay scrubs pagExecutionReport and restores pag.
    const recipeLaunch = {
      id: "reuse-recipe",
      view: "Image",
      recipe: {
        mode: "text_to_image",
        model: MINIMAX_H3.id,
        prompt: "stormy coastline",
        loras: [],
        normalizedSettings: {},
        rawAdapterSettings: {
          pag: { scale: 1.2, layers: [10, 25], start: 0.05, stop: 0.65, adaptiveScale: 0.2 },
          pagExecutionReport: {
            scale: 1.2,
            adaptiveScale: 0.2,
            layers: [10, 25],
            start: 0.05,
            stop: 0.65,
            activeSteps: 4,
            totalSteps: 6,
          },
        },
      },
    };
    await render(baseContext({ createImageJob: context.createImageJob, studioLaunch: recipeLaunch }));
    expect(field(container, "PAG scale").value).toBe("1.2");
    expect(field(container, "PAG blocks").value).toBe("10, 25");
    expect(field(container, "PAG window start").value).toBe("0.05");
    expect(field(container, "PAG window stop").value).toBe("0.65");
    expect(field(container, "PAG adaptive scale").value).toBe("0.2");
    await submitGenerate();
    const recipeJob = context.createImageJob.mock.calls.at(-1)[0];
    expect(recipeJob.advanced.pag).toEqual({
      scale: 1.2,
      layers: [10, 25],
      start: 0.05,
      stop: 0.65,
      adaptiveScale: 0.2,
    });
    expect(recipeJob.advanced).not.toHaveProperty("pagExecutionReport");

    // 4. Imported workflow PNG prefill restores pag into Studio controls while scrubbing pagExecutionReport.
    const shareEnvelope = {
      schemaVersion: 1,
      sceneworksWorkflow: "image",
      mode: "text_to_image",
      model: QWEN_2_1_TURBO.id,
      prompt: "ceramic teapot",
      negativePrompt: "",
      width: 1024,
      height: 1024,
      loras: [],
      inputs: [],
      advanced: {
        pag: { scale: 1.4, layers: [8, 16], start: 0.1, stop: 0.7, adaptiveScale: 0.15 },
        pagExecutionReport: {
          scale: 1.4,
          adaptiveScale: 0.15,
          layers: [8, 16],
          start: 0.1,
          stop: 0.7,
          activeSteps: 6,
          totalSteps: 8,
        },
      },
    };
    const report = { model: { state: "resolved", catalogId: QWEN_2_1_TURBO.id }, loras: [], styles: [], inputs: [], runnable: true };
    const importedRecipe = recipeFromWorkflowShare(shareEnvelope, report, { models: [QWEN_2_1_TURBO] });
    expect(importedRecipe.rawAdapterSettings).not.toHaveProperty("pagExecutionReport");
    await render(
      baseContext({
        createImageJob: context.createImageJob,
        studioLaunch: { id: "png-import", view: "Image", recipe: importedRecipe },
      }),
    );
    expect(field(container, "PAG scale").value).toBe("1.4");
    expect(field(container, "PAG blocks").value).toBe("8, 16");
    expect(field(container, "PAG window start").value).toBe("0.1");
    expect(field(container, "PAG window stop").value).toBe("0.7");
    expect(field(container, "PAG adaptive scale").value).toBe("0.15");
    await submitGenerate();
    const importedJob = context.createImageJob.mock.calls.at(-1)[0];
    expect(importedJob.advanced.pag).toEqual({
      scale: 1.4,
      layers: [8, 16],
      start: 0.1,
      stop: 0.7,
      adaptiveScale: 0.15,
    });
    expect(importedJob.advanced).not.toHaveProperty("pagExecutionReport");
  });
});
