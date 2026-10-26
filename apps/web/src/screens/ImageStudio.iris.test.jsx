import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";

// Studio UI coverage for Iris-3B (sc-25679 / sc-25681, epic 25678): it rides the generic catalog
// machinery, so these assert the controls the catalog row declares — exactly the engine's surface:
// the solver-order sampler menu, the schedule shift (no scheduler menu), the CFG interval, the
// caption-overflow policy; no guidance method or tier. The release step default (100) is reachable,
// the CFG-bound controls (negative prompt, interval) are disabled and not sent while guidance is
// 1.0, every control reaches the job request, and the sub-1024 patch-grid hint appears.

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path) =>
      path === "/api/v1/host-capabilities"
        ? { memoryGb: 64, memoryKind: "unified", platform: "macos" }
        : {},
    ),
  };
});

import { AppContext } from "../context/AppContext.js";
import { ImageStudio } from "./ImageStudio.jsx";

// Mirrors the shipped catalog entry's control-bearing keys.
const IRIS = {
  id: "iris_3b",
  name: "Iris 3B",
  type: "image",
  family: "iris",
  capabilities: ["text_to_image"],
  image: {
    negativePromptRequiresGuidance: true,
    supportsSchedulerShift: true,
    supportsCfgInterval: true,
    captionOverflowPolicies: ["warn", "error", "silent"],
    patchGridVisibleBelow: 1024,
    maxPromptBatchImages: 8,
  },
  defaults: {
    resolution: "1024x1024",
    steps: 100,
    guidanceScale: 3.0,
    sampler: "dpmpp_2m",
    schedulerShift: 4.0,
    count: 1,
  },
  limits: {
    resolutions: ["1024x1024", "1152x896", "896x1152", "512x512"],
    count: [1, 2, 4, 8],
    minDimension: 16,
    maxDimension: 2048,
    requiresDimensionsMultipleOf: 16,
    samplers: ["dpmpp_2m", "euler"],
  },
  loraCompatibility: { families: ["iris"], types: ["style"], irisTask: "generation" },
  ui: { label: "Iris 3B" },
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [IRIS],
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
    (node) => node.querySelector("input, select, textarea") && node.firstChild?.textContent?.trim() === text,
  );
const field = (container, text) => labelled(container, text)?.querySelector("input, select, textarea");
const generateButton = () =>
  [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate");

function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

describe("ImageStudio — Iris-3B (sc-25679)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    await click(document.body.querySelector(".advanced-section-toggle"));
    await act(async () => {});
  }

  it("offers exactly the controls the engine honors, with the release step default reachable", async () => {
    await render(baseContext());
    const sampler = field(container, "Sampler");
    expect([...sampler.options].map((option) => option.value)).toEqual(["euler", "dpmpp_2m"]);
    expect(sampler.value).toBe("dpmpp_2m");
    expect(field(container, "Scheduler")).toBeUndefined();
    expect(field(container, "Schedule shift").value).toBe("4");
    expect(field(container, "Guidance method")).toBeUndefined();
    expect(field(container, "Guidance interval")).toBeTruthy();
    const overflow = container.querySelector('select[aria-label="Caption overflow"]');
    expect([...overflow.options].map((option) => option.value)).toEqual(["default", "warn", "error", "silent"]);
    const steps = field(container, "Steps");
    expect(steps.getAttribute("max")).toBe("100");
    expect(steps.getAttribute("placeholder")).toBe("100");
    expect(field(container, "Guidance")).toBeTruthy();
    expect(field(container, "Width override").getAttribute("step")).toBe("16");
  });

  it("disables the negative prompt and sends none while guidance is 1.0", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-iris" }));
    await render(baseContext({ createImageJob }));
    const negative = field(container, "Negative prompt");
    expect(negative.disabled).toBe(false);
    await act(async () => setTextarea(negative, "blurry"));

    await act(async () => setInput(field(container, "Guidance"), "1"));
    expect(field(container, "Negative prompt").disabled).toBe(true);
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].negativePrompt).toBe("");

    await act(async () => setInput(field(container, "Guidance"), "3"));
    expect(field(container, "Negative prompt").disabled).toBe(false);
    await click(generateButton());
    expect(createImageJob.mock.calls[1][0].negativePrompt).toBe("blurry");
  });

  const intervalInput = () => field(container, "Guidance interval");

  it("sends every control to the job request", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-iris" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setSelect(field(container, "Sampler"), "euler"));
    await act(async () => setInput(field(container, "Schedule shift"), "2.5"));
    await act(async () => setInput(field(container, "Steps"), "30"));
    await act(async () => setInput(field(container, "Guidance"), "4.5"));
    await act(async () => setInput(intervalInput(), "0.1, 0.9"));
    await act(async () =>
      setSelect(container.querySelector('select[aria-label="Caption overflow"]'), "error"),
    );
    await act(async () => setTextarea(field(container, "Negative prompt"), "blurry"));
    await click(generateButton());
    const request = createImageJob.mock.calls[0][0];
    expect(request.model).toBe("iris_3b");
    expect(request.negativePrompt).toBe("blurry");
    expect(request.advanced).toMatchObject({
      sampler: "euler",
      schedulerShift: 2.5,
      steps: 30,
      guidanceScale: 4.5,
      cfgInterval: [0.1, 0.9],
      captionOverflow: "error",
    });
  });

  it("omits the defaults, and disables and withholds the interval while guidance is 1.0", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-iris" }));
    await render(baseContext({ createImageJob }));
    await click(generateButton());
    const defaults = createImageJob.mock.calls[0][0].advanced;
    expect(defaults.cfgInterval).toBeUndefined();
    expect(defaults.captionOverflow).toBeUndefined();
    // The shift axis rides on its own (there is no scheduler picker to gate it on).
    expect(defaults.schedulerShift).toBe(4);

    await act(async () => setInput(intervalInput(), "0.2, 0.8"));
    await act(async () => setInput(field(container, "Guidance"), "1"));
    expect(intervalInput().disabled).toBe(true);
    await click(generateButton());
    expect(createImageJob.mock.calls[1][0].advanced.cfgInterval).toBeUndefined();
  });

  it("refuses an impossible interval before submitting", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-iris" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setInput(intervalInput(), "0.9, 0.1"));
    expect(container.textContent).toContain("Need 0 ≤ start < end ≤ 1.");
    await click(generateButton());
    expect(createImageJob).not.toHaveBeenCalled();
  });

  it("hints at the patch grid below 1024 and never restricts the size", async () => {
    await render(baseContext());
    const hint = () =>
      [...container.querySelectorAll('[role="note"]')].find((node) => node.textContent.includes("patch grid"));
    expect(hint()).toBeUndefined();
    await act(async () => setSelect(field(container, "Aspect"), "512x512"));
    expect(hint()?.textContent).toBe("Below 1024px a side, Iris 3B shows a visible patch grid.");
    await act(async () => setSelect(field(container, "Aspect"), "1152x896"));
    expect(hint()?.textContent).toMatch(/patch grid/);
    await act(async () => setSelect(field(container, "Aspect"), "1024x1024"));
    expect(hint()).toBeUndefined();
    // The free size override still admits a sub-1024 size on the 16-px grid.
    expect(field(container, "Width override").getAttribute("min")).toBe("16");
  });

  it("restores the recorded controls from a recipe", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-iris" }));
    await render(
      baseContext({
        createImageJob,
        studioLaunch: {
          id: "launch-1",
          view: "Image",
          recipe: {
            model: "iris_3b",
            prompt: "a lighthouse",
            normalizedSettings: { width: 1024, height: 1024 },
            rawAdapterSettings: {
              sampler: "euler",
              schedulerShift: 6,
              cfgInterval: [0.25, 0.75],
              captionOverflow: "silent",
            },
          },
        },
      }),
    );
    await click(generateButton());
    const advanced = createImageJob.mock.calls.at(-1)?.[0]?.advanced ?? {};
    expect(advanced).toMatchObject({
      sampler: "euler",
      schedulerShift: 6,
      cfgInterval: [0.25, 0.75],
      captionOverflow: "silent",
    });
  });

  it("restores the controls from the workspace's saved settings", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-image-project_1",
      JSON.stringify({
        model: "iris_3b",
        schedulerShift: 5,
        cfgInterval: "0.2, 0.8",
        captionOverflow: "silent",
      }),
    );
    const createImageJob = vi.fn(async () => ({ id: "job-iris" }));
    await render(baseContext({ createImageJob }));
    expect(intervalInput().value).toBe("0.2, 0.8");
    expect(field(container, "Schedule shift").value).toBe("5");
    await click(generateButton());
    expect(createImageJob.mock.calls.at(-1)[0].advanced).toMatchObject({
      schedulerShift: 5,
      cfgInterval: [0.2, 0.8],
      captionOverflow: "silent",
    });
    // …and the writer keeps them in the snapshot.
    const snapshot = JSON.parse(window.localStorage.getItem("sceneworks-studio-image-project_1") || "{}");
    expect(snapshot).toMatchObject({ cfgInterval: "0.2, 0.8", captionOverflow: "silent" });
  });

  async function runBatch(text) {
    await click(container.querySelector(".batch-toggle"));
    await act(async () => {});
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Batch prompts"]'), text));
    const run = [...document.body.querySelectorAll("button")].find((b) => b.textContent.includes("Run batch"));
    await click(run);
    await act(async () => {});
  }

  // sc-25681: Iris renders a prompt list in ONE pass, so the prompt batch packs into native jobs of
  // up to floor(8 / count) same-size prompts; `prompt` is the first, `prompts` the whole chunk.
  // Mutation that reds this: `promptBatchSize` returning 1 (one job per prompt).
  it("packs the prompt batch into native batch jobs of up to 8 / count prompts", async () => {
    const createImageJob = vi.fn(async () => ({ id: `job-${createImageJob.mock.calls.length}` }));
    await render(baseContext({ createImageJob }));
    await act(async () => setInput(field(container, "Variations"), "4"));
    await runBatch("a red fox\na blue whale\na green frog\n[512x512] a gold bird");
    const requests = createImageJob.mock.calls.map(([request]) => request);
    expect(requests.map((request) => [request.prompt, request.prompts, request.count])).toEqual([
      ["a red fox", ["a red fox", "a blue whale"], 4],
      ["a green frog", undefined, 4],
      ["a gold bird", undefined, 4],
    ]);
    expect(requests[2]).toMatchObject({ width: 512, height: 512 });
  });

  it("packs equivalent ratios together and separates different resolved sizes", async () => {
    const createImageJob = vi.fn(async () => ({ id: `job-${createImageJob.mock.calls.length}` }));
    await render(baseContext({ createImageJob }));
    await runBatch("[4:3] a red fox\n[8:6] a blue whale\n[3:4] a green frog\n[1:1] a gold bird");
    const requests = createImageJob.mock.calls.map(([request]) => request);
    expect(requests.map(({ prompt, prompts, width, height }) => [prompt, prompts, width, height])).toEqual([
      ["a red fox", ["a red fox", "a blue whale"], 1152, 896],
      ["a green frog", undefined, 896, 1152],
      ["a gold bird", undefined, 1024, 1024],
    ]);
  });

  // Mutation that reds this: drop the `promptBatch` branch of the recipe effect.
  it("reopens a native batch image as the whole prompt batch", async () => {
    await render(
      baseContext({
        studioLaunch: {
          id: "launch-batch",
          view: "Image",
          recipe: {
            model: "iris_3b",
            prompt: "a blue whale",
            normalizedSettings: { width: 1024, height: 1024 },
            rawAdapterSettings: { promptBatch: ["a red fox", "a blue whale"] },
          },
        },
      }),
    );
    expect(container.querySelector('textarea[aria-label="Batch prompts"]')?.value).toBe("a red fox\na blue whale");
  });
});
