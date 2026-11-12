import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";
import { evaluateModelDimensions } from "../resolutionOverride.js";
import { fixedStepsForModel } from "../samplerOptions.js";

// SW-7100 (feature SW-F363): MiniMax-H3-Image text-to-image in the advanced Image Studio, from the
// SHIPPED catalog entry — the published 16 MP presets up to a 6272 px edge, no Steps / Guidance /
// Negative prompt control, and the §IV.2 "MiniMax-H3-Image" attribution on the generation surface.

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
const MANIFEST_PATH = resolve(HERE, "../../../../config/manifests/builtin.models.jsonc");
const manifestModels = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8")).models;
const installed = (id) => ({
  ...manifestModels.find((model) => model.id === id),
  installState: "installed",
  usable: true,
});

const H3_IMAGE = installed("minimax_h3_image");

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
    imageModels: [H3_IMAGE],
    models: [H3_IMAGE],
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

describe("ImageStudio — MiniMax-H3-Image (SW-7100)", () => {
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

  it("offers the published presets and hides steps, guidance and negative prompt", async () => {
    await render(baseContext());
    const presets = [...field(container, "Aspect").options].map((option) => option.value);
    expect(presets).toEqual(H3_IMAGE.limits.resolutions);
    expect(presets).toContain("6272x2688");
    expect(field(container, "Aspect").value).toBe("2048x2048");
    for (const absent of ["Steps", "Guidance", "Negative prompt", "Guidance interval"]) {
      expect(field(container, absent), absent).toBeUndefined();
    }
    expect(field(container, "Seed")).toBeTruthy();
    expect(field(container, "Width override").getAttribute("max")).toBe("6272");
    expect(container.querySelector(".studio-model-attribution")?.textContent).toBe(
      "Powered by MiniMax-H3-Image",
    );
  });

  it("sends a 6272x2688 text-to-image job with no steps, guidance or negative prompt", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-h3-image" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a harbour at dawn"));
    await act(async () => setSelect(field(container, "Aspect"), "6272x2688"));
    await act(async () => setInput(field(container, "Seed"), "11"));
    await click(generateButton());
    const request = createImageJob.mock.calls[0][0];
    expect(request).toMatchObject({
      model: "minimax_h3_image",
      mode: "text_to_image",
      width: 6272,
      height: 2688,
      negativePrompt: "",
    });
    expect(String(request.seed)).toBe("11");
    expect(request.advanced?.steps).toBeUndefined();
    expect(request.advanced?.guidanceScale).toBeUndefined();
  });
});

describe("ImageStudio — MiniMax-H3-Image sampler (SW-7106)", () => {
  let container;
  let root;
  const OTHER = installed("qwen_image_2_1_turbo");

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context, { open = true } = {}) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    if (open) await click(document.body.querySelector(".advanced-section-toggle"));
    await act(async () => {});
  }

  const samplerSelect = () => container.querySelector('select[aria-label="MiniMax-H3-Image sampler"]');

  it("sits in the collapsed Advanced section, defaulting to LCM at 1.1 / 0.9 / 0", async () => {
    await render(baseContext(), { open: false });
    expect(samplerSelect()).toBeNull();
    await click(document.body.querySelector(".advanced-section-toggle"));
    expect(samplerSelect().value).toBe("lcm");
    expect([...samplerSelect().options].map((option) => option.textContent)).toEqual([
      "Euler (official)",
      "LCM (stochastic re-noise)",
    ]);
    expect(field(container, "Noise scale start").value).toBe("1.1");
    expect(field(container, "Noise scale end").value).toBe("0.9");
    expect(field(container, "Noise clip (std, 0 = off)").value).toBe("0");
  });

  it("sends the default LCM block 1.1 / 0.9 / 0 with a new job", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-h3-default" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a harbour at dawn"));
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.h3ImageSampler).toEqual({
      sampler: "lcm",
      sNoise: 1.1,
      sNoiseEnd: 0.9,
      noiseClipStd: 0,
    });
  });

  it("replays an old job recorded without the block as Euler", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-h3-old" }));
    await render(baseContext({
      createImageJob,
      studioLaunch: { id: "reuse-old", view: "Image", recipe: { mode: "text_to_image", model: H3_IMAGE.id, prompt: "a cup", loras: [], normalizedSettings: {}, rawAdapterSettings: {} } },
    }));
    expect(samplerSelect().value).toBe("euler");
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.h3ImageSampler).toEqual({ sampler: "euler" });
  });

  it("reveals the LCM fields at their defaults and sends the block with the job", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-h3-lcm" }));
    await render(baseContext({ createImageJob }));
    expect(field(container, "Noise scale start").value).toBe("1.1");
    expect(field(container, "Noise scale end").value).toBe("0.9");
    expect(field(container, "Noise clip (std, 0 = off)").value).toBe("0");
    await act(async () => setInput(field(container, "Noise clip (std, 0 = off)"), "1.5"));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a harbour at dawn"));
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.h3ImageSampler).toEqual({
      sampler: "lcm",
      sNoise: 1.1,
      sNoiseEnd: 0.9,
      noiseClipStd: 1.5,
    });
  });

  it("sends Euler explicitly and refuses an out-of-range LCM value before submitting", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-h3" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setSelect(samplerSelect(), "euler"));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a harbour at dawn"));
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.h3ImageSampler).toEqual({ sampler: "euler" });
    await act(async () => setSelect(samplerSelect(), "lcm"));
    await act(async () => setInput(field(container, "Noise scale start"), "0"));
    expect(container.textContent).toContain("Noise scale start must be between 0.01 and 2.");
    await click(generateButton());
    expect(createImageJob).toHaveBeenCalledTimes(1);
  });

  it("saves LCM 1.1 / 0.9 / 0 in a recipe preset and restores it intact", async () => {
    const buttonWithText = (text) => [...container.querySelectorAll("button")].find((b) => b.textContent.includes(text));
    const context = baseContext({ createImageJob: vi.fn(async () => ({ id: "job-h3-preset" })) });
    await render(context);
    await act(async () => setInput(container.querySelector('input[aria-label="Preset name"]'), "H3 LCM"));
    await click(buttonWithText("Save as preset"));
    await click([...container.querySelectorAll(".save-preset-scope-picker .compact-selector-item")].find((b) => b.textContent.includes("This project")));
    const preset = context.createPreset.mock.calls[0][0];
    const lcm = { sampler: "lcm", sNoise: 1.1, sNoiseEnd: 0.9, noiseClipStd: 0 };
    expect(preset.defaults.h3ImageSampler).toEqual(lcm);

    // The saved preset is selected; deselect it and move to Euler, then apply it again: every LCM
    // field comes back exactly.
    await render({ ...context, presets: [preset] }, { open: false });
    await click([...container.querySelectorAll(".preset-chips button")].find((b) => b.textContent === "None"));
    await act(async () => setSelect(samplerSelect(), "euler"));
    expect(field(container, "Noise scale start")).toBeUndefined();
    await click(buttonWithText("H3 LCM"));
    expect(samplerSelect().value).toBe("lcm");
    expect(field(container, "Noise scale start").value).toBe("1.1");
    expect(field(container, "Noise scale end").value).toBe("0.9");
    expect(field(container, "Noise clip (std, 0 = off)").value).toBe("0");
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a harbour at dawn"));
    await click(generateButton());
    expect(context.createImageJob.mock.calls.at(-1)[0].advanced.h3ImageSampler).toEqual(lcm);
  });

  it("restores the sampler from a gallery recipe (reuse settings)", async () => {
    const lcm = { sampler: "lcm", sNoise: 1.3, sNoiseEnd: 0.5, noiseClipStd: 2 };
    const createImageJob = vi.fn(async () => ({ id: "job-h3-reuse" }));
    await render(baseContext({
      createImageJob,
      studioLaunch: { id: "reuse", view: "Image", recipe: { mode: "text_to_image", model: H3_IMAGE.id, prompt: "a cup", loras: [], normalizedSettings: {}, rawAdapterSettings: { h3ImageSampler: lcm } } },
    }));
    expect(samplerSelect().value).toBe("lcm");
    expect(field(container, "Noise scale start").value).toBe("1.3");
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.h3ImageSampler).toEqual(lcm);
  });

  it("is hidden for every other model", async () => {
    await render(baseContext({ imageModels: [OTHER], models: [OTHER] }));
    expect(samplerSelect()).toBeNull();
    expect(field(container, "Noise scale start")).toBeUndefined();
  });
});

describe("MiniMax-H3-Image geometry and steps are this model's alone (SW-7100)", () => {
  const other = installed("qwen_image_2_1_turbo");
  const plain = { id: "plain", limits: { resolutions: ["1024x1024"] } };

  it("accepts a 6272 px edge for H3-Image and refuses it for other models", () => {
    const at = (model) =>
      evaluateModelDimensions({ model, resolution: "1024x1024", widthOverride: "6272", heightOverride: "2688" });
    expect(at(H3_IMAGE).invalid).toBe(false);
    expect(at(other).invalid).toBe(true);
    expect(at(plain).invalid).toBe(true);
    expect(at(plain).constraints.max).toBe(4096);
  });

  it("fixes steps only where hardMinSteps equals hardMaxSteps", () => {
    expect(fixedStepsForModel(H3_IMAGE)).toBe(6);
    const fixed = manifestModels.filter((model) => fixedStepsForModel(model) != null).map((model) => model.id);
    expect(fixed).toEqual(["minimax_h3_image"]);
    expect(fixedStepsForModel(other)).toBeNull();
    expect(fixedStepsForModel({ limits: { hardMaxSteps: 100000 } })).toBeNull();
  });
});
