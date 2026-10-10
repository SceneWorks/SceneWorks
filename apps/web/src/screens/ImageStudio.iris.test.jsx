import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

// Studio UI coverage for Iris-3B (sc-25679, epic 25678): it rides the generic catalog machinery, so
// these assert only the controls this story owns — no control the engine ignores is offered
// (sampler / scheduler / guidance method / tier), the release step default (100) is reachable, and
// the negative prompt (the CFG unconditional) is disabled and not sent while guidance is 1.0.

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
  image: { negativePromptRequiresGuidance: true },
  defaults: { resolution: "1024x1024", steps: 100, guidanceScale: 3.0, count: 1 },
  limits: {
    resolutions: ["1024x1024", "1152x896", "896x1152", "512x512"],
    count: [1, 2, 4, 8],
    minDimension: 16,
    maxDimension: 2048,
    requiresDimensionsMultipleOf: 16,
  },
  loraCompatibility: { families: [], types: [] },
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

  it("offers only the controls the engine honors, with the release step default reachable", async () => {
    await render(baseContext());
    expect(field(container, "Sampler")).toBeUndefined();
    expect(field(container, "Scheduler")).toBeUndefined();
    expect(field(container, "Guidance method")).toBeUndefined();
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
});
