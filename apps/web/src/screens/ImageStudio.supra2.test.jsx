import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

// Studio UI coverage for Supra2-IMG (sc-24215, epic 24212): the catalog row declares exactly the
// published interface — prompt, seed, guidance, steps, image count at a native 256x256 with a
// 128-token Flan-T5 prompt. So the studio shows those controls and no others (no negative prompt,
// sampler, scheduler, guidance method, shift, interval or caption policy), states the native size
// and the prompt budget, and sends every control on the job request.

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
const SUPRA2 = {
  id: "supra2_img",
  name: "Supra2-IMG",
  type: "image",
  family: "supra2",
  capabilities: ["text_to_image"],
  image: { supportsNegativePrompt: false, maxPromptTokens: 128, nativeCountBatch: true },
  defaults: { resolution: "256x256", steps: 50, guidanceScale: 3.0, count: 1 },
  limits: {
    resolutions: ["256x256"],
    count: [1, 2, 4, 8],
    minDimension: 256,
    maxDimension: 256,
    hardMaxSteps: 100000,
  },
  loraCompatibility: { families: [], types: [] },
  ui: { label: "Supra2-IMG" },
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
    imageModels: [SUPRA2],
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

describe("ImageStudio — Supra2-IMG (sc-24215)", () => {
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

  it("offers exactly the published controls and states the native size and prompt budget", async () => {
    await render(baseContext());
    expect([...field(container, "Aspect").options].map((option) => option.value)).toEqual(["256x256"]);
    expect(container.textContent).toContain("Supra2-IMG renders at its native 256×256 only.");
    expect(container.textContent).toContain(
      "Supra2-IMG reads up to 128 prompt tokens; anything longer is truncated and the image is flagged.",
    );
    const steps = field(container, "Steps");
    // The declared ceiling (gen-core's sanity cap), not the generic 80-step clamp.
    expect(steps.getAttribute("max")).toBe("100000");
    expect(steps.getAttribute("placeholder")).toBe("50");
    expect(field(container, "Guidance")).toBeTruthy();
    expect(field(container, "Seed")).toBeTruthy();
    expect(field(container, "Variations")).toBeTruthy();
    for (const absent of [
      "Negative prompt",
      "Sampler",
      "Scheduler",
      "Schedule shift",
      "Guidance method",
      "Guidance interval",
    ]) {
      expect(field(container, absent), absent).toBeUndefined();
    }
    expect(container.querySelector('select[aria-label="Caption overflow"]')).toBeNull();
  });

  it("sends prompt, seed, guidance, steps and image count at 256x256", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-supra2" }));
    await render(baseContext({ createImageJob }));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a red fox"));
    await act(async () => setInput(field(container, "Variations"), "4"));
    await act(async () => setInput(field(container, "Seed"), "7"));
    await act(async () => setInput(field(container, "Steps"), "30"));
    await act(async () => setInput(field(container, "Guidance"), "1"));
    await click(generateButton());
    const request = createImageJob.mock.calls[0][0];
    expect(request).toMatchObject({
      model: "supra2_img",
      mode: "text_to_image",
      width: 256,
      height: 256,
      count: 4,
      negativePrompt: "",
    });
    expect(String(request.seed)).toBe("7");
    expect(request.prompt).toContain("a red fox");
    expect(request.advanced).toMatchObject({ steps: 30, guidanceScale: 1 });
  });
});
