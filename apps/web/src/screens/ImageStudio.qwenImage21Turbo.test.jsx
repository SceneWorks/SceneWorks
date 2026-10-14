// sc-25908 (epic 25899) — Qwen Image 2.1 Turbo in the advanced Image Studio.
//
// Turbo serves everything base serves (E3), keyed on the catalog — the family, the declared limits
// and capabilities of the SHIPPED manifest entry — never on base's id. The rewriter entries are fed
// through `models` (the whole catalog), as production does.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

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

const TURBO = installed("qwen_image_2_1_turbo");
const REWRITE_T2I = installed("qwen_image_2_1_pe_t2i");
const REWRITE_I2I = installed("qwen_image_2_1_pe_i2i");
const REFS = Array.from({ length: 11 }, (_, index) => ({
  id: `ref-${index}`,
  projectId: "project_1",
  type: "image",
  displayName: `Ref ${index}`,
  status: {},
}));

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: REFS,
    characters: [],
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    qwenRewritePrompt: vi.fn(async () => ({ refinedPrompt: "a long descriptive prompt" })),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [TURBO],
    models: [TURBO, REWRITE_T2I, REWRITE_I2I],
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

const field = (container, labelText) => {
  const label = [...container.querySelectorAll("label")].find((node) =>
    node.textContent.trim().startsWith(labelText),
  );
  return label?.querySelector("input, select, textarea");
};
const buttonWithText = (text) =>
  [...document.body.querySelectorAll("button")].find((button) => button.textContent.includes(text));
const generateButton = () =>
  [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate");

function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

describe("ImageStudio with Qwen Image 2.1 Turbo (sc-25908)", () => {
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

  async function render(context = baseContext()) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    return context;
  }
  const typePrompt = (text) =>
    act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), text));

  it("offers the seven presets, seeded on the 1:1 default", async () => {
    await render();
    const aspect = field(container, "Aspect");
    expect([...aspect.options].map((option) => option.value)).toEqual(TURBO.limits.resolutions);
    expect(aspect.options).toHaveLength(7);
    expect(aspect.value).toBe("2048x2048");
  });

  // *Mutation that reds this:* keying `qwenRewriterAppliesTo` on base's id (the pre-sc-25908 gate).
  it("offers the Qwen rewriter in Prompt tools and rewrites with Turbo as the target", async () => {
    const context = await render();
    await typePrompt("a lighthouse");
    await click(buttonWithText("Qwen rewriter"));
    await click(buttonWithText("Rewrite prompt"));
    expect(context.qwenRewritePrompt).toHaveBeenCalledWith(
      expect.objectContaining({ modelId: "qwen_image_2_1_turbo", prompt: "a lighthouse" }),
    );
  });

  it("defaults steps to the distilled 8 and sends another count unchanged", async () => {
    const context = await render();
    await click(document.body.querySelector(".advanced-section-toggle"));
    const steps = field(container, "Steps");
    expect(steps.getAttribute("placeholder")).toBe("8");
    expect(steps.getAttribute("min")).toBe("1");
    await typePrompt("a lighthouse");
    await act(async () => setInput(steps, "20"));
    await click(generateButton());
    expect(context.createImageJob.mock.calls[0][0].advanced.steps).toBe(20);
  });

  it("offers the transparency toggle and its prompt wording", async () => {
    const context = await render();
    await click(document.body.querySelector(".advanced-section-toggle"));
    await typePrompt("a courier");
    const toggle = container.querySelector(".transparency-toggle input");
    expect(toggle).toBeTruthy();
    await click(toggle);
    await click(container.querySelector(".transparency-prompt-hint button"));
    expect(container.querySelector('textarea[aria-label="Prompt"]').value).toContain(
      "The background is transparent.",
    );
    await click(generateButton());
    expect(context.createImageJob.mock.calls[0][0].advanced.transparentBackground).toBe(true);
  });

  // *Mutation that reds this:* dropping `referenceOverflowError` from the Generate validation —
  // eleven references then reach the enqueue gate, which 400s them.
  it("shows the ordered reference list at the cap of ten and blocks an eleventh", async () => {
    const context = await render();
    await typePrompt("put the jacket from image 2 on the person in image 1");
    await click([...document.body.querySelectorAll(".mode-tabs button")].find((b) => b.textContent === "Edit"));
    expect(container.textContent).toContain("Reference images (up to 10, in order)");

    const pick = async (count) => {
      await click(
        [...document.body.querySelectorAll(".asset-picker-head button")].find((button) =>
          ["Select images", "Edit references"].includes(button.textContent),
        ),
      );
      const dialog = document.body.querySelector('[role="dialog"]');
      const cards = [...dialog.querySelectorAll(".asset-picker-card")];
      for (const card of cards) {
        if (card.classList.contains("selected")) await click(card);
      }
      for (const card of cards.slice(0, count)) await click(card);
      await click([...dialog.querySelectorAll("button")].find((button) => button.textContent === "Use Selection"));
    };

    await pick(11);
    expect(container.textContent).toContain("takes up to 10 reference images — remove 1.");
    expect(generateButton().disabled).toBe(true);

    await pick(2);
    const ordinals = [...container.querySelectorAll(".ordered-reference-ordinal")].map((node) => node.textContent);
    expect(ordinals).toEqual(["Image 1", "Image 2"]);
    expect(container.textContent).not.toContain("takes up to 10 reference images");
    expect(generateButton().disabled).toBe(false);
    await click(generateButton());
    expect(context.createImageJob.mock.calls[0][0].referenceAssetIds).toHaveLength(2);
  });
});
