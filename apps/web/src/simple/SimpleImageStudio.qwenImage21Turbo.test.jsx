// sc-25908 (epic 25899) — Qwen Image 2.1 Turbo in the SIMPLE image studio.
//
// Turbo is the distilled sibling of Qwen Image 2.1 under its own catalog id, and serves everything
// base serves (E3). Every control below is driven by the SHIPPED manifest entry — a hand fixture
// would only prove the fixture — and the rewriters are fed the way production feeds them: in the
// whole catalog (`models`), never in `imageModels`, which holds generation pickers only and so can
// never contain a `type: "utility"` rewriter entry.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

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

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    recentVideoAssets: [],
    jobs: [],
    imageModels: [TURBO],
    videoModels: [],
    audioModels: [],
    models: [TURBO, REWRITE_T2I, REWRITE_I2I],
    loras: [],
    imageLocalJobs: [],
    videoLocalJobs: [],
    audioLocalJobs: [],
    visibleWorkers: [],
    macCapabilities: null,
    theme: "light",
    changeTheme: () => {},
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createVideoJob: vi.fn(async () => null),
    createAudioJob: vi.fn(async () => null),
    createModelDownloadJob: vi.fn(async () => null),
    createLoraDownloadJob: vi.fn(async () => null),
    jobAction: vi.fn(async () => {}),
    rememberLocalGenerationJob: vi.fn(),
    refinePrompt: vi.fn(),
    qwenRewritePrompt: vi.fn(async () => ({ refinedPrompt: "a long descriptive prompt" })),
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

const buttonWithText = (container, text) =>
  [...container.querySelectorAll("button")].find((button) => button.textContent.includes(text));

describe("SimpleImageStudio with Qwen Image 2.1 Turbo (sc-25908)", () => {
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

  async function openImage(context = baseContext()) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <SimpleShell
            accent="teal"
            lockedToSimple={false}
            onAccentChange={() => {}}
            onModeChange={() => {}}
            onSimpleDefaultChange={() => {}}
            simpleDefault
          />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    return context;
  }

  async function openAdvanced() {
    const fold = container.querySelector("details.su-advanced");
    expect(fold, "the advanced disclosure renders").toBeTruthy();
    await act(async () => {
      fold.open = true;
      fold.dispatchEvent(new Event("toggle", { bubbles: false }));
    });
  }

  // *Mutation that reds this:* looking the rewriter up in `imageModels` (the pre-sc-25908 code) —
  // the utility entry is never there — or keying the rewriter gate on base's id alone.
  it("offers the Qwen rewriter tile and sends the rewrite with Turbo as the target", async () => {
    const context = await openImage();
    await act(async () => setTextarea(container.querySelector("#su-image-prompt"), "a lighthouse"));
    const tile = buttonWithText(container, "Qwen rewriter");
    expect(tile, "the rewriter tile renders for Turbo").toBeTruthy();
    await click(tile);
    await click(buttonWithText(container, "Rewrite prompt"));
    expect(context.qwenRewritePrompt).toHaveBeenCalledWith(
      expect.objectContaining({ modelId: "qwen_image_2_1_turbo", prompt: "a lighthouse" }),
    );
  });

  it("offers the seven presets in the size sheet", async () => {
    await openImage();
    const trigger = [...container.querySelectorAll(".su-select")].find((node) =>
      node.textContent.includes("1:1"),
    );
    await click(trigger);
    const tiles = [...container.querySelectorAll(".su-sheet .su-option-tile")];
    expect(tiles).toHaveLength(TURBO.limits.resolutions.length);
    expect(tiles).toHaveLength(7);
    for (const preset of TURBO.limits.resolutions) {
      const [width, height] = preset.split("x");
      expect(
        tiles.some((tile) => tile.textContent.includes(width) && tile.textContent.includes(height)),
        preset,
      ).toBe(true);
    }
  });

  // *Mutation that reds this:* reading the step default from base's entry (40) or clamping the
  // field to the distilled count.
  it("defaults steps to the distilled 8, floors them at 1, and sends another count unchanged", async () => {
    const context = await openImage();
    await openAdvanced();
    const steps = container.querySelector("#su-image-steps");
    expect(steps.getAttribute("placeholder")).toBe("8");
    expect(steps.getAttribute("min")).toBe("1");

    await act(async () => setTextarea(container.querySelector("#su-image-prompt"), "a lighthouse"));
    await act(async () => setInput(steps, "20"));
    await click(container.querySelector(".su-generate"));
    expect(context.createImageJob.mock.calls[0][0].advanced.steps).toBe(20);
  });

  it("opens the ordered reference list at Turbo's cap of ten", async () => {
    await openImage();
    await openAdvanced();
    const label = container.querySelector('label[for="su-image-add-reference"]');
    expect(label?.textContent).toBe("References (up to 10, in order)");
  });

  it("offers the transparency toggle with the prompt wording it needs", async () => {
    const context = await openImage();
    const toggle = container.querySelector(".su-transparency-toggle input");
    expect(toggle).toBeTruthy();
    await act(async () => setTextarea(container.querySelector("#su-image-prompt"), "a courier"));
    await click(toggle);
    await click(buttonWithText(container, "Ask for transparency"));
    expect(container.querySelector("#su-image-prompt").value).toContain("The background is transparent.");
    await click(container.querySelector(".su-generate"));
    expect(context.createImageJob.mock.calls[0][0].advanced.transparentBackground).toBe(true);
  });
});
