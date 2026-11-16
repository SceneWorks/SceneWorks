import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";
import { writeSimpleStudioSnapshot } from "../simpleStudioStore.js";

// SW-7100 (feature SW-F363): MiniMax-H3-Image in the Simple Image Studio, from the SHIPPED catalog
// entry — the published presets up to 6272 px, no Steps / Guidance / Negative prompt, and the
// §IV.2 "MiniMax-H3-Image" attribution.

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
const H3_IMAGE = {
  ...manifestModels.find((model) => model.id === "minimax_h3_image"),
  installState: "installed",
  usable: true,
};

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    recentVideoAssets: [],
    jobs: [],
    imageModels: [H3_IMAGE],
    videoModels: [],
    audioModels: [],
    models: [H3_IMAGE],
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
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

describe("SimpleImageStudio with MiniMax-H3-Image (SW-7100)", () => {
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

  it("shows the attribution and offers every published preset in the size sheet", async () => {
    await openImage();
    expect(container.textContent).toContain("Powered by MiniMax-H3-Image");
    const trigger = [...container.querySelectorAll(".su-select")].find((node) =>
      node.textContent.includes("1:1"),
    );
    await click(trigger);
    const tiles = [...container.querySelectorAll(".su-sheet .su-option-tile")];
    expect(tiles).toHaveLength(H3_IMAGE.limits.resolutions.length);
    expect(tiles.some((tile) => tile.textContent.includes("6272") && tile.textContent.includes("2688"))).toBe(true);
  });

  it("hides steps, guidance and negative prompt, accepts a 6272 px width and sends no steps", async () => {
    const context = await openImage();
    await openAdvanced();
    expect(container.querySelector("#su-image-steps")).toBeNull();
    expect(container.querySelector("#su-image-guidance")).toBeNull();
    expect(container.querySelector("#su-image-negative")).toBeNull();
    expect(container.querySelector("#su-image-seed")).toBeTruthy();
    const width = container.querySelector("#su-image-width");
    expect(width.getAttribute("max")).toBe("6272");
    await act(async () => setTextarea(container.querySelector("#su-image-prompt"), "a harbour at dawn"));
    await act(async () => setInput(width, "6272"));
    await act(async () => setInput(container.querySelector("input[aria-label=\"Custom height\"]"), "2688"));
    await click(container.querySelector(".su-generate"));
    const request = context.createImageJob.mock.calls[0][0];
    expect(request).toMatchObject({ model: "minimax_h3_image", width: 6272, height: 2688 });
    expect(request.advanced?.steps).toBeUndefined();
    expect(request.advanced?.guidanceScale).toBeUndefined();
  });

  // SW-7101: the Edit tab takes up to sixteen ordered references — the armed tile is <Picture 1>,
  // the rest ride the ordered list in the Advanced fold — each with an Insert <Picture N> action,
  // and the edit keeps the t2i surface (no steps / guidance / negative, the LCM default sampler).
  it("edits from sixteen ordered references with <Picture N> insertion", async () => {
    const refs = Array.from({ length: 17 }, (_, index) => ({
      id: `ref-${index}`,
      projectId: "project-1",
      type: "image",
      name: `Ref ${index}`,
      status: {},
    }));
    writeSimpleStudioSnapshot("project-1", {
      image: {
        mode: "edit_image",
        model: "minimax_h3_image",
        prompt: "Put the hat from",
        referenceAssetId: "ref-0",
        extraReferenceIds: refs.slice(1, 17).map((ref) => ref.id),
        advancedOpen: true,
      },
    });
    const context = await openImage(baseContext({ assets: refs, recentImageAssets: refs }));
    expect(container.querySelector('label[for="su-image-add-reference"]')?.textContent).toBe(
      "References (up to 16, in order)",
    );
    // The seventeenth restored reference is dropped: the list is the model's cap.
    expect(container.querySelectorAll(".ordered-reference")).toHaveLength(16);
    expect(container.querySelector("#su-image-steps")).toBeNull();
    expect(container.querySelector("#su-image-guidance")).toBeNull();
    expect(container.querySelector("#su-image-negative")).toBeNull();
    await click(container.querySelector('button[aria-label="Insert <Picture 3> into the prompt"]'));
    expect(container.querySelector("#su-image-prompt").value).toBe("Put the hat from <Picture 3>");
    await click(container.querySelector(".su-generate"));
    const request = context.createImageJob.mock.calls[0][0];
    // One ordered list, the armed reference first — the same payload the full studio sends.
    expect(request).toMatchObject({ model: "minimax_h3_image", mode: "edit_image", sourceAssetId: null });
    expect(request.referenceAssetIds).toEqual(refs.slice(0, 16).map((ref) => ref.id));
    // Simple sends no sampler block for the edit, as for text-to-image: the worker runs (and
    // records) the LCM 1.1 / 0.9 default.
    expect(request.advanced?.h3ImageSampler).toBeUndefined();
    expect(request.advanced?.steps).toBeUndefined();
  });
});
