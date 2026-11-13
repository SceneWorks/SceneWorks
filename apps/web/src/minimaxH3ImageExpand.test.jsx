import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, unmountRoot } from "./testUtils/dom.js";
import { H3_IMAGE_EXPAND_COPY, h3ImageProposedResolution } from "./minimaxH3ImageExpand.js";
import { writeSimpleStudioSnapshot } from "./simpleStudioStore.js";

// SW-7102 (feature SW-F363): MiniMax-H3-Image's prompt expander in both Image Studio views, from
// the SHIPPED catalog entries. The Expand prompt tool appears only for MiniMax-H3-Image with the
// expander installed; the expansion replaces the prompt only on Apply; the proposed size is a
// separate offer mapped onto the image model's own presets.

vi.mock("./api.js", async (importOriginal) => {
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

import { AppContext } from "./context/AppContext.js";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { SimpleShell } from "./simple/SimpleShell.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");
const manifestModels = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8")).models;
const withState = (id, installState) => ({
  ...manifestModels.find((model) => model.id === id),
  installState,
  usable: installState !== "missing",
});
const H3_IMAGE = withState("minimax_h3_image", "installed");
const PE = withState("minimax_h3_image_pe", "installed");
const QWEN_TURBO = withState("qwen_image_2_1_turbo", "installed");

const EXPANSION = {
  refinedPrompt: "A lighthouse on a basalt headland in thick sea fog, its beam a pale cone.",
  rewriteSuggestion: { whRatio: "16:9", workflow: "t2i", rewriterModelId: "minimax_h3_image_pe" },
};

function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((button) => button.textContent.trim().includes(text));

function context(models, overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    recentImageAssets: [],
    recentVideoAssets: [],
    jobs: [],
    imageModels: models.filter((model) => model.type === "image"),
    videoModels: [],
    audioModels: [],
    models,
    loras: [],
    imageLocalJobs: [],
    videoLocalJobs: [],
    audioLocalJobs: [],
    visibleWorkers: [],
    macCapabilities: null,
    gpuOptions: [],
    presets: [],
    latestImageAssets: [],
    studioLaunch: null,
    theme: "light",
    changeTheme: () => {},
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createVideoJob: vi.fn(async () => null),
    createAudioJob: vi.fn(async () => null),
    createModelDownloadJob: vi.fn(async () => null),
    createLoraDownloadJob: vi.fn(async () => null),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    jobAction: vi.fn(async () => {}),
    rememberLocalGenerationJob: vi.fn(),
    refinePrompt: vi.fn(),
    qwenRewritePrompt: vi.fn(),
    h3ImageExpandPrompt: vi.fn(async () => EXPANSION),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    importAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    setPreviewAsset: vi.fn(),
    requestedGpu: "",
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    ...overrides,
  };
}

describe("h3ImageProposedResolution", () => {
  it("offers the preset whose ratio matches, closest in area to the current size", () => {
    expect(h3ImageProposedResolution("16:9", H3_IMAGE, "2048x2048")).toBe("5472x3072");
    expect(h3ImageProposedResolution("21:9", H3_IMAGE, "2048x2048")).toBe("6272x2688");
    expect(h3ImageProposedResolution("3:4", H3_IMAGE, "2048x2048")).toBe("3552x4736");
    // Two 1:1 presets: the one nearer the current area wins.
    expect(h3ImageProposedResolution("1:1", H3_IMAGE, "2048x2048")).toBe("2048x2048");
    expect(h3ImageProposedResolution("1:1", H3_IMAGE, "4096x4096")).toBe("4096x4096");
  });

  it("falls back to the ratio at the current area inside the model's edge limits", () => {
    expect(h3ImageProposedResolution("5:4", H3_IMAGE, "2048x2048")).toBe("2288x1824");
    const [width, height] = h3ImageProposedResolution("40:1", H3_IMAGE, "4096x4096")
      .split("x")
      .map(Number);
    expect(width).toBeLessThanOrEqual(H3_IMAGE.limits.maxDimension);
    expect(height % 16).toBe(0);
  });

  it("offers nothing for an unreadable ratio", () => {
    for (const ratio of ["", "adaptive", "0:1", "wide", undefined]) {
      expect(h3ImageProposedResolution(ratio, H3_IMAGE, "2048x2048")).toBe("");
    }
  });
});

describe("Expand prompt in the advanced Image Studio (SW-7102)", () => {
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

  async function render(value) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={value}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("expands, reviews and applies only on accept, with the size a separate offer", async () => {
    const value = context([H3_IMAGE, PE]);
    await render(value);
    const prompt = container.querySelector('textarea[aria-label="Prompt"]');
    await act(async () => setTextarea(prompt, "a lighthouse in fog"));
    await click(buttonWithText(container, "Expand prompt"));
    await click(buttonWithText(container.querySelector(".qwen-rewrite-control"), "Expand prompt"));
    await act(async () => {});
    expect(value.h3ImageExpandPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: "a lighthouse in fog",
        modelId: "minimax_h3_image",
        projectId: "project_1",
        sourceAssetIds: [],
      }),
    );
    // Reviewed, not applied: the user's prompt is untouched until Apply.
    expect(prompt.value).toBe("a lighthouse in fog");
    expect(container.querySelector(".qwen-rewrite-draft").value).toBe(EXPANSION.refinedPrompt);
    expect(container.textContent).toContain("Proposed size: 16:9 (5472x3072)");
    await click(buttonWithText(container, "Apply"));
    expect(container.querySelector('textarea[aria-label="Prompt"]').value).toBe(EXPANSION.refinedPrompt);
    // Applying the prompt did not change the size; the size is its own accept.
    await click(document.body.querySelector(".advanced-section-toggle"));
    const aspect = [...container.querySelectorAll("label")]
      .find((node) => node.querySelector("select") && node.firstChild?.textContent?.trim() === "Aspect")
      ?.querySelector("select");
    expect(aspect.value).toBe("2048x2048");
  });

  // Width/Height overrides outrank the Aspect preset, so accepting a proposed size has to clear
  // them (preset) or take them over (non-preset), or the render keeps the old override size.
  async function expandThenUseSize(ratio, saved) {
    window.localStorage.setItem(
      "sceneworks-studio-image-project_1",
      JSON.stringify({ model: "minimax_h3_image", ...saved }),
    );
    const value = context([H3_IMAGE, PE], {
      h3ImageExpandPrompt: vi.fn(async () => ({
        ...EXPANSION,
        rewriteSuggestion: { ...EXPANSION.rewriteSuggestion, whRatio: ratio },
      })),
    });
    await render(value);
    await act(async () =>
      setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a lighthouse in fog"),
    );
    await click(buttonWithText(container, "Expand prompt"));
    await click(buttonWithText(container.querySelector(".qwen-rewrite-control"), "Expand prompt"));
    await act(async () => {});
    await click(buttonWithText(container, "Use this size"));
    await click(buttonWithText(document.body, "Generate"));
    return value.createImageJob.mock.calls[0][0];
  }

  it("submits an accepted preset size even with Width/Height overrides set", async () => {
    const job = await expandThenUseSize("16:9", { widthOverride: "2048", heightOverride: "2048" });
    expect(job).toMatchObject({ width: 5472, height: 3072 });
  });

  it("writes an accepted non-preset size into the override fields", async () => {
    const job = await expandThenUseSize("5:4", { widthOverride: "2048", heightOverride: "2048" });
    expect(job).toMatchObject({ width: 2288, height: 1824 });
  });

  it("does not ground the expander in a reference retained from another model", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-image-project_1",
      JSON.stringify({ model: "minimax_h3_image", referenceAssetId: "retained-asset" }),
    );
    const value = context([H3_IMAGE, PE]);
    await render(value);
    await act(async () =>
      setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a lighthouse in fog"),
    );
    await click(buttonWithText(container, "Expand prompt"));
    await click(buttonWithText(container.querySelector(".qwen-rewrite-control"), "Expand prompt"));
    await act(async () => {});
    expect(value.h3ImageExpandPrompt).toHaveBeenCalledWith(
      expect.objectContaining({ sourceAssetIds: [] }),
    );
  });

  // SW-7101: in the edit, the expander is grounded in exactly the ordered references the render
  // conditions on, in their <Picture N> order.
  it("grounds an edit's expansion in the ordered references the render sends", async () => {
    const refs = ["ref-a", "ref-b", "ref-c"];
    window.localStorage.setItem(
      "sceneworks-studio-image-project_1",
      JSON.stringify({ mode: "edit_image", model: "minimax_h3_image", referenceAssetIds: refs }),
    );
    const assets = refs.map((id) => ({ id, projectId: "project_1", type: "image", displayName: id, status: {} }));
    const value = context([H3_IMAGE, PE], { assets });
    await render(value);
    await act(async () =>
      setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "<Picture 1> in the coat from <Picture 3>"),
    );
    await click(buttonWithText(container, "Expand prompt"));
    await click(buttonWithText(container.querySelector(".qwen-rewrite-control"), "Expand edit instruction"));
    await act(async () => {});
    expect(value.h3ImageExpandPrompt).toHaveBeenCalledWith(
      expect.objectContaining({ modelId: "minimax_h3_image", sourceAssetIds: refs }),
    );
  });

  it("keeps the original prompt when the user declines the expansion", async () => {
    const value = context([H3_IMAGE, PE]);
    await render(value);
    const prompt = container.querySelector('textarea[aria-label="Prompt"]');
    await act(async () => setTextarea(prompt, "a lighthouse in fog"));
    await click(buttonWithText(container, "Expand prompt"));
    await click(buttonWithText(container.querySelector(".qwen-rewrite-control"), "Expand prompt"));
    await act(async () => {});
    await click(buttonWithText(container, "Keep original"));
    expect(container.querySelector('textarea[aria-label="Prompt"]').value).toBe("a lighthouse in fog");
  });

  it("offers no Expand prompt tool without the expander or for another model", async () => {
    await render(context([H3_IMAGE, withState("minimax_h3_image_pe", "missing")]));
    expect(buttonWithText(container, "Expand prompt")).toBeUndefined();
    await render(context([QWEN_TURBO, PE]));
    expect(buttonWithText(container, "Expand prompt")).toBeUndefined();
  });
});

describe("Expand prompt in the Simple Image Studio (SW-7102)", () => {
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

  async function openImage(value) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={value}>
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
  }

  it("expands, applies on accept, and offers the proposed size separately", async () => {
    const value = context([H3_IMAGE, PE]);
    await openImage(value);
    const prompt = container.querySelector("#su-image-prompt");
    await act(async () => setTextarea(prompt, "a lighthouse in fog"));
    await click(buttonWithText(container, "Expand prompt"));
    await click(buttonWithText(container.querySelector(".qwen-rewrite-control"), "Expand prompt"));
    await act(async () => {});
    expect(value.h3ImageExpandPrompt).toHaveBeenCalledWith(
      expect.objectContaining({ prompt: "a lighthouse in fog", modelId: "minimax_h3_image" }),
    );
    expect(container.querySelector("#su-image-prompt").value).toBe("a lighthouse in fog");
    expect(container.textContent).toContain("Proposed size: 16:9 (5472x3072)");
    await click(buttonWithText(container, "Use this size"));
    await click(buttonWithText(container, "Apply"));
    expect(container.querySelector("#su-image-prompt").value).toBe(EXPANSION.refinedPrompt);
    await click(container.querySelector(".su-generate"));
    expect(value.createImageJob.mock.calls[0][0]).toMatchObject({
      model: "minimax_h3_image",
      prompt: EXPANSION.refinedPrompt,
      width: 5472,
      height: 3072,
    });
  });

  // SW-7101: Simple's edit grounds the expander in its ordered list — the armed tile first.
  it("grounds an edit's expansion in the ordered references the render sends", async () => {
    const refs = ["ref-a", "ref-b"].map((id) => ({ id, projectId: "project_1", type: "image", name: id, status: {} }));
    writeSimpleStudioSnapshot("project_1", {
      image: { mode: "edit_image", model: "minimax_h3_image", referenceAssetId: "ref-a", extraReferenceIds: ["ref-b"] },
    });
    const value = context([H3_IMAGE, PE], { assets: refs, recentImageAssets: refs });
    await openImage(value);
    await act(async () => setTextarea(container.querySelector("#su-image-prompt"), "<Picture 2> on <Picture 1>"));
    await click(buttonWithText(container, "Expand prompt"));
    await click(buttonWithText(container.querySelector(".qwen-rewrite-control"), "Expand edit instruction"));
    await act(async () => {});
    expect(value.h3ImageExpandPrompt).toHaveBeenCalledWith(
      expect.objectContaining({ modelId: "minimax_h3_image", sourceAssetIds: ["ref-a", "ref-b"] }),
    );
  });

  it("offers no Expand prompt tile without the expander", async () => {
    await openImage(context([H3_IMAGE, withState("minimax_h3_image_pe", "missing")]));
    expect(buttonWithText(container, "Expand prompt")).toBeUndefined();
  });
});

describe("the Expand prompt wording", () => {
  it("says the prompt is not replaced on its own and names the references as <Picture N>", () => {
    expect(H3_IMAGE_EXPAND_COPY.hint).toContain("not changed on its own");
    expect(H3_IMAGE_EXPAND_COPY.editHint(2)).toContain("<Picture 1>…<Picture 2>");
    expect(H3_IMAGE_EXPAND_COPY.editHint(1)).toContain("1 reference in order");
  });
});
