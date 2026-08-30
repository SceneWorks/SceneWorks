import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import { flushSync } from "react-dom";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async () => ({})),
  };
});

import { AppContext } from "../context/AppContext.js";
import { VideoStudio } from "./VideoStudio.jsx";

const LTX = {
  id: "ltx_2_3",
  name: "LTX 2.3",
  type: "video",
  family: "ltx-video",
  capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
  defaults: { duration: 6, resolution: "768x512", fps: 25 },
  limits: {},
  quantization: {},
  loraCompatibility: {},
  ui: {},
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createPersonDetectionJob: vi.fn(),
    createPersonTrackJob: vi.fn(),
    createVideoJob: vi.fn(),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    importAsset: vi.fn(),
    latestVideoAssets: [],
    recentVideoAssets: [],
    studioLaunch: null,
    loras: [],
    jobs: [],
    videoLocalJobs: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setPreviewAsset: vi.fn(),
    personTracks: [],
    personReadiness: {},
    presets: [],
    requestedGpu: "",
    saveTrackCorrections: vi.fn(),
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    videoModels: [LTX],
    ...overrides,
  };
}

async function doubleClick(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  });
}

const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === text);

// Save-as-preset is an inline pair in the style-preset row (sc-15372): the dropdown opens a
// scope menu, and picking a scope both sets it and fires the save.
const saveButton = (container) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Save as preset"));
const nameInput = (container) => container.querySelector('input[aria-label="Preset name"]');
const saveWithScope = async (container, scopeLabel) => {
  await click(saveButton(container));
  await click(
    [...container.querySelectorAll(".save-preset-scope-picker .compact-selector-item")].find((b) =>
      b.textContent.includes(scopeLabel),
    ),
  );
};
// The Advanced disclosure is collapsed by default — open it before touching a knob that lives
// inside it.
const openAdvanced = async (container) => {
  const toggle = container.querySelector(".advanced-section-toggle");
  if (toggle) {
    await click(toggle);
  }
};

// sc-15398 — `quality` (fast/balanced/best) is read by exactly ONE adapter: `svd_steps` maps it to
// 15/25/30 inference steps for `svd_video`. Every other video engine ignores the key, so the
// control is gated on the adapter that honors it. This pair is what makes the gate meaningful:
// asserting only the hidden case would pass just as well if the segment were deleted outright.
describe("VideoStudio Quality segment adapter gate (sc-15398)", () => {
  let container;
  let root;

  const SVD = {
    id: "svd",
    name: "Stable Video Diffusion",
    type: "video",
    family: "svd",
    adapter: "svd_video",
    capabilities: ["image_to_video"],
    promptless: true,
    defaults: { duration: 4, fps: 7, resolution: "1024x576", quality: "balanced" },
    limits: {},
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("shows Quality in Advanced for the SVD adapter, which honors it", async () => {
    await render(baseContext({ videoModels: [SVD] }));
    await openAdvanced(container);

    const segment = container.querySelector(".quality-segment");
    expect(segment).toBeTruthy();
    expect([...segment.querySelectorAll("button")].map((b) => b.textContent.trim())).toEqual([
      "Draft",
      "Balanced",
      "Final",
    ]);
  });

  it("hides Quality for an adapter that never reads the key", async () => {
    await render(baseContext({ videoModels: [LTX] }));
    await openAdvanced(container);

    expect(container.querySelector(".quality-segment")).toBeNull();
    // The Steps override — which beats `quality` even on SVD — is still there, so the run is not
    // left without a way to trade speed for fidelity.
    expect(
      [...container.querySelectorAll("label")].some((label) => label.textContent.trim().startsWith("Steps")),
    ).toBe(true);
  });
});

describe("VideoStudio Save as Preset", () => {
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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // The Advanced disclosure belongs INSIDE the work-panel, not floating beneath it:
  // the Purpose zone is one elevated card. It shipped detached in sc-10477 and was
  // pulled back in by sc-10490 — assert the nesting so it cannot drift again.
  it("nests the Advanced disclosure inside the work-panel", async () => {
    await render(baseContext());

    expect(document.body.querySelector(".work-panel .advanced-section")).toBeTruthy();
    expect(document.body.querySelector(".studio-shell > .advanced-section")).toBeNull();
  });

  it("snapshots the video config into a text_to_video preset without the seed", async () => {
    const context = baseContext();
    await render(context);

    const input = nameInput(container);
    expect(input).toBeTruthy();
    await act(async () => setInput(input, "Push In"));
    await saveWithScope(container, "This project");

    expect(context.createPreset).toHaveBeenCalledTimes(1);
    const payload = context.createPreset.mock.calls[0][0];
    // Video Studio now opens on Text→Video (sc-5716), matching Image Studio's Text→Image default.
    expect(payload).toMatchObject({
      id: "push_in",
      name: "Push In",
      scope: "project",
      workflow: "text_to_video",
      model: "ltx_2_3",
    });
    expect(payload.defaults.prompt).toBe("Camera slowly pushes in while the scene comes alive");
    expect(payload.defaults).not.toHaveProperty("seed");
    expect(container.textContent).toContain('Saved "Push In" to this project.');
  });

  it("blocks a duplicate name client-side before calling the API", async () => {
    const context = baseContext({
      presets: [
        {
          id: "push_in",
          name: "Push In",
          scope: "project",
          workflow: "image_to_video",
          model: "ltx_2_3",
          modes: ["image_to_video"],
        },
      ],
    });
    await render(context);

    await act(async () => setInput(nameInput(container), "Push In"));
    await saveWithScope(container, "This project");

    expect(context.createPreset).not.toHaveBeenCalled();
    expect(container.textContent).toContain("already exists");
  });

  it("shows a string preset-save rejection", async () => {
    const context = baseContext({ createPreset: vi.fn(() => Promise.reject("Preset service is unavailable.")) });
    await render(context);

    await act(async () => setInput(nameInput(container), "Unavailable preset"));
    await saveWithScope(container, "This project");

    expect(container.textContent).toContain("Preset service is unavailable.");
  });

  it("normalizes an undefined preset-save rejection into a safe error message", async () => {
    const context = baseContext({ createPreset: vi.fn(() => Promise.reject(undefined)) });
    await render(context);

    await act(async () => setInput(nameInput(container), "Unavailable preset"));
    await saveWithScope(container, "This project");

    expect(container.textContent).toContain("Could not save this preset.");
  });
});

describe("VideoStudio video_bridge", () => {
  let container;
  let root;

  // A bridge-capable video model with a non-LTX id so the IC-LoRA preset gate
  // (requiresLtxIcLora, which keys on ltx_2_3) doesn't block submission here —
  // this test exercises the new input wiring, not the LTX IC-LoRA requirement.
  const BRIDGE_MODEL = {
    id: "bridge_model",
    name: "Bridge Model",
    type: "video",
    family: "ltx-video",
    capabilities: ["image_to_video", "text_to_video", "extend_clip", "video_bridge"],
    defaults: { duration: 6, resolution: "768x512", fps: 25 },
    limits: {},
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  const leftClip = { id: "vid_left", type: "video", projectId: "project_1", displayName: "Left Clip" };
  const rightClip = { id: "vid_right", type: "video", projectId: "project_1", displayName: "Right Clip" };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("submits both clip ids when bridging two clips", async () => {
    const context = baseContext({
      videoModels: [BRIDGE_MODEL],
      assets: [leftClip, rightClip],
      selectedAsset: leftClip,
    });
    await render(context);

    // Switch to Bridge mode; the left clip is pre-filled from the selected asset.
    const modeControl = container.querySelector(".mode-control");
    await click(buttonWithText(modeControl, "Bridge"));

    // Drive the right-clip picker (the only "Select clip" button; the left
    // picker shows "Change" because it already has a value).
    await click(buttonWithText(container, "Select clip"));
    const modal = document.querySelector(".asset-picker-modal");
    expect(modal).toBeTruthy();
    const rightOption = [...modal.querySelectorAll('[role="option"]')].find((el) =>
      el.textContent.includes("Right Clip"),
    );
    await doubleClick(rightOption);

    // Render.
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "video_bridge",
      sourceClipAssetId: "vid_left",
      bridgeRightClipAssetId: "vid_right",
    });
  });
});

describe("VideoStudio fit mode (sc-6139)", () => {
  let container;
  let root;

  const source = { id: "img_src", type: "image", projectId: "project_1", displayName: "Source" };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const modeButton = (label) => buttonWithText(container.querySelector(".mode-control"), label);
  const fitField = () => container.querySelector(".fit-mode-field");
  const fitButton = (label) => fitField() && buttonWithText(fitField(), label);

  it("offers Crop/Pad only (no Outpaint) in the image-conditioned modes", async () => {
    const context = baseContext({ assets: [source], selectedAsset: source });
    await render(context);

    // Default Text → Video has no starting image, so no fit control.
    expect(fitField()).toBeFalsy();

    await click(modeButton("Image → Video"));
    expect([...fitField().querySelectorAll("button")].map((b) => b.textContent.trim())).toEqual([
      "Crop",
      "Pad",
    ]);

    await click(modeButton("First → Last"));
    expect(fitField()).toBeTruthy();
    expect(fitButton("Outpaint")).toBeFalsy();
  });

  it("threads the chosen fitMode into the image_to_video payload (default crop)", async () => {
    const context = baseContext({ assets: [source], selectedAsset: source });
    await render(context);
    await click(modeButton("Image → Video"));

    // Default selection is crop.
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].fitMode).toBe("crop");

    // Choosing Pad threads through on the next submit.
    await click(fitButton("Pad"));
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[1][0].fitMode).toBe("pad");
  });

  it("keeps a secondary-field upload from overwriting the primary source selection", async () => {
    let importOptions;
    function Host() {
      const [assets, setAssets] = React.useState([source]);
      const [selectedAssetId, setSelectedAssetId] = React.useState(source.id);
      const selectedAsset = assets.find((asset) => asset.id === selectedAssetId) ?? null;
      const importAsset = async (_file, options) => {
        importOptions = options;
        const imported = {
          id: "img_last",
          type: "image",
          projectId: "project_1",
          displayName: "Uploaded Last Frame",
        };
        setAssets((items) => [imported, ...items]);
        if (options.select !== false) {
          setSelectedAssetId(imported.id);
        }
        return imported;
      };
      return (
        <AppContext.Provider
          value={baseContext({ assets, importAsset, selectedAsset, selectedAssetId })}
        >
          <VideoStudio />
        </AppContext.Provider>
      );
    }

    await act(async () => root.render(<Host />));
    await act(async () => {});
    await click(modeButton("First → Last"));
    const fieldByLabel = (label) =>
      [...container.querySelectorAll(".asset-picker-field")].find(
        (field) => field.querySelector(".asset-picker-label")?.textContent === label,
      );
    await click(buttonWithText(fieldByLabel("Last frame"), "Select image"));
    const modal = document.querySelector(".media-source-modal");
    await click([...modal.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent.includes("File Upload")));
    const input = modal.querySelector('input[type="file"]');
    const file = new File(["last"], "last.png", { type: "image/png" });
    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    expect(importOptions).toEqual({ select: false, throwOnError: true });
    expect(fieldByLabel("First frame").textContent).toContain("Source");
    expect(fieldByLabel("First frame").textContent).not.toContain("Uploaded Last Frame");
    expect(fieldByLabel("Last frame").textContent).toContain("Uploaded Last Frame");
  });

  it("omits fitMode for non-image-conditioned modes", async () => {
    const context = baseContext({ assets: [source], selectedAsset: source });
    await render(context);
    // Text → Video is the default mode; it carries no starting image to fit.
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].fitMode).toBeUndefined();
  });
});

describe("VideoStudio Bernini task modes", () => {
  let container;
  let root;

  // Bernini exposes the full planner video surface (sc-4703). No `macSupport` here so
  // the (gating-off) test env leaves the mode buttons enabled — `capabilities` gates submit.
  const BERNINI = {
    id: "bernini",
    name: "Bernini",
    type: "video",
    family: "bernini",
    capabilities: [
      "text_to_video",
      "video_to_video",
      "reference_to_video",
      "reference_video_to_video",
      "multi_video_to_video",
      "ads2v",
    ],
    defaults: { duration: 5, resolution: "848x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["848x480", "480x848"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  const clip = { id: "vid_src", type: "video", projectId: "project_1", displayName: "Source Clip" };
  const clip2 = { id: "vid_src_2", type: "video", projectId: "project_1", displayName: "Source Clip Two" };
  const refClip = { id: "vid_ref", type: "video", projectId: "project_1", displayName: "Reference Clip" };
  const refA = { id: "img_ref_a", type: "image", projectId: "project_1", displayName: "Reference A" };
  const refB = { id: "img_ref_b", type: "image", projectId: "project_1", displayName: "Reference B" };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // Scope to the mode's media-input band so the unrelated upscaler card (which also
  // has a "Source clip" picker in the results rail) doesn't leak into the assertions.
  const pickerLabels = () =>
    [...(container.querySelector(".studio-source-band")?.querySelectorAll(".asset-picker-label") ?? [])].map(
      (el) => el.textContent,
    );
  const modeButton = (label) => buttonWithText(container.querySelector(".mode-control"), label);

  it("shows the right media slots for each mode", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, refA, refB] });
    await render(context);

    await click(modeButton("Video → Video"));
    expect(pickerLabels()).toContain("Source clip");
    expect(pickerLabels()).not.toContain("Reference images");

    await click(modeButton("Reference → Video"));
    expect(pickerLabels()).toContain("Reference images");
    expect(pickerLabels()).not.toContain("Source clip");

    await click(modeButton("Reference + Video"));
    expect(pickerLabels()).toEqual(expect.arrayContaining(["Source clip", "Reference images"]));

    // mv2v: a single multi-clip picker, no single "Source clip" or reference images.
    await click(modeButton("Multi-Clip → Video"));
    expect(pickerLabels()).toContain("Source clips");
    expect(pickerLabels()).not.toContain("Source clip");
    expect(pickerLabels()).not.toContain("Reference images");

    // ads2v: source clip + reference video + reference images.
    await click(modeButton("Clip + Ref Video"));
    expect(pickerLabels()).toEqual(
      expect.arrayContaining(["Source clip", "Reference video", "Reference images"]),
    );
  });

  it("uses the Image Studio source flow for both clip and image pickers", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, refA] });
    await render(context);

    await click(modeButton("Video → Video"));
    await click(buttonWithText(container, "Select clip"));
    let modal = document.querySelector(".media-source-modal");
    expect([...modal.querySelectorAll('[role="tab"]')].map((el) => el.textContent.trim())).toEqual([
      "Assets1",
      "File Upload",
      "Character0",
    ]);
    await click([...modal.querySelectorAll('[role="tab"]')].find((el) => el.textContent.includes("File Upload")));
    expect(modal.querySelector('input[type="file"]').accept).toBe("video/*");
    await click(buttonWithText(modal, "Cancel"));

    await click(modeButton("Reference → Video"));
    await click(buttonWithText(container, "Select images"));
    modal = document.querySelector(".media-source-modal");
    await click([...modal.querySelectorAll('[role="tab"]')].find((el) => el.textContent.includes("File Upload")));
    const imageInput = modal.querySelector('input[type="file"]');
    expect(imageInput.accept).toBe("image/*");
    expect(imageInput.multiple).toBe(true);
  });

  it("keeps Render disabled until the required reference image is selected", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, refA, refB] });
    await render(context);
    await click(modeButton("Reference → Video"));

    // No reference selected yet.
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);

    await click(buttonWithText(container, "Select images"));
    const modal = document.querySelector(".asset-picker-modal");
    const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Reference A"));
    await click(option);
    await click(buttonWithText(modal, "Use Selection"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
  });

  it("submits the source clip for video_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip], selectedAsset: clip });
    await render(context);
    await click(modeButton("Video → Video"));
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({ mode: "video_to_video", sourceClipAssetId: "vid_src" });
    expect(payload.referenceAssetIds).toEqual([]);
  });

  // sc-17160: the audio-reference list is on every video job body, present-but-empty when the
  // user picked none, so a replay reader never has to tell "absent" from "empty" and the picker
  // (sc-17161) has a field to fill rather than a payload shape to invent.
  it("carries the audio-reference list on the submitted payload", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip], selectedAsset: clip });
    await render(context);
    await click(modeButton("Video → Video"));
    await click(buttonWithText(container, "Render clip"));

    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.referenceAudioAssetIds).toEqual([]);
  });

  it("submits all chosen reference images for reference_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [refA, refB] });
    await render(context);
    await click(modeButton("Reference → Video"));

    await click(buttonWithText(container, "Select images"));
    const modal = document.querySelector(".asset-picker-modal");
    for (const name of ["Reference A", "Reference B"]) {
      const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes(name));
      await click(option);
    }
    await click(buttonWithText(modal, "Use Selection"));
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.mode).toBe("reference_to_video");
    expect(payload.referenceAssetIds).toEqual(["img_ref_a", "img_ref_b"]);
    expect(payload.sourceClipAssetId).toBeNull();
  });

  it("submits both clip and references for reference_video_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, refA], selectedAsset: clip });
    await render(context);
    await click(modeButton("Reference + Video"));

    await click(buttonWithText(container, "Select images"));
    const modal = document.querySelector(".asset-picker-modal");
    const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Reference A"));
    await click(option);
    await click(buttonWithText(modal, "Use Selection"));
    await click(buttonWithText(container, "Render clip"));

    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "reference_video_to_video",
      sourceClipAssetId: "vid_src",
    });
    expect(payload.referenceAssetIds).toEqual(["img_ref_a"]);
  });

  it("requires at least two clips before submitting multi_video_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, clip2], selectedAsset: clip });
    await render(context);
    await click(modeButton("Multi-Clip → Video"));

    // One clip auto-selected from selectedAsset isn't enough — mv2v needs >=2.
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);

    await click(buttonWithText(container, "Select clips"));
    const modal = document.querySelector(".asset-picker-modal");
    for (const name of ["Source Clip", "Source Clip Two"]) {
      const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes(name));
      await click(option);
    }
    await click(buttonWithText(modal, "Use Selection"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.mode).toBe("multi_video_to_video");
    expect(payload.sourceClipAssetIds).toEqual(["vid_src", "vid_src_2"]);
    expect(payload.sourceClipAssetId).toBeNull();
    expect(payload.referenceAssetIds).toEqual([]);
  });

  it("submits source clip, reference video, and references for ads2v", async () => {
    const context = baseContext({
      videoModels: [BERNINI],
      assets: [clip, refClip, refA],
      selectedAsset: clip,
    });
    await render(context);
    await click(modeButton("Clip + Ref Video"));

    // Source clip auto-selected; reference video + a reference image still required.
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);

    // The source clip picker shows "Change" (auto-selected), so the first "Select clip"
    // button in document order is the empty reference-video picker.
    await click(buttonWithText(container, "Select clip"));
    let modal = document.querySelector(".asset-picker-modal");
    const refClipOption = [...modal.querySelectorAll('[role="option"]')].find((el) =>
      el.textContent.includes("Reference Clip"),
    );
    await click(refClipOption);
    await click(buttonWithText(modal, "Use Selection"));

    // Pick a reference image.
    await click(buttonWithText(container, "Select images"));
    modal = document.querySelector(".asset-picker-modal");
    const refImageOption = [...modal.querySelectorAll('[role="option"]')].find((el) =>
      el.textContent.includes("Reference A"),
    );
    await click(refImageOption);
    await click(buttonWithText(modal, "Use Selection"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    await click(buttonWithText(container, "Render clip"));

    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "ads2v",
      sourceClipAssetId: "vid_src",
      referenceClipAssetId: "vid_ref",
    });
    expect(payload.referenceAssetIds).toEqual(["img_ref_a"]);
    expect(payload.sourceClipAssetIds).toEqual([]);
  });

  it("disables an editing mode on a model that does not support it", async () => {
    const context = baseContext({ videoModels: [LTX], assets: [clip] });
    await render(context);
    await click(modeButton("Reference → Video"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(true);
    expect(container.textContent).toContain("does not support this mode");
  });
});

describe("VideoStudio SCAIL-2 character animation + replacement backend", () => {
  let container;
  let root;

  const SCAIL2 = {
    id: "scail2_14b",
    name: "SCAIL-2",
    type: "video",
    family: "scail2",
    capabilities: ["animate_character", "replace_person"],
    defaults: { duration: 5, resolution: "832x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480", "480x832"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
    hasVariantMatrix: true,
    variants: [
      { variant: "q4", installState: "installed" },
      { variant: "q8", installState: "installed" },
      { variant: "bf16", installState: "installed" },
    ],
  };
  const SCAIL2_MULTI_REFERENCE = {
    ...SCAIL2,
    ui: { scail2MultiReference: true },
  };
  // A Wan-VACE-style replace-capable model — the default replacement backend SCAIL-2 augments.
  const WAN = {
    id: "wan_2_2",
    name: "Wan 2.2",
    type: "video",
    family: "wan-video",
    capabilities: ["image_to_video", "text_to_video", "replace_person"],
    defaults: { duration: 5, resolution: "832x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480", "480x832"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  const clip = { id: "vid_src", type: "video", projectId: "project_1", displayName: "Driving Clip" };
  const character = { id: "img_ref_a", type: "image", projectId: "project_1", displayName: "Character A" };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const pickerLabels = () =>
    [...(container.querySelector(".studio-source-band")?.querySelectorAll(".asset-picker-label") ?? [])].map(
      (el) => el.textContent,
    );
  const modeButton = (label) => buttonWithText(container.querySelector(".mode-control"), label);

  it("shows the driving video + reference character slots for animate_character", async () => {
    const context = baseContext({ videoModels: [SCAIL2], assets: [clip, character] });
    await render(context);
    await click(modeButton("Animate character"));

    expect(pickerLabels()).toEqual(expect.arrayContaining(["Driving video", "Reference character"]));
  });

  it("submits the driving clip + reference character for animate_character", async () => {
    const context = baseContext({ videoModels: [SCAIL2], assets: [clip, character] });
    await render(context);
    await click(modeButton("Animate character"));

    // Both inputs required → Render stays disabled until selected.
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);

    await click(buttonWithText(container, "Select clip"));
    let modal = document.querySelector(".asset-picker-modal");
    await doubleClick(
      [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Driving Clip")),
    );

    await click(buttonWithText(container, "Select image"));
    modal = document.querySelector(".asset-picker-modal");
    await doubleClick(
      [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Character A")),
    );

    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "animate_character",
      model: "scail2_14b",
      sourceClipAssetId: "vid_src",
    });
    expect(payload.referenceAssetIds).toEqual(["img_ref_a"]);
  });

  it("keeps SCAIL-2 multi-character selection descriptor-gated, ordered, and bounded", async () => {
    const characters = [
      character,
      { id: "img_ref_b", type: "image", projectId: "project_1", displayName: "Character B" },
      { id: "img_ref_c", type: "image", projectId: "project_1", displayName: "Character C" },
      { id: "img_ref_d", type: "image", projectId: "project_1", displayName: "Character D" },
      { id: "img_ref_e", type: "image", projectId: "project_1", displayName: "Character E" },
      { id: "img_ref_f", type: "image", projectId: "project_1", displayName: "Character F" },
      { id: "img_ref_g", type: "image", projectId: "project_1", displayName: "Character G" },
    ];
    const context = baseContext({ videoModels: [SCAIL2_MULTI_REFERENCE], assets: [clip, ...characters] });
    await render(context);
    await click(modeButton("Animate character"));

    expect(pickerLabels()).toContain("Reference characters (ordered, up to 6)");
    await click(buttonWithText(container, "Select clip"));
    let modal = document.querySelector(".asset-picker-modal");
    await doubleClick(
      [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Driving Clip")),
    );

    await click(buttonWithText(container, "Select images"));
    modal = document.querySelector(".asset-picker-modal");
    for (const asset of characters) {
      await click(
        [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes(asset.displayName)),
      );
    }
    await click(buttonWithText(modal, "Use Selection"));

    // All seven remain visible/selectable so the user can correct the choice; the studio must not
    // silently trim one before serialization, and the explicit validation blocks submission.
    expect(container.textContent).toContain("SCAIL-2 supports at most 6 reference characters");
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);
  });

  it("offers SCAIL-2 as a replacement engine when 2+ backends can replace", async () => {
    const context = baseContext({ videoModels: [WAN, SCAIL2], assets: [clip] });
    await render(context);
    await click(modeButton("Replace person"));

    const engineLabel = [...container.querySelectorAll("label")].find((el) =>
      el.textContent.includes("Replacement engine"),
    );
    expect(engineLabel).toBeTruthy();
    const engineSelect = engineLabel.querySelector("select");
    expect([...engineSelect.options].map((o) => o.value)).toEqual(
      expect.arrayContaining(["wan_2_2", "scail2_14b"]),
    );

    // Choosing SCAIL-2 switches the active model and surfaces the full-character note.
    const setSelect = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set;
    await act(async () => {
      setSelect.call(engineSelect, "scail2_14b");
      engineSelect.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    expect(container.textContent).toContain("SCAIL-2 full-character replacement");
  });

  it("does not offer a Replacement mode SCAIL-2 cannot honor, and never sends one (sc-20262)", async () => {
    // The knobs sweep (inference #669) found this select rendered and user-settable for SCAIL-2
    // while EVERY scail2 conditioning site in the worker emits `ReplacementMode::default()`
    // literally — so the choice never reached the engine. sc-20262 made the engine REFUSE a
    // non-default mode, which turns a silently-dropped knob into an instant refusal. Hiding it is
    // the honest surface. sc-20262 is also the story that would make it honored; if that lands,
    // this test and `replacementModeApplies` go together.
    //
    // The panel already carried a guidance strip saying the mode "does not apply" — PROSE IS NOT A
    // GATE, and that strip is why this looked covered. It is asserted below as copy, alongside the
    // control's absence, so neither can stand in for the other.
    const modeLabel = () =>
      [...container.querySelectorAll("label")].find((el) =>
        el.textContent.includes("Replacement mode"),
      );

    // Wan-VACE inpaints the masked region and DOES honor the mode: the control is unchanged there.
    // This leg is what makes the assertion per-model rather than a global removal.
    const context = baseContext({ videoModels: [WAN, SCAIL2], assets: [clip] });
    await render(context);
    await click(modeButton("Replace person"));
    expect(modeLabel(), "Wan-VACE honors the mode and must keep the control").toBeTruthy();
    expect([...modeLabel().querySelector("select").options].map((o) => o.value)).toEqual([
      "face_only",
      "full_person_keep_outfit",
      "full_person_replace_outfit",
    ]);

    // Switching the engine to SCAIL-2 withdraws it.
    const engineSelect = [...container.querySelectorAll("label")]
      .find((el) => el.textContent.includes("Replacement engine"))
      .querySelector("select");
    const setSelectValue = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set;
    await act(async () => {
      setSelectValue.call(engineSelect, "scail2_14b");
      engineSelect.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    expect(modeLabel(), "SCAIL-2 must not offer a mode it cannot honor").toBeFalsy();
    // The explanation survives the control — withdrawing it silently would be its own gap.
    expect(container.textContent).toContain("SCAIL-2 full-character replacement");
    expect(container.textContent).toContain("no Replacement mode to choose");

    // …and switching back restores it, so this is a gate rather than a one-way delete.
    await act(async () => {
      setSelectValue.call(engineSelect, "wan_2_2");
      engineSelect.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    expect(modeLabel(), "the control returns for an engine that honors it").toBeTruthy();
  });

  it("sends the default Replacement mode for SCAIL-2 even when one was set elsewhere (sc-20262)", async () => {
    // Hiding a control does not unset the state behind it. A user who picks "Full Person, Keep
    // Outfit" on Wan-VACE and then switches the engine to SCAIL-2 would otherwise send that value
    // into a job whose engine now REFUSES it — the same shape `referenceAudioAssetIds` was fixed
    // for (a selection made on Ref2VA riding into a model with no audio cap once the picker
    // unmounted). Restored from a snapshot because that is the cheapest way to reach a submittable
    // replace_person job, and it is also the realistic path: the value is persisted per project.
    window.localStorage.setItem(
      "sceneworks-studio-video-project_1",
      JSON.stringify({
        mode: "replace_person",
        model: "scail2_14b",
        sourceClipAssetId: "vid_src",
        personTrackId: "track_1",
        characterId: "char_1",
        replacementMode: "full_person_keep_outfit",
      }),
    );
    const context = baseContext({
      videoModels: [WAN, SCAIL2],
      assets: [clip, character],
      characters: [{ id: "char_1", name: "Ada", projectId: "project_1" }],
      personTracks: [
        {
          id: "track_1",
          name: "Selected person",
          projectId: "project_1",
          sourceAssetId: "vid_src",
          status: {},
          frames: [],
        },
      ],
    });
    await render(context);

    // Precondition: the snapshot really did restore a NON-default mode, so the assertion below is
    // about the gate rather than about a value that was never set.
    expect(context.createVideoJob).not.toHaveBeenCalled();
    // The submit CTA's LABEL is per-mode ("Replace person" here), so target the CTA by class
    // rather than by a label that changes with the mode under test.
    const submitCta = container.querySelector(".prompt-cta");
    expect(submitCta.disabled, "the restored snapshot must be submittable").toBe(false);
    await click(submitCta);

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.mode).toBe("replace_person");
    expect(payload.model).toBe("scail2_14b");
    expect(payload.replacementMode).toBe("face_only");
    // The recipe's human-readable echo follows the same gate, so a replayed SCAIL-2 recipe cannot
    // display a mode the job never carried.
    expect(payload.advanced.replacementModeLabel).toBe("Face Only");
  });

  it("hides the replacement engine picker when only one backend can replace", async () => {
    const context = baseContext({ videoModels: [WAN], assets: [clip] });
    await render(context);
    await click(modeButton("Replace person"));

    const engineLabel = [...container.querySelectorAll("label")].find((el) =>
      el.textContent.includes("Replacement engine"),
    );
    expect(engineLabel).toBeFalsy();
  });
});

// sc-5716: under active Mac gating, mode tabs must be gated on mode availability (does any model
// serve it), not on the selected model — otherwise switching to a model-specific mode (Replace
// person / Animate character) snapped to a model whose other modes were disabled and trapped the
// user. These render the studio in MLX-required mode with per-model `macSupport.features.videoModes`.
describe("VideoStudio Mac mode gating (sc-5716)", () => {
  let container;
  let root;

  // Every UI mode, so each Mac model can declare an explicit per-mode boolean (mirrors the API's
  // `macSupport.features.videoModes`, populated for every VIDEO_UI_MODE on a Mac-routed model).
  const ALL_VIDEO_MODES = [
    "image_to_video",
    "text_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
    "video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    "multi_video_to_video",
    "ads2v",
    "animate_character",
  ];

  // A Mac-routed model: `capabilities` = the modes it serves, and `macSupport.features.videoModes`
  // sets those true and every other mode false (the MLX-eligibility the API computes per model).
  const macModel = (id, name, served) => ({
    id,
    name,
    type: "video",
    family: id,
    capabilities: served,
    defaults: { duration: 5, resolution: "832x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480", "480x832"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
    macSupport: {
      supported: true,
      features: {
        videoModes: Object.fromEntries(ALL_VIDEO_MODES.map((m) => [m, served.includes(m)])),
      },
    },
  });

  const MAC_CAPS = {
    macGatingActive: true,
    platform: "darwin",
    notAvailableLabel: "Not available on Mac (MLX only)",
    features: {},
    training: { supportedKernels: [], lokrOnWanSupported: false },
  };

  // Wan serves t2v / i2v / replace; SCAIL-2 serves only animate_character + replace (the trap model).
  const WAN = macModel("wan_2_2", "Wan 2.2", ["image_to_video", "text_to_video", "replace_person"]);
  const SCAIL2 = macModel("scail2_14b", "SCAIL-2", ["animate_character", "replace_person"]);

  const clip = { id: "vid_src", type: "video", projectId: "project_1", displayName: "Driving Clip" };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const modeButton = (label) => buttonWithText(container.querySelector(".mode-control"), label);
  const modelSelect = () => container.querySelector(".settings-field-model select");

  it("does not trap the user after switching to a model-specific mode", async () => {
    const context = baseContext({
      videoModels: [WAN, SCAIL2],
      assets: [clip],
      macCapabilities: MAC_CAPS,
    });
    await render(context);

    // Opens on Text→Video, served by Wan — the picker reflects the selected model.
    expect(modelSelect().value).toBe("wan_2_2");

    // Switch to Animate character → snaps the model to SCAIL-2 (the only model that serves it).
    await click(modeButton("Animate character"));
    expect(modelSelect().value).toBe("scail2_14b");

    // The bug: with SCAIL-2 selected, every other tab used to be disabled with no way back.
    // Now Text→Video / Image→Video stay enabled because Wan serves them.
    expect(modeButton("Text → Video").disabled).toBe(false);
    expect(modeButton("Image → Video").disabled).toBe(false);

    // And leaving the mode snaps back to a model that serves the target mode.
    await click(modeButton("Text → Video"));
    expect(modeButton("Text → Video").className).toContain("active");
    expect(modelSelect().value).toBe("wan_2_2");
  });

  it("disables a mode tab only when no available model serves it", async () => {
    const context = baseContext({
      videoModels: [SCAIL2],
      assets: [clip],
      macCapabilities: MAC_CAPS,
    });
    await render(context);

    // Only SCAIL-2 is installed (animate_character + replace_person). Enter animate_character so it
    // isn't the active tab being checked, then assert the unserved modes are disabled.
    await click(modeButton("Animate character"));
    expect(modeButton("Animate character").disabled).toBe(false);
    expect(modeButton("Replace person").disabled).toBe(false);
    expect(modeButton("Text → Video").disabled).toBe(true);
    expect(modeButton("Image → Video").disabled).toBe(true);
    expect(modeButton("Video → Video").disabled).toBe(true);
  });

  it("filters the model picker to models that serve the active mode", async () => {
    const context = baseContext({
      videoModels: [WAN, SCAIL2],
      assets: [clip],
      macCapabilities: MAC_CAPS,
    });
    await render(context);

    // Text→Video: only Wan serves it.
    expect([...modelSelect().options].map((o) => o.value)).toEqual(["wan_2_2"]);

    // Animate character: only SCAIL-2 serves it.
    await click(modeButton("Animate character"));
    expect([...modelSelect().options].map((o) => o.value)).toEqual(["scail2_14b"]);

    // Replace person: both backends serve it.
    await click(modeButton("Replace person"));
    expect([...modelSelect().options].map((o) => o.value)).toEqual(
      expect.arrayContaining(["wan_2_2", "scail2_14b"]),
    );
  });
});

// sc-19570 — the OFF-MAC half of the same gate, at the SCREEN. The predicate-level guards in
// `modelEligibility.test.js` proved `videoModelServesMode` consults `candleVideoModeBlock`; they
// could not prove the Advanced Video Studio consults `videoModelServesMode`. It did not: this
// screen kept a LOCAL `modelServesMode` that read the Mac block only, and `macModeTabBlocked` was
// `macGating && …`, false off-Mac by construction. So the off-Mac gate shipped in the shared
// predicate — reaching the Simple studio, the screen gate and the download offers — while the
// Advanced shell rendered every MLX-only tab and the user only found out at Generate, eating the
// new 400 instead of never seeing the tab.
//
// The observation the other guards never make is ABSENCE: what is NOT on the reachable surface.
// Asserting a submit is refused would pass on the defect. So this collects the tabs a user can
// actually click and asserts the stranded five are missing from that set, and collects the model
// picker's options and asserts the candle-unservable model is missing from those.
describe("VideoStudio off-Mac mode gating (sc-19570)", () => {
  let container;
  let root;

  const ALL_VIDEO_MODES = [
    "image_to_video",
    "text_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
    "video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    "multi_video_to_video",
    "ads2v",
    "animate_character",
  ];

  // A video model in the shape `GET /api/v1/models` serves it: `capabilities` is what the manifest
  // DECLARES, `candleSupport` is what `model_candle_support` computed from the real routing
  // predicate. The two differ for exactly the stranded pairs — declaration is not reachability.
  const model = (id, name, declared, candleServed) => ({
    id,
    name,
    type: "video",
    family: id,
    capabilities: declared,
    defaults: { duration: 5, resolution: "832x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480", "480x832"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
    candleSupport: {
      supported: candleServed.length > 0,
      features: {
        videoModes: Object.fromEntries(
          ALL_VIDEO_MODES.map((m) => [m, candleServed.includes(m)]),
        ),
      },
    },
  });

  // LTX 2.3 as shipped: it declares six modes and candle serves only `text_to_video`. The other
  // five are the pairs that queued forever off-Mac.
  const LTX_DECLARED = [
    "text_to_video",
    "image_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
  ];
  const LTX23 = model("ltx_2_3", "LTX 2.3", LTX_DECLARED, ["text_to_video"]);
  // `wan_2_2_vace_fun_14b`: its ONLY advertised mode is candle-unclaimable, so the whole model has
  // no off-Mac lane — `candleSupport.supported: false`, the case `candleAvailableModels` exists for
  // and which NOTHING imported before this.
  const VACE_FUN = model("wan_2_2_vace_fun_14b", "Wan 2.2 VACE-Fun 14B", ["replace_person"], []);
  // Bernini keeps a served non-default mode on the board, so "the stranded tabs are gone" cannot be
  // satisfied by a gate that simply disabled everything but the active tab.
  const BERNINI = model("bernini", "Bernini", ["text_to_video", "video_to_video"], [
    "text_to_video",
    "video_to_video",
  ]);

  // Off-Mac: the platform-intrinsic switch the API derives from `!is_mac`. `macGatingActive` stays
  // false — this is Windows/Linux, where every macGating helper is inert.
  const OFF_MAC_CAPS = {
    macGatingActive: false,
    candleGatingActive: true,
    platform: "linux",
    notAvailableLabel: "Not available on Mac (MLX only)",
    features: {},
    training: { supportedKernels: [], lokrOnWanSupported: false },
  };
  // The same client before `GET /api/v1/capabilities/mac` answers — and the permanent value on a
  // Mac. Every helper inert; the studio must look exactly as it did.
  const INERT_CAPS = { ...OFF_MAC_CAPS, candleGatingActive: false, platform: "" };

  const clip = {
    id: "vid_src",
    type: "video",
    projectId: "project_1",
    displayName: "Driving Clip",
  };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // The tabs a user can actually reach: rendered AND not disabled. This is the set the assertions
  // below check for absence in — a disabled tab is not a way into a mode.
  const reachableModeTabs = () =>
    [...container.querySelectorAll(".mode-control button")]
      .filter((button) => !button.disabled)
      .map((button) => button.textContent.trim());
  const modeButton = (label) =>
    [...container.querySelectorAll(".mode-control button")].find(
      (button) => button.textContent.trim() === label,
    );
  const modelSelect = () => container.querySelector(".settings-field-model select");

  const STRANDED_TABS = ["Image → Video", "First → Last", "Extend", "Bridge", "Replace person"];

  it("does not offer an MLX-only mode tab off-Mac", async () => {
    await render(
      baseContext({
        videoModels: [LTX23, BERNINI],
        assets: [clip],
        macCapabilities: OFF_MAC_CAPS,
      }),
    );

    // ABSENT — not "present but refuses on submit". Every one of these was reachable before.
    const reachable = reachableModeTabs();
    for (const label of STRANDED_TABS) {
      expect(
        reachable,
        `"${label}" has no off-Mac lane on any installed model and must not be reachable`,
      ).not.toContain(label);
      // Rendered-but-disabled is how this screen has always expressed an unavailable mode, and the
      // tooltip must name the platform doing the gating — off-Mac the Mac sentence is backwards.
      expect(modeButton(label).disabled).toBe(true);
      expect(modeButton(label).title).toBe(
        "No installed model supports this mode on this platform (macOS/MLX only).",
      );
    }
    // The served modes are untouched, so the gate narrowed the tab list rather than the board.
    expect(reachable).toContain("Text → Video");
    expect(reachable).toContain("Video → Video");
  });

  it("does not list a model with no off-Mac lane at all in the picker", async () => {
    await render(
      baseContext({
        videoModels: [BERNINI, VACE_FUN],
        assets: [clip],
        macCapabilities: OFF_MAC_CAPS,
      }),
    );

    expect([...modelSelect().options].map((option) => option.value)).not.toContain(
      "wan_2_2_vace_fun_14b",
    );
    // …and its only mode is unreachable too, so the model cannot be reached the long way round.
    expect(reachableModeTabs()).not.toContain("Replace person");
  });

  it("engages the sc-5947 download gate when the platform filter empties the catalog", async () => {
    // `baseVideoModels` was `macVideoModels.length ? macVideoModels : videoModels` — it restored the
    // UNFILTERED catalog exactly when the platform filter had emptied it. `modelReady` read off it,
    // so with every installed model off-Mac-unserveable the gate stayed `ready` and the user got the
    // whole Studio with a picker of models that serve no mode: every tab disabled, no download offer,
    // and nothing saying why. Degraded rather than hung, but it hid the one screen that could fix it.
    //
    // Pre-existing, and unreachable before this story: `macAvailableModels` alone empties the list
    // only for a Mac user whose every video model is torch-only. `candleAvailableModels` makes it
    // the ordinary case for a Windows/Linux user with an MLX-only catalog.
    await render(
      baseContext({
        videoModels: [VACE_FUN],
        assets: [clip],
        macCapabilities: OFF_MAC_CAPS,
      }),
    );

    const gate = container.querySelector(".model-availability-gate");
    expect(gate, "the sc-5947 gate must replace the studio, not sit alongside it").toBeTruthy();
    expect(gate.textContent).toContain("Video Studio needs a video model");
    // The Studio itself must be GONE, not merely covered — the model picker is the tell, because
    // that is the control the old fallback repopulated with unserveable models.
    expect(modelSelect(), "the studio's model picker must not render behind the gate").toBeFalsy();
  });

  it("does not engage the download gate for the same catalog when gating is inert", async () => {
    // The anti-vacuity leg for the test above: same single model, same `candleSupport.supported:
    // false`, switch off. Without this, a screen that gated unconditionally — on a Mac, where
    // VACE-Fun renders fine — would satisfy the assertion above.
    await render(
      baseContext({
        videoModels: [VACE_FUN],
        assets: [clip],
        macCapabilities: INERT_CAPS,
      }),
    );

    expect(container.querySelector(".model-availability-gate")).toBeFalsy();
    expect([...modelSelect().options].map((option) => option.value)).toContain(
      "wan_2_2_vace_fun_14b",
    );
  });

  it("leaves every tab and model reachable when off-Mac gating is inert", async () => {
    await render(
      baseContext({
        videoModels: [LTX23, BERNINI, VACE_FUN],
        assets: [clip],
        macCapabilities: INERT_CAPS,
      }),
    );

    // Same models, same `candleSupport` blocks, gating switch off: nothing is hidden. Without this
    // leg the assertions above would be satisfied by a screen that hid these unconditionally — on a
    // Mac, where every one of them renders.
    const reachable = reachableModeTabs();
    for (const label of [...STRANDED_TABS, "Text → Video", "Video → Video"]) {
      expect(reachable, `${label} must stay reachable when gating is inert`).toContain(label);
    }
    await click(modeButton("Replace person"));
    expect([...modelSelect().options].map((option) => option.value)).toContain(
      "wan_2_2_vace_fun_14b",
    );
  });
});

describe("VideoStudio MLX quant-tier picker (sc-12165)", () => {
  let container;
  let root;

  const MAC_CAPS = {
    macGatingActive: true,
    platform: "darwin",
    notAvailableLabel: "Not available on Mac (MLX only)",
    features: {},
    training: { supportedKernels: [], lokrOnWanSupported: false },
  };

  const tieredVideoModel = (installed, quantization = {}) => ({
    id: "bernini",
    name: "Bernini",
    type: "video",
    family: "bernini",
    adapter: "bernini",
    capabilities: ["text_to_video"],
    defaults: { duration: 5, resolution: "832x480", fps: 16 },
    limits: { durations: [5], fps: [16], resolutions: ["832x480"] },
    quantization,
    loraCompatibility: {},
    ui: {},
    hasVariantMatrix: true,
    variants: ["q4", "q8", "bf16", "nvfp4"].map((tier) => ({
      variant: tier,
      installState: installed.includes(tier) ? "installed" : "missing",
    })),
    macSupport: {
      supported: true,
      features: { videoModes: { text_to_video: true } },
    },
  });
  const scailTierModel = (installed) => ({
    ...tieredVideoModel(installed),
    id: "scail2_14b",
    name: "SCAIL-2",
    family: "scail2",
    adapter: "scail2",
    capabilities: ["animate_character"],
  });
  const replayAssets = [
    { id: "driving", type: "video", projectId: "project_1", displayName: "Driving" },
    { id: "character", type: "image", projectId: "project_1", displayName: "Character" },
  ];
  const tierReplay = (model, mlxQuantize, id = `tier-replay-${model}-${mlxQuantize}`) => ({
    id,
    view: "Video",
    recipe: {
      mode: model === "scail2_14b" ? "animate_character" : "text_to_video",
      model,
      prompt: "Replay this exact tier",
      normalizedSettings:
        model === "scail2_14b"
          ? { sourceClipAssetId: "driving", referenceAssetIds: ["character"] }
          : {},
      rawAdapterSettings: { mlxQuantize },
    },
  });

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const openAdvanced = async () => {
    const toggle = container.querySelector(".advanced-section-toggle");
    if (toggle?.getAttribute("aria-expanded") !== "true") {
      await click(toggle);
    }
  };
  const tierPicker = () => container.querySelector("label.quant-tier-picker select");
  const quantizationPicker = () =>
    [...container.querySelectorAll("label")]
      .find((label) => label.textContent.trim().startsWith("Quantization"))
      ?.querySelector("select") ?? null;

  it("shows all possible MLX tiers and honors the shared generation-quality default", async () => {
    await render(
      baseContext({
        videoModels: [tieredVideoModel(["q4", "q8"])],
        macCapabilities: MAC_CAPS,
      }),
    );

    // Quant tier is an everyday settings-bar knob, not an Advanced one (sc-15374) — no disclosure
    // needs opening. Quality is gone from that row entirely, and on a Bernini (non-SVD) adapter it
    // is not in Advanced either (sc-15398) — nothing reads the key there.
    expect(container.querySelector(".settings-bar-row label.quant-tier-picker")).toBeTruthy();
    expect(container.querySelector(".settings-bar-row .quality-segment")).toBeNull();
    await openAdvanced();
    expect(container.querySelector(".quality-segment")).toBeNull();

    // All bits-based tiers appear (NVFP4 filtered on MLX); bf16 is declared-but-not-installed → disabled.
    expect([...tierPicker().options].map((option) => option.value)).toEqual(["q4", "q8", "bf16"]);
    const disabledByTier = Object.fromEntries(
      [...tierPicker().options].map((option) => [option.value, option.disabled]),
    );
    expect(disabledByTier).toEqual({ q4: false, q8: false, bf16: true });
    expect(tierPicker().value).toBe("q8");

    // Even with a single installed tier the picker now shows (others disabled), and q4 still rides the payload.
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    const context = baseContext({
      videoModels: [tieredVideoModel(["q4"])],
      macCapabilities: MAC_CAPS,
    });
    await render(context);
    await openAdvanced();

    expect(tierPicker()).toBeTruthy();
    expect(tierPicker().value).toBe("q4");
    const soloInstalled = Object.fromEntries(
      [...tierPicker().options].map((option) => [option.value, option.disabled]),
    );
    expect(soloInstalled).toEqual({ q4: false, q8: true, bf16: true });
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced.mlxQuantize).toBe(4);
  });

  it("uses the native tier picker on Candle Wan and never emits Torch GGUF quantization", async () => {
    const wan = {
      ...tieredVideoModel(["q4", "q8"], {
        variants: { "gguf-q4_k_m": { label: "Torch Q4" } },
      }),
      id: "wan_2_2",
      name: "Wan2.2",
      family: "wan-video",
      adapter: "wan_video",
    };
    const context = baseContext({ videoModels: [wan], macCapabilities: null });
    await render(context);

    expect(tierPicker()).toBeTruthy();
    expect(quantizationPicker()).toBeNull();
    setSelect(tierPicker(), "q4");
    await act(async () => {});
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 4 });
    expect(context.createVideoJob.mock.calls[0][0].advanced).not.toHaveProperty("quantization");
  });

  it("routes the exact LTX-2.5 variant and measured q4/bf16 tiers through Candle", async () => {
    const ltx25 = {
      ...tieredVideoModel(["q4", "q8", "bf16"]),
      id: "ltx_2_5",
      name: "LTX-2.5",
      family: "ltx-video",
      adapter: "ltx_video",
      defaults: { duration: 6, resolution: "768x512", fps: 25, steps: 8 },
      limits: { steps: [8, 30] },
    };
    const context = baseContext({ videoModels: [ltx25], macCapabilities: null });
    await render(context);

    expect(tierPicker()).toBeTruthy();
    expect([...tierPicker().options].map((option) => option.value)).toEqual(["q4", "bf16"]);
    setSelect(tierPicker(), "bf16");
    await openAdvanced();
    expect(
      [...container.querySelectorAll("label")].find((label) =>
        label.textContent.includes("Enhance prompt before generation"),
      ),
    ).toBeFalsy();
    expect(container.querySelector('select[aria-label="Text encoder model"]')).toBeNull();
    setSelect(container.querySelector('select[aria-label="LTX-2.5 transformer"]'), "dev");
    await act(async () => {});
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({
      mlxQuantize: 0,
      transformerVariant: "dev",
    });
  });

  it("offers every admitted q4/q8/bf16 execution tier for Candle SCAIL-2", async () => {
    const scail = {
      ...tieredVideoModel(["q4", "q8", "bf16"]),
      id: "scail2_14b",
      name: "SCAIL-2",
      family: "scail2",
      adapter: "scail2",
    };
    const context = baseContext({ videoModels: [scail], macCapabilities: null });
    await render(context);

    expect(tierPicker()).toBeTruthy();
    expect(quantizationPicker()).toBeNull();
    expect([...tierPicker().options].map((option) => option.value)).toEqual(["q4", "q8", "bf16"]);
    expect(tierPicker().value).toBe("bf16");
    setSelect(tierPicker(), "q4");
    await act(async () => {});
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 4 });
  });

  it("admits Candle SCAIL-2 q4/q8 when those exact packages are installed", async () => {
    const scail = {
      ...tieredVideoModel(["q4", "q8"]),
      id: "scail2_14b",
      name: "SCAIL-2",
      family: "scail2",
      adapter: "scail2",
    };
    const context = baseContext({ videoModels: [scail], macCapabilities: null });
    await render(context);

    expect(tierPicker()).toBeTruthy();
    expect([...tierPicker().options].map((option) => option.value)).toEqual(["q4", "q8", "bf16"]);
    expect(tierPicker().value).toBe("q8");
    const renderButton = buttonWithText(container, "Render clip");
    expect(renderButton.disabled).toBe(false);
    await click(renderButton);
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 8 });

    // The engine id alone never creates a selectable precision. A stale/non-matrix catalog row
    // remains picker-free and blocked instead of manufacturing a payload for an incomplete install.
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    const noVariants = { ...scail, hasVariantMatrix: false, variants: [] };
    const noVariantsContext = baseContext({ videoModels: [noVariants], macCapabilities: null });
    await render(noVariantsContext);
    expect(tierPicker()).toBeNull();
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);
    expect(noVariantsContext.createVideoJob).not.toHaveBeenCalled();
  });

  for (const [tier, mlxQuantize] of [
    ["Q4", 4],
    ["Q8", 8],
  ]) {
    it(`replays the exact admitted Candle SCAIL-2 ${tier} recipe`, async () => {
      const context = baseContext({
        videoModels: [scailTierModel(["q4", "q8", "bf16"])],
        assets: replayAssets,
        macCapabilities: null,
        studioLaunch: tierReplay("scail2_14b", mlxQuantize),
      });
      await render(context);

      expect(tierPicker().value).toBe(tier.toLowerCase());
      const renderButton = buttonWithText(container, "Render clip");
      expect(renderButton.disabled).toBe(false);
      await click(renderButton);
      expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize });
    });
  }

  it("submits an admitted exact replay tier while switching between native backends", async () => {
    const createVideoJob = vi.fn();
    const common = {
      videoModels: [scailTierModel(["q4", "q8", "bf16"])],
      assets: replayAssets,
      createVideoJob,
      studioLaunch: tierReplay("scail2_14b", 8, "candle-to-mlx-q8"),
    };
    await render(baseContext({ ...common, macCapabilities: null }));

    expect(tierPicker().value).toBe("q8");
    expect(buttonWithText(container, "Render clip").disabled).toBe(false);

    // Q8 is admitted on both native backends, so the exact replay remains submit-ready during the
    // backend transition instead of being silently downgraded to bf16.
    flushSync(() => {
      root.render(
        <AppContext.Provider value={baseContext({ ...common, macCapabilities: MAC_CAPS })}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    container
      .querySelector("form")
      .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    expect(createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 8 });

    await act(async () => {});
    expect(tierPicker().value).toBe("q8");
    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    await click(buttonWithText(container, "Render clip"));
    expect(createVideoJob.mock.calls[1][0].advanced).toMatchObject({ mlxQuantize: 8 });
  });

  it("keeps replay blocked when capability refresh removes its model and reapplies it on return", async () => {
    const bernini = tieredVideoModel(["q4", "q8"]);
    const fallback = {
      ...tieredVideoModel(["q4"]),
      id: "fallback_video",
      name: "Fallback Video",
    };
    const createVideoJob = vi.fn();
    const studioLaunch = tierReplay("bernini", 8, "refresh-removes-bernini-q8");
    const contextFor = (videoModels) =>
      baseContext({
        videoModels,
        createVideoJob,
        macCapabilities: MAC_CAPS,
        studioLaunch,
    });
    await render(contextFor([bernini, fallback]));
    expect(tierPicker().value).toBe("q8");

    // The capability response and a native form submit can land in the same turn. The stale target
    // is no longer available even before the automatic model-snap effect writes the fallback.
    flushSync(() => {
      root.render(
        <AppContext.Provider value={contextFor([fallback])}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    container
      .querySelector("form")
      .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    expect(createVideoJob).not.toHaveBeenCalled();
    await act(async () => {});
    expect(container.querySelector(".settings-field-model select").value).toBe("fallback_video");
    expect(tierPicker().value).toBe("q4");
    expect(container.textContent).toContain("requires Bernini at Q8");
    expect(container.textContent).toContain("not available on the current backend");
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(createVideoJob).not.toHaveBeenCalled();

    // No retirement occurred. When the authoritative catalog returns, the exact target and tier
    // return too; replay does not silently become a new Fallback Video Q4 generation.
    await render(contextFor([bernini, fallback]));
    expect(container.querySelector(".settings-field-model select").value).toBe("bernini");
    expect(tierPicker().value).toBe("q8");
    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    await click(buttonWithText(container, "Render clip"));
    expect(createVideoJob.mock.calls[0][0]).toMatchObject({
      model: "bernini",
      advanced: { mlxQuantize: 8 },
    });
  });

  it("withholds catalog-removal recovery until fallback model state and its shared tier align", async () => {
    const bernini = tieredVideoModel(["q8"]);
    const fallback = {
      ...tieredVideoModel(["q8"]),
      id: "fallback_video",
      name: "Fallback Video",
    };
    const createVideoJob = vi.fn();
    const studioLaunch = tierReplay("bernini", 8, "shared-q8-catalog-removal");
    const contextFor = (videoModels) =>
      baseContext({
        videoModels,
        createVideoJob,
        macCapabilities: MAC_CAPS,
        studioLaunch,
      });
    await render(contextFor([bernini, fallback]));
    expect(tierPicker().value).toBe("q8");

    flushSync(() => {
      root.render(
        <AppContext.Provider value={contextFor([fallback])}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });

    // The derived catalog model and tier happen to look ready, but payload `model` state still
    // names removed Bernini until the passive snap. No recovery action may retire replay here.
    expect(
      [...container.querySelectorAll("button")].some((button) =>
        button.textContent.includes("Start new generation with Fallback Video"),
      ),
    ).toBe(false);
    container
      .querySelector("form")
      .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    expect(createVideoJob).not.toHaveBeenCalled();

    await act(async () => {});
    expect(container.querySelector(".settings-field-model select").value).toBe("fallback_video");
    expect(tierPicker().value).toBe("q8");
    const startNew = buttonWithText(
      container,
      "Start new generation with Fallback Video Q8 (balanced)",
    );
    expect(startNew).toBeTruthy();
    await click(startNew);
    await click(buttonWithText(container, "Render clip"));
    expect(createVideoJob.mock.calls[0][0]).toMatchObject({
      model: "fallback_video",
      advanced: { mlxQuantize: 8 },
    });
  });

  it("blocks a mode fallback until an explicit action starts a new exact-tier generation", async () => {
    const bernini = tieredVideoModel(["q4", "q8"]);
    const imageVideo = {
      ...tieredVideoModel(["q4"]),
      id: "image_video",
      name: "Image Video",
      capabilities: ["image_to_video"],
      macSupport: {
        supported: true,
        features: { videoModes: { image_to_video: true } },
      },
    };
    const source = {
      id: "source_image",
      type: "image",
      projectId: "project_1",
      displayName: "Source Image",
    };
    const createVideoJob = vi.fn();
    const replay = tierReplay("bernini", 8, "mode-fallback-bernini-q8");
    replay.recipe.normalizedSettings = { sourceAssetId: source.id };
    const context = baseContext({
      videoModels: [bernini, imageVideo],
      assets: [source],
      createVideoJob,
      macCapabilities: MAC_CAPS,
      studioLaunch: replay,
    });
    await render(context);

    await click(buttonWithText(container.querySelector(".mode-control"), "Image → Video"));
    expect(container.querySelector(".settings-field-model select").value).toBe("image_video");
    expect(tierPicker().value).toBe("q4");
    expect(container.textContent).toContain("requires Bernini at Q8");
    expect(container.textContent).toContain("Image Video is active for the current mode");
    const renderButton = buttonWithText(container, "Render clip");
    expect(renderButton.disabled).toBe(true);
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(createVideoJob).not.toHaveBeenCalled();

    const startNew = buttonWithText(
      container,
      "Start new generation with Image Video Q4 (smallest)",
    );
    expect(startNew).toBeTruthy();
    await click(startNew);
    expect(container.textContent).not.toContain("requires Bernini at Q8");
    expect(renderButton.disabled).toBe(false);
    await click(renderButton);
    expect(createVideoJob.mock.calls[0][0]).toMatchObject({
      mode: "image_to_video",
      model: "image_video",
      sourceAssetId: "source_image",
      advanced: { mlxQuantize: 4 },
    });
  });

  it("retires a rejected replay through the Replace Person engine selector", async () => {
    const scail = {
      ...scailTierModel(["q4", "q8", "bf16"]),
      capabilities: ["replace_person"],
    };
    const wan = {
      ...tieredVideoModel(["q4"]),
      id: "wan_2_2",
      name: "Wan 2.2",
      family: "wan-video",
      adapter: "wan_video",
      capabilities: ["replace_person"],
    };
    const context = baseContext({
      videoModels: [wan, scail],
      assets: replayAssets,
      characters: [{ id: "char_1", name: "Ada", projectId: "project_1" }],
      personTracks: [
        {
          id: "track_1",
          name: "Selected person",
          projectId: "project_1",
          sourceAssetId: "driving",
          status: {},
          frames: [],
        },
      ],
      macCapabilities: null,
      studioLaunch: {
        id: "replace-scail-q8",
        view: "Video",
        recipe: {
          mode: "replace_person",
          model: "scail2_14b",
          prompt: "Replay this exact replacement",
          normalizedSettings: {
            sourceClipAssetId: "driving",
            personTrackId: "track_1",
            characterId: "char_1",
          },
          rawAdapterSettings: { mlxQuantize: 8 },
        },
      },
    });
    await render(context);

    expect(container.textContent).not.toContain("This recipe was recorded at Q8");
    const engineSelect = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.includes("Replacement engine"))
      .querySelector("select");
    setSelect(engineSelect, "wan_2_2");
    await act(async () => {});
    expect(container.textContent).not.toContain("This recipe was recorded at Q8");

    setSelect(engineSelect, "scail2_14b");
    await act(async () => {});
    expect(container.textContent).not.toContain("This recipe was recorded at Q8");
    expect(tierPicker().value).toBe("bf16");
    const submitCta = container.querySelector(".prompt-cta");
    expect(submitCta.disabled).toBe(false);
    await click(submitCta);
    expect(context.createVideoJob.mock.calls[0][0]).toMatchObject({
      mode: "replace_person",
      model: "scail2_14b",
      advanced: { mlxQuantize: 0 },
    });
  });

  it("refuses an unavailable Candle SCAIL-2 tier after a recipe switches models", async () => {
    const initial = tieredVideoModel(["q4"]);
    const context = baseContext({
      videoModels: [initial, scailTierModel(["q4", "q8", "bf16"])],
      assets: replayAssets,
      macCapabilities: null,
      studioLaunch: tierReplay("scail2_14b", 8, "switch-to-scail-q8"),
    });
    await render(context);

    expect(container.querySelector(".settings-field-model select").value).toBe("scail2_14b");
    expect(tierPicker().value).toBe("q8");
    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    expect(container.textContent).not.toContain("This recipe was recorded at Q8");
  });

  it("replays the exact accepted bf16 tier for Candle SCAIL-2", async () => {
    const context = baseContext({
      videoModels: [scailTierModel(["q4", "q8", "bf16"])],
      assets: replayAssets,
      macCapabilities: null,
      studioLaunch: tierReplay("scail2_14b", 0, "scail-bf16"),
    });
    await render(context);

    expect(tierPicker().value).toBe("bf16");
    const renderButton = buttonWithText(container, "Render clip");
    expect(renderButton.disabled).toBe(false);
    await click(renderButton);
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 0 });
  });

  for (const [tier, mlxQuantize] of [
    ["q4", 4],
    ["q8", 8],
  ]) {
    it(`keeps an exact SCAIL-2 ${tier} recipe selectable and replayable on MLX`, async () => {
      const context = baseContext({
        videoModels: [scailTierModel(["q4", "q8"])],
        assets: replayAssets,
        macCapabilities: MAC_CAPS,
        studioLaunch: tierReplay("scail2_14b", mlxQuantize, `scail-mlx-${tier}`),
      });
      await render(context);

      expect(tierPicker().value).toBe(tier);
      const renderButton = buttonWithText(container, "Render clip");
      expect(renderButton.disabled).toBe(false);
      await click(renderButton);
      expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize });
    });
  }

  it("refuses any recorded native tier that is unavailable for its active model", async () => {
    const bernini = tieredVideoModel(["q4"]);
    const context = baseContext({
      videoModels: [bernini],
      macCapabilities: MAC_CAPS,
      studioLaunch: tierReplay("bernini", 8, "bernini-missing-q8"),
    });
    await render(context);

    expect(tierPicker().value).toBe("q4");
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);
    expect(container.textContent).toContain("recorded at Q8");
    expect(container.textContent).toContain("not installed and available for Bernini on MLX");
    expect(context.createVideoJob).not.toHaveBeenCalled();
  });

  it("keeps installed q4/q8 selectable for SCAIL-2 on MLX", async () => {
    const scail = {
      ...tieredVideoModel(["q4", "q8"]),
      id: "scail2_14b",
      name: "SCAIL-2",
      family: "scail2",
      adapter: "scail2",
    };
    const context = baseContext({ videoModels: [scail], macCapabilities: MAC_CAPS });
    await render(context);

    expect([...tierPicker().options].map((option) => option.value)).toEqual(["q4", "q8", "bf16"]);
    expect(Object.fromEntries([...tierPicker().options].map((option) => [option.value, option.disabled])))
      .toEqual({ q4: false, q8: false, bf16: true });
    setSelect(tierPicker(), "q8");
    await act(async () => {});
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 8 });
  });

  it("keeps the candle-only NVFP4 tier out of the MLX video picker (sc-11042)", async () => {
    const context = baseContext({
      videoModels: [tieredVideoModel(["q4", "nvfp4"])],
      macCapabilities: MAC_CAPS,
    });
    await render(context);
    await openAdvanced();

    // The picker shows the bits-based tiers, but NVFP4 is filtered out even though it's installed.
    expect(tierPicker()).toBeTruthy();
    const optionValues = [...tierPicker().options].map((option) => option.value);
    expect(optionValues).toEqual(["q4", "q8", "bf16"]);
    expect(optionValues).not.toContain("nvfp4");
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 4 });
    expect(context.createVideoJob.mock.calls[0][0].advanced).not.toHaveProperty("quantTier");
  });

  it("emits a selected high tier, warns about memory, and restores the video/model sticky", async () => {
    const model = tieredVideoModel(["q4", "q8", "bf16"]);
    const context = baseContext({ videoModels: [model], macCapabilities: MAC_CAPS });
    await render(context);
    await openAdvanced();

    setSelect(tierPicker(), "q8");
    await act(async () => {});
    expect(container.querySelector(".quant-tier-memory-note")?.textContent).toContain(
      "may run out of memory",
    );
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced.mlxQuantize).toBe(8);

    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await render(baseContext({ videoModels: [model], macCapabilities: MAC_CAPS }));
    await openAdvanced();
    expect(tierPicker().value).toBe("q8");
  });

  it("keeps MLX tiers disjoint from the torch/GGUF quantization control and payload", async () => {
    const both = tieredVideoModel(["q4", "q8"], {
      variants: { q4_k_m: { label: "Q4_K_M" } },
    });
    const mlxContext = baseContext({ videoModels: [both], macCapabilities: MAC_CAPS });
    await render(mlxContext);
    await openAdvanced();

    expect(tierPicker()).toBeTruthy();
    expect(quantizationPicker()).toBeNull();
    setSelect(tierPicker(), "q8");
    await act(async () => {});
    await click(buttonWithText(container, "Render clip"));
    expect(mlxContext.createVideoJob.mock.calls[0][0].advanced).toMatchObject({ mlxQuantize: 8 });
    expect(mlxContext.createVideoJob.mock.calls[0][0].advanced).not.toHaveProperty("quantization");

    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    const torchContext = baseContext({ videoModels: [both] });
    await render(torchContext);
    await openAdvanced();

    expect(tierPicker()).toBeNull();
    expect(quantizationPicker()).toBeTruthy();
    setSelect(quantizationPicker(), "q4_k_m");
    await act(async () => {});
    await click(buttonWithText(container, "Render clip"));
    expect(torchContext.createVideoJob.mock.calls[0][0].advanced.quantization).toBe("q4_k_m");
    expect(torchContext.createVideoJob.mock.calls[0][0].advanced).not.toHaveProperty("mlxQuantize");
  });
});

// Lightning fast-4-step toggle for Wan2.2 A14B MoE (T2V + I2V) — epic 10043, sc-10048.
// Renders default-on for the two A14B engines only; maps to advanced.lightning; when on it
// governs (disables) the manual steps/guidance controls.
describe("VideoStudio Lightning toggle (sc-10048)", () => {
  let container;
  let root;

  // The two A14B MoE engines that honor advanced.lightning (constants.js WAN_A14B_LIGHTNING_MODEL_IDS).
  const WAN_T2V_14B = {
    id: "wan_2_2_t2v_14b",
    name: "Wan2.2 14B (T2V)",
    type: "video",
    family: "wan-video",
    capabilities: ["text_to_video"],
    defaults: { duration: 5, resolution: "832x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480", "480x832"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };
  const WAN_I2V_14B = {
    ...WAN_T2V_14B,
    id: "wan_2_2_i2v_14b",
    name: "Wan2.2 14B (I2V)",
    capabilities: ["image_to_video", "text_to_video"],
  };
  // Dense 5B (single-expert) — ignores advanced.lightning, so no toggle.
  const WAN_5B = {
    ...WAN_T2V_14B,
    id: "wan_2_2",
    name: "Wan 2.2 (5B)",
    capabilities: ["text_to_video"],
  };
  // A non-Wan engine (LTX) — no toggle.
  const NON_WAN = {
    id: "ltx_2_3",
    name: "LTX 2.3",
    type: "video",
    family: "ltx-video",
    capabilities: ["text_to_video"],
    defaults: { duration: 6, resolution: "768x512", fps: 25 },
    limits: {},
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // Open the Advanced disclosure where the toggle + steps/guidance live.
  const openAdvanced = async () => {
    await click(container.querySelector(".advanced-section-toggle"));
  };
  const lightningLabel = () =>
    [...container.querySelectorAll(".lightning-toggle label")].find((el) =>
      el.textContent.includes("Lightning"),
    );
  const lightningCheckbox = () => lightningLabel()?.querySelector('input[type="checkbox"]');
  const labeledInput = (label) =>
    [...container.querySelectorAll("label")]
      .find((el) => el.textContent.trim().startsWith(label))
      ?.querySelector("input");

  it("renders default-on for Wan A14B T2V", async () => {
    const context = baseContext({ videoModels: [WAN_T2V_14B] });
    await render(context);
    await openAdvanced();

    const box = lightningCheckbox();
    expect(box).toBeTruthy();
    expect(box.checked).toBe(true);
  });

  it("renders default-on for Wan A14B I2V", async () => {
    const context = baseContext({ videoModels: [WAN_I2V_14B] });
    await render(context);
    await openAdvanced();

    const box = lightningCheckbox();
    expect(box).toBeTruthy();
    expect(box.checked).toBe(true);
  });

  it("is not rendered for the dense Wan 5B", async () => {
    const context = baseContext({ videoModels: [WAN_5B] });
    await render(context);
    await openAdvanced();

    expect(lightningCheckbox()).toBeFalsy();
  });

  it("is not rendered for a non-Wan engine", async () => {
    const context = baseContext({ videoModels: [NON_WAN] });
    await render(context);
    await openAdvanced();

    expect(lightningCheckbox()).toBeFalsy();
  });

  it("sends advanced.lightning = true by default for Wan A14B T2V", async () => {
    const context = baseContext({ videoModels: [WAN_T2V_14B] });
    await render(context);

    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.model).toBe("wan_2_2_t2v_14b");
    expect(payload.advanced.lightning).toBe(true);
    // On → the worker governs steps/guidance, so the UI suppresses the overrides.
    expect(payload.advanced.steps).toBeUndefined();
    expect(payload.advanced.guidanceScale).toBeUndefined();
  });

  it("sends advanced.lightning = false when the toggle is turned off", async () => {
    const context = baseContext({ videoModels: [WAN_T2V_14B] });
    await render(context);
    await openAdvanced();

    await act(async () => lightningCheckbox().click());
    expect(lightningCheckbox().checked).toBe(false);

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.advanced.lightning).toBe(false);
  });

  it("disables the manual steps/guidance controls while Lightning is on and re-enables them off", async () => {
    const context = baseContext({ videoModels: [WAN_T2V_14B] });
    await render(context);
    await openAdvanced();

    // On (default): governed by the recipe.
    expect(labeledInput("Steps").disabled).toBe(true);
    expect(labeledInput("Guidance").disabled).toBe(true);

    // Off: the user's steps/guidance controls become active.
    await act(async () => lightningCheckbox().click());
    expect(labeledInput("Steps").disabled).toBe(false);
    expect(labeledInput("Guidance").disabled).toBe(false);
  });

  // sc-19502 — a distilled model bakes its sigma waypoints into training, so LTX-2.3 renders at
  // exactly 8 steps and BOTH backends now refuse anything else. A free Steps box here would let the
  // user type a number the enqueue gate 400s on, which is the unsatisfiable-control defect the
  // story names.
  const DISTILLED_LTX = {
    ...NON_WAN,
    defaults: { duration: 6, resolution: "768x512", fps: 25, steps: 8 },
    limits: { steps: [8] },
    ui: { label: "LTX-2.3" },
  };
  const LTX25 = {
    ...NON_WAN,
    id: "ltx_2_5",
    name: "LTX-2.5",
    adapter: "ltx_video",
    video: { supportsGuidance: false, supportsNegativePrompt: false },
    defaults: { duration: 6, resolution: "768x512", fps: 25, steps: 8 },
    limits: { steps: [8, 30] },
    hasVariantMatrix: true,
    variants: ["q4", "q8", "bf16"].map((variant) => ({
      variant,
      installState: "installed",
    })),
    // Runtime discovery is adapter-wide today. The UI must not mistake these LTX-2.3 options for
    // an LTX-2.5 capability merely because they appear on the enriched catalog entry.
    textEncoderOptions: [
      { id: "default", label: "Gemma 3 default", isDefault: true },
      { id: "ltx_amoral_gemma_3_12b", label: "Amoral Gemma 3", isDefault: false },
    ],
    ui: { label: "LTX-2.5" },
  };
  const LTX25_MAC_CAPS = {
    macGatingActive: true,
    candleGatingActive: false,
    platform: "macos",
    features: {},
    training: { supportedKernels: [], lokrOnWanSupported: false },
  };

  it("declares LTX-2.5 as opt-in with its Gemma 4 prompt guide while LTX-2.3 stays recommended", () => {
    const manifestPath = resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../../../config/manifests/builtin.models.jsonc",
    );
    const models = JSON5.parse(readFileSync(manifestPath, "utf8")).models;
    const ltx25 = models.find((entry) => entry.id === "ltx_2_5");
    const ltx23 = models.find((entry) => entry.id === "ltx_2_3");

    expect(ltx25?.recommended).toBe(false);
    expect(ltx25?.ui?.recommendedFor).toEqual([
      "image_to_video",
      "text_to_video",
      "first_last_frame",
      "extend_clip",
      "video_bridge",
      "replace_person",
    ]);
    expect(ltx25?.ui?.promptGuide).toEqual({
      title: "LTX-2.5 Gemma 4 Prompt Guide",
      path: "/prompt-guides/ltx-2-5.md",
      sources: [
        {
          label: "Lightricks LTX-2.5 model card",
          url: "https://huggingface.co/Lightricks/LTX-2.5",
        },
        {
          label: "Lightricks LTX-2.5 Diffusers model card",
          url: "https://huggingface.co/Lightricks/LTX-2.5-Diffusers",
        },
        {
          label: "Lightricks Gemma 4 text-to-video prompt contract",
          url: "https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_t2v_system_prompt.txt",
        },
        {
          label: "Lightricks Gemma 4 image-to-video prompt contract",
          url: "https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_i2v_system_prompt.txt",
        },
      ],
    });
    expect(ltx23?.recommended).toBe(true);
  });

  it("keeps LTX-2.3 as the default and confines LTX-2.5 execution controls to its Advanced panel", async () => {
    const context = baseContext({
      videoModels: [LTX25, DISTILLED_LTX],
      macCapabilities: LTX25_MAC_CAPS,
    });
    await render(context);

    expect(container.querySelector(".settings-field-model select").value).toBe("ltx_2_3");
    await openAdvanced();
    expect(container.querySelector('[aria-label="LTX-2.5 VAE decoder"]')).toBeNull();
    expect(container.querySelector('[aria-label="LTX-2.5 auto duration"]')).toBeNull();
    expect(container.querySelector('[aria-label="LTX-2.5 temporal upsampling"]')).toBeNull();
  });

  it("submits LTX-2.5 decoder, duration-head bounds, and temporal upsampling without a manual duration", async () => {
    const context = baseContext({ videoModels: [LTX25], macCapabilities: LTX25_MAC_CAPS });
    await render(context);
    await openAdvanced();

    const decoder = container.querySelector('[aria-label="LTX-2.5 VAE decoder"]');
    const autoDuration = container.querySelector('[aria-label="LTX-2.5 auto duration"]');
    const minDuration = container.querySelector('[aria-label="LTX-2.5 auto duration minimum"]');
    const maxDuration = container.querySelector('[aria-label="LTX-2.5 auto duration maximum"]');
    const temporal = container.querySelector('[aria-label="LTX-2.5 temporal upsampling"]');

    expect(decoder.value).toBe("conv");
    expect(autoDuration.checked).toBe(false);
    expect(minDuration.disabled).toBe(true);
    expect(maxDuration.disabled).toBe(true);
    expect(temporal.value).toBe("0");

    await act(async () => setSelect(decoder, "diffusion"));
    await act(async () => autoDuration.click());
    expect(minDuration.disabled).toBe(false);
    expect(maxDuration.disabled).toBe(false);
    await act(async () => setInput(minDuration, "5"));
    await act(async () => setInput(maxDuration, "12"));
    await act(async () => setSelect(temporal, "2"));

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).not.toHaveProperty("duration");
    expect(payload.advanced).toMatchObject({
      vaeDecoder: "diffusion",
      autoDuration: true,
      autoDurationMinSeconds: 5,
      autoDurationMaxSeconds: 12,
      temporalUpsampleRounds: 2,
    });
  });

  it("returns LTX-2.5 to an explicit duration when the duration picker changes", async () => {
    const context = baseContext({ videoModels: [LTX25], macCapabilities: LTX25_MAC_CAPS });
    await render(context);
    await openAdvanced();
    await act(async () => container.querySelector('[aria-label="LTX-2.5 auto duration"]').click());

    const durationPicker = container.querySelector(".settings-field-count select");
    await act(async () => setSelect(durationPicker, "8"));
    await click(buttonWithText(container, "Render clip"));

    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.duration).toBe(8);
    expect(payload.advanced.autoDuration).toBe(false);
  });

  it("routes the LTX-2.5 dev selector into the request and pins its 30-step row", async () => {
    const context = baseContext({ videoModels: [LTX25], macCapabilities: LTX25_MAC_CAPS });
    await render(context);
    await openAdvanced();

    const variant = container.querySelector('select[aria-label="LTX-2.5 transformer"]');
    expect(variant.value).toBe("distilled");
    await act(async () => setSelect(variant, "dev"));
    const steps = labeledInput("Steps");
    expect(steps.disabled).toBe(true);
    expect(steps.placeholder).toBe("30 (fixed schedule)");
    expect(labeledInput("Guidance")).toBeFalsy();
    expect(labeledInput("Video CFG")).toBeFalsy();
    expect(labeledInput("Video STG")).toBeFalsy();
    expect(labeledInput("Video rescale")).toBeFalsy();

    const negative = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.trim().startsWith("Negative prompt"))
      ?.querySelector("textarea");
    expect(negative).toBeTruthy();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      setter.call(negative, "washed out");
      negative.dispatchEvent(new window.Event("input", { bubbles: true }));
    });

    const renderButton = buttonWithText(container, "Render clip");
    expect(renderButton.disabled, container.textContent).toBe(false);
    await click(renderButton);
    expect(context.createVideoJob, container.textContent).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      model: "ltx_2_5",
      negativePrompt: "washed out",
      advanced: { transformerVariant: "dev" },
    });
    expect(payload.advanced.steps).toBeUndefined();
    expect(payload.advanced.guidanceScale).toBeUndefined();
  });

  it("keeps distilled LTX-2.5 sealed while exposing only its stock prompt enhancer", async () => {
    const context = baseContext({ videoModels: [LTX25], macCapabilities: LTX25_MAC_CAPS });
    await render(context);
    await openAdvanced();

    const steps = labeledInput("Steps");
    expect(steps.disabled).toBe(true);
    expect(steps.placeholder).toBe("8 (fixed schedule)");
    expect(labeledInput("Guidance")).toBeFalsy();
    expect(
      [...container.querySelectorAll("label")].find((label) =>
        label.textContent.trim().startsWith("Negative prompt"),
      ),
    ).toBeFalsy();
    expect(labeledInput("Video CFG")).toBeFalsy();
    expect(labeledInput("Video STG")).toBeFalsy();
    expect(labeledInput("Video rescale")).toBeFalsy();
    expect(container.querySelector('select[aria-label="Text encoder model"]')).toBeNull();

    const enhance = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.includes("Enhance prompt before generation"))
      ?.querySelector('input[type="checkbox"]');
    expect(enhance).toBeTruthy();
    expect(container.textContent).toContain("stock Gemma-4 enhancer");
    await act(async () => enhance.click());

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.negativePrompt).toBe("");
    expect(payload.advanced).toMatchObject({
      transformerVariant: "distilled",
      enhancePrompt: true,
    });
    for (const absent of [
      "steps",
      "guidanceScale",
      "textEncoderModel",
      "videoCfgGuidanceScale",
      "videoStgGuidanceScale",
      "videoRescaleScale",
    ]) {
      expect(payload.advanced).not.toHaveProperty(absent);
    }
  });

  it("pins the Steps control for a model that declares a single legal step count", async () => {
    const context = baseContext({ videoModels: [DISTILLED_LTX] });
    await render(context);
    await openAdvanced();

    const steps = labeledInput("Steps");
    expect(steps.disabled).toBe(true);
    // The value stays VISIBLE rather than hidden: the user should know the render is 8 steps. This
    // is the third case in the screen's taxonomy — not the transient Lightning disable, not the
    // permanently-absent `supportsGuidance` hide.
    expect(steps.placeholder).toBe("8 (fixed schedule)");
    expect(steps.title).toMatch(/fixed 8-step schedule/);
  });

  it("leaves the Steps control free for a model that declares no step menu", async () => {
    // The overwhelmingly common case. A blanket pin would remove a working control from every
    // other video model, so this is the half that keeps the test above honest.
    const context = baseContext({ videoModels: [NON_WAN] });
    await render(context);
    await openAdvanced();

    expect(labeledInput("Steps").disabled).toBe(false);
  });

  it("never emits an off-menu advanced.steps after switching to a pinned model", async () => {
    // The real path to a stale value: type 30 while an UNPINNED model is selected, then switch to
    // the distilled one. `stepsOverride` genuinely holds "30" at that point, so this exercises the
    // suppression rather than an empty box (which would pass no matter what the code did).
    const unpinned = { ...NON_WAN, id: "wan_2_2", name: "Wan 2.2" };
    const context = baseContext({ videoModels: [unpinned, DISTILLED_LTX] });
    await render(context);
    await openAdvanced();

    const modelPicker = () =>
      [...container.querySelectorAll("label")]
        .find((el) => el.textContent.trim().startsWith("Model"))
        ?.querySelector("select");

    await act(async () => setSelect(modelPicker(), "wan_2_2"));
    expect(labeledInput("Steps").disabled).toBe(false);
    await act(async () => setInput(labeledInput("Steps"), "30"));
    expect(labeledInput("Steps").value).toBe("30");

    // Switch to the distilled model: the control pins and the stale 30 must not reach the payload.
    await act(async () => setSelect(modelPicker(), "ltx_2_3"));
    expect(labeledInput("Steps").disabled).toBe(true);

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.model).toBe("ltx_2_3");
    // Omitting `steps` is what tells the engine to run its baked schedule, and it cannot 400.
    expect(payload.advanced.steps).toBeUndefined();
  });

  // sc-19502 — the MULTI-entry half of the same constraint. `steps_limit_error` refuses off-menu
  // counts by MEMBERSHIP, not by a floor, so a two-entry menu makes a free-text box exactly as
  // unsatisfiable as it is for the pinned case: the user can type 30 and the enqueue gate 400s.
  // Every other reader on this seam is already set-shaped, so the studio must be too.
  const MENU_LTX = {
    ...NON_WAN,
    id: "ltx_menu",
    name: "LTX (two schedules)",
    // Fixed point: `defaults.steps` is ON the menu, the manifest invariant the Rust side asserts
    // (`shipped_manifest_step_limits_are_what_core_reads`). It is NOT `limits.steps[0]` by
    // definition — declared second here so a `[0]`-as-default regression cannot pass.
    defaults: { duration: 6, resolution: "768x512", fps: 25, steps: 12 },
    limits: { steps: [4, 12] },
    ui: { label: "LTX Menu" },
  };
  const labeledSelect = (label) =>
    [...container.querySelectorAll("label")]
      .find((el) => el.textContent.trim().startsWith(label))
      ?.querySelector("select");

  it("offers a menu-shaped Steps picker for a model declaring more than one legal count", async () => {
    const context = baseContext({ videoModels: [MENU_LTX] });
    await render(context);
    await openAdvanced();

    // A free-text box here would be the "UI looser than the gate" desync. It must be a picker.
    expect(labeledInput("Steps")).toBeNull();
    const steps = labeledSelect("Steps");
    expect(steps).toBeTruthy();
    expect(steps.disabled).toBe(false);
    // Exactly the declared menu, in declared order, plus the cleared "model default" entry the
    // panel's own `cleared values → model default` hint promises. Nothing else is renderable.
    expect([...steps.options].map((option) => option.value)).toEqual(["", "4", "12"]);
    // The cleared entry shows `defaults.steps` (12) — NOT `limits.steps[0]` (4). If the empty option
    // ever started advertising the menu's first entry, this is the assertion that catches it.
    expect(steps.options[0].textContent).toBe("12 (model default)");
    expect(steps.value).toBe("");
    expect(steps.title).toMatch(/renders at 4 or 12 steps only/);
  });

  it("never emits an off-menu advanced.steps for a multi-entry menu, and does emit an on-menu one", async () => {
    // Same real staleness path as the pinned case: type 30 against an unmenued model, then switch.
    const unmenued = { ...NON_WAN, id: "wan_2_2", name: "Wan 2.2" };
    const context = baseContext({ videoModels: [unmenued, MENU_LTX] });
    await render(context);
    await openAdvanced();

    const modelPicker = () =>
      [...container.querySelectorAll("label")]
        .find((el) => el.textContent.trim().startsWith("Model"))
        ?.querySelector("select");

    await act(async () => setSelect(modelPicker(), "wan_2_2"));
    await act(async () => setInput(labeledInput("Steps"), "30"));
    expect(labeledInput("Steps").value).toBe("30");

    await act(async () => setSelect(modelPicker(), "ltx_menu"));
    // The stale 30 is off this model's menu, so the picker falls back to the cleared entry rather
    // than to the menu's first option — which is what a `<select>` with an unmatched value shows,
    // and would silently assert `limits.steps[0]` is the default.
    expect(labeledSelect("Steps").value).toBe("");

    await click(buttonWithText(container, "Render clip"));
    const stale = context.createVideoJob.mock.calls[0][0];
    expect(stale.model).toBe("ltx_menu");
    expect(stale.advanced.steps).toBeUndefined();

    // …and the picker is a working control, not a decoration: an on-menu pick reaches the payload.
    // Without this half, suppressing `steps` unconditionally would pass the assertion above.
    await act(async () => setSelect(labeledSelect("Steps"), "4"));
    expect(labeledSelect("Steps").value).toBe("4");
    await click(buttonWithText(container, "Render clip"));
    const picked = context.createVideoJob.mock.calls[1][0];
    expect(picked.advanced.steps).toBe(4);
  });

  it("does not emit advanced.lightning for a non-Wan engine", async () => {
    const context = baseContext({ videoModels: [NON_WAN] });
    await render(context);

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.advanced.lightning).toBeUndefined();
  });
});

describe("VideoStudio runtime text-encoder selector (sc-13800)", () => {
  let container;
  let root;

  const DEFAULT_ENCODER = {
    id: "default",
    label: "Shipped Gemma 3 12B (default)",
    description: "Uses the Gemma text encoder installed with LTX.",
    isDefault: true,
  };
  const AMORAL_ENCODER = {
    id: "ltx_amoral_gemma_3_12b",
    label: "Amoral Gemma 3 12B (operator staged)",
    description: "Uses the complete alternate Gemma snapshot already staged by the operator.",
    isDefault: false,
  };
  const LTX_WITH_ENCODERS = {
    ...LTX,
    adapter: "ltx_video",
    textEncoderOptions: [DEFAULT_ENCODER, AMORAL_ENCODER],
  };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const encoderSelect = () => container.querySelector('select[aria-label="Text encoder model"]');
  const enhanceCheckbox = () =>
    [...container.querySelectorAll("label")]
      .find((label) => label.textContent.includes("Enhance prompt before generation"))
      ?.querySelector('input[type="checkbox"]');

  it("is model-driven and hides an alternate the runtime did not enumerate", async () => {
    await render(
      baseContext({
        videoModels: [
          {
            ...LTX_WITH_ENCODERS,
            textEncoderOptions: [DEFAULT_ENCODER],
          },
        ],
      }),
    );
    await openAdvanced(container);

    expect(enhanceCheckbox()).toBeTruthy();
    expect(encoderSelect()).toBeTruthy();
    expect([...encoderSelect().options].map((option) => option.value)).toEqual(["default"]);
    expect(container.textContent).toContain(
      "Complete operator-staged alternates appear here after Models is refreshed.",
    );

    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await render(baseContext({ videoModels: [{ ...LTX, adapter: "ltx_video" }] }));
    await openAdvanced(container);
    expect(encoderSelect()).toBeNull();
  });

  it("derives a future model's default from capability metadata instead of a hard-coded id", async () => {
    const futureModel = {
      ...LTX,
      id: "future_video",
      adapter: "future_video",
      textEncoderOptions: [
        {
          id: "future_alternate",
          label: "Future alternate",
          description: "A staged future alternate.",
          isDefault: false,
        },
        {
          id: "future_bundled",
          label: "Future bundled encoder",
          description: "The future model's bundled encoder.",
          isDefault: true,
        },
      ],
    };
    const context = baseContext({ videoModels: [futureModel] });
    await render(context);
    await openAdvanced(container);

    expect(encoderSelect().value).toBe("future_bundled");
    await click(enhanceCheckbox());
    await act(async () => setSelect(encoderSelect(), "future_alternate"));
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced.textEncoderModel).toBe(
      "future_alternate",
    );
  });

  it("does not carry one model's encoder id into a model with a disjoint option namespace", async () => {
    const futureModel = {
      ...LTX,
      id: "future_video",
      name: "Future Video",
      adapter: "future_video",
      textEncoderOptions: [
        {
          id: "future_bundled",
          label: "Future bundled encoder",
          description: "The future model's bundled encoder.",
          isDefault: true,
        },
      ],
    };
    await render(baseContext({ videoModels: [LTX_WITH_ENCODERS, futureModel] }));
    await openAdvanced(container);
    await click(enhanceCheckbox());
    await act(async () => setSelect(encoderSelect(), AMORAL_ENCODER.id));

    const modelSelect = container.querySelector(".settings-field-model select");
    await act(async () => setSelect(modelSelect, futureModel.id));
    expect(encoderSelect().value).toBe("future_bundled");
    expect(container.textContent).not.toContain("Previously selected encoder (not staged)");
  });

  it("applies and clears a preset text-encoder default through functional setter updates", async () => {
    await render(
      baseContext({
        videoModels: [LTX_WITH_ENCODERS],
        presets: [
          {
            id: "amoral_prompt",
            name: "Amoral prompt",
            scope: "project",
            workflow: "text_to_video",
            model: "ltx_2_3",
            modes: ["text_to_video"],
            defaults: {
              enhancePrompt: true,
              textEncoderModel: AMORAL_ENCODER.id,
            },
          },
        ],
      }),
    );

    await click(buttonWithText(container, "Amoral prompt"));
    await openAdvanced(container);
    expect(enhanceCheckbox().checked).toBe(true);
    expect(encoderSelect().value).toBe(AMORAL_ENCODER.id);

    const presetAxis = container.querySelector(".style-axis-presets");
    await click(buttonWithText(presetAxis, "None"));
    expect(enhanceCheckbox().checked).toBe(false);
    expect(encoderSelect().value).toBe(DEFAULT_ENCODER.id);
  });

  it("emits the exact non-default selector with prompt enhancement, never the legacy boolean", async () => {
    const context = baseContext({ videoModels: [LTX_WITH_ENCODERS] });
    await render(context);
    await openAdvanced(container);

    expect(encoderSelect().disabled).toBe(true);
    await click(enhanceCheckbox());
    expect(encoderSelect().disabled).toBe(false);
    await act(async () => setSelect(encoderSelect(), AMORAL_ENCODER.id));
    await act(async () => {});
    expect(
      JSON.parse(window.localStorage.getItem("sceneworks-studio-video-project_1")),
    ).toMatchObject({
      model: "ltx_2_3",
      enhancePrompt: true,
      textEncoderModel: AMORAL_ENCODER.id,
    });
    await click(buttonWithText(container, "Render clip"));

    const advanced = context.createVideoJob.mock.calls[0][0].advanced;
    expect(advanced).toMatchObject({
      enhancePrompt: true,
      textEncoderModel: "ltx_amoral_gemma_3_12b",
    });
    expect(advanced.useUncensoredEnhancer).toBeUndefined();
  });

  it("keeps the shipped encoder as the omission-based default", async () => {
    const context = baseContext({ videoModels: [LTX_WITH_ENCODERS] });
    await render(context);
    await openAdvanced(container);
    await click(enhanceCheckbox());
    await click(buttonWithText(container, "Render clip"));

    const advanced = context.createVideoJob.mock.calls[0][0].advanced;
    expect(advanced.enhancePrompt).toBe(true);
    expect(advanced.textEncoderModel).toBeUndefined();
  });

  it("round-trips the stable selector through a recorded recipe", async () => {
    const context = baseContext({
      videoModels: [LTX_WITH_ENCODERS],
      studioLaunch: {
        id: "replay-text-encoder",
        view: "Video",
        recipe: {
          mode: "text_to_video",
          model: "ltx_2_3",
          prompt: "A camera glides through a forest",
          normalizedSettings: {},
          rawAdapterSettings: {
            enhancePrompt: true,
            textEncoderModel: AMORAL_ENCODER.id,
          },
        },
      },
    });
    await render(context);
    await openAdvanced(container);

    expect(enhanceCheckbox().checked).toBe(true);
    expect(encoderSelect().value).toBe(AMORAL_ENCODER.id);
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced).toMatchObject({
      enhancePrompt: true,
      textEncoderModel: AMORAL_ENCODER.id,
    });
  });

  it("rehydrates a legacy boolean recipe into the stable selector", async () => {
    const context = baseContext({
      videoModels: [LTX_WITH_ENCODERS],
      studioLaunch: {
        id: "replay-legacy-text-encoder",
        view: "Video",
        recipe: {
          mode: "text_to_video",
          model: "ltx_2_3",
          prompt: "A slow aerial reveal",
          normalizedSettings: {},
          rawAdapterSettings: {
            enhancePrompt: true,
            useUncensoredEnhancer: true,
          },
        },
      },
    });
    await render(context);
    await openAdvanced(container);

    expect(encoderSelect().value).toBe(AMORAL_ENCODER.id);
    await click(buttonWithText(container, "Render clip"));
    const advanced = context.createVideoJob.mock.calls[0][0].advanced;
    expect(advanced.textEncoderModel).toBe(AMORAL_ENCODER.id);
    expect(advanced.useUncensoredEnhancer).toBeUndefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Krea Realtime 14B — the Video Studio surface (sc-8445, epic 8431 S12).
//
// The catalog entry (sc-8444) drives most of this screen generically, so these tests exist to
// PIN that the generic path actually lands the model's real numbers, and to guard the one place
// the generic path is not enough: the `video_to_video` clip-strength knob.
//
// Krea's three capabilities split three ways on `advanced.videoConditioningStrength`:
//   * text_to_video  — no conditioning media, no strength.
//   * image_to_video — the reference still only warms the AR KV cache; the engine reads the image
//                      and never `strength` (`krea_realtime_conditioning` emits `strength: None`).
//                      A documented NO-OP, so the control must stay hidden there.
//   * video_to_video — the source clip drives a strength-controlled AR init, so the worker DOES
//                      read the key (default 1.0). Before this story the control rendered only for
//                      extend_clip/video_bridge, so the knob was unreachable and every Krea v2v ran
//                      at 1.0 whatever the user wanted.
// ─────────────────────────────────────────────────────────────────────────────────────────────
describe("VideoStudio Krea Realtime 14B surface (sc-8445)", () => {
  let container;
  let root;

  const ALL_VIDEO_MODES = [
    "image_to_video",
    "text_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
    "video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    "multi_video_to_video",
    "ads2v",
    "animate_character",
  ];

  const KREA_CAPABILITIES = ["text_to_video", "image_to_video", "video_to_video"];

  // Transcribed from `config/manifests/builtin.models.jsonc`. The first test below holds every
  // field of this fixture to the SHIPPED entry, so the behavioural tests cannot pass against a
  // fixture that has drifted away from what a user's catalog actually serves.
  const KREA = {
    id: "krea_realtime_14b",
    name: "Krea Realtime 14B",
    type: "video",
    family: "krea-realtime",
    adapter: "krea_realtime",
    capabilities: KREA_CAPABILITIES,
    // Krea Realtime is CFG-free, so it declares BOTH axes absent (sc-8445). Every other video model
    // declares no `video` block at all, and absent means TRUE — see the supportsGuidance derivation.
    video: { supportsGuidance: false, supportsNegativePrompt: false },
    defaults: {
      duration: 4,
      fps: 24,
      resolution: "832x480",
      quality: "balanced",
      steps: 6,
      sampler: "default",
      scheduler: "default",
    },
    limits: {
      durations: [2, 3, 4, 5],
      recommendedMaxDuration: 5,
      hardMaxDuration: 5,
      fps: [24],
      requiresDimensionsMultipleOf: 16,
      resolutions: ["832x480", "480x832", "1280x720", "720x1280"],
      samplers: ["default"],
      schedulers: ["default"],
    },
    quantization: {},
    loraCompatibility: { families: ["krea-realtime"], types: ["style", "motion", "character", "enhance"] },
    ui: {},
    macSupport: {
      supported: true,
      features: {
        videoModes: Object.fromEntries(
          ALL_VIDEO_MODES.map((value) => [value, KREA_CAPABILITIES.includes(value)]),
        ),
      },
    },
  };

  // Bernini also serves `video_to_video`, but its worker route never reads
  // `videoConditioningStrength` — the discriminator that proves the new control is gated on the
  // ENGINE and not merely on the mode.
  const BERNINI_V2V = {
    id: "bernini",
    name: "Bernini",
    type: "video",
    family: "bernini",
    adapter: "bernini",
    capabilities: ["text_to_video", "video_to_video"],
    defaults: { duration: 5, resolution: "848x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["848x480", "480x848"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  // The Wan A14B engine that DOES carry the Lightning toggle — the discriminator for "Krea has no
  // lightning LoRA", so the assertion cannot pass just because the toggle never renders anywhere.
  const WAN_A14B = {
    id: "wan_2_2_t2v_14b",
    name: "Wan 2.2 T2V A14B",
    type: "video",
    family: "wan-video",
    adapter: "wan_video",
    capabilities: ["text_to_video"],
    defaults: { duration: 5, resolution: "832x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  const MAC_CAPS = {
    macGatingActive: true,
    platform: "darwin",
    notAvailableLabel: "Not available on Mac (MLX only)",
    features: {},
    training: { supportedKernels: [], lokrOnWanSupported: false },
  };

  const still = { id: "img_still", type: "image", projectId: "project_1", displayName: "Opening Still" };
  const clip = { id: "vid_src", type: "video", projectId: "project_1", displayName: "Source Clip" };

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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const modeButton = (label) => buttonWithText(container.querySelector(".mode-control"), label);
  const labelStartingWith = (text) =>
    [...container.querySelectorAll("label")].find((el) => el.textContent.trim().startsWith(text));
  const selectLabelled = (text) => labelStartingWith(text)?.querySelector("select");
  const optionValues = (select) => [...select.options].map((option) => option.value);
  const clipStrengthInput = () => labelStartingWith("Clip strength")?.querySelector("input");
  // The disclosure's open state survives a re-render into the same root, so a blind toggle would
  // CLOSE it on the second model of a two-model discriminator test. Open only when it is shut.
  const ensureAdvancedOpen = async () => {
    const toggle = container.querySelector(".advanced-section-toggle");
    if (toggle && toggle.getAttribute("aria-expanded") !== "true") {
      await click(toggle);
    }
  };

  it("keeps the fixture identical to the shipped krea_realtime_14b manifest entry", () => {
    const manifestPath = resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../../../config/manifests/builtin.models.jsonc",
    );
    const manifest = JSON5.parse(readFileSync(manifestPath, "utf8"));
    const models = Array.isArray(manifest) ? manifest : manifest.models;
    const shipped = models.find((entry) => entry.id === "krea_realtime_14b");

    expect(shipped, "krea_realtime_14b must be in builtin.models.jsonc").toBeTruthy();
    expect(shipped.adapter).toBe(KREA.adapter);
    expect(shipped.family).toBe(KREA.family);
    expect(shipped.type).toBe("video");
    expect(shipped.capabilities).toEqual(KREA.capabilities);
    expect(shipped.defaults).toEqual(KREA.defaults);
    expect(shipped.limits).toEqual(KREA.limits);
    expect(shipped.video).toEqual(KREA.video);
    // sc-15017: the LoRA tests below ride this block, so hold it to the shipped entry too —
    // otherwise they could pass against a fixture that had drifted away from the real catalog.
    expect(shipped.loraCompatibility).toEqual(KREA.loraCompatibility);
  });

  // sc-15017 — Wan-family LoRA selection for Krea Realtime. No bespoke krea control: this is the
  // generic `generationStudio` picker, and the only thing that lights it up is family
  // compatibility. Krea declares its OWN `krea-realtime` family, so a Wan LoRA is offered only
  // because `loraMatchesModel` mirrors the backend's one-directional extra-compatible registry.
  describe("Wan-family LoRA selection", () => {
    const ORIGAMI = {
      id: "origami_wan",
      name: "Origami Wan Style",
      family: "wan-video",
      scope: "global",
      installState: "installed",
    };
    const KREA_NATIVE_LORA = {
      id: "krea_native",
      name: "Krea Native Style",
      family: "krea-realtime",
      scope: "global",
      installState: "installed",
    };
    const loraPicker = () => container.querySelector(".lora-picker");
    const openPicker = async () => {
      const add = loraPicker().querySelector(".lora-add");
      await act(async () => add.click());
    };

    it("offers a Wan-family LoRA when Krea Realtime is the selected model", async () => {
      await render(
        baseContext({
          videoModels: [KREA],
          macCapabilities: MAC_CAPS,
          loras: [ORIGAMI, KREA_NATIVE_LORA],
        }),
      );
      await openPicker();
      expect(loraPicker().textContent).toContain("Origami Wan Style");
      // Its own family is still offered — the extra is additive, not a swap.
      expect(loraPicker().textContent).toContain("Krea Native Style");
    });

    it("does NOT offer a Krea-Realtime LoRA on a Wan model (the registry is one-directional)", async () => {
      await render(
        baseContext({
          videoModels: [{ ...WAN_A14B, loraCompatibility: { families: ["wan-video"] } }],
          macCapabilities: MAC_CAPS,
          loras: [ORIGAMI, KREA_NATIVE_LORA],
        }),
      );
      await openPicker();
      // The control: the Wan model DOES offer its own family, so the absence below is the
      // direction of the relation and not an empty / broken picker.
      expect(loraPicker().textContent).toContain("Origami Wan Style");
      expect(loraPicker().textContent).not.toContain("Krea Native Style");
    });
  });

  it("offers Krea Realtime in the picker with exactly its three capability tabs enabled", async () => {
    await render(baseContext({ videoModels: [KREA], macCapabilities: MAC_CAPS }));

    const picker = selectLabelled("Model");
    expect(optionValues(picker)).toEqual(["krea_realtime_14b"]);
    expect(picker.value).toBe("krea_realtime_14b");

    // Mode tabs are gated at the MODE level under Mac gating, so a tab is enabled exactly when
    // some available model serves it. With Krea the only model, that is exactly its capabilities.
    const enabled = ALL_VIDEO_MODES.filter((value) => {
      const label = {
        text_to_video: "Text → Video",
        image_to_video: "Image → Video",
        first_last_frame: "First → Last",
        extend_clip: "Extend",
        video_bridge: "Bridge",
        replace_person: "Replace person",
        video_to_video: "Video → Video",
        reference_to_video: "Reference → Video",
        reference_video_to_video: "Reference + Video",
        multi_video_to_video: "Multi-Clip → Video",
        ads2v: "Clip + Ref Video",
        animate_character: "Animate character",
      }[value];
      return !modeButton(label).disabled;
    });
    expect(enabled.sort()).toEqual([...KREA_CAPABILITIES].sort());
  });

  it("surfaces the manifest's own duration / fps / resolution / steps, not the screen's fallbacks", async () => {
    await render(baseContext({ videoModels: [KREA], macCapabilities: MAC_CAPS }));
    await ensureAdvancedOpen();

    const durations = selectLabelled("Duration");
    // `limits.durations[0]` is 2 and the DEFAULT is 4 — asserting the selected value is 4 is
    // therefore a real discriminator, not the "first entry is the default" tautology.
    expect(optionValues(durations)).toEqual(["2", "3", "4", "5"]);
    expect(durations.value).toBe("4");
    expect(durations.value).not.toBe(String(KREA.limits.durations[0]));
    // The screen falls back to [4, 6, 8, 10] when a model declares no durations; Krea's lattice
    // shares none of its endpoints, so a broken wiring shows up here rather than passing silently.
    expect(optionValues(durations)).not.toEqual(["4", "6", "8", "10"]);

    const fps = selectLabelled("Frames");
    // The screen's fallback fps menu is [24, 25, 30] and Krea's is the single 24 — same trap:
    // the VALUE 24 alone would pass under the fallback, the one-entry MENU would not.
    expect(optionValues(fps)).toEqual(["24"]);
    expect(fps.value).toBe("24");

    const resolutions = selectLabelled("Resolution");
    expect(optionValues(resolutions)).toEqual(["832x480", "480x832", "1280x720", "720x1280"]);
    expect(resolutions.value).toBe("832x480");

    // The Advanced steps control reflects the model's own few-step recipe as its placeholder.
    const steps = labelStartingWith("Steps").querySelector("input");
    expect(steps.placeholder).toBe("6");
    expect(steps.disabled).toBe(false);
  });

  it("shows no Lightning toggle for Krea while still showing it for a Wan A14B engine", async () => {
    await render(baseContext({ videoModels: [KREA], macCapabilities: MAC_CAPS }));
    await ensureAdvancedOpen();
    expect(container.textContent).not.toContain("Lightning");

    // Same screen, same disclosure, a model that DOES carry the toggle — so the assertion above
    // cannot be passing merely because the control was removed or renamed.
    await render(baseContext({ videoModels: [WAN_A14B] }));
    await ensureAdvancedOpen();
    expect(container.textContent).toContain("Lightning");
  });

  it("offers the clip strength for Krea video_to_video and sends it", async () => {
    const context = baseContext({
      videoModels: [KREA],
      macCapabilities: MAC_CAPS,
      assets: [clip],
      selectedAsset: clip,
    });
    await render(context);
    await click(modeButton("Video → Video"));
    await ensureAdvancedOpen();

    const strength = clipStrengthInput();
    expect(strength, "Krea v2v must expose the clip strength the worker reads").toBeTruthy();
    await act(async () => setInput(strength, "0.65"));
    await click(buttonWithText(container, "Render clip"));

    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({ mode: "video_to_video", model: "krea_realtime_14b", sourceClipAssetId: "vid_src" });
    expect(payload.advanced.videoConditioningStrength).toBe(0.65);
  });

  it("hides the clip strength on a v2v model whose worker route ignores it", async () => {
    const context = baseContext({ videoModels: [BERNINI_V2V], assets: [clip], selectedAsset: clip });
    await render(context);
    await click(modeButton("Video → Video"));
    await ensureAdvancedOpen();

    // Same mode, different engine: Bernini's v2v never reads `videoConditioningStrength`, so the
    // control must not appear — the gate is the adapter, not the tab.
    expect(clipStrengthInput()).toBeFalsy();

    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].advanced.videoConditioningStrength).toBeUndefined();
  });

  it("hides the clip strength for Krea image_to_video, whose strength is a documented no-op", async () => {
    const context = baseContext({
      videoModels: [KREA],
      macCapabilities: MAC_CAPS,
      assets: [still],
      selectedAsset: still,
    });
    await render(context);
    await click(modeButton("Image → Video"));
    await ensureAdvancedOpen();

    // The i2v reference still only warms the AR KV cache — the engine reads the image and never
    // `strength`. Offering the knob here would be a control that silently does nothing.
    expect(clipStrengthInput()).toBeFalsy();

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({ mode: "image_to_video", model: "krea_realtime_14b", sourceAssetId: "img_still" });
    expect(payload.advanced.videoConditioningStrength).toBeUndefined();
  });

  // acceptance item 1's install-state half. `videoModels` is what App.jsx has ALREADY filtered
  // through `generationModelsForType` (installState complete), so a krea that is still `missing`
  // reaches this screen as an empty picker list plus a catalog entry to offer — exactly the shape
  // reproduced here. Without this the only evidence for the krea-specific gate was a browser
  // session nobody can re-run.
  it("gates the studio and offers Krea for download when it is not installed", async () => {
    const missingKrea = { ...KREA, installState: "missing", downloadSizeLabel: "18.9 GB" };
    const context = baseContext({
      videoModels: [],
      models: [missingKrea],
      macCapabilities: MAC_CAPS,
    });
    await render(context);

    const gate = container.querySelector(".model-availability-gate");
    expect(gate, "an empty video picker must render the availability gate").toBeTruthy();
    expect(gate.textContent).toContain("Video Studio needs a video model");
    // The studio body is replaced, not merely annotated.
    expect(container.querySelector(".mode-control")).toBeFalsy();

    // Krea is offered specifically — `videoModelUsable` accepts it (video type, Mac-supported,
    // serves >=1 mode), so the gate names it rather than falling back to "nothing supports this".
    const offers = [...gate.querySelectorAll(".model-availability-offer")].map((el) => el.textContent);
    expect(offers.length).toBe(1);
    expect(offers[0]).toContain("Krea Realtime 14B");
    expect(offers[0]).toContain("18.9 GB");
  });

  // sc-8445 major: Krea is CFG-free — `generate_krea_realtime` sets `negative_prompt: None` and
  // never forwards a guidance scale — so offering either control would be a knob that silently does
  // nothing. Driven off the manifest `video` block, not an id check.
  //
  // ⚠️ The payload half is asserted after the values have actually been SET on another model and
  // carried across the model switch. Both fields persist in studio state (and restore from a
  // recipe), so a user who sets guidance on Bernini and switches to Krea really does arrive here
  // with a non-empty override — which is the leak the gate has to plug. Asserting on a freshly
  // mounted Krea would assert the empty default and pass against no implementation at all.
  it("hides Guidance and Negative prompt for a CFG-free engine and sends neither", async () => {
    const context = baseContext({ videoModels: [BERNINI_V2V, KREA] });
    await render(context);
    await ensureAdvancedOpen();

    // On the guidance-taking model first: set both, so the state is genuinely non-empty.
    await act(async () => setInput(labelStartingWith("Guidance").querySelector("input"), "7.5"));
    const negative = labelStartingWith("Negative prompt").querySelector("textarea");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      setter.call(negative, "blurry, low quality");
      negative.dispatchEvent(new window.Event("input", { bubbles: true }));
    });

    // Switch to the CFG-free engine. Both controls go away…
    await act(async () => setSelect(selectLabelled("Model"), "krea_realtime_14b"));
    await ensureAdvancedOpen();
    expect(labelStartingWith("Guidance")).toBeFalsy();
    expect(labelStartingWith("Negative prompt")).toBeFalsy();

    // …and neither retained value reaches the worker.
    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.model).toBe("krea_realtime_14b");
    expect(payload.negativePrompt).toBe("");
    expect(payload.advanced.guidanceScale).toBeUndefined();
  });

  // The other half of the same rule, and the reason `!== false` rather than `=== true`: a video
  // model that declares NO `video` block keeps both controls, so this change is behaviour-neutral
  // for Wan / LTX / SCAIL-2 / Bernini. Without this leg the test above would also pass against an
  // implementation that hid the pair for every model.
  //
  // Bernini rather than the Wan A14B fixture on purpose: `wan_2_2_t2v_14b` is in
  // WAN_A14B_LIGHTNING_MODEL_IDS, and Lightning defaults ON and suppresses guidance itself — so it
  // would have proved nothing about the new `video` gate.
  it("keeps Guidance and Negative prompt for a model that declares no video axes", async () => {
    const context = baseContext({ videoModels: [BERNINI_V2V] });
    await render(context);
    await ensureAdvancedOpen();

    const guidance = labelStartingWith("Guidance")?.querySelector("input");
    expect(guidance).toBeTruthy();
    await act(async () => setInput(guidance, "6.5"));
    const negative = labelStartingWith("Negative prompt")?.querySelector("textarea");
    expect(negative).toBeTruthy();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      setter.call(negative, "blurry");
      negative.dispatchEvent(new window.Event("input", { bubbles: true }));
    });

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.negativePrompt).toBe("blurry");
    expect(payload.advanced.guidanceScale).toBe(6.5);
  });

  // SVD is the OTHER video engine with no guidance / negative axis: `generate_svd`
  // (video_jobs/svd.rs) hard-sets `negative_prompt: None` and `guidance: None` because it drives
  // its own frame-wise CFG ramp. `promptless` already hid its prompt field; these two controls
  // stayed visible and permanently dead until sc-8445. Same `video` block, no new machinery.
  const SVD = {
    id: "svd",
    name: "Stable Video Diffusion",
    type: "video",
    family: "svd",
    adapter: "svd_video",
    capabilities: ["image_to_video"],
    promptless: true,
    video: { supportsGuidance: false, supportsNegativePrompt: false },
    defaults: { duration: 4, resolution: "1024x576", fps: 7 },
    limits: {},
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  it("hides both axes for SVD too, and the manifest says so", async () => {
    const manifestPath = resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../../../config/manifests/builtin.models.jsonc",
    );
    const manifest = JSON5.parse(readFileSync(manifestPath, "utf8"));
    const models = Array.isArray(manifest) ? manifest : manifest.models;

    // The fixture is honest about the shipped entry…
    const shippedSvd = models.find((entry) => entry.id === "svd");
    expect(shippedSvd?.video).toEqual(SVD.video);

    // …and the polarity claim the manifest/schema comments make is checked against the real
    // population rather than trusted: exactly these entries declare the block, every other video
    // model declares nothing and therefore keeps both controls.
    //
    // Both MiniMax-H3 partitions joined in sc-17158: the sc-17242 spike enumerated the reference
    // pipeline's complete 19-parameter input surface and neither `guidance_scale` nor
    // `negative_prompt` is on it, so the axes genuinely do not exist. The non-declaring count is
    // deliberately asserted separately. LTX-2.5 declares both absent for its default distilled
    // surface; Video Studio reopens only negative conditioning when dev is explicitly selected.
    const videoEntries = models.filter((entry) => entry.type === "video");
    const declaring = videoEntries.filter((entry) => entry.video).map((entry) => entry.id).sort();
    expect(declaring).toEqual(["krea_realtime_14b", "ltx_2_5", "minimax_h3", "minimax_h3_ref", "svd"]);
    expect(videoEntries.length - declaring.length).toBe(8);

    const context = baseContext({ videoModels: [SVD] });
    await render(context);
    await ensureAdvancedOpen();
    expect(labelStartingWith("Guidance")).toBeFalsy();
    expect(labelStartingWith("Negative prompt")).toBeFalsy();
  });

  // ⚠️ Regression guard for a bug this PR introduced. `PresetStackPreview` is captioned
  // "Prompt sent" and exists so the user sees exactly what will be generated. Once submit started
  // sending `negativePrompt: ""` for a CFG-free engine, the preview's `Negative:` line became a
  // claim about something that is NOT sent. General presets are filtered on `kind === "general"`
  // alone, so a preset carrying `defaults.negativePrompt` reaches Krea.
  it("drops the stack preview's Negative line on a CFG-free engine but keeps it otherwise", async () => {
    const stackPreset = {
      id: "film_look",
      name: "Film Look",
      kind: "general",
      prompt: { suffix: "Kodak Portra 400" },
      defaults: { negativePrompt: "blurry, low quality" },
    };
    const context = baseContext({ videoModels: [BERNINI_V2V, KREA], presets: [stackPreset] });
    await render(context);

    const chip = [...container.querySelectorAll(".general-preset-chips .preset-chip")].find(
      (el) => el.textContent.trim() === "Film Look",
    );
    expect(chip, "the general preset must be offered").toBeTruthy();
    await click(chip);

    // Bernini takes a negative prompt, so the preview states it AND the run sends it.
    let preview = container.querySelector(".preset-stack-preview");
    expect(preview.textContent).toContain("Kodak Portra 400");
    expect(preview.textContent).toContain("Negative: blurry, low quality");
    await click(buttonWithText(container, "Render clip"));
    expect(context.createVideoJob.mock.calls[0][0].negativePrompt).toBe("blurry, low quality");

    // Same stack, CFG-free engine: the composed PROMPT is still shown (it is still sent), but the
    // Negative line is gone — because the payload no longer carries it.
    await act(async () => setSelect(selectLabelled("Model"), "krea_realtime_14b"));
    preview = container.querySelector(".preset-stack-preview");
    expect(preview.textContent).toContain("Kodak Portra 400");
    expect(preview.textContent).not.toContain("Negative:");

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[1][0];
    expect(payload.model).toBe("krea_realtime_14b");
    expect(payload.negativePrompt).toBe("");
  });

  it("submits Krea in each of its three advertised modes", async () => {
    const context = baseContext({
      videoModels: [KREA],
      macCapabilities: MAC_CAPS,
      assets: [still, clip],
    });
    await render(context);

    // text_to_video — the default tab, no conditioning media.
    await click(buttonWithText(container, "Render clip"));

    await click(modeButton("Image → Video"));
    await click(buttonWithText(container, "Select image"));
    let modal = document.querySelector(".asset-picker-modal");
    await click([...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Opening Still")));
    await click(buttonWithText(modal, "Use Selection"));
    await click(buttonWithText(container, "Render clip"));

    await click(modeButton("Video → Video"));
    await click(buttonWithText(container, "Select clip"));
    modal = document.querySelector(".asset-picker-modal");
    await click([...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Source Clip")));
    await click(buttonWithText(modal, "Use Selection"));
    await click(buttonWithText(container, "Render clip"));

    const modes = context.createVideoJob.mock.calls.map(([payload]) => payload.mode);
    expect(modes).toEqual(["text_to_video", "image_to_video", "video_to_video"]);
    for (const [payload] of context.createVideoJob.mock.calls) {
      expect(payload.model).toBe("krea_realtime_14b");
    }
  });
});
