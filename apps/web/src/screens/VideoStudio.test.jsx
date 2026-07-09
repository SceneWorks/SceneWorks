import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

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

const saveButton = (container) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Save as Preset"));
const nameInput = (container) => container.querySelector('input[aria-label="Preset name"]');
// Save-as-Preset folds into the Advanced disclosure (collapsed by default), matching
// Image Studio — open it before touching the preset controls.
const openAdvanced = async (container) => {
  const toggle = container.querySelector(".advanced-section-toggle");
  if (toggle) {
    await click(toggle);
  }
};

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

  it("snapshots the video config into a text_to_video preset without the seed", async () => {
    const context = baseContext();
    await render(context);
    await openAdvanced(container);

    const input = nameInput(container);
    expect(input).toBeTruthy();
    await act(async () => setInput(input, "Push In"));
    await click(saveButton(container));

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
    await openAdvanced(container);

    await act(async () => setInput(nameInput(container), "Push In"));
    await click(saveButton(container));

    expect(context.createPreset).not.toHaveBeenCalled();
    expect(container.textContent).toContain("already exists");
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

  it("does not emit advanced.lightning for a non-Wan engine", async () => {
    const context = baseContext({ videoModels: [NON_WAN] });
    await render(context);

    await click(buttonWithText(container, "Render clip"));
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.advanced.lightning).toBeUndefined();
  });
});
