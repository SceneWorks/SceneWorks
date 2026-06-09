import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

async function doubleClick(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  });
}

const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === text);

function setInput(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

const saveButton = (container) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Save as Preset"));
const nameInput = (container) => container.querySelector('input[aria-label="Preset name"]');

describe("VideoStudio Save as Preset", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
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

  it("snapshots the video config into an image_to_video preset without the seed", async () => {
    const context = baseContext();
    await render(context);

    const input = nameInput(container);
    expect(input).toBeTruthy();
    await act(async () => setInput(input, "Push In"));
    await click(saveButton(container));

    expect(context.createPreset).toHaveBeenCalledTimes(1);
    const payload = context.createPreset.mock.calls[0][0];
    expect(payload).toMatchObject({
      id: "push_in",
      name: "Push In",
      scope: "project",
      workflow: "image_to_video",
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
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
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
