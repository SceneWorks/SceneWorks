import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

// Pose loaders + any best-effort fetches must never touch the network on mount.
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async () => ({})),
  };
});

import { AppContext } from "../context/AppContext.js";
import { ImageStudio } from "./ImageStudio.jsx";
import { VideoStudio } from "./VideoStudio.jsx";

// sc-11962 — S3: restart-restore must not clobber a restored studio snapshot when the
// model / LoRA / preset catalogs resolve asynchronously AFTER the studio has mounted.
//
// The failure mode: on the first mount after an app restart the studio restores its
// settings from localStorage, then empty catalogs (still loading) make the model
// "snap" to a fallback and the preset / general-stack / LoRA prune + the sampler /
// resolution snaps fire against the empty catalog — silently reverting the restored
// values to model/preset defaults. These tests seed a snapshot, mount with EMPTY
// catalogs, then feed the real catalogs in AFTER mount (in several orderings) and
// assert every restored value survives.

// ---- Image fixtures ----------------------------------------------------------------
const Z_IMAGE = {
  id: "z_image_turbo",
  name: "Z Image Turbo",
  type: "image",
  family: "z-image",
  capabilities: ["text_to_image"],
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024"], samplers: ["default"], schedulers: ["default"] },
  loraCompatibility: {},
  ui: {},
};
const FLUX = {
  id: "flux2_dev",
  name: "FLUX.2 Dev",
  type: "image",
  family: "flux2",
  capabilities: ["text_to_image"],
  defaults: { resolution: "1024x1024", sampler: "euler", scheduler: "shift", steps: 20, guidanceScale: 3.5 },
  limits: {
    resolutions: ["1024x1024", "1536x1536"],
    samplers: ["default", "euler", "dpmpp_2m"],
    schedulers: ["default", "shift", "karras"],
  },
  loraCompatibility: {},
  ui: {},
};
const IMAGE_LORA = { id: "lora-a", name: "Lora A", family: "flux2", scope: "global", installState: "installed" };
const IMAGE_BASE_PRESET = {
  id: "cine",
  name: "Cinematic",
  kind: "model",
  scope: "global",
  workflow: "text_to_image",
  modes: ["text_to_image"],
  model: "flux2_dev",
  loras: [],
  defaults: {},
};
const IMAGE_GENERAL_PRESET = {
  id: "enhance",
  name: "Enhance",
  kind: "general",
  scope: "global",
  defaults: { prompt: { append: "cinematic lighting" } },
};

// The full snapshot a user leaves behind: a non-default model (NOT the catalog's first
// entry, so a fallback snap is visibly wrong), a resolution + sampler + scheduler that
// only the restored model declares, and a restored LoRA + base preset + general stack.
const IMAGE_SNAPSHOT = {
  mode: "text_to_image",
  prompt: "a restored studio prompt",
  model: "flux2_dev",
  resolution: "1536x1536",
  sampler: "dpmpp_2m",
  scheduler: "karras",
  steps: 24,
  guidanceScale: 5,
  ipAdapterScale: 0.85,
  selectedLoraIds: ["lora-a"],
  loraWeights: { "lora-a": 0.9 },
  selectedPresetId: "cine",
  generalStackIds: ["enhance"],
  advancedOpen: false,
};

function imageContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createImageJob: vi.fn(),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    magicPrompt: vi.fn(),
    imageCaption: vi.fn(),
    imageDescribe: vi.fn(),
    createModelDownloadJob: vi.fn(),
    createLoraDownloadJob: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [],
    models: [],
    jobs: [],
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
    promptBatches: [],
    createPromptBatch: vi.fn(),
    updatePromptBatch: vi.fn(),
    deletePromptBatch: vi.fn(),
    requestedGpu: "",
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    visibleWorkers: [],
    ...overrides,
  };
}

const IMAGE_FULL = {
  imageModels: [Z_IMAGE, FLUX],
  models: [Z_IMAGE, FLUX],
  loras: [IMAGE_LORA],
  presets: [IMAGE_BASE_PRESET, IMAGE_GENERAL_PRESET],
};

// ---- Video fixtures ----------------------------------------------------------------
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
const WAN = {
  id: "wan_t2v",
  name: "Wan T2V",
  type: "video",
  family: "wan",
  capabilities: ["text_to_video", "image_to_video"],
  defaults: { duration: 6, resolution: "1280x720", fps: 24 },
  limits: {
    resolutions: ["768x512", "1280x720"],
    samplers: ["default", "unipc", "dpmpp_2m"],
    schedulers: ["default", "karras"],
  },
  quantization: {},
  loraCompatibility: {},
  ui: {},
};
const VIDEO_LORA = { id: "vlora-a", name: "Vid Lora A", family: "wan", scope: "global", installState: "installed" };
const VIDEO_BASE_PRESET = {
  id: "vcine",
  name: "Video Cinematic",
  kind: "model",
  scope: "global",
  workflow: "text_to_video",
  modes: ["text_to_video"],
  model: "wan_t2v",
  loras: [],
  defaults: {},
};
const VIDEO_GENERAL_PRESET = {
  id: "venhance",
  name: "Video Enhance",
  kind: "general",
  scope: "global",
  defaults: { prompt: { append: "cinematic" } },
};

const VIDEO_SNAPSHOT = {
  mode: "text_to_video",
  prompt: "a restored video prompt",
  model: "wan_t2v",
  resolution: "1280x720",
  sampler: "unipc",
  scheduler: "karras",
  selectedLoraIds: ["vlora-a"],
  loraWeights: { "vlora-a": 0.9 },
  selectedPresetId: "vcine",
  generalStackIds: ["venhance"],
  advancedOpen: false,
};

function videoContext(overrides = {}) {
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
    videoModels: [],
    models: [],
    ...overrides,
  };
}

const VIDEO_FULL = {
  videoModels: [LTX, WAN],
  models: [LTX, WAN],
  loras: [VIDEO_LORA],
  presets: [VIDEO_BASE_PRESET, VIDEO_GENERAL_PRESET],
};

// ---- shared helpers ----------------------------------------------------------------
function seedSnapshot(studio, snapshot) {
  window.localStorage.setItem(`sceneworks-studio-${studio}-project_1`, JSON.stringify(snapshot));
}
function readSnapshot(studio) {
  return JSON.parse(window.localStorage.getItem(`sceneworks-studio-${studio}-project_1`) || "{}");
}
const modelSelectValue = () => document.body.querySelector(".settings-field-model select")?.value;
const aspectSelectValue = () => document.body.querySelector(".settings-field-aspect select")?.value;
const activeBasePresetName = () =>
  document.body.querySelector(".preset-chips:not(.general-preset-chips) button.preset-chip.active")?.textContent?.trim();
const activeGeneralNames = () =>
  [...document.body.querySelectorAll(".general-preset-chips button.preset-chip.active")].map((n) => n.textContent.trim());
const loraSlotNames = () =>
  [...document.body.querySelectorAll(".lora-slot .lora-slot-meta strong")].map((n) => n.textContent.trim());

describe("studio restart-restore does not clobber the restored snapshot (sc-11962)", () => {
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

  // Render a sequence of context snapshots into the SAME root, flushing effects
  // between each — this is how "catalogs resolve after mount" is simulated.
  async function renderSequence(Studio, contexts) {
    for (const ctx of contexts) {
      await act(async () => {
        root.render(
          <AppContext.Provider value={ctx}>
            <Studio />
          </AppContext.Provider>,
        );
      });
      await act(async () => {});
    }
  }

  // ---- Image Studio ----------------------------------------------------------------
  it("Image: restores model + resolution + sampler + preset + stack + LoRA when catalogs arrive after mount", async () => {
    seedSnapshot("image", IMAGE_SNAPSHOT);
    await renderSequence(ImageStudio, [imageContext(), imageContext(IMAGE_FULL)]);

    // Open Advanced so the LoRA slot renders.
    await click(document.body.querySelector(".advanced-section-toggle"));

    expect(modelSelectValue()).toBe("flux2_dev");
    expect(aspectSelectValue()).toBe("1536x1536");
    expect(activeBasePresetName()).toBe("Cinematic");
    expect(activeGeneralNames()).toEqual(["Enhance"]);
    expect(loraSlotNames()).toContain("Lora A");

    const snap = readSnapshot("image");
    expect(snap.model).toBe("flux2_dev");
    expect(snap.resolution).toBe("1536x1536");
    expect(snap.sampler).toBe("dpmpp_2m");
    expect(snap.scheduler).toBe("karras");
    expect(snap.steps).toBe(24);
    expect(snap.guidanceScale).toBe(5);
    expect(snap.ipAdapterScale).toBe(0.85);
    expect(snap.selectedLoraIds).toEqual(["lora-a"]);
    expect(snap.selectedPresetId).toBe("cine");
    expect(snap.generalStackIds).toEqual(["enhance"]);
  });

  it("Image: survives a LoRA-first / models+presets-later load ordering", async () => {
    seedSnapshot("image", IMAGE_SNAPSHOT);
    await renderSequence(ImageStudio, [
      imageContext(),
      imageContext({ loras: [IMAGE_LORA] }), // loras resolve first, models/presets still empty
      imageContext(IMAGE_FULL), // then everything else
    ]);

    expect(modelSelectValue()).toBe("flux2_dev");
    expect(aspectSelectValue()).toBe("1536x1536");
    expect(activeBasePresetName()).toBe("Cinematic");
    expect(activeGeneralNames()).toEqual(["Enhance"]);

    const snap = readSnapshot("image");
    expect(snap.model).toBe("flux2_dev");
    expect(snap.resolution).toBe("1536x1536");
    expect(snap.sampler).toBe("dpmpp_2m");
    expect(snap.selectedLoraIds).toEqual(["lora-a"]);
    expect(snap.selectedPresetId).toBe("cine");
    expect(snap.generalStackIds).toEqual(["enhance"]);
  });

  it("Image: survives a models-first / loras+presets-later load ordering", async () => {
    seedSnapshot("image", IMAGE_SNAPSHOT);
    await renderSequence(ImageStudio, [
      imageContext(),
      imageContext({ imageModels: [Z_IMAGE, FLUX], models: [Z_IMAGE, FLUX] }), // models first
      imageContext(IMAGE_FULL),
    ]);

    expect(modelSelectValue()).toBe("flux2_dev");
    expect(aspectSelectValue()).toBe("1536x1536");

    const snap = readSnapshot("image");
    expect(snap.model).toBe("flux2_dev");
    expect(snap.resolution).toBe("1536x1536");
    expect(snap.sampler).toBe("dpmpp_2m");
    expect(snap.selectedLoraIds).toEqual(["lora-a"]);
    expect(snap.selectedPresetId).toBe("cine");
    expect(snap.generalStackIds).toEqual(["enhance"]);
  });

  // ---- Video Studio ----------------------------------------------------------------
  it("Video: restores model + resolution + sampler + preset + stack + LoRA when catalogs arrive after mount", async () => {
    seedSnapshot("video", VIDEO_SNAPSHOT);
    await renderSequence(VideoStudio, [videoContext(), videoContext(VIDEO_FULL)]);

    await click(document.body.querySelector(".advanced-section-toggle"));

    expect(modelSelectValue()).toBe("wan_t2v");
    expect(aspectSelectValue()).toBe("1280x720");
    expect(activeBasePresetName()).toBe("Video Cinematic");
    expect(activeGeneralNames()).toEqual(["Video Enhance"]);
    expect(loraSlotNames()).toContain("Vid Lora A");

    const snap = readSnapshot("video");
    expect(snap.model).toBe("wan_t2v");
    expect(snap.resolution).toBe("1280x720");
    expect(snap.sampler).toBe("unipc");
    expect(snap.scheduler).toBe("karras");
    expect(snap.selectedLoraIds).toEqual(["vlora-a"]);
    expect(snap.selectedPresetId).toBe("vcine");
    expect(snap.generalStackIds).toEqual(["venhance"]);
  });

  it("Video: survives a LoRA-first / models+presets-later load ordering", async () => {
    seedSnapshot("video", VIDEO_SNAPSHOT);
    await renderSequence(VideoStudio, [
      videoContext(),
      videoContext({ loras: [VIDEO_LORA] }),
      videoContext(VIDEO_FULL),
    ]);

    expect(modelSelectValue()).toBe("wan_t2v");
    expect(aspectSelectValue()).toBe("1280x720");

    const snap = readSnapshot("video");
    expect(snap.model).toBe("wan_t2v");
    expect(snap.resolution).toBe("1280x720");
    expect(snap.sampler).toBe("unipc");
    expect(snap.selectedLoraIds).toEqual(["vlora-a"]);
    expect(snap.selectedPresetId).toBe("vcine");
    expect(snap.generalStackIds).toEqual(["venhance"]);
  });
});
