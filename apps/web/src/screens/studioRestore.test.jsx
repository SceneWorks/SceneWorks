import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, unmountRoot, setSelect } from "../testUtils/dom.js";

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

  // ---- The OTHER side of the sc-11962 fix (S4 / sc-11963) --------------------------
  // S3 made the reference-tuning + advanced-defaults resets fire only on a genuine model
  // VALUE change, not on an async catalog "snap" that leaves the model unchanged — the
  // tests above lock that async loads DON'T reset. This locks the complementary guarantee:
  // once the catalogs are LOADED (restore already settled, ready-gated writer live), a real
  // USER model change STILL resets both the reference tuning and the advanced defaults to the
  // newly-chosen model's declared values. A one-shot skip bug that silenced the async snap
  // would also (wrongly) swallow this real change, so the two halves must be tested together.
  it("Image: a genuine post-load user model change STILL resets reference tuning + advanced defaults", async () => {
    seedSnapshot("image", IMAGE_SNAPSHOT);
    // Catalogs resolve after mount → the restore settles with no reset (the S3 guarantee).
    await renderSequence(ImageStudio, [imageContext(), imageContext(IMAGE_FULL)]);

    // Sanity: the restored FLUX tuning/defaults survived the async catalog arrival.
    expect(modelSelectValue()).toBe("flux2_dev");
    let snap = readSnapshot("image");
    expect(snap.resolution).toBe("1536x1536");
    expect(snap.sampler).toBe("dpmpp_2m");
    expect(snap.ipAdapterScale).toBe(0.85);

    // A genuine USER model change (FLUX → Z-Image) through the picker.
    await act(async () => {
      setSelect(document.body.querySelector(".settings-field-model select"), "z_image_turbo");
    });
    await act(async () => {});

    // Advanced-defaults reset: Z-Image only serves 1024x1024 + the "default" sampler, so the
    // restored FLUX resolution/sampler snap to Z-Image's declared values.
    // Reference-tuning reset: Z-Image declares no referenceStrengthDefault, so ipAdapterScale
    // falls back to the model-agnostic 0.6 — proving the restored 0.85 was NOT preserved here.
    expect(modelSelectValue()).toBe("z_image_turbo");
    snap = readSnapshot("image");
    expect(snap.model).toBe("z_image_turbo");
    expect(snap.resolution).toBe("1024x1024");
    expect(snap.sampler).toBe("default");
    expect(snap.ipAdapterScale).toBe(0.6);
  });

  // Video counterpart: VideoStudio has no ipAdapter reference-tuning, but its duration/
  // resolution/fps snap must still fire on a genuine user model change. The restore preserves
  // WAN's 1280x720 (valid for WAN); switching to LTX — which does not serve 1280x720 — snaps
  // the resolution to LTX's declared default, so a real user change is not swallowed.
  it("Video: a genuine post-load user model change snaps a now-invalid resolution to the new model default", async () => {
    seedSnapshot("video", VIDEO_SNAPSHOT);
    await renderSequence(VideoStudio, [videoContext(), videoContext(VIDEO_FULL)]);

    // Sanity: WAN's restored resolution survived the async catalog arrival.
    expect(modelSelectValue()).toBe("wan_t2v");
    expect(readSnapshot("video").resolution).toBe("1280x720");

    // User switches to LTX, whose limits do not include 1280x720.
    await act(async () => {
      setSelect(document.body.querySelector(".settings-field-model select"), "ltx_2_3");
    });
    await act(async () => {});

    expect(modelSelectValue()).toBe("ltx_2_3");
    const snap = readSnapshot("video");
    expect(snap.model).toBe("ltx_2_3");
    // LTX declares defaults.resolution "768x512" and no resolution list → the invalid restored
    // 1280x720 snaps to that default rather than lingering.
    expect(snap.resolution).toBe("768x512");
  });
});

// sc-12034 — the OTHER late-catalog corner (pre-existing, not introduced by S3): on a FRESH
// mount (NO saved snapshot) whose model catalog arrives AFTER mount, the reference-tuning knobs
// (ipAdapterScale / controlnetScale / trueCfgScale …) can settle at the model-agnostic fallbacks
// (0.6 / 0.8 / 4.0) instead of the model's DECLARED defaults. `model` initializes to the hardcoded
// fallback id while the catalog is empty; when the catalog arrives it CONTAINS that id (a valid
// installed model), so the model id never changes and the model-change reset never re-fires — the
// declared defaults are never applied. The fix re-applies them the first time the model resolves,
// but ONLY on a fresh mount (a restored snapshot's tuning, and a recipe's injected tuning, survive).
describe("reference-tuning declared defaults apply on a fresh late-catalog mount (sc-12034)", () => {
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

  // The fallback model id ("z_image_turbo" — ImageStudio's hardcoded default) that a fresh mount
  // lands on before any catalog loads. It DECLARES its own reference-tuning defaults, distinct from
  // the generic 0.6 / 0.8 / 4.0, so a knob left at the generic value proves the declared default
  // was never applied.
  const Z_IMAGE_TUNED = {
    ...Z_IMAGE,
    ui: {
      referenceStrengthDefault: 0.75,
      identityStructure: { default: 0.9 },
      variationStrength: { default: 6 },
    },
  };

  it("applies the model's DECLARED reference-tuning defaults, not the generic fallbacks", async () => {
    // No seeded snapshot → fresh mount. First render: empty catalog (model = fallback id).
    // Second render: the catalog arrives and CONTAINS that same id, so the id never changes.
    await renderSequence(ImageStudio, [
      imageContext(),
      imageContext({ imageModels: [Z_IMAGE_TUNED], models: [Z_IMAGE_TUNED] }),
    ]);

    expect(modelSelectValue()).toBe("z_image_turbo");
    const snap = readSnapshot("image");
    // Declared defaults, NOT the generic 0.6 / 0.8 / 4.0.
    expect(snap.ipAdapterScale).toBe(0.75);
    expect(snap.controlnetScale).toBe(0.9);
    expect(snap.trueCfgScale).toBe(6);
  });

  it("does NOT clobber a restored snapshot's tuning on a late-catalog mount", async () => {
    // A restored snapshot (ipAdapterScale present) is authoritative — the fresh-mount resolver must
    // stay disarmed so the restored value survives the async catalog arrival, even though the model
    // id (z_image_turbo) is stable and would otherwise look like a fresh mount.
    seedSnapshot("image", {
      mode: "text_to_image",
      model: "z_image_turbo",
      ipAdapterScale: 0.42,
      controlnetScale: 0.31,
      trueCfgScale: 2.5,
    });
    await renderSequence(ImageStudio, [
      imageContext(),
      imageContext({ imageModels: [Z_IMAGE_TUNED], models: [Z_IMAGE_TUNED] }),
    ]);

    expect(modelSelectValue()).toBe("z_image_turbo");
    const snap = readSnapshot("image");
    expect(snap.ipAdapterScale).toBe(0.42);
    expect(snap.controlnetScale).toBe(0.31);
    expect(snap.trueCfgScale).toBe(2.5);
  });
});
