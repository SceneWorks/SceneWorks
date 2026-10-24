// The viewer's "Re-Generate" button routes an `autoGenerate` launch into Image Studio, which must
// enqueue the replayed recipe straight to the queue (no form event, no click) with a fresh random
// seed, once the recipe has fully hydrated. On a guard abort it stays on the prefilled studio instead
// of silently dropping the attempt.
import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mountRoot, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, apiFetch: vi.fn(async () => ({})) };
});

import { AppContext } from "../context/AppContext.js";
import { ImageStudio } from "./ImageStudio.jsx";

const MODEL = {
  id: "krea_2_raw",
  name: "Krea 2 Raw",
  type: "image",
  family: "krea2",
  capabilities: ["text_to_image"],
  installState: "installed",
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024", "1536x1024"], samplers: ["default"], schedulers: ["default"] },
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createImageJob: vi.fn(async () => ({ id: "job_1" })),
    createPreset: vi.fn(),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [MODEL],
    models: [MODEL],
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

// A launch exactly as App's sendAssetRecipeToQueue emits it: the recipe, a null replay seed (fresh
// random seed — never the source image's own), and the autoGenerate flag that asks the studio to
// fire submit by itself.
function regenerateLaunch() {
  return {
    id: "launch_regenerate_1",
    view: "Image",
    recipe: {
      mode: "text_to_image",
      model: MODEL.id,
      prompt: "a lighthouse in heavy fog",
      negativePrompt: "",
      loras: [],
      seed: 1234, // the SOURCE image's recorded seed — must NOT be reused
      normalizedSettings: { width: 1024, height: 1024, count: 1 },
      rawAdapterSettings: { steps: 28, guidanceScale: 4.5 },
    },
    replaySeed: null,
    autoGenerate: true,
  };
}

describe("Image Studio auto-generate (Re-Generate)", () => {
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
  }

  it("enqueues the replayed recipe once with a fresh random seed and lands on the Queue", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job_1" }));
    const setActiveView = vi.fn();
    const context = baseContext({ createImageJob, setActiveView, studioLaunch: regenerateLaunch() });

    await render(context);

    expect(createImageJob).toHaveBeenCalledTimes(1);
    // replaySeed null → the studio leaves seed blank → builder emits null = "random new seed".
    // The source image's own seed (1234) must never leak through a Re-Generate.
    expect(createImageJob.mock.calls[0][0].seed).toBeNull();
    expect(createImageJob.mock.calls[0][0].prompt).toBe("a lighthouse in heavy fog");
    expect(setActiveView).toHaveBeenCalledWith("Queue");
  });

  it("falls back to the prefilled studio when the enqueue is aborted, instead of pretending it queued", async () => {
    const createImageJob = vi.fn(async () => null); // beforeCreate declined / no job created
    const setActiveView = vi.fn();
    const context = baseContext({ createImageJob, setActiveView, studioLaunch: regenerateLaunch() });

    await render(context);

    expect(createImageJob).toHaveBeenCalledTimes(1);
    expect(setActiveView).toHaveBeenCalledWith("Image");
    expect(setActiveView).not.toHaveBeenCalledWith("Queue");
  });
});