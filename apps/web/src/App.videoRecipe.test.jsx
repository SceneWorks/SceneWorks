import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { withAppContext, FakeEventSource, response, settle, field, openAdvancedSection } from "./main.testSupport.jsx";

// sc-12324: "Use this recipe" on a video asset. A generated clip records a full recipe, and the
// user must be able to re-run it — with or without the original seed — exactly as they can an
// image. These assert the relaunched FORM against the recorded recipe field-for-field; a test
// that still passes with the recipe dropped is worthless (epic 1788 has shipped seven of those),
// so each one pins values the defaults do not produce.

const VIDEO_MODELS = [
  {
    id: "ltx_2_3",
    name: "LTX",
    type: "video",
    capabilities: ["text_to_video", "image_to_video", "first_last_frame"],
    limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
  },
  {
    id: "wan_i2v",
    name: "Wan I2V",
    type: "video",
    capabilities: ["image_to_video", "text_to_video"],
    limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
  },
  {
    id: "bernini_2",
    name: "Bernini 2",
    type: "video",
    capabilities: ["ads2v", "multi_video_to_video", "reference_to_video", "text_to_video"],
    limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
  },
];

// A recipe shaped exactly as `build_video_sidecar_parts` assembles one: resolved values under
// `normalizedSettings`, the client's `advanced` block verbatim under `rawAdapterSettings`.
function videoRecipe(overrides = {}) {
  const { normalizedSettings = {}, rawAdapterSettings = {}, ...rest } = overrides;
  return {
    mode: "text_to_video",
    model: "wan_i2v",
    adapter: "mlx_wan",
    prompt: "the hero walks through rain",
    negativePrompt: "blurry, warped hands",
    seed: 4242,
    loras: [],
    normalizedSettings: {
      duration: 8,
      fps: 30,
      width: 1280,
      height: 720,
      quality: "best",
      family: "wan-video",
      ...normalizedSettings,
    },
    rawAdapterSettings: {
      resolution: "1280x720",
      steps: 41,
      guidanceScale: 7.5,
      ...rawAdapterSettings,
    },
    ...rest,
  };
}

function renderStudio(context = {}) {
  return withAppContext(
    {
      activeProject: { id: "project-1", name: "Noir" },
      assets: [],
      characters: [],
      createPersonDetectionJob: () => {},
      createPersonTrackJob: () => {},
      createVideoJob: () => {},
      gpuOptions: ["auto"],
      latestVideoAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      rememberLocalGenerationJob: () => {},
      personTracks: [],
      purgeAsset: () => {},
      presets: [],
      requestedGpu: "auto",
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
      videoModels: VIDEO_MODELS,
      ...context,
    },
    <VideoStudio />,
  );
}

function activeMode() {
  return document.body.querySelector(".mode-tab.active")?.textContent.trim();
}

describe("video recipe replay (sc-12324)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  async function launch(context) {
    root = createRoot(container);
    await act(async () => {
      root.render(renderStudio(context));
    });
    await settle();
  }

  // The core AC. Every value here differs from the studio's default, so a dropped recipe fails.
  it("restores the recorded settings field-for-field", async () => {
    // A snapshot the recipe must OVERRIDE — replay reproduces the recipe, not a hybrid of the
    // recipe and whatever the studio was last left on.
    window.localStorage.setItem(
      "sceneworks-studio-video-project-1",
      JSON.stringify({ mode: "text_to_video", model: "ltx_2_3", resolution: "768x512", duration: 4, fps: 24 }),
    );

    await launch({
      studioLaunch: { id: "launch-1", view: "Video", assetId: "asset-1", recipe: videoRecipe(), replaySeed: null },
    });

    expect(field(container, "Model").value).toBe("wan_i2v");
    expect(field(container, "Resolution").value).toBe("1280x720");
    expect(field(container, "Duration").value).toBe("8");
    expect(field(container, "Prompt").value).toBe("the hero walks through rain");

    await openAdvancedSection();
    expect(field(container, "Negative prompt").value).toBe("blurry, warped hands");
    expect(field(container, "Frames").value).toBe("30");
    expect(field(container, "Steps").value).toBe("41");
    expect(field(container, "Guidance").value).toBe("7.5");
  });

  // The explicit ask: re-run with or without the same seed.
  it("replays the exact seed when Keep seed was checked", async () => {
    await launch({
      studioLaunch: { id: "launch-1", view: "Video", assetId: "asset-1", recipe: videoRecipe(), replaySeed: 4242 },
    });
    await openAdvancedSection();
    expect(field(container, "Seed").value).toBe("4242");
  });

  it("leaves the seed random for a close variation when Keep seed was not checked", async () => {
    await launch({
      studioLaunch: { id: "launch-1", view: "Video", assetId: "asset-1", recipe: videoRecipe(), replaySeed: null },
    });
    // A blank seed is also the studio's default, so on its own it would pass even if the recipe
    // never applied. Pin a non-default field from the same recipe to keep this honest.
    expect(field(container, "Model").value).toBe("wan_i2v");
    await openAdvancedSection();
    // Blank, NOT the recipe's own 4242 — the default re-run is a variation, matching the image lane.
    expect(field(container, "Seed").value).toBe("");
  });

  // Seed 0 is falsy: a `replaySeed &&` guard would silently randomize it.
  it("replays a seed of 0 rather than reading it as absent", async () => {
    await launch({
      studioLaunch: {
        id: "launch-1",
        view: "Video",
        assetId: "asset-1",
        recipe: videoRecipe({ seed: 0 }),
        replaySeed: 0,
      },
    });
    await openAdvancedSection();
    expect(field(container, "Seed").value).toBe("0");
  });

  it("restores the recorded mode, not the snapshot's", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-video-project-1",
      JSON.stringify({ mode: "text_to_video", model: "ltx_2_3" }),
    );
    await launch({
      studioLaunch: {
        id: "launch-1",
        view: "Video",
        assetId: "asset-1",
        recipe: videoRecipe({ mode: "image_to_video", normalizedSettings: { sourceAssetId: "asset-src" } }),
        replaySeed: null,
      },
    });
    expect(activeMode()).toBe("Image → Video");
  });

  // sc-12345's fields. These were dropped at the worker's fact boundary, so ads2v — the densest
  // mode — could not be replayed at all: its reference video and subject images were unrecoverable.
  it("restores the multi-source ids that ads2v needs", async () => {
    const assets = [
      { id: "clip-main", type: "video", displayName: "Main", file: { path: "a.mp4", width: 1280, height: 720 } },
      { id: "clip-ref", type: "video", displayName: "Ref clip", file: { path: "b.mp4", width: 1280, height: 720 } },
      { id: "ref-1", type: "image", displayName: "Ref 1", file: { path: "c.png", width: 1024, height: 1024 } },
    ];
    await launch({
      assets,
      latestVideoAssets: assets,
      studioLaunch: {
        id: "launch-1",
        view: "Video",
        assetId: "asset-1",
        recipe: videoRecipe({
          mode: "ads2v",
          model: "bernini_2",
          normalizedSettings: {
            sourceClipAssetId: "clip-main",
            referenceClipAssetId: "clip-ref",
            referenceAssetIds: ["ref-1"],
          },
        }),
        replaySeed: null,
      },
    });

    expect(activeMode()).toBe("Clip + Ref Video");
    expect(field(container, "Model").value).toBe("bernini_2");
    // The reference ids reach the form: the CTA is not blocked on missing inputs, which is the
    // observable proof that ads2v's sources were actually restored — `hasInputs` demands the
    // source clip AND the reference clip AND at least one reference image.
    expect(container.querySelector(".prompt-cta").disabled).toBe(false);
  });

  // The I2V aspect snap re-snaps resolution to the source image's aspect whenever the source id
  // changes. Seeding a recipe's source would trip it and overwrite the recorded resolution.
  it("keeps the recorded resolution when replaying an image_to_video recipe", async () => {
    const assets = [
      // A square source: the snap would pull resolution to the nearest square-ish option.
      { id: "asset-src", type: "image", displayName: "Src", file: { path: "s.png", width: 1024, height: 1024 } },
    ];
    await launch({
      assets,
      latestVideoAssets: assets,
      studioLaunch: {
        id: "launch-1",
        view: "Video",
        assetId: "asset-1",
        recipe: videoRecipe({
          mode: "image_to_video",
          normalizedSettings: { sourceAssetId: "asset-src" },
          rawAdapterSettings: { resolution: "1280x720" },
        }),
        replaySeed: null,
      },
    });
    expect(field(container, "Resolution").value).toBe("1280x720");
  });

  // The recipe records the REQUESTED resolution in `advanced.resolution` and the RESOLVED dims in
  // normalizedSettings. Video's <select> only holds `limits.resolutions`, so seeding the resolved
  // dims would show a blank control and snap away. Re-submitting the request re-resolves the same.
  it("seeds the requested resolution, not the resolved dims", async () => {
    await launch({
      studioLaunch: {
        id: "launch-1",
        view: "Video",
        assetId: "asset-1",
        recipe: videoRecipe({
          // Resolved to an off-menu 1264x720 by the stride floor; the user picked 1280x720.
          normalizedSettings: { width: 1264, height: 720 },
          rawAdapterSettings: { resolution: "1280x720" },
        }),
        replaySeed: null,
      },
    });
    expect(field(container, "Resolution").value).toBe("1280x720");
  });

  // Product decision: a recipe whose model is gone still restores its settings, and says so rather
  // than letting the swap read as the recipe's own choice.
  it("names an uninstalled model instead of silently switching, and still restores the settings", async () => {
    await launch({
      studioLaunch: {
        id: "launch-1",
        view: "Video",
        assetId: "asset-1",
        recipe: videoRecipe({ model: "mochi_1_uninstalled" }),
        replaySeed: null,
      },
    });

    expect(container.textContent).toContain("mochi_1_uninstalled");
    expect(container.textContent).toContain("isn’t installed");
    // The picker landed on a real, mode-serving model rather than a phantom id.
    expect(VIDEO_MODELS.some((m) => m.id === field(container, "Model").value)).toBe(true);
    // …and the rest of the recipe still came through.
    expect(field(container, "Duration").value).toBe("8");
    await openAdvancedSection();
    expect(field(container, "Frames").value).toBe("30");
  });

  // The whole capability, end to end through the real App: a generated clip in the Library →
  // fullscreen preview → "Use this recipe" → Video Studio → re-submit. Asserting the SUBMITTED
  // PAYLOAD is the real proof — that payload is what actually reproduces the clip, and it also
  // covers App.jsx's routing (a video's recipe must reach Video Studio, not Image Studio), which
  // the component-level tests above bypass by handing VideoStudio a studioLaunch directly.
  it("replays a library clip's recipe through preview into Video Studio and re-submits it", async () => {
    const videoPayloads = [];
    const createdJobs = [];
    const generatedClip = {
      id: "asset-clip",
      projectId: "project-1",
      generationSetId: "genset-clip",
      type: "video",
      displayName: "Hero walks through rain",
      createdAt: "2026-07-16T00:00:00Z",
      file: { path: "assets/videos/hero.mp4", mimeType: "video/mp4", width: 1280, height: 720 },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
      recipe: {
        mode: "text_to_video",
        model: "wan_i2v",
        adapter: "mlx_wan",
        prompt: "the hero walks through rain",
        negativePrompt: "blurry, warped hands",
        seed: 4242,
        loras: [{ id: "grain_style", weight: 0.65 }],
        normalizedSettings: { duration: 8, fps: 30, width: 1280, height: 720, quality: "best" },
        rawAdapterSettings: { resolution: "1280x720", steps: 41, guidanceScale: 7.5, motion: "pan left" },
      },
    };

    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(response([generatedClip]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response(VIDEO_MODELS));
      }
      if (path.endsWith("/loras")) {
        return Promise.resolve(
          response([{ id: "grain_style", name: "Grain", family: "wan-video", scope: "global" }]),
        );
      }
      if (path.endsWith("/video/jobs") && options.method === "POST") {
        const payload = JSON.parse(options.body);
        videoPayloads.push(payload);
        const job = { id: "video-job-replay", type: "video_generate", status: "queued", payload };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Assets").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".asset-tile").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    await settle();
    // Keep the seed, so this is a byte-for-byte rerun rather than a variation.
    await act(async () => {
      document.body.querySelector(".preview-keep-seed input").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Use this recipe").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector("form.studio-shell").requestSubmit();
    });
    await settle();

    // The re-submitted job reproduces the recorded recipe field-for-field.
    expect(videoPayloads[0]).toMatchObject({
      projectId: "project-1",
      mode: "text_to_video",
      model: "wan_i2v",
      prompt: "the hero walks through rain",
      negativePrompt: "blurry, warped hands",
      seed: 4242,
      duration: 8,
      fps: 30,
      width: 1280,
      height: 720,
      quality: "best",
      recipePresetId: null,
      loras: [expect.objectContaining({ id: "grain_style", weight: 0.65 })],
      advanced: expect.objectContaining({
        resolution: "1280x720",
        steps: 41,
        guidanceScale: 7.5,
        motion: "pan left",
      }),
    });
  });

  // The launch effect that handles asset/preset launches also depends on the selected asset. A
  // recipe branch folded into it would re-fire on selection change and clobber user edits, so the
  // recipe effect is keyed on the launch id alone — and must not adopt the replayed clip as a source.
  it("does not fall through to the asset branch when the replayed clip is selected", async () => {
    const assets = [{ id: "asset-1", type: "video", displayName: "The clip", file: { path: "a.mp4" } }];
    await launch({
      assets,
      latestVideoAssets: assets,
      selectedAssetId: "asset-1",
      studioLaunch: {
        id: "launch-1",
        view: "Video",
        assetId: "asset-1",
        // A NON-default mode on purpose: text_to_video is the studio default, so asserting it
        // would pass even with the recipe dropped entirely.
        recipe: videoRecipe({ mode: "first_last_frame", model: "ltx_2_3" }),
        replaySeed: null,
      },
    });
    // The recipe's mode wins. Without the recipe guard on the launch effect, the asset branch
    // would run setMode(undefined) here (the launch carries no `mode`) and leave no tab active,
    // while also adopting the replayed clip as its own source clip.
    expect(activeMode()).toBe("First → Last");
  });
});
