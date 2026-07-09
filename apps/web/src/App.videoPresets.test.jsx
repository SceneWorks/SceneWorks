import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { withAppContext, FakeEventSource, response, settle, field, changeField } from "./main.testSupport.jsx";

describe("SceneWorks app shell", () => {
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

  it("applies preset defaults to video jobs", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createPersonDetectionJob: () => {},
          createPersonTrackJob: () => {},
          createVideoJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          latestVideoAssets: [],
          loras: [{ id: "video_motion", name: "Video Motion" }],
          setPreviewAsset: () => {},
          rememberLocalGenerationJob: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            {
              id: "dream_motion",
              name: "Dream Motion",
              workflow: "image_to_video",
              model: "ltx_2_3",
              defaults: { duration: 8, fps: 30, resolution: "1280x720", quality: "best", negativePrompt: "jitter" },
              prompt: { suffix: "smooth camera motion" },
              builtInLoras: [{ id: "video_motion" }],
              ui: { description: "Soft camera motion." },
            },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
          videoModels: [
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    // Video Studio opens on Text→Video (sc-5716); this preset targets image_to_video.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image → Video").click();
    });
    await settle();

    // sc-5875: presets are opt-in — select it explicitly before its defaults apply.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent.trim() === "Dream Motion").click();
    });
    await settle();

    expect(container.textContent).toContain("Dream Motion");
    expect(container.textContent).toContain("Soft camera motion.");
    expect(container.textContent).toContain("Adds: smooth camera motion");
    expect(container.textContent).toContain("Preset LoRA applied at generation: Video Motion");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Render clip").click();
    });

    expect(createVideoJob).toHaveBeenCalledWith(
      expect.objectContaining({
        duration: 8,
        fps: 30,
        width: 1280,
        height: 720,
        quality: "best",
        negativePrompt: "jitter",
        recipePresetId: "dream_motion",
        // Preset prompt prefix/suffix and preset LoRAs are now folded in
        // server-side from recipePresetId, so the client sends the raw prompt
        // and only its own picker selections (none here).
        prompt: "Camera slowly pushes in while the scene comes alive",
        loras: [],
        advanced: expect.objectContaining({
          resolution: "1280x720",
        }),
      }),
    );
    const submittedAdvanced = createVideoJob.mock.calls[0][0].advanced;
    expect(submittedAdvanced).not.toHaveProperty("recipePresetName");
    expect(submittedAdvanced).not.toHaveProperty("recipePresetPrompt");
  });

  it("lets a promptless video model (SVD) submit without a text prompt", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
            characters: [],
            createPersonDetectionJob: () => {},
            createPersonTrackJob: () => {},
            createVideoJob,
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            rememberLocalGenerationJob: () => {},
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            requestedGpu: "auto",
            selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "svd",
                name: "Stable Video Diffusion",
                type: "video",
                family: "svd",
                capabilities: ["image_to_video"],
                promptless: true,
                defaults: { duration: 4, fps: 7, resolution: "1024x576", quality: "balanced" },
                limits: { durations: [4], fps: [6, 7, 8], resolutions: ["1024x576", "576x1024"] },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    // Video Studio opens on Text→Video (sc-5716); SVD is image_to_video-only, so switch tabs.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image → Video").click();
    });
    await settle();

    // The prompt field advertises that no prompt is needed for promptless models.
    const promptField = document.body.querySelector("textarea[aria-label='Prompt']");
    expect(promptField.placeholder).toContain("No prompt needed");

    // With a source image selected and an empty prompt, Render clip is enabled
    // and submits (a text-prompted model would be blocked here).
    const generate = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Render clip");
    expect(generate.disabled).toBe(false);
    await act(async () => {
      generate.click();
    });
    expect(createVideoJob).toHaveBeenCalled();
  });

  it("filters video presets by mode and selected model", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createPersonDetectionJob: () => {},
          createPersonTrackJob: () => {},
          createVideoJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          latestVideoAssets: [],
          loras: [],
          setPreviewAsset: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            { id: "ltx_motion", name: "LTX Motion", workflow: "image_to_video", model: "ltx_2_3" },
            { id: "ltx_story", name: "LTX Story", workflow: "text_to_video", model: "ltx_2_3" },
            { id: "wan_motion", name: "Wan Motion", workflow: "image_to_video", model: "wan_2_2" },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
          videoModels: [
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
            {
              id: "wan_2_2",
              name: "Wan2.2",
              type: "video",
              capabilities: ["image_to_video", "text_to_video"],
              defaults: { duration: 5, fps: 24, resolution: "1280x720", quality: "balanced" },
              limits: { durations: [4, 5], fps: [24], resolutions: ["1280x720"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    // Video Studio opens on Text→Video (sc-5716); enter Image→Video to filter the image_to_video
    // presets, then the test walks Text→Video and a model switch as before.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image → Video").click();
    });
    await settle();

    expect(container.textContent).toContain("LTX Motion");
    expect(container.textContent).not.toContain("Wan Motion");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Text → Video").click();
    });
    await settle();

    expect(container.textContent).toContain("LTX Story");
    expect(container.textContent).not.toContain("LTX Motion");

    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });
    await changeField(field(container, "Model"), "wan_2_2");
    await settle();

    // No preset selected → no guidance strip (it only appears when a preset is active).
    expect(document.body.querySelector(".guidance-strip")).toBeNull();
    expect(container.textContent).not.toContain("LTX Story");
  });

  it("uses preset modes as the Video Studio picker surface", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          assets: [
            { id: "image-1", type: "image", displayName: "Frame One" },
            { id: "image-2", type: "image", displayName: "Frame Two" },
          ],
          characters: [],
          createPersonDetectionJob: () => {},
          createPersonTrackJob: () => {},
          createVideoJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          latestVideoAssets: [],
          loras: [],
          setPreviewAsset: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            {
              id: "camera_bridge",
              name: "Camera Bridge",
              workflow: "image_to_video",
              modes: ["image_to_video", "first_last_frame"],
              model: "ltx_2_3",
            },
            {
              id: "start_frame",
              name: "Start Frame",
              workflow: "image_to_video",
              modes: ["image_to_video"],
              model: "ltx_2_3",
            },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
          videoModels: [
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    // Video Studio opens on Text→Video (sc-5716); these presets target image_to_video /
    // first_last_frame, so enter Image→Video to surface them.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image → Video").click();
    });
    await settle();

    expect(container.textContent).toContain("Camera Bridge");
    expect(container.textContent).toContain("Start Frame");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "First → Last").click();
    });
    await settle();

    expect(container.textContent).toContain("Camera Bridge");
    expect(container.textContent).not.toContain("Start Frame");
  });

  it("creates, edits, duplicates, and archives presets from the manager", async () => {
    const createPreset = vi.fn(async (payload) => payload);
    const updatePreset = vi.fn(async (id, payload) => ({ ...payload, id }));
    const duplicatePreset = vi.fn(async (id) => ({ id: `${id}_copy` }));
    const deletePreset = vi.fn(async (id) => ({ id, archived: true }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          createPreset,
          deletePreset,
          duplicatePreset,
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          loras: [
            { id: "cinematic_detail", name: "Cinematic Detail", family: "z-image", scope: "builtin", defaultWeight: 0.55 },
            { id: "global_detail", name: "Global Detail", family: "z-image", scope: "global", defaultWeight: 0.7 },
            { id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "global" },
          ],
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              scope: "builtin",
              workflow: "text_to_image",
              model: "z_image_turbo",
              loras: [{ id: "cinematic_detail", weight: 0.5 }],
              ui: { description: "Built in cinematic finish." },
            },
            {
              id: "moody",
              name: "Moody",
              scope: "global",
              workflow: "text_to_image",
              model: "z_image_turbo",
              ui: { description: "Low key color." },
            },
          ],
          updatePreset,
          videoModels: [{ id: "ltx_2_3", name: "LTX", type: "video" }],
          setActiveView: () => {},
          },
          <PresetManagerScreen />,
        ),
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "New Preset").click();
    });
    await changeField(field(container, "Name"), "Soft Morning");
    // New flow: open the LoRA picker, then click the compatible LoRA row to add it.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".lora-pick-row")]
        .find((button) => button.textContent.includes("Global Detail"))
        .click();
    });
    await changeField(field(container, "Weight"), "0.35");
    expect(field(container, "ID").value).toBe("soft_morning");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Create Preset").click();
    });
    expect(createPreset).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "soft_morning",
        name: "Soft Morning",
        scope: "global",
        loras: [{ id: "global_detail", weight: 0.35 }],
        modes: ["text_to_image", "character_image"],
      }),
    );
    expect(container.textContent).toContain("Ready");
    expect(container.textContent).not.toContain("Qwen Only");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "New Preset").click();
    });
    await changeField(field(container, "Name"), "Plain Morning");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".lora-pick-row")]
        .find((button) => button.textContent.includes("Global Detail"))
        .click();
    });
    expect(document.body.querySelector(".lora-choice-list").textContent).toContain("Global Detail");
    await act(async () => {
      [...document.body.querySelectorAll(".lora-choice button")].find((button) => button.textContent === "Remove").click();
    });
    expect(document.body.querySelector(".lora-choice-list")).toBeNull();

    await act(async () => {
      [...document.body.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await changeField(field(container, "Description"), "Richer low key color.");
    expect(container.textContent).not.toContain("Queue Import");
    expect(field(container, "Source URL")).toBeUndefined();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "New Preset").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await changeField(field(container, "Description"), "Richer low key color.");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save Preset").click();
    });
    expect(updatePreset).toHaveBeenCalledWith("moody", expect.objectContaining({ ui: { description: "Richer low key color." } }), "global");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Duplicate").click();
    });
    expect(duplicatePreset).toHaveBeenCalledWith("moody", "global");

    await act(async () => {
      [...document.body.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Archive").click();
    });
    expect(deletePreset).toHaveBeenCalledWith("moody", "global");
  });

  it("explains preset save blockers and selected LoRA warning states", async () => {
    const updatePreset = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          createPreset: () => {},
          deletePreset: () => {},
          duplicatePreset: () => {},
          imageModels: [],
          loras: [{ id: "pending_style", name: "Pending Style", family: "z-image", scope: "global", installState: "missing" }],
          presets: [
            {
              id: "blocked",
              name: "Blocked",
              scope: "global",
              workflow: "text_to_image",
              model: "z_image_turbo",
              loras: [{ id: "pending_style" }],
            },
          ],
          updatePreset,
          videoModels: [],
          setActiveView: () => {},
          },
          <PresetManagerScreen />,
        ),
      );
    });

    expect(container.textContent).toContain("No models");
    expect(container.textContent).not.toContain("No model selected");
    expect(container.textContent).toContain("Pending Style");
    expect(container.textContent).toContain("Missing or still importing");
    expect(container.textContent).toContain("Save blocked: pending_style has not finished importing.");
    expect(field(container, "Weight").disabled).toBe(true);
    expect([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save Preset").disabled).toBe(true);
    expect(updatePreset).not.toHaveBeenCalled();
  });
});
