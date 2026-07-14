import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { withAppContext, FakeEventSource, response, settle, field, changeField, openAdvancedSection } from "./main.testSupport.jsx";

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

  // sc-10516. A studio resolves `selectedPresetId` only against `availablePresets`, which
  // filters on the current mode AND model — so a launch that carried just the id selected
  // nothing and fell back to None. It must set all three together. And when a saved studio
  // snapshot exists, the first hydrate pass used to be skipped wholesale, which swallowed
  // the launched preset's defaults; only the snapshot's OWN preset may skip it.
  it("launches a preset into Video Studio, switching mode + model and applying its defaults", async () => {
    // A snapshot for a DIFFERENT preset — the launched one must still hydrate.
    window.localStorage.setItem(
      "sceneworks-studio-video-project-1",
      JSON.stringify({ mode: "text_to_video", model: "ltx_2_3", selectedPresetId: "some_other_preset" }),
    );

    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
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
            presets: [
              {
                id: "bridge",
                name: "Bridge Clip",
                workflow: "image_to_video",
                modes: ["image_to_video"],
                model: "wan_i2v",
                defaults: { mode: "image_to_video", duration: 8, fps: 30, resolution: "1280x720", steps: 41 },
              },
            ],
            requestedGpu: "auto",
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            // The launch must move OFF the default model, not just off the default mode.
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                capabilities: ["text_to_video", "image_to_video"],
                limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
              },
              {
                id: "wan_i2v",
                name: "Wan I2V",
                type: "video",
                capabilities: ["image_to_video", "text_to_video"],
                limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
              },
            ],
            studioLaunch: {
              id: "launch-1",
              view: "Video",
              presetId: "bridge",
              presetModel: "wan_i2v",
              presetMode: "image_to_video",
            },
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    const activeChip = [...document.body.querySelectorAll(".preset-chip.active")].map((chip) => chip.textContent.trim());
    expect(activeChip).toEqual(["Bridge Clip"]);
    expect(field(container, "Model").value).toBe("wan_i2v");
    expect(field(container, "Duration").value).toBe("8");

    await openAdvancedSection();
    expect(field(container, "Steps").value).toBe("41");
  });

  // epic 11949 Phase 3: a general (model-agnostic) preset appears in its own chip group on
  // any model and toggles into a stack, independently of the single-select model preset,
  // without changing the model. (Composition into the prompt lands in Phase 4.)
  it("stacks a general preset onto any model without changing the model", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            characters: [],
            createVideoJob: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            rememberLocalGenerationJob: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [
              {
                id: "film_look",
                name: "Film Look",
                kind: "general",
                prompt: { suffix: "Kodak Portra 400" },
                defaults: { aspect: "16:9" },
              },
            ],
            requestedGpu: "auto",
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                capabilities: ["text_to_video", "image_to_video"],
                limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    // The general preset lives in its own chip group, not the model-preset row.
    const generalGroup = container.querySelector(".general-preset-chips");
    expect(generalGroup).not.toBeNull();
    const chip = [...generalGroup.querySelectorAll(".preset-chip")].find((c) => c.textContent.trim() === "Film Look");
    expect(chip).toBeDefined();
    expect(chip.classList.contains("active")).toBe(false);

    const modelBefore = field(container, "Model").value;
    await act(async () => chip.click());

    // Toggling activates the chip and leaves the model untouched (Phase 3 is state only).
    const activeGeneral = [...container.querySelectorAll(".general-preset-chips .preset-chip.active")].map((c) =>
      c.textContent.trim(),
    );
    expect(activeGeneral).toEqual(["Film Look"]);
    expect(field(container, "Model").value).toBe(modelBefore);

    // epic 11949 Phase 4: toggling a general preset surfaces a live composed-prompt preview
    // showing exactly what the run will send — the safeguard against messy concatenation.
    const preview = container.querySelector(".preset-stack-preview");
    expect(preview).not.toBeNull();
    expect(preview.querySelector(".preset-stack-prompt p").textContent).toContain("Kodak Portra 400");
    expect(preview.textContent).toContain("Film Look");
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
    // The preset's installed LoRA is seeded into the visible picker, so the strip no longer
    // claims it's applied invisibly at generation.
    expect(container.textContent).not.toContain("Preset LoRA applied at generation");

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
        // The preset prompt prefix/suffix is still folded in server-side from recipePresetId,
        // so the client sends the raw prompt. The preset's LoRA, though, is now a visible
        // picker selection: it rides in `loras` at its resolved weight and the client flags
        // presetLorasResolvedClientSide so the server won't re-merge it.
        prompt: "Camera slowly pushes in while the scene comes alive",
        presetLorasResolvedClientSide: true,
        loras: [expect.objectContaining({ id: "video_motion", weight: 0.8 })],
        advanced: expect.objectContaining({
          resolution: "1280x720",
        }),
      }),
    );
    const submittedAdvanced = createVideoJob.mock.calls[0][0].advanced;
    expect(submittedAdvanced).not.toHaveProperty("recipePresetName");
    expect(submittedAdvanced).not.toHaveProperty("recipePresetPrompt");
  });

  // epic 11949 Phase 5: with a general preset stacked, the client composes the prompt + negative
  // and flags presetPromptResolvedClientSide so the server takes them verbatim (no double-fold).
  it("sends the client-composed prompt + flag when a general preset is stacked", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            characters: [],
            createPersonDetectionJob: () => {},
            createPersonTrackJob: () => {},
            createVideoJob,
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            rememberLocalGenerationJob: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [
              {
                id: "film_look",
                name: "Film Look",
                kind: "general",
                prompt: { suffix: "Kodak Portra 400" },
                defaults: { negativePrompt: "flicker" },
              },
            ],
            requestedGpu: "auto",
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                capabilities: ["text_to_video", "image_to_video"],
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

    // Toggle the general preset into the stack (no base model preset selected).
    await act(async () => {
      [...container.querySelectorAll(".general-preset-chips .preset-chip")]
        .find((chip) => chip.textContent.trim() === "Film Look")
        .click();
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Render clip").click();
    });

    expect(createVideoJob).toHaveBeenCalledWith(
      expect.objectContaining({
        // base=none, so the user's prompt + the general's append fragment, composed client-side.
        prompt: "Camera slowly pushes in while the scene comes alive, Kodak Portra 400",
        negativePrompt: "flicker",
        presetPromptResolvedClientSide: true,
      }),
    );
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

    const clickButton = async (label) => {
      await act(async () => {
        [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === label).click();
      });
    };
    const cardFor = (name) =>
      [...document.body.querySelectorAll(".preset-card")].find((card) => card.textContent.includes(name));
    const loraWeightSlider = () => document.body.querySelector(".lora-slot-weight input[type=range]");

    await clickButton("New preset");
    await changeField(field(container, "Name"), "Soft Morning");
    // Open the LoRA picker, then click the compatible LoRA row to add it.
    await clickButton("Add LoRA");
    await act(async () => {
      [...document.body.querySelectorAll(".lora-pick-row")]
        .find((button) => button.textContent.includes("Global Detail"))
        .click();
    });
    await changeField(loraWeightSlider(), "0.35");
    expect(field(container, "ID").value).toBe("soft_morning");
    await clickButton("Create preset");
    expect(createPreset).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "soft_morning",
        name: "Soft Morning",
        scope: "global",
        loras: [{ id: "global_detail", weight: 0.35 }],
        modes: ["text_to_image", "character_image"],
        // "Text" persists workflow + the sub-mode the studios restore (sc-10514).
        workflow: "text_to_image",
        defaults: expect.objectContaining({ mode: "text_to_image" }),
      }),
    );
    // The Qwen LoRA doesn't share a family with Z-Image, so it never reaches the picker.
    expect(container.textContent).not.toContain("Qwen Only");

    // Adding then removing a LoRA leaves no slot behind.
    expect(document.body.querySelector(".lora-slot").textContent).toContain("Global Detail");
    await clickButton("×");
    expect(document.body.querySelector(".lora-slot")).toBeNull();

    await clickButton("All presets");
    await act(async () => {
      cardFor("Moody").querySelector(".secondary-action").click();
    });
    await changeField(field(container, "Description"), "Richer low key color.");
    await clickButton("Save preset");
    expect(updatePreset).toHaveBeenCalledWith("moody", expect.objectContaining({ ui: { description: "Richer low key color." } }), "global");

    await clickButton("All presets");
    await act(async () => {
      cardFor("Moody").querySelector("[aria-label='Duplicate Moody']").click();
    });
    expect(duplicatePreset).toHaveBeenCalledWith("moody", "global");

    await act(async () => {
      cardFor("Moody").querySelector("[aria-label='Archive Moody']").click();
    });
    expect(deletePreset).toHaveBeenCalledWith("moody", "global");

    // Built-in presets stay read-only: duplicate only, no Edit and no Archive.
    expect(cardFor("Cinematic").querySelector("[aria-label='Duplicate Cinematic']")).not.toBeNull();
    expect(cardFor("Cinematic").querySelector("[aria-label='Archive Cinematic']")).toBeNull();
    expect(cardFor("Cinematic").textContent).not.toContain("Edit");
  });

  // sc-11964 (S5): the source/reference/character/person-track selections are USER choices, so
  // they persist in the studio snapshot and restore across a FULL app restart — not just via the
  // in-session keep-alive. The restart is simulated by seeding the snapshot, mounting while the
  // model/asset/character/track catalogs are still empty (the restart-restore window, when the
  // writer is ready-gated off), then letting the catalogs resolve on a re-render.
  const readVideoSnapshot = () =>
    JSON.parse(window.localStorage.getItem("sceneworks-studio-video-project-1") ?? "{}");

  const berniniModel = {
    id: "bernini",
    name: "Bernini",
    type: "video",
    capabilities: ["text_to_video", "reference_video_to_video"],
    defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
    limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
  };

  const renderVideoStudio = async (context) => {
    const value = {
      activeProject: { id: "project-1", name: "Noir" },
      createPersonDetectionJob: () => {},
      createPersonTrackJob: () => {},
      createVideoJob: () => {},
      gpuOptions: ["auto"],
      latestVideoAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      rememberLocalGenerationJob: () => {},
      purgeAsset: () => {},
      presets: [],
      requestedGpu: "auto",
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
      assets: [],
      characters: [],
      personTracks: [],
      videoModels: [],
      ...context,
    };
    // Mirror App.jsx's derivation exactly (App.jsx:767): an explicit `selectedAssetId` resolves
    // to that asset, otherwise `selectedAsset` FALLS BACK to assets[0] (the newest). Reproducing
    // that fallback is what makes the sc-11964 clobber path real in the test — without it the
    // selectedAsset-sync effect never fires and the restore assertion is a false green.
    const assets = value.assets ?? [];
    const selectedAssetId = value.selectedAssetId ?? null;
    value.selectedAssetId = selectedAssetId;
    value.selectedAsset = assets.find((asset) => asset.id === selectedAssetId) ?? assets[0] ?? null;
    await act(async () => {
      root.render(withAppContext(value, <VideoStudio />));
    });
    await settle();
  };

  it("restores source-clip / reference / character / person-track selections after a restart", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-video-project-1",
      JSON.stringify({
        mode: "reference_video_to_video",
        model: "bernini",
        sourceClipAssetId: "clip-1",
        referenceAssetIds: ["ref-1", "ref-2"],
        characterId: "char-1",
        characterLookId: "look-1",
        personTrackId: "track-1",
        replacementMode: "full_person_keep_outfit",
        trackName: "Hero",
      }),
    );

    root = createRoot(container);
    // Restart window: everything still resolving, so the ready-gated writer stays off.
    await renderVideoStudio({});
    // The catalogs land after mount.
    await renderVideoStudio({
      videoModels: [berniniModel],
      assets: [
        { id: "clip-1", type: "video", displayName: "Source Clip One" },
        { id: "ref-1", type: "image", displayName: "Ref One" },
        { id: "ref-2", type: "image", displayName: "Ref Two" },
      ],
      characters: [{ id: "char-1", name: "Hero Char", looks: [{ id: "look-1", name: "Look A" }] }],
      personTracks: [{ id: "track-1", sourceAssetId: "clip-1", name: "Hero" }],
    });

    // The restored source clip + reference images surface in their pickers (UI-level restore).
    expect(container.textContent).toContain("Source Clip One");
    expect(container.textContent).toContain("Ref One");
    expect(container.textContent).toContain("Ref Two");

    // The restored character resolves too (advanced holds the Character select).
    await openAdvancedSection();
    expect(field(container, "Character").value).toBe("char-1");

    // And the settled snapshot round-trips every selection — proving they were seeded from the
    // restore, not reset to defaults, and are persisted for the next restart.
    const snapshot = readVideoSnapshot();
    expect(snapshot.sourceClipAssetId).toBe("clip-1");
    expect(snapshot.referenceAssetIds).toEqual(["ref-1", "ref-2"]);
    expect(snapshot.characterId).toBe("char-1");
    expect(snapshot.characterLookId).toBe("look-1");
    expect(snapshot.personTrackId).toBe("track-1");
    expect(snapshot.replacementMode).toBe("full_person_keep_outfit");
    expect(snapshot.trackName).toBe("Hero");
  });

  // sc-11964 (S5) regression guard: on a cold restart nothing is explicitly selected, so App
  // derives `selectedAsset = assets[0]` (the NEWEST asset). The selectedAsset-sync effect must
  // NOT push that newest asset onto the restored source when the restored source isn't the newest
  // one — otherwise the two most important persisted fields silently fail to restore. Here the
  // restored clip ("clip-old") is deliberately NOT assets[0] ("clip-new"), so a regression would
  // clobber it to "clip-new".
  it("keeps a restored source clip that is NOT the newest asset (no assets[0] clobber)", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-video-project-1",
      JSON.stringify({
        mode: "reference_video_to_video",
        model: "bernini",
        sourceClipAssetId: "clip-old",
      }),
    );

    root = createRoot(container);
    // Restart window: catalogs empty, no explicit selection.
    await renderVideoStudio({});
    // Catalogs land. App's refreshAssets auto-selects the default/newest asset once the catalog
    // lands (`setSelectedAssetId((current) => current ?? defaultAsset.id)`, App.jsx:1270), so
    // selectedAssetId resolves to the NEWEST clip ("clip-new") — NOT the restored source. This is
    // what the real app does; without it the sync effect never fires and the assertion false-greens.
    // With the pre-fix gate this auto-default clobbers the restored source to "clip-new"; the
    // one-shot auto-default skip must keep "clip-old".
    await renderVideoStudio({
      videoModels: [berniniModel],
      selectedAssetId: "clip-new",
      assets: [
        { id: "clip-new", type: "video", displayName: "Newest Clip" },
        { id: "clip-old", type: "video", displayName: "Restored Clip" },
      ],
    });

    // The restored (non-newest) clip survives; the newest clip did not hijack the source.
    expect(container.textContent).toContain("Restored Clip");
    expect(readVideoSnapshot().sourceClipAssetId).toBe("clip-old");
  });

  // sc-11964 (S5) regression guard for the shared character-drop effect (generationStudio.jsx):
  // deleting the LAST-and-only selected character empties the catalog. A `.length` guard can't
  // tell "still loading" from "genuinely empty" and would leave the now-dangling characterId in
  // place. The loaded-once latch must still drop it once the catalog has landed at least once.
  it("drops the stale characterId when the last-and-only character is deleted", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-video-project-1",
      JSON.stringify({
        mode: "reference_video_to_video",
        model: "bernini",
        characterId: "char-1",
        characterLookId: "look-1",
      }),
    );

    root = createRoot(container);
    // Restart window: catalogs empty.
    await renderVideoStudio({});
    // The character catalog lands with the restored character present — the loaded-once latch trips
    // and the restored characterId is kept (it still resolves).
    await renderVideoStudio({
      videoModels: [berniniModel],
      characters: [{ id: "char-1", name: "Hero Char", looks: [{ id: "look-1", name: "Look A" }] }],
    });
    expect(readVideoSnapshot().characterId).toBe("char-1");

    // The user deletes their last-and-only character: the catalog is now legitimately empty. The
    // now-dangling characterId must drop (a pre-fix `.length` guard would leave it stale).
    await renderVideoStudio({
      videoModels: [berniniModel],
      characters: [],
    });

    const snapshot = readVideoSnapshot();
    expect(snapshot.characterId).toBe("");
    expect(snapshot.characterLookId).toBe("");
  });

  it("drops restored selections whose asset / character / track no longer exists", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-video-project-1",
      JSON.stringify({
        mode: "reference_video_to_video",
        model: "bernini",
        sourceClipAssetId: "clip-gone",
        referenceAssetIds: ["ref-1", "ref-gone"],
        characterId: "char-gone",
        characterLookId: "look-gone",
        personTrackId: "track-gone",
      }),
    );

    root = createRoot(container);
    await renderVideoStudio({});
    // Only ref-1 survives; the source clip, one reference, the character and the track are gone.
    await renderVideoStudio({
      videoModels: [berniniModel],
      assets: [{ id: "ref-1", type: "image", displayName: "Ref One" }],
      characters: [{ id: "char-1", name: "Hero Char", looks: [] }],
      personTracks: [{ id: "track-1", sourceAssetId: "clip-1", name: "Hero" }],
    });

    // The dangling source clip cleanly drops to the empty state; the surviving reference stays.
    expect(container.textContent).toContain("No source clip selected");
    expect(container.textContent).toContain("Ref One");
    expect(container.textContent).not.toContain("Ref Two");

    const snapshot = readVideoSnapshot();
    expect(snapshot.sourceClipAssetId).toBe("");
    expect(snapshot.referenceAssetIds).toEqual(["ref-1"]);
    expect(snapshot.characterId).toBe("");
    expect(snapshot.characterLookId).toBe("");
    expect(snapshot.personTrackId).toBe("");
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

    // The manager is the landing state now — open the blocked preset in the editor.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-card")]
        .find((card) => card.textContent.includes("Blocked"))
        .querySelector(".secondary-action")
        .click();
    });

    // The pinned model isn't in this install's catalog: say so instead of rendering an
    // empty select that would silently rewrite the preset's model on save.
    expect(field(container, "Model").textContent).toContain("z_image_turbo — not installed");
    expect(container.textContent).toContain("declares no preset-capable workflow");
    expect(container.textContent).toContain("Pending Style");
    expect(container.textContent).toContain("Missing or still importing");
    expect(container.textContent).toContain("Save blocked: pending_style has not finished importing.");
    expect(document.body.querySelector(".lora-slot-weight input[type=range]").disabled).toBe(true);
    expect([...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === "Save preset").disabled).toBe(true);
    expect(updatePreset).not.toHaveBeenCalled();
  });

  it("hides an unmet requirement but explains a broken value", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            createPreset: () => {},
            deletePreset: () => {},
            duplicatePreset: () => {},
            imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["text_to_image"] }],
            loras: [],
            presets: [],
            updatePreset: () => {},
            videoModels: [],
            setActiveView: () => {},
          },
          <PresetManagerScreen />,
        ),
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === "New preset").click();
    });

    const saveButton = () =>
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === "Create preset");

    // A REQUIREMENT (empty name): the CTA is disabled, but the form already marks the
    // field required — repeating it as a warning would be noise (sc-10500).
    expect(saveButton().disabled).toBe(true);
    expect(container.querySelector(".inline-warning")).toBeNull();

    // An ERROR (a value that is present but out of the range the backend enforces):
    // the CTA is disabled AND says why.
    await changeField(field(container, "Name"), "Overcooked");
    expect(saveButton().disabled).toBe(false);
    await openAdvancedSection();
    await changeField(field(container, "Steps"), "500");
    expect(saveButton().disabled).toBe(true);
    expect(container.textContent).toContain("Steps must be a whole number between 1 and 200.");
  });
});
