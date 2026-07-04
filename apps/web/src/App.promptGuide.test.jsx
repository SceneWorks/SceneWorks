import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { withAppContext, withImageStudioContext, settle, field, changeField } from "./main.testSupport.jsx";

describe("prompt guide popup (sc-1817)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    // Prompt guides are static assets fetched by path; echo the path back so
    // each guide's rendered body is distinguishable in assertions.
    global.fetch = vi.fn((url) => Promise.resolve({ ok: true, text: async () => `# Guide\n\nfetched ${url}` }));
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  const guideButton = () =>
    [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === "Prompt guide");

  it("opens the selected image model's guide without submitting the form", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Z Guide", path: "/prompt-guides/z-image-turbo.md" } },
            },
            {
              id: "qwen_image",
              name: "Qwen",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Qwen Guide", path: "/prompt-guides/qwen-image.md" } },
            },
          ],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();

    expect(document.body.querySelector("[role=dialog]")).not.toBeNull();
    expect(document.body.querySelector("#prompt-guide-title").textContent).toBe("Z Guide");
    expect(document.body.textContent).toContain("fetched /prompt-guides/z-image-turbo.md");
    expect(createImageJob).not.toHaveBeenCalled();
  });

  it("renders the new model's guide after switching models", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Z Guide", path: "/prompt-guides/z-image-turbo.md" } },
            },
            {
              id: "qwen_image",
              name: "Qwen",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Qwen Guide", path: "/prompt-guides/qwen-image.md" } },
            },
          ],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();
    expect(document.body.querySelector("#prompt-guide-title").textContent).toBe("Z Guide");

    await act(async () => {
      document.body.querySelector(".modal-close").click();
    });
    await changeField(field(container, "Model"), "qwen_image");
    await settle();

    await act(async () => {
      guideButton().click();
    });
    await settle();
    expect(document.body.querySelector("#prompt-guide-title").textContent).toBe("Qwen Guide");
    expect(document.body.textContent).toContain("fetched /prompt-guides/qwen-image.md");
  });

  it("falls back to the generic image guide when the model declares none", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", capabilities: ["text_to_image"] }],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();

    expect(global.fetch).toHaveBeenCalledWith("/prompt-guides/generic-image.md");
    expect(document.body.querySelector("#prompt-guide-title").textContent).toBe("Image Prompt Guide");
  });

  it("falls back to the generic video guide and does not submit the video form", async () => {
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
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            rememberLocalGenerationJob: () => {},
            requestedGpu: "auto",
            selectedAsset: null,
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                family: "ltx-video",
                capabilities: ["text_to_video"],
                defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();

    expect(global.fetch).toHaveBeenCalledWith("/prompt-guides/generic-video.md");
    expect(document.body.querySelector("#prompt-guide-title").textContent).toBe("Video Prompt Guide");
    expect(createVideoJob).not.toHaveBeenCalled();
  });
});
