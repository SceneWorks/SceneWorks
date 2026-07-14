import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { withAppContext, withImageStudioContext, FakeEventSource, response, settle, field, changeField } from "./main.testSupport.jsx";

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

  it("ignores duplicate image submits while job creation is in flight", async () => {
    let resolveJob;
    const createImageJob = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveJob = resolve;
        }),
    );
    const onLocalJobCreated = vi.fn();
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onLocalJobCreated,
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
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });

    expect(createImageJob).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveJob({ id: "image-job-1" });
    });
    await settle();

    expect(onLocalJobCreated).toHaveBeenCalledWith({ id: "image-job-1" });
  });

  it("remembers studio settings per workspace across remounts", async () => {
    const imageProps = {
      activeProject: { id: "project-1", name: "Noir" },
      assets: [],
      characters: [],
      createImageJob: () => {},
      deleteAsset: () => {},
      gpuOptions: ["auto"],
      imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
      latestAssets: [],
      localJobs: [],
      loras: [],
      onPreview: () => {},
      purgeAsset: () => {},
      requestedGpu: "auto",
      selectedAsset: null,
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
    };
    const promptField = () => document.body.querySelector("textarea[aria-label='Prompt']");

    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext(imageProps));
    });
    const defaultPrompt = promptField().value;
    await changeField(promptField(), "a custom remembered prompt");
    await settle();

    // Leaving the studio and returning to the same workspace restores the prompt.
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext(imageProps));
    });
    expect(promptField().value).toBe("a custom remembered prompt");

    // A different workspace starts from its own settings, not workspace-1's.
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext({ ...imageProps, activeProject: { id: "project-2", name: "Other" } }));
    });
    expect(promptField().value).toBe(defaultPrompt);
  });

  it("keeps completed image progress visible until the generated asset renders", async () => {
    const completedJob = {
      id: "image-job-1",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      elapsedSeconds: 8,
      requestedGpu: "auto",
      payload: { prompt: "long alley" },
      result: { generationSetId: "gen-1", assetIds: ["asset-1"] },
    };
    const imageProps = {
      activeProject: { id: "project-1", name: "Noir" },
      assets: [],
      characters: [],
      createImageJob: () => {},
      deleteAsset: () => {},
      gpuOptions: ["auto"],
      imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
      latestAssets: [],
      localJobs: [completedJob],
      loras: [],
      onPreview: () => {},
      purgeAsset: () => {},
      requestedGpu: "auto",
      selectedAsset: null,
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext(imageProps));
    });

    // Card stays visible while the completed job waits for its asset to arrive.
    expect(document.body.querySelector(".worker-progress-card")).not.toBeNull();
    expect(container.textContent).not.toContain("No fresh image batch");

    const generatedAsset = {
      id: "asset-1",
      type: "image",
      displayName: "Generated Image",
      generationSetId: "gen-1",
      status: {},
    };
    await act(async () => {
      root.render(withImageStudioContext({ ...imageProps, assets: [generatedAsset], latestAssets: [generatedAsset] }));
    });

    // Once the asset surfaces in latestAssets the card collapses out of the stack
    // (selectStackedJobs + resultVisible). The asset itself is in Recent Assets.
    expect(document.body.querySelector(".worker-progress-card")).toBeNull();
    expect(container.textContent).not.toContain("No fresh image batch");
  });

  it("reconstructs running image batch slots from a generation set without asset ids", async () => {
    const localJob = {
      id: "image-job-1",
      type: "image_generate",
      status: "running",
      stage: "generating",
      progress: 0.82,
      elapsedSeconds: 8,
      requestedGpu: "auto",
      payload: { prompt: "long alley", count: 3 },
      result: { generationSetId: "gen-1", expectedCount: 3 },
    };
    const assets = [
      {
        id: "asset-2",
        projectId: "project-1",
        type: "image",
        displayName: "Generated",
        generationSetId: "gen-1",
        file: { path: "runs/run_0007/assets/images/generated_0002.png", mimeType: "image/png" },
        status: {},
      },
      {
        id: "asset-1",
        projectId: "project-1",
        type: "image",
        displayName: "Generated",
        generationSetId: "gen-1",
        file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
        status: {},
      },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets,
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [localJob],
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

    const images = [...document.body.querySelectorAll(".worker-progress-card__thumb-media")].map((image) => image.getAttribute("src"));
    expect(images[0]).toContain("/api/v1/projects/project-1/files/assets/images/generated_0001.png");
    expect(images[1]).toContain("/api/v1/projects/project-1/files/runs/run_0007/assets/images/generated_0002.png");
    // 2 completed thumbnails + 1 skeleton slot = 3 total cells.
    expect(document.body.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBe(1);
  });

  it("cancels a running image job from the studio progress card", async () => {
    const runningJob = {
      id: "image-job-cancel",
      type: "image_generate",
      status: "running",
      stage: "generating",
      progress: 0.4,
      requestedGpu: "auto",
      payload: { prompt: "cancel me" },
      result: {},
    };
    const onCancelJob = vi.fn();
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [runningJob],
          loras: [],
          onCancelJob,
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    const cancelButton = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Cancel");
    expect(cancelButton).not.toBeUndefined();

    await act(async () => {
      cancelButton.click();
    });

    expect(onCancelJob).toHaveBeenCalledWith(expect.objectContaining({ id: "image-job-cancel" }));
  });

  it("hides the cancel control once an image job reaches a terminal state", async () => {
    const completedJob = {
      id: "image-job-done",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      requestedGpu: "auto",
      payload: { prompt: "all done" },
      result: {},
    };
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [completedJob],
          loras: [],
          onCancelJob: () => {},
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect([...document.body.querySelectorAll("button")].some((button) => button.textContent === "Cancel run")).toBe(false);
  });

  it("hides completed image progress with stale missing result metadata", async () => {
    const staleCompletedJob = {
      id: "image-job-stale",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      elapsedSeconds: 8,
      requestedGpu: "auto",
      updatedAt: "2026-05-18T00:00:00Z",
      payload: { prompt: "missing result metadata" },
      result: {},
    };

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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [staleCompletedJob],
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

    expect(container.textContent).not.toContain("Finished. Fetching result...");
    expect(container.textContent).toContain("No fresh image batch");
  });

  it("removes a canceled image job's progress card and placeholder thumbnails", async () => {
    const canceledJob = {
      id: "image-job-canceled",
      type: "image_generate",
      status: "canceled",
      stage: "canceled",
      progress: 0.5,
      requestedGpu: "auto",
      payload: { prompt: "abandon ship", count: 4 },
      result: { expectedCount: 4 },
    };
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [canceledJob],
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
    await settle();

    expect(document.body.querySelector(".worker-progress-card")).toBeNull();
    expect(document.body.querySelector(".review-placeholder")).toBeNull();
    expect(container.textContent).not.toContain("Canceled #");
    expect(container.textContent).toContain("No fresh image batch");
  });

  it("stacks multiple image runs, each with its own progress card and slots", async () => {
    const runningJob = {
      id: "image-job-run",
      type: "image_generate",
      status: "running",
      stage: "generating",
      progress: 0.5,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:00:00Z",
      payload: { prompt: "first run", count: 2 },
      result: { generationSetId: "gen-1", expectedCount: 2 },
    };
    const queuedJob = {
      id: "image-job-queue",
      type: "image_generate",
      status: "queued",
      progress: 0,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:01:00Z",
      payload: { prompt: "second run", count: 3 },
      result: { expectedCount: 3 },
    };
    const renderedAsset = {
      id: "asset-1",
      projectId: "project-1",
      type: "image",
      displayName: "Generated",
      generationSetId: "gen-1",
      file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
      status: {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [renderedAsset],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [runningJob, queuedJob],
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
    await settle();

    expect(document.body.querySelectorAll(".worker-progress-card").length).toBe(2);
    // Running run renders its one finished image alongside its remaining slot.
    expect(document.body.querySelector(".worker-progress-card__thumb-media")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated_0001.png",
    );
    // Queued run shows its own pending skeleton slots while it waits.
    expect(document.body.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBeGreaterThan(0);
    expect(container.textContent).not.toContain("No fresh image batch");
  });

  it("holds a completed run above the queue until the next run starts", async () => {
    const completedJob = {
      id: "image-job-done",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:00:00Z",
      completedAt: "2026-05-27T10:00:30Z",
      payload: { prompt: "finished run" },
      result: { generationSetId: "gen-1", assetIds: ["asset-1"] },
    };
    const nextJob = {
      id: "image-job-next",
      type: "image_generate",
      status: "queued",
      progress: 0,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:01:00Z",
      payload: { prompt: "next run", count: 2 },
      result: { expectedCount: 2 },
    };
    const renderedAsset = {
      id: "asset-1",
      projectId: "project-1",
      type: "image",
      displayName: "Generated",
      generationSetId: "gen-1",
      file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
      status: {},
    };
    const baseProps = {
      activeProject: { id: "project-1", name: "Noir" },
      assets: [renderedAsset],
      characters: [],
      createImageJob: () => {},
      deleteAsset: () => {},
      gpuOptions: ["auto"],
      imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
      latestAssets: [renderedAsset],
      loras: [],
      onPreview: () => {},
      purgeAsset: () => {},
      requestedGpu: "auto",
      selectedAsset: null,
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
    };

    root = createRoot(container);
    // Completed run with a run still queued behind it: both stay, completed on top.
    await act(async () => {
      root.render(withImageStudioContext({ ...baseProps, localJobs: [completedJob, nextJob] }));
    });
    await settle();
    expect(document.body.querySelectorAll(".worker-progress-card").length).toBe(2);
    // The queued next run shows skeleton slots for its expected outputs.
    expect(document.body.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBeGreaterThan(0);

    // The next run starts: the completed run slides out and the running run remains.
    await act(async () => {
      root.render(
        withImageStudioContext({
          ...baseProps,
          localJobs: [completedJob, { ...nextJob, status: "running", stage: "generating", progress: 0.3 }],
        }),
      );
    });
    await settle();
    expect(document.body.querySelectorAll(".worker-progress-card").length).toBe(1);
    expect(document.body.querySelector(".worker-progress-card.running")).not.toBeNull();
    expect(document.body.querySelector(".worker-progress-card.completed")).toBeNull();
  });

  it("submits compatible image LoRAs while capping simple user selections at four", async () => {
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [
            { id: "built_in", name: "Built In", family: "z-image", scope: "builtin", defaultWeight: 0.6 },
            { id: "global_style", name: "Global Style", family: "z-image", scope: "global" },
            { id: "project_mira", name: "Project Mira", family: "z-image", scope: "project", files: ["mira.safetensors"] },
            { id: "third_user", name: "Third User", family: "z-image", scope: "global" },
            { id: "fourth_user", name: "Fourth User", family: "z-image", scope: "global" },
            { id: "fifth_user", name: "Fifth User", family: "z-image", scope: "global" },
            { id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "global" },
            { id: "missing_lora", name: "Missing LoRA", family: "z-image", scope: "global", installState: "missing" },
          ],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    expect(container.textContent).not.toContain("Built In");
    expect(container.textContent).not.toContain("Qwen Only");
    expect(container.textContent).not.toContain("Missing LoRA");

    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });

    // Add-on-demand LoRA picker (UI-refinement 3b): open the "Add LoRA" dropdown and click a
    // compatible row to add each LoRA. built_in (scope "builtin") doesn't count toward the
    // four-user cap, so we can add it plus four user LoRAs.
    const addLora = async (name) => {
      await act(async () => {
        [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
      });
      await act(async () => {
        [...document.body.querySelectorAll(".lora-pick-row")]
          .find((button) => button.textContent.includes(name))
          .click();
      });
    };
    await addLora("Built In");
    await addLora("Global Style");
    await addLora("Project Mira");
    await addLora("Third User");
    await addLora("Fourth User");

    expect(container.textContent).toContain("Built In");

    // At the four-user cap, the fifth user LoRA's Add row is disabled in the dropdown.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    const fifthRow = [...document.body.querySelectorAll(".lora-pick-row")].find((button) =>
      button.textContent.includes("Fifth User"),
    );
    expect(fifthRow.disabled).toBe(true);

    // The Show-incompatible toggle reveals incompatible LoRAs in the same dropdown.
    await act(async () => {
      document.body.querySelector('.lora-picker .checkline input[type="checkbox"]').click();
    });
    expect(
      [...document.body.querySelectorAll(".lora-pick-row")].some((button) =>
        button.textContent.includes("Qwen Only"),
      ),
    ).toBe(true);
    expect(container.textContent).not.toContain("Missing LoRA");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        loras: [
          expect.objectContaining({ id: "built_in", scope: "builtin", weight: 0.6 }),
          expect.objectContaining({ id: "global_style", scope: "global" }),
          expect.objectContaining({ id: "project_mira", scope: "project", files: ["mira.safetensors"] }),
          expect.objectContaining({ id: "third_user", scope: "global" }),
          expect.objectContaining({ id: "fourth_user", scope: "global" }),
        ],
      }),
    );
  });

  it("excludes cross-family LoRAs from a Kolors selection (sc-1927)", async () => {
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
          // Mirrors the Kolors manifest entry: family "kolors", LoRA families ["kolors"].
          imageModels: [
            { id: "kolors", name: "Kolors", type: "image", family: "kolors", loraCompatibility: { families: ["kolors"] }, capabilities: ["text_to_image"] },
          ],
          latestAssets: [],
          loras: [
            { id: "z_style", name: "Z Style", family: "z-image", scope: "global" },
            { id: "kolors_style", name: "Kolors Style", family: "kolors", scope: "global" },
          ],
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
      document.body.querySelector(".advanced-section-toggle").click();
    });
    // Open the Add-LoRA dropdown to inspect which LoRAs the model offers as compatible.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });

    const loraNames = [...document.body.querySelectorAll(".lora-pick-row strong")].map((node) => node.textContent);
    // A kolors-family model must not offer a z-image LoRA as compatible.
    expect(loraNames).toContain("Kolors Style");
    expect(loraNames).not.toContain("Z Style");
  });

  it("blocks image submit when a visible incompatible LoRA is selected", async () => {
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [{ id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "builtin" }],
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
      document.body.querySelector(".advanced-section-toggle").click();
    });
    // Reveal the incompatible LoRA in the dropdown, then add it via its Add row.
    await act(async () => {
      document.body.querySelector('.lora-picker .checkline input[type="checkbox"]').click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".lora-pick-row")]
        .find((button) => button.textContent.includes("Qwen Only"))
        .click();
    });

    const generate = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate");
    expect(container.textContent).toContain("Generate is blocked");
    expect(container.textContent).toContain("Qwen Only");
    expect(generate.disabled).toBe(true);

    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });
    await settle();

    expect(document.body.querySelector(".advanced-section.open")).toBeTruthy();
    expect(container.textContent).toContain("Qwen Only");

    await act(async () => {
      generate.click();
    });

    expect(createImageJob).not.toHaveBeenCalled();
  });

  it("applies preset defaults and seeds the preset's LoRAs into the visible picker for image jobs", async () => {
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [
            {
              id: "cinematic_detail",
              name: "Cinematic Detail",
              family: "z-image",
              scope: "builtin",
              defaultWeight: 0.55,
            },
          ],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              model: "z_image_turbo",
              workflow: "text_to_image",
              defaults: { count: 2, resolution: "1280x720", negativePrompt: "flat lighting" },
              prompt: { suffix: "cinematic lighting" },
              builtInLoras: [{ id: "cinematic_detail", weight: 0.4 }],
              ui: { description: "Balanced cinematic color, contrast, and detail." },
            },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    // sc-5875: presets are opt-in — select it explicitly before its defaults apply.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent.trim() === "Cinematic").click();
    });

    expect(container.textContent).toContain("Cinematic");
    expect(container.textContent).toContain("Balanced cinematic color, contrast, and detail.");
    expect(container.textContent).toContain("Adds: cinematic lighting");
    // The preset's installed LoRA is now seeded into the visible picker (not hidden), so the
    // guidance strip no longer claims it's "applied at generation" — it's a normal selection.
    expect(container.textContent).not.toContain("Preset LoRA applied at generation");
    // Open Advanced (where the LoRA rail lives): the preset's LoRA is a real selection sitting
    // at the preset's weight (0.4), ready to retune.
    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });
    expect(document.body.querySelector(".lora-slot-weight-value").textContent).toBe("0.40");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // The seeded preset LoRA rides in `loras` at its preset weight, and the client tells the
    // server it already resolved the preset's LoRAs so the server won't re-merge them.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        count: 2,
        width: 1280,
        height: 720,
        negativePrompt: "flat lighting",
        prompt: "A cinematic frame of a neon street at midnight",
        recipePresetId: "cinematic",
        presetLorasResolvedClientSide: true,
        loras: [expect.objectContaining({ id: "cinematic_detail", weight: 0.4 })],
        advanced: { resolution: "1280x720" },
      }),
    );

    // Deselecting the preset (→ None) removes the LoRA it seeded — the picker restores to the
    // pre-preset selection (empty here), so the seed is a reversible overlay, not a one-way dump.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent.trim() === "None").click();
    });
    expect(document.body.querySelector(".lora-slot-weight-value")).toBeNull();
  });

  it("prefills Image Studio from a saved generation recipe launch without reusing the seed", async () => {
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
              family: "z-image",
              limits: {
                resolutions: ["1024x1024", "1536x1024"],
                samplers: ["default", "euler"],
                schedulers: ["default", "shift"],
              },
            },
          ],
          latestAssets: [],
          loras: [{ id: "ready_style", name: "Ready Style", family: "z-image", scope: "global" }],
          launchRequest: {
            id: "recipe-replay-1",
            view: "Image",
            recipe: {
              mode: "text_to_image",
              model: "z_image_turbo",
              prompt: "mist over a glass atrium",
              negativePrompt: "flat lighting",
              seed: 1234,
              loras: [{ id: "ready_style", weight: 0.65 }],
              normalizedSettings: { width: 1536, height: 1024, count: 3 },
              rawAdapterSettings: {
                steps: 14,
                guidanceScale: 2.5,
                sampler: "euler",
                scheduler: "shift",
                schedulerShift: 2.2,
              },
            },
          },
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [{ id: "cinematic", name: "Cinematic", model: "z_image_turbo", workflow: "text_to_image" }],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "text_to_image",
        model: "z_image_turbo",
        prompt: "mist over a glass atrium",
        negativePrompt: "flat lighting",
        seed: null,
        count: 3,
        width: 1536,
        height: 1024,
        recipePresetId: null,
        loras: [expect.objectContaining({ id: "ready_style", weight: 0.65 })],
        advanced: expect.objectContaining({
          resolution: "1536x1024",
          steps: 14,
          guidanceScale: 2.5,
          sampler: "euler",
          scheduler: "shift",
          schedulerShift: 2.2,
        }),
      }),
    );
  });

  it("aspect picker reflects the selected model's trained buckets", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto", "1"],
          imageModels: [
            {
              id: "sensenova_u1_8b",
              name: "SenseNova-U1 8B",
              type: "image",
              family: "sensenova-u1",
              defaults: { resolution: "2048x2048" },
              limits: { resolutions: ["2048x2048", "2720x1536", "1536x2720"] },
            },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    const aspect = field(container, "Aspect");
    const optionValues = [...aspect.querySelectorAll("option")].map((option) => option.value);
    expect(optionValues).toEqual(["2048x2048", "2720x1536", "1536x2720"]);
    // 1024x1024 isn't a SenseNova bucket, so the picker snaps to the model default.
    expect(aspect.value).toBe("2048x2048");
  });

  it("surfaces model and preset first and lets image generation run with no preset", async () => {
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
          gpuOptions: ["auto", "1"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [{ id: "cinematic_detail", name: "Cinematic Detail", family: "z-image", scope: "builtin", presetManaged: true }],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              model: "z_image_turbo",
              workflow: "text_to_image",
              defaults: { count: 2, negativePrompt: "flat lighting" },
              builtInLoras: [{ id: "cinematic_detail" }],
            },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    // Primary generation controls sit in the settings bar under the composer (UI-refinement 2b),
    // no longer in a right-hand rail; power-user knobs stay behind Advanced.
    const settingsLabels = [...document.body.querySelectorAll(".settings-bar label")].map(
      (label) => label.childNodes[0]?.textContent.trim(),
    );
    expect(settingsLabels).toEqual(expect.arrayContaining(["Model", "Variations", "Aspect"]));
    expect(document.body.querySelector(".prompt-input")).not.toBeNull();
    expect(document.body.querySelector(".preset-chips").textContent).toContain("None");
    // sc-5875: a fresh studio defaults to None, so the preset's count (2) is NOT
    // auto-applied — Variations shows the studio default (4) until a preset is picked.
    expect(field(container, "Variations").value).toBe("4");

    // Selecting the preset applies its defaults (count 2).
    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent.trim() === "Cinematic").click();
    });
    await settle();
    expect(field(container, "Variations").value).toBe("2");
    expect(field(container, "GPU")).toBeUndefined();
    expect(container.textContent).not.toContain("LoRAs");

    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")]
        .find((chip) => chip.textContent.trim() === "None")
        .click();
    });
    await settle();

    // With no preset selected the guidance strip renders nothing (the visible controls
    // already describe the run); it only appears once a preset is active.
    expect(document.body.querySelector(".guidance-strip")).toBeNull();
    expect(field(container, "Variations").value).toBe("4");

    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });

    expect(field(container, "GPU")).not.toBeUndefined();
    expect(container.textContent).toContain("LoRAs");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        count: 4,
        negativePrompt: "",
        recipePresetId: null,
        loras: [],
      }),
    );
    expect(createImageJob.mock.calls[0][0]).not.toHaveProperty("stylePreset");
  });

  it("threads the Image Studio upscale controls into enabled image jobs", async () => {
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
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });

    expect(document.body.querySelector('.upscale-toggle input[type="checkbox"]').checked).toBe(false);
    expect(field(container, "Scale").disabled).toBe(true);
    expect(field(container, "Engine").disabled).toBe(true);

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledTimes(1);
    expect(createImageJob.mock.calls[0][0]).not.toHaveProperty("upscale");

    await act(async () => {
      document.body.querySelector('.upscale-toggle input[type="checkbox"]').click();
    });
    await changeField(field(container, "Scale"), "4");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        upscale: {
          enabled: true,
          factor: 4,
          engine: "real-esrgan",
        },
      }),
    );

    // AuraSR is dropped as an offered engine (sc-3668 / sc-5499) and SeedVR2 needs its platform
    // capability flag, so under the default capabilities the picker offers only Real-ESRGAN — there
    // is no `aura-sr` option to select.
    expect([...field(container, "Engine").querySelectorAll("option")].map((option) => option.value)).toEqual([
      "real-esrgan",
    ]);
  });

  it("submits a Kolors character job with the approved reference and IP-Adapter scale", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "kolors",
              name: "Kolors",
              type: "image",
              family: "kolors",
              capabilities: ["text_to_image", "edit_image", "character_image", "style_variations"],
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-1", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
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
    await settle();

    // The approved reference renders as a selected identity thumbnail.
    expect(container.textContent).toContain("Reference identity");
    expect(document.body.querySelector(".reference-thumb.active")).not.toBeNull();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        characterId: "char-1",
        referenceAssetId: "ref-1",
        count: 4,
        advanced: { resolution: "1024x1024", ipAdapterScale: 0.6 },
      }),
    );
  });

  it("exposes the InstantID Identity structure slider and submits its tuned defaults", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "instantid_realvisxl",
              name: "InstantID (RealVisXL)",
              type: "image",
              family: "sdxl",
              capabilities: ["character_image"],
              ui: {
                referenceStrengthDefault: 0.8,
                identityStructure: { label: "Identity structure", default: 0.8, min: 0.3, max: 1.0, step: 0.05 },
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-iid", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
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
    await settle();

    // The second (InstantID-only) slider renders; the strength slider is relabeled.
    expect(container.textContent).toContain("Identity structure");
    expect(container.textContent).toContain("Identity strength");
    expect(container.textContent).not.toContain("Reference strength");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // Tuned InstantID defaults flow through advanced: ipAdapterScale 0.8 (raised from
    // the global 0.6) + controlnetConditioningScale 0.8 (the IdentityNet lock).
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        referenceAssetId: "ref-1",
        advanced: { resolution: "1024x1024", ipAdapterScale: 0.8, controlnetConditioningScale: 0.8 },
      }),
    );
  });

  it("offers the InstantID View angle picker and submits the chosen angle", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "instantid_realvisxl",
              name: "InstantID (RealVisXL)",
              type: "image",
              family: "sdxl",
              capabilities: ["character_image"],
              ui: {
                referenceStrengthDefault: 0.8,
                identityStructure: { label: "Identity structure", default: 0.8, min: 0.3, max: 1.0, step: 0.05 },
                viewAngles: [
                  { id: "three_quarter_left", label: "Three-quarter left" },
                  { id: "left_profile", label: "Left profile" },
                ],
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-va", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
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
    await settle();

    // The dropdown lists "Match reference" plus the model's declared angles.
    const angleOptions = [...field(container, "View angle").options].map((option) => option.textContent);
    expect(angleOptions).toContain("Match reference");
    expect(angleOptions).toContain("Left profile");

    await changeField(field(container, "View angle"), "left_profile");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // The chosen angle rides advanced.viewAngle for the worker's landmark pack.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        referenceAssetId: "ref-1",
        advanced: expect.objectContaining({ viewAngle: "left_profile", ipAdapterScale: 0.8 }),
      }),
    );
  });

  it("surfaces the FLUX Variation slider alongside Reference strength and submits both knobs", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "flux_dev",
              name: "FLUX.1 [dev]",
              type: "image",
              family: "flux",
              capabilities: ["character_image"],
              ui: {
                // FLUX exposes BOTH the IP-Adapter reference-strength slider
                // (no override → global 0.6 default; the manifest sets 0.7 in
                // production but this fixture intentionally omits that to verify
                // the picker still renders correctly without a tuned default)
                // AND the Variation slider for true_cfg_scale (sc-2017).
                variationStrength: { label: "Variation", default: 4.0, min: 1.0, max: 10.0, step: 0.5 },
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-flux", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
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
    await settle();

    // Both sliders are visible for FLUX: Reference strength (IP-Adapter) AND
    // Variation (the trueCfgScale knob, which is FLUX's real-CFG lever since
    // base FLUX is guidance-distilled).
    expect(container.textContent).toContain("Reference strength");
    expect(container.textContent).toContain("Variation");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // Both knobs ride advanced: ipAdapterScale falls back to the global 0.6
    // default (no per-model override) and trueCfgScale follows the model's
    // declared default (4.0).
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        referenceAssetId: "ref-1",
        advanced: expect.objectContaining({ ipAdapterScale: 0.6, trueCfgScale: 4.0 }),
      }),
    );
  });

  it("surfaces the FLUX.2-dev Enhance prompt toggle (ui.promptEnhance) and submits advanced.enhancePrompt", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "flux2_dev",
              name: "FLUX.2 [dev]",
              type: "image",
              family: "flux2-dev",
              capabilities: ["text_to_image", "character_image"],
              // The model declares its built-in prompt upsampler — the manifest gates the toggle.
              ui: { promptEnhance: true },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-flux2dev", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
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
    await settle();

    // The toggle lives in the (collapsed-by-default) Advanced disclosure — open it first.
    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });

    // The manifest-driven "Enhance prompt" toggle renders for flux2_dev (off by default).
    const enhanceToggle = document.body.querySelector(".prompt-enhance-toggle input");
    expect(enhanceToggle).not.toBeNull();
    expect(enhanceToggle.checked).toBe(false);
    expect(container.textContent).toContain("Enhance prompt");

    // Turn it on, then generate.
    await act(async () => {
      enhanceToggle.click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // The toggle rides advanced.enhancePrompt; the worker threads it into the dev GenerationRequest.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        advanced: expect.objectContaining({ enhancePrompt: true }),
      }),
    );
  });

  it("hides the Reference strength slider for Qwen and submits trueCfgScale alone", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "qwen_image_edit_2511",
              name: "Qwen Image Edit (2511)",
              type: "image",
              family: "qwen-image",
              capabilities: ["character_image"],
              ui: {
                // Qwen-Image-Edit's variation knob is trueCfgScale; the IP-Adapter
                // reference-strength slider would be a no-op here (the worker
                // adapter doesn't read ipAdapterScale). Hide the slider AND drop
                // it from the submit payload (sc-2017).
                hideReferenceStrength: true,
                variationStrength: { label: "Variation", default: 4.0, min: 1.0, max: 10.0, step: 0.5 },
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-qwen", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
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
    await settle();

    // The no-op Reference-strength slider is hidden; only Variation renders.
    expect(container.textContent).not.toContain("Reference strength");
    expect(container.textContent).not.toContain("Identity strength");
    expect(container.textContent).toContain("Variation");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // advanced carries trueCfgScale but explicitly NOT ipAdapterScale.
    const lastCall = createImageJob.mock.calls.at(-1)[0];
    expect(lastCall.advanced.trueCfgScale).toBe(4.0);
    expect(lastCall.advanced).not.toHaveProperty("ipAdapterScale");
  });

  it("limits the character image model picker to reference-capable models", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [{ id: "char-1", name: "Mira", type: "person", looks: [], approvedReferences: [] }],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] },
            { id: "flux_dev", name: "FLUX", type: "image", capabilities: ["text_to_image"] },
          ],
          latestAssets: [],
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

    // Text mode lists every image model.
    let modelOptions = [...field(container, "Model").options].map((option) => option.textContent);
    expect(modelOptions).toContain("Kolors");
    expect(modelOptions).toContain("FLUX");
    const sceneSuggestions = [...document.body.querySelectorAll(".suggestion")].map((button) => button.textContent).join("|");

    await act(async () => {
      [...document.body.querySelectorAll(".mode-tabs button")].find((button) => button.textContent === "With character").click();
    });
    await settle();

    // Character mode hides models without a reference (IP-Adapter) engine.
    modelOptions = [...field(container, "Model").options].map((option) => option.textContent);
    expect(modelOptions).toContain("Kolors");
    expect(modelOptions).not.toContain("FLUX");

    // Suggestions swap to the variation-oriented set in character mode.
    const characterSuggestions = [...document.body.querySelectorAll(".suggestion")].map((button) => button.textContent).join("|");
    expect(characterSuggestions).not.toBe(sceneSuggestions);
  });

  it("seeds a character-aware default prompt from the character's notes", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            { id: "char-1", name: "Mira", type: "person", description: "A grizzled detective in a trench coat", looks: [], approvedReferences: [] },
          ],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
          launchRequest: { id: "launch-prompt-1", view: "Image", characterId: "char-1", mode: "character_image" },
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
    await settle();

    expect(document.body.querySelector(".prompt-input").value).toBe("A grizzled detective in a trench coat");
  });

  it("falls back to a type-specific default prompt when the character has no notes", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [{ id: "char-2", name: "Echo", type: "creature", looks: [], approvedReferences: [] }],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
          launchRequest: { id: "launch-prompt-2", view: "Image", characterId: "char-2", mode: "character_image" },
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
    await settle();

    expect(document.body.querySelector(".prompt-input").value).toBe("The creature in a new setting, varied pose, natural lighting");
  });

  it("keeps an edited prompt when switching into character mode", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            { id: "char-1", name: "Mira", type: "person", description: "A grizzled detective", looks: [], approvedReferences: [] },
          ],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
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

    await changeField(document.body.querySelector(".prompt-input"), "my own deliberate scene");
    await act(async () => {
      [...document.body.querySelectorAll(".mode-tabs button")].find((button) => button.textContent === "With character").click();
    });
    await changeField(field(container, "Character"), "char-1");
    await settle();

    // The user's wording survives entering character mode.
    expect(document.body.querySelector(".prompt-input").value).toBe("my own deliberate scene");
  });

  it("generates without a reference and warns when the character has no approved reference image", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [{ id: "char-2", name: "Echo", type: "creature", looks: [], approvedReferences: [] }],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
          launchRequest: { id: "launch-2", view: "Image", characterId: "char-2", mode: "character_image" },
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
    await settle();

    expect(container.textContent).toContain("No approved reference");
    expect(document.body.querySelector(".reference-thumb")).toBeNull();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        characterId: "char-2",
        referenceAssetId: null,
        advanced: { resolution: "1024x1024" },
      }),
    );
  });

  it("blocks image presets whose managed LoRAs do not match the selected model", async () => {
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
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ],
          latestAssets: [],
          loras: [
            {
              id: "qwen_detail",
              name: "Qwen Detail",
              family: "qwen-image",
              scope: "builtin",
              presetManaged: true,
            },
          ],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              workflow: "text_to_image",
              builtInLoras: [{ id: "qwen_detail", weight: 0.4 }],
            },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    // sc-5875: presets are opt-in — select it explicitly to exercise the managed-LoRA mismatch block.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent.trim() === "Cinematic").click();
    });

    const generate = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate");
    expect(container.textContent).toContain("Preset cannot run with Z-Image");
    expect(container.textContent).toContain("qwen_detail");
    expect(generate.disabled).toBe(true);

    await act(async () => {
      generate.click();
    });

    expect(createImageJob).not.toHaveBeenCalled();

    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });
    await changeField(field(container, "Model"), "qwen_image");
    await settle();

    expect(container.textContent).not.toContain("Preset cannot run with Qwen Image");
    expect(generate.disabled).toBe(false);

    await act(async () => {
      generate.click();
    });

    expect(createImageJob).toHaveBeenCalledWith(expect.objectContaining({ model: "qwen_image", recipePresetId: "cinematic" }));
  });

  it("blocks video presets whose managed LoRAs do not match the selected model", async () => {
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
          loras: [{ id: "wan_motion", name: "Wan Motion", family: "wan-video", scope: "builtin", presetManaged: true }],
          setPreviewAsset: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            {
              id: "dream_motion",
              name: "Dream Motion",
              workflow: "image_to_video",
              model: "ltx_2_3",
              builtInLoras: [{ id: "wan_motion" }],
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
              family: "ltx-video",
              capabilities: ["image_to_video"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              loraCompatibility: { families: ["ltx-video"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });

    // Video Studio opens on Text→Video (sc-5716); this preset targets image_to_video, so switch
    // to that tab to exercise the managed-LoRA mismatch surface.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image → Video").click();
    });
    await settle();

    // sc-5875: presets are opt-in — select it explicitly to exercise the managed-LoRA mismatch block.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent.trim() === "Dream Motion").click();
    });
    await settle();

    const generate = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Render clip");
    expect(container.textContent).toContain("Preset cannot run with LTX");
    expect(container.textContent).toContain("wan_motion");
    expect(generate.disabled).toBe(true);

    await act(async () => {
      generate.click();
    });

    expect(createVideoJob).not.toHaveBeenCalled();
  });

  it("offers a Wan A14B quantization selector and threads the choice into the video job (sc-1982)", async () => {
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
                id: "wan_2_2_t2v_14b",
                name: "Wan2.2 14B (T2V)",
                type: "video",
                family: "wan-video",
                capabilities: ["text_to_video"],
                defaults: { duration: 5, fps: 16, resolution: "832x480", quality: "balanced" },
                limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480"] },
                loraCompatibility: { families: ["wan-video"] },
                quantization: {
                  defaults: { mps: "gguf-q8_0", cuda: "gguf-q4_k_m" },
                  variants: {
                    "gguf-q8_0": { format: "gguf", label: "GGUF Q8_0 (near-lossless)" },
                    "gguf-q4_k_m": { format: "gguf", label: "GGUF Q4_K_M (smallest)" },
                  },
                },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll(".mode-control button")].find((button) => button.textContent === "Text → Video").click();
    });
    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });

    const quantSelect = field(container, "Quantization");
    expect(quantSelect).toBeTruthy();
    const optionLabels = [...quantSelect.querySelectorAll("option")].map((option) => option.textContent);
    expect(optionLabels).toContain("GGUF Q8_0 (near-lossless)");
    expect(optionLabels).toContain("GGUF Q4_K_M (smallest)");
    expect(optionLabels[0]).toContain("Auto");

    await changeField(quantSelect, "gguf-q4_k_m");
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Render clip").click();
    });

    expect(createVideoJob).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "wan_2_2_t2v_14b",
        advanced: expect.objectContaining({ quantization: "gguf-q4_k_m" }),
      }),
    );
  });

  it("surfaces compatible LoRAs in the Video Studio picker and sends the selection to the job", async () => {
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
            loras: [
              { id: "ltx_style", name: "LTX Style", family: "ltx-video", scope: "global", installState: "installed" },
              { id: "z_glow", name: "Z Glow", family: "z-image", scope: "global", installState: "installed" },
            ],
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            rememberLocalGenerationJob: () => {},
            requestedGpu: "auto",
            selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                family: "ltx-video",
                capabilities: ["image_to_video"],
                defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
                loraCompatibility: { families: ["ltx-video"] },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    // Video Studio opens on Text→Video (sc-5716); this LTX model is image_to_video-only, so switch
    // to that tab before exercising the LoRA picker + submit.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image → Video").click();
    });
    await settle();

    await act(async () => {
      document.body.querySelector(".advanced-section-toggle").click();
    });
    await settle();

    // Add-on-demand LoRA picker (UI-refinement 3b): open the dropdown. Only the ltx-video LoRA is
    // compatible; the z-image one is filtered out.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    const pickNames = [...document.body.querySelectorAll(".lora-pick-row strong")].map((node) => node.textContent);
    expect(pickNames).toContain("LTX Style");
    expect(pickNames).not.toContain("Z Glow");

    // Adding a LoRA drops in a slot with its weight slider, defaulting to the LoRA weight (0.8).
    await act(async () => {
      [...document.body.querySelectorAll(".lora-pick-row")]
        .find((button) => button.textContent.includes("LTX Style"))
        .click();
    });
    await settle();
    const weightSlider = document.body.querySelector(".lora-slot-weight input[type=range]");
    expect(weightSlider).toBeTruthy();
    // The weight slider is the shared bidirectional -2..2 range (LORA_WEIGHT_*), not the
    // old 0..2 — slider LoRAs run negative for the inverse direction, and the range must
    // match the studios, the Preset Manager, and the recipe-preset normalizer (a preset
    // LoRA stuck at the old center of 0 generated at scale 0 and looked "not applied").
    expect(weightSlider.getAttribute("min")).toBe("-2");
    expect(weightSlider.getAttribute("max")).toBe("2");
    expect(document.body.querySelector(".lora-slot-weight-value").textContent).toBe("0.80");
    await changeField(weightSlider, "0.5");

    const generate = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Render clip");
    expect(generate.disabled).toBe(false);

    await act(async () => {
      generate.click();
    });

    expect(createVideoJob).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "ltx_2_3",
        loras: [expect.objectContaining({ id: "ltx_style", weight: 0.5 })],
      }),
    );
  });

  it("always exposes the preset selector in the Video Studio even with no presets", async () => {
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
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
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
                capabilities: ["image_to_video", "text_to_video"],
                defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
                loraCompatibility: { families: ["ltx-video"] },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    expect(container.textContent).toContain("Style preset");
    const noneChip = [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent === "None");
    expect(noneChip).toBeTruthy();
  });

  it("keeps Qwen selected when applying a Qwen image preset", async () => {
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
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            { id: "qwen_detail", name: "Qwen Detail", model: "qwen_image", workflow: "text_to_image", defaults: { count: 1 } },
            { id: "cinematic", name: "Cinematic", model: "z_image_turbo", workflow: "text_to_image", defaults: { count: 4 } },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.textContent).toContain("Qwen Detail");
    expect(container.textContent).not.toContain("Cinematic");

    // sc-5875: presets are opt-in — select the Qwen preset; the model must stay on Qwen.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-chip")].find((chip) => chip.textContent.trim() === "Qwen Detail").click();
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(expect.objectContaining({ model: "qwen_image", recipePresetId: "qwen_detail" }));
  });

  it("offers SenseNova-U1 in edit mode via its edit_image capability", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["text_to_image"] },
            {
              id: "sensenova_u1_8b",
              name: "SenseNova-U1 8B",
              type: "image",
              family: "sensenova-u1",
              capabilities: ["text_to_image", "edit_image"],
              limits: { resolutions: ["2048x2048"] },
            },
            {
              id: "sensenova_u1_8b_fast",
              name: "SenseNova-U1 8B Fast",
              type: "image",
              family: "sensenova-u1",
              capabilities: ["text_to_image", "edit_image"],
              limits: { resolutions: ["2048x2048"] },
            },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll(".mode-tabs button")].find((button) => button.textContent === "Edit").click();
    });
    await settle();

    const modelValues = [...field(container, "Model").querySelectorAll("option")].map((option) => option.value);
    expect(modelValues).toContain("sensenova_u1_8b");
    // The distilled fast variant also edits, so it appears in the edit picker.
    expect(modelValues).toContain("sensenova_u1_8b_fast");
    // The text-to-image-only model is filtered out of the edit-mode picker.
    expect(modelValues).not.toContain("z_image_turbo");
    // The selected model resets to an edit-capable one, so Generate doesn't submit
    // the (filtered-out) text default and get rejected by the worker.
    expect(field(container, "Model").value).toBe("sensenova_u1_8b");
  });

  it("uses preset modes as the Image Studio picker surface", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["edit_image"] }],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              model: "z_image_turbo",
              workflow: "text_to_image",
              modes: ["text_to_image", "edit_image", "character_image"],
            },
            {
              id: "portrait_only",
              name: "Portrait Only",
              model: "z_image_turbo",
              workflow: "text_to_image",
              modes: ["character_image"],
            },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.textContent).toContain("Cinematic");
    expect(container.textContent).not.toContain("Portrait Only");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Edit").click();
    });
    await settle();

    expect(container.textContent).toContain("Cinematic");
    expect(container.textContent).not.toContain("Portrait Only");
  });

  it("drops variations to 1 in edit mode and restores 4 for text", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["text_to_image", "edit_image"] },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(field(container, "Variations").value).toBe("4");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Edit").click();
    });
    await settle();

    expect(field(container, "Variations").value).toBe("1");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Text").click();
    });
    await settle();

    expect(field(container, "Variations").value).toBe("4");
  });

});
