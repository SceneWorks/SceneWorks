import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { withImageStudioContext, FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { styleTextForId } from "./data/styleCatalog.js";
import { composeStyledPrompt } from "./styleComposer.js";

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

  it("gates Image Studio behind a model download when no image model is present", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          purgeAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();
    // The studio form is replaced by the availability gate.
    expect(container.textContent).toContain("Image Studio needs an image model");
    expect(document.body.querySelector(".model-availability-gate")).not.toBeNull();
    expect(document.body.querySelector(".studio-shell")).toBeNull();
  });

  it("keeps image generation in the studio and shows local progress", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.35,
          elapsedSeconds: 4,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight" },
        };
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    expect(container.textContent).toContain("Generate Image");
    expect(container.textContent).toContain("Running");
    expect(container.textContent).not.toContain("Jobs and GPUs");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Assets").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();

    expect(container.textContent).toContain("Generate Image");
    expect(container.textContent).toContain("Running");
  });

  it("replays a library asset recipe through fullscreen preview into Image Studio with a random seed", async () => {
    const createdJobs = [];
    const imagePayloads = [];
    const generatedAsset = {
      id: "asset-replay",
      projectId: "project-1",
      generationSetId: "genset-replay",
      type: "image",
      displayName: "Atrium still",
      file: { path: "assets/images/atrium.png", mimeType: "image/png" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
      generationSet: {
        recipe: {
          mode: "text_to_image",
          model: "z_image_turbo",
          prompt: "mist over a glass atrium",
          negativePrompt: "flat lighting",
          seed: 1234,
          loras: [{ id: "ready_style", weight: 0.65 }],
          normalizedSettings: { width: 1536, height: 1024, count: 3 },
          rawAdapterSettings: { steps: 14, guidanceScale: 2.5 },
        },
      },
      recipe: { prompt: "asset fallback prompt" },
    };
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(response([generatedAsset]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              family: "z-image",
              limits: { resolutions: ["1024x1024", "1536x1024"] },
            },
          ]),
        );
      }
      if (path.endsWith("/loras")) {
        return Promise.resolve(response([{ id: "ready_style", name: "Ready Style", family: "z-image", scope: "global" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const payload = JSON.parse(options.body);
        imagePayloads.push(payload);
        const job = {
          id: "image-job-replay",
          type: "image_generate",
          status: "queued",
          stage: "queued",
          progress: 0,
          elapsedSeconds: 0,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: payload.requestedGpu,
          payload,
        };
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Assets").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".asset-tile").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use this recipe").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    expect(imagePayloads[0]).toMatchObject({
      projectId: "project-1",
      projectName: "Noir",
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
      }),
    });
  });

  it("replays a STYLED recipe: re-selects the style, recomposes the identical prompt, no double-wrap (sc-13132)", async () => {
    const createdJobs = [];
    const imagePayloads = [];
    const STYLE_ID = "ghibli-style";
    const RAW_PROMPT = "a fox by lantern light";
    // The prompt the ORIGINAL styled generate produced (the composer runs client-side). The recipe
    // records this verbatim in `prompt`, plus the styleId + raw prompt under rawAdapterSettings.
    const composedPrompt = composeStyledPrompt({ styleText: styleTextForId(STYLE_ID), userPrompt: RAW_PROMPT });
    const generatedAsset = {
      id: "asset-styled",
      projectId: "project-1",
      generationSetId: "genset-styled",
      type: "image",
      displayName: "Styled still",
      file: { path: "assets/images/styled.png", mimeType: "image/png" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
      generationSet: {
        recipe: {
          mode: "text_to_image",
          model: "z_image_turbo",
          prompt: composedPrompt,
          negativePrompt: "",
          seed: 42,
          loras: [],
          normalizedSettings: { width: 1024, height: 1024, count: 1 },
          rawAdapterSettings: { styleId: STYLE_ID, stylePrompt: RAW_PROMPT },
        },
      },
      recipe: { prompt: composedPrompt },
    };
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(response([generatedAsset]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              family: "z-image",
              limits: { resolutions: ["1024x1024", "1536x1024"] },
            },
          ]),
        );
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const payload = JSON.parse(options.body);
        imagePayloads.push(payload);
        const job = {
          id: "image-job-styled",
          type: "image_generate",
          status: "queued",
          stage: "queued",
          progress: 0,
          elapsedSeconds: 0,
          projectId: "project-1",
          projectName: "Noir",
          payload,
        };
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Assets").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".asset-tile").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use this recipe").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    // The re-run reproduces the IDENTICAL composed prompt — the picker was re-selected (else the
    // prompt would be the bare RAW prompt) AND the box held the RAW prompt (else recomposing the
    // already-composed prompt would nest a second `Style:` block).
    expect(imagePayloads[0].prompt).toBe(composedPrompt);
    expect(imagePayloads[0].prompt.match(/^Style:/gm)?.length ?? 0).toBe(1);
    // The style id round-trips onto the fresh recipe so the NEXT replay stays reproducible too.
    expect(imagePayloads[0].advanced.styleId).toBe(STYLE_ID);
    expect(imagePayloads[0].advanced.stylePrompt).toBe(RAW_PROMPT);
    expect(imagePayloads[0].presetPromptResolvedClientSide).toBe(true);
  });

  it("shows completed image batch items before the whole job finishes", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.35,
          elapsedSeconds: 4,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight", count: 4 },
        };
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    const partialAsset = {
      id: "asset-1",
      projectId: "project-1",
      generationSetId: "genset-1",
      type: "image",
      displayName: "Generated #1",
      file: { path: "assets/images/generated-1.png", mimeType: "image/png" },
      status: { favorite: false, rejected: false, trashed: false },
    };
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          ...createdJobs[0],
          status: "saving",
          stage: "saving",
          progress: 0.82,
          result: {
            generationSetId: "genset-1",
            assetIds: ["asset-1"],
            assets: [partialAsset],
            expectedCount: 4,
          },
        }),
      });
    });
    await settle();

    expect(document.body.querySelector(".worker-progress-card__thumb-media")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated-1.png",
    );
    // 1 completed asset + 3 pending slots = 4 total cells; skeletons fill the gaps.
    expect(document.body.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBe(3);
  });

  it("reconstructs running image batch slots from partial asset records", async () => {
    const createdJobs = [];
    let currentAssets = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(response(currentAssets));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.35,
          elapsedSeconds: 4,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight", count: 4 },
        };
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    currentAssets = [
      {
        id: "asset-1",
        projectId: "project-1",
        generationSetId: "genset-1",
        type: "image",
        displayName: "Generated #1",
        file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
        status: { favorite: false, rejected: false, trashed: false },
      },
    ];
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          ...createdJobs[0],
          status: "running",
          stage: "generating",
          progress: 0.48,
          message: "Running Z-Image 2 of 4.",
          result: {
            generationSetId: "genset-1",
            assetIds: ["asset-1"],
            expectedCount: 4,
          },
        }),
      });
    });
    await settle();

    expect(container.textContent).toContain("Running Z-Image 2 of 4.");
    expect(document.body.querySelector(".worker-progress-card__thumb-media")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated_0001.png",
    );
    // 1 completed thumbnail + 3 pending skeleton slots = 4 total cells.
    expect(document.body.querySelectorAll(".worker-progress-card__thumb-cell:not(.skeleton)").length).toBe(1);
    expect(document.body.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBe(3);
  });

  it("evicts the per-job asset-refresh bookkeeping once a job goes terminal (sc-8944)", async () => {
    const createdJobs = [];
    let assetRefreshFetches = 0;
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/assets")) {
        assetRefreshFetches += 1;
        return Promise.resolve(response([]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.35,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight", count: 1 },
        };
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    // A completed generation-set fires the initial assets refresh and records the
    // per-job dedupe entry (job.id -> generationSetId).
    function emitCompleted(generationSetId) {
      return act(async () => {
        FakeEventSource.instances[0].listeners["job.updated"]({
          data: JSON.stringify({
            ...createdJobs[0],
            status: "completed",
            stage: "completed",
            progress: 1,
            result: { generationSetId, assetIds: [] },
          }),
        });
      });
    }

    await emitCompleted("genset-1");
    await settle();
    const afterFirst = assetRefreshFetches;
    expect(afterFirst).toBeGreaterThan(0);

    // Re-emitting the SAME generation-set for the SAME (now terminal) job id must
    // refresh again: the terminal eviction (sc-8944) cleared the dedupe entry, so the
    // set is treated as new. Without eviction the entry would still dedupe this away.
    await emitCompleted("genset-1");
    await settle();
    expect(assetRefreshFetches).toBeGreaterThan(afterFirst);
  });

  it("shows local generation failures without duplicating the global banner", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.25,
          elapsedSeconds: 3,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight" },
        };
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({ ...createdJobs[0], status: "failed", stage: "failed", progress: 0.25, error: "Adapter crashed" }),
      });
    });
    await settle();

    expect(container.textContent).toContain("Adapter crashed");
    expect(container.textContent).not.toContain("image generate: Adapter crashed");
  });

});
