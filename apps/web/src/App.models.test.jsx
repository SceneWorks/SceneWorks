import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, eventUrl } from "./main.jsx";
import { liveElapsedSeconds } from "./formatting.js";
import { withModelManagerContext, FakeEventSource, response, settle, field, loraPanel, modelImportPanel, buttonInside, changeField, changeFile, selectModelTab, familyFilter } from "./main.testSupport.jsx";

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

  it("keeps video generation in the studio and shows local progress", async () => {
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
        return Promise.resolve(
          response([
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ]),
        );
      }
      if (path.endsWith("/video/jobs") && options.method === "POST") {
        const job = {
          id: "video-job-1",
          type: "video_generate",
          status: "queued",
          stage: "queued",
          progress: 0,
          elapsedSeconds: 0,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "Camera slowly pushes in while the scene comes alive" },
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
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Text → Video").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".video-studio form").requestSubmit();
    });
    await settle();

    expect(container.textContent).toContain("Generate Video");
    expect(container.textContent).toContain("Queued");
    expect(container.textContent).not.toContain("Jobs and GPUs");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Assets").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();

    expect(container.textContent).toContain("Generate Video");
    expect(container.textContent).toContain("Queued");
  });

  it("keeps model downloads on the Models page and shows local progress", async () => {
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
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            {
              id: "z_image_turbo",
              name: "Z-Image Turbo",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloadSizeLabel: "12 GB",
              downloads: [{ provider: "huggingface", repo: "Tongyi-MAI/Z-Image-Turbo" }],
            },
          ]),
        );
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      if (path.endsWith("/models/z_image_turbo/download") && options.method === "POST") {
        const job = {
          id: "download-job-1",
          type: "model_download",
          status: "downloading",
          stage: "downloading",
          progress: 0.5,
          elapsedSeconds: 12,
          requestedGpu: "auto",
          assignedGpu: "cpu",
          payload: { modelId: "z_image_turbo", modelName: "Z-Image Turbo" },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Download 12 GB").click();
    });
    await settle();

    expect(container.textContent).toContain("Model Import");
    expect(container.textContent).toContain("Downloading");
    expect(container.textContent).not.toContain("Jobs and GPUs");
  });

  it("flips the MLX button to ready when a model conversion completes", async () => {
    // `mlxConversionState` is derived server-side from the converted artifact on disk, so the
    // Models row only learns the conversion landed if the catalog is refetched when the
    // model_convert job completes. Without that refetch the button stays on "Convert to MLX"
    // until the app restarts.
    let conversionState = "needs_conversion";
    global.fetch.mockImplementation((url) => {
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
        return Promise.resolve(
          response([
            {
              id: "anima_2b",
              name: "Anima 2B",
              type: "image",
              family: "anima",
              installState: "installed",
              mlxConversionState: conversionState,
            },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();

    const mlxButton = () => [...container.querySelectorAll(".mlx-status button")][0];
    expect(mlxButton().textContent).toBe("Convert to MLX");

    conversionState = "converted";
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "convert-job-1",
          type: "model_convert",
          status: "completed",
          payload: { modelId: "anima_2b", modelName: "Anima 2B" },
        }),
      });
    });
    await settle();
    await settle();

    expect(mlxButton().textContent).toBe("MLX ready");
    expect(mlxButton().disabled).toBe(true);
    expect(container.textContent).toContain("Converted to MLX and ready.");
  });

  it("adds the imported model's row when a model import completes", async () => {
    // sc-10688, the same gap sc-10679 closed for `model_convert`. A completed model_import upserts
    // the model into `user.models.jsonc`, and the catalog is derived from that manifest server-side.
    // Without a refetch on the SSE `job.updated` the Models page never grows the imported row until
    // the app restarts. (The import route is 403-gated today under sc-7081; this locks the wiring in
    // so epic 7080's re-enable does not have to rediscover it.)
    let models = [{ id: "anima_2b", name: "Anima 2B", type: "image", family: "anima", installState: "installed" }];
    global.fetch.mockImplementation((url) => {
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
        return Promise.resolve(response(models));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();

    const cardTitles = () => [...container.querySelectorAll(".model-card-title strong")].map((node) => node.textContent);
    expect(cardTitles()).toEqual(["Anima 2B"]);

    models = [
      ...models,
      { id: "imported_checkpoint", name: "Imported Checkpoint", type: "image", family: "anima", installState: "installed" },
    ];
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "import-job-1",
          type: "model_import",
          status: "completed",
          payload: { modelId: "imported_checkpoint", modelName: "Imported Checkpoint" },
        }),
      });
    });
    await settle();
    await settle();

    expect(cardTitles()).toEqual(["Anima 2B", "Imported Checkpoint"]);
  });

  it("keeps LoRA imports on the Models page and shows local progress", async () => {
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
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      if (path.endsWith("/loras/import") && options.method === "POST") {
        const job = {
          id: "lora-import-job-1",
          type: "lora_import",
          status: "running",
          stage: "downloading",
          progress: 0.25,
          payload: { loraId: "detail_lora" },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    await selectModelTab("LoRAs");
    const panel = loraPanel(container);
    await act(async () => buttonInside(panel, "URL").click());
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });
    await settle();

    expect(container.textContent).toContain("LoRA imports in progress");
    expect(container.textContent).toContain("Running");
    expect(container.textContent).not.toContain("Jobs and GPUs");
  });

  it("refreshes the project LoRA overlay when a LoRA import completes", async () => {
    global.fetch.mockImplementation((url) => {
      const parsed = new URL(url);
      const path = parsed.pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    global.fetch.mockClear();
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-1",
          type: "lora_import",
          status: "completed",
          projectId: "project-1",
          payload: { loraId: "detail_lora" },
        }),
      });
    });
    await settle();
    await settle();

    const loraRequests = global.fetch.mock.calls
      .map(([url]) => new URL(url))
      .filter((url) => url.pathname.endsWith("/loras"));
    expect(loraRequests.some((url) => url.search === "")).toBe(true);
    expect(loraRequests.some((url) => url.search === "?projectId=project-1")).toBe(true);
  });

  it("refreshes the project LoRA overlay when LoRA training completes", async () => {
    global.fetch.mockImplementation((url) => {
      const parsed = new URL(url);
      const path = parsed.pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    global.fetch.mockClear();
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "train-job-1",
          type: "lora_train",
          status: "completed",
          projectId: "project-1",
          payload: { dryRun: false, outputName: "Portrait Set LoRA" },
          result: { loraRegistered: true, loraId: "lora_portrait_set" },
        }),
      });
    });
    await settle();
    await settle();

    const loraRequests = global.fetch.mock.calls
      .map(([url]) => new URL(url))
      .filter((url) => url.pathname.endsWith("/loras"));
    expect(loraRequests.some((url) => url.search === "")).toBe(true);
    expect(loraRequests.some((url) => url.search === "?projectId=project-1")).toBe(true);
  });

  it("shows the global banner for failed LoRA imports on the Models page", async () => {
    global.fetch.mockImplementation((url) => {
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
        return Promise.resolve(
          response([
            { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    await selectModelTab("LoRAs");
    await changeField(familyFilter(container), "z-image");
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-1",
          type: "lora_import",
          status: "failed",
          error: "Import worker crashed",
          payload: { loraId: "qwen_detail", family: "qwen-image" },
        }),
      });
    });
    await settle();

    expect(container.textContent).toContain("lora import: Import worker crashed");
    expect(container.textContent).toContain("1 LoRA import is hidden by this family filter.");
  });

  it("clears a stale LoRA import banner when a later import completes", async () => {
    global.fetch.mockImplementation((url) => {
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
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-1",
          type: "lora_import",
          status: "failed",
          createdAt: "2026-05-18T00:00:00Z",
          error: "LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest",
          payload: { loraId: "detail_lora", family: "z-image" },
        }),
      });
    });
    await settle();
    expect(container.textContent).toContain("lora import: LoRA manifestPath must target");

    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-2",
          type: "lora_import",
          status: "completed",
          createdAt: "2026-05-18T00:00:01Z",
          payload: { loraId: "detail_lora", family: "z-image" },
        }),
      });
    });
    await settle();

    expect(container.textContent).not.toContain("lora import: LoRA manifestPath must target");
  });

  // sc-4198 regression: notices are kept per-kind, so a LoRA-import banner and a
  // LoRA-train banner coexist, and clearing one (here, the LoRA import's later
  // completion) leaves the unrelated, still-relevant notice untouched — unlike the
  // old single `error` string where any clear wiped everything.
  it("keeps an unrelated notice when a LoRA import notice clears", async () => {
    global.fetch.mockImplementation((url) => {
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
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    // A LoRA import fails (kind "lora-import") and a LoRA training run completes but
    // fails to register (kind "lora-train") — both banners show at once.
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-1",
          type: "lora_import",
          status: "failed",
          createdAt: "2026-05-18T00:00:00Z",
          error: "manifest write failed",
          payload: { loraId: "detail_lora", family: "z-image" },
        }),
      });
    });
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-train-job-1",
          type: "lora_train",
          status: "completed",
          createdAt: "2026-05-18T00:00:01Z",
          payload: { dryRun: false },
          result: { loraRegistered: false, loraRegistrationError: "registry locked" },
        }),
      });
    });
    await settle();
    expect(container.textContent).toContain("lora import: manifest write failed");
    expect(container.textContent).toContain("lora training: registry locked");

    // A later LoRA import completes — only the lora-import banner clears; the
    // unrelated lora-train notice must survive.
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-2",
          type: "lora_import",
          status: "completed",
          createdAt: "2026-05-18T00:00:02Z",
          payload: { loraId: "detail_lora", family: "z-image" },
        }),
      });
    });
    await settle();

    expect(container.textContent).not.toContain("lora import: manifest write failed");
    expect(container.textContent).toContain("lora training: registry locked");
  });

  it("rejects oversized LoRA uploads before posting from the Models page", async () => {
    let importCalls = 0;
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
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/loras/import") && options.method === "POST") {
        importCalls += 1;
        return Promise.resolve(response({ id: "should-not-create" }));
      }
      return Promise.resolve(response([]));
    });
    const loraFile = new File(["lora"], "too-large.safetensors", { type: "application/octet-stream" });
    Object.defineProperty(loraFile, "size", { configurable: true, value: 2 * 1024 * 1024 * 1024 + 1 });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    await selectModelTab("LoRAs");
    await act(async () => {
      buttonInside(loraPanel(container), "Upload").click();
    });
    const panel = loraPanel(container);
    await changeFile(field(panel, "LoRA File"), loraFile);
    await act(async () => {
      buttonInside(loraPanel(container), "Queue Import").click();
    });

    expect(container.textContent).toContain("Uploaded LoRA file exceeds the 2GB limit");
    expect(importCalls).toBe(0);
  });

  it("keeps Preset Manager LoRA acquisition on the Models page", async () => {
    let importCalls = 0;
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
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/recipe-presets")) {
        return Promise.resolve(
          response([{ id: "moody", name: "Moody", scope: "global", workflow: "text_to_image", model: "z_image_turbo" }]),
        );
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response([]));
      }
      if (path.endsWith("/loras/import") && options.method === "POST") {
        importCalls += 1;
        return Promise.resolve(response({ id: "lora-import-job-1", type: "lora_import", status: "running" }));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Presets").click();
    });
    await settle();

    expect(container.textContent).toContain("Reusable generation setups");

    // The LoRA stack (and its "go acquire one" empty state) lives in the editor now,
    // so open the preset before asserting on it.
    await act(async () => {
      [...document.body.querySelectorAll(".preset-card")]
        .find((card) => card.textContent.includes("Moody"))
        .querySelector(".secondary-action")
        .click();
    });
    await settle();

    expect(container.textContent).toContain("No uploaded LoRAs yet. Manage LoRAs on the Models page.");
    expect(container.textContent).not.toContain("Import LoRA");
    expect(container.textContent).not.toContain("Queue Import");
    expect(field(container, "Source URL")).toBeUndefined();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Open Models").click();
    });
    await settle();
    await selectModelTab("LoRAs");

    expect(container.textContent).toContain("Models");
    expect(container.textContent).toContain("Import LoRA");
    expect(importCalls).toBe(0);
  });

  it("queues LoRA URL imports from the Models page", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "detail_lora" } }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    const panel = loraPanel(container);
    await act(async () => buttonInside(panel, "URL").click());
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await changeField(field(panel, "Name"), "Detail LoRA");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceUrl: "https://example.com/loras/detail.safetensors",
        name: "Detail LoRA",
        scope: "global",
      }),
    );
    expect(onImportLora.mock.calls[0][0]).not.toHaveProperty("family");
    expect(container.textContent).toContain("LoRA import queued for detail_lora.");
  });

  it("keeps Models LoRA import family independent from the list filter", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "detail_lora" } }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [
            { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    expect(familyFilter(container).value).toBe("all");
    await changeField(familyFilter(container), "qwen-image");
    const panel = loraPanel(container);
    await act(async () => buttonInside(panel, "URL").click());
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportLora.mock.calls[0][0]).not.toHaveProperty("family");

    await changeField(field(panel, "Family"), "z-image");
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportLora.mock.calls[1][0]).toEqual(
      expect.objectContaining({
        family: "z-image",
      }),
    );
  });

  it("clears an explicit Models LoRA import family when the model family disappears", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "detail_lora" } }));
    const renderScreen = (models) =>
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models,
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );

    root = createRoot(container);
    await act(async () => {
      renderScreen([
        { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" },
        { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
      ]);
    });

    await selectModelTab("LoRAs");
    const panel = loraPanel(container);
    await changeField(field(panel, "Family"), "qwen-image");
    expect(field(panel, "Family").value).toBe("qwen-image");

    await act(async () => {
      renderScreen([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]);
    });

    expect(field(loraPanel(container), "Family").value).toBe("");
  });

  it("shows model download size estimates and unavailable states before download", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [
            {
              id: "z_image_turbo",
              name: "Z-Image Turbo",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloadSizeLabel: "30.6 GB",
              downloadSizeEstimated: true,
              downloads: [{ provider: "huggingface", repo: "Tongyi-MAI/Z-Image-Turbo" }],
            },
            {
              id: "local_unknown",
              name: "Unknown Size",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloads: [{ provider: "huggingface", repo: "owner/unknown" }],
            },
            {
              id: "exact_size",
              name: "Exact Size",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloadSizeLabel: "8.0 GB",
              downloadSizeEstimated: false,
              downloads: [{ provider: "huggingface", repo: "owner/exact" }],
            },
          ],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    // The card footer surfaces each model's download size (the old repo/size <dl> is gone).
    expect(container.textContent).toContain("~30.6 GB");
    expect(container.textContent).toContain("8.0 GB");
    expect(container.textContent).toContain("Unavailable");
    expect([...document.body.querySelectorAll("button")].some((button) => button.textContent === "Download ~30.6 GB")).toBe(true);
    expect([...document.body.querySelectorAll("button")].some((button) => button.textContent === "Download 8.0 GB")).toBe(true);
    expect(container.textContent).not.toContain("~8.0 GB");
  });

  it("marks listed LoRAs unavailable when the backend reports missing install state", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [
            { id: "ready_style", name: "Ready Style", family: "z-image", scope: "global", installState: "installed" },
            { id: "broken_style", name: "Broken Style", family: "z-image", scope: "global", installState: "missing" },
          ],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image", installState: "installed" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    const rows = [...document.body.querySelectorAll(".lora-row")];
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("Ready Style");
    expect(rows[0].textContent).toContain("installed");
    expect(rows[0].classList.contains("unavailable")).toBe(false);
    expect(rows[1].textContent).toContain("Broken Style");
    // A registered user LoRA with no local files reads "unavailable" and carries the
    // danger-tinted row treatment.
    expect(rows[1].textContent).toContain("unavailable");
    expect(rows[1].classList.contains("unavailable")).toBe(true);
  });

  it("confirms and deletes models and LoRAs from the Models page", async () => {
    const onDeleteModel = vi.fn(async () => ({ removedManifestEntry: true, warnings: ["Recipe presets reference this model: Moody"] }));
    const onDeleteLora = vi.fn(async () => ({ removedManifestEntry: true, warnings: ["Recipe presets reference this lora: Moody"] }));
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [{ id: "ready_style", name: "Ready Style", family: "z-image", scope: "global", installState: "installed", removable: true }],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image", installState: "installed", removable: true }],
          onDeleteLora,
          onDeleteModel,
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
          recipePresets: [{ id: "moody", name: "Moody", model: "z_image_turbo", loras: [{ id: "ready_style" }] }],
        }),
      );
    });

    await act(async () => {
      document.body.querySelector(".model-card .danger-action").click();
    });

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Delete model "Z-Image Turbo"?'));
    expect(confirm.mock.calls[0][0]).toContain("Referenced by presets: Moody.");
    expect(onDeleteModel).toHaveBeenCalledWith(expect.objectContaining({ id: "z_image_turbo" }));
    expect(container.textContent).toContain("Removed the registry entry for Z-Image Turbo.");

    await selectModelTab("LoRAs");
    await act(async () => {
      document.body.querySelector(".lora-row .danger-action").click();
    });

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Delete lora "Ready Style"?'));
    expect(onDeleteLora).toHaveBeenCalledWith(expect.objectContaining({ id: "ready_style" }));
    expect(container.textContent).toContain("Removed the registry entry for Ready Style.");
  });

  it("advances elapsed seconds for active job snapshots between server updates", () => {
    const job = {
      id: "image-job-1",
      status: "running",
      elapsedSeconds: 57,
      startedAt: "2026-05-18T20:00:00Z",
    };

    expect(liveElapsedSeconds(job, Date.parse("2026-05-18T20:02:05Z"))).toBe(125);
  });

  it("resets the Models LoRA form after queueing and allows another import while one is pending", async () => {
    const onImportLora = vi.fn(async (payload) => ({
      id: `lora-import-job-${onImportLora.mock.calls.length}`,
      type: "lora_import",
      status: "queued",
      progress: 0,
      payload: { ...payload, loraId: `detail_lora_${onImportLora.mock.calls.length}` },
    }));

    function Harness() {
      const [jobs, setJobs] = React.useState([]);
      async function importLora(payload) {
        const job = await onImportLora(payload);
        setJobs((items) => [job, ...items]);
        return job;
      }
      return withModelManagerContext({
        activeProject: { id: "project-1", name: "Noir" },
        jobs,
        loras: [],
        models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
        onDownloadModel: () => {},
        onImportLora: importLora,
        onOpenQueue: () => {},
      });
    }

    root = createRoot(container);
    await act(async () => {
      root.render(<Harness />);
    });

    await selectModelTab("LoRAs");
    const panel = () => loraPanel(container);
    await act(async () => buttonInside(panel(), "URL").click());
    await changeField(field(panel(), "Source URL"), "https://example.com/loras/one.safetensors");
    await changeField(field(panel(), "Name"), "First Detail");
    await act(async () => {
      buttonInside(panel(), "Queue Import").click();
    });

    expect(field(panel(), "Source URL").value).toBe("");
    expect(field(panel(), "Name").value).toBe("");
    expect(container.textContent).toContain("LoRA imports");
    expect(container.textContent).toContain("detail_lora_1");
    expect(container.textContent).not.toContain("No user LoRAs");

    await changeField(field(panel(), "Source URL"), "https://example.com/loras/two.safetensors");
    await act(async () => {
      buttonInside(panel(), "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("detail_lora_2");
  });

  it("queues LoRA file uploads from the Models page with project scope", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "uploaded_detail" } }));
    const loraFile = new File(["lora"], "detail.safetensors", { type: "application/octet-stream" });
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "running",
              stage: "downloading",
              progress: 0.3,
              payload: { loraId: "existing_import" },
            },
          ],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    await act(async () => {
      buttonInside(loraPanel(container), "Upload").click();
    });
    const panel = loraPanel(container);
    await changeField(field(panel, "Scope"), "project");
    await changeFile(field(panel, "LoRA File"), loraFile);
    await act(async () => {
      buttonInside(loraPanel(container), "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledWith(
      expect.objectContaining({
        file: loraFile,
        scope: "project",
      }),
    );
    expect(onImportLora.mock.calls[0][0]).not.toHaveProperty("family");
    expect(container.textContent).toContain("LoRA Import");
    expect(container.textContent).toContain("Running");
  });

  it("keeps failed Models LoRA imports visible inline", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "failed",
              stage: "failed",
              progress: 0.4,
              error: "Adapter crashed",
              payload: { loraId: "broken_detail", family: "z-image" },
            },
          ],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    expect(container.textContent).toContain("LoRA imports");
    expect(container.textContent).toContain("broken_detail");
    expect(container.textContent).toContain("Adapter crashed");
    expect(container.textContent).not.toContain("No user LoRAs");
  });

  it("hides failed Models LoRA imports superseded by a completed retry", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [
            {
              id: "lora-import-job-2",
              type: "lora_import",
              status: "completed",
              stage: "completed",
              progress: 1,
              createdAt: "2026-05-18T00:00:01Z",
              payload: { loraId: "detail_lora", family: "z-image" },
            },
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "failed",
              stage: "failed",
              progress: 0.4,
              createdAt: "2026-05-18T00:00:00Z",
              error: "LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest",
              payload: { loraId: "detail_lora", family: "z-image" },
            },
          ],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    expect(container.textContent).not.toContain("LoRA manifestPath must target");
    expect(container.textContent).not.toContain("LoRA imports");
    expect(container.textContent).toContain("No user LoRAs yet");
  });

  it("shows Models page LoRA import errors and resets the queueing state", async () => {
    const onImportLora = vi.fn(async () => {
      throw new Error("LoRA sourceUrl must use http or https");
    });
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    const panel = loraPanel(container);
    await act(async () => buttonInside(panel, "URL").click());
    await changeField(field(panel, "Source URL"), "file:///tmp/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(container.textContent).toContain("LoRA sourceUrl must use http or https");
    expect(buttonInside(loraPanel(container), "Queue Import").disabled).toBe(false);
  });

  it("hides the model import form while uploads are disabled pending conversion support (sc-7081)", async () => {
    // The model upload/import form is gated off on every platform (MODEL_IMPORT_ENABLED)
    // until the compatibility + conversion pipeline (epic 7080) lands — an imported
    // checkpoint has no runnable engine today, and the API refuses the request too.
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: null,
          jobs: [],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onImportModel: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    expect(modelImportPanel(container)).toBeNull();
    expect(container.textContent).not.toContain("Import model");
    // The image model's card is on the default (Image) tab.
    expect(container.textContent).toContain("Z-Image Turbo");
    // LoRA import is unaffected — its form is on the LoRAs tab.
    await selectModelTab("LoRAs");
    expect(loraPanel(container)).not.toBeNull();
  });

  it("renders unassociated models with a needs-family badge", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: null,
          jobs: [],
          loras: [],
          models: [{ id: "imported_custom", name: "Imported Custom", type: "image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onImportModel: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    expect(container.textContent).toContain("needs family");
    expect(container.textContent).toContain("unassociated");
  });

  it("shows in-progress model imports inline", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: null,
          jobs: [
            {
              id: "model-import-job-1",
              type: "model_import",
              status: "downloading",
              stage: "downloading",
              progress: 0.42,
              payload: { modelId: "custom_model", name: "Custom Model" },
            },
          ],
          loras: [],
          models: [],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onImportModel: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    await selectModelTab("LoRAs");
    expect(container.textContent).toContain("Model imports in progress");
    expect(container.textContent).toContain("Model Import");
    expect(container.textContent).toContain("Downloading");
  });

  it("adds the SSE ticket as a query parameter", () => {
    expect(eventUrl("/api/v1/jobs/events", "stream-ticket")).toContain("ticket=stream-ticket");
  });

});
