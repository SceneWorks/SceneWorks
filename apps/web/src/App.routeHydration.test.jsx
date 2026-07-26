import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

function deferred() {
  let resolve;
  const promise = new Promise((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function requestPath(url) {
  return new URL(url, window.location.origin).pathname;
}

function requestCounts() {
  return global.fetch.mock.calls.reduce((counts, [url]) => {
    const path = requestPath(url);
    counts[path] = (counts[path] ?? 0) + 1;
    return counts;
  }, {});
}

describe("route-owned App hydration (sc-14783)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  function installFetch(overrides = {}) {
    global.fetch = vi.fn((url, options = {}) => {
      const requestUrl = new URL(url);
      const path = requestUrl.pathname;
      if (overrides[path]) return overrides[path](requestUrl, options);
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
        return Promise.resolve(response([{ id: "project-a", name: "Project A" }]));
      }
      if (path.endsWith("/training/targets")) {
        return Promise.resolve(response({ schemaVersion: 1, targets: [] }));
      }
      if (path.endsWith("/training/presets")) {
        return Promise.resolve(response({ schemaVersion: 1, presets: [] }));
      }
      if (path.endsWith("/jobs") || path.endsWith("/workers")) {
        return Promise.resolve(response([]));
      }
      return Promise.resolve(response([]));
    });
  }

  async function renderApp() {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
  }

  async function navigate(label) {
    const button = [...container.querySelectorAll(".nav-item")].find(
      (candidate) => candidate.textContent.trim() === label,
    );
    expect(button, `navigation button ${label}`).toBeTruthy();
    await act(async () => {
      button.click();
    });
    await settle();
  }

  it("hydrates late global/project domains once, on first consumer navigation", async () => {
    installFetch();
    await renderApp();

    let counts = requestCounts();
    expect(counts["/api/v1/projects/project-a/assets"]).toBe(1);
    [
      "/api/v1/models",
      "/api/v1/loras",
      "/api/v1/recipe-presets",
      "/api/v1/prompt-batches",
      "/api/v1/training/targets",
      "/api/v1/training/presets",
      "/api/v1/projects/project-a/training/datasets",
      "/api/v1/projects/project-a/timelines",
      "/api/v1/projects/project-a/voices",
      "/api/v1/projects/project-a/characters",
      "/api/v1/projects/project-a/person-tracks",
    ].forEach((path) => expect(counts[path]).toBeUndefined());

    await navigate("Train");
    counts = requestCounts();
    expect(counts["/api/v1/models"]).toBe(1);
    expect(counts["/api/v1/loras"]).toBeUndefined();
    expect(counts["/api/v1/training/targets"]).toBe(1);
    expect(counts["/api/v1/training/presets"]).toBe(1);
    expect(counts["/api/v1/projects/project-a/training/datasets"]).toBe(1);

    await navigate("Video Editor");
    await navigate("Audio");
    await navigate("Characters");
    await navigate("Image");
    await navigate("Video");
    counts = requestCounts();
    expect(counts["/api/v1/loras"]).toBe(1);
    expect(counts["/api/v1/projects/project-a/timelines"]).toBe(1);
    expect(counts["/api/v1/recipe-presets"]).toBe(1);
    expect(counts["/api/v1/projects/project-a/voices"]).toBe(1);
    expect(counts["/api/v1/projects/project-a/characters"]).toBe(1);
    expect(counts["/api/v1/prompt-batches"]).toBe(1);
    expect(counts["/api/v1/projects/project-a/person-tracks"]).toBe(1);

    await navigate("Assets");
    await navigate("Train");
    counts = requestCounts();
    expect(counts["/api/v1/projects/project-a/assets"]).toBe(1);
    expect(counts["/api/v1/models"]).toBe(1);
    expect(counts["/api/v1/training/targets"]).toBe(1);
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  it("hydrates and renders character training datasets on first Characters navigation", async () => {
    installFetch({
      "/api/v1/projects/project-a/characters": () =>
        Promise.resolve(response([{ id: "character-a", name: "Mara", projectId: "project-a" }])),
      "/api/v1/projects/project-a/training/datasets": () =>
        Promise.resolve(response([{
          id: "dataset-a",
          name: "Mara portrait set",
          characterId: "character-a",
          itemCount: 4,
          status: "draft",
        }])),
    });
    await renderApp();

    expect(requestCounts()["/api/v1/projects/project-a/training/datasets"]).toBeUndefined();
    await navigate("Characters");
    expect(requestCounts()["/api/v1/projects/project-a/training/datasets"]).toBe(1);

    const assetsTab = [...container.querySelectorAll('[role="tab"]')].find(
      (candidate) => candidate.textContent.trim() === "Assets",
    );
    expect(assetsTab).toBeTruthy();
    await act(async () => {
      assetsTab.click();
    });
    await settle();

    expect(container.textContent).toContain("Mara portrait set");
  });

  it("hydrates Library detail catalogs after assets settle so direct-entry controls are complete", async () => {
    const assetRequest = deferred();
    installFetch({
      "/api/v1/projects/project-a/assets": () => assetRequest.promise,
      "/api/v1/projects/project-a/characters": () =>
        Promise.resolve(response([{ id: "character-a", name: "Mara", projectId: "project-a" }])),
      "/api/v1/models": () =>
        Promise.resolve(response([{
          id: "vqa-model",
          name: "Vision QA",
          type: "image",
          capabilities: ["vqa"],
        }])),
    });
    await renderApp();

    let counts = requestCounts();
    expect(counts["/api/v1/projects/project-a/characters"]).toBeUndefined();
    expect(counts["/api/v1/models"]).toBeUndefined();

    await act(async () => {
      assetRequest.resolve(response([{
        id: "library-image",
        projectId: "project-a",
        type: "image",
        displayName: "Library portrait",
        file: { path: "portrait.png", mimeType: "image/png" },
        status: { favorite: false, rating: 0, rejected: false, trashed: false },
      }]));
    });
    await settle();

    counts = requestCounts();
    expect(counts["/api/v1/projects/project-a/characters"]).toBe(1);
    expect(counts["/api/v1/models"]).toBe(1);
    expect(container.querySelector('select[aria-label="Target character"]')?.textContent).toContain("Mara");
    expect(container.querySelector('textarea[aria-label="Ask about this image"]')).toBeTruthy();
  });

  it("retains Library's pending asset hydration when Queue becomes the active consumer", async () => {
    const assetsRequest = deferred();
    let assetSignal;
    installFetch({
      "/api/v1/projects/project-a/assets": (_url, options) => {
        assetSignal = options.signal;
        return assetsRequest.promise;
      },
      "/api/v1/jobs": () =>
        Promise.resolve(response([{
          id: "historical-job",
          projectId: "project-a",
          projectName: "Project A",
          type: "image_generate",
          status: "completed",
          createdAt: "2026-07-26T01:00:00Z",
          updatedAt: "2026-07-26T01:01:00Z",
          payload: { prompt: "historical" },
          result: { assetIds: ["historical-asset"] },
        }])),
    });
    await renderApp();
    expect(requestCounts()["/api/v1/projects/project-a/assets"]).toBe(1);
    expect(assetSignal?.aborted).toBe(false);

    await navigate("Queue");
    expect(assetSignal?.aborted).toBe(false);
    expect(requestCounts()["/api/v1/projects/project-a/assets"]).toBe(1);

    await act(async () => {
      assetsRequest.resolve(response([{
        id: "historical-asset",
        projectId: "project-a",
        type: "image",
        displayName: "Historical thumbnail",
        file: { path: "historical.png", mimeType: "image/png" },
        status: { favorite: false, rating: 0, rejected: false, trashed: false },
      }]));
    });
    await settle();

    expect(
      container.querySelector('button[aria-label="Historical thumbnail"]'),
    ).toBeTruthy();
  });

  it("retries a failed hydration domain when its route is revisited", async () => {
    let modelAttempts = 0;
    installFetch({
      "/api/v1/projects/project-a/assets": () =>
        Promise.resolve(response([{
          id: "library-image",
          projectId: "project-a",
          type: "image",
          displayName: "Library image",
          status: {},
        }])),
      "/api/v1/models": () => {
        modelAttempts += 1;
        return modelAttempts === 1
          ? Promise.reject(new Error("model catalog unavailable"))
          : Promise.resolve(response([{ id: "model-a", name: "Recovered Model", type: "image" }]));
      },
    });
    await renderApp();

    expect(container.textContent).toContain("model catalog unavailable");
    expect(modelAttempts).toBe(1);
    await navigate("Models");
    expect(modelAttempts).toBe(2);
    expect(container.textContent).not.toContain("model catalog unavailable");
    expect(container.textContent).toContain("Recovered Model");
  });

  it("does not apply a late recipe launch when required catalog hydration fails", async () => {
    const asset = {
      id: "asset-recipe",
      projectId: "project-a",
      type: "image",
      displayName: "Recipe source",
      file: { path: "source.png", mimeType: "image/png" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
      generationSet: {
        recipe: {
          mode: "text_to_image",
          model: "missing-model",
          prompt: "must not be applied after failure",
        },
      },
    };
    installFetch({
      "/api/v1/projects/project-a/assets": () => Promise.resolve(response([asset])),
      "/api/v1/models": () => Promise.reject(new Error("model catalog unavailable")),
    });
    await renderApp();

    const tile = container.querySelector(".asset-tile");
    expect(tile).toBeTruthy();
    await act(async () => {
      tile.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    await settle();
    const useRecipe = [...document.body.querySelectorAll("button")].find(
      (button) => button.textContent === "Use this recipe",
    );
    expect(useRecipe).toBeTruthy();
    await act(async () => {
      useRecipe.click();
    });
    await settle();

    expect(container.textContent).toContain("model catalog unavailable");
    expect(container.querySelector('.image-studio textarea')?.value ?? "").not.toContain(
      "must not be applied after failure",
    );
  });

  it("refreshes only model-dependent domains for completed model and LoRA jobs", async () => {
    installFetch();
    await renderApp();
    expect(FakeEventSource.instances).toHaveLength(1);
    const source = FakeEventSource.instances[0];
    const before = requestCounts();

    await act(async () => {
      source.listeners["job.updated"]({
        data: JSON.stringify({
          id: "model-job",
          projectId: "project-a",
          status: "completed",
          type: "model_import",
          createdAt: "2026-07-26T01:00:00Z",
          updatedAt: "2026-07-26T01:00:00Z",
        }),
        lastEventId: "10",
      });
    });
    await settle();
    let counts = requestCounts();
    expect(counts["/api/v1/models"]).toBe((before["/api/v1/models"] ?? 0) + 1);

    await act(async () => {
      source.listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-job",
          projectId: "project-a",
          status: "completed",
          type: "lora_import",
          createdAt: "2026-07-26T01:01:00Z",
          updatedAt: "2026-07-26T01:01:00Z",
        }),
        lastEventId: "11",
      });
    });
    await settle();
    counts = requestCounts();
    expect(counts["/api/v1/models"]).toBe((before["/api/v1/models"] ?? 0) + 2);
    expect(counts["/api/v1/loras"]).toBe((before["/api/v1/loras"] ?? 0) + 1);

    [
      "/api/v1/projects",
      "/api/v1/jobs",
      "/api/v1/workers",
      "/api/v1/projects/project-a/assets",
      "/api/v1/training/targets",
      "/api/v1/training/presets",
      "/api/v1/projects/project-a/training/datasets",
      "/api/v1/prompt-batches",
    ].forEach((path) => expect(counts[path] ?? 0).toBe(before[path] ?? 0));
  });

  it("does not let a successful sibling domain erase an unrelated error", async () => {
    installFetch({
      "/api/v1/models": () => Promise.reject(new Error("model catalog unavailable")),
      "/api/v1/loras": () => Promise.resolve(response([{ id: "lora-a" }])),
    });
    await renderApp();
    await navigate("Models");

    expect(container.textContent).toContain("model catalog unavailable");
    expect(requestCounts()["/api/v1/loras"]).toBe(1);

    await navigate("Assets");
    expect(container.textContent).toContain("model catalog unavailable");
  });

  it("aborts and suppresses stale route-domain work on a project switch", async () => {
    const projectACharacters = deferred();
    let projectASignal;
    installFetch({
      "/api/v1/projects": () =>
        Promise.resolve(
          response([
            { id: "project-a", name: "Project A" },
            { id: "project-b", name: "Project B" },
          ]),
        ),
      "/api/v1/projects/project-a/characters": (_url, options) => {
        projectASignal = options.signal;
        return projectACharacters.promise;
      },
      "/api/v1/projects/project-b/characters": () =>
        Promise.resolve(
          response([{ id: "character-b", name: "Character B", projectId: "project-b" }]),
        ),
    });
    await renderApp();
    await navigate("Characters");
    expect(projectASignal?.aborted).toBe(false);

    await act(async () => {
      container.querySelector(".project-pill").click();
    });
    const projectB = [...container.querySelectorAll('[role="option"]')].find(
      (option) => option.textContent.includes("Project B"),
    );
    expect(projectB).toBeTruthy();
    await act(async () => {
      projectB.click();
    });
    await settle();

    expect(projectASignal.aborted).toBe(true);
    expect(container.textContent).toContain("Character B");
    await act(async () => {
      projectACharacters.resolve(
        response([{ id: "character-a", name: "Character A", projectId: "project-a" }]),
      );
    });
    await settle();
    expect(container.textContent).not.toContain("Character A");
    expect(container.textContent).toContain("Character B");
  });
});
