import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, reject, resolve };
}

describe("independent project and asset bootstrap (sc-14754)", () => {
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

  function installFetch({
    assets = Promise.resolve(response([])),
    models = Promise.resolve(response([])),
    presets = Promise.resolve(response([])),
    scopedPresets = presets,
  } = {}) {
    global.fetch = vi.fn((url) => {
      const requestUrl = new URL(url);
      const path = requestUrl.pathname;
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
      if (path.endsWith("/projects/project-a/assets")) {
        return assets;
      }
      if (path.endsWith("/models")) {
        return models;
      }
      if (path.endsWith("/recipe-presets")) {
        return requestUrl.searchParams.has("projectId") ? scopedPresets : presets;
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(
          response([
            {
              id: "job-running",
              projectId: "project-a",
              status: "running",
              type: "image",
            },
          ]),
        );
      }
      if (path.endsWith("/workers")) {
        return Promise.resolve(
          response([
            {
              id: "worker-a",
              status: "idle",
              gpuId: "gpu-a",
              capabilities: ["image_generate"],
            },
          ]),
        );
      }
      if (path.endsWith("/loras")) {
        return Promise.resolve(response([{ id: "lora-a", name: "LoRA A" }]));
      }
      if (path.endsWith("/training/targets")) {
        return Promise.resolve(
          response({ schemaVersion: 1, targets: [{ id: "target-a" }] }),
        );
      }
      if (path.endsWith("/training/presets")) {
        return Promise.resolve(
          response({ schemaVersion: 1, presets: [{ id: "training-preset-a" }] }),
        );
      }
      if (path.endsWith("/prompt-batches")) {
        return Promise.resolve(response([{ id: "batch-a", name: "Batch A" }]));
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

  it("renders Library assets before its deferred detail catalogs settle", async () => {
    const models = deferred();
    const presets = deferred();
    installFetch({
      assets: Promise.resolve(
        response([
          {
            id: "asset-a",
            projectId: "project-a",
            type: "image",
            origin: "image_studio",
            displayName: "Hero Asset",
            status: {},
          },
        ]),
      ),
      models: models.promise,
      presets: presets.promise,
      scopedPresets: Promise.resolve(
        response([{ id: "project-preset", name: "Project Preset" }]),
      ),
    });

    await renderApp();

    expect(container.querySelector(".project-pill")?.textContent).toContain("Project A");
    expect(container.textContent).toContain("Hero Asset");
    expect(container.textContent).not.toContain("Loading assets");
    // Other independently settling domains commit even though the two slow
    // catalogs are still held open.
    expect(container.textContent).toContain("1 worker");
    expect(container.textContent).toContain("Queue 1");

    const paths = global.fetch.mock.calls.map(([url]) => new URL(url).pathname);
    expect(paths).toContain("/api/v1/projects/project-a/assets");
    expect(paths).toContain("/api/v1/models");
    expect(paths).not.toContain("/api/v1/recipe-presets");
    expect(paths.indexOf("/api/v1/projects")).toBeLessThan(
      paths.indexOf("/api/v1/projects/project-a/assets"),
    );

    await act(async () => {
      models.resolve(response([{ id: "model-a", name: "Model A", type: "image" }]));
      presets.resolve(response([{ id: "global-preset", name: "Global Preset" }]));
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll(".nav-item")]
        .find((button) => button.textContent.includes("Presets"))
        ?.click();
    });
    await settle();
    expect(container.textContent).toContain("Project Preset");
    expect(container.textContent).not.toContain("Global Preset");
  });

  it("keeps an empty Library isolated from unrelated inactive catalogs", async () => {
    installFetch();

    await renderApp();
    expect(container.textContent).toContain("No assets in this view");

    expect(container.querySelector(".project-pill")?.textContent).toContain("Project A");
    expect(container.textContent).toContain("No assets in this view");
    const paths = global.fetch.mock.calls.map(([url]) => new URL(url).pathname);
    expect(paths).not.toContain("/api/v1/models");
    expect(paths).not.toContain("/api/v1/recipe-presets");
    expect(paths).not.toContain("/api/v1/training/targets");
    expect(paths).not.toContain("/api/v1/training/presets");
    expect(paths).not.toContain("/api/v1/prompt-batches");
  });

  it("shows loading and error states before unrelated catalogs settle", async () => {
    const assets = deferred();
    const models = deferred();
    const presets = deferred();
    installFetch({
      assets: assets.promise,
      models: models.promise,
      presets: presets.promise,
    });

    await renderApp();
    expect(container.textContent).toContain("Loading assets");

    await act(async () => {
      assets.reject(new Error("asset catalog unavailable"));
    });
    await settle();

    expect(container.textContent).toContain(
      "Couldn't load assets: asset catalog unavailable",
    );

    await act(async () => {
      models.resolve(response([]));
      presets.resolve(response([]));
    });
    await settle();
  });
});
