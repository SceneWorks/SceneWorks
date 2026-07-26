import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAppStatic } from "./context/AppContext.js";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

const { projectAssetRenderHistory } = vi.hoisted(() => ({
  projectAssetRenderHistory: [],
}));

vi.mock("./screens/VideoStudio.jsx", () => ({
  VideoStudio() {
    const { activeProject, assets, selectedAssetId } = useAppStatic();
    projectAssetRenderHistory.push({
      projectId: activeProject?.id ?? null,
      assetIds: assets.map((asset) => asset.id),
      selectedAssetId,
    });
    return (
      <output data-testid="project-asset-probe">
        {JSON.stringify({
          assetIds: assets.map((asset) => asset.id),
          selectedAssetId,
        })}
      </output>
    );
  },
}));

import { App } from "./main.jsx";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, reject, resolve };
}

describe("project asset render boundary", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    projectAssetRenderHistory.length = 0;

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
        return Promise.resolve(
          response([
            { id: "project-a", name: "Project A" },
            { id: "project-b", name: "Project B" },
          ]),
        );
      }
      if (path.endsWith("/projects/project-a/assets")) {
        return Promise.resolve(
          response([{ id: "asset-a", type: "image", displayName: "Asset A", status: {} }]),
        );
      }
      if (path.endsWith("/projects/project-b/assets")) {
        // Keep B's catalog pending so this test observes its very first render.
        return new Promise(() => {});
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("never exposes project A assets or raw selection to project B while B loads", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")]
        .find((button) => button.textContent === "Video")
        ?.click();
    });
    await settle();

    const probe = () =>
      JSON.parse(document.body.querySelector("[data-testid='project-asset-probe']").textContent);
    expect(probe()).toEqual({ assetIds: ["asset-a"], selectedAssetId: "asset-a" });

    await act(async () => {
      document.body.querySelector(".project-pill")?.click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".project-menu-item")]
        .find((item) => item.textContent.includes("Project B"))
        ?.click();
    });

    expect(probe()).toEqual({ assetIds: [], selectedAssetId: null });
    const projectBRenders = projectAssetRenderHistory.filter(
      (snapshot) => snapshot.projectId === "project-b",
    );
    expect(projectBRenders.length).toBeGreaterThan(0);
    expect(projectBRenders).toEqual(
      projectBRenders.map(() => ({
        projectId: "project-b",
        assetIds: [],
        selectedAssetId: null,
      })),
    );
  });

  it("does not fetch assets for a background-project SSE update", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    expect(
      global.fetch.mock.calls.some(([url]) =>
        new URL(url).pathname.endsWith("/projects/project-b/assets"),
      ),
    ).toBe(false);

    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "background-job",
          projectId: "project-b",
          status: "running",
          createdAt: "2026-07-25T12:00:00Z",
          result: { assetIds: ["background-asset"] },
        }),
      });
    });
    await settle();

    expect(
      global.fetch.mock.calls.some(([url]) =>
        new URL(url).pathname.endsWith("/projects/project-b/assets"),
      ),
    ).toBe(false);
  });

  it("aborts project A and ignores its non-abort-aware failure after switching to B", async () => {
    const projectAAssets = deferred();
    const baseFetch = global.fetch;
    let projectASignal;
    global.fetch = vi.fn((url, options) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/projects/project-a/assets")) {
        projectASignal = options?.signal;
        return projectAAssets.promise;
      }
      if (path.endsWith("/projects/project-b/assets")) {
        return Promise.resolve(
          response([
            {
              id: "asset-b",
              projectId: "project-b",
              type: "image",
              origin: "image_studio",
              displayName: "Asset B",
              status: {},
            },
          ]),
        );
      }
      return baseFetch(url, options);
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    expect(projectASignal?.aborted).toBe(false);

    await act(async () => {
      document.body.querySelector(".project-pill")?.click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".project-menu-item")]
        .find((item) => item.textContent.includes("Project B"))
        ?.click();
    });
    await settle();

    expect(projectASignal?.aborted).toBe(true);
    expect(container.textContent).toContain("Asset B");

    // Simulate a transport that ignores AbortSignal and reports a late ordinary
    // failure. The old request must not replace B's settled state or error UI.
    await act(async () => {
      projectAAssets.reject(new Error("stale A failure"));
    });
    await settle();

    expect(container.textContent).toContain("Asset B");
    expect(container.textContent).not.toContain("stale A failure");
  });
});
