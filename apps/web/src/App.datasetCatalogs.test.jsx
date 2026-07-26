import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { navSections, viewTitles } from "./App.jsx";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

describe("Dataset Catalogs global app navigation", () => {
  let container;
  let root;
  let requests;

  beforeEach(() => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    requests = [];
    vi.stubGlobal("fetch", vi.fn((url, options = {}) => {
      const path = new URL(url).pathname;
      requests.push({ path, options, body: options.body ? JSON.parse(options.body) : null });
      if (path.endsWith("/health")) return Promise.resolve(response({ status: "ok", authRequired: false }));
      if (path.endsWith("/access")) return Promise.resolve(response({ authRequired: false }));
      if (path.endsWith("/projects")) return Promise.resolve(response([]));
      if (path.endsWith("/ui-preferences") && (!options.method || options.method === "GET")) {
        return Promise.resolve(response({ activeView: "DatasetCatalogs" }));
      }
      if (path.endsWith("/ui-preferences")) return Promise.resolve(response({ activeView: "DatasetCatalogs" }));
      if (path.endsWith("/catalogs")) return Promise.resolve(response([]));
      return Promise.resolve(response([]));
    }));
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("registers a top-level title and Library navigation item", () => {
    const item = navSections.flatMap((section) => section.items).find((candidate) => candidate.id === "DatasetCatalogs");
    expect(item?.label).toBe("Dataset Catalogs");
    expect(viewTitles.DatasetCatalogs.blurb).toContain("independently");
  });

  it("restores the page after restart and bypasses the first-project gate", async () => {
    root = createRoot(container);
    await act(async () => root.render(<App />));
    await settle();
    await settle();

    expect(container.querySelector(".nav-item.active")?.textContent).toContain("Dataset Catalogs");
    expect(container.textContent).toContain("No catalogs attached");
    expect(container.textContent).not.toContain("Create your first workspace");
    expect(requests.some((request) => request.path === "/api/v1/catalogs")).toBe(true);
    expect(requests.some(
      (request) =>
        request.path === "/api/v1/ui-preferences" &&
        request.options.method === "PUT" &&
        request.body?.activeView === "DatasetCatalogs",
    )).toBe(true);
  });
});
