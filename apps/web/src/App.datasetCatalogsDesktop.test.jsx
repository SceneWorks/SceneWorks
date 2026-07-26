import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("./runtime.js", () => ({
  isDesktop: true,
  tauriInvoke: invoke,
}));

import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

describe("Dataset Catalogs desktop setup bypass", () => {
  let container;
  let root;

  beforeEach(() => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    invoke.mockImplementation((command) => Promise.resolve(
      command === "get_storage_setup" ? { setupCompleted: false } : null,
    ));
    vi.stubGlobal("fetch", vi.fn((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) return Promise.resolve(response({ status: "ok", authRequired: false }));
      if (path.endsWith("/access")) return Promise.resolve(response({ authRequired: false }));
      if (path.endsWith("/projects")) return Promise.resolve(response([]));
      if (path.endsWith("/ui-preferences") && (!options.method || options.method === "GET")) {
        return Promise.resolve(response({ activeView: "DatasetCatalogs" }));
      }
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

  it("opens global catalogs with setup incomplete and no project", async () => {
    root = createRoot(container);
    await act(async () => root.render(<App />));
    await settle();
    await settle();

    expect(invoke).toHaveBeenCalledWith("get_storage_setup");
    expect(container.querySelector(".nav-item.active")?.textContent).toContain("Dataset Catalogs");
    expect(container.textContent).toContain("No catalogs attached");
    expect(container.textContent).not.toContain("Choose where SceneWorks stores");
    expect(container.textContent).not.toContain("Create your first workspace");
  });
});
