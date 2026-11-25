import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("react-konva", async () => {
  const React = await import("react");
  const passthrough = (name) => ({ children }) => React.createElement("div", { "data-konva": name }, children);
  return {
    Stage: passthrough("stage"),
    Layer: passthrough("layer"),
    Image: () => null,
    Rect: () => null,
    Line: () => null,
    Transformer: () => null,
  };
});

import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

describe("Image Editor page lifetime", () => {
  let container;
  let root;

  function mockFetch() {
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) return Promise.resolve(response({ status: "ok", authRequired: false }));
      if (path.endsWith("/access")) return Promise.resolve(response({ authRequired: false }));
      if (path.endsWith("/jobs/events/ticket")) return Promise.resolve(response({ ticket: "stream-ticket" }));
      if (path.endsWith("/projects")) return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
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

  async function clickButton(label) {
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === label)?.click();
    });
    await settle();
  }

  const editor = () => document.body.querySelector(".ie-shell");
  const modelsSurface = () => document.body.querySelector(".models-surface");
  const shortcutsPanel = () => document.body.querySelector(".image-editor-shortcuts");

  const confirmDialog = () => document.body.querySelector(".app-confirm-modal");

  async function openImageEditor() {
    await clickButton("Image Editor");
    for (let i = 0; i < 40 && !editor(); i += 1) {
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 25));
      });
    }
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    mockFetch();
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("releases a clean editor on navigation and creates a fresh page on return", async () => {
    await renderApp();
    await openImageEditor();

    const editorBefore = editor();
    expect(editorBefore).not.toBeNull();

    await clickButton("⌨");
    expect(shortcutsPanel()).not.toBeNull();

    await clickButton("Models");

    expect(confirmDialog()).toBeNull();
    expect(modelsSurface()).not.toBeNull();

    expect(editor()).toBeNull();

    await openImageEditor();

    expect(confirmDialog()).toBeNull();

    expect(editor()).not.toBe(editorBefore);

    expect(shortcutsPanel()).toBeNull();

    expect(modelsSurface()).toBeNull();
  });
});
