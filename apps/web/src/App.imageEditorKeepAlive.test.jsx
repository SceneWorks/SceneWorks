import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The Image Editor is React.lazy'd and imports react-konva (whose node build pulls the
// native `canvas` package, unusable in jsdom). The empty-state editor never mounts a
// <Stage>, so stub react-konva to keep konva off the graph — mirroring the ImageEditor
// unit test. This lets the FULL <App /> mount the (lazy) editor to exercise keep-alive.
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

// sc-11968 (epic 11958, R4): under selective keep-alive the Image Editor stays MOUNTED
// (hidden) across navigation, so its working image / edit chain / undo history — all held
// in component state — survive a round trip to another screen and back. jsdom can't decode
// a real bitmap, so we prove the MECHANISM: the editor's DOM node is the SAME instance
// across the round trip (never unmounted) AND a piece of its component-only UI state (the
// keyboard-shortcuts panel, which has no persistence to re-hydrate from) is still there.
describe("Image Editor keep-alive edit survival (sc-11968)", () => {
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

  // Click a top-level nav item / any button by exact visible text. The sidebar renders
  // before the workspace, so a nav button wins over an identically-labelled inner button.
  async function clickButton(label) {
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === label)?.click();
    });
    await settle();
  }

  const editor = () => document.body.querySelector(".ie-shell");
  const modelsSurface = () => document.body.querySelector(".models-surface");
  const shortcutsPanel = () => document.body.querySelector(".image-editor-shortcuts");

  // Navigate to the (lazy) Image Editor and wait for the dynamic import + Suspense to
  // resolve (the first visit transforms + loads the editor chunk, which takes real time).
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

  it("keeps the editor mounted (same node) across a nav round trip, preserving in-editor state", async () => {
    await renderApp();
    await openImageEditor();

    const editorBefore = editor();
    expect(editorBefore).not.toBeNull();

    // Leave a piece of component-only UI state on the editor: open the keyboard-shortcuts
    // panel (a boolean React state with nothing to re-hydrate it from). If this survives a
    // round trip, the editor was never unmounted — so its working image / edits / undo
    // history (same kind of component state) survive too.
    await clickButton("⌨");
    expect(shortcutsPanel()).not.toBeNull();

    // Navigate to an OUT screen (Models unmounts on nav away) and back.
    await clickButton("Models");
    expect(modelsSurface()).not.toBeNull();
    // The editor stays resident (hidden) while another view is active.
    expect(editor()).not.toBeNull();

    await openImageEditor();
    // Same DOM node → never unmounted/remounted.
    expect(editor()).toBe(editorBefore);
    // …and its in-editor state is intact with no re-hydrate.
    expect(shortcutsPanel()).not.toBeNull();
    // Contrast: the OUT screen was torn down on nav away.
    expect(modelsSurface()).toBeNull();
  });
});
