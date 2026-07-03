import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// True-fullscreen mode for FullscreenPreview. The mechanism (native Tauri window vs
// HTML5 Fullscreen API) lives behind runtime.js:setViewerFullscreen, which we mock so
// the component logic — the is-fullscreen class, the Esc-to-exit handler, the context
// menu toggle, and the browser fullscreenchange sync — can be exercised without Tauri
// or a real fullscreen element. Mirrors the mocking style of the context-menu suite.

const actionMocks = vi.hoisted(() => ({
  saveAssetAs: vi.fn(),
  revealAsset: vi.fn(),
}));

vi.mock("../assetActions.js", () => ({
  saveAssetAs: actionMocks.saveAssetAs,
  revealAsset: actionMocks.revealAsset,
}));

// Mutable runtime state driven per-suite. The component reads `isDesktop` at module
// load, so tests set this BEFORE importing the component (below). `pageFullscreen`
// backs the mocked isPageFullscreen() so we can simulate the browser leaving
// fullscreen natively.
const runtimeState = vi.hoisted(() => ({ isDesktop: true, pageFullscreen: false }));
const runtimeMocks = vi.hoisted(() => ({ setViewerFullscreen: vi.fn(() => Promise.resolve()) }));

vi.mock("../runtime.js", () => ({
  get isDesktop() {
    return runtimeState.isDesktop;
  },
  tauriInvoke: vi.fn(),
  setViewerFullscreen: runtimeMocks.setViewerFullscreen,
  isPageFullscreen: () => runtimeState.pageFullscreen,
}));

const imageAsset = {
  id: "asset-img",
  projectId: "project-1",
  displayName: "Plate",
  type: "image",
  status: {},
  file: { path: "assets/images/plate.png", mimeType: "image/png" },
};

let container;
let root;
let FullscreenPreview;

const noop = () => {};

function baseProps(overrides = {}) {
  return {
    asset: imageAsset,
    deleteAsset: noop,
    nextAsset: null,
    onClose: noop,
    onPreviewAsset: noop,
    previousAsset: null,
    purgeAsset: noop,
    updateAssetStatus: noop,
    ...overrides,
  };
}

async function renderPreview(props) {
  root = createRoot(container);
  await act(async () => {
    root.render(<FullscreenPreview {...props} />);
  });
}

// Flush the microtask chain in enterFullscreen/exitFullscreen (they await the mocked
// setViewerFullscreen before flipping state).
async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

function modal() {
  return document.body.querySelector(".preview-modal");
}

function isFullscreen() {
  return modal().classList.contains("is-fullscreen");
}

async function clickEnterFullscreen() {
  await act(async () => {
    document.body.querySelector(".preview-fullscreen-button").click();
  });
  await flush();
}

async function pressEscape() {
  await act(async () => {
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  });
  await flush();
}

async function rightClickStage() {
  const stage = document.body.querySelector(".preview-modal-stage");
  await act(async () => {
    stage.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 120, clientY: 90 }));
  });
}

function menu() {
  return document.body.querySelector(".preview-context-menu");
}

function menuItemLabels() {
  return [...document.body.querySelectorAll(".preview-context-menu > .preview-context-menu-item")].map((b) =>
    b.textContent.trim(),
  );
}

async function loadComponent(isDesktop) {
  runtimeState.isDesktop = isDesktop;
  runtimeState.pageFullscreen = false;
  vi.resetModules();
  ({ FullscreenPreview } = await import("./assetPanels.jsx"));
}

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  actionMocks.saveAssetAs.mockReset();
  actionMocks.revealAsset.mockReset();
  runtimeMocks.setViewerFullscreen.mockClear();
  runtimeMocks.setViewerFullscreen.mockImplementation(() => Promise.resolve());
});

afterEach(() => {
  if (root) {
    act(() => root.unmount());
    root = null;
  }
  container.remove();
});

describe("FullscreenPreview fullscreen (desktop)", () => {
  beforeEach(async () => {
    await loadComponent(true);
  });

  it("enters fullscreen from the top-bar button", async () => {
    await renderPreview(baseProps());
    expect(isFullscreen()).toBe(false);

    await clickEnterFullscreen();

    expect(runtimeMocks.setViewerFullscreen).toHaveBeenCalledWith(true);
    expect(isFullscreen()).toBe(true);
    // The Esc hint appears only in fullscreen.
    expect(document.body.querySelector(".preview-fullscreen-hint")).not.toBeNull();
  });

  it("exits fullscreen on Escape without closing the preview", async () => {
    const onClose = vi.fn();
    await renderPreview(baseProps({ onClose }));
    await clickEnterFullscreen();
    expect(isFullscreen()).toBe(true);
    runtimeMocks.setViewerFullscreen.mockClear();

    await pressEscape();

    expect(runtimeMocks.setViewerFullscreen).toHaveBeenCalledWith(false);
    expect(isFullscreen()).toBe(false);
    // First Esc only leaves fullscreen — it must NOT close the preview.
    expect(onClose).not.toHaveBeenCalled();
    expect(document.body.querySelector(".preview-fullscreen-hint")).toBeNull();
  });

  it("Escape closes an open context menu before it exits fullscreen", async () => {
    await renderPreview(baseProps());
    await clickEnterFullscreen();
    await rightClickStage();
    expect(menu()).not.toBeNull();
    runtimeMocks.setViewerFullscreen.mockClear();

    await pressEscape();

    // The menu's own Esc handler wins; we stay in fullscreen.
    expect(menu()).toBeNull();
    expect(isFullscreen()).toBe(true);
    expect(runtimeMocks.setViewerFullscreen).not.toHaveBeenCalled();
  });

  it("offers a fullscreen toggle in the right-click menu that flips label with state", async () => {
    await renderPreview(baseProps());

    await rightClickStage();
    expect(menuItemLabels()).toContain("Full screen");
    // Enter fullscreen via the menu item.
    await act(async () => {
      [...document.body.querySelectorAll(".preview-context-menu-item")]
        .find((b) => b.textContent.trim() === "Full screen")
        .click();
    });
    await flush();
    expect(isFullscreen()).toBe(true);

    // Re-opening the menu now offers the exit label.
    await rightClickStage();
    expect(menuItemLabels()).toContain("Exit full screen");
  });

  it("leaves fullscreen when the preview unmounts mid-fullscreen", async () => {
    await renderPreview(baseProps());
    await clickEnterFullscreen();
    runtimeMocks.setViewerFullscreen.mockClear();

    await act(async () => root.unmount());
    root = null;

    expect(runtimeMocks.setViewerFullscreen).toHaveBeenCalledWith(false);
  });
});

describe("FullscreenPreview fullscreen (remote browser)", () => {
  beforeEach(async () => {
    await loadComponent(false);
  });

  it("syncs state when the browser leaves fullscreen natively", async () => {
    await renderPreview(baseProps());
    await clickEnterFullscreen();
    expect(isFullscreen()).toBe(true);

    // Simulate the browser exiting fullscreen (Esc / F11): fullscreenElement clears
    // and a fullscreenchange event fires.
    runtimeState.pageFullscreen = false;
    await act(async () => {
      document.dispatchEvent(new Event("fullscreenchange"));
    });
    await flush();

    expect(isFullscreen()).toBe(false);
  });
});
