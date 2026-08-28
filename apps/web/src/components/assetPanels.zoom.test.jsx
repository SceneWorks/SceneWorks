import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const actionMocks = vi.hoisted(() => ({
  assetCanCarryWorkflow: vi.fn(() => true),
  revealAsset: vi.fn(),
  saveAssetAs: vi.fn(),
}));

vi.mock("../assetActions.js", () => ({
  assetCanCarryWorkflow: actionMocks.assetCanCarryWorkflow,
  revealAsset: actionMocks.revealAsset,
  saveAssetAs: actionMocks.saveAssetAs,
}));

vi.mock("../runtime.js", () => ({
  isDesktop: true,
  isPageFullscreen: () => false,
  setViewerFullscreen: vi.fn(() => Promise.resolve()),
  tauriInvoke: vi.fn(),
}));

import { FullscreenPreview } from "./assetPanels.jsx";

const original = {
  id: "original",
  displayName: "Original",
  file: { mimeType: "image/png", path: "original.png" },
  projectId: "project-1",
  status: {},
  type: "image",
};
const edited = {
  ...original,
  id: "edited",
  displayName: "Edited",
  file: { mimeType: "image/png", path: "edited.png" },
  recipe: { mode: "edit_image" },
};

let container;
let root;

async function renderPreview() {
  root = createRoot(container);
  await act(async () => {
    root.render(
      <FullscreenPreview
        asset={edited}
        deleteAsset={() => {}}
        nextAsset={null}
        onClose={() => {}}
        onPreviewAsset={() => {}}
        previousAsset={null}
        purgeAsset={() => {}}
        sourceAsset={original}
        updateAssetStatus={() => {}}
      />,
    );
  });
}

function viewport() {
  return document.body.querySelector(".preview-zoom-viewport");
}

async function wheel(node) {
  await act(async () => {
    node.dispatchEvent(new WheelEvent("wheel", { bubbles: true, cancelable: true, clientX: 10, clientY: 10, deltaY: -1 }));
  });
}

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  if (root) {
    act(() => root.unmount());
    root = null;
  }
  container.remove();
});

describe("FullscreenPreview zoom viewport lifecycle (sc-21905)", () => {
  it("rebinds the native wheel listener to the replacement viewport after compare mode", async () => {
    const addSpy = vi.spyOn(HTMLElement.prototype, "addEventListener");
    const removeSpy = vi.spyOn(HTMLElement.prototype, "removeEventListener");
    try {
      await renderPreview();
      const firstViewport = viewport();

      await act(async () => {
        document.body.querySelector(".preview-compare-toggle").click();
      });
      expect(viewport()).toBeNull();

      await act(async () => {
        document.body.querySelector(".preview-compare-toggle").click();
      });
      const replacementViewport = viewport();
      expect(replacementViewport).not.toBe(firstViewport);
      await wheel(replacementViewport);
      expect(replacementViewport.classList).toContain("zoomed");

      const wheelAddsOn = (node) =>
        addSpy.mock.calls.filter(([type], index) => type === "wheel" && addSpy.mock.instances[index] === node);
      const wheelRemovesOn = (node) =>
        removeSpy.mock.calls.filter(([type], index) => type === "wheel" && removeSpy.mock.instances[index] === node);
      const firstWheelAdds = wheelAddsOn(firstViewport);
      const replacementWheelAdds = wheelAddsOn(replacementViewport);
      expect(firstWheelAdds).toHaveLength(1);
      expect(replacementWheelAdds).toHaveLength(1);
      expect(wheelRemovesOn(firstViewport)).toContainEqual(["wheel", firstWheelAdds[0][1]]);
    } finally {
      addSpy.mockRestore();
      removeSpy.mockRestore();
    }
  });
});
