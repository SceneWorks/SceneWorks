// The viewer's "Re-Generate" button. It appears on an IMAGE asset carrying a recorded recipe
// (the same precondition as "Use this recipe") and hands the asset straight to the caller, whose
// job is to enqueue it with a fresh seed. It is image-only and recipe-gated — video/vector re-runs,
// and any asset with no recorded recipe, offer nothing.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../assetActions.js", () => ({
  saveAssetAs: vi.fn(),
  revealAsset: vi.fn(),
  assetCanCarryWorkflow: () => false,
}));
vi.mock("../runtime.js", () => ({
  isDesktop: false,
  tauriInvoke: vi.fn(),
  setViewerFullscreen: vi.fn(() => Promise.resolve()),
  isPageFullscreen: () => false,
}));

import { FullscreenPreview } from "./assetPanels.jsx";

const noop = () => {};

function asset(overrides = {}) {
  return {
    id: "asset-img",
    projectId: "project-1",
    displayName: "Plate",
    type: "image",
    status: {},
    file: { path: "assets/images/plate.png", mimeType: "image/png" },
    ...overrides,
  };
}

describe("FullscreenPreview — the Re-Generate action", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const render = async (props) =>
    act(async () =>
      root.render(
        <FullscreenPreview
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
          {...props}
        />,
      ),
    );

  const regenerateButtons = () =>
    [...document.querySelectorAll("button")].filter(
      (node) => node.textContent.trim() === "Re-Generate",
    );

  it("offers Re-Generate on an image carrying a recorded recipe, and hands the asset back", async () => {
    const onRegenerate = vi.fn();
    await render({
      asset: asset({ recipe: { mode: "text_to_image", prompt: "a lighthouse", seed: 42 } }),
      onRegenerate,
      onUseRecipe: vi.fn(),
    });
    expect(regenerateButtons()).toHaveLength(1);
    await act(async () =>
      regenerateButtons()[0].dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    expect(onRegenerate).toHaveBeenCalledWith(expect.objectContaining({ id: "asset-img" }));
  });

  it("keeps the viewer mounted and suppresses repeat clicks while submission is pending", async () => {
    let finish;
    const onRegenerate = vi.fn(() => new Promise((resolve) => { finish = resolve; }));
    const onClose = vi.fn();
    await render({ asset: asset({ recipe: { model: "model", prompt: "lighthouse" } }), onRegenerate, onClose, onUseRecipe: vi.fn() });
    const viewer = document.querySelector('[role="dialog"]');
    await act(async () => {
      regenerateButtons()[0].click();
      regenerateButtons()[0].click();
    });
    expect(onRegenerate).toHaveBeenCalledOnce();
    expect(regenerateButtons()[0].disabled).toBe(true);
    await act(async () => finish({ id: "new-job" }));
    expect(regenerateButtons()[0].disabled).toBe(false);
    expect(document.querySelector('[role="dialog"]')).toBe(viewer);
    expect(onClose).not.toHaveBeenCalled();
  });

  it("shows submission errors inside the viewer and allows retry", async () => {
    const onRegenerate = vi.fn().mockRejectedValueOnce(new Error("Host unavailable")).mockResolvedValueOnce({ id: "job" });
    await render({ asset: asset({ recipe: { model: "model", prompt: "lighthouse" } }), onRegenerate, onUseRecipe: vi.fn() });
    await act(async () => regenerateButtons()[0].click());
    expect(document.querySelector('[role="alert"]').textContent).toBe("Host unavailable");
    await act(async () => regenerateButtons()[0].click());
    expect(document.querySelector('[role="alert"]')).toBeNull();
    expect(onRegenerate).toHaveBeenCalledTimes(2);
  });

  it("offers Re-Generate only when wired", async () => {
    // Not wired (the prop is absent) — mirrors the suite fixtures that don't pass it — so no button.
    await render({
      asset: asset({ recipe: { mode: "text_to_image", prompt: "a lighthouse", seed: 42 } }),
      onUseRecipe: vi.fn(),
    });
    expect(regenerateButtons()).toHaveLength(0);
  });

  it("offers nothing for an image with no recorded recipe", async () => {
    await render({ asset: asset(), onRegenerate: vi.fn(), onUseRecipe: vi.fn() });
    expect(regenerateButtons()).toHaveLength(0);
  });

  it("offers nothing for a video with a recipe (the straight-to-queue lane is image-only)", async () => {
    await render({
      asset: asset({ type: "video", recipe: { mode: "text_to_video", prompt: "a lighthouse", seed: 7 } }),
      onRegenerate: vi.fn(),
      onUseRecipe: vi.fn(),
    });
    expect(regenerateButtons()).toHaveLength(0);
  });
});