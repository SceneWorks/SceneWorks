import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssetPickerField } from "./components/AssetPicker.jsx";
import { FullscreenPreview, PREVIEW_FIT_VIEW, PREVIEW_MAX_SCALE, PREVIEW_MIN_SCALE, assetSeed, clampPan, zoomView } from "./components/assetPanels.jsx";
import { FakeEventSource, response, changeField } from "./main.testSupport.jsx";

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
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
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("selects duplicate-titled assets through the thumbnail asset picker", async () => {
    const onChange = vi.fn();
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Shot", createdAt: "2026-05-19T09:00:00Z", recipe: { mode: "text_to_image" } },
      { id: "image-beta", type: "image", displayName: "Shot", createdAt: "2026-05-19T09:05:00Z", recipe: { mode: "edit_image" } },
      { id: "clip-gamma", type: "video", displayName: "Shot", createdAt: "2026-05-19T09:10:00Z", file: { mimeType: "video/mp4" } },
      { id: "upload-delta", type: "upload", displayName: "Plate", createdAt: "2026-05-19T09:15:00Z" },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        <AssetPickerField
          assets={assets}
          buttonLabel="Select image"
          emptyLabel="No source image selected"
          label="Source"
          onChange={onChange}
          value=""
        />,
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Select image").click();
    });

    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull();
    expect(document.body.textContent).toContain("Images 2");
    expect(document.body.textContent).toContain("Video 1");
    expect(document.body.textContent).toContain("Uploads 1");
    expect(document.body.textContent).toContain("Renders 2");

    await act(async () => {
      [...document.body.querySelectorAll(".asset-picker-toolbar button")].find((button) => button.textContent.includes("Video")).click();
    });

    expect(document.body.querySelectorAll(".asset-picker-card")).toHaveLength(1);
    expect(document.body.querySelector('[title="clip-gamma"]')).not.toBeNull();

    await act(async () => {
      [...document.body.querySelectorAll(".asset-picker-toolbar button")].find((button) => button.textContent.includes("All")).click();
    });
    await changeField(document.body.querySelector('[aria-label="Search assets"]'), "plate");

    expect(document.body.querySelectorAll(".asset-picker-card")).toHaveLength(1);
    expect(document.body.textContent).toContain("Plate");

    await act(async () => {
      document.body.querySelector(".modal-backdrop").dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    });

    expect(document.body.querySelector('[role="dialog"]')).toBeNull();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Select image").click();
    });

    const cards = [...document.body.querySelectorAll(".asset-picker-card")];
    await act(async () => {
      cards[1].click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });

    expect(onChange).toHaveBeenCalledWith("image-beta");

    await act(async () => {
      root.render(
        <AssetPickerField
          assets={assets}
          buttonLabel="Select image"
          emptyLabel="No source image selected"
          label="Source"
          onChange={onChange}
          value="image-beta"
        />,
      );
    });

    expect(container.textContent).toContain("image-beta".slice(-6));

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Change").click();
    });
    await act(async () => {
      document.body.querySelector('[role="dialog"]').dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });

    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
  });

  it("dismisses FullscreenPreview via Escape and backdrop click", async () => {
    const onClose = vi.fn();
    const noop = () => {};
    const asset = {
      id: "asset-a",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/plate.png" },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={onClose}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull();

    await act(async () => {
      document.body.querySelector('[role="dialog"]').dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);

    await act(async () => {
      document.body.querySelector(".modal-backdrop").dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("toggles FullscreenPreview between original and upscaled variants", async () => {
    const noop = () => {};
    const original = {
      id: "asset-original",
      projectId: "project-1",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/original.png" },
    };
    const upscaled = {
      id: "asset-upscaled",
      projectId: "project-1",
      displayName: "Plate (2x upscaled)",
      type: "image",
      status: {},
      file: { path: "assets/images/upscaled.png" },
      lineage: { sourceAssetId: "asset-original", parents: ["asset-original"] },
      extra: { isUpscaled: true, upscaledFromAssetId: "asset-original", factor: 2, engine: "real-esrgan" },
      variants: { original, upscaled: null },
    };
    upscaled.variants.upscaled = upscaled;

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={upscaled}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    expect(document.body.querySelector(".preview-modal img").getAttribute("src")).toContain("upscaled.png");
    expect(document.body.textContent).toContain("Original");
    expect(document.body.textContent).toContain("Upscaled");

    await act(async () => {
      [...document.body.querySelectorAll(".preview-variant-toggle button")].find((button) => button.textContent === "Original").click();
    });

    expect(document.body.querySelector(".preview-modal img").getAttribute("src")).toContain("original.png");
  });

  it("toggles a side-by-side Original↔Edited compare for edits with a resolvable source", async () => {
    const noop = () => {};
    const original = {
      id: "asset-original",
      projectId: "project-1",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/original.png" },
    };
    const edited = {
      id: "asset-edited",
      projectId: "project-1",
      displayName: "Plate (edited)",
      type: "image",
      status: {},
      file: { path: "assets/images/edited.png" },
      recipe: { mode: "edit_image", model: "z_image_turbo" },
      lineage: { sourceAssetId: "asset-original", parents: ["asset-original"] },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={edited}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          sourceAsset={original}
          updateAssetStatus={noop}
        />,
      );
    });

    // Single view to start; a compare toggle is offered because the source resolves.
    expect(document.body.querySelector(".preview-compare")).toBeNull();
    const toggle = document.body.querySelector(".preview-compare-toggle");
    expect(toggle).not.toBeNull();

    await act(async () => {
      toggle.click();
    });

    // Both the original and edited image render side by side, each labelled.
    const compare = document.body.querySelector(".preview-compare");
    expect(compare).not.toBeNull();
    const srcs = [...compare.querySelectorAll("img")].map((img) => img.getAttribute("src"));
    expect(srcs.some((src) => src.includes("original.png"))).toBe(true);
    expect(srcs.some((src) => src.includes("edited.png"))).toBe(true);
    expect(compare.textContent).toContain("Original");
    expect(compare.textContent).toContain("Edited");

    // Toggling off collapses back to the single edited view.
    await act(async () => {
      document.body.querySelector(".preview-compare-toggle").click();
    });
    expect(document.body.querySelector(".preview-compare")).toBeNull();
  });

  it("offers no compare toggle when an edit's source is gone (purged) or the asset isn't an edit", async () => {
    const noop = () => {};
    const base = {
      id: "asset-edited",
      projectId: "project-1",
      displayName: "Plate (edited)",
      type: "image",
      status: {},
      file: { path: "assets/images/edited.png" },
      lineage: { sourceAssetId: "asset-original" },
    };

    // An edit whose source is no longer in the gallery → App resolves sourceAsset to null.
    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={{ ...base, recipe: { mode: "edit_image", model: "z_image_turbo" } }}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          sourceAsset={null}
          updateAssetStatus={noop}
        />,
      );
    });
    expect(document.body.querySelector(".preview-compare-toggle")).toBeNull();

    // A non-edit asset never gets the compare toggle, even with a source present.
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={{ ...base, recipe: { mode: "text_to_image", model: "z_image_turbo" } }}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          sourceAsset={{ id: "asset-original", projectId: "project-1", type: "image", status: {}, file: { path: "x.png" } }}
          updateAssetStatus={noop}
        />,
      );
    });
    expect(document.body.querySelector(".preview-compare-toggle")).toBeNull();
  });

  // sc-8728 zoom math is extracted into pure helpers so the cursor-anchoring can be
  // asserted precisely (jsdom has no layout, so DOM-level anchoring can't be exercised).
  describe("preview zoom helpers (sc-8728)", () => {
    it("zoomView keeps the point under the cursor stationary", () => {
      const view = { scale: 1, x: 0, y: 0 };
      const pointer = { x: 100, y: 60 };
      const zoomed = zoomView(view, pointer, 2);
      expect(zoomed.scale).toBe(2);
      // The image-pixel under the cursor must map back to the same stage pixel.
      const imgX = (pointer.x - zoomed.x) / zoomed.scale;
      const imgY = (pointer.y - zoomed.y) / zoomed.scale;
      const beforeX = (pointer.x - view.x) / view.scale;
      const beforeY = (pointer.y - view.y) / view.scale;
      expect(imgX).toBeCloseTo(beforeX, 6);
      expect(imgY).toBeCloseTo(beforeY, 6);
    });

    it("zoomView clamps to the min/max scale and is a no-op at the clamp", () => {
      const atMax = { scale: PREVIEW_MAX_SCALE, x: 0, y: 0 };
      expect(zoomView(atMax, { x: 10, y: 10 }, 2)).toBe(atMax);
      const belowMin = zoomView({ scale: PREVIEW_MIN_SCALE, x: 0, y: 0 }, { x: 10, y: 10 }, 0.5);
      expect(belowMin.scale).toBe(PREVIEW_MIN_SCALE);
    });

    it("clampPan pins the view at fit scale and keeps the image on-screen when zoomed", () => {
      const pinned = clampPan({ scale: 1, x: 40, y: -25 }, 200, 100);
      expect(pinned.x).toBe(0);
      expect(pinned.y).toBe(0);
      // At 2x the image is 400x200 over a 200x100 stage → offset in [-200..0]/[-100..0].
      expect(clampPan({ scale: 2, x: 50, y: 50 }, 200, 100)).toMatchObject({ x: 0, y: 0 });
      expect(clampPan({ scale: 2, x: -999, y: -999 }, 200, 100)).toMatchObject({ x: -200, y: -100 });
    });
  });

  it("shows zoom controls for image previews and Fit resets the view (sc-8728)", async () => {
    const noop = () => {};
    const asset = {
      id: "asset-zoom",
      projectId: "project-1",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/plate.png" },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    const controls = document.body.querySelector(".preview-zoom-controls");
    expect(controls).not.toBeNull();
    const inner = document.body.querySelector(".preview-zoom-inner");
    // Starts at fit (identity transform); Zoom out is disabled at min scale.
    expect(inner.style.transform).toContain(`scale(${PREVIEW_FIT_VIEW.scale})`);
    const zoomOut = controls.querySelector('[aria-label="Zoom out"]');
    const zoomIn = controls.querySelector('[aria-label="Zoom in"]');
    const fit = controls.querySelector('[aria-label="Fit to view"]');
    expect(zoomOut.disabled).toBe(true);

    await act(async () => {
      zoomIn.click();
    });
    // Scale increased above fit; zoom out is now enabled.
    expect(inner.style.transform).not.toContain(`scale(${PREVIEW_MIN_SCALE})`);
    expect(controls.querySelector('[aria-label="Zoom out"]').disabled).toBe(false);

    await act(async () => {
      fit.click();
    });
    expect(document.body.querySelector(".preview-zoom-inner").style.transform).toContain(`scale(${PREVIEW_FIT_VIEW.scale})`);
  });

  it("resets zoom to fit when switching to a different asset (sc-8728)", async () => {
    const noop = () => {};
    const base = {
      projectId: "project-1",
      displayName: "Plate",
      type: "image",
      status: {},
    };
    const assetA = { ...base, id: "asset-a", file: { path: "assets/images/a.png" } };
    const assetB = { ...base, id: "asset-b", file: { path: "assets/images/b.png" } };

    root = createRoot(container);
    const renderWith = (asset) =>
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );

    await act(async () => {
      renderWith(assetA);
    });
    await act(async () => {
      document.body.querySelector('.preview-zoom-controls [aria-label="Zoom in"]').click();
    });
    expect(document.body.querySelector(".preview-zoom-inner").style.transform).not.toContain(`scale(${PREVIEW_MIN_SCALE})`);

    // Navigating to a new asset id must reset the view back to fit.
    await act(async () => {
      renderWith(assetB);
    });
    expect(document.body.querySelector(".preview-zoom-inner").style.transform).toContain(`scale(${PREVIEW_FIT_VIEW.scale})`);
  });

  it("renders native video without any zoom UI (sc-8728)", async () => {
    const noop = () => {};
    const asset = {
      id: "asset-video",
      projectId: "project-1",
      displayName: "Clip",
      type: "video",
      status: {},
      file: { path: "assets/videos/clip.mp4", mimeType: "video/mp4" },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    expect(document.body.querySelector(".preview-modal video")).not.toBeNull();
    expect(document.body.querySelector(".preview-modal video").hasAttribute("controls")).toBe(true);
    expect(document.body.querySelector(".preview-zoom-controls")).toBeNull();
    expect(document.body.querySelector(".preview-zoom-viewport")).toBeNull();
  });

  it("offers recipe reuse from FullscreenPreview image assets", async () => {
    const noop = () => {};
    const onUseRecipe = vi.fn();
    const asset = {
      id: "asset-recipe",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/plate.png" },
      generationSet: {
        recipe: {
          mode: "text_to_image",
          model: "z_image_turbo",
          prompt: "mist over a glass atrium",
        },
      },
      recipe: { prompt: "asset fallback" },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          onUseRecipe={onUseRecipe}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use this recipe").click();
    });

    // Second arg carries the keep-seed choice; this asset has no seed so the toggle
    // is hidden and defaults to false (random-seed variation).
    expect(onUseRecipe).toHaveBeenCalledWith(asset, { keepSeed: false });
  });

  // sc-12324: a generated clip records a full recipe, but the affordance was hard-gated to
  // images, so there was no way to re-run one. Videos re-run exactly as images do — keep-seed
  // included, which is the explicit ask.
  it("offers recipe reuse, with keep-seed, from FullscreenPreview video assets", async () => {
    const noop = () => {};
    const onUseRecipe = vi.fn();
    const asset = {
      id: "asset-clip",
      displayName: "Hero walks through rain",
      type: "video",
      status: {},
      file: { path: "assets/videos/hero.mp4" },
      recipe: {
        mode: "text_to_video",
        model: "wan_i2v",
        prompt: "the hero walks through rain",
        seed: 4242,
      },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          onUseRecipe={onUseRecipe}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    // The clip records a seed, so the keep-seed toggle is offered — turn it on and reuse.
    await act(async () => {
      document.body.querySelector(".preview-keep-seed input").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use this recipe").click();
    });

    expect(onUseRecipe).toHaveBeenCalledWith(asset, { keepSeed: true });
  });

  it("resolves each image's own seed, preferring the per-asset recipe over the shared set seed", () => {
    // Two siblings of one generation set: the set recipe carries the batch's base seed
    // (7), while each image records its own seed. assetSeed must return the per-image
    // value so navigating between siblings updates the seed (and reuse reproduces the
    // exact image), not the constant set seed.
    const setRecipe = { model: "z_image_turbo", seed: 7 };
    const first = { id: "a", recipe: { model: "z_image_turbo", seed: 7 }, generationSet: { recipe: setRecipe } };
    const second = { id: "b", recipe: { model: "z_image_turbo", seed: 42 }, generationSet: { recipe: setRecipe } };
    expect(assetSeed(first)).toBe(7);
    expect(assetSeed(second)).toBe(42);
    // Falls back to the set seed when the asset has no per-asset recipe, and honors seed 0.
    expect(assetSeed({ id: "c", generationSet: { recipe: setRecipe } })).toBe(7);
    expect(assetSeed({ id: "d", recipe: { seed: 0 } })).toBe(0);
    expect(assetSeed({ id: "e" })).toBeNull();
  });

  it("offers a keep-seed toggle only when the asset carries a seed", async () => {
    const noop = () => {};
    const onUseRecipe = vi.fn();
    const asset = {
      id: "asset-seed",
      displayName: "Atrium",
      type: "image",
      status: {},
      file: { path: "assets/images/atrium.png" },
      recipe: { mode: "text_to_image", model: "z_image_turbo", prompt: "mist", seed: 1234 },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          onUseRecipe={onUseRecipe}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    // The toggle appears because the recipe carries a seed.
    expect(document.body.querySelector(".preview-keep-seed")).not.toBeNull();

    // Toggle keep-seed on, then reuse the recipe — the choice rides on onUseRecipe.
    await act(async () => {
      document.body.querySelector(".preview-keep-seed input[type=\"checkbox\"]").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use this recipe").click();
    });

    expect(onUseRecipe).toHaveBeenCalledWith(asset, { keepSeed: true });
  });

  // sc-8730: the FullscreenPreview "Edit" button invokes onEditImage, which App now
  // wires to sendAssetToImageEditor (the Image Editor canvas route) instead of the
  // Image Studio edit_image route. This proves the button → onEditImage contract the
  // repointed call site depends on.
  it("routes the FullscreenPreview Edit button through onEditImage", async () => {
    const noop = () => {};
    const onEditImage = vi.fn();
    const asset = {
      id: "asset-edit",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/plate.png" },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onEditImage={onEditImage}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Edit").click();
    });

    expect(onEditImage).toHaveBeenCalledWith(asset);
  });

  it("reports the scroll direction when navigating the FullscreenPreview", async () => {
    const noop = () => {};
    const onPreviewAsset = vi.fn();
    const asset = { id: "asset-b", displayName: "Plate", type: "image", status: {}, file: { path: "b.png" } };
    const previous = { id: "asset-a", displayName: "Prev", type: "image", status: {}, file: { path: "a.png" } };
    const next = { id: "asset-c", displayName: "Next", type: "image", status: {}, file: { path: "c.png" } };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={next}
          onClose={noop}
          onPreviewAsset={onPreviewAsset}
          previousAsset={previous}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    await act(async () => {
      document.body.querySelector(".preview-nav-button.next").click();
    });
    expect(onPreviewAsset).toHaveBeenLastCalledWith(next, "next");

    await act(async () => {
      document.body.querySelector(".preview-nav-button.previous").click();
    });
    expect(onPreviewAsset).toHaveBeenLastCalledWith(previous, "previous");
  });

});
