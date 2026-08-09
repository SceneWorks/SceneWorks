import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resolutionUniverseKey, useReferenceAspectSnap } from "./useReferenceAspectSnap.js";

// Unit coverage for the shared guard behind Image Studio's img2img panel and the describe picker.
// The screen-level tests (App.imageStudioReferenceSnap, ReferenceCaptionPicker.snap) drive it
// through the UI; these pin the cases the UI cannot stage — a reference whose asset resolves a
// render later (sc-8220), a probe that never loads, and a missing callback.
const ASSET = {
  id: "asset-reference",
  projectId: "project-1",
  type: "image",
  displayName: "Portrait reference",
  file: { path: "assets/uploads/portrait.jpg", mimeType: "image/jpeg" },
  status: { favorite: false, rating: 0, rejected: false, trashed: false },
};

// jsdom's Image never loads. `loads` false models a reference whose URL never decodes.
function stubImageDecode({ width = 1080, height = 1920, loads = true } = {}) {
  class ProbeImage {
    constructor() {
      this.naturalWidth = 0;
      this.naturalHeight = 0;
      this.onload = null;
      this.onerror = null;
    }
    set src(value) {
      this._src = value;
      if (!value || !loads) return;
      Promise.resolve().then(() => {
        this.naturalWidth = width;
        this.naturalHeight = height;
        this.onload?.();
      });
    }
    get src() {
      return this._src;
    }
  }
  global.Image = ProbeImage;
}

function Probe(props) {
  useReferenceAspectSnap(props);
  return null;
}

describe("resolutionUniverseKey", () => {
  const MODEL = { id: "krea_2_turbo", limits: { resolutions: ["1024x1024", "768x1024", "2048x2048"] } };

  it("is a value, so an equal-but-new catalog object is inert", () => {
    expect(resolutionUniverseKey(MODEL)).toBe(
      resolutionUniverseKey({ ...MODEL, limits: { resolutions: [...MODEL.limits.resolutions] } }),
    );
  });

  it("changes when the model changes", () => {
    expect(resolutionUniverseKey(MODEL)).not.toBe(
      resolutionUniverseKey({ ...MODEL, id: "z_image_turbo" }),
    );
    // Same id, different declared buckets (a manifest update) is also a real change.
    expect(resolutionUniverseKey(MODEL)).not.toBe(
      resolutionUniverseKey({ ...MODEL, limits: { resolutions: ["1024x1024"] } }),
    );
  });

  it("reflects the DECLARED buckets, so gating cannot reach it", () => {
    // The regression guard for the sc-16020 tier axis: the key names the model's declared universe,
    // so switching quant tier — which re-gates the picker's visible list, trimming >1536² buckets —
    // leaves the key untouched and cannot re-fire the snap over the user's Aspect.
    expect(resolutionUniverseKey(MODEL)).toBe("krea_2_turbo|1024x1024,768x1024,2048x2048");
    // A gated list would have dropped 2048x2048; the key keeps it.
    expect(resolutionUniverseKey(MODEL)).toContain("2048x2048");
  });

  it("falls back to the caller's default list when the model declares none", () => {
    expect(resolutionUniverseKey(undefined, ["1024x1024"])).toBe("|1024x1024");
    expect(resolutionUniverseKey({ id: "m" }, ["1024x1024"])).toBe("m|1024x1024");
  });
});

describe("useReferenceAspectSnap", () => {
  let container;
  let root;
  const originalImage = global.Image;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    stubImageDecode();
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    global.Image = originalImage;
    vi.restoreAllMocks();
  });

  const render = (props) => act(async () => root.render(<Probe {...props} />));

  it("snaps once the asset resolves a render later, and not again (sc-8220 import/drag path)", async () => {
    const onReferenceImageLoaded = vi.fn();
    // The id is set a render BEFORE the asset lands in the list — the import/drag shape.
    await render({ referenceId: ASSET.id, assets: [], onReferenceImageLoaded, snapKey: "a" });
    expect(onReferenceImageLoaded).not.toHaveBeenCalled();

    await render({ referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "a" });
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);
    expect(onReferenceImageLoaded).toHaveBeenCalledWith(1080, 1920);

    // Further list churn must not re-snap.
    await render({ referenceId: ASSET.id, assets: [{ ...ASSET }], onReferenceImageLoaded, snapKey: "a" });
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);
  });

  it("stays eligible when the probe never loads", async () => {
    stubImageDecode({ loads: false });
    const onReferenceImageLoaded = vi.fn();
    await render({ referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "a" });
    expect(onReferenceImageLoaded).not.toHaveBeenCalled();

    // A failed probe must not consume the pick: once decoding works, the snap still happens.
    stubImageDecode();
    await render({ referenceId: ASSET.id, assets: [{ ...ASSET }], onReferenceImageLoaded, snapKey: "a" });
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);
  });

  it("re-snaps for a new snapKey but not for a repeated one", async () => {
    const onReferenceImageLoaded = vi.fn();
    await render({ referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "a" });
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);

    await render({ referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "b" });
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(2);

    await render({ referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "b" });
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(2);
  });

  it("suppresses while disabled without re-arming", async () => {
    const onReferenceImageLoaded = vi.fn();
    const props = { referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "a" };
    await render(props);
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);

    // Inoperative, not cleared — the caller's panel is hidden. Going back must not re-snap: that is
    // the Text -> Edit -> Text round-trip that re-clobbered the user's Aspect.
    await render({ ...props, enabled: false });
    await render(props);
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);
  });

  it("clearing the reference re-arms it", async () => {
    const onReferenceImageLoaded = vi.fn();
    await render({ referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "a" });
    await render({ referenceId: "", assets: [ASSET], onReferenceImageLoaded, snapKey: "a" });
    await render({ referenceId: ASSET.id, assets: [ASSET], onReferenceImageLoaded, snapKey: "a" });
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(2);
  });

  it("does nothing without a callback instead of throwing out of an image onload", async () => {
    // The hook wraps the callback, which defeats probeImageDimensions' own callable check, and a
    // throw from onload lands outside React's tree where no error boundary catches it.
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    const uncaught = vi.fn();
    process.on("uncaughtException", uncaught);
    try {
      await render({ referenceId: ASSET.id, assets: [ASSET], snapKey: "a" });
      await act(async () => {});
    } finally {
      process.off("unhandledRejection", unhandled);
      process.off("uncaughtException", uncaught);
    }
    expect(unhandled).not.toHaveBeenCalled();
    expect(uncaught).not.toHaveBeenCalled();
  });
});
