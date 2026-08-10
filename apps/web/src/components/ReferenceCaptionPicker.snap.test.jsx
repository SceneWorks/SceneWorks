import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ReferenceCaptionPicker from "./ReferenceCaptionPicker.jsx";

// sc-18255: the describe/caption picker shares the reference aspect auto-preset seam with Image
// Studio's img2img panel. Its probe is keyed on the asset LIST as well as the reference id, and the
// list gets a new array identity on every asset refresh — so an unguarded re-run re-snapped the
// Aspect over whatever the user had chosen since. The snap must fire once per (reference, snapKey).
const REFERENCE_ASSET = {
  id: "asset-reference",
  projectId: "project-1",
  type: "image",
  displayName: "Portrait reference",
  file: { path: "assets/uploads/portrait.jpg", mimeType: "image/jpeg" },
  status: { favorite: false, rating: 0, rejected: false, trashed: false },
};

// jsdom's Image never loads, so probeImageDimensions would never fire its callback.
function stubImageDecode(naturalWidth, naturalHeight) {
  class ProbeImage {
    constructor() {
      this.naturalWidth = 0;
      this.naturalHeight = 0;
      this.onload = null;
      this.onerror = null;
    }
    set src(value) {
      this._src = value;
      if (!value) return;
      Promise.resolve().then(() => {
        this.naturalWidth = naturalWidth;
        this.naturalHeight = naturalHeight;
        this.onload?.();
      });
    }
    get src() {
      return this._src;
    }
  }
  global.Image = ProbeImage;
}

describe("ReferenceCaptionPicker aspect snap", () => {
  let container;
  let root;
  const originalImage = global.Image;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    stubImageDecode(1080, 1920);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    global.Image = originalImage;
    vi.restoreAllMocks();
  });

  const buttonByText = (text) =>
    [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === text);

  function picker(assets, onReferenceImageLoaded, snapKey) {
    return (
      <ReferenceCaptionPicker
        onCaption={() => Promise.resolve("")}
        onApply={() => {}}
        onReferenceImageLoaded={onReferenceImageLoaded}
        referenceSnapKey={snapKey}
        referenceAssets={assets}
        projectId="project-1"
      />
    );
  }

  // The picker's modal renders into document.body, outside the container.
  const modalButtonByText = (text) =>
    [...document.body.querySelectorAll(".asset-picker-modal button")].find(
      (button) => button.textContent.trim() === text,
    );

  async function pick() {
    await act(async () =>
      buttonByText("Select reference image").dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    await act(async () =>
      document.body
        .querySelector(".asset-picker-modal .asset-picker-card")
        .dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    await act(async () =>
      modalButtonByText("Use Selection").dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
  }

  it("snaps once per reference, not on every asset-list refresh (sc-18255)", async () => {
    const onReferenceImageLoaded = vi.fn();
    await act(async () => root.render(picker([REFERENCE_ASSET], onReferenceImageLoaded, "1024x1024,768x1024")));
    await pick();
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);
    expect(onReferenceImageLoaded).toHaveBeenCalledWith(1080, 1920);

    // Two asset-list refreshes with the same reference still picked: no further snaps.
    await act(async () =>
      root.render(picker([{ ...REFERENCE_ASSET }], onReferenceImageLoaded, "1024x1024,768x1024")),
    );
    await act(async () =>
      root.render(
        picker(
          [{ ...REFERENCE_ASSET }, { ...REFERENCE_ASSET, id: "asset-generated" }],
          onReferenceImageLoaded,
          "1024x1024,768x1024",
        ),
      ),
    );
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);
  });

  it("snaps again when the caller's option set changes (sc-18255)", async () => {
    const onReferenceImageLoaded = vi.fn();
    await act(async () => root.render(picker([REFERENCE_ASSET], onReferenceImageLoaded, "1024x1024,768x1024")));
    await pick();
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(1);

    // A real option-set change (the user switched models) must re-snap onto the new buckets.
    await act(async () =>
      root.render(picker([REFERENCE_ASSET], onReferenceImageLoaded, "1024x1024,896x1600")),
    );
    expect(onReferenceImageLoaded).toHaveBeenCalledTimes(2);
  });
});
