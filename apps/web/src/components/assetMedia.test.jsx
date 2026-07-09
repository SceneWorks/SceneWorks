import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AssetMedia, AssetThumbnail, MissingMedia, posterUrl } from "./assetMedia.jsx";

const imageAsset = {
  id: "img",
  type: "image",
  displayName: "one.png",
  file: { path: "assets/one.png", mimeType: "image/png" },
  projectId: "p1",
};

const videoAsset = {
  id: "vid",
  type: "video",
  displayName: "clip.mp4",
  file: { path: "assets/clip.mp4", mimeType: "video/mp4" },
  projectId: "p1",
};

function fireContextMenu(el) {
  const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
  el.dispatchEvent(event);
  return event;
}

describe("thumbnail native context-menu suppression (sc-8731)", () => {
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
  });

  it("suppresses the native menu on an image thumbnail", async () => {
    await act(() => root.render(<AssetThumbnail asset={imageAsset} />));
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    const event = fireContextMenu(img);
    expect(event.defaultPrevented).toBe(true);
  });

  it("suppresses the native menu on a video-poster thumbnail", async () => {
    await act(() => root.render(<AssetThumbnail asset={videoAsset} />));
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    const event = fireContextMenu(img);
    expect(event.defaultPrevented).toBe(true);
  });

  it("suppresses the native menu on the deleted-asset placeholder", async () => {
    await act(() => root.render(<MissingMedia />));
    const placeholder = container.querySelector(".asset-thumb-missing");
    expect(placeholder).not.toBeNull();
    const event = fireContextMenu(placeholder);
    expect(event.defaultPrevented).toBe(true);
  });

  it("does NOT suppress the native menu on the full-size AssetMedia (owned by sc-8729)", async () => {
    await act(() => root.render(<AssetMedia asset={imageAsset} />));
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    const event = fireContextMenu(img);
    expect(event.defaultPrevented).toBe(false);
  });
});

describe("posterUrl poster-existence gating (sc-10468)", () => {
  it("uses the server-advertised posterUrl when present", () => {
    const asset = {
      type: "video",
      url: "/api/v1/projects/p1/files/assets/clip.mp4",
      posterUrl: "/api/v1/projects/p1/files/assets/clip.poster.jpg",
      file: { path: "assets/clip.mp4", mimeType: "video/mp4" },
      projectId: "p1",
    };
    expect(posterUrl(asset)).toContain("/api/v1/projects/p1/files/assets/clip.poster.jpg");
  });

  it("returns '' for a normalized video with no poster, so it never probes .poster.jpg", () => {
    // A persisted asset always carries a server `url`; the missing posterUrl means
    // the poster genuinely does not exist — the source of the startup 404 spam.
    const asset = {
      type: "video",
      url: "/api/v1/projects/p1/files/assets/clip.mp4",
      file: { path: "assets/clip.mp4", mimeType: "video/mp4" },
      projectId: "p1",
    };
    expect(posterUrl(asset)).toBe("");
  });

  it("still derives the poster path for a transient asset without a server url", () => {
    // Live-job assets the server hasn't normalized keep the old behavior.
    const asset = {
      type: "video",
      file: { path: "assets/clip.mp4", mimeType: "video/mp4" },
      projectId: "p1",
    };
    expect(posterUrl(asset)).toContain(".poster.jpg");
  });
});
