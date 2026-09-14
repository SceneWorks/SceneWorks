// sc-18790: OpenEXR assets are stills the browser cannot decode.
//
// The two properties that matter, and the reason they are separate:
//   1. PREVIEW paints the server's tone-mapped derivative, because an <img> pointed at a
//      scene-linear float EXR renders nothing in any browser.
//   2. DOWNLOAD is untouched — assetUrl() still resolves to the original bytes. If preview and
//      download ever collapsed onto one URL, the user would silently receive a tone-mapped 8-bit
//      proxy in place of the grading asset, which is the whole failure this story exists to avoid.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import {
  AssetMedia,
  assetDisplayUrl,
  assetIsHdrSource,
  assetNativeSize,
  assetUrl,
} from "./assetMedia.jsx";

const exrAsset = {
  id: "hdr",
  type: "image",
  displayName: "frame_00000.exr",
  file: { path: "assets/frame_00000.exr", mimeType: "image/x-exr" },
  projectId: "p1",
};

const pngAsset = {
  id: "sdr",
  type: "image",
  displayName: "one.png",
  file: { path: "assets/one.png", mimeType: "image/png" },
  projectId: "p1",
};

let container = null;

function render(element) {
  container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return container;
}

afterEach(() => {
  if (container) {
    container.remove();
    container = null;
  }
});

describe("HDR (OpenEXR) asset media", () => {
  it("recognizes an EXR by mime and by path, and nothing else", () => {
    expect(assetIsHdrSource(exrAsset)).toBe(true);
    // A stored asset whose mime was never recorded still resolves by extension.
    expect(
      assetIsHdrSource({ file: { path: "assets/plate.EXR" }, projectId: "p1" }),
    ).toBe(true);
    expect(assetIsHdrSource(pngAsset)).toBe(false);
    expect(assetIsHdrSource(null)).toBe(false);
  });

  it("routes preview to the derivative while download keeps the original bytes", () => {
    const display = assetDisplayUrl(exrAsset);
    const download = assetUrl(exrAsset);

    expect(display).toContain("thumbnail=");
    expect(display).not.toBe(download);
    // The download URL must still address the stored .exr itself — never the derivative.
    expect(download).toContain("frame_00000.exr");
    expect(download).not.toContain("thumbnail=");

    // An ordinary image is unaffected: preview and download are the same original URL, so this
    // change cannot have quietly rerouted every still through the thumbnail cache.
    expect(assetDisplayUrl(pngAsset)).toBe(assetUrl(pngAsset));
  });

  it("paints the derivative and labels the preview as a proxy", () => {
    const root = render(<AssetMedia asset={exrAsset} />);
    const img = root.querySelector("img");
    expect(img).not.toBeNull();
    expect(img.getAttribute("src")).toContain("thumbnail=");
    expect(img.getAttribute("title")).toMatch(/HDR source/i);
  });

  it("leaves an ordinary image untouched", () => {
    const root = render(<AssetMedia asset={pngAsset} />);
    const img = root.querySelector("img");
    expect(img).not.toBeNull();
    expect(img.getAttribute("src")).not.toContain("thumbnail=");
    expect(img.getAttribute("title")).toBeNull();
  });
});

describe("HDR native dimensions", () => {
  it("prefers the server-recorded size over decoding a bounded derivative", () => {
    // The derivative is capped at 384px, so reading ITS naturalWidth would size an edit job from a
    // thumbnail. The recorded value is the only correct source for a format the browser cannot
    // decode at all.
    expect(
      assetNativeSize({ file: { path: "a.exr", mimeType: "image/x-exr", width: 1920, height: 1080 } }),
    ).toEqual({ width: 1920, height: 1080 });
  });

  it("returns null when no size was recorded, rather than inventing one", () => {
    expect(assetNativeSize(exrAsset)).toBeNull();
    expect(assetNativeSize({ file: { width: 0, height: 0 } })).toBeNull();
    expect(assetNativeSize(null)).toBeNull();
  });
});
