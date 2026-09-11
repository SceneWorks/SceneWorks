import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AssetMedia, AssetThumbnail, MissingMedia, assetCanRenderAsImage, assetUrl, posterUrl, thumbnailUrl } from "./assetMedia.jsx";

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

const audioAsset = {
  id: "aud",
  type: "audio",
  displayName: "line.wav",
  origin: "audio_studio",
  file: { path: "assets/line.wav", mimeType: "audio/wav", duration: 3, sampleRate: 24000, channels: 1 },
  projectId: "p1",
};

const vectorAsset = {
  id: "vector", type: "vector", projectId: "p1", displayName: "mark.svg",
  file: { path: "assets/vector.svg", mimeType: "image/svg+xml" },
  preview: { path: "assets/preview.png", mimeType: "image/png" },
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
    expect(img.getAttribute("loading")).toBe("lazy");
    expect(img.getAttribute("decoding")).toBe("async");
    expect(new URL(img.src).searchParams.get("thumbnail")).toBe("384");
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
    expect(new URL(img.src).searchParams.has("thumbnail")).toBe(false);
    expect(img.hasAttribute("loading")).toBe(false);
    const event = fireContextMenu(img);
    expect(event.defaultPrevented).toBe(false);
  });
});

describe("bounded thumbnail URLs (sc-14797)", () => {
  it("uses the derivative only for shared grid thumbnails", () => {
    expect(new URL(thumbnailUrl(imageAsset)).searchParams.get("thumbnail")).toBe("384");
    expect(new URL(thumbnailUrl(videoAsset)).searchParams.get("thumbnail")).toBe("384");
    expect(new URL(posterUrl(videoAsset)).searchParams.has("thumbnail")).toBe(false);
  });
});

describe("vector PNG-only rendering (sc-22257)", () => {
  it("uses only the worker PNG preview and cannot enter generic image conditioning", () => {
    expect(assetCanRenderAsImage(vectorAsset)).toBe(false);
    expect(thumbnailUrl(vectorAsset)).toContain("assets/preview.png");
    expect(thumbnailUrl(vectorAsset)).not.toContain("vector.svg");
  });
});

// AssetMedia audio rendering (SceneWorks Audio Studio, epic 13400 A5 / sc-13405):
// a type:audio asset renders a playable <audio> element, mirroring the <video>
// branch — never an <audio> for image/video assets.
describe("AssetMedia audio rendering (epic 13400 A5)", () => {
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

  it("renders a playable <audio> element for a type:audio asset", async () => {
    await act(() => root.render(<AssetMedia asset={audioAsset} />));
    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();
    expect(audio.getAttribute("src")).toContain("assets/line.wav");
    // Audio has no poster/first-frame, so no <img> or <video> is produced.
    expect(container.querySelector("video")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
  });

  it("does NOT render an <audio> element for image or video assets", async () => {
    await act(() => root.render(<AssetMedia asset={imageAsset} />));
    expect(container.querySelector("audio")).toBeNull();
    expect(container.querySelector("img")).not.toBeNull();

    await act(() => root.render(<AssetMedia asset={videoAsset} />));
    expect(container.querySelector("audio")).toBeNull();
    expect(container.querySelector("video")).not.toBeNull();
  });

  it("uses WebKit-safe inline metadata playback for generated MP4 video", async () => {
    await act(() => root.render(<AssetMedia asset={videoAsset} />));
    const video = container.querySelector("video");
    expect(video).not.toBeNull();
    expect(video.controls).toBe(true);
    // `playsInline` is the WebKit-safe half; `muted` never was. It used to be hard-set here, which
    // made every clip in the app play silent — see `assetMedia.audio.test.jsx` (sc-17161), where
    // the joint audio+video families make that a wrong answer rather than a harmless one. Still
    // PINNED, in the other direction: a surface that needs a mute passes `muted` explicitly.
    expect(video.muted).toBe(false);
    expect(video.playsInline).toBe(true);
    expect(video.preload).toBe("metadata");
    expect(video.getAttribute("src")).toContain("clip.mp4");
  });

  it("honors controls={false} so a custom transport can drive the element", async () => {
    await act(() => root.render(<AssetMedia asset={audioAsset} controls={false} />));
    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();
    expect(audio.hasAttribute("controls")).toBe(false);
  });

  it("renders an <audio> when only the mimeType marks it as audio (no type)", async () => {
    const byMime = { id: "m", file: { path: "assets/clip.mp3", mimeType: "audio/mpeg" }, projectId: "p1" };
    await act(() => root.render(<AssetMedia asset={byMime} />));
    expect(container.querySelector("audio")).not.toBeNull();
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

// Live denoise preview frames (sc-16905) arrive as complete data: URLs. Every URL producer must
// pass them through byte-identical — an API prefix, media ticket, or thumbnail param anywhere in
// the chain corrupts the base64 body.
describe("data URL passthrough", () => {
  const asset = { id: "live-denoise-preview", type: "image", url: "data:image/jpeg;base64,QUJD", __interim: true };

  it("assetUrl returns the data URL unchanged", () => {
    expect(assetUrl(asset)).toBe(asset.url);
  });

  it("thumbnailUrl returns the data URL unchanged (no thumbnail param, no ticket)", () => {
    expect(thumbnailUrl(asset)).toBe(asset.url);
  });
});
