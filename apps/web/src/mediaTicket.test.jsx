// sc-8810: media URLs must authenticate in remote-auth mode. Element-driven
// requests (<img src>, <video src>, <a download>) cannot send the token header,
// so every media-URL producer appends the short-lived media ticket minted from
// POST /api/v1/files/ticket. These tests pin the central helper and the
// producers that ride on it (assetUrl/posterUrl, DocumentReader's fetch).
import React, { act, startTransition, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { API_BASE_URL, getMediaTicket, setMediaTicket, withMediaTicket } from "./api.js";
import { mediaResponseWithTicketRetry } from "./api/media.js";
import { AssetMedia, AssetThumbnail, assetUrl, posterUrl } from "./components/assetMedia.jsx";
import { AssetDetail } from "./components/assetPanels.jsx";
import { keypointSourceImageUrl } from "./keypointLibrary.js";

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

afterEach(() => {
  setMediaTicket("");
});

describe("withMediaTicket", () => {
  it("is a no-op while no ticket is set (desktop/loopback and auth-off modes)", () => {
    setMediaTicket("");
    expect(withMediaTicket("http://x/api/v1/projects/p1/files/a.png")).toBe(
      "http://x/api/v1/projects/p1/files/a.png",
    );
    expect(withMediaTicket("")).toBe("");
  });

  it("appends the ticket as a query param, joining on existing query strings", () => {
    setMediaTicket("abc123");
    expect(getMediaTicket()).toBe("abc123");
    expect(withMediaTicket("http://x/files/a.png")).toBe("http://x/files/a.png?ticket=abc123");
    expect(withMediaTicket("http://x/files/a.png?v=2")).toBe("http://x/files/a.png?v=2&ticket=abc123");
  });

  it("leaves empty URLs alone even with a ticket set", () => {
    setMediaTicket("abc123");
    expect(withMediaTicket("")).toBe("");
    expect(withMediaTicket(null)).toBe(null);
  });
});

describe("in-flight media reads", () => {
  it("retries a 401 only if the owned URL has acquired a new ticket", async () => {
    let release;
    const original = global.fetch;
    global.fetch = vi.fn().mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }))
      .mockResolvedValueOnce({ ok: true, status: 200 });
    try {
      setMediaTicket("old");
      const pending = mediaResponseWithTicketRetry(() => assetUrl(imageAsset));
      setMediaTicket("new");
      release({ ok: false, status: 401 });
      expect((await pending).status).toBe(200);
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(global.fetch.mock.calls[1][0]).toContain("ticket=new");
      global.fetch.mockClear().mockResolvedValue({ ok: false, status: 401 });
      expect((await mediaResponseWithTicketRetry(() => "blob:unchanged")).status).toBe(401);
      expect(global.fetch).toHaveBeenCalledTimes(1);
    } finally { global.fetch = original; }
  });
});

describe("asset URL producers carry the media ticket", () => {
  it("assetUrl appends the ticket to file-path URLs", () => {
    setMediaTicket("t0ken");
    expect(assetUrl(imageAsset)).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/one.png?ticket=t0ken`,
    );
  });

  it("assetUrl appends the ticket to normalized `url` assets", () => {
    setMediaTicket("t0ken");
    expect(assetUrl({ ...imageAsset, url: "/api/v1/projects/p1/files/assets/one.png" })).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/one.png?ticket=t0ken`,
    );
  });

  it("assetUrl stays bare without a ticket", () => {
    expect(assetUrl(imageAsset)).toBe(`${API_BASE_URL}/api/v1/projects/p1/files/assets/one.png`);
  });

  it("posterUrl swaps the extension BEFORE the ticket query string", () => {
    setMediaTicket("t0ken");
    expect(posterUrl(videoAsset)).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/clip.poster.jpg?ticket=t0ken`,
    );
  });

  it("keypointSourceImageUrl appends the ticket", () => {
    setMediaTicket("t0ken");
    const url = keypointSourceImageUrl("assets/keypoints/asset_x.png");
    expect(url).toContain("/files/assets/keypoints/asset_x.png?ticket=t0ken");
  });
});

// sc-9063: when the media-ticket mint fails, data loads anyway and un-ticketed
// thumbnails 401 into the "deleted" placeholder. The placeholder must be
// per-URL, not permanent — once a retried mint lands and the src gains the
// ticket query param, the thumbnail retries instead of sticking on the marker.
describe("thumbnail placeholders retry when the media URL changes (sc-9063)", () => {
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

  it("recovers from the deleted-asset marker once a ticketed URL arrives", async () => {
    setMediaTicket("");
    await act(async () => root.render(<AssetThumbnail asset={imageAsset} />));
    const img = container.querySelector("img");
    expect(img).not.toBeNull();

    // Un-ticketed URL 401s in remote-auth mode → thumbnail degrades to the marker.
    await act(async () => {
      img.dispatchEvent(new window.Event("error"));
    });
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector(".asset-thumb-missing")).not.toBeNull();

    // A mint retry lands: the ticketed src is a new URL, so the thumb retries.
    setMediaTicket("late-ticket");
    await act(async () => root.render(<AssetThumbnail asset={{ ...imageAsset }} />));
    const retried = container.querySelector("img");
    expect(retried).not.toBeNull();
    expect(retried.src).toContain("ticket=late-ticket");
    expect(container.querySelector(".asset-thumb-missing")).toBeNull();
  });
});

describe("DocumentReader authenticates its document fetch (sc-8810)", () => {
  let container;
  let root;
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    global.fetch = originalFetch;
  });

  it("keeps an in-flight document read and retries only a stale unauthorized request", async () => {
    setMediaTicket("old-ticket");
    let release;
    global.fetch = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }))
      .mockResolvedValueOnce({ ok: true, json: async () => ({ segments: [{ type: "text", text: "current" }] }) });
    const asset = { id: "doc", type: "document", projectId: "p1", file: { path: "assets/doc.json", mimeType: "application/json" } };
    await act(async () => root.render(<AssetDetail asset={asset} availableTags={[]} />));
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch.mock.calls[0][0]).toContain("ticket=old-ticket");
    await act(async () => setMediaTicket("new-ticket"));
    expect(global.fetch).toHaveBeenCalledTimes(1);
    await act(async () => { release({ ok: false, status: 401 }); });
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch.mock.calls[1][0]).toContain("ticket=new-ticket");
    expect(container.textContent).toContain("current");
    await act(async () => setMediaTicket("next-ticket"));
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  it("retries a completed unauthorized document read when a fresh ticket arrives", async () => {
    setMediaTicket("expired-ticket");
    global.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 401 })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ segments: [{ type: "text", text: "recovered" }] }) });
    const asset = { id: "doc", type: "document", projectId: "p1", file: { path: "assets/doc.json", mimeType: "application/json" } };
    await act(async () => root.render(<AssetDetail asset={asset} availableTags={[]} />));
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch.mock.calls[0][0]).toContain("ticket=expired-ticket");
    expect(container.textContent).toContain("401");
    await act(async () => setMediaTicket("fresh-ticket"));
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch.mock.calls[1][0]).toContain("ticket=fresh-ticket");
    expect(container.textContent).toContain("recovered");
  });

  it("does not retry an in-flight read against a suspended replacement document", async () => {
    let release;
    const never = new Promise(() => {});
    global.fetch = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }))
      .mockResolvedValueOnce({ ok: true, json: async () => ({ segments: [] }) });
    const asset = (name) => ({ id: name, type: "document", projectId: "p1", file: { path: `assets/${name}.json`, mimeType: "application/json" } });
    function Block({ pending }) { if (pending) throw never; return null; }
    function Tree({ name, pending }) {
      return <Suspense fallback={null}><AssetDetail asset={asset(name)} availableTags={[]} /><Block pending={pending} /></Suspense>;
    }
    setMediaTicket("old-ticket");
    await act(async () => root.render(<Tree name="first" pending={false} />));
    await act(async () => setMediaTicket("new-ticket"));
    await act(async () => { startTransition(() => root.render(<Tree name="second" pending />)); });
    await act(async () => { release({ ok: false, status: 401 }); });
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch.mock.calls[1][0]).toContain("/assets/first.json?ticket=new-ticket");
    expect(global.fetch.mock.calls[1][0]).not.toContain("second.json");
  });

  it.each([401, 500, 200])("does not retarget or publish an old %s response after a replacement document commits", async (oldStatus) => {
    const NativeAbortController = globalThis.AbortController;
    const pendingAborts = [];
    // Hold passive cleanup's abort to exercise the commit-to-cleanup ownership window.
    class DeferredAbortController extends NativeAbortController {
      abort(...args) { pendingAborts.push(() => super.abort(...args)); }
    }
    vi.stubGlobal("AbortController", DeferredAbortController);
    let releaseFirst;
    let releaseSecond;
    global.fetch = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { releaseFirst = resolve; }))
      .mockImplementationOnce(() => new Promise((resolve) => { releaseSecond = resolve; }))
      .mockResolvedValue({ ok: true, json: async () => ({ segments: [{ type: "text", text: "wrong retry" }] }) });
    const asset = (name) => ({ id: name, type: "document", projectId: "p1", file: { path: `assets/${name}.json`, mimeType: "application/json" } });
    try {
      setMediaTicket("same-ticket");
      await act(async () => root.render(<AssetDetail asset={asset("first")} availableTags={[]} />));
      await act(async () => root.render(<AssetDetail asset={asset("second")} availableTags={[]} />));
      expect(global.fetch).toHaveBeenCalledTimes(2);
      await act(async () => { releaseFirst({
        ok: oldStatus === 200,
        status: oldStatus,
        json: async () => ({ segments: [{ type: "text", text: "stale first document" }] }),
      }); });
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(container.textContent).not.toContain("stale first document");
      expect(container.textContent).not.toContain("wrong retry");
      expect(container.textContent).not.toContain("Couldn't load document");
      await act(async () => { releaseSecond({ ok: true, json: async () => ({ segments: [{ type: "text", text: "second document" }] }) }); });
      expect(container.textContent).toContain("second document");
    } finally {
      pendingAborts.forEach((abort) => abort());
      vi.unstubAllGlobals();
    }
  });

  it("fetches the document through a ticketed URL", async () => {
    setMediaTicket("d0cticket");
    global.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ segments: [{ type: "text", text: "hello" }] }),
    }));
    const documentAsset = {
      id: "doc",
      type: "document",
      displayName: "story.json",
      file: { path: "assets/documents/story.json", mimeType: "application/json" },
      projectId: "p1",
    };
    await act(async () =>
      root.render(
        <AssetDetail
          asset={documentAsset}
          deleteAsset={() => {}}
          purgeAsset={() => {}}
          onPreview={() => {}}
          onSendImage={() => {}}
          onSendVideo={() => {}}
          onSendEditor={() => {}}
          updateAssetStatus={() => {}}
          updateAssetTags={() => {}}
          availableTags={[]}
        />,
      ),
    );
    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url] = global.fetch.mock.calls[0];
    expect(url).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/documents/story.json?ticket=d0cticket`,
    );
    expect(container.textContent).toContain("hello");
  });
});

describe("finite media-credential rotation", () => {
  it("updates mounted playback and retains position and playing state for the same file", async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    const ref = React.createRef();
    const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
    try {
      setMediaTicket("old-client-ticket");
      act(() => root.render(<AssetMedia asset={videoAsset} ref={ref} />));
      const video = container.querySelector("video");
      expect(ref.current).toBe(video);
      video.currentTime = 42;
      video.playbackRate = 1.5;
      Object.defineProperty(video, "paused", { configurable: true, value: false });
      act(() => setMediaTicket("new-client-ticket"));
      expect(video.src).toContain("ticket=new-client-ticket");
      video.currentTime = 0; // the browser resets it while loading the new URL
      act(() => video.dispatchEvent(new Event("loadedmetadata")));
      expect(video.currentTime).toBe(42);
      expect(video.playbackRate).toBe(1.5);
      expect(play).toHaveBeenCalledOnce();
      act(() => root.render(<AssetMedia asset={{ ...videoAsset, file: { path: "assets/other.mp4" } }} ref={ref} />));
      video.currentTime = 0;
      act(() => video.dispatchEvent(new Event("loadedmetadata")));
      expect(video.currentTime).toBe(0);
      expect(play).toHaveBeenCalledOnce();
    } finally {
      act(() => root.unmount());
      container.remove();
      play.mockRestore();
    }
  });
});
