import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MediaBin } from "./MediaBin.jsx";
import { ProgramMonitor } from "./ProgramMonitor.jsx";
import { StoryboardStrip } from "./StoryboardStrip.jsx";

const videoAsset = {
  id: "video-1",
  type: "video",
  displayName: "Generated clip",
  origin: "video_studio",
  projectId: "project-1",
  url: "/api/v1/projects/project-1/files/generated.mp4",
  posterUrl: "/api/v1/projects/project-1/files/generated.poster.jpg",
  file: { path: "generated.mp4", mimeType: "video/mp4" },
};

describe("editor thumbnail surfaces (sc-13632)", () => {
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

  it("renders video posters in MediaBin without mounting full video media", async () => {
    const onAddToTrack = vi.fn();
    await act(() => root.render(<MediaBin assets={[videoAsset]} onAddToTrack={onAddToTrack} />));

    const poster = container.querySelector(".ve-bin-thumb");
    expect(poster?.tagName).toBe("IMG");
    expect(poster.getAttribute("src")).toContain("generated.poster.jpg");
    expect(container.querySelector("video")).toBeNull();
    expect(container.querySelector('[aria-label="AI generated"]')).not.toBeNull();

    await act(() => container.querySelector(".ve-bin-item").click());
    expect(onAddToTrack).toHaveBeenCalledWith(videoAsset);
  });

  it("makes the MediaBin cap visible and lets users reveal the remaining assets", async () => {
    const assets = Array.from({ length: 61 }, (_, index) => ({
      ...videoAsset,
      id: `video-${index}`,
      displayName: `Clip ${index}`,
    }));
    await act(() => root.render(<MediaBin assets={assets} />));

    expect(container.querySelectorAll(".ve-bin-item")).toHaveLength(60);
    expect(container.textContent).toContain("Showing 60 of 61");

    const showMore = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Show more",
    );
    await act(() => showMore.click());

    expect(container.querySelectorAll(".ve-bin-item")).toHaveLength(61);
    expect(container.textContent).toContain("Showing 61 of 61");
  });

  it("renders video posters in StoryboardStrip without mounting full video media", async () => {
    const clip = {
      id: "clip-1",
      assetId: videoAsset.id,
      displayName: "Opening shot",
      timelineStart: 0,
      timelineEnd: 2,
    };
    const onSelectKey = vi.fn();
    await act(() =>
      root.render(
        <StoryboardStrip
          assetsById={new Map([[videoAsset.id, videoAsset]])}
          clips={[clip]}
          onSelectKey={onSelectKey}
          selectedItemId={clip.id}
        />,
      ),
    );

    const poster = container.querySelector(".ve-key-media");
    expect(poster?.tagName).toBe("IMG");
    expect(poster.getAttribute("src")).toContain("generated.poster.jpg");
    expect(container.querySelector("video")).toBeNull();
    expect(container.querySelector(".ve-key").classList.contains("selected")).toBe(true);

    await act(() => container.querySelector(".ve-key").click());
    expect(onSelectKey).toHaveBeenCalledWith(clip);
  });

  it("keeps ProgramMonitor on full video media for playback", async () => {
    await act(() =>
      root.render(
        <ProgramMonitor
          selectedAsset={videoAsset}
          aspectClass="landscape"
          clipLabel="Opening shot"
          isPlaying={false}
        />,
      ),
    );

    const video = container.querySelector(".ve-program-media");
    expect(video?.tagName).toBe("VIDEO");
    expect(video.getAttribute("src")).toContain("generated.mp4");
    expect(video.getAttribute("poster")).toContain("generated.poster.jpg");
  });
});
