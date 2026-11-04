import React, { act, createRef } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { setMediaTicket } from "../../api.js";
import { TimelineAudio } from "./TimelineAudio.jsx";

const playback = vi.hoisted(() => ({ attach: vi.fn(), update: vi.fn(), pause: vi.fn(), dispose: vi.fn(), mediaError: vi.fn() }));
vi.mock("../../timelineAudioPlayback.js", () => ({ createTimelineAudioPlayback: () => playback }));

it.each(["audio", "video"])("rotates a %s-backed timeline audio element without losing position or play state", async (type) => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  const asset = { id: "a", type, projectId: "p", file: { path: type === "audio" ? "a.wav" : "a.mp4", mimeType: type === "audio" ? "audio/wav" : "video/mp4" } };
  const placement = { key: "layer", assetId: "a", track: { id: "track" }, item: {} };
  try {
    setMediaTicket("old");
    await act(async () => root.render(<TimelineAudio ref={createRef()} plan={{ placements: [placement] }} assetsById={new Map([["a", asset]])} time={0} playing />));
    const media = container.querySelector("audio");
    expect(media).not.toBeNull();
    expect(container.querySelector("video")).toBeNull();
    media.currentTime = 12;
    media.playbackRate = 1.25;
    Object.defineProperty(media, "paused", { configurable: true, value: false });
    await act(async () => setMediaTicket("new"));
    expect(container.querySelector("audio")).toBe(media);
    expect(media.src).toContain("ticket=new");
    media.currentTime = 0;
    await act(async () => media.dispatchEvent(new Event("loadedmetadata")));
    expect(media.currentTime).toBe(12);
    expect(media.playbackRate).toBe(1.25);
    expect(play).toHaveBeenCalledOnce();
  } finally {
    await act(async () => root.unmount());
    container.remove();
    setMediaTicket("");
    play.mockRestore();
  }
});
