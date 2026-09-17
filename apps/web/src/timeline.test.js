import { describe, expect, it } from "vitest";

import { audioPreviewState } from "./timeline.js";

describe("audioPreviewState", () => {
  const item = {
    sourceIn: 0.25,
    timelineStart: 2,
    timelineEnd: 6,
    speed: 1,
    volume: 0.5,
    fadeInSeconds: 1,
    fadeOutSeconds: 2,
  };
  const track = { id: "ambience", gain: 0.8, muted: false };

  it("maps timeline placement and trim to source time and mix gain", () => {
    expect(audioPreviewState(item, track, 2.5)).toEqual({
      afterPlacement: false,
      beforePlacement: false,
      currentTime: 0.75,
      hasPositiveGain: true,
      muted: false,
      playbackRate: 1,
      volume: 0.2,
    });
  });

  it("honors track mute, solo, and placement boundaries", () => {
    expect(audioPreviewState(item, { ...track, muted: true }, 3).muted).toBe(true);
    expect(audioPreviewState(item, track, 3, { dialogue: true }).muted).toBe(true);
    expect(audioPreviewState(item, track, 3, { ambience: true }).muted).toBe(false);
    expect(audioPreviewState(item, track, 1.5).muted).toBe(true);
    expect(audioPreviewState(item, track, 6).muted).toBe(true);
  });

  it("distinguishes an intentional zero gain from a zero-volume fade boundary", () => {
    expect(audioPreviewState({ ...item, volume: 0 }, track, 3).hasPositiveGain).toBe(false);
    expect(audioPreviewState(item, { ...track, gain: 0 }, 3).hasPositiveGain).toBe(false);
    expect(audioPreviewState(item, track, item.timelineStart).hasPositiveGain).toBe(true);
    expect(audioPreviewState(item, track, item.timelineStart).volume).toBe(0);
  });
});
