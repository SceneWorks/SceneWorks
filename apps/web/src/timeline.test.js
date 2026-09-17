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
      muted: false,
      playbackRate: 1,
      gain: 0.2,
    });
  });

  it("preserves the export-supported above-unity gain product", () => {
    expect(audioPreviewState(
      { ...item, volume: 2, fadeInSeconds: 0, fadeOutSeconds: 0 },
      { ...track, gain: 4 },
      3,
    ).gain).toBe(8);
    expect(audioPreviewState({ ...item, volume: 0 }, { ...track, gain: 4 }, 3).gain).toBe(0);
    expect(audioPreviewState(
      { ...item, volume: 99, fadeInSeconds: 0, fadeOutSeconds: 0 },
      { ...track, gain: 99 },
      3,
    ).gain).toBe(8);
  });

  it("applies fade-out and pins the source time at the source-out boundary", () => {
    expect(audioPreviewState(item, track, 5).gain).toBeCloseTo(0.2);
    expect(audioPreviewState(item, track, 6)).toMatchObject({
      afterPlacement: true,
      currentTime: 4.25,
      muted: true,
    });
  });

  it("honors track mute, solo, and placement boundaries", () => {
    expect(audioPreviewState(item, { ...track, muted: true }, 3).muted).toBe(true);
    expect(audioPreviewState(item, track, 3, { dialogue: true }).muted).toBe(true);
    expect(audioPreviewState(item, track, 3, { ambience: true }).muted).toBe(false);
    expect(audioPreviewState(item, track, 1.5).muted).toBe(true);
    expect(audioPreviewState(item, track, 6).muted).toBe(true);
  });
});
