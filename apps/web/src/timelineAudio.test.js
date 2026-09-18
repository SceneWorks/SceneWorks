import { describe, expect, it } from "vitest";
import { timelineAudioPlan, timelineAudioState } from "./timelineAudio.js";

const item = (id, overrides = {}) => ({ id, assetId: id, timelineStart: 0, timelineEnd: 4, sourceIn: 0, sourceOut: 4, speed: 1, ...overrides });
const cut = () => ({ tracks: [
  { id: "track_main", kind: "video", items: [item("picture", { generatedAudio: "include" })] },
  { id: "dialogue", kind: "audio", items: [item("voice", { timelineStart: 0.5, timelineEnd: 3.5, sourceOut: 3 })] },
  { id: "music", kind: "audio", gain: 0.25, items: [item("bed")] },
] });

describe("assembled audio export contract", () => {
  it("includes all overlapping layers independent of selection and ignores overlays and mute-default picture", () => {
    const timeline = cut();
    timeline.tracks.push({ id: "overlay", kind: "overlay", items: [item("overlay", { generatedAudio: "include" })] });
    const plan = timelineAudioPlan(timeline);
    expect(plan.duration).toBe(4);
    expect(plan.placements.map((p) => p.assetId).sort()).toEqual(["bed", "picture", "voice"]);
    expect(plan.placements.filter((p) => timelineAudioState(p, 1).active)).toHaveLength(3);
    delete timeline.tracks[0].items[0].generatedAudio;
    timeline.tracks[2].muted = true;
    expect(timelineAudioPlan(timeline).placements.map((p) => p.assetId)).toEqual(["voice"]);
  });

  it("applies trim, rate, full eightfold gain and multiplicative overlapping fades, then cuts at source out", () => {
    const timeline = cut();
    timeline.tracks[1].gain = 4;
    timeline.tracks[1].items = [item("voice", { timelineStart: 1, timelineEnd: 4, sourceIn: 2, sourceOut: 5, speed: 2, volume: 2, fadeInSeconds: 2, fadeOutSeconds: 2 })];
    const placement = timelineAudioPlan(timeline).placements.find((p) => p.assetId === "voice");
    expect(timelineAudioState(placement, 0)).toEqual({ active: false, currentTime: 2, gain: 0 });
    expect(timelineAudioState(placement, 2)).toEqual({ active: true, currentTime: 4, gain: 4 });
    expect(timelineAudioState(placement, 2.25).gain).toBeCloseTo(8 * 1.25 / 2 * 1.75 / 2);
    expect(timelineAudioState(placement, 2.5)).toEqual({ active: false, currentTime: 5, gain: 0 });
    expect(timelineAudioState(placement, 4).active).toBe(false);
  });

  it("retains local solo audition without changing the saved export mix", () => {
    const timeline = cut();
    expect(timelineAudioPlan(timeline, { dialogue: true }).placements.map((p) => p.assetId)).toEqual(["voice"]);
    expect(timelineAudioPlan(timeline).placements).toHaveLength(3);
    timeline.tracks[1].muted = true;
    expect(timelineAudioPlan(timeline, { dialogue: true }).placements).toHaveLength(0);
  });

  it("moves audio with exported crossfades and caps beds at picture duration", () => {
    const timeline = cut();
    timeline.tracks[0].items = [item("a", { timelineEnd: 2 }), item("b", { timelineStart: 2, transitionIn: { type: "crossfade", duration: 1 }, generatedAudio: "include" })];
    timeline.tracks[1].items = [item("voice", { timelineStart: 2, timelineEnd: 6 })];
    const plan = timelineAudioPlan(timeline);
    expect(plan.duration).toBe(3);
    expect(plan.toPictureTime(2)).toBe(1);
    expect(plan.toTimelineTime(1)).toBe(2);
    expect(plan.placements.find((p) => p.assetId === "voice")).toMatchObject({ start: 1, span: 2 });
    expect(plan.placements.find((p) => p.assetId === "bed").span).toBe(3);
  });

  it("retains leading gaps and excludes audio beyond the last picture even on longer audio tracks", () => {
    const timeline = cut();
    timeline.tracks[0].items[0] = item("picture", { timelineStart: 1, timelineEnd: 3 });
    timeline.tracks[1].items = [item("late", { timelineStart: 4, timelineEnd: 8 })];
    const plan = timelineAudioPlan(timeline);
    expect(plan.duration).toBe(3);
    expect(plan.placements.map((p) => p.assetId)).toEqual(["bed"]);
    expect(plan.placements[0].span).toBe(3);
  });
});
