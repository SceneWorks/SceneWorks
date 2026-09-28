import { afterEach, describe, expect, it, vi } from "vitest";
import { timelineAudioPlan } from "./timelineAudio.js";
import { createTimelineAudioPlayback } from "./timelineAudioPlayback.js";

export function fakeAudioContext(events = []) {
  const contexts = [];
  const node = () => ({ connect: vi.fn(), disconnect: vi.fn() });
  class Context {
    constructor() {
      this.destination = {};
      this.currentTime = 0;
      this.gains = [];
      this.sources = [];
      this.createGain = vi.fn(() => { const gain = { ...node(), gain: { value: 1, cancelScheduledValues: vi.fn(), setValueAtTime: vi.fn() } }; this.gains.push(gain); return gain; });
      this.createMediaElementSource = vi.fn((element) => { const source = { ...node(), element }; this.sources.push(source); return source; });
      this.createDynamicsCompressor = vi.fn(() => ({ ...node(), ...Object.fromEntries(["threshold", "knee", "ratio", "attack", "release"].map((key) => [key, { value: 0 }])) }));
      this.resume = vi.fn(async () => { events.push("resume"); });
      this.close = vi.fn(async () => {});
      contexts.push(this);
    }
  }
  vi.stubGlobal("AudioContext", Context);
  return contexts;
}
afterEach(() => vi.unstubAllGlobals());

describe("timeline audio transport", () => {
  function setup() {
    const events = [];
    const contexts = fakeAudioContext(events);
    const onError = vi.fn();
    const playback = createTimelineAudioPlayback(onError);
    const plan = timelineAudioPlan({ tracks: [{ id: "audio", kind: "audio", gain: 4, items: [0, 1].map((start, i) => ({ id: String(i), assetId: String(i), timelineStart: start, timelineEnd: 3, sourceIn: 0.5, sourceOut: 3.5, volume: 2, speed: 1 })) }] });
    const elements = plan.placements.map((placement) => {
      const element = { currentTime: 0, paused: true, ended: false, playbackRate: 1,
        play: vi.fn(function () { this.paused = false; events.push("play"); return Promise.resolve(); }),
        pause: vi.fn(function () { this.paused = true; }) };
      playback.attach(placement.key, element, placement);
      return element;
    });
    return { playback, onError, plan, contexts, elements, events };
  }

  it("unlocks all layers on the first gesture, schedules overlap, seeks, stops exactly and replays without rebinding", async () => {
    const { playback, plan, contexts, elements, events } = setup();
    expect(contexts).toHaveLength(0);
    expect(playback.start(0)).toBe(true);
    expect(events).toEqual(["resume", "play", "play"]);
    expect(contexts[0].gains.map((g) => g.gain.value)).toEqual([8, 0]);
    expect(contexts[0].gains[0].gain.setValueAtTime).toHaveBeenLastCalledWith(0, 3);
    await Promise.resolve();
    expect(elements.map((e) => e.paused)).toEqual([false, true]);
    playback.update(plan.placements, 1.5, true, true);
    expect(elements.map((e) => e.currentTime)).toEqual([2, 1]);
    expect(elements.map((e) => e.paused)).toEqual([false, false]);
    expect(contexts[0].gains.map((g) => g.gain.value)).toEqual([8, 8]);
    expect(contexts[0].gains[1].gain.setValueAtTime).toHaveBeenLastCalledWith(0, 1.5);
    playback.update(plan.placements, 3, true);
    expect(elements.map((e) => e.paused)).toEqual([true, true]);
    expect(elements.map((e) => e.currentTime)).toEqual([3.5, 2.5]);
    await Promise.resolve();
    playback.pause();
    playback.start(0);
    expect(elements[0].currentTime).toBe(0.5);
    expect(contexts).toHaveLength(1);
    expect(contexts[0].createMediaElementSource).toHaveBeenCalledTimes(2);
    playback.dispose();
    expect(contexts[0].close).toHaveBeenCalledTimes(1);
    expect(contexts[0].sources.every((node) => node.disconnect.mock.calls.length === 1)).toBe(true);
  });

  it("silences removed layers and pause/foreground loss; playback failures stop the whole mix", async () => {
    const { playback, plan, contexts, elements, onError } = setup();
    playback.start(1.5);
    await Promise.resolve();
    playback.attach(plan.placements[0].key, null);
    expect(elements[0].paused).toBe(true);
    expect(contexts[0].gains[0].gain.value).toBe(0);
    playback.update(plan.placements, 2, false);
    expect(elements[1].paused).toBe(true);
    expect(contexts[0].gains[1].gain.value).toBe(0);
    elements[1].play.mockRejectedValueOnce(new Error("blocked"));
    playback.start(1.5);
    await Promise.resolve();
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: "blocked" }));
    expect(elements[1].paused).toBe(true);
    playback.dispose();
  });

  it("reports unsupported mixing instead of silently clamping boosted gain", () => {
    const { playback, onError, elements } = setup();
    vi.stubGlobal("AudioContext", undefined);
    expect(playback.start(0)).toBe(false);
    expect(onError).toHaveBeenCalled();
    expect(elements.every((e) => e.paused && e.muted)).toBe(true);
  });
});
