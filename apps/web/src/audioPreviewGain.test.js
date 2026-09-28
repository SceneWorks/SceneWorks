import { describe, expect, it, vi } from "vitest";

import { acquireAudioPreviewGain } from "./audioPreviewGain.js";

function fakeAudioContext() {
  const contexts = [];
  class FakeAudioContext {
    constructor() {
      this.state = "suspended";
      this.destination = { kind: "destination" };
      this.source = { connect: vi.fn(), disconnect: vi.fn() };
      this.gainNode = { connect: vi.fn(), disconnect: vi.fn(), gain: { value: 1 } };
      this.close = vi.fn(async () => {});
      this.resume = vi.fn(async () => { this.state = "running"; });
      this.createMediaElementSource = vi.fn(() => this.source);
      this.createGain = vi.fn(() => this.gainNode);
      contexts.push(this);
    }
  }
  return { contexts, FakeAudioContext };
}

describe("audio preview gain graph", () => {
  it("applies the full gain, mute, and lifecycle without duplicate media sources", async () => {
    const { contexts, FakeAudioContext } = fakeAudioContext();
    const media = document.createElement("audio");
    const first = acquireAudioPreviewGain(media, FakeAudioContext);
    const second = acquireAudioPreviewGain(media, FakeAudioContext);

    expect(contexts).toHaveLength(1);
    expect(contexts[0].createMediaElementSource).toHaveBeenCalledTimes(1);
    expect(contexts[0].source.connect).toHaveBeenCalledWith(contexts[0].gainNode);
    expect(contexts[0].gainNode.connect).toHaveBeenCalledWith(contexts[0].destination);

    expect(first.setGain(8, false)).toBe(true);
    expect(media.volume).toBe(1);
    expect(media.muted).toBe(false);
    expect(contexts[0].gainNode.gain.value).toBe(8);

    first.setGain(0.2, false);
    expect(contexts[0].gainNode.gain.value).toBe(0.2);
    first.setGain(4, true);
    expect(contexts[0].gainNode.gain.value).toBe(0);
    await first.resume();
    expect(contexts[0].resume).toHaveBeenCalledTimes(1);

    first.release();
    await Promise.resolve();
    expect(contexts[0].close).not.toHaveBeenCalled();
    second.release();
    await Promise.resolve();
    expect(contexts[0].source.disconnect).toHaveBeenCalledTimes(1);
    expect(contexts[0].gainNode.disconnect).toHaveBeenCalledTimes(1);
    expect(contexts[0].close).toHaveBeenCalledTimes(1);

    const retired = acquireAudioPreviewGain(media, FakeAudioContext);
    expect(retired.supported).toBe(false);
    expect(contexts).toHaveLength(1);
  });

  it("reuses a graph when React immediately reacquires a released element", async () => {
    const { contexts, FakeAudioContext } = fakeAudioContext();
    const media = document.createElement("audio");
    const first = acquireAudioPreviewGain(media, FakeAudioContext);
    first.release();
    const second = acquireAudioPreviewGain(media, FakeAudioContext);
    await Promise.resolve();

    expect(contexts).toHaveLength(1);
    expect(contexts[0].createMediaElementSource).toHaveBeenCalledTimes(1);
    expect(contexts[0].close).not.toHaveBeenCalled();

    second.release();
    await Promise.resolve();
  });

  it("refuses a misleading boosted native fallback when Web Audio is unavailable", () => {
    const media = document.createElement("audio");
    const lease = acquireAudioPreviewGain(media, null);

    expect(lease.supported).toBe(false);
    expect(lease.setGain(0.4, false)).toBe(true);
    expect(media.volume).toBe(0.4);
    expect(media.muted).toBe(false);
    expect(lease.setGain(2, false)).toBe(false);
    expect(media.volume).toBe(1);
    expect(media.muted).toBe(true);
  });
});
