import { describe, expect, it } from "vitest";
import { isAiAudioTrack } from "./editorUtils.js";

describe("isAiAudioTrack", () => {
  it("does not infer AI provenance from a track name or array position", () => {
    expect(isAiAudioTrack({ id: "second", name: "Music", items: [] }, new Map())).toBe(false);
  });

  it("accepts explicit track provenance", () => {
    expect(
      isAiAudioTrack({ id: "generated", provenance: { source: "audio_studio" }, items: [] }, new Map()),
    ).toBe(true);
  });

  it("derives provenance from generated backing assets", () => {
    const assets = new Map([
      ["generated-audio", { id: "generated-audio", origin: "audio_studio", recipe: { model: "musicgen" } }],
    ]);
    expect(
      isAiAudioTrack({ id: "audio", items: [{ id: "clip", assetId: "generated-audio" }] }, assets),
    ).toBe(true);
  });
});
