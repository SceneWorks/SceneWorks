import { describe, expect, it } from "vitest";
import {
  audioExpectedTakes,
  audioJobMode,
  audioRunChips,
  audioRunGroups,
  audioRunModelName,
  audioTakeTitle,
  formatClock,
  formatRelativeTime,
} from "./audioTakes.js";

// Run grouping + header derivations for the Audio Studio redesign (epic 14361 / sc-14364).
// Everything here is derived from the job's OWN payload and the live asset catalog — these
// pin that a header can never claim a setting the run didn't carry.

const KOKORO = { id: "kokoro_82m", name: "Kokoro 82M", audio: { voices: [{ id: "af_heart" }] } };
const ACESTEP = { id: "acestep", name: "ACE-Step Turbo", audio: { editModes: ["extend"] } };
const MOSS = { id: "moss_sfx", name: "MOSS SoundEffect", audio: { languages: ["en"] } };

describe("formatClock", () => {
  it("renders m:ss and clamps junk to 0:00", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(9.7)).toBe("0:09");
    expect(formatClock(125)).toBe("2:05");
    expect(formatClock(Number.NaN)).toBe("0:00");
    expect(formatClock(-4)).toBe("0:00");
  });
});

describe("formatRelativeTime", () => {
  const now = Date.parse("2026-07-24T12:00:00Z");
  it("steps from 'just now' up to days", () => {
    expect(formatRelativeTime("2026-07-24T11:59:50Z", now)).toBe("just now");
    expect(formatRelativeTime("2026-07-24T11:58:00Z", now)).toBe("2 min ago");
    expect(formatRelativeTime("2026-07-24T09:00:00Z", now)).toBe("3 h ago");
    expect(formatRelativeTime("2026-07-22T12:00:00Z", now)).toBe("2 d ago");
  });

  it("is empty for a missing or unparseable timestamp", () => {
    expect(formatRelativeTime(null, now)).toBe("");
    expect(formatRelativeTime("not a date", now)).toBe("");
  });
});

describe("audioJobMode", () => {
  it("reads Voice Clone off the reference clip, ahead of every other signal", () => {
    const job = { payload: { referenceAudioAssetId: "asset-1", voice: "af_heart", bpm: 120 } };
    expect(audioJobMode(job, KOKORO)).toBe("voiceclone");
  });

  it("reads Music off the music sub-block or an edit band", () => {
    expect(audioJobMode({ payload: { bpm: 128 } })).toBe("music");
    expect(audioJobMode({ payload: { musicalKey: "C minor" } })).toBe("music");
    expect(audioJobMode({ payload: { editMode: "extend" } })).toBe("music");
  });

  it("reads Speech off a voice or a dialogue script", () => {
    expect(audioJobMode({ payload: { voice: "af_heart" } })).toBe("speech");
    expect(audioJobMode({ payload: { script: [{ text: "hi" }] } })).toBe("speech");
  });

  it("falls back to the model's advertised capabilities when the payload is bare", () => {
    // A bare prompt-only run: the model's own capability block breaks the tie.
    expect(audioJobMode({ payload: { prompt: "a door slam" } }, KOKORO)).toBe("speech");
    expect(audioJobMode({ payload: { prompt: "a piano loop" } }, ACESTEP)).toBe("music");
    expect(audioJobMode({ payload: { prompt: "a door slam" } }, MOSS)).toBe("sfx");
    // No model at all ⇒ the residual text→audio generator.
    expect(audioJobMode({ payload: { prompt: "a door slam" } }, null)).toBe("sfx");
  });
});

describe("audioRunChips", () => {
  it("lists only the settings the run actually submitted", () => {
    expect(
      audioRunChips({
        payload: { voice: "af_heart", language: "en-US", targetDurationSecs: 12, seed: 4821 },
      }),
    ).toEqual(["af_heart", "en-US", "12 s", "seed 4821"]);
  });

  it("carries the music sub-block and omits cleared knobs", () => {
    expect(audioRunChips({ payload: { bpm: 128, musicalKey: "C minor", seed: null } })).toEqual([
      "128 BPM",
      "C minor",
    ]);
  });

  it("keeps seed 0 — a real, reproducible seed, not an empty value", () => {
    expect(audioRunChips({ payload: { seed: 0 } })).toEqual(["seed 0"]);
  });

  it("is empty for a run with no recorded payload", () => {
    expect(audioRunChips({})).toEqual([]);
  });
});

describe("audioRunModelName", () => {
  it("prefers the catalog label and falls back to the recorded id", () => {
    expect(audioRunModelName({ payload: { model: "kokoro_82m" } }, [KOKORO])).toBe("Kokoro 82M");
    // The model was uninstalled since the run — the header still names what produced it.
    expect(audioRunModelName({ payload: { model: "kokoro_82m" } }, [])).toBe("kokoro_82m");
  });
});

describe("audioRunGroups", () => {
  const assets = [
    { id: "a1", type: "audio", displayName: "Take 1" },
    { id: "a2", type: "audio", displayName: "Take 2" },
  ];
  const jobs = [
    {
      id: "job-running",
      status: "running",
      createdAt: "2026-07-24T12:00:00Z",
      payload: { model: "acestep", bpm: 128 },
      result: { expectedCount: 3 },
    },
    {
      id: "job-done",
      status: "completed",
      createdAt: "2026-07-24T11:00:00Z",
      payload: { model: "kokoro_82m", voice: "af_heart" },
      result: { assetIds: ["a1", "a2"] },
    },
  ];

  it("resolves one group per run, in lane order, with its takes and header facts", () => {
    const groups = audioRunGroups(jobs, assets, [KOKORO, ACESTEP]);
    expect(groups.map((group) => group.id)).toEqual(["job-running", "job-done"]);

    expect(groups[0].running).toBe(true);
    expect(groups[0].mode).toBe("music");
    expect(groups[0].modeLabel).toBe("Music");
    expect(groups[0].modelName).toBe("ACE-Step Turbo");
    expect(groups[0].takes).toEqual([]);

    expect(groups[1].running).toBe(false);
    expect(groups[1].mode).toBe("speech");
    expect(groups[1].takes.map((asset) => asset.id)).toEqual(["a1", "a2"]);
    expect(groups[1].chips).toEqual(["af_heart"]);
    expect(groups[1].createdAt).toBe("2026-07-24T11:00:00Z");
  });

  it("is empty for an empty lane", () => {
    expect(audioRunGroups([], assets, [])).toEqual([]);
  });
});

describe("audioExpectedTakes", () => {
  it("reports only a declared, positive count", () => {
    expect(audioExpectedTakes({ result: { expectedCount: 3 } })).toBe(3);
    expect(audioExpectedTakes({ result: { expectedCount: 0 } })).toBeNull();
    expect(audioExpectedTakes({})).toBeNull();
  });
});

describe("audioTakeTitle", () => {
  it("prefers the run's prompt, falling back to the asset's display name", () => {
    expect(audioTakeTitle({ payload: { prompt: "  rain at the door  " } }, { displayName: "x" })).toBe(
      "rain at the door",
    );
    expect(audioTakeTitle({ payload: { prompt: "   " } }, { displayName: "Imported clip" })).toBe(
      "Imported clip",
    );
    expect(audioTakeTitle(null, {})).toBe("Untitled clip");
  });
});
