// sc-17161 — the client half of the video `limits` seam.
//
// Every assertion that names a MiniMax-H3 number reads it out of the REAL manifest bytes rather
// than a retyped copy: a retyped fixture is the false green that let GH #2074 ship a mode nobody
// could submit, and the caps/steps floors here have exactly the same failure shape (a control that
// renders but cannot produce an accepted job).

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAX_REFERENCE_ASSETS,
  DEFAULT_MAX_REFERENCE_AUDIO_ASSETS,
  DEFAULT_MAX_SOURCE_CLIP_ASSETS,
  formatDurationOption,
  formatDurationSeconds,
  minStepsForModel,
  referenceCaps,
  referenceLimitError,
} from "./videoModelLimits.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");
const manifest = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
const byId = (id) => manifest.models.find((model) => model.id === id);
const minimaxH3 = byId("minimax_h3");
const minimaxH3Ref = byId("minimax_h3_ref");

describe("referenceCaps", () => {
  it("falls back to the pre-sc-17160 blankets for a model that declares nothing", () => {
    expect(referenceCaps({ limits: {} })).toEqual({
      images: DEFAULT_MAX_REFERENCE_ASSETS,
      clips: DEFAULT_MAX_SOURCE_CLIP_ASSETS,
      audio: DEFAULT_MAX_REFERENCE_AUDIO_ASSETS,
      combined: null,
      clipsDeclared: false,
    });
    // Audio is the one that must be ZERO: it is surface no engine before MiniMax-H3 consumed, so
    // the picker stays hidden until a model declares it rather than offering a silent no-op.
    expect(referenceCaps({}).audio).toBe(0);
    expect(referenceCaps(null).audio).toBe(0);
  });

  it("reads MiniMax-H3 Ref2VA's four caps from the shipped manifest", () => {
    const caps = referenceCaps(minimaxH3Ref);
    expect(caps).toEqual({
      images: minimaxH3Ref.limits.maxReferenceAssets,
      clips: minimaxH3Ref.limits.maxSourceClipAssets,
      audio: minimaxH3Ref.limits.maxReferenceAudioAssets,
      combined: minimaxH3Ref.limits.maxCombinedReferenceAssets,
      clipsDeclared: true,
    });
    // The shape that makes the combined cap irreducible: the per-list caps sum PAST it, so a
    // selection can satisfy all three pickers and still be refused.
    expect(caps.images + caps.clips + caps.audio).toBeGreaterThan(caps.combined);
  });

  it("gives the text-to-video partition zero of every reference kind", () => {
    // `minimax_h3` has no reference-conditioning path at all, so every list is 0 and no reference
    // picker can appear for it — an engine can never be handed conditioning it would ignore.
    expect(referenceCaps(minimaxH3)).toMatchObject({ images: 0, clips: 0, audio: 0 });
  });

  it("treats a non-integer or negative declaration as undeclared", () => {
    expect(referenceCaps({ limits: { maxReferenceAssets: "9" } }).images).toBe(
      DEFAULT_MAX_REFERENCE_ASSETS,
    );
    expect(referenceCaps({ limits: { maxSourceClipAssets: -1 } }).clipsDeclared).toBe(false);
    expect(referenceCaps({ limits: { maxReferenceAudioAssets: 2.5 } }).audio).toBe(0);
  });

  it("separates 'declares a clip cap' from 'resolves to a non-zero clip cap'", () => {
    // The blanket default is 8, so `clips > 0` is true for a model that never said it takes
    // reference clips — which is why the picker keys on `clipsDeclared`.
    const undeclared = referenceCaps({ limits: {} });
    expect(undeclared.clips).toBeGreaterThan(0);
    expect(undeclared.clipsDeclared).toBe(false);
  });
});

describe("referenceLimitError", () => {
  const caps = referenceCaps(minimaxH3Ref);
  const modelName = minimaxH3Ref.name;

  it("admits a selection inside every cap", () => {
    expect(referenceLimitError({ modelName, caps, images: 9, clips: 2, audio: 1 })).toBeNull();
    expect(referenceLimitError({ modelName, caps })).toBeNull();
  });

  it("names the list that is over its own cap", () => {
    const message = referenceLimitError({ modelName, caps, images: caps.images + 1 });
    expect(message).toContain(modelName);
    expect(message).toContain("reference images");
    expect(message).toContain(String(caps.images));
    expect(message).toContain("Remove 1.");
  });

  it("refuses a kind the model takes none of, naming the kind", () => {
    const t2v = referenceCaps(minimaxH3);
    const message = referenceLimitError({ modelName: minimaxH3.name, caps: t2v, audio: 1 });
    expect(message).toContain("takes no reference audio clips");
  });

  it("agrees in number: a single over-cap selection reads as singular", () => {
    // Audio caps at 3 (and t2v at 0), so one selected file is an ordinary path — the refusal must
    // not read "1 are selected. Remove them" (sc-17137 review B2).
    const t2v = referenceCaps(minimaxH3);
    const singular = referenceLimitError({ modelName: minimaxH3.name, caps: t2v, audio: 1 });
    expect(singular).toContain("but 1 is selected. Remove it,");
    expect(singular).not.toContain("are selected");
    const plural = referenceLimitError({ modelName: minimaxH3.name, caps: t2v, audio: 2 });
    expect(plural).toContain("but 2 are selected. Remove them,");
    const overCap = referenceLimitError({ modelName, caps, images: caps.images + 2 });
    expect(overCap).toContain(`but ${caps.images + 2} are selected. Remove 2.`);
  });

  it("refuses a combined over-selection in which every individual list fits", () => {
    // 9 + 3 + 3 = 15 files: each picker is exactly at its own cap, and only the combined ceiling
    // can refuse it. This is the assertion the per-list checks structurally cannot make.
    const message = referenceLimitError({
      modelName,
      caps,
      images: caps.images,
      clips: caps.clips,
      audio: caps.audio,
    });
    expect(message).not.toBeNull();
    expect(message).toContain(`up to ${caps.combined} reference files in total`);
    expect(message).toContain(`Remove ${caps.images + caps.clips + caps.audio - caps.combined}.`);
  });

  it("does not invent a combined ceiling for a model that declares none", () => {
    const open = referenceCaps({ limits: {} });
    expect(referenceLimitError({ modelName: "Bernini", caps: open, images: 8, clips: 8 })).toBeNull();
  });
});

describe("minStepsForModel", () => {
  it("reads the declared floor for both MiniMax-H3 partitions", () => {
    expect(minStepsForModel(minimaxH3)).toBe(minimaxH3.limits.hardMinSteps);
    expect(minStepsForModel(minimaxH3Ref)).toBe(minimaxH3Ref.limits.hardMinSteps);
    // Not the historical hardcoded 1 the Steps input used to allow.
    expect(minStepsForModel(minimaxH3)).toBeGreaterThan(1);
  });

  it("falls back to 1 when a model declares no floor", () => {
    expect(minStepsForModel({ limits: {} })).toBe(1);
    expect(minStepsForModel(null)).toBe(1);
    expect(minStepsForModel({ limits: { hardMinSteps: 0 } })).toBe(1);
  });
});

describe("duration labels", () => {
  it("renders the fourteen shipped MiniMax-H3 rungs as 2dp seconds plus their frame count", () => {
    const fps = minimaxH3.defaults.fps;
    const labels = minimaxH3.limits.durations.map((value) => formatDurationOption(value, fps));
    expect(labels).toHaveLength(14);
    expect(labels[0]).toBe("5.17s · 124 frames");
    expect(labels.at(-1)).toBe("14.38s · 345 frames");
    // Every rung's frame count must land on the 17n+5 lattice — this is what makes the menu a
    // lattice rather than a rounded slider, and it is derived from the shipped seconds.
    for (const [index, duration] of minimaxH3.limits.durations.entries()) {
      const frames = Math.round(duration * fps);
      expect(frames % 17, `rung ${index} (${duration}s) is off the 17n+5 lattice`).toBe(5);
    }
  });

  it("never labels a duration the model does not offer, including the advertised 15s", () => {
    // MiniMax's own docs advertise 15.0s; the next lattice rung is 362 frames = 15.083s, so 15s
    // is refused rather than delivered as 14.375s. It must therefore not appear in the menu.
    expect(minimaxH3.limits.durations).not.toContain(15);
    expect(minimaxH3.limits.hardMaxDuration).toBe(Math.max(...minimaxH3.limits.durations));
    const labels = minimaxH3.limits.durations.map((value) =>
      formatDurationOption(value, minimaxH3.defaults.fps),
    );
    expect(labels.some((label) => label.startsWith("15"))).toBe(false);
  });

  it("leaves integral durations alone and degrades without an fps", () => {
    expect(formatDurationSeconds(8)).toBe("8");
    expect(formatDurationSeconds(8.0)).toBe("8");
    expect(formatDurationOption(6, 25)).toBe("6s · 150 frames");
    expect(formatDurationOption(6, undefined)).toBe("6s");
    expect(formatDurationOption(6, 0)).toBe("6s");
    expect(formatDurationSeconds("nope")).toBe("");
  });
});
