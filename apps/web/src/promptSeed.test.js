import { describe, expect, it } from "vitest";
import {
  DEFAULT_SCENE_PROMPT,
  promptHintFor,
  promptSeedFor,
  seedsNegativeInMode,
} from "./promptSeed.js";

describe("promptSeedFor (sc-10760)", () => {
  it("returns a booru model's defaultPrompt quality prefix", () => {
    expect(promptSeedFor({ defaultPrompt: "masterpiece, best quality, " })).toBe(
      "masterpiece, best quality, ",
    );
  });

  it("falls back to the generic scene default when no defaultPrompt is declared", () => {
    expect(promptSeedFor({ defaultNegativePrompt: "worst quality" })).toBe(DEFAULT_SCENE_PROMPT);
    expect(promptSeedFor({})).toBe(DEFAULT_SCENE_PROMPT);
    expect(promptSeedFor(undefined)).toBe(DEFAULT_SCENE_PROMPT);
    expect(promptSeedFor(null)).toBe(DEFAULT_SCENE_PROMPT);
  });

  it("treats an empty-string defaultPrompt as absent (no stale prefix)", () => {
    expect(promptSeedFor({ defaultPrompt: "" })).toBe(DEFAULT_SCENE_PROMPT);
  });

  it("ignores a non-string defaultPrompt", () => {
    expect(promptSeedFor({ defaultPrompt: 42 })).toBe(DEFAULT_SCENE_PROMPT);
  });
});

describe("seedsNegativeInMode (sc-3857, sc-10760)", () => {
  it("seeds the booru negative in character AND text-to-image", () => {
    expect(seedsNegativeInMode("character_image")).toBe(true);
    expect(seedsNegativeInMode("text_to_image")).toBe(true);
  });

  it("does not seed in edit mode or unknown modes", () => {
    expect(seedsNegativeInMode("edit_image")).toBe(false);
    expect(seedsNegativeInMode("batch")).toBe(false);
    expect(seedsNegativeInMode(undefined)).toBe(false);
  });
});

describe("promptHintFor (sc-10760)", () => {
  it("returns the declared booru hint", () => {
    expect(promptHintFor({ promptHint: "Booru-tag model: keep the quality prefix" })).toBe(
      "Booru-tag model: keep the quality prefix",
    );
  });

  it("returns null when no hint is declared", () => {
    expect(promptHintFor({})).toBe(null);
    expect(promptHintFor(undefined)).toBe(null);
    expect(promptHintFor({ promptHint: "" })).toBe(null);
  });
});
