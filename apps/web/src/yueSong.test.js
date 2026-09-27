import { describe, expect, it } from "vitest";
import YUE_TOP_TAGS from "./data/yue-top-200-tags.json";
import {
  YUE_TAG_CATEGORIES,
  addGenreTags,
  genreTagsPrompt,
  isIclSongModel,
  isSegmentedSongModel,
  parseGenreTags,
  splitGenreTagLine,
  yueLyricsForSubmit,
  yueTagSuggestions,
} from "./yueSong.js";

describe("yueSong (sc-19385)", () => {
  it("packs sections as upstream `[label]\\n…` blocks, dropping empty ones", () => {
    expect(
      yueLyricsForSubmit([
        { label: "verse", text: " line one\nline two " },
        { label: "chorus", text: "   " },
        { label: "bridge", text: "b" },
      ]),
    ).toBe("[verse]\nline one\nline two\n\n[bridge]\nb");
    expect(yueLyricsForSubmit([])).toBe("");
  });

  it("splits tags on commas/newlines only and dedupes case-insensitively", () => {
    expect(parseGenreTags("pop,  bright   vocal\nfemale,,")).toEqual(["pop", "bright vocal", "female"]);
    expect(addGenreTags(["Pop"], ["pop", "rock"])).toEqual(["Pop", "rock"]);
    expect(genreTagsPrompt(["uplifting", "bright vocal"])).toBe("uplifting bright vocal");
  });

  it("splits a refined tag line: commas win, else greedy multi-word upstream tags, else words", () => {
    expect(splitGenreTagLine("uplifting female airy vocal")).toEqual(["uplifting", "female", "airy vocal"]);
    expect(splitGenreTagLine("inspiring female uplifting pop airy vocal electronic bright vocal vocal")).toEqual([
      "inspiring",
      "female",
      "uplifting",
      "pop",
      "airy vocal",
      "electronic",
      "bright vocal",
      "vocal",
    ]);
    expect(splitGenreTagLine("dream pop, airy vocal")).toEqual(["dream pop", "airy vocal"]);
  });

  it("offers upstream's five categories, case-folded, in upstream order", () => {
    expect(Object.keys(YUE_TOP_TAGS)).toEqual(YUE_TAG_CATEGORIES.map((category) => category.id));
    const genre = yueTagSuggestions("genre");
    expect(genre.slice(0, 3)).toEqual(["Pop", "rock", "electronic"]);
    expect(new Set(genre.map((tag) => tag.toLowerCase())).size).toBe(genre.length);
  });

  it("gates on the capability flags, never an id", () => {
    const cot = { id: "x", audio: { supportsSegmentedLyrics: true } };
    const icl = { id: "y", audio: { supportsSegmentedLyrics: true, conditioning: ["ReferenceAudio"] } };
    const clone = { id: "yue_en_icl", audio: { conditioning: ["ReferenceAudio"] } };
    expect([cot, icl, clone].map(isSegmentedSongModel)).toEqual([true, true, false]);
    expect([cot, icl, clone].map(isIclSongModel)).toEqual([false, true, false]);
  });
});
