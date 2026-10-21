import { describe, expect, it } from "vitest";

import { collectCaptionSidecars, isJsonCaptionSidecar, parseJsonCaptionSidecar } from "./captionSidecars.js";

describe("parseJsonCaptionSidecar", () => {
  it("turns string fields into named captions and lifts caption as the caption text", () => {
    expect(parseJsonCaptionSidecar('{"caption":" a red fox ","short":"fox","tags":"animal, red"}')).toEqual({
      ok: true,
      captions: { short: "fox", tags: "animal, red" },
      caption: "a red fox",
    });
  });

  it("falls back to text when there is no caption key", () => {
    expect(parseJsonCaptionSidecar('{"text":"a cat"}')).toEqual({ ok: true, captions: { text: "a cat" }, caption: "a cat" });
  });

  it.each([
    ["not JSON", "a cat"],
    ["an array", '["a"]'],
    ["null", "null"],
    ["a string", '"a cat"'],
    ["a number value", '{"short":"a","n":3}'],
    ["a nested value", '{"short":{"x":"y"}}'],
    ["an empty object", "{}"],
    ["a padded field name", '{" short":"a"}'],
    ["a too-long field name", `{"${"x".repeat(65)}":"a"}`],
  ])("rejects %s", (_name, text) => {
    expect(parseJsonCaptionSidecar(text).ok).toBe(false);
  });

  it("recognizes .json by extension or type", () => {
    expect(isJsonCaptionSidecar({ name: "Mira_01.JSON" })).toBe(true);
    expect(isJsonCaptionSidecar({ name: "x", type: "application/json" })).toBe(true);
    expect(isJsonCaptionSidecar({ name: "Mira_01.txt", type: "text/plain" })).toBe(false);
  });
});

describe("collectCaptionSidecars", () => {
  it("pairs by stem; a .txt caption wins over the JSON caption, named fields still apply", () => {
    const { byStem, ignored } = collectCaptionSidecars([
      { name: "Mira_01.txt", text: "from txt" },
      { name: "Mira_01.json", text: '{"caption":"from json","short":"mira"}' },
      { name: "Mira_02.json", text: '{"caption":"json only","tags":"t"}' },
      { name: "bad.json", text: "[1,2]" },
      { name: "Mira_03.txt", text: "   " },
    ]);
    expect(byStem.get("mira_01")).toEqual({ text: "from txt", captions: { short: "mira" } });
    expect(byStem.get("mira_02")).toEqual({ text: "json only", captions: { tags: "t" } });
    expect(byStem.has("mira_03")).toBe(false);
    expect(byStem.has("bad")).toBe(false);
    expect(ignored).toEqual(["bad.json"]);
  });
});
