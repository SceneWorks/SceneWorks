import { describe, expect, it } from "vitest";
import { extractFamilies } from "./presetUtils.js";

describe("extractFamilies", () => {
  it("reads the supported shapes in precedence order", () => {
    expect(extractFamilies({ families: ["a"], compatibleFamilies: ["b"] })).toEqual(["a"]);
    expect(extractFamilies({ compatibleFamilies: ["b"], modelFamilies: ["c"] })).toEqual(["b"]);
    expect(extractFamilies({ modelFamilies: ["c"] })).toEqual(["c"]);
    expect(extractFamilies({ compatibility: { families: ["d"] } })).toEqual(["d"]);
    expect(extractFamilies({ family: "e" })).toEqual(["e"]);
  });

  it("returns an empty array when nothing is set", () => {
    expect(extractFamilies(undefined)).toEqual([]);
    expect(extractFamilies({})).toEqual([]);
  });

  it("returns raw values without normalizing casing or separators", () => {
    expect(extractFamilies({ families: ["Z_Image", "Qwen-Image"] })).toEqual(["Z_Image", "Qwen-Image"]);
  });

  it("ignores manifest metadata unless includeManifest is set", () => {
    const job = { payload: { manifestEntry: { families: ["z-image"] }, family: "qwen-image" } };
    expect(extractFamilies(job)).toEqual([]);
    expect(extractFamilies(job, { includeManifest: true })).toEqual(["z-image"]);
  });

  it("falls back through manifest fields then payload.family", () => {
    expect(extractFamilies({ payload: { family: "qwen-image" } }, { includeManifest: true })).toEqual(["qwen-image"]);
    expect(
      extractFamilies({ payload: { manifestEntry: { compatibility: { families: ["z-image"] } } } }, { includeManifest: true }),
    ).toEqual(["z-image"]);
  });

  it("prefers top-level fields over manifest even with includeManifest", () => {
    const job = { families: ["top"], payload: { manifestEntry: { families: ["manifest"] } } };
    expect(extractFamilies(job, { includeManifest: true })).toEqual(["top"]);
  });
});
