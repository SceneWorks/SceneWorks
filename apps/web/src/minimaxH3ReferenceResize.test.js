import { describe, expect, it } from "vitest";

import { isMinimaxH3Entry, recipeReferenceResize } from "./minimaxH3ReferenceResize.js";

describe("MiniMax-H3 reference resize replay (sc-26691)", () => {
  const h3 = (raw) => ({ model: "minimax_h3_ref", rawAdapterSettings: { minimaxH3Tier: "q8", ...raw } });

  it("carries the recorded mode and short edge", () => {
    expect(recipeReferenceResize(h3({ referenceImageResizeMode: "match" }))).toEqual({
      referenceImageResizeMode: "match",
    });
    expect(
      recipeReferenceResize(h3({ referenceImageResizeMode: "max", referenceImageShortEdge: 1024 })),
    ).toEqual({ referenceImageResizeMode: "max", referenceImageShortEdge: 1024 });
  });

  it("reads a recorded H3 asset without the key as the legacy diffusers render", () => {
    expect(recipeReferenceResize(h3({}))).toEqual({ referenceImageResizeMode: "diffusers" });
    // An imported H3 checkpoint's asset is recognised by the H3 arm's own tier record.
    expect(
      recipeReferenceResize({ model: "singularity_fl2va", rawAdapterSettings: { minimaxH3Tier: "bf16" } }),
    ).toEqual({ referenceImageResizeMode: "diffusers" });
  });

  it("leaves non-H3, unrecorded and malformed recipes to the engine defaults", () => {
    expect(recipeReferenceResize({ model: "wan_i2v", rawAdapterSettings: {} })).toBeNull();
    expect(recipeReferenceResize({ model: "minimax_h3", rawAdapterSettings: {} })).toBeNull();
    expect(recipeReferenceResize(h3({ referenceImageResizeMode: "crop" }))).toBeNull();
    expect(recipeReferenceResize(null)).toBeNull();
  });

  it("recognises stock and imported H3 entries", () => {
    expect(isMinimaxH3Entry({ id: "minimax_h3" })).toBe(true);
    expect(isMinimaxH3Entry({ id: "community_h3", family: "minimax-h3" })).toBe(true);
    expect(isMinimaxH3Entry({ id: "bernini", family: "bernini" })).toBe(false);
  });
});
