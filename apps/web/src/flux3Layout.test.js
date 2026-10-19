import { describe, expect, it } from "vitest";
import { reconcileLayoutReferences, moveLayoutReference, orderedLayoutAssetIds } from "./flux3Layout.js";

describe("layout attachment identity", () => {
  it("keeps duplicate attachments distinct across reorder, removal and replay", () => {
    const original = reconcileLayoutReferences([], ["a", "a", "b"]);
    expect(new Set(original.map(({ id }) => id)).size).toBe(3);
    const moved = moveLayoutReference(original, 1, 0);
    expect(reconcileLayoutReferences(moved, ["a", "a", "b"])).toEqual(moved);
    expect(reconcileLayoutReferences(moved, ["b", "a"])).toEqual([original[2], original[1]]);
    expect(reconcileLayoutReferences([], ["a", "a", "b"], original.map(({ id }) => id))).toEqual(original);
    expect(reconcileLayoutReferences([], ["a"])[0].id).not.toBe(original[0].id);
  });

  it("uses complete source, plural, singular S3 ordering without deduplicating", () => {
    expect(orderedLayoutAssetIds({ sourceAssetId: "a", referenceAssetIds: ["b", "a"], referenceAssetId: "b" }))
      .toEqual(["a", "b", "a", "b"]);
  });
});
