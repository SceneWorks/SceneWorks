import { describe, expect, it } from "vitest";
import { assetMatchesCharacter, characterAssetIds } from "./characterMembership.js";

const CHARACTER_ID = "char_1";

// A character whose approved-reference list and plain-reference list point at
// distinct assets, so tests can prove the predicate consults BOTH (superset).
const CHARACTER = {
  id: CHARACTER_ID,
  approvedReferences: [{ assetId: "asset_approved" }],
  references: [{ assetId: "asset_reference" }, { id: "asset_reference_alt_shape" }],
};

describe("assetMatchesCharacter", () => {
  it("matches an asset only in approvedReferences (superset semantics)", () => {
    expect(assetMatchesCharacter({ id: "asset_approved" }, CHARACTER_ID, CHARACTER)).toBe(true);
  });

  it("matches an asset only in references", () => {
    expect(assetMatchesCharacter({ id: "asset_reference" }, CHARACTER_ID, CHARACTER)).toBe(true);
  });

  it("honors both { assetId } and { id } reference shapes", () => {
    expect(assetMatchesCharacter({ id: "asset_reference_alt_shape" }, CHARACTER_ID, CHARACTER)).toBe(true);
  });

  it("matches via recipe.normalizedSettings.characterId without any reference list", () => {
    const asset = { id: "asset_recipe", recipe: { normalizedSettings: { characterId: CHARACTER_ID } } };
    expect(assetMatchesCharacter(asset, CHARACTER_ID, null)).toBe(true);
  });

  it("matches via metadata.characterReferences[].characterId without any reference list", () => {
    const asset = { id: "asset_meta", metadata: { characterReferences: [{ characterId: CHARACTER_ID }] } };
    expect(assetMatchesCharacter(asset, CHARACTER_ID, null)).toBe(true);
  });

  it("rejects a non-member asset", () => {
    expect(assetMatchesCharacter({ id: "asset_other" }, CHARACTER_ID, CHARACTER)).toBe(false);
  });

  it("rejects when characterId is missing", () => {
    expect(assetMatchesCharacter({ id: "asset_approved" }, "", CHARACTER)).toBe(false);
    expect(assetMatchesCharacter({ id: "asset_approved" }, undefined, CHARACTER)).toBe(false);
  });

  it("does not match a list entry when the asset has no id", () => {
    // A reference-less asset must not accidentally match via `undefined === undefined`.
    expect(assetMatchesCharacter({}, CHARACTER_ID, CHARACTER)).toBe(false);
  });

  it("does not match a different character's recipe/metadata", () => {
    const asset = {
      id: "asset_other_char",
      recipe: { normalizedSettings: { characterId: "char_2" } },
      metadata: { characterReferences: [{ characterId: "char_2" }] },
    };
    expect(assetMatchesCharacter(asset, CHARACTER_ID, CHARACTER)).toBe(false);
  });
});

describe("characterAssetIds", () => {
  it("unions approvedReferences and references and drops empty ids", () => {
    const ids = characterAssetIds({
      approvedReferences: [{ assetId: "a" }, { id: "b" }, {}],
      references: [{ assetId: "c" }],
    });
    expect(ids).toEqual(new Set(["a", "b", "c"]));
  });

  it("tolerates a null/undefined character", () => {
    expect(characterAssetIds(null)).toEqual(new Set());
    expect(characterAssetIds(undefined)).toEqual(new Set());
  });
});
