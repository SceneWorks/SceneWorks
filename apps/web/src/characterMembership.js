// Single source of truth for the "does this asset belong to this character?"
// predicate (sc-8857). Previously this logic was triplicated across the
// Image-Edit source picker (AssetPicker), the dataset add dialog
// (DatasetAddDialog), and the asset detail-panel linker (assetPanels) — and the
// copies had already drifted: only AssetPicker consulted `approvedReferences`,
// so an asset promoted to a character's *approved* reference list (a stronger
// membership signal than a plain reference) matched in the picker but silently
// failed to match in the other two surfaces.
//
// An asset belongs to a character when ANY of the following hold:
//   - it was generated for the character (recipe.normalizedSettings.characterId),
//   - it was generated referencing the character
//     (metadata.characterReferences[].characterId), or
//   - it is in the character's approved-reference OR reference list.
//
// The membership-list check is a superset: `approvedReferences` ∪ `references`.
// A reference entry may be shaped `{ assetId }` or `{ id }`, so both are honored.

// The set of asset ids the character explicitly references, unioning the
// approved-reference and plain-reference lists.
export function characterAssetIds(character) {
  return new Set(
    [
      ...(character?.approvedReferences ?? []),
      ...(character?.references ?? []),
    ]
      .map((reference) => reference?.assetId ?? reference?.id)
      .filter(Boolean),
  );
}

// True when `asset` belongs to the character identified by `characterId`.
// `character` (optional) supplies the reference lists for the membership-list
// check; without it, only the recipe/metadata signals are consulted.
export function assetMatchesCharacter(asset, characterId, character = null) {
  if (!characterId) {
    return false;
  }
  if (asset?.recipe?.normalizedSettings?.characterId === characterId) {
    return true;
  }
  if ((asset?.metadata?.characterReferences ?? []).some((reference) => reference?.characterId === characterId)) {
    return true;
  }
  return Boolean(asset?.id) && characterAssetIds(character).has(asset.id);
}
