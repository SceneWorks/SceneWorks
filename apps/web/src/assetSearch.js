// Free-text Library search across an asset's display name, prompt, and tags.
// Case-insensitive substring match; an empty needle matches everything so callers
// can pass the query unconditionally. `needle` is expected pre-lowercased/trimmed.
export function assetMatchesSearch(asset, needle) {
  if (!needle) {
    return true;
  }
  const displayName = asset.displayName ?? "";
  const prompt = asset.recipe?.prompt ?? "";
  const tags = Array.isArray(asset.tags) ? asset.tags.join(" ") : "";
  return `${displayName} ${prompt} ${tags}`.toLowerCase().includes(needle);
}

