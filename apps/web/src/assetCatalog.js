// The Library catalog is loaded as SUMMARY items (`GET .../assets?view=summary`): every field a
// grid, picker, badge, search or membership check reads, minus the replay bulk (the embedded
// generation set's recipe, LoRA lists, model provenance, most adapter settings). A summary
// item is marked `listView: "summary"`; anything that replays, regenerates or inspects an asset
// resolves the full envelope first (`ensureFullAsset`), and the opened/selected asset is
// upgraded in place (`useFullAssetHydration`).
//
// After the first read the catalog is refreshed incrementally: the API returns only the assets
// changed since the previous cursor plus the ids to drop, and `mergeAssetChanges` folds them
// into the catalog the app already holds.

export const SUMMARY_LIST_VIEW = "summary";

export function isSummaryAsset(asset) {
  return asset?.listView === SUMMARY_LIST_VIEW;
}

// One shape for both answers the listing gives: the incremental envelope, or a bare array (an
// API without incremental support), which is a full replacement with no cursor to continue from.
export function normalizeAssetChanges(payload) {
  if (Array.isArray(payload)) {
    return { reset: true, cursor: null, assets: payload, removedIds: [] };
  }
  return {
    reset: payload?.reset !== false,
    cursor: typeof payload?.cursor === "string" && payload.cursor ? payload.cursor : null,
    assets: Array.isArray(payload?.assets) ? payload.assets : [],
    removedIds: Array.isArray(payload?.removedIds) ? payload.removedIds : [],
  };
}

// Newest first, matching the API's `order by created_at desc` (byte order of ISO strings).
function newerFirst(left, right) {
  const a = String(left?.createdAt ?? "");
  const b = String(right?.createdAt ?? "");
  return a > b ? -1 : a < b ? 1 : 0;
}

// Fold incremental changes into a newest-first catalog: removed ids leave, changed assets are
// replaced where they stand, and new assets are merged in by creation time (ahead of existing
// assets with the same timestamp). Untouched entries keep their identity, so an asset already
// upgraded to its full envelope stays upgraded until it next changes.
export function mergeAssetChanges(current, { assets = [], removedIds = [] } = {}) {
  if (!assets.length && !removedIds.length) {
    return current;
  }
  const removed = new Set(removedIds);
  const changed = new Map();
  for (const asset of assets) {
    if (asset?.id) changed.set(asset.id, asset);
  }
  const kept = [];
  for (const asset of current) {
    if (changed.has(asset.id)) {
      kept.push(changed.get(asset.id));
      changed.delete(asset.id);
    } else if (!removed.has(asset.id)) {
      kept.push(asset);
    }
  }
  if (!changed.size) {
    return kept;
  }
  const added = [...changed.values()].sort(newerFirst);
  const merged = [];
  let index = 0;
  for (const asset of added) {
    while (index < kept.length && newerFirst(kept[index], asset) < 0) {
      merged.push(kept[index]);
      index += 1;
    }
    merged.push(asset);
  }
  while (index < kept.length) {
    merged.push(kept[index]);
    index += 1;
  }
  return merged;
}

// The full envelope for an asset that may be a catalog summary. Full assets pass through.
export async function ensureFullAsset(asset, loadAsset) {
  if (!isSummaryAsset(asset)) {
    return asset;
  }
  const full = await loadAsset(asset.projectId, asset.id);
  if (!full?.id) {
    throw new Error("This asset is no longer available.");
  }
  return full;
}
