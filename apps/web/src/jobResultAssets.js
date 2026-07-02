// Single source of truth for "resolve a job's produced asset records against the
// live catalog" (sc-8853). Four near-identical copies had drifted across
// ImageStudio (image, batch-slot ordered), VideoStudio (video), QueueScreen
// (type-agnostic small-row) and characterPanels (image, catalog-only). Unifying
// here keeps a worker result-contract change — batch-slot ordering, the
// assetIds/assets/generationSetId fallbacks — a one-place edit.
//
// The resolution strategy: a job reports its outputs three ways, tried in order.
//   1. `result.assetIds` — the worker emits these in batch-slot order, so the
//      array order IS the review-slot order (preserve it verbatim).
//   2. `result.assets` — the embedded asset records (used before the catalog has
//      caught up); merged against the catalog so a saved record wins once known.
//   3. `result.generationSetId` — every catalog asset tagged with the set. This
//      is the streaming path (partial outputs surface as each image is saved) and
//      is the only branch that needs an explicit batch-index sort, since the
//      catalog is not inherently slot-ordered.

// Best-effort batch-slot index for an asset, used to order the generationSetId
// fallback back into worker-emitted slot order. Prefers explicit `batchIndex`
// fields, then a `_NNNN` filename suffix, then a trailing `#N` in the display
// name; unknown assets sort last (Infinity).
export function assetBatchIndex(asset) {
  const candidates = [
    asset?.batchIndex,
    asset?.recipe?.batchIndex,
    asset?.recipe?.normalizedSettings?.batchIndex,
    asset?.lineage?.batchIndex,
  ];
  for (const candidate of candidates) {
    const value = Number(candidate);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  const basename = String(asset?.file?.path ?? "").split(/[\\/]/).pop() ?? "";
  const fileMatch = basename.match(/_(\d{4})\.[^.]+$/);
  if (fileMatch) {
    return Number(fileMatch[1]) - 1;
  }
  const nameMatch = String(asset?.displayName ?? "").match(/#(\d+)\s*$/);
  return nameMatch ? Number(nameMatch[1]) - 1 : Number.POSITIVE_INFINITY;
}

// Resolve a job's produced assets against the live `assets` catalog.
//
// Options (each call site opts into the behavior it historically had):
//   - `type`             — only keep assets of this media type ("image" |
//                          "video"). Omit for the type-agnostic queue small-row,
//                          which shows whatever a job produced.
//   - `sortByBatchIndex` — sort the `generationSetId` fallback into batch-slot
//                          order. Only the image lane needs this; the assetIds
//                          branch is already slot-ordered by the worker, so this
//                          flag never applies there.
//   - `mergeResultAssets`— consult `result.assets` and let an embedded record
//                          stand in for a not-yet-catalogued asset. The image,
//                          video and queue lanes do this; characterPanels reads
//                          the catalog only (a partial-output stream where only
//                          catalog membership matters), so it passes `false`.
export function resolveJobResultAssets(job, assets, options = {}) {
  const { type, sortByBatchIndex = false, mergeResultAssets = true } = options;
  const list = Array.isArray(assets) ? assets : [];
  if (!job?.result) return [];

  const catalogById = new Map(list.map((asset) => [asset.id, asset]));
  const ofType = (asset) => (type ? asset?.type === type : Boolean(asset));

  if (mergeResultAssets) {
    // An embedded record wins only until the catalog knows the id; a catalogued
    // asset (fuller record) then takes over.
    const resultAssets = (Array.isArray(job.result.assets) ? job.result.assets : []).filter((asset) =>
      type ? asset?.type === type : true,
    );
    const resultById = new Map(resultAssets.map((asset) => [asset.id, catalogById.get(asset.id) ?? asset]));

    const assetIds = job.result.assetIds ?? [];
    if (assetIds.length) {
      // assetIds already carry batch-slot order — never re-sort this branch.
      return assetIds.map((id) => resultById.get(id) ?? catalogById.get(id)).filter(ofType);
    }
    if (resultAssets.length) {
      return resultAssets.map((asset) => catalogById.get(asset.id) ?? asset);
    }
  } else {
    // Catalog-only path (characterPanels): an id not yet in the catalog is
    // dropped, so a parallel job's newer set never masks this job's partials.
    const assetIds = job.result.assetIds ?? [];
    if (assetIds.length) {
      return assetIds.map((id) => catalogById.get(id)).filter(ofType);
    }
  }

  if (job.result.generationSetId) {
    const inSet = list.filter(
      (asset) => ofType(asset) && asset.generationSetId === job.result.generationSetId,
    );
    return sortByBatchIndex
      ? inSet.sort((left, right) => assetBatchIndex(left) - assetBatchIndex(right))
      : inSet;
  }
  return [];
}
