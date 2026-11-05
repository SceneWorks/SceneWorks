

// Upper bound on images in a FLUX.2 multi-reference edit, matching the worker's MAX_EDIT_REFERENCES
// (image_jobs/flux2.rs). The working image takes one slot, so the editor allows up to 3 user refs.
export const MAX_EDIT_REFERENCES = 4;

// Compose the multi-reference edit's `referenceAssetIds`: the working image (staged as the scratch
// source) FIRST so it anchors the edit, then the user's reference images — trimmed, de-duped, and
// capped at `max` total. The worker prefers a non-empty `referenceAssetIds` list over `sourceAssetId`
// (image_jobs/flux2.rs `flux2_edit_reference_ids`), so the working scratch id must lead the list or
// the working image is dropped from the joint conditioning. Pure. Empty source → empty list.
export function editReferenceIds(sourceAssetId, refIds, max = MAX_EDIT_REFERENCES) {
  const ids = [sourceAssetId, ...(refIds ?? [])]
    .map((id) => (typeof id === "string" ? id.trim() : ""))
    .filter(Boolean);
  return Array.from(new Set(ids)).slice(0, max);
}

// Import every selected reference independently so one bad file does not discard
// the successful imports. importAsset normally reports request errors itself; the
// dialog still needs an exact failure count instead of silently filtering them.
export async function importEditorReferenceFiles(files, importAsset) {
  const results = await Promise.allSettled(
    Array.from(files ?? []).map((file) => importAsset(file, { throwOnError: true })),
  );
  const assets = [];
  let failureCount = 0;
  for (const result of results) {
    if (result.status === "fulfilled" && result.value?.id) {
      assets.push(result.value);
    } else {
      failureCount += 1;
    }
  }
  return { assets, failureCount };
}

// Whether a model accepts an inpaint mask — the manifest tags it `image_inpaint`
// (sc-2476). Gates the mask tool in the editor. Pure.
export function modelIsInpaintCapable(model) {
  return (model?.capabilities ?? []).includes("image_inpaint");
}
