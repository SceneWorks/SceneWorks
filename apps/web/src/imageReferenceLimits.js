// Per-model ordered-reference limits for the image lanes (sc-24113, epic 24107).
//
// Before this, the Image Editor's reference rail was bounded by ONE module-level constant
// (`MAX_EDIT_REFERENCES = 4`, mirroring the worker's FLUX.2 activation budget), applied to every
// multi-reference model in the catalog. Qwen-Image 2.1 takes ten, so the cap had to become a
// per-model reading — and the manifest already had the key for it on the video side
// (`limits.maxReferenceAssets`), now read on the image side too.
//
// The fallback is the caller's own constant rather than a number chosen here: a model that declares
// nothing must keep exactly the cap it had, so nothing about FLUX.2 / SenseNova / Krea moves.

// How many references this model accepts, or `fallback` when it declares none.
//
// A declared 0 is meaningful ("this model takes none") and is returned as 0, which is why this
// cannot be written as `declared || fallback`.
export function maxReferencesForModel(model, fallback) {
  const declared = model?.limits?.maxReferenceAssets;
  return Number.isInteger(declared) && declared >= 0 ? declared : fallback;
}

// Move the item at `from` to `to` in a new array, or return the input unchanged when either index
// is out of range (sc-24113).
//
// Reordering is a first-class action for this family rather than a convenience. The engine's
// template numbers the references (`<image1>` …) and its block-causal attention makes each one
// visible only to what follows, so swapping two references is a DIFFERENT request, not a cosmetic
// reshuffle. Before this the only way to reorder was to remove everything and re-add it in the
// right order, which is a reasonable thing for a user to give up on.
//
// Pure and total: never mutates, never throws, and a no-op move returns an equal array so a caller
// can set state unconditionally.
export function moveReference(ids, from, to) {
  if (!Array.isArray(ids)) return [];
  if (!Number.isInteger(from) || !Number.isInteger(to)) return ids.slice();
  if (from < 0 || to < 0 || from >= ids.length || to >= ids.length) return ids.slice();
  const next = ids.slice();
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  return next;
}

// The 1-based label the engine's own template uses for the reference at `index`.
//
// Shown next to every attached reference so the ordering the request depends on is VISIBLE. The
// prompt conventions for this family name images directly ("use the second image as a mask"), so a
// user who cannot see which one is second cannot write the prompt.
export function referenceOrdinalLabel(index) {
  return `Image ${index + 1}`;
}
