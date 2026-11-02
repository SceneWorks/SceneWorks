// sc-26691 — MiniMax-H3's image-reference resize policy on "Use this recipe".
//
// The Video Studio exposes no reference-resize control (nor the short edge it governs), but a
// replay must still reproduce how the recorded clip encoded its references. The worker records the
// RESOLVED `referenceImageResizeMode` in every H3 asset's `rawAdapterSettings` (beside
// `minimaxH3Tier`, which only the H3 arm writes). A recorded H3 asset WITHOUT the key was rendered
// before the mode existed, under the forced short edge — `diffusers` — so its replay asks for that
// rather than silently taking today's `match` default.

export const REFERENCE_IMAGE_RESIZE_MODES = ["match", "max", "diffusers"];
export const LEGACY_REFERENCE_IMAGE_RESIZE_MODE = "diffusers";

/** Whether a catalog entry renders through MiniMax-H3 (stock `minimax_h3*` or an imported H3). */
export function isMinimaxH3Entry(entry) {
  return Boolean(
    (typeof entry?.id === "string" && entry.id.startsWith("minimax_h3")) ||
      entry?.family === "minimax-h3",
  );
}

/**
 * The reference-sizing knobs a video recipe replays with, or `null` when the recipe is not a
 * recorded MiniMax-H3 render (another family, or a hand-authored / shared recipe with no H3
 * provenance — those take the engine defaults like any new request).
 */
export function recipeReferenceResize(recipe) {
  const raw = recipe?.rawAdapterSettings ?? {};
  const recordedH3 =
    typeof raw.minimaxH3Tier === "string" ||
    (typeof recipe?.model === "string" && recipe.model.startsWith("minimax_h3") &&
      Object.prototype.hasOwnProperty.call(raw, "referenceImageResizeMode"));
  if (!recordedH3) return null;
  const recorded = raw.referenceImageResizeMode;
  if (recorded !== undefined && !REFERENCE_IMAGE_RESIZE_MODES.includes(recorded)) return null;
  const restored = { referenceImageResizeMode: recorded ?? LEGACY_REFERENCE_IMAGE_RESIZE_MODE };
  if (Number.isInteger(raw.referenceImageShortEdge)) {
    restored.referenceImageShortEdge = raw.referenceImageShortEdge;
  }
  return restored;
}
