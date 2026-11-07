import { isMossImage, isMossEdit } from "./mossImageIdentity.js";

export function mossSeededRecipeReplay(asset, recipe, replaySeed) {
  if (isMossEdit(recipe?.model, recipe?.mode) && recipe?.rawAdapterSettings?.mossImageAdvanced) {
    const raw = recipe.rawAdapterSettings;
    // Every output reopens the whole conditioning batch, including outputs from later rows.
    if (Array.isArray(raw.promptBatch) && raw.promptBatch.length) {
      recipe = { ...recipe, prompt: raw.promptBatch[0] };
    }
    if (raw.mossImageAdvanced.noise) return { recipe, replaySeed: null };
    const baseSeed = asset?.generationSet?.recipe?.seed ?? raw.mossImageEditExecutionReport?.base?.samples?.[0]?.seed;
    return { recipe, replaySeed: replaySeed == null ? null : baseSeed ?? replaySeed };
  }
  const batch = recipe?.rawAdapterSettings?.promptBatch;
  if (replaySeed == null || !isMossImage(recipe?.model) || !Array.isArray(batch) || batch.length < 2) {
    return { recipe, replaySeed };
  }
  // A batch expands its base seed by output index; a sibling's own seed cannot
  // replace that base without changing every prompt/seed pair in the batch.
  const baseSeed = asset?.generationSet?.recipe?.seed;
  if (baseSeed != null && baseSeed !== "") return { recipe, replaySeed: baseSeed };
  // Older/random-seed sets may lack the original base. Replay the selected
  // image alone, whose persisted prompt is already styled, instead of shifting a batch.
  const rawAdapterSettings = { ...recipe.rawAdapterSettings };
  for (const key of ["promptBatch", "styleId", "stylePrompt"]) delete rawAdapterSettings[key];
  return { replaySeed, recipe: { ...recipe, prompt: asset?.recipe?.prompt ?? recipe.prompt,
    normalizedSettings: { ...recipe.normalizedSettings, count: 1 }, rawAdapterSettings } };
}
