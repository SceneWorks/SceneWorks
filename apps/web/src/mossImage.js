// Product wire spelling follows gen-core::moss_image::MossImageOptions.
export const isMossImage = (id) => id === "moss_image_base" || id === "moss_image_sft";
export const mossDefaults = () => ({ acceleration: "off", max_sequence_length: 4096 });
export const mossAccelerated = (options) => Boolean(options?.acceleration && options.acceleration !== "off");
export function mossSeededRecipeReplay(asset, recipe, replaySeed) {
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
export function mossWireOptions(options) {
  const value = { ...mossDefaults(), ...options };
  if (mossAccelerated(value)) {
    delete value.training_shift;
    delete value.uncond_skip_layers;
    delete value.slg;
  }
  return value;
}
// Hydrate inactive ordinary controls through the same state setters/snapshots as
// active controls, so selecting None restores the author's previous settings.
export function mossPresetWithOrdinarySettings(preset) {
  const ordinary = preset?.defaults?.mossOrdinarySettings;
  if (!ordinary) return preset;
  const { options, ...controls } = ordinary;
  return { ...preset, defaults: { ...preset.defaults, ...controls,
    mossImage: { ...options, acceleration: preset.defaults.mossImage?.acceleration ?? "off" },
  } };
}
export function mossComponentPurpose(options) {
  return mossAccelerated(options) ? `moss_dmd_${options.acceleration}` : null;
}
export function mossComponentState(model, options) {
  const purpose = mossComponentPurpose(options);
  if (!purpose) return { ready: true, purpose: null, state: "installed" };
  const state = model?.conditionalPurposes?.[purpose]?.installState ?? "missing";
  return { ready: state === "installed", purpose, state };
}

