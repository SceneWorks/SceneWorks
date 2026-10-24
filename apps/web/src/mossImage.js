import { mossAdvancedWire } from "./mossImageAdvanced.js";
// Product wire spelling follows gen-core::moss_image::MossImageOptions.
export const isMossImage = (id) => id === "moss_image_base" || id === "moss_image_sft";
export const mossDefaults = () => ({ acceleration: "off", max_sequence_length: 4096 });
export const mossEditDefaults = () => ({ reference_max_edge: 512, use_kv_cache: true });
export const mossEditGenerationDefaults = () => ({ width: 512, height: 512, steps: 30, guidanceScale: 4, schedulerShift: 5 });

export const isMossEdit = (model, mode) => isMossImage(model) && mode === "edit_image";

export const mossAccelerated = (options) => Boolean(options?.acceleration && options.acceleration !== "off");
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
export function mossComponentPurpose(options, mode) {
  if (mode === "edit_image") return "moss_image_edit";
  return mossAccelerated(options) ? `moss_dmd_${options.acceleration}` : null;
}
export function mossComponentState(model, options, mode) {
  const purpose = mossComponentPurpose(options, mode);
  if (!purpose) return { ready: true, purpose: null, state: "installed" };
  const component = model?.conditionalPurposes?.[purpose];
  const state = component?.blocked ? "blocked" : component?.installState ?? "missing";
  return { ready: state === "installed", purpose, state };
}
export function mossEditControlsError({ steps, guidanceScale, schedulerShift, max_sequence_length = 4096, training_shift = 1 }) {
  if (steps != null && steps !== "" && (!Number.isInteger(Number(steps)) || Number(steps) < 1)) return "MOSS edit steps must be a positive integer.";
  if (guidanceScale != null && guidanceScale !== "" && (!Number.isFinite(Number(guidanceScale)) || Number(guidanceScale) < 0)) return "MOSS edit guidance must be a finite nonnegative number.";
  if (schedulerShift != null && schedulerShift !== "" && (!Number.isFinite(Number(schedulerShift)) || Number(schedulerShift) <= 0)) return "MOSS edit schedule shift must be a finite positive number.";
  if (!Number.isInteger(Number(max_sequence_length)) || Number(max_sequence_length) < 1 || Number(max_sequence_length) > 4096) return "MOSS edit context limit must be an integer in 1–4096.";
  if (!Number.isFinite(Number(training_shift)) || Number(training_shift) <= 0) return "MOSS training shift must be finite and positive.";
  return "";
}
export function mossEditWireOptions(options) {
  const value = { ...mossEditDefaults(), ...options };
  if (!Number.isInteger(value.reference_max_edge) || value.reference_max_edge < 16 || value.reference_max_edge > 4096) {
    throw new Error("MOSS edit reference maximum edge must be an integer in 16–4096.");
  }
  if (typeof value.use_kv_cache !== "boolean" || Object.keys(value).some((key) => !["reference_max_edge", "use_kv_cache"].includes(key))) {
    throw new Error("MOSS edit controls accept only reference_max_edge and boolean use_kv_cache.");
  }
  return value;
}
// Shared by Studio and the eager Library/editor builder without importing ordinary caption tooling.
export function buildMossEditJobAdvanced(state) {
  return {
    resolution: state.resolution,
    mossImage: { acceleration: "off", max_sequence_length: state.mossImage?.max_sequence_length ?? 4096, training_shift: state.mossImage?.training_shift ?? 1, uncond_skip_layers: [] },
    mossImageEdit: mossEditWireOptions(state.mossImageEdit),
    ...(state.mossImageAdvanced || state.mossPromptBatch?.length ? { mossImageAdvanced: mossAdvancedWire(state.mossImageAdvanced ?? { output: "images" }) } : {}),
    ...(state.mossPromptBatch?.length ? { promptBatch: state.mossPromptBatch.slice() } : {}),
    steps: state.stepsOverride === "" || state.stepsOverride == null ? 30 : Number(state.stepsOverride),
    guidanceScale: state.guidanceOverride === "" || state.guidanceOverride == null ? 4 : Number(state.guidanceOverride),
    schedulerShift: state.schedulerShift == null || state.schedulerShift === "" ? 5 : Number(state.schedulerShift),
    // Effective native rows replay directly; a scalar raw Style prompt cannot describe the batch.
    ...(state.styleId ? { styleId: state.styleId, ...(state.styleUserPrompt === null ? {} : { stylePrompt: state.styleUserPrompt ?? "" }) } : {}),
  };
}
// References are semantic slots. Repeating an asset intentionally occupies multiple slots.
export function mossEditReferenceIds({ referenceAssetIds = [], sourceAssetId, referenceAssetId }) {
  const ids = [...referenceAssetIds];
  for (const alias of [referenceAssetId, sourceAssetId]) {
    if (alias == null || alias === "") continue;
    if (!ids.length) ids.push(alias);
    else if (ids[0] !== alias) throw new Error("Source/reference aliases conflict with the first ordered MOSS reference.");
  }
  if (!ids.length || ids.length > 4 || ids.some((id) => typeof id !== "string" || !id || id.trim() !== id)) {
    throw new Error("MOSS editing requires 1–4 ordered image references.");
  }
  return ids;
}
