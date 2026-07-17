// Reading a recorded generation recipe back into a studio form (sc-12324).
//
// A recipe is the only record of what a user asked for, and both studios replay one, so the
// field readers live here rather than privately in either. The recipe shape is assembled by
// `build_video_sidecar_parts` / its image twin in `crates/sceneworks-core/src/project_store.rs`:
// `normalizedSettings` holds the values the app RESOLVED, `rawAdapterSettings` holds the
// `advanced` block the client sent verbatim. That split is why the two resolution readers below
// disagree on purpose — see `recipeRequestedResolution`.

export function finiteRecipeNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

// The dims the app actually RAN at, preferring the resolved `normalizedSettings` and falling
// back to the requested string. Image Studio's order: it has explicit Width/Height overrides,
// so it can represent a resolved value that no dropdown option carries.
export function recipeResolution(recipe) {
  const settings = recipe?.normalizedSettings ?? {};
  const width = finiteRecipeNumber(settings.width);
  const height = finiteRecipeNumber(settings.height);
  if (width && height) {
    return `${width}x${height}`;
  }
  return rawResolutionString(recipe);
}

// The dims the user PICKED, preferring `rawAdapterSettings.resolution` — the `advanced.resolution`
// string the client sent, which is always one of the model's `limits.resolutions` options.
//
// Video Studio's order, and the opposite of `recipeResolution` by design. Video binds resolution
// to a <select> over `limits.resolutions` with a snap effect that discards any off-list value, so
// seeding the RESOLVED dims would show a blank control and then snap away — the recorded geometry
// is post-`normalized_dimensions` (stride floor + maxPixels fit, sc-12294), which need not be a
// dropdown option. Seeding the requested string still reproduces the clip exactly: normalization
// is a pure function of (w, h, manifest), so the same request re-resolves the same way.
//
// Falls back to the resolved dims when `advanced` was never recorded (procedural/stub assets,
// whose raw settings are built without the `advanced` passthrough).
export function recipeRequestedResolution(recipe) {
  return rawResolutionString(recipe) ?? recipeResolution(recipe);
}

function rawResolutionString(recipe) {
  const raw = recipe?.rawAdapterSettings?.resolution;
  return typeof raw === "string" && raw.includes("x") ? raw : null;
}

// A recipe's LoRAs are either bare id strings or `{id, weight}` objects, depending on how the
// job was submitted.
export function recipeLoraId(lora) {
  return typeof lora === "string" ? lora : (lora?.id ?? lora?.loraId);
}

export function recipeLoraWeight(lora) {
  if (typeof lora === "string") {
    return undefined;
  }
  return finiteRecipeNumber(lora?.weight) ?? undefined;
}

// The `{ loraIds, loraWeights }` a studio seeds its picker with. A LoRA with no recorded weight
// is selected but left to its default rather than pinned to a made-up number.
export function recipeLoraSelection(recipe) {
  const loras = Array.isArray(recipe?.loras) ? recipe.loras : [];
  return {
    loraIds: loras.map(recipeLoraId).filter(Boolean),
    loraWeights: Object.fromEntries(
      loras
        .map((lora) => [recipeLoraId(lora), recipeLoraWeight(lora)])
        .filter(([id, weight]) => id && weight !== undefined),
    ),
  };
}
