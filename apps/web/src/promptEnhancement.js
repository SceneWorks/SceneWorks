const SUPPORTED_MODES_BY_BACKEND = new Map([
  // MLX's native FLUX.2 edit provider owns the character/reference route as well as ordinary edits.
  ["mlx", new Set(["text_to_image", "edit_image", "character_image"])],
  // Candle FLUX.2-dev deliberately owns only sourceless base generation and canonical edit_image.
  // Its bespoke edit predicate rejects the legacy reference/image_to_image aliases and character
  // route; advertising enhancement there would let a referenceAssetId fall through to plain base.
  ["candle", new Set(["text_to_image", "edit_image"])],
]);

// The catalog flag says the model family owns the control. The remaining inputs are live route
// facts: prompt enhancement is implemented only by FLUX.2-dev's base/edit providers, on both native
// backends, and never by its separate strict-control provider.
export function promptEnhancementAvailable({ model, backend, mode, strictControlActive = false }) {
  const supportedModes = SUPPORTED_MODES_BY_BACKEND.get(backend);
  return Boolean(
    model?.id === "flux2_dev" &&
      model?.ui?.promptEnhance === true &&
      supportedModes?.has(mode) &&
      !strictControlActive,
  );
}
