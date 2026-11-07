const SUPPORTED_MODES_BY_BACKEND = new Map([
  // MLX's native FLUX.2 edit provider owns the character/reference route as well as ordinary edits.
  ["mlx", new Set(["text_to_image", "edit_image", "character_image"])],
  // Candle FLUX.2-dev deliberately owns only sourceless base generation and canonical edit_image.
  // Its bespoke edit predicate rejects the legacy reference/image_to_image aliases and character
  // route; advertising enhancement there would let a referenceAssetId fall through to plain base.
  ["candle", new Set(["text_to_image", "edit_image"])],
]);

// The catalog flag says the model family owns the control. The remaining inputs are live route
// facts: FLUX.2-dev owns base/edit; FLUX 3 Klein owns its current text-to-image route with a
// separately installed optional upsampler. Neither grants enhancement to strict-control routes.
export function promptEnhancementAvailable({ model, backend, mode, strictControlActive = false }) {
  if (model?.ui?.promptEnhance !== true || strictControlActive) return false;
  if (model?.id === "flux3_klein" || model?.id === "flux3_klein_base") {
    // Boxes and references choose the native upsampler's four modes from request geometry;
    // they are not a second user-selected mode that could disagree with the attached inputs.
    return (backend === "mlx" || backend === "candle") &&
      ["text_to_image", "edit_image", "character_image", "image_to_image"].includes(mode);
  }
  const supportedModes = SUPPORTED_MODES_BY_BACKEND.get(backend);
  return Boolean(model?.id === "flux2_dev" && supportedModes?.has(mode));
}
