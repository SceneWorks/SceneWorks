const SUPPORTED_BACKENDS = new Set(["mlx", "candle"]);
const SUPPORTED_MODES = new Set([
  "text_to_image",
  "edit_image",
  "character_image",
  "style_variations",
]);

// The catalog flag says the model family owns the control. The remaining inputs are live route
// facts: prompt enhancement is implemented only by FLUX.2-dev's base/edit providers, on both native
// backends, and never by its separate strict-control provider.
export function promptEnhancementAvailable({ model, backend, mode, strictControlActive = false }) {
  return Boolean(
    model?.id === "flux2_dev" &&
      model?.ui?.promptEnhance === true &&
      SUPPORTED_BACKENDS.has(backend) &&
      SUPPORTED_MODES.has(mode) &&
      !strictControlActive,
  );
}
