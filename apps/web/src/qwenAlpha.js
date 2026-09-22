// Transparency (RGBA output) request adapter — the web mirror of
// `crates/sceneworks-worker/src/qwen_alpha.rs` (sc-24113, epic 24107).
//
// Qwen-Image 2.1's VAE can decode straight to four channels, and sc-24111 already widened the whole
// SceneWorks egress path to carry that — the worker types a buffer by its channel count, the PNG
// writer preserves alpha, the upscale/detail passes route the alpha plane around the 3-channel
// models, and the Image Editor composites and exports without flattening onto black. What was
// missing is the ingress half: a way for the user to ask for it.
//
// This module is deliberately the ONLY place the web spells the names the S4 RGBA contract
// (inference sc-24111, PR #1009) turns on, for the same reason the Rust module is the only place
// the worker does. Its mirror is `crates/sceneworks-worker/src/qwen_alpha.rs`, and tests on both
// sides pin the two copies together.
//
// # Transparency is PROMPT-DRIVEN; the toggle only opens the surface
//
// The thing a reader will assume wrongly: there is NO transparency flag or mode upstream. The 2.1
// VAE is always four-channel. `output_channels: Rgba` does not ask the model for a transparent
// background — it asks for the fourth channel to survive the decode instead of being composited
// over white. Whether that channel holds a cut-out is decided by the PROMPT, using the model card's
// own convention.
//
// So the toggle alone can hand a user a fully opaque RGBA PNG and look broken. It is paired with
// `transparencyPromptSuggestion`, which the UI offers as visible, EDITABLE wording — never appended
// silently, because a prompt the user cannot see is a prompt they cannot fix.

/// The manifest/descriptor flag a model sets to advertise four-channel decode
/// (`Capabilities::supports_alpha_output`, surfaced on the catalog entry).
export const MODEL_ALPHA_CAPABILITY_FLAG = "supportsAlphaOutput";

/// `GenerationRequest::output_channels` — an ENUM (`Rgb` | `Rgba`), not a channel count.
export const REQUEST_OUTPUT_CHANNELS = "output_channels";

/// The two `gen_core::OutputChannels` variants. `Rgb` is the contract's default.
export const OUTPUT_CHANNELS_RGB = "rgb";
export const OUTPUT_CHANNELS_RGBA = "rgba";

/// The model card's own wording for asking 2.1 for a transparent background. Byte-identical to the
/// Rust mirror's `TRANSPARENCY_PROMPT_HINT`.
export const TRANSPARENCY_PROMPT_HINT =
  "The background is transparent. This is an RGBA image with transparency.";

// The `advanced` key the Studio and Editor set when the toggle is on. This one is OURS — a
// SceneWorks request axis, not an engine field — so it is stable and not part of the pair above.
export const ADVANCED_TRANSPARENT_BACKGROUND = "transparentBackground";

export const CHANNELS_RGB = 3;
export const CHANNELS_RGBA = 4;

// Does this catalog model advertise alpha output?
//
// Strictly `=== true`. Unlike the sc-15299 generation axes (`supportsGuidance` and friends, where
// ABSENT MEANS TRUE), a missing flag here means "no": transparency is a capability a model either
// has or does not, and defaulting it on would offer a cut-out toggle on every model in the catalog
// and fail the render at the worker.
export function modelSupportsAlphaOutput(model) {
  return model?.[MODEL_ALPHA_CAPABILITY_FLAG] === true;
}

// Should the transparency toggle be shown at all for this model?
//
// Separate from the value so a stale sticky `true` on a non-capable model is inert — the same shape
// `showPidToggle` / `bf16Precision` use in ImageStudio. The toggle is hidden rather than disabled:
// a control that can never apply to this model is noise, not information.
export function showTransparencyToggle(model) {
  return modelSupportsAlphaOutput(model);
}

// The `advanced` fragment for a transparency request.
//
// Returns an EMPTY object when the toggle is off OR the model cannot serve it. Two consequences,
// both deliberate:
//
//   * An opaque render puts NOTHING new on the wire, so every existing lane's payload is
//     byte-identical and no worker sees a field it does not know.
//   * A stale sticky `true` carried over from a transparency-capable model cannot leak onto a model
//     that would refuse it — the gate is re-evaluated against the CURRENTLY selected model at build
//     time, not at toggle time.
export function transparencyAdvanced(model, transparentBackground) {
  if (!transparentBackground || !modelSupportsAlphaOutput(model)) {
    return {};
  }
  return { [ADVANCED_TRANSPARENT_BACKGROUND]: true };
}

// The channel count a request resolves to, for display and for the recipe.
export function resolveOutputChannels(model, transparentBackground) {
  return transparentBackground && modelSupportsAlphaOutput(model) ? CHANNELS_RGBA : CHANNELS_RGB;
}

// The `gen_core::OutputChannels` variant a resolved channel count names.
//
// An enum, not the count. The count is how SceneWorks reasons about the buffer it gets BACK; the
// request says `Rgb` or `Rgba`.
export function outputChannelsVariant(channels) {
  return channels === CHANNELS_RGBA ? OUTPUT_CHANNELS_RGBA : OUTPUT_CHANNELS_RGB;
}

// The prompt the user would end up sending, given their own prompt and the transparency toggle —
// or `null` when there is nothing to offer.
//
// `null` covers transparency-off and the case where the prompt ALREADY says it. The second matters:
// a user who typed their own transparency wording must not be offered a near-duplicate sentence,
// and re-running a recipe must not accrete the hint once per run.
//
// This never applies on its own. The caller shows the result as editable text the user accepts —
// see the module note on why a silent append would be the wrong behaviour.
export function transparencyPromptSuggestion(prompt, transparentBackground) {
  if (!transparentBackground) return null;
  const text = typeof prompt === "string" ? prompt : "";
  const lowered = text.toLowerCase();
  if (lowered.includes("transparent") || lowered.includes("rgba")) return null;
  const trimmed = text.replace(/\s+$/u, "");
  if (!trimmed) return TRANSPARENCY_PROMPT_HINT;
  const separator = /[.!?]$/u.test(trimmed) ? " " : ". ";
  return `${trimmed}${separator}${TRANSPARENCY_PROMPT_HINT}`;
}

// Read the toggle back out of a stored recipe / `advanced` block.
//
// Anything that is not boolean `true` is off. A toggle that arrives malformed from an old recipe or
// a hand-edited payload must not silently change what the render emits.
export function transparencyRequested(advanced) {
  return advanced?.[ADVANCED_TRANSPARENT_BACKGROUND] === true;
}
