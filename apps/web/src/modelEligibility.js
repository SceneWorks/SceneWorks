// Shared per-screen model-eligibility predicates (sc-5946). Each Studio already
// derived "can this model serve this screen?" inline; those predicates are lifted here
// verbatim so the per-Studio availability gates and the screens themselves agree on
// exactly which models qualify. Predicates are capability + Mac-gating only (no install
// state) — callers layer installState via the helpers at the bottom.
import { macModelBlock, macModelFeatureBlock, macVideoModeBlock } from "./macGating.js";
import { VISION_CAPTION_MODEL_ID } from "./constants.js";

// Image generation modes a model can serve (ImageStudio mode tabs).
const IMAGE_MODES = ["text_to_image", "edit_image", "character_image", "style_variations"];

// Video generation modes a model can advertise (VideoStudio modeOptions).
export const VIDEO_MODES = [
  "text_to_video",
  "image_to_video",
  "first_last_frame",
  "extend_clip",
  "video_bridge",
  "replace_person",
  "video_to_video",
  "reference_to_video",
  "reference_video_to_video",
  "multi_video_to_video",
  "ads2v",
  "animate_character",
];

// Image Studio — mirrors ImageStudio.jsx `imageModelServesMode`. A model serves a mode
// when it declares the capability and, under active Mac gating, that feature is MLX-routed
// for it (the macModelFeatureBlock calls are no-ops off Mac).
export function imageModelServesMode(model, mode, caps) {
  const capabilities = model?.capabilities ?? [];
  if (mode === "edit_image") {
    return (
      (capabilities.includes("edit_image") || capabilities.includes("image_edit")) &&
      !macModelFeatureBlock(model, caps, "edit")
    );
  }
  if (mode === "character_image") {
    return capabilities.includes("character_image") && !macModelFeatureBlock(model, caps, "reference");
  }
  if (mode === "style_variations") {
    return capabilities.includes("style_variations") && !macModelFeatureBlock(model, caps, "reference");
  }
  return capabilities.includes("text_to_image");
}

// Usable on Image Studio: an image-type model that isn't Mac-blocked and serves ≥1 mode.
export function imageModelUsable(model, caps) {
  return (
    model?.type === "image" &&
    !macModelBlock(model, caps) &&
    IMAGE_MODES.some((mode) => imageModelServesMode(model, mode, caps))
  );
}

// Video Studio — mirrors VideoStudio.jsx `modelServesMode`.
export function videoModelServesMode(model, mode, caps) {
  return Boolean(model?.capabilities?.includes(mode)) && !macVideoModeBlock(model, caps, mode);
}

// Usable on Video Studio: a video-type model that isn't Mac-blocked and serves ≥1 mode.
export function videoModelUsable(model, caps) {
  return (
    model?.type === "video" &&
    !macModelBlock(model, caps) &&
    VIDEO_MODES.some((mode) => videoModelServesMode(model, mode, caps))
  );
}

// Document Studio — mirrors DocumentStudio.jsx `modelSupportsInterleave` (SenseNova-U1).
export function documentModelUsable(model, caps) {
  return (
    model?.type === "image" &&
    !macModelBlock(model, caps) &&
    Array.isArray(model?.capabilities) &&
    model.capabilities.includes("interleave")
  );
}

// Reference-image → JSON caption (epic 8102, sc-8110). The vision captioner is a single,
// catalog-pinned utility model (`vision_caption_qwen3vl_8b`), so usability is "this IS the
// captioner model AND it can run here", not a capability sweep. Two gates, mirroring the
// magic-prompt model gate:
//   * macModelBlock — the active-gating Rust/MLX oracle (a no-op off Mac / in observe mode).
//   * macOnly — cross-platform as of sc-8116 (epic 8103): the catalog flips `macOnly: false` now that
//     the candle qwen3_vl vision tower (candle-llm sc-8080) is in the backend-candle graph, so this
//     branch is a no-op label and the feature lights up on Windows/Linux too. The guard is kept
//     defensively: if some future catalog entry re-sets `macOnly: true` it still hides off Mac, and
//     `caps.platform` is the API host's OS (mac_capabilities in workers.rs) — empty (`""`) pre-load
//     doesn't block, matching the no-op-pre-load convention of the macGating helpers.
export function visionCaptionModelUsable(model, caps) {
  if (model?.id !== VISION_CAPTION_MODEL_ID) {
    return false;
  }
  if (macModelBlock(model, caps)) {
    return false;
  }
  if (model?.macOnly === true) {
    const platform = caps?.platform ?? "";
    if (platform && platform !== "macos") {
      return false;
    }
  }
  return true;
}

// Character Studio — mirrors CharacterStudio.jsx angle/pose predicates.
export function angleModelUsable(model, caps) {
  return !macModelBlock(model, caps) && Array.isArray(model?.ui?.viewAngles) && model.ui.viewAngles.length > 0;
}

export function poseModelUsable(model, caps) {
  return !macModelBlock(model, caps) && Boolean(model?.ui?.poseLibrary);
}

// Strict-control modes the selected backbone advertises (sc-8245). The single source of truth is the
// model's `ui.controlModes` — mirrored into constants.js from the manifest (the worker's
// STRICT_CONTROL_ENGINES `supported_kinds`). Canonical-ordered (pose, canny, depth) and de-duplicated so
// the picker renders deterministically regardless of manifest ordering; unknown modes are dropped (the
// worker only admits pose/canny/depth). Empty array ⇒ the backbone supports no strict control and the
// panel hides. Pure (model in → modes out), so the picker and its gate share exactly one notion of
// "supported_kinds".
export const CONTROL_MODE_ORDER = ["pose", "canny", "depth"];

export function supportedControlModes(model) {
  const declared = Array.isArray(model?.ui?.controlModes) ? model.ui.controlModes : [];
  const present = new Set(
    declared
      .filter((mode) => typeof mode === "string")
      .map((mode) => mode.trim().toLowerCase()),
  );
  return CONTROL_MODE_ORDER.filter((mode) => present.has(mode));
}

// Character Studio needs Angles OR Poses support.
export function characterModelUsable(model, caps) {
  return angleModelUsable(model, caps) || poseModelUsable(model, caps);
}

// Does the user have ≥1 present (installed or incomplete) model usable on this screen?
// Matches what the Studios' pickers actually offer (they source from models with
// installState !== "missing"), so a screen is never gated while its picker still shows a
// model. Only a fully-missing catalog gates the screen.
export function hasUsableModelFor(models, predicate, caps) {
  // `usable !== false` keeps the screen-gate in lockstep with the pickers, which
  // exclude not-yet-runnable external base models (sc-10667).
  return (models ?? []).some(
    (model) => model?.installState !== "missing" && model?.usable !== false && predicate(model, caps),
  );
}

// The models to OFFER for download when a screen is gated: catalog models usable on the
// screen that aren't installed, recommended-first. Falls back to any eligible-but-not-
// installed model so every screen has at least the models that would unlock it.
export function downloadOffersFor(models, predicate, caps) {
  const eligible = (models ?? []).filter(
    (model) => model?.installState !== "installed" && predicate(model, caps),
  );
  const recommended = eligible.filter((model) => model?.recommended === true);
  return recommended.length ? recommended : eligible;
}
