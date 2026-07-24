import { buildImageJobRequest } from "../imageJobRequest.js";
import { composeStyledPrompt } from "../styleComposer.js";
import { styleTextForId } from "../data/styleCatalog.js";
import { defaultTierSelection, installedTiers } from "../quantTier.js";
import { suggestTier } from "../tierSuggestion.js";
import { readLastTier } from "../lastTierStore.js";
import { readDefaultGenerationQuality } from "../generationQuality.js";

// Job-request builders for the Simple UI (design handoff "Simple UI for creative studios").
//
// The Simple studios expose a REDUCED control surface, not a different generation
// contract: a run submitted from Simple must be the same job the full workspace would
// submit with those same visible settings, defaulted everywhere else. So the image
// path delegates to the SAME pure `buildImageJobRequest` the advanced Image Studio (and
// its batch run) uses — this module only supplies the neutral defaults for the knobs
// Simple doesn't show. That way an advanced knob added later can't silently change
// meaning here: it either keeps its omit-when-default rule (and Simple is unaffected)
// or the shared builder's tests catch the drift.
//
// Video/audio have no extracted builder upstream (their payloads are assembled inline in
// VideoStudio/AudioStudio), so those two are written out here against the same DTOs the
// studios post — kept deliberately small and free of the mode-specific branches Simple
// never reaches.

// Quant tier is the ONE "advanced" knob Simple still has to resolve, because omitting it
// would silently ignore the app-wide "Default quality" setting the Simple Settings screen
// writes. This mirrors the advanced studios' seed effect exactly: the per-(screen,model)
// sticky first, then the global quality setting, then the capability-aware Auto suggestion —
// all clamped to what's actually installed.
export function resolveSimpleTier(model, screen, options = {}) {
  const { convRotEligible = false, nvfp4Eligible = false, unifiedMemoryGb = null } = options;
  const tierOptions = { convRotEligible, nvfp4Eligible };
  const available = installedTiers(model, tierOptions);
  if (available.length === 0) {
    return { quantTier: "", tierExplicit: false };
  }
  const sticky = readLastTier(screen, model?.id ?? "");
  const quantTier =
    defaultTierSelection(model, sticky, {
      ...tierOptions,
      defaultQuality: readDefaultGenerationQuality(),
      autoTier: suggestTier(model, unifiedMemoryGb),
    }) ?? "";
  return {
    quantTier,
    // Only a DELIBERATE prior pick (the sticky) marks the tier explicit, so the worker's
    // capability clamp may still downtier a derived default that doesn't fit (sc-10733).
    tierExplicit: Boolean(quantTier) && sticky === quantTier,
  };
}

// Neutral values for every advanced knob the Simple Image Studio does not render. Each one is
// the value that makes `buildImageJobAdvanced`'s omit-when-default rule drop the key entirely,
// so a Simple payload's `advanced` is exactly `{ resolution }` (+ styleId when a style is picked,
// + mlxQuantize when a tier resolves) — byte-identical to an advanced run with untouched controls.
const IMAGE_DEFAULTS = Object.freeze({
  negativePrompt: "",
  seed: "",
  posePayload: [],
  recipePresetId: null,
  presetPromptResolvedClientSide: false,
  characterId: null,
  characterLookId: null,
  multiReference: false,
  editSecondPair: null,
  controlPreprocessSourceId: null,
  referenceAssetIds: [],
  fitMode: "crop",
  editInpaintCapable: false,
  referenceAssetId: null,
  supportsImg2img: false,
  img2imgReferenceAssetId: null,
  loras: [],
  upscaleEnabled: false,
  upscaleFactor: 2,
  upscaleEngine: "",
  upscaleSoftness: 0,
  sampler: "default",
  scheduler: "default",
  schedulerShift: "",
  stepsOverride: "",
  guidanceOverride: "",
  guidanceMethod: "cfg",
  flashAttn: true,
  promptEnhance: false,
  enhancePrompt: false,
  precisionToggle: false,
  bf16Precision: false,
  showTierPicker: false,
  showPidToggle: false,
  usePid: false,
  pidTarget: "4k",
  hideReferenceStrength: false,
  ipAdapterScale: 0,
  identityStructure: false,
  controlnetScale: 0,
  variationStrength: false,
  trueCfgScale: 0,
  img2imgStrength: 0,
  supportsTextStyle: false,
  textStyleGain: 1,
  viewAngles: null,
  viewAngle: "",
  faceRestore: false,
  controlActive: false,
  activeControlMode: "pose",
  controlPassthroughId: null,
  effectiveControlScale: 0,
  controlOverlayId: null,
  multiPhaseActive: false,
  phases: [],
  sendStructured: false,
  submitCaption: null,
  submitBackend: null,
  resolutionOverride: null,
});

/**
 * Parse a "WIDTHxHEIGHT" resolution string into numbers. Returns null when unparseable
 * so callers can refuse to submit rather than post a NaN geometry.
 */
export function parseResolutionPair(resolution) {
  const [width, height] = String(resolution ?? "")
    .toLowerCase()
    .split("x")
    .map((part) => Number.parseInt(part, 10));
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

/**
 * The Image Studio job request for a Simple run.
 * @param {object} input
 * @param {string} input.prompt
 * @param {"text_to_image"|"edit_image"} input.mode
 * @param {string} input.model - catalog model id.
 * @param {string} input.resolution - "WIDTHxHEIGHT".
 * @param {number} input.count - variations.
 * @param {string|null} input.styleId - Style Catalog group/sub-style id, or null for "None".
 * @param {string|null} input.sourceAssetId - the edit source (edit_image only).
 * @param {string} [input.quantTier] / @param {boolean} [input.tierExplicit] - from resolveSimpleTier.
 * @returns {object|null} the payload for createImageJob, or null when the resolution is invalid.
 */
export function buildSimpleImageRequest({
  prompt,
  mode,
  model,
  resolution,
  count,
  styleId,
  sourceAssetId,
  quantTier = "",
  tierExplicit = false,
}) {
  const size = parseResolutionPair(resolution);
  if (!size) {
    return null;
  }
  return buildImageJobRequest({
    ...IMAGE_DEFAULTS,
    promptToSend: prompt,
    submitIntent: prompt,
    resolution,
    width: size.width,
    height: size.height,
    mode,
    model,
    count,
    // The composer wants the style's free text; the id rides `advanced` so a Simple run
    // replays into the advanced Style picker unchanged (sc-13132).
    styleText: styleId ? styleTextForId(styleId) : null,
    styleId: styleId || null,
    sourceAssetId: sourceAssetId || null,
    quantTier,
    tierExplicit,
  });
}

/**
 * The Video Studio job request for a Simple run. Mirrors VideoStudio.submit()'s payload with
 * every mode-specific channel Simple doesn't expose left at its empty default, so the DTO
 * shape the API deserializes is unchanged.
 */
export function buildSimpleVideoRequest({
  prompt,
  mode,
  model,
  resolution,
  duration,
  fps,
  styleId,
  sourceAssetId,
}) {
  const size = parseResolutionPair(resolution);
  if (!size) {
    return null;
  }
  const styleText = styleId ? styleTextForId(styleId) : null;
  const styleApplied = Boolean(styleText && styleText.trim());
  return {
    mode,
    // The Style Catalog wrap is applied client-side, exactly as VideoStudio does (sc-13136);
    // `presetPromptResolvedClientSide` then tells the server to skip its own fold.
    prompt: styleApplied ? composeStyledPrompt({ styleText, userPrompt: prompt }) : prompt,
    negativePrompt: "",
    model,
    duration: Number(duration),
    fps: Number(fps),
    width: size.width,
    height: size.height,
    quality: "balanced",
    seed: null,
    recipePresetId: null,
    presetPromptResolvedClientSide: styleApplied || undefined,
    characterId: null,
    characterLookId: null,
    sourceAssetId: mode === "image_to_video" ? sourceAssetId || null : null,
    lastFrameAssetId: null,
    sourceClipAssetId: null,
    sourceClipAssetIds: [],
    bridgeRightClipAssetId: null,
    referenceAssetIds: [],
    referenceClipAssetId: null,
    personTrackId: null,
    replacementMode: "face_only",
    loras: [],
    advanced: {
      resolution,
      ...(styleApplied ? { styleId, stylePrompt: prompt } : {}),
    },
  };
}

/**
 * The Audio Studio job request for a Simple run. Mirrors AudioStudio.submit()'s base payload
 * ({ model, prompt, language, targetDurationSecs, seed }) plus the single mode-specific field
 * Simple surfaces — the Speech voice.
 */
export function buildSimpleAudioRequest({ model, prompt, mode, voice, durationSecs }) {
  const target = Number(durationSecs);
  const payload = {
    model,
    prompt: String(prompt ?? "").trim(),
    // Omitted when the model declares no length cap, so it synthesizes its natural length.
    targetDurationSecs: durationSecs != null && Number.isFinite(target) && target > 0 ? target : undefined,
    seed: null,
  };
  if (mode === "speech" && voice) {
    payload.voice = voice;
  }
  return payload;
}
