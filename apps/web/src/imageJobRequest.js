import { buildImageJobAdvanced } from "./imageJobAdvanced.js";
import { effectiveFitMode } from "./components/FitModeControl.jsx";
import { upscaleEngineHasSoftness } from "./upscaleEngines.js";

// sc-11219 (F-031): single pure builder for the Image Studio job request, shared by both the
// single "Generate" submit and the batch run. It used to be duplicated — `submit()` held the
// canonical ~105-line createImageJob payload and `buildBatchJobRequest` was a hand-copied
// near-twin that had DRIFTED: the batch copy dropped the top-level `referenceAssetId` img2img
// branch and omitted `pidTarget` + the `supportsImg2img`/`img2imgReferenceAssetId`/
// `img2imgStrength` trio from `advanced`. So a batch on an img2img-capable model (Krea 2 Turbo)
// silently ignored the reference image and a PiD "2K" batch rendered at the 4K default. Extracting
// one builder makes the two paths byte-identical for the same visible settings; the ONLY intended
// difference is the per-item prompt / structured-caption / per-prompt-resolution override, which
// each caller resolves and passes in (promptToSend / submitIntent / sendStructured / submitCaption
// / submitBackend / resolutionOverride).
//
// This is the single-submit payload verbatim (the correct reference), parameterized only by those
// overrides. Every omit-when-default and mode gate is preserved; `advanced` is delegated to the
// already-pure buildImageJobAdvanced so its guards stay in one place.
export function buildImageJobRequest(state) {
  const {
    // Prompt / resolution overrides — the one legitimate single-vs-batch difference.
    promptToSend,
    submitIntent,
    sendStructured,
    submitCaption,
    submitBackend,
    // A per-prompt [WxH] directive (sc-10063) overrides the studio resolution for a batch job.
    resolutionOverride,
    resolution,
    // Shared studio settings (identical for both paths).
    mode,
    negativePrompt,
    model,
    count,
    seed,
    posePayload,
    width,
    height,
    recipePresetId,
    characterId,
    characterLookId,
    multiReference,
    editSecondPair,
    sourceAssetId,
    controlPreprocessSourceId,
    referenceAssetIds,
    fitMode,
    editInpaintCapable,
    referenceAssetId,
    supportsImg2img,
    img2imgReferenceAssetId,
    loras,
    upscaleEnabled,
    upscaleFactor,
    upscaleEngine,
    upscaleSoftness,
    // Advanced knobs (delegated to buildImageJobAdvanced).
    sampler,
    scheduler,
    schedulerShift,
    stepsOverride,
    guidanceOverride,
    guidanceMethod,
    flashAttn,
    promptEnhance,
    enhancePrompt,
    precisionToggle,
    bf16Precision,
    showTierPicker,
    quantTier,
    showPidToggle,
    usePid,
    pidTarget,
    hideReferenceStrength,
    ipAdapterScale,
    identityStructure,
    controlnetScale,
    variationStrength,
    trueCfgScale,
    img2imgStrength,
    viewAngles,
    viewAngle,
    faceRestore,
    controlActive,
    activeControlMode,
    controlPassthroughId,
    effectiveControlScale,
    controlOverlayId,
  } = state;

  return {
    mode,
    prompt: promptToSend,
    negativePrompt,
    model,
    count: posePayload.length ? 1 : count,
    seed: seed === "" ? null : Number(seed),
    // A per-prompt [WxH] directive (sc-10063) overrides the studio resolution for this job.
    width: resolutionOverride?.width ?? width,
    height: resolutionOverride?.height ?? height,
    recipePresetId,
    characterId: mode === "character_image" ? characterId || null : null,
    characterLookId: mode === "character_image" ? characterLookId || null : null,
    // edit_image: a single source image, except for a multi-reference model (sc-6211,
    // FLUX.2-dev) whose source picker is replaced by the multi-image reference picker below.
    // text_to_image strict-control (sc-8245): canny/depth in preprocess (derive) mode send the
    // uploaded control image here as the source the worker auto-derives the map FROM
    // (strict_control.rs `resolve_control_source`). Passthrough mode uses `advanced.controlImage`.
    sourceAssetId:
      mode === "edit_image" && !multiReference
        ? // A two-reference edit sends the ordered [image1, image2] pair as referenceAssetIds instead
          // (epic 10871 P1.3), so the single sourceAssetId is dropped when a second image is chosen.
          editSecondPair
          ? null
          : sourceAssetId || null
        : controlPreprocessSourceId,
    // Multi-reference edit (sc-6211): the plural reference set the FLUX.2-dev edit conditions on.
    // Only sent in edit_image mode for a multiReference model; the worker routes a non-empty list
    // to Conditioning::MultiReference (one image ⇒ a normal single-reference edit). The Krea
    // two-reference edit (epic 10871 P1.3) reuses this channel with the ordered [image1, image2] pair.
    referenceAssetIds:
      mode === "edit_image" && multiReference && referenceAssetIds.length
        ? referenceAssetIds
        : (editSecondPair ?? undefined),
    // Fit mode applies to edits only; coerced so a stale "outpaint" never reaches a
    // non-inpaint model (epic 2551). Omitted for non-edit modes (worker default crop).
    fitMode: mode === "edit_image" ? effectiveFitMode(fitMode, editInpaintCapable) : undefined,
    // character_image: the IP-Adapter identity reference. Otherwise, on an img2img-capable model
    // (Krea 2 Turbo, sc-8593), the reference picked in the "Start from an image" panel — sent so the
    // worker's krea arm routes it to img2img latent-init (advanced.strength below).
    referenceAssetId:
      mode === "character_image"
        ? referenceAssetId || null
        : supportsImg2img
          ? img2imgReferenceAssetId || null
          : null,
    loras,
    ...(upscaleEnabled
      ? {
          upscale: {
            enabled: true,
            factor: upscaleFactor,
            engine: upscaleEngine,
            // SeedVR2-only detail/softness knob (sc-4815); omitted for engines that ignore it.
            ...(upscaleEngineHasSoftness(upscaleEngine) ? { softness: upscaleSoftness } : {}),
          },
        }
      : {}),
    // advanced payload (sc-8854, F-052): assembled by the pure buildImageJobAdvanced
    // builder. Every omit-when-default rule (which keeps saved recipes byte-identical)
    // lives in imageJobAdvanced.js and is covered by imageJobAdvanced.test.js.
    advanced: buildImageJobAdvanced({
      resolution: resolutionOverride
        ? `${resolutionOverride.width}x${resolutionOverride.height}`
        : resolution,
      sendStructured,
      submitIntent,
      submitCaption,
      submitBackend,
      sampler,
      scheduler,
      schedulerShift,
      stepsOverride,
      guidanceOverride,
      guidanceMethod,
      flashAttn,
      promptEnhance,
      enhancePrompt,
      precisionToggle,
      bf16Precision,
      showTierPicker,
      quantTier,
      showPidToggle,
      usePid,
      pidTarget,
      mode,
      referenceAssetId,
      hideReferenceStrength,
      ipAdapterScale,
      // img2img (sc-8593): emit advanced.strength when an img2img-capable model has a reference.
      supportsImg2img,
      img2imgReferenceAssetId,
      img2imgStrength,
      identityStructure,
      controlnetScale,
      variationStrength,
      trueCfgScale,
      viewAngles,
      viewAngle,
      posePayload,
      faceRestore,
      controlActive,
      activeControlMode,
      controlPassthroughId,
      effectiveControlScale,
      controlOverlayId,
    }),
  };
}
