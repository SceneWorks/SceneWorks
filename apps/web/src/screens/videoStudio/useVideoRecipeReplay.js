import { DEFAULT_MOTION, LTX25_AUTO_DURATION_MIN_SECONDS, LTX25_AUTO_DURATION_MAX_SECONDS, amoralTextEncoderId, clampLtx25AutoDurationSeconds, clampLtx25TemporalUpsampleRounds } from "./videoTierHelpers.js";
import { useEffect } from "react";
import { noPresetId } from "../../presetUtils.js";
import { VIDEO_MODES } from "../../modelEligibility.js";
import { recipeReferenceResize } from "../../minimaxH3ReferenceResize.js";
import { quantizeTier } from "../../quantTier.js";
import { finiteRecipeNumber, recipeLoraSelection, recipeRequestedResolution } from "../../recipeFields.js";
import { restoreVedaSettings } from "../../vedaControls.js";

// Extracted from VideoStudio along its existing concern boundary.
export function useVideoRecipeReplay({
  launchRequest,
  setRecipeTierRequest,
  setExpectedImportedModelRef,
  setReplayReferenceResize,
  setSelectedPresetId,
  setMode,
  macVideoModels,
  setModel,
  setRecipeModelNotice,
  setStyleId,
  setPrompt,
  setNegativePrompt,
  setSeed,
  setResolution,
  setDuration,
  setFps,
  setQuality,
  setSelectedLoraIds,
  setLoraWeights,
  setVeda,
  setStepsOverride,
  setGuidanceOverride,
  setSampler,
  setScheduler,
  setSchedulerShift,
  setMotion,
  setLtxPipeline,
  setDistilledVariant,
  setTransformerVariant,
  setVaeDecoder,
  setAutoDuration,
  setAutoDurationMinSeconds,
  setAutoDurationMaxSeconds,
  setTemporalUpsampleRounds,
  setPrecision,
  setEnhancePrompt,
  setTextEncoderModel,
  model,
  setQuantization,
  setLightning,
  setLtxVideoCfg,
  setLtxVideoStg,
  setLtxVideoRescale,
  setVideoConditioningStrength,
  setBridgeRightVideoConditioningStrength,
  videoModels,
  setSourceAssetOrigin,
  setSourceAssetId,
  setLastFrameAssetId,
  setSourceClipAssetId,
  setBridgeRightClipAssetId,
  setSourceClipAssetIds,
  setReferenceAssetIds,
  setReferenceAudioAssetIds,
  setReferenceClipAssetId,
  setFitMode,
  setCharacterId,
  setCharacterLookId,
  setPersonTrackId,
  setReplacementMode,
  lastI2vAssetIdRef
}) {
  // sc-12324: replay a recorded recipe — the viewer's "Use this recipe" on a video asset. Restores
  // the form field-for-field so a clip can be re-run with or without its original seed.
  //
  // Its OWN effect, keyed on the launch id ALONE, mirroring ImageStudio. The launch effect above
  // also depends on the selected asset, so folding this into it would re-apply the recipe on every
  // selection change and silently overwrite whatever the user had since edited.
  //
  // Absent values reset to their defaults rather than keeping the current form's — a replay must
  // reproduce the recipe, not a hybrid of the recipe and whatever was on screen.
  useEffect(() => {
    if (launchRequest?.view !== "Video" || !launchRequest.recipe) {
      setRecipeTierRequest(null);
      // A non-recipe launch is a new request, not a continuation of the last replay: carrying its
      // original imported identity onto a different selection would be refused as a mismatch.
      if (launchRequest?.view === "Video") {
        setExpectedImportedModelRef(null);
        setReplayReferenceResize(null);
      }
      return;
    }
    const recipe = launchRequest.recipe;
    setExpectedImportedModelRef(recipe.importedModelRef ?? null);
    // How the recorded H3 clip encoded its references (`diffusers` for a pre-sc-26691 asset).
    setReplayReferenceResize(recipeReferenceResize(recipe));
    const settings = recipe.normalizedSettings ?? {};
    const rawSettings = recipe.rawAdapterSettings ?? {};
    const { loraIds, loraWeights: recipeWeights } = recipeLoraSelection(recipe);
    // Fold a mode the tabs no longer expose back to text_to_video, so a replay never lands on a
    // missing tab (the image lane's normalizeImageMode does the same).
    const nextMode = VIDEO_MODES.includes(recipe.mode) ? recipe.mode : "text_to_video";
    const sourceAssetIdFromRecipe = settings.sourceAssetId ?? "";
    const lastFrameAssetIdFromRecipe = settings.lastFrameAssetId ?? "";

    // A recipe and a preset are mutually exclusive: the recipe already IS the settings, and a
    // lingering preset's hydrate pass would layer its defaults back over them.
    setSelectedPresetId(noPresetId);
    setMode(nextMode);
    // An uninstalled model is filtered out of the catalog upstream, so setting it would leave the
    // picker on a phantom id; the mode-snap effect then moves to a model that serves the mode. Say
    // so rather than letting the swap look like the recipe's own choice.
    const recipeModelAvailable =
      !recipe.model || macVideoModels.some((item) => item.id === recipe.model);
    if (recipe.model && recipeModelAvailable) {
      setModel(recipe.model);
    }
    setRecipeModelNotice(recipeModelAvailable ? "" : (recipe.model ?? ""));

    // Style Catalog round-trip (sc-13136): re-select the picker to the recorded style id ONLY when
    // the raw pre-style prompt was also recorded, and seed the box with THAT raw prompt (not the
    // composed recipe.prompt) so the next submit recomposes the identical Subject:/Style: prompt
    // with no double-wrap. A styleless recipe clears any stale selection and uses recipe.prompt.
    const restoredStyleId = rawSettings.styleId ?? null;
    const hasRawStylePrompt = restoredStyleId != null && typeof rawSettings.stylePrompt === "string";
    setStyleId(hasRawStylePrompt ? restoredStyleId : null);
    setPrompt(hasRawStylePrompt ? rawSettings.stylePrompt : String(recipe.prompt ?? ""));
    setNegativePrompt(String(recipe.negativePrompt ?? ""));
    // Seed stays random by default, so "Use this recipe" makes a close variation; the viewer's
    // "Keep seed" resolves replaySeed to THIS clip's own seed for an exact rerun. Guard with
    // `!= null` so a legitimate seed of 0 replays instead of reading as absent.
    const replaySeed = launchRequest.replaySeed;
    setSeed(replaySeed != null && replaySeed !== "" ? String(replaySeed) : "");

    // The resolution the user PICKED, not the resolved dims — see recipeRequestedResolution. The
    // model-limits snap re-checks it against the model's options, so a stale one still snaps.
    const resolutionFromRecipe = recipeRequestedResolution(recipe);
    if (resolutionFromRecipe) {
      setResolution(resolutionFromRecipe);
    }
    const durationValue = finiteRecipeNumber(settings.duration);
    if (durationValue) {
      setDuration(durationValue);
    }
    const fpsValue = finiteRecipeNumber(settings.fps);
    if (fpsValue) {
      setFps(fpsValue);
    }
    if (settings.quality) {
      setQuality(settings.quality);
    }
    setSelectedLoraIds(loraIds);
    setLoraWeights(recipeWeights);

    // `advanced` passthrough: every real *_raw_settings builder clones it, so these are the knobs
    // exactly as the client sent them. Steps/guidance record the OVERRIDE — blank means "no
    // override", which correctly replays as blank and lets the engine re-derive its own default.
    setVeda(restoreVedaSettings(rawSettings));
    setStepsOverride(rawSettings.steps ?? "");
    setGuidanceOverride(rawSettings.guidanceScale ?? "");
    setSampler(rawSettings.sampler ?? "default");
    setScheduler(rawSettings.scheduler ?? "default");
    setSchedulerShift(rawSettings.schedulerShift ?? 3.0);
    setMotion(rawSettings.motion ?? DEFAULT_MOTION);
    setLtxPipeline(rawSettings.ltxPipeline ?? "auto");
    setDistilledVariant(rawSettings.distilledVariant ?? "1.1");
    setTransformerVariant(rawSettings.transformerVariant ?? "distilled");
    setVaeDecoder(rawSettings.vaeDecoder === "diffusion" ? "diffusion" : "conv");
    setAutoDuration(rawSettings.autoDuration === true);
    setAutoDurationMinSeconds(
      clampLtx25AutoDurationSeconds(
        rawSettings.autoDurationMinSeconds,
        LTX25_AUTO_DURATION_MIN_SECONDS,
      ),
    );
    setAutoDurationMaxSeconds(
      clampLtx25AutoDurationSeconds(
        rawSettings.autoDurationMaxSeconds,
        LTX25_AUTO_DURATION_MAX_SECONDS,
      ),
    );
    setTemporalUpsampleRounds(
      clampLtx25TemporalUpsampleRounds(rawSettings.temporalUpsampleRounds),
    );
    setPrecision(rawSettings.precision ?? "fp8");
    setEnhancePrompt(rawSettings.enhancePrompt === true);
    setTextEncoderModel(
      rawSettings.textEncoderModel ??
        (rawSettings.useUncensoredEnhancer === true ? amoralTextEncoderId : null),
      recipe.model && recipeModelAvailable ? recipe.model : model,
    );
    setQuantization(rawSettings.quantization ?? "auto");
    setLightning(rawSettings.lightning ?? true);
    setLtxVideoCfg(rawSettings.videoCfgGuidanceScale ?? "");
    setLtxVideoStg(rawSettings.videoStgGuidanceScale ?? "");
    setLtxVideoRescale(rawSettings.videoRescaleScale ?? "");
    setVideoConditioningStrength(rawSettings.videoConditioningStrength ?? "");
    setBridgeRightVideoConditioningStrength(rawSettings.bridgeRightVideoConditioningStrength ?? "");

    // The exact native tier the clip was generated at. Record the request against the model this
    // replay will actually activate; the reactive validator above applies it only when that exact
    // tier is selectable on the active backend. It deliberately does NOT write picker state here:
    // same-model q4/q8 SCAIL Candle replays used to escape the narrowed tier set through this setter,
    // then serialize no mlxQuantize and silently run bf16.
    const recipeTier = quantizeTier(rawSettings.mlxQuantize);
    if (recipeTier) {
      const recipeTierModelId = recipe.model && recipeModelAvailable ? recipe.model : model;
      setRecipeTierRequest({
        modelId: recipeTierModelId,
        modelName:
          videoModels.find((item) => item.id === recipeTierModelId)?.name ?? recipeTierModelId,
        tier: recipeTier,
      });
    } else {
      setRecipeTierRequest(null);
    }

    // Sources. A deleted asset is left to the existing `hasInputs` + validation machinery to
    // surface — the same path a user hits by clearing a picker, rather than a bespoke gate.
    // A recipe's recorded source is the user's own earlier pick, restored verbatim.
    setSourceAssetOrigin(sourceAssetIdFromRecipe ? "explicit" : "none");
    setSourceAssetId(sourceAssetIdFromRecipe);
    setLastFrameAssetId(lastFrameAssetIdFromRecipe);
    setSourceClipAssetId(settings.sourceClipAssetId ?? "");
    setBridgeRightClipAssetId(settings.bridgeRightClipAssetId ?? "");
    setSourceClipAssetIds(Array.isArray(settings.sourceClipAssetIds) ? settings.sourceClipAssetIds : []);
    setReferenceAssetIds(Array.isArray(settings.referenceAssetIds) ? settings.referenceAssetIds : []);
    setReferenceAudioAssetIds(
      Array.isArray(settings.referenceAudioAssetIds) ? settings.referenceAudioAssetIds : [],
    );
    setReferenceClipAssetId(settings.referenceClipAssetId ?? "");
    setFitMode(settings.fitMode ?? "crop");
    setCharacterId(settings.characterId ?? "");
    setCharacterLookId(settings.characterLookId ?? "");
    setPersonTrackId(settings.personTrackId ?? "");
    setReplacementMode(settings.replacementMode ?? "face_only");

    // The recipe's resolution is authoritative, but seeding a source image would otherwise trip the
    // I2V aspect snap above and overwrite it. Point that snap's ref at the restored id so it reads
    // as "no change" — the same mechanism that preserves a restored snapshot's resolution.
    lastI2vAssetIdRef.current = sourceAssetIdFromRecipe || lastFrameAssetIdFromRecipe;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [launchRequest?.id]);
}
