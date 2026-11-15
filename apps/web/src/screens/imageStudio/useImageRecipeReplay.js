import { useEffect, useRef } from "react";
import { recipeToStudioPatch } from "./recipeToStudioPatch.js";
import { isFlux3ImageModel } from "../../modelEligibility.js";

export function useImageRecipeReplay({
  model,
  models,
  mode,
  referenceAssetIds,
  launchRequest,
  setRecipeTierRequest,
  setFlux3Layout,
  recipeMode,
  setOrderedCharacterReferences,
  setFlux3ReferenceSlots,
  mossEditTransition,
  setReferenceDimensionIntent,
  setWidthOverride,
  setHeightOverride,
  skipReferenceTuningReset,
  referenceTuningDeclaredArmed,
  setSelectedPresetId,
  generalStackIds,
  toggleGeneralPreset,
  setMode,
  advancedDefaultsModel,
  skipAdvancedDefaultsReset,
  setModel,
  setStyleId,
  setAppearance,
  setPrompt,
  setCaption,
  setPromptMode,
  setMagicPromptBackend,
  promptEdited,
  setNegativePrompt,
  setSeed,
  setCount,
  setResolution,
  setSelectedLoraIds,
  setLoraWeights,
  setMossImage,
  setMossImageEdit,
  setMossImageAdvanced,
  setMossPromptBatch,
  setMossDmdSteps,
  setStepsOverride,
  setGuidanceOverride,
  setSampler,
  setScheduler,
  setSchedulerShift,
  setCfgInterval,
  setCaptionOverflow,
  setBatchMode,
  setBatchPromptsText,
  setGuidanceMethod,
  setCharacterId,
  setCharacterLookId,
  setReferenceAssetId,
  setReferenceAssetIds,
  setIpAdapterScale,
  ipAdapterScale,
  setControlnetScale,
  controlnetScale,
  setTrueCfgScale,
  trueCfgScale,
  setViewAngle,
  setEnhancePrompt,
  setTextEncoderModel,
  setUsePid,
  setPidTarget,
  setDecoder,
  setFaceRestore,
  setImg2imgStrength,
  img2imgStrength,
  setTextStyleGain,
  textStyleGain,
  setWorkflowPoses,
  setSelectedPoseIds,
  setPoseLibraryReplayOpen,
  setControlMode,
  setControlScale,
  controlScale,
  setMultiPhasePhases,
  setMultiPhaseEnabled,
  setSourceAssetId,
  setFitMode,
  setUpscaleEnabled,
  setUpscaleFactor,
  handleUpscaleEngineChange,
  setUpscaleSoftness,
  setUpscaleRestore,
  setHiresFixEnabled,
  setHiresFixSteps,
  setHiresFixDenoisingStrength,
  setHiresFixUpscaleBy,
  setHiresFixCfgScale
}) {
  // A recipe whose model the catalog has not delivered yet keeps its FLUX.3-only restore here
  // until the model resolves (applied only if it resolves to a FLUX.3 image model).
  const pendingFlux3Recipe = useRef(null);
  // Live values the user can change after a replay starts; the deferred restore yields to them.
  const live = useRef({});
  live.current = { model, mode, referenceAssetIds };
  useEffect(() => {
    const pending = pendingFlux3Recipe.current;
    if (!pending || pending.stage === 2) return;
    // Stage 0 is the commit that applied the replay (state not yet updated); stage 1 is the first
    // commit that shows the replayed state, which is the baseline a later user edit is measured from.
    if (pending.stage === 0) { pending.stage = 1; return; }
    pending.baseline = { ...live.current };
    pending.stage = 2;
  });
  useEffect(() => {
    const patch = recipeToStudioPatch(launchRequest, { model, models, recipeMode,
      mossEditActive: mossEditTransition.current.active, advancedDefaultsModel: advancedDefaultsModel.current,
      ipAdapterScale, controlnetScale, trueCfgScale, img2imgStrength, textStyleGain, controlScale });
    pendingFlux3Recipe.current = patch?.flags.flux3Pending
      ? { launchRequest, initialValues: patch.values, stage: 0, baseline: null } : null;
    if (!patch) return;
    const { values, flags } = patch;
    mossEditTransition.current.replay = flags.mossEditReplay;
    skipReferenceTuningReset.current = flags.skipReferenceTuningReset;
    referenceTuningDeclaredArmed.current = flags.referenceTuningDeclaredArmed;
    if (flags.skipAdvancedDefaultsReset) skipAdvancedDefaultsReset.current = true;
    promptEdited.current = flags.promptEdited;
    if (flags.clearGeneralStack) for (const id of generalStackIds) toggleGeneralPreset(id);
    const setters = {
      recipeTierRequest: setRecipeTierRequest,
      flux3Layout: setFlux3Layout,
      orderedCharacterReferences: setOrderedCharacterReferences,
      flux3ReferenceSlots: setFlux3ReferenceSlots,
      referenceDimensionIntent: setReferenceDimensionIntent,
      widthOverride: setWidthOverride,
      heightOverride: setHeightOverride,
      selectedPresetId: setSelectedPresetId,
      mode: setMode,
      model: setModel,
      styleId: setStyleId,
      appearance: setAppearance,
      prompt: setPrompt,
      caption: setCaption,
      promptMode: setPromptMode,
      magicPromptBackend: setMagicPromptBackend,
      negativePrompt: setNegativePrompt,
      seed: setSeed,
      count: setCount,
      resolution: setResolution,
      selectedLoraIds: setSelectedLoraIds,
      loraWeights: setLoraWeights,
      mossImage: setMossImage,
      mossImageEdit: setMossImageEdit,
      mossImageAdvanced: setMossImageAdvanced,
      mossPromptBatch: setMossPromptBatch,
      mossDmdSteps: setMossDmdSteps,
      stepsOverride: setStepsOverride,
      guidanceOverride: setGuidanceOverride,
      sampler: setSampler,
      scheduler: setScheduler,
      schedulerShift: setSchedulerShift,
      cfgInterval: setCfgInterval,
      captionOverflow: setCaptionOverflow,
      batchMode: setBatchMode,
      batchPromptsText: setBatchPromptsText,
      guidanceMethod: setGuidanceMethod,
      characterId: setCharacterId,
      characterLookId: setCharacterLookId,
      referenceAssetId: setReferenceAssetId,
      referenceAssetIds: setReferenceAssetIds,
      ipAdapterScale: setIpAdapterScale,
      controlnetScale: setControlnetScale,
      trueCfgScale: setTrueCfgScale,
      viewAngle: setViewAngle,
      enhancePrompt: setEnhancePrompt,
      textEncoderModel: ({ value, modelId }) => setTextEncoderModel(value, modelId),
      usePid: setUsePid,
      pidTarget: setPidTarget,
      decoder: setDecoder,
      faceRestore: setFaceRestore,
      img2imgStrength: setImg2imgStrength,
      textStyleGain: setTextStyleGain,
      workflowPoses: setWorkflowPoses,
      selectedPoseIds: setSelectedPoseIds,
      poseLibraryReplayOpen: setPoseLibraryReplayOpen,
      controlMode: setControlMode,
      controlScale: setControlScale,
      multiPhasePhases: setMultiPhasePhases,
      multiPhaseEnabled: setMultiPhaseEnabled,
      sourceAssetId: setSourceAssetId,
      fitMode: setFitMode,
      upscaleEngine: handleUpscaleEngineChange,
      upscaleEnabled: setUpscaleEnabled,
      upscaleFactor: setUpscaleFactor,
      upscaleSoftness: setUpscaleSoftness,
      upscaleRestore: setUpscaleRestore,
      hiresFixEnabled: setHiresFixEnabled,
      hiresFixSteps: setHiresFixSteps,
      hiresFixDenoisingStrength: setHiresFixDenoisingStrength,
      hiresFixUpscaleBy: setHiresFixUpscaleBy,
      hiresFixCfgScale: setHiresFixCfgScale,
    };
    for (const [key, value] of Object.entries(values)) setters[key]?.(value);
  // The launch id owns one-shot hydration; local values must not replay the launch.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [launchRequest?.id]);

  useEffect(() => {
    const pending = pendingFlux3Recipe.current;
    if (!pending) return;
    const resolved = models.find((entry) => entry.id === pending.launchRequest.recipe?.model);
    if (!resolved) return;
    pendingFlux3Recipe.current = null;
    if (!isFlux3ImageModel(resolved)) return;
    // A user action after the replay started wins over the deferred restore.
    const { baseline } = pending;
    const userActed = baseline && (
      baseline.model !== live.current.model || baseline.mode !== live.current.mode ||
      JSON.stringify(baseline.referenceAssetIds) !== JSON.stringify(live.current.referenceAssetIds));
    if (userActed) return;
    // Re-derive through the same function the loaded-catalog path uses and apply exactly what the
    // resolved model changes, so late and loaded replays end in the same Studio state.
    const patch = recipeToStudioPatch(pending.launchRequest, { model, models, recipeMode,
      mossEditActive: mossEditTransition.current.active, advancedDefaultsModel: advancedDefaultsModel.current,
      ipAdapterScale, controlnetScale, trueCfgScale, img2imgStrength, textStyleGain, controlScale });
    if (!patch) return;
    const flux3Setters = {
      flux3ReferenceSlots: setFlux3ReferenceSlots,
      orderedCharacterReferences: setOrderedCharacterReferences,
      referenceAssetIds: setReferenceAssetIds,
      mode: setMode,
      referenceDimensionIntent: setReferenceDimensionIntent,
      widthOverride: setWidthOverride,
      heightOverride: setHeightOverride,
    };
    for (const [key, setter] of Object.entries(flux3Setters)) {
      if (!Object.hasOwn(patch.values, key)) continue;
      if (JSON.stringify(patch.values[key]) === JSON.stringify(pending.initialValues[key])) continue;
      setter(patch.values[key]);
    }
  // Runs when the catalog changes; the pending ref scopes it to the one launch awaiting a model.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);
}
