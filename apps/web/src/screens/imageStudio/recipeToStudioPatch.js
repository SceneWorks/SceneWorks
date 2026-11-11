import { emptyAppearance, normalizeAppearance } from "../../data/appearanceCatalog.js";
import { mossAdvancedDefaults } from "../../mossImageAdvanced.js";
import { isMossEdit, mossEditDefaults, mossDefaults } from "../../mossImage.js";
import { deserializePhases } from "../../imageMultiPhase.js";
import { reconcileLayoutReferences, recordedLayoutReferences } from "../../flux3Layout.js";
import { isFlux3ImageModel } from "../../modelEligibility.js";
import { orderCaption, validateCaption } from "../../ideogramCaption.js";
import { workflowPoseRecords } from "../../poseLibrary.js";
import { noPresetId } from "../../presetUtils.js";
import { quantizeTier } from "../../quantTier.js";
import { finiteRecipeNumber, recipeImageDimensionIntent, recipeLoraSelection, recipeResolution } from "../../recipeFields.js";
import { irisControlsFromRecord } from "../../irisRestore.js";

// Pure launch normalization: absence stays absent; repeated writes resolve in their
// original order (ordinary MOSS settings intentionally override generic knobs).
export function recipeToStudioPatch(launchRequest, { model, models = [], recipeMode, mossEditActive, advancedDefaultsModel, ipAdapterScale, controlnetScale, trueCfgScale, img2imgStrength, textStyleGain, controlScale }) {
  const values = {};
  const flags = {};

  if (launchRequest?.view !== "Image" || !launchRequest.recipe) {
    return null;
  }
  const recipe = launchRequest.recipe;
  const recipeModel = models.find((entry) => entry.id === recipe.model);
  const flux3Recipe = isFlux3ImageModel(recipeModel);
  if (recipe.uiPreserveComposedPrompt === true) flags.clearGeneralStack = true;
  const settings = recipe.normalizedSettings ?? {};
  const rawSettings = recipe.rawAdapterSettings ?? {};
  const replayPromptBatch = typeof rawSettings.appearancePrompt === "string" && Array.isArray(rawSettings.appearancePromptBatch) ? rawSettings.appearancePromptBatch : rawSettings.promptBatch;
  const restoredTier = rawSettings.quantTier ??
    (rawSettings.convRot === true ? "int8-convrot" : quantizeTier(rawSettings.mlxQuantize));
  values.recipeTierRequest = restoredTier ? { model: recipe.model, tier: restoredTier } : null;
  // Restore the authoring document, never the worker's rounded/locked caption table.
  values.flux3Layout = rawSettings.flux3_layout ?? rawSettings.flux3Layout?.authoring ?? { version: 1, boxes: [] };
  const recordedReferences = flux3Recipe ? recordedLayoutReferences(rawSettings) : null;
  const orderedReferences = rawSettings.referenceAssetIds ?? recordedReferences?.assetIds;
  // Reference-conditioned text uses the existing edit UI. Character jobs retain their actual
  // identity/mode while the existing plural reference carrier preserves every attachment.
  const nextMode = flux3Recipe && orderedReferences?.length && recipeMode(recipe) === "text_to_image"
    ? "edit_image" : recipeMode(recipe);
  const restoredReferenceAssets = orderedReferences ?? (nextMode === "edit_image"
    ? [rawSettings.referenceAssetId ?? launchRequest.sourceAssetId ?? launchRequest.assetId ?? settings.sourceAssetId].filter(Boolean)
    : [rawSettings.referenceAssetId ?? launchRequest.referenceAssetId].filter(Boolean));
  values.orderedCharacterReferences = flux3Recipe && nextMode === "character_image" && Array.isArray(orderedReferences);
  values.flux3ReferenceSlots = reconcileLayoutReferences([], restoredReferenceAssets,
    rawSettings.flux3_reference_ids ?? recordedReferences?.referenceIds);
  const dimensionIntent = recipeImageDimensionIntent(recipe, recipeModel);
  const replayMossEdit = isMossEdit(recipe.model ?? model, nextMode);
  flags.mossEditReplay = replayMossEdit !== mossEditActive;
  const resolutionFromRecipe = recipeResolution(recipe);
  values.referenceDimensionIntent = dimensionIntent
    ? { ...dimensionIntent, model: recipe.model, mode: nextMode } : null;
  // A launch replaces stale custom sides, including when either requested axis was omitted.
  values.widthOverride = dimensionIntent?.width ?? "";
  values.heightOverride = dimensionIntent?.height ?? "";
  const { loraIds, loraWeights: loraWeightMap } = recipeLoraSelection(recipe);

  // Arm the model-change reset skip only when this recipe actually changes the model. A
  // same-model recipe never triggers that effect; leaving `true` behind would make the NEXT
  // manual model switch consume a stale skip and retain this recipe's transient pose records.
  flags.skipReferenceTuningReset = Boolean(recipe.model && recipe.model !== model);
  // A recipe injects its own reference tuning below (setIpAdapterScale/…), so disarm the
  // fresh-mount declared-defaults resolver (sc-12034) — it must not overwrite the recipe values
  // once the catalog resolves, even on a late-catalog mount where `skip*` was already consumed.
  flags.referenceTuningDeclaredArmed = false;
  values.selectedPresetId = noPresetId;
  // Native batch rows already contain the effective preset/style text captured by the worker.
  if (replayMossEdit && rawSettings.mossImageAdvanced && replayPromptBatch?.length) {
    flags.clearGeneralStack = true;
  }
  values.mode = nextMode;
  if (recipe.model) {
    if (recipe.model !== advancedDefaultsModel) {
      flags.skipAdvancedDefaultsReset = true;
    }
    values.model = recipe.model;
  }
  // Structured-prompt round-trip (sc-6147): a structured model's recipe carries
  // the full caption under rawAdapterSettings.structuredPrompt. Rehydrate the
  // builder (caption + intent + magic-prompt backend) instead of dropping the
  // serialized JSON into the plain prompt box. Falls back to the plain `prompt`
  // path when the blob is absent/invalid (older assets, non-structured models).
  const structuredRecipe = rawSettings.structuredPrompt;
  const restoredCaption = structuredRecipe?.caption ?? null;
  // Style Catalog round-trip (sc-13132): re-select the Style picker to the recorded style id (a
  // group id or a sub-style id) — but ONLY when the raw pre-style prompt was also recorded, so
  // submit can recompose from it. A styleless recipe, or a partial one carrying a styleId with
  // no stylePrompt, clears any stale selection so its already-composed recipe.prompt is never
  // re-wrapped. Since sc-13224 (structured captions ARE styled), these branches CAN overlap for a
  // structured styled recipe: both restoredCaption and hasRawStylePrompt may be truthy. That is
  // correct — restoredStyleId is restored just below, and the structured-caption branch takes
  // precedence, restoring the PRE-injection caption so submit re-injects the style exactly once.
  const restoredStyleId = rawSettings.styleId ?? null;
  const hasRawStylePrompt =
    restoredStyleId != null && typeof rawSettings.stylePrompt === "string";
  values.styleId = hasRawStylePrompt ? restoredStyleId : null;
  const hasRawAppearance = rawSettings.appearance && typeof rawSettings.appearancePrompt === "string";
  values.appearance = hasRawAppearance ? normalizeAppearance(rawSettings.appearance) : emptyAppearance();
  if (hasRawAppearance) {
    values.styleId = restoredStyleId;
    // Appearance carriers already contain the preset fold. Replaying them must not fold
    // whatever general presets happen to be active in the destination Studio a second time.
    flags.clearGeneralStack = true;
  }
  if (replayMossEdit && rawSettings.mossImageAdvanced && Array.isArray(replayPromptBatch) && replayPromptBatch.length) {
    values.prompt = replayPromptBatch[0];
  } else if (restoredCaption && validateCaption(restoredCaption).ok) {
    values.caption = orderCaption(restoredCaption);
    values.promptMode = "form";
    values.magicPromptBackend = structuredRecipe.magicPromptBackend ?? null;
    // The intent (original idea) seeds the plain box; the serialized caption is
    // authoritative for generation and is rebuilt from `caption` on submit.
    values.prompt = String(structuredRecipe.intent ?? "");
  } else if (hasRawAppearance) {
    values.prompt = rawSettings.appearancePrompt;
  } else if (hasRawStylePrompt) {
    // Styled recipe (sc-13132): seed the box with the RAW pre-style prompt, NOT the composed
    // `recipe.prompt`. With the picker re-selected above, submit recomposes the identical
    // `Subject:`/`Style:` prompt — recording the raw prompt is what prevents a double-wrap
    // (composing over the already-composed prompt would nest a second `Style:` block).
    values.prompt = rawSettings.stylePrompt;
  } else {
    values.prompt = String(recipe.prompt ?? "");
  }
  flags.promptEdited = true;
  values.negativePrompt = String(hasRawAppearance ? rawSettings.appearanceNegativePrompt ?? "" : recipe.negativePrompt ?? "");
  // Recipe replay leaves Seed random by default so "Use this recipe" makes a close
  // variation instead of a byte-for-byte rerun. When the launcher passes a replaySeed
  // (viewer "Keep seed" toggle), the launcher resolves this image's own seed or
  // the original base when replaying its whole MOSS batch. Seed 0 is valid.
  const replaySeed = launchRequest.replaySeed;
  values.seed = replaySeed != null && replaySeed !== "" ? String(replaySeed) : "";
  const countValue = finiteRecipeNumber(settings.count);
  if (countValue) {
    values.count = countValue;
  }
  if (resolutionFromRecipe) {
    values.resolution = resolutionFromRecipe;
  } else if (replayMossEdit) {
    values.resolution = "512x512";
  }
  if (replayMossEdit) { values.widthOverride = settings.width ?? ""; values.heightOverride = settings.height ?? ""; }
  values.selectedLoraIds = loraIds;
  values.loraWeights = loraWeightMap;
  values.mossImage = rawSettings.mossImage ?? mossDefaults();
  values.mossImageEdit = rawSettings.mossImageEdit ?? mossEditDefaults();
  values.mossImageAdvanced = rawSettings.mossImageAdvanced ?? mossAdvancedDefaults();
  values.mossPromptBatch = replayMossEdit && rawSettings.mossImageAdvanced ? replayPromptBatch ?? [] : [];
  values.mossDmdSteps = rawSettings.mossImage?.acceleration && rawSettings.mossImage.acceleration !== "off" ? rawSettings.steps ?? 8 : 8;
  values.stepsOverride = replayMossEdit ? rawSettings.steps ?? 30 : rawSettings.mossImage?.acceleration && rawSettings.mossImage.acceleration !== "off" ? "" : rawSettings.steps ?? rawSettings.numInferenceSteps ?? "";
  values.guidanceOverride = rawSettings.guidanceScale ?? (replayMossEdit ? 4 : "");
  values.sampler = rawSettings.sampler ?? "default";
  values.scheduler = rawSettings.scheduler ?? "default";
  values.schedulerShift = rawSettings.schedulerShift ?? rawSettings.timestepShift ?? (replayMossEdit ? 5 : null);
  // sc-25681: the recorded CFG interval / caption-overflow policy (absent = the model's own).
  values.cfgInterval = rawSettings.cfgInterval?.join?.(", ") ?? "";
  values.captionOverflow = rawSettings.captionOverflow ?? "default";
  // sc-25681: an image of a native prompt batch reopens the whole batch.
  const batch = replayPromptBatch;
  const replayBatch = !(replayMossEdit && rawSettings.mossImageAdvanced) && Array.isArray(batch) && batch.length > 1;
  values.batchMode = replayBatch;
  if (replayBatch) {
    values.batchPromptsText = batch.join(batch.some((p) => p.includes("\n")) ? "\n---\n" : "\n");
  }
  values.guidanceMethod = rawSettings.guidanceMethod ?? "cfg";
  if (rawSettings.mossOrdinarySettings) {
    const ordinary = rawSettings.mossOrdinarySettings;
    values.mossImage = { ...ordinary.options, acceleration: rawSettings.mossImage?.acceleration ?? "off" };
    values.stepsOverride = ordinary.steps ?? ""; values.guidanceOverride = ordinary.guidanceScale ?? ""; values.negativePrompt = ordinary.negativePrompt ?? ""; values.schedulerShift = ordinary.schedulerShift ?? null;
  }
  values.characterId = settings.characterId ?? "";
  values.characterLookId = settings.characterLookId ?? "";
  values.referenceAssetId = settings.referenceAssetId ?? rawSettings.referenceAssetId ?? recordedReferences?.assetIds[0] ?? launchRequest.referenceAssetId ?? "";
  // sc-15952 / sc-26687: the ordered reference images. Guarded on the KEY rather than assigned
  // unconditionally — a recipe that records neither spelling must not clear a selection the user
  // had already made, while an explicit empty list is authoritative and does clear it.
  //
  // `referenceAssetIds` is a TOP-LEVEL request key, not part of `advanced`, so it never reached
  // `rawAdapterSettings` and this reader could never see it: "Use this recipe" silently rebuilt an
  // edit with the ordered multi-reference conditioning dropped. The recipe now records the list in
  // `normalizedSettings` — the replay record — and that spelling is authoritative. The `advanced`
  // spelling is kept as the fallback for a shared-workflow envelope that carries it there.
  const recordedReferenceIds =
    Array.isArray(settings.referenceAssetIds)
      ? settings.referenceAssetIds
      : Array.isArray(rawSettings.referenceAssetIds)
        ? rawSettings.referenceAssetIds
        : null;
  // FLUX.3 layout recipes restore their recorded layout references (main, sc-26233).
  if (flux3Recipe) {
    values.referenceAssetIds = restoredReferenceAssets.filter(Boolean);
  } else if (isMossEdit(recipe.model ?? model, nextMode)) {
    // Missing identities stay visible and block submission; never drop a slot or use the output.
    values.referenceAssetIds = recordedReferenceIds ?? [];
  } else if (recordedReferenceIds) {
    values.referenceAssetIds = recordedReferenceIds.filter(Boolean);
  }
  values.ipAdapterScale = rawSettings.ipAdapterScale ?? settings.ipAdapterScale ?? ipAdapterScale;
  values.controlnetScale = rawSettings.controlnetConditioningScale ?? rawSettings.controlnetScale ?? settings.controlnetScale ?? controlnetScale;
  values.trueCfgScale = rawSettings.trueCfgScale ?? settings.trueCfgScale ?? trueCfgScale;
  values.viewAngle = rawSettings.viewAngle ?? settings.viewAngle ?? "";
  // The rest of the allow-listed knobs (sc-15951). Every one of these travels in a shared
  // image's envelope AND is recorded on a generated asset's recipe, so leaving them out made
  // both replay paths quietly produce a different image — the recipe said "PiD decoder on,
  // control strength 0.9" and the studio rendered with neither.
  //
  // ORDERING. `setControlMode` / `setControlScale` / `setTextStyleGain` are also written by the
  // model-change reset above (and by its late-catalog twin), which `setModel` a few lines up
  // will trigger on the NEXT commit. Both are disarmed for this path by the `skip*` /
  // `referenceTuningDeclaredArmed` flags set at the top of this effect, which is what lets the
  // assignments here stand — see the ordering test in `ImageStudio.recipeKnobs.test.jsx`, which
  // replays a control recipe onto a DIFFERENT model and asserts the recipe's values survive the
  // reset rather than the model's defaults.
  //
  // The booleans hydrate from `=== true` rather than falling back to their sticky values: each
  // is emitted by `buildImageJobAdvanced` only when it is ON, so an absent key means the run had
  // it off, and inheriting the user's current toggle would replay someone else's recipe with a
  // decoder they never used. The numbers keep the current value when absent, matching the
  // tuning knobs above.
  //
  // `replayNumber` keeps the current value for an ABSENT knob rather than for a zero one:
  // `finiteRecipeNumber(null)` is 0 (`Number(null) === 0`), so reading these through it alone
  // would turn "the run recorded nothing" into "the run recorded zero" — a 0.0 control strength
  // is a real, very different setting.
  const replayNumber = (value, fallback) =>
    value === null || value === undefined || value === ""
      ? fallback
      : (finiteRecipeNumber(value) ?? fallback);
  values.enhancePrompt = rawSettings.enhancePrompt === true;
  values.textEncoderModel = { value: rawSettings.textEncoderModel ?? null, modelId: recipe.model ?? model };
  values.usePid = rawSettings.usePid === true;
  values.pidTarget = rawSettings.pidTarget === "2k" ? "2k" : "4k";
  values.decoder = typeof rawSettings.decoder === "string" ? rawSettings.decoder : "native";
  values.faceRestore = rawSettings.faceRestore === true;
  values.img2imgStrength = replayNumber(rawSettings.strength, img2imgStrength);
  values.textStyleGain = replayNumber(rawSettings.textStyleGain, textStyleGain);
  const replayPoses = workflowPoseRecords(rawSettings.poses, launchRequest.id);
  values.workflowPoses = replayPoses;
  values.selectedPoseIds = replayPoses.map((pose) => pose.id);
  values.poseLibraryReplayOpen = replayPoses.length > 0;
  if (replayPoses.length) {
    // A shared pose selection is strict pose conditioning even when controlMode was omitted
    // (pose is the worker default). Never strand replayed poses behind canny or depth.
    values.controlMode = "pose";
  } else if (rawSettings.controlMode) {
    values.controlMode = rawSettings.controlMode;
  }
  values.controlScale = replayNumber(rawSettings.controlScale, controlScale);
  // Multi-phase denoise (epic 13879 S5). The recorded list is index-keyed against the job's own
  // LoRA stack; the editor is id-keyed, so `deserializePhases` resolves it against the ids this
  // launch is selecting in the very same commit. Absent → the editor is cleared AND switched
  // off, because a stale schedule left enabled would silently re-shape the next render.
  const restoredPhases = deserializePhases(rawSettings.phases ?? [], loraIds);
  values.multiPhasePhases = restoredPhases;
  values.multiPhaseEnabled = restoredPhases.length > 0;
  if (nextMode === "edit_image") {
    values.sourceAssetId = restoredReferenceAssets[0] ?? launchRequest.sourceAssetId ?? launchRequest.assetId ?? settings.sourceAssetId ?? "";
    values.fitMode = rawSettings.fitMode ?? settings.fitMode ?? "crop";
  }
  const upscale = rawSettings.upscale ?? settings.upscale;
  values.upscaleEnabled = Boolean(upscale?.enabled);
  if (upscale?.factor) {
    values.upscaleFactor = upscale.factor;
  }
  if (upscale?.engine) {
    values.upscaleEngine = upscale.engine;
  }
  if (typeof upscale?.softness === "number") {
    values.upscaleSoftness = upscale.softness;
  }
  // Reopen an Iris restoration's applied controls (sc-25683).
  const irisControls = irisControlsFromRecord(upscale);
  if (irisControls) {
    values.upscaleRestore = irisControls;
  }
  const hiresFix = launchRequest.hiresFix ?? rawSettings.hiresFix ?? settings.hiresFix;
  values.hiresFixEnabled = Boolean(hiresFix?.enabled);
  values.hiresFixSteps = hiresFix?.steps ?? 0;
  values.hiresFixDenoisingStrength = hiresFix?.denoisingStrength ?? 0.7;
  values.hiresFixUpscaleBy = hiresFix?.upscaleBy ?? 2;
  values.hiresFixCfgScale = hiresFix?.cfgScale ?? "";

  return { values, flags };
}
