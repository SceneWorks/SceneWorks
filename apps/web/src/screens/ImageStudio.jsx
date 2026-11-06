import { useAppearance } from "../hooks/useAppearance.js";
import { composeNegativePrompt } from "../data/appearanceCatalog.js";
import { useMediaTicket } from "../hooks/useMediaTicket.js";
import { useStudioAdvancedSettings } from "../hooks/useStudioAdvancedSettings.js";
import { useStudioPromptGuide } from "../hooks/useStudioPromptGuide.js";
import { StudioPromptGuide, CharacterLookFields } from "../components/StudioFields.jsx";
import { useSamplerControls } from "../hooks/useSamplerControls.js";
import { preferredOption, preferredResolution, formatResolutionLabel } from "../studioDefaults.js";
import { ImageAdvancedPanel } from "./imageStudio/ImageAdvancedPanel.jsx";
import { ImagePromptComposer } from "./imageStudio/ImagePromptComposer.jsx";
import { ImageGenerationSettings } from "./imageStudio/ImageGenerationSettings.jsx";
import { ImageReviewPanel } from "./imageStudio/ImageReviewPanel.jsx";
import { useStructuredPrompt } from "./imageStudio/useStructuredPrompt.js";
import { useMossImageTransition } from "./imageStudio/useMossImageTransition.js";
import { useMossImageState } from "./imageStudio/useMossImageState.js";
import { useImageRecipeReplay } from "./imageStudio/useImageRecipeReplay.js";
import { useImageBatchRun } from "./imageStudio/useImageBatchRun.js";

import { mossAdvancedIssue, mossPositivePromptIssue } from "../mossImageAdvancedValidation.js";
import { isMossImage, mossDefaults, mossWireOptions, mossPresetWithOrdinarySettings, mossEditWireOptions, mossEditReferenceIds, mossEditControlsError } from "../mossImage.js";
import { MossImageEditControls } from "../components/MossImageEditControls.jsx";
import { MossReferencePicker } from "../components/MossReferencePicker.jsx";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AssetPickerField, ImageEditSourcePickerField } from "../components/AssetPicker.jsx";

import { AssetMedia } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";

import { MultiPhaseEditor } from "../components/MultiPhaseEditor.jsx";
import { buildTurboFinishPhases, modelSupportsMultiPhase, multiPhaseIssues as computeMultiPhaseIssues, serializePhases } from "../imageMultiPhase.js";
import { WorkPanel } from "../components/WorkPanel.jsx";

import { PoseLibraryPicker } from "../components/PoseLibraryPicker.jsx";

import { linkedGroupIssues, missingKeys, parsePromptResolution } from "../promptBatch.js";
import {
  dimensionConstraintMessage,
  evaluateModelDimensions,
  isFlux3ReferenceRequest,
} from "../resolutionOverride.js";
// sc-24113. `minStepsForModel` lives in videoModelLimits.js only because video declared
// `limits.hardMinSteps` first; the reader is model-agnostic and reading it here is what stops this
// screen from offering a step count the enqueue gate will refuse.

import { OrderedReferenceList } from "../components/OrderedReferenceList.jsx";

import { isFlux3LayoutModel, orderedLayoutAssetIds, reconcileLayoutReferences, moveLayoutReference, bindLayoutReferenceIds } from "../flux3Layout.js";
import { maxReferencesForModel } from "../imageReferenceLimits.js";
import { pidDecodeHeadsUp } from "../pidDecodeNotice.js";
import { promptEnhancementAvailable } from "../promptEnhancement.js";

import {
  DEFAULT_SCENE_PROMPT,
  promptHintFor,
  promptSeedFor,
  seedsNegativeInMode,
} from "../promptSeed.js";
import { emptyCaption, injectStyleIntoCaption, serializeCaption, validateCaption } from "../ideogramCaption.js";
import { buildImageJobRequest, composeImageJobPrompt, imageJobReferences } from "../imageJobRequest.js";
import { reconcileDecoderSelection } from "../imageDecoderSelection.js";
import { usePoseLibrary, useUserPoseLoader, poseRecordToPayload, WORKFLOW_POSE_CATEGORY } from "../poseLibrary.js";

const PROMPT_SUGGESTION_POOL = [
  "Barista pouring espresso, morning light",
  "Runner cresting a dune at dawn",
  "Dewdrop on a fern, soft bokeh",
  "Watchmaker at her bench, warm tungsten",
  "Cyclist on a wet cobblestone street, neon reflections",
  "Cellist mid-bow, theater spotlight from above",
  "Glassblower shaping a vessel, kiln glow",
  "Fox watching from the edge of a snowy forest",
  "Surfer at golden hour, backlit spray",
  "Quiet kitchen window, herbs in low afternoon light",
  "Vintage typewriter on a roll-top desk, dust motes",
  "Lighthouse beam slicing through coastal fog",
];

// Character (IP-Adapter) variations: the reference image supplies identity, so
// these describe scene / pose / lighting to vary rather than a standalone subject.
const CHARACTER_SUGGESTION_POOL = [
  "studio portrait, soft key light",
  "in a sunlit park, candid",
  "city street at dusk, cinematic",
  "seated at a wooden desk, warm light",
  "walking through a busy market, natural light",
  "close-up, dramatic side lighting",
  "outdoors at golden hour, backlit",
  "neutral grey backdrop, even studio lighting",
];

function pickSuggestions(count, pool = PROMPT_SUGGESTION_POOL) {
  const available = [...pool];
  const result = [];
  for (let index = 0; index < count && available.length; index += 1) {
    const pick = Math.floor(Math.random() * available.length);
    result.push(available.splice(pick, 1)[0]);
  }
  return result;
}

// Seeded into the prompt when entering character mode (only when untouched). The
// character's own notes win if present; otherwise a neutral, type-appropriate
// variation prompt — identity still comes from the reference image, not this text.
function defaultCharacterPrompt(character) {
  const note = (character?.description ?? "").trim();
  if (note) {
    return note;
  }
  switch (character?.type) {
    case "creature":
      return "The creature in a new setting, varied pose, natural lighting";
    case "object":
      return "The object from a fresh angle and setting, studio lighting";
    default:
      return "Portrait of the character, varied pose and expression, natural lighting";
  }
}
import { finiteNumberOrUndefined, findModelEditLora, loraIsInstalled, serializeLora, composePreset } from "../presetUtils.js";
import { PresetStackPreview, useGenerationStudio, useQuantTierPicker, useSavePreset } from "../components/generationStudio.jsx";
import { useAppContext } from "../context/AppContext.js";
import { useBatchPromptState } from "./imageStudio/useBatchPromptState.js";
import { resolveCfgInterval } from "./imageStudio/samplingControls.js";
import { usePidState, useStrictControlState } from "./imageStudio/useImageControlState.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { imageBatchValidation, imageGenerateValidation } from "../imageStudioValidation.js";
import { useValidation } from "../validation/useValidation.js";
import { ValidationSummary } from "../validation/Validation.jsx";
import {
  downloadOffersFor,
  imageModelUsable,
  supportedControlModes,
  visionCaptionModelUsable,
} from "../modelEligibility.js";
import { ControlPanel } from "../components/ControlPanel.jsx";
import { pidToggleVisible } from "../pidEligibility.js";
import { allPossibleTiers, installedTiers, isBelowFloor, modelQualityFloor, nativeTierEligibility, tierPickerOptions } from "../quantTier.js";
import { suggestTier } from "../tierSuggestion.js";
import { useHostMemory } from "../hooks/useHostMemory.js";
import { hostMemoryGbForBackend } from "../hostMemory.js";
import { MAX_LISTED_JOBS, PROMPT_REFINE_MODEL_ID, qwenRewriterAppliesTo, QWEN_REWRITE_I2I_MODEL_ID, QWEN_REWRITE_T2I_MODEL_ID, VISION_CAPTION_MODEL_ID } from "../constants.js";
import { parseResolution, pickClosestResolution } from "../resolutionMatch.js";
import { fitsResolutionOptions } from "../resolutionMemory.js";

import {
  DEFAULT_MAC_CAPABILITIES,
  macAvailableModels,
  macGatingActive,
  macModelFeatureBlock,
} from "../macGating.js";
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import { resolutionUniverseKey, useReferenceAspectSnap } from "../hooks/useReferenceAspectSnap.js";
import { resolveJobResultAssets } from "../jobResultAssets.js";
import { availableUpscaleEngines as upscaleEnginesForPlatform, upscaleEngineIsRestoration, upscaleFactorsForEngine, useUpscaleEngineFallback } from "../upscaleEngines.js";
import { IRIS_RESTORE_DEFAULTS, irisRestoreControls, restorePlanBodyForRenders } from "../irisRestore.js";

import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { StyledPromptPreview } from "../components/StyledPromptPreview.jsx";
import { styleHintForId, styleTextForId } from "../data/styleCatalog.js";
import { guidanceMethodDefaultFromModel, guidanceMethodOptionsFromModel, negativePromptInertAtGuidance, guidanceIsOff, patchGridHint, nativeResolutionHint, promptTokenLimitHint, schedulerShiftDefaultFromModel } from "../samplerOptions.js";

// Used only for models that don't declare limits.resolutions (e.g. user-imported).
const DEFAULT_RESOLUTION_OPTIONS = ["768x768", "1024x1024", "1280x720", "720x1280"];
// Screen identity for the per-(screen, model) sticky quant-tier store (sc-10727). Matches this
// studio's `loadStudioSettings`/`useStudioSettingsWriter` key so the sticky namespace is stable and
// distinct from Video/Character studios. Change this and existing users lose their Image sticky.
const TIER_SCREEN = "image";
// Studio sub-modes a saved preset may restore (the "type") — the tabs the mode
// segmented control actually exposes. Edit lives in its own workflow; text and
// character share the text_to_image workflow.
const IMAGE_MODES = ["text_to_image", "edit_image", "character_image"];

// sc-12034: the reference-tuning knobs the fresh-mount declared-defaults resolver owns (see the
// resolver + the recipe disarm below). A recipe or a saved preset that injects any of these must
// disarm that resolver so its injected values survive the async catalog arrival instead of being
// overwritten by the selected model's DECLARED defaults once the catalog resolves.
const REFERENCE_TUNING_PRESET_KEYS = [
  "ipAdapterScale",
  "controlnetScale",
  "trueCfgScale",
  "controlScale",
  "textStyleGain",
];

// Above this many resolved images a batch run needs explicit confirmation, so a stray
// value or an over-eager cross-product can't silently queue a huge job (sc-9957).
// This is a CONFIRMATION threshold only: it never caps how many images a run may
// resolve — an over-cap run just gets one "N images?" gate before it starts.
const BATCH_RENDER_CAP = 100;
// How many of a confirmed run's jobs may be in flight at once — the submit window
// `waitForBatchSlot` holds the fan-out behind. Distinct from BATCH_RENDER_CAP: when the
// two were one constant, confirming a 250-image run queued 100 jobs and then held the
// remaining 150 until the first 100 finished, because the window could only refill as
// jobs settled. The window only bounds the client's in-flight id set (the worker stays
// serial either way), so it is the queue depth the app can hold and show — a run at or
// under it queues in full the moment it is confirmed, and a larger run feeds in windows
// of that size.
const BATCH_ACTIVE_JOB_LIMIT = MAX_LISTED_JOBS;
// A styled Cartesian batch can be enormous even though its queue state remains bounded.
// Yield after this many preflight prompts so the run state commits and Stop can interrupt
// a clean (no-overage) stream before we ever reach its Cartesian suffix.
const BATCH_PREFLIGHT_YIELD_INTERVAL = BATCH_RENDER_CAP;

// Fold any mode the tabs no longer expose (a legacy `style_variations` snapshot,
// an unknown value) back to text_to_image so a restored studio never lands on a
// missing tab. style_variations was a no-op duplicate of text_to_image (sc-5950).
function normalizeImageMode(mode) {
  return IMAGE_MODES.includes(mode) ? mode : "text_to_image";
}

// Greatest common divisor, for reducing a W×H resolution to an aspect ratio (sc-5997).
function gcd(a, b) {
  let x = Math.abs(Math.round(a));
  let y = Math.abs(Math.round(b));
  while (y) {
    [x, y] = [y, x % y];
  }
  return x;
}

function recipeMode(recipe) {
  return normalizeImageMode(recipe?.mode);
}

// Image Studio review slots: images in worker-emitted batch-slot order (sc-8853;
// the generationSetId fallback is the only branch needing an explicit sort).
function jobResultAssets(job, assets) {
  return resolveJobResultAssets(job, assets, { type: job.payload?.advanced?.mossImageAdvanced?.output === "latents" ? "tensor" : "image", sortByBatchIndex: true });
}

function jobExpectedCount(job, completedCount) {
  const expected = Number(job.result?.expectedCount ?? job.result?.count ?? job.payload?.count);
  return Number.isFinite(expected) && expected > 0 ? Math.max(expected, completedCount) : completedCount;
}

export function ImageStudio() {
  useMediaTicket();
  const {
    activeProject,
    assets,
    characters,
    createImageJob,
    createPreset,
    refinePrompt,
    // sc-24113's Qwen 2.1 prompt rewriter. App.jsx has always passed it (it is in the provider's
    // value list beside `refinePrompt`), but this destructure never picked it up, so
    // `qwenRewriteAvailable`'s `typeof qwenRewritePrompt === "function"` guard read an undefined
    // global: a ReferenceError at render in a module that is not in strict-undefined territory only
    // because eslint caught it first. Destructured here, beside the sibling it mirrors.
    qwenRewritePrompt,
    magicPrompt,
    imageCaption,
    imageDescribe,
    createModelDownloadJob,
    downloadConditionalComponents,
    createLoraDownloadJob,
    createLoraImportJob,
    deleteAsset,
    purgeAsset,
    gpuOptions,
    imageModels,
    models = [],
    // API token for the Iris restoration plan preview (sc-25683).
    token = "",
    modelCatalogStatus = "ready",
    jobs = [],
    importAsset,
    latestImageAssets,
    recentImageAssets,
    studioLaunch,
    imageLocalJobs = [],
    loras = [],
    jobAction,
    rememberLocalGenerationJob,
    setActiveView,
    setPreviewAsset,
    presets = [],
    promptBatches = [],
    createPromptBatch,
    updatePromptBatch,
    deletePromptBatch,
    requestedGpu,
    selectedAsset,
    setRequestedGpu,
    updateAssetStatus,
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
    // App supplies false until GET /capabilities/mac has returned real platform facts.
    // Legacy/test providers omit the field and already pass authoritative fixtures.
    macCapabilitiesAuthoritative = true,
    macCapabilitiesError = "",
    macCapabilitiesLoading = false,
    refreshMacCapabilities,
    visibleWorkers = [],
    preferencesHydrated,
  } = useAppContext();
  // Prompt-refinement model catalog entry (sc-5605) — drives the "download the
  // refinement model" affordance in RefinePromptControl when Refine fails because the
  // model isn't provisioned on the native worker.
  const refineModel = useMemo(
    () => models.find((entry) => entry.id === PROMPT_REFINE_MODEL_ID),
    [models],
  );
  // Vision-captioner catalog entry (sc-8107) — drives the reference-image caption flow (sc-8108).
  const visionModel = useMemo(
    () => models.find((entry) => entry.id === VISION_CAPTION_MODEL_ID),
    [models],
  );
  // Reference-image caption gate (sc-8110): the picker + "Generate JSON from image" button only goes
  // live once the vision captioner is present (installed/incomplete) AND usable on this platform
  // (visionCaptionModelUsable respects macOnly + Mac gating). When it's absent the section renders the
  // shared ModelAvailabilityGate download offer instead of a button that would only fail on click —
  // ONE coherent gate, formalizing sc-8108's inline error-driven affordance. `ready` matches the
  // catalog state (hasUsableModelFor counts non-missing models); offers come from downloadOffersFor.
  const visionCaptionReady =
    Boolean(visionModel) &&
    visionModel.installState !== "missing" &&
    visionCaptionModelUsable(visionModel, macCapabilities);
  const visionCaptionOffers = useMemo(
    () => downloadOffersFor(models, visionCaptionModelUsable, macCapabilities),
    [models, macCapabilities],
  );
  // Recent Assets list (sc-2088). When the new context value is available, use
  // the bounded 20-most-recent list; fall back to the legacy single-generation
  // list for test contexts that haven't migrated. The existing useGenerationStudio
  // selectStackedJobs() machinery collapses a completed job out of the stack as
  // soon as its assets surface here, so the worker card disappearing matches the
  // spec ("when the current worker completes its assets are added to recent
  // assets, the worker disappears").
  const latestAssets = recentImageAssets ?? latestImageAssets;
  const launchRequest = studioLaunch;
  const nonRecipeLaunchIdRef = useRef(null);
  const trackedLocalJobs = imageLocalJobs;
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onLocalJobCreated = (job) => rememberLocalGenerationJob("image", job);
  const onOpenPresets = () => setActiveView("Presets");
  const onOpenQueue = () => setActiveView("Queue");
  const onPreview = setPreviewAsset;
  // Last-used settings for this workspace, restored on mount. The component is keyed
  // by workspace in App.jsx, so this reads the right snapshot per workspace.
  const saved = useMemo(() => loadStudioSettings("image", activeProject?.id ?? null), [activeProject?.id]);
  const [sceneSuggestions] = useState(() => pickSuggestions(4));
  const [characterSuggestions] = useState(() => pickSuggestions(4, CHARACTER_SUGGESTION_POOL));
  const [mode, setMode] = useState(() => normalizeImageMode(saved.mode));
  const [prompt, setPrompt] = useState(saved.prompt ?? DEFAULT_SCENE_PROMPT);
  // sc-13130: the Style Catalog selection, an entry id from styles.json (or null for "None" /
  // pass-through). Lives next to `prompt` and persists via the same studio saved-state mechanism.
  // Kept as a bare id (not the full entry) so the sc-13132 recipe/replay rehydration can extend
  // it cleanly; the payload fold resolves the id → prompt text via styleTextForId at build time.
  const [styleId, setStyleId] = useState(saved.styleId ?? null);
  // True once the user types or picks a suggestion, so the character-mode default
  // prompt never clobbers their own wording. A restored prompt counts as edited so
  // re-entering character mode doesn't overwrite it.
  const promptEdited = useRef(saved.prompt != null);
  // Structured JSON-caption prompt (Ideogram 4, epic 4725). `caption` is the
  // typed model from ideogramCaption.js; `promptMode` selects the builder form,
  // raw-JSON edit, or the plain-text fallback. Only used when the selected model
  // declares `structuredPrompt`; the plain `prompt` state doubles as the
  // plain-text / magic-prompt seed.
  const [caption, setCaption] = useState(() => saved.structuredCaption ?? emptyCaption());
  const [promptMode, setPromptMode] = useState(saved.promptMode ?? "form");
  // The magic-prompt backend (utility model id) that drafted the current caption, recorded in the
  // structured-prompt recipe (sc-5997). Null until the user runs magic-prompt.
  const [magicPromptBackend, setMagicPromptBackend] = useState(saved.magicPromptBackend ?? null);
  const setPromptFromUser = (value) => {
    promptEdited.current = true;
    setPrompt(value);
    // Editing the idea clears a stale auto-expand error (sc-6501).
    setSubmitError("");
  };
  const [count, setCount] = useState(saved.count ?? 4);

  // Batch Prompt Processing (epic 9952). Batch mode is orthogonal to the T2I/Edit/
  // Character tab — it swaps the single prompt for a list of {{templated}} prompts run
  // as one batch against the current settings. State persists like the rest of the
  // studio; the fan-out on "Run batch" is wired in sc-9956 (slice 4).
  const {
    batchMode, setBatchMode, batchPromptsText, setBatchPromptsText,
    batchVariableValues, setBatchVariableValues, batchName, setBatchName,
    batchScope, setBatchScope, loadedBatchId, batchError, setBatchError,
    batchBusy, batchRun, setBatchRun, batchConfirmPending, setBatchConfirmPending,
    batchPrompts, batchVariables, batchJobCount, handleSaveBatch,
    handleLoadBatch, handleDeleteBatch, handleImportBatch, handleNewBatch,
  } = useBatchPromptState({ saved, createPromptBatch, updatePromptBatch, deletePromptBatch });

  const [model, setModel] = useState(saved.model ?? imageModels[0]?.id ?? "z_image_turbo");
  const [textEncoderSelection, setTextEncoderSelection] = useState({
    modelId: saved.model ?? null,
    id: saved.textEncoderModel ?? null,
  });
  const textEncoderModel =
    textEncoderSelection.modelId === model ? textEncoderSelection.id : null;
  const setTextEncoderModel = (next, modelId = model) =>
    setTextEncoderSelection((current) => {
      const currentId = current.modelId === modelId ? current.id : null;
      return {
        modelId,
        id: typeof next === "function" ? next(currentId) : next,
      };
    });
  const {
    mossImage,
    setMossImage,
    mossImageEdit,
    setMossImageEdit,
    mossImageAdvanced,
    setMossImageAdvanced,
    mossPromptBatch,
    setMossPromptBatch,
    mossDmdSteps,
    setMossDmdSteps,
    mossDownload,
    moss,
    mossEdit,
    mossDmd,
    mossComponents,
    mossDownloadProgress,
    setMossField,
    downloadMossAcceleration,
    mossEditTransition
  } = useMossImageState({
    model,
    saved,
  mode,
  imageModels,
  activeProject,
  token,
  jobs,
  downloadConditionalComponents
  });

  const {
    advancedOpen,
    setAdvancedOpen,
    seed,
    setSeed,
    negativePrompt,
    setNegativePrompt,
    sampler,
    setSampler,
    scheduler,
    setScheduler,
    schedulerShift,
    setSchedulerShift,
    stepsOverride,
    setStepsOverride,
    guidanceOverride,
    setGuidanceOverride,
    toggleAdvanced
  } = useStudioAdvancedSettings(saved, { schedulerShift: mossEdit ? 5 : null, steps: mossEdit ? 30 : "", guidanceScale: mossEdit ? 4 : "" });

  const [resolution, setResolution] = useState(saved.resolution ?? (mossEdit ? "512x512" : "1024x1024"));
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.id ?? "");
  // Multi-image reference set for a multi-reference edit (sc-6211, FLUX.2-dev). Drives the plural
  // `referenceAssetIds` payload when the model declares `ui.multiReference` and the user is in
  // edit_image mode (replaces the single source picker). Empty for every other model/mode.
  const [referenceAssetIds, setReferenceAssetIds] = useState(() =>
    Array.isArray(saved.referenceAssetIds) ? saved.referenceAssetIds : [],
  );
  const [flux3Layout, setFlux3Layout] = useState(saved.flux3Layout ?? { version: 1, boxes: [] });
  const [flux3ReferenceSlots, setFlux3ReferenceSlots] = useState(saved.flux3ReferenceSlots ?? []);
  // Explicit ordered ownership survives 2 -> 1 -> 0; length never chooses the character carrier.
  const [orderedCharacterReferences, setOrderedCharacterReferences] = useState(saved.orderedCharacterReferences === true);
  // Optional SECOND source for a Krea-style two-reference edit (epic 10871 P1.3). Any two images —
  // the `sourceAssetId` above is image 1 (required), this is image 2 (optional), a fixed order (the
  // instruction describes how to combine them). Only surfaced when the model declares
  // `ui.editReferences`; empty otherwise. When set, submit sends an ordered
  // `referenceAssetIds: [image1, image2]` pair instead of the single `sourceAssetId`.
  const [editSecondAssetId, setEditSecondAssetId] = useState("");
  // Edit fit mode (epic 2551): how the source is fitted to the output W×H. Never stretch.
  const [fitMode, setFitMode] = useState(saved.fitMode ?? "crop");
  const [characterId, setCharacterId] = useState(saved.characterId ?? "");
  const [characterLookId, setCharacterLookId] = useState(saved.characterLookId ?? "");
  // Character reference (IP-Adapter / InstantID) — the approved reference image whose
  // identity is carried across variations. `ipAdapterScale` rides in `advanced`; for
  // InstantID, `controlnetScale` (IdentityNet landmark lock) rides there too.
  const [referenceAssetId, setReferenceAssetId] = useState(saved.referenceAssetId ?? "");
  const [ipAdapterScale, setIpAdapterScale] = useState(saved.ipAdapterScale ?? 0.6);
  // img2img reference-guided generation (epic 8588 slice A, sc-8593): the reference picked in the
  // "Start from an image" panel (lifted from ReferenceCaptionPicker) drives generation at this
  // strength on an img2img-capable model (Krea 2 Turbo). Distinct from the character `referenceAssetId`
  // above — same picker, different purpose. Default 0.5 (the full-range slider midpoint; the usable
  // band is model-specific, so no clamp beyond the slider's 0–1).
  const [img2imgReferenceAssetId, setImg2imgReferenceAssetId] = useState("");
  const referenceChangeState = useRef(null);
  const changeReference = useCallback((field, id, setter, explicit = true) => {
    const current = referenceChangeState.current;
    if (current?.available) {
      // A singular approved choice replaces the ordered set even when its old singular mirror
      // already names that asset. Bind the OLD slots before dropping any attachment occurrence.
      setFlux3Layout((layout) => bindLayoutReferenceIds(layout, current.references));
      if ((field === "referenceAssetId" && current.mode === "character_image") ||
          (field === "sourceAssetId" && current.mode === "edit_image")) {
        const ids = id ? [id] : [];
        setReferenceAssetIds(ids);
        setFlux3ReferenceSlots(reconcileLayoutReferences(current.references, ids));
        if (field === "referenceAssetId") setOrderedCharacterReferences(explicit);
      }
      if (field === "characterId" && current.characterId !== id) {
        setReferenceAssetIds([]);
        setReferenceAssetId("");
        setFlux3ReferenceSlots([]);
        setOrderedCharacterReferences(false);
      }
    }
    setter(id);
  }, []);
  const changeReferenceAssetId = useCallback((id, explicit = true) => changeReference("referenceAssetId", id, setReferenceAssetId, explicit), [changeReference]);
  const changeCharacterId = useCallback((id) => changeReference("characterId", id, setCharacterId), [changeReference]);
  const changeSourceAssetId = useCallback((id) => changeReference("sourceAssetId", id, setSourceAssetId), [changeReference]);
  const changeEditSecondAssetId = useCallback((id) => changeReference("editSecondAssetId", id, setEditSecondAssetId), [changeReference]);
  const changeImg2imgReferenceAssetId = useCallback((id) => changeReference("img2imgReferenceAssetId", id, setImg2imgReferenceAssetId), [changeReference]);
  const [img2imgStrength, setImg2imgStrength] = useState(saved.img2imgStrength ?? 0.5);
  // Krea "text style" tap-reweight gain (sc-11878): the ui.textStyleGain slider (Krea/Qwen-family).
  // 1.0 = no-op; >1 warmer/richer (early Qwen3-VL taps), <1 late-biased. Resets to the model default on
  // model change like the other tuning knobs; emitted to advanced.textStyleGain only when off default.
  const [textStyleGain, setTextStyleGain] = useState(saved.textStyleGain ?? 1.0);
  // Krea 2 multi-phase denoise (epic 13879 S5, sc-13885). `multiPhaseEnabled` opts the job into the
  // multi-phase lane; `multiPhasePhases` is the editor's ordered phase list (LoRA refs stored by
  // stable id — serializePhases resolves them to indices at emit time). `multiPhaseOpen` is the
  // experimental disclosure's collapsed/expanded state (default collapsed, busy-page guidance).
  const [multiPhaseEnabled, setMultiPhaseEnabled] = useState(saved.multiPhaseEnabled ?? false);
  const [multiPhasePhases, setMultiPhasePhases] = useState(
    Array.isArray(saved.multiPhasePhases) ? saved.multiPhasePhases : [],
  );
  const [multiPhaseOpen, setMultiPhaseOpen] = useState(saved.multiPhaseOpen ?? false);
  const [controlnetScale, setControlnetScale] = useState(saved.controlnetScale ?? 0.8);
  // Variation knob for backbones whose CFG is decoupled from IP-Adapter:
  // FLUX (true_cfg_scale alongside ipAdapterScale) and Qwen-Image-Edit (true_cfg_scale
  // *replaces* the IP-Adapter slider because Qwen's edit pipeline doesn't use one).
  // Per-model spec rides in ui.variationStrength; resets to the model default on
  // model change like the other tuning knobs (sc-2017).
  const [trueCfgScale, setTrueCfgScale] = useState(saved.trueCfgScale ?? 4.0);
  // InstantID canonical head angle ("" = match the reference's own angle). Rides advanced.viewAngle.
  const [viewAngle, setViewAngle] = useState(saved.viewAngle ?? "");
  // Pose library: selected pose ids. When non-empty, the job carries advanced.poses
  // (one image per pose) instead of the normal variations count. Transient (not saved).
  const [selectedPoseIds, setSelectedPoseIds] = useState([]);
  const [workflowPoses, setWorkflowPoses] = useState([]);
  const [poseLibraryReplayOpen, setPoseLibraryReplayOpen] = useState(false);
  // Strict-control panel (epic 8236, sc-8245). The selected control type (pose / canny / depth),
  // gated to the backbone's `ui.controlModes`. Pose reuses `selectedPoseIds`; canny/depth use a
  // control-image asset + a preprocess-vs-passthrough toggle. `controlScale` (advanced.controlScale)
  // is the control-lock strength. All reset to the model's defaults on model change (below).
  const {
    controlMode, setControlMode, controlImageAssetId, setControlImageAssetId,
    controlImagePassthrough, setControlImagePassthrough, controlScale,
    setControlScale, controlOverlayId, setControlOverlayId,
  } = useStrictControlState(saved);
  // false = preprocess (worker auto-derives the map from the image → request.sourceAssetId);
  // true = use-as-is (the user-supplied map passes through verbatim → advanced.controlImage).
  // Selected trained ControlNet overlay id (sc-10165 B4) — for backbones whose pose control rides a
  // registered overlay (Krea 2 Turbo). Flows to advanced.controlWeights.overlayId; the API resolves it.
  // Configurable sampler / scheduler (epic 1753). Restored from per-workspace
  // settings; reset to the selected model's manifest defaults whenever the
  // model changes.

  // sc-25681: the CFG interval (two raw inputs, "" = the model's own bound) and the caption-
  // overflow policy ("default" = the checkpoint's own), for a model that declares them (Iris-3B).
  const [cfgInterval, setCfgInterval] = useState(saved.cfgInterval ?? "");
  const [captionOverflow, setCaptionOverflow] = useState(saved.captionOverflow ?? "default");
  // Guidance method (epic 7434). "cfg" is the engine-standard no-op default; the
  // picker only surfaces alternatives a model advertises on the active backend
  // (CFG++ on the SDXL family / MLX). Rides `advanced.guidanceMethod`.
  const [guidanceMethod, setGuidanceMethod] = useState(saved.guidanceMethod ?? "cfg");
  // Steps / guidance: previously worker-only knobs surfaced via this same
  // advanced panel. "" represents "use the model default" so the user can
  // clear the override.

  // Advanced resolution override: a custom Width/Height to experiment beyond the model's
  // pre-declared Aspect options (e.g. Krea 2 up to 4K). "" = "use the Aspect dropdown" for
  // that axis, mirroring the Steps/Guidance overrides. Effective dims are derived below and
  // ride the existing top-level width/height payload; the backend caps each at 256–4096.
  const [widthOverride, setWidthOverride] = useState(saved.widthOverride ?? "");
  const [heightOverride, setHeightOverride] = useState(saved.heightOverride ?? "");
  // Recipe-only optional axes. Never infer authored intent from the actual output dimensions.
  const [referenceDimensionIntent, setReferenceDimensionIntent] = useState(saved.referenceDimensionIntent ?? null);
  const handleResolutionChange = (value) => {
    setReferenceDimensionIntent(null);
    setResolution(value);
  };
  // Flash attention (sc-3674): fused attention on the candle (Windows/CUDA) SDXL backend — faster +
  // less VRAM. Per-payload (sent in `advanced.flashAttn`); the worker honors it only on candle, and
  // ignores it on every other backend. Default on. Sticky pref (persisted), not model-reset.
  const [flashAttn, setFlashAttn] = useState(saved.flashAttn ?? true);
  // FLUX.2-dev "Enhance prompt" (sc-6135): the model's built-in Mistral3 caption upsampler rewrites
  // the prompt before encoding — text-only for txt2img, reference-aware for edit. Per-payload
  // (`advanced.enhancePrompt`); only flux2_dev acts on it. Sticky pref (persisted), default off.
  const [enhancePrompt, setEnhancePrompt] = useState(saved.enhancePrompt ?? false);
  // Boogu precision toggle (sc-6568): off = the packed Q8 default; on emits `advanced.mlxQuantize: 0`
  // (the full-precision bf16 build, fetched on demand by the worker). Sticky pref, default off (Q8).
  const [pickedBf16Precision, setBf16Precision] = useState(saved.bf16Precision ?? false);
  const [recipeTierRequest, setRecipeTierRequest] = useState(null);
  // Transparency / RGBA output (sc-24113): on emits `advanced.transparentBackground: true`, which
  // the worker's `qwen_alpha` adapter turns into a four-channel decode and the sc-24111 egress path
  // writes as a genuinely transparent PNG. Sticky pref, default off; the toggle only RENDERS when
  // the selected model advertises alpha output, so a stale `true` carried over from Qwen 2.1 to a
  // model that cannot serve it is inert — `transparencyAdvanced` re-checks the capability against
  // the CURRENTLY selected model at payload-build time, exactly like `bf16Precision`/`usePid`.
  const [transparentBackground, setTransparentBackground] = useState(
    saved.transparentBackground ?? false,
  );
  // Generation-time quant-tier toggle (sc-8515, epic 8506): for a model with MORE THAN ONE
  // quant tier installed (sc-8508 per-variant install state), the advanced panel renders a
  // picker so the user can A/B a bf16 vs Q8 vs Q4 build. The picked tier rides
  // `advanced.mlxQuantize` (bf16→0, q8→8, q4→4); the worker's resolve_quant + generator cache
  // route to it (reload-always — the cache evicts + reloads on a heavy-tier switch). `quantTier`
  // holds the selected tier key ("" = no picker / not applicable). The user's last EXPLICIT pick is
  // persisted per (screen, model) in `lastTierStore` (epic 10721 / sc-10727) — project-independent,
  // so re-entering a model on this screen restores the tier you last generated with, in any
  // workspace and across app restarts. It seeds the picker as the top rung below a same-session pick
  // and above the model's base default (see the seed effect + `defaultTierSelection`).
  // PiD decoder toggle (epic 7840, sc-7851): off = the model's native VAE decode; on emits
  // `advanced.usePid: true`, routing decode through the optional PiD pixel-diffusion decoder
  // (decode + 2K/4K super-resolve, non-commercial output). Sticky pref, default off; the
  // toggle only renders + emits when the model is PiD-eligible AND its checkpoint is installed
  // (showPidToggle), so a stale `true` on a non-eligible model is inert — mirrors bf16Precision.
  // PiD output tier (epic 7840, sc-10054): PiD always super-resolves the base latent 4×, so this picks
  // the effective base — "4k" keeps the requested base (~4096 output, the pre-tier behavior), "2k" caps
  // it (~2048 output, faster + less GPU memory). Sticky pref, default "4k". Rides `advanced.pidTarget`
  // (emitted only when the PiD toggle is shown+on AND "2k" is picked — "4k" is the worker default).
  const { usePid, setUsePid, pidTarget, setPidTarget } = usePidState(saved);
  // Experimental alternate terminal decoder. Native is intentionally represented as an omitted
  // request key so the established provider path stays byte-for-byte identical.
  const [decoder, setDecoder] = useState(saved.decoder ?? "native");
  const [faceRestore, setFaceRestore] = useState(false);
  // User-created poses (reserved global project) join the built-in library in both
  // the picker and the id→keypoints resolver below, so saved poses can generate.
  const loadUserPoses = useUserPoseLoader();
  const { byId: poseById } = usePoseLibrary({ loadUserPoses, extraPoses: workflowPoses });
  const [upscaleEnabled, setUpscaleEnabled] = useState(saved.upscaleEnabled ?? false);
  const [upscaleFactor, setUpscaleFactor] = useState(saved.upscaleFactor ?? 2);
  const [upscaleEngine, setUpscaleEngine] = useState(saved.upscaleEngine ?? "real-esrgan");
  // SeedVR2 detail/softness knob (0..1, sc-4815) — only used by the seedvr2 engine.
  const [upscaleSoftness, setUpscaleSoftness] = useState(saved.upscaleSoftness ?? 0);
  // Iris restoration controls (sc-25683) — only used by the `iris` engine, which is sized by `scale`.
  const [upscaleRestore, setUpscaleRestore] = useState(() =>
    irisRestoreControls(saved.upscaleRestore ?? IRIS_RESTORE_DEFAULTS),
  );
  const [restorePlanStatus, setRestorePlanStatus] = useState({ ready: false, blocked: false, error: null });
  const [hiresFixEnabled, setHiresFixEnabled] = useState(saved.hiresFixEnabled ?? false);
  const [hiresFixSteps, setHiresFixSteps] = useState(saved.hiresFixSteps ?? 0);
  const [hiresFixDenoisingStrength, setHiresFixDenoisingStrength] = useState(
    saved.hiresFixDenoisingStrength ?? 0.7,
  );
  const [hiresFixUpscaleBy, setHiresFixUpscaleBy] = useState(saved.hiresFixUpscaleBy ?? 2);
  const [hiresFixCfgScale, setHiresFixCfgScale] = useState(saved.hiresFixCfgScale ?? "");
  const [submitting, setSubmitting] = useState(false);
  // Auto-expand state (sc-6501): when a structured model is in plain-text mode, Generate first
  // expands the idea into a JSON caption via magic-prompt. `expanding` drives the button label;
  // `submitError` surfaces an expansion failure (e.g. the utility model isn't installed) without
  // ever falling back to sending raw plain text.
  const [expanding, setExpanding] = useState(false);
  const [submitError, setSubmitError] = useState("");

  // Prompt tools (epic UI-refinement): which inline prompt-tool panel is open —
  // null | "describe" (reference-image caption) | "refine" (rewrite my prompt).
  // Replaces the always-rendered ReferenceCaptionPicker + RefinePromptControl pair
  // with two toggle tiles; only one panel opens at a time.
  const [promptTool, setPromptTool] = useState(null);
  const togglePromptTool = useCallback(
    (tool) => setPromptTool((current) => (current === tool ? null : tool)),
    [],
  );
  const depthControlAssets = useMemo(
    () =>
      assets.filter(
        (asset) =>
          asset.type === "depth" &&
          asset.projectId === activeProject?.id &&
          !asset.status?.trashed &&
          !asset.status?.rejected,
      ),
    [assets, activeProject?.id],
  );
  const editImageAssets = useMemo(
    () =>
      assets.filter(
        (asset) =>
          (asset.type === "image" || asset.type === "frame") &&
          asset.projectId === activeProject?.id &&
          !asset.status?.trashed &&
          !asset.status?.rejected,
      ),
    [assets, activeProject?.id],
  );
  const selectedAssetEditableSourceId = useMemo(
    () =>
      selectedAsset?.id && editImageAssets.some((asset) => asset.id === selectedAsset.id)
        ? selectedAsset.id
        : "",
    [editImageAssets, selectedAsset?.id],
  );

  function handleModeChange(nextMode) {
    if (nextMode === "edit_image") {
      setCount(1);
    } else if (nextMode === "text_to_image" || nextMode === "character_image") {
      setCount(4);
    }
    setMode(nextMode);
  }

  const handleUpscaleEngineChange = useCallback((nextEngine) => {
    setUpscaleEngine(nextEngine);
    const factors = upscaleFactorsForEngine(nextEngine);
    if (!factors.includes(upscaleFactor)) {
      setUpscaleFactor(factors[0]);
    }
  }, [upscaleFactor]);

  // Engines offered in the picker; AuraSR is dropped on every platform (sc-3668 / sc-5499).
  // Iris restoration (sc-25683) is offered only once its own `iris_3b_restore` task is installed.
  const availableUpscaleEngines = upscaleEnginesForPlatform(macCapabilities, models);
  const irisUpscale = upscaleEnabled && upscaleEngineIsRestoration(upscaleEngine);
  // If a restored/saved engine is gated out (e.g. a stale saved AuraSR selection), fall back to the
  // default real-esrgan engine so the user never submits an aura-sr job the native workers refuse (sc-8853).
  useUpscaleEngineFallback({
    macCapabilities,
    models,
    upscaleEngine,
    setUpscaleEngine,
    upscaleFactor,
    setUpscaleFactor,
  });

  // PiD, post-generation Upscale, and Hires.fix are mutually exclusive super-resolution paths.
  // If restored state carries PiD plus either option, keep the explicit upscale process and drop PiD.
  useEffect(() => {
    if (usePid && (upscaleEnabled || hiresFixEnabled)) {
      setUsePid(false);
    }
    if (upscaleEnabled && hiresFixEnabled) {
      setUpscaleEnabled(false);
    }
  }, [usePid, upscaleEnabled, hiresFixEnabled, setUsePid]);

  useEffect(() => {
    if (mode === "edit_image" && selectedAssetEditableSourceId) {
      changeSourceAssetId(selectedAssetEditableSourceId);
    }
  }, [mode, selectedAssetEditableSourceId, changeSourceAssetId]);

  useEffect(() => {
    if (launchRequest?.view !== "Image") {
      return;
    }
    if (launchRequest.recipe) {
      return;
    }
    const freshLaunch = nonRecipeLaunchIdRef.current !== launchRequest.id;
    nonRecipeLaunchIdRef.current = launchRequest.id;
    if (freshLaunch) {
      // A fresh non-recipe launch is a new intent entering this keep-alive studio. Workflow pose
      // records belong to the recipe launch that created them; carrying them into a character/edit
      // launch would silently submit advanced.poses (and its face-restoration opt-in) from the last
      // workflow. The id guard prevents an equivalent context-object rerender from clearing poses
      // the user picked while editing this same launch.
      setSelectedPoseIds([]);
      setWorkflowPoses([]);
      setPoseLibraryReplayOpen(false);
      setFaceRestore(false);
      setFlux3Layout({ version: 1, boxes: [] });
      setFlux3ReferenceSlots([]);
      setOrderedCharacterReferences(false);
      setReferenceDimensionIntent(null);
    }
    if (launchRequest.characterId) {
      setMode(normalizeImageMode(launchRequest.mode ?? "character_image"));
      setCharacterId(launchRequest.characterId);
      setCharacterLookId(launchRequest.lookId ?? "");
      if (launchRequest.referenceAssetId) {
        setReferenceAssetId(launchRequest.referenceAssetId);
      }
      return;
    }
    if (launchRequest.assetId !== selectedAsset?.id) {
      return;
    }
    setMode(normalizeImageMode(launchRequest.mode));
    // Preselect the family-matched edit model resolved at launch time (App.jsx). It's
    // edit-capable by construction, so the availableModels snap-to-first effect leaves
    // it in place; when absent the snap falls back to the default edit model.
    if (launchRequest.model) {
      setModel(launchRequest.model);
    }
    if (launchRequest.mode === "edit_image" && selectedAssetEditableSourceId) {
      changeSourceAssetId(selectedAssetEditableSourceId);
      if (isFlux3LayoutModel(launchRequest.model ?? referenceChangeState.current?.model)) {
        // A launch can switch models in this same commit, before changeSource sees the new model.
        setReferenceAssetIds([selectedAssetEditableSourceId]);
        setFlux3ReferenceSlots(reconcileLayoutReferences([], [selectedAssetEditableSourceId]));
      }
    }
  }, [launchRequest, selectedAsset?.id, selectedAssetEditableSourceId, changeSourceAssetId]);

  // Mac UI gating (sc-3486): on a Mac in MLX-required mode, hide torch-only models from the
  // picker so the user can't select something that would only error. Inert elsewhere.
  const macImageModels = useMemo(
    () => macAvailableModels(imageModels, macCapabilities),
    [imageModels, macCapabilities],
  );
  const macGating = macGatingActive(macCapabilities);
  const imageModelServesMode = useCallback((item, value) => {
    const capabilitiesDeclared = Array.isArray(item?.capabilities);
    const caps = capabilitiesDeclared ? item.capabilities : [];
    if (value === "edit_image") {
      return (
        (caps.includes("edit_image") || caps.includes("image_edit")) &&
        !macModelFeatureBlock(item, macCapabilities, "edit")
      );
    }
    if (value === "character_image") {
      // Only models with a reference-image (IP-Adapter) engine can preserve a
      // character's identity from one reference; gate the picker to them.
      return caps.includes("character_image") && !macModelFeatureBlock(item, macCapabilities, "reference");
    }
    // text_to_image: only models that declare a real sourceless T2I path (sc-5549).
    // Without this gate the Text tab leaked edit-only models (run a degraded
    // sourceless edit) and reference-only identity models (MLX-ineligible without a
    // reference → strand on "Waiting for an available worker"); both classes lack
    // text_to_image. Mirrors the per-capability gating the other three modes use.
    // Older installed-model records predate the capability array and historically
    // represented the Text workflow. Preserve that compatibility, while treating
    // every explicit capability list — including [] — as authoritative (so
    // unsupported, edit-only, and reference-only models can never leak into Text).
    return !capabilitiesDeclared || caps.includes("text_to_image");
  }, [macCapabilities]);
  const modelsForMode = useCallback(
    (value) => macImageModels.filter((item) => imageModelServesMode(item, value)),
    [imageModelServesMode, macImageModels],
  );
  const availableModels = useMemo(
    () => modelsForMode(mode),
    [mode, modelsForMode],
  );
  const pickerModels = availableModels;
  // Model-availability gate (sc-5947): when the user has no mac-available image model at all,
  // show recommended image-model downloads instead of the studio. `ready` matches the picker
  // across any of its modes; offers come from the full catalog via imageModelUsable,
  // recommended-first. A mode with no compatible installed model stays inside the
  // studio so the user can switch tabs, but renders an empty, actionable picker.
  const modelReady = macImageModels.length > 0;
  const modelOffers = useMemo(
    () => downloadOffersFor(models, imageModelUsable, macCapabilities),
    [models, macCapabilities],
  );
  const modelDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );
  // When the mode change filters out the current model (e.g. Lens-Turbo is the
  // text default but isn't edit-capable), snap to the first available model so
  // the dropdown's displayed option matches the value actually submitted.
  useEffect(() => {
    if (pickerModels.length && !pickerModels.some((item) => item.id === model)) {
      setModel(pickerModels[0].id);
    }
  }, [pickerModels, model]);
  // Resolve only through the current mode's capability-filtered set. A restored
  // selection can be stale until the snap effect runs; it must never become the
  // effective model for either rendering defaults or submitting a job.
  const selectedModel = availableModels.find((item) => item.id === model);
  const selectedModelServesMode = Boolean(selectedModel);
  const tierOptions = useMemo(
    () => nativeTierEligibility(selectedModel, visibleWorkers),
    [selectedModel, visibleWorkers],
  );
  // Qwen-Image 2.1's OFFICIAL prompt rewriters (sc-24113, epic 24107). Two optional, explicitly
  // downloaded checkpoints, and WHICH ONE applies is decided by the request rather than by a
  // picker: references attached ⇒ the editing rewriter, none ⇒ the text-to-image one. The catalog
  // ids mirror `crates/sceneworks-worker/src/qwen_prompt_rewrite.rs`'s `Rewriter::catalog_id`.
  // Name an asset the way the rest of this screen does, for the ordered-reference rail.
  const assetLabelFor = useCallback(
    (id) => editImageAssets.find((asset) => asset.id === id)?.name ?? id,
    [editImageAssets],
  );
  const qwenRewriteReferenceIds = useMemo(() => {
    // The SAME ordered list the Generate button will send, so the rewrite sees what the render
    // will. `referenceAssetIds` is the plural multi-reference picker; `referenceAssetId` /
    // `sourceAssetId` are the single-reference and edit-source flows, which the engine treats as a
    // one-reference edit.
    if (mode === "edit_image" && referenceAssetIds.length) return referenceAssetIds;
    const single = mode === "edit_image" ? sourceAssetId : referenceAssetId;
    return single ? [single] : [];
  }, [mode, referenceAssetIds, referenceAssetId, sourceAssetId]);
  const qwenRewriteModel = useMemo(() => {
    if (!qwenRewriterAppliesTo(selectedModel)) return undefined;
    const id = qwenRewriteReferenceIds.length
      ? QWEN_REWRITE_I2I_MODEL_ID
      : QWEN_REWRITE_T2I_MODEL_ID;
    return models.find((entry) => entry.id === id);
  }, [models, selectedModel, qwenRewriteReferenceIds.length]);
  // The tile renders ONLY when the matching rewriter is already installed. This is the acceptance
  // criterion "direct prompting works with neither rewriter installed — no download, no prompt to
  // install, no degraded path": with nothing installed there is no affordance at all, not a
  // disabled button that nags.
  const qwenRewriteAvailable =
    Boolean(qwenRewriteModel) &&
    qwenRewriteModel.installState !== "missing" &&
    typeof qwenRewritePrompt === "function";
  const catalogTextEncoderOptions = selectedModel?.textEncoderOptions ?? [];
  // A replayed/saved authored id remains a real control value even if the refreshed catalog can no
  // longer advertise this surface. Keep it visible and submit it unchanged so the server rejects a
  // stale choice; never make disappearance look like an implicit switch to the model encoder.
  const supportsTextEncoderSelection =
    catalogTextEncoderOptions.length > 0 || Boolean(textEncoderModel);
  const textEncoderOptions =
    catalogTextEncoderOptions.length > 0
      ? catalogTextEncoderOptions
      : [
          {
            id: "default",
            label: "Model encoder (default)",
            description: "Uses the encoder paired with this model.",
            isDefault: true,
          },
        ];
  const defaultTextEncoderId =
    textEncoderOptions.find((option) => option.isDefault)?.id ??
    textEncoderOptions[0]?.id ??
    "default";
  const selectedTextEncoderModel = textEncoderModel ?? defaultTextEncoderId;
  const selectedTextEncoderAvailable = textEncoderOptions.some(
    (option) => option.id === selectedTextEncoderModel,
  );
  const modeLabel =
    mode === "edit_image" ? "Edit" : mode === "character_image" ? "With character" : "Text";
  // Booru-convention prompt hint (sc-10760): non-null for danbooru-tag models (Anima, Illustrious)
  // that declare `ui.promptHint`; rendered under the prompt box with a link into the prompt guide.
  const promptHint = promptHintFor(selectedModel?.ui);
  // Prompt guide for the selected model; fall back to the generic image guide
  // when a model declares none, so the button is always useful (sc-1817).
  const { guideOpen, openGuide, closeGuide, promptGuide } = useStudioPromptGuide(selectedModel, {
    title: "Image Prompt Guide",
    path: "/prompt-guides/generic-image.md",
  });
  // Reference-tuning hints declared by the model (ui.*). InstantID raises the
  // reference-strength default and exposes a second "Identity structure" slider
  // (controlnetConditioningScale); models without these keys (e.g. Kolors) keep the
  // single reference-strength slider at the global default.
  const identityStructure = selectedModel?.ui?.identityStructure;
  // Optional label/range override for the primary reference-strength slider (sc-8278: klein maps it
  // to image-guidance over 1.0–2.5). Absent ⇒ the legacy "Reference strength" 0–1 slider.
  const referenceStrengthCfg = selectedModel?.ui?.referenceStrength;
  // img2img / reference-guided generation (epic 8588 slice A, sc-8593): a `ui.img2img` flag (Krea 2
  // Turbo) — a UI toggle like poseLibrary/multiReference, NOT a `capabilities` value (z-image already
  // uses "image_to_image" for its distinct edit-mode img2img, so a capability gate would collide).
  // Turns the shared "Start from an image" picker double-duty: the same reference can be described into
  // a prompt AND/OR guide the render via a strength slider, without needing the vision captioner.
  // `ui.img2imgStrength` optionally overrides the slider label/range.
  const supportsImg2img = Boolean(selectedModel?.ui?.img2img);
  const img2imgStrengthConfig = selectedModel?.ui?.img2imgStrength ?? null;
  // Krea "text style" control (sc-11878): the `ui.textStyleGain` slider descriptor. Presence gates the
  // control — only Krea (Qwen-Image-family) declares it, so the slider (and advanced.textStyleGain) are
  // Krea-only. Absent ⇒ no slider, no payload key.
  const textStyleGainConfig = selectedModel?.ui?.textStyleGain ?? null;
  const supportsTextStyle = Boolean(textStyleGainConfig);
  // Whether the edit model can outpaint (generate the padded border) — only models that
  // accept an inpaint mask (image_inpaint, SDXL family). Gates the Outpaint fit option.
  const editInpaintCapable = (selectedModel?.capabilities ?? []).includes("image_inpaint");
  // Canonical head angles the model can render from a frontal reference (InstantID).
  const viewAngles = Array.isArray(selectedModel?.ui?.viewAngles) ? selectedModel.ui.viewAngles : null;
  // Whether the model supports the OpenPose pose library (InstantID).
  const poseLibrary = Boolean(selectedModel?.ui?.poseLibrary);
  // Strict-control modes the selected backbone advertises (sc-8245) — canonical-ordered, gated to
  // `ui.controlModes` (the manifest / STRICT_CONTROL_ENGINES `supported_kinds`). Empty ⇒ no strict
  // control ⇒ the panel hides. The control panel surfaces only in text-to-image mode (its conditioning
  // is its own input image / pose, distinct from the edit / character source).
  const controlModes = useMemo(() => supportedControlModes(selectedModel), [selectedModel]);
  const controlScaleConfig = selectedModel?.ui?.controlScale ?? null;
  // Backbones whose pose control rides a registered trained overlay (sc-10165 B4) advertise
  // `ui.controlOverlay` (Krea 2 Turbo); the ControlPanel then shows a self-fetching overlay picker.
  const controlOverlayBaseModel = selectedModel?.ui?.controlOverlay ? selectedModel.id : null;
  // The control type actually in effect: the user's pick when the backbone still supports it, else the
  // first supported mode. Decouples the gating (derived) from the raw state so a backbone switch that
  // strands an unsupported pick degrades gracefully even before the reset effect runs.
  const activeControlMode = controlModes.includes(controlMode) ? controlMode : (controlModes[0] ?? null);
  const showControlPanel = mode === "text_to_image" && controlModes.length > 0;
  const effectiveControlScale =
    typeof controlScale === "number" ? controlScale : controlScaleConfig?.default ?? 0.9;

  // Pose-library + strict-control conditioning, derived once so the single Generate
  // (submit) and the batch run (buildBatchJobRequest) share them (sc-9980). Pose emits one
  // image per selected pose instead of `count` variations; canny/depth carry a control image
  // routed by the preprocess (derive → sourceAssetId) vs passthrough (advanced.controlImage) toggle.
  const usePosePayload =
    (mode === "character_image" && referenceAssetId && poseLibrary) ||
    (showControlPanel && activeControlMode === "pose");
  const posePayload =
    usePosePayload && selectedPoseIds.length
      ? selectedPoseIds
          .map((id) => poseById[id])
          .filter(Boolean)
          .map(poseRecordToPayload)
      : [];
  const controlActive = showControlPanel && Boolean(activeControlMode);
  const controlIsImageMode = controlActive && activeControlMode !== "pose";
  // A saved depth-map asset (Iris-3B, sc-25682) IS a depth control map: it is offered only in depth
  // mode and always rides as the passthrough `advanced.controlImage` — the worker derives the
  // near-bright control image from its raw map (it is never re-estimated as if it were a photo).
  const controlAssetIsDepth =
    activeControlMode === "depth" && depthControlAssets.some((asset) => asset.id === controlImageAssetId);
  const effectiveControlPassthrough = controlImagePassthrough || controlAssetIsDepth;
  const controlPreprocessSourceId =
    controlIsImageMode && !effectiveControlPassthrough && controlImageAssetId ? controlImageAssetId : null;
  const controlPassthroughId =
    controlIsImageMode && effectiveControlPassthrough && controlImageAssetId ? controlImageAssetId : null;

  // Images each resolved-prompt job emits: the pose count when poses are selected (they
  // replace `count` variations), else `count`. Total batch image count feeds the run label
  // and the cardinality cap.
  const batchImagesPerPrompt = posePayload.length || count;
  const batchTotal = batchJobCount * batchImagesPerPrompt;

  // A pending large-run confirmation is for one specific total — reset it whenever the batch
  // size changes so the user always re-confirms against the current count.
  useEffect(() => {
    setBatchConfirmPending(false);
  }, [batchTotal, setBatchConfirmPending]);
  // Whether the model exposes its built-in prompt upsampler ("Enhance prompt" toggle) — FLUX.2-dev.
  // Whether the model ships a packed default + a hosted full-precision bf16 build, exposing the
  // Studio "Full precision (bf16)" toggle (sc-6568) — Boogu Base/Turbo/Edit.
  const precisionToggle = Boolean(selectedModel?.ui?.precisionToggle);
  const precisionReplay = precisionToggle && recipeTierRequest?.model === model &&
    ["bf16", "q8"].includes(recipeTierRequest?.tier);
  const bf16Precision = precisionReplay ? recipeTierRequest.tier === "bf16" : pickedBf16Precision;
  // Quant-tier picker state (sc-8515, generalized). `availableTiers` is the SELECTABLE set — tiers that
  // are installed AND complete on disk, and the ONLY tiers we ever send to the worker (the user's bottom
  // line: never generate off a tier that isn't in the cache). `possibleTiers` is the FULL display set —
  // every tier the model can have, installed or not — so an un-downloaded tier renders disabled rather
  // than silently missing. The picker shows whenever there is MORE THAN ONE possible tier (so a user with
  // one installed tier still sees the others, greyed), provided at least one is actually installed to
  // select. Boogu's `precisionToggle` is orthogonal — those models have no tier matrix, so `possibleTiers`
  // is empty and this stays hidden for them.
  const availableTiers = useMemo(
    () => installedTiers(selectedModel, tierOptions),
    [selectedModel, tierOptions],
  );
  // Capability-aware "Auto" default (epic 10721): the highest-fidelity tier that fits this machine's
  // memory. Fed to `defaultTierSelection` as the base when the global quality setting is Auto (the
  // default), so a small model (SANA-Sprint) defaults to bf16 and a heavy one on a small Mac to what
  // fits — instead of a flat q8. `null` memory (probe pending / unavailable) leans to the highest tier;
  // the worker's capability downtier (sc-10733) still clamps a non-explicit pick to what actually fits.
  const hostMemory = useHostMemory();
  const activeBackend = macCapabilities?.macGatingActive ? "mlx" : "candle";
  const decoderOptions = useMemo(
    () =>
      macCapabilitiesAuthoritative
        ? (selectedModel?.decoders?.byBackend?.[activeBackend] ?? [])
        : null,
    [selectedModel, activeBackend, macCapabilitiesAuthoritative],
  );
  const showDecoderPicker = Boolean(decoderOptions?.length);
  // A restored alternate selection must survive the window where App still has only its
  // inert fallback. During that window the empty Candle option list is not evidence that the
  // decoder is incompatible, and omitting the selection from a request would silently change
  // the user's run. Native requests remain safe and need not wait for this optional surface.
  const decoderCapabilitiesPending = !macCapabilitiesAuthoritative && decoder !== "native";
  useEffect(() => {
    if (!selectedModel || !macCapabilitiesAuthoritative) return;
    const reconciled = reconcileDecoderSelection(decoder, decoderOptions);
    if (reconciled !== decoder) setDecoder(reconciled);
  }, [decoder, decoderOptions, selectedModel, macCapabilitiesAuthoritative]);
  useEffect(() => {
    if (decoder !== "native" && usePid) {
      setUsePid(false);
    }
  }, [decoder, usePid, setUsePid]);
  // Route-aware prompt enhancement: both native FLUX.2-dev base/edit providers implement it, while
  // Candle's unsupported character/reference aliases, Klein, an unknown backend, and the separate
  // strict-control provider fail closed. `posePayload` is the strict-pose route even when the control
  // panel itself is not open (character recipe replay).
  const promptEnhanceStrictControlActive =
    posePayload.length > 0 ||
    (controlActive && activeControlMode !== "pose") ||
    Boolean(controlPassthroughId || controlOverlayId);
  const promptEnhance = promptEnhancementAvailable({
    model: selectedModel,
    backend: activeBackend,
    mode,
    strictControlActive: promptEnhanceStrictControlActive,
  });
  const unifiedMemoryGb = hostMemoryGbForBackend(hostMemory, activeBackend);
  const autoTier = useMemo(
    () => suggestTier(selectedModel, unifiedMemoryGb, { backend: activeBackend }),
    [selectedModel, unifiedMemoryGb, activeBackend],
  );
  const { quantTier: pickedTier, tierSwitching, handleTierChange: pickTier, tierExplicit: pickedExplicit, tierSuggested, confirmTier } =
    useQuantTierPicker({
      screen: TIER_SCREEN,
      resolvedTier: precisionReplay ? null : recipeTierRequest?.tier,
      model,
      selectedModel,
      availableTiers,
      tierOptions,
      autoTier,
      useGenerationQuality: true,
      preferencesHydrated,
    });
  // Keep a replay's exact numeric format, not this user's sticky/default. Unavailable recipes
  // stay blocked until an explicit new pick; no worker or missing artifact can silently change it.
  const quantTier = pickedTier;
  const tierExplicit = Boolean(recipeTierRequest) || pickedExplicit;
  const replayTierUnavailable = recipeTierRequest && !precisionReplay &&
    (recipeTierRequest.model !== model || !availableTiers.includes(quantTier));
  const handleTierChange = (tier) => {
    if (!availableTiers.includes(tier)) return;
    setRecipeTierRequest(null);
    pickTier(tier);
  };
  const possibleTiers = useMemo(
    () => allPossibleTiers(selectedModel, tierOptions),
    [selectedModel, tierOptions],
  );
  const tierPickerItems = useMemo(
    () => tierPickerOptions(selectedModel, tierOptions),
    [selectedModel, tierOptions],
  );
  const showTierPicker = useMemo(
    () => possibleTiers.length > 1 && availableTiers.length > 0,
    [possibleTiers, availableTiers],
  );
  // Per-model quality floor (sc-10731): the model's minimum-fidelity tier (`minQualityTier`) and whether
  // the CURRENT pick sits below it. The DEFAULT is already clamped up to the floor (defaultTierSelection),
  // so this only fires when the user EXPLICITLY picks a below-floor tier — a deliberate quality/creative
  // choice we HONOR, but flag with a non-blocking advisory (never silently switch their tier).
  const qualityFloor = useMemo(() => modelQualityFloor(selectedModel), [selectedModel]);
  const tierBelowFloor = useMemo(
    () => showTierPicker && isBelowFloor(quantTier, selectedModel),
    [showTierPicker, quantTier, selectedModel],
  );
  // PiD decoder toggle visibility (epic 7840, sc-7851): the model's latent space has a PiD
  // backbone (ui.pid) AND that backbone's PiD checkpoint is installed. Hidden otherwise — for
  // non-eligible models (e.g. SenseNova) and for eligible models whose checkpoint isn't
  // downloaded yet (provisioned by sc-7852), where the worker would no-op to the native VAE.
  const showPidToggle = useMemo(() => pidToggleVisible(selectedModel, models), [selectedModel, models]);
  // Whether the model supports multi-image reference editing (sc-6211) — edit_image mode shows a
  // multi-select reference picker (plural `referenceAssetIds`) instead of the single source picker.
  // FLUX.2-dev only (its DiT sequence-gated chunking keeps the multi-reference edit under 96 GB).
  const multiReference = Boolean(selectedModel?.ui?.multiReference);
  // Mage Edit keeps its required source distinct from its optional ordered reference set. Unlike
  // `multiReference`, this never replaces `sourceAssetId`; it adds `referenceAssetIds` alongside it.
  const sourceWithMultiReference = Boolean(selectedModel?.ui?.sourceWithMultiReference);
  // sc-25908: the model's declared reference cap (`limits.maxReferenceAssets` — 10 for the Qwen
  // Image 2.1 family) on the multi-reference picker, which replaces the source, so the count is
  // exactly the list sent. Null for a model that declares none (FLUX.2-dev): no cap is shown and
  // the enqueue gate stays the only bound, as before.
  // Counted over DISTINCT ids, as the enqueue gate counts them (`qwen_image_2_1_reference_ids`
  // de-duplicates, first occurrence kept) — a restored recipe can carry a repeated id, and blocking a
  // request the API accepts would be a client stricter than the gate.
  const maxStudioReferences = multiReference ? maxReferencesForModel(selectedModel, null) : null;
  const distinctReferenceCount = mossEdit ? referenceAssetIds.length : new Set(referenceAssetIds).size;
  let mossEditError = "";
  if (mossEdit) {
    try {
      mossEditWireOptions(mossImageEdit);
      mossEditReferenceIds({ referenceAssetIds });
      mossEditError = mossEditControlsError({ steps: stepsOverride, guidanceScale: guidanceOverride, schedulerShift, max_sequence_length: mossImage.max_sequence_length ?? 4096, training_shift: mossImage.training_shift ?? 1 });
      mossEditError ||= mossAdvancedIssue(mossImageAdvanced, { assets, upscaleEnabled, prompt, prompts: mossPromptBatch, negativePrompt, count, width: widthOverride || Number(resolution.split("x")[0]), height: heightOverride || Number(resolution.split("x")[1]), context: mossImage.max_sequence_length ?? 4096 });
      if (referenceAssetIds.some((id) => !editImageAssets.some((asset) => asset.id === id))) mossEditError = "A MOSS reference is missing from this project. Reacquire that exact slot before editing.";
    } catch (error) { mossEditError = error.message; }
  }
  const referenceOverflowError =
    mode === "edit_image" && maxStudioReferences != null && distinctReferenceCount > maxStudioReferences
      ? `${selectedModel?.name ?? "This model"} takes up to ${maxStudioReferences} reference images — remove ${
          distinctReferenceCount - maxStudioReferences
        }.`
      : null;
  // Krea-style two-reference edit (epic 10871 P1.3): a model whose `ui.editReferences` adds an optional
  // SECOND source to the single-source edit — any two images, image 1 (required) + image 2 (optional),
  // fixed order. Only in single-source edit mode (never alongside the flat `multiReference`
  // multi-select). Null → the plain single-source edit for every other model/mode.
  const editReferences =
    mode === "edit_image" && !multiReference ? (selectedModel?.ui?.editReferences ?? null) : null;
  // The ordered [image1, image2] pair, sent as `referenceAssetIds` when a second image is chosen;
  // null → the single `sourceAssetId` path. Image 1 is required too (a second image with no first is
  // meaningless).
  const editSecondPair =
    editReferences && sourceAssetId && editSecondAssetId ? [sourceAssetId, editSecondAssetId] : null;
  // Drop a stale second-image selection when the two-reference edit surface goes away (model/mode
  // change), so it can never leak into a payload for a model that doesn't support it.
  useEffect(() => {
    if (!editReferences) {
      setEditSecondAssetId("");
    }
  }, [editReferences]);
  // Mac UI gating (sc-3486): disable the per-model feature controls the selected model can't run
  // in the Rust/MLX flow on Mac, so the user never reaches a `mlx_unsupported` error after submit.
  const macEditBlock = macModelFeatureBlock(selectedModel, macCapabilities, "edit");
  const macReferenceBlock = macModelFeatureBlock(selectedModel, macCapabilities, "reference");
  const macPoseBlock = macModelFeatureBlock(selectedModel, macCapabilities, "pose");
  const macActiveModeBlock = (() => {
    if (mode === "edit_image") return macEditBlock;
    if (mode === "character_image") return macReferenceBlock;
    return null;
  })();
  const macModeTabBlock = (value) => {
    if (!macGating || value === mode || modelsForMode(value).length) return null;
    return {
      blocked: true,
      text: "No available Mac model supports this mode.",
    };
  };
  // Variation slider spec (FLUX / Qwen). When declared, the model exposes a
  // trueCfgScale knob alongside (FLUX) or instead of (Qwen, via hideReferenceStrength)
  // the IP-Adapter reference-strength slider (sc-2017).
  const variationStrength = selectedModel?.ui?.variationStrength;
  const hideReferenceStrength = Boolean(selectedModel?.ui?.hideReferenceStrength);
  // Structured JSON-caption surface (Ideogram 4, epic 4725). When the model
  // declares `structuredPrompt`, the prompt hero swaps the plain textarea for the
  // builder and the engine receives the canonically-ordered JSON caption string.
  const structuredPromptModel = Boolean(selectedModel?.structuredPrompt);
  // Booru-tag models (Anima, Illustrious) declare `captionStyle: "tags"` — they are trained on
  // comma-separated tags, and their own promptHint warns that "a plain sentence renders low-effort
  // art". Every Style Catalog entry is a 600–900-char English prose paragraph, so the axis does not
  // fit them and the picker is hidden.
  //
  // The gate is DERIVED, never a state clear. Two reasons:
  //  - The user's pick survives passing through a tag model and comes back when they switch to a
  //    prose model, matching how the picker behaves across every other model change.
  //  - A render-time gate covers the paths a [model]-effect clear cannot: a localStorage restore and
  //    a recipe replay both seed styleId with no model change, and on a fresh mount `selectedModel`
  //    is briefly undefined while the catalog resolves (sc-11962 / sc-12034).
  // Downstream consumers read `effectiveStyleId`, so the axis cannot leak into a submit.
  const tagConventionModel = selectedModel?.captionStyle === "tags";
  const styleAxisAvailable = !tagConventionModel;
  const effectiveStyleId = styleAxisAvailable ? styleId : null;
  const appearance = useAppearance({ model: selectedModel, initial: saved.appearance, enabled: styleAxisAvailable && !structuredPromptModel && !(mossEdit && (mossImageAdvanced?.positive || mossImageAdvanced?.negative || mossImageAdvanced?.negativePrompts?.length)), prompt });
  const { setSelection: setAppearance } = appearance;
  // sc-13366: when a style is selected, replace the generic scene/character suggestion pills with a
  // single hint pill carrying that style's tailored subject prompt (styleThumbnailPrompts.json) — a
  // strong, style-fitting starting point. Falls back to the normal suggestions with no style.
  const styleHint = styleHintForId(effectiveStyleId);
  const suggestions = styleHint
    ? [styleHint]
    : mode === "character_image"
      ? characterSuggestions
      : sceneSuggestions;
  const captionValidation = useMemo(
    () => (structuredPromptModel ? validateCaption(caption, { plainText: prompt }) : null),
    [structuredPromptModel, caption, prompt],
  );
  // Structured mode is active when a structured model is selected and the user
  // isn't in the plain-text fallback tab.
  const structuredActive = structuredPromptModel && promptMode !== "plain";
  // A non-empty caption: at least a high-level description, a background, or one
  // element carrying content — guards Generate against the empty-but-valid skeleton.
  const captionHasContent = useMemo(() => {
    if (!structuredActive) return false;
    const cd = caption?.compositional_deconstruction ?? {};
    if (String(caption?.high_level_description ?? "").trim()) return true;
    if (String(cd.background ?? "").trim()) return true;
    return (Array.isArray(cd.elements) ? cd.elements : []).some(
      (el) => (el?.desc && String(el.desc).trim()) || (el?.type === "text" && el?.text && String(el.text).trim()),
    );
  }, [structuredActive, caption]);
  // Reset the reference tuning to the selected model's declared defaults whenever the
  // model changes, so InstantID starts at its tuned 0.8/0.8 and Kolors at 0.6, and the
  // view angle never carries over to a model that doesn't support it.
  //
  // Tracks the PREVIOUS model VALUE rather than a one-shot bool (sc-11962): a one-shot is
  // consumed on the mount pass, so a later async model "snap" (catalog arriving) would be
  // mistaken for a user model change and reset the restored tuning. Keying on the actual
  // value means an async catalog that leaves the model unchanged never fires the reset.
  // Seeded from the model when restoring (so the restored tuning survives mount) and null
  // otherwise (so a fresh mount still gets the model's declared defaults). `skip*` stays
  // for the recipe path, which sets the model AND the tuning together.
  const skipReferenceTuningReset = useRef(false);
  const referenceTuningModel = useRef(saved.ipAdapterScale != null ? model : null);
  // One declared-default policy for a model switch and the first late catalog resolution.
  // The effects retain their distinct ownership/skip gates and model-only transient resets.
  const applyDeclaredReferenceDefaults = useCallback((ui) => {
    setIpAdapterScale(typeof ui.referenceStrengthDefault === "number" ? ui.referenceStrengthDefault : 0.6);
    setControlnetScale(typeof ui.identityStructure?.default === "number" ? ui.identityStructure.default : 0.8);
    setTrueCfgScale(typeof ui.variationStrength?.default === "number" ? ui.variationStrength.default : 4.0);
    setControlScale(typeof ui.controlScale?.default === "number" ? ui.controlScale.default : null);
    setTextStyleGain(typeof ui.textStyleGain?.default === "number" ? ui.textStyleGain.default : 1.0);
  }, [setControlScale]);
  useEffect(() => {
    if (referenceTuningModel.current === model) {
      return;
    }
    referenceTuningModel.current = model;
    if (skipReferenceTuningReset.current) {
      skipReferenceTuningReset.current = false;
      return;
    }
    const ui = imageModels.find((item) => item.id === model)?.ui ?? {};
    applyDeclaredReferenceDefaults(ui);
    setViewAngle("");
    setSelectedPoseIds([]);
    setWorkflowPoses([]);
    setPoseLibraryReplayOpen(false);
    // Re-gate the strict-control panel to the new backbone: snap the control type to a supported mode
    // (an unsupported pick — e.g. canny on a pose-only backbone — resets to the first supported one),
    // reset the control-scale to the model's manifest default, and clear a stale control image.
    const nextModes = supportedControlModes(imageModels.find((item) => item.id === model));
    setControlMode((current) => (nextModes.includes(current) ? current : nextModes[0] ?? "pose"));
    setControlImageAssetId("");
    setControlImagePassthrough(false);
    // Clear a stale overlay pick so an id trained for a different backbone can't leak into a submit
    // (sc-10165 B4); the picker re-fetches for the new backbone.
    setControlOverlayId(null);
    // Model identity owns this reset; a catalog refresh must not erase in-progress tuning.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);
  // sc-12034: On a FRESH mount (no restored snapshot) whose model catalog arrives AFTER mount, the
  // reset effect above ran once with an empty `ui` (the catalog hadn't surfaced the model yet) and
  // settled the reference-tuning knobs at the model-agnostic fallbacks (0.6 / 0.8 / 4.0). Because
  // the fallback model id is a valid installed model that never changes, `model` never changes and
  // that effect never re-fires — so the model's DECLARED defaults are never applied. Re-apply them
  // ONCE, the first time the selected model actually resolves in the catalog.
  //
  // Fresh-mount ONLY: a restored snapshot seeds this disarmed (its tuning is authoritative and must
  // survive, mirroring the `referenceTuningModel` seed above), and the recipe path disarms it too
  // (it injects its own tuning). Mirrors the declared-default writes above; the clears
  // (viewAngle / poses / control image) are omitted because on a fresh mount they are already at
  // their initial values and the model never changed, so there is nothing stale to clear.
  const referenceTuningDeclaredArmed = useRef(saved.ipAdapterScale == null);
  useEffect(() => {
    if (!referenceTuningDeclaredArmed.current) {
      return;
    }
    const resolved = imageModels.find((item) => item.id === model);
    if (!resolved) {
      return; // Catalog hasn't surfaced this model yet — wait for it to arrive.
    }
    referenceTuningDeclaredArmed.current = false;
    if (skipReferenceTuningReset.current) {
      // A recipe/preset injection is applying its own tuning in this same commit — don't override it.
      return;
    }
    const ui = resolved.ui ?? {};
    applyDeclaredReferenceDefaults(ui);
  }, [model, imageModels, applyDeclaredReferenceDefaults]);
  // Approved reference images for the selected character (the IP-Adapter identity
  // source). Resolve the full asset from the catalog so thumbnails render even when
  // the character payload only carries assetIds.
  const characterReferences = useMemo(() => {
    const character = characters.find((item) => item.id === characterId);
    return (character?.approvedReferences ?? []).map((reference) => ({
      assetId: reference.assetId,
      role: reference.role ?? null,
      asset: reference.asset ?? assets.find((item) => item.id === reference.assetId) ?? null,
    }));
  }, [characters, characterId, assets]);
  // Keep the selected reference valid: default to the first approved reference when
  // none is chosen or the current one no longer belongs to this character.
  useEffect(() => {
    if (mode !== "character_image" || (isFlux3LayoutModel(model) && orderedCharacterReferences)) {
      return;
    }
    if (characterReferences.some((reference) => reference.assetId === referenceAssetId)) {
      return;
    }
    changeReferenceAssetId(characterReferences[0]?.assetId ?? "", false);
  }, [mode, model, orderedCharacterReferences, characterReferences, referenceAssetId, changeReferenceAssetId]);
  // Seed a character-appropriate default prompt when entering character mode, unless
  // the user has already typed/picked their own. The generic text-to-image default
  // ("neon street at midnight") makes no sense for character variations.
  useEffect(() => {
    if (mode !== "character_image" || !characterId || promptEdited.current) {
      return;
    }
    const character = characters.find((item) => item.id === characterId);
    if (character) {
      setPrompt(defaultCharacterPrompt(character));
    }
  }, [mode, characterId, characters]);
  // Seed the model's curated default negative prompt into an EMPTY negative box (sc-3857, sc-10760).
  // Originally character-mode only (InstantID/RealVisXL declares one to fight its shiny/over-saturated
  // look); now also text-to-image, so booru models (Anima, Illustrious) get their booru negative there
  // too — a bare negative was a big reason their anime renders looked worse. Only fills an empty box, so
  // it never clobbers a typed, restored, or preset negative.
  useEffect(() => {
    if (!seedsNegativeInMode(mode) || negativePrompt !== "") {
      return;
    }
    const entry = imageModels.find((item) => item.id === model);
    // An engine with no negative-prompt axis (sc-15299) never gets one seeded: the box is hidden
    // for it and the value is not sent, so seeding would only plant a ghost the user cannot see.
    // Read off the entry rather than the `supportsNegativePrompt` const below — that binding is
    // declared later in this component body, so touching it from a dep array would be a TDZ hit.
    if (entry?.image?.supportsNegativePrompt === false) {
      return;
    }
    const ui = entry?.ui ?? {};
    if (typeof ui.defaultNegativePrompt === "string" && ui.defaultNegativePrompt) {
      setNegativePrompt(ui.defaultNegativePrompt);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, model]);
  // Seed the model's booru quality prefix (`ui.defaultPrompt`) into an UNEDITED prompt box in
  // text-to-image (sc-10760). Anima/Illustrious are danbooru-tag models that render low-effort art from a
  // bare sentence; opening with `masterpiece, best quality,` and building on it is what their model cards
  // recommend. Mirrors the character-mode prompt seed: only when `!promptEdited`, so it replaces the
  // throwaway scene default, never the user's own wording. A model WITHOUT a defaultPrompt restores the
  // generic scene default, so a stale prefix never lingers after switching to a non-booru model.
  useEffect(() => {
    if (mode !== "text_to_image" || promptEdited.current) {
      return;
    }
    const ui = imageModels.find((item) => item.id === model)?.ui ?? {};
    setPrompt(promptSeedFor(ui));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, model]);
  // The model's advertised buckets, MEMORY-GATED for this host (sc-13959): a resolution above the
  // historical 1536² ceiling is hidden when its predicted peak won't fit `unifiedMemoryGb` (unified
  // memory on a Mac / GPU VRAM off Mac) on the active backend — so a 48 GB Mac isn't offered a 2048²
  // that OOMs while a 128 GB Mac gets the full range. A no-op for ≤1536²-only models, an unknown
  // memory reading, and models with no declared memory floor (see resolutionMemory.js). The
  // downstream snap/default effects keep the selection valid within the gated list.
  const resolutionOptions = useMemo(() => {
    const declared = selectedModel?.limits?.resolutions?.length
      ? selectedModel.limits.resolutions
      : DEFAULT_RESOLUTION_OPTIONS;
    const gated = fitsResolutionOptions(selectedModel, declared, unifiedMemoryGb, {
      backend: activeBackend,
      tier: quantTier,
    });
    // Never collapse to an empty picker: if the gate somehow trims everything (it never trims
    // ≤1536², so this is defensive), fall back to the declared list.
    return gated.length > 0 ? gated : declared;
  }, [selectedModel, unifiedMemoryGb, activeBackend, quantTier]);
  // Reference-image auto-preset (sc-8109, epic 8102): when a captioning reference
  // image's natural dimensions become known, snap the resolution picker to whichever
  // option best matches its aspect ratio. The caption's bboxes are normalized 0–1000
  // to the FRAME, so a reference grounded for (say) 4:5 but rendered at 16:9 comes out
  // wrong-shaped — matching the aspect keeps the captured composition valid. This is a
  // plain seam: the reference-image upload handler (the picker UI itself lands in
  // sc-8108) calls onReferenceImageLoaded(width, height) once the image has loaded.
  // setResolution still leaves the picker fully user-overridable.
  const onReferenceImageLoaded = useCallback(
    (referenceWidth, referenceHeight) => {
      const match = pickClosestResolution(referenceWidth, referenceHeight, resolutionOptions);
      if (match) setResolution(match);
    },
    [resolutionOptions],
  );
  // The reference snap's dedupe key, shared with the describe picker. It names which model's option
  // UNIVERSE we are in — deliberately the DECLARED buckets, not the memory-gated `resolutionOptions`
  // above. The gate also varies with the picked quant tier (sc-16020), and a tier switch re-snapping
  // over the user's Aspect is the very clobber sc-18255 fixes. If the gate ever trims the selected
  // bucket, the validity effect below re-picks it. A VALUE, not an identity: `selectedModel`'s
  // identity churns on every model-catalog refetch (see useReferenceAspectSnap).
  const resolutionSnapKey = useMemo(
    () => resolutionUniverseKey(selectedModel, DEFAULT_RESOLUTION_OPTIONS),
    [selectedModel],
  );
  // Auto-preset the Aspect from the picked img2img reference (sc-10195), through the same guarded
  // hook the describe picker uses. Gated on the SAME condition that renders the tile (below), because
  // `img2imgReferenceAssetId` is never cleared on a model or mode change: without the gate, a
  // reference armed on Krea would keep steering the Aspect on a model that has no img2img panel at
  // all. This rides `enabled`, NOT an emptied id — an emptied id re-arms the snap, which would
  // re-clobber the Aspect on a mode round-trip (Text → Edit → Text with the reference still armed).
  const img2imgSnapEligible = supportsImg2img && mode === "text_to_image";
  useReferenceAspectSnap({
    referenceId: img2imgReferenceAssetId,
    assets: editImageAssets,
    onReferenceImageLoaded,
    snapKey: resolutionSnapKey,
    enabled: img2imgSnapEligible,
  });
  // Sampler / scheduler menus declared by the model, gated to the ACTIVE backend
  // (epic 7114 P5): `macGatingActive` is the worker `mlx_required` master switch, so
  // it picks the manifest's `mlx.limits` override on Mac/MLX and the `candle.limits`
  // override on the Windows/Linux candle build (e.g. Lens exposes the curated menu
  // only on candle; SDXL only on MLX). The advanced panel hides the dropdowns when
  // the menu has fewer than 2 options (epic 1753 §7.4).
  const { samplerOptions, schedulerOptions, samplerDefault, schedulerDefault, showSchedulerPicker } = useSamplerControls(selectedModel, activeBackend, { sampler, scheduler, setSampler, setScheduler, emptyValue: "default" });

  const guidanceMethodOptions = useMemo(
    () => guidanceMethodOptionsFromModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );

  const showGuidanceMethodPicker = guidanceMethodOptions.length > 1;
  // Which generation axes the selected engine actually has, declared in the manifest `image`
  // sub-block (sc-15299) — the image-lane sibling of the `video` block sc-8445 added for Krea
  // Realtime, and of `audio.supportsGuidance` / `audio.supportsNegativePrompt`. Neither an id check
  // nor an engine link: the catalog says it, mirroring the engine descriptor's own
  // `Capabilities { supports_guidance, supports_negative_prompt }` that the worker's
  // `resolve_guidance` / `resolve_negative_prompt` already gate on (image_jobs/base.rs) — a
  // guidance-distilled engine drops both on the floor, so offering them is a knob that does nothing.
  //
  // ⚠️ ABSENT MEANS TRUE, the same polarity as `video` and the OPPOSITE of `audio`. Most image
  // engines take both a CFG scale and a negative prompt and declare nothing, so `!== false` is what
  // keeps them byte-identical to their pre-sc-15299 behaviour; only an engine that genuinely lacks
  // the axis declares it false.
  //
  // HIDDEN, not disabled: the Wan Lightning precedent in Video Studio DISABLES Steps/Guidance
  // because that inertness is TRANSIENT (turn Lightning off and the control works again). A missing
  // axis is permanent for the model, so a forever-dead input is just clutter — Audio Studio hides
  // these two for the same reason (`showGuidance` / `showNegative`).
  const supportsGuidance = selectedModel?.image?.supportsGuidance !== false;
  const supportsNegativePrompt = selectedModel?.image?.supportsNegativePrompt !== false;
  // sc-25679: a model whose negative prompt IS the CFG unconditional
  // (`image.negativePromptRequiresGuidance`, Iris-3B) refuses one with CFG off. The field stays
  // visible (the axis exists — raise Guidance and it applies) but is DISABLED and not sent while
  // the effective guidance is 1.0: a transient inertness, the Wan-Lightning precedent above.
  const negativePromptCfgOff = negativePromptInertAtGuidance(selectedModel, guidanceOverride);
  // sc-25681: the remaining declared sampling controls. A schedule-shift axis without a scheduler
  // menu shows the Schedule shift control on its own; the CFG interval is disabled (and not sent)
  // while CFG is off, exactly like the CFG-bound negative prompt; the caption-overflow menu only
  // appears for a model that offers policies.
  const supportsSchedulerShift = selectedModel?.image?.supportsSchedulerShift === true;
  // `null` = no explicit value yet (a fresh workspace, or a recipe that recorded none): the model's
  // own default, which for a shift-axis model (Iris-3B) is its checkpoint's flow shift — not the
  // historical blanket 3.0, which would be SENT for it and silently change the schedule.
  const effectiveSchedulerShift = schedulerShift ?? schedulerShiftDefaultFromModel(selectedModel);
  const showSchedulerShift = supportsSchedulerShift || (showSchedulerPicker && scheduler !== "default");
  const supportsCfgInterval = supportsGuidance && selectedModel?.image?.supportsCfgInterval;
  const cfgIntervalDisabled = guidanceIsOff(selectedModel, guidanceOverride);
  const { interval: resolvedCfgInterval, error: cfgIntervalError } = resolveCfgInterval({
    model: supportsGuidance ? selectedModel : null,
    text: cfgInterval,
    guidanceOverride,
  });
  const captionOverflowChoices = selectedModel?.image?.captionOverflowPolicies ?? [];
  const outgoingCaptionOverflow = captionOverflowChoices.includes(captionOverflow)
    ? captionOverflow
    : "default";
  const hiresFixModelCapable = Boolean(
    supportsImg2img || selectedModel?.family === "sdxl",
  );
  const hiresFixAccelerationBlocked =
    model === "realvisxl_lightning" || ["lightning", "lcm", "hyper"].includes(sampler);
  const hiresFixInputBlocked =
    Boolean(img2imgReferenceAssetId) ||
    posePayload.length > 0 ||
    Boolean(controlPreprocessSourceId) ||
    Boolean(controlPassthroughId) ||
    (multiPhaseEnabled && modelSupportsMultiPhase(selectedModel));
  const hiresFixAvailable =
    mode === "text_to_image" && hiresFixModelCapable && !hiresFixAccelerationBlocked;
  const hiresFixDisabled =
    !hiresFixAvailable || usePid || hiresFixInputBlocked;
  const hiresFixDisabledReason = !hiresFixModelCapable
    ? "The selected model does not support the img2img second pass required by Hires.fix."
    : mode !== "text_to_image"
      ? "Hires.fix is only available for text-to-image generation."
      : hiresFixAccelerationBlocked
        ? "Lightning, LCM, and Hyper samplers do not accept the required img2img second pass."
        : usePid
          ? "Disable PiD before enabling Hires.fix."
          : hiresFixInputBlocked
            ? "Remove the starting image, strict control, or multi-phase denoise before enabling Hires.fix."
            : undefined;
  useEffect(() => {
    // Keep restored state while the model catalog is still loading. Once the selected model is
    // resolved, incompatible modes/samplers/inputs turn the option off instead of submitting a
    // payload the worker must reject.
    if (
      hiresFixEnabled &&
      selectedModel &&
      (!hiresFixAvailable || hiresFixInputBlocked)
    ) {
      setHiresFixEnabled(false);
    }
  }, [
    hiresFixEnabled,
    selectedModel,
    hiresFixAvailable,
    hiresFixInputBlocked,
  ]);
  const advancedDefaultsModel = useRef(model);
  const skipAdvancedDefaultsReset = useRef(false);

  useEffect(() => {
    if (advancedDefaultsModel.current === model) {
      return;
    }
    const previousModel = advancedDefaultsModel.current;
    advancedDefaultsModel.current = model;
    if (skipAdvancedDefaultsReset.current) {
      skipAdvancedDefaultsReset.current = false;
      return;
    }
    if (mossEdit && isMossImage(previousModel)) return;
    setMossImage(mossDefaults());
    setMossDmdSteps(8);
    setSampler(samplerDefault);
    setScheduler(schedulerDefault);
    setSchedulerShift(schedulerShiftDefaultFromModel(selectedModel));
    setCfgInterval("");
    setCaptionOverflow("default");
    setGuidanceMethod(
      preferredOption(guidanceMethodDefaultFromModel(selectedModel), guidanceMethodOptions, [], { empty: "default" }),
    );
    setStepsOverride("");
    setGuidanceOverride("");
  }, [model, resolutionOptions, samplerOptions, schedulerOptions, samplerDefault, schedulerDefault, guidanceMethodOptions, selectedModel, mossEdit, setMossImage, setMossDmdSteps, setSampler, setScheduler, setSchedulerShift, setStepsOverride, setGuidanceOverride]);
  useMossImageTransition({
    model,
    mossEditTransition,
  mossEdit,
  mossImage,
  resolution,
  stepsOverride,
  guidanceOverride,
  schedulerShift,
  negativePrompt,
  widthOverride,
  heightOverride,
  setResolution,
  setStepsOverride,
  setGuidanceOverride,
  setSchedulerShift,
  setWidthOverride,
  setHeightOverride,
  setMossImageEdit,
  setMossImage,
  setNegativePrompt
  });

  // Snap the guidance method back to "cfg" when the current choice isn't honored by
  // the active backend for this model (e.g. switching off the SDXL family drops
  // CFG++) — the N3 guard at the UI layer, so an unsupported method is never sent.
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (guidanceMethodOptions.includes(guidanceMethod)) {
      return;
    }
    setGuidanceMethod(
      preferredOption(guidanceMethodDefaultFromModel(selectedModel), guidanceMethodOptions, [], { empty: "default" }),
    );
  }, [guidanceMethodOptions, guidanceMethod, selectedModel]);
  // A sticky or replayed `true` must not survive a route change that hides the control. This keeps
  // visible state, persisted state, and the submit-time gate aligned instead of silently carrying an
  // old request into a backend/model that will reject it.
  useEffect(() => {
    if (!promptEnhance && enhancePrompt) {
      setEnhancePrompt(false);
    }
  }, [enhancePrompt, promptEnhance]);
  // Preserve resolution across models; if unavailable, choose the closest supported
  // aspect ratio and size instead of resetting the composition to the model default.
  // Guard on a resolved model (sc-11962): before the catalog loads `resolutionOptions`
  // falls back to DEFAULT_RESOLUTION_OPTIONS, so an un-guarded snap would revert a
  // restored model-specific resolution (e.g. 1536x1536) during restart-restore.
  //
  // Exception: a FLUX 3 automatic-reference-sized recipe records the reference plan's
  // resolved canvas (for example 80x64), not a user aspect choice. Snapping that value to
  // the closest bucket would silently change the requested composition, and the automatic
  // intent already owns the emitted axes. Fall back to the model's preferred bucket there.
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (resolutionOptions.includes(resolution)) {
      return;
    }
    const autoReferenceIntent = referenceDimensionIntent?.model === model &&
      referenceDimensionIntent?.mode === mode &&
      (referenceDimensionIntent.width === null || referenceDimensionIntent.height === null);
    const dims = !autoReferenceIntent && parseResolution(resolution);
    setResolution(
      (dims && pickClosestResolution(dims.width, dims.height, resolutionOptions))
        || preferredResolution(selectedModel, resolutionOptions),
    );
  }, [resolutionOptions, resolution, selectedModel, referenceDimensionIntent, model, mode]);
  // Keep the selected quant tier valid for the active model (sc-8515). When the current tier is
  // still installed for this model, leave it; otherwise snap to the model's default selection
  // (sticky-for-this-(screen,model) → declared default → q8 base → q4 → first installed). Clears "" when
  // no tier is installed / the model has no matrix, so a stale tier never leaks into the payload.
  // The sticky rung (sc-10727) is read straight from the persistent per-(screen,model) store, so it
  // survives restarts and is honored above the base default whenever that tier is still installed.
  // Keyed on `model` (not `selectedModel`) plus the installed-tier list so a catalog refresh that
  // newly installs a second tier re-derives the default without churning on every render.
  const {
    availablePresets,
    selectedPreset,
    selectedPresetId,
    setSelectedPresetId,
    availableGeneralPresets,
    generalStack,
    generalStackIds,
    toggleGeneralPreset,
    presetPromptParts,
    presetLoraDetails,
    presetValidationResult,
    localJobs,
    selectedLoraIds,
    setSelectedLoraIds,
    loraWeights,
    setLoraWeights,
    compatibleLoras,
    selectedLoras,
    userSelectedLoraCount,
    selectedLoraValidationResult,
    loraEmptyMessage,
    toggleLora,
    effectiveLoraWeight,
    setLoraWeight,
  } = useGenerationStudio({
    mode,
    presets,
    selectedModel,
    loras,
    models: imageModels,
    model,
    setModel,
    fallbackModelId: "z_image_turbo",
    characters,
    characterId,
    setCharacterId: changeCharacterId,
    setCharacterLookId,
    assets,
    latestAssets,
    trackedLocalJobs,
    initialPresetId: saved.selectedPresetId ?? null,
    advancedOpen,
    setAdvancedOpen,
    initialSelectedLoraIds: saved.selectedLoraIds ?? [],
    initialLoraWeights: saved.loraWeights ?? {},
    initialGeneralStackIds: saved.generalStackIds ?? [],
  });

  // The effective generation inputs once the general-preset stack is folded in (epic 11949).
  // The live preview shows exactly this; a client-authoritative submit sends it (Phase 5).
  const composedStack = useMemo(
    () =>
      composePreset({
        base: selectedPreset,
        generalStack,
        userText: prompt,
        userNegative: negativePrompt,
        resolutionOptions,
      }),
    [selectedPreset, generalStack, prompt, negativePrompt, resolutionOptions],
  );
  if (mossEdit && !mossEditError) mossEditError = mossPositivePromptIssue(mossImageAdvanced, {
    prompt, prompts: mossPromptBatch,
    effectivePrompt: composeImageJobPrompt({ promptToSend: composedStack.prompt, sendStructured: false, styleText: styleTextForId(effectiveStyleId), appearance: appearance.selection, model: selectedModel }).prompt,
  });
  const stackAddsNegative = generalStack.some((preset) => Boolean(preset?.defaults?.negativePrompt));
  const stackAddsCount = generalStack.some((preset) => Number.isFinite(Number(preset?.defaults?.count)));

  // ---- Krea-style image edit LoRA (epic 10871, P4.1) ----
  // The Krea 2 edit surface REQUIRES a dual-conditioning `image_edit` LoRA (R5) that the base
  // can't edit without. We MANAGE it for the user — no manual LoRA picking — rather than leave it
  // in the picker: auto-applied to the payload when installed, or surfaced as a one-click download
  // when not. `findModelEditLora` returns null for edit models that need no such LoRA (Qwen-Image-
  // Edit, FLUX.2), so this whole block is inert for them.
  const editLora = useMemo(
    () => (mode === "edit_image" ? findModelEditLora(loras, selectedModel) : null),
    [mode, loras, selectedModel],
  );
  const editLoraInstalled = loraIsInstalled(editLora);
  // The managed LoRA is applied automatically; hide it from the manual picker so it isn't
  // double-shown or accidentally toggled. Still deduped at payload time in case a saved selection
  // carries it.
  const managedEditLoraId = editLora && editLoraInstalled ? editLora.id : null;
  const editLoraRequiredMissing = Boolean(editLora) && !editLoraInstalled;
  const [editLoraDownloadRequested, setEditLoraDownloadRequested] = useState(false);
  // Clear the transient "requested" state once the download lands (installState flips) or the
  // edit LoRA leaves the picture (model/mode change).
  useEffect(() => {
    if (!editLoraRequiredMissing) {
      setEditLoraDownloadRequested(false);
    }
  }, [editLoraRequiredMissing]);
  const requestEditLoraDownload = useCallback(() => {
    if (!editLora) {
      return;
    }
    setEditLoraDownloadRequested(true);
    createLoraDownloadJob?.(editLora);
  }, [editLora, createLoraDownloadJob]);
  // Serialize the outgoing LoRA payload, appending the auto-applied (managed) edit LoRA unless a
  // saved selection already carries it — the worker's edit lane requires it (R5). Used by both the
  // single-generate and batch submit paths so they stay identical.
  const buildLorasPayload = useCallback(() => {
    const out = selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) }));
    if (managedEditLoraId && !out.some((lora) => lora.id === managedEditLoraId)) {
      out.push(serializeLora(editLora, { weight: effectiveLoraWeight(editLora) }));
    }
    return out;
  }, [selectedLoras, effectiveLoraWeight, managedEditLoraId, editLora]);
  // The manual LoRA picker hides the managed edit LoRA (it's applied automatically in the source
  // band above), so it can't be double-shown or accidentally toggled off.
  const pickerCompatibleLoras = managedEditLoraId
    ? compatibleLoras.filter((lora) => lora.id !== managedEditLoraId)
    : compatibleLoras;
  const pickerSelectedLoras = managedEditLoraId
    ? selectedLoras.filter((lora) => lora.id !== managedEditLoraId)
    : selectedLoras;
  const pickerSelectedLoraIds = managedEditLoraId
    ? selectedLoraIds.filter((id) => id !== managedEditLoraId)
    : selectedLoraIds;

  // ---- Krea 2 multi-phase denoise (epic 13879 S5, sc-13885) ----
  // Shown only for the model that advertises the "acceleration" LoRA compat (krea_2_raw, sc-13882)
  // AND in text-to-image mode — matching the worker gate (`request.model == krea_2_raw`, t2i only;
  // multi-phase renders from pure noise, so edit/pose/reference are rejected). `multiPhaseActive`
  // additionally requires the editor toggle to be on: only then does `advanced.phases` ride the
  // request, so every other job is byte-for-byte unchanged. The per-phase LoRA toggles reference the
  // job's OWN selected LoRAs (`buildLorasPayload` order == `request.loras` == the worker's
  // `LoadSpec::adapters`); serializePhases resolves each stored LoRA id to its current index there,
  // so the emitted `loras:[{ index, weight? }]` is always valid, reindexed on reorder, pruned on
  // deselect. Weighed against the SAME `selectedLoras` the payload maps, so the indices line up.
  const supportsMultiPhase = modelSupportsMultiPhase(selectedModel);
  const multiPhaseAvailable = supportsMultiPhase && mode === "text_to_image";
  const multiPhaseActive = multiPhaseAvailable && multiPhaseEnabled;
  const serializedPhases = useMemo(
    () => serializePhases(multiPhasePhases, selectedLoras),
    [multiPhasePhases, selectedLoras],
  );
  const multiPhaseValidationIssues = useMemo(
    () => computeMultiPhaseIssues({ enabled: multiPhaseActive, phases: multiPhasePhases }),
    [multiPhaseActive, multiPhasePhases],
  );
  const applyTurboFinishPreset = useCallback(() => {
    setMultiPhasePhases(buildTurboFinishPhases(selectedLoras));
    setMultiPhaseEnabled(true);
    setMultiPhaseOpen(true);
  }, [selectedLoras]);

  // sc-10516: a preset launch (Presets → "Use in Studio"). `availablePresets` filters on
  // mode + model, so the preset only resolves once both match — set them alongside the id.
  // Changing the model otherwise wipes the steps/guidance overrides (the advanced-defaults
  // reset effect above), which would clobber the very defaults the preset is about to
  // apply, so suppress that reset exactly as the recipe path does.
  // Kept separate from the recipe effect below, which clears the preset instead.
  useEffect(() => {
    if (launchRequest?.view !== "Image" || !launchRequest.presetId) {
      return;
    }
    if (IMAGE_MODES.includes(launchRequest.presetMode)) {
      setMode(launchRequest.presetMode);
    }
    if (launchRequest.presetModel && launchRequest.presetModel !== advancedDefaultsModel.current) {
      skipAdvancedDefaultsReset.current = true;
      setModel(launchRequest.presetModel);
    }
    setSelectedPresetId(launchRequest.presetId);
  }, [launchRequest, setSelectedPresetId]);
  // A general-preset launch (epic 11949): toggle it into the stack without touching the model
  // or mode. The chip is available in every studio, so a Video-bound user can re-add it there.
  useEffect(() => {
    if (launchRequest?.view !== "Image" || !launchRequest.presetGeneralId) {
      return;
    }
    if (!generalStackIds.includes(launchRequest.presetGeneralId)) {
      toggleGeneralPreset(launchRequest.presetGeneralId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [launchRequest?.id]);
  useImageRecipeReplay({
    model,
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
  });

  const [dropdownWidth, dropdownHeight] = resolution.split("x").map((value) => Number(value));
  // Per-model geometry constraints for the free Width/Height override (sc-14058). Native-resolution
  // families (Mage-Flow) declare a ÷16 stride and a 512–2048 native range; every other model keeps the
  // global 256–4096 envelope with no stride. Drives the override inputs' min/max/step affordance below.
  // Match the references actually emitted by buildImageJobRequest, not hidden/stale picker state.
  const jobReferences = imageJobReferences({ model, mode, multiReference, orderedCharacterReferences, sourceWithMultiReference,
    editSecondPair, sourceAssetId, controlPreprocessSourceId, referenceAssetIds, referenceAssetId,
    supportsImg2img, img2imgReferenceAssetId });
  const flux3LayoutAvailable = isFlux3LayoutModel(selectedModel);
  const layoutAssetIds = orderedLayoutAssetIds(jobReferences);
  const layoutReferences = flux3LayoutAvailable
    ? reconcileLayoutReferences(flux3ReferenceSlots, layoutAssetIds) : [];
  // Keep hidden model state intact. Reconcile only the active attachment sequence, without
  // deduplicating it. Updating our own state here avoids an effect/submit identity race.
  if (flux3LayoutAvailable && layoutAssetIds.length &&
      (layoutReferences.length !== flux3ReferenceSlots.length || layoutReferences.some((slot, index) =>
        slot.assetId !== flux3ReferenceSlots[index]?.assetId || slot.id !== flux3ReferenceSlots[index]?.id))) {
    setFlux3ReferenceSlots(layoutReferences);
  }
  // Stable change callbacks read the OLD complete sequence before their setter runs.
  // This snapshot also covers approved-reference fallback effects without changing effect cadence.
  referenceChangeState.current = { available: flux3LayoutAvailable, model, mode, references: layoutReferences,
    referenceAssetId, characterId, sourceAssetId, editSecondAssetId, img2imgReferenceAssetId };
  const orderedCharacterActive = flux3LayoutAvailable && mode === "character_image" && orderedCharacterReferences;
  const characterReferenceError = orderedCharacterActive && !referenceAssetIds.length
    ? "Select at least one reference image for this character request." : null;
  const changeReferenceAssetIds = (ids) => {
    if (flux3LayoutAvailable) {
      setFlux3Layout(bindLayoutReferenceIds(flux3Layout, layoutReferences));
      setFlux3ReferenceSlots(reconcileLayoutReferences(layoutReferences, ids));
    }
    setReferenceAssetIds(ids);
    if (flux3LayoutAvailable && mode === "character_image") {
      setOrderedCharacterReferences(true);
      setReferenceAssetId(ids[0] ?? "");
    }
  };
  const moveFlux3Reference = (from, to) => {
    setFlux3Layout(bindLayoutReferenceIds(flux3Layout, layoutReferences));
    setFlux3ReferenceSlots(moveLayoutReference(layoutReferences, from, to));
    const ids = moveLayoutReference(referenceAssetIds, from, to);
    setReferenceAssetIds(ids);
    if (orderedCharacterActive) setReferenceAssetId(ids[0] ?? "");
  };
  const removeFlux3Reference = (index) => {
    setFlux3Layout(bindLayoutReferenceIds(flux3Layout, layoutReferences));
    setFlux3ReferenceSlots(layoutReferences.filter((_, slotIndex) => slotIndex !== index));
    const ids = referenceAssetIds.filter((_, slotIndex) => slotIndex !== index);
    setReferenceAssetIds(ids);
    if (orderedCharacterActive) setReferenceAssetId(ids[0] ?? "");
  };
  const flux3ReferenceRequest = isFlux3ReferenceRequest(model, Object.values(jobReferences).flat());
  const activeDimensionIntent = flux3ReferenceRequest && referenceDimensionIntent?.model === model &&
    referenceDimensionIntent?.mode === mode ? referenceDimensionIntent : null;
  useEffect(() => {
    if (referenceDimensionIntent && (referenceDimensionIntent.model !== model ||
        referenceDimensionIntent.mode !== mode || (selectedModel && !flux3ReferenceRequest))) {
      setReferenceDimensionIntent(null);
    }
  }, [referenceDimensionIntent, model, mode, selectedModel, flux3ReferenceRequest]);
  // A non-empty Width/Height override wins for that axis; empty falls back to the Aspect
  // dropdown. The resulting dims flow through the existing top-level width/height payload,
  // so submit() and the batch builder need no further change. Logic lives in the pure,
  // unit-tested evaluateModelDimensions helper, which layers the model's range + stride on top.
  const dimensionEval = evaluateModelDimensions({
    model: selectedModel,
    resolution,
    widthOverride,
    heightOverride,
    referenceRequest: flux3ReferenceRequest,
    dimensionIntent: activeDimensionIntent,
  });
  const { width, height, invalid: dimensionsInvalid, constraints: dimensionConstraints } = dimensionEval;
  // A precise, user-facing ERROR (never a raw requirement): names this model's actual range/stride.
  const dimensionError = dimensionConstraintMessage(dimensionEval);
  // sc-25681: a model that shows its patch grid below a declared side (Iris-3B, 1024) says so
  // whenever the effective width or height is under it.
  const patchGridNotice = patchGridHint(selectedModel, width, height);
  // sc-24215: a fixed native size and a declared prompt-token budget (Supra2-IMG: 256×256, 128
  // Flan-T5 tokens) are stated where the user picks the size and writes the prompt.
  const nativeResolutionNotice = nativeResolutionHint(selectedModel);
  const promptTokenNotice = promptTokenLimitHint(selectedModel);
  const hiresFixTargetWidth = Math.round(width * hiresFixUpscaleBy);
  const hiresFixTargetHeight = Math.round(height * hiresFixUpscaleBy);
  const hiresFixTargetInvalid =
    hiresFixEnabled && (hiresFixTargetWidth > 4096 || hiresFixTargetHeight > 4096);
  const hiresFixTargetMessage = hiresFixTargetInvalid
    ? `Hires.fix target ${hiresFixTargetWidth}×${hiresFixTargetHeight} exceeds the 4096px image limit.`
    : "";

  // PiD high-res decode heads-up (sc-10144): PiD super-resolves the base render 4×, so a large base at
  // the default 4K tier is a multi-minute (auto-tiled above 4096², sc-10087) decode that can look hung.
  // Surfaced inline under the PiD output tier so the user knows the long decode is progressing, not
  // stuck. null when PiD is off, on the fast 2K tier, or below the multi-minute threshold.
  const pidDecodeNotice = pidDecodeHeadsUp({ usePid, pidTarget, width, height });

  // Magic-prompt expansion (sc-5997): expand the plain-text idea into an editable caption via the
  // native utility model (same backend as Refine), recording which model drafted it. Returns the
  // cleaned caption (aspect_ratio + bboxes stripped); the builder applies it and switches to the
  // form. Only wired when a structured model is selected.
  const magicModelMissing = refineModel?.installState === "missing";
  const {
    onMagicExpand,
    onImageCaption,
    onImageDescribe
  } = useStructuredPrompt({
    model,
    magicPrompt,
  gcd,
  width,
  height,
  setMagicPromptBackend,
  imageCaption,
  activeProject,
  selectedModel,
  imageDescribe
  });

  // Save-as-Preset + the preset-default hydrate pass (sc-8937 — shared with the Video
  // studio via useSavePreset). The [key, setter] pairs are restored through the
  // remember/clear snapshot machinery, so switching to None (or another preset) puts
  // the user's prior value back. Only keys the preset actually carries are applied, so
  // older presets (which only stored count/resolution/negativePrompt) keep working and
  // full-snapshot presets restore the prompt, cfg, sampler, reference + upscale knobs.
  // The model is intentionally absent — presets never switch the model.
  const {
    presetName,
    setPresetName,
    presetScope,
    setPresetScope,
    savingPreset,
    presetSaveMessage,
    setPresetSaveMessage,
    handleSaveAsPreset,
  } = useSavePreset({
    saved,
    selectedPreset: moss ? mossPresetWithOrdinarySettings(selectedPreset) : selectedPreset,
    presetApplicationId: isMossImage(launchRequest?.presetModel) && launchRequest?.presetId ? launchRequest.id : null,
    setSelectedPresetId,
    presets,
    mode,
    model,
    selectedLoras,
    effectiveLoraWeight,
    createPreset,
    activeProject,
    setMode,
    presetDefaultFields: [
      ["mossImage", setMossImage],
      ["mossImageEdit", setMossImageEdit],
      ["mossImageAdvanced", setMossImageAdvanced],
      ["mossPromptBatch", setMossPromptBatch],
      ["mossDmdSteps", setMossDmdSteps],
      ["prompt", setPrompt],
      ["negativePrompt", setNegativePrompt],
      ["resolution", setResolution],
      ["count", setCount],
      ["guidanceScale", setGuidanceOverride],
      ["steps", setStepsOverride],
      ["sampler", setSampler],
      ["scheduler", setScheduler],
      ["schedulerShift", setSchedulerShift],
      [
        "cfgInterval",
        (value) => setCfgInterval(value?.join?.(", ") ?? ""),
      ],
      ["captionOverflow", setCaptionOverflow],
      ["guidanceMethod", setGuidanceMethod],
      ["textEncoderModel", setTextEncoderModel],
      ["ipAdapterScale", setIpAdapterScale],
      ["img2imgStrength", setImg2imgStrength],
      ["textStyleGain", setTextStyleGain],
      ["controlnetScale", setControlnetScale],
      ["trueCfgScale", setTrueCfgScale],
      ["viewAngle", setViewAngle],
      ["upscaleEnabled", setUpscaleEnabled],
      ["upscaleFactor", setUpscaleFactor],
      ["upscaleEngine", setUpscaleEngine],
      ["upscaleSoftness", setUpscaleSoftness],
      ["upscaleRestore", (value) => setUpscaleRestore(irisRestoreControls(value))],
      ["hiresFixEnabled", setHiresFixEnabled],
      ["hiresFixSteps", setHiresFixSteps],
      ["hiresFixDenoisingStrength", setHiresFixDenoisingStrength],
      ["hiresFixUpscaleBy", setHiresFixUpscaleBy],
      ["hiresFixCfgScale", setHiresFixCfgScale],
    ],
    // Restore the saved sub-mode ("type"). Edit presets only surface in edit mode, so
    // this only ever flips between text/character within one workflow.
    modeIsPresetable: (savedMode) => IMAGE_MODES.includes(savedMode),
    onApplyDefaults: (defaults) => {
      // Filling the prompt box counts as a user edit, so character mode's default
      // prompt won't clobber the restored prompt.
      if (Object.prototype.hasOwnProperty.call(defaults, "prompt")) {
        promptEdited.current = true;
      }
      // sc-12034: a saved preset that carries its own reference-tuning (a character-mode preset
      // stores ipAdapterScale / controlnetScale / trueCfgScale — see buildDefaults) must be
      // protected exactly like a recipe (the disarm at the recipe injection below). If the preset
      // resolves in the narrow window BEFORE the model catalog first surfaces the model, disarm the
      // fresh-mount declared-defaults resolver so it can't overwrite the applied preset tuning when
      // the catalog arrives on a later render. Only disarm when the preset actually carries a tuning
      // key — a preset with no tuning must still let the model's DECLARED defaults resolve.
      if (REFERENCE_TUNING_PRESET_KEYS.some((key) => Object.prototype.hasOwnProperty.call(defaults, key))) {
        referenceTuningDeclaredArmed.current = false;
      }
    },
    buildDefaults: () => ({
      ...(moss ? { mossImage: mossEdit ? { acceleration: "off", max_sequence_length: mossImage.max_sequence_length ?? 4096, uncond_skip_layers: [] } : mossWireOptions(mossImage), mossDmdSteps,
        mossOrdinarySettings: { options: { ...mossImage, acceleration: "off" }, steps: stepsOverride, guidanceScale: guidanceOverride, negativePrompt, schedulerShift } } : {}),
      ...(mossEdit ? { mossImageEdit, mossImageAdvanced, mossPromptBatch } : {}),
      prompt,
      // Same axis gate as the job payload (sc-15299): a preset saved on an engine with no
      // negative-prompt / guidance axis must not bank a value the studio never showed and the
      // worker never reads — applying that preset later on a guidance-taking model would
      // resurrect a "negative prompt" the user never typed there.
      negativePrompt: supportsNegativePrompt ? negativePrompt : "",
      resolution,
      count,
      mode,
      guidanceScale: supportsGuidance && !mossDmd ? finiteNumberOrUndefined(guidanceOverride) : undefined,
      steps: mossDmd ? Number(mossDmdSteps) : finiteNumberOrUndefined(stepsOverride),
      sampler,
      scheduler,
      schedulerShift: mossDmd ? undefined : effectiveSchedulerShift,
      ...(resolvedCfgInterval ? { cfgInterval: resolvedCfgInterval } : {}),
      ...(outgoingCaptionOverflow !== "default" ? { captionOverflow: outgoingCaptionOverflow } : {}),
      guidanceMethod,
      ...(supportsTextEncoderSelection &&
      selectedTextEncoderModel !== defaultTextEncoderId
        ? { textEncoderModel: selectedTextEncoderModel }
        : {}),
      upscaleEnabled,
      upscaleFactor,
      upscaleEngine,
      upscaleSoftness,
      upscaleRestore,
      hiresFixEnabled,
      hiresFixSteps,
      hiresFixDenoisingStrength,
      hiresFixUpscaleBy,
      hiresFixCfgScale,
      // Reference/identity knobs only matter for the character flow; keep them
      // out of plain text/edit presets so they don't carry irrelevant state.
      ...(mode === "character_image"
        ? { ipAdapterScale, controlnetScale, trueCfgScale, viewAngle }
        : {}),
    }),
  });

  useStudioSettingsWriter("image", activeProject?.id ?? null, {
    mode,
    prompt,
    mossImage,
    mossImageEdit,
    mossImageAdvanced,
    mossPromptBatch,
    mossDmdSteps,
    styleId,
    appearance: appearance.selection,
    structuredCaption: caption,
    promptMode,
    magicPromptBackend,
    count,
    advancedOpen,
    model,
    textEncoderModel,
    seed,
    negativePrompt,
    resolution,
    widthOverride,
    heightOverride,
    fitMode,
    referenceAssetIds,
    referenceDimensionIntent,
    orderedCharacterReferences,
    ...(isFlux3LayoutModel(model) && mode === "character_image" ? { characterId, characterLookId, referenceAssetId } : {}),
    flux3Layout,
    flux3ReferenceSlots,
    ipAdapterScale,
    controlnetScale,
    trueCfgScale,
    viewAngle,
    upscaleEnabled,
    upscaleFactor,
    upscaleEngine,
    upscaleSoftness,
    upscaleRestore,
    hiresFixEnabled,
    hiresFixSteps,
    hiresFixDenoisingStrength,
    hiresFixUpscaleBy,
    hiresFixCfgScale,
    selectedLoraIds,
    loraWeights,
    // Krea 2 multi-phase denoise (sc-13885): persist the editor toggle, phase list, and disclosure
    // state alongside the rest of the studio snapshot (LoRA refs are by stable id, so they survive
    // reload regardless of the selected-LoRA list's async hydration).
    multiPhaseEnabled,
    multiPhasePhases,
    multiPhaseOpen,
    selectedPresetId,
    generalStackIds,
    batchMode,
    batchPromptsText,
    batchVariableValues,
    batchName,
    batchScope,
    loadedBatchId,
    sampler,
    scheduler,
    schedulerShift,
    cfgInterval,
    captionOverflow,
    guidanceMethod,
    steps: stepsOverride,
    guidanceScale: guidanceOverride,
    flashAttn,
    enhancePrompt,
    bf16Precision,
    transparentBackground,
    usePid,
    pidTarget,
    decoder,
  },
  // Suppress the live writer until the model catalog has loaded (sc-11962), so a
  // transient defaults-reset during the restart-restore/settle window can't be
  // persisted over the restored snapshot. When there are no models the studio shows
  // the availability gate (no editable form), so nothing meaningful is lost by waiting.
  // ALSO gated on ui-preferences hydration (sc-15425): before the GET lands the localStorage
  // cache may be empty (a relaunched desktop app has a new origin), and persisting through it
  // would overwrite the durable copy with catalog defaults before anything read it.
  // A restored alternate decoder also waits for authoritative platform facts. Otherwise the
  // pending fallback can reconcile it to native and make that transient loss durable.
  preferencesHydrated && imageModels.length > 0 && !decoderCapabilitiesPending);

  // Each stacked run carries its already-resolved completed assets + the
  // expected count, which the WorkerProgressCard image-grid variant uses to
  // render thumbnails + skeleton cells (sc-2088 — replaces the explicit slot
  // construction the legacy JobProgressCard wrapper needed).
  const localJobGroups = useMemo(
    () =>
      localJobs.map((job) => {
        const completedAssets = jobResultAssets(job, assets);
        const expectedCount = jobExpectedCount(job, completedAssets.length);
        return { job, completedAssets, expectedCount };
      }),
    [assets, localJobs],
  );

  // Both entry points use the same admission checks, then their own prompt/form rules.
  // Layout references and model geometry are derived once above and passed unchanged
  // to the shared request builder; exact replay tiers and MOSS components must be ready.
  function preflightError(kind) {
    if (!mossComponents.ready || mossEditError) return mossEditError || (mossEdit ? "Install or repair the exact MOSS edit adapter before editing." : "Install the selected MOSS 8-step components before generating.");
    if (!selectedModelServesMode) return `Install or select a model that supports ${modeLabel} generation.`;
    if (replayTierUnavailable) return "Recipe tier unavailable. Install it on a supported worker or choose another tier.";
    if (characterReferenceError) return characterReferenceError;
    if (decoderCapabilitiesPending) return "Waiting for engine capabilities before using the restored decoder.";
    if (dimensionsInvalid || hiresFixTargetInvalid) return hiresFixTargetMessage || dimensionError || "Width and height must each be between 256 and 4096.";
    if (cfgIntervalError) return cfgIntervalError;
    if (macActiveModeBlock) return macActiveModeBlock.text;
    const validity = kind === "batch" ? batchValidity : generateValidity;
    if (!validity.ready) return validity.surfaced[0]?.message || "Complete the required fields before generating.";
    if (kind === "single" && structuredActive && !captionValidation?.ok) return "Review the caption before generating.";
    return null;
  }

  async function submit(event) {
    event.preventDefault();
    // Batch mode never routes a stray Enter through single generation.
    if (batchMode || submitting) return;
    const error = preflightError("single");
    if (error) {
      setSubmitError(error);
      return;
    }
    setSubmitting(true);
    try {
      // posePayload / controlActive / controlPreprocessSourceId / controlPassthroughId are
      // derived at component scope (shared with the batch run, sc-9980).
      // Resolve the prompt + structured-caption payload. Structured models (Ideogram 4) are
      // JSON-caption-only: raw plain text is out-of-distribution and renders the "Image blocked by
      // safety filter" placeholder (sc-6307/sc-6501). So a structured model ALWAYS sends a JSON
      // caption — the builder caption in form/JSON mode, or an auto-expanded caption when the user is
      // in plain-text mode. Plain text is never submitted raw to a structured engine.
      let promptToSend = prompt;
      let sendStructured = false;
      let submitCaption = caption;
      let submitBackend = magicPromptBackend;
      let submitIntent = prompt;
      if (structuredPromptModel) {
        if (structuredActive) {
          sendStructured = true;
          promptToSend = serializeCaption(caption);
        } else {
          // Plain-text mode for a structured model → auto-expand the idea into an editable caption
          // (silent auto-expand, surfaced in the Builder) before generating.
          const idea = prompt.trim();
          if (!idea) {
            return;
          }
          if (typeof magicPrompt !== "function" || magicModelMissing) {
            setSubmitError(
              "Plain text can't be sent to this model. Download the prompt-refiner model to auto-expand your idea into a caption, or build one in the Builder.",
            );
            return;
          }
          let expanded;
          setExpanding(true);
          try {
            expanded = await onMagicExpand(idea);
          } catch (e) {
            setSubmitError(e?.message || "Couldn't expand the prompt into a caption. Try the Builder.");
            return;
          } finally {
            setExpanding(false);
          }
          // Surface the expanded caption editable in the Builder regardless of validity.
          setCaption(expanded);
          setPromptMode("form");
          if (!validateCaption(expanded).ok) {
            setSubmitError("The auto-generated caption needs a tweak — review it in the Builder and generate again.");
            return;
          }
          sendStructured = true;
          submitCaption = expanded;
          submitBackend = PROMPT_REFINE_MODEL_ID;
          submitIntent = idea;
          promptToSend = serializeCaption(expanded);
        }
        setSubmitError("");
      }
      // sc-11219 (F-031): the single Generate payload is built by the shared buildJobRequest so
      // it stays byte-identical to a batch run with the same visible settings. The only per-call
      // difference is the resolved prompt / structured-caption override threaded in here.
      const job = await createImageJob(
        buildJobRequest({
          promptToSend,
          submitIntent,
          sendStructured,
          submitCaption,
          submitBackend,
        }),
      );
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }

  // sc-11219 (F-031): the single shared image-job-request builder used by BOTH single Generate
  // (submit) and the batch run. It resolves the current studio state into the args the pure
  // buildImageJobRequest assembles, so the two paths emit a byte-identical payload for the same
  // visible settings. The ONLY intended difference is the per-item prompt / structured-caption /
  // per-prompt-resolution override passed in `overrides` (promptToSend / submitIntent /
  // sendStructured / submitCaption / submitBackend / resolutionOverride). This replaced a
  // hand-copied batch twin that had drifted (dropped the img2img reference + pidTarget/img2img
  // advanced knobs), silently ignoring an img2img reference and PiD "2K" tier on batch runs.
  const activeStyleText = styleTextForId(effectiveStyleId);
  const composeJobPrompt = useCallback(
    (overrides = {}) => {
      const stackActive = generalStack.length > 0;
      const isStructured = overrides.sendStructured ?? false;
      const foldPrompt = (stackActive || (mossEdit && mossPromptBatch.length > 0)) && !isStructured;
      const promptToSend =
        foldPrompt && overrides.promptToSend != null
          ? composePreset({
              base: selectedPreset,
              generalStack,
              userText: overrides.promptToSend,
              resolutionOptions,
            }).prompt
          : overrides.promptToSend;
      const { prompt: composedPrompt } = composeImageJobPrompt({
        promptToSend,
        sendStructured: isStructured,
        styleText: activeStyleText,
        appearance: appearance.selection,
        model: selectedModel,
      });
      return { composedPrompt, foldPrompt, isStructured, promptToSend };
    },
    [activeStyleText, generalStack, resolutionOptions, selectedPreset, mossEdit, mossPromptBatch.length, appearance.selection, selectedModel],
  );

  const buildJobRequest = (overrides = {}) => {
    // Fold the general-preset stack (epic 11949) into this request. The prompt is composed per
    // call so it wraps either the single prompt or a per-batch-item prompt; a structured JSON
    // caption can't take flat fragments, so the prompt fold is skipped there (the stack's
    // negative/aspect/count still apply). When the stack folds the prompt, the client is
    // authoritative — presetPromptResolvedClientSide tells the server to skip its own fold.
    const stackActive = generalStack.length > 0;
    const { foldPrompt, isStructured, promptToSend } = composeJobPrompt(overrides);
    const stackResolution = stackActive && composedStack.resolution ? parseResolution(composedStack.resolution) : null;
    return buildImageJobRequest({
      flux3Layout,
      flux3ReferenceIds: layoutReferences.map(({ id }) => id),
      // Overrides — the one legitimate single-vs-batch difference.
      promptToSend,
      submitIntent: overrides.submitIntent,
      sendStructured: isStructured,
      submitCaption: overrides.submitCaption,
      submitBackend: overrides.submitBackend,
      // A per-prompt [WxH] batch directive wins; otherwise the stack's aspect drives resolution.
      resolutionOverride: overrides.resolutionOverride ?? stackResolution,
      // Shared studio settings (identical for both paths).
      mossImage,
      mossImageEdit,
      mossImageAdvanced,
      mossPromptBatch: mossEdit ? mossPromptBatch.map((text) => composeJobPrompt({ ...overrides, promptToSend: text }).promptToSend) : mossPromptBatch,
      mossDmdSteps,
      mossOrdinarySettings: mossDmd ? { options: { ...mossImage, acceleration: "off" }, steps: stepsOverride, guidanceScale: guidanceOverride, negativePrompt, schedulerShift } : undefined,
      resolution,
      mode,
      // An engine with no negative-prompt axis (sc-15299) gets an empty one rather than whatever a
      // preset stack composed or a replayed recipe restored — the field is hidden for it, so
      // sending text the user cannot see or edit (and the worker's `resolve_negative_prompt`
      // discards anyway) would be a ghost input recorded on this job's recipe.
      negativePrompt: supportsNegativePrompt && !negativePromptCfgOff && !mossDmd
        ? stackActive
          ? composedStack.negativePrompt
          : negativePrompt
        : "",
      model,
      count: stackActive && composedStack.count != null ? composedStack.count : count,
      seed,
      posePayload,
      width,
      height,
      recipePresetId: selectedPreset?.id ?? null,
      presetPromptResolvedClientSide: foldPrompt,
      // sc-13130: the selected Style Catalog entry's prompt text (or null for None). The pure
      // builder applies composeStyledPrompt as the LAST wrap — after the preset fold above has
      // produced `promptToSend` — so the style's `Style:` block wraps the already-preset-composed
      // user prompt as `Subject:`. Null → pass-through (prompt sent unchanged). Structured
      // caption models ignore it (the builder skips composition when sendStructured is true).
      styleText: activeStyleText,
      // sc-13132: the opaque style id travels with the recipe so replay can re-select the picker.
      styleId: effectiveStyleId,
      appearance: appearance.selection,
      appearanceModel: selectedModel,
      appearanceNegativeEnabled: supportsNegativePrompt && !negativePromptCfgOff && !mossDmd,
      characterId,
      characterLookId,
      multiReference,
      orderedCharacterReferences,
      sourceWithMultiReference,
      editSecondPair,
      sourceAssetId: mossEdit ? undefined : sourceAssetId,
      controlPreprocessSourceId,
      referenceAssetIds,
      fitMode,
      editInpaintCapable,
      referenceAssetId: mossEdit ? undefined : referenceAssetId,
      supportsImg2img,
      img2imgReferenceAssetId,
      loras: buildLorasPayload(),
      upscaleEnabled,
      upscaleFactor,
      upscaleEngine,
      upscaleSoftness,
      upscaleRestore,
      hiresFixEnabled,
      hiresFixSteps,
      hiresFixDenoisingStrength,
      hiresFixUpscaleBy,
      hiresFixCfgScale: supportsGuidance ? hiresFixCfgScale : "",
      sampler,
      scheduler,
      schedulerShift: effectiveSchedulerShift,
      supportsSchedulerShift,
      cfgInterval: resolvedCfgInterval,
      captionOverflow: outgoingCaptionOverflow,
      stepsOverride,
      // Same rule for the guidance axis (sc-15299): a CFG-free engine never receives a scale, so
      // the override is blanked before it reaches the builder and `advanced.guidanceScale` is
      // omitted entirely. The model-switch reset already clears it in the ordinary case, but a
      // replayed recipe (and a preset launch) restores model + guidance together with that reset
      // suppressed — which is the path that really leaks a stale scale onto a CFG-free engine.
      guidanceOverride: supportsGuidance ? guidanceOverride : "",
      guidanceMethod,
      supportsTextEncoderSelection,
      textEncoderModel: selectedTextEncoderModel,
      defaultTextEncoderId,
      flashAttn,
      promptEnhance,
      enhancePrompt,
      precisionToggle,
      bf16Precision,
      selectedModel,
      transparentBackground,
      showTierPicker,
      // Never send a tier that isn't installed-and-complete. The seed effect keeps `quantTier` clamped to
      // an installed tier, but gate the OUTGOING value on the selectable set as a hard belt so no state —
      // a race, a stale sticky, a torn tier newly reported missing — can leak an uninstalled tier into the
      // payload (the worker would then default/crash against a tier the user never downloaded). Mirrors the
      // guard VideoStudio/Editor already apply. Empty string ⇒ `tierQuantize("")` is null ⇒ no mlxQuantize.
      quantTier: availableTiers.includes(quantTier) ? quantTier : "",
      // sc-10733: the tier is a DELIBERATE pick (not the pure global/base default) when it equals this
      // (screen, model)'s persisted sticky — derived by `useQuantTierPicker`, which the picker also uses
      // to label a non-explicit tier as a suggestion (sc-22246). The worker honors an explicit pick (never
      // silently downtiers it); only a non-explicit default is capability-clamped.
      tierExplicit,
      showPidToggle,
      usePid,
      pidTarget,
      showDecoderPicker,
      decoder,
      hideReferenceStrength,
      ipAdapterScale,
      identityStructure,
      controlnetScale,
      variationStrength,
      trueCfgScale,
      img2imgStrength,
      supportsTextStyle,
      textStyleGain,
      viewAngles,
      viewAngle,
      faceRestore,
      controlActive,
      activeControlMode,
      controlPassthroughId,
      effectiveControlScale,
      controlOverlayId,
      // Krea 2 multi-phase denoise (sc-13885): the resolved gate + the serialized phase list. The
      // advanced builder emits `advanced.phases` only when active + non-empty, so a normal
      // single-phase Raw job (editor off, or any other model/mode) stays byte-for-byte unchanged.
      multiPhaseActive,
      phases: serializedPhases,
    });
  };

  // One image-job request for a single resolved batch prompt. Thin adapter over the shared
  // buildJobRequest: resolves the per-prompt override defaults (structured-caption payload and a
  // per-prompt [WxH] directive, sc-10063) against studio state, then delegates. `count`
  // multiplies within each job UNLESS poses are selected, in which case each job emits one image
  // per pose (images = jobs × posePayload.length). For a structured-caption model (sc-9980) the
  // caller passes the per-prompt auto-expanded caption via `opts`.
  const buildBatchJobRequest = (resolvedPrompt, opts = {}) =>
    buildJobRequest({
      promptToSend: opts.promptToSend ?? resolvedPrompt,
      submitIntent: resolvedPrompt,
      sendStructured: opts.sendStructured ?? false,
      submitCaption: opts.submitCaption ?? caption,
      submitBackend: opts.submitBackend ?? magicPromptBackend,
      resolutionOverride: opts.resolution ?? null,
    });

  // The Iris post-processing plan covers what the next run renders: the Studio size for a single
  // Generate, every distinct per-prompt [WxH] size (or the Studio size) for a prompt batch (S2).
  const restorePlanBody = useMemo(
    () =>
      restorePlanBodyForRenders({
        width,
        height,
        resolutions: batchMode ? batchPrompts.map((line) => parsePromptResolution(line, resolutionOptions).resolution) : [],
      }),
    [batchMode, batchPrompts, height, width, resolutionOptions],
  );
  const batchMissingKeys = missingKeys(batchPrompts, batchVariables);
  const batchGroupIssues = linkedGroupIssues(batchPrompts);
  // Prompt lines whose leading [WxH] directive (sc-10063) breaks the SELECTED model's geometry — the
  // SAME per-model range + stride the single-generate gate enforces (sc-14058), not the raw backend
  // envelope. For a native-resolution model (Mage-Flow) an off-range OR off-÷16-stride size blocks the
  // Run; every other model keeps the global 256–4096 / no-stride check. Reuses evaluateModelDimensions
  // so the batch and single paths cannot drift.
  const batchResolutionIssues = batchPrompts
    .map((line) => parsePromptResolution(line, resolutionOptions).resolution)
    .filter(
      (res) =>
        res &&
        evaluateModelDimensions({
          model: selectedModel,
          resolution: `${res.width}x${res.height}`,
          widthOverride: "",
          heightOverride: "",
          referenceRequest: flux3ReferenceRequest,
        }).invalid,
    );
  // A structured-caption model can batch, but only if the prompt-refiner is available to
  // auto-write a caption per resolved prompt (sc-9980).
  const batchStructuredExpandBlocked =
    structuredPromptModel && (magicModelMissing || typeof magicPrompt !== "function");
  const styleSelected = typeof activeStyleText === "string" && activeStyleText.trim() !== "";
  const stylePreviewActive = !structuredPromptModel && (styleSelected || appearance.active);
  const appearanceNegativePreview = supportsNegativePrompt && !negativePromptCfgOff && !mossDmd
    ? composeNegativePrompt(generalStack.length ? composedStack.negativePrompt : negativePrompt, appearance.resolved.negative) : "";
  // sc-13224: structured JSON-caption models apply the Style axis by merging into the caption's
  // `style_description.aesthetics` (see imageJobRequest.js), so the outgoing prompt is the injected,
  // re-serialized caption. Compute it here so the budget guard measures the ACTUAL string sent (the
  // caption grows against the 4000-char cap once a style is merged in). Only when a structured model
  // is in caption mode with a style selected; a null/empty style is a pass-through.
  const structuredStyleActive = structuredActive && styleSelected;
  const structuredStyledPrompt = structuredStyleActive
    ? serializeCaption(injectStyleIntoCaption(caption, activeStyleText))
    : null;
  // One summary per CTA (epic 10644): the button's `disabled` and the message it owes the
  // user come from the same issue list and cannot drift. Two independent rule sets — the
  // batch's problems must never disable single-image Generate. The drafts gather the
  // already-computed sub-results the rules turn into issues.
  const batchDraft = useMemo(
    () => ({
      activeProject,
      batchStructuredExpandBlocked,
      batchTotal,
      missingKeys: batchMissingKeys,
      groupIssues: batchGroupIssues,
      resolutionIssues: batchResolutionIssues,
      // Per-model range + stride for the inline [WxH] directive — the SAME constraints the free
      // Width/Height override uses (sc-14058). A native-resolution model narrows the batch check to its
      // own 512–2048 / ÷16 envelope; every other model gets the global 256–4096 / no-stride default.
      dimensionConstraints,
      decoderCapabilitiesPending,
      restorePlan: irisUpscale ? restorePlanStatus : null,
    }),
    [
      activeProject,
      batchStructuredExpandBlocked,
      batchTotal,
      batchMissingKeys,
      batchGroupIssues,
      batchResolutionIssues,
      dimensionConstraints,
      decoderCapabilitiesPending,
      irisUpscale,
      restorePlanStatus,
    ],
  );
  // sc-13131 / sc-13133: the live composed-prompt preview for the selected Style Catalog entry, and
  // the budget the composed string spends against the backend cap. ANTI-DRIFT: composeJobPrompt is
  // the shared prompt-only primitive used by both this preview and buildJobRequest, so the
  // previewed/measured string is byte-for-byte the prompt that will be sent without rebuilding the
  // entire request on every keystroke (preset stack folds into the prompt FIRST, the style's
  // Subject:/Style: wrap is applied LAST — see imageJobRequest.js). Its memo dependencies track the
  // prompt text, selected style, and active preset stack. Only active for
  // free-text models with a style actually selected: structured-caption models (Ideogram) merge the
  // style into the caption's `aesthetics` instead (sc-13224), so there's no Subject:/Style: prose
  // to preview, and a null/empty styleText is a pass-through with nothing extra to preview and no
  // style-composition budget to guard.
  const styledPreviewPrompt = useMemo(
    () =>
      stylePreviewActive
        ? composeJobPrompt({ promptToSend: prompt }).composedPrompt
        : null,
    [composeJobPrompt, prompt, stylePreviewActive],
  );
  const generateDraft = useMemo(
    () => ({
      activeProject,
      structuredActive,
      captionHasContent,
      tensorConditioning: mossEdit && Boolean(mossImageAdvanced.positive),
      prompt,
      // sc-13133 / sc-13224: measure the COMPOSED outgoing prompt against the cap, but only when a
      // style is active (styleless behavior unchanged). For prose that is the Subject:/Style:
      // composition; for a structured model it is the style-injected, re-serialized caption. Either
      // string is exactly what the run submits, so the cap is measured on IT.
      styleActive: stylePreviewActive || structuredStyleActive,
      composedPrompt: styledPreviewPrompt ?? structuredStyledPrompt ?? "",
      mode,
      characterId,
      // Edit needs a source (single) or ≥1 reference (multiReference); a required edit LoRA must be
      // downloaded first. Both silently gate Generate — the empty picker / the source-band download
      // note are the visible affordances.
      editSourceMissing:
        mode === "edit_image" && (multiReference ? !referenceAssetIds.length : !sourceAssetId),
      editLoraMissing: editLoraRequiredMissing,
      presetMissing: presetValidationResult.missing,
      presetIncompatible: presetValidationResult.incompatible,
      loraIncompatible: selectedLoraValidationResult.incompatible,
      modelName: selectedModel?.name,
      // Multi-phase denoise problems (sc-13885): an enabled-but-broken phase list blocks Generate.
      multiPhaseIssues: multiPhaseValidationIssues,
      // Free Width/Height override out of the model's range or off its ÷16 stride (sc-14058). A precise
      // ERROR message (not a raw requirement) so the dead Generate button states its reason pre-emptively,
      // not only on click. null when the dimensions are valid.
      dimensionError,
      referenceOverflowError,
      decoderCapabilitiesPending,
      restorePlan: irisUpscale ? restorePlanStatus : null,
    }),
    [
      activeProject,
      structuredActive,
      captionHasContent,
      mossEdit, mossImageAdvanced,
      prompt,
      stylePreviewActive,
      structuredStyleActive,
      styledPreviewPrompt,
      structuredStyledPrompt,
      mode,
      characterId,
      multiReference,
      referenceAssetIds,
      sourceAssetId,
      editLoraRequiredMissing,
      presetValidationResult,
      selectedLoraValidationResult,
      selectedModel,
      multiPhaseValidationIssues,
      dimensionError,
      referenceOverflowError,
      decoderCapabilitiesPending,
      irisUpscale,
      restorePlanStatus,
    ],
  );
  const batchValidity = useValidation(imageBatchValidation, batchDraft, undefined);
  const generateValidity = useValidation(imageGenerateValidation, generateDraft, undefined);
  // `submitting` is a busy gate, not a rule. `batchTotal === 0` is a requirement (silent),
  // already folded into batchValidity — no need to repeat it here.
  const batchRunDisabled =
    !batchValidity.ready || !selectedModelServesMode || replayTierUnavailable || Boolean(characterReferenceError) || !mossComponents.ready || Boolean(mossEditError) || Boolean(batchRun?.submitting);
  // The two conditions whose message has its own home stay explicit gates: a structured
  // caption's field errors live in the builder, and a Mac block prints its own note.
  const {
    replaceBatchRun,
    runBatch,
    cancelBatchRun,
    batchRunProgress
  } = useImageBatchRun({
    batchRun,
  activeProject,
  jobs,
  jobAction,
  setBatchRun,
  BATCH_ACTIVE_JOB_LIMIT,
  preflightError,
  setBatchError,
  batchTotal,
  BATCH_RENDER_CAP,
  setBatchConfirmPending,
  batchPrompts,
  batchVariables,
  resolutionOptions,
  composeJobPrompt,
  stylePreviewActive,
  structuredPromptModel,
  batchJobCount,
  BATCH_PREFLIGHT_YIELD_INTERVAL,
  selectedModel,
  buildBatchJobRequest,
  onMagicExpand,
  createImageJob
  });
  const generateDisabled =
    submitting ||
    replayTierUnavailable ||
    Boolean(characterReferenceError) ||
    !mossComponents.ready ||
    Boolean(mossEditError) ||
    !selectedModelServesMode ||
    !generateValidity.ready ||
    (structuredActive && !captionValidation?.ok) ||
    Boolean(macActiveModeBlock);

  return (
    <ModelAvailabilityGate
      ready={modelReady}
      initializing={modelCatalogStatus === "idle" || modelCatalogStatus === "loading"}
      title="Image Studio needs an image model"
      description="Download a recommended image model to start generating."
      offers={modelOffers}
      downloadJobs={modelDownloadJobs}
      onDownload={createModelDownloadJob}
      onOpenModels={() => setActiveView("Models")}
      onOpenQueue={onOpenQueue}
      onCancelJob={onCancelJob}
    >
    <section className="page-frame image-studio">
      <form className="studio-shell" onSubmit={submit}>
        <WorkPanel className="studio-work-panel">
          <ImagePromptComposer
            mode={mode}
            handleModeChange={handleModeChange}
            macModeTabBlock={macModeTabBlock}
            batchMode={batchMode}
            setBatchMode={setBatchMode}
            openGuide={openGuide}
            onOpenPresets={onOpenPresets}
            batchPromptsText={batchPromptsText}
            resolutionOptions={resolutionOptions}
            setBatchPromptsText={setBatchPromptsText}
            batchVariableValues={batchVariableValues}
            setBatchVariableValues={setBatchVariableValues}
            batchImagesPerPrompt={batchImagesPerPrompt}
            promptBatches={promptBatches}
            activeProject={activeProject}
            batchName={batchName}
            setBatchName={setBatchName}
            batchScope={batchScope}
            setBatchScope={setBatchScope}
            loadedBatchId={loadedBatchId}
            handleSaveBatch={handleSaveBatch}
            handleNewBatch={handleNewBatch}
            handleLoadBatch={handleLoadBatch}
            handleDeleteBatch={handleDeleteBatch}
            handleImportBatch={handleImportBatch}
            batchBusy={batchBusy}
            batchError={batchError}
            batchValidity={batchValidity}
            batchTotal={batchTotal}
            batchRun={batchRun}
            batchRunProgress={batchRunProgress}
            cancelBatchRun={cancelBatchRun}
            replaceBatchRun={replaceBatchRun}
            batchConfirmPending={batchConfirmPending}
            runBatch={runBatch}
            setBatchConfirmPending={setBatchConfirmPending}
            batchRunDisabled={batchRunDisabled}
            structuredPromptModel={structuredPromptModel}
            caption={caption}
            setCaption={setCaption}
            captionValidation={captionValidation}
            promptMode={promptMode}
            setPromptMode={setPromptMode}
            prompt={prompt}
            setPromptFromUser={setPromptFromUser}
            magicPrompt={magicPrompt}
            onMagicExpand={onMagicExpand}
            magicModelMissing={magicModelMissing}
            refineModel={refineModel}
            createModelDownloadJob={createModelDownloadJob}
            onReferenceImageLoaded={onReferenceImageLoaded}
            resolutionSnapKey={resolutionSnapKey}
            imageCaption={imageCaption}
            onImageCaption={onImageCaption}
            editImageAssets={editImageAssets}
            characters={characters}
            importAsset={importAsset}
            visionCaptionReady={visionCaptionReady}
            visionCaptionOffers={visionCaptionOffers}
            modelDownloadJobs={modelDownloadJobs}
            setActiveView={setActiveView}
            onOpenQueue={onOpenQueue}
            onCancelJob={onCancelJob}
            generateDisabled={generateDisabled}
            submitting={submitting}
            expanding={expanding}
            replayTierUnavailable={replayTierUnavailable}
            quantTier={quantTier}
            setRecipeTierRequest={setRecipeTierRequest}
            submitError={submitError}
            promptTokenNotice={promptTokenNotice}
            promptHint={promptHint}
            suggestions={suggestions}
            supportsImg2img={supportsImg2img}
            promptTool={promptTool}
            togglePromptTool={togglePromptTool}
            img2imgReferenceAssetId={img2imgReferenceAssetId}
            changeImg2imgReferenceAssetId={changeImg2imgReferenceAssetId}
            img2imgStrengthConfig={img2imgStrengthConfig}
            setImg2imgStrength={setImg2imgStrength}
            img2imgStrength={img2imgStrength}
            imageDescribe={imageDescribe}
            qwenRewriteAvailable={qwenRewriteAvailable}
            onImageDescribe={onImageDescribe}
            promptGuide={promptGuide}
            model={model}
            refinePrompt={refinePrompt}
            handleResolutionChange={handleResolutionChange}
            qwenRewriteReferenceIds={qwenRewriteReferenceIds}
            qwenRewriteModel={qwenRewriteModel}
            qwenRewritePrompt={qwenRewritePrompt}
            flux3LayoutAvailable={flux3LayoutAvailable}
            flux3Layout={flux3Layout}
            setFlux3Layout={setFlux3Layout}
            width={width}
            height={height}
            layoutReferences={layoutReferences}
            assets={assets}
          />

        {mode === "edit_image" || mode === "character_image" ? (
          <div className="studio-source-band">
            {mode === "edit_image" || orderedCharacterActive ? (
              <>
                {mossEdit ? <>
                  <MossReferencePicker assets={editImageAssets} characters={characters} projectId={activeProject?.id} importAsset={importAsset} assetIds={referenceAssetIds} onChange={setReferenceAssetIds} max={maxStudioReferences} />
                  <MossImageEditControls model={selectedModel} token={token} jobs={jobs} options={mossImageEdit} onChange={setMossImageEdit} advanced={mossImageAdvanced} onAdvancedChange={setMossImageAdvanced} assets={assets} importAsset={importAsset} promptBatch={mossPromptBatch} onPromptBatchChange={(batch) => { setMossPromptBatch(batch); if (batch.length) setPrompt(batch[0]); }} />
                  {mossEditError ? <p role="alert">{mossEditError}</p> : null}
                </> : multiReference ? (
                  // sc-6211: FLUX.2-dev multi-reference edit — pick 1–N reference images that the
                  // model combines/edits (Conditioning::MultiReference). Sends the plural
                  // `referenceAssetIds`; a single pick reduces to the normal single-reference edit.
                  <AssetPickerField
                    assets={editImageAssets}
                    buttonLabel="Select images"
                    changeLabel="Edit references"
                    emptyLabel="No reference images selected"
                    label={
                      maxStudioReferences != null
                        ? `Reference images (up to ${maxStudioReferences}, in order)`
                        : "Reference images"
                    }
                    multiple
                    onChange={changeReferenceAssetIds}
                    values={referenceAssetIds}
                  />
                ) : null}
                {/* sc-24113: the ORDER of these references is part of the request for a model whose
                    template numbers them, so show it and let the user change it. Instance owners
                    keep removal available through the last reference and the explicit empty state. */}
                {mossEdit ? null : multiReference ? (
                  <OrderedReferenceList
                    assetIds={referenceAssetIds}
                    itemKeys={flux3LayoutAvailable ? layoutReferences.map(({ id }) => id) : undefined}
                    onMove={flux3LayoutAvailable ? moveFlux3Reference : undefined}
                    onRemove={flux3LayoutAvailable ? removeFlux3Reference : undefined}
                    labelFor={assetLabelFor}
                    onChange={changeReferenceAssetIds}
                  />
                ) : (
                  <>
                    <ImageEditSourcePickerField
                      assets={editImageAssets}
                      buttonLabel="Select image"
                      characters={characters}
                      emptyLabel="No source image selected"
                      importAsset={importAsset}
                      label={editReferences ? "Image 1" : "Source image"}
                      onChange={changeSourceAssetId}
                      projectId={activeProject?.id}
                      value={sourceAssetId}
                    />
                    {/* Optional second source for a two-reference edit (epic 10871 P1.3). Any two
                        images in a fixed order: the source above is image 1 (required); this is
                        image 2 (optional). Only rendered when the model declares `ui.editReferences`. */}
                    {editReferences ? (
                      <>
                        <ImageEditSourcePickerField
                          assets={editImageAssets}
                          buttonLabel="Select image"
                          characters={characters}
                          clearable
                          emptyLabel="No second image selected (optional)"
                          importAsset={importAsset}
                          label={editReferences.secondaryLabel ?? "Image 2 (optional)"}
                          onChange={changeEditSecondAssetId}
                          projectId={activeProject?.id}
                          value={editSecondAssetId}
                        />
                        {editReferences.secondaryHint ? (
                          <p className="field-hint">{editReferences.secondaryHint}</p>
                        ) : null}
                      </>
                    ) : null}
                    {sourceWithMultiReference ? (
                      <AssetPickerField
                        assets={editImageAssets}
                        buttonLabel="Select references"
                        changeLabel="Edit references"
                        emptyLabel="No additional references selected (optional)"
                        label="Additional references (optional)"
                        multiple
                        onChange={changeReferenceAssetIds}
                        values={referenceAssetIds}
                      />
                    ) : null}
                    {sourceWithMultiReference ? (
                      <OrderedReferenceList
                        assetIds={referenceAssetIds}
                        labelFor={assetLabelFor}
                        onChange={changeReferenceAssetIds}
                      />
                    ) : null}
                  </>
                )}
                <FitModeControl
                  value={effectiveFitMode(fitMode, editInpaintCapable)}
                  onChange={setFitMode}
                  inpaintCapable={editInpaintCapable}
                />
                {/* Krea-style edit LoRA (epic 10871, P4.1): managed for the user — no manual
                    picking. Applied automatically once installed; a one-click download when not
                    (the base can't edit without it, R5). Inert for edit models that need none. */}
                {editLora ? (
                  editLoraInstalled ? (
                    <>
                      <p className="field-hint" role="status">
                        <Icon.Sparkle size={13} /> {editLora.name} is applied automatically for editing.
                      </p>
                      {/* Identity strength (sc-11798): the managed edit LoRA is hidden from the manual
                          picker, so expose its apply weight here. `effectiveLoraWeight`/`setLoraWeight`
                          already back the value, and `buildLorasPayload` serializes it into the edit
                          LoRA's payload `weight` — higher = stronger identity/edit conditioning. */}
                      <div className="lora-slot-weight edit-lora-strength">
                        <label>
                          <span>Identity strength</span>
                          <span className="lora-slot-weight-value">
                            {effectiveLoraWeight(editLora).toFixed(2)}
                          </span>
                        </label>
                        <input
                          aria-label={`${editLora.name} identity strength`}
                          max="2"
                          min="0"
                          onChange={(event) => setLoraWeight(editLora.id, Number(event.target.value))}
                          step="0.05"
                          type="range"
                          value={effectiveLoraWeight(editLora)}
                        />
                      </div>
                    </>
                  ) : (
                    <div className="inline-warning edit-lora-download">
                      <span>
                        {editLora.name} is required to edit — the base can’t edit without it.
                      </span>
                      <button
                        type="button"
                        className="secondary-action"
                        onClick={requestEditLoraDownload}
                        disabled={editLoraDownloadRequested}
                      >
                        {editLoraDownloadRequested ? "Downloading…" : "Download"}
                      </button>
                    </div>
                  )
                ) : null}
              </>
            ) : null}

            {mode === "character_image" ? (
              <>
                <div className="control-grid compact-controls">
                  <CharacterLookFields characters={characters} characterId={characterId} lookId={characterLookId}
                    onCharacterChange={changeCharacterId} onLookChange={setCharacterLookId} characterPlaceholder="Select character" />

                </div>
                {flux3LayoutAvailable && !orderedCharacterReferences ? <button type="button"
                  onClick={() => changeReferenceAssetIds(referenceAssetId ? [referenceAssetId] : [])}>Edit reference images</button> : null}
                {characterReferenceError ? <p className="inline-warning" role="alert">{characterReferenceError}</p> : null}
                {characterId ? (
                  characterReferences.length ? (
                    <div className="character-reference-picker">
                      <span className="reference-picker-label">Reference identity</span>
                      <div className="reference-thumb-row">
                        {characterReferences.map((reference) => (
                          <button
                            aria-label={`Use ${reference.asset?.displayName ?? reference.assetId} as reference`}
                            aria-pressed={reference.assetId === referenceAssetId}
                            className={reference.assetId === referenceAssetId ? "reference-thumb active" : "reference-thumb"}
                            key={reference.assetId}
                            onClick={() => changeReferenceAssetId(reference.assetId)}
                            title={reference.asset?.displayName ?? reference.assetId}
                            type="button"
                          >
                            {reference.asset ? <AssetMedia asset={reference.asset} controls={false} /> : <span>Missing asset</span>}
                          </button>
                        ))}
                      </div>
                      {hideReferenceStrength ? null : (
                        <label className="reference-strength">
                          {referenceStrengthCfg?.label ??
                            (identityStructure ? "Identity strength" : "Reference strength")}
                          <input
                            max={referenceStrengthCfg?.max ?? 1}
                            min={referenceStrengthCfg?.min ?? 0}
                            onChange={(event) => setIpAdapterScale(Number(event.target.value))}
                            step={referenceStrengthCfg?.step ?? 0.05}
                            type="range"
                            value={ipAdapterScale}
                          />
                          <span>{ipAdapterScale.toFixed(2)}</span>
                        </label>
                      )}
                      {identityStructure ? (
                        <label className="reference-strength">
                          {identityStructure.label ?? "Identity structure"}
                          <input
                            max={identityStructure.max ?? 1}
                            min={identityStructure.min ?? 0}
                            onChange={(event) => setControlnetScale(Number(event.target.value))}
                            step={identityStructure.step ?? 0.05}
                            type="range"
                            value={controlnetScale}
                          />
                          <span>{controlnetScale.toFixed(2)}</span>
                        </label>
                      ) : null}
                      {variationStrength ? (
                        <label className="reference-strength">
                          {variationStrength.label ?? "Variation"}
                          <input
                            max={variationStrength.max ?? 10}
                            min={variationStrength.min ?? 1}
                            onChange={(event) => setTrueCfgScale(Number(event.target.value))}
                            step={variationStrength.step ?? 0.5}
                            type="range"
                            value={trueCfgScale}
                          />
                          <span>{trueCfgScale.toFixed(2)}</span>
                        </label>
                      ) : null}
                      {viewAngles ? (
                        <label className="reference-strength">
                          View angle
                          <select onChange={(event) => setViewAngle(event.target.value)} value={viewAngle}>
                            <option value="">Match reference</option>
                            {viewAngles.map((angle) => (
                              <option key={angle.id} value={angle.id}>
                                {angle.label}
                              </option>
                            ))}
                          </select>
                        </label>
                      ) : null}
                      {poseLibrary && macPoseBlock ? (
                        <p className="mac-gating-note">{macPoseBlock.text}</p>
                      ) : poseLibrary ? (
                        <details
                          className="pose-library-details"
                          onToggle={(event) => setPoseLibraryReplayOpen(event.currentTarget.open)}
                          open={poseLibraryReplayOpen || undefined}
                        >
                          <summary>
                            Pose library{selectedPoseIds.length ? ` · ${selectedPoseIds.length} selected` : ""}
                          </summary>
                          <PoseLibraryPicker
                            extraPoses={workflowPoses}
                            loadUserPoses={loadUserPoses}
                            onClear={() => setSelectedPoseIds([])}
                            onToggle={(id) =>
                              setSelectedPoseIds((ids) =>
                                ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id],
                              )
                            }
                            selectedIds={selectedPoseIds}
                            preferredCategory={workflowPoses.length ? WORKFLOW_POSE_CATEGORY : null}
                          />
                          <label className="checkline">
                            <input checked={faceRestore} onChange={(event) => setFaceRestore(event.target.checked)} type="checkbox" />
                            Restore face (sharper identity; off keeps the raw render)
                          </label>
                          <p className="muted">Selecting poses generates one image per pose (overrides Variations).</p>
                        </details>
                      ) : null}
                      <div className="guidance-strip">
                        <strong>Identity from reference</strong>
                        <span>
                          {identityStructure
                            ? "InstantID holds this person's face from the reference while the prompt drives the scene. Identity strength tunes likeness; Identity structure locks face geometry. Set a View angle to rotate the head (profiles, up/down, diagonals) with identity preserved. Raise Variations and leave the seed blank to explore takes."
                            : variationStrength && hideReferenceStrength
                            ? "Qwen's dual-control architecture (semantic + appearance) carries this reference's subject across new scenes and poses. Variation steers prompt-vs-reference balance: higher = more prompt-driven, lower = closer to the reference. Raise Variations and leave the seed blank to explore takes."
                            : variationStrength
                            ? "This reference's identity is carried across every variation. Reference strength tunes how strongly the reference conditions the result; Variation steers prompt adherence (raise for more variety, lower for closer to the reference). Raise Variations and leave the seed blank to explore takes."
                            : "This reference's identity is carried across every variation. Raise Variations and leave the seed blank to explore different takes."}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="guidance-strip">
                      <strong>No approved reference</strong>
                      <span>Approve a reference image for this character in Character Studio to generate identity-preserving variations. Generating now uses the prompt only.</span>
                    </div>
                  )
                ) : (
                  <div className="guidance-strip">
                    <strong>Select a character</strong>
                    <span>Choose a character with an approved reference image to copy its identity across variations.</span>
                  </div>
                )}
              </>
            ) : null}
          </div>
        ) : null}

          {/* Generation settings (UI-refinement 2b): the everyday knobs — Model, Aspect,
              Variations, Style preset — sit in a bar directly under the composer instead of a
              detached right rail. Power-user knobs fold into Advanced below; the results area
              reclaims the full width (single-column .studio-results). */}
          <ImageGenerationSettings
            selectedModel={selectedModel}
            setModel={setModel}
            model={model}
            pickerModels={pickerModels}
            modeLabel={modeLabel}
            setActiveView={setActiveView}
            createModelDownloadJob={createModelDownloadJob}
            mossEdit={mossEdit}
            mossDmd={mossDmd}
            setMossField={setMossField}
            mossComponents={mossComponents}
            mossDownload={mossDownload}
            mossDownloadProgress={mossDownloadProgress}
            downloadMossAcceleration={downloadMossAcceleration}
            handleResolutionChange={handleResolutionChange}
            resolution={resolution}
            resolutionOptions={resolutionOptions}
            formatResolutionLabel={formatResolutionLabel}
            patchGridNotice={patchGridNotice}
            nativeResolutionNotice={nativeResolutionNotice}
            setCount={setCount}
            count={count}
            showTierPicker={showTierPicker}
            quantTier={quantTier}
            handleTierChange={handleTierChange}
            tierPickerItems={tierPickerItems}
            tierSwitching={tierSwitching}
            tierSuggested={tierSuggested}
            confirmTier={confirmTier}
            tierBelowFloor={tierBelowFloor}
            qualityFloor={qualityFloor}
            tierExplicit={tierExplicit}
            pickerSelectedLoras={pickerSelectedLoras}
            pickerSelectedLoraIds={pickerSelectedLoraIds}
            pickerCompatibleLoras={pickerCompatibleLoras}
            userSelectedLoraCount={userSelectedLoraCount}
            toggleLora={toggleLora}
            effectiveLoraWeight={effectiveLoraWeight}
            setLoraWeight={setLoraWeight}
            loraEmptyMessage={loraEmptyMessage}
            createLoraDownloadJob={createLoraDownloadJob}
            activeProject={activeProject}
            createLoraImportJob={createLoraImportJob}
            models={models}
            appearance={appearance}
            styleAxisAvailable={styleAxisAvailable}
            styleId={styleId}
            setStyleId={setStyleId}
            selectedPreset={selectedPreset}
            availablePresets={availablePresets}
            setSelectedPresetId={setSelectedPresetId}
            availableGeneralPresets={availableGeneralPresets}
            generalStackIds={generalStackIds}
            toggleGeneralPreset={toggleGeneralPreset}
            presetPromptParts={presetPromptParts}
            presetLoraDetails={presetLoraDetails}
            presetName={presetName}
            setPresetName={setPresetName}
            savingPreset={savingPreset}
            presetSaveMessage={presetSaveMessage}
            setPresetSaveMessage={setPresetSaveMessage}
            handleSaveAsPreset={handleSaveAsPreset}
            presetScope={presetScope}
            setPresetScope={setPresetScope}

          />

          {/* sc-13131: the EXACT composed prompt (Subject:/Style:, preserved sibling directives,
              and the own-`Style:` MERGE) the run will send once a style is active — reuses
              buildJobRequest so it can never drift from the payload. Sits under the Style axis row.
              Hidden when no style applies. */}
          <StyledPromptPreview active={stylePreviewActive} composedPrompt={styledPreviewPrompt} appearanceActive={appearance.active} negativePrompt={appearance.active ? appearanceNegativePreview : undefined} />

          {macActiveModeBlock ? <p className="mac-gating-note">{macActiveModeBlock.text}</p> : null}

          {decoderCapabilitiesPending && macCapabilitiesError ? (
            <div className="inline-warning decoder-capability-recovery" role="alert">
              <span>{macCapabilitiesError} The restored decoder has not been changed.</span>
              <div className="decoder-capability-recovery-actions">
                <button
                  className="secondary-action"
                  disabled={macCapabilitiesLoading || typeof refreshMacCapabilities !== "function"}
                  onClick={() => refreshMacCapabilities?.()}
                  type="button"
                >
                  {macCapabilitiesLoading ? "Retrying…" : "Retry"}
                </button>
                <button
                  className="secondary-action"
                  onClick={() => setDecoder("native")}
                  type="button"
                >
                  Use Native VAE
                </button>
              </div>
            </div>
          ) : null}

          {/* `stackAddsNegative` is ANDed with the engine's negative-prompt axis (sc-15299), the
              same way VideoStudio does it: the submit path blanks the composed negative for a
              CFG-free engine, so previewing "+ negative prompt" would advertise a contribution
              this job will not make. */}
          <PresetStackPreview
            generalStack={generalStack}
            composed={composedStack}
            stackAddsNegative={supportsNegativePrompt && stackAddsNegative}
            stackAddsCount={stackAddsCount}
          />

          {/* Strict-control panel (epic 8236, sc-8245): pose / canny / depth structure lock for the
              text-to-image backbones whose `ui.controlModes` advertises it. Hidden when the backbone
              supports no strict control. Pose reuses the library picker (one image per pose); canny/depth
              take an uploaded control image + a preprocess-vs-use-as-is toggle. The request wiring lives in
              submit() — controlMode / sourceAssetId|advanced.controlImage / advanced.controlScale.
              Sits directly above Advanced (studio-cleanup sc-15369): it is a disclosure of the same
              kind, so it belongs beside the other one rather than in a source band above the settings. */}
          {showControlPanel ? (
            <ControlPanel
              supportedModes={controlModes}
              controlMode={activeControlMode}
              onControlModeChange={setControlMode}
              selectedPoseIds={selectedPoseIds}
              onTogglePose={(id) =>
                setSelectedPoseIds((ids) =>
                  ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id],
                )
              }
              onClearPoses={() => setSelectedPoseIds([])}
              loadUserPoses={loadUserPoses}
              extraPoses={workflowPoses}
              preferredPoseCategory={workflowPoses.length ? WORKFLOW_POSE_CATEGORY : null}
              revealPoseLibraryToken={workflowPoses.length ? launchRequest?.id : null}
              poseReplayOpen={poseLibraryReplayOpen}
              poseBlockText={macPoseBlock ? macPoseBlock.text : null}
              controlImageAssetId={controlImageAssetId}
              onControlImageChange={setControlImageAssetId}
              controlImagePassthrough={effectiveControlPassthrough}
              onControlImagePassthroughChange={setControlImagePassthrough}
              controlImageAssets={
                activeControlMode === "depth" ? [...depthControlAssets, ...editImageAssets] : editImageAssets
              }
              importAsset={importAsset}
              projectId={activeProject?.id}
              characters={characters}
              controlScaleConfig={controlScaleConfig}
              controlScale={effectiveControlScale}
              onControlScaleChange={setControlScale}
              controlOverlayBaseModel={controlOverlayBaseModel}
              selectedOverlayId={controlOverlayId}
              onOverlayChange={setControlOverlayId}
            />
          ) : null}

          <ImageAdvancedPanel
            toggleAdvanced={toggleAdvanced}
            advancedOpen={advancedOpen}
            setRequestedGpu={setRequestedGpu}
            requestedGpu={requestedGpu}
            gpuOptions={gpuOptions}
            mossEdit={mossEdit}
            mossImageAdvanced={mossImageAdvanced}
            setSeed={setSeed}
            seed={seed}
            dimensionConstraints={dimensionConstraints}
            setReferenceDimensionIntent={setReferenceDimensionIntent}
            setWidthOverride={setWidthOverride}
            activeDimensionIntent={activeDimensionIntent}
            dropdownWidth={dropdownWidth}
            widthOverride={widthOverride}
            setHeightOverride={setHeightOverride}
            dropdownHeight={dropdownHeight}
            heightOverride={heightOverride}
            moss={moss}
            mossDmd={mossDmd}
            mossImage={mossImage}
            setMossField={setMossField}

            setSampler={setSampler}
            sampler={sampler}
            samplerOptions={samplerOptions}

            setScheduler={setScheduler}
            scheduler={scheduler}
            schedulerOptions={schedulerOptions}
            showSchedulerShift={showSchedulerShift}
            setSchedulerShift={setSchedulerShift}
            effectiveSchedulerShift={effectiveSchedulerShift}
            selectedModel={selectedModel}
            setMossDmdSteps={setMossDmdSteps}
            setStepsOverride={setStepsOverride}
            mossDmdSteps={mossDmdSteps}
            stepsOverride={stepsOverride}
            supportsGuidance={supportsGuidance}
            setGuidanceOverride={setGuidanceOverride}
            guidanceOverride={guidanceOverride}
            supportsCfgInterval={supportsCfgInterval}
            cfgIntervalDisabled={cfgIntervalDisabled}
            setCfgInterval={setCfgInterval}
            cfgInterval={cfgInterval}
            cfgIntervalError={cfgIntervalError}
            captionOverflowChoices={captionOverflowChoices}
            setCaptionOverflow={setCaptionOverflow}
            outgoingCaptionOverflow={outgoingCaptionOverflow}
            supportsTextStyle={supportsTextStyle}
            textStyleGainConfig={textStyleGainConfig}
            setTextStyleGain={setTextStyleGain}
            textStyleGain={textStyleGain}
            supportsTextEncoderSelection={supportsTextEncoderSelection}
            setTextEncoderModel={setTextEncoderModel}
            selectedTextEncoderModel={selectedTextEncoderModel}
            textEncoderOptions={textEncoderOptions}
            selectedTextEncoderAvailable={selectedTextEncoderAvailable}
            defaultTextEncoderId={defaultTextEncoderId}
            showGuidanceMethodPicker={showGuidanceMethodPicker}
            setGuidanceMethod={setGuidanceMethod}
            guidanceMethod={guidanceMethod}
            guidanceMethodOptions={guidanceMethodOptions}
            flashAttn={flashAttn}
            setFlashAttn={setFlashAttn}
            promptEnhance={promptEnhance}
            enhancePrompt={enhancePrompt}
            setEnhancePrompt={setEnhancePrompt}
            precisionToggle={precisionToggle}
            showTierPicker={showTierPicker}
            bf16Precision={bf16Precision}
            setRecipeTierRequest={setRecipeTierRequest}
            setBf16Precision={setBf16Precision}
            transparentBackground={transparentBackground}
            setTransparentBackground={setTransparentBackground}
            prompt={prompt}
            setPromptFromUser={setPromptFromUser}
            showPidToggle={showPidToggle}
            usePid={usePid}
            upscaleEnabled={upscaleEnabled}
            hiresFixEnabled={hiresFixEnabled}
            setUsePid={setUsePid}
            setDecoder={setDecoder}
            setPidTarget={setPidTarget}
            pidTarget={pidTarget}
            pidDecodeNotice={pidDecodeNotice}
            showDecoderPicker={showDecoderPicker}
            decoder={decoder}
            decoderOptions={decoderOptions}
            setUpscaleEnabled={setUpscaleEnabled}
            setHiresFixEnabled={setHiresFixEnabled}
            upscaleEngine={upscaleEngine}
            setUpscaleFactor={setUpscaleFactor}
            upscaleFactor={upscaleFactor}
            handleUpscaleEngineChange={handleUpscaleEngineChange}
            availableUpscaleEngines={availableUpscaleEngines}
            setUpscaleRestore={setUpscaleRestore}
            setRestorePlanStatus={setRestorePlanStatus}
            restorePlanBody={restorePlanBody}
            token={token}
            upscaleRestore={upscaleRestore}
            setUpscaleSoftness={setUpscaleSoftness}
            upscaleSoftness={upscaleSoftness}
            hiresFixDisabledReason={hiresFixDisabledReason}
            hiresFixDisabled={hiresFixDisabled}
            setHiresFixSteps={setHiresFixSteps}
            hiresFixSteps={hiresFixSteps}
            setHiresFixDenoisingStrength={setHiresFixDenoisingStrength}
            hiresFixDenoisingStrength={hiresFixDenoisingStrength}
            setHiresFixUpscaleBy={setHiresFixUpscaleBy}
            hiresFixUpscaleBy={hiresFixUpscaleBy}
            hiresFixTargetInvalid={hiresFixTargetInvalid}
            hiresFixTargetWidth={hiresFixTargetWidth}
            hiresFixTargetHeight={hiresFixTargetHeight}
            setHiresFixCfgScale={setHiresFixCfgScale}
            hiresFixCfgScale={hiresFixCfgScale}
            supportsNegativePrompt={supportsNegativePrompt}
            negativePromptCfgOff={negativePromptCfgOff}
            setNegativePrompt={setNegativePrompt}
            negativePrompt={negativePrompt}
          />

          {/* Krea 2 multi-phase denoise (epic 13879 S5, sc-13885). Its own EXPERIMENTAL disclosure,
              default collapsed (busy-page guidance), shown only for the model that advertises the
              acceleration LoRA compat (krea_2_raw) in text-to-image mode — matching the worker gate.
              The phase editor references the LoRAs picked in the Advanced section above by index. */}
          {multiPhaseAvailable ? (
            <AdvancedSection
              className="multiphase-section"
              hint="experimental — split the Raw denoise into phases"
              label="Multi-phase denoise (experimental)"
              onToggle={() => setMultiPhaseOpen((value) => !value)}
              open={multiPhaseOpen}
            >
              <MultiPhaseEditor
                enabled={multiPhaseEnabled}
                onApplyTurboFinish={applyTurboFinishPreset}
                onChange={setMultiPhasePhases}
                onToggleEnabled={setMultiPhaseEnabled}
                phases={multiPhasePhases}
                selectedLoras={selectedLoras}
              />
            </AdvancedSection>
          ) : null}

          {/* The preset/LoRA problems that used to be three separate .inline-warning
              paragraphs (sc-10649). Requirements — no project, empty prompt — stay silent. */}
          <ValidationSummary issues={generateValidity.surfaced} label="Generate errors" />

        </WorkPanel>

        <div className="studio-results">
          <ImageReviewPanel
            localJobGroups={localJobGroups}
            onPreview={onPreview}
            onCancelJob={onCancelJob}
            onOpenQueue={onOpenQueue}
            latestAssets={latestAssets}
            deleteAsset={deleteAsset}
            purgeAsset={purgeAsset}
            updateAssetStatus={updateAssetStatus}
          />
        </div>
      </form>
      <StudioPromptGuide open={guideOpen} guide={promptGuide} modelName={selectedModel?.name} onClose={closeGuide} />
    </section>
    </ModelAvailabilityGate>
  );
}
