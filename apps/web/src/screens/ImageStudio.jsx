import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AssetPickerField, ImageEditSourcePickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia, assetUrl } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { PoseLibraryPicker } from "../components/PoseLibraryPicker.jsx";
import { RefinePromptControl } from "../components/RefinePromptControl.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../components/StudioUpdateNotice.jsx";
import StructuredPromptBuilder from "../components/StructuredPromptBuilder.jsx";
import ReferenceCaptionPicker from "../components/ReferenceCaptionPicker.jsx";
import BatchPromptPanel from "../components/BatchPromptPanel.jsx";
import {
  cardinality,
  expandBatch,
  extractKeys,
  linkedGroupIssues,
  missingKeys,
  parsePromptResolution,
  splitPromptLines,
} from "../promptBatch.js";
import {
  MAX_IMAGE_DIMENSION,
  MIN_IMAGE_DIMENSION,
  resolveEffectiveDimensions,
} from "../resolutionOverride.js";
import { pidDecodeHeadsUp } from "../pidDecodeNotice.js";
import { batchItemStatus, summarizeBatchRun } from "../batchOps.js";
import {
  DEFAULT_SCENE_PROMPT,
  promptHintFor,
  promptSeedFor,
  seedsNegativeInMode,
} from "../promptSeed.js";
import {
  emptyCaption,
  injectStyleIntoCaption,
  orderCaption,
  parseMagicPromptCaption,
  parseVisionCaption,
  serializeCaption,
  validateCaption,
} from "../ideogramCaption.js";
import { buildImageJobRequest } from "../imageJobRequest.js";
import { usePoseLibrary, useUserPoseLoader } from "../poseLibrary.js";

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
import {
  finiteNumberOrUndefined,
  findModelEditLora,
  loraIsInstalled,
  serializeLora,
  noPresetId,
  composePreset,
} from "../presetUtils.js";
import {
  LoraPickerSection,
  onPromptKeyDown,
  PresetGuidanceStrip,
  PresetStackPreview,
  SavePresetPanel,
  useGenerationStudio,
  useSavePreset,
} from "./generationStudio.jsx";
import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import {
  batchPromptBudgetMessage,
  batchPromptBudgetOverages,
  imageBatchValidation,
  imageGenerateValidation,
} from "../imageStudioValidation.js";
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
import {
  allPossibleTiers,
  defaultTierSelection,
  installedTiers,
  isBelowFloor,
  modelQualityFloor,
  tierLabel,
  tierPickerOptions,
} from "../quantTier.js";
import { suggestTier } from "../tierSuggestion.js";
import { useUnifiedMemoryGb } from "../hooks/useUnifiedMemoryGb.js";
import { readLastTier, writeLastTier } from "../lastTierStore.js";
import { readDefaultGenerationQuality } from "../generationQuality.js";
import { PROMPT_REFINE_MODEL_ID, VISION_CAPTION_MODEL_ID, VISION_CAPTION_MODEL_REPO } from "../constants.js";
import { parseResolution, pickClosestResolution } from "../resolutionMatch.js";
import { fitsResolutionOptions } from "../resolutionMemory.js";
import { finiteRecipeNumber, recipeLoraSelection, recipeResolution } from "../recipeFields.js";
import {
  DEFAULT_MAC_CAPABILITIES,
  macAvailableModels,
  macGatingActive,
  macModelFeatureBlock,
} from "../macGating.js";
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import { resolveJobResultAssets } from "../jobResultAssets.js";
import {
  availableUpscaleEngines as upscaleEnginesForPlatform,
  upscaleEngineHasSoftness,
  upscaleFactorsForEngine,
  useUpscaleEngineFallback,
} from "../upscaleEngines.js";
import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { StylePicker } from "../components/StylePicker.jsx";
import { StyledPromptPreview } from "../components/StyledPromptPreview.jsx";
import { STYLE_GROUPS, styleHintForId, styleTextForId } from "../data/styleCatalog.js";
import {
  GUIDANCE_METHOD_LABELS,
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  guidanceMethodDefaultFromModel,
  guidanceMethodOptionsFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  schedulerShiftDefaultFromModel,
  stepsDefaultFromModel,
} from "../samplerOptions.js";

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
const BATCH_RENDER_CAP = 100;

// Join a saved batch's prompts back into the authoring textarea: multi-line prompts
// round-trip through the `---` delimiter, a flat list joins on newlines.
function batchTextFromPrompts(prompts) {
  const list = Array.isArray(prompts) ? prompts : [];
  return list.join(list.some((prompt) => prompt.includes("\n")) ? "\n---\n" : "\n");
}

function preferredOption(defaultValue, options) {
  return options.includes(defaultValue) ? defaultValue : options[0] ?? "default";
}

function preferredResolution(model, options) {
  const modelDefault = model?.defaults?.resolution;
  return options.includes(modelDefault)
    ? modelDefault
    : options.includes("1024x1024")
      ? "1024x1024"
      : options[0];
}

function formatResolutionLabel(value) {
  const [width, height] = String(value).split("x");
  return height ? `${width} × ${height}` : value;
}

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
  return resolveJobResultAssets(job, assets, { type: "image", sortByBatchIndex: true });
}

function jobExpectedCount(job, completedCount) {
  const expected = Number(job.result?.expectedCount ?? job.result?.count ?? job.payload?.count);
  return Number.isFinite(expected) && expected > 0 ? Math.max(expected, completedCount) : completedCount;
}

export function ImageStudio() {
  const {
    activeProject,
    assets,
    characters,
    createImageJob,
    createPreset,
    refinePrompt,
    magicPrompt,
    imageCaption,
    imageDescribe,
    createModelDownloadJob,
    createLoraDownloadJob,
    deleteAsset,
    purgeAsset,
    gpuOptions,
    imageModels,
    models = [],
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
    visibleWorkers = [],
  } = useAppContext();
  // Krea 2 INT8-ConvRot eligibility (sc-9300, epic 9083): the candle-only tier is offered ONLY when a
  // live worker advertises the `int8_convrot` capability — which the worker emits solely on the candle
  // lane AND when its GPU clears the sm_89 compute-cap floor (gpu.rs). So macOS/MLX and pre-Ada NVIDIA
  // hosts (where no worker advertises it) HIDE the tier gracefully rather than only failing at submit.
  const convRotEligible = useMemo(
    () =>
      visibleWorkers.some(
        (worker) =>
          worker?.status !== "offline" &&
          Array.isArray(worker?.capabilities) &&
          worker.capabilities.includes("int8_convrot"),
      ),
    [visibleWorkers],
  );
  // NVFP4 eligibility (sc-11042, epic 11037): identical shape to the ConvRot gate above — the
  // candle-only FP4 tier is offered ONLY when a live worker advertises the `nvfp4` capability, which
  // the worker emits solely on the candle lane AND when its GPU clears the sm_120 consumer-Blackwell
  // compute-cap floor (gpu.rs). macOS/MLX (no FP4 hardware) and pre-Blackwell NVIDIA hosts HIDE the
  // tier rather than only failing at submit. Hiding is the picker-side gate; the worker independently
  // re-checks the cap at tier-select, so this is UX, not the security boundary.
  const nvfp4Eligible = useMemo(
    () =>
      visibleWorkers.some(
        (worker) =>
          worker?.status !== "offline" &&
          Array.isArray(worker?.capabilities) &&
          worker.capabilities.includes("nvfp4"),
      ),
    [visibleWorkers],
  );
  const tierOptions = useMemo(
    () => ({ convRotEligible, nvfp4Eligible }),
    [convRotEligible, nvfp4Eligible],
  );
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
  const [batchMode, setBatchMode] = useState(saved.batchMode ?? false);
  const [batchPromptsText, setBatchPromptsText] = useState(saved.batchPromptsText ?? "");
  const [batchVariableValues, setBatchVariableValues] = useState(saved.batchVariableValues ?? {});
  const [batchName, setBatchName] = useState(saved.batchName ?? "");
  const [batchScope, setBatchScope] = useState(saved.batchScope ?? "global");
  const [loadedBatchId, setLoadedBatchId] = useState(saved.loadedBatchId ?? null);
  const [batchError, setBatchError] = useState("");
  const [batchBusy, setBatchBusy] = useState(false);
  // An in-flight / just-finished batch run: { submitting, items: [{ prompt, jobId }] }.
  // Progress + cancel are derived off the live jobs feed, mirroring the asset batch (sc-6112).
  const [batchRun, setBatchRun] = useState(null);
  // True once a run over BATCH_RENDER_CAP is awaiting the user's explicit confirmation.
  const [batchConfirmPending, setBatchConfirmPending] = useState(false);
  // Set by Stop/Cancel to break the (possibly slow, structured) enqueue loop mid-flight.
  const batchAbortRef = useRef(false);

  const batchPrompts = useMemo(() => splitPromptLines(batchPromptsText), [batchPromptsText]);
  const batchVariables = useMemo(
    () =>
      extractKeys(batchPrompts).map((key) => ({
        key,
        // The value editor keeps a trailing empty slot; drop blanks so saved batches
        // and the run payload carry only real values (the engine ignores them anyway).
        values: (batchVariableValues[key] ?? []).filter((value) => value.trim() !== ""),
      })),
    [batchPrompts, batchVariableValues],
  );
  // Number of resolved-prompt jobs (pose-independent). Image count = jobs × images-per-prompt,
  // computed as batchTotal once the pose payload is known (poses replace `count`).
  const batchJobCount = useMemo(
    () => cardinality(batchPrompts, batchVariables, 1),
    [batchPrompts, batchVariables],
  );

  const applyBatchContent = useCallback(({ prompts, variables, lastValues, name }) => {
    setBatchPromptsText(batchTextFromPrompts(prompts));
    const values = {};
    for (const variable of variables ?? []) {
      if (variable?.key) values[variable.key] = Array.isArray(variable.values) ? variable.values : [];
    }
    for (const [key, vals] of Object.entries(lastValues ?? {})) {
      if (!(key in values) && Array.isArray(vals)) values[key] = vals;
    }
    setBatchVariableValues(values);
    if (name !== undefined) setBatchName(name ?? "");
    setBatchError("");
  }, []);

  const handleSaveBatch = useCallback(async () => {
    setBatchBusy(true);
    setBatchError("");
    try {
      const payload = {
        name: batchName.trim(),
        scope: batchScope,
        prompts: batchPrompts,
        variables: batchVariables,
        lastValues: Object.fromEntries(batchVariables.map((variable) => [variable.key, variable.values])),
      };
      const result = loadedBatchId
        ? await updatePromptBatch(loadedBatchId, payload, batchScope)
        : await createPromptBatch(payload);
      if (result?.id) setLoadedBatchId(result.id);
    } catch (err) {
      setBatchError(err.message);
    } finally {
      setBatchBusy(false);
    }
  }, [batchName, batchScope, batchPrompts, batchVariables, loadedBatchId, updatePromptBatch, createPromptBatch]);

  const handleLoadBatch = useCallback(
    (batch) => {
      applyBatchContent(batch);
      setBatchScope(batch.scope === "project" ? "project" : "global");
      setLoadedBatchId(batch.id ?? null);
    },
    [applyBatchContent],
  );

  const handleDeleteBatch = useCallback(
    async (batch) => {
      setBatchError("");
      try {
        await deletePromptBatch(batch.id, batch.scope);
        setLoadedBatchId((current) => (current === batch.id ? null : current));
      } catch (err) {
        setBatchError(err.message);
      }
    },
    [deletePromptBatch],
  );

  const handleImportBatch = useCallback(
    (payload) => {
      applyBatchContent(payload);
      setLoadedBatchId(null);
    },
    [applyBatchContent],
  );

  // Detach from the loaded batch and clear the authoring fields so the next Save creates a
  // brand-new batch. Needed because loadedBatchId now persists in the studio snapshot: after
  // a restart the panel restores its last-loaded batch and Save is stuck on "Update" with no
  // way back to a blank slate. Scope is left as-is (a user preference, not batch content).
  const handleNewBatch = useCallback(() => {
    setBatchPromptsText("");
    setBatchVariableValues({});
    setBatchName("");
    setLoadedBatchId(null);
    setBatchError("");
  }, []);
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  const [model, setModel] = useState(saved.model ?? imageModels[0]?.id ?? "z_image_turbo");
  const [seed, setSeed] = useState(saved.seed ?? "");
  const [negativePrompt, setNegativePrompt] = useState(saved.negativePrompt ?? "");
  const [resolution, setResolution] = useState(saved.resolution ?? "1024x1024");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.id ?? "");
  // Multi-image reference set for a multi-reference edit (sc-6211, FLUX.2-dev). Drives the plural
  // `referenceAssetIds` payload when the model declares `ui.multiReference` and the user is in
  // edit_image mode (replaces the single source picker). Empty for every other model/mode.
  const [referenceAssetIds, setReferenceAssetIds] = useState(() =>
    Array.isArray(saved.referenceAssetIds) ? saved.referenceAssetIds : [],
  );
  // Optional SECOND source for a Krea-style two-reference edit (epic 10871 P1.3). Any two images —
  // the `sourceAssetId` above is image 1 (required), this is image 2 (optional), a fixed order (the
  // instruction describes how to combine them). Only surfaced when the model declares
  // `ui.editReferences`; empty otherwise. When set, submit sends an ordered
  // `referenceAssetIds: [image1, image2]` pair instead of the single `sourceAssetId`.
  const [editSecondAssetId, setEditSecondAssetId] = useState("");
  // Edit fit mode (epic 2551): how the source is fitted to the output W×H. Never stretch.
  const [fitMode, setFitMode] = useState(saved.fitMode ?? "crop");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  // Character reference (IP-Adapter / InstantID) — the approved reference image whose
  // identity is carried across variations. `ipAdapterScale` rides in `advanced`; for
  // InstantID, `controlnetScale` (IdentityNet landmark lock) rides there too.
  const [referenceAssetId, setReferenceAssetId] = useState("");
  const [ipAdapterScale, setIpAdapterScale] = useState(saved.ipAdapterScale ?? 0.6);
  // img2img reference-guided generation (epic 8588 slice A, sc-8593): the reference picked in the
  // "Start from an image" panel (lifted from ReferenceCaptionPicker) drives generation at this
  // strength on an img2img-capable model (Krea 2 Turbo). Distinct from the character `referenceAssetId`
  // above — same picker, different purpose. Default 0.5 (the full-range slider midpoint; the usable
  // band is model-specific, so no clamp beyond the slider's 0–1).
  const [img2imgReferenceAssetId, setImg2imgReferenceAssetId] = useState("");
  const [img2imgStrength, setImg2imgStrength] = useState(saved.img2imgStrength ?? 0.5);
  // Krea "text style" tap-reweight gain (sc-11878): the ui.textStyleGain slider (Krea/Qwen-family).
  // 1.0 = no-op; >1 warmer/richer (early Qwen3-VL taps), <1 late-biased. Resets to the model default on
  // model change like the other tuning knobs; emitted to advanced.textStyleGain only when off default.
  const [textStyleGain, setTextStyleGain] = useState(saved.textStyleGain ?? 1.0);
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
  // Strict-control panel (epic 8236, sc-8245). The selected control type (pose / canny / depth),
  // gated to the backbone's `ui.controlModes`. Pose reuses `selectedPoseIds`; canny/depth use a
  // control-image asset + a preprocess-vs-passthrough toggle. `controlScale` (advanced.controlScale)
  // is the control-lock strength. All reset to the model's defaults on model change (below).
  const [controlMode, setControlMode] = useState(saved.controlMode ?? "pose");
  const [controlImageAssetId, setControlImageAssetId] = useState("");
  // false = preprocess (worker auto-derives the map from the image → request.sourceAssetId);
  // true = use-as-is (the user-supplied map passes through verbatim → advanced.controlImage).
  const [controlImagePassthrough, setControlImagePassthrough] = useState(false);
  const [controlScale, setControlScale] = useState(saved.controlScale ?? null);
  // Selected trained ControlNet overlay id (sc-10165 B4) — for backbones whose pose control rides a
  // registered overlay (Krea 2 Turbo). Flows to advanced.controlWeights.overlayId; the API resolves it.
  const [controlOverlayId, setControlOverlayId] = useState(null);
  // Configurable sampler / scheduler (epic 1753). Restored from per-workspace
  // settings; reset to the selected model's manifest defaults whenever the
  // model changes.
  const [sampler, setSampler] = useState(saved.sampler ?? "default");
  const [scheduler, setScheduler] = useState(saved.scheduler ?? "default");
  const [schedulerShift, setSchedulerShift] = useState(saved.schedulerShift ?? 3.0);
  // Guidance method (epic 7434). "cfg" is the engine-standard no-op default; the
  // picker only surfaces alternatives a model advertises on the active backend
  // (CFG++ on the SDXL family / MLX). Rides `advanced.guidanceMethod`.
  const [guidanceMethod, setGuidanceMethod] = useState(saved.guidanceMethod ?? "cfg");
  // Steps / guidance: previously worker-only knobs surfaced via this same
  // advanced panel. "" represents "use the model default" so the user can
  // clear the override.
  const [stepsOverride, setStepsOverride] = useState(saved.steps ?? "");
  const [guidanceOverride, setGuidanceOverride] = useState(saved.guidanceScale ?? "");
  // Advanced resolution override: a custom Width/Height to experiment beyond the model's
  // pre-declared Aspect options (e.g. Krea 2 up to 4K). "" = "use the Aspect dropdown" for
  // that axis, mirroring the Steps/Guidance overrides. Effective dims are derived below and
  // ride the existing top-level width/height payload; the backend caps each at 256–4096.
  const [widthOverride, setWidthOverride] = useState(saved.widthOverride ?? "");
  const [heightOverride, setHeightOverride] = useState(saved.heightOverride ?? "");
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
  const [bf16Precision, setBf16Precision] = useState(saved.bf16Precision ?? false);
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
  const [quantTier, setQuantTier] = useState("");
  // Brief "loading <tier>" hint shown right after a switch (reload-always): switching a heavy
  // tier evicts + reloads on the worker, so we surface a transient loading note rather than
  // implying an instant swap. Cleared on a short timer; purely cosmetic.
  const [tierSwitching, setTierSwitching] = useState("");
  // PiD decoder toggle (epic 7840, sc-7851): off = the model's native VAE decode; on emits
  // `advanced.usePid: true`, routing decode through the optional PiD pixel-diffusion decoder
  // (decode + 2K/4K super-resolve, non-commercial output). Sticky pref, default off; the
  // toggle only renders + emits when the model is PiD-eligible AND its checkpoint is installed
  // (showPidToggle), so a stale `true` on a non-eligible model is inert — mirrors bf16Precision.
  const [usePid, setUsePid] = useState(saved.usePid ?? false);
  // PiD output tier (epic 7840, sc-10054): PiD always super-resolves the base latent 4×, so this picks
  // the effective base — "4k" keeps the requested base (~4096 output, the pre-tier behavior), "2k" caps
  // it (~2048 output, faster + less GPU memory). Sticky pref, default "4k". Rides `advanced.pidTarget`
  // (emitted only when the PiD toggle is shown+on AND "2k" is picked — "4k" is the worker default).
  const [pidTarget, setPidTarget] = useState(saved.pidTarget === "2k" ? "2k" : "4k");
  const [faceRestore, setFaceRestore] = useState(false);
  // User-created poses (reserved global project) join the built-in library in both
  // the picker and the id→keypoints resolver below, so saved poses can generate.
  const loadUserPoses = useUserPoseLoader();
  const { byId: poseById } = usePoseLibrary({ loadUserPoses });
  const [upscaleEnabled, setUpscaleEnabled] = useState(saved.upscaleEnabled ?? false);
  const [upscaleFactor, setUpscaleFactor] = useState(saved.upscaleFactor ?? 2);
  const [upscaleEngine, setUpscaleEngine] = useState(saved.upscaleEngine ?? "real-esrgan");
  // SeedVR2 detail/softness knob (0..1, sc-4815) — only used by the seedvr2 engine.
  const [upscaleSoftness, setUpscaleSoftness] = useState(saved.upscaleSoftness ?? 0);
  const [submitting, setSubmitting] = useState(false);
  // Auto-expand state (sc-6501): when a structured model is in plain-text mode, Generate first
  // expands the idea into a JSON caption via magic-prompt. `expanding` drives the button label;
  // `submitError` surfaces an expansion failure (e.g. the utility model isn't installed) without
  // ever falling back to sending raw plain text.
  const [expanding, setExpanding] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [guideOpen, setGuideOpen] = useState(false);
  // Prompt tools (epic UI-refinement): which inline prompt-tool panel is open —
  // null | "describe" (reference-image caption) | "refine" (rewrite my prompt).
  // Replaces the always-rendered ReferenceCaptionPicker + RefinePromptControl pair
  // with two toggle tiles; only one panel opens at a time.
  const [promptTool, setPromptTool] = useState(null);
  const togglePromptTool = useCallback(
    (tool) => setPromptTool((current) => (current === tool ? null : tool)),
    [],
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

  function handleUpscaleEngineChange(nextEngine) {
    setUpscaleEngine(nextEngine);
    const factors = upscaleFactorsForEngine(nextEngine);
    if (!factors.includes(upscaleFactor)) {
      setUpscaleFactor(factors[0]);
    }
  }

  // Engines offered in the picker; AuraSR is dropped on every platform (sc-3668 / sc-5499).
  const availableUpscaleEngines = upscaleEnginesForPlatform(macCapabilities);
  // If a restored/saved engine is gated out (e.g. a stale saved AuraSR selection), fall back to the
  // default real-esrgan engine so the user never submits an aura-sr job the native workers refuse (sc-8853).
  useUpscaleEngineFallback({
    macCapabilities,
    upscaleEngine,
    setUpscaleEngine,
    upscaleFactor,
    setUpscaleFactor,
  });

  // PiD decode and Upscale both super-resolve, so they're mutually exclusive in the UI (each
  // disables the other while active). If a saved/preset state carries both on, drop PiD (keep
  // Upscale) so neither checkbox is left permanently disabled.
  useEffect(() => {
    if (usePid && upscaleEnabled) {
      setUsePid(false);
    }
  }, [usePid, upscaleEnabled]);

  useEffect(() => {
    if (mode === "edit_image" && selectedAssetEditableSourceId) {
      setSourceAssetId(selectedAssetEditableSourceId);
    }
  }, [mode, selectedAssetEditableSourceId]);

  useEffect(() => {
    if (launchRequest?.view !== "Image") {
      return;
    }
    if (launchRequest.recipe) {
      return;
    }
    if (launchRequest.characterId) {
      setMode(launchRequest.mode ?? "character_image");
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
    setMode(launchRequest.mode);
    // Preselect the family-matched edit model resolved at launch time (App.jsx). It's
    // edit-capable by construction, so the availableModels snap-to-first effect leaves
    // it in place; when absent the snap falls back to the default edit model.
    if (launchRequest.model) {
      setModel(launchRequest.model);
    }
    if (launchRequest.mode === "edit_image" && selectedAssetEditableSourceId) {
      setSourceAssetId(selectedAssetEditableSourceId);
    }
  }, [launchRequest?.id, selectedAsset?.id, selectedAssetEditableSourceId]);

  // Mac UI gating (sc-3486): on a Mac in MLX-required mode, hide torch-only models from the
  // picker so the user can't select something that would only error. Inert elsewhere.
  const macImageModels = useMemo(
    () => macAvailableModels(imageModels, macCapabilities),
    [imageModels, macCapabilities],
  );
  const macGating = macGatingActive(macCapabilities);
  const imageModelServesMode = useCallback((item, value) => {
    const caps = item?.capabilities ?? [];
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
    return caps.includes("text_to_image");
  }, [macCapabilities]);
  const modelsForMode = useCallback(
    (value) => macImageModels.filter((item) => imageModelServesMode(item, value)),
    [imageModelServesMode, macImageModels],
  );
  const availableModels = useMemo(
    () => modelsForMode(mode),
    [mode, modelsForMode],
  );
  const pickerModels = mode === "text_to_image" && availableModels.length === 0 ? macImageModels : availableModels;
  // Model-availability gate (sc-5947): when the user has no mac-available image model at all,
  // show recommended image-model downloads instead of the studio. `ready` matches the picker
  // (which falls back to all macImageModels for the text tab); offers come from the full catalog
  // via imageModelUsable, recommended-first.
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
  const selectedModel = imageModels.find((item) => item.id === model);
  // Booru-convention prompt hint (sc-10760): non-null for danbooru-tag models (Anima, Illustrious)
  // that declare `ui.promptHint`; rendered under the prompt box with a link into the prompt guide.
  const promptHint = promptHintFor(selectedModel?.ui);
  // Prompt guide for the selected model; fall back to the generic image guide
  // when a model declares none, so the button is always useful (sc-1817).
  const promptGuide = selectedModel?.ui?.promptGuide ?? {
    title: "Image Prompt Guide",
    path: "/prompt-guides/generic-image.md",
  };
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
          .map((pose) => ({ id: pose.id, keypoints: pose.keypoints }))
      : [];
  const controlActive = showControlPanel && Boolean(activeControlMode);
  const controlIsImageMode = controlActive && activeControlMode !== "pose";
  const controlPreprocessSourceId =
    controlIsImageMode && !controlImagePassthrough && controlImageAssetId ? controlImageAssetId : null;
  const controlPassthroughId =
    controlIsImageMode && controlImagePassthrough && controlImageAssetId ? controlImageAssetId : null;

  // Images each resolved-prompt job emits: the pose count when poses are selected (they
  // replace `count` variations), else `count`. Total batch image count feeds the run label
  // and the cardinality cap.
  const batchImagesPerPrompt = posePayload.length || count;
  const batchTotal = batchJobCount * batchImagesPerPrompt;

  // A pending large-run confirmation is for one specific total — reset it whenever the batch
  // size changes so the user always re-confirms against the current count.
  useEffect(() => {
    setBatchConfirmPending(false);
  }, [batchTotal]);
  // Whether the model exposes its built-in prompt upsampler ("Enhance prompt" toggle) — FLUX.2-dev.
  const promptEnhance = Boolean(selectedModel?.ui?.promptEnhance);
  // Whether the model ships a packed default + a hosted full-precision bf16 build, exposing the
  // Studio "Full precision (bf16)" toggle (sc-6568) — Boogu Base/Turbo/Edit.
  const precisionToggle = Boolean(selectedModel?.ui?.precisionToggle);
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
  const unifiedMemoryGb = useUnifiedMemoryGb();
  const autoTier = useMemo(
    () => suggestTier(selectedModel, unifiedMemoryGb),
    [selectedModel, unifiedMemoryGb],
  );
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
    setIpAdapterScale(typeof ui.referenceStrengthDefault === "number" ? ui.referenceStrengthDefault : 0.6);
    setControlnetScale(typeof ui.identityStructure?.default === "number" ? ui.identityStructure.default : 0.8);
    setTrueCfgScale(typeof ui.variationStrength?.default === "number" ? ui.variationStrength.default : 4.0);
    setViewAngle("");
    setSelectedPoseIds([]);
    // Re-gate the strict-control panel to the new backbone: snap the control type to a supported mode
    // (an unsupported pick — e.g. canny on a pose-only backbone — resets to the first supported one),
    // reset the control-scale to the model's manifest default, and clear a stale control image.
    const nextModes = supportedControlModes(imageModels.find((item) => item.id === model));
    setControlMode((current) => (nextModes.includes(current) ? current : nextModes[0] ?? "pose"));
    setControlScale(typeof ui.controlScale?.default === "number" ? ui.controlScale.default : null);
    // Krea text-style gain: snap to the new model's declared default (1.0 = no-op when undeclared).
    setTextStyleGain(typeof ui.textStyleGain?.default === "number" ? ui.textStyleGain.default : 1.0);
    setControlImageAssetId("");
    setControlImagePassthrough(false);
    // Clear a stale overlay pick so an id trained for a different backbone can't leak into a submit
    // (sc-10165 B4); the picker re-fetches for the new backbone.
    setControlOverlayId(null);
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
    setIpAdapterScale(typeof ui.referenceStrengthDefault === "number" ? ui.referenceStrengthDefault : 0.6);
    setControlnetScale(typeof ui.identityStructure?.default === "number" ? ui.identityStructure.default : 0.8);
    setTrueCfgScale(typeof ui.variationStrength?.default === "number" ? ui.variationStrength.default : 4.0);
    setControlScale(typeof ui.controlScale?.default === "number" ? ui.controlScale.default : null);
    setTextStyleGain(typeof ui.textStyleGain?.default === "number" ? ui.textStyleGain.default : 1.0);
  }, [model, imageModels]);
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
    if (mode !== "character_image") {
      return;
    }
    if (characterReferences.some((reference) => reference.assetId === referenceAssetId)) {
      return;
    }
    setReferenceAssetId(characterReferences[0]?.assetId ?? "");
  }, [mode, characterReferences, referenceAssetId]);
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
    const ui = imageModels.find((item) => item.id === model)?.ui ?? {};
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
    const backend = macCapabilities?.macGatingActive ? "mlx" : "candle";
    const gated = fitsResolutionOptions(selectedModel, declared, unifiedMemoryGb, { backend });
    // Never collapse to an empty picker: if the gate somehow trims everything (it never trims
    // ≤1536², so this is defensive), fall back to the declared list.
    return gated.length > 0 ? gated : declared;
  }, [selectedModel, unifiedMemoryGb, macCapabilities]);
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
  // Auto-preset the Aspect from the picked img2img reference (sc-10195): matching the reference's
  // aspect keeps the latent-init composition valid (a 4:5 reference rendered at 16:9 comes out
  // wrong-shaped). Mirrors the describe picker's probe (sc-8109/8220) — keyed on the id AND the asset
  // list so a freshly imported reference re-runs once it resolves. User Aspect override still wins.
  useEffect(() => {
    if (!img2imgReferenceAssetId) return;
    const asset = editImageAssets.find((item) => item.id === img2imgReferenceAssetId);
    const src = asset && assetUrl(asset);
    if (!src || typeof Image === "undefined") return;
    const probe = new Image();
    probe.onload = () => {
      if (probe.naturalWidth && probe.naturalHeight) {
        onReferenceImageLoaded(probe.naturalWidth, probe.naturalHeight);
      }
    };
    probe.src = src;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [img2imgReferenceAssetId, editImageAssets]);
  // Sampler / scheduler menus declared by the model, gated to the ACTIVE backend
  // (epic 7114 P5): `macGatingActive` is the worker `mlx_required` master switch, so
  // it picks the manifest's `mlx.limits` override on Mac/MLX and the `candle.limits`
  // override on the Windows/Linux candle build (e.g. Lens exposes the curated menu
  // only on candle; SDXL only on MLX). The advanced panel hides the dropdowns when
  // the menu has fewer than 2 options (epic 1753 §7.4).
  const activeBackend = macCapabilities?.macGatingActive ? "mlx" : "candle";
  const samplerOptions = useMemo(
    () => samplerOptionsFromModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );
  const schedulerOptions = useMemo(
    () => schedulerOptionsFromModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );
  const guidanceMethodOptions = useMemo(
    () => guidanceMethodOptionsFromModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );
  const showSamplerPicker = samplerOptions.length > 1;
  const showSchedulerPicker = schedulerOptions.length > 1;
  const showGuidanceMethodPicker = guidanceMethodOptions.length > 1;
  const advancedDefaultsModel = useRef(model);
  const skipAdvancedDefaultsReset = useRef(false);
  useEffect(() => {
    if (advancedDefaultsModel.current === model) {
      return;
    }
    advancedDefaultsModel.current = model;
    if (skipAdvancedDefaultsReset.current) {
      skipAdvancedDefaultsReset.current = false;
      return;
    }
    setSampler(preferredOption(samplerDefaultFromModel(selectedModel), samplerOptions));
    setScheduler(preferredOption(schedulerDefaultFromModel(selectedModel), schedulerOptions));
    setSchedulerShift(schedulerShiftDefaultFromModel(selectedModel));
    setGuidanceMethod(
      preferredOption(guidanceMethodDefaultFromModel(selectedModel), guidanceMethodOptions),
    );
    setResolution(preferredResolution(selectedModel, resolutionOptions));
    setStepsOverride("");
    setGuidanceOverride("");
  }, [
    model,
    resolutionOptions,
    samplerOptions,
    schedulerOptions,
    guidanceMethodOptions,
    selectedModel,
  ]);
  // Snap the sampler / scheduler back to the model's declared default when the
  // current value is no longer in the menu (e.g. user switched to a sealed
  // model whose only option is "default"). Mirrors the resolution-snap effect.
  // Guard on a resolved model (sc-11962): before the model catalog loads,
  // `samplerOptions` falls back to ["default"], so an un-guarded snap would revert a
  // restored non-default sampler during the restart-restore window and never recover.
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (samplerOptions.includes(sampler)) {
      return;
    }
    setSampler(preferredOption(samplerDefaultFromModel(selectedModel), samplerOptions));
  }, [samplerOptions, sampler, selectedModel]);
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (schedulerOptions.includes(scheduler)) {
      return;
    }
    setScheduler(preferredOption(schedulerDefaultFromModel(selectedModel), schedulerOptions));
  }, [schedulerOptions, scheduler, selectedModel]);
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
      preferredOption(guidanceMethodDefaultFromModel(selectedModel), guidanceMethodOptions),
    );
  }, [guidanceMethodOptions, guidanceMethod, selectedModel]);
  // Keep the selected resolution valid for the current model's buckets. Switching
  // to a model whose options exclude the current value snaps to its default (or
  // 1024x1024, then the first option) rather than leaving a stale, unselectable value.
  // Guard on a resolved model (sc-11962): before the catalog loads `resolutionOptions`
  // falls back to DEFAULT_RESOLUTION_OPTIONS, so an un-guarded snap would revert a
  // restored model-specific resolution (e.g. 1536x1536) during restart-restore.
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (resolutionOptions.includes(resolution)) {
      return;
    }
    setResolution(preferredResolution(selectedModel, resolutionOptions));
  }, [resolutionOptions, resolution, selectedModel]);
  // Keep the selected quant tier valid for the active model (sc-8515). When the current tier is
  // still installed for this model, leave it; otherwise snap to the model's default selection
  // (sticky-for-this-(screen,model) → declared default → q8 base → q4 → first installed). Clears "" when
  // no tier is installed / the model has no matrix, so a stale tier never leaks into the payload.
  // The sticky rung (sc-10727) is read straight from the persistent per-(screen,model) store, so it
  // survives restarts and is honored above the base default whenever that tier is still installed.
  // Keyed on `model` (not `selectedModel`) plus the installed-tier list so a catalog refresh that
  // newly installs a second tier re-derives the default without churning on every render.
  const availableTiersKey = availableTiers.join(",");
  useEffect(() => {
    if (availableTiers.includes(quantTier)) {
      return;
    }
    setQuantTier(
      defaultTierSelection(selectedModel, readLastTier(TIER_SCREEN, model), {
        ...tierOptions,
        // Rung 3 (sc-10728): the app-wide default-generation-quality setting is the base default below
        // the per-(screen,model) sticky. Read fresh here (like readLastTier) so a change made in Settings
        // is picked up the next time this effect derives a default — no stale in-memory copy.
        defaultQuality: readDefaultGenerationQuality(),
        // When that setting is Auto (the default), the base is this capability-aware suggestion.
        autoTier,
      }) ?? "",
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, availableTiersKey, autoTier]);
  // Switch the active quant tier (sc-8515): persist it as this (screen, model)'s last EXPLICIT tier
  // (sc-10727 — sticky) and surface a brief "loading <tier>" note (reload-always — the worker evicts
  // + reloads a heavy tier on the next generation; there is no co-residence). The note self-clears.
  const tierSwitchTimer = useRef(null);
  useEffect(() => () => clearTimeout(tierSwitchTimer.current), []);
  const handleTierChange = useCallback(
    (nextTier) => {
      // Only an installed-and-complete tier can be selected — the disabled options in the dropdown are
      // shown for discoverability, never as a pickable target. This is the belt behind the native
      // `<option disabled>` (which already blocks selection) so no code path can strand the state on a
      // tier that isn't in the cache.
      if (nextTier === quantTier || !availableTiers.includes(nextTier)) {
        return;
      }
      setQuantTier(nextTier);
      writeLastTier(TIER_SCREEN, model, nextTier);
      setTierSwitching(nextTier);
      clearTimeout(tierSwitchTimer.current);
      tierSwitchTimer.current = setTimeout(() => setTierSwitching(""), 1500);
    },
    [model, quantTier, availableTiers],
  );
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
    showIncompatibleLoras,
    setShowIncompatibleLoras,
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
    setCharacterId,
    setCharacterLookId,
    assets,
    latestAssets,
    trackedLocalJobs,
    initialPresetId: saved.selectedPresetId ?? null,
    advancedOpen,
    setAdvancedOpen,
    initialSelectedLoraIds: saved.selectedLoraIds ?? [],
    initialLoraWeights: saved.loraWeights ?? {},
    initialShowIncompatibleLoras: saved.showIncompatibleLoras ?? false,
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
  }, [launchRequest?.id]);
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
  useEffect(() => {
    if (launchRequest?.view !== "Image" || !launchRequest.recipe) {
      return;
    }
    const recipe = launchRequest.recipe;
    const settings = recipe.normalizedSettings ?? {};
    const rawSettings = recipe.rawAdapterSettings ?? {};
    const nextMode = recipeMode(recipe);
    const resolutionFromRecipe = recipeResolution(recipe);
    const { loraIds, loraWeights: loraWeightMap } = recipeLoraSelection(recipe);

    skipReferenceTuningReset.current = true;
    // A recipe injects its own reference tuning below (setIpAdapterScale/…), so disarm the
    // fresh-mount declared-defaults resolver (sc-12034) — it must not overwrite the recipe values
    // once the catalog resolves, even on a late-catalog mount where `skip*` was already consumed.
    referenceTuningDeclaredArmed.current = false;
    setSelectedPresetId(noPresetId);
    setMode(nextMode);
    if (recipe.model) {
      if (recipe.model !== advancedDefaultsModel.current) {
        skipAdvancedDefaultsReset.current = true;
      }
      setModel(recipe.model);
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
    setStyleId(hasRawStylePrompt ? restoredStyleId : null);
    if (restoredCaption && validateCaption(restoredCaption).ok) {
      setCaption(orderCaption(restoredCaption));
      setPromptMode("form");
      setMagicPromptBackend(structuredRecipe.magicPromptBackend ?? null);
      // The intent (original idea) seeds the plain box; the serialized caption is
      // authoritative for generation and is rebuilt from `caption` on submit.
      setPrompt(String(structuredRecipe.intent ?? ""));
    } else if (hasRawStylePrompt) {
      // Styled recipe (sc-13132): seed the box with the RAW pre-style prompt, NOT the composed
      // `recipe.prompt`. With the picker re-selected above, submit recomposes the identical
      // `Subject:`/`Style:` prompt — recording the raw prompt is what prevents a double-wrap
      // (composing over the already-composed prompt would nest a second `Style:` block).
      setPrompt(rawSettings.stylePrompt);
    } else {
      setPrompt(String(recipe.prompt ?? ""));
    }
    promptEdited.current = true;
    setNegativePrompt(String(recipe.negativePrompt ?? ""));
    // Recipe replay leaves Seed random by default so "Use this recipe" makes a close
    // variation instead of a byte-for-byte rerun. When the launcher passes a replaySeed
    // (viewer "Keep seed" toggle), it is already resolved to THIS image's own seed —
    // replay it verbatim for an exact reproduction. Guard with `!= null` so seed 0 is honored.
    const replaySeed = launchRequest.replaySeed;
    setSeed(replaySeed != null && replaySeed !== "" ? String(replaySeed) : "");
    const countValue = finiteRecipeNumber(settings.count);
    if (countValue) {
      setCount(countValue);
    }
    if (resolutionFromRecipe) {
      setResolution(resolutionFromRecipe);
    }
    setSelectedLoraIds(loraIds);
    setLoraWeights(loraWeightMap);
    setStepsOverride(rawSettings.steps ?? rawSettings.numInferenceSteps ?? "");
    setGuidanceOverride(rawSettings.guidanceScale ?? "");
    setSampler(rawSettings.sampler ?? "default");
    setScheduler(rawSettings.scheduler ?? "default");
    setSchedulerShift(rawSettings.schedulerShift ?? rawSettings.timestepShift ?? 3.0);
    setGuidanceMethod(rawSettings.guidanceMethod ?? "cfg");
    setCharacterId(settings.characterId ?? "");
    setCharacterLookId(settings.characterLookId ?? "");
    setReferenceAssetId(rawSettings.referenceAssetId ?? launchRequest.referenceAssetId ?? "");
    setIpAdapterScale(rawSettings.ipAdapterScale ?? settings.ipAdapterScale ?? ipAdapterScale);
    setControlnetScale(rawSettings.controlnetConditioningScale ?? rawSettings.controlnetScale ?? settings.controlnetScale ?? controlnetScale);
    setTrueCfgScale(rawSettings.trueCfgScale ?? settings.trueCfgScale ?? trueCfgScale);
    setViewAngle(rawSettings.viewAngle ?? settings.viewAngle ?? "");
    setSelectedPoseIds([]);
    if (nextMode === "edit_image") {
      setSourceAssetId(launchRequest.sourceAssetId ?? launchRequest.assetId ?? settings.sourceAssetId ?? "");
      setFitMode(rawSettings.fitMode ?? settings.fitMode ?? "crop");
    }
    const upscale = rawSettings.upscale ?? settings.upscale;
    setUpscaleEnabled(Boolean(upscale?.enabled));
    if (upscale?.factor) {
      setUpscaleFactor(upscale.factor);
    }
    if (upscale?.engine) {
      handleUpscaleEngineChange(upscale.engine);
    }
    if (typeof upscale?.softness === "number") {
      setUpscaleSoftness(upscale.softness);
    }
  }, [launchRequest?.id]);
  const [dropdownWidth, dropdownHeight] = resolution.split("x").map((value) => Number(value));
  // A non-empty Width/Height override wins for that axis; empty falls back to the Aspect
  // dropdown. The resulting dims flow through the existing top-level width/height payload,
  // so submit() and the batch builder need no further change. Logic lives in the pure,
  // unit-tested resolveEffectiveDimensions helper.
  const { width, height, invalid: dimensionsInvalid } = resolveEffectiveDimensions({
    resolution,
    widthOverride,
    heightOverride,
  });

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
  const onMagicExpand = useCallback(
    async (idea) => {
      if (typeof magicPrompt !== "function") {
        throw new Error("Magic-prompt is unavailable.");
      }
      const divisor = gcd(width, height) || 1;
      const aspectRatio = Number.isFinite(width) && Number.isFinite(height) ? `${width / divisor}:${height / divisor}` : "1:1";
      const raw = await magicPrompt({ prompt: idea, modelId: model, aspectRatio });
      const { caption: expanded, error } = parseMagicPromptCaption(raw);
      if (error || !expanded) {
        throw new Error(error || "Magic-prompt returned an unusable caption.");
      }
      setMagicPromptBackend(PROMPT_REFINE_MODEL_ID);
      return expanded;
    },
    [magicPrompt, model, width, height],
  );

  // Reference-image → JSON caption (epic 8102, sc-8108): run the worker's `image_caption` vision job on
  // the picked reference asset and parse the reply into an editable caption. Uses `parseVisionCaption`
  // (strips the non-schema `aspect_ratio`, KEEPS the grounded bboxes — they are derived from the actual
  // image, unlike magic-prompt's guessed boxes). Throws on a malformed/non-caption reply so the builder
  // surfaces the error and lets the user retry, mirroring the magic-prompt error UX. C1: the image is
  // captioning-only — it is consumed here to produce JSON and never passed to generation.
  const onImageCaption = useCallback(
    // Single id (string) for one reference, or an ARRAY of ids for a mood board (sc-8595): >1 synthesizes
    // ONE Ideogram JSON caption from the shared style, exactly one keeps the scalar single-image path.
    async (source) => {
      if (typeof imageCaption !== "function") {
        throw new Error("Image captioning is unavailable.");
      }
      if (!activeProject?.id) {
        throw new Error("Open a project first.");
      }
      const ids = Array.isArray(source) ? source.filter(Boolean) : [source].filter(Boolean);
      const multi = ids.length > 1;
      const raw = await imageCaption({
        sourceAssetId: multi ? undefined : ids[0],
        sourceAssetIds: multi ? ids : undefined,
        projectId: activeProject.id,
        model: VISION_CAPTION_MODEL_REPO,
      });
      const { caption: parsed, error } = parseVisionCaption(raw);
      if (error || !parsed) {
        throw new Error(error || "The image did not produce a usable caption.");
      }
      setMagicPromptBackend(VISION_CAPTION_MODEL_ID);
      return parsed;
    },
    [imageCaption, activeProject?.id],
  );

  // Reference-image → plain-text description (epic 8203, sc-8208): the NON-structured sibling of
  // onImageCaption. Runs the worker's `image_describe` job on the picked reference and resolves to the
  // raw description text — prose by default, or booru tags when the model declares `captionStyle:"tags"`
  // (sc-8205). The shared picker drops the returned text into the prompt textarea. Gated to
  // text-to-image only, like the caption flow. C1: the image is consumed to produce the prompt and is
  // never passed to generation.
  const describeCaptionStyle = selectedModel?.captionStyle;
  const onImageDescribe = useCallback(
    // The shared picker passes a single asset id (string) for one reference, or an ARRAY of ids for a
    // mood board (sc-8595). Normalize: >1 rides `sourceAssetIds` (worker synthesizes one prompt from the
    // shared style), exactly one collapses to the scalar `sourceAssetId` (the unchanged single path).
    async (source) => {
      if (typeof imageDescribe !== "function") {
        throw new Error("Image description is unavailable.");
      }
      if (!activeProject?.id) {
        throw new Error("Open a project first.");
      }
      const ids = Array.isArray(source) ? source.filter(Boolean) : [source].filter(Boolean);
      const multi = ids.length > 1;
      const text = await imageDescribe({
        sourceAssetId: multi ? undefined : ids[0],
        sourceAssetIds: multi ? ids : undefined,
        projectId: activeProject.id,
        model: VISION_CAPTION_MODEL_REPO,
        captionStyle: describeCaptionStyle,
      });
      const trimmed = (text || "").trim();
      if (!trimmed) {
        throw new Error("The image did not produce a usable description.");
      }
      setMagicPromptBackend(VISION_CAPTION_MODEL_ID);
      return trimmed;
    },
    [imageDescribe, activeProject?.id, describeCaptionStyle],
  );

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
    selectedPreset,
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
      ["prompt", setPrompt],
      ["negativePrompt", setNegativePrompt],
      ["resolution", setResolution],
      ["count", setCount],
      ["guidanceScale", setGuidanceOverride],
      ["steps", setStepsOverride],
      ["sampler", setSampler],
      ["scheduler", setScheduler],
      ["schedulerShift", setSchedulerShift],
      ["guidanceMethod", setGuidanceMethod],
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
      prompt,
      negativePrompt,
      resolution,
      count,
      mode,
      guidanceScale: finiteNumberOrUndefined(guidanceOverride),
      steps: finiteNumberOrUndefined(stepsOverride),
      sampler,
      scheduler,
      schedulerShift,
      guidanceMethod,
      upscaleEnabled,
      upscaleFactor,
      upscaleEngine,
      upscaleSoftness,
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
    styleId,
    structuredCaption: caption,
    promptMode,
    magicPromptBackend,
    count,
    advancedOpen,
    model,
    seed,
    negativePrompt,
    resolution,
    widthOverride,
    heightOverride,
    fitMode,
    referenceAssetIds,
    ipAdapterScale,
    controlnetScale,
    trueCfgScale,
    viewAngle,
    upscaleEnabled,
    upscaleFactor,
    upscaleEngine,
    upscaleSoftness,
    selectedLoraIds,
    loraWeights,
    showIncompatibleLoras,
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
    guidanceMethod,
    steps: stepsOverride,
    guidanceScale: guidanceOverride,
    flashAttn,
    enhancePrompt,
    bf16Precision,
    usePid,
    pidTarget,
  },
  // Suppress the live writer until the model catalog has loaded (sc-11962), so a
  // transient defaults-reset during the restart-restore/settle window can't be
  // persisted over the restored snapshot. When there are no models the studio shows
  // the availability gate (no editable form), so nothing meaningful is lost by waiting.
  imageModels.length > 0);

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

  async function submit(event) {
    event.preventDefault();
    // Batch mode runs through its own "Run batch" action (sc-9956), never the single
    // Generate submit — guard so a stray Enter in a batch field can't queue one image.
    if (batchMode) {
      return;
    }
    if (submitting) {
      return;
    }
    if (dimensionsInvalid) {
      setSubmitError("Width and height must each be between 256 and 4096.");
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
  const buildJobRequest = (overrides = {}) => {
    // Fold the general-preset stack (epic 11949) into this request. The prompt is composed per
    // call so it wraps either the single prompt or a per-batch-item prompt; a structured JSON
    // caption can't take flat fragments, so the prompt fold is skipped there (the stack's
    // negative/aspect/count still apply). When the stack folds the prompt, the client is
    // authoritative — presetPromptResolvedClientSide tells the server to skip its own fold.
    const stackActive = generalStack.length > 0;
    const isStructured = overrides.sendStructured ?? false;
    const foldPrompt = stackActive && !isStructured;
    const promptToSend =
      foldPrompt && overrides.promptToSend != null
        ? composePreset({ base: selectedPreset, generalStack, userText: overrides.promptToSend, resolutionOptions })
            .prompt
        : overrides.promptToSend;
    const stackResolution = stackActive && composedStack.resolution ? parseResolution(composedStack.resolution) : null;
    return buildImageJobRequest({
      // Overrides — the one legitimate single-vs-batch difference.
      promptToSend,
      submitIntent: overrides.submitIntent,
      sendStructured: isStructured,
      submitCaption: overrides.submitCaption,
      submitBackend: overrides.submitBackend,
      // A per-prompt [WxH] batch directive wins; otherwise the stack's aspect drives resolution.
      resolutionOverride: overrides.resolutionOverride ?? stackResolution,
      // Shared studio settings (identical for both paths).
      resolution,
      mode,
      negativePrompt: stackActive ? composedStack.negativePrompt : negativePrompt,
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
      styleText: styleTextForId(effectiveStyleId),
      // sc-13132: the opaque style id travels with the recipe so replay can re-select the picker.
      styleId: effectiveStyleId,
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
      loras: buildLorasPayload(),
      upscaleEnabled,
      upscaleFactor,
      upscaleEngine,
      upscaleSoftness,
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
      // Never send a tier that isn't installed-and-complete. The seed effect keeps `quantTier` clamped to
      // an installed tier, but gate the OUTGOING value on the selectable set as a hard belt so no state —
      // a race, a stale sticky, a torn tier newly reported missing — can leak an uninstalled tier into the
      // payload (the worker would then default/crash against a tier the user never downloaded). Mirrors the
      // guard VideoStudio/Editor already apply. Empty string ⇒ `tierQuantize("")` is null ⇒ no mlxQuantize.
      quantTier: availableTiers.includes(quantTier) ? quantTier : "",
      // sc-10733: the tier is a DELIBERATE pick (not the pure global/base default) when it equals this
      // (screen, model)'s persisted sticky — a prior explicit pick, which `handleTierChange` writes and
      // the seed effect reads back into `quantTier`. The worker honors an explicit pick (never silently
      // downtiers it); only a non-explicit default is capability-clamped. Gate on the same installed-set
      // membership so an uninstalled sticky is never flagged explicit.
      tierExplicit:
        availableTiers.includes(quantTier) && readLastTier(TIER_SCREEN, model) === quantTier,
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

  // Fan out one image job per resolved prompt (mirrors the asset batch, sc-6112): each
  // posts independently so the worker runs them serially with its between-image cache
  // release, and progress/cancel read the live jobs feed.
  async function runBatch(confirmed = false) {
    if (batchRun?.submitting || !activeProject) {
      return;
    }
    const resolved = expandBatch(batchPrompts, batchVariables);
    if (!resolved.length) {
      return;
    }
    const promptBudgetOverages = batchPromptBudgetOverages(
      stylePreviewActive && !structuredPromptModel
        ? resolved.map(({ prompt: resolvedPrompt }) => {
            const { prompt: cleanPrompt } = parsePromptResolution(resolvedPrompt);
            return buildBatchJobRequest(cleanPrompt).prompt;
          })
        : [],
    );
    if (promptBudgetOverages.length) {
      setBatchError(batchPromptBudgetMessage(promptBudgetOverages));
      setBatchConfirmPending(false);
      return;
    }
    if (dimensionsInvalid) {
      setBatchError("Width and height must each be between 256 and 4096.");
      return;
    }
    setBatchError("");
    // Soft cap: a large run must be confirmed once, showing the exact image count.
    if (!confirmed && resolved.length * batchImagesPerPrompt > BATCH_RENDER_CAP) {
      setBatchConfirmPending(true);
      return;
    }
    setBatchConfirmPending(false);
    batchAbortRef.current = false;
    // Items carry `error` so not-yet-submitted rows read as pending, not failed, while a
    // (possibly slow, structured) enqueue is in flight. Updated after each post so progress
    // ticks up live.
    const items = resolved.map((entry) => ({ prompt: entry.prompt, jobId: null, error: false }));
    setBatchRun({ submitting: true, items: items.map((item) => ({ ...item })) });
    for (let i = 0; i < resolved.length; i += 1) {
      if (batchAbortRef.current) {
        break;
      }
      const entry = resolved[i];
      // Strip a leading [WxH] directive (sc-10063): the model gets the clean prompt, the job
      // gets that per-prompt resolution.
      const { prompt: cleanPrompt, resolution } = parsePromptResolution(entry.prompt);
      try {
        let request;
        if (structuredPromptModel) {
          // Structured-caption models (Ideogram 4) reject raw plain text, so auto-expand each
          // resolved prompt into a JSON caption first (sc-9980) — N sequential refine calls.
          // A prompt that fails to expand fails only that item; the rest continue.
          const expanded = await onMagicExpand(cleanPrompt);
          if (!validateCaption(expanded).ok) {
            throw new Error("Auto-generated caption was invalid.");
          }
          request = buildBatchJobRequest(cleanPrompt, {
            promptToSend: serializeCaption(expanded),
            sendStructured: true,
            submitCaption: expanded,
            submitBackend: PROMPT_REFINE_MODEL_ID,
            resolution,
          });
        } else {
          request = buildBatchJobRequest(cleanPrompt, { resolution });
        }
        const job = await createImageJob(request);
        items[i] = { prompt: cleanPrompt, jobId: job?.id ?? null, error: !job?.id };
      } catch {
        items[i] = { prompt: cleanPrompt, jobId: null, error: true };
      }
      setBatchRun({ submitting: true, items: items.map((item) => ({ ...item })) });
    }
    setBatchRun({ submitting: false, items });
  }

  // Stop the enqueue loop (if still running) and cancel every still-pending job in the run;
  // completed/failed items are left as-is.
  function cancelBatchRun() {
    batchAbortRef.current = true;
    if (!batchRun) {
      return;
    }
    for (const item of batchRun.items) {
      if (!item.jobId) {
        continue;
      }
      const status = batchItemStatus(item.jobId, jobs);
      if (status !== "queued" && status !== "running") {
        continue;
      }
      const job = jobs.find((entry) => entry.id === item.jobId);
      if (job) {
        jobAction(job, "cancel");
      }
    }
  }

  const batchRunProgress = batchRun ? summarizeBatchRun(batchRun.items, jobs) : null;
  const batchMissingKeys = missingKeys(batchPrompts, batchVariables);
  const batchGroupIssues = linkedGroupIssues(batchPrompts);
  // Prompt lines whose leading [WxH] directive (sc-10063) is out of the backend 256–4096
  // range — block the run and name the offending size.
  const batchResolutionIssues = batchPrompts
    .map((line) => parsePromptResolution(line).resolution)
    .filter(
      (res) =>
        res &&
        (res.width < MIN_IMAGE_DIMENSION ||
          res.width > MAX_IMAGE_DIMENSION ||
          res.height < MIN_IMAGE_DIMENSION ||
          res.height > MAX_IMAGE_DIMENSION),
    );
  // A structured-caption model can batch, but only if the prompt-refiner is available to
  // auto-write a caption per resolved prompt (sc-9980).
  const batchStructuredExpandBlocked =
    structuredPromptModel && (magicModelMissing || typeof magicPrompt !== "function");
  const activeStyleText = styleTextForId(effectiveStyleId);
  const styleSelected = typeof activeStyleText === "string" && activeStyleText.trim() !== "";
  const stylePreviewActive = !structuredPromptModel && styleSelected;
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
      minDimension: MIN_IMAGE_DIMENSION,
      maxDimension: MAX_IMAGE_DIMENSION,
    }),
    [
      activeProject,
      batchStructuredExpandBlocked,
      batchTotal,
      batchMissingKeys,
      batchGroupIssues,
      batchResolutionIssues,
    ],
  );
  // sc-13131 / sc-13133: the live composed-prompt preview for the selected Style Catalog entry, and
  // the budget the composed string spends against the backend cap. ANTI-DRIFT: we do NOT re-derive
  // the composition here — we run the SAME buildJobRequest the single Generate submit calls (with the
  // live prompt as promptToSend) and read its `.prompt`, so the previewed/measured string is
  // byte-for-byte the prompt that will be sent (preset stack folds into the prompt FIRST, the style's
  // Subject:/Style: wrap is applied LAST — see imageJobRequest.js). It recomputes every render, so
  // it tracks the prompt text, the selected style, and the active preset stack live. Only active for
  // free-text models with a style actually selected: structured-caption models (Ideogram) merge the
  // style into the caption's `aesthetics` instead (sc-13224), so there's no Subject:/Style: prose
  // to preview, and a null/empty styleText is a pass-through with nothing extra to preview and no
  // style-composition budget to guard.
  const styledPreviewPrompt = stylePreviewActive ? buildJobRequest({ promptToSend: prompt }).prompt : null;
  const generateDraft = useMemo(
    () => ({
      activeProject,
      structuredActive,
      captionHasContent,
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
    }),
    [
      activeProject,
      structuredActive,
      captionHasContent,
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
    ],
  );
  const batchValidity = useValidation(imageBatchValidation, batchDraft, undefined);
  const generateValidity = useValidation(imageGenerateValidation, generateDraft, undefined);
  // `submitting` is a busy gate, not a rule. `batchTotal === 0` is a requirement (silent),
  // already folded into batchValidity — no need to repeat it here.
  const batchRunDisabled = !batchValidity.ready || Boolean(batchRun?.submitting);
  // The two conditions whose message has its own home stay explicit gates: a structured
  // caption's field errors live in the builder, and a Mac block prints its own note.
  const generateDisabled =
    submitting ||
    !generateValidity.ready ||
    (structuredActive && !captionValidation?.ok) ||
    Boolean(macActiveModeBlock);

  return (
    <ModelAvailabilityGate
      ready={modelReady}
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
          <div className="prompt-hero-top">
            <div className="mode-tabs" role="tablist" aria-label="Image mode">
              {[
                ["text_to_image", "Text"],
                ["edit_image", "Edit"],
                ["character_image", "With character"],
              ].map(([value, label]) => {
                const macBlock = macModeTabBlock(value);
                const active = mode === value;
                return (
                  <button
                    className={active ? "mode-tab active" : "mode-tab"}
                    key={value}
                    role="tab"
                    aria-selected={active}
                    onClick={() => handleModeChange(value)}
                    type="button"
                    disabled={Boolean(macBlock)}
                    title={macBlock ? macBlock.text : undefined}
                  >
                    {value === "text_to_image" ? <Icon.Sparkle size={13} /> : null}
                    {label}
                  </button>
                );
              })}
            </div>
            <button
              aria-pressed={batchMode}
              className={batchMode ? "batch-toggle active" : "batch-toggle"}
              onClick={() => setBatchMode((on) => !on)}
              title="Run a list of prompts as one batch with the current settings"
              type="button"
            >
              <Icon.Stars size={13} /> Batch
            </button>
            <div className="prompt-hero-links">
              <button className="hero-link" onClick={() => setGuideOpen(true)} type="button">
                <Icon.Book size={14} /> Prompt guide
              </button>
              {onOpenPresets ? (
                <button className="hero-link" onClick={onOpenPresets} type="button">
                  <Icon.Folder size={14} /> Saved presets
                </button>
              ) : null}
            </div>
          </div>

          {batchMode ? (
            <div className="prompt-input-row batch">
              <BatchPromptPanel
                promptsText={batchPromptsText}
                onPromptsTextChange={setBatchPromptsText}
                variableValues={batchVariableValues}
                onVariableValuesChange={setBatchVariableValues}
                count={batchImagesPerPrompt}
                batches={promptBatches}
                projectId={activeProject?.id ?? null}
                name={batchName}
                onNameChange={setBatchName}
                scope={batchScope}
                onScopeChange={setBatchScope}
                loadedBatchId={loadedBatchId}
                onSave={handleSaveBatch}
                onNew={handleNewBatch}
                onLoad={handleLoadBatch}
                onDelete={handleDeleteBatch}
                onImport={handleImportBatch}
                busy={batchBusy}
                error={batchError}
              />
              <div className="batch-run">
                {/* The batch's blocking problems, still one at a time in priority order —
                    but now the winning message and the disabled button read from the same
                    summary, so they cannot disagree (sc-10649). The empty-batch hint is a
                    silent requirement, rendered here as its own empty-state affordance. */}
                {batchValidity.surfaced.length ? (
                  <p className="batch-warning">{batchValidity.surfaced[0].message}</p>
                ) : batchTotal === 0 ? (
                  <p className="batch-hint">Add at least one prompt to run a batch.</p>
                ) : null}
                {batchRun ? (
                  <div className="batch-run-progress" aria-live="polite">
                    <span>
                      {batchRun.submitting
                        ? `Queued ${batchRunProgress.total - batchRunProgress.pending}/${batchRunProgress.total}`
                        : `${batchRunProgress.done}/${batchRunProgress.total} done`}
                      {batchRunProgress.failed ? ` · ${batchRunProgress.failed} failed` : ""}
                    </span>
                    {batchRun.submitting ? (
                      <button className="batch-btn ghost" onClick={cancelBatchRun} type="button">
                        Stop
                      </button>
                    ) : batchRunProgress.active > 0 ? (
                      <button className="batch-btn ghost" onClick={cancelBatchRun} type="button">
                        Cancel remaining
                      </button>
                    ) : (
                      <button className="batch-btn ghost" onClick={() => setBatchRun(null)} type="button">
                        Clear
                      </button>
                    )}
                  </div>
                ) : null}
                {batchConfirmPending ? (
                  <div className="batch-confirm" role="alertdialog">
                    <span>Queue {batchTotal} images? That’s a large batch.</span>
                    <button className="prompt-cta" onClick={() => runBatch(true)} type="button">
                      <Icon.Play size={14} /> Queue {batchTotal}
                    </button>
                    <button className="batch-btn ghost" onClick={() => setBatchConfirmPending(false)} type="button">
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    className="prompt-cta"
                    disabled={batchRunDisabled}
                    onClick={() => runBatch(false)}
                    type="button"
                  >
                    <Icon.Play size={14} />
                    {batchRun?.submitting ? "Queueing…" : `Run batch · ${batchTotal}`}
                  </button>
                )}
              </div>
            </div>
          ) : (
          <div className={`prompt-input-row${structuredPromptModel ? " structured" : ""}`}>
            {structuredPromptModel ? (
              <StructuredPromptBuilder
                caption={caption}
                onCaptionChange={setCaption}
                validation={captionValidation}
                mode={promptMode}
                onModeChange={setPromptMode}
                plainText={prompt}
                onPlainTextChange={setPromptFromUser}
                onMagicExpand={magicPrompt ? onMagicExpand : undefined}
                magicModelMissing={magicModelMissing}
                onDownloadMagicModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
                // sc-8109 seam: the reference-image picker calls this with the uploaded image's
                // natural dimensions to auto-preset the resolution to the nearest aspect.
                onReferenceImageLoaded={onReferenceImageLoaded}
                // Reference-image → JSON caption (epic 8102, sc-8108). Gated to text-to-image ONLY:
                // edit/character modes condition on their own source/identity image, so a fresh
                // scene caption written from a different reference would conflict. The image is
                // captioning-only (C1) — never sent to generation.
                onImageCaption={mode === "text_to_image" && imageCaption ? onImageCaption : undefined}
                referenceAssets={editImageAssets}
                referenceCharacters={characters}
                importAsset={importAsset}
                projectId={activeProject?.id ?? ""}
                // Reference-image caption gate (sc-8110): the section's availability is now driven by the
                // catalog (visionCaptionReady) through the shared ModelAvailabilityGate, not an inline
                // error-after-click affordance. When the captioner is missing, the gate offers a download.
                visionCaptionReady={visionCaptionReady}
                visionCaptionOffers={visionCaptionOffers}
                visionCaptionDownloadJobs={modelDownloadJobs}
                onDownloadModel={createModelDownloadJob}
                onOpenModels={() => setActiveView("Models")}
                onOpenQueue={onOpenQueue}
                onCancelJob={onCancelJob}
              />
            ) : (
              <textarea
                aria-label="Prompt"
                className="prompt-input"
                onChange={(event) => setPromptFromUser(event.target.value)}
                onKeyDown={onPromptKeyDown}
                placeholder="Describe your shot — subject, lighting, mood, lens…"
                value={prompt}
              />
            )}
            <button className="prompt-cta" disabled={generateDisabled} type="submit">
              <Icon.Sparkle size={14} />
              {submitting ? (expanding ? "Expanding…" : "Queueing…") : "Generate"}
            </button>
          </div>
          )}
          {/* Auto-expand failure (sc-6501): a structured model couldn't turn the plain-text idea
              into a caption (e.g. the prompt-refiner model isn't installed). We never fall back to
              sending raw plain text, so surface the reason and the path forward. */}
          {submitError ? (
            <p className="structured-error" role="alert">
              {submitError}
            </p>
          ) : null}

          {/* Booru-convention prompt hint (sc-10760): danbooru-tag models (Anima, Illustrious) declare
              `ui.promptHint`, so the studio nudges toward the quality prefix + tag-style prompting the model
              was trained on (a bare sentence renders low-effort art). Opens the existing prompt-guide modal.
              Free-text models only. */}
          {!structuredPromptModel && promptHint ? (
            <p className="prompt-hint">
              {promptHint}{" "}
              <button className="prompt-hint-link" onClick={() => setGuideOpen(true)} type="button">
                Prompt guide →
              </button>
            </p>
          ) : null}

          {/* Scene suggestions sit directly under the prompt (UI-refinement 4a). Free-text
              prompts only; structured models get the builder + (later) magic-prompt. */}
          {structuredPromptModel ? null : (
            <div className="suggestion-row">
              <span className="suggestion-row-label">Try:</span>
              {suggestions.map((suggestion) => (
                <button
                  className="suggestion"
                  key={suggestion}
                  onClick={() => setPromptFromUser(suggestion)}
                  type="button"
                >
                  <Icon.Sparkle size={11} />
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Style Catalog picker + its composed-prompt preview moved into the Style axis row of the
              settings bar (sc-13135) — see the .settings-bar-style-axis block below. */}

          {/* Prompt tools (UI-refinement 1b; restructured sc-10195): a framed strip of up to THREE
              distinct tiles, one panel open at a time (all free-text only — structured models excluded).
              1) "Image reference" — img2img reference-guided generation (a picked image + strength slider
                 VISUALLY guides the render). Shown only for img2img-capable models (`supportsImg2img`).
              2) "Prompt from image" — the reference→text describe flow + mood board (epic 8203/8595): the
                 vision captioner writes prompt TEXT (captioning-only, never sent to generation). Gated on
                 the macOS-first captioner being platform-eligible.
              3) "Refine my prompt" — RefinePromptControl (sc-2041).
              img2img and describe used to share ONE overloaded tile (sc-8593); splitting them makes the
              "guide the render vs. write a prompt" choice explicit (Michael, on-device). */}
          {(() => {
            // img2img "Image reference" tile — a picked image + strength that SEED the render
            // (latent-init). Available for every `ui.img2img` model in text-to-image mode, INCLUDING
            // structured-prompt models (Ideogram, epic 8588 A4.4 sc-10192): the caption builder replaces
            // the free-text prompt tools, but reference-guided generation is orthogonal to how the prompt
            // is authored, so the tile coexists with the JSON-caption builder. Needs no vision captioner.
            const img2imgAvailable = supportsImg2img && mode === "text_to_image";
            const imageRefActive = img2imgAvailable && promptTool === "imageReference";
            const img2imgTile = img2imgAvailable ? (
              <button
                type="button"
                className={imageRefActive ? "prompt-tool active" : "prompt-tool"}
                aria-pressed={imageRefActive}
                onClick={() => togglePromptTool("imageReference")}
              >
                <span className="prompt-tool-title">
                  <Icon.Image size={15} /> Image reference
                  {/* Armed indicator: the panel is a collapsible accordion, so once a reference is
                      picked and the tile is collapsed there's otherwise no sign it's still driving
                      every generation. Show "On" whenever a reference is set, regardless of expansion. */}
                  {img2imgReferenceAssetId ? <span className="prompt-tool-flag">On</span> : null}
                </span>
                <span className="prompt-tool-desc">Guide the render with an image (image-to-image)</span>
              </button>
            ) : null;
            const img2imgPanel = imageRefActive ? (
              <div className="prompt-tool-panel">
                <div className="structured-reference">
                  <p className="structured-hint">
                    Pick an image to guide the render (image-to-image). A higher reference strength
                    stays closer to it; lower lets the prompt take over.
                  </p>
                  <ImageEditSourcePickerField
                    assets={editImageAssets}
                    buttonLabel="Select reference image"
                    changeLabel="Change reference"
                    characters={characters}
                    clearable
                    emptyLabel="No reference image selected"
                    importAsset={importAsset}
                    label="Reference image"
                    onChange={setImg2imgReferenceAssetId}
                    projectId={activeProject?.id}
                    value={img2imgReferenceAssetId}
                  />
                  {img2imgReferenceAssetId ? (
                    <label className="reference-strength img2img-strength">
                      {img2imgStrengthConfig?.label ?? "Reference strength"}
                      <input
                        max={img2imgStrengthConfig?.max ?? 1}
                        min={img2imgStrengthConfig?.min ?? 0}
                        onChange={(event) => setImg2imgStrength(Number(event.target.value))}
                        step={img2imgStrengthConfig?.step ?? 0.05}
                        type="range"
                        value={img2imgStrength}
                      />
                      <span>{Number(img2imgStrength).toFixed(2)}</span>
                    </label>
                  ) : null}
                </div>
              </div>
            ) : null;

            // Structured-prompt models (Ideogram) get ONLY the img2img tile in this strip: "Prompt from
            // image" is served by the caption builder's own image→caption picker (epic 8102) and "Refine
            // my prompt" by its magic-expand, so rendering them here would duplicate those. Nothing to
            // show when the model doesn't advertise img2img.
            if (structuredPromptModel) {
              return img2imgAvailable ? (
                <div className="prompt-tools">
                  <div className="prompt-tools-head">
                    <span className="prompt-tools-title">Prompt tools</span>
                    <span className="hairline" />
                  </div>
                  <div className="prompt-tools-tiles">{img2imgTile}</div>
                  {img2imgPanel}
                </div>
              ) : null;
            }

            const describeAvailable =
              mode === "text_to_image" &&
              typeof imageDescribe === "function" &&
              (visionCaptionReady || visionCaptionOffers.length > 0);
            const describeActive = describeAvailable && promptTool === "describe";
            const refineActive = promptTool === "refine";
            return (
              <div className="prompt-tools">
                <div className="prompt-tools-head">
                  <span className="prompt-tools-title">Prompt tools</span>
                  <span className="hairline" />
                </div>
                <div className="prompt-tools-tiles">
                  {img2imgTile}
                  {describeAvailable ? (
                    <button
                      type="button"
                      className={describeActive ? "prompt-tool active" : "prompt-tool"}
                      aria-pressed={describeActive}
                      onClick={() => togglePromptTool("describe")}
                    >
                      <span className="prompt-tool-title">
                        <Icon.Image size={15} /> Prompt from image
                      </span>
                      <span className="prompt-tool-desc">Caption a reference (or mood board) into an editable prompt</span>
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className={refineActive ? "prompt-tool active" : "prompt-tool"}
                    aria-pressed={refineActive}
                    onClick={() => togglePromptTool("refine")}
                  >
                    <span className="prompt-tool-title">
                      <Icon.Wand size={15} /> Refine my prompt
                    </span>
                    <span className="prompt-tool-desc">Rewrite what you typed for clarity &amp; detail</span>
                  </button>
                </div>
                {img2imgPanel}
                {describeActive ? (
                  <div className="prompt-tool-panel">
                    <ReferenceCaptionPicker
                      onCaption={onImageDescribe}
                      onApply={(text) => setPromptFromUser(text)}
                      onReferenceImageLoaded={onReferenceImageLoaded}
                      referenceAssets={editImageAssets}
                      referenceCharacters={characters}
                      importAsset={importAsset}
                      projectId={activeProject?.id ?? ""}
                      hint="The image is only used to write the prompt — it isn’t sent to generation."
                      buttonLabel="✨ Describe image"
                      busyLabel="Describing…"
                      showMoodBoard={visionCaptionReady}
                      emptyMessage="The image did not produce a usable description. Try another reference."
                      errorFallback="Could not describe the image."
                      gateDescription="Download the vision captioner to turn a reference image into a prompt. It runs locally on the native worker; the image is only used to write the prompt."
                      visionCaptionReady={visionCaptionReady}
                      visionCaptionOffers={visionCaptionOffers}
                      visionCaptionDownloadJobs={modelDownloadJobs}
                      onDownloadModel={createModelDownloadJob}
                      onOpenModels={() => setActiveView("Models")}
                      onOpenQueue={onOpenQueue}
                      onCancelJob={onCancelJob}
                    />
                  </div>
                ) : null}
                {refineActive ? (
                  <div className="prompt-tool-panel">
                    <RefinePromptControl
                      autoStart
                      guidePath={promptGuide.path}
                      modelId={model}
                      onApply={setPromptFromUser}
                      prompt={prompt}
                      refinePrompt={refinePrompt}
                      refineModel={refineModel}
                      onDownloadRefineModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
                      workflow="image"
                    />
                  </div>
                ) : null}
              </div>
            );
          })()}

        {mode === "edit_image" || mode === "character_image" ? (
          <div className="studio-source-band">
            {mode === "edit_image" ? (
              <>
                {multiReference ? (
                  // sc-6211: FLUX.2-dev multi-reference edit — pick 1–N reference images that the
                  // model combines/edits (Conditioning::MultiReference). Sends the plural
                  // `referenceAssetIds`; a single pick reduces to the normal single-reference edit.
                  <AssetPickerField
                    assets={editImageAssets}
                    buttonLabel="Select images"
                    changeLabel="Edit references"
                    emptyLabel="No reference images selected"
                    label="Reference images"
                    multiple
                    onChange={setReferenceAssetIds}
                    values={referenceAssetIds}
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
                      onChange={setSourceAssetId}
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
                          onChange={setEditSecondAssetId}
                          projectId={activeProject?.id}
                          value={editSecondAssetId}
                        />
                        {editReferences.secondaryHint ? (
                          <p className="field-hint">{editReferences.secondaryHint}</p>
                        ) : null}
                      </>
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
                  <label>
                    Character
                    <select onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
                      <option value="">Select character</option>
                      {characters.map((character) => (
                        <option key={character.id} value={character.id}>
                          {character.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Look
                    <select onChange={(event) => setCharacterLookId(event.target.value)} value={characterLookId}>
                      <option value="">Default look</option>
                      {(characters.find((character) => character.id === characterId)?.looks ?? []).map((look) => (
                        <option key={look.id} value={look.id}>
                          {look.name}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
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
                            onClick={() => setReferenceAssetId(reference.assetId)}
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
                        <details className="pose-library-details">
                          <summary>
                            Pose library{selectedPoseIds.length ? ` · ${selectedPoseIds.length} selected` : ""}
                          </summary>
                          <PoseLibraryPicker
                            loadUserPoses={loadUserPoses}
                            onClear={() => setSelectedPoseIds([])}
                            onToggle={(id) =>
                              setSelectedPoseIds((ids) =>
                                ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id],
                              )
                            }
                            selectedIds={selectedPoseIds}
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

        {/* Strict-control panel (epic 8236, sc-8245): pose / canny / depth structure lock for the
            text-to-image backbones whose `ui.controlModes` advertises it. Hidden when the backbone
            supports no strict control. Pose reuses the library picker (one image per pose); canny/depth
            take an uploaded control image + a preprocess-vs-use-as-is toggle. The request wiring lives in
            submit() — controlMode / sourceAssetId|advanced.controlImage / advanced.controlScale. */}
        {showControlPanel ? (
          <div className="studio-source-band">
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
              poseBlockText={macPoseBlock ? macPoseBlock.text : null}
              controlImageAssetId={controlImageAssetId}
              onControlImageChange={setControlImageAssetId}
              controlImagePassthrough={controlImagePassthrough}
              onControlImagePassthroughChange={setControlImagePassthrough}
              controlImageAssets={editImageAssets}
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
          </div>
        ) : null}

          {/* Generation settings (UI-refinement 2b): the everyday knobs — Model, Aspect,
              Variations, Style preset — sit in a bar directly under the composer instead of a
              detached right rail. Power-user knobs fold into Advanced below; the results area
              reclaims the full width (single-column .studio-results). */}
          <div className="settings-bar">
            <div className="settings-bar-row">
              <label className="settings-field settings-field-model">
                Model
                <StudioUpdateBadge item={selectedModel} />
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {pickerModels.map((item) => (
                    <option key={item.id} value={item.id}>
                      {updateOptionLabel(item)}
                    </option>
                  ))}
                </select>
                <StudioUpdateNotice item={selectedModel} onUpdate={createModelDownloadJob} />
              </label>
              <label className="settings-field settings-field-aspect">
                Aspect
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((option) => (
                    <option key={option} value={option}>{formatResolutionLabel(option)}</option>
                  ))}
                </select>
              </label>
              <label className="settings-field settings-field-count">
                Variations
                <input min="1" max="8" onChange={(event) => setCount(Number(event.target.value))} type="number" value={count} />
              </label>
            </div>
            {/* Style axis (sc-13135): the Style Catalog picker sits FIRST in this row, followed by the
                model's Style presets — both are style controls, so they share one row instead of the
                catalog picker floating in a standalone row under the composer. The Style Catalog
                composes the prompt (free-text) or merges into the caption (Ideogram, sc-13224); "None"
                resets to pass-through. Hidden for booru-tag models, whose convention the catalog's prose
                entries do not fit. NB: distinct from Krea's numeric "text style" (textStyleGain). */}
            <div className="settings-bar-styles settings-bar-style-axis">
              {styleAxisAvailable ? (
                <div className="style-axis-field style-axis-catalog">
                  <span className="settings-bar-label">Style</span>
                  <StylePicker groups={STYLE_GROUPS} selectedId={styleId} onSelect={setStyleId} label="Style" />
                </div>
              ) : null}
              <div className="style-axis-field style-axis-presets">
                <span className="settings-bar-label">Style preset</span>
                <div className="preset-chips">
                  <button
                    className={!selectedPreset ? "preset-chip active" : "preset-chip"}
                    onClick={() => setSelectedPresetId(noPresetId)}
                    type="button"
                  >
                    None
                  </button>
                  {availablePresets.map((preset) => (
                    <button
                      className={selectedPreset?.id === preset.id ? "preset-chip active" : "preset-chip"}
                      key={preset.id}
                      onClick={() => setSelectedPresetId(preset.id)}
                      type="button"
                    >
                      {preset.name ?? preset.id}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            {availableGeneralPresets.length ? (
              <div className="settings-bar-styles">
                <span className="settings-bar-label">General</span>
                <div className="preset-chips general-preset-chips">
                  {availableGeneralPresets.map((preset) => (
                    <button
                      className={generalStackIds.includes(preset.id) ? "preset-chip active" : "preset-chip"}
                      key={preset.id}
                      onClick={() => toggleGeneralPreset(preset.id)}
                      type="button"
                    >
                      {preset.name ?? preset.id}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          {/* sc-13131: the EXACT composed prompt (Subject:/Style:, preserved sibling directives,
              and the own-`Style:` MERGE) the run will send once a style is active — reuses
              buildJobRequest so it can never drift from the payload. Sits under the Style axis row.
              Hidden when no style applies. */}
          <StyledPromptPreview active={stylePreviewActive} composedPrompt={styledPreviewPrompt} />

          {macActiveModeBlock ? <p className="mac-gating-note">{macActiveModeBlock.text}</p> : null}

          <PresetGuidanceStrip
            selectedPreset={selectedPreset}
            presetPromptParts={presetPromptParts}
            presetLoraDetails={presetLoraDetails}
          />

          <PresetStackPreview
            generalStack={generalStack}
            composed={composedStack}
            stackAddsNegative={stackAddsNegative}
            stackAddsCount={stackAddsCount}
          />

          <AdvancedSection
            hint="cleared values → model default"
            onToggle={() => setAdvancedOpen((value) => !value)}
            open={advancedOpen}
          >
            <div className="advanced-panel">
              <label>
                GPU
                <select onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
                  {gpuOptions.map((gpu) => (
                    <option key={gpu} value={gpu}>
                      {gpu === "auto" ? "Auto" : gpu}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
              </label>
              <label>
                Width override
                <input
                  min="256"
                  max="4096"
                  onChange={(event) => setWidthOverride(event.target.value)}
                  placeholder={String(dropdownWidth)}
                  type="number"
                  value={widthOverride}
                />
              </label>
              <label>
                Height override
                <input
                  min="256"
                  max="4096"
                  onChange={(event) => setHeightOverride(event.target.value)}
                  placeholder={String(dropdownHeight)}
                  type="number"
                  value={heightOverride}
                />
              </label>
              {showSamplerPicker ? (
                <label>
                  Sampler
                  <select onChange={(event) => setSampler(event.target.value)} value={sampler}>
                    {samplerOptions.map((key) => (
                      <option key={key} value={key}>
                        {SAMPLER_LABELS[key] ?? key}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {showSchedulerPicker ? (
                <label>
                  Scheduler
                  <select onChange={(event) => setScheduler(event.target.value)} value={scheduler}>
                    {schedulerOptions.map((key) => (
                      <option key={key} value={key}>
                        {SCHEDULER_LABELS[key] ?? key}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {showSchedulerPicker && scheduler !== "default" ? (
                <label>
                  Schedule shift
                  <input
                    max="10"
                    min="0.1"
                    onChange={(event) => setSchedulerShift(Number(event.target.value))}
                    step="0.1"
                    type="number"
                    value={schedulerShift}
                  />
                </label>
              ) : null}
              <label>
                Steps
                <input
                  min="1"
                  max="80"
                  onChange={(event) => setStepsOverride(event.target.value)}
                  placeholder={String(stepsDefaultFromModel(selectedModel) ?? "")}
                  type="number"
                  value={stepsOverride}
                />
              </label>
              <label>
                Guidance
                <input
                  min="0"
                  max="30"
                  onChange={(event) => setGuidanceOverride(event.target.value)}
                  placeholder={(() => {
                    const value = guidanceDefaultFromModel(selectedModel);
                    return value == null ? "" : String(value);
                  })()}
                  step="0.1"
                  type="number"
                  value={guidanceOverride}
                />
              </label>
              {supportsTextStyle ? (
                <label className="text-style-gain">
                  {textStyleGainConfig?.label ?? "Text style"}
                  <input
                    max={textStyleGainConfig?.max ?? 1.75}
                    min={textStyleGainConfig?.min ?? 0.25}
                    onChange={(event) => setTextStyleGain(Number(event.target.value))}
                    step={textStyleGainConfig?.step ?? 0.05}
                    type="range"
                    value={textStyleGain}
                  />
                  <span>{Number(textStyleGain).toFixed(2)}</span>
                </label>
              ) : null}
              {showGuidanceMethodPicker ? (
                <label>
                  Guidance method
                  <select
                    onChange={(event) => setGuidanceMethod(event.target.value)}
                    value={guidanceMethod}
                  >
                    {guidanceMethodOptions.map((key) => (
                      <option key={key} value={key}>
                        {GUIDANCE_METHOD_LABELS[key] ?? key}
                      </option>
                    ))}
                  </select>
                  {guidanceMethod === "cfg_pp" ? (
                    <span className="field-hint">
                      CFG++ reparameterizes guidance — use a low CFG (~1.5–2.5); high
                      values over-saturate.
                    </span>
                  ) : null}
                </label>
              ) : null}
              <label
                className="checkline flash-attn-toggle"
                title="Fused flash-attention on the candle (Windows/CUDA) SDXL backend — faster and lower VRAM. Ignored on other backends."
              >
                <input
                  checked={flashAttn}
                  onChange={(event) => setFlashAttn(event.target.checked)}
                  type="checkbox"
                />
                Flash attention
              </label>
              {promptEnhance ? (
                <label
                  className="checkline prompt-enhance-toggle"
                  title="Have FLUX.2-dev's built-in LLM rewrite (upsample) your prompt before generating — text-only for new images, and reference-aware when editing. Distinct from the Refine button; off by default."
                >
                  <input
                    checked={enhancePrompt}
                    onChange={(event) => setEnhancePrompt(event.target.checked)}
                    type="checkbox"
                  />
                  Enhance prompt
                </label>
              ) : null}
              {showTierPicker ? (
                <label className="quant-tier-picker" title="Switch which installed quant tier generates, for A/B comparison. Higher precision = larger memory footprint; switching a heavy tier reloads it before the next generation. Tiers you haven't downloaded are shown but disabled — install them from the Models page to enable.">
                  Quant tier
                  <select
                    onChange={(event) => handleTierChange(event.target.value)}
                    value={quantTier}
                  >
                    {tierPickerItems.map((item) => (
                      <option key={item.tier} value={item.tier} disabled={item.disabled}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                  {tierSwitching ? (
                    <span className="field-hint" role="status">
                      Loading {tierLabel(tierSwitching)}…
                    </span>
                  ) : null}
                  {tierBelowFloor ? (
                    <span className="field-hint quant-tier-floor-note">
                      {tierLabel(quantTier)} is below the {tierLabel(qualityFloor)} recommended for{" "}
                      {selectedModel?.name ?? "this model"} — it can look washed or lose fine detail
                      here (quantization error is amplified under CFG). Your pick is honored.
                    </span>
                  ) : null}
                </label>
              ) : null}
              {precisionToggle && !showTierPicker ? (
                <label
                  className="checkline boogu-precision-toggle"
                  title="Use the full-precision bf16 build instead of the default Q8. Higher fidelity, but a much larger download (~38 GB per variant, fetched on demand) that needs a larger Mac (≈96 GB unified memory). Off = the Q8 default (~23 GB, 64 GB-class Mac)."
                >
                  <input
                    checked={bf16Precision}
                    onChange={(event) => setBf16Precision(event.target.checked)}
                    type="checkbox"
                  />
                  Full precision (bf16)
                </label>
              ) : null}
              {showPidToggle ? (
                <>
                  <label
                    className="checkline pid-decoder-toggle"
                    title="Decode this generation through NVIDIA's PiD pixel-diffusion decoder instead of the model's VAE: it decodes and super-resolves in one pass to 2K or 4K (pick the tier at right — sharper detail, but slower and more memory). Non-commercial use only — PiD output is licensed for research/evaluation, unlike the rest of the pipeline. Off = the model's native VAE at the selected resolution."
                  >
                    <input
                      checked={usePid}
                      disabled={upscaleEnabled}
                      onChange={(event) => setUsePid(event.target.checked)}
                      type="checkbox"
                    />
                    PiD decoder <span className="badge badge-nc">Non-Commercial</span>
                  </label>
                  {usePid ? (
                    <label
                      className="pid-target-select"
                      title="PiD super-resolves the base render 4×, so this sets the output size: 4K (~4096px, max detail) or 2K (~2048px, faster and less GPU memory). Both are super-resolved from the model's latent."
                    >
                      Output
                      <select
                        onChange={(event) => setPidTarget(event.target.value)}
                        value={pidTarget}
                      >
                        <option value="4k">4K · max detail</option>
                        <option value="2k">2K · faster</option>
                      </select>
                      {pidDecodeNotice ? (
                        <span className="field-hint pid-decode-hint" role="status">
                          {pidDecodeNotice.message}
                        </span>
                      ) : null}
                    </label>
                  ) : null}
                </>
              ) : null}
              <label
                className="checkline upscale-toggle"
                title={usePid ? "Disabled while the PiD decoder is on — PiD already super-resolves to 2K/4K." : undefined}
              >
                <input
                  checked={upscaleEnabled}
                  disabled={usePid}
                  onChange={(event) => setUpscaleEnabled(event.target.checked)}
                  type="checkbox"
                />
                Upscale
              </label>
              <label>
                Scale
                <select disabled={!upscaleEnabled || usePid} onChange={(event) => setUpscaleFactor(Number(event.target.value))} value={upscaleFactor}>
                  {upscaleFactorsForEngine(upscaleEngine).map((factor) => (
                    <option key={factor} value={factor}>
                      {factor}x
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Engine
                <select disabled={!upscaleEnabled || usePid} onChange={(event) => handleUpscaleEngineChange(event.target.value)} value={upscaleEngine}>
                  {availableUpscaleEngines.map((engine) => (
                    <option key={engine.key} value={engine.key}>
                      {engine.label}
                    </option>
                  ))}
                </select>
              </label>
              {upscaleEngineHasSoftness(upscaleEngine) ? (
                <label title="Higher restores more detail from a degraded source; 0 keeps it faithful.">
                  Detail
                  <input
                    aria-label="SeedVR2 detail (softness)"
                    disabled={!upscaleEnabled || usePid}
                    max="1"
                    min="0"
                    onChange={(event) => setUpscaleSoftness(Number(event.target.value))}
                    step="0.05"
                    type="range"
                    value={upscaleSoftness}
                  />
                  <span>{upscaleSoftness.toFixed(2)}</span>
                </label>
              ) : null}
              <label className="prompt-field">
                Negative prompt
                <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
              </label>
              <LoraPickerSection
                selectedModel={selectedModel}
                selectedLoras={pickerSelectedLoras}
                selectedLoraIds={pickerSelectedLoraIds}
                compatibleLoras={pickerCompatibleLoras}
                userSelectedLoraCount={userSelectedLoraCount}
                showIncompatibleLoras={showIncompatibleLoras}
                setShowIncompatibleLoras={setShowIncompatibleLoras}
                toggleLora={toggleLora}
                effectiveLoraWeight={effectiveLoraWeight}
                setLoraWeight={setLoraWeight}
                loraEmptyMessage={loraEmptyMessage}
                onUpdateLora={createLoraDownloadJob}
              />
              {/* Save-as-preset folds into Advanced with the rest of the power-user
                  knobs (UI-refinement 2b). */}
              <SavePresetPanel
                presetName={presetName}
                setPresetName={setPresetName}
                savingPreset={savingPreset}
                presetSaveMessage={presetSaveMessage}
                setPresetSaveMessage={setPresetSaveMessage}
                onSave={handleSaveAsPreset}
                presetScope={presetScope}
                setPresetScope={setPresetScope}
                activeProject={activeProject}
              />
            </div>
          </AdvancedSection>

          {/* The preset/LoRA problems that used to be three separate .inline-warning
              paragraphs (sc-10649). Requirements — no project, empty prompt — stay silent. */}
          <ValidationSummary issues={generateValidity.surfaced} label="Generate errors" />

        </WorkPanel>

        <div className="studio-results">
          <section className="review-panel">
            <div className="review-panel-head">
              <h2>Latest batch</h2>
              <span className="kbd-hint">
                <kbd>⌘</kbd>
                <kbd>↵</kbd>
                to generate
              </span>
            </div>
            {localJobGroups.length ? (
              <div className="worker-progress-card-stack local-job-stack">
                {localJobGroups.map(({ job, completedAssets, expectedCount }) => (
                  <WorkerProgressCard
                    key={job.id}
                    job={job}
                    thumbnailsVariant="image-grid"
                    thumbnailAssets={completedAssets}
                    expectedThumbnailCount={expectedCount}
                    onThumbnailClick={(asset) => onPreview(asset, completedAssets)}
                    onCancel={onCancelJob}
                    onOpenQueue={onOpenQueue}
                  />
                ))}
              </div>
            ) : null}
            {latestAssets.length ? (
              <div className="recent-assets">
                {localJobGroups.length ? <h3 className="recent-assets__title">Recent Assets</h3> : null}
                <div className="review-grid">
                  {latestAssets.map((asset) => (
                    <AssetCard
                      asset={asset}
                      deleteAsset={deleteAsset}
                      key={asset.id}
                      onPreview={(previewed) => onPreview(previewed, latestAssets)}
                      purgeAsset={purgeAsset}
                      updateAssetStatus={updateAssetStatus}
                    />
                  ))}
                </div>
              </div>
            ) : localJobGroups.length ? null : (
              <div className="empty-panel">No fresh image batch</div>
            )}
          </section>
        </div>
      </form>
      {guideOpen ? (
        <PromptGuideModal guide={promptGuide} modelName={selectedModel?.name} onClose={() => setGuideOpen(false)} />
      ) : null}
    </section>
    </ModelAvailabilityGate>
  );
}
