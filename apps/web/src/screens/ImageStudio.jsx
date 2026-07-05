import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AssetPickerField, ImageEditSourcePickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { PoseLibraryPicker } from "../components/PoseLibraryPicker.jsx";
import { RefinePromptControl } from "../components/RefinePromptControl.jsx";
import StructuredPromptBuilder from "../components/StructuredPromptBuilder.jsx";
import ReferenceCaptionPicker from "../components/ReferenceCaptionPicker.jsx";
import BatchPromptPanel from "../components/BatchPromptPanel.jsx";
import { cardinality, expandBatch, extractKeys, missingKeys, splitPromptLines } from "../promptBatch.js";
import { batchItemStatus, summarizeBatchProgress } from "../batchOps.js";
import {
  emptyCaption,
  orderCaption,
  parseMagicPromptCaption,
  parseVisionCaption,
  serializeCaption,
  validateCaption,
} from "../ideogramCaption.js";
import { buildImageJobAdvanced } from "../imageJobAdvanced.js";
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
  serializeLora,
  noPresetId,
} from "../presetUtils.js";
import {
  LoraPickerSection,
  onPromptKeyDown,
  PresetGuidanceStrip,
  PresetValidationWarnings,
  SavePresetPanel,
  useGenerationStudio,
  useSavePreset,
} from "./generationStudio.jsx";
import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import {
  downloadOffersFor,
  imageModelUsable,
  supportedControlModes,
  visionCaptionModelUsable,
} from "../modelEligibility.js";
import { ControlPanel } from "../components/ControlPanel.jsx";
import { pidToggleVisible } from "../pidEligibility.js";
import {
  defaultTierSelection,
  installedTiers,
  shouldShowTierPicker,
  tierLabel,
} from "../quantTier.js";
import { PROMPT_REFINE_MODEL_ID, VISION_CAPTION_MODEL_ID, VISION_CAPTION_MODEL_REPO } from "../constants.js";
import { pickClosestResolution } from "../resolutionMatch.js";
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
// Studio sub-modes a saved preset may restore (the "type") — the tabs the mode
// segmented control actually exposes. Edit lives in its own workflow; text and
// character share the text_to_image workflow.
const IMAGE_MODES = ["text_to_image", "edit_image", "character_image"];

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

function finiteRecipeNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function recipeResolution(recipe) {
  const settings = recipe?.normalizedSettings ?? {};
  const width = finiteRecipeNumber(settings.width);
  const height = finiteRecipeNumber(settings.height);
  if (width && height) {
    return `${width}x${height}`;
  }
  const rawResolution = recipe?.rawAdapterSettings?.resolution;
  return typeof rawResolution === "string" && rawResolution.includes("x") ? rawResolution : null;
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

function recipeLoraId(lora) {
  return typeof lora === "string" ? lora : lora?.id ?? lora?.loraId;
}

function recipeLoraWeight(lora) {
  if (typeof lora === "string") {
    return undefined;
  }
  return finiteRecipeNumber(lora?.weight) ?? undefined;
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
  const tierOptions = useMemo(() => ({ convRotEligible }), [convRotEligible]);
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
  const [prompt, setPrompt] = useState(saved.prompt ?? "A cinematic frame of a neon street at midnight");
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
  const suggestions = mode === "character_image" ? characterSuggestions : sceneSuggestions;
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

  const batchPrompts = useMemo(() => splitPromptLines(batchPromptsText), [batchPromptsText]);
  const batchVariables = useMemo(
    () => extractKeys(batchPrompts).map((key) => ({ key, values: batchVariableValues[key] ?? [] })),
    [batchPrompts, batchVariableValues],
  );
  const batchTotal = useMemo(
    () => cardinality(batchPrompts, batchVariables, count),
    [batchPrompts, batchVariables, count],
  );

  // A pending large-run confirmation is for one specific count — reset it whenever the
  // batch size changes so the user always re-confirms against the current total.
  useEffect(() => {
    setBatchConfirmPending(false);
  }, [batchTotal]);

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
  // Edit fit mode (epic 2551): how the source is fitted to the output W×H. Never stretch.
  const [fitMode, setFitMode] = useState(saved.fitMode ?? "crop");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  // Character reference (IP-Adapter / InstantID) — the approved reference image whose
  // identity is carried across variations. `ipAdapterScale` rides in `advanced`; for
  // InstantID, `controlnetScale` (IdentityNet landmark lock) rides there too.
  const [referenceAssetId, setReferenceAssetId] = useState("");
  const [ipAdapterScale, setIpAdapterScale] = useState(saved.ipAdapterScale ?? 0.6);
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
  // holds the selected tier key ("" = no picker / not applicable). Last-used is persisted PER
  // MODEL in `lastUsedTiers` so re-entering a model restores the tier you last generated with.
  const [lastUsedTiers, setLastUsedTiers] = useState(
    saved.lastUsedTiers && typeof saved.lastUsedTiers === "object" ? saved.lastUsedTiers : {},
  );
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
  // The control type actually in effect: the user's pick when the backbone still supports it, else the
  // first supported mode. Decouples the gating (derived) from the raw state so a backbone switch that
  // strands an unsupported pick degrades gracefully even before the reset effect runs.
  const activeControlMode = controlModes.includes(controlMode) ? controlMode : (controlModes[0] ?? null);
  const showControlPanel = mode === "text_to_image" && controlModes.length > 0;
  const effectiveControlScale =
    typeof controlScale === "number" ? controlScale : controlScaleConfig?.default ?? 0.9;
  // Whether the model exposes its built-in prompt upsampler ("Enhance prompt" toggle) — FLUX.2-dev.
  const promptEnhance = Boolean(selectedModel?.ui?.promptEnhance);
  // Whether the model ships a packed default + a hosted full-precision bf16 build, exposing the
  // Studio "Full precision (bf16)" toggle (sc-6568) — Boogu Base/Turbo/Edit.
  const precisionToggle = Boolean(selectedModel?.ui?.precisionToggle);
  // Installed quant tiers of the active model + whether the tier picker should show (sc-8515).
  // The picker renders only when MORE THAN ONE tier is installed; a single installed tier keeps
  // the studio unchanged (no toggle). Boogu's `precisionToggle` is orthogonal — those models are
  // single-download (no variant matrix), so they never hit this path.
  const availableTiers = useMemo(
    () => installedTiers(selectedModel, tierOptions),
    [selectedModel, tierOptions],
  );
  const showTierPicker = useMemo(
    () => shouldShowTierPicker(selectedModel, tierOptions),
    [selectedModel, tierOptions],
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
  // view angle never carries over to a model that doesn't support it. Skip the mount
  // run when restoring a snapshot so the user's saved tuning survives.
  const skipReferenceTuningReset = useRef(saved.ipAdapterScale != null);
  useEffect(() => {
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
    setControlImageAssetId("");
    setControlImagePassthrough(false);
  }, [model]);
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
  // Seed the model's curated default negative prompt when entering character mode
  // with an empty box (sc-3857). InstantID/RealVisXL declares one to fight its
  // shiny/over-saturated look; running character mode with an empty negative was
  // the main reason Image Studio output trailed Character Studio. Only fills an
  // empty box, so it never clobbers a typed, restored, or preset negative.
  useEffect(() => {
    if (mode !== "character_image" || negativePrompt !== "") {
      return;
    }
    const ui = imageModels.find((item) => item.id === model)?.ui ?? {};
    if (typeof ui.defaultNegativePrompt === "string" && ui.defaultNegativePrompt) {
      setNegativePrompt(ui.defaultNegativePrompt);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, model]);
  const resolutionOptions = useMemo(
    () =>
      selectedModel?.limits?.resolutions?.length
        ? selectedModel.limits.resolutions
        : DEFAULT_RESOLUTION_OPTIONS,
    [selectedModel],
  );
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
  useEffect(() => {
    if (samplerOptions.includes(sampler)) {
      return;
    }
    setSampler(preferredOption(samplerDefaultFromModel(selectedModel), samplerOptions));
  }, [samplerOptions, sampler, selectedModel]);
  useEffect(() => {
    if (schedulerOptions.includes(scheduler)) {
      return;
    }
    setScheduler(preferredOption(schedulerDefaultFromModel(selectedModel), schedulerOptions));
  }, [schedulerOptions, scheduler, selectedModel]);
  // Snap the guidance method back to "cfg" when the current choice isn't honored by
  // the active backend for this model (e.g. switching off the SDXL family drops
  // CFG++) — the N3 guard at the UI layer, so an unsupported method is never sent.
  useEffect(() => {
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
  useEffect(() => {
    if (resolutionOptions.includes(resolution)) {
      return;
    }
    setResolution(preferredResolution(selectedModel, resolutionOptions));
  }, [resolutionOptions, resolution, selectedModel]);
  // Keep the selected quant tier valid for the active model (sc-8515). When the current tier is
  // still installed for this model, leave it; otherwise snap to the model's default selection
  // (last-used-for-this-model → declared default → q4 → first installed). Also clears to "" when
  // no tier is installed / the model has no matrix, so a stale tier never leaks into the payload.
  // Keyed on `model` (not `selectedModel`) plus the installed-tier list so a catalog refresh that
  // newly installs a second tier re-derives the default without churning on every render.
  const availableTiersKey = availableTiers.join(",");
  useEffect(() => {
    if (availableTiers.includes(quantTier)) {
      return;
    }
    setQuantTier(defaultTierSelection(selectedModel, lastUsedTiers[model], tierOptions) ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, availableTiersKey]);
  // Switch the active quant tier (sc-8515): persist it as this model's last-used tier and surface
  // a brief "loading <tier>" note (reload-always — the worker evicts + reloads a heavy tier on the
  // next generation; there is no co-residence). The note is cosmetic and self-clears.
  const tierSwitchTimer = useRef(null);
  useEffect(() => () => clearTimeout(tierSwitchTimer.current), []);
  const handleTierChange = useCallback(
    (nextTier) => {
      if (nextTier === quantTier) {
        return;
      }
      setQuantTier(nextTier);
      setLastUsedTiers((prev) => ({ ...prev, [model]: nextTier }));
      setTierSwitching(nextTier);
      clearTimeout(tierSwitchTimer.current);
      tierSwitchTimer.current = setTimeout(() => setTierSwitching(""), 1500);
    },
    [model, quantTier],
  );
  const {
    availablePresets,
    selectedPreset,
    selectedPresetId,
    setSelectedPresetId,
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
  });
  useEffect(() => {
    if (launchRequest?.view !== "Image" || !launchRequest.recipe) {
      return;
    }
    const recipe = launchRequest.recipe;
    const settings = recipe.normalizedSettings ?? {};
    const rawSettings = recipe.rawAdapterSettings ?? {};
    const nextMode = recipeMode(recipe);
    const resolutionFromRecipe = recipeResolution(recipe);
    const recipeLoras = Array.isArray(recipe.loras) ? recipe.loras : [];
    const loraIds = recipeLoras.map(recipeLoraId).filter(Boolean);
    const loraWeightMap = Object.fromEntries(
      recipeLoras
        .map((lora) => [recipeLoraId(lora), recipeLoraWeight(lora)])
        .filter(([id, weight]) => id && weight !== undefined),
    );

    skipReferenceTuningReset.current = true;
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
    if (restoredCaption && validateCaption(restoredCaption).ok) {
      setCaption(orderCaption(restoredCaption));
      setPromptMode("form");
      setMagicPromptBackend(structuredRecipe.magicPromptBackend ?? null);
      // The intent (original idea) seeds the plain box; the serialized caption is
      // authoritative for generation and is rebuilt from `caption` on submit.
      setPrompt(String(structuredRecipe.intent ?? ""));
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
  const [width, height] = resolution.split("x").map((value) => Number(value));

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
    async (sourceAssetId) => {
      if (typeof imageCaption !== "function") {
        throw new Error("Image captioning is unavailable.");
      }
      if (!activeProject?.id) {
        throw new Error("Open a project first.");
      }
      const raw = await imageCaption({
        sourceAssetId,
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
    async (sourceAssetId) => {
      if (typeof imageDescribe !== "function") {
        throw new Error("Image description is unavailable.");
      }
      if (!activeProject?.id) {
        throw new Error("Open a project first.");
      }
      const text = await imageDescribe({
        sourceAssetId,
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
    structuredCaption: caption,
    promptMode,
    magicPromptBackend,
    count,
    advancedOpen,
    model,
    seed,
    negativePrompt,
    resolution,
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
    lastUsedTiers,
  });

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
    setSubmitting(true);
    try {
      // Pose library: when poses are selected, the job emits one image per pose
      // (advanced.poses) instead of `count` variations. Two pose surfaces share this payload:
      //   * character_image — InstantID pose set (needs an approved reference).
      //   * text_to_image — the strict-control panel's pose mode (sc-8245): a Fun-Union backbone
      //     conditions each render on the selected library skeleton; no reference needed.
      const usePosePayload =
        (mode === "character_image" && referenceAssetId && poseLibrary) ||
        (showControlPanel && activeControlMode === "pose");
      const posePayload =
        usePosePayload && selectedPoseIds.length
          ? selectedPoseIds.map((id) => poseById[id]).filter(Boolean).map((pose) => ({ id: pose.id, keypoints: pose.keypoints }))
          : [];
      // Strict-control conditioning (sc-8245). Active only for a text-to-image control backbone.
      // Pose flows through `posePayload` above; canny/depth carry the control type + the uploaded
      // control image, routed by the preprocess-vs-passthrough toggle:
      //   * preprocess (derive) → request `sourceAssetId` (the worker auto-derives the map).
      //   * use-as-is (passthrough) → `advanced.controlImage` (the map is fed verbatim).
      const controlActive = showControlPanel && Boolean(activeControlMode);
      const controlIsImageMode = controlActive && activeControlMode !== "pose";
      const controlPreprocessSourceId =
        controlIsImageMode && !controlImagePassthrough && controlImageAssetId ? controlImageAssetId : null;
      const controlPassthroughId =
        controlIsImageMode && controlImagePassthrough && controlImageAssetId ? controlImageAssetId : null;
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
      const job = await createImageJob({
        mode,
        prompt: promptToSend,
        negativePrompt,
        model,
        count: posePayload.length ? 1 : count,
        seed: seed === "" ? null : Number(seed),
        width,
        height,
        recipePresetId: selectedPreset?.id ?? null,
        characterId: mode === "character_image" ? characterId || null : null,
        characterLookId: mode === "character_image" ? characterLookId || null : null,
        // edit_image: a single source image, except for a multi-reference model (sc-6211,
        // FLUX.2-dev) whose source picker is replaced by the multi-image reference picker below.
        // text_to_image strict-control (sc-8245): canny/depth in preprocess (derive) mode send the
        // uploaded control image here as the source the worker auto-derives the map FROM
        // (strict_control.rs `resolve_control_source`). Passthrough mode uses `advanced.controlImage`.
        sourceAssetId:
          mode === "edit_image" && !multiReference
            ? sourceAssetId || null
            : controlPreprocessSourceId,
        // Multi-reference edit (sc-6211): the plural reference set the FLUX.2-dev edit conditions on.
        // Only sent in edit_image mode for a multiReference model; the worker routes a non-empty list
        // to Conditioning::MultiReference (one image ⇒ a normal single-reference edit).
        referenceAssetIds:
          mode === "edit_image" && multiReference && referenceAssetIds.length ? referenceAssetIds : undefined,
        // Fit mode applies to edits only; coerced so a stale "outpaint" never reaches a
        // non-inpaint model (epic 2551). Omitted for non-edit modes (worker default crop).
        fitMode: mode === "edit_image" ? effectiveFitMode(fitMode, editInpaintCapable) : undefined,
        referenceAssetId: mode === "character_image" ? referenceAssetId || null : null,
        loras: selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
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
        // builder so this async submit() stays focused on prompt resolution + the API
        // call. Every omit-when-default rule (which keeps saved recipes byte-identical)
        // lives in imageJobAdvanced.js and is covered by imageJobAdvanced.test.js.
        advanced: buildImageJobAdvanced({
          resolution,
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
          mode,
          referenceAssetId,
          hideReferenceStrength,
          ipAdapterScale,
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
        }),
      });
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }

  // One image-job request for a single resolved batch prompt. Reuses the current
  // studio settings (model, loras, upscale, reference/source per mode, and the whole
  // advanced knob set via the tested buildImageJobAdvanced). Batch deliberately omits
  // the single-shot pose-library / strict-control conditioning and never sends a
  // structured caption — each resolved prompt is a plain prompt. `count` still
  // multiplies within each job (images = jobs × count).
  const buildBatchJobRequest = (resolvedPrompt) => ({
    mode,
    prompt: resolvedPrompt,
    negativePrompt,
    model,
    count,
    seed: seed === "" ? null : Number(seed),
    width,
    height,
    recipePresetId: selectedPreset?.id ?? null,
    characterId: mode === "character_image" ? characterId || null : null,
    characterLookId: mode === "character_image" ? characterLookId || null : null,
    sourceAssetId: mode === "edit_image" && !multiReference ? sourceAssetId || null : null,
    referenceAssetIds:
      mode === "edit_image" && multiReference && referenceAssetIds.length ? referenceAssetIds : undefined,
    fitMode: mode === "edit_image" ? effectiveFitMode(fitMode, editInpaintCapable) : undefined,
    referenceAssetId: mode === "character_image" ? referenceAssetId || null : null,
    loras: selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
    ...(upscaleEnabled
      ? {
          upscale: {
            enabled: true,
            factor: upscaleFactor,
            engine: upscaleEngine,
            ...(upscaleEngineHasSoftness(upscaleEngine) ? { softness: upscaleSoftness } : {}),
          },
        }
      : {}),
    advanced: buildImageJobAdvanced({
      resolution,
      sendStructured: false,
      submitIntent: resolvedPrompt,
      submitCaption: caption,
      submitBackend: magicPromptBackend,
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
      mode,
      referenceAssetId,
      hideReferenceStrength,
      ipAdapterScale,
      identityStructure,
      controlnetScale,
      variationStrength,
      trueCfgScale,
      viewAngles,
      viewAngle,
      posePayload: [],
      faceRestore,
      controlActive: false,
      activeControlMode,
      controlPassthroughId: null,
      effectiveControlScale,
    }),
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
    // Soft cap: a large run must be confirmed once, showing the exact image count.
    if (!confirmed && resolved.length * count > BATCH_RENDER_CAP) {
      setBatchConfirmPending(true);
      return;
    }
    setBatchConfirmPending(false);
    setBatchRun({ submitting: true, items: resolved.map((entry) => ({ prompt: entry.prompt, jobId: null })) });
    const items = [];
    for (const entry of resolved) {
      try {
        const job = await createImageJob(buildBatchJobRequest(entry.prompt));
        items.push({ prompt: entry.prompt, jobId: job?.id ?? null });
      } catch {
        items.push({ prompt: entry.prompt, jobId: null });
      }
    }
    setBatchRun({ submitting: false, items });
  }

  // Cancel every still-pending job in the current run; completed/failed items are left.
  function cancelBatchRun() {
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

  const batchRunProgress = batchRun ? summarizeBatchProgress(batchRun.items, jobs) : null;
  const batchMissingKeys = missingKeys(batchPrompts, batchVariables);
  const batchRunDisabled =
    !activeProject ||
    structuredPromptModel ||
    batchTotal === 0 ||
    batchMissingKeys.length > 0 ||
    Boolean(batchRun?.submitting);

  const generateDisabled =
    submitting ||
    !activeProject ||
    // Structured models gate on a valid, non-empty caption; everyone else on a
    // non-empty prompt. (Plain-text fallback falls through to the prompt check.)
    (structuredActive ? !captionValidation?.ok || !captionHasContent : !prompt.trim()) ||
    (mode === "character_image" && !characterId) ||
    Boolean(macActiveModeBlock) ||
    !presetValidationResult.ok ||
    !selectedLoraValidationResult.ok;

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
    <section className="main-surface image-studio">
      <form className="studio-shell" onSubmit={submit}>
        <div className="surface-header hero studio-prompt-hero">
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
                count={count}
                batches={promptBatches}
                projectId={activeProject?.id ?? null}
                name={batchName}
                onNameChange={setBatchName}
                scope={batchScope}
                onScopeChange={setBatchScope}
                loadedBatchId={loadedBatchId}
                onSave={handleSaveBatch}
                onLoad={handleLoadBatch}
                onDelete={handleDeleteBatch}
                onImport={handleImportBatch}
                busy={batchBusy}
                error={batchError}
              />
              <div className="batch-run">
                {structuredPromptModel ? (
                  <p className="batch-warning">
                    Batch mode isn’t available for structured-caption models yet.
                  </p>
                ) : batchMissingKeys.length > 0 ? (
                  <p className="batch-warning">
                    Fill in a value for {batchMissingKeys.map((key) => `{{${key}}}`).join(", ")} to run.
                  </p>
                ) : batchTotal === 0 ? (
                  <p className="batch-hint">Add at least one prompt to run a batch.</p>
                ) : null}
                {batchRun ? (
                  <div className="batch-run-progress" aria-live="polite">
                    <span>
                      {batchRunProgress.done}/{batchRunProgress.total} done
                      {batchRunProgress.failed ? ` · ${batchRunProgress.failed} failed` : ""}
                    </span>
                    {batchRunProgress.allDone ? (
                      <button className="batch-btn ghost" onClick={() => setBatchRun(null)} type="button">
                        Clear
                      </button>
                    ) : (
                      <button className="batch-btn ghost" onClick={cancelBatchRun} type="button">
                        Cancel remaining
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

          {/* Prompt tools (UI-refinement 1b): one framed strip of two equally-weighted toggle
              tiles that replace the disjointed ReferenceCaptionPicker card + lone Refine link.
              Tile A wraps the reference-image → plain-text describe flow (epic 8203, sc-8208):
              gated to text-to-image and hidden unless the macOS-first captioner is platform-
              eligible (ready, or an install offer exists — both false off-Mac). C1: captioning-
              only, never sent to generation. Tile B wraps RefinePromptControl (sc-2041). Only
              one panel opens at a time; both are free-text only (structured models excluded). */}
          {structuredPromptModel ? null : (() => {
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
                  {describeAvailable ? (
                    <button
                      type="button"
                      className={describeActive ? "prompt-tool active" : "prompt-tool"}
                      aria-pressed={describeActive}
                      onClick={() => togglePromptTool("describe")}
                    >
                      <span className="prompt-tool-title">
                        <Icon.Image size={15} /> Start from an image
                      </span>
                      <span className="prompt-tool-desc">Caption a reference into an editable prompt</span>
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
                  <ImageEditSourcePickerField
                    assets={editImageAssets}
                    buttonLabel="Select image"
                    characters={characters}
                    emptyLabel="No source image selected"
                    importAsset={importAsset}
                    label="Source image"
                    onChange={setSourceAssetId}
                    projectId={activeProject?.id}
                    value={sourceAssetId}
                  />
                )}
                <FitModeControl
                  value={effectiveFitMode(fitMode, editInpaintCapable)}
                  onChange={setFitMode}
                  inpaintCapable={editInpaintCapable}
                />
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
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {pickerModels.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
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
            <div className="settings-bar-styles">
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

          {macActiveModeBlock ? <p className="mac-gating-note">{macActiveModeBlock.text}</p> : null}

          <PresetGuidanceStrip
            selectedPreset={selectedPreset}
            presetPromptParts={presetPromptParts}
            presetLoraDetails={presetLoraDetails}
          />

          <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
            <Icon.ChevDown className={advancedOpen ? "chev-rotate open" : "chev-rotate"} size={14} />
            {advancedOpen ? "Hide advanced" : "Advanced"}
          </button>

            {advancedOpen ? (
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
                  <label className="quant-tier-picker" title="Switch which installed quant tier generates, for A/B comparison. Higher precision = larger memory footprint; switching a heavy tier reloads it before the next generation.">
                    Quant tier
                    <select
                      onChange={(event) => handleTierChange(event.target.value)}
                      value={quantTier}
                    >
                      {availableTiers.map((tier) => (
                        <option key={tier} value={tier}>
                          {tierLabel(tier)}
                        </option>
                      ))}
                    </select>
                    {tierSwitching ? (
                      <span className="field-hint" role="status">
                        Loading {tierLabel(tierSwitching)}…
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
                  <label
                    className="checkline pid-decoder-toggle"
                    title="Decode this generation through NVIDIA's PiD pixel-diffusion decoder instead of the model's VAE: it decodes and super-resolves in one pass, so output comes out at 2K/4K (sharper detail, but slower and more memory). Non-commercial use only — PiD output is licensed for research/evaluation, unlike the rest of the pipeline. Off = the model's native VAE at the selected resolution."
                  >
                    <input
                      checked={usePid}
                      disabled={upscaleEnabled}
                      onChange={(event) => setUsePid(event.target.checked)}
                      type="checkbox"
                    />
                    PiD decoder · 2K/4K <span className="badge badge-nc">Non-Commercial</span>
                  </label>
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
                  selectedLoras={selectedLoras}
                  selectedLoraIds={selectedLoraIds}
                  compatibleLoras={compatibleLoras}
                  userSelectedLoraCount={userSelectedLoraCount}
                  showIncompatibleLoras={showIncompatibleLoras}
                  setShowIncompatibleLoras={setShowIncompatibleLoras}
                  toggleLora={toggleLora}
                  effectiveLoraWeight={effectiveLoraWeight}
                  setLoraWeight={setLoraWeight}
                  loraEmptyMessage={loraEmptyMessage}
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
            ) : null}

          <PresetValidationWarnings presetValidationResult={presetValidationResult} selectedModel={selectedModel} />
          {selectedLoraValidationResult.incompatible.length ? (
            <p className="inline-warning">
              Generate is blocked because these selected LoRAs are incompatible with {selectedModel?.name ?? "the selected model"}: {selectedLoraValidationResult.incompatible.join(", ")}.
            </p>
          ) : null}
        </div>

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
