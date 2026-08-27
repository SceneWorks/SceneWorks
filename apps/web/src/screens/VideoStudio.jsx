import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { parseResolution, pickClosestResolution } from "../resolutionMatch.js";
import { AssetPickerField, ImageEditSourcePickerField, VideoSourcePickerField } from "../components/AssetPicker.jsx";
import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { StudioLoraImportPanel } from "../components/StudioLoraImportPanel.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { RefinePromptControl } from "../components/RefinePromptControl.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../components/StudioUpdateNotice.jsx";
import { VideoUpscalePanel } from "./VideoUpscalePanel.jsx";
import { StyledPromptPreview } from "../components/StyledPromptPreview.jsx";
import { STYLE_GROUPS, styleTextForId } from "../data/styleCatalog.js";
import { composeStyledPrompt } from "../styleComposer.js";
import { resolveJobResultAssets } from "../jobResultAssets.js";

const MOTIONS = [
  "static",
  "slow push-in",
  "pull out",
  "pan left",
  "pan right",
  "tilt up",
  "tilt down",
  "handheld",
];
// Named so the initial state and the recipe replay's "absent → default" reset can't drift apart.
const DEFAULT_MOTION = "slow push-in";

// Resolve a video job's result assets against the live catalog so the
// WorkerProgressCard video-player variant can play the finished clip (sc-2089).
// Shares the unified resolver (sc-8853); the video lane keeps the generationSetId
// fallback in catalog order (no batch-slot sort — that is image-only).
function jobVideoResultAssets(job, assets) {
  return resolveJobResultAssets(job, assets, { type: "video" });
}
import {
  finiteNumberOrUndefined,
  loraLooksLikeIcLora,
  noPresetId,
  serializeLora,
  composePreset,
} from "../presetUtils.js";
import {
  LoraPickerSection,
  onPromptKeyDown,
  PresetStackPreview,
  SavePresetPanel,
  ModeTabs,
  StyleAxisRow,
  TierPickerField,
  useGenerationStudio,
  useQuantTierPicker,
  useSavePreset,
} from "../components/generationStudio.jsx";
import { ReplacePersonPanel } from "./ReplacePersonPanel.jsx";
import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import {
  replacementModeApplies,
  SCAIL2_MODEL_ID,
  videoGenerateValidation,
} from "../videoStudioValidation.js";
import { useValidation } from "../validation/useValidation.js";
import { ValidationSummary } from "../validation/Validation.jsx";
import {
  VIDEO_MODES,
  downloadOffersFor,
  videoModelServesMode,
  videoModelUsable,
} from "../modelEligibility.js";
import {
  defaultTurboVariant,
  modelIsMinimaxH3,
  selectedTurboVariant,
  turboRecipeSummary,
  turboVariantsForModel,
} from "../minimaxH3Turbo.js";
import { PROMPT_REFINE_MODEL_ID, WAN_A14B_LIGHTNING_MODEL_IDS } from "../constants.js";
import {
  DEFAULT_MAC_CAPABILITIES,
  macAvailableModels,
  macGatingActive,
} from "../macGating.js";
import { candleAvailableModels, candleGatingActive } from "../candleGating.js";
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import { qualityChoices } from "../jobTypes.js";
import {
  allPossibleTiers,
  installedTiers,
  quantizeTier,
  tierLabel,
  tierPickerOptions,
  tierQuantize,
} from "../quantTier.js";
import { suggestTier } from "../tierSuggestion.js";
import { useHostMemory } from "../hooks/useHostMemory.js";
import { hostMemoryGbForBackend } from "../hostMemory.js";
import {
  finiteRecipeNumber,
  recipeLoraSelection,
  recipeRequestedResolution,
} from "../recipeFields.js";
import {
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  stepsDefaultFromModel,
  stepsMenuFromModel,
} from "../samplerOptions.js";
import {
  formatDurationOption,
  minStepsForModel,
  referenceCaps,
  referenceLimitError,
} from "../videoModelLimits.js";
import { ModelAttribution } from "../components/ModelAttribution.jsx";

// "8" / "4 or 8" / "4, 8, or 12" — the same phrasing `humanized_number_menu` gives the enqueue
// gate's own rejection (crates/sceneworks-core/src/video_request.rs, sc-19502), so the Steps
// picker's tooltip states the legal set the way a 400 from that gate would.
function humanizedNumberMenu(menu) {
  if (menu.length <= 1) return String(menu[0] ?? "");
  if (menu.length === 2) return `${menu[0]} or ${menu[1]}`;
  return `${menu.slice(0, -1).join(", ")}, or ${menu[menu.length - 1]}`;
}

const ltxVideoModelId = "ltx_2_3";
const ltx25VideoModelId = "ltx_2_5";
const ltxIcLoraModelIds = new Set([ltxVideoModelId, "ltx_2_3_eros", ltx25VideoModelId]);
// Keep this list to native Candle engines that publish a real Model Manager variant matrix. The
// picker only enables entries whose individual install is complete (`installedTiers`); in particular,
// adding SCAIL-2 here never fabricates q4/q8 for a dense-only or partial local snapshot.
const candleTierModelIds = new Set([
  "wan_2_2",
  "wan_2_2_t2v_14b",
  "wan_2_2_i2v_14b",
  "scail2_14b",
  ltx25VideoModelId,
]);
// sc-20969 terminal acceptance admits exactly SCAIL-2's shipped q4/q8/bf16 Candle packages. Keep
// this execution allowlist literal and local so an unrelated future catalog tier cannot become
// runnable merely by appearing in the manifest. MLX continues to use the complete installed tier set.
const SCAIL2_CANDLE_PRODUCT_TIERS = Object.freeze(["q4", "q8", "bf16"]);
const legacyDefaultTextEncoderId = "default";
const amoralTextEncoderId = "ltx_amoral_gemma_3_12b";
const ltxIcLoraRequiredModes = new Set(["extend_clip", "video_bridge", "replace_person"]);
const TIER_SCREEN = "video";
const MAX_SCAIL2_REFERENCE_CHARACTERS = 6;

function videoExecutionTierModel(model, backend) {
  if (backend !== "candle" || model?.id !== SCAIL2_MODEL_ID) {
    return model;
  }
  const accepted = (tier) => SCAIL2_CANDLE_PRODUCT_TIERS.includes(tier);
  return {
    ...model,
    variants: Array.isArray(model.variants)
      ? model.variants.filter((variant) => accepted(variant?.variant))
      : model.variants,
    runtimeQuantTiers: Array.isArray(model.runtimeQuantTiers)
      ? model.runtimeQuantTiers.filter(accepted)
      : model.runtimeQuantTiers,
    mlxTiers: Array.isArray(model.mlxTiers)
      ? model.mlxTiers.filter(accepted)
      : model.mlxTiers,
    mlxTierStates: Array.isArray(model.mlxTierStates)
      ? model.mlxTierStates.filter((state) => accepted(state?.tier))
      : model.mlxTierStates,
  };
}

function unavailableRecipeTierMessage(model, backend, tier) {
  const recorded = tierLabel(tier);
  if (
    backend === "candle" &&
    model?.id === SCAIL2_MODEL_ID &&
    !SCAIL2_CANDLE_PRODUCT_TIERS.includes(tier)
  ) {
    return `This recipe was recorded at ${recorded}, which is not enabled for SCAIL-2 Candle generation. Replay it on MLX, or install an admitted SCAIL-2 tier in Model Manager before starting a new generation.`;
  }
  return `This recipe was recorded at ${recorded}, but that tier is not installed and available for ${model?.name ?? "the selected model"} on ${backend === "mlx" ? "MLX" : "Candle"}. Install or repair that tier in Model Manager before replaying this recipe.`;
}

// Video sub-modes that map onto a recipe workflow. extend_clip / replace_person
// aren't recipe workflows, so "Save as Preset" is gated to these.
const VIDEO_PRESET_MODES = ["image_to_video", "text_to_video", "first_last_frame"];

export function VideoStudio() {
  const {
    activeProject,
    assets,
    characters,
    createPersonDetectionJob,
    createPersonTrackJob,
    createVideoJob,
    createVideoUpscaleJob,
    createPreset,
    refinePrompt,
    createModelDownloadJob,
    createLoraDownloadJob,
    createLoraImportJob,
    deleteAsset,
    purgeAsset,
    gpuOptions,
    importAsset,
    latestVideoAssets,
    recentVideoAssets,
    studioLaunch,
    loras = [],
    jobs = [],
    videoLocalJobs = [],
    jobAction,
    rememberLocalGenerationJob,
    setActiveView,
    setPreviewAsset,
    personTracks = [],
    personReadiness = {},
    presets = [],
    requestedGpu,
    saveTrackCorrections,
    selectedAsset,
    selectedAssetId,
    setRequestedGpu,
    updateAssetStatus,
    videoModels,
    models = [],
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
    preferencesHydrated,
  } = useAppContext();
  // Prompt-refinement model catalog entry (sc-5605) — drives the "download the
  // refinement model" affordance in RefinePromptControl when Refine fails because the
  // model isn't provisioned on the native worker.
  const refineModel = useMemo(
    () => models.find((entry) => entry.id === PROMPT_REFINE_MODEL_ID),
    [models],
  );
  // Recent Assets (sc-2089) — 20 most recent video assets in the active
  // project. Falls back to the legacy single-generation list for test
  // contexts that haven't migrated.
  const latestAssets = recentVideoAssets ?? latestVideoAssets;
  const launchRequest = studioLaunch;
  const trackedLocalJobs = videoLocalJobs;
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onLocalJobCreated = (job) => rememberLocalGenerationJob("video", job);
  const onOpenPresets = () => setActiveView("Presets");
  const onOpenQueue = () => setActiveView("Queue");
  const onPreview = setPreviewAsset;
  // Last-used settings for this workspace, restored on mount. The component is keyed
  // by workspace in App.jsx, so this reads the right snapshot per workspace.
  const saved = useMemo(() => loadStudioSettings("video", activeProject?.id ?? null), [activeProject?.id]);
  const [motion, setMotion] = useState(saved.motion ?? DEFAULT_MOTION);
  // Memoize the per-type catalog splits (sc-8939): both feed a dozen pickers/trays and
  // re-filtering the full catalog on every render (including unrelated state churn) is
  // needless. Recompute only when the catalog changes; stable identities also keep the
  // downstream memoized offers/consumers from thrashing.
  const imageAssets = useMemo(
    () => assets.filter((asset) => asset.type === "image" || asset.type === "frame"),
    [assets],
  );
  const videoAssets = useMemo(() => assets.filter((asset) => asset.type === "video"), [assets]);
  // Library audio tracks the reference-audio picker can offer (sc-17161) — the audio twin of
  // `videoAssets`, type-scoped so the picker never offers a render as a "voice".
  const audioAssets = useMemo(() => assets.filter((asset) => asset.type === "audio"), [assets]);
  // Open on Text→Video for parity with Image Studio's Text→Image default and the
  // launch-request fallback below (sc-5716); the prior image_to_video default was the odd one out.
  const [mode, setMode] = useState(saved.mode ?? "text_to_video");
  const [prompt, setPrompt] = useState(saved.prompt ?? "Camera slowly pushes in while the scene comes alive");
  // Style Catalog selection (sc-13136): an entry id from styles.json (a group id or a sub-style id)
  // or null for "None"/pass-through. The id is opaque here; styleTextForId resolves it to the
  // composer's free-text at build time. Persisted via saved-state, mirroring the Image Studio.
  const [styleId, setStyleId] = useState(saved.styleId ?? null);
  const [quality, setQuality] = useState(saved.quality ?? "balanced");
  const [ltxPipeline, setLtxPipeline] = useState(saved.ltxPipeline ?? "auto");
  const [distilledVariant, setDistilledVariant] = useState(saved.distilledVariant ?? "1.1");
  const [transformerVariant, setTransformerVariant] = useState(saved.transformerVariant ?? "distilled");
  const [precision, setPrecision] = useState(saved.precision ?? "fp8");
  const [enhancePrompt, setEnhancePrompt] = useState(saved.enhancePrompt ?? false);
  const [textEncoderSelection, setTextEncoderSelection] = useState({
    modelId: saved.model ?? null,
    id: saved.textEncoderModel ?? null,
  });
  const [quantization, setQuantization] = useState(saved.quantization ?? "auto");
  // MLX generation tier (sc-12165), separate from the torch/GGUF `quantization` state above.
  // The explicit pick is persisted per (video, model), outside the workspace settings snapshot.
  // The model a replayed recipe asked for, when it isn't installed (sc-12324). Its settings still
  // restore; the mode-snap effect moves the picker to a model that serves the mode. Named rather
  // than silent so the swap doesn't read as the recipe's own choice.
  const [recipeModelNotice, setRecipeModelNotice] = useState("");
  // A recorded native tier is an exact replay request, not a preference. Keep it separate from the
  // picker state so an unavailable/disallowed replay can block instead of being clamped to whatever
  // installed tier the current model would normally seed.
  const [recipeTierRequest, setRecipeTierRequest] = useState(null);
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  const [model, setModel] = useState(saved.model ?? videoModels[0]?.id ?? ltxVideoModelId);
  // Every USER model picker must retire replay-only state. Automatic catalog/mode/recipe snaps use
  // the raw state setter below and preserve the replay while its target model becomes active.
  const setUserModel = useCallback((nextModel) => {
    setRecipeModelNotice("");
    setRecipeTierRequest(null);
    setModel(nextModel);
  }, []);
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
  const [guideOpen, setGuideOpen] = useState(false);
  // Platform UI gating: hide whole models with no lane on THIS platform and snap off one if
  // selected. Two composed partitions, one per platform, each a no-op on the other's platform:
  //   * `macAvailableModels` (sc-3486) — torch-only models (e.g. SVD) on a gated Mac.
  //   * `candleAvailableModels` (sc-19570) — models with no candle lane at all off-Mac (e.g.
  //     `wan_2_2_vace_fun_14b`, whose only advertised mode is candle-unclaimable). Before this the
  //     export existed and NOTHING imported it, so whole-model hiding was dead in the one screen
  //     that matters and the picker still listed a model no off-Mac worker can claim.
  const macVideoModels = useMemo(
    () => candleAvailableModels(macAvailableModels(videoModels, macCapabilities), macCapabilities),
    [videoModels, macCapabilities],
  );
  useEffect(() => {
    if (macVideoModels.length && !macVideoModels.some((item) => item.id === model)) {
      setModel(macVideoModels[0].id);
    }
  }, [macVideoModels, model]);
  const selectedModel = videoModels.find((item) => item.id === model) ?? videoModels[0];
  const ltx25Dev = model === ltx25VideoModelId && transformerVariant === "dev";
  // Multi-reference SCAIL-2 needs a reference/mask pair per character. Keep this source-ready UI
  // behind the descriptor-derived manifest flag: the currently pinned engine descriptor does not
  // advertise the paired contract yet, so a normal catalog remains on the existing single picker
  // until the final inference-main pin makes the capability truthful.
  const scail2MultiReferenceEnabled =
    model === "scail2_14b" && selectedModel?.ui?.scail2MultiReference === true;
  // Runtime-curated selector surface (sc-13800). The API emits only complete encoders the worker can
  // resolve; Video Studio stays adapter-agnostic and future models can expose the same shape.
  const textEncoderOptions = selectedModel?.textEncoderOptions ?? [];
  // LTX-2.5's tier-local Gemma-4 generation encoder is not selectable. Runtime option discovery
  // is shared by the `ltx_video` adapter and can therefore carry LTX-2.3's Gemma-3/amoral choices
  // onto this catalog entry; exposing them would produce a request the 2.5 provider rejects.
  const supportsTextEncoderSelection =
    model !== ltx25VideoModelId && textEncoderOptions.length > 0;
  // The separately downloaded stock Gemma-4 enhancer is an MLX-only 2.5 capability. Keep the
  // simple opt-in checkbox while omitting the unrelated 2.3 encoder picker; off-Mac the Candle
  // provider has no prompt-enhancement input, so the control and payload both disappear. Require
  // a positive host identity: before capabilities load, showing a backend-specific control would
  // risk briefly offering it to an off-Mac client.
  const supportsStockPromptEnhancement =
    model === ltx25VideoModelId && ["macos", "darwin"].includes(macCapabilities?.platform);
  const supportsPromptEnhancement =
    supportsTextEncoderSelection || supportsStockPromptEnhancement;
  const defaultTextEncoderId =
    textEncoderOptions.find((option) => option.isDefault)?.id ??
    textEncoderOptions[0]?.id ??
    legacyDefaultTextEncoderId;
  const selectedTextEncoderModel =
    textEncoderModel == null ||
    (textEncoderModel === legacyDefaultTextEncoderId &&
      !textEncoderOptions.some((option) => option.id === legacyDefaultTextEncoderId))
      ? defaultTextEncoderId
      : textEncoderModel;
  const selectedTextEncoderAvailable = textEncoderOptions.some(
    (option) => option.id === selectedTextEncoderModel,
  );
  // Models gated on the selected tab, not tabs on the selected model (sc-5716). "Serves" is the
  // SHARED `videoModelServesMode` from modelEligibility.js — this screen used to keep a local copy
  // that read the Mac block only, and that is exactly how sc-19570's off-Mac gate got added to the
  // shared predicate (and so to the Simple studio, the screen gate and the download offers) while
  // the Advanced shell kept rendering MLX-only tabs off-Mac. One authority, three layers:
  // declaration + `macVideoModeBlock` + `candleVideoModeBlock`. The mode tabs, the model picker and
  // the snap-on-mode-switch effect all derive from it so the user is never trapped on a mode whose
  // model can't serve the others.
  const macGating = macGatingActive(macCapabilities);
  const candleGating = candleGatingActive(macCapabilities);
  // sc-19570 — NO `macVideoModels.length ? macVideoModels : videoModels` FALLBACK, and its removal
  // is the point rather than a tidy-up. That fallback restored the UNFILTERED catalog precisely when
  // the platform filter had emptied it — i.e. exactly when every installed video model is blocked on
  // this host. `modelReady` read off it, so it stayed `true`, the sc-5947 gate never engaged, and
  // the user got the full Studio with a picker of models that cannot serve any mode: every tab
  // disabled by `modeTabBlocked`, no download offer, and no statement of why. Degraded rather than
  // hung, but it hid the one screen that could fix it.
  //
  // Pre-existing, and it did not fire before this story because `macAvailableModels` alone empties
  // the list only for a Mac user whose every video model is torch-only. `candleAvailableModels`
  // makes it reachable for an ordinary Windows/Linux user with an MLX-only catalog, which is the
  // common case off-Mac.
  //
  // `ImageStudio.jsx:765` is the precedent and settles the semantics: `modelReady =
  // macImageModels.length > 0`, filtered list, no fallback. This is the video twin.
  //
  // Not a behaviour change when no gate is engaged: `macAvailableModels` and `candleAvailableModels`
  // both return the input list unfiltered when their gate is inactive (and before the capabilities
  // endpoint responds), so `macVideoModels === videoModels` and the two expressions are identical.
  // The difference appears only when a gate is active AND has filtered everything out — the case the
  // gate exists for.
  const modelsForMode = (value) =>
    macVideoModels.filter((item) => videoModelServesMode(item, value, macCapabilities));
  // Model-availability gate (sc-5947): when the user has no PLATFORM-available video model at all,
  // show recommended video-model downloads instead of the studio. `ready` matches the picker; offers
  // come from the full catalog via videoModelUsable, recommended-first.
  const modelReady = macVideoModels.length > 0;
  const modelOffers = useMemo(
    () => downloadOffersFor(models, videoModelUsable, macCapabilities),
    [models, macCapabilities],
  );
  const modelDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );
  // Prompt guide for the selected model; fall back to the generic video guide
  // when a model declares none, so the button is always useful (sc-1817).
  const promptGuide = selectedModel?.ui?.promptGuide ?? {
    title: "Video Prompt Guide",
    path: "/prompt-guides/generic-video.md",
  };
  const [duration, setDuration] = useState(saved.duration ?? selectedModel?.defaults?.duration ?? 6);
  const [resolution, setResolution] = useState(saved.resolution ?? selectedModel?.defaults?.resolution ?? "768x512");
  const [fps, setFps] = useState(saved.fps ?? selectedModel?.defaults?.fps ?? 25);
  const [seed, setSeed] = useState(saved.seed ?? "");
  const [negativePrompt, setNegativePrompt] = useState(saved.negativePrompt ?? "");
  // Configurable sampler / scheduler (epic 1753). The Wan diffusers (torch)
  // adapter applies these; MLX-backed video paths advertise default-only via
  // mlx.limits and the picker hides itself there.
  const [sampler, setSampler] = useState(saved.sampler ?? "default");
  const [scheduler, setScheduler] = useState(saved.scheduler ?? "default");
  const [schedulerShift, setSchedulerShift] = useState(saved.schedulerShift ?? 3.0);
  const [stepsOverride, setStepsOverride] = useState(saved.steps ?? "");
  const [guidanceOverride, setGuidanceOverride] = useState(saved.guidanceScale ?? "");
  // Lightning fast-4-step toggle for Wan2.2 A14B MoE (T2V + I2V) — epic 10043, sc-10048.
  // Default ON: the worker (sc-10047) reads `advanced.lightning` and, when on, derives the
  // 4-step / CFG-off distilled recipe; when off it honors the user's steps/guidance (or the
  // native multi-step CFG default). Only the two A14B engines honor it (see showLightning),
  // so the dense 5B and non-Wan models never see the control. Persisted per-workspace.
  const [lightning, setLightning] = useState(saved.lightning ?? true);
  // Which MiniMax-H3 models have already had the default-on turbo variant applied (sc-18727).
  //
  // Turbo IS a LoRA selection — the control and the generic LoRA picker write the same
  // `selectedLoraIds`, which is the only way the two entry points can't disagree (sc-18727's "route
  // both through one resolver"). That makes "default on" a ONE-SHOT seed rather than a default
  // value: without this marker, re-selecting the default every time the studio saw an empty
  // selection would silently undo a deliberate "Off", and undo it identically whether the user
  // turned it off in the turbo control or deselected the adapter in the picker.
  //
  // Persisted in the studio snapshot (mirrored to server ui-preferences, so it survives a desktop
  // relaunch — a localStorage-only marker would re-seed on every launch and re-defeat "Off").
  // Per-model because the two partitions take different adapters.
  const [turboSeededModels, setTurboSeededModels] = useState(saved.turboSeededModels ?? []);
  // LTX-2.3 native guidance knobs (epic 1753 sc-1769). The native ltx-core
  // path has no diffusers scheduler to swap — these three values (cfg + STG +
  // rescale) drive its sealed MultiModalGuiderParams instead.
  const [ltxVideoCfg, setLtxVideoCfg] = useState(saved.videoCfgGuidanceScale ?? "");
  const [ltxVideoStg, setLtxVideoStg] = useState(saved.videoStgGuidanceScale ?? "");
  const [ltxVideoRescale, setLtxVideoRescale] = useState(saved.videoRescaleScale ?? "");
  // Clip-conditioning strengths for the LTX IC-LoRA extend/bridge paths (sc-3522,
  // sc-3755) and for Krea Realtime's video_to_video (sc-8445). The worker reads these from
  // `advanced` (default 1.0 when absent): the source/left clip uses videoConditioningStrength,
  // the bridge right clip uses bridgeRightVideoConditioningStrength.
  const [videoConditioningStrength, setVideoConditioningStrength] = useState(saved.videoConditioningStrength ?? "");
  const [bridgeRightVideoConditioningStrength, setBridgeRightVideoConditioningStrength] = useState(
    saved.bridgeRightVideoConditioningStrength ?? "",
  );
  // Source/reference/character/person-track selections are USER selections, so they persist in
  // the studio snapshot and restore across a full restart (sc-11964). When the snapshot carries a
  // restored source, it WINS at seed time: VideoStudio mounts lazily (keep-alive), usually AFTER
  // App's startup auto-default has already set `selectedAssetId`/`selectedAsset` to the newest
  // asset (App.jsx:1270/768), so seeding from the live `selectedAsset` here would silently clobber
  // the restored source with the newest asset. Only when there is NO restored source at all do we
  // fall back to seeding from the live `selectedAsset` context (the historical behavior). A launch
  // (sendAssetToVideo) still sets the source directly in an effect below, overriding this seed. A
  // one-shot restore-validation effect drops any restored id that no longer resolves once the asset
  // / person-track catalogs land.
  const hasRestoredSource = Boolean(saved.sourceAssetId || saved.sourceClipAssetId);
  const [sourceAssetId, setSourceAssetId] = useState(
    hasRestoredSource
      ? (saved.sourceAssetId ?? "")
      : (["image", "frame"].includes(selectedAsset?.type) ? selectedAsset.id : ""),
  );
  // How the starting image is fitted to the output resolution for the image-conditioned
  // modes (sc-6139), mirroring Image Studio Edit. Crop/Pad only — video has no inpaint
  // mask, so Outpaint is hidden (`inpaintCapable={false}`). Default crop = fill, not stretch.
  const [fitMode, setFitMode] = useState(saved.fitMode ?? "crop");
  const [lastFrameAssetId, setLastFrameAssetId] = useState(saved.lastFrameAssetId ?? "");
  const [sourceClipAssetId, setSourceClipAssetId] = useState(
    hasRestoredSource
      ? (saved.sourceClipAssetId ?? "")
      : (selectedAsset?.type === "video" ? selectedAsset.id : ""),
  );
  const [bridgeRightClipAssetId, setBridgeRightClipAssetId] = useState(saved.bridgeRightClipAssetId ?? "");
  // Subject reference images for Bernini's reference-driven video modes
  // (reference_to_video / reference_video_to_video / ads2v, sc-4703 / sc-5425). 1–N images.
  const [referenceAssetIds, setReferenceAssetIds] = useState(saved.referenceAssetIds ?? []);
  // Multiple source clips for Bernini's multi-source-video edit (multi_video_to_video, sc-5425).
  const [sourceClipAssetIds, setSourceClipAssetIds] = useState(saved.sourceClipAssetIds ?? []);
  // Reference video for Bernini's ads2v mode (sc-5425): a second source clip distinct from the
  // edited source clip (sourceClipAssetId).
  const [referenceClipAssetId, setReferenceClipAssetId] = useState(saved.referenceClipAssetId ?? "");
  // Reference AUDIO clips for a multi-modal reference mode (sc-17160). Held and replayed here so a
  // re-run rebuilds the same conditioning; the picker that populates it is sc-17161's.
  const [referenceAudioAssetIds, setReferenceAudioAssetIds] = useState(
    saved.referenceAudioAssetIds ?? [],
  );
  const [characterId, setCharacterId] = useState(saved.characterId ?? "");
  const [characterLookId, setCharacterLookId] = useState(saved.characterLookId ?? "");
  const [personTrackId, setPersonTrackId] = useState(saved.personTrackId ?? "");
  const [replacementMode, setReplacementMode] = useState(saved.replacementMode ?? "face_only");
  const [selectedDetectionId, setSelectedDetectionId] = useState(saved.selectedDetectionId ?? "");
  const [trackName, setTrackName] = useState(saved.trackName ?? "Selected person");
  const [comparisonMode, setComparisonMode] = useState("side_by_side");
  const [abSide, setAbSide] = useState("replacement");
  const [submitting, setSubmitting] = useState(false);
  const capabilities = selectedModel?.capabilities ?? [];
  const supportsMode = capabilities.includes(mode);
  // GGUF quantization variants the torch adapter can load (sc-1982). Declared in
  // the model manifest's `quantization.variants`; "auto" defers to the worker's
  // per-platform default (Q8_0 on MPS, Q4_K_M on CUDA).
  const quantVariants = Object.entries(selectedModel?.quantization?.variants ?? {});
  const supportsQuantization = quantVariants.length > 0;
  // Lightning is only meaningful for the two Wan2.2 A14B MoE engines (T2V + I2V); the dense
  // 5B and every non-Wan engine ignore `advanced.lightning`, so hide the control there
  // (sc-10048, epic 10043). When it's shown and on, the worker governs steps/guidance with the
  // 4-step recipe, so the manual Steps/Guidance inputs are disabled to reflect that.
  const showLightning = WAN_A14B_LIGHTNING_MODEL_IDS.has(selectedModel?.id);
  const lightningActive = showLightning && lightning;
  // Clip-conditioning strength (`advanced.videoConditioningStrength`). The LTX/Wan IC-LoRA clip
  // modes always honor it, so those two modes show it for whichever model serves them. The
  // `video_to_video` tab is different: the key is honored there ONLY by Krea Realtime, whose v2v
  // drives a strength-controlled autoregressive init (`FewStepSchedule::for_strength`, wired in
  // `video_jobs/krea_realtime.rs::krea_realtime_conditioning`). Bernini — the other v2v model —
  // never reads the key, so gating on the ADAPTER keeps the control off a model that would
  // silently ignore it, the same shape as the `adapter === "ltx_video"` guidance knobs below.
  //
  // Krea's image_to_video deliberately gets NO strength control: an i2v reference still only warms
  // the AR KV cache and the engine reads the image alone, never `strength` — a documented no-op
  // (sc-8440 / sc-8443), pinned worker-side by `krea_realtime_conditioning` emitting
  // `strength: None`. Showing it there would be a knob that does nothing.
  const showClipStrength =
    ["extend_clip", "video_bridge"].includes(mode) ||
    (mode === "video_to_video" && selectedModel?.adapter === "krea_realtime");
  // `quality` (fast/balanced/best) is read by exactly ONE adapter: `svd_steps` (video_jobs/svd.rs)
  // maps it to 15/25/30 inference steps. Every other video engine — LTX, Wan, Bernini, Krea
  // Realtime — never reads the key, so this gates on the ADAPTER for the same reason
  // `showClipStrength` does: a knob that does nothing on the selected model shouldn't be shown
  // (sc-15398). The field itself stays in the payload, the snapshot, and preset defaults —
  // recipe replay round-trips it — so this hides the control, it does not drop the value.
  //
  // Note the manifest `steps[quality]` override `svd_steps` documents is dead: every `steps` in
  // builtin.models.jsonc is a scalar, so the builtin ladder always wins. And an explicit
  // `advanced.steps` beats quality outright, which is why the control says so.
  const showQualitySegment = selectedModel?.adapter === "svd_video";
  // Which generation axes the selected engine actually has, declared in the manifest `video`
  // sub-block (sc-8445) — the video-lane mirror of how Audio Studio reads `audio.supportsGuidance`
  // / `audio.supportsNegativePrompt`. Neither an id check nor an engine link: the catalog says it.
  //
  // ⚠️ ABSENT MEANS TRUE, the opposite polarity to the audio block. Every other video model takes
  // both a CFG scale and a negative prompt and declares nothing, so `!== false` is what keeps them
  // byte-identical to their pre-sc-8445 behaviour; only an engine that genuinely lacks the axis
  // declares it false. Krea Realtime is the one that does: it is CFG-free (Self-Forcing baked
  // guidance out), so `generate_krea_realtime` runs a single batch-1 forward, sets
  // `negative_prompt: None`, and never forwards a guidance scale.
  //
  // HIDDEN, not disabled: the lightning precedent below disables Steps/Guidance because that
  // inertness is TRANSIENT (turn Lightning off and the control works again). A missing axis is
  // permanent for the model, so a forever-dead input is just clutter — Audio Studio hides these two
  // for the same reason (`showGuidance` / `showNegative`).
  // LTX-2.5's distilled path is CFG-free; only the dev transformer consumes these two fields.
  // Other catalog entries retain the manifest-driven absent-means-true contract above.
  const supportsGuidance =
    model === ltx25VideoModelId
      ? ltx25Dev
      : selectedModel?.video?.supportsGuidance !== false;
  const supportsNegativePrompt =
    model === ltx25VideoModelId
      ? ltx25Dev
      : selectedModel?.video?.supportsNegativePrompt !== false;
  // `limits.steps` — the exact set of step counts the model can render (sc-19502). Distilled models
  // bake their sigma waypoints into training: LTX-2.3 runs 8 and nothing else, and BOTH backends now
  // refuse anything off the menu, so an unpinned Steps box here would let the user type a number the
  // enqueue gate 400s on.
  //
  // PINNED (disabled + the value shown), not hidden. The lightning precedent above disables because
  // the inertness is transient, and `supportsGuidance` hides because the axis is missing entirely.
  // This is a third case: the axis is real and the value is worth seeing — the user should know the
  // render is 8 steps — but it is not theirs to move. Hiding it would answer "why is there no Steps
  // control?" with nothing, and leaving it editable would be the silently-ignored knob this story
  // exists to remove.
  const stepsMenu = stepsMenuFromModel(selectedModel);
  const stepsPinnedValue =
    model === ltx25VideoModelId
      ? ltx25Dev
        ? 30
        : 8
      : stepsMenu?.length === 1
        ? stepsMenu[0]
        : null;
  const stepsPinned = stepsPinnedValue !== null;
  // A menu with MORE than one entry is a CHOICE, not a pin — but the gate refuses off-menu counts
  // exactly as hard there, so a free-text box would be the same "UI looser than the gate" desync in
  // a different shape. Render the declared set as a picker, the way `fpsOptions` renders `limits.fps`
  // below. No shipped model declares a multi-entry menu today, so this path is latent; it exists
  // because every OTHER reader on this seam is already set-shaped (`allowed_steps` and
  // `humanized_number_menu` in crates/sceneworks-core/src/video_request.rs, `stepsMenuFromModel`,
  // `checkInMenu` in PresetManagerScreen, gen-core's `supported_steps`) and leaving this one
  // singleton-only would make the studio the single seam that silently reopens the defect.
  const stepsChoice =
    model !== ltx25VideoModelId && stepsMenu !== null && stepsMenu.length > 1
      ? stepsMenu
      : null;
  // The generic LTX-2.3 guider knobs map to MultiModalGuiderParams. LTX-2.5 has its own sealed
  // distilled/dev sampling contracts, so retaining or emitting these values there would be inert.
  const showsLegacyLtxGuidanceControls =
    selectedModel?.adapter === "ltx_video" && model !== ltx25VideoModelId;
  // Whether the override currently held is something the selected model can actually render. A
  // number typed against a PREVIOUS model survives the switch (the same staleness `stepsPinned`
  // suppresses), and a `<select>` whose `value` matches no `<option>` displays its first one — which
  // would quietly assert `limits.steps[0]` is the default. It is not; `defaults.steps` is, and the
  // manifest fixed-point invariant (`shipped_manifest_step_limits_are_what_core_reads`) guarantees
  // that default is itself on the menu. So an off-menu override falls back to the empty
  // "model default" option and, below, is kept out of the payload rather than 400ing.
  const stepsOffMenu =
    stepsChoice !== null && stepsOverride !== "" && !stepsChoice.includes(Number(stepsOverride));
  const implementedMode = [
    "image_to_video",
    "text_to_video",
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
  ].includes(mode);
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
    // Recipe replay seeds the picker wholesale rather than toggling one LoRA at a time (sc-12324).
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
    models: videoModels,
    model,
    setModel,
    fallbackModelId: ltxVideoModelId,
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
    initialGeneralStackIds: saved.generalStackIds ?? [],
  });
  // ── MiniMax-H3 turbo (sc-18726 / sc-18727) ────────────────────────────────────────────────
  //
  // Every value below is derived from the LoRA catalog and the live selection; there is no separate
  // turbo state to keep in sync. `turboVariants` is already narrowed to installed + compatible by
  // `compatibleLoras`, so the control can only ever offer an adapter that would actually enqueue.
  const turboVariants = useMemo(
    () => turboVariantsForModel(selectedModel, compatibleLoras),
    [selectedModel, compatibleLoras],
  );
  const activeTurboVariant = selectedTurboVariant(turboVariants, selectedLoraIds);
  // Shown whenever the model is MiniMax-H3, even with nothing installed: an absent control would
  // answer "why is this render two and a half hours?" with silence. With no variant installed the
  // control renders the reason and the Model Manager pointer instead of an empty menu.
  const showTurbo = modelIsMinimaxH3(selectedModel);
  // Default-on seed. Runs once per model (see `turboSeededModels`) and only once the LoRA catalog
  // has actually resolved — seeding against an empty `turboVariants` during the restart-restore
  // window would mark the model seeded and leave turbo permanently off, the same sc-11962 trap the
  // preset/LoRA prunes above guard.
  useEffect(() => {
    if (!showTurbo || !turboVariants.length) return;
    if (turboSeededModels.includes(model)) return;
    setTurboSeededModels((seeded) => (seeded.includes(model) ? seeded : [...seeded, model]));
    // A variant already selected — a replayed recipe, a restored snapshot, a preset — is the
    // caller's choice and the seed must not stack a SECOND accelerator on top of it. Two adapters
    // asking for different schedules is refused by the worker, so seeding blind here would turn
    // "replay this render" into a hard enqueue failure.
    if (selectedTurboVariant(turboVariants, selectedLoraIds)) return;
    const preferred = defaultTurboVariant(selectedModel, turboVariants);
    if (!preferred) return;
    setSelectedLoraIds((ids) => (ids.includes(preferred.id) ? ids : [...ids, preferred.id]));
  }, [
    showTurbo,
    turboVariants,
    turboSeededModels,
    model,
    selectedModel,
    selectedLoraIds,
    setSelectedLoraIds,
  ]);
  // Selecting a variant REPLACES any other turbo adapter rather than stacking: two accelerators ask
  // for two different schedules and the worker refuses the pair by name
  // (`resolve_turbo_recipe`), so letting the control build that payload would offer a selection that
  // can only fail. Plain (non-accelerator) LoRAs are untouched — a style LoRA rides alongside turbo.
  const selectTurboVariant = useCallback(
    (id) => {
      const turboIds = new Set(turboVariants.map((variant) => variant.id));
      setSelectedLoraIds((ids) => {
        const withoutTurbo = ids.filter((existing) => !turboIds.has(existing));
        return id ? [...withoutTurbo, id] : withoutTurbo;
      });
    },
    [turboVariants, setSelectedLoraIds],
  );
  // Sampler / scheduler menus declared by the model. Video Wan torch
  // declares the full menu; sealed paths (LTX native, MLX) drop to
  // default-only and the picker hides. Gated to the ACTIVE backend (epic 7114 P5):
  // `macGating` is the worker `mlx_required` master switch, so the menu reflects the
  // manifest's `mlx.limits` override on Mac/MLX and `candle.limits` on the candle build.
  const activeBackend = macGating ? "mlx" : "candle";
  // The same hosted q4/q8/bf16 matrices drive both native video backends. Candle Wan tiers must use
  // this explicit picker too: sending the unrelated Torch GGUF `advanced.quantization` field to the
  // Candle worker would otherwise be ignored and silently replaced by its default tier.
  const nativeTierLane =
    activeBackend === "mlx" ||
    (activeBackend === "candle" && candleTierModelIds.has(selectedModel?.id));
  const scail2CandleTierLane =
    activeBackend === "candle" && selectedModel?.id === SCAIL2_MODEL_ID;
  // A Video Studio-only projection: retain the complete catalog object for every non-tier concern,
  // but narrow every tier vocabulary the shared picker understands. The original model object still
  // reaches Model Manager unchanged, so q4/q8 install and repair metadata remain visible there.
  const executionTierModel = useMemo(
    () => videoExecutionTierModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );
  const tierOptions = useMemo(
    () => ({ convRotEligible: false, nvfp4Eligible: false }),
    [],
  );
  const availableTiers = useMemo(
    () => (nativeTierLane ? installedTiers(executionTierModel, tierOptions) : []),
    [nativeTierLane, executionTierModel, tierOptions],
  );
  // The full display set (all possible tiers, installed or not) + the picker option list with
  // un-downloaded tiers disabled — same show-all/disable-unavailable rule as Image Studio. `availableTiers`
  // stays the SELECTABLE/send set. Ordinarily the picker needs more than one possible tier; the
  // explicit SCAIL-2 Candle tier surface remains visible so users can see which shipped packages
  // are admitted even when a particular local install is incomplete.
  const possibleTiers = useMemo(
    () => (nativeTierLane ? allPossibleTiers(executionTierModel, tierOptions) : []),
    [nativeTierLane, executionTierModel, tierOptions],
  );
  const tierPickerItems = useMemo(
    () => (nativeTierLane ? tierPickerOptions(executionTierModel, tierOptions) : []),
    [nativeTierLane, executionTierModel, tierOptions],
  );
  const showTierPicker = useMemo(
    () =>
      nativeTierLane &&
      possibleTiers.length > 0 &&
      availableTiers.length > 0 &&
      (possibleTiers.length > 1 || scail2CandleTierLane),
    [nativeTierLane, possibleTiers, availableTiers, scail2CandleTierLane],
  );
  const baseExecutionTierBlockMessage =
    scail2CandleTierLane && availableTiers.length === 0
      ? "SCAIL-2 Candle generation requires an installed q4, q8, or bf16 tier. Install or repair one in Model Manager."
      : null;
  const hostMemory = useHostMemory();
  const nativeMemoryGb = hostMemoryGbForBackend(hostMemory, activeBackend);
  const autoTier = useMemo(
    () => suggestTier(executionTierModel, nativeMemoryGb, { backend: activeBackend }),
    [executionTierModel, nativeMemoryGb, activeBackend],
  );
  // Seed from the per-(video, model) sticky, then the global quality/Auto policy, clamped to installed.
  // A model transition always re-seeds even when both models happen to expose the same tier list.
  const {
    quantTier,
    setQuantTier,
    tierSwitching,
    handleTierChange,
  } = useQuantTierPicker({
    screen: TIER_SCREEN,
    model,
    selectedModel: executionTierModel,
    availableTiers,
    tierOptions,
    autoTier,
    useGenerationQuality: true,
    reseedOnModelChange: true,
  });
  const availableTierKey = availableTiers.join(",");
  const recipeTierTargetModel = recipeTierRequest
    ? videoModels.find((item) => item.id === recipeTierRequest.modelId)
    : null;
  const recipeTierTargetAvailable = Boolean(
    recipeTierRequest && macVideoModels.some((item) => item.id === recipeTierRequest.modelId),
  );
  // Re-activate a replay target after a transient capability/catalog refresh when it can still
  // serve the current mode, then apply its exact tier. A user mode change that the target cannot
  // serve deliberately keeps the automatic fallback model active; the global replay guard below
  // still refuses submission until the user explicitly starts a new generation.
  useEffect(() => {
    if (
      recipeTierRequest &&
      selectedModel?.id !== recipeTierRequest.modelId &&
      recipeTierTargetAvailable &&
      recipeTierTargetModel &&
      videoModelServesMode(recipeTierTargetModel, mode, macCapabilities)
    ) {
      setModel(recipeTierRequest.modelId);
      return;
    }
    if (
      recipeTierRequest &&
      recipeTierTargetAvailable &&
      recipeTierRequest.modelId === selectedModel?.id &&
      availableTiers.includes(recipeTierRequest.tier)
    ) {
      setQuantTier(recipeTierRequest.tier);
    }
    // The stable key represents exact tier availability; the array identity changes with renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    recipeTierRequest,
    recipeTierTargetModel,
    recipeTierTargetAvailable,
    selectedModel?.id,
    mode,
    activeBackend,
    availableTierKey,
  ]);
  const handleExecutionTierChange = useCallback(
    (nextTier) => {
      setRecipeTierRequest(null);
      handleTierChange(nextTier);
    },
    [handleTierChange],
  );
  const replayTierTargetsActiveModel = Boolean(
    recipeTierRequest && recipeTierRequest.modelId === selectedModel?.id,
  );
  const replayTierTargetName =
    recipeTierTargetModel?.name ?? recipeTierRequest?.modelName ?? recipeTierRequest?.modelId;
  const replayTierBlockMessage = recipeTierRequest
    ? !recipeTierTargetAvailable
      ? `This recipe requires ${replayTierTargetName} at ${tierLabel(recipeTierRequest.tier)}, but that model is not available on the current backend. Generation stays blocked while the recipe replay is active.`
      : !replayTierTargetsActiveModel
        ? `This recipe requires ${replayTierTargetName} at ${tierLabel(recipeTierRequest.tier)}, but ${selectedModel?.name ?? model} is active for the current mode. Generation stays blocked while the recipe replay is active.`
      : !availableTiers.includes(recipeTierRequest.tier)
      ? unavailableRecipeTierMessage(
          selectedModel,
          activeBackend,
          recipeTierRequest.tier,
        )
      : quantTier !== recipeTierRequest.tier
        ? `Preparing this recipe's exact ${tierLabel(recipeTierRequest.tier)} tier. Generation stays blocked until that tier is selected.`
        : null
    : null;
  const executionTierBlockMessage = replayTierBlockMessage ?? baseExecutionTierBlockMessage;
  const canStartNewBf16FromReplay =
    replayTierTargetsActiveModel &&
    scail2CandleTierLane &&
    recipeTierRequest.tier !== "bf16" &&
    !SCAIL2_CANDLE_PRODUCT_TIERS.includes(recipeTierRequest.tier) &&
    availableTiers.includes("bf16");
  const canStartNewGenerationFromReplay = Boolean(
    replayTierBlockMessage &&
      model === selectedModel?.id &&
      macVideoModels.some((item) => item.id === model) &&
      !baseExecutionTierBlockMessage &&
      (!nativeTierLane || availableTiers.includes(quantTier)),
  );
  const startNewGenerationLabel = canStartNewBf16FromReplay
    ? "Use bf16 for a new generation"
    : `Start new generation with ${selectedModel?.name ?? model}${
        nativeTierLane ? ` ${tierLabel(quantTier)}` : ""
      }`;
  const handleStartNewGenerationFromReplay = useCallback(() => {
    // Recheck the same identity/tier contract at the click boundary. During a catalog refresh,
    // `selectedModel` can already derive the fallback while `model` still carries the removed id;
    // clearing replay in that transient render would advertise one model and submit another.
    if (
      !recipeTierRequest ||
      !replayTierBlockMessage ||
      model !== selectedModel?.id ||
      !macVideoModels.some((item) => item.id === model) ||
      baseExecutionTierBlockMessage ||
      (nativeTierLane && !availableTiers.includes(quantTier))
    ) {
      return;
    }
    setRecipeModelNotice("");
    setRecipeTierRequest(null);
    if (nativeTierLane && availableTiers.includes(quantTier)) {
      handleTierChange(quantTier);
    }
  }, [
    recipeTierRequest,
    replayTierBlockMessage,
    model,
    selectedModel?.id,
    macVideoModels,
    baseExecutionTierBlockMessage,
    nativeTierLane,
    availableTiers,
    quantTier,
    handleTierChange,
  ]);
  const showTorchQuantization = activeBackend !== "mlx" && !nativeTierLane && supportsQuantization;
  const selectedTierQuantize =
    nativeTierLane && availableTiers.includes(quantTier) ? tierQuantize(quantTier) : null;
  const tierHasMemoryRisk = showTierPicker && ["q8", "bf16"].includes(quantTier);
  const samplerOptions = useMemo(
    () => samplerOptionsFromModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );
  const schedulerOptions = useMemo(
    () => schedulerOptionsFromModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );
  const showSamplerPicker = samplerOptions.length > 1;
  const showSchedulerPicker = schedulerOptions.length > 1;
  // Guard on a resolved model (sc-11962): before the video catalog loads `samplerOptions`
  // falls back to ["default"], so an un-guarded snap would revert a restored non-default
  // sampler during the restart-restore window and never recover once the catalog lands.
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (samplerOptions.includes(sampler)) {
      return;
    }
    const preferred = samplerOptions.includes(samplerDefaultFromModel(selectedModel))
      ? samplerDefaultFromModel(selectedModel)
      : samplerOptions[0];
    setSampler(preferred);
  }, [samplerOptions, sampler, selectedModel]);
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (schedulerOptions.includes(scheduler)) {
      return;
    }
    const preferred = schedulerOptions.includes(schedulerDefaultFromModel(selectedModel))
      ? schedulerDefaultFromModel(selectedModel)
      : schedulerOptions[0];
    setScheduler(preferred);
  }, [schedulerOptions, scheduler, selectedModel]);
  const requiresLtxIcLora = ltxIcLoraModelIds.has(selectedModel?.id) && ltxIcLoraRequiredModes.has(mode);
  const hasLtxIcLora = selectedLoras.some((lora) => loraLooksLikeIcLora(lora));

  // Sync the source from a genuine USER asset-selection TRANSITION after mount — but NEVER from
  // App's non-user auto-default (sc-11964). App derives `selectedAsset = assets.find(id ===
  // selectedAssetId) ?? assets[0]` (App.jsx:768) and `refreshAssets` auto-selects the newest asset
  // once the catalog lands at STARTUP (`setSelectedAssetId((current) => current ?? defaultAsset.id)`,
  // App.jsx:1270) — regardless of the active view. VideoStudio mounts LAZILY (keep-alive: it only
  // mounts when the user first navigates to it), so it almost always mounts AFTER that startup
  // auto-default has already fired, i.e. with `selectedAssetId` ALREADY set to the newest asset. A
  // plain "does selectedAssetId resolve" gate would then push that newest asset onto a source
  // restored from the snapshot and clobber it (empirically: restored "clip-old" -> "clip-new").
  //
  // Model: track the previously-synced selectedAssetId in a ref and sync ONLY on a real post-mount
  // change (selectedAssetId !== prevRef). When a restored source is present, the auto-default must
  // never count as that transition — whether it is ALREADY present at first mount (late mount, the
  // primary flow) OR arrives after mount while the studio was mounted during the restart window
  // (early mount, selectedAssetId still null at mount). So the ref seeds to a sentinel that absorbs
  // the FIRST resolved selection (the auto-default) exactly once, then tracks transitions normally.
  // With NO restored source the ref seeds to null, so the first resolved selection IS a transition
  // and the source defaults to the selected asset exactly as before. A launch (sendAssetToVideo)
  // sets the source directly in the effect below, independent of this sync.
  const AUTO_DEFAULT_PENDING = undefined;
  const prevSelectedAssetIdRef = useRef(hasRestoredSource ? AUTO_DEFAULT_PENDING : null);
  useEffect(() => {
    // Wait for the selection to resolve to a real asset before treating it as a transition; a
    // selectedAssetId whose asset the catalog hasn't landed yet is not yet a user-visible pick.
    if (!selectedAssetId || selectedAsset?.id !== selectedAssetId) {
      return;
    }
    const prevSelectedAssetId = prevSelectedAssetIdRef.current;
    prevSelectedAssetIdRef.current = selectedAssetId;
    // Absorb the first resolved selection (the restart auto-default) once when a restored source
    // is present, so navigating INTO Video Studio can't clobber it. Also a no-op when the value
    // hasn't actually changed since the last sync.
    if (prevSelectedAssetId === AUTO_DEFAULT_PENDING || selectedAssetId === prevSelectedAssetId) {
      return;
    }
    if (selectedAsset.type === "image" || selectedAsset.type === "frame") {
      setSourceAssetId(selectedAsset.id);
    }
    if (selectedAsset.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
    // Asset object refreshes must not replay a user-selection transition.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAssetId, selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (launchRequest?.view !== "Video") {
      return;
    }
    // A recipe launch is owned by the recipe effect below. It carries an `assetId` for selection
    // context but no `mode`, so without this it would fall through to the asset path and both
    // setMode(undefined) and adopt the replayed clip as its own source clip (sc-12324).
    if (launchRequest.recipe) {
      return;
    }
    // sc-10516: a preset launch. `availablePresets` filters on mode + model, so the
    // preset only resolves once both match — set them alongside the id, then let
    // useSavePreset's hydrate effect apply its `defaults`. Returns before the asset
    // paths below, which would otherwise fall through to setMode(undefined).
    if (launchRequest.presetId) {
      if (VIDEO_PRESET_MODES.includes(launchRequest.presetMode)) {
        setMode(launchRequest.presetMode);
      }
      if (launchRequest.presetModel) {
        setModel(launchRequest.presetModel);
      }
      setSelectedPresetId(launchRequest.presetId);
      return;
    }
    if (launchRequest.characterId) {
      setMode(launchRequest.mode ?? "text_to_video");
      setCharacterId(launchRequest.characterId);
      setCharacterLookId(launchRequest.lookId ?? "");
      return;
    }
    if (launchRequest.assetId !== selectedAsset?.id) {
      return;
    }
    setMode(launchRequest.mode);
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "image" || selectedAsset?.type === "frame") {
      setSourceAssetId(selectedAsset.id);
    }
    // A persistent launch is applied once per launch/asset identity, not on catalog object refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [launchRequest?.id, selectedAsset?.id, selectedAsset?.type]);

  // Restore-time validation of the persisted asset selections (sc-11964). The snapshot seeds
  // sourceAssetId / referenceAssetIds / etc. at mount, but the asset catalog resolves
  // asynchronously after a restart. Once it first lands, drop any restored id that no longer
  // resolves to a real asset so we never carry a dangling reference to a deleted one. A ref
  // latches on the first non-empty catalog so this validates only the RESTORED values — it
  // won't fight a later user selection whose freshly generated asset the catalog hasn't caught
  // up to yet.
  const restoredAssetsValidatedRef = useRef(false);
  useEffect(() => {
    if (restoredAssetsValidatedRef.current || assets.length === 0) {
      return;
    }
    restoredAssetsValidatedRef.current = true;
    const assetExists = (id) => assets.some((asset) => asset.id === id);
    const dropMissing = (setter) => setter((current) => (current && assetExists(current) ? current : ""));
    dropMissing(setSourceAssetId);
    dropMissing(setLastFrameAssetId);
    dropMissing(setSourceClipAssetId);
    dropMissing(setBridgeRightClipAssetId);
    dropMissing(setReferenceClipAssetId);
    setReferenceAssetIds((current) => (current.some((id) => !assetExists(id)) ? current.filter(assetExists) : current));
    setSourceClipAssetIds((current) => (current.some((id) => !assetExists(id)) ? current.filter(assetExists) : current));
  }, [assets]);

  // Same restore-time validation for the restored person-track selection (sc-11964): once the
  // person-track catalog first lands, drop personTrackId if it no longer resolves to a real track.
  const restoredPersonTrackValidatedRef = useRef(false);
  useEffect(() => {
    if (restoredPersonTrackValidatedRef.current || personTracks.length === 0) {
      return;
    }
    restoredPersonTrackValidatedRef.current = true;
    setPersonTrackId((current) => (current && personTracks.some((track) => track.id === current) ? current : ""));
  }, [personTracks]);

  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    setDuration((current) => {
      const options = selectedModel.limits?.durations ?? [4, 6, 8, 10];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.duration ?? options[0];
    });
    setResolution((current) => {
      const options = selectedModel.limits?.resolutions ?? ["768x512"];
      return options.includes(current) ? current : selectedModel.defaults?.resolution ?? options[0];
    });
    setFps((current) => {
      const options = selectedModel.limits?.fps ?? [24, 25, 30];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.fps ?? options[0];
    });
    // Reconcile defaults only when model identity changes, not on catalog object refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel?.id]);

  // I2V: when the user picks a source image (or first/last frame) after mount,
  // snap resolution to whichever option in the model's list best matches the
  // image's aspect ratio. The ref tracks the last-seen id so polling-driven
  // assets refreshes don't re-fire, and so the saved snapshot's resolution is
  // preserved when the asset id is just being restored on mount.
  const i2vSourceAssetId = sourceAssetId || lastFrameAssetId;
  const lastI2vAssetIdRef = useRef(i2vSourceAssetId);
  useEffect(() => {
    if (i2vSourceAssetId === lastI2vAssetIdRef.current) {
      return;
    }
    lastI2vAssetIdRef.current = i2vSourceAssetId;
    if (!i2vSourceAssetId) return;
    if (!["image_to_video", "first_last_frame"].includes(mode)) return;
    const asset = assets.find((item) => item.id === i2vSourceAssetId);
    const width = asset?.file?.width;
    const height = asset?.file?.height;
    if (!width || !height) return;
    const match = pickClosestResolution(width, height, selectedModel?.limits?.resolutions);
    if (match) setResolution(match);
    // Source/model identities drive this probe; an unrelated asset catalog refresh does not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [i2vSourceAssetId, mode, selectedModel?.id, assets]);

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
      return;
    }
    const recipe = launchRequest.recipe;
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
    setStepsOverride(rawSettings.steps ?? "");
    setGuidanceOverride(rawSettings.guidanceScale ?? "");
    setSampler(rawSettings.sampler ?? "default");
    setScheduler(rawSettings.scheduler ?? "default");
    setSchedulerShift(rawSettings.schedulerShift ?? 3.0);
    setMotion(rawSettings.motion ?? DEFAULT_MOTION);
    setLtxPipeline(rawSettings.ltxPipeline ?? "auto");
    setDistilledVariant(rawSettings.distilledVariant ?? "1.1");
    setTransformerVariant(rawSettings.transformerVariant ?? "distilled");
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

  // Models are gated on the selected tab (sc-5716): when the active mode isn't served by the current
  // model, snap to the first model that serves it so the user can always leave a mode. Generalizes
  // the old per-mode snaps (replace_person → first replace-capable model; animate_character →
  // scail2_14b) to every mode, including the Bernini editing/reference modes. A no-op when the
  // current model already serves the mode (e.g. an LTX image_to_video → text_to_video switch) or
  // when no model serves it (a reduced catalog) — there's nothing to snap to.
  useEffect(() => {
    if (videoModelServesMode(selectedModel, mode, macCapabilities)) {
      return;
    }
    const fallback = modelsForMode(mode)[0];
    if (fallback && fallback.id !== model) {
      setModel(fallback.id);
    }
    // videoModelServesMode / modelsForMode close over videoModels + macCapabilities, captured below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, model, selectedModel, videoModels, macCapabilities]);

  // Save-as-Preset + the preset-default hydrate pass (sc-8937 — shared with the Image
  // studio via useSavePreset). The [key, setter] pairs are restored through the
  // remember/clear snapshot machinery, so switching to None (or another preset) puts
  // the user's prior value back. Only keys the preset carries are applied, so older
  // presets keep working and full-snapshot presets restore the prompt, cfg, sampler,
  // and the native LTX guidance knobs. The model is intentionally absent — presets
  // never switch the model.
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
      ["duration", setDuration],
      ["fps", setFps],
      ["quality", setQuality],
      ["guidanceScale", setGuidanceOverride],
      ["steps", setStepsOverride],
      ["sampler", setSampler],
      ["scheduler", setScheduler],
      ["schedulerShift", setSchedulerShift],
      ["precision", setPrecision],
      ["quantization", setQuantization],
      ["ltxPipeline", setLtxPipeline],
      ["distilledVariant", setDistilledVariant],
      ["transformerVariant", setTransformerVariant],
      ["enhancePrompt", setEnhancePrompt],
      ["textEncoderModel", setTextEncoderModel],
      ["motion", setMotion],
      ["videoCfgGuidanceScale", setLtxVideoCfg],
      ["videoStgGuidanceScale", setLtxVideoStg],
      ["videoRescaleScale", setLtxVideoRescale],
    ],
    // Restore the saved sub-mode ("type") when it's a generatable video workflow.
    modeIsPresetable: (savedMode) => VIDEO_PRESET_MODES.includes(savedMode),
    // Video gates saving to the presetable modes; blocks the rest with a message.
    extraSaveGuard: () =>
      VIDEO_PRESET_MODES.includes(mode)
        ? null
        : "Switch to Image, Text, or First/Last mode to save a preset.",
    buildDefaults: () => ({
      prompt,
      negativePrompt,
      resolution,
      duration,
      fps,
      quality,
      mode,
      guidanceScale: finiteNumberOrUndefined(guidanceOverride),
      steps: finiteNumberOrUndefined(stepsOverride),
      sampler,
      scheduler,
      schedulerShift,
      precision,
      quantization,
      ltxPipeline,
      distilledVariant,
      transformerVariant,
      ...(supportsPromptEnhancement ? { enhancePrompt } : {}),
      ...(supportsTextEncoderSelection ? { textEncoderModel: selectedTextEncoderModel } : {}),
      motion,
      ...(showsLegacyLtxGuidanceControls
        ? {
            videoCfgGuidanceScale: finiteNumberOrUndefined(ltxVideoCfg),
            videoStgGuidanceScale: finiteNumberOrUndefined(ltxVideoStg),
            videoRescaleScale: finiteNumberOrUndefined(ltxVideoRescale),
          }
        : {}),
    }),
  });

  useStudioSettingsWriter("video", activeProject?.id ?? null, {
    motion,
    mode,
    prompt,
    styleId,
    quality,
    ltxPipeline,
    distilledVariant,
    transformerVariant,
    precision,
    enhancePrompt,
    textEncoderModel,
    quantization,
    advancedOpen,
    selectedLoraIds,
    loraWeights,
    model,
    duration,
    resolution,
    fps,
    seed,
    negativePrompt,
    selectedPresetId,
    generalStackIds,
    sampler,
    scheduler,
    schedulerShift,
    steps: stepsOverride,
    guidanceScale: guidanceOverride,
    lightning,
    // The MiniMax-H3 turbo default-on one-shot marker (sc-18727). The SELECTION itself already
    // persists through `selectedLoraIds`; this records only that the seed has fired, so a
    // deliberate "Off" is not re-seeded on the next mount or the next relaunch.
    turboSeededModels,
    videoCfgGuidanceScale: ltxVideoCfg,
    videoStgGuidanceScale: ltxVideoStg,
    videoRescaleScale: ltxVideoRescale,
    videoConditioningStrength,
    bridgeRightVideoConditioningStrength,
    fitMode,
    // User asset/reference/character/person-track selections (sc-11964). These are USER choices,
    // not model defaults, so they persist here and restore across a full restart — kept out of the
    // defaults-reset path. Restore-validation (above) drops any id whose asset/track is gone.
    sourceAssetId,
    lastFrameAssetId,
    sourceClipAssetId,
    bridgeRightClipAssetId,
    referenceAssetIds,
    referenceAudioAssetIds,
    sourceClipAssetIds,
    referenceClipAssetId,
    characterId,
    characterLookId,
    personTrackId,
    replacementMode,
    selectedDetectionId,
    trackName,
  },
  // Suppress the live writer until the video catalog has loaded (sc-11962), so a transient
  // defaults-reset during the restart-restore/settle window can't overwrite the restored
  // snapshot before the async catalogs settle.
  // Plus ui-preferences hydration (sc-15425) — see the same gate in ImageStudio.
  preferencesHydrated && videoModels.length > 0);

  useEffect(() => {
    if (mode !== "replace_person") {
      return;
    }
    const firstMatchingTrack = personTracks.find((track) => track.sourceAssetId === sourceClipAssetId);
    if (firstMatchingTrack && !personTracks.some((track) => track.id === personTrackId)) {
      setPersonTrackId(firstMatchingTrack.id);
    }
  }, [mode, personTracks, personTrackId, sourceClipAssetId]);

  const modeOptions = [
    // Text→Video first, mirroring Image Studio (Text → Image first) and the default mode (sc-5716).
    ["text_to_video", "Text → Video"],
    ["image_to_video", "Image → Video"],
    ["first_last_frame", "First → Last"],
    ["extend_clip", "Extend"],
    ["video_bridge", "Bridge"],
    ["replace_person", "Replace person"],
    // Bernini planner editing / reference-driven video modes (sc-4703) + multi-source
    // modes (sc-5425). Enabled only on models whose capabilities include them (today:
    // Bernini); disabled elsewhere, the same per-model gating as Replace person / the
    // LTX clip modes.
    ["video_to_video", "Video → Video"],
    ["reference_to_video", "Reference → Video"],
    ["reference_video_to_video", "Reference + Video"],
    ["multi_video_to_video", "Multi-Clip → Video"],
    ["ads2v", "Clip + Ref Video"],
    // SCAIL-2 character animation (epic 5439 / sc-5449): a reference character image + a driving
    // video → the character animated with the driving motion. Enabled only on the model whose
    // capabilities include it (today: scail2_14b); the same per-model gating as the others.
    ["animate_character", "Animate character"],
  ];
  // Platform UI gating (sc-3486, sc-3773, sc-5716, sc-19570): mode tabs are gated at the MODE
  // level, not on the selected model. A tab is disabled only under active platform gating when NO
  // available model serves the mode (mode-level availability across `macVideoModels`) — never on
  // the selected model's `videoModes`, which used to trap the user on replace_person /
  // animate_character with no way back. The active tab is always left enabled so a reduced catalog
  // can't strand you on a disabled tab; the per-mode block still gates the in-mode model picker +
  // submit (via `modelsForMode` / `supportsMode`).
  //
  // sc-19570 — PLATFORM-AGNOSTIC. This read `macGating &&`, which is false off-Mac, so every
  // MLX-only tab (LTX 2.3's Image→Video / First-Last-Frame / Extend / Bridge / Replace) stayed
  // enabled on Windows/Linux and the user only learned at Generate. Either gate disables the tab now.
  //
  // The two are mutually exclusive, so AT MOST one is ever true — not "exactly one", which this
  // comment used to claim and which is false in two ordinary states. `candleGatingActive` is
  // `!is_mac` and `macGatingActive` is the SCENEWORKS_MLX_REQUIRED rollout flag, so BOTH are false
  // (a) before `GET /api/v1/capabilities/mac` responds, since `DEFAULT_MAC_CAPABILITIES` sets both
  // false, and (b) permanently on a Mac still in observe mode. In both windows this predicate is
  // inert and the tab list falls back to the manifest declaration alone. That is deliberate — a
  // client that has not yet been told the platform must not invent a gate — but it means the
  // declaration IS the whole answer there, and copy that says otherwise misdescribes the screen.
  const modeTabBlocked = (value) =>
    (macGating || candleGating) && modelsForMode(value).length === 0;
  // The tab tooltip names the platform that is doing the gating — off-Mac the honest sentence is
  // the inverse of the Mac one (the pair works, just not here).
  const modeTabBlockedText = candleGating
    ? "No installed model supports this mode on this platform (macOS/MLX only)."
    : "No installed model supports this mode on macOS.";
  const matchingTracks = useMemo(
    () =>
      mode === "replace_person"
        ? personTracks.filter((track) => track.sourceAssetId === sourceClipAssetId)
        : [],
    [mode, personTracks, sourceClipAssetId],
  );
  const latestDetectionJob = useMemo(() => {
    if (mode !== "replace_person" || !activeProject?.id || !sourceClipAssetId) {
      return null;
    }
    let latest = null;
    for (const job of jobs) {
      if (
        job.type === "person_detect" &&
        job.status === "completed" &&
        job.projectId === activeProject.id &&
        job.payload?.sourceAssetId === sourceClipAssetId &&
        (!latest || job.createdAt.localeCompare(latest.createdAt) > 0)
      ) {
        latest = job;
      }
    }
    return latest;
  }, [activeProject?.id, jobs, mode, sourceClipAssetId]);
  const detectionResult = latestDetectionJob?.result ?? null;
  const representativeFrame = assets.find((asset) => asset.id === detectionResult?.frameAssetId);
  const selectedDetection = detectionResult?.detections?.find((item) => item.id === selectedDetectionId) ?? detectionResult?.detections?.[0];
  const selectedTrack = personTracks.find((track) => track.id === personTrackId);
  const comparisonAsset = latestAssets.find((asset) => asset.recipe?.mode === "replace_person");
  const comparisonSource = assets.find((asset) => asset.id === comparisonAsset?.lineage?.sourceClipAssetId);
  // Per-model reference-media caps (sc-17160), read at the FORM (sc-17161). Before this the four
  // caps had Rust readers only, so an over-selection was submittable and came back as a 400.
  const refCaps = useMemo(() => referenceCaps(selectedModel), [selectedModel]);
  // Which reference pickers a MODE + MODEL pair can actually feed. `reference_to_video` was the
  // only mode serving multi-modal references when MiniMax-H3 Ref2VA arrived; the audio picker keys
  // on the declared cap alone (default 0) because no other model takes audio references, and the
  // clip picker keys on the cap being DECLARED, not merely non-zero — the blanket default is 8, so
  // a value check would offer reference clips on Bernini's r2v, whose engine encodes images only.
  const showAudioReferences = mode === "reference_to_video" && refCaps.audio > 0;
  const showReferenceClips = mode === "reference_to_video" && refCaps.clipsDeclared && refCaps.clips > 0;
  // The reference media this mode will actually SEND. The gate counts these rather than the raw
  // state so the client refusal and the payload can never disagree about what was selected — the
  // same single-expression rule `videoStudioValidation.js` exists to keep.
  const outgoingReferenceAssetIds = [
    "reference_to_video",
    "reference_video_to_video",
    "ads2v",
    "animate_character",
  ].includes(mode)
    ? referenceAssetIds
    : [];
  const outgoingSourceClipAssetIds =
    mode === "multi_video_to_video" || showReferenceClips ? sourceClipAssetIds : [];
  // The audio references the selected MODEL will carry, gated on its declared
  // `limits.maxReferenceAudioAssets` (default 0) rather than a hardcoded mode list — which modes
  // take audio references is a model fact, which is why the payload slot was never mode-gated.
  // Before this the payload sent the raw state, so a selection made on Ref2VA rode along into a
  // job for a model that declares no audio cap at all.
  const outgoingReferenceAudioAssetIds = refCaps.audio > 0 ? referenceAudioAssetIds : [];
  const referenceLimitMessage = referenceLimitError({
    modelName: selectedModel?.name,
    caps: refCaps,
    images: outgoingReferenceAssetIds.length,
    clips: outgoingSourceClipAssetIds.length,
    // The count the visible PICKER can change, not the raw state. `referenceAudioAssetIds` outlives
    // the picker — it is persisted per project and restored on mount — so counting it raw meant
    // selecting one audio reference on Ref2VA and then switching mode or model disabled Generate
    // with "…but 1 are selected. Remove them", while the only control that could remove them had
    // just unmounted. That refusal was unclearable, and it survived a restart. A cap the form on
    // screen cannot violate must not be able to refuse.
    audio: showAudioReferences ? referenceAudioAssetIds.length : 0,
  });
  // sc-19574 — the audio-only Ref2VA shape, named so the Generate gate can SAY why rather than
  // leaving the user to infer it from an empty image zone next to a full audio one. Counted off the
  // outgoing lists so a stale audio selection on a model that declares no audio cap can't raise it.
  const audioOnlyReferenceSet =
    mode === "reference_to_video" &&
    outgoingReferenceAudioAssetIds.length > 0 &&
    referenceAssetIds.length === 0 &&
    outgoingSourceClipAssetIds.length === 0;
  const hasInputs =
    mode === "text_to_video" ||
    (mode === "image_to_video" && sourceAssetId) ||
    (mode === "first_last_frame" && sourceAssetId && lastFrameAssetId) ||
    (mode === "extend_clip" && sourceClipAssetId) ||
    (mode === "video_bridge" && sourceClipAssetId && bridgeRightClipAssetId) ||
    (mode === "replace_person" && sourceClipAssetId && personTrackId && characterId) ||
    // Bernini editing / reference-driven modes (sc-4703).
    (mode === "video_to_video" && sourceClipAssetId) ||
    // `reference_to_video` needs at least one VISUAL reference — an image or a video clip. Audio
    // references ride along and can never be the only one.
    //
    // sc-17159 widened this from images-alone because MiniMax-H3 Ref2VA also takes clips and audio;
    // the clip half was right and the audio half was not. sc-19574 settled it against the reference
    // implementation: diffusers `MiniMaxH3` refuses `set(kinds) == {"audio"}` outright — an audio
    // reference never reaches the conditioner, so an audio-only set leaves the visual stream
    // unconditioned — and the worker refuses it too (sc-19508). Enabling Generate for a shape three
    // layers down rejects is exactly the "the product offers it and then says no" gap this closes;
    // `validate_video_job` now 400s the same shape with the same rule.
    //
    // Bernini is unaffected: it declares no clip or audio caps, so `outgoingSourceClipAssetIds` is
    // empty there and this stays "needs an image".
    (mode === "reference_to_video" &&
      (referenceAssetIds.length > 0 || outgoingSourceClipAssetIds.length > 0)) ||
    (mode === "reference_video_to_video" && sourceClipAssetId && referenceAssetIds.length > 0) ||
    // Bernini multi-source modes (sc-5425): mv2v needs >=2 clips; ads2v needs a source
    // clip, a reference video, and >=1 reference image.
    (mode === "multi_video_to_video" && sourceClipAssetIds.length >= 2) ||
    (mode === "ads2v" && sourceClipAssetId && referenceClipAssetId && referenceAssetIds.length > 0) ||
    // SCAIL-2 character animation (sc-5449): a driving video + a reference character image.
    // Once the paired multi-reference descriptor is live, its source-position table admits 1–6
    // ordered character images. Keep all seven in state to show a rejection; never truncate one.
    (mode === "animate_character" &&
      sourceClipAssetId &&
      referenceAssetIds.length > 0 &&
      (!scail2MultiReferenceEnabled || referenceAssetIds.length <= MAX_SCAIL2_REFERENCE_CHARACTERS));
  const scail2ReferenceOverflow =
    mode === "animate_character" &&
    scail2MultiReferenceEnabled &&
    referenceAssetIds.length > MAX_SCAIL2_REFERENCE_CHARACTERS;
  // Don't let Replace Person queue a job the readiness endpoint says no live
  // worker can run — that would sit unclaimable instead of honoring the gate.
  const replaceReady = mode !== "replace_person" || personReadiness?.replace?.ready !== false;
  // Image-conditioned models (e.g. Stable Video Diffusion) take no text prompt;
  // they animate the source image, so don't gate submission on prompt text.
  const promptless = Boolean(selectedModel?.promptless);
  const [width, height] = resolution.split("x").map((value) => Number(value));
  const durationOptions = selectedModel?.limits?.durations ?? [4, 6, 8, 10];
  const resolutionOptions = useMemo(
    () =>
      selectedModel?.limits?.resolutions ?? [
        "768x512",
        "640x640",
        "1280x720",
        "720x1280",
      ],
    [selectedModel?.limits?.resolutions],
  );

  // Effective inputs once the general-preset stack folds in (epic 11949); drives the live
  // preview and (Phase 5) the client-authoritative submit.
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
  const stackActive = generalStack.length > 0;

  // Style Catalog axis (sc-13136): mirror the Image Studio's Style picker into the Video Studio.
  // composeStyledPrompt wraps the outgoing prompt LAST — AFTER the general-preset stack fold above —
  // so the style's Subject:/Style: block sits around the already-preset-composed prompt. Video
  // has no structured-caption or batch modes (the two the image lane gates the composer off for), so
  // the exclusions here are a promptless (image-conditioned) model, which sends no text prompt to
  // wrap, and a booru-tag model, whose comma-separated-tag convention the catalog's prose entries do
  // not fit (mirrors the Image Studio gate). No video model declares `captionStyle: "tags"` today, so
  // the tag arm is inert — it is here so a future tag-convention video model inherits the gate rather
  // than silently acquiring a mismatched axis. `stylePromptBase` is the SAME base the styleless submit
  // sends, so the live preview and the submit compose from one string and can never drift.
  const styleAxisAvailable = !promptless && selectedModel?.captionStyle !== "tags";
  const activeStyleText = styleTextForId(styleAxisAvailable ? styleId : null);
  const styleApplied = styleAxisAvailable && typeof activeStyleText === "string" && activeStyleText.trim() !== "";
  const stylePromptBase = stackActive ? composedStack.prompt : prompt;
  const composedStylePrompt = styleApplied
    ? composeStyledPrompt({ styleText: activeStyleText, userPrompt: stylePromptBase })
    : null;

  // One summary gates Generate and carries every reason it might be dead, so the button
  // and the messages can't drift — the bug this screen used to embody, where `canSubmit`
  // and `blockedMessage` re-derived the same rules side by side (epic 10644).
  const videoDraft = useMemo(
    () => ({
      activeProject,
      promptless,
      prompt,
      // sc-13136: measure the COMPOSED outgoing prompt against the backend cap, but only when a
      // style is active (styleless behavior unchanged). `composedStylePrompt` is the exact string
      // submitted, so the readout and the blocking Generate error can never disagree.
      styleActive: styleApplied,
      composedPrompt: composedStylePrompt ?? "",
      supportsMode,
      implementedMode,
      hasInputs,
      requiresLtxIcLora,
      hasLtxIcLora,
      replaceReady,
      scail2ReferenceOverflow,
      referenceLimitMessage,
      audioOnlyReferenceSet,
      executionTierBlockMessage,
      modelName: selectedModel?.name,
      presetMissing: presetValidationResult.missing,
      presetIncompatible: presetValidationResult.incompatible,
      loraIncompatible: selectedLoraValidationResult.incompatible,
    }),
    [
      activeProject,
      promptless,
      prompt,
      styleApplied,
      composedStylePrompt,
      supportsMode,
      implementedMode,
      hasInputs,
      requiresLtxIcLora,
      hasLtxIcLora,
      replaceReady,
      scail2ReferenceOverflow,
      referenceLimitMessage,
      audioOnlyReferenceSet,
      executionTierBlockMessage,
      selectedModel,
      presetValidationResult,
      selectedLoraValidationResult,
    ],
  );
  const videoValidity = useValidation(videoGenerateValidation, videoDraft, undefined);
  const stackAddsNegative = generalStack.some((preset) => Boolean(preset?.defaults?.negativePrompt));
  const stackAddsCount = generalStack.some((preset) => Number.isFinite(Number(preset?.defaults?.count)));
  const fpsOptions = selectedModel?.limits?.fps ?? [24, 25, 30];
  // `limits.hardMinSteps` (sc-19426) had no web reader — the Steps input hardcoded min="1".
  const minSteps = minStepsForModel(selectedModel);
  const durationHint =
    selectedModel?.ui?.durationHint ??
    (selectedModel?.limits?.recommendedMaxDuration ? `Recommended: ${selectedModel.limits.recommendedMaxDuration}s or less.` : "");
  const replacementModeLabels = {
    face_only: "Face Only",
    full_person_keep_outfit: "Full Person, Keep Outfit",
    full_person_replace_outfit: "Full Person, Replace Outfit",
  };

  async function submit(event) {
    event.preventDefault();
    // The button and validation summary already expose this refusal. Keep the submit boundary exact
    // as well so a direct/stale form submit cannot enqueue a SCAIL-2 Candle request that no product
    // tier is allowed to claim.
    if (submitting || executionTierBlockMessage) {
      return;
    }
    setSubmitting(true);
    try {
      // Fold the general-preset stack (epic 11949): send the composed prompt + negative and,
      // when a general sets aspect, the snapped resolution. The client is authoritative for the
      // composed prompt, so presetPromptResolvedClientSide tells the server to skip its fold.
      // (Video has no `count` field, so the stack's variations don't apply here.)
      const stackResolution = stackActive && composedStack.resolution ? parseResolution(composedStack.resolution) : null;
      const job = await createVideoJob({
        mode,
        // sc-13136: when a Style Catalog entry is active the composer has already wrapped the
        // (stack-folded) prompt LAST — send that composed string verbatim. Falls back to the plain
        // stack-folded / raw prompt when no style is applied.
        prompt: styleApplied ? composedStylePrompt : stackActive ? composedStack.prompt : prompt,
        // An engine with no negative-prompt axis gets an empty one rather than whatever a preset
        // stack composed — the field is hidden for it, so sending text the user cannot see or edit
        // (and the worker discards anyway) would be a ghost input on the recipe.
        negativePrompt: supportsNegativePrompt ? (stackActive ? composedStack.negativePrompt : negativePrompt) : "",
        model,
        duration: Number(duration),
        fps: Number(fps),
        width: stackResolution?.width ?? width,
        height: stackResolution?.height ?? height,
        quality,
        seed: seed === "" ? null : Number(seed),
        recipePresetId: selectedPreset?.id ?? null,
        // The studio seeds a selected preset's LoRAs into the visible `loras` (generationStudio's
        // preset-LoRA seed effect), so the client is authoritative for preset LoRAs — tell the
        // server to skip its own merge so edits/removals stick. Parity with the Image Studio.
        presetLorasResolvedClientSide: selectedPreset ? true : undefined,
        // The client composed the prompt (preset stack and/or Style Catalog wrap), so tell the
        // server to skip its own fold and send the composed string as-is (sc-13136).
        presetPromptResolvedClientSide: stackActive || styleApplied || undefined,
        characterId: characterId || null,
        characterLookId: characterLookId || null,
        sourceAssetId: ["image_to_video", "first_last_frame"].includes(mode) ? sourceAssetId || null : null,
        // Crop/Pad fit for the starting image (sc-6139) — only the image-conditioned
        // modes carry it; `effectiveFitMode(_, false)` coerces any stale outpaint back to
        // crop (video has no inpaint mask). Other modes omit it (DTO defaults to crop).
        fitMode: ["image_to_video", "first_last_frame"].includes(mode)
          ? effectiveFitMode(fitMode, false)
          : undefined,
        lastFrameAssetId: mode === "first_last_frame" ? lastFrameAssetId || null : null,
        sourceClipAssetId: [
          "extend_clip",
          "replace_person",
          "video_bridge",
          "video_to_video",
          "reference_video_to_video",
          "ads2v",
          // SCAIL-2 character animation (sc-5449): the driving video.
          "animate_character",
        ].includes(mode)
          ? sourceClipAssetId || null
          : null,
        // Bernini multi-source clips (sc-5425) — mv2v carries the array, and so does a
        // reference→video on a model that DECLARES a reference-clip cap (MiniMax-H3 Ref2VA takes
        // up to 3 clips alongside its images and audio, sc-17160).
        sourceClipAssetIds: outgoingSourceClipAssetIds,
        bridgeRightClipAssetId: mode === "video_bridge" ? bridgeRightClipAssetId || null : null,
        // Bernini subject references (sc-4703 / sc-5425) — the reference-driven modes + ads2v carry
        // them; SCAIL-2 character animation (sc-5449) carries the reference character image.
        referenceAssetIds: outgoingReferenceAssetIds,
        // Reference AUDIO clips (sc-17160). Gated on the MODEL's declared cap, not on a hardcoded
        // mode list like the two above: which modes take audio references is a model fact,
        // declared as `limits.maxReferenceAudioAssets` and enforced server-side, and a model that
        // takes none refuses a non-empty list at enqueue. Sending the raw state meant a selection
        // made on Ref2VA rode into a job for a model with no audio cap once the picker unmounted.
        referenceAudioAssetIds: outgoingReferenceAudioAssetIds,
        // Bernini ads2v reference video (sc-5425).
        referenceClipAssetId: mode === "ads2v" ? referenceClipAssetId || null : null,
        personTrackId: mode === "replace_person" ? personTrackId || null : null,
        // Gated on the MODEL as well as the mode (sc-20262), for the same reason
        // `referenceAudioAssetIds` above is: hiding a control does not unset the state behind it,
        // so a mode picked on a Wan-VACE engine would otherwise ride into a SCAIL-2 job once the
        // control unmounted — and SCAIL-2's engine refuses a non-default mode. `replacementMode`
        // is only meaningful where `replacementModeApplies`.
        replacementMode:
          mode === "replace_person" && replacementModeApplies(model) ? replacementMode : "face_only",
        loras: selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
        advanced: {
          resolution,
          durationHint,
          motion,
          selectedPersonTrack: selectedTrack ?? null,
          // The recipe's human-readable echo of the field above, so it follows the same gate — a
          // replayed SCAIL-2 recipe must not display a Replacement mode the job never carried.
          replacementModeLabel:
            replacementModeLabels[
              replacementModeApplies(model) ? replacementMode : "face_only"
            ],
          // Style Catalog round-trip (sc-13136, mirrors image sc-13132): record the picked style id
          // and the RAW pre-style prompt so replay re-selects the picker and recomposes the identical
          // prompt without double-wrapping. Rides advanced → rawAdapterSettings (cloned verbatim by
          // the worker; no backend change). Emitted only when a style is applied so styleless recipes
          // stay byte-identical.
          ...(styleApplied ? { styleId, stylePrompt: stylePromptBase } : {}),
          ...(model === ltxVideoModelId ? { ltxPipeline, distilledVariant, precision } : {}),
          ...(model === ltx25VideoModelId ? { transformerVariant } : {}),
          ...(supportsPromptEnhancement && enhancePrompt
            ? { enhancePrompt: true }
            : {}),
          ...(supportsTextEncoderSelection &&
          enhancePrompt &&
          selectedTextEncoderModel !== defaultTextEncoderId
            ? { textEncoderModel: selectedTextEncoderModel }
            : {}),
          ...(showTorchQuantization && quantization !== "auto" ? { quantization } : {}),
          ...(selectedTierQuantize !== null ? { mlxQuantize: selectedTierQuantize } : {}),
          // Configurable sampler / scheduler (epic 1753). Sealed adapters
          // (LTX native, MLX) silently fall back to default; only the Wan
          // diffusers (torch) path actually applies these.
          ...(sampler && sampler !== "default" ? { sampler } : {}),
          ...(scheduler && scheduler !== "default" ? { scheduler } : {}),
          // Schedule shift (time-shift mu) only pairs with a curated (non-default)
          // scheduler — it shapes that schedule; the default scheduler keeps the
          // engine's resolution-native shift (epic 7114).
          ...(scheduler &&
          scheduler !== "default" &&
          Number.isFinite(Number(schedulerShift))
            ? { schedulerShift: Number(schedulerShift) }
            : {}),
          // Lightning fast-4-step toggle for Wan2.2 A14B MoE (sc-10048, epic 10043). Only the two
          // A14B engines honor it; emit the explicit bool for them (worker sc-10047 reads
          // `advanced.lightning`: absent → defaults on, false → off). When on the worker derives
          // the 4-step/CFG-off recipe, so we suppress the manual steps/guidance overrides below to
          // keep the payload consistent with the recipe the UI is reflecting.
          ...(showLightning ? { lightning } : {}),
          // `stepsPinned` suppresses the override for the same reason `lightningActive` does: the
          // count is not the caller's to set (sc-19502). Omitting `steps` entirely — rather than
          // sending the pinned value — is what "use the baked schedule" means to the engine, and it
          // also means a stale number left in the box by a previously-selected model can never leak
          // into the payload and 400.
          //
          // `stepsOffMenu` is the multi-entry half of the same suppression: the picker below shows
          // such a stale value as "model default", so emitting it anyway would send a count the UI
          // is not displaying AND that the enqueue gate refuses.
          ...(!lightningActive &&
          !stepsPinned &&
          !stepsOffMenu &&
          stepsOverride !== "" &&
          Number.isFinite(Number(stepsOverride))
            ? { steps: Number(stepsOverride) }
            : {}),
          ...(supportsGuidance &&
          !lightningActive &&
          guidanceOverride !== "" &&
          Number.isFinite(Number(guidanceOverride))
            ? { guidanceScale: Number(guidanceOverride) }
            : {}),
          // LTX native guidance knobs (epic 1753 sc-1769). Only emitted for
          // the LTX adapter — the worker would silently ignore them on other
          // adapters but keeping the payload tight avoids surprise overrides.
          ...(showsLegacyLtxGuidanceControls && ltxVideoCfg !== "" && Number.isFinite(Number(ltxVideoCfg))
            ? { videoCfgGuidanceScale: Number(ltxVideoCfg) }
            : {}),
          ...(showsLegacyLtxGuidanceControls && ltxVideoStg !== "" && Number.isFinite(Number(ltxVideoStg))
            ? { videoStgGuidanceScale: Number(ltxVideoStg) }
            : {}),
          ...(showsLegacyLtxGuidanceControls && ltxVideoRescale !== "" && Number.isFinite(Number(ltxVideoRescale))
            ? { videoRescaleScale: Number(ltxVideoRescale) }
            : {}),
          // Clip-conditioning strengths (sc-3522, sc-3755; sc-8445 for Krea Realtime v2v). The
          // worker reads these from `advanced`, defaulting to 1.0 when absent — extend uses the
          // source-clip strength, bridge uses both left and right, and Krea Realtime's v2v uses
          // the source-clip strength. `showClipStrength` is the same predicate that renders the
          // control, so the payload can never carry a strength the user was never offered.
          ...(showClipStrength &&
          videoConditioningStrength !== "" &&
          Number.isFinite(Number(videoConditioningStrength))
            ? { videoConditioningStrength: Number(videoConditioningStrength) }
            : {}),
          ...(mode === "video_bridge" &&
          bridgeRightVideoConditioningStrength !== "" &&
          Number.isFinite(Number(bridgeRightVideoConditioningStrength))
            ? { bridgeRightVideoConditioningStrength: Number(bridgeRightVideoConditioningStrength) }
            : {}),
        },
      });
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }

  const generateDisabled = submitting || !videoValidity.ready;
  const renderLabel = mode === "replace_person" ? "Replace person" : "Render clip";

  return (
    <ModelAvailabilityGate
      ready={modelReady}
      title="Video Studio needs a video model"
      description="Download a recommended video model to start generating."
      offers={modelOffers}
      downloadJobs={modelDownloadJobs}
      onDownload={createModelDownloadJob}
      onOpenModels={() => setActiveView("Models")}
      onOpenQueue={onOpenQueue}
      onCancelJob={onCancelJob}
    >
    <section className="page-frame video-studio">
      <form className="studio-shell" onSubmit={submit}>
        <WorkPanel className="studio-work-panel">
          <div className="prompt-hero-top">
            <ModeTabs
              className="mode-tabs mode-control"
              label="Video mode"
              options={modeOptions}
              mode={mode}
              onChange={setMode}
              blockFor={(value, active) => !active && modeTabBlocked(value)
                ? { text: modeTabBlockedText }
                : null}
            />
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

          <div className="prompt-input-row">
            <textarea
              aria-label="Prompt"
              className="prompt-input"
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={onPromptKeyDown}
              placeholder={
                promptless
                  ? "No prompt needed — this model animates the source image. Pick a first frame below and generate."
                  : "Describe the motion — what moves, where the camera goes, how it feels…"
              }
              value={prompt}
            />
            <button className="prompt-cta" disabled={generateDisabled} type="submit">
              <Icon.Sparkle size={14} />
              {submitting ? "Queueing…" : renderLabel}
            </button>
          </div>

          {promptless ? null : (
            <RefinePromptControl
              guidePath={promptGuide.path}
              modelId={model}
              onApply={setPrompt}
              prompt={prompt}
              refinePrompt={refinePrompt}
              refineModel={refineModel}
              onDownloadRefineModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
              workflow="video"
            />
          )}

          {/* Style Catalog picker + its composed-prompt preview moved into the Style axis row of the
              settings bar (sc-13135) — see the .settings-bar-style-axis block below. */}

          <div className="motion-row">
            <span className="motion-row-label">Motion:</span>
            {MOTIONS.map((option) => (
              <button
                className={motion === option ? "motion-chip active" : "motion-chip"}
                key={option}
                onClick={() => setMotion(option)}
                type="button"
              >
                <span aria-hidden="true" className="motion-arrow">→</span>
                {option}
              </button>
            ))}
          </div>

          {mode !== "text_to_video" ? (
          <div className="studio-source-band">
            {mode === "image_to_video" || mode === "first_last_frame" ? (
              <ImageEditSourcePickerField
                assets={imageAssets}
                buttonLabel="Select image"
                characters={characters}
                emptyLabel="No first frame selected"
                eyebrow="Video Studio"
                importAsset={importAsset}
                label="First frame"
                onChange={setSourceAssetId}
                projectId={activeProject?.id}
                value={sourceAssetId}
              />
            ) : null}

            {mode === "first_last_frame" ? (
              <ImageEditSourcePickerField
                assets={imageAssets}
                buttonLabel="Select image"
                characters={characters}
                emptyLabel="No last frame selected"
                eyebrow="Video Studio"
                importAsset={importAsset}
                label="Last frame"
                onChange={setLastFrameAssetId}
                projectId={activeProject?.id}
                value={lastFrameAssetId}
              />
            ) : null}

            {mode === "image_to_video" || mode === "first_last_frame" ? (
              <FitModeControl
                value={effectiveFitMode(fitMode, false)}
                onChange={setFitMode}
                inpaintCapable={false}
              />
            ) : null}

            {mode === "extend_clip" ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                characters={characters}
                emptyLabel="No source clip selected"
                importAsset={importAsset}
                label="Source clip"
                onChange={setSourceClipAssetId}
                projectId={activeProject?.id}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "video_bridge" ? (
              <>
                <VideoSourcePickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  characters={characters}
                  emptyLabel="No left clip selected"
                  importAsset={importAsset}
                  label="Left clip"
                  onChange={setSourceClipAssetId}
                  projectId={activeProject?.id}
                  value={sourceClipAssetId}
                />
                <VideoSourcePickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  characters={characters}
                  emptyLabel="No right clip selected"
                  importAsset={importAsset}
                  label="Right clip"
                  onChange={setBridgeRightClipAssetId}
                  projectId={activeProject?.id}
                  value={bridgeRightClipAssetId}
                />
              </>
            ) : null}

            {["video_to_video", "reference_video_to_video", "ads2v"].includes(mode) ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                characters={characters}
                emptyLabel="No source clip selected"
                importAsset={importAsset}
                label="Source clip"
                onChange={setSourceClipAssetId}
                projectId={activeProject?.id}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "multi_video_to_video" ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clips"
                characters={characters}
                changeLabel="Edit clips"
                emptyLabel="No source clips selected"
                importAsset={importAsset}
                label="Source clips"
                multiple
                onChange={setSourceClipAssetIds}
                projectId={activeProject?.id}
                values={sourceClipAssetIds}
              />
            ) : null}

            {mode === "ads2v" ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                characters={characters}
                emptyLabel="No reference video selected"
                importAsset={importAsset}
                label="Reference video"
                onChange={setReferenceClipAssetId}
                projectId={activeProject?.id}
                value={referenceClipAssetId}
              />
            ) : null}

            {["reference_to_video", "reference_video_to_video", "ads2v"].includes(mode) ? (
              <ImageEditSourcePickerField
                assets={imageAssets}
                buttonLabel="Select images"
                characters={characters}
                changeLabel="Edit references"
                emptyLabel="No reference images selected"
                eyebrow="Video Studio"
                importAsset={importAsset}
                label="Reference images"
                multiple
                onChange={setReferenceAssetIds}
                projectId={activeProject?.id}
                values={referenceAssetIds}
              />
            ) : null}

            {/* Reference VIDEO clips (sc-17160 / sc-17161). Only for a model that declares a
                reference-clip cap: MiniMax-H3 Ref2VA conditions on motion and pacing from up to 3
                clips, where Bernini's r2v encodes reference images alone and would silently ignore
                them — the invisible-in-the-output failure the per-model caps exist to prevent. */}
            {showReferenceClips ? (
              <VideoSourcePickerField
                assets={videoAssets}
                buttonLabel="Select clips"
                characters={characters}
                changeLabel="Edit clips"
                emptyLabel={`No reference clips selected (up to ${refCaps.clips})`}
                importAsset={importAsset}
                label="Reference clips"
                multiple
                onChange={setSourceClipAssetIds}
                projectId={activeProject?.id}
                values={sourceClipAssetIds}
              />
            ) : null}

            {/* Reference AUDIO clips (sc-17160 landed the payload field; this is the control that
                makes it reachable). Gated on the declared cap alone — it defaults to 0, so the
                picker appears only for a model that says it conditions on audio. Uses the plain
                AssetPickerField with categories hidden, the same shape Audio Studio's reference
                voice uses: the media pickers' All/Images/Video tabs carry no audio bucket.
                `importAsset` + `mediaKind` give it the same local-file import the image/clip
                pickers above carry (sc-17137 review B3): a project with no audio assets would
                otherwise offer an empty grid with no way in. */}
            {showAudioReferences ? (
              <AssetPickerField
                assets={audioAssets}
                buttonLabel="Select audio"
                changeLabel="Edit audio"
                emptyLabel={`No reference audio selected (up to ${refCaps.audio})`}
                importAsset={importAsset}
                label="Reference audio"
                mediaKind="audio"
                multiple
                onChange={setReferenceAudioAssetIds}
                showCategories={false}
                values={referenceAudioAssetIds}
              />
            ) : null}

            {mode === "animate_character" ? (
              <>
                <VideoSourcePickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  characters={characters}
                  emptyLabel="No driving video selected"
                  importAsset={importAsset}
                  label="Driving video"
                  onChange={setSourceClipAssetId}
                  projectId={activeProject?.id}
                  value={sourceClipAssetId}
                />
                {/* The paired-reference UI stays descriptor-gated until the matching inference pin is
                    live. Its ordered array maps to strict Reference,Mask pairs in both workers. */}
                <ImageEditSourcePickerField
                  assets={imageAssets}
                  buttonLabel={scail2MultiReferenceEnabled ? "Select images" : "Select image"}
                  characters={characters}
                  changeLabel={scail2MultiReferenceEnabled ? "Edit characters" : "Change character"}
                  emptyLabel={scail2MultiReferenceEnabled ? "No reference characters selected" : "No reference character selected"}
                  eyebrow="Video Studio"
                  importAsset={importAsset}
                  label={scail2MultiReferenceEnabled ? "Reference characters (ordered, up to 6)" : "Reference character"}
                  multiple={scail2MultiReferenceEnabled}
                  onChange={(ids) =>
                    setReferenceAssetIds(
                      scail2MultiReferenceEnabled ? ids : ids ? [ids] : [],
                    )
                  }
                  projectId={activeProject?.id}
                  value={referenceAssetIds[0] ?? ""}
                  values={referenceAssetIds}
                />
                {scail2ReferenceOverflow ? (
                  <p className="inline-warning" role="alert">
                    SCAIL-2 supports at most {MAX_SCAIL2_REFERENCE_CHARACTERS} reference characters. Remove one before rendering.
                  </p>
                ) : null}
              </>
            ) : null}

            {mode === "replace_person" ? (
              <ReplacePersonPanel
                characters={characters}
                createPersonDetectionJob={createPersonDetectionJob}
                createPersonTrackJob={createPersonTrackJob}
                createModelDownloadJob={createModelDownloadJob}
                importAsset={importAsset}
                personReadiness={personReadiness}
                projectId={activeProject?.id}
                detectionResult={detectionResult}
                matchingTracks={matchingTracks}
                personTrackId={personTrackId}
                replacementMode={replacementMode}
                representativeFrame={representativeFrame}
                saveTrackCorrections={saveTrackCorrections}
                selectedDetection={selectedDetection}
                selectedTrack={selectedTrack}
                setPersonTrackId={setPersonTrackId}
                setReplacementMode={setReplacementMode}
                setSelectedDetectionId={setSelectedDetectionId}
                setSourceClipAssetId={setSourceClipAssetId}
                setTrackName={setTrackName}
                sourceClipAssetId={sourceClipAssetId}
                trackName={trackName}
                videoAssets={videoAssets}
                videoModels={videoModels}
                model={model}
                setModel={setUserModel}
              />
            ) : null}
          </div>
        ) : null}

          <div className="settings-bar">
            <div className="settings-bar-row">
              <label className="settings-field settings-field-model">
                Model
                <StudioUpdateBadge item={selectedModel} />
                <select
                  onChange={(event) => setUserModel(event.target.value)}
                  value={model}
                >
                  {/* Models gated on the selected tab (sc-5716): show only models that serve the
                      active mode, falling back to the full available list if none do (a reduced
                      catalog) so the picker is never empty. */}
                  {(modelsForMode(mode).length ? modelsForMode(mode) : macVideoModels).map((item) => (
                    <option key={item.id} value={item.id}>
                      {updateOptionLabel(item)}
                    </option>
                  ))}
                </select>
                <StudioUpdateNotice item={selectedModel} onUpdate={createModelDownloadJob} />
                {/* Licence-required attribution (sc-17227 §IV.2, landed on the generation surfaces
                    by sc-17161). The Models card alone is one screen a user may never revisit after
                    installing; this is where the model is used. Reads the manifest field — never a
                    second hard-coded copy of a string a licence specifies. */}
                <ModelAttribution model={selectedModel} className="studio-model-attribution" />
                {recipeModelNotice ? (
                  <span className="field-hint" role="status">
                    This clip was made with “{recipeModelNotice}”, which isn’t installed. Its
                    settings were restored — pick a model to re-run it.
                  </span>
                ) : null}
              </label>
              <label className="settings-field settings-field-aspect">
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((value) => (
                    <option key={value} value={value}>
                      {value.replace("x", " × ")}
                    </option>
                  ))}
                </select>
              </label>
              <label className="settings-field settings-field-count">
                Duration
                {/* A MENU, never a range: `limits.durations` is the model's exact renderable set,
                    and MiniMax-H3 is the case that makes the distinction visible — its fourteen
                    `17n + 5` lattice rungs are the ONLY lengths the checkpoint renders, so a
                    slider would emit durations the engine refuses (15.0s among them, which its own
                    docs advertise and which sits between the last rung and the next). The label
                    carries the frame count because that is what the lattice is counting. */}
                <select onChange={(event) => setDuration(Number(event.target.value))} value={duration}>
                  {durationOptions.map((value) => (
                    <option key={value} value={value}>
                      {formatDurationOption(value, fps)}
                    </option>
                  ))}
                </select>
              </label>
              {/* Quant tier, not the abstract Fast/Balanced/Best segment (studio-cleanup
                  sc-15374): the settings bar names the concrete thing that will load. Quality
                  itself is still a live payload/preset field and moves to Advanced. */}
              {showTierPicker ? (
                <TierPickerField
                  className="settings-field settings-field-tier"
                  value={quantTier}
                  onChange={handleExecutionTierChange}
                  items={tierPickerItems}
                  tierSwitching={tierSwitching}
                  tierLabel={tierLabel}
                  title="Switch which installed MLX quant tier generates. Higher precision uses more memory; switching a heavy tier reloads it before the next generation."
                  warning={tierHasMemoryRisk ? (
                    <span className="field-hint quant-tier-memory-note">
                      Higher MLX video tiers may run out of memory on long or high-resolution clips.
                      Your pick is honored.
                    </span>
                  ) : null}
                />
              ) : null}
            </div>
            <LoraPickerSection
              selectedModel={selectedModel}
              selectedLoras={selectedLoras}
              selectedLoraIds={selectedLoraIds}
              compatibleLoras={compatibleLoras}
              userSelectedLoraCount={userSelectedLoraCount}
              toggleLora={toggleLora}
              effectiveLoraWeight={effectiveLoraWeight}
              setLoraWeight={setLoraWeight}
              loraEmptyMessage={loraEmptyMessage}
              onUpdateLora={createLoraDownloadJob}
              importPanel={(
                <StudioLoraImportPanel
                  activeProject={activeProject}
                  createLoraImportJob={createLoraImportJob}
                  models={models}
                />
              )}
            />
            {/* Style axis (sc-13135): the Style Catalog picker leads this row (hidden for promptless
                image-conditioned models and for booru-tag models), followed by the model's Style
                presets — mirrors the Image Studio. The catalog wraps the outgoing prompt
                (Subject:/Style:); "None" resets. */}
            <StyleAxisRow
              available={styleAxisAvailable}
              groups={STYLE_GROUPS}
              styleId={styleId}
              onStyleChange={setStyleId}
              selectedPreset={selectedPreset}
              presets={availablePresets}
              onPresetChange={setSelectedPresetId}
              generalPresets={availableGeneralPresets}
              generalStackIds={generalStackIds}
              onToggleGeneral={toggleGeneralPreset}
              noPresetValue={noPresetId}
              presetPromptParts={presetPromptParts}
              presetLoraDetails={presetLoraDetails}
              savePreset={(
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
                  saveDisabled={!VIDEO_PRESET_MODES.includes(mode)}
                  saveTitle={VIDEO_PRESET_MODES.includes(mode) ? undefined : "Presets are available in Image→Video, Text→Video, or First/Last mode."}
                />
              )}
            />
          </div>

          {/* sc-13136: the EXACT composed prompt the run will send once a style is active — recomputed
              from the same base the submit uses so it can never drift. Sits under the Style axis row.
              Hidden when no style applies. */}
          <StyledPromptPreview active={styleApplied} composedPrompt={composedStylePrompt} />

          {/* `stackAddsNegative` is ANDed with the engine's negative-prompt axis (sc-8445). This
              panel is captioned "Prompt sent" and exists so the user sees exactly what will be
              generated; general presets are filtered on `kind === "general"` alone, so a preset
              carrying `defaults.negativePrompt` reaches a CFG-free model too. Since submit now sends
              `negativePrompt: ""` for such a model, showing the stack's `Negative:` line here would
              assert something that is NOT sent — the same false-copy class this story exists to
              remove. (Only the negative needs the guard: the preview's other extras are aspect and
              count, which every video engine honors.) */}
          <PresetStackPreview
            generalStack={generalStack}
            composed={composedStack}
            stackAddsNegative={supportsNegativePrompt && stackAddsNegative}
            stackAddsCount={stackAddsCount}
          />

          {durationHint ? <p className="helper-copy">{durationHint}</p> : null}

          <AdvancedSection
            hint="cleared values → model default"
            onToggle={() => setAdvancedOpen((value) => !value)}
            open={advancedOpen}
          >
            <div className="advanced-panel">
              <label>
                Frames
                <select onChange={(event) => setFps(Number(event.target.value))} value={fps}>
                  {fpsOptions.map((value) => (
                    <option key={value} value={value}>
                      {value} fps
                    </option>
                  ))}
                </select>
              </label>
              {showLightning ? (
                <div className="lightning-toggle">
                  <label className="checkline">
                    <input
                      checked={lightning}
                      onChange={(event) => setLightning(event.target.checked)}
                      type="checkbox"
                    />
                    Lightning (fast 4-step)
                  </label>
                  <p className="helper-copy">
                    {lightning
                      ? "On: ~10× faster, 4 steps, CFG off, small quality trade-off. Steps and guidance are governed by the recipe."
                      : "Off: full multi-step quality with CFG (slower). Use the Steps and Guidance controls below."}
                  </p>
                </div>
              ) : null}
              {/* MiniMax-H3 turbo (sc-18727). A VARIANT selector, not a toggle: the three published
                  fl2v adapters carry three different (NFE, video shift) pairs, so "on" is not one
                  state. Writes the SAME `selectedLoraIds` the LoRA picker writes — the control and
                  the picker are two views of one selection, which is why they cannot disagree.
                  Default-on (seeded once per model): sc-18729 measured 2.42 h against 12.6 min at
                  the model's default canvas. */}
              {showTurbo ? (
                <div className="lightning-toggle">
                  <label>
                    Turbo (step-distilled)
                    <select
                      aria-label="Turbo (step-distilled)"
                      disabled={!turboVariants.length}
                      onChange={(event) => selectTurboVariant(event.target.value)}
                      value={activeTurboVariant?.id ?? ""}
                    >
                      <option value="">Off — {selectedModel?.defaults?.steps ?? 50} steps</option>
                      {turboVariants.map((variant) => (
                        <option key={variant.id} value={variant.id}>
                          {variant.name} — {variant.sampling.steps} steps
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="helper-copy">
                    {!turboVariants.length
                      ? "No turbo adapter installed for this model. Install one from the LoRA library to render in 4–8 steps instead of the full schedule."
                      : activeTurboVariant
                        ? `On: ${turboRecipeSummary(activeTurboVariant)}. Roughly 7–12× faster (measured 12.6 min against 2.42 h at 1344×768). The distilled checkpoints are trained at 544p/768p and upstream is still improving their detail, so this is a different sample rather than the same one faster — turn it off for the reference schedule.`
                        : "Off: the full schedule at the model's own sigma shift. Slow — a default-canvas clip measured 2.42 h."}
                  </p>
                </div>
              ) : null}
              {model === ltxVideoModelId ? (
                <>
                  <label>
                    LTX pipeline
                    <select onChange={(event) => setLtxPipeline(event.target.value)} value={ltxPipeline}>
                      <option value="auto">Auto (follow quality)</option>
                      <option value="distilled">Distilled (single-stage)</option>
                      <option value="two_stage">Two-stage (dev + upscaler)</option>
                    </select>
                  </label>
                  <label>
                    Distilled variant
                    <select onChange={(event) => setDistilledVariant(event.target.value)} value={distilledVariant}>
                      <option value="1.1">1.1 (newer aesthetic + audio)</option>
                      <option value="1.0">1.0 (original)</option>
                    </select>
                  </label>
                  <label>
                    Precision
                    <select onChange={(event) => setPrecision(event.target.value)} value={precision}>
                      <option value="fp8">FP8 (lower VRAM)</option>
                      <option value="bf16">BF16 (higher quality, CPU offload)</option>
                    </select>
                  </label>
                </>
              ) : null}
              {model === ltx25VideoModelId ? (
                <label>
                  Transformer
                  <select
                    aria-label="LTX-2.5 transformer"
                    onChange={(event) => setTransformerVariant(event.target.value)}
                    value={transformerVariant}
                  >
                    <option value="distilled">Distilled (default, 8 steps)</option>
                    <option value="dev">Dev (guided, 30 steps)</option>
                  </select>
                  <p className="helper-copy">
                    {ltx25Dev
                      ? "Dev uses the guided 30-step transformer and the bundled stage-two distilled refinement."
                      : "Distilled is the default packed transformer and does not apply the refinement adapter twice."}
                  </p>
                </label>
              ) : null}
              {supportsPromptEnhancement ? (
                <>
                  <div className="lightning-toggle">
                    <label className="checkline">
                      <input
                        checked={enhancePrompt}
                        onChange={(event) => setEnhancePrompt(event.target.checked)}
                        type="checkbox"
                      />
                      Enhance prompt before generation
                    </label>
                    <p className="helper-copy">
                      {supportsStockPromptEnhancement
                        ? "Rewrites the prompt with LTX-2.5's separately downloaded stock Gemma-4 enhancer."
                        : "Rewrites the prompt with the selected text encoder before the model encodes it."}
                    </p>
                  </div>
                  {supportsTextEncoderSelection ? (
                    <>
                      <label>
                        Text encoder model
                        <select
                          aria-label="Text encoder model"
                          disabled={!enhancePrompt}
                          onChange={(event) => setTextEncoderModel(event.target.value)}
                          value={selectedTextEncoderModel}
                        >
                          {textEncoderOptions.map((option) => (
                            <option key={option.id} value={option.id}>
                              {option.label}
                            </option>
                          ))}
                          {!selectedTextEncoderAvailable &&
                          selectedTextEncoderModel !== defaultTextEncoderId ? (
                            <option disabled value={selectedTextEncoderModel}>
                              Previously selected encoder (not staged)
                            </option>
                          ) : null}
                        </select>
                      </label>
                      <p className="helper-copy">
                        {!selectedTextEncoderAvailable &&
                        selectedTextEncoderModel !== defaultTextEncoderId
                          ? "The recorded encoder is not staged on this worker. Choose the shipped default or stage the alternate before rendering."
                          : textEncoderOptions.length > 1
                            ? "Only complete encoders already staged for this worker are listed."
                            : "The shipped encoder is the default. Complete operator-staged alternates appear here after Models is refreshed."}
                      </p>
                    </>
                  ) : null}
                </>
              ) : null}
              {showsLegacyLtxGuidanceControls ? (
                <>
                  <label>
                    Video CFG
                    <input
                      min="0"
                      max="30"
                      onChange={(event) => setLtxVideoCfg(event.target.value)}
                      placeholder="4.0"
                      step="0.1"
                      type="number"
                      value={ltxVideoCfg}
                    />
                  </label>
                  <label>
                    Video STG
                    <input
                      min="0"
                      max="10"
                      onChange={(event) => setLtxVideoStg(event.target.value)}
                      placeholder="0.0"
                      step="0.1"
                      type="number"
                      value={ltxVideoStg}
                    />
                  </label>
                  <label>
                    Video rescale
                    <input
                      min="0"
                      max="2"
                      onChange={(event) => setLtxVideoRescale(event.target.value)}
                      placeholder="0.7"
                      step="0.05"
                      type="number"
                      value={ltxVideoRescale}
                    />
                  </label>
                </>
              ) : null}
              {showClipStrength ? (
                <>
                  <label>
                    {mode === "video_bridge" ? "Left clip strength" : "Clip strength"}
                    <input
                      min="0"
                      max="1"
                      onChange={(event) => setVideoConditioningStrength(event.target.value)}
                      placeholder="1.0"
                      step="0.05"
                      type="number"
                      value={videoConditioningStrength}
                    />
                  </label>
                  {mode === "video_bridge" ? (
                    <label>
                      Right clip strength
                      <input
                        min="0"
                        max="1"
                        onChange={(event) => setBridgeRightVideoConditioningStrength(event.target.value)}
                        placeholder="1.0"
                        step="0.05"
                        type="number"
                        value={bridgeRightVideoConditioningStrength}
                      />
                    </label>
                  ) : null}
                </>
              ) : null}
              {showQualitySegment ? (
                <label
                  className="video-quality-field"
                  title="Step-count preset for Stable Video Diffusion: Draft 15, Balanced 25, Final 30. An explicit Steps value below overrides it."
                >
                  Quality
                  <div className="quality-segment" role="radiogroup" aria-label="Quality">
                    {qualityChoices.map(([value, label]) => (
                      <button
                        aria-checked={quality === value}
                        className={quality === value ? "active" : ""}
                        key={value}
                        onClick={() => setQuality(value)}
                        role="radio"
                        type="button"
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </label>
              ) : null}
              {showTorchQuantization ? (
                <label>
                  Quantization
                  <select onChange={(event) => setQuantization(event.target.value)} value={quantization}>
                    <option value="auto">Auto (per-platform default)</option>
                    {quantVariants.map(([id, variant]) => (
                      <option key={id} value={id}>
                        {variant?.label ?? id}
                      </option>
                    ))}
                    <option value="none">Full precision (unquantized)</option>
                  </select>
                </label>
              ) : null}
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
                {stepsChoice && !ltx25Dev ? (
                  <select
                    disabled={lightningActive}
                    onChange={(event) => setStepsOverride(event.target.value)}
                    title={
                      lightningActive
                        ? "Governed by Lightning (fast 4-step). Turn Lightning off to set steps."
                        : `${selectedModel?.ui?.label ?? selectedModel?.name ?? "This model"} is distilled: it renders at ${humanizedNumberMenu(stepsChoice)} steps only.`
                    }
                    value={lightningActive || stepsOffMenu ? "" : stepsOverride}
                  >
                    {/* The cleared state, exactly as for the free-text box above and as the panel's
                        own "cleared values → model default" hint promises: no `advanced.steps` is
                        sent and the engine runs `defaults.steps`. It is deliberately NOT
                        `stepsChoice[0]` — `limits.steps[0]` is not a default — and it is safe
                        because the manifest invariant pins `defaults.steps` onto the menu. */}
                    <option value="">
                      {stepsDefaultFromModel(selectedModel) == null
                        ? "Model default"
                        : `${stepsDefaultFromModel(selectedModel)} (model default)`}
                    </option>
                    {stepsChoice.map((value) => (
                      <option key={value} value={String(value)}>
                        {value}
                      </option>
                    ))}
                  </select>
                ) : (
                  /* The `min` floor is the MODEL's, not a blanket 1 (sc-19426 / sc-17161).
                     MiniMax-H3 declares 2: the unit is model evaluations (NFE — the engine appends
                     the terminal sigma itself), and a 1-evaluation schedule is a single Euler jump
                     from pure noise, so the floor is the corrected sc-18726 product judgement,
                     REFUSED rather than raised — the form has to say so instead of letting it be
                     typed and then 400'd at enqueue. `hardMinSteps` and `limits.steps` are INDEPENDENT axes
                     (sc-19502): the floor bounds an open range, the menu enumerates a closed set,
                     and this branch is the one a model reaches when it declares no menu — so the
                     floor still has to be honoured here even though the pinned/menu cases above
                     express their own, tighter constraint. */
                  <input
                    min={String(minSteps)}
                    max="80"
                    disabled={lightningActive || stepsPinned}
                    onChange={(event) => setStepsOverride(event.target.value)}
                    placeholder={
                      lightningActive
                        ? "4 (Lightning)"
                        : stepsPinned
                          ? `${stepsPinnedValue} (fixed schedule)`
                          : /* Turbo supplies a step count but does NOT seize the control, unlike
                               Lightning above: upstream's own spec table lists the 8-step MiniMax-H3
                               adapter as "8 / 4", so a caller who knows the checkpoint may run it
                               shorter, and `minimax_h3_sampling` honours an explicit
                               `advanced.steps` over the variant's default for exactly that reason.
                               So the placeholder REPORTS the recipe's count while the box stays
                               editable — a knob honoured rather than rejected. */
                            activeTurboVariant
                            ? `${activeTurboVariant.sampling.steps} (Turbo)`
                            : String(stepsDefaultFromModel(selectedModel) ?? "")
                    }
                    title={
                      lightningActive
                        ? "Governed by Lightning (fast 4-step). Turn Lightning off to set steps."
                        : stepsPinned
                          ? `${selectedModel?.ui?.label ?? selectedModel?.name ?? "This model"} is distilled: it runs a fixed ${stepsPinnedValue}-step schedule baked into its weights and cannot render any other step count.`
                          : activeTurboVariant
                            ? `${activeTurboVariant.name} is distilled for ${activeTurboVariant.sampling.steps} steps, which is what runs when this is blank. You can still set your own count.`
                            : minSteps > 1
                              ? `${selectedModel?.name ?? "This model"} needs at least ${minSteps} steps.`
                              : undefined
                    }
                    type="number"
                    value={lightningActive || stepsPinned ? "" : stepsOverride}
                  />
                )}
              </label>
              {supportsGuidance ? (
                <label>
                  Guidance
                  <input
                    min="0"
                    max="30"
                    disabled={lightningActive}
                    onChange={(event) => setGuidanceOverride(event.target.value)}
                    placeholder={lightningActive ? "off (Lightning)" : (() => {
                      const value = guidanceDefaultFromModel(selectedModel);
                      return value == null ? "" : String(value);
                    })()}
                    step="0.1"
                    title={lightningActive ? "Governed by Lightning (fast 4-step). Turn Lightning off to set guidance." : undefined}
                    type="number"
                    value={lightningActive ? "" : guidanceOverride}
                  />
                </label>
              ) : null}
              <label>
                Character
                <select onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
                  <option value="">No character</option>
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
              {supportsNegativePrompt ? (
                <label className="prompt-field">
                  Negative prompt
                  <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
                </label>
              ) : null}
              {characterId ? (
                <div className="guidance-strip">
                  <strong>Character reference</strong>
                  <span>
                    Character and look are saved with the recipe; LTX image conditioning uses IC-LoRA when the selected preset includes one.
                  </span>
                </div>
              ) : null}
              {/* Video upscale (super-resolve an existing clip) folds into Advanced — it
                  previously lived in the render rail this layout removes. It operates on a
                  selected existing asset, independent of the current generation payload. */}
              <VideoUpscalePanel
                characters={characters}
                createVideoUpscaleJob={createVideoUpscaleJob}
                importAsset={importAsset}
                macCapabilities={macCapabilities}
                onSubmitted={(job) => {
                  onLocalJobCreated(job);
                  onOpenQueue();
                }}
                selectedAsset={selectedAsset}
                projectId={activeProject?.id}
                videoAssets={videoAssets}
              />
            </div>
          </AdvancedSection>

          {/* Every reason Generate is dead — mode/preset/LoRA/worker problems — in one chip
              row, from the same summary that gates the button (sc-10650). Project, prompt
              and inputs are silent requirements: their empty fields show it. */}
          <ValidationSummary issues={videoValidity.surfaced} label="Generate errors" />
          {canStartNewGenerationFromReplay ? (
            <button
              className="secondary-action"
              onClick={handleStartNewGenerationFromReplay}
              type="button"
            >
              {startNewGenerationLabel}
            </button>
          ) : null}

        </WorkPanel>

        <div className="studio-results">
          <section className="review-panel">
            <div className="review-panel-head">
              <h2>Latest batch</h2>
              <span className="kbd-hint">
                <kbd>⌘</kbd>
                <kbd>↵</kbd>
                to render
              </span>
            </div>
            {localJobs.length ? (
              <div className="worker-progress-card-stack local-job-stack">
                {localJobs.map((job) => {
                  const jobAssets = jobVideoResultAssets(job, assets);
                  return (
                    <WorkerProgressCard
                      key={job.id}
                      job={job}
                      thumbnailsVariant="video-player"
                      thumbnailAssets={jobAssets}
                      onThumbnailClick={(asset) => onPreview(asset, jobAssets)}
                      onCancel={onCancelJob}
                      onOpenQueue={onOpenQueue}
                    />
                  );
                })}
              </div>
            ) : null}
            {latestAssets.length ? (
              <div className="recent-assets">
                {localJobs.length ? <h3 className="recent-assets__title">Recent Assets</h3> : null}
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
            ) : localJobs.length ? null : (
              <div className="empty-panel">No fresh clip batch</div>
            )}
          </section>

          {/* Replace-person A/B / side-by-side review (video-specific, no Image Studio
              equivalent) — surfaces the latest replacement clip against its source. */}
          {comparisonAsset?.recipe?.mode === "replace_person" && comparisonSource ? (
            <div className="comparison-panel">
              <div className="comparison-toolbar">
                <div className="segmented-control compact-segment" aria-label="Comparison mode">
                  <button className={comparisonMode === "side_by_side" ? "active" : ""} onClick={() => setComparisonMode("side_by_side")} type="button">
                    Side by Side
                  </button>
                  <button className={comparisonMode === "ab" ? "active" : ""} onClick={() => setComparisonMode("ab")} type="button">
                    A/B
                  </button>
                </div>
                {comparisonMode === "ab" ? (
                  <div className="segmented-control compact-segment" aria-label="A/B source">
                    <button className={abSide === "original" ? "active" : ""} onClick={() => setAbSide("original")} type="button">
                      A
                    </button>
                    <button className={abSide === "replacement" ? "active" : ""} onClick={() => setAbSide("replacement")} type="button">
                      B
                    </button>
                  </div>
                ) : null}
              </div>
              {comparisonMode === "side_by_side" ? (
                <div className="comparison-grid">
                  <div>
                    <p className="eyebrow">Original</p>
                    <AssetMedia asset={comparisonSource} />
                  </div>
                  <div>
                    <p className="eyebrow">Replacement</p>
                    <AssetMedia asset={comparisonAsset} />
                  </div>
                </div>
              ) : (
                <div className="comparison-single">
                  <p className="eyebrow">{abSide === "original" ? "A Original" : "B Replacement"}</p>
                  <AssetMedia asset={abSide === "original" ? comparisonSource : comparisonAsset} />
                </div>
              )}
            </div>
          ) : null}
        </div>
      </form>
      {guideOpen ? (
        <PromptGuideModal guide={promptGuide} modelName={selectedModel?.name} onClose={() => setGuideOpen(false)} />
      ) : null}
    </section>
    </ModelAvailabilityGate>
  );
}
