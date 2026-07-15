import React, { useEffect, useMemo, useRef, useState } from "react";
import { parseResolution, pickClosestResolution } from "../resolutionMatch.js";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { RefinePromptControl } from "../components/RefinePromptControl.jsx";
import { VideoUpscalePanel } from "./VideoUpscalePanel.jsx";
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
  PresetGuidanceStrip,
  PresetStackPreview,
  SavePresetPanel,
  useGenerationStudio,
  useSavePreset,
} from "./generationStudio.jsx";
import { ReplacePersonPanel } from "./ReplacePersonPanel.jsx";
import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { videoGenerateValidation } from "../videoStudioValidation.js";
import { useValidation } from "../validation/useValidation.js";
import { ValidationSummary } from "../validation/Validation.jsx";
import { downloadOffersFor, videoModelUsable } from "../modelEligibility.js";
import { PROMPT_REFINE_MODEL_ID, WAN_A14B_LIGHTNING_MODEL_IDS } from "../constants.js";
import {
  DEFAULT_MAC_CAPABILITIES,
  macAvailableModels,
  macGatingActive,
  macVideoModeBlock,
} from "../macGating.js";
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import { qualityChoices } from "../jobTypes.js";
import {
  defaultTierSelection,
  installedTiers,
  shouldShowTierPicker,
  tierLabel,
  tierQuantize,
} from "../quantTier.js";
import { readLastTier, writeLastTier } from "../lastTierStore.js";
import {
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  stepsDefaultFromModel,
} from "../samplerOptions.js";

const ltxVideoModelId = "ltx_2_3";
const ltxIcLoraRequiredModes = new Set(["extend_clip", "video_bridge"]);
// Keep Video Studio's MLX lane q4-first (sc-10859). Unlike Image Studio, it deliberately does not
// inherit the app-wide generation-quality setting: video activation peaks need a larger safety margin.
const TIER_SCREEN = "video";
const VIDEO_DEFAULT_TIER = "q4";

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
    deleteAsset,
    purgeAsset,
    gpuOptions,
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
  const [motion, setMotion] = useState(saved.motion ?? "slow push-in");
  // Memoize the per-type catalog splits (sc-8939): both feed a dozen pickers/trays and
  // re-filtering the full catalog on every render (including unrelated state churn) is
  // needless. Recompute only when the catalog changes; stable identities also keep the
  // downstream memoized offers/consumers from thrashing.
  const imageAssets = useMemo(
    () => assets.filter((asset) => asset.type === "image" || asset.type === "frame"),
    [assets],
  );
  const videoAssets = useMemo(() => assets.filter((asset) => asset.type === "video"), [assets]);
  // Open on Text→Video for parity with Image Studio's Text→Image default and the
  // launch-request fallback below (sc-5716); the prior image_to_video default was the odd one out.
  const [mode, setMode] = useState(saved.mode ?? "text_to_video");
  const [prompt, setPrompt] = useState(saved.prompt ?? "Camera slowly pushes in while the scene comes alive");
  const [quality, setQuality] = useState(saved.quality ?? "balanced");
  const [ltxPipeline, setLtxPipeline] = useState(saved.ltxPipeline ?? "auto");
  const [distilledVariant, setDistilledVariant] = useState(saved.distilledVariant ?? "1.1");
  const [precision, setPrecision] = useState(saved.precision ?? "fp8");
  const [quantization, setQuantization] = useState(saved.quantization ?? "auto");
  // MLX generation tier (sc-12165), separate from the torch/GGUF `quantization` state above.
  // The explicit pick is persisted per (video, model), outside the workspace settings snapshot.
  const [quantTier, setQuantTier] = useState("");
  const [tierSwitching, setTierSwitching] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  const [model, setModel] = useState(saved.model ?? videoModels[0]?.id ?? ltxVideoModelId);
  const [guideOpen, setGuideOpen] = useState(false);
  // Mac UI gating (sc-3486): hide torch-only video models (e.g. SVD) and snap off one if selected.
  const macVideoModels = useMemo(
    () => macAvailableModels(videoModels, macCapabilities),
    [videoModels, macCapabilities],
  );
  useEffect(() => {
    if (macVideoModels.length && !macVideoModels.some((item) => item.id === model)) {
      setModel(macVideoModels[0].id);
    }
  }, [macVideoModels, model]);
  const selectedModel = videoModels.find((item) => item.id === model) ?? videoModels[0];
  // Models gated on the selected tab, not tabs on the selected model (sc-5716). A model "serves" a
  // mode when it declares the capability AND, under active Mac gating, that mode is MLX-routed for
  // it (`macVideoModeBlock` is a no-op off-Mac, so there this is pure capability). The mode tabs,
  // the model picker, and the snap-on-mode-switch effect all derive from this so the user is never
  // trapped on a mode whose model can't serve the others.
  const macGating = macGatingActive(macCapabilities);
  const baseVideoModels = macVideoModels.length ? macVideoModels : videoModels;
  const modelServesMode = (item, value) =>
    Boolean(item?.capabilities?.includes(value)) && !macVideoModeBlock(item, macCapabilities, value);
  const modelsForMode = (value) => baseVideoModels.filter((item) => modelServesMode(item, value));
  // Model-availability gate (sc-5947): when the user has no mac-available video model at all,
  // show recommended video-model downloads instead of the studio. `ready` matches the picker
  // (which falls back to all baseVideoModels); offers come from the full catalog via
  // videoModelUsable, recommended-first.
  const modelReady = baseVideoModels.length > 0;
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
  // LTX-2.3 native guidance knobs (epic 1753 sc-1769). The native ltx-core
  // path has no diffusers scheduler to swap — these three values (cfg + STG +
  // rescale) drive its sealed MultiModalGuiderParams instead.
  const [ltxVideoCfg, setLtxVideoCfg] = useState(saved.videoCfgGuidanceScale ?? "");
  const [ltxVideoStg, setLtxVideoStg] = useState(saved.videoStgGuidanceScale ?? "");
  const [ltxVideoRescale, setLtxVideoRescale] = useState(saved.videoRescaleScale ?? "");
  // Clip-conditioning strengths for the LTX IC-LoRA extend/bridge paths (sc-3522,
  // sc-3755). The worker reads these from `advanced` (default 1.0 when absent):
  // the source/left clip uses videoConditioningStrength, the bridge right clip
  // uses bridgeRightVideoConditioningStrength.
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
    loraWeights,
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
    initialShowIncompatibleLoras: saved.showIncompatibleLoras ?? false,
    initialGeneralStackIds: saved.generalStackIds ?? [],
  });
  // Sampler / scheduler menus declared by the model. Video Wan torch
  // declares the full menu; sealed paths (LTX native, MLX) drop to
  // default-only and the picker hides. Gated to the ACTIVE backend (epic 7114 P5):
  // `macGating` is the worker `mlx_required` master switch, so the menu reflects the
  // manifest's `mlx.limits` override on Mac/MLX and `candle.limits` on the candle build.
  const activeBackend = macGating ? "mlx" : "candle";
  // MLX tier installs and the torch/GGUF quantization variants can coexist on one catalog model,
  // but they target different worker lanes. Only derive an MLX tier while the MLX lane is active;
  // the existing quantization picker owns the non-MLX lane. `convRotEligible: false` keeps the
  // candle-only INT8-ConvRot image tier out of this MLX video control.
  const mlxTierLane = activeBackend === "mlx";
  const tierOptions = useMemo(
    () => ({ convRotEligible: false, defaultQuality: VIDEO_DEFAULT_TIER }),
    [],
  );
  const availableTiers = useMemo(
    () => (mlxTierLane ? installedTiers(selectedModel, tierOptions) : []),
    [mlxTierLane, selectedModel, tierOptions],
  );
  const showTierPicker = useMemo(
    () => mlxTierLane && shouldShowTierPicker(selectedModel, tierOptions),
    [mlxTierLane, selectedModel, tierOptions],
  );
  const showTorchQuantization = !mlxTierLane && supportsQuantization;
  const selectedMlxQuantize =
    mlxTierLane && availableTiers.includes(quantTier) ? tierQuantize(quantTier) : null;
  const tierHasMemoryRisk = showTierPicker && ["q8", "bf16"].includes(quantTier);

  // Seed from the per-(video, model) sticky, then the video-specific q4 base, clamped to installed.
  // A model transition always re-seeds even when both models happen to expose the same tier list.
  const availableTiersKey = availableTiers.join(",");
  const quantTierModelRef = useRef(null);
  useEffect(() => {
    const modelChanged = quantTierModelRef.current !== model;
    quantTierModelRef.current = model;
    if (!modelChanged && availableTiers.includes(quantTier)) {
      return;
    }
    setQuantTier(
      defaultTierSelection(selectedModel, readLastTier(TIER_SCREEN, model), tierOptions) ?? "",
    );
    // `availableTiersKey` is the stable install-state dependency; the remaining values are read from
    // the render that produced it, matching Image Studio's catalog-refresh seed behavior.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, availableTiersKey]);

  const tierSwitchTimer = useRef(null);
  useEffect(() => () => clearTimeout(tierSwitchTimer.current), []);
  const handleTierChange = (nextTier) => {
    if (nextTier === quantTier) {
      return;
    }
    setQuantTier(nextTier);
    writeLastTier(TIER_SCREEN, model, nextTier);
    setTierSwitching(nextTier);
    clearTimeout(tierSwitchTimer.current);
    tierSwitchTimer.current = setTimeout(() => setTierSwitching(""), 1500);
  };
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
  const requiresLtxIcLora = selectedModel?.id === ltxVideoModelId && ltxIcLoraRequiredModes.has(mode);
  const hasLtxIcLora = presetLoraDetails.some((lora) => !lora.missing && loraLooksLikeIcLora(lora));

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
  }, [selectedAssetId, selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (launchRequest?.view !== "Video") {
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
  }, [i2vSourceAssetId, mode, selectedModel?.id, assets]);

  // Models are gated on the selected tab (sc-5716): when the active mode isn't served by the current
  // model, snap to the first model that serves it so the user can always leave a mode. Generalizes
  // the old per-mode snaps (replace_person → first replace-capable model; animate_character →
  // scail2_14b) to every mode, including the Bernini editing/reference modes. A no-op when the
  // current model already serves the mode (e.g. an LTX image_to_video → text_to_video switch) or
  // when no model serves it (a reduced catalog) — there's nothing to snap to.
  useEffect(() => {
    if (modelServesMode(selectedModel, mode)) {
      return;
    }
    const fallback = modelsForMode(mode)[0];
    if (fallback && fallback.id !== model) {
      setModel(fallback.id);
    }
    // modelServesMode / modelsForMode close over videoModels + macCapabilities, captured below.
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
      motion,
      videoCfgGuidanceScale: finiteNumberOrUndefined(ltxVideoCfg),
      videoStgGuidanceScale: finiteNumberOrUndefined(ltxVideoStg),
      videoRescaleScale: finiteNumberOrUndefined(ltxVideoRescale),
    }),
  });

  useStudioSettingsWriter("video", activeProject?.id ?? null, {
    motion,
    mode,
    prompt,
    quality,
    ltxPipeline,
    distilledVariant,
    precision,
    quantization,
    advancedOpen,
    selectedLoraIds,
    loraWeights,
    showIncompatibleLoras,
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
  videoModels.length > 0);

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
  // Mac UI gating (sc-3486, sc-3773, sc-5716): mode tabs are gated at the MODE level, not on the
  // selected model. A tab is disabled only under active Mac gating when NO available model serves
  // the mode (mode-level availability across `macVideoModels`) — never on the selected model's
  // `videoModes`, which used to trap the user on replace_person / animate_character with no way
  // back. Off-Mac `macGating` is false so tabs are never disabled here. The active tab is always
  // left enabled so a reduced catalog can't strand you on a disabled tab. `macVideoModeBlock` still
  // gates the in-mode model picker + submit (via `modelsForMode` / `supportsMode`).
  const macModeTabBlocked = (value) => macGating && modelsForMode(value).length === 0;
  const matchingTracks = personTracks.filter((track) => track.sourceAssetId === sourceClipAssetId);
  const latestDetectionJob = jobs
    .filter(
      (job) =>
        job.type === "person_detect" &&
        job.status === "completed" &&
        job.projectId === activeProject?.id &&
        job.payload?.sourceAssetId === sourceClipAssetId,
    )
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
  const detectionResult = latestDetectionJob?.result ?? null;
  const representativeFrame = assets.find((asset) => asset.id === detectionResult?.frameAssetId);
  const selectedDetection = detectionResult?.detections?.find((item) => item.id === selectedDetectionId) ?? detectionResult?.detections?.[0];
  const selectedTrack = personTracks.find((track) => track.id === personTrackId);
  const comparisonAsset = latestAssets.find((asset) => asset.recipe?.mode === "replace_person");
  const comparisonSource = assets.find((asset) => asset.id === comparisonAsset?.lineage?.sourceClipAssetId);
  const hasInputs =
    mode === "text_to_video" ||
    (mode === "image_to_video" && sourceAssetId) ||
    (mode === "first_last_frame" && sourceAssetId && lastFrameAssetId) ||
    (mode === "extend_clip" && sourceClipAssetId) ||
    (mode === "video_bridge" && sourceClipAssetId && bridgeRightClipAssetId) ||
    (mode === "replace_person" && sourceClipAssetId && personTrackId && characterId) ||
    // Bernini editing / reference-driven modes (sc-4703).
    (mode === "video_to_video" && sourceClipAssetId) ||
    (mode === "reference_to_video" && referenceAssetIds.length > 0) ||
    (mode === "reference_video_to_video" && sourceClipAssetId && referenceAssetIds.length > 0) ||
    // Bernini multi-source modes (sc-5425): mv2v needs >=2 clips; ads2v needs a source
    // clip, a reference video, and >=1 reference image.
    (mode === "multi_video_to_video" && sourceClipAssetIds.length >= 2) ||
    (mode === "ads2v" && sourceClipAssetId && referenceClipAssetId && referenceAssetIds.length > 0) ||
    // SCAIL-2 character animation (sc-5449): a driving video + a reference character image.
    (mode === "animate_character" && sourceClipAssetId && referenceAssetIds.length > 0);
  // Don't let Replace Person queue a job the readiness endpoint says no live
  // worker can run — that would sit unclaimable instead of honoring the gate.
  const replaceReady = mode !== "replace_person" || personReadiness?.replace?.ready !== false;
  // Image-conditioned models (e.g. Stable Video Diffusion) take no text prompt;
  // they animate the source image, so don't gate submission on prompt text.
  const promptless = Boolean(selectedModel?.promptless);
  // One summary gates Generate and carries every reason it might be dead, so the button
  // and the messages can't drift — the bug this screen used to embody, where `canSubmit`
  // and `blockedMessage` re-derived the same rules side by side (epic 10644).
  const videoDraft = useMemo(
    () => ({
      activeProject,
      promptless,
      prompt,
      supportsMode,
      implementedMode,
      hasInputs,
      requiresLtxIcLora,
      hasLtxIcLora,
      replaceReady,
      modelName: selectedModel?.name,
      presetMissing: presetValidationResult.missing,
      presetIncompatible: presetValidationResult.incompatible,
      loraIncompatible: selectedLoraValidationResult.incompatible,
    }),
    [
      activeProject,
      promptless,
      prompt,
      supportsMode,
      implementedMode,
      hasInputs,
      requiresLtxIcLora,
      hasLtxIcLora,
      replaceReady,
      selectedModel,
      presetValidationResult,
      selectedLoraValidationResult,
    ],
  );
  const videoValidity = useValidation(videoGenerateValidation, videoDraft, undefined);
  const [width, height] = resolution.split("x").map((value) => Number(value));
  const durationOptions = selectedModel?.limits?.durations ?? [4, 6, 8, 10];
  const resolutionOptions = selectedModel?.limits?.resolutions ?? ["768x512", "640x640", "1280x720", "720x1280"];

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
  const stackAddsNegative = generalStack.some((preset) => Boolean(preset?.defaults?.negativePrompt));
  const stackAddsCount = generalStack.some((preset) => Number.isFinite(Number(preset?.defaults?.count)));
  const fpsOptions = selectedModel?.limits?.fps ?? [24, 25, 30];
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
    if (submitting) {
      return;
    }
    setSubmitting(true);
    try {
      // Fold the general-preset stack (epic 11949): send the composed prompt + negative and,
      // when a general sets aspect, the snapped resolution. The client is authoritative for the
      // composed prompt, so presetPromptResolvedClientSide tells the server to skip its fold.
      // (Video has no `count` field, so the stack's variations don't apply here.)
      const stackActive = generalStack.length > 0;
      const stackResolution = stackActive && composedStack.resolution ? parseResolution(composedStack.resolution) : null;
      const job = await createVideoJob({
        mode,
        prompt: stackActive ? composedStack.prompt : prompt,
        negativePrompt: stackActive ? composedStack.negativePrompt : negativePrompt,
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
        presetPromptResolvedClientSide: stackActive || undefined,
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
        // Bernini multi-source clips (sc-5425) — only mv2v carries the array.
        sourceClipAssetIds: mode === "multi_video_to_video" ? sourceClipAssetIds : [],
        bridgeRightClipAssetId: mode === "video_bridge" ? bridgeRightClipAssetId || null : null,
        // Bernini subject references (sc-4703 / sc-5425) — the reference-driven modes + ads2v carry
        // them; SCAIL-2 character animation (sc-5449) carries the reference character image.
        referenceAssetIds: [
          "reference_to_video",
          "reference_video_to_video",
          "ads2v",
          "animate_character",
        ].includes(mode)
          ? referenceAssetIds
          : [],
        // Bernini ads2v reference video (sc-5425).
        referenceClipAssetId: mode === "ads2v" ? referenceClipAssetId || null : null,
        personTrackId: mode === "replace_person" ? personTrackId || null : null,
        replacementMode: mode === "replace_person" ? replacementMode : "face_only",
        loras: selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
        advanced: {
          resolution,
          durationHint,
          motion,
          selectedPersonTrack: selectedTrack ?? null,
          replacementModeLabel: replacementModeLabels[replacementMode],
          ...(model === ltxVideoModelId ? { ltxPipeline, distilledVariant, precision } : {}),
          ...(showTorchQuantization && quantization !== "auto" ? { quantization } : {}),
          ...(selectedMlxQuantize !== null ? { mlxQuantize: selectedMlxQuantize } : {}),
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
          ...(!lightningActive && stepsOverride !== "" && Number.isFinite(Number(stepsOverride))
            ? { steps: Number(stepsOverride) }
            : {}),
          ...(!lightningActive && guidanceOverride !== "" && Number.isFinite(Number(guidanceOverride))
            ? { guidanceScale: Number(guidanceOverride) }
            : {}),
          // LTX native guidance knobs (epic 1753 sc-1769). Only emitted for
          // the LTX adapter — the worker would silently ignore them on other
          // adapters but keeping the payload tight avoids surprise overrides.
          ...(selectedModel?.adapter === "ltx_video" && ltxVideoCfg !== "" && Number.isFinite(Number(ltxVideoCfg))
            ? { videoCfgGuidanceScale: Number(ltxVideoCfg) }
            : {}),
          ...(selectedModel?.adapter === "ltx_video" && ltxVideoStg !== "" && Number.isFinite(Number(ltxVideoStg))
            ? { videoStgGuidanceScale: Number(ltxVideoStg) }
            : {}),
          ...(selectedModel?.adapter === "ltx_video" && ltxVideoRescale !== "" && Number.isFinite(Number(ltxVideoRescale))
            ? { videoRescaleScale: Number(ltxVideoRescale) }
            : {}),
          // LTX IC-LoRA clip-conditioning strengths (sc-3522, sc-3755). The worker
          // reads these from `advanced`, defaulting to 1.0 when absent — extend uses
          // the source-clip strength, bridge uses both left and right.
          ...(["extend_clip", "video_bridge"].includes(mode) &&
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
            <div className="mode-tabs mode-control" role="tablist" aria-label="Video mode">
              {modeOptions.map(([value, label]) => {
                // Disabled only when no available model serves this mode on Mac — and never the
                // active tab, so the user can always switch away (sc-5716).
                const blocked = value !== mode && macModeTabBlocked(value);
                const active = mode === value;
                return (
                  <button
                    className={active ? "mode-tab active" : "mode-tab"}
                    key={value}
                    role="tab"
                    aria-selected={active}
                    onClick={() => setMode(value)}
                    type="button"
                    disabled={blocked}
                    title={blocked ? "No installed model supports this mode on macOS." : undefined}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
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
              <AssetPickerField
                assets={imageAssets}
                buttonLabel="Select image"
                emptyLabel="No first frame selected"
                label="First frame"
                onChange={setSourceAssetId}
                value={sourceAssetId}
              />
            ) : null}

            {mode === "first_last_frame" ? (
              <AssetPickerField
                assets={imageAssets}
                buttonLabel="Select image"
                emptyLabel="No last frame selected"
                label="Last frame"
                onChange={setLastFrameAssetId}
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
              <AssetPickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                emptyLabel="No source clip selected"
                label="Source clip"
                onChange={setSourceClipAssetId}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "video_bridge" ? (
              <>
                <AssetPickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  emptyLabel="No left clip selected"
                  label="Left clip"
                  onChange={setSourceClipAssetId}
                  value={sourceClipAssetId}
                />
                <AssetPickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  emptyLabel="No right clip selected"
                  label="Right clip"
                  onChange={setBridgeRightClipAssetId}
                  value={bridgeRightClipAssetId}
                />
              </>
            ) : null}

            {["video_to_video", "reference_video_to_video", "ads2v"].includes(mode) ? (
              <AssetPickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                emptyLabel="No source clip selected"
                label="Source clip"
                onChange={setSourceClipAssetId}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "multi_video_to_video" ? (
              <AssetPickerField
                assets={videoAssets}
                buttonLabel="Select clips"
                changeLabel="Edit clips"
                emptyLabel="No source clips selected"
                label="Source clips"
                multiple
                onChange={setSourceClipAssetIds}
                values={sourceClipAssetIds}
              />
            ) : null}

            {mode === "ads2v" ? (
              <AssetPickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                emptyLabel="No reference video selected"
                label="Reference video"
                onChange={setReferenceClipAssetId}
                value={referenceClipAssetId}
              />
            ) : null}

            {["reference_to_video", "reference_video_to_video", "ads2v"].includes(mode) ? (
              <AssetPickerField
                assets={imageAssets}
                buttonLabel="Select images"
                changeLabel="Edit references"
                emptyLabel="No reference images selected"
                label="Reference images"
                multiple
                onChange={setReferenceAssetIds}
                values={referenceAssetIds}
              />
            ) : null}

            {mode === "animate_character" ? (
              <>
                <AssetPickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  emptyLabel="No driving video selected"
                  label="Driving video"
                  onChange={setSourceClipAssetId}
                  value={sourceClipAssetId}
                />
                {/* One character today; the worker reads the first reference. Multi-reference is
                    experimental and tracked separately (sc-5583), so this stays a single image. */}
                <AssetPickerField
                  assets={imageAssets}
                  buttonLabel="Select image"
                  changeLabel="Change character"
                  emptyLabel="No reference character selected"
                  label="Reference character"
                  onChange={(id) => setReferenceAssetIds(id ? [id] : [])}
                  value={referenceAssetIds[0] ?? ""}
                />
              </>
            ) : null}

            {mode === "replace_person" ? (
              <ReplacePersonPanel
                createPersonDetectionJob={createPersonDetectionJob}
                createPersonTrackJob={createPersonTrackJob}
                personReadiness={personReadiness}
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
                setModel={setModel}
              />
            ) : null}
          </div>
        ) : null}

          <div className="settings-bar">
            <div className="settings-bar-row">
              <label className="settings-field settings-field-model">
                Model
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {/* Models gated on the selected tab (sc-5716): show only models that serve the
                      active mode, falling back to the full available list if none do (a reduced
                      catalog) so the picker is never empty. */}
                  {(modelsForMode(mode).length ? modelsForMode(mode) : baseVideoModels).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
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
                <select onChange={(event) => setDuration(Number(event.target.value))} value={duration}>
                  {durationOptions.map((value) => (
                    <option key={value} value={value}>
                      {value}s
                    </option>
                  ))}
                </select>
              </label>
              <label className="settings-field">
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
              {selectedModel?.adapter === "ltx_video" ? (
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
              {["extend_clip", "video_bridge"].includes(mode) ? (
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
              {showTierPicker ? (
                <label
                  className="quant-tier-picker"
                  title="Switch which installed MLX quant tier generates. Higher precision uses more memory; switching a heavy tier reloads it before the next generation."
                >
                  Quant tier
                  <select onChange={(event) => handleTierChange(event.target.value)} value={quantTier}>
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
                  {tierHasMemoryRisk ? (
                    <span className="field-hint quant-tier-memory-note">
                      Higher MLX video tiers may run out of memory on long or high-resolution clips.
                      Your pick is honored.
                    </span>
                  ) : null}
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
                <input
                  min="1"
                  max="80"
                  disabled={lightningActive}
                  onChange={(event) => setStepsOverride(event.target.value)}
                  placeholder={lightningActive ? "4 (Lightning)" : String(stepsDefaultFromModel(selectedModel) ?? "")}
                  title={lightningActive ? "Governed by Lightning (fast 4-step). Turn Lightning off to set steps." : undefined}
                  type="number"
                  value={lightningActive ? "" : stepsOverride}
                />
              </label>
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
              {characterId ? (
                <div className="guidance-strip">
                  <strong>Character reference</strong>
                  <span>
                    Character and look are saved with the recipe; LTX image conditioning uses IC-LoRA when the selected preset includes one.
                  </span>
                </div>
              ) : null}
              {/* Save-as-preset folds into Advanced with the rest of the power-user
                  knobs, matching Image Studio. Gated to the presetable video modes. */}
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
              {/* Video upscale (super-resolve an existing clip) folds into Advanced — it
                  previously lived in the render rail this layout removes. It operates on a
                  selected existing asset, independent of the current generation payload. */}
              <VideoUpscalePanel
                createVideoUpscaleJob={createVideoUpscaleJob}
                macCapabilities={macCapabilities}
                onSubmitted={(job) => {
                  onLocalJobCreated(job);
                  onOpenQueue();
                }}
                selectedAsset={selectedAsset}
                videoAssets={videoAssets}
              />
            </div>
          </AdvancedSection>

          {/* Every reason Generate is dead — mode/preset/LoRA/worker problems — in one chip
              row, from the same summary that gates the button (sc-10650). Project, prompt
              and inputs are silent requirements: their empty fields show it. */}
          <ValidationSummary issues={videoValidity.surfaced} label="Generate errors" />

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
