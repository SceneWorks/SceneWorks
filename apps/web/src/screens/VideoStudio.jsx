import React, { useEffect, useMemo, useRef, useState } from "react";
import { pickClosestResolution } from "../resolutionMatch.js";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
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

function formatGpuLabel(requestedGpu) {
  if (!requestedGpu || requestedGpu === "auto") {
    return "auto GPU";
  }
  return `GPU ${requestedGpu}`;
}

function estimateRenderSeconds(durationSeconds, quality) {
  // Rough heuristic: every clip second ~3s on Balanced, ±50% for Draft/Final.
  const base = Math.max(1, Number(durationSeconds) || 6) * 3;
  if (quality === "fast") return Math.round(base * 0.5);
  if (quality === "best") return Math.round(base * 1.5);
  return Math.round(base);
}

function formatPlaybackTime(seconds) {
  const safeSeconds = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  return `${minutes}:${String(safeSeconds % 60).padStart(2, "0")}`;
}

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
import { ReplacePersonPanel } from "./ReplacePersonPanel.jsx";
import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
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
    setSelectedAssetId,
    setPreviewAsset,
    personTracks = [],
    personReadiness = {},
    presets = [],
    requestedGpu,
    saveTrackCorrections,
    selectedAsset,
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
  const onSendToEditor = (asset) => {
    if (asset?.id) {
      setSelectedAssetId(asset.id);
    }
    setActiveView("Editor");
  };
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
  const [sourceAssetId, setSourceAssetId] = useState(["image", "frame"].includes(selectedAsset?.type) ? selectedAsset.id : "");
  // How the starting image is fitted to the output resolution for the image-conditioned
  // modes (sc-6139), mirroring Image Studio Edit. Crop/Pad only — video has no inpaint
  // mask, so Outpaint is hidden (`inpaintCapable={false}`). Default crop = fill, not stretch.
  const [fitMode, setFitMode] = useState(saved.fitMode ?? "crop");
  const [lastFrameAssetId, setLastFrameAssetId] = useState("");
  const [sourceClipAssetId, setSourceClipAssetId] = useState(selectedAsset?.type === "video" ? selectedAsset.id : "");
  const [bridgeRightClipAssetId, setBridgeRightClipAssetId] = useState("");
  // Subject reference images for Bernini's reference-driven video modes
  // (reference_to_video / reference_video_to_video / ads2v, sc-4703 / sc-5425). 1–N images.
  const [referenceAssetIds, setReferenceAssetIds] = useState([]);
  // Multiple source clips for Bernini's multi-source-video edit (multi_video_to_video, sc-5425).
  const [sourceClipAssetIds, setSourceClipAssetIds] = useState([]);
  // Reference video for Bernini's ads2v mode (sc-5425): a second source clip distinct from the
  // edited source clip (sourceClipAssetId).
  const [referenceClipAssetId, setReferenceClipAssetId] = useState("");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  const [personTrackId, setPersonTrackId] = useState("");
  const [replacementMode, setReplacementMode] = useState("face_only");
  const [selectedDetectionId, setSelectedDetectionId] = useState("");
  const [trackName, setTrackName] = useState("Selected person");
  const [comparisonMode, setComparisonMode] = useState("side_by_side");
  const [abSide, setAbSide] = useState("replacement");
  const [submitting, setSubmitting] = useState(false);
  const previewVideoRef = useRef(null);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewTime, setPreviewTime] = useState(0);
  const [previewDuration, setPreviewDuration] = useState(0);
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
  });
  // Sampler / scheduler menus declared by the model. Video Wan torch
  // declares the full menu; sealed paths (LTX native, MLX) drop to
  // default-only and the picker hides. Gated to the ACTIVE backend (epic 7114 P5):
  // `macGating` is the worker `mlx_required` master switch, so the menu reflects the
  // manifest's `mlx.limits` override on Mac/MLX and `candle.limits` on the candle build.
  const activeBackend = macGating ? "mlx" : "candle";
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
  useEffect(() => {
    if (samplerOptions.includes(sampler)) {
      return;
    }
    const preferred = samplerOptions.includes(samplerDefaultFromModel(selectedModel))
      ? samplerDefaultFromModel(selectedModel)
      : samplerOptions[0];
    setSampler(preferred);
  }, [samplerOptions, sampler, selectedModel]);
  useEffect(() => {
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

  useEffect(() => {
    if (selectedAsset?.type === "image" || selectedAsset?.type === "frame") {
      setSourceAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
  }, [selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (launchRequest?.view !== "Video") {
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
  });

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
  const canSubmit = Boolean(
    activeProject &&
      (promptless || prompt.trim()) &&
      supportsMode &&
      implementedMode &&
      hasInputs &&
      presetValidationResult.ok &&
      selectedLoraValidationResult.ok &&
      (!requiresLtxIcLora || hasLtxIcLora) &&
      replaceReady,
  );
  const [width, height] = resolution.split("x").map((value) => Number(value));
  const durationOptions = selectedModel?.limits?.durations ?? [4, 6, 8, 10];
  const resolutionOptions = selectedModel?.limits?.resolutions ?? ["768x512", "640x640", "1280x720", "720x1280"];
  const fpsOptions = selectedModel?.limits?.fps ?? [24, 25, 30];
  const durationHint =
    selectedModel?.ui?.durationHint ??
    (selectedModel?.limits?.recommendedMaxDuration ? `Recommended: ${selectedModel.limits.recommendedMaxDuration}s or less.` : "");
  const blockedMessage = !supportsMode
    ? `${selectedModel?.name ?? "Selected model"} does not support this mode.`
    : !implementedMode
      ? "This entry point is reserved for the next runtime slice."
      : !hasInputs
        ? "Required inputs are missing."
        : requiresLtxIcLora && !hasLtxIcLora
          ? "LTX video-conditioned generation needs an installed IC-LoRA preset."
          : !replaceReady
            ? "No live GPU worker can run person replacement yet."
            : "";
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
      const job = await createVideoJob({
        mode,
        prompt,
        negativePrompt,
        model,
        duration: Number(duration),
        fps: Number(fps),
        width,
        height,
        quality,
        seed: seed === "" ? null : Number(seed),
        recipePresetId: selectedPreset?.id ?? null,
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
          ...(supportsQuantization && quantization !== "auto" ? { quantization } : {}),
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

  const generateDisabled = submitting || !canSubmit;
  const renderLabel = mode === "replace_person" ? "Replace person" : "Render clip";
  const previewAsset = latestAssets[0] ?? null;
  const estimateSeconds = estimateRenderSeconds(duration, quality);
  const gpuLabel = formatGpuLabel(requestedGpu);
  const previewCanPlay = assetCanRenderAsVideo(previewAsset);
  const previewTotalSeconds = previewDuration || Number(previewAsset?.file?.duration) || Number(duration) || 0;
  const previewProgress = previewTotalSeconds ? `${Math.min(100, (previewTime / previewTotalSeconds) * 100)}%` : "0%";

  useEffect(() => {
    setPreviewPlaying(false);
    setPreviewTime(0);
    setPreviewDuration(0);
  }, [previewAsset?.id]);

  function togglePreviewPlayback() {
    const video = previewVideoRef.current;
    if (!video || !previewCanPlay) {
      return;
    }
    if (video.paused) {
      video.play().catch(() => setPreviewPlaying(false));
      return;
    }
    video.pause();
  }

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
    <section className="main-surface video-studio">
      <form className="studio-shell" onSubmit={submit}>
        <div className="surface-header hero studio-prompt-hero video-prompt-hero">
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

        <div className="video-results">
          <div className="video-main-stack">
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

            <div className="video-preview-card">
              <div className="video-preview-stage">
                {previewAsset ? (
                  <AssetMedia
                    asset={previewAsset}
                    controls={false}
                    onEnded={() => setPreviewPlaying(false)}
                    onLoadedMetadata={(event) => setPreviewDuration(event.currentTarget.duration || 0)}
                    onPause={() => setPreviewPlaying(false)}
                    onPlay={() => setPreviewPlaying(true)}
                    onTimeUpdate={(event) => setPreviewTime(event.currentTarget.currentTime || 0)}
                    ref={previewVideoRef}
                  />
                ) : (
                  <span className="video-preview-empty">No clip rendered yet — set up the prompt above and hit Render</span>
                )}
              </div>

              <div className="video-playback-bar">
                <button
                  aria-label={previewPlaying ? "Pause preview" : "Play preview"}
                  className="play-btn"
                  disabled={!previewCanPlay}
                  onClick={togglePreviewPlayback}
                  type="button"
                >
                  {previewPlaying ? <Icon.Pause size={14} /> : <Icon.Play size={14} />}
                </button>
                <div aria-hidden="true" className="video-playback-scrub">
                  <span className="video-playback-scrub-fill" style={{ width: previewProgress }} />
                </div>
                <span className="video-playback-time">
                  {formatPlaybackTime(previewTime)} / {formatPlaybackTime(previewTotalSeconds)}
                </span>
                <span className="video-playback-estimate">~{estimateSeconds}s on {gpuLabel}</span>
                <button
                  className="send-editor-btn"
                  disabled={!previewAsset || !onSendToEditor}
                  onClick={() => previewAsset && onSendToEditor?.(previewAsset)}
                  type="button"
                >
                  <Icon.Editor size={14} /> Send to editor
                </button>
              </div>
            </div>

            {videoAssets.length ? (
              <div className="recent-clips-card">
                <div className="recent-clips-head">
                  <h3>Recent clips</h3>
                  <span className="meta">{localJobs.length || latestAssets.length} this session</span>
                </div>
                <div className="recent-clips-strip">
                  {videoAssets.slice(0, 4).map((asset) => (
                    <button className="tray-item" key={asset.id} onClick={() => onPreview(asset, videoAssets.slice(0, 4))} type="button">
                      <AssetMedia asset={asset} />
                      <span>{asset.displayName}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

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

            {latestAssets.length > 1 ? (
              <div className="review-grid video-review-grid">
                {latestAssets.slice(1).map((asset) => (
                  <AssetCard
                    asset={asset}
                    deleteAsset={deleteAsset}
                    key={asset.id}
                    onPreview={(previewed) => onPreview(previewed, latestAssets.slice(1))}
                    purgeAsset={purgeAsset}
                    updateAssetStatus={updateAssetStatus}
                  />
                ))}
              </div>
            ) : null}

            {blockedMessage ? <p className="inline-warning">{blockedMessage}</p> : null}
            <PresetValidationWarnings presetValidationResult={presetValidationResult} selectedModel={selectedModel} />
            {selectedLoraValidationResult.incompatible.length ? (
              <p className="inline-warning">
                Generate is blocked because these selected LoRAs are incompatible with {selectedModel?.name ?? "the selected model"}: {selectedLoraValidationResult.incompatible.join(", ")}.
              </p>
            ) : null}
          </div>

          <div className="video-rail">
            <aside className="render-rail">
              <div className="preset-rail-head">
                <h3>Render settings</h3>
                <span className="preset-rail-model-tag">{selectedModel?.name ?? "—"}</span>
              </div>

              <label>
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

              <div className="style-preset-strip">
                <span className="style-preset-label">Style preset</span>
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

              <label>
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

              <div className="control-grid preset-rail-row">
                <label>
                  Duration
                  <select onChange={(event) => setDuration(Number(event.target.value))} value={duration}>
                    {durationOptions.map((value) => (
                      <option key={value} value={value}>
                        {value}s
                      </option>
                    ))}
                  </select>
                </label>
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
              </div>

              <label>
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((value) => (
                    <option key={value} value={value}>
                      {value.replace("x", " × ")}
                    </option>
                  ))}
                </select>
              </label>

              <PresetGuidanceStrip
                selectedPreset={selectedPreset}
                presetPromptParts={presetPromptParts}
                presetLoraDetails={presetLoraDetails}
              />

              {durationHint ? <p className="helper-copy">{durationHint}</p> : null}

              <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
                <Icon.ChevDown className={advancedOpen ? "chev-rotate open" : "chev-rotate"} size={14} />
                {advancedOpen ? "Hide advanced" : "Advanced"}
              </button>

              {advancedOpen ? (
                <div className="advanced-panel">
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
                  {supportsQuantization ? (
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
                </div>
              ) : null}
            </aside>

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

            <aside className="tips-card">
              <h3>Tips</h3>
              <ul>
                <li>Short clips (4–6s) compose better in the editor.</li>
                <li>Describe the motion, not just the scene.</li>
                <li>Pick a motion chip above to guide the camera.</li>
              </ul>
            </aside>

            <aside className="keyboard-card">
              <h3>Keyboard</h3>
              <dl>
                <div className="kbd-row">
                  <span>Render</span>
                  <span className="kbd-keys">
                    <kbd>⌘</kbd>
                    <kbd>↵</kbd>
                  </span>
                </div>
              </dl>
            </aside>
          </div>
        </div>
      </form>
      {guideOpen ? (
        <PromptGuideModal guide={promptGuide} modelName={selectedModel?.name} onClose={() => setGuideOpen(false)} />
      ) : null}
    </section>
    </ModelAvailabilityGate>
  );
}
