import { useStudioAdvancedSettings } from "../hooks/useStudioAdvancedSettings.js";
import { useStudioPromptGuide } from "../hooks/useStudioPromptGuide.js";
import { StudioPromptGuide } from "../components/StudioFields.jsx";
import { VideoSourcePanel } from "./videoStudio/VideoSourcePanel.jsx";
import { VideoPromptComposer } from "./videoStudio/VideoPromptComposer.jsx";
import { VideoReviewPanel } from "./videoStudio/VideoReviewPanel.jsx";
import { VideoGenerationSettings } from "./videoStudio/VideoGenerationSettings.jsx";
import { VideoAdvancedPanel } from "./videoStudio/VideoAdvancedPanel.jsx";
import { useVideoSubmission } from "./videoStudio/useVideoSubmission.js";
import { useVideoPresetState } from "./videoStudio/useVideoPresetState.js";
import { useVideoTurboControls } from "./videoStudio/useVideoTurboControls.js";
import { useVideoModeSelection } from "./videoStudio/useVideoModeSelection.js";
import { useVideoRecipeReplay } from "./videoStudio/useVideoRecipeReplay.js";
import { useVideoReferences } from "./videoStudio/useVideoReferences.js";
import { useVideoExecutionTier } from "./videoStudio/useVideoExecutionTier.js";
import { useLtx25Controls } from "./videoStudio/useLtx25Controls.js";
import { useVideoModelSelection } from "./videoStudio/useVideoModelSelection.js";
import { DEFAULT_MOTION, ltxVideoModelId, ltx25VideoModelId, ltxIcLoraModelIds, ltxIcLoraRequiredModes, MAX_SCAIL2_REFERENCE_CHARACTERS } from "./videoStudio/videoTierHelpers.js";

import { useSamplerControls } from "../hooks/useSamplerControls.js";
import { VedaOutcome } from "../components/VedaOutcome.jsx";
import React, { useEffect, useMemo, useState } from "react";

import { AdvancedSection } from "../components/AdvancedSection.jsx";

import { WorkPanel } from "../components/WorkPanel.jsx";

import { StyledPromptPreview } from "../components/StyledPromptPreview.jsx";
import { styleTextForId } from "../data/styleCatalog.js";
import { composeStyledPrompt } from "../styleComposer.js";

import { finiteNumberOrUndefined, loraLooksLikeIcLora, composePreset } from "../presetUtils.js";
import { PresetStackPreview, useGenerationStudio } from "../components/generationStudio.jsx";

import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { videoGenerateValidation } from "../videoStudioValidation.js";
import { useValidation } from "../validation/useValidation.js";
import { ValidationSummary } from "../validation/Validation.jsx";

import { PROMPT_REFINE_MODEL_ID, WAN_A14B_LIGHTNING_MODEL_IDS } from "../constants.js";
import { DEFAULT_MAC_CAPABILITIES } from "../macGating.js";

import { loadStudioSettings } from "../hooks/useStudioSettings.js";

import { stepsMenuFromModel } from "../samplerOptions.js";
import { minStepsForModel, referenceCaps, referenceLimitError } from "../videoModelLimits.js";

import { restoreVedaSettings, vedaControlError } from "../vedaControls.js";

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
    modelCatalogStatus = "ready",
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
  const [veda, setVeda] = useState(() => restoreVedaSettings(saved));
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
    schedulerShift: rawSchedulerShift,
    setSchedulerShift,
    schedulerShiftExplicit,
    stepsOverride,
    setStepsOverride,
    guidanceOverride,
    setGuidanceOverride,
    toggleAdvanced
  } = useStudioAdvancedSettings(saved);
  const {
    model,
    setModel,
    expectedImportedModelRef,
    setExpectedImportedModelRef,
    replayReferenceResize,
    setReplayReferenceResize,
    setUserModel,
    macVideoModels,
    selectedModel,
    supportsLastOnly,
    acceptsLastOnly,
    importedLastOnly,
    macGating,
    candleGating,
    modelsForMode,
    modelReady,
    modelOffers,
    modelDownloadJobs
  } = useVideoModelSelection({
    videoModels,
  saved,
  setRecipeModelNotice,
  setRecipeTierRequest,
  macCapabilities,
  models,
  jobs
  });

  const {
    ltxPipeline,
    setLtxPipeline,
    distilledVariant,
    setDistilledVariant,
    transformerVariant,
    setTransformerVariant,
    vaeDecoder,
    setVaeDecoder,
    autoDuration,
    setAutoDuration,
    autoDurationMinSeconds,
    setAutoDurationMinSeconds,
    autoDurationMaxSeconds,
    setAutoDurationMaxSeconds,
    temporalUpsampleRounds,
    setTemporalUpsampleRounds,
    precision,
    setPrecision,
    enhancePrompt,
    setEnhancePrompt,
    textEncoderModel,
    setTextEncoderModel,
    ltx25Dev,
    ltx25AutoDurationEnabled,
    textEncoderOptions,
    supportsTextEncoderSelection,
    supportsStockPromptEnhancement,
    supportsPromptEnhancement,
    defaultTextEncoderId,
    selectedTextEncoderModel,
    selectedTextEncoderAvailable,
    ltxVideoCfg,
    setLtxVideoCfg,
    ltxVideoStg,
    setLtxVideoStg,
    ltxVideoRescale,
    setLtxVideoRescale,
    videoConditioningStrength,
    setVideoConditioningStrength,
    bridgeRightVideoConditioningStrength,
    setBridgeRightVideoConditioningStrength,
    showsLegacyLtxGuidanceControls
  } = useLtx25Controls({
    saved,
  model,
  selectedModel,
  macCapabilities
  });

  // Multi-reference SCAIL-2 needs a reference/mask pair per character. Keep this source-ready UI
  // behind the descriptor-derived manifest flag: the currently pinned engine descriptor does not
  // advertise the paired contract yet, so a normal catalog remains on the existing single picker
  // until the final inference-main pin makes the capability truthful.
  const scail2MultiReferenceEnabled =
    model === "scail2_14b" && selectedModel?.ui?.scail2MultiReference === true;

  // Prompt guide for the selected model; fall back to the generic video guide
  // when a model declares none, so the button is always useful (sc-1817).
  const { guideOpen, openGuide, closeGuide, promptGuide } = useStudioPromptGuide(selectedModel, {
    title: "Video Prompt Guide",
    path: "/prompt-guides/generic-video.md",
  });
  const [duration, setDuration] = useState(saved.duration ?? selectedModel?.defaults?.duration ?? 6);
  const [resolution, setResolution] = useState(saved.resolution ?? selectedModel?.defaults?.resolution ?? "768x512");
  const [fps, setFps] = useState(saved.fps ?? selectedModel?.defaults?.fps ?? 25);

  // Configurable sampler / scheduler (epic 1753). The Wan diffusers (torch)
  // adapter applies these; MLX-backed video paths advertise default-only via
  // mlx.limits and the picker hides itself there.

  // Lightning fast-4-step toggle for Wan2.2 A14B MoE (T2V + I2V) — epic 10043, sc-10048.
  // Default ON: the worker (sc-10047) reads `advanced.lightning` and, when on, derives the
  // 4-step / CFG-off distilled recipe; when off it honors the user's steps/guidance (or the
  // native multi-step CFG default). Only the two A14B engines honor it (see showLightning),
  // so the dense 5B and non-Wan models never see the control. Persisted per-workspace.
  const [lightning, setLightning] = useState(saved.lightning ?? true);

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
  const hasRestoredSource = Boolean(saved.sourceAssetId || saved.sourceClipAssetId || (supportsLastOnly(videoModels.find((entry) => entry.id === saved.model)) && saved.lastFrameAssetId));
  const restoredSourceAssetId = hasRestoredSource ? (saved.sourceAssetId ?? "") : "";
  const ambientSourceAssetId = !hasRestoredSource && ["image", "frame"].includes(selectedAsset?.type) ? selectedAsset.id : "";
  const [sourceAssetId, setSourceAssetId] = useState(restoredSourceAssetId || ambientSourceAssetId);
  // Where the current FIRST frame came from. "ambient" = the studio adopted the app's current
  // asset selection on the user's behalf (the seed above, or the selection sync below);
  // "explicit" = the user picked it here, or a recipe restored it. A last-only-capable IMPORTED
  // model must not condition on a first frame the user never chose: choosing the LAST frame frames
  // the request around it, and an ambient first frame would ride along as an invented first frame
  // (sc-26687 S8). The API accepts a last-only request, so the studio has to be able to send one.
  //
  // It is STATE, not a ref, so it is persisted with the snapshot and can never drift from the value
  // the request is built from. A snapshot written before this field existed restores as "explicit",
  // which is the behaviour that shipped.
  const [sourceAssetOrigin, setSourceAssetOrigin] = useState(
    // A snapshot that records the origin is authoritative. One that does NOT is not proof that the
    // user chose the frame: the studio's ambient seed is the common producer of a saved
    // sourceAssetId, so a snapshot without the origin is read as ambient. That only ever withholds
    // a frame from an IMPORTED last-only request, so no other model is affected — and the import
    // platform ships with this epic, so there is no such legacy state in the wild either.
    saved.sourceAssetOrigin === "explicit"
      ? "explicit"
      : (restoredSourceAssetId ? "ambient" : (ambientSourceAssetId ? "ambient" : "none")),
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
  // LTX-2.5 keeps guidance provider-owned on both packed variants. Only negative conditioning is
  // variant-specific: the distilled default has no negative branch, while dev consumes one through
  // its fixed native multimodal guider. Other catalog entries retain absent-means-true above.
  const schedulerShift = selectedModel?.video?.supportsSchedulerShift !== false && (schedulerShiftExplicit || Boolean(scheduler && scheduler !== "default"))
    ? rawSchedulerShift ?? selectedModel?.defaults?.schedulerShift ?? "" : "";
  const supportsGuidance = selectedModel?.video?.supportsGuidance !== false;
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
  const {
    turboSeededModels,
    turboVariants,
    activeTurboVariant,
    showTurbo,
    selectTurboVariant
  } = useVideoTurboControls({
    saved,
  selectedModel,
  compatibleLoras,
  selectedLoraIds,
  model,
  setSelectedLoraIds
  });

  const {
    activeBackend,
    tierPickerItems,
    showTierPicker,
    quantTier,
    tierSwitching,
    handleExecutionTierChange,
    executionTierBlockMessage,
    canStartNewGenerationFromReplay,
    startNewGenerationLabel,
    handleStartNewGenerationFromReplay,
    showTorchQuantization,
    selectedTierQuantize,
    tierHasMemoryRisk
  } = useVideoExecutionTier({
    macGating,
  selectedModel,
  model,
  preferencesHydrated,
  recipeTierRequest,
  videoModels,
  macVideoModels,
  mode,
  macCapabilities,
  setModel,
  setRecipeTierRequest,
  setRecipeModelNotice,
  supportsQuantization
  });

  const { samplerOptions, schedulerOptions, showSchedulerPicker } = useSamplerControls(selectedModel, activeBackend, { sampler, scheduler, setSampler, setScheduler });

  const requiresLtxIcLora = ltxIcLoraModelIds.has(selectedModel?.id) && ltxIcLoraRequiredModes.has(mode);
  const hasLtxIcLora = selectedLoras.some((lora) => loraLooksLikeIcLora(lora));

  const {
    lastI2vAssetIdRef
  } = useVideoReferences({
    hasRestoredSource,
  selectedAssetId,
  selectedAsset,
  setSourceAssetOrigin,
  setSourceAssetId,
  setSourceClipAssetId,
  launchRequest,
  setMode,
  setModel,
  setSelectedPresetId,
  setCharacterId,
  setCharacterLookId,
  assets,
  setLastFrameAssetId,
  setBridgeRightClipAssetId,
  setReferenceClipAssetId,
  setReferenceAssetIds,
  setSourceClipAssetIds,
  personTracks,
  setPersonTrackId,
  selectedModel,
  setDuration,
  setResolution,
  setFps,
  sourceAssetId,
  lastFrameAssetId,
  mode
  });

  useVideoRecipeReplay({
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
  });

  useVideoModeSelection({
    selectedModel,
  mode,
  macCapabilities,
  modelsForMode,
  model,
  setModel,
  videoModels
  });

  const {
    presetName,
    setPresetName,
    presetScope,
    setPresetScope,
    savingPreset,
    presetSaveMessage,
    setPresetSaveMessage,
    handleSaveAsPreset
  } = useVideoPresetState({
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
  setPrompt,
  setNegativePrompt,
  setResolution,
  setDuration,
  setFps,
  setQuality,
  setGuidanceOverride,
  setStepsOverride,
  setVeda,
  setSampler,
  setScheduler,
  setSchedulerShift,
  setPrecision,
  setQuantization,
  setLtxPipeline,
  setDistilledVariant,
  setTransformerVariant,
  setVaeDecoder,
  setAutoDuration,
  setAutoDurationMinSeconds,
  setAutoDurationMaxSeconds,
  setTemporalUpsampleRounds,
  setEnhancePrompt,
  setTextEncoderModel,
  setMotion,
  setLtxVideoCfg,
  setLtxVideoStg,
  setLtxVideoRescale,
  prompt,
  negativePrompt,
  resolution,
  duration,
  fps,
  quality,
  guidanceOverride,
  veda,
  stepsOverride,
  sampler,
  scheduler,
  schedulerShift,
  schedulerShiftExplicit,
  precision,
  quantization,
  ltxPipeline,
  distilledVariant,
  transformerVariant,
  vaeDecoder,
  autoDuration,
  autoDurationMinSeconds,
  autoDurationMaxSeconds,
  temporalUpsampleRounds,
  supportsPromptEnhancement,
  enhancePrompt,
  supportsTextEncoderSelection,
  selectedTextEncoderModel,
  motion,
  showsLegacyLtxGuidanceControls,
  ltxVideoCfg,
  ltxVideoStg,
  ltxVideoRescale,
  styleId,
  textEncoderModel,
  advancedOpen,
  selectedLoraIds,
  loraWeights,
  seed,
  selectedPresetId,
  generalStackIds,
  lightning,
  turboSeededModels,
  videoConditioningStrength,
  bridgeRightVideoConditioningStrength,
  fitMode,
  sourceAssetId,
  sourceAssetOrigin,
  expectedImportedModelRef,
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
  preferencesHydrated,
  videoModels
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
    (mode === "image_to_video" && (sourceAssetId || (acceptsLastOnly && lastFrameAssetId))) ||
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
  const vedaSteps = finiteNumberOrUndefined(stepsOverride) ?? activeTurboVariant?.sampling?.steps ?? selectedModel?.defaults?.steps ?? 50;
  const vedaBlockMessage = vedaControlError(showTurbo, veda);
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
      vedaBlockMessage,
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
      vedaBlockMessage,
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

  const {
    submit
  } = useVideoSubmission({
    submitting,
  executionTierBlockMessage,
  setSubmitting,
  stackActive,
  composedStack,
  createVideoJob,
  mode,
  styleApplied,
  composedStylePrompt,
  prompt,
  supportsNegativePrompt,
  negativePrompt,
  model,
  expectedImportedModelRef,
  ltx25AutoDurationEnabled,
  duration,
  fps,
  width,
  height,
  quality,
  seed,
  selectedPreset,
  characterId,
  characterLookId,
  importedLastOnly,
  sourceAssetOrigin,
  lastFrameAssetId,
  sourceAssetId,
  fitMode,
  acceptsLastOnly,
  sourceClipAssetId,
  outgoingSourceClipAssetIds,
  bridgeRightClipAssetId,
  outgoingReferenceAssetIds,
  outgoingReferenceAudioAssetIds,
  referenceClipAssetId,
  personTrackId,
  replacementMode,
  selectedLoras,
  effectiveLoraWeight,
  veda,
  resolution,
  durationHint,
  motion,
  selectedTrack,
  replacementModeLabels,
  styleId,
  stylePromptBase,
  replayReferenceResize,
  selectedModel,
  ltxPipeline,
  distilledVariant,
  precision,
  transformerVariant,
  vaeDecoder,
  autoDuration,
  autoDurationMinSeconds,
  autoDurationMaxSeconds,
  temporalUpsampleRounds,
  supportsPromptEnhancement,
  enhancePrompt,
  supportsTextEncoderSelection,
  selectedTextEncoderModel,
  defaultTextEncoderId,
  showTorchQuantization,
  quantization,
  selectedTierQuantize,
  sampler,
  scheduler,
  schedulerShift,
  showLightning,
  lightning,
  lightningActive,
  stepsPinned,
  stepsOffMenu,
  stepsOverride,
  supportsGuidance,
  guidanceOverride,
  showsLegacyLtxGuidanceControls,
  ltxVideoCfg,
  ltxVideoStg,
  ltxVideoRescale,
  showClipStrength,
  videoConditioningStrength,
  bridgeRightVideoConditioningStrength,
  onLocalJobCreated
  });

  const generateDisabled = submitting || !videoValidity.ready;
  const renderLabel = mode === "replace_person" ? "Replace person" : "Render clip";

  return (
    <ModelAvailabilityGate
      ready={modelReady}
      initializing={modelCatalogStatus === "idle" || modelCatalogStatus === "loading"}
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
          <VideoPromptComposer
            modeOptions={modeOptions}
            mode={mode}
            setMode={setMode}
            modeTabBlocked={modeTabBlocked}
            modeTabBlockedText={modeTabBlockedText}
            openGuide={openGuide}
            onOpenPresets={onOpenPresets}
            setPrompt={setPrompt}
            promptless={promptless}
            prompt={prompt}
            generateDisabled={generateDisabled}
            submitting={submitting}
            renderLabel={renderLabel}
            promptGuide={promptGuide}
            model={model}
            refinePrompt={refinePrompt}
            refineModel={refineModel}
            createModelDownloadJob={createModelDownloadJob}
            motion={motion}
            setMotion={setMotion}
          />

          <VideoSourcePanel
            mode={mode}
            imageAssets={imageAssets}
            characters={characters}
            importAsset={importAsset}
            setSourceAssetOrigin={setSourceAssetOrigin}
            setSourceAssetId={setSourceAssetId}
            activeProject={activeProject}
            sourceAssetId={sourceAssetId}
            acceptsLastOnly={acceptsLastOnly}
            setLastFrameAssetId={setLastFrameAssetId}
            importedLastOnly={importedLastOnly}
            sourceAssetOrigin={sourceAssetOrigin}
            lastFrameAssetId={lastFrameAssetId}
            fitMode={fitMode}
            setFitMode={setFitMode}
            videoAssets={videoAssets}
            setSourceClipAssetId={setSourceClipAssetId}
            sourceClipAssetId={sourceClipAssetId}
            setBridgeRightClipAssetId={setBridgeRightClipAssetId}
            bridgeRightClipAssetId={bridgeRightClipAssetId}
            setSourceClipAssetIds={setSourceClipAssetIds}
            sourceClipAssetIds={sourceClipAssetIds}
            setReferenceClipAssetId={setReferenceClipAssetId}
            referenceClipAssetId={referenceClipAssetId}
            setReferenceAssetIds={setReferenceAssetIds}
            referenceAssetIds={referenceAssetIds}
            showReferenceClips={showReferenceClips}
            refCaps={refCaps}
            showAudioReferences={showAudioReferences}
            audioAssets={audioAssets}
            setReferenceAudioAssetIds={setReferenceAudioAssetIds}
            referenceAudioAssetIds={referenceAudioAssetIds}
            scail2MultiReferenceEnabled={scail2MultiReferenceEnabled}
            scail2ReferenceOverflow={scail2ReferenceOverflow}
            createPersonDetectionJob={createPersonDetectionJob}
            createPersonTrackJob={createPersonTrackJob}
            createModelDownloadJob={createModelDownloadJob}
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
            setTrackName={setTrackName}
            trackName={trackName}
            videoModels={videoModels}
            model={model}
            setUserModel={setUserModel}
          />

          <VideoGenerationSettings
            selectedModel={selectedModel}
            setUserModel={setUserModel}
            model={model}
            modelsForMode={modelsForMode}
            mode={mode}
            macVideoModels={macVideoModels}
            createModelDownloadJob={createModelDownloadJob}
            recipeModelNotice={recipeModelNotice}
            setResolution={setResolution}
            resolution={resolution}
            resolutionOptions={resolutionOptions}
            setDuration={setDuration}
            setAutoDuration={setAutoDuration}
            duration={duration}
            durationOptions={durationOptions}
            fps={fps}
            showTierPicker={showTierPicker}
            quantTier={quantTier}
            handleExecutionTierChange={handleExecutionTierChange}
            tierPickerItems={tierPickerItems}
            tierSwitching={tierSwitching}
            tierHasMemoryRisk={tierHasMemoryRisk}
            selectedLoras={selectedLoras}
            selectedLoraIds={selectedLoraIds}
            compatibleLoras={compatibleLoras}
            userSelectedLoraCount={userSelectedLoraCount}
            toggleLora={toggleLora}
            effectiveLoraWeight={effectiveLoraWeight}
            setLoraWeight={setLoraWeight}
            loraEmptyMessage={loraEmptyMessage}
            createLoraDownloadJob={createLoraDownloadJob}
            activeProject={activeProject}
            createLoraImportJob={createLoraImportJob}
            models={models}
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
            onToggle={toggleAdvanced}
            open={advancedOpen}
          >
            <VideoAdvancedPanel
            showTurbo={showTurbo}
            veda={veda}
            setVeda={setVeda}
            vedaSteps={vedaSteps}
            setFps={setFps}
            fps={fps}
            fpsOptions={fpsOptions}
            showLightning={showLightning}
            lightning={lightning}
            setLightning={setLightning}
            turboVariants={turboVariants}
            selectTurboVariant={selectTurboVariant}
            activeTurboVariant={activeTurboVariant}
            selectedModel={selectedModel}
            model={model}
            setLtxPipeline={setLtxPipeline}
            ltxPipeline={ltxPipeline}
            setDistilledVariant={setDistilledVariant}
            distilledVariant={distilledVariant}
            setPrecision={setPrecision}
            precision={precision}
            setTransformerVariant={setTransformerVariant}
            transformerVariant={transformerVariant}
            ltx25Dev={ltx25Dev}
            setVaeDecoder={setVaeDecoder}
            vaeDecoder={vaeDecoder}
            autoDuration={autoDuration}
            setAutoDuration={setAutoDuration}
            setAutoDurationMinSeconds={setAutoDurationMinSeconds}
            setAutoDurationMaxSeconds={setAutoDurationMaxSeconds}
            autoDurationMinSeconds={autoDurationMinSeconds}
            autoDurationMaxSeconds={autoDurationMaxSeconds}
            setTemporalUpsampleRounds={setTemporalUpsampleRounds}
            temporalUpsampleRounds={temporalUpsampleRounds}
            supportsPromptEnhancement={supportsPromptEnhancement}
            enhancePrompt={enhancePrompt}
            setEnhancePrompt={setEnhancePrompt}
            supportsStockPromptEnhancement={supportsStockPromptEnhancement}
            supportsTextEncoderSelection={supportsTextEncoderSelection}
            setTextEncoderModel={setTextEncoderModel}
            selectedTextEncoderModel={selectedTextEncoderModel}
            textEncoderOptions={textEncoderOptions}
            selectedTextEncoderAvailable={selectedTextEncoderAvailable}
            defaultTextEncoderId={defaultTextEncoderId}
            showsLegacyLtxGuidanceControls={showsLegacyLtxGuidanceControls}
            setLtxVideoCfg={setLtxVideoCfg}
            ltxVideoCfg={ltxVideoCfg}
            setLtxVideoStg={setLtxVideoStg}
            ltxVideoStg={ltxVideoStg}
            setLtxVideoRescale={setLtxVideoRescale}
            ltxVideoRescale={ltxVideoRescale}
            showClipStrength={showClipStrength}
            mode={mode}
            setVideoConditioningStrength={setVideoConditioningStrength}
            videoConditioningStrength={videoConditioningStrength}
            setBridgeRightVideoConditioningStrength={setBridgeRightVideoConditioningStrength}
            bridgeRightVideoConditioningStrength={bridgeRightVideoConditioningStrength}
            showQualitySegment={showQualitySegment}
            quality={quality}
            setQuality={setQuality}
            showTorchQuantization={showTorchQuantization}
            setQuantization={setQuantization}
            quantization={quantization}
            quantVariants={quantVariants}
            setRequestedGpu={setRequestedGpu}
            requestedGpu={requestedGpu}
            gpuOptions={gpuOptions}
            seed={seed}
            setSeed={setSeed}
            sampler={sampler}
            scheduler={scheduler}
            samplerOptions={samplerOptions}
            schedulerOptions={schedulerOptions}
            setSampler={setSampler}
            setScheduler={setScheduler}
            showSchedulerPicker={showSchedulerPicker}
            setSchedulerShift={setSchedulerShift}
            schedulerShift={schedulerShift}
            stepsChoice={stepsChoice}
            lightningActive={lightningActive}
            stepsOffMenu={stepsOffMenu}
            stepsOverride={stepsOverride}
            stepsPinned={stepsPinned}
            setStepsOverride={setStepsOverride}
            minSteps={minSteps}
            stepsPinnedValue={stepsPinnedValue}
            supportsGuidance={supportsGuidance}
            setGuidanceOverride={setGuidanceOverride}
            guidanceOverride={guidanceOverride}
            setCharacterId={setCharacterId}
            characterId={characterId}
            characters={characters}
            setCharacterLookId={setCharacterLookId}
            characterLookId={characterLookId}
            supportsNegativePrompt={supportsNegativePrompt}
            setNegativePrompt={setNegativePrompt}
            negativePrompt={negativePrompt}
            createVideoUpscaleJob={createVideoUpscaleJob}
            importAsset={importAsset}
            macCapabilities={macCapabilities}
            onLocalJobCreated={onLocalJobCreated}
            onOpenQueue={onOpenQueue}
            selectedAsset={selectedAsset}
            activeProject={activeProject}
            videoAssets={videoAssets}
          />
          </AdvancedSection>

          {/* Every reason Generate is dead — mode/preset/LoRA/worker problems — in one chip
              row, from the same summary that gates the button (sc-10650). Project, prompt
              and inputs are silent requirements: their empty fields show it. */}
          <ValidationSummary issues={videoValidity.surfaced} label="Generate errors" />
          <VedaOutcome
            label="Recorded recipe attention"
            raw={launchRequest?.recipe?.rawAdapterSettings}
          />
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

        <VideoReviewPanel
            localJobs={localJobs}
            assets={assets}
            onPreview={onPreview}
            onCancelJob={onCancelJob}
            onOpenQueue={onOpenQueue}
            latestAssets={latestAssets}
            deleteAsset={deleteAsset}
            purgeAsset={purgeAsset}
            updateAssetStatus={updateAssetStatus}
            comparisonAsset={comparisonAsset}
            comparisonSource={comparisonSource}
            comparisonMode={comparisonMode}
            setComparisonMode={setComparisonMode}
            abSide={abSide}
            setAbSide={setAbSide}
          />
      </form>
      <StudioPromptGuide open={guideOpen} guide={promptGuide} modelName={selectedModel?.name} onClose={closeGuide} />
    </section>
    </ModelAvailabilityGate>
  );
}
