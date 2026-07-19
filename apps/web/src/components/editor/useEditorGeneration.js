import { useEffect, useMemo, useState } from "react";
import { qualityChoices } from "../../jobTypes.js";
import {
  samplerOptionsFromModel,
  schedulerOptionsFromModel,
  samplerDefaultFromModel,
  schedulerDefaultFromModel,
  schedulerShiftDefaultFromModel,
} from "../../samplerOptions.js";
import { installedTiers, tierQuantize } from "../../quantTier.js";
import { serializeLora, buildStudioPresetPayload } from "../../presetUtils.js";
import { WAN_A14B_LIGHTNING_MODEL_IDS } from "../../constants.js";
import { useGenerationStudio } from "../../screens/generationStudio.jsx";
import { loadStudioSettings, useStudioSettingsWriter } from "../../hooks/useStudioSettings.js";
import { MOTIONS, DEFAULT_MOTION } from "./editorUtils.js";

const LTX_VIDEO_MODEL_ID = "ltx_2_3";
const EDITOR_DEFAULT_TIER = "q4";
// Preset filtering is by workflow/mode; the editor drives multiple video modes, so it
// borrows the most common video preset workflow for the availability filter.
const PRESET_FILTER_MODE = "image_to_video";

const RESOLUTION_FALLBACK = ["1280x720", "768x512", "768x768", "720x1280", "1024x576", "512x768"];
const FPS_FALLBACK = [24, 25, 30];
const DURATION_FALLBACK = [4, 6, 8, 10];

// Owns the editor's generation-settings object and reuses the Video Studio machinery
// (useGenerationStudio for presets + LoRA selection, plus the model-derived option
// helpers and tier logic) so a clip generated from the editor behaves identically to
// one generated in Video Studio (epic 12798). Returns everything GenerationRail needs
// to render, plus buildBasePayload() which the screen merges with the timeline context.
export function useEditorGeneration({ context }) {
  const {
    activeProject,
    videoModels = [],
    models = [],
    loras = [],
    presets = [],
    assets = [],
    characters = [],
    recentVideoAssets = [],
    createPreset,
    createModelDownloadJob,
    createLoraDownloadJob,
  } = context;

  const projectId = activeProject?.id ?? null;
  // Seed from the editor's own scope, falling back to the Video Studio snapshot so the
  // editor opens with the user's last studio choices but never clobbers that snapshot.
  const saved = useMemo(() => {
    const studio = loadStudioSettings("video", projectId);
    const own = loadStudioSettings("editor-video", projectId);
    return { ...studio, ...own };
  }, [projectId]);

  const fallbackModelId = videoModels[0]?.id ?? LTX_VIDEO_MODEL_ID;
  const [model, setModel] = useState(saved.model ?? fallbackModelId);
  const [quality, setQuality] = useState(saved.quality ?? "balanced");
  const [resolution, setResolution] = useState(saved.resolution ?? "1280x720");
  const [fps, setFps] = useState(saved.fps ?? 24);
  const [duration, setDuration] = useState(saved.duration ?? 6);
  const [motion, setMotion] = useState(MOTIONS.includes(saved.motion) ? saved.motion : DEFAULT_MOTION);
  const [seed, setSeed] = useState(saved.seed ?? "");
  const [prompt, setPrompt] = useState(saved.prompt ?? "Continue the action, matching motion and lighting");
  const [negativePrompt, setNegativePrompt] = useState(saved.negativePrompt ?? "");
  const [sampler, setSampler] = useState(saved.sampler ?? "default");
  const [scheduler, setScheduler] = useState(saved.scheduler ?? "default");
  const [schedulerShift, setSchedulerShift] = useState(saved.schedulerShift ?? 3.0);
  const [stepsOverride, setStepsOverride] = useState(saved.steps ?? "");
  const [guidanceOverride, setGuidanceOverride] = useState(saved.guidanceScale ?? "");
  const [quantTier, setQuantTier] = useState(saved.quantTier ?? EDITOR_DEFAULT_TIER);
  const [lightning, setLightning] = useState(saved.lightning ?? true);
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);

  const selectedModel = useMemo(
    () => videoModels.find((item) => item.id === model) ?? videoModels[0] ?? null,
    [videoModels, model],
  );

  const studio = useGenerationStudio({
    mode: PRESET_FILTER_MODE,
    presets,
    selectedModel,
    loras,
    models: videoModels.length ? videoModels : models,
    model,
    setModel,
    fallbackModelId,
    characters,
    characterId: "",
    setCharacterId: () => {},
    setCharacterLookId: () => {},
    assets,
    latestAssets: recentVideoAssets,
    trackedLocalJobs: [],
    initialPresetId: saved.selectedPresetId ?? null,
    advancedOpen,
    setAdvancedOpen,
    initialSelectedLoraIds: saved.selectedLoraIds ?? [],
    initialLoraWeights: saved.loraWeights ?? {},
    initialShowIncompatibleLoras: saved.showIncompatibleLoras ?? false,
    initialGeneralStackIds: saved.generalStackIds ?? [],
  });

  // Option menus derived from the selected model (null backend → base limits menu).
  const resolutionOptions = selectedModel?.limits?.resolutions ?? RESOLUTION_FALLBACK;
  const fpsOptions = selectedModel?.limits?.fps ?? FPS_FALLBACK;
  const durationOptions = selectedModel?.limits?.durations ?? DURATION_FALLBACK;
  const samplerOptions = useMemo(() => samplerOptionsFromModel(selectedModel, null), [selectedModel]);
  const schedulerOptions = useMemo(() => schedulerOptionsFromModel(selectedModel, null), [selectedModel]);
  const availableTiers = useMemo(
    () => installedTiers(selectedModel, { convRotEligible: false, nvfp4Eligible: false, defaultQuality: EDITOR_DEFAULT_TIER }),
    [selectedModel],
  );

  const showSamplerPicker = samplerOptions.length > 1;
  const showSchedulerPicker = schedulerOptions.length > 1;
  const showTierPicker = availableTiers.length > 1;
  const showLightning = WAN_A14B_LIGHTNING_MODEL_IDS.has(model);
  const lightningActive = showLightning && lightning;

  // Re-clamp the discrete selects to the current model's menus when the model changes,
  // and reset sampler/scheduler defaults + tier into range (mirrors VideoStudio's clamp).
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (!resolutionOptions.includes(resolution)) {
      setResolution(selectedModel.defaults?.resolution ?? resolutionOptions[0]);
    }
    if (!fpsOptions.includes(Number(fps))) {
      setFps(selectedModel.defaults?.fps ?? fpsOptions[0]);
    }
    if (!durationOptions.includes(Number(duration))) {
      setDuration(selectedModel.defaults?.duration ?? durationOptions[0]);
    }
    if (!samplerOptions.includes(sampler)) {
      setSampler(samplerDefaultFromModel(selectedModel));
    }
    if (!schedulerOptions.includes(scheduler)) {
      setScheduler(schedulerDefaultFromModel(selectedModel));
      setSchedulerShift(schedulerShiftDefaultFromModel(selectedModel));
    }
    if (availableTiers.length && !availableTiers.includes(quantTier)) {
      setQuantTier(availableTiers.includes(EDITOR_DEFAULT_TIER) ? EDITOR_DEFAULT_TIER : availableTiers[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel?.id]);

  const [width, height] = String(resolution).split("x").map(Number);
  const motionPct = Math.round(((MOTIONS.indexOf(motion) + 1) / MOTIONS.length) * 100);

  // Apply a chosen preset's non-LoRA defaults to the scalar controls (the LoRA seeding
  // is handled inside useGenerationStudio). Runs on preset change only.
  const selectedPresetId = studio.selectedPreset?.id ?? null;
  useEffect(() => {
    const defaults = studio.selectedPreset?.defaults;
    if (!defaults) {
      return;
    }
    if (defaults.resolution) setResolution(defaults.resolution);
    if (defaults.duration != null) setDuration(defaults.duration);
    if (defaults.fps != null) setFps(defaults.fps);
    if (defaults.quality) setQuality(defaults.quality);
    if (defaults.negativePrompt != null) setNegativePrompt(defaults.negativePrompt);
    if (defaults.motion && MOTIONS.includes(defaults.motion)) setMotion(defaults.motion);
    if (defaults.sampler) setSampler(defaults.sampler);
    if (defaults.scheduler) setScheduler(defaults.scheduler);
    if (defaults.schedulerShift != null) setSchedulerShift(defaults.schedulerShift);
    if (defaults.steps != null) setStepsOverride(defaults.steps);
    if (defaults.guidanceScale != null) setGuidanceOverride(defaults.guidanceScale);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPresetId]);

  function randomizeSeed() {
    setSeed(String(Math.floor(Math.random() * 1_000_000_000)));
  }

  // Save the current settings as a reusable preset (createPreset from context). Scope is
  // "project" when a project is open, else "global". Returns the created preset or null.
  async function savePreset(name, scope = activeProject ? "project" : "global") {
    if (!name || !name.trim() || typeof createPreset !== "function") {
      return null;
    }
    const payload = buildStudioPresetPayload({
      name: name.trim(),
      scope,
      mode: PRESET_FILTER_MODE,
      model,
      loras: studio.selectedLoras.map((lora) => serializeLora(lora, { weight: studio.effectiveLoraWeight(lora) })),
      defaults: {
        resolution,
        duration: Number(duration),
        fps: Number(fps),
        quality,
        negativePrompt,
        motion,
        sampler,
        scheduler,
        schedulerShift: Number(schedulerShift),
        ...(stepsOverride !== "" ? { steps: Number(stepsOverride) } : {}),
        ...(guidanceOverride !== "" ? { guidanceScale: Number(guidanceOverride) } : {}),
      },
    });
    return createPreset(payload);
  }

  // Persist the editor's own generation-settings snapshot (separate scope so it never
  // overwrites the richer Video Studio snapshot). Gated on a loaded catalog (sc-11962).
  useStudioSettingsWriter(
    "editor-video",
    projectId,
    {
      model,
      quality,
      resolution,
      fps,
      duration,
      motion,
      seed,
      prompt,
      negativePrompt,
      sampler,
      scheduler,
      schedulerShift,
      steps: stepsOverride,
      guidanceScale: guidanceOverride,
      quantTier,
      lightning,
      advancedOpen,
      selectedLoraIds: studio.selectedLoraIds,
      loraWeights: studio.loraWeights,
      showIncompatibleLoras: studio.showIncompatibleLoras,
      selectedPresetId: studio.selectedPresetId,
      generalStackIds: studio.generalStackIds,
    },
    videoModels.length > 0,
  );

  // The generation fields common to every editor action. The screen merges this with the
  // per-action mode + source asset + advanced.timelineAction/timelineContext. Blank /
  // default / non-finite values are omitted so the engine re-derives its own defaults.
  function buildBasePayload() {
    const mlxQuantize = tierQuantize(quantTier);
    const advanced = {
      resolution,
      motion,
      ...(showTierPicker && mlxQuantize !== null ? { mlxQuantize } : {}),
      ...(sampler && sampler !== "default" ? { sampler } : {}),
      ...(scheduler && scheduler !== "default" ? { scheduler } : {}),
      ...(scheduler && scheduler !== "default" && Number.isFinite(Number(schedulerShift))
        ? { schedulerShift: Number(schedulerShift) }
        : {}),
      ...(showLightning ? { lightning } : {}),
      ...(!lightningActive && stepsOverride !== "" && Number.isFinite(Number(stepsOverride))
        ? { steps: Number(stepsOverride) }
        : {}),
      ...(!lightningActive && guidanceOverride !== "" && Number.isFinite(Number(guidanceOverride))
        ? { guidanceScale: Number(guidanceOverride) }
        : {}),
    };
    return {
      model,
      quality,
      prompt,
      duration: Number(duration),
      fps: Number(fps),
      width: width || undefined,
      height: height || undefined,
      seed: seed === "" ? null : Number(seed),
      negativePrompt,
      recipePresetId: studio.selectedPreset?.id ?? null,
      presetLorasResolvedClientSide: studio.selectedPreset ? true : undefined,
      loras: studio.selectedLoras.map((lora) => serializeLora(lora, { weight: studio.effectiveLoraWeight(lora) })),
      advanced,
    };
  }

  return {
    studio,
    selectedModel,
    videoModels,
    createModelDownloadJob,
    createLoraDownloadJob,
    // scalar controls
    model,
    setModel,
    quality,
    setQuality,
    qualityChoices,
    resolution,
    setResolution,
    resolutionOptions,
    fps,
    setFps,
    fpsOptions,
    duration,
    setDuration,
    durationOptions,
    motion,
    setMotion,
    motionPct,
    seed,
    setSeed,
    randomizeSeed,
    prompt,
    setPrompt,
    negativePrompt,
    setNegativePrompt,
    // advanced
    advancedOpen,
    setAdvancedOpen,
    sampler,
    setSampler,
    samplerOptions,
    showSamplerPicker,
    scheduler,
    setScheduler,
    schedulerOptions,
    showSchedulerPicker,
    schedulerShift,
    setSchedulerShift,
    stepsOverride,
    setStepsOverride,
    guidanceOverride,
    setGuidanceOverride,
    quantTier,
    setQuantTier,
    availableTiers,
    showTierPicker,
    lightning,
    setLightning,
    showLightning,
    lightningActive,
    width,
    height,
    // payload + preset save
    buildBasePayload,
    savePreset,
  };
}
