// Training target/preset config helpers + label maps (sc-4199). Extracted
// verbatim from TrainingStudio.jsx: the option/label lookup tables, the
// preset selection helpers, and the two pure config builders the screen used to
// bury — configDraftFromTarget (target/preset → form draft) and
// trainingConfigSnapshot (form draft → worker payload). No React, no app state.

import { issue } from "../validation/issues.js";
import {
  asText,
  compactObject,
  normalizeTrainingAdapterVersion,
  numberFromDraft,
  numericDraft,
} from "./drafts.js";

export const defaultGpuOptions = ["auto"];
export const defaultOptimizerOptions = ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"];
export const timestepTypeOptions = ["sigmoid", "linear", "uniform", "weighted"];
const sd3TimestepTypeOptions = [...timestepTypeOptions, "default", "logit_normal"];
export const timestepBiasOptions = ["balanced", "high_noise", "low_noise"];
export const lossTypeOptions = ["mse", "mae"];
// Learning-rate schedulers the worker actually honors (constant holds the LR
// fixed; linear/cosine decay it over the run). Distinct from the timestep/noise
// scheduler above. The target's `limits.lrSchedulers` overrides this fallback.
export const lrSchedulerOptions = ["constant", "linear", "cosine"];
// Relative-mode weight noising (epic 2123, sc-24826): after every optimizer update the trainer
// adds N(0,1)·sigma·rms(w) to each adapter tensor. Off (empty/0) by default; enabling it seeds the
// upstream-suggested strength. The max is the API's bound (WEIGHT_NOISE_SIGMA_MAX in
// crates/sceneworks-core/src/training.rs) — a Rust parity test reads this line, keep them equal.
export const weightNoiseSigmaMax = 0.1;
export const weightNoiseSigmaSuggested = 0.0125;
// Annealed gradient noise (epic 2123, sc-24827): on optimizer update t the trainer adds
// N(0,1)·eta/(1+t)^gamma to every adapter gradient (after the norm clip). Off (empty/0) by default;
// enabling it seeds the upstream eta and the paper's gamma. The maxima are the API's bounds
// (GRADIENT_NOISE_ETA_MAX / GRADIENT_NOISE_GAMMA_MAX in crates/sceneworks-core/src/training.rs) — a
// Rust parity test reads these lines, keep them equal.
export const gradientNoiseEtaMax = 0.1;
export const gradientNoiseEtaSuggested = 0.01;
export const gradientNoiseGammaMax = 1;
export const gradientNoiseGammaDefault = 0.55;
// Multi-resolution buckets with per-bucket repeat counts (epic 2123, sc-2127): each dataset image
// trains at every listed resolution, `repeats` times per epoch per row (so 512/768/1024 at 16/4/1
// is a 16:4:1 per-image mix). Off (null draft) by default. These are the API's limits
// (RESOLUTION_BUCKETS_MAX / RESOLUTION_BUCKET_REPEATS_MAX / RESOLUTION_BUCKET_STRIDE in
// crates/sceneworks-core/src/training.rs) — a Rust parity test reads these lines, keep them equal.
export const resolutionBucketsMax = 8;
export const resolutionBucketRepeatsMax = 100;
export const resolutionBucketStride = 32;
// Depth anchoring (epic 2123, sc-2125): an auxiliary loss that decodes the model's prediction with
// a tiny decoder, runs a frozen Depth Anything V2 on it, and pulls its depth toward the training
// image's. Off (empty weight) by default; enabling seeds the upstream DA2-Small weight. These
// bounds are the API's (crates/sceneworks-core/src/training/depth_anchoring.rs) — a Rust parity
// test reads these lines, keep them equal.
export const depthAnchoringWeightMax = 1;
export const depthAnchoringWeightSuggested = 0.1;
export const depthAnchoringEveryMax = 16;
export const depthAnchoringEveryDefault = 2;
export const depthAnchoringModelOptions = ["small", "base", "large"];
// The LTX-2.5 workflows that generate no video (API: DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS) — depth
// anchoring has no video x0 to decode there, so the engine refuses it (sc-24830).
export const depthAnchoringNoVideoLtxWorkflows = ["v2a_lora", "t2a_lora", "audio_extend_lora", "audio_inpainting_lora", "audio_suffix_lora", "a2a_ic_lora"];
export const depthAnchoringModelLabels = {
  small: "Small (fast)",
  base: "Base",
  large: "Large (much smaller weight)",
};

// Face losses (epic 2123, sc-24831): the ArcFace identity loss pulls the decoded prediction's face
// toward the training images' identity; the FaceMesh landmark loss keeps the face shape. Off (empty
// weight) by default; enabling seeds the upstream starting weight. These bounds are the API's
// (crates/sceneworks-core/src/training/face_losses.rs) — a Rust parity test reads these lines, keep
// them equal.
export const faceLossWeightMax = 1;
export const faceLossWeightSuggested = 0.1;
export const faceLossEveryMax = 16;
export const faceLossEveryDefault = 2;
export const identityLossMinCosDefault = 0.2;
export const identityLossReferenceOptions = ["dataset_average", "per_image"];
export const identityLossReferenceLabels = {
  dataset_average: "Average of all images",
  per_image: "Each image's own face",
};

// Whether the target's trainer on the serving platform honors the identity / face-landmark loss —
// the API projects the platform-effective `limits.supportsIdentityLoss` /
// `limits.supportsFaceLandmarkLoss` (pinned to the trainer descriptors by a worker test).
export function targetSupportsIdentityLoss(target) {
  return target?.limits?.supportsIdentityLoss === true;
}
export function targetSupportsFaceLandmarkLoss(target) {
  return target?.limits?.supportsFaceLandmarkLoss === true;
}

// Why a face loss (`label`) cannot run for this target + draft even though the target advertises
// it, or null — mirrors the API's `face_loss_combination_refusal` (sc-24831), the same rules as
// depth anchoring's: a full base fine-tune trains no adapter step, and an LTX-2.5 workflow with no
// generated video has nothing to decode.
export function faceLossCombinationRefusal(target, configDraft, label) {
  if (isFullFinetuneNetworkType(configDraft?.networkType)) {
    return `${label} trains a LoRA/LoKr adapter only, not a full fine-tune`;
  }
  const workflow = asText(configDraft?.ltxWorkflow).trim();
  if (target?.baseModel === "ltx_2_5" && depthAnchoringNoVideoLtxWorkflows.includes(workflow)) {
    return `${label} needs a generated video stream; the LTX-2.5 workflow ${workflow} generates none`;
  }
  return null;
}

// Whether the identity / face-landmark control is offered for this target + draft.
export function identityLossAvailable(target, configDraft) {
  return targetSupportsIdentityLoss(target) && faceLossCombinationRefusal(target, configDraft, "Identity loss") === null;
}
export function faceLandmarkLossAvailable(target, configDraft) {
  return targetSupportsFaceLandmarkLoss(target) && faceLossCombinationRefusal(target, configDraft, "Face landmark loss") === null;
}

// Subject-masked loss weighting (epic 2123, sc-24828): each image's subject mask weights the
// trainer's per-element loss — background cells by the background weight, subject cells by the
// subject weight. Off by default. The bound and defaults are the API's (SUBJECT_MASK_WEIGHT_MAX and
// SUBJECT_MASK_*_WEIGHT_DEFAULT in crates/sceneworks-core/src/training.rs) — a Rust parity test
// reads these lines, keep them equal. Background may be 0 (drop it); subject must be > 0.
export const subjectMaskWeightMax = 1;
export const subjectMaskBackgroundWeightDefault = 0.1;
export const subjectMaskSubjectWeightDefault = 1;

// Mask coverage of a dataset for masked loss, from its /subject-masks report: `usable` images carry
// a non-empty mask (an all-black "no subject" mask counts as missing — the worker refuses it too).
// `null` when no report is loaded (unknown coverage).
export function subjectMaskCoverage(report) {
  if (!report || !Array.isArray(report.items)) {
    return null;
  }
  const total = report.items.length;
  const usable = report.items.filter((item) => item?.hasMask && !item?.empty).length;
  const empty = report.items.filter((item) => item?.hasMask && item?.empty).length;
  return { total, usable, empty, missing: total - usable, complete: total > 0 && usable === total };
}

// The target's advertised training resolutions, ascending (empty when it advertises none).
function targetResolutions(target) {
  const values = target?.limits?.resolutions;
  return Array.isArray(values) ? values.map(Number).filter(Number.isFinite).sort((a, b) => a - b) : [];
}

// Draft rows (string-typed, like every other draft field) from a stored bucket list; null = off.
export function resolutionBucketsDraft(value) {
  return Array.isArray(value)
    ? value.map((row) => ({ resolution: numericDraft(row?.resolution), repeats: numericDraft(row?.repeats) }))
    : null;
}

// The rows the "Multi-resolution buckets" toggle seeds: every resolution the target trains at up to
// the current one (or just the current one when the target advertises none), one repeat each, so a
// fresh list trains exactly the sizes the target already offers and the user tunes the mix.
export function seedResolutionBuckets(target, resolution) {
  const current = numberFromDraft(resolution);
  const offered = targetResolutions(target).filter((value) => current === null || value <= current);
  const resolutions = (offered.length ? offered : [current ?? 1024]).slice(-resolutionBucketsMax);
  return resolutions.map((value) => ({ resolution: String(value), repeats: "1" }));
}

// Whether the target's trainer honors weight noise / gradient noise — the targets endpoint's
// `limits.supportsWeightNoise` / `limits.supportsGradientNoise` (pinned to the trainer descriptors
// by a worker test). Absent means unsupported.
export function targetSupportsWeightNoise(target) {
  return target?.limits?.supportsWeightNoise === true;
}
export function targetSupportsGradientNoise(target) {
  return target?.limits?.supportsGradientNoise === true;
}

// Whether the target's trainer on the serving platform honors multi-resolution buckets — the
// platform-effective `limits.supportsResolutionBuckets` (pinned to the trainer descriptors by a
// worker test; withheld for LTX-2.5). Absent means unsupported.
export function targetSupportsResolutionBuckets(target) {
  return target?.limits?.supportsResolutionBuckets === true;
}

// Whether the target's trainer on the serving platform honors depth anchoring — the same
// mechanism as weight noise: the API projects the platform-effective `limits.supportsDepthAnchoring`
// (pinned to the trainer descriptors by a worker test). Absent means unsupported.
export function targetSupportsDepthAnchoring(target) {
  return target?.limits?.supportsDepthAnchoring === true;
}

// Why depth anchoring cannot run for this target + draft even though the target advertises it, or
// null — mirrors the API's `depth_anchoring_combination_refusal` (sc-24830): a full base fine-tune
// trains no adapter step, and an LTX-2.5 workflow with no generated video has nothing to decode.
export function depthAnchoringCombinationRefusal(target, configDraft) {
  if (isFullFinetuneNetworkType(configDraft?.networkType)) {
    return "Depth anchoring trains a LoRA/LoKr adapter only, not a full fine-tune";
  }
  const workflow = asText(configDraft?.ltxWorkflow).trim();
  if (target?.baseModel === "ltx_2_5" && depthAnchoringNoVideoLtxWorkflows.includes(workflow)) {
    return `Depth anchoring needs a generated video stream; the LTX-2.5 workflow ${workflow} generates none`;
  }
  return null;
}

// Whether the depth-anchoring control is offered for this target + draft.
export function depthAnchoringAvailable(target, configDraft) {
  return targetSupportsDepthAnchoring(target) && depthAnchoringCombinationRefusal(target, configDraft) === null;
}

// Whether the target's trainer on the serving platform honors subject-masked loss (sc-24828) — the
// API advertises `limits.supportsSubjectMaskLoss` per target (pinned to the trainer descriptors on
// both platforms by a worker test). Absent means unsupported.
export function targetSupportsSubjectMaskLoss(target) {
  return target?.limits?.supportsSubjectMaskLoss === true;
}
export const optimizerLabels = {
  adam: "Adam",
  adamw: "AdamW",
  adamw8bit: "AdamW 8-bit",
  prodigy: "Prodigy",
  prodigyopt: "Prodigy",
  rose: "Rose",
};
// Adapter network parameterization. `lora` is the universal default; `lokr`
// (LyCORIS Kronecker) is offered only on targets whose `limits.networkTypes`
// advertise it (epic 2193).
// `full` is not an adapter at all (sc-14056): it trains every base weight and writes a fine-tuned
// checkpoint. Only Mage-Flow targets advertise it in `limits.networkTypes`.
export const networkTypeLabels = {
  lora: "LoRA",
  lokr: "LoKr (LyCORIS Kronecker)",
  full: "Full base fine-tune",
};
// The quality vocabulary the built-in preset registry actually emits. Quality is an
// attribute of a preset rather than a standalone hyperparameter — each tier is a
// sibling preset with its own rank/alpha/LR/steps/resolution (sc-10483).
export const qualityPresetLabels = {
  balanced: "Balanced",
  conservative: "Conservative",
  low_vram: "Low VRAM",
};
// Versions of the ostris de-distill training adapter (Z-Image-Turbo only). The
// worker maps these to the matching repo file; legacy "v2-default" normalizes to v2.
export const trainingAdapterVersionOptions = ["v1", "v2"];
export const trainingAdapterVersionLabels = {
  v1: "v1 — stable (smaller)",
  v2: "v2 — experimental (heavier de-distill)",
};

export const ltx25ValidationDefaults = Object.freeze({
  width: 960,
  height: 544,
  frames: 89,
  fps: 24,
  steps: 30,
  videoCfgScale: 3.0,
  audioCfgScale: 7.0,
  videoStgScale: 1.0,
  audioStgScale: 1.0,
  stgBlocks: [28],
  guidanceRescale: 0.7,
  videoModalityGuidanceScale: 3.0,
  audioModalityGuidanceScale: 3.0,
  generateAudio: true,
});

const condition = (type, probability, extra = {}) => ({ type, probability, ...extra });
const generated = (conditions = []) => ({ isGenerated: true, conditions });
const frozen = () => ({ isGenerated: false, conditions: [] });
const LTX25_WORKFLOW_PLANS = Object.freeze({
  i2v_lora: { video: generated([condition("firstFrame", 0.5)]), audio: generated() },
  t2v_lora: { video: generated(), audio: generated() },
  v2a_lora: { video: frozen(), audio: generated() },
  a2v_lora: { video: generated(), audio: frozen() },
  t2a_lora: { video: null, audio: generated() },
  video_extend_lora: { video: generated([condition("prefix", 1, { temporalBoundary: 8 })]), audio: generated() },
  video_inpainting_lora: { video: generated([condition("mask", 1, { tensorKey: "video_mask" })]), audio: null },
  video_outpainting_lora: { video: generated([condition("spatialCrop", 1, { spatialRegion: [0, 0, 288, 576] })]), audio: null },
  video_suffix_lora: { video: generated([condition("suffix", 1, { temporalBoundary: 8 })]), audio: generated() },
  audio_extend_lora: { video: null, audio: generated([condition("prefix", 1, { temporalBoundary: 8 })]) },
  audio_inpainting_lora: { video: null, audio: generated([condition("mask", 1, { tensorKey: "audio_mask" })]) },
  audio_suffix_lora: { video: null, audio: generated([condition("suffix", 1, { temporalBoundary: 8 })]) },
  av2av_ic_lora: {
    video: generated([condition("reference", 1, { tensorKey: "video_reference" })]),
    audio: generated([condition("reference", 1, { tensorKey: "audio_reference" })]),
  },
  v2v_ic_lora: {
    video: generated([
      condition("reference", 1, { tensorKey: "video_reference" }),
      condition("firstFrame", 0.2),
    ]),
    audio: null,
  },
  a2a_ic_lora: { video: null, audio: generated([condition("reference", 1, { tensorKey: "audio_reference" })]) },
});

function cloneJson(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

export function ltx25WorkflowPlan(workflow, previousVideo = null, previousAudio = null) {
  const canonical = LTX25_WORKFLOW_PLANS[workflow] ?? LTX25_WORKFLOW_PLANS.t2v_lora;
  const mergeModality = (next, previous) => {
    if (!next) return null;
    return {
      isGenerated: next.isGenerated,
      conditions: next.conditions.map((entry) => {
        const old = previous?.conditions?.find((candidate) => candidate?.type === entry.type);
        return old ? { ...entry, ...old, type: entry.type } : { ...entry };
      }),
    };
  };
  return {
    video: mergeModality(cloneJson(canonical.video), previousVideo),
    audio: mergeModality(cloneJson(canonical.audio), previousAudio),
  };
}

function ltxValidationSnapshot(value = {}) {
  const merged = { ...ltx25ValidationDefaults, ...value };
  return {
    width: numberFromDraft(merged.width),
    height: numberFromDraft(merged.height),
    frames: numberFromDraft(merged.frames),
    fps: numberFromDraft(merged.fps),
    steps: numberFromDraft(merged.steps),
    videoCfgScale: numberFromDraft(merged.videoCfgScale),
    audioCfgScale: numberFromDraft(merged.audioCfgScale),
    videoStgScale: numberFromDraft(merged.videoStgScale),
    audioStgScale: numberFromDraft(merged.audioStgScale),
    stgBlocks: Array.isArray(merged.stgBlocks)
      ? merged.stgBlocks.map(Number)
      : String(merged.stgBlocks).split(",").map((item) => Number(item.trim())).filter(Number.isFinite),
    guidanceRescale: numberFromDraft(merged.guidanceRescale),
    videoModalityGuidanceScale: numberFromDraft(merged.videoModalityGuidanceScale),
    audioModalityGuidanceScale: numberFromDraft(merged.audioModalityGuidanceScale),
    generateAudio: Boolean(merged.generateAudio),
  };
}

function validateLtxValidationControls(value, issues) {
  const validation = ltxValidationSnapshot(value);
  const integerRange = (field, min, max, label = optionLabel(field)) => {
    const current = validation[field];
    if (!Number.isInteger(current) || current < min || current > max) {
      issues.push(issue.error("ltxValidation", `${label} must be an integer between ${min} and ${max}`));
      return false;
    }
    return true;
  };
  const finiteRange = (field, min, max, label = optionLabel(field)) => {
    const current = validation[field];
    if (!Number.isFinite(current) || current < min || current > max) {
      issues.push(issue.error("ltxValidation", `${label} must be between ${min} and ${max}`));
    }
  };

  for (const field of ["width", "height"]) {
    if (integerRange(field, 32, 4096) && validation[field] % 32 !== 0) {
      issues.push(issue.error("ltxValidation", `${optionLabel(field)} must be aligned to 32 pixels`));
    }
  }
  if (integerRange("frames", 1, 257) && validation.frames % 8 !== 1) {
    issues.push(issue.error("ltxValidation", "Frames must satisfy frames % 8 = 1"));
  }
  integerRange("fps", 1, 120, "FPS");
  integerRange("steps", 1, 100);
  for (const field of [
    "videoCfgScale", "audioCfgScale", "videoStgScale", "audioStgScale",
    "videoModalityGuidanceScale", "audioModalityGuidanceScale",
  ]) {
    finiteRange(field, 0, 20);
  }
  finiteRange("guidanceRescale", 0, 1);

  if (validation.stgBlocks.length !== 1
    || !Number.isInteger(validation.stgBlocks[0])
    || validation.stgBlocks[0] < 0
    || validation.stgBlocks[0] > 47) {
    issues.push(issue.error("ltxValidation", "STG blocks must contain one integer block index between 0 and 47"));
  }
}

function ltxModalitySnapshot(modality) {
  if (!modality) return undefined;
  const spatialRegion = (value) => {
    if (Array.isArray(value)) return value.map(Number);
    const text = String(value ?? "").trim();
    return text ? text.split(",").map((part) => Number(part.trim())).filter(Number.isFinite) : undefined;
  };
  return {
    isGenerated: Boolean(modality.isGenerated),
    conditions: (modality.conditions ?? []).map((entry) => compactObject({
      type: entry.type,
      probability: numberFromDraft(entry.probability),
      temporalBoundary: numberFromDraft(entry.temporalBoundary),
      spatialRegion: spatialRegion(entry.spatialRegion),
      tensorKey: asText(entry.tensorKey).trim(),
      spatialScaleFactor: numberFromDraft(entry.spatialScaleFactor),
      temporalScaleFactor: numberFromDraft(entry.temporalScaleFactor),
    })).map((entry) => (entry.spatialRegion?.length ? entry : compactObject({ ...entry, spatialRegion: undefined }))),
  };
}

export function rangeOptions(limits, key) {
  return Array.isArray(limits?.[key]) ? limits[key] : [];
}

export function optimizerLabel(value) {
  return optimizerLabels[value] ?? value;
}

// The native SD3 trainers intentionally add diffusers' default/logit-normal schedule to the shared
// flow set. Every other target must stay on the shared set; exposing logit-normal globally created
// plans Anima/Mage rejected only after model load. A recognized target default/current value remains
// selected because it is already in the owning target's exact set—unsupported stale values are not
// smuggled back into the menu.
export function timestepTypeOptionsForTarget(target) {
  return target?.kernel === "sd3_lora" ? sd3TimestepTypeOptions : timestepTypeOptions;
}

export function networkTypeLabel(value) {
  return networkTypeLabels[value] ?? value;
}

export function optionLabel(value) {
  return String(value ?? "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function qualityPresetLabel(value) {
  return qualityPresetLabels[value] ?? optionLabel(value);
}

function presetSortValue(preset) {
  const order = Number(preset?.ui?.order);
  return Number.isFinite(order) ? order : 999;
}

// Two presets belong to the same group when they differ only by quality tier. Preset
// ids spell this out as `<target>.<recipe>.<optimizer>.<quality>`, but key off the
// fields rather than the id so a renamed id can't silently regroup the registry.
function presetGroupKey(preset) {
  const recipe = (preset?.recommendedFor ?? []).join("+");
  return `${preset?.targetId ?? ""}|${recipe}|${preset?.optimizer ?? ""}`;
}

// The quality tiers reachable from `preset`, in preset display order. Most groups in
// the built-in registry are single-tier; only a handful offer a real choice, so
// callers should treat a length below 2 as "nothing to pick" (sc-10483).
export function qualityTiersForPreset(presets, preset) {
  if (!preset) {
    return [];
  }
  const key = presetGroupKey(preset);
  return (presets ?? [])
    .filter((item) => presetGroupKey(item) === key)
    .slice()
    .sort((left, right) => presetSortValue(left) - presetSortValue(right));
}

// The sibling preset carrying `tier`, or null when the group doesn't offer it.
export function presetForQualityTier(presets, preset, tier) {
  return qualityTiersForPreset(presets, preset).find((item) => item.qualityPreset === tier) ?? null;
}

export function presetsForTarget(presets, targetId) {
  return (presets ?? [])
    .filter((preset) => preset.targetId === targetId)
    .slice()
    .sort((left, right) => presetSortValue(left) - presetSortValue(right) || left.name.localeCompare(right.name));
}

export function defaultPresetForTarget(presets, targetId) {
  const targetPresets = presetsForTarget(presets, targetId);
  return targetPresets.find((preset) => preset.ui?.default) ?? targetPresets[0] ?? null;
}

// sc-15036: the output kind is a per-RUN property, not a per-target one. The Mage target offers
// `networkType: "full"`, and a full run produces a base checkpoint (a model) rather than an adapter
// — the same `training_output_kind` derivation the Rust plan builder applies. Mirrored here so the
// Studio names what THIS run will produce instead of what the target usually produces.
const FULL_FINETUNE_NETWORK_TYPE = "full";

export function isFullFinetuneNetworkType(networkType) {
  return String(networkType ?? "").trim().toLowerCase() === FULL_FINETUNE_NETWORK_TYPE;
}

export function outputKindLabel(target, networkType) {
  const targetKind = String(target?.outputKind ?? "output").toLowerCase();
  const kind = targetKind === "lora" && isFullFinetuneNetworkType(networkType) ? "base_checkpoint" : targetKind;
  if (kind === "lora") {
    return "LoRA";
  }
  return kind.replaceAll("_", " ");
}

export function configDraftFromTarget(target, dataset, gpuOptions, triggerPhrase = "", preset = null, previousDraft = {}) {
  const defaults = preset?.config ?? target?.defaults ?? {};
  const advanced = defaults.advanced ?? {};
  const firstGpu = gpuOptions[0] ?? "";
  const requestedGpu = asText(advanced.requestedGpu || firstGpu);
  const outputLabel = outputKindLabel(target, advanced.networkType);
  const ltxWorkflow = asText(advanced.ltxWorkflow || "t2v_lora");
  const ltxPlan = ltx25WorkflowPlan(ltxWorkflow, advanced.ltxVideo, advanced.ltxAudio);
  return {
    outputName: previousDraft.outputName ?? (dataset?.name ? `${dataset.name} ${outputLabel}` : ""),
    triggerWord: triggerPhrase || asText(defaults.triggerWord),
    outputScope: asText(advanced.outputScope),
    qualityPreset: asText(advanced.qualityPreset),
    requestedGpu: gpuOptions.includes(requestedGpu) ? requestedGpu : firstGpu,
    rank: numericDraft(defaults.rank),
    alpha: numericDraft(defaults.alpha),
    networkType: asText(advanced.networkType || "lora"),
    // LoKr block-decomposition factor; -1 = auto. Only consumed when networkType
    // is lokr (the worker ignores it otherwise).
    decomposeFactor: numericDraft(advanced.decomposeFactor ?? -1),
    optimizer: asText(defaults.optimizer),
    learningRate: numericDraft(defaults.learningRate),
    weightDecay: numericDraft(advanced.weightDecay),
    lrScheduler: asText(advanced.lrScheduler || "constant"),
    lrWarmupSteps: numericDraft(advanced.lrWarmupSteps),
    // Empty = weight noising off (the default); see weightNoiseSigmaSuggested.
    weightNoiseSigma: numericDraft(advanced.weightNoiseSigma),
    // Empty = gradient noise off (the default); the toggle seeds eta + gamma together.
    gradientNoiseEta: numericDraft(advanced.gradientNoiseEta),
    gradientNoiseGamma: numericDraft(advanced.gradientNoiseGamma),
    // null = one resolution (the default); see seedResolutionBuckets.
    resolutionBuckets: resolutionBucketsDraft(advanced.resolutionBuckets),
    // Empty weight = depth anchoring off (the default); see depthAnchoringWeightSuggested. The
    // other knobs only matter (and only reach the job) while it is on.
    depthAnchoringWeight: numericDraft(advanced.depthAnchoringWeight),
    depthAnchoringModel: asText(advanced.depthAnchoringModel || depthAnchoringModelOptions[0]),
    depthAnchoringMinT: numericDraft(advanced.depthAnchoringMinT),
    depthAnchoringMaxT: numericDraft(advanced.depthAnchoringMaxT),
    depthAnchoringEvery: numericDraft(advanced.depthAnchoringEvery),
    // Empty weight = the face loss is off (the default); see faceLossWeightSuggested.
    identityLossWeight: numericDraft(advanced.identityLossWeight),
    identityLossMinT: numericDraft(advanced.identityLossMinT),
    identityLossMaxT: numericDraft(advanced.identityLossMaxT),
    identityLossEvery: numericDraft(advanced.identityLossEvery),
    identityLossMinCos: numericDraft(advanced.identityLossMinCos),
    identityLossReference: asText(advanced.identityLossReference || identityLossReferenceOptions[0]),
    faceLandmarkLossWeight: numericDraft(advanced.faceLandmarkLossWeight),
    faceLandmarkLossMinT: numericDraft(advanced.faceLandmarkLossMinT),
    faceLandmarkLossMaxT: numericDraft(advanced.faceLandmarkLossMaxT),
    faceLandmarkLossEvery: numericDraft(advanced.faceLandmarkLossEvery),
    // Subject-masked loss: off unless the target/preset turns it on; the weights seed the defaults.
    subjectMaskLoss: advanced.subjectMaskLoss === true,
    subjectMaskBackgroundWeight: numericDraft(
      advanced.subjectMaskBackgroundWeight ?? subjectMaskBackgroundWeightDefault,
    ),
    subjectMaskSubjectWeight: numericDraft(advanced.subjectMaskSubjectWeight ?? subjectMaskSubjectWeightDefault),
    steps: numericDraft(defaults.steps),
    timestepType: asText(advanced.timestepType || "sigmoid"),
    timestepBias: asText(advanced.timestepBias || "balanced"),
    lossType: asText(advanced.lossType || "mse"),
    trainingAdapterRepo: asText(advanced.trainingAdapterRepo),
    trainingAdapterVersion: normalizeTrainingAdapterVersion(advanced.trainingAdapterVersion),
    gradientCheckpointing: advanced.gradientCheckpointing !== false,
    resolution: numericDraft(defaults.resolution),
    precision: asText(advanced.mixedPrecision),
    saveEvery: numericDraft(defaults.saveEvery),
    sampleEvery: numericDraft(advanced.sampleEvery),
    sampleSteps: numericDraft(advanced.sampleSteps),
    sampleGuidanceScale: numericDraft(advanced.sampleGuidanceScale),
    sampleCount: numericDraft(advanced.sampleCount ?? defaultSampleCount),
    // Prefilled with the preset's prompts when it carries them, otherwise the
    // trigger-derived defaults. The screen keeps this in sync with the trigger
    // phrase until the user edits it (configPromptsFollowTrigger).
    samplePrompts: promptListToLines(
      Array.isArray(advanced.samplePrompts) && advanced.samplePrompts.length
        ? advanced.samplePrompts
        : samplePromptsFromTrigger(triggerPhrase || asText(defaults.triggerWord)),
    ),
    // Batch size and gradient accumulation have inputs in the Advanced grid
    // (sc-10689), so a bad value there is now fixable. The `?? default` floor
    // guarantees the box is never empty: a target/preset whose defaults omit either
    // field would otherwise seed "" and fail the `> 0` rule with no way to clear it.
    batchSize: numericDraft(defaults.batchSize ?? defaultBatchSize),
    gradientAccumulation: numericDraft(defaults.gradientAccumulation ?? defaultGradientAccumulation),
    seed: numericDraft(defaults.seed),
    ...(target?.baseModel === "ltx_2_5"
      ? {
          ltxWorkflow,
          ltxVideo: ltxPlan.video,
          ltxAudio: ltxPlan.audio,
          ltxValidation: ltxValidationSnapshot(advanced.ltxValidation),
        }
      : {}),
  };
}

// Resolution buckets: optional (null = off), but when on the list must hold 1..max rows, each an
// on-stride resolution the target trains at (not repeated) with a whole repeat count in
// 1..resolutionBucketRepeatsMax — the API's exact rules (E6), so the button never enables a job
// the API would refuse.
function validateResolutionBuckets(rows, target, issues) {
  if (!Array.isArray(rows)) {
    return;
  }
  const field = "resolutionBuckets";
  // A list carried over from another target (or a preset) on a target that cannot honor it would be
  // refused by the API — block it here with the same reason.
  if (!targetSupportsResolutionBuckets(target)) {
    issues.push(issue.error(field, "This target does not support multi-resolution buckets — turn them off or pick a supporting target"));
    return;
  }
  if (!rows.length) {
    issues.push(issue.error(field, "Add at least one resolution bucket, or turn multi-resolution buckets off"));
    return;
  }
  if (rows.length > resolutionBucketsMax) {
    issues.push(issue.error(field, `At most ${resolutionBucketsMax} resolution buckets`));
  }
  const offered = targetResolutions(target);
  const seen = new Set();
  rows.forEach((row, index) => {
    const n = index + 1;
    const resolution = numberFromDraft(row?.resolution);
    const repeats = numberFromDraft(row?.repeats);
    if (
      resolution === null ||
      !Number.isInteger(resolution) ||
      resolution <= 0 ||
      resolution % resolutionBucketStride !== 0
    ) {
      issues.push(issue.error(field, `Bucket ${n}: resolution must be a positive multiple of ${resolutionBucketStride}`));
    } else if (offered.length && !offered.includes(resolution)) {
      issues.push(issue.error(field, `Bucket ${n}: this target trains at ${offered.join(", ")}`));
    } else if (seen.has(resolution)) {
      issues.push(issue.error(field, `Bucket ${n}: resolution ${resolution} is already listed`));
    }
    if (resolution !== null) {
      seen.add(resolution);
    }
    if (repeats === null || !Number.isInteger(repeats) || repeats < 1 || repeats > resolutionBucketRepeatsMax) {
      issues.push(issue.error(field, `Bucket ${n}: repeats must be a whole number from 1 to ${resolutionBucketRepeatsMax}`));
    }
  });
}

// Decide how the config-draft basis effect should react to a basis change (sc-11970).
// The basis is keyed on (targetId, datasetId, defaultPresetId). Distinguishing a genuine
// USER action from an ASYNC catalog load is the whole point (mirrors S3, sc-11962):
//   - target OR dataset changed  → the user switched context → fully re-SEED the draft.
//   - ONLY the default preset id changed (target+dataset stable) → the trainingPresets
//     catalog just resolved async. If the user has already customized fields, MERGE the
//     newly-available preset defaults UNDER their edits rather than wiping them ("merge").
//     With no pending edits it's safe to seed the freshly-resolved preset ("seed").
//   - nothing changed → "noop".
// `prev`/`next` are `{ targetId, datasetId, presetId }` basis records.
export function configReseedDecision(prev, next, customizedFieldCount = 0) {
  const previous = prev ?? {};
  const userChange = previous.targetId !== next.targetId || previous.datasetId !== next.datasetId;
  if (!userChange && previous.presetId === next.presetId) {
    return "noop";
  }
  if (!userChange && customizedFieldCount > 0) {
    return "merge";
  }
  return "seed";
}

// Overlay the user's customized field values onto a freshly-seeded draft (sc-11970). Used
// on the "merge" path above so an async preset-catalog load re-seeds the fields the user
// did NOT touch from the now-available preset while preserving the ones they did.
export function mergeCustomizedConfigDraft(seeded, current = {}, customizedFields = new Set()) {
  const merged = { ...seeded };
  for (const field of customizedFields) {
    if (Object.prototype.hasOwnProperty.call(current, field)) {
      merged[field] = current[field];
    }
  }
  return merged;
}

// The training config's rule set, in the shape `useValidation` wants: a pure
// `(draft, ctx) => Issue[]` living beside the draft it validates (epic 10644).
//
// Every issue blocks Start training, but they don't deserve the same screen space. An
// unfilled field is a `requirement` — you can see the empty box, so the screen stays
// quiet and the "Needs input" pill carries it. A number the user cleared or drove
// non-positive is an `error`: nothing on the form explains the dead button, so it earns
// a chip and outlines its input.
//
// sc-10492 dropped both as noise; sc-10501 brought the errors back and this is where
// that distinction became the app's vocabulary rather than one screen's helper.
export function configValidation(
  configDraft,
  {
    activeDataset,
    selectedTarget,
    datasetNotReady = false,
    missingControlModels = [],
    subjectMaskReport = null,
  } = {},
) {
  const issues = [];
  if (!selectedTarget) {
    issues.push(issue.requirement("target", "Select a training target"));
  }
  if (!activeDataset?.id) {
    issues.push(issue.requirement("dataset", "Select a saved dataset"));
  }
  if (!configDraft.outputName?.trim()) {
    issues.push(issue.requirement("outputName", `Name the ${outputKindLabel(selectedTarget, configDraft.networkType)} output`));
  }
  if (!configDraft.triggerWord?.trim()) {
    issues.push(issue.requirement("triggerWord", "Add a trigger phrase"));
  }
  // The field name is the draft key, so `invalidProps` can outline the very input the
  // chip is talking about. Every field below has an input in ConfigureJobPanel — the
  // basic grid (steps, saveEvery) or the Advanced disclosure (the rest, including
  // batchSize and gradientAccumulation as of sc-10689) — so every error names a field
  // the user can reach and clear, which is the epic's premise (10644 R5).
  for (const [field, label] of [
    ["rank", "Rank"],
    ["alpha", "Alpha"],
    ["learningRate", "Learning rate"],
    ["steps", "Steps"],
    ["resolution", "Resolution"],
    ["batchSize", "Batch size"],
    ["gradientAccumulation", "Gradient accumulation"],
    ["saveEvery", "Checkpoint cadence"],
  ]) {
    const value = numberFromDraft(configDraft[field]);
    if (!value || value <= 0) {
      issues.push(issue.error(field, `${label} must be greater than zero`));
    }
  }
  // Weight noising: optional, but when set it must sit inside the API's bound (same max, E6),
  // and it only perturbs adapter weights, so a full base fine-tune cannot use it (E5).
  if (String(configDraft.weightNoiseSigma ?? "").trim()) {
    const sigma = numberFromDraft(configDraft.weightNoiseSigma);
    if (sigma === null || sigma < 0 || sigma > weightNoiseSigmaMax) {
      issues.push(issue.error("weightNoiseSigma", `Weight noise must be between 0 and ${weightNoiseSigmaMax}`));
    } else if (sigma > 0 && isFullFinetuneNetworkType(configDraft.networkType)) {
      issues.push(issue.error("weightNoiseSigma", "Weight noise only applies to LoRA/LoKr adapters, not a full fine-tune"));
    } else if (sigma > 0 && selectedTarget && !targetSupportsWeightNoise(selectedTarget)) {
      // The toggle is hidden for such a target, so this names no input (field null): the value can
      // only arrive from a carried-over draft, and the API would refuse it anyway.
      issues.push(issue.error(null, "This target does not support weight noise — clear it or pick a supporting target"));
    }
  }
  // Gradient noise (sc-24827): same bounds as the API (E6); eta is adapter-only (E5).
  if (String(configDraft.gradientNoiseEta ?? "").trim()) {
    const eta = numberFromDraft(configDraft.gradientNoiseEta);
    if (eta === null || eta < 0 || eta > gradientNoiseEtaMax) {
      issues.push(issue.error("gradientNoiseEta", `Gradient noise eta must be between 0 and ${gradientNoiseEtaMax}`));
    } else if (eta > 0 && isFullFinetuneNetworkType(configDraft.networkType)) {
      issues.push(issue.error("gradientNoiseEta", "Gradient noise only applies to LoRA/LoKr adapters, not a full fine-tune"));
    } else if (eta > 0 && selectedTarget && !targetSupportsGradientNoise(selectedTarget)) {
      // As for weight noise: the controls are hidden for such a target, so this names no input.
      issues.push(issue.error(null, "This target does not support gradient noise — clear it or pick a supporting target"));
    }
  }
  if (String(configDraft.gradientNoiseGamma ?? "").trim()) {
    const gamma = numberFromDraft(configDraft.gradientNoiseGamma);
    if (gamma === null || gamma < 0 || gamma > gradientNoiseGammaMax) {
      issues.push(issue.error("gradientNoiseGamma", `Gradient noise gamma must be between 0 and ${gradientNoiseGammaMax}`));
    }
  }
  validateResolutionBuckets(configDraft.resolutionBuckets, selectedTarget, issues);
  for (const [field, message] of depthAnchoringIssues(configDraft, selectedTarget)) {
    issues.push(issue.error(field, message));
  }
  for (const [field, message] of faceLossIssues(configDraft, selectedTarget)) {
    issues.push(issue.error(field, message));
  }
  // Subject-masked loss: the weights sit inside the API's bounds (same max, E6), and every image
  // needs a non-empty subject mask — the worker refuses the job otherwise, so incomplete coverage
  // blocks Start here too (only when the report is loaded; unknown coverage is left to the worker).
  if (configDraft.subjectMaskLoss && selectedTarget && !targetSupportsSubjectMaskLoss(selectedTarget)) {
    // The toggle is hidden for such a target, so a `true` can only be carried over from another
    // target's draft; it must block Start (the API refuses it too).
    issues.push(issue.error(null, "This target does not support subject-masked loss — pick a supporting target"));
  } else if (configDraft.subjectMaskLoss) {
    const background = numberFromDraft(configDraft.subjectMaskBackgroundWeight);
    if (background === null || background < 0 || background > subjectMaskWeightMax) {
      issues.push(
        issue.error("subjectMaskBackgroundWeight", `Background weight must be between 0 and ${subjectMaskWeightMax}`),
      );
    }
    const subject = numberFromDraft(configDraft.subjectMaskSubjectWeight);
    if (subject === null || subject <= 0 || subject > subjectMaskWeightMax) {
      issues.push(
        issue.error(
          "subjectMaskSubjectWeight",
          `Subject weight must be greater than 0 and at most ${subjectMaskWeightMax}`,
        ),
      );
    }
    const coverage = subjectMaskCoverage(subjectMaskReport);
    if (coverage && !coverage.complete) {
      issues.push(
        issue.error(
          "subjectMaskLoss",
          `Subject masks are missing for ${coverage.missing} of ${coverage.total} images — generate subject masks first`,
        ),
      );
    }
  }
  // Whether the chosen dataset is trainable is part of "can this job run", so it belongs
  // in the Train button's one validity summary rather than a separate `disabled` term.
  // The screen passes the already-computed gate (trainBlockedByReadiness keeps its
  // bias-to-warn rule in datasetReadiness.js); a `needs_attention` gate is deliberately
  // NOT surfaced here — DatasetDoctorReadout's headline already carries it, and a chip
  // would only repeat it. field is null: the fix is in Data Sets, not an input on this form.
  if (datasetNotReady) {
    issues.push(issue.error(null, "This dataset isn’t ready to train yet — open Data Sets to add or fix images."));
  }
  // A ControlNet run renders its per-image condition with a preprocessor model, and those resolvers
  // are cache-only since epic 17625 — a missing one is a job-time failure, not a mid-run download.
  // So it is part of "can this job run" and belongs in the one validity summary that gates Start,
  // exactly like `datasetNotReady` above, rather than a separate `disabled` term that could drift
  // from what the panel says. field is null: the fix is a download in the panel's notice, not an
  // input on this form — ConfigureJobPanel renders the offer right below the ControlNet note.
  if (missingControlModels.length > 0) {
    const names = missingControlModels.map((model) => model?.name ?? model?.id).filter(Boolean);
    issues.push(
      issue.error(
        null,
        `Install ${names.join(" and ")} to render this run's control condition.`,
      ),
    );
  }
  if (selectedTarget?.baseModel === "ltx_2_5") {
    const workflows = Array.isArray(selectedTarget?.limits?.ltxWorkflows)
      ? selectedTarget.limits.ltxWorkflows
      : [];
    if (!workflows.includes(configDraft.ltxWorkflow)) {
      issues.push(issue.error("ltxWorkflow", "Choose a supported LTX-2.5 workflow"));
    }
    const missingPrepared = (activeDataset?.items ?? []).filter(
      (item) => !String(item?.ltxPreparedBundlePath ?? "").trim(),
    );
    if (missingPrepared.length) {
      issues.push(issue.error(null, `Attach an LTX prepared bundle to all ${missingPrepared.length} unprepared dataset item(s).`));
    }
    for (const modality of [configDraft.ltxVideo, configDraft.ltxAudio].filter(Boolean)) {
      for (const condition of modality.conditions ?? []) {
        const probability = Number(condition.probability);
        if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
          issues.push(issue.error("ltxWorkflow", "LTX condition probability must be between 0 and 1"));
        }
        if (["mask", "reference"].includes(condition.type) && !String(condition.tensorKey ?? "").trim()) {
          issues.push(issue.error("ltxWorkflow", `${optionLabel(condition.type)} conditions require a tensor key`));
        }
        if (["prefix", "suffix"].includes(condition.type) && Number(condition.temporalBoundary) <= 0) {
          issues.push(issue.error("ltxWorkflow", "LTX temporal boundaries must be greater than zero"));
        }
        if (condition.type === "spatialCrop") {
          const region = Array.isArray(condition.spatialRegion)
            ? condition.spatialRegion.map(Number)
            : String(condition.spatialRegion ?? "").split(",").map((part) => Number(part.trim()));
          if (region.length !== 4 || region.some((value) => !Number.isFinite(value)) || region[2] <= region[0] || region[3] <= region[1]) {
            issues.push(issue.error("ltxWorkflow", "LTX spatial regions require y1,x1,y2,x2 with positive area"));
          }
        }
        for (const scale of [condition.spatialScaleFactor, condition.temporalScaleFactor]) {
          if (scale !== undefined && scale !== "" && Number(scale) <= 0) {
            issues.push(issue.error("ltxWorkflow", "LTX condition scale factors must be greater than zero"));
          }
        }
      }
    }
    validateLtxValidationControls(configDraft.ltxValidation, issues);
  }
  return issues;
}

export function samplePromptsFromTrigger(triggerWord) {
  const trigger = String(triggerWord ?? "").trim() || "the trained subject";
  return [
    `${trigger}, studio portrait, soft key light, detailed face`,
    `${trigger}, full body fashion editorial photo, natural pose`,
    `${trigger}, cinematic outdoor portrait, golden hour`,
    `${trigger}, close-up character portrait, dramatic rim light`,
  ];
}

// Default number of preview images rendered per sample step (sc-8671). Matches
// the four trigger-derived default prompts, so the out-of-the-box behavior is
// unchanged when neither knob is touched. The backends cap the prompt pool at
// this count (one preview per prompt, truncated — never padded).
export const defaultSampleCount = 4;

// Safety-net defaults for the two hyperparameters the panel now exposes (sc-10689).
// Every built-in target/preset already ships explicit values (the Rust `TrainingConfig`
// contract types both as required, so the API can't omit them), and that per-model
// value is what `configDraftFromTarget` uses when present. These only fill the box for
// a source that omits the field — a loosened contract or a user-authored preset — where
// `1` (batch of one, no accumulation) is the universally safe floor: minimum VRAM,
// always fits, and never larger than any advertised `limits.batchSize` range.
export const defaultBatchSize = 1;
export const defaultGradientAccumulation = 1;

// The sample-prompts textarea holds one prompt per line; the worker payload wants
// a string array. These two convert between the draft string and the array.
export function promptLinesToList(text) {
  return String(text ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export function promptListToLines(list) {
  return (Array.isArray(list) ? list : []).join("\n");
}

export function trainingConfigSnapshot({ activeDataset, configDraft, selectedPreset, selectedTarget, dryRun = true }) {
  const defaults = selectedTarget?.defaults ?? {};
  const networkType = asText(configDraft.networkType).trim() || "lora";
  const isFullFinetune = isFullFinetuneNetworkType(networkType);
  const fullFinetuneConfig = isFullFinetune ? defaults.advanced?.fullFinetuneConfig : null;
  const defaultAdvanced = { ...(defaults.advanced ?? {}) };
  delete defaultAdvanced.fullFinetuneConfig;
  // The user-edited prompt pool, one per line. Empty falls back to the trigger-derived
  // defaults so previews still render (and {trigger} substitution is preserved). The
  // backends cap this pool at sampleCount (one preview per prompt), so the pool can hold
  // more prompts than render.
  const editedPrompts = promptLinesToList(configDraft.samplePrompts);
  const samplePrompts = editedPrompts.length ? editedPrompts : samplePromptsFromTrigger(configDraft.triggerWord);
  const ltxWorkflow = asText(configDraft.ltxWorkflow).trim();
  const ltxPlan = selectedTarget?.baseModel === "ltx_2_5"
    ? ltx25WorkflowPlan(ltxWorkflow, configDraft.ltxVideo, configDraft.ltxAudio)
    : null;
  const advanced = compactObject({
    ...defaultAdvanced,
    networkType,
    // LoKr factor only matters for lokr; omit it otherwise so lora jobs stay clean.
    decomposeFactor: networkType === "lokr" ? numberFromDraft(configDraft.decomposeFactor) : undefined,
    weightDecay: numberFromDraft(configDraft.weightDecay),
    lrScheduler: asText(configDraft.lrScheduler).trim() || "constant",
    lrWarmupSteps: numberFromDraft(configDraft.lrWarmupSteps),
    // Omitted when off (empty draft), so a default job's snapshot is unchanged.
    weightNoiseSigma: numberFromDraft(configDraft.weightNoiseSigma),
    // Gradient noise: omitted when off; gamma only travels with an eta (it means nothing alone).
    gradientNoiseEta: numberFromDraft(configDraft.gradientNoiseEta),
    gradientNoiseGamma: String(configDraft.gradientNoiseEta ?? "").trim()
      ? numberFromDraft(configDraft.gradientNoiseGamma)
      : undefined,
    // Omitted when off (null draft); an empty "on" list is sent as [] and refused by both sides.
    resolutionBuckets: Array.isArray(configDraft.resolutionBuckets)
      ? configDraft.resolutionBuckets.map((row) => ({
          resolution: numberFromDraft(row?.resolution),
          repeats: numberFromDraft(row?.repeats),
        }))
      : undefined,
    ...depthAnchoringSnapshot(configDraft),
    ...faceLossSnapshot(configDraft),
    // Omitted entirely when off, so a default job's snapshot is unchanged.
    subjectMaskLoss: configDraft.subjectMaskLoss ? true : undefined,
    subjectMaskBackgroundWeight: configDraft.subjectMaskLoss
      ? numberFromDraft(configDraft.subjectMaskBackgroundWeight)
      : undefined,
    subjectMaskSubjectWeight: configDraft.subjectMaskLoss
      ? numberFromDraft(configDraft.subjectMaskSubjectWeight)
      : undefined,
    timestepType: asText(configDraft.timestepType).trim(),
    timestepBias: asText(configDraft.timestepBias).trim(),
    lossType: asText(configDraft.lossType).trim(),
    // Preset-only advanced keys (the submit spreads target defaults, not the
    // preset), so carry the de-distill adapter through explicitly — the worker
    // only fuses it when config.advanced.trainingAdapterRepo is present.
    trainingAdapterRepo: asText(configDraft.trainingAdapterRepo).trim(),
    trainingAdapterVersion: asText(configDraft.trainingAdapterVersion).trim(),
    // Platform-effective target metadata carries any backend-specific full-tune requirement. MLX
    // has no override and keeps the submitted draft; Candle Mage advertises f32/no-checkpointing.
    gradientCheckpointing:
      fullFinetuneConfig?.gradientCheckpointing ?? Boolean(configDraft.gradientCheckpointing),
    mixedPrecision: fullFinetuneConfig?.mixedPrecision ?? asText(configDraft.precision).trim(),
    sampleEvery: numberFromDraft(configDraft.sampleEvery),
    sampleSteps: numberFromDraft(configDraft.sampleSteps),
    sampleGuidanceScale: numberFromDraft(configDraft.sampleGuidanceScale),
    sampleCount: numberFromDraft(configDraft.sampleCount),
    samplePrompts,
    // Provenance only: no backend reads `advanced.qualityPreset`. The tier is carried
    // for real by presetId/presetVersion, which pin the hyperparameters below (sc-10483).
    qualityPreset: configDraft.qualityPreset,
    outputScope: configDraft.outputScope,
    requestedGpu: configDraft.requestedGpu,
    ...(ltxPlan
      ? {
          ltxWorkflow,
          ltxVideo: ltxModalitySnapshot(ltxPlan.video),
          ltxAudio: ltxModalitySnapshot(ltxPlan.audio),
          ltxValidation: ltxValidationSnapshot(configDraft.ltxValidation),
        }
      : {}),
  });
  return {
    targetId: selectedTarget.id,
    datasetId: activeDataset.id,
    datasetVersion: activeDataset.version,
    outputName: configDraft.outputName.trim(),
    dryRun,
    outputScope: configDraft.outputScope,
    qualityPreset: configDraft.qualityPreset,
    requestedGpu: configDraft.requestedGpu,
    presetId: selectedPreset?.id,
    presetVersion: selectedPreset?.version,
    config: {
      rank: numberFromDraft(configDraft.rank),
      alpha: numberFromDraft(configDraft.alpha),
      learningRate: numberFromDraft(configDraft.learningRate),
      steps: numberFromDraft(configDraft.steps),
      batchSize: numberFromDraft(configDraft.batchSize),
      gradientAccumulation: numberFromDraft(configDraft.gradientAccumulation),
      resolution: numberFromDraft(configDraft.resolution),
      saveEvery: numberFromDraft(configDraft.saveEvery),
      seed: numberFromDraft(configDraft.seed),
      optimizer: asText(configDraft.optimizer).trim(),
      triggerWord: asText(configDraft.triggerWord).trim(),
      advanced,
    },
  };
}

// Depth anchoring is on whenever the draft carries a weight (empty = off).
export function depthAnchoringEnabled(configDraft) {
  return String(configDraft?.depthAnchoringWeight ?? "").trim() !== "";
}

// Field issues for the depth-anchoring knobs — the API's bounds (E6): weight in
// [0, depthAnchoringWeightMax], a known model size, a [0, 1] noise window with min <= max, and a
// whole alternation period in [1, depthAnchoringEveryMax]. Checked only while enabled.
export function depthAnchoringIssues(configDraft, selectedTarget) {
  if (!depthAnchoringEnabled(configDraft)) return [];
  const issues = [];
  const weight = numberFromDraft(configDraft.depthAnchoringWeight);
  if (weight === null || weight < 0 || weight > depthAnchoringWeightMax) {
    issues.push(["depthAnchoringWeight", `Depth anchoring weight must be between 0 and ${depthAnchoringWeightMax}`]);
  } else if (weight > 0 && selectedTarget && !targetSupportsDepthAnchoring(selectedTarget)) {
    // The toggle is hidden for such a target, so this names no input (field null): the value can
    // only arrive from a carried-over draft, and the API would refuse it anyway.
    issues.push([null, "This target does not support depth anchoring — clear it or pick a supporting target"]);
    return issues;
  } else if (weight > 0 && selectedTarget) {
    const refusal = depthAnchoringCombinationRefusal(selectedTarget, configDraft);
    if (refusal) {
      issues.push(["depthAnchoringWeight", refusal]);
      return issues;
    }
  }
  if (!depthAnchoringModelOptions.includes(asText(configDraft.depthAnchoringModel).trim())) {
    issues.push(["depthAnchoringModel", `Depth model must be one of ${depthAnchoringModelOptions.join(", ")}`]);
  }
  const window = {};
  for (const field of ["depthAnchoringMinT", "depthAnchoringMaxT"]) {
    if (!String(configDraft[field] ?? "").trim()) continue;
    const t = numberFromDraft(configDraft[field]);
    if (t === null || t < 0 || t > 1) {
      issues.push([field, "Noise window bounds must be between 0 and 1"]);
    } else {
      window[field] = t;
    }
  }
  const minT = window.depthAnchoringMinT ?? 0;
  const maxT = window.depthAnchoringMaxT ?? 1;
  if (minT > maxT) {
    issues.push(["depthAnchoringMaxT", "Noise window max must be at least its min"]);
  }
  if (String(configDraft.depthAnchoringEvery ?? "").trim()) {
    const every = numberFromDraft(configDraft.depthAnchoringEvery);
    if (every === null || !Number.isInteger(every) || every < 1 || every > depthAnchoringEveryMax) {
      issues.push(["depthAnchoringEvery", `Alternation period must be a whole number from 1 to ${depthAnchoringEveryMax}`]);
    }
  }
  return issues;
}

// The depth-anchoring keys a job snapshot carries: none while off, so a default job's snapshot
// is unchanged; the weight plus every set knob while on (unset knobs take the API defaults).
export function depthAnchoringSnapshot(configDraft) {
  if (!depthAnchoringEnabled(configDraft)) return {};
  return compactObject({
    depthAnchoringWeight: numberFromDraft(configDraft.depthAnchoringWeight),
    depthAnchoringModel: asText(configDraft.depthAnchoringModel).trim(),
    depthAnchoringMinT: numberFromDraft(configDraft.depthAnchoringMinT),
    depthAnchoringMaxT: numberFromDraft(configDraft.depthAnchoringMaxT),
    depthAnchoringEvery: numberFromDraft(configDraft.depthAnchoringEvery),
  });
}

// A face loss is on whenever the draft carries its weight (empty = off).
export function identityLossEnabled(configDraft) {
  return String(configDraft?.identityLossWeight ?? "").trim() !== "";
}
export function faceLandmarkLossEnabled(configDraft) {
  return String(configDraft?.faceLandmarkLossWeight ?? "").trim() !== "";
}

// Shared schedule checks for one face loss (weight / noise window / alternation period), the API's
// bounds (E6). `prefix` is the draft-key prefix ("identityLoss" / "faceLandmarkLoss"). Marks the
// returned issues `blocked` when the loss cannot run here at all (unsupported target or refused
// combination), so the caller skips its loss-specific knobs.
function faceLossScheduleIssues(configDraft, prefix, label, supported, selectedTarget) {
  const issues = [];
  const weightKey = `${prefix}Weight`;
  const weight = numberFromDraft(configDraft[weightKey]);
  if (weight === null || weight < 0 || weight > faceLossWeightMax) {
    issues.push([weightKey, `${label} weight must be between 0 and ${faceLossWeightMax}`]);
  } else if (weight > 0 && !supported) {
    // The control is hidden for such a target, so this names no input (field null): the value can
    // only arrive from a carried-over draft, and the API would refuse it anyway.
    issues.push([null, `This target does not support the ${label.toLowerCase()} — clear it or pick a supporting target`]);
    issues.blocked = true;
    return issues;
  } else if (weight > 0 && selectedTarget) {
    const refusal = faceLossCombinationRefusal(selectedTarget, configDraft, label);
    if (refusal) {
      issues.push([weightKey, refusal]);
      issues.blocked = true;
      return issues;
    }
  }
  const window = {};
  for (const field of [`${prefix}MinT`, `${prefix}MaxT`]) {
    if (!String(configDraft[field] ?? "").trim()) continue;
    const t = numberFromDraft(configDraft[field]);
    if (t === null || t < 0 || t > 1) {
      issues.push([field, "Noise window bounds must be between 0 and 1"]);
    } else {
      window[field] = t;
    }
  }
  if ((window[`${prefix}MinT`] ?? 0) > (window[`${prefix}MaxT`] ?? 1)) {
    issues.push([`${prefix}MaxT`, "Noise window max must be at least its min"]);
  }
  const everyKey = `${prefix}Every`;
  if (String(configDraft[everyKey] ?? "").trim()) {
    const every = numberFromDraft(configDraft[everyKey]);
    if (every === null || !Number.isInteger(every) || every < 1 || every > faceLossEveryMax) {
      issues.push([everyKey, `Alternation period must be a whole number from 1 to ${faceLossEveryMax}`]);
    }
  }
  return issues;
}

// Field issues for the face-loss knobs (checked only while each loss is enabled).
export function faceLossIssues(configDraft, selectedTarget) {
  const issues = [];
  if (identityLossEnabled(configDraft)) {
    const supported = !selectedTarget || targetSupportsIdentityLoss(selectedTarget);
    const scheduleIssues = faceLossScheduleIssues(configDraft, "identityLoss", "Identity loss", supported, selectedTarget);
    issues.push(...scheduleIssues);
    if (!scheduleIssues.blocked) {
      if (String(configDraft.identityLossMinCos ?? "").trim()) {
        const minCos = numberFromDraft(configDraft.identityLossMinCos);
        if (minCos === null || minCos < -1 || minCos > 1) {
          issues.push(["identityLossMinCos", "Identity gate must be between -1 and 1"]);
        }
      }
      if (!identityLossReferenceOptions.includes(asText(configDraft.identityLossReference).trim())) {
        issues.push(["identityLossReference", `Identity reference must be one of ${identityLossReferenceOptions.join(", ")}`]);
      }
    }
  }
  if (faceLandmarkLossEnabled(configDraft)) {
    const supported = !selectedTarget || targetSupportsFaceLandmarkLoss(selectedTarget);
    issues.push(...faceLossScheduleIssues(configDraft, "faceLandmarkLoss", "Face landmark loss", supported, selectedTarget));
  }
  return issues;
}

// The face-loss keys a job snapshot carries: none while off (a default job's snapshot is
// unchanged); the weight plus every set knob while on (unset knobs take the API defaults).
export function faceLossSnapshot(configDraft) {
  const snapshot = {};
  if (identityLossEnabled(configDraft)) {
    Object.assign(snapshot, compactObject({
      identityLossWeight: numberFromDraft(configDraft.identityLossWeight),
      identityLossMinT: numberFromDraft(configDraft.identityLossMinT),
      identityLossMaxT: numberFromDraft(configDraft.identityLossMaxT),
      identityLossEvery: numberFromDraft(configDraft.identityLossEvery),
      identityLossMinCos: numberFromDraft(configDraft.identityLossMinCos),
      identityLossReference: asText(configDraft.identityLossReference).trim(),
    }));
  }
  if (faceLandmarkLossEnabled(configDraft)) {
    Object.assign(snapshot, compactObject({
      faceLandmarkLossWeight: numberFromDraft(configDraft.faceLandmarkLossWeight),
      faceLandmarkLossMinT: numberFromDraft(configDraft.faceLandmarkLossMinT),
      faceLandmarkLossMaxT: numberFromDraft(configDraft.faceLandmarkLossMaxT),
      faceLandmarkLossEvery: numberFromDraft(configDraft.faceLandmarkLossEvery),
    }));
  }
  return snapshot;
}
