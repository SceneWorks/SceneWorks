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
export const timestepTypeOptions = ["sigmoid", "linear", "weighted"];
export const timestepBiasOptions = ["balanced", "high_noise", "low_noise"];
export const lossTypeOptions = ["mse", "mae"];
// Learning-rate schedulers the worker actually honors (constant holds the LR
// fixed; linear/cosine decay it over the run). Distinct from the timestep/noise
// scheduler above. The target's `limits.lrSchedulers` overrides this fallback.
export const lrSchedulerOptions = ["constant", "linear", "cosine"];
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
export const networkTypeLabels = {
  lora: "LoRA",
  lokr: "LoKr (LyCORIS Kronecker)",
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

export function rangeOptions(limits, key) {
  return Array.isArray(limits?.[key]) ? limits[key] : [];
}

export function optimizerLabel(value) {
  return optimizerLabels[value] ?? value;
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

export function outputKindLabel(target) {
  const kind = String(target?.outputKind ?? "output").toLowerCase();
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
  const outputLabel = outputKindLabel(target);
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
  };
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
export function configValidation(configDraft, { activeDataset, selectedTarget, datasetNotReady = false } = {}) {
  const issues = [];
  if (!selectedTarget) {
    issues.push(issue.requirement("target", "Select a training target"));
  }
  if (!activeDataset?.id) {
    issues.push(issue.requirement("dataset", "Select a saved dataset"));
  }
  if (!configDraft.outputName?.trim()) {
    issues.push(issue.requirement("outputName", `Name the ${outputKindLabel(selectedTarget)} output`));
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
  // Whether the chosen dataset is trainable is part of "can this job run", so it belongs
  // in the Train button's one validity summary rather than a separate `disabled` term.
  // The screen passes the already-computed gate (trainBlockedByReadiness keeps its
  // bias-to-warn rule in datasetReadiness.js); a `needs_attention` gate is deliberately
  // NOT surfaced here — DatasetDoctorReadout's headline already carries it, and a chip
  // would only repeat it. field is null: the fix is in Data Sets, not an input on this form.
  if (datasetNotReady) {
    issues.push(issue.error(null, "This dataset isn’t ready to train yet — open Data Sets to add or fix images."));
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
  // The user-edited prompt pool, one per line. Empty falls back to the trigger-derived
  // defaults so previews still render (and {trigger} substitution is preserved). The
  // backends cap this pool at sampleCount (one preview per prompt), so the pool can hold
  // more prompts than render.
  const editedPrompts = promptLinesToList(configDraft.samplePrompts);
  const samplePrompts = editedPrompts.length ? editedPrompts : samplePromptsFromTrigger(configDraft.triggerWord);
  const advanced = compactObject({
    ...(defaults.advanced ?? {}),
    networkType,
    // LoKr factor only matters for lokr; omit it otherwise so lora jobs stay clean.
    decomposeFactor: networkType === "lokr" ? numberFromDraft(configDraft.decomposeFactor) : undefined,
    weightDecay: numberFromDraft(configDraft.weightDecay),
    lrScheduler: asText(configDraft.lrScheduler).trim() || "constant",
    lrWarmupSteps: numberFromDraft(configDraft.lrWarmupSteps),
    timestepType: asText(configDraft.timestepType).trim(),
    timestepBias: asText(configDraft.timestepBias).trim(),
    lossType: asText(configDraft.lossType).trim(),
    // Preset-only advanced keys (the submit spreads target defaults, not the
    // preset), so carry the de-distill adapter through explicitly — the worker
    // only fuses it when config.advanced.trainingAdapterRepo is present.
    trainingAdapterRepo: asText(configDraft.trainingAdapterRepo).trim(),
    trainingAdapterVersion: asText(configDraft.trainingAdapterVersion).trim(),
    gradientCheckpointing: Boolean(configDraft.gradientCheckpointing),
    mixedPrecision: asText(configDraft.precision).trim(),
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
