// Iris-3B DEPTH (sc-25687) and RESTORATION (sc-25688) training — the web half of the
// `advanced.iris` (depth) / `advanced.iris_restore` (restoration) options-bag contracts owned by
// `crates/sceneworks-core/src/training/iris/{depth,restore}.rs`.
//
// Same schema-driven shape as irisOptions.js (generation): each objective names exactly its
// client-settable keys (`DEPTH_CLIENT_OPTION_KEYS` / `RESTORE_CLIENT_OPTION_KEYS`), the draft holds
// every key as an input-friendly value, and the snapshot emits only the keys that are filled AND in
// play for this run — the engine refuses an unknown key or a knob that does not apply.
//
// Pure: no React, no app state.

import { issue } from "../validation/issues.js";
import { asText, numberFromDraft } from "./drafts.js";
import { IRIS_OPTION_SPECS, RESUME_OR_LOAD_MESSAGE, irisObjective, parseOption, parseOptionsBag } from "./irisOptions.js";

const pick = (keys) => Object.fromEntries(keys.map((key) => [key, IRIS_OPTION_SPECS[key]]));
const choice = (label, ...values) => ({ kind: "choice", values, label });
const atLeastZero = (label) => ({ kind: "number", check: (x) => x >= 0, expect: "at least 0", label });
const percentile = (label) => ({ kind: "number", check: (x) => x >= 0 && x <= 100, expect: "between 0 and 100", label });
const SHARED = [
  "betas", "gradient_clip", "ema_enabled", "ema_decay", "num_epochs", "nan_loss_tolerance",
  "keep_last_checkpoints", "milestone_steps", "resume_data_policy",
];

export const IRIS_TASK_SPECS = Object.freeze({
  depth: Object.freeze({
    depth_size: { kind: "size", label: "Training size" },
    resize: choice("Resize", "cover", "none"),
    crop: choice("Crop", "center", "random"),
    hflip_prob: { ...IRIS_OPTION_SPECS.text_dropout, label: "Horizontal flip probability" },
    gradient_weight: atLeastZero("Gradient loss weight"),
    low_percentile: percentile("Low percentile"),
    high_percentile: percentile("High percentile"),
    clip_target: { kind: "bool", label: "Clip the target to the percentiles" },
    loss_reduction: choice("Loss reduction", "valid_pixels", "per_image"),
    invalid_examples: choice("Invalid examples", "reject", "skip"),
    min_valid_pixels: { kind: "int", min: 2, label: "Minimum valid pixels" },
    reducer_init: choice("Depth head init", "torch_default", "channel_mean"),
    ...pick([...SHARED, "muon_momentum", "muon_nesterov", "muon_adjust_lr", "auto_lr", "base_batch_size", "override_lr_on_resume", "preview_weights", "export_weights"]),
    preview_items: { kind: "int", min: 0, label: "Preview items" },
  }),
  restoration: Object.freeze({
    loss_l2: atLeastZero("Pixel (L2) loss weight"),
    loss_lpips: atLeastZero("LPIPS loss weight"),
    loss_gan: atLeastZero("GAN loss weight"),
    gan_real_label: { ...IRIS_OPTION_SPECS.ema_decay, label: "GAN real label" },
    lr_critic: atLeastZero("Critic learning rate"),
    critic_weight_decay: atLeastZero("Critic weight decay"),
    critic_betas: { kind: "betas", label: "Critic betas" },
    critic_gradient_clip: { ...IRIS_OPTION_SPECS.gradient_clip, label: "Critic gradient clip" },
    crop: choice("Crop", "random", "center", "none"),
    critic_dtype: choice("Critic precision", "bf16", "fp32"),
    init: choice("Start from", "export", "parent"),
    lora_init: choice("Adapter init", "gaussian", "kaiming"),
    ...pick([...SHARED, "preview_weights", "export_weights", "export_dtype"]),
    degradation: { kind: "json", label: "Synthetic degradation overrides" },
  }),
});

// The request bag key each objective's options travel under.
export const IRIS_TASK_BAG_KEYS = Object.freeze({ depth: "iris", restoration: "iris_restore" });

function taskRun(objective, configDraft = {}) {
  const bag = configDraft.irisTask ?? {};
  return {
    bag,
    // Depth trains without an EMA by default; restoration with one.
    emaOn: objective === "depth" ? bag.ema_enabled === true : bag.ema_enabled !== false,
    resuming: Boolean(asText(configDraft.resumeFromJobId).trim()),
    full: !["lora", "lokr"].includes(asText(configDraft.networkType).trim().toLowerCase()),
    parent: asText(configDraft.depthInit).trim() === "parent",
    muon: asText(configDraft.optimizer).trim().toLowerCase() === "muon",
    autoLr: !["", "none"].includes(asText(bag.auto_lr).trim()),
  };
}

// Whether `key` applies to this run (not rendered, sent or validated otherwise).
export function irisTaskOptionInPlay(objective, key, configDraft) {
  const run = taskRun(objective, configDraft);
  switch (key) {
    case "reducer_init":
      return run.parent;
    case "lora_init":
      return !run.full;
    case "ema_decay":
      return run.emaOn;
    case "muon_momentum":
    case "muon_nesterov":
    case "muon_adjust_lr":
      return run.muon;
    case "base_batch_size":
      return run.autoLr;
    case "resume_data_policy":
    case "override_lr_on_resume":
      return run.resuming;
    default:
      return true;
  }
}

function taskBag(objective, configDraft) {
  return parseOptionsBag(
    IRIS_TASK_SPECS[objective] ?? {},
    configDraft.irisTask,
    (key) => irisTaskOptionInPlay(objective, key, configDraft),
    "irisTask",
    taskRun(objective, configDraft).emaOn,
  );
}

export function irisTaskOptionsSnapshot(objective, configDraft = {}) {
  return taskBag(objective, configDraft).values;
}

// The depth run trains at `depth_size`; `resolution` must mirror its larger side.
export function irisDepthResolution(configDraft = {}) {
  const size = parseOption("depth_size", configDraft.irisTask?.depth_size, IRIS_TASK_SPECS.depth).value;
  return size ? Math.max(...size) : numberFromDraft(configDraft.resolution);
}

// The row key of the target's declared memory-floor table for this run: the micro-batch pixels
// (batch × training height × width) when the table is keyed by them, else the resolution.
export function irisFloorSize(target, configDraft = {}) {
  const depth = irisObjective(target) === "depth";
  const resolution = depth ? irisDepthResolution(configDraft) : numberFromDraft(configDraft.resolution);
  if (target?.memoryFloors?.key !== "microBatchPixels") return resolution;
  const size = depth
    ? parseOption("depth_size", configDraft.irisTask?.depth_size, IRIS_TASK_SPECS.depth).value
    : null;
  const batch = numberFromDraft(configDraft.batchSize) ?? 1;
  return batch * (size ? size[0] * size[1] : resolution * resolution);
}

// The request-level keys riding beside the bag.
export function irisTaskRequestKeys(objective, configDraft = {}) {
  const resumeFromJobId = asText(configDraft.resumeFromJobId).trim();
  const loadFromJobId = asText(configDraft.loadFromJobId).trim();
  const out = resumeFromJobId ? { resumeFromJobId } : {};
  if (objective === "depth") {
    out.depthInit = taskRun(objective, configDraft).parent ? "parent" : "export";
    if (loadFromJobId && out.depthInit === "export") out.loadFromJobId = loadFromJobId;
  }
  return out;
}

// One optimizer step of a restoration run is a generator window plus a critic window, so its
// micro-step fields are multiples of 2 × gradient accumulation (limits.microStepMultiple).
export function microStepMultiple(target, accumulation) {
  return target?.limits?.microStepMultiple === "2xGradientAccumulation" ? 2 * accumulation : accumulation;
}

export function irisTaskValidation(objective, configDraft = {}) {
  const { issues } = taskBag(objective, configDraft);
  if (objective === "depth") {
    const run = taskRun(objective, configDraft);
    const low = numberFromDraft(run.bag.low_percentile) ?? 2;
    const high = numberFromDraft(run.bag.high_percentile) ?? 98;
    if (low >= high) issues.push(issue.error("irisTask.low_percentile", "Low percentile must be below the high percentile"));
    if (asText(configDraft.loadFromJobId).trim() && (run.resuming || run.parent)) {
      issues.push(issue.error("loadFromJobId", run.resuming ? RESUME_OR_LOAD_MESSAGE : "A previous run's weights need the depth export start"));
    }
  }
  return issues;
}

// ---- Training resources (restoration) -----------------------------------------------------

export function irisTrainingResourceIds(target) {
  return (Array.isArray(target?.trainingResources) ? target.trainingResources : [])
    .map((resource) => resource?.modelId)
    .filter(Boolean);
}

// ---- Pre-launch dataset check -------------------------------------------------------------

export const POLICY_TEXT = {
  reject: "Reject: the run refuses to start, naming the first invalid item.",
  skip: "Skip: invalid items are dropped with a warning.",
  pairs: "Aligned pairs: each item trains on its degraded image.",
  synthetic: "Clean images: degraded inputs are synthesized.",
};

// The worker's report (`iris_task_dataset_report`; shape pinned by irisDatasetReport.fixture.json,
// which a worker test compares with a real report): `ok` / `invalid` / `errors` are counts and
// `trainable` is whether the run would start under its policy.
export function irisCheckSummary(report) {
  if (!report) return null;
  const items = Array.isArray(report.items) ? report.items : [];
  return {
    counts: `${report.ok ?? 0} trainable · ${report.invalid ?? 0} invalid · ${report.errors ?? 0} unreadable`,
    ready: report.trainable === true,
    state: report.trainable === true ? "Ready to train" : "Blocked — this run would not start",
    policy: POLICY_TEXT[report.policy] ?? asText(report.policy),
    problems: items.filter((item) => item?.status !== "ok").slice(0, 5),
    configError: report.configError || null,
  };
}

// ---- Sidecars -----------------------------------------------------------------------------

export const IRIS_SIDECAR_KINDS = ["depth", "mask", "degraded"];

function stem(name) {
  const base = asText(name).split(/[\\/]/).pop();
  const dot = base.lastIndexOf(".");
  return (dot > 0 ? base.slice(0, dot) : base).toLowerCase();
}

// The item stem a sidecar file names: `<stem>.ext`, or for depth/mask `<stem>_depth.ext` /
// `<stem>.depth.ext` (`_mask` / `.mask`).
export function sidecarStem(fileName, kind) {
  const name = stem(fileName);
  const suffix = kind === "degraded" ? null : new RegExp(`[._]${kind}$`);
  return suffix ? name.replace(suffix, "") : name;
}

// Match bulk-picked files to saved dataset items by stem (the item's display name or stored file
// name). `{ matches: [{ item, file }], unmatched: [fileName] }`.
export function matchSidecarFiles(items, files, kind) {
  const byStem = new Map();
  for (const item of items ?? []) {
    if (!item?.id) continue;
    for (const key of [stem(item.displayName), stem(item.path)]) {
      if (key && !byStem.has(key)) byStem.set(key, item);
    }
  }
  const matches = [];
  const unmatched = [];
  for (const file of files ?? []) {
    const item = byStem.get(sidecarStem(file?.name, kind));
    if (item) matches.push({ item, file });
    else unmatched.push(file?.name);
  }
  return { matches, unmatched };
}

export function itemSidecarState(item) {
  return {
    depth: Boolean(item?.irisDepth?.path),
    mask: Boolean(item?.irisDepth?.maskPath),
    degraded: Boolean(item?.irisDegradedImage),
  };
}

// The objective's options bag inside an `advanced` object (`iris` for depth, `iris_restore` for
// restoration), or undefined.
export function irisTaskBagOf(objective, advancedObject) {
  if (objective === "depth") return advancedObject?.iris;
  if (objective === "restoration") return advancedObject?.iris_restore;
  return undefined;
}
