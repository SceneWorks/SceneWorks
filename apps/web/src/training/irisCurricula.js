// Iris-3B curricula, REPA, holdout, fp16 and distributed training (sc-25689) — the web half of
// `crates/sceneworks-core/src/training/iris/curricula.rs`. The S11 keys ride in the same
// `advanced.iris` bag as the S7 options (irisOptions.js); nested ones (`repa`, `dist`, `perf`,
// `tracking`) are drafted under dotted keys (`repa.weight`, `dist.world.world_size`) and nested
// again by the snapshot. A key the run's own choices make inert is not in play: never rendered,
// sent or validated (E3). Pure: no React.

const between = (min) => ({ check: (x) => x >= min, expect: `at least ${min}` });
const text = (label, maxLength = 64) => ({ kind: "text", maxLength, label });
const int = (label, min = 1) => ({ kind: "int", min, label });
const choice = (label, values) => ({ kind: "choice", values, label });
const bool = (label) => ({ kind: "bool", label });

export const IRIS_CURRICULA_SPECS = Object.freeze({
  stage: text("Curriculum stage"),
  parent_stage: text("Parent stage"),
  run_name: text("Run name", 200),
  "tracking.project": text("Tracking project", 200),
  "tracking.tags": { kind: "strings", label: "Tracking tags" },
  shape_policy: choice("Shape policy", ["fixed", "bucket", "area"]),
  aspect_ratio_bucket: choice("Aspect buckets", ["shared21-512", "shared21-1024"]),
  shape_align: int("Shape alignment"),
  shape_max_ratio: { kind: "number", ...between(1), label: "Max aspect ratio" },
  rope_aspect: choice("RoPE law", ["square", "isotropic"]),
  "repa.weight": { kind: "number", ...between(0), label: "REPA weight" },
  "repa.variant": choice("REPA variant", ["repa", "irepa"]),
  "repa.spatial_norm_gamma": { kind: "number", label: "iREPA spatial-norm gamma" },
  "repa.proj_hidden_dim": int("Projector width"),
  "repa.teacher_image_size": int("Teacher image size", 14),
  "repa.teacher_match_student": bool("Teacher sees the student's resolution"),
  "repa.layer": int("Aligned block"),
  activation_checkpointing: choice("Activation checkpointing", ["none", "full", "selective_op", "selective_layer"]),
  ac_selective_every: int("Checkpoint every Nth block"),
  val_every_steps: int("Validate every (optimizer steps)", 0),
  val_samples: int("Holdout samples per validation", 0),
  log_every: int("Log every (optimizer steps)"),
  log_block_grad_norms: bool("Log per-block gradient norms"),
  save_every_epochs: int("Save every N epochs", 0),
  resume_optimizer: choice("Resume optimizer state", ["full", "core"]),
  expected_global_batch: int("Expected global batch"),
  scale_warmup_by_world: bool("Scale warmup by rank count"),
  "perf.lazy_sync": bool("Lazy sync (non-finite guard on the clipped norm)"),
  preview_at_step_one: bool("Also preview at step 1"),
  preview_order: int("Preview solver order"),
  "dist.strategy": choice("Strategy", ["ddp", "fsdp2"]),
  "dist.sharding": choice("Sharding", ["full", "grad_op", "hybrid", "hybrid_grad_op"]),
  "dist.mesh.dp_shard": int("Ranks per sharded copy", 0),
  "dist.mesh.dp_replicate": int("Replicated copies", 0),
  "dist.param_dtype": choice("Gathered parameter precision", ["bf16", "fp16", "fp32"]),
  "dist.reduce_dtype": choice("Reduction precision", ["fp32", "bf16"]),
  "dist.reshard_after_forward": bool("Reshard after forward"),
  "dist.checkpoint_io": choice("Checkpoint writing", ["auto", "rank0", "dcp"]),
  "dist.world.world_size": int("Ranks"),
  "dist.world.share_device": bool("Share one device between ranks"),
  "dist.world.timeout_secs": int("Rank timeout (s)"),
  // sc-25690: the engine refuses 0 for both (min 1).
  "dist.world.rendezvous_timeout_secs": int("Rank start-up timeout (s)"),
  "dist.world.cancel_grace_secs": int("Cancel grace (s)"),
});

export const IRIS_CURRICULA_KEYS = Object.freeze(Object.keys(IRIS_CURRICULA_SPECS));

const str = (value) => (value === undefined || value === null ? "" : String(value)).trim();

// Read a dotted key from a nested bag (`dist.world.world_size`).
export function dottedValue(source, key) {
  return key.split(".").reduce((value, part) => (value && typeof value === "object" ? value[part] : undefined), source);
}

// Nest dotted keys back into objects for the request.
export function nestDotted(flat) {
  const out = {};
  for (const [key, value] of Object.entries(flat)) {
    const parts = key.split(".");
    let node = out;
    parts.slice(0, -1).forEach((part) => {
      node = node[part] ??= {};
    });
    node[parts.at(-1)] = value;
  }
  return out;
}

function previewsOn(draft) {
  return Number(draft.sampleEvery) > 0 && (draft.iris?.upstream_validation_prompts === true || str(draft.samplePrompts) !== "");
}

// Whether an S11 key applies to this run (see the module comment).
export function curriculaInPlay(key, draft = {}) {
  const iris = draft.iris ?? {};
  const fsdp = str(iris["dist.strategy"]) === "fsdp2";
  switch (key) {
    case "parent_stage":
      return str(iris.stage) !== "";
    case "aspect_ratio_bucket":
      return str(iris.shape_policy) === "bucket";
    case "shape_align":
    case "shape_max_ratio":
      return str(iris.shape_policy) === "area";
    case "repa.spatial_norm_gamma":
      return Number(iris["repa.weight"]) > 0 && str(iris["repa.variant"]) === "irepa";
    case "ac_selective_every":
      return str(iris.activation_checkpointing) === "selective_layer";
    case "val_samples":
      return Number(iris.val_every_steps) > 0;
    case "resume_optimizer":
      return str(draft.resumeFromJobId) !== "";
    case "preview_at_step_one":
    case "preview_order":
      return previewsOn(draft);
    case "dist.sharding":
    case "dist.param_dtype":
    case "dist.reshard_after_forward":
      return fsdp;
    case "dist.mesh.dp_shard":
    case "dist.mesh.dp_replicate":
      return fsdp && str(iris["dist.sharding"]).startsWith("hybrid");
    default:
      return key.startsWith("repa.") && key !== "repa.weight" ? Number(iris["repa.weight"]) > 0 : true;
  }
}

// The global batch a draft trains with (`batch × ranks × accumulation`).
export function irisGlobalBatch(draft = {}) {
  const ranks = Math.max(1, Number(draft.iris?.["dist.world.world_size"]) || 1);
  return Math.max(1, Number(draft.batchSize) || 1) * ranks * Math.max(1, Number(draft.gradientAccumulation) || 1);
}

// Client-side rules the server would refuse anyway, flagged on their field.
export function curriculaIssues(draft = {}) {
  const iris = draft.iris ?? {};
  const out = [];
  if (str(iris["dist.checkpoint_io"]) === "dcp" && str(iris["dist.strategy"]) !== "fsdp2") {
    out.push(["iris.dist.checkpoint_io", "Per-rank (dcp) checkpoints need the fsdp2 strategy"]);
  }
  const expected = Number(iris.expected_global_batch);
  if (str(iris.expected_global_batch) && expected !== irisGlobalBatch(draft)) {
    out.push([
      "iris.expected_global_batch",
      `Global batch ${irisGlobalBatch(draft)} (batch × ranks × accumulation) does not match the expected ${expected}`,
    ]);
  }
  return out;
}

// Assign the holdout flag of dataset items: `{ [selectionId]: true | false }` overrides.
export function itemSplit(holdoutById, selectionId, savedItem) {
  if (holdoutById && selectionId in holdoutById) return holdoutById[selectionId] ? "holdout" : "train";
  return savedItem?.split;
}

// The grouped validation records of a run's metrics (`val/loss`, `val/loss_t0..3`).
export function validationRows(metrics) {
  return (metrics?.validation ?? []).map((record) => ({
    step: record.step,
    loss: record["val/loss"],
    bands: [0, 1, 2, 3].map((band) => record[`val/loss_t${band}`]),
  }));
}
