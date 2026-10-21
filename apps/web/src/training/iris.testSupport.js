// Test-only fixtures for the Iris-3B training target (sc-25685/sc-25686). Mirrors the JSON the API
// serializes from `crates/sceneworks-core/src/training/iris.rs::iris_generation_target()` so the
// web tests exercise the real contract shape (camelCase, `memoryFloors` flattened to the top).

export const IRIS_CLIENT_OPTION_KEYS = Object.freeze([
  "init", "prediction", "flow_shift", "shift_law", "shift_base_tokens", "logit_mean",
  "logit_std", "x_pred_sigma_min", "text_dropout", "gradient_clip", "betas", "muon_momentum",
  "muon_nesterov", "muon_adjust_lr", "auto_lr", "base_batch_size", "ema_enabled", "ema_decay",
  "num_epochs", "nan_loss_tolerance", "caption_field", "caption_fields", "on_caption_overflow",
  "text_conditioning", "keep_last_checkpoints", "milestone_steps", "resume_data_policy",
  "override_lr_on_resume", "preview_weights", "preview_negative_prompt", "export_weights",
  "export_dtype", "upstream_validation_prompts",
]);

export function irisTarget() {
  return {
    id: "iris_3b_generation",
    name: "Iris 3B",
    modality: "image",
    outputKind: "lora",
    family: "iris",
    baseModel: "iris_3b",
    baseModelRepo: "speridlabs/iris-3b",
    kernel: "iris_train",
    defaults: {
      rank: 16,
      alpha: 16,
      learningRate: 0.0001,
      steps: 1000,
      batchSize: 1,
      gradientAccumulation: 1,
      resolution: 256,
      saveEvery: 250,
      seed: 42,
      optimizer: "adamw",
      advanced: {
        mixedPrecision: "bf16",
        networkType: "lora",
        decomposeFactor: -1,
        timestepType: "logit_normal",
        timestepBias: "balanced",
        lossType: "mse",
        weightDecay: 0.0,
        lrScheduler: "constant",
        lrWarmupSteps: 0,
        gradientCheckpointing: false,
        sampleEvery: 250,
        sampleSteps: 30,
        sampleGuidanceScale: 3.0,
        sampleCount: 4,
        qualityPreset: "balanced",
        outputScope: "project",
        requestedGpu: "auto",
        iris: {
          init: "weights",
          text_conditioning: "cached",
          text_dropout: 0.1,
          gradient_clip: 0.5,
          ema_enabled: true,
          ema_decay: 0.9999,
          on_caption_overflow: "warn",
          caption_field: "caption",
          caption_fields: [],
          keep_last_checkpoints: 2,
          milestone_steps: [],
          nan_loss_tolerance: 20,
          betas: [0.9, 0.95],
          auto_lr: "none",
          base_batch_size: 256,
          preview_negative_prompt: "",
          upstream_validation_prompts: false,
          export_dtype: "fp32",
        },
      },
    },
    limits: {
      rank: [1, 128],
      alpha: [1, 128],
      steps: [1, 200000],
      batchSize: [1, 64],
      resolutions: [256, 512, 1024],
      sampleCount: [1, 4],
      optimizers: ["adamw", "muon"],
      networkTypes: ["lora", "lokr", "full"],
      lrSchedulers: ["constant", "cosine"],
      timestepTypes: ["logit_normal", "uniform"],
      timestepBiases: ["balanced"],
      lossTypes: ["mse"],
      mixedPrecisions: ["bf16", "f32"],
      gradientCheckpointing: false,
      microStepFields: ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
      inits: ["weights", "random"],
      outputScopes: ["project", "global"],
    },
    ui: { label: "Iris 3B", recommendedFor: ["character", "style", "concept"], datasetModality: "image" },
    memoryFloors: {
      unit: "GiB",
      basis: "Floors are GiB.",
      lanes: {
        mlx: {
          adapter: { 256: 16, 512: 24, 1024: 40 },
          full: { 256: 80, 512: 88, 1024: 104 },
          full_ac: { 256: 80, 512: 88, 1024: 96 },
        },
        candle: {
          adapter: { 256: 24, 512: 32, 1024: 48 },
          full: { 256: 80, 512: 88, 1024: 104 },
        },
      },
      provenance: {
        mlx: {
          adapter: { 256: "declared", 512: "declared", 1024: "declared" },
          full: { 256: "extrapolated", 512: "measured", 1024: "measured" },
          full_ac: { 256: "extrapolated", 512: "extrapolated", 1024: "measured" },
        },
        candle: {
          adapter: { 256: "declared", 512: "declared", 1024: "declared" },
          full: { 256: "extrapolated", 512: "measured", 1024: "extrapolated" },
        },
      },
    },
  };
}

// A terminal Iris lora_train job as the jobs list carries it (`payload.plan` is the TrainingPlan).
export function irisJob({
  id,
  status = "completed",
  networkType = "lora",
  outputName = `Run ${id}`,
  projectId = "p1",
  targetId = "iris_3b_generation",
  result = {},
  type = "lora_train",
  createdAt = "2026-10-01T00:00:00Z",
} = {}) {
  return {
    id,
    type,
    status,
    projectId,
    createdAt,
    payload: {
      outputName,
      plan: {
        target: { targetId, kernel: "iris_train" },
        config: { advanced: { networkType } },
      },
    },
    result,
  };
}

// The depth / restoration targets (sc-25687 / sc-25688), mirroring
// `crates/sceneworks-core/src/training/iris/{depth,restore}.rs` as serialized by the API.
const TASK_LIMITS = {
  lrSchedulers: ["constant", "cosine"],
  mixedPrecisions: ["bf16", "f32"],
  gradientCheckpointingLanes: ["candle"],
  microStepFields: ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
  outputScopes: ["project", "global"],
};

export function irisDepthTarget() {
  return {
    id: "iris_3b_depth",
    name: "Iris 3B Depth",
    modality: "image",
    outputKind: "base_checkpoint",
    family: "iris",
    baseModel: "iris_3b_depth",
    kernel: "iris_train",
    irisObjective: "depth",
    depthTraining: { sizeMin: 64, sizeMax: 2048, sizeMultiple: 16, inits: ["export", "parent"] },
    defaults: {
      rank: 1, alpha: 1, learningRate: 3e-5, steps: 320000, batchSize: 1, gradientAccumulation: 32,
      resolution: 640, saveEvery: 32000, seed: 42, optimizer: "adamw",
      advanced: {
        mixedPrecision: "bf16", networkType: "full", lossType: "l1", weightDecay: 0.01, lrScheduler: "constant",
        lrWarmupSteps: 0, gradientCheckpointing: false, sampleEvery: 32000, outputScope: "project",
        requestedGpu: "auto", depthInit: "export",
        iris: {
          depth_size: [480, 640], resize: "cover", crop: "center", hflip_prob: 0.5, gradient_weight: 5,
          low_percentile: 2, high_percentile: 98, clip_target: false, loss_reduction: "valid_pixels",
          invalid_examples: "reject", min_valid_pixels: 2, betas: [0.9, 0.999], gradient_clip: 1,
          ema_enabled: false, nan_loss_tolerance: 20, keep_last_checkpoints: 2, milestone_steps: [],
          preview_items: 4, preview_weights: "raw", export_weights: "raw",
        },
      },
    },
    limits: { ...TASK_LIMITS, optimizers: ["adamw", "muon"], networkTypes: ["full"], lossTypes: ["l1"] },
    ui: { label: "Iris 3B Depth", objective: "depth" },
    memoryFloors: {
      unit: "GiB", key: "microBatchPixels", basis: "Depth floors.",
      lanes: {
        mlx: { 16384: 64, 307200: 80, 589824: 96 },
        candle: { 16384: 68, 307200: 84, 589824: 100, 1048576: 111, full_ac: { 307200: 68, 1048576: 68 } },
      },
      provenance: {
        mlx: { 16384: "declared", 307200: "declared", 589824: "declared" },
        candle: {
          16384: "declared", 307200: "extrapolated", 589824: "extrapolated", 1048576: "measured_spill",
          full_ac: { 307200: "extrapolated", 1048576: "measured" },
        },
      },
    },
  };
}

export function irisRestoreTarget() {
  return {
    id: "iris_3b_restore",
    name: "Iris 3B Restoration",
    modality: "image",
    outputKind: "base_checkpoint",
    family: "iris",
    baseModel: "iris_3b_restore",
    kernel: "iris_train",
    irisObjective: "restoration",
    trainingResources: [
      { modelId: "iris_restore_train_vgg16", component: "lpips_vgg16" },
      { modelId: "iris_restore_train_lpips_lin", component: "lpips_lin" },
      { modelId: "iris_restore_train_critic_trunk", component: "critic_trunk" },
    ],
    defaults: {
      rank: 16, alpha: 16, learningRate: 1e-5, steps: 10000, batchSize: 1, gradientAccumulation: 1,
      resolution: 1024, saveEvery: 1000, seed: 42, optimizer: "adamw",
      advanced: {
        mixedPrecision: "bf16", networkType: "full", decomposeFactor: -1, lossType: "mse", weightDecay: 0.01,
        lrScheduler: "constant", lrWarmupSteps: 0, gradientCheckpointing: false, sampleEvery: 1000,
        outputScope: "project", requestedGpu: "auto",
        iris_restore: {
          loss_l2: 1, loss_lpips: 5, loss_gan: 0.5, gan_real_label: 0.8, betas: [0.9, 0.999], gradient_clip: 1,
          critic_gradient_clip: 1, ema_enabled: true, ema_decay: 0.999, crop: "random", critic_dtype: "bf16",
          init: "export", nan_loss_tolerance: 20, keep_last_checkpoints: 2, milestone_steps: [], export_dtype: "fp32",
        },
      },
    },
    limits: {
      ...TASK_LIMITS, optimizers: ["adamw"], networkTypes: ["full", "lora", "lokr"], lossTypes: ["mse"],
      resolutions: [256, 512, 1024], microStepMultiple: "2xGradientAccumulation",
    },
    ui: { label: "Iris 3B Restoration", objective: "restoration" },
    memoryFloors: {
      unit: "GiB", key: "microBatchPixels", basis: "Restoration floors.",
      lanes: { mlx: { full: { 65536: 92, 1048576: 128 }, adapter: { 65536: 28, 1048576: 56 } } },
      provenance: {
        mlx: { full: { 65536: "declared", 1048576: "lower_bound" }, adapter: { 65536: "declared", 1048576: "declared" } },
      },
    },
  };
}
