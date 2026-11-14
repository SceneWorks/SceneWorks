//! Training creation, plan configuration and family-specific options.
//!
//! The effective target and the selected preset seed the Studio's draft. The API validates the
//! submitted config as a snapshot. An absent advanced field can therefore have a different
//! legacy engine fallback from the selected target's published default.
use crate::*;
use DefaultRule::{Absent, Inherit, Literal, Provider, Required, Target};
use IntentClass::{Context, MachineBudget, WhatToMake};
use Predicate::{Any, TrainingId, TrainingKernel, TrainingTarget};
use Shape::{Bool, Integer, List, Number, Numbers, Object, OneOf, Strings, Text};

const TARGET: Predicate = TrainingTarget("id");
const GENERATION: Predicate = TrainingId("iris_3b_generation");
const DEPTH: Predicate = TrainingId("iris_3b_depth");
const RESTORE: Predicate = TrainingId("iris_3b_restore");
const GENERATION_OR_DEPTH: Predicate = Any(&[GENERATION, DEPTH]);
const FLUX3: Predicate = Any(&[
    TrainingKernel("flux3_klein_lora"),
    TrainingKernel("flux3_klein_edit_lora"),
]);
const MOSS: Predicate = TrainingKernel("moss_image_lora");
const LTX25: Predicate = TrainingId("ltx_2_5_video_lora");

const MICRO_STEPS: &str = "iris::validate_micro_steps (generation/depth: gradientAccumulation; restoration: 2 × gradientAccumulation)";
const TECHNIQUE_COMPAT: &str =
    "JSON null is absence (training::technique_value). Presets may supply a non-neutral value.";
const LTX_CONDITION: &str = "trainingConfig::ltx25WorkflowPlan / LTX condition_from_value";
const IRIS_GENERATION_RULE: &str = "training::iris::validate_iris_training_config";
const IRIS_DEPTH_RULE: &str = "training::iris::depth::validate_depth_config";
const IRIS_RESTORE_RULE: &str = "training::iris::restore::validate_restore_config";
const CURRICULA_RULE: &str = "training::iris::curricula::validate_curricula_config";
const DEGRADATION_RULE: &str = "gen_core_iris::restoration_train::parse_degradation / training::iris::restore::validate_restore_dataset";
const FLUX3_RULE: &str = "training::flux3::validate";
// Every nested default is either an authored target field, a verified JSON literal, or the
// named native resolver. This keeps stage, EMA, model and backend-dependent defaults qualified.
// The common preflight accepts numeric strings and boolean strings. Target-specific validators
// can require strict JSON numbers/booleans before that worker compatibility layer runs.

macro_rules! submit {
    ($key:literal, $shape:expr, $default:expr, $intent:expr, $binding:literal) => {
        OptionSpec::new(
            concat!("training.", $key),
            "training",
            $key,
            $shape,
            $default,
            $intent,
            $binding,
        )
    };
}
macro_rules! config {
    ($key:literal, $shape:expr, $intent:expr, $field:literal) => {
        OptionSpec::new(
            concat!("training.config.", $key),
            "training.config",
            $key,
            $shape,
            Target(concat!("defaults.", $key)),
            $intent,
            concat!("TrainingConfig.", $field),
        )
        .applies(TARGET)
    };
}
macro_rules! advanced {
    ($key:literal, $shape:expr, $default:expr, $intent:expr, $binding:expr) => {
        OptionSpec::new(
            concat!("training.advanced.", $key),
            "training.config.advanced",
            $key,
            $shape,
            $default,
            $intent,
            $binding,
        )
        .applies(TARGET)
    };
}
macro_rules! common {
    ($key:literal, $shape:expr, $intent:expr, $field:literal, $fallback:literal) => {
        advanced!($key, $shape, Target(concat!("defaults.advanced.", $key)), $intent, concat!("TrainingConfig.", $field))
             .compat(concat!("Effective target/preset default; absent in a legacy plan: ", $fallback,
                ". training_jobs::map_training_config and preflight_string/bool/u32/i32/f32 own legacy coercions. Target validation can be stricter."))
    };
}
macro_rules! nested {
    ($parent:literal, $key:literal, $shape:expr, $default:expr, $intent:expr, $predicate:expr, $rule:expr) => {
        OptionSpec::new(
            concat!("training.advanced.", $parent, ".", $key),
            concat!("training.config.advanced.", $parent),
            $key,
            $shape,
            $default,
            $intent,
            concat!("TrainingConfig.model_options.", $parent, ".", $key),
        )
        .applies($predicate)
        .limits(&[$rule])
    };
}
macro_rules! technique {
    ($key:literal, $shape:expr, $fallback:literal, $flag:literal, $binding:literal, $rule:expr) => {
        advanced!(
            $key,
            $shape,
            Target(concat!("defaults.advanced.", $key)),
            WhatToMake,
            $binding
        )
        .applies(TrainingTarget(concat!("limits.", $flag)))
        .limits(&[$rule])
        .compat(concat!(
            "Target/preset default. Null or absence in the submitted snapshot resolves to ",
            $fallback,
            " (training::technique_value)."
        ))
    };
}
macro_rules! effect {
    ($when:literal, $action:literal, $target:literal) => {
        Effect {
            when: $when,
            action: $action,
            target: $target,
        }
    };
}

pub(crate) const SPECS: &[OptionSpec] = &[
    submit!("targetId", Text, Required, WhatToMake, "LoraTrainingRequest.target_id → TrainingPlan.target"),
    submit!("datasetId", Text, Required, WhatToMake, "LoraTrainingRequest.dataset_id → TrainingPlan.dataset"),
    submit!("datasetVersion", Integer, Inherit("dataset.version"), Context, "LoraTrainingRequest.dataset_version"),
    submit!("presetId", Text, Absent, Context, "TrainingPresetProvenance.preset_id"),
    submit!("presetVersion", Integer, Inherit("preset.version"), Context, "TrainingPresetProvenance.preset_version")
        .limits(&["training::validate_training_submission: matching presetId/version"]),
    submit!("outputName", Text, Required, Context, "LoraTrainingRequest.output_name → training artifact name"),
    submit!("dryRun", Bool, Literal("true"), Context, "LoraTrainingRequest.dry_run")
        .effects(&[effect!("true", "validate plan without training", "lora_train")])
        .limits(&["training::validate_training_submission: control training requires false"]),
    submit!("config", Object("TrainingConfig"), Required, WhatToMake, "TrainingPlan.config"),
    submit!("licenseAcknowledged", Bool, Literal("false"), Context, "training_license_acknowledgment_error")
        .applies(TrainingTarget("ui.requiresLicenseAcknowledgment")),
    // The Studio sends these extra properties alongside the authoritative advanced carriers.
    submit!("outputScope", Text, Absent, Context, "LoraTrainingRequest.extra.outputScope")
        .compat("Studio echo; training_output_scope reads config.advanced.outputScope"),
    submit!("requestedGpu", Text, Absent, Context, "LoraTrainingRequest.extra.requestedGpu")
        .compat("Studio echo; training_requested_gpu reads config.advanced.requestedGpu"),
    submit!("qualityPreset", Text, Absent, Context, "LoraTrainingRequest.extra.qualityPreset")
        .compat("Studio quality label; presetId and presetVersion record preset identity"),
    config!("rank", Integer, WhatToMake, "rank").limits(&["target.limits.rank","training::validate_training_config"]),
    config!("alpha", Number, WhatToMake, "alpha").limits(&["target.limits.alpha","training::training_alpha_f32"])
        .compat("Finite positive f32; targets advertising integral alpha limits require a whole number"),
    config!("learningRate", Number, WhatToMake, "learning_rate").limits(&["training::validate_training_config: finite, positive"]),
    config!("steps", Integer, WhatToMake, "steps").limits(&["target.limits.steps","training::validate_training_config",MICRO_STEPS]),
    config!("batchSize", Integer, WhatToMake, "batch_size").limits(&["target.limits.batchSize","training::validate_training_config"]),
    config!("gradientAccumulation", Integer, WhatToMake, "gradient_accumulation").limits(&["training::validate_training_config",MICRO_STEPS]),
    config!("resolution", Integer, WhatToMake, "resolution").limits(&["target.limits.resolutions","training::validate_training_config"]),
    config!("saveEvery", Integer, Context, "save_every").limits(&[MICRO_STEPS]),
    config!("seed", Integer, WhatToMake, "seed").limits(&["training::moss_training_seed","training::flux3::validate"])
        .compat("MOSS and FLUX.3 require nonnegative seeds, including zero. Generic legacy lowering maps negatives to zero."),
    config!("optimizer", Text, WhatToMake, "optimizer").limits(&["target.limits.optimizers"])
        .compat("The engine normalizes accepted values adamw8bit to adamw and prodigyopt to prodigy; submit still validates target choices."),
    config!("triggerWord", Text, WhatToMake, "trigger_word")
        .compat("Optional on the wire; may be populated from a preset")
        .effects(&[effect!("nonempty", "include trigger in captions and output metadata", "TrainingPlan.output.trigger_words")]),
    config!("advanced", Object("TrainingConfig.advanced"), WhatToMake, "model_options"),
    common!("weightDecay", Number, WhatToMake, "weight_decay", "0"),
    common!("lrScheduler", Text, WhatToMake, "lr_scheduler", "constant").limits(&["target.limits.lrSchedulers","training::SUPPORTED_LR_SCHEDULERS"]),
    common!("lrWarmupSteps", Integer, WhatToMake, "lr_warmup_steps", "0").limits(&[MICRO_STEPS]),
    common!("networkType", Text, WhatToMake, "network_type", "lora").limits(&["target.limits.networkTypes"])
        .effects(&[effect!("full", "set true", "TrainingConfig.full_finetune"),effect!("full", "select base checkpoint for an adapter target", "TrainingPlan.target.output_kind")]),
    common!("decomposeFactor", Integer, WhatToMake, "decompose_factor", "-1").limits(&["training::validate_moss_training_field_shapes"]),
    common!("loraTargetModules", Strings, WhatToMake, "lora_target_modules", "[]").limits(&["training::validate_moss_training_field_shapes","training::flux3::validate"])
        .effects(&[effect!("empty", "use native module selection", "adapter target modules")]),
    common!("timestepType", Text, WhatToMake, "timestep_type", "sigmoid; MOSS uses moss_stage3_v1").limits(&["target.limits.timestepTypes"]),
    common!("timestepBias", Text, WhatToMake, "timestep_bias", "balanced").limits(&["target.limits.timestepBiases"]),
    common!("lossType", Text, WhatToMake, "loss_type", "mse").limits(&["target.limits.lossTypes"]),
    common!("mixedPrecision", Text, MachineBudget, "train_dtype", "bf16").limits(&["target.limits.mixedPrecisions","target.defaults.advanced.fullFinetuneConfig"])
        .effects(&[effect!("Iris fp16", "preserve fp16 after generic dtype normalization", "TrainingConfig.train_dtype"),
            effect!("Candle LTX", "set f32", "TrainingConfig.train_dtype")]),
    common!("gradientCheckpointing", Bool, MachineBudget, "gradient_checkpointing", "false")
        .limits(&["target.limits.gradientCheckpointing","target.defaults.advanced.fullFinetuneConfig"])
        .effects(&[effect!("candle_requires_gradient_checkpointing", "force on", "TrainingConfig.gradient_checkpointing")]),
    common!("cacheLatents", Bool, MachineBudget, "cache_latents", "true"),
    common!("cacheTextEmbeddings", Bool, MachineBudget, "cache_text_embeddings", "true"),
    common!("resume", Bool, Context, "resume", "false").limits(&["training::validate_training_submission / worker training preflight"]),
    common!("sampleEvery", Integer, Context, "sample_every", "0").limits(&[MICRO_STEPS,"training::flux3::validate"])
        .effects(&[effect!("0", "disable preview sampling", "training previews")]),
    common!("sampleSteps", Integer, Context, "sample_steps", "20"),
    common!("sampleGuidanceScale", Number, Context, "sample_guidance_scale", "1"),
    common!("sampleCount", Integer, Context, "sample_prompts", "4").limits(&["target.limits.sampleCount"])
        .effects(&[effect!("present", "truncate the prompt pool to this count", "training.advanced.samplePrompts")]),
    common!("samplePrompts", Strings, Context, "sample_prompts", "[]")
        .effects(&[effect!("present", "trim and discard empty prompts before applying sampleCount", "TrainingConfig.sample_prompts")]),
    common!("controlType", Text, WhatToMake, "control_type", "None").applies(TrainingKernel("krea_control")).limits(&["target.limits.controlTypes"]),
    advanced!("memoryLimitBytes", Integer, Absent, MachineBudget, "TrainingConfig.memory_limit_bytes").limits(&["positive u64"])
        .compat("Unsigned integers and trimmed numeric strings; null is absence"),
    advanced!("memoryLimitFraction", Number, Absent, MachineBudget, "TrainingConfig.memory_limit_fraction").limits(&["finite fraction in (0,1] representable as a positive f32"])
        .compat("Numbers and trimmed numeric strings; null is absence"),
    advanced!("outputScope", Text, Target("defaults.advanced.outputScope"), Context, "training_output_scope").limits(&["target.limits.outputScopes"]).compat("Absent or blank resolves to project"),
    advanced!("requestedGpu", Text, Target("defaults.advanced.requestedGpu"), MachineBudget, "training_requested_gpu").compat("Absent resolves to auto"),
    advanced!("personGate", Bool, Literal("false"), WhatToMake, "control_training_jobs::PersonFilter")
        .applies(TrainingTarget("limits.controlTypes"))
        .limits(&["control training person detector confidence floor"]),
    advanced!("qualityPreset", Text, Target("defaults.advanced.qualityPreset"), Context, "TrainingPlan.config.advanced.qualityPreset")
        .compat("Preset provenance label; no native trainer consumes it"),
    advanced!("optimizerDCoef", Number, Absent, WhatToMake, "TrainingConfig.model_options.optimizerDCoef")
        .compat("Built-in Prodigy preset carries 1.0; no native optimizer field consumes it"),
    advanced!("numFrames", Integer, Target("defaults.advanced.numFrames"), WhatToMake, "TrainingConfig.model_options.numFrames")
        .applies(TrainingTarget("defaults.advanced.numFrames")).compat("Legacy video preset field; carried in the plan"),
    advanced!("baseModelRepo", Text, Target("defaults.advanced.baseModelRepo"), Context, "TrainingConfig.model_options.baseModelRepo")
        .applies(TrainingKernel("lens_lora")).compat("Legacy Lens target metadata. Target identity supplies the plan base repository."),
    advanced!("backend", Text, Target("defaults.advanced.backend"), Context, "TrainingConfig.model_options.backend")
        .applies(TrainingKernel("sd3_lora")).internal(),
    advanced!("fullFinetuneConfig", Object("platform full-finetune requirements"), Target("defaults.advanced.fullFinetuneConfig"), Context, "training::validate_training_config_for_target")
        .applies(TrainingTarget("defaults.advanced.fullFinetuneConfig")).internal()
        .compat("Target-authored requirements. Studio removes this object and applies its fields to the submitted config."),
    advanced!("modelOptions", Object("MOSS training options"), Target("defaults.advanced.modelOptions"), WhatToMake, "TrainingConfig.model_options")
        .applies(MOSS).limits(&["training::validate_moss_training_field_shapes"]),
    nested!("modelOptions","vae_posterior",Text,Literal("\"posterior_mean_v1\""),WhatToMake,MOSS,"training::validate_moss_training_field_shapes"),
    advanced!("flux3", Object("FLUX.3 source options"), Target("defaults.advanced.flux3"), WhatToMake, "TrainingConfig.model_options.flux3").applies(FLUX3).limits(&[FLUX3_RULE]),
    // FLUX.3 source profile (gen_core_flux3::training::options).
    nested!("flux3", "mode", Text, Target("defaults.advanced.flux3.mode"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "caption_dropout", Number, Literal("0"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "max_sequence_length", Integer, Literal("512"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "center_crop", Bool, Literal("false"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "random_flip", Bool, Literal("false"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "weighting_scheme", Text, Inherit("training.advanced.timestepType"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "logit_mean", Number, Literal("0"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "logit_std", Number, Literal("1"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "mode_scale", Number, Literal("1.29"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "max_grad_norm", Number, Literal("1"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "lr_scheduler", Text, Inherit("training.advanced.lrScheduler"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "lr_num_cycles", Integer, Literal("1"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "lr_power", Number, Literal("1"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "with_prior_preservation", Bool, Literal("false"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "use_rslora", Bool, Literal("false"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "use_dora", Bool, Literal("false"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "lora_dropout", Number, Literal("0"), WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "rank_pattern", Object("empty map"), Absent, WhatToMake, FLUX3, FLUX3_RULE),
    nested!("flux3", "alpha_pattern", Object("empty map"), Absent, WhatToMake, FLUX3, FLUX3_RULE),
    // Target capability flags are installed by training::builtin_training_targets.
    technique!("weightNoiseSigma", Number, "0", "supportsWeightNoise", "TrainingConfig.weight_noise_sigma", "training::WEIGHT_NOISE_SIGMA_MAX [0,0.1]"),
    technique!("gradientNoiseEta", Number, "0", "supportsGradientNoise", "TrainingConfig.gradient_noise_eta", "training::GRADIENT_NOISE_ETA_MAX [0,0.1]"),
    technique!("gradientNoiseGamma", Number, "0.55", "supportsGradientNoise", "TrainingConfig.gradient_noise_gamma", "training::GRADIENT_NOISE_GAMMA_MAX [0,1]"),
    technique!("subjectMaskLoss", Bool, "false", "supportsSubjectMaskLoss", "TrainingConfig.subject_mask_loss", "training::subject_mask_loss_weights"),
    technique!("subjectMaskBackgroundWeight", Number, "0.1", "supportsSubjectMaskLoss", "TrainingConfig.subject_mask_loss", "training::subject_mask_loss_weights [0,1]"),
    technique!("subjectMaskSubjectWeight", Number, "1", "supportsSubjectMaskLoss", "TrainingConfig.subject_mask_loss", "training::subject_mask_loss_weights [0,1]"),
    technique!("depthAnchoringWeight", Number, "0", "supportsDepthAnchoring", "TrainingConfig.depth_anchoring", "depth_anchoring::depth_anchoring_settings: [0,1]").effects(&[effect!("0", "disable loss", "depth_anchoring")]),
    technique!("depthAnchoringMinT", Number, "0", "supportsDepthAnchoring", "TrainingConfig.depth_anchoring", "depth_anchoring::depth_anchoring_settings: [0,1], min <= max"),
    technique!("depthAnchoringMaxT", Number, "0.5", "supportsDepthAnchoring", "TrainingConfig.depth_anchoring", "depth_anchoring::depth_anchoring_settings: [0,1], min <= max"),
    technique!("depthAnchoringEvery", Integer, "2", "supportsDepthAnchoring", "TrainingConfig.depth_anchoring", "depth_anchoring::depth_anchoring_settings: [1,16]"),
    technique!("bodyProportionWeight", Number, "0", "supportsBodyProportionLoss", "TrainingConfig.body_losses.proportion", "body_losses::body_loss_settings: [0,1]").effects(&[effect!("0", "disable loss", "body_losses.proportion")]),
    technique!("bodyProportionMinT", Number, "0", "supportsBodyProportionLoss", "TrainingConfig.body_losses.proportion", "body_losses::body_loss_settings: [0,1], min <= max"),
    technique!("bodyProportionMaxT", Number, "0.5", "supportsBodyProportionLoss", "TrainingConfig.body_losses.proportion", "body_losses::body_loss_settings: [0,1], min <= max"),
    technique!("bodyProportionEvery", Integer, "2", "supportsBodyProportionLoss", "TrainingConfig.body_losses.proportion", "body_losses::body_loss_settings: [1,16]"),
    technique!("bodyShapeWeight", Number, "0", "supportsBodyShapeLoss", "TrainingConfig.body_losses.shape", "body_losses::body_loss_settings: [0,1]").effects(&[effect!("0", "disable loss", "body_losses.shape")]),
    technique!("bodyShapeMinT", Number, "0.4", "supportsBodyShapeLoss", "TrainingConfig.body_losses.shape", "body_losses::body_loss_settings: [0,1], min <= max"),
    technique!("bodyShapeMaxT", Number, "0.8", "supportsBodyShapeLoss", "TrainingConfig.body_losses.shape", "body_losses::body_loss_settings: [0,1], min <= max"),
    technique!("bodyShapeEvery", Integer, "2", "supportsBodyShapeLoss", "TrainingConfig.body_losses.shape", "body_losses::body_loss_settings: [1,16]"),
    technique!("normalWeight", Number, "0", "supportsNormalLoss", "TrainingConfig.body_losses.normal", "body_losses::body_loss_settings: [0,1]").effects(&[effect!("0", "disable loss", "body_losses.normal")]),
    technique!("normalMinT", Number, "0.4", "supportsNormalLoss", "TrainingConfig.body_losses.normal", "body_losses::body_loss_settings: [0,1], min <= max"),
    technique!("normalMaxT", Number, "0.8", "supportsNormalLoss", "TrainingConfig.body_losses.normal", "body_losses::body_loss_settings: [0,1], min <= max"),
    technique!("normalEvery", Integer, "2", "supportsNormalLoss", "TrainingConfig.body_losses.normal", "body_losses::body_loss_settings: [1,16]"),
    technique!("identityLossWeight", Number, "0", "supportsIdentityLoss", "TrainingConfig.identity_loss", "face_losses::identity_loss_settings: [0,1]").effects(&[effect!("0", "disable loss", "identity_loss")]),
    technique!("identityLossMinT", Number, "0", "supportsIdentityLoss", "TrainingConfig.identity_loss", "face_losses::identity_loss_settings: [0,1], min <= max"),
    technique!("identityLossMaxT", Number, "0.5", "supportsIdentityLoss", "TrainingConfig.identity_loss", "face_losses::identity_loss_settings: [0,1], min <= max"),
    technique!("identityLossEvery", Integer, "2", "supportsIdentityLoss", "TrainingConfig.identity_loss", "face_losses::identity_loss_settings: [1,16]"),
    technique!("faceLandmarkLossWeight", Number, "0", "supportsFaceLandmarkLoss", "TrainingConfig.face_landmark_loss", "face_losses::face_landmark_loss_settings: [0,1]").effects(&[effect!("0", "disable loss", "face_landmark_loss")]),
    technique!("faceLandmarkLossMinT", Number, "0", "supportsFaceLandmarkLoss", "TrainingConfig.face_landmark_loss", "face_losses::face_landmark_loss_settings: [0,1], min <= max"),
    technique!("faceLandmarkLossMaxT", Number, "0.5", "supportsFaceLandmarkLoss", "TrainingConfig.face_landmark_loss", "face_losses::face_landmark_loss_settings: [0,1], min <= max"),
    technique!("faceLandmarkLossEvery", Integer, "2", "supportsFaceLandmarkLoss", "TrainingConfig.face_landmark_loss", "face_losses::face_landmark_loss_settings: [1,16]"),
    technique!("vaeAnchorWeight", Number, "0", "supportsVaeAnchorLoss", "TrainingConfig.vae_anchor", "latent_perceptual::latent_loss_settings: [0,10]").effects(&[effect!("0", "disable loss", "vae_anchor")]),
    technique!("vaeAnchorMinT", Number, "0", "supportsVaeAnchorLoss", "TrainingConfig.vae_anchor", "latent_perceptual::latent_loss_settings: [0,1], min <= max"),
    technique!("vaeAnchorMaxT", Number, "0.5", "supportsVaeAnchorLoss", "TrainingConfig.vae_anchor", "latent_perceptual::latent_loss_settings: [0,1], min <= max"),
    technique!("vaeAnchorEvery", Integer, "1", "supportsVaeAnchorLoss", "TrainingConfig.vae_anchor", "latent_perceptual::latent_loss_settings: [1,16]"),
    technique!("latentLpipsWeight", Number, "0", "supportsLatentLpipsLoss", "TrainingConfig.latent_lpips", "latent_perceptual::latent_loss_settings: [0,10]").effects(&[effect!("0", "disable loss", "latent_lpips")]),
    technique!("latentLpipsMinT", Number, "0", "supportsLatentLpipsLoss", "TrainingConfig.latent_lpips", "latent_perceptual::latent_loss_settings: [0,1], min <= max"),
    technique!("latentLpipsMaxT", Number, "0.5", "supportsLatentLpipsLoss", "TrainingConfig.latent_lpips", "latent_perceptual::latent_loss_settings: [0,1], min <= max"),
    technique!("latentLpipsEvery", Integer, "1", "supportsLatentLpipsLoss", "TrainingConfig.latent_lpips", "latent_perceptual::latent_loss_settings: [1,16]"),
    technique!("bodyProportionIncludeHead", Bool, "false", "supportsBodyProportionLoss", "TrainingConfig.body_losses.include_head", "body_losses::body_loss_settings"),
    technique!("bodyShapeMinCos", Number, "0.2", "supportsBodyShapeLoss", "TrainingConfig.body_losses.shape_min_cos", "body_losses::body_loss_settings [-1,1]"),
    technique!("normalRestrictToSubject", Bool, "false", "supportsNormalLoss", "TrainingConfig.body_losses.normal_restrict_to_subject", "body_losses::body_loss_settings: mask required when enabled"),
    technique!("identityLossMinCos", Number, "0.2", "supportsIdentityLoss", "TrainingConfig.identity_loss.min_cos", "face_losses::identity_loss_settings [-1,1]"),
    technique!("identityLossReference", Text, "\"dataset_average\"", "supportsIdentityLoss", "TrainingConfig.identity_loss.reference", "face_losses::IDENTITY_LOSS_REFERENCES"),
    technique!("depthAnchoringModel", Text, "\"small\"", "supportsDepthAnchoring", "TrainingConfig.depth_anchoring.model_size", "depth_anchoring::DEPTH_ANCHORING_MODELS"),
    advanced!("resolutionBuckets",List("ResolutionBucketConfig"),Absent,WhatToMake,"TrainingConfig.resolution_buckets or aspect_ratio_buckets").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]).compat(TECHNIQUE_COMPAT).effects(&[effect!("both bucket lists present", "refuse ambiguous geometry", "TrainingConfig.aspect_ratio_buckets")]),
    OptionSpec::new("training.advanced.resolutionBuckets[].resolution", "training.config.advanced.resolutionBuckets[]", "resolution", Integer, Required, WhatToMake, "TrainingConfig.resolution_buckets[].resolution").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]),
    OptionSpec::new("training.advanced.resolutionBuckets[].width", "training.config.advanced.resolutionBuckets[]", "width", Integer, Required, WhatToMake, "TrainingConfig.aspect_ratio_buckets[].width").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]),
    OptionSpec::new("training.advanced.resolutionBuckets[].height", "training.config.advanced.resolutionBuckets[]", "height", Integer, Required, WhatToMake, "TrainingConfig.aspect_ratio_buckets[].height").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]),
    OptionSpec::new("training.advanced.resolutionBuckets[].repeats", "training.config.advanced.resolutionBuckets[]", "repeats", Integer, Required, WhatToMake, "TrainingConfig.resolution_buckets[].repeats / aspect_ratio_buckets[].repeats").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]),
    advanced!("aspectRatioBuckets",List("AspectRatioBucket"),Absent,WhatToMake,"TrainingConfig.aspect_ratio_buckets").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]).compat(TECHNIQUE_COMPAT).effects(&[effect!("both bucket lists present", "refuse ambiguous geometry", "TrainingConfig.aspect_ratio_buckets")]),
    OptionSpec::new("training.advanced.aspectRatioBuckets[].width", "training.config.advanced.aspectRatioBuckets[]", "width", Integer, Required, WhatToMake, "TrainingConfig.aspect_ratio_buckets[].width").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]),
    OptionSpec::new("training.advanced.aspectRatioBuckets[].height", "training.config.advanced.aspectRatioBuckets[]", "height", Integer, Required, WhatToMake, "TrainingConfig.aspect_ratio_buckets[].height").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]),
    OptionSpec::new("training.advanced.aspectRatioBuckets[].repeats", "training.config.advanced.aspectRatioBuckets[]", "repeats", Integer, Required, WhatToMake, "TrainingConfig.resolution_buckets[].repeats / aspect_ratio_buckets[].repeats").applies(TrainingTarget("limits.supportsResolutionBuckets")).limits(&["training::parse_resolution_bucket_config","training_jobs::map_aspect_ratio_buckets"]),
    advanced!("trainingAdapterRepo",Text,Absent,WhatToMake,"TrainingConfig.training_adapter").applies(TrainingTarget("limits.supportsTrainingAdapter")).limits(&["training_adapter::training_adapter_model"]).compat("Null and blank are absent. With the supported repo, v2-default is an alias of v2."),
    advanced!("trainingAdapterVersion",Text,Inherit("training_adapter::training_adapter_model (v2 when a repo is selected)"),WhatToMake,"TrainingConfig.training_adapter").applies(TrainingTarget("limits.supportsTrainingAdapter")).limits(&["training_adapter::training_adapter_model"]).compat("Null and blank are absent. With the supported repo, v2-default is an alias of v2."),
    // LTX-2.5 modality and validation configuration (ltx25WorkflowPlan and native LTX parser).
    advanced!("ltxWorkflow",Text,Target("defaults.advanced.ltxWorkflow"),WhatToMake,"TrainingConfig.model_options.ltxWorkflow").applies(LTX25).limits(&["target.limits.ltxWorkflows"]),
    advanced!("ltxVideo",Object("LTX video modality"),Inherit("target.ui.ltxWorkflowPlans[ltxWorkflow].video"),WhatToMake,"TrainingConfig.model_options.ltxVideo").applies(LTX25).limits(&["target.limits.ltxWorkflows"]),
    advanced!("ltxAudio",Object("LTX audio modality"),Inherit("target.ui.ltxWorkflowPlans[ltxWorkflow].audio"),WhatToMake,"TrainingConfig.model_options.ltxAudio").applies(LTX25).limits(&["target.limits.ltxWorkflows"]),
    advanced!("ltxValidation",Object("Ltx25ValidationConfig"),Target("defaults.advanced.ltxValidation"),Context,"TrainingConfig.model_options.ltxValidation").applies(LTX25).limits(&["target.limits.ltxWorkflows"]),
    nested!("ltxVideo", "isGenerated", Bool, Inherit("target.ui.ltxWorkflowPlans[ltxWorkflow].video.isGenerated"), WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo", "conditions", List("LTX modality conditions"), Inherit("target.ui.ltxWorkflowPlans[ltxWorkflow].video.conditions"), WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo.conditions[]", "type", Text, Required, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo.conditions[]", "probability", Number, Required, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo.conditions[]", "tensorKey", Text, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo.conditions[]", "temporalBoundary", Integer, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo.conditions[]", "spatialRegion", Numbers, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo.conditions[]", "spatialScaleFactor", Integer, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxVideo.conditions[]", "temporalScaleFactor", Integer, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio", "isGenerated", Bool, Inherit("target.ui.ltxWorkflowPlans[ltxWorkflow].audio.isGenerated"), WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio", "conditions", List("LTX modality conditions"), Inherit("target.ui.ltxWorkflowPlans[ltxWorkflow].audio.conditions"), WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio.conditions[]", "type", Text, Required, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio.conditions[]", "probability", Number, Required, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio.conditions[]", "tensorKey", Text, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio.conditions[]", "temporalBoundary", Integer, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio.conditions[]", "spatialRegion", Numbers, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio.conditions[]", "spatialScaleFactor", Integer, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxAudio.conditions[]", "temporalScaleFactor", Integer, Absent, WhatToMake, LTX25, LTX_CONDITION),
    nested!("ltxValidation", "width", Integer, Target("defaults.advanced.ltxValidation.width"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "height", Integer, Target("defaults.advanced.ltxValidation.height"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "frames", Integer, Target("defaults.advanced.ltxValidation.frames"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "fps", Integer, Target("defaults.advanced.ltxValidation.fps"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "steps", Integer, Target("defaults.advanced.ltxValidation.steps"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "videoCfgScale", Number, Target("defaults.advanced.ltxValidation.videoCfgScale"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "audioCfgScale", Number, Target("defaults.advanced.ltxValidation.audioCfgScale"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "videoStgScale", Number, Target("defaults.advanced.ltxValidation.videoStgScale"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "audioStgScale", Number, Target("defaults.advanced.ltxValidation.audioStgScale"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "stgBlocks", Numbers, Target("defaults.advanced.ltxValidation.stgBlocks"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "guidanceRescale", Number, Target("defaults.advanced.ltxValidation.guidanceRescale"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "videoModalityGuidanceScale", Number, Target("defaults.advanced.ltxValidation.videoModalityGuidanceScale"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "audioModalityGuidanceScale", Number, Target("defaults.advanced.ltxValidation.audioModalityGuidanceScale"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    nested!("ltxValidation", "generateAudio", Bool, Target("defaults.advanced.ltxValidation.generateAudio"), Context, LTX25, "validateLtxValidationControls / LTX validation_from_value"),
    // Iris generation, downstream depth and restoration use different allowlists.
    advanced!("iris", Object("Iris generation or depth options"), Target("defaults.advanced.iris"), WhatToMake, "TrainingConfig.model_options.iris").applies(GENERATION_OR_DEPTH).limits(&["training::iris::iris_options"]),
    advanced!("iris_restore", Object("Iris restoration options"), Target("defaults.advanced.iris_restore"), WhatToMake, "TrainingConfig.model_options.iris_restore").applies(RESTORE).limits(&[IRIS_RESTORE_RULE]),
    advanced!("depthInit", Text, Target("defaults.advanced.depthInit"), WhatToMake, "training::iris::depth::depth_init").applies(DEPTH).limits(&[IRIS_DEPTH_RULE]),
    advanced!("resumeFromJobId", Text, Absent, Context, "Iris checkpoint selection → resume_from").applies(TrainingKernel("iris_train")).limits(&["iris_training::resolve_iris_run_sources"]),
    advanced!("loadFromJobId", Text, Absent, WhatToMake, "Iris prior full run → load_from").applies(GENERATION_OR_DEPTH).limits(&[IRIS_GENERATION_RULE]),
    nested!("iris", "init", Text, Target("defaults.advanced.iris.init"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Start from", "text"),
    nested!("iris", "prediction", Text, Inherit("Iris model config.flow.prediction"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Prediction target", "text"),
    nested!("iris", "flow_shift", Number, Inherit("Iris model config.flow.shift"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Flow shift", "number"),
    nested!("iris", "shift_law", Text, Inherit("Iris model config.flow.shift_law"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Shift law", "text"),
    nested!("iris", "shift_base_tokens", Integer, Provider("gen_core_iris::IrisTrainPlan::resolve: flow_defaults.shift_base_tokens"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Shift base tokens", "number"),
    nested!("iris", "logit_mean", Number, Provider("gen_core_iris::IrisTrainPlan::resolve: flow_defaults.logit_mean"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Logit-normal mean", "number"),
    nested!("iris", "logit_std", Number, Provider("gen_core_iris::IrisTrainPlan::resolve: flow_defaults.logit_std"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Logit-normal std", "number"),
    nested!("iris", "x_pred_sigma_min", Number, Provider("gen_core_iris::IrisTrainPlan::resolve: flow_defaults.x_pred_sigma_min"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("x-prediction sigma floor", "number"),
    nested!("iris", "text_dropout", Number, Target("defaults.advanced.iris.text_dropout"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Caption dropout", "number"),
    nested!("iris", "gradient_clip", Number, Target("defaults.advanced.iris.gradient_clip"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Gradient clip", "number"),
    nested!("iris", "betas", Numbers, Target("defaults.advanced.iris.betas"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Betas", "list"),
    nested!("iris", "muon_momentum", Number, Literal("0.95"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Muon momentum", "number"),
    nested!("iris", "muon_nesterov", Bool, Literal("true"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Muon Nesterov", "checkbox"),
    nested!("iris", "muon_adjust_lr", Text, Literal("\"rms_norm\""), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Muon LR adjustment", "text"),
    nested!("iris", "auto_lr", Text, Literal("\"none\""), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Auto LR scaling", "text"),
    nested!("iris", "base_batch_size", Integer, Literal("256"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Base batch size", "number"),
    nested!("iris", "ema_enabled", Bool, Target("defaults.advanced.iris.ema_enabled"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("EMA", "checkbox"),
    nested!("iris", "ema_decay", Number, Target("defaults.advanced.iris.ema_decay"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").compat("Depth omits this default when EMA is disabled; its engine fallback is 0.9999").ui("EMA decay", "number"),
    nested!("iris", "num_epochs", Integer, Absent, WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Epochs", "number"),
    nested!("iris", "nan_loss_tolerance", Integer, Target("defaults.advanced.iris.nan_loss_tolerance"), Context, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("NaN-loss tolerance", "number"),
    nested!("iris", "caption_field", Text, Target("defaults.advanced.iris.caption_field"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Caption field", "text"),
    nested!("iris", "caption_fields", Strings, Target("defaults.advanced.iris.caption_fields"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Caption fields", "list"),
    nested!("iris", "on_caption_overflow", Text, Target("defaults.advanced.iris.on_caption_overflow"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Caption overflow", "text"),
    nested!("iris", "text_conditioning", Text, Target("defaults.advanced.iris.text_conditioning"), MachineBudget, GENERATION, IRIS_GENERATION_RULE).ui("Text conditioning", "text"),
    nested!("iris", "keep_last_checkpoints", Integer, Target("defaults.advanced.iris.keep_last_checkpoints"), Context, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Keep last checkpoints", "number"),
    nested!("iris", "milestone_steps", Numbers, Target("defaults.advanced.iris.milestone_steps"), Context, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Milestone steps", "list"),
    nested!("iris", "resume_data_policy", Text, Literal("\"exact\""), Context, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Resume data policy", "text"),
    nested!("iris", "override_lr_on_resume", Bool, Literal("false"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Override LR on resume", "checkbox"),
    nested!("iris", "preview_weights", Text, Provider("gen_core_iris::IrisTrainPlan::resolve / DepthTrainPlan::resolve: artifact and EMA selection"), Context, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Preview weights", "text"),
    nested!("iris", "preview_negative_prompt", Text, Target("defaults.advanced.iris.preview_negative_prompt"), Context, GENERATION, IRIS_GENERATION_RULE).ui("Preview negative prompt", "text"),
    nested!("iris", "export_weights", Text, Provider("gen_core_iris::IrisTrainPlan::resolve / DepthTrainPlan::resolve: artifact and EMA selection"), WhatToMake, GENERATION_OR_DEPTH, "training::iris::validate_iris_training_config / depth::validate_depth_config").ui("Export weights", "text"),
    nested!("iris", "export_dtype", Text, Target("defaults.advanced.iris.export_dtype"), WhatToMake, GENERATION, IRIS_GENERATION_RULE).ui("Export precision", "text"),
    nested!("iris", "upstream_validation_prompts", OneOf("boolean or string list"), Target("defaults.advanced.iris.upstream_validation_prompts"), Context, GENERATION, IRIS_GENERATION_RULE).effects(&[effect!("enabled", "replace sample prompts and refuse a nonempty user prompt pool", "training.advanced.samplePrompts")]),
    nested!("iris", "depth_size", Numbers, Target("defaults.advanced.iris.depth_size"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).effects(&[effect!("present", "require resolution to equal the larger side", "training.config.resolution")]).ui("Training size", "list"),
    nested!("iris", "resize", Text, Target("defaults.advanced.iris.resize"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Resize", "text"),
    nested!("iris", "crop", Text, Target("defaults.advanced.iris.crop"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Crop", "text"),
    nested!("iris", "hflip_prob", Number, Target("defaults.advanced.iris.hflip_prob"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Horizontal flip probability", "number"),
    nested!("iris", "gradient_weight", Number, Target("defaults.advanced.iris.gradient_weight"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Gradient loss weight", "number"),
    nested!("iris", "low_percentile", Number, Target("defaults.advanced.iris.low_percentile"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Low percentile", "number"),
    nested!("iris", "high_percentile", Number, Target("defaults.advanced.iris.high_percentile"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("High percentile", "number"),
    nested!("iris", "clip_target", Bool, Target("defaults.advanced.iris.clip_target"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Clip the target to the percentiles", "checkbox"),
    nested!("iris", "loss_reduction", Text, Target("defaults.advanced.iris.loss_reduction"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Loss reduction", "text"),
    nested!("iris", "invalid_examples", Text, Target("defaults.advanced.iris.invalid_examples"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Invalid examples", "text"),
    nested!("iris", "min_valid_pixels", Integer, Target("defaults.advanced.iris.min_valid_pixels"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Minimum valid pixels", "number"),
    nested!("iris", "reducer_init", Text, Provider("gen_core_iris::train::depth::DepthTrainPlan::resolve"), WhatToMake, DEPTH, IRIS_DEPTH_RULE).ui("Depth head init", "text"),
    nested!("iris", "preview_items", Integer, Target("defaults.advanced.iris.preview_items"), Context, DEPTH, IRIS_DEPTH_RULE).ui("Preview items", "number"),
    nested!("iris", "stage", Text, Absent, WhatToMake, GENERATION, CURRICULA_RULE).ui("Curriculum stage", "text"),
    nested!("iris", "parent_stage", Text, Inherit("gen_core_iris::train::upstream::stage_preset(stage).parent"), WhatToMake, GENERATION, CURRICULA_RULE).ui("Parent stage", "text"),
    nested!("iris", "run_name", Text, Absent, Context, GENERATION, CURRICULA_RULE).ui("Run name", "text"),
    nested!("iris", "tracking", Object("Iris tracking options"), Absent, Context, GENERATION, CURRICULA_RULE),
    nested!("iris", "shape_policy", Text, Provider("gen_core_iris::IrisTrainPlan::resolve: fixed or translated stage shape"), WhatToMake, GENERATION, CURRICULA_RULE).ui("Shape policy", "text"),
    nested!("iris", "aspect_ratio_bucket", Text, Literal("\"shared21-512\""), WhatToMake, GENERATION, CURRICULA_RULE).ui("Aspect buckets", "text"),
    nested!("iris", "shape_align", Integer, Literal("32"), MachineBudget, GENERATION, CURRICULA_RULE).ui("Shape alignment", "number"),
    nested!("iris", "shape_max_ratio", Number, Literal("4"), WhatToMake, GENERATION, CURRICULA_RULE).ui("Max aspect ratio", "number"),
    nested!("iris", "rope_aspect", Text, Provider("gen_core_iris::IrisTrainPlan::resolve: model rope configuration"), WhatToMake, GENERATION, CURRICULA_RULE).ui("RoPE law", "text"),
    nested!("iris", "repa", Object("Iris repa options"), Absent, WhatToMake, GENERATION, CURRICULA_RULE),
    nested!("iris", "activation_checkpointing", Text, Inherit("training.advanced.gradientCheckpointing: full when true, none when false"), MachineBudget, GENERATION, CURRICULA_RULE).ui("Activation checkpointing", "text"),
    nested!("iris", "ac_selective_every", Integer, Literal("2"), MachineBudget, GENERATION, CURRICULA_RULE).ui("Checkpoint every Nth block", "number"),
    nested!("iris", "expected_global_batch", Integer, Absent, WhatToMake, GENERATION, CURRICULA_RULE).ui("Expected global batch", "number"),
    nested!("iris", "scale_warmup_by_world", Bool, Literal("true"), WhatToMake, GENERATION, CURRICULA_RULE).ui("Scale warmup by rank count", "checkbox"),
    nested!("iris", "dist", Object("Iris dist options"), Absent, MachineBudget, GENERATION, CURRICULA_RULE),
    nested!("iris", "resume_optimizer", Text, Literal("\"full\""), WhatToMake, GENERATION, CURRICULA_RULE).ui("Resume optimizer state", "text"),
    nested!("iris", "save_every_epochs", Integer, Literal("0"), Context, GENERATION, CURRICULA_RULE).ui("Save every N epochs", "number"),
    nested!("iris", "log_every", Integer, Literal("20"), Context, GENERATION, CURRICULA_RULE).ui("Log every (optimizer steps)", "number"),
    nested!("iris", "log_block_grad_norms", Bool, Literal("false"), Context, GENERATION, CURRICULA_RULE).ui("Log per-block gradient norms", "checkbox"),
    nested!("iris", "val_every_steps", Integer, Literal("0"), Context, GENERATION, CURRICULA_RULE).ui("Validate every (optimizer steps)", "number"),
    nested!("iris", "val_samples", Integer, Literal("8192"), Context, GENERATION, CURRICULA_RULE).ui("Holdout samples per validation", "number"),
    nested!("iris", "perf", Object("Iris perf options"), Absent, Context, GENERATION, CURRICULA_RULE),
    nested!("iris", "preview_at_step_one", Bool, Literal("false"), Context, GENERATION, CURRICULA_RULE).ui("Also preview at step 1", "checkbox"),
    nested!("iris", "preview_order", Integer, Literal("2"), Context, GENERATION, CURRICULA_RULE).ui("Preview solver order", "number"),
    nested!("iris_restore", "loss_l2", Number, Target("defaults.advanced.iris_restore.loss_l2"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Pixel (L2) loss weight", "number"),
    nested!("iris_restore", "loss_lpips", Number, Target("defaults.advanced.iris_restore.loss_lpips"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("LPIPS loss weight", "number"),
    nested!("iris_restore", "loss_gan", Number, Target("defaults.advanced.iris_restore.loss_gan"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("GAN loss weight", "number"),
    nested!("iris_restore", "gan_real_label", Number, Target("defaults.advanced.iris_restore.gan_real_label"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("GAN real label", "number"),
    nested!("iris_restore", "lr_critic", Number, Inherit("training.config.learningRate"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Critic learning rate", "number"),
    nested!("iris_restore", "betas", Numbers, Target("defaults.advanced.iris_restore.betas"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Betas", "list"),
    nested!("iris_restore", "critic_betas", Numbers, Inherit("training.advanced.iris_restore.betas"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Critic betas", "list"),
    nested!("iris_restore", "critic_weight_decay", Number, Inherit("training.advanced.weightDecay"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Critic weight decay", "number"),
    nested!("iris_restore", "gradient_clip", Number, Target("defaults.advanced.iris_restore.gradient_clip"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Gradient clip", "number"),
    nested!("iris_restore", "critic_gradient_clip", Number, Target("defaults.advanced.iris_restore.critic_gradient_clip"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Critic gradient clip", "number"),
    nested!("iris_restore", "ema_enabled", Bool, Target("defaults.advanced.iris_restore.ema_enabled"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("EMA", "checkbox"),
    nested!("iris_restore", "ema_decay", Number, Target("defaults.advanced.iris_restore.ema_decay"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("EMA decay", "number"),
    nested!("iris_restore", "crop", Text, Target("defaults.advanced.iris_restore.crop"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Crop", "text"),
    nested!("iris_restore", "degradation", Object("Iris degradation options"), Provider("gen_core_iris::degradation::DegradationConfig::default"), WhatToMake, RESTORE, IRIS_RESTORE_RULE),
    nested!("iris_restore", "critic_dtype", Text, Target("defaults.advanced.iris_restore.critic_dtype"), MachineBudget, RESTORE, IRIS_RESTORE_RULE).ui("Critic precision", "text"),
    nested!("iris_restore", "init", Text, Target("defaults.advanced.iris_restore.init"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Start from", "text"),
    nested!("iris_restore", "num_epochs", Integer, Absent, WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Epochs", "number"),
    nested!("iris_restore", "nan_loss_tolerance", Integer, Target("defaults.advanced.iris_restore.nan_loss_tolerance"), Context, RESTORE, IRIS_RESTORE_RULE).ui("NaN-loss tolerance", "number"),
    nested!("iris_restore", "keep_last_checkpoints", Integer, Target("defaults.advanced.iris_restore.keep_last_checkpoints"), Context, RESTORE, IRIS_RESTORE_RULE).ui("Keep last checkpoints", "number"),
    nested!("iris_restore", "milestone_steps", Numbers, Target("defaults.advanced.iris_restore.milestone_steps"), Context, RESTORE, IRIS_RESTORE_RULE).ui("Milestone steps", "list"),
    nested!("iris_restore", "resume_data_policy", Text, Literal("\"exact\""), Context, RESTORE, IRIS_RESTORE_RULE).ui("Resume data policy", "text"),
    nested!("iris_restore", "export_weights", Text, Provider("gen_core_iris::restoration_train::RestoreTrainPlan::resolve.export_weights"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Export weights", "text"),
    nested!("iris_restore", "export_dtype", Text, Target("defaults.advanced.iris_restore.export_dtype"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Export precision", "text"),
    nested!("iris_restore", "preview_weights", Text, Provider("gen_core_iris::restoration_train::RestoreTrainPlan::resolve.preview_weights"), Context, RESTORE, IRIS_RESTORE_RULE).ui("Preview weights", "text"),
    nested!("iris_restore", "lora_init", Text, Provider("gen_core_iris::restoration_train::RestoreTrainPlan::resolve.lora_init"), WhatToMake, RESTORE, IRIS_RESTORE_RULE).ui("Adapter init", "text"),
    // Iris nested controls. The worker owns paths and distributed process identity.
    nested!("iris.tracking", "project", Text, Absent, Context, GENERATION, "training::iris::curricula::validate_tracking").ui("Tracking project", "text"),
    nested!("iris.tracking", "tags", Strings, Absent, Context, GENERATION, "training::iris::curricula::validate_tracking").ui("Tracking tags", "list"),
    nested!("iris.repa", "weight", Number, Literal("0"), WhatToMake, GENERATION, "training::iris::curricula::validate_repa").ui("REPA weight", "number"),
    nested!("iris.repa", "variant", Text, Literal("\"repa\""), WhatToMake, GENERATION, "training::iris::curricula::validate_repa").ui("REPA variant", "text"),
    nested!("iris.repa", "spatial_norm_gamma", Number, Literal("1"), WhatToMake, GENERATION, "training::iris::curricula::validate_repa").ui("iREPA spatial-norm gamma", "number"),
    nested!("iris.repa", "proj_hidden_dim", Integer, Literal("2048"), WhatToMake, GENERATION, "training::iris::curricula::validate_repa").ui("Projector width", "number"),
    nested!("iris.repa", "teacher_image_size", Integer, Literal("224"), WhatToMake, GENERATION, "training::iris::curricula::validate_repa").ui("Teacher image size", "number"),
    nested!("iris.repa", "teacher_match_student", Bool, Literal("false"), WhatToMake, GENERATION, "training::iris::curricula::validate_repa").ui("Teacher sees the student's resolution", "checkbox"),
    nested!("iris.repa", "layer", Integer, Provider("gen_core_iris::train::repa::RepaPlan::parse: default layer"), WhatToMake, GENERATION, "training::iris::curricula::validate_repa").ui("Aligned block", "number"),
    nested!("iris.perf", "lazy_sync", Bool, Literal("false"), MachineBudget, GENERATION, "training::iris::curricula::validate_perf").ui("Lazy sync (non-finite guard on the clipped norm)", "checkbox"),
    nested!("iris.dist", "strategy", Text, Literal("\"ddp\""), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Strategy", "text"),
    nested!("iris.dist", "sharding", Text, Literal("\"full\""), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Sharding", "text"),
    nested!("iris.dist", "mesh", Object("distributed mesh"), Provider("gen_core_iris::train::dist::parse_dist"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist"),
    nested!("iris.dist", "param_dtype", Text, Literal("\"bf16\""), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Gathered parameter precision", "text"),
    nested!("iris.dist", "reduce_dtype", Text, Literal("\"fp32\""), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Reduction precision", "text"),
    nested!("iris.dist", "reshard_after_forward", Bool, Literal("true"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Reshard after forward", "checkbox"),
    nested!("iris.dist", "checkpoint_io", Text, Literal("\"auto\""), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Checkpoint writing", "text"),
    nested!("iris.dist", "world", Object("distributed world"), Provider("gen_core_iris::train::dist::parse_dist"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist"),
    nested!("iris.dist.world", "world_size", Integer, Provider("gen_core_iris::train::dist::parse_dist"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Ranks", "number"),
    nested!("iris.dist.world", "share_device", Bool, Literal("false"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Share one device between ranks", "checkbox"),
    nested!("iris.dist.world", "timeout_secs", Integer, Literal("5400"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Rank timeout (s)", "number"),
    nested!("iris.dist.world", "rendezvous_timeout_secs", Integer, Literal("600"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Rank start-up timeout (s)", "number"),
    nested!("iris.dist.world", "cancel_grace_secs", Integer, Literal("120"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Cancel grace (s)", "number"),
    nested!("iris.dist.mesh", "dp_shard", Integer, Literal("0"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Ranks per sharded copy", "number"),
    nested!("iris.dist.mesh", "dp_replicate", Integer, Literal("0"), MachineBudget, GENERATION, "training::iris::curricula::validate_dist").ui("Replicated copies", "number"),
    nested!("iris_restore.degradation", "final_sinc_prob", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "use_hflip", Bool, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "use_rot", Bool, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "use_sharpener", Bool, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "second_blur_prob", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "stage2_scale", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "resize_back", Bool, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "kernel_list", Strings, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "kernel_prob", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "sinc_prob", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "blur_sigma", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "betag_range", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "betap_range", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "resize_prob", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "resize_range", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "gaussian_noise_prob", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "noise_range", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "poisson_scale_range", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "gray_noise_prob", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "jpeg_range", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "kernel_list2", Strings, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "kernel_prob2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "sinc_prob2", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "blur_sigma2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "betag_range2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "betap_range2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "resize_prob2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "resize_range2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "gaussian_noise_prob2", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "noise_range2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "poisson_scale_range2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "gray_noise_prob2", Number, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris_restore.degradation", "jpeg_range2", Numbers, Provider("gen_core_iris::degradation::DegradationConfig::default (HYPIR sd2_train recipe)"), WhatToMake, RESTORE, DEGRADATION_RULE),
    nested!("iris", "resume_from", Text, Absent, Context, GENERATION_OR_DEPTH, IRIS_GENERATION_RULE).internal().compat("API resolves an authorized job id to this path and refuses client-supplied paths"),
    nested!("iris", "load_from", Text, Absent, Context, GENERATION_OR_DEPTH, IRIS_GENERATION_RULE).internal().compat("API resolves an authorized job id to this path and refuses client-supplied paths"),
    nested!("iris_restore", "resume_from", Text, Absent, Context, RESTORE, IRIS_RESTORE_RULE).internal().compat("API resolves an authorized job id to this path and refuses client-supplied paths"),
    nested!("fullFinetuneConfig", "mixedPrecision", Text, Target("defaults.advanced.fullFinetuneConfig.mixedPrecision"), Context, TrainingTarget("defaults.advanced.fullFinetuneConfig"), "training::apply_candle_training_defaults").internal(),
    nested!("fullFinetuneConfig", "gradientCheckpointing", Bool, Target("defaults.advanced.fullFinetuneConfig.gradientCheckpointing"), Context, TrainingTarget("defaults.advanced.fullFinetuneConfig"), "training::apply_candle_training_defaults").internal(),
];
