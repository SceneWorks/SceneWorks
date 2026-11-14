//! One-way lowering from the shared validator's effective request into native option fields.
//!
//! The authoring payload remains available to the upgrader and persistence layer. Execution
//! code receives domain structs built from this input, not the mutable `advanced` JSON bag.
//! This module does not coerce, clamp, default or decide compatibility: those rules belong to
//! `sceneworks_options::request_validation::resolve` before `BindingInput` is constructed.

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::contracts::JsonObject;
use crate::image_request::ControlWeightsFields;
use crate::video_request::ReferenceImageResizeMode;
use sceneworks_options::{by_wire, DefaultRule, OptionSpec, Shape};

/// How the value reached the effective request. Omission remains distinct from an authored
/// value that happens to equal the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Authored {
        spelling: &'static str,
    },
    Default(DefaultRule),
    /// A worker-owned probe derived from already-validated values. It is never persisted as an
    /// authored choice or substituted into a public request.
    Internal(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bound<T> {
    pub value: T,
    pub origin: Origin,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingError {
    Undeclared {
        location: String,
        key: String,
    },
    WrongShape {
        id: &'static str,
        expected: &'static str,
    },
    MissingRequired {
        id: &'static str,
    },
    InvalidEffectiveValue {
        id: &'static str,
        expected: &'static str,
    },
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undeclared { location, key } => {
                write!(formatter, "undeclared option {location}.{key}")
            }
            Self::WrongShape { id, expected } => {
                write!(formatter, "option {id} is not declared as {expected}")
            }
            Self::MissingRequired { id } => write!(formatter, "required option {id} is missing"),
            Self::InvalidEffectiveValue { id, expected } => {
                write!(formatter, "resolved option {id} must be {expected}")
            }
        }
    }
}

impl std::error::Error for BindingError {}

/// The syntactic state of an option in a saved request or a bare routing probe. Routing must
/// retain malformed intent so a different provider cannot silently claim the request. Policy,
/// coercion, and defaults are applied by the shared resolver for full submissions.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteField<T> {
    Absent,
    Null,
    Valid(T),
    Invalid,
}

/// A persisted film job can predate the resize-mode claim stamp. Keep absence distinct from a
/// stamped value when the film controller decides what a later attempt should reproduce.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedVideoResizeModeFact {
    pub mode: RouteField<ReferenceImageResizeMode>,
}

impl PersistedVideoResizeModeFact {
    pub fn from_payload_syntax(payload: &JsonObject) -> Self {
        let mode = match payload.get("advanced") {
            None => RouteField::Absent,
            Some(Value::Null) => RouteField::Null,
            Some(Value::Object(advanced)) => match advanced.get("referenceImageResizeMode") {
                None => RouteField::Absent,
                Some(Value::Null) => RouteField::Null,
                Some(Value::String(value)) => ReferenceImageResizeMode::parse(value)
                    .map(RouteField::Valid)
                    .unwrap_or(RouteField::Invalid),
                Some(_) => RouteField::Invalid,
            },
            Some(_) => RouteField::Invalid,
        };
        Self { mode }
    }
}

impl<T> RouteField<T> {
    pub fn is_absent_or_null(&self) -> bool {
        matches!(self, Self::Absent | Self::Null)
    }

    pub fn is_present_nonnull(&self) -> bool {
        matches!(self, Self::Valid(_) | Self::Invalid)
    }
}

/// The two authored MOSS controls that appearance folding must inspect before it rewrites the
/// prompt. This is a one-time authoring projection; the rewritten request later enters portable
/// validation and the resolved execution binder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AppearanceAuthoredInputs {
    pub has_embedding_or_per_prompt_negative: bool,
    pub moss_accelerated: bool,
    /// Presence includes explicit null, which blocks a preset from replacing authored intent.
    pub moss_image_present: bool,
}

/// A PiD source override can name either component independently. Keep each carrier's syntax
/// so admission can refuse malformed source claims without re-reading authored JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutePidCheckpoint {
    pub repo: RouteField<String>,
    pub filename: RouteField<String>,
}

impl AppearanceAuthoredInputs {
    pub fn from_payload_syntax(payload: &JsonObject) -> Self {
        let advanced = payload.get("advanced").and_then(Value::as_object);
        let moss_advanced = advanced
            .and_then(|advanced| advanced.get("mossImageAdvanced"))
            .and_then(Value::as_object);
        let has_embedding_or_per_prompt_negative = moss_advanced.is_some_and(|inputs| {
            ["positive", "negative"]
                .iter()
                .any(|key| inputs.get(*key).is_some_and(|value| !value.is_null()))
                || inputs
                    .get("negativePrompts")
                    .and_then(Value::as_array)
                    .is_some_and(|values| !values.is_empty())
        });
        let moss_accelerated = advanced
            .and_then(|advanced| advanced.get("mossImage"))
            .and_then(|value| value.get("acceleration"))
            .and_then(Value::as_str)
            .is_some_and(|mode| mode != "off");
        let moss_image_present =
            advanced.is_some_and(|advanced| advanced.contains_key("mossImage"));
        Self {
            has_embedding_or_per_prompt_negative,
            moss_accelerated,
            moss_image_present,
        }
    }
}

/// A one-time typed projection for routing probes, including old or malformed saved payloads.
/// Structured carriers expose only the shape routing needs; execution binds their full domain
/// structs after the shared resolver has returned an effective request.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteOptionFacts {
    pub poses: RouteField<usize>,
    /// Pose routing also checks each entry's container shape without parsing its keypoints.
    pub poses_all_objects: RouteField<usize>,
    pub poses_any_object: RouteField<bool>,
    pub phases: RouteField<usize>,
    pub control_mode: RouteField<String>,
    pub control_image: RouteField<String>,
    pub control_weights: RouteField<ControlWeightsFields>,
    pub control_scale: RouteField<f64>,
    pub true_cfg_scale: RouteField<f64>,
    pub image_guidance_scale: RouteField<f64>,
    pub ip_adapter_scale: RouteField<f64>,
    pub reference_strength: RouteField<f64>,
    pub mlx_quantize: RouteField<i64>,
    pub quant_tier: RouteField<String>,
    pub quant: RouteField<String>,
    pub quantization: RouteField<String>,
    pub decoder: RouteField<String>,
    pub enhance_prompt: RouteField<bool>,
    pub use_pid: RouteField<bool>,
    pub pid_checkpoint: RouteField<RoutePidCheckpoint>,
    pub pid_gemma: RouteField<String>,
    pub veda_enabled: RouteField<bool>,
    pub use_uncensored_enhancer: RouteField<bool>,
    pub text_encoder_model: RouteField<String>,
    pub model_path: RouteField<String>,
    /// Legacy source hints used only to match an already-loaded model at claim time.
    pub model_repo: RouteField<String>,
    pub repo: RouteField<String>,
    pub tile_control_net_repo: RouteField<String>,
    pub strength: RouteField<f64>,
    pub hires_fix_enabled: RouteField<bool>,
    pub yue2_decoder: RouteField<String>,
    pub conv_rot: RouteField<bool>,
    pub angle_set: RouteField<bool>,
}

impl RouteOptionFacts {
    /// Parse only JSON syntax. A malformed `advanced` object marks every advanced carrier invalid
    /// because treating it as empty could make an unsupported route appear eligible.
    pub fn from_payload_syntax(payload: &JsonObject) -> Self {
        fn parent<'a>(root: &'a JsonObject, key: &str) -> Result<Option<&'a JsonObject>, ()> {
            match root.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Object(map)) => Ok(Some(map)),
                Some(_) => Err(()),
            }
        }
        fn field<T>(
            map: Result<Option<&JsonObject>, ()>,
            key: &str,
            decode: impl FnOnce(&Value) -> Option<T>,
        ) -> RouteField<T> {
            match map {
                Err(()) => RouteField::Invalid,
                Ok(None) => RouteField::Absent,
                Ok(Some(map)) => match map.get(key) {
                    None => RouteField::Absent,
                    Some(Value::Null) => RouteField::Null,
                    Some(value) => decode(value)
                        .map(RouteField::Valid)
                        .unwrap_or(RouteField::Invalid),
                },
            }
        }
        fn text(value: &Value) -> Option<String> {
            value.as_str().map(str::to_owned)
        }
        fn boolean(value: &Value) -> Option<bool> {
            value.as_bool()
        }
        fn count(value: &Value) -> Option<usize> {
            value.as_array().map(Vec::len)
        }
        fn legacy_integer(value: &Value) -> Option<i64> {
            sceneworks_options::legacy_integer(value)?.as_i64()
        }
        fn legacy_number(value: &Value) -> Option<f64> {
            sceneworks_options::legacy_number(value)
        }
        let advanced = parent(payload, "advanced");
        let veda = match advanced {
            Err(()) => Err(()),
            Ok(None) => Ok(None),
            Ok(Some(map)) => parent(map, "veda"),
        };
        let hires_fix = parent(payload, "hiresFix");
        let yue2 = parent(payload, "yue2");
        Self {
            poses: field(advanced, "poses", count),
            poses_all_objects: field(advanced, "poses", |value| {
                value
                    .as_array()
                    .and_then(|items| items.iter().all(Value::is_object).then_some(items.len()))
            }),
            poses_any_object: field(advanced, "poses", |value| {
                value
                    .as_array()
                    .map(|items| items.iter().any(Value::is_object))
            }),
            phases: field(advanced, "phases", count),
            control_mode: field(advanced, "controlMode", text),
            control_image: field(advanced, "controlImage", text),
            control_weights: field(advanced, "controlWeights", |v| {
                serde_json::from_value(v.clone()).ok()
            }),
            control_scale: field(advanced, "controlScale", legacy_number),
            true_cfg_scale: field(advanced, "trueCfgScale", legacy_number),
            image_guidance_scale: field(advanced, "imageGuidanceScale", legacy_number),
            ip_adapter_scale: field(advanced, "ipAdapterScale", legacy_number),
            reference_strength: field(advanced, "referenceStrength", legacy_number),
            mlx_quantize: field(advanced, "mlxQuantize", legacy_integer),
            quant_tier: field(advanced, "quantTier", text),
            quant: field(advanced, "quant", text),
            quantization: field(advanced, "quantization", text),
            decoder: field(advanced, "decoder", text),
            enhance_prompt: field(advanced, "enhancePrompt", boolean),
            use_pid: field(advanced, "usePid", |value| {
                Some(crate::portable_sources::pid_requested(Some(value)))
            }),
            pid_checkpoint: field(advanced, "pidCheckpoint", |value| {
                value.as_object().map(|checkpoint| RoutePidCheckpoint {
                    repo: field(Ok(Some(checkpoint)), "repo", text),
                    filename: field(Ok(Some(checkpoint)), "filename", text),
                })
            }),
            pid_gemma: field(advanced, "pidGemma", text),
            veda_enabled: field(veda, "enabled", boolean),
            use_uncensored_enhancer: field(advanced, "useUncensoredEnhancer", boolean),
            text_encoder_model: field(advanced, "textEncoderModel", text),
            model_path: field(advanced, "modelPath", text),
            model_repo: field(advanced, "modelRepo", text),
            repo: field(advanced, "repo", text),
            tile_control_net_repo: field(advanced, "tileControlNetRepo", text),
            strength: field(advanced, "strength", legacy_number),
            hires_fix_enabled: field(hires_fix, "enabled", boolean),
            yue2_decoder: field(yue2, "decoder", text),
            conv_rot: field(advanced, "convRot", boolean),
            angle_set: field(advanced, "angleSet", |value| {
                Some(sceneworks_options::legacy_image_flag(value))
            }),
        }
    }

    /// Build the portable companion-source selector from an already-resolved request. A caller
    /// supplies facts projected from the owner-issued effective payload, never authored input.
    pub fn auxiliary_selection(&self) -> crate::portable_sources::AuxOptionSelection {
        use crate::portable_sources::AuxOptionSelection;
        fn truth(field: &RouteField<bool>) -> bool {
            matches!(field, RouteField::Valid(true))
        }
        fn text(field: &RouteField<String>) -> Option<String> {
            match field {
                RouteField::Valid(value) => Some(value.clone()),
                _ => None,
            }
        }
        AuxOptionSelection {
            poses_present: matches!(self.poses, RouteField::Valid(count) if count > 0),
            poses_contain_object: truth(&self.poses_any_object),
            use_pid: truth(&self.use_pid),
            decoder: text(&self.decoder),
            enhance_prompt: truth(&self.enhance_prompt),
            control_mode: text(&self.control_mode),
            control_image: text(&self.control_image),
            veda_enabled: truth(&self.veda_enabled),
            use_uncensored_enhancer: truth(&self.use_uncensored_enhancer),
            text_encoder_model: text(&self.text_encoder_model),
            training: None,
        }
    }

    pub fn artifact_selection(
        &self,
        authored: &Self,
        model_id: Option<&str>,
    ) -> Result<crate::model_artifacts::artifact_selection::PortableArtifactOptions, String> {
        let yue2_decoder = if model_id == Some("yue2") {
            // An omitted decoder lets the installed co-requisite win. A materialized literal
            // "standard" default is not an authored choice and must not override that install.
            match &authored.yue2_decoder {
                RouteField::Absent | RouteField::Null => None,
                RouteField::Valid(value) => Some(value.clone()),
                RouteField::Invalid => return Err("YuE2 decoder choice is not a string".into()),
            }
        } else {
            None
        };
        Ok(
            crate::model_artifacts::artifact_selection::PortableArtifactOptions {
                conv_rot: matches!(self.conv_rot, RouteField::Valid(true)),
                yue2_decoder,
            },
        )
    }
}

/// A bare training route probe cannot run submission resolution because it may omit the target
/// catalog facts. Preserve malformed parent objects so unsupported training intent fails closed.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRouteFacts {
    pub network_type: RouteField<String>,
    pub resume: RouteField<bool>,
    pub base_model_path: RouteField<String>,
    /// Authored REPA weight affects the frozen source identity. Keep malformed archived plans
    /// distinct from omission; the source projector uses only a valid JSON number.
    pub repa_weight: RouteField<f64>,
}

impl TrainingRouteFacts {
    pub fn from_plan_syntax(plan: &JsonObject) -> Self {
        let advanced = plan
            .get("config")
            .ok_or(())
            .and_then(|value| value.as_object().ok_or(()))
            .and_then(|config| config.get("advanced").ok_or(()))
            .and_then(|value| value.as_object().ok_or(()));
        let missing_advanced = plan
            .get("config")
            .and_then(Value::as_object)
            .is_some_and(|config| !config.contains_key("advanced"));
        let network_type = match advanced {
            Ok(advanced) => match advanced.get("networkType") {
                None => RouteField::Absent,
                Some(Value::Null) => RouteField::Null,
                Some(Value::String(value)) => RouteField::Valid(value.clone()),
                Some(_) => RouteField::Invalid,
            },
            // An omitted advanced object is ordinary absence; malformed or missing config is
            // invalid because no training route can prove that the plan is well formed.
            Err(()) if missing_advanced => RouteField::Absent,
            Err(()) => RouteField::Invalid,
        };
        let resume = match advanced {
            Ok(advanced) => match advanced.get("resume") {
                None => RouteField::Absent,
                Some(Value::Null) => RouteField::Null,
                Some(Value::Bool(value)) => RouteField::Valid(*value),
                Some(_) => RouteField::Invalid,
            },
            Err(()) if missing_advanced => RouteField::Absent,
            Err(()) => RouteField::Invalid,
        };
        let repa_weight = match advanced {
            Ok(advanced) => match advanced.get("iris") {
                None => RouteField::Absent,
                Some(Value::Null) => RouteField::Null,
                Some(Value::Object(iris)) => match iris.get("repa") {
                    None => RouteField::Absent,
                    Some(Value::Null) => RouteField::Null,
                    Some(Value::Object(repa)) => match repa.get("weight") {
                        None => RouteField::Absent,
                        Some(Value::Null) => RouteField::Null,
                        Some(value) => value
                            .as_f64()
                            .map(RouteField::Valid)
                            .unwrap_or(RouteField::Invalid),
                    },
                    Some(_) => RouteField::Invalid,
                },
                Some(_) => RouteField::Invalid,
            },
            Err(()) if missing_advanced => RouteField::Absent,
            Err(()) => RouteField::Invalid,
        };
        let base_model_path = match plan.get("target") {
            None => RouteField::Absent,
            Some(Value::Null) => RouteField::Null,
            Some(Value::Object(target)) => match target.get("baseModelPath") {
                None => RouteField::Absent,
                Some(Value::Null) => RouteField::Null,
                Some(Value::String(path)) => RouteField::Valid(path.clone()),
                Some(_) => RouteField::Invalid,
            },
            Some(_) => RouteField::Invalid,
        };
        Self {
            network_type,
            resume,
            base_model_path,
            repa_weight,
        }
    }
}

/// Training controls that API admission, portable routing and worker orchestration inspect.
/// The shared training resolver owns their defaults and refusals; this only lowers its effective
/// payload once into typed fields. The engine's complete validated model options travel through
/// its existing wire contract separately.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IrisTrainingOptions {
    pub network_type: Option<Bound<String>>,
    pub mixed_precision: Option<Bound<String>>,
    pub control_type: Option<Bound<String>>,
    pub person_gate: Option<Bound<bool>>,
    pub resume_from_job_id: Option<Bound<String>>,
    pub load_from_job_id: Option<Bound<String>>,
    pub output_scope: Option<Bound<String>>,
    pub requested_gpu: Option<Bound<String>>,
    pub depth_init: Option<Bound<String>>,
    pub stage: Option<Bound<String>>,
    pub parent_stage: Option<Bound<String>>,
    pub resume_data_policy: Option<Bound<String>>,
    pub resume_optimizer: Option<Bound<String>>,
    pub dist_strategy: Option<Bound<String>>,
    pub dist_world_size: Option<Bound<i64>>,
    pub dist_share_device: Option<Bound<bool>>,
    pub dist_timeout_secs: Option<Bound<i64>>,
    pub expected_global_batch: Option<Bound<i64>>,
    /// Portable Iris policy has applied topology defaults once at the binding boundary.
    pub dist_topology: IrisDistTopology,
    pub repa_weight: Option<Bound<f64>>,
    /// Portable Iris policy decides whether the validated REPA settings activate the loss.
    pub repa_active: bool,
    pub activation_checkpointing: Option<Bound<String>>,
    /// The portable memory-floor policy treats full and selective-op as full recomputation.
    pub full_activation_checkpointing: bool,
    pub restore_init: Option<Bound<String>>,
    pub restore_resume_data_policy: Option<Bound<String>>,
    /// Validated opaque engine ABI; callers may pass it to the engine parser, not read its keys.
    pub restore_degradation_wire: Option<Bound<JsonObject>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrisDistTopology {
    pub world_size: u64,
    pub share_device: bool,
    pub timeout_secs: u64,
}

impl Default for IrisDistTopology {
    fn default() -> Self {
        let (world_size, share_device, timeout_secs) =
            crate::training::iris::curricula::iris_dist_world(&JsonObject::new());
        Self {
            world_size,
            share_device,
            timeout_secs,
        }
    }
}

impl IrisTrainingOptions {
    pub fn bind(input: &BindingInput<'_>) -> Result<Self, BindingError> {
        let dist_topology = BindingInput::parent(input.effective, "training.config.advanced.iris")
            .map(crate::training::iris::curricula::iris_dist_world)
            .map(
                |(world_size, share_device, timeout_secs)| IrisDistTopology {
                    world_size,
                    share_device,
                    timeout_secs,
                },
            )
            .unwrap_or_default();
        let repa_active = BindingInput::parent(input.effective, "training.config.advanced.iris")
            .is_some_and(crate::training::iris::curricula::iris_repa_enabled);
        let activation_checkpointing =
            input.text("training.config.advanced.iris", "activation_checkpointing")?;
        let full_activation_checkpointing =
            crate::training::iris::iris_full_activation_checkpointing_mode(
                activation_checkpointing
                    .as_ref()
                    .map(|bound| bound.value.as_str()),
            );
        Ok(Self {
            network_type: input.text("training.config.advanced", "networkType")?,
            mixed_precision: input.text("training.config.advanced", "mixedPrecision")?,
            control_type: input.text("training.config.advanced", "controlType")?,
            person_gate: input.boolean("training.config.advanced", "personGate")?,
            resume_from_job_id: input.text("training.config.advanced", "resumeFromJobId")?,
            load_from_job_id: input.text("training.config.advanced", "loadFromJobId")?,
            output_scope: input.text("training.config.advanced", "outputScope")?,
            requested_gpu: input.text("training.config.advanced", "requestedGpu")?,
            depth_init: input.text("training.config.advanced", "depthInit")?,
            stage: input.text("training.config.advanced.iris", "stage")?,
            parent_stage: input.text("training.config.advanced.iris", "parent_stage")?,
            resume_data_policy: input
                .text("training.config.advanced.iris", "resume_data_policy")?,
            resume_optimizer: input.text("training.config.advanced.iris", "resume_optimizer")?,
            dist_strategy: input.text("training.config.advanced.iris.dist", "strategy")?,
            dist_world_size: input
                .integer("training.config.advanced.iris.dist.world", "world_size")?,
            dist_share_device: input
                .boolean("training.config.advanced.iris.dist.world", "share_device")?,
            dist_timeout_secs: input
                .integer("training.config.advanced.iris.dist.world", "timeout_secs")?,
            expected_global_batch: input
                .integer("training.config.advanced.iris", "expected_global_batch")?,
            dist_topology,
            repa_weight: input.number("training.config.advanced.iris.repa", "weight")?,
            repa_active,
            activation_checkpointing,
            full_activation_checkpointing,
            restore_init: input.text("training.config.advanced.iris_restore", "init")?,
            restore_resume_data_policy: input.text(
                "training.config.advanced.iris_restore",
                "resume_data_policy",
            )?,
            restore_degradation_wire: input
                .structured("training.config.advanced.iris_restore", "degradation")?,
        })
    }

    pub fn repa_enabled(&self) -> bool {
        self.repa_active
    }

    pub fn auxiliary_selection(
        &self,
        plan: &crate::training::TrainingPlan,
        entry_id: &str,
    ) -> crate::portable_sources::TrainingAuxSelection {
        let matches_entry = plan.target.base_model == entry_id;
        let iris_kernel =
            matches_entry && crate::training::iris::is_iris_training_kernel(&plan.target.kernel);
        crate::portable_sources::TrainingAuxSelection {
            base_model: plan.target.base_model.clone(),
            depth_init_parent: matches_entry
                && entry_id == "iris_3b_depth"
                && self.depth_init.as_ref().map(|value| value.value.as_str()) == Some("parent"),
            restore_init_parent: matches_entry
                && entry_id == "iris_3b_restore"
                && self.restore_init.as_ref().map(|value| value.value.as_str()) == Some("parent"),
            iris_repa_enabled: iris_kernel && self.repa_enabled(),
            iris_kernel,
        }
    }
}

/// The complete, validated advanced object handed to the native trainer ABI. Its JSON shape is
/// an engine contract; execution policy reads the named fields on `TrainingExecutionOptions`.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedModelOptionsWire(JsonObject);

impl ValidatedModelOptionsWire {
    pub fn into_json(self) -> JsonObject {
        self.0
    }

    pub fn as_json_for_engine(&self) -> &JsonObject {
        &self.0
    }

    /// Iris task trainers accept exactly their objective's nested options bag. Preserve the
    /// historical empty-object fallback for missing or malformed saved bags.
    pub fn for_iris_task(&self, objective: crate::training::iris::IrisObjective) -> JsonObject {
        let key = objective.options_key();
        let mut selected = JsonObject::new();
        selected.insert(
            key.to_owned(),
            self.0
                .get(key)
                .filter(|value| value.is_object())
                .cloned()
                .unwrap_or_else(|| Value::Object(JsonObject::new())),
        );
        selected
    }
}

/// One-time native training binding from the portable owner's effective request. The shared
/// owner resolves common defaults; its existing technique parsers supply the named native
/// settings, so no worker repeats their policy or reads the authored advanced bag.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingExecutionOptions {
    pub common: crate::training::ResolvedTrainingAdvanced,
    pub iris: IrisTrainingOptions,
    pub ltx_workflow: Option<Bound<String>>,
    pub subject_mask_loss_weights: Option<(f64, f64)>,
    pub depth_anchoring: Option<crate::training::depth_anchoring::DepthAnchoringSettings>,
    pub body_losses: crate::training::body_losses::BodyLossSettings,
    pub vae_anchor: Option<crate::training::latent_perceptual::LatentLossSettings>,
    pub latent_lpips: Option<crate::training::latent_perceptual::LatentLossSettings>,
    pub identity_loss: Option<crate::training::face_losses::IdentityLossSettings>,
    pub face_landmark_loss: Option<crate::training::face_losses::FaceLossSchedule>,
    pub training_adapter: Option<&'static crate::training::depth_anchoring::AuxTrainingModel>,
    pub buckets: Option<crate::training::ResolutionBucketConfig>,
    pub model_options_wire: ValidatedModelOptionsWire,
}

impl TrainingExecutionOptions {
    pub fn bind(
        authored: &JsonObject,
        resolved: &crate::request_validation::ResolvedRequest,
    ) -> Result<Self, String> {
        let effective = resolved.as_payload();
        let carrier = effective
            .get("plan")
            .and_then(Value::as_object)
            .unwrap_or(effective);
        let advanced = carrier
            .get("config")
            .and_then(Value::as_object)
            .and_then(|config| config.get("advanced"))
            .and_then(Value::as_object)
            .ok_or_else(|| "resolved training request has no config.advanced object".to_owned())?;
        let target = carrier.get("target").and_then(Value::as_object);
        let kernel = target
            .and_then(|target| target.get("kernel"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let base_model = target
            .and_then(|target| target.get("baseModel"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let common = crate::training::resolve_advanced_for_model(advanced, kernel, base_model)
            .map_err(|error| error.to_string())?;
        Self::from_effective(authored, effective, advanced, kernel, common)
    }

    /// Preview and Iris helpers sometimes have a validated bare config rather than a plan.
    /// Resolve its common fields through the same portable owner before binding techniques.
    pub fn for_config(
        config: &crate::training::TrainingConfig,
        kernel: &str,
        base_model: &str,
    ) -> Result<Self, String> {
        let common =
            crate::training::resolve_advanced_for_model(&config.advanced, kernel, base_model)
                .map_err(|error| error.to_string())?;
        let mut effective_advanced = config.advanced.clone();
        let lowered = serde_json::to_value(&common).map_err(|error| error.to_string())?;
        effective_advanced.extend(
            lowered
                .as_object()
                .ok_or_else(|| "resolved training controls are not an object".to_owned())?
                .clone(),
        );
        let mut effective = JsonObject::new();
        effective.insert(
            "config".to_owned(),
            serde_json::json!({ "advanced": effective_advanced }),
        );
        let mut authored = JsonObject::new();
        authored.insert(
            "config".to_owned(),
            serde_json::json!({ "advanced": config.advanced }),
        );
        let advanced = effective["config"]["advanced"]
            .as_object()
            .expect("constructed training advanced object");
        Self::from_effective(&authored, &effective, advanced, kernel, common)
    }

    fn from_effective(
        authored: &JsonObject,
        effective: &JsonObject,
        advanced: &JsonObject,
        kernel: &str,
        common: crate::training::ResolvedTrainingAdvanced,
    ) -> Result<Self, String> {
        let iris = IrisTrainingOptions::bind(&BindingInput::new(authored, effective))
            .map_err(|error| error.to_string())?;
        let technique_error = |error: crate::training::TrainingPlanError| error.to_string();
        Ok(Self {
            common,
            iris,
            ltx_workflow: BindingInput::new(authored, effective)
                .text("training.config.advanced", "ltxWorkflow")
                .map_err(|error| error.to_string())?,
            subject_mask_loss_weights: crate::training::subject_mask_loss_weights(advanced)
                .map_err(technique_error)?,
            depth_anchoring: crate::training::depth_anchoring::depth_anchoring_settings(advanced)
                .map_err(technique_error)?,
            body_losses: crate::training::body_losses::body_loss_settings(advanced)
                .map_err(technique_error)?,
            vae_anchor: crate::training::latent_perceptual::latent_loss_settings(
                &crate::training::latent_perceptual::VAE_ANCHOR,
                advanced,
            )
            .map_err(technique_error)?,
            latent_lpips: crate::training::latent_perceptual::latent_loss_settings(
                &crate::training::latent_perceptual::LATENT_LPIPS,
                advanced,
            )
            .map_err(technique_error)?,
            identity_loss: crate::training::face_losses::identity_loss_settings(advanced)
                .map_err(technique_error)?,
            face_landmark_loss: crate::training::face_losses::face_landmark_loss_settings(advanced)
                .map_err(technique_error)?,
            training_adapter: crate::training::training_adapter::training_adapter_model(advanced)
                .map_err(technique_error)?,
            buckets: crate::training::parse_training_bucket_config(advanced)?,
            model_options_wire: ValidatedModelOptionsWire(
                crate::training::training_model_options_wire(
                    BindingInput::parent(authored, "training.config.advanced").ok_or_else(
                        || "authored training request has no config.advanced object".to_owned(),
                    )?,
                    kernel,
                ),
            ),
        })
    }
}

/// Ephemeral input used only at the boundary where the shared validator has already returned
/// its effective clone. Domain-specific lowering consumes this and retains typed `Bound<T>`
/// fields. Neither raw map is stored in an execution request.
#[derive(Debug, Clone, Copy)]
pub struct BindingInput<'a> {
    authored: &'a JsonObject,
    effective: &'a JsonObject,
}

impl<'a> BindingInput<'a> {
    pub fn new(authored: &'a JsonObject, effective: &'a JsonObject) -> Self {
        Self {
            authored,
            effective,
        }
    }

    fn parent(root: &'a JsonObject, location: &str) -> Option<&'a Map<String, Value>> {
        // Registry training locations are logical paths. Submission payloads store `config`
        // at the root or under `plan`; neither serializes a literal `training` wrapper.
        let (mut map, path) = if let Some(path) = location.strip_prefix("training.") {
            (
                root.get("plan").and_then(Value::as_object).unwrap_or(root),
                path,
            )
        } else {
            (root, location)
        };
        for part in path.split('.') {
            map = map.get(part)?.as_object()?;
        }
        Some(map)
    }

    fn entry(
        &self,
        location: &str,
        key: &str,
        expected: &'static str,
        accepts: impl FnOnce(Shape) -> bool,
    ) -> Result<Option<(&'static OptionSpec, &'a Value, Origin)>, BindingError> {
        let spec = by_wire(location, key).ok_or_else(|| BindingError::Undeclared {
            location: location.to_owned(),
            key: key.to_owned(),
        })?;
        if !accepts(spec.shape) {
            return Err(BindingError::WrongShape {
                id: spec.id,
                expected,
            });
        }
        let value = Self::parent(self.effective, location).and_then(|map| {
            // The shared resolver writes canonical keys. Reading an old alias here still
            // preserves the original contract if a compatibility route carried it through.
            map.get(spec.key)
                .or_else(|| spec.aliases.iter().find_map(|alias| map.get(*alias)))
        });
        let Some(value) = value else {
            return if spec.default == DefaultRule::Required {
                Err(BindingError::MissingRequired { id: spec.id })
            } else {
                Ok(None)
            };
        };
        let origin = Self::parent(self.authored, location)
            .and_then(|map| {
                std::iter::once(spec.key)
                    .chain(spec.aliases.iter().copied())
                    .find(|spelling| map.contains_key(*spelling))
            })
            .map(|spelling| Origin::Authored { spelling })
            .unwrap_or(Origin::Default(spec.default));
        Ok(Some((spec, value, origin)))
    }

    pub fn boolean(&self, location: &str, key: &str) -> Result<Option<Bound<bool>>, BindingError> {
        self.entry(location, key, "boolean", |shape| shape == Shape::Bool)?
            .map(|(spec, value, origin)| {
                value.as_bool().map(|value| Bound { value, origin }).ok_or(
                    BindingError::InvalidEffectiveValue {
                        id: spec.id,
                        expected: "boolean",
                    },
                )
            })
            .transpose()
    }

    pub fn integer(&self, location: &str, key: &str) -> Result<Option<Bound<i64>>, BindingError> {
        self.entry(location, key, "integer", |shape| shape == Shape::Integer)?
            .map(|(spec, value, origin)| {
                value.as_i64().map(|value| Bound { value, origin }).ok_or(
                    BindingError::InvalidEffectiveValue {
                        id: spec.id,
                        expected: "integer",
                    },
                )
            })
            .transpose()
    }

    /// Preserve a meaningful explicit JSON null separately from omission. A caller can then
    /// refuse or suppress a route fallback without mistaking null for an absent option.
    pub fn nullable_integer(
        &self,
        location: &str,
        key: &str,
    ) -> Result<Option<Bound<Option<i64>>>, BindingError> {
        self.entry(location, key, "integer or null", |shape| {
            shape == Shape::Integer
        })?
        .map(|(spec, value, origin)| {
            if value.is_null() {
                return Ok(Bound {
                    value: None,
                    origin,
                });
            }
            value
                .as_i64()
                .map(|value| Bound {
                    value: Some(value),
                    origin,
                })
                .ok_or(BindingError::InvalidEffectiveValue {
                    id: spec.id,
                    expected: "integer or null",
                })
        })
        .transpose()
    }

    pub fn number(&self, location: &str, key: &str) -> Result<Option<Bound<f64>>, BindingError> {
        self.entry(location, key, "finite number", |shape| {
            shape == Shape::Number
        })?
        .map(|(spec, value, origin)| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| Bound { value, origin })
                .ok_or(BindingError::InvalidEffectiveValue {
                    id: spec.id,
                    expected: "finite number",
                })
        })
        .transpose()
    }

    /// Bind an alias that the portable owner deliberately retains as a separate provider input.
    /// SenseNova interleave uses `timestepShift` itself, even when `schedulerShift` is also present.
    /// The ordinary `number` accessor follows the canonical key and would lose that distinction.
    pub fn number_exact_alias(
        &self,
        location: &str,
        alias: &'static str,
    ) -> Result<Option<Bound<f64>>, BindingError> {
        let spec = by_wire(location, alias)
            .filter(|spec| spec.shape == Shape::Number && spec.aliases.contains(&alias))
            .ok_or_else(|| BindingError::Undeclared {
                location: location.to_owned(),
                key: alias.to_owned(),
            })?;
        let Some(value) = Self::parent(self.effective, location).and_then(|map| map.get(alias))
        else {
            return Ok(None);
        };
        let value = value.as_f64().filter(|value| value.is_finite()).ok_or(
            BindingError::InvalidEffectiveValue {
                id: spec.id,
                expected: "finite number",
            },
        )?;
        let origin =
            if Self::parent(self.authored, location).is_some_and(|map| map.contains_key(alias)) {
                Origin::Authored { spelling: alias }
            } else {
                Origin::Default(spec.default)
            };
        Ok(Some(Bound { value, origin }))
    }

    pub fn text(&self, location: &str, key: &str) -> Result<Option<Bound<String>>, BindingError> {
        self.entry(location, key, "text", |shape| {
            matches!(shape, Shape::Text | Shape::OneOf(_))
        })?
        .map(|(spec, value, origin)| {
            value
                .as_str()
                .map(|value| Bound {
                    value: value.to_owned(),
                    origin,
                })
                .ok_or(BindingError::InvalidEffectiveValue {
                    id: spec.id,
                    expected: "text",
                })
        })
        .transpose()
    }

    /// Decode only a registered structured field into its domain type. This is a one-time
    /// boundary lowering, not an execution-time typed→JSON→typed roundtrip.
    pub fn structured<T: DeserializeOwned>(
        &self,
        location: &str,
        key: &str,
    ) -> Result<Option<Bound<T>>, BindingError> {
        self.entry(location, key, "structured option", |shape| {
            matches!(
                shape,
                Shape::Object(_) | Shape::List(_) | Shape::Strings | Shape::Numbers
            )
        })?
        .map(|(spec, value, origin)| {
            serde_json::from_value(value.clone())
                .map(|value| Bound { value, origin })
                .map_err(|_| BindingError::InvalidEffectiveValue {
                    id: spec.id,
                    expected: "structured option",
                })
        })
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(value: Value) -> JsonObject {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn authored_default_and_alias_remain_distinct() {
        let authored = obj(json!({"advanced":{"steps":"8"}}));
        let effective = obj(json!({"advanced":{"steps":8,"guidanceScale":5.0}}));
        let input = BindingInput::new(&authored, &effective);
        assert_eq!(
            input.integer("advanced", "steps").unwrap(),
            Some(Bound {
                value: 8,
                origin: Origin::Authored { spelling: "steps" }
            })
        );
        assert_eq!(
            input
                .number("advanced", "guidanceScale")
                .unwrap()
                .unwrap()
                .value,
            5.0
        );
        assert!(matches!(
            input
                .number("advanced", "guidanceScale")
                .unwrap()
                .unwrap()
                .origin,
            Origin::Default(_)
        ));
    }

    #[test]
    fn no_coercion_or_fallback_is_invented_after_resolution() {
        let authored = obj(json!({"advanced":{"steps":"broken"}}));
        let effective = obj(json!({"advanced":{"steps":"broken"}}));
        let input = BindingInput::new(&authored, &effective);
        assert!(matches!(
            input.integer("advanced", "steps"),
            Err(BindingError::InvalidEffectiveValue { .. })
        ));
        assert!(matches!(
            input.integer("advanced", "__unknown"),
            Err(BindingError::Undeclared { .. })
        ));
    }

    #[test]
    fn explicit_null_suppresses_an_absent_value_fallback() {
        let authored = obj(json!({"advanced":{"mlxQuantize":null}}));
        let effective = authored.clone();
        let input = BindingInput::new(&authored, &effective);
        assert_eq!(
            input.nullable_integer("advanced", "mlxQuantize").unwrap(),
            Some(Bound {
                value: None,
                origin: Origin::Authored {
                    spelling: "mlxQuantize"
                }
            })
        );
        let omitted = obj(json!({"advanced":{}}));
        assert_eq!(
            BindingInput::new(&omitted, &omitted)
                .nullable_integer("advanced", "mlxQuantize")
                .unwrap(),
            None
        );
    }

    #[test]
    fn sense_nova_interleave_alias_keeps_its_own_value() {
        let authored = obj(json!({"advanced":{"schedulerShift":1.25,"timestepShift":2.5}}));
        let input = BindingInput::new(&authored, &authored);
        assert_eq!(
            input
                .number("advanced", "schedulerShift")
                .unwrap()
                .unwrap()
                .value,
            1.25
        );
        assert_eq!(
            input
                .number_exact_alias("advanced", "timestepShift")
                .unwrap()
                .unwrap()
                .value,
            2.5
        );
        let canonical_only = obj(json!({"advanced":{"schedulerShift":1.25}}));
        assert!(BindingInput::new(&canonical_only, &canonical_only)
            .number_exact_alias("advanced", "timestepShift")
            .unwrap()
            .is_none());
    }

    #[test]
    fn appearance_authoring_projection_preserves_legacy_missing_and_null_behavior() {
        let empty = AppearanceAuthoredInputs::from_payload_syntax(&obj(json!({})));
        assert_eq!(empty, AppearanceAuthoredInputs::default());
        let nulls = AppearanceAuthoredInputs::from_payload_syntax(&obj(json!({
            "advanced":{"mossImageAdvanced":{"positive":null,"negative":null,"negativePrompts":[]},
                "mossImage":{"acceleration":"off"}}
        })));
        assert!(!nulls.has_embedding_or_per_prompt_negative);
        assert!(!nulls.moss_accelerated);
        assert!(nulls.moss_image_present);
        let supplied = AppearanceAuthoredInputs::from_payload_syntax(&obj(json!({
            "advanced":{"mossImageAdvanced":{"negativePrompts":["bad"]},
                "mossImage":{"acceleration":"unmerged_bf16"}}
        })));
        assert!(supplied.has_embedding_or_per_prompt_negative);
        assert!(supplied.moss_accelerated);
        assert!(supplied.moss_image_present);
        let malformed = AppearanceAuthoredInputs::from_payload_syntax(&obj(json!({
            "advanced":{"mossImageAdvanced":[],"mossImage":{"acceleration":17}}
        })));
        assert!(!malformed.has_embedding_or_per_prompt_negative);
        assert!(!malformed.moss_accelerated);
        assert!(malformed.moss_image_present);
        let explicit_null = AppearanceAuthoredInputs::from_payload_syntax(&obj(json!({
            "advanced":{"mossImage":null}
        })));
        assert!(explicit_null.moss_image_present);
    }

    #[test]
    fn pid_route_projection_retains_nested_source_carrier_shape() {
        let facts = RouteOptionFacts::from_payload_syntax(&obj(json!({
            "advanced":{"pidCheckpoint":{"repo":"owner/model","filename":null},"pidGemma":"gemma"}
        })));
        let RouteField::Valid(checkpoint) = facts.pid_checkpoint else {
            panic!("checkpoint object must bind");
        };
        assert_eq!(checkpoint.repo, RouteField::Valid("owner/model".into()));
        assert_eq!(checkpoint.filename, RouteField::Null);
        assert_eq!(facts.pid_gemma, RouteField::Valid("gemma".into()));
        let malformed = RouteOptionFacts::from_payload_syntax(&obj(json!({
            "advanced":{"pidCheckpoint":[],"pidGemma":17}
        })));
        assert_eq!(malformed.pid_checkpoint, RouteField::Invalid);
        assert_eq!(malformed.pid_gemma, RouteField::Invalid);
    }

    #[test]
    fn route_projection_preserves_absent_null_valid_and_malformed_intent() {
        let facts = RouteOptionFacts::from_payload_syntax(&obj(json!({
            "advanced": {
                "poses": [], "controlMode": null, "controlImage": 17,
                "controlWeights":{"overlayId":"pose-1","_catalogAuthorized":true},
                "mlxQuantize": 4, "veda": {"enabled": true}
            },
            "hiresFix": {"enabled": false}
        })));
        assert_eq!(facts.poses, RouteField::Valid(0));
        assert_eq!(facts.poses_all_objects, RouteField::Valid(0));
        assert_eq!(facts.poses_any_object, RouteField::Valid(false));
        assert_eq!(facts.phases, RouteField::Absent);
        assert_eq!(facts.control_mode, RouteField::Null);
        assert_eq!(facts.control_image, RouteField::Invalid);
        assert!(
            matches!(facts.control_weights, RouteField::Valid(ref weights)
            if weights.overlay_id.valid().map(String::as_str) == Some("pose-1")
                && weights.catalog_authorized.valid() == Some(&true))
        );
        assert_eq!(facts.mlx_quantize, RouteField::Valid(4));
        assert_eq!(facts.veda_enabled, RouteField::Valid(true));
        assert_eq!(facts.hires_fix_enabled, RouteField::Valid(false));
        let auxiliary = facts.auxiliary_selection();
        assert!(!auxiliary.poses_present);
        assert!(!auxiliary.poses_contain_object);
        assert!(auxiliary.veda_enabled);

        let malformed = RouteOptionFacts::from_payload_syntax(&obj(json!({"advanced": []})));
        assert_eq!(malformed.poses, RouteField::Invalid);
        assert_eq!(malformed.poses_all_objects, RouteField::Invalid);
        assert_eq!(malformed.poses_any_object, RouteField::Invalid);
        assert_eq!(malformed.control_mode, RouteField::Invalid);
        assert_eq!(malformed.veda_enabled, RouteField::Invalid);
    }

    #[test]
    fn training_logical_paths_bind_root_and_plan_payloads() {
        for authored in [
            obj(json!({"config":{"advanced":{"iris":{"stage":"S2"}}}})),
            obj(json!({"plan":{"config":{"advanced":{"iris":{"stage":"S2"}}}}})),
        ] {
            let bound = BindingInput::new(&authored, &authored)
                .text("training.config.advanced.iris", "stage")
                .unwrap()
                .unwrap();
            assert_eq!(bound.value, "S2");
            assert_eq!(bound.origin, Origin::Authored { spelling: "stage" });
        }
    }

    #[test]
    fn nested_training_controls_bind_without_reinterpreting_the_engine_wire() {
        let authored = obj(json!({"plan":{"config":{"advanced":{
            "networkType":"full", "mixedPrecision":"fp16", "controlType":"canny",
            "personGate":true, "resumeFromJobId":"job-r", "loadFromJobId":"job-l",
            "outputScope":"project", "requestedGpu":"auto",
            "iris":{"stage":"S2", "expected_global_batch":8, "repa":{"weight":0.25},
                "dist":{"strategy":"ddp", "world":{"world_size":2,"share_device":true,"timeout_secs":600}}},
            "iris_restore":{"init":"parent","degradation":{"use_hflip":false}}
        }}}}));
        let bound = IrisTrainingOptions::bind(&BindingInput::new(&authored, &authored)).unwrap();
        assert_eq!(bound.network_type.as_ref().unwrap().value, "full");
        assert_eq!(bound.control_type.as_ref().unwrap().value, "canny");
        assert!(bound.person_gate.as_ref().unwrap().value);
        assert_eq!(bound.resume_from_job_id.as_ref().unwrap().value, "job-r");
        assert_eq!(bound.load_from_job_id.as_ref().unwrap().value, "job-l");
        assert_eq!(bound.output_scope.as_ref().unwrap().value, "project");
        assert_eq!(bound.requested_gpu.as_ref().unwrap().value, "auto");
        assert_eq!(bound.stage.as_ref().unwrap().value, "S2");
        assert_eq!(bound.dist_strategy.as_ref().unwrap().value, "ddp");
        assert_eq!(bound.dist_world_size.as_ref().unwrap().value, 2);
        assert!(bound.dist_share_device.as_ref().unwrap().value);
        assert_eq!(bound.dist_timeout_secs.as_ref().unwrap().value, 600);
        assert_eq!(bound.expected_global_batch.as_ref().unwrap().value, 8);
        assert_eq!(bound.dist_topology.world_size, 2);
        assert!(bound.dist_topology.share_device);
        assert_eq!(bound.dist_topology.timeout_secs, 600);
        assert!(bound.repa_enabled());
        assert_eq!(bound.restore_init.as_ref().unwrap().value, "parent");
        assert_eq!(
            bound.restore_degradation_wire.unwrap().value,
            obj(json!({"use_hflip":false}))
        );
    }

    #[test]
    fn training_route_probe_retains_missing_and_malformed_network_intent() {
        let plan = obj(json!({"config":{"advanced":{"networkType":"LoKr","resume":true}}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&plan).network_type,
            RouteField::Valid("LoKr".to_owned())
        );
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&plan).resume,
            RouteField::Valid(true)
        );
        let absent = obj(json!({"config":{"advanced":{}}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&absent).network_type,
            RouteField::Absent
        );
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&absent).resume,
            RouteField::Absent
        );
        let malformed = obj(json!({"config":{"advanced":[]}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&malformed).network_type,
            RouteField::Invalid
        );
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&malformed).resume,
            RouteField::Invalid
        );
    }

    #[test]
    fn training_source_probe_preserves_authored_repa_weight_shape() {
        for (value, expected) in [
            (json!(0.5), RouteField::Valid(0.5)),
            (json!(0), RouteField::Valid(0.0)),
            (json!(-0.5), RouteField::Valid(-0.5)),
            (json!("0.5"), RouteField::Invalid),
            (Value::Null, RouteField::Null),
        ] {
            let plan = obj(json!({"config":{"advanced":{"iris":{"repa":{"weight":value}}}}}));
            assert_eq!(
                TrainingRouteFacts::from_plan_syntax(&plan).repa_weight,
                expected
            );
        }
        let absent = obj(json!({"config":{"advanced":{"iris":{"repa":{}}}}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&absent).repa_weight,
            RouteField::Absent
        );
        let malformed = obj(json!({"config":{"advanced":{"iris":{"repa":[]}}}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&malformed).repa_weight,
            RouteField::Invalid
        );
    }

    #[test]
    fn persisted_video_resize_mode_keeps_unclaimed_and_invalid_history_distinct() {
        for (payload, expected) in [
            (json!({}), RouteField::Absent),
            (json!({"advanced":{}}), RouteField::Absent),
            (json!({"advanced":null}), RouteField::Null),
            (
                json!({"advanced":{"referenceImageResizeMode":"match"}}),
                RouteField::Valid(ReferenceImageResizeMode::Match),
            ),
            (
                json!({"advanced":{"referenceImageResizeMode":"diffusers"}}),
                RouteField::Valid(ReferenceImageResizeMode::Diffusers),
            ),
            (
                json!({"advanced":{"referenceImageResizeMode":"typo"}}),
                RouteField::Invalid,
            ),
        ] {
            assert_eq!(
                PersistedVideoResizeModeFact::from_payload_syntax(&obj(payload)).mode,
                expected
            );
        }
    }

    #[test]
    fn artifact_choice_reads_yue2_decoder_from_its_own_request_block() {
        let payload = obj(json!({"advanced":{"decoder":"image-vae"},"yue2":{"decoder":"legacy"}}));
        let facts = RouteOptionFacts::from_payload_syntax(&payload);
        assert_eq!(
            facts
                .artifact_selection(&facts, Some("yue2"))
                .unwrap()
                .yue2_decoder,
            Some("legacy".to_owned())
        );
        assert_eq!(
            facts
                .artifact_selection(&facts, Some("image"))
                .unwrap()
                .yue2_decoder,
            None
        );
        let malformed = RouteOptionFacts::from_payload_syntax(&obj(json!({"yue2":{"decoder":7}})));
        assert!(malformed
            .artifact_selection(&malformed, Some("yue2"))
            .is_err());
        let omitted = RouteOptionFacts::from_payload_syntax(&obj(json!({})));
        let effective_default =
            RouteOptionFacts::from_payload_syntax(&obj(json!({"yue2":{"decoder":"standard"}})));
        assert_eq!(
            effective_default
                .artifact_selection(&omitted, Some("yue2"))
                .unwrap()
                .yue2_decoder,
            None
        );
    }

    #[test]
    fn pid_route_probe_uses_its_narrow_legacy_spelling() {
        for (wire, expected) in [
            (json!(" TrUe "), true),
            (json!("false"), false),
            (json!(1), false),
        ] {
            let payload = obj(json!({"advanced":{"usePid":wire}}));
            assert_eq!(
                RouteOptionFacts::from_payload_syntax(&payload).use_pid,
                RouteField::Valid(expected)
            );
        }
    }

    #[test]
    fn loaded_model_hints_preserve_saved_repo_syntax() {
        let facts = RouteOptionFacts::from_payload_syntax(&obj(json!({
            "advanced":{"modelRepo":" org/model ","repo":"","modelPath":"ignored"}
        })));
        assert_eq!(
            facts.model_repo,
            RouteField::Valid(" org/model ".to_owned())
        );
        assert_eq!(facts.repo, RouteField::Valid(String::new()));
        let malformed = RouteOptionFacts::from_payload_syntax(&obj(json!({
            "advanced":{"modelRepo":7,"repo":null}
        })));
        assert_eq!(malformed.model_repo, RouteField::Invalid);
        assert_eq!(malformed.repo, RouteField::Null);
    }

    #[test]
    fn bare_route_probe_uses_portable_legacy_numeric_and_angle_parsers() {
        let facts = RouteOptionFacts::from_payload_syntax(&obj(json!({
            "advanced": {
                "mlxQuantize": " 4 ",
                "referenceStrength": "0.45",
                "trueCfgScale": "2.5",
                "angleSet": "false"
            }
        })));
        assert_eq!(facts.mlx_quantize, RouteField::Valid(4));
        assert_eq!(facts.reference_strength, RouteField::Valid(0.45));
        assert_eq!(facts.true_cfg_scale, RouteField::Valid(2.5));
        assert_eq!(facts.angle_set, RouteField::Valid(true));
        let malformed = RouteOptionFacts::from_payload_syntax(&obj(json!({
            "advanced":{"mlxQuantize":"4.5","trueCfgScale":"nan","angleSet":{}}
        })));
        assert_eq!(malformed.mlx_quantize, RouteField::Invalid);
        assert_eq!(malformed.true_cfg_scale, RouteField::Invalid);
        assert_eq!(malformed.angle_set, RouteField::Valid(false));
    }

    #[test]
    fn iris_task_wire_keeps_only_the_selected_object() {
        let wire = ValidatedModelOptionsWire(obj(json!({
            "iris": {"stage": "s1"},
            "iris_restore": {"init": "parent"},
            "unused": true
        })));
        assert_eq!(
            wire.for_iris_task(crate::training::iris::IrisObjective::Generation),
            obj(json!({"iris":{"stage":"s1"}}))
        );
        assert_eq!(
            wire.for_iris_task(crate::training::iris::IrisObjective::Restoration),
            obj(json!({"iris_restore":{"init":"parent"}}))
        );
        let malformed = ValidatedModelOptionsWire(obj(json!({"iris": false})));
        assert_eq!(
            malformed.for_iris_task(crate::training::iris::IrisObjective::Depth),
            obj(json!({"iris":{}}))
        );
    }

    #[test]
    fn iris_activation_checkpointing_uses_owner_memory_floor_policy() {
        for (mode, full) in [
            ("full", true),
            ("selective_op", true),
            ("selective_layer", false),
            ("none", false),
        ] {
            let payload = obj(json!({"config":{"advanced":{"iris":{
                "activation_checkpointing": mode
            }}}}));
            let bound = IrisTrainingOptions::bind(&BindingInput::new(&payload, &payload)).unwrap();
            assert_eq!(bound.full_activation_checkpointing, full, "{mode}");
        }
    }

    #[test]
    fn training_family_wire_and_flux3_defaults_match_source_profiles() {
        let empty = JsonObject::new();
        let flux3 = crate::training::resolve_advanced_for_model(
            &empty,
            "flux3_klein_lora",
            "flux3_klein_base",
        )
        .unwrap();
        assert_eq!(flux3.timestep_type, "uniform");
        assert_eq!(flux3.weight_decay, 0.0001);
        assert_eq!(
            crate::training::training_model_options_wire(&empty, "flux3_klein_lora"),
            obj(json!({"flux3":{"mode":"captioned"}}))
        );
        let authored = obj(json!({
            "resume": true,
            "flux3": {"mode":"edit", "caption_dropout":0.2},
            "unrelated": "ignored"
        }));
        assert_eq!(
            crate::training::training_model_options_wire(&authored, "flux3_klein_edit_lora"),
            obj(json!({"flux3":{"mode":"edit","caption_dropout":0.2}}))
        );
        assert_eq!(
            crate::training::training_model_options_wire(&empty, "moss_image_lora"),
            obj(json!({"vae_posterior":"posterior_mean_v1"}))
        );
        assert_eq!(
            crate::training::training_model_options_wire(&authored, "iris_train"),
            authored
        );
    }
}
