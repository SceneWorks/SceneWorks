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

impl<T> RouteField<T> {
    pub fn is_absent_or_null(&self) -> bool {
        matches!(self, Self::Absent | Self::Null)
    }

    pub fn is_present_nonnull(&self) -> bool {
        matches!(self, Self::Valid(_) | Self::Invalid)
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
    pub veda_enabled: RouteField<bool>,
    pub use_uncensored_enhancer: RouteField<bool>,
    pub text_encoder_model: RouteField<String>,
    pub model_path: RouteField<String>,
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
            control_scale: field(advanced, "controlScale", |v| {
                v.as_f64().filter(|n| n.is_finite())
            }),
            true_cfg_scale: field(advanced, "trueCfgScale", |v| {
                v.as_f64().filter(|n| n.is_finite())
            }),
            image_guidance_scale: field(advanced, "imageGuidanceScale", |v| {
                v.as_f64().filter(|n| n.is_finite())
            }),
            ip_adapter_scale: field(advanced, "ipAdapterScale", |v| {
                v.as_f64().filter(|n| n.is_finite())
            }),
            reference_strength: field(advanced, "referenceStrength", |v| {
                v.as_f64().filter(|n| n.is_finite())
            }),
            mlx_quantize: field(advanced, "mlxQuantize", Value::as_i64),
            quant_tier: field(advanced, "quantTier", text),
            quant: field(advanced, "quant", text),
            quantization: field(advanced, "quantization", text),
            decoder: field(advanced, "decoder", text),
            enhance_prompt: field(advanced, "enhancePrompt", boolean),
            use_pid: field(advanced, "usePid", |value| {
                Some(crate::portable_sources::pid_requested(Some(value)))
            }),
            veda_enabled: field(veda, "enabled", boolean),
            use_uncensored_enhancer: field(advanced, "useUncensoredEnhancer", boolean),
            text_encoder_model: field(advanced, "textEncoderModel", text),
            model_path: field(advanced, "modelPath", text),
            tile_control_net_repo: field(advanced, "tileControlNetRepo", text),
            strength: field(advanced, "strength", |v| {
                v.as_f64().filter(|n| n.is_finite())
            }),
            hires_fix_enabled: field(hires_fix, "enabled", boolean),
            yue2_decoder: field(yue2, "decoder", text),
            conv_rot: field(advanced, "convRot", boolean),
            angle_set: field(advanced, "angleSet", boolean),
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
    pub base_model_path: RouteField<String>,
}

impl TrainingRouteFacts {
    pub fn from_plan_syntax(plan: &JsonObject) -> Self {
        let network_type = plan
            .get("config")
            .ok_or(())
            .and_then(|value| value.as_object().ok_or(()))
            .and_then(|config| config.get("advanced").ok_or(()))
            .and_then(|value| value.as_object().ok_or(()));
        let network_type = match network_type {
            Ok(advanced) => match advanced.get("networkType") {
                None => RouteField::Absent,
                Some(Value::Null) => RouteField::Null,
                Some(Value::String(value)) => RouteField::Valid(value.clone()),
                Some(_) => RouteField::Invalid,
            },
            // An omitted advanced object is ordinary absence; malformed or missing config is
            // invalid because no training route can prove that the plan is well formed.
            Err(())
                if plan
                    .get("config")
                    .and_then(Value::as_object)
                    .is_some_and(|config| !config.contains_key("advanced")) =>
            {
                RouteField::Absent
            }
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
            base_model_path,
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
    pub depth_init: Option<Bound<String>>,
    pub stage: Option<Bound<String>>,
    pub parent_stage: Option<Bound<String>>,
    pub resume_data_policy: Option<Bound<String>>,
    pub resume_optimizer: Option<Bound<String>>,
    pub dist_strategy: Option<Bound<String>>,
    pub repa_weight: Option<Bound<f64>>,
    pub restore_init: Option<Bound<String>>,
    pub restore_resume_data_policy: Option<Bound<String>>,
    /// Validated opaque engine ABI; callers may pass it to the engine parser, not read its keys.
    pub restore_degradation_wire: Option<Bound<JsonObject>>,
}

impl IrisTrainingOptions {
    pub fn bind(input: &BindingInput<'_>) -> Result<Self, BindingError> {
        Ok(Self {
            network_type: input.text("training.config.advanced", "networkType")?,
            mixed_precision: input.text("training.config.advanced", "mixedPrecision")?,
            depth_init: input.text("training.config.advanced", "depthInit")?,
            stage: input.text("training.config.advanced.iris", "stage")?,
            parent_stage: input.text("training.config.advanced.iris", "parent_stage")?,
            resume_data_policy: input
                .text("training.config.advanced.iris", "resume_data_policy")?,
            resume_optimizer: input.text("training.config.advanced.iris", "resume_optimizer")?,
            dist_strategy: input.text("training.config.advanced.iris.dist", "strategy")?,
            repa_weight: input.number("training.config.advanced.iris.repa", "weight")?,
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
        self.repa_weight
            .as_ref()
            .is_some_and(|weight| weight.value > 0.0)
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
            "networkType":"full", "mixedPrecision":"fp16",
            "iris":{"stage":"S2", "repa":{"weight":0.25},
                "dist":{"strategy":"ddp"}},
            "iris_restore":{"init":"parent","degradation":{"use_hflip":false}}
        }}}}));
        let bound = IrisTrainingOptions::bind(&BindingInput::new(&authored, &authored)).unwrap();
        assert_eq!(bound.network_type.as_ref().unwrap().value, "full");
        assert_eq!(bound.stage.as_ref().unwrap().value, "S2");
        assert_eq!(bound.dist_strategy.as_ref().unwrap().value, "ddp");
        assert!(bound.repa_enabled());
        assert_eq!(bound.restore_init.as_ref().unwrap().value, "parent");
        assert_eq!(
            bound.restore_degradation_wire.unwrap().value,
            obj(json!({"use_hflip":false}))
        );
    }

    #[test]
    fn training_route_probe_retains_missing_and_malformed_network_intent() {
        let plan = obj(json!({"config":{"advanced":{"networkType":"LoKr"}}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&plan).network_type,
            RouteField::Valid("LoKr".to_owned())
        );
        let absent = obj(json!({"config":{"advanced":{}}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&absent).network_type,
            RouteField::Absent
        );
        let malformed = obj(json!({"config":{"advanced":[]}}));
        assert_eq!(
            TrainingRouteFacts::from_plan_syntax(&malformed).network_type,
            RouteField::Invalid
        );
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
}
