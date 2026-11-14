//! One-way lowering from the shared validator's effective request into native option fields.
//!
//! The authoring payload remains available to the upgrader and persistence layer. Execution
//! code receives domain structs built from this input, not the mutable `advanced` JSON bag.
//! This module does not coerce, clamp, default or decide compatibility: those rules belong to
//! `sceneworks_options::request_validation::resolve` before `BindingInput` is constructed.

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::contracts::JsonObject;
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
    pub phases: RouteField<usize>,
    pub control_mode: RouteField<String>,
    pub control_image: RouteField<String>,
    pub control_weights: RouteField<()>,
    pub control_scale: RouteField<f64>,
    pub true_cfg_scale: RouteField<f64>,
    pub image_guidance_scale: RouteField<f64>,
    pub ip_adapter_scale: RouteField<f64>,
    pub reference_strength: RouteField<f64>,
    pub mlx_quantize: RouteField<i64>,
    pub quant_tier: RouteField<String>,
    pub decoder: RouteField<String>,
    pub enhance_prompt: RouteField<bool>,
    pub use_pid: RouteField<bool>,
    pub veda_enabled: RouteField<bool>,
    pub use_uncensored_enhancer: RouteField<bool>,
    pub text_encoder_model: RouteField<String>,
    pub strength: RouteField<f64>,
    pub hires_fix_enabled: RouteField<bool>,
    pub conv_rot: RouteField<bool>,
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
        Self {
            poses: field(advanced, "poses", count),
            poses_all_objects: field(advanced, "poses", |value| {
                value
                    .as_array()
                    .and_then(|items| items.iter().all(Value::is_object).then_some(items.len()))
            }),
            phases: field(advanced, "phases", count),
            control_mode: field(advanced, "controlMode", text),
            control_image: field(advanced, "controlImage", text),
            control_weights: field(advanced, "controlWeights", |v| v.as_object().map(|_| ())),
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
            decoder: field(advanced, "decoder", text),
            enhance_prompt: field(advanced, "enhancePrompt", boolean),
            use_pid: field(advanced, "usePid", boolean),
            veda_enabled: field(veda, "enabled", boolean),
            use_uncensored_enhancer: field(advanced, "useUncensoredEnhancer", boolean),
            text_encoder_model: field(advanced, "textEncoderModel", text),
            strength: field(advanced, "strength", |v| {
                v.as_f64().filter(|n| n.is_finite())
            }),
            hires_fix_enabled: field(hires_fix, "enabled", boolean),
            conv_rot: field(advanced, "convRot", boolean),
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
        let mut map = root;
        for part in location.split('.') {
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
                "mlxQuantize": 4, "veda": {"enabled": true}
            },
            "hiresFix": {"enabled": false}
        })));
        assert_eq!(facts.poses, RouteField::Valid(0));
        assert_eq!(facts.poses_all_objects, RouteField::Valid(0));
        assert_eq!(facts.phases, RouteField::Absent);
        assert_eq!(facts.control_mode, RouteField::Null);
        assert_eq!(facts.control_image, RouteField::Invalid);
        assert_eq!(facts.mlx_quantize, RouteField::Valid(4));
        assert_eq!(facts.veda_enabled, RouteField::Valid(true));
        assert_eq!(facts.hires_fix_enabled, RouteField::Valid(false));

        let malformed = RouteOptionFacts::from_payload_syntax(&obj(json!({"advanced": []})));
        assert_eq!(malformed.poses, RouteField::Invalid);
        assert_eq!(malformed.poses_all_objects, RouteField::Invalid);
        assert_eq!(malformed.control_mode, RouteField::Invalid);
        assert_eq!(malformed.veda_enabled, RouteField::Invalid);
    }
}
