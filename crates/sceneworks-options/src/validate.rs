//! Raw authored-value checks driven by the registry. Defaults, host facts and unknown extension
//! fields are left intact. In particular a malformed known value cannot become an omitted value.
use crate::{registry, Shape};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeError {
    pub field: String,
    pub detail: String,
}

pub fn shape(kind: &str, payload: &Value) -> Result<(), ShapeError> {
    if let Some(value) = payload
        .get("advanced")
        .and_then(|advanced| advanced.get("steps"))
    {
        let steps = value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse::<u64>().ok());
        if steps.is_none_or(|steps| steps == 0 || steps > u64::from(u32::MAX)) {
            return Err(ShapeError {
                field: "advanced.steps".to_owned(),
                detail:
                    "advanced.steps must be a positive unsigned 32-bit integer or numeric string"
                        .to_owned(),
            });
        }
    }
    for key in [
        "referenceAssetIds",
        "sourceClipAssetIds",
        "referenceAudioAssetIds",
        "sourceAssetIds",
    ] {
        if let Some(Value::Array(values)) = payload.get(key) {
            if values.iter().any(|value| {
                value
                    .as_str()
                    .is_none_or(|id| id.trim().is_empty() || id != id.trim())
            }) {
                return Err(ShapeError {
                    field: key.to_owned(),
                    detail: format!(
                        "{key} must contain nonblank string ids without surrounding whitespace"
                    ),
                });
            }
        }
    }
    for spec in registry() {
        let location = if spec.location == "advanced" || spec.location.starts_with("advanced.") {
            spec.location
        } else if spec.location == kind {
            ""
        } else if let Some(location) = spec.location.strip_prefix(&format!("{kind}.")) {
            location
        } else {
            continue;
        };
        let mut parents = vec![(String::new(), payload)];
        if !location.is_empty() {
            for segment in location.split('.') {
                let array = segment.ends_with("[]");
                let key = segment.trim_end_matches("[]");
                parents = parents
                    .into_iter()
                    .flat_map(|(path, parent)| {
                        let prefix = if path.is_empty() {
                            key.to_owned()
                        } else {
                            format!("{path}.{key}")
                        };
                        match parent.get(key) {
                            Some(Value::Array(values)) if array => values
                                .iter()
                                .enumerate()
                                .map(|(i, value)| (format!("{prefix}[{i}]"), value))
                                .collect(),
                            Some(value) if !array => vec![(prefix, value)],
                            _ => Vec::new(),
                        }
                    })
                    .collect();
            }
        }
        for (path, parent) in parents {
            for key in std::iter::once(spec.key).chain(spec.aliases.iter().copied()) {
                let Some(value) = parent.get(key).filter(|value| !value.is_null()) else {
                    continue;
                };
                // Host-owned receipts remain readable; their producer owns their schema.
                if !spec.user_settable {
                    continue;
                }
                let numeric_string = spec.compatibility.contains("numeric string");
                let number = || {
                    value
                        .as_f64()
                        .or_else(|| {
                            numeric_string
                                .then(|| value.as_str()?.trim().parse::<f64>().ok())
                                .flatten()
                        })
                        .filter(|v| v.is_finite())
                };
                let valid = match spec.shape {
                    Shape::Bool if spec.compatibility.contains("Legacy image flag") => {
                        value.is_boolean()
                            || value.is_number()
                            || value.is_string()
                            || value.is_array()
                    }
                    Shape::Bool => value.is_boolean(),
                    Shape::Integer => {
                        value.as_i64().is_some()
                            || value.as_u64().is_some()
                            || (numeric_string
                                && value
                                    .as_str()
                                    .and_then(|v| v.trim().parse::<i64>().ok())
                                    .is_some())
                    }
                    Shape::Number => number().is_some(),
                    Shape::Text => value.is_string(),
                    Shape::OneOf(description) if description.starts_with("JSON value retained") => {
                        true
                    }
                    Shape::OneOf("2 | 4") => matches!(value.as_u64(), Some(2 | 4)),
                    Shape::OneOf(_) if spec.location == "promptRefine" => value.is_string(),
                    Shape::OneOf(description)
                        if !spec.compatibility.contains("Unknown or absent")
                            && description.contains(" | ")
                            && !description.contains("empty legacy alias") =>
                    {
                        value.as_str().is_some_and(|text| {
                            description.split(" | ").any(|choice| choice == text)
                        })
                    }
                    Shape::OneOf(_) => value.is_string(),
                    Shape::Strings => value
                        .as_array()
                        .is_some_and(|a| a.iter().all(Value::is_string)),
                    Shape::Numbers => value
                        .as_array()
                        .is_some_and(|a| a.iter().all(|v| v.as_f64().is_some_and(f64::is_finite))),
                    Shape::Object(_) => value.is_object(),
                    Shape::List(_) => value.is_array(),
                };
                if !valid {
                    let field = if path.is_empty() {
                        key.to_owned()
                    } else {
                        format!("{path}.{key}")
                    };
                    return Err(ShapeError {
                        detail: format!("{field} has an invalid value for {:?}", spec.shape),
                        field,
                    });
                }
            }
        }
    }
    Ok(())
}
