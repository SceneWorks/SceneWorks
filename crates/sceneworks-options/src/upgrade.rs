//! Lossless upgrade of stored and external option carriers. This module does not resolve
//! defaults, select models, validate availability, or replace the caller's original document.
use serde::Serialize;
use serde_json::{Map, Value};
use std::fmt;

pub const OPTIONS_SCHEMA_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carrier {
    Request,
    Job,
    Recipe,
    StudioSettings,
    SimpleSettings,
    FilmPlan,
    CompiledFilmPlan,
    TrainingPlan,
    ShareEnvelope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpgradeError {
    pub code: &'static str,
    pub field: String,
    pub message: String,
}
impl fmt::Display for UpgradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl std::error::Error for UpgradeError {}

fn refusal(code: &'static str, field: &str, message: impl Into<String>) -> UpgradeError {
    UpgradeError {
        code,
        field: field.into(),
        message: message.into(),
    }
}
fn version(
    object: &Map<String, Value>,
    key: &str,
    supported: u64,
    path: &str,
) -> Result<(), UpgradeError> {
    if let Some(value) = object.get(key) {
        let field = format!("{path}{key}");
        let actual = value.as_u64().ok_or_else(|| {
            refusal(
                "InvalidSchemaVersion",
                &field,
                "Expected a positive integer schema version",
            )
        })?;
        if actual == 0 || actual > supported {
            return Err(refusal("UnsupportedSchemaVersion", &field, format!("Schema version {actual} is unsupported; this reader supports through {supported}")));
        }
    }
    Ok(())
}

fn alias_number(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Number(value) => value.as_f64()?,
        Value::String(value) if !value.trim().is_empty() => value.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    (number.is_finite() && number.abs() <= 9_007_199_254_740_991.0).then_some(number)
}

fn aliases_equal(left: &Value, right: &Value, shape: crate::Shape) -> bool {
    left == right
        || (matches!(shape, crate::Shape::Number | crate::Shape::Integer)
            && alias_number(left)
                .zip(alias_number(right))
                .is_some_and(|(left, right)| left == right))
}

/// Original spellings remain available to older consumers. Canonical names are added only
/// when an alias was authored, never for an omitted field. Even context/telemetry aliases
/// survive byte-value-identically. Conflicts refuse rather than choosing an arbitrary winner.
fn aliases(
    object: &mut Map<String, Value>,
    location: &str,
    path: &str,
) -> Result<(), UpgradeError> {
    for spec in
        crate::registry().filter(|spec| spec.location == location && !spec.aliases.is_empty())
    {
        let mut authored: Option<(&str, Value)> =
            object.get(spec.key).map(|v| (spec.key, v.clone()));
        for alias in spec.aliases {
            if let Some(value) = object.get(*alias) {
                if let Some((name, existing)) = &authored {
                    if !aliases_equal(existing, value, spec.shape) {
                        return Err(refusal(
                            "ConflictingAliases",
                            &format!("{path}{}", spec.key),
                            format!("{name} and {alias} contain different authored values"),
                        ));
                    }
                } else {
                    authored = Some((alias, value.clone()));
                }
            }
        }
        if let Some((_, value)) = authored {
            object.entry(spec.key.to_owned()).or_insert(value);
        }
    }
    Ok(())
}

fn walk(value: &mut Value, path: &str, depth: usize) -> Result<(), UpgradeError> {
    if depth > 64 {
        return Err(refusal(
            "CarrierTooDeep",
            path,
            "Option carrier exceeds 64 nested levels",
        ));
    }
    match value {
        Value::Object(object) => {
            version(object, "optionsSchemaVersion", OPTIONS_SCHEMA_VERSION, path)?;
            if object.contains_key("planVersion")
                && object.contains_key("target")
                && object.contains_key("dataset")
            {
                version(object, "planVersion", 1, path)?;
                version(object, "schemaVersion", 1, path)?;
            }
            // Do not reinterpret arbitrary provenance, telemetry, prompts, or model manifests.
            // Only the declared option carriers receive alias normalization.
            for (key, child) in object.iter_mut() {
                let child_path = format!("{path}{key}.");
                match key.as_str() {
                    "lowResStart" => {
                        if let Some(recipe) = child.as_object() {
                            if !recipe.contains_key("version") {
                                return Err(refusal(
                                    "MissingRecipeVersion",
                                    &format!("{child_path}version"),
                                    "lowResStart requires an explicit recipe version",
                                ));
                            }
                            version(recipe, "version", 1, &child_path)?;
                        }
                    }
                    "advanced" | "rawAdapterSettings" => {
                        if let Some(options) = child.as_object_mut() {
                            aliases(options, "advanced", &child_path)?;
                        }
                        walk(child, &child_path, depth + 1)?;
                    }
                    "payload" => {
                        if let Some(request) = child.as_object_mut() {
                            aliases(request, "image", &child_path)?;
                        }
                        walk(child, &child_path, depth + 1)?;
                    }
                    "plan" | "normalizedSettings" | "settings" | "recipe" | "config" | "model"
                    | "shots" | "scenes" | "segments" | "generationSet" => {
                        walk(child, &child_path, depth + 1)?
                    }
                    _ => {}
                }
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter_mut().enumerate() {
                walk(child, &format!("{path}{index}."), depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Upgrade a copy; on refusal the caller's stored original is untouched. Existing film
/// audio-intent migrations remain owned by the film parser, and are never invented here.
pub fn upgrade(input: &Value, carrier: Carrier) -> Result<Value, UpgradeError> {
    upgrade_owned(input.clone(), carrier)
}

fn upgrade_owned(mut result: Value, carrier: Carrier) -> Result<Value, UpgradeError> {
    let original = result
        .as_object()
        .ok_or_else(|| refusal("InvalidCarrier", "", "Expected an object"))?;
    let supported = match carrier {
        Carrier::FilmPlan => Some(3),
        Carrier::CompiledFilmPlan => Some(7),
        Carrier::TrainingPlan => Some(1),
        Carrier::ShareEnvelope => Some(2),
        _ => None,
    };
    if let Some(supported) = supported {
        version(original, "schemaVersion", supported, "")?;
    }
    let object = result.as_object_mut().expect("checked object");
    if matches!(carrier, Carrier::Request | Carrier::Job) {
        aliases(object, "image", "")?;
    }
    if matches!(carrier, Carrier::StudioSettings) {
        aliases(object, "advanced", "")?;
    }
    walk(&mut result, "", 0)?;
    Ok(result)
}

pub fn upgrade_object(
    input: &Map<String, Value>,
    carrier: Carrier,
) -> Result<Map<String, Value>, UpgradeError> {
    match upgrade_owned(Value::Object(input.clone()), carrier)? {
        Value::Object(object) => Ok(object),
        _ => unreachable!("upgrade preserves the carrier object"),
    }
}

/// Data for the browser's lossless adapter, generated from the same alias declarations.
pub fn metadata() -> Value {
    serde_json::json!({
        "optionsSchemaVersion": OPTIONS_SCHEMA_VERSION,
        "recipeVersions": {"lowResStart": 1},
        "aliases": crate::registry().filter(|s| !s.aliases.is_empty()).map(|s|
            serde_json::json!({"location": s.location, "key": s.key, "aliases": s.aliases, "numeric": matches!(s.shape, crate::Shape::Number | crate::Shape::Integer)})
        ).collect::<Vec<_>>()
    })
}

/// Fresh generation keeps authored intent and logical source bindings, but cannot reuse the
/// previous machine's admission choices or completed-output evidence. Exact replay is separate.
pub fn strip_machine_decisions(payload: &mut Map<String, Value>) {
    if let Some(advanced) = payload.get_mut("advanced").and_then(Value::as_object_mut) {
        for key in [
            "memoryStrategy",
            "memoryEvidenceRevision",
            "realModelInference",
            "checkpointWeightFacts",
            "mossImageExecutionReport",
            "moss_image_execution_report",
            "mossImageEditExecutionReport",
            "moss_image_edit_execution_report",
            "mossImageSampleIndex",
            "promptEnhancement",
        ] {
            advanced.remove(key);
        }
    }
}
