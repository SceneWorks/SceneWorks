//! Measured builtin facts. The shipped manifest owns product choices; this store owns the
//! calibration and memory facts projected into the effective model catalog.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use serde_json::Value;

const BACKEND_FIELDS: &[&str] = &[
    "calibrations",
    "control",
    "imageActivationGbPerMegapixel",
    "measured",
    "measuredSiblingActiveDiskFraction",
    "memoryDeclarationWithhold",
    "memoryStrategyCapabilities",
    "memoryStrategyContract",
    "memoryStrategyStructuralExemptions",
    "minMemoryGb",
    "minMemoryGbByTier",
    "sequentialPeakGb",
    "supportsSequentialOffload",
    "turboFit",
    "vramGbByTier",
    "vramMeasuredPixels",
];
const DOWNLOAD_IDENTITY: &[&str] = &[
    "provider",
    "repo",
    "revision",
    "variant",
    "files",
    "platforms",
];
const MEASURED_FOOTPRINT: &[&str] = &["residentMemoryBytes", "peakMemoryBytes", "measuredPixels"];

static EVIDENCE: LazyLock<BTreeMap<String, Value>> = LazyLock::new(|| {
    serde_json::from_str(include_str!(
        "../../../../config/manifests/registry/evidence.json"
    ))
    .expect("build-validated builtin memory evidence")
});

pub fn builtin_memory_evidence(id: &str) -> Option<&'static Value> {
    EVIDENCE.get(id)
}

/// Restore the shipped effective model shape from the separate evidence authority. This is
/// idempotent, and intentionally overwrites user-supplied measured fields for builtin IDs.
pub fn project_builtin_memory_evidence(model: &mut Value) -> Result<(), String> {
    let id = model
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| "model projection has no id".to_owned())?
        .to_owned();
    let Some(evidence) = builtin_memory_evidence(&id) else {
        return Ok(());
    };
    let evidence = evidence
        .as_object()
        .ok_or_else(|| format!("{id} evidence is not an object"))?;
    for (location, fields) in evidence {
        match location.as_str() {
            "model" => {
                let target = model
                    .as_object_mut()
                    .ok_or_else(|| format!("{id} model is not an object"))?;
                target.extend(fields.as_object().expect("validated root evidence").clone());
            }
            "mlx" | "candle" => {
                let target = model
                    .get_mut(location)
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| format!("{id} has no {location} backend for evidence"))?;
                target.extend(
                    fields
                        .as_object()
                        .expect("validated backend evidence")
                        .clone(),
                );
            }
            "downloads" => {
                let target = model
                    .get_mut("downloads")
                    .and_then(Value::as_array_mut)
                    .ok_or_else(|| format!("{id} has no downloads for measured footprint"))?;
                for entry in fields.as_array().expect("validated download evidence") {
                    let mut matches = target
                        .iter_mut()
                        .filter(|candidate| download_identity_matches(candidate, entry));
                    let download = matches
                        .next()
                        .ok_or_else(|| format!("{id} has no product download for evidence"))?;
                    if matches.next().is_some() {
                        return Err(format!("{id} measured download identity is ambiguous"));
                    }
                    let footprint = download
                        .get_mut("footprint")
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| format!("{id} download has no product footprint"))?;
                    footprint.extend(
                        entry["footprint"]
                            .as_object()
                            .expect("validated measured footprint")
                            .clone(),
                    );
                }
            }
            _ => unreachable!("build-validated evidence location"),
        }
    }
    Ok(())
}

fn download_identity_matches(candidate: &Value, evidence: &Value) -> bool {
    // The original five identity fields win. A platform pin is required only when those fields
    // collide (FLUX3 Klein's macOS and CUDA local derivations share the source artifact).
    DOWNLOAD_IDENTITY[..5]
        .iter()
        .all(|key| candidate.get(*key) == evidence.get(*key))
        && evidence.get("platforms").map_or(true, |platforms| {
            candidate.get("platforms") == Some(platforms)
        })
}

/// Build-time validation covers *all* legacy rows, even when a model is not yet represented by
/// a typed ModelSpec. Thus a later product migration cannot surface previously ignored bad data.
pub fn validate_evidence_document(contents: &str, manifest: &Value) -> Result<(), String> {
    let document: Value = serde_json::from_str(contents).map_err(|error| error.to_string())?;
    let rows = document
        .as_object()
        .ok_or_else(|| "memory evidence must be an object".to_owned())?;
    let models = manifest
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| "builtin models manifest has no models array".to_owned())?;
    let mut known = BTreeMap::new();
    for model in models {
        let id = model
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "builtin model has no id".to_owned())?;
        if known.insert(id, model).is_some() {
            return Err(format!("duplicate builtin model {id}"));
        }
    }
    for (id, record) in rows {
        let model = known
            .get(id.as_str())
            .ok_or_else(|| format!("memory evidence has unknown model {id}"))?;
        let record = record
            .as_object()
            .ok_or_else(|| format!("{id} evidence must be an object"))?;
        if record.is_empty() {
            return Err(format!("{id} evidence is empty"));
        }
        for (location, fields) in record {
            match location.as_str() {
                "model" => {
                    let fields = fields
                        .as_object()
                        .ok_or_else(|| format!("{id} root evidence must be an object"))?;
                    if fields.len() != 1
                        || !fields
                            .get("minMemoryGb")
                            .is_some_and(|value| value.as_u64().is_some())
                    {
                        return Err(format!("{id} has invalid root memory evidence"));
                    }
                }
                "mlx" | "candle" => validate_backend(id, location, fields, model)?,
                "downloads" => validate_downloads(id, fields, model)?,
                _ => return Err(format!("{id} has unknown evidence location {location}")),
            }
        }
    }
    Ok(())
}

fn validate_backend(id: &str, backend: &str, fields: &Value, model: &Value) -> Result<(), String> {
    if !model.get(backend).is_some_and(Value::is_object) {
        return Err(format!("{id} has evidence for absent {backend} backend"));
    }
    let fields = fields
        .as_object()
        .ok_or_else(|| format!("{id}/{backend} evidence must be an object"))?;
    if fields.is_empty() {
        return Err(format!("{id}/{backend} evidence is empty"));
    }
    for (name, value) in fields {
        if !BACKEND_FIELDS.contains(&name.as_str()) {
            return Err(format!("{id}/{backend} has unknown evidence field {name}"));
        }
        let valid = match name.as_str() {
            "minMemoryGb" | "vramMeasuredPixels" => value.as_u64().is_some(),
            "measured" | "supportsSequentialOffload" => value.is_boolean(),
            "calibrations" => value.is_array(),
            "imageActivationGbPerMegapixel" | "measuredSiblingActiveDiskFraction" => {
                value.as_f64().is_some()
            }
            _ => value.is_object(),
        };
        if !valid {
            return Err(format!("{id}/{backend}.{name} has invalid value type"));
        }
    }
    Ok(())
}

fn validate_downloads(id: &str, fields: &Value, model: &Value) -> Result<(), String> {
    let fields = fields
        .as_array()
        .ok_or_else(|| format!("{id} download evidence must be an array"))?;
    if fields.is_empty() {
        return Err(format!("{id} download evidence is empty"));
    }
    let downloads = model
        .get("downloads")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{id} model has no downloads"))?;
    let mut seen = BTreeSet::new();
    for entry in fields {
        let entry_fields = entry
            .as_object()
            .ok_or_else(|| format!("{id} download evidence row is not an object"))?;
        if entry_fields
            .keys()
            .any(|name| name != "footprint" && !DOWNLOAD_IDENTITY.contains(&name.as_str()))
            || !["provider", "repo", "revision", "files"]
                .iter()
                .all(|name| entry_fields.contains_key(*name))
            || !entry_fields.contains_key("footprint")
        {
            return Err(format!("{id} has invalid measured download identity"));
        }
        let identity = serde_json::to_string(
            &DOWNLOAD_IDENTITY
                .iter()
                .map(|name| &entry[*name])
                .collect::<Vec<_>>(),
        )
        .expect("serializable identity");
        if !seen.insert(identity) {
            return Err(format!("{id} has duplicate measured download identity"));
        }
        if downloads
            .iter()
            .filter(|candidate| download_identity_matches(candidate, entry))
            .count()
            != 1
        {
            return Err(format!("{id} has no unique product download for evidence"));
        }
        let footprint = entry["footprint"]
            .as_object()
            .ok_or_else(|| format!("{id} measured footprint is not an object"))?;
        if footprint.is_empty()
            || footprint.iter().any(|(name, value)| {
                !MEASURED_FOOTPRINT.contains(&name.as_str())
                    || !(value.as_u64().is_some()
                        || ((name == "residentMemoryBytes" || name == "peakMemoryBytes")
                            && value.is_null()))
            })
        {
            return Err(format!("{id} has invalid measured footprint"));
        }
    }
    Ok(())
}
