//! Version normalization for user model manifests. Callers that persist the
//! normalized value must hold the shared `<manifest>.lock` across read and write.

use serde_json::{Map, Value};

pub const MODEL_MANIFEST_SCHEMA_VERSION: u64 = 2;
pub const MODEL_MANIFEST_SCHEMA_URL: &str =
    "https://sceneworks.local/schemas/model-manifest.schema.json";

/// Accept an old user model manifest without losing its entries or extension
/// fields, and make its in-memory representation the current version. A later
/// writer can atomically persist the returned value under the manifest lock.
pub fn normalize_user_model_manifest(root: &mut Value) -> Result<bool, String> {
    let object = root
        .as_object_mut()
        .ok_or_else(|| "model manifest must be a JSON object".to_owned())?;
    normalize_user_model_manifest_object(object)
}

/// Read-side entry point for every consumer of a user model manifest: strip JSONC comments, parse,
/// and normalize. An unknown future `schemaVersion` (or malformed header) is refused here, so a
/// read-only consumer can never interpret a newer file as the current version.
pub fn parse_user_model_manifest(raw: &str) -> Result<Value, String> {
    let mut root: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw))
        .map_err(|error| format!("model manifest is not valid JSON: {error}"))?;
    normalize_user_model_manifest(&mut root)?;
    Ok(root)
}

pub fn normalize_user_model_manifest_object(root: &mut Map<String, Value>) -> Result<bool, String> {
    let version = match root.get("schemaVersion") {
        None => 1, // Old user JSONC files sometimes omitted the header entirely.
        Some(Value::Number(number)) => number
            .as_u64()
            .ok_or_else(|| "model manifest schemaVersion must be a positive integer".to_owned())?,
        Some(_) => return Err("model manifest schemaVersion must be a positive integer".to_owned()),
    };
    if version != 1 && version != MODEL_MANIFEST_SCHEMA_VERSION {
        return Err(format!(
            "unsupported model manifest schemaVersion {version}; supported versions are 1 and {MODEL_MANIFEST_SCHEMA_VERSION}"
        ));
    }
    if root.get("models").is_some_and(|models| !models.is_array()) {
        return Err("model manifest models must be an array".to_owned());
    }
    if version == MODEL_MANIFEST_SCHEMA_VERSION {
        return Ok(false);
    }
    if let Some(models) = root.get_mut("models").and_then(Value::as_array_mut) {
        for model in models.iter_mut().filter_map(Value::as_object_mut) {
            drop_inert_builtin_memory_keys(model);
        }
    }
    root.insert(
        "schemaVersion".to_owned(),
        Value::from(MODEL_MANIFEST_SCHEMA_VERSION),
    );
    Ok(true)
}

/// Backend memory-evidence keys v2 forbids on a manifest row (APPREG-4: memory data lives in the
/// evidence store). Mirrors the `schemaVersion: 2` rule in `model-manifest.schema.json`.
const V2_FORBIDDEN_BACKEND_MEMORY_KEYS: [&str; 16] = [
    "minMemoryGb",
    "minMemoryGbByTier",
    "memoryStrategyContract",
    "memoryStrategyCapabilities",
    "memoryStrategyStructuralExemptions",
    "calibrations",
    "vramGbByTier",
    "vramMeasuredPixels",
    "supportsSequentialOffload",
    "sequentialPeakGb",
    "measured",
    "imageActivationGbPerMegapixel",
    "turboFit",
    "control",
    "memoryDeclarationWithhold",
    "measuredSiblingActiveDiskFraction",
];
const V2_FORBIDDEN_FOOTPRINT_KEYS: [&str; 3] =
    ["residentMemoryBytes", "peakMemoryBytes", "measuredPixels"];

/// A v1 override of a REGISTERED builtin may carry hand-edited memory evidence. The registry
/// projection never reads it (`ModelSpec::project` applies only author choices, and
/// `project_builtin_memory_evidence` overwrites measured fields for builtin ids), and v2 forbids
/// it, so dropping it changes no behavior while keeping every author choice (name, hidden, ui,
/// limits, defaults) and every other key. Rows that are not registered builtins (imported,
/// trained, custom) are preserved verbatim: the evidence store has no row for their ids, so
/// their own memory keys are the only memory facts the worker admits them with.
fn drop_inert_builtin_memory_keys(model: &mut Map<String, Value>) {
    let registered = model
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| crate::registry::builtin_model_spec(id).is_some());
    if !registered {
        return;
    }
    model.remove("minMemoryGb");
    for backend in ["mlx", "candle"] {
        if let Some(block) = model.get_mut(backend).and_then(Value::as_object_mut) {
            for key in V2_FORBIDDEN_BACKEND_MEMORY_KEYS {
                block.remove(key);
            }
        }
    }
    if let Some(downloads) = model.get_mut("downloads").and_then(Value::as_array_mut) {
        for footprint in downloads
            .iter_mut()
            .filter_map(|download| download.get_mut("footprint"))
            .filter_map(Value::as_object_mut)
        {
            for key in V2_FORBIDDEN_FOOTPRINT_KEYS {
                footprint.remove(key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn old_user_rows_upgrade_without_changing_identity_or_metadata() {
        let mut root = json!({
            "schemaVersion": 1,
            "models": [
                {"id":"imported", "importPlan":{"planId":"p"}, "paths":{"model":"/a"}},
                {"id":"trained", "baseModel":"flux_schnell", "hidden":true}
            ],
            "operatorExtension": {"keep": true}
        });
        let original_models = root["models"].clone();
        assert!(normalize_user_model_manifest(&mut root).unwrap());
        assert_eq!(root["schemaVersion"], 2);
        assert_eq!(root["models"], original_models);
        assert_eq!(root["operatorExtension"], json!({"keep":true}));
        assert!(!normalize_user_model_manifest(&mut root).unwrap());
    }

    #[test]
    fn v1_builtin_override_drops_inert_memory_evidence_but_keeps_author_choices() {
        let memory = json!({
            "minMemoryGb": 24, "vramGbByTier": {"q8": 20}, "calibrations": [{"abi": 3}],
            "supportsSequentialOffload": true, "quantize": 4
        });
        let mut root = json!({
            "schemaVersion": 1,
            "models": [
                {"id": "z_image_turbo", "name": "My Z", "hidden": true,
                 "ui": {"label": "Mine"}, "defaults": {"steps": 6},
                 "minMemoryGb": 30, "mlx": memory.clone(), "candle": memory.clone(),
                 "downloads": [{"repo": "x/y", "footprint": {"diskSizeBytes": 5,
                     "residentMemoryBytes": 4, "peakMemoryBytes": 6, "measuredPixels": 7}}]},
                {"id": "imported", "importPlan": {"planId": "p"}, "minMemoryGb": 30,
                 "mlx": memory.clone(),
                 "downloads": [{"repo": "x/z", "footprint": {"peakMemoryBytes": 6}}]}
            ]
        });
        let imported = root["models"][1].clone();
        assert!(normalize_user_model_manifest(&mut root).unwrap());
        let builtin = &root["models"][0];
        assert_eq!(builtin["name"], "My Z");
        assert_eq!(builtin["hidden"], true);
        assert_eq!(builtin["ui"], json!({"label": "Mine"}));
        assert_eq!(builtin["defaults"], json!({"steps": 6}));
        assert!(builtin.get("minMemoryGb").is_none());
        for backend in ["mlx", "candle"] {
            assert_eq!(builtin[backend], json!({"quantize": 4}), "{backend}");
        }
        assert_eq!(
            builtin["downloads"][0]["footprint"],
            json!({"diskSizeBytes": 5})
        );
        assert_eq!(
            root["models"][1], imported,
            "user-only rows are preserved verbatim"
        );
    }

    #[test]
    fn read_side_parse_refuses_future_versions_and_upgrades_old_ones() {
        let future = "// user\n{\"schemaVersion\": 3, \"models\": [{\"id\": \"later\"}]}";
        assert!(parse_user_model_manifest(future)
            .unwrap_err()
            .contains("unsupported model manifest schemaVersion 3"));
        let old = "{\"models\": [{\"id\": \"legacy\"}]} // trailing";
        let parsed = parse_user_model_manifest(old).unwrap();
        assert_eq!(parsed["schemaVersion"], 2);
        assert_eq!(parsed["models"][0]["id"], "legacy");
    }

    #[test]
    fn future_and_malformed_versions_refuse_without_mutating() {
        for value in [json!(3), json!("2"), json!(0), json!(-1), json!(1.5)] {
            let mut root = json!({"schemaVersion":value,"models":[]});
            let original = root.clone();
            assert!(normalize_user_model_manifest(&mut root).is_err());
            assert_eq!(root, original);
        }
        let mut no_version = json!({"models":[{"id":"legacy"}]});
        assert!(normalize_user_model_manifest(&mut no_version).unwrap());
        assert_eq!(no_version["schemaVersion"], 2);
        let mut invalid_models = json!({"schemaVersion":1,"models":{}});
        assert!(normalize_user_model_manifest(&mut invalid_models).is_err());
    }
}
