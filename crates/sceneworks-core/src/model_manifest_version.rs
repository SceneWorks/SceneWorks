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
    root.insert(
        "schemaVersion".to_owned(),
        Value::from(MODEL_MANIFEST_SCHEMA_VERSION),
    );
    Ok(true)
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
