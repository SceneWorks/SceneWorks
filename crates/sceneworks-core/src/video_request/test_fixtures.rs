use serde_json::Value;

/// The `models` array of the SHIPPED manifest — the exact bytes the app embeds and seeds.
pub(crate) fn builtin_video_models() -> Vec<Value> {
    let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw)).expect("parses as JSON");
    manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array")
        .iter()
        .filter(|m| m.get("type").and_then(Value::as_str) == Some("video"))
        .cloned()
        .collect()
}
