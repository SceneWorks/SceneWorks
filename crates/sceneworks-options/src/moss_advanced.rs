//! Pure product transport for MOSS advanced input carriers; artifact authorization stays native.
use crate::contracts::JsonObject;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_length: usize,
    pub sha256: String,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MossImageOutputKind {
    #[default]
    Images,
    Latents,
}
pub const KEY: &str = "mossImageAdvanced";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorRef {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub asset_id: String,
    pub sha256: String,
    /// Measured by the server. Never trusted without re-reading the owned artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TensorMetadata>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EmbeddingsRef {
    pub values: TensorRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<TensorRef>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EditAdvanced<O = MossImageOutputKind> {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positive: Option<EmbeddingsRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative: Option<EmbeddingsRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompts: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<TensorRef>,
    #[serde(default)]
    pub output: O,
}
/// Fold the public typed carrier into the native carrier, preserving the old duplicate refusal.
pub fn normalize_carrier(payload: &mut JsonObject) -> Result<(), String> {
    let Some(value) = payload.remove(KEY).filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let advanced = payload
        .entry("advanced")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .ok_or("advanced must be an object")?;
    if advanced.contains_key(KEY) {
        return Err("supply mossImageAdvanced once, not at both request boundaries".into());
    }
    advanced.insert(KEY.into(), value);
    Ok(())
}
pub fn options(payload: &JsonObject) -> Result<Option<EditAdvanced>, String> {
    let mut canonical = payload.clone();
    normalize_carrier(&mut canonical)?;
    let payload = &canonical;
    let Some(value) = payload.get("advanced").and_then(|v| v.get(KEY)) else {
        return Ok(None);
    };
    if !crate::moss_validation::is_moss(
        payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    ) || !crate::moss_validation::is_edit(payload)
    {
        return Err("mossImageAdvanced is supported only by explicit MOSS edit_image".into());
    }
    let options: EditAdvanced = serde_json::from_value(value.clone())
        .map_err(|e| format!("invalid mossImageAdvanced: {e}"))?;
    if options.positive.is_some()
        && (payload
            .get("prompt")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
            || payload
                .get("prompts")
                .and_then(Value::as_array)
                .is_some_and(|p| !p.is_empty()))
    {
        return Err(
            "positive embeddings replace both prompt and prompts; clear both text carriers".into(),
        );
    }
    if options.negative.is_some() && options.negative_prompts.is_some() {
        return Err("negative embeddings and negativePrompts are mutually exclusive".into());
    }
    if (options.negative.is_some() || options.negative_prompts.is_some())
        && payload
            .get("negativePrompt")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
    {
        return Err("advanced negative inputs replace the broadcast negativePrompt".into());
    }
    if options.output == MossImageOutputKind::Latents
        && payload
            .get("upscale")
            .and_then(|v| v.get("enabled"))
            .and_then(Value::as_bool)
            == Some(true)
    {
        return Err(
            "MOSS latent output cannot use inline upscale; disable upscale or select images output"
                .into(),
        );
    }
    Ok(Some(options))
}
pub fn batch_hint(payload: &JsonObject) -> usize {
    payload
        .get("advanced")
        .and_then(|v| v.get(KEY))
        .or_else(|| payload.get(KEY))
        .and_then(|v| v.get("positive"))
        .and_then(|v| v.get("values"))
        .and_then(|v| v.get("metadata"))
        .and_then(|v| v.get("shape"))
        .and_then(|v| v.get(0))
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(1)
}
