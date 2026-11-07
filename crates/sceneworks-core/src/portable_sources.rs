//! Pure selected auxiliary source IDs shared by API admission and worker guard.

use crate::contracts::{JobType, JsonObject};
use serde_json::Value;

pub fn required_portable_aux_source_ids(
    job_type: &JobType,
    payload: &JsonObject,
    entry: &Value,
    backend: &str,
) -> Result<Vec<String>, String> {
    let id = entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or("model entry lacks id")?;
    let primary_model = payload.get("model").and_then(Value::as_str) == Some(id);
    let advanced = payload.get("advanced").and_then(Value::as_object);
    let mut ids = Vec::new();
    let has_ref = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let character_reference =
        payload.get("mode").and_then(Value::as_str) == Some("character_image") && has_ref;
    if primary_model && matches!(job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        if character_reference && matches!(id, "flux_dev" | "flux_schnell") {
            ids.extend(["flux_ip_adapter", "flux_ip_encoder"].map(str::to_owned));
        }
        if character_reference
            && matches!(
                id,
                "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2"
            )
        {
            ids.extend(["sdxl_ip_adapter", "sdxl_ip_encoder"].map(str::to_owned));
        }
        if let Some(decoder) = advanced
            .and_then(|value| value.get("decoder"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty() && *value != "native")
        {
            let component = entry
                .pointer(&format!("/decoders/byBackend/{backend}"))
                .and_then(Value::as_array)
                .and_then(|options| {
                    options
                        .iter()
                        .find(|option| option.get("id").and_then(Value::as_str) == Some(decoder))
                })
                .and_then(|option| option.get("componentId"))
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{id} cannot use decoder {decoder} on {backend}"))?;
            ids.push(component.to_owned());
        }
        if matches!(id, "flux3_klein" | "flux3_klein_base")
            && advanced
                .and_then(|value| value.get("enhancePrompt"))
                .and_then(Value::as_bool)
                == Some(true)
        {
            ids.push("prompt_upsampler".to_owned());
        }
        if advanced
            .and_then(|value| value.get("poses"))
            .and_then(Value::as_array)
            .is_some_and(|poses| !poses.is_empty())
        {
            ids.push("strict_control".to_owned());
            if advanced
                .and_then(|value| value.get("controlMode"))
                .and_then(Value::as_str)
                == Some("depth")
            {
                ids.push("depth_anything_v2_small".to_owned());
            }
        }
    }
    if primary_model && *job_type == JobType::ImageDetail {
        ids.push("sdxl_detail_tile_control".to_owned());
    }
    if payload.get("editMode").and_then(Value::as_str) == Some("cover") && id.starts_with("acestep")
    {
        ids.push("sft_cover".to_owned());
    }
    if matches!(id, "minimax_h3" | "minimax_h3_ref")
        && advanced
            .and_then(|value| value.get("veda"))
            .and_then(|value| value.get("enabled"))
            .and_then(Value::as_bool)
            == Some(true)
    {
        ids.push("veda_predictor".to_owned());
    }
    if id.starts_with("ltx")
        && backend == "mlx"
        && advanced.is_some_and(|value| {
            value.get("useUncensoredEnhancer").and_then(Value::as_bool) == Some(true)
                || value.get("textEncoderModel").and_then(Value::as_str)
                    == Some("ltx_amoral_gemma_3_12b")
        })
    {
        ids.push("ltx_amoral_gemma_3_12b".to_owned());
    }
    if matches!(job_type, JobType::LoraTrain | JobType::ControlTraining) {
        let plan: crate::training::TrainingPlan = serde_json::from_value(
            payload
                .get("plan")
                .cloned()
                .ok_or("training job lacks resolved plan")?,
        )
        .map_err(|error| format!("invalid resolved training plan: {error}"))?;
        if plan.target.base_model == id {
            if id == "ltx_2_3" {
                ids.push("ltx_training_text_encoder".to_owned());
            }
            if crate::training::iris::is_iris_training_kernel(&plan.target.kernel) {
                let options = crate::training::iris::iris_options(&plan.config)
                    .map_err(|error| error.to_string())?;
                if crate::training::iris::curricula::iris_repa_enabled(&options) {
                    ids.push("repa_teacher".to_owned());
                }
            }
        }
    }
    ids.sort();
    ids.dedup();
    Ok(ids)
}
