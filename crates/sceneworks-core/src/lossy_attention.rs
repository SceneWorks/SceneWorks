//! `advanced.lossyAttention` (SW-6132) on the standalone job types that have no typed request:
//! image upscale / Iris restoration, video upscale and dataset upscale.
//!
//! The studio lanes validate the option against the model's catalog capability. A standalone job
//! names an ENGINE instead, so this maps the engine to the candle providers it runs and judges them
//! by the same checked-in projection of inference's `SHARED_ATTENTION_ROUTING`
//! ([`crate::registry::shared_attention_facts`]). The API calls it at enqueue and the worker before
//! it applies the opt-in, so a job is never accepted and then run without the kernel it asked for.
use serde_json::Value;

use crate::contracts::{JobType, JsonObject};
use sceneworks_options::lossy_attention::{
    admission_refusal, payload_lossy_attention, LossyAttentionWire,
};
use sceneworks_options::request_validation::ValidationRefusal;

/// The candle providers a standalone job's engine runs (`None`: the engine has no candle attention
/// the option could reach — Real-ESRGAN is an ONNX convolutional upscaler).
fn standalone_engine_providers(job_type: &JobType, payload: &JsonObject) -> Option<&'static str> {
    let engine = payload
        .get("engine")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("real-esrgan");
    match job_type {
        JobType::ImageUpscale if engine.eq_ignore_ascii_case("seedvr2") => Some("seedvr2"),
        JobType::ImageUpscale if sceneworks_options::image_restoration::is_iris_engine(engine) => {
            Some("iris_3b_restore")
        }
        // Video upscaling is SeedVR2 only.
        JobType::VideoUpscale => Some("seedvr2"),
        _ => None,
    }
}

/// Whether `job_type` is one of the standalone job types judged here.
pub fn is_standalone_job(job_type: &JobType) -> bool {
    matches!(
        job_type,
        JobType::ImageUpscale | JobType::VideoUpscale | JobType::DatasetUpscale
    )
}

/// The admitted `advanced.lossyAttention` of a standalone job (`Ok(None)` when it requests
/// nothing), or the typed refusal: a malformed value, an engine without candle attention, or a
/// kernel the engine's attention can never take.
pub fn standalone_lossy_attention(
    job_type: &JobType,
    payload: &JsonObject,
) -> Result<Option<LossyAttentionWire>, ValidationRefusal> {
    let Some(wire) = payload_lossy_attention(payload)? else {
        return Ok(None);
    };
    let engine = payload
        .get("engine")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|engine| !engine.is_empty())
        .unwrap_or(match job_type {
            JobType::VideoUpscale => "seedvr2",
            _ => "real-esrgan",
        });
    let capability = standalone_engine_providers(job_type, payload)
        .and_then(|provider| crate::registry::shared_attention_facts().capability_for([provider]));
    match admission_refusal(engine, &wire, capability.as_ref()) {
        Some(refusal) => Err(refusal),
        None => Ok(Some(wire)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(value: Value) -> JsonObject {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn standalone_engines_take_the_option_only_where_their_attention_serves_it() {
        let int8 = json!({ "int8": {} });
        for (job_type, engine) in [
            (JobType::ImageUpscale, Some("seedvr2")),
            (JobType::ImageUpscale, Some("iris")),
            (JobType::VideoUpscale, None),
        ] {
            let mut body = json!({ "sourceAssetId": "a", "advanced": { "lossyAttention": int8 } });
            if let Some(engine) = engine {
                body["engine"] = json!(engine);
            }
            let admitted = standalone_lossy_attention(&job_type, &payload(body)).unwrap();
            assert!(admitted.unwrap().int8.is_some(), "{job_type:?} {engine:?}");
        }
        // Real-ESRGAN (the default image and dataset engine) has no candle attention.
        for (job_type, body) in [
            (
                JobType::ImageUpscale,
                json!({ "advanced": { "lossyAttention": int8 } }),
            ),
            (
                JobType::ImageUpscale,
                json!({ "engine": "real-esrgan", "advanced": { "lossyAttention": int8 } }),
            ),
            (
                JobType::DatasetUpscale,
                json!({ "advanced": { "lossyAttention": int8 } }),
            ),
        ] {
            let refusal = standalone_lossy_attention(&job_type, &payload(body)).unwrap_err();
            assert_eq!(refusal.field, "advanced.lossyAttention", "{job_type:?}");
        }
        // Absent, null or empty requests nothing, whatever the engine.
        for body in [
            json!({}),
            json!({ "advanced": { "lossyAttention": null } }),
            json!({ "advanced": { "lossyAttention": {} } }),
        ] {
            assert!(
                standalone_lossy_attention(&JobType::ImageUpscale, &payload(body))
                    .unwrap()
                    .is_none()
            );
        }
        // A malformed value is refused by its field.
        let bad = json!({ "engine": "seedvr2", "advanced": { "lossyAttention": { "sol": {} } } });
        assert!(standalone_lossy_attention(&JobType::ImageUpscale, &payload(bad)).is_err());
    }

    /// Codex P2 on #3528: the shared validation admits a video upscale by its engine; it no longer
    /// demands a generation-model capability the job has no entry for.
    #[test]
    fn shared_validation_admits_video_upscale_by_engine() {
        let empty = JsonObject::new();
        let admitted = payload(json!({ "sourceAssetId": "clip", "factor": 2,
                                       "advanced": { "lossyAttention": { "int8": {} } } }));
        crate::request_validation::validate("videoUpscale", &admitted, &empty).unwrap();
        crate::request_validation::resolve("videoUpscale", &admitted, &empty).unwrap();
        let malformed = payload(json!({ "sourceAssetId": "clip", "factor": 2,
                                        "advanced": { "lossyAttention": { "sol": {} } } }));
        assert!(crate::request_validation::validate("videoUpscale", &malformed, &empty).is_err());
        // A generation kind still requires the model capability (an empty entry has none).
        let image = payload(json!({ "projectId": "p", "model": "m", "prompt": "x",
                                    "advanced": { "lossyAttention": { "int8": {} } } }));
        let refusal = crate::request_validation::validate("image", &image, &empty).unwrap_err();
        assert_eq!(refusal.field, "advanced.lossyAttention");
    }
}
