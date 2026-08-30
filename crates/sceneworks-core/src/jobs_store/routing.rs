//! Backend routing / gating / catalog logic split out of the `jobs_store` god module
//! (sc-8816). This is a pure code move: the SQLite jobs/workers store and the SQL-coupled
//! dispatch stay in `jobs_store.rs`, while the backend-eligibility predicates, the Mac
//! support/capability probes, the routed-model/kernel catalog, and the gap classifiers live
//! here. No routing decision, catalog membership, or public API changed.

pub(crate) mod candle;
pub(crate) mod catalog;
pub(crate) mod gaps;
pub(crate) mod matrix;
pub(crate) mod mlx;

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::contracts::{ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage};

/// Every built-in SenseNova-U1 variant. Keep routing, understanding, and gap
/// classification on one list so adding an Infographic/distilled tier cannot
/// silently fall through to a generic unsupported reason.
pub(crate) const SENSENOVA_MODEL_IDS: &[&str] = &[
    "sensenova_u1_8b",
    "sensenova_u1_8b_infographic_v2",
    "sensenova_u1_8b_infographic_v3",
    "sensenova_u1_8b_fast",
    "sensenova_u1_8b_infographic_v2_fast",
    "sensenova_u1_8b_infographic_v3_fast",
];

/// Every video mode exposed by the production UI/router catalog.
///
/// The capability dumper consumes this accessor instead of carrying a second mode list, so a newly
/// shipped mode cannot be omitted from matching-platform descriptor evidence.
pub fn video_ui_modes() -> &'static [&'static str] {
    catalog::VIDEO_UI_MODES
}

/// Descriptor conditioning alternatives required by a structurally valid request for `mode`.
/// Each inner slice is one required group; at least one conditioning kind in every group must be
/// present across the route's resolved descriptors.
pub fn video_mode_conditioning_requirements(mode: &str) -> &'static [&'static [&'static str]] {
    match mode {
        "image_to_video" => &[&["reference"]],
        "first_last_frame" => &[&["keyframe"]],
        "video_to_video" | "multi_video_to_video" => &[&["videoClip"]],
        // Subject-reference conditioning is consumable through EITHER descriptor surface
        // (sc-18650): the heterogeneous `multiReference` bundle (bernini, and the scail/edit
        // convention), or the ordered singular `reference` kind — MiniMax-H3's omni-reference
        // surface, where gen-core has no heterogeneous-reference variant and the engines instead
        // declare the ordered vec `[Keyframe, Reference, ReferenceVideo, ReferenceAudio]` (the
        // request's own order carries the semantics). Same in-group alternatives idiom as
        // `replace_person` below.
        "reference_to_video" => &[&["multiReference", "reference"]],
        "reference_video_to_video" | "ads2v" => &[&["videoClip"], &["multiReference"]],
        "replace_person" | "animate_character" => {
            &[&["controlClip"], &["reference", "multiReference"]]
        }
        "extend_clip" | "video_bridge" => &[&["controlClip", "keyframe", "videoClip"]],
        _ => &[],
    }
}

/// Build the canonical, structurally complete production request used to probe one video route.
/// The payload contains the same required asset and provider-adapter seams enforced by the Rust API
/// before enqueue.
pub fn canonical_video_route_probe(model: &str, mode: &str) -> Result<JobSnapshot, String> {
    let (job_type, payload) = match mode {
        "image_to_video" => (
            JobType::VideoGenerate,
            json!({ "model": model, "mode": mode, "sourceAssetId": "probe" }),
        ),
        "first_last_frame" => (
            JobType::VideoGenerate,
            json!({
                "model": model,
                "mode": mode,
                "sourceAssetId": "probe",
                "lastFrameAssetId": "probe-end"
            }),
        ),
        "extend_clip" => (
            JobType::VideoExtend,
            json!({ "model": model, "mode": mode, "sourceClipAssetId": "probe" }),
        ),
        "video_bridge" => (
            JobType::VideoBridge,
            json!({
                "model": model,
                "mode": mode,
                "sourceClipAssetId": "probe",
                "bridgeRightClipAssetId": "probe-right"
            }),
        ),
        "replace_person" => (
            JobType::PersonReplace,
            json!({
                "model": model,
                "mode": mode,
                "sourceClipAssetId": "probe",
                "personTrackId": "probe-person",
                "characterId": "probe-character"
            }),
        ),
        "animate_character" => (
            JobType::VideoGenerate,
            json!({
                "model": model,
                "mode": mode,
                "referenceAssetId": "probe",
                "sourceClipAssetId": "probe-clip"
            }),
        ),
        "video_to_video" => (
            JobType::VideoGenerate,
            json!({ "model": model, "mode": mode, "sourceClipAssetId": "probe" }),
        ),
        "reference_to_video" => (
            JobType::VideoGenerate,
            json!({ "model": model, "mode": mode, "referenceAssetIds": ["probe"] }),
        ),
        "reference_video_to_video" => (
            JobType::VideoGenerate,
            json!({
                "model": model,
                "mode": mode,
                "sourceClipAssetId": "probe",
                "referenceAssetIds": ["probe-reference"]
            }),
        ),
        "multi_video_to_video" => (
            JobType::VideoGenerate,
            json!({
                "model": model,
                "mode": mode,
                "sourceClipAssetIds": ["probe-a", "probe-b"]
            }),
        ),
        "ads2v" => (
            JobType::VideoGenerate,
            json!({
                "model": model,
                "mode": mode,
                "sourceClipAssetId": "probe",
                "referenceClipAssetId": "probe-reference-video",
                "referenceAssetIds": ["probe-reference-image"]
            }),
        ),
        "text_to_video" => (
            JobType::VideoGenerate,
            json!({ "model": model, "mode": mode }),
        ),
        other => return Err(format!("unknown production video mode {other:?}")),
    };
    let mut payload = payload
        .as_object()
        .cloned()
        .expect("canonical video payload is an object");
    // Native LTX clip append/control is structurally available only with its in-context adapter.
    // Capability facts must probe the complete runnable shape; omitting this mandatory input makes
    // both backends falsely report extend/bridge/replacement as unsupported.
    if matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "ltx_2_5")
        && matches!(mode, "extend_clip" | "video_bridge" | "replace_person")
    {
        payload.insert(
            "loras".to_owned(),
            json!([{ "conditioningRole": "ic_lora" }]),
        );
    }
    Ok(JobSnapshot {
        id: "video-route-probe".to_owned(),
        job_type,
        status: JobStatus::Queued,
        project_id: None,
        project_name: None,
        payload,
        result: Map::new(),
        requested_gpu: "auto".to_owned(),
        assigned_gpu: None,
        worker_id: None,
        progress: ContractNumber::from(0_u64),
        stage: ProgressStage::Queued,
        message: String::new(),
        error: None,
        eta_seconds: None,
        elapsed_seconds: None,
        attempts: 0,
        source_job_id: None,
        duplicate_of_job_id: None,
        cancel_requested: false,
        created_at: String::new(),
        updated_at: String::new(),
        started_at: None,
        completed_at: None,
        canceled_at: None,
        last_heartbeat_at: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: None,
        title: None,
        extra: BTreeMap::new(),
    })
}

/// Evaluate one canonical video request through the production backend predicate.
pub fn video_backend_mode_supported(
    backend: &str,
    model: &str,
    mode: &str,
) -> Result<bool, String> {
    let job = canonical_video_route_probe(model, mode)?;
    match backend {
        "mlx" => Ok(mlx::video_job_is_mlx_eligible(&job)),
        "candle" => Ok(candle::video_job_is_candle_eligible(&job)),
        other => Err(format!("unknown native video backend {other:?}")),
    }
}

/// True when a payload key contains a non-blank string.
pub(super) fn has_nonempty_string(payload: &Map<String, Value>, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// True when an optional string carrier is either non-blank or has a malformed non-string shape.
/// Missing, `null`, and a blank string are all the product-level "not supplied" representation.
pub(super) fn has_nonempty_or_malformed_string(payload: &Map<String, Value>, key: &str) -> bool {
    match payload.get(key) {
        None | Some(Value::Null) => false,
        Some(Value::String(value)) => !value.trim().is_empty(),
        Some(_) => true,
    }
}

/// True when an optional string carrier is present in a shape that is neither a string nor `null`.
/// Missing, `null`, and any string (blank included) are well-formed "not supplied or supplied"
/// encodings; a number/bool/array/object is an authored value no route can consume and must fail
/// closed. Use this where BOTH the populated and the absent carrier are eligible on the same gate,
/// so [`has_nonempty_or_malformed_string`] cannot separate malformed from populated (sc-20525).
pub(super) fn has_malformed_optional_string(payload: &Map<String, Value>, key: &str) -> bool {
    !matches!(
        payload.get(key),
        None | Some(Value::Null) | Some(Value::String(_))
    )
}

/// True when a payload key contains a non-empty JSON array.
pub(super) fn has_nonempty_array(payload: &Map<String, Value>, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
}

/// True when an optional array carrier is either non-empty or has a malformed non-array shape.
/// Missing, `null`, and an empty array are all the product-level "not supplied" representation.
/// Use this for unsupported carriers that must fail closed instead of being silently ignored.
pub(super) fn has_nonempty_or_malformed_array(payload: &Map<String, Value>, key: &str) -> bool {
    match payload.get(key) {
        None | Some(Value::Null) => false,
        Some(Value::Array(values)) => !values.is_empty(),
        Some(_) => true,
    }
}

/// True when a payload array contains at least one non-blank string id.
pub(super) fn has_nonempty_string_array(payload: &Map<String, Value>, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .any(|value| !value.trim().is_empty())
        })
}

/// Validate the mutually-exclusive image-reference carriers used by native conditioned-image
/// routes and return the selected reference count. Missing, null, blank strings, and an empty plural
/// array are absent. A malformed plural array, blank/non-string plural member, an oversized plural
/// set, a malformed scalar, or more than one populated carrier fails closed.
pub(super) fn conditioned_reference_count(
    payload: &Map<String, Value>,
    allow_source: bool,
    max_plural: usize,
) -> Option<usize> {
    let plural_count = match payload.get("referenceAssetIds") {
        None | Some(Value::Null) => 0,
        Some(Value::Array(values)) if values.is_empty() => 0,
        Some(Value::Array(values))
            if values.len() <= max_plural
                && values
                    .iter()
                    .all(|value| value.as_str().is_some_and(|id| !id.trim().is_empty())) =>
        {
            values.len()
        }
        Some(_) => return None,
    };
    let singular = match payload.get("referenceAssetId") {
        None | Some(Value::Null) => false,
        Some(Value::String(value)) => !value.trim().is_empty(),
        Some(_) => return None,
    };
    let source = match payload.get("sourceAssetId") {
        None | Some(Value::Null) => false,
        Some(Value::String(value)) => !value.trim().is_empty(),
        Some(_) => return None,
    };
    if source && !allow_source {
        return None;
    }
    let active = usize::from(plural_count > 0) + usize::from(singular) + usize::from(source);
    if active != 1 {
        return None;
    }
    Some(if plural_count > 0 { plural_count } else { 1 })
}

/// True when `payload[object_key][array_key]` is a non-empty array.
pub(super) fn has_nonempty_nested_array(
    payload: &Map<String, Value>,
    object_key: &str,
    array_key: &str,
) -> bool {
    payload
        .get(object_key)
        .and_then(Value::as_object)
        .and_then(|object| object.get(array_key))
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
}

/// Nested equivalent of [`has_nonempty_or_malformed_array`]. A malformed parent object or child
/// carrier fails closed; a missing/null parent and a missing/null/empty child mean "not supplied".
pub(super) fn has_nonempty_or_malformed_nested_array(
    payload: &Map<String, Value>,
    object_key: &str,
    array_key: &str,
) -> bool {
    match payload.get(object_key) {
        None | Some(Value::Null) => false,
        Some(Value::Object(object)) => has_nonempty_or_malformed_array(object, array_key),
        Some(_) => true,
    }
}

/// True when an optional nested carrier is present with any non-null value. A malformed parent
/// object also fails closed. These carriers are emitted only when their feature is active, so an
/// empty string/object or wrong scalar is malformed rather than an alternate representation of
/// "not supplied"; callers must not silently ignore it.
pub(super) fn has_nonnull_or_malformed_nested_carrier(
    payload: &Map<String, Value>,
    object_key: &str,
    carrier_key: &str,
) -> bool {
    match payload.get(object_key) {
        None | Some(Value::Null) => false,
        Some(Value::Object(object)) => object
            .get(carrier_key)
            .is_some_and(|value| !value.is_null()),
        Some(_) => true,
    }
}

/// True when an optional advanced numeric knob is present but is neither a finite JSON number nor a
/// finite numeric string. Missing and null mean "use the model default"; malformed authored values
/// fail closed so a submitted control can never be silently replaced by that default.
pub(super) fn has_malformed_optional_nested_number(
    payload: &Map<String, Value>,
    object_key: &str,
    number_key: &str,
) -> bool {
    match payload.get(object_key) {
        None | Some(Value::Null) => false,
        Some(Value::Object(object)) => match object.get(number_key) {
            None | Some(Value::Null) => false,
            Some(Value::Number(value)) => value.as_f64().is_none_or(|value| !value.is_finite()),
            Some(Value::String(value)) => value
                .trim()
                .parse::<f64>()
                .map_or(true, |value| !value.is_finite()),
            Some(_) => true,
        },
        Some(_) => true,
    }
}

/// Unsupported structural carriers for Krea's reference edit lane. LoRAs, quant selection,
/// guidance, and text-style controls are deliberately allowed; mask/control/pose/phase inputs have
/// no consumer in the edit provider and therefore fail closed.
pub(super) fn krea_edit_has_unsupported_carrier(payload: &Map<String, Value>) -> bool {
    ["controls", "controlnets"]
        .iter()
        .any(|key| has_nonempty_or_malformed_array(payload, key))
        || has_nonempty_or_malformed_string(payload, "maskAssetId")
        || ["poses", "phases"]
            .iter()
            .any(|key| has_nonempty_or_malformed_nested_array(payload, "advanced", key))
        || [
            "controlMode",
            "controlImage",
            "controlScale",
            "controlWeights",
        ]
        .iter()
        .any(|key| has_nonnull_or_malformed_nested_carrier(payload, "advanced", key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_ltx_clip_probes_include_the_required_ic_lora_and_honor_eros_withdrawal() {
        for model in ["ltx_2_3", "ltx_2_3_eros", "ltx_2_5"] {
            for mode in ["extend_clip", "video_bridge", "replace_person"] {
                let job = canonical_video_route_probe(model, mode).unwrap();
                let loras = job.payload["loras"].as_array().unwrap();
                assert!(crate::video_request::loras_contain_ltx_ic_lora(loras));
                assert!(video_backend_mode_supported("mlx", model, mode).unwrap());
                assert_eq!(
                    video_backend_mode_supported("candle", model, mode).unwrap(),
                    model != "ltx_2_3_eros",
                    "Candle must admit supported LTX products and reject withdrawn Eros for {mode}"
                );
            }
            for mode in ["text_to_video", "image_to_video", "first_last_frame"] {
                assert!(canonical_video_route_probe(model, mode)
                    .unwrap()
                    .payload
                    .get("loras")
                    .is_none());
            }
        }
    }

    #[test]
    fn ltx_2_5_candle_admission_rejects_unpromoted_q8_across_base_and_advanced_modes() {
        let modes = [
            "text_to_video",
            "image_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "replace_person",
        ];
        for mode in modes {
            for tier in [json!(4), json!("4"), json!(0), json!("0"), json!(-1)] {
                let mut job = canonical_video_route_probe("ltx_2_5", mode).unwrap();
                job.payload
                    .insert("advanced".to_owned(), json!({ "mlxQuantize": tier }));
                job.payload
                    .entry("loras".to_owned())
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .unwrap()
                    .push(json!({ "id": "style" }));
                assert!(
                    candle::video_job_is_candle_eligible(&job),
                    "LTX-2.5 must preserve q4/bf16 tier {tier} with a user LoRA in {mode}"
                );
            }

            for tier in [json!(8), json!("8")] {
                let mut job = canonical_video_route_probe("ltx_2_5", mode).unwrap();
                job.payload
                    .insert("advanced".to_owned(), json!({ "mlxQuantize": tier }));
                assert!(
                    !candle::video_job_is_candle_eligible(&job),
                    "LTX-2.5 q8 must remain unrouted until inference promotes terminal evidence: {mode} {tier}"
                );
            }
        }

        for tier in [json!(3), json!(16), json!("bf16")] {
            let mut ltx25 = catalog::video_mode_probe_payload("ltx_2_5", "text_to_video");
            ltx25.insert("advanced".to_owned(), json!({ "mlxQuantize": tier }));
            assert!(
                !candle::video_request_candle_eligible("ltx_2_5", &ltx25),
                "LTX-2.5 must fail closed outside q4/bf16: {tier}"
            );

            let mut ltx23 = catalog::video_mode_probe_payload("ltx_2_3", "text_to_video");
            ltx23.insert("advanced".to_owned(), json!({ "mlxQuantize": tier }));
            assert!(
                !candle::video_request_candle_eligible("ltx_2_3", &ltx23),
                "LTX-2.3 must not inherit LTX-2.5's bf16 tier: {tier}"
            );
        }

        for tier in [json!(4), json!(8), json!("8")] {
            let mut ltx23 = catalog::video_mode_probe_payload("ltx_2_3", "text_to_video");
            ltx23.insert("advanced".to_owned(), json!({ "mlxQuantize": tier }));
            assert!(
                candle::video_request_candle_eligible("ltx_2_3", &ltx23),
                "LTX-2.3 packed tier behavior must remain unchanged: {tier}"
            );
        }
    }
}
