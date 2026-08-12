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
        "reference_to_video" => &[&["multiReference"]],
        "reference_video_to_video" | "ads2v" => &[&["videoClip"], &["multiReference"]],
        "replace_person" | "animate_character" => {
            &[&["controlClip"], &["reference", "multiReference"]]
        }
        "extend_clip" | "video_bridge" => &[&["controlClip", "keyframe", "videoClip"]],
        _ => &[],
    }
}

/// Build the canonical, structurally complete production request used to probe one video route.
/// The payload contains the same required asset seams enforced by the Rust API before enqueue.
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
    let payload = payload
        .as_object()
        .cloned()
        .expect("canonical video payload is an object");
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
