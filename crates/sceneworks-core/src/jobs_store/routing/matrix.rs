//! Generated cross-backend capability inventory (sc-18473).
//!
//! This module deliberately evaluates the routing predicates instead of copying their answers.
//! The checked-in JSON is a reviewable build artifact; its unit test rebuilds it from the embedded
//! manifests, router, descriptor-derived preview facts, and exception register. Source digests also
//! cover the API, worker dispatch, and web gates so a change at any contract boundary requires an
//! intentional regeneration.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::contracts::{ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage};
use crate::jsonc::strip_jsonc_comments;

use super::candle::{
    image_job_is_candle_eligible, training_job_is_candle_eligible, video_job_is_candle_eligible,
};
use super::catalog::{CANDLE_ROUTED_TRAINING_KERNELS, MLX_ROUTED_TRAINING_KERNELS, VIDEO_UI_MODES};
use super::mlx::{
    image_job_is_mlx_eligible, training_job_is_mlx_eligible, video_job_is_mlx_eligible,
};

const MANIFEST: &str = include_str!("../../../../../config/manifests/builtin.models.jsonc");
const PREVIEW: &str = include_str!("../../../../../config/manifests/builtin.preview-support.jsonc");
const EXCEPTIONS: &str = include_str!("../../../../../config/backend-capabilities/exceptions.json");
const MLX_DESCRIPTOR_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/capabilities.mlx.json");
const CANDLE_DESCRIPTOR_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/capabilities.candle.json");
const AUDIO_DESCRIPTOR_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/audio/capabilities.candle.json");

const ROUTING_CATALOG: &str = include_str!("catalog.rs");
const ROUTING_MLX: &str = include_str!("mlx.rs");
const ROUTING_CANDLE: &str = include_str!("candle.rs");
const ROUTING_GAPS: &str = include_str!("gaps.rs");
const API_VALIDATION: &str = include_str!("../../../../../apps/rust-api/src/jobs.rs");
const WORKER_DISPATCH: &str = include_str!("../../../../sceneworks-worker/src/lib.rs");
const WORKER_IMAGE_DISPATCH: &str =
    include_str!("../../../../sceneworks-worker/src/image_jobs/base.rs");
const WORKER_ENGINE_TABLE: &str = include_str!("../../../../sceneworks-worker/src/engines.rs");
const WEB_IMAGE_REQUEST: &str = include_str!("../../../../../apps/web/src/imageJobRequest.js");
const WEB_MAC_GATING: &str = include_str!("../../../../../apps/web/src/macGating.js");
const WEB_PREVIEW_GATING: &str = include_str!("../../../../../apps/web/src/previewSupport.js");

const GENERATOR: &str = "cargo run -p sceneworks-core --bin dump-backend-capability-matrix";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendCapabilityMatrix {
    pub schema_version: u32,
    pub generated_by: String,
    pub sources: BTreeMap<String, String>,
    pub models: Vec<ModelCapabilityRow>,
    pub gpu_job_types: Vec<JobCapabilityRow>,
    pub training_kernels: Vec<TrainingCapabilityRow>,
    pub exceptions: Vec<ExceptionRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapabilityRow {
    pub id: String,
    pub family: String,
    pub model_type: String,
    pub manifest_operations: Vec<String>,
    pub operation_and_mode: Vec<CapabilityCell>,
    pub conditioning_shape: Vec<CapabilityCell>,
    pub user_adapters: Vec<CapabilityCell>,
    pub precision_tier: Vec<CapabilityCell>,
    pub preview: CapabilityCell,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobCapabilityRow {
    pub job_type: String,
    pub category: String,
    pub support: CapabilityCell,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCapabilityRow {
    pub kernel: String,
    pub network_type: String,
    pub support: CapabilityCell,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityCell {
    pub capability: String,
    pub mlx: Option<bool>,
    pub candle: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity_obligation: Option<ParityObligation>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub preserved_candle_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityObligation {
    pub work_item: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authority: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExceptionRecord {
    pub id: String,
    pub category: String,
    pub approver: String,
    pub approved_date: String,
    pub user_facing_behavior: String,
    pub revisit_condition: String,
    pub cells: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExceptionRegister {
    schema_version: u32,
    records: Vec<ExceptionRecord>,
}

#[derive(Debug, Deserialize)]
struct ManifestRoot {
    models: Vec<ManifestModel>,
}

#[derive(Debug, Deserialize)]
struct ManifestModel {
    id: String,
    #[serde(default)]
    family: String,
    #[serde(rename = "type", default)]
    model_type: String,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    downloads: Vec<ManifestDownload>,
    #[serde(default)]
    ui: Value,
    #[serde(rename = "loraCompatibility", default)]
    lora_compatibility: Value,
    #[serde(default)]
    mlx: Value,
}

#[derive(Debug, Deserialize)]
struct ManifestDownload {
    #[serde(default)]
    variant: String,
}

#[derive(Debug, Deserialize)]
struct PreviewRoot {
    models: BTreeMap<String, BTreeMap<String, bool>>,
}

/// Build the complete, deterministic matrix without loading model weights.
pub fn backend_capability_matrix() -> Result<BackendCapabilityMatrix, String> {
    let manifest: ManifestRoot = serde_json::from_str(&strip_jsonc_comments(MANIFEST))
        .map_err(|error| format!("parse builtin.models.jsonc: {error}"))?;
    let preview: PreviewRoot = serde_json::from_str(&strip_jsonc_comments(PREVIEW))
        .map_err(|error| format!("parse builtin.preview-support.jsonc: {error}"))?;
    let exceptions: ExceptionRegister = serde_json::from_str(EXCEPTIONS)
        .map_err(|error| format!("parse backend capability exception register: {error}"))?;
    if exceptions.schema_version != 1 {
        return Err(format!(
            "unsupported backend capability exception schema {}",
            exceptions.schema_version
        ));
    }
    validate_exceptions(&exceptions.records)?;

    let mut models = Vec::with_capacity(manifest.models.len());
    for model in &manifest.models {
        models.push(model_row(model, preview.models.get(&model.id))?);
    }
    models.sort_by(|left, right| left.id.cmp(&right.id));

    let gpu_job_types = gpu_job_rows();
    let training_kernels = training_rows();
    validate_obligations(
        &models,
        &gpu_job_types,
        &training_kernels,
        &exceptions.records,
    )?;

    Ok(BackendCapabilityMatrix {
        schema_version: 1,
        generated_by: GENERATOR.to_owned(),
        sources: source_digests(),
        models,
        gpu_job_types,
        training_kernels,
        exceptions: exceptions.records,
    })
}

fn model_row(
    model: &ManifestModel,
    preview: Option<&BTreeMap<String, bool>>,
) -> Result<ModelCapabilityRow, String> {
    let mut manifest_operations = model.capabilities.clone();
    if model.ui.get("img2img").and_then(Value::as_bool) == Some(true) {
        manifest_operations.push("img2img".to_owned());
    }
    manifest_operations.sort();
    manifest_operations.dedup();

    let is_image = model.model_type == "image";
    let is_video = model.model_type == "video";
    let mut operation_and_mode = Vec::new();
    let mut conditioning_shape = Vec::new();
    let mut user_adapters = Vec::new();
    let mut precision_tier = Vec::new();

    if is_image {
        operation_and_mode.push(image_cell(model, "text_to_image", json!({}), "operation")?);
        operation_and_mode.push(image_cell(
            model,
            "edit_image",
            json!({ "mode": "edit_image", "sourceAssetId": "probe" }),
            "operation",
        )?);
        operation_and_mode.push(image_cell(
            model,
            "character_image",
            json!({ "mode": "character_image", "referenceAssetId": "probe" }),
            "operation",
        )?);
        if model.ui.get("promptEnhance").and_then(Value::as_bool) == Some(true) {
            operation_and_mode.push(cell(
                "prompt_enhancement".to_owned(),
                true,
                false,
                gap_for(&model.id, "operation", "prompt_enhancement"),
            ));
        }
        conditioning_shape.push(image_cell(
            model,
            "img2img",
            json!({ "mode": "text_to_image", "referenceAssetId": "probe" }),
            "conditioning",
        )?);
        conditioning_shape.push(image_cell(
            model,
            "masked_edit",
            json!({
                "mode": "edit_image",
                "sourceAssetId": "probe",
                "maskAssetId": "probe-mask"
            }),
            "conditioning",
        )?);
        conditioning_shape.push(image_cell(
            model,
            "strict_pose",
            json!({ "advanced": { "poses": [{}] } }),
            "conditioning",
        )?);
        user_adapters.push(image_cell(
            model,
            "lora",
            json!({ "loras": [{ "id": "probe", "networkType": "lora" }] }),
            "adapter",
        )?);
        user_adapters.push(image_cell(
            model,
            "lokr",
            json!({ "loras": [{ "id": "probe", "networkType": "lokr" }] }),
            "adapter",
        )?);
        if model.lora_compatibility.is_null() {
            for adapter in &mut user_adapters {
                adapter.mlx = Some(false);
                adapter.candle = Some(false);
                adapter.parity_obligation = None;
                adapter.preserved_candle_only = false;
            }
        }
        for (tier, bits) in [("q4", 4), ("q8", 8)] {
            let mut tier_cell = image_cell(
                model,
                tier,
                json!({ "advanced": { "mlxQuantize": bits } }),
                "precision",
            )?;
            let manifest_has_tier = model
                .downloads
                .iter()
                .any(|download| download.variant == tier)
                || model
                    .mlx
                    .get("quantize")
                    .and_then(Value::as_u64)
                    .is_some_and(|quantize| quantize == bits);
            if !manifest_has_tier {
                tier_cell.mlx = Some(false);
                tier_cell.candle = Some(false);
                tier_cell.parity_obligation = None;
            }
            precision_tier.push(tier_cell);
        }
        precision_tier.push(image_cell(model, "bf16", json!({}), "precision")?);
    } else if is_video {
        for mode in VIDEO_UI_MODES {
            let payload = video_payload(mode);
            let job_type = video_job_type(mode);
            let job = probe_job(job_type, &model.id, payload)?;
            operation_and_mode.push(cell(
                (*mode).to_owned(),
                video_job_is_mlx_eligible(&job),
                video_job_is_candle_eligible(&job),
                gap_for(&model.id, "video", mode),
            ));
        }
        for network_type in ["lora", "lokr"] {
            let job = probe_job(
                JobType::VideoGenerate,
                &model.id,
                json!({
                    "mode": "text_to_video",
                    "loras": [{ "id": "probe", "networkType": network_type }]
                }),
            )?;
            user_adapters.push(cell(
                network_type.to_owned(),
                video_job_is_mlx_eligible(&job),
                video_job_is_candle_eligible(&job),
                gap_for(&model.id, "video-adapter", network_type),
            ));
        }
        let dense = probe_job(
            JobType::VideoGenerate,
            &model.id,
            json!({ "mode": "text_to_video" }),
        )?;
        precision_tier.push(cell(
            "bf16".to_owned(),
            video_job_is_mlx_eligible(&dense),
            video_job_is_candle_eligible(&dense),
            gap_for(&model.id, "video", "bf16"),
        ));
        let quant = probe_job(
            JobType::VideoGenerate,
            &model.id,
            json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 4 } }),
        )?;
        precision_tier.push(cell(
            "q4".to_owned(),
            video_job_is_mlx_eligible(&quant),
            video_job_is_candle_eligible(&quant),
            gap_for(&model.id, "video", "q4"),
        ));
    }

    // Utility/audio descriptors are shipped models too. Their model-specific operation is recorded
    // from the manifest; backend execution is represented in the exhaustive GPU-job section below.
    if !is_image && !is_video {
        for operation in &manifest_operations {
            operation_and_mode.push(CapabilityCell {
                capability: operation.clone(),
                mlx: None,
                candle: None,
                parity_obligation: None,
                preserved_candle_only: false,
            });
        }
    }

    let preview = CapabilityCell {
        capability: "live_preview".to_owned(),
        mlx: preview.and_then(|backends| backends.get("mlx")).copied(),
        candle: preview.and_then(|backends| backends.get("candle")).copied(),
        parity_obligation: match (
            preview.and_then(|backends| backends.get("mlx")).copied(),
            preview.and_then(|backends| backends.get("candle")).copied(),
        ) {
            (Some(true), Some(false) | None) => Some(gap_for(&model.id, "preview", "live_preview")),
            _ => None,
        },
        preserved_candle_only: matches!(
            (
                preview.and_then(|backends| backends.get("mlx")).copied(),
                preview.and_then(|backends| backends.get("candle")).copied()
            ),
            (Some(false) | None, Some(true))
        ),
    };

    Ok(ModelCapabilityRow {
        id: model.id.clone(),
        family: model.family.clone(),
        model_type: model.model_type.clone(),
        manifest_operations,
        operation_and_mode,
        conditioning_shape,
        user_adapters,
        precision_tier,
        preview,
    })
}

fn image_cell(
    model: &ManifestModel,
    name: &str,
    shape: Value,
    category: &str,
) -> Result<CapabilityCell, String> {
    let job_type = if name == "edit_image" || name == "masked_edit" {
        JobType::ImageEdit
    } else {
        JobType::ImageGenerate
    };
    let job = probe_job(job_type, &model.id, shape)?;
    Ok(cell(
        name.to_owned(),
        image_job_is_mlx_eligible(&job),
        image_job_is_candle_eligible(&job),
        gap_for(&model.id, category, name),
    ))
}

fn cell(
    capability: String,
    mlx: bool,
    candle: bool,
    obligation: ParityObligation,
) -> CapabilityCell {
    CapabilityCell {
        capability,
        mlx: Some(mlx),
        candle: Some(candle),
        parity_obligation: (mlx && !candle).then_some(obligation),
        preserved_candle_only: candle && !mlx,
    }
}

fn gap_for(model: &str, category: &str, capability: &str) -> ParityObligation {
    let (work_item, authority) = match category {
        "adapter" => ("sc-18477", None),
        "video" | "video-adapter" => {
            let authority = (model == "krea_realtime_14b").then_some("epic-8433");
            ("sc-18478", authority)
        }
        "preview" => {
            let story = if model.starts_with("wan_")
                || matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "svd" | "bernini")
            {
                "sc-18478"
            } else {
                "sc-18476"
            };
            (story, Some("epic-16948"))
        }
        "precision" => ("sc-18476", Some("epic-9083")),
        "conditioning" if matches!(model, "sana_1600m" | "sana_sprint_1600m") => {
            ("sc-18475", Some("epic-8588"))
        }
        "conditioning" => ("sc-18476", Some("epic-8588")),
        "operation" if model == "flux2_dev" && capability == "prompt_enhancement" => {
            ("sc-18474", None)
        }
        "operation" => ("sc-18476", None),
        "training" => ("sc-18479", None),
        "utility" => ("sc-18480", None),
        _ => ("sc-18480", None),
    };
    ParityObligation {
        work_item: work_item.to_owned(),
        url: shortcut_url(work_item),
        authority: authority.map(str::to_owned),
    }
}

fn shortcut_url(work_item: &str) -> String {
    if let Some(id) = work_item.strip_prefix("sc-") {
        format!("https://app.shortcut.com/trefry/story/{id}")
    } else if let Some(id) = work_item.strip_prefix("epic-") {
        format!("https://app.shortcut.com/trefry/epic/{id}")
    } else {
        String::new()
    }
}

fn video_payload(mode: &str) -> Value {
    match mode {
        "image_to_video" => json!({ "mode": mode, "sourceAssetId": "probe" }),
        "first_last_frame" => json!({
            "mode": mode,
            "sourceAssetId": "probe",
            "endAssetId": "probe-end"
        }),
        "extend_clip" => json!({ "mode": mode, "sourceClipAssetId": "probe" }),
        "video_bridge" => json!({
            "mode": mode,
            "sourceClipAssetId": "probe",
            "bridgeRightClipAssetId": "probe-right"
        }),
        "replace_person" => json!({
            "mode": mode,
            "sourceClipAssetId": "probe",
            "personTrackId": "probe-person",
            "characterId": "probe-character"
        }),
        "animate_character" => json!({
            "mode": mode,
            "referenceAssetId": "probe",
            "sourceClipAssetId": "probe-clip"
        }),
        _ => json!({ "mode": mode }),
    }
}

fn video_job_type(mode: &str) -> JobType {
    match mode {
        "extend_clip" => JobType::VideoExtend,
        "video_bridge" => JobType::VideoBridge,
        "replace_person" => JobType::PersonReplace,
        _ => JobType::VideoGenerate,
    }
}

fn probe_job(job_type: JobType, model: &str, payload: Value) -> Result<JobSnapshot, String> {
    let mut payload = payload
        .as_object()
        .cloned()
        .ok_or_else(|| "probe payload must be an object".to_owned())?;
    payload.insert("model".to_owned(), Value::String(model.to_owned()));
    Ok(JobSnapshot {
        id: "capability-matrix-probe".to_owned(),
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

fn gpu_job_rows() -> Vec<JobCapabilityRow> {
    // Complete JobType set minus jobs_store::NON_GPU_JOB_TYPES. The source-level architecture test
    // below mechanically proves that this list stays complete when JobType grows.
    let rows = [
        ("placeholder", "utility", false, false),
        ("image_generate", "per-model", true, true),
        ("image_edit", "per-model", true, true),
        ("image_vqa", "per-model", true, true),
        ("image_interleave", "per-model", true, true),
        ("video_generate", "per-model", true, true),
        ("video_extend", "per-model", true, true),
        ("video_bridge", "per-model", true, true),
        ("person_detect", "utility", true, true),
        ("person_track", "utility", true, true),
        ("person_replace", "per-model", true, true),
        ("audio_generate", "per-model", false, true),
        ("pose_detect", "utility", true, true),
        ("kps_extract", "utility", true, true),
        ("image_upscale", "utility", true, true),
        ("image_detail", "per-model", true, false),
        ("image_segment", "utility", true, false),
        ("video_upscale", "utility", true, true),
        ("frame_extract", "utility", true, true),
        ("timeline_export", "utility", true, true),
        ("lora_train", "per-kernel", true, true),
        ("control_training", "per-kernel", false, true),
        ("training_caption", "utility", true, true),
        ("dataset_analysis", "utility", true, false),
        ("catalog_analysis", "utility", true, true),
        ("dataset_upscale", "utility", true, true),
        ("dataset_face_analysis", "utility", true, true),
        ("face_likeness_compare", "utility", true, true),
        ("prompt_refine", "utility", true, true),
    ];
    rows.into_iter()
        .map(|(job_type, category, mlx, candle)| JobCapabilityRow {
            job_type: job_type.to_owned(),
            category: category.to_owned(),
            support: cell(
                "representative_request".to_owned(),
                mlx,
                candle,
                gap_for("", "utility", job_type),
            ),
        })
        .collect()
}

fn training_rows() -> Vec<TrainingCapabilityRow> {
    let mut kernels: BTreeSet<&str> = MLX_ROUTED_TRAINING_KERNELS.iter().copied().collect();
    kernels.extend(CANDLE_ROUTED_TRAINING_KERNELS.iter().copied());
    kernels.insert("wan_moe_lora");
    let mut rows = Vec::new();
    for kernel in kernels {
        for network_type in ["lora", "lokr"] {
            let base_model = if kernel == "wan_moe_lora" {
                "wan_2_2_t2v_14b"
            } else {
                "probe"
            };
            let payload = json!({
                "plan": {
                    "target": { "kernel": kernel, "baseModel": base_model },
                    "config": { "advanced": { "networkType": network_type } }
                }
            });
            let mlx = probe_job(JobType::LoraTrain, "", payload.clone())
                .is_ok_and(|job| training_job_is_mlx_eligible(&job));
            let candle_job_type = if kernel == "krea_control" {
                JobType::ControlTraining
            } else {
                JobType::LoraTrain
            };
            let candle = probe_job(candle_job_type, "", payload)
                .is_ok_and(|job| training_job_is_candle_eligible(&job));
            rows.push(TrainingCapabilityRow {
                kernel: kernel.to_owned(),
                network_type: network_type.to_owned(),
                support: cell(
                    "train".to_owned(),
                    mlx,
                    candle,
                    gap_for("", "training", kernel),
                ),
            });
        }
    }
    rows
}

fn validate_exceptions(records: &[ExceptionRecord]) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for record in records {
        if !ids.insert(record.id.as_str()) {
            return Err(format!("duplicate exception id {:?}", record.id));
        }
        for (field, value) in [
            ("id", record.id.as_str()),
            ("category", record.category.as_str()),
            ("approver", record.approver.as_str()),
            ("approvedDate", record.approved_date.as_str()),
            ("userFacingBehavior", record.user_facing_behavior.as_str()),
            ("revisitCondition", record.revisit_condition.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("exception {:?} has empty {field}", record.id));
            }
        }
        if record.cells.is_empty() {
            return Err(format!("exception {:?} names no matrix cells", record.id));
        }
    }
    Ok(())
}

fn validate_obligations(
    models: &[ModelCapabilityRow],
    jobs: &[JobCapabilityRow],
    training: &[TrainingCapabilityRow],
    exceptions: &[ExceptionRecord],
) -> Result<(), String> {
    let exception_cells: BTreeSet<&str> = exceptions
        .iter()
        .flat_map(|record| record.cells.iter().map(String::as_str))
        .collect();
    let mut missing = Vec::new();
    let mut check = |path: String, cell: &CapabilityCell| {
        if cell.mlx == Some(true)
            && cell.candle != Some(true)
            && cell.parity_obligation.is_none()
            && !exception_cells.contains(path.as_str())
        {
            missing.push(path.clone());
        }
        if cell.candle == Some(true) && cell.mlx != Some(true) && !cell.preserved_candle_only {
            missing.push(format!("{path} (Candle-only capability not preserved)"));
        }
    };
    for model in models {
        for (axis, cells) in [
            ("operationAndMode", model.operation_and_mode.as_slice()),
            ("conditioningShape", model.conditioning_shape.as_slice()),
            ("userAdapters", model.user_adapters.as_slice()),
            ("precisionTier", model.precision_tier.as_slice()),
        ] {
            for cell in cells {
                check(
                    format!("models/{}/{axis}/{}", model.id, cell.capability),
                    cell,
                );
            }
        }
        check(format!("models/{}/preview", model.id), &model.preview);
    }
    for job in jobs {
        check(format!("gpuJobTypes/{}", job.job_type), &job.support);
    }
    for row in training {
        check(
            format!("trainingKernels/{}/{}", row.kernel, row.network_type),
            &row.support,
        );
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "backend capability matrix contains untracked parity cells: {missing:?}"
        ))
    }
}

fn source_digests() -> BTreeMap<String, String> {
    [
        ("manifest", MANIFEST),
        ("routerCatalog", ROUTING_CATALOG),
        ("routerMlx", ROUTING_MLX),
        ("routerCandle", ROUTING_CANDLE),
        ("routerGaps", ROUTING_GAPS),
        ("apiValidation", API_VALIDATION),
        ("workerDispatch", WORKER_DISPATCH),
        ("workerImageDispatch", WORKER_IMAGE_DISPATCH),
        ("workerEngineTable", WORKER_ENGINE_TABLE),
        ("descriptorMlxFacts", MLX_DESCRIPTOR_FACTS),
        ("descriptorCandleFacts", CANDLE_DESCRIPTOR_FACTS),
        ("descriptorAudioFacts", AUDIO_DESCRIPTOR_FACTS),
        ("descriptorPreviewFacts", PREVIEW),
        ("webImageRequest", WEB_IMAGE_REQUEST),
        ("webMacGating", WEB_MAC_GATING),
        ("webPreviewGating", WEB_PREVIEW_GATING),
        ("exceptionRegister", EXCEPTIONS),
    ]
    .into_iter()
    .map(|(name, source)| (name.to_owned(), sha256(source)))
    .collect()
}

fn sha256(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKED_IN: &str = include_str!("../../../../../config/backend-capabilities/matrix.json");

    #[test]
    fn checked_in_matrix_matches_all_authoritative_sources() {
        let expected: BackendCapabilityMatrix =
            serde_json::from_str(CHECKED_IN).expect("checked-in capability matrix parses");
        let actual = backend_capability_matrix().expect("capability matrix generates");
        assert_eq!(
            expected, actual,
            "backend capability matrix drifted; run `{GENERATOR} > config/backend-capabilities/matrix.json`"
        );
    }

    #[test]
    fn every_gpu_job_type_is_in_the_matrix() {
        let known = known_job_types();
        let non_gpu: BTreeSet<&str> = crate::jobs_store::NON_GPU_JOB_TYPES
            .iter()
            .copied()
            .collect();
        let expected: BTreeSet<_> = known
            .iter()
            .map(String::as_str)
            .filter(|name| !non_gpu.contains(name))
            .collect();
        let actual: BTreeSet<_> = gpu_job_rows().into_iter().map(|row| row.job_type).collect();
        let expected: BTreeSet<_> = expected.into_iter().map(str::to_owned).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn mutation_guards_reject_untracked_gap_and_incomplete_exception() {
        let mut broken = cell(
            "mutated".to_owned(),
            true,
            false,
            gap_for("", "utility", "mutated"),
        );
        broken.parity_obligation = None;
        let jobs = [JobCapabilityRow {
            job_type: "mutated".to_owned(),
            category: "utility".to_owned(),
            support: broken,
        }];
        assert!(validate_obligations(&[], &jobs, &[], &[]).is_err());

        let incomplete = ExceptionRecord {
            id: "ex-1".to_owned(),
            category: "".to_owned(),
            approver: "owner".to_owned(),
            approved_date: "2026-08-10".to_owned(),
            user_facing_behavior: "disabled".to_owned(),
            revisit_condition: "implementation lands".to_owned(),
            cells: vec!["gpuJobTypes/mutated".to_owned()],
        };
        assert!(validate_exceptions(&[incomplete]).is_err());
    }

    fn known_job_types() -> BTreeSet<String> {
        let body = include_str!("../../contracts.rs")
            .split_once("pub enum JobType {")
            .expect("contracts.rs declares JobType")
            .1
            .split_once("\n    }")
            .expect("JobType closes")
            .0;
        body.lines()
            .filter_map(|line| {
                let (_, wire) = line.split_once("=>")?;
                Some(wire.trim().strip_prefix('"')?.split('"').next()?.to_owned())
            })
            .collect()
    }
}
