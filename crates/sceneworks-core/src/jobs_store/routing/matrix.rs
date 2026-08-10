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

use crate::contracts::{
    ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage, WorkerCapability,
    WorkerSnapshot, WorkerStatus,
};
use crate::jsonc::strip_jsonc_comments;

use super::catalog::VIDEO_UI_MODES;

const MANIFEST: &str = include_str!("../../../../../config/manifests/builtin.models.jsonc");
const PREVIEW: &str = include_str!("../../../../../config/manifests/builtin.preview-support.jsonc");
const EXCEPTIONS: &str = include_str!("../../../../../config/backend-capabilities/exceptions.json");
const MLX_DESCRIPTOR_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/capabilities.mlx.json");
const CANDLE_DESCRIPTOR_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/capabilities.candle.json");
const AUDIO_DESCRIPTOR_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/audio/capabilities.candle.json");
const MLX_RUNTIME_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/runtime/capabilities.mlx.json");
const CANDLE_RUNTIME_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/runtime/capabilities.candle.json");

const ROUTING_CATALOG: &str = include_str!("catalog.rs");
const ROUTING_MLX: &str = include_str!("mlx.rs");
const ROUTING_CANDLE: &str = include_str!("candle.rs");
const ROUTING_GAPS: &str = include_str!("gaps.rs");
const API_VALIDATION: &str = include_str!("../../../../../apps/rust-api/src/jobs.rs");
const WORKER_DISPATCH: &str = include_str!("../../../../sceneworks-worker/src/lib.rs");
const WORKER_IMAGE_DISPATCH: &str =
    include_str!("../../../../sceneworks-worker/src/image_jobs/base.rs");
const WORKER_ENGINE_TABLE: &str = include_str!("../../../../sceneworks-worker/src/engines.rs");
const WORKER_GPU_CAPABILITIES: &str = include_str!("../../../../sceneworks-worker/src/gpu.rs");
const WORKER_TRAINING_DISPATCH: &str =
    include_str!("../../../../sceneworks-worker/src/training_jobs.rs");
const SCHEDULER: &str = include_str!("../../jobs_store.rs");
const TRAINING_CATALOG: &str = include_str!("../../training.rs");
const WEB_IMAGE_REQUEST: &str = include_str!("../../../../../apps/web/src/imageJobRequest.js");
const WEB_IMAGE_ADVANCED: &str = include_str!("../../../../../apps/web/src/imageJobAdvanced.js");
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
    pub requests: Vec<CapabilityCell>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCapabilityRow {
    pub target: String,
    pub base_model: String,
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
    pub authority: String,
    pub approved_date: String,
    pub user_facing_behavior: String,
    pub revisit_condition: String,
    pub cells: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExceptionRegister {
    schema_version: u32,
    #[serde(default)]
    authorized_approvers: Vec<AuthorizedApprover>,
    records: Vec<ExceptionRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AuthorizedApprover {
    name: String,
    authority: String,
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
}

#[derive(Debug, Deserialize)]
struct ManifestDownload {
    #[serde(default)]
    variant: String,
    #[serde(default)]
    platforms: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeDescriptorFacts {
    schema_version: u32,
    generated_from: RuntimeProvenance,
    model_mappings: BTreeMap<String, String>,
    trainer_mappings: BTreeMap<String, String>,
    worker_capabilities: Vec<String>,
    snapshot: RuntimeSnapshot,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeProvenance {
    inference_revision: String,
}

#[derive(Debug, Deserialize)]
struct RuntimeSnapshot {
    backend: String,
    generator_capabilities: Vec<GeneratorCapabilityFacts>,
    trainer_capabilities: Vec<TrainerCapabilityFacts>,
    #[serde(default)]
    captioner_ids: Vec<String>,
    #[serde(default)]
    image_embedder_ids: Vec<String>,
    #[serde(default)]
    text_llm_ids: Vec<String>,
    #[serde(default)]
    audio_generator_capabilities: Vec<GeneratorCapabilityFacts>,
    #[serde(default)]
    audio_voice_embedder_ids: Vec<String>,
    #[serde(default)]
    audio_transform_ids: Vec<String>,
    #[serde(default)]
    audio_transcriber_ids: Vec<String>,
    #[serde(default)]
    audio_embedder_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GeneratorCapabilityFacts {
    id: String,
    backend: String,
    modality: String,
    #[serde(default)]
    conditioning: Vec<String>,
    supports_lora: bool,
    supports_lokr: bool,
    #[serde(default)]
    supported_quants: Vec<String>,
    supports_preview: bool,
}

#[derive(Debug, Deserialize)]
struct TrainerCapabilityFacts {
    id: String,
    backend: String,
    supports_lora: bool,
    supports_lokr: bool,
    supports_control: bool,
    supports_full_finetune: bool,
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
    let mlx_facts = runtime_facts(MLX_RUNTIME_FACTS, "mlx")?;
    let candle_facts = runtime_facts(CANDLE_RUNTIME_FACTS, "candle")?;
    validate_runtime_pair(&mlx_facts, &candle_facts)?;
    validate_exceptions(&exceptions)?;

    let mut models = Vec::with_capacity(manifest.models.len());
    for model in &manifest.models {
        models.push(model_row(
            model,
            preview.models.get(&model.id),
            &mlx_facts,
            &candle_facts,
        )?);
    }
    models.sort_by(|left, right| left.id.cmp(&right.id));

    let gpu_job_types = gpu_job_rows(&manifest, &mlx_facts, &candle_facts)?;
    let training_kernels = training_rows(&mlx_facts, &candle_facts)?;
    validate_obligations(
        &models,
        &gpu_job_types,
        &training_kernels,
        &exceptions.records,
    )?;

    Ok(BackendCapabilityMatrix {
        schema_version: 2,
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
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<ModelCapabilityRow, String> {
    let mut manifest_operations = model.capabilities.clone();
    if model.ui.get("img2img").and_then(Value::as_bool) == Some(true) {
        manifest_operations.push("image_to_image".to_owned());
    }
    if model.ui.get("promptEnhance").and_then(Value::as_bool) == Some(true) {
        manifest_operations.push("prompt_enhancement".to_owned());
    }
    if model.model_type == "video" {
        manifest_operations.extend(VIDEO_UI_MODES.iter().map(|mode| (*mode).to_owned()));
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
        for operation in &manifest_operations {
            operation_and_mode.push(operation_cell(model, operation, mlx_facts, candle_facts)?);
        }
        let conditioning = descriptor_conditioning_union(model, mlx_facts, candle_facts)?;
        for shape in conditioning {
            conditioning_shape.push(conditioning_cell(model, &shape, mlx_facts, candle_facts)?);
        }
        for adapter in ["lora", "lokr"] {
            user_adapters.push(adapter_cell(model, adapter, mlx_facts, candle_facts)?);
        }
        if model.lora_compatibility.is_null() {
            for adapter in &mut user_adapters {
                adapter.mlx = Some(false);
                adapter.candle = Some(false);
                adapter.parity_obligation = None;
                adapter.preserved_candle_only = false;
            }
        }
        for tier in precision_union(model, mlx_facts, candle_facts)? {
            precision_tier.push(precision_cell(model, &tier, mlx_facts, candle_facts)?);
        }
    } else if is_video {
        for mode in &manifest_operations {
            let payload = video_payload(mode);
            let job_type = video_job_type(mode);
            let job = probe_job(job_type, &model.id, payload)?;
            operation_and_mode.push(routed_cell(
                mode,
                &model.id,
                "video",
                &job,
                mlx_facts,
                candle_facts,
                true,
            )?);
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
            let mlx = backend_supports(&job, mlx_facts)?
                && descriptor_supports_adapter(mlx_facts, &model.id, network_type);
            let candle = backend_supports(&job, candle_facts)?
                && descriptor_supports_adapter(candle_facts, &model.id, network_type);
            user_adapters.push(cell(
                network_type.to_owned(),
                mlx,
                candle,
                gap_for(&model.id, "video-adapter", network_type),
            ));
        }
        for tier in precision_union(model, mlx_facts, candle_facts)? {
            precision_tier.push(precision_cell(model, &tier, mlx_facts, candle_facts)?);
        }
        for shape in descriptor_conditioning_union(model, mlx_facts, candle_facts)? {
            conditioning_shape.push(conditioning_cell(model, &shape, mlx_facts, candle_facts)?);
        }
    }

    if !is_image && !is_video {
        operation_and_mode.extend(utility_model_cells(model, mlx_facts, candle_facts)?);
    }

    let mlx_preview = descriptor_preview(model, mlx_facts)
        .or_else(|| preview.and_then(|backends| backends.get("mlx")).copied());
    let candle_preview = descriptor_preview(model, candle_facts)
        .or_else(|| preview.and_then(|backends| backends.get("candle")).copied());
    let preview = CapabilityCell {
        capability: "live_preview".to_owned(),
        mlx: mlx_preview,
        candle: candle_preview,
        parity_obligation: match (mlx_preview, candle_preview) {
            (Some(true), Some(false) | None) => Some(gap_for(&model.id, "preview", "live_preview")),
            _ => None,
        },
        preserved_candle_only: matches!(
            (mlx_preview, candle_preview),
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

fn runtime_facts(source: &str, expected_backend: &str) -> Result<RuntimeDescriptorFacts, String> {
    let facts: RuntimeDescriptorFacts = serde_json::from_str(source)
        .map_err(|error| format!("parse {expected_backend} runtime descriptor facts: {error}"))?;
    if facts.schema_version != 1 {
        return Err(format!(
            "unsupported {expected_backend} runtime descriptor schema {}",
            facts.schema_version
        ));
    }
    if facts.snapshot.backend != expected_backend {
        return Err(format!(
            "{expected_backend} runtime artifact contains {:?} backend",
            facts.snapshot.backend
        ));
    }
    if facts.generated_from.inference_revision.trim().is_empty() {
        return Err(format!(
            "{expected_backend} runtime artifact has no inference revision"
        ));
    }
    if facts.worker_capabilities.is_empty() {
        return Err(format!(
            "{expected_backend} runtime artifact has no production worker capabilities"
        ));
    }
    Ok(facts)
}

fn validate_runtime_pair(
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<(), String> {
    if mlx.generated_from.inference_revision != candle.generated_from.inference_revision {
        return Err(format!(
            "runtime descriptor revisions differ: mlx={} candle={}",
            mlx.generated_from.inference_revision, candle.generated_from.inference_revision
        ));
    }
    if mlx.model_mappings != candle.model_mappings {
        return Err("matching-platform production model mappings differ".to_owned());
    }
    if mlx.trainer_mappings != candle.trainer_mappings {
        return Err("matching-platform production trainer mappings differ".to_owned());
    }
    for facts in [mlx, candle] {
        if facts.snapshot.generator_capabilities.is_empty() {
            return Err(format!(
                "{} runtime artifact has no generator descriptors",
                facts.snapshot.backend
            ));
        }
        if facts.snapshot.trainer_capabilities.is_empty() {
            return Err(format!(
                "{} runtime artifact has no trainer descriptors",
                facts.snapshot.backend
            ));
        }
        for descriptor in &facts.snapshot.generator_capabilities {
            if descriptor.backend != facts.snapshot.backend || descriptor.modality.trim().is_empty()
            {
                return Err(format!(
                    "{} runtime generator {:?} has backend/modality drift",
                    facts.snapshot.backend, descriptor.id
                ));
            }
        }
    }
    Ok(())
}

fn generator_descriptor<'a>(
    facts: &'a RuntimeDescriptorFacts,
    model: &str,
) -> Option<&'a GeneratorCapabilityFacts> {
    let engine = facts.model_mappings.get(model)?;
    facts
        .snapshot
        .generator_capabilities
        .iter()
        .find(|descriptor| descriptor.id == *engine)
}

fn descriptor_preview(model: &ManifestModel, facts: &RuntimeDescriptorFacts) -> Option<bool> {
    generator_descriptor(facts, &model.id).map(|descriptor| descriptor.supports_preview)
}

fn descriptor_supports_adapter(
    facts: &RuntimeDescriptorFacts,
    model: &str,
    network_type: &str,
) -> bool {
    generator_descriptor(facts, model).is_some_and(|descriptor| match network_type {
        "lora" => descriptor.supports_lora,
        "lokr" => descriptor.supports_lokr,
        _ => false,
    })
}

fn backend_worker(facts: &RuntimeDescriptorFacts) -> Result<WorkerSnapshot, String> {
    let capabilities = facts
        .worker_capabilities
        .iter()
        .map(|capability| {
            serde_json::from_value::<WorkerCapability>(Value::String(capability.clone())).map_err(
                |error| {
                    format!(
                        "{} runtime artifact contains invalid worker capability {capability:?}: {error}",
                        facts.snapshot.backend
                    )
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WorkerSnapshot {
        id: format!("capability-matrix-{}", facts.snapshot.backend),
        gpu_id: if facts.snapshot.backend == "mlx" {
            "mlx".to_owned()
        } else {
            "0".to_owned()
        },
        gpu_name: None,
        status: WorkerStatus::Idle,
        current_job_id: None,
        capabilities,
        loaded_models: Vec::new(),
        utilization: None,
        status_reason: None,
        registered_at: String::new(),
        last_seen_at: String::new(),
        extra: BTreeMap::new(),
    })
}

fn backend_supports(job: &JobSnapshot, facts: &RuntimeDescriptorFacts) -> Result<bool, String> {
    Ok(super::super::worker_supports_job(
        &backend_worker(facts)?,
        job,
    ))
}

fn routed_cell(
    capability: &str,
    model: &str,
    category: &str,
    job: &JobSnapshot,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
    require_descriptor: bool,
) -> Result<CapabilityCell, String> {
    let mlx = backend_supports(job, mlx_facts)?
        && (!require_descriptor || generator_descriptor(mlx_facts, model).is_some());
    let candle = backend_supports(job, candle_facts)?
        && (!require_descriptor || generator_descriptor(candle_facts, model).is_some());
    Ok(cell(
        capability.to_owned(),
        mlx,
        candle,
        gap_for(model, category, capability),
    ))
}

fn operation_cell(
    model: &ManifestModel,
    operation: &str,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<CapabilityCell, String> {
    let (job_type, payload, require_descriptor) = match operation {
        "text_to_image" => (JobType::ImageGenerate, json!({ "mode": operation }), true),
        "edit_image" => (
            JobType::ImageEdit,
            json!({ "mode": operation, "sourceAssetId": "probe" }),
            true,
        ),
        "style_variations" | "image_to_image" => (
            JobType::ImageGenerate,
            json!({ "mode": "text_to_image", "referenceAssetId": "probe" }),
            true,
        ),
        "character_image" => (
            JobType::ImageGenerate,
            json!({ "mode": operation, "referenceAssetId": "probe" }),
            true,
        ),
        "vqa" => (
            JobType::ImageVqa,
            json!({ "sourceAssetId": "probe", "question": "probe" }),
            true,
        ),
        "interleave" => (
            JobType::ImageInterleave,
            json!({ "prompt": "probe", "sourceAssetIds": ["probe"] }),
            true,
        ),
        "image_inpaint" => (
            JobType::ImageEdit,
            json!({
                "mode": "edit_image",
                "sourceAssetId": "probe",
                "maskAssetId": "probe-mask"
            }),
            true,
        ),
        "image_detail" => (
            JobType::ImageDetail,
            json!({ "sourceAssetId": "probe", "engine": "controlnet-tile" }),
            true,
        ),
        "prompt_enhancement" => (
            JobType::PromptRefine,
            json!({ "prompt": "probe", "model": "prompt_refine_anubis_8b" }),
            false,
        ),
        other => {
            return Err(format!(
                "manifest operation {other:?} for image model {:?} has no canonical production request",
                model.id
            ));
        }
    };
    let job = probe_job(job_type, &model.id, payload)?;
    routed_cell(
        operation,
        &model.id,
        "operation",
        &job,
        mlx_facts,
        candle_facts,
        require_descriptor,
    )
}

fn descriptor_conditioning_union(
    model: &ManifestModel,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<Vec<String>, String> {
    let mut shapes = BTreeSet::new();
    for facts in [mlx, candle] {
        if let Some(descriptor) = generator_descriptor(facts, &model.id) {
            shapes.extend(descriptor.conditioning.iter().cloned());
        }
    }
    if model.ui.get("img2img").and_then(Value::as_bool) == Some(true) {
        shapes.insert("reference".to_owned());
    }
    if model.ui.get("multiReference").and_then(Value::as_bool) == Some(true)
        || model
            .ui
            .get("sourceWithMultiReference")
            .and_then(Value::as_bool)
            == Some(true)
    {
        shapes.insert("multiReference".to_owned());
    }
    if model.ui.get("poseLibrary").and_then(Value::as_bool) == Some(true) {
        shapes.insert("control".to_owned());
    }
    Ok(shapes.into_iter().collect())
}

fn conditioning_payload(model_type: &str, shape: &str) -> Result<(JobType, Value), String> {
    let image_edit = || {
        (
            JobType::ImageEdit,
            json!({ "mode": "edit_image", "sourceAssetId": "probe" }),
        )
    };
    let pair = match shape {
        "reference" if model_type == "video" => (
            JobType::VideoGenerate,
            json!({ "mode": "image_to_video", "sourceAssetId": "probe" }),
        ),
        "reference" => (
            JobType::ImageGenerate,
            json!({ "mode": "text_to_image", "referenceAssetId": "probe" }),
        ),
        "multiReference" | "reduxRefs" => (
            JobType::ImageEdit,
            json!({ "mode": "edit_image", "referenceAssetIds": ["probe-a", "probe-b"] }),
        ),
        "control" => (
            JobType::ImageGenerate,
            json!({ "mode": "text_to_image", "advanced": { "poses": [{}] } }),
        ),
        "depth" => (
            JobType::ImageGenerate,
            json!({ "mode": "text_to_image", "advanced": { "depthAssetId": "probe" } }),
        ),
        "mask" => {
            let (kind, mut payload) = image_edit();
            payload["maskAssetId"] = Value::String("probe-mask".to_owned());
            (kind, payload)
        }
        "keyframe" => (
            JobType::VideoGenerate,
            json!({ "mode": "first_last_frame", "sourceAssetId": "probe", "endAssetId": "probe-end" }),
        ),
        "videoClip" => (
            JobType::VideoExtend,
            json!({ "mode": "extend_clip", "sourceClipAssetId": "probe" }),
        ),
        "controlClip" | "videoSync" => (
            JobType::VideoGenerate,
            json!({ "mode": "animate_character", "sourceClipAssetId": "probe", "referenceAssetIds": ["probe-ref"] }),
        ),
        "conversationHistory" => (
            JobType::ImageInterleave,
            json!({ "prompt": "probe", "conversationHistory": [] }),
        ),
        other => {
            return Err(format!(
                "descriptor conditioning {other:?} has no SceneWorks canonical request"
            ));
        }
    };
    Ok(pair)
}

fn conditioning_cell(
    model: &ManifestModel,
    shape: &str,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<CapabilityCell, String> {
    let (job_type, payload) = conditioning_payload(&model.model_type, shape)?;
    let job = probe_job(job_type, &model.id, payload)?;
    let supports = |facts: &RuntimeDescriptorFacts| {
        generator_descriptor(facts, &model.id)
            .is_some_and(|descriptor| descriptor.conditioning.iter().any(|kind| kind == shape))
    };
    let mlx = supports(mlx_facts) && backend_supports(&job, mlx_facts)?;
    let candle = supports(candle_facts) && backend_supports(&job, candle_facts)?;
    Ok(cell(
        shape.to_owned(),
        mlx,
        candle,
        gap_for(&model.id, "conditioning", shape),
    ))
}

fn adapter_cell(
    model: &ManifestModel,
    adapter: &str,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<CapabilityCell, String> {
    let job_type = if model.model_type == "video" {
        JobType::VideoGenerate
    } else {
        JobType::ImageGenerate
    };
    let mode = if model.model_type == "video" {
        "text_to_video"
    } else {
        "text_to_image"
    };
    let job = probe_job(
        job_type,
        &model.id,
        json!({ "mode": mode, "loras": [{ "id": "probe", "networkType": adapter }] }),
    )?;
    let mlx = descriptor_supports_adapter(mlx_facts, &model.id, adapter)
        && backend_supports(&job, mlx_facts)?;
    let candle = descriptor_supports_adapter(candle_facts, &model.id, adapter)
        && backend_supports(&job, candle_facts)?;
    Ok(cell(
        adapter.to_owned(),
        mlx,
        candle,
        gap_for(&model.id, "adapter", adapter),
    ))
}

fn precision_union(
    model: &ManifestModel,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<Vec<String>, String> {
    let mut tiers: BTreeSet<String> = model
        .downloads
        .iter()
        .map(|download| download.variant.as_str())
        .filter(|variant| !variant.is_empty() && *variant != "training")
        .map(str::to_owned)
        .collect();
    for facts in [mlx, candle] {
        if let Some(descriptor) = generator_descriptor(facts, &model.id) {
            tiers.insert("bf16".to_owned());
            tiers.extend(descriptor.supported_quants.iter().cloned());
        }
    }
    Ok(tiers.into_iter().collect())
}

fn manifest_tier_support(model: &ManifestModel, tier: &str, backend: &str) -> bool {
    let matching: Vec<&ManifestDownload> = model
        .downloads
        .iter()
        .filter(|download| download.variant == tier)
        .collect();
    if matching.is_empty() {
        return true;
    }
    matching.iter().any(|download| {
        download.platforms.is_empty()
            || download.platforms.iter().any(|platform| {
                if backend == "mlx" {
                    platform == "macos"
                } else {
                    matches!(platform.as_str(), "windows" | "linux")
                }
            })
    })
}

fn descriptor_tier_support(facts: &RuntimeDescriptorFacts, model: &str, tier: &str) -> bool {
    generator_descriptor(facts, model).is_some_and(|descriptor| {
        tier == "bf16"
            || descriptor
                .supported_quants
                .iter()
                .any(|quant| quant == tier)
    })
}

fn precision_payload(model: &ManifestModel, tier: &str) -> Result<(JobType, Value), String> {
    let mode = if model.model_type == "video" {
        "text_to_video"
    } else {
        "text_to_image"
    };
    let job_type = if model.model_type == "video" {
        JobType::VideoGenerate
    } else {
        JobType::ImageGenerate
    };
    let payload = match tier {
        "bf16" => json!({ "mode": mode }),
        "q4" => json!({ "mode": mode, "advanced": { "mlxQuantize": 4 } }),
        "q8" => json!({ "mode": mode, "advanced": { "mlxQuantize": 8 } }),
        "nvfp4" => json!({ "mode": mode, "advanced": { "quantTier": "nvfp4" } }),
        "int8-convrot" => json!({ "mode": mode, "advanced": { "convRot": true } }),
        other => {
            return Err(format!(
                "manifest precision tier {other:?} has no canonical request"
            ))
        }
    };
    Ok((job_type, payload))
}

fn precision_cell(
    model: &ManifestModel,
    tier: &str,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<CapabilityCell, String> {
    let (job_type, payload) = precision_payload(model, tier)?;
    let job = probe_job(job_type, &model.id, payload)?;
    let support = |facts: &RuntimeDescriptorFacts| {
        let backend = facts.snapshot.backend.as_str();
        let descriptor = descriptor_tier_support(facts, &model.id, tier);
        let artifact_only = model
            .downloads
            .iter()
            .any(|download| download.variant == tier);
        manifest_tier_support(model, tier, backend) && (descriptor || artifact_only)
    };
    let mlx = support(mlx_facts) && backend_supports(&job, mlx_facts)?;
    let candle = support(candle_facts) && backend_supports(&job, candle_facts)?;
    Ok(cell(
        tier.to_owned(),
        mlx,
        candle,
        gap_for(&model.id, "precision", tier),
    ))
}

fn provider_registered(facts: &RuntimeDescriptorFacts, id: &str) -> Option<&'static str> {
    let snapshot = &facts.snapshot;
    if snapshot
        .audio_generator_capabilities
        .iter()
        .any(|descriptor| descriptor.id == id)
    {
        return Some("audio_generator");
    }
    for (kind, ids) in [
        ("captioner", snapshot.captioner_ids.as_slice()),
        ("image_embedder", snapshot.image_embedder_ids.as_slice()),
        ("text_llm", snapshot.text_llm_ids.as_slice()),
        (
            "audio_voice_embedder",
            snapshot.audio_voice_embedder_ids.as_slice(),
        ),
        ("audio_transform", snapshot.audio_transform_ids.as_slice()),
        (
            "audio_transcriber",
            snapshot.audio_transcriber_ids.as_slice(),
        ),
        ("audio_embedder", snapshot.audio_embedder_ids.as_slice()),
    ] {
        if ids.iter().any(|candidate| candidate == id) {
            return Some(kind);
        }
    }
    None
}

fn provider_alias(id: &str) -> &str {
    match id {
        "prompt_refine_anubis_8b" => "anubis_8b",
        "joycaption_beta_one" => "joy_caption",
        "clip_vit_l14" => "clip_vit_l14",
        "vision_caption_qwen3vl_8b" => "qwen3vl_8b",
        other => other,
    }
}

fn utility_model_cells(
    model: &ManifestModel,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<Vec<CapabilityCell>, String> {
    let engine_request = match model.id.as_str() {
        "real_esrgan" => Some((
            "engine:real-esrgan",
            JobType::ImageUpscale,
            json!({ "engine": "real-esrgan", "sourceAssetId": "probe" }),
        )),
        "seedvr2_upscaler" => Some((
            "engine:seedvr2",
            JobType::ImageUpscale,
            json!({ "engine": "seedvr2", "sourceAssetId": "probe" }),
        )),
        "aura_sr_v2" => Some((
            "engine:aura-sr",
            JobType::ImageUpscale,
            json!({ "engine": "aura-sr", "sourceAssetId": "probe" }),
        )),
        "sam3_person_segment" => Some((
            "smart_segment",
            JobType::ImageSegment,
            json!({ "sourceAssetId": "probe", "box": [0, 0, 1, 1] }),
        )),
        "sam2_person_segment" => Some((
            "person_segment",
            JobType::PersonDetect,
            json!({ "sourceAssetId": "probe" }),
        )),
        "person_detector" => Some((
            "person_detect",
            JobType::PersonDetect,
            json!({ "sourceAssetId": "probe" }),
        )),
        "dwpose_pose_detector" => Some((
            "pose_detect",
            JobType::PoseDetect,
            json!({ "sourceAssetId": "probe" }),
        )),
        "controlnet_tile_sdxl" => Some((
            "image_detail",
            JobType::ImageDetail,
            json!({ "model": "sdxl", "engine": "controlnet-tile", "sourceAssetId": "probe" }),
        )),
        id if id.starts_with("pid_") => Some((
            "person_detect",
            JobType::PersonDetect,
            json!({ "sourceAssetId": "probe", "model": id }),
        )),
        "prompt_refine_anubis_8b" => Some((
            "prompt_refine",
            JobType::PromptRefine,
            json!({ "prompt": "probe" }),
        )),
        "joycaption_beta_one" => Some((
            "training_caption",
            JobType::TrainingCaption,
            json!({ "captioner": "joy_caption", "sourceAssetId": "probe" }),
        )),
        "vision_caption_qwen3vl_8b" => Some((
            "training_caption",
            JobType::TrainingCaption,
            json!({ "captioner": "qwen3vl_8b", "sourceAssetId": "probe" }),
        )),
        "clip_vit_l14" => Some((
            "dataset_analysis",
            JobType::DatasetAnalysis,
            json!({ "datasetId": "probe" }),
        )),
        _ => None,
    };
    if let Some((name, job_type, payload)) = engine_request {
        let job = probe_job(job_type, &model.id, payload)?;
        return Ok(vec![routed_cell(
            name,
            &model.id,
            "utility",
            &job,
            mlx_facts,
            candle_facts,
            false,
        )?]);
    }

    let alias = provider_alias(&model.id);
    let mlx_provider = provider_registered(mlx_facts, alias);
    let candle_provider = provider_registered(candle_facts, alias);
    if mlx_provider.is_none() && candle_provider.is_none() {
        return Err(format!(
            "shipped utility/audio model {:?} is absent from runtime providers and has no canonical request",
            model.id
        ));
    }
    let kind = mlx_provider
        .or(candle_provider)
        .expect("one provider exists");
    // Audio is candle-native on both bundles. It belongs in the Candle column even when the
    // matching-platform MLX snapshot carries that side registry.
    let mlx = mlx_provider.is_some() && !kind.starts_with("audio_");
    let candle = candle_provider.is_some() || kind.starts_with("audio_");
    Ok(vec![cell(
        format!("provider:{kind}"),
        mlx,
        candle,
        gap_for(&model.id, "utility", kind),
    )])
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
        "operation"
            if matches!(model, "sana_1600m" | "sana_sprint_1600m")
                && capability == "image_to_image" =>
        {
            ("sc-18475", Some("epic-8588"))
        }
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

fn manifest_model_for_operation<'a>(
    manifest: &'a ManifestRoot,
    operation: &str,
) -> Result<&'a str, String> {
    manifest
        .models
        .iter()
        .find(|model| model.capabilities.iter().any(|item| item == operation))
        .map(|model| model.id.as_str())
        .ok_or_else(|| format!("no shipped model declares operation {operation:?}"))
}

fn manifest_model_of_type<'a>(
    manifest: &'a ManifestRoot,
    model_type: &str,
) -> Result<&'a str, String> {
    manifest
        .models
        .iter()
        .find(|model| model.model_type == model_type)
        .map(|model| model.id.as_str())
        .ok_or_else(|| format!("no shipped model has type {model_type:?}"))
}

fn gpu_job_rows(
    manifest: &ManifestRoot,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<Vec<JobCapabilityRow>, String> {
    let image = manifest_model_for_operation(manifest, "text_to_image")?;
    let edit = manifest_model_for_operation(manifest, "edit_image")?;
    let vqa = manifest_model_for_operation(manifest, "vqa")?;
    let interleave = manifest_model_for_operation(manifest, "interleave")?;
    let video = manifest_model_for_operation(manifest, "text_to_video")?;
    let detail = manifest_model_for_operation(manifest, "image_detail")?;
    let audio = manifest_model_of_type(manifest, "audio")?;
    let training_target = crate::training::builtin_training_targets()
        .targets
        .into_iter()
        .next()
        .ok_or_else(|| "production training catalog is empty".to_owned())?;
    let control_target = crate::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.kernel == "krea_control")
        .ok_or_else(|| "production training catalog has no control target".to_owned())?;

    let specs: Vec<(&str, &str, JobType, &str, Value)> = vec![
        (
            "placeholder",
            "utility",
            JobType::Placeholder,
            "",
            json!({}),
        ),
        (
            "image_generate",
            "per-model",
            JobType::ImageGenerate,
            image,
            json!({ "mode": "text_to_image" }),
        ),
        (
            "image_edit",
            "per-model",
            JobType::ImageEdit,
            edit,
            json!({ "mode": "edit_image", "sourceAssetId": "probe" }),
        ),
        (
            "image_vqa",
            "per-model",
            JobType::ImageVqa,
            vqa,
            json!({ "sourceAssetId": "probe", "question": "probe" }),
        ),
        (
            "image_interleave",
            "per-model",
            JobType::ImageInterleave,
            interleave,
            json!({ "prompt": "probe" }),
        ),
        (
            "video_generate",
            "per-model",
            JobType::VideoGenerate,
            video,
            json!({ "mode": "text_to_video" }),
        ),
        (
            "video_extend",
            "per-model",
            JobType::VideoExtend,
            video,
            json!({ "mode": "extend_clip", "sourceClipAssetId": "probe" }),
        ),
        (
            "video_bridge",
            "per-model",
            JobType::VideoBridge,
            video,
            json!({ "mode": "video_bridge", "sourceClipAssetId": "probe", "bridgeRightClipAssetId": "probe-right" }),
        ),
        (
            "person_detect",
            "utility",
            JobType::PersonDetect,
            "person_detector",
            json!({ "sourceAssetId": "probe" }),
        ),
        (
            "person_track",
            "utility",
            JobType::PersonTrack,
            "person_detector",
            json!({ "sourceAssetId": "probe" }),
        ),
        (
            "person_replace",
            "per-model",
            JobType::PersonReplace,
            video,
            json!({ "mode": "replace_person", "sourceClipAssetId": "probe", "personTrackId": "probe-person", "characterId": "probe-character" }),
        ),
        (
            "audio_generate",
            "per-model",
            JobType::AudioGenerate,
            audio,
            json!({ "prompt": "probe" }),
        ),
        (
            "pose_detect",
            "utility",
            JobType::PoseDetect,
            "dwpose_pose_detector",
            json!({ "sourceAssetId": "probe" }),
        ),
        (
            "kps_extract",
            "utility",
            JobType::KpsExtract,
            "person_detector",
            json!({ "sourceAssetId": "probe" }),
        ),
        (
            "image_detail",
            "per-model",
            JobType::ImageDetail,
            detail,
            json!({ "sourceAssetId": "probe", "engine": "controlnet-tile" }),
        ),
        (
            "image_segment",
            "utility",
            JobType::ImageSegment,
            "sam3_person_segment",
            json!({ "sourceAssetId": "probe", "box": [0, 0, 1, 1] }),
        ),
        (
            "video_upscale",
            "utility",
            JobType::VideoUpscale,
            "seedvr2_upscaler",
            json!({ "engine": "seedvr2", "sourceAssetId": "probe" }),
        ),
        (
            "frame_extract",
            "utility",
            JobType::FrameExtract,
            "",
            json!({ "sourceAssetId": "probe" }),
        ),
        (
            "timeline_export",
            "utility",
            JobType::TimelineExport,
            "",
            json!({ "timelineId": "probe" }),
        ),
        (
            "lora_train",
            "per-kernel",
            JobType::LoraTrain,
            "",
            training_payload(&training_target, "lora"),
        ),
        (
            "control_training",
            "per-kernel",
            JobType::ControlTraining,
            "",
            training_payload(&control_target, "control"),
        ),
        (
            "training_caption",
            "utility",
            JobType::TrainingCaption,
            "joycaption_beta_one",
            json!({ "captioner": "joy_caption", "sourceAssetId": "probe" }),
        ),
        (
            "dataset_analysis",
            "utility",
            JobType::DatasetAnalysis,
            "clip_vit_l14",
            json!({ "datasetId": "probe" }),
        ),
        (
            "catalog_analysis",
            "utility",
            JobType::CatalogAnalysis,
            "clip_vit_l14",
            json!({ "catalogId": "probe" }),
        ),
        (
            "dataset_upscale",
            "utility",
            JobType::DatasetUpscale,
            "real_esrgan",
            json!({ "datasetId": "probe", "engine": "real-esrgan" }),
        ),
        (
            "dataset_face_analysis",
            "utility",
            JobType::DatasetFaceAnalysis,
            "person_detector",
            json!({ "datasetId": "probe" }),
        ),
        (
            "face_likeness_compare",
            "utility",
            JobType::FaceLikenessCompare,
            "person_detector",
            json!({ "sourceAssetId": "probe", "referenceAssetId": "probe-ref" }),
        ),
        (
            "prompt_refine",
            "utility",
            JobType::PromptRefine,
            "prompt_refine_anubis_8b",
            json!({ "prompt": "probe" }),
        ),
    ];

    let mut rows = Vec::new();
    for (job_type, category, kind, model, payload) in specs {
        let job = probe_job(kind, model, payload)?;
        rows.push(JobCapabilityRow {
            job_type: job_type.to_owned(),
            category: category.to_owned(),
            requests: vec![routed_cell(
                "representative_request",
                model,
                "utility",
                &job,
                mlx_facts,
                candle_facts,
                false,
            )?],
        });
    }
    let mut upscale_requests = Vec::new();
    for (capability, engine) in [
        ("engine:real-esrgan", "real-esrgan"),
        ("engine:seedvr2", "seedvr2"),
        ("engine:aura-sr", "aura-sr"),
    ] {
        let job = probe_job(
            JobType::ImageUpscale,
            engine,
            json!({ "engine": engine, "sourceAssetId": "probe" }),
        )?;
        upscale_requests.push(routed_cell(
            capability,
            engine,
            "utility",
            &job,
            mlx_facts,
            candle_facts,
            false,
        )?);
    }
    rows.push(JobCapabilityRow {
        job_type: "image_upscale".to_owned(),
        category: "utility".to_owned(),
        requests: upscale_requests,
    });
    rows.sort_by(|left, right| left.job_type.cmp(&right.job_type));
    Ok(rows)
}

fn training_payload(target: &crate::training::TrainingTarget, network_type: &str) -> Value {
    json!({
        "dryRun": false,
        "plan": {
            "target": {
                "targetId": target.id,
                "kernel": target.kernel,
                "baseModel": target.base_model
            },
            "config": { "advanced": { "networkType": network_type } }
        }
    })
}

fn target_network_types(target: &crate::training::TrainingTarget) -> Result<Vec<String>, String> {
    let mut types = target
        .limits
        .get("networkTypes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if types.is_empty() {
        types.push(if target.kernel == "krea_control" {
            "control".to_owned()
        } else {
            "lora".to_owned()
        });
    }
    types.sort();
    types.dedup();
    if types
        .iter()
        .any(|kind| !matches!(kind.as_str(), "lora" | "lokr" | "control" | "full"))
    {
        return Err(format!(
            "training target {:?} contains an unknown network type: {types:?}",
            target.id
        ));
    }
    Ok(types)
}

fn trainer_supports(facts: &RuntimeDescriptorFacts, target: &str, network_type: &str) -> bool {
    let Some(engine) = facts.trainer_mappings.get(target) else {
        return false;
    };
    facts
        .snapshot
        .trainer_capabilities
        .iter()
        .find(|descriptor| descriptor.id == *engine)
        .is_some_and(|descriptor| {
            descriptor.backend == facts.snapshot.backend
                && match network_type {
                    "lora" => descriptor.supports_lora,
                    "lokr" => descriptor.supports_lokr,
                    "control" => descriptor.supports_control,
                    "full" => descriptor.supports_full_finetune,
                    _ => false,
                }
        })
}

fn training_rows(
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<Vec<TrainingCapabilityRow>, String> {
    let mut rows = Vec::new();
    for target in crate::training::builtin_training_targets().targets {
        for network_type in target_network_types(&target)? {
            let job_type = if network_type == "control" {
                JobType::ControlTraining
            } else {
                JobType::LoraTrain
            };
            let job = probe_job(job_type, "", training_payload(&target, &network_type))?;
            let mlx = trainer_supports(mlx_facts, &target.id, &network_type)
                && backend_supports(&job, mlx_facts)?;
            let candle = trainer_supports(candle_facts, &target.id, &network_type)
                && backend_supports(&job, candle_facts)?;
            rows.push(TrainingCapabilityRow {
                target: target.id.clone(),
                base_model: target.base_model.clone(),
                kernel: target.kernel.clone(),
                network_type: network_type.clone(),
                support: cell(
                    "train".to_owned(),
                    mlx,
                    candle,
                    gap_for(&target.id, "training", &network_type),
                ),
            });
        }
    }
    rows.sort_by(|left, right| {
        (&left.target, &left.network_type).cmp(&(&right.target, &right.network_type))
    });
    Ok(rows)
}

fn validate_exceptions(register: &ExceptionRegister) -> Result<(), String> {
    let authorities: BTreeSet<(&str, &str)> = register
        .authorized_approvers
        .iter()
        .map(|entry| (entry.name.as_str(), entry.authority.as_str()))
        .collect();
    if authorities.len() != register.authorized_approvers.len() {
        return Err("duplicate authorized approver/authority pair".to_owned());
    }
    let mut ids = BTreeSet::new();
    let mut cells = BTreeSet::new();
    for record in &register.records {
        if !ids.insert(record.id.as_str()) {
            return Err(format!("duplicate exception id {:?}", record.id));
        }
        for (field, value) in [
            ("id", record.id.as_str()),
            ("category", record.category.as_str()),
            ("approver", record.approver.as_str()),
            ("authority", record.authority.as_str()),
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
        if !matches!(
            record.category.as_str(),
            "operation"
                | "conditioning"
                | "adapter"
                | "precision"
                | "preview"
                | "training"
                | "utility"
        ) {
            return Err(format!(
                "exception {:?} has unknown category {:?}",
                record.id, record.category
            ));
        }
        if !authorities.contains(&(record.approver.as_str(), record.authority.as_str())) {
            return Err(format!(
                "exception {:?} is not approved by an authorized owner/authority pair",
                record.id
            ));
        }
        if !valid_iso_date(&record.approved_date) {
            return Err(format!(
                "exception {:?} has invalid ISO approvedDate {:?}",
                record.id, record.approved_date
            ));
        }
        for path in &record.cells {
            if !cells.insert(path.as_str()) {
                return Err(format!(
                    "matrix cell {path:?} appears in multiple exceptions"
                ));
            }
        }
    }
    Ok(())
}

fn valid_iso_date(value: &str) -> bool {
    let mut parts = value.split('-');
    let (Some(year), Some(month), Some(day), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    if year.len() != 4 || month.len() != 2 || day.len() != 2 {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        year.parse::<u32>(),
        month.parse::<u32>(),
        day.parse::<u32>(),
    ) else {
        return false;
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=max_day).contains(&day)
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
    let mut all_cells = BTreeMap::new();
    let mut check = |path: String, cell: &CapabilityCell| {
        all_cells.insert(path.clone(), cell.clone());
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
        for request in &job.requests {
            check(
                format!("gpuJobTypes/{}/{}", job.job_type, request.capability),
                request,
            );
        }
    }
    for row in training {
        check(
            format!("trainingKernels/{}/{}", row.target, row.network_type),
            &row.support,
        );
    }
    for path in exception_cells {
        match all_cells.get(path) {
            None => missing.push(format!("{path} (exception names no exact matrix cell)")),
            Some(cell) if cell.mlx != Some(true) || cell.candle == Some(true) => missing.push(
                format!("{path} (exception is not attached to an MLX-only cell)"),
            ),
            Some(_) => {}
        }
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
        ("scheduler", SCHEDULER),
        ("trainingCatalog", TRAINING_CATALOG),
        ("workerDispatch", WORKER_DISPATCH),
        ("workerImageDispatch", WORKER_IMAGE_DISPATCH),
        ("workerEngineTable", WORKER_ENGINE_TABLE),
        ("workerGpuCapabilities", WORKER_GPU_CAPABILITIES),
        ("workerTrainingDispatch", WORKER_TRAINING_DISPATCH),
        ("descriptorMlxFacts", MLX_DESCRIPTOR_FACTS),
        ("descriptorCandleFacts", CANDLE_DESCRIPTOR_FACTS),
        ("descriptorAudioFacts", AUDIO_DESCRIPTOR_FACTS),
        ("descriptorMlxRuntime", MLX_RUNTIME_FACTS),
        ("descriptorCandleRuntime", CANDLE_RUNTIME_FACTS),
        ("descriptorPreviewFacts", PREVIEW),
        ("webImageRequest", WEB_IMAGE_REQUEST),
        ("webImageAdvanced", WEB_IMAGE_ADVANCED),
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
        let manifest: ManifestRoot = serde_json::from_str(&strip_jsonc_comments(MANIFEST)).unwrap();
        let mlx = runtime_facts(MLX_RUNTIME_FACTS, "mlx").unwrap();
        let candle = runtime_facts(CANDLE_RUNTIME_FACTS, "candle").unwrap();
        let actual: BTreeSet<_> = gpu_job_rows(&manifest, &mlx, &candle)
            .unwrap()
            .into_iter()
            .map(|row| row.job_type)
            .collect();
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
            requests: vec![broken],
        }];
        assert!(validate_obligations(&[], &jobs, &[], &[]).is_err());

        let incomplete = ExceptionRecord {
            id: "ex-1".to_owned(),
            category: "".to_owned(),
            approver: "owner".to_owned(),
            authority: "team:image".to_owned(),
            approved_date: "2026-08-10".to_owned(),
            user_facing_behavior: "disabled".to_owned(),
            revisit_condition: "implementation lands".to_owned(),
            cells: vec!["gpuJobTypes/mutated".to_owned()],
        };
        let register = ExceptionRegister {
            schema_version: 1,
            authorized_approvers: vec![AuthorizedApprover {
                name: "owner".to_owned(),
                authority: "team:image".to_owned(),
            }],
            records: vec![incomplete],
        };
        assert!(validate_exceptions(&register).is_err());
    }

    #[test]
    fn every_manifest_operation_has_an_exact_matrix_cell() {
        let matrix = backend_capability_matrix().unwrap();
        for model in &matrix.models {
            let cells: BTreeSet<&str> = model
                .operation_and_mode
                .iter()
                .map(|cell| cell.capability.as_str())
                .collect();
            for operation in &model.manifest_operations {
                assert!(
                    cells.contains(operation.as_str()),
                    "{} operation {operation} has no canonical cell",
                    model.id
                );
            }
        }
    }

    #[test]
    fn audit_regressions_are_derived_from_production_truth() {
        let matrix = backend_capability_matrix().unwrap();
        let krea = matrix
            .models
            .iter()
            .find(|row| row.id == "krea_2_turbo")
            .unwrap();
        let convrot = krea
            .precision_tier
            .iter()
            .find(|cell| cell.capability == "int8-convrot")
            .expect("shipping Krea int8-convrot tier is represented");
        assert_eq!((convrot.mlx, convrot.candle), (Some(false), Some(true)));
        assert!(convrot.preserved_candle_only);

        let mage = matrix
            .training_kernels
            .iter()
            .find(|row| row.target == "mage_flow_base_lora" && row.network_type == "full")
            .expect("production Mage target offers full fine-tuning");
        assert_eq!(mage.support.mlx, Some(true));

        let upscale = matrix
            .gpu_job_types
            .iter()
            .find(|row| row.job_type == "image_upscale")
            .unwrap();
        let aura = upscale
            .requests
            .iter()
            .find(|cell| cell.capability == "engine:aura-sr")
            .unwrap();
        assert_eq!((aura.mlx, aura.candle), (Some(false), Some(false)));
        for engine in ["engine:real-esrgan", "engine:seedvr2"] {
            let request = upscale
                .requests
                .iter()
                .find(|cell| cell.capability == engine)
                .unwrap();
            assert_eq!((request.mlx, request.candle), (Some(true), Some(true)));
        }

        for model_id in ["sana_1600m", "sana_sprint_1600m"] {
            let model = matrix.models.iter().find(|row| row.id == model_id).unwrap();
            for cell in [
                model
                    .operation_and_mode
                    .iter()
                    .find(|cell| cell.capability == "image_to_image")
                    .unwrap(),
                model
                    .conditioning_shape
                    .iter()
                    .find(|cell| cell.capability == "reference")
                    .unwrap(),
            ] {
                assert_eq!(
                    cell.parity_obligation
                        .as_ref()
                        .map(|item| item.work_item.as_str()),
                    Some("sc-18475"),
                    "SANA img2img belongs end-to-end to sc-18475"
                );
            }
        }
    }

    #[test]
    fn exception_register_rejects_bad_dates_authority_duplicates_and_non_mlx_cells() {
        let make = |id: &str, date: &str, cell: &str| ExceptionRecord {
            id: id.to_owned(),
            category: "utility".to_owned(),
            approver: "Capability Owner".to_owned(),
            authority: "team:runtime".to_owned(),
            approved_date: date.to_owned(),
            user_facing_behavior: "The control is disabled with an explanation.".to_owned(),
            revisit_condition: "Revisit when the native implementation ships.".to_owned(),
            cells: vec![cell.to_owned()],
        };
        let authorized = || {
            vec![AuthorizedApprover {
                name: "Capability Owner".to_owned(),
                authority: "team:runtime".to_owned(),
            }]
        };
        let bad_date = ExceptionRegister {
            schema_version: 1,
            authorized_approvers: authorized(),
            records: vec![make("bad-date", "2026-02-30", "missing")],
        };
        assert!(validate_exceptions(&bad_date).is_err());

        let mut unauthorized = make("unauthorized", "2026-08-10", "missing");
        unauthorized.authority = "team:not-authorized".to_owned();
        assert!(validate_exceptions(&ExceptionRegister {
            schema_version: 1,
            authorized_approvers: authorized(),
            records: vec![unauthorized],
        })
        .is_err());

        assert!(validate_exceptions(&ExceptionRegister {
            schema_version: 1,
            authorized_approvers: authorized(),
            records: vec![
                make("one", "2026-08-10", "same"),
                make("two", "2026-08-10", "same"),
            ],
        })
        .is_err());

        let matrix = backend_capability_matrix().unwrap();
        let exception = make(
            "wrong-cell",
            "2026-08-10",
            "models/krea_2_turbo/precisionTier/int8-convrot",
        );
        assert!(validate_obligations(
            &matrix.models,
            &matrix.gpu_job_types,
            &matrix.training_kernels,
            &[exception]
        )
        .is_err());
        let missing = make(
            "missing-cell",
            "2026-08-10",
            "models/does-not-exist/preview",
        );
        assert!(validate_obligations(
            &matrix.models,
            &matrix.gpu_job_types,
            &matrix.training_kernels,
            &[missing]
        )
        .is_err());
    }

    #[test]
    fn rich_runtime_mutations_change_descriptor_and_dispatch_answers() {
        let original = runtime_facts(MLX_RUNTIME_FACTS, "mlx").unwrap();
        let upscale = probe_job(
            JobType::ImageUpscale,
            "real_esrgan",
            json!({ "engine": "real-esrgan", "sourceAssetId": "probe" }),
        )
        .unwrap();
        assert!(backend_supports(&upscale, &original).unwrap());

        let mut value: Value = serde_json::from_str(MLX_RUNTIME_FACTS).unwrap();
        value["workerCapabilities"]
            .as_array_mut()
            .unwrap()
            .retain(|capability| capability.as_str() != Some("image_upscale"));
        let missing_dispatch =
            runtime_facts(&serde_json::to_string(&value).unwrap(), "mlx").unwrap();
        assert!(!backend_supports(&upscale, &missing_dispatch).unwrap());

        let engine = original.model_mappings.get("sana_1600m").unwrap();
        let descriptors = value["snapshot"]["generator_capabilities"]
            .as_array_mut()
            .unwrap();
        descriptors
            .iter_mut()
            .find(|descriptor| descriptor["id"].as_str() == Some(engine))
            .unwrap()["conditioning"] = json!([]);
        let missing_descriptor =
            runtime_facts(&serde_json::to_string(&value).unwrap(), "mlx").unwrap();
        assert!(generator_descriptor(&original, "sana_1600m")
            .unwrap()
            .conditioning
            .iter()
            .any(|kind| kind == "reference"));
        assert!(!generator_descriptor(&missing_descriptor, "sana_1600m")
            .unwrap()
            .conditioning
            .iter()
            .any(|kind| kind == "reference"));
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
