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
const API_GENERATION: &str = include_str!("../../../../../apps/rust-api/src/generation.rs");
const API_CONTRACT_ENTRY: &str = include_str!("../../../../../apps/rust-api/src/lib.rs");
const API_DTO: &str = include_str!("../../../../../apps/rust-api/src/dto.rs");
const WORKER_DISPATCH: &str = include_str!("../../../../sceneworks-worker/src/lib.rs");
const WORKER_IMAGE_DISPATCH: &str =
    include_str!("../../../../sceneworks-worker/src/image_jobs/base.rs");
const WORKER_ENGINE_TABLE: &str = include_str!("../../../../sceneworks-worker/src/engines.rs");
const WORKER_GPU_CAPABILITIES: &str = include_str!("../../../../sceneworks-worker/src/gpu.rs");
const WORKER_VIDEO_DISPATCH: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/mod.rs");
const WORKER_VIDEO_WAN: &str = include_str!("../../../../sceneworks-worker/src/video_jobs/wan.rs");
const WORKER_VIDEO_VACE: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/vace.rs");
const WORKER_VIDEO_LTX: &str = include_str!("../../../../sceneworks-worker/src/video_jobs/ltx.rs");
const WORKER_VIDEO_SVD: &str = include_str!("../../../../sceneworks-worker/src/video_jobs/svd.rs");
const WORKER_VIDEO_BERNINI: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/bernini.rs");
const WORKER_VIDEO_SCAIL2: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/scail2.rs");
const WORKER_VIDEO_KREA: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/krea_realtime.rs");
const WORKER_VIDEO_MOCHI: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/mochi.rs");
const WORKER_VIDEO_CANDLE: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/candle.rs");
const WORKER_VIDEO_SEEDVR2: &str =
    include_str!("../../../../sceneworks-worker/src/video_jobs/seedvr2.rs");
const WORKER_AUDIO_DISPATCH: &str = include_str!("../../../../sceneworks-worker/src/audio_jobs.rs");
const WORKER_UTILITY_DISPATCH: &str =
    include_str!("../../../../sceneworks-worker/src/upscale_jobs.rs");
const WORKER_TRAINING_DISPATCH: &str =
    include_str!("../../../../sceneworks-worker/src/training_jobs.rs");
const SCHEDULER: &str = include_str!("../../jobs_store.rs");
const TRAINING_CATALOG: &str = include_str!("../../training.rs");
const WEB_IMAGE_REQUEST: &str = include_str!("../../../../../apps/web/src/imageJobRequest.js");
const WEB_IMAGE_ADVANCED: &str = include_str!("../../../../../apps/web/src/imageJobAdvanced.js");
const WEB_MAC_GATING: &str = include_str!("../../../../../apps/web/src/macGating.js");
const WEB_PREVIEW_GATING: &str = include_str!("../../../../../apps/web/src/previewSupport.js");
const WEB_SIMPLE_JOBS: &str = include_str!("../../../../../apps/web/src/simple/simpleJobs.js");
const WEB_SIMPLE_VIDEO: &str =
    include_str!("../../../../../apps/web/src/simple/SimpleVideoStudio.jsx");
const WEB_VIDEO_VALIDATION: &str =
    include_str!("../../../../../apps/web/src/videoStudioValidation.js");
const WEB_VIDEO_STUDIO: &str = include_str!("../../../../../apps/web/src/screens/VideoStudio.jsx");
const WEB_SIMPLE_AUDIO: &str =
    include_str!("../../../../../apps/web/src/simple/simpleAudioParts.jsx");
const WEB_SIMPLE_AUDIO_STUDIO: &str =
    include_str!("../../../../../apps/web/src/simple/SimpleAudioStudio.jsx");
const WEB_AUDIO_STUDIO: &str = include_str!("../../../../../apps/web/src/screens/AudioStudio.jsx");
const WEB_UPSCALE_ENGINES: &str = include_str!("../../../../../apps/web/src/upscaleEngines.js");
const WEB_VIDEO_UPSCALE: &str =
    include_str!("../../../../../apps/web/src/screens/VideoUpscalePanel.jsx");

const GENERATOR: &str = "cargo run -p sceneworks-core --bin dump-backend-capability-matrix";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendCapabilityMatrix {
    pub schema_version: u32,
    pub generated_by: String,
    pub summary: MatrixSummary,
    pub sources: BTreeMap<String, String>,
    pub models: Vec<ModelCapabilityRow>,
    pub gpu_job_types: Vec<JobCapabilityRow>,
    pub training_kernels: Vec<TrainingCapabilityRow>,
    pub exceptions: Vec<ExceptionRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatrixSummary {
    pub model_count: usize,
    pub cell_count: usize,
    pub mlx_only_cell_count: usize,
    pub candle_only_cell_count: usize,
    pub exception_count: usize,
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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeDescriptorFacts {
    schema_version: u32,
    generated_from: RuntimeProvenance,
    model_mappings: BTreeMap<String, String>,
    video_model_mappings: Vec<VideoModelMapping>,
    trainer_mappings: BTreeMap<String, String>,
    worker_capabilities: Vec<String>,
    snapshot: RuntimeSnapshot,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct VideoModelMapping {
    model_id: String,
    mode: String,
    engine_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeProvenance {
    inference_revision: String,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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
    supports_prompt_enhancement: bool,
}

#[derive(Debug, Clone, Deserialize)]
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
    let summary = matrix_summary(
        &models,
        &gpu_job_types,
        &training_kernels,
        exceptions.records.len(),
    );

    Ok(BackendCapabilityMatrix {
        schema_version: 2,
        generated_by: GENERATOR.to_owned(),
        summary,
        sources: source_digests(),
        models,
        gpu_job_types,
        training_kernels,
        exceptions: exceptions.records,
    })
}

fn matrix_summary(
    models: &[ModelCapabilityRow],
    jobs: &[JobCapabilityRow],
    training: &[TrainingCapabilityRow],
    exception_count: usize,
) -> MatrixSummary {
    let mut cell_count = 0;
    let mut mlx_only_cell_count = 0;
    let mut candle_only_cell_count = 0;
    let mut count = |cell: &CapabilityCell| {
        cell_count += 1;
        if cell.mlx == Some(true) && cell.candle != Some(true) {
            mlx_only_cell_count += 1;
        }
        if cell.candle == Some(true) && cell.mlx != Some(true) {
            candle_only_cell_count += 1;
        }
    };
    for model in models {
        for cells in [
            model.operation_and_mode.as_slice(),
            model.conditioning_shape.as_slice(),
            model.user_adapters.as_slice(),
            model.precision_tier.as_slice(),
        ] {
            for cell in cells {
                count(cell);
            }
        }
        count(&model.preview);
    }
    for job in jobs {
        for cell in &job.requests {
            count(cell);
        }
    }
    for row in training {
        count(&row.support);
    }
    MatrixSummary {
        model_count: models.len(),
        cell_count,
        mlx_only_cell_count,
        candle_only_cell_count,
        exception_count,
    }
}

fn model_row(
    model: &ManifestModel,
    preview: Option<&BTreeMap<String, bool>>,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<ModelCapabilityRow, String> {
    let manifest_operations = manifested_operations(model);
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
    } else if is_video {
        for mode in &manifest_operations {
            let payload = video_payload(&model.id, mode);
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
            user_adapters.push(adapter_cell(model, network_type, mlx_facts, candle_facts)?);
        }
    }

    if !is_image && !is_video {
        operation_and_mode.extend(utility_model_cells(model, mlx_facts, candle_facts)?);
    }

    // Descriptor axes apply to every registered generator modality. In particular, audio
    // generators live in the Candle audio registry even in a macOS runtime snapshot, and utility
    // manifest rows such as MMAudio still carry generator conditioning that must not disappear.
    for shape in descriptor_conditioning_union(model, mlx_facts, candle_facts)? {
        conditioning_shape.push(conditioning_cell(model, &shape, mlx_facts, candle_facts)?);
    }
    for tier in precision_union(model, mlx_facts, candle_facts)? {
        precision_tier.push(precision_cell(model, &tier, mlx_facts, candle_facts)?);
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
            (Some(true), Some(false) | None) => Some(gap_for(
                &model.id,
                if is_video { "video-preview" } else { "preview" },
                "live_preview",
            )),
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

fn manifested_operations(model: &ManifestModel) -> Vec<String> {
    let mut operations = model.capabilities.clone();
    if model.ui.get("img2img").and_then(Value::as_bool) == Some(true) {
        operations.push("image_to_image".to_owned());
    }
    if model.ui.get("promptEnhance").and_then(Value::as_bool) == Some(true) {
        operations.push("prompt_enhancement".to_owned());
    }
    if model.model_type == "video" {
        operations.extend(VIDEO_UI_MODES.iter().map(|mode| (*mode).to_owned()));
    }
    operations.sort();
    operations.dedup();
    operations
}

fn runtime_facts(source: &str, expected_backend: &str) -> Result<RuntimeDescriptorFacts, String> {
    let facts: RuntimeDescriptorFacts = serde_json::from_str(source)
        .map_err(|error| format!("parse {expected_backend} runtime descriptor facts: {error}"))?;
    if facts.schema_version != 2 {
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
    if facts.video_model_mappings.is_empty() {
        return Err(format!(
            "{expected_backend} runtime artifact has no production video model mappings"
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
        for descriptor in &facts.snapshot.audio_generator_capabilities {
            if descriptor.backend.trim().is_empty() || descriptor.modality != "audio" {
                return Err(format!(
                    "{} runtime audio generator {:?} has backend/modality drift",
                    facts.snapshot.backend, descriptor.id
                ));
            }
        }
    }
    // The mapping catalog is intentionally shared across backends: a mapping may name an MLX-only
    // engine (for example Qwen Image Edit) and therefore be absent from the Candle registry. It is
    // drift only when neither matching-platform registry declares the mapped engine.
    for (model, engine) in mlx.model_mappings.iter().chain(&candle.model_mappings) {
        if descriptor_by_engine(mlx, engine).is_none()
            && descriptor_by_engine(candle, engine).is_none()
        {
            return Err(format!(
                "production model mapping {model:?} -> {engine:?} names no MLX or Candle descriptor"
            ));
        }
    }
    for facts in [mlx, candle] {
        let mut keys = BTreeSet::new();
        for mapping in &facts.video_model_mappings {
            if mapping.model_id.trim().is_empty()
                || mapping.mode.trim().is_empty()
                || mapping.engine_ids.is_empty()
            {
                return Err(format!(
                    "{} runtime artifact contains an incomplete video mapping",
                    facts.snapshot.backend
                ));
            }
            if !keys.insert((&mapping.model_id, &mapping.mode)) {
                return Err(format!(
                    "{} runtime artifact repeats video mapping {:?}/{:?}",
                    facts.snapshot.backend, mapping.model_id, mapping.mode
                ));
            }
            let mut conditioning = BTreeSet::new();
            for engine in &mapping.engine_ids {
                let Some(descriptor) = descriptor_by_engine(facts, engine) else {
                    return Err(format!(
                        "{} video mapping {:?}/{:?} names missing descriptor {:?}",
                        facts.snapshot.backend, mapping.model_id, mapping.mode, engine
                    ));
                };
                if !matches!(descriptor.modality.as_str(), "video" | "both") {
                    return Err(format!(
                        "{} video mapping {:?}/{:?} names non-video descriptor {:?}",
                        facts.snapshot.backend, mapping.model_id, mapping.mode, engine
                    ));
                }
                conditioning.extend(descriptor.conditioning.iter().map(String::as_str));
            }
            for alternatives in super::video_mode_conditioning_requirements(&mapping.mode) {
                if !alternatives
                    .iter()
                    .any(|required| conditioning.contains(required))
                {
                    return Err(format!(
                        "{} video mapping {:?}/{:?} descriptors cannot satisfy required conditioning alternatives {:?}",
                        facts.snapshot.backend, mapping.model_id, mapping.mode, alternatives
                    ));
                }
            }
        }
    }
    Ok(())
}

fn generator_descriptors<'a>(
    facts: &'a RuntimeDescriptorFacts,
    model: &str,
) -> Vec<&'a GeneratorCapabilityFacts> {
    let mut engine_ids = BTreeSet::new();
    if let Some(engine) = facts.model_mappings.get(model) {
        engine_ids.insert(engine.as_str());
    }
    for mapping in facts
        .video_model_mappings
        .iter()
        .filter(|mapping| mapping.model_id == model)
    {
        engine_ids.extend(mapping.engine_ids.iter().map(String::as_str));
    }
    let mut descriptors: Vec<_> = engine_ids
        .into_iter()
        .filter_map(|engine| descriptor_by_engine(facts, engine))
        .collect();
    if descriptors.is_empty() {
        let provider = provider_alias(model);
        descriptors.extend(
            facts
                .snapshot
                .audio_generator_capabilities
                .iter()
                .find(|descriptor| descriptor.id == provider),
        );
    }
    descriptors
}

fn descriptor_is_native_to_snapshot(
    descriptor: &GeneratorCapabilityFacts,
    facts: &RuntimeDescriptorFacts,
) -> bool {
    descriptor.backend == facts.snapshot.backend
}

fn native_generator_descriptors<'a>(
    facts: &'a RuntimeDescriptorFacts,
    model: &str,
) -> Vec<&'a GeneratorCapabilityFacts> {
    generator_descriptors(facts, model)
        .into_iter()
        .filter(|descriptor| descriptor_is_native_to_snapshot(descriptor, facts))
        .collect()
}

fn native_video_route_descriptors<'a>(
    facts: &'a RuntimeDescriptorFacts,
    model: &str,
    mode: &str,
) -> Vec<&'a GeneratorCapabilityFacts> {
    facts
        .video_model_mappings
        .iter()
        .filter(|mapping| mapping.model_id == model && mapping.mode == mode)
        .flat_map(|mapping| mapping.engine_ids.iter())
        .filter_map(|engine| descriptor_by_engine(facts, engine))
        .filter(|descriptor| descriptor_is_native_to_snapshot(descriptor, facts))
        .collect()
}

fn descriptor_modality<'a>(
    model: &'a ManifestModel,
    mlx: &'a RuntimeDescriptorFacts,
    candle: &'a RuntimeDescriptorFacts,
) -> Result<Option<&'a str>, String> {
    let modalities: BTreeSet<&str> = [mlx, candle]
        .into_iter()
        .flat_map(|facts| generator_descriptors(facts, &model.id))
        .map(|descriptor| descriptor.modality.as_str())
        .collect();
    if modalities.contains("audio") {
        if modalities.len() == 1 {
            return Ok(Some("audio"));
        }
        return Err(format!(
            "runtime descriptors disagree on modality for {:?}: {modalities:?}",
            model.id
        ));
    }
    // `both` is a provider-level image/video descriptor breadth, while the shipped manifest row
    // still has one product operation family. Bernini is deliberately `both` on MLX and `video`
    // on Candle; its canonical SceneWorks request remains the manifest's video operation.
    if matches!(model.model_type.as_str(), "image" | "video") {
        return Ok(Some(model.model_type.as_str()));
    }
    if modalities.len() > 1 {
        return Err(format!(
            "runtime descriptors disagree on modality for {:?}: {modalities:?}",
            model.id
        ));
    }
    Ok(modalities.into_iter().next())
}

fn descriptor_by_engine<'a>(
    facts: &'a RuntimeDescriptorFacts,
    engine: &str,
) -> Option<&'a GeneratorCapabilityFacts> {
    facts
        .snapshot
        .generator_capabilities
        .iter()
        .chain(facts.snapshot.audio_generator_capabilities.iter())
        .find(|descriptor| descriptor.id == engine)
}

fn descriptor_preview(model: &ManifestModel, facts: &RuntimeDescriptorFacts) -> Option<bool> {
    let descriptors = native_generator_descriptors(facts, &model.id);
    (!descriptors.is_empty()).then(|| {
        descriptors
            .iter()
            .any(|descriptor| descriptor.supports_preview)
    })
}

fn descriptor_supports_adapter(
    facts: &RuntimeDescriptorFacts,
    model: &str,
    video_mode: Option<&str>,
    network_type: &str,
) -> bool {
    let descriptors = video_mode.map_or_else(
        || native_generator_descriptors(facts, model),
        |mode| native_video_route_descriptors(facts, model, mode),
    );
    descriptors
        .into_iter()
        .any(|descriptor| match network_type {
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

/// The Qwen-Edit and FLUX.2 edit workers are bespoke SceneWorks lanes rather than registered
/// generator descriptors, so their conditioning axes are absent from inference runtime facts. Keep
/// this exception exact: the shared production lane table is consumed by both scheduler and worker,
/// and only the modes/shapes that those concrete lanes implement may supplement descriptor truth.
fn bespoke_image_lane_support(
    backend: &str,
    model: &str,
    category: &str,
    capability: &str,
    job: &JobSnapshot,
) -> bool {
    use super::candle::CandleImageLane;

    if job.payload.get("model").and_then(Value::as_str) != Some(model) {
        return false;
    }
    if backend == "mlx" {
        return model == "flux2_dev"
            && category == "conditioning"
            && capability == "multiReference"
            && super::mlx::image_job_is_mlx_eligible(job);
    }
    if backend != "candle" {
        return false;
    }
    let Some(lane) = super::candle::image_job_candle_lane(job) else {
        return false;
    };
    match lane {
        CandleImageLane::QwenEdit => {
            matches!(
                model,
                "qwen_image_edit"
                    | "qwen_image_edit_2509"
                    | "qwen_image_edit_2511"
                    | "qwen_image_edit_2511_lightning"
            ) && matches!(
                (category, capability),
                ("operation", "edit_image" | "character_image")
                    | ("conditioning", "reference" | "multiReference")
                    | ("adapter", "lora" | "lokr")
            )
        }
        CandleImageLane::Flux2Edit => {
            let exact_model = matches!(
                model,
                "flux2_dev" | "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2"
            );
            exact_model
                && category == "conditioning"
                && (capability == "reference"
                    || (model == "flux2_dev" && capability == "multiReference"))
        }
        _ => false,
    }
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
    let evaluate = |facts: &RuntimeDescriptorFacts| -> Result<bool, String> {
        let routed = backend_supports(job, facts)?;
        if !routed || !require_descriptor {
            return Ok(routed);
        }
        let descriptors = if category == "video" {
            native_video_route_descriptors(facts, model, capability)
        } else {
            native_generator_descriptors(facts, model)
        };
        if descriptors.is_empty() && category == "video" {
            return Err(format!(
                "production {} router admits shipped video route {model:?}/{capability:?}, but the matching-platform artifact has no descriptor mapping",
                facts.snapshot.backend
            ));
        }
        Ok(!descriptors.is_empty()
            || bespoke_image_lane_support(
                &facts.snapshot.backend,
                model,
                category,
                capability,
                job,
            ))
    };
    let mlx = evaluate(mlx_facts)?;
    let candle = evaluate(candle_facts)?;
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
    let (job_type, payload, require_descriptor) = operation_request(model, operation)?;
    let job = probe_job(job_type, &model.id, payload)?;
    if operation == "prompt_enhancement" {
        let supports = |facts: &RuntimeDescriptorFacts| {
            native_generator_descriptors(facts, &model.id)
                .into_iter()
                .any(|descriptor| descriptor.supports_prompt_enhancement)
        };
        let mlx = supports(mlx_facts) && backend_supports(&job, mlx_facts)?;
        let candle = supports(candle_facts) && backend_supports(&job, candle_facts)?;
        return Ok(cell(
            operation.to_owned(),
            mlx,
            candle,
            gap_for(&model.id, "operation", operation),
        ));
    }
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

fn operation_request(
    model: &ManifestModel,
    operation: &str,
) -> Result<(JobType, Value, bool), String> {
    let request = match operation {
        "text_to_image" => (JobType::ImageGenerate, json!({ "mode": operation }), true),
        "edit_image" => (
            JobType::ImageEdit,
            json!({ "mode": operation, "sourceAssetId": "probe" }),
            true,
        ),
        "style_variations" | "image_to_image" if model.id.starts_with("flux2_") => (
            JobType::ImageGenerate,
            json!({ "mode": operation, "referenceAssetId": "probe" }),
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
            JobType::ImageGenerate,
            json!({
                "mode": "text_to_image",
                "advanced": { "enhancePrompt": true }
            }),
            true,
        ),
        other => {
            return Err(format!(
                "manifest operation {other:?} for image model {:?} has no canonical production request",
                model.id
            ));
        }
    };
    Ok(request)
}

fn descriptor_conditioning_union(
    model: &ManifestModel,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<Vec<String>, String> {
    let mut shapes = BTreeSet::new();
    for facts in [mlx, candle] {
        for descriptor in generator_descriptors(facts, &model.id) {
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

fn canonical_model_request(
    model: &ManifestModel,
    descriptor_modality: Option<&str>,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<(JobType, Value), String> {
    if descriptor_modality == Some("audio") || model.model_type == "audio" {
        return Ok((JobType::AudioGenerate, json!({ "prompt": "probe" })));
    }
    if model.model_type == "video" {
        for mode in VIDEO_UI_MODES {
            let job = super::canonical_video_route_probe(&model.id, mode)?;
            let mapped = !native_video_route_descriptors(mlx, &model.id, mode).is_empty()
                || !native_video_route_descriptors(candle, &model.id, mode).is_empty();
            if mapped && (backend_supports(&job, mlx)? || backend_supports(&job, candle)?) {
                let mut payload = job.payload;
                payload.remove("model");
                return Ok((job.job_type, Value::Object(payload)));
            }
        }
        return Err(format!(
            "video model {:?} has no routed descriptor-backed canonical mode",
            model.id
        ));
    }
    if model.model_type == "image" {
        for operation in [
            "text_to_image",
            "edit_image",
            "character_image",
            "image_to_image",
            "style_variations",
            "image_inpaint",
            "image_detail",
            "vqa",
            "interleave",
        ] {
            if manifested_operations(model)
                .iter()
                .any(|item| item == operation)
            {
                let (job_type, payload, _) = operation_request(model, operation)?;
                return Ok((job_type, payload));
            }
        }
    }
    Err(format!(
        "model {:?} has descriptor axes but no canonical generation request",
        model.id
    ))
}

fn conditioning_payload(
    model: &ManifestModel,
    shape: &str,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<(JobType, Value), String> {
    let modality = descriptor_modality(model, mlx, candle)?;
    let (mut job_type, mut payload) = canonical_model_request(model, modality, mlx, candle)?;
    let operations = manifested_operations(model);

    match shape {
        "referenceAudio" | "voiceEmbedding" if modality == Some("audio") => {
            payload["referenceAudioAssetId"] = Value::String("probe-audio".to_owned());
        }
        "audioEdit" if modality == Some("audio") => {
            payload["sourceAudioAssetId"] = Value::String("probe-audio".to_owned());
            payload["editMode"] = Value::String("cover".to_owned());
        }
        "audioEditRegions" if modality == Some("audio") => {
            payload["sourceAudioAssetId"] = Value::String("probe-audio".to_owned());
            payload["editMode"] = Value::String("inpaint".to_owned());
            payload["editRegionStartSecs"] = json!(1.0);
            payload["editRegionEndSecs"] = json!(2.0);
        }
        "videoSync" if modality == Some("audio") => {
            payload["sourceClipAssetId"] = Value::String("probe-video".to_owned());
        }
        "conversationHistory" if modality == Some("audio") => {
            payload["conversationHistory"] = json!([{ "role": "user", "text": "probe" }]);
        }
        "reference" if model.model_type == "video" => {
            job_type = JobType::VideoGenerate;
            payload = video_payload(&model.id, "image_to_video");
        }
        "reference"
            if (model.id.starts_with("sensenova_")
                || model.id.starts_with("qwen_image_edit")
                || model.id.starts_with("flux2_"))
                && operations.iter().any(|item| item == "character_image") =>
        {
            let (kind, character, _) = operation_request(model, "character_image")?;
            job_type = kind;
            payload = character;
        }
        "reference" => {
            if job_type != JobType::ImageEdit {
                payload["referenceAssetId"] = Value::String("probe".to_owned());
            }
        }
        "multiReference" | "reduxRefs" => {
            if operations.iter().any(|item| item == "character_image") {
                let (kind, mut character, _) = operation_request(model, "character_image")?;
                // Plural references replace the singular carrier in the canonical character probe.
                character
                    .as_object_mut()
                    .expect("canonical character request is an object")
                    .remove("referenceAssetId");
                character["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
                job_type = kind;
                payload = character;
            } else if operations.iter().any(|item| item == "edit_image") {
                let (kind, mut edit, _) = operation_request(model, "edit_image")?;
                // The product's source-with-multi-reference edit shape always keeps the primary
                // source; optional ordered references augment it. A plural list alone is not a
                // structurally valid Mage-Flow edit request.
                edit["sourceAssetId"] = Value::String("probe-source".to_owned());
                edit["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
                job_type = kind;
                payload = edit;
            } else {
                payload["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
            }
        }
        "control" => {
            payload["advanced"] = json!({ "poses": [{}] });
        }
        "depth" => {
            payload["advanced"] = json!({ "depthAssetId": "probe" });
        }
        "mask" => {
            let (kind, mut edit, _) = operation_request(model, "edit_image")?;
            edit["maskAssetId"] = Value::String("probe-mask".to_owned());
            job_type = kind;
            payload = edit;
        }
        "keyframe" => {
            job_type = JobType::VideoGenerate;
            payload = video_payload(&model.id, "first_last_frame");
        }
        "videoClip" => {
            job_type = JobType::VideoExtend;
            payload = video_payload(&model.id, "extend_clip");
        }
        "controlClip" | "videoSync" => {
            job_type = JobType::VideoGenerate;
            payload = video_payload(&model.id, "animate_character");
        }
        "conversationHistory" => {
            job_type = JobType::ImageInterleave;
            payload = json!({
                "prompt": "probe",
                "conversationHistory": [{ "role": "user", "text": "probe" }]
            });
        }
        other => {
            return Err(format!(
                "descriptor conditioning {other:?} for modality {modality:?} has no SceneWorks canonical request"
            ));
        }
    }
    Ok((job_type, payload))
}

fn conditioning_cell(
    model: &ManifestModel,
    shape: &str,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<CapabilityCell, String> {
    if model.model_type == "video" {
        // Derive the probe set from the production mode contract rather than maintaining a second
        // shape-to-mode table. A descriptor axis with no production semantic is an error; an axis
        // whose semantic exists but which this model never routes is represented as false/false.
        // This distinction keeps broad `both` descriptors (Bernini's still-image `reference`, for
        // example) visible without falsely claiming that the video wrapper constructs that shape.
        let requirement_shape = match shape {
            "reduxRefs" => "multiReference",
            other => other,
        };
        let modes: Vec<&str> = VIDEO_UI_MODES
            .iter()
            .copied()
            .filter(|mode| {
                super::video_mode_conditioning_requirements(mode)
                    .iter()
                    .any(|alternatives| alternatives.contains(&requirement_shape))
                    // PersonReplace's public request names a tracked person; the production worker
                    // resolves that track to the engine's Mask conditioning internally. It is the
                    // one descriptor shape intentionally realized rather than supplied by the API.
                    || (shape == "mask" && *mode == "replace_person")
            })
            .collect();
        if modes.is_empty() {
            return Err(format!(
                "descriptor video conditioning {shape:?} for {:?} has no production mode semantic",
                model.id
            ));
        }
        let supports = |facts: &RuntimeDescriptorFacts| -> Result<bool, String> {
            for mode in &modes {
                let job = super::canonical_video_route_probe(&model.id, mode)?;
                let descriptor_supports = native_video_route_descriptors(facts, &model.id, mode)
                    .into_iter()
                    .any(|descriptor| descriptor.conditioning.iter().any(|kind| kind == shape));
                if descriptor_supports && backend_supports(&job, facts)? {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        let mlx = supports(mlx_facts)?;
        let candle = supports(candle_facts)?;
        return Ok(cell(
            shape.to_owned(),
            mlx,
            candle,
            gap_for(&model.id, "video-conditioning", shape),
        ));
    }

    let (job_type, payload) = conditioning_payload(model, shape, mlx_facts, candle_facts)?;
    let job = probe_job(job_type, &model.id, payload)?;
    let supports = |facts: &RuntimeDescriptorFacts| {
        let descriptors = native_generator_descriptors(facts, &model.id);
        descriptors
            .into_iter()
            .any(|descriptor| descriptor.conditioning.iter().any(|kind| kind == shape))
            || bespoke_image_lane_support(
                &facts.snapshot.backend,
                &model.id,
                "conditioning",
                shape,
                &job,
            )
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
    if model.model_type == "video" {
        let supports = |facts: &RuntimeDescriptorFacts| -> Result<bool, String> {
            for mode in VIDEO_UI_MODES {
                let descriptor_supports =
                    descriptor_supports_adapter(facts, &model.id, Some(mode), adapter);
                if !descriptor_supports {
                    continue;
                }
                let mut job = super::canonical_video_route_probe(&model.id, mode)?;
                job.payload.insert(
                    "loras".to_owned(),
                    json!([{ "id": "probe", "networkType": adapter }]),
                );
                if backend_supports(&job, facts)? {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        return Ok(cell(
            adapter.to_owned(),
            supports(mlx_facts)?,
            supports(candle_facts)?,
            gap_for(&model.id, "video-adapter", adapter),
        ));
    }

    let modality = descriptor_modality(model, mlx_facts, candle_facts)?;
    let (job_type, mut payload) =
        canonical_model_request(model, modality, mlx_facts, candle_facts)?;
    payload["loras"] = json!([{ "id": "probe", "networkType": adapter }]);
    let job = probe_job(job_type, &model.id, payload)?;
    let supports = |facts: &RuntimeDescriptorFacts| {
        descriptor_supports_adapter(facts, &model.id, None, adapter)
            || bespoke_image_lane_support(
                &facts.snapshot.backend,
                &model.id,
                "adapter",
                adapter,
                &job,
            )
    };
    let mlx = supports(mlx_facts) && backend_supports(&job, mlx_facts)?;
    let candle = supports(candle_facts) && backend_supports(&job, candle_facts)?;
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
        for descriptor in generator_descriptors(facts, &model.id) {
            if matches!(descriptor.modality.as_str(), "image" | "video" | "both") {
                tiers.insert("bf16".to_owned());
            }
            tiers.extend(descriptor.supported_quants.iter().cloned());
        }
    }
    Ok(tiers.into_iter().collect())
}

fn manifest_artifact_tier_support(model: &ManifestModel, tier: &str, backend: &str) -> bool {
    model
        .downloads
        .iter()
        .filter(|download| download.variant == tier)
        .any(|download| {
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

fn descriptor_tier_support(
    facts: &RuntimeDescriptorFacts,
    model: &str,
    video_mode: Option<&str>,
    tier: &str,
) -> bool {
    let descriptors = video_mode.map_or_else(
        || native_generator_descriptors(facts, model),
        |mode| native_video_route_descriptors(facts, model, mode),
    );
    descriptors.into_iter().any(|descriptor| {
        tier == "bf16"
            || descriptor
                .supported_quants
                .iter()
                .any(|quant| quant == tier)
    })
}

fn precision_payload(
    model: &ManifestModel,
    tier: &str,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<(JobType, Value), String> {
    let modality = descriptor_modality(model, mlx, candle)?;
    let (job_type, mut payload) = canonical_model_request(model, modality, mlx, candle)?;
    let advanced = match tier {
        "bf16" => None,
        "q4" => Some(json!({ "mlxQuantize": 4 })),
        "q8" => Some(json!({ "mlxQuantize": 8 })),
        "nvfp4" => Some(json!({ "quantTier": "nvfp4" })),
        "int8-convrot" => Some(json!({ "convRot": true })),
        other => {
            return Err(format!(
                "manifest precision tier {other:?} has no canonical request"
            ))
        }
    };
    if let Some(advanced) = advanced {
        payload["advanced"] = advanced;
    }
    Ok((job_type, payload))
}

fn precision_cell(
    model: &ManifestModel,
    tier: &str,
    mlx_facts: &RuntimeDescriptorFacts,
    candle_facts: &RuntimeDescriptorFacts,
) -> Result<CapabilityCell, String> {
    let (job_type, payload) = precision_payload(model, tier, mlx_facts, candle_facts)?;
    let job = probe_job(job_type, &model.id, payload)?;
    let video_mode = (model.model_type == "video")
        .then(|| job.payload.get("mode").and_then(Value::as_str))
        .flatten();
    let support = |facts: &RuntimeDescriptorFacts| {
        let backend = facts.snapshot.backend.as_str();
        let descriptor = descriptor_tier_support(facts, &model.id, video_mode, tier);
        // Runtime descriptors and exact backend-specific manifest artifacts are independent
        // authorities. A macOS-only exact tier must not veto a native Candle descriptor merely
        // because Candle installs an unvarianted whole-repository snapshot (and vice versa).
        // The production scheduler predicate below remains mandatory for either source of truth.
        descriptor || manifest_artifact_tier_support(model, tier, backend)
    };
    let mlx = support(mlx_facts) && backend_supports(&job, mlx_facts)?;
    let candle = support(candle_facts) && backend_supports(&job, candle_facts)?;
    Ok(cell(
        tier.to_owned(),
        mlx,
        candle,
        gap_for(
            &model.id,
            if model.model_type == "video" {
                "video-precision"
            } else {
                "precision"
            },
            tier,
        ),
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
        "video-conditioning" => ("sc-18478", Some("epic-8588")),
        "video-precision" => ("sc-18478", Some("epic-9083")),
        "video-preview" => ("sc-18478", Some("epic-16948")),
        category if category == "video" || category.starts_with("video-") => {
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
        // sc-18479 closes the current matrix. Future residual training gaps belong to the
        // parity-closure story rather than the completed implementation story.
        "training" => ("sc-18481", None),
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

fn video_payload(model: &str, mode: &str) -> Value {
    let mut job = super::canonical_video_route_probe(model, mode)
        .expect("VIDEO_UI_MODES contains only canonical production modes");
    job.payload.remove("model");
    Value::Object(job.payload)
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
    validate_probe_structure(&job_type, &payload)?;
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

fn validate_probe_structure(
    job_type: &JobType,
    payload: &Map<String, Value>,
) -> Result<(), String> {
    let nonempty = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    let mode = payload.get("mode").and_then(Value::as_str);
    let require = |condition: bool, detail: &str| {
        condition.then_some(()).ok_or_else(|| {
            format!("canonical {job_type:?} probe is structurally invalid: {detail}")
        })
    };

    match job_type {
        JobType::ImageGenerate => {
            require(nonempty("model"), "model is required")?;
            require(
                mode.is_some_and(|value| !value.trim().is_empty()),
                "mode is required",
            )?;
        }
        JobType::ImageEdit => {
            require(nonempty("model"), "model is required")?;
            require(mode == Some("edit_image"), "mode must be edit_image")?;
            require(nonempty("sourceAssetId"), "sourceAssetId is required")?;
        }
        JobType::ImageVqa => {
            require(nonempty("model"), "model is required")?;
            require(nonempty("sourceAssetId"), "sourceAssetId is required")?;
            require(nonempty("question"), "question is required")?;
        }
        JobType::ImageInterleave => {
            require(nonempty("model"), "model is required")?;
            require(nonempty("prompt"), "prompt is required")?;
        }
        JobType::VideoGenerate => {
            require(nonempty("model"), "model is required")?;
            require(
                mode.is_some_and(|value| !value.trim().is_empty()),
                "mode is required",
            )?;
            if matches!(mode, Some("image_to_video" | "first_last_frame")) {
                require(nonempty("sourceAssetId"), "sourceAssetId is required")?;
            }
            if mode == Some("first_last_frame") {
                require(nonempty("lastFrameAssetId"), "lastFrameAssetId is required")?;
            }
            if matches!(
                mode,
                Some("video_to_video" | "reference_video_to_video" | "ads2v")
            ) {
                require(
                    nonempty("sourceClipAssetId"),
                    "sourceClipAssetId is required",
                )?;
            }
            if matches!(
                mode,
                Some("reference_to_video" | "reference_video_to_video" | "ads2v")
            ) {
                require(
                    payload
                        .get("referenceAssetIds")
                        .and_then(Value::as_array)
                        .is_some_and(|ids| !ids.is_empty()),
                    "referenceAssetIds is required",
                )?;
            }
            if mode == Some("multi_video_to_video") {
                require(
                    payload
                        .get("sourceClipAssetIds")
                        .and_then(Value::as_array)
                        .is_some_and(|ids| ids.len() >= 2),
                    "at least two sourceClipAssetIds are required",
                )?;
            }
            if mode == Some("ads2v") {
                require(
                    nonempty("referenceClipAssetId"),
                    "referenceClipAssetId is required",
                )?;
            }
            if mode == Some("animate_character") {
                require(nonempty("referenceAssetId"), "referenceAssetId is required")?;
                require(
                    nonempty("sourceClipAssetId"),
                    "sourceClipAssetId is required",
                )?;
            }
        }
        JobType::VideoExtend => {
            require(mode == Some("extend_clip"), "mode must be extend_clip")?;
            require(
                nonempty("sourceClipAssetId"),
                "sourceClipAssetId is required",
            )?;
        }
        JobType::VideoBridge => {
            require(mode == Some("video_bridge"), "mode must be video_bridge")?;
            require(
                nonempty("sourceClipAssetId"),
                "sourceClipAssetId is required",
            )?;
            require(
                nonempty("bridgeRightClipAssetId"),
                "bridgeRightClipAssetId is required",
            )?;
        }
        JobType::PersonReplace => {
            require(
                mode == Some("replace_person"),
                "mode must be replace_person",
            )?;
            require(
                nonempty("sourceClipAssetId"),
                "sourceClipAssetId is required",
            )?;
            require(nonempty("personTrackId"), "personTrackId is required")?;
            require(nonempty("characterId"), "characterId is required")?;
        }
        JobType::AudioGenerate => {
            require(nonempty("model"), "model is required")?;
            require(nonempty("prompt"), "prompt is required")?;
            if nonempty("sourceAudioAssetId") {
                require(nonempty("editMode"), "editMode is required for audio edit")?;
            }
        }
        _ => {}
    }
    Ok(())
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
        ("apiGeneration", API_GENERATION),
        ("apiContractEntry", API_CONTRACT_ENTRY),
        ("apiDto", API_DTO),
        ("scheduler", SCHEDULER),
        ("trainingCatalog", TRAINING_CATALOG),
        ("workerDispatch", WORKER_DISPATCH),
        ("workerImageDispatch", WORKER_IMAGE_DISPATCH),
        ("workerEngineTable", WORKER_ENGINE_TABLE),
        ("workerGpuCapabilities", WORKER_GPU_CAPABILITIES),
        ("workerVideoDispatch", WORKER_VIDEO_DISPATCH),
        ("workerVideoWan", WORKER_VIDEO_WAN),
        ("workerVideoVace", WORKER_VIDEO_VACE),
        ("workerVideoLtx", WORKER_VIDEO_LTX),
        ("workerVideoSvd", WORKER_VIDEO_SVD),
        ("workerVideoBernini", WORKER_VIDEO_BERNINI),
        ("workerVideoScail2", WORKER_VIDEO_SCAIL2),
        ("workerVideoKreaRealtime", WORKER_VIDEO_KREA),
        ("workerVideoMochi", WORKER_VIDEO_MOCHI),
        ("workerVideoCandle", WORKER_VIDEO_CANDLE),
        ("workerVideoSeedvr2", WORKER_VIDEO_SEEDVR2),
        ("workerAudioDispatch", WORKER_AUDIO_DISPATCH),
        ("workerUtilityDispatch", WORKER_UTILITY_DISPATCH),
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
        ("webSimpleJobs", WEB_SIMPLE_JOBS),
        ("webSimpleVideo", WEB_SIMPLE_VIDEO),
        ("webVideoValidation", WEB_VIDEO_VALIDATION),
        ("webVideoStudio", WEB_VIDEO_STUDIO),
        ("webSimpleAudio", WEB_SIMPLE_AUDIO),
        ("webSimpleAudioStudio", WEB_SIMPLE_AUDIO_STUDIO),
        ("webAudioStudio", WEB_AUDIO_STUDIO),
        ("webUpscaleEngines", WEB_UPSCALE_ENGINES),
        ("webVideoUpscale", WEB_VIDEO_UPSCALE),
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
    fn bespoke_conditioning_authority_is_exact_and_route_backed() {
        let qwen_character = probe_job(
            JobType::ImageGenerate,
            "qwen_image_edit_2511",
            json!({ "mode": "character_image", "referenceAssetIds": ["a", "b"] }),
        )
        .unwrap();
        assert!(bespoke_image_lane_support(
            "candle",
            "qwen_image_edit_2511",
            "conditioning",
            "multiReference",
            &qwen_character,
        ));
        assert!(!bespoke_image_lane_support(
            "candle",
            "qwen_image",
            "conditioning",
            "multiReference",
            &qwen_character,
        ));
        assert!(!bespoke_image_lane_support(
            "candle",
            "qwen_image_edit_2509",
            "conditioning",
            "multiReference",
            &qwen_character,
        ));
        assert!(!bespoke_image_lane_support(
            "candle",
            "qwen_image_edit_2511",
            "conditioning",
            "control",
            &qwen_character,
        ));

        let flux_multi = probe_job(
            JobType::ImageGenerate,
            "flux2_dev",
            json!({ "mode": "character_image", "referenceAssetIds": ["a", "b"] }),
        )
        .unwrap();
        for backend in ["mlx", "candle"] {
            assert!(bespoke_image_lane_support(
                backend,
                "flux2_dev",
                "conditioning",
                "multiReference",
                &flux_multi,
            ));
        }
        assert!(!bespoke_image_lane_support(
            "candle",
            "flux2_dev",
            "conditioning",
            "control",
            &flux_multi,
        ));

        let malformed = probe_job(
            JobType::ImageGenerate,
            "flux2_dev",
            json!({ "mode": "reference", "referenceAssetId": " " }),
        )
        .unwrap();
        assert!(!bespoke_image_lane_support(
            "candle",
            "flux2_dev",
            "conditioning",
            "reference",
            &malformed,
        ));
    }

    #[test]
    fn sc18476_conditioning_cells_match_production_routes() {
        let matrix = backend_capability_matrix().unwrap();
        for model in [
            "sensenova_u1_8b",
            "sensenova_u1_8b_infographic_v2",
            "sensenova_u1_8b_infographic_v3",
            "sensenova_u1_8b_fast",
            "sensenova_u1_8b_infographic_v2_fast",
            "sensenova_u1_8b_infographic_v3_fast",
            "qwen_image_edit_2511",
            "qwen_image_edit_2511_lightning",
            "flux2_dev",
        ] {
            let row = matrix
                .models
                .iter()
                .find(|row| row.id == model)
                .unwrap_or_else(|| panic!("missing matrix row for {model}"));
            for shape in ["reference", "multiReference"] {
                let cell = row
                    .conditioning_shape
                    .iter()
                    .find(|cell| cell.capability == shape)
                    .unwrap_or_else(|| panic!("missing {model}/{shape} matrix cell"));
                assert_eq!(
                    (cell.mlx, cell.candle),
                    (Some(true), Some(true)),
                    "{model}/{shape} must match the strict native route"
                );
            }
        }
        for model in ["qwen_image_edit_2511", "qwen_image_edit_2511_lightning"] {
            let row = matrix.models.iter().find(|row| row.id == model).unwrap();
            for operation in ["edit_image", "character_image"] {
                let cell = row
                    .operation_and_mode
                    .iter()
                    .find(|cell| cell.capability == operation)
                    .unwrap();
                assert_eq!((cell.mlx, cell.candle), (Some(true), Some(true)));
            }
        }
    }

    #[test]
    fn sc18477_bespoke_qwen_edit_adapter_cells_match_production_routes() {
        let matrix = backend_capability_matrix().unwrap();
        for model in ["qwen_image_edit_2511", "qwen_image_edit_2511_lightning"] {
            let row = matrix.models.iter().find(|row| row.id == model).unwrap();
            for adapter in ["lora", "lokr"] {
                let cell = row
                    .user_adapters
                    .iter()
                    .find(|cell| cell.capability == adapter)
                    .unwrap();
                assert_eq!(
                    (cell.mlx, cell.candle),
                    (Some(true), Some(true)),
                    "{model}/{adapter} must match the bespoke Qwen-Edit route"
                );
                assert!(cell.parity_obligation.is_none());
            }
        }
    }

    #[test]
    fn generated_summary_is_derived_from_every_matrix_section() {
        let matrix = backend_capability_matrix().unwrap();
        assert_eq!(
            matrix.summary,
            matrix_summary(
                &matrix.models,
                &matrix.gpu_job_types,
                &matrix.training_kernels,
                matrix.exceptions.len(),
            )
        );

        let mut models = matrix.models.clone();
        let both = models
            .iter_mut()
            .flat_map(|row| row.operation_and_mode.iter_mut())
            .find(|cell| cell.mlx == Some(true) && cell.candle == Some(true))
            .expect("matrix has a both-backend operation");
        both.candle = Some(false);
        let mutated = matrix_summary(
            &models,
            &matrix.gpu_job_types,
            &matrix.training_kernels,
            matrix.exceptions.len(),
        );
        assert_eq!(mutated.cell_count, matrix.summary.cell_count);
        assert_eq!(
            mutated.mlx_only_cell_count,
            matrix.summary.mlx_only_cell_count + 1
        );
        assert_eq!(
            mutated.candle_only_cell_count,
            matrix.summary.candle_only_cell_count
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
    fn future_training_gaps_are_attributed_to_the_closure_story() {
        let obligation = gap_for("synthetic_training_base", "training", "synthetic_lora");
        assert_eq!(obligation.work_item, "sc-18481");
        assert_eq!(
            obligation.url,
            "https://app.shortcut.com/trefry/story/18481"
        );
        assert_ne!(obligation.work_item, "sc-18479");
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
        let flux2 = matrix
            .models
            .iter()
            .find(|row| row.id == "flux2_dev")
            .unwrap();
        let enhancement = flux2
            .operation_and_mode
            .iter()
            .find(|cell| cell.capability == "prompt_enhancement")
            .expect("FLUX.2 prompt enhancement is represented");
        assert_eq!(
            (enhancement.mlx, enhancement.candle),
            (Some(true), Some(true))
        );
        assert!(enhancement.parity_obligation.is_none());
        assert!(!enhancement.preserved_candle_only);

        let mut video_mlx_only = 0;
        for video in matrix.models.iter().filter(|row| row.model_type == "video") {
            for cell in video
                .operation_and_mode
                .iter()
                .chain(&video.conditioning_shape)
                .chain(&video.user_adapters)
                .chain(&video.precision_tier)
                .chain(std::iter::once(&video.preview))
                .filter(|cell| cell.mlx == Some(true) && cell.candle != Some(true))
            {
                video_mlx_only += 1;
                assert_eq!(
                    cell.parity_obligation
                        .as_ref()
                        .map(|item| item.work_item.as_str()),
                    Some("sc-18478"),
                    "video capability {}/{} must stay with the video parity story",
                    video.id,
                    cell.capability
                );
            }
        }
        assert!(
            video_mlx_only > 0,
            "the shipped matrix must retain explicit MLX-only video parity obligations"
        );

        // LTX clip append/control routes require the IC-LoRA carried by the canonical probe. The
        // operation matrix must evaluate that complete runnable shape, not rebuild a bare payload
        // that both production routers correctly reject.
        for model_id in ["ltx_2_3", "ltx_2_3_eros"] {
            let row = matrix.models.iter().find(|row| row.id == model_id).unwrap();
            for mode in ["extend_clip", "video_bridge", "replace_person"] {
                let cell = row
                    .operation_and_mode
                    .iter()
                    .find(|cell| cell.capability == mode)
                    .unwrap_or_else(|| panic!("{model_id}/{mode} is represented"));
                assert_eq!(
                    (cell.mlx, cell.candle),
                    (Some(true), Some(true)),
                    "{model_id}/{mode} uses the complete IC-LoRA probe on both backends"
                );
                assert!(cell.parity_obligation.is_none());
            }
        }

        let vace_fun = matrix
            .models
            .iter()
            .find(|row| row.id == "wan_2_2_vace_fun_14b")
            .expect("the shipped VACE-Fun row is represented");
        let replace = vace_fun
            .operation_and_mode
            .iter()
            .find(|cell| cell.capability == "replace_person")
            .expect("VACE-Fun replace_person operation is represented");
        assert_eq!((replace.mlx, replace.candle), (Some(true), Some(true)));
        for shape in ["controlClip", "reference"] {
            let cell = vace_fun
                .conditioning_shape
                .iter()
                .find(|cell| cell.capability == shape)
                .expect("VACE-Fun descriptor conditioning is represented");
            assert_eq!((cell.mlx, cell.candle), (Some(true), Some(true)));
        }
        for capability in ["lora", "lokr"] {
            let cell = vace_fun
                .user_adapters
                .iter()
                .find(|cell| cell.capability == capability)
                .expect("VACE-Fun adapter axis is represented");
            assert_eq!((cell.mlx, cell.candle), (Some(true), Some(true)));
        }
        for (tier, expected) in [
            ("bf16", (Some(true), Some(true))),
            ("q4", (Some(true), Some(false))),
            ("q8", (Some(true), Some(false))),
        ] {
            let cell = vace_fun
                .precision_tier
                .iter()
                .find(|cell| cell.capability == tier)
                .expect("VACE-Fun precision axis is represented");
            assert_eq!((cell.mlx, cell.candle), expected);
        }

        // Bernini's `both` descriptor also advertises singular Reference for its still-image
        // renderer, but the shipped video wrapper intentionally constructs MultiReference from
        // `referenceAssetIds`. Keep the rich axis visible without turning a non-existent video
        // request into support on either backend.
        let bernini = matrix
            .models
            .iter()
            .find(|row| row.id == "bernini")
            .expect("the shipped Bernini video row is represented");
        for (shape, expected) in [
            ("reference", (Some(false), Some(false))),
            ("multiReference", (Some(true), Some(true))),
            ("videoClip", (Some(true), Some(true))),
        ] {
            let cell = bernini
                .conditioning_shape
                .iter()
                .find(|cell| cell.capability == shape)
                .unwrap_or_else(|| panic!("Bernini descriptor axis {shape} is represented"));
            assert_eq!((cell.mlx, cell.candle), expected, "Bernini {shape}");
        }

        // Exact macOS bf16 downloads must not suppress an independently shipped Candle dense
        // descriptor. Bernini additionally publishes q4/q8 inside its unvarianted Candle snapshot,
        // and both production worker lanes resolve those tier subdirectories.
        for model_id in [
            "bernini",
            "bernini_image",
            "ltx_2_3",
            "sana_1600m",
            "sana_sprint_1600m",
            "scail2_14b",
        ] {
            let row = matrix.models.iter().find(|row| row.id == model_id).unwrap();
            let dense = row
                .precision_tier
                .iter()
                .find(|cell| cell.capability == "bf16")
                .unwrap_or_else(|| panic!("{model_id} dense precision is represented"));
            assert_eq!(
                (dense.mlx, dense.candle),
                (Some(true), Some(true)),
                "{model_id} dense precision follows both native descriptors"
            );
        }
        for model_id in ["bernini", "bernini_image"] {
            let row = matrix.models.iter().find(|row| row.id == model_id).unwrap();
            for tier in ["q4", "q8"] {
                let cell = row
                    .precision_tier
                    .iter()
                    .find(|cell| cell.capability == tier)
                    .unwrap_or_else(|| panic!("{model_id} {tier} precision is represented"));
                assert_eq!(
                    (cell.mlx, cell.candle),
                    (Some(true), Some(true)),
                    "{model_id} {tier} follows the descriptor and production tier resolver"
                );
            }
        }

        // Both SCAIL production modes accept user adapters. The model-level adapter axis is the union
        // of descriptor-backed production modes, not an accident of mode ordering.
        let scail = matrix
            .models
            .iter()
            .find(|row| row.id == "scail2_14b")
            .unwrap();
        for adapter in ["lora", "lokr"] {
            let cell = scail
                .user_adapters
                .iter()
                .find(|cell| cell.capability == adapter)
                .unwrap_or_else(|| panic!("SCAIL {adapter} axis is represented"));
            assert_eq!(
                (cell.mlx, cell.candle),
                (Some(true), Some(true)),
                "SCAIL {adapter} is served through animate_character"
            );
        }

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

        let mage_edit = matrix
            .models
            .iter()
            .find(|row| row.id == "mage_flow_edit")
            .unwrap();
        for tier in ["q4", "q8"] {
            let cell = mage_edit
                .precision_tier
                .iter()
                .find(|cell| cell.capability == tier)
                .unwrap();
            assert_eq!(
                (cell.mlx, cell.candle),
                (Some(true), Some(true)),
                "Mage Edit precision must use a structurally valid edit probe"
            );
        }
        for shape in ["reference", "multiReference"] {
            let cell = mage_edit
                .conditioning_shape
                .iter()
                .find(|cell| cell.capability == shape)
                .unwrap();
            assert_eq!(
                (cell.mlx, cell.candle),
                (Some(true), Some(true)),
                "Mage Edit conditioning must retain its required primary source"
            );
        }

        let chatterbox = matrix
            .models
            .iter()
            .find(|row| row.id == "chatterbox_tts")
            .unwrap();
        for shape in ["referenceAudio", "voiceEmbedding"] {
            let cell = chatterbox
                .conditioning_shape
                .iter()
                .find(|cell| cell.capability == shape)
                .expect("Chatterbox descriptor conditioning is represented");
            assert_eq!((cell.mlx, cell.candle), (Some(false), Some(true)));
            assert!(cell.preserved_candle_only);
        }

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
                assert_eq!((cell.mlx, cell.candle), (Some(true), Some(true)));
                assert!(
                    cell.parity_obligation.is_none(),
                    "fulfilled SANA img2img parity must carry no open obligation"
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
        assert!(generator_descriptors(&original, "sana_1600m")[0]
            .conditioning
            .iter()
            .any(|kind| kind == "reference"));
        assert!(!generator_descriptors(&missing_descriptor, "sana_1600m")[0]
            .conditioning
            .iter()
            .any(|kind| kind == "reference"));

        let manifest: ManifestRoot = serde_json::from_str(&strip_jsonc_comments(MANIFEST)).unwrap();
        let flux2 = manifest
            .models
            .iter()
            .find(|model| model.id == "flux2_dev")
            .unwrap();
        let candle = runtime_facts(CANDLE_RUNTIME_FACTS, "candle").unwrap();
        let original_cell =
            operation_cell(flux2, "prompt_enhancement", &original, &candle).unwrap();
        assert_eq!(
            (original_cell.mlx, original_cell.candle),
            (Some(true), Some(true))
        );

        let mut no_enhancement: Value = serde_json::from_str(MLX_RUNTIME_FACTS).unwrap();
        let engine = original.model_mappings.get("flux2_dev").unwrap();
        no_enhancement["snapshot"]["generator_capabilities"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|descriptor| descriptor["id"].as_str() == Some(engine))
            .unwrap()["supports_prompt_enhancement"] = Value::Bool(false);
        let no_enhancement =
            runtime_facts(&serde_json::to_string(&no_enhancement).unwrap(), "mlx").unwrap();
        let mutated_cell =
            operation_cell(flux2, "prompt_enhancement", &no_enhancement, &candle).unwrap();
        assert_eq!(
            (mutated_cell.mlx, mutated_cell.candle),
            (Some(false), Some(true))
        );
        assert!(mutated_cell.preserved_candle_only);
        assert!(mutated_cell.parity_obligation.is_none());

        let sana = manifest
            .models
            .iter()
            .find(|model| model.id == "sana_1600m")
            .unwrap();
        assert!(manifest_artifact_tier_support(sana, "bf16", "mlx"));
        assert!(!manifest_artifact_tier_support(sana, "bf16", "candle"));
        let dense = precision_cell(sana, "bf16", &original, &candle).unwrap();
        assert_eq!((dense.mlx, dense.candle), (Some(true), Some(true)));

        // Removing the Candle descriptor loses Candle dense support even though the manifest still
        // contains a macOS bf16 artifact. Conversely, removing the MLX descriptor does not erase the
        // exact macOS artifact. These mutations guard both independent sides of the union.
        let mut no_candle_descriptor = candle.clone();
        no_candle_descriptor.model_mappings.remove("sana_1600m");
        no_candle_descriptor
            .video_model_mappings
            .retain(|mapping| mapping.model_id != "sana_1600m");
        let dense = precision_cell(sana, "bf16", &original, &no_candle_descriptor).unwrap();
        assert_eq!(dense.candle, Some(false));

        let mut no_mlx_descriptor = original.clone();
        no_mlx_descriptor.model_mappings.remove("sana_1600m");
        no_mlx_descriptor
            .video_model_mappings
            .retain(|mapping| mapping.model_id != "sana_1600m");
        let dense = precision_cell(sana, "bf16", &no_mlx_descriptor, &candle).unwrap();
        assert_eq!(dense.mlx, Some(true));

        // Neither authority bypasses production routing: a snapshot without image dispatch cannot
        // claim the precision cell even when both its descriptor and manifest artifact remain.
        let mut no_candle_dispatch = candle.clone();
        no_candle_dispatch
            .worker_capabilities
            .retain(|capability| capability != "image_generate");
        let dense = precision_cell(sana, "bf16", &original, &no_candle_dispatch).unwrap();
        assert_eq!(dense.candle, Some(false));

        let scail = manifest
            .models
            .iter()
            .find(|model| model.id == "scail2_14b")
            .unwrap();
        let adapter = adapter_cell(scail, "lora", &original, &candle).unwrap();
        assert_eq!((adapter.mlx, adapter.candle), (Some(true), Some(true)));
        let mut no_candle_animation = candle.clone();
        no_candle_animation.video_model_mappings.retain(|mapping| {
            mapping.model_id != "scail2_14b" || mapping.mode != "animate_character"
        });
        let adapter = adapter_cell(scail, "lora", &original, &no_candle_animation).unwrap();
        assert_eq!(
            adapter.candle,
            Some(true),
            "replace_person independently preserves SCAIL LoRA support"
        );

        let mut bad_mlx = runtime_facts(MLX_RUNTIME_FACTS, "mlx").unwrap();
        let mut bad_candle = runtime_facts(CANDLE_RUNTIME_FACTS, "candle").unwrap();
        for facts in [&mut bad_mlx, &mut bad_candle] {
            facts.model_mappings.insert(
                "flux2_dev".to_owned(),
                "missing-from-both-native-catalogs".to_owned(),
            );
        }
        assert!(validate_runtime_pair(&bad_mlx, &bad_candle).is_err());

        let mut wrong_video_shape = runtime_facts(MLX_RUNTIME_FACTS, "mlx").unwrap();
        wrong_video_shape
            .video_model_mappings
            .iter_mut()
            .find(|mapping| {
                mapping.model_id == "wan_2_2_i2v_14b" && mapping.mode == "image_to_video"
            })
            .unwrap()
            .engine_ids = vec!["wan2_2_t2v_14b".to_owned()];
        assert!(validate_runtime_pair(&wrong_video_shape, &candle).is_err());
    }

    #[test]
    fn every_routed_shipped_video_mode_has_an_exact_descriptor_join() {
        let manifest: ManifestRoot = serde_json::from_str(&strip_jsonc_comments(MANIFEST)).unwrap();
        let matrix = backend_capability_matrix().unwrap();
        let mlx = runtime_facts(MLX_RUNTIME_FACTS, "mlx").unwrap();
        let candle = runtime_facts(CANDLE_RUNTIME_FACTS, "candle").unwrap();
        for model in manifest
            .models
            .iter()
            .filter(|model| model.model_type == "video")
        {
            let row = matrix.models.iter().find(|row| row.id == model.id).unwrap();
            for mode in VIDEO_UI_MODES {
                let job = super::super::canonical_video_route_probe(&model.id, mode).unwrap();
                for facts in [&mlx, &candle] {
                    if backend_supports(&job, facts).unwrap() {
                        let descriptors = native_video_route_descriptors(facts, &model.id, mode);
                        assert!(
                            !descriptors.is_empty(),
                            "{} routed {:?}/{mode:?} without a generated video descriptor join",
                            facts.snapshot.backend,
                            model.id
                        );
                        assert!(row
                            .operation_and_mode
                            .iter()
                            .any(|cell| cell.capability == *mode));
                    }
                }
            }
        }

        for (model, supported_mode, rejected_mode, required_conditioning) in [
            ("wan_2_2_t2v_14b", "text_to_video", "image_to_video", None),
            (
                "wan_2_2_i2v_14b",
                "image_to_video",
                "text_to_video",
                Some("reference"),
            ),
        ] {
            let supported =
                super::super::canonical_video_route_probe(model, supported_mode).unwrap();
            let rejected = super::super::canonical_video_route_probe(model, rejected_mode).unwrap();
            assert!(backend_supports(&supported, &mlx).unwrap());
            assert!(!backend_supports(&rejected, &mlx).unwrap());
            let descriptors = native_video_route_descriptors(&mlx, model, supported_mode);
            assert_eq!(descriptors.len(), 1);
            if let Some(shape) = required_conditioning {
                assert!(descriptors[0].conditioning.iter().any(|item| item == shape));
            } else {
                assert!(descriptors[0].conditioning.is_empty());
            }
        }
    }

    #[test]
    fn deleting_any_routed_video_mapping_fails_closed() {
        let mlx = runtime_facts(MLX_RUNTIME_FACTS, "mlx").unwrap();
        let candle = runtime_facts(CANDLE_RUNTIME_FACTS, "candle").unwrap();
        for original in [&mlx, &candle] {
            for index in 0..original.video_model_mappings.len() {
                let mapping = &original.video_model_mappings[index];
                let job =
                    super::super::canonical_video_route_probe(&mapping.model_id, &mapping.mode)
                        .unwrap();
                assert!(backend_supports(&job, original).unwrap());
                let mut mutated = original.clone();
                mutated.video_model_mappings.remove(index);
                assert!(routed_cell(
                    &mapping.mode,
                    &mapping.model_id,
                    "video",
                    &job,
                    if original.snapshot.backend == "mlx" {
                        &mutated
                    } else {
                        &mlx
                    },
                    if original.snapshot.backend == "candle" {
                        &mutated
                    } else {
                        &candle
                    },
                    true,
                )
                .is_err());
            }
        }
    }

    #[test]
    fn drift_digest_covers_video_audio_and_utility_owners() {
        let sources = source_digests();
        for required in [
            "apiGeneration",
            "apiContractEntry",
            "apiDto",
            "workerVideoDispatch",
            "workerVideoWan",
            "workerVideoVace",
            "workerVideoLtx",
            "workerVideoSvd",
            "workerVideoBernini",
            "workerVideoScail2",
            "workerVideoKreaRealtime",
            "workerVideoMochi",
            "workerVideoCandle",
            "workerVideoSeedvr2",
            "workerAudioDispatch",
            "workerUtilityDispatch",
            "webSimpleJobs",
            "webSimpleVideo",
            "webVideoValidation",
            "webVideoStudio",
            "webSimpleAudio",
            "webSimpleAudioStudio",
            "webAudioStudio",
            "webUpscaleEngines",
            "webVideoUpscale",
        ] {
            assert!(
                sources.contains_key(required),
                "missing source digest {required}"
            );
        }
    }

    #[test]
    fn every_descriptor_axis_for_a_shipped_generator_has_a_matrix_cell() {
        let matrix = backend_capability_matrix().unwrap();
        let manifest: ManifestRoot = serde_json::from_str(&strip_jsonc_comments(MANIFEST)).unwrap();
        let mlx = runtime_facts(MLX_RUNTIME_FACTS, "mlx").unwrap();
        let candle = runtime_facts(CANDLE_RUNTIME_FACTS, "candle").unwrap();

        for model in &manifest.models {
            let row = matrix.models.iter().find(|row| row.id == model.id).unwrap();
            let descriptors: Vec<_> = [&mlx, &candle]
                .into_iter()
                .flat_map(|facts| generator_descriptors(facts, &model.id))
                .collect();
            if descriptors.is_empty() {
                continue;
            }
            let conditioning: BTreeSet<&str> = row
                .conditioning_shape
                .iter()
                .map(|cell| cell.capability.as_str())
                .collect();
            let precision: BTreeSet<&str> = row
                .precision_tier
                .iter()
                .map(|cell| cell.capability.as_str())
                .collect();
            let adapters: BTreeSet<&str> = row
                .user_adapters
                .iter()
                .map(|cell| cell.capability.as_str())
                .collect();
            for descriptor in descriptors {
                for shape in &descriptor.conditioning {
                    assert!(
                        conditioning.contains(shape.as_str()),
                        "{} descriptor conditioning {shape} has no matrix cell",
                        model.id
                    );
                }
                for tier in &descriptor.supported_quants {
                    assert!(
                        precision.contains(tier.as_str()),
                        "{} descriptor precision {tier} has no matrix cell",
                        model.id
                    );
                }
                if descriptor.supports_lora {
                    assert!(
                        adapters.contains("lora"),
                        "{} loses descriptor LoRA",
                        model.id
                    );
                }
                if descriptor.supports_lokr {
                    assert!(
                        adapters.contains("lokr"),
                        "{} loses descriptor LoKr",
                        model.id
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_probe_validation_rejects_missing_operation_inputs() {
        assert!(probe_job(
            JobType::ImageEdit,
            "mage_flow_edit",
            json!({ "mode": "edit_image", "referenceAssetIds": ["probe"] })
        )
        .is_err());
        assert!(probe_job(
            JobType::VideoGenerate,
            "ltx_2_3",
            json!({ "mode": "first_last_frame", "sourceAssetId": "probe" })
        )
        .is_err());
        assert!(probe_job(
            JobType::AudioGenerate,
            "chatterbox_tts",
            json!({ "referenceAudioAssetId": "probe" })
        )
        .is_err());
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
