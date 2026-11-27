//! PIPE-18 route reconciliation (SW-6931, SW-F331 E1, SW-6983): a metadata-only join, not a
//! capability authority and not an execution or device-fit claim.
//!
//! The two sides are written independently:
//!
//! * **Product side (authored):** the public operations and authored conditioning carriers of
//!   `config/manifests/builtin.models.jsonc`, the per-backend routing admission of
//!   `config/manifests/registry/route-policy.json`, the vector providers each product names, the
//!   native job arms of `crates/sceneworks-worker/src/utility_dispatch.rs`, and the production
//!   claim gate with its bespoke worker lanes. No field the core build copies out of a capability
//!   export is read on this side.
//! * **Engine side:** the hardware-free `sga.capabilities/2` exports in
//!   `config/engine-capabilities/sga/{macos-mlx,candle-cuda,candle-cpu,ios-media,ios-llm-only}.json`.
//!
//! Every product route is joined by provider, mode, conditioning and output on each backend that
//! admits it. A route with neither a registry binding nor a bespoke worker lane is missing. An
//! approved backend exception may explain a missing cell; such cells are reported separately and
//! never dropped.
use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::contracts::{
    ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage, WorkerCapability,
    WorkerSnapshot, WorkerStatus,
};
use crate::registry::{route_bindings, ModelSpec};
use crate::sga_capabilities::{
    CapabilitiesExportV2, ExportedRouteV2, CANDLE_CPU_JSON, CANDLE_CUDA_JSON, IOS_LLM_ONLY_JSON,
    IOS_MEDIA_JSON, MACOS_MLX_JSON,
};

const DISPATCH: &str = include_str!("../../../../sceneworks-worker/src/utility_dispatch.rs");
const PRODUCT_MANIFEST: &str = include_str!("../../../../../config/manifests/builtin.models.jsonc");
const ROUTE_POLICY: &str =
    include_str!("../../../../../config/manifests/registry/route-policy.json");

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteCoverage {
    pub schema_version: u32,
    pub evidence_kind: &'static str,
    pub sources: BTreeMap<String, String>,
    pub jobs: Vec<JobLifecycle>,
    pub cells: Vec<RouteCell>,
    pub registries: Vec<RegistrySurface>,
    pub profile_exports: BTreeMap<String, Value>,
    pub model_aliases: Vec<Value>,
    /// Product routes with no registry binding and no bespoke lane, and no approved exception.
    pub missing: Vec<String>,
    /// Product routes with no binding that an approved backend exception explains.
    pub exceptions: Vec<ExceptionCell>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteCell {
    pub platform: String,
    pub backend: String,
    pub model: String,
    pub axis: &'static str,
    pub mode: String,
    pub conditioning: String,
    pub output: String,
    pub output_variants: Vec<String>,
    pub canonical_request: Option<Value>,
    pub expected_provider_ids: Vec<String>,
    pub providers: Vec<String>,
    pub binding: &'static str,
    pub owner: &'static str,
    pub entry_point: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExceptionCell {
    pub cell: String,
    pub exception: String,
    pub decision: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegistrySurface {
    pub platform: String,
    pub backend: String,
    pub surface: String,
    pub provider: String,
    pub descriptor: Option<Value>,
    pub owner: &'static str,
    pub public_cells: Vec<String>,
    pub disposition: &'static str,
    pub entry_point: String,
    pub callable_entry: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobLifecycle {
    pub job_type: String,
    pub lifecycle: &'static str,
    pub output: &'static str,
    pub owner: &'static str,
    pub entry_point: &'static str,
    /// Platforms whose worker advertises this job arm.
    pub advertised_by: Vec<String>,
}

/// One platform's engine registries, read from its canonical `sga.capabilities/2` export.
struct Engine {
    export: CapabilitiesExportV2,
    platform: String,
    backend: String,
    audio_backend: String,
    worker_capabilities: Vec<String>,
    worker: WorkerSnapshot,
}

impl Engine {
    fn parse(source: &str) -> Result<Self, String> {
        let export =
            CapabilitiesExportV2::from_json_bytes(source.as_bytes()).map_err(|e| e.to_string())?;
        let platform = export.platform.clone();
        let backend = export.backend.clone();
        let audio_backend = export
            .audio_backend
            .clone()
            .ok_or_else(|| format!("{} export has no audio_backend", export.feature_set))?;
        let worker_capabilities: Vec<String> =
            route_bindings::platform_worker_capabilities(&backend)
                .iter()
                .map(|cap| (*cap).to_owned())
                .collect();
        let capabilities = worker_capabilities
            .iter()
            .map(|value| {
                serde_json::from_value::<WorkerCapability>(json!(value)).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let worker = WorkerSnapshot {
            id: format!("route-coverage-{backend}"),
            platform: None,
            gpu_id: if backend == "mlx" { "mlx" } else { "0" }.to_owned(),
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
        };
        Ok(Self {
            export,
            platform,
            backend,
            audio_backend,
            worker_capabilities,
            worker,
        })
    }

    fn route(&self, id: &str) -> Option<&ExportedRouteV2> {
        self.export.routes.get(id)
    }

    fn audio_route(&self, id: &str) -> Option<&ExportedRouteV2> {
        self.export.audio_routes.get(id)
    }

    fn conditioning(route: &ExportedRouteV2) -> BTreeSet<&str> {
        route
            .conditioning
            .kinds
            .iter()
            .map(String::as_str)
            .collect()
    }

    /// The registered media generator the platform maps a public image model to.
    fn mapped_generator(&self, model: &str) -> Option<&ExportedRouteV2> {
        if let Some(id) = route_bindings::image_route_id(model, |candidate| {
            self.export.routes.contains_key(candidate)
        }) {
            return self.route(id);
        }
        route_bindings::bespoke_image_route_id(model).and_then(|id| self.route(id))
    }

    fn video_engines(&self, model: &str, mode: &str) -> Vec<String> {
        route_bindings::video_route_ids(&self.backend, model, mode)
            .iter()
            .map(|id| (*id).to_owned())
            .collect()
    }

    fn surface_ids(&self, surface: &str) -> Vec<String> {
        match surface {
            "generator_ids" => self.export.routes.keys().cloned().collect(),
            "audio_generator_ids" => self.export.audio_routes.keys().cloned().collect(),
            "trainer_ids" => self.export.trainers.keys().cloned().collect(),
            "captioner_ids" => self.export.captioners.keys().cloned().collect(),
            "image_embedder_ids" => self.export.embedders.image.keys().cloned().collect(),
            "text_embedder_ids" => self.export.embedders.text.keys().cloned().collect(),
            "text_llm_ids" => self.export.llm_models.keys().cloned().collect(),
            "transform_ids" => self.export.transforms.image.keys().cloned().collect(),
            "audio_transform_ids" => self.export.transforms.audio.keys().cloned().collect(),
            "audio_voice_embedder_ids" => self.export.embedders.voice.keys().cloned().collect(),
            "audio_transcriber_ids" => self.export.transcribers.keys().cloned().collect(),
            "audio_embedder_ids" => self.export.embedders.audio.keys().cloned().collect(),
            "snapshot_preparer_backends" => self.export.snapshot_preparer_backends.clone(),
            "audio_snapshot_preparer_backends" => {
                self.export.audio_snapshot_preparer_backends.clone()
            }
            _ => Vec::new(),
        }
    }

    fn has_surface_id(&self, surface: &str, id: &str) -> bool {
        match surface {
            "generator_ids" => self.export.routes.contains_key(id),
            "audio_generator_ids" => self.export.audio_routes.contains_key(id),
            "trainer_ids" => self.export.trainers.contains_key(id),
            "captioner_ids" => self.export.captioners.contains_key(id),
            "image_embedder_ids" => self.export.embedders.image.contains_key(id),
            "text_embedder_ids" => self.export.embedders.text.contains_key(id),
            "text_llm_ids" => self.export.llm_models.contains_key(id),
            "transform_ids" => self.export.transforms.image.contains_key(id),
            "audio_transform_ids" => self.export.transforms.audio.contains_key(id),
            "audio_voice_embedder_ids" => self.export.embedders.voice.contains_key(id),
            "audio_transcriber_ids" => self.export.transcribers.contains_key(id),
            "audio_embedder_ids" => self.export.embedders.audio.contains_key(id),
            "snapshot_preparer_backends" => self
                .export
                .snapshot_preparer_backends
                .iter()
                .any(|item| item == id),
            "audio_snapshot_preparer_backends" => self
                .export
                .audio_snapshot_preparer_backends
                .iter()
                .any(|item| item == id),
            _ => false,
        }
    }

    fn surface_descriptor(&self, surface: &str, id: &str) -> Option<Value> {
        match surface {
            "generator_ids" => self
                .export
                .routes
                .get(id)
                .and_then(|r| serde_json::to_value(r).ok()),
            "audio_generator_ids" => self
                .export
                .audio_routes
                .get(id)
                .and_then(|r| serde_json::to_value(r).ok()),
            "trainer_ids" => self
                .export
                .trainers
                .get(id)
                .and_then(|t| serde_json::to_value(t).ok()),
            "text_llm_ids" => self
                .export
                .llm_models
                .get(id)
                .and_then(|m| serde_json::to_value(m).ok()),
            _ => None,
        }
    }

    fn claims(&self, job: &JobSnapshot) -> bool {
        crate::jobs_store::worker_supports_job(&self.worker, job)
    }
}

const ALL_SURFACES: [&str; 14] = [
    "generator_ids",
    "audio_generator_ids",
    "trainer_ids",
    "captioner_ids",
    "image_embedder_ids",
    "text_embedder_ids",
    "text_llm_ids",
    "transform_ids",
    "audio_transform_ids",
    "audio_voice_embedder_ids",
    "audio_transcriber_ids",
    "audio_embedder_ids",
    "snapshot_preparer_backends",
    "audio_snapshot_preparer_backends",
];

fn probe(job_type: JobType, model: &str, payload: Value) -> JobSnapshot {
    let mut payload = payload.as_object().cloned().unwrap_or_default();
    payload.insert("model".into(), json!(model));
    payload
        .entry("prompt".to_owned())
        .or_insert_with(|| json!("probe"));
    JobSnapshot {
        id: "route-coverage-probe".to_owned(),
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
    }
}

fn field<'a>(spec: &'a ModelSpec, key: &str) -> Option<&'a Value> {
    spec.product.fields.get(key)
}

fn ui<'a>(spec: &'a ModelSpec, key: &str) -> Option<&'a Value> {
    field(spec, "ui").and_then(|ui| ui.get(key))
}

/// The public operations a product row authors.
fn operations(spec: &ModelSpec) -> Vec<String> {
    let mut operations: BTreeSet<String> = field(spec, "capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    if ui(spec, "img2img").and_then(Value::as_bool) == Some(true) {
        operations.insert("image_to_image".into());
    }
    operations.into_iter().collect()
}

/// The production request for one image operation (the API's canonical shape).
fn image_request(id: &str, operation: &str) -> Option<(JobType, Value)> {
    Some(match operation {
        "text_to_image" => (JobType::ImageGenerate, json!({ "mode": operation })),
        "edit_image" => (
            JobType::ImageEdit,
            json!({ "mode": operation, "sourceAssetId": "probe" }),
        ),
        "image_to_image" if id == "z_image_edit" => (
            JobType::ImageEdit,
            json!({ "mode": "edit_image", "sourceAssetId": "probe" }),
        ),
        "image_to_image" if id.starts_with("flux2_") => (
            JobType::ImageGenerate,
            json!({ "mode": operation, "referenceAssetId": "probe" }),
        ),
        "image_to_image" => (
            JobType::ImageGenerate,
            json!({ "mode": "text_to_image", "referenceAssetId": "probe" }),
        ),
        "character_image" => (
            JobType::ImageGenerate,
            json!({ "mode": operation, "referenceAssetId": "probe" }),
        ),
        "vqa" => (
            JobType::ImageVqa,
            json!({ "sourceAssetId": "probe", "question": "probe" }),
        ),
        "interleave" => (
            JobType::ImageInterleave,
            json!({ "sourceAssetIds": ["probe"] }),
        ),
        "image_inpaint" => (
            JobType::ImageEdit,
            json!({ "mode": "edit_image", "sourceAssetId": "probe", "maskAssetId": "probe-mask" }),
        ),
        "image_detail" => (
            JobType::ImageDetail,
            json!({ "sourceAssetId": "probe", "engine": "controlnet-tile" }),
        ),
        _ => return None,
    })
}

/// The descriptor conditioning that registers an image operation on a mapped generator.
fn image_operation_conditioning(operation: &str) -> &'static [&'static str] {
    match operation {
        "character_image" | "image_to_image" => &["reference", "multiReference"],
        _ => &[],
    }
}

/// A Candle request that lands on a bespoke production lane is served by that lane: the generic
/// text-to-image lane refuses every conditioning carrier.
fn candle_bespoke_lane(job: &JobSnapshot) -> Option<String> {
    use super::candle::{image_job_candle_lane, CandleImageLane};
    match image_job_candle_lane(job) {
        None | Some(CandleImageLane::TextToImage) => None,
        Some(lane) => Some(format!("{lane:?}")),
    }
}

fn conditioning_meaning(spec: &ModelSpec, mode: &str) -> &'static str {
    let id = spec.product.id.as_str();
    let family = spec.product.family.as_str();
    match mode {
        "control" => "structural ControlNet (pose/control image), not VAE initialization",
        "multiReference" => {
            "ordered multiple references; source retained for source-with-reference edit"
        }
        "image_inpaint" => "source image plus inpainting mask",
        "image_to_image" => "source-image VAE initialization and strength blend",
        "character_image" if family == "flux" && !id.starts_with("pulid") => {
            "XLabs IP-adapter image embedding, not VAE initialization"
        }
        "character_image" if id.starts_with("pulid") || id.starts_with("instantid") => {
            "identity reference (PuLID/InstantID), not VAE initialization"
        }
        "character_image" => {
            "reference-conditioned image; provider selector defines identity versus edit"
        }
        "edit_image" => {
            "source-image edit; provider selector defines VAE init versus reference encoding"
        }
        "image_to_svg" => "image prefix to native token driver",
        "first_last_frame" => "first/last endpoint keyframes",
        "image_to_video" => "first-frame reference",
        "vqa" | "interleave" => "source image(s) for native understanding",
        "ReferenceAudio" | "AudioEdit" | "AudioEditRegions" | "VoiceEmbedding" => {
            "audio conditioning (distinct reference/edit/voice carriers)"
        }
        _ if spec.product.model_type == "video" => {
            "video conditioning by mode (see canonical request and descriptor requirements)"
        }
        _ => "text-only prompt; see canonical request",
    }
}

fn output(spec: &ModelSpec, mode: &str) -> String {
    match mode {
        "vqa" => "text".into(),
        "interleave" => "ordered-text/image-document".into(),
        _ => match spec.product.model_type.as_str() {
            "vector" => "canonical-svg+derived-preview+terminal-metadata",
            "image" => "image (provider alpha/count contract retained)",
            "video" => "video (provider audio/count contract retained)",
            "audio" => "audio",
            _ => "structured-metadata",
        }
        .into(),
    }
}

fn output_variants(spec: &ModelSpec, mode: &str) -> Vec<String> {
    let kind = spec.product.model_type.as_str();
    if matches!(mode, "vqa" | "interleave") || (kind != "image" && kind != "video") {
        return vec![output(spec, mode)];
    }
    match spec.product.family.as_str() {
        "qwen-image-2-1" => vec![
            "RGB PNG; count".into(),
            "native straight-alpha RGBA PNG; transparentBackground; count".into(),
        ],
        "moss_image" => vec![
            "RGB PNG; one prompt × count".into(),
            "RGB PNGs; prompts × count with per-output seeds".into(),
        ],
        "minimax-h3" => vec!["joint video+synchronized stereo audio".into()],
        "ltx-video" => vec!["video".into(), "joint video+audio".into()],
        _ if kind == "video" => vec!["video; source audio passthrough where supported".into()],
        _ => vec!["RGB image(s); count; editor/detail retains source alpha where supported".into()],
    }
}

fn owner(spec: &ModelSpec) -> Result<&'static str, String> {
    let product = &spec.product;
    Ok(match product.model_type.as_str() {
        "vector" => "ENG-7/APP-5",
        "audio" => "ENG-14/APP-6",
        "image" | "video" => match product.family.as_str() {
            "flux" | "pulid" => "ENG-1a/APP-5",
            "chroma" | "flux2-dev" | "flux2-klein" | "lens" | "anima" | "sd3" | "sana"
            | "boogu" | "qwen-image" | "qwen-image-2-1" | "z-image" | "kolors" | "sdxl"
            | "instantid" => "ENG-2/ENG-5/APP-5",
            "wan-video" | "scail2" | "svd" | "mochi" => "ENG-3/ENG-5/APP-6",
            "krea_2" => "ENG-4b/APP-5",
            "flux3" | "moss_image" | "iris" | "ideogram" | "supra2" => "ENG-5/APP-5",
            "ltx-video" | "minimax-h3" | "minimax-h3-image" | "mage-flow" | "sensenova-u1"
            | "bernini" | "krea-realtime" => "ENG-6/APP-5/APP-6",
            other => return Err(format!("no migration owner for {}/{other}", product.id)),
        },
        other => {
            return Err(format!(
                "unclassified model type {other} for {}",
                product.id
            ))
        }
    })
}

fn registry_owner(
    surface: &str,
    provider: &str,
    export: &CapabilitiesExportV2,
) -> Result<&'static str, String> {
    if surface.starts_with("audio_") {
        return Ok("ENG-14");
    }
    Ok(match surface {
        "text_llm_ids" if provider.contains("starvector") => "ENG-7/APP-5",
        "text_llm_ids" => "ENG-14",
        "trainer_ids" => "ENG-8/ENG-9",
        "captioner_ids" | "image_embedder_ids" | "text_embedder_ids" => "ENG-11",
        "transform_ids" => "ENG-6/ENG-10",
        "snapshot_preparer_backends" => "ENG-9",
        "generator_ids" => {
            let family = export
                .routes
                .get(provider)
                .map(|r| r.family.as_str())
                .ok_or_else(|| format!("generator {provider} has no route family"))?;
            match family {
                "flux" | "pulid" => "ENG-1a",
                "chroma" | "flux2" | "lens" | "anima" | "sd3" | "stable-diffusion-3" | "sana"
                | "boogu" | "qwen-image" | "qwen-image-2-1" | "z-image" | "kolors" | "sdxl"
                | "instantid" => "ENG-2/ENG-5",
                "wan" | "scail2" | "svd" | "mochi" => "ENG-3/ENG-5",
                "krea_2" => "ENG-4b",
                "flux3" | "moss_image" | "iris" | "ideogram" | "supra2" => "ENG-5",
                "ltx" | "minimax_h3" | "minimax_h3_image" | "mage_flow" | "sensenova-u1"
                | "bernini" | "krea_realtime" | "seedvr2" => "ENG-6",
                "starvector" => "ENG-7/APP-5",
                other => return Err(format!("unowned native generator {provider}/{other}")),
            }
        }
        _ => return Err(format!("unowned registry surface {surface}")),
    })
}

fn registry_entry(platform: &str, surface: &str, provider: &str) -> String {
    let (lane, load, run) = match surface {
        "generator_ids" => ("media()", "load", "generate(GenerationRequest)"),
        "transform_ids" => ("media()", "load_transform", "transform(TransformRequest)"),
        "trainer_ids" => ("media()", "load_trainer", "train(TrainingRequest)"),
        "captioner_ids" => ("media()", "load_captioner", "caption(CaptionRequest)"),
        "image_embedder_ids" => ("media()", "load_image_embedder", "embed_image"),
        "text_embedder_ids" => ("media()", "load_text_embedder", "embed_text"),
        "text_llm_ids" => ("text()", "load_textllm", "generate/stream"),
        "audio_generator_ids" => ("audio()", "load", "generate(GenerationRequest)"),
        "audio_transform_ids" => ("audio()", "load_audio_transform", "transform"),
        "audio_voice_embedder_ids" => ("audio()", "load_voice_embedder", "embed"),
        "audio_transcriber_ids" => ("audio()", "load_transcriber", "transcribe"),
        "audio_embedder_ids" => ("audio()", "load_audio_embedder", "embed"),
        "snapshot_preparer_backends" => ("preparers()", "prepare_snapshot", "snapshot preparation"),
        "audio_snapshot_preparer_backends" => (
            "audio_preparers()",
            "prepare_snapshot",
            "snapshot preparation",
        ),
        _ => return format!("unclassified registry surface: {surface}"),
    };
    format!("runtime-{platform}::catalog().{lane}.{load}({provider}) -> {run}")
}

fn lifecycle(job: JobType) -> Result<(&'static str, &'static str, &'static str), String> {
    use JobType::*;
    Ok(match &job {
        ImageGenerate | ImageEdit | ImageDetail | ImageUpscale => ("image", "image", "APP-5"),
        VectorGenerate => (
            "vector-token",
            "canonical-svg+derived-preview+terminal-metadata",
            "APP-5",
        ),
        ImageVqa => ("understanding", "text", "APP-6"),
        ImageInterleave => (
            "conditional-generation",
            "ordered-text/image-document",
            "APP-6",
        ),
        VideoGenerate | VideoExtend | VideoBridge | PersonReplace | VideoUpscale => {
            ("video", "video+optional-audio", "APP-6")
        }
        AudioGenerate => ("audio", "audio", "ENG-14/APP-6"),
        LoraTrain | ControlTraining => (
            "training",
            "adapter/checkpoint+training-metadata",
            "ENG-8/APP-6",
        ),
        ModelDownload | LoraDownload => ("installer", "installed-model-metadata", "APP-6"),
        ModelImport | LoraImport | ModelConvert => {
            ("import/conversion", "model/adapter-metadata", "APP-6")
        }
        DatasetParquetImport => ("io", "dataset-items", "APP-6"),
        FrameExtract => ("media-utility", "image-assets", "APP-6"),
        TimelineExport => ("media-utility", "video", "APP-6"),
        DatasetUpscale => ("image-utility", "dataset-images", "APP-6"),
        DatasetSubjectMask | ImageSegment => ("vision-utility", "mask", "APP-6"),
        ImageDepth => ("vision-utility", "float32-npy-depth+preview", "APP-6"),
        PersonDetect | PersonTrack | PoseDetect | KpsExtract | DatasetAnalysis
        | CatalogAnalysis | DatasetFaceAnalysis | FaceLikenessCompare => {
            ("vision-utility", "structured-metadata", "APP-6")
        }
        TrainingCaption | PromptRefine | DatasetBatchDescribe => {
            ("language-utility", "text+metadata", "APP-6")
        }
        Placeholder => ("utility", "placeholder-result", "APP-6"),
        Unknown(name) => return Err(format!("unclassified job type {name}")),
    })
}

/// The approved backend exception, if any, naming this product cell.
fn approved_exception(model: &str, axis: &str, value: &str) -> Option<(String, String)> {
    let key = format!(
        "models/{model}/{}/{value}",
        if axis == "operation" {
            "operationAndMode"
        } else {
            "conditioningShape"
        }
    );
    crate::registry::approved_backend_exceptions()
        .iter()
        .find(|record| record.cells.contains(&key))
        .map(|record| (key, record.id.clone()))
}

/// Does the product route this public cell on `engine`'s backend? Image and video cells follow
/// the authored routing predicates for the canonical request: the generic lanes (gated by the
/// authored `route-policy.json`) and every bespoke lane. Vector cells need an available authored
/// provider; audio follows the authored platform scope.
fn product_admits(
    spec: &ModelSpec,
    engine: &Engine,
    job: Option<&JobSnapshot>,
) -> Result<bool, String> {
    let backend = engine.backend.as_str();
    Ok(match (spec.product.model_type.as_str(), job) {
        ("video", Some(job)) => {
            if backend == "mlx" {
                super::mlx::video_job_is_mlx_eligible(job)
            } else {
                super::candle::video_job_is_candle_eligible(job)
            }
        }
        ("image", Some(job)) => {
            if backend == "mlx" {
                super::gaps::job_is_any_mlx_eligible(job)
            } else {
                super::gaps::job_is_any_candle_eligible(job)
            }
        }
        ("vector", _) => field(spec, "vector")
            .and_then(|vector| vector["providers"].get(backend))
            .and_then(|provider| provider["available"].as_bool())
            .unwrap_or(false),
        ("audio", _) => {
            backend == "mlx" || field(spec, "macOnly").and_then(Value::as_bool) != Some(true)
        }
        _ => false,
    })
}

/// An unbound product cell is missing unless an approved backend exception explains it. The
/// register records deliberate MLX-only cells, so it can only explain a Candle-side gap; an MLX
/// loss is always missing.
fn record_unbound(
    missing: &mut Vec<String>,
    exceptions: &mut Vec<ExceptionCell>,
    (model, axis, value): (&str, &str, &str),
    candle: bool,
    key: String,
) {
    match approved_exception(model, axis, value).filter(|_| candle) {
        Some((decision, exception)) => exceptions.push(ExceptionCell {
            cell: key,
            exception,
            decision,
        }),
        None => missing.push(key),
    }
}

struct Binding {
    providers: Vec<String>,
    kind: &'static str,
    entry_point: String,
    expected: Vec<String>,
    request: Option<Value>,
}

impl Binding {
    fn none(expected: Vec<String>, request: Option<Value>) -> Self {
        Self {
            providers: Vec::new(),
            kind: "unbound",
            entry_point: "no registry binding and no bespoke lane".into(),
            expected,
            request,
        }
    }
}

fn request_json(job: &JobSnapshot) -> Value {
    json!({ "jobType": job.job_type, "payload": job.payload })
}

/// The canonical production request for one authored image cell.
fn image_probe(spec: &ModelSpec, axis: &str, mode: &str) -> Option<JobSnapshot> {
    let id = spec.product.id.as_str();
    let operations = operations(spec);
    let has = |op: &str| operations.iter().any(|o| o == op);
    let (job_type, payload) = if axis == "conditioning" {
        match mode {
            "control" => {
                let op = ["text_to_image", "character_image", "edit_image"]
                    .into_iter()
                    .find(|op| has(op))?;
                let (job_type, mut payload) = image_request(id, op)?;
                payload["advanced"] = json!({ "poses": [{}] });
                (job_type, payload)
            }
            "multiReference" if has("character_image") => {
                let (job_type, mut payload) = image_request(id, "character_image")?;
                payload.as_object_mut()?.remove("referenceAssetId");
                payload["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
                (job_type, payload)
            }
            "multiReference" if has("edit_image") => {
                let (job_type, mut payload) = image_request(id, "edit_image")?;
                payload["sourceAssetId"] = json!("probe-source");
                payload["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
                (job_type, payload)
            }
            "multiReference" => (
                JobType::ImageEdit,
                json!({
                    "mode": "edit_image",
                    "referenceAssetIds": ["probe-a", "probe-b"],
                    "modelManifestEntry": Value::Object(spec.product.fields.clone().into_iter().collect()),
                }),
            ),
            _ => return None,
        }
    } else {
        image_request(id, mode)?
    };
    Some(probe(job_type, id, payload))
}

fn bind_image(
    spec: &ModelSpec,
    engine: &Engine,
    axis: &str,
    mode: &str,
    job: &JobSnapshot,
) -> Binding {
    let id = spec.product.id.as_str();
    let mapped = engine.mapped_generator(id);
    let expected: Vec<String> = mapped
        .map(|route| vec![route.route_id.clone()])
        .unwrap_or_default();
    if !engine.claims(job) {
        return Binding::none(expected, Some(request_json(job)));
    }
    let registered = mapped.is_some_and(|route| {
        let kinds = Engine::conditioning(route);
        let required: &[&str] = if axis == "conditioning" {
            &[]
        } else {
            image_operation_conditioning(mode)
        };
        let shape_ok = axis != "conditioning" || kinds.contains(mode);
        shape_ok && (required.is_empty() || required.iter().any(|kind| kinds.contains(kind)))
    });
    if registered {
        return Binding {
            providers: expected.clone(),
            kind: "registry",
            entry_point: if matches!(mode, "vqa" | "interleave") {
                format!("crates/sceneworks-worker/src/sensenova_jobs.rs::run_{mode}_job")
            } else {
                "crates/sceneworks-worker/src/image_jobs.rs".into()
            },
            expected,
            request: Some(request_json(job)),
        };
    }
    let bespoke = if engine.backend == "candle" {
        candle_bespoke_lane(job)
    } else {
        let carrier = job.payload.contains_key("referenceAssetId")
            || job.payload.contains_key("referenceAssetIds")
            || job.payload.contains_key("sourceAssetId")
            || job.payload.contains_key("advanced");
        let strict_control = mode != "control"
            || ui(spec, "controlModes")
                .and_then(Value::as_array)
                .is_some_and(|modes| modes.iter().any(|m| m == "pose"));
        (carrier && strict_control).then(|| "MlxBespoke".to_owned())
    };
    match bespoke {
        Some(lane) => Binding {
            providers: vec![format!("{id}::{lane}")],
            kind: "bespoke-lane",
            entry_point: if engine.backend == "candle" {
                "crates/sceneworks-core/src/jobs_store/routing/candle.rs::image_job_candle_lane -> crates/sceneworks-worker/src/image_jobs/candle.rs".into()
            } else {
                "crates/sceneworks-core/src/jobs_store/routing/mlx.rs::image_job_is_mlx_eligible -> crates/sceneworks-worker/src/image_jobs/mlx.rs".into()
            },
            expected,
            request: Some(request_json(job)),
        },
        None => Binding::none(expected, Some(request_json(job))),
    }
}

fn bind_video(spec: &ModelSpec, engine: &Engine, mode: &str, job: &JobSnapshot) -> Binding {
    let id = spec.product.id.as_str();
    let engines = engine.video_engines(id, mode);
    let registered: Vec<&ExportedRouteV2> =
        engines.iter().filter_map(|e| engine.route(e)).collect();
    let kinds: BTreeSet<&str> = registered
        .iter()
        .flat_map(|route| Engine::conditioning(route))
        .collect();
    let requirements_met = super::video_mode_conditioning_requirements(mode)
        .iter()
        .all(|group| group.iter().any(|kind| kinds.contains(kind)));
    let bound = !registered.is_empty()
        && registered.len() == engines.len()
        && requirements_met
        && engine.claims(job);
    if bound {
        Binding {
            providers: engines.clone(),
            kind: "registry",
            entry_point: "crates/sceneworks-worker/src/video_jobs/mod.rs".into(),
            expected: engines,
            request: Some(request_json(job)),
        }
    } else {
        Binding::none(engines, Some(request_json(job)))
    }
}

fn bind_vector(spec: &ModelSpec, engine: &Engine, mode: &str) -> Result<Binding, String> {
    let id = spec.product.id.as_str();
    let llm_provider = field(spec, "vector")
        .and_then(|vector| vector["providers"][engine.backend.as_str()]["id"].as_str())
        .ok_or_else(|| format!("{id} names no {} vector provider", engine.backend))?
        .to_owned();
    let job = probe(
        JobType::VectorGenerate,
        id,
        json!({ "mode": mode, "sourceAssetId": "probe" }),
    );
    let route_ok = engine.route(id).is_some_and(|route| {
        route.modality == "vector" && route.modes.iter().any(|candidate| candidate == mode)
    });
    let llm_ok = engine.has_surface_id("text_llm_ids", &llm_provider);
    let bound = route_ok && llm_ok && engine.claims(&job);
    Ok(if bound {
        Binding {
            providers: vec![id.to_owned(), llm_provider.clone()],
            kind: "registry",
            entry_point: "crates/sceneworks-worker/src/vector_jobs.rs::run_vector_job".into(),
            expected: vec![llm_provider],
            request: Some(request_json(&job)),
        }
    } else {
        Binding::none(vec![llm_provider], Some(request_json(&job)))
    })
}

const AUDIO_SURFACES: [&str; 5] = [
    "audio_generator_ids",
    "audio_transform_ids",
    "audio_voice_embedder_ids",
    "audio_transcriber_ids",
    "audio_embedder_ids",
];

fn bind_audio(spec: &ModelSpec, engine: &Engine, axis: &str, mode: &str) -> Binding {
    let id = spec.product.id.as_str();
    let surface = AUDIO_SURFACES
        .iter()
        .find(|surface| engine.has_surface_id(surface, id));
    let carrier_ok = axis != "conditioning"
        || engine
            .audio_route(id)
            .map(Engine::conditioning)
            .is_some_and(|kinds| {
                let mut camel = mode.to_owned();
                camel[..1].make_ascii_lowercase();
                kinds.contains(camel.as_str())
            })
        || surface.is_some_and(|s| *s != "audio_generator_ids");
    match surface {
        Some(surface) if carrier_ok => Binding {
            providers: vec![id.to_owned()],
            kind: "registry",
            entry_point: format!(
                "crates/sceneworks-worker/src/audio_jobs.rs <- {}",
                registry_entry(&engine.platform, surface, id)
            ),
            expected: vec![id.to_owned()],
            request: None,
        },
        _ => Binding::none(vec![id.to_owned()], None),
    }
}

/// The product cells one public model authors: `(axis, value)`.
fn product_cells(spec: &ModelSpec) -> Vec<(&'static str, String)> {
    let mut cells: Vec<(&'static str, String)> = operations(spec)
        .into_iter()
        .map(|operation| ("operation", operation))
        .collect();
    match spec.product.model_type.as_str() {
        "image" => {
            if ui(spec, "controlModes")
                .and_then(Value::as_array)
                .is_some_and(|modes| !modes.is_empty())
            {
                cells.push(("conditioning", "control".into()));
            }
            if ui(spec, "multiReference").and_then(Value::as_bool) == Some(true)
                || ui(spec, "sourceWithMultiReference").and_then(Value::as_bool) == Some(true)
            {
                cells.push(("conditioning", "multiReference".into()));
            }
        }
        "audio" => {
            cells.push(("operation", "audio".into()));
            for kind in field(spec, "audio")
                .and_then(|audio| audio["conditioning"].as_array())
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                cells.push(("conditioning", kind.to_owned()));
            }
        }
        _ => {}
    }
    cells
}

/// Join every authored product route to the canonical `sga.capabilities/2` exports and bespoke
/// lanes. Metadata only: no model or backend is constructed, allocated or run.
pub fn generation_route_coverage() -> Result<RouteCoverage, String> {
    reconcile_with_profiles(
        MACOS_MLX_JSON,
        CANDLE_CUDA_JSON,
        IOS_MEDIA_JSON,
        IOS_LLM_ONLY_JSON,
        CANDLE_CPU_JSON,
    )
}

#[cfg(test)]
fn reconcile(mlx_source: &str, candle_source: &str) -> Result<RouteCoverage, String> {
    reconcile_with_profiles(
        mlx_source,
        candle_source,
        IOS_MEDIA_JSON,
        IOS_LLM_ONLY_JSON,
        CANDLE_CPU_JSON,
    )
}

fn reconcile_with_profiles(
    mlx_source: &str,
    candle_source: &str,
    ios_media_source: &str,
    ios_llm_only_source: &str,
    candle_cpu_source: &str,
) -> Result<RouteCoverage, String> {
    let engines = [Engine::parse(mlx_source)?, Engine::parse(candle_source)?];
    let ios_media = CapabilitiesExportV2::from_json_bytes(ios_media_source.as_bytes())
        .map_err(|e| e.to_string())?;
    let ios_llm_only = CapabilitiesExportV2::from_json_bytes(ios_llm_only_source.as_bytes())
        .map_err(|e| e.to_string())?;
    let candle_cpu = CapabilitiesExportV2::from_json_bytes(candle_cpu_source.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut sources = BTreeMap::new();
    for (name, source) in [
        ("productManifest", PRODUCT_MANIFEST),
        ("productRoutePolicy", ROUTE_POLICY),
        ("jobDispatcher", DISPATCH),
        ("sgaMacosMlx", mlx_source),
        ("sgaCandleCuda", candle_source),
        ("sgaCandleCpu", candle_cpu_source),
        ("sgaIosMedia", ios_media_source),
        ("sgaIosLlmOnly", ios_llm_only_source),
    ] {
        sources.insert(name.to_owned(), format!("{:x}", Sha256::digest(source)));
    }
    let mut cells = Vec::new();
    let mut missing = Vec::new();
    let mut exceptions = Vec::new();
    let public: Vec<&ModelSpec> = crate::registry::builtin_model_specs()
        .filter(|spec| field(spec, "componentOnly").and_then(Value::as_bool) != Some(true))
        .filter(|spec| {
            matches!(
                spec.product.model_type.as_str(),
                "image" | "video" | "vector" | "audio"
            )
        })
        .collect();
    for spec in &public {
        let id = spec.product.id.as_str();
        for (axis, value) in product_cells(spec) {
            let job = match spec.product.model_type.as_str() {
                "image" => match image_probe(spec, axis, &value) {
                    Some(job) => Some(job),
                    None => continue,
                },
                "video" => Some(super::canonical_video_route_probe(id, &value)?),
                _ => None,
            };
            let mut admitted_anywhere = false;
            for engine in &engines {
                if !product_admits(spec, engine, job.as_ref())? {
                    if engine.backend == "candle" {
                        if let Some((decision, exception)) = approved_exception(id, axis, &value) {
                            exceptions.push(ExceptionCell {
                                cell: format!(
                                    "{}/{}/{id}/{axis}/{value}",
                                    engine.platform, engine.backend
                                ),
                                exception,
                                decision: format!("not routed on Candle: {decision}"),
                            });
                        }
                    }
                    continue;
                }
                admitted_anywhere = true;
                let binding = match (spec.product.model_type.as_str(), job.as_ref()) {
                    ("image", Some(job)) => bind_image(spec, engine, axis, &value, job),
                    ("video", Some(job)) => bind_video(spec, engine, &value, job),
                    ("vector", _) => bind_vector(spec, engine, &value)?,
                    _ => bind_audio(spec, engine, axis, &value),
                };
                let backend = if spec.product.model_type == "audio" {
                    engine.audio_backend.clone()
                } else {
                    engine.backend.clone()
                };
                let key = format!("{}/{backend}/{id}/{axis}/{value}", engine.platform);
                if binding.kind == "unbound" {
                    record_unbound(
                        &mut missing,
                        &mut exceptions,
                        (id, axis, &value),
                        engine.backend == "candle",
                        key,
                    );
                }
                cells.push(RouteCell {
                    platform: engine.platform.clone(),
                    backend,
                    model: id.to_owned(),
                    axis,
                    conditioning: conditioning_meaning(spec, &value).into(),
                    output: output(spec, &value),
                    output_variants: output_variants(spec, &value),
                    mode: value.clone(),
                    canonical_request: binding.request,
                    expected_provider_ids: binding.expected,
                    providers: binding.providers,
                    binding: binding.kind,
                    owner: owner(spec)?,
                    entry_point: binding.entry_point,
                });
            }
            if !admitted_anywhere {
                let key = format!("product/{id}/{axis}/{value}");
                record_unbound(
                    &mut missing,
                    &mut exceptions,
                    (id, axis, &value),
                    false,
                    key,
                );
            }
        }
    }
    let mut registries = Vec::new();
    let mut model_aliases = Vec::new();
    for engine in &engines {
        for &(model, alias_target) in route_bindings::IMAGE_MODEL_ALIASES
            .iter()
            .chain(route_bindings::BESPOKE_IMAGE_ROUTE_ALIASES.iter())
        {
            let resolved = engine
                .mapped_generator(model)
                .map(|r| r.route_id.as_str())
                .unwrap_or(alias_target);
            let registered = engine.route(resolved).is_some();
            model_aliases.push(json!({
                "platform": engine.platform,
                "backend": engine.backend,
                "model": model,
                "provider": resolved,
                "registeredMediaGenerator": registered,
                "publicModel": public.iter().any(|spec| spec.product.id == model),
                "entryPoint": if registered {
                    "runtime bundle media registry (public eligibility checked separately)"
                } else {
                    "production bespoke selector; alias alone does not establish eligibility"
                },
                "owner": "ENG-7"
            }));
        }
        for surface in ALL_SURFACES {
            let backend = if surface.starts_with("audio_") {
                engine.audio_backend.as_str()
            } else {
                engine.backend.as_str()
            };
            for provider in engine.surface_ids(surface) {
                let public_cells: Vec<_> = cells
                    .iter()
                    .filter(|c| {
                        c.platform == engine.platform
                            && c.backend == backend
                            && c.providers.iter().any(|p| p == &provider)
                    })
                    .map(|c| format!("{}/{}/{}/{}", c.backend, c.model, c.axis, c.mode))
                    .collect();
                let callable_entry = registry_entry(&engine.platform, surface, &provider);
                if callable_entry.starts_with("unclassified registry surface:") {
                    return Err(callable_entry);
                }
                registries.push(RegistrySurface {
                    platform: engine.platform.clone(),
                    backend: backend.into(),
                    surface: surface.to_owned(),
                    provider: provider.clone(),
                    descriptor: engine.surface_descriptor(surface, &provider),
                    owner: registry_owner(surface, &provider, &engine.export)?,
                    disposition: if public_cells.is_empty() {
                        "direct-engine/bundle-consumer"
                    } else {
                        "public-and-direct-engine"
                    },
                    public_cells,
                    entry_point: format!(
                        "runtime-{}::catalog().snapshot().{surface}",
                        engine.platform
                    ),
                    callable_entry,
                });
            }
        }
    }
    // Non-desktop shipping profiles come directly from their hardware-free `sga.capabilities/2`
    // exports (`ios-media.json`, `ios-llm-only.json`, `candle-cpu.json`).
    let mut profile_exports = BTreeMap::new();
    profile_exports.insert(
        "ios-media".into(),
        json!({
            "platform": ios_media.platform,
            "backend": ios_media.backend,
            "routes": ios_media.routes.keys().collect::<Vec<_>>(),
            "llmModels": ios_media.llm_models.keys().collect::<Vec<_>>(),
            "imageTransforms": ios_media.transforms.image.keys().collect::<Vec<_>>(),
            "snapshotPreparerBackends": ios_media.snapshot_preparer_backends,
        }),
    );
    profile_exports.insert(
        "ios-llm-only".into(),
        json!({
            "platform": ios_llm_only.platform,
            "backend": ios_llm_only.backend,
            "routes": ios_llm_only.routes.keys().collect::<Vec<_>>(),
            "llmModels": ios_llm_only.llm_models.keys().collect::<Vec<_>>(),
            "imageTransforms": ios_llm_only.transforms.image.keys().collect::<Vec<_>>(),
        }),
    );
    profile_exports.insert(
        "candle-cpu".into(),
        json!({
            "platform": candle_cpu.platform,
            "backend": candle_cpu.backend,
            "routesCount": candle_cpu.routes.len(),
            "audioRoutesCount": candle_cpu.audio_routes.len(),
            "llmModels": candle_cpu.llm_models.keys().collect::<Vec<_>>(),
        }),
    );
    for (surface, ids) in [
        (
            "generator_ids",
            ios_media.routes.keys().cloned().collect::<Vec<_>>(),
        ),
        (
            "text_llm_ids",
            ios_media.llm_models.keys().cloned().collect::<Vec<_>>(),
        ),
        (
            "transform_ids",
            ios_media
                .transforms
                .image
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
        ),
        (
            "snapshot_preparer_backends",
            ios_media.snapshot_preparer_backends.clone(),
        ),
    ] {
        for provider in ids {
            let callable_entry = registry_entry("ios", surface, &provider);
            registries.push(RegistrySurface {
                platform: "ios".into(),
                backend: "mlx".into(),
                surface: surface.to_owned(),
                provider: provider.clone(),
                descriptor: if surface == "generator_ids" {
                    ios_media
                        .routes
                        .get(&provider)
                        .and_then(|r| serde_json::to_value(r).ok())
                } else {
                    None
                },
                owner: "ENG-13",
                public_cells: Vec::new(),
                disposition: "direct-engine/bundle-consumer; sga.capabilities/2 export",
                entry_point:
                    "runtime-ios::catalog() (media-enabled; LLM-only omits generators/transforms)"
                        .into(),
                callable_entry,
            });
        }
    }
    for surface in ALL_SURFACES {
        let cpu_ids: Vec<String> = match surface {
            "generator_ids" => candle_cpu.routes.keys().cloned().collect(),
            "audio_generator_ids" => candle_cpu.audio_routes.keys().cloned().collect(),
            "trainer_ids" => candle_cpu.trainers.keys().cloned().collect(),
            "captioner_ids" => candle_cpu.captioners.keys().cloned().collect(),
            "image_embedder_ids" => candle_cpu.embedders.image.keys().cloned().collect(),
            "text_embedder_ids" => candle_cpu.embedders.text.keys().cloned().collect(),
            "text_llm_ids" => candle_cpu.llm_models.keys().cloned().collect(),
            "transform_ids" => candle_cpu.transforms.image.keys().cloned().collect(),
            "audio_transform_ids" => candle_cpu.transforms.audio.keys().cloned().collect(),
            "audio_voice_embedder_ids" => candle_cpu.embedders.voice.keys().cloned().collect(),
            "audio_transcriber_ids" => candle_cpu.transcribers.keys().cloned().collect(),
            "audio_embedder_ids" => candle_cpu.embedders.audio.keys().cloned().collect(),
            "snapshot_preparer_backends" => candle_cpu.snapshot_preparer_backends.clone(),
            "audio_snapshot_preparer_backends" => {
                candle_cpu.audio_snapshot_preparer_backends.clone()
            }
            _ => Vec::new(),
        };
        for provider in cpu_ids {
            let callable_entry = registry_entry("cpu", surface, &provider);
            registries.push(RegistrySurface {
                platform: "cpu".into(),
                backend: "candle".into(),
                surface: surface.to_owned(),
                provider: provider.clone(),
                descriptor: if surface == "generator_ids" {
                    candle_cpu
                        .routes
                        .get(&provider)
                        .and_then(|r| serde_json::to_value(r).ok())
                } else {
                    None
                },
                owner: registry_owner(surface, &provider, &candle_cpu)?,
                public_cells: Vec::new(),
                disposition: "direct-engine/bundle-consumer; sga.capabilities/2 export",
                entry_point:
                    "runtime-cpu::catalog() (default media+audio; profiles may disable either)"
                        .into(),
                callable_entry,
            });
        }
    }
    let jobs = JobType::known_values()
        .iter()
        .map(|wire| {
            let job: JobType = serde_json::from_value(json!(wire)).map_err(|e| e.to_string())?;
            let (lifecycle, output, owner) = lifecycle(job.clone())?;
            let advertised_by = engines
                .iter()
                .filter(|engine| engine.worker_capabilities.iter().any(|cap| cap == *wire))
                .map(|engine| engine.platform.clone())
                .collect();
            Ok(JobLifecycle {
                job_type: job.as_str().to_owned(),
                lifecycle,
                output,
                owner,
                entry_point:
                    "crates/sceneworks-worker/src/utility_dispatch.rs::dispatch_utility_job",
                advertised_by,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RouteCoverage {
        schema_version: 3,
        evidence_kind: "metadata-only; registration is not execution or device-fit evidence",
        sources,
        jobs,
        cells,
        registries,
        profile_exports,
        model_aliases,
        missing,
        exceptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn without_starvector(source: &str) -> String {
        let mut value: Value = serde_json::from_str(source).unwrap();
        value["routes"]
            .as_object_mut()
            .unwrap()
            .remove("starvector_1b");
        value["routes"]
            .as_object_mut()
            .unwrap()
            .remove("starvector_8b");
        value["llm_models"]
            .as_object_mut()
            .unwrap()
            .retain(|id, _| !id.contains("starvector"));
        serde_json::to_string(&value).unwrap()
    }

    #[test]
    fn every_product_route_has_a_binding_or_an_approved_exception() {
        let report = generation_route_coverage().unwrap();
        assert!(report.missing.is_empty(), "{:#?}", report.missing);
        assert_eq!(report.jobs.len(), JobType::known_values().len());
        // Every exception-covered cell is a Candle-side gap with no binding, named by a record.
        let approved: BTreeSet<&str> = crate::registry::approved_backend_exceptions()
            .iter()
            .map(|record| record.id.as_str())
            .collect();
        for exception in &report.exceptions {
            assert!(exception.cell.starts_with("cuda/candle/"), "{exception:?}");
            assert!(
                approved.contains(exception.exception.as_str()),
                "{exception:?}"
            );
            assert!(
                !report.cells.iter().any(|c| format!(
                    "{}/{}/{}/{}/{}",
                    c.platform, c.backend, c.model, c.axis, c.mode
                ) == exception.cell
                    && c.binding != "unbound"),
                "an exception must not cover a bound cell: {exception:?}"
            );
        }
        assert!(report.cells.iter().any(|c| c.model == "flux_dev"
            && c.axis == "conditioning"
            && c.mode == "control"
            && c.binding != "unbound"));
        assert!(report
            .cells
            .iter()
            .any(|c| c.model == "pulid_flux_dev" && c.binding == "registry"));
        assert!(report
            .cells
            .iter()
            .any(|c| c.model == "instantid_realvisxl" && c.binding == "registry"));
    }

    #[test]
    fn starvector_routes_bind_all_four_cells_via_modality_vector_and_fail_closed_when_removed() {
        let baseline = generation_route_coverage().unwrap();
        for (platform, backend, model) in [
            ("macos", "mlx", "starvector_1b"),
            ("cuda", "candle", "starvector_1b"),
            ("macos", "mlx", "starvector_8b"),
            ("cuda", "candle", "starvector_8b"),
        ] {
            let cell = baseline
                .cells
                .iter()
                .find(|c| {
                    c.platform == platform
                        && c.backend == backend
                        && c.model == model
                        && c.axis == "operation"
                        && c.mode == "image_to_svg"
                })
                .unwrap_or_else(|| {
                    panic!("missing baseline StarVector cell {platform}/{backend}/{model}")
                });
            assert_eq!(cell.binding, "registry");
            assert!(cell.providers.iter().any(|p| p == model));
        }

        // Negative control: removing the Modality::Vector routes alone must fail all 4 cells
        // even if llm_models entries are still present.
        let strip_vector_routes = |src: &str| {
            let mut value: Value = serde_json::from_str(src).unwrap();
            value["routes"]
                .as_object_mut()
                .unwrap()
                .remove("starvector_1b");
            value["routes"]
                .as_object_mut()
                .unwrap()
                .remove("starvector_8b");
            serde_json::to_string(&value).unwrap()
        };
        let without_routes = reconcile(
            &strip_vector_routes(MACOS_MLX_JSON),
            &strip_vector_routes(CANDLE_CUDA_JSON),
        )
        .unwrap();
        assert_eq!(
            without_routes.missing,
            [
                "macos/mlx/starvector_1b/operation/image_to_svg",
                "cuda/candle/starvector_1b/operation/image_to_svg",
                "macos/mlx/starvector_8b/operation/image_to_svg",
                "cuda/candle/starvector_8b/operation/image_to_svg",
            ]
        );

        let report = reconcile(
            &without_starvector(MACOS_MLX_JSON),
            &without_starvector(CANDLE_CUDA_JSON),
        )
        .unwrap();
        assert_eq!(
            report.missing,
            [
                "macos/mlx/starvector_1b/operation/image_to_svg",
                "cuda/candle/starvector_1b/operation/image_to_svg",
                "macos/mlx/starvector_8b/operation/image_to_svg",
                "cuda/candle/starvector_8b/operation/image_to_svg",
            ]
        );
        let expected: BTreeSet<_> = report
            .cells
            .iter()
            .filter(|c| c.model.starts_with("starvector_") && c.binding == "unbound")
            .flat_map(|c| c.expected_provider_ids.iter().map(String::as_str))
            .collect();
        assert_eq!(
            expected,
            [
                "mlx-starvector-1b",
                "mlx-starvector-8b",
                "candle-starvector-1b",
                "candle-starvector-8b"
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn candle_qwen_edit_aliases_and_newly_registered_bespoke_routes_reconcile() {
        let candle = crate::sga_capabilities::candle_cuda();
        for alias in super::super::candle::CANDLE_QWEN_EDIT_ROUTE_ALIASES {
            let route = candle
                .routes
                .get(*alias)
                .unwrap_or_else(|| panic!("missing Candle registered Qwen edit alias {alias}"));
            assert!(
                route.modes.iter().any(|m| m == "image_to_image")
                    && route.modes.iter().any(|m| m == "multi_reference"),
                "Candle Qwen edit route {alias} must advertise image_to_image + multi_reference"
            );
            assert!(route.capabilities.supports_preview);
        }
        for bespoke_route in [
            "pulid_flux",
            "instantid",
            "flux1_dev_control",
            "flux1_dev_ip_adapter",
            "flux2_dev_edit",
            "flux2_klein_9b_edit",
            "sdxl_control",
            "sdxl_detail",
            "sdxl_edit",
            "sdxl_ip_adapter",
            "kolors_control",
            "kolors_ip_adapter",
            "qwen_image_control",
            "z_image_control",
            "z_image_turbo_control",
        ] {
            assert!(
                candle.routes.contains_key(bespoke_route),
                "Candle sga.capabilities/2 must register promoted route {bespoke_route}"
            );
        }
        let baseline = generation_route_coverage().unwrap();
        for model in ["qwen_image_edit_2511", "qwen_image_edit_2511_lightning"] {
            let cell = baseline
                .cells
                .iter()
                .find(|c| {
                    c.platform == "cuda"
                        && c.backend == "candle"
                        && c.model == model
                        && c.axis == "operation"
                        && c.mode == "edit_image"
                })
                .unwrap_or_else(|| panic!("missing cuda/candle/{model}/operation/edit_image"));
            assert_eq!(cell.binding, "registry");
            assert_eq!(cell.providers, vec![model.to_owned()]);
        }
    }

    #[test]
    fn every_native_dispatch_arm_is_joined_to_a_lifecycle() {
        use syn::visit::Visit;
        struct Dispatch {
            variants: BTreeSet<String>,
        }
        fn variants(pattern: &syn::Pat, set: &mut BTreeSet<String>) {
            match pattern {
                syn::Pat::Path(path) => {
                    if path
                        .path
                        .segments
                        .iter()
                        .rev()
                        .nth(1)
                        .is_some_and(|part| part.ident == "JobType")
                    {
                        set.insert(path.path.segments.last().unwrap().ident.to_string());
                    }
                }
                syn::Pat::Or(or) => {
                    for case in &or.cases {
                        variants(case, set);
                    }
                }
                _ => (),
            }
        }
        impl<'ast> Visit<'ast> for Dispatch {
            fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
                if matches!(node.expr.as_ref(), syn::Expr::Field(field) if matches!(&field.member, syn::Member::Named(name) if name == "job_type"))
                {
                    for arm in &node.arms {
                        if arm.guard.is_none() {
                            variants(&arm.pat, &mut self.variants);
                        }
                    }
                    return;
                }
                syn::visit::visit_expr_match(self, node);
            }
        }
        let syntax = syn::parse_file(DISPATCH).unwrap();
        let function = syntax
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(f) if f.sig.ident == "dispatch_utility_job" => Some(f),
                _ => None,
            })
            .expect("the public worker dispatch owner must exist");
        let mut dispatch = Dispatch {
            variants: BTreeSet::new(),
        };
        dispatch.visit_block(&function.block);
        let expected: BTreeSet<_> = JobType::known_values()
            .iter()
            .map(|wire| {
                format!(
                    "{:?}",
                    serde_json::from_value::<JobType>(json!(wire)).unwrap()
                )
            })
            .collect();
        assert_eq!(
            dispatch.variants, expected,
            "comments and nested handler text cannot replace a native job arm"
        );
    }

    #[test]
    fn bespoke_lanes_keep_their_conditioning_and_output_contracts() {
        let report = generation_route_coverage().unwrap();
        let flux_ip = report
            .cells
            .iter()
            .find(|c| c.backend == "candle" && c.model == "flux_dev" && c.mode == "character_image")
            .unwrap();
        assert!(flux_ip
            .providers
            .iter()
            .any(|p| p.ends_with("::FluxIpAdapter")));
        assert!(flux_ip.conditioning.contains("IP-adapter"));
        let vqa = report
            .cells
            .iter()
            .find(|c| c.model == "sensenova_u1_8b" && c.mode == "vqa")
            .unwrap();
        assert_eq!(vqa.output, "text");
        assert!(vqa.entry_point.ends_with("sensenova_jobs.rs::run_vqa_job"));
        let ios: Vec<_> = report
            .registries
            .iter()
            .filter(|r| r.platform == "ios" && r.surface == "generator_ids")
            .map(|r| r.provider.as_str())
            .collect();
        assert_eq!(ios.len(), 7);
        assert!(!ios.iter().any(|id| id.starts_with("flux1")));
    }

    #[test]
    fn an_exception_explains_a_candle_gap_but_never_hides_an_mlx_loss() {
        // Krea Realtime is MLX-only by an approved sequencing decision: its Candle cells are
        // reported as exception-covered. Dropping its MLX route from sga.capabilities/2 must
        // surface the MLX cells as missing; the register cannot explain them away.
        fn without_krea_realtime(source: &str) -> String {
            let mut value: Value = serde_json::from_str(source).unwrap();
            value["routes"]
                .as_object_mut()
                .unwrap()
                .remove("krea_realtime_14b");
            serde_json::to_string(&value).unwrap()
        }
        let krea = |r: &RouteCoverage| -> Vec<String> {
            r.exceptions
                .iter()
                .filter(|e| e.cell.contains("/krea_realtime_14b/"))
                .map(|e| e.cell.clone())
                .collect()
        };
        let baseline = generation_route_coverage().unwrap();
        assert!(krea(&baseline)
            .iter()
            .all(|cell| cell.starts_with("cuda/candle/")));
        assert!(!krea(&baseline).is_empty());
        let report = reconcile(
            &without_krea_realtime(MACOS_MLX_JSON),
            &without_krea_realtime(CANDLE_CUDA_JSON),
        )
        .unwrap();
        assert_eq!(
            report.missing,
            [
                "macos/mlx/krea_realtime_14b/operation/image_to_video",
                "macos/mlx/krea_realtime_14b/operation/text_to_video",
                "macos/mlx/krea_realtime_14b/operation/video_to_video",
            ]
        );
        assert_eq!(krea(&report), krea(&baseline));
    }

    #[test]
    fn unowned_job_fails_closed() {
        assert!(lifecycle(JobType::Unknown("new-job".into())).is_err());
    }
}
