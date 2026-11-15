//! PIPE-18 route reconciliation (SW-6931, SW-F331 E1): a metadata-only join, not a capability
//! authority and not an execution or device-fit claim.
//!
//! The two sides are written independently:
//!
//! * **Product side (authored):** the public operations and authored conditioning carriers of
//!   `config/manifests/builtin.models.jsonc`, the per-backend routing admission of
//!   `config/manifests/registry/route-policy.json`, the vector providers each product names, the
//!   native job arms of `crates/sceneworks-worker/src/utility_dispatch.rs`, and the production
//!   claim gate with its bespoke worker lanes. No field the core build copies out of a capability
//!   dump is read on this side.
//! * **Engine side:** the platform registries in
//!   `config/engine-capabilities/runtime/capabilities.{mlx,candle}.json`, written by the worker's
//!   `engine_capability_facts.rs` dumper.
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
use crate::registry::ModelSpec;

const DISPATCH: &str = include_str!("../../../../sceneworks-worker/src/utility_dispatch.rs");
const MLX_RUNTIME_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/runtime/capabilities.mlx.json");
const CANDLE_RUNTIME_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/runtime/capabilities.candle.json");
const PRODUCT_MANIFEST: &str = include_str!("../../../../../config/manifests/builtin.models.jsonc");
const ROUTE_POLICY: &str =
    include_str!("../../../../../config/manifests/registry/route-policy.json");
const PROFILE_RECEIPT: &str =
    include_str!("../../../tests/fixtures/generation-route-profiles.json");

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteCoverage {
    pub schema_version: u32,
    pub evidence_kind: &'static str,
    pub sources: BTreeMap<String, String>,
    pub jobs: Vec<JobLifecycle>,
    pub cells: Vec<RouteCell>,
    pub registries: Vec<RegistrySurface>,
    pub profile_source_receipt: Value,
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
    /// Platforms whose worker advertises this job arm in its registry dump.
    pub advertised_by: Vec<String>,
}

/// One platform's engine registries, read from its runtime capability dump.
struct Engine {
    raw: Value,
    platform: String,
    backend: String,
    audio_backend: String,
    worker: WorkerSnapshot,
}

impl Engine {
    fn parse(source: &str) -> Result<Self, String> {
        let raw: Value = serde_json::from_str(source).map_err(|e| e.to_string())?;
        let snapshot = raw["snapshot"].as_object().ok_or("missing snapshot")?;
        let text = |key: &str| {
            snapshot
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("snapshot has no {key}"))
        };
        let platform = text("platform")?;
        let backend = text("backend")?;
        let audio_backend = text("audio_backend")?;
        let capabilities = raw["workerCapabilities"]
            .as_array()
            .ok_or("missing worker capabilities")?
            .iter()
            .map(|value| {
                serde_json::from_value::<WorkerCapability>(value.clone()).map_err(|e| e.to_string())
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
            raw,
            platform,
            backend,
            audio_backend,
            worker,
        })
    }

    fn snapshot(&self) -> &Map<String, Value> {
        self.raw["snapshot"].as_object().expect("parsed snapshot")
    }

    fn ids(&self, surface: &str) -> BTreeSet<&str> {
        self.snapshot()
            .get(surface)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect()
    }

    fn descriptor(&self, list: &str, id: &str) -> Option<&Value> {
        self.snapshot()
            .get(list)
            .and_then(Value::as_array)
            .and_then(|items| items.iter().find(|d| d["id"] == id))
    }

    fn conditioning(descriptor: &Value) -> BTreeSet<&str> {
        descriptor["conditioning"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect()
    }

    /// The registered media generator the platform maps a public model to.
    fn mapped_generator(&self, model: &str) -> Option<&Value> {
        let engine = self.raw["modelMappings"].get(model)?.as_str()?;
        self.ids("generator_ids")
            .contains(engine)
            .then(|| self.descriptor("generator_capabilities", engine))
            .flatten()
    }

    fn video_engines(&self, model: &str, mode: &str) -> Vec<String> {
        self.raw["videoModelMappings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|row| row["modelId"] == model && row["mode"] == mode)
            .flat_map(|row| row["engineIds"].as_array().into_iter().flatten())
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    }

    fn claims(&self, job: &JobSnapshot) -> bool {
        crate::jobs_store::worker_supports_job(&self.worker, job)
    }
}

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
            "ltx-video" | "minimax-h3" | "mage-flow" | "sensenova-u1" | "bernini"
            | "krea-realtime" => "ENG-6/APP-5/APP-6",
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
    snapshot: &Map<String, Value>,
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
            let family = snapshot["generator_capabilities"]
                .as_array()
                .and_then(|ds| ds.iter().find(|d| d["id"] == provider))
                .and_then(|d| d["family"].as_str())
                .ok_or_else(|| format!("generator {provider} has no descriptor family"))?;
            match family {
                "flux" | "pulid" => "ENG-1a",
                "chroma" | "flux2" | "lens" | "anima" | "sd3" | "stable-diffusion-3" | "sana"
                | "boogu" | "qwen-image" | "qwen-image-2-1" | "z-image" | "kolors" | "sdxl" => {
                    "ENG-2/ENG-5"
                }
                "wan" | "scail2" | "svd" | "mochi" => "ENG-3/ENG-5",
                "krea_2" => "ENG-4b",
                "flux3" | "moss_image" | "iris" | "ideogram" | "supra2" => "ENG-5",
                "ltx" | "minimax_h3" | "mage_flow" | "sensenova-u1" | "bernini"
                | "krea_realtime" | "seedvr2" => "ENG-6",
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
            // The ordered reference list rides the product's own reference surface: its
            // character request, else its source edit, else (MOSS) a reference-only edit.
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
        .and_then(|d| d["id"].as_str())
        .map(|engine_id| vec![engine_id.to_owned()])
        .unwrap_or_default();
    if !engine.claims(job) {
        return Binding::none(expected, Some(request_json(job)));
    }
    let registered = mapped.is_some_and(|descriptor| {
        let kinds = Engine::conditioning(descriptor);
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
        // The MLX router has no lane enum, but it refuses every carrier it has no lane for, so
        // its admission of the carrier-bearing request is the lane's claim. A pose folded into a
        // reference is not a strict control lane: that also needs an authored pose control mode.
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
    let registered: Vec<&Value> = engines
        .iter()
        .filter_map(|e| {
            engine
                .ids("generator_ids")
                .contains(e.as_str())
                .then(|| engine.descriptor("generator_capabilities", e))
                .flatten()
        })
        .collect();
    let kinds: BTreeSet<&str> = registered
        .iter()
        .flat_map(|d| Engine::conditioning(d))
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
    let provider = field(spec, "vector")
        .and_then(|vector| vector["providers"][engine.backend.as_str()]["id"].as_str())
        .ok_or_else(|| format!("{id} names no {} vector provider", engine.backend))?
        .to_owned();
    let job = probe(
        JobType::VectorGenerate,
        id,
        json!({ "mode": mode, "sourceAssetId": "probe" }),
    );
    let bound = engine.ids("text_llm_ids").contains(provider.as_str()) && engine.claims(&job);
    Ok(if bound {
        Binding {
            providers: vec![provider.clone()],
            kind: "registry",
            entry_point: "crates/sceneworks-worker/src/vector_jobs.rs::run_vector_job".into(),
            expected: vec![provider],
            request: Some(request_json(&job)),
        }
    } else {
        Binding::none(vec![provider], Some(request_json(&job)))
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
        .find(|surface| engine.ids(surface).contains(id));
    let carrier_ok = axis != "conditioning"
        || engine
            .descriptor("audio_generator_capabilities", id)
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
            // A strict control carrier is authored through `ui.controlModes`; a bare pose library
            // folds the pose into the reference carrier instead.
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

/// Join every authored product route to the platform registries and bespoke lanes.
/// Metadata only: no model or backend is constructed, allocated or run.
pub fn generation_route_coverage() -> Result<RouteCoverage, String> {
    reconcile(MLX_RUNTIME_FACTS, CANDLE_RUNTIME_FACTS)
}

fn reconcile(mlx_source: &str, candle_source: &str) -> Result<RouteCoverage, String> {
    let engines = [Engine::parse(mlx_source)?, Engine::parse(candle_source)?];
    let mut sources = BTreeMap::new();
    for (name, source) in [
        ("productManifest", PRODUCT_MANIFEST),
        ("productRoutePolicy", ROUTE_POLICY),
        ("jobDispatcher", DISPATCH),
        ("mlxRuntimeRegistries", mlx_source),
        ("candleRuntimeRegistries", candle_source),
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
                    // The approved register records deliberate MLX-only product cells. Report the
                    // Candle side it explains, so an approved difference stays visible.
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
            // An authored public cell that no backend's routing admits is unreachable product
            // surface, not a backend difference.
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
        let snapshot = engine.snapshot();
        for (model, provider) in engine.raw["modelMappings"]
            .as_object()
            .ok_or("missing model mappings")?
        {
            let registered = provider
                .as_str()
                .is_some_and(|p| engine.ids("generator_ids").contains(p));
            model_aliases.push(json!({ "platform": engine.platform, "backend": engine.backend,
                "model": model, "provider": provider, "registeredMediaGenerator": registered,
                "publicModel": public.iter().any(|spec| spec.product.id == *model),
                "entryPoint": if registered { "runtime bundle media registry (public eligibility checked separately)" }
                    else { "production bespoke selector; alias alone does not establish eligibility" },
                "owner": "ENG-7" }));
        }
        for (surface, ids) in snapshot
            .iter()
            .filter(|(k, _)| k.ends_with("_ids") || k.ends_with("_backends"))
        {
            let backend = if surface.starts_with("audio_") {
                engine.audio_backend.as_str()
            } else {
                engine.backend.as_str()
            };
            for id in ids.as_array().ok_or("registry IDs must be arrays")? {
                let provider = id.as_str().ok_or("registry ID must be a string")?;
                let public_cells: Vec<_> = cells
                    .iter()
                    .filter(|c| {
                        c.platform == engine.platform
                            && c.backend == backend
                            && c.providers.iter().any(|p| p == provider)
                    })
                    .map(|c| format!("{}/{}/{}/{}", c.backend, c.model, c.axis, c.mode))
                    .collect();
                let callable_entry = registry_entry(&engine.platform, surface, provider);
                if callable_entry.starts_with("unclassified registry surface:") {
                    return Err(callable_entry);
                }
                registries.push(RegistrySurface {
                    platform: engine.platform.clone(),
                    backend: backend.into(),
                    surface: surface.clone(),
                    provider: provider.into(),
                    descriptor: snapshot
                        .get(match surface.as_str() {
                            "audio_generator_ids" => "audio_generator_capabilities",
                            "trainer_ids" => "trainer_capabilities",
                            _ => "generator_capabilities",
                        })
                        .and_then(Value::as_array)
                        .and_then(|ds| ds.iter().find(|d| d["id"] == provider))
                        .cloned(),
                    owner: registry_owner(surface, provider, snapshot)?,
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
    // The mobile and CPU consumers submit no SceneWorks jobs. Keep their independently
    // source-identified profile contracts rather than projecting desktop product eligibility.
    let profile_source_receipt: Value =
        serde_json::from_str(PROFILE_RECEIPT).map_err(|e| e.to_string())?;
    for (surface, ids) in profile_source_receipt["ios"]
        .as_object()
        .ok_or("missing iOS profile")?
    {
        for id in ids.as_array().ok_or("profile IDs must be arrays")? {
            let provider = id.as_str().ok_or("invalid profile ID")?;
            registries.push(RegistrySurface {
                platform: "ios".into(),
                backend: "mlx".into(),
                surface: surface.clone(),
                provider: provider.into(),
                descriptor: None,
                owner: "ENG-13",
                public_cells: Vec::new(),
                disposition: "direct-engine/bundle-consumer; source-identified profile",
                entry_point:
                    "runtime-ios::catalog() (media-enabled; LLM-only omits generators/transforms)"
                        .into(),
                callable_entry: registry_entry("ios", surface, provider),
            });
        }
    }
    let cpu_registries: Vec<_> = registries
        .iter()
        .filter(|r| r.platform == "cuda" && r.surface != "text_llm_ids")
        .map(|r| RegistrySurface {
            platform: "cpu".into(),
            backend: r.backend.clone(),
            surface: r.surface.clone(),
            provider: r.provider.clone(),
            descriptor: r.descriptor.clone(),
            owner: r.owner,
            public_cells: Vec::new(),
            disposition:
                "direct-engine/bundle-consumer; shared Candle catalog, separate device admission",
            entry_point:
                "runtime-cpu::catalog() (default media+audio; profiles may disable either)".into(),
            callable_entry: registry_entry("cpu", &r.surface, &r.provider),
        })
        .collect();
    registries.extend(cpu_registries);
    for id in profile_source_receipt["cpu"]["text_llm_ids"]
        .as_array()
        .ok_or("missing CPU LLM profile")?
    {
        let provider = id.as_str().ok_or("invalid CPU LLM ID")?;
        registries.push(RegistrySurface {
            platform: "cpu".into(),
            backend: "candle".into(),
            surface: "text_llm_ids".into(),
            provider: provider.into(),
            descriptor: None,
            owner: if provider.contains("starvector") {
                "ENG-7/APP-5"
            } else {
                "ENG-14"
            },
            public_cells: Vec::new(),
            disposition: "direct-engine/bundle-consumer; source-identified profile",
            entry_point: "runtime-cpu::catalog().text()".into(),
            callable_entry: registry_entry("cpu", "text_llm_ids", provider),
        });
    }
    let jobs = JobType::known_values()
        .iter()
        .map(|wire| {
            let job: JobType = serde_json::from_value(json!(wire)).map_err(|e| e.to_string())?;
            let (lifecycle, output, owner) = lifecycle(job.clone())?;
            let advertised_by = engines
                .iter()
                .filter(|engine| {
                    engine.raw["workerCapabilities"]
                        .as_array()
                        .is_some_and(|caps| caps.iter().any(|cap| cap == *wire))
                })
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
        schema_version: 2,
        evidence_kind: "metadata-only; registration is not execution or device-fit evidence",
        sources,
        jobs,
        cells,
        registries,
        profile_source_receipt,
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
        value["snapshot"]["text_llm_ids"]
            .as_array_mut()
            .unwrap()
            .retain(|id| !id.as_str().unwrap().contains("starvector"));
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
            .any(|c| c.model == "pulid_flux_dev" && c.binding != "unbound"));
    }

    #[test]
    fn removing_starvector_names_exactly_four_backend_provider_cells() {
        let report = reconcile(
            &without_starvector(MLX_RUNTIME_FACTS),
            &without_starvector(CANDLE_RUNTIME_FACTS),
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
        // reported as exception-covered. Dropping its MLX registrations must surface the MLX
        // cells as missing; the register cannot explain them away.
        fn without_krea_realtime(source: &str) -> String {
            let mut value: Value = serde_json::from_str(source).unwrap();
            value["videoModelMappings"]
                .as_array_mut()
                .unwrap()
                .retain(|row| row["modelId"] != "krea_realtime_14b");
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
            &without_krea_realtime(MLX_RUNTIME_FACTS),
            &without_krea_realtime(CANDLE_RUNTIME_FACTS),
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
