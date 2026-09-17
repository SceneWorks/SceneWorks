//! Vector Studio request, provider, and safe-publication boundary.
//!
//! The route supplies only typed raster/text conditioning. A mode-specific native provider streams
//! the SVG through [`MultimodalVectorProviderAdapter`]; the worker does not create a staging
//! directory until that stream has completed without cancellation. The source is then parsed into
//! a deliberately small inert SVG subset, canonicalized, rendered through resvg (which has no
//! network/resource loader), and published as an SVG+PNG directory rename.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gen_core::core_llm::{
    Content, ImageRef, LoadSpec as TextLoadSpec, Message, ModelRequirements, Role, Sampling,
    StarVectorFinishReason, StarVectorOutput, StarVectorRequest, StarVectorStreamEvent,
    StarVectorTier, TextLlmRequest,
};
use quick_xml::events::Event;
use quick_xml::{Reader, XmlVersion};
use resvg::usvg;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::*;

const MAX_SVG_BYTES: usize = 256 * 1024;
const MAX_SVG_DEPTH: usize = 32;
const MAX_SVG_ELEMENTS: usize = 2_000;
const MAX_SVG_ATTRIBUTES: usize = 4_096;
const MAX_SVG_ATTRIBUTES_PER_ELEMENT: usize = 12;
const MAX_SVG_ATTRIBUTE_VALUE_BYTES: usize = 64 * 1024;
const MAX_SVG_ATTRIBUTE_VALUE_BYTES_TOTAL: usize = 192 * 1024;
const MAX_SVG_PATH_DATA_BYTES: usize = 160 * 1024;
const MAX_SVG_PATH_COMMANDS: usize = 16_384;
const MAX_SVG_PATH_NUMBERS: usize = 65_536;
const MAX_SVG_POINT_NUMBERS: usize = 32_768;
const MAX_SVG_TRANSFORM_NUMBERS: usize = 4_096;
const MAX_SVG_DASH_NUMBERS: usize = 4_096;
const MAX_SVG_RESOURCES: usize = 128;
const MAX_SVG_RESOURCE_REFERENCES: usize = 256;
const MAX_SVG_GRADIENT_STOPS: usize = 512;
const MAX_SVG_FILTER_PRIMITIVES: usize = 64;
const MAX_SVG_FILTER_BLUR_SIGMA: f64 = 64.0;
// Eight maximum-size RGBA renderer surfaces (128 MiB). The parsed-tree check below charges the
// actual clipped layer for every application, result, and input surface before resvg can allocate.
const MAX_SVG_FILTER_OFFSCREEN_PIXELS: u64 =
    MAX_PREVIEW_DIMENSION as u64 * MAX_PREVIEW_DIMENSION as u64 * 8;
const MAX_SVG_COORDINATE_MAGNITUDE: f64 = 1_000_000.0;
const MAX_SVG_VIEWBOX_ORIGIN_MAGNITUDE: f64 = 1_000_000.0;
const MAX_PREVIEW_DIMENSION: u32 = 2_048;
const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";
const VECTOR_SANITIZER_VERSION: &str = "sceneworks-inert-svg-v1";
const VECTOR_RENDERER_VERSION: &str = "resvg-0.45";
const CANCEL_MESSAGE: &str = "Vector generation canceled before publication.";
const STARVECTOR_ADAPTER_ID: &str = "starvector";

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn terminal_result(
    terminal: &TerminalProviderOutcome,
    canonical: Option<(&Path, &[u8])>,
    preview: Option<(&Path, &[u8])>,
    transcript: Option<(&Path, &[u8])>,
) -> Value {
    json!({
        "accepted": terminal.publishable(),
        "finishReason": terminal.finish_reason,
        "generatedTokens": terminal.generated_tokens,
        "generatedBytes": terminal.generated_bytes,
        "latencySeconds": terminal.latency_seconds,
        "providerId": terminal.provider_id,
        "modelId": terminal.model_id,
        "modelRepository": terminal.model_repository,
        "modelRevision": terminal.model_revision,
        "backend": terminal.backend,
        "providerTranscriptPath": transcript.map(|(path, _)| path.to_string_lossy().into_owned()),
        "providerTranscriptSha256": transcript.map(|(_, bytes)| sha256_hex(bytes)),
        "canonicalSvgPath": canonical.map(|(path, _)| path.to_string_lossy().into_owned()),
        "canonicalSvgSha256": canonical.map(|(_, bytes)| sha256_hex(bytes)),
        "previewPngPath": preview.map(|(path, _)| path.to_string_lossy().into_owned()),
        "previewPngSha256": preview.map(|(_, bytes)| sha256_hex(bytes)),
        "resultContainsInlineSvg": false,
    })
}

fn terminal_generation_limit_result(
    terminal: &TerminalProviderOutcome,
    transcript: (&Path, &[u8]),
) -> WorkerResult<Value> {
    if !matches!(
        terminal.finish_reason,
        "token_limit" | "byte_limit" | "wall_time_limit"
    ) {
        return Err(WorkerError::Engine(
            "terminal generation rejection lacks a typed bounded finish reason".to_owned(),
        ));
    }
    let mut evidence = terminal_result(terminal, None, None, Some(transcript));
    let object = evidence.as_object_mut().ok_or_else(|| {
        WorkerError::Engine("terminal generation rejection evidence must be an object".to_owned())
    })?;
    object.insert("outcome".to_owned(), Value::String("rejected".to_owned()));
    object.insert(
        "rejectionStage".to_owned(),
        Value::String("generation_limit".to_owned()),
    );
    object.insert(
        "rejectionCode".to_owned(),
        Value::String(terminal.finish_reason.to_owned()),
    );
    object.insert(
        "rejectionReason".to_owned(),
        Value::String(format!(
            "native StarVector stopped at the {}",
            terminal.finish_reason
        )),
    );
    Ok(evidence)
}

fn terminal_transcript_bytes(terminal: &TerminalProviderOutcome) -> WorkerResult<Vec<u8>> {
    serde_json::to_vec(&json!({
        "providerId": terminal.provider_id,
        "modelId": terminal.model_id,
        "modelRepository": terminal.model_repository,
        "modelRevision": terminal.model_revision,
        "backend": terminal.backend,
        "finishReason": terminal.finish_reason,
        "generatedTokens": terminal.generated_tokens,
        "generatedBytes": terminal.generated_bytes,
        "latencySeconds": terminal.latency_seconds,
    }))
    .map_err(|error| {
        WorkerError::Engine(format!("serialize StarVector terminal transcript: {error}"))
    })
}

fn add_source_raster_evidence(
    mut value: Value,
    source_raster: Option<(&Path, &[u8])>,
) -> WorkerResult<Value> {
    let object = value.as_object_mut().ok_or_else(|| {
        WorkerError::Engine("StarVector terminal evidence must be a JSON object".to_owned())
    })?;
    match source_raster {
        Some((path, bytes)) => {
            object.insert(
                "sourceRasterPath".to_owned(),
                Value::String(path.to_string_lossy().into_owned()),
            );
            object.insert(
                "sourceRasterSha256".to_owned(),
                Value::String(sha256_hex(bytes)),
            );
        }
        None => {
            object.insert("sourceRasterPath".to_owned(), Value::Null);
            object.insert("sourceRasterSha256".to_owned(), Value::Null);
        }
    }
    Ok(value)
}

#[derive(Clone, Copy)]
struct StarVectorModelIdentity {
    model_id: &'static str,
    repository: &'static str,
    revision: &'static str,
    tier: StarVectorTier,
    mlx_provider_id: &'static str,
    candle_provider_id: &'static str,
}

const STARVECTOR_MODELS: &[StarVectorModelIdentity] = &[
    StarVectorModelIdentity {
        model_id: "starvector_1b",
        repository: "starvector/starvector-1b-im2svg",
        revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
        tier: StarVectorTier::OneB,
        mlx_provider_id: "mlx-starvector-1b",
        candle_provider_id: "candle-starvector-1b",
    },
    // Kept here so the bridge already has one exact identity authority when the terminal catalog
    // batch admits 8B. Until that catalog row and the permanent inference pin land, no job can
    // select it and the runtime catalog remains fail-closed.
    StarVectorModelIdentity {
        model_id: "starvector_8b",
        repository: "starvector/starvector-8b-im2svg",
        revision: "518beea8dcb5f7a37c5911e92d1d62a76beee7f9",
        tier: StarVectorTier::EightB,
        mlx_provider_id: "mlx-starvector-8b",
        candle_provider_id: "candle-starvector-8b",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VectorMode {
    ImageToSvg,
    TextToSvg,
}

impl VectorMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ImageToSvg => "image_to_svg",
            Self::TextToSvg => "text_to_svg",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct VectorSampling {
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: u32,
    pub(crate) repetition_penalty: f32,
    pub(crate) repetition_context: u32,
    pub(crate) seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct VectorDetailBudget {
    pub(crate) max_new_tokens: u32,
    pub(crate) max_svg_bytes: u32,
    pub(crate) max_wall_time_ms: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VectorJobPayload {
    project_id: String,
    mode: VectorMode,
    model: String,
    #[serde(default)]
    source_asset_id: Option<String>,
    #[serde(default)]
    prompt: String,
    sampling: VectorSampling,
    detail_budget: VectorDetailBudget,
    model_manifest_entry: Value,
    #[serde(default)]
    workflow: Option<Value>,
}

/// Backend-neutral input for the later runtime bridge. The adapter deliberately owns multimodal
/// composition: callers provide the confined raster path and disclosed text guidance separately,
/// and a backend maps them into its native text-provider request without leaking engine types here.
#[derive(Clone, Debug)]
pub(crate) struct VectorProviderRequest {
    pub(crate) mode: VectorMode,
    pub(crate) model: String,
    pub(crate) source_path: Option<PathBuf>,
    pub(crate) prompt: String,
    pub(crate) sampling: VectorSampling,
    pub(crate) detail_budget: VectorDetailBudget,
}

/// Injected multimodal text-provider seam. Implementations must poll `cancel` while decoding and
/// emit only UTF-8 source fragments. Production uses [`NativeStarVectorProvider`]; tests inject
/// small structural providers without loading weights.
pub(crate) trait MultimodalVectorProviderAdapter: Send + Sync {
    fn provider_id(&self) -> &str;
    fn supports_mode(&self, mode: VectorMode) -> bool;
    fn unavailable_reason(&self) -> Option<&str> {
        None
    }
    fn generate_svg(
        &self,
        request: &VectorProviderRequest,
        cancel: &gen_core::CancelFlag,
        progress: tokio::sync::watch::Sender<u32>,
        on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
    ) -> WorkerResult<()>;

    /// Native StarVector providers record their typed terminal outcome here.  The normal worker
    /// path deliberately does not depend on it; the post-pin campaign alone requests it.
    fn terminal_outcome(&self) -> Option<TerminalProviderOutcome> {
        None
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TerminalProviderOutcome {
    finish_reason: &'static str,
    generated_tokens: u32,
    generated_bytes: usize,
    latency_seconds: f64,
    provider_id: String,
    model_id: &'static str,
    model_repository: &'static str,
    model_revision: &'static str,
    backend: &'static str,
}

impl TerminalProviderOutcome {
    fn publishable(&self) -> bool {
        matches!(self.finish_reason, "complete_root" | "eos")
    }
}

#[derive(Clone)]
struct NativeStarVectorProvider {
    identity: StarVectorModelIdentity,
    backend: &'static str,
    inference_provider_id: &'static str,
    weights_dir: PathBuf,
    terminal_outcome: Arc<Mutex<Option<TerminalProviderOutcome>>>,
}

impl NativeStarVectorProvider {
    fn resolve(settings: &Settings, payload: &VectorJobPayload) -> WorkerResult<Self> {
        if payload.mode != VectorMode::ImageToSvg {
            return Err(WorkerError::InvalidPayload(
                "native StarVector currently serves image_to_svg only".to_owned(),
            ));
        }
        if !payload.prompt.trim().is_empty() {
            return Err(WorkerError::InvalidPayload(
                "native StarVector image_to_svg does not accept text guidance".to_owned(),
            ));
        }
        let identity = starvector_model_identity(&payload.model)?;
        manifest_binds_starvector_identity(&payload.model_manifest_entry, identity)?;
        let backend = active_starvector_backend(settings)?;
        let inference_provider_id = match backend {
            "mlx" => identity.mlx_provider_id,
            "candle" => identity.candle_provider_id,
            _ => unreachable!("active_starvector_backend returns a closed set"),
        };
        let provider = payload
            .model_manifest_entry
            .pointer(&format!("/vector/providers/{backend}"))
            .and_then(Value::as_object)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "selected StarVector model has no {backend} provider declaration"
                ))
            })?;
        if provider.get("available").and_then(Value::as_bool) != Some(true) {
            let reason = provider
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("provider_not_linked");
            return Err(WorkerError::InvalidPayload(format!(
                "vector_backend_unavailable: {reason}"
            )));
        }
        if provider.get("id").and_then(Value::as_str) != Some(inference_provider_id) {
            return Err(WorkerError::InvalidPayload(format!(
                "selected StarVector model does not bind exact {backend} provider {inference_provider_id}"
            )));
        }
        let weights_dir = crate::model_jobs::huggingface_receipt_weights_dir_at_revision(
            &settings.data_dir,
            identity.repository,
            identity.revision,
            Some(identity.model_id),
            Some("default"),
        )
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "vector_model_unavailable: exact receipt-backed snapshot {}@{} is missing or unproven",
                identity.repository, identity.revision
            ))
        })?;
        Ok(Self {
            identity: *identity,
            backend,
            inference_provider_id,
            weights_dir,
            terminal_outcome: Arc::new(Mutex::new(None)),
        })
    }
}

impl MultimodalVectorProviderAdapter for NativeStarVectorProvider {
    fn provider_id(&self) -> &str {
        STARVECTOR_ADAPTER_ID
    }

    fn supports_mode(&self, mode: VectorMode) -> bool {
        mode == VectorMode::ImageToSvg
    }

    fn generate_svg(
        &self,
        request: &VectorProviderRequest,
        cancel: &gen_core::CancelFlag,
        progress: tokio::sync::watch::Sender<u32>,
        on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
    ) -> WorkerResult<()> {
        let source_path = request.source_path.as_deref().ok_or_else(|| {
            WorkerError::InvalidPayload("image_to_svg source path is missing".to_owned())
        })?;
        let image = crate::prompt_refine_jobs::load_caption_image_ref(source_path)?;
        let text_cancel = gen_core::core_llm::CancelFlag::new();
        if cancel.is_cancelled() {
            text_cancel.cancel();
        }
        let typed_request = native_starvector_request(request, image, text_cancel.clone())?;
        let spec = TextLoadSpec::dense(self.weights_dir.to_string_lossy().into_owned());
        let requirements = ModelRequirements::from_request(&typed_request.text_request);
        let expected_provider_id = self.inference_provider_id.to_owned();
        let identity = self.identity;
        let expected_tier = identity.tier;
        let backend = self.backend;
        let cancel = cancel.clone();
        let load_context = format!(
            "{backend} StarVector provider {} load failed",
            self.inference_provider_id
        );
        let started = Instant::now();
        let generation = mirror_vector_cancel(cancel, text_cancel, || {
            tokio::runtime::Handle::current().block_on(crate::refine_model_cache::with_cached_refiner(
                spec,
                requirements,
                load_context,
                move |model| {
                    if model.descriptor().id != expected_provider_id {
                        return Err(WorkerError::Engine(format!(
                            "StarVector model-first resolution selected {}, expected {expected_provider_id}",
                            model.descriptor().id
                        )));
                    }
                    let provider = model.as_starvector_provider().ok_or_else(|| {
                        WorkerError::Engine(format!(
                            "text provider {expected_provider_id} exposes no typed StarVector view"
                        ))
                    })?;
                    if provider.starvector_descriptor().tier != expected_tier {
                        return Err(WorkerError::Engine(format!(
                            "text provider {expected_provider_id} exposes the wrong StarVector tier"
                        )));
                    }
                    let mut events = Vec::new();
                    let output = provider
                        .generate_svg(&typed_request, &mut |event| {
                            if let StarVectorStreamEvent::Progress { generated_tokens } = &event {
                                // Only counters leave this private source boundary. A watch channel
                                // stores one value even if decode outruns the async publisher.
                                record_vector_token_progress(
                                    &progress, *generated_tokens, typed_request.text_request.max_new_tokens,
                                );
                            } else {
                                events.push(event);
                            }
                        })
                        .map_err(classify_starvector_error)?;
                    let terminal = TerminalProviderOutcome {
                        finish_reason: terminal_finish_reason(output.finish_reason),
                        generated_tokens: output.generated_tokens,
                        generated_bytes: output.generated_bytes,
                        latency_seconds: started.elapsed().as_secs_f64(),
                        provider_id: expected_provider_id.clone(),
                        model_id: identity.model_id,
                        model_repository: identity.repository,
                        model_revision: identity.revision,
                        backend,
                    };
                    let source = validate_native_starvector_generation(output, events)?;
                    Ok((source, terminal))
                },
            ))
        })?;
        let (source, terminal) = generation;
        *self.terminal_outcome.lock().map_err(|_| {
            WorkerError::Engine("StarVector terminal outcome lock poisoned".to_owned())
        })? = Some(terminal);
        for (text, index) in source {
            on_source(&text, index)?;
        }
        Ok(())
    }

    fn terminal_outcome(&self) -> Option<TerminalProviderOutcome> {
        self.terminal_outcome.lock().ok()?.clone()
    }
}

const fn terminal_finish_reason(reason: StarVectorFinishReason) -> &'static str {
    match reason {
        StarVectorFinishReason::CompleteRoot => "complete_root",
        StarVectorFinishReason::Eos => "eos",
        StarVectorFinishReason::TokenLimit => "token_limit",
        StarVectorFinishReason::ByteLimit => "byte_limit",
        StarVectorFinishReason::WallTimeLimit => "wall_time_limit",
        StarVectorFinishReason::Cancelled => "cancelled",
    }
}

fn starvector_model_identity(model_id: &str) -> WorkerResult<&'static StarVectorModelIdentity> {
    STARVECTOR_MODELS
        .iter()
        .find(|identity| identity.model_id == model_id)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "selected vector model {model_id} has no exact native StarVector identity"
            ))
        })
}

fn manifest_binds_starvector_identity(
    manifest: &Value,
    identity: &StarVectorModelIdentity,
) -> WorkerResult<()> {
    if manifest.get("id").and_then(Value::as_str) != Some(identity.model_id)
        || manifest.get("type").and_then(Value::as_str) != Some("vector")
        || manifest.get("adapter").and_then(Value::as_str) != Some(STARVECTOR_ADAPTER_ID)
    {
        return Err(WorkerError::InvalidPayload(
            "selected model manifest does not bind the exact native StarVector identity".to_owned(),
        ));
    }
    let exact_download = manifest
        .get("downloads")
        .and_then(Value::as_array)
        .is_some_and(|downloads| {
            downloads.iter().any(|download| {
                download.get("coRequisite").and_then(Value::as_bool) != Some(true)
                    && download.get("repo").and_then(Value::as_str) == Some(identity.repository)
                    && download.get("revision").and_then(Value::as_str) == Some(identity.revision)
            })
        });
    if !exact_download {
        return Err(WorkerError::InvalidPayload(format!(
            "selected StarVector manifest does not bind {}@{}",
            identity.repository, identity.revision
        )));
    }
    Ok(())
}

fn active_starvector_backend(settings: &Settings) -> WorkerResult<&'static str> {
    #[cfg(target_os = "macos")]
    {
        if settings.backend_mlx_enabled && settings.gpu_id == "mlx" {
            return Ok("mlx");
        }
        Err(WorkerError::InvalidPayload(
            "vector_backend_unavailable: native MLX worker is disabled".to_owned(),
        ))
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        if settings.backend_candle_enabled && settings.gpu_id != "cpu" && settings.gpu_id != "mlx" {
            return Ok("candle");
        }
        Err(WorkerError::InvalidPayload(
            "vector_backend_unavailable: native Candle worker is disabled".to_owned(),
        ))
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
    {
        let _ = settings;
        Err(WorkerError::InvalidPayload(
            "vector_backend_unavailable: no native StarVector backend is linked".to_owned(),
        ))
    }
}

fn native_starvector_request(
    request: &VectorProviderRequest,
    image: ImageRef,
    cancel: gen_core::core_llm::CancelFlag,
) -> WorkerResult<StarVectorRequest> {
    if request.mode != VectorMode::ImageToSvg || !request.prompt.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "native StarVector requires image-only conditioning".to_owned(),
        ));
    }
    let max_svg_bytes = usize::try_from(request.detail_budget.max_svg_bytes)
        .map_err(|_| WorkerError::InvalidPayload("maxSvgBytes does not fit usize".to_owned()))?;
    if max_svg_bytes == 0 || max_svg_bytes > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(format!(
            "native StarVector maxSvgBytes must be 1..={MAX_SVG_BYTES}"
        )));
    }
    let top_k = usize::try_from(request.sampling.top_k)
        .map_err(|_| WorkerError::InvalidPayload("sampling.topK does not fit usize".to_owned()))?;
    let repetition_context =
        usize::try_from(request.sampling.repetition_context).map_err(|_| {
            WorkerError::InvalidPayload("sampling.repetitionContext does not fit usize".to_owned())
        })?;
    let text_request = TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Image(image)],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling {
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            top_k,
            repetition_penalty: request.sampling.repetition_penalty,
            repetition_context,
        },
        max_new_tokens: request.detail_budget.max_new_tokens,
        seed: request.sampling.seed,
        cancel,
        ..TextLlmRequest::default()
    };
    Ok(StarVectorRequest::new(
        text_request,
        max_svg_bytes,
        Duration::from_millis(request.detail_budget.max_wall_time_ms),
    ))
}

/// Mirror the worker queue's established generation cancel flag into core-llm's independent flag
/// while a cached text provider loads and decodes. The monitor owns no model and is joined before
/// return; it exists only because the two tensor-free contracts intentionally use distinct flag
/// types.
fn mirror_vector_cancel<R>(
    source: gen_core::CancelFlag,
    target: gen_core::core_llm::CancelFlag,
    run: impl FnOnce() -> WorkerResult<R>,
) -> WorkerResult<R> {
    let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    std::thread::scope(|scope| {
        let finished_for_monitor = finished.clone();
        let target_for_monitor = target.clone();
        let monitor = scope.spawn(move || {
            while !finished_for_monitor.load(std::sync::atomic::Ordering::Acquire) {
                if source.is_cancelled() {
                    target_for_monitor.cancel();
                    break;
                }
                std::thread::park_timeout(Duration::from_millis(10));
            }
        });
        let result = run();
        finished.store(true, std::sync::atomic::Ordering::Release);
        monitor.thread().unpark();
        let _ = monitor.join();
        result
    })
}

fn classify_starvector_error(error: gen_core::core_llm::Error) -> WorkerError {
    if matches!(error, gen_core::core_llm::Error::Canceled) {
        WorkerError::Canceled(CANCEL_MESSAGE.to_owned())
    } else {
        WorkerError::Engine(format!("native StarVector generation failed: {error}"))
    }
}

fn validate_native_starvector_generation(
    output: StarVectorOutput,
    events: Vec<StarVectorStreamEvent>,
) -> WorkerResult<Vec<(String, u32)>> {
    let mut source = Vec::new();
    let mut done = None;
    for event in events {
        match event {
            StarVectorStreamEvent::Progress { .. } => {}
            StarVectorStreamEvent::Source { text, index } => {
                if done.is_some() {
                    return Err(WorkerError::Engine(
                        "native StarVector emitted source after Done".to_owned(),
                    ));
                }
                source.push((text, index));
            }
            StarVectorStreamEvent::Done {
                finish_reason,
                generated_tokens,
                generated_bytes,
            } => {
                if done
                    .replace((finish_reason, generated_tokens, generated_bytes))
                    .is_some()
                {
                    return Err(WorkerError::Engine(
                        "native StarVector emitted more than one Done event".to_owned(),
                    ));
                }
            }
        }
    }
    let (event_reason, event_tokens, event_bytes) = done
        .ok_or_else(|| WorkerError::Engine("native StarVector emitted no Done event".to_owned()))?;
    if (event_reason, event_tokens, event_bytes)
        != (
            output.finish_reason,
            output.generated_tokens,
            output.generated_bytes,
        )
    {
        return Err(WorkerError::Engine(
            "native StarVector Done counters disagree with its output".to_owned(),
        ));
    }
    match output.finish_reason {
        StarVectorFinishReason::CompleteRoot | StarVectorFinishReason::Eos => {
            let svg = output.svg.ok_or_else(|| {
                WorkerError::Engine(
                    "native StarVector completed without a publishable SVG".to_owned(),
                )
            })?;
            let streamed = source
                .iter()
                .map(|(text, _)| text.as_str())
                .collect::<String>();
            if streamed != svg || streamed.len() != output.generated_bytes {
                return Err(WorkerError::Engine(
                    "native StarVector streamed source disagrees with its output".to_owned(),
                ));
            }
            Ok(source)
        }
        // The provider's typed terminal outcome is retained by the native adapter.  Normal job
        // execution turns this into the same failure/cancellation outcome as before; the sealed
        // terminal campaign can instead record the non-publishable result with no attachments.
        StarVectorFinishReason::Cancelled
        | StarVectorFinishReason::TokenLimit
        | StarVectorFinishReason::ByteLimit
        | StarVectorFinishReason::WallTimeLimit => Ok(Vec::new()),
    }
}

fn ensure_provider_available(provider: &dyn MultimodalVectorProviderAdapter) -> WorkerResult<()> {
    if let Some(reason) = provider.unavailable_reason() {
        return Err(WorkerError::InvalidPayload(format!(
            "vector_backend_unavailable: {reason}"
        )));
    }
    Ok(())
}

fn manifest_declares_mode(payload: &VectorJobPayload) -> bool {
    payload
        .model_manifest_entry
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|capabilities| {
            capabilities
                .iter()
                .any(|capability| capability.as_str() == Some(payload.mode.as_str()))
        })
}

struct CollectedSvgSource {
    source: Option<String>,
    terminal: Option<TerminalProviderOutcome>,
}

fn record_vector_token_progress(
    progress: &tokio::sync::watch::Sender<u32>,
    count: u32,
    maximum: u32,
) {
    progress.send_modify(|previous| *previous = (*previous).max(count.min(maximum)));
}

/// Poll a single latest counter at most four times a second. The heartbeat runner remains
/// responsible for cancellation and task ownership while publication is pending.
async fn with_vector_progress<R, G, P, F>(
    progress: tokio::sync::watch::Receiver<u32>,
    maximum: u32,
    cancel: gen_core::CancelFlag,
    generation: G,
    mut publish: P,
) -> WorkerResult<R>
where
    G: std::future::Future<Output = WorkerResult<R>>,
    P: FnMut(u32, u32) -> F,
    F: std::future::Future<Output = WorkerResult<()>>,
{
    tokio::pin!(generation);
    let mut interval = tokio::time::interval(Duration::from_millis(250));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut reported = 0;
    loop {
        tokio::select! {
            result = &mut generation => return result,
            _ = interval.tick() => {
                let count = (*progress.borrow()).min(maximum);
                if count <= reported || cancel.is_cancelled() { continue; }
                if let Err(error) = publish(count, maximum).await {
                    // A progress POST failure cannot detach ongoing decode. Keep polling the
                    // heartbeat/owned task while cooperative cancel unwinds, with its usual bound.
                    cancel.cancel();
                    let _ = tokio::time::timeout(
                        crate::progress::CANCEL_JOIN_GRACE + crate::progress::CANCEL_JOIN_ABANDON,
                        &mut generation,
                    ).await;
                    return Err(error);
                }
                reported = count;
            }
        }
    }
}

fn vector_token_progress(count: u32, maximum: u32) -> ProgressRequest {
    let fraction = f64::from(count) / f64::from(maximum.max(1));
    let mut payload = progress_payload(
        JobStatus::Running,
        ProgressStage::Generating,
        0.25 + 0.45 * fraction,
        &format!("Generating SVG: {count} / {maximum} tokens."),
        None,
        None,
        None,
    );
    payload
        .extra
        .insert("generatedTokens".to_owned(), json!(count));
    payload
        .extra
        .insert("maxNewTokens".to_owned(), json!(maximum));
    payload
}

fn collect_svg_source(
    provider: &dyn MultimodalVectorProviderAdapter,
    request: &VectorProviderRequest,
    cancel: &gen_core::CancelFlag,
    progress: tokio::sync::watch::Sender<u32>,
) -> WorkerResult<CollectedSvgSource> {
    if request.model.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "vector model id is empty".to_owned(),
        ));
    }
    match request.mode {
        VectorMode::ImageToSvg => {
            let source_path = request.source_path.as_deref().ok_or_else(|| {
                WorkerError::InvalidPayload("image_to_svg source path is missing".to_owned())
            })?;
            if !source_path.is_file() {
                return Err(WorkerError::InvalidPayload(format!(
                    "image_to_svg source is no longer available: {}",
                    source_path.display()
                )));
            }
        }
        VectorMode::TextToSvg if request.prompt.trim().is_empty() => {
            return Err(WorkerError::InvalidPayload(
                "text_to_svg prompt is empty".to_owned(),
            ));
        }
        VectorMode::TextToSvg => {}
    }
    if !request.sampling.temperature.is_finite()
        || !request.sampling.top_p.is_finite()
        || !request.sampling.repetition_penalty.is_finite()
    {
        return Err(WorkerError::InvalidPayload(
            "vector sampling values must be finite".to_owned(),
        ));
    }
    if !provider.supports_mode(request.mode) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider {} does not declare {}",
            provider.provider_id(),
            request.mode.as_str()
        )));
    }
    let max_bytes = usize::try_from(request.detail_budget.max_svg_bytes)
        .map_err(|_| WorkerError::InvalidPayload("maxSvgBytes does not fit usize".to_owned()))?;
    let mut source = String::new();
    let mut previous_source_index = None;
    provider.generate_svg(request, cancel, progress, &mut |fragment, index| {
        if cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        // All native providers emit the static SVG prefix at zero. Decoder tokens that
        // produce no visible text leave gaps after it; they still count in token progress.
        let invalid_index = match previous_source_index {
            None => index != 0,
            Some(previous) => index <= previous,
        };
        if invalid_index {
            return Err(WorkerError::Engine(format!(
                "vector provider emitted non-monotonic source index {index}; previous {previous_source_index:?}"
            )));
        }
        previous_source_index = Some(index);
        let next_len = source
            .len()
            .checked_add(fragment.len())
            .ok_or_else(|| WorkerError::InvalidPayload("SVG byte count overflow".to_owned()))?;
        if next_len > max_bytes {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG exceeds the {max_bytes}-byte detail budget"
            )));
        }
        source.push_str(fragment);
        Ok(())
    })?;
    if cancel.is_cancelled() {
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    let terminal = provider.terminal_outcome();
    if let Some(terminal) = &terminal {
        if !terminal.publishable() {
            if terminal.finish_reason == "cancelled" {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            if std::env::var("SCENEWORKS_TERMINAL_CAMPAIGN").as_deref() == Ok("1") {
                return Ok(CollectedSvgSource {
                    source: None,
                    terminal: Some(terminal.clone()),
                });
            }
            return match terminal.finish_reason {
                "token_limit" => Err(WorkerError::Engine(
                    "native StarVector stopped at the token limit; no partial SVG was published"
                        .to_owned(),
                )),
                "byte_limit" => Err(WorkerError::Engine(
                    "native StarVector stopped at the byte limit; no partial SVG was published"
                        .to_owned(),
                )),
                "wall_time_limit" => Err(WorkerError::Engine(
                    "native StarVector stopped at the wall-time limit; no partial SVG was published"
                        .to_owned(),
                )),
                _ => Err(WorkerError::Engine("unknown StarVector terminal outcome".to_owned())),
            };
        }
    }
    if source.trim().is_empty() {
        return Err(WorkerError::Engine(
            "vector provider returned no SVG source".to_owned(),
        ));
    }
    Ok(CollectedSvgSource {
        source: Some(source),
        terminal,
    })
}

pub(crate) async fn run_vector_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let payload: VectorJobPayload = serde_json::from_value(Value::Object(job.payload.clone()))
        .map_err(|error| WorkerError::InvalidPayload(format!("invalid VectorRequest: {error}")))?;
    if let Some(error) = crate::vector_admission::vector_admission_error(
        &payload.model_manifest_entry,
        &settings.gpu_id,
    )
    .await
    {
        return Err(WorkerError::InvalidPayload(error));
    }
    let provider = NativeStarVectorProvider::resolve(settings, &payload)?;
    run_vector_job_with_provider(api, settings, job, Arc::new(provider)).await
}

pub(crate) async fn run_vector_job_with_provider(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    provider: Arc<dyn MultimodalVectorProviderAdapter>,
) -> WorkerResult<()> {
    let payload: VectorJobPayload = serde_json::from_value(Value::Object(job.payload.clone()))
        .map_err(|error| WorkerError::InvalidPayload(format!("invalid VectorRequest: {error}")))?;
    if !manifest_declares_mode(&payload) {
        return Err(WorkerError::InvalidPayload(
            "selected model manifest does not declare the requested vector mode".to_owned(),
        ));
    }
    let manifest_adapter = payload
        .model_manifest_entry
        .get("adapter")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if manifest_adapter != provider.provider_id() {
        return Err(WorkerError::InvalidPayload(format!(
            "selected model declares provider {manifest_adapter}, but worker resolved {}",
            provider.provider_id()
        )));
    }
    // Keep this before project lookup, heartbeat, or any status transition: the catalog may be
    // installable before the feature train permanently pins the matching native providers, but a
    // queued/stale job must never make an unavailable backend look claimed.
    ensure_provider_available(provider.as_ref())?;
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&payload.project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_path = match (&payload.mode, payload.source_asset_id.as_deref()) {
        (VectorMode::ImageToSvg, Some(source_asset_id)) => {
            Some(store.resolve_asset_media_path(&payload.project_id, source_asset_id)?)
        }
        (VectorMode::ImageToSvg, None) => {
            return Err(WorkerError::InvalidPayload(
                "image_to_svg requires sourceAssetId".to_owned(),
            ))
        }
        (VectorMode::TextToSvg, None) => None,
        (VectorMode::TextToSvg, Some(_)) => {
            return Err(WorkerError::InvalidPayload(
                "text_to_svg does not accept sourceAssetId".to_owned(),
            ))
        }
    };
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.10,
            "Resolving vector provider and conditioning.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Generating,
            0.25,
            "Streaming SVG from the native vector provider.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let request = VectorProviderRequest {
        mode: payload.mode,
        model: payload.model.clone(),
        source_path,
        prompt: payload.prompt.clone(),
        sampling: payload.sampling.clone(),
        detail_budget: payload.detail_budget.clone(),
    };
    let cancel = gen_core::CancelFlag::new();
    let blocking_cancel = cancel.clone();
    let blocking_provider = provider.clone();
    let blocking_request = request.clone();
    let (progress_tx, progress_rx) = tokio::sync::watch::channel(0);
    let task = tokio::task::spawn_blocking(move || {
        collect_svg_source(
            blocking_provider.as_ref(),
            &blocking_request,
            &blocking_cancel,
            progress_tx,
        )
    });
    let generation = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel.clone()),
        CANCEL_MESSAGE,
        "vector provider stream",
        no_cancel_ack(),
        task,
    );
    let collected = with_vector_progress(
        progress_rx,
        request.detail_budget.max_new_tokens,
        cancel,
        generation,
        |count, maximum| async move {
            update_job(api, &job.id, vector_token_progress(count, maximum))
                .await
                .map(|_| ())
        },
    )
    .await?;
    let source_raster = match request.source_path.as_deref() {
        Some(path) => Some((path.to_owned(), tokio::fs::read(path).await?)),
        None => None,
    };

    if collected.source.is_none() {
        let terminal = collected.terminal.ok_or_else(|| {
            WorkerError::Engine("non-publishable vector result lacks terminal evidence".to_owned())
        })?;
        let evidence_dir = project_path.join(".terminal-evidence").join(&job.id);
        let staging = evidence_dir.with_extension("tmp");
        let transcript_path = evidence_dir.join("provider-terminal.json");
        let transcript = terminal_transcript_bytes(&terminal)?;
        let evidence_write: WorkerResult<()> = async {
            tokio::fs::create_dir_all(&staging).await?;
            tokio::fs::write(staging.join("provider-terminal.json"), &transcript).await?;
            tokio::fs::rename(&staging, &evidence_dir).await?;
            Ok(())
        }
        .await;
        if evidence_write.is_err() {
            let _ = tokio::fs::remove_dir_all(&staging).await;
        }
        evidence_write?;
        let result = json!({
            "terminalEvidence": add_source_raster_evidence(
                terminal_generation_limit_result(
                    &terminal,
                    (&transcript_path, &transcript),
                )?,
                source_raster.as_ref().map(|(path, bytes)| (path.as_path(), bytes.as_slice())),
            )?,
        })
        .as_object()
        .cloned()
        .expect("terminal evidence result is an object");
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Completed,
                ProgressStage::Completed,
                1.0,
                "Native vector provider reached a bounded terminal outcome without publication.",
                None,
                Some(result),
                None,
            ),
        )
        .await?;
        return Ok(());
    }
    let source = collected
        .source
        .expect("checked non-empty publishable vector source");

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Rendering,
            0.75,
            "Sanitizing SVG and rendering its preview.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let canonical = match sanitize_svg(&source) {
        Ok(canonical) => canonical,
        Err(WorkerError::InvalidPayload(reason))
            if std::env::var("SCENEWORKS_TERMINAL_CAMPAIGN").as_deref() == Ok("1") =>
        {
            let terminal = collected.terminal.as_ref().ok_or_else(|| {
                WorkerError::Engine(
                    "terminal campaign sanitizer rejection lacks provider terminal evidence"
                        .to_owned(),
                )
            })?;
            let evidence_dir = project_path.join(".terminal-evidence").join(&job.id);
            let staging = evidence_dir.with_extension("tmp");
            let transcript_path = evidence_dir.join("provider-terminal.json");
            let rejected_svg_path = evidence_dir.join("provider-output.svg");
            let transcript = terminal_transcript_bytes(terminal)?;
            let evidence_write: WorkerResult<()> = async {
                tokio::fs::create_dir_all(&staging).await?;
                tokio::fs::write(staging.join("provider-terminal.json"), &transcript).await?;
                tokio::fs::write(staging.join("provider-output.svg"), source.as_bytes()).await?;
                tokio::fs::rename(&staging, &evidence_dir).await?;
                Ok(())
            }
            .await;
            if evidence_write.is_err() {
                let _ = tokio::fs::remove_dir_all(&staging).await;
            }
            evidence_write?;
            let mut terminal_evidence =
                terminal_result(terminal, None, None, Some((&transcript_path, &transcript)));
            let object = terminal_evidence.as_object_mut().ok_or_else(|| {
                WorkerError::Engine("terminal rejection evidence must be an object".to_owned())
            })?;
            object.insert("accepted".to_owned(), Value::Bool(false));
            object.insert("outcome".to_owned(), Value::String("rejected".to_owned()));
            object.insert(
                "rejectionStage".to_owned(),
                Value::String("sanitizer".to_owned()),
            );
            object.insert("rejectionReason".to_owned(), Value::String(reason));
            object.insert(
                "rejectedSvgPath".to_owned(),
                Value::String(rejected_svg_path.to_string_lossy().into_owned()),
            );
            object.insert(
                "rejectedSvgSha256".to_owned(),
                Value::String(sha256_hex(source.as_bytes())),
            );
            let result = json!({
                "terminalEvidence": add_source_raster_evidence(
                    terminal_evidence,
                    source_raster
                        .as_ref()
                        .map(|(path, bytes)| (path.as_path(), bytes.as_slice())),
                )?,
            })
            .as_object()
            .cloned()
            .expect("terminal rejection result is an object");
            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Completed,
                    ProgressStage::Completed,
                    1.0,
                    "Native vector output was rejected by the SVG sanitizer without publication.",
                    None,
                    Some(result),
                    None,
                ),
            )
            .await?;
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    let created_at = now_rfc3339();
    let asset_id = fresh_asset_id();
    let generation_set_id = format!("genset_{}", uuid::Uuid::new_v4().simple());
    let base = project_path
        .join("assets")
        .join("images")
        .join(&generation_set_id);
    let staging = base.join(format!(".{asset_id}.tmp"));
    let published = base.join(&asset_id);
    let svg_path = staging.join("vector.svg");
    let preview_path = staging.join("preview.png");
    let transcript_path = staging.join("provider-terminal.json");

    let publish_result: WorkerResult<()> = async {
        tokio::fs::create_dir_all(&staging).await?;
        tokio::fs::write(&svg_path, canonical.svg.as_bytes()).await?;
        render_preview(
            &canonical.svg,
            canonical.width,
            canonical.height,
            &preview_path,
        )
        .await?;
        if let Some(terminal) = &collected.terminal {
            tokio::fs::write(&transcript_path, terminal_transcript_bytes(terminal)?).await?;
        }
        check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
        tokio::fs::rename(&staging, &published).await?;
        Ok(())
    }
    .await;
    if publish_result.is_err() {
        let _ = tokio::fs::remove_dir_all(&staging).await;
    }
    publish_result?;

    let terminal_evidence = if std::env::var("SCENEWORKS_TERMINAL_CAMPAIGN").as_deref() == Ok("1") {
        let terminal = collected.terminal.as_ref().ok_or_else(|| {
            WorkerError::Engine(
                "terminal campaign native result lacks provider terminal evidence".to_owned(),
            )
        })?;
        let canonical_disk_path = published.join("vector.svg");
        let preview_disk_path = published.join("preview.png");
        let transcript_disk_path = published.join("provider-terminal.json");
        let canonical_bytes = tokio::fs::read(&canonical_disk_path).await?;
        let preview_bytes = tokio::fs::read(&preview_disk_path).await?;
        let transcript_bytes = tokio::fs::read(&transcript_disk_path).await?;
        Some(add_source_raster_evidence(
            terminal_result(
                terminal,
                Some((&canonical_disk_path, &canonical_bytes)),
                Some((&preview_disk_path, &preview_bytes)),
                Some((&transcript_disk_path, &transcript_bytes)),
            ),
            source_raster
                .as_ref()
                .map(|(path, bytes)| (path.as_path(), bytes.as_slice())),
        )?)
    } else {
        None
    };

    let media_path = format!("assets/images/{generation_set_id}/{asset_id}/vector.svg");
    let preview_path = format!("assets/images/{generation_set_id}/{asset_id}/preview.png");
    let fact = json!({
        "assetId": asset_id,
        "type": "vector",
        "mediaPath": media_path,
        "mimeType": "image/svg+xml",
        "width": canonical.width,
        "height": canonical.height,
        "createdAt": created_at,
        "mode": payload.mode.as_str(),
        "model": payload.model,
        "adapter": provider.provider_id(),
        "prompt": payload.prompt,
        "negativePrompt": "",
        "sourceAssetId": payload.source_asset_id,
        "sampling": payload.sampling,
        "detailBudget": payload.detail_budget,
        "sanitizerVersion": VECTOR_SANITIZER_VERSION,
        "rendererVersion": VECTOR_RENDERER_VERSION,
        "workflow": payload.workflow,
        "count": 1,
        "normalizedWidth": canonical.width,
        "normalizedHeight": canonical.height,
        "preview": { "path": preview_path, "mimeType": "image/png", "width": canonical.width, "height": canonical.height },
    });
    let mut result = json!({
        "generationSetId": generation_set_id,
        "expectedCount": 1,
        "adapter": provider.provider_id(),
        "model": fact["model"],
        "generationSet": {
            "id": generation_set_id,
            "mode": fact["mode"],
            "model": fact["model"],
            "prompt": fact["prompt"],
            "negativePrompt": "",
            "count": 1,
            "createdAt": created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("vector result is an object");
    if let Some(terminal_evidence) = terminal_evidence {
        result.insert("terminalEvidence".to_owned(), terminal_evidence);
    }
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Vector SVG stored with a PNG preview.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

#[derive(Debug)]
struct CanonicalSvg {
    svg: String,
    width: u32,
    height: u32,
}

/// Process-local terminal-campaign view of the already-authoritative sanitizer.
/// This is intentionally doc-hidden and is not part of the HTTP/API surface.
#[doc(hidden)]
pub struct TerminalSanitizedSvg {
    pub canonical_svg: String,
    pub width: u32,
    pub height: u32,
}

/// Delegate raw UTF-8 bytes to the production SVG sanitizer without a model.
#[doc(hidden)]
pub fn terminal_sanitize_svg_bytes(input: &[u8]) -> Result<TerminalSanitizedSvg, String> {
    let canonical = sanitize_svg_bytes(input).map_err(|error| error.to_string())?;
    Ok(TerminalSanitizedSvg {
        canonical_svg: canonical.svg,
        width: canonical.width,
        height: canonical.height,
    })
}

/// Atomically publish the canonical SVG and preview pair for an inert result.
#[doc(hidden)]
pub async fn terminal_write_sanitized_pair(
    value: &TerminalSanitizedSvg,
    destination: &Path,
) -> Result<(PathBuf, PathBuf), String> {
    terminal_write_sanitized_pair_with_preview_size(value, destination, None).await
}

/// Terminal comparison raster only: fit the intrinsic SVG into a bounded square without changing
/// canonical source or the ordinary product preview. The same resvg renderer handles both paths.
#[doc(hidden)]
pub async fn terminal_write_sanitized_pair_with_preview_size(
    value: &TerminalSanitizedSvg,
    destination: &Path,
    preview_size: Option<u32>,
) -> Result<(PathBuf, PathBuf), String> {
    if preview_size.is_some_and(|size| size == 0 || size > MAX_PREVIEW_DIMENSION) {
        return Err(format!(
            "terminal preview size must be 1..={MAX_PREVIEW_DIMENSION}"
        ));
    }
    let parent = destination
        .parent()
        .ok_or_else(|| "terminal sanitizer destination has no parent".to_owned())?;
    let staging = parent.join(format!(
        ".terminal-sanitize-{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    let result: WorkerResult<(PathBuf, PathBuf)> = async {
        tokio::fs::create_dir_all(&staging).await?;
        let svg = staging.join("canonical.svg");
        let preview = staging.join("preview.png");
        tokio::fs::write(&svg, value.canonical_svg.as_bytes()).await?;
        render_preview_with_size(
            &value.canonical_svg,
            value.width,
            value.height,
            &preview,
            preview_size,
        )
        .await?;
        tokio::fs::rename(&staging, destination).await?;
        Ok((
            destination.join("canonical.svg"),
            destination.join("preview.png"),
        ))
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_dir_all(&staging).await;
    }
    result.map_err(|error| error.to_string())
}

#[derive(Default)]
struct SanitizerBudget {
    attributes: usize,
    attribute_value_bytes: usize,
    path_data_bytes: usize,
    path_commands: usize,
    path_numbers: usize,
    point_numbers: usize,
    transform_numbers: usize,
    dash_numbers: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SvgResourceKind {
    ClipPath,
    LinearGradient,
    Filter,
}

#[derive(Debug)]
struct SvgResourceReference {
    source: Option<String>,
    target: String,
    expected: SvgResourceKind,
}

#[derive(Debug)]
struct SvgResourceCatalog {
    ids: BTreeMap<String, SvgResourceKind>,
    width: u32,
    height: u32,
}

fn sanitize_svg(input: &str) -> WorkerResult<CanonicalSvg> {
    sanitize_svg_bytes(input.as_bytes())
}

#[derive(Debug)]
struct SvgIndexFrame {
    hidden: bool,
    filter_id: Option<String>,
    resource_owner: Option<String>,
}

fn attribute_value<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value.as_str())
}

fn validate_resource_id(value: &str, label: &str) -> WorkerResult<()> {
    let mut bytes = value.bytes();
    if value.len() > 128
        || !bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} is not a bounded local identifier"
        )));
    }
    Ok(())
}

fn local_url_target(value: &str) -> Option<&str> {
    value
        .strip_prefix("url(#")
        .and_then(|value| value.strip_suffix(')'))
        .filter(|value| !value.is_empty())
}

fn local_href_target(value: &str) -> Option<&str> {
    value.strip_prefix('#').filter(|value| !value.is_empty())
}

fn validate_filter_input(value: &str, results: &BTreeSet<String>, label: &str) -> WorkerResult<()> {
    if matches!(value, "SourceGraphic" | "SourceAlpha") || results.contains(value) {
        Ok(())
    } else {
        Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} must reference an earlier filter result"
        )))
    }
}

fn index_filter_primitive(
    name: &str,
    attrs: &[(String, String)],
    filter_id: Option<&str>,
    filter_results: &mut BTreeMap<String, BTreeSet<String>>,
    filter_primitives: &mut usize,
) -> WorkerResult<()> {
    if !matches!(
        name,
        "feGaussianBlur" | "feOffset" | "feMerge" | "feMergeNode"
    ) {
        return Ok(());
    }
    let filter_id = filter_id.ok_or_else(|| {
        WorkerError::InvalidPayload(format!("provider SVG element <{name}> is outside a filter"))
    })?;
    *filter_primitives = filter_primitives.checked_add(1).ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG filter primitive count overflow".to_owned())
    })?;
    if *filter_primitives > MAX_SVG_FILTER_PRIMITIVES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG exceeds the filter-primitive budget".to_owned(),
        ));
    }
    let results = filter_results.entry(filter_id.to_owned()).or_default();
    if let Some(input) = attribute_value(attrs, "in") {
        validate_filter_input(input, results, name)?;
    } else if matches!(name, "feGaussianBlur" | "feOffset" | "feMergeNode") {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {name} requires an input"
        )));
    }
    if let Some(result) = attribute_value(attrs, "result") {
        if !matches!(name, "feGaussianBlur" | "feOffset") {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG {name} cannot name a result"
            )));
        }
        validate_resource_id(result, "filter result")?;
        if !results.insert(result.to_owned()) {
            return Err(WorkerError::InvalidPayload(
                "provider SVG filter result is duplicated".to_owned(),
            ));
        }
    } else if matches!(name, "feGaussianBlur" | "feOffset") {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {name} requires a result"
        )));
    }
    Ok(())
}

fn visit_resource(
    id: &str,
    edges: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    complete: &mut BTreeSet<String>,
) -> WorkerResult<()> {
    if complete.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id.to_owned()) {
        return Err(WorkerError::InvalidPayload(
            "provider SVG resource cycle is not allowed".to_owned(),
        ));
    }
    if let Some(targets) = edges.get(id) {
        for target in targets {
            visit_resource(target, edges, visiting, complete)?;
        }
    }
    visiting.remove(id);
    complete.insert(id.to_owned());
    Ok(())
}

fn validate_resource_cycles(references: &[SvgResourceReference]) -> WorkerResult<()> {
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for reference in references {
        if let Some(source) = &reference.source {
            edges
                .entry(source.clone())
                .or_default()
                .push(reference.target.clone());
        }
    }
    let mut visiting = BTreeSet::new();
    let mut complete = BTreeSet::new();
    for id in edges.keys() {
        visit_resource(id, &edges, &mut visiting, &mut complete)?;
    }
    Ok(())
}

fn index_svg_resources(input: &str) -> WorkerResult<SvgResourceCatalog> {
    let mut reader = Reader::from_reader(input.as_bytes());
    reader.config_mut().trim_text(true);
    let mut buffer = Vec::new();
    let mut stack: Vec<SvgIndexFrame> = Vec::new();
    let mut budget = SanitizerBudget::default();
    let mut elements = 0usize;
    let mut ids = BTreeMap::new();
    let mut id_counts = BTreeMap::new();
    let mut references = Vec::new();
    let mut filter_results = BTreeMap::new();
    let mut resources = 0usize;
    let mut gradient_stops = 0usize;
    let mut filter_primitives = 0usize;
    let mut dimensions = None;
    let mut xlink_namespace_bound = false;

    loop {
        let (event, empty) = match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => (event, false),
            Ok(Event::Empty(event)) => (event, true),
            Ok(Event::End(_)) => {
                stack.pop();
                buffer.clear();
                continue;
            }
            Ok(Event::Eof) => break,
            Ok(_) => {
                buffer.clear();
                continue;
            }
            Err(error) => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG is malformed: {error}"
                )))
            }
        };
        if stack.len() >= MAX_SVG_DEPTH {
            return Err(WorkerError::InvalidPayload(
                "provider SVG exceeds the element nesting budget".to_owned(),
            ));
        }
        elements += 1;
        if elements > MAX_SVG_ELEMENTS {
            return Err(WorkerError::InvalidPayload(
                "provider SVG exceeds the element budget".to_owned(),
            ));
        }
        let name = std::str::from_utf8(event.name().as_ref())
            .map_err(|_| WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned()))?
            .to_owned();
        let is_root = stack.is_empty();
        let attrs = source_attributes(
            &reader,
            &event,
            source_attribute_limit(&name, is_root),
            &mut budget,
        )?;
        if is_root && name == "svg" {
            let mut root_attrs = attrs.clone();
            normalize_root_dimensions(&mut root_attrs)?;
            dimensions = Some(svg_dimensions(&root_attrs)?);
            xlink_namespace_bound =
                attribute_value(&attrs, "xmlns:xlink") == Some("http://www.w3.org/1999/xlink");
        }
        if attrs.iter().any(|(key, _)| key == "xlink:href") && !xlink_namespace_bound {
            return Err(WorkerError::InvalidPayload(
                "provider SVG xlink:href requires the exact root xmlns:xlink binding".to_owned(),
            ));
        }
        if let Some(id) = attribute_value(&attrs, "id") {
            *id_counts.entry(id.to_owned()).or_insert(0usize) += 1;
        }
        let hidden = stack.last().is_some_and(|frame| frame.hidden)
            || (name == "g" && attribute_value(&attrs, "display") == Some("none"));
        let resource_kind = if hidden {
            None
        } else {
            match name.as_str() {
                "clipPath" => Some(SvgResourceKind::ClipPath),
                "linearGradient" => Some(SvgResourceKind::LinearGradient),
                "filter" => Some(SvgResourceKind::Filter),
                _ => None,
            }
        };
        let resource_id = resource_kind
            .map(|kind| {
                let id = attribute_value(&attrs, "id").ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "provider SVG resource <{name}> requires an id"
                    ))
                })?;
                validate_resource_id(id, "resource id")?;
                resources = resources.checked_add(1).ok_or_else(|| {
                    WorkerError::InvalidPayload("provider SVG resource count overflow".to_owned())
                })?;
                if resources > MAX_SVG_RESOURCES {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the resource budget".to_owned(),
                    ));
                }
                if ids.insert(id.to_owned(), kind).is_some() {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG resource id is duplicated".to_owned(),
                    ));
                }
                Ok(id.to_owned())
            })
            .transpose()?;
        if !hidden && name == "stop" {
            gradient_stops = gradient_stops.checked_add(1).ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG gradient stop count overflow".to_owned())
            })?;
            if gradient_stops > MAX_SVG_GRADIENT_STOPS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the gradient-stop budget".to_owned(),
                ));
            }
        }
        let filter_id = if name == "filter" {
            resource_id.clone()
        } else {
            stack.last().and_then(|frame| frame.filter_id.clone())
        };
        let resource_owner = resource_id
            .clone()
            .or_else(|| stack.last().and_then(|frame| frame.resource_owner.clone()));
        if !hidden {
            index_filter_primitive(
                &name,
                &attrs,
                filter_id.as_deref(),
                &mut filter_results,
                &mut filter_primitives,
            )?;
            for (key, value) in &attrs {
                if key == "style" {
                    for declaration in value.split(';') {
                        let Some((property, property_value)) = declaration.split_once(':') else {
                            continue;
                        };
                        if property.trim() != "fill" {
                            continue;
                        }
                        let Some(target) = local_url_target(property_value.trim()) else {
                            continue;
                        };
                        validate_resource_id(target, "resource reference")?;
                        if references.len() >= MAX_SVG_RESOURCE_REFERENCES {
                            return Err(WorkerError::InvalidPayload(
                                "provider SVG exceeds the resource-reference budget".to_owned(),
                            ));
                        }
                        references.push(SvgResourceReference {
                            source: resource_owner.clone(),
                            target: target.to_owned(),
                            expected: SvgResourceKind::LinearGradient,
                        });
                    }
                }
                let expected = match key.as_str() {
                    "fill" => local_url_target(value).map(|_| SvgResourceKind::LinearGradient),
                    "clip-path" => local_url_target(value).map(|_| SvgResourceKind::ClipPath),
                    "filter" => local_url_target(value).map(|_| SvgResourceKind::Filter),
                    "href" | "xlink:href" if name == "linearGradient" => {
                        local_href_target(value).map(|_| SvgResourceKind::LinearGradient)
                    }
                    _ => None,
                };
                if let Some(expected) = expected {
                    let target = if matches!(key.as_str(), "href" | "xlink:href") {
                        local_href_target(value)
                    } else {
                        local_url_target(value)
                    }
                    .expect("matched above");
                    validate_resource_id(target, "resource reference")?;
                    if references.len() >= MAX_SVG_RESOURCE_REFERENCES {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG exceeds the resource-reference budget".to_owned(),
                        ));
                    }
                    references.push(SvgResourceReference {
                        source: resource_owner.clone(),
                        target: target.to_owned(),
                        expected,
                    });
                }
            }
        }
        if !empty {
            stack.push(SvgIndexFrame {
                hidden,
                filter_id,
                resource_owner,
            });
        }
        buffer.clear();
    }
    let (width, height) = dimensions
        .ok_or_else(|| WorkerError::InvalidPayload("provider output has no SVG root".to_owned()))?;
    for reference in &references {
        if id_counts.get(&reference.target) != Some(&1)
            || ids.get(&reference.target) != Some(&reference.expected)
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG local resource reference #{} is unresolved or has the wrong type",
                reference.target
            )));
        }
    }
    validate_resource_cycles(&references)?;
    Ok(SvgResourceCatalog { ids, width, height })
}

fn sanitize_svg_bytes(input: &[u8]) -> WorkerResult<CanonicalSvg> {
    if input.len() > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG exceeds the 256 KiB sanitizer budget".to_owned(),
        ));
    }
    let input = std::str::from_utf8(input)
        .map_err(|_| WorkerError::InvalidPayload("provider SVG is not valid UTF-8".to_owned()))?;
    let resources = index_svg_resources(input)?;
    let mut reader = Reader::from_reader(input.as_bytes());
    reader.config_mut().trim_text(true);
    let mut buffer = Vec::new();
    let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let mut stack: Vec<RawSvgElement> = Vec::new();
    let mut elements = 0usize;
    let mut budget = SanitizerBudget::default();
    let mut dimensions = None;
    let mut root_seen = false;
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => {
                let event_name = event.name();
                let raw_name = std::str::from_utf8(event_name.as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                if stack.len() >= MAX_SVG_DEPTH {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element nesting budget".to_owned(),
                    ));
                }
                elements += 1;
                if elements > MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element budget".to_owned(),
                    ));
                }
                let is_root = stack.is_empty();
                let source_attrs = source_attributes(
                    &reader,
                    &event,
                    source_attribute_limit(&raw_name, is_root),
                    &mut budget,
                )?;
                let kind =
                    classify_element(&raw_name, stack.last(), &source_attrs, false, root_seen)?;
                if kind.emitted() {
                    let (name, mut attrs) = canonical_emitted_element(
                        kind,
                        &raw_name,
                        source_attrs,
                        is_root,
                        &resources,
                        &mut budget,
                    )?;
                    if is_root {
                        root_seen = true;
                        if !attrs
                            .iter()
                            .any(|(key, value)| key == "xmlns" && value == SVG_NAMESPACE)
                        {
                            attrs.push(("xmlns".to_owned(), SVG_NAMESPACE.to_owned()));
                            attrs.sort_unstable();
                        }
                        if attrs.len() > MAX_SVG_ATTRIBUTES_PER_ELEMENT {
                            return Err(WorkerError::InvalidPayload(
                                "provider SVG element <svg> exceeds the per-element attribute budget"
                                    .to_owned(),
                            ));
                        }
                        dimensions = Some(svg_dimensions(&attrs)?);
                    }
                    write_start(&mut output, &name, &attrs, false);
                } else if kind == RawSvgElementKind::Hidden {
                    validate_hidden_attributes(&raw_name, &source_attrs, &mut budget)?;
                } else {
                    validate_discarded_attributes(kind, &source_attrs)?;
                }
                stack.push(RawSvgElement {
                    name: raw_name,
                    kind,
                });
            }
            Ok(Event::Empty(event)) => {
                let event_name = event.name();
                let raw_name = std::str::from_utf8(event_name.as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                if stack.len() >= MAX_SVG_DEPTH {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element nesting budget".to_owned(),
                    ));
                }
                elements += 1;
                if elements > MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element budget".to_owned(),
                    ));
                }
                let source_attrs = source_attributes(
                    &reader,
                    &event,
                    source_attribute_limit(&raw_name, false),
                    &mut budget,
                )?;
                let kind =
                    classify_element(&raw_name, stack.last(), &source_attrs, true, root_seen)?;
                if kind.emitted() {
                    let (name, attrs) = canonical_emitted_element(
                        kind,
                        &raw_name,
                        source_attrs,
                        false,
                        &resources,
                        &mut budget,
                    )?;
                    write_start(&mut output, &name, &attrs, true);
                } else if kind == RawSvgElementKind::Hidden {
                    validate_hidden_attributes(&raw_name, &source_attrs, &mut budget)?;
                } else {
                    validate_discarded_attributes(kind, &source_attrs)?;
                }
            }
            Ok(Event::End(event)) => {
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                let Some(open) = stack.pop() else {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has mismatched tags".to_owned(),
                    ));
                };
                if open.name != name {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has mismatched tags".to_owned(),
                    ));
                }
                if open.kind.emitted() {
                    output.push_str("</");
                    output.push_str(&name);
                    output.push('>');
                }
            }
            Ok(Event::Text(text)) => {
                let bytes: &[u8] = text.as_ref();
                let current = stack.last().map(|element| element.kind);
                let dc_format = current == Some(RawSvgElementKind::DcFormat);
                if dc_format {
                    if bytes != b"image/svg+xml" {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG dc:format metadata is not recognized".to_owned(),
                        ));
                    }
                } else if current == Some(RawSvgElementKind::Metadata) {
                    // Exporters such as potrace place inert provenance text directly in the
                    // metadata container. The entire container is omitted from canonical output;
                    // its children and attributes still pass the strict metadata classifiers.
                } else if matches!(
                    current,
                    Some(RawSvgElementKind::Title | RawSvgElementKind::DcTitle)
                ) {
                    // Plain title text is inert metadata and is deliberately omitted.
                } else if !bytes.iter().all(u8::is_ascii_whitespace) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG text nodes are not supported".to_owned(),
                    ));
                }
            }
            Ok(Event::Comment(_)) | Ok(Event::Decl(_)) => {}
            Ok(Event::Eof) => break,
            Ok(Event::CData(_))
            | Ok(Event::DocType(_))
            | Ok(Event::PI(_))
            | Ok(Event::GeneralRef(_)) => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG contains a disallowed XML construct".to_owned(),
                ));
            }
            Err(error) => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG is malformed: {error}"
                )))
            }
        }
        buffer.clear();
    }
    if !stack.is_empty() || dimensions.is_none() {
        return Err(WorkerError::InvalidPayload(
            "provider SVG is incomplete".to_owned(),
        ));
    }
    // A second parse through the renderer proves the canonical subset is renderable before any
    // write. usvg does not load network resources, and our whitelist already removed every URL.
    let tree = usvg::Tree::from_str(&output, &usvg::Options::default()).map_err(|error| {
        WorkerError::InvalidPayload(format!("provider SVG cannot be rendered: {error}"))
    })?;
    let (width, height) = dimensions.expect("validated above");
    validate_filter_render_budget(
        &tree,
        width,
        height,
        resvg::tiny_skia::Transform::identity(),
    )?;
    Ok(CanonicalSvg {
        svg: output,
        width,
        height,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RawSvgElementKind {
    Retained,
    Defs,
    ResourceDefs,
    ClipPath,
    LinearGradient,
    GradientStop,
    Filter,
    FeGaussianBlur,
    FeOffset,
    FeMerge,
    FeMergeNode,
    Hidden,
    NamedView,
    Grid,
    Metadata,
    RdfLower,
    RdfUpper,
    CcLower,
    CcUpper,
    DcFormat,
    DcType,
    DcTitle,
    Title,
}

impl RawSvgElementKind {
    fn emitted(self) -> bool {
        matches!(
            self,
            Self::Retained
                | Self::ResourceDefs
                | Self::ClipPath
                | Self::LinearGradient
                | Self::GradientStop
                | Self::Filter
                | Self::FeGaussianBlur
                | Self::FeOffset
                | Self::FeMerge
                | Self::FeMergeNode
        )
    }
}

#[derive(Debug)]
struct RawSvgElement {
    name: String,
    kind: RawSvgElementKind,
}

fn source_attribute_limit(name: &str, is_root: bool) -> usize {
    if (is_root && name == "svg") || name == "sodipodi:namedview" {
        24
    } else {
        MAX_SVG_ATTRIBUTES_PER_ELEMENT
    }
}

fn source_attributes(
    reader: &Reader<&[u8]>,
    event: &quick_xml::events::BytesStart<'_>,
    per_element_limit: usize,
    budget: &mut SanitizerBudget,
) -> WorkerResult<Vec<(String, String)>> {
    let mut attrs = Vec::new();
    for attribute in event.attributes().with_checks(true) {
        let attribute = attribute.map_err(|error| {
            WorkerError::InvalidPayload(format!("provider SVG attribute is malformed: {error}"))
        })?;
        let key = std::str::from_utf8(attribute.key.as_ref())
            .map_err(|_| {
                WorkerError::InvalidPayload("provider SVG attribute is not UTF-8".to_owned())
            })?
            .to_owned();
        let value = attribute
            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("provider SVG attribute is invalid: {error}"))
            })?
            .into_owned();
        if value.len() > MAX_SVG_ATTRIBUTE_VALUE_BYTES {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} exceeds the per-value byte budget"
            )));
        }
        budget.attributes = budget.attributes.checked_add(1).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG attribute count overflow".to_owned())
        })?;
        budget.attribute_value_bytes = budget
            .attribute_value_bytes
            .checked_add(value.len())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG attribute bytes overflow".to_owned())
            })?;
        if budget.attributes > MAX_SVG_ATTRIBUTES {
            return Err(WorkerError::InvalidPayload(
                "provider SVG exceeds the total attribute budget".to_owned(),
            ));
        }
        if budget.attribute_value_bytes > MAX_SVG_ATTRIBUTE_VALUE_BYTES_TOTAL {
            return Err(WorkerError::InvalidPayload(
                "provider SVG exceeds the total attribute-value byte budget".to_owned(),
            ));
        }
        if attrs.iter().any(|(existing, _)| existing == &key) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is duplicated"
            )));
        }
        attrs.push((key, value));
    }
    if attrs.len() > per_element_limit {
        return Err(WorkerError::InvalidPayload(
            "provider SVG element exceeds the per-element attribute budget".to_owned(),
        ));
    }
    Ok(attrs)
}

fn one_number(value: &str, label: &str, allow_px: bool) -> WorkerResult<f64> {
    let values = parse_number_list(value, allow_px, label, true)?;
    if values.len() != 1 {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} must contain exactly one finite number"
        )));
    }
    Ok(values[0])
}

fn number_or_percentage(value: &str, label: &str) -> WorkerResult<f64> {
    if let Some(number) = value.strip_suffix('%') {
        one_number(number, label, false)
    } else {
        one_number(value, label, false)
    }
}

fn canonical_stop_style(value: &str) -> WorkerResult<Vec<(String, String)>> {
    let mut attrs = Vec::new();
    let declarations = value.split(';').collect::<Vec<_>>();
    for (index, declaration) in declarations.iter().enumerate() {
        if declaration.trim().is_empty() {
            if index + 1 == declarations.len() && value.ends_with(';') {
                continue;
            }
            return Err(WorkerError::InvalidPayload(
                "provider SVG gradient stop style is malformed".to_owned(),
            ));
        }
        let (key, value) = declaration.split_once(':').ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG gradient stop style is malformed".to_owned())
        })?;
        let key = key.trim();
        let value = value.trim();
        if !matches!(key, "stop-color" | "stop-opacity")
            || value.is_empty()
            || value.contains(':')
            || unsafe_attribute_value(value)
            || attrs.iter().any(|(existing, _)| existing == key)
        {
            return Err(WorkerError::InvalidPayload(
                "provider SVG gradient stop style is not allowed".to_owned(),
            ));
        }
        if key == "stop-opacity" {
            let opacity = one_number(value, "stop-opacity", false)?;
            if !(0.0..=1.0).contains(&opacity) {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stop-opacity must be between zero and one".to_owned(),
                ));
            }
        }
        attrs.push((key.to_owned(), value.to_owned()));
    }
    if attrs.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "provider SVG gradient stop style is empty".to_owned(),
        ));
    }
    Ok(attrs)
}

fn canonical_resource_element(
    kind: RawSvgElementKind,
    name: &str,
    source_attrs: Vec<(String, String)>,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    let mut attrs = Vec::new();
    let expected_id_kind = match kind {
        RawSvgElementKind::ClipPath => Some(SvgResourceKind::ClipPath),
        RawSvgElementKind::LinearGradient => Some(SvgResourceKind::LinearGradient),
        RawSvgElementKind::Filter => Some(SvgResourceKind::Filter),
        _ => None,
    };
    for (key, value) in source_attrs {
        if unsafe_attribute_value(&value)
            && !(kind == RawSvgElementKind::GradientStop && key == "style")
            && !(matches!(key.as_str(), "href" | "xlink:href")
                && local_href_target(&value).is_some())
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG resource attribute {key} is not allowed"
            )));
        }
        match kind {
            RawSvgElementKind::ResourceDefs | RawSvgElementKind::FeMerge => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG element <{name}> does not allow attributes"
                )));
            }
            RawSvgElementKind::ClipPath => match key.as_str() {
                "id" => attrs.push((key, value)),
                "clipPathUnits"
                    if matches!(value.as_str(), "userSpaceOnUse" | "objectBoundingBox") =>
                {
                    attrs.push((key, value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG clipPath attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::LinearGradient => match key.as_str() {
                "id" => attrs.push((key, value)),
                "x1" | "y1" | "x2" | "y2" => {
                    number_or_percentage(&value, &key)?;
                    attrs.push((key, value));
                }
                "gradientUnits"
                    if matches!(value.as_str(), "userSpaceOnUse" | "objectBoundingBox") =>
                {
                    attrs.push((key, value));
                }
                "gradientTransform" => {
                    let numbers = validate_transform_list(&value)?;
                    budget.transform_numbers = budget
                        .transform_numbers
                        .checked_add(numbers)
                        .ok_or_else(|| {
                            WorkerError::InvalidPayload(
                                "provider SVG transform number overflow".to_owned(),
                            )
                        })?;
                    if budget.transform_numbers > MAX_SVG_TRANSFORM_NUMBERS {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG exceeds the transform-number budget".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                "href" | "xlink:href" => {
                    let target = local_href_target(&value).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG gradient href must be a local fragment".to_owned(),
                        )
                    })?;
                    if resources.ids.get(target) != Some(&SvgResourceKind::LinearGradient) {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG gradient href has the wrong resource type".to_owned(),
                        ));
                    }
                    attrs.push(("href".to_owned(), value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG linearGradient attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::GradientStop => match key.as_str() {
                "offset" => {
                    let percent = value.ends_with('%');
                    let offset = number_or_percentage(&value, "gradient stop offset")?;
                    let maximum = if percent { 100.0 } else { 1.0 };
                    if !(0.0..=maximum).contains(&offset) {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG gradient stop offset is outside its range".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                "stop-color" => attrs.push((key, value)),
                "stop-opacity" => {
                    let opacity = one_number(&value, "stop-opacity", false)?;
                    if !(0.0..=1.0).contains(&opacity) {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG stop-opacity must be between zero and one".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                "style" => {
                    for style_attr in canonical_stop_style(&value)? {
                        if attrs.iter().any(|(existing, _)| existing == &style_attr.0) {
                            return Err(WorkerError::InvalidPayload(
                                "provider SVG gradient stop style duplicates an attribute"
                                    .to_owned(),
                            ));
                        }
                        attrs.push(style_attr);
                    }
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG gradient stop attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::Filter => match key.as_str() {
                "id" | "width" | "height" => attrs.push((key, value)),
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG filter attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::FeGaussianBlur => match key.as_str() {
                "in" | "result" => attrs.push((key, value)),
                "stdDeviation" => {
                    let values = parse_number_list(&value, false, "filter blur", true)?;
                    if values.len() > 2
                        || values
                            .iter()
                            .any(|value| *value < 0.0 || *value > MAX_SVG_FILTER_BLUR_SIGMA)
                    {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG filter blur exceeds its bounded range".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG feGaussianBlur attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::FeOffset => match key.as_str() {
                "in" | "result" => attrs.push((key, value)),
                "dx" | "dy" => {
                    one_number(&value, &key, false)?;
                    attrs.push((key, value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG feOffset attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::FeMergeNode => match key.as_str() {
                "in" => attrs.push((key, value)),
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG feMergeNode attribute {key} is not allowed"
                    )))
                }
            },
            _ => unreachable!("resource canonicalizer called for {kind:?}"),
        }
    }
    if let Some(expected) = expected_id_kind {
        let id = attribute_value(&attrs, "id").ok_or_else(|| {
            WorkerError::InvalidPayload(format!("provider SVG resource <{name}> requires an id"))
        })?;
        if resources.ids.get(id) != Some(&expected) {
            return Err(WorkerError::InvalidPayload(
                "provider SVG resource id differs from the validated index".to_owned(),
            ));
        }
    }
    if kind == RawSvgElementKind::Filter {
        let width = attribute_value(&attrs, "width").ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG filter requires width".to_owned())
        })?;
        let height = attribute_value(&attrs, "height").ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG filter requires height".to_owned())
        })?;
        bounded_filter_axis(width, resources.width, "width")?;
        bounded_filter_axis(height, resources.height, "height")?;
    }
    let mut keys = BTreeSet::new();
    if attrs.iter().any(|(key, _)| !keys.insert(key.as_str())) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> has duplicate canonical attributes"
        )));
    }
    attrs.sort_unstable();
    Ok((name.to_owned(), attrs))
}

fn bounded_filter_axis(value: &str, viewport: u32, label: &str) -> WorkerResult<u32> {
    let pixels = if let Some(number) = value.strip_suffix('%') {
        let percentage = one_number(number, label, false)?;
        if !(0.0..=200.0).contains(&percentage) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG filter {label} exceeds 200%"
            )));
        }
        f64::from(viewport) * percentage / 100.0
    } else {
        one_number(value, label, false)?
    };
    if pixels <= 0.0 || pixels > f64::from(MAX_PREVIEW_DIMENSION) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG filter {label} exceeds the renderer dimension budget"
        )));
    }
    Ok(pixels.ceil() as u32)
}

fn canonical_emitted_element(
    kind: RawSvgElementKind,
    name: &str,
    attrs: Vec<(String, String)>,
    is_root: bool,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    if kind == RawSvgElementKind::Retained {
        canonical_element(name, attrs, is_root, resources, budget)
    } else {
        canonical_resource_element(kind, name, attrs, resources, budget)
    }
}

fn validate_hidden_attributes(
    element: &str,
    attrs: &[(String, String)],
    budget: &mut SanitizerBudget,
) -> WorkerResult<()> {
    for (key, value) in attrs {
        let inert = matches!(key.as_str(), "id" | "class" | "data-name");
        let hidden_group = element == "g"
            && matches!(
                key.as_str(),
                "display" | "overflow" | "x" | "y" | "width" | "height"
            );
        if !inert && !hidden_group && !allowed_attribute(element, key) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG hidden attribute {key} is not allowed"
            )));
        }
        let local_paint = key == "fill" && local_url_target(value).is_some();
        if !local_paint && unsafe_attribute_value(value) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG hidden attribute {key} is not allowed"
            )));
        }
        match key.as_str() {
            "display" if value != "none" => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG hidden display must be none".to_owned(),
                ))
            }
            "overflow" if !matches!(value.as_str(), "visible" | "hidden") => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG hidden overflow is not recognized".to_owned(),
                ))
            }
            "width" | "height" | "x" | "y" if value.ends_with('%') => {
                let percentage = number_or_percentage(value, key)?;
                if percentage.abs() > 100.0 {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG hidden percentage exceeds its budget".to_owned(),
                    ));
                }
            }
            _ if !inert && !hidden_group && !local_paint => {
                validate_attribute_resource_budget(element, key, value, budget)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn canonical_element(
    name: &str,
    mut source_attrs: Vec<(String, String)>,
    is_root: bool,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    if !matches!(
        name,
        "svg" | "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
    ) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> is not allowed"
        )));
    }
    if is_root {
        normalize_root_dimensions(&mut source_attrs)?;
    }
    let mut attrs = Vec::new();
    for (key, value) in source_attrs {
        if stripped_retained_attribute(name, &key, &value, is_root)? {
            continue;
        }
        if key == "style" {
            let style_attrs = canonical_presentation_style(name, &value, resources, budget)?;
            if style_attrs
                .iter()
                .any(|(key, _)| attrs.iter().any(|(existing, _)| existing == key))
            {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG style duplicates an attribute".to_owned(),
                ));
            }
            attrs.extend(style_attrs);
            continue;
        }
        let reference_kind = match key.as_str() {
            "fill" if local_url_target(&value).is_some() => Some(SvgResourceKind::LinearGradient),
            "clip-path" => Some(SvgResourceKind::ClipPath),
            "filter" => Some(SvgResourceKind::Filter),
            _ => None,
        };
        if let Some(expected) = reference_kind {
            if matches!(key.as_str(), "clip-path" | "filter")
                && !matches!(
                    name,
                    "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
                )
            {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} is not allowed on <{name}>"
                )));
            }
            let target = local_url_target(&value).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} must be an exact local resource reference"
                ))
            })?;
            if resources.ids.get(target) != Some(&expected) {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} has an unresolved or mistyped resource"
                )));
            }
            attrs.push((key, value));
            continue;
        }
        // The default namespace changes the meaning of unprefixed elements, so retain only SVG's
        // namespace on the root. It is syntax, never a dereferenced resource.
        let namespace_ok = key == "xmlns" && is_root && name == "svg" && value == SVG_NAMESPACE;
        if !allowed_attribute(name, &key)
            || (key == "xmlns" && !namespace_ok)
            || (!namespace_ok && unsafe_attribute_value(&value))
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is not allowed"
            )));
        }
        validate_attribute_resource_budget(name, &key, &value, budget)?;
        if attrs.iter().any(|(existing, _)| existing == &key) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is duplicated"
            )));
        }
        attrs.push((key, value));
    }
    if attrs.len() > MAX_SVG_ATTRIBUTES_PER_ELEMENT {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> exceeds the per-element attribute budget"
        )));
    }
    attrs.sort_unstable();
    Ok((name.to_owned(), attrs))
}

fn classify_element(
    name: &str,
    parent: Option<&RawSvgElement>,
    attrs: &[(String, String)],
    empty: bool,
    root_seen: bool,
) -> WorkerResult<RawSvgElementKind> {
    let invalid = || {
        WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> is not allowed in this context"
        ))
    };
    let Some(parent) = parent else {
        return if !root_seen && !empty && name == "svg" {
            Ok(RawSvgElementKind::Retained)
        } else {
            Err(WorkerError::InvalidPayload(
                "provider output root must be a non-empty <svg> element".to_owned(),
            ))
        };
    };
    if parent.kind == RawSvgElementKind::Hidden {
        return if matches!(
            name,
            "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
        ) {
            Ok(RawSvgElementKind::Hidden)
        } else if name == "title" {
            Ok(RawSvgElementKind::Title)
        } else {
            Err(invalid())
        };
    }
    if parent.kind == RawSvgElementKind::Retained
        && matches!(
            name,
            "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
        )
    {
        if name == "g" && attribute_value(attrs, "display") == Some("none") {
            return Ok(RawSvgElementKind::Hidden);
        }
        return Ok(RawSvgElementKind::Retained);
    }
    match (parent.kind, parent.name.as_str(), name, empty) {
        (RawSvgElementKind::Retained, "svg", "defs", true) => Ok(RawSvgElementKind::Defs),
        (RawSvgElementKind::Retained, "svg", "defs", false) => Ok(RawSvgElementKind::ResourceDefs),
        (RawSvgElementKind::ResourceDefs, "defs", "clipPath", false) => {
            Ok(RawSvgElementKind::ClipPath)
        }
        (RawSvgElementKind::ResourceDefs, "defs", "linearGradient", _) => {
            Ok(RawSvgElementKind::LinearGradient)
        }
        (RawSvgElementKind::ResourceDefs, "defs", "filter", false) => Ok(RawSvgElementKind::Filter),
        (
            RawSvgElementKind::ClipPath,
            "clipPath",
            "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon",
            _,
        ) => Ok(RawSvgElementKind::Retained),
        (RawSvgElementKind::LinearGradient, "linearGradient", "stop", true) => {
            Ok(RawSvgElementKind::GradientStop)
        }
        (RawSvgElementKind::Filter, "filter", "feGaussianBlur", true) => {
            Ok(RawSvgElementKind::FeGaussianBlur)
        }
        (RawSvgElementKind::Filter, "filter", "feOffset", true) => Ok(RawSvgElementKind::FeOffset),
        (RawSvgElementKind::Filter, "filter", "feMerge", false) => Ok(RawSvgElementKind::FeMerge),
        (RawSvgElementKind::FeMerge, "feMerge", "feMergeNode", true) => {
            Ok(RawSvgElementKind::FeMergeNode)
        }
        (RawSvgElementKind::Retained, "svg", "sodipodi:namedview", _) => {
            Ok(RawSvgElementKind::NamedView)
        }
        (RawSvgElementKind::Retained, _, "title", _) => Ok(RawSvgElementKind::Title),
        (RawSvgElementKind::NamedView, _, "inkscape:grid", true) => Ok(RawSvgElementKind::Grid),
        (RawSvgElementKind::Retained, "svg", "metadata", false) => Ok(RawSvgElementKind::Metadata),
        (RawSvgElementKind::Metadata, _, "rdf:rdf", false) => Ok(RawSvgElementKind::RdfLower),
        (RawSvgElementKind::Metadata, _, "rdf:RDF", false) => Ok(RawSvgElementKind::RdfUpper),
        (RawSvgElementKind::RdfLower, _, "cc:work", false) => Ok(RawSvgElementKind::CcLower),
        (RawSvgElementKind::RdfUpper, _, "cc:Work", false) => Ok(RawSvgElementKind::CcUpper),
        (RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper, _, "dc:format", false) => {
            Ok(RawSvgElementKind::DcFormat)
        }
        (RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper, _, "dc:type", true) => {
            Ok(RawSvgElementKind::DcType)
        }
        (RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper, _, "dc:title", _) => {
            Ok(RawSvgElementKind::DcTitle)
        }
        _ => Err(invalid()),
    }
}

fn stripped_retained_attribute(
    element: &str,
    key: &str,
    value: &str,
    is_root: bool,
) -> WorkerResult<bool> {
    if key == "xmlns:cc" {
        if is_root
            && element == "svg"
            && matches!(
                value,
                "http://creativecommons.org/ns#" | "http://web.resource.org/cc/"
            )
        {
            return Ok(true);
        }
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG attribute {key} is not allowed"
        )));
    }
    let known_namespace = match key {
        "xmlns:dc" => Some("http://purl.org/dc/elements/1.1/"),
        "xmlns:rdf" => Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
        "xmlns:svg" => Some(SVG_NAMESPACE),
        "xmlns:sodipodi" => Some("http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"),
        "xmlns:inkscape" => Some("http://www.inkscape.org/namespaces/inkscape"),
        "xmlns:xlink" => Some("http://www.w3.org/1999/xlink"),
        "xmlns:ns1" => Some("http://sozi.baierouge.fr"),
        _ => None,
    };
    if let Some(expected) = known_namespace {
        if !is_root || element != "svg" || value != expected {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is not allowed"
            )));
        }
        return Ok(true);
    }
    if is_root && element == "svg" {
        match key {
            "x" | "y" => {
                let values = parse_number_list(value, true, key, true)?;
                if values.as_slice() != [0.0] {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG outer {key} must be zero"
                    )));
                }
                return Ok(true);
            }
            "xml:space" if matches!(value, "default" | "preserve") => return Ok(true),
            "enable-background" => {
                validate_inert_enable_background(value)?;
                return Ok(true);
            }
            // This editor path is never emitted or dereferenced. Attribute byte budgets still
            // bound it, including legacy Windows paths containing backslashes.
            "sodipodi:docbase" => return Ok(true),
            _ => {}
        }
    }
    let stripped = match element {
        "svg" if is_root => matches!(
            key,
            "id" | "version"
                | "inkscape:version"
                | "sodipodi:docname"
                | "sodipodi:version"
                | "class"
                | "data-z"
                | "data-slots"
                | "data-tags"
        ),
        "g" => matches!(
            key,
            "inkscape:label" | "inkscape:groupmode" | "id" | "class" | "data-name"
        ),
        "path" => matches!(
            key,
            "id" | "class" | "data-name" | "inkscape:connector-curvature"
        ),
        "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon" => {
            matches!(key, "id" | "class" | "data-name")
        }
        _ => false,
    };
    if stripped && unsafe_attribute_value(value) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG attribute {key} is not allowed"
        )));
    }
    Ok(stripped)
}

fn validate_discarded_attributes(
    kind: RawSvgElementKind,
    attrs: &[(String, String)],
) -> WorkerResult<()> {
    for (key, value) in attrs {
        let allowed = match kind {
            RawSvgElementKind::Defs => key == "id",
            RawSvgElementKind::NamedView => matches!(
                key.as_str(),
                "id" | "pagecolor"
                    | "bordercolor"
                    | "borderopacity"
                    | "inkscape:pageopacity"
                    | "inkscape:pageshadow"
                    | "inkscape:zoom"
                    | "inkscape:cx"
                    | "inkscape:cy"
                    | "inkscape:document-units"
                    | "inkscape:current-layer"
                    | "showgrid"
                    | "units"
                    | "inkscape:window-width"
                    | "inkscape:window-height"
                    | "inkscape:window-x"
                    | "inkscape:window-y"
                    | "inkscape:window-maximized"
                    | "objecttolerance"
                    | "gridtolerance"
                    | "guidetolerance"
            ),
            RawSvgElementKind::Grid => matches!(
                key.as_str(),
                "type"
                    | "id"
                    | "empspacing"
                    | "visible"
                    | "enabled"
                    | "snapvisiblegridlinesonly"
                    | "units"
                    | "spacingx"
                    | "spacingy"
                    | "originx"
                    | "originy"
            ),
            RawSvgElementKind::Metadata => key == "id",
            RawSvgElementKind::RdfLower | RawSvgElementKind::RdfUpper => false,
            RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper => {
                key == "rdf:about" && value.is_empty()
            }
            RawSvgElementKind::DcFormat | RawSvgElementKind::DcTitle | RawSvgElementKind::Title => {
                false
            }
            RawSvgElementKind::DcType => {
                key == "rdf:resource" && value == "http://purl.org/dc/dcmitype/StillImage"
            }
            RawSvgElementKind::Retained
            | RawSvgElementKind::ResourceDefs
            | RawSvgElementKind::ClipPath
            | RawSvgElementKind::LinearGradient
            | RawSvgElementKind::GradientStop
            | RawSvgElementKind::Filter
            | RawSvgElementKind::FeGaussianBlur
            | RawSvgElementKind::FeOffset
            | RawSvgElementKind::FeMerge
            | RawSvgElementKind::FeMergeNode
            | RawSvgElementKind::Hidden => {
                unreachable!("emitted or hidden attributes are validated separately")
            }
        };
        let known_metadata_uri = kind == RawSvgElementKind::DcType && allowed;
        if !allowed || (!known_metadata_uri && unsafe_attribute_value(value)) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG discarded attribute {key} is not allowed"
            )));
        }
    }
    Ok(())
}

fn normalize_root_dimensions(attrs: &mut [(String, String)]) -> WorkerResult<()> {
    let viewbox = attrs
        .iter()
        .find(|(key, _)| key == "viewBox")
        .map(|(_, value)| parse_viewbox_values(value))
        .transpose()?;
    for (key, value) in attrs
        .iter_mut()
        .filter(|(key, _)| matches!(key.as_str(), "width" | "height"))
    {
        if value.contains('%') {
            if value != "100%" {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG {key} percentage must be exactly 100%"
                )));
            }
            let values = viewbox.ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "provider SVG percentage dimensions require viewBox".to_owned(),
                )
            })?;
            *value = if key == "width" {
                values[2].to_string()
            } else {
                values[3].to_string()
            };
        } else if let Some(pixels) = physical_length_in_pixels(value)? {
            *value = pixels.to_string();
        }
    }
    Ok(())
}

fn physical_length_in_pixels(value: &str) -> WorkerResult<Option<f64>> {
    let Some((number, end)) = parse_svg_number(value, 0) else {
        return Ok(None);
    };
    let Some(unit) = value.get(end..) else {
        return Ok(None);
    };
    let factor = if unit.eq_ignore_ascii_case("in") {
        96.0
    } else if unit.eq_ignore_ascii_case("cm") {
        96.0 / 2.54
    } else if unit.eq_ignore_ascii_case("mm") {
        96.0 / 25.4
    } else if unit.eq_ignore_ascii_case("q") {
        96.0 / 101.6
    } else if unit.eq_ignore_ascii_case("pt") {
        96.0 / 72.0
    } else if unit.eq_ignore_ascii_case("pc") {
        16.0
    } else {
        return Ok(None);
    };
    let pixels = number * factor;
    if !pixels.is_finite() || pixels <= 0.0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG dimensions must be finite positive numbers".to_owned(),
        ));
    }
    validate_coordinate(pixels, "dimension")?;
    Ok(Some(pixels))
}

fn validate_inert_enable_background(value: &str) -> WorkerResult<()> {
    let Some(numbers) = value.strip_prefix("new ") else {
        return Err(WorkerError::InvalidPayload(
            "provider SVG enable-background is not recognized".to_owned(),
        ));
    };
    let values = parse_number_list(numbers, false, "enable-background", true)?;
    if values.len() != 4 || values[2] <= 0.0 || values[3] <= 0.0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG enable-background must be new plus four bounded viewport numbers"
                .to_owned(),
        ));
    }
    Ok(())
}

fn canonical_presentation_style(
    element: &str,
    style: &str,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<Vec<(String, String)>> {
    let lower = style.to_ascii_lowercase();
    let compact = lower
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    if style.trim().is_empty()
        || style
            .chars()
            .any(|character| matches!(character, '\\' | '{' | '}' | '@'))
        || compact.contains("/*")
        || compact.contains("*/")
        || compact.contains("expression(")
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG style is not allowed".to_owned(),
        ));
    }
    let mut attrs = Vec::new();
    let mut seen_properties = Vec::new();
    let declarations = style.split(';').collect::<Vec<_>>();
    for (index, declaration) in declarations.iter().enumerate() {
        if declaration.trim().is_empty() {
            if index + 1 == declarations.len() && style.ends_with(';') {
                continue;
            }
            return Err(WorkerError::InvalidPayload(
                "provider SVG style declaration is malformed".to_owned(),
            ));
        }
        let (key, value) = declaration.split_once(':').ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG style declaration is malformed".to_owned())
        })?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty()
            || value.contains(':')
            || seen_properties.iter().any(|existing| existing == &key)
        {
            return Err(WorkerError::InvalidPayload(
                "provider SVG style declaration is malformed".to_owned(),
            ));
        }
        seen_properties.push(key);
        if matches!(
            (key, value),
            ("display", "inline") | ("stroke-miterlimit", "4") | ("stroke-dasharray", "none")
        ) {
            continue;
        }
        if key == "enable-background" && element == "svg" {
            validate_inert_enable_background(value)?;
            continue;
        }
        if key == "fill" {
            if let Some(target) = local_url_target(value) {
                if resources.ids.get(target) != Some(&SvgResourceKind::LinearGradient) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG fill has an unresolved or mistyped resource".to_owned(),
                    ));
                }
                attrs.push((key.to_owned(), value.to_owned()));
                continue;
            }
        }
        if !allowed_presentation_property(key) || value.is_empty() || unsafe_attribute_value(value)
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG style property {key} is not allowed"
            )));
        }
        validate_attribute_resource_budget(element, key, value, budget)?;
        attrs.push((key.to_owned(), value.to_owned()));
    }
    Ok(attrs)
}

fn allowed_presentation_property(key: &str) -> bool {
    matches!(
        key,
        "fill"
            | "fill-rule"
            | "clip-rule"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "opacity"
    )
}

fn allowed_attribute(element: &str, attribute: &str) -> bool {
    let common = matches!(
        attribute,
        "fill"
            | "fill-rule"
            | "clip-rule"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "opacity"
            | "transform"
    );
    common
        || match element {
            "svg" => matches!(
                attribute,
                "xmlns" | "width" | "height" | "viewBox" | "preserveAspectRatio" | "overflow"
            ),
            "path" => attribute == "d",
            "rect" => matches!(attribute, "x" | "y" | "width" | "height" | "rx" | "ry"),
            "circle" => matches!(attribute, "cx" | "cy" | "r"),
            "ellipse" => matches!(attribute, "cx" | "cy" | "rx" | "ry"),
            "line" => matches!(attribute, "x1" | "x2" | "y1" | "y2"),
            "polyline" | "polygon" => attribute == "points",
            "g" => false,
            _ => false,
        }
}

fn validate_attribute_resource_budget(
    element: &str,
    key: &str,
    value: &str,
    budget: &mut SanitizerBudget,
) -> WorkerResult<()> {
    match key {
        "fill-rule" | "clip-rule" if !matches!(value, "nonzero" | "evenodd") => {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} must be nonzero or evenodd"
            )));
        }
        "d" => {
            budget.path_data_bytes =
                budget
                    .path_data_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("provider SVG path bytes overflow".to_owned())
                    })?;
            if budget.path_data_bytes > MAX_SVG_PATH_DATA_BYTES {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the path-data byte budget".to_owned(),
                ));
            }
            let (commands, numbers) = scan_path_data(value)?;
            budget.path_commands = budget.path_commands.checked_add(commands).ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG path command overflow".to_owned())
            })?;
            budget.path_numbers = budget.path_numbers.checked_add(numbers).ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG path number overflow".to_owned())
            })?;
            if budget.path_commands > MAX_SVG_PATH_COMMANDS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the path-command budget".to_owned(),
                ));
            }
            if budget.path_numbers > MAX_SVG_PATH_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the path-number budget".to_owned(),
                ));
            }
        }
        "points" => {
            let values = parse_number_list(value, false, "points", true)?;
            if values.len() < 2 || values.len() % 2 != 0 {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG points must contain coordinate pairs".to_owned(),
                ));
            }
            budget.point_numbers =
                budget
                    .point_numbers
                    .checked_add(values.len())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("provider SVG point count overflow".to_owned())
                    })?;
            if budget.point_numbers > MAX_SVG_POINT_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the point-number budget".to_owned(),
                ));
            }
        }
        "transform" => {
            let numbers = validate_transform_list(value)?;
            budget.transform_numbers =
                budget
                    .transform_numbers
                    .checked_add(numbers)
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG transform number overflow".to_owned(),
                        )
                    })?;
            if budget.transform_numbers > MAX_SVG_TRANSFORM_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the transform-number budget".to_owned(),
                ));
            }
        }
        "viewBox" => {
            parse_viewbox_values(value)?;
        }
        "preserveAspectRatio" => {
            validate_preserve_aspect_ratio(value)?;
        }
        "overflow" if value != "hidden" => {
            return Err(WorkerError::InvalidPayload(
                "provider SVG root overflow must be hidden".to_owned(),
            ));
        }
        "stroke-miterlimit" => {
            let values = parse_number_list(value, false, key, true)?;
            if values.len() != 1 || values[0] < 1.0 {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stroke-miterlimit must be one finite number at least one"
                        .to_owned(),
                ));
            }
        }
        "stroke-dasharray" if value != "none" => {
            let values = parse_number_list(value, true, key, true)?;
            if values.iter().any(|value| *value < 0.0) || values.iter().all(|value| *value == 0.0) {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stroke-dasharray must be none or non-negative finite numbers with a positive sum".to_owned(),
                ));
            }
            budget.dash_numbers =
                budget
                    .dash_numbers
                    .checked_add(values.len())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("provider SVG dash number overflow".to_owned())
                    })?;
            if budget.dash_numbers > MAX_SVG_DASH_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the dash-number budget".to_owned(),
                ));
            }
        }
        "stroke-dashoffset" => {
            let values = parse_number_list(value, true, key, true)?;
            if values.len() != 1 {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stroke-dashoffset must be one finite number".to_owned(),
                ));
            }
        }
        "stroke-linecap" if !matches!(value, "butt" | "round" | "square") => {
            return Err(WorkerError::InvalidPayload(
                "provider SVG stroke-linecap is not recognized".to_owned(),
            ));
        }
        "stroke-linejoin" if !matches!(value, "miter" | "round" | "bevel") => {
            return Err(WorkerError::InvalidPayload(
                "provider SVG stroke-linejoin is not recognized".to_owned(),
            ));
        }
        _ if coordinate_attribute(element, key) => {
            let values = parse_number_list(value, true, key, true)?;
            if values.len() != 1 {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} must contain exactly one finite number"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_preserve_aspect_ratio(value: &str) -> WorkerResult<()> {
    let mut tokens = value.split_ascii_whitespace();
    let first = tokens.next().ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG preserveAspectRatio is empty".to_owned())
    })?;
    let align = if first == "defer" {
        tokens.next().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG preserveAspectRatio defer requires an alignment".to_owned(),
            )
        })?
    } else {
        first
    };
    if align == "none" {
        if tokens.next().is_none() {
            return Ok(());
        }
    } else if matches!(
        align,
        "xMinYMin"
            | "xMidYMin"
            | "xMaxYMin"
            | "xMinYMid"
            | "xMidYMid"
            | "xMaxYMid"
            | "xMinYMax"
            | "xMidYMax"
            | "xMaxYMax"
    ) {
        match tokens.next() {
            None => return Ok(()),
            Some("meet" | "slice") if tokens.next().is_none() => return Ok(()),
            _ => {}
        }
    }
    Err(WorkerError::InvalidPayload(
        "provider SVG preserveAspectRatio must use the SVG alignment grammar".to_owned(),
    ))
}

fn coordinate_attribute(element: &str, key: &str) -> bool {
    key == "stroke-width"
        || match element {
            "svg" => matches!(key, "width" | "height"),
            "rect" => matches!(key, "x" | "y" | "width" | "height" | "rx" | "ry"),
            "circle" => matches!(key, "cx" | "cy" | "r"),
            "ellipse" => matches!(key, "cx" | "cy" | "rx" | "ry"),
            "line" => matches!(key, "x1" | "x2" | "y1" | "y2"),
            _ => false,
        }
}

fn scan_path_data(value: &str) -> WorkerResult<(usize, usize)> {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut commands = 0usize;
    let mut numbers = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() || bytes[index] == b',' {
            index += 1;
            continue;
        }
        if matches!(
            bytes[index],
            b'M' | b'm'
                | b'Z'
                | b'z'
                | b'L'
                | b'l'
                | b'H'
                | b'h'
                | b'V'
                | b'v'
                | b'C'
                | b'c'
                | b'S'
                | b's'
                | b'Q'
                | b'q'
                | b'T'
                | b't'
                | b'A'
                | b'a'
        ) {
            commands += 1;
            index += 1;
            continue;
        }
        let (number, end) = parse_svg_number(value, index).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG path data is malformed".to_owned())
        })?;
        validate_coordinate(number, "path")?;
        numbers += 1;
        index = end;
    }
    if commands == 0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG path data contains no commands".to_owned(),
        ));
    }
    Ok((commands, numbers))
}

fn parse_number_list(
    value: &str,
    allow_px: bool,
    label: &str,
    enforce_coordinate_budget: bool,
) -> WorkerResult<Vec<f64>> {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut values = Vec::new();
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() || bytes[index] == b',' {
            index += 1;
            continue;
        }
        let (number, mut end) = parse_svg_number(value, index).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "provider SVG {label} must contain only finite numbers"
            ))
        })?;
        if allow_px && value.as_bytes().get(end..end + 2) == Some(b"px") {
            end += 2;
        }
        if end < bytes.len()
            && !bytes[end].is_ascii_whitespace()
            && bytes[end] != b','
            && !(bytes[end] == b'+' || bytes[end] == b'-')
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG {label} uses an unsupported unit or token"
            )));
        }
        if enforce_coordinate_budget {
            validate_coordinate(number, label)?;
        }
        values.push(number);
        index = end;
    }
    if values.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} contains no numbers"
        )));
    }
    Ok(values)
}

fn parse_svg_number(value: &str, start: usize) -> Option<(f64, usize)> {
    let bytes = value.as_bytes();
    let mut index = start;
    if matches!(bytes.get(index), Some(b'+' | b'-')) {
        index += 1;
    }
    let integer_start = index;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        index += 1;
    }
    let mut digits = index - integer_start;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        digits += index - fraction_start;
    }
    if digits == 0 {
        return None;
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let exponent_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == exponent_start {
            return None;
        }
    }
    let number = value.get(start..index)?.parse::<f64>().ok()?;
    number.is_finite().then_some((number, index))
}

fn validate_coordinate(number: f64, label: &str) -> WorkerResult<()> {
    if !number.is_finite() || number.abs() > MAX_SVG_COORDINATE_MAGNITUDE {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} exceeds the coordinate-magnitude budget"
        )));
    }
    Ok(())
}

fn validate_transform_list(value: &str) -> WorkerResult<usize> {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut total_numbers = 0usize;
    while index < bytes.len() {
        while bytes
            .get(index)
            .is_some_and(|byte| byte.is_ascii_whitespace() || *byte == b',')
        {
            index += 1;
        }
        if index == bytes.len() {
            break;
        }
        let name_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_alphabetic) {
            index += 1;
        }
        let name = value.get(name_start..index).unwrap_or_default();
        if !matches!(
            name,
            "matrix" | "translate" | "scale" | "rotate" | "skewX" | "skewY"
        ) {
            return Err(WorkerError::InvalidPayload(
                "provider SVG transform function is not allowed".to_owned(),
            ));
        }
        while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if bytes.get(index) != Some(&b'(') {
            return Err(WorkerError::InvalidPayload(
                "provider SVG transform is malformed".to_owned(),
            ));
        }
        index += 1;
        let args_start = index;
        while bytes.get(index).is_some_and(|byte| *byte != b')') {
            index += 1;
        }
        if bytes.get(index) != Some(&b')') {
            return Err(WorkerError::InvalidPayload(
                "provider SVG transform is incomplete".to_owned(),
            ));
        }
        let args = value.get(args_start..index).unwrap_or_default();
        let count = parse_number_list(args, false, "transform", true)?.len();
        let valid_arity = match name {
            "matrix" => count == 6,
            "translate" | "scale" => matches!(count, 1 | 2),
            "rotate" => matches!(count, 1 | 3),
            "skewX" | "skewY" => count == 1,
            _ => false,
        };
        if !valid_arity {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG transform {name} has an invalid argument count"
            )));
        }
        total_numbers += count;
        index += 1;
    }
    if total_numbers == 0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG transform contains no numbers".to_owned(),
        ));
    }
    Ok(total_numbers)
}

fn unsafe_attribute_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    let compact = lower
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    compact.contains("url(")
        || compact.contains("expression(")
        || lower.contains('\\')
        || compact.contains("/*")
        || compact.contains("*/")
        || lower.contains(';')
        || lower.contains('{')
        || lower.contains('}')
        || lower.contains('@')
        || lower.contains("javascript:")
        || lower.contains("data:")
        || lower.contains("file:")
        || lower.contains("http:")
        || lower.contains("https:")
        || lower.contains("ftp:")
        || lower.contains("ws:")
        || lower.contains("wss:")
        || compact.contains("@import")
        || lower.contains("//")
}

fn svg_dimensions(attrs: &[(String, String)]) -> WorkerResult<(u32, u32)> {
    let named = |name: &str| {
        attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    };
    let viewbox = named("viewBox").map(parse_viewbox).transpose()?;
    let width = named("width")
        .map(parse_dimension)
        .transpose()?
        .or_else(|| viewbox.map(|pair| pair.0))
        .unwrap_or(512);
    let height = named("height")
        .map(parse_dimension)
        .transpose()?
        .or_else(|| viewbox.map(|pair| pair.1))
        .unwrap_or(512);
    if width == 0 || height == 0 || width > MAX_PREVIEW_DIMENSION || height > MAX_PREVIEW_DIMENSION
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG dimensions must be 1..={MAX_PREVIEW_DIMENSION}"
        )));
    }
    Ok((width, height))
}

fn parse_dimension(value: &str) -> WorkerResult<u32> {
    let values = parse_number_list(value, true, "dimension", true)?;
    let parsed = values
        .first()
        .copied()
        .filter(|value| *value > 0.0)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG dimensions must be finite positive numbers".to_owned(),
            )
        })?;
    if values.len() != 1 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG dimensions must contain exactly one number".to_owned(),
        ));
    }
    Ok(parsed.ceil() as u32)
}

fn parse_viewbox(value: &str) -> WorkerResult<(u32, u32)> {
    let numbers = parse_viewbox_values(value)?;
    Ok((numbers[2].ceil() as u32, numbers[3].ceil() as u32))
}

fn parse_viewbox_values(value: &str) -> WorkerResult<[f64; 4]> {
    // Parse finite SVG numbers first, then apply the viewBox-specific origin/span bounds below so
    // an overlarge origin cannot be hidden behind the generic shape-coordinate refusal.
    let numbers = parse_number_list(value, false, "viewBox", false)?;
    let numbers: [f64; 4] = numbers.try_into().map_err(|_| {
        WorkerError::InvalidPayload(
            "provider SVG viewBox must contain four finite numbers".to_owned(),
        )
    })?;
    if numbers[0].abs() > MAX_SVG_VIEWBOX_ORIGIN_MAGNITUDE
        || numbers[1].abs() > MAX_SVG_VIEWBOX_ORIGIN_MAGNITUDE
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG viewBox origin exceeds the origin-magnitude budget".to_owned(),
        ));
    }
    if numbers[2] <= 0.0
        || numbers[3] <= 0.0
        || numbers[2] > f64::from(MAX_PREVIEW_DIMENSION)
        || numbers[3] > f64::from(MAX_PREVIEW_DIMENSION)
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG viewBox dimensions must be 1..={MAX_PREVIEW_DIMENSION}"
        )));
    }
    Ok(numbers)
}

fn write_start(output: &mut String, name: &str, attrs: &[(String, String)], empty: bool) {
    output.push('<');
    output.push_str(name);
    for (key, value) in attrs {
        output.push(' ');
        output.push_str(key);
        output.push_str("=\"");
        for character in value.chars() {
            match character {
                '&' => output.push_str("&amp;"),
                '<' => output.push_str("&lt;"),
                '"' => output.push_str("&quot;"),
                _ => output.push(character),
            }
        }
        output.push('"');
    }
    output.push_str(if empty { "/>" } else { ">" });
}

async fn render_preview(svg: &str, width: u32, height: u32, path: &Path) -> WorkerResult<()> {
    render_preview_with_size(svg, width, height, path, None).await
}

async fn render_preview_with_size(
    svg: &str,
    width: u32,
    height: u32,
    path: &Path,
    preview_size: Option<u32>,
) -> WorkerResult<()> {
    let svg = svg.to_owned();
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).map_err(|error| {
            WorkerError::InvalidPayload(format!("provider SVG cannot be rendered: {error}"))
        })?;
        let (width, height, transform) = match preview_size {
            Some(size) => {
                let intrinsic = tree.size();
                let scale = (size as f32 / intrinsic.width()).min(size as f32 / intrinsic.height());
                let x = (size as f32 - intrinsic.width() * scale) / 2.0;
                let y = (size as f32 - intrinsic.height() * scale) / 2.0;
                (
                    size,
                    size,
                    resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y),
                )
            }
            None => (width, height, resvg::tiny_skia::Transform::identity()),
        };
        validate_filter_render_budget(&tree, width, height, transform)?;
        let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG preview dimensions are invalid".to_owned())
        })?;
        // resvg draws only the already-sanitized inert geometry into a fixed-size PNG.
        resvg::render(&tree, transform, &mut pixmap.as_mut());
        pixmap
            .save_png(path)
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))
    })
    .await
    .map_err(|error| task_join_error("SVG preview render", error))??;
    Ok(())
}

fn filter_input_surface_count(kind: &usvg::filter::Kind) -> WorkerResult<u64> {
    match kind {
        // Pinned resvg's box blur allocates one RGBA back buffer. Its IIR path allocates one f64
        // value per pixel, equivalent to two RGBA surfaces, so charge the larger scratch cost in
        // addition to the primitive result charged by the caller.
        usvg::filter::Kind::GaussianBlur(_) => Ok(2),
        usvg::filter::Kind::Offset(_) => Ok(1),
        usvg::filter::Kind::Merge(merge) => u64::try_from(merge.inputs().len()).map_err(|_| {
            WorkerError::InvalidPayload("provider SVG filter input count overflow".to_owned())
        }),
        _ => Err(WorkerError::InvalidPayload(
            "provider SVG parsed to an unsupported filter primitive".to_owned(),
        )),
    }
}

fn clipped_filter_layer_pixels(
    group: &usvg::Group,
    target_width: u32,
    target_height: u32,
    root_transform: resvg::tiny_skia::Transform,
) -> WorkerResult<u64> {
    let bbox = group
        .abs_layer_bounding_box()
        .transform(root_transform)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG filter layer has an invalid transform".to_owned(),
            )
        })?
        .to_int_rect();
    // Keep this identical to resvg::render's max_bbox and render_group intersection. The parsed
    // abs_layer_bounding_box already contains objectBoundingBox filter resolution and every SVG
    // transform, while root_transform contains the requested preview fit.
    let left = bbox.left().max(-(target_width as i32) * 2);
    let top = bbox.top().max(-(target_height as i32) * 2);
    let right = bbox.right().min((target_width as i32) * 3);
    let bottom = bbox.bottom().min((target_height as i32) * 3);
    if right <= left || bottom <= top {
        return Ok(0);
    }
    let width = u64::try_from(right - left).map_err(|_| {
        WorkerError::InvalidPayload("provider SVG filter layer width overflow".to_owned())
    })?;
    let height = u64::try_from(bottom - top).map_err(|_| {
        WorkerError::InvalidPayload("provider SVG filter layer height overflow".to_owned())
    })?;
    width.checked_mul(height).ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG filter layer pixels overflow".to_owned())
    })
}

fn validate_filter_render_budget(
    tree: &usvg::Tree,
    target_width: u32,
    target_height: u32,
    root_transform: resvg::tiny_skia::Transform,
) -> WorkerResult<()> {
    fn visit(
        group: &usvg::Group,
        target_width: u32,
        target_height: u32,
        root_transform: resvg::tiny_skia::Transform,
        total: &mut u64,
    ) -> WorkerResult<()> {
        if !group.filters().is_empty() {
            let layer_pixels =
                clipped_filter_layer_pixels(group, target_width, target_height, root_transform)?;
            // resvg allocates the isolated group layer once. Its filter evaluator retains every
            // primitive result and can clone every input surface while producing the next result.
            let mut surfaces = 1u64;
            for filter in group.filters() {
                for primitive in filter.primitives() {
                    surfaces = surfaces.checked_add(1).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG filter surface count overflow".to_owned(),
                        )
                    })?;
                    let inputs = filter_input_surface_count(primitive.kind())?;
                    surfaces = surfaces.checked_add(inputs).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG filter surface count overflow".to_owned(),
                        )
                    })?;
                }
            }
            *total = total
                .checked_add(layer_pixels.checked_mul(surfaces).ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "provider SVG filter offscreen pixels overflow".to_owned(),
                    )
                })?)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "provider SVG filter offscreen pixels overflow".to_owned(),
                    )
                })?;
            if *total > MAX_SVG_FILTER_OFFSCREEN_PIXELS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG filters exceed the total offscreen-pixel budget".to_owned(),
                ));
            }
        }
        for child in group.children() {
            if let usvg::Node::Group(child) = child {
                visit(child, target_width, target_height, root_transform, total)?;
            }
        }
        Ok(())
    }

    let mut total = 0u64;
    visit(
        tree.root(),
        target_width,
        target_height,
        root_transform,
        &mut total,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_request(mode: VectorMode, prompt: &str) -> VectorProviderRequest {
        VectorProviderRequest {
            mode,
            model: "starvector_1b".to_owned(),
            source_path: None,
            prompt: prompt.to_owned(),
            sampling: VectorSampling {
                temperature: 0.2,
                top_p: 0.9,
                top_k: 0,
                repetition_penalty: 1.0,
                repetition_context: 0,
                seed: Some(7),
            },
            detail_budget: VectorDetailBudget {
                max_new_tokens: 128,
                max_svg_bytes: 4_096,
                max_wall_time_ms: 1_000,
            },
        }
    }

    fn invalid_detail(result: WorkerResult<CanonicalSvg>) -> String {
        match result.expect_err("fixture must be rejected") {
            WorkerError::InvalidPayload(detail) => detail,
            other => panic!("expected invalid payload, got {other}"),
        }
    }

    fn terminal_fixture(
        finish_reason: StarVectorFinishReason,
        source: &str,
        svg: Option<&str>,
    ) -> (StarVectorOutput, Vec<StarVectorStreamEvent>) {
        let generated_tokens = u32::from(!source.is_empty());
        let generated_bytes = source.len();
        let mut events = Vec::new();
        if !source.is_empty() {
            events.push(StarVectorStreamEvent::Source {
                text: source.to_owned(),
                index: 0,
            });
        }
        events.push(StarVectorStreamEvent::Done {
            finish_reason,
            generated_tokens,
            generated_bytes,
        });
        (
            StarVectorOutput {
                svg: svg.map(str::to_owned),
                generated_tokens,
                generated_bytes,
                finish_reason,
            },
            events,
        )
    }

    struct CancelingProvider;

    impl MultimodalVectorProviderAdapter for CancelingProvider {
        fn provider_id(&self) -> &str {
            "starvector"
        }

        fn supports_mode(&self, mode: VectorMode) -> bool {
            mode == VectorMode::TextToSvg
        }

        fn generate_svg(
            &self,
            _request: &VectorProviderRequest,
            cancel: &gen_core::CancelFlag,
            _progress: tokio::sync::watch::Sender<u32>,
            on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
        ) -> WorkerResult<()> {
            on_source("<svg xmlns=\"http://www.w3.org/2000/svg\">", 0)?;
            cancel.cancel();
            on_source("<rect width=\"1\" height=\"1\"/></svg>", 1)?;
            Ok(())
        }
    }

    struct NativeSourceSequence([u32; 3]);

    impl MultimodalVectorProviderAdapter for NativeSourceSequence {
        fn provider_id(&self) -> &str {
            "starvector"
        }

        fn supports_mode(&self, mode: VectorMode) -> bool {
            mode == VectorMode::TextToSvg
        }

        fn generate_svg(
            &self,
            request: &VectorProviderRequest,
            _cancel: &gen_core::CancelFlag,
            progress: tokio::sync::watch::Sender<u32>,
            on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
        ) -> WorkerResult<()> {
            // One hidden decoder token separates the static prefix and first visible delta.
            // Validate native output/events, then forward the original indices just as the
            // production adapter does. Source fragments and token counts stay independent.
            let svg = "<svg></svg>";
            let output = StarVectorOutput {
                svg: Some(svg.to_owned()),
                generated_tokens: 3,
                generated_bytes: svg.len(),
                finish_reason: StarVectorFinishReason::CompleteRoot,
            };
            let mut events = Vec::new();
            for (index, fragment) in self.0.into_iter().zip(["<svg", ">", "</svg>"]) {
                events.push(StarVectorStreamEvent::Source {
                    text: fragment.to_owned(),
                    index,
                });
            }
            for generated_tokens in 1..=3 {
                record_vector_token_progress(
                    &progress,
                    generated_tokens,
                    request.detail_budget.max_new_tokens,
                );
            }
            events.push(StarVectorStreamEvent::Done {
                finish_reason: output.finish_reason,
                generated_tokens: output.generated_tokens,
                generated_bytes: output.generated_bytes,
            });
            for (fragment, index) in validate_native_starvector_generation(output, events)? {
                on_source(&fragment, index)?;
            }
            Ok(())
        }
    }

    #[test]
    fn native_hidden_token_source_gaps_reach_product_collection() {
        let request = vector_request(VectorMode::TextToSvg, "a compact mark");
        let (progress, observed) = tokio::sync::watch::channel(0);
        let collected = collect_svg_source(
            &NativeSourceSequence([0, 2, 3]),
            &request,
            &gen_core::CancelFlag::new(),
            progress,
        )
        .expect("native hidden-token indices remain valid at the product boundary");
        assert_eq!(collected.source.as_deref(), Some("<svg></svg>"));
        assert_eq!(
            *observed.borrow(),
            3,
            "progress counts actual decoder tokens"
        );
    }

    #[test]
    fn native_source_indices_still_reject_duplicates_regressions_and_missing_prefix() {
        let request = vector_request(VectorMode::TextToSvg, "a compact mark");
        for indices in [[0, 0, 3], [0, 3, 2], [1, 2, 3]] {
            let result = collect_svg_source(
                &NativeSourceSequence(indices),
                &request,
                &gen_core::CancelFlag::new(),
                tokio::sync::watch::channel(0).0,
            );
            assert!(
                matches!(result, Err(WorkerError::Engine(message)) if message.contains("source index")),
                "invalid sequence {indices:?} must remain rejected",
            );
        }
    }

    #[tokio::test]
    async fn vector_progress_is_live_bounded_monotonic_and_private() {
        let (tx, rx) = tokio::sync::watch::channel(0);
        let (release, blocked) = tokio::sync::oneshot::channel();
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completed = finished.clone();
        let generation = async move {
            blocked.await.expect("test releases provider");
            completed.store(true, std::sync::atomic::Ordering::Release);
            Ok(())
        };
        // Exercise the same bounded sink used by the native provider callback. Older counts
        // cannot regress progress, and a bad provider cannot exceed the declared maximum.
        record_vector_token_progress(&tx, 1, 10);
        let mut release = Some(release);
        let mut observations = Vec::new();
        tokio::time::timeout(
            Duration::from_secs(3),
            with_vector_progress(
                rx,
                10,
                gen_core::CancelFlag::new(),
                generation,
                |count, maximum| {
                    assert!(
                        !finished.load(std::sync::atomic::Ordering::Acquire),
                        "progress must precede decode completion"
                    );
                    observations.push(count);
                    let payload = vector_token_progress(count, maximum);
                    assert_eq!(payload.extra["generatedTokens"], count);
                    assert_eq!(payload.extra["maxNewTokens"], 10);
                    let encoded = serde_json::to_string(&payload).expect("serializes");
                    assert!(!encoded.contains("<svg"));
                    if count == 1 {
                        record_vector_token_progress(&tx, 0, 10);
                        assert_eq!(*tx.borrow(), 1);
                        record_vector_token_progress(&tx, 1000, 10);
                    } else {
                        release
                            .take()
                            .expect("one completion")
                            .send(())
                            .expect("provider held");
                    }
                    std::future::ready(Ok(()))
                },
            ),
        )
        .await
        .expect("live progress completes without waiting for provider")
        .expect("progress succeeds");
        assert_eq!(observations, [1, 10]);
        assert!(finished.load(std::sync::atomic::Ordering::Acquire));
    }

    #[tokio::test]
    async fn vector_progress_post_failure_cancels_and_joins_generation() {
        let (tx, rx) = tokio::sync::watch::channel(1);
        let cancel = gen_core::CancelFlag::new();
        let provider_cancel = cancel.clone();
        let joined = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let provider_joined = joined.clone();
        let generation = async move {
            while !provider_cancel.is_cancelled() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            provider_joined.store(true, std::sync::atomic::Ordering::Release);
            Ok(())
        };
        let result = with_vector_progress(rx, 10, cancel.clone(), generation, |_, _| {
            std::future::ready(Err(WorkerError::Engine("progress unavailable".to_owned())))
        })
        .await;
        drop(tx);
        assert!(result.is_err());
        assert!(cancel.is_cancelled());
        assert!(joined.load(std::sync::atomic::Ordering::Acquire));
    }

    #[tokio::test]
    async fn vector_progress_never_posts_after_cancellation() {
        let (_tx, rx) = tokio::sync::watch::channel(3);
        let cancel = gen_core::CancelFlag::new();
        cancel.cancel();
        let result: WorkerResult<()> = with_vector_progress(
            rx,
            10,
            cancel,
            async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()))
            },
            |_, _| {
                panic!("cancelled generation must not post Running progress");
                #[allow(unreachable_code)]
                std::future::ready(Ok(()))
            },
        )
        .await;
        assert!(matches!(result, Err(WorkerError::Canceled(_))));
    }

    #[test]
    fn native_request_is_image_only_and_rejects_text_guidance() {
        let image = ImageRef::new(1, 1, vec![1, 2, 3]).expect("valid RGB pixel");
        let request = native_starvector_request(
            &vector_request(VectorMode::ImageToSvg, ""),
            image.clone(),
            gen_core::core_llm::CancelFlag::new(),
        )
        .expect("image-only request");
        assert_eq!(request.text_request.messages.len(), 1);
        assert_eq!(request.text_request.messages[0].role, Role::User);
        assert!(matches!(
            request.text_request.messages[0].content.as_slice(),
            [Content::Image(actual)] if actual == &image
        ));
        assert!(!request.has_text());

        let error = native_starvector_request(
            &vector_request(VectorMode::ImageToSvg, "undisclosed guidance"),
            image,
            gen_core::core_llm::CancelFlag::new(),
        )
        .expect_err("StarVector image-to-SVG must remain image-only");
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
    }

    #[test]
    fn exact_starvector_manifest_identity_is_fail_closed() {
        let identity = starvector_model_identity("starvector_1b").expect("known identity");
        let exact = serde_json::json!({
            "id": identity.model_id,
            "type": "vector",
            "adapter": "starvector",
            "downloads": [{
                "repo": identity.repository,
                "revision": identity.revision
            }]
        });
        manifest_binds_starvector_identity(&exact, identity).expect("exact immutable identity");

        for crossed in [
            serde_json::json!({
                "id": "starvector_8b",
                "type": "vector",
                "adapter": "starvector",
                "downloads": [{"repo": identity.repository, "revision": identity.revision}]
            }),
            serde_json::json!({
                "id": identity.model_id,
                "type": "vector",
                "adapter": "starvector",
                "downloads": [{"repo": identity.repository, "revision": "main"}]
            }),
            serde_json::json!({
                "id": identity.model_id,
                "type": "vector",
                "adapter": "starvector",
                "downloads": [{
                    "repo": identity.repository,
                    "revision": identity.revision,
                    "coRequisite": true
                }]
            }),
        ] {
            assert!(manifest_binds_starvector_identity(&crossed, identity).is_err());
        }
        assert!(starvector_model_identity("unregistered-starvector").is_err());
    }

    #[test]
    fn native_terminal_events_publish_only_complete_root_or_eos() {
        let svg = "<svg/>";
        for reason in [
            StarVectorFinishReason::CompleteRoot,
            StarVectorFinishReason::Eos,
        ] {
            let (output, events) = terminal_fixture(reason, svg, Some(svg));
            assert_eq!(
                validate_native_starvector_generation(output, events)
                    .expect("complete document is publishable"),
                vec![(svg.to_owned(), 0)]
            );
        }

        for reason in [
            StarVectorFinishReason::TokenLimit,
            StarVectorFinishReason::ByteLimit,
            StarVectorFinishReason::WallTimeLimit,
        ] {
            let (output, events) = terminal_fixture(reason, "<svg>", None);
            assert!(
                validate_native_starvector_generation(output, events)
                    .expect("typed bounded outcome")
                    .is_empty(),
                "bounded partial output must never reach publication"
            );
        }

        let (output, events) = terminal_fixture(StarVectorFinishReason::Cancelled, "<svg>", None);
        assert!(validate_native_starvector_generation(output, events)
            .expect("typed cancellation outcome")
            .is_empty());
        assert_eq!(
            terminal_finish_reason(StarVectorFinishReason::TokenLimit),
            "token_limit"
        );
    }

    #[test]
    fn terminal_evidence_never_advertises_files_for_a_bounded_outcome() {
        let terminal = TerminalProviderOutcome {
            finish_reason: "token_limit",
            generated_tokens: 7,
            generated_bytes: 23,
            latency_seconds: 0.01,
            provider_id: "mlx-starvector-1b".to_owned(),
            model_id: "starvector_1b",
            model_repository: "starvector/starvector-1b-im2svg",
            model_revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
            backend: "mlx",
        };
        let result = terminal_generation_limit_result(
            &terminal,
            (Path::new("provider-terminal.json"), b"transcript"),
        )
        .expect("typed generation rejection");
        assert_eq!(result["accepted"], false);
        assert_eq!(result["outcome"], "rejected");
        assert_eq!(result["rejectionStage"], "generation_limit");
        assert_eq!(result["rejectionCode"], "token_limit");
        assert_eq!(result["finishReason"], "token_limit");
        assert_ne!(result["providerTranscriptSha256"], Value::Null);
        assert!(result["canonicalSvgPath"].is_null());
        assert!(result["previewPngPath"].is_null());
        assert_eq!(result["providerTranscriptPath"], "provider-terminal.json");
        assert_eq!(result["resultContainsInlineSvg"], false);

        let cancelled = TerminalProviderOutcome {
            finish_reason: "cancelled",
            ..terminal
        };
        assert!(terminal_generation_limit_result(
            &cancelled,
            (Path::new("provider-terminal.json"), b"transcript"),
        )
        .is_err());
    }

    #[test]
    fn native_terminal_events_require_one_consistent_final_done() {
        let svg = "<svg/>";
        let (output, mut events) =
            terminal_fixture(StarVectorFinishReason::CompleteRoot, svg, Some(svg));
        events.pop();
        assert!(validate_native_starvector_generation(output.clone(), events).is_err());

        let (output, mut events) =
            terminal_fixture(StarVectorFinishReason::CompleteRoot, svg, Some(svg));
        events.push(StarVectorStreamEvent::Done {
            finish_reason: output.finish_reason,
            generated_tokens: output.generated_tokens,
            generated_bytes: output.generated_bytes,
        });
        assert!(validate_native_starvector_generation(output.clone(), events).is_err());

        let (output, mut events) =
            terminal_fixture(StarVectorFinishReason::CompleteRoot, svg, Some(svg));
        if let Some(StarVectorStreamEvent::Done {
            generated_bytes, ..
        }) = events.last_mut()
        {
            *generated_bytes += 1;
        }
        assert!(validate_native_starvector_generation(output, events).is_err());
    }

    #[test]
    fn cancellation_during_streamed_svg_never_reaches_publication() {
        let request = vector_request(VectorMode::TextToSvg, "a compact mark");
        let cancel = gen_core::CancelFlag::new();
        let output = collect_svg_source(
            &CancelingProvider,
            &request,
            &cancel,
            tokio::sync::watch::channel(0).0,
        );
        assert!(matches!(output, Err(WorkerError::Canceled(_))));

        // This is the exact publication boundary used by `run_vector_job_with_provider`: staging
        // starts only after `collect_svg_source` returns Ok. A canceled stream has no source to
        // sanitize, render, or rename, so neither member of the asset pair can exist.
        let temp = tempfile::tempdir().expect("temp dir");
        let published = temp.path().join("asset");
        if let Ok(CollectedSvgSource {
            source: Some(source),
            ..
        }) = output
        {
            std::fs::create_dir_all(&published).expect("publication dir");
            std::fs::write(published.join("vector.svg"), source).expect("source writes");
        }
        assert!(!published.exists());
    }

    #[test]
    fn canonicalizes_inert_svg_and_accepts_safe_px_dimensions() {
        let valid = sanitize_svg("<svg height=\"8px\" width=\"12px\" xmlns=\"http://www.w3.org/2000/svg\"><rect height=\"4\" width=\"3\"/></svg>")
            .expect("valid inert fixture");
        assert_eq!((valid.width, valid.height), (12, 8));
        assert!(
            valid.svg.contains("height=\"8px\" width=\"12px\""),
            "attributes are canonicalized"
        );
    }

    #[test]
    fn inert_root_namespace_declarations_are_dropped_without_broadening_svg_policy() {
        let declarations = concat!(
            r#"xmlns:dc="http://purl.org/dc/elements/1.1/" "#,
            r#"xmlns:cc="http://creativecommons.org/ns#" "#,
            r#"xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" "#,
            r#"xmlns:svg="http://www.w3.org/2000/svg" "#,
            r#"xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" "#,
            r#"xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" "#,
            r#"xmlns:xlink="http://www.w3.org/1999/xlink" "#,
            r#"xmlns:ns1="http://sozi.baierouge.fr""#,
        );
        let input =
            format!(r#"<svg {declarations} viewBox="0 0 2 2"><path d="M0 0H2V2H0Z"/></svg>"#);
        let canonical = sanitize_svg(&input).expect("inert namespace declarations");
        assert!(canonical
            .svg
            .contains(&format!("xmlns=\"{SVG_NAMESPACE}\"")));
        assert!(
            !canonical.svg.contains("xmlns:"),
            "declarations are not retained"
        );
        let legacy_cc = sanitize_svg(
            r#"<svg xmlns:cc="http://web.resource.org/cc/"><path d="M0 0H1V1Z"/></svg>"#,
        )
        .expect("legacy Creative Commons declaration is inert");
        assert!(!legacy_cc.svg.contains("xmlns:cc"));

        for input in [
            r#"<svg xmlns="https://example.invalid/svg"><path d="M0 0H1V1Z"/></svg>"#,
            r#"<svg xmlns:evil="https://example.invalid"><evil:path/></svg>"#,
            r#"<svg xmlns:dc="https://example.invalid"><path dc:href="https://example.invalid/a" d="M0 0H1V1Z"/></svg>"#,
            r#"<svg><path xmlns:evil="https://example.invalid" d="M0 0H1V1Z"/></svg>"#,
            r#"<svg xmlns:xml="http://www.w3.org/XML/1998/namespace"><path d="M0 0H1V1Z"/></svg>"#,
            r#"<svg xmlns:bad:prefix="https://example.invalid"><path d="M0 0H1V1Z"/></svg>"#,
            r#"<svg xmlns:xlink="https://example.invalid"><path d="M0 0H1V1Z"/></svg>"#,
            r##"<svg xmlns:xlink="http://www.w3.org/1999/xlink"><path xlink:href="#x" d="M0 0H1V1Z"/></svg>"##,
        ] {
            assert!(sanitize_svg(input).is_err(), "accepted {input}");
        }
    }

    #[tokio::test]
    async fn archived_upstream_case_strips_editor_metadata_and_keeps_visible_geometry() {
        let raw = include_bytes!("../tests/fixtures/starvector/upstream-34835393738-case-02.svg");
        let canonical = terminal_sanitize_svg_bytes(raw).expect("actual archived upstream case");
        assert_eq!((canonical.width, canonical.height), (90, 90));
        for retained in [
            "fill=\"#ffffff\"",
            "fill-opacity=\"1\"",
            "stroke=\"#000000\"",
            "stroke-opacity=\"1\"",
            "stroke-width=\"1.25\"",
            "stroke-linecap=\"butt\"",
            "stroke-linejoin=\"miter\"",
            "transform=\"translate(0,-962.35975)\"",
        ] {
            assert!(
                canonical.canonical_svg.contains(retained),
                "missing {retained}"
            );
        }
        for removed in [
            "xmlns:",
            "<metadata",
            "<defs",
            "sodipodi:",
            "inkscape:",
            "style=",
            "stroke-miterlimit=",
            "stroke-dasharray=",
        ] {
            assert!(
                !canonical.canonical_svg.contains(removed),
                "retained {removed}"
            );
        }
        let temp = tempfile::tempdir().expect("temp dir");
        let (_, preview) = terminal_write_sanitized_pair_with_preview_size(
            &canonical,
            &temp.path().join("actual-upstream-case-02"),
            Some(512),
        )
        .await
        .expect("actual fixture renders through the canonical CLI seam");
        let pixels = image::open(preview).expect("preview PNG").to_rgba8();
        assert_eq!(pixels.dimensions(), (512, 512));
        assert!(
            pixels.pixels().any(|pixel| pixel[3] > 0),
            "actual visible geometry renders"
        );
    }

    #[test]
    fn editor_and_rdf_discarding_is_exact_and_contextual() {
        let valid = concat!(
            "<svg>",
            "<defs id=\"defs4\"/>",
            "<sodipodi:namedview id=\"base\" objecttolerance=\"10\" gridtolerance=\"10\" guidetolerance=\"10\"/>",
            "<metadata id=\"metadata7\"><rdf:rdf><cc:work rdf:about=\"\">",
            "<dc:format>image/svg+xml</dc:format>",
            "<dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>",
            "<dc:title>archived title</dc:title>",
            "</cc:work></rdf:rdf></metadata>",
            "<g><title>visible path description</title><rect id=\"shape\" class=\"geometry\" width=\"1\" height=\"1\"/></g>",
            "</svg>",
        );
        let canonical = sanitize_svg(valid).expect("exact inert editor and RDF metadata");
        assert!(canonical.svg.contains("<rect"));
        for discarded in ["<defs", "namedview", "metadata", "rdf:", "cc:", "dc:"] {
            assert!(!canonical.svg.contains(discarded));
        }

        let potrace = sanitize_svg(concat!(
            "<svg><metadata>",
            "Created by potrace 1.11, written by Peter Selinger 2001-2013",
            "</metadata><g><path d=\"M0 0H1V1Z\"/></g></svg>",
        ))
        .expect("exact inert potrace provenance metadata");
        assert!(potrace.svg.contains("<path"));
        assert!(!potrace.svg.contains("Created by potrace"));

        for malicious in [
            "<svg>Created by potrace 1.11, written by Peter Selinger 2001-2013<path d=\"M0 0H1V1Z\"/></svg>",
            "<svg><g>Created by potrace 1.11, written by Peter Selinger 2001-2013</g></svg>",
            "<svg><metadata><script/></metadata></svg>",
            "<svg><metadata><foreignObject/></metadata></svg>",
            "<svg><metadata><rdf:rdf><cc:work rdf:about=\"\"><dc:creator/></cc:work></rdf:rdf></metadata></svg>",
            "<svg><g><metadata><rdf:rdf></rdf:rdf></metadata></g></svg>",
            "<svg><g><sodipodi:namedview></sodipodi:namedview></g></svg>",
            "<svg><inkscape:grid/></svg>",
            "<svg><metadata onclick=\"alert(1)\"><rdf:rdf></rdf:rdf></metadata></svg>",
            "<svg><sodipodi:namedview pagecolor=\"url(https://example.invalid/a)\"></sodipodi:namedview></svg>",
            "<svg><metadata><rdf:rdf></metadata></rdf:rdf></svg>",
            "<svg><path dc:format=\"image/svg+xml\" d=\"M0 0H1V1Z\"/></svg>",
            "<svg><g inkscape:unknown=\"1\"/></svg>",
            "<svg><rect onclick=\"alert(1)\"/></svg>",
            "<svg><title onclick=\"alert(1)\">metadata</title></svg>",
        ] {
            assert!(sanitize_svg(malicious).is_err(), "accepted {malicious}");
        }
    }

    #[test]
    fn inline_presentation_style_is_strict_and_omits_only_exact_defaults() {
        let canonical = sanitize_svg(concat!(
            "<svg><path ",
            "style=\"fill:#ffffff;fill-opacity:1;stroke:#000000;stroke-opacity:1;",
            "stroke-width:1.25;stroke-linecap:butt;stroke-linejoin:miter;",
            "display:inline;stroke-miterlimit:4;stroke-dasharray:none\" ",
            "d=\"M0 0H1V1Z\"/></svg>",
        ))
        .expect("strict presentation style");
        for retained in [
            "fill=\"#ffffff\"",
            "fill-opacity=\"1\"",
            "stroke=\"#000000\"",
            "stroke-opacity=\"1\"",
            "stroke-width=\"1.25\"",
            "stroke-linecap=\"butt\"",
            "stroke-linejoin=\"miter\"",
        ] {
            assert!(canonical.svg.contains(retained), "missing {retained}");
        }
        for omitted in [
            "style=",
            "display=",
            "stroke-miterlimit=",
            "stroke-dasharray=",
        ] {
            assert!(!canonical.svg.contains(omitted), "retained {omitted}");
        }
        let default_only = sanitize_svg(
            "<svg><g style=\"display:inline;stroke-miterlimit:4;stroke-dasharray:none\"></g></svg>",
        )
        .expect("default-only style is safely omitted");
        assert!(!default_only.svg.contains("style="));
        let visible_stroke = sanitize_svg(
            "<svg><path style=\"stroke:black;stroke-miterlimit:10;stroke-dasharray:0.12,0.06;stroke-dashoffset:0\" d=\"M0 0H1\"/></svg>",
        )
        .expect("bounded visible stroke presentation");
        for retained in [
            "stroke-miterlimit=\"10\"",
            "stroke-dasharray=\"0.12,0.06\"",
            "stroke-dashoffset=\"0\"",
        ] {
            assert!(visible_stroke.svg.contains(retained), "missing {retained}");
        }

        for malicious in [
            "<svg><rect style=\"fill:url(https://example.invalid/a)\"/></svg>",
            "<svg><rect style=\"@import 'https://example.invalid/a'\"/></svg>",
            "<svg><rect style=\"fill:expression(alert(1))\"/></svg>",
            "<svg><rect style=\"--paint:red\"/></svg>",
            "<svg><rect style=\"fill\"/></svg>",
            "<svg><rect style=\"fill:red;;stroke:black\"/></svg>",
            "<svg><rect style=\"fill:red;fill:blue\"/></svg>",
            "<svg><rect style=\"display:inline;display:inline\"/></svg>",
            "<svg><rect style=\"stroke-miterlimit:4;stroke-miterlimit:4\"/></svg>",
            "<svg><rect fill=\"red\" style=\"fill:blue\"/></svg>",
            "<svg><rect style=\"fill:blue\" fill=\"red\"/></svg>",
            "<svg><rect style=\"fill:r\\65 d\"/></svg>",
            "<svg><rect style=\"fill:red/*comment*/\"/></svg>",
            "<svg><rect stroke-miterlimit=\"0.5\"/></svg>",
            "<svg><rect stroke-miterlimit=\"10px\"/></svg>",
            "<svg><rect style=\"stroke-miterlimit:10px\"/></svg>",
            "<svg><rect stroke-dasharray=\"0 0\"/></svg>",
            "<svg><rect stroke-dashoffset=\"1 2\"/></svg>",
            "<svg><rect stroke-linecap=\"null\"/></svg>",
            "<svg><rect stroke-linejoin=\"null\"/></svg>",
        ] {
            assert!(sanitize_svg(malicious).is_err(), "accepted {malicious}");
        }
    }

    #[test]
    fn inert_export_root_attributes_are_validated_then_removed() {
        let canonical = sanitize_svg(concat!(
            "<svg x=\"0px\" y=\"0\" xml:space=\"preserve\" ",
            "enable-background=\"new 0 0 48 48\" ",
            "sodipodi:version=\"0.32\" sodipodi:docbase=\"C:\\\\archive\\\\icons\" ",
            "style=\"enable-background:new 0 0 48 48;\" viewBox=\"0 0 48 48\">",
            "<path d=\"M0 0H48V48Z\"/></svg>",
        ))
        .expect("inert export metadata");
        for removed in [
            " x=",
            " y=",
            "xml:space",
            "enable-background",
            "sodipodi:",
            "style=",
        ] {
            assert!(!canonical.svg.contains(removed), "retained {removed}");
        }
        for invalid in [
            "<svg x=\"1\"><path d=\"M0 0H1\"/></svg>",
            "<svg xml:space=\"other\"><path d=\"M0 0H1\"/></svg>",
            "<svg enable-background=\"new 0 0 -1 1\"><path d=\"M0 0H1\"/></svg>",
            "<svg style=\"enable-background:url(https://example.invalid)\"><path d=\"M0 0H1\"/></svg>",
        ] {
            assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn root_percent_dimensions_use_exact_validated_viewbox_spans() {
        let canonical = sanitize_svg(
            "<svg width=\"100%\" height=\"100%\" viewBox=\"0 0 1.25 2.5\"><rect width=\"1.25\" height=\"2.5\"/></svg>",
        )
        .expect("exact root percentages");
        assert_eq!((canonical.width, canonical.height), (2, 3));
        assert!(canonical.svg.contains("width=\"1.25\""));
        assert!(canonical.svg.contains("height=\"2.5\""));
        let numeric = sanitize_svg("<svg width=\"12.50px\" height=\"8.00\"></svg>")
            .expect("numeric dimensions remain unchanged");
        assert!(numeric.svg.contains("width=\"12.50px\""));
        assert!(numeric.svg.contains("height=\"8.00\""));
        let physical = sanitize_svg(
            "<svg width=\"210mm\" height=\"297mm\" viewBox=\"0 0 210 297\"><rect width=\"210\" height=\"297\"/></svg>",
        )
        .expect("standard absolute physical dimensions");
        assert_eq!((physical.width, physical.height), (794, 1123));
        assert!(physical.svg.contains("width=\"793.7007874015749\""));
        assert!(physical.svg.contains("height=\"1122.5196850393702\""));
        for dimension in [
            "1in", "1IN", "2.54cm", "2.54CM", "25.4mm", "25.4MM", "101.6Q", "101.6q", "72pt",
            "72PT", "6pc", "6PC",
        ] {
            let canonical = sanitize_svg(&format!(
                "<svg width=\"{dimension}\" height=\"1px\"><rect width=\"1\" height=\"1\"/></svg>"
            ))
            .expect("standard absolute length unit");
            assert_eq!(canonical.width, 96, "{dimension}");
        }

        for invalid in [
            "<svg width=\"99%\" viewBox=\"0 0 10 10\"></svg>",
            "<svg width=\"100.0%\" viewBox=\"0 0 10 10\"></svg>",
            "<svg width=\"100%\"></svg>",
            "<svg width=\"100%\" viewBox=\"0 0 0 10\"></svg>",
            "<svg height=\"100%\" viewBox=\"0 0 10 2049\"></svg>",
            "<svg><rect width=\"100%\" height=\"1\"/></svg>",
            "<svg width=\"1em\" height=\"1\"></svg>",
        ] {
            assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn root_preserve_aspect_ratio_uses_only_the_svg_alignment_grammar() {
        for value in ["none", "xMidYMid", "xMinYMax meet", "defer xMaxYMin slice"] {
            let canonical = sanitize_svg(&format!(
                "<svg viewBox=\"0 0 16 8\" preserveAspectRatio=\"{value}\"><rect width=\"16\" height=\"8\"/></svg>"
            ))
            .expect("standard preserveAspectRatio value");
            assert_eq!((canonical.width, canonical.height), (16, 8));
            assert!(canonical
                .svg
                .contains(&format!("preserveAspectRatio=\"{value}\"")));
        }

        for value in [
            "",
            "xMidYMid meet slice",
            "none meet",
            "xCenterYCenter",
            "xMidYMid url(https://example.invalid/a)",
            "xMidYMid javascript:alert(1)",
        ] {
            assert!(
                sanitize_svg(&format!(
                    "<svg viewBox=\"0 0 16 8\" preserveAspectRatio=\"{value}\"></svg>"
                ))
                .is_err(),
                "accepted {value:?}"
            );
        }
    }

    #[test]
    fn discarded_content_obeys_all_attribute_and_element_budgets() {
        let attributes = |count: usize| {
            (0..count)
                .map(|index| format!(" x{index}=\"v\""))
                .collect::<String>()
        };
        for input in [
            format!("<svg{}></svg>", attributes(25)),
            format!(
                "<svg><sodipodi:namedview{}></sodipodi:namedview></svg>",
                attributes(25)
            ),
            format!("<svg><g{}/></svg>", attributes(13)),
        ] {
            assert!(
                invalid_detail(sanitize_svg(&input)).contains("per-element attribute budget"),
                "wrong refusal for capped element"
            );
        }

        let namedview = concat!(
            "<sodipodi:namedview id=\"x\" pagecolor=\"x\" bordercolor=\"x\" borderopacity=\"1\" ",
            "inkscape:pageopacity=\"0\" inkscape:pageshadow=\"2\" inkscape:zoom=\"1\" ",
            "inkscape:cx=\"1\" inkscape:cy=\"1\" inkscape:document-units=\"in\" ",
            "inkscape:current-layer=\"x\" showgrid=\"false\" units=\"in\" ",
            "inkscape:window-width=\"1\" inkscape:window-height=\"1\" inkscape:window-x=\"1\" ",
            "inkscape:window-y=\"1\" inkscape:window-maximized=\"1\"></sodipodi:namedview>",
        );
        let global = format!(
            "<svg>{}</svg>",
            namedview.repeat(MAX_SVG_ATTRIBUTES / 18 + 1)
        );
        assert!(invalid_detail(sanitize_svg(&global)).contains("total attribute budget"));

        let oversized = "x".repeat(MAX_SVG_ATTRIBUTE_VALUE_BYTES + 1);
        assert!(invalid_detail(sanitize_svg(&format!(
            "<svg><metadata id=\"{oversized}\"><rdf:rdf></rdf:rdf></metadata></svg>"
        )))
        .contains("per-value byte budget"));

        let value = "x".repeat(50_000);
        let total = format!(
            "<svg>{}</svg>",
            format!("<metadata id=\"{value}\"><rdf:rdf></rdf:rdf></metadata>").repeat(4)
        );
        assert!(invalid_detail(sanitize_svg(&total)).contains("total attribute-value byte budget"));

        let elements = format!("<svg>{}</svg>", "<defs id=\"x\"/>".repeat(MAX_SVG_ELEMENTS));
        assert!(invalid_detail(sanitize_svg(&elements)).contains("element budget"));
    }

    #[tokio::test]
    async fn fill_rules_preserve_holes_and_inherited_geometry() {
        for rule in ["nonzero", "evenodd"] {
            let input = format!(
                r#"<svg width="16" height="16"><g fill-rule="{rule}" clip-rule="{rule}"><path fill="red" d="M0 0H16V16H0Z M4 4H12V12H4Z"/></g></svg>"#
            );
            let canonical = sanitize_svg(&input).expect("inert fill rule");
            assert!(canonical.svg.contains(&format!("fill-rule=\"{rule}\"")));
            assert!(canonical.svg.contains(&format!("clip-rule=\"{rule}\"")));
            let temp = tempfile::tempdir().expect("temp dir");
            let preview = temp.path().join("preview.png");
            render_preview(&canonical.svg, 16, 16, &preview)
                .await
                .expect("render rule");
            let pixels = image::open(preview).expect("PNG").to_rgba8();
            assert_eq!(pixels.get_pixel(1, 1)[3], 255);
            assert_eq!(
                pixels.get_pixel(8, 8)[3],
                if rule == "evenodd" { 0 } else { 255 }
            );
        }
    }

    #[test]
    fn fill_rules_reject_invalid_enums_and_resources() {
        for attribute in ["fill-rule", "clip-rule"] {
            for value in [
                "",
                "inherit",
                "EvenOdd",
                "evenodd nonzero",
                "url(#mask)",
                "url(https://example.invalid/a)",
                "file:///tmp/a",
                "e\\76enodd",
                "evenodd;fill:red",
            ] {
                let input = format!(r#"<svg><path {attribute}="{value}" d="M0 0H1V1Z"/></svg>"#);
                assert!(
                    sanitize_svg(&input).is_err(),
                    "accepted {attribute}={value}"
                );
            }
        }
        for input in [
            r#"<svg><path fill-rule="evenodd" clip-path="url(#mask)" d="M0 0H1V1Z"/></svg>"#,
            r#"<svg><clipPath clip-rule="evenodd"/></svg>"#,
            r#"<svg><path fill-rule="evenodd" onclick="alert(1)" d="M0 0H1V1Z"/></svg>"#,
        ] {
            assert!(sanitize_svg(input).is_err(), "{input}");
        }
    }

    #[test]
    fn fill_rules_still_consume_attribute_budgets() {
        let input = format!(
            "<svg>{}</svg>",
            r#"<g fill-rule="evenodd" clip-rule="nonzero" fill="red"/>"#
                .repeat(MAX_SVG_ATTRIBUTES / 3 + 1)
        );
        assert!(invalid_detail(sanitize_svg(&input)).contains("total attribute budget"));
        let value = "evenodd".repeat(MAX_SVG_ATTRIBUTE_VALUE_BYTES / 7 + 1);
        let input = format!(r#"<svg fill-rule="{value}"/>"#);
        assert!(invalid_detail(sanitize_svg(&input)).contains("per-value byte budget"));
    }

    #[test]
    fn visible_dash_patterns_obey_a_separate_number_budget() {
        let values = "1 ".repeat(MAX_SVG_DASH_NUMBERS + 1);
        let input = format!("<svg><path stroke-dasharray=\"{values}\" d=\"M0 0H1\"/></svg>");
        assert!(invalid_detail(sanitize_svg(&input)).contains("dash-number budget"));
    }

    #[tokio::test]
    async fn upstream_fill_rule_svg_keeps_source_and_gets_bounded_comparison_preview() {
        let raw = include_bytes!("../tests/fixtures/starvector/upstream-34829516753-case-00.svg");
        let canonical = terminal_sanitize_svg_bytes(raw).expect("actual upstream case accepted");
        assert_eq!((canonical.width, canonical.height), (80, 80));
        assert!(canonical.canonical_svg.contains("fill-rule=\"evenodd\""));
        assert!(canonical.canonical_svg.contains("clip-rule=\"evenodd\""));
        assert!(canonical.canonical_svg.contains("viewBox=\"0 0 80 80\""));
        let temp = tempfile::tempdir().expect("temp dir");
        let (_, intrinsic) =
            terminal_write_sanitized_pair(&canonical, &temp.path().join("intrinsic"))
                .await
                .expect("intrinsic pair");
        let (source, comparison) = terminal_write_sanitized_pair_with_preview_size(
            &canonical,
            &temp.path().join("comparison"),
            Some(512),
        )
        .await
        .expect("comparison pair");
        assert_eq!(
            image::open(intrinsic).expect("PNG").to_rgba8().dimensions(),
            (80, 80)
        );
        let pixels = image::open(comparison).expect("PNG").to_rgba8();
        assert_eq!(pixels.dimensions(), (512, 512));
        assert!(
            pixels.pixels().any(|pixel| pixel[3] > 0),
            "actual geometry renders"
        );
        assert_eq!(
            std::fs::read_to_string(source).expect("canonical source"),
            canonical.canonical_svg
        );
        for size in [0, MAX_PREVIEW_DIMENSION + 1] {
            let destination = temp.path().join(format!("invalid-{size}"));
            assert!(terminal_write_sanitized_pair_with_preview_size(
                &canonical,
                &destination,
                Some(size)
            )
            .await
            .is_err());
            assert!(!destination.exists());
        }
        // A rectangular viewport is fitted and centered, not stretched to fill the square.
        let rectangle = terminal_sanitize_svg_bytes(
            br#"<svg viewBox="0 0 16 8"><rect width="16" height="8" fill="red"/></svg>"#,
        )
        .expect("rectangle");
        let (_, preview) = terminal_write_sanitized_pair_with_preview_size(
            &rectangle,
            &temp.path().join("rectangle"),
            Some(32),
        )
        .await
        .expect("fitted rectangle");
        let pixels = image::open(preview).expect("PNG").to_rgba8();
        assert_eq!(pixels.get_pixel(16, 2)[3], 0);
        assert_eq!(pixels.get_pixel(16, 16)[3], 255);
        assert_eq!(pixels.get_pixel(16, 29)[3], 0);
    }

    #[tokio::test]
    async fn upstream_case_four_preserves_viewbox_aspect_ratio_rendering() {
        let raw = include_bytes!("../tests/fixtures/starvector/upstream-34841832913-case-04.svg");
        let canonical =
            terminal_sanitize_svg_bytes(raw).expect("actual upstream case four accepted");
        assert_eq!((canonical.width, canonical.height), (1000, 1000));
        assert!(canonical
            .canonical_svg
            .contains("viewBox=\"0 0 1000 1000\""));
        assert!(canonical
            .canonical_svg
            .contains("preserveAspectRatio=\"xMidYMid\""));
        let temp = tempfile::tempdir().expect("temp dir");
        let (_, preview) = terminal_write_sanitized_pair_with_preview_size(
            &canonical,
            &temp.path().join("actual-upstream-case-04"),
            Some(512),
        )
        .await
        .expect("actual case renders through the canonical CLI seam");
        let pixels = image::open(preview).expect("preview PNG").to_rgba8();
        assert_eq!(pixels.dimensions(), (512, 512));
        assert!(
            pixels.pixels().any(|pixel| pixel[3] > 0),
            "actual visible geometry renders"
        );
    }

    fn comparison_pixels(svg: &str, size: u32) -> Vec<u8> {
        let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).expect("comparison SVG");
        let intrinsic = tree.size();
        let scale = (size as f32 / intrinsic.width()).min(size as f32 / intrinsic.height());
        let x = (size as f32 - intrinsic.width() * scale) / 2.0;
        let y = (size as f32 - intrinsic.height() * scale) / 2.0;
        let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size).expect("comparison pixmap");
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y),
            &mut pixmap.as_mut(),
        );
        pixmap.data().to_vec()
    }

    #[test]
    fn bounded_internal_resources_preserve_all_five_upstream_renders() {
        let cases = [
            include_str!("../tests/fixtures/starvector/upstream-34847121425-case-01.svg"),
            include_str!("../tests/fixtures/starvector/upstream-34847121425-case-03.svg"),
            include_str!("../tests/fixtures/starvector/upstream-34847121425-case-04.svg"),
            include_str!("../tests/fixtures/starvector/upstream-34847121425-case-09.svg"),
            include_str!("../tests/fixtures/starvector/upstream-34847121425-case-17.svg"),
        ];
        for (index, raw) in cases.into_iter().enumerate() {
            let canonical = sanitize_svg(raw).unwrap_or_else(|error| {
                panic!("upstream internal-resource case {index} failed: {error}")
            });
            let tree = usvg::Tree::from_str(&canonical.svg, &usvg::Options::default())
                .expect("canonical comparison SVG");
            let intrinsic = tree.size();
            let scale = (512.0 / intrinsic.width()).min(512.0 / intrinsic.height());
            let x = (512.0 - intrinsic.width() * scale) / 2.0;
            let y = (512.0 - intrinsic.height() * scale) / 2.0;
            validate_filter_render_budget(
                &tree,
                512,
                512,
                resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y),
            )
            .unwrap_or_else(|error| {
                panic!("upstream internal-resource case {index} exceeded preview budget: {error}")
            });
            assert_eq!(
                comparison_pixels(raw, 512),
                comparison_pixels(&canonical.svg, 512),
                "upstream internal-resource case {index} changed visible pixels"
            );
            assert_eq!(
                canonical.svg,
                sanitize_svg(raw).expect("repeat sanitation").svg,
                "upstream internal-resource case {index} is not byte deterministic"
            );
        }
    }

    #[test]
    fn internal_resource_graph_is_typed_resolved_acyclic_and_forward_safe() {
        let forward = concat!(
            "<svg width=\"8\" height=\"8\"><rect width=\"8\" height=\"8\" fill=\"url(#g)\"/>",
            "<defs><linearGradient id=\"g\"><stop offset=\"0\" stop-color=\"red\"/>",
            "<stop offset=\"1\" stop-color=\"blue\"/></linearGradient></defs></svg>"
        );
        sanitize_svg(forward).expect("forward local resource reference");
        for invalid in [
            "<svg><path fill=\"url(#missing)\" d=\"M0 0H1\"/></svg>",
            "<svg><defs><clipPath id=\"x\"><rect/></clipPath></defs><path fill=\"url(#x)\" d=\"M0 0H1\"/></svg>",
            "<svg><defs><clipPath id=\"x\"><rect/></clipPath><filter id=\"x\" width=\"1\" height=\"1\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"1\"/></filter></defs></svg>",
            "<svg><defs><linearGradient id=\"a\" href=\"#b\"/><linearGradient id=\"b\" href=\"#a\"/></defs></svg>",
            "<svg><defs><linearGradient id=\"a\" href=\"#a\"/></defs></svg>",
            "<svg><defs><clipPath id=\"c\"><rect clip-path=\"url(#c)\"/></clipPath></defs></svg>",
            "<svg><defs><linearGradient id=\"a\" href=\"#b\" xlink:href=\"#b\"/><linearGradient id=\"b\"/></defs></svg>",
            "<svg><defs><linearGradient id=\"a\" href=\"https://example.invalid/g\"/></defs></svg>",
            "<svg><defs><linearGradient id=\"a\"/></defs><path fill=\"url( #a )\" d=\"M0 0H1\"/></svg>",
            "<svg><defs><linearGradient id=\"a\"><stop offset=\"2\" stop-color=\"red\"/></linearGradient></defs></svg>",
            "<svg><defs><linearGradient id=\"a\"><stop offset=\"0\" style=\"stop-color:red;;stop-opacity:1\"/></linearGradient></defs></svg>",
            "<svg><defs><linearGradient id=\"a\"><stop offset=\"0\" style=\"stop-color:url(#a)\"/></linearGradient></defs></svg>",
        ] {
            assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
        }
        let unbound_xlink = "<svg><defs><linearGradient id=\"a\" xlink:href=\"#b\"/><linearGradient id=\"b\"/></defs></svg>";
        assert!(
            invalid_detail(sanitize_svg(unbound_xlink)).contains("exact root xmlns:xlink binding"),
            "unbound xlink prefix must fail closed"
        );
    }

    #[test]
    fn internal_resource_and_filter_costs_are_independently_bounded() {
        let resources = (0..=MAX_SVG_RESOURCES)
            .map(|index| format!("<clipPath id=\"c{index}\"><rect/></clipPath>"))
            .collect::<String>();
        assert!(invalid_detail(sanitize_svg(&format!(
            "<svg><defs>{resources}</defs></svg>"
        )))
        .contains("resource budget"));

        let references = (0..=MAX_SVG_RESOURCE_REFERENCES)
            .map(|_| "<path fill=\"url(#g)\" d=\"M0 0H1\"/>".to_owned())
            .collect::<String>();
        let input = format!(
            "<svg><defs><linearGradient id=\"g\"><stop offset=\"0\" stop-color=\"red\"/></linearGradient></defs>{references}</svg>"
        );
        assert!(invalid_detail(sanitize_svg(&input)).contains("resource-reference budget"));

        let stops = "<stop offset=\"0\" stop-color=\"red\"/>".repeat(MAX_SVG_GRADIENT_STOPS + 1);
        assert!(invalid_detail(sanitize_svg(&format!(
            "<svg><defs><linearGradient id=\"g\">{stops}</linearGradient></defs></svg>"
        )))
        .contains("gradient-stop budget"));

        let mut primitives = String::new();
        let mut input = "SourceAlpha".to_owned();
        for index in 0..=MAX_SVG_FILTER_PRIMITIVES {
            let result = format!("r{index}");
            primitives.push_str(&format!(
                "<feOffset in=\"{input}\" result=\"{result}\" dx=\"1\" dy=\"1\"/>"
            ));
            input = result;
        }
        let svg = format!(
            "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\">{primitives}</filter></defs></svg>"
        );
        assert!(invalid_detail(sanitize_svg(&svg)).contains("filter-primitive budget"));

        for invalid in [
            "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\"><feGaussianBlur in=\"later\" result=\"b\" stdDeviation=\"1\"/></filter></defs></svg>",
            "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"65\"/></filter></defs></svg>",
            "<svg width=\"2048\" height=\"2048\"><defs><filter id=\"f\" width=\"200%\" height=\"200%\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"1\"/></filter></defs></svg>",
            "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\"><feOffset in=\"SourceAlpha\" result=\"b\"/><feOffset in=\"b\" result=\"b\"/></filter></defs></svg>",
        ] {
            assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
        }

        let scaled = "<svg width=\"900\" height=\"600\"><defs><filter id=\"f\" width=\"200%\" height=\"200%\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"1\"/></filter></defs><g filter=\"url(#f)\" transform=\"scale(1000)\"><rect width=\"900\" height=\"600\"/></g></svg>";
        assert!(
            invalid_detail(sanitize_svg(scaled)).contains("offscreen-pixel budget"),
            "resolved transformed filter layer must be charged"
        );

        let filter = "<filter id=\"f\" width=\"100%\" height=\"100%\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"1\"/></filter>";
        let applications =
            "<g filter=\"url(#f)\"><rect width=\"2048\" height=\"2048\"/></g>".repeat(3);
        let repeated = format!(
            "<svg width=\"2048\" height=\"2048\"><defs>{filter}</defs>{applications}</svg>"
        );
        let detail = invalid_detail(sanitize_svg(&repeated));
        assert!(
            detail.contains("offscreen-pixel budget"),
            "filter applications must be charged independently: {detail}"
        );

        let case_17 = sanitize_svg(include_str!(
            "../tests/fixtures/starvector/upstream-34847121425-case-17.svg"
        ))
        .expect("case 17 stays within its intrinsic render budget");
        let tree = usvg::Tree::from_str(&case_17.svg, &usvg::Options::default())
            .expect("case 17 canonical tree");
        let intrinsic = tree.size();
        let scale = (MAX_PREVIEW_DIMENSION as f32 / intrinsic.width())
            .min(MAX_PREVIEW_DIMENSION as f32 / intrinsic.height());
        let x = (MAX_PREVIEW_DIMENSION as f32 - intrinsic.width() * scale) / 2.0;
        let y = (MAX_PREVIEW_DIMENSION as f32 - intrinsic.height() * scale) / 2.0;
        let detail = validate_filter_render_budget(
            &tree,
            MAX_PREVIEW_DIMENSION,
            MAX_PREVIEW_DIMENSION,
            resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y),
        )
        .expect_err("oversized case 17 preview must be rejected")
        .to_string();
        assert!(
            detail.contains("offscreen-pixel budget"),
            "preview transforms must be included in the allocation bound: {detail}"
        );
    }

    #[test]
    fn hidden_geometry_is_omitted_but_remains_strictly_parsed_and_bounded() {
        let accepted = sanitize_svg(
            "<svg><g display=\"none\"><rect width=\"100%\" height=\"100%\" fill=\"url(#missing)\"/></g><path d=\"M0 0H1\"/></svg>",
        )
        .expect("bounded hidden geometry");
        assert!(!accepted.svg.contains("display"));
        assert!(!accepted.svg.contains("missing"));
        for invalid in [
            "<svg><g display=\"none\"><script/></g></svg>",
            "<svg><g display=\"none\"><text>hidden</text></g></svg>",
            "<svg><g display=\"none\"><use href=\"#x\"/></g></svg>",
            "<svg><g display=\"none\"><rect fill=\"url(https://example.invalid/a)\"/></g></svg>",
            "<svg><g display=\"none\" onclick=\"alert(1)\"><rect/></g></svg>",
        ] {
            assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
        }
        let paths = format!(
            "<svg><g display=\"none\"><path d=\"{}\"/></g></svg>",
            "M".repeat(MAX_SVG_PATH_COMMANDS + 1)
        );
        assert!(invalid_detail(sanitize_svg(&paths)).contains("path-command budget"));
    }

    #[test]
    fn rejects_active_resources_css_references_and_text() {
        for malicious in [
            "<svg><script>alert(1)</script></svg>",
            "<svg><rect fill=\"url(https://example.invalid/a)\"/></svg>",
            "<svg><rect fill=\"u r l( data:image/png;base64,AA== )\"/></svg>",
            "<svg><rect fill=\"u\\72l(https://example.invalid/a)\"/></svg>",
            "<svg><rect fill=\"u/*escaped*/rl(https://example.invalid/a)\"/></svg>",
            "<svg><rect fill=\"file:///tmp/payload\"/></svg>",
            "<svg><rect fill=\"http://example.invalid/a\"/></svg>",
            "<svg><style>rect { fill: red }</style></svg>",
            "<svg><use href=\"#shape\"/></svg>",
            "<svg><rect/>not inert</svg>",
            "<svg><rect onclick=\"alert(1)\"/></svg>",
            "<svg xmlns=\"https://example.invalid/not-svg\"/>",
        ] {
            assert!(sanitize_svg(malicious).is_err(), "{malicious}");
        }
    }

    #[test]
    fn enforces_byte_utf8_node_depth_and_attribute_budgets() {
        assert_eq!(
            invalid_detail(sanitize_svg_bytes(&vec![b' '; MAX_SVG_BYTES + 1])),
            "provider SVG exceeds the 256 KiB sanitizer budget"
        );
        assert_eq!(
            invalid_detail(sanitize_svg_bytes(&[0xff])),
            "provider SVG is not valid UTF-8"
        );

        let nodes = format!("<svg>{}</svg>", "<g/>".repeat(MAX_SVG_ELEMENTS));
        assert!(invalid_detail(sanitize_svg(&nodes)).contains("element budget"));

        let depth = format!(
            "<svg>{}{}</svg>",
            "<g>".repeat(MAX_SVG_DEPTH),
            "</g>".repeat(MAX_SVG_DEPTH)
        );
        assert!(invalid_detail(sanitize_svg(&depth)).contains("nesting budget"));

        let nine_attributes = concat!(
            " fill=\"#000\" fill-opacity=\"1\" stroke=\"#000\" stroke-opacity=\"1\"",
            " stroke-width=\"1\" stroke-linecap=\"round\" stroke-linejoin=\"round\"",
            " opacity=\"1\" transform=\"scale(1)\""
        );
        let elements = (MAX_SVG_ATTRIBUTES / 9) + 1;
        let attributes = format!(
            "<svg>{}</svg>",
            format!("<g{nine_attributes}/>").repeat(elements)
        );
        assert!(invalid_detail(sanitize_svg(&attributes)).contains("total attribute budget"));

        let per_element = concat!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\" viewBox=\"0 0 1 1\"",
            " fill=\"#000\" fill-opacity=\"1\" stroke=\"#000\" stroke-opacity=\"1\"",
            " stroke-width=\"1\" stroke-linecap=\"round\" stroke-linejoin=\"round\"",
            " opacity=\"1\" transform=\"scale(1)\"></svg>"
        );
        assert!(invalid_detail(sanitize_svg(per_element)).contains("per-element attribute budget"));

        let value = "x".repeat(MAX_SVG_ATTRIBUTE_VALUE_BYTES + 1);
        assert!(
            invalid_detail(sanitize_svg(&format!("<svg><g fill=\"{value}\"/></svg>")))
                .contains("per-value byte budget")
        );

        let value = "x".repeat(50_000);
        let total_values = format!(
            "<svg><g fill=\"{value}\"/><g fill=\"{value}\"/><g fill=\"{value}\"/><g fill=\"{value}\"/></svg>"
        );
        assert!(invalid_detail(sanitize_svg(&total_values))
            .contains("total attribute-value byte budget"));
    }

    #[test]
    fn enforces_path_command_number_and_path_byte_budgets() {
        let commands = "M".repeat(MAX_SVG_PATH_COMMANDS + 1);
        assert!(invalid_detail(sanitize_svg(&format!(
            "<svg><path d=\"{commands}\"/></svg>"
        )))
        .contains("path-command budget"));

        let numbers_per_path = (MAX_SVG_PATH_NUMBERS / 3) + 1;
        let number_path = format!("M{}", " 0".repeat(numbers_per_path));
        let number_svg = format!(
            "<svg><path d=\"{number_path}\"/><path d=\"{number_path}\"/><path d=\"{number_path}\"/></svg>"
        );
        assert!(invalid_detail(sanitize_svg(&number_svg)).contains("path-number budget"));

        let bytes_per_path = (MAX_SVG_PATH_DATA_BYTES / 3) + 1;
        let byte_path = format!("M{}", " 0".repeat(bytes_per_path / 2));
        let byte_svg = format!(
            "<svg><path d=\"{byte_path}\"/><path d=\"{byte_path}\"/><path d=\"{byte_path}\"/></svg>"
        );
        assert!(invalid_detail(sanitize_svg(&byte_svg)).contains("path-data byte budget"));
    }

    #[test]
    fn enforces_points_transform_coordinate_dimension_and_viewbox_budgets() {
        let points_per_element = (MAX_SVG_POINT_NUMBERS / 2) + 2;
        let points = "0 ".repeat(points_per_element);
        let points_svg =
            format!("<svg><polyline points=\"{points}\"/><polyline points=\"{points}\"/></svg>");
        assert!(invalid_detail(sanitize_svg(&points_svg)).contains("point-number budget"));

        let transform = "translate(0)".repeat(MAX_SVG_TRANSFORM_NUMBERS + 1);
        assert!(invalid_detail(sanitize_svg(&format!(
            "<svg><g transform=\"{transform}\"/></svg>"
        )))
        .contains("transform-number budget"));

        assert!(invalid_detail(sanitize_svg(
            "<svg><rect x=\"1000001\" width=\"1\" height=\"1\"/></svg>"
        ))
        .contains("coordinate-magnitude budget"));
        assert!(
            invalid_detail(sanitize_svg("<svg width=\"2049px\" height=\"1px\"></svg>"))
                .contains("dimensions must be")
        );
        assert!(
            invalid_detail(sanitize_svg("<svg viewBox=\"1000001 0 1 1\"></svg>"))
                .contains("viewBox origin")
        );
    }

    #[tokio::test]
    async fn renders_a_bounded_png_preview_from_the_canonical_svg() {
        let canonical = sanitize_svg(
            "<svg width=\"12\" height=\"8\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"12\" height=\"8\" fill=\"#ff0000\"/></svg>",
        )
        .expect("valid fixture");
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("preview.png");
        render_preview(&canonical.svg, canonical.width, canonical.height, &path)
            .await
            .expect("preview writes");
        let preview = image::open(&path).expect("preview decodes").to_rgba8();
        assert_eq!(preview.dimensions(), (12, 8));
        assert!(
            preview.get_pixel(0, 0)[0] > 200,
            "preview rendered the red rectangle"
        );
    }

    #[tokio::test]
    async fn terminal_seam_delegates_to_sanitizer_and_publishes_only_the_pair() {
        assert!(terminal_sanitize_svg_bytes(&[0xff]).is_err());
        let value = terminal_sanitize_svg_bytes(
            b"<svg width=\"12\" height=\"8\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"12\" height=\"8\" fill=\"#00ff00\"/></svg>",
        )
        .expect("valid inert SVG");
        let temp = tempfile::tempdir().expect("temp dir");
        let destination = temp.path().join("published");
        let (svg, preview) = terminal_write_sanitized_pair(&value, &destination)
            .await
            .expect("atomic terminal publication");
        assert_eq!(
            svg.file_name().and_then(|name| name.to_str()),
            Some("canonical.svg")
        );
        assert_eq!(
            preview.file_name().and_then(|name| name.to_str()),
            Some("preview.png")
        );
        assert!(svg.is_file() && preview.is_file());
        assert!(
            std::fs::read_dir(temp.path())
                .expect("read temp")
                .all(|entry| !entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .contains(".tmp")),
            "terminal publication leaves no staging residue"
        );
    }
}
