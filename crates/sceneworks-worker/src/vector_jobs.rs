//! Vector Studio request, provider, and safe-publication boundary.
//!
//! The route supplies only typed raster/text conditioning. A mode-specific native provider streams
//! the SVG through [`MultimodalVectorProviderAdapter`]; the worker does not create a staging
//! directory until that stream has completed without cancellation. The source is then parsed into
//! a deliberately small inert SVG subset, canonicalized, rendered through resvg (which has no
//! network/resource loader), and published as an SVG+PNG directory rename.

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
            None,
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
                        .generate_svg(&typed_request, &mut |event| events.push(event))
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

fn collect_svg_source(
    provider: &dyn MultimodalVectorProviderAdapter,
    request: &VectorProviderRequest,
    cancel: &gen_core::CancelFlag,
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
    let mut expected_index = 0u32;
    provider.generate_svg(request, cancel, &mut |fragment, index| {
        if cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        if index != expected_index {
            return Err(WorkerError::Engine(format!(
                "vector provider emitted non-monotonic source index {index}; expected {expected_index}"
            )));
        }
        expected_index = expected_index.saturating_add(1);
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
            if std::env::var("SCENEWORKS_TERMINAL_CAMPAIGN").as_deref() == Ok("1") {
                return Ok(CollectedSvgSource {
                    source: None,
                    terminal: Some(terminal.clone()),
                });
            }
            return match terminal.finish_reason {
                "cancelled" => Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned())),
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
    let task = tokio::task::spawn_blocking(move || {
        collect_svg_source(
            blocking_provider.as_ref(),
            &blocking_request,
            &blocking_cancel,
        )
    });
    let collected = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        CANCEL_MESSAGE,
        "vector provider stream",
        no_cancel_ack(),
        task,
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
                terminal_result(&terminal, None, None, Some((&transcript_path, &transcript))),
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
    let canonical = sanitize_svg(&source)?;
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
        render_preview(&value.canonical_svg, value.width, value.height, &preview).await?;
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
}

fn sanitize_svg(input: &str) -> WorkerResult<CanonicalSvg> {
    sanitize_svg_bytes(input.as_bytes())
}

fn sanitize_svg_bytes(input: &[u8]) -> WorkerResult<CanonicalSvg> {
    if input.len() > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG exceeds the 256 KiB sanitizer budget".to_owned(),
        ));
    }
    let input = std::str::from_utf8(input)
        .map_err(|_| WorkerError::InvalidPayload("provider SVG is not valid UTF-8".to_owned()))?;
    let mut reader = Reader::from_reader(input.as_bytes());
    reader.config_mut().trim_text(true);
    let mut buffer = Vec::new();
    let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let mut stack = Vec::new();
    let mut elements = 0usize;
    let mut budget = SanitizerBudget::default();
    let mut dimensions = None;
    let mut root_seen = false;
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => {
                let (name, mut attrs) = canonical_element(&reader, &event, &mut budget)?;
                if stack.is_empty() && (root_seen || name != "svg") {
                    return Err(WorkerError::InvalidPayload(
                        "provider output root must be <svg>".to_owned(),
                    ));
                }
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
                if stack.is_empty() {
                    root_seen = true;
                    if !attrs
                        .iter()
                        .any(|(key, value)| key == "xmlns" && value == SVG_NAMESPACE)
                    {
                        attrs.push(("xmlns".to_owned(), SVG_NAMESPACE.to_owned()));
                        attrs.sort_unstable();
                    }
                    dimensions = Some(svg_dimensions(&attrs)?);
                }
                write_start(&mut output, &name, &attrs, false);
                stack.push(name);
            }
            Ok(Event::Empty(event)) => {
                let (name, attrs) = canonical_element(&reader, &event, &mut budget)?;
                if stack.is_empty() || name == "svg" {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has an invalid empty root".to_owned(),
                    ));
                }
                elements += 1;
                if elements > MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element budget".to_owned(),
                    ));
                }
                write_start(&mut output, &name, &attrs, true);
            }
            Ok(Event::End(event)) => {
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                if stack.pop().as_deref() != Some(name.as_str()) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has mismatched tags".to_owned(),
                    ));
                }
                output.push_str("</");
                output.push_str(&name);
                output.push('>');
            }
            Ok(Event::Text(text)) => {
                let bytes: &[u8] = text.as_ref();
                if !bytes.iter().all(u8::is_ascii_whitespace) {
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
    usvg::Tree::from_str(&output, &usvg::Options::default()).map_err(|error| {
        WorkerError::InvalidPayload(format!("provider SVG cannot be rendered: {error}"))
    })?;
    let (width, height) = dimensions.expect("validated above");
    Ok(CanonicalSvg {
        svg: output,
        width,
        height,
    })
}

fn canonical_element(
    reader: &Reader<&[u8]>,
    event: &quick_xml::events::BytesStart<'_>,
    budget: &mut SanitizerBudget,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    let name = std::str::from_utf8(event.name().as_ref())
        .map_err(|_| WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned()))?
        .to_owned();
    if !matches!(
        name.as_str(),
        "svg" | "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
    ) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> is not allowed"
        )));
    }
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
        // The root namespace is the one URL-looking value admitted by this inert subset; it names
        // SVG syntax and is never dereferenced. Every actual resource-bearing URL is rejected.
        let namespace_ok = key == "xmlns" && value == SVG_NAMESPACE;
        if !allowed_attribute(&name, &key)
            || (key == "xmlns" && !namespace_ok)
            || (!namespace_ok && unsafe_attribute_value(&value))
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is not allowed"
            )));
        }
        validate_attribute_resource_budget(&name, &key, &value, budget)?;
        attrs.push((key, value));
    }
    if attrs.len() > MAX_SVG_ATTRIBUTES_PER_ELEMENT {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> exceeds the per-element attribute budget"
        )));
    }
    attrs.sort_unstable();
    Ok((name, attrs))
}

fn allowed_attribute(element: &str, attribute: &str) -> bool {
    let common = matches!(
        attribute,
        "fill"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "opacity"
            | "transform"
    );
    common
        || match element {
            "svg" => matches!(attribute, "xmlns" | "width" | "height" | "viewBox"),
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
    let svg = svg.to_owned();
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).map_err(|error| {
            WorkerError::InvalidPayload(format!("provider SVG cannot be rendered: {error}"))
        })?;
        let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG preview dimensions are invalid".to_owned())
        })?;
        // resvg draws only the already-sanitized inert geometry into a fixed-size PNG.
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        pixmap
            .save_png(path)
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))
    })
    .await
    .map_err(|error| task_join_error("SVG preview render", error))??;
    Ok(())
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
            on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
        ) -> WorkerResult<()> {
            on_source("<svg xmlns=\"http://www.w3.org/2000/svg\">", 0)?;
            cancel.cancel();
            on_source("<rect width=\"1\" height=\"1\"/></svg>", 1)?;
            Ok(())
        }
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
        let result = terminal_result(&terminal, None, None, None);
        assert_eq!(result["accepted"], false);
        assert_eq!(result["finishReason"], "token_limit");
        assert_eq!(result["providerTranscriptSha256"], Value::Null);
        assert!(result["canonicalSvgPath"].is_null());
        assert!(result["previewPngPath"].is_null());
        assert!(result["providerTranscriptPath"].is_null());
        assert_eq!(result["resultContainsInlineSvg"], false);
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
        let output = collect_svg_source(&CancelingProvider, &request, &cancel);
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
            "<svg><rect style=\"fill: red\"/></svg>",
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
