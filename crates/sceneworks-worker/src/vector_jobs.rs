//! Vector Studio request, provider, and safe-publication boundary.
//!
//! The route supplies only typed raster/text conditioning. A mode-specific native provider streams
//! the SVG through [`MultimodalVectorProviderAdapter`]; the worker does not create a staging
//! directory until that stream has completed without cancellation. The source is then parsed into
//! bounded static SVG, canonicalized through the same usvg tree that resvg renders, and published
//! as an SVG+PNG directory rename. Active content and external resources never cross this boundary.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gen_core::core_llm::{
    Content, ImageRef, LoadSpec as TextLoadSpec, Message, ModelRequirements, Role, Sampling,
    StarVectorFinishReason, StarVectorOutput, StarVectorRequest, StarVectorStreamEvent,
    StarVectorTier, TextLlmRequest,
};
use resvg::usvg;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::*;

#[path = "svg_sanitize.rs"]
mod svg_sanitize;
use svg_sanitize::*;

const MAX_PREVIEW_DIMENSION: u32 = 2_048;
const VECTOR_RENDERER_VERSION: &str = "resvg-0.45";
const TERMINAL_COMPARISON_SIZE: u32 = 512;
const CANCEL_MESSAGE: &str = "Vector generation canceled before publication.";
const STARVECTOR_ADAPTER_ID: &str = "starvector";

fn sha256_hex(bytes: &[u8]) -> String {
    sceneworks_core::hashing::sha256_bytes(bytes)
}

fn terminal_result(
    terminal: &TerminalProviderOutcome,
    product_latency_seconds: Option<f64>,
    canonical: Option<(&Path, &[u8])>,
    preview: Option<(&Path, &[u8])>,
    comparison: Option<(&Path, &[u8])>,
    transcript: Option<(&Path, &[u8])>,
) -> Value {
    json!({
        "accepted": terminal.publishable(),
        "finishReason": terminal.finish_reason,
        "generatedTokens": terminal.generated_tokens,
        "generatedBytes": terminal.generated_bytes,
        // Retain the original key for historical receipts. It has always measured only the
        // native provider call, not completed product output.
        "latencySeconds": terminal.latency_seconds,
        "providerLatencySeconds": terminal.latency_seconds,
        "productLatencySeconds": product_latency_seconds,
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
        "comparisonPngPath": comparison.map(|(path, _)| path.to_string_lossy().into_owned()),
        "comparisonPngSha256": comparison.map(|(_, bytes)| sha256_hex(bytes)),
        "resultContainsInlineSvg": false,
    })
}

fn terminal_generation_limit_result(
    terminal: &TerminalProviderOutcome,
    product_latency_seconds: f64,
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
    let mut evidence = terminal_result(
        terminal,
        Some(product_latency_seconds),
        None,
        None,
        None,
        Some(transcript),
    );
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
        "providerLatencySeconds": terminal.latency_seconds,
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

/// Build the flat vector asset fact consumed by the API's generated-asset
/// persistence boundary. Kept public for the API contract test: the test must
/// exercise the exact producer shape rather than a hand-written approximation.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn build_vector_asset_fact(
    asset_id: &str,
    generation_set_id: &str,
    width: u32,
    height: u32,
    created_at: &str,
    mode: &str,
    model: &str,
    adapter: &str,
    prompt: &str,
    source_asset_id: Option<&str>,
    sampling: Value,
    detail_budget: Value,
    workflow: Option<Value>,
) -> Value {
    let title: String = prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = format!(
        "{} #1",
        if title.is_empty() {
            "Generated vector"
        } else {
            title
        }
    );
    json!({
        "assetId": asset_id,
        "type": "vector",
        "mediaPath": format!("assets/images/{generation_set_id}/{asset_id}/vector.svg"),
        "mimeType": "image/svg+xml",
        "width": width,
        "height": height,
        "displayName": display_name,
        "createdAt": created_at,
        "mode": mode,
        "model": model,
        "adapter": adapter,
        "prompt": prompt,
        "negativePrompt": "",
        "sourceAssetId": source_asset_id,
        "sampling": sampling,
        "detailBudget": detail_budget,
        "sanitizerVersion": VECTOR_SANITIZER_VERSION,
        "rendererVersion": VECTOR_RENDERER_VERSION,
        "workflow": workflow,
        "count": 1,
        "normalizedWidth": width,
        "normalizedHeight": height,
        "preview": {
            "path": format!("assets/images/{generation_set_id}/{asset_id}/preview.png"),
            "mimeType": "image/png",
            "width": width,
            "height": height,
        },
    })
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

    /// The KV cache of the provider's last decode, whichever way it ended (sc-20682, sc-20688):
    /// what [`collect_svg_source`] records as the job's `llm_kv_cache` event. `None` for a
    /// provider that runs no native LLM decode.
    fn kv_cache_record(&self) -> Option<crate::llm_kv_cache::KvCacheRecord> {
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
    kv_cache: Arc<Mutex<Option<crate::llm_kv_cache::KvCacheRecord>>>,
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
            kv_cache: Arc::new(Mutex::new(None)),
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
        let kv_cache = self.kv_cache.clone();
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
                    let (output, events) = decode_native_starvector(
                        provider, &typed_request, &progress, backend, &kv_cache,
                    )?;
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

    fn kv_cache_record(&self) -> Option<crate::llm_kv_cache::KvCacheRecord> {
        self.kv_cache.lock().ok()?.clone()
    }
}

/// Decode one SVG on the typed StarVector `provider`, publishing its token progress, and keep the
/// decode's KV-cache record in `kv_cache` whichever way it ends — completed, or refused, canceled
/// or failed by the engine (sc-20682, sc-20688).
fn decode_native_starvector(
    provider: &dyn gen_core::core_llm::StarVectorProvider,
    typed_request: &StarVectorRequest,
    progress: &tokio::sync::watch::Sender<u32>,
    backend: &'static str,
    kv_cache: &Mutex<Option<crate::llm_kv_cache::KvCacheRecord>>,
) -> WorkerResult<(StarVectorOutput, Vec<StarVectorStreamEvent>)> {
    // sc-24029: StarVector decodes on the same resident `TextLlm` with the same
    // per-token-growing KV cache. `Progress` is the decode's per-token event, so
    // counting its arrivals bounds the cache the same way, and this callback is used
    // because it is the only hook interleaved with the decode — NOT because the
    // allocator is thread-local. It is not: MLX's freed-buffer cache is
    // PROCESS-GLOBAL (inference `crates/llm/mlx-llm/src/starvector_8b.rs` declines to
    // clear it in `unload` for exactly that reason), so up to once per 16 streamed
    // token events — plus once at decode end — this clear also discards buffers a
    // CONCURRENT image render had cached, forcing that render to re-allocate. The
    // terminal clear fires when this function returns.
    let mut cache_bound = crate::mlx_decode_cache::DecodeCacheBound::mlx();
    let mut events = Vec::new();
    let generation = provider.generate_svg(typed_request, &mut |event| {
        if let StarVectorStreamEvent::Progress { generated_tokens } = &event {
            cache_bound.note_event();
            // Only counters leave this private source boundary. A watch channel
            // stores one value even if decode outruns the async publisher.
            record_vector_token_progress(
                progress,
                *generated_tokens,
                typed_request.text_request.max_new_tokens,
            );
        } else {
            events.push(event);
        }
    });
    let record = crate::llm_kv_cache::KvCacheRecord::of(
        backend,
        typed_request.text_request.kv_compression,
        &generation,
        |output| output.kv_cache.as_ref(),
    );
    *kv_cache.lock().map_err(|_| {
        WorkerError::Engine("StarVector KV-cache record lock poisoned".to_owned())
    })? = Some(record);
    let output = generation.map_err(classify_starvector_error)?;
    Ok((output, events))
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

    #[cfg(not(native_lane))]
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
        progress: Some(crate::load_progress::text_observer()),
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
            // SceneWorks exposes no presence-penalty knob; 0.0 is the contract's "disabled"
            // value and preserves the sampling behaviour from before the field existed.
            presence_penalty: 0.0,
            presence_penalty_generated_only: false,
            repetition_penalty: request.sampling.repetition_penalty,
            repetition_context,
        },
        max_new_tokens: request.detail_budget.max_new_tokens,
        seed: request.sampling.seed,
        cancel,
        // sc-20682: the worker-wide compressed-KV opt-in (off unless the operator enabled it).
        kv_compression: crate::llm_kv_cache::worker_default_policy(),
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
    job_id: &str,
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
    let generated = provider.generate_svg(request, cancel, progress, &mut |fragment, index| {
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
    });
    // sc-20682 / sc-20688: the decode's KV cache, as the job's `llm_kv_cache` telemetry event,
    // whichever way the decode ended — published, rejected at a terminal limit, refused at memory
    // admission, canceled or failed.
    if let Some(record) = provider.kv_cache_record() {
        record.emit(job_id);
    }
    generated?;
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
    // This worker-local product timer includes request decoding, source loading, generation,
    // sanitization, both renders, atomic publication, and sealed-artifact read-back. The route
    // measures submit-to-Completed separately because queue time and the completion update live
    // outside this function's observable interval.
    let product_started = Instant::now();
    let terminal_campaign = std::env::var("SCENEWORKS_TERMINAL_CAMPAIGN").as_deref() == Ok("1");
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
    let blocking_job_id = job.id.clone();
    let task = crate::load_progress::spawn_blocking(move || {
        collect_svg_source(
            &blocking_job_id,
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
                    product_started.elapsed().as_secs_f64(),
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
            let mut terminal_evidence = terminal_result(
                terminal,
                Some(product_started.elapsed().as_secs_f64()),
                None,
                None,
                None,
                Some((&transcript_path, &transcript)),
            );
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
    let comparison_path = staging.join("comparison-512.png");
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
        if terminal_campaign {
            render_preview_with_size(
                &canonical.svg,
                canonical.width,
                canonical.height,
                &comparison_path,
                Some(TERMINAL_COMPARISON_SIZE),
            )
            .await?;
        }
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

    let terminal_evidence = if terminal_campaign {
        let terminal = collected.terminal.as_ref().ok_or_else(|| {
            WorkerError::Engine(
                "terminal campaign native result lacks provider terminal evidence".to_owned(),
            )
        })?;
        let canonical_disk_path = published.join("vector.svg");
        let preview_disk_path = published.join("preview.png");
        let comparison_disk_path = published.join("comparison-512.png");
        let transcript_disk_path = published.join("provider-terminal.json");
        let canonical_bytes = tokio::fs::read(&canonical_disk_path).await?;
        let preview_bytes = tokio::fs::read(&preview_disk_path).await?;
        let comparison_bytes = tokio::fs::read(&comparison_disk_path).await?;
        let transcript_bytes = tokio::fs::read(&transcript_disk_path).await?;
        Some(add_source_raster_evidence(
            terminal_result(
                terminal,
                Some(product_started.elapsed().as_secs_f64()),
                Some((&canonical_disk_path, &canonical_bytes)),
                Some((&preview_disk_path, &preview_bytes)),
                Some((&comparison_disk_path, &comparison_bytes)),
                Some((&transcript_disk_path, &transcript_bytes)),
            ),
            source_raster
                .as_ref()
                .map(|(path, bytes)| (path.as_path(), bytes.as_slice())),
        )?)
    } else {
        None
    };

    let fact = build_vector_asset_fact(
        &asset_id,
        &generation_set_id,
        canonical.width,
        canonical.height,
        &created_at,
        payload.mode.as_str(),
        &payload.model,
        provider.provider_id(),
        &payload.prompt,
        payload.source_asset_id.as_deref(),
        json!(payload.sampling),
        json!(payload.detail_budget),
        payload.workflow.clone(),
    );
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
    crate::load_progress::spawn_blocking(move || {
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
#[path = "vector_jobs_tests.rs"]
mod tests;
