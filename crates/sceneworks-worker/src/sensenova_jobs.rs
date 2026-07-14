//! SenseNova-U1 understanding + interleave jobs (epic 3180, sc-3905 — Path B).
//!
//! These are the two SenseNova-U1 modes the mlx-gen `Generator` contract can't express
//! (`GenerationOutput` is Images/Video only), so they bypass the registry and call the public
//! [`T2iModel`](runtime_macos::providers::sensenova::T2iModel) methods directly:
//!
//! * **VQA** (`image_vqa`): one source image + a question → a text answer. No asset write.
//! * **Interleave / Document Studio** (`image_interleave`): a prompt (+ optional source images) →
//!   ordered text + generated images, persisted as image assets plus an
//!   [`InterleavedDocument`](sceneworks_core::contracts::InterleavedDocument) `document` asset.
//!
//! Each backend assembles its own concrete `T2iModel` from the provider's public re-exports (the
//! `Generator` crate has no public constructor): macOS drives `runtime_macos::providers::sensenova::T2iModel`; the
//! candle (Windows/CUDA) lane drives `runtime_cuda::providers::sensenova::T2iModel` via `load_understanding`
//! (sc-5501), retiring the off-Mac Python torch VQA/interleave path. The handler shape, request
//! parsing, and document assembly are shared and backend-neutral.
//!
//! Parity: VQA mirrors the Python `SenseNovaU1Adapter.answer_question`; interleave mirrors
//! `generate_interleaved` / `_write_interleaved_document` (request fields, the interleave resolution
//! buckets + think/no-think system protocol, and the response/asset shapes). The understanding +
//! generation model loads dense (no distill LoRA, no quantization) exactly as the torch adapter does
//! (`_load_model(distill_lora=None)`) — the full base model.
//!
//! When neither the MLX nor the candle engine is linked (a non-macOS build with `backend-candle`
//! off), the `image_vqa` / `image_interleave` capabilities are not advertised and the handlers are
//! stubs that error loudly (the Python torch worker serves these modes there).

use super::*;
// The macOS (MLX) and candle (Windows/CUDA) handlers parse an `ImageRequest`; the torch-defer stub
// doesn't.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
use sceneworks_core::image_request::ImageRequest;

// CARVE-OUT(epic 3720): backend-specific; absorbed by TextLlm in Phase 5.
// VQA + Document-Studio interleave bypass the `Generator` registry and drive the concrete unified
// `T2iModel` directly (text / text+images output the neutral `GenerationOutput` contract can't
// express). macOS drives `runtime_macos::providers::sensenova::T2iModel`; off-Mac the candle (Windows/CUDA) lane drives
// `runtime_cuda::providers::sensenova::T2iModel` (the sibling carve-outs, sc-5501) — retiring the off-Mac torch
// VQA/interleave path. Each lane keeps its own engine-typed imports; the document-assembly +
// request-parsing helpers below are backend-neutral and shared.
#[cfg(target_os = "macos")]
use mlx_rs::ops::divide;
#[cfg(target_os = "macos")]
use mlx_rs::Array;
#[cfg(target_os = "macos")]
use runtime_macos::media::image::{decoded_to_image, resize_bicubic_u8};
#[cfg(target_os = "macos")]
use runtime_macos::media::tokenizer::TextTokenizer;
#[cfg(target_os = "macos")]
use runtime_macos::media::Image;
#[cfg(target_os = "macos")]
use runtime_macos::providers::sensenova::{
    load_raw, load_tokenizer, smart_resize, NeoChatConfig, Sampler, T2iModel, T2iOptions,
    INTERLEAVE_RESOLUTIONS, INTERLEAVE_SYSTEM_MESSAGE,
};

// Candle (Windows/CUDA) understanding path (sc-5501). `Image` is the neutral `gen_core::Image`
// (`load_reference_image`'s return); the source-image preprocessing + tensor construction live in
// `image_to_chw01_candle` (the `image` crate's resampler + `runtime_cuda::media::candle_core`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use gen_core::Image;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::media::candle_core::{DType, Device, Tensor};
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::sensenova::{
    load_understanding, smart_resize, tensor_to_image, Sampler, T2iOptions, INTERLEAVE_RESOLUTIONS,
    INTERLEAVE_SYSTEM_MESSAGE,
};

/// The adapter id recorded on the generated assets + the interleaved document, matching the
/// per-backend `adapter_label` the image rows use: `mlx_sensenova` on macOS, `candle_sensenova` on
/// the candle lane (sc-5576).
#[cfg(target_os = "macos")]
const SENSENOVA_ADAPTER: &str = "mlx_sensenova";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const SENSENOVA_ADAPTER: &str = "candle_sensenova";

/// Terminal cancel messages for the two SenseNova understanding jobs. Shared between the
/// `run_blocking_with_heartbeat` keepalive (which posts them when the engine finishes AFTER the
/// flag tripped) and the engine-error mapping below (which posts them when the engine's per-token /
/// per-step cancel check surfaces the typed `Canceled` itself), so the terminal status message is
/// identical whichever side observes the trip first (sc-9123).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
const VQA_CANCEL_MESSAGE: &str = "Visual question answering canceled by user.";
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
const INTERLEAVE_CANCEL_MESSAGE: &str = "Interleaved generation canceled by user.";

/// Lift a SenseNova engine error into the worker's error space, preserving the TYPED cancellation:
/// the engine's per-token/per-step cancel check surfaces `Error::Canceled` when the keepalive trips
/// the threaded `CancelFlag` (sc-9123), and it must map to [`WorkerError::Canceled`] — not a
/// stringified `Engine` error — so `run_blocking_with_heartbeat` posts the terminal `Canceled`
/// instead of failing the job. Everything else keeps the old `Engine(context: error)` shape.
#[cfg(target_os = "macos")]
fn map_sensenova_engine_error(
    error: runtime_macos::media::Error,
    context: &str,
    cancel_message: &str,
) -> WorkerError {
    match error {
        runtime_macos::media::Error::Canceled => WorkerError::Canceled(cancel_message.to_owned()),
        other => WorkerError::Engine(format!("{context}: {other}")),
    }
}

/// Candle sibling of the macOS mapping above (same typed-cancel contract, `CandleError::Canceled`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn map_sensenova_engine_error(
    error: runtime_cuda::media::CandleError,
    context: &str,
    cancel_message: &str,
) -> WorkerError {
    match error {
        runtime_cuda::media::CandleError::Canceled => {
            WorkerError::Canceled(cancel_message.to_owned())
        }
        other => WorkerError::Engine(format!("{context}: {other}")),
    }
}

// ===========================================================================
// VQA (image_vqa)
// ===========================================================================

/// Visual question answering: a text answer about one source image. Mirrors the Python
/// `SenseNovaU1Adapter.answer_question` — same request fields, same `{answer, question,
/// sourceAssetId, model, realModelInference}` result, no asset write. The source image resolves
/// only through the project sidecar/DB (`load_reference_image`), so there is no client-supplied
/// path escape.
///
/// The async orchestration (request parse, progress/heartbeat posts, cancel plumbing, result
/// assembly) is backend-neutral and shared; the only per-backend piece is the blocking
/// load → preprocess → `vqa` → think-strip seam, injected as [`vqa_generate`] (one cfg-gated impl
/// per engine — sc-8839, the F-037 dedupe).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) async fn run_vqa_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let model_id = if request.model.trim().is_empty() {
        "sensenova_u1_8b".to_owned()
    } else {
        request.model.clone()
    };
    let question = job
        .payload
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Visual question answering requires a question.".to_owned())
        })?
        .to_owned();
    let source_asset_id = job
        .payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Visual question answering requires a source image asset.".to_owned(),
            )
        })?
        .to_owned();

    // VQA latency ~ output tokens + input vision tokens; both default low and are tunable per
    // request (top-level payload, mirroring the Python adapter — NOT under `advanced`).
    let max_new_tokens = payload_int(&job.payload, "maxNewTokens", 256, 16, 2048) as usize;
    let max_image_pixels = payload_int(
        &job.payload,
        "maxImagePixels",
        768 * 768,
        256 * 256,
        2048 * 2048,
    );

    let weights_dir = resolve_weights_dir(&request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SenseNova-U1 weights not found".to_owned()))?;

    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let backend = backend_label(&settings.gpu_id).to_owned();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing visual question.",
            None,
            &backend,
        ),
    )
    .await?;

    // Decode the source image on the async side (Send `Image` moves into the blocking task).
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &source_asset_id,
        &project_path,
    )?;

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.6,
            "Analyzing image.",
            None,
            &backend,
        ),
    )
    .await?;

    let job_id = job.id.clone();
    let question_for_vqa = question.clone();
    // Keep the worker heartbeat alive across the blocking VLM load + generation (a cold
    // SenseNova-U1 8B load + long answer easily exceeds the API's 90s stale-sweep) so the in-flight
    // job is never falsely marked `interrupted` (sc-8390). The engine checks the threaded
    // `CancelFlag` before each decoded token (mlx-gen #634 / its candle sibling), so the keepalive's
    // cancel poll actually STOPS a multi-minute answer mid-rollout instead of only waiting for it
    // (sc-9123, the sc-8804 F-003 residual).
    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    let answer = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        VQA_CANCEL_MESSAGE,
        "VQA task join",
        crate::no_cancel_ack(),
        tokio::task::spawn_blocking(move || -> WorkerResult<String> {
            vqa_generate(
                &weights_dir,
                &source,
                &question_for_vqa,
                max_new_tokens,
                max_image_pixels,
                &job_id,
                &task_cancel,
            )
        }),
    )
    .await?;

    let result = vqa_result_json(&answer, &question, &source_asset_id, &model_id);

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Answer ready.",
            Some(result),
            &backend,
        ),
    )
    .await?;
    Ok(())
}

/// macOS (MLX) VQA seam: the blocking load → preprocess → `vqa` → think-strip body run inside the
/// shared handler's `spawn_blocking`. Builds the dense `runtime_macos::providers::sensenova::T2iModel` via
/// [`load_sensenova_model`] and calls `T2iModel::vqa`. The candle sibling below mirrors it exactly
/// bar the engine-typed load/preprocess (sc-8839).
#[cfg(target_os = "macos")]
fn vqa_generate(
    weights_dir: &Path,
    source: &Image,
    question: &str,
    max_new_tokens: usize,
    max_image_pixels: i64,
    job_id: &str,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<String> {
    emit_load_event("image_pipeline_load_start", job_id, "sensenova_u1_8b", 0);
    let (model, tokenizer) = load_sensenova_model(weights_dir)?;
    emit_load_event("image_pipeline_load_complete", job_id, "sensenova_u1_8b", 0);
    // ImageNet-normalized inside `vqa`; pass [3,H,W] in [0,1], 32-aligned, within the understanding
    // pixel budget (default 768², `load_image_native` min 256²).
    let pixel_values = image_to_chw01(source, 256 * 256, max_image_pixels)?;
    let answer = model
        .vqa(
            &tokenizer,
            question,
            std::slice::from_ref(&pixel_values),
            max_new_tokens,
            Sampler::Greedy,
            Some(cancel),
        )
        .map_err(|error| {
            map_sensenova_engine_error(error, "SenseNova VQA failed", VQA_CANCEL_MESSAGE)
        })?;
    Ok(strip_reasoning(&answer))
}

/// Candle (Windows/CUDA) VQA seam — the off-Mac sibling of [`vqa_generate`] (sc-5501). Builds the
/// dense `runtime_cuda::providers::sensenova::T2iModel` via `load_understanding` and calls `T2iModel::vqa`,
/// retiring the Python torch `image_vqa` path off-Mac. Same shape as the macOS seam; only the
/// engine-typed load/preprocess differ.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn vqa_generate(
    weights_dir: &Path,
    source: &Image,
    question: &str,
    max_new_tokens: usize,
    max_image_pixels: i64,
    job_id: &str,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<String> {
    emit_load_event("image_pipeline_load_start", job_id, "sensenova_u1_8b", 0);
    let (model, tokenizer) = load_understanding(weights_dir)
        .map_err(|error| WorkerError::Engine(format!("SenseNova-U1 load: {error}")))?;
    emit_load_event("image_pipeline_load_complete", job_id, "sensenova_u1_8b", 0);
    // ImageNet-normalized inside `vqa`; pass [3,H,W] in [0,1], 32-aligned, within the understanding
    // pixel budget (default 768², min 256²).
    let pixel_values = image_to_chw01_candle(source, 256 * 256, max_image_pixels)?;
    let answer = model
        .vqa(
            &tokenizer,
            question,
            std::slice::from_ref(&pixel_values),
            max_new_tokens,
            Sampler::Greedy,
            Some(cancel),
        )
        .map_err(|error| {
            map_sensenova_engine_error(error, "SenseNova VQA failed", VQA_CANCEL_MESSAGE)
        })?;
    Ok(strip_reasoning(&answer))
}

/// Off macOS without the candle backend the in-process engine is unavailable; `image_vqa` is served
/// by the Python torch worker (macOS runs the MLX engine; Windows/CUDA runs candle when enabled).
#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
pub(crate) async fn run_vqa_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "image_vqa runs on the macOS MLX worker, the candle (Windows/CUDA) worker, not this worker"
            .to_owned(),
    ))
}

// ===========================================================================
// Interleave / Document Studio (image_interleave)
// ===========================================================================

/// The resolved interleave rollout knobs threaded from the shared handler into the per-backend
/// [`interleave_generate`] seam. Bundled into one struct so the seam stays a small arity and both
/// engine impls read the exact same resolved values (sc-8839, the F-037 dedupe). The pure
/// [`resolve_interleave_params`] resolver is unit-tested on the macOS lane.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
#[derive(Debug, PartialEq)]
struct InterleaveParams {
    /// Snapped bucket width (a 32-aligned interleave-resolution bucket).
    width: i32,
    /// Snapped bucket height.
    height: i32,
    steps: usize,
    cfg_scale: f32,
    img_cfg_scale: f32,
    timestep_shift: f32,
    max_new_tokens: usize,
    max_images: usize,
    think_mode: bool,
    /// Interleave system protocol message (think/no-think), resolved to the request override else
    /// [`INTERLEAVE_SYSTEM_MESSAGE`].
    system_message: String,
    seed: i64,
}

/// Resolve every interleave rollout knob from the request + top-level payload: snap the requested
/// W×H to the nearest [`INTERLEAVE_RESOLUTIONS`] bucket, overlay the `advanced` defaults/overrides
/// (upstream `examples/interleave/inference.py` @238d6cf), and pick the think/no-think system
/// message. Pure + backend-neutral (the shared handler's brain), so both engine seams see identical
/// values and it can be unit-tested without weights or a device (sc-8839).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn resolve_interleave_params(request: &ImageRequest, payload: &JsonObject) -> InterleaveParams {
    let advanced = &request.advanced;
    // Snap the requested W×H to the nearest interleave bucket by aspect ratio (log-space), mirroring
    // the Python `interleave_resolution_for`. Defaults 2048×1152 (16:9), clamped 256..4096.
    let req_width = payload_int(payload, "width", 2048, 256, 4096);
    let req_height = payload_int(payload, "height", 1152, 256, 4096);
    let (width, height) = interleave_resolution_snap(req_width, req_height);
    InterleaveParams {
        width,
        height,
        // Upstream interleave defaults (examples/interleave/inference.py @238d6cf).
        steps: advanced_int(advanced, "numInferenceSteps", 50, 1, 100) as usize,
        cfg_scale: advanced_float(advanced, "guidanceScale", 4.0),
        img_cfg_scale: advanced_float(advanced, "imageGuidanceScale", 1.0),
        timestep_shift: advanced_float(advanced, "timestepShift", 3.0),
        max_new_tokens: advanced_int(advanced, "maxNewTokens", 2048, 64, 8192) as usize,
        max_images: advanced_int(advanced, "maxImages", 6, 1, 10) as usize,
        // Non-Think by default: the document is the deliverable, so skip the chain-of-thought.
        think_mode: advanced
            .get("thinkMode")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        system_message: advanced
            .get("systemMessage")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| INTERLEAVE_SYSTEM_MESSAGE.to_owned()),
        seed: resolve_seed(request, 0),
    }
}

/// Interleaved text-image generation: one model rollout yields ordered text + images, persisted as
/// a `document` asset whose segments reference the generated image assets in order. Mirrors the
/// Python `generate_interleaved` → `_write_interleaved_document` contract (request fields, resolution
/// buckets, think/no-think protocol, asset/result shapes). The base understanding+generation model
/// loads dense (no distill LoRA) — interleave needs the full model, never the distilled gen LoRA.
///
/// The async orchestration (request parse, resolution snap, progress/heartbeat posts, pre-rollout
/// cancel, document-write hand-off) is backend-neutral and shared; the only per-backend piece is the
/// blocking load → preprocess → `interleave_gen` → decode seam, injected as [`interleave_generate`]
/// (one cfg-gated impl per engine — sc-8839, the F-037 dedupe).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) async fn run_interleave_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let prompt = request.prompt.trim().to_owned();
    if prompt.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Interleaved generation requires a prompt.".to_owned(),
        ));
    }
    let model_id = if request.model.trim().is_empty() {
        "sensenova_u1_8b".to_owned()
    } else {
        request.model.clone()
    };
    // Resolve every rollout knob (resolution snap + advanced overlay defaults/overrides) once, so
    // both backend seams read the exact same values (sc-8839). Pure + backend-neutral → unit-tested.
    let params = resolve_interleave_params(&request, &job.payload);

    let source_asset_ids: Vec<String> = job
        .payload
        .get("sourceAssetIds")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    let weights_dir = resolve_weights_dir(&request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SenseNova-U1 weights not found".to_owned()))?;

    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    tokio::fs::create_dir_all(project_path.join("assets").join("documents")).await?;
    tokio::fs::create_dir_all(project_path.join("assets").join("images")).await?;
    let backend = backend_label(&settings.gpu_id).to_owned();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing interleaved document.",
            None,
            &backend,
        ),
    )
    .await?;

    // Decode the optional source images on the async side (Send moves into the blocking task).
    let mut input_images = Vec::with_capacity(source_asset_ids.len());
    for asset_id in &source_asset_ids {
        input_images.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            asset_id,
            &project_path,
        )?);
    }

    // Early-exit on a cancel that arrived before the rollout even started (the mid-rollout case is
    // covered by the CancelFlag threaded into `interleave_gen` below, sc-9123). `check_cancel`
    // marks the job canceled + returns `Canceled` on a cancel; a real API error still propagates.
    match check_cancel(api, &job.id, INTERLEAVE_CANCEL_MESSAGE).await {
        Ok(()) => {}
        Err(WorkerError::Canceled(_)) => return Ok(()),
        Err(other) => return Err(other),
    }

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.45,
            "Composing interleaved document.",
            None,
            &backend,
        ),
    )
    .await?;
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;

    // The engine's `interleave_gen` is a single synchronous rollout with no per-segment callback,
    // so (like the Python adapter's single `interleave_gen` call) the document streams as one final
    // result rather than incrementally. The whole rollout runs on a blocking thread; the decoded
    // images come back as Send `Image`s for asset writing on the async side. The only per-backend
    // piece — load → preprocess → `interleave_gen` → decode — is the `interleave_generate` seam.
    let job_id = job.id.clone();
    let prompt_for_gen = prompt.clone();
    // The document-write hand-off below needs these resolved scalars after `params` moves into the
    // rollout closure; snapshot the `Copy` fields now (`system_message` is rollout-only, so `params`
    // itself still moves whole into the seam closure).
    let (
        width,
        height,
        steps,
        cfg_scale,
        img_cfg_scale,
        timestep_shift,
        max_new_tokens,
        max_images,
        think_mode,
        seed,
    ) = (
        params.width,
        params.height,
        params.steps,
        params.cfg_scale,
        params.img_cfg_scale,
        params.timestep_shift,
        params.max_new_tokens,
        params.max_images,
        params.think_mode,
        params.seed,
    );
    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    let interleave_task =
        tokio::task::spawn_blocking(move || -> WorkerResult<(String, Vec<Image>)> {
            interleave_generate(
                &weights_dir,
                &input_images,
                &prompt_for_gen,
                &params,
                &job_id,
                &task_cancel,
            )
        });
    // Keep the worker heartbeat alive across the blocking VLM load + interleave rollout (cold 8B
    // load + multi-image generation easily exceeds the API's 90s stale-sweep) so the in-flight job
    // is never falsely marked `interrupted` (sc-8390). Cancelable mid-rollout via the threaded
    // flag (sc-9123).
    let (generated_text, images) = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        INTERLEAVE_CANCEL_MESSAGE,
        "interleave task join",
        crate::no_cancel_ack(),
        interleave_task,
    )
    .await?;

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.9,
            "Saving interleaved document.",
            None,
            &backend,
        ),
    )
    .await?;

    // The document assembly synchronously PNG-encodes up to `max_images` multi-megapixel images
    // (`write_image_asset`) and does the document `fs::write`/`rename` — multi-second work that must
    // not run on the async runtime thread. Move it onto the blocking pool under the heartbeat wrapper
    // so the in-flight job keeps beating across the encode + IO and is never falsely swept as
    // `interrupted` during those seconds (sc-8838, sc-8390). Ownership is moved into the closure.
    // Cancel flag stays `None` (sc-9123 decision): this is bounded CPU encode + filesystem IO with
    // no loop worth interrupting, and aborting a half-written document asset mid-rename is worse
    // than letting the seconds-long write finish.
    let job_owned = job.clone();
    let write_task = tokio::task::spawn_blocking(move || {
        write_interleaved_document(
            request,
            job_owned,
            project_path,
            prompt,
            model_id,
            seed,
            max_images,
            width,
            height,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            max_new_tokens,
            think_mode,
            &generated_text,
            images,
        )
    });
    let result = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        None,
        "",
        "interleave document write join",
        crate::no_cancel_ack(),
        write_task,
    )
    .await?;

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Interleaved document ready.",
            Some(result),
            &backend,
        ),
    )
    .await?;
    Ok(())
}

/// macOS (MLX) interleave seam: the blocking load → preprocess → `interleave_gen` → decode body run
/// inside the shared handler's `spawn_blocking`. Builds the dense `runtime_macos::providers::sensenova::T2iModel` via
/// [`load_sensenova_model`] and calls `T2iModel::interleave_gen`, decoding each model-space image
/// with `decoded_to_image`. The engine's `interleave_gen` takes `width`/`height` as `i32`, an
/// `init_noises: None`, and a no-op `on_progress` sink — arg shapes intrinsic to the MLX API, not
/// shared with the candle sibling (sc-8839).
#[cfg(target_os = "macos")]
fn interleave_generate(
    weights_dir: &Path,
    input_images: &[Image],
    prompt: &str,
    params: &InterleaveParams,
    job_id: &str,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<(String, Vec<Image>)> {
    emit_load_event("image_pipeline_load_start", job_id, "sensenova_u1_8b", 0);
    let (model, tokenizer) = load_sensenova_model(weights_dir)?;
    emit_load_event("image_pipeline_load_complete", job_id, "sensenova_u1_8b", 0);

    // Source images: [3,H,W] in [0,1], 32-aligned. Bounds mirror the torch `interleave_gen`
    // (`load_image_native` min 512², max min(2048², 4096²/n)).
    let n = input_images.len().max(1) as i64;
    let max_pixels = (2048 * 2048).min((4096 * 4096) / n);
    let mut input_arrays = Vec::with_capacity(input_images.len());
    for image in input_images {
        input_arrays.push(image_to_chw01(image, 512 * 512, max_pixels)?);
    }

    let opts = T2iOptions {
        cfg_scale: params.cfg_scale,
        img_cfg_scale: params.img_cfg_scale,
        num_steps: params.steps,
        timestep_shift: params.timestep_shift,
        seed: params.seed as u64,
        think_mode: params.think_mode,
        ..Default::default()
    };
    // The engine checks the threaded flag per decoded text token and per denoise step (mlx-gen
    // #634), so the keepalive's cancel poll stops a multi-minute rollout cooperatively (sc-9123).
    // Progress stays a no-op sink: this seam reports liveness via the Busy heartbeat, not per-step
    // job progress.
    let out = model
        .interleave_gen(
            &tokenizer,
            prompt,
            &input_arrays,
            params.width,
            params.height,
            &opts,
            &params.system_message,
            params.max_new_tokens,
            params.max_images,
            None,
            cancel,
            &mut |_| {},
        )
        .map_err(|error| {
            map_sensenova_engine_error(
                error,
                "SenseNova interleave failed",
                INTERLEAVE_CANCEL_MESSAGE,
            )
        })?;
    // The generated images are model-space [-1,1] `[1,3,H,W]` arrays — decode each to RGB8 exactly
    // as the `Generator` image path does (`decoded_to_image`).
    let mut decoded = Vec::with_capacity(out.images.len());
    for image in &out.images {
        decoded.push(decoded_to_image(image).map_err(|error| {
            WorkerError::InvalidPayload(format!("SenseNova interleave decode: {error}"))
        })?);
    }
    Ok((out.text, decoded))
}

/// Candle (Windows/CUDA) interleave seam — the off-Mac sibling of [`interleave_generate`] (sc-5501).
/// Builds the dense `runtime_cuda::providers::sensenova::T2iModel` via `load_understanding` and calls
/// `T2iModel::interleave_gen`, decoding each model-space tensor with `tensor_to_image`, retiring the
/// Python torch `image_interleave` path off-Mac. The candle engine's `interleave_gen` takes
/// `width`/`height` as `usize` and no `init_noises`/`on_progress` args — arg shapes intrinsic to the
/// candle API, not shared with the MLX sibling (sc-8839).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn interleave_generate(
    weights_dir: &Path,
    input_images: &[Image],
    prompt: &str,
    params: &InterleaveParams,
    job_id: &str,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<(String, Vec<Image>)> {
    emit_load_event("image_pipeline_load_start", job_id, "sensenova_u1_8b", 0);
    let (model, tokenizer) = load_understanding(weights_dir)
        .map_err(|error| WorkerError::Engine(format!("SenseNova-U1 load: {error}")))?;
    emit_load_event("image_pipeline_load_complete", job_id, "sensenova_u1_8b", 0);

    // Source images: [3,H,W] in [0,1], 32-aligned. Bounds mirror the torch `interleave_gen`
    // (`load_image_native` min 512², max min(2048², 4096²/n)).
    let n = input_images.len().max(1) as i64;
    let max_pixels = (2048 * 2048).min((4096 * 4096) / n);
    let mut input_arrays = Vec::with_capacity(input_images.len());
    for image in input_images {
        input_arrays.push(image_to_chw01_candle(image, 512 * 512, max_pixels)?);
    }

    let opts = T2iOptions {
        cfg_scale: params.cfg_scale,
        img_cfg_scale: params.img_cfg_scale,
        num_steps: params.steps,
        timestep_shift: params.timestep_shift,
        seed: params.seed as u64,
        think_mode: params.think_mode,
        ..Default::default()
    };
    // The engine polls the threaded flag between text tokens / denoise steps, and the keepalive's
    // cancel poll trips it (sc-9123).
    let out = model
        .interleave_gen(
            &tokenizer,
            prompt,
            &input_arrays,
            params.width as usize,
            params.height as usize,
            &opts,
            &params.system_message,
            params.max_new_tokens,
            params.max_images,
            cancel,
        )
        .map_err(|error| {
            map_sensenova_engine_error(
                error,
                "SenseNova interleave failed",
                INTERLEAVE_CANCEL_MESSAGE,
            )
        })?;
    // The generated images are model-space [-1,1] `[1,3,H,W]` tensors — decode each to RGB8.
    let mut decoded = Vec::with_capacity(out.images.len());
    for image in &out.images {
        decoded.push(tensor_to_image(image).map_err(|error| {
            WorkerError::InvalidPayload(format!("SenseNova interleave decode: {error}"))
        })?);
    }
    Ok((out.text, decoded))
}

/// Off macOS without the candle backend the in-process engine is unavailable; `image_interleave` is
/// served by the Python torch worker (macOS runs MLX; Windows/CUDA runs candle when enabled).
#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
pub(crate) async fn run_interleave_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "image_interleave runs on the macOS MLX worker, the candle (Windows/CUDA) worker, not this worker"
            .to_owned(),
    ))
}

// ---------------------------------------------------------------------------
// Document assembly (macOS): write the generated images as ordinary image assets, split the model
// text on `<image>` markers into ordered segments, then write the `InterleavedDocument` body + the
// `document` asset fact. Mirrors the Python `_write_interleaved_document` / `_build_interleaved_segments`.
// ---------------------------------------------------------------------------

// Runs the sync PNG encodes (up to 10 multi-megapixel images via `write_image_asset`) plus the
// document `fs::write`/`rename`; all call sites hand it to `spawn_blocking` under
// `run_blocking_with_heartbeat`, so it must own its inputs (no borrows into the async frame) and
// never touch the runtime thread itself (sc-8838).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
#[allow(clippy::too_many_arguments)]
fn write_interleaved_document(
    request: ImageRequest,
    job: JobSnapshot,
    project_path: PathBuf,
    prompt: String,
    model_id: String,
    seed: i64,
    max_images: usize,
    width: i32,
    height: i32,
    steps: usize,
    cfg_scale: f32,
    img_cfg_scale: f32,
    timestep_shift: f32,
    max_new_tokens: usize,
    think_mode: bool,
    generated_text: &str,
    images: Vec<Image>,
) -> WorkerResult<JsonObject> {
    let resolution = format!("{width}x{height}");
    // Flat telemetry mirroring the Python `raw_settings` (advanced overlay + resolved knobs).
    let mut raw_settings = request.advanced.clone();
    raw_settings.insert("realModelInference".to_owned(), Value::Bool(true));
    raw_settings.insert("repo".to_owned(), Value::String(model_repo_for(&request)));
    raw_settings.insert("numInferenceSteps".to_owned(), json!(steps));
    raw_settings.insert("guidanceScale".to_owned(), json!(cfg_scale));
    raw_settings.insert("imageGuidanceScale".to_owned(), json!(img_cfg_scale));
    raw_settings.insert("timestepShift".to_owned(), json!(timestep_shift));
    raw_settings.insert("maxImages".to_owned(), json!(max_images));
    raw_settings.insert("maxNewTokens".to_owned(), json!(max_new_tokens));
    raw_settings.insert("thinkMode".to_owned(), Value::Bool(think_mode));
    raw_settings.insert("resolution".to_owned(), Value::String(resolution.clone()));

    // Generated images persist as ordinary image assets — the worker saves the PNG + reports facts,
    // and the Rust API builds + indexes their sidecars. The document references them in order.
    let plan = ImagePlan::with_count(&request, images.len() as u32);
    let mut image_raw_settings = raw_settings.clone();
    image_raw_settings.insert("interleaved".to_owned(), Value::Bool(true));
    let mut image_writes: Vec<Value> = Vec::with_capacity(images.len());
    for (index, image) in images.into_iter().enumerate() {
        let fact = write_image_asset(
            &plan,
            index,
            seed,
            image.width,
            image.height,
            image.pixels,
            SENSENOVA_ADAPTER,
            image_raw_settings.clone(),
            &project_path,
        )?;
        image_writes.push(Value::Object(fact));
    }
    let image_asset_ids: Vec<String> = image_writes
        .iter()
        .filter_map(|write| write.get("assetId").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();

    let segments = build_interleaved_segments(generated_text, &image_writes);

    let created_at = crate::now_rfc3339();
    let document_id = format!("doc_{}", Uuid::new_v4().simple());
    let media_rel = format!("assets/documents/{document_id}.json");
    let document_body = json!({
        "schemaVersion": 1,
        "id": document_id,
        "projectId": request.project_id,
        "jobId": job.id,
        "model": model_id,
        "prompt": prompt,
        "createdAt": created_at,
        "segments": segments,
    });
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = media_path.with_extension("tmp.json");
    std::fs::write(
        &temp_path,
        serde_json::to_vec_pretty(&document_body)
            .map_err(|error| WorkerError::InvalidPayload(format!("serialize document: {error}")))?,
    )?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let display_name: String = {
        let trimmed: String = prompt.chars().take(56).collect();
        if trimmed.trim().is_empty() {
            "Interleaved document".to_owned()
        } else {
            trimmed
        }
    };
    let asset_id = fresh_asset_id();
    let document_write = json!({
        "type": "document",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "application/json",
        "displayName": display_name,
        "createdAt": created_at,
        "mode": "interleave",
        "model": model_id,
        "adapter": SENSENOVA_ADAPTER,
        "prompt": prompt,
        "negativePrompt": "",
        "seed": seed,
        "loras": [],
        "rawAdapterSettings": raw_settings,
        "maxImages": max_images,
        "resolution": resolution,
        "imageCount": image_asset_ids.len(),
        "parents": image_asset_ids,
    });

    let mut asset_writes = image_writes;
    asset_writes.push(document_write);
    let expected_count = asset_writes.len();

    Ok(json!({
        "documentId": document_id,
        "documentAssetId": asset_id,
        "imageAssetIds": image_asset_ids,
        "segments": segments,
        "model": model_id,
        "realModelInference": true,
        "generationSetId": plan.genset_id,
        "expectedCount": expected_count,
        "generationSet": plan.generation_set,
        "assetWrites": asset_writes,
    })
    .as_object()
    .cloned()
    .expect("json! object literal"))
}

/// Split the model output on its inline `<image>` markers and slot the generated image assets in
/// order: text[0], image[0], text[1], image[1], …. Mirrors the Python `_build_interleaved_segments`
/// (reads each image fact's `assetId` + `mediaPath`).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn build_interleaved_segments(generated_text: &str, image_writes: &[Value]) -> Vec<Value> {
    let mut segments = Vec::new();
    for (index, part) in generated_text.split("<image>").enumerate() {
        let text = part.trim();
        if !text.is_empty() {
            segments.push(json!({ "type": "text", "text": text }));
        }
        if let Some(write) = image_writes.get(index) {
            segments.push(json!({
                "type": "image",
                "assetId": write.get("assetId").cloned().unwrap_or(Value::Null),
                "path": write.get("mediaPath").cloned().unwrap_or(Value::Null),
            }));
        }
    }
    segments
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Assemble the VQA `{answer, question, sourceAssetId, model, realModelInference}` result object —
/// the shared backend-neutral response shape both engine handlers post (mirrors the Python
/// `SenseNovaU1Adapter.answer_question` result). Pure, so it is unit-tested cross-platform (sc-8839).
#[cfg(any(target_os = "macos", test, feature = "backend-candle"))]
fn vqa_result_json(
    answer: &str,
    question: &str,
    source_asset_id: &str,
    model_id: &str,
) -> JsonObject {
    json!({
        "answer": answer,
        "question": question,
        "sourceAssetId": source_asset_id,
        "model": model_id,
        "realModelInference": true,
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Assemble the concrete unified `T2iModel` + tokenizer for a SenseNova-U1 snapshot, replicating the
/// engine's private `load_inner` from public re-exports. Loads dense bf16 with NO distill LoRA and
/// NO quantization — the understanding (VQA) + interleave paths use the full base model, exactly as
/// the torch adapter's `_load_model(distill_lora=None)`, keeping the VQA decode bit-identical.
#[cfg(target_os = "macos")]
fn load_sensenova_model(weights_dir: &Path) -> WorkerResult<(T2iModel, TextTokenizer)> {
    let cfg = NeoChatConfig::from_dir(weights_dir)
        .map_err(|error| WorkerError::InvalidPayload(format!("SenseNova-U1 config: {error}")))?;
    let weights = load_raw(weights_dir)
        .map_err(|error| WorkerError::InvalidPayload(format!("SenseNova-U1 weights: {error}")))?;
    let model = T2iModel::from_weights(&weights, &cfg).map_err(|error| {
        WorkerError::InvalidPayload(format!("SenseNova-U1 model build: {error}"))
    })?;
    let tokenizer = load_tokenizer(weights_dir)
        .map_err(|error| WorkerError::InvalidPayload(format!("SenseNova-U1 tokenizer: {error}")))?;
    Ok((model, tokenizer))
}

/// Decode an [`Image`] (RGB8 HWC) to a `[3,H,W]` f32 tensor in `[0,1]`, smart-resized to a
/// 32-aligned bucket within `[min_pixels, max_pixels]`. Replicates the engine's private
/// `image_to_chw01` (its `preprocess_image` ImageNet-normalizes internally, so this stays in
/// `[0,1]`). VQA passes the understanding budget; interleave passes the it2i source budget.
#[cfg(target_os = "macos")]
fn image_to_chw01(img: &Image, min_pixels: i64, max_pixels: i64) -> WorkerResult<Array> {
    let (in_w, in_h) = (img.width as i32, img.height as i32);
    let (out_h, out_w) = smart_resize(in_h, in_w, 32, min_pixels, max_pixels);
    // gen-core drift (sc-9940): imageops::resize_*_u8 became fallible.
    let hwc = resize_bicubic_u8(
        &img.pixels,
        in_h as usize,
        in_w as usize,
        out_h as usize,
        out_w as usize,
    )
    .map_err(|error| WorkerError::InvalidPayload(format!("image resize: {error}")))?;
    let hwc = Array::from_slice(&hwc, &[out_h, out_w, 3]);
    let chw = hwc
        .transpose_axes(&[2, 0, 1])
        .map_err(|error| WorkerError::InvalidPayload(format!("image transpose: {error}")))?;
    divide(&chw, Array::from_f32(255.0))
        .map_err(|error| WorkerError::InvalidPayload(format!("image normalize: {error}")))
}

/// Candle sibling of [`image_to_chw01`] (sc-5501): decode an [`Image`] (RGB8 HWC) to a `[3,H,W]` f32
/// candle `Tensor` in `[0,1]`, smart-resized (Lanczos3, the worker's house resampler) to a 32-aligned
/// bucket within `[min_pixels, max_pixels]`. The understanding `T2iModel::{vqa, interleave_gen}`
/// ImageNet-normalize internally, so this stays in `[0,1]`. Built on **CPU**; the engine relocates it
/// to the model's device (candle treats each `new_cuda(0)` handle as distinct, so building it on the
/// worker's own device handle would mismatch the model's).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn image_to_chw01_candle(img: &Image, min_pixels: i64, max_pixels: i64) -> WorkerResult<Tensor> {
    let (out_h, out_w) = smart_resize(
        img.height as i32,
        img.width as i32,
        32,
        min_pixels,
        max_pixels,
    );
    let rgb =
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone()).ok_or_else(|| {
            WorkerError::InvalidPayload("reference image buffer size mismatch".to_owned())
        })?;
    let resized = image::imageops::resize(
        &rgb,
        out_w as u32,
        out_h as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let (rw, rh) = (resized.width() as usize, resized.height() as usize);
    // [H,W,3] u8 -> [3,H,W] f32 in [0,1], on CPU (the engine moves it to the model device).
    let hwc = Tensor::from_vec(resized.into_raw(), (rh, rw, 3), &Device::Cpu)
        .and_then(|tensor| tensor.to_dtype(DType::F32))
        .map_err(|error| WorkerError::Engine(format!("image tensor: {error}")))?;
    hwc.permute((2, 0, 1))
        .and_then(|chw| chw / 255.0)
        .map_err(|error| WorkerError::Engine(format!("image normalize: {error}")))
}

/// The SenseNova-U1 repo (manifest `repo` else the default), for the document telemetry.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn model_repo_for(request: &ImageRequest) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("sensenova/SenseNova-U1-8B-MoT")
        .to_owned()
}

/// Snap a requested W×H to the nearest interleave bucket by aspect ratio in log-space (ties resolve
/// to the first bucket). Mirrors the Python `snap_to_aspect_bucket` over the same
/// `INTERLEAVE_RESOLUTIONS` table (priority order).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn interleave_resolution_snap(width: i64, height: i64) -> (i32, i32) {
    let target = (width.max(1) as f64 / height.max(1) as f64).ln();
    let mut best = INTERLEAVE_RESOLUTIONS[0].1;
    let mut best_distance = f64::INFINITY;
    for &(_, (bucket_w, bucket_h)) in INTERLEAVE_RESOLUTIONS {
        let distance = (target - (bucket_w as f64 / bucket_h as f64).ln()).abs();
        if distance < best_distance {
            best_distance = distance;
            best = (bucket_w, bucket_h);
        }
    }
    best
}

/// Drop any `<think>…</think>` reasoning so only the answer is returned — removes complete think
/// blocks and any dangling/unclosed one (reasoning truncated by `max_new_tokens`). Mirrors the
/// Python `SenseNovaU1Adapter._strip_reasoning`. Used by the macOS VQA handler; also unit-tested
/// cross-platform (it is pure string logic), so it compiles under `test` off macOS too.
#[cfg(any(target_os = "macos", test, feature = "backend-candle"))]
fn strip_reasoning(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            // Complete block: drop `<think>…</think>` and continue after it.
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            // Dangling/unclosed block (truncated by max_new_tokens): drop everything from
            // `<think>` on.
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_owned()
}

/// `safe_int` over a top-level payload field: parse (int / float / numeric string) else `default`,
/// then clamp to `[lo, hi]`. Used by the macOS handlers; also unit-tested cross-platform.
#[cfg(any(target_os = "macos", test, feature = "backend-candle"))]
fn payload_int(payload: &JsonObject, key: &str, default: i64, lo: i64, hi: i64) -> i64 {
    payload
        .get(key)
        .and_then(json_to_i64)
        .unwrap_or(default)
        .clamp(lo, hi)
}

/// `safe_int` over an `advanced` field (same parse/clamp as [`payload_int`]).
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn advanced_int(advanced: &JsonObject, key: &str, default: i64, lo: i64, hi: i64) -> i64 {
    payload_int(advanced, key, default, lo, hi)
}

/// `_advanced_float`: parse an `advanced` field as f32 else `default`.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn advanced_float(advanced: &JsonObject, key: &str, default: f32) -> f32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
}

#[cfg(any(target_os = "macos", test, feature = "backend-candle"))]
fn json_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_f64().map(|float| float as i64))
        .or_else(|| value.as_str()?.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-9123: the engine's per-token/per-step cancel check surfaces the TYPED `Canceled`, and the
    /// worker mapping must preserve it as `WorkerError::Canceled` (with the shared terminal message)
    /// so `run_blocking_with_heartbeat` posts the terminal `Canceled` instead of failing the job.
    /// Any other engine error keeps the old `Engine("context: error")` shape.
    #[cfg(target_os = "macos")]
    #[test]
    fn engine_cancel_maps_to_worker_canceled_with_terminal_message() {
        let mapped = map_sensenova_engine_error(
            runtime_macos::media::Error::Canceled,
            "SenseNova VQA failed",
            VQA_CANCEL_MESSAGE,
        );
        assert!(
            matches!(mapped, WorkerError::Canceled(ref m) if m == VQA_CANCEL_MESSAGE),
            "typed engine Canceled must map to WorkerError::Canceled, got {mapped:?}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn engine_failure_keeps_the_engine_error_shape() {
        let mapped = map_sensenova_engine_error(
            runtime_macos::media::Error::Msg("boom".to_owned()),
            "SenseNova interleave failed",
            INTERLEAVE_CANCEL_MESSAGE,
        );
        assert!(
            matches!(mapped, WorkerError::Engine(ref m) if m == "SenseNova interleave failed: boom"),
            "non-cancel engine errors must stay Engine(context: error), got {mapped:?}"
        );
    }

    /// Candle-lane sibling of the two mappings above (`CandleError::Canceled` / `CandleError::Msg`).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn engine_cancel_maps_to_worker_canceled_with_terminal_message_candle() {
        let mapped = map_sensenova_engine_error(
            runtime_cuda::media::CandleError::Canceled,
            "SenseNova VQA failed",
            VQA_CANCEL_MESSAGE,
        );
        assert!(
            matches!(mapped, WorkerError::Canceled(ref m) if m == VQA_CANCEL_MESSAGE),
            "typed engine Canceled must map to WorkerError::Canceled, got {mapped:?}"
        );
        let failed = map_sensenova_engine_error(
            runtime_cuda::media::CandleError::Msg("boom".to_owned()),
            "SenseNova interleave failed",
            INTERLEAVE_CANCEL_MESSAGE,
        );
        assert!(
            matches!(failed, WorkerError::Engine(ref m) if m == "SenseNova interleave failed: boom"),
            "non-cancel engine errors must stay Engine(context: error), got {failed:?}"
        );
    }

    #[test]
    fn strip_reasoning_removes_think_blocks() {
        assert_eq!(strip_reasoning("<think>reasoning</think>answer"), "answer");
        assert_eq!(
            strip_reasoning("before <think>mid</think> after"),
            "before  after"
        );
        // Dangling/unclosed block (truncated by max_new_tokens) is dropped entirely.
        assert_eq!(strip_reasoning("answer<think>cut off"), "answer");
        // No think block: returned trimmed, unchanged.
        assert_eq!(strip_reasoning("  plain answer  "), "plain answer");
    }

    #[test]
    fn payload_int_parses_clamps_and_defaults() {
        let map = json!({ "a": 500, "b": "1024", "c": 3.0, "d": "bad" })
            .as_object()
            .cloned()
            .unwrap();
        assert_eq!(payload_int(&map, "a", 256, 16, 2048), 500);
        assert_eq!(payload_int(&map, "b", 256, 16, 2048), 1024);
        assert_eq!(
            payload_int(&map, "c", 256, 16, 2048),
            16,
            "3 clamps up to lo"
        );
        assert_eq!(
            payload_int(&map, "d", 256, 16, 2048),
            256,
            "unparseable → default"
        );
        assert_eq!(payload_int(&map, "missing", 768, 16, 2048), 768);
        assert_eq!(
            payload_int(&map, "a", 256, 16, 400),
            400,
            "500 clamps to hi"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn interleave_resolution_snaps_to_aspect_bucket() {
        // Exact 16:9 → its bucket.
        assert_eq!(interleave_resolution_snap(2048, 1152), (2048, 1152));
        // Square-ish → 1:1.
        assert_eq!(interleave_resolution_snap(1000, 1000), (1536, 1536));
        // Tall portrait → 9:16.
        assert_eq!(interleave_resolution_snap(1152, 2048), (1152, 2048));
        // Extreme wide → 3:1.
        assert_eq!(interleave_resolution_snap(3000, 1000), (2592, 864));
    }

    /// sc-8839: the shared VQA result assembly (`vqa_result_json`) both engine handlers post. Pure,
    /// so it is asserted cross-platform (no weights / device / backend cfg).
    #[test]
    fn vqa_result_json_has_the_shared_answer_shape() {
        let result = vqa_result_json("the answer", "the question", "asset_src", "sensenova_u1_8b");
        assert_eq!(
            result.get("answer").and_then(Value::as_str),
            Some("the answer")
        );
        assert_eq!(
            result.get("question").and_then(Value::as_str),
            Some("the question")
        );
        assert_eq!(
            result.get("sourceAssetId").and_then(Value::as_str),
            Some("asset_src")
        );
        assert_eq!(
            result.get("model").and_then(Value::as_str),
            Some("sensenova_u1_8b")
        );
        assert_eq!(
            result.get("realModelInference").and_then(Value::as_bool),
            Some(true)
        );
    }

    /// sc-8839: the shared handler's rollout-knob resolver (`resolve_interleave_params`) — the single
    /// backend-neutral "brain" both engine seams now read. Asserts the resolution snap, the advanced
    /// overlay overrides, and the resolved system message / seed with a fake payload (no weights /
    /// device), which is the shared-driver behavior the per-backend seams then consume identically.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_interleave_params_snaps_resolution_and_overlays_advanced() {
        // Overrides supplied via the `advanced` overlay + a portrait W×H + explicit seed.
        let payload = json!({
            "projectId": "proj_1",
            "prompt": "a document",
            "width": 1152,
            "height": 2048,
            "seed": 7,
            "advanced": {
                "numInferenceSteps": 20,
                "guidanceScale": 6.5,
                "imageGuidanceScale": 2.0,
                "timestepShift": 1.5,
                "maxNewTokens": 512,
                "maxImages": 3,
                "thinkMode": true,
                "systemMessage": "custom protocol"
            }
        })
        .as_object()
        .cloned()
        .unwrap();
        let request = ImageRequest::from_payload(&payload);
        let params = resolve_interleave_params(&request, &payload);
        assert_eq!(
            params,
            InterleaveParams {
                // Portrait 1152×2048 snaps to the 9:16 bucket.
                width: 1152,
                height: 2048,
                steps: 20,
                cfg_scale: 6.5,
                img_cfg_scale: 2.0,
                timestep_shift: 1.5,
                max_new_tokens: 512,
                max_images: 3,
                think_mode: true,
                system_message: "custom protocol".to_owned(),
                seed: 7,
            }
        );
    }

    /// sc-8839: an empty `advanced` overlay resolves to the upstream interleave defaults (the
    /// non-think document-studio defaults), and the default 2048×1152 payload snaps to the 16:9
    /// bucket. Complements the override test above.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_interleave_params_uses_upstream_defaults_when_advanced_is_empty() {
        let payload = json!({ "projectId": "proj_1", "prompt": "hello", "seed": 0 })
            .as_object()
            .cloned()
            .unwrap();
        let request = ImageRequest::from_payload(&payload);
        let params = resolve_interleave_params(&request, &payload);
        assert_eq!(
            params,
            InterleaveParams {
                width: 2048,
                height: 1152,
                steps: 50,
                cfg_scale: 4.0,
                img_cfg_scale: 1.0,
                timestep_shift: 3.0,
                max_new_tokens: 2048,
                max_images: 6,
                think_mode: false,
                system_message: INTERLEAVE_SYSTEM_MESSAGE.to_owned(),
                seed: 0,
            }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_segments_interleaves_text_and_images() {
        let writes = vec![
            json!({ "assetId": "asset_a", "mediaPath": "assets/images/g/a.png" }),
            json!({ "assetId": "asset_b", "mediaPath": "assets/images/g/b.png" }),
        ];
        let segments = build_interleaved_segments("intro<image>middle<image>end", &writes);
        assert_eq!(segments.len(), 5);
        assert_eq!(segments[0], json!({ "type": "text", "text": "intro" }));
        assert_eq!(
            segments[1],
            json!({ "type": "image", "assetId": "asset_a", "path": "assets/images/g/a.png" })
        );
        assert_eq!(segments[2], json!({ "type": "text", "text": "middle" }));
        assert_eq!(
            segments[3],
            json!({ "type": "image", "assetId": "asset_b", "path": "assets/images/g/b.png" })
        );
        assert_eq!(segments[4], json!({ "type": "text", "text": "end" }));
    }

    /// The HF-cache snapshot dir for a cached repo (test helper).
    #[cfg(target_os = "macos")]
    fn hf_snapshot(model_dir: &str) -> PathBuf {
        let home = std::env::var("HOME").expect("HOME set");
        std::fs::read_dir(
            PathBuf::from(home).join(format!(".cache/huggingface/hub/{model_dir}/snapshots")),
        )
        .expect("HF cache snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a snapshot dir")
    }

    /// A synthetic RGB8 gradient (test source image).
    #[cfg(target_os = "macos")]
    fn gradient_image(width: u32, height: u32) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push((x % 256) as u8);
                pixels.push((y % 256) as u8);
                pixels.push(((x + y) % 256) as u8);
            }
        }
        Image {
            width,
            height,
            pixels,
        }
    }

    /// Real-weights smoke: SenseNova-U1 VQA. Loads the dense base `T2iModel` (~35GB
    /// `sensenova/SenseNova-U1-8B-MoT`), preprocesses a synthetic image, and asserts the answer
    /// text is non-empty (post think-strip). Needs the HF cache + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_vqa_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SenseNova-U1-8B-MoT weights (~35GB) + Metal device"]
    fn sensenova_vqa_real_weights_answers_non_empty() {
        let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT");
        let (model, tokenizer) = load_sensenova_model(&snapshot).expect("load model");
        let image = gradient_image(512, 512);
        let pixel_values = image_to_chw01(&image, 256 * 256, 768 * 768).expect("preprocess");
        let answer = model
            .vqa(
                &tokenizer,
                "What colors appear in this image?",
                std::slice::from_ref(&pixel_values),
                64,
                Sampler::Greedy,
                None,
            )
            .expect("vqa");
        let answer = strip_reasoning(&answer);
        assert!(
            !answer.is_empty(),
            "VQA answer should be non-empty: {answer:?}"
        );
    }

    /// Real-weights smoke: SenseNova-U1 interleave. Loads the dense base `T2iModel`, runs a short
    /// think-mode interleave rollout (mirroring the engine's own real-weight test, which reliably
    /// emits an image), decodes the generated image(s), and asserts ≥1 image + a valid segment set
    /// with at least one image segment. Small 512² + 8 steps for speed (production buckets are
    /// 1536²+ / 50 steps). Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_interleave_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SenseNova-U1-8B-MoT weights (~35GB) + Metal device"]
    fn sensenova_interleave_real_weights_produces_document() {
        let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT");
        let (model, tokenizer) = load_sensenova_model(&snapshot).expect("load model");
        let opts = T2iOptions {
            cfg_scale: 4.0,
            img_cfg_scale: 1.0,
            num_steps: 8,
            timestep_shift: 3.0,
            seed: 42,
            think_mode: true,
            ..Default::default()
        };
        // Budget mirrors the engine's own passing interleave real-weight test (512 new tokens,
        // generous max_images) so the think-mode rollout reliably reaches an `<img>`.
        let out = model
            .interleave_gen(
                &tokenizer,
                "Generate an illustration of a single red circle on a white background, then briefly describe it.",
                &[],
                512,
                512,
                &opts,
                INTERLEAVE_SYSTEM_MESSAGE,
                512,
                4,
                None,
                &gen_core::CancelFlag::new(),
                &mut |_| {},
            )
            .expect("interleave_gen");
        assert!(!out.images.is_empty(), "expected >= 1 generated image");
        let mut image_writes = Vec::new();
        for (index, image) in out.images.iter().enumerate() {
            let decoded = decoded_to_image(image).expect("decode");
            assert_eq!(
                decoded.pixels.len(),
                (decoded.width * decoded.height * 3) as usize
            );
            image_writes.push(json!({
                "assetId": format!("asset_{index}"),
                "mediaPath": format!("assets/images/g/{index}.png"),
            }));
        }
        let segments = build_interleaved_segments(&out.text, &image_writes);
        assert!(
            segments.iter().any(|s| s["type"] == "image"),
            "document should contain >= 1 image segment: {segments:?}"
        );
    }

    /// sc-9960 (epic 9959) S0 engine-load de-risk: prove the NEW `Infographic-V2` checkpoint
    /// (`sensenova/SenseNova-U1-8B-MoT-Infographic-V2`, ~33GB dense bf16) loads and runs on the
    /// CURRENTLY pinned `mlx-gen-sensenova` with NO engine change. Static analysis already showed
    /// V2's `config.json` is byte-identical to V1's and its tensor namespace adds nothing new, so
    /// this is expected to be a no-op like the LTX-2.3 dense tier — but S0's AC requires an actual
    /// render + VQA, not just the static diff. Exercises the full stack in one shot: config +
    /// sharded weight load + tokenizer (V2 ships slow-only vocab.json/merges.txt, no tokenizer.json —
    /// a `load_tokenizer` failure here means a derived-tokenizer overlay is an S1 conversion step,
    /// NOT an architecture problem), the understanding path (VQA), and the generation path
    /// (interleave decode + non-degenerate render check). Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_v2_infographic_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SenseNova-U1-8B-MoT-Infographic-V2 weights (~33GB) + Metal device"]
    fn sensenova_v2_infographic_real_weights_load_render_vqa() {
        let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT-Infographic-V2");
        let (model, tokenizer) =
            load_sensenova_model(&snapshot).expect("load V2 model on the current pinned engine");

        // Understanding path: VQA over a synthetic image must return non-empty text post think-strip.
        let image = gradient_image(512, 512);
        let pixel_values = image_to_chw01(&image, 256 * 256, 768 * 768).expect("preprocess");
        let answer = model
            .vqa(
                &tokenizer,
                "What colors appear in this image?",
                std::slice::from_ref(&pixel_values),
                64,
                Sampler::Greedy,
                None,
            )
            .expect("V2 vqa");
        let answer = strip_reasoning(&answer);
        assert!(
            !answer.is_empty(),
            "V2 VQA answer should be non-empty: {answer:?}"
        );

        // Generation path: the PRIMARY t2i render (what the Infographic-V2 checkpoint is tuned for).
        // 512² + 16 steps for speed (production is 2048²/50); we only assert the decode is a real,
        // non-degenerate image, not infographic quality (that's S4 on-device validation).
        let opts = T2iOptions {
            cfg_scale: 4.0,
            img_cfg_scale: 1.0,
            num_steps: 16,
            timestep_shift: 3.0,
            seed: 42,
            think_mode: false,
            ..Default::default()
        };
        let out = model
            .generate(
                &tokenizer,
                "an infographic poster about the water cycle, clean vector style",
                512,
                512,
                &opts,
                None,
                None,
            )
            .expect("V2 t2i generate");
        let decoded = decoded_to_image(&out.image).expect("decode");
        assert_eq!(
            decoded.pixels.len(),
            (decoded.width * decoded.height * 3) as usize
        );
        // Non-degenerate check: a NaN/all-black/flat decode collapses the per-pixel std toward 0.
        let n = decoded.pixels.len() as f64;
        let mean = decoded.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
        let std = (decoded
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        assert!(
            std > 10.0,
            "V2 render looks degenerate (std {std:.2}, mean {mean:.2}) — possible NaN / all-black / flat decode"
        );
    }

    /// sc-8771 on-device verify (MLX, epic 8506 Group-B): drive the ACTUAL packed worker seam against
    /// the downloaded SceneWorks/sensenova-u1-8b-mlx **q4** turnkey tier. SenseNova-U1 is a unified MoT
    /// (no separate TE/VAE) whose whole backbone packs into a flat `q4/model.safetensors`; the worker's
    /// `standard_tier_subdir` (covered by `resolves_flat_unified_backbone_tiers`) resolves the `q4/`
    /// subdir from the tier root, then `load_sensenova_model` loads it — the shared `load_raw` +
    /// `T2iModel::from_weights` auto-detect the packed `.scales` sidecars (mlx-gen #623) and build the
    /// quantized decoder stack with NO dense bf16 transient. A short think-mode interleave rollout then
    /// renders a real image, proving the packed q4 tier both loads AND generates non-degenerately.
    /// Small 512² + 8 steps for speed (production is 1536²+/50 steps). Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_q4_tier_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight MLX smoke; needs the SceneWorks/sensenova-u1-8b-mlx q4 tier cached + a Metal device"]
    fn sensenova_q4_tier_real_weights_produces_image() {
        // Resolve the packed `q4/` tier subdir from the turnkey root — the same selection the worker's
        // `standard_tier_subdir` makes for a default job when only the q4 tier is installed (the q8
        // default, sc-10726, clamps down to the sole installed tier here).
        let root = hf_snapshot("models--SceneWorks--sensenova-u1-8b-mlx");
        let q4_dir = root.join("q4");
        assert!(
            q4_dir.join("model.safetensors").is_file(),
            "packed q4 tier not found at {} — download it: \
             hf download SceneWorks/sensenova-u1-8b-mlx --include 'q4/*'",
            q4_dir.display()
        );
        let (model, tokenizer) = load_sensenova_model(&q4_dir).expect("load packed q4 model");
        let opts = T2iOptions {
            cfg_scale: 4.0,
            img_cfg_scale: 1.0,
            num_steps: 8,
            timestep_shift: 3.0,
            seed: 42,
            think_mode: true,
            ..Default::default()
        };
        let out = model
            .interleave_gen(
                &tokenizer,
                "Generate an illustration of a single red circle on a white background, then briefly describe it.",
                &[],
                512,
                512,
                &opts,
                INTERLEAVE_SYSTEM_MESSAGE,
                512,
                4,
                None,
                &gen_core::CancelFlag::new(),
                &mut |_| {},
            )
            .expect("interleave_gen");
        assert!(
            !out.images.is_empty(),
            "packed q4 tier should render >= 1 image"
        );
        let decoded = decoded_to_image(&out.images[0]).expect("decode");
        assert_eq!(
            decoded.pixels.len(),
            (decoded.width * decoded.height * 3) as usize
        );
        // Non-degenerate check: a NaN/all-black/flat decode collapses the per-pixel std toward 0.
        let n = decoded.pixels.len() as f64;
        let mean = decoded.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
        let std = (decoded
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        assert!(
            std > 10.0,
            "packed q4 render looks degenerate (std {std:.2}, mean {mean:.2}) — possible NaN / all-black / flat decode"
        );
    }

    /// sc-8775 on-device verify (MLX, epic 8506 Group-B): the DISTILLED `_fast` variant's own packed
    /// re-host. `SceneWorks/sensenova-u1-8b-fast-mlx` ships the 8-step distill LoRA PRE-MERGED into the
    /// generation path and then packed, so its `q4/model.safetensors` is a distinct checkpoint from the
    /// base re-host. The worker's `standard_tier_subdir` resolves the flat `q4/` subdir identically to
    /// the base (covered by `resolves_flat_unified_backbone_tiers`); `load_sensenova_model`'s shared
    /// `load_raw` + `T2iModel::from_weights` auto-detect the packed `.scales` sidecars and build the
    /// quantized decoder stack with the merged gen-path weights (NO dense transient, NO load-time
    /// merge). A short 8-NFE / CFG-1.0 T2I rollout — the distilled defaults — then renders a real image,
    /// proving the pre-merged packed tier both loads AND generates non-degenerately (a bad merge would
    /// corrupt the gen path and flatten the decode). The marker-gated `load_fast` skip that the engine
    /// registry uses for this tier is separately proven in mlx-gen's `prequantize_real_weights`
    /// (`SC8771_MODEL=sensenova_u1_8b_fast`). Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_fast_q4_tier_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight MLX smoke; needs the SceneWorks/sensenova-u1-8b-fast-mlx q4 tier cached + a Metal device"]
    fn sensenova_fast_q4_tier_real_weights_produces_image() {
        let root = hf_snapshot("models--SceneWorks--sensenova-u1-8b-fast-mlx");
        let q4_dir = root.join("q4");
        assert!(
            q4_dir.join("model.safetensors").is_file(),
            "packed fast q4 tier not found at {} — download it: \
             hf download SceneWorks/sensenova-u1-8b-fast-mlx --include 'q4/*'",
            q4_dir.display()
        );
        // The distill LoRA is baked in; the marker rides the tier for provenance/loader gating.
        assert!(
            q4_dir.join("distill_merged.json").is_file(),
            "pre-merged fast tier missing its distill_merged.json marker at {}",
            q4_dir.display()
        );
        let (model, tokenizer) = load_sensenova_model(&q4_dir).expect("load packed fast q4 model");
        // Distilled defaults: 8 NFE at CFG 1.0 (the `_fast` variant's manifest defaults).
        let opts = T2iOptions {
            cfg_scale: 1.0,
            img_cfg_scale: 1.0,
            num_steps: 8,
            timestep_shift: 3.0,
            seed: 42,
            ..Default::default()
        };
        let out = model
            .generate(
                &tokenizer,
                "a single red circle on a white background, flat vector illustration",
                512,
                512,
                &opts,
                None,
                None,
            )
            .expect("fast t2i generate");
        let decoded = decoded_to_image(&out.image).expect("decode");
        assert_eq!(
            decoded.pixels.len(),
            (decoded.width * decoded.height * 3) as usize
        );
        // Non-degenerate check: a NaN/all-black/flat decode (e.g. a corrupt merge) collapses std → 0.
        let n = decoded.pixels.len() as f64;
        let mean = decoded.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
        let std = (decoded
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        assert!(
            std > 10.0,
            "packed fast q4 render looks degenerate (std {std:.2}, mean {mean:.2}) — possible NaN / all-black / flat decode or a bad distill merge"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_segments_trailing_image_without_marker_is_appended() {
        // One `<image>` marker but two images generated. Mirrors the Python splitter exactly:
        // split → ["only one", ""]; index 0 slots image[0] after the text, index 1 (empty text)
        // still slots image[1] because `index < len(image_writes)`. So the extra image trails.
        let writes = vec![
            json!({ "assetId": "asset_a", "mediaPath": "a.png" }),
            json!({ "assetId": "asset_b", "mediaPath": "b.png" }),
        ];
        let segments = build_interleaved_segments("only one<image>", &writes);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], json!({ "type": "text", "text": "only one" }));
        assert_eq!(segments[1]["assetId"], "asset_a");
        assert_eq!(segments[2]["assetId"], "asset_b");
    }
}
