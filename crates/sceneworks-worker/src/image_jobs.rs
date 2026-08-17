//! Native MLX image generation jobs — runtime pipeline + Z-Image inference (epic 3018).
//!
//! Parses the job into an [`ImageRequest`], generates `count` images, saves each PNG
//! into the project's `assets/images/`, and reports flat "facts" the Rust API turns
//! into indexed assets. The API's `persist_reported_assets` (apps/rust-api jobs.rs)
//! runs on EVERY progress update — idempotently building each sidecar via
//! `build_image_sidecar_parts` and indexing project.db — so emitting the accumulating
//! `assetWrites` per image is what streams results into the gallery as they land.
//!
//! On macOS, engine-backed families (`z_image_turbo` — sc-3022; `flux_schnell` /
//! `flux_dev` — sc-3023; `qwen_image` — sc-3024 / strict pose sc-3575) run **real**
//! in-process inference via the linked mlx-gen
//! engine; other models (and non-macOS) fall back to a procedural stub (sc-3020), so
//! the pipeline stays cross-platform-testable and each new family just adds a row to
//! the [`crate::engines::MODEL_TABLE`] dispatch table + links its provider crate.

use super::*;
// Used only by the generation harness in base.rs (the metrics builders), which is
// itself `include!`d only on macOS / the backend-candle lane — so gate the import to
// match, or the Linux-no-candle "neither" build sees it as unused (`-D warnings`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::contracts::GenerationMetrics;
use sceneworks_core::image_request::ImageRequest;
use sceneworks_core::workflow_png::write_workflow_chunk;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
use sceneworks_core::workflow_share::trusted_lora_for_share;
use sceneworks_core::workflow_share::{
    embeddable_workflow_share, WorkflowAssetFacts, WorkflowLora, OMITTED_PHASES,
};
use std::sync::Arc;

// Backend-neutral contract types come from the canonical inference release. The selected runtime
// bundle explicitly owns its provider catalog; this product module names only contract types and
// the few bespoke utility APIs that do not implement the general registry traits.
// Contract types for the generation harness — shared by the macOS MLX path AND the Windows candle
// lane (sc-3675), so broadened from macOS-only. `gen_core` is a direct worker dep on every platform.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, Conditioning, GenerationOutput, GenerationRequest,
    Generator, Image, LoadPhase, LoadSpec, Progress, Quant, WeightsSource,
};
// `IdentityWeights` (the PuLID-FLUX `LoadSpec::identity` seam, sc-8827) is used only by the macOS MLX
// PuLID path (`image_jobs/pulid.rs`); gate it so the candle lane's `-D warnings` sees no unused import.
#[cfg(target_os = "macos")]
use gen_core::IdentityWeights;
// `AdapterKind` (LoRA/LoKr classification) was MLX-only until sc-5126 introduced the first candle
// adapter lane; it now serves the shared MLX and candle adapter loaders, so the import lives in the
// shared block above. `ControlKind` (ControlNet conditioning) was MLX-only until sc-8304: the candle
// strict-control trio (`candle_strict_control.rs`) now shares the cross-platform `strict_control.rs`
// `(engine_id, supported_kinds)` table + `preprocess_control_entry`, so `ControlKind` is in scope on the
// candle build too.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::ControlKind;

// Provider-specific utilities below are explicit exports of the selected runtime bundle; ordinary
// generators, trainers, captioners, and embedders are reached only through `inference_runtime`.
// InstantID (sc-3345) is a bespoke provider, not a general `Generator`, so it is reached through
// the runtime bundle's named utility export. The `runtime_macos::media::weights::Weights` loader and
// concrete InstantID API stay MLX-typed until the face stack moves onto a neutral contract.
#[cfg(target_os = "macos")]
use runtime_macos::media::weights::Weights;
#[cfg(target_os = "macos")]
use runtime_macos::providers::instantid::{
    BodyPoint, InstantId, InstantIdPaths, InstantIdRequest, FACE_RESTORE_PROMPT,
};
// The Windows/CUDA sibling: the candle InstantID provider (sc-5491, epic 5480), retiring the Python
// `_vendor/instantid` off-Mac. The same bespoke by-name reference (`InstantId::load`) is owned by
// `runtime-cuda` rather than the general media registry. The SCRFD + ArcFace FaceEmbedder the model
// composes (`candle-gen-face`, sc-5490) rides in transitively
// via `candle-gen-instantid` and is used directly (not through the registry), so it needs no direct
// worker dep. The candle `with_face` loads the face pair from THEIR DIRECTORY, so there is no
// `Weights::from_file` import on this lane (the MLX `Weights` loader above stays macOS-only).
// `InstantIdPaths`/`InstantIdRequest`/`BodyPoint` resolve to the candle crate's types, but the
// conditioning types they carry (`WeightsSource`, `Image`, `CancelFlag`, `Progress`) are the SHARED
// `gen_core` contract — the single-rev skew gate (sc-4482) is what makes the worker's `gen_core::Image`
// the exact type `InstantId::generate` consumes.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::instantid::{
    BodyPoint, InstantId, InstantIdPaths, InstantIdRequest, SdxlComponents,
    COMPONENT_TOKENIZER_CLIP_BIGG, COMPONENT_TOKENIZER_CLIP_L, COMPONENT_VAE_FP16_FIX,
    FACE_RESTORE_PROMPT,
};
// SDXL IP-Adapter-Plus reference provider (sc-5488, epic 5480) — the candle (Windows/CUDA) reference-
// conditioning sibling of the InstantID lane, living in `candle-gen-sdxl` (it composes that crate's
// IP-Adapter Resampler + the new CLIP ViT-H image encoder + a pure-IP denoise). Candle-only: macOS keeps
// the MLX SDXL IP path (the registry `SdxlSubMode::Ip`), so these named types resolve only off-Mac.
// The bespoke reference route (`image_jobs/sdxl_ipadapter.rs`) uses this named utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::sdxl::{
    IpAdapterSdxl, IpAdapterSdxlPaths, IpAdapterSdxlRequest, SdxlEdit, SdxlEditPaths,
    SdxlEditRequest,
};
// FLUX.2-klein reference / img2img edit provider (sc-5487, epic 5480) — the candle (Windows/CUDA) FLUX.2
// edit lane (the sibling of the SDXL edit lane above), living in `candle-gen-flux2` (Kontext-style
// reference token-concat over the txt2img FLUX.2 stack + the VAE encoder). Candle-only: macOS keeps the
// MLX `flux2_klein_9b_edit` registry path. The bespoke edit route
// (`image_jobs/flux2_edit_candle.rs`) uses this named utility export. The same crate carries `Flux2Control`
// (FLUX.2-dev Fun-Controlnet-Union strict-pose VACE branch, sc-7460) the candle pose route
// (`image_jobs/flux2_control_candle.rs`, sc-7736) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::flux2::{
    Flux2Control, Flux2ControlPaths, Flux2ControlRequest, Flux2Edit, Flux2EditPaths,
    Flux2EditRequest,
};
// Kolors IP-Adapter-Plus reference provider (sc-5488, epic 5480) — the candle (Windows/CUDA) Kolors
// sibling of the SDXL IP lane, living in `candle-gen-kolors` (it reuses candle-gen-sdxl's vendored IP
// UNet + the CLIP ViT-L/14-336 image encoder, with the Kolors ChatGLM3 conditioning + leading-Euler
// sampler). Candle-only: macOS keeps the MLX Kolors IP path (the registry `Reference` route), so these
// named types resolve only off-Mac. The bespoke reference route
// (`image_jobs/kolors_ipadapter.rs`) uses this named utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::kolors::{
    IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest,
};
// FLUX XLabs IP-Adapter reference provider (sc-5872, epic 5480) — the candle (Windows/CUDA) FLUX sibling
// of the SDXL/Kolors IP lanes, living in `candle-gen-flux` (the forked FLUX DiT with the per-double-block
// XLabs seam + the pooled CLIP-ViT-L image encoder). Candle-only: macOS keeps the MLX FLUX XLabs IP path
// (epic 3621, the registry `Reference` route). The bespoke route
// (`image_jobs/flux_ipadapter.rs`) uses this named utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::flux::{IpAdapterFlux, IpAdapterFluxPaths, IpAdapterFluxRequest};
// FLUX.1-dev strict-control Fun-Controlnet-Union provider (sc-8412, epic 8236) — the candle (Windows/CUDA)
// FLUX.1-dev sibling of the FLUX.2 / Z-Image / Qwen strict-control lanes, living in `candle-gen-flux` (the
// Shakker Union-Pro-2.0 residual-emitter control branch overlaid on the FLUX.1-dev base via the
// compose-ready DiT seam). Candle-only: macOS keeps the MLX `flux1_dev_control` registry generator
// (flux1_control.rs). The bespoke control route (`image_jobs/flux1_control_candle.rs`) uses this
// named utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::flux::{Flux1ControlPaths, Flux1ControlRequest, Flux1DevControl};
// Qwen-Image 2512-Fun-Controlnet-Union (strict control) provider (sc-5489 origin / sc-8350 repoint, epic
// 8236) — the candle (Windows/CUDA) strict-control lane. As of sc-8350 this rides the input-agnostic
// `QwenFunControl` VACE engine on the Qwen-Image-2512 base (the InstantX `QwenControl` is retired on the
// candle lane; the engine stays in the crate, unused by the worker). Candle-only: macOS keeps the MLX
// `qwen_image_control` registry generator. The bespoke control route
// (`image_jobs/qwen_control.rs`) uses this named utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::qwen_image::{
    QwenFunControl, QwenFunControlPaths, QwenFunControlRequest,
};
// Qwen-Image-Edit provider (sc-5487, epic 5480) — the candle (Windows/CUDA) reference-edit lane (the
// last family of sc-5487; SDXL + FLUX.2-klein edit already shipped). Candle-only: macOS keeps the MLX
// `qwen_image_edit` registry path. The bespoke edit route (`image_jobs/qwen_edit_candle.rs`) uses
// this named utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::qwen_image::{QwenEdit, QwenEditPaths, QwenEditRequest};
// Kolors ControlNet (strict pose) provider (sc-5489, epic 5480) — the candle (Windows/CUDA) Kolors
// sibling of the Qwen strict-pose lane, living in `candle-gen-kolors` (it reuses candle-gen-sdxl's
// vendored UNet + the SDXL `ControlNet`, with the Kolors ChatGLM3 conditioning + leading-Euler sampler).
// Candle-only: macOS keeps the MLX Kolors ControlNet path. The bespoke pose route
// (`image_jobs/kolors_control.rs`) uses this named utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::kolors::{KolorsControl, KolorsControlPaths, KolorsControlRequest};
// Z-Image Fun-ControlNet (strict pose) provider (sc-5489, epic 5480) — the candle (Windows/CUDA)
// Z-Image sibling of the Qwen/Kolors strict-pose lanes, living in `candle-gen-z-image` (the VACE-style
// dual-injection control on the vendored DiT). Candle-only: macOS keeps the MLX `z_image_turbo_control`
// registry generator. The bespoke pose route (`image_jobs/zimage_control.rs`) uses this named utility
// export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::z_image::{ZImageControl, ZImageControlPaths, ZImageControlRequest};
// PuLID-FLUX face-identity provider (sc-5492, epic 5480) — the candle (Windows/CUDA) sibling of the
// macOS `pulid_flux` registry generator, living in `candle-gen-pulid` (the EVA02-CLIP tower + IDFormer
// + the 20 PerceiverAttentionCA modules injected into the forked FLUX DiT via the post-block
// `DitImageInjector` seam, composing the gen-core FaceEmbedder + the BiSeNet `face_features_image`).
// Candle-only: macOS keeps the inventory-registered `pulid_flux` MLX generator; the candle `PulidFlux`
// is a bespoke provider referenced BY NAME (like `InstantId`), so no `as _;` anchor is needed — this is
// the named-type import the bespoke route (`image_jobs/pulid_candle.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::pulid::{PulidFlux, PulidFluxPaths, PulidFluxRequest};

/// The stub adapter id recorded on generated assets (matches the contract fixture
/// `tests/fixtures/rust_migration_contracts/sidecars/asset-image.sceneworks.json`).
const STUB_ADAPTER: &str = "procedural_preview";
/// The adapter id recorded on assets produced by the candle (Windows/CUDA) SDXL lane (sc-3678).
/// Used by the generic candle per-asset stream and its route-derived generation-set label.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_ADAPTER: &str = "candle_sdxl";
// Shared by the MLX path and every adapter-capable candle image lane: all cap a job's total LoRAs at
// MAX_JOB_LORAS (`resolve_adapters`), so the const is available on the Windows candle build too.
// The web pickers enforce a lower user-selectable cap (presetUtils.MAX_USER_JOB_LORAS) that leaves
// headroom for an auto-applied builtin within this total (sc-8936).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MAX_JOB_LORAS: usize = 5;

// The engine dispatch table + its `ModelRow`/`mlx_model` join moved to the all-targets
// `engines` module (sc-3723); the two descriptor-duplicating flags it used to carry
// (`supports_guidance`/`supports_negative_prompt`) are now read from the linked gen_core
// descriptor via `ResolvedModel`. Shared by the macOS MLX path and the Windows candle lane
// (sc-5096) — the join is backend-neutral, so `generate_candle_stream` resolves repo/steps/guidance
// through the same `mlx_model` lookup the MLX path uses.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::engines::{mlx_model, ResolvedModel};

/// Parse the request-selected terminal decoder. Missing, null, empty, and `native` all preserve the
/// provider's built-in decoder byte-for-byte; every other value must be a bounded string id.
fn requested_decoder_id(
    advanced: &sceneworks_core::contracts::JsonObject,
) -> WorkerResult<Option<&str>> {
    match advanced.get("decoder") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.trim().is_empty() || value == "native" => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.as_str())),
        Some(_) => Err(WorkerError::InvalidPayload(
            "advanced.decoder must be a decoder id string (or 'native')".to_owned(),
        )),
    }
}

/// Dispatch handler for `JobType::ImageGenerate`: generate, save, and stream image
/// assets through the Rust GPU worker.
///
/// Takes no `reqwest::Client`: its only use was forwarding one to the inline-upscale post-pass, and
/// both upscalers became cache-only resolvers (sc-17633 / sc-17632). The likeness/tier staging this
/// handler still triggers builds its own context inside `image_jobs/base.rs`.
const PROMPT_ENHANCEMENT_FACT_KEY: &str = "promptEnhancement";
const PROMPT_ENHANCE_MAX_TOKENS: u64 = 2048;
const PROMPT_ENHANCE_MAX_TEMPERATURE: f64 = 2.0;

fn parse_prompt_enhancement_fields(
    advanced: &JsonObject,
) -> WorkerResult<(bool, Option<f32>, Option<u32>)> {
    if advanced.contains_key(PROMPT_ENHANCEMENT_FACT_KEY) {
        return Err(WorkerError::InvalidPayload(format!(
            "advanced.{PROMPT_ENHANCEMENT_FACT_KEY} is worker-owned"
        )));
    }
    let enabled = match advanced.get("enhancePrompt") {
        None => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => {
            return Err(WorkerError::InvalidPayload(
                "advanced.enhancePrompt must be a boolean".to_owned(),
            ));
        }
    };
    let temperature = advanced
        .get("enhanceTemperature")
        .map(|value| {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "advanced.enhanceTemperature must be a finite number".to_owned(),
                    )
                })?;
            if !(0.0..=PROMPT_ENHANCE_MAX_TEMPERATURE).contains(&value) {
                return Err(WorkerError::InvalidPayload(format!(
                    "advanced.enhanceTemperature must be between 0 and {PROMPT_ENHANCE_MAX_TEMPERATURE}"
                )));
            }
            Ok(value as f32)
        })
        .transpose()?;
    let max_tokens = advanced
        .get("enhanceMaxTokens")
        .map(|value| {
            let value = value.as_u64().ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "advanced.enhanceMaxTokens must be an integer".to_owned(),
                )
            })?;
            if !(1..=PROMPT_ENHANCE_MAX_TOKENS).contains(&value) {
                return Err(WorkerError::InvalidPayload(format!(
                    "advanced.enhanceMaxTokens must be between 1 and {PROMPT_ENHANCE_MAX_TOKENS}"
                )));
            }
            Ok(value as u32)
        })
        .transpose()?;
    if !enabled && (temperature.is_some() || max_tokens.is_some()) {
        return Err(WorkerError::InvalidPayload(
            "prompt-enhancement tuning requires advanced.enhancePrompt=true".to_owned(),
        ));
    }
    Ok((enabled, temperature, max_tokens))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn prompt_enhancement_has_edit_input(request: &ImageRequest) -> bool {
    request
        .source_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty())
        || request
            .reference_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        || !request.reference_asset_ids.is_empty()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn prompt_enhancement_has_reference_input(request: &ImageRequest) -> bool {
    request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty())
        || !request.reference_asset_ids.is_empty()
}

fn validate_prompt_enhancement_route(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<()> {
    let mode = request.mode.as_str();

    #[cfg(target_os = "macos")]
    {
        let _ = settings;
        if !matches!(
            mode,
            "text_to_image" | "edit_image" | "character_image" | "style_variations"
        ) {
            return Err(WorkerError::InvalidPayload(format!(
                "prompt enhancement on MLX does not support image mode {mode}"
            )));
        }
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        if !settings.backend_candle_enabled {
            return Err(WorkerError::InvalidPayload(
                "prompt enhancement requires the enabled native Candle backend on this worker"
                    .to_owned(),
            ));
        }
        if !matches!(mode, "text_to_image" | "edit_image") {
            return Err(WorkerError::InvalidPayload(format!(
                "prompt enhancement on Candle supports only text_to_image and edit_image; mode {mode} is unsupported"
            )));
        }
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        let _ = (mode, settings);
        Err(WorkerError::InvalidPayload(
            "prompt enhancement requires a native MLX or Candle image backend".to_owned(),
        ))
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        if mode == "text_to_image" && prompt_enhancement_has_edit_input(request) {
            return Err(WorkerError::InvalidPayload(
                "prompt enhancement text_to_image cannot include source or reference image assets"
                    .to_owned(),
            ));
        }
        if mode == "edit_image" && !prompt_enhancement_has_edit_input(request) {
            return Err(WorkerError::InvalidPayload(
                "prompt enhancement edit_image requires a source or reference image asset"
                    .to_owned(),
            ));
        }
        if matches!(mode, "character_image" | "style_variations")
            && !prompt_enhancement_has_reference_input(request)
        {
            return Err(WorkerError::InvalidPayload(format!(
                "prompt enhancement {mode} requires a reference image asset"
            )));
        }
        Ok(())
    }
}

/// Re-check the backend and route at the worker trust boundary. Raw queue writes and legacy stored
/// jobs need the same fail-closed behavior as typed API creates, including a build with no native
/// image backend. The route shape is checked before any weight or project asset load.
fn validate_prompt_enhancement_request(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<()> {
    let (enabled, _, _) = parse_prompt_enhancement_fields(&request.advanced)?;
    if !enabled {
        return Ok(());
    }
    if request.model != "flux2_dev" {
        return Err(WorkerError::InvalidPayload(
            "prompt enhancement is supported only by FLUX.2-dev; FLUX.2-Klein and other models reject it"
                .to_owned(),
        ));
    }
    let strict_control = request
        .advanced
        .get("poses")
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
        || request.advanced.contains_key("controlWeights")
        || request.advanced.contains_key("controlImage")
        || request.advanced.contains_key("controlMode");
    if strict_control {
        return Err(WorkerError::InvalidPayload(
            "prompt enhancement cannot be combined with FLUX.2-dev strict control".to_owned(),
        ));
    }
    validate_prompt_enhancement_route(request, settings)
}

pub(crate) async fn run_image_generate_job(
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
    validate_hires_fix_request(&request)?;
    validate_prompt_enhancement_request(&request, settings)?;
    if let Some(decoder_id) = requested_decoder_id(&request.advanced)? {
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        {
            let backend = if cfg!(target_os = "macos") {
                "mlx"
            } else {
                "candle"
            };
            let provider_id = sceneworks_core::decoder_support::provider_id_for_backend(
                &request.model_manifest_entry,
                backend,
            )
            .or_else(|| mlx_model(&request.model).map(|model| model.engine_id().to_owned()))
            .unwrap_or_else(|| request.model.clone());
            validate_selected_decoder_request(&provider_id, decoder_id, &request.advanced)?;
        }
        #[cfg(not(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        )))]
        {
            return Err(WorkerError::InvalidPayload(format!(
                "decoder '{decoder_id}' is unavailable because this worker has no compatible image backend"
            )));
        }
    }
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    tokio::fs::create_dir_all(project_path.join("assets").join("images")).await?;

    // sc-8091: when the Image Studio "Upscale" toggle is on, each generated image also yields a
    // second "(Nx upscaled)" asset, so the generation set expects twice as many images. The inline
    // upscale post-pass only runs where the upscaler engines compile (macOS / candle); the
    // stub-only build keeps the base count.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    let upscale_mult: u32 = if request.upscale.enabled { 2 } else { 1 };
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let upscale_mult: u32 = 1;

    // Resolve the MLX dispatch branch once, then bake that branch's real total into
    // the plan so the generation set + streamed `expectedCount` match what lands in
    // the gallery.
    #[cfg(target_os = "macos")]
    let route = prepare_image_route(&request, settings)?;
    // Whether — and from what — every image this job writes embeds its sanitized workflow
    // (sc-15948). Resolved once here so the base write and the inline-upscale write share one
    // answer, and read live off the config dir so flipping the Settings toggle takes effect on the
    // next job rather than the next launch.
    let workflow_source = workflow_source(settings, &job.payload);

    #[cfg(target_os = "macos")]
    let plan = ImagePlan::with_count_and_adapter(
        &request,
        route.as_ref().map_or(request.count, |route| {
            route.kind().image_count(&request, settings)
        }) * upscale_mult,
        route
            .as_ref()
            .map_or(STUB_ADAPTER, |route| route.kind().adapter_label(&request)),
        workflow_source,
    );
    // Windows/CUDA candle lane: resolve the candle dispatch branch once and bake THAT branch's real
    // total into the plan, exactly as the macOS arm does with `resolve_image_route`. An InstantID
    // angle/pose set produces N images (the active angle collection's length, or the pose count) and
    // every strict-pose control lane produces one image per pose (`pose_entries().len()`), not
    // `request.count` — so the generation set + streamed `expectedCount` match what lands in the gallery
    // (sc-5491 InstantID; sc-11171 F-009 strict-pose). `resolve_candle_image_route` returns `None` when
    // candle is disabled, so any other job (or a disabled backend) keeps `request.count`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let route = prepare_candle_image_route(&request, settings)?;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let plan = ImagePlan::with_count_and_adapter(
        &request,
        route.as_ref().map_or(request.count, |route| {
            route.kind().image_count(&request, settings)
        }) * upscale_mult,
        route
            .as_ref()
            .map_or(STUB_ADAPTER, |route| route.kind().adapter_label(&request)),
        workflow_source,
    );
    #[cfg(all(
        not(target_os = "macos"),
        not(all(not(target_os = "macos"), feature = "backend-candle"))
    ))]
    let plan = ImagePlan::with_count(&request, request.count * upscale_mult, workflow_source);

    let mut plan = plan;
    if plan.workflow_source.is_some() {
        plan.model_hash =
            trusted_imported_model_hash(api, settings, job, plan.adapter, &request).await;
    }

    // Pre-flight LoRA family-compat guardrail (sc-3027): reject an incompatible LoRA
    // (e.g. a Flux LoRA on an SDXL model, or a Wan 5B LoRA on the 14B base) before any
    // heavy load, with the same message the Python worker raised — instead of failing
    // deep in the engine's strict adapter loader. Network-type handling (peft LoKr AND third-party
    // LyCORIS both apply on MLX now, epic 3641) is done by routing + `classify_adapter` + the engine.
    sceneworks_core::lora_family::validate_lora_compatibility(
        &request.loras,
        Some(plan.family.as_str()),
        plan.adapter,
        Some(request.model.as_str()),
    )
    .map_err(WorkerError::InvalidPayload)?;

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        let route_applies_loras = {
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            {
                route
                    .as_ref()
                    .is_some_and(|route| route.kind().applies_request_loras(&request))
            }
            #[cfg(target_os = "macos")]
            {
                route
                    .as_ref()
                    .is_some_and(|route| route.kind().applies_request_loras())
            }
        };
        // sc-18477: a request-owned adapter stack is part of the generation contract, not an
        // optional hint. Bespoke routes historically bypassed the generic LoadSpec and several of
        // them therefore rendered successfully while silently omitting request.loras. Fail before
        // any model/conditioning load unless the selected executable route explicitly consumes the
        // stack. Each route moves onto the allow-list only in the same change that wires its actual
        // provider load, which makes this guard fail closed for direct worker callers as well as API
        // submissions.
        if !request.loras.is_empty() && !route_applies_loras {
            // Label with the route KIND, not the prepared route itself: the prepared value carries
            // pinned load payloads and is deliberately not `Debug`, and the kind is what names the
            // lane in the refusal anyway.
            let route_label = route
                .as_ref()
                .map(|selected| format!("{:?}", selected.kind()))
                .unwrap_or_else(|| "unavailable".to_owned());
            return Err(WorkerError::InvalidPayload(format!(
                "{} cannot apply the selected LoRA/LoKr stack through the resolved {} image route; \
                 choose a model/request shape whose active backend supports adapters",
                request.model, route_label,
            )));
        }
        if plan.workflow_source.is_some() && route_applies_loras {
            plan.loras = trusted_loras_for_share(api, settings, job, &request).await;
        }
    }

    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            &format!("Preparing {} image(s).", plan.image_count),
            None,
            backend,
        ),
    )
    .await?;

    let mut asset_writes: Vec<Value> = Vec::with_capacity(plan.image_count as usize);

    // Real in-process MLX inference on macOS for engine-backed models; otherwise the
    // procedural stub (keeps non-macOS + not-yet-ported models working).
    #[cfg(target_os = "macos")]
    let handled = if let Some(route) = route {
        match route.kind() {
            ImageRoute::ZImageControl => {
                // Z-Image strict-pose (advanced.poses) → Fun-Controlnet-Union, one image per pose.
                generate_zimage_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::ZImageBaseControl => {
                // Base (full-CFG) Z-Image strict control (advanced.poses on `z_image`) → base
                // Fun-Controlnet-Union, one image per pose (sc-8251).
                generate_zimage_base_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::QwenControl => {
                // Qwen strict-pose (advanced.poses) → InstantX ControlNet-Union, one image per pose.
                generate_qwen_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KolorsControl => {
                // Kolors strict-pose (advanced.poses + a reference) → the combined pose ControlNet
                // + IP-Adapter identity + img2img pass (sc-4766 / engine sc-5012), one image per pose.
                generate_kolors_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KreaControl => {
                // Krea 2 Turbo strict-pose (advanced.poses on `krea_2_turbo`) → the trained control-branch
                // overlay on the frozen dense base (sc-8465, epic 8459 S5), one image per pose. The MLX
                // twin of the candle `CandleImageRoute::KreaControl` lane.
                generate_krea_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux1DevControl => {
                // FLUX.1-dev strict control (advanced.poses) → Shakker Union-Pro-2.0, one image per pose
                // (pose / canny / depth via advanced.controlMode).
                generate_flux1_dev_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux2DevControl => {
                // FLUX.2-dev strict-pose (advanced.poses) → Fun-Controlnet-Union, one image per pose.
                generate_flux2_dev_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux2Edit => {
                // FLUX.2-klein edit/reference (mode edit_image or a reference) → edit variant.
                generate_flux2_edit_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::QwenEdit => {
                // Qwen-Image-Edit (mode edit_image / Character-Studio reference / best-effort
                // pose / angle set) → the engine's `qwen_image_edit` model (sc-3397).
                generate_qwen_edit_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KreaEdit => {
                // Krea 2 Raw Kontext-style edit (mode edit_image + a source) → the `krea_2_edit`
                // engine: source as in-context VAE tokens + Qwen3-VL grounding (epic 10871, sc-10882).
                generate_krea_edit_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KreaTurboOnRaw => {
                // Krea 2 Raw t2i + the accelerator (turbo) LoRA (sc-13882) → the distilled Turbo
                // sampling regime (fixed mu 1.15 / ~8 steps / CFG-off) on the Raw base + LoRA additive
                // (epic 13879 S3, sc-13883). Routes to the `krea_2_turbo` engine while loading the Raw
                // weights — the engine-id sibling of the `krea_2_turbo_edit` vs `krea_2_edit` edit pick.
                generate_krea_turbo_on_raw_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KreaMultiPhase => {
                // Krea 2 Raw t2i + an explicit `advanced.phases` list (epic 13879 S4, sc-13884) → the
                // multi-phase denoise driver: ONE Raw trajectory / global sigma schedule, per-phase
                // guidance (CFG on/off) + per-phase toggling of the job's load-time LoRA stack (by
                // index). Reference/edit/pose/PiD shapes are rejected loudly before the load (renders
                // from pure noise). Takes precedence over the S3 turbo-on-Raw regime above.
                generate_krea_multiphase_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KreaImportedControl => {
                // Imported single-file Krea 2 checkpoint + strict-pose set: the trained pose
                // control-branch overlay rides the file-loaded imported DiT (the imported twin of
                // the `KreaControl` arm above), one pose-locked image per pose.
                let PreparedImageRoute::KreaImportedControl(sources) = route else {
                    unreachable!("Krea imported-control route missing its prepared sources")
                };
                generate_krea_imported_control_stream(
                    api,
                    settings,
                    job,
                    PreparedFileDispatch {
                        plan: &plan,
                        sources: *sources,
                    },
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KreaImported => {
                // Imported/user single-file Krea 2 checkpoint (epic 14015 S0c, sc-14018): pair the
                // imported DiT with a resident `krea_2` base tier (shared TE/VAE/tokenizer) and load via
                // the S0b MLX native single-file entrypoint. txt2img, `count` renders each its own seed.
                let PreparedImageRoute::KreaImported(sources) = route else {
                    unreachable!("Krea imported route missing its prepared sources")
                };
                generate_krea_imported_stream(
                    api,
                    settings,
                    job,
                    PreparedFileDispatch {
                        plan: &plan,
                        sources: *sources,
                    },
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::MageFinetuned => {
                // A full base fine-tune's own checkpoint (sc-15036): pair the trained transformer
                // with the installed Mage-Flow base's shared text encoder + VAE and render through
                // `load_finetuned`. txt2img, `count` renders each its own seed.
                let PreparedImageRoute::MageFinetuned(transformer) = route else {
                    unreachable!("Mage fine-tuned route missing its prepared transformer")
                };
                generate_mage_finetuned_stream(
                    api,
                    settings,
                    job,
                    *transformer,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::SdxlImported => {
                let PreparedImageRoute::SdxlImported(sources) = route else {
                    unreachable!("SDXL imported route missing its prepared sources")
                };
                generate_sdxl_imported_stream(
                    api,
                    settings,
                    job,
                    PreparedFileDispatch {
                        plan: &plan,
                        sources: *sources,
                    },
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::InstantId => {
                // InstantID identity-preserving character image (sc-3345): single identity or
                // grouped angle/pose sets, on RealVisXL + IdentityNet + the native face stack.
                generate_instantid_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::PulidFlux => {
                // PuLID-FLUX face-identity character image (sc-3344): FLUX.1-dev backbone +
                // EVA/IDFormer/CA injection via the native face stack, one image per seed.
                generate_pulid_flux_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::SdxlAdvanced => {
                // SDXL reference (IP-Adapter) / img2img edit / inpaint / outpaint (epic 3041,
                // sc-3060) → the engine's advanced conditioning paths.
                generate_sdxl_advanced_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::SensenovaEdit => {
                // SenseNova-U1 instruction edit + Character Studio on the unified
                // `sensenova_u1_8b` / `_fast` ids (sc-3900).
                generate_sensenova_edit_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Bernini => {
                // Bernini still-image companion (sc-5424): t2i / i2i on the `bernini_image` id,
                // routed to the same `engine_id:"bernini"` planner+renderer with `frames:1`.
                generate_bernini_image_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::PoseControlBaseMissing => {
                // A strict-pose job on a WIRED MLX pose family (`WIRED_MLX_POSE_FAMILIES`) whose control
                // base/overlay snapshot is NOT installed (its `…_control_available` weight-gate failed, so
                // it fell through). Refuse loudly rather than silently rendering an unconditioned image via
                // the plain MLX lane and dropping the poses (sc-11796 for krea, generalized to every wired
                // family in sc-11814) — the MLX twin of the candle `PoseControlBaseMissing` reject.
                return Err(WorkerError::InvalidPayload(format!(
                    "strict pose (advanced.poses) requested for model '{}', but its control base snapshot \
                     is not installed — refusing rather than silently generating an unconditioned image; \
                     install the control base model to enable strict-pose generation",
                    request.model
                )));
            }
            ImageRoute::PoseReject => {
                // No-silent-T2I (sc-5968): a strict-pose job on an MLX model with NO pose-control lane
                // (e.g. a plain `sdxl` pose job with no reference — SDXL identity-pose ships via InstantID /
                // IP-Adapter) that `mlx_available` would otherwise render as plain txt2img, dropping the
                // poses. Refuse loudly — the MLX twin of the candle `PoseReject` reject.
                return Err(WorkerError::InvalidPayload(format!(
                    "strict pose (advanced.poses) is not supported for model '{}' on the MLX backend — \
                     refusing rather than silently generating an unconditioned image (wired MLX pose \
                     families: {}; SDXL identity-pose runs via InstantID)",
                    request.model,
                    WIRED_MLX_POSE_FAMILIES.join(", ")
                )));
            }
            ImageRoute::Mlx => {
                generate_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
        }
        true
    } else {
        false
    };
    // Windows/CUDA candle execution path (sc-3675, epic 3672). The macOS dispatch above is MLX-bound;
    // this branch executes the single route selected by `resolve_candle_image_route`, covering the
    // generic registered-generator stream plus the bespoke edit, reference, identity, control,
    // imported, and ComfyUI lanes. Every route uses the same neutral assetWrites/progress/cancellation
    // harness. Gated on `backend_candle_enabled` (default off), so disabling Candle preserves the stub
    // behavior.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let handled = match route {
        Some(route) => {
            match route.kind() {
                // InstantID (sc-5491, epic 5480): the candle InstantID provider's bespoke path (the
                // off-Mac sibling of the macOS `ImageRoute::InstantId` arm).
                CandleImageRoute::InstantId => {
                    generate_instantid_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // SDXL img2img / inpaint / outpaint edit (sc-5487) — diverted before the txt2img arm
                // because `sdxl`/`realvisxl` ARE candle txt2img ids (an `edit_image` job would otherwise
                // be caught there and lose the source/mask). Disjoint from the IP-Adapter lane.
                CandleImageRoute::SdxlEdit => {
                    generate_candle_sdxl_edit_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // FLUX.2-klein reference / img2img edit (sc-5487) — `flux2_klein_9b` IS a candle txt2img
                // id, so an `edit_image` job must divert here first. No torch path for klein edit.
                CandleImageRoute::Flux2Edit => {
                    generate_candle_flux2_edit_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Qwen-Image-Edit reference / dual-latent edit (sc-5487) — `qwen_image_edit` is its own
                // model id, routed to the bespoke candle QwenEdit stream (disjoint from the qwen control
                // lane, which is `qwen_image` + `advanced.poses`).
                CandleImageRoute::QwenEdit => {
                    generate_candle_qwen_edit_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Z-Image img2img / edit (sc-6595) — `z_image_turbo` IS a candle txt2img id, so an
                // `edit_image` job must divert here first (disjoint from the Z-Image control lane).
                CandleImageRoute::ZimageEdit => {
                    // The catalog alias resolves to the registered Turbo provider so edit requests
                    // participate in the same shared request-scope memory lifecycle as text-to-image.
                    generate_candle_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // In-place ComfyUI Z-Image base (sc-10668, epic 10451): an `external_base_*` id whose
                // forwarded row carries the DiT/TE/VAE component paths — render the user's ComfyUI weights
                // in place via `runtime_cuda::providers::z_image::load_from_comfyui_components`.
                CandleImageRoute::ZimageComfyui => {
                    let PreparedCandleImageRoute::ZimageComfyui(sources) = route else {
                        unreachable!("Z-Image ComfyUI route missing its prepared sources")
                    };
                    generate_candle_zimage_comfyui_stream(
                        api,
                        settings,
                        job,
                        PreparedFileDispatch {
                            plan: &plan,
                            sources: *sources,
                        },
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::QwenImageComfyui => {
                    let PreparedCandleImageRoute::QwenImageComfyui(sources) = route else {
                        unreachable!("Qwen ComfyUI route missing its prepared sources")
                    };
                    generate_candle_qwen_comfyui_stream(
                        api,
                        settings,
                        job,
                        PreparedFileDispatch {
                            plan: &plan,
                            sources: *sources,
                        },
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // In-place ComfyUI FLUX.2-dev fp8-mixed base (sc-10680, epic 10451): an `external_base_*`
                // id whose forwarded row carries the DiT component path — render the user's ComfyUI
                // weights in place via `runtime_cuda::providers::flux2::load_from_comfyui_dit` (inline-scale fp8 dequant
                // + BFL→diffusers remap; TE/VAE/tokenizer from a resident FLUX.2-dev snapshot).
                CandleImageRoute::Flux2Comfyui => {
                    let PreparedCandleImageRoute::Flux2Comfyui(sources) = route else {
                        unreachable!("FLUX.2 ComfyUI route missing its prepared sources")
                    };
                    generate_candle_flux2_comfyui_stream(
                        api,
                        settings,
                        job,
                        PreparedFileDispatch {
                            plan: &plan,
                            sources: *sources,
                        },
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Bernini still-image companion (sc-10996, epic 6562): `bernini_image` t2i / i2i on the
                // `engine_id:"bernini"` planner+renderer with `frames:1` — the Windows/CUDA sibling of the
                // macOS `ImageRoute::Bernini` arm. NOT `is_candle_engine` (its engine is `Modality::Video`),
                // so it has its own bespoke stream that forces `frames:1` + the engine task string, exactly
                // like the MLX `generate_bernini_image_stream`.
                CandleImageRoute::Bernini => {
                    generate_candle_bernini_image_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // SDXL IP-Adapter-Plus reference conditioning (sc-5488) — diverted before the txt2img arm
                // (else the reference silently drops on the shared `sdxl`/`realvisxl` txt2img id).
                CandleImageRoute::SdxlIpAdapter => {
                    generate_candle_sdxl_ipadapter_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Kolors IP-Adapter-Plus reference conditioning (sc-5488).
                CandleImageRoute::KolorsIpAdapter => {
                    generate_candle_kolors_ipadapter_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // FLUX XLabs IP-Adapter reference conditioning (sc-5872).
                CandleImageRoute::FluxIpAdapter => {
                    generate_candle_flux_ipadapter_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // PuLID-FLUX face identity (sc-5492) — `pulid_flux_dev` is its own model id (never an
                // `is_candle_engine` txt2img id), routed to the bespoke candle PulidFlux stream.
                CandleImageRoute::Pulid => {
                    generate_candle_pulid_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Qwen-Image strict-pose ControlNet (sc-5489) — diverted before the txt2img arm (else the
                // poses silently drop on the shared `qwen_image` txt2img id).
                CandleImageRoute::QwenControl => {
                    generate_candle_qwen_control_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Kolors strict-pose ControlNet (sc-5489).
                CandleImageRoute::KolorsControl => {
                    generate_candle_kolors_control_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Z-Image strict-pose Fun-ControlNet (sc-5489).
                CandleImageRoute::ZimageControl => {
                    generate_candle_zimage_control_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // FLUX.2-dev strict-pose Fun-Controlnet-Union (sc-7736, epic 6564) — `flux2_dev` +
                // `advanced.poses` is the bespoke candle Flux2Control lane, diverted before the txt2img arm.
                CandleImageRoute::Flux2Control => {
                    generate_candle_flux2_control_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // FLUX.1-dev strict-control Shakker Union-Pro-2.0 (sc-8412, epic 8236) — `flux_dev` +
                // `advanced.poses` is the bespoke candle Flux1DevControl lane, diverted before the txt2img arm.
                CandleImageRoute::Flux1Control => {
                    generate_candle_flux1_control_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Krea 2 pose-ControlNet (sc-8464, epic 8459) — `krea_2_turbo` + `advanced.poses` is the
                // bespoke candle Krea2Control lane (a trained control-branch overlay on the frozen Turbo
                // base), diverted before the registry txt2img arm.
                CandleImageRoute::KreaControl => {
                    generate_candle_krea_control_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Krea 2 Kontext-style dual-conditioned edit (epic 10871) — `krea_2_raw` + `edit_image` +
                // a source, routed to the bespoke candle KreaEdit stream (disjoint from the Krea control
                // lane, which is `krea_2_turbo` + `advanced.poses`).
                CandleImageRoute::KreaEdit => {
                    generate_candle_krea_edit_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Krea 2 Raw t2i + an explicit `advanced.phases` list (epic 13879 S4, sc-13884; candle
                // sc-13887) → the multi-phase denoise driver: ONE Raw trajectory / global sigma schedule,
                // per-phase guidance (CFG on/off) + per-phase toggling of the job's load-time LoRA stack (by
                // index). Reference/edit/pose/PiD shapes are rejected loudly before the load (renders from
                // pure noise). Takes precedence over the S3 turbo-on-Raw regime. The candle engine honors
                // the backend-agnostic `GenerationRequest::phases` (inference PR #204); the handler is the
                // SAME backend-neutral `generate_krea_multiphase_stream` the macOS `KreaMultiPhase` arm runs.
                CandleImageRoute::KreaMultiPhase => {
                    generate_krea_multiphase_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Krea 2 Raw t2i + the accelerator (turbo) LoRA (sc-13882) → the distilled Turbo sampling
                // regime (fixed mu 1.15 / ~8 steps / CFG-off) on the Raw base + LoRA additive (epic 13879
                // S3, sc-13883; candle sc-13887). Routes to the `krea_2_turbo` candle engine while loading
                // the Raw weights — the engine keys the regime on that descriptor id (inference PR #204).
                // The handler is the SAME backend-neutral `generate_krea_turbo_on_raw_stream` the macOS
                // `KreaTurboOnRaw` arm runs.
                CandleImageRoute::KreaTurboOnRaw => {
                    generate_krea_turbo_on_raw_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // Imported/user Krea 2 single-file t2i (sc-14023): pair the imported bare DiT with
                // the resident Krea base tier and load it through the selected runtime's native-file
                // entrypoint. The resolver has already proved this is a non-builtin, single-file,
                // unconditioned request; keep it distinct from the builtin registry path.
                CandleImageRoute::KreaImported => {
                    let PreparedCandleImageRoute::KreaImported(sources) = route else {
                        unreachable!("Krea imported route missing its prepared sources")
                    };
                    generate_krea_imported_stream(
                        api,
                        settings,
                        job,
                        PreparedFileDispatch {
                            plan: &plan,
                            sources: *sources,
                        },
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::KreaImportedControl => {
                    generate_krea_imported_control_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::MageFinetuned => {
                    let PreparedCandleImageRoute::MageFinetuned(transformer) = route else {
                        unreachable!("Mage fine-tuned route missing its prepared transformer")
                    };
                    generate_mage_finetuned_stream(
                        api,
                        settings,
                        job,
                        *transformer,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::SdxlImported => {
                    let PreparedCandleImageRoute::SdxlImported(sources) = route else {
                        unreachable!("SDXL imported route missing its prepared sources")
                    };
                    generate_sdxl_imported_stream(
                        api,
                        settings,
                        job,
                        PreparedFileDispatch {
                            plan: &plan,
                            sources: *sources,
                        },
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
                // No-silent-T2I (sc-5968): a strict-pose job on a candle model with NO pose lane (e.g.
                // sdxl) must be REJECTED with a clear error, not silently rendered as plain txt2img (poses
                // dropped) and not rerouted. The candle worker CLAIMS these (jobs_store
                // `image_job_candle_pose_reject`) precisely to fail them loudly here. SDXL identity-pose
                // ships via InstantID; the wired candle pose families are `WIRED_CANDLE_POSE_FAMILIES`.
                CandleImageRoute::PoseReject => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "strict pose (advanced.poses) is not supported for model '{}' on the candle backend — \
                         refusing rather than silently generating an unconditioned image (wired candle pose \
                         families: {}; SDXL identity-pose runs via InstantID)",
                        request.model,
                        WIRED_CANDLE_POSE_FAMILIES.join(", ")
                    )));
                }
                // No-silent-T2I (sc-11171, F-008): a strict-pose job on a WIRED candle pose family whose
                // control base snapshot is NOT installed (the family's `…_control_available` weight-gate
                // failed, so it fell through to here). The scheduler routed it to candle weight-blind
                // (`zimage_control_candle_eligible` & siblings check only the payload), so REFUSE loudly
                // rather than silently rendering plain txt2img and dropping the poses.
                CandleImageRoute::PoseControlBaseMissing => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "strict pose (advanced.poses) requested for model '{}' on the candle backend, but \
                         its control base snapshot is not installed — refusing rather than silently \
                         generating an unconditioned image; install the control base model to enable \
                         strict-pose generation",
                        request.model
                    )));
                }
                // Registry-driven candle generation. Mage Edit is named separately by the resolver so
                // an edit without its required source can never fall through as plain T2I; both variants
                // use the same generic stream once their request shapes are resolved.
                CandleImageRoute::MageEdit
                | CandleImageRoute::SenseNovaEdit
                | CandleImageRoute::KolorsEdit
                | CandleImageRoute::CandleTxt2Img => {
                    generate_candle_stream(
                        api,
                        settings,
                        job,
                        &plan,
                        &project_path,
                        backend,
                        &mut asset_writes,
                    )
                    .await?;
                }
            }
            true
        }
        // Candle disabled (default) or no candle engine matched → stub exactly as before.
        None => false,
    };
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let handled = false;

    // An MLX-routed model id whose weights/snapshot didn't resolve must fail
    // loudly with a precise re-download error instead of completing the job
    // with procedural stub output (sc-4176, epic 3482 "unsupported jobs error
    // loudly"). `mlx_available` is the last dispatch arm, so reaching here
    // with a known engine model means exactly that its weights are unusable.
    // Model ids outside the engine families still stub (test models,
    // not-yet-ported families, non-macOS lanes).
    #[cfg(target_os = "macos")]
    if !handled {
        if let Some(gap) = mlx_weights_gap(&request, settings) {
            return Err(WorkerError::InvalidPayload(gap));
        }
    }

    if !handled {
        if request.hires_fix.enabled {
            return Err(WorkerError::InvalidPayload(
                "Hires.fix requires a native image engine with img2img support; this job resolved only to the procedural stub."
                    .to_owned(),
            ));
        }
        generate_stub_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
    }

    // sc-8091: Image Studio "Upscale" toggle. The native worker never ported the Python inline-upscale
    // path, so the UI's `upscale` request was silently dropped (images came out at the base size). Mirror
    // Python: after the base images land, upscale each with the selected engine and append a second
    // "(Nx upscaled)" asset. Gated to where the upscaler engines compile (macOS / candle).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    if request.upscale.enabled {
        apply_inline_upscale(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
    }

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Generated {} image(s).", plan.image_count),
            Some(streaming_result(&plan, &asset_writes)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Procedural stub generation (sc-3020): a deterministic per-seed gradient per image.
async fn generate_stub_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    for index in 0..request.count as usize {
        check_cancel(api, &job.id, "Image generation canceled by user.").await?;
        let seed = resolve_seed(request, index);
        let pixels = stub_rgb8(request.width, request.height, seed);
        // Encode + write the asset PNG off the async runtime thread (sc-8909 / F-107).
        let plan_for_task = plan.clone();
        let raw_settings = stub_raw_settings(request);
        let (width, height) = (request.width, request.height);
        let project_path_for_task = project_path.to_owned();
        let fact = tokio::task::spawn_blocking(move || {
            write_image_asset(
                &plan_for_task,
                index,
                seed,
                width,
                height,
                pixels,
                STUB_ADAPTER,
                raw_settings,
                &project_path_for_task,
            )
        })
        .await
        .map_err(|error| crate::task_join_error("stub image asset write task", error))??;
        asset_writes.push(Value::Object(fact));
        let progress = 0.1 + 0.85 * ((index + 1) as f64 / request.count as f64);
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Running,
                ProgressStage::Generating,
                progress,
                &format!("Generated image {}/{}.", index + 1, request.count),
                Some(streaming_result(plan, asset_writes)),
                backend,
            ),
        )
        .await?;
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    }
    Ok(())
}

/// Per-job invariants shared across every image in the generation set.
///
/// `Clone` so the per-image asset writers can move an owned copy into a `spawn_blocking` PNG-encode
/// task (sc-8909 / F-107) — the plan is a few strings + one small generation-set `Value`, negligible
/// next to the encode it hands off the async runtime thread.
#[derive(Clone)]
pub(crate) struct ImagePlan {
    pub(crate) request: ImageRequest,
    pub(crate) genset_id: String,
    pub(crate) created_at: String,
    pub(crate) family: String,
    pub(crate) slug: String,
    pub(crate) generation_set: Value,
    /// Backend adapter label selected by the resolved dispatch route. Keeping it in the plan makes
    /// generation-set telemetry use the same one-time route decision as per-asset generation.
    pub(crate) adapter: &'static str,
    /// Number of images this job produces. Usually `request.count`, but a FLUX.2 angle
    /// set is 11 and a pose set is the pose count (sc-3030) — the generation set's
    /// `count`/`expectedCount` reflect this so the gallery streams against the real
    /// total, not the requested `count`.
    image_count: u32,
    /// The job's raw `payload_json`, or `None` when nothing should be embedded (epic 15945,
    /// sc-15948). One field for both halves of the decision, resolved ONCE per job by
    /// [`workflow_source`], so the two write seams that read it cannot disagree about whether the
    /// user opted out — and a `None` here means both take the byte-identical `save_with_format`
    /// path.
    ///
    /// The RAW payload rather than a re-serialization of [`ImageRequest`]: the envelope's
    /// allow-list is defined against the payload's own keys (`sceneworks_core::workflow_share`),
    /// and the epic's decision is to source from `jobs.payload_json` because the recipe is lossy.
    /// `Arc` because every per-image asset writer clones the plan into a `spawn_blocking` encode
    /// task and a payload carries the resolved model manifest.
    pub(crate) workflow_source: Option<Arc<JsonObject>>,
    /// Worker-proven SHA-256 of the exact imported checkpoint selected by the resolved route.
    /// Never populated from the request payload.
    pub(crate) model_hash: Option<String>,
    /// Worker-resolved LoRAs in inference order. Names come from exact filenames, weights use the
    /// same parser as the engine adapter specs, and hashes come from those exact files.
    pub(crate) loras: Vec<WorkflowLora>,
}

/// Resolve the exact single-file checkpoint for an imported Krea/SDXL route. The adapter label is
/// the worker's route decision, so a client cannot opt an arbitrary manifest path into attribution.
fn imported_checkpoint_file_for_share(
    adapter: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> Option<PathBuf> {
    if !matches!(
        adapter,
        "mlx_krea_imported" | "candle_krea_imported" | "mlx_sdxl_imported" | "candle_sdxl_imported"
    ) {
        return None;
    }
    let raw_path = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .or_else(|| {
            request
                .model_manifest_entry
                .get("paths")
                .and_then(|paths| paths.get("model"))
        })
        .and_then(Value::as_str)?;
    let path = crate::paths::normalize_app_managed_model_path(
        settings,
        raw_path,
        "Imported checkpoint attribution",
    )
    .ok()?;
    if path.is_file() {
        return path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
            .then_some(path);
    }
    let mut found = None;
    for entry in std::fs::read_dir(path).ok()?.filter_map(Result::ok) {
        let candidate = entry.path();
        if candidate.is_file()
            && candidate
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
        {
            if found.is_some() {
                return None;
            }
            found = Some(candidate);
        }
    }
    found
}

fn cached_model_hash_for_file(file: &Path, marker: &JsonObject) -> Option<String> {
    let identity = model_file_identity(file)?;
    (marker.get("modelFileName").and_then(Value::as_str) == Some(identity.name.as_str())
        && marker.get("modelFileBytes").and_then(Value::as_u64) == Some(identity.bytes)
        && marker.get("modelFileModifiedNanos").and_then(Value::as_str)
            == Some(identity.modified_nanos.as_str()))
    .then(|| {
        marker
            .get("modelFileSha256")
            .and_then(Value::as_str)
            .and_then(normalize_sha256)
    })
    .flatten()
}

/// First non-empty of installedPath/sourcePath/path/source.path on a LoRA spec.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
pub(crate) fn lora_path(lora: &Value) -> Option<PathBuf> {
    for key in ["installedPath", "sourcePath", "path"] {
        if let Some(value) = lora
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(PathBuf::from(value));
        }
    }
    lora.get("source")
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The exact adapter filename a manifest declares when its LoRA path is a directory.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
pub(crate) fn declared_adapter_file(lora: &Value) -> Option<&str> {
    lora.get("files")
        .and_then(Value::as_array)
        .and_then(|files| files.first())
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// Resolve the exact adapter file inference will load. Attribution calls this same function before
/// hashing, so the digest cannot drift onto a sibling checkpoint or an unconfined client path.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
fn resolve_adapter_file(lora: &Value, settings: &Settings) -> WorkerResult<PathBuf> {
    let raw = lora_path(lora)
        .ok_or_else(|| WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned()))?;
    let path = crate::normalize_app_managed_lora_path(settings, &raw)?;
    let file = if path.is_dir() {
        crate::resolve_adapter_in_dir(&path, declared_adapter_file(lora)).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "LoRA has no .safetensors under {}",
                path.display()
            ))
        })?
    } else {
        path
    };
    if !file.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "LoRA file is missing: {}",
            file.display()
        )));
    }
    Ok(file)
}

/// Resolve and pin the exact adapter entry inference will load. Directory-valued imports are first
/// confined as directories, then their selected child is independently pinned and confined so a
/// child symlink cannot inherit trust from its parent.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_prepared_adapter_file(
    lora: &Value,
    settings: &Settings,
) -> WorkerResult<gen_core::PinnedWeightsFile> {
    let raw = lora_path(lora)
        .ok_or_else(|| WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned()))?;
    let confined = crate::normalize_app_managed_lora_path(settings, &raw)?;
    let candidate = if confined.is_dir() {
        let directory = confined;
        crate::resolve_adapter_in_dir(&directory, declared_adapter_file(lora)).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "LoRA has no .safetensors under {}",
                directory.display()
            ))
        })?
    } else {
        raw
    };
    crate::paths::pin_app_managed_model_file(settings, &candidate, "LoRA file")
}

/// The exact weight parser used by every adapter lane and by the gallery attribution renderer.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
fn lora_weight(lora: &Value) -> f64 {
    lora.get("weight")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(0.8)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
fn existing_lora_install_marker(lora: &Value, settings: &Settings, file: &Path) -> Option<PathBuf> {
    // Explicit HF catalog downloads execute from the shared hub cache, while their durable receipt
    // lives under data/loras/<catalog-id>. Prefer that trusted app-owned receipt when it names the
    // same repo; it is also the marker the API reads when resolving imported hashes.
    let source = lora.get("source").and_then(Value::as_object);
    let provider = source
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str);
    let repo = source
        .and_then(|source| source.get("repo"))
        .or_else(|| lora.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if provider == Some("huggingface") {
        if let (Some(id), Some(repo)) = (
            lora.get("id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty()),
            repo,
        ) {
            let candidate = settings
                .data_dir
                .join("loras")
                .join(crate::paths::safe_download_dir(id))
                .join(INSTALL_MARKER);
            let matches_repo = std::fs::read(&candidate)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|marker| {
                    marker
                        .get("repo")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .as_deref()
                == Some(repo);
            if matches_repo {
                return Some(candidate);
            }
        }
    }
    file.ancestors()
        .take(8)
        .map(|directory| directory.join(INSTALL_MARKER))
        .find(|candidate| candidate.is_file())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
fn cached_lora_hash_for_file(file: &Path, marker: &JsonObject) -> Option<String> {
    let identity = model_file_identity(file)?;
    (marker.get("loraFileName").and_then(Value::as_str) == Some(identity.name.as_str())
        && marker.get("loraFileBytes").and_then(Value::as_u64) == Some(identity.bytes)
        && marker.get("loraFileModifiedNanos").and_then(Value::as_str)
            == Some(identity.modified_nanos.as_str()))
    .then(|| {
        marker
            .get("loraFileSha256")
            .and_then(Value::as_str)
            .and_then(normalize_sha256)
    })
    .flatten()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
fn civitai_lora_key(file: &Path, used: &mut std::collections::HashSet<String>) -> String {
    let stem = file
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("lora");
    let mut base = String::with_capacity(stem.len().min(120));
    let mut last_was_separator = false;
    for character in stem.chars().take(120) {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            base.push(character);
            last_was_separator = false;
        } else if !last_was_separator {
            base.push('_');
            last_was_separator = true;
        }
    }
    let base = base.trim_matches('_');
    let base = if base.is_empty() { "lora" } else { base };
    let mut candidate = base.to_owned();
    let mut suffix = 2_u32;
    while !used.insert(candidate.clone()) {
        candidate = format!("{base}_{suffix}");
        suffix += 1;
    }
    candidate
}

/// Hash the exact resolved adapter bytes, using an existing install marker as a cheap cache when
/// available. A missing/malformed marker or a hashing failure loses attribution only; it never
/// turns a successful generation into an error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
async fn trusted_lora_hash(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    lora: &Value,
    file: &Path,
) -> Option<String> {
    let marker_path = existing_lora_install_marker(lora, settings, file);
    let mut marker = match marker_path.as_ref() {
        Some(path) => tokio::fs::read(path)
            .await
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok()),
        None => None,
    };
    if let Some(hash) = marker
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|object| cached_lora_hash_for_file(file, object))
    {
        return Some(hash);
    }

    let identity_before = model_file_identity(file)?;
    let hash = match sha256_file(api, settings, &job.id, file).await {
        Ok(hash) => hash,
        Err(error) => {
            tracing::warn!(path = %file.display(), %error, "LoRA attribution hash failed");
            return None;
        }
    };
    let identity_after = model_file_identity(file)?;
    if identity_after != identity_before {
        tracing::warn!(path = %file.display(), "LoRA changed while attribution hash was being computed");
        return None;
    }

    if let (Some(marker_path), Some(object)) =
        (marker_path, marker.as_mut().and_then(Value::as_object_mut))
    {
        object.insert(
            "loraFileName".to_owned(),
            Value::String(identity_after.name),
        );
        object.insert(
            "loraFileBytes".to_owned(),
            Value::Number(identity_after.bytes.into()),
        );
        object.insert(
            "loraFileModifiedNanos".to_owned(),
            Value::String(identity_after.modified_nanos),
        );
        object.insert("loraFileSha256".to_owned(), Value::String(hash.clone()));
        if let Ok(bytes) = serde_json::to_vec_pretty(&marker) {
            if let Err(error) = tokio::fs::write(&marker_path, bytes).await {
                tracing::warn!(path = %marker_path.display(), %error, "could not cache LoRA attribution hash");
            }
        }
    }
    Some(hash)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
#[cfg_attr(test, allow(dead_code))]
async fn trusted_loras_for_share(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> Vec<WorkflowLora> {
    let mut used_names = std::collections::HashSet::new();
    let mut loras = Vec::with_capacity(request.loras.len());
    for raw in &request.loras {
        let file = match resolve_adapter_file(raw, settings) {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(%error, "LoRA attribution could not resolve the inference file");
                continue;
            }
        };
        let name = civitai_lora_key(&file, &mut used_names);
        let weight = lora_weight(raw);
        let hash = trusted_lora_hash(api, settings, job, raw, &file).await;
        if let Some(lora) = trusted_lora_for_share(raw, name, weight, hash) {
            loras.push(lora);
        }
    }
    loras
}

/// Read the digest retained by model import, or backfill one legacy marker once. Hashing failure is
/// attribution failure, not generation failure: the image remains usable and merely lacks a card.
async fn trusted_imported_model_hash(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    adapter: &str,
    request: &ImageRequest,
) -> Option<String> {
    let file = imported_checkpoint_file_for_share(adapter, request, settings)?;
    let marker_path = file.parent()?.join(INSTALL_MARKER);
    let bytes = tokio::fs::read(&marker_path).await.ok()?;
    let mut marker = serde_json::from_slice::<Value>(&bytes).ok()?;
    let object = marker.as_object_mut()?;
    if let Some(hash) = cached_model_hash_for_file(&file, object) {
        return Some(hash);
    }

    let identity_before = model_file_identity(&file)?;
    let hash = match sha256_file(api, settings, &job.id, &file).await {
        Ok(hash) => hash,
        Err(error) => {
            tracing::warn!(path = %file.display(), %error, "checkpoint attribution hash failed");
            return None;
        }
    };
    let identity_after = model_file_identity(&file)?;
    if identity_after != identity_before {
        tracing::warn!(path = %file.display(), "checkpoint changed while attribution hash was being computed");
        return None;
    }
    object.insert(
        "modelFileName".to_owned(),
        Value::String(identity_after.name),
    );
    object.insert(
        "modelFileBytes".to_owned(),
        Value::Number(identity_after.bytes.into()),
    );
    object.insert(
        "modelFileModifiedNanos".to_owned(),
        Value::String(identity_after.modified_nanos),
    );
    object.insert("modelFileSha256".to_owned(), Value::String(hash.clone()));
    match serde_json::to_vec_pretty(&marker) {
        Ok(bytes) => {
            if let Err(error) = tokio::fs::write(&marker_path, bytes).await {
                tracing::warn!(path = %marker_path.display(), %error, "could not cache checkpoint attribution hash");
            }
        }
        Err(error) => {
            tracing::warn!(path = %marker_path.display(), %error, "could not serialize checkpoint attribution marker");
        }
    }
    Some(hash)
}

/// The workflow-envelope source for one job, or `None` for "write the file exactly as today"
/// (sc-15948).
///
/// Two independent reasons to embed nothing, deliberately collapsed into one `Option` at the top
/// of the job rather than re-decided at each write:
///
/// * the user turned `embedWorkflowInImages` off (read live off the config dir — see
///   `sceneworks_core::app_paths::embed_workflow_in_images`);
/// * there is no payload to describe. A stub or dry-run write with an empty payload has nothing to
///   say about how the image was made, and an envelope of bare fallbacks is worse than no chunk —
///   so absence is an absence, never an error.
///
/// Each branch logs its reason at `debug`. Collapsing the two into one `Option` is right for the
/// callers — both mean "write the file exactly as today" — but it leaves "there is no chunk in this
/// PNG" with two indistinguishable causes, and a user asking why an image has no recipe needs to
/// know whether they turned it off or the payload was empty. One line each is the whole diagnostic.
pub(crate) fn workflow_source(
    settings: &Settings,
    job_payload: &JsonObject,
) -> Option<Arc<JsonObject>> {
    if job_payload.is_empty() {
        tracing::debug!(
            reason = "empty_job_payload",
            "not embedding a workflow: the job carries no payload to describe"
        );
        return None;
    }
    if !sceneworks_core::app_paths::embed_workflow_in_images(&settings.config_dir) {
        tracing::debug!(
            reason = "preference_off",
            config_dir = %settings.config_dir.display(),
            "not embedding a workflow: `embedWorkflowInImages` did not resolve to true"
        );
        return None;
    }
    tracing::debug!(
        keys = job_payload.len(),
        "embedding the sanitized workflow in every image this job writes"
    );
    Some(Arc::new(job_payload.clone()))
}

/// The workflow envelope for an inline-upscaled variant (sc-15948): the generation's own payload
/// with the APPLIED pass overlaid.
///
/// The inline upscale is a sub-step of the generate job, not a job of its own, so there is no second
/// payload to build from — but the base generation's envelope alone would describe an image that was
/// never written. The overlay is what makes the difference honest, and it is the *applied* record
/// rather than the requested one: `write_upscaled_asset`'s caller normalizes the engine id (anything
/// unknown, including the dropped `aura-sr`, becomes `real-esrgan`) and clamps the factor to 2 or 4,
/// and it is that pass the file came out of. It is the same `Value` the fact's
/// `rawAdapterSettings.upscale` records, so a shared image and its sidecar cannot disagree.
///
/// Geometry deliberately stays the GENERATION geometry rather than the upscaled file's. The envelope
/// is a recipe: "render 1024² then upscale 2x" replays to this image, where "render 2048² then
/// upscale 2x" would replay to something twice the size.
///
/// Compiled under `test` on every platform (unlike `write_upscaled_asset`, which needs an upscaler
/// backend) so the lineage contract is tested on every build, not only where candle or MLX compiles.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
pub(crate) fn upscaled_workflow_share(
    request: &ImageRequest,
    base_fact: &JsonObject,
    job_payload: &JsonObject,
    upscale_record: &Value,
) -> Option<sceneworks_core::workflow_share::WorkflowShare> {
    let base_u32 = |key: &str| {
        base_fact
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    };
    let mut overlay = job_payload.clone();
    overlay.insert("upscale".to_owned(), upscale_record.clone());
    // `upscale` is the ONLY key overlaid. Everything else — the payload's own `width`/`height`
    // included — is left exactly as the base image's own envelope reads it, so two envelopes
    // describing one generation cannot disagree about the render that produced it. The base fact
    // supplies the geometry fallback for a payload that omitted it, which is the actual written
    // size of the image this variant was upscaled FROM.
    embeddable_workflow_share(
        &WorkflowAssetFacts {
            mode: request.mode.clone(),
            model: request.model.clone(),
            prompt: request.prompt.clone(),
            negative_prompt: request.negative_prompt.clone(),
            seed: base_fact.get("seed").and_then(Value::as_i64).unwrap_or(0),
            width: base_u32("width"),
            height: base_u32("height"),
        },
        &overlay,
    )
}

/// The workflow envelope for the standalone detail pass (sc-15948).
///
/// Unlike the inline upscale, `image_detail` IS its own job with its own payload, so nothing here is
/// inherited from the generation that produced the source image — there is no base-generation
/// envelope in play at all. What the payload cannot supply, the resolved pass does: the mode, the
/// SDXL backbone that actually ran, the detail prompt/negative (which live under `advanced` and are
/// what the model saw, not the payload's absent top-level prompt), the seed and the output geometry.
/// `advanced.strength` and `advanced.cnScale` — the two knobs the Detail UI exposes — travel through
/// the sc-15946 allow-list, and the source image rides as an input SHAPE rather than as the local
/// asset id `sourceAssetId` names.
///
/// Compiled under `test` everywhere for the same reason as [`upscaled_workflow_share`]:
/// `image_jobs/detail.rs` compiles on macOS only, and the lineage contract should not go untested on
/// every other platform.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
pub(crate) fn detail_workflow_share(
    job_payload: &JsonObject,
    model: &str,
    prompt: &str,
    negative_prompt: &str,
    seed: i64,
    width: u32,
    height: u32,
) -> Option<sceneworks_core::workflow_share::WorkflowShare> {
    embeddable_workflow_share(
        &WorkflowAssetFacts {
            mode: "image_detail".to_owned(),
            model: model.to_owned(),
            prompt: prompt.to_owned(),
            negative_prompt: negative_prompt.to_owned(),
            seed,
            width: Some(width),
            height: Some(height),
        },
        job_payload,
    )
}

/// The workflow envelope for the STANDALONE `image_upscale` job (sc-15948).
///
/// The fourth write seam, and the one carrying the asset class users share most: `single_child_asset.rs`
/// wrote this PNG with a bare `save_with_format` and no chunk at all, so an upscaled image — the
/// version people actually post — was the one image in the app with no recipe inside it. The story's
/// rule for a derived pass is "where the pass is a distinct job, use that job's payload", and this is
/// exactly that: its own `JobType::ImageUpscale` row with its own payload.
///
/// So, like [`detail_workflow_share`] and unlike [`upscaled_workflow_share`], nothing is inherited
/// from whatever generated the source image. There is no prompt and no model in an upscale — the
/// "model" IS the engine that ran — and `sourceAssetId` rides as an input SHAPE rather than as a
/// local id.
///
/// The pass is the APPLIED one: `engine_id` has already been canonicalized and validated by the
/// caller (`real-esrgan` / `seedvr2`; the dropped `aura-sr` is rejected outright) and `factor` has
/// already been through `resolve_image_upscale_factor`, so what lands here cannot name an engine that
/// does not exist or a factor nobody offers. `softness` is SeedVR2's knob and is `None` for
/// Real-ESRGAN, which ignores it — recording `softness: 0.0` on an engine that has no such control
/// would be inventing a fact.
///
/// Geometry is the SOURCE image's, matching [`upscaled_workflow_share`]: the envelope is a recipe, so
/// "this 1024² image, upscaled 2x" is what reproduces the file, where the written 2048² would read as
/// an instruction to upscale something already upscaled.
///
/// Compiled under `test` on every platform (unlike `upscale_jobs.rs`, which needs an upscaler
/// backend) so the contract is tested on every build rather than only where candle or MLX compiles.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
pub(crate) fn standalone_upscale_workflow_share(
    job_payload: &JsonObject,
    engine_id: &str,
    factor: u8,
    softness: Option<f32>,
    seed: i64,
    source_width: u32,
    source_height: u32,
) -> Option<sceneworks_core::workflow_share::WorkflowShare> {
    let mut upscale = json!({ "enabled": true, "engine": engine_id, "factor": factor });
    if let Some(softness) = softness {
        upscale["softness"] = json!(softness);
    }
    // `upscale` is the only key overlaid: the job payload's own `factor` / `engine` are the
    // REQUESTED values and are not envelope fields, so the record built here is the only thing that
    // describes the pass.
    let mut overlay = job_payload.clone();
    overlay.insert("upscale".to_owned(), upscale);
    embeddable_workflow_share(
        &WorkflowAssetFacts {
            mode: "image_upscale".to_owned(),
            // The engine IS the model for this pass. The payload carries no `model` key (see
            // `buildUpscaleJobBody`), so this is what travels.
            model: engine_id.to_owned(),
            prompt: String::new(),
            negative_prompt: String::new(),
            seed,
            width: Some(source_width),
            height: Some(source_height),
        },
        &overlay,
    )
}

impl ImagePlan {
    /// Test-only convenience: a plan whose image count is the request count, embedding nothing.
    /// Production always goes through [`ImagePlan::with_count`] (the FLUX.2 angle/pose sets need an
    /// effective count that differs from `request.count`).
    #[cfg(test)]
    fn new(request: &ImageRequest) -> Self {
        Self::with_count_and_adapter(request, request.count, adapter_id(request), None)
    }

    /// Build a plan whose generation set reports `image_count` images (see the field).
    pub(crate) fn with_count(
        request: &ImageRequest,
        image_count: u32,
        workflow_source: Option<Arc<JsonObject>>,
    ) -> Self {
        Self::with_count_and_adapter(request, image_count, adapter_id(request), workflow_source)
    }

    /// Build a plan with the count and adapter label selected by the already-resolved route.
    fn with_count_and_adapter(
        request: &ImageRequest,
        image_count: u32,
        adapter: &'static str,
        workflow_source: Option<Arc<JsonObject>>,
    ) -> Self {
        let genset_id = format!("genset_{}", Uuid::new_v4().simple());
        let created_at = now_rfc3339();
        let family = resolve_family(request);
        let slug = slugify(&request.prompt, "image", Some(42));
        let generation_set = json!({
            "id": genset_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": image_count,
            "createdAt": created_at,
        });
        Self {
            request: request.clone(),
            genset_id,
            created_at,
            family,
            slug,
            generation_set,
            adapter,
            image_count,
            workflow_source,
            model_hash: None,
            loras: Vec::new(),
        }
    }
}

/// Add the worker-resolved denoise count to the sanitized share envelope when the request did not
/// already carry one.
///
/// Several generation lanes choose a real default after request parsing and record it as
/// `raw_settings.numInferenceSteps`. The original workflow source is the user request, so without
/// this narrow overlay a generated PNG can omit the step count even though the worker knows exactly
/// what ran. Civitai then rejects the entire A1111 `parameters` block.
///
/// Only the single trusted field is copied, and only when `numInferenceSteps` was absent from the raw
/// request. That provenance check prevents an API caller from forging an internal telemetry value.
/// Multi-phase schedules remain untouched: one numeric A1111 `Steps` value cannot represent them.
fn resolved_steps_for_share(
    workflow_source: &JsonObject,
    raw_settings: &JsonObject,
    has_multi_phase: bool,
) -> Option<u32> {
    let source_supplied_runtime_steps = workflow_source
        .get("advanced")
        .and_then(Value::as_object)
        .is_some_and(|advanced| advanced.contains_key("numInferenceSteps"));
    if has_multi_phase || source_supplied_runtime_steps {
        return None;
    }
    raw_settings
        .get("numInferenceSteps")
        .and_then(Value::as_u64)
        .filter(|steps| *steps >= 1)
        .and_then(|steps| u32::try_from(steps).ok())
}

/// Return a sampler name that the worker recorded after resolving the actual execution path.
///
/// `resolvedSampler` is deliberately separate from the request's `advanced.sampler`: some lanes
/// ignore that request setting and choose their sampler inside the runtime. Promotion is bound to
/// the worker-selected imported-Krea adapter, not to any field in the client payload; the Krea
/// execution seam overwrites this raw fact before writing the asset. Keep both the route and value
/// vocabularies narrow until another execution lane records a proven value of its own.
fn resolved_sampler_for_share<'a>(adapter: &str, raw_settings: &'a JsonObject) -> Option<&'a str> {
    if !matches!(adapter, "mlx_krea_imported" | "candle_krea_imported") {
        return None;
    }
    raw_settings
        .get("resolvedSampler")
        .and_then(Value::as_str)
        .filter(|sampler| matches!(*sampler, "euler"))
}

/// Save image `index` (its RGB8 `pixels`) under `assets/images/` and return the flat
/// fact the API turns into an indexed asset (every key here is consumed by
/// `build_image_sidecar_parts`). Shared by the stub and real paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_image_asset(
    plan: &ImagePlan,
    index: usize,
    seed: i64,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    adapter: &str,
    raw_settings: JsonObject,
    project_path: &Path,
) -> WorkerResult<JsonObject> {
    let request = &plan.request;
    let rgb_image = image::RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| WorkerError::InvalidPayload("image buffer size mismatch".to_owned()))?;

    // Sanitize the payload-supplied model id before it becomes a path component: it
    // arrives verbatim from the untrusted job payload, and a `../` / `\` / absolute id
    // would otherwise traverse out of the project dir here (F-003 / sc-11159). rust-api
    // now rejects such ids at enqueue, but the worker is the trust boundary and must
    // re-confine — slugify neutralizes any separator/`..` to a single readable component.
    let model_slug = slugify(&request.model, "model", None);
    let filename = format!(
        "{}_{}_{}_{:04}.png",
        &plan.created_at[..10],
        model_slug,
        plan.slug,
        index + 1
    );
    let media_rel = format!("assets/images/{}/{filename}", plan.genset_id);
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = media_path.with_extension("tmp.png");
    // The one funnel every generated image goes through, and therefore the one place the sanitized
    // workflow needs embedding (epic 15945, sc-15948). `None` — embedding off, or no payload to
    // describe — routes through the same `save_with_format` call this used to make, byte for byte.
    let share = plan.workflow_source.as_deref().and_then(|payload| {
        let mut share = embeddable_workflow_share(
            &WorkflowAssetFacts {
                mode: request.mode.clone(),
                model: request.model.clone(),
                prompt: request.prompt.clone(),
                negative_prompt: request.negative_prompt.clone(),
                // THIS image's seed, not the batch base the payload carries.
                seed,
                width: Some(width),
                height: Some(height),
            },
            payload,
        )?;
        let has_multi_phase = share.advanced.contains_key("phases")
            || share.omitted.iter().any(|field| field == OMITTED_PHASES);
        if let Some(steps) = resolved_steps_for_share(payload, &raw_settings, has_multi_phase) {
            share.advanced.insert("steps".to_owned(), json!(steps));
        }
        if let Some(sampler) = resolved_sampler_for_share(adapter, &raw_settings) {
            share.advanced.insert("sampler".to_owned(), json!(sampler));
        }
        share.model_hash = plan.model_hash.clone();
        // Replace request-derived hints with the exact adapter stack the worker resolved. This is
        // the only seam that can attach hashes, so client-supplied attribution fields never win.
        share.loras = plan.loras.clone();
        // This function only ever writes a BASE render. The inline-upscale post-pass writes its
        // output through `write_upscaled_asset` and keeps the base as its own retained asset, so a
        // `upscale.enabled: true` from the request would describe, on this file, a pass this file
        // never received — and describe it in the REQUESTED terms, which `apply_inline_upscale`
        // then clamps (factor to 2/4) and normalizes (a dropped engine to `real-esrgan`). Naming a
        // dropped engine at an unoffered factor is worse than saying nothing: sc-15952 prefills the
        // studio from this. The variant's own envelope carries the pass that actually ran, which is
        // the same reasoning `upscaled_workflow_share` applies from the other side.
        share.upscale = None;
        Some(share)
    });
    write_workflow_chunk(&rgb_image, &temp_path, share.as_ref())
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let title: String = request.prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = format!(
        "{} #{}",
        if title.is_empty() {
            "Generated image"
        } else {
            title
        },
        index + 1
    );

    // The source image an edit was derived from, so the Image Viewer can offer an
    // Original↔Edited side-by-side compare (the edit stays its own asset — this is
    // lineage, not a fold). Single-source edits carry it as `sourceAssetId`; the
    // multi-image reference pickers (FLUX.2-dev multi-select, Krea two-reference)
    // send `referenceAssetIds` with no scalar source, so anchor to the first
    // reference. Non-edit jobs keep their own `sourceAssetId` (usually none).
    let source_asset_id = request.source_asset_id.clone().or_else(|| {
        if request.mode == "edit_image" {
            request.reference_asset_ids.first().cloned()
        } else {
            None
        }
    });

    let fact = json!({
        "assetId": fresh_asset_id(),
        "type": "image",
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "width": width,
        "height": height,
        "normalizedWidth": request.width,
        "normalizedHeight": request.height,
        "count": plan.image_count,
        "family": plan.family,
        "seed": seed,
        "index": index,
        "displayName": display_name,
        "createdAt": now_rfc3339(),
        "mode": request.mode,
        "model": request.model,
        "adapter": adapter,
        "prompt": request.prompt,
        "negativePrompt": request.negative_prompt,
        "loras": request.loras,
        "stylePreset": request.style_preset,
        "characterId": request.character_id,
        "characterLookId": request.character_look_id,
        "sourceAssetId": source_asset_id,
        "rawAdapterSettings": raw_settings,
    });
    Ok(fact.as_object().cloned().expect("json! object literal"))
}

/// Normalise the UI's upscale engine id to the canonical worker id. SeedVR2 stays itself;
/// everything else (`real-esrgan` / `realesrgan` / the dropped `aura-sr` / unknown) maps to
/// Real-ESRGAN, so a bad engine string never hard-fails a whole generation.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn normalize_upscale_engine(engine: &str) -> &'static str {
    match engine.trim().to_ascii_lowercase().as_str() {
        "seedvr2" => "seedvr2",
        _ => "real-esrgan",
    }
}

/// Inline upscale post-pass (sc-8091): upscale every base image the generation produced and append a
/// second "(Nx upscaled)" asset, mirroring the Python worker. Reuses the same in-memory upscalers as the
/// standalone `image_upscale` job — Real-ESRGAN via `ort`, SeedVR2 via the registry generator — both of
/// which now RESOLVE already-installed weights instead of provisioning them on first use (sc-17633 /
/// sc-17632), which is why this pass needs no HTTP client. Runs after the base images have already been
/// streamed (so they persist even if a late upscale step errors and fails the job).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn apply_inline_upscale(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let factor: u8 = if request.upscale.factor == 4 { 4 } else { 2 };
    let engine_id = normalize_upscale_engine(&request.upscale.engine);
    let softness = request.upscale.softness();
    // The generate payload carries the *generation* model's manifest, not an upscaler one; pass Null
    // so the weight resolvers fall back to the default HF repos (download-on-first-use).
    let manifest = Value::Null;
    let cancel = CancelFlag::new();

    // Snapshot the base image assets (we append the upscaled variants as we go).
    let base_facts: Vec<JsonObject> = asset_writes
        .iter()
        .filter_map(Value::as_object)
        .filter(|fact| fact.get("type").and_then(Value::as_str) == Some("image"))
        .cloned()
        .collect();
    let total = base_facts.len();

    for (i, base_fact) in base_facts.iter().enumerate() {
        check_cancel(api, &job.id, "Image upscale canceled by user.").await?;

        let media_rel = base_fact
            .get("mediaPath")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                WorkerError::InvalidPayload("upscale source asset missing mediaPath".to_owned())
            })?;
        // Decode the base image off the async runtime thread (sc-8909 / F-107).
        let source_path = project_path.join(media_rel);
        let source = tokio::task::spawn_blocking(move || {
            crate::image_decode::decode_image_any(source_path)
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Upscale source could not be loaded: {error}"
                    ))
                })
                .map(|decoded| decoded.to_rgb8())
        })
        .await
        .map_err(|error| crate::task_join_error("upscale source decode task", error))??;
        let seed = base_fact.get("seed").and_then(Value::as_i64).unwrap_or(0);

        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Running,
                ProgressStage::Running,
                0.9,
                &format!(
                    "Upscaling image {}/{total} {factor}x with {engine_id}.",
                    i + 1
                ),
                Some(streaming_result(plan, asset_writes)),
                backend,
            ),
        )
        .await?;

        let upscaled = crate::upscale_jobs::upscale_image_in_memory(
            api,
            settings,
            job,
            &manifest,
            engine_id,
            factor,
            softness,
            seed.max(0) as u64,
            source,
            &cancel,
        )
        .await?;

        // Build the upscaled asset (including the blocking PNG encode) off the async runtime thread
        // (sc-8909 / F-107).
        let plan_for_task = plan.clone();
        let base_fact_for_task = base_fact.clone();
        let engine_for_task = engine_id.to_owned();
        let project_path_for_task = project_path.to_owned();
        let fact = tokio::task::spawn_blocking(move || {
            write_upscaled_asset(
                &plan_for_task,
                &base_fact_for_task,
                &upscaled,
                &engine_for_task,
                factor,
                softness,
                &project_path_for_task,
            )
        })
        .await
        .map_err(|error| crate::task_join_error("upscaled asset write task", error))??;
        asset_writes.push(Value::Object(fact));
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    }
    Ok(())
}

/// Write the upscaled variant of a base image as its own asset (sc-8091): same metadata as the base
/// fact, but a fresh `assetId`, the `_up{factor}x` file, the upscaled dimensions, a "(Nx upscaled)"
/// display-name suffix, and a `rawAdapterSettings.upscale` record (so preset-restore reads it back).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_upscaled_asset(
    plan: &ImagePlan,
    base_fact: &JsonObject,
    upscaled: &image::RgbImage,
    engine_id: &str,
    factor: u8,
    softness: f32,
    project_path: &Path,
) -> WorkerResult<JsonObject> {
    let request = &plan.request;
    let index = base_fact.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let (width, height) = (upscaled.width(), upscaled.height());

    // Sanitize the untrusted model id before it becomes a path component (F-003 / sc-11159),
    // mirroring `write_image_asset` so the upscaled variant is confined identically.
    let model_slug = slugify(&request.model, "model", None);
    let filename = format!(
        "{}_{}_{}_{:04}_up{factor}x.png",
        &plan.created_at[..10],
        model_slug,
        plan.slug,
        index + 1
    );
    let media_rel = format!("assets/images/{}/{filename}", plan.genset_id);
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // The APPLIED pass, not the requested one. Hoisted above the write because two things now read
    // it: the embedded workflow (sc-15948) and the `rawAdapterSettings.upscale` record below, and a
    // shared image must not describe a different pass from the one the sidecar records.
    let upscale_record = if engine_id == "seedvr2" {
        json!({ "enabled": true, "engine": engine_id, "factor": factor, "softness": softness })
    } else {
        json!({ "enabled": true, "engine": engine_id, "factor": factor })
    };

    let temp_path = media_path.with_extension("tmp.png");
    // The upscaled variant is the asset users share most, so it carries the workflow that produced
    // IT rather than a stale base-generation envelope (sc-15948) — see `upscaled_workflow_share`.
    let mut share = plan
        .workflow_source
        .as_deref()
        .and_then(|payload| upscaled_workflow_share(request, base_fact, payload, &upscale_record));
    if let Some(share) = share.as_mut() {
        share.model_hash = plan.model_hash.clone();
        share.loras = plan.loras.clone();
    }
    write_workflow_chunk(upscaled, &temp_path, share.as_ref())
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let base_display = base_fact
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("Generated image");
    let display_name = format!("{base_display} ({factor}x upscaled)");

    // rawAdapterSettings: the base settings + an `upscale` record (mirrors the Python worker so the
    // gallery / preset restore can read back the engine/factor/softness).
    let mut raw_settings = base_fact
        .get("rawAdapterSettings")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    raw_settings.insert("upscale".to_owned(), upscale_record);

    let mut fact = base_fact.clone();
    fact.insert("assetId".to_owned(), json!(fresh_asset_id()));
    fact.insert("mediaPath".to_owned(), json!(media_rel));
    fact.insert("width".to_owned(), json!(width));
    fact.insert("height".to_owned(), json!(height));
    fact.insert("displayName".to_owned(), json!(display_name));
    fact.insert("createdAt".to_owned(), json!(now_rfc3339()));
    fact.insert("rawAdapterSettings".to_owned(), Value::Object(raw_settings));
    // Link the upscaled variant back to its base image using the SAME lineage keys the standalone
    // `image_upscale` job writes (upscale_jobs.rs), so the Library / Recent-Batches fold and the
    // Original↔Upscaled A/B toggle collapse the pair (sc-10117). This previously wrote a bare
    // `upscaledFrom` field that nothing read (not the web `assetVariants.js`, not `project_store`) and
    // that was dropped at sidecar-build time, so inline upscales never folded with their originals.
    let source_asset_id = base_fact.get("assetId").cloned().unwrap_or(Value::Null);
    fact.insert("sourceAssetId".to_owned(), source_asset_id.clone());
    fact.insert("parents".to_owned(), json!([source_asset_id.clone()]));
    // Preserve any base `extra` (e.g. character metadata) and layer the upscale markers on top.
    let mut extra = base_fact
        .get("extra")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    extra.insert("isUpscaled".to_owned(), json!(true));
    extra.insert("upscaledFromAssetId".to_owned(), source_asset_id);
    extra.insert("factor".to_owned(), json!(factor));
    extra.insert("engine".to_owned(), json!(engine_id));
    fact.insert("extra".to_owned(), Value::Object(extra));
    Ok(fact)
}

/// The job-result shape the API streams from: `assetWrites` + the `generationSet`
/// fact drive `persist_reported_assets` (idempotent per progress update).
///
/// ACCEPTED TRADEOFF (sc-8953 / F-151): this deep-clones the whole `asset_writes` vec into the
/// result on every call, and the generation loop calls it on each `GenEvent::Step` — so the total
/// serialization work is O(images² · steps) as `asset_writes` grows one entry per finished image.
/// At current image counts (a handful per set) and step counts this is negligible next to the
/// generation itself, so it is left as-is; if sets grow large, stream this only on `Image` /
/// `Decoding` events (where the fact set actually changes) rather than on every step.
fn streaming_result(plan: &ImagePlan, asset_writes: &[Value]) -> JsonObject {
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": plan.image_count,
        "adapter": plan.adapter,
        "model": plan.request.model,
        "generationSet": plan.generation_set,
        "assetWrites": asset_writes,
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Request-only fallback used by test plans and the backend-less stub plan. Production MLX/candle
/// plans bind their adapter from the resolved route so bespoke IDs cannot fall through to the stub.
fn adapter_id(request: &ImageRequest) -> &'static str {
    #[cfg(target_os = "macos")]
    if let Some(model) = mlx_model(&request.model) {
        return model.adapter_label();
    }
    // Windows/CUDA candle lane (sc-3678, per-engine in sc-5096): report the candle adapter for the
    // wired family so the generation-set fact matches the per-asset `adapter` the candle path writes,
    // instead of falling through to the procedural-stub label. Routing (`worker_supports_job`) only
    // lets candle-eligible txt2img jobs reach this worker, so `is_candle_engine` here implies the
    // candle path ran.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if is_candle_engine(&request.model) {
        return candle_adapter_label(&request.model);
    }
    let _ = request;
    STUB_ADAPTER
}

fn stub_raw_settings(request: &ImageRequest) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(false));
    raw
}

/// The asset `family`: the resolved model manifest entry wins (the UI sends it), else
/// the linked mlx-gen descriptor's family on macOS, else empty.
fn resolve_family(request: &ImageRequest) -> String {
    if let Some(family) = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return family.to_owned();
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(family) = crate::inference_runtime::generators()
            .find(|registration| (registration.descriptor)().id == request.model)
            .map(|registration| (registration.descriptor)().family)
        {
            return family.to_owned();
        }
    }
    String::new()
}

fn hires_fix_target_dimension(dimension: u32, upscale_by: f32) -> u32 {
    (dimension as f64 * upscale_by as f64).round() as u32
}

/// Reject unsupported Hires.fix request shapes before any model load. Hires.fix is a plain
/// text-to-image second pass: existing edit/control inputs, PiD, multi-phase sampling, and the
/// separate post-generation upscaler are intentionally mutually exclusive rather than being
/// silently dropped by a provider.
fn validate_hires_fix_request(request: &ImageRequest) -> WorkerResult<()> {
    if request.hires_fix.is_disabled() {
        return Ok(());
    }
    if request.mode != "text_to_image" {
        return Err(WorkerError::InvalidPayload(
            "Hires.fix is only available for text-to-image generation.".to_owned(),
        ));
    }
    if request.upscale.enabled {
        return Err(WorkerError::InvalidPayload(
            "Hires.fix and the post-generation Upscale option are mutually exclusive.".to_owned(),
        ));
    }
    let family = resolve_family(request);
    let advertises_img2img = request
        .model_manifest_entry
        .get("ui")
        .and_then(|ui| ui.get("img2img"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if family != "sdxl" && !advertises_img2img {
        return Err(WorkerError::InvalidPayload(format!(
            "Hires.fix requires an img2img-capable model; '{}' does not advertise that capability.",
            request.model
        )));
    }
    let has_edit_or_control_input = request.source_asset_id.is_some()
        || request.reference_asset_id.is_some()
        || !request.reference_asset_ids.is_empty()
        || request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || request
            .advanced
            .get("poses")
            .and_then(Value::as_array)
            .is_some_and(|poses| !poses.is_empty())
        || request
            .advanced
            .get("controlImage")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty());
    if has_edit_or_control_input {
        return Err(WorkerError::InvalidPayload(
            "Hires.fix cannot be combined with image-edit, reference, character, mask, or strict-control inputs."
                .to_owned(),
        ));
    }
    if request
        .advanced
        .get("usePid")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(WorkerError::InvalidPayload(
            "Hires.fix and PiD super-resolution are mutually exclusive.".to_owned(),
        ));
    }
    if request
        .advanced
        .get("phases")
        .and_then(Value::as_array)
        .is_some_and(|phases| !phases.is_empty())
    {
        return Err(WorkerError::InvalidPayload(
            "Hires.fix cannot be combined with multi-phase sampling.".to_owned(),
        ));
    }
    let acceleration_sampler = request
        .advanced
        .get("sampler")
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .is_some_and(|sampler| matches!(sampler.as_str(), "lightning" | "lcm" | "hyper"));
    if request.model == "realvisxl_lightning" || acceleration_sampler {
        return Err(WorkerError::InvalidPayload(
            "Hires.fix is not supported by Lightning, LCM, or Hyper acceleration samplers because they do not accept the required img2img second pass."
                .to_owned(),
        ));
    }
    let upscale_by = request.hires_fix.effective_upscale_by();
    let target_width = hires_fix_target_dimension(request.width, upscale_by);
    let target_height = hires_fix_target_dimension(request.height, upscale_by);
    if target_width > 4096 || target_height > 4096 {
        return Err(WorkerError::InvalidPayload(format!(
            "Hires.fix target {}x{} exceeds the 4096px image limit; lower the base resolution or Upscale by value.",
            target_width, target_height
        )));
    }
    Ok(())
}

/// Resolve the seed for image `index`, matching the Python worker's `resolve_seed`:
/// a base `seed` (offset by index) wins, else an explicit per-image seed, else a
/// deterministic `sha256("{prompt}:{index}")` so a re-run reproduces.
pub(crate) fn resolve_seed(request: &ImageRequest, index: usize) -> i64 {
    if let Some(base) = request.seed {
        return base.wrapping_add(index as i64);
    }
    if let Some(seed) = request.seeds.get(index) {
        return *seed;
    }
    let digest = Sha256::digest(format!("{}:{}", request.prompt, index).as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as i64
}

/// Progress payload with the worker's real backend label (the shared
/// `progress_payload` hardcodes `cpu`; the MLX worker reports `mlx`).
pub(crate) fn image_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

pub(crate) fn backend_label(gpu_id: &str) -> &str {
    if gpu_id.trim().is_empty() {
        "cpu"
    } else {
        gpu_id
    }
}

/// First-detection handling for the in-loop image cancel poller (sc-5515): trip the
/// engine `CancelFlag` and post a NON-terminal "Cancelling…" update (indeterminate
/// progress; any completed thumbnails stay via the streamed result). The terminal
/// `Canceled` is posted only after the blocking generation actually stops (see
/// `consume_gen_events`), so the worker row — and therefore the next queued job — is
/// not freed until the GPU is genuinely idle, and the UI honestly shows "Cancelling…"
/// until completion. Best-effort: a failed status update here is non-fatal because the
/// post-run terminal write is what ultimately frees the worker.
//
// Gated to where `consume_gen_events` (its only caller) and the `CancelFlag` import live — the
// `include!`d `base.rs` block — so it isn't compiled (referencing the cfg-gated `CancelFlag`) on
// non-macOS / non-candle builds.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn begin_image_cancel(
    api: &ApiClient,
    job_id: &str,
    cancel: &CancelFlag,
    plan: &ImagePlan,
    asset_writes: &[Value],
    backend: &str,
) {
    cancel.cancel();
    let _ = update_job(
        api,
        job_id,
        image_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.0,
            "Cancelling — finishing the current image…",
            Some(streaming_result(plan, asset_writes)),
            backend,
        ),
    )
    .await;
}

/// Deterministic placeholder pixels: a vertical gradient from a per-seed base colour
/// to white, exactly `width * height * 3` RGB8 bytes.
fn stub_rgb8(width: u32, height: u32, seed: i64) -> Vec<u8> {
    let seed = seed as u64;
    let base = [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
    ];
    let span = height.saturating_sub(1).max(1) as f32;
    let mut buffer = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        let t = y as f32 / span;
        let row = [lerp(base[0], t), lerp(base[1], t), lerp(base[2], t)];
        for _ in 0..width {
            buffer.extend_from_slice(&row);
        }
    }
    buffer
}

fn lerp(a: u8, t: f32) -> u8 {
    let a = a as f32;
    (a + (255.0 - a) * t).round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Real in-process MLX inference (macOS, via mlx-gen): Z-Image (sc-3022) +
// FLUX.1 schnell/dev (sc-3023), driven by the engines::MODEL_TABLE dispatch table.
// ---------------------------------------------------------------------------

// Neutral generation harness + MLX routing. The streaming helpers (`start_cached_gen_stream` /
// `consume_gen_events` / `generate_one`) and a few resolvers are backend-neutral and shared by the
// Windows candle lane (sc-3675); the MLX-coupled fns inside (`generate_stream`, the `ResolvedModel`
// resolvers) carry their own `#[cfg(target_os = "macos")]`. So these two includes compile on macOS
// AND on the Windows candle build.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// MLX/candle generator stream helpers.
include!("image_jobs/stream.rs");

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
mod tier_resolver;
#[cfg(target_os = "macos")]
use tier_resolver::resolved_tier_is_complete;
#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
use tier_resolver::standard_tier_subdir_gated;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) use tier_resolver::INT8_CONVROT_TIER;
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) use tier_resolver::NVFP4_TIER;
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
use tier_resolver::{
    is_dense_te_tier, min_quality_floor, nvfp4_host_eligible, nvfp4_requested, nvfp4_selected,
    pick_loadable_tier, preferred_tier, standard_tier_subdir, tier_components_present,
    tier_quality_rank, tier_static_name, uses_standard_tier_layout,
};

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// base image routing (MLX) + neutral txt2img generation harness + the candle execution path.
include!("image_jobs/base.rs");
// Per-generation PiD (pixel-diffusion) super-resolving decoder routing (epic 7840, sc-7849). The
// weight-resolution helper (`resolve_pid_weights`) is backend-neutral, so it compiles on BOTH face
// backends: the generic MLX lanes (base.rs/qwen.rs `generate*`, macOS-only) AND the candle InstantID
// Angles/Poses lane (instantid.rs, sc-8373), which now decodes through the `sdxl` PiD student off-Mac.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/pid.rs");
// Shared strict-control driver (epic 8236, sc-8243): the `(engine_id, control_repo, supported_kinds)`
// single source of truth + the preprocess (pose/canny/depth/user-passthrough) → `Conditioning::Control`
// core the three MLX registry strict-control paths (zimage/flux2/qwen below) route through. Off-Mac the
// candle strict-control trio (`candle_strict_control.rs`, sc-8304) reuses the SAME table +
// `preprocess_control_entry` (pose/canny/depth), so this is gated to either platform (the candle build
// off-Mac, MLX on macOS) rather than macOS-only.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/strict_control.rs");
#[cfg(target_os = "macos")]
// Z-Image strict-pose and prompt augmentation helpers.
include!("image_jobs/zimage.rs");
#[cfg(target_os = "macos")]
// FLUX.2 edit routing and conditioning.
include!("image_jobs/flux2.rs");
#[cfg(target_os = "macos")]
// FLUX.1-dev strict-control (Shakker Union-Pro-2.0) routing.
include!("image_jobs/flux1_control.rs");
#[cfg(target_os = "macos")]
// Qwen control/edit routing.
include!("image_jobs/qwen.rs");
#[cfg(target_os = "macos")]
// Krea 2 Kontext-style image-edit routing (epic 10871).
include!("image_jobs/krea_edit.rs");
// Krea 2 single-phase turbo-on-Raw routing (epic 13879 S3, sc-13883): the accelerator LoRA on a
// `krea_2_raw` t2i job → the distilled Turbo sampling regime (fixed mu 1.15 / ~8 steps / CFG-off).
// Shared by the MLX path (macOS) AND the candle lane (Windows/Linux CUDA, sc-13887): the whole file
// is backend-neutral — it resolves + samples through the same registry harness (`mlx_model` +
// `start_cached_gen_stream`) both backends use, so no `_candle.rs` sibling is needed. The candle
// engine (inference PR #204) keys the turbo regime on the `krea_2_turbo` descriptor id exactly like
// MLX.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/krea_turbo_raw.rs");
// Krea 2 multi-phase denoise routing (epic 13879 S4, sc-13884): a `krea_2_raw` t2i job carrying an
// explicit `advanced.phases` list → the multi-phase driver (one Raw trajectory / global schedule,
// per-phase guidance + per-phase toggling of the job's load-time LoRA stack by index). Shared by the
// MLX path AND the candle lane (sc-13887): the multi-phase primitive consumes the backend-agnostic
// `GenerationRequest::phases` contract, which the candle Krea engine also honors (inference PR #204),
// so the shape-detection + `advanced.phases` parse/validate + engine invocation are all backend-neutral.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/krea_multiphase.rs");
#[cfg(target_os = "macos")]
// Krea 2 pose-ControlNet (MLX) strict-pose routing (sc-8465, epic 8459 S5).
include!("image_jobs/krea_control.rs");
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// Imported single-file Krea 2 checkpoint routing (epic 14015 S0c, sc-14018/sc-14023): a user-imported
// `krea_2`-family DiT single file → paired with a resident base tier (shared TE/VAE/tokenizer) and
// loaded through the selected runtime's native single-file entrypoint, bypassing the registry
// snapshot-dir path. Shared by MLX and Candle so global import acceptance always has a real route.
include!("image_jobs/krea_imported.rs");
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/sdxl_imported.rs");
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// Fine-tuned Mage-Flow base checkpoint routing (sc-15036, epic 14034 F6): the `transformer/`-shaped
// artifact a FULL base fine-tune writes, paired at load with the installed base's shared text
// encoder + VAE and rendered through the `load_finetuned` entrypoint that skips the pinned-
// checkpoint identity guard a fine-tune necessarily fails. The shared request path uses native MLX
// on macOS and the native Candle Mage engine on CUDA hosts.
include!("image_jobs/mage_finetuned.rs");
#[cfg(target_os = "macos")]
// SenseNova edit routing.
include!("image_jobs/sensenova.rs");
// Bernini still-image (t2i/i2i) routing. Included on macOS (the MLX `generate_bernini_image_stream`)
// AND the candle lane (sc-10996: `generate_candle_bernini_image_stream` + the shared task/raw-settings/
// generate-one helpers); each item inside is cfg-gated to its backend, so the neither build pulls in
// nothing. (Unlike the other macOS-only routing files above, whose candle siblings live in dedicated
// `*_candle.rs` files, Bernini keeps both lanes in one file to share the neutral still-image helpers.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/bernini.rs");
#[cfg(target_os = "macos")]
// SDXL advanced routing.
include!("image_jobs/sdxl.rs");
#[cfg(target_os = "macos")]
// Kolors advanced conditioning (img2img + IP-Adapter-Plus reference).
include!("image_jobs/kolors.rs");
// InstantID native routing — macOS (MLX) + the Windows/CUDA candle lane (sc-5491). The two engines'
// `InstantId` APIs differ only at the load boundary (with_face dir-vs-Weights, quantize, largest_face
// signature), cfg-split inside; the per-item generate/restore loop is backend-neutral over `gen_core`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/instantid.rs");
// SDXL IP-Adapter-Plus reference conditioning — the Windows/CUDA candle lane ONLY (sc-5488). macOS keeps
// the MLX SDXL IP path (sdxl.rs `SdxlSubMode::Ip`); there is no MLX `IpAdapterSdxl`, so this is
// candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod sdxl_ipadapter;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use sdxl_ipadapter::{generate_candle_sdxl_ipadapter_stream, sdxl_ipadapter_available};
// SDXL img2img / inpaint / outpaint edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps the
// MLX SDXL advanced path (sdxl.rs `SdxlSubMode::{Edit,Inpaint,Outpaint}`); the candle `SdxlEdit` is a
// bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod sdxl_edit_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use sdxl_edit_candle::{generate_candle_sdxl_edit_stream, sdxl_edit_candle_available};
// FLUX.2-klein reference / img2img edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps the
// MLX FLUX.2 edit path (flux2.rs `generate_flux2_edit_stream`); the candle `Flux2Edit` is a bespoke
// provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod flux2_edit_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use flux2_edit_candle::{flux2_edit_candle_available, generate_candle_flux2_edit_stream};
// Qwen-Image-Edit reference / dual-latent edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps
// the MLX Qwen-Image-Edit path (qwen.rs); the candle `QwenEdit` is a bespoke provider, so this is
// candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod qwen_edit_candle;
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) use qwen_edit_candle::resolve_qwen_edit_candle_base;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use qwen_edit_candle::{generate_candle_qwen_edit_stream, qwen_edit_candle_available};
// Krea 2 Kontext-style dual-conditioned image-edit — the Windows/CUDA candle lane ONLY (epic 10871).
// macOS keeps the MLX Krea edit path (krea_edit.rs, the `krea_2_edit` registry generator); the candle
// Krea edit is a bespoke pipeline, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod krea_edit_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use krea_edit_candle::{generate_candle_krea_edit_stream, krea_edit_candle_available};
// Kolors IP-Adapter-Plus reference conditioning — the Windows/CUDA candle lane ONLY (sc-5488). macOS
// keeps the MLX Kolors IP path (kolors.rs, the registry `Reference` route); the candle `IpAdapterKolors`
// is a bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod kolors_ipadapter;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use kolors_ipadapter::{generate_candle_kolors_ipadapter_stream, kolors_ipadapter_available};
// FLUX XLabs IP-Adapter reference conditioning — the Windows/CUDA candle lane ONLY (sc-5872). macOS keeps
// the MLX FLUX XLabs IP path (epic 3621, the registry `Reference` route); the candle `IpAdapterFlux` is a
// bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod flux_ipadapter;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use flux_ipadapter::{flux_ipadapter_available, generate_candle_flux_ipadapter_stream};
// Shared candle conditioning-overlay admission seam (sc-16069, epic 15448): the ONE gate every candle
// route that overlays a second network on the base (ControlNet / IP-Adapter / identity encoder) calls
// before it allocates. Those routes are diverted by `resolve_candle_image_route` around BOTH the
// `generate_candle_stream` `vram_gate` and the `generator_cache` `apply_residency_policy`, so eleven of
// them had no pre-flight check at all — no rejection, no warning, reactive CUDA OOM only. Must precede
// `candle_strict_control` (whose trait references the footprint type) and the conditioning lanes.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod conditioning_gate;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use conditioning_gate::{admit_conditioning_overlay, admit_conditioning_paths};
// Shared admission seam for bespoke single-base Candle routes (sc-16093). Built-in tiers use catalog
// catalog peaks; imported/ComfyUI checkpoints use an explicitly weaker on-disk weights floor.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod base_admission;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use base_admission::{
    admit_candle_base, admit_candle_base_floor, admit_candle_base_floor_with_resident_overlay,
    admit_candle_load_spec_floor, has_candle_tier_peak_row, prepare_cached_candle_base_floor,
    safetensors_tensor_bytes_with_prefixes, CandleBaseEvidence,
};
// Shared candle strict-control driver (sc-8304, epic 8236): the `CandleStrictControl` trait + the one
// `run_candle_strict_control` driver the candle trio (qwen/zimage/flux2 control below) route through —
// reusing the SAME `STRICT_CONTROL_ENGINES` table + `preprocess_control_entry` (pose/canny/depth) as the
// MLX `strict_control.rs`. Must precede the three lanes (they reference the trait + driver).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod candle_strict_control;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_strict_control::{run_candle_strict_control, CandleStrictControl};
// Qwen-Image 2512-Fun-Controlnet-Union (strict control) — the Windows/CUDA candle lane ONLY (sc-5489
// origin / sc-8350 repoint). macOS keeps the MLX `qwen_image_control` registry generator; the candle
// `QwenFunControl` is a bespoke provider (the InstantX `QwenControl` is retired on the candle lane).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod qwen_control;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use qwen_control::{generate_candle_qwen_control_stream, qwen_control_available};
// Kolors ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-5489). macOS keeps the MLX
// Kolors ControlNet path; the candle `KolorsControl` is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod kolors_control;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use kolors_control::{generate_candle_kolors_control_stream, kolors_control_available};
// Z-Image Fun-ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-5489). macOS keeps the
// MLX `z_image_turbo_control` registry generator; the candle `ZImageControl` is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod zimage_control;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use zimage_control::{generate_candle_zimage_control_stream, zimage_control_available};
// FLUX.2-dev Fun-Controlnet-Union (strict pose) — the Windows/CUDA candle lane ONLY (sc-7736, epic 6564).
// macOS keeps the MLX `flux2_dev_control` registry generator (flux2.rs); the candle `Flux2Control` is a
// bespoke provider (the dev VACE control branch over the Q4 dev DiT).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod flux2_control_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use flux2_control_candle::{flux2_control_candle_available, generate_candle_flux2_control_stream};
// FLUX.1-dev Shakker Union-Pro-2.0 (strict control) — the Windows/CUDA candle lane ONLY (sc-8412, epic
// 8236). macOS keeps the MLX `flux1_dev_control` registry generator (flux1_control.rs); the candle
// `Flux1DevControl` is a bespoke provider (the Shakker residual-emitter control branch over the dense
// bf16 dev DiT).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod flux1_control_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use flux1_control_candle::{flux1_control_candle_available, generate_candle_flux1_control_stream};
// Krea 2 pose-ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-8464, epic 8459). There is
// no MLX Krea control twin yet (8459 S5 / sc-8465); the candle `Krea2Control` loads a trained
// control-branch overlay on the frozen dense bf16 Turbo base.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod krea_control_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use krea_control_candle::{generate_candle_krea_control_stream, krea_control_candle_available};
// Z-Image img2img / edit — the Windows/CUDA candle lane ONLY (sc-6595). macOS keeps the MLX
// `z_image_turbo` registry generator's `Conditioning::Reference` img2img path; the candle `ZImageEdit`
// is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod zimage_edit_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use zimage_edit_candle::zimage_edit_candle_available;
// In-place ComfyUI Z-Image base txt2img — Windows/CUDA candle lane ONLY (sc-10668, epic 10451). Renders
// a user's ComfyUI Z-Image weights in place via `runtime_cuda::providers::z_image::load_from_comfyui_components`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod zimage_comfyui_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use zimage_comfyui_candle::generate_candle_zimage_comfyui_stream;
// Qwen-Image txt2img from an in-place ComfyUI DiT (plain fp8_e4m3fn → bf16) — the Windows/CUDA candle
// lane ONLY (sc-10670, epic 10451 Phase 2b). Sibling of the Z-Image comfyui lane; TE/VAE/tokenizer come
// from a resident `SceneWorks/qwen-image-mlx` snapshot tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod qwen_comfyui_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use qwen_comfyui_candle::generate_candle_qwen_comfyui_stream;
// FLUX.2-dev txt2img from an in-place ComfyUI fp8-mixed DiT (inline-scale fp8 dequant → f32, then
// quantized onto the GPU) — the Windows/CUDA candle lane ONLY (sc-10680, epic 10451 Phase 2e). Sibling
// of the Qwen-Image comfyui lane; the Mistral-3 TE / VAE / tokenizer come from a resident FLUX.2-dev
// snapshot tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod flux2_comfyui_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use flux2_comfyui_candle::generate_candle_flux2_comfyui_stream;
// Z-Image identity-init request gate for Image Studio "With Character" (sc-8409, epic 4406). Both
// backends now generate through their registered `z_image_turbo` provider; this candle-only helper
// preserves the off-Mac availability/base-resolution predicate while the generic stream owns Reference
// conditioning, adapters, provenance, and face-likeness scoring.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod zimage_identity_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use zimage_identity_candle::{zimage_identity_candle_available, zimage_identity_candle_strength};
// PuLID-FLUX face identity — the Windows/CUDA candle lane ONLY (sc-5492). macOS keeps the
// inventory-registered `pulid_flux` MLX generator (image_jobs/pulid.rs); the candle `PulidFlux` is a
// bespoke provider, so this file is candle-gated and distinct from the macOS route.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod pulid_candle;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use pulid_candle::{generate_candle_pulid_stream, pulid_candle_available};
#[cfg(target_os = "macos")]
// PuLID-FLUX native routing.
include!("image_jobs/pulid.rs");
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// image detail tile-ControlNet routing.
include!("image_jobs/detail.rs");

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
pub(crate) async fn run_image_detail_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "image_detail requires either the MLX or Candle inference backend".to_owned(),
    ))
}

#[cfg(test)]
mod tests;
