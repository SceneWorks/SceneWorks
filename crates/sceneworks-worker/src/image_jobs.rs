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
// `AdapterKind` (LoRA/LoKr classification) was MLX-only until sc-5126: the candle Lens lane is the
// first candle family to take LoRA/LoKr, so it now classifies adapters too and the import moved into
// the shared block above. `ControlKind` (ControlNet conditioning) was MLX-only until sc-8304: the candle
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
    BodyPoint, InstantId, InstantIdPaths, InstantIdRequest, FACE_RESTORE_PROMPT,
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
// Z-Image img2img / edit provider (sc-6595, epic 5480) — the candle (Windows/CUDA) sibling of the MLX
// `z_image_turbo` `Conditioning::Reference` img2img route, living in `candle-gen-z-image` (the Turbo DiT
// + a strength-derived source-latent init). Candle-only: macOS keeps the registered MLX generator's
// img2img path. The bespoke edit route (`image_jobs/zimage_edit_candle.rs`) uses this named runtime
// utility export.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::z_image::{ZImageEdit, ZImageEditPaths, ZImageEditRequest};
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
/// Used both per-asset (`generate_candle_stream`) and at the generation-set level (`adapter_id`)
/// so the sidecar + result agree on which backend produced the image.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_ADAPTER: &str = "candle_sdxl";
// Shared by the MLX path and the candle Lens lane (sc-5126) — both cap a job's total LoRAs at
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
/// Dispatch handler for `JobType::ImageGenerate`: generate, save, and stream image
/// assets through the Rust GPU worker.
pub(crate) async fn run_image_generate_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
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
    let route = resolve_image_route(&request, settings);
    #[cfg(target_os = "macos")]
    let plan = ImagePlan::with_count(
        &request,
        route.map_or(request.count, |route| route.image_count(&request, settings)) * upscale_mult,
    );
    // Windows/CUDA candle lane: resolve the candle dispatch branch once and bake THAT branch's real
    // total into the plan, exactly as the macOS arm does with `resolve_image_route`. An InstantID
    // angle/pose set produces N images (the active angle collection's length, or the pose count) and
    // every strict-pose control lane produces one image per pose (`pose_entries().len()`), not
    // `request.count` — so the generation set + streamed `expectedCount` match what lands in the gallery
    // (sc-5491 InstantID; sc-11171 F-009 strict-pose). `resolve_candle_image_route` returns `None` when
    // candle is disabled, so any other job (or a disabled backend) keeps `request.count`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let plan = {
        let count = resolve_candle_image_route(&request, settings)
            .map_or(request.count, |route| route.image_count(&request, settings));
        ImagePlan::with_count(&request, count * upscale_mult)
    };
    #[cfg(all(
        not(target_os = "macos"),
        not(all(not(target_os = "macos"), feature = "backend-candle"))
    ))]
    let plan = ImagePlan::with_count(&request, request.count * upscale_mult);

    // Pre-flight LoRA family-compat guardrail (sc-3027): reject an incompatible LoRA
    // (e.g. a Flux LoRA on an SDXL model, or a Wan 5B LoRA on the 14B base) before any
    // heavy load, with the same message the Python worker raised — instead of failing
    // deep in the engine's strict adapter loader. Network-type handling (peft LoKr AND third-party
    // LyCORIS both apply on MLX now, epic 3641) is done by routing + `classify_adapter` + the engine.
    sceneworks_core::lora_family::validate_lora_compatibility(
        &request.loras,
        Some(plan.family.as_str()),
        adapter_id(&request),
        Some(request.model.as_str()),
    )
    .map_err(WorkerError::InvalidPayload)?;

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
        match route {
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
    // candle is a narrow txt2img-only lane, so for a candle-engine model (sdxl/realvisxl) with the
    // backend enabled we run `generate_candle_stream` (same neutral assetWrites/progress/cancellation
    // harness). Gated on `backend_candle_enabled` (default off) so production routing is unchanged
    // until parity is accepted — otherwise it stubs exactly like before.
    // InstantID (sc-5491, epic 5480) is the exception to "txt2img-only": the candle InstantID provider
    // gets its own bespoke path (`generate_instantid_stream`, the off-Mac sibling of the macOS
    // `ImageRoute::InstantId` arm) — checked first since `instantid_realvisxl` is not an inventory
    // `is_candle_engine` id.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let handled = match resolve_candle_image_route(&request, settings) {
        Some(route) => {
            match route {
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
                    generate_candle_zimage_edit_stream(
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
                    generate_candle_zimage_comfyui_stream(
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
                CandleImageRoute::QwenImageComfyui => {
                    generate_candle_qwen_comfyui_stream(
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
                // In-place ComfyUI FLUX.2-dev fp8-mixed base (sc-10680, epic 10451): an `external_base_*`
                // id whose forwarded row carries the DiT component path — render the user's ComfyUI
                // weights in place via `runtime_cuda::providers::flux2::load_from_comfyui_dit` (inline-scale fp8 dequant
                // + BFL→diffusers remap; TE/VAE/tokenizer from a resident FLUX.2-dev snapshot).
                CandleImageRoute::Flux2Comfyui => {
                    generate_candle_flux2_comfyui_stream(
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
                // Z-Image identity-init for Image Studio "With Character" (sc-8409, epic 4406) — the
                // off-Mac sibling of the macOS generic lane's Z-Image identity img2img; reuses the candle
                // ZImageEdit engine with the identity `referenceAssetId` as the source-latent init + wires
                // the sc-4411 face-likeness scorer. Diverted before the txt2img arm (else the reference
                // silently drops).
                CandleImageRoute::ZimageIdentity => {
                    generate_candle_zimage_identity_stream(
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
                // No-silent-T2I (sc-5968): a strict-pose job on a candle model with NO pose lane (e.g.
                // sdxl) must be REJECTED with a clear error, not silently rendered as plain txt2img (poses
                // dropped) and not bounced to torch. The candle worker CLAIMS these (jobs_store
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
                // Plain candle txt2img (sc-3675, epic 3672): sdxl/realvisxl on the narrow txt2img lane.
                CandleImageRoute::CandleTxt2Img => {
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
            http_client,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
    }
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let _ = http_client;

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
    /// Number of images this job produces. Usually `request.count`, but a FLUX.2 angle
    /// set is 11 and a pose set is the pose count (sc-3030) — the generation set's
    /// `count`/`expectedCount` reflect this so the gallery streams against the real
    /// total, not the requested `count`.
    image_count: u32,
}

impl ImagePlan {
    /// Test-only convenience: a plan whose image count is the request count. Production
    /// always goes through [`ImagePlan::with_count`] (the FLUX.2 angle/pose sets need an
    /// effective count that differs from `request.count`).
    #[cfg(test)]
    fn new(request: &ImageRequest) -> Self {
        Self::with_count(request, request.count)
    }

    /// Build a plan whose generation set reports `image_count` images (see the field).
    pub(crate) fn with_count(request: &ImageRequest, image_count: u32) -> Self {
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
            image_count,
        }
    }
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
    rgb_image
        .save_with_format(&temp_path, image::ImageFormat::Png)
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
        "sourceAssetId": request.source_asset_id,
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
/// standalone `image_upscale` job — Real-ESRGAN via `ort`, SeedVR2 via the registry generator — provisioning
/// weights on first use. Runs after the base images have already been streamed (so they persist even if a
/// late upscale step errors and fails the job).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn apply_inline_upscale(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
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
            http_client,
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
    let temp_path = media_path.with_extension("tmp.png");
    upscaled
        .save_with_format(&temp_path, image::ImageFormat::Png)
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
    let upscale_record = if engine_id == "seedvr2" {
        json!({ "enabled": true, "engine": engine_id, "factor": factor, "softness": softness })
    } else {
        json!({ "enabled": true, "engine": engine_id, "factor": factor })
    };
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
        "adapter": adapter_id(&plan.request),
        "model": plan.request.model,
        "generationSet": plan.generation_set,
        "assetWrites": asset_writes,
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// The adapter id reported for the set (real engine on macOS for a linked family,
/// else the procedural stub).
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
#[cfg(target_os = "macos")]
// Krea 2 pose-ControlNet (MLX) strict-pose routing (sc-8465, epic 8459 S5).
include!("image_jobs/krea_control.rs");
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
include!("image_jobs/sdxl_ipadapter.rs");
// SDXL img2img / inpaint / outpaint edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps the
// MLX SDXL advanced path (sdxl.rs `SdxlSubMode::{Edit,Inpaint,Outpaint}`); the candle `SdxlEdit` is a
// bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/sdxl_edit_candle.rs");
// FLUX.2-klein reference / img2img edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps the
// MLX FLUX.2 edit path (flux2.rs `generate_flux2_edit_stream`); the candle `Flux2Edit` is a bespoke
// provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux2_edit_candle.rs");
// Qwen-Image-Edit reference / dual-latent edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps
// the MLX Qwen-Image-Edit path (qwen.rs); the candle `QwenEdit` is a bespoke provider, so this is
// candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/qwen_edit_candle.rs");
// Krea 2 Kontext-style dual-conditioned image-edit — the Windows/CUDA candle lane ONLY (epic 10871).
// macOS keeps the MLX Krea edit path (krea_edit.rs, the `krea_2_edit` registry generator); the candle
// Krea edit is a bespoke pipeline, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/krea_edit_candle.rs");
// Kolors IP-Adapter-Plus reference conditioning — the Windows/CUDA candle lane ONLY (sc-5488). macOS
// keeps the MLX Kolors IP path (kolors.rs, the registry `Reference` route); the candle `IpAdapterKolors`
// is a bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/kolors_ipadapter.rs");
// FLUX XLabs IP-Adapter reference conditioning — the Windows/CUDA candle lane ONLY (sc-5872). macOS keeps
// the MLX FLUX XLabs IP path (epic 3621, the registry `Reference` route); the candle `IpAdapterFlux` is a
// bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux_ipadapter.rs");
// Shared candle strict-control driver (sc-8304, epic 8236): the `CandleStrictControl` trait + the one
// `run_candle_strict_control` driver the candle trio (qwen/zimage/flux2 control below) route through —
// reusing the SAME `STRICT_CONTROL_ENGINES` table + `preprocess_control_entry` (pose/canny/depth) as the
// MLX `strict_control.rs`. Must precede the three lanes (they reference the trait + driver).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/candle_strict_control.rs");
// Qwen-Image 2512-Fun-Controlnet-Union (strict control) — the Windows/CUDA candle lane ONLY (sc-5489
// origin / sc-8350 repoint). macOS keeps the MLX `qwen_image_control` registry generator; the candle
// `QwenFunControl` is a bespoke provider (the InstantX `QwenControl` is retired on the candle lane).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/qwen_control.rs");
// Kolors ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-5489). macOS keeps the MLX
// Kolors ControlNet path; the candle `KolorsControl` is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/kolors_control.rs");
// Z-Image Fun-ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-5489). macOS keeps the
// MLX `z_image_turbo_control` registry generator; the candle `ZImageControl` is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/zimage_control.rs");
// FLUX.2-dev Fun-Controlnet-Union (strict pose) — the Windows/CUDA candle lane ONLY (sc-7736, epic 6564).
// macOS keeps the MLX `flux2_dev_control` registry generator (flux2.rs); the candle `Flux2Control` is a
// bespoke provider (the dev VACE control branch over the Q4 dev DiT).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux2_control_candle.rs");
// FLUX.1-dev Shakker Union-Pro-2.0 (strict control) — the Windows/CUDA candle lane ONLY (sc-8412, epic
// 8236). macOS keeps the MLX `flux1_dev_control` registry generator (flux1_control.rs); the candle
// `Flux1DevControl` is a bespoke provider (the Shakker residual-emitter control branch over the dense
// bf16 dev DiT).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux1_control_candle.rs");
// Krea 2 pose-ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-8464, epic 8459). There is
// no MLX Krea control twin yet (8459 S5 / sc-8465); the candle `Krea2Control` loads a trained
// control-branch overlay on the frozen dense bf16 Turbo base.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/krea_control_candle.rs");
// Z-Image img2img / edit — the Windows/CUDA candle lane ONLY (sc-6595). macOS keeps the MLX
// `z_image_turbo` registry generator's `Conditioning::Reference` img2img path; the candle `ZImageEdit`
// is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/zimage_edit_candle.rs");
// In-place ComfyUI Z-Image base txt2img — Windows/CUDA candle lane ONLY (sc-10668, epic 10451). Renders
// a user's ComfyUI Z-Image weights in place via `runtime_cuda::providers::z_image::load_from_comfyui_components`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/zimage_comfyui_candle.rs");
// Qwen-Image txt2img from an in-place ComfyUI DiT (plain fp8_e4m3fn → bf16) — the Windows/CUDA candle
// lane ONLY (sc-10670, epic 10451 Phase 2b). Sibling of the Z-Image comfyui lane; TE/VAE/tokenizer come
// from a resident `SceneWorks/qwen-image-mlx` snapshot tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/qwen_comfyui_candle.rs");
// FLUX.2-dev txt2img from an in-place ComfyUI fp8-mixed DiT (inline-scale fp8 dequant → f32, then
// quantized onto the GPU) — the Windows/CUDA candle lane ONLY (sc-10680, epic 10451 Phase 2e). Sibling
// of the Qwen-Image comfyui lane; the Mistral-3 TE / VAE / tokenizer come from a resident FLUX.2-dev
// snapshot tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux2_comfyui_candle.rs");
// Z-Image identity-init for Image Studio "With Character" — the Windows/CUDA candle lane ONLY (sc-8409,
// epic 4406). macOS keeps the MLX `z_image_turbo` generic-lane identity img2img (`generate_stream` ⇒
// `resolve_zimage_identity_init`); off-Mac this bespoke lane reuses the candle `ZImageEdit` engine with
// the identity `referenceAssetId` as the source-latent init + wires the sc-4411 face-likeness scorer.
// Reuses the sibling `zimage_edit_candle.rs` base/steps helpers, so it is included right after it.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/zimage_identity_candle.rs");
// PuLID-FLUX face identity — the Windows/CUDA candle lane ONLY (sc-5492). macOS keeps the
// inventory-registered `pulid_flux` MLX generator (image_jobs/pulid.rs); the candle `PulidFlux` is a
// bespoke provider, so this file is candle-gated and distinct from the macOS route.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/pulid_candle.rs");
#[cfg(target_os = "macos")]
// PuLID-FLUX native routing.
include!("image_jobs/pulid.rs");
#[cfg(target_os = "macos")]
// image detail tile-ControlNet routing.
include!("image_jobs/detail.rs");

/// Off macOS the in-process engine is unavailable; `image_detail` is served by the Python
/// torch worker (the `mlx` worker — the only one advertising this capability — is macOS-only).
#[cfg(not(target_os = "macos"))]
pub(crate) async fn run_image_detail_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "image_detail runs on the macOS MLX worker, not this worker".to_owned(),
    ))
}

#[cfg(test)]
mod tests;
