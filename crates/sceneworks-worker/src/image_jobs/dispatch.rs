//! Execution of the one prepared image route; heartbeat and job lifecycle stay in the parent.
use super::{ImagePlan, ImageRequest};
use crate::{ApiClient, JobSnapshot, Settings, WorkerError, WorkerResult};
use serde_json::Value;
use std::path::Path;

pub(super) enum ImageRouteOutcome {
    Handled,
    Unavailable,
}
#[cfg(target_os = "macos")]
use super::bernini::generate_bernini_image_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::bernini::generate_candle_bernini_image_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::candle::generate_candle_stream;
use super::checkpoint_plan::generate_checkpoint_plan_stream;
#[cfg(target_os = "macos")]
use super::flux1_control::generate_flux1_dev_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::flux1_control_candle::generate_candle_flux1_control_stream;
#[cfg(target_os = "macos")]
use super::flux2::generate_flux2_dev_control_stream;
#[cfg(target_os = "macos")]
use super::flux2::generate_flux2_edit_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::flux2_comfyui_candle::generate_candle_flux2_comfyui_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::flux2_control_candle::generate_candle_flux2_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::flux2_edit_candle::generate_candle_flux2_edit_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::flux_ipadapter::generate_candle_flux_ipadapter_stream;
use super::instantid::generate_instantid_stream;
#[cfg(target_os = "macos")]
use super::kolors::generate_kolors_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::kolors_control::generate_candle_kolors_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::kolors_ipadapter::generate_candle_kolors_ipadapter_stream;
#[cfg(target_os = "macos")]
use super::krea_control::generate_krea_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::krea_control_candle::generate_candle_krea_control_stream;
#[cfg(target_os = "macos")]
use super::krea_edit::generate_krea_edit_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::krea_edit_candle::generate_candle_krea_edit_stream;
use super::krea_imported::generate_krea_imported_control_stream;
use super::krea_imported::generate_krea_imported_stream;
use super::krea_multiphase::generate_krea_multiphase_stream;
use super::krea_turbo_raw::generate_krea_turbo_on_raw_stream;
use super::mage_finetuned::generate_mage_finetuned_stream;
#[cfg(target_os = "macos")]
use super::mlx::generate_stream;
#[cfg(target_os = "macos")]
use super::pulid::generate_pulid_flux_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::pulid_candle::generate_candle_pulid_stream;
#[cfg(target_os = "macos")]
use super::qwen::generate_qwen_control_stream;
#[cfg(target_os = "macos")]
use super::qwen::generate_qwen_edit_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::qwen_comfyui_candle::generate_candle_qwen_comfyui_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::qwen_control::generate_candle_qwen_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::qwen_edit_candle::generate_candle_qwen_edit_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::routing::CandleImageRoute;
#[cfg(target_os = "macos")]
use super::routing::ImageRoute;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::routing::PreparedCandleImageRoute;
use super::routing::PreparedFileDispatch;
#[cfg(target_os = "macos")]
use super::routing::PreparedImageRoute;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::routing::WIRED_CANDLE_POSE_FAMILIES;
#[cfg(target_os = "macos")]
use super::routing::WIRED_MLX_POSE_FAMILIES;
#[cfg(target_os = "macos")]
use super::sdxl::generate_sdxl_advanced_stream;
use super::sdxl_control::generate_sdxl_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::sdxl_edit_candle::generate_candle_sdxl_edit_stream;
use super::sdxl_imported::generate_sdxl_imported_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::sdxl_ipadapter::generate_candle_sdxl_ipadapter_stream;
#[cfg(target_os = "macos")]
use super::sensenova::generate_sensenova_edit_stream;
#[cfg(target_os = "macos")]
use super::zimage::generate_zimage_base_control_stream;
#[cfg(target_os = "macos")]
use super::zimage::generate_zimage_control_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::zimage_comfyui_candle::generate_candle_zimage_comfyui_stream;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::zimage_control::generate_candle_zimage_control_stream;
#[cfg(target_os = "macos")]
// Preserve the existing borrowed job context and one-shot loader ownership.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_mlx_image_route(
    route: Option<PreparedImageRoute>,
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<ImageRouteOutcome> {
    Ok(if let Some(route) = route {
        match route.kind() {
            ImageRoute::ZImageControl => {
                // Z-Image strict-pose (advanced.poses) → Fun-Controlnet-Union, one image per pose.
                generate_zimage_control_stream(
                    api,
                    settings,
                    job,
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
                )
                .await?;
            }
            ImageRoute::QwenControl => {
                // Qwen strict-pose (advanced.poses) → InstantX ControlNet-Union, one image per pose.
                generate_qwen_control_stream(
                    api,
                    settings,
                    job,
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux2DevControl => {
                // FLUX.2-dev strict-pose (advanced.poses) → Fun-Controlnet-Union, one image per pose.
                generate_flux2_dev_control_stream(
                    api,
                    settings,
                    job,
                    plan,
                    project_path,
                    backend,
                    asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux2Edit => {
                // FLUX.2-klein edit/reference (mode edit_image or a reference) → edit variant.
                generate_flux2_edit_stream(
                    api,
                    settings,
                    job,
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
                )
                .await?;
            }
            ImageRoute::CheckpointPlan => {
                // Plan-driven checkpoint (epic 20398, sc-20634): a persisted ImportPlanV1 resolved
                // and re-verified before load; provider selected by family/source/operation through
                // the registry. txt2img, `count` renders each its own seed.
                let PreparedImageRoute::CheckpointPlan(sources) = route else {
                    unreachable!("checkpoint plan route missing its prepared sources")
                };
                generate_checkpoint_plan_stream(
                    api,
                    settings,
                    job,
                    *sources,
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                        plan,
                        sources: *sources,
                    },
                    project_path,
                    backend,
                    asset_writes,
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
                        plan,
                        sources: *sources,
                    },
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                        plan,
                        sources: *sources,
                    },
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
                )
                .await?;
            }
            ImageRoute::SdxlControl => {
                generate_sdxl_control_stream(
                    api,
                    settings,
                    job,
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
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
                    plan,
                    project_path,
                    backend,
                    asset_writes,
                )
                .await?;
            }
            ImageRoute::FluxIpAdapterPoseReject => {
                return Err(WorkerError::InvalidPayload(
                    "FLUX.1 IP-Adapter reference conditioning cannot be combined with strict pose control; \
                     this backend has separate IP-Adapter and ControlNet providers, so refusing rather \
                     than silently dropping either requested conditioning".to_owned(),
                ));
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
                // No-silent-T2I (sc-5968): control intent on an MLX model with NO pose-control lane
                // that `mlx_available` would otherwise render as plain txt2img, dropping the intent.
                // Refuse loudly — the MLX twin of the candle `PoseReject` reject.
                return Err(WorkerError::InvalidPayload(format!(
                    "control intent (advanced.poses/controlMode/controlImage/controlWeights) is not supported for model '{}' on the MLX backend — \
                     refusing rather than silently generating an unconditioned image (wired MLX control \
                     families: {})",
                    request.model,
                    WIRED_MLX_POSE_FAMILIES.join(", ")
                )));
            }
            ImageRoute::Mlx => {
                generate_stream(
                    api,
                    settings,
                    job,
                    plan,
                    project_path,
                    backend,
                    asset_writes,
                )
                .await?;
            }
        }
        ImageRouteOutcome::Handled
    } else {
        ImageRouteOutcome::Unavailable
    })
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
// Preserve the existing borrowed job context and one-shot loader ownership.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_candle_image_route(
    route: Option<PreparedCandleImageRoute>,
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<ImageRouteOutcome> {
    Ok(match route {
        Some(route) => {
            match route.kind() {
                // InstantID (sc-5491, epic 5480): the candle InstantID provider's bespoke path (the
                // off-Mac sibling of the macOS `ImageRoute::InstantId` arm).
                CandleImageRoute::InstantId => {
                    generate_instantid_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::SdxlControl => {
                    generate_sdxl_control_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                            plan,
                            sources: *sources,
                        },
                        project_path,
                        backend,
                        asset_writes,
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
                            plan,
                            sources: *sources,
                        },
                        project_path,
                        backend,
                        asset_writes,
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
                            plan,
                            sources: *sources,
                        },
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                // Kolors IP-Adapter-Plus reference conditioning (sc-5488).
                CandleImageRoute::KolorsIpAdapter => {
                    generate_candle_kolors_ipadapter_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                // FLUX XLabs IP-Adapter reference conditioning (sc-5872).
                CandleImageRoute::FluxIpAdapter => {
                    generate_candle_flux_ipadapter_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                // Kolors strict-pose ControlNet (sc-5489).
                CandleImageRoute::KolorsControl => {
                    generate_candle_kolors_control_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::KolorsCompositeReject => {
                    return Err(WorkerError::InvalidPayload(
                        "Kolors Candle does not compose IP-Adapter identity with pose ControlNet or PiD; refusing the crossed request before model load".to_owned(),
                    ));
                }
                // Z-Image strict-pose Fun-ControlNet (sc-5489).
                CandleImageRoute::ZimageControl => {
                    generate_candle_zimage_control_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                            plan,
                            sources: *sources,
                        },
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::KreaImportedControl => {
                    generate_krea_imported_control_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                CandleImageRoute::CheckpointPlan => {
                    let PreparedCandleImageRoute::CheckpointPlan(sources) = route else {
                        unreachable!("checkpoint plan route missing its prepared sources")
                    };
                    generate_checkpoint_plan_stream(
                        api,
                        settings,
                        job,
                        *sources,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                        plan,
                        project_path,
                        backend,
                        asset_writes,
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
                            plan,
                            sources: *sources,
                        },
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
                // No-silent-T2I (sc-5968): a strict-pose job on a candle model with NO pose lane must be
                // REJECTED with a clear error, not silently rendered as plain txt2img (poses dropped) and
                // not rerouted. The candle worker CLAIMS these (jobs_store `image_job_candle_pose_reject`)
                // precisely to fail them loudly here.
                CandleImageRoute::PoseReject => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "control intent (advanced.poses/controlMode/controlImage/controlWeights) is not supported for model '{}' on the candle backend — \
                         refusing rather than silently generating an unconditioned image (wired candle control \
                         families: {})",
                        request.model,
                        WIRED_CANDLE_POSE_FAMILIES.join(", ")
                    )));
                }
                CandleImageRoute::FluxIpAdapterPoseReject => {
                    return Err(WorkerError::InvalidPayload(
                        "FLUX.1 IP-Adapter reference conditioning cannot be combined with strict pose control; \
                         this backend has separate IP-Adapter and ControlNet providers, so refusing rather \
                         than silently dropping either requested conditioning".to_owned(),
                    ));
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
                | CandleImageRoute::MossImageEdit
                | CandleImageRoute::QwenImage21Edit
                | CandleImageRoute::Flux3Reference
                | CandleImageRoute::SenseNovaEdit
                | CandleImageRoute::KolorsEdit
                | CandleImageRoute::CandleTxt2Img => {
                    generate_candle_stream(
                        api,
                        settings,
                        job,
                        plan,
                        project_path,
                        backend,
                        asset_writes,
                    )
                    .await?;
                }
            }
            ImageRouteOutcome::Handled
        }
        // Candle disabled (default) or no candle engine matched → stub exactly as before.
        None => ImageRouteOutcome::Unavailable,
    })
}
