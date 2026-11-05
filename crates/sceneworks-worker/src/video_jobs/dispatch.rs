//! Selected video-route execution; lifecycle, encoding and terminal updates stay in the parent.
#[cfg(target_os = "macos")]
use super::wan;
use super::DecodedVideo;
use serde_json::Value;

pub(super) struct VideoRouteOutcome {
    pub(super) decoded: DecodedVideo,
    pub(super) adapter: &'static str,
    pub(super) raw_settings: Value,
    pub(super) replacement_status: Option<Value>,
}
#[cfg(native_lane)]
use super::bernini_raw_settings;
#[cfg(target_os = "macos")]
use super::generate_bernini;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_bernini;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_minimax_h3;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_scail2;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_scail2_replace;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_video;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_wan_comfyui;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_wan_vace;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_wan_vace_extend_bridge;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::generate_candle_wan_vace_fun;
#[cfg(target_os = "macos")]
use super::generate_krea_realtime;
#[cfg(target_os = "macos")]
use super::generate_ltx;
#[cfg(target_os = "macos")]
use super::generate_minimax_h3;
#[cfg(target_os = "macos")]
use super::generate_mochi;
#[cfg(target_os = "macos")]
use super::generate_scail2;
#[cfg(target_os = "macos")]
use super::generate_scail2_replace;
#[cfg(native_lane)]
use super::generate_stub_video;
#[cfg(target_os = "macos")]
use super::generate_svd;
#[cfg(target_os = "macos")]
use super::generate_wan;
#[cfg(target_os = "macos")]
use super::generate_wan_vace;
#[cfg(target_os = "macos")]
use super::generate_wan_vace_extend_bridge;
#[cfg(target_os = "macos")]
use super::generate_wan_vace_fun;
#[cfg(target_os = "macos")]
use super::ltx_raw_settings;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::reject_unsupported_candle_video_route;
#[cfg(target_os = "macos")]
use super::resolve_wan_vace_model_dir;
#[cfg(native_lane)]
use super::scail2_raw_settings;
#[cfg(native_lane)]
use super::stub_raw_settings;
#[cfg(target_os = "macos")]
use super::svd_raw_settings;
#[cfg(target_os = "macos")]
use super::wan_raw_settings;
#[cfg(native_lane)]
use super::wan_vace_extend_raw_settings;
#[cfg(native_lane)]
use super::wan_vace_raw_settings;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::CandleVideoRoute;
#[cfg(native_lane)]
use super::VideoRequest;
#[cfg(target_os = "macos")]
use super::VideoRoute;
#[cfg(target_os = "macos")]
use super::BERNINI_ADAPTER;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::CANDLE_BERNINI_ADAPTER;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::CANDLE_MINIMAX_H3_ADAPTER;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::CANDLE_SCAIL2_ADAPTER;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::CANDLE_WAN_VACE_ADAPTER;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::CANDLE_WAN_VACE_FUN_ADAPTER;
#[cfg(target_os = "macos")]
use super::KREA_REALTIME_ADAPTER;
#[cfg(target_os = "macos")]
use super::LTX_ADAPTER;
#[cfg(target_os = "macos")]
use super::MINIMAX_H3_ADAPTER;
#[cfg(target_os = "macos")]
use super::MOCHI_ADAPTER;
#[cfg(target_os = "macos")]
use super::SCAIL2_ADAPTER;
#[cfg(native_lane)]
use super::STUB_ADAPTER;
#[cfg(target_os = "macos")]
use super::SVD_ADAPTER;
#[cfg(target_os = "macos")]
use super::WAN_ADAPTER;
#[cfg(target_os = "macos")]
use super::WAN_VACE_ADAPTER;
#[cfg(target_os = "macos")]
use super::WAN_VACE_FUN_ADAPTER;
#[cfg(target_os = "macos")]
use crate::WorkerError;
#[cfg(native_lane)]
use crate::{ApiClient, JobSnapshot, Settings, WorkerResult};
#[cfg(native_lane)]
use std::path::Path;
#[cfg(target_os = "macos")]
// Preserve the existing borrowed job context and one-shot loader ownership.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_mlx_video_route(
    route: VideoRoute,
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    seed: i64,
) -> WorkerResult<VideoRouteOutcome> {
    let (decoded, adapter, raw_settings, replacement_status) = match route {
        // sc-5452: SCAIL-2 is a higher-quality cross-identity replacement backend behind the same
        // YOLO11 → ByteTrack → SAM3 person-track pipeline. A `scail2_14b` person-replace job routes
        // to SCAIL-2 (the tracked person's masks + the character reference → the engine's
        // replacement conditioning, `replace_flag = true`); every other replace-capable model keeps
        // native Wan-VACE (sc-3521). Routed by model id, not weight availability: like the Wan-VACE
        // path, `generate_scail2_replace` resolves-or-errors loudly if the snapshot is unprovisioned
        // (a person-replace must never silently degrade to a different backend or the stub). Both
        // report the honest `replacementStatus` the asset sidecar folds in.
        VideoRoute::ReplacePersonScail2(engine_id) => {
            let (decoded, status, lightning) = generate_scail2_replace(
                api,
                settings,
                job,
                request,
                project_path,
                engine_id,
                backend,
            )
            .await?;
            (
                decoded,
                SCAIL2_ADAPTER,
                scail2_raw_settings(request, lightning),
                Some(status),
            )
        }
        VideoRoute::ReplacePersonWanVaceFun => {
            // sc-3459: the native dual-expert Wan2.2 VACE-Fun engine (`wan2_2_vace_fun_14b`,
            // mlx-gen sc-6604). Same replace_person conditioning as single-expert Wan-VACE, but the
            // dual-expert snapshot + engine — `generate_wan_vace_fun` resolves-or-errors loudly,
            // never falling back to the Wan2.1 `wan_vace` checkpoint.
            let (decoded, status) =
                generate_wan_vace_fun(api, settings, job, request, project_path, backend).await?;
            (
                decoded,
                WAN_VACE_FUN_ADAPTER,
                wan_vace_raw_settings(request, "wan2_2_vace_fun_14b"),
                Some(status),
            )
        }
        VideoRoute::ReplacePersonWanVace => {
            let (decoded, status) =
                generate_wan_vace(api, settings, job, request, project_path, backend).await?;
            (
                decoded,
                WAN_VACE_ADAPTER,
                wan_vace_raw_settings(request, "wan_vace"),
                Some(status),
            )
        }
        VideoRoute::WanVaceExtendBridge => {
            // sc-3812 (tier C): route Wan extend/bridge to native Wan-VACE for genuine motion
            // continuity (the model attends to real source frames, not one boundary still). Falls
            // back to the sc-3357 single-frame TI2V-5B keyframe path when the VACE snapshot is
            // unprovisioned, so the mode keeps working on the weights the user already has. The
            // engine substitution under the `wan_2_2` pick is recorded honestly in raw-settings.
            match resolve_wan_vace_model_dir(settings) {
                Ok(model_dir) => (
                    generate_wan_vace_extend_bridge(
                        api,
                        settings,
                        job,
                        request,
                        project_path,
                        backend,
                        model_dir,
                    )
                    .await?,
                    WAN_VACE_ADAPTER,
                    wan_vace_extend_raw_settings(request),
                    None,
                ),
                Err(_) => {
                    let engine_id = "wan2_2_ti2v_5b";
                    (
                        generate_wan(
                            api,
                            settings,
                            job,
                            request,
                            project_path,
                            engine_id,
                            backend,
                        )
                        .await?,
                        WAN_ADAPTER,
                        wan_raw_settings(request, engine_id),
                        None,
                    )
                }
            }
        }
        VideoRoute::Wan(engine_id) => (
            generate_wan(
                api,
                settings,
                job,
                request,
                project_path,
                engine_id,
                backend,
            )
            .await?,
            WAN_ADAPTER,
            wan_raw_settings(request, engine_id),
            None,
        ),
        VideoRoute::Ltx(engine_id) => {
            let (decoded, status) = generate_ltx(
                api,
                settings,
                job,
                request,
                project_path,
                engine_id,
                backend,
            )
            .await?;
            (decoded, LTX_ADAPTER, ltx_raw_settings(request), status)
        }
        VideoRoute::Svd(engine_id) => (
            generate_svd(
                api,
                settings,
                job,
                request,
                project_path,
                engine_id,
                backend,
            )
            .await?,
            SVD_ADAPTER,
            svd_raw_settings(request),
            None,
        ),
        VideoRoute::Bernini(engine_id) => {
            // Bernini (epic 4699 / sc-4707 + sc-4703): the full Qwen2.5-VL planner + Wan2.2-T2V-A14B
            // renderer. Serves text_to_video + the editing/reference video modes (video_to_video /
            // reference_to_video / reference_video_to_video); `generate_bernini` maps the SceneWorks mode
            // to the engine guidance task and resolves the source media into the planner conditioning.
            // (t2i/i2i image companion = a separate image-typed catalog id, tracked under epic 4699.)
            (
                generate_bernini(
                    api,
                    settings,
                    job,
                    request,
                    project_path,
                    engine_id,
                    backend,
                )
                .await?,
                BERNINI_ADAPTER,
                bernini_raw_settings(request),
                None,
            )
        }
        VideoRoute::Scail2(engine_id) => {
            // SCAIL-2 (epic 5439 / sc-5448): Wan2.1-14B I2V character animation. `generate_scail2`
            // segments the reference image + driving frames with native SAM3, paints the color-coded
            // masks, and maps the SceneWorks mode to the engine task (animate_character → animation;
            // replace_person → replacement is wired in sc-5452). No torch path (mac-only engine).
            // It returns the resolved `lightning` bool so the effective-recipe record uses the
            // same resolution instead of a second `unwrap_or_default` pass (F-118).
            let (decoded, lightning) = generate_scail2(
                api,
                settings,
                job,
                request,
                project_path,
                engine_id,
                backend,
            )
            .await?;
            (
                decoded,
                SCAIL2_ADAPTER,
                scail2_raw_settings(request, lightning),
                None,
            )
        }
        VideoRoute::KreaRealtime(engine_id) => {
            // Krea Realtime 14B (epic 8431 / sc-8443 S10): autoregressive self-forcing video whose
            // shipped checkpoint is Wan 2.1 T2V 14B weight-for-weight. `generate_krea_realtime` maps
            // the supplied media to the engine conditioning (a reference still → i2v `Reference`,
            // strength no-op; a source clip → v2v `VideoClip`, strength honored; else t2v) and drives
            // the shared `generate_video` heartbeat funnel. CFG off; Wan-family LoRAs ride
            // `LoadSpec::adapters` (sc-15017 S15). It returns its own `rawSettings` because the tier
            // that actually LOADED (sc-15258) and any partial LoRA application are only knowable
            // inside the arm — so the record names what actually ran, not what the request asked for.
            let (decoded, raw_settings) = generate_krea_realtime(
                api,
                settings,
                job,
                request,
                project_path,
                engine_id,
                backend,
            )
            .await?;
            (decoded, KREA_REALTIME_ADAPTER, raw_settings, None)
        }
        VideoRoute::Mochi(engine_id) => {
            // Mochi 1 (epic 1788 / sc-11992): 10B AsymmDiT text-to-video, true CFG, pre-quantized
            // tier dirs. `generate_mochi` fetches an opted-into tier on demand, resolves the tier
            // dir + its shared T5/VAE co-requisite, runs the pre-flight memory fit gate (the
            // untiled AsymmVAE decode scales with clip length and MLX's OOM handler is an
            // unmappable `exit(-1)`), then drives the shared `generate_video` funnel. It returns
            // its own `rawSettings` so the record names the tier that was actually loaded.
            let (decoded, raw_settings) =
                generate_mochi(api, settings, job, request, engine_id, backend).await?;
            (decoded, MOCHI_ADAPTER, raw_settings, None)
        }
        VideoRoute::MiniMaxH3(engine_id) => {
            // MiniMax-H3 (epic 17137 / sc-19508): joint audio+video in ONE denoise pass.
            // `generate_minimax_h3` maps the supplied media to the engine conditioning — 0/1/2
            // keyframes for t2va/fl2va, ordered image + audio references for Ref2VA — resolves
            // the tier's DiT partition and the upstream shared components (two different
            // repos), and drives the shared `generate_video` funnel. The synchronized stereo
            // soundtrack rides `DecodedVideo::audio` out of the funnel exactly as LTX's does;
            // no bespoke audio plumbing.
            //
            // It returns its own `rawSettings` because the tier that actually LOADED and the
            // task that actually DENOISED — i.e. which of the two structurally-identical
            // 18.78 GB DiT partitions ran — are only knowable inside the arm, and a
            // wrong-partition render is invisible in the output.
            let (decoded, raw_settings) = generate_minimax_h3(
                api,
                settings,
                job,
                request,
                project_path,
                engine_id,
                backend,
            )
            .await?;
            (decoded, MINIMAX_H3_ADAPTER, raw_settings, None)
        }
        VideoRoute::NoPlanLane => {
            return Err(WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:no-video-lane] checkpoint {:?}: no MLX video lane loads a \
                     checkpoint plan for the {:?} family, and a plan-backed entry never renders \
                     procedural stub output",
                sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(
                    &request.model_manifest_entry
                )
                .unwrap_or_default(),
                request
                    .model_manifest_entry
                    .get("family")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            )));
        }
        VideoRoute::Stub => {
            // An MLX-routed video model whose snapshot didn't resolve must fail
            // loudly with the resolver's precise error instead of completing with
            // procedural stub output (sc-4176, epic 3482 "unsupported jobs error
            // loudly"). replace_person above already follows this rule; the stub
            // remains only for ids outside the engine families (test models,
            // not-yet-ported families).
            wan::ensure_video_engine_weights(request, settings)?;
            (
                generate_stub_video(request, seed),
                STUB_ADAPTER,
                stub_raw_settings(request),
                None,
            )
        }
    };
    Ok(VideoRouteOutcome {
        decoded,
        adapter,
        raw_settings,
        replacement_status,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
// Preserve the existing borrowed job context and one-shot loader ownership.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_candle_video_route(
    route: CandleVideoRoute,
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    seed: i64,
) -> WorkerResult<VideoRouteOutcome> {
    let candle_route = route;
    let (decoded, adapter, raw_settings, replacement_status) = {
        reject_unsupported_candle_video_route(candle_route)?;
        match candle_route {
            // The rejection above is the execution backstop. Keep this arm unreachable so adding a
            // new match site cannot accidentally turn the unsupported route into successful output.
            CandleVideoRoute::UnsupportedEros => {
                unreachable!("unsupported Eros route was rejected")
            }
            // sc-6837 (epic 6563): SCAIL-2 is a distinct cross-identity replacement backend (NOT Wan-VACE)
            // behind the same person-track pipeline. A `scail2_14b` replace_person job routes to the candle
            // SCAIL-2 engine (the character reference + the tracked person's color masks, `replace_flag`),
            // mirroring the macOS `generate_scail2_replace` dispatch; every other replace-capable model
            // keeps candle Wan-VACE. Routed by model id, not weight availability — `generate_candle_scail2_
            // replace` resolves-or-errors loudly (a person-replace must never silently degrade to a stub).
            CandleVideoRoute::ReplacePersonScail2(engine_id) => {
                let (decoded, status, lightning) = generate_candle_scail2_replace(
                    api,
                    settings,
                    job,
                    request,
                    project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_SCAIL2_ADAPTER,
                    scail2_raw_settings(request, lightning),
                    Some(status),
                )
            }
            CandleVideoRoute::ReplacePersonWanVace => {
                // Candle Wan-VACE person replacement (sc-5494) — the candle equivalent of the MLX
                // `wan_vace` path. The router (`video_request_candle_vace_eligible`) already confirmed the
                // candle-VACE model + the source clip + person track before this job reached the candle
                // worker; `generate_candle_wan_vace` resolves-or-errors loudly (a person-replace must never
                // silently degrade to a stub).
                let (decoded, status) =
                    generate_candle_wan_vace(api, settings, job, request, project_path, backend)
                        .await?;
                (
                    decoded,
                    CANDLE_WAN_VACE_ADAPTER,
                    wan_vace_raw_settings(request, "wan_vace"),
                    Some(status),
                )
            }
            CandleVideoRoute::ReplacePersonWanVaceFun => {
                let (decoded, status) = generate_candle_wan_vace_fun(
                    api,
                    settings,
                    job,
                    request,
                    project_path,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_WAN_VACE_FUN_ADAPTER,
                    wan_vace_raw_settings(request, "wan2_2_vace_fun_14b"),
                    Some(status),
                )
            }
            CandleVideoRoute::AnimateScail2(engine_id) => {
                // sc-6837: SCAIL-2 standalone character animation — a reference character + a driving video →
                // an animated clip. `generate_candle_scail2` segments the reference + driving frames with the
                // candle SAM3 segmenter, paints the color-coded masks, and runs the `scail2_14b` engine
                // (`animate_character` → engine task `animation`). The candle sibling of the macOS
                // `generate_scail2`; resolves-or-errors loudly (no torch fallback — a distinct candle engine).
                let (decoded, adapter, raw_settings) = generate_candle_scail2(
                    api,
                    settings,
                    job,
                    request,
                    project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (decoded, adapter, raw_settings, None::<Value>)
            }
            CandleVideoRoute::WanVaceExtendBridge => {
                // Candle Wan-VACE extend/bridge (sc-5494): real source frames pinned at the kept positions +
                // a generated span (the candle equivalent of the MLX `generate_wan_vace_extend_bridge`).
                let decoded = generate_candle_wan_vace_extend_bridge(
                    api,
                    settings,
                    job,
                    request,
                    project_path,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_WAN_VACE_ADAPTER,
                    wan_vace_extend_raw_settings(request),
                    None::<Value>,
                )
            }
            CandleVideoRoute::Bernini(engine_id) => {
                // sc-10997 (epic 6562): the candle Bernini VIDEO lane — the full Qwen planner +
                // Wan2.2-T2V-A14B renderer. `generate_candle_bernini` maps the SceneWorks mode to the engine
                // `video_mode` task (t2v / v2v / r2v / rv2v / mv2v / ads2v) and resolves the source media into
                // the planner conditioning (`resolve_candle_bernini_conditioning`) — empty for t2v, one or
                // more `VideoClip`s for the edit modes, `MultiReference` for the reference modes. The off-Mac
                // sibling of the macOS `generate_bernini`; resolves-or-errors loudly (no torch fallback — a
                // distinct candle engine, GPU-val gated on the `SceneWorks/bernini` weights, sc-11003).
                let decoded = generate_candle_bernini(
                    api,
                    settings,
                    job,
                    request,
                    project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_BERNINI_ADAPTER,
                    bernini_raw_settings(request),
                    None::<Value>,
                )
            }
            CandleVideoRoute::CandleVideo => {
                let VideoRouteOutcome {
                    decoded,
                    adapter,
                    raw_settings,
                    replacement_status: status,
                } = generate_candle_video(api, settings, job, request, project_path, backend)
                    .await?;
                (decoded, adapter, raw_settings, status)
            }
            CandleVideoRoute::MiniMaxH3(engine_id) => {
                let (decoded, raw_settings) = generate_candle_minimax_h3(
                    api,
                    settings,
                    job,
                    request,
                    project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_MINIMAX_H3_ADAPTER,
                    raw_settings,
                    None::<Value>,
                )
            }
            CandleVideoRoute::WanComfyui => {
                let (decoded, adapter, raw_settings) =
                    generate_candle_wan_comfyui(api, settings, job, request, project_path, backend)
                        .await?;
                (decoded, adapter, raw_settings, None::<Value>)
            }
            CandleVideoRoute::Stub => (
                generate_stub_video(request, seed),
                STUB_ADAPTER,
                stub_raw_settings(request),
                None::<Value>,
            ),
        }
    };
    Ok(VideoRouteOutcome {
        decoded,
        adapter,
        raw_settings,
        replacement_status,
    })
}
