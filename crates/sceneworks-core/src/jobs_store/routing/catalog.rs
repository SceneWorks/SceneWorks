//! Model/kernel routing catalog: the per-backend routed-model and training-kernel lists,
//! the Mac support/capability probes, and their supporting types. Moved out of
//! `jobs_store.rs` (sc-8816) with no behavior change. List membership is pinned by the
//! snapshot tests at the bottom of this file.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::jobs_store::routing::gaps::{classify_image_gap, classify_video_gap, UnsupportedReason};
use crate::jobs_store::routing::mlx::{image_request_mlx_eligible, video_mode_is_mlx_eligible};

/// The user-facing affordance prefix the Mac UI shows in place of a torch-only control
/// (sc-3486). Centralised so the API, the web client, and the gap docs read identically.
pub const MAC_NOT_AVAILABLE_LABEL: &str = "Not available on Mac (Rust/MLX only)";

/// UI-facing per-model macOS support (sc-3486), derived from the same `*_mlx_eligible` routing
/// predicates as the [`mac_rust_supported`] job oracle — one source of truth, so what the UI
/// hides can never drift from what routing refuses. `supported` = at least one generation config
/// for this model routes to the in-process Rust/MLX flow on macOS, so the model stays in the
/// picker; `false` = a torch-only model the Mac UI hides/disables once gating is active (its
/// `reason` names the porting epic). The per-feature flags use "available in *some* MLX config"
/// semantics (they never over-gate a valid combination) so a control is disabled only when the
/// model can't use it on MLX at all; residual config-specific dead ends are caught by the
/// `mlx_unsupported` affordance at submit.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMacSupport {
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<UnsupportedReason>,
    pub features: ModelMacFeatures,
}

/// Per-feature macOS support for a model (sc-3486). Each flag mirrors the routing predicate for
/// that feature with "eligible in at least one config" semantics; `false` → disable that control
/// on Mac when gating is active. `video_modes` is populated only for video models.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMacFeatures {
    /// Pose conditioning (the pose picker): a non-empty `advanced.poses`, alone or with a
    /// reference. Base `qwen_image` strict-pose uses the MLX ControlNet path (epic 3401).
    pub pose: bool,
    /// Reference / IP-Adapter / `character_image` identity conditioning (`referenceAssetId`).
    pub reference: bool,
    /// img2img `edit_image` (`mode=edit_image` + a source/reference image).
    pub edit: bool,
    /// Third-party LyCORIS (LoHa / non-peft LoKr) adapters — now applied on every MLX provider
    /// (epic 3641: core loader sc-3642/3643, SDXL/Wan/LTX sc-3671), so `true` for MLX-routed models.
    pub lycoris: bool,
    /// Video-only: which `video_generate` modes route to MLX. Empty for non-video models.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub video_modes: BTreeMap<String, bool>,
}

/// Build a synthetic generation payload (`{ "model": ..., <entries> }`) for probing the routing
/// predicates without a full [`JobSnapshot`] — the UI-gating sibling of how the oracle reads a
/// real job's payload.
pub(crate) fn probe_payload(model: &str, entries: &[(&str, Value)]) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("model".to_owned(), Value::String(model.to_owned()));
    for (key, value) in entries {
        payload.insert((*key).to_owned(), value.clone());
    }
    payload
}

/// UI gating support for a model id of the given catalog `model_type` ("image" / "video" / other).
/// Non-image/video types (utility/infra: upscalers, captioners) are reported `supported` — their
/// Python-only *actions* are gated by [`mac_capabilities`] at the job-type level, not by hiding
/// the model from a picker. Same source of truth as [`mac_rust_supported`].
pub fn model_mac_support(model_id: &str, model_type: &str) -> ModelMacSupport {
    match model_type {
        "image" => image_model_mac_support(model_id),
        "video" => video_model_mac_support(model_id),
        _ => ModelMacSupport {
            supported: true,
            reason: None,
            features: ModelMacFeatures::default(),
        },
    }
}

pub(crate) fn image_model_mac_support(model: &str) -> ModelMacSupport {
    if !MLX_ROUTED_MODELS.contains(&model) {
        return ModelMacSupport {
            supported: false,
            reason: Some(classify_image_gap(&probe_payload(model, &[]))),
            features: ModelMacFeatures::default(),
        };
    }
    // "Available in some MLX config" probes — bias toward not-disabling so a valid combination
    // (e.g. a Z-Image reference, with or without a pose set — sc-3619) is never blocked. Any
    // residual config-only dead ends surface as the `mlx_unsupported` submit affordance.
    let pose = image_request_mlx_eligible(
        model,
        &probe_payload(model, &[("advanced", json!({ "poses": [{}] }))]),
    ) || image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("mode", json!("character_image")),
                ("referenceAssetId", json!("probe")),
                ("advanced", json!({ "poses": [{}] })),
            ],
        ),
    );
    let reference = image_request_mlx_eligible(
        model,
        &probe_payload(model, &[("referenceAssetId", json!("probe"))]),
    ) || image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("mode", json!("character_image")),
                ("referenceAssetId", json!("probe")),
            ],
        ),
    ) || image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("referenceAssetId", json!("probe")),
                ("advanced", json!({ "poses": [{}] })),
            ],
        ),
    );
    let edit = image_request_mlx_eligible(
        model,
        &probe_payload(
            model,
            &[
                ("mode", json!("edit_image")),
                ("sourceAssetId", json!("probe")),
            ],
        ),
    );
    ModelMacSupport {
        supported: true,
        reason: None,
        features: ModelMacFeatures {
            pose,
            reference,
            edit,
            // Third-party LyCORIS applies on every MLX provider now (epic 3641).
            lycoris: true,
            video_modes: BTreeMap::new(),
        },
    }
}

/// The `video_generate` modes the UI offers, in display order, so the gating mirrors
/// [`video_mode_is_mlx_eligible`] for every mode a Mac user could pick. The clip-conditioning
/// modes `extend_clip` / `video_bridge` are included (sc-3773) so the Mac UI gates them
/// per-model — MLX on the LTX IC-LoRA path, torch on Wan — rather than via a coarse global flag.
pub(crate) const VIDEO_UI_MODES: &[&str] = &[
    "text_to_video",
    "image_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
    // Bernini editing / reference-driven video modes (sc-4703) + multi-source modes
    // (sc-5425: `multi_video_to_video` / `ads2v`): only `bernini` is eligible (see
    // `video_mode_is_mlx_eligible`); they surface disabled on the other models, the same
    // per-model gating as `replace_person` / the LTX clip modes.
    "video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    "multi_video_to_video",
    "ads2v",
    // SCAIL-2 standalone character animation (epic 5439 / sc-5448): only `scail2_14b` is
    // eligible; surfaces disabled on the other models. Reference character + driving video
    // → animated clip. (Cross-identity replacement reuses `replace_person`, wired in sc-5452.)
    "animate_character",
];

pub(crate) fn video_model_mac_support(model: &str) -> ModelMacSupport {
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return ModelMacSupport {
            supported: false,
            reason: Some(classify_video_gap(&probe_payload(model, &[]))),
            features: ModelMacFeatures::default(),
        };
    }
    let video_modes = VIDEO_UI_MODES
        .iter()
        .map(|mode| ((*mode).to_owned(), video_mode_is_mlx_eligible(model, mode)))
        .collect();
    ModelMacSupport {
        supported: true,
        reason: None,
        features: ModelMacFeatures {
            video_modes,
            ..ModelMacFeatures::default()
        },
    }
}

/// macOS support for a non-model feature/sub-system (sc-3486): the infra job types that have no
/// in-process Rust path. `supported=false` carries the `reason` (the same `UnsupportedReason` the
/// `mlx_unsupported` event uses); when one of these is ported its flag flips to `true`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacFeatureSupport {
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<UnsupportedReason>,
}

impl MacFeatureSupport {
    // Declares a Mac feature gap with the reason + suggested port epic. Currently no
    // feature is gated (poseFromPhoto was the last, ported in sc-3487/flipped in
    // sc-4206) — kept as the gating vocabulary for the next torch-only surface that
    // appears before its Rust port lands, so a gap is declared the same way every time.
    #[allow(dead_code)]
    fn unsupported(feature: &str, detail: &str, suggested_epic: &str) -> Self {
        Self {
            supported: false,
            reason: Some(UnsupportedReason::new(
                None,
                feature,
                detail,
                Some(suggested_epic),
            )),
        }
    }
}

/// macOS training support (sc-3486): the kernels with a native mlx-gen Rust trainer, so the
/// Training studio can disable a base model whose kernel only runs on the Python torch trainer.
/// `lokr_on_wan_supported=false` mirrors the LoKr-on-Wan routing caveat.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacTrainingSupport {
    pub supported_kernels: Vec<String>,
    pub lokr_on_wan_supported: bool,
}

/// What the Mac UI needs to gate every non-model Python surface plus the master switch
/// (sc-3486). `mac_gating_active` is the rollout flag (`SCENEWORKS_MLX_REQUIRED`): when `false`
/// (Windows/Linux, or a Mac still in observe mode) the client applies no gating at all, so
/// non-Mac pickers are untouched. The per-feature entries are facts about the Rust flow
/// independent of the flag; the client only acts on them when `mac_gating_active`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacCapabilities {
    pub platform: String,
    pub mac_gating_active: bool,
    pub not_available_label: String,
    pub features: BTreeMap<String, MacFeatureSupport>,
    pub training: MacTrainingSupport,
}

/// Build the [`MacCapabilities`] surface for the given platform + gating flag. The feature set is
/// the non-model half of `docs/mac-rust-gaps.md` §5 (infra) plus the global feature gaps; keep it
/// in sync with the oracle's job-type arms.
pub fn mac_capabilities(platform: &str, mac_gating_active: bool) -> MacCapabilities {
    // `std::env::consts::OS` is `"macos"` (the API host's OS, passed by the capabilities handler);
    // accept the legacy `"darwin"` alias defensively. Drives the platform-intrinsic engine flags
    // (e.g. `imageUpscaleSeedvr2`, which is Mac-only) rather than the gating-rollout flag.
    let is_mac = matches!(platform, "macos" | "darwin");
    // SeedVR2 has a backend on Mac (native MLX) and on Windows + Linux (the candle CUDA/NVIDIA port:
    // Windows sc-5928, Linux sc-5160 — candle is CPU+CUDA cross-platform so Linux rides the Windows
    // port). Drives the platform-intrinsic `imageUpscaleSeedvr2` flag.
    let seedvr2_supported = is_mac || matches!(platform, "windows" | "linux");
    let mut features = BTreeMap::new();
    // Third-party LyCORIS (LoHa / non-peft LoKr) now applies on every MLX provider (epic 3641:
    // core loader sc-3642/3643 + SDXL/Wan/LTX sc-3671), so it is no longer a Mac feature gap — the
    // per-model `features.lycoris` flag is `true` and the web LyCORIS upload control is un-gated.
    features.insert(
        // Real-ESRGAN image upscaling is ported to the Rust worker (sc-3489), so the
        // Image Editor upscale tool works on a Python-free Mac. The tool stays available;
        // only the second engine (AuraSR) is dropped, gated per-engine below.
        "imageUpscale".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        // The AuraSR upscale engine (`engine=aura-sr`) is dropped on Mac (sc-3668, port-or-drop
        // spike): it is a 617M-param torch-only GigaGAN with no viable Rust path and only a marginal,
        // ~35-50x-slower quality difference vs the already-ported Real-ESRGAN x4. As of sc-5499 it is
        // also dropped as an OFFERED engine off-Mac — there is no native (MLX/candle) path and the
        // Python torch backend that served it on Windows/Linux is retired in Phase 7 (epic 5483), so
        // exposing it would point users at a path about to disappear. `supported: false` on every
        // platform (platform-intrinsic, like `imageUpscaleSeedvr2`), so the UI hides the engine
        // everywhere. Must agree with the AuraSR arm of `mac_rust_supported` (UI-hidden == routing
        // refuses): the native MLX/candle workers refuse it; only the (transitional) torch worker runs
        // an explicitly-submitted aura-sr job until Phase 7.
        "imageUpscaleAuraSr".to_owned(),
        MacFeatureSupport {
            supported: false,
            reason: Some(UnsupportedReason::new(
                None,
                "image_upscale (AuraSR)",
                "AuraSR is a legacy GAN upscaler, dropped as an offered engine on all platforms (sc-3668 / sc-5499); Real-ESRGAN is the cross-platform upscaler (SeedVR2 the high-fidelity option).",
                Some("sc-5499"),
            )),
        },
    );
    features.insert(
        // SeedVR2 (`engine=seedvr2`) is the one-step diffusion super-resolution upscaler — native MLX
        // on Mac (epic 4811 / sc-4815, in-process `mlx-gen-seedvr2`) and the candle CUDA/NVIDIA port on
        // Windows (sc-5928) + Linux (sc-5160) (epic 5482, `candle-gen-seedvr2`). Both back the same
        // `engine=seedvr2` image upscale + the net-new `video_upscale`. This flag is platform-intrinsic
        // (a backend exists, regardless of the gating rollout flag) so the web upscale picker offers
        // SeedVR2 on every platform that has a backend (Mac, Windows, Linux) and hides it only where
        // there is none (contrast AuraSR, which the UI hides only under active gating). Must agree with
        // the routing oracle (mlx OR candle claims seedvr2; a plain torch worker refuses it).
        "imageUpscaleSeedvr2".to_owned(),
        MacFeatureSupport {
            supported: seedvr2_supported,
            reason: if seedvr2_supported {
                None
            } else {
                // Unreachable on the three platforms that build a SeedVR2 backend (mac/windows/linux);
                // kept for any future platform that has neither MLX nor the candle CUDA/NVIDIA port.
                Some(UnsupportedReason::new(
                    None,
                    "image_upscale (SeedVR2)",
                    "SeedVR2 runs on Mac (native MLX) and Windows/Linux (the candle CUDA/NVIDIA backend); this platform has no SeedVR2 backend.",
                    Some("sc-5160"),
                ))
            },
        },
    );
    features.insert(
        // DWPose pose detection is ported to the Rust worker (sc-3487): RTMW whole-body
        // via `ort`/CoreML on the macOS MLX worker, so the Pose Library "create from
        // photo" flow runs Python-free. This must agree with the PoseDetect arm of
        // `mac_rust_supported` — what the UI hides can never drift from what routing
        // refuses (sc-4206 / F-CORE-2).
        "poseFromPhoto".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        // Person detection + tracking are ported to the Rust worker (sc-3488 /
        // sc-3633/3634/3709): native-MLX YOLO11 detection, SORT/ByteTrack track assembly,
        // and SAM2 per-frame segmentation all run in-process, so the Replace-Person
        // detect → track → mask flow works on a Python-free Mac. (The replace_person
        // video-gen half is gated per-model via each video model's `videoModes`.)
        "personDetect".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt
        // segmentation runs in-process on the macOS Rust worker, so the Image Editor
        // smart-select tool works on a Python-free Mac. Mac-only (no torch SAM3 image
        // path); must agree with the ImageSegment arm of `mac_rust_supported` — what the
        // UI shows == what routing accepts (sc-4206 / F-CORE-2).
        "imageSegment".to_owned(),
        MacFeatureSupport {
            supported: is_mac,
            reason: if is_mac {
                None
            } else {
                Some(UnsupportedReason::new(
                    None,
                    "image_segment (SAM3 smart-select)",
                    "smart-select segmentation runs on the native-MLX SAM3 stack (macOS only); there is no candle SAM3 image path yet.",
                    Some("sc-6105"),
                ))
            },
        },
    );
    features.insert(
        "datasetCaptioning".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    features.insert(
        // Video upscaling is net-new on Mac (epic 4811 / sc-4816): the native-MLX SeedVR2
        // engine gives SceneWorks its first video upscaler, running in-process on the macOS
        // MLX worker (zero-Python). There is no torch fallback (mac-only), so this feature is
        // the gate for the Video Studio "Upscale" action. Must agree with the VideoUpscale arm
        // of `mac_rust_supported` (what the UI shows == what routing accepts).
        "videoUpscale".to_owned(),
        MacFeatureSupport {
            supported: true,
            reason: None,
        },
    );
    // The former global `advancedVideoModes` flag is gone (sc-3773): every video mode — including
    // the LTX IC-LoRA clip-conditioning modes extend_clip / video_bridge — is now gated per-model
    // via each model's `macSupport.features.videoModes`, so a Mac user on LTX is no longer blocked
    // from a mode the in-process Rust worker can run.
    MacCapabilities {
        platform: platform.to_owned(),
        mac_gating_active,
        not_available_label: MAC_NOT_AVAILABLE_LABEL.to_owned(),
        features,
        training: MacTrainingSupport {
            supported_kernels: MLX_ROUTED_TRAINING_KERNELS
                .iter()
                .map(|kernel| (*kernel).to_owned())
                .collect(),
            lokr_on_wan_supported: false,
        },
    }
}

/// Models the in-process Rust MLX worker generates today, by id. This set grows
/// one family story at a time as each lands real generation in
/// `sceneworks-worker::image_jobs` — sc-3022 Z-Image, sc-3023 FLUX.1, sc-3024 Qwen,
/// sc-3025 FLUX.2, sc-3026 SDXL (live). A model id absent here is never routed to the
/// mlx worker, so the Python torch path stays authoritative for it.
pub(crate) const MLX_ROUTED_MODELS: &[&str] = &[
    "z_image_turbo",
    "z_image_edit",
    "flux_schnell",
    "flux_dev",
    "qwen_image",
    "qwen_image_edit",
    "qwen_image_edit_2509",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
    "flux2_klein_9b",
    "flux2_klein_9b_kv",
    "flux2_klein_9b_true_v2",
    // FLUX.2-dev (epic 5914) — MLX-only flagship, txt2img today (edit = sc-5919).
    "flux2_dev",
    "sdxl",
    "realvisxl",
    // RealVisXL Lightning (sc-6075): standalone few-step distilled SDXL checkpoint on the shared
    // `sdxl` engine. txt2img only — the engine's `lightning` accel sampler rejects reference/img2img
    // conditioning (`realvisxl_lightning_mlx_eligible`), so edit/reference shapes fall back to torch.
    "realvisxl_lightning",
    // InstantID on RealVisXL (sc-3345): single-identity + the 11-view angle set route to the
    // native `mlx-gen-instantid` provider. Pose-library + face-restore InstantID jobs are gated
    // OUT by `instantid_mlx_eligible` and stay on the torch `InstantIDAdapter` (engine sc-3117 /
    // sc-3380 not ported).
    "instantid_realvisxl",
    // PuLID-FLUX on FLUX.1-dev (sc-3344): the native `mlx-gen-pulid` registry generator serves
    // `character_image` with a reference face. Mirrors `pulid_flux_mlx_eligible`.
    "pulid_flux_dev",
    "chroma1_hd",
    "chroma1_base",
    "chroma1_flash",
    "sensenova_u1_8b",
    "sensenova_u1_8b_fast",
    // Kolors (epic 3090): the full surface runs on the Rust `kolors` engine model — T2I (sc-3875),
    // img2img (sc-4765), the IP-Adapter-Plus reference (sc-4767) and the strict-pose tier (sc-4766 /
    // engine sc-5012, the combined pose-ControlNet + IP-Adapter-identity + img2img pass).
    "kolors",
    // Microsoft Lens / Lens-Turbo (epic 3164 engine / sc-5105 cutover): pure T2I on the native
    // `mlx-gen-lens` engine (gpt-oss-20b MoE encoder + dual-stream MMDiT + Flux.2 VAE), retiring the
    // Python `/opt/lens-venv` transformers-5 sidecar on Mac. Both ids are always MLX-eligible
    // (`lens_mlx_eligible` — no conditioning surface to gate). Lens was the LAST whole-model
    // torch-only image family; with it routed, every image model here is MLX (`torch_only_image_model_epic`
    // now matches nothing).
    "lens",
    "lens_turbo",
    // Bernini still-image companion (epic 4699 / sc-5424): the image-typed catalog id
    // (`bernini_image`) routes its t2i / i2i (`edit_image`) jobs to the in-process Rust
    // worker, where the same `engine_id:"bernini"` planner+renderer runs with `frames:1`.
    // The video `bernini` id lives in `VIDEO_MLX_ROUTED_MODELS`, not here. Mirrors
    // `bernini_image_mlx_eligible`.
    "bernini_image",
    // Ideogram 4 (epic 4725): 9.3B single-stream flow DiT (asymmetric two-DiT CFG) + Qwen3-VL-8B
    // text encoder on the native `mlx-gen-ideogram` engine (id `ideogram_4`, adapter `mlx_ideogram`),
    // macOS-only (no torch backend). Text-to-image AND img2img/Remix + mask inpaint/outpaint edit
    // (sc-6303 — the engine + worker `resolve_ideogram_edit` edit path); both route to MLX. Mirrors
    // `ideogram_mlx_eligible`.
    "ideogram_4",
    // Ideogram 4 Turbo (mlx-gen #488) — the SAME base model + the bundled TurboTime LoRA the engine
    // installs at load (CFG-free, few-step, single DiT; engine id `ideogram_4_turbo`). Same routing +
    // edit surface as the base (the shared denoise serves both); registered so it reaches the picker
    // and routes to MLX for both T2I and edit (sc-6303). macOS-only (no torch backend).
    "ideogram_4_turbo",
    // Boogu-Image-0.1 (epic 6387): ~10.3B Lumina-Image-2.0 / OmniGen2-lineage flow DiT + Qwen3-VL-8B
    // condition encoder + FLUX.1 VAE on the native `mlx-gen-boogu` engine (adapter `mlx_boogu`),
    // macOS-only (no torch backend, Apache-2.0 ungated). All three route to MLX; mirror
    // `boogu_mlx_eligible`. Base + Turbo are text-to-image only; Edit adds the instruction
    // image-edit `Reference` path (`resolve_boogu_edit`).
    "boogu_image",
    "boogu_image_turbo",
    "boogu_image_edit",
    // Krea 2 Turbo (epic 7565 / sc-7572): native `mlx-gen-krea` text-to-image engine
    // (adapter `mlx_krea`) over the packed Q8/Q4 turnkey. CFG-free Turbo is text-to-image only.
    "krea_2_turbo",
    // Stable Diffusion 3.5 (epic 7841 / S2 sc-7871 worker MODEL_TABLE, surfaced S4 sc-7873): the three
    // native `mlx-gen-sd3` variants (adapter `mlx_sd3`), macOS-only (no torch backend, gated). All are
    // text-to-image only — Large + Medium run true CFG (28 / 40 steps), Turbo is the CFG-free few-step
    // distill (4 steps). `edit_image` has no source/reference path, so it's rejected (mirrors Krea/Lens).
    "sd3_5_large",
    "sd3_5_large_turbo",
    "sd3_5_medium",
    // SANA 1600M (epic 8485 / sc-8489): native `mlx-gen-sana` text-to-image engine (adapter
    // `mlx_sana`) over the un-gated `SceneWorks/Sana_1600M_1024px_mlx` MLX snapshot. macOS-only
    // (no torch backend). True-CFG text-to-image only (20 steps / guidance 4.5); `edit_image` has no
    // source/reference path, so it's rejected (mirrors Krea/SD3.5/Lens).
    "sana_1600m",
    // SANA-Sprint 1.6B (epic 8485 / sc-8490): the CFG-free few-step (default 2-step, SCM sampler +
    // guidance-embed trunk) SANA distillation — same `mlx-gen-sana` engine (adapter `mlx_sana`) over the
    // un-gated `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` MLX snapshot. macOS-only (no torch backend).
    // Text-to-image only; `edit_image` has no source/reference path, so it's rejected (mirrors base SANA).
    "sana_sprint_1600m",
];

/// The models the candle (Windows/CUDA) lane can serve (epic 3672 sc-3678 for SDXL; epic 5095
/// sc-5096 adds the four image families; sc-5126 adds Lens / Lens-Turbo; sc-5484 + sc-5576 add Chroma,
/// Kolors, and SenseNova-U1; sc-7459 adds the two FLUX.2-klein weight variants). Mirrors the worker's
/// `image_jobs::is_candle_engine`: SDXL/RealVisXL (`realvisxl` shares the candle `"sdxl"` engine via a
/// weights swap), plus z-image-turbo, FLUX.1 schnell/dev, FLUX.2-klein-9B (+ the `_kv` / `_true_v2`
/// weight variants, which share the candle `flux2_klein_9b` loader), Qwen-Image,
/// `lens`/`lens_turbo`, `chroma1_hd`/`_base`/`_flash`, `kolors`, and `sensenova_u1_8b`/`_fast` —
/// the base **txt2img** ids (plus the klein weight swaps). Deliberately narrow: candle is a gated
/// txt2img-only lane, so every conditioning shape AND every still-unwired weight variant (e.g.
/// `qwen_image_edit`) falls back to the Python torch worker — including the klein variants' OWN edit /
/// KV-cache shapes (`flux2_klein_9b_kv`'s reference-edit accel is out of scope; sc-7459 is txt2img weight
/// parity only). Lens is pure T2I (no conditioning at all) but — unlike the others — DOES advertise
/// quant + LoRA/LoKr, so it is also listed in [`CANDLE_QUANT_LORA_MODELS`] below to exempt it from the
/// quant/LoRA → torch fallbacks.
pub(crate) const CANDLE_ROUTED_MODELS: &[&str] = &[
    "sdxl",
    "realvisxl",
    // RealVisXL Lightning (sc-7176, the candle half of sc-6128): the standalone few-step distilled
    // checkpoint shares the candle `sdxl` engine via a weights swap (like `realvisxl`), driven on the
    // engine's few-step `lightning` (Euler-trailing, CFG-off) sampler the worker forces for this id.
    // **txt2img only** — its edit / reference / mask / pose shapes are rejected below
    // (`image_request_candle_eligible`) and fall back to the Python torch worker, exactly as the MLX
    // `realvisxl_lightning_mlx_eligible` gate restricts the macOS path (the accel sampler is
    // conditioning-incompatible).
    "realvisxl_lightning",
    "z_image_turbo",
    // Base (non-distilled, full-CFG) Z-Image (epic 8236, sc-8379 control + sc-8679 txt2img): same candle
    // `z_image_diffusers` family as Turbo. Now routed for BOTH the strict-control lane (`z_image` +
    // `advanced.poses` → `generate_candle_zimage_control_stream`, the base Fun-Controlnet-Union branch)
    // AND plain txt2img (sc-8679): the registered candle `z_image` base generator (shift-6.0 / ~50-step /
    // real CFG) runs a non-pose `z_image` job through the generic candle txt2img lane, the base sibling of
    // `z_image_turbo`. Edit/reference/mask shapes still defer to torch (`image_request_candle_eligible`).
    "z_image",
    "flux_schnell",
    "flux_dev",
    "flux2_klein_9b",
    // FLUX.2-klein weight variants (sc-7459, epic 6564 story 3): same candle `flux2_klein_9b`
    // loader/arch, a weights swap. `_kv` is a separately-distilled checkpoint with a full diffusers
    // tree (4-step, guidance 1.0); `_true_v2` is the wikeeyang undistilled fine-tune (24-step,
    // guidance 1.0) the convert-at-install lane assembles into a local diffusers dir (loaded via the
    // `modelPath` seam — candle converter sc-7459). **txt2img only**: `_kv`'s reference-edit / KV-cache
    // accel and every reference/edit shape are rejected below (`image_request_candle_eligible`) and
    // fall back to the Python torch worker.
    "flux2_klein_9b_kv",
    "flux2_klein_9b_true_v2",
    // FLUX.2-dev (epic 6564 sc-7458): the guidance-distilled 32B flagship, a SEPARATE candle engine
    // from klein (Mistral3 TE + 48/48/15360 DiT), registered by `candle-gen-flux2`'s `flux2_dev`
    // generator (sc-7457). Off-Mac the candle lane loads the dense `black-forest-labs/FLUX.2-dev`
    // diffusers snapshot and Q4-quantizes it at load (CPU-stage → quantize-onto-GPU; the 32B doesn't
    // fit the GPU dense) — the manifest `mlx.quantize: 4` + the dev descriptor's `supported_quants`
    // drive that through the shared `resolve_quant` gate, so it needs no per-payload quant request for
    // txt2img. Its edit / multi-reference shapes (`Flux2Edit::load_dev`) and strict-pose Fun-Controlnet-
    // Union (`Flux2Control`) are now candle lanes too (sc-7736): both branch out of
    // `image_job_is_candle_eligible` BEFORE the txt2img gate (the edit + control eligibility predicates).
    "flux2_dev",
    "qwen_image",
    "lens",
    "lens_turbo",
    // epic 3692 candle image families. Chroma's worker lane (#658) shipped without this router half, so
    // chroma jobs never reached the candle worker — added here with Kolors + SenseNova-U1 (sc-5576). All
    // pure **txt2img** on candle: their edit / IP-reference / pose-control / VQA shapes are rejected
    // below (`image_request_candle_eligible`) and fall back to the Python torch worker.
    "chroma1_hd",
    "chroma1_base",
    "chroma1_flash",
    "kolors",
    "sensenova_u1_8b",
    "sensenova_u1_8b_fast",
    // Ideogram 4 (sc-6597, epic 6561): the candle `candle-gen-ideogram` provider serves `ideogram_4`
    // (asymmetric two-DiT CFG) and `ideogram_4_turbo` (CFG-free single DiT + bundled TurboTime LoRA).
    // Pure **txt2img** on candle for now — edit / img2img / mask shapes are rejected below
    // (`image_request_candle_eligible`) and fall back to the Python torch worker until the candle edit
    // lane lands (sc-6598). These ids are also in `MLX_ROUTED_MODELS` (the macOS native engine).
    "ideogram_4",
    "ideogram_4_turbo",
    // Boogu-Image-0.1 (sc-7524, epic 6831): the candle `candle-gen-boogu` provider serves `boogu_image`
    // (Base, true-CFG), `boogu_image_turbo` (DMD few-step, CFG-free), and `boogu_image_edit` (single-
    // reference instruction TI2I). Base + Turbo are pure **txt2img** on candle; `boogu_image_edit`'s
    // `edit_image` shape is handled by the bespoke `boogu_edit_candle_eligible` branch in
    // `image_job_is_candle_eligible` (the source `Reference` is resolved in-lane by the worker's
    // `generate_candle_stream`, like Ideogram — NOT a separate bespoke stream). bf16-only (the provider
    // rejects on-the-fly quant), so a deliberate quant request defers below — boogu is intentionally NOT
    // in `CANDLE_QUANT_LORA_MODELS`. Apache-2.0 ungated. These ids are also in `MLX_ROUTED_MODELS` (the
    // macOS native engine); mirror `boogu_mlx_eligible`.
    "boogu_image",
    "boogu_image_turbo",
    "boogu_image_edit",
    // Krea 2 Turbo (epic 7565 P4, sc-7581): the candle `candle-gen-krea` provider serves `krea_2_turbo`
    // (12B single-stream rectified-flow DiT + Qwen3-VL-4B TE + Qwen-Image VAE, TDM-distilled CFG-free
    // few-step). txt2img + inference LoRA/LoKr — `image_request_candle_eligible` accepts the plain shape
    // and a LoRA (Krea is in `CANDLE_LORA_MODELS`; the provider merges a `krea_2_raw`-trained adapter at
    // Turbo inference, sc-7836), but rejects edit / reference / mask / quant (the provider advertises no
    // conditioning shapes and `supported_quants: &[]`). The MODEL_TABLE row + manifest entry are the MLX
    // twin (sc-7572); sc-7581 adds the candle lane (bf16 off the ungated public `krea/Krea-2-Turbo`, the
    // boogu pattern). This id is also in `MLX_ROUTED_MODELS`; mirror `krea_mlx_eligible`.
    "krea_2_turbo",
    // Stable Diffusion 3.5 (sc-7880, epic 7982): the candle `candle-gen-sd3` provider serves
    // `sd3_5_large` (8B MMDiT, true-CFG), `sd3_5_large_turbo` (ADD-distilled few-step, CFG-free), and
    // `sd3_5_medium` (2.5B MMDiT-X dual-attention). All three are pure **txt2img** on candle — the
    // generic `image_request_candle_eligible` gate accepts the plain shape and rejects edit / reference /
    // mask / pose / LoRA (the descriptor advertises `supports_lora: false`). The provider DOES advertise
    // Q4/Q8 (sc-7879), so an explicit quant request stays on the candle lane (listed in
    // `CANDLE_QUANT_MODELS` below). The MODEL_TABLE rows + manifest entries are the MLX twin (sc-7871);
    // these ids are also in `MLX_ROUTED_MODELS` (the macOS native engine); mirror `sd3_5_mlx_eligible`.
    "sd3_5_large",
    "sd3_5_large_turbo",
    "sd3_5_medium",
];

/// The candle image families that advertise on-the-fly Q4/Q8 quant AND LoRA/LoKr adapters — Lens /
/// Lens-Turbo (sc-5126), the first such candle family. For these a LoRA or an explicit quant request
/// does NOT force the job to the Python torch worker: the candle `generate_candle_stream` maps both
/// into the `LoadSpec` (descriptor-gated, see `ResolvedModel::supports_quant`/`supports_adapters`).
/// Every other candle family advertises neither, so a LoRA/quant request there still defers to torch.
pub(crate) const CANDLE_QUANT_LORA_MODELS: &[&str] = &["lens", "lens_turbo"];

/// The candle image families that advertise on-the-fly Q4/Q8 quant but NOT inference LoRA — Stable
/// Diffusion 3.5 (sc-7880): the `candle-gen-sd3` descriptor advertises `supported_quants: [Q4, Q8]`
/// with `supports_lora: false`. For these an explicit quant request stays on the candle lane (the
/// worker's `generate_candle_stream` resolves it descriptor-side via `model.supports_quant()`), but a
/// LoRA still defers to the Python torch worker. `CANDLE_QUANT_LORA_MODELS` (Lens) is the superset that
/// also keeps LoRAs on candle; these two lists are disjoint and the gate consults both.
pub(crate) const CANDLE_QUANT_MODELS: &[&str] =
    &["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"];

/// The candle image families that advertise inference LoRA/LoKr adapters but NOT on-the-fly quant —
/// Krea 2 Turbo (sc-7836): `candle-gen-krea` merges a `krea_2_raw`-trained LoRA/LoKr into the dense DiT
/// (`supports_lora`/`supports_lokr: true`), but ships `supported_quants: &[]` (dense bf16 only). For
/// these a LoRA stays on the candle lane (the worker's `generate_candle_stream` resolves it via
/// `model.supports_adapters()`) while an explicit quant request still defers to the Python torch
/// worker. The mirror of `CANDLE_QUANT_MODELS` (quant-not-LoRA); `CANDLE_QUANT_LORA_MODELS` (Lens) is
/// the both-set. The three lists are disjoint and the gate consults all three. (sc-7836 landed the
/// candle-gen engine merge + descriptor un-gate; the SceneWorks-side router un-gate here was missed
/// because the sc-7837 GPU validation ran through the `candle-gen-krea` test harness, not a real
/// `image_generate` submission — so a Krea LoRA job hit the no-torch-fallback gap instead of candle.)
pub(crate) const CANDLE_LORA_MODELS: &[&str] = &["krea_2_turbo"];

/// The video models the candle (Windows/CUDA) lane serves: the base txt2video engines `wan_2_2`
/// (→ candle `wan2_2_ti2v_5b`) and `ltx_2_3` (→ candle `ltx_2_3_distilled`) (epic 5095, sc-5097),
/// plus the Wan2.2 **14B** MoE pair `wan_2_2_t2v_14b` (text-only) and `wan_2_2_i2v_14b` (image→video)
/// (sc-5175), plus `svd` (→ candle `svd_xt`, image→video, sc-5493 / epic 5481). Mirrors the worker's
/// `video_jobs::candle_video_engine_id`. `ltx_2_3_eros` (sc-5495) now routes to candle for plain
/// text-to-video too — it's a full dense LTX-2.3 fine-tune → the same `ltx_2_3_distilled` engine, just
/// its own weights repo; every conditioned mode (first_last_frame / extend / bridge / replace) + LoRA
/// still stays on the Python torch worker. Note the 14B I2V and SVD are image→video, NOT txt2video —
/// see [`CANDLE_VIDEO_I2V_ROUTED_MODELS`].
pub(crate) const CANDLE_VIDEO_ROUTED_MODELS: &[&str] = &[
    "wan_2_2",
    "ltx_2_3",
    "ltx_2_3_eros",
    "wan_2_2_t2v_14b",
    "wan_2_2_i2v_14b",
    "svd",
];

/// The candle video models that run **image→video** (a source image is required), not txt2video: the
/// Wan2.2 14B I2V MoE (sc-5175) and SVD (`svd` → `svd_xt`, sc-5493). Their candle providers condition on
/// a source frame, so their eligibility gate requires `mode=image_to_video` + a non-empty
/// `sourceAssetId` — the inverse of the txt2video-only gate the 5B / T2V-14B / ltx ids use.
pub(crate) const CANDLE_VIDEO_I2V_ROUTED_MODELS: &[&str] = &["wan_2_2_i2v_14b", "svd"];

/// The candle video models eligible for the Wan-VACE advanced modes (sc-5494). These route to the
/// single candle `wan_vace` engine regardless of the user's wan pick. The SCAIL-2 person-replace
/// backend is MLX-only, so `scail2_*` is deliberately absent (those stay on the torch / mac worker).
pub(crate) const CANDLE_VIDEO_VACE_MODELS: &[&str] =
    &["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"];

/// Video models the in-process Rust MLX worker generates today (sc-3034 Wan2.2,
/// sc-3035 LTX-2.3 + audio, sc-3523 SVD-XT image→video). Mirrors
/// `MlxVideoAdapter._supported_models`. A model id absent here is never routed to the
/// mlx worker — the Python torch path stays authoritative for it.
pub(crate) const VIDEO_MLX_ROUTED_MODELS: &[&str] = &[
    "ltx_2_3",
    "ltx_2_3_eros",
    "wan_2_2",
    "wan_2_2_t2v_14b",
    "wan_2_2_i2v_14b",
    "svd",
    // Bernini (epic 4699 / sc-4707): full Qwen2.5-VL planner + Wan2.2-T2V-A14B
    // renderer, native MLX (engine id "bernini"). Slice A serves text_to_video
    // only; the editing/reference video modes (v2v/mv2v/r2v/rv2v/ads2v) are
    // net-new UI vocabulary tracked under sc-4703.
    "bernini",
    // SCAIL-2 (epic 5439 / sc-5448): Wan2.1-14B I2V end-to-end character animation,
    // native MLX (engine id "scail2_14b"). Serves the standalone `animate_character`
    // mode; cross-identity `replace_person` reuses the same engine, wired in sc-5452.
    "scail2_14b",
];

/// SceneWorks training kernels with a native mlx-gen Rust trainer (epic 3039):
/// the engine registers `z_image_turbo`/`sdxl`/`kolors`/`ltx_2_3`/`wan2_2_*` trainers,
/// which the worker reaches via these SceneWorks kernel ids (the mlx worker maps the
/// kernel and base model onto an engine trainer id). `kolors_lora` (SDXL U-Net plus
/// ChatGLM3) gained a native trainer in sc-4568, cut over here in sc-4732. `lens_lora`
/// gained a native mlx-gen-lens trainer in sc-5148, cut over here in sc-5180 (off-Mac
/// keeps the Python sidecar trainer). `krea_lora` is the native `mlx-gen-krea` trainer
/// (sc-7577); it has no torch path, so it is also listed in `MLX_ONLY_TRAINING_KERNELS`
/// (sc-7578). `sd3_lora` is the native `mlx-gen-sd3` trainer (sc-7883/7885; Large +
/// MMDiT-X Medium training bases), cut over here in sc-7884; it has no torch path either,
/// so it is also in `MLX_ONLY_TRAINING_KERNELS`. A kernel absent here is never routed to
/// the mlx worker.
pub(crate) const MLX_ROUTED_TRAINING_KERNELS: &[&str] = &[
    "z_image_lora",
    "sdxl_lora",
    "kolors_lora",
    "lens_lora",
    "krea_lora",
    "sd3_lora",
    "wan_lora",
    "wan_moe_lora",
    "ltx_mlx_lora",
];

/// SceneWorks training kernels with a native candle trainer that needs no base-model disambiguation
/// (sc-7817, epic 5164) — the off-Mac twin of [`MLX_ROUTED_TRAINING_KERNELS`]. The candle registry
/// holds trainers for `sdxl`, `z_image_turbo`, `lens`, `krea_2_raw` (the Krea 2 Raw 12B DiT, epic
/// 7565 P4 — sc-8614 wires it here), and the Wan **A14B T2V** MoE (`wan2_2_t2v_14b`); the first four
/// map straight from kernel, while `wan_moe_lora` is base-model gated (handled in
/// [`training_job_is_candle_eligible`]). UNLIKE the torch families (z-image/sdxl/lens/wan), Krea has
/// NO torch trainer — it is in BOTH this set and [`MLX_ONLY_TRAINING_KERNELS`] (Rust-only: mlx OR
/// candle, never torch). The dense Wan 5B + the I2V A14B have no candle trainer yet (sc-5167
/// follow-ups) and Kolors/LTX none at all — those kernels stay on the Python torch worker off-Mac.
pub(crate) const CANDLE_ROUTED_TRAINING_KERNELS: &[&str] =
    &["z_image_lora", "sdxl_lora", "lens_lora", "krea_lora"];

/// Training kernels with NO **torch** fallback — only a Rust worker can run them, so a torch worker
/// must refuse the job (leaving it queued for a Rust worker) rather than claim it and fail with "no
/// training kernel". `ltx_mlx_lora` was Apple-Silicon-only MLX-Python; epic 3039 (sc-3049) retired
/// that Python trainer, leaving the native Rust LTX trainer as the sole path. The torch families
/// (z-image/sdxl/wan) keep their Python trainer as the Windows path + Mac fallback, so they are
/// deliberately NOT listed here. `krea_lora` (epic 7565) is Rust-native with no torch trainer — but
/// UNLIKE LTX/SD3 it now ALSO has a candle trainer (sc-8614, P4), so it is the one member that runs
/// on EITHER Rust backend (mlx in-process on Mac, candle off-Mac); the [`worker_supports_job`] gate
/// exempts a candle worker for a `krea_lora` job it is candle-eligible for (it is also in
/// [`CANDLE_ROUTED_TRAINING_KERNELS`]), while torch is still refused. `sd3_lora` (epic 7841 T3
/// sc-7884) is MLX-native with no torch trainer and no candle trainer yet (the off-Mac/candle SD3.5
/// trainer is epic 7982), so — like LTX — only an mlx worker runs it today.
pub(crate) const MLX_ONLY_TRAINING_KERNELS: &[&str] = &["ltx_mlx_lora", "krea_lora", "sd3_lora"];

#[cfg(test)]
mod tests {
    //! Membership-parity regression guard (sc-8816): every routed-model / routed-kernel
    //! list relocated into this catalog module is pinned to a snapshot of its current
    //! contents so an accidental future edit (add / remove / reorder) is caught, and the
    //! documented superset invariant (every quant/lora id is also a candle-routed id) is
    //! asserted against current reality.
    use super::{
        CANDLE_LORA_MODELS, CANDLE_QUANT_LORA_MODELS, CANDLE_QUANT_MODELS, CANDLE_ROUTED_MODELS,
        CANDLE_ROUTED_TRAINING_KERNELS, CANDLE_VIDEO_I2V_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS,
        CANDLE_VIDEO_VACE_MODELS, MLX_ONLY_TRAINING_KERNELS, MLX_ROUTED_MODELS,
        MLX_ROUTED_TRAINING_KERNELS, VIDEO_MLX_ROUTED_MODELS,
    };

    const EXPECTED_MLX_ROUTED_MODELS: &[&str] = &[
        "z_image_turbo",
        "z_image_edit",
        "flux_schnell",
        "flux_dev",
        "qwen_image",
        "qwen_image_edit",
        "qwen_image_edit_2509",
        "qwen_image_edit_2511",
        "qwen_image_edit_2511_lightning",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
        "flux2_dev",
        "sdxl",
        "realvisxl",
        "realvisxl_lightning",
        "instantid_realvisxl",
        "pulid_flux_dev",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "kolors",
        "lens",
        "lens_turbo",
        "bernini_image",
        "ideogram_4",
        "ideogram_4_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        "krea_2_turbo",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sana_1600m",
        "sana_sprint_1600m",
    ];

    const EXPECTED_CANDLE_ROUTED_MODELS: &[&str] = &[
        "sdxl",
        "realvisxl",
        "realvisxl_lightning",
        "z_image_turbo",
        "z_image",
        "flux_schnell",
        "flux_dev",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
        "flux2_dev",
        "qwen_image",
        "lens",
        "lens_turbo",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "kolors",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "ideogram_4",
        "ideogram_4_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        "krea_2_turbo",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
    ];

    const EXPECTED_CANDLE_QUANT_LORA_MODELS: &[&str] = &["lens", "lens_turbo"];

    const EXPECTED_CANDLE_QUANT_MODELS: &[&str] =
        &["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"];

    const EXPECTED_CANDLE_LORA_MODELS: &[&str] = &["krea_2_turbo"];

    const EXPECTED_CANDLE_VIDEO_ROUTED_MODELS: &[&str] = &[
        "wan_2_2",
        "ltx_2_3",
        "ltx_2_3_eros",
        "wan_2_2_t2v_14b",
        "wan_2_2_i2v_14b",
        "svd",
    ];

    const EXPECTED_CANDLE_VIDEO_I2V_ROUTED_MODELS: &[&str] = &["wan_2_2_i2v_14b", "svd"];

    const EXPECTED_CANDLE_VIDEO_VACE_MODELS: &[&str] =
        &["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"];

    const EXPECTED_VIDEO_MLX_ROUTED_MODELS: &[&str] = &[
        "ltx_2_3",
        "ltx_2_3_eros",
        "wan_2_2",
        "wan_2_2_t2v_14b",
        "wan_2_2_i2v_14b",
        "svd",
        "bernini",
        "scail2_14b",
    ];

    const EXPECTED_MLX_ROUTED_TRAINING_KERNELS: &[&str] = &[
        "z_image_lora",
        "sdxl_lora",
        "kolors_lora",
        "lens_lora",
        "krea_lora",
        "sd3_lora",
        "wan_lora",
        "wan_moe_lora",
        "ltx_mlx_lora",
    ];

    const EXPECTED_CANDLE_ROUTED_TRAINING_KERNELS: &[&str] =
        &["z_image_lora", "sdxl_lora", "lens_lora", "krea_lora"];

    const EXPECTED_MLX_ONLY_TRAINING_KERNELS: &[&str] = &["ltx_mlx_lora", "krea_lora", "sd3_lora"];

    #[test]
    fn routed_model_lists_match_snapshot() {
        assert_eq!(MLX_ROUTED_MODELS, EXPECTED_MLX_ROUTED_MODELS);
        assert_eq!(CANDLE_ROUTED_MODELS, EXPECTED_CANDLE_ROUTED_MODELS);
        assert_eq!(CANDLE_QUANT_LORA_MODELS, EXPECTED_CANDLE_QUANT_LORA_MODELS);
        assert_eq!(CANDLE_QUANT_MODELS, EXPECTED_CANDLE_QUANT_MODELS);
        assert_eq!(CANDLE_LORA_MODELS, EXPECTED_CANDLE_LORA_MODELS);
        assert_eq!(
            CANDLE_VIDEO_ROUTED_MODELS,
            EXPECTED_CANDLE_VIDEO_ROUTED_MODELS
        );
        assert_eq!(
            CANDLE_VIDEO_I2V_ROUTED_MODELS,
            EXPECTED_CANDLE_VIDEO_I2V_ROUTED_MODELS
        );
        assert_eq!(CANDLE_VIDEO_VACE_MODELS, EXPECTED_CANDLE_VIDEO_VACE_MODELS);
        assert_eq!(VIDEO_MLX_ROUTED_MODELS, EXPECTED_VIDEO_MLX_ROUTED_MODELS);
        assert_eq!(
            MLX_ROUTED_TRAINING_KERNELS,
            EXPECTED_MLX_ROUTED_TRAINING_KERNELS
        );
        assert_eq!(
            CANDLE_ROUTED_TRAINING_KERNELS,
            EXPECTED_CANDLE_ROUTED_TRAINING_KERNELS
        );
        assert_eq!(
            MLX_ONLY_TRAINING_KERNELS,
            EXPECTED_MLX_ONLY_TRAINING_KERNELS
        );
    }

    #[test]
    fn quant_and_lora_models_are_candle_routed_supersets() {
        for id in CANDLE_QUANT_LORA_MODELS
            .iter()
            .chain(CANDLE_QUANT_MODELS)
            .chain(CANDLE_LORA_MODELS)
        {
            assert!(
                CANDLE_ROUTED_MODELS.contains(id),
                "{id} must also be in CANDLE_ROUTED_MODELS (superset invariant)"
            );
        }
    }
}
