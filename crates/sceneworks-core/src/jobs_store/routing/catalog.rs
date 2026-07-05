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

// ---------------------------------------------------------------------------------------------
// Per-model routing capability tables (sc-9495)
// ---------------------------------------------------------------------------------------------
//
// F-014 follow-up: the per-backend routed-model lists used to live as ~9 parallel `&[&str]`
// constants (MLX_ROUTED_MODELS, CANDLE_ROUTED_MODELS, CANDLE_QUANT_LORA_MODELS,
// CANDLE_QUANT_MODELS, CANDLE_LORA_MODELS + the 4 video variants), so a single model's routing
// facts were scattered across up to five edit sites — the "engine wired but router half missed"
// bug class (chroma sc-5576, krea sc-7836). They are now collapsed into ONE row per model in the
// tables below, and every legacy list constant is DERIVED from a table column at compile time via
// [`derive_model_list!`] — so adding or changing a model's routing is a single-row edit, and the
// list constants (which the routing predicates + oracles still `.contains()`/iterate exactly as
// before) can never drift from one another. The membership-parity test at the bottom of this file
// pins each derived constant against the pre-collapse snapshot (zero-diff guardrail), and a
// superset test asserts the quant/lora columns imply the candle-routed column.

/// One image model's per-backend routing capabilities (sc-9495). Each boolean is the model's
/// membership in what used to be a standalone routing list; the predicates in `mlx.rs` / `candle.rs`
/// / `gaps.rs` consult the DERIVED list constants below (byte-identical membership), so behavior is
/// unchanged — this struct is purely the single source those constants are generated from.
///
/// **Superset invariant (enforced):** `candle_quant`, `candle_lora`, and `candle_quant_lora` all
/// imply `candle_routed` — a model can only advertise on-the-fly quant / inference LoRA on the candle
/// lane if it is candle-routed at all. Encoded structurally by [`ModelCaps::new`] (a `debug_assert`
/// on every constructed row) and asserted exhaustively over the table by the
/// `quant_and_lora_columns_are_candle_routed_supersets` test.
#[derive(Clone, Copy)]
pub(crate) struct ModelCaps {
    /// Model id.
    pub(crate) id: &'static str,
    /// In-process Rust MLX worker generates this model on macOS (was `MLX_ROUTED_MODELS`).
    pub(crate) mlx_routed: bool,
    /// The candle (Windows/CUDA) lane serves this model's base txt2img (was `CANDLE_ROUTED_MODELS`).
    pub(crate) candle_routed: bool,
    /// Candle advertises on-the-fly Q4/Q8 quant but NOT inference LoRA (was `CANDLE_QUANT_MODELS`).
    pub(crate) candle_quant: bool,
    /// Candle advertises inference LoRA/LoKr but NOT on-the-fly quant (was `CANDLE_LORA_MODELS`).
    pub(crate) candle_lora: bool,
    /// Candle advertises BOTH on-the-fly quant AND inference LoRA (was `CANDLE_QUANT_LORA_MODELS`).
    pub(crate) candle_quant_lora: bool,
}

impl ModelCaps {
    const fn new(
        id: &'static str,
        mlx_routed: bool,
        candle_routed: bool,
        candle_quant: bool,
        candle_lora: bool,
        candle_quant_lora: bool,
    ) -> Self {
        // Superset invariant: any quant/lora capability implies the model is candle-routed. A const
        // `assert!` makes a violating row a COMPILE error (evaluated when the table `const` is built).
        assert!(
            candle_routed || !(candle_quant || candle_lora || candle_quant_lora),
            "quant/lora capability implies candle_routed (sc-9495 superset invariant)"
        );
        Self {
            id,
            mlx_routed,
            candle_routed,
            candle_quant,
            candle_lora,
            candle_quant_lora,
        }
    }
}

/// One video model's per-backend routing capabilities (sc-9495) — the video-namespace sibling of
/// [`ModelCaps`]. Collapses the 4 parallel video lists (`VIDEO_MLX_ROUTED_MODELS`,
/// `CANDLE_VIDEO_ROUTED_MODELS`, `CANDLE_VIDEO_I2V_ROUTED_MODELS`, `CANDLE_VIDEO_VACE_MODELS`) into
/// one row per model.
///
/// **Superset invariant (enforced):** `candle_video_i2v` and `candle_video_vace` both imply
/// `candle_video_routed` — the i2v-only and VACE-mode gates only ever run on a candle-video-routed
/// model. Encoded by [`VideoModelCaps::new`] + asserted by the superset test.
#[derive(Clone, Copy)]
pub(crate) struct VideoModelCaps {
    /// Video model id.
    pub(crate) id: &'static str,
    /// In-process Rust MLX worker generates this video model (was `VIDEO_MLX_ROUTED_MODELS`).
    pub(crate) video_mlx_routed: bool,
    /// The candle lane serves this video model's base txt2video (was `CANDLE_VIDEO_ROUTED_MODELS`).
    pub(crate) candle_video_routed: bool,
    /// Candle serves this model as image→video ONLY, not txt2video (was `CANDLE_VIDEO_I2V_ROUTED_MODELS`).
    pub(crate) candle_video_i2v: bool,
    /// Candle serves the Wan-VACE advanced modes for this model (was `CANDLE_VIDEO_VACE_MODELS`).
    pub(crate) candle_video_vace: bool,
}

impl VideoModelCaps {
    const fn new(
        id: &'static str,
        video_mlx_routed: bool,
        candle_video_routed: bool,
        candle_video_i2v: bool,
        candle_video_vace: bool,
    ) -> Self {
        assert!(
            candle_video_routed || !(candle_video_i2v || candle_video_vace),
            "candle video i2v/vace capability implies candle_video_routed (sc-9495 superset invariant)"
        );
        Self {
            id,
            video_mlx_routed,
            candle_video_routed,
            candle_video_i2v,
            candle_video_vace,
        }
    }
}

/// The one-row-per-model image routing table (sc-9495) — the single source the image list constants
/// below are derived from. Column meanings + the porting-story history that used to live as inline
/// comments on each standalone list are documented per-row here; each model is now ONE edit site.
///
/// Legend for the [`ModelCaps::new`] positional args:
/// `new(id, mlx_routed, candle_routed, candle_quant, candle_lora, candle_quant_lora)`.
pub(crate) const IMAGE_MODEL_CAPS: &[ModelCaps] = &[
    // sc-3022 Z-Image / sc-3023 FLUX.1 / sc-3024 Qwen / sc-3025 FLUX.2 / sc-3026 SDXL — the founding
    // MLX-routed families (grows one family story at a time as each lands real generation in
    // `sceneworks-worker::image_jobs`). CANDLE: SDXL sc-3678, the four families sc-5096.
    ModelCaps::new("z_image_turbo", true, true, false, false, false),
    // Base (non-distilled, full-CFG) Z-Image (epic 8236, sc-8379 control + sc-8679 txt2img): candle-only
    // (no MLX row) — the base sibling of `z_image_turbo` on the candle `z_image_diffusers` family.
    ModelCaps::new("z_image", false, true, false, false, false),
    // `z_image_edit` (epic 3529 / sc-3923): MLX-only edit id on Turbo weights.
    ModelCaps::new("z_image_edit", true, false, false, false, false),
    ModelCaps::new("flux_schnell", true, true, false, false, false),
    ModelCaps::new("flux_dev", true, true, false, false, false),
    ModelCaps::new("qwen_image", true, true, false, false, false),
    // Qwen-Image-Edit ids (sc-3397/3398): MLX edit siblings; candle serves them via the bespoke
    // `qwen_edit_candle_eligible` lane (NOT the txt2img gate), so they are NOT candle-routed txt2img ids.
    ModelCaps::new("qwen_image_edit", true, false, false, false, false),
    ModelCaps::new("qwen_image_edit_2509", true, false, false, false, false),
    ModelCaps::new("qwen_image_edit_2511", true, false, false, false, false),
    ModelCaps::new(
        "qwen_image_edit_2511_lightning",
        true,
        false,
        false,
        false,
        false,
    ),
    // FLUX.2-klein-9B + the `_kv` / `_true_v2` weight variants share the candle `flux2_klein_9b` loader
    // (sc-7459, a weights swap). **txt2img only** on candle: edit / KV-cache shapes defer to torch.
    ModelCaps::new("flux2_klein_9b", true, true, false, false, false),
    ModelCaps::new("flux2_klein_9b_kv", true, true, false, false, false),
    ModelCaps::new("flux2_klein_9b_true_v2", true, true, false, false, false),
    // FLUX.2-dev (epic 5914 MLX / epic 6564 sc-7458 candle) — the guidance-distilled 32B flagship.
    // A SEPARATE candle engine from klein (Mistral3 TE + 48/48/15360 DiT); Q4-quantized at load off-Mac.
    ModelCaps::new("flux2_dev", true, true, false, false, false),
    ModelCaps::new("sdxl", true, true, false, false, false),
    ModelCaps::new("realvisxl", true, true, false, false, false),
    // RealVisXL Lightning (MLX sc-6075 / candle sc-7176): standalone few-step distilled SDXL checkpoint
    // on the shared `sdxl` engine, few-step `lightning` accel sampler. **txt2img only** on both backends —
    // edit / reference / mask / pose shapes fall back to torch (accel sampler is conditioning-incompatible).
    ModelCaps::new("realvisxl_lightning", true, true, false, false, false),
    // InstantID on RealVisXL (sc-3345): MLX-only id — single-identity + the 11-view angle set route to
    // the native `mlx-gen-instantid` provider (candle serves it via the bespoke `instantid_candle_eligible`
    // lane, not the txt2img gate, so it is NOT a candle-routed txt2img id).
    ModelCaps::new("instantid_realvisxl", true, false, false, false, false),
    // PuLID-FLUX on FLUX.1-dev (sc-3344): MLX-only id — `character_image` with a reference face (candle
    // serves it via the bespoke `pulid_flux_candle_eligible` lane, not the txt2img gate).
    ModelCaps::new("pulid_flux_dev", true, false, false, false, false),
    // Chroma (epic 3531 / sc-3843 MLX; epic 3692 / sc-5576 candle). Pure txt2img on candle.
    ModelCaps::new("chroma1_hd", true, true, false, false, false),
    ModelCaps::new("chroma1_base", true, true, false, false, false),
    ModelCaps::new("chroma1_flash", true, true, false, false, false),
    // SenseNova-U1 (epic 3180 / sc-3900 MLX; sc-5576 candle). Pure txt2img on candle.
    ModelCaps::new("sensenova_u1_8b", true, true, false, false, false),
    ModelCaps::new("sensenova_u1_8b_fast", true, true, false, false, false),
    // Kolors (epic 3090): full surface on the Rust `kolors` engine (SDXL-family U-Net + ChatGLM3);
    // candle serves txt2img + bespoke IP/pose lanes (sc-5488/sc-5489).
    ModelCaps::new("kolors", true, true, false, false, false),
    // Microsoft Lens / Lens-Turbo (epic 3164 / sc-5105 MLX; sc-5126 candle): pure T2I family. UNLIKE the
    // other candle families it DOES advertise on-the-fly quant AND LoRA/LoKr, so `candle_quant_lora` is
    // set — the first (and, with SD3.5/Krea, one of the) candle families exempt from the quant/LoRA → torch
    // fallbacks. Lens was the LAST whole-model torch-only image family — once it routed, the per-model
    // torch-only image epic seam matched nothing and was retired (sc-8951).
    ModelCaps::new("lens", true, true, false, false, true),
    ModelCaps::new("lens_turbo", true, true, false, false, true),
    // Bernini still-image companion (epic 4699 / sc-5424): MLX-only id — `engine_id:"bernini"`
    // planner+renderer with `frames:1`. The video `bernini` id lives in the video table below.
    ModelCaps::new("bernini_image", true, false, false, false, false),
    // Ideogram 4 + Turbo (epic 4725 MLX; sc-6597 candle): 9.3B flow DiT + Qwen3-VL-8B TE. T2I + edit on
    // MLX (sc-6303); candle serves txt2img + the in-lane edit path (sc-6598) via the generic stream.
    // Candle advertises Q4/Q8 (sc-9607 flipped `supported_quants: [Q4, Q8]`, dropping the loader's
    // `spec.quantize` reject — a no-op on the already-packed q4/q8 turnkey), so `candle_quant` is set
    // (sc-9983 — the routing half of sc-9607, previously missed): a tier-select `mlxQuantize` stays on
    // candle. No inference LoRA on candle, so NOT quant/lora-exempt.
    ModelCaps::new("ideogram_4", true, true, true, false, false),
    ModelCaps::new("ideogram_4_turbo", true, true, true, false, false),
    // Boogu-Image-0.1 (epic 6387 MLX; sc-7524 candle): ~10.3B flow DiT + Qwen3-VL-8B + FLUX.1 VAE. Base +
    // Turbo are txt2img; Edit adds the instruction image-edit path. Candle advertises Q4/Q8 (sc-9607
    // flipped `supported_quants: [Q4, Q8]`, the packed-tier no-op), so `candle_quant` is set (sc-9983 —
    // the routing half of sc-9607). No inference LoRA on candle, so NOT quant/lora-exempt.
    ModelCaps::new("boogu_image", true, true, true, false, false),
    ModelCaps::new("boogu_image_turbo", true, true, true, false, false),
    ModelCaps::new("boogu_image_edit", true, true, true, false, false),
    // Krea 2 Turbo (epic 7565 / sc-7572 MLX; sc-7581 candle): 12B rectified-flow DiT, TDM-distilled
    // CFG-free. Candle advertises inference LoRA/LoKr (sc-7836 — merges a `krea_2_raw`-trained adapter at
    // Turbo inference) AND, since sc-9607, on-the-fly Q4/Q8 (`supported_quants: [Q4, Q8]`, a no-op on the
    // already-packed q4/q8 turnkey), so `candle_quant_lora` is set (sc-9983 — the routing half of sc-9607,
    // moving Krea from `candle_lora` to BOTH): a tier-select `mlxQuantize` AND a LoRA both stay on candle.
    ModelCaps::new("krea_2_turbo", true, true, false, false, true),
    // Stable Diffusion 3.5 Large / Large Turbo / Medium (epic 7841 / sc-7871 MLX; sc-7880 candle):
    // pure txt2img. Candle advertises Q4/Q8 (sc-7879) but NOT inference LoRA (`supports_lora: false`), so
    // `candle_quant` is set — an explicit quant request stays on candle while a LoRA still defers to torch.
    ModelCaps::new("sd3_5_large", true, true, true, false, false),
    ModelCaps::new("sd3_5_large_turbo", true, true, true, false, false),
    ModelCaps::new("sd3_5_medium", true, true, true, false, false),
    // SANA 1600M + SANA-Sprint (epic 8485 / sc-8489 / sc-8490): MLX-only txt2img (no torch/candle backend).
    ModelCaps::new("sana_1600m", true, false, false, false, false),
    ModelCaps::new("sana_sprint_1600m", true, false, false, false, false),
];

/// The one-row-per-model VIDEO routing table (sc-9495) — the single source the video list constants
/// below are derived from.
///
/// Legend for the [`VideoModelCaps::new`] positional args:
/// `new(id, video_mlx_routed, candle_video_routed, candle_video_i2v, candle_video_vace)`.
pub(crate) const VIDEO_MODEL_CAPS: &[VideoModelCaps] = &[
    // LTX-2.3 base + eros (sc-3035 MLX; sc-5097 / sc-5495 candle): txt2video on both backends.
    VideoModelCaps::new("ltx_2_3", true, true, false, false),
    VideoModelCaps::new("ltx_2_3_eros", true, true, false, false),
    // Wan2.2 TI2V-5B (sc-3034 MLX; sc-5097 candle): txt2video + VACE advanced modes on candle.
    VideoModelCaps::new("wan_2_2", true, true, false, true),
    // Wan2.2 14B MoE (sc-5175): T2V-14B is text-only; I2V-14B is image→video ONLY (candle_video_i2v).
    // Both are VACE-capable on candle.
    VideoModelCaps::new("wan_2_2_t2v_14b", true, true, false, true),
    VideoModelCaps::new("wan_2_2_i2v_14b", true, true, true, true),
    // SVD (`svd` → `svd_xt`, sc-3523 MLX; sc-5493 candle): image→video ONLY. Not a VACE model.
    VideoModelCaps::new("svd", true, true, true, false),
    // Bernini (epic 4699 / sc-4707): MLX-only — Qwen2.5-VL planner + Wan2.2-T2V-A14B renderer.
    VideoModelCaps::new("bernini", true, false, false, false),
    // SCAIL-2 (epic 5439 / sc-5448): MLX end-to-end character animation; the candle SCAIL-2 engine
    // (sc-6837) is a DISTINCT engine gated by its own predicates, NOT Wan-VACE membership — so its
    // candle-video columns are all false (it is deliberately absent from `CANDLE_VIDEO_*`).
    VideoModelCaps::new("scail2_14b", true, false, false, false),
];

/// Derive a `&'static [&'static str]` list constant from a boolean column of one of the capability
/// tables above (sc-9495). Expands to a compile-time-built array of exactly the ids whose column is
/// `true`, in table-row order, so the generated constant is a drop-in for the hand-written list it
/// replaced (same `&[&str]` type, consumed unchanged by the routing predicates). Every legacy list is
/// one macro invocation — the model rows are the single edit site.
macro_rules! derive_model_list {
    ($(#[$meta:meta])* $vis:vis $name:ident, $table:ident, $field:ident) => {
        $(#[$meta])*
        $vis const $name: &[&str] = {
            const fn count() -> usize {
                let mut n = 0;
                let mut i = 0;
                while i < $table.len() {
                    if $table[i].$field {
                        n += 1;
                    }
                    i += 1;
                }
                n
            }
            const N: usize = count();
            const fn build() -> [&'static str; N] {
                let mut out = [""; N];
                let mut i = 0;
                let mut j = 0;
                while i < $table.len() {
                    if $table[i].$field {
                        out[j] = $table[i].id;
                        j += 1;
                    }
                    i += 1;
                }
                out
            }
            &build()
        };
    };
}

derive_model_list! {
    /// Models the in-process Rust MLX worker generates today, by id (derived from
    /// [`IMAGE_MODEL_CAPS`]`.mlx_routed`, sc-9495). A model id absent here is never routed to the mlx
    /// worker, so the Python torch path stays authoritative for it.
    pub(crate) MLX_ROUTED_MODELS, IMAGE_MODEL_CAPS, mlx_routed
}

derive_model_list! {
    /// The models the candle (Windows/CUDA) lane can serve for base txt2img (derived from
    /// [`IMAGE_MODEL_CAPS`]`.candle_routed`, sc-9495). Mirrors the worker's `image_jobs::is_candle_engine`.
    /// Deliberately narrow: candle is a gated txt2img-only lane, so every conditioning shape falls back to
    /// the Python torch worker unless a bespoke candle lane in `image_job_is_candle_eligible` claims it.
    pub(crate) CANDLE_ROUTED_MODELS, IMAGE_MODEL_CAPS, candle_routed
}

derive_model_list! {
    /// The candle image families that advertise on-the-fly Q4/Q8 quant AND LoRA/LoKr adapters — Lens /
    /// Lens-Turbo and Krea 2 Turbo (derived from [`IMAGE_MODEL_CAPS`]`.candle_quant_lora`, sc-9495; Krea
    /// added sc-9983 once sc-9607 flipped its `supported_quants`). For these a LoRA or an explicit quant
    /// request does NOT force the job to torch. Subset of [`CANDLE_ROUTED_MODELS`].
    pub(crate) CANDLE_QUANT_LORA_MODELS, IMAGE_MODEL_CAPS, candle_quant_lora
}

derive_model_list! {
    /// The candle image families that advertise on-the-fly Q4/Q8 quant but NOT inference LoRA — Stable
    /// Diffusion 3.5, Ideogram 4 (+Turbo), and Boogu (Base/Turbo/Edit) (derived from
    /// [`IMAGE_MODEL_CAPS`]`.candle_quant`, sc-9495; the ideogram/boogu packed families added sc-9983 once
    /// sc-9607 flipped their `supported_quants`). Quant stays on candle; a LoRA still defers to torch.
    /// Disjoint from [`CANDLE_QUANT_LORA_MODELS`]; both are consulted by the gate. Subset of
    /// [`CANDLE_ROUTED_MODELS`].
    pub(crate) CANDLE_QUANT_MODELS, IMAGE_MODEL_CAPS, candle_quant
}

derive_model_list! {
    /// The candle image families that advertise inference LoRA/LoKr but NOT on-the-fly quant (derived from
    /// [`IMAGE_MODEL_CAPS`]`.candle_lora`, sc-9495). Currently EMPTY: Krea 2 Turbo was the sole member until
    /// sc-9983 moved it to [`CANDLE_QUANT_LORA_MODELS`] (sc-9607 gave it Q4/Q8 too). Kept as the vocabulary
    /// for the next candle family that advertises LoRA but not quant. The mirror of [`CANDLE_QUANT_MODELS`];
    /// both plus [`CANDLE_QUANT_LORA_MODELS`] are disjoint and all consulted by the gate. Subset of
    /// [`CANDLE_ROUTED_MODELS`].
    pub(crate) CANDLE_LORA_MODELS, IMAGE_MODEL_CAPS, candle_lora
}

derive_model_list! {
    /// The video models the candle (Windows/CUDA) lane serves for base txt2video (derived from
    /// [`VIDEO_MODEL_CAPS`]`.candle_video_routed`, sc-9495). Mirrors `video_jobs::candle_video_engine_id`.
    /// The 14B I2V + SVD are image→video (see [`CANDLE_VIDEO_I2V_ROUTED_MODELS`]).
    pub(crate) CANDLE_VIDEO_ROUTED_MODELS, VIDEO_MODEL_CAPS, candle_video_routed
}

derive_model_list! {
    /// The candle video models that run image→video ONLY (a source image is required), not txt2video —
    /// Wan2.2 14B I2V + SVD (derived from [`VIDEO_MODEL_CAPS`]`.candle_video_i2v`, sc-9495). Subset of
    /// [`CANDLE_VIDEO_ROUTED_MODELS`].
    pub(crate) CANDLE_VIDEO_I2V_ROUTED_MODELS, VIDEO_MODEL_CAPS, candle_video_i2v
}

derive_model_list! {
    /// The candle video models eligible for the Wan-VACE advanced modes (derived from
    /// [`VIDEO_MODEL_CAPS`]`.candle_video_vace`, sc-9495). These route to the single candle `wan_vace`
    /// engine regardless of the user's wan pick. The SCAIL-2 person-replace backend is a distinct candle
    /// engine, so `scail2_*` is deliberately absent. Subset of [`CANDLE_VIDEO_ROUTED_MODELS`].
    pub(crate) CANDLE_VIDEO_VACE_MODELS, VIDEO_MODEL_CAPS, candle_video_vace
}

derive_model_list! {
    /// Video models the in-process Rust MLX worker generates today (derived from
    /// [`VIDEO_MODEL_CAPS`]`.video_mlx_routed`, sc-9495). Mirrors `MlxVideoAdapter._supported_models`. A
    /// model id absent here is never routed to the mlx worker.
    pub(crate) VIDEO_MLX_ROUTED_MODELS, VIDEO_MODEL_CAPS, video_mlx_routed
}

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
    //! Membership-parity regression guard (sc-8816, strengthened sc-9495): every routed-model /
    //! routed-kernel list is pinned to a snapshot of its pre-collapse contents. The model lists are
    //! now DERIVED from [`IMAGE_MODEL_CAPS`] / [`VIDEO_MODEL_CAPS`] (sc-9495), so the parity test is
    //! the zero-diff proof that the table-driven derivation reproduces the OLD membership EXACTLY
    //! before the standalone lists were removed. Membership is compared as a SET (same elements, same
    //! count, no duplicates) because nothing in routing depends on list order — every consumer either
    //! `.contains()`es or iterates order-independently — so the table can carry a single canonical
    //! row order while each derived list still proves exact membership parity. The documented superset
    //! invariant (quant/lora ⊆ candle-routed; i2v/vace ⊆ candle-video-routed) is asserted over the
    //! tables directly.
    use std::collections::BTreeSet;

    use super::{
        CANDLE_LORA_MODELS, CANDLE_QUANT_LORA_MODELS, CANDLE_QUANT_MODELS, CANDLE_ROUTED_MODELS,
        CANDLE_ROUTED_TRAINING_KERNELS, CANDLE_VIDEO_I2V_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS,
        CANDLE_VIDEO_VACE_MODELS, IMAGE_MODEL_CAPS, MLX_ONLY_TRAINING_KERNELS, MLX_ROUTED_MODELS,
        MLX_ROUTED_TRAINING_KERNELS, VIDEO_MLX_ROUTED_MODELS, VIDEO_MODEL_CAPS,
    };

    /// Assert a table-derived list reproduces its pre-collapse snapshot EXACTLY as a set: same
    /// membership, same length (so no id was dropped, added, or duplicated). Order is intentionally
    /// not compared — see the module doc.
    fn assert_membership_parity(name: &str, derived: &[&str], expected: &[&str]) {
        let derived_set: BTreeSet<&str> = derived.iter().copied().collect();
        let expected_set: BTreeSet<&str> = expected.iter().copied().collect();
        assert_eq!(
            derived_set, expected_set,
            "{name}: table-derived membership must equal the pre-collapse snapshot (sc-9495 zero-diff)"
        );
        assert_eq!(
            derived.len(),
            derived_set.len(),
            "{name}: derived list has duplicate ids"
        );
        assert_eq!(
            derived.len(),
            expected.len(),
            "{name}: derived list length must equal the pre-collapse snapshot length"
        );
    }

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

    // sc-9983: Krea joins Lens as a BOTH-quant-and-LoRA candle family (sc-9607 flipped its
    // `supported_quants` to [Q4, Q8]; it already advertised inference LoRA via sc-7836).
    const EXPECTED_CANDLE_QUANT_LORA_MODELS: &[&str] = &["lens", "lens_turbo", "krea_2_turbo"];

    // sc-9983: ideogram/boogu join SD3.5 as quant-only candle families (sc-9607 flipped their
    // `supported_quants` to [Q4, Q8]; no inference LoRA on candle).
    const EXPECTED_CANDLE_QUANT_MODELS: &[&str] = &[
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "ideogram_4",
        "ideogram_4_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
    ];

    // sc-9983: Krea moved to CANDLE_QUANT_LORA_MODELS (BOTH), so the LoRA-only list is now empty.
    const EXPECTED_CANDLE_LORA_MODELS: &[&str] = &[];

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
        // The nine model lists are table-DERIVED (sc-9495): assert each reproduces the pre-collapse
        // snapshot EXACTLY as a set (zero membership diff) — this is the guardrail that proves the
        // collapse changed no routing decision.
        assert_membership_parity(
            "MLX_ROUTED_MODELS",
            MLX_ROUTED_MODELS,
            EXPECTED_MLX_ROUTED_MODELS,
        );
        assert_membership_parity(
            "CANDLE_ROUTED_MODELS",
            CANDLE_ROUTED_MODELS,
            EXPECTED_CANDLE_ROUTED_MODELS,
        );
        assert_membership_parity(
            "CANDLE_QUANT_LORA_MODELS",
            CANDLE_QUANT_LORA_MODELS,
            EXPECTED_CANDLE_QUANT_LORA_MODELS,
        );
        assert_membership_parity(
            "CANDLE_QUANT_MODELS",
            CANDLE_QUANT_MODELS,
            EXPECTED_CANDLE_QUANT_MODELS,
        );
        assert_membership_parity(
            "CANDLE_LORA_MODELS",
            CANDLE_LORA_MODELS,
            EXPECTED_CANDLE_LORA_MODELS,
        );
        assert_membership_parity(
            "CANDLE_VIDEO_ROUTED_MODELS",
            CANDLE_VIDEO_ROUTED_MODELS,
            EXPECTED_CANDLE_VIDEO_ROUTED_MODELS,
        );
        assert_membership_parity(
            "CANDLE_VIDEO_I2V_ROUTED_MODELS",
            CANDLE_VIDEO_I2V_ROUTED_MODELS,
            EXPECTED_CANDLE_VIDEO_I2V_ROUTED_MODELS,
        );
        assert_membership_parity(
            "CANDLE_VIDEO_VACE_MODELS",
            CANDLE_VIDEO_VACE_MODELS,
            EXPECTED_CANDLE_VIDEO_VACE_MODELS,
        );
        assert_membership_parity(
            "VIDEO_MLX_ROUTED_MODELS",
            VIDEO_MLX_ROUTED_MODELS,
            EXPECTED_VIDEO_MLX_ROUTED_MODELS,
        );
        // The training-kernel lists are keyed by kernel id (a separate namespace) and were NOT
        // collapsed into the per-model tables (sc-9495); they stay hand-written, still order-pinned.
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
        // Derived-list view of the invariant (unchanged from sc-8816): every quant/lora id is also a
        // candle-routed id.
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

    #[test]
    fn capability_table_encodes_superset_invariant() {
        // Table-level view of the superset invariant (sc-9495): assert it row-by-row over the source of
        // truth, not just the derived lists — so a future row that sets a quant/lora column without
        // candle_routed (or an i2v/vace column without candle_video_routed) is caught here in addition
        // to the `const fn new` compile-time `assert!`.
        for caps in IMAGE_MODEL_CAPS {
            if caps.candle_quant || caps.candle_lora || caps.candle_quant_lora {
                assert!(
                    caps.candle_routed,
                    "{}: quant/lora capability implies candle_routed (superset invariant)",
                    caps.id
                );
            }
            // The three candle-adapter columns are mutually exclusive by construction (quant-only,
            // lora-only, both) — the gate consults them as three disjoint lists.
            let adapter_flags = [caps.candle_quant, caps.candle_lora, caps.candle_quant_lora];
            assert!(
                adapter_flags.iter().filter(|flag| **flag).count() <= 1,
                "{}: candle_quant / candle_lora / candle_quant_lora are mutually exclusive",
                caps.id
            );
        }
        for caps in VIDEO_MODEL_CAPS {
            if caps.candle_video_i2v || caps.candle_video_vace {
                assert!(
                    caps.candle_video_routed,
                    "{}: candle video i2v/vace capability implies candle_video_routed (superset invariant)",
                    caps.id
                );
            }
        }
    }

    #[test]
    fn capability_tables_have_no_duplicate_ids() {
        let image_ids: BTreeSet<&str> = IMAGE_MODEL_CAPS.iter().map(|caps| caps.id).collect();
        assert_eq!(
            image_ids.len(),
            IMAGE_MODEL_CAPS.len(),
            "IMAGE_MODEL_CAPS has duplicate model ids"
        );
        let video_ids: BTreeSet<&str> = VIDEO_MODEL_CAPS.iter().map(|caps| caps.id).collect();
        assert_eq!(
            video_ids.len(),
            VIDEO_MODEL_CAPS.len(),
            "VIDEO_MODEL_CAPS has duplicate model ids"
        );
    }
}
