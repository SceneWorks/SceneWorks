//! Model/kernel routing catalog: the per-backend routed-model and training-kernel lists,
//! the Mac support/capability probes, and their supporting types. Moved out of
//! `jobs_store.rs` (sc-8816) with no behavior change. List membership is pinned by the
//! snapshot tests at the bottom of this file.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::jobs_store::routing::gaps::{classify_image_gap, classify_video_gap, UnsupportedReason};
use crate::jobs_store::routing::mlx::{image_request_mlx_eligible, video_mode_is_mlx_eligible};
use crate::jobs_store::routing::{
    has_nonempty_array, has_nonempty_nested_array, has_nonempty_string, has_nonempty_string_array,
};

/// The user-facing affordance prefix the Mac UI shows in place of a control with no native Mac lane
/// (sc-3486). Centralised so the API, the web client, and the gap docs read identically.
pub const MAC_NOT_AVAILABLE_LABEL: &str = "Not available on Mac (MLX only)";

/// UI-facing per-model macOS support (sc-3486), derived from the same `*_mlx_eligible` routing
/// predicates as the [`mac_rust_supported`] job oracle — one source of truth, so what the UI
/// hides can never drift from what routing refuses. `supported` = at least one generation config
/// for this model routes to the in-process Rust/MLX flow on macOS, so the model stays in the
/// picker; `false` = a model with no native Mac lane that the UI hides/disables once gating is active (its
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
/// Capability-specific *actions* are gated by [`mac_capabilities`] at the job-type level, not by hiding
/// the model from a picker. Same source of truth as [`mac_rust_supported`].
///
/// `family` is the model's catalog-declared architecture family (`None` for a builtin whose routing
/// is purely id-keyed). It cannot by itself select an imported loader: this core-level probe is
/// therefore authoritative for builtins only and leaves novel imported ids unsupported. The API's
/// catalog projection has the full manifest entry and, after applying these builtin defaults,
/// overwrites an imported row's `macSupport` from the exact family + `importSourceShape` provider
/// facts. This prevents a family sibling with a different filesystem shape from inheriting a route.
pub fn model_mac_support(
    model_id: &str,
    model_type: &str,
    family: Option<&str>,
) -> ModelMacSupport {
    match model_type {
        "image" => image_model_mac_support(model_id, family),
        "video" => video_model_mac_support(model_id),
        _ => ModelMacSupport {
            supported: true,
            reason: None,
            features: ModelMacFeatures::default(),
        },
    }
}

/// Builtin image support probe. Imported rows require their full source-shaped manifest entry, so
/// the API replaces this provisional verdict with exact provider-derived support during catalog
/// projection; the family argument is intentionally insufficient here.
pub(crate) fn image_model_mac_support(model: &str, _family: Option<&str>) -> ModelMacSupport {
    if !MLX_ROUTED_MODELS.contains(&model) {
        // A family token is not enough to choose an imported loader: source shape is mandatory.
        // Catalog projection applies exact provider facts once it has the full manifest entry.
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
/// per-model — native on the supported LTX/Wan paths, queued when unsupported — rather than via a
/// coarse global flag.
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
    // sc-4206) — kept as the gating vocabulary for the next unsupported native surface that
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
/// Training studio can disable a base model whose kernel has no native Mac trainer.
/// `lokr_on_wan_supported=false` mirrors the LoKr-on-Wan routing caveat.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacTrainingSupport {
    pub supported_kernels: Vec<String>,
    pub lokr_on_wan_supported: bool,
}

/// What the Mac UI needs to gate every non-model native gap plus the master switch
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
    // (e.g. `imageUpscaleSeedvr2`, which follows native MLX or Candle availability) rather than the
    // gating-rollout flag.
    let is_mac = matches!(platform, "macos" | "darwin");
    // SeedVR2 has a backend on Mac (native MLX) and on Windows + Linux (the candle CUDA/NVIDIA port:
    // Windows sc-5928, Linux sc-5160 — candle is CPU+CUDA cross-platform so Linux rides the Windows
    // port). Drives the platform-intrinsic `imageUpscaleSeedvr2` flag.
    let seedvr2_supported = is_mac || matches!(platform, "windows" | "linux");
    // SAM3 box smart-select follows the same native-backend footprint: MLX on Mac and Candle on
    // Windows/Linux. Other platforms have no linked native provider.
    let image_segment_supported = is_mac || matches!(platform, "windows" | "linux");
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
        // spike): it is a 617M-param legacy GigaGAN with no viable Rust path and only a marginal,
        // ~35-50x-slower quality difference vs the already-ported Real-ESRGAN x4. As of sc-5499 it is
        // also dropped as an OFFERED engine off-Mac — there is no native (MLX/candle) path and the
        // historical Python backend that served it on Windows/Linux is retired, so exposing it would
        // point users at a nonexistent path. `supported: false` on every
        // platform (platform-intrinsic, like `imageUpscaleSeedvr2`), so the UI hides the engine
        // everywhere. Must agree with the AuraSR arm of `mac_rust_supported` (UI-hidden == routing
        // refuses): the native MLX/candle workers refuse it, so an explicitly submitted AuraSR job
        // remains queued.
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
        // the routing oracle (mlx OR candle claims SeedVR2; every other descriptor refuses it).
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
        // Smart-select segmentation (epic 6087, sc-6105; sc-18480): SAM3 box-prompt
        // segmentation runs in-process on the native MLX worker on Mac and the Candle worker on
        // Windows/Linux. This platform-intrinsic flag must agree with worker advertisement and
        // dispatch so the UI never hides an executable lane or offers an absent one.
        "imageSegment".to_owned(),
        MacFeatureSupport {
            supported: image_segment_supported,
            reason: if image_segment_supported {
                None
            } else {
                Some(UnsupportedReason::new(
                    None,
                    "image_segment (SAM3 smart-select)",
                    "smart-select segmentation requires the native MLX backend on macOS or the Candle backend on Windows/Linux.",
                    Some("sc-18480"),
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
        // SeedVR2 gives SceneWorks its first video upscaler: native MLX on Mac (epic 4811 / sc-4816)
        // and native Candle/CUDA on Windows/Linux (sc-5928 / sc-5160). This platform feature gates
        // the Video Studio "Upscale" action and must agree with routing.
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
/// imply `candle_routed` — a model can only advertise a quant tier/select or inference LoRA on the
/// candle lane if it is candle-routed at all. Encoded structurally by [`ModelCaps::new`] (a `debug_assert`
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
    /// Candle accepts Q4/Q8 generation requests (either a packed-tier select or load-time quant) but
    /// NOT inference LoRA (was `CANDLE_QUANT_MODELS`).
    pub(crate) candle_quant: bool,
    /// Candle advertises inference LoRA/LoKr but NOT on-the-fly quant (was `CANDLE_LORA_MODELS`).
    pub(crate) candle_lora: bool,
    /// Candle accepts BOTH Q4/Q8 generation requests AND inference LoRA
    /// (was `CANDLE_QUANT_LORA_MODELS`).
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
    // Mage-Flow generation + instruction-edit (sc-14053): all six registry descriptors are linked by
    // runtime-cuda and advertise Q4/Q8 over the same complete dense snapshots (load-time DiT fold;
    // BF16 stays dense). User LoRA/LoKr now applies on both dense and packed tiers.
    ModelCaps::new("mage_flow_base", true, true, false, false, true),
    ModelCaps::new("mage_flow", true, true, false, false, true),
    ModelCaps::new("mage_flow_turbo", true, true, false, false, true),
    ModelCaps::new("mage_flow_edit_base", true, true, false, false, true),
    ModelCaps::new("mage_flow_edit", true, true, false, false, true),
    ModelCaps::new("mage_flow_edit_turbo", true, true, false, false, true),
    // sc-3022 Z-Image / sc-3023 FLUX.1 / sc-3024 Qwen / sc-3025 FLUX.2 / sc-3026 SDXL — the founding
    // MLX-routed families (grows one family story at a time as each lands real generation in
    // `sceneworks-worker::image_jobs`). CANDLE: SDXL sc-3678, the four families sc-5096.
    //
    // Z-Image's candle descriptor deliberately keeps `supported_quants: []`: it does not quantize a
    // dense checkpoint on the fly. That is NOT the capability this routing bit represents for its
    // turnkey. `advanced.mlxQuantize` selects an already-packed q4/q8/bf16 directory, and
    // candle-gen-z-image packed-detects those components from their config. The manifests advertise
    // `standardTierLayout` and measured candle Q4/BF16 VRAM. Keeping `candle_quant = false` therefore
    // rejected a valid Q4 tier select as `candle_unsupported` before the worker could load it.
    ModelCaps::new("z_image_turbo", true, true, false, false, true),
    // Base (non-distilled, full-CFG) Z-Image (epic 8236, sc-8379 control + sc-8679 txt2img). MLX-routed
    // on macOS AND candle-routed off-Mac. The worker registers a real `z_image` MLX engine (MODEL_TABLE:
    // shift-6.0 / ~50-step / real CFG, `mlx_z_image` adapter) plus base strict-control on Mac
    // (`ImageRoute::ZImageBaseControl`, `WIRED_MLX_POSE_FAMILIES`), and the manifest carries a full `mlx`
    // block — so a Mac generates base Z-Image in-process just like Turbo. The prior `mlx_routed: false`
    // here was a wiring gap left by sc-8320/sc-8251 (the base MLX engine + control landed, but this row
    // and `image_request_mlx_eligible` were never updated), so `model_mac_support` hid the model behind
    // "Not available on Mac (MLX only)" even though the Mac worker fully supports it.
    ModelCaps::new("z_image", true, true, false, false, true),
    // `z_image_edit` (epic 3529 / sc-3923): MLX-only edit id on Turbo weights.
    ModelCaps::new("z_image_edit", true, false, false, false, false),
    ModelCaps::new("flux_schnell", true, true, false, true, false),
    ModelCaps::new("flux_dev", true, true, false, true, false),
    // Base `qwen_image` candle txt2img is a turnkey packed-quant family (sc-8669 wired the q4/q8/bf16
    // subdirs into `STANDARD_TIER_MODELS`; sc-10969 measured the tiers), so a tier-select `mlxQuantize`
    // stays on candle — `candle_quant` is set (sc-11020, the routing half previously missed by sc-9983,
    // which flipped krea/ideogram/boogu but not qwen). User LoRA/LoKr applies on the packed tiers.
    ModelCaps::new("qwen_image", true, true, false, false, true),
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
    // (sc-7459, a weights swap). The klein/dev `edit_image` shapes do NOT reach this gate — they are
    // branched out to the bespoke candle `Flux2Edit` lane above (`flux2_edit_candle_eligible`), so these
    // caps describe the txt2img surface only.
    //
    // sc-10222 (epic 9083): `candle_quant = true` for the two ids that actually ship the standard
    // q4/q8/bf16 turnkey. Both are worker `STANDARD_TIER_MODELS` members whose manifests carry a
    // MEASURED `candle.vramGbByTier` (klein 46.2/50.6/75.5, dev 44.0/70.7/128.0) — the worker has served
    // those tiers on the candle txt2img lane since sc-9092, but this row still said `false`, so
    // `image_request_candle_eligible`'s `!supports_quant && candle_request_wants_quant` arm bounced every
    // explicit `advanced.mlxQuantize > 0` tier-select off the candle lane into the retired torch
    // fallback — and FLUX.2-klein has NO torch path at all. Cruelly, dev's own fit-gate tells a 48 GB
    // user to pick q4 (44.0) instead of the q8 default (70.7): the only tier that fits was the one tier
    // that could not be routed. This is the identical "engine wired, router half missed" skew sc-9983
    // (krea/ideogram/boogu) and sc-11020 (qwen_image) each closed; the diagnostic tell is exactly the one
    // recorded there — a family in `STANDARD_TIER_MODELS` with a `candle.vramGbByTier` block that is not
    // in a quant-capable routing set. sc-18477 adds user LoRA/LoKr on every published tier.
    //
    // The load `Quant` stays `None` on klein regardless (its `mlx.denseTextEncoderTier` declaration keeps its bf16 Qwen3 text
    // encoder full-precision); `mlxQuantize` here is a turnkey tier-SELECT — which pre-quantized subdir
    // `standard_tier_subdir` descends into — not an on-the-fly quantize.
    ModelCaps::new("flux2_klein_9b", true, true, false, false, true),
    ModelCaps::new("flux2_klein_9b_kv", true, true, false, false, true),
    // `_true_v2` stays `candle_quant = false`: it is the wikeeyang community fine-tune, installed by
    // convert-at-install from a single bf16 file into a FLAT `modelPath` dir. It ships no q4/q8/bf16
    // tier matrix (one `downloads[]` entry, `mlx.quantize: 8` on-the-fly), so there is no tier for a
    // pick to select — admitting one would only hand the dense converted tree to the legacy CPU-stage →
    // quantize-onto-GPU path on a shape nothing has validated.
    ModelCaps::new("flux2_klein_9b_true_v2", true, true, false, true, false),
    // FLUX.2-dev (epic 5914 MLX / epic 6564 sc-7458 candle) — the guidance-distilled 32B flagship.
    // A SEPARATE candle engine from klein (Mistral3 TE + 48/48/15360 DiT). Same sc-10222 tier-select
    // flip as klein above; its `SceneWorks/flux2-dev-mlx` turnkey is where the epic's headline
    // "packed Q4 load kills the ~105 GB dense CPU-staging peak" claim actually lands.
    ModelCaps::new("flux2_dev", true, true, false, false, true),
    // SDXL family (sc-10767, epic 9083 full-catalog parity): the candle lane serves the packed q4/q8
    // MLX tiers end-to-end — packed UNet (sc-9416), packed dual-CLIP (sc-9527), and LoRA/LoKr fold on a
    // packed tier (sc-9528) — and `candle-gen-sdxl` now advertises `supported_quants: [Q4, Q8]`. So
    // BOTH a quant tier-select AND an inference LoRA stay on candle → `candle_quant_lora`. Previously
    // `false, false, false` bounced quant requests (the sc-10726 q8 default) AND LoRA requests off the
    // candle lane into the retired torch fallback. bf16 still resolves to Quant::None (dense), verbatim.
    ModelCaps::new("sdxl", true, true, false, false, true),
    ModelCaps::new("realvisxl", true, true, false, false, true),
    // Illustrious-XL v1.0 / v2.0 (epic 10609): vanilla-SDXL anime finetunes on the shared `sdxl`
    // engine. Same routing surface as `realvisxl` — MLX + candle txt2img + packed q4/q8 (sc-10767).
    ModelCaps::new("illustrious_xl_v1", true, true, false, false, true),
    ModelCaps::new("illustrious_xl_v2", true, true, false, false, true),
    // RealVisXL Lightning (MLX sc-6075 / candle sc-7176): standalone few-step distilled SDXL checkpoint
    // on the shared `sdxl` engine, few-step `lightning` accel sampler. **txt2img only** on both backends —
    // edit / reference / mask / pose shapes are refused and remain queued (the accel sampler is
    // conditioning-incompatible).
    // sc-10812 (epic 9083): shares `candle-gen-sdxl`, which advertises `supported_quants: [Q4, Q8]` +
    // inference LoRA-on-packed after sc-10767 (packed UNet sc-9416 / dual-CLIP sc-9527 / adapter fold
    // sc-9528). Its `SceneWorks/realvisxl-lightning-mlx` turnkey ships the standard q4/q8/bf16 tiers
    // (standard_tier_subdir), so a quant tier-select AND a LoRA both stay on the candle lane for the
    // plain few-step txt2img shape → `candle_quant_lora`. bf16 still resolves to Quant::None (dense).
    ModelCaps::new("realvisxl_lightning", true, true, false, false, true),
    // InstantID on RealVisXL (sc-3345): identity-only id — single-identity + the 11-view angle set use
    // native MLX on Mac and the bespoke `instantid_candle_eligible` lane off-Mac. It is intentionally
    // NOT a candle-routed plain-txt2img id.
    ModelCaps::new("instantid_realvisxl", true, false, false, false, false),
    // PuLID-FLUX on FLUX.1-dev (sc-3344): `character_image` with a reference face runs through native
    // MLX or the bespoke `pulid_flux_candle_eligible` lane, not the plain txt2img gate.
    ModelCaps::new("pulid_flux_dev", true, false, false, false, false),
    // Chroma (epic 3531 / sc-3843 MLX; epic 3692 / sc-5576 candle). Pure txt2img on candle.
    ModelCaps::new("chroma1_hd", true, true, false, true, false),
    ModelCaps::new("chroma1_base", true, true, false, true, false),
    ModelCaps::new("chroma1_flash", true, true, false, true, false),
    // SenseNova-U1 (epic 3180 / sc-3900 MLX; sc-5576 candle). Pure txt2img on candle.
    //
    // sc-14249 (epic 9083): `candle_quant = true` across the whole family. `candle-gen-sensenova`
    // used to mmap its backbone at a hardcoded F32 and hard-reject `spec.quantize`, so the candle
    // lane could read only the dense `bf16/` tier — at DOUBLE its on-disk size (a measured 70.5 GB
    // peak on sm_120 for a 32.7 GiB checkpoint). It now packed-detects each of the 588 backbone
    // projections off the `.scales` sibling in the weights, so the turnkey's `q4/` and `q8/` tiers
    // load natively and an `advanced.mlxQuantize` is a turnkey tier-SELECT. Without this flip
    // `image_request_candle_eligible` would keep bouncing every tier pick off the candle lane, and
    // the worker's `standard_tier_subdir` descent could never reach the cheap tiers.
    //
    // NOT `candle_quant_lora`: the family advertises no candle inference LoRA (the fast ids' 8-step
    // distill LoRA is merged internally by the loader, never user-supplied).
    ModelCaps::new("sensenova_u1_8b", true, true, true, false, false),
    // Infographic-V2 (epic 9959): coexisting checkpoint refresh of the SAME NEO-unify engine as the
    // base id — routes identically (MLX full surface; candle txt2img + the sc-14249 tier matrix).
    ModelCaps::new(
        "sensenova_u1_8b_infographic_v2",
        true,
        true,
        true,
        false,
        false,
    ),
    // Infographic-V3 (epic 13095): tensor-identical checkpoint refresh of the SAME NEO-unify engine
    // (config + 1,116 tensor keys byte-identical to V2, verified) — routes identically to the base/V2
    // ids. Coexists with V2 (new id; V2 stays loadable for pinned recipes).
    ModelCaps::new(
        "sensenova_u1_8b_infographic_v3",
        true,
        true,
        true,
        false,
        false,
    ),
    ModelCaps::new("sensenova_u1_8b_fast", true, true, true, false, false),
    // Infographic-V2 8-step distilled variant (epic 9959): same fast engine, routes like the base fast.
    ModelCaps::new(
        "sensenova_u1_8b_infographic_v2_fast",
        true,
        true,
        true,
        false,
        false,
    ),
    // Infographic-V3 8-step distilled variant (epic 13095): same fast engine, routes like the base fast.
    ModelCaps::new(
        "sensenova_u1_8b_infographic_v3_fast",
        true,
        true,
        true,
        false,
        false,
    ),
    // Kolors (epic 3090): full surface on the Rust `kolors` engine (SDXL-family U-Net + ChatGLM3);
    // candle serves txt2img + bespoke IP/pose lanes (sc-5488/sc-5489). sc-10819 (epic 9083): the candle
    // `candle-gen-kolors` lane now serves the packed q4/q8 `SceneWorks/kolors-mlx` tiers end-to-end —
    // packed ChatGLM3 (the four GLM projections) + the vendored packed-detecting SDXL UNet, VAE dense —
    // and advertises `supported_quants: [Q4, Q8]`. So a quant tier-select stays on candle → `candle_quant`.
    // Kolors now applies LoRA/LoKr through the vendored adaptable SDXL UNet on dense and packed tiers;
    // bf16 still resolves to Quant::None (dense), verbatim.
    ModelCaps::new("kolors", true, true, false, false, true),
    // Microsoft Lens / Lens-Turbo (epic 3164 / sc-5105 MLX; sc-5126 candle): pure T2I family. It
    // advertises on-the-fly quant AND LoRA/LoKr, so `candle_quant_lora` is set. Lens was the LAST
    // whole-model torch-only image family — once it routed, the per-model
    // torch-only image epic seam matched nothing and was retired (sc-8951).
    ModelCaps::new("lens", true, true, false, false, true),
    ModelCaps::new("lens_turbo", true, true, false, false, true),
    // Bernini still-image companion (epic 4699 / sc-5424 MLX; sc-10996 candle): `engine_id:"bernini"`
    // planner+renderer with `frames:1`. Both backends wired — `mlx-gen-bernini` (macOS) +
    // `candle-gen-bernini` (Windows/CUDA, sc-10996 epic 6562: the full planner+renderer `bernini`
    // generator, `gen_core::load("bernini")` with `frames:1`), so `candle_routed = true`. Both still
    // tasks route to candle: t2i via the generic `image_request_candle_eligible` gate, i2i via the
    // `bernini_image` `edit_image` branch in `image_job_is_candle_eligible` (a `sourceAssetId` edit, the
    // bespoke `generate_candle_bernini_image_stream` lane — like the MLX `bernini_image_mlx_eligible`).
    // `candle_quant = true`: the descriptor advertises Q4/Q8 and the off-Mac worker resolves the
    // published `SceneWorks/bernini` bf16/q8/q4 tier subdirectories (sc-11003) in both its still and
    // video lanes. User adapters route to the renderer's high/low experts on every tier.
    ModelCaps::new("bernini_image", true, true, false, false, true),
    // Ideogram 4 + Turbo (epic 4725 MLX; sc-6597 candle): 9.3B flow DiT + Qwen3-VL-8B TE. T2I + edit on
    // MLX (sc-6303); candle serves txt2img + the in-lane edit path (sc-6598) via the generic stream.
    // Candle advertises Q4/Q8 (sc-9607 flipped `supported_quants: [Q4, Q8]`, dropping the loader's
    // `spec.quantize` reject — a no-op on the already-packed q4/q8 turnkey), so `candle_quant` is set
    // (sc-9983 — the routing half of sc-9607, previously missed): a tier-select `mlxQuantize` stays on
    // candle. User LoRA/LoKr stacks after the bundled TurboTime adapter when Turbo is selected.
    ModelCaps::new("ideogram_4", true, true, false, false, true),
    ModelCaps::new("ideogram_4_turbo", true, true, false, false, true),
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
    // Krea 2 Raw (epic 9992) — the undistilled 12B DiT exposed as a full-CFG generation model (52 steps /
    // guidance 3.5 + negative prompt), alongside the distilled Turbo (the Boogu base/turbo precedent).
    // Both backends wired: `mlx-gen-krea` (PR #656) + `candle-gen-krea` (sc-9994, candle-gen #350 —
    // `render_base`, two DiT forwards/step, the reference `sampling.py:129` CFG combine). Candle
    // advertises inference LoRA/LoKr (the shared Krea merge, sc-7836) AND on-the-fly Q4/Q8 (a no-op on
    // the already-packed q4/q8 turnkey, sc-9607), so `candle_quant_lora` is set — mirroring `krea_2_turbo`
    // exactly (Raw + Turbo share the arch / loader; only the DiT weights differ). `krea_2_raw` is ALSO the
    // LoRA-training base id (Path 1 unify); training routes via the trainer registry, independent of this
    // image-caps row.
    ModelCaps::new("krea_2_raw", true, true, false, false, true),
    // Stable Diffusion 3.5 Large / Large Turbo / Medium (epic 7841 / sc-7871 MLX; sc-7880 candle):
    // txt2img plus singular-reference latent-init img2img (epic 8588 A4 / sc-10189). Candle advertises
    // Q4/Q8 (sc-7879) and applies LoRA/LoKr on every tier.
    ModelCaps::new("sd3_5_large", true, true, false, false, true),
    ModelCaps::new("sd3_5_large_turbo", true, true, false, false, true),
    ModelCaps::new("sd3_5_medium", true, true, false, false, true),
    // SANA 1600M (epic 8485 / sc-8489 MLX; sc-11780/sc-18475 candle): NVIDIA's 1.6B Linear-DiT
    // true-CFG txt2img plus singular-reference non-edit img2img.
    // Both backends wired — `mlx-gen-sana` (macOS, MLX-packed q4/q8/bf16 turnkey) + `candle-gen-sana`
    // (Windows/CUDA + Linux, candle-gen #495 — loads the whole `Efficient-Large-Model/
    // Sana_1600M_1024px_diffusers` HF snapshot dense), so `candle_routed = true`. NOT
    // `candle_quant` / `candle_lora`: the candle base path advertises neither (dense bf16, no adapter
    // fold) — an `mlxQuantize` or LoRA request is refused off-Mac and remains queued.
    ModelCaps::new("sana_1600m", true, true, false, false, false),
    // SANA-Sprint 1.6B (epic 8485 / sc-8490 MLX; sc-11781 candle): NVIDIA's few-step CFG-FREE distill of
    // SANA — the SAME 1.6B Linear-DiT trunk with a guidance embedder, sampled by the SCM/TrigFlow
    // continuous-time consistency loop in 1–4 steps. Both backends wired — `mlx-gen-sana` (macOS,
    // MLX-packed q4/q8/bf16 turnkey) + `candle-gen-sana`'s Sprint pipeline (Windows/CUDA + Linux,
    // candle-gen #498 — loads the whole `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` HF
    // snapshot dense), so `candle_routed = true`, including singular-reference non-edit img2img. NOT
    // `candle_quant` / `candle_lora`: the
    // candle Sprint path advertises neither (the adapter rejects quant / LoRA / control) — an `mlxQuantize`
    // or LoRA request is refused off-Mac and remains queued.
    ModelCaps::new("sana_sprint_1600m", true, true, false, false, false),
    // Anima base / aesthetic / turbo (epic 10512): anime txt2img on BOTH backends — native-MLX (macOS,
    // install-time Q4/Q8 quant) and the candle off-Mac lane (sc-10676), which sc-10625 GPU-validated on
    // real CUDA (candle-gen #380). `candle_routed = true`; `candle_lora = true` because the candle engine
    // dense-folds a LoRA/LoKr onto the split_files/ DiT (descriptor `supports_lora`/`supports_lokr`,
    // validated in the anima GPU smoke). `candle_quant = false`: there is NO packed Q4/Q8 tier off-Mac —
    // the `anima_quant` converter is macOS-only and the NC license bars publishing one, and the candle
    // loader only CONSUMES an MLX-packed tier (it hard-rejects a quant request against the dense DiT). So
    // a deliberate `mlxQuantize` request stays off candle (defers rather than silently running dense —
    // the sc-10515 anti-lie posture); the worker also force-dense-loads Anima regardless of the manifest
    // default. `candle_quant_lora = false`: quant+LoRA together is unsupported on both lanes (sc-10578 —
    // folding an adapter into u32-packed codes is a separate additive-branch job). candle_lora ⟹
    // candle_routed keeps the sc-9495 superset invariant satisfied.
    ModelCaps::new("anima_base", true, true, false, true, false),
    ModelCaps::new("anima_aesthetic", true, true, false, true, false),
    ModelCaps::new("anima_turbo", true, true, false, true, false),
];

/// The one-row-per-model VIDEO routing table (sc-9495) — the single source the video list constants
/// below are derived from.
///
/// Legend for the [`VideoModelCaps::new`] positional args:
/// `new(id, video_mlx_routed, candle_video_routed, candle_video_i2v, candle_video_vace)`.
pub(crate) const VIDEO_MODEL_CAPS: &[VideoModelCaps] = &[
    // LTX-2.3 base (sc-18478): both native backends serve the provider's current mode surface.
    VideoModelCaps::new("ltx_2_3", true, true, false, false),
    // 10Eros remains MLX-only (sc-18902). Exact-head Candle/CUDA acceptance run 31766800005
    // rendered the dense, undistilled checkpoint through the single-pass distilled engine at the
    // shipped 8-step defaults and produced unresolved noise with no prompt subject. The required
    // cond_safe distill LoRA is a two-pass MLX recipe and only 768/3,320 keys match Candle's adapter
    // surface, so Candle must not claim this model until it has a complete validated recipe.
    VideoModelCaps::new("ltx_2_3_eros", true, false, false, false),
    // Wan2.2 TI2V-5B (sc-18478): T2V plus native I2V/FLF and user adapters on both backends. Its
    // legacy VACE membership remains for the established extend/bridge fallback.
    VideoModelCaps::new("wan_2_2", true, true, false, true),
    // Wan2.2 14B MoE (sc-5175): T2V-14B is text-only; I2V-14B is image→video ONLY (candle_video_i2v).
    // Both are VACE-capable on candle.
    VideoModelCaps::new("wan_2_2_t2v_14b", true, true, false, true),
    VideoModelCaps::new("wan_2_2_i2v_14b", true, true, true, true),
    // Wan2.2 VACE-Fun A14B (sc-18478): dedicated dual-expert replace-person providers on both native
    // backends. Like SCAIL-2 below, this is deliberately absent from the base Candle and generic
    // single-expert VACE columns: its dedicated predicate admits PersonReplace only.
    VideoModelCaps::new("wan_2_2_vace_fun_14b", true, false, false, false),
    // SVD (`svd` → `svd_xt`, sc-3523 MLX; sc-5493 candle): image→video ONLY. Not a VACE model.
    VideoModelCaps::new("svd", true, true, true, false),
    // Bernini (epic 4699 / sc-4707 MLX; sc-10997 candle): Qwen2.5-VL planner + Wan2.2-T2V-A14B
    // renderer. Both backends wired — `mlx-gen-bernini` (macOS) + `candle-gen-bernini` (Windows/CUDA,
    // the full planner+renderer `bernini` generator, `gen_core::load("bernini")`), so
    // `candle_video_routed = true`. Serves t2v + the editing/reference/multi-source modes (v2v / r2v /
    // rv2v / mv2v / ads2v) on both lanes; the candle worker routes those via the dedicated
    // `bernini_video_candle_eligible` gate + the `CandleVideoRoute::Bernini` dispatch (NOT the generic
    // wan/ltx txt2video arm — Bernini is a distinct engine). Not an i2v/VACE model, so those columns
    // stay false. Off-Mac packed-tier select is deferred until the `SceneWorks/bernini` tier
    // layout lands (sc-11003) — the candle lane loads the converted snapshot dense today.
    VideoModelCaps::new("bernini", true, true, false, false),
    // SCAIL-2 (epic 5439 / sc-5448): MLX end-to-end character animation; the candle SCAIL-2 engine
    // (sc-6837) is a DISTINCT engine gated by its own predicates, NOT Wan-VACE membership — so its
    // candle-video columns are all false (it is deliberately absent from `CANDLE_VIDEO_*`).
    VideoModelCaps::new("scail2_14b", true, false, false, false),
    // Mochi 1 (epic 1788 / sc-11991): 10B AsymmDiT text-to-video on BOTH backends — `mlx-gen-mochi`
    // (macOS) and `candle-gen-mochi` (Windows/CUDA + Linux), which ingests the SAME mlx-affine tiers
    // via the A6 `.scales`-detect seam. Both register the SAME engine id (`mochi_1`), unlike LTX's
    // `ltx_2_3` + `ltx_2_3_distilled` split.
    //
    // `candle_video_routed = true` is load-bearing: the MLX descriptor is `mac_only: true` but the
    // CANDLE descriptor is `mac_only: false`, so Mochi must NOT be hard mac-gated app-side — the
    // off-Mac lane is real and CUDA-validated on Blackwell (sc-11990). Per-backend gating is exactly
    // what this table's two routed columns express, so no new mechanism is needed.
    //
    // t2v ONLY (`conditioning: []` on both descriptors): NOT an i2v model and NOT a VACE model, so
    // those columns stay false — `video_request_candle_eligible`'s non-i2v arm then requires an
    // explicit `text_to_video` mode and rejects a stray source image, and `video_mode_is_mlx_eligible`
    // carries the matching t2v-only arm. Absent from `CANDLE_VIDEO_LORA_MODELS` because both
    // descriptors set `supports_lora`/`supports_lokr` = false, so a LoRA-carrying job is refused.
    VideoModelCaps::new("mochi_1", true, true, false, false),
    // Krea Realtime 14B (epic 8431 / sc-8444): the autoregressive Self-Forcing Wan-2.1-14B video
    // engine. `video_mlx_routed = true` — S10 (sc-8443) wired the real MLX route
    // (`video_jobs/krea_realtime.rs` + the `resolve_video_route` arm), so without this row
    // `video_job_is_mlx_eligible` refuses the job AND `video_model_mac_support` reports the
    // `classify_video_gap` "no MLX engine" reason, which is simply untrue for this model.
    //
    // Every CANDLE column is false, and that is load-bearing rather than a default: there is no
    // `candle-gen-krea-realtime` at all (parity is a deliberately separate follow-up epic), the MLX
    // descriptor is `mac_only: true`, and `run_video_generate_job` already fails a non-mac krea job
    // loudly rather than routing it elsewhere. So it is neither candle-routed nor a candle i2v/VACE
    // model. The positional row has the same all-false Candle shape as `scail2_14b`, but NOT the same
    // meaning: SCAIL-2 is unioned through its distinct-engine predicate below; Krea has no such lane.
    VideoModelCaps::new("krea_realtime_14b", true, false, false, false),
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
    /// worker, so an absent id remains unclaimed unless family-based imported routing admits it.
    pub(crate) MLX_ROUTED_MODELS, IMAGE_MODEL_CAPS, mlx_routed
}

/// Historical architecture-family surface retained only for test oracles that compare the former
/// family gate with the import compatibility allow-list. Production imported routing does not read
/// this list: it selects an exact backend + family + `importSourceShape` + operation row from the
/// checked-in provider facts below. Builtins remain id-routed through [`MLX_ROUTED_MODELS`]. Keep
/// this test fixture aligned with the importable families and with at least one exact MLX provider
/// registration for each listed family.
///
/// `mage-flow` (sc-15036, epic 14034 F6) joins for a different reason than the other two: its
/// non-builtin ids are not community imports but this app's OWN full base fine-tunes — the
/// `transformer/`-shaped checkpoint a `networkType: "full"` training run produces, registered into
/// the model catalog and loaded by `image_jobs::mage_finetuned` against the installed base's shared
/// text encoder + VAE. The pinned MLX and Candle runtimes both expose the same `load_finetuned`
/// seam, so this family is listed for both native backends in these fixtures. Production still
/// admits one only when the stamped source is the registered `transformer_directory` shape, and the
/// exact provider-facts route below is what decides per backend.
/// The token is the MANIFEST family spelling (`mage-flow`), matching the builtin entries — not the
/// underscored id prefix.
#[cfg(test)]
pub(crate) const MLX_ROUTED_FAMILIES: &[&str] = &["krea_2", "mage-flow", "sdxl"];

/// Test oracle for whether the generated provider facts contain at least one MLX route for a
/// family. It does not authorize a request without an exact source-shaped provider match.
pub(crate) fn image_family_is_mlx_routed(family: &str) -> bool {
    imported_provider_routes("mlx", family).next().is_some()
}

/// Architecture families whose existing Candle engine can serve a non-builtin imported/user
/// same-family model. This is deliberately separate from [`CANDLE_ROUTED_MODELS`]: imported
/// checkpoints have novel ids and are claimed only after the scheduler validates their manifest
/// family and supported request shape. Seeded by the descriptor-gated Krea single-file lane
/// (sc-14023).
#[cfg(test)]
pub(crate) const CANDLE_ROUTED_FAMILIES: &[&str] = &["krea_2", "mage-flow", "sdxl"];

/// Whether `id` names a builtin image model (a row in [`IMAGE_MODEL_CAPS`]). The route-by-family
/// path applies only to non-builtin (imported/user) ids, so a builtin's id-keyed routing is never
/// overridden by its family (sc-14019).
pub fn is_builtin_image_model(id: &str) -> bool {
    IMAGE_MODEL_CAPS.iter().any(|caps| caps.id == id)
}

const MLX_ENGINE_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/capabilities.mlx.json");
const CANDLE_ENGINE_FACTS: &str =
    include_str!("../../../../../config/engine-capabilities/capabilities.candle.json");

/// Checked-in projection of one exact imported-source registration. It is generated from the
/// inference registry and therefore describes the provider that really validates and loads the
/// source, rather than a SceneWorks family table.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedProviderSurface {
    pub family: String,
    pub source: String,
    pub operation: String,
    pub provider_id: String,
    #[serde(default)]
    pub conditioning: Vec<String>,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    #[serde(default)]
    pub supported_quants: Vec<String>,
    pub supports_kv_cache: bool,
    pub supports_sequential_offload: bool,
    pub registry_cached: bool,
}

#[derive(Debug, Deserialize)]
struct ImportedProviderFactsFile {
    backend: String,
    #[serde(default)]
    imports: Vec<ImportedProviderSurface>,
}

fn imported_provider_facts() -> &'static Result<Vec<ImportedProviderFactsFile>, String> {
    static FACTS: OnceLock<Result<Vec<ImportedProviderFactsFile>, String>> = OnceLock::new();
    FACTS.get_or_init(|| {
        [MLX_ENGINE_FACTS, CANDLE_ENGINE_FACTS]
            .into_iter()
            .map(|raw| {
                serde_json::from_str(raw).map_err(|error| {
                    format!("engine import capability facts are malformed: {error}")
                })
            })
            .collect()
    })
}

/// Provider-derived imported routes for one backend and family. Empty means unsupported.
pub fn imported_provider_routes(
    backend: &str,
    family: &str,
) -> std::vec::IntoIter<&'static ImportedProviderSurface> {
    imported_provider_facts()
        .as_ref()
        .ok()
        .into_iter()
        .flat_map(|files| files.iter())
        .filter(move |file| file.backend == backend)
        .flat_map(|file| file.imports.iter())
        .filter(|route| route.family == family)
        .collect::<Vec<_>>()
        .into_iter()
}

fn imported_provider_route(
    backend: &str,
    family: &str,
    source: &str,
    operation: &str,
) -> Option<&'static ImportedProviderSurface> {
    imported_provider_routes(backend, family)
        .find(|route| route.source == source && route.operation == operation)
}

/// Exact structural source shape stamped by the component scanner/importer/training registrar.
/// Absence or an unknown label fails closed; routing must never choose a sibling route from family
/// identity alone.
/// The single imported provider operation a payload selects. Exactly one wins, and the order is the
/// contract: a phase list is a MultiPhase job even when it also carries poses, and so on down.
fn imported_payload_operation(payload: &Map<String, Value>) -> &'static str {
    if has_nonempty_nested_array(payload, "advanced", "phases") {
        "multi_phase"
    } else if has_nonempty_nested_array(payload, "advanced", "poses") {
        "pose"
    } else if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        "edit"
    } else {
        "generate"
    }
}

/// Whether `backend`'s checked-in engine facts DECLARE an imported provider route for the exact
/// (family, source, operation) this payload selects.
///
/// Declaration only. It deliberately stops short of the adapter / quant / conditioning checks
/// [`imported_image_request_provider_eligible`] goes on to make, so a caller can compare "the facts
/// say this backend serves this shape" against an independently measured "the worker actually
/// routes it". That comparison is what lets the capability matrix prove a declared lane is really
/// reachable instead of restating one side of the same lookup.
pub fn imported_backend_declared_route(
    payload: &Map<String, Value>,
    backend: &str,
) -> Option<&'static ImportedProviderSurface> {
    let entry = payload
        .get("modelManifestEntry")
        .and_then(Value::as_object)?;
    let family = entry.get("family").and_then(Value::as_str)?;
    let source = imported_source_shape(entry)?;
    imported_provider_route(backend, family, source, imported_payload_operation(payload))
}

/// Convenience form of [`imported_backend_declared_route`] for callers that only need existence.
pub fn imported_backend_declares_route(payload: &Map<String, Value>, backend: &str) -> bool {
    imported_backend_declared_route(payload, backend).is_some()
}

/// The persisted checkpoint identity a plan-backed manifest entry carries (`importPlan.checkpointId`,
/// epic 20398). The worker's plan-driven route is the only consumer that loads through it; the
/// scheduler treats it purely as the entry's source hint.
pub fn checkpoint_plan_checkpoint_id(entry: &Map<String, Value>) -> Option<&str> {
    entry
        .get("importPlan")
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("checkpointId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
}

fn imported_source_shape(entry: &Map<String, Value>) -> Option<&str> {
    match entry.get("importSourceShape").and_then(Value::as_str) {
        Some(
            source @ ("transformer_file"
            | "fused_checkpoint"
            | "transformer_directory"
            | "comfy_ui_tree"),
        ) => Some(source),
        _ => None,
    }
}

/// Explicit load-time quant requested by the shared image payload. A named tier takes precedence
/// over the legacy MLX bit count. Unknown names are terminal rather than silently coerced.
fn imported_requested_quant(
    payload: &Map<String, Value>,
    ignore_quant_tier: bool,
) -> Result<Option<&'static str>, ()> {
    let advanced = payload.get("advanced").and_then(Value::as_object);
    let named = advanced
        .and_then(|advanced| {
            if ignore_quant_tier {
                advanced.get("quant")
            } else {
                advanced.get("quantTier").or_else(|| advanced.get("quant"))
            }
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase);
    let bits = advanced
        .and_then(|advanced| advanced.get("mlxQuantize"))
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        });
    if let Some(named) = named.as_deref() {
        return match named {
            "nvfp4" => Ok(Some("nvfp4")),
            "q4" => Ok(Some("q4")),
            "q8" => Ok(Some("q8")),
            "bf16" | "dense" => Ok(None),
            _ => Err(()),
        };
    }
    match bits {
        Some(1..=4) => Ok(Some("q4")),
        Some(5..) => Ok(Some("q8")),
        Some(i64::MIN..=0) | None => Ok(None),
    }
}

/// Whether an imported image request's `advanced` object carries user control intent that must be consumed by an exact
/// Pose provider route. `controlImage` is material whenever it is present and non-null (including an
/// invalid empty/non-string value, which must fail closed). `controlMode` is material when its string
/// value is non-empty after trimming; an explicit non-string value is invalid and therefore material
/// too. Empty strings and JSON null retain the historical no-control shape.
///
/// This helper deliberately does not infer Pose from either field. A caller must independently select
/// Pose from a non-empty `advanced.poses` set and prove that exact source/backend operation exists.
pub fn imported_control_intent_is_material(advanced: &Map<String, Value>) -> bool {
    if advanced
        .get("controlImage")
        .is_some_and(|value| !value.is_null())
    {
        return true;
    }
    match advanced.get("controlMode") {
        None | Some(Value::Null) => false,
        Some(Value::String(mode)) => !mode.trim().is_empty(),
        Some(_) => true,
    }
}

/// The imported Pose surface currently assembles a pose-only Krea control branch. Omission/empty
/// keeps that proven default; any explicit non-pose or malformed mode must fail before claim rather
/// than reach a worker predicate that cannot consume it. When provider facts gain per-control-kind
/// metadata this check can be derived from that registration instead.
pub fn imported_pose_control_mode_is_supported(advanced: &Map<String, Value>) -> bool {
    match advanced.get("controlMode") {
        None | Some(Value::Null) => true,
        Some(Value::String(mode)) if mode.trim().is_empty() => true,
        Some(Value::String(mode)) => mode.trim().eq_ignore_ascii_case("pose"),
        Some(_) => false,
    }
}

/// Fail-closed request gate for a non-builtin imported image model. The exact source shape and
/// operation select one provider row; capabilities are never unioned across sibling family routes.
pub fn imported_image_request_provider_eligible(
    model: &str,
    payload: &Map<String, Value>,
    backend: &str,
) -> bool {
    if is_builtin_image_model(model) {
        return false;
    }
    let Some(entry) = payload.get("modelManifestEntry").and_then(Value::as_object) else {
        return false;
    };
    let Some(family) = entry.get("family").and_then(Value::as_str) else {
        return false;
    };
    let Some(source) = imported_source_shape(entry) else {
        return false;
    };
    // The source hint: an installed/linked path for the bespoke imported lanes, or — for a
    // plan-backed entry (epic 20398, sc-20634) — the persisted checkpoint identity the worker's
    // plan-driven route resolves through the checkpoint plan store. Either is a claim that the
    // worker then verifies fail-closed; neither is consumed for loading here.
    let has_nonempty_path = entry
        .get("modelPath")
        .and_then(Value::as_str)
        .or_else(|| {
            entry
                .get("paths")
                .and_then(Value::as_object)
                .and_then(|paths| paths.get("model"))
                .and_then(Value::as_str)
        })
        .or_else(|| entry.get("installedPath").and_then(Value::as_str))
        .or_else(|| {
            entry
                .get("source")
                .and_then(Value::as_object)
                .and_then(|source| source.get("path"))
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .is_some_and(|path| !path.is_empty());
    if !has_nonempty_path && checkpoint_plan_checkpoint_id(entry).is_none() {
        return false;
    }

    let has_reference_list = has_nonempty_string_array(payload, "referenceAssetIds");
    let has_loras = has_nonempty_array(payload, "loras");
    let has_poses = has_nonempty_nested_array(payload, "advanced", "poses");
    let has_phases = has_nonempty_nested_array(payload, "advanced", "phases");
    let is_edit = payload.get("mode").and_then(Value::as_str) == Some("edit_image");
    if has_nonempty_string(payload, "characterId")
        || has_nonempty_string(payload, "characterLookId")
    {
        return false;
    }

    let operation = imported_payload_operation(payload);
    // An explicit user control map/mode is semantically material. It may only accompany a selected
    // Pose operation; never flatten it into Generate/Edit/MultiPhase. The exact route lookup and
    // Control-conditioning check below then prove that this source/backend can consume the request.
    if payload
        .get("advanced")
        .and_then(Value::as_object)
        .is_some_and(imported_control_intent_is_material)
        && operation != "pose"
    {
        return false;
    }
    if operation == "pose"
        && payload
            .get("advanced")
            .and_then(Value::as_object)
            .is_some_and(|advanced| !imported_pose_control_mode_is_supported(advanced))
    {
        return false;
    }
    let Some(route) = imported_provider_route(backend, family, source, operation) else {
        return false;
    };
    if has_loras && !(route.supports_lora || route.supports_lokr) {
        return false;
    }
    let ignore_quant_tier = family == "flux2" && source == "comfy_ui_tree";
    match imported_requested_quant(payload, ignore_quant_tier) {
        Ok(Some(quant)) if !route.supported_quants.iter().any(|value| value == quant) => {
            return false;
        }
        Err(()) => return false,
        Ok(_) => {}
    }

    if has_phases {
        return !is_edit
            && !has_poses
            && !has_reference_list
            && !has_nonempty_string(payload, "referenceAssetId")
            && !has_nonempty_string(payload, "sourceAssetId")
            && !has_nonempty_string(payload, "maskAssetId");
    }
    if has_poses {
        return !is_edit
            && route.conditioning.iter().any(|kind| kind == "control")
            && !has_reference_list
            && !has_nonempty_string(payload, "sourceAssetId")
            && !has_nonempty_string(payload, "maskAssetId");
    }
    if !is_edit
        && (has_nonempty_string(payload, "sourceAssetId")
            || has_nonempty_string(payload, "maskAssetId"))
    {
        return false;
    }
    if has_nonempty_string(payload, "maskAssetId")
        && !route.conditioning.iter().any(|kind| kind == "mask")
    {
        return false;
    }
    if has_reference_list
        && !route
            .conditioning
            .iter()
            .any(|kind| kind == "multi_reference")
    {
        return false;
    }
    let has_single_reference = has_nonempty_string(payload, "referenceAssetId")
        || has_nonempty_string(payload, "sourceAssetId");
    if has_single_reference && !route.conditioning.iter().any(|kind| kind == "reference") {
        return false;
    }
    !is_edit || has_reference_list || has_single_reference
}

/// Per-backend capabilities of the native single-file (imported) lane, the axis
/// [`imported_image_request_family_eligible`] keys request-shape admission on. One named struct —
/// not positional bools — so the mlx.rs / candle.rs call sites and the worker's mirrored
/// `KREA_IMPORTED_SUPPORTS_*` constants cannot silently transpose capabilities.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImportedImageBackendCaps {
    /// The backend's native single-file loader takes an `adapters` slice, so it serves LoRAs and
    /// the Kontext edit surface (whose required identity-edit LoRA is itself an adapter).
    pub(crate) adapters: bool,
    /// The backend can assemble the Krea pose ControlNet branch around a FILE-LOADED DiT: the MLX
    /// runtime's `load_control_from_native_dit_file` folds the trained `Krea2ControlBranch` (a
    /// `control_scale`-scaled residual, architecturally independent of the DiT's weights) onto the
    /// imported transformer, so a same-shape fine-tune serves strict-pose sets.
    pub(crate) pose_control: bool,
}

/// The MLX backend's imported-lane capabilities (mlx.rs / the Mac worker's macOS constants).
#[cfg(test)]
pub(crate) const MLX_IMPORTED_CAPS: ImportedImageBackendCaps = ImportedImageBackendCaps {
    adapters: true,
    pose_control: true,
};

/// The candle backend's imported-lane capabilities (candle.rs / the worker's candle constants).
#[cfg(test)]
pub(crate) const CANDLE_IMPORTED_CAPS: ImportedImageBackendCaps = ImportedImageBackendCaps {
    adapters: true,
    pose_control: true,
};

/// Shared fail-closed request-shape gate for a non-builtin imported image id that reuses a native
/// engine by family. The worker owns filesystem confinement and verifies that the recorded path
/// resolves to exactly one safetensors file; the scheduler only admits the matching family, a
/// non-empty installed path hint, and a request shape the imported-family handlers implement.
///
/// `caps` is the selected backend's native single-file loader capability surface
/// ([`MLX_IMPORTED_CAPS`] / [`CANDLE_IMPORTED_CAPS`]): `adapters` admits LoRAs + the Kontext edit
/// surface, `pose_control` admits a strict-pose set on the `krea_2` family. img2img (a single
/// `referenceAssetId` on a non-edit mode, resolved by the worker's `resolve_img2img_init_generic`)
/// needs no capability, so it is admitted on **both** backends.
#[cfg(test)]
pub(crate) fn imported_image_request_family_eligible(
    model: &str,
    payload: &Map<String, Value>,
    routed_families: &[&str],
    caps: ImportedImageBackendCaps,
) -> bool {
    if is_builtin_image_model(model) {
        return false;
    }
    let Some(entry) = payload.get("modelManifestEntry").and_then(Value::as_object) else {
        return false;
    };
    let family = entry.get("family").and_then(Value::as_str);
    if !family.is_some_and(|family| routed_families.contains(&family)) {
        return false;
    }
    let has_nonempty_path = entry
        .get("modelPath")
        .and_then(Value::as_str)
        .or_else(|| {
            entry
                .get("paths")
                .and_then(Value::as_object)
                .and_then(|paths| paths.get("model"))
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .is_some_and(|path| !path.is_empty());
    if !has_nonempty_path {
        return false;
    }
    let has_reference_list = has_nonempty_string_array(payload, "referenceAssetIds");
    let has_loras = has_nonempty_array(payload, "loras");
    let has_poses = has_nonempty_nested_array(payload, "advanced", "poses");
    let has_phases = has_nonempty_nested_array(payload, "advanced", "phases");
    let has_hires_fix = payload
        .get("hiresFix")
        .and_then(Value::as_object)
        .and_then(|hires| hires.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // A fused SDXL checkpoint contains the complete denoiser + dual text encoders + VAE, so its
    // single-file lane is not Krea's bare-transformer assembly surface. Both native loaders accept
    // UNet LoRA/LoKr adapters, but the worker lane intentionally claims only the descriptor's common
    // txt2img surface; edit/reference/control requests keep flowing to their established specialized
    // routes instead of silently dropping conditioning.
    if family == Some("sdxl") {
        return payload.get("mode").and_then(Value::as_str).map(str::trim) != Some("edit_image")
            && !has_reference_list
            && !has_nonempty_string(payload, "referenceAssetId")
            && !has_nonempty_string(payload, "sourceAssetId")
            && !has_poses
            && !has_phases
            && !has_nonempty_string(payload, "maskAssetId")
            && !has_nonempty_string(payload, "characterId")
            && !has_nonempty_string(payload, "characterLookId");
    }

    // A Mage-Flow non-builtin id is a full base FINE-TUNE (sc-15036, epic 14034 F6) — the
    // `transformer/`-shaped checkpoint a `networkType: "full"` training run writes, paired at load
    // with the installed base's shared text encoder + VAE. The Mage generator advertises no
    // conditioning on the non-edit variants, and `mlx_gen_mage::load_finetuned` refuses adapters
    // outright, so this claims the plain **txt2img** surface ONLY. Everything else keeps flowing to
    // its established route rather than being silently flattened to t2i here — including LoRAs,
    // which are rejected on EVERY backend (unlike Krea's adapter-capable single-file lane, hence the
    // explicit check rather than relying on `adapters_supported`).
    if family == Some("mage-flow") {
        return payload.get("mode").and_then(Value::as_str).map(str::trim) != Some("edit_image")
            && !has_reference_list
            && !has_loras
            && !has_nonempty_string(payload, "referenceAssetId")
            && !has_nonempty_string(payload, "sourceAssetId")
            && !has_poses
            && !has_phases
            && !has_nonempty_string(payload, "maskAssetId")
            && !has_nonempty_string(payload, "characterId")
            && !has_nonempty_string(payload, "characterLookId");
    }

    // Never on this bare-transformer lane, on any backend: multi-phase, inpaint mask, and
    // character / look identity conditioning (all need base-tier components it does not stage).
    if has_phases
        || has_nonempty_string(payload, "maskAssetId")
        || has_nonempty_string(payload, "characterId")
        || has_nonempty_string(payload, "characterLookId")
    {
        return false;
    }
    let is_edit = payload.get("mode").and_then(Value::as_str).map(str::trim) == Some("edit_image");

    // Imported Krea hires currently serves only unconditioned t2i. The generic two-pass helper has
    // no imported edit/img2img/pose conditioning inputs, so reject those combinations at claim time
    // just as the worker does instead of returning a render that silently dropped them.
    if has_hires_fix && (is_edit || has_poses || has_nonempty_string(payload, "referenceAssetId")) {
        return false;
    }

    // Strict pose (a non-empty `advanced.poses`): the imported pose-control surface. The trained
    // Krea pose branch is a `control_scale`-scaled residual folded onto the frozen DiT at load —
    // architecturally independent of the DiT's weights — so it composes with a same-shape imported
    // fine-tune exactly as with the builtin base. Admitted only where the worker can actually
    // assemble it: the `krea_2` family on a pose-control-capable backend (`caps.pose_control`),
    // outside edit mode, without the plural reference set / bare `sourceAssetId` the pose render
    // loop would silently drop (a single `referenceAssetId` is the identity-likeness scoring
    // source, the builtin `krea_control_available` semantics). LoRAs ride the adapter path on the
    // imported DiT under the branch. Mirrors the worker's `krea_imported_available` pose arm.
    if has_poses {
        return family == Some("krea_2")
            && caps.pose_control
            && !is_edit
            && !has_reference_list
            && !has_nonempty_string(payload, "sourceAssetId")
            && (!has_loras || caps.adapters);
    }
    // LoRAs ride the native single-file loader's adapter path; reject them on any backend whose
    // provider does not advertise that load-time seam.
    if has_loras && !caps.adapters {
        return false;
    }

    if is_edit {
        // Kontext-style edit (sc-14119): the adapter-capable backend + a conditioning image, which
        // can arrive as the singular `referenceAssetId`, the plural scene+person set, or a
        // `sourceAssetId` — the same priority the worker's `edit_reference_ids` resolves. The
        // required `krea2_identity_edit` LoRA is enforced worker-side.
        return caps.adapters
            && (has_reference_list
                || has_nonempty_string(payload, "referenceAssetId")
                || has_nonempty_string(payload, "sourceAssetId"));
    }

    // Non-edit t2i / img2img: img2img rides a single `referenceAssetId` (the worker's
    // `resolve_img2img_init_generic`, sc-14071). The plural set is the edit surface, and a bare
    // `sourceAssetId` is not read on the img2img path (it would silently render plain t2i), so both
    // are rejected outside edit mode — mirroring the worker's `krea_imported_available`.
    if has_reference_list || has_nonempty_string(payload, "sourceAssetId") {
        return false;
    }
    true
}

/// Minimal probe payload for a non-builtin image request of `family` at `model`, carrying exactly
/// the fields [`imported_image_request_provider_eligible`] reads: the manifest entry (family + a
/// non-empty installed path) and, when `with_lora`, a single job LoRA. Everything else is absent, so
/// the probe is the PLAIN t2i shape — the surface every imported lane claims — plus/minus adapters.
fn imported_image_lora_probe(
    model: &str,
    family: &str,
    source: &str,
    with_lora: bool,
) -> Map<String, Value> {
    let mut payload = json!({
        "model": model,
        "modelManifestEntry": {
            "id": model,
            "family": family,
            "importSourceShape": source,
            "paths": { "model": "/probe" }
        }
    });
    if with_lora {
        payload
            .as_object_mut()
            .expect("probe payload is an object")
            .insert("loras".to_owned(), json!([{ "id": "probe" }]));
    }
    payload
        .as_object()
        .expect("probe payload is an object")
        .clone()
}

/// Whether a non-builtin (imported / fine-tuned) image model may ADVERTISE LoRA compatibility on a
/// deployment offering the given native lanes — the advertisement-side twin of
/// [`imported_image_request_provider_eligible`] (sc-14135 follow-up; the class sc-15328 named).
///
/// * `None` — the family is not served by the route-by-family path at all, so this oracle has no
///   opinion and the entry must be left exactly as-is (its problem, if any, is that it renders
///   nothing, not that it over-advertises adapters).
/// * `Some(true)` — some available lane claims the model WITH a LoRA attached.
/// * `Some(false)` — some available lane claims the plain t2i shape but NONE claims it with a LoRA.
///   Advertising `loraCompatibility` here is the exact hang: the API accepts the submission and no
///   worker ever claims it, so the job sits on "Waiting for an available GPU worker" forever.
///
/// 🔴 Derived by asking the REAL gate — the same function the scheduler calls, with the same
/// exact provider facts used by both routers. Restating the per-family verdict as its own table here
/// is precisely how the advertisement and the gate drift apart again, so it is computed, never
/// copied.
pub fn imported_image_model_lora_advertisement(
    model: &str,
    family: &str,
    source: &str,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) -> Option<bool> {
    let claims = |with_lora: bool| {
        let payload = imported_image_lora_probe(model, family, source, with_lora);
        (mlx_lane_available && imported_image_request_provider_eligible(model, &payload, "mlx"))
            || (candle_lane_available
                && imported_image_request_provider_eligible(model, &payload, "candle"))
    };
    if !claims(false) {
        return None;
    }
    Some(claims(true))
}

derive_model_list! {
    /// The models the candle (Windows/CUDA) lane can serve for base txt2img (derived from
    /// [`IMAGE_MODEL_CAPS`]`.candle_routed`, sc-9495). Mirrors the worker's `image_jobs::is_candle_engine`.
    /// Deliberately limited to the unconditioned base surface: every other request shape must be
    /// claimed explicitly by a descriptor-backed or bespoke lane in `image_job_is_candle_eligible`.
    pub(crate) CANDLE_ROUTED_MODELS, IMAGE_MODEL_CAPS, candle_routed
}

/// Built-in image model ids whose native, unconditioned shape is routed to the Candle worker.
///
/// This is the public read-only oracle for worker/scheduler parity checks. Callers must still apply
/// request-shape routing: some ids (currently `bernini_image`) use a bespoke Candle lane rather than
/// the generic txt2img engine, and imported `external_base_*` rows are routed from their forwarded
/// manifest instead of this built-in catalog.
pub fn candle_routed_image_models() -> &'static [&'static str] {
    CANDLE_ROUTED_MODELS
}

derive_model_list! {
    /// The candle image families that accept Q4/Q8 generation requests AND LoRA/LoKr adapters
    /// (derived from [`IMAGE_MODEL_CAPS`]`.candle_quant_lora`). For these a LoRA or an explicit quant
    /// request stays on candle instead of being refused. Subset of [`CANDLE_ROUTED_MODELS`].
    pub(crate) CANDLE_QUANT_LORA_MODELS, IMAGE_MODEL_CAPS, candle_quant_lora
}

derive_model_list! {
    /// The candle image families that accept Q4/Q8 generation requests but NOT inference LoRA —
    /// either by selecting a pre-packed tier (including Z-Image) or by honoring load-time quant.
    /// Derived from [`IMAGE_MODEL_CAPS`]`.candle_quant` (sc-9495). Quant stays on candle; a LoRA is
    /// refused and remains queued.
    /// Disjoint from [`CANDLE_QUANT_LORA_MODELS`]; both are consulted by the gate. Subset of
    /// [`CANDLE_ROUTED_MODELS`].
    pub(crate) CANDLE_QUANT_MODELS, IMAGE_MODEL_CAPS, candle_quant
}

derive_model_list! {
    /// The candle image families that advertise inference LoRA/LoKr but NOT Q4/Q8 generation requests
    /// (derived from [`IMAGE_MODEL_CAPS`]`.candle_lora`). The mirror of [`CANDLE_QUANT_MODELS`]; both
    /// plus [`CANDLE_QUANT_LORA_MODELS`] are disjoint and all are consulted by the gate. Subset of
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

/// Whether `model` has ANY MLX video route (sc-18814). The `video_mlx_routed` column of
/// [`VIDEO_MODEL_CAPS`], asked without a payload — the mode-level question is
/// [`super::mlx::video_mode_is_mlx_eligible`].
///
/// Exists so the video memory gate (`crate::video_request::video_admission_surface`) can state its
/// per-family backend surface from the routing catalog — the backend authority — instead of from
/// the manifest's advisory `mlx` / `candle` hint objects, which describe intent rather than
/// routing.
pub(crate) fn video_model_is_mlx_video_routed(model: &str) -> bool {
    VIDEO_MLX_ROUTED_MODELS.contains(&model)
}

/// Whether `model` has ANY candle video route (sc-18814) — the candle half of
/// [`video_model_is_mlx_video_routed`], and **not** simply the `candle_video_routed` column.
///
/// Two families reach candle through their OWN distinct engines rather than through
/// `CANDLE_VIDEO_ROUTED_MODELS`, so reading that column alone would report them unroutable:
///
/// * `scail2_14b` — `candle_video_routed = false` by design (it is not a Wan-VACE model), yet
///   [`super::candle::scail2_animate_candle_eligible`] /
///   [`super::candle::scail2_replace_candle_eligible`] route it to the distinct candle SCAIL-2
///   engine (sc-6837), which the worker dispatches as `CandleVideoRoute::AnimateScail2` /
///   `::ReplacePersonScail2`.
/// * `bernini` — in the column, but reached via
///   [`super::candle::bernini_video_candle_eligible`] rather than the generic txt2video arm.
///
/// `krea_realtime_14b` is the one video family with genuinely NO candle engine: there is no
/// `candle-gen-krea-realtime` at all, its MLX descriptor is `mac_only`, and
/// `run_video_generate_job` fails an off-Mac krea job loudly rather than routing it elsewhere.
///
/// Pinned against the per-model predicates by `crate::video_request`'s
/// `video_admission_surface_matches_the_routing_catalog`.
pub(crate) fn video_model_has_candle_video_route(model: &str) -> bool {
    CANDLE_VIDEO_ROUTED_MODELS.contains(&model) || CANDLE_SCAIL2_VIDEO_MODELS.contains(&model)
}

/// Video models served off-Mac by a DISTINCT candle engine that is deliberately absent from
/// [`CANDLE_VIDEO_ROUTED_MODELS`] (sc-6837): the candle SCAIL-2 engine, gated by
/// [`super::candle::scail2_animate_candle_eligible`] /
/// [`super::candle::scail2_replace_candle_eligible`] rather than by table membership.
pub(crate) const CANDLE_SCAIL2_VIDEO_MODELS: &[&str] = &["scail2_14b"];

/// SceneWorks training kernels with a native mlx-gen Rust trainer (epic 3039):
/// the engine registers `z_image_turbo`/`sdxl`/`kolors`/`ltx_2_3`/`wan2_2_*` trainers,
/// which the worker reaches via these SceneWorks kernel ids (the mlx worker maps the
/// kernel and base model onto an engine trainer id). This list describes MLX routing only; several
/// entries also have Candle trainers. A kernel absent here is never routed to the MLX worker.
///
/// Public (sc-15277) so the worker's `engine_trainer_id` invariant can be DERIVED from this list
/// rather than restating it: a kernel that is routed here but has no trainer mapping is claimed by
/// the mlx worker and then immediately failed, which is exactly how `mage_flow_lora` shipped broken.
pub const MLX_ROUTED_TRAINING_KERNELS: &[&str] = &[
    "z_image_lora",
    "sdxl_lora",
    "kolors_lora",
    "lens_lora",
    "krea_lora",
    "sd3_lora",
    "wan_lora",
    "wan_moe_lora",
    "ltx_mlx_lora",
    // Anima (epic 10512, sc-10522): the native `mlx-gen-anima` LoRA/LoKr trainer (DiT + `llm_adapter`
    // conditioner). Candle reached parity in sc-18479.
    "anima_lora",
    // Mage-Flow adapters and full base fine-tunes have native MLX and Candle trainers.
    "mage_flow_lora",
];

/// SceneWorks training kernels with a native candle trainer that needs no base-model disambiguation
/// (sc-7817, epic 5164) — the off-Mac twin of [`MLX_ROUTED_TRAINING_KERNELS`]. The registry includes
/// Kolors, SD3.5, Anima, Mage, Wan TI2V-5B, and both Wan A14B experts as of sc-18479. Families that
/// share a kernel remain base-model gated by [`training_job_is_candle_eligible`]. LTX keeps its
/// historical `ltx_mlx_lora` kernel name on both native backends.
/// `krea_control` (epic 10159, B2 sc-10163 / B1 sc-10162) is the Krea 2 pose-ControlNet branch trainer
/// (candle-gen-krea `ControlTrainer`, dispatched under `krea_2_control`); it has no MLX trainer, so —
/// like `krea_lora` — it is also in [`MLX_ONLY_TRAINING_KERNELS`] but NOT
/// [`MLX_ROUTED_TRAINING_KERNELS`] (the MLX control lane is B5/sc-10177).
pub(crate) const CANDLE_ROUTED_TRAINING_KERNELS: &[&str] = &[
    "z_image_lora",
    "sdxl_lora",
    "kolors_lora",
    "lens_lora",
    "krea_lora",
    "krea_control",
    "sd3_lora",
    "ltx_mlx_lora",
    "wan_lora",
    "wan_moe_lora",
    "anima_lora",
    "mage_flow_lora",
];

/// Native-only training kernels — only a Rust worker can run them, so a generic worker descriptor
/// must refuse the job (leaving it queued for a Rust worker) rather than claim it and fail with "no
/// training kernel". Despite the constant's historical name, this means native-Rust-only rather than
/// MLX-backend-only: most members have both MLX and Candle trainers. Candle workers are admitted
/// by the [`worker_supports_job`] exception when the kernel is also in
/// [`CANDLE_ROUTED_TRAINING_KERNELS`], while a generic worker remains refused. `krea_control` is the
/// intentional Candle-only exception; the historical constant name remains for compatibility.
pub(crate) const MLX_ONLY_TRAINING_KERNELS: &[&str] = &[
    "ltx_mlx_lora",
    "krea_lora",
    "sd3_lora",
    "anima_lora",
    "krea_control",
    "mage_flow_lora",
];

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
        image_family_is_mlx_routed, image_model_mac_support, imported_control_intent_is_material,
        imported_image_model_lora_advertisement, imported_image_request_family_eligible,
        imported_image_request_provider_eligible, imported_provider_routes, is_builtin_image_model,
        CANDLE_IMPORTED_CAPS, CANDLE_LORA_MODELS, CANDLE_QUANT_LORA_MODELS, CANDLE_QUANT_MODELS,
        CANDLE_ROUTED_FAMILIES, CANDLE_ROUTED_MODELS, CANDLE_ROUTED_TRAINING_KERNELS,
        CANDLE_VIDEO_I2V_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS, CANDLE_VIDEO_VACE_MODELS,
        IMAGE_MODEL_CAPS, MLX_IMPORTED_CAPS, MLX_ONLY_TRAINING_KERNELS, MLX_ROUTED_FAMILIES,
        MLX_ROUTED_MODELS, MLX_ROUTED_TRAINING_KERNELS, VIDEO_MLX_ROUTED_MODELS, VIDEO_MODEL_CAPS,
    };

    #[test]
    fn imported_control_intent_distinguishes_material_values_without_inventing_pose() {
        for advanced in [
            serde_json::json!({ "controlImage": "asset" }),
            serde_json::json!({ "controlImage": "" }),
            serde_json::json!({ "controlImage": 7 }),
            serde_json::json!({ "controlMode": "pose" }),
            serde_json::json!({ "controlMode": " canny " }),
            serde_json::json!({ "controlMode": false }),
        ] {
            assert!(imported_control_intent_is_material(
                advanced.as_object().unwrap()
            ));
        }
        for advanced in [
            serde_json::json!({}),
            serde_json::json!({ "controlImage": null }),
            serde_json::json!({ "controlMode": null }),
            serde_json::json!({ "controlMode": "  " }),
        ] {
            assert!(!imported_control_intent_is_material(
                advanced.as_object().unwrap()
            ));
        }
    }

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
        "mage_flow_base",
        "mage_flow",
        "mage_flow_turbo",
        "mage_flow_edit_base",
        "mage_flow_edit",
        "mage_flow_edit_turbo",
        "z_image_turbo",
        "z_image",
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
        "illustrious_xl_v1",
        "illustrious_xl_v2",
        "realvisxl_lightning",
        "instantid_realvisxl",
        "pulid_flux_dev",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_infographic_v3_fast",
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
        "krea_2_raw",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sana_1600m",
        "sana_sprint_1600m",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
    ];

    const EXPECTED_CANDLE_ROUTED_MODELS: &[&str] = &[
        "mage_flow_base",
        "mage_flow",
        "mage_flow_turbo",
        "mage_flow_edit_base",
        "mage_flow_edit",
        "mage_flow_edit_turbo",
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
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
        // sc-10996 (epic 6562): the candle Bernini still-image companion joins the routed set.
        "bernini_image",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "kolors",
        "sensenova_u1_8b",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3_fast",
        "ideogram_4",
        "ideogram_4_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        "krea_2_turbo",
        "krea_2_raw",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        // sc-11780 (epic 8485): the candle SANA 1600M provider (candle-gen #495) joins the routed set —
        // true-CFG txt2img plus singular-reference img2img on the whole
        // `Efficient-Large-Model/Sana_1600M_1024px_diffusers` snapshot.
        "sana_1600m",
        // sc-11781 (epic 8485): the candle SANA-Sprint provider (candle-gen #498) joins the routed set too —
        // CFG-free 1–4 step SCM/TrigFlow txt2img plus singular-reference img2img on the whole
        // `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` snapshot.
        "sana_sprint_1600m",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
    ];

    // sc-9983: Krea joins Lens as a BOTH-quant-and-LoRA candle family (sc-9607 flipped its
    // `supported_quants` to [Q4, Q8]; it already advertised inference LoRA via sc-7836). sc-9994 adds the
    // Raw variant (candle-gen #350) with the same both-set advertisement. sc-10767: the SDXL family
    // (sdxl/realvisxl/illustrious v1+v2) joins the both-set — the candle packed q4/q8 tier (sc-9416/9527)
    // + adapter-on-packed fold (sc-9528) are wired and now advertised. sc-10812: realvisxl_lightning (the
    // few-step distilled sibling on the SAME `sdxl` engine / descriptor) joins the both-set too — quant +
    // LoRA stay on candle for its plain txt2img shape.
    const EXPECTED_CANDLE_QUANT_LORA_MODELS: &[&str] = &[
        "bernini_image",
        "mage_flow_base",
        "mage_flow",
        "mage_flow_turbo",
        "mage_flow_edit_base",
        "mage_flow_edit",
        "mage_flow_edit_turbo",
        "z_image_turbo",
        "z_image",
        "qwen_image",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_dev",
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
        "realvisxl_lightning",
        "lens",
        "lens_turbo",
        "krea_2_turbo",
        "krea_2_raw",
        "kolors",
        "ideogram_4",
        "ideogram_4_turbo",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
    ];

    // Z-Image Turbo/base select already-packed q4/q8/bf16 directories; this is not on-the-fly quant,
    // but it is still a valid `mlxQuantize` request and therefore belongs in the quant routing set.
    // The adapter-capable packed families moved to `EXPECTED_CANDLE_QUANT_LORA_MODELS` above;
    // this list retains families that accept quant requests without user adapters. Historically,
    // these rows covered Ideogram, Kolors, Qwen, FLUX.2, SD3.5, and Z-Image as quant-only.
    const EXPECTED_CANDLE_QUANT_MODELS: &[&str] = &[
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        // sc-11020: qwen_image's turnkey q4/q8/bf16 packed tiers (sc-8669, measured sc-10969) load on
        // the candle txt2img lane, so a tier-select stays on candle. Qwen now appears in the combined
        // quant+adapter list above.
        // sc-10222: FLUX.2-klein 9B/`_kv` + FLUX.2-dev — the same missed router half, for the last
        // `STANDARD_TIER_MODELS` families still carrying it. `_true_v2` is deliberately absent (a flat
        // convert-at-install dir with no tier matrix); see the caps rows for the full reasoning.
        // sc-14249: the whole SenseNova-U1 family, once `candle-gen-sensenova` gained the packed
        // q4/q8 load path (it was dense-f32-only, and only the bf16 tier was readable at all).
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_infographic_v3_fast",
    ];

    // sc-9983: Krea moved to CANDLE_QUANT_LORA_MODELS (BOTH). sc-10676: Anima is the LoRA-only candle
    // family — the off-Mac engine dense-folds a LoRA/LoKr onto the split_files/ DiT, but advertises NO
    // candle quant (no packed tier off-Mac), so it is LoRA-only rather than BOTH.
    const EXPECTED_CANDLE_LORA_MODELS: &[&str] = &[
        "flux_schnell",
        "flux_dev",
        "flux2_klein_9b_true_v2",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
    ];

    const EXPECTED_CANDLE_VIDEO_ROUTED_MODELS: &[&str] = &[
        "wan_2_2",
        "ltx_2_3",
        "wan_2_2_t2v_14b",
        "wan_2_2_i2v_14b",
        "svd",
        // Bernini VIDEO lane (sc-10997, epic 6562): the full candle planner+renderer serves t2v + the
        // editing/reference/multi-source modes. Not in the i2v/VACE subsets (a distinct engine).
        "bernini",
        // Mochi 1 (sc-11991): the candle descriptor is `mac_only: false` and ingests the same hosted
        // mlx-affine tiers, so the off-Mac t2v lane is real. Not in the i2v/VACE subsets (t2v only).
        "mochi_1",
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
        "wan_2_2_vace_fun_14b",
        "svd",
        "bernini",
        "scail2_14b",
        "mochi_1",
        "krea_realtime_14b",
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
        "anima_lora",
        // sc-14056: Mage-Flow's native mlx trainer is routed to the mlx worker. Before this, the
        // kernel was in `MLX_ONLY_TRAINING_KERNELS` but in NO routed set, so every worker refused it
        // and Mage training jobs queued forever.
        "mage_flow_lora",
    ];

    const EXPECTED_CANDLE_ROUTED_TRAINING_KERNELS: &[&str] = &[
        "z_image_lora",
        "sdxl_lora",
        "kolors_lora",
        "lens_lora",
        "krea_lora",
        "krea_control",
        "sd3_lora",
        "ltx_mlx_lora",
        "wan_lora",
        "wan_moe_lora",
        "anima_lora",
        "mage_flow_lora",
    ];

    const EXPECTED_MLX_ONLY_TRAINING_KERNELS: &[&str] = &[
        "ltx_mlx_lora",
        "krea_lora",
        "sd3_lora",
        "anima_lora",
        "krea_control",
        "mage_flow_lora",
    ];

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

    // --- route-by-family for imported models (sc-14019, epic 14015) -------------

    /// The MLX-routed *family* allow-list, pinned like the model lists above. Adding a family here is
    /// a deliberate edit (it makes every imported same-family checkpoint Mac-routable), so it must be
    /// mirrored here — the guardrail that a family is never silently added to the import surface.
    const EXPECTED_MLX_ROUTED_FAMILIES: &[&str] = &["krea_2", "mage-flow", "sdxl"];
    const EXPECTED_CANDLE_ROUTED_FAMILIES: &[&str] = &["krea_2", "mage-flow", "sdxl"];

    #[test]
    fn mlx_routed_families_match_snapshot() {
        assert_eq!(MLX_ROUTED_FAMILIES, EXPECTED_MLX_ROUTED_FAMILIES);
    }

    #[test]
    fn candle_routed_families_match_snapshot() {
        assert_eq!(CANDLE_ROUTED_FAMILIES, EXPECTED_CANDLE_ROUTED_FAMILIES);
        assert!(CANDLE_ROUTED_FAMILIES.contains(&"krea_2"));
        assert!(!CANDLE_ROUTED_FAMILIES.contains(&"z-image"));
    }

    #[test]
    fn mlx_routed_families_have_a_same_family_builtin_engine() {
        // Every allow-listed family must actually be one an MLX-routed builtin defines an engine for
        // (the routing catalog has no id→family map, so this asserts the family token is real: at
        // least one builtin id begins with the family token, and it is MLX-routed). krea_2 →
        // krea_2_turbo / krea_2_raw. This catches a typo'd or dead family token in the allow-list.
        //
        // sc-15036: the token compared is the *manifest* `family` string, whose separator is not
        // uniform across families — `krea_2` and `sdxl` use `_`, while `mage-flow` (and `z-image`,
        // `qwen-image`, …) use `-`, and every id uses `_`. Normalize the separator before the
        // prefix check so the guard keeps asserting "this family token names a real MLX-routed
        // builtin" instead of accidentally asserting "the family token happens to be spelled with
        // underscores".
        for family in MLX_ROUTED_FAMILIES {
            let prefix = family.replace('-', "_");
            let has_routed_builtin = MLX_ROUTED_MODELS
                .iter()
                .any(|id| id.starts_with(&prefix) && is_builtin_image_model(id));
            assert!(
                has_routed_builtin,
                "MLX_ROUTED_FAMILIES lists '{family}' but no MLX-routed builtin id shares that family prefix"
            );
        }
    }

    #[test]
    fn candle_routed_families_have_a_same_family_builtin_engine() {
        for family in CANDLE_ROUTED_FAMILIES {
            let prefix = family.replace('-', "_");
            let has_routed_builtin = CANDLE_ROUTED_MODELS
                .iter()
                .any(|id| id.starts_with(&prefix) && is_builtin_image_model(id));
            assert!(
                has_routed_builtin,
                "CANDLE_ROUTED_FAMILIES lists '{family}' but no Candle-routed builtin id shares that family prefix"
            );
        }
    }

    #[test]
    fn mlx_routed_families_agree_with_import_supported_families() {
        // sc-14019: `MLX_ROUTED_FAMILIES` (routing, here) and `base_weights::IMPORT_SUPPORTED_FAMILIES`
        // (the model-import compatibility gate) are two independent `krea_2` allow-lists that must NOT
        // silently drift. The route surface is the union of the MLX and Candle family lists: a family
        // accepted by the import gate must have at least one native route, and a routed family must be
        // admitted by the import gate.
        let routed: BTreeSet<&str> = MLX_ROUTED_FAMILIES
            .iter()
            .chain(CANDLE_ROUTED_FAMILIES.iter())
            .copied()
            .collect();
        let importable: BTreeSet<&str> = crate::base_weights::IMPORT_SUPPORTED_FAMILIES
            .iter()
            .copied()
            .collect();
        assert_eq!(
            routed, importable,
            "the native route-by-family union and IMPORT_SUPPORTED_FAMILIES disagree; add a family \
             to the import gate and at least one native route list (or remove it from both)"
        );
    }

    #[test]
    fn imported_same_family_model_requires_an_exact_provider_source() {
        // A novel imported id plus a family token is not enough to select a loader. The catalog
        // projection has the full manifest entry and applies the exact provider surface; this
        // family-only core probe must stay fail-closed.
        let imported_id = "user_kreamania_variant5"; // a novel id, never a builtin
        assert!(!is_builtin_image_model(imported_id));
        assert!(!MLX_ROUTED_MODELS.contains(&imported_id));

        let support = image_model_mac_support(imported_id, Some("krea_2"));
        assert!(!support.supported);
        assert!(support.reason.is_some());

        let exact = serde_json::json!({
            "model": imported_id,
            "modelManifestEntry": {
                "id": imported_id,
                "family": "krea_2",
                "importSourceShape": "transformer_file",
                "paths": { "model": "/app/models/imports/kreamania_variant5" }
            }
        });
        assert!(imported_image_request_provider_eligible(
            imported_id,
            exact.as_object().expect("probe is an object"),
            "mlx"
        ));

        let mut missing_source = exact;
        missing_source["modelManifestEntry"]
            .as_object_mut()
            .expect("manifest entry is an object")
            .remove("importSourceShape");
        assert!(!imported_image_request_provider_eligible(
            imported_id,
            missing_source.as_object().expect("probe is an object"),
            "mlx"
        ));
    }

    /// sc-20634: a plan-backed manifest entry (`importPlan.checkpointId`, no installed path) is an
    /// admissible imported Generate claim on its own; blank/missing identities and a missing source
    /// shape stay fail-closed.
    #[test]
    fn plan_backed_entry_is_an_imported_provider_claim_without_an_installed_path() {
        let imported_id = "user_linked_kreamania";
        let plan_backed = serde_json::json!({
            "model": imported_id,
            "modelManifestEntry": {
                "id": imported_id,
                "family": "krea_2",
                "importSourceShape": "transformer_file",
                "importPlan": { "checkpointId": "linked/root-0123456789abcdef/kreamania.safetensors" }
            }
        });
        assert!(imported_image_request_provider_eligible(
            imported_id,
            plan_backed.as_object().unwrap(),
            "mlx"
        ));
        assert!(imported_image_request_provider_eligible(
            imported_id,
            plan_backed.as_object().unwrap(),
            "candle"
        ));

        let mut blank = plan_backed.clone();
        blank["modelManifestEntry"]["importPlan"]["checkpointId"] = serde_json::json!("   ");
        assert!(!imported_image_request_provider_eligible(
            imported_id,
            blank.as_object().unwrap(),
            "mlx"
        ));
        let mut no_plan = plan_backed.clone();
        no_plan["modelManifestEntry"]
            .as_object_mut()
            .unwrap()
            .remove("importPlan");
        assert!(!imported_image_request_provider_eligible(
            imported_id,
            no_plan.as_object().unwrap(),
            "mlx"
        ));
        let mut no_shape = plan_backed;
        no_shape["modelManifestEntry"]
            .as_object_mut()
            .unwrap()
            .remove("importSourceShape");
        assert!(!imported_image_request_provider_eligible(
            imported_id,
            no_shape.as_object().unwrap(),
            "mlx"
        ));
    }

    #[test]
    fn imported_non_routed_family_model_stays_unsupported() {
        // A non-MLX-routed family (or no family) does NOT get the family path — it stays hidden with
        // the standard "not available on Mac" gap reason, exactly as before sc-14019.
        assert!(!image_family_is_mlx_routed("z-image")); // detector family for an unsupported import
        let unsupported = image_model_mac_support("user_zimage_import", Some("z-image"));
        assert!(!unsupported.supported);
        assert!(unsupported.reason.is_some());

        let no_family = image_model_mac_support("user_mystery_import", None);
        assert!(!no_family.supported);
    }

    #[test]
    fn builtin_image_routing_is_unaffected_by_family_argument() {
        // A builtin short-circuits on its id: passing a (possibly wrong) family never changes its
        // verdict, so builtin id-keyed routing stays byte-identical (sc-14019 constraint).
        let with_family = image_model_mac_support("krea_2_turbo", Some("krea_2"));
        let without_family = image_model_mac_support("krea_2_turbo", None);
        let mismatched_family = image_model_mac_support("krea_2_turbo", Some("z-image"));
        assert!(with_family.supported);
        assert_eq!(with_family, without_family);
        assert_eq!(with_family, mismatched_family);
    }

    /// The shared imported-family gate follows each backend's provider surface. At the current pin,
    /// MLX and Candle both take adapters and can assemble strict Krea pose control around an imported
    /// DiT, so t2i, img2img, adapter, edit, and pose shapes are admitted symmetrically.
    #[test]
    fn imported_family_gate_is_adapter_capability_aware() {
        let imported_id = "user_kreamania_variant5";
        let payload = |extra: serde_json::Value| -> serde_json::Map<String, serde_json::Value> {
            let mut base = serde_json::json!({
                "model": imported_id,
                "modelManifestEntry": {
                    "id": imported_id,
                    "family": "krea_2",
                    "paths": { "model": "/app/models/imports/kreamania_variant5" }
                },
            });
            base.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            base.as_object().unwrap().clone()
        };
        let mlx = |p: &serde_json::Map<String, serde_json::Value>| {
            imported_image_request_family_eligible(
                imported_id,
                p,
                MLX_ROUTED_FAMILIES,
                MLX_IMPORTED_CAPS,
            )
        };
        let candle = |p: &serde_json::Map<String, serde_json::Value>| {
            imported_image_request_family_eligible(
                imported_id,
                p,
                CANDLE_ROUTED_FAMILIES,
                CANDLE_IMPORTED_CAPS,
            )
        };

        // Plain t2i + non-edit img2img: eligible on BOTH backends (no adapter needed).
        for p in [
            payload(serde_json::json!({ "mode": "text_to_image" })),
            payload(serde_json::json!({ "referenceAssetId": "asset-1" })),
        ] {
            assert!(mlx(&p), "t2i/img2img eligible on MLX");
            assert!(candle(&p), "t2i/img2img eligible on candle");
        }
        let plain_hires = payload(serde_json::json!({ "hiresFix": { "enabled": true } }));
        assert!(mlx(&plain_hires) && candle(&plain_hires));

        // LoRAs + edit: both providers expose the adapter path.
        let t2i_lora = payload(serde_json::json!({ "loras": [{ "id": "adapter" }] }));
        assert!(mlx(&t2i_lora), "LoRA t2i eligible on MLX");
        assert!(candle(&t2i_lora), "LoRA t2i eligible on candle");

        let edit = payload(serde_json::json!({ "mode": "edit_image", "sourceAssetId": "s" }));
        assert!(mlx(&edit), "edit eligible on MLX");
        assert!(candle(&edit), "edit eligible on candle");

        // Strict pose: both providers assemble the pose branch around the file-loaded DiT.
        let pose = payload(serde_json::json!({ "advanced": { "poses": [{}] } }));
        assert!(mlx(&pose), "pose eligible on MLX (imported pose control)");
        assert!(
            candle(&pose),
            "pose eligible on candle (imported pose control)"
        );

        // Pose + a single reference: the Character-Studio pose-library shape (the reference is the
        // identity-likeness scoring source, the builtin `krea_control_available` semantics).
        let pose_with_reference = payload(serde_json::json!({
            "mode": "character_image",
            "referenceAssetId": "asset-1",
            "advanced": { "poses": [{}] },
        }));
        assert!(
            mlx(&pose_with_reference),
            "pose + likeness reference on MLX"
        );
        assert!(
            candle(&pose_with_reference),
            "pose + likeness reference on candle"
        );

        // Pose + LoRAs: adapters install on the imported DiT under the branch on both providers.
        let pose_with_lora = payload(serde_json::json!({
            "loras": [{ "id": "style" }],
            "advanced": { "poses": [{}] },
        }));
        assert!(mlx(&pose_with_lora), "pose + LoRA on MLX");
        assert!(candle(&pose_with_lora), "pose + LoRA on candle");

        // The imported two-pass helper cannot preserve conditioning yet, so both scheduler lanes
        // fail closed instead of accepting an img2img/edit/pose request and returning t2i output.
        for (label, extra) in [
            (
                "img2img + hires",
                serde_json::json!({
                    "referenceAssetId": "asset-1", "hiresFix": { "enabled": true }
                }),
            ),
            (
                "edit + hires",
                serde_json::json!({
                    "mode": "edit_image", "sourceAssetId": "s",
                    "hiresFix": { "enabled": true }
                }),
            ),
            (
                "pose + hires",
                serde_json::json!({
                    "advanced": { "poses": [{}] }, "hiresFix": { "enabled": true }
                }),
            ),
        ] {
            let p = payload(extra);
            assert!(!mlx(&p) && !candle(&p), "{label} rejected on both");
        }

        // Shapes the pose render loop would silently drop stay rejected on EVERY backend: the
        // plural reference set, a bare source, edit mode, and the base-tier identity shapes.
        for (label, extra) in [
            (
                "pose + reference list",
                serde_json::json!({ "referenceAssetIds": ["a"], "advanced": { "poses": [{}] } }),
            ),
            (
                "pose + source",
                serde_json::json!({ "sourceAssetId": "s", "advanced": { "poses": [{}] } }),
            ),
            (
                "pose + edit mode",
                serde_json::json!({
                    "mode": "edit_image",
                    "sourceAssetId": "s",
                    "advanced": { "poses": [{}] },
                }),
            ),
            (
                "pose + mask",
                serde_json::json!({ "maskAssetId": "m", "advanced": { "poses": [{}] } }),
            ),
            (
                "pose + character",
                serde_json::json!({ "characterId": "c", "advanced": { "poses": [{}] } }),
            ),
            (
                "pose + phases",
                serde_json::json!({ "advanced": { "poses": [{}], "phases": [{}] } }),
            ),
        ] {
            let p = payload(extra);
            assert!(!mlx(&p) && !candle(&p), "{label} rejected on both");
        }
    }

    /// The imported model projection derives its affordances from the exact registered source,
    /// never from a family-wide hardcoded feature flag. Krea transformer files expose the MLX pose
    /// route; the registered SDXL fused checkpoint and Mage directory sources do not.
    #[test]
    fn imported_provider_pose_surface_follows_the_exact_registry() {
        let krea = imported_provider_routes("mlx", "krea_2")
            .filter(|route| route.source == "transformer_file")
            .collect::<Vec<_>>();
        assert!(!krea.is_empty());
        assert!(krea.iter().any(|route| {
            route.operation == "pose" && route.conditioning.iter().any(|kind| kind == "control")
        }));
        assert!(krea.iter().any(|route| route.operation == "edit"));
        assert!(krea
            .iter()
            .any(|route| route.supports_lora || route.supports_lokr));

        for (family, source) in [
            ("sdxl", "fused_checkpoint"),
            ("mage-flow", "transformer_directory"),
        ] {
            let routes = imported_provider_routes("mlx", family)
                .filter(|route| route.source == source)
                .collect::<Vec<_>>();
            assert!(!routes.is_empty(), "{family} exact source is registered");
            assert!(
                routes.iter().all(|route| route.operation != "pose"),
                "{family} exact source must keep the pose picker dark"
            );
        }
    }

    /// sc-15036 (epic 14034 F6) — a full base fine-tune's exact transformer-directory source must
    /// be claimable for plain txt2img and REFUSED for every shape that lane cannot serve.
    ///
    /// This is the routing half of "selectable at generation": before this, a `mage-flow`-family
    /// non-builtin id fell through `image_model_mac_support` to "not available on Mac", so the
    /// Studio hid the model entirely.
    ///
    /// Discriminating in both directions on ONE entry: t2i eligible, and each of edit / reference /
    /// LoRA / pose / mask / character refused — a gate that simply returned `true` for the family
    /// (the generic Krea-shaped arm) would pass the first assertion and fail the LoRA one, because
    /// `mlx_gen_mage::load_finetuned` refuses adapters outright on EVERY backend.
    #[test]
    fn a_fine_tuned_mage_flow_base_is_native_routable_and_claims_txt2img_only() {
        let finetune_id = "finetune_9f3c";
        let payload = |extra: serde_json::Value| {
            let mut value = serde_json::json!({
                "model": finetune_id,
                "modelManifestEntry": {
                    "id": finetune_id,
                    "family": "mage-flow",
                    "importSourceShape": "transformer_directory",
                    "paths": { "model": "/app/models/finetunes/finetune_9f3c" }
                }
            });
            value
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            value.as_object().unwrap().clone()
        };
        let mlx = |p: &serde_json::Map<String, serde_json::Value>| {
            imported_image_request_provider_eligible(finetune_id, p, "mlx")
        };
        let candle = |p: &serde_json::Map<String, serde_json::Value>| {
            imported_image_request_family_eligible(
                finetune_id,
                p,
                CANDLE_ROUTED_FAMILIES,
                CANDLE_IMPORTED_CAPS,
            )
        };

        // The family-only probe stays closed because it cannot select a source. The exact stamped
        // manifest entry is what admits this provider route.
        assert!(
            !image_model_mac_support(finetune_id, Some("mage-flow")).supported,
            "family identity alone must not invent a Mage loader"
        );

        assert!(
            mlx(&payload(serde_json::json!({ "mode": "text_to_image" }))),
            "plain txt2img is the shape the MLX fine-tuned lane serves"
        );
        assert!(
            candle(&payload(serde_json::json!({ "mode": "text_to_image" }))),
            "plain txt2img is the shape the Candle fine-tuned lane serves"
        );

        for (label, extra) in [
            (
                "edit",
                serde_json::json!({ "mode": "edit_image", "sourceAssetId": "s" }),
            ),
            (
                "reference",
                serde_json::json!({ "referenceAssetId": "asset-1" }),
            ),
            (
                "reference list",
                serde_json::json!({ "referenceAssetIds": ["a"] }),
            ),
            ("source", serde_json::json!({ "sourceAssetId": "a" })),
            (
                "lora",
                serde_json::json!({ "loras": [{ "id": "adapter" }] }),
            ),
            ("pose", serde_json::json!({ "advanced": { "poses": [{}] } })),
            (
                "phases",
                serde_json::json!({ "advanced": { "phases": [{}] } }),
            ),
            ("mask", serde_json::json!({ "maskAssetId": "m" })),
            ("character", serde_json::json!({ "characterId": "c" })),
            ("look", serde_json::json!({ "characterLookId": "l" })),
        ] {
            assert!(
                !mlx(&payload(extra.clone())) && !candle(&payload(extra)),
                "{label} must not be flattened to t2i on either native backend"
            );
        }

        // The generated Mage full-fine-tune seam is present on both pinned runtimes.
        assert!(
            CANDLE_ROUTED_FAMILIES.contains(&"mage-flow"),
            "Candle must advertise the generated Mage full-fine-tune family"
        );
        // ...but the family fixture is only an oracle. Production selects an EXACT provider-facts
        // route, and the checked-in candle facts register no `mage-flow` import row, so the
        // provider path stays closed on candle even though the runtime exposes the seam.
        assert!(
            !imported_image_request_provider_eligible(
                finetune_id,
                &payload(serde_json::json!({ "mode": "text_to_image" })),
                "candle"
            ),
            "no candle mage-flow import route is registered, so the exact provider lookup must refuse"
        );

        // A BUILTIN Mage id keeps its id-keyed routing — the family path applies only to
        // non-builtins, so the tiered snapshot lane is untouched.
        assert!(!imported_image_request_provider_eligible(
            "mage_flow_base",
            &payload(serde_json::json!({ "mode": "text_to_image" })),
            "mlx"
        ));
    }

    #[test]
    fn imported_sdxl_family_gate_claims_t2i_on_both_backends_without_dropping_conditioning() {
        let imported_id = "community_xl";
        let payload = |extra: serde_json::Value| {
            let mut value = serde_json::json!({
                "model": imported_id,
                "modelManifestEntry": {
                    "id": imported_id,
                    "family": "sdxl",
                    "paths": { "model": "/app/models/imports/community-xl" }
                }
            });
            value
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            value.as_object().unwrap().clone()
        };
        let eligible = |p: &serde_json::Map<String, serde_json::Value>| {
            (
                imported_image_request_family_eligible(
                    imported_id,
                    p,
                    MLX_ROUTED_FAMILIES,
                    MLX_IMPORTED_CAPS,
                ),
                imported_image_request_family_eligible(
                    imported_id,
                    p,
                    CANDLE_ROUTED_FAMILIES,
                    CANDLE_IMPORTED_CAPS,
                ),
            )
        };

        assert_eq!(eligible(&payload(serde_json::json!({}))), (true, true));
        assert_eq!(
            eligible(&payload(
                serde_json::json!({ "loras": [{ "id": "style" }] })
            )),
            (true, true),
            "both fused SDXL loaders accept UNet adapters"
        );
        for conditioned in [
            payload(serde_json::json!({ "referenceAssetId": "asset-1" })),
            payload(serde_json::json!({ "mode": "edit_image", "sourceAssetId": "asset-1" })),
            payload(serde_json::json!({ "advanced": { "poses": [{}] } })),
        ] {
            assert_eq!(eligible(&conditioned), (false, false));
        }
    }

    /// The `models` array of the SHIPPED manifest — the exact bytes the app embeds and seeds, and
    /// the exact bytes the web LoRA picker reads `loraCompatibility` out of.
    fn builtin_image_models() -> Vec<serde_json::Value> {
        let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc present");
        let manifest: serde_json::Value =
            serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw)).expect("parses as JSON");
        manifest
            .get("models")
            .and_then(serde_json::Value::as_array)
            .expect("models array")
            .iter()
            .filter(|m| m.get("type").and_then(serde_json::Value::as_str) == Some("image"))
            .cloned()
            .collect()
    }

    /// The IMPORTED-side half of the class guard below, which reads `builtin.models.jsonc` and so
    /// can only ever see BUILTIN rows. A non-builtin (imported / fine-tuned) entry has no manifest
    /// row to read: its `loraCompatibility` is SYNTHESIZED from the family token by
    /// `lora_family::apply_model_manifest_defaults`, which is why the class escaped that guard
    /// entirely and shipped three separate hangs.
    ///
    /// This pins the real matrix [`imported_image_model_lora_advertisement`] computes off the gate,
    /// so a gate change that silently flips a family's adapter verdict lands here rather than in a
    /// queue that never drains. `mlx` = a macOS deployment (MLX lane only), `candle` = Windows/Linux
    /// (candle lane only) — the two shipped topologies.
    #[test]
    fn imported_lora_advertisement_tracks_which_lane_can_claim_an_adapter() {
        let source = |family: &str| match family {
            "krea_2" => "transformer_file",
            "sdxl" => "fused_checkpoint",
            "mage-flow" => "transformer_directory",
            _ => "transformer_file",
        };
        let mlx = |family: &str| {
            imported_image_model_lora_advertisement(
                "user_import",
                family,
                source(family),
                true,
                false,
            )
        };
        let candle = |family: &str| {
            imported_image_model_lora_advertisement(
                "user_import",
                family,
                source(family),
                false,
                true,
            )
        };

        // krea_2 — THE REPORTED BUG. The MLX single-file entrypoint takes adapters (inference #211);
        // the candle provider now does too, so both lanes must advertise truthfully.
        assert_eq!(mlx("krea_2"), Some(true));
        assert_eq!(
            candle("krea_2"),
            Some(true),
            "an imported Krea 2 checkpoint advertises the adapter lane Candle can claim"
        );

        // sdxl — a fused checkpoint; both native loaders accept UNet adapters, so the
        // advertisement is honest on both lanes and must be left alone.
        assert_eq!(mlx("sdxl"), Some(true));
        assert_eq!(candle("sdxl"), Some(true));

        // mage-flow — the two answers differ, and the difference is the point. `Some(false)` means
        // "this lane serves the family and refuses adapters"; `None` means "this backend is not on
        // the seam at all, so it has no opinion to advertise". MLX declares a `mage-flow` imported
        // provider whose seam rejects adapters, so it answers `Some(false)`. Candle declares no
        // `mage-flow` imported provider in its engine facts, so the base claim never lands and the
        // honest answer is `None` — advertising `Some(false)` there would imply a lane exists.
        assert_eq!(
            mlx("mage-flow"),
            Some(false),
            "a Mage fine-tune renders t2i on MLX but refuses adapters"
        );
        assert_eq!(
            candle("mage-flow"),
            None,
            "candle declares no mage-flow imported provider, so it advertises no opinion"
        );

        // A family the route-by-family path does not serve at all: no opinion, entry untouched.
        assert_eq!(mlx("flux"), None);
        assert_eq!(candle("z-image"), None);

        assert_eq!(
            imported_image_model_lora_advertisement(
                "wrong_shape",
                "krea_2",
                "fused_checkpoint",
                true,
                true,
            ),
            None,
            "a sibling source shape must not inherit Krea transformer-file adapters"
        );

        // A BUILTIN id keeps its id-keyed routing and is never touched by this oracle, whatever
        // family token it carries — otherwise the projection would rewrite shipped manifest rows.
        assert_eq!(
            imported_image_model_lora_advertisement(
                "krea_2_turbo",
                "krea_2",
                "transformer_file",
                true,
                false,
            ),
            None,
            "builtins route by id; the advertisement oracle must have no opinion on them"
        );
    }

    /// **THE CLASS GUARD (sc-15328).** If a model advertises LoRA compatibility, then some backend
    /// must actually be able to CLAIM a render that carries one.
    ///
    /// This is the sixth "shipped but unreachable" defect in epic 14034, and every one of them
    /// passed its own tests, because each layer was correct in isolation. sc-15328 itself: the
    /// manifest advertised `loraCompatibility.families = ["mage-flow"]` on the three Mage generation
    /// variants, the picker therefore offered a trained Mage adapter as "installed and compatible",
    /// `mage_flow_mlx_eligible` refused any payload with a non-empty `loras`, and every Mage
    /// `ModelCaps` row is `candle_lora: false`. No lane claimed the job. It did not fail — it sat
    /// `queued` / "Waiting for an available worker." **forever**, next to an idle mlx worker, with
    /// no error and no terminal state. That is strictly worse than a rejection.
    ///
    /// **Derived from the real tables on both sides**, never a restated list:
    ///
    ///   * the ADVERTISEMENT is read out of the shipped `builtin.models.jsonc` bytes — the same
    ///     `loraCompatibility.families` key `presetUtils.js`'s `modelLoraFamilies` reads to decide
    ///     what the picker offers;
    ///   * the CLAIM is the real routing predicates, `image_request_mlx_eligible` and
    ///     `image_request_candle_eligible`, which are what `worker_supports_job` consults.
    ///
    /// So a NEW model advertising adapters is covered the moment its manifest row lands — nobody has
    /// to remember to extend a list here. This is the LoRA-carrying sibling of
    /// `every_mlx_routed_model_has_a_dispatch_arm` (sc-10523's stranded-Anima guard), and it probes
    /// the same three conditioning shapes so that an edit-only or reference-only model is judged on
    /// a shape it actually serves.
    ///
    /// The second assertion is what keeps the first honest: every advertised model must be claimable
    /// on some probe WITHOUT a LoRA too. Without it, a model that no lane could claim under any
    /// shape would make its half of the first assertion vacuous — the probes would be wrong rather
    /// than the routing, and the test would be measuring nothing.
    #[test]
    fn every_lora_advertising_image_model_is_claimable_with_a_lora() {
        // The three conditioning shapes, each in a with-LoRA and a without-LoRA form. Anything a
        // model gates on beyond these is out of scope for what this guard can decide.
        let probes = |model: &str,
                      with_lora: bool|
         -> Vec<serde_json::Map<String, serde_json::Value>> {
            [
                serde_json::json!({ "model": model, "mode": "text_to_image" }),
                serde_json::json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset-1" }),
                serde_json::json!({ "model": model, "mode": "character_image", "referenceAssetId": "asset-1" }),
            ]
            .into_iter()
            .map(|shape| {
                let mut payload = shape.as_object().expect("probe is an object").clone();
                if with_lora {
                    payload.insert(
                        "loras".to_owned(),
                        serde_json::json!([{ "id": "adapter-1", "weight": 0.8 }]),
                    );
                }
                payload
            })
            .collect()
        };
        let claimable = |model: &str, payload: &serde_json::Map<String, serde_json::Value>| {
            super::super::mlx::image_request_mlx_eligible(model, payload)
                || super::super::candle::image_request_candle_eligible(model, payload)
        };

        // Every image model whose SHIPPED manifest row offers adapters to the picker.
        let advertised: Vec<String> = builtin_image_models()
            .iter()
            .filter(|model| {
                model
                    .get("loraCompatibility")
                    .and_then(|lora| lora.get("families"))
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|families| !families.is_empty())
            })
            .filter_map(|model| {
                model
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        assert!(
            !advertised.is_empty(),
            "no image model advertises `loraCompatibility` — the manifest key was renamed or the \
             filter is wrong, and this guard is asserting nothing"
        );

        // Anti-vacuity: the probe shapes must be right for every advertised model, so that a
        // failure below is about the LoRA and not about the probe.
        let unroutable: Vec<&String> = advertised
            .iter()
            .filter(|model| {
                !probes(model, false)
                    .iter()
                    .any(|payload| claimable(model, payload))
            })
            .collect();
        assert!(
            unroutable.is_empty(),
            "these models are unclaimable by every backend even WITHOUT a LoRA, so the with-LoRA \
             assertion below would be vacuous for them — fix the probe shapes or the routing: \
             {unroutable:?}"
        );

        let stranded: Vec<&String> = advertised
            .iter()
            .filter(|model| {
                !probes(model, true)
                    .iter()
                    .any(|payload| claimable(model, payload))
            })
            .collect();
        assert!(
            stranded.is_empty(),
            "these image models advertise `loraCompatibility` in builtin.models.jsonc — so the \
             picker offers adapters for them as \"installed and compatible\" — but NO backend will \
             claim a render carrying one (neither `image_request_mlx_eligible` nor \
             `image_request_candle_eligible`). A user who attaches an adapter gets a job that \
             queues FOREVER with no error and an idle worker (sc-15328). Either make a lane claim \
             it, or stop advertising `loraCompatibility` on the model: {stranded:?}"
        );
    }
}
