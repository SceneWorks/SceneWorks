//! Model/kernel routing catalog: the per-backend routed-model and training-kernel lists,
//! the Mac support/capability probes, and their supporting types. Moved out of
//! `jobs_store.rs` (sc-8816) with no behavior change. List membership is pinned by the
//! snapshot tests at the bottom of this file.

use std::collections::BTreeMap;

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
/// is purely id-keyed). It is consulted only for the image route-by-family path (sc-14019): a
/// non-builtin (imported/user) image model whose family is MLX-routed is Mac-routable even though
/// its novel id is in no routing table. Builtin routing is unaffected (a builtin short-circuits on
/// its id).
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

pub(crate) fn image_model_mac_support(model: &str, family: Option<&str>) -> ModelMacSupport {
    if !MLX_ROUTED_MODELS.contains(&model) {
        // Route-by-family for non-builtin (imported/user) models (sc-14019, epic 14015): a model
        // whose id is in no routing table but whose declared `family` is an MLX-routed family reuses
        // that family's existing in-process engine, so it is Mac-routable rather than hidden behind
        // "Not available on Mac (MLX only)". Guarded on `!is_builtin_image_model` so builtin id-keyed
        // routing is never altered — a builtin always short-circuits on `MLX_ROUTED_MODELS` above (or,
        // if some future builtin is candle-only, keeps its id-keyed not-supported verdict here).
        if !is_builtin_image_model(model) && family.is_some_and(image_family_is_mlx_routed) {
            // Probe the SHARED imported-family claim gate with a synthetic manifest entry (the
            // declared family + a placeholder install path), so the pose affordance below can never
            // drift from what the scheduler actually admits — the same one-dispatch-table
            // discipline the builtin probes at the bottom of this function follow.
            let imported_probe = |extra: &[(&str, Value)]| {
                let mut payload = probe_payload(model, extra);
                payload.insert(
                    "modelManifestEntry".to_owned(),
                    json!({ "family": family, "paths": { "model": "probe" } }),
                );
                payload
            };
            // Strict pose: admitted for the `krea_2` family on MLX (the native control entrypoint
            // assembles the pose branch around the file-loaded DiT), refused for every other
            // imported family — derived from the gate itself, mirroring the builtin pose probe's
            // two shapes (a plain pose set, and a Character-Studio pose set with a reference).
            let pose = imported_image_request_family_eligible(
                model,
                &imported_probe(&[("advanced", json!({ "poses": [{}] }))]),
                MLX_ROUTED_FAMILIES,
                MLX_IMPORTED_CAPS,
            ) || imported_image_request_family_eligible(
                model,
                &imported_probe(&[
                    ("mode", json!("character_image")),
                    ("referenceAssetId", json!("probe")),
                    ("advanced", json!({ "poses": [{}] })),
                ]),
                MLX_ROUTED_FAMILIES,
                MLX_IMPORTED_CAPS,
            );
            return ModelMacSupport {
                supported: true,
                reason: None,
                // The imported single-file checkpoint is a bare diffusion transformer paired with a
                // resident base tier. On MLX its native loader takes adapters (inference #211), so the
                // lane serves job LoRAs (`lycoris`, sc-14111) AND the Kontext edit surface (`edit`,
                // sc-14119). `pose` lights per-family from the claim-gate probe above (the Krea pose
                // control branch composes with a file-loaded same-shape DiT); `reference`
                // (IP-Adapter / character identity) stays false — identity conditioning needs
                // base-tier components this lane does not stage; img2img is surfaced separately via
                // the `ui.img2img` manifest flag, not `reference`.
                features: ModelMacFeatures {
                    pose,
                    edit: true,
                    lycoris: true,
                    ..ModelMacFeatures::default()
                },
            };
        }
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
        // MLX worker. There is no fallback (mac-only), so this feature is
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
    // in `CANDLE_QUANT_MODELS`. NOT `candle_quant_lora`: neither advertises candle inference LoRA.
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
    // InstantID on RealVisXL (sc-3345): MLX-only id — single-identity + the 11-view angle set route to
    // the native `mlx-gen-instantid` provider (candle serves it via the bespoke `instantid_candle_eligible`
    // lane, not the txt2img gate, so it is NOT a candle-routed txt2img id).
    ModelCaps::new("instantid_realvisxl", true, false, false, false, false),
    // PuLID-FLUX on FLUX.1-dev (sc-3344): MLX-only id — `character_image` with a reference face (candle
    // serves it via the bespoke `pulid_flux_candle_eligible` lane, not the txt2img gate).
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
    // Microsoft Lens / Lens-Turbo (epic 3164 / sc-5105 MLX; sc-5126 candle): pure T2I family. UNLIKE the
    // other candle families it DOES advertise on-the-fly quant AND LoRA/LoKr, so `candle_quant_lora` is
    // set — the first (and, with SD3.5/Krea, one of the) candle families exempt from quant/LoRA
    // refusal. Lens was the LAST whole-model torch-only image family — once it routed, the per-model
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
    // pure txt2img. Candle advertises Q4/Q8 (sc-7879) and applies LoRA/LoKr on every tier.
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
    // LTX-2.3 base + eros (sc-3035 MLX; sc-5097 / sc-5495 candle): txt2video on both backends.
    VideoModelCaps::new("ltx_2_3", true, true, false, false),
    VideoModelCaps::new("ltx_2_3_eros", true, true, false, false),
    // Wan2.2 TI2V-5B (sc-3034 MLX; sc-5097 candle): txt2video + VACE advanced modes on candle.
    VideoModelCaps::new("wan_2_2", true, true, false, true),
    // Wan2.2 14B MoE (sc-5175): T2V-14B is text-only; I2V-14B is image→video ONLY (candle_video_i2v).
    // Both are VACE-capable on candle.
    VideoModelCaps::new("wan_2_2_t2v_14b", true, true, false, true),
    VideoModelCaps::new("wan_2_2_i2v_14b", true, true, true, true),
    // Wan2.2 VACE-Fun A14B (sc-3459 / sc-6604): the shipped dual-expert model is a dedicated
    // MLX replace-person engine. The worker resolves it before the generic Wan-VACE fallback and
    // fails loudly off-Mac, so only the MLX column is true; its sole admitted mode is pinned by
    // `video_mode_is_mlx_eligible`.
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
    // model — the same all-false candle shape `scail2_14b` carries, and for the same reason.
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

/// Architecture families whose MLX-routed builtins define an in-process image engine that a
/// **non-builtin (imported/user) same-family model reuses** (sc-14019, epic 14015). A builtin's
/// routing is decided by its id ([`MLX_ROUTED_MODELS`]); an imported checkpoint carries a novel id
/// that is in no table, so [`image_model_mac_support`] falls back to its declared *family* — if the
/// family is one of these, it routes to that family's existing MLX engine. Seeded with `krea_2`
/// (its builtins `krea_2_turbo` / `krea_2_raw` already route to the Krea MLX engine). This is an
/// explicit allow-list, not derived from the id table, because the routing catalog is keyed on ids
/// alone and has no id→family map; keep it aligned with the same-family builtins above and with the
/// import gate (`sceneworks_core::base_weights::IMPORT_SUPPORTED_FAMILIES`). Pinned by the
/// `EXPECTED_MLX_ROUTED_FAMILIES` snapshot test.
///
/// `mage-flow` (sc-15036, epic 14034 F6) joins for a different reason than the other two: its
/// non-builtin ids are not community imports but this app's OWN full base fine-tunes — the
/// `transformer/`-shaped checkpoint a `networkType: "full"` training run produces, registered into
/// the model catalog and loaded by `image_jobs::mage_finetuned` against the installed base's shared
/// text encoder + VAE. It is deliberately absent from [`CANDLE_ROUTED_FAMILIES`]: the Mage engine is
/// MLX-only (`mac_only: true`), and there is no candle Mage generator to route a fine-tune to.
/// The token is the MANIFEST family spelling (`mage-flow`), matching the builtin entries — not the
/// underscored id prefix.
pub(crate) const MLX_ROUTED_FAMILIES: &[&str] = &["krea_2", "mage-flow", "sdxl"];

/// Whether an image `family` string routes to an in-process MLX engine via the route-by-family path
/// (sc-14019) — i.e. it is a member of [`MLX_ROUTED_FAMILIES`]. Consulted only for non-builtin ids.
pub(crate) fn image_family_is_mlx_routed(family: &str) -> bool {
    MLX_ROUTED_FAMILIES.contains(&family)
}

/// Architecture families whose existing Candle engine can serve a non-builtin imported/user
/// same-family model. This is deliberately separate from [`CANDLE_ROUTED_MODELS`]: imported
/// checkpoints have novel ids and are claimed only after the scheduler validates their manifest
/// family and supported request shape. Seeded by the descriptor-gated Krea single-file lane
/// (sc-14023).
pub(crate) const CANDLE_ROUTED_FAMILIES: &[&str] = &["krea_2", "sdxl"];

/// Whether `id` names a builtin image model (a row in [`IMAGE_MODEL_CAPS`]). The route-by-family
/// path applies only to non-builtin (imported/user) ids, so a builtin's id-keyed routing is never
/// overridden by its family (sc-14019).
pub(crate) fn is_builtin_image_model(id: &str) -> bool {
    IMAGE_MODEL_CAPS.iter().any(|caps| caps.id == id)
}

/// Per-backend capabilities of the native single-file (imported) lane, the axis
/// [`imported_image_request_family_eligible`] keys request-shape admission on. One named struct —
/// not positional bools — so the mlx.rs / candle.rs call sites and the worker's mirrored
/// `KREA_IMPORTED_SUPPORTS_*` constants cannot silently transpose capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImportedImageBackendCaps {
    /// The backend's native single-file loader takes an `adapters` slice: the MLX entrypoint
    /// (`mlx_gen_krea::load_from_native_dit_file`, inference #211) does, so it serves LoRAs
    /// (sc-14111) AND the Kontext edit surface (sc-14119, whose required `krea2_identity_edit`
    /// LoRA IS an adapter); the candle entrypoint takes no adapters yet (sc-14135), so the candle
    /// imported lane stays **t2i / img2img only**.
    pub(crate) adapters: bool,
    /// The backend can assemble the Krea pose ControlNet branch around a FILE-LOADED DiT: the MLX
    /// runtime's `load_control_from_native_dit_file` folds the trained `Krea2ControlBranch` (a
    /// `control_scale`-scaled residual, architecturally independent of the DiT's weights) onto the
    /// imported transformer, so a same-shape fine-tune serves strict-pose sets; the candle
    /// single-file path has no control parameter, so candle keeps rejecting pose.
    pub(crate) pose_control: bool,
}

/// The MLX backend's imported-lane capabilities (mlx.rs / the Mac worker's macOS constants).
pub(crate) const MLX_IMPORTED_CAPS: ImportedImageBackendCaps = ImportedImageBackendCaps {
    adapters: true,
    pose_control: true,
};

/// The candle backend's imported-lane capabilities (candle.rs / the worker's candle constants):
/// no adapters (sc-14135), no pose control — its native single-file entrypoint threads neither.
pub(crate) const CANDLE_IMPORTED_CAPS: ImportedImageBackendCaps = ImportedImageBackendCaps {
    adapters: false,
    pose_control: false,
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
    // LoRAs ride the native single-file loader's adapter path — MLX only today (sc-14135 is the
    // candle follow-up), so reject them on a backend without adapter support.
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
/// the fields [`imported_image_request_family_eligible`] reads: the manifest entry (family + a
/// non-empty installed path) and, when `with_lora`, a single job LoRA. Everything else is absent, so
/// the probe is the PLAIN t2i shape — the surface every imported lane claims — plus/minus adapters.
fn imported_image_lora_probe(model: &str, family: &str, with_lora: bool) -> Map<String, Value> {
    let mut payload = json!({
        "model": model,
        "modelManifestEntry": {
            "id": model,
            "family": family,
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
/// [`imported_image_request_family_eligible`] (sc-14135 follow-up; the class sc-15328 named).
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
/// per-lane capability surfaces the two routers pass ([`MLX_IMPORTED_CAPS`] from `mlx.rs` /
/// [`CANDLE_IMPORTED_CAPS`] from `candle.rs`). Restating the per-family verdict as its own table
/// here is precisely how the advertisement and the gate drift apart again, so it is computed,
/// never copied.
pub fn imported_image_model_lora_advertisement(
    model: &str,
    family: &str,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) -> Option<bool> {
    let claims = |with_lora: bool| {
        let payload = imported_image_lora_probe(model, family, with_lora);
        (mlx_lane_available
            && imported_image_request_family_eligible(
                model,
                &payload,
                MLX_ROUTED_FAMILIES,
                MLX_IMPORTED_CAPS,
            ))
            || (candle_lane_available
                && imported_image_request_family_eligible(
                    model,
                    &payload,
                    CANDLE_ROUTED_FAMILIES,
                    CANDLE_IMPORTED_CAPS,
                ))
    };
    if !claims(false) {
        return None;
    }
    Some(claims(true))
}

derive_model_list! {
    /// The models the candle (Windows/CUDA) lane can serve for base txt2img (derived from
    /// [`IMAGE_MODEL_CAPS`]`.candle_routed`, sc-9495). Mirrors the worker's `image_jobs::is_candle_engine`.
    /// Deliberately narrow: candle is a gated txt2img-only lane, so every conditioning shape is
    /// refused unless a bespoke candle lane in `image_job_is_candle_eligible` claims it.
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
    /// The candle image families that advertise on-the-fly Q4/Q8 quant AND LoRA/LoKr adapters — Lens /
    /// Lens-Turbo and Krea 2 Turbo (derived from [`IMAGE_MODEL_CAPS`]`.candle_quant_lora`, sc-9495; Krea
    /// added sc-9983 once sc-9607 flipped its `supported_quants`). For these a LoRA or an explicit quant
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
/// has no off-Mac trainer). `krea_lora` is the native `mlx-gen-krea` trainer
/// (sc-7577); it is native-only, so it is also listed in `MLX_ONLY_TRAINING_KERNELS`
/// (sc-7578). `sd3_lora` is the native `mlx-gen-sd3` trainer (sc-7883/7885; Large +
/// MMDiT-X Medium training bases), cut over here in sc-7884; it is native-only too,
/// so it is also in `MLX_ONLY_TRAINING_KERNELS`. A kernel absent here is never routed to
/// the mlx worker.
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
    // conditioner). It is native-only, so it is also in `MLX_ONLY_TRAINING_KERNELS`.
    "anima_lora",
    // Mage-Flow (epic 14034, sc-14055 LoRA/LoKr + sc-14056 full base fine-tune): the native
    // `mlx-gen-mage` trainer. It is native-only, so it is ALSO in `MLX_ONLY_TRAINING_KERNELS` — and
    // that pairing is exactly why this entry is load-bearing: `MLX_ONLY_TRAINING_KERNELS` makes every
    // generic/candle worker refuse the job, so until the kernel appears HERE no worker claims it at
    // all and a Mage training job queues forever (sc-14056). The kernel serves both the LoRA path and
    // the `networkType: "full"` base fine-tune; the full path is additionally admission-gated by the
    // worker's unified-memory pre-flight.
    "mage_flow_lora",
];

/// SceneWorks training kernels with a native candle trainer that needs no base-model disambiguation
/// (sc-7817, epic 5164) — the off-Mac twin of [`MLX_ROUTED_TRAINING_KERNELS`]. The candle registry
/// holds trainers for `sdxl`, `z_image_turbo`, `lens`, `krea_2_raw` (the Krea 2 Raw 12B DiT, epic
/// 7565 P4 — sc-8614 wires it here), and the Wan **A14B T2V** MoE (`wan2_2_t2v_14b`); the first four
/// map straight from kernel, while `wan_moe_lora` is base-model gated (handled in
/// [`training_job_is_candle_eligible`]). Krea is in BOTH this set and
/// [`MLX_ONLY_TRAINING_KERNELS`] (Rust-only: mlx OR candle). The dense Wan 5B + the I2V A14B have no
/// candle trainer yet (sc-5167 follow-ups), and Kolors has none; unsupported kernels remain queued
/// off-Mac. LTX uses its historical `ltx_mlx_lora` kernel name on both native backends.
/// `krea_control` (epic 10159, B2 sc-10163 / B1 sc-10162) is the Krea 2 pose-ControlNet branch trainer
/// (candle-gen-krea `ControlTrainer`, dispatched under `krea_2_control`); it has no MLX trainer, so —
/// like `krea_lora` — it is also in [`MLX_ONLY_TRAINING_KERNELS`] but NOT
/// [`MLX_ROUTED_TRAINING_KERNELS`] (the MLX control lane is B5/sc-10177).
pub(crate) const CANDLE_ROUTED_TRAINING_KERNELS: &[&str] = &[
    "z_image_lora",
    "sdxl_lora",
    "lens_lora",
    "krea_lora",
    "krea_control",
    "ltx_mlx_lora",
];

/// Native-only training kernels — only a Rust worker can run them, so a generic worker descriptor
/// must refuse the job (leaving it queued for a Rust worker) rather than claim it and fail with "no
/// training kernel". Despite the constant's historical name, this means native-Rust-only rather than
/// MLX-backend-only: LTX and Krea have both MLX and candle trainers. Their candle workers are admitted
/// by the [`worker_supports_job`] exception when the kernel is also in
/// [`CANDLE_ROUTED_TRAINING_KERNELS`], while a generic worker remains refused. `sd3_lora` (epic 7841 T3
/// sc-7884) is MLX-native with no candle trainer yet (the off-Mac/candle SD3.5 trainer is epic
/// 7982), so only an mlx worker runs it today. `mage_flow_lora` was registered as a
/// training-target contract in sc-14054 ahead of its native trainer; the `mlx-gen-mage` trainer
/// landed in sc-14055 (LoRA/LoKr) and sc-14056 (full base fine-tune), and sc-14056 added the kernel
/// to [`MLX_ROUTED_TRAINING_KERNELS`] so an mlx worker finally claims it. It stays here because there
/// is no candle Mage trainer — a generic/candle worker must leave the job queued rather than claim it
/// and fail with "no training kernel".
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
        image_family_is_mlx_routed, image_model_mac_support,
        imported_image_model_lora_advertisement, imported_image_request_family_eligible,
        is_builtin_image_model, CANDLE_IMPORTED_CAPS, CANDLE_LORA_MODELS, CANDLE_QUANT_LORA_MODELS,
        CANDLE_QUANT_MODELS, CANDLE_ROUTED_FAMILIES, CANDLE_ROUTED_MODELS,
        CANDLE_ROUTED_TRAINING_KERNELS, CANDLE_VIDEO_I2V_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS,
        CANDLE_VIDEO_VACE_MODELS, IMAGE_MODEL_CAPS, MLX_IMPORTED_CAPS, MLX_ONLY_TRAINING_KERNELS,
        MLX_ROUTED_FAMILIES, MLX_ROUTED_MODELS, MLX_ROUTED_TRAINING_KERNELS,
        VIDEO_MLX_ROUTED_MODELS, VIDEO_MODEL_CAPS,
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
        "ltx_2_3_eros",
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
        "lens_lora",
        "krea_lora",
        "krea_control",
        "ltx_mlx_lora",
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
    const EXPECTED_CANDLE_ROUTED_FAMILIES: &[&str] = &["krea_2", "sdxl"];

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
    fn imported_same_family_model_routes_via_family_path() {
        // An imported model id that is in NO routing table, declaring family krea_2, resolves as
        // Mac-routable (supported) via the family path — not hidden behind "not available on Mac".
        let imported_id = "user_kreamania_variant5"; // a novel id, never a builtin
        assert!(!is_builtin_image_model(imported_id));
        assert!(!MLX_ROUTED_MODELS.contains(&imported_id));

        let support = image_model_mac_support(imported_id, Some("krea_2"));
        assert!(
            support.supported,
            "an imported krea_2-family model should route via the family path"
        );
        assert!(support.reason.is_none());
        // The MLX imported lane serves job LoRAs (sc-14111) and the Kontext edit surface (sc-14119)
        // via the native single-file loader's adapter path, plus strict-pose sets via the native
        // control entrypoint (the trained pose branch folds onto the file-loaded DiT), so
        // `lycoris` + `edit` + `pose` are advertised. IP-Adapter/character `reference` stays
        // unclaimed (identity conditioning needs base-tier components this lane does not stage;
        // img2img is a separate `ui.img2img` flag, not `reference`).
        assert!(support.features.lycoris);
        assert!(support.features.edit);
        assert!(support.features.pose);
        assert!(!support.features.reference);
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

    /// The shared imported-family gate is backend-capability-aware (sc-14111 / sc-14119): LoRAs and
    /// the Kontext edit surface are admitted ONLY when the backend's native single-file loader takes
    /// adapters (`adapters_supported = true`, MLX / #211); with `false` (candle, until sc-14135) they
    /// are rejected so the candle imported lane stays t2i / img2img only. img2img (a non-edit single
    /// `referenceAssetId`) and plain t2i need no adapter, so they are admitted on BOTH.
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

        // LoRAs + edit: MLX only (the adapter path), rejected on candle.
        let t2i_lora = payload(serde_json::json!({ "loras": [{ "id": "adapter" }] }));
        assert!(mlx(&t2i_lora), "LoRA t2i eligible on MLX");
        assert!(
            !candle(&t2i_lora),
            "LoRA t2i NOT eligible on candle (sc-14135)"
        );

        let edit = payload(serde_json::json!({ "mode": "edit_image", "sourceAssetId": "s" }));
        assert!(mlx(&edit), "edit eligible on MLX");
        assert!(!candle(&edit), "edit NOT eligible on candle (sc-14135)");

        // Strict pose: the imported pose-control surface — MLX assembles the pose branch around the
        // file-loaded DiT (`load_control_from_native_dit_file`), candle has no control parameter on
        // its single-file path, so the pose set stays candle-rejected (a candle-required deployment
        // fails it terminally via the `candle_unsupported` sweep rather than stranding it queued).
        let pose = payload(serde_json::json!({ "advanced": { "poses": [{}] } }));
        assert!(mlx(&pose), "pose eligible on MLX (imported pose control)");
        assert!(!candle(&pose), "pose NOT eligible on candle");

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
        assert!(!candle(&pose_with_reference), "still candle-rejected");

        // Pose + LoRAs: adapters install on the imported DiT under the branch (MLX only).
        let pose_with_lora = payload(serde_json::json!({
            "loras": [{ "id": "style" }],
            "advanced": { "poses": [{}] },
        }));
        assert!(mlx(&pose_with_lora), "pose + LoRA on MLX");
        assert!(!candle(&pose_with_lora), "pose + LoRA candle-rejected");

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

    /// The imported model's Mac-support badge derives its pose affordance from the SAME claim gate
    /// the scheduler runs (never a hardcoded feature flag): a `krea_2`-family import lights the pose
    /// picker (the MLX native control entrypoint assembles the branch around the file-loaded DiT);
    /// an `sdxl`- or `mage-flow`-family import keeps it dark (no control lane composes with those).
    #[test]
    fn imported_family_badge_pose_follows_the_claim_gate() {
        let krea = image_model_mac_support("user_kreamania_variant5", Some("krea_2"));
        assert!(krea.supported);
        assert!(krea.features.pose, "krea_2 import lights the pose picker");
        assert!(krea.features.edit && krea.features.lycoris);

        for family in ["sdxl", "mage-flow"] {
            let support = image_model_mac_support("user_other_import", Some(family));
            assert!(support.supported);
            assert!(
                !support.features.pose,
                "{family} import must keep the pose picker dark"
            );
        }
    }

    /// sc-15036 (epic 14034 F6) — a full base fine-tune's catalog entry must be Mac-routable and
    /// claimable for plain txt2img, and REFUSED for every shape the fine-tuned lane cannot serve.
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
    fn a_fine_tuned_mage_flow_base_is_mac_routable_and_claims_txt2img_only() {
        let finetune_id = "finetune_9f3c";
        let payload = |extra: serde_json::Value| {
            let mut value = serde_json::json!({
                "model": finetune_id,
                "modelManifestEntry": {
                    "id": finetune_id,
                    "family": "mage-flow",
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
            imported_image_request_family_eligible(
                finetune_id,
                p,
                MLX_ROUTED_FAMILIES,
                MLX_IMPORTED_CAPS,
            )
        };

        // The novel id is in NO routing table; the family path is what makes it Mac-routable.
        assert!(
            image_model_mac_support(finetune_id, Some("mage-flow")).supported,
            "a fine-tuned Mage base must not be hidden behind \"not available on Mac\""
        );
        assert!(
            !image_model_mac_support(finetune_id, None).supported,
            "...and it is the DECLARED FAMILY that admits it, not the id"
        );

        assert!(
            mlx(&payload(serde_json::json!({ "mode": "text_to_image" }))),
            "plain txt2img is the shape the fine-tuned lane serves"
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
                !mlx(&payload(extra)),
                "{label} must keep flowing to its established route, not be flattened to t2i here"
            );
        }

        // Mage is MLX-only (`mac_only` descriptor, no candle Mage engine), so the candle family
        // list must NOT admit it — otherwise a candle host would claim a job it cannot load.
        assert!(
            !CANDLE_ROUTED_FAMILIES.contains(&"mage-flow"),
            "there is no candle Mage engine to route a fine-tune to"
        );
        assert!(!imported_image_request_family_eligible(
            finetune_id,
            &payload(serde_json::json!({ "mode": "text_to_image" })),
            CANDLE_ROUTED_FAMILIES,
            CANDLE_IMPORTED_CAPS
        ));

        // A BUILTIN Mage id keeps its id-keyed routing — the family path applies only to
        // non-builtins, so the tiered snapshot lane is untouched.
        assert!(!imported_image_request_family_eligible(
            "mage_flow_base",
            &payload(serde_json::json!({ "mode": "text_to_image" })),
            MLX_ROUTED_FAMILIES,
            MLX_IMPORTED_CAPS
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
        let mlx = |family: &str| {
            imported_image_model_lora_advertisement("user_import", family, true, false)
        };
        let candle = |family: &str| {
            imported_image_model_lora_advertisement("user_import", family, false, true)
        };

        // krea_2 — THE REPORTED BUG. The MLX single-file entrypoint takes adapters (inference #211);
        // the candle one does not yet (sc-14135), so a candle-only host must not advertise them.
        assert_eq!(mlx("krea_2"), Some(true));
        assert_eq!(
            candle("krea_2"),
            Some(false),
            "an imported Krea 2 checkpoint renders t2i on candle but CANNOT take a LoRA — \
             advertising one is the hang this oracle exists to prevent"
        );

        // sdxl — a fused checkpoint; both native loaders accept UNet adapters, so the
        // advertisement is honest on both lanes and must be left alone.
        assert_eq!(mlx("sdxl"), Some(true));
        assert_eq!(candle("sdxl"), Some(true));

        // mage-flow — adapters refused on EVERY backend (`mlx_gen_mage::load_finetuned`), and there
        // is no candle Mage engine at all, so it is not even routable there.
        assert_eq!(
            mlx("mage-flow"),
            Some(false),
            "a Mage fine-tune renders t2i on MLX but refuses adapters on every backend"
        );
        assert_eq!(
            candle("mage-flow"),
            None,
            "mage-flow is absent from CANDLE_ROUTED_FAMILIES — not routable, so no opinion"
        );

        // A family the route-by-family path does not serve at all: no opinion, entry untouched.
        assert_eq!(mlx("flux"), None);
        assert_eq!(candle("z-image"), None);

        // A BUILTIN id keeps its id-keyed routing and is never touched by this oracle, whatever
        // family token it carries — otherwise the projection would rewrite shipped manifest rows.
        assert_eq!(
            imported_image_model_lora_advertisement("krea_2_turbo", "krea_2", true, false),
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
