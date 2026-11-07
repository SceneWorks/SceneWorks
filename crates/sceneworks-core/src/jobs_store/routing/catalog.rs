//! Model/kernel routing catalog: the per-backend routed-model and training-kernel lists,
//! the Mac support/capability probes, and their supporting types. Moved out of
//! `jobs_store.rs` (sc-8816) with no behavior change. Image/video membership comes from typed
//! route policy; native training membership joins product targets with runtime trainer exports.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{LazyLock, OnceLock};

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::checkpoint_weight_facts::NVFP4_CODEC_ID;
use crate::contracts::JobType;

use crate::jobs_store::routing::candle::video_mode_is_candle_eligible;
use crate::jobs_store::routing::gaps::{
    classify_candle_video_gap, classify_image_gap, classify_video_gap, UnsupportedReason,
};
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
    let supported = crate::registry::builtin_model_spec(model).map_or_else(
        || LEGACY_IMAGE_MLX_ROUTED.contains(&model),
        |spec| {
            spec.routing
                .as_ref()
                .is_some_and(crate::registry::RoutePolicy::mlx_routed)
                // Some shipped variant descriptors have no row in the historical router list.
                // The request probes below still decide the offered modes and conditioning.
                || (spec.routing.is_none() && spec.routes.contains_key("mlx"))
        },
    );
    if !supported {
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
            // Product admission probes include supplemental reference adapters that are not
            // declared by the primary generator descriptor.
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
    let routed = crate::registry::builtin_model_spec(model).map_or_else(
        || LEGACY_VIDEO_MLX_ROUTED.contains(&model),
        |spec| {
            spec.routing
                .as_ref()
                .is_some_and(crate::registry::RoutePolicy::mlx_routed)
        },
    );
    if !routed {
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

/// The [`JobType`] a video `mode` is enqueued as. The three advanced modes get their own job
/// types; everything else rides the base `video_generate`.
///
/// It lives HERE, in core, because three call sites need the SAME mapping and a re-typed copy is
/// the false-green shape this epic keeps hitting: `create_video_job` (which builds the real job),
/// [`video_mode_probe_payload`] (which builds the synthetic one the UI-gating probes ask the claim
/// predicates about), and any future caller that must reach a claim predicate from a `(model, mode)`
/// pair. If the API's mapping and the probe's mapping ever disagree, the gating oracle answers a
/// question about a job type that is never enqueued — the gate reads green and the user still hangs.
pub fn video_job_type_for_mode(mode: &str) -> JobType {
    match mode {
        "extend_clip" => JobType::VideoExtend,
        "video_bridge" => JobType::VideoBridge,
        "replace_person" => JobType::PersonReplace,
        _ => JobType::VideoGenerate,
    }
}

/// The canonical MINIMAL well-formed request for `(model, mode)` — the synthetic payload that
/// carries exactly the conditioning media `validate_video_job`'s per-mode required-asset `match`
/// demands for that mode, and nothing else (sc-19570).
///
/// This is the video sibling of [`probe_payload`]'s image probes, and it exists because the two
/// lanes' predicates are shaped differently. `video_mode_is_mlx_eligible` is `(model, mode)` and
/// ignores the payload entirely, so the Mac gate could map [`VIDEO_UI_MODES`] straight onto it. The
/// candle predicates are payload-shaped — `video_request_candle_eligible` requires the mode's source
/// image, `video_request_candle_vace_eligible` the clip/track/character set,
/// `scail2_animate_candle_eligible` the reference + driving clip — so asking them "does this model
/// serve this mode?" requires reconstructing the request. Restating each lane's answer as a
/// hand-maintained table instead is precisely what sc-19570's acceptance forbids.
///
/// MINIMAL, not maximal, in both directions:
/// * every key a mode's gate REQUIRES is present, so the probe never under-reports (hiding a mode
///   that works off-Mac is a worse regression than the hang this closes), and
/// * no key it does not require is present, because several gates REJECT stray conditioning — a
///   `sourceAssetId` on a `text_to_video` probe, or a `referenceAssetId` alongside it, both make
///   `video_request_candle_eligible` return false for reasons that have nothing to do with the mode.
///
/// The ids are placeholders: no gate resolves them, every one only asks whether the slot is filled.
pub(crate) fn video_mode_probe_payload(model: &str, mode: &str) -> Map<String, Value> {
    let probe = json!("probe");
    let media: Vec<(&str, Value)> = match mode {
        "text_to_video" => vec![],
        "image_to_video" => vec![("sourceAssetId", probe.clone())],
        "first_last_frame" => vec![
            ("sourceAssetId", probe.clone()),
            ("lastFrameAssetId", probe.clone()),
        ],
        "extend_clip" => vec![("sourceClipAssetId", probe.clone())],
        "video_bridge" => vec![
            ("sourceClipAssetId", probe.clone()),
            ("bridgeRightClipAssetId", probe.clone()),
        ],
        "replace_person" => vec![
            ("sourceClipAssetId", probe.clone()),
            ("personTrackId", probe.clone()),
            ("characterId", probe.clone()),
        ],
        "video_to_video" => vec![("sourceClipAssetId", probe.clone())],
        "reference_to_video" => vec![("referenceAssetIds", json!(["probe"]))],
        "reference_video_to_video" => vec![
            ("sourceClipAssetId", probe.clone()),
            ("referenceAssetIds", json!(["probe"])),
        ],
        "multi_video_to_video" => vec![("sourceClipAssetIds", json!(["probe-a", "probe-b"]))],
        "ads2v" => vec![
            ("sourceClipAssetId", probe.clone()),
            ("referenceClipAssetId", probe.clone()),
            ("referenceAssetIds", json!(["probe"])),
        ],
        "animate_character" => vec![
            ("referenceAssetIds", json!(["probe"])),
            ("sourceClipAssetId", probe.clone()),
        ],
        // An unknown mode probes as bare — every lane's per-mode arm refuses it, which is the
        // right answer for a mode no surface offers.
        _ => vec![],
    };
    let mut payload = probe_payload(model, &media);
    payload.insert("mode".to_owned(), Value::String(mode.to_owned()));
    // LTX's extend / bridge / replacement providers are the IC-LoRA keyframe-append paths, not a
    // separate checkpoint, so BOTH lanes require `loras_contain_ltx_ic_lora` for those three modes
    // (`video_request_is_mlx_eligible`, `ltx_replace_candle_eligible`,
    // `video_request_candle_eligible`). An adapter is not conditioning media and `validate_video_job`
    // does not demand it — but this payload is not only a well-formedness probe, it is what the CLAIM
    // predicates are asked, and they do.
    //
    // Without it this function UNDER-REPORTS, which the doc above names as the worse of the two
    // failure directions: `ModelCandleSupport` would report LTX extend/bridge/replacement as
    // candle-unclaimable and the Video Studio would hide three tabs off-Mac that the lane genuinely
    // serves. The class guard `every_declared_video_capability_is_claimable_by_some_lane` reads the
    // same shape and would call the same six (model, mode) pairs advertised-but-unclaimable.
    //
    // Confined to the LTX pair deliberately: `video_request_candle_vace_eligible` REJECTS a
    // LoRA-bearing advanced job for every model except the dedicated VACE-Fun provider, so attaching
    // this unconditionally would invert the answer for `wan_2_2` — the exact over-report the "no key
    // it does not require" half of the contract above forbids.
    if matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "ltx_2_5")
        && matches!(mode, "extend_clip" | "video_bridge" | "replace_person")
    {
        payload.insert(
            "loras".to_owned(),
            json!([{ "id": "probe-ltx-ic-lora", "icLora": true }]),
        );
    }
    payload
}

/// UI-facing per-model CANDLE (Windows/Linux) support — the off-Mac twin of [`ModelMacSupport`]
/// (sc-19570), derived from the same claim predicates the candle worker's `worker_supports_job`
/// arm consults, so what the UI hides off-Mac can never drift from what routing refuses there.
///
/// **Why it had to exist.** [`ModelMacSupport`] had no sibling, so off-Mac
/// `videoModelServesMode` collapsed to `capabilities.includes(mode)` — the manifest declaration
/// alone, with nothing asking whether a lane exists on that platform. Every MLX-only,
/// candle-unclaimable, Windows/Linux-installable pair was therefore offered as a Video Studio tab
/// on Windows and Linux, and submitting one produced a job that sat `queued` /
/// "Waiting for an available worker." forever: no mlx worker exists off-Mac to claim it, the candle
/// worker's own gate refuses it, and both enforce sweeps default to **warn**.
///
/// Scoped to `videoModes`, which is the whole of the measured gap: the block is emitted for video
/// models and reports nothing about image features. `supported: false` means no [`VIDEO_UI_MODES`]
/// entry is candle-claimable for this model at all, so the model itself is hidden off-Mac rather
/// than shown with every tab disabled.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCandleSupport {
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<UnsupportedReason>,
    pub features: ModelCandleFeatures,
}

/// Per-feature off-Mac support for a model (sc-19570) — the candle mirror of [`ModelMacFeatures`],
/// carrying the one feature axis the off-Mac gap was measured on. `video_modes` is populated only
/// for video models.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCandleFeatures {
    /// Video-only: which `video_generate` modes the candle lane claims. Empty for non-video models.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub video_modes: BTreeMap<String, bool>,
}

/// Off-Mac (candle) UI gating support for a model id of the given catalog `model_type` — the
/// sibling of [`model_mac_support`] (sc-19570). Converted builtins take availability from their
/// static Candle route. Legacy video models expose per-mode verdicts in `features.video_modes`;
/// other legacy rows keep their original supported/empty-features projection.
pub fn model_candle_support(model_id: &str, model_type: &str) -> ModelCandleSupport {
    if model_type == "image" {
        if let Some(spec) = crate::registry::builtin_model_spec(model_id) {
            if spec.routes.contains_key("candle") {
                return ModelCandleSupport {
                    supported: true,
                    reason: None,
                    features: ModelCandleFeatures::default(),
                };
            }
        }
    }
    match model_type {
        "video" => video_model_candle_support(model_id),
        _ => ModelCandleSupport {
            supported: true,
            reason: None,
            features: ModelCandleFeatures::default(),
        },
    }
}

pub(crate) fn video_model_candle_support(model: &str) -> ModelCandleSupport {
    let video_modes: BTreeMap<String, bool> = VIDEO_UI_MODES
        .iter()
        .map(|mode| {
            (
                (*mode).to_owned(),
                video_mode_is_candle_eligible(model, mode),
            )
        })
        .collect();
    if video_modes.values().any(|eligible| *eligible) {
        return ModelCandleSupport {
            supported: true,
            reason: None,
            features: ModelCandleFeatures { video_modes },
        };
    }
    // No mode at all: the model has no candle video engine off-Mac, so the picker hides it rather
    // than offering a model whose every tab is disabled. Derived, not listed — the verdict is the
    // per-mode map's own `any`, so a model that gains its first candle mode un-hides itself.
    ModelCandleSupport {
        supported: false,
        reason: Some(classify_candle_video_gap(&probe_payload(model, &[]))),
        features: ModelCandleFeatures::default(),
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
    /// The training kernels an off-Mac (candle) worker can run (sc-24161). The off-Mac twin of
    /// `supported_kernels`: the client applies it only when `candle_gating_active`, so a target
    /// whose kernel has no candle trainer is not offered on a host where the API would refuse it
    /// at Start.
    pub candle_supported_kernels: Vec<String>,
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
    /// sc-19570 — the off-Mac twin of `mac_gating_active`, and the master switch for the
    /// [`ModelCandleSupport`] block on `GET /api/v1/models`. **Platform-intrinsic, not a rollout
    /// flag**: it is `true` exactly when the host is not a Mac, because that is when the candle
    /// lane is the only backend that can register (`mlx_required` is documented "absent on
    /// Windows/Linux/Docker" — MLX is an Apple-silicon runtime). It deliberately does NOT read
    /// `candle_required`, which defaults OFF and only governs the enforce sweeps: the modes this
    /// gates are unreachable off-Mac whether or not a deployment opted into terminal gap
    /// reporting, so tying the gate to that flag would leave the default deployment — the one the
    /// defect was measured on — still offering tabs that hang.
    pub candle_gating_active: bool,
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
        // Iris-3B restoration (`engine=iris`, sc-25683, epic 25678): the `iris_3b_restore` transform
        // is registered by BOTH native bundles (`mlx-gen-iris` on Mac, `candle-gen-iris` on
        // Windows/Linux NVIDIA), so — exactly like SeedVR2 — the flag is platform-intrinsic: a
        // backend exists on Mac/Windows/Linux regardless of the gating rollout flag. Whether the
        // restoration task is INSTALLED is a separate, per-host fact the web reads from the model
        // catalog (`iris_3b_restore.installState`); this flag only says a lane exists. Must agree with
        // the routing oracle (mlx OR candle claims `engine=iris`; every other descriptor refuses it).
        "imageUpscaleIris".to_owned(),
        MacFeatureSupport {
            supported: seedvr2_supported,
            reason: if seedvr2_supported {
                None
            } else {
                Some(UnsupportedReason::new(
                    None,
                    "image_upscale (Iris restoration)",
                    "Iris restoration runs on Mac (native MLX) and Windows/Linux (the Candle CUDA/NVIDIA backend); this platform has no Iris backend.",
                    Some("sc-25683"),
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
        // Iris-3B monocular depth (sc-25682): the `iris_3b_depth` task export runs in-process on
        // the native MLX worker on Mac and the Candle worker on Windows/Linux (same native-backend
        // footprint as smart-select). Platform-intrinsic: whether the task export is INSTALLED is
        // the model catalog's `installState`, not this flag.
        "imageDepth".to_owned(),
        MacFeatureSupport {
            supported: image_segment_supported,
            reason: if image_segment_supported {
                None
            } else {
                Some(UnsupportedReason::new(
                    None,
                    "image_depth (Iris-3B depth)",
                    "Iris-3B depth requires the native MLX backend on macOS or the Candle backend on Windows/Linux.",
                    Some("sc-25682"),
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
        // sc-19570: the candle lane is the only backend off-Mac, so the per-mode `candleSupport`
        // gate engages exactly there. Derived from the same `is_mac` the platform-intrinsic engine
        // flags above use, never from `candle_required`.
        candle_gating_active: !is_mac,
        not_available_label: MAC_NOT_AVAILABLE_LABEL.to_owned(),
        features,
        training: MacTrainingSupport {
            supported_kernels: ordered_training_kernels(&MLX_ROUTED_TRAINING_KERNELS, "mlx"),
            candle_supported_kernels: ordered_training_kernels(
                &CANDLE_ROUTED_TRAINING_KERNELS,
                "candle",
            ),
            lokr_on_wan_supported: false,
        },
    }
}

// ---------------------------------------------------------------------------------------------
// Per-model routing capability tables (sc-9495)
// ---------------------------------------------------------------------------------------------
//
// The schema-validated route-policy.json owns SceneWorks router admission. Build.rs compiles its
// typed rows to the static slices consumed by existing job predicates; the native descriptor
// export separately owns engine breadth. The frozen membership test below checks that the policy
// migration preserves every existing route and catches the five formerly missing MLX gate rows.

/// One image model's SceneWorks router admission policy. Build.rs derives the static list
/// constants from these generated rows; the checked-in typed JSON is the sole authored owner.
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
    /// Candle accepts Q4/Q8 generation requests (either a packed-tier select or load-time quant;
    /// was `CANDLE_QUANT_MODELS`). This may overlap `candle_lora` when both capabilities work
    /// independently but their composition has not been admitted.
    pub(crate) candle_quant: bool,
    /// Candle advertises inference LoRA/LoKr (was `CANDLE_LORA_MODELS`). This may overlap
    /// `candle_quant`; overlap alone does not admit a quantized adapter composition.
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

// Generated from the schema-validated router product policy in the static ModelSpec build.
include!(concat!(env!("OUT_DIR"), "/model-route-policy.rs"));

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

/// Legacy family-gate test fixtures derive from the exact imported-provider registrations.
/// Production imported routing also selects the provider by source and operation.
#[cfg(test)]
pub(crate) static MLX_ROUTED_FAMILIES: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| legacy_test_routed_families("mlx"));

/// Test oracle for whether the generated provider facts contain at least one MLX route for a
/// family. It does not authorize a request without an exact source-shaped provider match.
pub(crate) fn image_family_is_mlx_routed(family: &str) -> bool {
    imported_provider_routes("mlx", family).next().is_some()
}

#[cfg(test)]
pub(crate) static CANDLE_ROUTED_FAMILIES: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| legacy_test_routed_families("candle"));

#[cfg(test)]
fn legacy_test_routed_families(backend: &str) -> Vec<&'static str> {
    crate::base_weights::IMPORT_SUPPORTED_FAMILIES
        .iter()
        .copied()
        .filter(|family| imported_provider_routes(backend, family).next().is_some())
        .collect()
}

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
/// [`imported_image_request_provider_eligible`] goes on to make. The focused test compares
/// descriptor declaration with the production route claim for every imported source and operation.
#[cfg(test)]
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
#[cfg(test)]
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

/// [`checkpoint_plan_checkpoint_id`] reached through a request payload's forwarded
/// `modelManifestEntry` — the shape the routing predicates hold (sc-20651).
pub(crate) fn checkpoint_plan_checkpoint_id_of_payload_entry(
    payload: &Map<String, Value>,
) -> Option<&str> {
    payload
        .get("modelManifestEntry")
        .and_then(Value::as_object)
        .and_then(checkpoint_plan_checkpoint_id)
}

/// Whether one backend lane claims this plan-backed H3 video probe WITH a compatible adapter.
/// The catalog's advertisement derives from the same predicates the scheduler consults, so the
/// UI can never offer an adapter stack no lane will claim.
pub fn imported_video_lane_claims_with_lora(lane: &str, probe: &Map<String, Value>) -> bool {
    match lane {
        "mlx" => super::mlx::video_request_is_mlx_eligible(&JobType::VideoGenerate, probe),
        "candle" => super::candle::video_request_is_candle_eligible(&JobType::VideoGenerate, probe),
        _ => false,
    }
}

/// Whether ANY valid supported shape of this plan-backed imported H3 entry is claimable WITH a
/// compatible adapter on the given lane pair. Every UI mode is probed through the SAME
/// mode-conditioning machinery the support projection uses — a reference-partition (Ref2VA)
/// import is advertised through the reference shape it actually serves, a hybrid through its
/// base shape — so a text-only probe can never withdraw an adapter surface the entry genuinely
/// claims on some other shape.
pub fn imported_video_adapter_surface_claimable(
    entry: &Map<String, Value>,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) -> bool {
    let Some(id) = entry.get("id").and_then(Value::as_str) else {
        return false;
    };
    VIDEO_UI_MODES.iter().any(|mode| {
        let mut payload = video_mode_probe_payload(id, mode);
        payload.insert(
            "modelManifestEntry".to_owned(),
            Value::Object(entry.clone()),
        );
        payload.insert(
            "loras".to_owned(),
            Value::Array(vec![json!({ "id": "minimax_h3_turbo_4step_v01" })]),
        );
        (mlx_lane_available
            && super::mlx::video_request_is_mlx_eligible(&JobType::VideoGenerate, &payload))
            || (candle_lane_available
                && super::candle::video_request_is_candle_eligible(
                    &JobType::VideoGenerate,
                    &payload,
                ))
    })
}

/// The video shapes an IMPORTED dense MiniMax-H3 DiT (sc-26680) is claimable on — on BOTH native
/// lanes, because both providers load a staged DiT directory through the same `transformer`
/// component seam.
///
/// Keyed on the forwarded `modelManifestEntry` ([`crate::minimax_h3_import::entry_is_imported_minimax_h3`]),
/// never on the model id, and narrow on every axis because a claim is a promise a worker arm exists:
/// * **id** — never a builtin video id. A builtin id names the shipped weights; a plan forwarded
///   under one would render the imported bytes under another checkpoint's name (or, on a lane that
///   ignored the plan, the shipped bytes under the imported one).
/// * **shape** — the explicitly declared FL2VA, Ref2VA, or hybrid operation union. Tensor shapes
///   and model names never imply task compatibility. Builtin predicates validate conditioning,
///   while the imported entry and plan remain the weight identity.
/// * **adapters** — the native H3 LoRA/LoKr and turbo surface. The same partition predicate owns
///   request shape; source topology, target namespace and pruning-basis compatibility are validated
///   by the native installer against the selected imported task, never by substituting stock weights.
///
/// This is a claim, not a load: the worker still resolves and re-verifies the plan, the binding and
/// the resident components, and refuses by name when it cannot.
pub(crate) fn plan_backed_minimax_h3_video_eligible(
    model: &str,
    payload: &Map<String, Value>,
) -> bool {
    let Some(entry) = payload.get("modelManifestEntry").and_then(Value::as_object) else {
        return false;
    };
    if !crate::minimax_h3_import::entry_is_imported_minimax_h3(entry) {
        return false;
    }
    if matches!(
        model,
        crate::minimax_h3_import::MINIMAX_H3_BASE_PARTITION_MODEL
            | crate::minimax_h3_import::MINIMAX_H3_REFERENCE_PARTITION_MODEL
    ) || VIDEO_MLX_ROUTED_MODELS.contains(&model)
        || CANDLE_VIDEO_ROUTED_MODELS.contains(&model)
    {
        return false;
    }
    let has_references = payload.get("mode").and_then(Value::as_str) == Some("reference_to_video");
    let Ok(partition) = crate::minimax_h3_import::entry_partition_model(entry, has_references)
    else {
        return false;
    };
    super::candle::video_request_candle_eligible(partition, payload)
}

/// Per-mode Mac support for an imported, plan-backed video entry (sc-26680) — the
/// [`video_model_mac_support`] twin that asks the REAL MLX claim predicate with the entry forwarded,
/// exactly as a submitted job forwards it. The id-keyed form cannot answer for an imported model:
/// its probe carries no entry, so every plan-backed predicate answers `false` and the Video Studio
/// would hide a model the lane serves.
pub fn imported_video_model_mac_support(entry: &Map<String, Value>) -> ModelMacSupport {
    let model = entry.get("id").and_then(Value::as_str).unwrap_or_default();
    let video_modes: BTreeMap<String, bool> = VIDEO_UI_MODES
        .iter()
        .map(|mode| {
            let payload = imported_video_mode_probe_payload(model, mode, entry);
            let job_type = video_job_type_for_mode(mode);
            (
                (*mode).to_owned(),
                super::mlx::video_request_is_mlx_eligible(&job_type, &payload),
            )
        })
        .collect();
    if video_modes.values().any(|eligible| *eligible) {
        return ModelMacSupport {
            supported: true,
            reason: None,
            features: ModelMacFeatures {
                video_modes,
                ..ModelMacFeatures::default()
            },
        };
    }
    video_model_mac_support(model)
}

/// The off-Mac twin of [`imported_video_model_mac_support`].
pub fn imported_video_model_candle_support(entry: &Map<String, Value>) -> ModelCandleSupport {
    let model = entry.get("id").and_then(Value::as_str).unwrap_or_default();
    let video_modes: BTreeMap<String, bool> = VIDEO_UI_MODES
        .iter()
        .map(|mode| {
            let payload = imported_video_mode_probe_payload(model, mode, entry);
            let job_type = video_job_type_for_mode(mode);
            (
                (*mode).to_owned(),
                super::candle::video_request_is_candle_eligible(&job_type, &payload),
            )
        })
        .collect();
    if video_modes.values().any(|eligible| *eligible) {
        return ModelCandleSupport {
            supported: true,
            reason: None,
            features: ModelCandleFeatures { video_modes },
        };
    }
    video_model_candle_support(model)
}

/// [`video_mode_probe_payload`] with the imported entry forwarded as `modelManifestEntry`.
fn imported_video_mode_probe_payload(
    model: &str,
    mode: &str,
    entry: &Map<String, Value>,
) -> Map<String, Value> {
    let mut payload = video_mode_probe_payload(model, mode);
    payload.insert(
        "modelManifestEntry".to_owned(),
        Value::Object(entry.clone()),
    );
    payload
}

/// The path a bespoke imported lane actually LOADS from, in the order the entry may spell it.
/// `None` when every spelling is absent or blank.
///
/// Deliberately excludes the provenance-only `source.path`, which [`imported_entry_installed_path`]
/// accepts as a last resort: an import writes `source.path` as a data-dir-RELATIVE breadcrumb
/// (`models/imports/<name>`), and a LINKED entry may carry one describing where its bytes came from
/// while owning no installed bytes at all. Treating that as a loadable path made a linked entry
/// claim a bespoke family lane it does not have — and pointed that lane at a path it must not read
/// (sc-20636 review).
pub fn imported_entry_loadable_path(entry: &Map<String, Value>) -> Option<&str> {
    entry
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
        .map(str::trim)
        .filter(|path| !path.is_empty())
}

/// Every spelling of an installed/linked path an entry may carry, including the provenance-only
/// `source.path` fallback. `None` when all of them are absent or blank.
///
/// The ADMISSION view: "does this entry describe bytes anywhere", used to decide whether an
/// imported entry is worth pricing. A caller deciding whether a lane can LOAD the entry wants
/// [`imported_entry_loadable_path`] instead — see its note on why the `source.path` fallback must
/// not answer that question.
pub fn imported_entry_installed_path(entry: &Map<String, Value>) -> Option<&str> {
    imported_entry_loadable_path(entry).or_else(|| {
        entry
            .get("source")
            .and_then(Value::as_object)
            .and_then(|source| source.get("path"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|path| !path.is_empty())
    })
}

/// The **source codec** an imported manifest entry's weights are stored in — the engine's stable
/// identity (`"nvfp4-v1"`, `"int8-per-row-v1"`, `"convrot-int8-native"`, …). Registered decoders
/// use the persisted header classification. Native H3 ConvRot instead requires the format stamp
/// AND matching assembly encoding derived from the strictly validated source inventory.
///
/// One place turns those persisted source proofs into the codec vocabulary, so admission and the
/// worker cannot disagree about the source format. `None` means the entry provides no known,
/// proved codec identity; it never asserts dense storage, dense execution, or a default tier.
///
/// **This is the SOURCE fact only.** It says nothing about how a load would execute the weights: a
/// checkpoint stored `nvfp4-v1` reports `nvfp4-v1` here on a Mac, on CPU, on datacenter `sm_100`
/// and on a pre-Blackwell GPU, every one of which would run the dense BF16 fallback. What a run
/// actually materialized is a separate fact carried on the asset receipt and the telemetry row (see
/// [`crate::checkpoint_weight_facts`]), and it is never derived from this one.
///
/// The lowercase normalization preserves the case-insensitive compare this replaced; the persisted
/// value is always written lowercase by the import job, so it only ever matters for an entry
/// hand-edited into a manifest.
pub fn imported_entry_source_codec(entry: &Map<String, Value>) -> Option<&'static str> {
    let label = entry
        .get("importQuantFormat")
        .and_then(Value::as_str)?
        .trim()
        .to_ascii_lowercase();
    if matches!(label.as_str(), "convrot_int8" | "asym_w4a8_int8")
        && crate::minimax_h3_import::entry_is_imported_minimax_h3(entry)
        && crate::minimax_h3_import::entry_assembly(entry).is_some_and(|assembly| {
            assembly.get("sourceEncoding").and_then(Value::as_str) == Some(label.as_str())
        })
    {
        // The strict compiled source inventory supplies this assembly stamp. Coarse header
        // classification or a family name alone never establishes a rotated operand format.
        return Some(if label == "asym_w4a8_int8" {
            crate::checkpoint_weight_facts::W4A8_INT8_CODEC_ID
        } else {
            crate::checkpoint_weight_facts::CONVROT_INT8_CODEC_ID
        });
    }
    crate::base_weights::QuantFormat::from_label(&label)?.source_codec_id()
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
    // Admission gates on what the checkpoint's weights are STORED in — the source codec — and on
    // nothing else (sc-21484, epic 11037). It deliberately does not consult, and cannot consult,
    // what a load would MATERIALIZE them as: that is a per-load receipt produced by the engine
    // after admission, and a gate that guessed at it would be deciding routability from a
    // prediction. `imported_entry_source_codec` is the one place the persisted classification is
    // turned into the cross-repo codec vocabulary, so this gate and the worker's route agree on the
    // question they are asking.
    let native_nvfp4 = imported_entry_source_codec(entry) == Some(NVFP4_CODEC_ID);
    // Native Kitchen NVFP4 is a Candle/CUDA source encoding. MLX has no consumer for its packed
    // E2M1 weights and blocked scales, so it must never win the family route merely because dense
    // Krea imports are cross-platform.
    if native_nvfp4 && backend != "candle" {
        return false;
    }
    // The source hint: an installed/linked path for the bespoke imported lanes, or — for a
    // plan-backed entry (epic 20398, sc-20634) — the persisted checkpoint identity the worker's
    // plan-driven route resolves through the checkpoint plan store. Either is a claim that the
    // worker then verifies fail-closed; neither is consumed for loading here.
    let has_nonempty_path = imported_entry_installed_path(entry).is_some();
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
    // Native NVFP4 is proven only on the Generate surface. The Candle Krea loader additionally
    // installs low-rank LoRA/LoKr adapters as residuals around packed Nvfp4Linear modules, so that
    // exact provider/source combination does not mutate the packed base. Other families retain the
    // fail-closed adapter guard until their native loaders prove the same contract. Unsupported Krea
    // adapter forms are rejected by the loader's typed validation.
    let native_nvfp4_adapters_supported = family == "krea_2" && backend == "candle";
    if native_nvfp4 && (operation != "generate" || (has_loras && !native_nvfp4_adapters_supported))
    {
        return false;
    }
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
    if native_nvfp4 {
        let advanced = payload.get("advanced").and_then(Value::as_object);
        let named = advanced
            .and_then(|advanced| advanced.get("quantTier").or_else(|| advanced.get("quant")))
            .and_then(Value::as_str)
            .map(str::trim);
        if named.is_some_and(|quant| !quant.eq_ignore_ascii_case("nvfp4"))
            || advanced.is_some_and(|advanced| advanced.contains_key("mlxQuantize"))
        {
            return false;
        }
    }
    match imported_requested_quant(payload, ignore_quant_tier) {
        // `supported_quants` advertises the provider's quantize-on-LOAD tiers. For a native NVFP4
        // checkpoint the tier is not a load-time instruction at all — the stored codec is a
        // property of the file and the engine's compiled plan owns which projections execute
        // packed (sc-21485: the klein single-file provider advertises only q4/q8 here and takes
        // no quant on its LoadSpec). The `native_nvfp4` block above has already vetted that the
        // request names nvfp4 or nothing, so requiring the label in `supported_quants` too would
        // close the exact route the engine registers.
        Ok(Some("nvfp4")) if native_nvfp4 => {}
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
    /// The candle image families that accept Q4/Q8 generation requests, either by selecting a
    /// pre-packed tier (including Z-Image) or by honoring load-time quant. Derived from
    /// [`IMAGE_MODEL_CAPS`]`.candle_quant` (sc-9495). A row may also appear in
    /// [`CANDLE_LORA_MODELS`] when both capabilities work independently; only
    /// [`CANDLE_QUANT_LORA_MODELS`] admits their composition. Subset of [`CANDLE_ROUTED_MODELS`].
    pub(crate) CANDLE_QUANT_MODELS, IMAGE_MODEL_CAPS, candle_quant
}

derive_model_list! {
    /// The candle image families that advertise inference LoRA/LoKr (derived from
    /// [`IMAGE_MODEL_CAPS`]`.candle_lora`). A row may also appear in [`CANDLE_QUANT_MODELS`] when
    /// both capabilities work independently; only [`CANDLE_QUANT_LORA_MODELS`] admits their
    /// composition. Subset of [`CANDLE_ROUTED_MODELS`].
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
    CANDLE_VIDEO_ROUTED_MODELS.contains(&model)
        || crate::registry::builtin_model_spec(model).is_some_and(|spec| {
            spec.product.family == "scail2"
                && spec
                    .video_routes
                    .get("candle")
                    .is_some_and(|routes| !routes.is_empty())
        })
}

/// SceneWorks training kernels with a registered MLX trainer supporting at least one technique
/// offered by a shipped target. The product target owns its kernel and offered techniques; the
/// runtime export owns the backend-local trainer identity and executable descriptor.
///
/// Public (sc-15277) so the worker's `engine_trainer_id` invariant can be checked from this set
/// rather than restating it: a kernel that is routed here but has no trainer mapping is claimed by
/// the mlx worker and then immediately failed, which is exactly how `mage_flow_lora` shipped broken.
pub static MLX_ROUTED_TRAINING_KERNELS: LazyLock<BTreeSet<String>> =
    LazyLock::new(|| crate::registry::native_training_kernels("mlx"));

/// Off-Mac twin of [`MLX_ROUTED_TRAINING_KERNELS`], projected from the Candle trainer export.
/// Families sharing a kernel remain base-model gated by [`training_job_is_candle_eligible`].
/// `krea_control` has a Candle control trainer but no MLX control trainer.
pub(crate) static CANDLE_ROUTED_TRAINING_KERNELS: LazyLock<BTreeSet<String>> =
    LazyLock::new(|| crate::registry::native_training_kernels("candle"));

/// Public capability-array order captured before the native route tables were derived. Only
/// ordering lives here: the set supplied to `ordered_training_kernels` decides membership. The
/// two ranks differ for LTX/Wan because the historical arrays had different backend ordering.
const TRAINING_KERNEL_PRESENTATION_ORDER: &[(&str, u8, u8)] = &[
    ("moss_image_lora", 0, 0),
    ("z_image_lora", 1, 1),
    ("sdxl_lora", 2, 2),
    ("kolors_lora", 3, 3),
    ("lens_lora", 4, 4),
    ("krea_lora", 5, 5),
    ("krea_control", u8::MAX, 6),
    ("sd3_lora", 6, 7),
    ("wan_lora", 7, 9),
    ("wan_moe_lora", 8, 10),
    ("ltx_mlx_lora", 9, 8),
    ("anima_lora", 10, 11),
    ("mage_flow_lora", 11, 12),
    ("iris_train", 12, 13),
    ("qwen_image_2_1_lora", 13, 14),
    ("qwen_image_2_1_edit_lora", 14, 15),
    ("flux3_klein_lora", 15, 16),
    ("flux3_klein_edit_lora", 16, 17),
];

fn ordered_training_kernels(kernels: &BTreeSet<String>, backend: &str) -> Vec<String> {
    let rank = |kernel: &str| {
        TRAINING_KERNEL_PRESENTATION_ORDER
            .iter()
            .find(|(id, _, _)| *id == kernel)
            .map(|(_, mlx, candle)| if backend == "mlx" { *mlx } else { *candle })
            .unwrap_or(u8::MAX)
    };
    let mut ordered: Vec<_> = kernels.iter().cloned().collect();
    ordered.sort_by(|left, right| rank(left).cmp(&rank(right)).then_with(|| left.cmp(right)));
    ordered
}

/// Whether an off-Mac (candle) worker can run this training kernel at all. A real training run on a
/// non-macOS host for a kernel that is NOT candle-routed has no worker that will ever claim it, so the
/// submit path refuses it up front instead of leaving it queued forever (sc-24159).
pub fn training_kernel_is_candle_routed(kernel: &str) -> bool {
    CANDLE_ROUTED_TRAINING_KERNELS.contains(kernel)
}

/// Both native MOSS lanes train adapters for the exact selected Base/SFT checkpoint only.
pub(crate) fn moss_training_plan_is_supported(plan: &Map<String, Value>) -> bool {
    let base = plan
        .get("target")
        .and_then(Value::as_object)
        .and_then(|target| target.get("baseModel"))
        .and_then(Value::as_str);
    if !matches!(base, Some("moss_image_base" | "moss_image_sft")) {
        return false;
    }
    let advanced = match plan.get("config") {
        None => None,
        Some(Value::Object(config)) => match config.get("advanced") {
            None => None,
            Some(Value::Object(advanced)) => Some(advanced),
            _ => return false,
        },
        _ => return false,
    };
    let method = match advanced.and_then(|advanced| advanced.get("networkType")) {
        None => "lora",
        Some(Value::String(value)) if value.trim().is_empty() => "lora",
        Some(Value::String(value)) => value.trim(),
        _ => return false,
    };
    method.eq_ignore_ascii_case("lora") || method.eq_ignore_ascii_case("lokr")
}

/// Native-only training policy — only a Rust worker can run these kernels, so a generic descriptor
/// must refuse the job (leaving it queued for a Rust worker) rather than claim it and fail with "no
/// training kernel". Despite the constant's historical name, this means native-Rust-only rather than
/// MLX-backend-only: most members have both MLX and Candle trainers. Candle workers are admitted
/// by the [`worker_supports_job`] exception when the kernel is also in
/// [`CANDLE_ROUTED_TRAINING_KERNELS`], while a generic worker remains refused. `krea_control` is the
/// intentional Candle-only exception; the historical constant name remains for compatibility.
/// Native runtime exports cannot derive this policy: registration of a native trainer does not
/// establish whether a separate generic trainer is absent.
pub(crate) const MLX_ONLY_TRAINING_KERNELS: &[&str] = &[
    "moss_image_lora",
    "ltx_mlx_lora",
    "krea_lora",
    "sd3_lora",
    "anima_lora",
    "krea_control",
    "mage_flow_lora",
    "iris_train",
    // Qwen Image 2.1 (sc-24159/sc-24160): native MLX + Candle trainers and no generic one, so a
    // generic worker refuses it while the candle exception admits it (it is candle-routed).
    "qwen_image_2_1_lora",
    // Qwen Image 2.1 edit (sc-24161/sc-24162): native MLX + Candle trainers and no generic one, so
    // a generic worker refuses it while the candle exception admits it (it is candle-routed).
    "qwen_image_2_1_edit_lora",
    "flux3_klein_lora",
    "flux3_klein_edit_lora",
];

#[cfg(test)]
mod tests {
    //! Membership-parity regression guard (sc-8816, strengthened sc-9495): model lists are
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
        imported_entry_source_codec, imported_image_model_lora_advertisement,
        imported_image_request_family_eligible, imported_image_request_provider_eligible,
        imported_provider_routes, imported_requested_quant, is_builtin_image_model,
        video_job_type_for_mode, video_mode_probe_payload, video_model_candle_support,
        video_model_mac_support, CANDLE_IMPORTED_CAPS, CANDLE_LORA_MODELS,
        CANDLE_QUANT_LORA_MODELS, CANDLE_QUANT_MODELS, CANDLE_ROUTED_FAMILIES,
        CANDLE_ROUTED_MODELS, CANDLE_VIDEO_I2V_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS,
        CANDLE_VIDEO_VACE_MODELS, IMAGE_MODEL_CAPS, MLX_IMPORTED_CAPS, MLX_ROUTED_FAMILIES,
        MLX_ROUTED_MODELS, VIDEO_MLX_ROUTED_MODELS, VIDEO_MODEL_CAPS, VIDEO_UI_MODES,
    };
    use crate::contracts::JobType;
    use crate::jobs_store::routing::gaps::video_request_is_claimable_on_platform;
    use crate::jobs_store::JobSnapshot;

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

    #[test]
    fn routed_model_lists_match_typed_policy() {
        // The generated compatibility lists must be exact projections of the sole authored
        // route policy, including its legacy aliases. No second model-ID snapshot is needed.
        let policy: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../config/manifests/registry/route-policy.json"
        ))
        .unwrap();
        for (rows, field, list) in [
            ("image", "mlxRouted", MLX_ROUTED_MODELS),
            ("image", "candleRouted", CANDLE_ROUTED_MODELS),
            ("image", "candleQuantLora", CANDLE_QUANT_LORA_MODELS),
            ("image", "candleQuant", CANDLE_QUANT_MODELS),
            ("image", "candleLora", CANDLE_LORA_MODELS),
            ("video", "candleVideoRouted", CANDLE_VIDEO_ROUTED_MODELS),
            ("video", "candleVideoI2v", CANDLE_VIDEO_I2V_ROUTED_MODELS),
            ("video", "candleVideoVace", CANDLE_VIDEO_VACE_MODELS),
            ("video", "videoMlxRouted", VIDEO_MLX_ROUTED_MODELS),
        ] {
            let expected: BTreeSet<_> = policy[rows]
                .as_array()
                .unwrap()
                .iter()
                .filter(|row| row[field] == true)
                .map(|row| row["id"].as_str().unwrap())
                .collect();
            let actual: BTreeSet<_> = list.iter().copied().collect();
            assert_eq!(actual, expected, "{rows}/{field} route-policy projection");
            assert_eq!(
                actual.len(),
                list.len(),
                "{rows}/{field} duplicate route id"
            );
        }
    }

    #[test]
    fn native_training_capability_arrays_keep_the_frozen_public_order() {
        let old: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../apps/web/src/testUtils/catalogBaseline/old-catalog.json"
        ))
        .unwrap();
        let actual = serde_json::to_value(super::mac_capabilities("macos", true)).unwrap();
        for key in ["supportedKernels", "candleSupportedKernels"] {
            assert_eq!(
                actual["training"][key], old["catalog"]["platforms"]["macos"]["training"][key],
                "{key} public order"
            );
        }
    }

    #[test]
    fn frozen_ui_models_with_missing_old_rows_reach_the_native_mlx_predicate() {
        for id in [
            "sensenova_u1_8b_infographic_v2",
            "sensenova_u1_8b_infographic_v2_fast",
            "sensenova_u1_8b_infographic_v3",
            "sensenova_u1_8b_infographic_v3_fast",
        ] {
            assert!(MLX_ROUTED_MODELS.contains(&id), "{id} router policy");
            assert!(
                crate::jobs_store::routing::mlx::image_request_mlx_eligible(
                    id,
                    &super::probe_payload(id, &[])
                ),
                "{id} native MLX request"
            );
        }
        let id = "qwen_image_edit_2511_lightning";
        assert!(MLX_ROUTED_MODELS.contains(&id));
        assert!(
            crate::jobs_store::routing::mlx::image_request_mlx_eligible(
                id,
                &super::probe_payload(
                    id,
                    &[
                        ("mode", serde_json::json!("edit_image")),
                        ("sourceAssetId", serde_json::json!("asset")),
                    ]
                )
            ),
            "{id} native MLX edit request"
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
            // A combined-capability row is complete in itself rather than duplicated into the two
            // standalone lists. The standalone lists MAY overlap: that precisely encodes a family
            // whose dense adapter and packed-tier paths are independently admitted while their
            // composition is not.
            assert!(
                !caps.candle_quant_lora || !(caps.candle_quant || caps.candle_lora),
                "{}: candle_quant_lora must not be duplicated into standalone columns",
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
        for family in MLX_ROUTED_FAMILIES.iter() {
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
        for family in CANDLE_ROUTED_FAMILIES.iter() {
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
    fn imported_kitchen_nvfp4_is_candle_generate_only() {
        let imported_id = "user_kreamania_variant7";
        let payload = |extra: serde_json::Value| {
            let mut value = serde_json::json!({
                "model": imported_id,
                "modelManifestEntry": {
                    "id": imported_id,
                    "family": "krea_2",
                    "importSourceShape": "transformer_file",
                    "importQuantFormat": "nvfp4",
                    "paths": { "model": "/app/models/imports/kreamania_variant7" }
                }
            });
            value
                .as_object_mut()
                .expect("payload is an object")
                .extend(extra.as_object().expect("extra is an object").clone());
            value.as_object().expect("payload is an object").clone()
        };

        assert!(imported_image_request_provider_eligible(
            imported_id,
            &payload(serde_json::json!({"mode": "text_to_image"})),
            "candle"
        ));
        assert!(imported_image_request_provider_eligible(
            imported_id,
            &payload(serde_json::json!({"referenceAssetId": "asset-1"})),
            "candle"
        ));
        assert!(imported_image_request_provider_eligible(
            imported_id,
            &payload(serde_json::json!({"loras": [{"id": "adapter"}]})),
            "candle"
        ));
        assert!(!imported_image_request_provider_eligible(
            imported_id,
            &payload(serde_json::json!({"mode": "text_to_image"})),
            "mlx"
        ));
        for unsupported in [
            serde_json::json!({"mode": "edit_image", "sourceAssetId": "asset-1"}),
            serde_json::json!({"advanced": {"poses": [{}]}}),
            serde_json::json!({"advanced": {"quantTier": "q8"}}),
        ] {
            assert!(!imported_image_request_provider_eligible(
                imported_id,
                &payload(unsupported),
                "candle"
            ));
        }
    }

    /// Admission reads the SOURCE codec, in the cross-repo codec vocabulary, and never a tier
    /// (sc-21484, epic 11037).
    #[test]
    fn admission_reads_the_source_codec_not_the_requested_tier() {
        use crate::checkpoint_weight_facts::NVFP4_CODEC_ID;
        use serde_json::{json, Map};

        let entry = |quant: Option<&str>| {
            let mut entry = Map::new();
            entry.insert("family".to_owned(), json!("krea_2"));
            if let Some(quant) = quant {
                entry.insert("importQuantFormat".to_owned(), json!(quant));
            }
            entry
        };

        assert_eq!(
            imported_entry_source_codec(&entry(Some("nvfp4"))),
            Some(NVFP4_CODEC_ID)
        );
        assert_eq!(
            imported_entry_source_codec(&entry(Some("nvfp4"))),
            Some("nvfp4-v1")
        );
        // Case tolerance is preserved from the compare this replaced.
        assert_eq!(
            imported_entry_source_codec(&entry(Some(" NVFP4 "))),
            Some(NVFP4_CODEC_ID)
        );
        assert_eq!(
            imported_entry_source_codec(&entry(Some("int8_tensorwise_per_row"))),
            Some("int8-per-row-v1")
        );
        assert_eq!(
            imported_entry_source_codec(&entry(Some("bf16"))),
            Some("dense-bf16-v1")
        );
        // A REQUEST TIER is not a source codec, and neither is an absent or unknown stamp.
        for not_a_codec in [None, Some("q4"), Some("q8"), Some("nvfp4-v1"), Some("")] {
            assert_eq!(
                imported_entry_source_codec(&entry(not_a_codec)),
                None,
                "{not_a_codec:?} must not resolve to a source codec"
            );
        }
        // The gate's own question, spelled out: an NVFP4-stored entry is native-SOURCE, which is
        // the only thing admission may decide from.
        assert!(imported_entry_source_codec(&entry(Some("nvfp4"))) == Some(NVFP4_CODEC_ID));
        assert!(imported_entry_source_codec(&entry(Some("bf16"))) != Some(NVFP4_CODEC_ID));
    }

    /// Saved-job round trip across the whole tier vocabulary (sc-21484, epic 11037).
    ///
    /// Surfacing the source codec and the execution representation as separate facts must not move
    /// the REQUEST shapes: a job saved before this change — `advanced.quantTier` of `bf16`/`q8`/`q4`
    /// (or `mlxQuantize` bits, or nothing at all) — has to admit exactly as it did, and the tier a
    /// payload resolves to has to be the same string. The new facts describe a load; they are not
    /// an input to one, and nothing in the routing gate reads them.
    ///
    /// The bit ladder is pinned here too, because `bits => tier` is the re-derivation that has
    /// failed open five times in this epic: NVFP4 is reachable ONLY by name, never from a count.
    #[test]
    fn saved_job_tier_request_shapes_round_trip_unchanged() {
        let dense_id = "user_kreamania_dense";
        let payload = |advanced: serde_json::Value| {
            let mut value = serde_json::json!({
                "model": dense_id,
                "mode": "text_to_image",
                "modelManifestEntry": {
                    "id": dense_id,
                    "family": "krea_2",
                    "importSourceShape": "transformer_file",
                    "paths": { "model": "/app/models/imports/kreamania_dense" }
                }
            });
            if let Some(advanced) = advanced.as_object() {
                value.as_object_mut().expect("payload is an object").insert(
                    "advanced".to_owned(),
                    serde_json::Value::Object(advanced.clone()),
                );
            }
            value.as_object().expect("payload is an object").clone()
        };

        // Named tiers resolve to themselves; `bf16`/`dense` resolve to "no explicit tier".
        for (advanced, expected) in [
            (serde_json::json!({ "quantTier": "q4" }), Ok(Some("q4"))),
            (serde_json::json!({ "quantTier": "q8" }), Ok(Some("q8"))),
            (serde_json::json!({ "quantTier": "bf16" }), Ok(None)),
            (serde_json::json!({ "quantTier": "dense" }), Ok(None)),
            (
                serde_json::json!({ "quantTier": "nvfp4" }),
                Ok(Some("nvfp4")),
            ),
            (serde_json::json!({}), Ok(None)),
            // The bit ladder, unchanged — and NVFP4 is not on it at any width.
            (serde_json::json!({ "mlxQuantize": 4 }), Ok(Some("q4"))),
            (serde_json::json!({ "mlxQuantize": 8 }), Ok(Some("q8"))),
            (serde_json::json!({ "mlxQuantize": "4" }), Ok(Some("q4"))),
            (serde_json::json!({ "mlxQuantize": 0 }), Ok(None)),
            (serde_json::json!({ "quantTier": "nonsense" }), Err(())),
        ] {
            assert_eq!(
                imported_requested_quant(&payload(advanced.clone()), false),
                expected,
                "saved payload {advanced} must resolve to {expected:?}"
            );
        }

        // And the same payloads still admit through the provider gate for a dense import.
        for advanced in [
            serde_json::json!({ "quantTier": "q4" }),
            serde_json::json!({ "quantTier": "q8" }),
            serde_json::json!({ "quantTier": "bf16" }),
            serde_json::json!({ "mlxQuantize": 8 }),
            serde_json::json!({}),
        ] {
            assert!(
                imported_image_request_provider_eligible(
                    dense_id,
                    &payload(advanced.clone()),
                    "candle"
                ),
                "a saved dense job with advanced {advanced} must still admit"
            );
        }
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
                MLX_ROUTED_FAMILIES.as_slice(),
                MLX_IMPORTED_CAPS,
            )
        };
        let candle = |p: &serde_json::Map<String, serde_json::Value>| {
            imported_image_request_family_eligible(
                imported_id,
                p,
                CANDLE_ROUTED_FAMILIES.as_slice(),
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
                CANDLE_ROUTED_FAMILIES.as_slice(),
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
            !candle(&payload(serde_json::json!({ "mode": "text_to_image" }))),
            "Candle has no registered Mage imported provider"
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

        // The generated Mage full-fine-tune seam exists in the pinned Candle runtime, but no
        // imported-provider registration currently makes that source claimable there.
        assert!(
            !CANDLE_ROUTED_FAMILIES.contains(&"mage-flow"),
            "test family projection must follow registered imported providers"
        );
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
                    MLX_ROUTED_FAMILIES.as_slice(),
                    MLX_IMPORTED_CAPS,
                ),
                imported_image_request_family_eligible(
                    imported_id,
                    p,
                    CANDLE_ROUTED_FAMILIES.as_slice(),
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
    pub(super) fn builtin_image_models() -> Vec<serde_json::Value> {
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

    /// The VIDEO half of [`builtin_image_models`] — the shipped rows the Video Studio reads
    /// `capabilities` out of to build its mode tabs.
    fn builtin_video_models() -> Vec<serde_json::Value> {
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
            .filter(|m| m.get("type").and_then(serde_json::Value::as_str) == Some("video"))
            .cloned()
            .collect()
    }

    /// A `video_generate`-family [`JobSnapshot`] carrying the media `mode` requires, so a routing
    /// gate is judged on a shape it actually serves rather than on an empty payload it would refuse
    /// for a second reason. Mirrors the API's own `mode` → `JobType` map in `create_video_job`.
    ///
    /// `text_to_video` deliberately carries NO `sourceAssetId`: `video_request_candle_eligible`
    /// rejects a stray source image on the non-i2v lane, so adding one would make the candle half
    /// of the probe answer `false` for a reason that has nothing to do with the capability.
    fn video_capability_probe(model: &str, mode: &str) -> JobSnapshot {
        // sc-19570 folded the hand-maintained copies of this table into the production
        // `video_mode_probe_payload` / `video_job_type_for_mode`, which the off-Mac gating oracle
        // now builds its per-mode verdicts from. Delegating rather than keeping a parallel copy is
        // the point: this guard and the gate it sits next to must judge the SAME request shape, or
        // one can be green about a payload the other never sees.
        let payload = video_mode_probe_payload(model, mode);
        let job_type = match video_job_type_for_mode(mode) {
            JobType::VideoExtend => "video_extend",
            JobType::VideoBridge => "video_bridge",
            JobType::PersonReplace => "person_replace",
            _ => "video_generate",
        };
        serde_json::from_value(serde_json::json!({
            "id": "job_video_probe",
            "type": job_type,
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-08-14T00:00:00Z",
            "updatedAt": "2026-08-14T00:00:00Z"
        }))
        .expect("valid video job")
    }

    /// The (model, capability) pairs the shipped manifest ADVERTISES that no lane will claim.
    ///
    /// Not a suppression list — an inventory that the guard asserts is EXACT, so fixing an entry
    /// turns the test red until the entry is deleted, and a NEW divergence turns it red too. Empty
    /// is the target state; every row here is a filed, surfaced defect, never a quiet exemption.
    /// EMPTY, which is the target state and is now the SHIPPED state (sc-19504).
    ///
    /// Its one and only row was `wan_2_2_i2v_14b` + `first_last_frame`, recorded by sc-17159 because
    /// the choice between "serve it" and "stop advertising it" was a product/engine call. sc-19504
    /// made it: the capability was WITHDRAWN from the manifest. Neither engine has a keyframe path
    /// on the A14B — both the MLX and the candle I2V-A14B descriptors declare a single
    /// `ConditioningKind::Reference`, and each `build_i2v_y` hard-pins its 4-channel temporal mask
    /// to latent frame 0 — and the checkpoint's `in_dim 36` patch embedding was trained on
    /// "frame 0 image, rest zeros", so a second pinned frame is out-of-distribution rather than an
    /// unwired flag. The 5B's FLF rides a mask-blend architecture the A14B does not have.
    /// **THE CLASS GUARD (sc-17159, GH #2074).** A video mode in a shipped model's `capabilities`
    /// is a promise to the user — the Video Studio builds its mode tabs from that array. This
    /// asserts the promise is keepable: every advertised capability is a mode the Mac gate can
    /// reason about ([`VIDEO_UI_MODES`]) AND one some lane will actually CLAIM.
    ///
    /// GH #2074 is the shape: SCAIL-2's `animate_character` was wired through the catalog,
    /// `VIDEO_UI_MODES`, `video_mode_is_mlx_eligible`, the candle claim gate and the worker's
    /// dispatch — everything but the API allow-list — and 400'd on every submission from the moment
    /// it shipped. This guard's own run found two more: `wan_2_2_vace_fun_14b` had no
    /// [`VIDEO_MODEL_CAPS`] row at all despite a dedicated `VideoRoute::ReplacePersonWanVaceFun` arm
    /// (fixed in sc-17159), and `wan_2_2_i2v_14b`'s `first_last_frame` (withdrawn in sc-19504).
    ///
    /// It is the DECLARATION half only. It proves an advertised mode is claimable; it cannot prove
    /// that an UNadvertised one is refused, because `VIDEO_JOB_MODES` is global and any caller may
    /// name any admitted mode against any model. That half is the enqueue-time no-lane gate
    /// ([`super::super::gaps::video_request_is_claimable_by_any_lane`], sc-19504), whose own guard is
    /// `a_video_mode_no_lane_serves_is_refused_at_submission` (apps/rust-api tests/jobs.rs).
    ///
    /// Derived from the real tables on BOTH sides, never a restated list: the ADVERTISEMENT is read
    /// out of the shipped `builtin.models.jsonc` bytes, and the CLAIM is the real predicates
    /// [`video_job_is_mlx_eligible`] / [`video_job_is_candle_eligible`] that `worker_supports_job`
    /// consults. So a new family is covered the moment its manifest row lands.
    ///
    /// The API allow-list is the sixth surface and lives in a different crate; its half of this
    /// guard is `every_declared_video_capability_is_submittable` (apps/rust-api tests/jobs.rs),
    /// which reads the same manifest bytes against the real `VIDEO_JOB_MODES` constant.
    #[test]
    fn every_declared_video_capability_is_claimable_by_some_lane() {
        let models = builtin_video_models();
        assert!(
            models.len() >= 12,
            "the shipped video catalog shrank unexpectedly ({}) — this guard reads the real \
             manifest and would be asserting almost nothing",
            models.len()
        );

        let mut unclaimable: Vec<(String, String)> = Vec::new();
        let mut advertised_pairs = 0_usize;
        for model in &models {
            let id = model
                .get("id")
                .and_then(serde_json::Value::as_str)
                .expect("every model row has an id");
            let capabilities = model
                .get("capabilities")
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| panic!("{id}: every shipped video model declares capabilities"));
            assert!(
                !capabilities.is_empty(),
                "{id}: a video model with no capabilities serves no mode at all"
            );
            for capability in capabilities {
                let mode = capability
                    .as_str()
                    .unwrap_or_else(|| panic!("{id}: capabilities entries are strings"));
                advertised_pairs += 1;
                // The Mac gate builds `macSupport.features.videoModes` by mapping VIDEO_UI_MODES;
                // a capability outside it is invisible to `macVideoModeBlock` and to the studio's
                // own mode list, so the tab can never be offered however well it is routed.
                assert!(
                    VIDEO_UI_MODES.contains(&mode),
                    "{id} advertises `{mode}` but it is not in VIDEO_UI_MODES, so the Video Studio \
                     has no tab for it and the Mac gate has no entry for it"
                );
                let job = video_capability_probe(id, mode);
                let claimable = super::super::mlx::video_job_is_mlx_eligible(&job)
                    || super::super::candle::video_job_is_candle_eligible(&job);
                if !claimable {
                    unclaimable.push((id.to_owned(), mode.to_owned()));
                }
            }
        }
        assert!(
            advertised_pairs >= 30,
            "only {advertised_pairs} (model, capability) pairs were probed — the manifest read is \
             wrong and this guard is vacuous"
        );

        let found: BTreeSet<(String, String)> = unclaimable.into_iter().collect();
        assert!(
            found.is_empty(),
            "advertised video modes have no claiming lane: {found:?} (GH #2074)"
        );
    }

    /// **THE OFF-MAC CLASS GUARD (sc-19570).** The sibling above proves an advertised mode is
    /// claimable by SOME lane, anywhere. That is the platform-independent question, and it passes
    /// for every pair below — which is exactly why they hung: they are claimable on a Mac and
    /// claimable NOWHERE on Windows or Linux, where no `mlx` worker can ever register. The Video
    /// Studio offered each as a tab off-Mac (`macGatingActive` is false there, so
    /// `videoModelServesMode` collapsed to `capabilities.includes(mode)`), and submitting one
    /// produced a job that sat `queued` / "Waiting for an available worker." with no terminal state.
    ///
    /// The inventory is exact: it is not a suppression list, it is the measured set, and every member must be BOTH hidden (a `false` in the model's
    /// `candleSupport.features.videoModes`, which `candleVideoModeBlock` reads) AND judged
    /// unreachable on both off-Mac platforms (`video_request_is_claimable_on_platform`). Hiding
    /// alone would not do: `VIDEO_JOB_MODES` is global and the MCP tool, a raw REST caller or a
    /// recipe replay can still name any admitted mode against any model.
    ///
    /// **The unreachability verdict is an EXECUTION outcome, never a status code.** A client may be
    /// platform-aware — hiding a tab off-Mac is fine, and is half of sc-19570's value — but the
    /// HTTP surface may not be: `POST /api/v1/video/jobs` answers `201` for these pairs on every
    /// host, and `JobsStore::fail_platform_unreachable_jobs` then fails the JOB terminal with a
    /// legible reason. sc-19570 shipped this as a platform-conditional `400` first and that was
    /// ruled out. Do not re-read this predicate at an HTTP boundary.
    ///
    /// It asserts the Mac side too, in the same loop and on the same pairs, because "fix the other
    /// platform" is one edit away from "break this one".
    ///
    /// Derived on both sides: the ADVERTISEMENT is read from the shipped `builtin.models.jsonc`
    /// bytes, and the verdicts come from the real predicates. Nothing here restates a mode list.
    #[test]
    fn every_declared_video_mode_with_no_off_mac_lane_is_hidden_and_unreachable() {
        let models = builtin_video_models();
        assert!(
            models.len() >= 12,
            "the shipped video catalog shrank unexpectedly ({}) — this guard reads the real \
             manifest and would be asserting almost nothing",
            models.len()
        );

        let mut checked = 0_usize;
        for model in &models {
            let id = model
                .get("id")
                .and_then(serde_json::Value::as_str)
                .expect("every model row has an id");
            let candle_support = video_model_candle_support(id);
            let mac_support = video_model_mac_support(id);
            for capability in model
                .get("capabilities")
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| panic!("{id}: every shipped video model declares capabilities"))
            {
                let mode = capability
                    .as_str()
                    .unwrap_or_else(|| panic!("{id}: capabilities entries are strings"));
                checked += 1;
                let job_type = video_job_type_for_mode(mode);
                let payload = video_mode_probe_payload(id, mode);
                let mlx = super::super::mlx::video_request_is_mlx_eligible(&job_type, &payload);
                let candle =
                    super::super::candle::video_request_is_candle_eligible(&job_type, &payload);
                // The `candleSupport` block the web reads must agree with the predicate, per mode,
                // for EVERY advertised pair — not only the stranded ones.
                assert_eq!(
                    candle_support
                        .features
                        .video_modes
                        .get(mode)
                        .copied()
                        .unwrap_or(false),
                    candle,
                    "{id} + {mode}: candleSupport.features.videoModes disagrees with the real \
                     candle claim predicate, so the off-Mac UI gate and routing would diverge"
                );
                if !(mlx && !candle) {
                    continue;
                }
                // 1. HIDDEN off-Mac: either the whole model is candle-unsupported (so the picker
                //    drops it) or this specific mode is `false` (so the tab never renders).
                assert!(
                    !candle_support.supported
                        || candle_support.features.video_modes.get(mode) == Some(&false),
                    "{id} + {mode} has no off-Mac lane but candleSupport still offers it — the \
                     Video Studio would show the tab on Windows/Linux"
                );
                // 2. UNREACHABLE on BOTH off-Mac platforms, for the callers that never see a tab at
                //    all — so the sweep terminates the job instead of letting it queue forever.
                for os in ["windows", "linux"] {
                    assert!(
                        !video_request_is_claimable_on_platform(&job_type, &payload, os),
                        "{id} + {mode} would still be judged CLAIMABLE on {os}, where nothing can \
                         claim it — the sweep would leave it queued forever"
                    );
                }
                // 3. UNCHANGED on the Mac, where the pair genuinely renders.
                assert!(
                    video_request_is_claimable_on_platform(&job_type, &payload, "macos"),
                    "{id} + {mode} must still be accepted on macOS — refusing it there would break \
                     a working combination to fix a different platform"
                );
                assert_eq!(
                    mac_support.features.video_modes.get(mode),
                    Some(&true),
                    "{id} + {mode} must stay offered on a gated Mac"
                );
            }
        }
        assert!(
            checked >= 30,
            "only {checked} advertised (model, mode) pairs were probed — the manifest read is \
             wrong and this guard is vacuous"
        );
    }

    /// The other half of the same mechanism: a pair the candle lane DOES claim must stay claimable
    /// off-Mac. The cheapest way to get a platform verdict wrong is to condemn too much — a
    /// `wan_2_2` `extend_clip` on Windows renders through candle Wan-VACE today, and an over-broad
    /// predicate would have the sweep fail it terminal on the host that serves it.
    ///
    /// Includes two pairs no model ADVERTISES but candle serves anyway (`wan_2_2_t2v_14b` +
    /// `extend_clip`, `wan_2_2_i2v_14b` + `replace_person`), for the reason sc-19504 gave the
    /// platform-independent gate: this is not a capability gate, and a capability-shaped one would
    /// condemn working shapes.
    #[test]
    fn a_candle_served_video_mode_is_still_claimable_off_mac() {
        let served = [
            ("wan_2_2", "text_to_video"),
            ("wan_2_2", "extend_clip"),
            ("wan_2_2", "video_bridge"),
            ("wan_2_2", "replace_person"),
            ("wan_2_2_t2v_14b", "text_to_video"),
            ("wan_2_2_t2v_14b", "extend_clip"),
            ("wan_2_2_i2v_14b", "image_to_video"),
            ("wan_2_2_i2v_14b", "extend_clip"),
            ("wan_2_2_i2v_14b", "video_bridge"),
            ("wan_2_2_i2v_14b", "replace_person"),
            ("ltx_2_3", "text_to_video"),
            // `ltx_2_3_eros` is deliberately NOT here: SC-18902 withdrew its failed candle route,
            // so its modes are asserted MLX-only in MLX_ONLY_ADVERTISED_PAIRS above instead.
            ("svd", "image_to_video"),
            ("mochi_1", "text_to_video"),
            ("bernini", "text_to_video"),
            ("bernini", "video_to_video"),
            ("bernini", "reference_to_video"),
            ("bernini", "reference_video_to_video"),
            ("bernini", "multi_video_to_video"),
            ("bernini", "ads2v"),
            ("scail2_14b", "animate_character"),
            ("scail2_14b", "replace_person"),
        ];
        for (model, mode) in served {
            let job_type = video_job_type_for_mode(mode);
            let payload = video_mode_probe_payload(model, mode);
            assert!(
                super::super::candle::video_mode_is_candle_eligible(model, mode),
                "{model} + {mode} is candle-served — the probe payload must reach the real gate"
            );
            for os in ["windows", "linux", "macos"] {
                assert!(
                    video_request_is_claimable_on_platform(&job_type, &payload, os),
                    "{model} + {mode} must stay claimable on {os}"
                );
            }
        }
    }

    /// A video model's `ui.recommendedFor` is a SECOND advertisement of the same promise — the
    /// studio reads it to highlight modes — and nothing derived it from `capabilities`, so the two
    /// could drift silently (sc-19504: `wan_2_2_i2v_14b` listed `first_last_frame` in BOTH, and
    /// withdrawing it from one would have left the other still recommending the mode that hangs).
    ///
    /// A subset, not an equality: recommending fewer modes than a model serves is an editorial
    /// choice (`ltx_2_3_eros` recommends 2 of its 6). Recommending one it does not serve is not.
    /// Scoped to video because an image model's `recommendedFor` is a different vocabulary
    /// (`character` / `style`), not a mode list.
    #[test]
    fn every_recommended_video_mode_is_one_the_model_declares() {
        let models = builtin_video_models();
        let mut checked = 0_usize;
        for model in &models {
            let id = model
                .get("id")
                .and_then(serde_json::Value::as_str)
                .expect("every model row has an id");
            let capabilities: BTreeSet<&str> = model
                .get("capabilities")
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| panic!("{id}: every shipped video model declares capabilities"))
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect();
            let Some(recommended) = model
                .get("ui")
                .and_then(|ui| ui.get("recommendedFor"))
                .and_then(serde_json::Value::as_array)
            else {
                continue;
            };
            for mode in recommended.iter().filter_map(serde_json::Value::as_str) {
                checked += 1;
                assert!(
                    capabilities.contains(mode),
                    "{id} recommends `{mode}` in ui.recommendedFor but does not declare it in \
                     `capabilities` — the studio highlights a mode the model does not serve, and \
                     the capability guards (which read `capabilities`) cannot see it"
                );
            }
        }
        assert!(
            checked >= 12,
            "only {checked} recommended modes were checked — the manifest read is wrong and this \
             guard is vacuous"
        );
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

#[cfg(test)]
mod moss_native_training_routing_tests {
    use super::*;
    use crate::contracts::{JobSnapshot, JobType};
    use crate::jobs_store::routing::{candle, mlx};

    fn job(plan: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id":"moss-routing", "type":"lora_train", "status":"queued",
            "payload":{"plan":plan}, "result":{}, "requestedGpu":"auto",
            "progress":0, "stage":"queued", "message":"", "attempts":0,
            "cancelRequested":false, "createdAt":"", "updatedAt":""
        }))
        .unwrap()
    }

    #[test]
    fn moss_exact_checkpoint_adapters_route_to_both_native_lanes_not_generic() {
        let candle_worker: crate::contracts::WorkerSnapshot = serde_json::from_value(json!({
            "id":"moss-candle", "gpuId":"0", "gpuName":null, "status":"idle",
            "currentJobId":null, "capabilities":["image_generate", "candle"],
            "loadedModels":[], "registeredAt":"", "lastSeenAt":""
        }))
        .unwrap();
        for base in ["moss_image_base", "moss_image_sft"] {
            let entry = super::tests::builtin_image_models()
                .iter()
                .find(|entry| entry.get("id").and_then(Value::as_str) == Some(base))
                .unwrap()
                .clone();
            for method in ["lora", "lokr"] {
                let mut generation = job(json!({}));
                generation.job_type = JobType::ImageGenerate;
                generation.payload = json!({
                    "model":base, "mode":"text_to_image", "prompt":"a ceramic cup",
                    "width":64, "height":64, "modelManifestEntry":entry,
                    "loras":[{"id":"trained-moss", "networkType":method, "weight":0.65}],
                    "advanced":{"mossImage":{"acceleration":"off"}}
                })
                .as_object()
                .unwrap()
                .clone();
                let entry = generation.payload["modelManifestEntry"]
                    .as_object()
                    .unwrap();
                assert!(
                    crate::moss_image::native_request(&generation.payload, entry)
                        .unwrap()
                        .is_some()
                );
                assert!(
                    candle::image_job_is_candle_eligible(&generation),
                    "{base}/{method}"
                );
                assert!(crate::jobs_store::worker_supports_job(
                    &candle_worker,
                    &generation
                ));
                assert_eq!(generation.payload["loras"][0]["networkType"], method);
                assert_eq!(generation.payload["loras"][0]["weight"], 0.65);
                // Admission of custom adapters must not enable packed weights or conditioning.
                for field in ["sourceAssetId", "referenceAssetId", "maskAssetId"] {
                    let mut invalid = generation.clone();
                    invalid
                        .payload
                        .insert(field.into(), json!("unsupported-asset"));
                    assert!(!candle::image_job_is_candle_eligible(&invalid));
                    assert!(!crate::jobs_store::worker_supports_job(
                        &candle_worker,
                        &invalid
                    ));
                }
                let mut edit = generation.clone();
                edit.payload.insert("mode".into(), json!("edit_image"));
                assert!(!candle::image_job_is_candle_eligible(&edit));
                for bits in [4, 8] {
                    let mut packed = generation.clone();
                    packed.payload.get_mut("advanced").unwrap()["mlxQuantize"] = json!(bits);
                    assert!(!candle::image_job_is_candle_eligible(&packed));
                    assert!(!crate::jobs_store::worker_supports_job(
                        &candle_worker,
                        &packed
                    ));
                }
            }
        }
        let targets: Vec<_> = crate::training::builtin_training_targets()
            .targets
            .into_iter()
            .filter(|target| target.kernel == "moss_image_lora")
            .collect();
        assert!(["moss_image_base", "moss_image_sft"]
            .iter()
            .all(|base| targets.iter().any(|target| target.base_model == *base)));
        for target in targets {
            let plan = json!({"target":{"kernel":target.kernel,"baseModel":target.base_model},
                "config":target.defaults});
            for method in ["lora", "lokr", " LoKR ", ""] {
                let mut selected = plan.clone();
                selected["config"]["advanced"]["networkType"] = json!(method);
                let selected = job(selected);
                assert!(mlx::training_job_is_mlx_eligible(&selected));
                assert!(candle::training_job_is_candle_eligible(&selected));
                assert!(mlx::training_kernel_is_mlx_only(&selected));
            }
            for method in [json!("full"), json!("control"), json!("unknown"), json!(42)] {
                let mut invalid = plan.clone();
                invalid["config"]["advanced"]["networkType"] = method;
                let invalid = job(invalid);
                assert!(!mlx::training_job_is_mlx_eligible(&invalid));
                assert!(!candle::training_job_is_candle_eligible(&invalid));
                assert!(mlx::training_kernel_is_mlx_only(&invalid));
            }
            let mut foreign = plan.clone();
            foreign["target"]["baseModel"] = json!("qwen_image_2_1");
            let foreign = job(foreign);
            assert!(!mlx::training_job_is_mlx_eligible(&foreign));
            assert!(!candle::training_job_is_candle_eligible(&foreign));
            let mut control = job(plan);
            control.job_type = JobType::ControlTraining;
            assert!(!mlx::training_job_is_mlx_eligible(&control));
            assert!(!candle::training_job_is_candle_eligible(&control));
        }
    }
}
