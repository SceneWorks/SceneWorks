//! Candle (off-Mac / CUDA) routing predicates. Moved out of `jobs_store.rs` (sc-8816) with
//! no behavior change.

use serde_json::{Map, Value};

use crate::contracts::{JobSnapshot, JobType, WorkerSnapshot};
use crate::image_request::MAX_JOB_POSES;
use crate::jobs_store::routing::catalog::{
    imported_image_request_family_eligible, CANDLE_IMPORTED_CAPS, CANDLE_LORA_MODELS,
    CANDLE_QUANT_LORA_MODELS, CANDLE_QUANT_MODELS, CANDLE_ROUTED_FAMILIES, CANDLE_ROUTED_MODELS,
    CANDLE_ROUTED_TRAINING_KERNELS, CANDLE_VIDEO_I2V_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS,
    CANDLE_VIDEO_VACE_MODELS,
};
use crate::jobs_store::routing::mlx::{
    instantid_mlx_eligible, pulid_flux_mlx_eligible, upscale_job_is_mlx_eligible,
    video_upscale_job_is_mlx_eligible,
};
use crate::jobs_store::routing::{
    conditioned_reference_count, has_malformed_optional_nested_number, has_nonempty_array,
    has_nonempty_nested_array, has_nonempty_or_malformed_array,
    has_nonempty_or_malformed_nested_array, has_nonempty_or_malformed_string, has_nonempty_string,
    has_nonempty_string_array, has_nonnull_or_malformed_nested_carrier,
    krea_edit_has_unsupported_carrier,
};

/// Candle video models whose provider descriptor advertises user-LoRA inference, so a video job
/// carrying `request.loras` stays on the candle lane instead of being refused. Wan-14B applies
/// adapters per MoE expert, while base LTX installs additive residuals on its video-attention
/// projections. Wan-5B now applies dense/packed LoRA and LoKr, and Bernini exposes its provider
/// slot. 10Eros remains MLX-only after SC-18902's failed Candle acceptance; SVD and Mochi advertise
/// no adapter slot. Mirror of the candle-gen descriptors — kept in lockstep the same way
/// `CANDLE_VIDEO_ROUTED_MODELS` mirrors the routed engines.
pub(crate) const CANDLE_VIDEO_LORA_MODELS: &[&str] = &[
    "wan_2_2",
    "wan_2_2_t2v_14b",
    "wan_2_2_i2v_14b",
    "ltx_2_3",
    "bernini",
];

/// Does this image job belong on the candle (Windows/CUDA) image lane (epic 3672, sc-3678)? The base
/// `generate_candle_stream` drives plain text-to-image, and the bespoke lanes branched out below add
/// the conditioned shapes ported under epic 5480 — SDXL/FLUX.2/Qwen `edit_image` (sc-5487), IP-Adapter
/// reference (sc-5488/sc-5872), InstantID/PuLID identity (sc-5491/sc-5492), and strict-pose ControlNet
/// (sc-5489). Anything still without a candle lane (an unsupported family or shape, or an adapter on
/// a family whose descriptor advertises no adapter support) is refused here and remains queued for a
/// capable native worker.
///
/// Like the MLX twin [`image_job_is_mlx_eligible`], this accepts BOTH `image_generate` and the distinct
/// `image_edit` job type (the Image Studio/Editor "plain Image Edit": `mode == "edit_image"` +
/// `sourceAssetId`, epic 2427) — the engine dispatches the SdxlEdit/Flux2Edit/QwenEdit lanes by payload
/// model+mode, not job type, so both job types route through the same per-model predicates. Without
/// `image_edit` here a plain Image Edit was wrongly enforce-failed `candle_unsupported` off-Mac instead
/// of reaching its candle edit lane (the sc-5487 lanes were validated only via `image_generate` jobs, so
/// the gap was invisible). The conditioning signals mirror the worker's `sdxl_sub_mode` / `pose_entries`
/// exactly, so the router and worker agree on the lane boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CandleImageLane {
    ImportedFamily,
    InstantId,
    SdxlEdit,
    Flux2Edit,
    QwenEdit,
    SenseNovaEdit,
    ZImageEdit,
    ZImageIdentity,
    IdeogramEdit,
    IdeogramImg2Img,
    BooguEdit,
    MageEdit,
    BooguImg2Img,
    KreaEdit,
    BerniniEdit,
    SdxlIpAdapter,
    KolorsIpAdapter,
    FluxIpAdapter,
    QwenControl,
    KolorsControl,
    KolorsEdit,
    ZImageControl,
    ZImageImg2Img,
    Sd3Img2Img,
    SanaImg2Img,
    Flux1Control,
    Flux2Control,
    KreaControl,
    KreaImg2Img,
    Pulid,
    TextToImage,
}

struct CandleImageRoute {
    lane: CandleImageLane,
    models: ModelMatch,
    shape: fn(&Map<String, Value>) -> bool,
}

enum ModelMatch {
    Any(&'static [&'static str]),
    Family(fn(&str) -> bool),
}

impl CandleImageRoute {
    fn matches(&self, model: &str, payload: &Map<String, Value>) -> bool {
        let model_matches = match self.models {
            ModelMatch::Any(models) => models.contains(&model),
            ModelMatch::Family(predicate) => predicate(model),
        };
        model_matches && (self.shape)(payload)
    }
}

const CANDLE_IMAGE_ROUTES: &[CandleImageRoute] = &[
    CandleImageRoute {
        lane: CandleImageLane::InstantId,
        models: ModelMatch::Any(&["instantid_realvisxl"]),
        shape: instantid_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::SdxlEdit,
        models: ModelMatch::Family(is_sdxl_family_candle_model),
        shape: sdxl_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::Flux2Edit,
        models: ModelMatch::Any(&[
            "flux2_klein_9b",
            "flux2_klein_9b_kv",
            "flux2_klein_9b_true_v2",
        ]),
        shape: flux2_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::Flux2Edit,
        models: ModelMatch::Any(&["flux2_dev"]),
        shape: flux2_dev_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::QwenEdit,
        models: ModelMatch::Any(&[
            "qwen_image_edit",
            "qwen_image_edit_2509",
            "qwen_image_edit_2511",
            "qwen_image_edit_2511_lightning",
        ]),
        shape: qwen_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::SenseNovaEdit,
        models: ModelMatch::Any(&[
            "sensenova_u1_8b",
            "sensenova_u1_8b_infographic_v2",
            "sensenova_u1_8b_infographic_v3",
            "sensenova_u1_8b_fast",
            "sensenova_u1_8b_infographic_v2_fast",
            "sensenova_u1_8b_infographic_v3_fast",
        ]),
        shape: sensenova_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::ZImageEdit,
        models: ModelMatch::Any(&["z_image_turbo", "z_image_edit"]),
        shape: zimage_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::ZImageIdentity,
        models: ModelMatch::Any(&["z_image_turbo"]),
        shape: zimage_identity_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::IdeogramEdit,
        models: ModelMatch::Any(&["ideogram_4", "ideogram_4_turbo"]),
        shape: ideogram_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::IdeogramImg2Img,
        models: ModelMatch::Any(&["ideogram_4", "ideogram_4_turbo"]),
        shape: ideogram_img2img_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::BooguEdit,
        models: ModelMatch::Any(&["boogu_image_edit"]),
        shape: boogu_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::MageEdit,
        models: ModelMatch::Any(&[
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ]),
        shape: mage_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::BooguImg2Img,
        models: ModelMatch::Any(&["boogu_image", "boogu_image_turbo"]),
        shape: boogu_img2img_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::KreaEdit,
        models: ModelMatch::Any(&["krea_2_raw", "krea_2_turbo"]),
        shape: krea_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::BerniniEdit,
        models: ModelMatch::Any(&["bernini_image"]),
        shape: bernini_image_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::SdxlIpAdapter,
        models: ModelMatch::Family(is_sdxl_family_candle_model),
        shape: sdxl_ipadapter_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::KolorsIpAdapter,
        models: ModelMatch::Any(&["kolors"]),
        shape: kolors_ipadapter_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::FluxIpAdapter,
        models: ModelMatch::Any(&["flux_dev", "flux_schnell"]),
        shape: flux_ipadapter_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::QwenControl,
        models: ModelMatch::Any(&["qwen_image"]),
        shape: qwen_control_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::KolorsControl,
        models: ModelMatch::Any(&["kolors"]),
        shape: kolors_control_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::KolorsEdit,
        models: ModelMatch::Any(&["kolors"]),
        shape: kolors_edit_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::ZImageControl,
        models: ModelMatch::Any(&["z_image_turbo", "z_image"]),
        shape: zimage_control_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::ZImageImg2Img,
        models: ModelMatch::Any(&["z_image", "z_image_turbo"]),
        shape: zimage_img2img_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::Sd3Img2Img,
        models: ModelMatch::Family(is_sd3_family_candle_model),
        shape: sd3_img2img_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::SanaImg2Img,
        models: ModelMatch::Any(&["sana_1600m", "sana_sprint_1600m"]),
        shape: sana_img2img_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::Flux1Control,
        models: ModelMatch::Any(&["flux_dev"]),
        shape: flux1_control_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::Flux2Control,
        models: ModelMatch::Any(&["flux2_dev"]),
        shape: flux2_dev_control_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::KreaControl,
        models: ModelMatch::Any(&["krea_2_turbo"]),
        shape: krea_control_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::KreaImg2Img,
        models: ModelMatch::Any(&["krea_2_turbo", "krea_2_raw"]),
        shape: krea_img2img_candle_eligible,
    },
    CandleImageRoute {
        lane: CandleImageLane::Pulid,
        models: ModelMatch::Any(&["pulid_flux_dev"]),
        shape: pulid_flux_candle_eligible,
    },
];

#[cfg(test)]
pub(crate) fn candle_image_route_lanes() -> Vec<CandleImageLane> {
    CANDLE_IMAGE_ROUTES.iter().map(|route| route.lane).collect()
}

#[cfg(test)]
pub(crate) fn candle_pose_route_models() -> Vec<&'static str> {
    CANDLE_IMAGE_ROUTES
        .iter()
        .filter(|route| {
            matches!(
                route.lane,
                CandleImageLane::QwenControl
                    | CandleImageLane::KolorsControl
                    | CandleImageLane::ZImageControl
                    | CandleImageLane::Flux1Control
                    | CandleImageLane::Flux2Control
                    | CandleImageLane::KreaControl
            )
        })
        .flat_map(|route| match route.models {
            ModelMatch::Any(models) => models.iter().copied(),
            ModelMatch::Family(_) => [].iter().copied(),
        })
        .collect()
}

/// Resolve the first matching Candle image lane. The order is part of the scheduler contract:
/// specialized conditioned lanes precede the generic registry text-to-image lane.
pub(crate) fn image_job_candle_lane(job: &JobSnapshot) -> Option<CandleImageLane> {
    if !matches!(job.job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        return None;
    }
    let model = job.payload.get("model").and_then(Value::as_str)?;

    if imported_image_request_family_eligible(
        model,
        &job.payload,
        CANDLE_ROUTED_FAMILIES,
        CANDLE_IMPORTED_CAPS,
    ) {
        return Some(CandleImageLane::ImportedFamily);
    }

    CANDLE_IMAGE_ROUTES
        .iter()
        .find(|route| route.matches(model, &job.payload))
        .map(|route| route.lane)
        .or_else(|| {
            image_request_candle_eligible(model, &job.payload)
                .then_some(CandleImageLane::TextToImage)
        })
}

/// Does this image job belong on the Candle (Windows/CUDA) image lane?
pub(crate) fn image_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    image_job_candle_lane(job).is_some()
}

/// Per-model candle txt2img-eligibility, factored out of [`image_job_is_candle_eligible`] so the
/// routing tests can probe it with synthetic payloads (parity with `image_request_mlx_eligible`).
pub(crate) fn image_request_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if !CANDLE_ROUTED_MODELS.contains(&model) {
        return false;
    }
    // sc-18475: the SANA specialized route above owns the only accepted reference shape. If the
    // singular carrier is present but not a non-empty string, do not let it fall through as txt2img.
    if matches!(model, "sana_1600m" | "sana_sprint_1600m")
        && (payload
            .get("referenceAssetId")
            .is_some_and(|value| match value.as_str() {
                Some(id) => id.trim().is_empty(),
                None => true,
            })
            || sana_has_unsupported_carrier(payload))
    {
        return false;
    }
    // These families' reference-bearing modes are owned exclusively by specialized routes above.
    // If their carrier is blank/malformed or another shape check fails, never reinterpret the job as
    // registered text-to-image and silently drop the user's conditioning intent.
    let reference_only_mode = matches!(
        payload.get("mode").and_then(Value::as_str),
        Some("reference" | "image_to_image" | "character_image" | "style_variations")
    );
    if reference_only_mode
        && (matches!(
            model,
            "flux2_dev" | "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2"
        ) || matches!(
            model,
            "qwen_image_edit"
                | "qwen_image_edit_2509"
                | "qwen_image_edit_2511"
                | "qwen_image_edit_2511_lightning"
        ) || model.starts_with("sensenova_u1_8b"))
    {
        return false;
    }
    // Base (non-distilled, full-CFG) Z-Image txt2img (sc-8679, epic 8236): the candle `z_image` base
    // generator (shift-6.0 / ~50-step / real CFG) is now a candle txt2img provider (`is_candle_engine`),
    // so a plain (non-pose, non-edit) `z_image` job routes to the generic candle txt2img lane here — the
    // base sibling of `z_image_turbo`. Its strict-pose control (`advanced.poses`) is still branched out by
    // `zimage_control_candle_eligible` in `image_job_is_candle_eligible` BEFORE this gate; its edit shapes
    // are rejected below with every other family. (The prior sc-8379 guard that hard-rejected base z_image
    // here — because no candle txt2img provider existed — is retired now that one does.)
    // img2img / inpaint / outpaint all arrive as `mode == "edit_image"` (+ a source); reject the
    // whole edit family up front (the worker's `sdxl_sub_mode` keys off the same mode).
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    // Any conditioning asset (img2img source, IP-Adapter reference, or inpaint mask) is refused by
    // this base lane. Applies to EVERY candle family including Lens (pure T2I — no conditioning
    // shapes in the Lens port).
    if has_nonempty_string(payload, "sourceAssetId")
        || has_nonempty_string(payload, "referenceAssetId")
        || has_nonempty_string_array(payload, "referenceAssetIds")
        || has_nonempty_string(payload, "maskAssetId")
    {
        return false;
    }
    // Adapter-capable families are derived from the same audited capability table as quant support.
    // Some accept both adapters and Q4/Q8 (including Z-Image, Qwen, FLUX.2, SD3.5, Lens, Krea, and
    // SDXL), while others accept only adapters or only quant. A quant value may select a pre-packed
    // tier rather than request on-the-fly quantization; the gate intentionally covers both forms.
    // The two capabilities remain decoupled and fail closed through their separate derived lists.
    let supports_lora =
        CANDLE_QUANT_LORA_MODELS.contains(&model) || CANDLE_LORA_MODELS.contains(&model);
    let supports_quant =
        CANDLE_QUANT_LORA_MODELS.contains(&model) || CANDLE_QUANT_MODELS.contains(&model);
    // LoRAs: not in the candle lane unless the audited family row advertises adapters.
    if !supports_lora
        && payload
            .get("loras")
            .and_then(Value::as_array)
            .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    // Strict-pose ControlNet (`advanced.poses`, object-shaped entries) is refused by this base lane.
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses {
        return false;
    }
    if has_nonempty_nested_array(payload, "advanced", "phases") {
        return false;
    }
    // A quant/tier request (`advanced.mlxQuantize` > 0) is refused UNLESS the family advertises quant.
    // The sc-3675/sc-5096 candle providers advertise `supported_quants: &[]` (dense bf16/fp16 only), so
    // an explicit quant request can't be honored — refuse it rather than silently running dense
    // (sc-5099). Lens (sc-5126), SD3.5 (sc-7880), Krea (sc-9607/sc-9983), the Ideogram/Boogu packed
    // families (sc-9607), Qwen-Image (sc-11020), and Z-Image advertise Q4/Q8, so their quant requests
    // stay on candle. For the packed families
    // the `mlxQuantize` value is a turnkey tier-SELECT (which pre-quantized q4/q8 subdir to load), a no-op
    // on the loader rather than a runtime quantize — but the gate is the same: quant-capable → stay.
    if !supports_quant && candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// Whether the request explicitly asks for on-the-fly quantization the candle backend can't do.
/// `advanced.mlxQuantize` is an optional advanced override (the web UI doesn't send it; the MLX path
/// otherwise defaults quant from the manifest) — so a payload-level value `> 0` is a deliberate quant
/// request. `<= 0` (dense) and absent both leave candle on its native dense path (sc-5099).
pub(crate) fn candle_request_wants_quant(payload: &Map<String, Value>) -> bool {
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("mlxQuantize"))
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .is_some_and(|bits| bits > 0)
}

fn candle_request_wants_torch_quantization(payload: &Map<String, Value>) -> bool {
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("quantization"))
        .and_then(Value::as_str)
        .is_some_and(|value| {
            let value = value.trim();
            !value.is_empty() && !value.eq_ignore_ascii_case("auto")
        })
}

/// Does this video job belong on the candle video lane? The candle wan/ltx providers drive plain
/// text-to-video, the 14B I2V's single source-image conditioning (sc-5175), SVD image→video (sc-5493),
/// **and** the Wan-VACE advanced modes — replace_person / extend / bridge (sc-5494, the `PersonReplace`
/// / `VideoExtend` / `VideoBridge` job types → the candle `wan_vace` engine). Model-advertised user
/// LoRAs remain on the compatible base lane; unsupported conditioning, or an adapter on a provider
/// with no adapter slot, is refused here and remains queued for a capable native worker. SCAIL-2
/// (`scail2_14b`) adds a DISTINCT candle engine off-Mac —
/// `animate_character` + `replace_person` (sc-6837, epic 6563) — gated separately (it is not a VACE
/// model). Bernini (`bernini`) adds another DISTINCT candle engine off-Mac — t2v + the editing/
/// reference/multi-source video modes (sc-10997, epic 6562) — also gated separately. The per-model
/// shape gates are [`video_request_candle_eligible`] (base),
/// [`video_request_candle_vace_eligible`] (VACE modes), [`bernini_video_candle_eligible`] (Bernini),
/// and [`scail2_animate_candle_eligible`] / [`scail2_replace_candle_eligible`].
pub(crate) fn video_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    if candle_request_wants_torch_quantization(&job.payload) {
        // `advanced.quantization` selects a Torch GGUF overlay. Native Candle tiers use
        // `mlxQuantize`; accepting this field would silently discard the user's explicit choice.
        return false;
    }
    match job.job_type {
        // The base txt2video / image→video lane (sc-5097 / sc-5175 / sc-5493), plus SCAIL-2 standalone
        // character animation (`animate_character`, sc-6837 — a distinct candle engine, not VACE).
        JobType::VideoGenerate => {
            video_request_candle_eligible(model, &job.payload)
                || scail2_animate_candle_eligible(model, &job.payload)
                // Bernini (sc-10997, epic 6562): t2v + the editing/reference/multi-source modes on the
                // distinct candle `bernini` engine — its own gate (not the generic txt2video path).
                || bernini_video_candle_eligible(model, &job.payload)
        }
        // replace_person → candle Wan-VACE (sc-5494) OR candle SCAIL-2 (sc-6837, routed by model id).
        JobType::PersonReplace => {
            video_request_candle_vace_eligible(model, &job.payload, &job.job_type)
                || scail2_replace_candle_eligible(model, &job.payload)
                || ltx_replace_candle_eligible(model, &job.payload)
        }
        // extend_clip / video_bridge → candle Wan-VACE only (sc-5494).
        JobType::VideoExtend | JobType::VideoBridge => {
            video_request_candle_eligible(model, &job.payload)
                || video_request_candle_vace_eligible(model, &job.payload, &job.job_type)
        }
        _ => false,
    }
}

/// Per-model candle txt2video-eligibility, factored out so the routing tests can probe it with
/// synthetic payloads (parity with `image_request_candle_eligible`).
pub(crate) fn video_request_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if !CANDLE_VIDEO_ROUTED_MODELS.contains(&model) {
        return false;
    }
    if candle_request_wants_torch_quantization(payload) {
        return false;
    }
    if matches!(model, "ltx_2_3" | "ltx_2_3_eros") {
        match payload.get("mode").and_then(Value::as_str) {
            Some("text_to_video") => {
                if has_nonempty_string(payload, "sourceAssetId")
                    || has_nonempty_string(payload, "lastFrameAssetId")
                    || has_nonempty_string(payload, "sourceClipAssetId")
                    || has_nonempty_string(payload, "bridgeRightClipAssetId")
                {
                    return false;
                }
            }
            Some("image_to_video") => {
                if !has_nonempty_string(payload, "sourceAssetId")
                    || has_nonempty_string(payload, "lastFrameAssetId")
                    || has_nonempty_string(payload, "sourceClipAssetId")
                    || has_nonempty_string(payload, "bridgeRightClipAssetId")
                {
                    return false;
                }
            }
            Some("first_last_frame") => {
                if !has_nonempty_string(payload, "sourceAssetId")
                    || !has_nonempty_string(payload, "lastFrameAssetId")
                    || has_nonempty_string(payload, "sourceClipAssetId")
                    || has_nonempty_string(payload, "bridgeRightClipAssetId")
                {
                    return false;
                }
            }
            Some("extend_clip") => {
                if !has_nonempty_string(payload, "sourceClipAssetId")
                    || has_nonempty_string(payload, "bridgeRightClipAssetId")
                    || has_nonempty_string(payload, "sourceAssetId")
                    || has_nonempty_string(payload, "lastFrameAssetId")
                    || !payload
                        .get("loras")
                        .and_then(Value::as_array)
                        .is_some_and(|loras| crate::video_request::loras_contain_ltx_ic_lora(loras))
                {
                    return false;
                }
            }
            Some("video_bridge") => {
                if !has_nonempty_string(payload, "sourceClipAssetId")
                    || !has_nonempty_string(payload, "bridgeRightClipAssetId")
                    || has_nonempty_string(payload, "sourceAssetId")
                    || has_nonempty_string(payload, "lastFrameAssetId")
                    || !payload
                        .get("loras")
                        .and_then(Value::as_array)
                        .is_some_and(|loras| crate::video_request::loras_contain_ltx_ic_lora(loras))
                {
                    return false;
                }
            }
            _ => return false,
        }
    } else if model == "wan_2_2" {
        match payload.get("mode").and_then(Value::as_str) {
            Some("text_to_video") => {
                if has_nonempty_string(payload, "sourceAssetId")
                    || has_nonempty_string(payload, "lastFrameAssetId")
                {
                    return false;
                }
            }
            Some("image_to_video") => {
                if !has_nonempty_string(payload, "sourceAssetId")
                    || has_nonempty_string(payload, "lastFrameAssetId")
                {
                    return false;
                }
            }
            Some("first_last_frame") => {
                if !has_nonempty_string(payload, "sourceAssetId")
                    || !has_nonempty_string(payload, "lastFrameAssetId")
                {
                    return false;
                }
            }
            _ => return false,
        }
        if has_nonempty_string(payload, "sourceClipAssetId")
            || has_nonempty_string(payload, "bridgeRightClipAssetId")
        {
            return false;
        }
    } else if CANDLE_VIDEO_I2V_ROUTED_MODELS.contains(&model) {
        // Wan 14B I2V is image→video ONLY (sc-5175): require the `image_to_video` mode + a source
        // image. A txt2video shape (no source) is rejected and remains queued.
        if payload.get("mode").and_then(Value::as_str) != Some("image_to_video") {
            return false;
        }
        if !has_nonempty_string(payload, "sourceAssetId") {
            return false;
        }
    } else {
        // txt2video only: the base `video_generate` mode defaults to `image_to_video`, so require an
        // explicit `text_to_video`. Every conditioned mode (i2v / first_last_frame / extend / bridge /
        // replace) is thereby excluded, as is a stray source image.
        if payload.get("mode").and_then(Value::as_str) != Some("text_to_video") {
            return false;
        }
        if has_nonempty_string(payload, "sourceAssetId") {
            return false;
        }
    }
    // Reference / inpaint-mask conditioning is never in the candle video lane (i2v needs only the
    // single source image; reference + mask are unsupported character / inpaint shapes).
    if has_nonempty_string(payload, "referenceAssetId")
        || has_nonempty_string(payload, "maskAssetId")
    {
        return false;
    }
    // User LoRAs on the candle video lane are gated by the provider descriptor. Wan-5B/14B and LTX
    // apply each `request.loras` entry from its file path. SVD and Mochi still reject rather than
    // silently dropping an adapter.
    if !CANDLE_VIDEO_LORA_MODELS.contains(&model)
        && payload
            .get("loras")
            .and_then(Value::as_array)
            .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    // `advanced.mlxQuantize` is a tier select for the published Wan q4/q8/bf16 matrices and for
    // LTX base's shared packed-q4 turnkey. Other video providers remain dense and fail closed.
    if candle_request_wants_quant(payload) && !candle_video_tier_select_eligible(model, payload) {
        return false;
    }
    true
}

fn candle_video_tier_select_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if matches!(model, "wan_2_2" | "wan_2_2_t2v_14b" | "wan_2_2_i2v_14b") {
        return true;
    }
    model == "ltx_2_3"
        && payload
            .get("advanced")
            .and_then(Value::as_object)
            .and_then(|advanced| advanced.get("mlxQuantize"))
            .and_then(|value| {
                value
                    .as_i64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
            })
            == Some(4)
}

/// Native LTX replace-person eligibility. Unlike the historical generic Wan-VACE substitution, this
/// route keeps the selected LTX/Eros model and forwards the tracked clip, masks, references, and
/// selected IC-LoRA to that model's `ControlClip`/keyframe-append provider path.
pub(crate) fn ltx_replace_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    // SC-18902 withdrew the Eros Candle route after the exact-head CUDA render proved that its
    // undistilled checkpoint is not compatible with Candle's single-pass distilled recipe. Keep
    // this newer advanced-mode route aligned with the same product decision as the base lane.
    if model != "ltx_2_3"
        || !has_nonempty_string(payload, "sourceClipAssetId")
        || !has_nonempty_string(payload, "personTrackId")
        || !has_nonempty_string(payload, "characterId")
        || !payload
            .get("loras")
            .and_then(Value::as_array)
            .is_some_and(|loras| crate::video_request::loras_contain_ltx_ic_lora(loras))
    {
        return false;
    }
    !candle_request_wants_quant(payload) || candle_video_tier_select_eligible(model, payload)
}

/// Candle Wan-VACE eligibility for the advanced video job types (sc-5494): `PersonReplace`
/// (replace_person), `VideoExtend` (extend_clip), `VideoBridge` (video_bridge). Routes to the candle
/// `wan_vace` engine when the model is VACE-capable and the per-mode source assets are present. The
/// dedicated VACE-Fun provider is admitted here only for PersonReplace and accepts user adapters;
/// generic VACE still rejects them. Factored out so the routing tests can probe it with synthetic
/// payloads (parity with [`video_request_candle_eligible`]).
pub(crate) fn video_request_candle_vace_eligible(
    model: &str,
    payload: &Map<String, Value>,
    job_type: &JobType,
) -> bool {
    if model == "wan_2_2_vace_fun_14b" {
        if !matches!(job_type, JobType::PersonReplace) {
            return false;
        }
    } else if !CANDLE_VIDEO_VACE_MODELS.contains(&model) {
        return false;
    }
    match job_type {
        // replace_person: the source control clip + the tracked person + the character references.
        JobType::PersonReplace => {
            if !has_nonempty_string(payload, "sourceClipAssetId")
                || !has_nonempty_string(payload, "personTrackId")
                || !has_nonempty_string(payload, "characterId")
            {
                return false;
            }
        }
        // extend_clip: the source clip whose tail anchors the continuation.
        JobType::VideoExtend => {
            if !has_nonempty_string(payload, "sourceClipAssetId") {
                return false;
            }
        }
        // video_bridge: both clips (the left tail + the right head) are pinned around the gap.
        JobType::VideoBridge => {
            if !has_nonempty_string(payload, "sourceClipAssetId")
                || !has_nonempty_string(payload, "bridgeRightClipAssetId")
            {
                return false;
            }
        }
        _ => return false,
    }
    // Only the dedicated VACE-Fun provider accepts adapters. On-the-fly quant remains unsupported.
    if model != "wan_2_2_vace_fun_14b" && has_nonempty_array(payload, "loras") {
        return false;
    }
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// Candle SCAIL-2 `animate_character` eligibility (sc-6837, epic 6563). SCAIL-2 is a DISTINCT candle
/// engine (NOT Wan-VACE), so it has its own gate rather than membership in [`CANDLE_VIDEO_VACE_MODELS`]:
/// the `scail2_14b` model + the `animate_character` mode + a reference character image
/// (`referenceAssetId` / `referenceAssetIds` / `sourceAssetId`) + a driving clip (`sourceClipAssetId`).
/// Inference LoRA / LoKr / LoHa + the Bias-Aware DPO LoRA + the lightx2v lightning diff-patch ARE on the
/// candle path now (sc-6838 — the provider merges them into the dense DiT), so a LoRA-bearing animate job
/// stays on candle. On-the-fly quantization is refused and remains queued (the provider is dense). Mirrors the
/// MLX `video_mode_is_mlx_eligible(scail2_14b, animate_character)` shape, expressed as a candle-claim
/// gate. Factored out so the routing tests can probe it (parity with [`video_request_candle_eligible`]).
pub(crate) fn scail2_animate_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if model != "scail2_14b" {
        return false;
    }
    if payload.get("mode").and_then(Value::as_str) != Some("animate_character") {
        return false;
    }
    let has_reference = has_nonempty_string(payload, "referenceAssetId")
        || has_nonempty_string(payload, "sourceAssetId")
        || has_nonempty_string_array(payload, "referenceAssetIds");
    if !has_reference {
        return false;
    }
    if !has_nonempty_string(payload, "sourceClipAssetId") {
        return false;
    }
    // Inference LoRA (DPO / lightning / user adapter) merges into the candle DiT (sc-6838), so a
    // LoRA-bearing animate job is candle-eligible — only on-the-fly quant is still refused.
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// Candle SCAIL-2 `replace_person` eligibility (sc-6837, epic 6563). The `scail2_14b` model behind a
/// `PersonReplace` job: the source control clip + the tracked person + the character references (the
/// same per-mode assets the Wan-VACE replace gate requires). Inference adapters use the same provider
/// seam as standalone animation (LoRA / LoKr / LoHa / diff-patch); only on-the-fly quant remains
/// unsupported. A distinct candle engine, so it is gated here rather than
/// added to [`CANDLE_VIDEO_VACE_MODELS`]. Factored out so the routing tests can probe it.
pub(crate) fn scail2_replace_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if model != "scail2_14b" {
        return false;
    }
    if !has_nonempty_string(payload, "sourceClipAssetId")
        || !has_nonempty_string(payload, "personTrackId")
        || !has_nonempty_string(payload, "characterId")
    {
        return false;
    }
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// Candle Bernini VIDEO eligibility (sc-10997, epic 6562). Bernini is a DISTINCT candle engine (the
/// full Qwen planner + Wan2.2-T2V-A14B renderer, `gen_core::load("bernini")`), so it has its own gate
/// rather than routing through the generic [`video_request_candle_eligible`] txt2video path — that
/// path only admits `text_to_video`, but Bernini also serves the editing/reference/multi-source modes.
/// Mirrors the MLX `video_mode_is_mlx_eligible(bernini, mode)` shape (mode-only, `bernini` id + the six
/// served modes), expressed as a candle-claim gate: `text_to_video` (base), `video_to_video` (v2v edit),
/// `reference_to_video` (r2v), `reference_video_to_video` (rv2v), `multi_video_to_video` (mv2v), and
/// `ads2v`. Routed on the model id + mode, not weight availability — the worker's dedicated
/// `CandleVideoRoute::Bernini` dispatch resolves-or-errors loudly if the `SceneWorks/bernini`
/// snapshot is unprovisioned (sc-11003), and validates the per-mode source media when it assembles the
/// conditioning. User LoRA/LoKr applies to the renderer's high/low experts; an explicit `mlxQuantize` remains
/// on Candle because the worker resolves the published bf16/q8/q4 tier subdirectories (sc-11003) —
/// there is no torch Bernini to fall back to. Factored out so
/// the routing tests can probe it with synthetic payloads (parity with [`video_request_candle_eligible`]).
pub(crate) fn bernini_video_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if model != "bernini" {
        return false;
    }
    matches!(
        payload.get("mode").and_then(Value::as_str),
        Some(
            "text_to_video"
                | "video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "multi_video_to_video"
                | "ads2v"
        )
    )
}

/// InstantID candle-routing conditions (sc-5491, epic 5480). The candle `candle-gen-instantid`
/// provider is the off-Mac sibling of `mlx-gen-instantid` and serves the IDENTICAL surface (single
/// identity, the angle set, pose-library mode, face-restore via `generate_pose` / `restore_face`), so
/// the gate is the same as [`instantid_mlx_eligible`]: a `character_image` job with a reference face.
/// Mirrors the candle worker's `instantid_available` gate so the router and worker agree.
pub(crate) fn instantid_candle_eligible(payload: &Map<String, Value>) -> bool {
    instantid_mlx_eligible(payload)
}

/// PuLID-FLUX candle-routing conditions (sc-5492, epic 5480). The candle `candle-gen-pulid` provider is
/// the off-Mac sibling of `mlx-gen-pulid` and serves the IDENTICAL surface (a `character_image` job with
/// a reference face → the PuLID identity injection on FLUX.1-dev), so the gate is the same as
/// [`pulid_flux_mlx_eligible`]. Mirrors the candle worker's `pulid_candle_available` gate so the router
/// and worker agree. `pulid_flux_dev` is a distinct model id (not `flux_dev`), so this never collides
/// with the FLUX XLabs IP-Adapter lane.
pub(crate) fn pulid_flux_candle_eligible(payload: &Map<String, Value>) -> bool {
    pulid_flux_mlx_eligible(payload)
}

/// The SDXL-family model ids whose conditioning shapes have a bespoke candle lane (edit + IP-Adapter).
///
/// NOT every id on the `sdxl` engine: `realvisxl_lightning` is txt2img-only (its accel sampler is
/// engine-incompatible with reference/img2img conditioning) and `instantid_realvisxl` has its own
/// bespoke lane. Must stay in lockstep with the worker's `is_sdxl_edit_candle_model` /
/// `is_sdxl_ipadapter_model` — a model the router sends to a lane the worker then rejects fails the
/// job rather than falling back.
pub(crate) fn is_sdxl_family_candle_model(model: &str) -> bool {
    matches!(
        model,
        "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2"
    )
}

/// SDXL img2img / inpaint / outpaint candle-routing conditions (sc-5487, epic 5480). The candle
/// `SdxlEdit` provider serves `edit_image` mode with a `sourceAssetId` on the sdxl family: img2img (no
/// mask), inpaint (+ `maskAssetId`), and outpaint (`fit_mode == "outpaint"`) all route to the one lane.
/// Disjoint from the IP-Adapter lane (which is `referenceAssetId` and NOT `edit_image`). Mirrors the
/// worker's `sdxl_edit_candle_available` gate (minus the local weight-resolve check) so the router and
/// worker agree. Candle-only — macOS keeps the MLX `SdxlSubMode::{Edit,Inpaint,Outpaint}` path.
pub(crate) fn sdxl_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "sourceAssetId")
}

/// FLUX.2 reference-route candle conditions. Edit/reference, character, and style requests all
/// use native token-concat conditioning and must carry a concrete source/reference id.
/// Mirrors the worker's `flux2_edit_candle_available` gate minus local weight resolution.
pub(crate) fn flux2_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    let mode = payload.get("mode").and_then(Value::as_str);
    if !matches!(
        mode,
        Some(
            "edit_image" | "reference" | "image_to_image" | "character_image" | "style_variations"
        )
    ) {
        return false;
    }
    conditioned_reference_count(
        payload,
        matches!(mode, Some("edit_image" | "image_to_image")),
        4,
    )
    .is_some()
        && !conditioned_edit_has_unsupported_carrier(payload, true, false, false)
        && !conditioned_true_cfg_is_malformed(payload)
}

fn flux2_dev_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    flux2_edit_candle_eligible(payload)
}

/// Qwen-Image-Edit candle-routing conditions (sc-5487, sc-18476). The candle `QwenEdit` provider
/// serves instruction edit plus ordered singular/plural reference and character/angle workflows on
/// the Qwen-Image-Edit family. Character Studio pose sets are supported only with exactly one identity
/// reference: each pose skeleton is appended as the second ordered edit reference. Masks, controls,
/// phases, malformed pose sets/CFG, and conflicting reference carriers are rejected rather than
/// dropped. Mirrors the worker's
/// `qwen_edit_candle_available` gate (minus local weight resolution); macOS keeps the corresponding
/// MLX `qwen_image_edit` path, including its pre-existing best-effort pose grouping.
pub(crate) fn qwen_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    let mode = payload.get("mode").and_then(Value::as_str);
    if !matches!(mode, Some("edit_image" | "character_image")) {
        return false;
    }
    let Some(reference_count) = conditioned_reference_count(payload, mode == Some("edit_image"), 5)
    else {
        return false;
    };
    let Some(pose_count) = strict_candle_pose_count(payload) else {
        return false;
    };
    (pose_count == 0 || (mode == Some("character_image") && reference_count == 1))
        && !conditioned_edit_has_unsupported_carrier(payload, true, false, true)
        && !conditioned_true_cfg_is_malformed(payload)
}

/// Strict Character Studio pose carrier shared by the newly conditioned Candle routes.
/// Missing/null/empty means no pose set; a supplied set must contain only objects and stay within the
/// API-wide pose bound. Workers parse those same objects with the production whole-body renderer.
fn strict_candle_pose_count(payload: &Map<String, Value>) -> Option<usize> {
    match payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
    {
        None | Some(Value::Null) => Some(0),
        Some(Value::Array(poses))
            if poses.len() <= MAX_JOB_POSES && poses.iter().all(Value::is_object) =>
        {
            Some(poses.len())
        }
        Some(_) => None,
    }
}

/// SenseNova-U1's registered Candle generator accepts structural Reference/MultiReference
/// conditioning plus image true-CFG. Unsupported blend-strength, control, mask, adapter, pose, and
/// phase carriers are rejected here so they can never reach the generic registered stream as T2I.
pub(crate) fn sensenova_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    let mode = payload.get("mode").and_then(Value::as_str);
    if !matches!(mode, Some("edit_image" | "character_image")) {
        return false;
    }
    conditioned_reference_count(payload, mode == Some("edit_image"), 5).is_some()
        && !conditioned_edit_has_unsupported_carrier(payload, false, true, false)
        && !conditioned_true_cfg_is_malformed(payload)
}

fn conditioned_true_cfg_is_malformed(payload: &Map<String, Value>) -> bool {
    ["trueCfgScale", "imageGuidanceScale"]
        .iter()
        .any(|key| has_malformed_optional_nested_number(payload, "advanced", key))
}

fn conditioned_edit_has_unsupported_carrier(
    payload: &Map<String, Value>,
    allow_loras: bool,
    reject_strength: bool,
    allow_poses: bool,
) -> bool {
    (!allow_loras && has_nonempty_or_malformed_array(payload, "loras"))
        || ["controls", "controlnets"]
            .iter()
            .any(|key| has_nonempty_or_malformed_array(payload, key))
        || ["maskAssetId"]
            .iter()
            .any(|key| has_nonempty_or_malformed_string(payload, key))
        || (!allow_poses && has_nonempty_or_malformed_nested_array(payload, "advanced", "poses"))
        || has_nonempty_or_malformed_nested_array(payload, "advanced", "phases")
        || [
            "controlMode",
            "controlImage",
            "controlScale",
            "controlWeights",
            "convRot",
            "quantTier",
        ]
        .iter()
        .any(|key| has_nonnull_or_malformed_nested_carrier(payload, "advanced", key))
        || (reject_strength
            && ["strength", "ipAdapterScale", "referenceStrength"]
                .iter()
                .any(|key| has_nonnull_or_malformed_nested_carrier(payload, "advanced", key)))
        || (reject_strength
            && ["strength", "referenceStrength"]
                .iter()
                .any(|key| payload.get(*key).is_some_and(|value| !value.is_null())))
}

/// Z-Image img2img / edit candle-routing conditions (sc-6595, epic 5480). The candle `ZImageEdit`
/// provider serves `edit_image` mode with a `sourceAssetId` on the z-image family — the Turbo weights'
/// img2img path (no mask / inpaint / outpaint). Same payload predicate as the other edit gates, gated to
/// the z-image family (`z_image_turbo` + the dedicated `z_image_edit` id) by the caller. Mirrors the
/// worker's `zimage_edit_candle_available` gate (minus the local weight-resolve check) so the router and
/// worker agree. Candle-only — macOS keeps the MLX `z_image_turbo` registry generator's `Reference` path.
pub(crate) fn zimage_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "sourceAssetId")
}

/// Krea 2 Kontext-style dual-conditioned image-edit candle-routing conditions (epic 10871). The bespoke
/// candle Krea edit lane (`generate_candle_krea_edit_stream`) serves `edit_image` mode on `krea_2_raw` —
/// the conditioning image rides as in-context VAE tokens AND grounds the Qwen3-VL vision tower, requiring
/// the `krea2_identity_edit` LoRA (checked worker-side, R5). The image can arrive as the two-reference
/// scene+person set (`referenceAssetIds`, scene = image 1, person = image 2, `sourceAssetId` null) or a
/// plain `sourceAssetId` — the same fields the worker's `krea_edit_candle_reference_ids` resolves (no
/// singular `referenceAssetId`, unlike the MLX lane). Gating on `sourceAssetId` alone stranded the two-ref
/// form (the candle worker owns the lane with no fallback), the off-Mac twin of the MLX
/// `krea_mlx_eligible` bug. Gated to `krea_2_raw` by the caller. Mirrors the worker's
/// `krea_edit_candle_available` gate (minus the local weight-resolve check) so the router and worker
/// agree. Candle-only — macOS keeps the MLX `krea_2_edit` registry generator's edit path (`krea_edit.rs`).
pub(crate) fn krea_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    conditioned_reference_count(payload, true, 2).is_some()
        && !krea_edit_has_unsupported_carrier(payload)
}

/// Krea 2 img2img (reference-guided latent-init) candle-routing conditions (sc-10134 Turbo, sc-10226 Raw;
/// epic 8588). The candle img2img lane serves a `krea_2_turbo` / `krea_2_raw` job in a **non-edit** mode
/// carrying a `referenceAssetId` — the "Start from an image" tile: the reference is VAE-encoded and blended
/// into the init latent at `advanced.strength`, then the denoise runs from `sigmas[start]` (CFG-free
/// `render_img2img` for Turbo, two-forward CFG `render_base_img2img` for Raw). Distinct from the Krea
/// **edit** lane (`edit_image` mode + the Kontext dual-conditioning, `krea_edit_candle_eligible`) and the
/// pose-control lane (`advanced.poses`, branched first). Gated to each id by the caller (Turbo + Raw have
/// separate branches so the precedence comments stay per-id). Mirrors the worker's img2img resolve in
/// `generate_candle_stream` (the `model_supports_img2img` + `resolve_img2img_init_generic` path).
pub(crate) fn krea_img2img_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
}

/// Z-Image img2img (reference-guided latent-init) candle-routing conditions — the registered `z_image`
/// **base** (sc-10265) and `z_image_turbo` (sc-11783), epic 8588. Both candle generators serve registry
/// img2img: a single `Conditioning::Reference` in a non-edit request VAE-encodes to the clean init latent
/// and denoises the reduced schedule tail (`candle-gen-z-image` `resolve_reference` + `init_time_step` —
/// base's `render_base` sc-8646, Turbo's `render` sc-11783). So a `z_image` / `z_image_turbo` job in a
/// non-edit mode with a `referenceAssetId` is candle-eligible; the worker resolves the init generically
/// (`model_supports_img2img`, sc-10134). Same payload shape as [`krea_img2img_candle_eligible`], gated to
/// the z-image ids by the caller (each has its own branch so the precedence comments stay per-id). NOT the
/// identity-init (`character_image` + `referenceStrength`), the `edit_image` masked-edit, or the
/// pose-control (`advanced.poses`) shapes — all branched first.
pub(crate) fn zimage_img2img_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
}

/// The three registered candle SD3.5 txt2img ids (`candle-gen-sd3`): Large (CFG), Large Turbo
/// (distilled), Medium (MMDiT-X CFG). Shared by the SD3.5 img2img branch (sc-11784) so the id set has
/// one home; each also rides the generic txt2img gate below (all three are in `CANDLE_ROUTED_MODELS`).
pub(crate) fn is_sd3_family_candle_model(model: &str) -> bool {
    matches!(model, "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium")
}

/// SD3.5 img2img (reference-guided latent-init) candle-routing conditions (sc-11784, epic 8588). The
/// registered `sd3_5_*` candle generators serve registry img2img — a single `Conditioning::Reference`
/// in a non-edit request VAE-encodes to the clean init latent and denoises the reduced schedule tail
/// (`candle-gen-sd3` `resolve_reference` + `init_time_step` + `render`, candle-gen #493; real CFG for
/// Large/Medium, the distilled loop for Turbo). So an SD3.5 job in a non-edit mode with a
/// `referenceAssetId` is candle-eligible; the worker resolves the init generically
/// (`model_supports_img2img`, sc-10134). Same payload shape as [`zimage_img2img_candle_eligible`], gated
/// to the SD3.5 ids by the caller. SD3.5 has no candle identity / `edit_image` / pose-control lane, so
/// the only reference shape on these ids is img2img — no earlier branch to keep precedence behind.
pub(crate) fn sd3_img2img_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
}

/// SANA base/Sprint non-edit img2img: exactly one singular reference, with no edit/control/adapter
/// carrier that the generic worker path would otherwise drop. Candle has no Q4/Q8 or LoRA surface for
/// these dense snapshots, so those request shapes remain unclaimed.
pub(crate) fn sana_img2img_candle_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
        && has_nonempty_string(payload, "referenceAssetId")
        && !sana_has_unsupported_carrier(payload)
}

/// SANA consumes only one singular reference. Optional empty/null carriers mean "absent"; any
/// non-empty unsupported or malformed carrier must not fall through to the generic txt2img lane,
/// whose worker request type would otherwise ignore it.
fn sana_has_unsupported_carrier(payload: &Map<String, Value>) -> bool {
    ["referenceAssetIds", "controls", "controlnets", "loras"]
        .iter()
        .any(|key| has_nonempty_or_malformed_array(payload, key))
        || ["sourceAssetId", "maskAssetId"]
            .iter()
            .any(|key| has_nonempty_or_malformed_string(payload, key))
        || ["poses", "phases"]
            .iter()
            .any(|key| has_nonempty_or_malformed_nested_array(payload, "advanced", key))
        || [
            "controlMode",
            "controlImage",
            "controlScale",
            "controlWeights",
            "convRot",
            "quantTier",
        ]
        .iter()
        .any(|key| has_nonnull_or_malformed_nested_carrier(payload, "advanced", key))
        || sana_has_unsupported_quant_carrier(payload)
}

fn sana_has_unsupported_quant_carrier(payload: &Map<String, Value>) -> bool {
    let Some(value) = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("mlxQuantize"))
    else {
        return false;
    };
    if value.is_null() {
        return false;
    }
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|bits| bits.trim().parse().ok()))
        .map_or(true, |bits| bits > 0)
}

/// Ideogram 4 img2img / Remix + mask inpaint / outpaint edit candle-routing conditions (sc-6598, epic
/// 6561). The candle `candle-gen-ideogram` provider serves `edit_image` mode with a `sourceAssetId` on
/// the ideogram family — img2img/Remix (source `Reference`), masked inpaint (`+ maskAssetId`), and
/// outpaint (`fit_mode == "outpaint"`, the worker synthesizes the border mask) all require a source.
/// Same payload predicate as the other edit gates (an optional mask / outpaint is resolved worker-side
/// in `resolve_ideogram_edit`). Gated to the ideogram family by the caller. The candle lane reuses the
/// generic `generate_candle_stream` (same engine as T2I), so there is no separate worker `*_available`
/// gate to mirror — the worker's `is_candle_engine` + in-lane edit resolve cover both. Candle-only —
/// macOS keeps the MLX `ideogram_4` registry generator's edit path (sc-6303).
pub(crate) fn ideogram_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "sourceAssetId")
}

/// Ideogram 4 / Turbo img2img (reference-guided latent-init) candle-routing conditions (sc-10261, epic
/// 8588). The registered `ideogram_4` / `ideogram_4_turbo` candle generators serve the `ui.img2img`
/// "Image reference" tile: a single `Conditioning::Reference` (no `Mask`) in a **non-edit** request is
/// VAE-encoded to the clean init latent and denoised from the strength-derived step to the schedule tail
/// (`candle-gen-ideogram` `resolve_edit` → `prepare_edit` with `mask = None`, sc-6598). So an ideogram
/// job in a non-edit mode with a `referenceAssetId` is candle-eligible; the worker resolves the init
/// generically (`model_supports_img2img`, sc-10134). Same payload predicate as
/// [`boogu_img2img_candle_eligible`] / [`sd3_img2img_candle_eligible`], gated to the ideogram ids by the
/// caller. DISJOINT from the [`ideogram_edit_candle_eligible`] Remix/inpaint lane (that arm is
/// `edit_image` + `sourceAssetId`). Mirrors the mlx generic img2img arm (sc-10192).
pub(crate) fn ideogram_img2img_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
}

/// Boogu instruction-edit candle-routing conditions (sc-7524, epic 6831). The candle `boogu_image_edit`
/// engine serves `edit_image` mode with a `sourceAssetId` — a single-reference instruction TI2I (the
/// source is VAE-encoded into the DiT reference latent AND read by the Qwen3-VL vision tower; no mask /
/// inpaint / outpaint, the descriptor accepts only `Reference`). Same payload predicate as the other edit
/// gates, gated to `boogu_image_edit` by the caller (only the Edit checkpoint edits — Base/Turbo are
/// T2I-only). Like Ideogram, the candle lane reuses the generic `generate_candle_stream` (the source is
/// resolved in-lane by `resolve_boogu_edit`), so there is no separate worker `*_available` gate to mirror
/// — the worker's `is_candle_engine` + in-lane edit resolve cover it. Candle-only — macOS keeps the MLX
/// `boogu_image_edit` registry generator's edit path.
pub(crate) fn boogu_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    // One source: the single `sourceAssetId`, or the plural `referenceAssetIds` multi-image picker
    // (sc-7645 — the Boogu DiT packs up to 5 references). Either routes the edit to candle.
    has_nonempty_string(payload, "sourceAssetId")
        || has_nonempty_string_array(payload, "referenceAssetIds")
}

/// Mage-Flow instruction-edit routing conditions (sc-14053). Every edit checkpoint accepts one
/// required primary `sourceAssetId` followed by optional ordered `referenceAssetIds`; the worker
/// turns that exact list into `Reference` / `MultiReference` conditioning for the registry generator.
pub(crate) fn mage_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// Boogu Base/Turbo img2img (reference-guided latent-init) candle-routing conditions (sc-11786, epic
/// 8588). The registered `boogu_image` (Base) and `boogu_image_turbo` candle generators serve registry
/// img2img: a single `Conditioning::Reference` in a **non-edit** request VAE-encodes to the clean init
/// latent and denoises the reduced schedule tail (`candle-gen-boogu` `resolve_reference` +
/// `init_time_step` — `render_base` / `render_turbo`). So a Base/Turbo job in a non-edit mode with a
/// `referenceAssetId` is candle-eligible; the worker resolves the init generically
/// (`model_supports_img2img`, sc-10134). Same payload predicate as [`zimage_img2img_candle_eligible`],
/// gated to the Boogu Base/Turbo ids by the caller. DISJOINT from the `boogu_image_edit` multi-reference
/// instruction-edit lane (`boogu_edit_candle_eligible`, a different engine id + `edit_image` mode + the
/// Qwen3-VL vision tower). Mirrors the mlx-gen-boogu img2img (sc-10191).
pub(crate) fn boogu_img2img_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
}

/// Bernini still-image i2i candle-routing conditions (sc-10996, epic 6562). The candle
/// `candle-gen-bernini` still lane serves `edit_image` mode with a `sourceAssetId` — the source is fed
/// to the engine as `Conditioning::Reference` (the planner ViT/VAE-encodes it into a structural
/// re-render; the reference strength is ignored). Same payload predicate as the other edit gates (a
/// `sourceAssetId` is required — an `edit_image` job with nothing to edit stays off candle), gated to
/// `bernini_image` by the caller. Unlike Ideogram/Boogu (which edit on their SAME txt2img engine in the
/// generic stream), Bernini has a DEDICATED worker stream (`generate_candle_bernini_image_stream`,
/// `frames:1`) — but the routing predicate is identical. The exact candle twin of the MLX
/// `bernini_image_mlx_eligible` (mlx.rs).
pub(crate) fn bernini_image_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "sourceAssetId")
}

/// SDXL IP-Adapter-Plus candle-routing conditions (sc-5488, epic 5480). The candle `IpAdapterSdxl`
/// provider serves PURE reference (image-prompt) conditioning on the sdxl family: a `referenceAssetId`
/// with NO img2img source / inpaint mask and NOT an `edit_image` (that advanced SDXL shape is the
/// sc-5487 `SdxlEdit` lane). Mirrors the worker's `sdxl_ipadapter_available` gate (minus the local
/// weight-resolve check) so the router and worker agree on the lane boundary. Candle-only — there is no
/// MLX `IpAdapterSdxl` (the MLX SDXL IP path is the registry `SdxlSubMode::Ip`), so this has no
/// `*_mlx_eligible` sibling.
pub(crate) fn sdxl_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
        && !has_nonempty_string(payload, "sourceAssetId")
        && !has_nonempty_string(payload, "maskAssetId")
}

/// Kolors IP-Adapter-Plus candle-routing conditions (sc-5488, epic 5480). The candle `IpAdapterKolors`
/// provider serves PURE reference (image-prompt) conditioning on the `kolors` family — the same payload
/// shape as the SDXL IP lane: a `referenceAssetId` with NO img2img source / inpaint mask and NOT an
/// `edit_image` (those advanced Kolors shapes are refused and remain queued). Mirrors the worker's
/// `kolors_ipadapter_available` gate (minus the local weight-resolve check) so the router and worker
/// agree on the lane boundary. Candle-only — the macOS Kolors IP path is the registry `Reference` route,
/// not a separate candle-eligible gate.
pub(crate) fn kolors_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
        && !has_nonempty_string(payload, "sourceAssetId")
        && !has_nonempty_string(payload, "maskAssetId")
}

/// FLUX XLabs IP-Adapter candle-routing conditions (sc-5872, epic 5480). The candle `IpAdapterFlux`
/// provider serves PURE reference (image-prompt) conditioning on the `flux_dev`/`flux_schnell` families
/// — the same payload shape as the SDXL/Kolors IP lanes: a `referenceAssetId` with NO img2img source /
/// inpaint mask and NOT an `edit_image` (those advanced FLUX shapes are refused and remain queued). Mirrors
/// the worker's `flux_ipadapter_available` gate (minus the local weight-resolve check) so the router and
/// worker agree on the lane boundary. Candle-only — the macOS FLUX IP path is the registry `Reference`
/// route (epic 3621), not a separate candle-eligible gate.
pub(crate) fn flux_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_string(payload, "referenceAssetId")
        && !has_nonempty_string(payload, "sourceAssetId")
        && !has_nonempty_string(payload, "maskAssetId")
}

/// Qwen-Image strict-pose ControlNet candle-routing conditions (sc-5489, epic 5480). The candle
/// `QwenControl` provider serves `qwen_image` + a non-empty object `advanced.poses` (one image per pose,
/// each conditioned on a DWPose skeleton), NOT an `edit_image`. A `referenceAssetId`, if present, is
/// ignored (identity comes from a character LoRA on the base, mirroring the MLX/torch
/// `QwenImageControlNetPipeline`). Mirrors the worker's `qwen_control_available` gate (minus the local
/// weight-resolve check) so the router and worker agree. Candle-only — the macOS path is the registry
/// `qwen_image_control` generator, not a separate candle-eligible gate.
pub(crate) fn qwen_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_nested_array(payload, "advanced", "poses")
}

/// Kolors strict-pose ControlNet candle-routing conditions (sc-5489, epic 5480). The candle
/// `KolorsControl` provider serves `kolors` + a non-empty `advanced.poses` (one image per pose, each
/// conditioned on a DWPose skeleton via the `Kwai-Kolors/Kolors-ControlNet-Pose` branch), NOT an
/// `edit_image`. Same shape as `qwen_control_candle_eligible` — the model gate (`kolors`) is applied at
/// the call site. Mirrors the worker's `kolors_control_available` gate (minus the local weight-resolve
/// check) so the router and worker agree. Candle-only — the macOS path is the MLX Kolors ControlNet.
pub(crate) fn kolors_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_nested_array(payload, "advanced", "poses")
}

/// Kolors source-image img2img/edit through the registered Candle generator. The existing pure
/// `referenceAssetId` IP-Adapter route and pose ControlNet route are intentionally separate and
/// precede this route. Only a singular `sourceAssetId` is accepted here; plural/reference/mask/
/// control carriers fail closed rather than being discarded by the generic stream.
pub(crate) fn kolors_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) == Some("edit_image")
        && has_nonempty_string(payload, "sourceAssetId")
        && !["referenceAssetIds", "controls", "controlnets", "loras"]
            .iter()
            .any(|key| has_nonempty_or_malformed_array(payload, key))
        && !["referenceAssetId", "maskAssetId"]
            .iter()
            .any(|key| has_nonempty_or_malformed_string(payload, key))
        && !["poses", "phases"]
            .iter()
            .any(|key| has_nonempty_or_malformed_nested_array(payload, "advanced", key))
        && [
            "controlMode",
            "controlImage",
            "controlScale",
            "controlWeights",
            "convRot",
            "quantTier",
        ]
        .iter()
        .all(|key| !has_nonnull_or_malformed_nested_carrier(payload, "advanced", key))
        && !has_malformed_optional_nested_number(payload, "advanced", "strength")
}

/// Z-Image strict-control Fun-ControlNet candle-routing conditions (sc-5489 origin / sc-8379 base, epic
/// 8236). The candle `ZImageControl` provider serves `z_image_turbo` OR the base `z_image` + a non-empty
/// `advanced.poses` (one image per pose, each conditioned on a DWPose skeleton via the VACE-style
/// Fun-Controlnet-Union branch — the Turbo or base checkpoint), NOT an `edit_image`. Same shape as the
/// qwen/kolors gates — the model gate (`z_image_turbo` / `z_image`) is applied at the call site (both call
/// this). Mirrors the worker's `zimage_control_available`. Candle-only — the macOS path is the MLX
/// `z_image_turbo_control` / `z_image_control` registry generators.
pub(crate) fn zimage_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_nested_array(payload, "advanced", "poses")
}

/// Z-Image identity-init (Image Studio "With Character") candle-routing conditions (sc-8409, epic 4406).
/// The candle `ZImageEdit` engine seeds the Turbo denoise from the chosen character `referenceAssetId`
/// latents (identity img2img) for a `character_image` job with `advanced.referenceStrength > 0`, that is
/// NOT an angle set (`advanced.angleSet`) and NOT a pose-library set (`advanced.poses`) — those are
/// `character_image` too but route to (and score on) their own candle lanes (InstantID angle/pose, the
/// Z-Image strict-control lane). The model gate (`z_image_turbo`) is applied at the call site. The
/// `referenceStrength > 0` engage condition mirrors the macOS `zimage_identity_strength` gate (zimage.rs,
/// sc-3146) EXACTLY, so candle routes the identity init precisely when the MLX generic lane runs it — a
/// With-Character job without a positive `referenceStrength` stays plain txt2img on both backends. Mirrors
/// the worker's `zimage_identity_candle_available` (minus the local weight-resolve check). Candle-only —
/// macOS keeps the MLX `z_image_turbo` generic-lane identity img2img (`resolve_zimage_identity_init`).
pub(crate) fn zimage_identity_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("character_image") {
        return false;
    }
    // A non-empty referenceAssetId is the identity source.
    if !has_nonempty_string(payload, "referenceAssetId") {
        return false;
    }
    // referenceStrength > 0 engages the identity init (parity with `zimage_identity_strength`); without a
    // positive strength the With-Character job stays plain txt2img.
    let reference_strength = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("referenceStrength"))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(0.0);
    if reference_strength <= 0.0 {
        return false;
    }
    // Angle / pose sets are `character_image` too but route to their own lanes — exclude both so this
    // plain With-Character gate never steals them (the worker sits this lane BEFORE the strict-control
    // lane). Mirrors the worker's `resolve_character_image_likeness_source` exclusions.
    let angle_set = match payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("angleSet"))
    {
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        _ => false,
    };
    if angle_set {
        return false;
    }
    !has_nonempty_nested_array(payload, "advanced", "poses")
}

/// FLUX.1-dev strict-control Shakker Union-Pro-2.0 candle-routing conditions (sc-8412, epic 8236). The
/// candle `Flux1DevControl` provider serves `flux_dev` + a non-empty `advanced.poses` (one image per pose,
/// each conditioned on a DWPose skeleton via the Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0` residual
/// branch on the dense bf16 dev base), NOT an `edit_image`. Same shape as the qwen/kolors/zimage/flux2
/// control gates — the model gate (`flux_dev`) is applied at the call site. Mirrors the worker's
/// `flux1_control_candle_available`. Candle-only — the macOS path is the MLX `flux1_dev_control` registry
/// generator (sc-8244).
pub(crate) fn flux1_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_nested_array(payload, "advanced", "poses")
}

/// FLUX.2-dev strict-pose Fun-Controlnet-Union candle-routing conditions (sc-7736, epic 6564). The candle
/// `Flux2Control` provider serves `flux2_dev` + a non-empty `advanced.poses` (one image per pose, each
/// conditioned on a DWPose skeleton via the VACE-style `FLUX.2-dev-Fun-Controlnet-Union` branch overlaid
/// on the Q4 dev DiT), NOT an `edit_image`. Same shape as the qwen/kolors/zimage control gates — the model
/// gate (`flux2_dev`) is applied at the call site. Mirrors the worker's `flux2_control_candle_available`.
/// Candle-only — the macOS path is the MLX `flux2_dev_control` registry generator (sc-6055).
pub(crate) fn flux2_dev_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    strict_candle_pose_count(payload).is_some_and(|count| count > 0)
}

/// Krea 2 pose-ControlNet candle-routing conditions (sc-8464, epic 8459). The candle `Krea2Control`
/// provider serves `krea_2_turbo` + a non-empty `advanced.poses` (one image per pose, each conditioned on
/// a DWPose skeleton via a trained control-branch overlay on the frozen Turbo base), NOT an `edit_image`.
/// Same shape as the qwen/kolors/zimage/flux control gates — the model gate (`krea_2_turbo`) is applied at
/// the call site. Mirrors the worker's `krea_control_candle_available`. Candle-only (there is no MLX Krea
/// control twin yet — 8459 S5 / sc-8465).
pub(crate) fn krea_control_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    has_nonempty_nested_array(payload, "advanced", "poses")
}

/// Candle-routed image models that HAVE a candle strict-control lane (sc-5489; flux2_dev sc-7736; base
/// z_image + flux_dev sc-8379 / sc-8412). A `advanced.poses` job on any OTHER candle-routed model has no
/// pose path on candle (plain-SDXL pose ships via InstantID, `instantid_realvisxl`, not `sdxl`).
pub(crate) const CANDLE_POSE_MODELS: &[&str] = &[
    "qwen_image",
    "kolors",
    "z_image_turbo",
    "z_image",
    "flux2_dev",
    "flux_dev",
    "krea_2_turbo",
];

pub(crate) fn model_has_candle_pose_lane(model: &str) -> bool {
    CANDLE_POSE_MODELS.contains(&model)
}

/// A strict-pose (`advanced.poses`) job on a **candle-routed model with no candle pose lane** —
/// `sdxl` / `realvisxl` / `chroma*` / `flux*` / `lens*` / `sensenova*` (everything outside the wired
/// pose families), not `edit_image` (sc-5968, epic 5483). No native lane has a pose path for these
/// models off-Mac (the historical `sdxl` adapter's OpenPose lived only in
/// the `instantid_realvisxl` adapter), so a generic claimant could silently drop the poses and render an unconditioned T2I
/// image. The candle worker therefore CLAIMS these (`worker_supports_job`) to REJECT them with a typed
/// error in the handler, while every other GPU descriptor declines them so candle reliably wins
/// and nothing silently mis-serves them. **Mac is unaffected:** `sdxl + poses` is MLX-served there
/// (`model_mac_support("sdxl").features.pose`), so the MLX worker claims it and other descriptors
/// decline. Pairs with the worker's `candle_unsupported_pose_reject` dispatch guard.
pub(crate) fn image_request_candle_pose_reject(model: &str, payload: &Map<String, Value>) -> bool {
    if !CANDLE_ROUTED_MODELS.contains(&model) || model_has_candle_pose_lane(model) {
        return false;
    }
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
}

/// [`image_request_candle_pose_reject`] on a [`JobSnapshot`].
pub(crate) fn image_job_candle_pose_reject(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageGenerate) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    image_request_candle_pose_reject(model, &job.payload)
}

/// Whether `worker` is the candle (Windows/CUDA) SDXL worker — identified by the `candle` marker
/// capability it self-advertises (`gpu::with_candle_capabilities`), mirroring the `nvidia` marker
/// the Rust GPU worker already emits. The candle worker runs on a real CUDA gpu index, not the
/// `mlx` sentinel, so it can't be recognized by `gpu_id`; the marker is the seam. When candle is
/// disabled the worker never advertises the marker, so this is always `false` and routing is
/// unchanged.
pub(crate) fn worker_is_candle(worker: &WorkerSnapshot) -> bool {
    worker
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "candle")
}

/// Epic 5164 / sc-7817 routing — does this `lora_train` job belong on the candle (Windows/CUDA +
/// Linux/NVIDIA) worker? The training sibling of
/// [`image_job_is_candle_eligible`]/[`video_job_is_candle_eligible`]: the candle engine has a native
/// trainer for the family. Both dry-run and real runs are eligible (the dry-run validates the same
/// resolved plan). `wan_moe_lora` accepts both A14B T2V and I2V bases, while `wan_lora` accepts the
/// dense TI2V-5B base. The resolved plan is stamped into the payload at submit (apps/rust-api
/// training.rs), so the kernel + base model are readable without touching the dataset or weights.
pub(crate) fn training_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    // The ControlNet studio job (epic 10159) trains through the SAME native executor keyed on the
    // resolved plan's kernel (`krea_control` ∈ [`CANDLE_ROUTED_TRAINING_KERNELS`]), so it is
    // candle-eligible on the same terms as a `lora_train` run.
    if !matches!(job.job_type, JobType::LoraTrain | JobType::ControlTraining) {
        return false;
    }
    let Some(plan) = job.payload.get("plan").and_then(Value::as_object) else {
        return false;
    };
    let target = plan.get("target").and_then(Value::as_object);
    let kernel = target
        .and_then(|target| target.get("kernel"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !CANDLE_ROUTED_TRAINING_KERNELS.contains(&kernel) {
        return false;
    }
    let base_model = target
        .and_then(|target| target.get("baseModel"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let network_type = plan
        .get("config")
        .and_then(Value::as_object)
        .and_then(|config| config.get("advanced"))
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("networkType"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("lora");
    let is_adapter =
        network_type.eq_ignore_ascii_case("lora") || network_type.eq_ignore_ascii_case("lokr");

    match kernel {
        "kolors_lora" => base_model == "kolors" && is_adapter,
        "sd3_lora" => matches!(base_model, "sd3_5_large" | "sd3_5_medium") && is_adapter,
        "wan_lora" => base_model == "wan_2_2" && network_type.eq_ignore_ascii_case("lora"),
        "wan_moe_lora" => match base_model {
            "wan_2_2_t2v_14b" => is_adapter,
            "wan_2_2_i2v_14b" => network_type.eq_ignore_ascii_case("lora"),
            _ => false,
        },
        "anima_lora" => base_model == "anima_base" && is_adapter,
        "mage_flow_lora" => {
            base_model == "mage_flow_base"
                && (is_adapter || network_type.eq_ignore_ascii_case("full"))
        }
        // Existing native families keep their established admission. Their trainers perform the
        // final typed validation; Krea ControlNet deliberately has no adapter-kind selector.
        _ => true,
    }
}

/// Whether an `image_upscale` job is candle-eligible (sc-5928 SeedVR2 + sc-5499 Real-ESRGAN, epic
/// 4811 / epic 5482): the candle worker serves **Real-ESRGAN** (`ort`/CUDA, the off-Mac sibling of
/// the Mac CoreML path — sc-5499) AND **SeedVR2** (`candle-gen-seedvr2`, sc-5928) off-Mac. This now
/// mirrors `upscale_job_is_mlx_eligible` exactly (the default `real-esrgan` engine + `seedvr2`);
/// `aura-sr` was dropped as an offered engine (sc-3668 Mac / sc-5499 off-Mac) so it has no candle
/// path — a candle worker refuses it, so it remains queued. Real-ESRGAN is candle-eligible, while
/// SeedVR2 is admitted only by its explicit native lanes.
pub(crate) fn upscale_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    upscale_job_is_mlx_eligible(job)
}

/// Whether a `video_upscale` job is candle-eligible (sc-5928, epic 4811 / epic 5482): the candle
/// SeedVR2 provider is the off-Mac video upscaler. Mirrors `video_upscale_job_is_mlx_eligible`
/// exactly (same engine set the worker's `run_video_upscale_job` accepts) — the engine defaults to
/// `seedvr2` when the payload omits it.
pub(crate) fn video_upscale_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    video_upscale_job_is_mlx_eligible(job)
}
