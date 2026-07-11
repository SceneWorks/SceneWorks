//! Candle (off-Mac / CUDA) routing predicates. Moved out of `jobs_store.rs` (sc-8816) with
//! no behavior change.

use serde_json::{Map, Value};

use crate::contracts::{JobSnapshot, JobType, WorkerSnapshot};
use crate::jobs_store::routing::catalog::{
    CANDLE_LORA_MODELS, CANDLE_QUANT_LORA_MODELS, CANDLE_QUANT_MODELS, CANDLE_ROUTED_MODELS,
    CANDLE_ROUTED_TRAINING_KERNELS, CANDLE_VIDEO_I2V_ROUTED_MODELS, CANDLE_VIDEO_ROUTED_MODELS,
    CANDLE_VIDEO_VACE_MODELS,
};
use crate::jobs_store::routing::mlx::{
    instantid_mlx_eligible, pulid_flux_mlx_eligible, upscale_job_is_mlx_eligible,
    video_upscale_job_is_mlx_eligible,
};

/// Candle video models whose provider descriptor advertises user-LoRA inference, so a video job
/// carrying `request.loras` stays on the candle lane instead of being refused. Today only the Wan-14B
/// MoE engines qualify: `candle-gen-wan`'s `wan14b` descriptor sets `supports_lora`, and the worker's
/// `candle_resolve_wan_adapters` applies each LoRA (including an external ComfyUI file) per MoE expert.
/// The wan-5B TI2V / LTX / SVD providers advertise no LoRA slot (sc-10539). Mirror of the candle-gen
/// descriptors — kept in lockstep the same way `CANDLE_VIDEO_ROUTED_MODELS` mirrors the routed engines.
pub(crate) const CANDLE_VIDEO_LORA_MODELS: &[&str] = &["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"];

/// Does this image job belong on the candle (Windows/CUDA) image lane (epic 3672, sc-3678)? The base
/// `generate_candle_stream` drives plain text-to-image, and the bespoke lanes branched out below add
/// the conditioned shapes ported under epic 5480 — SDXL/FLUX.2/Qwen `edit_image` (sc-5487), IP-Adapter
/// reference (sc-5488/sc-5872), InstantID/PuLID identity (sc-5491/sc-5492), and strict-pose ControlNet
/// (sc-5489). Anything still without a candle lane (a torch-only family, an unported shape, a LoRA on a
/// non-Lens family) falls back to the Python torch worker, so the candle worker refuses it here.
///
/// Like the MLX twin [`image_job_is_mlx_eligible`], this accepts BOTH `image_generate` and the distinct
/// `image_edit` job type (the Image Studio/Editor "plain Image Edit": `mode == "edit_image"` +
/// `sourceAssetId`, epic 2427) — the engine dispatches the SdxlEdit/Flux2Edit/QwenEdit lanes by payload
/// model+mode, not job type, so both job types route through the same per-model predicates. Without
/// `image_edit` here a plain Image Edit was wrongly enforce-failed `candle_unsupported` off-Mac instead
/// of reaching its candle edit lane (the sc-5487 lanes were validated only via `image_generate` jobs, so
/// the gap was invisible). The conditioning signals mirror the worker's `sdxl_sub_mode` / `pose_entries`
/// exactly, so the router and worker agree on the lane boundary.
pub(crate) fn image_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    // InstantID (sc-5491, epic 5480): the candle `candle-gen-instantid` provider serves the SAME
    // identity-preserving surface as the MLX path (single-identity character_image, the angle set,
    // pose-library mode, face-restore) — a bespoke `generate_instantid_stream` lane, NOT the
    // txt2img-only `image_request_candle_eligible` gate (which rejects `referenceAssetId`, which
    // InstantID requires). Branch it out before that gate. Retires the Python `_vendor/instantid`
    // off-Mac; the candle worker only advertises the `candle` marker when the backend is enabled, so a
    // candle-disabled box still falls these jobs back to the Python torch worker unchanged.
    if model == "instantid_realvisxl" {
        return instantid_candle_eligible(&job.payload);
    }
    // SDXL img2img / inpaint / outpaint edit (sc-5487, epic 5480): an sdxl-family `edit_image` job with
    // a source image is the bespoke candle `SdxlEdit` lane (`generate_candle_sdxl_edit_stream`), NOT
    // txt2img — the `image_request_candle_eligible` gate below rejects the whole `edit_image` family.
    // Branch it out first (disjoint from the IP-Adapter lane below, which is reference-only and not
    // `edit_image`). Mirrors the worker's `sdxl_edit_candle_available` gate.
    if is_sdxl_family_candle_model(model) && sdxl_edit_candle_eligible(&job.payload) {
        return true;
    }
    // FLUX.2-klein reference / img2img edit (sc-5487, epic 5480): a klein-family `edit_image` job with a
    // source image is the bespoke candle `Flux2Edit` lane (`generate_candle_flux2_edit_stream`), NOT
    // txt2img — the `image_request_candle_eligible` gate below rejects the whole `edit_image` family.
    // FLUX.2-klein has no torch path, so this is the only off-Mac edit lane for it. Mirrors the worker's
    // `flux2_edit_candle_available` gate.
    if matches!(model, "flux2_klein_9b" | "flux2_klein_9b_true_v2")
        && flux2_edit_candle_eligible(&job.payload)
    {
        return true;
    }
    // FLUX.2-dev edit (sc-7736, epic 6564): the 32B flagship `edit_image` job with a source is the SAME
    // bespoke candle `Flux2Edit` lane (`generate_candle_flux2_edit_stream` via `load_dev`, Q4 CPU-stage →
    // quantize-onto-GPU), NOT txt2img — the `image_request_candle_eligible` gate below rejects the whole
    // `edit_image` family. Branch it out first (the klein-edit reasoning, for the dev family). Same payload
    // predicate as klein. Mirrors the worker's `flux2_edit_candle_available` gate.
    if model == "flux2_dev" && flux2_edit_candle_eligible(&job.payload) {
        return true;
    }
    // Qwen-Image-Edit reference / dual-latent edit (sc-5487, epic 5480): a non-lightning Qwen-Image-Edit
    // `edit_image` job with a source image is the bespoke candle `QwenEdit` lane
    // (`generate_candle_qwen_edit_stream`), NOT txt2img — and `qwen_image_edit*` are not candle txt2img
    // ids (the gate below only knows `qwen_image`), so they would fall through to torch. Branch it out
    // first. Off-Mac this was a torch fallback. The `-2511_lightning` distill is the same `-2511` base
    // with the lightx2v 4-step LoRA folded into the MMDiT at load (sc-6220), so it routes to candle too.
    // Mirrors the worker's `qwen_edit_candle_available`.
    if matches!(
        model,
        "qwen_image_edit"
            | "qwen_image_edit_2509"
            | "qwen_image_edit_2511"
            | "qwen_image_edit_2511_lightning"
    ) && qwen_edit_candle_eligible(&job.payload)
    {
        return true;
    }
    // Z-Image img2img / edit (sc-6595, epic 5480): a z-image-family `edit_image` job with a source image
    // is the bespoke candle `ZImageEdit` lane (`generate_candle_zimage_edit_stream`), NOT txt2img — the
    // gate below rejects the whole `edit_image` family, and the dedicated `z_image_edit` id isn't even a
    // candle txt2img id (so a `z_image_edit` job would otherwise hit the "model not routed" gap and
    // misattribute to epic 3692). Branch it out first; disjoint from the Z-Image strict-pose control lane
    // below (that one is `advanced.poses`, not `edit_image`). Mirrors the worker's
    // `zimage_edit_candle_available`.
    if matches!(model, "z_image_turbo" | "z_image_edit")
        && zimage_edit_candle_eligible(&job.payload)
    {
        return true;
    }
    // Z-Image identity-init for Image Studio "With Character" (sc-8409, epic 4406): a `z_image_turbo`
    // `character_image` job with a `referenceAssetId` + `advanced.referenceStrength > 0` is the bespoke
    // candle `ZImageEdit` identity-init lane (`generate_candle_zimage_identity_stream`), NOT txt2img — the
    // `image_request_candle_eligible` gate below rejects any `referenceAssetId`, so without this the job
    // falls back to torch/MLX (off-Mac: plain txt2img, dropping the reference — the pre-existing gap this
    // story closes). Branch it out first; disjoint from the Z-Image edit lane above (`edit_image` +
    // `sourceAssetId`) and the strict-pose control lane below (`advanced.poses`, which this gate excludes).
    // Mirrors the worker's `zimage_identity_candle_available`.
    if model == "z_image_turbo" && zimage_identity_candle_eligible(&job.payload) {
        return true;
    }
    // Ideogram 4 img2img / Remix + mask inpaint / outpaint edit (sc-6598, epic 6561): an ideogram-family
    // `edit_image` job with a source image runs the candle `candle-gen-ideogram` edit path. Unlike the
    // other families above, Ideogram has no bespoke edit stream — it's the SAME engine for T2I and edit,
    // so the generic `generate_candle_stream` resolves the source `Reference` (+ optional `Mask`), exactly
    // as the MLX `generate_stream` handles Ideogram edit in-lane. The `image_request_candle_eligible` gate
    // below rejects the whole `edit_image` family, so branch it out here. Mirrors the worker's dispatch.
    if matches!(model, "ideogram_4" | "ideogram_4_turbo")
        && ideogram_edit_candle_eligible(&job.payload)
    {
        return true;
    }
    // Boogu instruction edit (sc-7524, epic 6831): a `boogu_image_edit` `edit_image` job with a source
    // image runs the candle `candle-gen-boogu` edit path. Like Ideogram (and unlike the SDXL/FLUX.2/Qwen/
    // Z-Image bespoke streams above), Boogu has no separate edit stream — the SAME registered
    // `boogu_image_edit` engine resolves the source `Reference` in the worker's `generate_candle_stream`
    // (the Qwen3-VL vision tower reads it + it VAE-encodes into the DiT reference latent), exactly as the
    // MLX `generate_stream` handles Boogu edit in-lane. The `image_request_candle_eligible` gate below
    // rejects the whole `edit_image` family, so branch it out here. Base/Turbo are pure T2I (the generic
    // gate). Mirrors the worker's dispatch + the MLX `boogu_mlx_eligible`.
    if model == "boogu_image_edit" && boogu_edit_candle_eligible(&job.payload) {
        return true;
    }
    // SDXL IP-Adapter-Plus reference conditioning (sc-5488, epic 5480): an sdxl-family model with a
    // reference image is a bespoke candle lane (`generate_candle_sdxl_ipadapter_stream`), NOT txt2img —
    // the `image_request_candle_eligible` gate below rejects `referenceAssetId`. Branch it out first
    // (pure IP only; img2img/inpaint/edit shapes are the SDXL edit lane above). Mirrors the worker's
    // `sdxl_ipadapter_available` gate.
    if is_sdxl_family_candle_model(model) && sdxl_ipadapter_candle_eligible(&job.payload) {
        return true;
    }
    // Kolors IP-Adapter-Plus reference conditioning (sc-5488, epic 5480): the `kolors` family with a
    // reference image is the same bespoke candle lane (`generate_candle_kolors_ipadapter_stream`), NOT
    // txt2img — branch it out before the gate (which rejects `referenceAssetId`). Pure IP only;
    // img2img/edit shapes stay on torch (sc-5487). Mirrors the worker's `kolors_ipadapter_available`.
    if model == "kolors" && kolors_ipadapter_candle_eligible(&job.payload) {
        return true;
    }
    // FLUX XLabs IP-Adapter reference conditioning (sc-5872, epic 5480): a `flux_dev`/`flux_schnell`
    // model with a reference image is the same bespoke candle lane (`generate_candle_flux_ipadapter_\
    // stream`), NOT txt2img — branch it out before the gate (which rejects `referenceAssetId`). Pure IP
    // only; img2img/edit shapes stay on torch (sc-5487). Mirrors the worker's `flux_ipadapter_available`.
    if matches!(model, "flux_dev" | "flux_schnell") && flux_ipadapter_candle_eligible(&job.payload)
    {
        return true;
    }
    // Qwen-Image strict-pose ControlNet (sc-5489, epic 5480): `qwen_image` + `advanced.poses` is a
    // bespoke candle lane (`generate_candle_qwen_control_stream`), NOT txt2img — the
    // `image_request_candle_eligible` gate below DEFERS any `advanced.poses` job to torch. Branch it out
    // first so `qwen_image` pose jobs reach candle (the kolors / z_image families follow below — all three
    // strict-pose families are now wired; plain-sdxl pose has no product route). Mirrors the worker's
    // `qwen_control_available`.
    if model == "qwen_image" && qwen_control_candle_eligible(&job.payload) {
        return true;
    }
    // Kolors strict-pose ControlNet (sc-5489, epic 5480): `kolors` + `advanced.poses` is the bespoke
    // candle lane (`generate_candle_kolors_control_stream`), NOT txt2img — the `image_request_candle_\
    // eligible` gate below DEFERS any `advanced.poses` job to torch. Branch it out first (the Qwen-control
    // reasoning, for the Kolors family). A pure-pose `kolors` job (no `referenceAssetId`) does NOT match
    // the `kolors_ipadapter_candle_eligible` branch above, so it reaches here. Mirrors the worker's
    // `kolors_control_available`.
    if model == "kolors" && kolors_control_candle_eligible(&job.payload) {
        return true;
    }
    // Z-Image strict-pose Fun-ControlNet (sc-5489, epic 5480): `z_image_turbo` + `advanced.poses` is the
    // bespoke candle lane (`generate_candle_zimage_control_stream`), NOT txt2img — the `image_request_\
    // candle_eligible` gate below DEFERS any `advanced.poses` job to torch. Branch it out first (the
    // Qwen/Kolors-control reasoning, for the last strict-pose family). Mirrors the worker's
    // `zimage_control_available`. With this all three control families (qwen / kolors / z_image) are wired.
    if model == "z_image_turbo" && zimage_control_candle_eligible(&job.payload) {
        return true;
    }
    // Base (non-distilled, full-CFG) Z-Image strict-control (sc-8379, epic 8236): `z_image` +
    // `advanced.poses` is the SAME bespoke candle `ZImageControl` lane as Turbo
    // (`generate_candle_zimage_control_stream`, base Fun-Controlnet-Union branch), NOT txt2img — branch it
    // out before the txt2img gate (which would defer the pose job to torch and has no base-z-image txt2img
    // provider anyway). Same payload shape as the Turbo gate. Mirrors the worker's `zimage_control_\
    // available` (which accepts both `z_image_turbo` and `z_image`).
    if model == "z_image" && zimage_control_candle_eligible(&job.payload) {
        return true;
    }
    // FLUX.1-dev strict-control Shakker Union-Pro-2.0 (sc-8412, epic 8236): `flux_dev` + `advanced.poses` is
    // the bespoke candle `Flux1DevControl` lane (`generate_candle_flux1_control_stream`), NOT txt2img — the
    // `image_request_candle_eligible` gate below DEFERS any `advanced.poses` job to torch, and the
    // pose-reject would otherwise claim-to-reject it (it now HAS a candle pose lane). Branch it out first (the
    // qwen/kolors/z_image/flux2-control reasoning, for the FLUX.1-dev family). A `flux_dev` reference job (a
    // `referenceAssetId`) is the FLUX XLabs IP-Adapter branch above; a pure-pose job reaches here. Mirrors
    // the worker's `flux1_control_candle_available`.
    if model == "flux_dev" && flux1_control_candle_eligible(&job.payload) {
        return true;
    }
    // FLUX.2-dev strict-pose Fun-Controlnet-Union (sc-7736, epic 6564): `flux2_dev` + `advanced.poses` is
    // the bespoke candle `Flux2Control` lane (`generate_candle_flux2_control_stream`), NOT txt2img — the
    // `image_request_candle_eligible` gate below DEFERS any `advanced.poses` job to torch, and the pose-
    // reject would otherwise claim-to-reject it (it now HAS a candle pose lane). Branch it out first (the
    // qwen/kolors/z_image-control reasoning, for the 4th wired strict-pose family). A flux2_dev edit job
    // (with a source) is the edit branch above; a pure-pose job (no source) reaches here. Mirrors the
    // worker's `flux2_control_candle_available`.
    if model == "flux2_dev" && flux2_dev_control_candle_eligible(&job.payload) {
        return true;
    }
    // PuLID-FLUX face identity (sc-5492, epic 5480): `pulid_flux_dev` is a distinct model id (not a
    // candle txt2img id), so the `image_request_candle_eligible` gate below would reject it; the candle
    // `candle-gen-pulid` provider serves it via a bespoke `generate_candle_pulid_stream` lane (the
    // off-Mac sibling of the macOS `pulid_flux` registry route). Branch it out, returning eligibility
    // directly — a non-character / reference-less job returns false → falls back to torch/MLX. Mirrors
    // the worker's `pulid_candle_available`.
    if model == "pulid_flux_dev" {
        return pulid_flux_candle_eligible(&job.payload);
    }
    image_request_candle_eligible(model, &job.payload)
}

/// Per-model candle txt2img-eligibility, factored out of [`image_job_is_candle_eligible`] so the
/// routing tests can probe it with synthetic payloads (parity with `image_request_mlx_eligible`).
pub(crate) fn image_request_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if !CANDLE_ROUTED_MODELS.contains(&model) {
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
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    // Any conditioning asset (img2img source, IP-Adapter reference, or inpaint mask) → torch. Applies
    // to EVERY candle family including Lens (pure T2I — no conditioning shapes in the Lens port).
    if has_nonempty_id("sourceAssetId")
        || has_nonempty_id("referenceAssetId")
        || has_nonempty_id("maskAssetId")
    {
        return false;
    }
    // Lens / Lens-Turbo and Krea 2 Turbo advertise Q4/Q8 + LoRA/LoKr, so a quant request OR a LoRA stays
    // on the candle lane for them (Krea gained Q4/Q8 in sc-9607, joining Lens in the both-set — sc-9983).
    // SD3.5 (sc-7880) and the Ideogram/Boogu packed families (sc-9607) advertise Q4/Q8 but NOT inference
    // LoRA (quant stays, LoRA defers). Every other candle family advertises neither and defers both. The
    // two capabilities are decoupled: `supports_lora` and `supports_quant` each consult the both-set plus
    // their own list.
    let supports_lora =
        CANDLE_QUANT_LORA_MODELS.contains(&model) || CANDLE_LORA_MODELS.contains(&model);
    let supports_quant =
        CANDLE_QUANT_LORA_MODELS.contains(&model) || CANDLE_QUANT_MODELS.contains(&model);
    // LoRAs: not in the candle lane unless the family advertises adapters (Lens / Krea).
    if !supports_lora
        && payload
            .get("loras")
            .and_then(Value::as_array)
            .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    // Strict-pose ControlNet (`advanced.poses`, object-shaped entries) → torch.
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses {
        return false;
    }
    // On-the-fly quantization (`advanced.mlxQuantize` > 0) → torch UNLESS the family advertises quant.
    // The sc-3675/sc-5096 candle providers advertise `supported_quants: &[]` (dense bf16/fp16 only), so
    // an explicit quant request can't be honored — route to Python rather than silently running dense
    // (sc-5099). Lens (sc-5126), SD3.5 (sc-7880), Krea (sc-9607/sc-9983), and the Ideogram/Boogu packed
    // families (sc-9607) advertise Q4/Q8, so their quant requests stay on candle. For the packed families
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

/// Does this video job belong on the candle video lane? The candle wan/ltx providers drive plain
/// text-to-video, the 14B I2V's single source-image conditioning (sc-5175), SVD image→video (sc-5493),
/// **and** the Wan-VACE advanced modes — replace_person / extend / bridge (sc-5494, the `PersonReplace`
/// / `VideoExtend` / `VideoBridge` job types → the candle `wan_vace` engine). Every other shape
/// (reference/mask/first-last-frame conditioning, LoRAs) must fall back to the Python torch worker, so
/// the candle worker refuses it here. SCAIL-2 (`scail2_14b`) adds a DISTINCT candle engine off-Mac —
/// `animate_character` + `replace_person` (sc-6837, epic 6563) — gated separately (it is not a VACE
/// model). The per-model shape gates are [`video_request_candle_eligible`] (base),
/// [`video_request_candle_vace_eligible`] (VACE modes), and
/// [`scail2_animate_candle_eligible`] / [`scail2_replace_candle_eligible`].
pub(crate) fn video_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    match job.job_type {
        // The base txt2video / image→video lane (sc-5097 / sc-5175 / sc-5493), plus SCAIL-2 standalone
        // character animation (`animate_character`, sc-6837 — a distinct candle engine, not VACE).
        JobType::VideoGenerate => {
            video_request_candle_eligible(model, &job.payload)
                || scail2_animate_candle_eligible(model, &job.payload)
        }
        // replace_person → candle Wan-VACE (sc-5494) OR candle SCAIL-2 (sc-6837, routed by model id).
        JobType::PersonReplace => {
            video_request_candle_vace_eligible(model, &job.payload, &job.job_type)
                || scail2_replace_candle_eligible(model, &job.payload)
        }
        // extend_clip / video_bridge → candle Wan-VACE only (sc-5494).
        JobType::VideoExtend | JobType::VideoBridge => {
            video_request_candle_vace_eligible(model, &job.payload, &job.job_type)
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
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    if CANDLE_VIDEO_I2V_ROUTED_MODELS.contains(&model) {
        // Wan 14B I2V is image→video ONLY (sc-5175): require the `image_to_video` mode + a source
        // image. A txt2video shape (no source) is rejected so a mis-picked text job stays on torch.
        if payload.get("mode").and_then(Value::as_str) != Some("image_to_video") {
            return false;
        }
        if !has_nonempty_id("sourceAssetId") {
            return false;
        }
    } else {
        // txt2video only: the base `video_generate` mode defaults to `image_to_video`, so require an
        // explicit `text_to_video`. Every conditioned mode (i2v / first_last_frame / extend / bridge /
        // replace) is thereby excluded, as is a stray source image.
        if payload.get("mode").and_then(Value::as_str) != Some("text_to_video") {
            return false;
        }
        if has_nonempty_id("sourceAssetId") {
            return false;
        }
    }
    // Reference / inpaint-mask conditioning is never in the candle video lane (i2v needs only the
    // single source image; reference + mask are the character / inpaint shapes that stay on torch).
    if has_nonempty_id("referenceAssetId") || has_nonempty_id("maskAssetId") {
        return false;
    }
    // User LoRAs on the candle video lane are gated by the provider descriptor: the Wan-14B MoE
    // engines (`wan_2_2_t2v_14b` / `wan_2_2_i2v_14b`) advertise `supports_lora` and their worker path
    // (`candle_resolve_wan_adapters`) applies each `request.loras` entry from its file path — so a wan-14B
    // job carrying user LoRAs stays on candle. Every other candle video provider (wan-5B TI2V, LTX, SVD)
    // advertises no LoRA slot, so a LoRA there is refused here (no torch fallback — epic 8283). sc-10539.
    if !CANDLE_VIDEO_LORA_MODELS.contains(&model)
        && payload
            .get("loras")
            .and_then(Value::as_array)
            .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    // On-the-fly quantization → torch (the candle video providers are dense; sc-5099).
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// Candle Wan-VACE eligibility for the advanced video job types (sc-5494): `PersonReplace`
/// (replace_person), `VideoExtend` (extend_clip), `VideoBridge` (video_bridge). Routes to the candle
/// `wan_vace` engine when the model is VACE-capable and the per-mode source assets are present. LoRA /
/// on-the-fly quant are not in the candle video lane (the VACE provider rejects them). Factored out so
/// the routing tests can probe it with synthetic payloads (parity with [`video_request_candle_eligible`]).
pub(crate) fn video_request_candle_vace_eligible(
    model: &str,
    payload: &Map<String, Value>,
    job_type: &JobType,
) -> bool {
    if !CANDLE_VIDEO_VACE_MODELS.contains(&model) {
        return false;
    }
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    match job_type {
        // replace_person: the source control clip + the tracked person + the character references.
        JobType::PersonReplace => {
            if !has_nonempty_id("sourceClipAssetId")
                || !has_nonempty_id("personTrackId")
                || !has_nonempty_id("characterId")
            {
                return false;
            }
        }
        // extend_clip: the source clip whose tail anchors the continuation.
        JobType::VideoExtend => {
            if !has_nonempty_id("sourceClipAssetId") {
                return false;
            }
        }
        // video_bridge: both clips (the left tail + the right head) are pinned around the gap.
        JobType::VideoBridge => {
            if !has_nonempty_id("sourceClipAssetId") || !has_nonempty_id("bridgeRightClipAssetId") {
                return false;
            }
        }
        _ => return false,
    }
    // LoRAs / on-the-fly quant are not in the candle video lane (the VACE provider rejects them).
    if payload
        .get("loras")
        .and_then(Value::as_array)
        .is_some_and(|loras| !loras.is_empty())
    {
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
/// stays on candle. On-the-fly quantization is still torch (the candle provider is dense). Mirrors the
/// MLX `video_mode_is_mlx_eligible(scail2_14b, animate_character)` shape, expressed as a candle-claim
/// gate. Factored out so the routing tests can probe it (parity with [`video_request_candle_eligible`]).
pub(crate) fn scail2_animate_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if model != "scail2_14b" {
        return false;
    }
    if payload.get("mode").and_then(Value::as_str) != Some("animate_character") {
        return false;
    }
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    let has_reference = has_nonempty_id("referenceAssetId")
        || has_nonempty_id("sourceAssetId")
        || payload
            .get("referenceAssetIds")
            .and_then(Value::as_array)
            .is_some_and(|ids| {
                ids.iter()
                    .any(|v| v.as_str().is_some_and(|s| !s.trim().is_empty()))
            });
    if !has_reference {
        return false;
    }
    if !has_nonempty_id("sourceClipAssetId") {
        return false;
    }
    // Inference LoRA (DPO / lightning / user adapter) merges into the candle DiT (sc-6838), so a
    // LoRA-bearing animate job is candle-eligible — only on-the-fly quant still falls back to torch.
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
}

/// Candle SCAIL-2 `replace_person` eligibility (sc-6837, epic 6563). The `scail2_14b` model behind a
/// `PersonReplace` job: the source control clip + the tracked person + the character references (the
/// same per-mode assets the Wan-VACE replace gate requires). No LoRA / on-the-fly quant (the provider
/// rejects them; inference LoRA is sc-6838). A distinct candle engine, so it is gated here rather than
/// added to [`CANDLE_VIDEO_VACE_MODELS`]. Factored out so the routing tests can probe it.
pub(crate) fn scail2_replace_candle_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if model != "scail2_14b" {
        return false;
    }
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    if !has_nonempty_id("sourceClipAssetId")
        || !has_nonempty_id("personTrackId")
        || !has_nonempty_id("characterId")
    {
        return false;
    }
    if payload
        .get("loras")
        .and_then(Value::as_array)
        .is_some_and(|loras| !loras.is_empty())
    {
        return false;
    }
    if candle_request_wants_quant(payload) {
        return false;
    }
    true
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
    payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// FLUX.2-klein edit candle-routing conditions (sc-5487, epic 5480). The candle `Flux2Edit` provider
/// serves `edit_image` mode with a `sourceAssetId` on the klein family — Kontext-style reference
/// token-concat editing (no mask / inpaint / outpaint; that masked shape is the SDXL edit lane's). Same
/// payload predicate as `sdxl_edit_candle_eligible`, gated to the klein family by the caller. Mirrors the
/// worker's `flux2_edit_candle_available` gate (minus the local weight-resolve check) so the router and
/// worker agree. Candle-only — macOS keeps the MLX `flux2_klein_9b_edit` registry path.
pub(crate) fn flux2_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// Qwen-Image-Edit candle-routing conditions (sc-5487, epic 5480). The candle `QwenEdit` provider
/// serves `edit_image` mode with a `sourceAssetId` on the non-lightning Qwen-Image-Edit family —
/// dual-latent reference editing (no mask / inpaint / outpaint; that masked shape is the SDXL edit
/// lane's). Same payload predicate as `flux2_edit_candle_eligible`, gated to the qwen-edit family by the
/// caller. Mirrors the worker's `qwen_edit_candle_available` gate (minus the local weight-resolve check)
/// so the router and worker agree. Candle-only — macOS keeps the MLX `qwen_image_edit` registry path.
pub(crate) fn qwen_edit_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }
    payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
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
    payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
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
    payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
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
    let single = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let plural = payload
        .get("referenceAssetIds")
        .and_then(Value::as_array)
        .is_some_and(|ids| {
            ids.iter()
                .any(|v| v.as_str().is_some_and(|s| !s.trim().is_empty()))
        });
    single || plural
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
    let non_empty = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    non_empty("referenceAssetId") && !non_empty("sourceAssetId") && !non_empty("maskAssetId")
}

/// Kolors IP-Adapter-Plus candle-routing conditions (sc-5488, epic 5480). The candle `IpAdapterKolors`
/// provider serves PURE reference (image-prompt) conditioning on the `kolors` family — the same payload
/// shape as the SDXL IP lane: a `referenceAssetId` with NO img2img source / inpaint mask and NOT an
/// `edit_image` (those advanced Kolors shapes are sc-5487, still torch). Mirrors the worker's
/// `kolors_ipadapter_available` gate (minus the local weight-resolve check) so the router and worker
/// agree on the lane boundary. Candle-only — the macOS Kolors IP path is the registry `Reference` route,
/// not a separate candle-eligible gate.
pub(crate) fn kolors_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let non_empty = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    non_empty("referenceAssetId") && !non_empty("sourceAssetId") && !non_empty("maskAssetId")
}

/// FLUX XLabs IP-Adapter candle-routing conditions (sc-5872, epic 5480). The candle `IpAdapterFlux`
/// provider serves PURE reference (image-prompt) conditioning on the `flux_dev`/`flux_schnell` families
/// — the same payload shape as the SDXL/Kolors IP lanes: a `referenceAssetId` with NO img2img source /
/// inpaint mask and NOT an `edit_image` (those advanced FLUX shapes are sc-5487, still torch). Mirrors
/// the worker's `flux_ipadapter_available` gate (minus the local weight-resolve check) so the router and
/// worker agree on the lane boundary. Candle-only — the macOS FLUX IP path is the registry `Reference`
/// route (epic 3621), not a separate candle-eligible gate.
pub(crate) fn flux_ipadapter_candle_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let non_empty = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    non_empty("referenceAssetId") && !non_empty("sourceAssetId") && !non_empty("maskAssetId")
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
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
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
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
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
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
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
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    if !has_reference {
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
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    !has_poses
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
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
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
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
}

/// Candle-routed image models that HAVE a candle strict-control lane (sc-5489; flux2_dev sc-7736; base
/// z_image + flux_dev sc-8379 / sc-8412). A `advanced.poses` job on any OTHER candle-routed model has no
/// pose path on candle (plain-SDXL pose ships via InstantID, `instantid_realvisxl`, not `sdxl`).
pub(crate) fn model_has_candle_pose_lane(model: &str) -> bool {
    matches!(
        model,
        "qwen_image" | "kolors" | "z_image_turbo" | "z_image" | "flux2_dev" | "flux_dev"
    )
}

/// A strict-pose (`advanced.poses`) job on a **candle-routed model with no candle pose lane** —
/// `sdxl` / `realvisxl` / `chroma*` / `flux*` / `lens*` / `sensenova*` (everything but the three wired
/// pose families), not `edit_image` (sc-5968, epic 5483). Neither candle nor the co-resident torch
/// worker has a pose path for these models off-Mac (the torch `sdxl` adapter's OpenPose lives only in
/// the `instantid_realvisxl` adapter), so torch would silently drop the poses → an unconditioned T2I
/// image. The candle worker therefore CLAIMS these (`worker_supports_job`) to REJECT them with a typed
/// error in the handler, and the co-resident torch worker DECLINES them (below) so candle reliably wins
/// and nothing silently mis-serves them. **Mac is unaffected:** `sdxl + poses` is MLX-served there
/// (`model_mac_support("sdxl").features.pose`), so the MLX worker claims it and only the torch/`mps`
/// worker declines. Pairs with the worker's `candle_unsupported_pose_reject` dispatch guard.
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
/// Linux/NVIDIA) worker (vs the Python torch worker)? The training sibling of
/// [`image_job_is_candle_eligible`]/[`video_job_is_candle_eligible`]: the candle engine has a native
/// trainer for the family. Both dry-run and real runs are eligible (the dry-run validates the same
/// resolved plan). `wan_moe_lora` is candle-eligible ONLY for the **T2V** A14B base model
/// (`wan_2_2_t2v_14b`) — the candle Wan trainer is registered under `wan2_2_t2v_14b` only; the I2V
/// A14B and the dense `wan_lora` 5B have no candle trainer, so they stay on torch. UNLIKE the mlx Wan
/// path, the candle Wan trainer DOES support LoKr (its `build_lokr_targets` merge), so there is no
/// LoKr-on-Wan exclusion here. The resolved plan is stamped into the payload at submit (apps/rust-api
/// training.rs), so the kernel + base model are readable without touching the dataset or weights.
pub(crate) fn training_job_is_candle_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::LoraTrain) {
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
    if CANDLE_ROUTED_TRAINING_KERNELS.contains(&kernel) {
        return true;
    }
    // The A14B MoE: candle registers only the T2V trainer (`wan2_2_t2v_14b`). The I2V A14B base
    // model has no candle trainer, so it stays on torch.
    if kernel == "wan_moe_lora" {
        let base_model = target
            .and_then(|target| target.get("baseModel"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        return base_model == "wan_2_2_t2v_14b";
    }
    false
}

/// Whether an `image_upscale` job is candle-eligible (sc-5928 SeedVR2 + sc-5499 Real-ESRGAN, epic
/// 4811 / epic 5482): the candle worker serves **Real-ESRGAN** (`ort`/CUDA, the off-Mac sibling of
/// the Mac CoreML path — sc-5499) AND **SeedVR2** (`candle-gen-seedvr2`, sc-5928) off-Mac. This now
/// mirrors `upscale_job_is_mlx_eligible` exactly (the default `real-esrgan` engine + `seedvr2`);
/// `aura-sr` was dropped as an offered engine (sc-3668 Mac / sc-5499 off-Mac) so it has no candle
/// path — a candle worker refuses it (it runs only on the Python torch worker until Phase 7). Note
/// Real-ESRGAN keeps its torch path as a co-resident fallback (the torch worker is NOT refused it,
/// unlike SeedVR2), so a Real-ESRGAN job may run on whichever worker claims it first.
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
