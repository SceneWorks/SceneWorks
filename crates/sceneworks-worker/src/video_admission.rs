//! The worker half of the video memory gate (sc-18814, epic 18803).
//!
//! `sceneworks_core::video_request` owns the video admission POLICY — which family is routed on
//! which lane, which geometries have to be graded, what a refusal says. It cannot own the
//! SELECTION, because `sceneworks-core` deliberately carries no gen-core dependency and the
//! ladder selector (`crate::memory_strategy::select_strategy`) is gen-core-typed. This module is
//! the bridge: it implements core's [`VideoStrategySelector`] seam by building candidates from
//! the loaded provider's own [`MemoryProviderContract`] and calling that one shared selector.
//!
//! **Epic decision 3 stands** (sc-18814, reaffirmed at activity-19060): the video gate is
//! `video_request.rs`, not a unified `mlx_fit_gate`. Ordering, first-fit and margin grading are
//! not re-implemented here — `select_strategy` does all three, exactly as it does for
//! `mlx_fit_gate` (the MLX image lane) and `candle_memory_strategy` (the candle image lane).
//!
//! **Request selection is evidence-gated; prediction is exact-or-floor.** The request entry point
//! first requires a full curve match across catalog/provider/lane/tier/mode/reference shape and
//! count/output FPS/overlay/rung/load-shape/ABI/fingerprint/closure/decode-regime. A candidate
//! whose identity matches a packaged fitted curve then uses
//! its three residual-bounded affine cross laws
//! (`fixed + perMpx*mpx + perMpxFrame*mpx*frames + maxResidual`). The internal selector retains
//! a floor for legacy load-time/sequential behavior and focused candidate tests, but it is never
//! request-scoped evidence. A request mismatch — including geometry outside the measured
//! area-by-voxel hull — keeps direct generation and receives no memory context. The floor remains
//! the established
//! [`crate::mlx_fit_gate::estimate_floor_weights_bytes`] plus activation-headroom lower bound,
//! strengthened when the selected provider exports a larger decode working-set profile. Requested
//! rows key `gen_core::MemoryGeometry` to the real clip length; synthetic cap rows key it to the cap
//! they evaluate. The eventual provider run context still receives the actual request geometry,
//! and the image lane remains untouched at one frame.

use gen_core::tiling::VideoDecodeMemoryProfile;
use gen_core::{
    Conditioning, MemoryBackend, MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryNumericTier,
    MemoryProviderContract, MemoryRunContext, MemorySelection, MemoryStrategy, OffloadPolicy,
};
use sceneworks_core::memory_calibration::{Ltx25Decoder, Ltx25TransformerVariant, StrategyRung};
use sceneworks_core::video_memory_curves::{
    VideoCurveBackend, VideoCurveDecodePass, VideoCurveGeometry, VideoCurveLoadShape,
    VideoCurveQuery, VideoMemoryCurveBundle,
};
use sceneworks_core::video_request::{
    VideoAdmissionGeometry, VideoDecodePass, VideoLane, VideoRungSelection, VideoStrategySelector,
};
use sha2::Digest;

use crate::memory_strategy::{Budget, Candidate, CandidateBasis, RequestScope, Selection};

pub(crate) const BERNINI_R2V_RECEIPT_DOMAIN: &str = "bernini-r2v-references-v2";
pub(crate) const BERNINI_R2V_SEAL_DOMAIN: &str = "bernini-r2v-request-seal-v1";
pub(crate) const BERNINI_MV2V_RECEIPT_DOMAIN: &str = "bernini-mv2v-clips-v1";
pub(crate) const BERNINI_MV2V_SEAL_DOMAIN: &str = "bernini-mv2v-request-seal-v1";
pub(crate) const BERNINI_ADS2V_RECEIPT_DOMAIN: &str = "bernini-ads2v-sources-v1";
pub(crate) const BERNINI_ADS2V_SEAL_DOMAIN: &str = "bernini-ads2v-request-seal-v1";
pub(crate) const BERNINI_ADAPTER_RECEIPT_AXIS_DOMAIN: &str = "bernini-adapter-receipt-v1";
// sc-20799. Outside Bernini and LTX the video carriers were keyed by SHAPE ALONE — a bare `"other"`
// plus `conditioning.len()` — so a one-image `MultiReference` and an eight-image one, or two
// different driving clips and masks, shared a single admission identity and could borrow each
// other's admitted peak. These seal the carriers the way Bernini's do: a content-INDEPENDENT
// evidence axis (order, counts, native shapes, and the exact knobs that change the working set)
// plus a request-only byte seal appended after `+`, which `video_curve_overlay` strips before curve
// lookup. Sealed BEFORE any fitted curve exists for these cells, so no curve can be fitted against
// a borrowable identity.
pub(crate) const WAN_VACE_REPLACE_RECEIPT_DOMAIN: &str = "wan-vace-replace-person-carrier-v1";
pub(crate) const WAN_VACE_REPLACE_SEAL_DOMAIN: &str = "wan-vace-replace-person-request-seal-v1";
pub(crate) const SCAIL2_CARRIER_RECEIPT_DOMAIN: &str = "scail2-carrier-v1";
pub(crate) const SCAIL2_CARRIER_SEAL_DOMAIN: &str = "scail2-carrier-request-seal-v1";
pub(crate) const KREA_V2V_RECEIPT_DOMAIN: &str = "krea-realtime-v2v-clip-v1";
pub(crate) const KREA_V2V_SEAL_DOMAIN: &str = "krea-realtime-v2v-request-seal-v1";

fn backend_axis(lane: VideoLane) -> &'static str {
    match lane {
        VideoLane::Mlx => "mlx",
        VideoLane::Candle => "candle",
    }
}

/// Seal one ordered image list into the evidence axis (native shapes) and the byte seal.
fn seal_images(role: &str, images: &[gen_core::Image], seal: &mut sha2::Sha256) -> String {
    seal.update((role.len() as u64).to_le_bytes());
    seal.update(role.as_bytes());
    let shapes = images
        .iter()
        .enumerate()
        .map(|(ordinal, image)| {
            seal.update((ordinal as u32).to_le_bytes());
            seal.update(image.width.to_le_bytes());
            seal.update(image.height.to_le_bytes());
            seal.update(&image.pixels);
            format!("{ordinal}:{}x{}", image.width, image.height)
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{role}-{}:{shapes}", images.len())
}

/// Worker seal for the single-expert Wan-VACE `replace_person` carrier
/// (`video_jobs::vace::build_vace_conditioning`): one `ControlClip` of driving frames plus its
/// per-frame mask, then one `Reference` per character image. The frame/mask cardinality, the
/// per-frame geometry, the masking strength and the replacement mode all move the working set, and
/// the reference ORDER is load order, so all of them are evidence axes.
pub(crate) fn wan_vace_replace_person_receipt(
    lane: VideoLane,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let [Conditioning::ControlClip {
        frames,
        mask,
        masking_strength,
        start_frame,
        mode,
    }, references @ ..] = conditioning
    else {
        return Err(
            "Wan-VACE replace_person admission requires a leading ControlClip carrier".to_owned(),
        );
    };
    if frames.len() != mask.len() || frames.is_empty() {
        return Err(format!(
            "Wan-VACE replace_person admission requires equal non-empty frame/mask counts, got {} and {}",
            frames.len(),
            mask.len()
        ));
    }
    if *start_frame != 0 {
        return Err(format!(
            "Wan-VACE replace_person admission requires a frame-0 control clip, got {start_frame}"
        ));
    }
    let mut reference_images: Vec<gen_core::Image> = Vec::with_capacity(references.len());
    for entry in references {
        let Conditioning::Reference { image, strength } = entry else {
            return Err(
                "Wan-VACE replace_person admission accepts only Reference entries after its \
                 ControlClip"
                    .to_owned(),
            );
        };
        if strength.is_some() {
            return Err(
                "Wan-VACE replace_person references carry no strength; a populated one is a \
                 different workload"
                    .to_owned(),
            );
        }
        reference_images.push(image.clone());
    }
    let mut seal = sha2::Sha256::new();
    seal.update(WAN_VACE_REPLACE_SEAL_DOMAIN.as_bytes());
    seal.update(masking_strength.to_bits().to_le_bytes());
    let frames_axis = seal_images("control-frames", frames, &mut seal);
    let mask_axis = seal_images("control-mask", mask, &mut seal);
    let reference_axis = seal_images("references", &reference_images, &mut seal);
    Ok(format!(
        "{WAN_VACE_REPLACE_RECEIPT_DOMAIN}:backend-{}:composite-{width}x{height}:{frames_axis}:{mask_axis}:strength-{:08x}:mode-{mode:?}:frame-0:{reference_axis}+{WAN_VACE_REPLACE_SEAL_DOMAIN}-{:x}",
        backend_axis(lane),
        masking_strength.to_bits(),
        seal.finalize()
    ))
}

/// Worker seal for both SCAIL-2 carriers (`animate_character` and `replace_person`), which share
/// one physical assembly built by `video_jobs::scail2`: a `Reference` identity still, its painted
/// `Mask`, and a `ControlClip` of driving frames plus per-frame color masks. The engine TASK is a
/// separate axis because animation and replacement are different working sets over the same shapes.
pub(crate) fn scail2_carrier_receipt(
    lane: VideoLane,
    mode: &str,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let task = match mode {
        "animate_character" => "animation",
        "replace_person" => "replacement",
        other => return Err(format!("SCAIL-2 admission has no carrier for mode {other}")),
    };
    let [Conditioning::Reference {
        image: reference,
        strength: reference_strength,
    }, Conditioning::Mask {
        image: reference_mask,
    }, Conditioning::ControlClip {
        frames,
        mask,
        masking_strength,
        start_frame,
        mode: replacement_mode,
    }] = conditioning
    else {
        return Err(
            "SCAIL-2 admission requires exactly one Reference, one Mask, and one ControlClip, in \
             that order"
                .to_owned(),
        );
    };
    if reference_strength.is_some() {
        return Err(
            "the SCAIL-2 identity reference carries no strength; a populated one is a different \
             workload"
                .to_owned(),
        );
    }
    if frames.len() != mask.len() || frames.is_empty() {
        return Err(format!(
            "SCAIL-2 admission requires equal non-empty driving frame/mask counts, got {} and {}",
            frames.len(),
            mask.len()
        ));
    }
    if *start_frame != 0 {
        return Err(format!(
            "SCAIL-2 admission requires a frame-0 driving clip, got {start_frame}"
        ));
    }
    let mut seal = sha2::Sha256::new();
    seal.update(SCAIL2_CARRIER_SEAL_DOMAIN.as_bytes());
    seal.update(task.as_bytes());
    seal.update(masking_strength.to_bits().to_le_bytes());
    let reference_axis = seal_images("reference", std::slice::from_ref(reference), &mut seal);
    let reference_mask_axis = seal_images(
        "reference-mask",
        std::slice::from_ref(reference_mask),
        &mut seal,
    );
    let frames_axis = seal_images("driving-frames", frames, &mut seal);
    let mask_axis = seal_images("driving-mask", mask, &mut seal);
    Ok(format!(
        "{SCAIL2_CARRIER_RECEIPT_DOMAIN}:backend-{}:task-{task}:composite-{width}x{height}:{reference_axis}:{reference_mask_axis}:{frames_axis}:{mask_axis}:strength-{:08x}:mode-{replacement_mode:?}:frame-0+{SCAIL2_CARRIER_SEAL_DOMAIN}-{:x}",
        backend_axis(lane),
        masking_strength.to_bits(),
        seal.finalize()
    ))
}

/// Worker seal for the Krea Realtime video-to-video carrier
/// (`video_jobs::krea_realtime::krea_realtime_conditioning`): one `VideoClip` whose frame count and
/// per-frame geometry set the autoregressive init working set, plus the v2v strength — the one knob
/// that distinguishes v2v from the i2v cache warm, which drops strength entirely.
pub(crate) fn krea_v2v_clip_receipt(
    lane: VideoLane,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let [Conditioning::VideoClip {
        frames,
        frame_idx,
        strength,
    }] = conditioning
    else {
        return Err(
            "Krea Realtime video_to_video admission requires exactly one VideoClip".to_owned(),
        );
    };
    if *frame_idx != 0 || frames.is_empty() {
        return Err(format!(
            "Krea Realtime v2v admission requires a non-empty frame-0 clip, got {} frames at {frame_idx}",
            frames.len()
        ));
    }
    let mut seal = sha2::Sha256::new();
    seal.update(KREA_V2V_SEAL_DOMAIN.as_bytes());
    seal.update(strength.to_bits().to_le_bytes());
    let frames_axis = seal_images("clip", frames, &mut seal);
    Ok(format!(
        "{KREA_V2V_RECEIPT_DOMAIN}:backend-{}:composite-{width}x{height}:{frames_axis}:frame-0:strength-{:08x}+{KREA_V2V_SEAL_DOMAIN}-{:x}",
        backend_axis(lane),
        strength.to_bits(),
        seal.finalize()
    ))
}

pub(crate) fn bernini_adapter_receipt_axis(component_id: &str) -> String {
    let mut digest = sha2::Sha256::new();
    digest.update(BERNINI_ADAPTER_RECEIPT_AXIS_DOMAIN.as_bytes());
    digest.update(component_id.as_bytes());
    format!(
        "{BERNINI_ADAPTER_RECEIPT_AXIS_DOMAIN}-{:x}",
        digest.finalize()
    )
}

fn round_half_even(value: f64) -> i64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor as i64
    } else if fraction > 0.5 || (floor as i64) % 2 != 0 {
        floor as i64 + 1
    } else {
        floor as i64
    }
}

fn bernini_vit_shape(width: u32, height: u32) -> (i64, i64) {
    const FACTOR: i64 = 28;
    const MIN_PIXELS: i64 = 3136;
    const MAX_PIXELS: i64 = 50176;
    let (width, height) = (f64::from(width), f64::from(height));
    let mut effective_height = round_half_even(height / FACTOR as f64) * FACTOR;
    let mut effective_width = round_half_even(width / FACTOR as f64) * FACTOR;
    if effective_height * effective_width > MAX_PIXELS {
        let scale = ((height * width) / MAX_PIXELS as f64).sqrt();
        effective_height = FACTOR.max((height / scale / FACTOR as f64).floor() as i64 * FACTOR);
        effective_width = FACTOR.max((width / scale / FACTOR as f64).floor() as i64 * FACTOR);
    } else if effective_height * effective_width < MIN_PIXELS {
        let scale = (MIN_PIXELS as f64 / (height * width)).sqrt();
        effective_height = (height * scale / FACTOR as f64).ceil() as i64 * FACTOR;
        effective_width = (width * scale / FACTOR as f64).ceil() as i64 * FACTOR;
    }
    (effective_width, effective_height)
}

/// Exact full-Bernini source preprocessing: both clip frames and reference images pass through
/// the max-624, stride-16 VAE transform before conditioning. The renderer-only provider instead
/// uses output geometry for both; SceneWorks resolves the full `bernini` provider, so mixing the
/// two is an evidence mismatch rather than a conservative approximation.
fn bernini_full_vae_shape(width: u32, height: u32) -> (i64, i64) {
    const MAX_SIZE: f64 = 624.0;
    const STRIDE: i64 = 16;
    let (width, height) = (f64::from(width), f64::from(height));
    let mut scale = (MAX_SIZE / width.max(height)).min(1.0);
    scale = scale.max(1.0 / width.min(height));
    let make_divisible = |value: f64| {
        let scaled = round_half_even(value);
        STRIDE.max(round_half_even(scaled as f64 / STRIDE as f64) * STRIDE)
    };
    let mut effective = (
        make_divisible(width * scale),
        make_divisible(height * scale),
    );
    if effective.0.max(effective.1) > MAX_SIZE as i64 {
        scale = MAX_SIZE / effective.0.max(effective.1) as f64;
        effective = (
            make_divisible(effective.0 as f64 * scale),
            make_divisible(effective.1 as f64 * scale),
        );
    }
    effective
}

fn bernini_packed_source_tokens(frames: usize, width: u32, height: u32) -> Result<u64, String> {
    const VAE_TEMPORAL_SCALE: u64 = 4;
    const VAE_DIT_SPATIAL_STRIDE: u64 = 16;
    let frames =
        u64::try_from(frames).map_err(|_| "Bernini source frame count overflow".to_owned())?;
    if u64::from(width) % VAE_DIT_SPATIAL_STRIDE != 0
        || u64::from(height) % VAE_DIT_SPATIAL_STRIDE != 0
    {
        return Err(format!(
            "Bernini source geometry {width}x{height} does not land on the exact VAE/DiT token grid"
        ));
    }
    let latent_frames = frames.saturating_sub(1) / VAE_TEMPORAL_SCALE + 1;
    latent_frames
        .checked_mul(u64::from(width) / VAE_DIT_SPATIAL_STRIDE)
        .and_then(|tokens| tokens.checked_mul(u64::from(height) / VAE_DIT_SPATIAL_STRIDE))
        .ok_or_else(|| "Bernini packed source-token count overflow".to_owned())
}

/// Worker-side independent twin of the provider's ordered reference receipt. The first axis is the
/// content-independent memory evidence identity (count/order/native and effective shapes); the
/// second is a request-only byte seal. Curve lookup strips only the seal, so same-shape references
/// share memory evidence while post-admission content mutation still fails provider configure.
pub(crate) fn bernini_r2v_reference_receipt(
    lane: VideoLane,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let (clip, images) = match conditioning {
        [Conditioning::MultiReference { images }] => (None, images.as_slice()),
        [
            Conditioning::VideoClip {
                frames,
                frame_idx,
                strength,
            },
            Conditioning::MultiReference { images },
        ] => (
            Some((frames.as_slice(), *frame_idx, *strength)),
            images.as_slice(),
        ),
        _ => {
            return Err(
                "Bernini reference admission requires one MultiReference, optionally preceded by exactly one VideoClip"
                    .to_owned(),
            )
        }
    };
    if !(1..=8).contains(&images.len()) {
        return Err(format!(
            "Bernini R2V admission requires 1-8 flattened images, got {}",
            images.len()
        ));
    }
    let backend = match lane {
        VideoLane::Mlx => "mlx",
        VideoLane::Candle => "candle",
    };
    const FULL_PREPROCESS_AXIS: &str = "source-preprocess-full-vae624-v1";
    let mut request_seal = sha2::Sha256::new();
    request_seal.update(BERNINI_R2V_SEAL_DOMAIN.as_bytes());
    let mut entries = Vec::with_capacity(images.len() + usize::from(clip.is_some()));
    let has_video = clip.is_some();
    let mut packed_tokens = 0_u64;
    if let Some((frames, frame_idx, strength)) = clip {
        if frame_idx != 0
            || strength.to_bits() != 1.0_f32.to_bits()
            || !matches!(frames.len(), 45 | 61 | 77)
        {
            return Err(
                "Bernini RV2V admission requires one normalized 45/61/77-frame VideoClip at frame 0 with strength 1"
                    .to_owned(),
            );
        }
        request_seal.update(b"video-1");
        request_seal.update(frame_idx.to_le_bytes());
        request_seal.update(strength.to_bits().to_le_bytes());
        for (index, frame) in frames.iter().enumerate() {
            let expected = u64::from(width)
                .checked_mul(u64::from(height))
                .and_then(|pixels| pixels.checked_mul(3))
                .and_then(|pixels| usize::try_from(pixels).ok())
                .ok_or_else(|| "Bernini RV2V clip geometry overflow".to_owned())?;
            if frame.width != width || frame.height != height || frame.pixels.len() != expected {
                return Err(format!(
                    "Bernini RV2V clip frame {index} is not exact output-sized RGB8"
                ));
            }
            request_seal.update((index as u32).to_le_bytes());
            request_seal.update(frame.width.to_le_bytes());
            request_seal.update(frame.height.to_le_bytes());
            request_seal.update(&frame.pixels);
        }
        let (vae_width, vae_height) = bernini_full_vae_shape(width, height);
        let tokens =
            bernini_packed_source_tokens(frames.len(), vae_width as u32, vae_height as u32)?;
        packed_tokens = packed_tokens
            .checked_add(tokens)
            .ok_or_else(|| "Bernini combined source-token count overflow".to_owned())?;
        entries.push(format!(
            "video-1:frames-{};native-{width}x{height};vae-{}x{}x{};tokens-{tokens}",
            frames.len(),
            (frames.len() - 1) / 4 + 1,
            vae_width / 8,
            vae_height / 8
        ));
    }
    for (index, image) in images.iter().enumerate() {
        let expected = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|pixels| usize::try_from(pixels).ok())
            .ok_or_else(|| "Bernini R2V reference geometry overflow".to_owned())?;
        if image.pixels.len() != expected {
            return Err(format!(
                "Bernini R2V reference {index} has {} RGB bytes, expected {expected}",
                image.pixels.len()
            ));
        }
        let (vit_width, vit_height) = bernini_vit_shape(image.width, image.height);
        let (vae_width, vae_height) = bernini_full_vae_shape(image.width, image.height);
        request_seal.update(image.width.to_le_bytes());
        request_seal.update(image.height.to_le_bytes());
        request_seal.update(&image.pixels);
        if has_video {
            let tokens = bernini_packed_source_tokens(1, vae_width as u32, vae_height as u32)?;
            packed_tokens = packed_tokens
                .checked_add(tokens)
                .ok_or_else(|| "Bernini combined source-token count overflow".to_owned())?;
            entries.push(format!(
                "{index}:native-{}x{};vit-{vit_width}x{vit_height};vae-{vae_width}x{vae_height};tokens-{tokens}",
                image.width, image.height
            ));
        } else {
            entries.push(format!(
                "{index}:native-{}x{};vit-{vit_width}x{vit_height};vae-{vae_width}x{vae_height}",
                image.width, image.height
            ));
        }
    }
    let packed_surface = has_video.then(|| format!("packed-source-tokens-{packed_tokens}:"));
    Ok(format!(
        "{BERNINI_R2V_RECEIPT_DOMAIN}:backend-{backend}:{FULL_PREPROCESS_AXIS}:count-{}:{}{}+{BERNINI_R2V_SEAL_DOMAIN}-{:x}",
        images.len(),
        packed_surface.as_deref().unwrap_or_default(),
        entries.join("|"),
        request_seal.finalize()
    ))
}

fn bernini_mv2v_source_ids(count: usize) -> Vec<String> {
    if count <= 5 {
        return (1..=count).map(|id| id.to_string()).collect();
    }
    (0..count)
        .map(|index| {
            let id = 1.0 + 4.0 * index as f64 / (count - 1) as f64;
            if id.fract() == 0.0 {
                format!("{id:.0}")
            } else {
                format!("{id:.6}")
            }
        })
        .collect()
}

fn bernini_ads2v_source_ids(count: usize) -> Vec<String> {
    debug_assert!((3..=10).contains(&count));
    if count <= 3 {
        return (1..=count).map(|id| id.to_string()).collect();
    }
    (0..count)
        .map(|index| {
            let id = 1.0 + 2.0 * index as f64 / (count - 1) as f64;
            if id.fract() == 0.0 {
                format!("{id:.0}")
            } else {
                format!("{id:.6}")
            }
        })
        .collect()
}

/// Worker twin of the full provider's ADS2V receipt. The role labels make the source and
/// reference clips non-interchangeable; image count/order and backend preprocessing remain exact.
pub(crate) fn bernini_ads2v_source_receipt(
    lane: VideoLane,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    let [Conditioning::VideoClip {
        frames: source_frames,
        frame_idx: source_index,
        strength: source_strength,
    }, Conditioning::VideoClip {
        frames: reference_frames,
        frame_idx: reference_index,
        strength: reference_strength,
    }, Conditioning::MultiReference { images }] = conditioning
    else {
        return Err("Bernini ADS2V admission requires [source VideoClip, reference VideoClip, MultiReference 1-8 ordered images]".to_owned());
    };
    if !(1..=8).contains(&images.len()) {
        return Err("Bernini ADS2V admission requires 1-8 flattened images".to_owned());
    }
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|pixels| usize::try_from(pixels).ok())
        .ok_or_else(|| "Bernini ADS2V clip geometry overflow".to_owned())?;
    let backend = match lane {
        VideoLane::Mlx => "mlx",
        VideoLane::Candle => "candle",
    };
    let (clip_vae_width, clip_vae_height) = bernini_full_vae_shape(width, height);
    let mut seal = sha2::Sha256::new();
    seal.update(BERNINI_ADS2V_SEAL_DOMAIN.as_bytes());
    let mut tokens_total = 0_u64;
    let mut entries = Vec::with_capacity(images.len() + 2);
    for (role, frames, frame_idx, strength) in [
        ("source-video", source_frames, source_index, source_strength),
        (
            "reference-video",
            reference_frames,
            reference_index,
            reference_strength,
        ),
    ] {
        if *frame_idx != 0
            || strength.to_bits() != 1.0_f32.to_bits()
            || !matches!(frames.len(), 45 | 61 | 77)
        {
            return Err(format!("Bernini ADS2V {role} clip must be normalized 45/61/77 frames at frame 0 with strength 1"));
        }
        seal.update(role.as_bytes());
        seal.update(frame_idx.to_le_bytes());
        seal.update(strength.to_bits().to_le_bytes());
        for (frame_index, frame) in frames.iter().enumerate() {
            if frame.width != width || frame.height != height || frame.pixels.len() != expected {
                return Err(format!(
                    "Bernini ADS2V {role} clip frame {frame_index} is not exact output-sized RGB8"
                ));
            }
            seal.update((frame_index as u32).to_le_bytes());
            seal.update(frame.width.to_le_bytes());
            seal.update(frame.height.to_le_bytes());
            seal.update(&frame.pixels);
        }
        let tokens = bernini_packed_source_tokens(
            frames.len(),
            clip_vae_width as u32,
            clip_vae_height as u32,
        )?;
        tokens_total = tokens_total
            .checked_add(tokens)
            .ok_or_else(|| "Bernini ADS2V combined source-token overflow".to_owned())?;
        entries.push(format!(
            "{role}:frames-{};native-{width}x{height};vae-{}x{}x{};tokens-{tokens}",
            frames.len(),
            (frames.len() - 1) / 4 + 1,
            clip_vae_width / 8,
            clip_vae_height / 8
        ));
    }
    for (index, image) in images.iter().enumerate() {
        let expected = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|pixels| usize::try_from(pixels).ok())
            .ok_or_else(|| "Bernini ADS2V image geometry overflow".to_owned())?;
        if image.pixels.len() != expected {
            return Err(format!(
                "Bernini ADS2V image {index} has malformed RGB8 bytes"
            ));
        }
        seal.update((index as u32).to_le_bytes());
        seal.update(image.width.to_le_bytes());
        seal.update(image.height.to_le_bytes());
        seal.update(&image.pixels);
        let (vit_width, vit_height) = bernini_vit_shape(image.width, image.height);
        let (vae_width, vae_height) = bernini_full_vae_shape(image.width, image.height);
        let tokens = bernini_packed_source_tokens(1, vae_width as u32, vae_height as u32)?;
        tokens_total = tokens_total
            .checked_add(tokens)
            .ok_or_else(|| "Bernini ADS2V combined source-token overflow".to_owned())?;
        entries.push(format!("image-{}:native-{}x{};vit-{vit_width}x{vit_height};vae-{vae_width}x{vae_height};tokens-{tokens}", index + 1, image.width, image.height));
    }
    let count = images.len() + 2;
    Ok(format!("{BERNINI_ADS2V_RECEIPT_DOMAIN}:backend-{backend}:source-preprocess-full-vae624-v1:count-{count}:packed-source-tokens-{tokens_total}:source-ids-{}:{}+{BERNINI_ADS2V_SEAL_DOMAIN}-{:x}", bernini_ads2v_source_ids(count).join(","), entries.join("|"), seal.finalize()))
}

/// Worker twin of the provider's MV2V receipt.  Clips remain ordered and independently sealed;
/// curve lookup receives the complete normalized VAE/DiT and source-id surface but not RGB bytes.
pub(crate) fn bernini_mv2v_clip_receipt(
    lane: VideoLane,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    if !(2..=8).contains(&conditioning.len())
        || conditioning
            .iter()
            .any(|entry| !matches!(entry, Conditioning::VideoClip { .. }))
    {
        return Err(
            "Bernini MV2V admission requires exactly 2-8 ordered VideoClips and no images"
                .to_owned(),
        );
    }
    let backend = match lane {
        VideoLane::Mlx => "mlx",
        VideoLane::Candle => "candle",
    };
    let mut seal = sha2::Sha256::new();
    seal.update(BERNINI_MV2V_SEAL_DOMAIN.as_bytes());
    let mut total_tokens = 0_u64;
    let mut entries = Vec::with_capacity(conditioning.len());
    let (vae_width, vae_height) = bernini_full_vae_shape(width, height);
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|pixels| usize::try_from(pixels).ok())
        .ok_or_else(|| "Bernini MV2V clip geometry overflow".to_owned())?;
    for (clip_index, entry) in conditioning.iter().enumerate() {
        let Conditioning::VideoClip {
            frames,
            frame_idx,
            strength,
        } = entry
        else {
            unreachable!()
        };
        if *frame_idx != 0
            || strength.to_bits() != 1.0_f32.to_bits()
            || !matches!(frames.len(), 45 | 61 | 77)
        {
            return Err(format!("Bernini MV2V clip {clip_index} requires normalized 45/61/77 frames at frame 0 with strength 1"));
        }
        seal.update((clip_index as u32).to_le_bytes());
        seal.update(frame_idx.to_le_bytes());
        seal.update(strength.to_bits().to_le_bytes());
        for (frame_index, frame) in frames.iter().enumerate() {
            if frame.width != width || frame.height != height || frame.pixels.len() != expected {
                return Err(format!("Bernini MV2V clip {clip_index} frame {frame_index} is not exact output-sized RGB8"));
            }
            seal.update((frame_index as u32).to_le_bytes());
            seal.update(frame.width.to_le_bytes());
            seal.update(frame.height.to_le_bytes());
            seal.update(&frame.pixels);
        }
        let tokens =
            bernini_packed_source_tokens(frames.len(), vae_width as u32, vae_height as u32)?;
        total_tokens = total_tokens
            .checked_add(tokens)
            .ok_or_else(|| "Bernini MV2V combined source-token count overflow".to_owned())?;
        entries.push(format!(
            "video-{}:frames-{};native-{width}x{height};vae-{}x{}x{};tokens-{tokens}",
            clip_index + 1,
            frames.len(),
            (frames.len() - 1) / 4 + 1,
            vae_width / 8,
            vae_height / 8,
        ));
    }
    Ok(format!(
        "{BERNINI_MV2V_RECEIPT_DOMAIN}:backend-{backend}:source-preprocess-full-vae624-v1:count-{}:packed-source-tokens-{total_tokens}:source-ids-{}:{}+{BERNINI_MV2V_SEAL_DOMAIN}-{:x}",
        conditioning.len(), bernini_mv2v_source_ids(conditioning.len()).join(","), entries.join("|"), seal.finalize()
    ))
}

/// Content-independent overlay identity used for fitted video-curve lookup. Request byte seals are
/// deliberately excluded; all other axes remain exact.
fn video_curve_overlay(overlay: Option<&str>) -> Option<String> {
    let axes = overlay?
        .split('+')
        .filter(|axis| {
            !axis.starts_with(BERNINI_R2V_SEAL_DOMAIN)
                && !axis.starts_with(BERNINI_MV2V_SEAL_DOMAIN)
                && !axis.starts_with(BERNINI_ADS2V_SEAL_DOMAIN)
                && !axis.starts_with(WAN_VACE_REPLACE_SEAL_DOMAIN)
                && !axis.starts_with(SCAIL2_CARRIER_SEAL_DOMAIN)
                && !axis.starts_with(KREA_V2V_SEAL_DOMAIN)
        })
        .collect::<Vec<_>>();
    (!axes.is_empty()).then(|| axes.join("+"))
}

/// Test-only view of [`video_curve_overlay`], so the sealed-carrier tests can assert that curve
/// lookup strips exactly the request seal and keeps every evidence axis.
#[cfg(test)]
pub(crate) fn video_curve_overlay_for_test(overlay: Option<&str>) -> Option<String> {
    video_curve_overlay(overlay)
}

type VideoDecodeProfileResolver = fn(
    VideoLane,
    &str,
    VideoAdmissionGeometry,
    MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String>;

#[derive(Clone, Copy, Debug)]
struct ResolvedVideoDecodeProfile {
    profile: VideoDecodeMemoryProfile,
    evidence_revision: &'static str,
}

fn no_video_decode_profile(
    _lane: VideoLane,
    _provider_id: &str,
    _geometry: VideoAdmissionGeometry,
    _selection: MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String> {
    Ok(None)
}

/// Resolve the exact provider-owned decode working set for the candidate being graded.
///
/// The selected MLX Wan rung-2 carrier has a narrower profile derived from the same provider planner
/// that executes the request. Every other supported candidate uses the provider's conservative
/// single-pass profile. A runtime bundle that exposes no profile returns `None`, preserving the
/// historical weights-plus-headroom floor; provider validation errors fail closed instead of being
/// rewritten as an unprofiled estimate.
fn packaged_video_decode_profile(
    lane: VideoLane,
    provider_id: &str,
    geometry: VideoAdmissionGeometry,
    selection: MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String> {
    let frames = geometry.estimate_frames().max(1);
    #[cfg(target_os = "macos")]
    if lane == VideoLane::Mlx {
        if selection.strategy == MemoryStrategy::BoundedDecode {
            if let (Some(tile_edge), Some(overlap)) = (
                selection.parameters.decode_tile_edge,
                selection.parameters.decode_overlap,
            ) {
                let selected = runtime_macos::selected_video_decode_memory_profile(
                    provider_id,
                    geometry.width,
                    geometry.height,
                    frames,
                    tile_edge,
                    overlap,
                )
                .map_err(|error| {
                    format!(
                        "{provider_id}: selected video decode profile rejected the admitted carrier: {error}"
                    )
                })?;
                if let Some(profile) = selected {
                    return Ok(Some(ResolvedVideoDecodeProfile {
                        profile,
                        evidence_revision: "video-provider-selected-decode-profile-v1",
                    }));
                }
            }
            // No provider-selected profile means the runtime bundle has not exported a
            // load-bearing working set for this exact carrier. Do not apply the conservative
            // single-pass profile to a bounded-decode candidate: that would erase the very saving
            // the rung selects. The unchanged generic floor remains the honest fallback.
            return Ok(None);
        }
        return Ok(runtime_macos::conservative_video_decode_memory_profile(
            provider_id,
            geometry.width,
            geometry.height,
            frames,
        )
        .map(|profile| ResolvedVideoDecodeProfile {
            profile,
            evidence_revision: "video-provider-conservative-decode-profile-v1",
        }));
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if lane == VideoLane::Candle {
        if selection.strategy == MemoryStrategy::BoundedDecode {
            return Ok(None);
        }
        return Ok(runtime_cuda::conservative_video_decode_memory_profile(
            provider_id,
            geometry.width,
            geometry.height,
            frames,
        )
        .map(|profile| ResolvedVideoDecodeProfile {
            profile,
            evidence_revision: "video-provider-conservative-decode-profile-v1",
        }));
    }
    #[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
    let _ = (lane, provider_id, selection, frames);
    Ok(None)
}

/// A video request's identity for the selector, minus the geometry (which arrives per-call).
pub(crate) struct VideoRequestIdentity<'a> {
    /// Catalog model id. Kept distinct from `route`: two catalog entries may share one engine but
    /// do not thereby share measured artifact/overlay memory (LTX base vs Eros is the live case).
    pub(crate) model_id: &'a str,
    /// Catalog family, kept separate from the provider descriptor's implementation family. A
    /// custom/imported LTX-family entry cannot inherit the built-in base model's measurements.
    pub(crate) model_family: &'a str,
    /// The resolved engine id — the same `resolved_route` spelling every other admission cell uses.
    pub(crate) route: &'a str,
    /// Calibration mode. A text-to-video curve must not silently price an image/keyframe-conditioned
    /// request whose encoder/residency surface was not measured.
    pub(crate) mode: &'a str,
    /// Exact reference count and overlay identity. The only promoted curve currently covers zero
    /// references and no overlay, but these still travel through evidence identity so a future
    /// calibrated surface cannot accidentally inherit the base T2V cell.
    pub(crate) reference_count: u32,
    /// Input carrier shape. Count alone cannot distinguish image conditioning from another future
    /// reference transport, so both are part of curve/evidence identity.
    pub(crate) reference_shape: &'a str,
    /// Overlay identity. Provider-only video modes use the same sealed axis as resident
    /// adapters/enhancers (`provider_video_mode:<resolved mode>`), so a catalog T2V request
    /// cannot borrow the base carrier's curve after its provider workload changes.
    pub(crate) overlay: Option<&'a str>,
    /// Output rate is an evidence axis even when a fitted peak is frame-count based.
    pub(crate) fps: u32,
    pub(crate) lane: VideoLane,
    pub(crate) tier: MemoryNumericTier,
    pub(crate) transformer_variant: Option<Ltx25TransformerVariant>,
    pub(crate) decoder: Option<Ltx25Decoder>,
    /// The contract's live calibration ABI. Carried separately from the optional calibration
    /// identity so an ABI mismatch fails the fitted curve even if a malformed/legacy identity was
    /// minted with a misleading fingerprint.
    pub(crate) calibration_abi: u32,
    /// The live compile-closure digest of the provider being admitted (sc-17774). Both sides carry
    /// the same value on a route with no measured cell, which states plainly that there is no
    /// measured closure to be current against.
    pub(crate) expected_closure_digest: &'a str,
}

/// Core's [`VideoStrategySelector`] seam, answered by the shared ladder selector.
pub(crate) struct LadderVideoSelector<'a> {
    identity: VideoRequestIdentity<'a>,
    contract: &'a MemoryProviderContract,
    budget: Option<Budget>,
    /// The activation-headroom allowance the caller already charges this load. Supplied, never
    /// derived here — see the module doc.
    headroom_bytes: u64,
    /// The shared backend-neutral fitted-curve container. `None` is a normal fail-open state: every
    /// rung retains its pre-existing weights-plus-headroom floor candidate.
    curves: Option<&'a VideoMemoryCurveBundle>,
    /// Measured memory anchors (sc-22507, epic 22505). When the fitted curves miss, a matching
    /// anchor derives a per-phase estimate for the requested geometry analytically — including a
    /// `(geometry, frames)` cell that was never measured. `None` fails open to the floor.
    anchors: Option<&'a sceneworks_core::memory_anchor::MemoryAnchorStore>,
    /// Backend bundle resolver for the provider's load-bearing decode working set. Tests default to
    /// `None` so focused curve/floor fixtures do not accidentally inherit a real provider profile.
    decode_profile: VideoDecodeProfileResolver,
    /// Provider-resident bytes captured as the conservative committed delta around the exact cold
    /// load. Fitted/floor laws model the complete run peak, while the post-load budget is
    /// incremental, so every estimate candidate is reduced by this fixed attribution exactly once.
    attributable_resident_bytes: u64,
    /// A provider claiming more attributable resident bytes than a candidate's complete peak is
    /// an accounting contradiction, not a zero-byte request. Remember it so the admission funnel
    /// can fail closed after core returns through its non-error selector seam.
    accounting_error: Option<String>,
    profile_error: std::cell::RefCell<Option<String>>,
    /// Every `(geometry, selection)` the selector chose, so the caller can recover the selected
    /// PARAMETERS for the geometry core reports as binding. Core's seam returns a rung; the
    /// per-request knobs need the whole `MemorySelection`, and re-deriving it would be a second
    /// selection.
    selections: Vec<VideoSelectedCandidate>,
    /// Unwidened resident candidate for every geometry core graded. The refusal guard consumes the
    /// exact binding geometry's value, including any provider profile, instead of recomputing a
    /// profile-blind weights floor after selection.
    resident_floors: Vec<(VideoAdmissionGeometry, u64)>,
}

#[derive(Clone, Debug)]
struct VideoSelectedCandidate {
    binding_geometry: VideoAdmissionGeometry,
    selection: MemorySelection,
    /// Raw fitted/floor peak before the shared estimate margin. This is the exact demand basis the
    /// provider lifecycle context must receive; `Selection::needed_gb` is already widened.
    predicted_peak_bytes: u64,
    evidence_revision: String,
}

impl<'a> LadderVideoSelector<'a> {
    #[cfg(test)]
    pub(crate) fn new(
        identity: VideoRequestIdentity<'a>,
        contract: &'a MemoryProviderContract,
        budget: Option<Budget>,
        headroom_bytes: u64,
        attributable_resident_bytes: u64,
    ) -> Self {
        Self::with_curve_bundle(
            identity,
            contract,
            budget,
            headroom_bytes,
            attributable_resident_bytes,
            sceneworks_core::video_memory_curves::packaged_video_memory_curves(),
        )
    }

    fn with_curve_bundle(
        identity: VideoRequestIdentity<'a>,
        contract: &'a MemoryProviderContract,
        budget: Option<Budget>,
        headroom_bytes: u64,
        attributable_resident_bytes: u64,
        curves: Option<&'a VideoMemoryCurveBundle>,
    ) -> Self {
        Self {
            identity,
            contract,
            budget,
            headroom_bytes,
            curves,
            anchors: sceneworks_core::memory_anchor::packaged_memory_anchors(),
            decode_profile: no_video_decode_profile,
            attributable_resident_bytes,
            accounting_error: None,
            profile_error: std::cell::RefCell::new(None),
            selections: Vec::new(),
            resident_floors: Vec::new(),
        }
    }

    /// Replace the packaged anchor store (tests only): focused floor/curve fixtures must be able
    /// to prove the anchor path carried — or did not carry — a selection.
    #[cfg(test)]
    fn with_anchor_store(
        mut self,
        anchors: Option<&'a sceneworks_core::memory_anchor::MemoryAnchorStore>,
    ) -> Self {
        self.anchors = anchors;
        self
    }

    fn with_profiles(
        identity: VideoRequestIdentity<'a>,
        contract: &'a MemoryProviderContract,
        budget: Option<Budget>,
        headroom_bytes: u64,
        attributable_resident_bytes: u64,
        curves: Option<&'a VideoMemoryCurveBundle>,
        decode_profile: VideoDecodeProfileResolver,
    ) -> Self {
        let mut selector = Self::with_curve_bundle(
            identity,
            contract,
            budget,
            headroom_bytes,
            attributable_resident_bytes,
            curves,
        );
        selector.decode_profile = decode_profile;
        selector
    }

    /// The gen-core backend this lane grades under. Exhaustive on [`VideoLane`] so a new lane
    /// cannot compile without choosing one — the same posture
    /// `memory_strategy::stale_measured_margin` takes on [`MemoryBackend`].
    pub(crate) const fn backend(&self) -> MemoryBackend {
        match self.identity.lane {
            VideoLane::Mlx => MemoryBackend::Mlx,
            VideoLane::Candle => MemoryBackend::Candle,
        }
    }
}

/// Which phase carries a rung's peak at one geometry.
///
/// A property of the GEOMETRY, not of the model: it is measured to flip inside a single model's
/// envelope (text binds at 11,904 latent tokens, decode at 14,080 — sc-18812). It is therefore
/// derived on every call and never cached, and no code here may assume a model has "a" binding
/// phase. This is the MLX-side answer to the question `KreaTurboPhasePeaks::binding_phase()`
/// answers on the candle side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VideoBindingPhase {
    Conditioning,
    Denoise,
    Decode,
}

/// The per-phase peaks one rung's prediction is made of.
///
/// **Deliberately not reduced to a scalar until the last possible moment.** sc-18810 measured every
/// candidate temporal form missing the AGGREGATE peak by at least 10.26 GiB — about 94x the
/// replicate noise floor — while the same forms land at 0.019–0.44 GiB *per phase*. A prediction
/// that is accurate at all is therefore phase-resolved, and the admission number is the **max over
/// phases**, exactly the discipline `KreaTurboPhasePeaks::peak_gb` already applies on the candle
/// side.
///
/// The fitted laws are **affine cross curves**
/// (`fixedGb + perMpxGb*mpx + perMpxFrameGb*mpx*frames + maxResidualGb`) with large,
/// phase-specific intercepts and conservative per-phase residual floors. A through-origin scalar
/// ratio like `mlx_fit_gate`'s `scaled(bytes) = bytes * scale` cannot express them, which is why
/// this seam keeps the three values apart rather than handing one number across. On an inapplicable
/// curve all three values carry the same historical weights+headroom floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PhasePeaks {
    pub(crate) conditioning_bytes: u64,
    pub(crate) denoise_bytes: u64,
    pub(crate) decode_bytes: u64,
}

impl PhasePeaks {
    /// The admission peak — the max over phases, taken at the single point a scalar is
    /// unavoidable (gen-core's `MemoryEvidence::predicted_peak_bytes` is one number).
    pub(crate) const fn peak_bytes(self) -> u64 {
        let mut peak = self.conditioning_bytes;
        if self.denoise_bytes > peak {
            peak = self.denoise_bytes;
        }
        if self.decode_bytes > peak {
            peak = self.decode_bytes;
        }
        peak
    }

    /// Which phase binds AT THIS GEOMETRY. Ties resolve to the LATER phase, matching
    /// `mlx_fit_gate::binding_phase` so the two never disagree on the same triple.
    pub(crate) const fn binding_phase(self) -> VideoBindingPhase {
        let mut phase = VideoBindingPhase::Conditioning;
        let mut peak = self.conditioning_bytes;
        if self.denoise_bytes >= peak {
            phase = VideoBindingPhase::Denoise;
            peak = self.denoise_bytes;
        }
        if self.decode_bytes >= peak {
            phase = VideoBindingPhase::Decode;
        }
        phase
    }
}

/// The weights+headroom floor for one engaged composition, expressed per phase.
///
/// The fallback scalar is exactly
/// `mlx_fit_gate::estimate_floor_weights_bytes(contract, engaged) + headroom_bytes`, the same
/// number the cold-load residency gate already charges this load. It is phase-blind, so all three
/// phases carry it and [`PhasePeaks::peak_bytes`] returns that scalar unchanged.
fn floor_phase_peaks(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    headroom_bytes: u64,
) -> PhasePeaks {
    let floor = crate::mlx_fit_gate::estimate_floor_weights_bytes(contract, engaged)
        .saturating_add(headroom_bytes);
    PhasePeaks {
        conditioning_bytes: floor,
        denoise_bytes: floor,
        decode_bytes: floor,
    }
}

/// The historical activation floor, strengthened by the provider's own decode working set when the
/// selected runtime bundle exposes one. The generic allowance remains a lower bound: a decode-only
/// profile cannot prove that conditioning or denoise need less activation memory. Conversely, a
/// profile above that allowance is load-bearing and may create a real geometry-dependent refusal.
fn profiled_floor_phase_peaks(
    selector: &LadderVideoSelector<'_>,
    geometry: VideoAdmissionGeometry,
    selection: MemorySelection,
    engaged: &[MemoryStrategy],
) -> (PhasePeaks, Option<&'static str>) {
    let generic = floor_phase_peaks(selector.contract, engaged, selector.headroom_bytes);
    let resolved = match (selector.decode_profile)(
        selector.identity.lane,
        &selector.contract.provider_id,
        geometry,
        selection,
    ) {
        Ok(resolved) => resolved,
        Err(error) => {
            *selector.profile_error.borrow_mut() = Some(error);
            return (generic, None);
        }
    };
    let Some(resolved) = resolved else {
        return (generic, None);
    };
    let weights = crate::mlx_fit_gate::estimate_floor_weights_bytes(selector.contract, engaged);
    let Some(profiled) = resolved
        .profile
        .checked_composed_peak(weights, selector.contract.asset_facts.decoder_bytes)
    else {
        *selector.profile_error.borrow_mut() = Some(format!(
            "{} decode profile cannot compose contract weights {} with decoder bytes {}; refusing inconsistent provider accounting",
            selector.contract.provider_id,
            weights,
            selector.contract.asset_facts.decoder_bytes,
        ));
        return (generic, None);
    };
    let floor = generic.peak_bytes().max(profiled);
    (
        PhasePeaks {
            conditioning_bytes: floor,
            denoise_bytes: floor,
            decode_bytes: floor,
        },
        Some(resolved.evidence_revision),
    )
}

fn curve_backend(lane: VideoLane) -> VideoCurveBackend {
    match lane {
        VideoLane::Mlx => VideoCurveBackend::Mlx,
        VideoLane::Candle => VideoCurveBackend::Candle,
    }
}

fn curve_load_shape(load_shape: gen_core::LoadShape) -> VideoCurveLoadShape {
    match load_shape {
        gen_core::LoadShape::EagerMaterialization => VideoCurveLoadShape::EagerMaterialization,
        gen_core::LoadShape::DeferredMaterialization => {
            VideoCurveLoadShape::DeferredMaterialization
        }
    }
}

fn curve_decode_pass(decode_pass: VideoDecodePass) -> VideoCurveDecodePass {
    match decode_pass {
        VideoDecodePass::SinglePass => VideoCurveDecodePass::SinglePass,
        VideoDecodePass::Tiled => VideoCurveDecodePass::Tiled,
        VideoDecodePass::Unmodelled => VideoCurveDecodePass::Unmodelled,
    }
}

/// Whether the anchor's evidence is CURRENT for the contract being admitted.
///
/// The fitted-curve path demotes to the floor when the contract carries no calibration identity,
/// when the calibration ABI moved, or when the fingerprint no longer names the measured closure.
/// The anchor path must demote on exactly the same events: a pin bump or ABI change that stales the
/// evidence cannot be allowed to leave the anchor derivation live while the fitted curve falls.
///
/// The fingerprint compared here is the calibration campaign's, which is what the anchor's source
/// record carries. sc-22511 re-keys currency on the model's own loader closure; this helper is the
/// single seam that changes when it does. Do not grow a second currency notion beside it.
fn anchor_currency_matches(
    selector: &LadderVideoSelector<'_>,
    anchor: &sceneworks_core::memory_anchor::MemoryAnchor,
) -> bool {
    selector
        .contract
        .calibration
        .as_ref()
        .is_some_and(|calibration| {
            calibration.abi == selector.identity.calibration_abi
                && calibration.fingerprint == anchor.source.calibration_fingerprint
        })
}

/// Derive per-phase peaks from the measured memory anchor for this
/// `(model, tier, lane, transformer variant, decoder)` coordinate (sc-22507, epic 22505), for the
/// exact regime of the candidate being graded.
///
/// Deliberately NOT hull-restricted: the whole point of the anchor + analytic derivation is that
/// a request at a `(geometry, frames)` never measured is priced from the anchor plus architecture
/// facts instead of falling to the phase-blind floor. Identity stays strict — model, family, route,
/// provider, lane, tier, pipeline variant/decoder, mode, overlay-free, reference-free, and current
/// calibration — and the anchor store itself is validated against the retained evidence it was
/// extracted from at load.
fn anchor_derived_phase_peaks<'a>(
    selector: &LadderVideoSelector<'a>,
    geometry: VideoAdmissionGeometry,
    engaged: &[MemoryStrategy],
) -> Option<(PhasePeaks, &'a str)> {
    use sceneworks_core::memory_anchor::{AnchorBackend, AnchorDeriveRequest};

    let store = selector.anchors?;
    let identity = &selector.identity;
    // The anchors were measured overlay-free with zero references; a differently-conditioned
    // surface must not borrow them.
    if identity.overlay.is_some() || identity.reference_count != 0 {
        return None;
    }
    let backend = match identity.lane {
        VideoLane::Mlx => AnchorBackend::Mlx,
        VideoLane::Candle => AnchorBackend::Candle,
    };
    // The pipeline cell is part of the lookup key, exactly as it is for the fitted curve: the
    // retained corpus has no dev-vs-distilled pair at a common regime, so a request on another
    // variant/decoder has no measured basis here and must fall to the floor.
    let anchor = store.anchor_for(
        identity.model_id,
        backend,
        crate::mlx_fit_gate::plan_tier_key(identity.tier),
        identity.transformer_variant?,
        identity.decoder?,
    )?;
    if anchor.model_family != identity.model_family
        || anchor.mode != identity.mode
        || anchor.route != identity.route
        || anchor.provider != selector.contract.provider_id
        || !anchor_currency_matches(selector, anchor)
    {
        return None;
    }
    // `decode_tiled` is keyed on the ENGAGED RUNG, not on `geometry.decode_pass`, because the rung
    // is what actually bounds the decoder's working set: the rung is the selected memory strategy
    // the provider will execute, while `decode_pass` describes how the caller intends to walk the
    // clip. A `VideoDecodePass::Tiled` request without the rung is therefore priced by the
    // single-pass voxel law — an over-estimate, and the safe direction. The fitted path keys on
    // both because its curves are measured per `(rung, decode_pass)` cell; the anchor derivation
    // has one law per regime and only the rung selects between them.
    let derived = anchor.derive_video_phase_peaks(AnchorDeriveRequest {
        width: geometry.width,
        height: geometry.height,
        frames: geometry.estimate_frames(),
        decode_tiled: engaged.contains(&MemoryStrategy::BoundedDecode),
        transformer_windowed: engaged.contains(&MemoryStrategy::BoundedTransformerResidency),
        deferred_materialization: selector.contract.load_shape
            == gen_core::LoadShape::DeferredMaterialization,
    })?;
    Some((
        PhasePeaks {
            conditioning_bytes: derived.conditioning,
            denoise_bytes: derived.denoise,
            decode_bytes: derived.decode,
        },
        anchor.id.as_str(),
    ))
}

/// Prefer the exact fitted per-phase curve for this cell; otherwise preserve the established
/// weights-plus-headroom floor byte-for-byte. The lookup itself owns every fail-closed identity and
/// geometry check, including lane/closure/load-shape/mode and the measured area-by-voxel hull.
///
/// There is deliberately no binding-phase flip band here. Each phase has its own fitted law and the
/// scalar is the max over those three evaluations at this exact geometry, so a phase crossing is a
/// result of the measured curves rather than an extrapolation away from one scalar anchor.
fn fitted_or_floor_phase_peaks<'a>(
    selector: &LadderVideoSelector<'a>,
    geometry: VideoAdmissionGeometry,
    strategy: MemoryStrategy,
    engaged: &[MemoryStrategy],
) -> (
    PhasePeaks,
    CandidateBasis,
    &'a str,
    Option<&'a str>,
    Option<&'static str>,
) {
    let fitted = selector
        .curves
        .zip(selector.contract.calibration.as_ref())
        .and_then(|(bundle, calibration)| {
            if calibration.abi != selector.identity.calibration_abi {
                return None;
            }
            let transformer_variant = selector.identity.transformer_variant?;
            let decoder = selector.identity.decoder?;
            let curve_overlay = video_curve_overlay(selector.identity.overlay);
            bundle.evaluate(VideoCurveQuery {
                model_id: selector.identity.model_id,
                model_family: selector.identity.model_family,
                route: selector.identity.route,
                provider: &selector.contract.provider_id,
                backend: curve_backend(selector.identity.lane),
                tier: crate::mlx_fit_gate::plan_tier_key(selector.identity.tier),
                transformer_variant,
                decoder,
                mode: selector.identity.mode,
                reference_shape: selector.identity.reference_shape,
                reference_count: selector.identity.reference_count,
                frames_per_second: selector.identity.fps,
                overlay: curve_overlay.as_deref(),
                rung: rung_of(strategy),
                load_shape: curve_load_shape(selector.contract.load_shape),
                closure_digest: selector.identity.expected_closure_digest,
                calibration_abi: selector.identity.calibration_abi,
                calibration_fingerprint: &calibration.fingerprint,
                decode_pass: curve_decode_pass(geometry.decode_pass),
                geometry: VideoCurveGeometry {
                    width: geometry.width,
                    height: geometry.height,
                    frames: geometry.estimate_frames(),
                    batch: geometry.batch,
                },
            })
        });
    if let Some(evaluation) = fitted {
        // This candidate is the narrowly ratified binding-phase exemption documented and pinned in
        // `ladder_margin_policy`: every phase has its own residual-bounded law and the scalar below
        // is their request-geometry maximum.
        return (
            PhasePeaks {
                conditioning_bytes: evaluation.phases.conditioning,
                denoise_bytes: evaluation.phases.denoise,
                decode_bytes: evaluation.phases.decode,
            },
            CandidateBasis::EstimateFittedCurve,
            evaluation.closure_digest,
            Some(evaluation.curve_id),
            None,
        );
    }
    // Anchor + analytic derivation (sc-22507): between the exact fitted curve and the
    // phase-blind floor. A never-measured (geometry, frames) cell is priced from the one
    // measured anchor for this (model, tier, lane); the shared selector then applies its
    // ordinary backend estimate margin, exactly as it does for fitted-curve candidates.
    if let Some((derived, anchor_id)) = anchor_derived_phase_peaks(selector, geometry, engaged) {
        return (
            derived,
            CandidateBasis::EstimateAnchorDerived,
            selector.identity.expected_closure_digest,
            Some(anchor_id),
            None,
        );
    }
    let selection = MemorySelection {
        strategy,
        parameters: crate::mlx_fit_gate::estimate_floor_smallest_parameters(
            selector.contract,
            engaged,
        )
        .unwrap_or_default(),
        tier: selector.identity.tier,
    };
    let (floor, profile_revision) =
        profiled_floor_phase_peaks(selector, geometry, selection, engaged);
    (
        floor,
        CandidateBasis::EstimateFloor,
        selector.identity.expected_closure_digest,
        None,
        profile_revision,
    )
}

/// The gen-core geometry for one video admission cell.
///
/// Requested rows retain the actual clip length because `MemoryGeometry` is calibration/evidence
/// identity: f9/chunk8 and f25/chunk8 must not collide. A synthetic single-pass-cap row uses the
/// interior cap it evaluates, so the non-monotonic cap peak can still bind the request.
fn video_memory_geometry(geometry: VideoAdmissionGeometry, reference_count: u32) -> MemoryGeometry {
    MemoryGeometry {
        width: geometry.width,
        height: geometry.height,
        batch: geometry.batch.max(1),
        frames: geometry.estimate_frames().max(1),
        reference_count,
    }
}

impl VideoStrategySelector for LadderVideoSelector<'_> {
    fn select(&mut self, geometry: VideoAdmissionGeometry) -> VideoRungSelection {
        let memory_geometry = video_memory_geometry(geometry, self.identity.reference_count);
        let backend = self.backend();
        let calibration_fingerprint = self
            .contract
            .calibration
            .as_ref()
            .map(|identity| identity.fingerprint.as_str());

        // One exact-fitted-or-floor candidate per rung the provider's OWN contract can execute. A rung
        // the provider has not wired is never offered — predicting a saving the provider will
        // silently ignore is how a staged prediction turns into a SIGKILL
        // (`mlx_fit_gate::engine_supports_sequential`'s rationale, applied through the contract).
        //
        // The SHARED selector enforces it, and nothing here restates the check. Two candidate-side
        // guards were written, found unkillable by individual mutation, and removed rather than
        // kept with tests shaped to match them:
        //
        // * `support == MemoryStrategySupport::Implemented` — gen-core's `validate_selection`
        //   rejects every non-`Implemented` support at `memory_strategy.rs:1458`.
        // * `contract.validate_selection(&selection)` — `memory_strategy::candidate_exclusion`
        //   runs exactly that call on every candidate (`memory_strategy.rs:466`) and excludes it
        //   as `Invalid`.
        //
        // `mlx_fit_gate::synthesize_estimate_ladder` carries both as a pre-filter; here they would
        // be a second copy of selection policy, which is what epic decision 3 says not to build.
        // `a_rung_whose_prerequisite_is_unmet_is_not_offered` pins that the shared exclusion is in
        // force on the video lane, rather than assuming it.
        let mut synthesized = Vec::new();
        for strategy in MemoryStrategy::ALL {
            let engaged = self.contract.engaged_composition(strategy);
            let Some(parameters) =
                crate::mlx_fit_gate::estimate_floor_smallest_parameters(self.contract, &engaged)
            else {
                continue;
            };
            let selection = MemorySelection {
                strategy,
                parameters,
                tier: self.identity.tier,
            };
            // Phase-resolved for as long as possible: the scalar is taken only here, where
            // gen-core's evidence type forces one.
            let (phase_peaks, basis, closure_digest, curve_id, profile_revision) =
                fitted_or_floor_phase_peaks(self, geometry, strategy, &engaged);
            if self.profile_error.borrow().is_some() {
                return VideoRungSelection::Undecidable;
            }
            let absolute_predicted_peak_bytes = phase_peaks.peak_bytes();
            let Some(predicted_peak_bytes) =
                absolute_predicted_peak_bytes.checked_sub(self.attributable_resident_bytes)
            else {
                self.accounting_error = Some(format!(
                    "{} live resident attribution {} exceeds modeled total peak {} for {:?}; \
                     refusing an inconsistent video budget",
                    self.identity.route,
                    self.attributable_resident_bytes,
                    absolute_predicted_peak_bytes,
                    strategy,
                ));
                return VideoRungSelection::Undecidable;
            };
            if strategy == MemoryStrategy::Resident {
                self.resident_floors.push((geometry, predicted_peak_bytes));
            }
            let evidence = crate::mlx_fit_gate::estimate_evidence(
                self.contract,
                backend,
                self.identity.tier,
                self.identity.mode,
                self.identity.overlay,
                memory_geometry,
                selection,
                predicted_peak_bytes,
                calibration_fingerprint,
            );
            synthesized.push((
                selection,
                evidence,
                phase_peaks,
                basis,
                closure_digest,
                curve_id,
                profile_revision,
            ));
        }
        if synthesized.is_empty() {
            return VideoRungSelection::Undecidable;
        }

        let candidates = synthesized
            .iter()
            .map(
                |(selection, evidence, _, basis, closure_digest, _, _)| Candidate {
                    selection: *selection,
                    evidence,
                    closure_digest,
                    basis: *basis,
                    // sc-22508: only the weights+headroom floor decomposes into a counted term and
                    // a flat activation allowance, and `floor_phase_peaks` built this peak as
                    // exactly `estimate_floor_weights_bytes + headroom_bytes`. Fitted-curve and
                    // anchor-derived peaks are phase-resolved and carry no such split.
                    //
                    // MLX ONLY. The headroom term earns a doubling where the modelled allowance is
                    // a guess AND an overshoot is fatal (the MLX allocator aborts the worker). On
                    // candle an allocation failure is a recoverable `Err` and the corpus's
                    // deterministic live-allocation accounting bounds the residual at 2%, so
                    // charging a whole extra headroom there would be strictly MORE conservative
                    // than the blanket policy this story retires — which E3 does not license.
                    unmodeled_activation_bytes: (matches!(basis, CandidateBasis::EstimateFloor)
                        && backend == MemoryBackend::Mlx)
                        .then_some(self.headroom_bytes),
                },
            )
            .collect::<Vec<_>>();

        match crate::memory_strategy::select_strategy(
            RequestScope {
                resolved_route: self.identity.route,
                backend: self.identity.lane.as_key(),
                tier: self.identity.tier,
                mode: self.identity.mode,
                overlay: self.identity.overlay,
                geometry: memory_geometry,
                expected_closure_digest: self.identity.expected_closure_digest,
            },
            self.contract,
            self.budget,
            &candidates,
        ) {
            Selection::Selected {
                selection,
                needed_gb,
                available_gb,
            } => {
                tracing::info!(
                    event = "video_memory_strategy_selected",
                    route = self.identity.route,
                    backend = self.identity.lane.as_key(),
                    strategy = ?selection.strategy,
                    request_frames = geometry.frames,
                    estimate_frames = memory_geometry.frames,
                    decode_pass_frames = geometry.decode_pass_frames,
                    decode_pass = ?geometry.decode_pass,
                    geometry_role = ?geometry.role,
                    // Recomputed for THIS geometry's selected rung, never cached: which phase
                    // binds is a geometry property and flips inside one model's envelope.
                    binding_phase = ?synthesized
                        .iter()
                        .find(|(candidate, ..)| candidate.strategy == selection.strategy)
                        .map(|(_, _, peaks, ..)| peaks.binding_phase()),
                    curve_id = synthesized
                        .iter()
                        .find(|(candidate, ..)| candidate.strategy == selection.strategy)
                        .and_then(|(_, _, _, _, _, curve_id, _)| *curve_id)
                        .unwrap_or("none"),
                    needed_gb,
                    available_gb,
                );
                let selected = synthesized
                    .iter()
                    .find(|(candidate, ..)| *candidate == selection)
                    .expect("the shared selector can only return a submitted video candidate");
                self.selections.push(VideoSelectedCandidate {
                    binding_geometry: geometry,
                    selection,
                    predicted_peak_bytes: selected.1.predicted_peak_bytes,
                    evidence_revision: selected
                        .5
                        .or(selected.6)
                        .unwrap_or("video-estimate-floor-v1")
                        .to_owned(),
                });
                VideoRungSelection::Selected {
                    rung: rung_of(selection.strategy),
                    needed_gb,
                    available_gb,
                }
            }
            Selection::Reject {
                needed_gb,
                available_gb,
            } => VideoRungSelection::Reject {
                needed_gb,
                available_gb,
            },
            // No gradable candidate survived: never block without evidence.
            Selection::Unverified { .. } => VideoRungSelection::Undecidable,
        }
    }
}

/// What the video funnel does with the gate's verdict.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct VideoAdmissionOutcome {
    /// The per-request rung knobs to put on `GenerationRequest::memory`. `None` ⇒ leave the field
    /// exactly as it was before this gate existed (the provider's own defaults) — which is what
    /// [`VideoAdmission::NotRouted`], [`VideoAdmission::Undecidable`], and a selected
    /// [`StrategyRung::Resident`] all produce, so those paths are byte-identical to today.
    pub(crate) memory: Option<gen_core::GenerationMemory>,
    /// Exact selected contract/evidence handed to provider safety and request-scope lifecycle.
    /// Present for a contract-backed Resident selection too, even when `memory` is `None` to
    /// preserve the provider's historical request defaults.
    pub(crate) context: Option<MemoryRunContext>,
    /// The house-convention refusal, or `None` to run.
    pub(crate) refusal: Option<String>,
}

/// Route one video generation through the video gate and turn its verdict into per-request rung
/// knobs (sc-18814). The video lane's counterpart to the image lane's
/// `mlx_fit_gate::evaluate_request`, and deliberately at the same position: after the load, before
/// `generate`.
///
/// # The non-regression guard on floor refusal
///
/// A floor-only refusal here can only fire on a job that already cleared the PRE-load gate
/// (`mlx_fit_gate::apply_residency_policy` → `too_big_error`), because that gate runs first on the
/// cold-load path. The ladder is a superset of that gate's two rungs; suppressing the estimate-only
/// margin band preserves the established behavior. [`refusal_is_a_margin_artifact`] deliberately
/// excludes fitted peaks above that floor, so an applicable measured curve can still make a real,
/// geometry-dependent refusal.
///
/// The pinned provider bundle owns the contract and decode profile. Absence remains a fail-open
/// state; callers must never synthesize a compatibility contract because that could manufacture a
/// refusal for a carrier the provider has not promised to execute.
pub(crate) fn admit_video_generation(
    generator: &dyn gen_core::Generator,
    request: VideoAdmissionInputs<'_>,
) -> VideoAdmissionOutcome {
    admit_video_generation_with_curves_and_profiles(
        generator,
        request,
        sceneworks_core::video_memory_curves::packaged_video_memory_curves(),
        packaged_video_decode_profile,
        true,
    )
}

/// Whether a request has sealed packaged evidence before the worker pays for a live-memory probe.
/// This is deliberately independent of the post-load budget: an unsupported mode/shape/rate must
/// preserve direct generation even when that platform's budget probe would itself fail.
pub(crate) fn packaged_video_evidence_covers_request(
    generator: &dyn gen_core::Generator,
    request: &VideoAdmissionInputs<'_>,
) -> bool {
    if request.reference_shape.trim().is_empty()
        || (request.reference_shape == "none") != (request.reference_count == 0)
    {
        return false;
    }
    let Some(contract) = generator.memory_strategy_contract() else {
        return false;
    };
    curve_evidence_covers_request(
        sceneworks_core::video_memory_curves::packaged_video_memory_curves(),
        contract,
        request,
    ) || anchor_evidence_covers_request(
        sceneworks_core::memory_anchor::packaged_memory_anchors(),
        contract,
        request,
    )
}

/// Whether the packaged anchor store carries the measured anchor this request's
/// `(model, tier, lane, transformer variant, decoder)` coordinate derives from (sc-22507).
/// Coverage is identity-only — no geometry hull — so a never-measured `(geometry, frames)` request
/// still reaches the ladder and its anchor-derived candidate instead of bypassing the gate.
///
/// This mirrors [`anchor_derived_phase_peaks`]'s identity and currency guards exactly. A gate that
/// admitted a request the derivation then refuses would only push it to the floor, but a gate that
/// stayed open across a staled calibration would state the wrong thing about the evidence.
fn anchor_evidence_covers_request(
    anchors: Option<&sceneworks_core::memory_anchor::MemoryAnchorStore>,
    contract: &MemoryProviderContract,
    request: &VideoAdmissionInputs<'_>,
) -> bool {
    let Some(anchors) = anchors else {
        return false;
    };
    if request.overlay.is_some() || request.reference_count != 0 {
        return false;
    }
    let (Some(transformer_variant), Some(decoder)) = (request.transformer_variant, request.decoder)
    else {
        return false;
    };
    let backend = match request.lane {
        VideoLane::Mlx => sceneworks_core::memory_anchor::AnchorBackend::Mlx,
        VideoLane::Candle => sceneworks_core::memory_anchor::AnchorBackend::Candle,
    };
    anchors
        .anchor_for(
            request.model_id,
            backend,
            crate::mlx_fit_gate::plan_tier_key(request.tier),
            transformer_variant,
            decoder,
        )
        .is_some_and(|anchor| {
            anchor.model_family == request.model_family
                && anchor.mode == request.mode
                && anchor.route == request.route
                && anchor.provider == contract.provider_id
                && contract.calibration.as_ref().is_some_and(|calibration| {
                    calibration.abi == gen_core::MEMORY_CALIBRATION_ABI
                        && calibration.fingerprint == anchor.source.calibration_fingerprint
                })
        })
}

/// Test seam for exact fixture contracts/curves. Production always calls
/// [`admit_video_generation`] and therefore consumes only the validated packaged bundle.
#[cfg(test)]
fn admit_video_generation_with_curves(
    generator: &dyn gen_core::Generator,
    request: VideoAdmissionInputs<'_>,
    curves: Option<&VideoMemoryCurveBundle>,
) -> VideoAdmissionOutcome {
    admit_video_generation_with_curves_and_profiles(
        generator,
        request,
        curves,
        no_video_decode_profile,
        false,
    )
}

fn admit_video_generation_with_curves_and_profiles(
    generator: &dyn gen_core::Generator,
    request: VideoAdmissionInputs<'_>,
    curves: Option<&VideoMemoryCurveBundle>,
    decode_profile: VideoDecodeProfileResolver,
    require_request_evidence: bool,
) -> VideoAdmissionOutcome {
    if request.reference_shape.trim().is_empty()
        || (request.reference_shape == "none") != (request.reference_count == 0)
    {
        return VideoAdmissionOutcome::default();
    }
    let contract = generator.memory_strategy_contract();
    if require_request_evidence && !packaged_video_evidence_covers_request(generator, &request) {
        if bernini_memory_attempt(&request) {
            return VideoAdmissionOutcome {
                refusal: Some(bernini_evidence_refusal(&request, contract)),
                ..VideoAdmissionOutcome::default()
            };
        }
        return VideoAdmissionOutcome::default();
    }
    // Provider safety requires a same-moment post-load budget snapshot. A pre-load total-only probe
    // cannot describe unrelated committed bytes or credit already-resident provider bytes exactly
    // once, so a lane without this snapshot fails open instead of forging a context.
    let Some(runtime) = request.runtime else {
        return VideoAdmissionOutcome::default();
    };
    // No provider contract ⇒ no declared rungs ⇒ nothing for the ladder to select between. Fail
    // open, exactly as `mlx_fit_gate` does when a generator publishes no contract.
    let Some(contract) = contract else {
        return VideoAdmissionOutcome::default();
    };
    // The request must have one fully matching fitted curve before any estimate/floor candidate is
    // allowed into the selector. This replaces the historical exact-T2V predicate: a future mode
    // is admitted by adding its own sealed curve, not by weakening a mode/reference/FPS `if`.
    // The production entry already checked the packaged bundle before probing. Test seams can
    // intentionally exercise legacy floor behavior with their own fixture bundle.
    let attributable_resident_bytes = runtime
        .provider_resident_bytes
        .min(runtime.budget.committed_bytes)
        .min(contract.total_resident_bytes());
    let mut selector = LadderVideoSelector::with_profiles(
        VideoRequestIdentity {
            model_id: request.model_id,
            model_family: request.model_family,
            route: request.route,
            mode: request.mode,
            reference_count: request.reference_count,
            reference_shape: request.reference_shape,
            overlay: request.overlay,
            fps: request.fps,
            lane: request.lane,
            tier: request.tier,
            transformer_variant: request.transformer_variant,
            decoder: request.decoder,
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: request.expected_closure_digest,
        },
        contract,
        runtime.selector_budget(),
        request.headroom_bytes,
        attributable_resident_bytes,
        curves,
        decode_profile,
    );
    let verdict = sceneworks_core::video_request::video_admission(
        request.model_id,
        request.lane,
        request.width,
        request.height,
        request.frames,
        request.decode_chunk_size,
        &mut selector,
    );
    if let Some(error) = selector.accounting_error {
        return VideoAdmissionOutcome {
            memory: None,
            context: None,
            refusal: Some(error),
        };
    }
    if let Some(error) = selector.profile_error.into_inner() {
        return VideoAdmissionOutcome {
            memory: None,
            context: None,
            refusal: Some(error),
        };
    }
    let resident_floors = selector.resident_floors;
    let selections = selector.selections;
    match verdict {
        sceneworks_core::video_request::VideoAdmission::NotRouted
        | sceneworks_core::video_request::VideoAdmission::Undecidable => {
            VideoAdmissionOutcome::default()
        }
        sceneworks_core::video_request::VideoAdmission::Admitted { rung, geometry, .. } => {
            let selected = selections.iter().find(|candidate| {
                candidate.binding_geometry == geometry
                    && rung_of(candidate.selection.strategy) == rung
            });
            let Some(selected) = selected else {
                tracing::error!(
                    event = "video_memory_selection_lost",
                    route = request.route,
                    ?rung,
                    ?geometry,
                    "the binding video selection was absent from the selector transcript"
                );
                return VideoAdmissionOutcome::default();
            };
            let calibration = contract.calibration.as_ref();
            // Video candidates are fitted-curve or floor syntheses graded behind the shared
            // estimate margin, and the selector transcript keeps no per-candidate measured-cell
            // basis — so no optimized video selection may claim Calibrated authority here.
            let optimization_authority = if selected.selection.strategy.is_optimized() {
                gen_core::MemoryOptimizationAuthority::Estimated
            } else {
                gen_core::MemoryOptimizationAuthority::Resident
            };
            let context = MemoryRunContext {
                selection: selected.selection,
                optimization_authority,
                calibration_abi: calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |id| id.abi),
                calibration_fingerprint: calibration
                    .map(|id| id.fingerprint.clone())
                    .unwrap_or_default(),
                load_shape: contract.load_shape,
                mode: crate::memory_strategy::memory_mode_from_mode_key(request.mode),
                has_reference: request.reference_count > 0,
                use_pid: false,
                // Phase-resolved evidence is not a multi-phase request modifier. LTX's canonical
                // reference-free T2V scope carries this false.
                has_phases: false,
                // Provider safety needs the ACTUAL request geometry even when the interior
                // single-pass cap supplied the binding peak and selection.
                geometry: MemoryGeometry {
                    width: request.width,
                    height: request.height,
                    batch: 1,
                    frames: request.frames,
                    reference_count: request.reference_count,
                },
                overlay: request.overlay.map(str::to_owned),
                budget: runtime.budget,
                predicted_peak_bytes: selected.predicted_peak_bytes,
                cache_state: runtime.cache_state,
                evidence_revision: selected.evidence_revision.clone(),
            };
            tracing::info!(
                event = "video_memory_context_built",
                route = request.route,
                cache_state = ?runtime.cache_state,
                load_policy = ?runtime.load_policy,
                incremental_predicted_peak_bytes = selected.predicted_peak_bytes,
                attributable_resident_bytes,
                binding_frames = selected.binding_geometry.estimate_frames(),
                request_frames = request.frames,
            );
            VideoAdmissionOutcome {
                memory: contract.generation_memory(&selected.selection),
                context: Some(context),
                refusal: None,
            }
        }
        sceneworks_core::video_request::VideoAdmission::Refused {
            message,
            needed_gb,
            geometry,
            ..
        } => {
            let Some(resident_floor_bytes) = resident_floors
                .iter()
                .find_map(|(graded, bytes)| (*graded == geometry).then_some(*bytes))
            else {
                tracing::error!(
                    event = "video_memory_resident_floor_lost",
                    route = request.route,
                    ?geometry,
                    "the binding refusal was absent from the resident floor transcript"
                );
                return VideoAdmissionOutcome {
                    memory: None,
                    context: None,
                    refusal: Some(message),
                };
            };
            // This scope check is load-bearing on fitted curves: a measured phase peak may exceed
            // the resident weights floor, in which case the refusal is real and must survive.
            if refusal_is_a_margin_artifact(
                needed_gb,
                resident_floor_bytes,
                crate::memory_strategy::floor_admitted_peak_bytes(
                    contract.backend.backend_kind(),
                    resident_floor_bytes,
                    // Mirrors the MLX-only headroom declaration on the floor candidates above, so
                    // this guard compares against the SAME admitted ceiling the selector graded.
                    if contract.backend.backend_kind() == MemoryBackend::Mlx {
                        Some(request.headroom_bytes)
                    } else {
                        None
                    },
                ),
                runtime.selector_budget().and_then(Budget::effective_gb),
            ) {
                tracing::info!(
                    event = "video_memory_strategy_refusal_suppressed",
                    route = request.route,
                    backend = request.lane.as_key(),
                    frames = request.frames,
                    needed_gb,
                    resident_floor_bytes,
                    "ladder rejected inside the estimate-margin band on a peak that IS the \
                     weights+headroom floor; the unwidened floor still fits, so the pre-existing \
                     load gate keeps owning refusal (sc-18814)"
                );
                return VideoAdmissionOutcome::default();
            }
            VideoAdmissionOutcome {
                memory: None,
                context: None,
                refusal: Some(message),
            }
        }
    }
}

fn bernini_memory_attempt(request: &VideoAdmissionInputs<'_>) -> bool {
    request.model_id == "bernini"
        && (matches!(
            request.mode,
            "video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "multi_video_to_video"
        ) || request.reference_count > 0)
}

fn bernini_reference_receipt_has_video(axis: &str) -> bool {
    axis.split_once(':')
        .is_some_and(|(_, suffix)| suffix.contains(":video-1:"))
}

fn bernini_rv2v_receipt_matches_surface(
    axis: &str,
    width: u32,
    height: u32,
    frames: u32,
    reference_count: u32,
) -> bool {
    let (vae_width, vae_height) = bernini_full_vae_shape(width, height);
    let Ok(video_tokens) =
        bernini_packed_source_tokens(frames as usize, vae_width as u32, vae_height as u32)
    else {
        return false;
    };
    let video_marker = format!(
        "video-1:frames-{frames};native-{width}x{height};vae-{}x{}x{};tokens-{video_tokens}",
        (frames - 1) / 4 + 1,
        vae_width / 8,
        vae_height / 8,
    );
    let Some((declared_total, token_surface)) = axis
        .split_once(":packed-source-tokens-")
        .and_then(|(_, suffix)| suffix.split_once(':'))
    else {
        return false;
    };
    let Ok(declared_total) = declared_total.parse::<u64>() else {
        return false;
    };
    let tokens: Vec<_> = token_surface
        .split(";tokens-")
        .skip(1)
        .filter_map(|suffix| {
            suffix
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse::<u64>()
                .ok()
        })
        .collect();
    axis.contains("source-preprocess-full-vae624-v1")
        && axis.matches("video-1:").count() == 1
        && axis.contains(&video_marker)
        && tokens.len() == reference_count as usize
        && tokens
            .iter()
            .try_fold(0_u64, |sum, token| sum.checked_add(*token))
            == Some(declared_total)
}

fn bernini_surface_is_exact(
    request: &VideoAdmissionInputs<'_>,
    contract: Option<&MemoryProviderContract>,
) -> bool {
    let Some(contract) = contract else {
        return false;
    };
    let expected_adapter = contract
        .resident_components()
        .iter()
        .find(|component| component.kind == gen_core::MemoryComponentKind::AdapterStack)
        .map(|component| bernini_adapter_receipt_axis(&component.id));
    let adapter_axis = request.overlay.and_then(|overlay| {
        overlay
            .split('+')
            .find(|axis| axis.starts_with(BERNINI_ADAPTER_RECEIPT_AXIS_DOMAIN))
    });
    let expected_provider_mode = match request.mode {
        "video_to_video" => "provider_video_mode:v2v",
        "reference_to_video" => "provider_video_mode:r2v",
        "reference_video_to_video" => "provider_video_mode:rv2v",
        "multi_video_to_video" => "provider_video_mode:mv2v",
        "ads2v" => "provider_video_mode:ads2v",
        _ => return false,
    };
    let reference_prefix = match request.lane {
        VideoLane::Mlx => {
            "bernini-r2v-references-v2:backend-mlx:source-preprocess-full-vae624-v1:count-"
        }
        VideoLane::Candle => {
            "bernini-r2v-references-v2:backend-candle:source-preprocess-full-vae624-v1:count-"
        }
    };
    let mv2v_prefix = match request.lane {
        VideoLane::Mlx => {
            "bernini-mv2v-clips-v1:backend-mlx:source-preprocess-full-vae624-v1:count-"
        }
        VideoLane::Candle => {
            "bernini-mv2v-clips-v1:backend-candle:source-preprocess-full-vae624-v1:count-"
        }
    };
    let ads2v_prefix = match request.lane {
        VideoLane::Mlx => {
            "bernini-ads2v-sources-v1:backend-mlx:source-preprocess-full-vae624-v1:count-"
        }
        VideoLane::Candle => {
            "bernini-ads2v-sources-v1:backend-candle:source-preprocess-full-vae624-v1:count-"
        }
    };
    let axes: Vec<_> = request
        .overlay
        .unwrap_or_default()
        .split('+')
        .filter(|axis| !axis.is_empty())
        .collect();
    let reference_axis = axes
        .iter()
        .copied()
        .find(|axis| axis.starts_with(reference_prefix));
    let seal_axis = axes
        .iter()
        .copied()
        .find(|axis| axis.starts_with(BERNINI_R2V_SEAL_DOMAIN));
    let reference_axis_count = axes
        .iter()
        .filter(|axis| axis.starts_with(reference_prefix))
        .count();
    let seal_axis_count = axes
        .iter()
        .filter(|axis| axis.starts_with(BERNINI_R2V_SEAL_DOMAIN))
        .count();
    let mv2v_axis = axes
        .iter()
        .copied()
        .find(|axis| axis.starts_with(mv2v_prefix));
    let mv2v_seal = axes
        .iter()
        .copied()
        .find(|axis| axis.starts_with(BERNINI_MV2V_SEAL_DOMAIN));
    let mv2v_axis_count = axes
        .iter()
        .filter(|axis| axis.starts_with(mv2v_prefix))
        .count();
    let mv2v_seal_count = axes
        .iter()
        .filter(|axis| axis.starts_with(BERNINI_MV2V_SEAL_DOMAIN))
        .count();
    let ads2v_axis = axes
        .iter()
        .copied()
        .find(|axis| axis.starts_with(ads2v_prefix));
    let ads2v_seal = axes
        .iter()
        .copied()
        .find(|axis| axis.starts_with(BERNINI_ADS2V_SEAL_DOMAIN));
    let ads2v_axis_count = axes
        .iter()
        .filter(|axis| axis.starts_with(ads2v_prefix))
        .count();
    let ads2v_seal_count = axes
        .iter()
        .filter(|axis| axis.starts_with(BERNINI_ADS2V_SEAL_DOMAIN))
        .count();
    let reference_count = reference_axis.and_then(|axis| {
        axis.strip_prefix(reference_prefix)?
            .split_once(':')?
            .0
            .parse::<u32>()
            .ok()
    });
    let receipt_has_video = reference_axis.is_some_and(bernini_reference_receipt_has_video);
    let overlay_is_exact = adapter_axis == expected_adapter.as_deref()
        && axes.contains(&expected_provider_mode)
        && axes.iter().all(|axis| {
            *axis == expected_provider_mode
                || Some(*axis) == expected_adapter.as_deref()
                || axis.starts_with(reference_prefix)
                || axis.starts_with(BERNINI_R2V_SEAL_DOMAIN)
                || axis.starts_with(mv2v_prefix)
                || axis.starts_with(BERNINI_MV2V_SEAL_DOMAIN)
                || axis.starts_with(ads2v_prefix)
                || axis.starts_with(BERNINI_ADS2V_SEAL_DOMAIN)
        });
    let carrier_is_exact = match request.mode {
        "video_to_video" => {
            request.reference_count == 1
                && request.reference_shape == "video"
                && reference_axis_count == 0
                && seal_axis_count == 0
        }
        "reference_to_video" => {
            (1..=8).contains(&request.reference_count)
                && request.reference_shape == "multi_image"
                && reference_count == Some(request.reference_count)
                && !receipt_has_video
                && reference_axis_count == 1
                && seal_axis.is_some()
                && seal_axis_count == 1
        }
        "reference_video_to_video" => {
            (2..=9).contains(&request.reference_count)
                && request.reference_shape == "video+multi_image"
                && reference_count == Some(request.reference_count - 1)
                && receipt_has_video
                && reference_axis.is_some_and(|axis| {
                    bernini_rv2v_receipt_matches_surface(
                        axis,
                        request.width,
                        request.height,
                        request.frames,
                        request.reference_count,
                    )
                })
                && reference_axis_count == 1
                && seal_axis.is_some()
                && seal_axis_count == 1
        }
        "multi_video_to_video" => {
            (2..=8).contains(&request.reference_count)
                && request.reference_shape == "multi_video"
                && mv2v_axis.is_some_and(|axis| {
                    axis.contains(&format!(":count-{}:", request.reference_count))
                        && axis.matches("video-").count() == request.reference_count as usize
                        && axis.contains(&format!(
                            "source-ids-{}",
                            bernini_mv2v_source_ids(request.reference_count as usize).join(",")
                        ))
                })
                && mv2v_axis_count == 1
                && mv2v_seal.is_some()
                && mv2v_seal_count == 1
                && reference_axis_count == 0
                && seal_axis_count == 0
        }
        "ads2v" => {
            (3..=10).contains(&request.reference_count)
                && request.reference_shape == "ads2v"
                && ads2v_axis.is_some_and(|axis| {
                    axis.contains(&format!(":count-{}:", request.reference_count))
                        && axis.matches("source-video:").count() == 1
                        && axis.matches("reference-video:").count() == 1
                        && axis.matches("image-").count()
                            == request.reference_count.saturating_sub(2) as usize
                        && axis.contains(&format!(
                            "source-ids-{}",
                            bernini_ads2v_source_ids(request.reference_count as usize).join(",")
                        ))
                })
                && ads2v_axis_count == 1
                && ads2v_seal.is_some()
                && ads2v_seal_count == 1
                && reference_axis_count == 0
                && seal_axis_count == 0
                && mv2v_axis_count == 0
                && mv2v_seal_count == 0
        }
        _ => false,
    };
    request.model_id == "bernini"
        && carrier_is_exact
        && request.fps == 16
        && matches!(
            (request.width, request.height),
            (848, 480) | (480, 848) | (1280, 720) | (720, 1280)
        )
        && matches!(request.frames, 45 | 61 | 77)
        && request.tier.precision == gen_core::Precision::Bf16
        && matches!(
            request.tier.quant,
            None | Some(gen_core::Quant::Q4) | Some(gen_core::Quant::Q8)
        )
        && overlay_is_exact
}

fn bernini_evidence_refusal(
    request: &VideoAdmissionInputs<'_>,
    contract: Option<&MemoryProviderContract>,
) -> String {
    if !bernini_surface_is_exact(request, contract) {
        return "Bernini memory admission refused: exact surface requires V2V, R2V, RV2V, MV2V, or ADS2V [source VideoClip, reference VideoClip, ordered MultiReference 1-8 images]; FPS16, frames 45/61/77, one public geometry, supported tier, backend-specific source receipt, and the exact loaded adapter receipt".to_owned();
    }
    format!(
        "Bernini {} memory admission refused: no current calibrated evidence matches route={}, lane={}, tier={:?}, geometry={}x{} frames={} references={} shape={} overlay={:?}",
        request.mode,
        request.route,
        request.lane.as_key(),
        request.tier.quant,
        request.width,
        request.height,
        request.frames,
        request.reference_count,
        request.reference_shape,
        request.overlay
    )
}

fn curve_evidence_covers_request(
    curves: Option<&VideoMemoryCurveBundle>,
    contract: &MemoryProviderContract,
    request: &VideoAdmissionInputs<'_>,
) -> bool {
    let Some(curves) = curves else {
        return false;
    };
    let Some(calibration) = contract.calibration.as_ref() else {
        return false;
    };
    let (Some(transformer_variant), Some(decoder)) = (request.transformer_variant, request.decoder)
    else {
        return false;
    };
    let geometries = sceneworks_core::video_request::video_admission_geometries(
        request.model_id,
        request.lane,
        request.width,
        request.height,
        request.frames,
        request.decode_chunk_size,
    );
    let curve_overlay = video_curve_overlay(request.overlay);
    curves.curves.iter().any(|curve| {
        curve.model_id == request.model_id
            && curve.model_family == request.model_family
            && curve.route == request.route
            && curve.provider == contract.provider_id
            && curve.backend == curve_backend(request.lane)
            && curve.tier == crate::mlx_fit_gate::plan_tier_key(request.tier)
            && curve.transformer_variant == transformer_variant
            && curve.decoder == decoder
            && curve.mode == request.mode
            && curve.reference_shape == request.reference_shape
            && curve.reference_count == request.reference_count
            && curve.frames_per_second.contains(&request.fps)
            && curve.overlay.as_deref() == curve_overlay.as_deref()
            && curve.calibration_abi == gen_core::MEMORY_CALIBRATION_ABI
            && curve.calibration_fingerprint == calibration.fingerprint
            && geometries.iter().all(|geometry| {
                curves
                    .evaluate(VideoCurveQuery {
                        model_id: request.model_id,
                        model_family: request.model_family,
                        route: request.route,
                        provider: &contract.provider_id,
                        backend: curve_backend(request.lane),
                        tier: crate::mlx_fit_gate::plan_tier_key(request.tier),
                        transformer_variant,
                        decoder,
                        mode: request.mode,
                        reference_shape: request.reference_shape,
                        reference_count: request.reference_count,
                        frames_per_second: request.fps,
                        overlay: curve_overlay.as_deref(),
                        rung: curve.rung,
                        load_shape: curve_load_shape(contract.load_shape),
                        closure_digest: request.expected_closure_digest,
                        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
                        calibration_fingerprint: &calibration.fingerprint,
                        decode_pass: curve_decode_pass(geometry.decode_pass),
                        geometry: VideoCurveGeometry {
                            width: geometry.width,
                            height: geometry.height,
                            frames: geometry.estimate_frames(),
                            batch: geometry.batch,
                        },
                    })
                    .is_some()
            })
    })
}

/// Whether a ladder rejection may be suppressed as a pure **estimate-margin artifact** on a
/// **floor-shaped peak** — the only refusal `admit_video_generation`'s non-regression guard is
/// entitled to swallow.
///
/// Both conjuncts are load-bearing and neither implies the other:
///
/// 1. **`needed_gb` must not exceed the widened resident floor.** This is the scope check, and it
///    is the whole reason this function exists rather than the budget comparison alone. On a
///    fallback [`floor_phase_peaks`] candidate this holds; on a fitted affine per-phase curve it
///    may not. A guard that compared only the floor to the budget would suppress a genuine
///    all-rungs-reject on a host that fits the weights but not the fitted phase peak, return
///    [`VideoAdmissionOutcome::default()`], and run the job resident into an OOM. Comparing the
///    REJECTED peak keeps the suppression attached to the claim its name makes.
/// 2. **The unwidened resident floor must fit.** This is the non-regression condition proper: it
///    is the same weights+headroom-vs-budget comparison `mlx_fit_gate::fit_decision` already
///    applies to this load, so a job the pre-load gate admits today is never newly refused here.
///
/// No budget signal ⇒ `false`: with nothing to compare against, the ladder cannot have rejected on
/// a margin (`select_strategy` returns `Unverified`, not `Reject`), and suppressing on no evidence
/// would be the inverse of the house never-block-without-evidence posture.
fn refusal_is_a_margin_artifact(
    needed_gb: f64,
    resident_floor_bytes: u64,
    admitted_floor_bytes: u64,
    available_gb: Option<f64>,
) -> bool {
    let Some(available_gb) = available_gb else {
        return false;
    };
    // The caller produces `admitted_floor_bytes` through the SAME policy function `select_strategy`
    // graded the floor candidate with (sc-22508), so the two sides of the comparison are produced
    // by one conversion rather than two roundings.
    let widened_floor_gb = crate::memory_strategy::peak_bytes_to_gb(admitted_floor_bytes);
    let floor_gb = crate::memory_strategy::peak_bytes_to_gb(resident_floor_bytes);
    needed_gb <= widened_floor_gb && floor_gb <= available_gb
}

/// Everything `admit_video_generation` needs that is not on the generator.
pub(crate) struct VideoAdmissionInputs<'a> {
    pub(crate) model_id: &'a str,
    /// The resolved catalog family, not the engine descriptor family.
    pub(crate) model_family: &'a str,
    /// The resolved engine id.
    pub(crate) route: &'a str,
    /// Evidence-mode key (`text_to_video`, `image_to_video`, or another explicitly measured mode).
    pub(crate) mode: &'a str,
    pub(crate) reference_count: u32,
    /// Exact carrier used by the references. Must be `none` iff `reference_count` is zero.
    pub(crate) reference_shape: &'a str,
    pub(crate) overlay: Option<&'a str>,
    pub(crate) lane: VideoLane,
    pub(crate) tier: MemoryNumericTier,
    pub(crate) transformer_variant: Option<Ltx25TransformerVariant>,
    pub(crate) decoder: Option<Ltx25Decoder>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    /// Provider-resolved VAE temporal chunk. `None` means one invocation sees the whole clip; zero
    /// is normalized by core to the providers' minimum of one.
    pub(crate) decode_chunk_size: Option<u32>,
    /// Output FPS is part of the measured surface even though the affine curve is frame-count
    /// keyed. SC-18810 exercised 24 and 30 fps; values outside that envelope fail open.
    pub(crate) fps: u32,
    pub(crate) runtime: Option<VideoRuntimeMemoryState>,
    pub(crate) headroom_bytes: u64,
    pub(crate) expected_closure_digest: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VideoRuntimeMemoryState {
    pub(crate) budget: MemoryBudget,
    pub(crate) cache_state: MemoryCacheState,
    pub(crate) load_policy: OffloadPolicy,
    /// Fixed backend-committed delta captured around this generator's cold load. This is not
    /// recomputed from a historical external baseline on warm requests.
    pub(crate) provider_resident_bytes: u64,
}

impl VideoRuntimeMemoryState {
    fn selector_budget(self) -> Option<Budget> {
        const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        Some(Budget {
            available_gb: self
                .budget
                .total_bytes
                .saturating_sub(self.budget.committed_bytes) as f64
                / BYTES_PER_GIB,
            reclaimable_gb: self.budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
            total_gb: self.budget.total_bytes as f64 / BYTES_PER_GIB,
            reserved_headroom_gb: self.budget.reserved_headroom_bytes as f64 / BYTES_PER_GIB,
        })
    }
}

/// Capture the same post-load MLX budget snapshot the image request gate uses. The generator-cache
/// callback supplies the fixed cold-load provider delta and cache state separately, so admission
/// can credit only this provider's already-resident bytes while leaving unrelated live allocations
/// charged.
#[cfg(target_os = "macos")]
pub(crate) fn live_video_runtime_state(
    engine_id: &str,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    Ok(Some(VideoRuntimeMemoryState {
        // Keep the image lane's canonical foreign/system reserve. The caller removes this exact
        // amount from the fallback activation allowance, yielding 16 + 2 rather than 18 + 2;
        // fitted active-memory curves likewise retain the 2 GiB reserve exactly once.
        budget: crate::mlx_fit_gate::live_request_budget(engine_id)?,
        cache_state,
        load_policy,
        provider_resident_bytes,
    }))
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn candle_budget_from_total_free(
    raw_total_bytes: u64,
    raw_free_bytes: u64,
    cap_bytes: Option<u64>,
) -> Option<MemoryBudget> {
    if raw_total_bytes == 0 || raw_free_bytes > raw_total_bytes {
        return None;
    }
    let raw_committed = raw_total_bytes.saturating_sub(raw_free_bytes);
    let total_bytes = cap_bytes.unwrap_or(raw_total_bytes).min(raw_total_bytes);
    Some(MemoryBudget {
        total_bytes,
        committed_bytes: raw_committed.min(total_bytes),
        reclaimable_bytes: 0,
        reserved_headroom_bytes: (crate::fit_gate::DEDICATED_VRAM_ALLOCATOR_SLACK_GB
            * 1024.0
            * 1024.0
            * 1024.0)
            .ceil() as u64,
    })
}

/// Candle reads CUDA's driver allocation counters synchronously after load, on the serialized
/// generator thread. This is the same snapshot used for both live committed pressure and the fixed
/// cold-load provider attribution; an emulation cap changes total capacity but never erases raw
/// committed bytes.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle", not(test)))]
pub(crate) fn live_video_runtime_state(
    _engine_id: &str,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    let (free, total) =
        runtime_cuda::media::candle_core::cuda::cudarc::driver::result::mem_get_info().map_err(
            |error| crate::WorkerError::Engine(format!("CUDA VRAM snapshot failed: {error}")),
        )?;
    let cap_bytes = crate::vram_gate::cuda_vram_cap_gb()
        .map(|gb| (gb * 1024.0 * 1024.0 * 1024.0).floor() as u64);
    Ok(
        candle_budget_from_total_free(total as u64, free as u64, cap_bytes).map(|budget| {
            VideoRuntimeMemoryState {
                budget,
                cache_state,
                load_policy,
                provider_resident_bytes,
            }
        }),
    )
}

// Candle unit tests inject backend-neutral generators and create no CUDA context. Give those seams
// a deterministic synthetic device budget; non-test binaries retain the physical driver snapshot
// above, which the exact-head CUDA acceptance run exercises with real weights.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle", test))]
pub(crate) fn live_video_runtime_state(
    _engine_id: &str,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    const TEST_TOTAL_BYTES: u64 = 24 * 1024 * 1024 * 1024;
    Ok(
        candle_budget_from_total_free(TEST_TOTAL_BYTES, TEST_TOTAL_BYTES, None).map(|budget| {
            VideoRuntimeMemoryState {
                budget,
                cache_state,
                load_policy,
                provider_resident_bytes,
            }
        }),
    )
}

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
pub(crate) fn live_video_runtime_state(
    _engine_id: &str,
    _cache_state: MemoryCacheState,
    _load_policy: OffloadPolicy,
    _provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    Ok(None)
}

/// Which lane this build's video path executes on.
#[cfg(target_os = "macos")]
pub(crate) const LANE: VideoLane = VideoLane::Mlx;
#[cfg(not(target_os = "macos"))]
pub(crate) const LANE: VideoLane = VideoLane::Candle;

/// Bridge the gen-core strategy back to the rung spelling `sceneworks-core` returns. The inverse of
/// `memory_strategy::strategy_from_rung`, and exhaustive for the same reason.
const fn rung_of(strategy: MemoryStrategy) -> StrategyRung {
    match strategy {
        MemoryStrategy::Resident => StrategyRung::Resident,
        MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
        MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
        MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency => StrategyRung::BoundedTransformerResidency,
    }
}

#[cfg(test)]
mod tests;
