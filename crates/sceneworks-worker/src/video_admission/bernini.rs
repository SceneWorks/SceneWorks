//! video_admission bernini owner.
use super::*;

pub(crate) const BERNINI_R2V_RECEIPT_DOMAIN: &str = "bernini-r2v-references-v2";

pub(crate) const BERNINI_R2V_SEAL_DOMAIN: &str = "bernini-r2v-request-seal-v1";

pub(crate) const BERNINI_MV2V_RECEIPT_DOMAIN: &str = "bernini-mv2v-clips-v1";

pub(crate) const BERNINI_MV2V_SEAL_DOMAIN: &str = "bernini-mv2v-request-seal-v1";

pub(crate) const BERNINI_ADS2V_RECEIPT_DOMAIN: &str = "bernini-ads2v-sources-v1";

pub(crate) const BERNINI_ADS2V_SEAL_DOMAIN: &str = "bernini-ads2v-request-seal-v1";

pub(crate) const BERNINI_ADAPTER_RECEIPT_AXIS_DOMAIN: &str = "bernini-adapter-receipt-v1";

pub(crate) fn bernini_adapter_receipt_axis(component_id: &str) -> String {
    let mut digest = sha2::Sha256::new();
    digest.update(BERNINI_ADAPTER_RECEIPT_AXIS_DOMAIN.as_bytes());
    digest.update(component_id.as_bytes());
    format!(
        "{BERNINI_ADAPTER_RECEIPT_AXIS_DOMAIN}-{:x}",
        digest.finalize()
    )
}

pub(super) fn round_half_even(value: f64) -> i64 {
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

pub(super) fn bernini_vit_shape(width: u32, height: u32) -> (i64, i64) {
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
pub(super) fn bernini_full_vae_shape(width: u32, height: u32) -> (i64, i64) {
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

pub(super) fn bernini_packed_source_tokens(
    frames: usize,
    width: u32,
    height: u32,
) -> Result<u64, String> {
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

pub(super) fn bernini_mv2v_source_ids(count: usize) -> Vec<String> {
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

pub(super) fn bernini_ads2v_source_ids(count: usize) -> Vec<String> {
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

pub(super) fn bernini_memory_attempt(request: &VideoAdmissionInputs<'_>) -> bool {
    request.model_id == "bernini"
        && (matches!(
            request.mode,
            "video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "multi_video_to_video"
        ) || request.reference_count > 0)
}

pub(super) fn bernini_reference_receipt_has_video(axis: &str) -> bool {
    axis.split_once(':')
        .is_some_and(|(_, suffix)| suffix.contains(":video-1:"))
}

pub(super) fn bernini_rv2v_receipt_matches_surface(
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

pub(super) fn bernini_surface_is_exact(
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

/// Why a Bernini request is outside the SUPPORTED SURFACE — never "why it is unmeasured".
///
/// sc-22512 deleted the second arm this function used to have, which formatted
/// "no current calibrated evidence matches route=…, lane=…, tier=…" for a well-formed request on a
/// coordinate nobody had captured. That message was the last place in video admission where the
/// absence of a measurement produced a denial instead of an abstention (E8), and its only remaining
/// caller now guards on `!bernini_surface_is_exact`, so the arm was also dead.
pub(super) fn bernini_surface_refusal() -> String {
    "Bernini memory admission refused: exact surface requires V2V, R2V, RV2V, MV2V, or ADS2V [source VideoClip, reference VideoClip, ordered MultiReference 1-8 images]; FPS16, frames 45/61/77, one public geometry, supported tier, backend-specific source receipt, and the exact loaded adapter receipt".to_owned()
}
