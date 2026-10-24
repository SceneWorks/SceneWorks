//! Bernini MLX adoption of the shared memory-strategy contract (sc-15528) — the **complete** ladder
//! over a dual-expert Wan2.2-A14B trunk.
//!
//! Both registered Bernini providers adopt it. `bernini` is the entry the SceneWorks catalog's
//! `bernini_image` maps onto (the worker sets `frames: Some(1)`); `bernini_renderer` is the
//! renderer-only sibling. They share one architecture, one staged phase order, one tiled VAE decode,
//! one bounded-attention primitive and one block stream, and differ only in whether the Qwen2.5-VL
//! planner runs — so they share one contract builder. Sharing code is explicitly NOT what makes a
//! catalog entry Verified: each entry still owes its own per-tier evidence.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | Both experts co-resident for the whole denoise — today's shipped behaviour |
//! | 1 Staged residency | Missing | The provider releases phases unconditionally, but exposes no request-selected staged-residency lever |
//! | 2 Bounded decode | Implemented | [`TilingConfig::spatial_only`] over the [`DECODE_TILE_EDGES`] ladder, replacing `TilingConfig::auto` |
//! | 3 Bounded attention | Implemented (trunk) | [`mlx_gen::attention::sdpa_budgeted_bhsd`] through both trunk SDPA seams, on both experts |
//! | 4 Bounded transformer residency | Implemented (deferred-materialization loads) | [`mlx_gen::block_residency::run_windowed`] over the **80-block** trunk |
//!
//! ## Why the trunk is 80 blocks and not 40 (sc-16354)
//!
//! `bernini.rs` refuses a single-expert snapshot outright (`if !config.dual_model { return Err(…) }`)
//! and every `dual_model` Wan config is `num_layers: 40`, so the trunk is `2 x 40 = 80` blocks,
//! code-enforced, and this is the only shape the family has. sc-16354 raised the hypothesis that a
//! naive per-expert window would bound the active expert while the **idle** expert's 40 blocks stayed
//! fully resident — buying at most half of what the arithmetic suggests, and possibly nothing at the
//! request level.
//!
//! This implementation does not inherit that conclusion; it removes its precondition. Rung 4 loads
//! **both** experts deferred ([`WanTransformer::from_weights_deferred`](mlx_gen_wan::WanTransformer::from_weights_deferred)),
//! so neither expert holds any blocks and there is no idle half to pay for. The window is therefore
//! declared and validated over the whole 80-block trunk in one global index space — high-noise blocks
//! are `0..40`, low-noise blocks are `40..80` — rather than as two independent 40-block plans.
//!
//! That global indexing is load-bearing and it constrains the published cadence domain: the shared
//! [`MlxRequestScopeCore`](mlx_gen::request_scope::MlxRequestScopeCore) requires a window start
//! aligned to the window size, and the low expert's first block is index 40, so **every published
//! window size must divide 40**. [`TRANSFORMER_WINDOW_SIZES`] is exactly the set of proper divisors
//! of 40, and [`the_published_window_sizes_all_divide_one_expert`](self) pins it.
//!
//! The cheap half of the same problem — releasing the *high* expert at the monotone boundary switch
//! so only one expert is resident during the denoise — is NOT landed here and is not claimed: see
//! `_RUNG_ONE_IS_UNCONDITIONAL`. Rung 4 achieves strictly more (zero blocks of both experts), so
//! nothing in this ladder depends on it.
//!
//! ## The cost side of rung 4, priced honestly
//!
//! A window plan is re-walked once per forward, and the guided-velocity modes run several full
//! forwards per step — `VitMode::VaeTxtVitWapg`, the `bernini_image` default, runs four. At the
//! variant's real 40 denoise steps that is `40 blocks x 40 steps x 4 passes = 6400` window
//! materializations against one expert per render. That is a latency consequence to price into the
//! chosen cadence, not a correctness problem, and it is why [`TRANSFORMER_WINDOW_SIZE`] defaults to a
//! wide cadence rather than to 1. The full domain down to 1 is still published, because narrowing a
//! mechanism's advertised surface on a latency judgement the caller is better placed to make is not
//! this contract's job.
//!
//! ## Disclosure: an optimized request may EVICT the warm cross-request cache
//!
//! Rung 4 is request-scoped and never materializes a block at all, so a warm generator that served a
//! windowed request has no block residency to hand the next one — every subsequent request re-reads
//! the stack it needs. Stated here rather than discovered as a latency regression.
//!
//! ## What is NOT declared, and why
//!
//! * **PiD decode.** Bernini decodes through the Wan z16 `AutoencoderKLWan`; the PiD students are the
//!   FLUX and Qwen-Image ones. The crate has no `mlx-gen-pid` dependency, so `use_pid` is rejected at
//!   admission rather than silently degraded to the native decode.
//! * **Adapters.** The descriptor advertises `supports_lora: false`, and Wan MERGES adapter deltas
//!   into the weight map at load, so a streamed block re-read from the snapshot would silently carry
//!   none of them. [`WanBlockStream::new`](mlx_gen_wan::WanBlockStream) refuses an adapted load.
//! * **`TransformerComponent::TextEncoder` / `Both`.** See [`TRANSFORMER_WINDOW_COMPONENTS`].

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode,
    MemoryNumericTier, MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport,
    ResidentRequestMemory, Result as CoreResult, TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec};
use mlx_gen_wan::config::WanModelConfig;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The full Bernini pipeline — the provider the SceneWorks `bernini_image` catalog entry resolves to.
pub const FULL_ID: &str = crate::bernini::MODEL_ID;
/// The renderer-only sibling.
pub const RENDERER_ID: &str = crate::pipeline::MODEL_ID;

/// Both adopting providers, in registration order.
pub const PROVIDER_IDS: [&str; 2] = [RENDERER_ID, FULL_ID];

/// Calibration identity for the weights-free registry conformance walk.
///
/// Deliberately distinct from [`MEMORY_CALIBRATION_FINGERPRINT`]: this one describes *declaration*
/// behaviour over a synthetic spec, and must never be mistaken for a measured production cell.
///
/// `-v2` (sc-18609): the registered route fixture changed shape — it now declares the single-phase,
/// one-frame still-image route it actually executes instead of claiming `has_phases`. Versioning the
/// *declaration* key rather than [`MEMORY_CALIBRATION_FINGERPRINT`] keeps the (still unminted)
/// production identity from being restamped by a route correction.
const STATIC_CALIBRATION: &str = "bernini-mlx-registry-behavior-v3";

/// The production calibration identity of the `(bernini, q4)` cell — the one this family's
/// evidence campaign names.
///
/// Since sc-22737 this is one member of the per-`(provider, tier)` family
/// [`production_calibration_fingerprint`] mints, not the only string the crate can publish. It is
/// retained BYTE-FOR-BYTE because it is the key an existing record is filed under; the format the
/// function interpolates reproduces it exactly for that cell.
///
/// A published fingerprint is a key for evidence, never a claim that evidence exists:
/// `MemoryEvidence::optimized_eligibility` still refuses every optimized fit for a cell with no
/// record, so the resident path runs and no selector claims a saving this repository cannot show.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "bernini-image-q4-mlx-dual-expert-ladder-v1";

/// Rung 2 — production decode tile edges, in **output pixels**.
///
/// `TilingConfig` converts to latent by dividing by the VAE's spatial scale (8 for the z16
/// `AutoencoderKLWan`), so every published edge is a multiple of 8 and lands on a whole latent cell.
/// The floor is geometric rather than measured: a tile must exceed twice the overlap by at least one
/// latent cell or successive tiles do not advance, which puts the smallest admissible edge at
/// `2 * DECODE_OVERLAP + 8 = 136`. 192 is the first published multiple comfortably above it.
pub const DECODE_TILE_EDGES: &[u32] = &[512, 448, 384, 320, 256, 192];

/// The default edge when a request enables rung 2 without naming one.
pub const DECODE_TILE_EDGE: u32 = 512;

/// The single published overlap. One overlap per route keeps admission able to reject a geometry
/// assembled for a different one.
pub const DECODE_OVERLAP: u32 = 64;
pub const ADVERTISED_GEOMETRIES: &[(u32, u32)] =
    &[(848, 480), (480, 848), (1280, 720), (720, 1280)];

/// Content-independent memory-evidence identity and request-only byte-seal domains.
pub const R2V_REFERENCE_RECEIPT_DOMAIN: &str = "bernini-r2v-references-v2";
pub const R2V_REFERENCE_SEAL_DOMAIN: &str = "bernini-r2v-request-seal-v1";
/// Ordered full-length source-video identity for the MV2V route.  This is deliberately a
/// different domain from R2V/RV2V: a curve for one source carrier must never be reused for N.
pub const MV2V_RECEIPT_DOMAIN: &str = "bernini-mv2v-clips-v1";
pub const MV2V_SEAL_DOMAIN: &str = "bernini-mv2v-request-seal-v1";
/// ADS2V has two role-distinct videos followed by 1-8 ordered images.  It must never reuse the
/// neighbouring MV2V or RV2V curve identity merely because their flattened source counts match.
pub const ADS2V_RECEIPT_DOMAIN: &str = "bernini-ads2v-sources-v1";
pub const ADS2V_SEAL_DOMAIN: &str = "bernini-ads2v-request-seal-v1";

fn is_reference_receipt_axis(axis: &str) -> bool {
    axis.strip_prefix(R2V_REFERENCE_RECEIPT_DOMAIN)
        .is_some_and(|suffix| suffix.starts_with(":backend-mlx:source-preprocess-"))
}

fn is_mv2v_receipt_axis(axis: &str) -> bool {
    axis.strip_prefix(MV2V_RECEIPT_DOMAIN)
        .is_some_and(|suffix| suffix.starts_with(":backend-mlx:source-preprocess-"))
}

fn is_ads2v_receipt_axis(axis: &str) -> bool {
    axis.strip_prefix(ADS2V_RECEIPT_DOMAIN)
        .is_some_and(|suffix| suffix.starts_with(":backend-mlx:source-preprocess-"))
}

fn reference_receipt_has_video(axis: &str) -> bool {
    axis.split_once(':')
        .is_some_and(|(_, suffix)| suffix.contains(":video-1:"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourcePreprocess {
    RendererOutput,
    FullVae,
}

fn source_preprocess(provider_id: &str) -> CoreResult<SourcePreprocess> {
    match provider_id {
        RENDERER_ID => Ok(SourcePreprocess::RendererOutput),
        FULL_ID => Ok(SourcePreprocess::FullVae),
        _ => Err(CoreError::Unsupported(format!(
            "unknown Bernini provider {provider_id}"
        ))),
    }
}

fn source_preprocess_axis(provider_id: &str) -> CoreResult<&'static str> {
    match source_preprocess(provider_id)? {
        SourcePreprocess::RendererOutput => Ok("source-preprocess-renderer-output-v1"),
        SourcePreprocess::FullVae => Ok("source-preprocess-full-vae624-v1"),
    }
}

fn source_vae_dims(provider_id: &str, width: u32, height: u32) -> CoreResult<(u32, u32)> {
    match source_preprocess(provider_id)? {
        SourcePreprocess::RendererOutput => Ok((width, height)),
        SourcePreprocess::FullVae => {
            let (width, height) = crate::vae_preprocess::resize_dims(
                i64::from(width),
                i64::from(height),
                crate::vae_preprocess::VAE_MAX_SIZE,
                crate::vae_preprocess::VAE_MIN_SIZE,
                crate::vae_preprocess::VAE_STRIDE,
            );
            Ok((width as u32, height as u32))
        }
    }
}

fn image_source_vae_dims(
    provider_id: &str,
    output_width: u32,
    output_height: u32,
    image_width: u32,
    image_height: u32,
) -> CoreResult<(u32, u32)> {
    match source_preprocess(provider_id)? {
        SourcePreprocess::RendererOutput => {
            source_vae_dims(provider_id, output_width, output_height)
        }
        SourcePreprocess::FullVae => source_vae_dims(provider_id, image_width, image_height),
    }
}

fn rv2v_receipt_matches_context(provider_id: &str, axis: &str, context: &MemoryRunContext) -> bool {
    let Ok((vae_width, vae_height)) =
        source_vae_dims(provider_id, context.geometry.width, context.geometry.height)
    else {
        return false;
    };
    let Ok(video_tokens) =
        packed_source_tokens(context.geometry.frames as usize, vae_width, vae_height)
    else {
        return false;
    };
    let latent_frames = (context.geometry.frames - 1)
        / u32::try_from(crate::VAE_TILING.temporal_scale).unwrap_or(u32::MAX)
        + 1;
    let video_marker = format!(
        "video-1:frames-{};native-{}x{};vae-{}x{}x{};tokens-{video_tokens}",
        context.geometry.frames,
        context.geometry.width,
        context.geometry.height,
        latent_frames,
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
    axis.contains(source_preprocess_axis(provider_id).unwrap_or_default())
        && axis.matches("video-1:").count() == 1
        && axis.contains(&video_marker)
        && tokens.len() == context.geometry.reference_count as usize
        && tokens
            .iter()
            .try_fold(0_u64, |sum, token| sum.checked_add(*token))
            == Some(declared_total)
}

fn packed_source_tokens(frames: usize, width: u32, height: u32) -> CoreResult<u64> {
    let temporal_scale = u64::try_from(crate::VAE_TILING.temporal_scale)
        .map_err(|_| CoreError::Unsupported("bernini temporal scale is invalid".into()))?;
    let spatial_patch_stride = u64::try_from(crate::VAE_TILING.spatial_scale)
        .map_err(|_| CoreError::Unsupported("bernini spatial scale is invalid".into()))?
        .checked_mul(2)
        .ok_or_else(|| CoreError::Unsupported("bernini source-token stride overflow".into()))?;
    let frames = u64::try_from(frames)
        .map_err(|_| CoreError::Unsupported("bernini source frame count overflow".into()))?;
    let latent_frames = if crate::VAE_TILING.causal_temporal {
        frames.saturating_sub(1) / temporal_scale + 1
    } else {
        frames.div_ceil(temporal_scale)
    };
    if u64::from(width) % spatial_patch_stride != 0 || u64::from(height) % spatial_patch_stride != 0
    {
        return Err(CoreError::Unsupported(format!(
            "bernini source geometry {width}x{height} does not land on the exact VAE/DiT token grid"
        )));
    }
    latent_frames
        .checked_mul(u64::from(width) / spatial_patch_stride)
        .and_then(|tokens| tokens.checked_mul(u64::from(height) / spatial_patch_stride))
        .ok_or_else(|| CoreError::Unsupported("bernini packed source-token count overflow".into()))
}

fn mv2v_clips(request: &GenerationRequest) -> CoreResult<Vec<mlx_gen::gen_core::VideoClipRef<'_>>> {
    if request.video_mode.as_deref() != Some("mv2v")
        || request.image_reference_count() != 0
        || !(2..=8).contains(&request.video_clips().len())
    {
        return Err(CoreError::Unsupported(
            "bernini mv2v requires exactly 2-8 VideoClips, no images, and video_mode=mv2v".into(),
        ));
    }
    let clips = request
        .conditioning
        .iter()
        .map(|conditioning| match conditioning {
            mlx_gen::gen_core::Conditioning::VideoClip {
                frames,
                frame_idx,
                strength,
            } => Ok(mlx_gen::gen_core::VideoClipRef {
                frames,
                frame_idx: *frame_idx,
                strength: *strength,
            }),
            _ => Err(CoreError::Unsupported(
                "bernini mv2v requires an ordered VideoClip-only carrier".into(),
            )),
        })
        .collect::<CoreResult<Vec<_>>>()?;
    let expected_pixels = u64::from(request.width)
        .checked_mul(u64::from(request.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|pixels| usize::try_from(pixels).ok())
        .ok_or_else(|| CoreError::Unsupported("bernini mv2v clip geometry overflow".into()))?;
    for (clip_index, clip) in clips.iter().enumerate() {
        if clip.frame_idx != 0
            || clip.strength.to_bits() != 1.0_f32.to_bits()
            || Some(clip.frames.len()) != request.frames.map(|frames| frames as usize)
        {
            return Err(CoreError::Unsupported(format!(
                "bernini mv2v clip {clip_index} must be normalized full-length at frame 0 with strength 1"
            )));
        }
        for (frame_index, frame) in clip.frames.iter().enumerate() {
            if frame.width != request.width
                || frame.height != request.height
                || frame.pixels.len() != expected_pixels
            {
                return Err(CoreError::Unsupported(format!(
                    "bernini mv2v clip {clip_index} frame {frame_index} is not exact output-sized RGB8"
                )));
            }
        }
    }
    Ok(clips)
}

fn mv2v_source_ids(count: usize) -> Vec<String> {
    // The renderer is trained through id 5; the shipped config interpolates 6-8 sources over 1..5.
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

/// ADS2V was trained with its three required sources (source clip, reference clip, first image).
/// Additional ordered images therefore interpolate within that trained 1..=3 source-id range.
fn ads2v_source_ids(count: usize) -> Vec<String> {
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

fn ads2v_sources(
    request: &GenerationRequest,
) -> CoreResult<(
    mlx_gen::gen_core::VideoClipRef<'_>,
    mlx_gen::gen_core::VideoClipRef<'_>,
    &[mlx_gen::gen_core::Image],
)> {
    let [mlx_gen::gen_core::Conditioning::VideoClip {
        frames: source_frames,
        frame_idx: source_index,
        strength: source_strength,
    }, mlx_gen::gen_core::Conditioning::VideoClip {
        frames: reference_frames,
        frame_idx: reference_index,
        strength: reference_strength,
    }, mlx_gen::gen_core::Conditioning::MultiReference { images }] =
        request.conditioning.as_slice()
    else {
        return Err(CoreError::Unsupported("bernini ads2v requires [source VideoClip, reference VideoClip, MultiReference 1-8 ordered images]".into()));
    };
    if request.video_mode.as_deref() != Some("ads2v") || !(1..=8).contains(&images.len()) {
        return Err(CoreError::Unsupported(
            "bernini ads2v requires video_mode=ads2v and 1-8 flattened ordered images".into(),
        ));
    }
    let expected_pixels = u64::from(request.width)
        .checked_mul(u64::from(request.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|pixels| usize::try_from(pixels).ok())
        .ok_or_else(|| CoreError::Unsupported("bernini ads2v clip geometry overflow".into()))?;
    let expected_frames = request.frames.map(|frames| frames as usize);
    let clips = [
        (
            "source",
            mlx_gen::gen_core::VideoClipRef {
                frames: source_frames,
                frame_idx: *source_index,
                strength: *source_strength,
            },
        ),
        (
            "reference",
            mlx_gen::gen_core::VideoClipRef {
                frames: reference_frames,
                frame_idx: *reference_index,
                strength: *reference_strength,
            },
        ),
    ];
    for (role, clip) in &clips {
        if clip.frame_idx != 0
            || clip.strength.to_bits() != 1.0_f32.to_bits()
            || Some(clip.frames.len()) != expected_frames
        {
            return Err(CoreError::Unsupported(format!("bernini ads2v {role} clip must be normalized full-length at frame 0 with strength 1")));
        }
        for (frame_index, frame) in clip.frames.iter().enumerate() {
            if frame.width != request.width
                || frame.height != request.height
                || frame.pixels.len() != expected_pixels
            {
                return Err(CoreError::Unsupported(format!(
                    "bernini ads2v {role} clip frame {frame_index} is not exact output-sized RGB8"
                )));
            }
        }
    }
    Ok((clips[0].1, clips[1].1, images.as_slice()))
}

/// Exact ADS2V source identity. The independent clip-role labels and flattened image entries are
/// intentionally included in both the curve-keyable surface and request-only byte seal.
pub fn ads2v_source_receipt(provider_id: &str, request: &GenerationRequest) -> CoreResult<String> {
    let (source, reference, images) = ads2v_sources(request)?;
    let preprocess_axis = source_preprocess_axis(provider_id)?;
    let (clip_vae_width, clip_vae_height) =
        source_vae_dims(provider_id, request.width, request.height)?;
    let mut seal = Sha256::new();
    seal.update(ADS2V_SEAL_DOMAIN.as_bytes());
    let mut total_tokens = 0_u64;
    let mut entries = Vec::with_capacity(images.len() + 2);
    for (role, clip) in [("source-video", source), ("reference-video", reference)] {
        seal.update(role.as_bytes());
        seal.update(clip.frame_idx.to_le_bytes());
        seal.update(clip.strength.to_bits().to_le_bytes());
        for (frame_index, frame) in clip.frames.iter().enumerate() {
            seal.update((frame_index as u32).to_le_bytes());
            seal.update(frame.width.to_le_bytes());
            seal.update(frame.height.to_le_bytes());
            seal.update(&frame.pixels);
        }
        let tokens = packed_source_tokens(clip.frames.len(), clip_vae_width, clip_vae_height)?;
        total_tokens = total_tokens.checked_add(tokens).ok_or_else(|| {
            CoreError::Unsupported("bernini ads2v packed source-token count overflow".into())
        })?;
        entries.push(format!(
            "{role}:frames-{};native-{}x{};vae-{}x{}x{};tokens-{tokens}",
            clip.frames.len(),
            request.width,
            request.height,
            (clip.frames.len() - 1)
                / usize::try_from(crate::VAE_TILING.temporal_scale).map_err(|_| {
                    CoreError::Unsupported("bernini temporal scale is invalid".into())
                })?
                + 1,
            clip_vae_width / 8,
            clip_vae_height / 8
        ));
    }
    for (index, image) in images.iter().enumerate() {
        let expected_pixels = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|pixels| usize::try_from(pixels).ok())
            .ok_or_else(|| {
                CoreError::Unsupported("bernini ads2v image geometry overflow".into())
            })?;
        if image.pixels.len() != expected_pixels {
            return Err(CoreError::Unsupported(format!(
                "bernini ads2v image {index} has malformed RGB8 bytes"
            )));
        }
        seal.update((index as u32).to_le_bytes());
        seal.update(image.width.to_le_bytes());
        seal.update(image.height.to_le_bytes());
        seal.update(&image.pixels);
        let (vit_h, vit_w) = crate::vit_preprocess::smart_resize(
            i64::from(image.height),
            i64::from(image.width),
            crate::vit_preprocess::FACTOR,
            3136,
            50176,
        );
        let (vae_width, vae_height) = image_source_vae_dims(
            provider_id,
            request.width,
            request.height,
            image.width,
            image.height,
        )?;
        let tokens = packed_source_tokens(1, vae_width, vae_height)?;
        total_tokens = total_tokens.checked_add(tokens).ok_or_else(|| {
            CoreError::Unsupported("bernini ads2v packed source-token count overflow".into())
        })?;
        entries.push(format!(
            "image-{}:native-{}x{};vit-{}x{};vae-{}x{};tokens-{tokens}",
            index + 1,
            image.width,
            image.height,
            vit_w,
            vit_h,
            vae_width,
            vae_height
        ));
    }
    let count = images.len() + 2;
    Ok(format!("{ADS2V_RECEIPT_DOMAIN}:backend-mlx:{preprocess_axis}:count-{count}:packed-source-tokens-{total_tokens}:source-ids-{}:{}+{ADS2V_SEAL_DOMAIN}-{:x}", ads2v_source_ids(count).join(","), entries.join("|"), seal.finalize()))
}

/// Bind every ordered source clip's normalized shape, packed VAE/DiT surface and source-id schedule.
/// The byte seal rejects post-admission content or ordering mutation while the shape axis remains curve-keyable.
pub fn mv2v_clip_receipt(provider_id: &str, request: &GenerationRequest) -> CoreResult<String> {
    let clips = mv2v_clips(request)?;
    let preprocess_axis = source_preprocess_axis(provider_id)?;
    let (vae_width, vae_height) = source_vae_dims(provider_id, request.width, request.height)?;
    let mut seal = Sha256::new();
    seal.update(MV2V_SEAL_DOMAIN.as_bytes());
    let mut total_tokens = 0_u64;
    let mut entries = Vec::with_capacity(clips.len());
    for (clip_index, clip) in clips.iter().enumerate() {
        seal.update((clip_index as u32).to_le_bytes());
        seal.update(clip.frame_idx.to_le_bytes());
        seal.update(clip.strength.to_bits().to_le_bytes());
        for (frame_index, frame) in clip.frames.iter().enumerate() {
            seal.update((frame_index as u32).to_le_bytes());
            seal.update(frame.width.to_le_bytes());
            seal.update(frame.height.to_le_bytes());
            seal.update(&frame.pixels);
        }
        let tokens = packed_source_tokens(clip.frames.len(), vae_width, vae_height)?;
        total_tokens = total_tokens.checked_add(tokens).ok_or_else(|| {
            CoreError::Unsupported("bernini mv2v combined source-token count overflow".into())
        })?;
        let latent_frames = (clip.frames.len() - 1)
            / usize::try_from(crate::VAE_TILING.temporal_scale)
                .map_err(|_| CoreError::Unsupported("bernini temporal scale is invalid".into()))?
            + 1;
        entries.push(format!(
            "video-{}:frames-{};native-{}x{};vae-{}x{}x{};tokens-{tokens}",
            clip_index + 1,
            clip.frames.len(),
            request.width,
            request.height,
            latent_frames,
            vae_width / 8,
            vae_height / 8,
        ));
    }
    Ok(format!(
        "{MV2V_RECEIPT_DOMAIN}:backend-mlx:{preprocess_axis}:count-{}:packed-source-tokens-{total_tokens}:source-ids-{}:{}+{MV2V_SEAL_DOMAIN}-{:x}",
        clips.len(), mv2v_source_ids(clips.len()).join(","), entries.join("|"), seal.finalize()
    ))
}

fn r2v_sources(
    request: &GenerationRequest,
) -> CoreResult<(
    Option<mlx_gen::gen_core::VideoClipRef<'_>>,
    &[mlx_gen::gen_core::Image],
)> {
    let (clip, images) = match request.conditioning.as_slice() {
        [mlx_gen::gen_core::Conditioning::MultiReference { images }] => (None, images.as_slice()),
        [
            mlx_gen::gen_core::Conditioning::VideoClip {
                frames,
                frame_idx,
                strength,
            },
            mlx_gen::gen_core::Conditioning::MultiReference { images },
        ] => (
            Some(mlx_gen::gen_core::VideoClipRef {
                frames,
                frame_idx: *frame_idx,
                strength: *strength,
            }),
            images.as_slice(),
        ),
        _ => {
            return Err(CoreError::Unsupported(
                "bernini reference memory requires one MultiReference, optionally preceded by exactly one VideoClip"
                    .to_owned(),
            ))
        }
    };
    if !(1..=8).contains(&images.len()) {
        return Err(CoreError::Unsupported(format!(
            "bernini r2v requires 1-8 flattened reference images, got {}",
            images.len()
        )));
    }
    if let Some(clip) = clip {
        if clip.frame_idx != 0
            || clip.strength.to_bits() != 1.0_f32.to_bits()
            || Some(clip.frames.len()) != request.frames.map(|frames| frames as usize)
        {
            return Err(CoreError::Unsupported(
                "bernini rv2v requires one normalized full-length VideoClip at frame 0 with strength 1"
                    .to_owned(),
            ));
        }
        for (index, frame) in clip.frames.iter().enumerate() {
            let expected_pixels = u64::from(request.width)
                .checked_mul(u64::from(request.height))
                .and_then(|pixels| pixels.checked_mul(3))
                .and_then(|pixels| usize::try_from(pixels).ok())
                .ok_or_else(|| {
                    CoreError::Unsupported("bernini rv2v clip geometry overflow".into())
                })?;
            if frame.width != request.width
                || frame.height != request.height
                || frame.pixels.len() != expected_pixels
            {
                return Err(CoreError::Unsupported(format!(
                    "bernini rv2v clip frame {index} is not exact output-sized RGB8"
                )));
            }
        }
    }
    Ok((clip, images))
}

/// Bind the content-independent memory shape separately from the request-only byte seal. The two
/// axes travel together to provider safety, but only the shape axis is eligible for fitted-curve
/// lookup. This is recomputed at configure so post-admission content mutation is rejected.
pub fn r2v_reference_receipt(provider_id: &str, request: &GenerationRequest) -> CoreResult<String> {
    let (clip, images) = r2v_sources(request)?;
    let preprocess_axis = source_preprocess_axis(provider_id)?;
    let has_video = clip.is_some();
    let mut request_seal = Sha256::new();
    request_seal.update(R2V_REFERENCE_SEAL_DOMAIN.as_bytes());
    let mut entries = Vec::with_capacity(images.len() + usize::from(clip.is_some()));
    let mut packed_tokens = 0_u64;
    if let Some(clip) = clip {
        request_seal.update(b"video-1");
        request_seal.update(clip.frame_idx.to_le_bytes());
        request_seal.update(clip.strength.to_bits().to_le_bytes());
        for (index, frame) in clip.frames.iter().enumerate() {
            request_seal.update((index as u32).to_le_bytes());
            request_seal.update(frame.width.to_le_bytes());
            request_seal.update(frame.height.to_le_bytes());
            request_seal.update(&frame.pixels);
        }
        let (vae_width, vae_height) = source_vae_dims(provider_id, request.width, request.height)?;
        let tokens = packed_source_tokens(clip.frames.len(), vae_width, vae_height)?;
        packed_tokens = packed_tokens.checked_add(tokens).ok_or_else(|| {
            CoreError::Unsupported("bernini combined source-token count overflow".into())
        })?;
        let latent_frames = (clip.frames.len() - 1)
            / usize::try_from(crate::VAE_TILING.temporal_scale)
                .map_err(|_| CoreError::Unsupported("bernini temporal scale is invalid".into()))?
            + 1;
        entries.push(format!(
            "video-1:frames-{};native-{}x{};vae-{}x{}x{};tokens-{tokens}",
            clip.frames.len(),
            request.width,
            request.height,
            latent_frames,
            vae_width / 8,
            vae_height / 8
        ));
    }
    for (index, image) in images.iter().enumerate() {
        let expected_pixels = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|pixels| usize::try_from(pixels).ok())
            .ok_or_else(|| CoreError::Unsupported("bernini r2v image geometry overflow".into()))?;
        if image.pixels.len() != expected_pixels {
            return Err(CoreError::Unsupported(format!(
                "bernini r2v reference {index} has {} RGB bytes, expected {expected_pixels}",
                image.pixels.len()
            )));
        }
        let (vit_h, vit_w) = crate::vit_preprocess::smart_resize(
            i64::from(image.height),
            i64::from(image.width),
            crate::vit_preprocess::FACTOR,
            3136,
            50176,
        );
        let (vae_w, vae_h) = image_source_vae_dims(
            provider_id,
            request.width,
            request.height,
            image.width,
            image.height,
        )?;
        request_seal.update(image.width.to_le_bytes());
        request_seal.update(image.height.to_le_bytes());
        request_seal.update(&image.pixels);
        if has_video {
            let tokens = packed_source_tokens(1, vae_w, vae_h)?;
            packed_tokens = packed_tokens.checked_add(tokens).ok_or_else(|| {
                CoreError::Unsupported("bernini combined source-token count overflow".into())
            })?;
            entries.push(format!(
                "{index}:native-{}x{};vit-{}x{};vae-{}x{};tokens-{tokens}",
                image.width, image.height, vit_w, vit_h, vae_w, vae_h
            ));
        } else {
            entries.push(format!(
                "{index}:native-{}x{};vit-{}x{};vae-{}x{}",
                image.width, image.height, vit_w, vit_h, vae_w, vae_h
            ));
        }
    }
    let packed_surface = has_video.then(|| format!("packed-source-tokens-{packed_tokens}:"));
    Ok(format!(
        "{R2V_REFERENCE_RECEIPT_DOMAIN}:backend-mlx:{preprocess_axis}:count-{}:{}{}+{R2V_REFERENCE_SEAL_DOMAIN}-{:x}",
        images.len(),
        packed_surface.as_deref().unwrap_or_default(),
        entries.join("|"),
        request_seal.finalize()
    ))
}

fn receipt_count(axis: &str) -> Option<u32> {
    axis.strip_prefix(R2V_REFERENCE_RECEIPT_DOMAIN)?
        .split_once(":count-")?
        .1
        .split_once(':')?
        .0
        .parse()
        .ok()
}

fn reference_receipt_from_overlay(overlay: Option<&str>) -> Option<String> {
    let axes = overlay?.split('+').collect::<Vec<_>>();
    let evidence = axes
        .iter()
        .copied()
        .filter(|axis| is_reference_receipt_axis(axis))
        .collect::<Vec<_>>();
    let seals = axes
        .iter()
        .copied()
        .filter(|axis| axis.starts_with(R2V_REFERENCE_SEAL_DOMAIN))
        .collect::<Vec<_>>();
    (evidence.len() == 1 && seals.len() == 1).then(|| format!("{}+{}", evidence[0], seals[0]))
}

fn source_receipt_from_overlay(overlay: Option<&str>) -> Option<String> {
    let axes = overlay?.split('+').collect::<Vec<_>>();
    let evidence = axes
        .iter()
        .copied()
        .filter(|axis| {
            is_reference_receipt_axis(axis)
                || is_mv2v_receipt_axis(axis)
                || is_ads2v_receipt_axis(axis)
        })
        .collect::<Vec<_>>();
    let seals = axes
        .iter()
        .copied()
        .filter(|axis| {
            axis.starts_with(R2V_REFERENCE_SEAL_DOMAIN)
                || axis.starts_with(MV2V_SEAL_DOMAIN)
                || axis.starts_with(ADS2V_SEAL_DOMAIN)
        })
        .collect::<Vec<_>>();
    (evidence.len() == 1 && seals.len() == 1).then(|| format!("{}+{}", evidence[0], seals[0]))
}

/// Rung 3 — the shared constrained score budget. Bernini does not invent its own.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// Rung 4 — the published window cadences, in blocks.
///
/// **Every value divides 40**, which is not a style choice: the trunk is indexed globally across both
/// experts (`0..40` high, `40..80` low) and the shared request scope requires a window start aligned
/// to the window size, so a cadence that does not divide one expert's depth would leave the low
/// expert's first window mis-aligned. `40` itself is excluded because it degenerates to fully
/// resident (`BlockPlan::is_bounded` is false there), which is rung 0 wearing rung 4's name.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 5, 8, 10, 20];

/// The default cadence when a request enables rung 4 without naming one.
///
/// 10 rather than 1. At 1 the trunk is re-walked `40 x 40 x 4 = 6400` times per render for the
/// tightest possible bound; 10 costs a tenth of that for a bound still an order of magnitude below
/// resident. A caller that wants 1 can still ask for it — the domain publishes it.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 10;

/// The rung-4 component scope this family implements.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// The component scopes this family implements — the DiT trunk only, and the reason is structural
/// rather than inherited (sc-15794 says a scope decision must not be copied from a sibling).
///
/// Bernini's conditioning networks are the largest in the catalog: a Qwen2.5-VL-7B planner backbone, a
/// NEO vision embedder, a clip-diff head and a UMT5-XXL text encoder. They are also **completely
/// released before either expert is loaded** — `generate_impl` drops each one and calls `clear_cache`
/// at the phase boundary, unconditionally, on both providers. A `TextEncoder`-scoped window would
/// therefore bound a phase that has already returned its bytes by the time the request reaches its
/// high-water mark, which is the expert phase.
///
/// That is an argument about *where the peak is*, and it is backed by measurement already on this
/// branch's parent: `sequential_residency_real_weights.rs` bounds the observed peak to the expert
/// phase. It is **not** a claim that the conditioning phase is small — it is a claim that it is not
/// concurrent with the peak. `TextEncoder` and `Both` are consequently declared unimplemented and
/// rejected with a typed error rather than silently narrowed to `Dit`, and a future story that wants
/// them must first show a route where the conditioning phase IS the peak.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[TRANSFORMER_WINDOW_COMPONENT];

/// Architecture axes shared by both registered Bernini routes (epic SC-22657, E2).
///
/// Bernini runs the Wan 2.2 T2V-14B trunk, whose dimensions this workspace pins as
/// [`WanModelConfig::wan22_t2v_14b`]; the loader overlays a snapshot's own `config.json` over that
/// preset, and [`check_loaded_expert_depth`] already refuses a snapshot whose expert depth
/// disagrees, so the preset is the config both routes are built from.
///
/// `transformer_blocks` is ONE expert's depth, not [`trunk_blocks`]: a denoise step traverses the
/// low-noise expert or the high-noise one, never both, so 80 would double the trunk an activation
/// estimate sees. The VAE scales are the provider's own assigned geometry ([`crate::VAE_TILING`]),
/// which is a **video** autoencoder — hence a real `vae_temporal_scale` here, unlike the image
/// families.
///
/// When `spec` names a materialized snapshot **that carries a `config.json`**, this re-runs
/// `WanModelConfig::from_model_dir` — the parse `crate::pipeline` and `crate::bernini` both run at
/// load — so the published trunk axes are the snapshot's rather than the preset's. The
/// `config.json` existence check is not redundant with `materialized_root`: `from_model_dir` falls
/// back to the **TI2V-5B** preset when the file is absent, which is a different model from the one
/// this provider runs, so a config-less snapshot must fall back here instead.
///
/// `transformer_blocks` stays one expert's depth on both paths — the snapshot's `num_layers` is
/// per-expert, exactly as [`expert_blocks`] reads it from the preset. [`check_loaded_expert_depth`]
/// deliberately keeps enforcing the *pinned* depth: that guard is about rung 4's window index
/// space, not about what geometry a contract describes.
fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let wan = mlx_gen::architecture_facts::materialized_root(spec)
        .filter(|root| root.join("config.json").is_file())
        .and_then(|root| WanModelConfig::from_model_dir(root).ok())
        .unwrap_or_else(WanModelConfig::wan22_t2v_14b);
    let (_, patch_h, patch_w) = wan.patch_size;
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(wan.num_heads),
        // The exactness-gated helper, NOT `axis(wan.head_dim())` (SC-22667): `head_dim()` is a
        // plain `dim / num_heads`, which rounds a non-uniform stack into a fabricated width and
        // panics on a `"num_heads": 0` snapshot key before `axis` can decline it.
        head_dim: mlx_gen::architecture_facts::head_dim(wan.dim, wan.num_heads),
        transformer_blocks: mlx_gen::architecture_facts::axis(wan.num_layers),
        // A single scalar can only describe a square patch; an anisotropic one has no honest value.
        patch_size: (patch_h == patch_w)
            .then(|| mlx_gen::architecture_facts::axis(patch_h))
            .flatten(),
        latent_channels: mlx_gen::architecture_facts::axis(wan.vae_z_dim),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::VAE_TILING.spatial_scale),
        vae_temporal_scale: mlx_gen::architecture_facts::axis(crate::VAE_TILING.temporal_scale),
        // The Wan trunk is bf16-native: every matmul runs bf16 over an f32 residual stream.
        activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
    }
}

/// One Wan expert's block depth, read from the model config rather than written as `40`.
fn expert_blocks() -> usize {
    WanModelConfig::wan22_t2v_14b().num_layers
}

/// The whole trunk's block depth — both experts, in the one global index space rung 4 windows over.
pub fn trunk_blocks() -> usize {
    2 * expert_blocks()
}

/// Refuse rung 4 on a loaded snapshot whose expert depth is not the declared one (sc-18609).
///
/// The crate-private `expert_blocks` reads the pinned `wan22_t2v_14b` config, so every published cadence in
/// [`TRANSFORMER_WINDOW_SIZES`] and the trunk depth handed to the shared request scope are derived
/// from 40. A snapshot's `config.json` can nonetheless set `num_layers` alongside `dual_model`
/// (`WanModelConfig::apply_overrides`), and both loaders accept whatever it says. Before this guard
/// a 30-block dual snapshot would have been admitted against a divisors-of-40 domain and a declared
/// 80-block index space: cadence 20 would put the low expert's first window at global block 60 while
/// the real boundary is 30, i.e. a window spanning both experts. That is the declaration/reachability
/// gap the epic forbids, so the rung refuses with a typed error instead of mis-aligning mid-denoise.
///
/// This gates only the optimized route. The load itself, and rungs 0-3, are unaffected — none of
/// them index the trunk.
pub fn check_loaded_expert_depth(provider_id: &str, loaded_expert_blocks: usize) -> CoreResult<()> {
    if loaded_expert_blocks == expert_blocks() {
        return Ok(());
    }
    Err(CoreError::Unsupported(format!(
        "{provider_id}: bounded transformer residency is declared over a {}-block expert \
         ({}-block trunk) and publishes the cadences {TRANSFORMER_WINDOW_SIZES:?}; this snapshot \
         loads {loaded_expert_blocks} blocks per expert, so no published window is aligned to its \
         expert boundary",
        expert_blocks(),
        trunk_blocks()
    )))
}

/// Whether THIS load can execute rung 4.
///
/// Two independent facts decide it, and only one of them is a [`LoadShape`]:
///
/// 1. The window rebuilds blocks from the snapshot, so it needs a **re-openable source**. Both
///    providers reject anything but a `WeightsSource::Dir`, so a load that got this far is
///    re-openable by construction.
/// 2. The load must not have bulk-committed the stack. [`LoadShape::DeferredMaterialization`] is the
///    shared contract's declared prerequisite for rung 4, and it is checked per LOAD, not per
///    provider: a window over an already-materialized trunk bounds nothing, it *adds* a copy on top.
///
/// Adapters would be a third fact, but the descriptor advertises `supports_lora: false`, so an
/// adapted load cannot reach here. The refusal still lives in
/// [`WanBlockStream::new`](mlx_gen_wan::WanBlockStream) where the mechanism is, so a future provider
/// that flips the capability bit cannot silently ship un-adapted streamed blocks.
pub fn structurally_streamable(spec: &LoadSpec) -> bool {
    matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.adapters.is_empty()
}

fn known_provider(provider_id: &str) -> CoreResult<()> {
    PROVIDER_IDS.contains(&provider_id).then_some(()).ok_or_else(|| {
        CoreError::Msg(format!(
            "bernini memory strategy: unknown provider `{provider_id}`; expected one of {PROVIDER_IDS:?}"
        ))
    })
}

/// Artifact-tier label of a Bernini load: `bf16` dense, `q4`/`q8` prepacked. `None` for a packed
/// width this family does not ship.
pub fn calibration_tier_label(quant: Option<mlx_gen::Quant>) -> Option<&'static str> {
    match quant {
        None => Some("bf16"),
        Some(mlx_gen::Quant::Q4) => Some("q4"),
        Some(mlx_gen::Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// The route token one provider id contributes to its identity: the full planner+renderer pipeline
/// the `bernini` / `bernini_image` catalog entries resolve to, and the renderer-only sibling.
pub fn calibration_route(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        FULL_ID => Some("image"),
        RENDERER_ID => Some("renderer"),
        _ => None,
    }
}

/// The tier [`MEMORY_CALIBRATION_FINGERPRINT`] names.
pub const CALIBRATED_TIER: &str = "q4";

/// The production calibration identity for one `(provider, artifact-proven tier)` cell.
///
/// ## sc-22737 (epic sc-22723 E1/E4): every clean load publishes an identity
///
/// This returned `None` for **every** load until sc-22737, on the reasoning that a key without a
/// measurement behind it is a false claim. That conflated two different things. A fingerprint is a
/// *key for* evidence, not an assertion *that* evidence exists — the anchor store is what says
/// whether a cell has been measured — and while this published nothing, no anchor could bind to a
/// Bernini load at all: the measurement machinery had no name to record against, so the family was
/// permanently unmeasurable rather than merely unmeasured. `MemoryEvidence::optimized_eligibility`
/// still refuses every optimized fit for a cell with no record, so nothing here can claim a saving
/// this repository cannot show.
///
/// The key is **(provider, tier)**, and the tier is the one the ARTIFACT proves:
/// [`resolved_numeric_tier`] reads the renderer/planner manifests and the packed safetensors
/// headers, and refuses a prepacked root that also carries a load-time `quantize`, a manifest that
/// disagrees with its headers, and a dense root whose headers are packed. Every one of those is
/// `Err`, which becomes `None` here — **withheld on disagreement, never a failed load**. Keying on
/// `spec.quantize` instead would let a dense snapshot asked for as q4 publish the q4 identity.
///
/// The measured `(bernini, q4)` cell keeps [`MEMORY_CALIBRATION_FINGERPRINT`] unchanged; the format
/// below is the one that mints it, so the retained string is produced rather than special-cased
/// (`the_measured_cell_keeps_its_retained_string` asserts the two agree).
pub fn production_calibration_fingerprint(provider_id: &str, spec: &LoadSpec) -> Option<String> {
    let route = calibration_route(provider_id)?;
    if spec.precision != mlx_gen::Precision::Bf16 {
        return None;
    }
    let tier = calibration_tier_label(resolved_numeric_tier(spec).ok()?.quant)?;
    Some(format!("bernini-{route}-{tier}-mlx-dual-expert-ladder-v1"))
}

/// The public contract for a loaded provider.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let facts = crate::pipeline::component_footprint(spec)?;
    build_contract(
        provider_id,
        spec,
        conditioning_bytes(provider_id, spec, facts.text_encoder),
        facts.dit,
        facts.vae,
        production_calibration_fingerprint(provider_id, spec)
            .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape)),
    )
}

/// Declaration-equivalent contract used by weights-free registry conformance. Structure, parameter
/// domains and prerequisites are identical; only the measured asset facts are absent, and the
/// calibration identity is the static declaration key rather than a production one.
///
/// `pub` (sc-18609) for the same reason FLUX.1 and PuLID export theirs: this is the only contract in
/// the family that carries a calibration identity, and `standard_memory_behavior_context` refuses to
/// build a run context without one. An out-of-crate evidence runner therefore cannot construct the
/// declared route from `memory_strategy_contract` alone until a production cell is minted.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    build_contract(
        provider_id,
        spec,
        0,
        0,
        0,
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION,
            spec.load_shape,
        )),
    )
}

/// The conditioning bytes this provider actually loads.
///
/// [`component_footprint`](crate::pipeline::component_footprint) counts only `t5_encoder` because the
/// worker's fit gate consumes that split. The **full** pipeline additionally loads the Qwen2.5-VL-7B
/// planner backbone, the MLP connector, the ViT decoder and the MAR mask tokens before the experts —
/// on the shipped q4 tier that is another ~9 GB, and charging zero for it would under-price the
/// conditioning phase by more than the whole text encoder.
fn conditioning_bytes(provider_id: &str, spec: &LoadSpec, text_encoder_bytes: u64) -> u64 {
    if provider_id != FULL_ID {
        return text_encoder_bytes;
    }
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return text_encoder_bytes;
    };
    let planner = mlx_gen::gen_core::PerComponentBytes::from_root_subdirs(
        root,
        &[
            "qwen2_5_vl.safetensors",
            "connector.safetensors",
            "vit_decoder.safetensors",
            "mask_tokens.safetensors",
        ],
        &[],
        &[],
    );
    text_encoder_bytes.saturating_add(planner.text_encoder)
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    known_provider(provider_id)?;
    let streamable = structurally_streamable(spec);
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            // Unified memory: the wired-residency budget is what the staged phases release, weights
            // are mmap-backed and lazy per tensor, and MLX's lazy graph needs an explicit `eval`
            // before a phase drop (or a window drop) frees anything.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts(spec);
    contract.calibration = calibration;
    // Bernini's shipped default is BOTH experts co-resident, which is exactly what rung 0 means here,
    // so the resident selection preserves the load defaults rather than writing an all-disabled
    // block. The cross-phase staging that runs unconditionally is not a lever and is not claimed as
    // one: rung 1's lever is the expert sequencing.
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;

    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    contract.formula = MemoryFormulaKind::PhaseEnvelope { phases, variables };

    // No auxiliary resident network: no control branch, no IP-adapter, no identity encoder, and
    // `supports_lora: false`. `overlay_bytes` is therefore 0 as a positive statement, not an omission.
    contract.asset_facts.overlay_bytes = 0;
    contract.asset_facts.conditioning_bytes = conditioning_bytes;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.decoder_bytes = decoder_bytes;
    contract.asset_facts.base_bytes = conditioning_bytes
        .saturating_add(transformer_bytes)
        .saturating_add(decoder_bytes);

    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };

    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency => MemoryStrategySupport::Missing,
            MemoryStrategy::BoundedDecode => {
                capability.parameters.decode_tile_edges = DECODE_TILE_EDGES.to_vec();
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = TRANSFORMER_WINDOW_SIZES.to_vec();
                capability.parameters.transformer_window_components =
                    TRANSFORMER_WINDOW_COMPONENTS.to_vec();
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }

    // Deliberately NO rung-4 -> rung-1 `additional_prerequisites` edge. Chroma and Anima declare one
    // because their window bounds the DiT's *denoise-phase* residency only, so without the phase
    // release the conditioner and VAE sit alongside it and the request peak does not move. Bernini's
    // conditioning phase is already released unconditionally before either expert loads, so the
    // denoise phase IS the peak with or without rung 1 — and rung 4's deferred load holds zero blocks
    // of BOTH experts, which is strictly more than rung 1's sequencing achieves. Adding the edge here
    // would force a caller to pay rung 1's warm-cache eviction for a saving rung 4 already has, and
    // `MemoryStrategy::engages` is explicit that rung 4 does not universally engage rung 1.
    Ok(contract)
}

/// Reject a decode geometry outside the published domain.
fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> CoreResult<()> {
    let edge = edge.ok_or_else(|| {
        CoreError::Unsupported("bernini: bounded decode requires a tile edge".to_owned())
    })?;
    let overlap = overlap.ok_or_else(|| {
        CoreError::Unsupported("bernini: bounded decode requires a tile overlap".to_owned())
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) {
        return Err(CoreError::Unsupported(format!(
            "bernini: decode tile edge {edge} is outside the published domain {DECODE_TILE_EDGES:?}"
        )));
    }
    if overlap != DECODE_OVERLAP {
        return Err(CoreError::Unsupported(format!(
            "bernini: decode overlap {overlap} is not the published {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

fn validate_geometry(width: u32, height: u32) -> CoreResult<()> {
    if ADVERTISED_GEOMETRIES.contains(&(width, height)) {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "bernini: memory evidence requires one of the advertised geometries {ADVERTISED_GEOMETRIES:?}, got {width}x{height}"
        )))
    }
}

/// Reject a window cadence outside the published domain.
fn validate_window(size: u32) -> CoreResult<()> {
    if !TRANSFORMER_WINDOW_SIZES.contains(&size) {
        return Err(CoreError::Unsupported(format!(
            "bernini: transformer window {size} is outside the published domain \
             {TRANSFORMER_WINDOW_SIZES:?}; every published cadence divides one expert's \
             {}-block depth so the low expert's first window stays aligned in the {}-block trunk",
            expert_blocks(),
            trunk_blocks()
        )));
    }
    Ok(())
}

/// Reject an attention chunk outside the published domain.
fn validate_attention(size: u32) -> CoreResult<()> {
    if size != ATTENTION_CHUNK_SIZE {
        return Err(CoreError::Unsupported(format!(
            "bernini: attention chunk size {size} is not the published {ATTENTION_CHUNK_SIZE}"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let mode = context.mode.as_key();
        let axes: Vec<_> = context
            .overlay
            .as_deref()
            .unwrap_or_default()
            .split('+')
            .filter(|axis| !axis.is_empty())
            .collect();
        let expected_provider_mode = match mode {
            "video_to_video" if context.geometry.reference_count == 1 => "provider_video_mode:v2v",
            "reference_to_video" if (1..=8).contains(&context.geometry.reference_count) => {
                "provider_video_mode:r2v"
            }
            "reference_video_to_video" if (2..=9).contains(&context.geometry.reference_count) => {
                "provider_video_mode:rv2v"
            }
            "multi_video_to_video" if (2..=8).contains(&context.geometry.reference_count) => {
                "provider_video_mode:mv2v"
            }
            "ads2v" if (3..=10).contains(&context.geometry.reference_count) => {
                "provider_video_mode:ads2v"
            }
            _ => {
                return Err(CoreError::Unsupported(format!(
                    "{}: Bernini memory evidence covers exact video_to_video/one-clip, \
                     reference_to_video/1-8-image, reference_video_to_video/one-clip-plus-1-8-image, \
                     multi_video_to_video/2-8-clip, or \
                     ads2v/source-plus-reference-clip-plus-1-8-image routes",
                    contract.provider_id
                )))
            }
        };
        if !context.has_reference || !axes.contains(&expected_provider_mode) {
            return Err(CoreError::Unsupported(format!(
                "{}: Bernini memory evidence requires the exact provider task {expected_provider_mode}",
                contract.provider_id
            )));
        }
        let reference_axis = axes
            .iter()
            .copied()
            .find(|axis| is_reference_receipt_axis(axis));
        let seal_axis = axes
            .iter()
            .copied()
            .find(|axis| axis.starts_with(R2V_REFERENCE_SEAL_DOMAIN));
        let exact_receipt = reference_receipt_from_overlay(context.overlay.as_deref());
        let mv2v_axis = axes.iter().copied().find(|axis| is_mv2v_receipt_axis(axis));
        let mv2v_seal = axes
            .iter()
            .copied()
            .find(|axis| axis.starts_with(MV2V_SEAL_DOMAIN));
        let ads2v_axis = axes
            .iter()
            .copied()
            .find(|axis| is_ads2v_receipt_axis(axis));
        let ads2v_seal = axes
            .iter()
            .copied()
            .find(|axis| axis.starts_with(ADS2V_SEAL_DOMAIN));
        let reference_axis_count = axes
            .iter()
            .filter(|axis| is_reference_receipt_axis(axis))
            .count();
        let seal_axis_count = axes
            .iter()
            .filter(|axis| axis.starts_with(R2V_REFERENCE_SEAL_DOMAIN))
            .count();
        let mv2v_axis_count = axes
            .iter()
            .filter(|axis| is_mv2v_receipt_axis(axis))
            .count();
        let mv2v_seal_count = axes
            .iter()
            .filter(|axis| axis.starts_with(MV2V_SEAL_DOMAIN))
            .count();
        let ads2v_axis_count = axes
            .iter()
            .filter(|axis| is_ads2v_receipt_axis(axis))
            .count();
        let ads2v_seal_count = axes
            .iter()
            .filter(|axis| axis.starts_with(ADS2V_SEAL_DOMAIN))
            .count();
        let receipt_has_video = reference_axis.is_some_and(reference_receipt_has_video);
        if mode == "reference_to_video"
            && (reference_axis.and_then(receipt_count) != Some(context.geometry.reference_count)
                || receipt_has_video
                || seal_axis.is_none()
                || exact_receipt.is_none())
        {
            return Err(CoreError::Unsupported(format!(
                "{}: Bernini r2v memory evidence requires an ordered MLX reference receipt for all {} images",
                contract.provider_id, context.geometry.reference_count
            )));
        }
        if mode == "reference_video_to_video"
            && (reference_axis.and_then(receipt_count)
                != Some(context.geometry.reference_count - 1)
                || !receipt_has_video
                || !reference_axis.is_some_and(|axis| {
                    rv2v_receipt_matches_context(&contract.provider_id, axis, context)
                })
                || seal_axis.is_none()
                || exact_receipt.is_none())
        {
            return Err(CoreError::Unsupported(format!(
                "{}: Bernini rv2v memory evidence requires one normalized clip plus an ordered MLX receipt for all {} images",
                contract.provider_id,
                context.geometry.reference_count - 1
            )));
        }
        if mode == "video_to_video" && (reference_axis.is_some() || seal_axis.is_some()) {
            return Err(CoreError::Unsupported(
                "bernini V2V cannot carry an R2V image receipt".to_owned(),
            ));
        }
        if mode == "multi_video_to_video"
            && (source_receipt_from_overlay(context.overlay.as_deref()).is_none()
                || mv2v_axis.is_none()
                || mv2v_seal.is_none()
                || mv2v_axis_count != 1
                || mv2v_seal_count != 1
                || reference_axis_count != 0
                || seal_axis_count != 0
                || !mv2v_axis.is_some_and(|axis| {
                    axis.contains(&format!(":count-{}:", context.geometry.reference_count))
                        && axis.matches("video-").count()
                            == context.geometry.reference_count as usize
                        && axis.contains(&format!(
                            "source-ids-{}",
                            mv2v_source_ids(context.geometry.reference_count as usize).join(",")
                        ))
                }))
        {
            return Err(CoreError::Unsupported(format!(
                "{}: Bernini mv2v memory evidence requires ordered normalized receipts for all {} source clips",
                contract.provider_id, context.geometry.reference_count
            )));
        }
        if mode == "ads2v"
            && (source_receipt_from_overlay(context.overlay.as_deref()).is_none()
                || ads2v_axis.is_none()
                || ads2v_seal.is_none()
                || ads2v_axis_count != 1
                || ads2v_seal_count != 1
                || reference_axis_count != 0
                || seal_axis_count != 0
                || mv2v_axis_count != 0
                || mv2v_seal_count != 0
                || !ads2v_axis.is_some_and(|axis| {
                    axis.contains(&format!(":count-{}:", context.geometry.reference_count))
                        && axis.matches("source-video:").count() == 1
                        && axis.matches("reference-video:").count() == 1
                        && axis.matches("image-").count()
                            == context.geometry.reference_count.saturating_sub(2) as usize
                        && axis.contains(&format!(
                            "source-ids-{}",
                            ads2v_source_ids(context.geometry.reference_count as usize).join(",")
                        ))
                }))
        {
            return Err(CoreError::Unsupported(format!(
                "{}: Bernini ads2v memory evidence requires ordered source/reference clips and 1-8 ordered images",
                contract.provider_id
            )));
        }
        if axes.iter().any(|axis| {
            *axis != expected_provider_mode
                && !is_reference_receipt_axis(axis)
                && !axis.starts_with(R2V_REFERENCE_SEAL_DOMAIN)
                && !is_mv2v_receipt_axis(axis)
                && !axis.starts_with(MV2V_SEAL_DOMAIN)
                && !is_ads2v_receipt_axis(axis)
                && !axis.starts_with(ADS2V_SEAL_DOMAIN)
        }) {
            return Err(CoreError::Unsupported(
                "bernini MLX memory evidence contains an unknown or crossed overlay axis"
                    .to_owned(),
            ));
        }
        validate_geometry(context.geometry.width, context.geometry.height)?;
        if !matches!(context.geometry.frames, 45 | 61 | 77) {
            return Err(CoreError::Unsupported(format!(
                "{}: Bernini V2V memory evidence covers exactly 45, 61, or 77 frames, got {}",
                contract.provider_id, context.geometry.frames
            )));
        }
        // sc-15839 review defect, Resident+PiD admission. Bernini decodes through the Wan z16 VAE and
        // the crate carries no `mlx-gen-pid` dependency at all, so a PiD selection would silently
        // execute the ordinary native decode — a different strategy than the selector chose. Reject
        // at admission rather than degrade.
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: bernini decodes through the Wan z16 AutoencoderKLWan and has no PiD student; \
                 the overlay is not implementable on this provider",
                contract.provider_id
            )));
        }
        // sc-18609. `MemoryRunContext::has_phases` describes the request's multi-phase denoise
        // (`GenerationRequest::phases`, epic 13879), which only the Krea MLX family reads — NOT this
        // contract's `MemoryPhase` lifecycle, which is a different axis and is always present here.
        // Nothing in this crate reads `req.phases`, so a phased admission would record evidence for a
        // trajectory Bernini cannot run. Reject it rather than silently render the single-phase one.
        if context.has_phases {
            return Err(CoreError::Unsupported(format!(
                "{}: bernini runs a single-phase denoise; the multi-phase request trajectory is not \
                 implemented on this provider",
                contract.provider_id
            )));
        }
        // sc-15839 review defect, unconstrained batch geometry. `max_count: 1` on both descriptors,
        // so a batched admission would record evidence for a route `validate` rejects anyway.
        if context.geometry.batch != 1 {
            return Err(CoreError::Unsupported(format!(
                "{}: bernini renders one image per request (max_count = 1); got batch {}",
                contract.provider_id, context.geometry.batch
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedAttention) {
            if let Some(size) = context.selection.parameters.attention_chunk_size {
                validate_attention(size)?;
            }
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) {
            if !contract.lifecycle.transformer_window_materialization {
                return Err(CoreError::Unsupported(format!(
                    "{}: bounded transformer residency requires a DeferredMaterialization load",
                    contract.provider_id
                )));
            }
            // The scope AND the cadence are checked here as well as by the shared parameter
            // validator, so a request that reached the provider by another path still cannot ask for
            // a scope this family does not implement or a cadence that would mis-align the low
            // expert's first window.
            let component = context.selection.parameters.window_component();
            if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
                return Err(CoreError::Unsupported(format!(
                    "{}: transformer window component {component:?} is not implemented; bernini \
                     releases every conditioning network before either expert loads, so a window \
                     over them cannot move the request peak",
                    contract.provider_id
                )));
            }
            if let Some(size) = context.selection.parameters.transformer_window_size {
                validate_window(size)?;
            }
        }
        Ok(())
    };
    let loaded_tier = if contract.provider_id != FULL_ID
        || contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == STATIC_CALIBRATION)
    {
        fixture_tier(spec)
    } else {
        match resolved_numeric_tier(spec) {
            Ok(tier) => tier,
            Err(error) => {
                return MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                }
            }
        }
    };
    standard_memory_strategy_safety_check(contract, context, Some(loaded_tier), Some(&route_gate))
}

fn fixture_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

fn quant_manifest(path: &std::path::Path) -> CoreResult<Option<(i32, usize)>> {
    let bytes = std::fs::read(path).map_err(|error| {
        CoreError::Unsupported(format!(
            "bernini numeric tier cannot read {}: {error}",
            path.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        CoreError::Unsupported(format!(
            "bernini numeric tier cannot parse {}: {error}",
            path.display()
        ))
    })?;
    let Some(quant) = json.get("quantization") else {
        return Ok(None);
    };
    if quant.is_null() {
        return Ok(None);
    }
    let bits = quant.get("bits").and_then(serde_json::Value::as_i64);
    let group = quant.get("group_size").and_then(serde_json::Value::as_u64);
    let (Some(bits @ (4 | 8)), Some(64)) = (bits, group) else {
        return Err(CoreError::Unsupported(format!(
            "bernini numeric tier requires an exact q4/q8 group-64 quantization manifest in {}",
            path.display()
        )));
    };
    Ok(Some((bits as i32, 64)))
}

fn packed_header_bits(path: &std::path::Path, group_size: usize) -> CoreResult<Option<i32>> {
    use mlx_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype};

    let headers = safetensors_path_tensor_headers(path)?;
    let by_name = headers
        .iter()
        .map(|header| (header.name.as_str(), header))
        .collect::<BTreeMap<_, _>>();
    let packed_bases = headers
        .iter()
        .filter_map(|header| header.name.strip_suffix(".scales"))
        .collect::<Vec<_>>();
    if packed_bases.is_empty() {
        if headers
            .iter()
            .any(|header| header.name.ends_with(".weight") && header.dtype == Dtype::U32)
        {
            return Err(CoreError::Unsupported(format!(
                "bernini numeric tier found U32 weights without packed scale headers in {}",
                path.display()
            )));
        }
        return Ok(None);
    }
    for header in headers
        .iter()
        .filter(|header| header.name.ends_with(".weight") && header.dtype == Dtype::U32)
    {
        let base = header
            .name
            .strip_suffix(".weight")
            .expect("filtered weight suffix");
        if !by_name.contains_key(format!("{base}.scales").as_str())
            || !by_name.contains_key(format!("{base}.biases").as_str())
        {
            return Err(CoreError::Unsupported(format!(
                "bernini numeric tier found unpaired packed weight {} in {}",
                header.name,
                path.display()
            )));
        }
    }

    let mut exact_bits = None;
    for base in packed_bases {
        let weight_name = format!("{base}.weight");
        let scale_name = format!("{base}.scales");
        let bias_name = format!("{base}.biases");
        let weight = by_name.get(weight_name.as_str()).ok_or_else(|| {
            CoreError::Unsupported(format!("{scale_name} has no packed {weight_name}"))
        })?;
        let scales = by_name
            .get(scale_name.as_str())
            .expect("base came from scales");
        let biases = by_name.get(bias_name.as_str()).ok_or_else(|| {
            CoreError::Unsupported(format!("{scale_name} has no packed {bias_name}"))
        })?;
        let ([weight_out, words], [scale_out, groups], [bias_out, bias_groups]) = (
            weight.shape.as_slice(),
            scales.shape.as_slice(),
            biases.shape.as_slice(),
        ) else {
            return Err(CoreError::Unsupported(format!(
                "bernini packed header {base} must be rank two"
            )));
        };
        if weight.dtype != Dtype::U32
            || scales.dtype != Dtype::BF16
            || biases.dtype != Dtype::BF16
            || weight_out != scale_out
            || scale_out != bias_out
            || groups != bias_groups
            || *groups == 0
        {
            return Err(CoreError::Unsupported(format!(
                "bernini packed header {base} has crossed dtype or shape companions"
            )));
        }
        let input = groups.checked_mul(group_size).ok_or_else(|| {
            CoreError::Unsupported(format!("bernini packed header {base} input overflow"))
        })?;
        let numerator = words.checked_mul(32).ok_or_else(|| {
            CoreError::Unsupported(format!("bernini packed header {base} bit-width overflow"))
        })?;
        if numerator % input != 0 {
            return Err(CoreError::Unsupported(format!(
                "bernini packed header {base} does not encode an integral bit width"
            )));
        }
        let bits = i32::try_from(numerator / input).map_err(|_| {
            CoreError::Unsupported(format!("bernini packed header {base} bit width overflow"))
        })?;
        if !matches!(bits, 4 | 8) || exact_bits.is_some_and(|current| current != bits) {
            return Err(CoreError::Unsupported(format!(
                "bernini packed header {} mixes or declares unsupported {bits}-bit weights",
                path.display()
            )));
        }
        exact_bits = Some(bits);
    }
    Ok(exact_bits)
}

/// Resolve the numeric tier actually carried by the immutable Bernini snapshot. Prepacked tiers
/// require matching renderer/planner manifests and matching packed safetensors headers; legacy
/// dense snapshots retain the explicit load-time quantization behavior.
pub fn resolved_numeric_tier(spec: &LoadSpec) -> CoreResult<MemoryNumericTier> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Unsupported(
            "bernini numeric tier requires a checkpoint directory".to_owned(),
        ));
    };
    let renderer_manifest = quant_manifest(&root.join("config.json"))?;
    let planner_manifest = quant_manifest(&root.join("qwen2_5_vl_config.json"))?;
    if renderer_manifest != planner_manifest {
        return Err(CoreError::Unsupported(
            "bernini renderer and planner quantization manifests disagree".to_owned(),
        ));
    }
    let files = [
        root.join("high_noise_model.safetensors"),
        root.join("low_noise_model.safetensors"),
        root.join("qwen2_5_vl.safetensors"),
    ];
    let header_bits = files
        .iter()
        .map(|path| packed_header_bits(path, 64))
        .collect::<CoreResult<Vec<_>>>()?;
    let resolved_quant = match renderer_manifest {
        Some((bits, 64)) => {
            if spec.quantize.is_some() {
                return Err(CoreError::Unsupported(
                    "bernini prepacked tiers must not request a second load-time quantization"
                        .to_owned(),
                ));
            }
            if header_bits.iter().any(|actual| *actual != Some(bits)) {
                return Err(CoreError::Unsupported(format!(
                    "bernini q{bits} manifest disagrees with packed renderer/planner headers"
                )));
            }
            Some(match bits {
                4 => mlx_gen::Quant::Q4,
                8 => mlx_gen::Quant::Q8,
                _ => unreachable!("manifest validator accepts q4/q8 only"),
            })
        }
        None => {
            if header_bits.iter().any(Option::is_some) {
                return Err(CoreError::Unsupported(
                    "bernini dense manifest disagrees with packed renderer/planner headers"
                        .to_owned(),
                ));
            }
            spec.quantize
        }
        Some(_) => unreachable!("manifest validator accepts group 64 only"),
    };
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant: resolved_quant,
        component_precision_floors: &[],
    })
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        fixture_tier(spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("video_to_video".to_owned()),
            reference_count: 1,
            use_pid: false,
            // Single-phase, deliberately (sc-18609). See the `has_phases` gate in `safety_check`:
            // this axis is the request's multi-phase denoise trajectory, which Bernini does not
            // read, not the contract's `MemoryPhase` lifecycle, which it always runs.
            has_phases: false,
            overlay: Some("provider_video_mode:v2v".to_owned()),
        },
    )?;
    context.geometry.width = 848;
    context.geometry.height = 480;
    context.geometry.frames = 45;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free bernini memory behavior".to_owned();
    fixture.request.fps = Some(16);
    fixture.request.video_mode = Some("v2v".to_owned());
    fixture.request.conditioning.clear();
    fixture
        .request
        .conditioning
        .push(mlx_gen::gen_core::Conditioning::VideoClip {
            frames: vec![mlx_gen::gen_core::Image {
                width: 2,
                height: 2,
                pixels: vec![0; 12],
            }],
            frame_idx: 0,
            strength: 1.0,
        });
    fixture.request.phases = None;
    let images = vec![
        mlx_gen::gen_core::Image {
            width: 40,
            height: 24,
            pixels: vec![1; 40 * 24 * 3],
        },
        mlx_gen::gen_core::Image {
            width: 24,
            height: 40,
            pixels: vec![2; 24 * 40 * 3],
        },
    ];
    let mut r2v_request = fixture.request.clone();
    r2v_request.prompt = "weights-free bernini r2v memory behavior".to_owned();
    r2v_request.video_mode = Some("r2v".to_owned());
    r2v_request.conditioning = vec![mlx_gen::gen_core::Conditioning::MultiReference { images }];
    let reference_axis = r2v_reference_receipt(&contract.provider_id, &r2v_request)?;
    let mut r2v_context = fixture.context.clone();
    r2v_context.mode = MemoryMode::Other("reference_to_video".to_owned());
    r2v_context.geometry.reference_count = 2;
    r2v_context.overlay = Some(format!("provider_video_mode:r2v+{reference_axis}"));
    let mut rv2v_request = r2v_request.clone();
    rv2v_request.prompt = "weights-free bernini rv2v memory behavior".to_owned();
    rv2v_request.video_mode = Some("rv2v".to_owned());
    rv2v_request.width = 848;
    rv2v_request.height = 480;
    rv2v_request.frames = Some(45);
    let clip_frame = mlx_gen::gen_core::Image {
        width: rv2v_request.width,
        height: rv2v_request.height,
        pixels: vec![3; rv2v_request.width as usize * rv2v_request.height as usize * 3],
    };
    let mut conditioning = std::mem::take(&mut rv2v_request.conditioning);
    let images = match conditioning.pop() {
        Some(mlx_gen::gen_core::Conditioning::MultiReference { images })
            if conditioning.is_empty() =>
        {
            images
        }
        _ => unreachable!("R2V fixture has one MultiReference"),
    };
    rv2v_request.conditioning = vec![
        mlx_gen::gen_core::Conditioning::VideoClip {
            frames: vec![clip_frame; 45],
            frame_idx: 0,
            strength: 1.0,
        },
        mlx_gen::gen_core::Conditioning::MultiReference { images },
    ];
    let rv2v_reference_axis = r2v_reference_receipt(&contract.provider_id, &rv2v_request)?;
    let mut rv2v_context = fixture.context.clone();
    rv2v_context.mode = MemoryMode::Other("reference_video_to_video".to_owned());
    rv2v_context.geometry.reference_count = 3;
    rv2v_context.overlay = Some(format!("provider_video_mode:rv2v+{rv2v_reference_axis}"));
    let mut mv2v_request = fixture.request.clone();
    mv2v_request.prompt = "weights-free bernini mv2v memory behavior".to_owned();
    mv2v_request.video_mode = Some("mv2v".to_owned());
    mv2v_request.width = 848;
    mv2v_request.height = 480;
    mv2v_request.frames = Some(45);
    let mv2v_frame = mlx_gen::gen_core::Image {
        width: mv2v_request.width,
        height: mv2v_request.height,
        pixels: vec![4; mv2v_request.width as usize * mv2v_request.height as usize * 3],
    };
    mv2v_request.conditioning = (0..2)
        .map(|_| mlx_gen::gen_core::Conditioning::VideoClip {
            frames: vec![mv2v_frame.clone(); 45],
            frame_idx: 0,
            strength: 1.0,
        })
        .collect();
    let mv2v_axis = mv2v_clip_receipt(&contract.provider_id, &mv2v_request)?;
    let mut mv2v_context = fixture.context.clone();
    mv2v_context.mode = MemoryMode::Other("multi_video_to_video".to_owned());
    mv2v_context.geometry.reference_count = 2;
    mv2v_context.overlay = Some(format!("provider_video_mode:mv2v+{mv2v_axis}"));
    let mut ads2v_request = rv2v_request.clone();
    ads2v_request.prompt = "weights-free Bernini ads2v memory behavior".to_owned();
    ads2v_request.video_mode = Some("ads2v".to_owned());
    let clip = match &ads2v_request.conditioning[0] {
        mlx_gen::gen_core::Conditioning::VideoClip {
            frames,
            frame_idx,
            strength,
        } => mlx_gen::gen_core::Conditioning::VideoClip {
            frames: frames.clone(),
            frame_idx: *frame_idx,
            strength: *strength,
        },
        _ => unreachable!("RV2V fixture starts with a clip"),
    };
    let images = match ads2v_request.conditioning.pop() {
        Some(mlx_gen::gen_core::Conditioning::MultiReference { images }) => images,
        _ => unreachable!("RV2V fixture ends with images"),
    };
    ads2v_request.conditioning = vec![
        clip.clone(),
        clip,
        mlx_gen::gen_core::Conditioning::MultiReference { images },
    ];
    let ads2v_axis = ads2v_source_receipt(&contract.provider_id, &ads2v_request)?;
    let mut ads2v_context = fixture.context.clone();
    ads2v_context.mode = MemoryMode::Other("ads2v".to_owned());
    ads2v_context.geometry.reference_count = 4;
    ads2v_context.overlay = Some(format!("provider_video_mode:ads2v+{ads2v_axis}"));
    let load_spec = fixture.load_spec.clone();
    Ok(vec![
        fixture,
        MemoryBehaviorFixture {
            context: r2v_context,
            request: r2v_request,
            load_spec: load_spec.clone(),
        },
        MemoryBehaviorFixture {
            context: rv2v_context,
            request: rv2v_request,
            load_spec: load_spec.clone(),
        },
        MemoryBehaviorFixture {
            context: mv2v_context,
            request: mv2v_request,
            load_spec: load_spec.clone(),
        },
        MemoryBehaviorFixture {
            context: ads2v_context,
            request: ads2v_request,
            load_spec,
        },
    ])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub fn begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

/// Provider safety for a **loaded** generator (sc-18609).
///
/// Identical to the crate-private `safety_check` plus the one fact only a loaded generator has: the snapshot's real
/// expert depth. See [`check_loaded_expert_depth`].
pub fn loaded_safety_check(
    provider_id: &str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_expert_blocks: usize,
) -> MemorySafetyDecision {
    if contract.engages(
        context.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency,
    ) {
        if let Err(error) = check_loaded_expert_depth(provider_id, loaded_expert_blocks) {
            return MemorySafetyDecision::Reject {
                reason: error.to_string(),
            };
        }
    }
    safety_check(spec, contract, context)
}

/// Request scope for a **loaded** generator: [`begin_request`] behind [`loaded_safety_check`].
pub fn loaded_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_expert_blocks: usize,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        loaded_safety_check(provider_id, spec, contract, context, loaded_expert_blocks)
    {
        return Err(CoreError::Unsupported(reason));
    }
    begin_request(provider_id, spec, contract, context)
}

fn begin_with_cleanup(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let route = match context.mode.as_key() {
        "video_to_video" => BerniniMemoryRoute::Clip,
        "reference_to_video" => BerniniMemoryRoute::Images,
        "reference_video_to_video" => BerniniMemoryRoute::ClipAndImages,
        "multi_video_to_video" => BerniniMemoryRoute::MultiClip,
        "ads2v" => BerniniMemoryRoute::Ads2v,
        mode => {
            return Err(CoreError::Unsupported(format!(
                "bernini memory scope cannot configure crossed mode {mode}"
            )))
        }
    };
    let expected_reference_receipt = source_receipt_from_overlay(context.overlay.as_deref());
    // VideoClip is a temporal carrier and therefore maps to zero image references in the shared
    // core. MultiReference is flattened and stays equal to the exact 1-8 image count.
    let mut core_geometry = context.geometry;
    if matches!(
        route,
        BerniniMemoryRoute::Clip | BerniniMemoryRoute::MultiClip
    ) {
        core_geometry.reference_count = 0;
    } else if route == BerniniMemoryRoute::ClipAndImages {
        core_geometry.reference_count = core_geometry.reference_count.saturating_sub(1);
    } else if route == BerniniMemoryRoute::Ads2v {
        core_geometry.reference_count = core_geometry.reference_count.saturating_sub(2);
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        core_geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        // The WHOLE trunk, both experts, in one global index space — see the module docs. Derived
        // from the model config rather than written as 80.
        trunk_blocks(),
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{provider_id}: bernini has no PiD decoder"
                )));
            }
            validate_decode(Some(edge), Some(overlap))
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(BerniniMemoryRequestScope {
        inner: mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
        provider_id,
        route,
        expected_reference_receipt,
    })))
}

fn validate_video_to_video_request(request: &GenerationRequest) -> CoreResult<()> {
    if request.phases.is_some() {
        return Err(CoreError::Unsupported(
            "bernini memory scope does not implement multi-phase denoise requests".to_owned(),
        ));
    }
    if request.video_mode.as_deref() != Some("v2v") {
        return Err(CoreError::Unsupported(
            "bernini memory scope requires video_mode=v2v".to_owned(),
        ));
    }
    if request.fps.unwrap_or(16) != 16 {
        return Err(CoreError::Unsupported(
            "bernini memory scope requires FPS 16".to_owned(),
        ));
    }
    if request.count != 1
        || request.image_reference_count() != 0
        || request.video_clips().len() != 1
        || !matches!(
            request.conditioning.as_slice(),
            [mlx_gen::gen_core::Conditioning::VideoClip { .. }]
        )
    {
        return Err(CoreError::Unsupported(
            "bernini memory scope requires exactly one normalized VideoClip and no image references"
                .to_owned(),
        ));
    }
    if !matches!(request.frames, Some(45 | 61 | 77)) {
        return Err(CoreError::Unsupported(
            "bernini memory scope supports only the advertised 3/4/5 second frame counts at FPS 16"
                .to_owned(),
        ));
    }
    validate_geometry(request.width, request.height)?;
    Ok(())
}

fn validate_reference_to_video_request(
    provider_id: &str,
    request: &GenerationRequest,
) -> CoreResult<String> {
    if request.phases.is_some() {
        return Err(CoreError::Unsupported(
            "bernini r2v memory scope does not implement multi-phase denoise requests".to_owned(),
        ));
    }
    if request.video_mode.as_deref() != Some("r2v") {
        return Err(CoreError::Unsupported(
            "bernini r2v memory scope requires video_mode=r2v".to_owned(),
        ));
    }
    if request.fps.unwrap_or(16) != 16 || request.count != 1 {
        return Err(CoreError::Unsupported(
            "bernini r2v memory scope requires FPS16 and count 1".to_owned(),
        ));
    }
    if !matches!(request.frames, Some(45 | 61 | 77)) {
        return Err(CoreError::Unsupported(
            "bernini r2v memory scope supports only 45/61/77 frames".to_owned(),
        ));
    }
    validate_geometry(request.width, request.height)?;
    if !request.video_clips().is_empty()
        || !matches!(
            request.conditioning.as_slice(),
            [mlx_gen::gen_core::Conditioning::MultiReference { .. }]
        )
    {
        return Err(CoreError::Unsupported(
            "bernini r2v memory scope requires images only".to_owned(),
        ));
    }
    r2v_reference_receipt(provider_id, request)
}

fn validate_reference_video_to_video_request(
    provider_id: &str,
    request: &GenerationRequest,
) -> CoreResult<String> {
    if request.phases.is_some()
        || request.video_mode.as_deref() != Some("rv2v")
        || request.fps.unwrap_or(16) != 16
        || request.count != 1
        || !matches!(request.frames, Some(45 | 61 | 77))
        || request.video_clips().len() != 1
        || !(1..=8).contains(&request.image_reference_count())
        || !matches!(
            request.conditioning.as_slice(),
            [
                mlx_gen::gen_core::Conditioning::VideoClip { .. },
                mlx_gen::gen_core::Conditioning::MultiReference { .. }
            ]
        )
    {
        return Err(CoreError::Unsupported(
            "bernini rv2v requires one normalized VideoClip followed by one MultiReference with 1-8 images, count1, FPS16, and 45/61/77 frames"
                .to_owned(),
        ));
    }
    validate_geometry(request.width, request.height)?;
    r2v_reference_receipt(provider_id, request)
}

fn validate_multi_video_to_video_request(
    provider_id: &str,
    request: &GenerationRequest,
) -> CoreResult<String> {
    if request.phases.is_some()
        || request.fps.unwrap_or(16) != 16
        || request.count != 1
        || !matches!(request.frames, Some(45 | 61 | 77))
    {
        return Err(CoreError::Unsupported(
            "bernini mv2v requires count1, FPS16, and 45/61/77 frames".into(),
        ));
    }
    validate_geometry(request.width, request.height)?;
    mv2v_clip_receipt(provider_id, request)
}

fn validate_ads2v_request(provider_id: &str, request: &GenerationRequest) -> CoreResult<String> {
    if request.phases.is_some()
        || request.fps.unwrap_or(16) != 16
        || request.count != 1
        || !matches!(request.frames, Some(45 | 61 | 77))
    {
        return Err(CoreError::Unsupported(
            "bernini ads2v requires count1, FPS16, and 45/61/77 frames".into(),
        ));
    }
    validate_geometry(request.width, request.height)?;
    ads2v_source_receipt(provider_id, request)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BerniniMemoryRoute {
    Clip,
    Images,
    ClipAndImages,
    MultiClip,
    Ads2v,
}

struct BerniniMemoryRequestScope {
    inner: mlx_gen::request_scope::MlxRequestScopeCore,
    provider_id: &'static str,
    route: BerniniMemoryRoute,
    expected_reference_receipt: Option<String>,
}

impl MemoryRequestScope for BerniniMemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        match self.route {
            BerniniMemoryRoute::Clip => validate_video_to_video_request(request)?,
            BerniniMemoryRoute::Images => {
                let actual = validate_reference_to_video_request(self.provider_id, request)?;
                if self.expected_reference_receipt.as_deref() != Some(actual.as_str()) {
                    return Err(CoreError::Unsupported(
                        "bernini r2v references changed after admission".to_owned(),
                    ));
                }
            }
            BerniniMemoryRoute::ClipAndImages => {
                let actual = validate_reference_video_to_video_request(self.provider_id, request)?;
                if self.expected_reference_receipt.as_deref() != Some(actual.as_str()) {
                    return Err(CoreError::Unsupported(
                        "bernini rv2v sources changed after admission".to_owned(),
                    ));
                }
            }
            BerniniMemoryRoute::MultiClip => {
                let actual = validate_multi_video_to_video_request(self.provider_id, request)?;
                if self.expected_reference_receipt.as_deref() != Some(actual.as_str()) {
                    return Err(CoreError::Unsupported(
                        "bernini mv2v ordered sources changed after admission".to_owned(),
                    ));
                }
            }
            BerniniMemoryRoute::Ads2v => {
                let actual = validate_ads2v_request(self.provider_id, request)?;
                if self.expected_reference_receipt.as_deref() != Some(actual.as_str()) {
                    return Err(CoreError::Unsupported(
                        "bernini ads2v ordered sources changed after admission".to_owned(),
                    ));
                }
            }
        }
        self.inner.configure_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> CoreResult<()> {
        self.inner.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> CoreResult<()> {
        self.inner.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        mut geometry: MemoryGeometry,
    ) -> CoreResult<()> {
        if matches!(
            self.route,
            BerniniMemoryRoute::Clip | BerniniMemoryRoute::MultiClip
        ) {
            geometry.reference_count = 0;
        } else if self.route == BerniniMemoryRoute::ClipAndImages {
            geometry.reference_count = geometry.reference_count.saturating_sub(1);
        } else if self.route == BerniniMemoryRoute::Ads2v {
            geometry.reference_count = geometry.reference_count.saturating_sub(2);
        }
        self.inner.configure_decode(tile_edge, overlap, geometry)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> CoreResult<()> {
        self.inner.configure_attention(chunk_size)
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        self.inner
            .materialize_transformer_window(first_block, block_count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> CoreResult<()> {
        self.inner.finish(outcome)
    }
}

/// The registry rows for one adopting provider.
///
/// Both providers get the full set — contract, safety check, weights-free contract fixture and
/// behaviour — because a half-registered family is exactly the "declaration is not reachability"
/// hazard: a contract nothing resolves through cannot be walked by registry conformance, and a
/// behaviour registration without a contract fixture cannot be exercised weights-free.
macro_rules! memory_registration {
    ($registration:ident, $behavior:ident, $provider_id:expr) => {
        pub(crate) const $registration: mlx_gen::gen_core::MemoryRegistration =
            mlx_gen::gen_core::MemoryRegistration {
                provider_id: $provider_id,
                contract: |spec| {
                    crate::memory_strategy::memory_strategy_contract($provider_id, spec)
                },
                safety_check: crate::memory_strategy::safety_check,
            };
        pub(crate) const $behavior: mlx_gen::gen_core::MemoryBehaviorRegistration =
            mlx_gen::gen_core::MemoryBehaviorRegistration {
                provider_id: $provider_id,
                valid_fixtures: crate::memory_strategy::registered_valid_fixture,
                begin_request: |spec, contract, context| {
                    crate::memory_strategy::registered_begin_request(
                        $provider_id,
                        spec,
                        contract,
                        context,
                    )
                },
            };
    };
}

memory_registration!(
    RENDERER_MEMORY_REGISTRATION,
    RENDERER_MEMORY_BEHAVIOR,
    RENDERER_ID
);
memory_registration!(FULL_MEMORY_REGISTRATION, FULL_MEMORY_BEHAVIOR, FULL_ID);

// ── Request-side resolution: the shared `GenerationMemory` signal → this provider's levers ────────

/// Rung 3: the budget applied to every trunk SDPA seam.
///
/// A budget rather than an `AttentionPlan` because a `WanBlockStream` outlives the borrow a plan's
/// cancel flag would need. The denoise loop already checks cancellation once per step and a rung-3
/// chunk is far shorter than a step, so nothing is lost.
///
/// **Scope, stated precisely.** This covers both per-block SDPA seams of the denoise trunk — the
/// self-attention over the packed `[sources…, target]` sequence and the cross-attention over the
/// prompt streams — on both experts, resident or windowed. It does **not** cover the planner's
/// hand-rolled softmax (`qwen2_5_vl.rs`, `vision.rs`): those run in the conditioning phase, which is
/// fully released before either expert loads, so bounding them cannot move the request peak the way
/// bounding the trunk can. Tracked separately; see the story linked on sc-15528.
pub(crate) fn attention_budget(req: &GenerationRequest) -> mlx_gen::attention::AttentionBudget {
    match req.memory {
        Some(memory) if memory.chunk_attention => mlx_gen::attention::AttentionBudget::CONSTRAINED,
        _ => mlx_gen::attention::AttentionBudget::UNBOUNDED,
    }
}

/// Rung 4: the requested cadence, or `None` for the resident stack.
///
/// A scope this family does not implement — or a cadence outside the published
/// [`TRANSFORMER_WINDOW_SIZES`] — is a typed rejection rather than a silently narrowed (or silently
/// *widened*) execution.
pub(crate) fn transformer_window_size(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory
        .transformer_window_component
        .unwrap_or(TRANSFORMER_WINDOW_COMPONENT);
    if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "bernini implements only the {TRANSFORMER_WINDOW_COMPONENT:?} transformer window \
             component, got {component:?}"
        )));
    }
    let size = memory
        .transformer_window_size
        .unwrap_or(TRANSFORMER_WINDOW_SIZE);
    validate_window(size).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("bernini transformer window rejected: {error}"))
    })?;
    Ok(Some(size as usize))
}

/// Rung 2: the decode tiling for this request.
///
/// `None` means "this request did not select rung 2", and the caller then keeps
/// [`TilingConfig::auto`] — Bernini's shipped behaviour, which already tiles a large decode. Rung 2
/// is therefore *explicit geometry*, not "tiling versus no tiling": the A/B arms differ in exactly
/// one thing, the tile edge, and the harness must compare against the composition this extends
/// rather than against an untiled decode that never shipped.
pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    let edge = memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE);
    let overlap = memory.decode_overlap.unwrap_or(DECODE_OVERLAP);
    validate_decode(Some(edge), Some(overlap)).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("bernini decode tiling rejected: {error}"))
    })?;
    Ok(Some(TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    )))
}

/// Rung 1 has **no request-side resolver**, so it is deliberately declared `Missing`.
///
/// Both `generate_impl`s release every completed phase unconditionally: the UMT5 encoder is dropped
/// and `clear_cache`d before the source-VAE encode, the source-VAE encoder before the experts, and
/// the experts before the decode. `GenerationMemory::stage_residency` therefore selects nothing.
/// This older unconditional staging behavior is not a request-selected rung and cannot be admitted
/// as `StagedResidency` until a selector-controlled mechanism is wired through the provider.
///
/// A resolver that branched on the flag would be a lever over behaviour that does not vary, i.e. a
/// declaration with no enforcement behind it. The synchronized phase release remains useful
/// lifecycle behavior, but it does not establish the memory-ladder rung.
///
/// **What this does NOT include**, stated so it is not read as covered: releasing the *high* expert
/// at the boundary switch so only one expert is resident during the denoise. The switch is monotone,
/// so that is sound, and sc-16354 identifies it as the cheap half of the dual-expert problem — but it
/// needs `BVitExpert`/`BExpert` to own their transformer rather than borrow it, which is a larger
/// refactor than this story lands. Rung 4 already achieves strictly more (zero blocks of BOTH
/// experts), so nothing here depends on it. Tracked separately; see the story linked on sc-15528.
const _RUNG_ONE_IS_UNCONDITIONAL: () = ();

/// The strategy parameters this provider accepts, for a caller that wants the whole domain in one
/// value (the conformance tests and the SceneWorks evidence writer both key off this).
pub fn declared_parameters() -> mlx_gen::gen_core::MemoryStrategyParameters {
    mlx_gen::gen_core::MemoryStrategyParameters {
        stage_residency: None,
        decode_tile_edge: Some(DECODE_TILE_EDGE),
        decode_overlap: Some(DECODE_OVERLAP),
        attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{Conditioning, GenerationMemory};
    use mlx_gen::{OffloadPolicy, WeightsSource};

    fn spec(shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/bernini-contract".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(shape)
    }

    fn contract(provider_id: &str, shape: LoadShape) -> MemoryProviderContract {
        weights_free_memory_strategy_contract(provider_id, &spec(shape)).expect("contract")
    }

    /// AC (SC-22662): both registered Bernini routes publish the axes of the Wan A14B trunk and the
    /// video VAE they run, and pass the shared facts conformance check.
    ///
    /// The published `transformer_blocks` is deliberately ONE expert's depth, not the 80-block
    /// windowing index space: a denoise step traverses one expert.
    #[test]
    fn architecture_facts_follow_the_wan_a14b_config_and_the_video_vae() {
        for provider_id in PROVIDER_IDS {
            let contract = contract(provider_id, LoadShape::DeferredMaterialization);
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(40),
                    // 5120 / 40, derived by `WanModelConfig::head_dim`.
                    head_dim: Some(128),
                    transformer_blocks: Some(40),
                    // `patch_size` is `(1, 2, 2)`: the square spatial patch is 2.
                    patch_size: Some(2),
                    latent_channels: Some(16),
                    vae_spatial_scale: Some(8),
                    // A video autoencoder: four frames per latent unit.
                    vae_temporal_scale: Some(4),
                    activation_dtype_width: Some(2),
                },
                "{provider_id} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        assert_eq!(
            contract(FULL_ID, LoadShape::DeferredMaterialization)
                .architecture_facts
                .transformer_blocks,
            Some(expert_blocks() as u32),
            "the published depth is one expert's, never the 80-block windowing index space"
        );
    }

    fn spec_for_config(dir: &std::path::Path, config: &serde_json::Value) -> LoadSpec {
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(dir.to_path_buf()))
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    /// AC (SC-22662, review follow-up): on the **materialized** path the trunk axes come from the
    /// snapshot's own `config.json` — the file both Bernini loaders parse — rather than from the
    /// compile-time preset. The mirror fixture agrees with the weights-free path; a fixture with
    /// mutated keys publishes the mutated axes, which is what the unconditional
    /// `architecture_facts()` this replaced would fail.
    ///
    /// The third arm is the one that makes the `config.json` existence filter load-bearing: a
    /// materialized snapshot with **no** `config.json` must still publish the A14B preset.
    /// `WanModelConfig::from_model_dir` falls back to the TI2V-5B preset there, and publishing that
    /// would describe a 30-block, 24-head, z48 model this provider never runs.
    #[test]
    fn materialized_trunk_axes_come_from_the_snapshot_rather_than_the_preset() {
        let preset = WanModelConfig::wan22_t2v_14b();

        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&spec_for_config(mirror.path(), &preset.to_json())),
            architecture_facts(&spec(LoadShape::DeferredMaterialization)),
            "a snapshot mirroring the reference config must publish the preset's axes"
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = preset.to_json();
        mutated["num_layers"] = serde_json::json!(7);
        mutated["vae_z_dim"] = serde_json::json!(32);
        let mutated_facts = architecture_facts(&spec_for_config(mutated_dir.path(), &mutated));
        assert_eq!(
            (
                mutated_facts.transformer_blocks,
                mutated_facts.latent_channels
            ),
            (Some(7), Some(32)),
            "the materialized path must publish the snapshot's geometry, not the preset's"
        );

        let bare = tempfile::tempdir().unwrap();
        let bare_spec = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            bare.path().to_path_buf(),
        ))
        .with_load_shape(LoadShape::DeferredMaterialization);
        assert_eq!(
            architecture_facts(&bare_spec).transformer_blocks,
            Some(preset.num_layers as u32),
            "a config-less snapshot must keep the A14B preset, never fall through to TI2V-5B"
        );
    }

    /// **The dual-expert invariant (sc-16354).** The trunk rung 4 windows over is BOTH experts in one
    /// global index space, so the block count the shared request scope validates against is 80.
    ///
    /// A contract that declared 40 would accept a window at block 39 and reject one at block 40 — the
    /// low expert's first block — which is exactly the "windowing one expert while the other stays
    /// resident" shape the survey warned about, expressed as an off-by-one-expert admission bug.
    #[test]
    fn the_windowed_trunk_is_both_experts() {
        assert_eq!(expert_blocks(), 40, "a Wan A14B expert is 40 blocks");
        assert_eq!(
            trunk_blocks(),
            80,
            "rung 4 windows the whole dual-expert trunk, not one expert"
        );
    }

    /// Every published cadence divides one expert's depth, so the low expert's first window (global
    /// block 40) is aligned under the shared scope's `first_block % window == 0` rule. A cadence that
    /// did not divide 40 would be admitted by the contract and then rejected mid-denoise.
    #[test]
    fn the_published_window_sizes_all_divide_one_expert() {
        let expert = expert_blocks() as u32;
        for &size in TRANSFORMER_WINDOW_SIZES {
            assert_eq!(
                expert % size,
                0,
                "window {size} does not divide the {expert}-block expert, so the low expert's \
                 first window would be mis-aligned in the trunk index space"
            );
            assert!(size < expert, "window {size} degenerates to fully resident");
        }
        assert!(
            TRANSFORMER_WINDOW_SIZES.contains(&TRANSFORMER_WINDOW_SIZE),
            "the default cadence must be inside the published domain"
        );
    }

    /// Rung 4 is declared per LOAD. An eager load publishes it `Missing`; only a
    /// `DeferredMaterialization` load publishes it `Implemented`. Staged residency remains Missing
    /// on both shapes because the provider has no request-selected rung-1 lever.
    #[test]
    fn rung_four_is_declared_per_load_and_moves_nothing_else() {
        for provider_id in PROVIDER_IDS {
            let deferred = contract(provider_id, LoadShape::DeferredMaterialization);
            let eager = contract(provider_id, LoadShape::EagerMaterialization);
            for rung in [
                MemoryStrategy::Resident,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
            ] {
                assert_eq!(
                    deferred.capability(rung).map(|c| &c.support),
                    Some(&MemoryStrategySupport::Implemented),
                    "{provider_id}: {rung:?} must be implemented on a deferred load"
                );
                assert_eq!(
                    eager.capability(rung).map(|c| &c.support),
                    Some(&MemoryStrategySupport::Implemented),
                    "{provider_id}: {rung:?} must not depend on the load shape"
                );
            }
            assert_eq!(
                deferred
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|c| &c.support),
                Some(&MemoryStrategySupport::Implemented)
            );
            assert_eq!(
                eager
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|c| &c.support),
                Some(&MemoryStrategySupport::Missing),
                "{provider_id}: a window over an already-materialized trunk bounds nothing"
            );
        }
    }

    /// Rung 4 does NOT declare a rung-1 prerequisite on this family, and that is a deliberate
    /// departure from Chroma/Anima justified in the module docs. Pinning it keeps a later
    /// copy-paste from quietly importing an edge that would charge a caller for a saving rung 4
    /// already has.
    #[test]
    fn rung_four_does_not_require_rung_one_here() {
        let contract = contract(FULL_ID, LoadShape::DeferredMaterialization);
        assert!(
            contract.additional_prerequisites.is_empty(),
            "bernini declares no provider-specific prerequisite edges, got {:?}",
            contract.additional_prerequisites
        );
        assert!(
            !contract.engages(
                MemoryStrategy::BoundedTransformerResidency,
                MemoryStrategy::StagedResidency
            ),
            "rung 4 must not drag rung 1 in by cost order"
        );
    }

    /// Every published parameter is accepted and everything outside the domain is refused — on the
    /// production path, by the same validators admission uses.
    #[test]
    fn the_published_domains_are_the_accepted_domains() {
        for &edge in DECODE_TILE_EDGES {
            validate_decode(Some(edge), Some(DECODE_OVERLAP))
                .unwrap_or_else(|error| panic!("published edge {edge} refused: {error}"));
        }
        // Off-domain edges: one below the published floor, one that is not a published multiple, and
        // one from a PiD-shaped domain this provider does not have.
        for edge in [128_u32, 700, 1024] {
            assert!(
                validate_decode(Some(edge), Some(DECODE_OVERLAP)).is_err(),
                "edge {edge} is outside the published domain and must be refused"
            );
        }
        assert!(validate_decode(Some(DECODE_TILE_EDGE), Some(32)).is_err());
        assert!(validate_decode(None, Some(DECODE_OVERLAP)).is_err());
        assert!(validate_decode(Some(DECODE_TILE_EDGE), None).is_err());

        for &size in TRANSFORMER_WINDOW_SIZES {
            validate_window(size).unwrap_or_else(|error| panic!("window {size} refused: {error}"));
        }
        // 3, 6, 7 and 9 are all plausible cadences that do NOT divide 40.
        for size in [0_u32, 3, 6, 7, 9, 40, 41] {
            assert!(
                validate_window(size).is_err(),
                "window {size} must be refused: it does not divide one expert's depth"
            );
        }

        validate_attention(ATTENTION_CHUNK_SIZE).expect("the published chunk size");
        assert!(validate_attention(ATTENTION_CHUNK_SIZE + 1).is_err());
    }

    /// The request-side resolvers refuse the same values the contract refuses, so a request that
    /// bypassed admission still cannot execute an unpublished geometry.
    #[test]
    fn the_request_resolvers_refuse_what_the_contract_refuses() {
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };

        // Absent memory block: every lever off, nothing rejected.
        assert!(decode_tiling(&base).expect("no memory").is_none());
        assert!(transformer_window_size(&base).expect("no memory").is_none());
        assert!(attention_budget(&base).is_unbounded());

        let with = |memory: GenerationMemory| GenerationRequest {
            memory: Some(memory),
            ..base.clone()
        };

        // Rung 2 accepts the published domain and refuses everything else.
        let ok = with(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(384),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(decode_tiling(&ok).expect("published edge").is_some());
        let bad = with(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(700),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(decode_tiling(&bad).is_err(), "700 is not published");

        // Rung 4 accepts a published cadence, refuses an unpublished one, and refuses a scope this
        // family does not implement.
        let ok = with(GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(20),
            ..Default::default()
        });
        assert_eq!(transformer_window_size(&ok).expect("published"), Some(20));
        let bad = with(GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(3),
            ..Default::default()
        });
        assert!(
            transformer_window_size(&bad).is_err(),
            "3 does not divide 40"
        );
        let scope = with(GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_component: Some(TransformerComponent::TextEncoder),
            ..Default::default()
        });
        let error = match transformer_window_size(&scope) {
            Ok(_) => panic!("the TextEncoder scope is not implemented and must be refused"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("TextEncoder"), "{error}");

        // Rung 4 defaulting: an enabled block with no cadence takes the published default, not 1.
        let defaulted = with(GenerationMemory {
            stream_transformer_blocks: true,
            ..Default::default()
        });
        assert_eq!(
            transformer_window_size(&defaulted).expect("default"),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );

        // Rung 3 is a plain switch. Rung 1 has no resolver by design — see
        // `_RUNG_ONE_IS_UNCONDITIONAL`.
        let chunked = with(GenerationMemory {
            chunk_attention: true,
            ..Default::default()
        });
        assert!(!attention_budget(&chunked).is_unbounded());
    }

    /// No PiD route, no resident overlay, one output per request — each stated positively rather than by
    /// omission, because the sc-15839 review found all three shipped as silent gaps elsewhere.
    #[test]
    fn the_absent_capabilities_are_declared_absent() {
        for provider_id in PROVIDER_IDS {
            let contract = contract(provider_id, LoadShape::DeferredMaterialization);
            assert!(
                contract.pid_decode_routes.is_none(),
                "{provider_id}: bernini has no PiD route"
            );
            assert_eq!(
                contract.asset_facts.overlay_bytes, 0,
                "{provider_id}: bernini loads no auxiliary resident network"
            );
            assert_eq!(
                contract.resident_request_memory,
                ResidentRequestMemory::PreserveLoadDefaults
            );
        }
    }

    /// The exact declared surface both variants publish across the whole MLX registry-load matrix
    /// (sc-18609) — 3 artifact tiers x 2 offload policies x 2 load shapes, per provider.
    ///
    /// The per-rung counts are the family's real shape, not a copy of a neighbour's: rung 1 is
    /// Missing because staging is unconditional, while rung 4 rides the load shape ALONE. It is 6
    /// rather than 3 per provider
    /// because `structurally_streamable` does not consult `OffloadPolicy` — both descriptors advertise
    /// `supports_sequential_offload: false` (the family is unconditionally staged, so the policy is
    /// not a rung-1 lever here), and gating rung 4 on a control the provider does not consume would declare a
    /// dependency that does not exist.
    #[test]
    fn both_variants_publish_the_exact_declared_surface_ladder() {
        use mlx_gen::gen_core::{LoadShape as Shape, MemoryContractSurfaceTier};
        use std::collections::BTreeSet;

        let registry = crate::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        assert_eq!(
            surfaces.len(),
            24,
            "2 providers x 12 registry-load surfaces"
        );

        for provider_id in PROVIDER_IDS {
            let provider: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider.len(), 12, "{provider_id}");
            let selectors: BTreeSet<_> = provider
                .iter()
                .map(|surface| surface.selector.id())
                .collect();
            let expected: BTreeSet<_> = [
                "bf16:resident:eager",
                "bf16:resident:deferred",
                "bf16:sequential:eager",
                "bf16:sequential:deferred",
                "q4:resident:eager",
                "q4:resident:deferred",
                "q4:sequential:eager",
                "q4:sequential:deferred",
                "q8:resident:eager",
                "q8:resident:deferred",
                "q8:sequential:eager",
                "q8:sequential:deferred",
            ]
            .into_iter()
            .collect();
            assert_eq!(selectors, expected, "{provider_id}");

            let count = |strategy| {
                provider
                    .iter()
                    .filter(|surface| {
                        surface
                            .contract
                            .capability(strategy)
                            .expect("the complete bernini ladder")
                            .support
                            == MemoryStrategySupport::Implemented
                    })
                    .count()
            };
            assert_eq!(count(MemoryStrategy::Resident), 12, "{provider_id}");
            assert_eq!(count(MemoryStrategy::StagedResidency), 0, "{provider_id}");
            assert_eq!(count(MemoryStrategy::BoundedDecode), 12, "{provider_id}");
            assert_eq!(count(MemoryStrategy::BoundedAttention), 12, "{provider_id}");
            assert_eq!(
                count(MemoryStrategy::BoundedTransformerResidency),
                6,
                "{provider_id}: rung 4 rides the load shape alone"
            );

            for surface in &provider {
                assert!(!surface.composed, "{provider_id}");
                assert!(
                    matches!(
                        surface.resolved_artifact_tier(),
                        MemoryContractSurfaceTier::Bf16
                            | MemoryContractSurfaceTier::Q4
                            | MemoryContractSurfaceTier::Q8
                    ),
                    "{provider_id}: MLX facts omit NVFP4"
                );
                let windowed = surface
                    .contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support
                    == MemoryStrategySupport::Implemented;
                assert_eq!(
                    windowed,
                    surface.selector.load_shape == Shape::DeferredMaterialization,
                    "{provider_id} [{}]",
                    surface.selector.id()
                );
                if windowed {
                    assert_eq!(
                        surface
                            .contract
                            .capability(MemoryStrategy::BoundedTransformerResidency)
                            .unwrap()
                            .parameters
                            .transformer_window_sizes,
                        TRANSFORMER_WINDOW_SIZES.to_vec(),
                        "{provider_id}"
                    );
                }
                assert!(
                    surface
                        .contract
                        .calibration
                        .as_ref()
                        .is_some_and(|identity| identity.fingerprint == STATIC_CALIBRATION),
                    "{provider_id}: a declaration surface carries the declaration key only"
                );
            }
        }
    }

    /// A phased request trajectory is refused, not silently rendered single-phase. Nothing in this
    /// crate reads `GenerationRequest::phases`, so admitting one would record evidence for a
    /// trajectory Bernini cannot run.
    #[test]
    fn a_phased_request_trajectory_is_refused() {
        for provider_id in PROVIDER_IDS {
            let spec = spec(LoadShape::DeferredMaterialization);
            let contract = contract(provider_id, LoadShape::DeferredMaterialization);
            let mut context = registered_valid_fixture(
                &spec,
                &contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap()
            .remove(0)
            .context;
            assert!(!context.has_phases);
            assert_eq!(
                safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Accept
            );
            context.has_phases = true;
            assert!(matches!(
                safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Reject { .. }
            ));
            assert!(
                registered_begin_request(provider_id, &spec, &contract, &context).is_err(),
                "{provider_id}"
            );
        }
    }

    /// The loaded-trunk-depth guard refuses exactly the depths whose experts no published cadence can
    /// align to, and refuses nothing else. Mutated one depth at a time rather than as a set.
    #[test]
    fn the_loaded_trunk_depth_guard_admits_only_the_declared_expert() {
        for provider_id in PROVIDER_IDS {
            check_loaded_expert_depth(provider_id, expert_blocks())
                .expect("the declared depth must be admitted");
            for depth in [0_usize, 1, 20, 30, 39, 41, 80] {
                let error = check_loaded_expert_depth(provider_id, depth)
                    .expect_err("an undeclared expert depth must be refused")
                    .to_string();
                assert!(error.contains(&depth.to_string()), "{error}");
            }
        }
    }

    /// An unknown provider id is refused rather than silently handed the family contract.
    #[test]
    fn an_unknown_provider_is_refused() {
        assert!(weights_free_memory_strategy_contract(
            "bernini_imaginary",
            &spec(LoadShape::DeferredMaterialization)
        )
        .is_err());
    }

    /// No production calibration is minted until a `MEMORY_EVIDENCE_V1` record exists, so no selector
    /// can reach an optimized fit on evidence this repository cannot show. This test is what will
    /// redden — deliberately — on the commit that mints the first cell.
    #[test]
    fn no_production_calibration_is_claimed_without_evidence() {
        for provider_id in PROVIDER_IDS {
            let contract =
                memory_strategy_contract(provider_id, &spec(LoadShape::DeferredMaterialization))
                    .expect("contract");
            assert!(
                contract.calibration.is_none(),
                "{provider_id}: a production calibration identity without a measured record would \
                 let a stale fit be selected"
            );
        }
        // The weights-free walk still gets a declaration key, so registry conformance can run.
        assert!(contract(FULL_ID, LoadShape::DeferredMaterialization)
            .calibration
            .is_some());
    }

    #[test]
    fn v2v_scope_rejects_plain_and_crossed_video_requests() {
        let clip = Conditioning::VideoClip {
            frames: vec![mlx_gen::gen_core::Image {
                width: 2,
                height: 2,
                pixels: vec![0; 12],
            }],
            frame_idx: 0,
            strength: 1.0,
        };
        let mut valid = GenerationRequest {
            prompt: "v2v".to_owned(),
            width: 848,
            height: 480,
            frames: Some(45),
            fps: Some(16),
            video_mode: Some("v2v".to_owned()),
            conditioning: vec![clip.clone()],
            ..Default::default()
        };
        assert!(validate_video_to_video_request(&valid).is_ok());
        valid.video_mode = None;
        assert!(validate_video_to_video_request(&valid).is_err());
        valid.video_mode = Some("v2v".to_owned());
        valid.fps = Some(24);
        assert!(validate_video_to_video_request(&valid).is_err());
        valid.fps = Some(16);
        valid.conditioning.clear();
        assert!(validate_video_to_video_request(&valid).is_err());
    }

    #[test]
    fn v2v_scope_admits_only_advertised_geometry_and_frames() {
        for &(width, height) in ADVERTISED_GEOMETRIES {
            for frames in [45, 61, 77] {
                let request = GenerationRequest {
                    prompt: "v2v".to_owned(),
                    width,
                    height,
                    frames: Some(frames),
                    fps: Some(16),
                    video_mode: Some("v2v".to_owned()),
                    conditioning: vec![Conditioning::VideoClip {
                        frames: vec![mlx_gen::gen_core::Image {
                            width: 2,
                            height: 2,
                            pixels: vec![0; 12],
                        }],
                        frame_idx: 0,
                        strength: 1.0,
                    }],
                    ..Default::default()
                };
                assert!(
                    validate_video_to_video_request(&request).is_ok(),
                    "{width}x{height}/{frames}"
                );
            }
        }
        let mut crossed = GenerationRequest {
            width: 640,
            height: 640,
            frames: Some(45),
            fps: Some(16),
            video_mode: Some("v2v".to_owned()),
            conditioning: vec![Conditioning::VideoClip {
                frames: vec![mlx_gen::gen_core::Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 12],
                }],
                frame_idx: 0,
                strength: 1.0,
            }],
            ..Default::default()
        };
        assert!(validate_video_to_video_request(&crossed).is_err());
        crossed.width = 848;
        crossed.height = 480;
        crossed.fps = Some(24);
        assert!(validate_video_to_video_request(&crossed).is_err());
    }

    /// Safety owns the context axis before a request scope is constructed. A valid request cannot
    /// rescue a context whose frame cell was crossed after selection.
    #[test]
    fn v2v_safety_binds_the_exact_context_frame_cell_before_configure() {
        for provider_id in PROVIDER_IDS {
            let spec = spec(LoadShape::DeferredMaterialization);
            let contract = contract(provider_id, LoadShape::DeferredMaterialization);
            let mut context =
                registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .remove(0)
                    .context;
            for frames in [45, 61, 77] {
                context.geometry.frames = frames;
                assert_eq!(
                    safety_check(&spec, &contract, &context),
                    MemorySafetyDecision::Accept,
                    "{provider_id}/{frames}"
                );
            }
            for frames in [1, 44, 46, 60, 62, 76, 78] {
                context.geometry.frames = frames;
                assert!(
                    matches!(
                        safety_check(&spec, &contract, &context),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{provider_id}/{frames} must be rejected before configure_request"
                );
                assert!(
                    registered_begin_request(provider_id, &spec, &contract, &context).is_err(),
                    "{provider_id}/{frames} crossed context reached request scope"
                );
            }
        }
    }

    fn r2v_request() -> GenerationRequest {
        GenerationRequest {
            prompt: "ordered subjects".to_owned(),
            width: 848,
            height: 480,
            frames: Some(45),
            fps: Some(16),
            video_mode: Some("r2v".to_owned()),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![
                    mlx_gen::gen_core::Image {
                        width: 640,
                        height: 360,
                        pixels: vec![11; 640 * 360 * 3],
                    },
                    mlx_gen::gen_core::Image {
                        width: 360,
                        height: 640,
                        pixels: vec![29; 360 * 640 * 3],
                    },
                ],
            }],
            ..Default::default()
        }
    }

    fn rv2v_request() -> GenerationRequest {
        let mut request = r2v_request();
        request.prompt = "ordered clip and subjects".to_owned();
        request.video_mode = Some("rv2v".to_owned());
        let [Conditioning::MultiReference { images }] = request.conditioning.as_mut_slice() else {
            unreachable!()
        };
        let images = std::mem::take(images);
        let frame = mlx_gen::gen_core::Image {
            width: request.width,
            height: request.height,
            pixels: vec![7; request.width as usize * request.height as usize * 3],
        };
        request.conditioning = vec![
            Conditioning::VideoClip {
                frames: vec![frame; request.frames.unwrap() as usize],
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::MultiReference { images },
        ];
        request
    }

    fn mv2v_request(count: usize) -> GenerationRequest {
        let mut request = rv2v_request();
        request.prompt = "ordered multi-source clips".to_owned();
        request.video_mode = Some("mv2v".to_owned());
        let frame = mlx_gen::gen_core::Image {
            width: request.width,
            height: request.height,
            pixels: vec![41; request.width as usize * request.height as usize * 3],
        };
        request.conditioning = (0..count)
            .map(|index| Conditioning::VideoClip {
                frames: vec![
                    mlx_gen::gen_core::Image {
                        pixels: vec![41 + index as u8; frame.pixels.len()],
                        ..frame.clone()
                    };
                    45
                ],
                frame_idx: 0,
                strength: 1.0,
            })
            .collect();
        request
    }

    #[test]
    fn mv2v_receipt_binds_count_order_shape_and_source_id_schedule() {
        let two = mv2v_request(2);
        let receipt = mv2v_clip_receipt(FULL_ID, &two).unwrap();
        assert!(receipt.contains("count-2:packed-source-tokens-"));
        assert!(receipt.contains("source-ids-1,2"));
        assert!(receipt.contains("video-1:frames-45;native-848x480;vae-12x"));
        assert!(receipt.contains("video-2:frames-45;native-848x480;vae-12x"));
        assert_eq!(mv2v_source_ids(5), ["1", "2", "3", "4", "5"]);
        assert_eq!(
            mv2v_source_ids(8),
            ["1", "1.571429", "2.142857", "2.714286", "3.285714", "3.857143", "4.428571", "5"]
        );
        let mut reversed = two.clone();
        reversed.conditioning.reverse();
        assert_ne!(receipt, mv2v_clip_receipt(FULL_ID, &reversed).unwrap());
        assert!(mv2v_clip_receipt(FULL_ID, &mv2v_request(1)).is_err());
    }

    #[test]
    fn ads2v_receipt_binds_roles_images_and_three_to_four_source_schedule() {
        let mut request = rv2v_request();
        request.video_mode = Some("ads2v".to_owned());
        let clip = request.conditioning[0].clone();
        let mut images = match request.conditioning.pop() {
            Some(mlx_gen::gen_core::Conditioning::MultiReference { images }) => images,
            _ => unreachable!(),
        };
        images.truncate(1);
        request.conditioning = vec![
            clip.clone(),
            clip,
            mlx_gen::gen_core::Conditioning::MultiReference { images },
        ];
        let mlx_gen::gen_core::Conditioning::VideoClip { frames, .. } =
            &mut request.conditioning[1]
        else {
            unreachable!()
        };
        frames[0].pixels[0] ^= 1;
        let three = ads2v_source_receipt(FULL_ID, &request).unwrap();
        assert!(three.contains("count-3:") && three.contains("source-ids-1,2,3"));
        let mut swapped = request.clone();
        swapped.conditioning.swap(0, 1);
        assert_ne!(three, ads2v_source_receipt(FULL_ID, &swapped).unwrap());
        let mlx_gen::gen_core::Conditioning::MultiReference { images } =
            &mut request.conditioning[2]
        else {
            unreachable!()
        };
        images.push(images[0].clone());
        let four = ads2v_source_receipt(FULL_ID, &request).unwrap();
        assert!(four.contains("source-ids-1,1.666667,2.333333,3"));
        assert_eq!(
            source_receipt_from_overlay(Some(&format!("provider_video_mode:ads2v+{four}"))),
            Some(four)
        );
    }

    fn distinct_references(count: usize) -> Vec<mlx_gen::gen_core::Image> {
        (0..count)
            .map(|index| mlx_gen::gen_core::Image {
                width: 28,
                height: 28,
                pixels: vec![u8::try_from(index + 1).unwrap(); 28 * 28 * 3],
            })
            .collect()
    }

    fn write_tier_file(path: &std::path::Path, bits: Option<i32>) {
        let mut tensors = serde_json::Map::new();
        let mut offset = 0_u64;
        let mut insert = |name: &str, dtype: &str, shape: &[usize], bytes: u64| {
            tensors.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        };
        match bits {
            Some(bits @ (4 | 8)) => {
                let words = usize::try_from(bits).unwrap() * 64 / 32;
                insert("block.weight", "U32", &[2, words], (2 * words * 4) as u64);
                insert("block.scales", "BF16", &[2, 1], 4);
                insert("block.biases", "BF16", &[2, 1], 4);
            }
            None => insert("block.weight", "BF16", &[2, 64], 256),
            Some(bits) => panic!("unsupported fixture q{bits}"),
        }
        let mut header = serde_json::to_vec(&tensors).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.resize(file.len() + usize::try_from(offset).unwrap(), 0);
        std::fs::write(path, file).unwrap();
    }

    fn tier_snapshot(bits: Option<i32>) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let quantization = bits.map(|bits| {
            serde_json::json!({
                "bits": bits,
                "group_size": 64,
            })
        });
        let config = serde_json::json!({ "quantization": quantization });
        std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
        std::fs::write(
            root.path().join("qwen2_5_vl_config.json"),
            config.to_string(),
        )
        .unwrap();
        for name in [
            "high_noise_model.safetensors",
            "low_noise_model.safetensors",
            "qwen2_5_vl.safetensors",
        ] {
            write_tier_file(&root.path().join(name), bits);
        }
        root
    }

    #[test]
    fn provider_owned_tier_resolver_reads_q4_q8_and_bf16_headers() {
        for (bits, expected) in [
            (Some(4), Some(mlx_gen::Quant::Q4)),
            (Some(8), Some(mlx_gen::Quant::Q8)),
            (None, None),
        ] {
            let root = tier_snapshot(bits);
            let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
            let tier = resolved_numeric_tier(&spec).unwrap();
            assert_eq!(tier.quant, expected, "fixture q{bits:?}");
            assert_eq!(
                crate::resolved_video_memory_numeric_tier(FULL_ID, &spec)
                    .unwrap()
                    .unwrap(),
                tier
            );

            let production = memory_strategy_contract(FULL_ID, &spec).unwrap();
            let declaration = weights_free_memory_strategy_contract(FULL_ID, &spec).unwrap();
            let mut context =
                registered_valid_fixture(&spec, &declaration, MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .remove(0)
                    .context;
            context.selection.tier = tier;
            context.optimization_authority =
                mlx_gen::gen_core::MemoryOptimizationAuthority::Estimated;
            // sc-22737: the context is seeded from the weights-free DECLARATION, whose identity is
            // `STATIC_CALIBRATION`, but the contract under test is the PRODUCTION one — which now
            // publishes a per-(provider, tier) key rather than nothing. Carry the production
            // identity, or the handshake rejects before any geometry clause is reached.
            context.calibration_fingerprint =
                production_calibration_fingerprint(FULL_ID, &spec).unwrap();
            assert_eq!(
                safety_check(&spec, &production, &context),
                MemorySafetyDecision::Accept,
                "provider safety must admit the resolved q{bits:?} tier"
            );
        }
    }

    // ------------------------------------------------------------------------------------------
    // sc-22737 (epic sc-22723 E1/E4): the per-(provider, tier) production identity.
    // ------------------------------------------------------------------------------------------

    /// The retained `(bernini, q4)` key is what the format string mints for that cell — so the
    /// generalization cannot silently restamp a record that is already filed under it.
    #[test]
    fn the_measured_cell_keeps_its_retained_string() {
        let root = tier_snapshot(Some(4));
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        assert_eq!(
            production_calibration_fingerprint(FULL_ID, &spec).as_deref(),
            Some(MEMORY_CALIBRATION_FINGERPRINT),
        );
    }

    /// Every shipped `(provider, tier)` publishes, and every cell publishes its OWN string: six
    /// loads, six distinct identities. Before sc-22737 all six published nothing, so no anchor
    /// could bind to a Bernini load on this lane at all.
    #[test]
    fn every_provider_and_tier_publishes_a_distinct_identity() {
        let mut seen = std::collections::BTreeSet::new();
        for bits in [None, Some(4), Some(8)] {
            let root = tier_snapshot(bits);
            let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
            for provider in PROVIDER_IDS {
                let fingerprint = production_calibration_fingerprint(provider, &spec)
                    .unwrap_or_else(|| panic!("{provider} q{bits:?} publishes an identity"));
                assert!(
                    seen.insert(fingerprint.clone()),
                    "{provider} q{bits:?} reuses {fingerprint}"
                );
                // The identity must reach the CONTRACT, not merely the table: the production
                // contract is what an evidence runner and the worker read.
                let contract = memory_strategy_contract(provider, &spec).unwrap();
                assert_eq!(
                    contract
                        .calibration
                        .as_ref()
                        .map(|identity| identity.fingerprint.as_str()),
                    Some(fingerprint.as_str()),
                    "{provider} q{bits:?}"
                );
            }
        }
        assert_eq!(seen.len(), 6);
    }

    /// The weights-free declaration surface may never publish a production string. A plan row
    /// naming the declaration key could otherwise be "satisfied" by a contract that loaded nothing.
    #[test]
    fn the_weights_free_declaration_cannot_publish_a_production_string() {
        let root = tier_snapshot(Some(4));
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let declaration = weights_free_memory_strategy_contract(FULL_ID, &spec).unwrap();
        let declared = declaration
            .calibration
            .as_ref()
            .unwrap()
            .fingerprint
            .clone();
        assert_eq!(declared, STATIC_CALIBRATION);
        for bits in [None, Some(4), Some(8)] {
            let root = tier_snapshot(bits);
            let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
            for provider in PROVIDER_IDS {
                assert_ne!(
                    production_calibration_fingerprint(provider, &spec).unwrap(),
                    declared,
                    "{provider} q{bits:?}"
                );
            }
        }
    }

    /// **Withheld, never failed.** Every shape whose tier the artifact does not prove publishes
    /// `None` and still loads: an unknown provider, a prepacked root asked for as a second
    /// quantization, a manifest that disagrees with its headers, and a root with no snapshot at
    /// all. Keying on `spec.quantize` instead of the resolved tier would publish a q4 identity for
    /// each of the middle two.
    #[test]
    fn a_tier_the_artifact_does_not_prove_is_withheld() {
        let root = tier_snapshot(Some(4));
        let base = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        assert!(production_calibration_fingerprint("not_a_bernini", &base).is_none());

        // A prepacked root that ALSO asks for a load-time quantization is not a shipped load. The
        // identity is withheld and the LOAD still succeeds — the contract publishes, uncalibrated.
        let mut requantized = base.clone();
        requantized.quantize = Some(mlx_gen::Quant::Q4);
        assert!(production_calibration_fingerprint(FULL_ID, &requantized).is_none());
        assert!(
            memory_strategy_contract(FULL_ID, &requantized)
                .unwrap()
                .calibration
                .is_none(),
            "a withheld identity must not fail the load"
        );

        // A manifest that disagrees with its packed headers proves no tier.
        let crossed = tier_snapshot(Some(4));
        write_tier_file(&crossed.path().join("qwen2_5_vl.safetensors"), Some(8));
        let spec = LoadSpec::new(WeightsSource::Dir(crossed.path().to_owned()));
        assert!(production_calibration_fingerprint(FULL_ID, &spec).is_none());

        // An absent snapshot proves no tier either, and asking costs nothing.
        let missing = LoadSpec::new(WeightsSource::Dir(
            root.path().join("no-such-snapshot").to_owned(),
        ));
        assert!(production_calibration_fingerprint(FULL_ID, &missing).is_none());
    }

    /// A tier is read off the ARTIFACT, so the dense root publishes `bf16` however the request is
    /// spelled — and a packed root never publishes the dense key.
    #[test]
    fn the_identity_names_the_artifacts_tier_not_the_requested_one() {
        let dense = tier_snapshot(None);
        let dense_spec = LoadSpec::new(WeightsSource::Dir(dense.path().to_owned()));
        assert_eq!(
            production_calibration_fingerprint(FULL_ID, &dense_spec).unwrap(),
            "bernini-image-bf16-mlx-dual-expert-ladder-v1"
        );
        let packed = tier_snapshot(Some(8));
        let packed_spec = LoadSpec::new(WeightsSource::Dir(packed.path().to_owned()));
        assert_eq!(
            production_calibration_fingerprint(FULL_ID, &packed_spec).unwrap(),
            "bernini-image-q8-mlx-dual-expert-ladder-v1"
        );
    }

    #[test]
    fn provider_owned_tier_resolver_rejects_manifest_header_disagreement() {
        let root = tier_snapshot(Some(4));
        write_tier_file(&root.path().join("qwen2_5_vl.safetensors"), Some(8));
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        assert!(resolved_numeric_tier(&spec).is_err());

        let root = tier_snapshot(Some(4));
        let crossed = serde_json::json!({
            "quantization": { "bits": 8, "group_size": 64 }
        });
        std::fs::write(
            root.path().join("qwen2_5_vl_config.json"),
            crossed.to_string(),
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        assert!(resolved_numeric_tier(&spec).is_err());
    }

    #[test]
    fn r2v_receipt_binds_order_bytes_and_mlx_effective_shapes() {
        let request = r2v_request();
        let receipt = r2v_reference_receipt(FULL_ID, &request).expect("exact receipt");
        assert_eq!(
            receipt,
            "bernini-r2v-references-v2:backend-mlx:source-preprocess-full-vae624-v1:count-2:0:native-640x360;vit-280x168;vae-624x352|1:native-360x640;vit-168x280;vae-352x624+bernini-r2v-request-seal-v1-cd11cf62ec83e85860e1790538062a88b39ae384d2956fd1dc54c0e45d6fa8f5"
        );
        let renderer_receipt =
            r2v_reference_receipt(RENDERER_ID, &request).expect("renderer receipt");
        assert!(renderer_receipt.contains("source-preprocess-renderer-output-v1"));
        assert!(renderer_receipt.contains("vae-848x480"));
        assert_ne!(receipt, renderer_receipt);

        let mut reversed = request.clone();
        let [Conditioning::MultiReference { images }] = reversed.conditioning.as_mut_slice() else {
            unreachable!()
        };
        images.reverse();
        assert_ne!(receipt, r2v_reference_receipt(FULL_ID, &reversed).unwrap());

        let mut duplicate = request;
        let [Conditioning::MultiReference { images }] = duplicate.conditioning.as_mut_slice()
        else {
            unreachable!()
        };
        images[1] = images[0].clone();
        assert!(r2v_reference_receipt(FULL_ID, &duplicate).is_ok());

        let mut crossed = r2v_request();
        crossed.conditioning.clear();
        assert!(r2v_reference_receipt(FULL_ID, &crossed).is_err());
        crossed.conditioning = vec![Conditioning::Reference {
            image: distinct_references(1).into_iter().next().unwrap(),
            strength: Some(1.0),
        }];
        assert!(r2v_reference_receipt(FULL_ID, &crossed).is_err());
    }

    #[test]
    fn r2v_request_scope_covers_every_public_geometry_duration_and_reference_bound() {
        for count in [1, 8] {
            for (width, height) in ADVERTISED_GEOMETRIES {
                for frames in [45, 61, 77] {
                    let mut request = r2v_request();
                    request.width = *width;
                    request.height = *height;
                    request.frames = Some(frames);
                    request.conditioning = vec![Conditioning::MultiReference {
                        images: distinct_references(count),
                    }];
                    validate_reference_to_video_request(FULL_ID, &request).unwrap_or_else(
                        |error| panic!("{count}/{width}x{height}/{frames}: {error}"),
                    );
                }
            }
        }
    }

    #[test]
    fn rv2v_binds_the_composite_packed_surface_and_cannot_borrow_r2v_evidence() {
        let request = rv2v_request();
        for quant in [None, Some(mlx_gen::Quant::Q4), Some(mlx_gen::Quant::Q8)] {
            let mut tier_spec = spec(LoadShape::DeferredMaterialization);
            tier_spec.quantize = quant;
            for provider_id in PROVIDER_IDS {
                let receipt =
                    r2v_reference_receipt(provider_id, &request).expect("exact RV2V receipt");
                let (total, image_tokens, preprocess) = if provider_id == RENDERER_ID {
                    (22_260, 1_590, "source-preprocess-renderer-output-v1")
                } else {
                    (12_012, 858, "source-preprocess-full-vae624-v1")
                };
                assert!(receipt.contains(&format!(
                    "{preprocess}:count-2:packed-source-tokens-{total}:"
                )));
                assert!(receipt.contains(&format!(";tokens-{image_tokens}")));
                let r2v_receipt = r2v_reference_receipt(provider_id, &r2v_request()).unwrap();
                assert_ne!(receipt, r2v_receipt);
                assert!(!r2v_receipt.contains("video-1"));
                assert!(!r2v_receipt.contains("packed-source-tokens"));
                let contract =
                    weights_free_memory_strategy_contract(provider_id, &tier_spec).unwrap();
                for strategy in [
                    MemoryStrategy::BoundedDecode,
                    MemoryStrategy::BoundedAttention,
                    MemoryStrategy::BoundedTransformerResidency,
                ] {
                    let mut context = registered_valid_fixture(&tier_spec, &contract, strategy)
                        .unwrap()
                        .remove(1)
                        .context;
                    context.mode = MemoryMode::Other("reference_video_to_video".to_owned());
                    context.geometry.reference_count = 3;
                    context.overlay = Some(format!("provider_video_mode:rv2v+{receipt}"));
                    assert_eq!(
                        safety_check(&tier_spec, &contract, &context),
                        MemorySafetyDecision::Accept,
                        "{provider_id}/{quant:?}/{strategy:?}"
                    );

                    let crossed_provider = if provider_id == RENDERER_ID {
                        FULL_ID
                    } else {
                        RENDERER_ID
                    };
                    let mut crossed_provider_context = context.clone();
                    crossed_provider_context.overlay = Some(format!(
                        "provider_video_mode:rv2v+{}",
                        r2v_reference_receipt(crossed_provider, &request).unwrap()
                    ));
                    assert!(matches!(
                        safety_check(&tier_spec, &contract, &crossed_provider_context),
                        MemorySafetyDecision::Reject { .. }
                    ));

                    let mut crossed = context.clone();
                    crossed.geometry.reference_count = 2;
                    assert!(matches!(
                        safety_check(&tier_spec, &contract, &crossed),
                        MemorySafetyDecision::Reject { .. }
                    ));
                    crossed = context.clone();
                    crossed.geometry.width = 1280;
                    crossed.geometry.height = 720;
                    assert!(matches!(
                        safety_check(&tier_spec, &contract, &crossed),
                        MemorySafetyDecision::Reject { .. }
                    ));
                    crossed = context.clone();
                    crossed.overlay = crossed.overlay.take().map(|overlay| {
                        overlay.replace(
                            &format!("packed-source-tokens-{total}"),
                            &format!("packed-source-tokens-{}", total + 1),
                        )
                    });
                    assert!(matches!(
                        safety_check(&tier_spec, &contract, &crossed),
                        MemorySafetyDecision::Reject { .. }
                    ));
                    crossed = context;
                    crossed.mode = MemoryMode::Other("reference_to_video".to_owned());
                    assert!(matches!(
                        safety_check(&tier_spec, &contract, &crossed),
                        MemorySafetyDecision::Reject { .. }
                    ));
                }
            }
        }

        let spec = spec(LoadShape::DeferredMaterialization);
        let contract = contract(FULL_ID, LoadShape::DeferredMaterialization);
        let mut context = registered_valid_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(1)
        .context;
        context.mode = MemoryMode::Other("reference_video_to_video".to_owned());
        context.geometry.reference_count = 3;
        let receipt = r2v_reference_receipt(FULL_ID, &request).unwrap();
        context.overlay = Some(format!("provider_video_mode:rv2v+{receipt}"));
        let mut scope = registered_begin_request(FULL_ID, &spec, &contract, &context)
            .unwrap()
            .expect("RV2V request scope");
        let mut exact = request.clone();
        scope.configure_request(&mut exact).unwrap();
        scope.enter_phase(MemoryPhase::Conditioning).unwrap();
        scope.leave_phase(MemoryPhase::Conditioning).unwrap();
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        let mut scope = registered_begin_request(FULL_ID, &spec, &contract, &context)
            .unwrap()
            .expect("RV2V request scope");
        let mut canceled = request.clone();
        scope.configure_request(&mut canceled).unwrap();
        scope.enter_phase(MemoryPhase::Conditioning).unwrap();
        scope.leave_phase(MemoryPhase::Conditioning).unwrap();
        scope.finish(MemoryRunOutcome::Canceled).unwrap();

        let mut mutated = request;
        let [Conditioning::VideoClip { frames, .. }, Conditioning::MultiReference { .. }] =
            mutated.conditioning.as_mut_slice()
        else {
            unreachable!()
        };
        frames[0].pixels[0] ^= 1;
        let mut scope = registered_begin_request(FULL_ID, &spec, &contract, &context)
            .unwrap()
            .expect("RV2V request scope");
        assert!(scope.configure_request(&mut mutated).is_err());
        scope
            .finish(MemoryRunOutcome::Error {
                message: "crossed clip".to_owned(),
            })
            .unwrap();
    }

    #[test]
    fn every_implemented_mlx_rung_exposes_v2v_r2v_rv2v_and_mv2v_behavior() {
        let spec = spec(LoadShape::DeferredMaterialization);
        for provider_id in PROVIDER_IDS {
            let contract = contract(provider_id, LoadShape::DeferredMaterialization);
            for strategy in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                let fixtures = registered_valid_fixture(&spec, &contract, strategy).unwrap();
                assert_eq!(fixtures.len(), 5, "{provider_id}/{strategy:?}");
                assert_eq!(fixtures[0].context.mode.as_key(), "video_to_video");
                assert_eq!(fixtures[1].context.mode.as_key(), "reference_to_video");
                assert_eq!(fixtures[1].context.geometry.reference_count, 2);
                assert_eq!(
                    fixtures[2].context.mode.as_key(),
                    "reference_video_to_video"
                );
                assert_eq!(fixtures[2].context.geometry.reference_count, 3);
                assert_eq!(fixtures[3].context.mode.as_key(), "multi_video_to_video");
                assert_eq!(fixtures[3].context.geometry.reference_count, 2);
                assert_eq!(
                    safety_check(&spec, &contract, &fixtures[1].context),
                    MemorySafetyDecision::Accept,
                    "{provider_id}/{strategy:?}"
                );
                assert_eq!(
                    safety_check(&spec, &contract, &fixtures[2].context),
                    MemorySafetyDecision::Accept,
                    "{provider_id}/{strategy:?}/rv2v"
                );
                assert_eq!(
                    safety_check(&spec, &contract, &fixtures[3].context),
                    MemorySafetyDecision::Accept,
                    "{provider_id}/{strategy:?}/mv2v"
                );
            }
        }
    }

    #[test]
    fn mv2v_safety_rejects_duplicate_crossed_and_malformed_source_axes_before_scope() {
        let spec = spec(LoadShape::DeferredMaterialization);
        for provider_id in PROVIDER_IDS {
            let contract = contract(provider_id, LoadShape::DeferredMaterialization);
            let fixtures =
                registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode).unwrap();
            let exact = fixtures[3].context.clone();
            assert_eq!(
                safety_check(&spec, &contract, &exact),
                MemorySafetyDecision::Accept,
                "{provider_id}: exact MV2V context"
            );
            let overlay = exact.overlay.clone().unwrap();
            let mv2v_axis = overlay
                .split('+')
                .find(|axis| is_mv2v_receipt_axis(axis))
                .unwrap();
            let mv2v_seal = overlay
                .split('+')
                .find(|axis| axis.starts_with(MV2V_SEAL_DOMAIN))
                .unwrap();
            let r2v_overlay = fixtures[1].context.overlay.as_deref().unwrap();
            let r2v_axis = r2v_overlay
                .split('+')
                .find(|axis| is_reference_receipt_axis(axis))
                .unwrap();
            let r2v_seal = r2v_overlay
                .split('+')
                .find(|axis| axis.starts_with(R2V_REFERENCE_SEAL_DOMAIN))
                .unwrap();

            for invalid_overlay in [
                format!("{overlay}+{mv2v_axis}"),
                format!("{overlay}+{mv2v_seal}"),
                format!("{overlay}+{r2v_axis}+{r2v_seal}"),
                format!(
                    "provider_video_mode:mv2v+{MV2V_RECEIPT_DOMAIN}:backend-mlx:malformed+{MV2V_SEAL_DOMAIN}-bad"
                ),
            ] {
                let mut crossed = exact.clone();
                crossed.overlay = Some(invalid_overlay);
                assert!(matches!(
                    safety_check(&spec, &contract, &crossed),
                    MemorySafetyDecision::Reject { .. }
                ));
            }
        }
    }

    #[test]
    fn r2v_scope_revalidates_references_and_finishes_every_outcome() {
        let spec = spec(LoadShape::DeferredMaterialization);
        let contract = contract(FULL_ID, LoadShape::DeferredMaterialization);
        let fixture = registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .remove(1);
        let mut empty_phases = fixture.request.clone();
        empty_phases.phases = Some(Vec::new());
        assert!(validate_reference_to_video_request(FULL_ID, &empty_phases).is_err());
        for outcome in [
            MemoryRunOutcome::Complete,
            MemoryRunOutcome::Canceled,
            MemoryRunOutcome::Error {
                message: "fixture error".to_owned(),
            },
        ] {
            let mut scope = registered_begin_request(FULL_ID, &spec, &contract, &fixture.context)
                .unwrap()
                .expect("scope");
            let mut request = fixture.request.clone();
            scope
                .configure_request(&mut request)
                .expect("exact request");
            scope.enter_phase(MemoryPhase::Conditioning).unwrap();
            scope.leave_phase(MemoryPhase::Conditioning).unwrap();
            scope.finish(outcome).unwrap();
        }

        let mut scope = registered_begin_request(FULL_ID, &spec, &contract, &fixture.context)
            .unwrap()
            .expect("scope");
        let mut mutated = fixture.request.clone();
        let [Conditioning::MultiReference { images }] = mutated.conditioning.as_mut_slice() else {
            unreachable!()
        };
        images[0].pixels[0] ^= 1;
        assert!(scope.configure_request(&mut mutated).is_err());
        scope
            .finish(MemoryRunOutcome::Error {
                message: "crossed references".to_owned(),
            })
            .unwrap();

        let mut scope = registered_begin_request(FULL_ID, &spec, &contract, &fixture.context)
            .unwrap()
            .expect("scope");
        let mut phased = fixture.request;
        phased.phases = Some(vec![mlx_gen::gen_core::GenerationPhase {
            steps: 1,
            ..Default::default()
        }]);
        assert!(scope.configure_request(&mut phased).is_err());
        scope
            .finish(MemoryRunOutcome::Error {
                message: "crossed phases".to_owned(),
            })
            .unwrap();
    }
}
