//! MLX LTX-2.3 video memory-provider contract (SC-19109).
//!
//! The declaration follows the SC-18813 source survey rather than projecting the image ladder onto
//! video:
//!
//! - rung 1 is the shipped, unconditional Gemma -> AvDiT -> decode phase staging;
//! - rung 2 is the shipped budgeted LTX VAE tiler;
//! - rung 3 is missing because inference attention remains one monolithic SDPA call;
//! - rung 4 is missing because the 48-block AvDiT has no block-window materialization path.
//!
//! `MemoryStrategy::Resident` is gen-core's mandatory protocol baseline: it means "apply no new
//! request controls", not "keep every model component co-resident". LTX's historical baseline
//! already performs rung-1 phase staging, so `ResidentRequestMemory::PreserveLoadDefaults` leaves
//! that physical path unchanged while an explicit `StagedResidency` selection emits
//! `stage_residency=true`. The current contract vocabulary cannot separately say "unconditional"
//! versus "selectable"; SC-18816 owns that descriptor-level tri-state. Until then, resident
//! calibration must measure the already-staged baseline rather than synthesize a co-resident peak.
//!
//! The production factory sizes the exact component files this loader can materialize. The separate
//! weights-free factory exists only for registry conformance and deliberately injects zero asset
//! facts; production registry resolution never consults it.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::{
    self, AdapterKind, AdapterResidencyMode, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategyPrerequisite,
    MemoryStrategySupport, Quant, ResidentRequestMemory, WeightsSource,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, Result};

use crate::config::{LtxConfig, SplitModel};
use crate::gemma::GemmaQuant;

/// Contract/execution identity introduced by SC-19109. The earlier SC-18808 capture predates this
/// shared-contract carrier and therefore cannot be reused as if its calibration semantics matched.
///
/// Since sc-22737 this is the identity of the **q8** cell specifically — the tier this family's
/// retained anchor is filed under — rather than one string shared by all three shipped tiers. See
/// [`production_calibration_fingerprint`]. Retained byte-for-byte so that record keeps its key.
pub const CALIBRATION_FINGERPRINT: &str = "sc-20772-ltx-2-3-mlx-memory-ladder-v2";

/// The tier [`CALIBRATION_FINGERPRINT`] names.
pub const CALIBRATED_TIER: &str = "q8";

/// Artifact-tier label of an LTX-2.3 load: `bf16` dense, `q4`/`q8` prepacked.
pub fn calibration_tier_label(quant: Option<Quant>) -> Option<&'static str> {
    match quant {
        None => Some("bf16"),
        Some(Quant::Q4) => Some("q4"),
        Some(Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// The production calibration identity for one shipped tier.
///
/// ## sc-22737 (epic sc-22723 E1/E4): the key is per TIER
///
/// It used to be one string for the whole route, so a q4 load and a bf16 load published the
/// identity of the q8 cell. That is worse than publishing nothing: an anchor measured on one tier
/// would be matched against loads of the other two, and the ladder for a 20GB checkpoint would be
/// priced from a 47GB one's peaks.
///
/// The tier is the one the ARTIFACT carries. Callers pass the tier
/// `numeric_tier_from_split` resolved, which reads the split checkpoint's own quantization and
/// REFUSES a `spec.quantize` that disagrees with it, so a crossed request never reaches here.
/// `None` for a packed width this family does not ship — withheld, never a failed load.
pub fn production_calibration_fingerprint(tier: &MemoryNumericTier) -> Option<String> {
    let label = calibration_tier_label(tier.quant)?;
    Some(if label == CALIBRATED_TIER {
        CALIBRATION_FINGERPRINT.to_owned()
    } else {
        format!("sc-20772-ltx-2-3-{label}-mlx-memory-ladder-v2")
    })
}
const STATIC_CALIBRATION_FINGERPRINT: &str = "sc-20772-ltx-2-3-mlx-registry-behavior-v2";

pub const DECODE_OVERLAP: u32 = 64;

const SHIPPED_GEOMETRIES: &[(u32, u32)] =
    &[(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)];
const SHIPPED_FRAME_COUNTS: &[u32] = &[
    97, 121, 145, 153, 177, 193, 201, 241, 249, 289, 297, 361, 377, 449,
];
const I2V_STRENGTH_BITS: u32 = 1.0_f32.to_bits();

fn frame_count_matches_fps(fps: u32, frames: u32) -> bool {
    match fps {
        24 => [97, 145, 193, 241, 289, 361].contains(&frames),
        25 => [97, 153, 201, 249, 297, 377].contains(&frames),
        30 => [121, 177, 241, 297, 361, 449].contains(&frames),
        _ => false,
    }
}

pub(crate) fn decode_tile_edges() -> Vec<u32> {
    crate::pipeline::LTX_VAE_SPATIAL_PX
        .iter()
        .map(|&edge| edge as u32)
        .collect()
}

fn checked_sum(label: &str, values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            mlx_gen::Error::Msg(format!(
                "ltx_2_3: {label} safetensors byte total overflows u64"
            ))
        })
    })
}

fn required_projected_safetensors_bytes(
    path: &std::path::Path,
    label: &str,
    projection: ResidentProjection,
) -> Result<u64> {
    let bytes = projected_safetensors_bytes(path, |_| projection).map_err(mlx_gen::Error::from)?;
    if bytes == 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3: {label} has no projected resident safetensors bytes at {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn uncensored_enhancer_resident_bytes(spec: &LoadSpec) -> Result<u64> {
    let Some(source) = spec.components.get("uncensored_enhancer") else {
        return Ok(0);
    };
    let WeightsSource::Dir(path) = source else {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored_enhancer must be a directory source".into(),
        ));
    };
    // The amoral Gemma loader retains packed affine tensors, but `load_embedding` dequantizes the
    // packed token table to dense bf16. MLX keeps the packed source graph alive until that lazy
    // dequantization is evaluated, so the conditioning load peak is the complete stored tensor
    // inventory plus the expanded dense table. Its autoregressive KV work is bounded separately by
    // the request-scope MAX_ENHANCE_TOKENS ceiling.
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .map_err(mlx_gen::Error::from)?;
    let quant = crate::model::resolve_gemma_quant(path)?.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored Gemma must declare affine quantization geometry".into(),
        )
    })?;
    if quant.group <= 0 || !matches!(quant.bits, 4 | 8) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: uncensored Gemma quantization must be q4/q8 with a positive group (got {}-bit group-{})",
            quant.bits, quant.group
        )));
    }
    let embedding_weights = headers
        .iter()
        .filter(|header| header.name.ends_with("embed_tokens.weight"))
        .collect::<Vec<_>>();
    let [weight] = embedding_weights.as_slice() else {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: uncensored Gemma must contain exactly one packed embed_tokens.weight (got {})",
            embedding_weights.len()
        )));
    };
    let base = weight
        .name
        .strip_suffix(".weight")
        .expect("the filtered embedding name has a weight suffix");
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    let scales = headers
        .iter()
        .find(|header| header.name == scales_name)
        .ok_or_else(|| {
            mlx_gen::Error::Unsupported(format!(
                "ltx_2_3: uncensored Gemma packed embedding is missing {scales_name}"
            ))
        })?;
    let biases = headers
        .iter()
        .find(|header| header.name == biases_name)
        .ok_or_else(|| {
            mlx_gen::Error::Unsupported(format!(
                "ltx_2_3: uncensored Gemma packed embedding is missing {biases_name}"
            ))
        })?;
    if weight.dtype != gen_core::weightsmeta::Dtype::U32
        || !matches!(
            scales.dtype,
            gen_core::weightsmeta::Dtype::F16
                | gen_core::weightsmeta::Dtype::BF16
                | gen_core::weightsmeta::Dtype::F32
        )
        || biases.dtype != scales.dtype
        || biases.shape != scales.shape
    {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored Gemma packed embedding has invalid weight/scales/biases dtypes or shapes"
                .to_string(),
        ));
    }
    let [out, groups] = scales.shape.as_slice() else {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored Gemma packed embedding scales must be rank two".to_string(),
        ));
    };
    let group = usize::try_from(quant.group).map_err(|_| {
        mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma group is unrepresentable".into())
    })?;
    let input = groups.checked_mul(group).ok_or_else(|| {
        mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma embedding input width overflows".into())
    })?;
    let packed_bits = input
        .checked_mul(usize::try_from(quant.bits).map_err(|_| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma bit width is unrepresentable".into())
        })?)
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma packed width overflows".into())
        })?;
    if !packed_bits.is_multiple_of(32) || weight.shape.as_slice() != [*out, packed_bits / 32] {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: uncensored Gemma packed embedding geometry disagrees with {}-bit group-{} config",
            quant.bits, quant.group
        )));
    }
    let dense_embedding_bytes = u64::try_from(*out)
        .ok()
        .and_then(|out| {
            u64::try_from(input)
                .ok()
                .and_then(|input| out.checked_mul(input))
        })
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma dense embedding bytes overflow".into())
        })?;
    let stored_bytes = headers.iter().try_fold(0_u64, |total, header| {
        total.checked_add(header.data_bytes).ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma stored bytes overflow".into())
        })
    })?;
    let bytes = stored_bytes
        .checked_add(dense_embedding_bytes)
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma load peak overflows u64".into())
        })?;
    if bytes == 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3: uncensored Gemma enhancer has no projected resident safetensors bytes at {}",
            path.display()
        )));
    }
    Ok(bytes)
}

struct AssetDeclaration {
    facts: MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
}

fn adapters_have_load_exact_additive_accounting(spec: &LoadSpec) -> Result<bool> {
    for adapter in &spec.adapters {
        if adapter.kind == AdapterKind::Lokr {
            return Ok(false);
        }
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)
            .map_err(mlx_gen::Error::from)?;
        let reconstructs_dense_delta = headers.iter().any(|tensor| {
            gen_core::weightsmeta::LOKR_TP_SUFFIXES
                .iter()
                .chain(gen_core::weightsmeta::LOHA_TP_SUFFIXES.iter())
                .any(|suffix| tensor.name.ends_with(suffix))
        });
        if reconstructs_dense_delta {
            return Ok(false);
        }
    }
    Ok(true)
}

fn production_asset_declaration(
    spec: &LoadSpec,
    gemma_dir: &std::path::Path,
) -> Result<AssetDeclaration> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 memory contract requires the split model directory used by the loader".into(),
        ));
    };

    // The conditioning phase includes the canonical dense Gemma snapshot, the LTX connector, the
    // request-scoped VAE encoder used by I2V / ordered first-last keyframes, and the optional uncensored Gemma artifact when that
    // exact overlay is loaded. The encoder is released after its fitted image latents materialize;
    // keeping it in this phase floor accounts its widest instant without carrying it into denoise.
    let conditioning_bytes = checked_sum(
        "conditioning",
        [
            required_projected_safetensors_bytes(
                gemma_dir,
                "Gemma text encoder",
                ResidentProjection::Bfloat16,
            )?,
            required_projected_safetensors_bytes(
                &root.join("connector.safetensors"),
                "connector",
                ResidentProjection::Bfloat16,
            )?,
            required_projected_safetensors_bytes(
                &root.join("vae_encoder.safetensors"),
                "video VAE encoder",
                ResidentProjection::Float32,
            )?,
            uncensored_enhancer_resident_bytes(spec)?,
        ],
    )?;

    // The upsampler runs between the two denoise passes while the AvDiT is live, so it belongs to
    // the denoise-phase base footprint rather than to decode.
    let transformer_bytes = checked_sum(
        "denoise",
        [
            required_projected_safetensors_bytes(
                &root.join("transformer.safetensors"),
                "AudioVideo transformer",
                ResidentProjection::Stored,
            )?,
            required_projected_safetensors_bytes(
                &root.join("upsampler.safetensors"),
                "latent upsampler",
                ResidentProjection::Stored,
            )?,
        ],
    )?;

    // The decoder phase excludes the request-scoped encoder charged above.
    let decoder_bytes = checked_sum(
        "decode",
        [
            required_projected_safetensors_bytes(
                &root.join("vae_decoder.safetensors"),
                "video VAE decoder",
                ResidentProjection::Float32,
            )?,
            required_projected_safetensors_bytes(
                &root.join("audio_vae.safetensors"),
                "audio VAE decoder",
                ResidentProjection::Float32,
            )?,
            required_projected_safetensors_bytes(
                &root.join("vocoder.safetensors"),
                "vocoder",
                ResidentProjection::Float32,
            )?,
        ],
    )?;

    let overlay_bytes = match gen_core::adapter_stack_resident_bytes(
        &spec.adapters,
        AdapterResidencyMode::Additive,
    ) {
        Some(bytes) => bytes,
        None => {
            return Err(mlx_gen::Error::Msg(
                "ltx_2_3: every additive adapter must have a non-zero load-exact safetensors size"
                    .into(),
            ));
        }
    };
    let resident_components = (overlay_bytes > 0)
        .then(|| MemoryResidentComponent {
            id: "adapter_stack".to_owned(),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        })
        .into_iter()
        .collect();

    let base_bytes = checked_sum(
        "base model",
        [conditioning_bytes, transformer_bytes, decoder_bytes],
    )?;
    Ok(AssetDeclaration {
        facts: MemoryAssetFacts {
            base_bytes,
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
        resident_components,
    })
}

fn quant_from_split(split: &SplitModel) -> Result<Option<Quant>> {
    if !split.quantized {
        return Ok(None);
    }
    match split.bits {
        4 => Ok(Some(Quant::Q4)),
        8 => Ok(Some(Quant::Q8)),
        bits => Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: split_model.json declares unsupported {bits}-bit transformer weights"
        ))),
    }
}

pub(crate) fn numeric_tier_from_split(
    spec: &LoadSpec,
    split: &SplitModel,
) -> Result<MemoryNumericTier> {
    let quant = quant_from_split(split)?;
    if let Some(requested) = spec.quantize {
        if quant != Some(requested) {
            return Err(mlx_gen::Error::Unsupported(format!(
                "ltx_2_3: requested {requested:?} does not match the checkpoint tier {quant:?}"
            )));
        }
    }
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> Result<MemoryNumericTier> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 numeric tier requires a split model directory".into(),
        ));
    };
    numeric_tier_from_split(spec, &SplitModel::from_model_dir(root)?)
}

/// Canonical request overlay spelling shared with the SceneWorks admission identity.
///
/// Adapter count is part of the identity (the auto-distill adapter is therefore not allowed to
/// borrow a plain-T2V cell), enhancer is a load-bearing asset axis, and the provider mode is a
/// request-only axis.  The latter is appended by SceneWorks because `LoadSpec` does not carry the
/// per-request `video_mode` field.
pub(crate) fn canonical_overlay(
    adapter_count: usize,
    enhancer: Option<&str>,
    video_mode: Option<&str>,
) -> Option<String> {
    let mut axes = Vec::new();
    if adapter_count > 0 {
        axes.push(format!("adapters:{adapter_count}"));
    }
    if let Some(enhancer) = enhancer {
        axes.push(format!("enhancer:{enhancer}"));
    }
    if let Some(video_mode) = video_mode {
        axes.push(format!("provider_video_mode:{video_mode}"));
    }
    (!axes.is_empty()).then(|| axes.join("+"))
}

pub(crate) fn route_overlay(spec: &LoadSpec) -> Option<String> {
    canonical_overlay(
        spec.adapters.len(),
        spec.components
            .contains_key("uncensored_enhancer")
            .then_some("uncensored"),
        None,
    )
}

/// Compare a request overlay to the loaded route without dropping request-only axes.
///
/// The provider contract is created from `LoadSpec`, so it can only know the loaded adapter and
/// enhancer axes.  `provider_video_mode:*` is carried by the request identity and is accepted only
/// as an additional, explicitly named axis.  Unknown or missing load axes still fail closed.
fn overlay_matches_loaded_route(actual: Option<&str>, expected: Option<&str>) -> bool {
    let load_axes = |overlay: Option<&str>| {
        overlay
            .into_iter()
            .flat_map(|value| value.split('+'))
            .filter(|axis| {
                !axis.starts_with("provider_video_mode:")
                    && !axis.starts_with("reference:image:")
                    && !axis.starts_with("reference:sheet:")
                    && !axis.starts_with("keyframe:")
                    && !axis.starts_with("clip:append:")
                    && !axis.starts_with("clip:replace:")
                    && axis != &"enhancer:standard"
            })
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let actual_axes = actual.into_iter().flat_map(|value| value.split('+'));
    if actual_axes.clone().any(|axis| {
        axis.starts_with("provider_video_mode:") && !matches!(axis, "provider_video_mode:no_audio")
    }) {
        return false;
    }
    load_axes(actual) == load_axes(expected)
}

fn reference_axis(width: u32, height: u32) -> String {
    format!("reference:image:{width}x{height}:strength:{I2V_STRENGTH_BITS:08x}")
}

fn first_last_axes(width: u32, height: u32) -> [String; 2] {
    [
        format!("keyframe:first:image:{width}x{height}:frame:0:strength:{I2V_STRENGTH_BITS:08x}"),
        format!("keyframe:last:image:{width}x{height}:frame:-1:strength:{I2V_STRENGTH_BITS:08x}"),
    ]
}

fn extend_clip_axis(frames: u32, width: u32, height: u32) -> String {
    format!("clip:append:frames:{frames}:image:{width}x{height}:frame:0:strength:{I2V_STRENGTH_BITS:08x}")
}
fn bridge_clip_axes(frames: u32, width: u32, height: u32) -> [String; 2] {
    [extend_clip_axis(frames, width, height), format!("clip:append:frames:{frames}:image:{width}x{height}:frame:-1:strength:{I2V_STRENGTH_BITS:08x}")]
}

/// The ordered replace-person receipt: the masked control clip, then the character contact sheet.
///
/// Byte-identical to the Candle sibling in `candle-gen-ltx`. Two axes rather than one because the
/// carriers are independently variable — the clip's mask mode and blend weight move without
/// changing the reference count, and a fifth reference changes the composite grid without touching
/// the clip. The reference axis names the **composite** the provider actually encodes (see
/// [`crate::conditioning::compose_ordered_character_references`]), so 1–4 sources at any input size
/// collapse to one target-sized image latent — the count is what changes the grid, not the sizes.
fn replace_person_axes(
    frames: u32,
    width: u32,
    height: u32,
    references: usize,
    masking_strength: f32,
    mode: gen_core::ReplacementMode,
) -> [String; 2] {
    [
        format!(
            "clip:replace:frames:{frames}:image:{width}x{height}:frame:0:mode:{}:mask:{:08x}",
            mode as u8,
            masking_strength.to_bits()
        ),
        format!("reference:sheet:{references}:image:{width}x{height}"),
    ]
}

/// Pull the exact LTX replace-person carrier out of a request: one masked `ControlClip` plus one
/// ordered 1–4 image `MultiReference`, in either order (the generator accepts both).
fn replace_person_receipt(request: &GenerationRequest) -> Option<String> {
    let (clip, images) = match request.conditioning.as_slice() {
        [gen_core::Conditioning::ControlClip { .. }, gen_core::Conditioning::MultiReference { images }]
        | [gen_core::Conditioning::MultiReference { images }, gen_core::Conditioning::ControlClip { .. }] => {
            (request.control_clip()?, images)
        }
        _ => return None,
    };
    let frames = request.frames.unwrap_or(0);
    let fitted =
        |image: &gen_core::Image| image.width == request.width && image.height == request.height;
    if clip.frames.len() != frames as usize
        || clip.mask.len() != clip.frames.len()
        || clip.frames.is_empty()
        || clip.start_frame != 0
        || !clip.masking_strength.is_finite()
        || !(0.0..=1.0).contains(&clip.masking_strength)
        || !clip.frames.iter().all(fitted)
        || !clip.mask.iter().all(fitted)
        || !(1..=4).contains(&images.len())
    {
        return None;
    }
    // The composite is the actual VAE input; refusing here keeps a degenerate reference (zero-sized
    // or short buffer) from being admitted and then failing mid-render.
    crate::conditioning::compose_ordered_character_references(
        images,
        request.width,
        request.height,
    )
    .ok()?;
    Some(
        replace_person_axes(
            frames,
            request.width,
            request.height,
            images.len(),
            clip.masking_strength,
            clip.mode,
        )
        .join("+"),
    )
}

/// The replace-person axes an overlay actually carries, in receipt order.
fn admitted_replace_person_axes(overlay: Option<&str>) -> String {
    overlay
        .into_iter()
        .flat_map(|value| value.split('+'))
        .filter(|axis| axis.starts_with("clip:replace:") || axis.starts_with("reference:sheet:"))
        .collect::<Vec<_>>()
        .join("+")
}

/// Does `axes` spell exactly the two replace-person axes this `geometry` can carry?
///
/// The mask blend weight and the [`gen_core::ReplacementMode`] have no geometry counterpart, so
/// they are checked structurally rather than reconstructed. Everything the geometry *does* fix —
/// frame count, output size, clip start and reference cardinality — is compared exactly, and the
/// request scope then compares the whole receipt string byte-for-byte.
fn replace_person_axes_shape_ok(axes: &str, geometry: gen_core::MemoryGeometry) -> bool {
    let axes = axes.split('+').collect::<Vec<_>>();
    let [clip, sheet] = axes.as_slice() else {
        return false;
    };
    if *sheet
        != format!(
            "reference:sheet:{}:image:{}x{}",
            geometry.reference_count, geometry.width, geometry.height
        )
    {
        return false;
    }
    let Some(tail) = clip.strip_prefix(&format!(
        "clip:replace:frames:{}:image:{}x{}:frame:0:mode:",
        geometry.frames, geometry.width, geometry.height
    )) else {
        return false;
    };
    let Some((mode, mask)) = tail.split_once(":mask:") else {
        return false;
    };
    if !matches!(mode, "0" | "1" | "2")
        || mask.len() != 8
        || !mask
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return false;
    }
    u32::from_str_radix(mask, 16)
        .ok()
        .map(f32::from_bits)
        .is_some_and(|weight| weight.is_finite() && (0.0..=1.0).contains(&weight))
}

fn admitted_reference_axis(overlay: Option<&str>) -> Option<&str> {
    overlay
        .into_iter()
        .flat_map(|value| value.split('+'))
        .find(|axis| axis.starts_with("reference:image:"))
}

fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                    MemoryStrategySupport::Missing
                }
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: decode_tile_edges(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

/// Architecture axes shared by both registered LTX-2 routes (epic SC-22657, E2).
///
/// [`LtxConfig::video_only_defaults`](crate::config::LtxConfig::video_only_defaults) mirrors the
/// `embedded_config.json` transformer block the loader parses on every load, and the measured 2.3 to
/// 2.5 delta is two booleans — not a dimension — so both routes publish one set of axes.
///
/// `patch_size` is declared **absent**: LTX's AvDiT has no patchify at all. `patchify_proj` is a
/// plain `Linear` over the 128-channel latent token, and every patch factor lives inside the VAE,
/// already folded into `vae_spatial_scale`. A structurally absent axis is declared `None`, never a
/// stand-in 1.
///
/// The VAE is a **video** autoencoder, so `vae_temporal_scale` is a real value here: eight frames
/// per latent unit over a x32 spatial scale, which `diff_vae::VIDEO_SCALE_FACTORS` restates.
///
/// When `spec` names a materialized snapshot directory this re-runs `LtxConfig::from_model_dir` —
/// the 2.3 loader's own `embedded_config.json` parse — so the published trunk axes are the
/// snapshot's rather than the preset's. On the weights-free surface there is nothing to read, and
/// the preset is what `from_model_dir` itself falls back to. The 2.5 route parses a different
/// source (its split bundle's transformer `__metadata__`) and goes through
/// [`architecture_facts_for`] with that config instead.
///
/// The three snapshot outcomes are kept apart because the loader keeps them apart (SC-22667):
///
/// * **absent** `embedded_config.json` — `from_model_dir` returns the preset itself
///   (`config.rs`'s early `Ok(Self::video_only_defaults())`), and `model.rs`'s `?` accepts it, so
///   the preset is the geometry the loader will build and publishing it is honest;
/// * **present and parseable** — the snapshot's own axes, as before;
/// * **present and unparseable** (malformed JSON, no `transformer` section, an unimplemented
///   `rope_type`) — `model.rs`'s `?` *errors*, so no load of this snapshot has the preset's
///   geometry. Falling back to the preset would publish 48 blocks x 3072 channels for a tree the
///   engine refuses to open. The trunk axes are declared **absent** instead; the VAE axes are
///   crate constants that no snapshot config can move, so they stand and the weights-free gate
///   still sees a declared axis.
pub(crate) fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let Some(root) = mlx_gen::architecture_facts::materialized_root(spec) else {
        return architecture_facts_for(&crate::config::LtxConfig::video_only_defaults(), spec);
    };
    match crate::config::LtxConfig::from_model_dir(root) {
        Ok(dit) => architecture_facts_for(&dit, spec),
        Err(_) => unreadable_trunk_architecture_facts(spec),
    }
}

/// Axes for a materialized tree whose own transformer config the loader cannot parse.
///
/// Every trunk axis comes from that config, so none of them is knowable — and E2 declares an
/// unknowable axis absent rather than substituting a preset the loader would never build. The VAE
/// axes and the activation width are not config-derived (crate constants and [`LoadSpec`]'s own
/// precision), so they remain declared.
pub(crate) fn unreadable_trunk_architecture_facts(
    spec: &LoadSpec,
) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: None,
        head_dim: None,
        transformer_blocks: None,
        patch_size: None,
        latent_channels: mlx_gen::architecture_facts::axis(crate::model::LATENT_CHANNELS),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::model::SPATIAL_SCALE),
        vae_temporal_scale: mlx_gen::architecture_facts::axis(crate::model::TEMPORAL_SCALE),
        activation_dtype_width: Some(activation_dtype_width(spec)),
    }
}

/// The shared LTX axis derivation over an already-resolved transformer config: the trunk axes off
/// `dit`, the video-VAE axes off this crate's constants, and the activation width off the precision
/// the loader will run `spec` at (SC-22667).
///
/// Both loaders select the DiT precision from `spec.precision` — `Bf16` (the `LoadSpec` default)
/// computes bf16 activations over quantized weights, `Fp32` computes f32 activations over the same
/// weights (`model.rs`, the quality target) — so the width is derived from that field rather than
/// pinned to the half-precision literal, which described only the default path.
pub(crate) fn architecture_facts_for(
    dit: &crate::config::LtxConfig,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: None,
        latent_channels: mlx_gen::architecture_facts::axis(crate::model::LATENT_CHANNELS),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::model::SPATIAL_SCALE),
        vae_temporal_scale: mlx_gen::architecture_facts::axis(crate::model::TEMPORAL_SCALE),
        activation_dtype_width: Some(activation_dtype_width(spec)),
    }
}

/// Bytes per activation element for the precision `model.rs` selects from `spec.precision`.
pub(crate) fn activation_dtype_width(spec: &LoadSpec) -> u32 {
    match spec.precision {
        mlx_gen::Precision::Bf16 => mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH,
        mlx_gen::Precision::Fp32 => mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH,
    }
}

fn build_contract(
    spec: &LoadSpec,
    asset_declaration: AssetDeclaration,
    calibration_fingerprint: &str,
) -> Result<MemoryProviderContract> {
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3 has no deferred/block-window loader; use EagerMaterialization".into(),
        ));
    }
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::FrameCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
    ];
    let formula = if asset_declaration.resident_components.is_empty() {
        MemoryFormulaKind::PhaseEnvelope {
            phases: phases.clone(),
            variables,
        }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: phases.clone(),
            variables,
            resident_components: asset_declaration.resident_components,
        }
    };
    Ok(MemoryProviderContract {
        phase_facts: None,
        architecture_facts: architecture_facts(spec),
        provider_id: crate::MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: strategies(),
        // LTX declares no decode-quality geometry policy table, so this route carries no semantic
        // decode authority — the fail-closed default every non-declaring provider contract uses.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        // LTX stages Gemma before the AvDiT for every render. A selected decode rung therefore
        // co-engages rung 1 even though the shared cost-order default intentionally does not.
        additional_prerequisites: vec![(
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        )],
        default_engagement_exclusions: Vec::new(),
        // Gen-core requires Resident as the no-new-control baseline. It does not assert literal
        // co-residency: preserving defaults keeps LTX's historical always-staged phase order and
        // automatic decode guard. SC-18816 will make that unconditional staging separately visible.
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula,
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts: asset_declaration.facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

pub(crate) fn contract_for_loaded(
    spec: &LoadSpec,
    split: &SplitModel,
    gemma_dir: &std::path::Path,
    gemma_quant: Option<GemmaQuant>,
) -> Result<Option<(MemoryProviderContract, MemoryNumericTier, Option<String>)>> {
    // The first production evidence campaign is calibrated against the canonical dense-bf16 Gemma
    // route. Quantized Gemma remains a supported generator input, but must fail open until it has a
    // separately identifiable and measured contract rather than borrowing the canonical evidence.
    if gemma_quant.is_some() || !adapters_have_load_exact_additive_accounting(spec)? {
        return Ok(None);
    }
    let tier = numeric_tier_from_split(spec, split)?;
    // sc-22737: the identity names the tier the checkpoint carries, not the route.
    let Some(fingerprint) = production_calibration_fingerprint(&tier) else {
        return Ok(None);
    };
    let contract = build_contract(
        spec,
        production_asset_declaration(spec, gemma_dir)?,
        &fingerprint,
    )?;
    Ok(Some((contract, tier, route_overlay(spec))))
}

pub fn memory_strategy_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 memory contract requires a split model directory".into(),
        ));
    };
    let split = SplitModel::from_model_dir(root)?;
    let tier = numeric_tier_from_split(spec, &split)?;
    let gemma_dir = crate::model::resolve_gemma_dir(spec.text_encoder.as_ref())?;
    if let Some(quant) = crate::model::resolve_gemma_quant(&gemma_dir)? {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: the calibrated memory contract requires canonical dense-bf16 Gemma; \
             the requested Gemma snapshot declares {}-bit group-{} quantization",
            quant.bits, quant.group
        )));
    }
    if !adapters_have_load_exact_additive_accounting(spec)? {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: calibrated memory admission supports additive LoRA factors but not \
             LoKr/LoHa routes that reconstruct dense bf16 deltas"
                .into(),
        ));
    }
    let fingerprint = production_calibration_fingerprint(&tier).ok_or_else(|| {
        mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: the checkpoint declares an unshipped packed tier {:?}",
            tier.quant
        ))
    })?;
    build_contract(
        spec,
        production_asset_declaration(spec, &gemma_dir)?,
        &fingerprint,
    )
}

pub(crate) fn weights_free_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        AssetDeclaration {
            facts: MemoryAssetFacts::default(),
            resident_components: Vec::new(),
        },
        STATIC_CALIBRATION_FINGERPRINT,
    )
    .map_err(Into::into)
}

/// LTX witnesses the shared MLX tiers under both offload policies, but only the eager half of the
/// materialization axis. `build_contract` rejects `DeferredMaterialization` outright — LTX has no
/// deferred/block-window loader — so publishing the deferred selectors would advertise a load
/// surface no contract can be built for, and the registry conformance walk (which constructs every
/// published selector) would fail the whole MLX catalog. The witness set is the provider's own
/// finite inventory, not the shared default.
pub(crate) fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

fn fixture_contract(contract: &MemoryProviderContract) -> bool {
    contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == STATIC_CALIBRATION_FINGERPRINT)
}

fn registered_tier(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> Result<MemoryNumericTier> {
    if fixture_contract(contract) {
        return Ok(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        });
    }
    resolved_numeric_tier(spec)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported("ltx_2_3: bounded decode requires a tile edge".into())
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported("ltx_2_3: bounded decode requires a tile overlap".into())
    })?;
    let edges = decode_tile_edges();
    if !edges.contains(&edge) {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_3: decode tile edge {edge} is outside the production domain {edges:?}"
        )));
    }
    if overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_3: decode overlap {overlap} is not the production overlap {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let t2v = context.mode.as_key() == "text_to_video"
            && context.geometry.reference_count == 0
            && !context.has_reference;
        let i2v = context.mode.as_key() == "image_to_video"
            && context.geometry.reference_count == 1
            && context.has_reference;
        let first_last = context.mode.as_key() == "first_last_frame"
            && context.geometry.reference_count == 2
            && context.has_reference;
        let extend = context.mode.as_key() == "extend_clip"
            && context.geometry.reference_count == 0
            && !context.has_reference;
        let bridge = context.mode.as_key() == "video_bridge"
            && context.geometry.reference_count == 0
            && !context.has_reference;
        let replace_person = context.mode.as_key() == "replace_person"
            && (1..=4).contains(&context.geometry.reference_count)
            && context.has_reference;
        if (!t2v && !i2v && !first_last && !extend && !bridge && !replace_person)
            || context.use_pid
            || context.has_phases
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: memory admission requires exact text_to_video/no-reference, image_to_video/one-reference, first_last_frame/two-keyframe, extend_clip or video_bridge IC-LoRA clip, or replace_person/masked-clip + 1-4 ordered references identity without PiD/phases"
                    .into(),
            ));
        }
        if !overlay_matches_loaded_route(context.overlay.as_deref(), expected_overlay) {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: memory overlay {:?} does not match the loaded route {:?}",
                context.overlay, expected_overlay
            )));
        }
        let geometry = context.geometry;
        if geometry.batch != 1
            || !SHIPPED_GEOMETRIES.contains(&(geometry.width, geometry.height))
            || !SHIPPED_FRAME_COUNTS.contains(&geometry.frames)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: unsupported memory geometry {}x{}x{} frames={}",
                geometry.width, geometry.height, geometry.batch, geometry.frames
            )));
        }
        let reference = admitted_reference_axis(context.overlay.as_deref());
        if t2v && reference.is_some() {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: text_to_video admission cannot carry an image reference receipt".into(),
            ));
        }
        if i2v && reference != Some(reference_axis(geometry.width, geometry.height).as_str()) {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: image_to_video admission requires the exact fitted image shape and fixed strength receipt"
                    .into(),
            ));
        }
        if first_last {
            let expected = first_last_axes(geometry.width, geometry.height).join("+");
            let actual = context
                .overlay
                .as_deref()
                .into_iter()
                .flat_map(|value| value.split('+'))
                .filter(|axis| axis.starts_with("keyframe:"))
                .collect::<Vec<_>>()
                .join("+");
            if actual != expected {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: first_last_frame admission requires the exact ordered fitted first/last keyframe receipt"
                        .into(),
                ));
            }
        }
        if extend
            && context
                .overlay
                .as_deref()
                .into_iter()
                .flat_map(|v| v.split('+'))
                .filter(|a| a.starts_with("clip:append:"))
                .collect::<Vec<_>>()
                .join("+")
                != extend_clip_axis(geometry.frames, geometry.width, geometry.height)
        {
            return Err(gen_core::Error::Unsupported("ltx_2_3: extend_clip admission requires the exact single IC-LoRA appended-clip receipt".into()));
        }
        if bridge {
            let actual = context
                .overlay
                .as_deref()
                .into_iter()
                .flat_map(|v| v.split('+'))
                .filter(|a| a.starts_with("clip:append:"))
                .collect::<Vec<_>>()
                .join("+");
            if actual
                != bridge_clip_axes(geometry.frames, geometry.width, geometry.height).join("+")
            {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: video_bridge admission requires exact ordered IC-LoRA clip endpoints"
                        .into(),
                ));
            }
        }
        if replace_person
            && !replace_person_axes_shape_ok(
                &admitted_replace_person_axes(context.overlay.as_deref()),
                geometry,
            )
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: replace_person admission requires the exact masked-clip + ordered character-sheet receipt"
                    .into(),
            ));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier),
        Some(&route_gate),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_tier(spec, contract) {
        Ok(tier) => safety_check(contract, tier, route_overlay(spec).as_deref(), context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let tier = registered_tier(spec, contract).map_err(gen_core::Error::from)?;
    let mut t2v_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("text_to_video".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: route_overlay(spec),
        },
    )?;
    t2v_context.geometry.width = 768;
    t2v_context.geometry.height = 512;
    t2v_context.geometry.frames = 153;
    let mut t2v = MemoryBehaviorFixture::new(t2v_context);
    t2v.request.width = 768;
    t2v.request.height = 512;
    t2v.request.frames = Some(153);
    t2v.request.fps = Some(25);

    let reference_receipt = reference_axis(768, 512);
    let i2v_overlay = match route_overlay(spec) {
        Some(load) => format!("{load}+{reference_receipt}"),
        None => reference_receipt,
    };
    let mut i2v_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("image_to_video".into()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some(i2v_overlay),
        },
    )?;
    i2v_context.geometry.width = 768;
    i2v_context.geometry.height = 512;
    i2v_context.geometry.frames = 153;
    let mut i2v = MemoryBehaviorFixture::new(i2v_context);
    i2v.request.width = 768;
    i2v.request.height = 512;
    i2v.request.frames = Some(153);
    i2v.request.fps = Some(25);
    i2v.request.conditioning = vec![gen_core::Conditioning::Reference {
        image: gen_core::Image {
            width: 768,
            height: 512,
            pixels: vec![0; 768 * 512 * 3],
        },
        strength: Some(1.0),
    }];
    let first_last_receipt = first_last_axes(768, 512).join("+");
    let first_last_overlay = match route_overlay(spec) {
        Some(load) => format!("{load}+{first_last_receipt}"),
        None => first_last_receipt,
    };
    let mut first_last_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("first_last_frame".into()),
            reference_count: 2,
            use_pid: false,
            has_phases: false,
            overlay: Some(first_last_overlay),
        },
    )?;
    first_last_context.geometry.width = 768;
    first_last_context.geometry.height = 512;
    first_last_context.geometry.frames = 153;
    let mut first_last = MemoryBehaviorFixture::new(first_last_context);
    first_last.request.width = 768;
    first_last.request.height = 512;
    first_last.request.frames = Some(153);
    first_last.request.fps = Some(25);
    first_last.request.conditioning = vec![
        gen_core::Conditioning::Keyframe {
            image: gen_core::Image {
                width: 768,
                height: 512,
                pixels: vec![0; 768 * 512 * 3],
            },
            frame_idx: 0,
            strength: 1.0,
        },
        gen_core::Conditioning::Keyframe {
            image: gen_core::Image {
                width: 768,
                height: 512,
                pixels: vec![0; 768 * 512 * 3],
            },
            frame_idx: -1,
            strength: 1.0,
        },
    ];
    let mut extend_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("extend_clip".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: Some(extend_clip_axis(153, 768, 512)),
        },
    )?;
    extend_context.geometry.width = 768;
    extend_context.geometry.height = 512;
    extend_context.geometry.frames = 153;
    let mut extend = MemoryBehaviorFixture::new(extend_context);
    extend.request.width = 768;
    extend.request.height = 512;
    extend.request.frames = Some(153);
    extend.request.fps = Some(25);
    extend.request.conditioning = vec![gen_core::Conditioning::VideoClip {
        frames: vec![
            gen_core::Image {
                width: 768,
                height: 512,
                pixels: vec![0; 768 * 512 * 3]
            };
            153
        ],
        frame_idx: 0,
        strength: 1.0,
    }];
    let mut bridge_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("video_bridge".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: Some(bridge_clip_axes(153, 768, 512).join("+")),
        },
    )?;
    bridge_context.geometry.width = 768;
    bridge_context.geometry.height = 512;
    bridge_context.geometry.frames = 153;
    let mut bridge = MemoryBehaviorFixture::new(bridge_context);
    bridge.request.width = 768;
    bridge.request.height = 512;
    bridge.request.frames = Some(153);
    bridge.request.fps = Some(25);
    let clip = vec![
        gen_core::Image {
            width: 768,
            height: 512,
            pixels: vec![0; 768 * 512 * 3]
        };
        153
    ];
    bridge.request.conditioning = vec![
        gen_core::Conditioning::VideoClip {
            frames: clip.clone(),
            frame_idx: 0,
            strength: 1.0,
        },
        gen_core::Conditioning::VideoClip {
            frames: clip,
            frame_idx: -1,
            strength: 1.0,
        },
    ];
    let replace_references = 2_usize;
    let replace_mask = 1.0_f32;
    let replace_mode = gen_core::ReplacementMode::FullPersonKeepOutfit;
    let replace_receipt = replace_person_axes(
        153,
        768,
        512,
        replace_references,
        replace_mask,
        replace_mode,
    )
    .join("+");
    let replace_overlay = match route_overlay(spec) {
        Some(load) => format!("{load}+{replace_receipt}"),
        None => replace_receipt,
    };
    let mut replace_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("replace_person".into()),
            reference_count: replace_references as u32,
            use_pid: false,
            has_phases: false,
            overlay: Some(replace_overlay),
        },
    )?;
    replace_context.geometry.width = 768;
    replace_context.geometry.height = 512;
    replace_context.geometry.frames = 153;
    let mut replace_person = MemoryBehaviorFixture::new(replace_context);
    replace_person.request.width = 768;
    replace_person.request.height = 512;
    replace_person.request.frames = Some(153);
    replace_person.request.fps = Some(25);
    let plate = gen_core::Image {
        width: 768,
        height: 512,
        pixels: vec![0; 768 * 512 * 3],
    };
    replace_person.request.conditioning = vec![
        gen_core::Conditioning::ControlClip {
            frames: vec![plate.clone(); 153],
            mask: vec![plate.clone(); 153],
            masking_strength: replace_mask,
            start_frame: 0,
            mode: replace_mode,
        },
        gen_core::Conditioning::MultiReference {
            images: vec![plate; replace_references],
        },
    ];
    Ok(vec![t2v, i2v, first_last, extend, bridge, replace_person])
}

fn begin_with_cleanup(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    transformer_blocks: usize,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, loaded_tier, expected_overlay, context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        crate::MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        transformer_blocks,
        |_use_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    Ok(Some(Box::new(LtxMemoryRequestScope {
        inner: mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
        expected_overlay: expected_overlay.map(str::to_owned),
        admitted_overlay: context.overlay.clone(),
        admitted_mode: context.mode.as_key().to_owned(),
    })))
}

/// LTX's calibrated evidence is deliberately narrower than the generator's complete capability
/// surface. Bind the admitted context to the actual request before the shared core installs any
/// selected controls, including axes that [`GenerationRequest::image_reference_count`] does not
/// represent (temporal conditioning and prompt enhancement).
struct LtxMemoryRequestScope {
    inner: mlx_gen::request_scope::MlxRequestScopeCore,
    expected_overlay: Option<String>,
    admitted_overlay: Option<String>,
    admitted_mode: String,
}

impl LtxMemoryRequestScope {
    fn validate_request(
        request: &GenerationRequest,
        expected_overlay: Option<&str>,
        admitted_overlay: Option<&str>,
        admitted_mode: &str,
    ) -> gen_core::Result<()> {
        let fitted = |image: &gen_core::Image, strength: f32| {
            image.width == request.width
                && image.height == request.height
                && strength.to_bits() == I2V_STRENGTH_BITS
        };
        let reference = match request.conditioning.as_slice() {
            [gen_core::Conditioning::Reference { image, strength }]
                if fitted(image, strength.or(request.strength).unwrap_or(1.0)) =>
            {
                Some(reference_axis(image.width, image.height))
            }
            _ => None,
        };
        let first_last = match request.conditioning.as_slice() {
            [gen_core::Conditioning::Keyframe {
                image: first,
                frame_idx: 0,
                strength: first_strength,
            }, gen_core::Conditioning::Keyframe {
                image: last,
                frame_idx: -1,
                strength: last_strength,
            }] if fitted(first, *first_strength) && fitted(last, *last_strength) => {
                Some(first_last_axes(first.width, first.height).join("+"))
            }
            _ => None,
        };
        let extend = match request.conditioning.as_slice() {
            [gen_core::Conditioning::VideoClip {
                frames,
                frame_idx: 0,
                strength,
            }] if frames.len() == request.frames.unwrap_or(0) as usize
                && *strength == 1.0
                && frames.iter().all(|image| {
                    image.width == request.width && image.height == request.height
                }) =>
            {
                Some(extend_clip_axis(
                    request.frames.unwrap_or(0),
                    request.width,
                    request.height,
                ))
            }
            _ => None,
        };
        let bridge = match request.conditioning.as_slice() {
            [gen_core::Conditioning::VideoClip {
                frames: left,
                frame_idx: 0,
                strength: left_strength,
            }, gen_core::Conditioning::VideoClip {
                frames: right,
                frame_idx: -1,
                strength: right_strength,
            }] if left.len() == request.frames.unwrap_or(0) as usize
                && right.len() == left.len()
                && *left_strength == 1.0
                && *right_strength == 1.0
                && left.iter().chain(right).all(|image| {
                    image.width == request.width && image.height == request.height
                }) =>
            {
                Some(
                    bridge_clip_axes(request.frames.unwrap_or(0), request.width, request.height)
                        .join("+"),
                )
            }
            _ => None,
        };
        match (admitted_mode, reference.as_deref(), first_last.as_deref(), extend.as_deref(), bridge.as_deref()) {
            ("text_to_video", None, None, None, None) if request.conditioning.is_empty() => {}
            ("image_to_video", Some(actual), None, None, None)
                if Some(actual) == admitted_reference_axis(admitted_overlay) => {}
            ("image_to_video", None, None, None, None) => {
                if request.conditioning.len() > 1 { return Err(gen_core::Error::Unsupported("ltx_2_3: image_to_video admission requires exactly one Reference".into())); }
                if let [gen_core::Conditioning::Reference { image, strength }] = request.conditioning.as_slice() {
                    if image.width != request.width || image.height != request.height { return Err(gen_core::Error::Unsupported("ltx_2_3: fitted reference shape does not match output".into())); }
                    if strength.or(request.strength).unwrap_or(1.0).to_bits() != I2V_STRENGTH_BITS { return Err(gen_core::Error::Unsupported("ltx_2_3: image_to_video admission requires fixed strength 1.0".into())); }
                }
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: image_to_video admission requires one fitted Reference with fixed strength".into(),
                ));
            }
            ("first_last_frame", None, Some(actual), None, None)
                if admitted_overlay
                    .into_iter()
                    .flat_map(|overlay| overlay.split('+'))
                    .filter(|axis| axis.starts_with("keyframe:"))
                    .collect::<Vec<_>>()
                    .join("+")
                    == actual => {}
            ("first_last_frame", None, None, None, None) => {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: first_last_frame admission requires ordered fitted Keyframes at 0 and -1 with fixed strength 1.0".into(),
                ));
            }
            ("extend_clip", None, None, Some(actual), None) if admitted_overlay.into_iter().flat_map(|overlay| overlay.split('+')).filter(|axis| axis.starts_with("clip:append:")).collect::<Vec<_>>().join("+") == actual => {}
            ("extend_clip", None, None, None, None) => return Err(gen_core::Error::Unsupported("ltx_2_3: extend_clip admission requires one fitted IC-LoRA VideoClip at frame 0 with strength 1.0".into())),
            ("video_bridge", None, None, None, Some(actual)) if admitted_overlay.into_iter().flat_map(|overlay| overlay.split('+')).filter(|axis| axis.starts_with("clip:append:")).collect::<Vec<_>>().join("+") == actual => {},
            ("video_bridge", None, None, None, None) => return Err(gen_core::Error::Unsupported("ltx_2_3: video_bridge admission requires ordered fitted IC-LoRA clips at 0 and -1 with strength 1.0".into())),
            // The receipt is rebuilt from the executing request and compared to the admitted
            // overlay, so the mask blend weight, mask mode and reference cardinality cannot change
            // after admission even though the geometry cannot see them.  Carrier order is free:
            // the generator accepts `ControlClip + MultiReference` in either order.
            ("replace_person", None, None, None, None)
                if replace_person_receipt(request).is_some_and(|actual| {
                    actual == admitted_replace_person_axes(admitted_overlay)
                }) => {}
            ("replace_person", None, None, None, None) => {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: replace_person admission requires one fitted masked ControlClip at frame 0 plus one ordered 1-4 image MultiReference matching the admitted receipt".into(),
                ))
            }
            _ => {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: request mode/reference receipt crossed the admitted memory identity"
                        .into(),
                ));
            }
        }
        // These controls are part of the exact request identity.  They are safe to carry through
        // a future variant-specific evidence cell; they must not be silently erased or borrow the
        // plain cell.  `SceneWorks` keeps them fail-open until that cell is packaged.
        let expected_has_enhancer = expected_overlay
            .into_iter()
            .flat_map(|overlay| overlay.split('+'))
            .any(|axis| axis == "enhancer:uncensored");
        let request_enhancer = if request.use_uncensored_enhancer {
            Some("uncensored")
        } else if request.enhance_prompt {
            Some("standard")
        } else {
            None
        };
        let admitted_enhancer = admitted_overlay
            .into_iter()
            .flat_map(|overlay| overlay.split('+'))
            .find_map(|axis| axis.strip_prefix("enhancer:"));
        if request_enhancer != admitted_enhancer {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: enhancer flavor does not match admitted overlay".into(),
            ));
        }
        if request.use_uncensored_enhancer && !expected_has_enhancer {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: uncensored enhancer does not match the loaded overlay".into(),
            ));
        }
        let admitted_video_mode = admitted_overlay
            .into_iter()
            .flat_map(|overlay| overlay.split('+'))
            .find_map(|axis| axis.strip_prefix("provider_video_mode:"));
        if let Some(video_mode) = request.video_mode.as_deref() {
            if video_mode != "no_audio" {
                return Err(gen_core::Error::Unsupported(format!(
                    "ltx_2_3: unsupported video_mode variant {video_mode:?}"
                )));
            }
        }
        if request.video_mode.as_deref() != admitted_video_mode {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: provider video mode does not match admitted overlay".into(),
            ));
        }
        if let Some(tokens) = request.enhance_max_tokens {
            if tokens == 0 || tokens > gen_core::generator::MAX_ENHANCE_TOKENS {
                return Err(gen_core::Error::Unsupported(format!(
                    "ltx_2_3: enhance_max_tokens must be in 1..={}, got {tokens}",
                    gen_core::generator::MAX_ENHANCE_TOKENS
                )));
            }
        }
        let fps = request.fps.unwrap_or(24);
        if !matches!(fps, 24 | 25 | 30) {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: calibrated memory route requires public fps 24, 25, or 30, got {fps}"
            )));
        }
        if !SHIPPED_GEOMETRIES.contains(&(request.width, request.height))
            || !frame_count_matches_fps(fps, request.frames.unwrap_or(1))
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: request geometry/frame count is outside the shipped LTX ladder".into(),
            ));
        }
        if request
            .sampler
            .as_deref()
            .is_some_and(|sampler| sampler != "euler")
            || request.scheduler.is_some()
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: conditioned memory admission preserves the native Euler/fixed schedule"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl MemoryRequestScope for LtxMemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        Self::validate_request(
            request,
            self.expected_overlay.as_deref(),
            self.admitted_overlay.as_deref(),
            &self.admitted_mode,
        )?;
        self.inner.configure_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.inner.configure_decode(tile_edge, overlap, geometry)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(chunk_size)
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.inner
            .materialize_transformer_window(first_block, block_count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.inner.finish(outcome)
    }
}

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    transformer_blocks: usize,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_with_cleanup(
        contract,
        loaded_tier,
        expected_overlay,
        transformer_blocks,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let tier = registered_tier(spec, contract).map_err(gen_core::Error::from)?;
    begin_with_cleanup(
        contract,
        tier,
        route_overlay(spec).as_deref(),
        LtxConfig::video_only_defaults().num_layers as usize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn decode_tiling(request: &GenerationRequest) -> Result<Option<TilingConfig>> {
    let Some(memory) = request.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: selected bounded decode is missing decode_tile_edge".into(),
        )
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: selected bounded decode is missing decode_overlap".into(),
        )
    })?;
    validate_decode(Some(edge), Some(overlap)).map_err(mlx_gen::Error::from)?;
    crate::pipeline::selected_tiling_budgeted_ltx(
        request.height as i32,
        request.width as i32,
        request.frames.unwrap_or(1) as i32,
        edge as i32,
        overlap as i32,
    )
    .map(Some)
}

pub(crate) const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::MODEL_ID,
    contract: |spec| memory_strategy_contract(spec).map_err(Into::into),
    safety_check: registered_safety_check,
};

pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        Conditioning, GenerationMemory, Image, MemoryStrategyParameters, Precision, ReplacementMode,
    };

    fn fixture_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-fixture".into())).with_quant(Quant::Q8)
    }

    /// AC (SC-22662): the LTX-2.3 route publishes the axes of the AvDiT and the video VAE this
    /// crate declares, and passes the shared facts conformance check.
    /// SC-22667: the activation width follows the precision `model.rs` selects the DiT path from —
    /// `Bf16` runs bf16 activations, the `Fp32` opt-in runs f32 ones — rather than the
    /// half-precision literal that described only the default path.
    ///
    /// Mutation that fails this: `activation_dtype_width: Some(HALF_ACTIVATION_WIDTH)` (the shape
    /// under review) — the `Fp32` assertion then reads 2.
    #[test]
    fn activation_width_follows_the_selected_precision() {
        let bf16 = fixture_spec();
        assert_eq!(architecture_facts(&bf16).activation_dtype_width, Some(2));
        let mut fp32 = bf16.clone();
        fp32.precision = Precision::Fp32;
        assert_eq!(architecture_facts(&fp32).activation_dtype_width, Some(4));
        assert_eq!(
            weights_free_memory_strategy_contract(&fp32)
                .unwrap()
                .architecture_facts
                .activation_dtype_width,
            Some(4)
        );
    }

    #[test]
    fn architecture_facts_follow_the_crate_transformer_and_vae_constants() {
        let contract = weights_free_memory_strategy_contract(&fixture_spec()).unwrap();
        assert_eq!(
            contract.architecture_facts,
            mlx_gen::gen_core::MemoryArchitectureFacts {
                attention_heads: Some(32),
                head_dim: Some(128),
                transformer_blocks: Some(48),
                // The AvDiT has no patchify: `patchify_proj` is a plain Linear over the
                // 128-channel latent token, and every patch factor lives inside the VAE.
                patch_size: None,
                latent_channels: Some(128),
                vae_spatial_scale: Some(32),
                // A video autoencoder: eight frames per latent unit.
                vae_temporal_scale: Some(8),
                activation_dtype_width: Some(2),
            },
            "ltx_2_3 architecture facts"
        );
        assert!(contract.architecture_facts.has_declared_architecture_axis());
        // The published scales ARE the released DiffVAE's `(time, height, width)` factors.
        assert_eq!(
            [
                contract.architecture_facts.vae_temporal_scale,
                contract.architecture_facts.vae_spatial_scale,
                contract.architecture_facts.vae_spatial_scale,
            ],
            crate::diff_vae::VIDEO_SCALE_FACTORS.map(|scale| Some(scale as u32))
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);
    }

    /// The `embedded_config.json` keys `LtxConfig::from_model_dir` reads for the three trunk axes
    /// this provider publishes. Emitted from a config value so the fixture cannot drift.
    fn embedded_config_json(cfg: &crate::config::LtxConfig) -> serde_json::Value {
        serde_json::json!({
            "transformer": {
                "num_attention_heads": cfg.num_attention_heads,
                "attention_head_dim": cfg.attention_head_dim,
                "num_layers": cfg.num_layers,
            }
        })
    }

    fn spec_for_embedded_config(dir: &std::path::Path, config: &serde_json::Value) -> LoadSpec {
        std::fs::write(dir.join("embedded_config.json"), config.to_string()).unwrap();
        LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
    }

    /// AC (SC-22662, review follow-up): on the **materialized** path the trunk axes come from the
    /// snapshot's own `embedded_config.json` — the file the loader parses — rather than from the
    /// compile-time preset. The mirror fixture agrees with the weights-free path; a fixture with
    /// mutated keys publishes the mutated axes, which is what the unconditional
    /// `architecture_facts()` this replaced would fail.
    #[test]
    fn materialized_trunk_axes_come_from_the_snapshot_rather_than_the_preset() {
        let preset = crate::config::LtxConfig::video_only_defaults();

        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&spec_for_embedded_config(
                mirror.path(),
                &embedded_config_json(&preset)
            )),
            architecture_facts(&fixture_spec()),
            "a snapshot mirroring the reference config must publish the preset's axes"
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = embedded_config_json(&preset);
        mutated["transformer"]["num_layers"] = serde_json::json!(7);
        mutated["transformer"]["attention_head_dim"] = serde_json::json!(64);
        let mutated_facts =
            architecture_facts(&spec_for_embedded_config(mutated_dir.path(), &mutated));
        assert_eq!(
            (mutated_facts.transformer_blocks, mutated_facts.head_dim),
            (Some(7), Some(64)),
            "the materialized path must publish the snapshot's geometry, not the preset's"
        );
    }

    /// SC-22667: the three snapshot outcomes are exactly the loader's. An ABSENT
    /// `embedded_config.json` is the one case `LtxConfig::from_model_dir` itself answers with the
    /// preset (and `model.rs`'s `?` accepts), so the preset is published there. A file that is
    /// PRESENT and unparseable — malformed JSON, or no `transformer` section — errors at load, so
    /// the trunk axes are declared absent instead of standing the preset in for a tree the engine
    /// refuses to open. The VAE axes are crate constants no config can move, so they survive.
    ///
    /// Mutation that fails this: the `from_model_dir(root).ok().unwrap_or_else(video_only_defaults)`
    /// shape under review — both unparseable fixtures then publish `Some(32)/Some(128)/Some(48)`.
    #[test]
    fn an_unparseable_embedded_config_declines_the_trunk_axes_instead_of_presetting() {
        let absent = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&LoadSpec::new(WeightsSource::Dir(
                absent.path().to_path_buf()
            ))),
            architecture_facts(&fixture_spec()),
            "an absent embedded_config.json is the loader's own preset fall-back"
        );

        for body in [
            "{ this is not json".to_owned(),
            serde_json::json!({ "vae": { "num_layers": 3 } }).to_string(),
        ] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("embedded_config.json"), &body).unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
            assert!(
                crate::config::LtxConfig::from_model_dir(dir.path()).is_err(),
                "fixture premise: the loader's own parse must reject {body}"
            );
            let facts = architecture_facts(&spec);
            assert_eq!(
                (
                    facts.attention_heads,
                    facts.head_dim,
                    facts.transformer_blocks,
                    facts.patch_size
                ),
                (None, None, None, None),
                "an unparseable {body} must not publish preset trunk axes"
            );
            assert_eq!(
                (
                    facts.latent_channels,
                    facts.vae_spatial_scale,
                    facts.vae_temporal_scale,
                    facts.activation_dtype_width
                ),
                (Some(128), Some(32), Some(8), Some(2)),
                "the VAE axes and the activation width are not config-derived"
            );
            assert!(facts.has_declared_architecture_axis());
        }
    }

    /// `ProviderRegistry::memory_contract_surfaces` constructs a contract for **every** selector the
    /// fixture publishes and fails the entire MLX catalog when one errors, so the published witness
    /// set must be exactly the set this provider can build. Asserting it here localizes the failure
    /// to LTX instead of surfacing it as eight red `mlx-gen-catalog` tests.
    #[test]
    fn every_published_contract_surface_builds_and_no_deferred_surface_is_published() {
        let surfaces = memory_contract_surface_specs();
        assert_eq!(
            surfaces.len(),
            gen_core::mlx_memory_contract_surface_specs().len() / 2,
            "the witness set is the eager half of the shared MLX surface"
        );
        for surface in &surfaces {
            assert_eq!(
                surface.selector.load_shape,
                LoadShape::EagerMaterialization,
                "{} has no deferred/block-window loader",
                surface.selector.id()
            );
            weights_free_memory_strategy_contract(&surface.spec).unwrap_or_else(|error| {
                panic!("surface {} must build: {error}", surface.selector.id())
            });
        }
        assert!(
            gen_core::mlx_memory_contract_surface_specs()
                .into_iter()
                .filter(|surface| surface.selector.load_shape == LoadShape::DeferredMaterialization)
                .all(|surface| weights_free_memory_strategy_contract(&surface.spec).is_err()),
            "a deferred surface that now builds must be published, not filtered out"
        );
    }

    #[test]
    fn survey_rungs_and_video_formula_are_declared_without_a_default_standin() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert!(matches!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        for missing in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert!(matches!(
                contract.capability(missing).unwrap().support,
                MemoryStrategySupport::Missing
            ));
        }
        assert!(contract.formula.uses(MemoryFormulaVariable::FrameCount));
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ]
        );

        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q8),
            component_precision_floors: &[],
        };
        let resident = contract
            .representative_selection(MemoryStrategy::Resident, tier, false)
            .unwrap();
        let staged = contract
            .representative_selection(MemoryStrategy::StagedResidency, tier, false)
            .unwrap();
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::Resident),
            [MemoryStrategy::Resident]
        );
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::StagedResidency),
            [MemoryStrategy::Resident, MemoryStrategy::StagedResidency]
        );
        assert_eq!(contract.generation_memory(&resident), None);
        assert_eq!(
            contract.generation_memory(&staged),
            Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
    }

    #[test]
    fn selected_decode_parameters_reach_the_real_tiling_carrier_and_mutations_fail() {
        let mut request = GenerationRequest {
            // Keep the carrier assertion below the smallest supported CI host's live MLX budget.
            // Long-clip temporal-budget behavior is covered by the injected-budget pipeline test;
            // this test owns only request-memory propagation and domain validation.
            width: 256,
            height: 256,
            frames: Some(25),
            count: 1,
            memory: Some(GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(256),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tiling = decode_tiling(&request).unwrap().unwrap();
        let spatial = tiling.spatial.unwrap();
        assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 64));

        request.memory.as_mut().unwrap().decode_tile_edge = Some(255);
        assert!(decode_tiling(&request).is_err());
        request.memory.as_mut().unwrap().decode_tile_edge = None;
        assert!(decode_tiling(&request).is_err());

        let contract = weights_free_memory_strategy_contract(&fixture_spec()).unwrap();
        let mut selection = contract
            .representative_selection(
                MemoryStrategy::BoundedDecode,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q8),
                    component_precision_floors: &[],
                },
                false,
            )
            .unwrap();
        selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(511),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert!(contract.validate_selection(&selection).is_err());
    }

    fn calibrated_t2v_scope_and_request() -> (Box<dyn MemoryRequestScope>, GenerationRequest) {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
            .unwrap();
        let scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        (scope, fixture.request)
    }

    #[test]
    fn calibrated_scope_rejects_zero_fps_before_installing_controls() {
        let (mut scope, request) = calibrated_t2v_scope_and_request();
        let mut zero_fps = GenerationRequest {
            fps: Some(0),
            ..request
        };
        let error = scope
            .configure_request(&mut zero_fps)
            .expect_err("calibrated MLX LTX admission must reject zero fps");
        assert!(
            matches!(error, gen_core::Error::Unsupported(_)),
            "got: {error:?}"
        );
        assert!(error
            .to_string()
            .contains("requires public fps 24, 25, or 30"));
        assert_eq!(zero_fps.memory, None);
    }

    #[test]
    fn first_last_fixture_binds_ordered_two_keyframe_identity() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "first_last_frame")
            .unwrap();
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        let mut accepted = fixture.request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.unwrap().stage_residency);

        let mut crossed = fixture.request;
        crossed.conditioning.swap(0, 1);
        let error = scope
            .configure_request(&mut crossed)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ordered fitted Keyframes"), "{error}");
        assert_eq!(crossed.memory, None);
    }

    #[test]
    fn bridge_fixture_binds_ordered_two_clip_identity() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "video_bridge")
            .unwrap();
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        let mut accepted = fixture.request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.unwrap().stage_residency);

        let mut crossed = fixture.request;
        crossed.conditioning.swap(0, 1);
        let error = scope
            .configure_request(&mut crossed)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ordered fitted IC-LoRA clips"), "{error}");
        assert_eq!(crossed.memory, None);
    }

    /// sc-20799: `replace_person` is implemented and advertised by this provider's generator, so it
    /// must also be admissible. It previously fell to `validate_request`'s catch-all, which made
    /// the mode unreachable behind an admitted memory rung. Mirrors the Candle sibling test.
    #[test]
    fn replace_person_fixture_binds_the_masked_clip_and_character_sheet_identity() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "replace_person")
            .expect("replace_person ships a behavior fixture");
        assert_eq!(fixture.context.geometry.reference_count, 2);
        assert!(fixture.context.has_reference);
        // The shared scope core compares this against the admitted geometry; if the two counters
        // disagreed the mode would be unreachable no matter what the arms below accept.
        assert_eq!(
            fixture.request.memory_reference_count(),
            fixture.context.geometry.reference_count
        );
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        let mut accepted = fixture.request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.unwrap().stage_residency);

        // Carrier order is free (the generator accepts either), so swapping must NOT fail.
        let mut swapped = fixture.request.clone();
        swapped.conditioning.swap(0, 1);
        scope.configure_request(&mut swapped).unwrap();

        for (label, cross) in [
            (
                "reference cardinality",
                Box::new(|request: &mut GenerationRequest| {
                    if let Conditioning::MultiReference { images } = &mut request.conditioning[1] {
                        images.push(images[0].clone());
                    }
                }) as Box<dyn Fn(&mut GenerationRequest)>,
            ),
            (
                "mask blend weight",
                Box::new(|request: &mut GenerationRequest| {
                    if let Conditioning::ControlClip {
                        masking_strength, ..
                    } = &mut request.conditioning[0]
                    {
                        *masking_strength = 0.5;
                    }
                }),
            ),
            (
                "replacement mode",
                Box::new(|request: &mut GenerationRequest| {
                    if let Conditioning::ControlClip { mode, .. } = &mut request.conditioning[0] {
                        *mode = gen_core::ReplacementMode::FaceOnly;
                    }
                }),
            ),
            (
                "clip start frame",
                Box::new(|request: &mut GenerationRequest| {
                    if let Conditioning::ControlClip { start_frame, .. } =
                        &mut request.conditioning[0]
                    {
                        *start_frame = 1;
                    }
                }),
            ),
        ] {
            let mut crossed = fixture.request.clone();
            cross(&mut crossed);
            let error = scope
                .configure_request(&mut crossed)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("replace_person admission requires"),
                "crossing the {label} must not be admitted: {error}"
            );
            assert_eq!(crossed.memory, None, "{label}");
        }
    }

    #[test]
    fn replace_person_admission_refuses_a_crossed_or_missing_receipt() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "replace_person")
            .unwrap();
        let geometry = fixture.context.geometry;
        assert!(replace_person_axes_shape_ok(
            &admitted_replace_person_axes(fixture.context.overlay.as_deref()),
            geometry
        ));
        let load = route_overlay(&spec);
        let with_load = |receipt: &str| match &load {
            Some(load) => format!("{load}+{receipt}"),
            None => receipt.to_owned(),
        };
        for crossed in [
            None,
            Some(with_load("")),
            // One axis only.
            Some(with_load(&format!(
                "reference:sheet:2:image:{}x{}",
                geometry.width, geometry.height
            ))),
            // A sheet that claims a different cardinality than the admitted geometry.
            Some(with_load(&format!(
                "clip:replace:frames:{}:image:{}x{}:frame:0:mode:1:mask:{:08x}+reference:sheet:3:image:{}x{}",
                geometry.frames,
                geometry.width,
                geometry.height,
                1.0_f32.to_bits(),
                geometry.width,
                geometry.height
            ))),
            // An out-of-range mask blend weight.
            Some(with_load(&format!(
                "clip:replace:frames:{}:image:{}x{}:frame:0:mode:1:mask:{:08x}+reference:sheet:2:image:{}x{}",
                geometry.frames,
                geometry.width,
                geometry.height,
                2.0_f32.to_bits(),
                geometry.width,
                geometry.height
            ))),
            // An undeclared replacement mode ordinal.
            Some(with_load(&format!(
                "clip:replace:frames:{}:image:{}x{}:frame:0:mode:9:mask:{:08x}+reference:sheet:2:image:{}x{}",
                geometry.frames,
                geometry.width,
                geometry.height,
                1.0_f32.to_bits(),
                geometry.width,
                geometry.height
            ))),
        ] {
            let mut context = fixture.context.clone();
            context.overlay = crossed.clone();
            assert!(
                matches!(
                    registered_safety_check(&spec, &contract, &context),
                    MemorySafetyDecision::Reject { .. }
                ),
                "{crossed:?} must not be admitted"
            );
        }
    }

    #[test]
    fn calibrated_scope_binds_empty_t2v_request_and_exact_fps_frame_pair_before_controls() {
        let (mut scope, request) = calibrated_t2v_scope_and_request();
        let mut accepted = request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.unwrap().stage_residency);

        let mut crossed_pair = GenerationRequest {
            fps: Some(24),
            ..request.clone()
        };
        let error = scope
            .configure_request(&mut crossed_pair)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside the shipped LTX ladder"));
        assert_eq!(crossed_pair.memory, None);

        let image = Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        for conditioning in [
            Conditioning::Keyframe {
                image: image.clone(),
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::VideoClip {
                frames: vec![image.clone()],
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::ControlClip {
                frames: vec![image.clone()],
                mask: vec![image],
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::FaceOnly,
            },
        ] {
            let expected_reference_count =
                u32::from(matches!(conditioning, Conditioning::Keyframe { .. }));
            let mut temporal_conditioning = GenerationRequest {
                conditioning: vec![conditioning],
                ..request.clone()
            };
            assert_eq!(
                temporal_conditioning.image_reference_count(),
                expected_reference_count
            );
            let error = scope
                .configure_request(&mut temporal_conditioning)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("request mode/reference receipt crossed")
                    || error.contains("accepts only the single fitted image Reference carrier")
                    || error.contains("does not fit admitted"),
                "{error}"
            );
            assert_eq!(temporal_conditioning.memory, None);
        }

        for variant in [
            GenerationRequest {
                enhance_prompt: true,
                ..request.clone()
            },
            GenerationRequest {
                use_uncensored_enhancer: true,
                ..request.clone()
            },
        ] {
            let mut variant = variant;
            let error = scope
                .configure_request(&mut variant)
                .unwrap_err()
                .to_string();
            assert!(error.contains("enhancer flavor does not match admitted overlay"));
            assert_eq!(variant.memory, None);
        }

        let mut enhancer_spec = fixture_spec();
        enhancer_spec.components.insert(
            "uncensored_enhancer".into(),
            WeightsSource::Dir("/nonexistent-enhancer-fixture".into()),
        );
        let enhancer_contract = weights_free_memory_strategy_contract(&enhancer_spec).unwrap();
        let enhancer_fixture = registered_valid_fixtures(
            &enhancer_spec,
            &enhancer_contract,
            MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .into_iter()
        .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
        .unwrap();
        let mut enhancer_scope = registered_begin_request(
            &enhancer_spec,
            &enhancer_contract,
            &enhancer_fixture.context,
        )
        .unwrap()
        .unwrap();
        let mut crossed_standard = GenerationRequest {
            enhance_prompt: true,
            ..enhancer_fixture.request.clone()
        };
        let error = enhancer_scope
            .configure_request(&mut crossed_standard)
            .unwrap_err()
            .to_string();
        assert!(error.contains("enhancer flavor does not match admitted overlay"));
        let mut admitted_uncensored = GenerationRequest {
            enhance_prompt: true,
            use_uncensored_enhancer: true,
            enhance_max_tokens: Some(gen_core::generator::MAX_ENHANCE_TOKENS),
            ..enhancer_fixture.request.clone()
        };
        enhancer_scope
            .configure_request(&mut admitted_uncensored)
            .unwrap();
        assert!(admitted_uncensored.memory.unwrap().stage_residency);
        let mut crossed_token_budget = GenerationRequest {
            enhance_prompt: true,
            use_uncensored_enhancer: true,
            enhance_max_tokens: Some(gen_core::generator::MAX_ENHANCE_TOKENS + 1),
            ..enhancer_fixture.request.clone()
        };
        let error = enhancer_scope
            .configure_request(&mut crossed_token_budget)
            .unwrap_err()
            .to_string();
        assert!(error.contains("enhance_max_tokens must be in 1..="));
        assert_eq!(crossed_token_budget.memory, None);

        let standard_spec = fixture_spec();
        let standard_contract = weights_free_memory_strategy_contract(&standard_spec).unwrap();
        let mut standard_context = registered_valid_fixtures(
            &standard_spec,
            &standard_contract,
            MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .into_iter()
        .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
        .unwrap()
        .context;
        standard_context.overlay = Some("enhancer:standard".into());
        let mut standard_scope =
            registered_begin_request(&standard_spec, &standard_contract, &standard_context)
                .unwrap()
                .unwrap();
        let mut admitted_standard = GenerationRequest {
            enhance_prompt: true,
            enhance_max_tokens: Some(gen_core::generator::MAX_ENHANCE_TOKENS),
            ..request.clone()
        };
        standard_scope
            .configure_request(&mut admitted_standard)
            .unwrap();
        assert!(admitted_standard.memory.unwrap().stage_residency);
        let mut crossed_uncensored = GenerationRequest {
            use_uncensored_enhancer: true,
            ..request.clone()
        };
        let error = standard_scope
            .configure_request(&mut crossed_uncensored)
            .unwrap_err()
            .to_string();
        assert!(error.contains("enhancer flavor does not match admitted overlay"));

        let mut no_audio = GenerationRequest {
            video_mode: Some("no_audio".into()),
            ..request.clone()
        };
        let error = scope
            .configure_request(&mut no_audio)
            .unwrap_err()
            .to_string();
        assert!(error.contains("provider video mode does not match admitted overlay"));
        assert_eq!(no_audio.memory, None);

        let no_audio_spec = fixture_spec();
        let no_audio_contract = weights_free_memory_strategy_contract(&no_audio_spec).unwrap();
        let mut no_audio_context = registered_valid_fixtures(
            &no_audio_spec,
            &no_audio_contract,
            MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .into_iter()
        .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
        .unwrap()
        .context;
        no_audio_context.overlay = Some("provider_video_mode:no_audio".into());
        let mut no_audio_scope =
            registered_begin_request(&no_audio_spec, &no_audio_contract, &no_audio_context)
                .unwrap()
                .unwrap();
        let mut admitted_no_audio = GenerationRequest {
            video_mode: Some("no_audio".into()),
            ..request.clone()
        };
        no_audio_scope
            .configure_request(&mut admitted_no_audio)
            .unwrap();
        assert!(admitted_no_audio.memory.unwrap().stage_residency);
        let mut crossed_plain = request.clone();
        let error = no_audio_scope
            .configure_request(&mut crossed_plain)
            .unwrap_err()
            .to_string();
        assert!(error.contains("provider video mode does not match admitted overlay"));

        for video_mode in ["video_only", "image_to_video"] {
            let mut variant = GenerationRequest {
                video_mode: Some(video_mode.into()),
                ..request.clone()
            };
            let error = scope
                .configure_request(&mut variant)
                .unwrap_err()
                .to_string();
            assert!(error.contains("unsupported video_mode variant"));
            assert_eq!(variant.memory, None);
        }

        for fps in [Some(0), Some(23), Some(26), Some(29), Some(31)] {
            let mut outside_envelope = GenerationRequest {
                fps,
                ..request.clone()
            };
            let error = scope
                .configure_request(&mut outside_envelope)
                .unwrap_err()
                .to_string();
            assert!(error.contains("requires public fps 24, 25, or 30"));
            assert_eq!(outside_envelope.memory, None);
        }

        // Provider-route rejection happens before the shared carrier mutates the request, and it
        // leaves the inner scope active so the caller can report the terminal error and run cleanup.
        scope
            .finish(MemoryRunOutcome::Error {
                message: "request route rejected".into(),
            })
            .unwrap();
        let mut after_finish = request;
        assert!(scope.configure_request(&mut after_finish).is_err());
    }

    #[test]
    fn calibrated_i2v_scope_binds_fitted_reference_strength_schedule_and_frame_identity() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "image_to_video")
            .unwrap();
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();

        let mut accepted = fixture.request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.unwrap().stage_residency);

        let reference = fixture.request.conditioning[0].clone();
        for (mut crossed, expected) in [
            (
                GenerationRequest {
                    conditioning: Vec::new(),
                    ..fixture.request.clone()
                },
                "requires one fitted Reference",
            ),
            (
                GenerationRequest {
                    conditioning: vec![reference.clone(), reference.clone()],
                    ..fixture.request.clone()
                },
                "requires exactly one Reference",
            ),
            (
                GenerationRequest {
                    conditioning: vec![Conditioning::Reference {
                        image: Image {
                            width: 512,
                            height: 768,
                            pixels: vec![0; 512 * 768 * 3],
                        },
                        strength: Some(1.0),
                    }],
                    ..fixture.request.clone()
                },
                "fitted reference shape",
            ),
            (
                GenerationRequest {
                    conditioning: vec![Conditioning::Reference {
                        image: Image {
                            width: 768,
                            height: 512,
                            pixels: vec![0; 768 * 512 * 3],
                        },
                        strength: Some(0.5),
                    }],
                    ..fixture.request.clone()
                },
                "requires fixed strength 1.0",
            ),
            (
                GenerationRequest {
                    fps: Some(24),
                    ..fixture.request.clone()
                },
                "outside the shipped LTX ladder",
            ),
            (
                GenerationRequest {
                    sampler: Some("heun".into()),
                    ..fixture.request.clone()
                },
                "native Euler/fixed schedule",
            ),
            (
                GenerationRequest {
                    scheduler: Some("linear".into()),
                    ..fixture.request.clone()
                },
                "native Euler/fixed schedule",
            ),
        ] {
            let error = scope
                .configure_request(&mut crossed)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "unexpected refusal: {error}");
            assert_eq!(crossed.memory, None);
        }

        let (mut t2v_scope, mut crossed_i2v) = calibrated_t2v_scope_and_request();
        crossed_i2v.conditioning = fixture.request.conditioning.clone();
        let error = t2v_scope
            .configure_request(&mut crossed_i2v)
            .unwrap_err()
            .to_string();
        assert!(error.contains("crossed the admitted memory identity"));

        let mut crossed_t2v = GenerationRequest {
            conditioning: Vec::new(),
            ..fixture.request
        };
        let error = scope
            .configure_request(&mut crossed_t2v)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires one fitted Reference"));
    }

    #[test]
    fn canonical_overlay_preserves_adapter_count_enhancer_and_provider_mode() {
        assert_eq!(canonical_overlay(0, None, None), None);
        assert_eq!(
            canonical_overlay(2, Some("uncensored"), Some("no_audio")).as_deref(),
            Some("adapters:2+enhancer:uncensored+provider_video_mode:no_audio")
        );
        assert!(overlay_matches_loaded_route(
            Some("adapters:2+enhancer:uncensored+provider_video_mode:no_audio"),
            Some("adapters:2+enhancer:uncensored")
        ));
        assert!(!overlay_matches_loaded_route(
            Some("adapters:1+enhancer:uncensored+provider_video_mode:no_audio"),
            Some("adapters:2+enhancer:uncensored")
        ));
        assert!(!overlay_matches_loaded_route(
            Some("provider_video_mode:video_only"),
            None
        ));
    }

    #[test]
    fn uncensored_enhancer_artifact_is_charged_to_conditioning_floor() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"quantization":{"mode":"affine","group_size":64,"bits":4}}"#,
        )
        .unwrap();
        let header = br#"{"model.embed_tokens.weight":{"dtype":"U32","shape":[2,16],"data_offsets":[0,128]},"model.embed_tokens.scales":{"dtype":"F16","shape":[2,2],"data_offsets":[128,136]},"model.embed_tokens.biases":{"dtype":"F16","shape":[2,2],"data_offsets":[136,144]},"model.layer.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[144,152]}}"#;
        let mut padded = header.to_vec();
        while !(padded.len() + 8).is_multiple_of(8) {
            padded.push(b' ');
        }
        let mut file = (padded.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&padded);
        file.extend_from_slice(&[0; 152]);
        std::fs::write(dir.path().join("model.safetensors"), file).unwrap();

        let mut spec = fixture_spec();
        spec.components.insert(
            "uncensored_enhancer".into(),
            WeightsSource::Dir(dir.path().to_path_buf()),
        );
        let enhancer_bytes = uncensored_enhancer_resident_bytes(&spec).unwrap();
        // Stored peak: 152 B. Expanded embedding: [2, 2*64] bf16 = 512 B. Both coexist while MLX
        // evaluates the lazy dequantization graph, so the conditioning floor is their exact sum.
        assert_eq!(enhancer_bytes, 664);

        let contract = build_contract(
            &spec,
            AssetDeclaration {
                facts: MemoryAssetFacts {
                    base_bytes: enhancer_bytes,
                    conditioning_bytes: enhancer_bytes,
                    ..Default::default()
                },
                resident_components: Vec::new(),
            },
            STATIC_CALIBRATION_FINGERPRINT,
        )
        .unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, enhancer_bytes);
        assert_eq!(contract.total_resident_bytes(), enhancer_bytes);
    }

    // ------------------------------------------------------------------------------------------
    // sc-22737 (epic sc-22723 E1/E4): the identity is per TIER, not per route.
    // ------------------------------------------------------------------------------------------

    fn ltx_tier(quant: Option<Quant>) -> MemoryNumericTier {
        MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
            quant,
            component_precision_floors: &[],
        }
    }

    /// The retained q8 key is unchanged, and the other two shipped tiers publish their OWN keys —
    /// so an anchor measured on one tier can never be matched against a load of another.
    #[test]
    fn each_shipped_tier_publishes_its_own_identity() {
        assert_eq!(
            production_calibration_fingerprint(&ltx_tier(Some(Quant::Q8))).as_deref(),
            Some(CALIBRATION_FINGERPRINT),
            "the measured cell keeps its retained key"
        );
        let mut seen = std::collections::BTreeSet::new();
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let fingerprint = production_calibration_fingerprint(&ltx_tier(quant))
                .unwrap_or_else(|| panic!("{quant:?} publishes an identity"));
            assert!(
                seen.insert(fingerprint.clone()),
                "{quant:?} reuses {fingerprint}"
            );
            assert_ne!(fingerprint, STATIC_CALIBRATION_FINGERPRINT);
        }
        assert_eq!(seen.len(), 3);
        assert_eq!(
            production_calibration_fingerprint(&ltx_tier(None)).unwrap(),
            "sc-20772-ltx-2-3-bf16-mlx-memory-ladder-v2"
        );
        assert_eq!(
            production_calibration_fingerprint(&ltx_tier(Some(Quant::Q4))).unwrap(),
            "sc-20772-ltx-2-3-q4-mlx-memory-ladder-v2"
        );
    }

    /// A packed width this family does not ship publishes nothing rather than a plausible key.
    /// NVFP4 is a real `Quant` variant with no MLX hardware behind it, so it is the honest probe.
    #[test]
    fn an_unshipped_packed_width_is_withheld() {
        assert!(production_calibration_fingerprint(&ltx_tier(Some(Quant::Nvfp4))).is_none());
    }

    #[test]
    fn uncensored_enhancer_rejects_config_that_disagrees_with_packed_embedding() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"quantization":{"mode":"affine","group_size":64,"bits":8}}"#,
        )
        .unwrap();
        let header = br#"{"model.embed_tokens.weight":{"dtype":"U32","shape":[2,16],"data_offsets":[0,128]},"model.embed_tokens.scales":{"dtype":"F16","shape":[2,2],"data_offsets":[128,136]},"model.embed_tokens.biases":{"dtype":"F16","shape":[2,2],"data_offsets":[136,144]}}"#;
        let mut padded = header.to_vec();
        while !(padded.len() + 8).is_multiple_of(8) {
            padded.push(b' ');
        }
        let mut file = (padded.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&padded);
        file.extend_from_slice(&[0; 144]);
        std::fs::write(dir.path().join("model.safetensors"), file).unwrap();
        let mut spec = fixture_spec();
        spec.components.insert(
            "uncensored_enhancer".into(),
            WeightsSource::Dir(dir.path().to_path_buf()),
        );
        let error = uncensored_enhancer_resident_bytes(&spec)
            .unwrap_err()
            .to_string();
        assert!(error.contains("geometry disagrees"), "{error}");
    }
}
