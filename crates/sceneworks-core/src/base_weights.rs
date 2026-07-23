//! Base-weight architecture detection for external ComfyUI files (sc-10662,
//! epic 10451 Phase 2).
//!
//! Phase 1 (sc-10452) scans a ComfyUI `loras/` subtree and classifies each
//! adapter with [`crate::lora_family::detect_lora_family`]. Phase 2 must reuse the
//! *base* weights in the sibling subtrees — `unet/`, `diffusion_models/`,
//! `text_encoders/`, `vae/`, `checkpoints/` — and **nothing classifies a base
//! file today**. `detect_model_family` only reads a diffusers `model_index.json`
//! or falls through to the LoRA classifier; a single-file ComfyUI base weight is
//! neither, so it returns `None` or misclassifies.
//!
//! Before Phase 2 can pick a per-family remap table or a dequant path it must
//! decide three things about a file, and **each one branches the downstream
//! policy**:
//!
//! 1. **Component role** — is this the diffusion transformer, a text encoder, a
//!    VAE, or an all-in-one checkpoint? ComfyUI stores them as separate files, so
//!    an assembled load needs all three roles located.
//! 2. **Architecture family** — z-image vs qwen-image vs wan vs flux2 …; the
//!    remap table is keyed on it.
//! 3. **On-disk quant format** — bf16 · plain fp8 · one of *three* distinct
//!    ComfyUI scaled/packed fp8/fp4 conventions · GGUF. The dequant math differs
//!    per convention, and there is no silent-fallback slack: a file mis-detected
//!    as plain-castable fp8 when it is actually scaled would decode to noise, not
//!    an error — the worst violation of the no-silent-fallback rule available.
//!
//! The detector is **header-only** (key names + dtypes; a 7 GB file costs a few
//! KB of I/O — the same posture as the Phase 1 LoRA scan) and **GPU-free**. It
//! emits a typed [`BaseWeightDetection::Recognized`] `(family, component, quant)`
//! or a [`BaseWeightDetection::Unrecognized`] carrying a reason — never a guess.
//! Filenames are deliberately **not** consulted: users rename files, civitai
//! downloads arrive with arbitrary names, and a file labelled `*_fp8_scaled` may
//! actually be `comfy_quant`-packed (`ideogram4_fp8_scaled` is).
//!
//! Out of scope (these are the Phase 2 implementation slices, not this story): the
//! `ComfyUI-keys → VarBuilder` remap seam, the per-family key tables, and the
//! actual dequant kernels.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use serde_json::Value;

use crate::lora_family::{read_safetensors_header, SafetensorsHeaderError};

/// Which part of a generation pipeline a base-weight file holds. ComfyUI stores
/// these as separate files (modern checkpoints are *not* fused), so an assembled
/// virtual model must locate each role; a legacy all-in-one `checkpoints/` file
/// carries several at once ([`ComponentRole::Checkpoint`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentRole {
    /// The diffusion backbone (DiT / UNet).
    Transformer,
    /// A prompt text encoder — an LLM decoder (`model.embed_tokens` + `model.layers`)
    /// or a T5-style encoder (`shared` + `encoder.block`).
    TextEncoder,
    /// A variational autoencoder (paired `encoder.`/`decoder.` conv stacks).
    Vae,
    /// A legacy all-in-one checkpoint carrying the transformer plus at least one
    /// of the VAE / text-encoder (SD1.5/SDXL-era, and the LTX-2.3 `checkpoints/*`
    /// audio+video bundles). Distinguished from the `diffusion_models/*` DiT-only
    /// sibling, which is [`ComponentRole::Transformer`].
    Checkpoint,
}

impl ComponentRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::TextEncoder => "text_encoder",
            Self::Vae => "vae",
            Self::Checkpoint => "checkpoint",
        }
    }
}

impl std::fmt::Display for ComponentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The on-disk numeric encoding of a base-weight file. The survey of a real
/// ComfyUI tree (sc-10662) found the epic's assumed "bf16 / plain-fp8 /
/// scaled-fp8 / fp4 / GGUF" is really **four incompatible scaled/packed
/// conventions**, told apart by *marker keys*, not by dtype — each needs its own
/// dequant path downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// All-`BF16` weights — load as-is (the Phase 2 prototype target).
    Bf16,
    /// All-`F16` weights.
    F16,
    /// All-`F32` weights (typical of VAEs).
    F32,
    /// Plain `F8_E4M3` with **no** scale companions — cast up at load. The tree's
    /// `qwen_image_*_fp8_e4m3fn` files (dtype set is exactly `{F8_E4M3}`).
    Fp8E4m3,
    /// ComfyUI **companion** scaled-fp8: per-tensor `.scale_weight`(+`.scale_input`)
    /// sibling tensors and a top-level `scaled_fp8` marker. `wan2.2_*_fp8_scaled`,
    /// the Kijai `*_KJ` variants (`.scale_weight` only), `umt5_*_scaled`. Dequant:
    /// `w = w_fp8.to(bf16) * scale_weight`.
    ScaledFp8Companion,
    /// FLUX.2 **inline**-scale fp8: each quantized Linear carries a
    /// `.weight`+`.weight_scale`+`.input_scale` triplet, with no `.scale_weight`
    /// companion and no `scaled_fp8` marker. `flux2_dev_fp8mixed` (mixed — some
    /// layers stay bf16). A different dequant path than [`Self::ScaledFp8Companion`].
    Fp8InlineScale,
    /// ComfyUI `comfy_quant` packed fp4/mxfp4: a `.comfy_quant` marker per Linear
    /// plus `.weight_scale`/`weight_scale_N` block scales over `U8`-packed nibbles.
    /// `gemma_3_12B_it_fp4_mixed`, `ideogram4_fp8_scaled` (packed despite the name).
    /// Distinguished from [`Self::Int8TensorwisePerRow`] by dtype: fp4/mxfp4 packs
    /// two nibbles per `U8` byte and carries **no** `I8` weight tensor.
    ComfyQuantPacked,
    /// ComfyUI `comfy_quant` **int8 tensorwise** (per-row). Despite riding the same
    /// `.comfy_quant` marker as [`Self::ComfyQuantPacked`], this convention stores
    /// each quantized Linear's weight as a plain `I8` tensor with an `F32`
    /// `.weight_scale` sibling — its `.comfy_quant` descriptor blob is
    /// `{"format":"int8_tensorwise","per_row":true}`. Told apart from the fp4 bucket
    /// **header-only, by dtype**: int8 carries a bulk of `I8` weight tensors, fp4
    /// carries none (it packs nibbles into `U8`). Both also carry bulk `U8`
    /// (int8 stores its `.comfy_quant` descriptors as small `U8` blobs), so `U8`
    /// alone cannot separate them — the `I8` weight dtype is the decisive signal.
    /// Loaded by Krea's descriptor-gated single-file loader (sc-14023); split out
    /// here (sc-14026) so the int8 Krea variant
    /// (`~/models/kreamania_variant4.safetensors`) is not mislabelled as the
    /// unloadable fp4 bucket.
    Int8TensorwisePerRow,
    /// GGUF container (`Q8_0`, `Q4_K_S`, …) — detected by the `GGUF` magic, not the
    /// extension. Has no safetensors header; family/component are read from GGUF
    /// metadata by the loader slice, not here.
    Gguf,
    /// `F8_E4M3` (or otherwise packed) weights that carry **no recognized scale
    /// marker** — fp8 tensors mixed with bulk `U8` companions or `F32`/`BF16`
    /// blocks under keys that match none of the conventions above. Explicitly
    /// **not** plain-castable: emitted so the downstream fails closed rather than
    /// casting to noise. (No file in the surveyed tree lands here — every real fp8
    /// file carries one of the four markers above — but it is the safe default for
    /// an unfamiliar scaled/packed export.)
    UnrecognizedScaling,
}

impl QuantFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::ScaledFp8Companion => "scaled_fp8_companion",
            Self::Fp8InlineScale => "fp8_inline_scale",
            Self::ComfyQuantPacked => "comfy_quant_packed",
            Self::Int8TensorwisePerRow => "int8_tensorwise_per_row",
            Self::Gguf => "gguf",
            Self::UnrecognizedScaling => "unrecognized_scaling",
        }
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A confident classification of a base-weight file. `family` is `None` when the
/// component role is known but the architecture is not (e.g. a VAE or text
/// encoder whose exact family the assembler pairs by the transformer's
/// requirement, or an unfamiliar DiT); the verdict is still usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseWeightVerdict {
    pub family: Option<String>,
    pub component: ComponentRole,
    pub quant: QuantFormat,
}

/// The typed outcome of classifying a base-weight file — never a bare guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseWeightDetection {
    Recognized(BaseWeightVerdict),
    /// The file parsed but matched no component-role signature. `reason` records
    /// what was seen so the surface can say *why* it is unusable (the sc-10509
    /// fail-closed-with-reason posture).
    Unrecognized {
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Import compatibility (sc-14019, epic 14015)
// ---------------------------------------------------------------------------

/// Architecture families the single-file **model-import** pipeline can assemble and load today
/// (sc-14019, epic 14015). An imported checkpoint of one of these families is registered as a
/// user model and routed to that family's existing in-process engine (the S0d family-routing path
/// in `jobs_store::routing::catalog`). Seeded with `krea_2` — the community Krea 2 DiT export the
/// detector recognizes by its `txtfusion.` marker, whose builtins already route to the Krea MLX
/// engine. Grows one entry per landed loader; keep it aligned with [`import_supported`]'s arms and
/// with `MLX_ROUTED_FAMILIES` (routing).
pub const IMPORT_SUPPORTED_FAMILIES: &[&str] = &["krea_2"];

/// Whether an imported community single-file **base checkpoint** described by `verdict` can be
/// assembled and run by a real engine today (sc-14019, epic 14015) — the compatibility gate behind
/// the model-import kill-switch (`apps/rust-api::model_import_enabled`). `Ok(())` means the
/// `(family, component, quant)` triple has a landed single-file import loader; `Err(reason)` is a
/// client-facing explanation of why the file is refused. There is **no silent fallback**: a triple
/// with no loader fails closed with a reason (the sc-10509 posture), never a best-effort load that
/// would decode to noise.
///
/// Written as a `match` with **one arm per supported loader** so a follow-on family / component /
/// quant (the S0c assembly slice and the epic's later families) is a single added arm, not a
/// rewrite. Today two triples are loadable: a Krea 2 transformer in either dense `bf16` or
/// descriptor-gated [`QuantFormat::Int8TensorwisePerRow`], both of which reuse the existing Krea
/// engine via the family-routing path. Everything else — an unrecognized/absent family, a
/// non-transformer component (VAE / text encoder / all-in-one checkpoint), or a deferred quant
/// (plain/scaled/inline fp8, `comfy_quant` fp4-packed, GGUF) — is refused with a specific reason.
pub fn import_supported(verdict: &BaseWeightVerdict) -> Result<(), String> {
    match (verdict.family.as_deref(), verdict.component, verdict.quant) {
        // --- Supported loaders: one arm per landed loader (add the next family/quant here) ---
        (
            Some("krea_2"),
            ComponentRole::Transformer,
            QuantFormat::Bf16 | QuantFormat::Int8TensorwisePerRow,
        ) => Ok(()),

        // --- Refusals: most specific first, each with an actionable, client-facing reason ---
        (None, _, _) => Err(
            "The architecture family could not be identified from the file, so the import is \
             refused rather than guessing at a loader."
                .to_owned(),
        ),
        (Some(family), _, _) if !IMPORT_SUPPORTED_FAMILIES.contains(&family) => Err(format!(
            "Model import does not yet support the '{family}' family. Supported today: {}.",
            IMPORT_SUPPORTED_FAMILIES.join(", ")
        )),
        (Some(family), component, _) if component != ComponentRole::Transformer => Err(format!(
            "Model import for the '{family}' family currently supports only the diffusion \
             transformer, not a {component} file."
        )),
        (Some(family), _, quant) => Err(format!(
            "Model import for the '{family}' family currently supports only dense bf16 or \
             descriptor-gated int8-per-row weights, not {quant}. Re-export the checkpoint in bf16."
        )),
    }
}

/// [`import_supported`] lifted over a whole [`BaseWeightDetection`]: a `Recognized` verdict defers to
/// `import_supported`, while an `Unrecognized` file is refused carrying the detector's own reason.
/// The single entry point the API/worker import gates call over a detected file (sc-14019).
pub fn import_detection_supported(detection: &BaseWeightDetection) -> Result<(), String> {
    match detection {
        BaseWeightDetection::Recognized(verdict) => import_supported(verdict),
        BaseWeightDetection::Unrecognized { reason } => Err(format!(
            "The file is not a recognized base-weight checkpoint ({reason}), so it cannot be \
             imported."
        )),
    }
}

/// The GGUF container magic — the first four bytes of every `.gguf` file. Detected
/// by content, not extension, per the story (a renamed `.bin`/`.sft` GGUF must
/// still classify).
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// True when `path` begins with the [`GGUF_MAGIC`]. A read error (missing/locked
/// file) is treated as "not GGUF" — the caller then attempts the safetensors
/// header and surfaces any real I/O error there.
pub fn is_gguf_file(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic).is_ok() && &magic == GGUF_MAGIC
}

/// Classify a base-weight file at `path`. GGUF is detected by magic first (it has
/// no safetensors header); everything else is classified from its safetensors
/// header alone via [`classify_base_header`].
pub fn detect_base_weight_file(path: &Path) -> Result<BaseWeightDetection, SafetensorsHeaderError> {
    if is_gguf_file(path) {
        return Ok(BaseWeightDetection::Recognized(BaseWeightVerdict {
            family: None,
            component: ComponentRole::Checkpoint,
            quant: QuantFormat::Gguf,
        }));
    }
    let header = read_safetensors_header(path)?;
    Ok(classify_base_header(&header))
}

/// Classify a parsed safetensors header. Pure over the header `Value` (tensor
/// name → `{dtype, shape, data_offsets}` map) so it is unit-testable without a
/// file on disk.
pub fn classify_base_header(header: &Value) -> BaseWeightDetection {
    let Some(entries) = header.as_object() else {
        return BaseWeightDetection::Unrecognized {
            reason: "safetensors header is not a JSON object".to_owned(),
        };
    };

    let mut keys: Vec<&str> = Vec::with_capacity(entries.len());
    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    for (name, tensor) in entries {
        if name == "__metadata__" {
            continue;
        }
        keys.push(name.as_str());
        if let Some(dtype) = tensor.get("dtype").and_then(Value::as_str) {
            *dtypes.entry(dtype.to_ascii_uppercase()).or_default() += 1;
        }
    }

    if keys.is_empty() {
        return BaseWeightDetection::Unrecognized {
            reason: "safetensors header declares no tensors".to_owned(),
        };
    }

    let quant = detect_quant_format(&keys, &dtypes);
    let component = detect_component_role(&keys);
    let family = detect_base_family(&keys);

    match component {
        Some(component) => BaseWeightDetection::Recognized(BaseWeightVerdict {
            family,
            component,
            quant,
        }),
        None => BaseWeightDetection::Unrecognized {
            reason: format!(
                "no recognized component-role signature (quant={quant}, {} tensors, dtypes={})",
                keys.len(),
                dtype_summary(&dtypes),
            ),
        },
    }
}

fn dtype_summary(dtypes: &BTreeMap<String, usize>) -> String {
    dtypes
        .iter()
        .map(|(name, count)| format!("{name}×{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// True when any tensor key contains `needle`.
fn any_key_contains(keys: &[&str], needle: &str) -> bool {
    keys.iter().any(|key| key.contains(needle))
}

// ---------------------------------------------------------------------------
// Quant format
// ---------------------------------------------------------------------------

/// Classify the on-disk quant convention from marker keys, then dtypes.
///
/// **Marker keys win over dtypes**, in strict precedence, because several
/// conventions share the `F8_E4M3` dtype and are separable only by the scale
/// tensors that ride alongside:
///
/// 1. `.comfy_quant` present → split by dtype: bulk `I8` weights ⇒ int8-tensorwise
///    per-row ([`QuantFormat::Int8TensorwisePerRow`], a loadable-in-principle quant);
///    otherwise fp4/mxfp4-packed `U8` nibbles ([`QuantFormat::ComfyQuantPacked`]).
/// 2. a top-level `scaled_fp8` marker or a `.scale_weight` companion →
///    [`QuantFormat::ScaledFp8Companion`] (wan / Kijai / umt5).
/// 3. a `.weight_scale`/`.input_scale` inline triplet →
///    [`QuantFormat::Fp8InlineScale`] (flux2).
///
/// Only if none matches do dtypes decide. `scale_shift` keys are deliberately
/// ignored: `scale_shift_table` is adaLN modulation (a real model weight in every
/// PixArt/LTX-style DiT), **not** a quant scale.
fn detect_quant_format(keys: &[&str], dtypes: &BTreeMap<String, usize>) -> QuantFormat {
    let count = |name: &str| dtypes.get(name).copied().unwrap_or(0);
    // `U8` holds packed nibbles when it appears in bulk; a stray one or two are
    // tokenizer bytes (`spiece_model`, `tekken_model`) or `I64` bookkeeping
    // (`num_batches_tracked`) that carry no numeric weight signal.
    let packed_u8 = count("U8") > 4;
    // Bulk `I8` weight tensors are the header-only signal that a `.comfy_quant`
    // file is int8-tensorwise-per-row rather than fp4/mxfp4-packed (sc-14026). The
    // int8 export stores each quantized Linear's weight as `I8` (variant4: 264 of
    // them); a genuine fp4 file packs nibbles into `U8` and carries no `I8` weight.
    // A `>4` floor (matching `packed_u8`) shrugs off a stray bookkeeping `I8`.
    let int8_bulk = count("I8") > 4;

    if any_key_contains(keys, ".comfy_quant") || keys.contains(&"comfy_quant") {
        // Both int8-tensorwise and fp4-packed ride the `.comfy_quant` marker (and
        // both carry bulk `U8` — int8's are its per-Linear descriptor blobs), so
        // the marker alone can't separate them. The `I8` weight dtype is decisive:
        // present ⇒ loadable int8-per-row (sc-14023 loader); absent ⇒ fp4 reject.
        if int8_bulk {
            return QuantFormat::Int8TensorwisePerRow;
        }
        return QuantFormat::ComfyQuantPacked;
    }
    let has_companion_scale = any_key_contains(keys, ".scale_weight")
        || any_key_contains(keys, ".scale_input")
        || keys.contains(&"scaled_fp8");
    if has_companion_scale {
        return QuantFormat::ScaledFp8Companion;
    }
    if any_key_contains(keys, ".weight_scale") || any_key_contains(keys, ".input_scale") {
        return QuantFormat::Fp8InlineScale;
    }

    let fp8 = count("F8_E4M3") + count("F8E4M3") + count("FLOAT8_E4M3FN");
    let bf16 = count("BF16");
    let f16 = count("F16") + count("FLOAT16");
    let f32 = count("F32") + count("FLOAT32");

    if fp8 > 0 {
        // Plain, castable fp8 is *only* a file whose weights are entirely fp8 with
        // nothing packed alongside. Any fp8 mixed with bulk U8 (LTX-2.3) or with a
        // scale scheme we could not name is NOT plain — fail closed rather than
        // cast to noise.
        if !packed_u8 && bf16 == 0 && f16 == 0 {
            return QuantFormat::Fp8E4m3;
        }
        return QuantFormat::UnrecognizedScaling;
    }
    if bf16 > 0 {
        return QuantFormat::Bf16;
    }
    if f16 > 0 {
        return QuantFormat::F16;
    }
    if f32 > 0 {
        return QuantFormat::F32;
    }
    QuantFormat::UnrecognizedScaling
}

// ---------------------------------------------------------------------------
// Component role
// ---------------------------------------------------------------------------

fn detect_component_role(keys: &[&str]) -> Option<ComponentRole> {
    let transformer = has_transformer_signature(keys);
    let vae = has_vae_signature(keys);
    let text = has_text_encoder_signature(keys);

    if transformer && (vae || text) {
        return Some(ComponentRole::Checkpoint);
    }
    if transformer {
        return Some(ComponentRole::Transformer);
    }
    if vae {
        return Some(ComponentRole::Vae);
    }
    if text {
        return Some(ComponentRole::TextEncoder);
    }
    None
}

/// A diffusion backbone if any known DiT family signature matches (see
/// [`detect_base_family`]) — reusing family detection keeps the two in lockstep so
/// a family we can name never fails the role check.
fn has_transformer_signature(keys: &[&str]) -> bool {
    detect_transformer_family(keys).is_some()
}

/// A VAE carries paired encoder/decoder conv stacks. All three tree conventions —
/// BFL ldm (`decoder.up.` / `encoder.down.` / `mid.attn`), diffusers
/// (`up_blocks`/`down_blocks`/`mid_block`+`resnets`), and Wan/Qwen 3D
/// (`upsamples`/`downsamples`/`middle`/`residual`) — share the `encoder.` +
/// `decoder.` + conv shape. Requiring *both* an encoder and a decoder conv keeps a
/// T5 text encoder (which has `encoder.block.` but no decoder) out.
fn has_vae_signature(keys: &[&str]) -> bool {
    let has_decoder_conv = any_key_contains(keys, "decoder.conv")
        || any_key_contains(keys, "decoder.up")
        || any_key_contains(keys, "decoder.upsamples")
        || any_key_contains(keys, "decoder.middle")
        || any_key_contains(keys, "decoder.mid");
    let has_encoder_conv = any_key_contains(keys, "encoder.conv")
        || any_key_contains(keys, "encoder.down")
        || any_key_contains(keys, "encoder.downsamples")
        || any_key_contains(keys, "encoder.middle")
        || any_key_contains(keys, "encoder.mid");
    has_decoder_conv && has_encoder_conv
}

/// A prompt text encoder: an LLM decoder (`model.embed_tokens` + `model.layers.*`
/// `q_proj`/`gate_proj`) or a T5-style encoder (`shared` embedding +
/// `encoder.block.*.SelfAttention`). The tokenizer-blob markers `spiece_model` /
/// `tekken_model` are corroborating but not required.
fn has_text_encoder_signature(keys: &[&str]) -> bool {
    let llm = any_key_contains(keys, "embed_tokens")
        && (any_key_contains(keys, ".self_attn.q_proj")
            || any_key_contains(keys, ".mlp.gate_proj")
            || any_key_contains(keys, ".mlp.down_proj"));
    let t5 = keys.contains(&"shared.weight")
        && any_key_contains(keys, "encoder.block.")
        && any_key_contains(keys, "SelfAttention");
    llm || t5
}

// ---------------------------------------------------------------------------
// Architecture family
// ---------------------------------------------------------------------------

/// Best-effort architecture family. Transformer/checkpoint families are load-
/// critical (they pick the remap table) and detected precisely; VAE and text-
/// encoder families are informational (the assembler pairs those by the
/// transformer's requirement) and left `None` when unknown.
fn detect_base_family(keys: &[&str]) -> Option<String> {
    if let Some(family) = detect_transformer_family(keys) {
        return Some(family.to_owned());
    }
    detect_encoder_or_vae_family(keys).map(str::to_owned)
}

/// The diffusion-backbone family, or `None`. Ordered by unique-marker specificity;
/// each family here carries a tensor-name segment that appears in no other family
/// we ship, so one hit is decisive (mirroring the LoRA detector's
/// `detect_unique_key_family` posture).
fn detect_transformer_family(keys: &[&str]) -> Option<&'static str> {
    // Z-Image (epic 1408): `context_refiner`/`noise_refiner`/`cap_embedder` are
    // unique. Shares the bare `layers.N.attention.qkv` fused-QKV layout with
    // Ideogram, so it must be checked by its own refiner markers first.
    if any_key_contains(keys, "noise_refiner.")
        || any_key_contains(keys, "context_refiner.")
        || any_key_contains(keys, "cap_embedder.")
    {
        return Some("z-image");
    }
    // Krea 2 (epic 8588): the ComfyUI-native MMDiT export carries a unique
    // `txtfusion.` text-fusion tower (`txtfusion.{layerwise,refiner}_blocks`,
    // `txtfusion.projector`) alongside BFL-style `blocks.N.attn.{wq,wk,wv,wo}`,
    // `qknorm`, and `mod.lin`. `txtfusion.` appears in no other family we ship, so
    // one hit is decisive. This classifies the community single-file DiT export
    // (`model.diffusion_model.*`, native keys) — the diffusers-key snapshot layout
    // (`transformer_blocks.*`) is loaded by the Krea snapshot path, not here.
    if any_key_contains(keys, "txtfusion.") {
        return Some("krea_2");
    }
    // Ideogram 4 (epic 6561): single-stream `layers.N.attention.qkv` +
    // `adaln_modulation` (lowercase) + `feed_forward.w`, with the unique
    // `embed_image_indicator` / `llm_cond_proj` / `adaln_proj` head keys.
    if any_key_contains(keys, "embed_image_indicator")
        || any_key_contains(keys, "llm_cond_proj")
        || any_key_contains(keys, "adaln_proj.")
    {
        return Some("ideogram");
    }
    // LTX-Video 2.3 (epic 5481/5495): PixArt-style `transformer_blocks` +
    // `adaln_single` + `scale_shift_table`, with the unique audio-video
    // `audio_embeddings_connector` / `audio_patchify_proj` / `patchify_proj`.
    if any_key_contains(keys, "audio_embeddings_connector")
        || any_key_contains(keys, "patchify_proj")
        || (any_key_contains(keys, "scale_shift_table")
            && any_key_contains(keys, "transformer_blocks"))
    {
        return Some("ltx-video");
    }
    // FLUX.2 (epic 6564): shared-modulation tensors across all blocks.
    if any_key_contains(keys, "double_stream_modulation_")
        || any_key_contains(keys, "single_stream_modulation.")
    {
        return Some("flux2");
    }
    // Anima (epic 10512): Cosmos-Predict2 `diffusion_model.blocks.` with the
    // Cosmos triple `adaln_modulation_{self_attn,cross_attn,mlp}`; checked before
    // Wan because it shares Wan's `blocks.`/`self_attn`/`cross_attn` prefix.
    if any_key_contains(keys, "adaln_modulation_self_attn")
        || any_key_contains(keys, "adaln_modulation_cross_attn")
        || any_key_contains(keys, "adaln_modulation_mlp")
    {
        return Some("anima");
    }
    // Wan 2.x (epic 5095): `blocks.N.{self_attn,cross_attn,ffn}` + per-block
    // `modulation`. The `.ffn.` + bare `modulation` pairing separates it from
    // Anima (adaln_modulation) and from LTX (attn1/attn2, no ffn).
    if any_key_contains(keys, ".self_attn.")
        && any_key_contains(keys, ".cross_attn.")
        && any_key_contains(keys, ".ffn.")
        && any_key_contains(keys, "blocks.")
    {
        return Some("wan-video");
    }
    // FLUX.1 and derivatives (longcat edit, chroma): double+single blocks with
    // `img_mod`/`txt_mod` per-block modulation (no shared-modulation tensors —
    // that is FLUX.2, handled above).
    if any_key_contains(keys, "double_blocks.")
        && any_key_contains(keys, "single_blocks.")
        && (any_key_contains(keys, "img_mod") || any_key_contains(keys, "txt_mod"))
    {
        return Some("flux");
    }
    // Dual-stream MMDiT with `img_mlp`/`txt_mlp` + joint attention `add_q_proj`.
    // Among the families we ship as a single-file base this is Qwen-Image /
    // Qwen-Image-Edit. SD3's `ff_context`/`context_embedder` layout is excluded so
    // an SD3 checkpoint is not mislabelled qwen.
    if any_key_contains(keys, "transformer_blocks.")
        && any_key_contains(keys, "add_q_proj")
        && (any_key_contains(keys, ".img_mlp.") || any_key_contains(keys, ".txt_mlp."))
        && !any_key_contains(keys, "context_embedder")
        && !any_key_contains(keys, ".ff_context.")
    {
        return Some("qwen-image");
    }
    None
}

/// Best-effort family label for text encoders and VAEs — informational only.
/// Returns `None` freely; several of these architectures are byte-identical
/// across families (the Wan and Qwen 3D VAEs share every key), so over-claiming
/// would be worse than `None`.
fn detect_encoder_or_vae_family(keys: &[&str]) -> Option<&'static str> {
    // T5 / UMT5 text encoder.
    if keys.contains(&"shared.weight") && any_key_contains(keys, "encoder.block.") {
        return Some("t5");
    }
    // Mistral-3 (flux2's text encoder): the `tekken` tokenizer + a `vision_tower`.
    if any_key_contains(keys, "tekken_model") || any_key_contains(keys, "vision_tower.") {
        return Some("mistral");
    }
    // Gemma-3: a `vision_model` sibling + Gemma's `mm_soft_emb_norm` projector.
    if any_key_contains(keys, "mm_soft_emb_norm") || any_key_contains(keys, "mm_input_projection") {
        return Some("gemma");
    }
    // Qwen3 LLM text encoder: per-head `self_attn.q_norm`/`k_norm` (Qwen3-specific)
    // with no vision tower.
    if any_key_contains(keys, "embed_tokens")
        && any_key_contains(keys, ".self_attn.q_norm.")
        && any_key_contains(keys, ".self_attn.k_norm.")
    {
        return Some("qwen3");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a safetensors-shaped header from `(name, dtype)` pairs. Only the
    /// fields the classifier reads (`dtype`) are populated; shapes/offsets are
    /// omitted because [`classify_base_header`] never inspects them.
    fn header(entries: &[(&str, &str)]) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        for (name, dtype) in entries {
            map.insert((*name).to_owned(), json!({ "dtype": dtype }));
        }
        Value::Object(map)
    }

    fn recognized(detection: BaseWeightDetection) -> BaseWeightVerdict {
        match detection {
            BaseWeightDetection::Recognized(verdict) => verdict,
            BaseWeightDetection::Unrecognized { reason } => {
                panic!("expected Recognized, got Unrecognized: {reason}")
            }
        }
    }

    // --- component role + family: the DiT prototype targets --------------------

    #[test]
    fn z_image_turbo_bf16_is_transformer() {
        // unet/z_image_turbo_bf16.safetensors (measured: 453 tensors, all BF16).
        let verdict = recognized(classify_base_header(&header(&[
            ("cap_embedder.0.weight", "BF16"),
            ("noise_refiner.0.attention.qkv.weight", "BF16"),
            ("context_refiner.0.attention.qkv.weight", "BF16"),
            ("layers.0.attention.qkv.weight", "BF16"),
            ("layers.0.attention.q_norm.weight", "BF16"),
            ("layers.0.feed_forward.w1.weight", "BF16"),
            ("layers.0.adaLN_modulation.1.weight", "BF16"),
            ("x_embedder.weight", "BF16"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("z-image"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Bf16);
    }

    #[test]
    fn krea2_native_bf16_single_file_is_transformer() {
        // ~/models/kreamania_variant5.safetensors (measured: 430 tensors, DiT-only,
        // 415 BF16 + 15 F32, ComfyUI-native keys, no quant markers) — the dense-bf16
        // community Krea 2 checkpoint that anchors the single-file import skeleton.
        let verdict = recognized(classify_base_header(&header(&[
            ("model.diffusion_model.blocks.0.attn.wq.weight", "BF16"),
            ("model.diffusion_model.blocks.0.attn.wk.weight", "BF16"),
            (
                "model.diffusion_model.blocks.0.attn.qknorm.qnorm.scale",
                "BF16",
            ),
            ("model.diffusion_model.blocks.0.mlp.gate.weight", "BF16"),
            ("model.diffusion_model.blocks.0.mod.lin", "BF16"),
            (
                "model.diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight",
                "BF16",
            ),
            ("model.diffusion_model.txtfusion.projector.weight", "F32"),
            ("model.diffusion_model.first.weight", "F32"),
            ("model.diffusion_model.last.linear.weight", "F32"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("krea_2"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Bf16);
    }

    #[test]
    fn krea2_native_int8_single_file_is_int8_tensorwise_per_row() {
        // ~/models/kreamania_variant4.safetensors (measured: 958 tensors, DiT-only,
        // int8 per-row `{"format":"int8_tensorwise","per_row":true}`: 264 `I8` weight
        // tensors + F32 `.weight_scale` siblings + 264 small `U8` `.comfy_quant`
        // descriptors). Family/component classify the same as the bf16 sibling; the quant
        // must resolve to `Int8TensorwisePerRow` (a loadable-in-principle quant, loader in
        // sc-14023) — NOT the fp4 `ComfyQuantPacked` reject bucket it was mislabelled as
        // before sc-14026. The header-only discriminator is the bulk of `I8` weights (both
        // conventions carry `.comfy_quant` and bulk `U8`, so those cannot separate them).
        let mut entries = vec![
            ("model.diffusion_model.blocks.0.mod.lin", "BF16"),
            ("model.diffusion_model.txtfusion.projector.weight", "F32"),
        ];
        // Bulk I8 quantized weights, each with an F32 `.weight_scale` and a small U8
        // `.comfy_quant` descriptor — the variant4 shape at reduced scale.
        let i8_weights: Vec<String> = (0..6)
            .map(|i| format!("model.diffusion_model.blocks.{i}.attn.wq.weight"))
            .collect();
        let scales: Vec<String> = (0..6)
            .map(|i| format!("model.diffusion_model.blocks.{i}.attn.wq.weight_scale"))
            .collect();
        let descriptors: Vec<String> = (0..6)
            .map(|i| format!("model.diffusion_model.blocks.{i}.attn.wq.comfy_quant"))
            .collect();
        for name in &i8_weights {
            entries.push((name.as_str(), "I8"));
        }
        for name in &scales {
            entries.push((name.as_str(), "F32"));
        }
        for name in &descriptors {
            entries.push((name.as_str(), "U8"));
        }
        let verdict = recognized(classify_base_header(&header(&entries)));
        assert_eq!(verdict.family.as_deref(), Some("krea_2"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Int8TensorwisePerRow);
    }

    #[test]
    fn krea2_comfy_quant_fp4_packed_u8_stays_packed() {
        // The fp4/mxfp4 counterpart to variant4, same krea_2 family: a `.comfy_quant`
        // export whose quantized weights are packed nibbles in bulk `U8` (plus F32
        // `.weight_scale` block scales) and carry **no** `I8` weight tensor. Must stay
        // `ComfyQuantPacked` (the unloadable reject bucket) — proving the sc-14026
        // discriminator flips only on the `I8` weight dtype, not on the shared
        // `.comfy_quant`/`U8` signals. (Real fp4 files, e.g. `gemma_3_12B_it_fp4_mixed`,
        // store weights this way; the F8_E4M3-modelled gemma/ideogram tests below cover
        // the same reject verdict from the other observed dtype.)
        let mut entries = vec![
            ("model.diffusion_model.blocks.0.mod.lin", "BF16"),
            ("model.diffusion_model.txtfusion.projector.weight", "F32"),
        ];
        let u8_weights: Vec<String> = (0..6)
            .map(|i| format!("model.diffusion_model.blocks.{i}.attn.wq.weight"))
            .collect();
        let scales: Vec<String> = (0..6)
            .map(|i| format!("model.diffusion_model.blocks.{i}.attn.wq.weight_scale"))
            .collect();
        let descriptors: Vec<String> = (0..6)
            .map(|i| format!("model.diffusion_model.blocks.{i}.attn.wq.comfy_quant"))
            .collect();
        for name in &u8_weights {
            entries.push((name.as_str(), "U8"));
        }
        for name in &scales {
            entries.push((name.as_str(), "F32"));
        }
        for name in &descriptors {
            entries.push((name.as_str(), "U8"));
        }
        let verdict = recognized(classify_base_header(&header(&entries)));
        assert_eq!(verdict.family.as_deref(), Some("krea_2"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::ComfyQuantPacked);
    }

    #[test]
    fn qwen_image_plain_fp8_is_transformer() {
        // diffusion_models/qwen_image_2512_fp8_e4m3fn (measured: 1933 tensors, all F8_E4M3).
        let verdict = recognized(classify_base_header(&header(&[
            ("model.diffusion_model.img_in.weight", "F8_E4M3"),
            (
                "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
                "F8_E4M3",
            ),
            (
                "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
                "F8_E4M3",
            ),
            (
                "model.diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight",
                "F8_E4M3",
            ),
            (
                "model.diffusion_model.transformer_blocks.0.txt_mlp.net.0.proj.weight",
                "F8_E4M3",
            ),
            (
                "model.diffusion_model.transformer_blocks.0.img_mod.1.weight",
                "F8_E4M3",
            ),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("qwen-image"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Fp8E4m3);
    }

    #[test]
    fn wan_fp8_scaled_is_companion_scaled() {
        // unet/wan2.2_t2v_high_noise_14B_fp8_scaled (measured: scale_weight+scale_input+scaled_fp8).
        let verdict = recognized(classify_base_header(&header(&[
            ("scaled_fp8", "F8_E4M3"),
            ("blocks.0.self_attn.q.weight", "F8_E4M3"),
            ("blocks.0.self_attn.q.scale_weight", "F32"),
            ("blocks.0.self_attn.q.scale_input", "F32"),
            ("blocks.0.cross_attn.k.weight", "F8_E4M3"),
            ("blocks.0.ffn.0.weight", "F8_E4M3"),
            ("blocks.0.modulation", "F16"),
            ("patch_embedding.weight", "F16"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("wan-video"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::ScaledFp8Companion);
    }

    #[test]
    fn wan_kijai_scale_weight_only_is_companion_scaled() {
        // Kijai variant carries `.scale_weight` but no `.scale_input`.
        let verdict = recognized(classify_base_header(&header(&[
            ("blocks.0.self_attn.q.weight", "F8_E4M3"),
            ("blocks.0.self_attn.q.scale_weight", "F32"),
            ("blocks.0.cross_attn.k.weight", "F8_E4M3"),
            ("blocks.0.ffn.0.weight", "F8_E4M3"),
            ("blocks.0.modulation", "F32"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("wan-video"));
        assert_eq!(verdict.quant, QuantFormat::ScaledFp8Companion);
    }

    #[test]
    fn flux2_dev_is_inline_scale() {
        // diffusion_models/flux2_dev_fp8mixed (measured: weight_scale+input_scale, no companion).
        let verdict = recognized(classify_base_header(&header(&[
            ("double_stream_modulation_img.lin.weight", "BF16"),
            ("single_stream_modulation.lin.weight", "BF16"),
            ("double_blocks.0.img_attn.qkv.weight", "BF16"),
            ("double_blocks.0.img_mlp.0.weight", "F8_E4M3"),
            ("double_blocks.0.img_mlp.0.weight_scale", "F32"),
            ("double_blocks.0.img_mlp.0.input_scale", "F32"),
            ("single_blocks.0.linear1.weight", "F8_E4M3"),
            ("single_blocks.0.linear1.weight_scale", "F32"),
            ("single_blocks.0.linear1.input_scale", "F32"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("flux2"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Fp8InlineScale);
    }

    #[test]
    fn ideogram_comfy_quant_packed_wins_over_inline_scale() {
        // diffusion_models/ideogram4_fp8_scaled — has BOTH `.comfy_quant` and
        // `.weight_scale`; comfy_quant must win (it is packed fp4, not inline fp8).
        let verdict = recognized(classify_base_header(&header(&[
            ("embed_image_indicator.weight", "BF16"),
            ("layers.0.attention.qkv.weight", "F8_E4M3"),
            ("layers.0.attention.qkv.comfy_quant", "U8"),
            ("layers.0.attention.qkv.weight_scale", "F32"),
            ("layers.0.feed_forward.w1.weight", "F8_E4M3"),
            ("layers.0.feed_forward.w1.comfy_quant", "U8"),
            ("layers.0.adaln_modulation.weight", "F8_E4M3"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("ideogram"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::ComfyQuantPacked);
    }

    #[test]
    fn fp8_with_bulk_u8_and_no_scale_marker_is_unrecognized_not_plain() {
        // Defensive: an fp8 export whose scale/packing rides under keys that match
        // none of the four recognized markers — fp8 mixed with bulk U8 companions.
        // Must NOT be classified as plain-castable fp8 (that would cast to noise);
        // it fails closed as UnrecognizedScaling. (No file in the surveyed tree is
        // actually shaped like this — every real fp8 file carries a marker — but
        // the fallback must stay safe for an unfamiliar future export.)
        // A Wan-shaped DiT so the family/role still resolve.
        let mut entries = vec![
            ("blocks.0.self_attn.q.weight", "F8_E4M3"),
            ("blocks.0.cross_attn.k.weight", "F8_E4M3"),
            ("blocks.0.ffn.0.weight", "F8_E4M3"),
            ("blocks.0.modulation", "BF16"),
        ];
        // Bulk U8 companions under a non-marker key (`.q8`), the packing signal.
        let u8_names: Vec<String> = (0..10)
            .map(|i| format!("blocks.{i}.self_attn.q.q8"))
            .collect();
        for name in &u8_names {
            entries.push((name.as_str(), "U8"));
        }
        let verdict = recognized(classify_base_header(&header(&entries)));
        assert_eq!(verdict.family.as_deref(), Some("wan-video"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::UnrecognizedScaling);
    }

    // --- component role: encoders, VAEs, checkpoints ---------------------------

    #[test]
    fn qwen3_text_encoder_is_text_encoder() {
        // text_encoders/qwen_3_4b (measured: embed_tokens + model.layers + q_norm/k_norm).
        let verdict = recognized(classify_base_header(&header(&[
            ("model.embed_tokens.weight", "BF16"),
            ("model.layers.0.self_attn.q_proj.weight", "BF16"),
            ("model.layers.0.self_attn.q_norm.weight", "BF16"),
            ("model.layers.0.self_attn.k_norm.weight", "BF16"),
            ("model.layers.0.mlp.gate_proj.weight", "BF16"),
            ("model.norm.weight", "BF16"),
        ])));
        assert_eq!(verdict.component, ComponentRole::TextEncoder);
        assert_eq!(verdict.family.as_deref(), Some("qwen3"));
        assert_eq!(verdict.quant, QuantFormat::Bf16);
    }

    #[test]
    fn umt5_scaled_is_text_encoder_companion_scaled() {
        // text_encoders/umt5_xxl_fp8_e4m3fn_scaled (measured: T5 encoder + scale_weight).
        let verdict = recognized(classify_base_header(&header(&[
            ("shared.weight", "F32"),
            ("encoder.block.0.layer.0.SelfAttention.q.weight", "F8_E4M3"),
            (
                "encoder.block.0.layer.0.SelfAttention.q.scale_weight",
                "F32",
            ),
            (
                "encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
                "F8_E4M3",
            ),
            ("encoder.final_layer_norm.weight", "F32"),
        ])));
        assert_eq!(verdict.component, ComponentRole::TextEncoder);
        assert_eq!(verdict.family.as_deref(), Some("t5"));
        assert_eq!(verdict.quant, QuantFormat::ScaledFp8Companion);
    }

    #[test]
    fn gemma_comfy_quant_is_text_encoder_packed() {
        // text_encoders/gemma_3_12B_it_fp4_mixed (measured: embed_tokens + comfy_quant + vision_model).
        let verdict = recognized(classify_base_header(&header(&[
            ("model.embed_tokens.weight", "BF16"),
            ("model.layers.0.self_attn.q_proj.weight", "F8_E4M3"),
            ("model.layers.0.self_attn.q_proj.comfy_quant", "U8"),
            ("model.layers.0.self_attn.q_proj.weight_scale", "F32"),
            ("model.layers.0.mlp.down_proj.weight", "F8_E4M3"),
            ("model.layers.0.mlp.down_proj.comfy_quant", "U8"),
            ("multi_modal_projector.mm_soft_emb_norm.weight", "BF16"),
        ])));
        assert_eq!(verdict.component, ComponentRole::TextEncoder);
        assert_eq!(verdict.family.as_deref(), Some("gemma"));
        assert_eq!(verdict.quant, QuantFormat::ComfyQuantPacked);
    }

    #[test]
    fn flux_ldm_vae_is_vae() {
        // vae/ae.safetensors (measured: encoder.down/decoder.up + mid.attn, all F32).
        let verdict = recognized(classify_base_header(&header(&[
            ("encoder.conv_in.weight", "F32"),
            ("encoder.down.0.block.0.conv1.weight", "F32"),
            ("encoder.mid.attn_1.q.weight", "F32"),
            ("decoder.conv_out.weight", "F32"),
            ("decoder.up.0.block.0.conv1.weight", "F32"),
            ("decoder.mid.attn_1.q.weight", "F32"),
        ])));
        assert_eq!(verdict.component, ComponentRole::Vae);
        assert_eq!(verdict.quant, QuantFormat::F32);
    }

    #[test]
    fn wan_3d_vae_is_vae() {
        // vae/wan_2.1_vae (measured: encoder.downsamples/decoder.upsamples/middle, all BF16).
        let verdict = recognized(classify_base_header(&header(&[
            ("encoder.conv1.weight", "BF16"),
            ("encoder.downsamples.0.residual.0.weight", "BF16"),
            ("encoder.middle.0.to_qkv.weight", "BF16"),
            ("decoder.conv1.weight", "BF16"),
            ("decoder.upsamples.0.residual.0.weight", "BF16"),
            ("decoder.middle.0.to_qkv.weight", "BF16"),
        ])));
        assert_eq!(verdict.component, ComponentRole::Vae);
        assert_eq!(verdict.quant, QuantFormat::Bf16);
    }

    #[test]
    fn ltx_checkpoint_is_all_in_one_checkpoint() {
        // checkpoints/ltx-2.3-22b-dev-fp8 (measured: audio_vae.* + model.diffusion_model.*).
        let verdict = recognized(classify_base_header(&header(&[
            ("model.diffusion_model.scale_shift_table", "F32"),
            ("model.diffusion_model.patchify_proj.weight", "BF16"),
            (
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                "F8_E4M3",
            ),
            ("audio_vae.encoder.conv_in.conv.weight", "F32"),
            ("audio_vae.decoder.conv_out.conv.weight", "F32"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("ltx-video"));
        assert_eq!(verdict.component, ComponentRole::Checkpoint);
    }

    // --- typed-negative + quant edge cases -------------------------------------

    #[test]
    fn unknown_architecture_is_unrecognized_with_reason() {
        let detection = classify_base_header(&header(&[
            ("some.mystery.tensor", "BF16"),
            ("another.mystery.tensor", "BF16"),
        ]));
        match detection {
            BaseWeightDetection::Unrecognized { reason } => {
                assert!(reason.contains("component-role"), "reason: {reason}");
            }
            BaseWeightDetection::Recognized(v) => panic!("expected Unrecognized, got {v:?}"),
        }
    }

    #[test]
    fn empty_header_is_unrecognized() {
        let detection = classify_base_header(&json!({ "__metadata__": {"format": "pt"} }));
        assert!(matches!(
            detection,
            BaseWeightDetection::Unrecognized { .. }
        ));
    }

    #[test]
    fn stray_u8_tokenizer_byte_does_not_defeat_bf16() {
        // mistral_3_small_flux2_bf16: BF16×494 + a single U8 (`tekken_model`).
        let verdict = recognized(classify_base_header(&header(&[
            ("model.embed_tokens.weight", "BF16"),
            ("model.layers.0.self_attn.q_proj.weight", "BF16"),
            ("model.layers.0.mlp.gate_proj.weight", "BF16"),
            ("vision_tower.patch_conv.weight", "BF16"),
            ("tekken_model", "U8"),
        ])));
        assert_eq!(verdict.component, ComponentRole::TextEncoder);
        assert_eq!(verdict.family.as_deref(), Some("mistral"));
        assert_eq!(verdict.quant, QuantFormat::Bf16);
    }

    /// End-to-end over the operator's real ComfyUI tree — the sc-10662 real-tree
    /// posture (mirrors Phase 1's `external_loras::tests::real_comfyui_tree`).
    /// Ignored by default (needs the local tree + is slow to enumerate); exercises
    /// the full path: GGUF magic, real safetensors headers, and the classifier.
    ///
    /// ```text
    /// SCENEWORKS_EXTERNAL_MODEL_ROOTS='C:\Users\Michael\ComfyUI-Shared\models' \
    ///   cargo test -p sceneworks-core --lib base_weights::tests::real_comfyui_base_tree -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn real_comfyui_base_tree() {
        use crate::external_roots::EXTERNAL_MODEL_ROOTS_ENV;
        use std::path::PathBuf;

        let root = std::env::var(EXTERNAL_MODEL_ROOTS_ENV)
            .ok()
            .and_then(|raw| std::env::split_paths(&raw).next())
            .expect("set SCENEWORKS_EXTERNAL_MODEL_ROOTS to the ComfyUI models dir");

        // Anchors: (relative path, expected family, component, quant). One per
        // family/quant/role class surveyed for sc-10662.
        let anchors: &[(&str, Option<&str>, ComponentRole, QuantFormat)] = &[
            (
                "unet/z_image_turbo_bf16.safetensors",
                Some("z-image"),
                ComponentRole::Transformer,
                QuantFormat::Bf16,
            ),
            (
                "diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors",
                Some("qwen-image"),
                ComponentRole::Transformer,
                QuantFormat::Fp8E4m3,
            ),
            (
                "unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                Some("wan-video"),
                ComponentRole::Transformer,
                QuantFormat::ScaledFp8Companion,
            ),
            (
                "diffusion_models/flux2_dev_fp8mixed.safetensors",
                Some("flux2"),
                ComponentRole::Transformer,
                QuantFormat::Fp8InlineScale,
            ),
            (
                "diffusion_models/ideogram4_fp8_scaled.safetensors",
                Some("ideogram"),
                ComponentRole::Transformer,
                QuantFormat::ComfyQuantPacked,
            ),
            (
                // Packed: this export carries `.comfy_quant` (+ `.weight_scale`).
                "diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
                Some("ltx-video"),
                ComponentRole::Transformer,
                QuantFormat::ComfyQuantPacked,
            ),
            (
                "text_encoders/qwen_3_4b.safetensors",
                Some("qwen3"),
                ComponentRole::TextEncoder,
                QuantFormat::Bf16,
            ),
            (
                "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                Some("t5"),
                ComponentRole::TextEncoder,
                QuantFormat::ScaledFp8Companion,
            ),
            (
                "text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
                Some("gemma"),
                ComponentRole::TextEncoder,
                QuantFormat::ComfyQuantPacked,
            ),
            (
                "vae/ae.safetensors",
                None,
                ComponentRole::Vae,
                QuantFormat::F32,
            ),
            (
                // Inline-scale: this export carries `.weight_scale`+`.input_scale`.
                "checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
                Some("ltx-video"),
                ComponentRole::Checkpoint,
                QuantFormat::Fp8InlineScale,
            ),
            (
                "unet/wan2.2_t2v_high_noise_14B_Q4_K_S.gguf",
                None,
                ComponentRole::Checkpoint,
                QuantFormat::Gguf,
            ),
        ];

        let mut failures = Vec::new();
        for (rel, family, component, quant) in anchors {
            let path: PathBuf = root.join(rel);
            if !path.exists() {
                println!("SKIP (absent): {rel}");
                continue;
            }
            match detect_base_weight_file(&path) {
                Ok(BaseWeightDetection::Recognized(v)) => {
                    println!(
                        "{rel} -> family={:?} component={} quant={}",
                        v.family, v.component, v.quant
                    );
                    if v.family.as_deref() != *family
                        || v.component != *component
                        || v.quant != *quant
                    {
                        failures.push(format!(
                            "{rel}: got ({:?},{},{}) want ({family:?},{component},{quant})",
                            v.family, v.component, v.quant
                        ));
                    }
                }
                Ok(BaseWeightDetection::Unrecognized { reason }) => {
                    failures.push(format!("{rel}: Unrecognized ({reason})"));
                }
                Err(e) => failures.push(format!("{rel}: header error {e}")),
            }
        }
        assert!(
            failures.is_empty(),
            "real-tree mismatches:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn scale_shift_table_alone_is_not_a_quant_scale() {
        // A pure-BF16 DiT carrying `scale_shift_table` (adaLN) must stay Bf16.
        let verdict = recognized(classify_base_header(&header(&[
            ("patchify_proj.weight", "BF16"),
            ("scale_shift_table", "BF16"),
            ("transformer_blocks.0.attn1.to_q.weight", "BF16"),
            ("transformer_blocks.0.attn2.to_k.weight", "BF16"),
        ])));
        assert_eq!(verdict.quant, QuantFormat::Bf16);
    }

    // --- import compatibility gate (sc-14019, epic 14015) -----------------------

    fn verdict(
        family: Option<&str>,
        component: ComponentRole,
        quant: QuantFormat,
    ) -> BaseWeightVerdict {
        BaseWeightVerdict {
            family: family.map(str::to_owned),
            component,
            quant,
        }
    }

    #[test]
    fn import_supported_accepts_krea2_transformer_bf16_and_int8_per_row() {
        for quant in [QuantFormat::Bf16, QuantFormat::Int8TensorwisePerRow] {
            assert!(
                import_supported(&verdict(Some("krea_2"), ComponentRole::Transformer, quant))
                    .is_ok(),
                "Krea 2 transformer {quant} has a landed single-file loader"
            );
        }
    }

    #[test]
    fn import_supported_refuses_krea2_transformer_deferred_quant() {
        // Same family + component, but a packed quant with no loader → refused with the quant named.
        let reason = import_supported(&verdict(
            Some("krea_2"),
            ComponentRole::Transformer,
            QuantFormat::ComfyQuantPacked,
        ))
        .expect_err("packed quant must be refused");
        assert!(
            reason.contains("bf16") && reason.contains("int8"),
            "reason should name the required quant: {reason}"
        );
        assert!(
            reason.contains(QuantFormat::ComfyQuantPacked.as_str()),
            "reason should name the rejected quant: {reason}"
        );
    }

    #[test]
    fn import_supported_refuses_krea2_wrong_component() {
        // A Krea-family VAE (or any non-transformer component) is not the transformer we load.
        for component in [
            ComponentRole::Vae,
            ComponentRole::TextEncoder,
            ComponentRole::Checkpoint,
        ] {
            let reason = import_supported(&verdict(Some("krea_2"), component, QuantFormat::Bf16))
                .expect_err("non-transformer component must be refused");
            assert!(
                reason.contains("transformer"),
                "reason should explain the transformer-only rule: {reason}"
            );
        }
    }

    #[test]
    fn import_supported_refuses_unsupported_and_absent_family() {
        // A recognized-but-unsupported family (z-image) is refused, naming the supported set.
        let z_reason = import_supported(&verdict(
            Some("z-image"),
            ComponentRole::Transformer,
            QuantFormat::Bf16,
        ))
        .expect_err("unsupported family must be refused");
        assert!(z_reason.contains("z-image"), "reason: {z_reason}");
        assert!(
            z_reason.contains("krea_2"),
            "reason should name the supported set: {z_reason}"
        );
        // A component with no family label (None) is refused rather than guessed at.
        assert!(import_supported(&verdict(
            None,
            ComponentRole::Transformer,
            QuantFormat::Bf16
        ))
        .is_err());
    }

    #[test]
    fn import_detection_supported_refuses_unrecognized_with_reason() {
        let detection = BaseWeightDetection::Unrecognized {
            reason: "no recognized component-role signature".to_owned(),
        };
        let reason =
            import_detection_supported(&detection).expect_err("unrecognized must be refused");
        assert!(
            reason.contains("no recognized component-role signature"),
            "the detector's own reason must be surfaced: {reason}"
        );
    }

    #[test]
    fn import_detection_supported_accepts_recognized_krea2_bf16() {
        let detection = BaseWeightDetection::Recognized(verdict(
            Some("krea_2"),
            ComponentRole::Transformer,
            QuantFormat::Bf16,
        ));
        assert!(import_detection_supported(&detection).is_ok());
    }

    #[test]
    fn import_supported_families_are_a_subset_of_the_ok_arms() {
        // Guardrail: every family the gate advertises must actually have an Ok triple, so the
        // advertised set and the `match` arms can never drift (add the family here + its arm together).
        for family in IMPORT_SUPPORTED_FAMILIES {
            assert!(
                import_supported(&verdict(
                    Some(family),
                    ComponentRole::Transformer,
                    QuantFormat::Bf16
                ))
                .is_ok(),
                "IMPORT_SUPPORTED_FAMILIES lists {family} but no bf16 transformer arm accepts it"
            );
        }
    }
}
