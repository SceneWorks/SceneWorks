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
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::lora_family::{is_hidden_file, read_safetensors_header, SafetensorsHeaderError};

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
    /// NVIDIA NVFP4 as serialized by the ComfyUI Kitchen converter: each quantized Linear is a
    /// U8 `[out,in/2]` E2M1 weight plus an F8_E4M3 blocked `.weight_scale` and scalar F32
    /// `.weight_scale_2`. The converter declares `quant_format=NVFP4` in safetensors metadata.
    /// This is not the descriptor-gated generic [`Self::ComfyQuantPacked`] bucket: the Krea Candle
    /// loader consumes this exact triplet directly and swaps Kitchen's nibble convention losslessly.
    Nvfp4,
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
            Self::Nvfp4 => "nvfp4",
            Self::Gguf => "gguf",
            Self::UnrecognizedScaling => "unrecognized_scaling",
        }
    }

    /// Parse the [`Self::as_str`] spelling back.
    ///
    /// The inverse of `as_str`, needed because the classification is *persisted* — a manifest
    /// entry's `importQuantFormat` is this string, written at import time by the header gate that
    /// proved it, and read back on every later request. Matching is exact and case-sensitive:
    /// these are values this code wrote, not user input, and a lenient parse would let a
    /// near-miss (`"NVFP4"`, `"nvfp4-v1"`) resolve to a classification nothing verified.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "bf16" => Some(Self::Bf16),
            "f16" => Some(Self::F16),
            "f32" => Some(Self::F32),
            "fp8_e4m3" => Some(Self::Fp8E4m3),
            "scaled_fp8_companion" => Some(Self::ScaledFp8Companion),
            "fp8_inline_scale" => Some(Self::Fp8InlineScale),
            "comfy_quant_packed" => Some(Self::ComfyQuantPacked),
            "int8_tensorwise_per_row" => Some(Self::Int8TensorwisePerRow),
            "nvfp4" => Some(Self::Nvfp4),
            "gguf" => Some(Self::Gguf),
            "unrecognized_scaling" => Some(Self::UnrecognizedScaling),
            _ => None,
        }
    }

    /// The engine's stable **source-codec id** for this classification, when the two vocabularies
    /// name the same thing (sc-21484, epic 11037).
    ///
    /// This is the one sanctioned bridge from SceneWorks' header classification to the cross-repo
    /// codec vocabulary in [`crate::checkpoint_weight_facts`]. It is a *lookup on the verified
    /// verdict*, never a derivation from a bit count: [`Self::Nvfp4`] is only ever reached through
    /// `metadata_declares_nvfp4(..) && has_kitchen_nvfp4_triplets(..)`, both halves required, and
    /// the codec id inherits exactly that proof. Nothing here reads `mlxQuantize`, a tier name, or
    /// a dtype width.
    ///
    /// # Why several arms are `None`
    ///
    /// `None` means "this build cannot name the engine codec for that classification", not "the
    /// file is dense". The fp8 family is the reason it exists: SceneWorks tells apart four fp8
    /// conventions by *marker shape* ([`Self::Fp8E4m3`], [`Self::ScaledFp8Companion`],
    /// [`Self::Fp8InlineScale`], [`Self::ComfyQuantPacked`]) while the engine's registered codecs
    /// are keyed by *scale topology* (`fp8-e4m3-scalar-v1`, `mxfp8-v1`, …). The two partitions do
    /// not line up one-to-one, and guessing a mapping would put a codec id that was never proved
    /// into a persisted fact a user reads. A caller that gets `None` must omit the source-codec
    /// fact, never substitute a request tier for it.
    pub fn source_codec_id(self) -> Option<&'static str> {
        use crate::checkpoint_weight_facts as facts;
        match self {
            Self::Bf16 => Some(facts::DENSE_BF16_CODEC_ID),
            Self::F16 => Some(facts::DENSE_F16_CODEC_ID),
            Self::F32 => Some(facts::DENSE_F32_CODEC_ID),
            Self::Int8TensorwisePerRow => Some(facts::INT8_PER_ROW_CODEC_ID),
            Self::Nvfp4 => Some(facts::NVFP4_CODEC_ID),
            Self::Gguf => Some(facts::GGUF_CONTAINER_CODEC_ID),
            Self::Fp8E4m3
            | Self::ScaledFp8Companion
            | Self::Fp8InlineScale
            | Self::ComfyQuantPacked
            | Self::UnrecognizedScaling => None,
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
/// `mage-flow` (sc-15036, epic 14034 F6) is the DIRECTORY-shaped member: unlike the two single-file
/// families, a loadable Mage-Flow backbone is a diffusers **transformer component directory**
/// ([`is_mage_flow_transformer_dir`]) — `config.json` plus its weight file — because the loader
/// reads its architecture from that config rather than inferring it from a base tier. A bare
/// `.safetensors` with no config sibling is therefore refused, which is what the arm below says.
pub const IMPORT_SUPPORTED_FAMILIES: &[&str] = &["krea_2", "mage-flow", "sdxl"];

/// The file a Mage-Flow transformer component directory carries its weights in (the diffusers
/// name the trainer's full-fine-tune writer emits).
pub const MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE: &str = "diffusion_pytorch_model.safetensors";
/// The architecture config a Mage-Flow transformer component directory carries.
pub const MAGE_FLOW_TRANSFORMER_CONFIG_FILE: &str = "config.json";

/// Whether `dir` is a **loadable** Mage-Flow transformer component directory (sc-15036) — the shape
/// a full base fine-tune (sc-14056) emits and the only shape `mlx_gen_mage::load_finetuned` can
/// read: the architecture `config.json` plus the diffusers weight file, both present.
///
/// Shared deliberately by the model-import gate and the worker's `mage_finetuned` render lane, so
/// "what the app will accept" and "what the engine can load" cannot drift apart. A directory that
/// carries only one of the two is a torn or partial artifact and is refused, loudly, before any
/// compute — not probed for at load time.
pub fn is_mage_flow_transformer_dir(dir: &Path) -> bool {
    dir.join(MAGE_FLOW_TRANSFORMER_CONFIG_FILE).is_file()
        && dir.join(MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE).is_file()
}

/// Resolve the one weight file carried by a user-import source without guessing across a model
/// tree. Accepted shapes are deliberately the same structural shapes the imported loaders own:
///
/// - a direct, non-hidden `.safetensors` file;
/// - an exact Mage-Flow transformer directory; or
/// - a flat install directory containing exactly one top-level, non-hidden `.safetensors` file and
///   no diffusers snapshot markers.
///
/// Recursive discovery is intentionally forbidden. A sharded/multi-file/diffusers tree cannot be
/// collapsed to whichever child `read_dir` happened to return first and mislabeled as a single-file
/// import route.
pub fn imported_model_primary_weight_file(source: &Path) -> Option<PathBuf> {
    let is_plain_safetensors_file = |path: &Path| {
        path.extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
            && !is_hidden_file(path)
            && std::fs::symlink_metadata(path)
                .ok()
                .is_some_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
    };
    if is_plain_safetensors_file(source) {
        return Some(source.to_path_buf());
    }
    if !source.is_dir() {
        return None;
    }
    if is_mage_flow_transformer_dir(source) {
        let config = source.join(MAGE_FLOW_TRANSFORMER_CONFIG_FILE);
        let weights = source.join(MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE);
        let config_is_regular = std::fs::symlink_metadata(config)
            .ok()
            .is_some_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink());
        return (config_is_regular && is_plain_safetensors_file(&weights)).then_some(weights);
    }
    if source.join("model_index.json").is_file()
        || source.join("config.json").is_file()
        || source.join("transformer").is_dir()
    {
        return None;
    }
    let mut found = None;
    for entry in std::fs::read_dir(source).ok()?.flatten() {
        let candidate = entry.path();
        if !is_plain_safetensors_file(&candidate) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(candidate);
    }
    found
}

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
/// rewrite. Today a Krea 2 transformer is loadable as dense `bf16`, descriptor-gated
/// [`QuantFormat::Int8TensorwisePerRow`], or Kitchen [`QuantFormat::Nvfp4`]; all three reuse the
/// existing Krea engine via the family-routing path. Everything else — an unrecognized/absent family, a
/// non-transformer component (VAE / text encoder / all-in-one checkpoint), or a deferred quant
/// (plain/scaled/inline fp8, `comfy_quant` fp4-packed, GGUF) — is refused with a specific reason.
pub fn import_supported(verdict: &BaseWeightVerdict) -> Result<(), String> {
    match (verdict.family.as_deref(), verdict.component, verdict.quant) {
        // --- Supported loaders: one arm per landed loader (add the next family/quant here) ---
        (
            Some("krea_2"),
            ComponentRole::Transformer,
            QuantFormat::Bf16 | QuantFormat::Int8TensorwisePerRow | QuantFormat::Nvfp4,
        ) => Ok(()),
        (
            Some("sdxl"),
            ComponentRole::Checkpoint,
            QuantFormat::F16 | QuantFormat::Bf16 | QuantFormat::F32,
        ) => Ok(()),
        // sc-15036: a dense Mage-Flow DiT. Loadable only as a transformer component DIRECTORY —
        // the caller must additionally pass the directory shape through
        // [`is_mage_flow_transformer_dir`]; this triple gate cannot see the config sibling.
        (Some("mage-flow"), ComponentRole::Transformer, QuantFormat::Bf16) => Ok(()),

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
        (Some("krea_2"), component, _) if component != ComponentRole::Transformer => Err(format!(
            "Model import for the 'krea_2' family requires a diffusion transformer, not a \
             {component} file."
        )),
        (Some("sdxl"), component, _) if component != ComponentRole::Checkpoint => Err(format!(
            "Model import for the 'sdxl' family requires a fused checkpoint containing the UNet, \
             both text encoders, and VAE, not a {component} file."
        )),
        (Some("mage-flow"), component, _) if component != ComponentRole::Transformer => {
            Err(format!(
                "Model import for the 'mage-flow' family requires a diffusion transformer, not a \
                 {component} file."
            ))
        }
        (Some("krea_2"), _, quant) => Err(format!(
            "Model import for the 'krea_2' family supports dense bf16, descriptor-gated \
             int8-per-row, or NVIDIA Kitchen NVFP4 weights, not {quant}. Re-export the checkpoint \
             in bf16."
        )),
        (Some("sdxl"), _, quant) => Err(format!(
            "Model import for the 'sdxl' family supports dense f16, bf16, or f32 fused checkpoints, \
             not {quant}."
        )),
        (Some("mage-flow"), _, quant) => Err(format!(
            "Model import for the 'mage-flow' family supports a dense bf16 transformer, not \
             {quant}. A pre-quantized q4/q8 Mage-Flow tier is installed through the model catalog, \
             not imported."
        )),
        (Some(family), _, _) => Err(format!(
            "Model import has no compatible loader arm for the '{family}' family."
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

    let quant = detect_quant_format(entries, &keys, &dtypes);
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
/// 1. explicit Kitchen `quant_format=NVFP4` metadata plus an exact U8/F8/F32 triplet surface →
///    [`QuantFormat::Nvfp4`]. Both pieces are required; metadata alone is never trusted.
/// 2. `.comfy_quant` present → split by dtype: bulk `I8` weights ⇒ int8-tensorwise
///    per-row ([`QuantFormat::Int8TensorwisePerRow`], a loadable-in-principle quant);
///    otherwise fp4/mxfp4-packed `U8` nibbles ([`QuantFormat::ComfyQuantPacked`]).
/// 3. a top-level `scaled_fp8` marker or a `.scale_weight` companion →
///    [`QuantFormat::ScaledFp8Companion`] (wan / Kijai / umt5).
/// 4. a `.weight_scale`/`.input_scale` inline triplet →
///    [`QuantFormat::Fp8InlineScale`] (flux2).
///
/// Only if none matches do dtypes decide. `scale_shift` keys are deliberately
/// ignored: `scale_shift_table` is adaLN modulation (a real model weight in every
/// PixArt/LTX-style DiT), **not** a quant scale.
fn detect_quant_format(
    entries: &serde_json::Map<String, Value>,
    keys: &[&str],
    dtypes: &BTreeMap<String, usize>,
) -> QuantFormat {
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

    if metadata_declares_nvfp4(entries) && has_kitchen_nvfp4_triplets(entries) {
        return QuantFormat::Nvfp4;
    }

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

fn metadata_declares_nvfp4(entries: &serde_json::Map<String, Value>) -> bool {
    entries
        .get("__metadata__")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("quant_format"))
        .and_then(Value::as_str)
        .is_some_and(|format| format.eq_ignore_ascii_case("nvfp4"))
}

fn tensor_header<'a>(
    entries: &'a serde_json::Map<String, Value>,
    name: &str,
) -> Option<(&'a str, Vec<u64>)> {
    let tensor = entries.get(name)?.as_object()?;
    let dtype = tensor.get("dtype")?.as_str()?;
    let shape = tensor
        .get("shape")?
        .as_array()?
        .iter()
        .map(Value::as_u64)
        .collect::<Option<Vec<_>>>()?;
    Some((dtype, shape))
}

fn round_up_u64(value: u64, multiple: u64) -> Option<u64> {
    value
        .checked_add(multiple.checked_sub(1)?)?
        .checked_div(multiple)?
        .checked_mul(multiple)
}

/// Header-only proof of the exact Kitchen NVFP4 tensor surface the Candle Krea loader accepts.
/// The global metadata prevents a coincidental U8 triplet scheme from being mislabeled, while these
/// shape/dtype checks prevent a forged or stale metadata string from opening the loader.
fn has_kitchen_nvfp4_triplets(entries: &serde_json::Map<String, Value>) -> bool {
    // The current Kitchen converter writes no per-layer descriptors. Descriptor-gated NVFP4 belongs
    // to the generic ComfyQuant bucket until its extra key surface has a consumer.
    if entries.keys().any(|name| name.ends_with(".comfy_quant")) {
        return false;
    }
    let u8_weights = entries
        .iter()
        .filter(|(name, tensor)| {
            name.ends_with(".weight")
                && tensor
                    .get("dtype")
                    .and_then(Value::as_str)
                    .is_some_and(|dtype| dtype.eq_ignore_ascii_case("u8"))
        })
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    let scale_2 = entries
        .keys()
        .filter(|name| name.ends_with(".weight_scale_2"))
        .map(String::as_str)
        .collect::<Vec<_>>();
    if u8_weights.len() <= 4 || u8_weights.len() != scale_2.len() {
        return false;
    }

    let valid_weight = |weight_name: &str| {
        let Some(base) = weight_name.strip_suffix(".weight") else {
            return false;
        };
        let Some((weight_dtype, weight_shape)) = tensor_header(entries, weight_name) else {
            return false;
        };
        let [rows, packed_cols] = weight_shape.as_slice() else {
            return false;
        };
        let Some(cols) = packed_cols.checked_mul(2) else {
            return false;
        };
        if !weight_dtype.eq_ignore_ascii_case("u8")
            || *rows == 0
            || cols == 0
            || *rows % 16 != 0
            || cols % 16 != 0
        {
            return false;
        }

        let Some((block_dtype, block_shape)) =
            tensor_header(entries, &format!("{base}.weight_scale"))
        else {
            return false;
        };
        let block_values = block_shape
            .iter()
            .try_fold(1u64, |product, dim| product.checked_mul(*dim));
        let expected_blocks = round_up_u64(*rows, 128).and_then(|scale_rows| {
            round_up_u64(cols / 16, 4).and_then(|scale_cols| scale_rows.checked_mul(scale_cols))
        });
        if !matches!((block_values, expected_blocks), (Some(actual), Some(expected)) if actual == expected)
            || !matches!(
                block_dtype.to_ascii_uppercase().as_str(),
                "F8_E4M3" | "F8E4M3" | "FLOAT8_E4M3FN"
            )
        {
            return false;
        }

        matches!(
            tensor_header(entries, &format!("{base}.weight_scale_2")),
            Some((dtype, shape)) if dtype.eq_ignore_ascii_case("f32") && shape.is_empty()
        )
    };

    u8_weights.iter().all(|name| valid_weight(name))
        && scale_2.iter().all(|name| {
            name.strip_suffix(".weight_scale_2")
                .is_some_and(|base| valid_weight(&format!("{base}.weight")))
        })
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
    // Classic SDXL LDM/A1111 fused checkpoint: a UNet under `model.diffusion_model.input_blocks`,
    // dual CLIP conditioners, and a `first_stage_model` VAE. The UNet input/output/middle block
    // grammar is distinct from modern DiTs; requiring the SDXL second conditioner prevents an SD1.5
    // checkpoint from being misrouted into the dual-CLIP SDXL loader.
    if any_key_contains(keys, "model.diffusion_model.input_blocks.")
        && any_key_contains(keys, "model.diffusion_model.middle_block.")
        && any_key_contains(keys, "conditioner.embedders.1.model.")
        && any_key_contains(keys, "first_stage_model.")
    {
        return Some("sdxl");
    }
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
    // Mage-Flow: a 12-block dual-stream MMDiT with explicit img/txt modulation linears.
    // Its broad `img_mlp`/`txt_mlp` + `add_q_proj` grammar overlaps Qwen-Image, so pin both the
    // Mage-specific modulation names and the exact published depth (block 11 exists, block 12
    // does not) before the broader Qwen arm below.
    if observed_numbered_indices(keys, "transformer_blocks.") == (0_u32..12).collect()
        && any_key_contains(keys, ".attn.add_q_proj.")
        && any_key_contains(keys, ".img_mlp.")
        && any_key_contains(keys, ".txt_mlp.")
        && any_key_contains(keys, ".img_mod.1.")
        && any_key_contains(keys, ".txt_mod.1.")
    {
        return Some("mage-flow");
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

fn observed_numbered_indices(keys: &[&str], marker: &str) -> std::collections::BTreeSet<u32> {
    keys.iter()
        .filter_map(|key| {
            let (_, suffix) = key.split_once(marker)?;
            suffix.split('.').next()?.parse().ok()
        })
        .collect()
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

    /// Build a safetensors-shaped header from `(name, dtype)` pairs. Ordinary dtype-based
    /// classification does not need shapes; the native-NVFP4 tests build their exact header below.
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
    fn fused_sdxl_a1111_file_is_checkpoint() {
        let verdict = recognized(classify_base_header(&header(&[
            (
                "model.diffusion_model.input_blocks.7.0.out_layers.3.weight",
                "F16",
            ),
            (
                "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
                "F16",
            ),
            (
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "F16",
            ),
            (
                "conditioner.embedders.1.model.transformer.resblocks.9.attn.in_proj_weight",
                "F16",
            ),
            ("first_stage_model.encoder.conv_in.weight", "F16"),
            ("first_stage_model.decoder.conv_out.weight", "F16"),
        ])));
        assert_eq!(verdict.family.as_deref(), Some("sdxl"));
        assert_eq!(verdict.component, ComponentRole::Checkpoint);
        assert_eq!(verdict.quant, QuantFormat::F16);
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

    /// The header classification → engine codec-id bridge (sc-21484). Pinned rather than left to
    /// the match arms because the whole point of the bridge is that `Nvfp4` reaches the codec
    /// vocabulary as `"nvfp4-v1"` and NEVER as the request tier `"nvfp4"` or as a `q4` alias.
    #[test]
    fn quant_format_maps_to_the_engine_source_codec_id() {
        use crate::checkpoint_weight_facts::{is_codec_id, NVFP4_CODEC_ID};

        assert_eq!(QuantFormat::Nvfp4.source_codec_id(), Some(NVFP4_CODEC_ID));
        assert_eq!(QuantFormat::Nvfp4.source_codec_id(), Some("nvfp4-v1"));
        assert_ne!(
            QuantFormat::Nvfp4.source_codec_id(),
            Some(QuantFormat::Nvfp4.as_str()),
            "the codec id and the tier/verdict spelling must stay different strings"
        );
        assert_eq!(
            QuantFormat::Int8TensorwisePerRow.source_codec_id(),
            Some("int8-per-row-v1")
        );
        assert_eq!(QuantFormat::Bf16.source_codec_id(), Some("dense-bf16-v1"));
        assert_eq!(
            QuantFormat::Gguf.source_codec_id(),
            Some("gguf-container-v1")
        );

        // The fp8 family has no proved one-to-one mapping, so it stays absent rather than guessing.
        for unmapped in [
            QuantFormat::Fp8E4m3,
            QuantFormat::ScaledFp8Companion,
            QuantFormat::Fp8InlineScale,
            QuantFormat::ComfyQuantPacked,
            QuantFormat::UnrecognizedScaling,
        ] {
            assert_eq!(
                unmapped.source_codec_id(),
                None,
                "{unmapped} has no proved codec id and must not invent one"
            );
        }

        // `as_str` round trips through `from_label` for every variant, so a persisted
        // `importQuantFormat` resolves back to the classification that wrote it.
        for quant in [
            QuantFormat::Bf16,
            QuantFormat::F16,
            QuantFormat::F32,
            QuantFormat::Fp8E4m3,
            QuantFormat::ScaledFp8Companion,
            QuantFormat::Fp8InlineScale,
            QuantFormat::ComfyQuantPacked,
            QuantFormat::Int8TensorwisePerRow,
            QuantFormat::Nvfp4,
            QuantFormat::Gguf,
            QuantFormat::UnrecognizedScaling,
        ] {
            assert_eq!(QuantFormat::from_label(quant.as_str()), Some(quant));
        }
        // A near-miss never resolves — including the codec id, which is a different vocabulary.
        for stranger in ["NVFP4", "nvfp4-v1", "q4", "", "int8-per-row-v1"] {
            assert_eq!(
                QuantFormat::from_label(stranger),
                None,
                "{stranger:?} must not parse as a verified classification"
            );
        }

        // Whatever is produced is always a real codec id.
        for quant in [
            QuantFormat::Bf16,
            QuantFormat::F16,
            QuantFormat::F32,
            QuantFormat::Int8TensorwisePerRow,
            QuantFormat::Nvfp4,
            QuantFormat::Gguf,
        ] {
            let codec_id = quant.source_codec_id().expect("mapped");
            assert!(
                is_codec_id(codec_id),
                "{quant} produced a non-codec {codec_id:?}"
            );
        }
    }

    #[test]
    fn krea2_kitchen_nvfp4_single_file_is_transformer() {
        let mut map = serde_json::Map::new();
        map.insert(
            "__metadata__".to_owned(),
            json!({"format": "pt", "quant_format": "NVFP4"}),
        );
        map.insert(
            "model.diffusion_model.txtfusion.projector.weight".to_owned(),
            json!({"dtype": "BF16", "shape": [128, 128]}),
        );
        for index in 0..6 {
            let base = format!("model.diffusion_model.blocks.{index}.attn.wq");
            map.insert(
                format!("{base}.weight"),
                json!({"dtype": "U8", "shape": [128, 32]}),
            );
            map.insert(
                format!("{base}.weight_scale"),
                json!({"dtype": "F8_E4M3", "shape": [128, 4]}),
            );
            map.insert(
                format!("{base}.weight_scale_2"),
                json!({"dtype": "F32", "shape": []}),
            );
        }

        let verdict = recognized(classify_base_header(&Value::Object(map)));
        assert_eq!(verdict.family.as_deref(), Some("krea_2"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Nvfp4);
    }

    #[test]
    fn nvfp4_metadata_without_exact_triplets_stays_refused_inline_fp8() {
        let mut map = serde_json::Map::new();
        map.insert(
            "__metadata__".to_owned(),
            json!({"format": "pt", "quant_format": "NVFP4"}),
        );
        map.insert(
            "model.diffusion_model.txtfusion.projector.weight".to_owned(),
            json!({"dtype": "BF16", "shape": [128, 128]}),
        );
        for index in 0..6 {
            let base = format!("model.diffusion_model.blocks.{index}.attn.wq");
            map.insert(
                format!("{base}.weight"),
                json!({"dtype": "U8", "shape": [128, 32]}),
            );
            map.insert(
                format!("{base}.weight_scale"),
                json!({"dtype": "F8_E4M3", "shape": [128, 4]}),
            );
        }

        let verdict = recognized(classify_base_header(&Value::Object(map)));
        assert_eq!(verdict.quant, QuantFormat::Fp8InlineScale);
    }

    #[test]
    #[ignore = "requires KREAMANIA_VARIANT7 to point at the local checkpoint"]
    fn validate_real_kreamania_variant7_nvfp4_checkpoint() {
        let path = PathBuf::from(
            std::env::var("KREAMANIA_VARIANT7")
                .expect("set KREAMANIA_VARIANT7 to kreamania_variant7.safetensors"),
        );
        let verdict = recognized(detect_base_weight_file(&path).expect("read safetensors header"));
        assert_eq!(verdict.family.as_deref(), Some("krea_2"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Nvfp4);
        import_supported(&verdict).expect("the real NVFP4 checkpoint must be importable");
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
    fn mage_flow_bf16_is_exact_twelve_block_transformer() {
        fn mage_keys() -> Vec<String> {
            let mut keys = (0..12)
                .map(|block| format!("transformer_blocks.{block}.attn.to_q.weight"))
                .collect::<Vec<_>>();
            keys.extend(
                [
                    "transformer_blocks.0.attn.add_q_proj.weight",
                    "transformer_blocks.0.img_mlp.net.0.proj.weight",
                    "transformer_blocks.0.txt_mlp.net.0.proj.weight",
                    "transformer_blocks.0.img_mod.1.weight",
                    "transformer_blocks.0.txt_mod.1.weight",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            keys
        }
        fn classify(keys: &[String]) -> BaseWeightVerdict {
            let entries = keys
                .iter()
                .map(|key| (key.as_str(), "BF16"))
                .collect::<Vec<_>>();
            recognized(classify_base_header(&header(&entries)))
        }
        fn classify_family(keys: &[String]) -> Option<String> {
            let entries = keys
                .iter()
                .map(|key| (key.as_str(), "BF16"))
                .collect::<Vec<_>>();
            match classify_base_header(&header(&entries)) {
                BaseWeightDetection::Recognized(verdict) => verdict.family,
                BaseWeightDetection::Unrecognized { .. } => None,
            }
        }

        let complete = mage_keys();
        let verdict = classify(&complete);
        assert_eq!(verdict.family.as_deref(), Some("mage-flow"));
        assert_eq!(verdict.component, ComponentRole::Transformer);
        assert_eq!(verdict.quant, QuantFormat::Bf16);

        for missing_block in [0, 5, 11] {
            let marker = format!("transformer_blocks.{missing_block}.");
            let mutated = complete
                .iter()
                .filter(|key| !key.contains(&marker))
                .cloned()
                .collect::<Vec<_>>();
            assert_ne!(
                classify_family(&mutated).as_deref(),
                Some("mage-flow"),
                "missing block {missing_block} must fail exact 0..11 detection"
            );
        }
        let mut deeper = complete.clone();
        deeper.push("transformer_blocks.12.attn.to_q.weight".to_owned());
        assert_ne!(classify_family(&deeper).as_deref(), Some("mage-flow"));

        for marker in [
            ".attn.add_q_proj.",
            ".img_mlp.",
            ".txt_mlp.",
            ".img_mod.1.",
            ".txt_mod.1.",
        ] {
            let mutated = complete
                .iter()
                .filter(|key| !key.contains(marker))
                .cloned()
                .collect::<Vec<_>>();
            assert_ne!(
                classify_family(&mutated).as_deref(),
                Some("mage-flow"),
                "missing Mage grammar marker {marker} must fail closed"
            );
        }
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
    fn import_supported_accepts_krea2_transformer_loadable_encodings() {
        for quant in [
            QuantFormat::Bf16,
            QuantFormat::Int8TensorwisePerRow,
            QuantFormat::Nvfp4,
        ] {
            assert!(
                import_supported(&verdict(Some("krea_2"), ComponentRole::Transformer, quant))
                    .is_ok(),
                "Krea 2 transformer {quant} has a landed single-file loader"
            );
        }
    }

    #[test]
    fn import_supported_accepts_dense_fused_sdxl_only() {
        for quant in [QuantFormat::F16, QuantFormat::Bf16, QuantFormat::F32] {
            assert!(
                import_supported(&verdict(Some("sdxl"), ComponentRole::Checkpoint, quant)).is_ok()
            );
        }
        assert!(import_supported(&verdict(
            Some("sdxl"),
            ComponentRole::Transformer,
            QuantFormat::F16
        ))
        .is_err());
        assert!(import_supported(&verdict(
            Some("sdxl"),
            ComponentRole::Checkpoint,
            QuantFormat::Fp8E4m3
        ))
        .is_err());
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
            reason.contains("bf16") && reason.contains("int8") && reason.contains("NVFP4"),
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

    /// sc-15036 — Mage-Flow is the DIRECTORY-shaped member of the import gate. The triple gate
    /// accepts a dense bf16 transformer (the shape a full base fine-tune writes) and refuses the
    /// wrong component and the pre-quantized tiers, each with its own reason.
    #[test]
    fn import_supported_accepts_a_dense_mage_flow_transformer_and_names_every_refusal() {
        assert!(import_supported(&verdict(
            Some("mage-flow"),
            ComponentRole::Transformer,
            QuantFormat::Bf16
        ))
        .is_ok());

        let wrong_component = import_supported(&verdict(
            Some("mage-flow"),
            ComponentRole::Vae,
            QuantFormat::Bf16,
        ))
        .expect_err("a VAE is not a Mage-Flow backbone");
        assert!(
            wrong_component.contains("transformer"),
            "reason: {wrong_component}"
        );

        for quant in [
            QuantFormat::Fp8E4m3,
            QuantFormat::Int8TensorwisePerRow,
            QuantFormat::Gguf,
        ] {
            let refused = import_supported(&verdict(
                Some("mage-flow"),
                ComponentRole::Transformer,
                quant,
            ))
            .expect_err("only dense bf16 is loadable");
            assert!(
                refused.contains(quant.as_str()),
                "the refusal must name the encoding it saw: {refused}"
            );
        }
    }

    /// sc-15036 — the directory-shape probe the import gate and the worker's render lane SHARE.
    /// Discriminating in three directions: both files present is accepted, and EITHER one missing
    /// is refused — a torn artifact must not register as a model that then fails at load.
    #[test]
    fn is_mage_flow_transformer_dir_requires_both_the_config_and_the_weights() {
        // Guarded rather than hand-cleaned: the trailing `remove_dir_all` this replaces was
        // skipped whenever an assertion below panicked, which is exactly when the leftovers
        // pile up (sc-17641).
        let root_guard = tempfile::tempdir().expect("temp dir for the probe fixtures");
        let root = root_guard.path();
        let case = |name: &str, files: &[&str]| {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            for file in files {
                fs::write(dir.join(file), b"x").unwrap();
            }
            dir
        };

        assert!(is_mage_flow_transformer_dir(&case(
            "complete",
            &[
                MAGE_FLOW_TRANSFORMER_CONFIG_FILE,
                MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE
            ]
        )));
        assert!(
            !is_mage_flow_transformer_dir(&case(
                "weights-only",
                &[MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE]
            )),
            "weights without the architecture config cannot be loaded"
        );
        assert!(
            !is_mage_flow_transformer_dir(&case(
                "config-only",
                &[MAGE_FLOW_TRANSFORMER_CONFIG_FILE]
            )),
            "a config with no weights is a torn run, not a checkpoint"
        );
        assert!(!is_mage_flow_transformer_dir(&case("empty", &[])));
        assert!(!is_mage_flow_transformer_dir(&root.join("absent")));
    }

    #[test]
    fn imported_model_primary_weight_file_accepts_only_exact_loader_shapes() {
        let root_guard = tempfile::tempdir().expect("import-shape fixtures");
        let root = root_guard.path();

        let direct = root.join("direct.safetensors");
        fs::write(&direct, b"weights").unwrap();
        assert_eq!(
            imported_model_primary_weight_file(&direct),
            Some(direct.clone())
        );

        let flat = root.join("flat");
        fs::create_dir(&flat).unwrap();
        let lone = flat.join("model.safetensors");
        fs::write(&lone, b"weights").unwrap();
        fs::write(flat.join("install.json"), b"{}").unwrap();
        assert_eq!(imported_model_primary_weight_file(&flat), Some(lone));

        let multiple = root.join("multiple");
        fs::create_dir(&multiple).unwrap();
        fs::write(multiple.join("a.safetensors"), b"a").unwrap();
        fs::write(multiple.join("b.safetensors"), b"b").unwrap();
        assert_eq!(imported_model_primary_weight_file(&multiple), None);

        let nested = root.join("nested");
        fs::create_dir_all(nested.join("child")).unwrap();
        fs::write(nested.join("child/model.safetensors"), b"weights").unwrap();
        assert_eq!(imported_model_primary_weight_file(&nested), None);

        let snapshot = root.join("snapshot");
        fs::create_dir(&snapshot).unwrap();
        fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();
        fs::write(snapshot.join("model_index.json"), b"{}").unwrap();
        assert_eq!(imported_model_primary_weight_file(&snapshot), None);

        let mage = root.join("mage");
        fs::create_dir(&mage).unwrap();
        fs::write(mage.join(MAGE_FLOW_TRANSFORMER_CONFIG_FILE), b"{}").unwrap();
        let mage_weights = mage.join(MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE);
        fs::write(&mage_weights, b"weights").unwrap();
        assert_eq!(
            imported_model_primary_weight_file(&mage),
            Some(mage_weights)
        );
    }

    #[test]
    fn import_supported_families_are_a_subset_of_the_ok_arms() {
        // Guardrail: every family the gate advertises must actually have an Ok triple, so the
        // advertised set and the `match` arms can never drift (add the family here + its arm together).
        for family in IMPORT_SUPPORTED_FAMILIES {
            let component = if *family == "sdxl" {
                ComponentRole::Checkpoint
            } else {
                ComponentRole::Transformer
            };
            assert!(
                import_supported(&verdict(Some(family), component, QuantFormat::Bf16)).is_ok(),
                "IMPORT_SUPPORTED_FAMILIES lists {family} but no bf16 transformer arm accepts it"
            );
        }
    }
}
