//! MLX **mapped logical-weight reader** and the engine's **codec table** (epic 20398, sc-20634;
//! ComfyUI descriptor codecs sc-20385).
//!
//! The shared seam every MLX family provider reads imported checkpoints through:
//!
//! 1. the provider supplies its adapter-owned [`LogicalKeyMapping`] (the same `mapping_id` its
//!    registry adapter row declares);
//! 2. [`plan_logical_weights`] reads the safetensors **header** plus the (small) `.comfy_quant`
//!    descriptor payloads and compiles a [`LogicalWeightPlan`] against [`baseline_codec_registry`]
//!    — an unmapped key, a key collision, a stored format without a registered codec, a malformed
//!    descriptor, a missing/mis-shaped scale companion, or bad MXFP8 block geometry is a typed
//!    refusal naming the exact tensor, before any MLX array exists;
//! 3. [`read_logical_weights`] loads the planned tensors, renames each to its canonical logical
//!    key, runs the planned codec **per layer** (mixed checkpoints dispatch tensor-by-tensor), and
//!    returns a [`LogicalWeightReceipt`] whose resident bytes are measured from the decoded arrays.
//!
//! Codecs are registered **once** per platform catalog via [`register_checkpoint_codecs`]; the
//! registry rows and [`CODEC_IMPLEMENTATION_IDS`] are kept in lockstep by the catalog conformance
//! test, so a declared codec without an implementation (or vice versa) cannot ship.
//!
//! # fp8 files and MLX's loader
//!
//! MLX has no fp8/e8m0 storage dtype, so its own safetensors parser rejects any file carrying
//! `F8_E4M3` / `F8_E5M2` / `F8_E8M0` tensors. Those files load through a **header-rewriting custom
//! reader** (`load_planned_file`): the on-disk header is re-presented to MLX with each fp8 dtype
//! renamed to `U8` at identical byte length (element size is 1 in both readings, so every offset is
//! unchanged and the payload is byte-identical), and MLX's normal lazy `load` primitive then serves
//! the raw bytes file-backed. Laziness is preserved deliberately — the deferred mode's bounded
//! block-window consumers reopen the file per window and must not copy the whole checkpoint into
//! host RAM per reopen. The codec then decodes the `U8` view on device: E4M3 through MLX's own
//! byte-accurate `from_fp8`, E5M2 as the top byte of an IEEE binary16 (`u16 << 8` viewed as f16),
//! E8M0 as `2^(byte − 127)` (`u32 << 23` viewed as f32), each multiplied by its planned scale in
//! f32 and cast once to the codec's resident bf16 — the same reference math as
//! `gen_core::comfy_quant`, which the goldens pin.
//!
//! Residency is priced **per layer** by the plan ([`gen_core::PlannedResidency`]): MLX has no
//! packed fp8/int8 matmul on this seam, so its policy is [`DenseResidencyPolicy`] — every
//! quantized layer prices at its dense bf16 form and every scale/descriptor companion is consumed.
//! The receipt's measured bytes equal the plan's `resident_bytes()`; that pair is the
//! packed-vs-dense pricing seam admission reads.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::OnceLock;

use gen_core::checkpoint_codec::{
    CheckpointCodecRegistration, CheckpointCodecRegistry, CodecResidencyReport,
    DenseResidencyPolicy, LogicalKeyMapping, LogicalReadMaterialization, LogicalTensorPlan,
    LogicalWeightPlan, LogicalWeightReceipt, ResidencyMode, ScalarScaleSource, TensorCodecSpec,
    WeightEncoding, DENSE_BF16_CODEC, DENSE_F16_CODEC, DENSE_F32_CODEC, FP8_E4M3_SCALAR_CODEC,
    FP8_E5M2_SCALAR_CODEC, INT8_PER_ROW_CODEC, MXFP8_CODEC, NVFP4_CODEC,
};
use gen_core::checkpoint_facts::ExecutionRepresentation;
use gen_core::weightsmeta::Dtype as HeaderDtype;
use gen_core::ProviderRegistryBuilder;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};

use crate::weights::Weights;
use crate::{Error, Result};

/// Descriptor payloads are tiny JSON blobs; anything above this is not a `.comfy_quant` tensor.
const MAX_DESCRIPTOR_BYTES: u64 = 65_536;

/// The codec rows this engine implements and registers. Register through
/// [`register_checkpoint_codecs`]; never per family crate.
pub const BASELINE_CODECS: &[CheckpointCodecRegistration] =
    gen_core::checkpoint_codec::BASELINE_CHECKPOINT_CODECS;

/// The codec ids this engine has a decode implementation for. Must equal the ids of
/// [`BASELINE_CODECS`] (the catalog test proves it).
pub const CODEC_IMPLEMENTATION_IDS: &[&str] = &[
    DENSE_BF16_CODEC.codec_id,
    DENSE_F16_CODEC.codec_id,
    DENSE_F32_CODEC.codec_id,
    FP8_E4M3_SCALAR_CODEC.codec_id,
    FP8_E5M2_SCALAR_CODEC.codec_id,
    MXFP8_CODEC.codec_id,
    INT8_PER_ROW_CODEC.codec_id,
    NVFP4_CODEC.codec_id,
];

/// Register the engine's codec table exactly once into a platform catalog.
pub fn register_checkpoint_codecs(mut builder: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    for codec in BASELINE_CODECS {
        builder = builder.register_checkpoint_codec(*codec);
    }
    builder
}

/// The validated registry the loaders plan against.
pub fn baseline_codec_registry() -> &'static CheckpointCodecRegistry {
    static REGISTRY: OnceLock<CheckpointCodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        CheckpointCodecRegistry::new(BASELINE_CODECS.iter().copied())
            .expect("the engine codec table is a valid registry")
    })
}

/// Header-plus-descriptors plan for one safetensors file under the given mapping. Refuses before
/// any array is created; the error names the exact on-disk tensor. MLX residency is always the
/// dense fallback ([`DenseResidencyPolicy`]) — this engine has no packed fp8/int8 matmul on the
/// codec seam.
pub fn plan_logical_weights(
    path: &Path,
    mapping: &dyn LogicalKeyMapping,
) -> Result<LogicalWeightPlan> {
    let headers = gen_core::safetensors_path_tensor_headers(path)?;
    let descriptors = gen_core::read_safetensors_tensor_payloads(
        path,
        |header| header.name.ends_with(".comfy_quant"),
        MAX_DESCRIPTOR_BYTES,
    )?;
    // A checkpoint declares its quantization per layer (`.comfy_quant` tensors), file-wide
    // (`__metadata__._quantization_metadata`, sc-20641), or not at all; both routes are read and
    // the compiler refuses a layer the two disagree about.
    let quantization_metadata = gen_core::safetensors_path_quantization_metadata(path)?;
    gen_core::compile_logical_weight_plan_with_metadata(
        &headers,
        &descriptors,
        quantization_metadata.as_deref(),
        mapping,
        baseline_codec_registry(),
        &DenseResidencyPolicy,
    )
    .map_err(|error| {
        Error::Msg(format!(
            "logical weight plan for {} ({}): {error}",
            path.display(),
            mapping.mapping_id()
        ))
    })
}

/// How a read leaves its payloads.
pub enum LogicalReadMode<'a> {
    /// Evaluate every planned tensor through the supplied materializer and measure what each
    /// codec left resident. Providers apply their adapter-owned shape normalization inside the
    /// closure (a lossless reshape is part of the canonical logical form) and then run their
    /// pin-guarded `Weights::materialize`, so residency is measured on the final resident arrays.
    /// Dequantizing codecs additionally evaluate **per tensor** inside the read, so one
    /// projection's source codes/scales graph is released before the next is decoded (a 12.8 GB
    /// packed DiT never holds source and reconstruction fully resident together).
    Eager(&'a mut dyn FnMut(&mut Weights) -> Result<()>),
    /// Leave payloads lazy; block-window and bounded loaders consume them under their own pin
    /// guards. The receipt reports `Deferred` and no residency rows.
    Deferred,
}

/// The reader's output: logical-keyed weights plus the receipt.
pub struct LogicalWeights {
    pub weights: Weights,
    pub receipt: LogicalWeightReceipt,
}

/// Load exactly the planned tensors from `path`, rename them to their logical keys, decode each
/// through its planned codec, and (eagerly) measure residency.
///
/// The file's tensor set must equal the plan's physical key surface (weights **and** companions):
/// the plan was compiled from this file's header, so any difference means the source changed
/// between planning and reading and the read refuses instead of loading a different checkpoint.
pub fn read_logical_weights(
    path: &Path,
    plan: &LogicalWeightPlan,
    mode: LogicalReadMode<'_>,
) -> Result<LogicalWeights> {
    // Adapter-declared logical transforms (sc-21547) are a Candle-side capability today: this
    // reader removes each physical tensor from the map as it decodes, so a one-to-many entry would
    // find its source already taken. Refuse by name rather than fail as a missing tensor — a
    // transformed plan is not a plan this engine can honour, and silently reading only the first
    // output would hand the model a fused matrix under a split key's name.
    if let Some(transformed) = plan
        .tensors
        .iter()
        .find(|tensor| tensor.transform.is_some())
    {
        return Err(Error::Msg(format!(
            "logical weight read of {}: logical key {:?} is a declared transform of physical \
             tensor {:?}, which this engine's reader does not implement; plan this checkpoint with \
             a mapping that declares no logical transforms",
            path.display(),
            transformed.logical_key,
            transformed.physical_key,
        )));
    }
    let mut physical = load_planned_file(path)?;
    let mut on_disk: Vec<&str> = physical.keys().map(String::as_str).collect();
    on_disk.sort_unstable();
    let mut planned: Vec<&str> = plan.all_physical_keys().collect();
    planned.sort_unstable();
    if on_disk != planned {
        let missing = planned
            .iter()
            .filter(|key| on_disk.binary_search(key).is_err())
            .count();
        let unplanned = on_disk
            .iter()
            .filter(|key| planned.binary_search(key).is_err())
            .count();
        return Err(Error::Msg(format!(
            "logical weight read of {}: tensor set changed since planning ({missing} planned \
             tensor(s) missing, {unplanned} unplanned tensor(s) present); refusing to load a \
             different checkpoint",
            path.display(),
        )));
    }

    let eager = matches!(mode, LogicalReadMode::Eager(_));
    let mut logical = Weights::empty();
    for tensor in &plan.tensors {
        let array = physical
            .remove(&tensor.physical_key)
            .ok_or_else(|| Error::MissingTensor(tensor.physical_key.clone()))?;
        let decoded = decode(tensor, array, &physical)?;
        if eager && !matches!(tensor.codec, TensorCodecSpec::Dense) {
            // MLX is lazy: materialize dequantizing projections one by one so the eager resident
            // load never retains a graph edge to every removed source code/scale buffer (which
            // would keep both the packed source and the dense reconstruction alive at once).
            decoded.eval()?;
        }
        logical.insert(tensor.logical_key.clone(), decoded);
    }
    drop(physical);

    let materialization = match mode {
        LogicalReadMode::Eager(materialize) => {
            materialize(&mut logical)?;
            LogicalReadMaterialization::Materialized
        }
        LogicalReadMode::Deferred => LogicalReadMaterialization::Deferred,
    };
    let residency = match materialization {
        LogicalReadMaterialization::Materialized => measure_residency(plan, &logical)?,
        // MLX reads are all-or-nothing at this seam: `Partial` (a candle block-streamed state,
        // sc-11045 fix round) is never produced here, and would carry no measured rows if it were.
        LogicalReadMaterialization::Partial | LogicalReadMaterialization::Deferred => Vec::new(),
    };
    Ok(LogicalWeights {
        weights: logical,
        receipt: LogicalWeightReceipt {
            mapping_id: plan.mapping_id,
            tensor_count: plan.tensors.len(),
            source_bytes: plan.source_bytes,
            materialization,
            residency,
            // MLX serves every quantized row through its dense-side decode as *planned* (the
            // residency policy is dense-only there), so there is never a packed pricing to demote.
            demotions: Vec::new(),
        },
    })
}

// =================================================================================================
// Per-layer codec decode (MLX ops; lazy until the caller evaluates).
// =================================================================================================

fn guard_shape(tensor: &LogicalTensorPlan, array: &Array, expected: &[usize]) -> Result<()> {
    let expected: Vec<i32> = expected.iter().map(|dim| *dim as i32).collect();
    if array.shape() != expected {
        return Err(Error::Msg(format!(
            "codec {}: tensor {:?} was planned with stored shape {expected:?} but the backend \
             loaded shape {:?}; the file changed between planning and reading",
            tensor.codec_id,
            tensor.physical_key,
            array.shape()
        )));
    }
    Ok(())
}

fn guard_dtype(tensor: &LogicalTensorPlan, array: &Array, expected: Dtype) -> Result<()> {
    if array.dtype() != expected {
        return Err(Error::Msg(format!(
            "codec {}: tensor {:?} was planned as {} but the backend loaded {:?}",
            tensor.codec_id,
            tensor.physical_key,
            tensor.encoding.label(),
            array.dtype()
        )));
    }
    Ok(())
}

fn companion<'a>(
    tensor: &LogicalTensorPlan,
    physical: &'a HashMap<String, Array>,
    key: &str,
) -> Result<&'a Array> {
    physical.get(key).ok_or_else(|| {
        Error::Msg(format!(
            "codec {}: tensor {:?} planned companion {key:?} which the loaded file no longer \
             carries",
            tensor.codec_id, tensor.physical_key
        ))
    })
}

/// Decode E4M3FN bytes (`U8` view) to f32 via MLX's byte-accurate converter — with one repair:
/// MLX's `from_fp8` reads the two all-ones codes `0x7F`/`0xFF` as ±480 (a NaN-free extension of
/// the format), while OCP E4M3**FN** (and `torch.float8_e4m3fn`, the producer of these files)
/// defines them as NaN. A corrupted payload must surface as NaN, not as a plausible ±480 weight,
/// so those two codes are substituted explicitly. Pinned by the 256-code cross-check test against
/// `gen_core::fp8_e4m3fn_to_f32`.
fn e4m3_bytes_to_f32(codes: &Array) -> Result<Array> {
    let decoded = codes.from_fp8(Dtype::Float32)?;
    let nan_code = mlx_rs::ops::logical_or(
        &mlx_rs::ops::eq(codes, Array::from_slice(&[0x7F_u8], &[]))?,
        &mlx_rs::ops::eq(codes, Array::from_slice(&[0xFF_u8], &[]))?,
    )?;
    Ok(mlx_rs::ops::r#where(
        &nan_code,
        Array::from_f32(f32::NAN),
        &decoded,
    )?)
}

/// Decode E5M2 bytes (`U8` view) to f32: E5M2 is IEEE binary16 truncated to its top byte, so
/// `u16(byte) << 8` reinterpreted as f16 is the exact value (inf/NaN codes included).
fn e5m2_bytes_to_f32(codes: &Array) -> Result<Array> {
    let shifted = mlx_rs::ops::multiply(
        &codes.as_dtype(Dtype::Uint16)?,
        Array::from_slice(&[256_u16], &[]),
    )?;
    Ok(shifted
        .view_dtype(Dtype::Float16)?
        .as_dtype(Dtype::Float32)?)
}

/// Decode E8M0 shared-exponent bytes (`U8` view) to f32: `2^(byte − 127)` = `u32(byte) << 23`
/// reinterpreted as f32, with the spec's byte-0 reading (`2^-127`) substituted where the shifted
/// form would read `0.0` — the same semantics as `gen_core::comfy_quant::e8m0_to_f32`.
fn e8m0_bytes_to_f32(codes: &Array) -> Result<Array> {
    let shifted = mlx_rs::ops::multiply(
        &codes.as_dtype(Dtype::Uint32)?,
        Array::from_slice(&[1_u32 << 23], &[]),
    )?;
    let scales = shifted.view_dtype(Dtype::Float32)?;
    let zero_code = mlx_rs::ops::eq(codes, Array::from_slice(&[0_u8], &[]))?;
    let scales = mlx_rs::ops::r#where(
        &zero_code,
        Array::from_f32(f32::from_bits(0x0040_0000)),
        &scales,
    )?;
    // Code 255 is NaN per the spec; the shifted form reads +inf (exponent 255, mantissa 0).
    let nan_code = mlx_rs::ops::eq(codes, Array::from_slice(&[0xFF_u8], &[]))?;
    Ok(mlx_rs::ops::r#where(
        &nan_code,
        Array::from_f32(f32::NAN),
        &scales,
    )?)
}

/// Un-swizzle a cuBLAS 128×4-tiled MXFP8 block-scale buffer back to `[rows, blocks]` row-major —
/// the exact inverse of `comfy_kitchen.float_utils.to_blocked`, as MLX reshapes/transposes.
fn unswizzle_block_scales(scales: &Array, rows: i32, blocks: i32) -> Result<Array> {
    let padded_rows = (rows + 127) / 128 * 128;
    let padded_cols = (blocks + 3) / 4 * 4;
    let row_tiles = padded_rows / 128;
    let col_tiles = padded_cols / 4;
    // to_blocked: view(rt, 128, ct, 4) → permute(0, 2, 1, 3) → reshape(-1, 4, 32, 4)
    //             → transpose(1, 2) → flatten. Invert step by step.
    let step1 = scales.reshape(&[row_tiles * col_tiles, 32, 4, 4])?;
    let step2 = step1.transpose_axes(&[0, 2, 1, 3])?; // (n, 4, 32, 4)
    let step3 = step2.reshape(&[row_tiles, col_tiles, 128, 4])?;
    let step4 = step3.transpose_axes(&[0, 2, 1, 3])?; // (rt, 128, ct, 4)
    let unblocked = step4.reshape(&[padded_rows, padded_cols])?;
    Ok(unblocked.index((..rows, ..blocks)))
}

/// Run the planned codec on one array — the per-layer dispatch. Dense codecs are byte-preserving:
/// each asserts the backend decoded exactly the stored dtype it planned for and returns the array
/// untouched, so no cast or substitution can hide inside it. Quantized codecs receive the raw `U8`
/// (fp8) or `I8` byte view and decode in f32 with the planned scales, casting once to the codec's
/// resident bf16.
fn decode(
    tensor: &LogicalTensorPlan,
    array: Array,
    physical: &HashMap<String, Array>,
) -> Result<Array> {
    // A block-padded row (MXFP8/NVFP4) whose adapter declared no logical shape would decode its
    // PADDING as weights — see [`LogicalTensorPlan::undeclared_padded_storage`]. Planning such a
    // file stays legal (the padded grid is a conservative pricing over-estimate, and the
    // memory-strategy paths that compile a plan for admission never materialize a tensor);
    // materializing it is not.
    if let Some(refusal) = tensor.undeclared_padded_storage_refusal() {
        return Err(Error::Msg(refusal));
    }
    // Rank floor for the arms below that index the logical shape positionally (`tensor.shape[0]` /
    // `tensor.shape[1]` in the MXFP8, NVFP4 and int8-per-row arms) — see
    // [`LogicalTensorPlan::matrix_rank_refusal`]. Checked before the per-arm `guard_shape` so a
    // wrong-rank plan is diagnosed as the rank defect it is rather than as a stored-shape drift, and
    // before any indexing so it refuses by name instead of panicking out of bounds. The refusal text
    // is gen-core's, so this engine and candle-gen refuse the same plan with the same diagnosis.
    if let Some(refusal) = tensor.matrix_rank_refusal() {
        return Err(Error::Msg(refusal));
    }
    // Every arm below produces the codec's DENSE resident form. MLX has no packed fp8/int8/fp4
    // matmul on this seam, so this engine plans [`DenseResidencyPolicy`] for every row — but the
    // reader must not *assume* that. A `Packed` entry reaching here (a caller planning with a
    // foreign policy, or a policy that grows a packed row) would dense-decode a layer admission
    // priced at its packed stored bytes: bf16 is 2 bytes/element against fp8's 1 and NVFP4's 0.5,
    // so the load silently costs 2–4× the admitted footprint. Refuse instead — the same direction
    // `candle_gen::logical_weights` takes for a packed plan on a build with no CUDA fp8 leg.
    if tensor.residency.mode != ResidencyMode::Dense {
        return Err(Error::Msg(format!(
            "codec {}: tensor {:?} was planned with {:?} residency ({} stored bytes), but this \
             engine decodes every codec to its dense resident form; replan with a dense residency \
             policy instead of silently materializing the dense reconstruction where the plan \
             priced the stored packing",
            tensor.codec_id,
            tensor.physical_key,
            tensor.residency.mode,
            tensor.residency.resident_bytes
        )));
    }
    match &tensor.codec {
        TensorCodecSpec::Dense => {
            let expected = match tensor.codec_id {
                id if id == DENSE_BF16_CODEC.codec_id => Dtype::Bfloat16,
                id if id == DENSE_F16_CODEC.codec_id => Dtype::Float16,
                id if id == DENSE_F32_CODEC.codec_id => Dtype::Float32,
                other => {
                    return Err(Error::Msg(format!(
                        "codec {other:?} is registered but this engine has no dense implementation \
                         for it (tensor {:?})",
                        tensor.physical_key
                    )))
                }
            };
            // Shape before dtype: planning and reading are two separate opens of the file, so a
            // same-key same-dtype tensor whose SHAPE changed in between would otherwise decode
            // silently and be handed to a model expecting the planned geometry.
            guard_shape(tensor, &array, &tensor.shape)?;
            guard_dtype(tensor, &array, expected)?;
            Ok(array)
        }
        TensorCodecSpec::ScalarFp8 { scale, .. } => {
            // fp8 arrives as the U8 byte view (MLX has no fp8 dtype; the loader re-presents it).
            guard_shape(tensor, &array, &tensor.shape)?;
            guard_dtype(tensor, &array, Dtype::Uint8)?;
            let values = match tensor.encoding {
                WeightEncoding::Fp8E4M3 => e4m3_bytes_to_f32(&array)?,
                WeightEncoding::Fp8E5M2 => e5m2_bytes_to_f32(&array)?,
                other => {
                    return Err(Error::Msg(format!(
                        "codec {}: tensor {:?} planned scalar-fp8 decode for non-fp8 encoding {}",
                        tensor.codec_id,
                        tensor.physical_key,
                        other.label()
                    )))
                }
            };
            let scaled = match scale {
                ScalarScaleSource::Unit => values,
                ScalarScaleSource::Companion { physical_key } => {
                    let scale = companion(tensor, physical, physical_key)?;
                    if scale.dtype() != Dtype::Float32 {
                        return Err(Error::Msg(format!(
                            "codec {}: companion {:?} must load as F32, got {:?}",
                            tensor.codec_id,
                            physical_key,
                            scale.dtype()
                        )));
                    }
                    // Scalar-fp8 means ONE per-tensor scale. Without this check a companion of any
                    // broadcastable shape multiplies through silently: a `[cols]` scale broadcasts
                    // across rows and a `[rows, 1]` one across columns, so a per-row or per-column
                    // convention mis-read as scalar-fp8 would decode to plausible-looking but wrong
                    // weights rather than refusing. `size()` (not the shape) is the right test —
                    // `[]`, `[1]` and `[1, 1]` are all the one scalar this codec means.
                    if scale.size() != 1 {
                        return Err(Error::Msg(format!(
                            "codec {}: companion {:?} must load as ONE F32 scalar (scalar-fp8 is a \
                             per-tensor scale), got shape {:?} with {} elements; a broadcastable \
                             multi-element scale is a different convention, not this codec",
                            tensor.codec_id,
                            physical_key,
                            scale.shape(),
                            scale.size()
                        )));
                    }
                    mlx_rs::ops::multiply(&values, scale)?
                }
            };
            Ok(scaled.as_dtype(Dtype::Bfloat16)?)
        }
        TensorCodecSpec::Mxfp8 {
            scale,
            stored_shape,
            ..
        } => {
            guard_shape(tensor, &array, stored_shape)?;
            guard_dtype(tensor, &array, Dtype::Uint8)?;
            let scales = companion(tensor, physical, scale)?;
            if scales.dtype() != Dtype::Uint8 {
                return Err(Error::Msg(format!(
                    "codec {}: companion {scale:?} must load as U8 E8M0 bytes, got {:?}",
                    tensor.codec_id,
                    scales.dtype()
                )));
            }
            let rows = stored_shape[0] as i32;
            let cols = stored_shape[1] as i32;
            let blocks = cols / gen_core::MXFP8_BLOCK as i32;
            let block = gen_core::MXFP8_BLOCK as i32;
            let values = e4m3_bytes_to_f32(&array)?.reshape(&[rows, blocks, block])?;
            let block_scales = e8m0_bytes_to_f32(&unswizzle_block_scales(scales, rows, blocks)?)?
                .reshape(&[rows, blocks, 1])?;
            let dense = mlx_rs::ops::multiply(&values, &block_scales)?.reshape(&[rows, cols])?;
            let logical = dense.index((..tensor.shape[0] as i32, ..tensor.shape[1] as i32));
            Ok(logical.as_dtype(Dtype::Bfloat16)?)
        }
        TensorCodecSpec::Nvfp4 {
            block_scale,
            global_scale,
            stored_shape,
            ..
        } => {
            // NVFP4 is decoded on the host through the gen-core reference: MLX has no 4-bit
            // element type to view the packed nibbles as, so the same reference decode both
            // backends share is the honest single implementation rather than a second
            // bit-twiddling path that could drift from it.
            let packed_shape = [stored_shape[0], stored_shape[1] / 2];
            guard_shape(tensor, &array, &packed_shape)?;
            guard_dtype(tensor, &array, Dtype::Uint8)?;
            let scales = companion(tensor, physical, block_scale)?;
            if scales.dtype() != Dtype::Uint8 {
                return Err(Error::Msg(format!(
                    "codec {}: companion {block_scale:?} must load as U8 E4M3 block-scale bytes, \
                     got {:?}",
                    tensor.codec_id,
                    scales.dtype()
                )));
            }
            let global = companion(tensor, physical, global_scale)?;
            if global.dtype() != Dtype::Float32 || global.size() != 1 {
                return Err(Error::Msg(format!(
                    "codec {}: companion {global_scale:?} must load as one F32 scalar, got {:?} \
                     with {} elements",
                    tensor.codec_id,
                    global.dtype(),
                    global.size()
                )));
            }
            let mut values = Vec::new();
            gen_core::decode_nvfp4(
                array.as_slice::<u8>(),
                scales.as_slice::<u8>(),
                global.as_slice::<f32>()[0],
                *stored_shape,
                [tensor.shape[0], tensor.shape[1]],
                &mut values,
            )
            .map_err(|error| {
                Error::Msg(format!(
                    "codec {}: tensor {:?}: {error}",
                    tensor.codec_id, tensor.physical_key
                ))
            })?;
            let dense =
                Array::from_slice(&values, &[tensor.shape[0] as i32, tensor.shape[1] as i32]);
            Ok(dense.as_dtype(Dtype::Bfloat16)?)
        }
        TensorCodecSpec::Int8PerRow { scale, .. } => {
            guard_shape(tensor, &array, &tensor.shape)?;
            guard_dtype(tensor, &array, Dtype::Int8)?;
            let rows = tensor.shape[0] as i32;
            let scale = companion(tensor, physical, scale)?;
            if scale.dtype() != Dtype::Float32 {
                return Err(Error::Msg(format!(
                    "codec {}: companion must load as F32, got {:?}",
                    tensor.codec_id,
                    scale.dtype()
                )));
            }
            // [rows] / [rows, 1] / scalar (single row) → [rows, 1] for the broadcast; the plan
            // validated the stored scale shape against the row count.
            let scale = scale.reshape(&[rows, 1])?;
            let dense = mlx_rs::ops::multiply(&array.as_dtype(Dtype::Float32)?, &scale)?;
            Ok(dense.as_dtype(Dtype::Bfloat16)?)
        }
    }
}

/// Resident bytes per codec, measured from the decoded arrays (dtype × shape after decode), not
/// from the header. Companion source bytes are attributed to their owner's codec row so a codec's
/// `source_bytes` covers everything it consumed.
fn measure_residency(
    plan: &LogicalWeightPlan,
    logical: &Weights,
) -> Result<Vec<CodecResidencyReport>> {
    let mut by_codec: BTreeMap<&'static str, CodecResidencyReport> = BTreeMap::new();
    let mut codec_by_owner: BTreeMap<&str, &'static str> = BTreeMap::new();
    // A transformed tensor contributes one plan entry per logical output, all naming the same
    // physical key (sc-21547), and exactly one of them carries `source_bytes` while its siblings
    // carry zero — the plan compiler's convention. So a plain sum totals the file once, and
    // de-duplicating by physical key here would be wrong: `plan.tensors` is sorted by logical key,
    // so the byte-carrying output is not generally the first sighting of a fused key. Tensor
    // counts and resident bytes stay per-logical: each output is separately decoded and separately
    // occupies memory.
    for tensor in &plan.tensors {
        codec_by_owner.insert(tensor.physical_key.as_str(), tensor.codec_id);
        let array = logical.require(&tensor.logical_key)?;
        let resident_bytes = u64::try_from(array.nbytes()).map_err(|_| {
            Error::Msg(format!(
                "tensor {:?} resident size overflows u64",
                tensor.logical_key
            ))
        })?;
        let report = by_codec
            .entry(tensor.codec_id)
            .or_insert(CodecResidencyReport {
                codec_id: tensor.codec_id,
                // MLX plans under `DenseResidencyPolicy` only — every codec row decodes to its
                // dense resident encoding here, so every row this backend reports is a
                // dense fallback and none of them may ever be labelled native (sc-21484). The
                // retained-companion assertion below is the same invariant, stated on bytes.
                representation: ExecutionRepresentation::DenseFallback,
                tensor_count: 0,
                source_bytes: 0,
                resident_bytes: 0,
            });
        report.tensor_count += 1;
        report.source_bytes = report.source_bytes.saturating_add(tensor.source_bytes);
        report.resident_bytes = report.resident_bytes.saturating_add(resident_bytes);
    }
    for companion in &plan.companions {
        let Some(codec_id) = codec_by_owner.get(companion.owner_physical_key.as_str()) else {
            continue;
        };
        // MLX plans under `DenseResidencyPolicy` only — there is no packed fp8/int8 matmul on
        // this seam — so every companion is consumed by its decode and retains nothing. Adding
        // the plan's own number in would make the receipt a *copy* of the plan on exactly the
        // row the receipt/plan pair exists to cross-check. Assert the invariant instead: if a
        // packed policy ever reaches this backend, the read refuses rather than silently
        // agreeing with a residency it never measured.
        //
        // Checked outside the `by_codec` lookup below, so a companion whose owner's row is absent
        // still cannot smuggle a retained residency past this refusal.
        if companion.resident_bytes != 0 {
            return Err(Error::Msg(format!(
                "companion {:?} (codec {codec_id}) was planned to retain {} resident bytes, but \
                 this backend decodes every codec through its dense fallback and measures no \
                 retained companion; replan with a dense residency policy, or teach \
                 `measure_residency` to measure the retained form",
                companion.physical_key, companion.resident_bytes
            )));
        }
        if let Some(report) = by_codec.get_mut(codec_id) {
            report.source_bytes = report.source_bytes.saturating_add(companion.source_bytes);
        }
    }
    Ok(by_codec.into_values().collect())
}

/// Whether a stored encoding (undescribed) has a registered codec on this engine.
pub fn encoding_is_supported(encoding: WeightEncoding) -> bool {
    baseline_codec_registry().for_encoding(encoding).is_some()
}

// =================================================================================================
// Loading — MLX's lazy safetensors loader, with a header-rewriting reader for fp8 files.
// =================================================================================================

/// Load every tensor of `path` as lazy MLX arrays. Files without fp8/e8m0 tensors go through the
/// ordinary `Weights::from_file` seam; files with them go through the custom reader that
/// re-presents those dtypes as `U8` (see the module docs) so MLX's lazy `load` primitive can serve
/// them file-backed.
fn load_planned_file(path: &Path) -> Result<HashMap<String, Array>> {
    let layout = gen_core::safetensors_file_tensor_locations(path)?;
    let needs_rewrite = layout.tensors.iter().any(|location| {
        matches!(
            location.header.dtype,
            HeaderDtype::F8_E4M3 | HeaderDtype::F8_E5M2 | HeaderDtype::F8_E8M0
        )
    });
    if !needs_rewrite {
        return Ok(Weights::from_file(path)?.into_tensors());
    }
    let header = rewrite_fp8_header_as_u8(&layout.header_json).map_err(|error| {
        Error::Msg(format!(
            "loading {}: cannot re-present fp8 header to MLX: {error}",
            path.display()
        ))
    })?;
    fp8_view::load_with_rewritten_header(path, header, layout.file_len)
}

/// Rewrite a safetensors header JSON so every `F8_E4M3` / `F8_E5M2` / `F8_E8M0` dtype reads `U8`,
/// preserving the exact byte length (the JSON shrinks — those names are longer than `U8` — and is
/// space-padded back; whitespace between JSON tokens is insignificant). Element size is 1 byte in
/// both readings, so every declared offset stays valid.
fn rewrite_fp8_header_as_u8(header_json: &[u8]) -> std::result::Result<Vec<u8>, String> {
    let mut json: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(header_json).map_err(|error| error.to_string())?;
    for (name, value) in json.iter_mut() {
        if name == "__metadata__" {
            continue;
        }
        let Some(object) = value.as_object_mut() else {
            continue;
        };
        let Some(dtype) = object.get("dtype").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if matches!(dtype, "F8_E4M3" | "F8_E5M2" | "F8_E8M0") {
            object.insert(
                "dtype".to_owned(),
                serde_json::Value::String("U8".to_owned()),
            );
        }
    }
    let mut rewritten = serde_json::to_vec(&json).map_err(|error| error.to_string())?;
    if rewritten.len() > header_json.len() {
        return Err(format!(
            "rewritten header is {} bytes but the original held {}",
            rewritten.len(),
            header_json.len()
        ));
    }
    rewritten.resize(header_json.len(), b' ');
    Ok(rewritten)
}

/// The custom `mlx_io_reader` presenting a byte-identical file whose header names fp8 dtypes `U8`.
mod fp8_view {
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    use mlx_rs::{Array, Stream};

    use crate::{Error, Result};

    struct RewrittenFile {
        file: std::fs::File,
        /// The full re-presented prefix: 8-byte little-endian header length + patched header JSON.
        prefix: Vec<u8>,
        file_len: u64,
        position: u64,
        good: bool,
        label: CString,
    }

    impl RewrittenFile {
        fn read_at(&mut self, data: *mut c_char, n: usize, offset: u64) {
            // SAFETY: mlx-c hands us the destination buffer it allocated for exactly `n` bytes.
            let out = unsafe { std::slice::from_raw_parts_mut(data.cast::<u8>(), n) };
            let mut filled = 0_usize;
            // Prefix region first (patched header), then the untouched file bytes.
            if offset < self.prefix.len() as u64 {
                let start = offset as usize;
                let take = (self.prefix.len() - start).min(n);
                out[..take].copy_from_slice(&self.prefix[start..start + take]);
                filled = take;
            }
            if filled < n {
                let file_offset = offset + filled as u64;
                if self
                    .file
                    .read_exact_at(&mut out[filled..], file_offset)
                    .is_err()
                {
                    // The vtable read cannot return an error; poison the stream so `good()`
                    // reports the failure and zero the buffer rather than leaking uninitialized
                    // memory.
                    out[filled..].fill(0);
                    self.good = false;
                }
            }
        }
    }

    unsafe extern "C" fn is_open(_ctx: *mut c_void) -> bool {
        true
    }

    unsafe extern "C" fn good(ctx: *mut c_void) -> bool {
        (*ctx.cast::<RewrittenFile>()).good
    }

    unsafe extern "C" fn tell(ctx: *mut c_void) -> usize {
        (*ctx.cast::<RewrittenFile>()).position as usize
    }

    unsafe extern "C" fn seek(ctx: *mut c_void, off: i64, whence: c_int) {
        let this = &mut *ctx.cast::<RewrittenFile>();
        let base = match whence {
            seek_whence::SEEK_SET => 0_i64,
            seek_whence::SEEK_CUR => this.position as i64,
            seek_whence::SEEK_END => this.file_len as i64,
            _ => {
                this.good = false;
                return;
            }
        };
        let target = base.saturating_add(off);
        if target < 0 {
            this.good = false;
            return;
        }
        this.position = target as u64;
    }

    unsafe extern "C" fn read(ctx: *mut c_void, data: *mut c_char, n: usize) {
        let this = &mut *ctx.cast::<RewrittenFile>();
        let position = this.position;
        this.read_at(data, n, position);
        this.position += n as u64;
    }

    unsafe extern "C" fn read_at_offset(ctx: *mut c_void, data: *mut c_char, n: usize, off: usize) {
        let this = &mut *ctx.cast::<RewrittenFile>();
        this.read_at(data, n, off as u64);
    }

    unsafe extern "C" fn write(ctx: *mut c_void, _data: *const c_char, _n: usize) {
        (*ctx.cast::<RewrittenFile>()).good = false;
    }

    unsafe extern "C" fn label(ctx: *mut c_void) -> *const c_char {
        (*ctx.cast::<RewrittenFile>()).label.as_ptr()
    }

    unsafe extern "C" fn free(ctx: *mut c_void) {
        drop(Box::from_raw(ctx.cast::<RewrittenFile>()));
    }

    /// `SEEK_*` as mlx-c passes them (`stdio.h` values; identical on every unix libc).
    mod seek_whence {
        use std::os::raw::c_int;
        pub const SEEK_SET: c_int = 0;
        pub const SEEK_CUR: c_int = 1;
        pub const SEEK_END: c_int = 2;
    }

    /// Load a safetensors file through MLX's lazy loader, serving `patched_header` in place of the
    /// on-disk header JSON (identical byte length; see the module docs). The reader object lives
    /// until MLX frees the last lazy array that references it.
    pub(super) fn load_with_rewritten_header(
        path: &Path,
        patched_header: Vec<u8>,
        file_len: u64,
    ) -> Result<HashMap<String, Array>> {
        let file = std::fs::File::open(path)?;
        let mut prefix = Vec::with_capacity(8 + patched_header.len());
        prefix.extend_from_slice(&(patched_header.len() as u64).to_le_bytes());
        prefix.extend_from_slice(&patched_header);
        let state = Box::new(RewrittenFile {
            file,
            prefix,
            file_len,
            position: 0,
            good: true,
            label: CString::new(format!("fp8-as-u8 view of {}", path.display()))
                .unwrap_or_else(|_| CString::new("fp8-as-u8 view").expect("static string")),
        });
        let vtable = mlx_sys::mlx_io_vtable {
            is_open: Some(is_open),
            good: Some(good),
            tell: Some(tell),
            seek: Some(seek),
            read: Some(read),
            read_at_offset: Some(read_at_offset),
            write: Some(write),
            label: Some(label),
            free: Some(free),
        };
        // Matches mlx-rs `Array::load_safetensors` (which routes file loads to the CPU stream).
        let stream = Stream::cpu();
        let mut arrays = HashMap::new();
        // SAFETY: the vtable functions match mlx-c's contract; `state` ownership transfers to the
        // reader, whose `free` callback reclaims the Box when MLX drops the last reference.
        unsafe {
            let reader = mlx_sys::mlx_io_reader_new(Box::into_raw(state).cast(), vtable);
            let mut map = mlx_sys::mlx_map_string_to_array_new();
            let mut metadata = mlx_sys::mlx_map_string_to_string_new();
            let status = mlx_sys::mlx_load_safetensors_reader(
                &mut map,
                &mut metadata,
                reader,
                stream.as_ptr(),
            );
            mlx_sys::mlx_map_string_to_string_free(metadata);
            if status != 0 {
                mlx_sys::mlx_map_string_to_array_free(map);
                mlx_sys::mlx_io_reader_free(reader);
                return Err(Error::Msg(format!(
                    "loading {} through the fp8-as-u8 MLX view failed",
                    path.display()
                )));
            }
            let iterator = mlx_sys::mlx_map_string_to_array_iterator_new(map);
            loop {
                let mut key: *const c_char = std::ptr::null();
                let mut value = mlx_sys::mlx_array_new();
                let status = mlx_sys::mlx_map_string_to_array_iterator_next(
                    &mut key as *mut *const _,
                    &mut value,
                    iterator,
                );
                match status {
                    0 => {
                        let key = std::ffi::CStr::from_ptr(key).to_string_lossy().into_owned();
                        arrays.insert(key, Array::from_ptr(value));
                    }
                    2 => {
                        mlx_sys::mlx_array_free(value);
                        break;
                    }
                    _ => {
                        mlx_sys::mlx_array_free(value);
                        mlx_sys::mlx_map_string_to_array_iterator_free(iterator);
                        mlx_sys::mlx_map_string_to_array_free(map);
                        mlx_sys::mlx_io_reader_free(reader);
                        return Err(Error::Msg(format!(
                            "loading {} through the fp8-as-u8 MLX view: array map iteration failed",
                            path.display()
                        )));
                    }
                }
            }
            mlx_sys::mlx_map_string_to_array_iterator_free(iterator);
            mlx_sys::mlx_map_string_to_array_free(map);
            // The lazy Load primitives hold their own shared reference; this handle is done.
            mlx_sys::mlx_io_reader_free(reader);
        }
        Ok(arrays)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::Path;

    /// Write a minimal safetensors file from `(name, dtype, shape, little-endian payload)` rows.
    pub(crate) fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header_entries = Vec::new();
        let mut body = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            let start = body.len();
            body.extend_from_slice(payload);
            let end = body.len();
            header_entries.push(format!(
                "{:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{end}]}}",
                name, dtype, shape
            ));
        }
        let mut header = format!("{{{}}}", header_entries.join(",")).into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + body.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&body);
        std::fs::write(path, file).expect("write safetensors fixture");
    }

    pub(crate) fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{bf16_bytes, write_safetensors};
    use super::*;
    use gen_core::checkpoint_codec::{
        CompanionRole, CompanionTensorPlan, PlannedResidency, ResidencyMode,
    };

    struct StripModel;

    impl LogicalKeyMapping for StripModel {
        fn mapping_id(&self) -> &'static str {
            "strip-model-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
    }

    /// [`StripModel`] that also **declares** the true logical shape of ONE key.
    ///
    /// A block-padded codec row (MXFP8/NVFP4) is only materializable when the adapter states the
    /// layer's true geometry — otherwise the plan can do nothing but carry the padded stored grid
    /// forward, and decoding it would promote padding to weights (`gen_core` refuses by name,
    /// identically on both engines). The fixtures using this build their padded layer at exactly
    /// the grid declared here, so the declaration is a true statement about the layer.
    struct StripModelDeclaring(&'static str, Vec<usize>);

    impl LogicalKeyMapping for StripModelDeclaring {
        fn mapping_id(&self) -> &'static str {
            "strip-model-declaring-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
        fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
            (logical_key == self.0).then(|| self.1.clone())
        }
    }

    fn fixture_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("logical-weights-{}-", std::process::id()))
            .tempdir()
            .expect("fixture dir")
    }

    /// bf16 value of `x` (round-to-nearest-even) — the reference resident form after an
    /// exactly-rounded f32 computation.
    fn to_bf16(x: f32) -> f32 {
        if x.is_nan() {
            return x;
        }
        let bits = x.to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
        f32::from_bits(rounded)
    }

    fn as_f32(weights: &Weights, key: &str) -> Vec<f32> {
        crate::weights::to_f32(weights.require(key).unwrap())
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    fn eager_read(path: &Path, plan: &LogicalWeightPlan) -> LogicalWeights {
        let mut materialize = |weights: &mut Weights| weights.materialize();
        read_logical_weights(path, plan, LogicalReadMode::Eager(&mut materialize)).expect("read")
    }

    #[test]
    fn baseline_registry_matches_the_implementation_table() {
        let registry = baseline_codec_registry();
        let mut registered: Vec<&str> = registry.codecs().map(|codec| codec.codec_id).collect();
        registered.sort_unstable();
        let mut implemented = CODEC_IMPLEMENTATION_IDS.to_vec();
        implemented.sort_unstable();
        assert_eq!(registered, implemented);
        assert!(encoding_is_supported(WeightEncoding::DenseBf16));
        // The undescribed fp8 cast is supported since sc-20385; undescribed int8 is not (the
        // int8 codec is descriptor-gated).
        assert!(encoding_is_supported(WeightEncoding::Fp8E4M3));
        assert!(encoding_is_supported(WeightEncoding::Fp8E5M2));
        assert!(!encoding_is_supported(WeightEncoding::Int8));
        assert!(!encoding_is_supported(WeightEncoding::UInt8));
        let catalog = register_checkpoint_codecs(ProviderRegistryBuilder::new())
            .build()
            .expect("codec table registers into an empty catalog");
        assert_eq!(
            catalog
                .checkpoint_codecs()
                .codecs()
                .copied()
                .collect::<Vec<_>>(),
            BASELINE_CODECS
        );
        // The table itself is gen-core's, not this crate's: a codec is a backend-portable
        // declaration, and two hand-kept copies could drift into a checkpoint one engine plans and
        // the other refuses with nothing comparing them (mlx-gen builds only on macOS and
        // candle-gen's quantized legs only under `cuda`, so no compilation sees both lists).
        assert_eq!(
            BASELINE_CODECS,
            gen_core::checkpoint_codec::BASELINE_CHECKPOINT_CODECS
        );
    }

    #[test]
    fn eager_read_renames_decodes_and_measures_resident_bytes_from_the_arrays() {
        let dir = fixture_dir();
        let path = dir.path().join("dense.safetensors");
        write_safetensors(
            &path,
            &[
                (
                    "model.b.weight",
                    "BF16",
                    &[2, 2],
                    bf16_bytes(&[1.0, 2.0, 3.0, 4.0]),
                ),
                (
                    "model.a.weight",
                    "BF16",
                    &[3],
                    bf16_bytes(&[0.5, -1.5, 8.0]),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &StripModel).expect("plan");
        assert_eq!(
            plan.logical_keys().collect::<Vec<_>>(),
            ["a.weight", "b.weight"]
        );
        assert_eq!(plan.source_bytes, 6 + 8);

        let mut materialized = false;
        let mut materialize = |weights: &mut Weights| {
            materialized = true;
            weights.materialize()
        };
        let LogicalWeights { weights, receipt } =
            read_logical_weights(&path, &plan, LogicalReadMode::Eager(&mut materialize))
                .expect("read");
        assert!(
            materialized,
            "eager mode must run the provider's materializer"
        );
        let mut keys: Vec<&str> = weights.keys().collect();
        keys.sort_unstable();
        assert_eq!(keys, ["a.weight", "b.weight"]);
        assert!(weights.get("model.a.weight").is_none());

        let a = weights.require("a.weight").unwrap();
        assert_eq!(a.dtype(), Dtype::Bfloat16);
        assert_eq!(as_f32(&weights, "a.weight"), [0.5, -1.5, 8.0]);
        let b = weights.require("b.weight").unwrap();
        assert_eq!(b.shape(), [2, 2]);

        assert_eq!(receipt.mapping_id, "strip-model-test");
        assert_eq!(receipt.tensor_count, 2);
        assert_eq!(receipt.source_bytes, 14);
        assert_eq!(
            receipt.materialization,
            LogicalReadMaterialization::Materialized
        );
        assert_eq!(receipt.residency.len(), 1);
        let report = &receipt.residency[0];
        assert_eq!(report.codec_id, "dense-bf16-v1");
        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.source_bytes, 14);
        let measured: usize = weights
            .keys()
            .map(|key| weights.get(key).unwrap().nbytes())
            .sum();
        assert_eq!(report.resident_bytes, measured as u64);
        assert_eq!(report.resident_bytes, 14);
        assert_eq!(receipt.resident_bytes(), 14);
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
    }

    /// Golden: the `fp8-e4m3-scalar-v1` codec row. Fixture bytes cover the format's landmark
    /// codes (1.0, max 448, min normal 2^-6, min subnormal 2^-9, negatives, ±0); the expected
    /// resident values are the gen-core reference decode (`e4m3(v) · scale` in f32, cast to bf16)
    /// — spec-derived, never measured off a device. Exact equality: both compute the same
    /// exactly-rounded f32 product, then the same f32→bf16 rounding.
    #[test]
    fn fp8_e4m3_scalar_golden_decodes_exactly_against_the_reference() {
        let dir = fixture_dir();
        let path = dir.path().join("e4m3.safetensors");
        let codes: Vec<u8> = vec![0x38, 0x7E, 0x08, 0x01, 0xB9, 0xC0, 0x00, 0x80];
        let scale = 0.03125_f32; // 2^-5, exact
        let descriptor = br#"{"format": "float8_e4m3fn"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[2, 4], codes.clone()),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    scale.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &StripModel).expect("plan");
        assert_eq!(plan.codec_ids(), ["fp8-e4m3-scalar-v1"]);
        let LogicalWeights { weights, receipt } = eager_read(&path, &plan);
        let got = as_f32(&weights, "q.weight");
        let mut expected = Vec::new();
        gen_core::decode_fp8_e4m3fn_scalar(&codes, scale, &mut expected);
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        assert_eq!(got, expected, "e4m3 golden must match the reference decode");
        assert_eq!(
            weights.require("q.weight").unwrap().dtype(),
            Dtype::Bfloat16
        );
        // Residency: 8 fp8 bytes + 4 scale + descriptor consumed → 16 resident bf16 bytes.
        assert_eq!(receipt.resident_bytes(), 16);
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
        let report = &receipt.residency[0];
        assert_eq!(report.codec_id, "fp8-e4m3-scalar-v1");
        assert_eq!(
            report.source_bytes,
            8 + 4 + descriptor.len() as u64,
            "the codec's source bytes cover the weight and its companions"
        );
    }

    /// Golden: the `fp8-e5m2-scalar-v1` codec row, plus the undescribed plain-cast form of the
    /// same codec (unit scale) in one file — per-layer dispatch inside one fp8 encoding family.
    #[test]
    fn fp8_e5m2_scalar_and_plain_cast_golden_decode_exactly() {
        let dir = fixture_dir();
        let path = dir.path().join("e5m2.safetensors");
        // E5M2 landmarks: 1.0, max 57344, min normal 2^-14, min subnormal 2^-16, -1.75, ±0, 3.0.
        let codes: Vec<u8> = vec![0x3C, 0x7B, 0x04, 0x01, 0xBF, 0x00, 0x80, 0x42];
        let scale = 0.25_f32;
        let descriptor = br#"{"format": "float8_e5m2"}"#;
        // A second, undescribed plain-cast E5M2 tensor (rank-1, as a cast bias would be).
        let plain: Vec<u8> = vec![0x3C, 0x40, 0xC2, 0x01];
        write_safetensors(
            &path,
            &[
                ("model.k.weight", "F8_E5M2", &[2, 4], codes.clone()),
                (
                    "model.k.weight_scale",
                    "F32",
                    &[1],
                    scale.to_le_bytes().to_vec(),
                ),
                (
                    "model.k.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
                ("model.k.bias", "F8_E5M2", &[4], plain.clone()),
            ],
        );
        let plan = plan_logical_weights(&path, &StripModel).expect("plan");
        assert_eq!(plan.codec_ids(), ["fp8-e5m2-scalar-v1"]);
        let by_key: std::collections::BTreeMap<&str, &LogicalTensorPlan> = plan
            .tensors
            .iter()
            .map(|tensor| (tensor.logical_key.as_str(), tensor))
            .collect();
        assert!(matches!(
            &by_key["k.weight"].codec,
            TensorCodecSpec::ScalarFp8 {
                scale: ScalarScaleSource::Companion { .. },
                ..
            }
        ));
        assert!(matches!(
            &by_key["k.bias"].codec,
            TensorCodecSpec::ScalarFp8 {
                scale: ScalarScaleSource::Unit,
                ..
            }
        ));
        let LogicalWeights { weights, receipt } = eager_read(&path, &plan);
        let mut expected = Vec::new();
        gen_core::decode_fp8_e5m2_scalar(&codes, scale, &mut expected);
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        assert_eq!(as_f32(&weights, "k.weight"), expected);
        let mut expected_plain = Vec::new();
        gen_core::decode_fp8_e5m2_scalar(&plain, 1.0, &mut expected_plain);
        let expected_plain: Vec<f32> = expected_plain.into_iter().map(to_bf16).collect();
        assert_eq!(as_f32(&weights, "k.bias"), expected_plain);
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
    }

    /// Golden: the `mxfp8-v1` codec row — stored [32, 64] (32-padded), logical = stored here, block
    /// scales in the cuBLAS swizzle. Expected values are the gen-core reference decode
    /// (`decode_mxfp8`), which itself is pinned against comfy_kitchen's eager reference. Exact:
    /// power-of-two scaling in f32 is exactly rounded.
    #[test]
    fn mxfp8_golden_unswizzles_scales_and_decodes_blocks_exactly() {
        let dir = fixture_dir();
        let path = dir.path().join("mxfp8.safetensors");
        // 64 rows exercises BOTH 32-row halves of a 128-row swizzle tile (the sub-tile transpose);
        // 96 columns = 3 blocks, so the swizzled scale matrix carries a padded fourth column.
        let stored = [64_usize, 96];
        // Deterministic non-trivial payload: cycle through fp8 codes, avoiding NaN (0x7F/0xFF).
        let values: Vec<u8> = (0..stored[0] * stored[1])
            .map(|index| {
                let byte = (index * 7 + 3) as u8;
                if byte & 0x7F == 0x7F {
                    0x38
                } else {
                    byte
                }
            })
            .collect();
        let scale_shape = gen_core::mxfp8_scale_shape([stored[0], stored[1]]);
        let mut scales = vec![0_u8; scale_shape[0] * scale_shape[1]];
        for row in 0..stored[0] {
            for block in 0..stored[1] / gen_core::MXFP8_BLOCK {
                // Exponents around 1.0: 2^-2 .. 2^2.
                scales[gen_core::mxfp8_swizzled_scale_index([stored[0], stored[1]], row, block)] =
                    125 + ((row + block) % 5) as u8;
            }
        }
        let descriptor = br#"{"format": "mxfp8"}"#;
        write_safetensors(
            &path,
            &[
                ("model.v.weight", "F8_E4M3", &stored, values.clone()),
                (
                    "model.v.weight_scale",
                    "F8_E8M0",
                    &scale_shape,
                    scales.clone(),
                ),
                (
                    "model.v.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        // The layer IS its stored 32-aligned grid here, and since sc-20651 the adapter has to say
        // so: an undeclared padded grid plans (for pricing) but refuses to materialize, because
        // nothing in the file distinguishes it from a grid with pad rows/columns in it.
        let plan = plan_logical_weights(&path, &StripModelDeclaring("v.weight", stored.to_vec()))
            .expect("plan");
        assert_eq!(plan.codec_ids(), ["mxfp8-v1"]);
        assert_eq!(plan.tensors[0].shape, vec![64, 96]);
        let LogicalWeights { weights, receipt } = eager_read(&path, &plan);
        let mut expected = Vec::new();
        gen_core::decode_mxfp8(
            &values,
            &scales,
            [stored[0], stored[1]],
            [stored[0], stored[1]],
            &mut expected,
        )
        .unwrap();
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        assert_eq!(as_f32(&weights, "v.weight"), expected);
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
        assert_eq!(receipt.resident_bytes(), (64 * 96 * 2) as u64);
    }

    /// MXFP8 padding/tails: an adapter-declared logical shape [5, 40] inside stored [32, 64] — a
    /// non-block-aligned column tail (40 = one full block + 8 elements of block 1) and padded rows.
    /// Padding carries poison values so a wrong unpad or wrong block/scale pairing shows up.
    #[test]
    fn mxfp8_unpads_declared_logical_shapes_with_non_block_aligned_tails() {
        struct DeclaredShape;
        impl LogicalKeyMapping for DeclaredShape {
            fn mapping_id(&self) -> &'static str {
                "declared-shape-test"
            }
            fn logical_key(&self, physical_key: &str) -> Option<String> {
                physical_key.strip_prefix("model.").map(str::to_owned)
            }
            fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
                (logical_key == "v.weight").then(|| vec![5, 40])
            }
        }
        let dir = fixture_dir();
        let path = dir.path().join("mxfp8-padded.safetensors");
        let stored = [32_usize, 64];
        let mut values = vec![0x7E_u8; stored[0] * stored[1]]; // 448 poison everywhere
        for row in 0..5 {
            for col in 0..40 {
                values[row * stored[1] + col] = 0x38 + ((row + col) % 4) as u8;
            }
        }
        let scale_shape = gen_core::mxfp8_scale_shape([stored[0], stored[1]]);
        let mut scales = vec![0xFF_u8; scale_shape[0] * scale_shape[1]]; // NaN poison
        for row in 0..5 {
            for block in 0..2 {
                scales[gen_core::mxfp8_swizzled_scale_index([stored[0], stored[1]], row, block)] =
                    126 + ((row + block) % 3) as u8;
            }
        }
        let descriptor = br#"{"format": "mxfp8"}"#;
        write_safetensors(
            &path,
            &[
                ("model.v.weight", "F8_E4M3", &stored, values.clone()),
                ("model.v.weight_scale", "U8", &scale_shape, scales.clone()),
                (
                    "model.v.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &DeclaredShape).expect("plan");
        assert_eq!(plan.tensors[0].shape, vec![5, 40]);
        assert_eq!(
            plan.tensors[0].residency,
            PlannedResidency {
                mode: ResidencyMode::Dense,
                resident_bytes: 5 * 40 * 2
            }
        );
        let LogicalWeights { weights, receipt } = eager_read(&path, &plan);
        let mut expected = Vec::new();
        gen_core::decode_mxfp8(
            &values,
            &scales,
            [stored[0], stored[1]],
            [5, 40],
            &mut expected,
        )
        .unwrap();
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        let got = as_f32(&weights, "v.weight");
        assert_eq!(weights.require("v.weight").unwrap().shape(), [5, 40]);
        assert_eq!(got, expected);
        assert!(
            got.iter().all(|value| value.is_finite() && *value < 400.0),
            "poison from the padding region leaked into the logical tensor"
        );
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
    }

    /// AC1 (MLX). The NVFP4 golden fixture decodes on the dense fallback to a **hand-derived**
    /// expectation: E2M1 grid value × E4M3 block scale × the `weight_scale_2` global, with both the
    /// grid and the scale values written out literally below rather than obtained from the decoder
    /// under test.
    ///
    /// The fixture is 16-padded on both axes (`[32, 64]` stored, `[20, 50]` declared logical), so
    /// the unpad is on the path and blocks that cover only padding carry NaN poison. It also
    /// exercises MLX's header-rewriting reader: the block scales are stored `F8_E4M3`, a dtype MLX
    /// has no array type for and re-presents as `U8`.
    #[test]
    fn nvfp4_golden_unswizzles_unpads_and_decodes_to_a_hand_derived_table() {
        struct DeclaredShape;
        impl LogicalKeyMapping for DeclaredShape {
            fn mapping_id(&self) -> &'static str {
                "declared-shape-test"
            }
            fn logical_key(&self, physical_key: &str) -> Option<String> {
                physical_key.strip_prefix("model.").map(str::to_owned)
            }
            fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
                (logical_key == "n.weight").then(|| vec![20, 50])
            }
        }
        // The E2M1 grid and the E4M3 scale values, written out by hand from the format tables.
        const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        const SCALE_BYTES: [u8; 4] = [0x38, 0x40, 0x30, 0x3C];
        const SCALE_VALUES: [f32; 4] = [1.0, 2.0, 0.5, 1.5];
        const GLOBAL: f32 = 0.25;

        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-padded.safetensors");
        let stored = [32_usize, 64];
        let logical = [20_usize, 50];
        let row_bytes = stored[1] / 2;

        // Real region: column c holds E2M1 code (c % 8). Padding keeps the zero code the format
        // requires; the block scales covering padding-only blocks carry NaN poison instead.
        let mut packed = vec![0_u8; stored[0] * row_bytes];
        for row in 0..logical[0] {
            for col in 0..logical[1] {
                let code = (col % 8) as u8;
                let byte = &mut packed[row * row_bytes + col / 2];
                // ComfyUI packs the even element in the HIGH nibble.
                if col.is_multiple_of(2) {
                    *byte = (*byte & 0x0F) | (code << 4);
                } else {
                    *byte = (*byte & 0xF0) | code;
                }
            }
        }
        let scale_shape = gen_core::nvfp4_scale_shape(stored);
        let mut scales = vec![0x7F_u8; scale_shape[0] * scale_shape[1]]; // E4M3 NaN poison
        for row in 0..logical[0] {
            for block in 0..logical[1].div_ceil(16) {
                scales[gen_core::nvfp4_swizzled_scale_index(stored, row, block)] =
                    SCALE_BYTES[(row + block) % 4];
            }
        }
        write_safetensors(
            &path,
            &[
                (
                    "model.n.weight",
                    "U8",
                    &[stored[0], row_bytes],
                    packed.clone(),
                ),
                ("model.n.weight_scale", "F8_E4M3", &scale_shape, scales),
                (
                    "model.n.weight_scale_2",
                    "F32",
                    &[],
                    GLOBAL.to_le_bytes().to_vec(),
                ),
                (
                    "model.n.comfy_quant",
                    "U8",
                    &[19],
                    br#"{"format": "nvfp4"}"#.to_vec(),
                ),
            ],
        );

        let plan = plan_logical_weights(&path, &DeclaredShape).expect("plan");
        assert_eq!(plan.codec_ids(), vec!["nvfp4-v1"]);
        assert_eq!(plan.tensors[0].shape, logical.to_vec());
        assert_eq!(
            plan.tensors[0].residency,
            PlannedResidency {
                mode: ResidencyMode::Dense,
                resident_bytes: (20 * 50 * 2) as u64
            },
            "MLX has no FP4 matmul on the codec seam: always the dense fallback"
        );

        let LogicalWeights { weights, receipt } = eager_read(&path, &plan);
        assert_eq!(weights.require("n.weight").unwrap().shape(), [20, 50]);
        let got = as_f32(&weights, "n.weight");
        for row in 0..logical[0] {
            for col in 0..logical[1] {
                let expected = to_bf16(E2M1[col % 8] * SCALE_VALUES[(row + col / 16) % 4] * GLOBAL);
                assert_eq!(got[row * logical[1] + col], expected, "row {row} col {col}");
            }
        }
        assert!(
            got.iter().all(|value| value.is_finite()),
            "NaN poison from a padding-only block scale leaked into the logical tensor"
        );
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
    }

    /// The int8 per-row codec row — the former bespoke Krea arm's exact math, now engine-level.
    #[test]
    fn int8_per_row_golden_decodes_codes_times_row_scale() {
        let dir = fixture_dir();
        let path = dir.path().join("int8.safetensors");
        let codes: Vec<u8> = vec![1, 0xFE, 3, 0xFC, 5, 0xFA]; // i8: 1, -2, 3, -4, 5, -6
        let scales = [0.5_f32, 2.0];
        let descriptor = br#"{"format": "int8_tensorwise", "per_row": true}"#;
        write_safetensors(
            &path,
            &[
                ("model.o.weight", "I8", &[2, 3], codes.clone()),
                (
                    "model.o.weight_scale",
                    "F32",
                    &[2, 1],
                    scales
                        .iter()
                        .flat_map(|scale| scale.to_le_bytes())
                        .collect(),
                ),
                (
                    "model.o.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &StripModel).expect("plan");
        assert_eq!(plan.codec_ids(), ["int8-per-row-v1"]);
        let LogicalWeights { weights, receipt } = eager_read(&path, &plan);
        assert_eq!(
            as_f32(&weights, "o.weight"),
            [0.5, -1.0, 1.5, -8.0, 10.0, -12.0]
        );
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
        assert_eq!(receipt.resident_bytes(), 12);
    }

    /// **sc-20651 feature-end review (major): the matrix arms refuse a non-rank-2 plan BY NAME.**
    ///
    /// The MXFP8, NVFP4 and int8-per-row arms index `tensor.shape[0]` / `tensor.shape[1]`
    /// positionally. [`read_logical_weights`] is a public entry over a caller-supplied
    /// [`LogicalWeightPlan`] whose fields are public, so a plan that did not come from the plan
    /// compiler reaches those arms unchecked — and before
    /// [`LogicalTensorPlan::matrix_rank_refusal`] it PANICKED out of bounds instead of refusing.
    ///
    /// This is the mirror of candle-gen's `matrix_codecs_refuse_a_non_matrix_logical_rank`: same
    /// three codecs, same rank-1 and rank-3 mutations, same refusal text (the message is gen-core's
    /// precisely so the two engines cannot drift — nothing compiles both, mlx-gen being macOS-only
    /// and candle-gen's quantized legs `cuda`-gated).
    #[test]
    fn matrix_codecs_refuse_a_non_matrix_logical_rank() {
        let dir = fixture_dir();

        // int8-per-row: the golden fixture's [2, 3] grid.
        let int8_path = dir.path().join("rank-int8.safetensors");
        let int8_descriptor = br#"{"format": "int8_tensorwise", "per_row": true}"#;
        write_safetensors(
            &int8_path,
            &[
                (
                    "model.o.weight",
                    "I8",
                    &[2, 3],
                    vec![1, 0xFE, 3, 0xFC, 5, 0xFA],
                ),
                (
                    "model.o.weight_scale",
                    "F32",
                    &[2, 1],
                    [0.5_f32, 2.0]
                        .iter()
                        .flat_map(|scale| scale.to_le_bytes())
                        .collect(),
                ),
                (
                    "model.o.comfy_quant",
                    "U8",
                    &[int8_descriptor.len()],
                    int8_descriptor.to_vec(),
                ),
            ],
        );
        let int8_plan = plan_logical_weights(&int8_path, &StripModel).expect("int8 plan");
        assert_eq!(int8_plan.codec_ids(), ["int8-per-row-v1"]);

        // MXFP8: a 32-aligned [32, 32] grid the adapter DECLARES, so the undeclared-padding refusal
        // is not what fires below.
        let mxfp8_path = dir.path().join("rank-mxfp8.safetensors");
        let mxfp8_scale_shape = gen_core::mxfp8_scale_shape([32, 32]);
        let mxfp8_descriptor = br#"{"format": "mxfp8"}"#;
        write_safetensors(
            &mxfp8_path,
            &[
                (
                    "model.v.weight",
                    "F8_E4M3",
                    &[32, 32],
                    vec![0x38_u8; 32 * 32],
                ),
                (
                    "model.v.weight_scale",
                    "U8",
                    &mxfp8_scale_shape,
                    vec![127_u8; mxfp8_scale_shape[0] * mxfp8_scale_shape[1]],
                ),
                (
                    "model.v.comfy_quant",
                    "U8",
                    &[mxfp8_descriptor.len()],
                    mxfp8_descriptor.to_vec(),
                ),
            ],
        );
        let mxfp8_plan =
            plan_logical_weights(&mxfp8_path, &StripModelDeclaring("v.weight", vec![32, 32]))
                .expect("mxfp8 plan");
        assert_eq!(mxfp8_plan.codec_ids(), ["mxfp8-v1"]);

        // NVFP4: the value golden's [32, 64] stored grid, declared UNPADDED so the padding-poison
        // check is not what fires below either — the rank is the only thing under test.
        let nvfp4_path = dir.path().join("rank-nvfp4.safetensors");
        let stored = [32_usize, 64];
        let nvfp4_scale_shape = gen_core::nvfp4_scale_shape(stored);
        let nvfp4_descriptor = br#"{"format": "nvfp4"}"#;
        write_safetensors(
            &nvfp4_path,
            &[
                (
                    "model.n.weight",
                    "U8",
                    &[stored[0], stored[1] / 2],
                    vec![0x12_u8; stored[0] * stored[1] / 2],
                ),
                (
                    "model.n.weight_scale",
                    "F8_E4M3",
                    &nvfp4_scale_shape,
                    vec![0x38_u8; nvfp4_scale_shape[0] * nvfp4_scale_shape[1]],
                ),
                (
                    "model.n.weight_scale_2",
                    "F32",
                    &[],
                    0.25_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.n.comfy_quant",
                    "U8",
                    &[nvfp4_descriptor.len()],
                    nvfp4_descriptor.to_vec(),
                ),
            ],
        );
        let nvfp4_plan = plan_logical_weights(
            &nvfp4_path,
            &StripModelDeclaring("n.weight", stored.to_vec()),
        )
        .expect("nvfp4 plan");
        assert_eq!(nvfp4_plan.codec_ids(), ["nvfp4-v1"]);

        for (codec, path, plan) in [
            ("int8-per-row-v1", int8_path.as_path(), &int8_plan),
            ("mxfp8-v1", mxfp8_path.as_path(), &mxfp8_plan),
            ("nvfp4-v1", nvfp4_path.as_path(), &nvfp4_plan),
        ] {
            // The control: the unmutated plan reads, so the refusals below are the RANK and not a
            // broken fixture.
            eager_read(path, plan);

            for wrong in [vec![6_usize], vec![1_usize, 2, 3]] {
                let mut forced = plan.clone();
                forced.tensors[0].shape = wrong.clone();
                let error = read_logical_weights(path, &forced, LogicalReadMode::Deferred)
                    .err()
                    .unwrap_or_else(|| panic!("{codec}: a rank-{} plan must refuse", wrong.len()))
                    .to_string();
                assert!(
                    error.contains(&format!("codec {codec}:"))
                        && error.contains("expected rank 2")
                        && error.contains(&format!("observed rank {}", wrong.len())),
                    "{codec}: the rank-{} refusal must name the codec, the expected rank and the \
                     observed rank, got: {error}",
                    wrong.len()
                );
            }
        }
    }

    /// One file, four codecs: dense bf16 + scalar e4m3 + mxfp8 + int8 per row dispatch per layer,
    /// and the receipt reports one residency row per codec with companion bytes attributed.
    #[test]
    fn mixed_checkpoint_dispatches_per_layer_with_one_residency_row_per_codec() {
        let dir = fixture_dir();
        let path = dir.path().join("mixed.safetensors");
        let e4m3_descriptor = br#"{"format": "float8_e4m3fn"}"#;
        let int8_descriptor = br#"{"format": "int8_tensorwise", "per_row": true}"#;
        let mxfp8_descriptor = br#"{"format": "mxfp8"}"#;
        let mx_stored = [32_usize, 32];
        let mx_values = vec![0x38_u8; 32 * 32];
        let mx_scale_shape = gen_core::mxfp8_scale_shape([32, 32]);
        let mut mx_scales = vec![0_u8; mx_scale_shape[0] * mx_scale_shape[1]];
        for row in 0..32 {
            mx_scales[gen_core::mxfp8_swizzled_scale_index([32, 32], row, 0)] = 128;
            // ×2
        }
        write_safetensors(
            &path,
            &[
                ("model.dense.weight", "BF16", &[2], bf16_bytes(&[1.0, -1.0])),
                (
                    "model.q.weight",
                    "F8_E4M3",
                    &[1, 4],
                    vec![0x38, 0x40, 0x48, 0xB8],
                ),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[e4m3_descriptor.len()],
                    e4m3_descriptor.to_vec(),
                ),
                ("model.o.weight", "I8", &[1, 2], vec![4, 0xFE]),
                (
                    "model.o.weight_scale",
                    "F32",
                    &[1],
                    0.5_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.o.comfy_quant",
                    "U8",
                    &[int8_descriptor.len()],
                    int8_descriptor.to_vec(),
                ),
                ("model.v.weight", "F8_E4M3", &mx_stored, mx_values),
                ("model.v.weight_scale", "U8", &mx_scale_shape, mx_scales),
                (
                    "model.v.comfy_quant",
                    "U8",
                    &[mxfp8_descriptor.len()],
                    mxfp8_descriptor.to_vec(),
                ),
            ],
        );
        // The MXFP8 layer is built at its stored 32-aligned grid; the adapter declares it, which
        // sc-20651 requires before a block-padded row may be materialized at all.
        let plan =
            plan_logical_weights(&path, &StripModelDeclaring("v.weight", mx_stored.to_vec()))
                .expect("plan");
        assert_eq!(
            plan.codec_ids(),
            [
                "dense-bf16-v1",
                "fp8-e4m3-scalar-v1",
                "int8-per-row-v1",
                "mxfp8-v1"
            ]
        );
        let LogicalWeights { weights, receipt } = eager_read(&path, &plan);
        assert_eq!(as_f32(&weights, "dense.weight"), [1.0, -1.0]);
        assert_eq!(as_f32(&weights, "q.weight"), [2.0, 4.0, 8.0, -2.0]);
        assert_eq!(as_f32(&weights, "o.weight"), [2.0, -1.0]);
        assert!(as_f32(&weights, "v.weight")
            .iter()
            .all(|value| *value == 2.0));
        assert_eq!(receipt.residency.len(), 4);
        assert_eq!(receipt.resident_bytes(), plan.resident_bytes());
        let by_codec: std::collections::BTreeMap<&str, &gen_core::CodecResidencyReport> = receipt
            .residency
            .iter()
            .map(|report| (report.codec_id, report))
            .collect();
        assert_eq!(by_codec["dense-bf16-v1"].resident_bytes, 4);
        assert_eq!(by_codec["fp8-e4m3-scalar-v1"].resident_bytes, 8);
        assert_eq!(by_codec["int8-per-row-v1"].resident_bytes, 4);
        assert_eq!(by_codec["mxfp8-v1"].resident_bytes, 32 * 32 * 2);
        // Companion bytes belong to their codec's source accounting.
        assert_eq!(
            by_codec["fp8-e4m3-scalar-v1"].source_bytes,
            4 + 4 + e4m3_descriptor.len() as u64
        );
        // Every byte of the data region is a codec's source byte.
        let total: u64 = receipt
            .residency
            .iter()
            .map(|report| report.source_bytes)
            .sum();
        assert_eq!(total, plan.source_bytes);
    }

    #[test]
    fn deferred_read_leaves_payloads_lazy_and_reports_no_residency() {
        let dir = fixture_dir();
        let path = dir.path().join("deferred.safetensors");
        write_safetensors(&path, &[("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0]))]);
        let plan = plan_logical_weights(&path, &StripModel).unwrap();
        let LogicalWeights { weights, receipt } =
            read_logical_weights(&path, &plan, LogicalReadMode::Deferred).unwrap();
        assert!(weights.get("w").is_some());
        assert_eq!(
            receipt.materialization,
            LogicalReadMaterialization::Deferred
        );
        assert!(receipt.residency.is_empty());
        assert_eq!(receipt.resident_bytes(), 0);
        assert_eq!(receipt.source_bytes, 4);
    }

    /// A deferred fp8 read stays lazy too: the codec graph is built over the file-backed U8 view
    /// and the payload is not evaluated until a consumer asks — the property the bounded
    /// block-window loaders rely on when they reopen a native file per window.
    #[test]
    fn deferred_fp8_read_still_decodes_correct_values_on_demand() {
        let dir = fixture_dir();
        let path = dir.path().join("deferred-fp8.safetensors");
        write_safetensors(
            &path,
            &[("model.p.weight", "F8_E4M3", &[2], vec![0x38, 0xC0])],
        );
        let plan = plan_logical_weights(&path, &StripModel).unwrap();
        let LogicalWeights { weights, receipt } =
            read_logical_weights(&path, &plan, LogicalReadMode::Deferred).unwrap();
        assert_eq!(
            receipt.materialization,
            LogicalReadMaterialization::Deferred
        );
        assert!(receipt.residency.is_empty());
        assert_eq!(as_f32(&weights, "p.weight"), [1.0, -2.0]);
    }

    #[test]
    fn planning_refuses_foreign_keys_descriptor_defects_and_unregistered_formats() {
        let dir = fixture_dir();
        let foreign = dir.path().join("foreign.safetensors");
        write_safetensors(&foreign, &[("other.w", "BF16", &[1], bf16_bytes(&[1.0]))]);
        let error = plan_logical_weights(&foreign, &StripModel).unwrap_err();
        assert!(
            error.to_string().contains("\"other.w\"")
                && error.to_string().contains("no canonical logical key"),
            "{error}"
        );

        // Packed u8 nibbles (comfy fp4-style) have no codec: refuse naming the format.
        let packed = dir.path().join("packed.safetensors");
        write_safetensors(
            &packed,
            &[
                ("model.ok", "BF16", &[1], bf16_bytes(&[1.0])),
                ("model.packed", "U8", &[4], vec![1, 2, 3, 4]),
            ],
        );
        let error = plan_logical_weights(&packed, &StripModel).unwrap_err();
        assert!(
            error.to_string().contains("\"model.packed\"")
                && error.to_string().contains("uint8")
                && error
                    .to_string()
                    .contains("no checkpoint codec is registered"),
            "{error}"
        );

        // A malformed descriptor names the exact layer and defect, from the payload bytes.
        let bad = dir.path().join("bad-descriptor.safetensors");
        let descriptor = br#"{"format": "int4_awq"}"#;
        write_safetensors(
            &bad,
            &[
                ("model.q.weight", "F8_E4M3", &[1, 2], vec![0x38, 0x40]),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let error = plan_logical_weights(&bad, &StripModel)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("\"model.q\"") && error.contains("int4_awq"),
            "{error}"
        );

        // sc-20641: `nvfp4` is no longer an unsupported format — it is a registered codec — but a
        // layer that DECLARES nvfp4 while storing an fp8 weight is still refused, by dtype.
        let mismatched = dir.path().join("nvfp4-dtype-mismatch.safetensors");
        let nvfp4_descriptor = br#"{"format": "nvfp4"}"#;
        write_safetensors(
            &mismatched,
            &[
                ("model.q.weight", "F8_E4M3", &[16, 16], vec![0x38; 256]),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[nvfp4_descriptor.len()],
                    nvfp4_descriptor.to_vec(),
                ),
            ],
        );
        let error = plan_logical_weights(&mismatched, &StripModel)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("model.q.weight")
                && error.contains("nvfp4")
                && error.contains("F8_E4M3"),
            "{error}"
        );

        // An fp8 weight with a scale companion but NO descriptor is the FLUX.2 inline convention.
        let inline = dir.path().join("inline.safetensors");
        write_safetensors(
            &inline,
            &[
                ("model.q.weight", "F8_E4M3", &[1, 2], vec![0x38, 0x40]),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    1.0_f32.to_le_bytes().to_vec(),
                ),
            ],
        );
        let error = plan_logical_weights(&inline, &StripModel)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("\"model.q.weight_scale\"") && error.contains("inline-scale"),
            "{error}"
        );
    }

    struct Reverse;

    impl LogicalKeyMapping for Reverse {
        fn mapping_id(&self) -> &'static str {
            "reverse-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            match physical_key {
                "model.a" => Some("zeta".to_owned()),
                "model.b" => Some("alpha".to_owned()),
                _ => None,
            }
        }
    }

    #[test]
    fn read_accepts_plans_whose_logical_order_differs_from_physical_order() {
        // The plan is sorted by logical key; the on-disk check must compare physical key SETS, not
        // positions, or a perfectly valid remap refuses as "changed since planning".
        let dir = fixture_dir();
        let path = dir.path().join("reorder.safetensors");
        write_safetensors(
            &path,
            &[
                ("model.a", "BF16", &[1], bf16_bytes(&[1.0])),
                ("model.b", "BF16", &[1], bf16_bytes(&[2.0])),
            ],
        );
        let plan = plan_logical_weights(&path, &Reverse).unwrap();
        assert_eq!(plan.logical_keys().collect::<Vec<_>>(), ["alpha", "zeta"]);
        assert_eq!(
            plan.physical_keys().collect::<Vec<_>>(),
            ["model.b", "model.a"]
        );
        let LogicalWeights { weights, .. } =
            read_logical_weights(&path, &plan, LogicalReadMode::Deferred)
                .expect("a reordering remap is not source drift");
        assert_eq!(as_f32(&weights, "alpha"), [2.0]);
    }

    /// **sc-20651 blocker 4: this engine refuses a `Packed` plan entry instead of dense-decoding it.**
    ///
    /// Every arm of `decode` produces the codec's dense resident form, because MLX has no packed
    /// fp8/int8/fp4 matmul on this seam — so this engine's own policy is
    /// [`DenseResidencyPolicy`]. The reader was *assuming* that rather than checking it. A `Packed`
    /// entry arriving from a foreign policy (or from a policy that grows a packed row) would
    /// materialize the dense reconstruction of a layer admission priced at its stored packing:
    /// bf16 is 2 bytes/element against fp8's 1, so the load silently costs twice the admitted
    /// footprint, with the receipt reporting the larger number long after admission committed to
    /// the smaller one.
    #[test]
    fn read_refuses_a_packed_plan_entry_rather_than_dense_decoding_it() {
        let dir = fixture_dir();
        let path = dir.path().join("packed-plan.safetensors");
        let descriptor = br#"{"format": "float8_e4m3fn"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[2, 4], vec![0x38; 8]),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    1.0_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );

        // The engine's own plan is dense, and reads.
        let dense = plan_logical_weights(&path, &StripModel).expect("plan");
        assert_eq!(dense.tensors[0].residency.mode, ResidencyMode::Dense);
        eager_read(&path, &dense);

        // The same file under a plan that priced the layer PACKED — the stored 8 bytes rather than
        // the 16 bytes of bf16 this engine would materialize.
        let mut packed = dense.clone();
        packed.tensors[0].residency = PlannedResidency {
            mode: ResidencyMode::Packed,
            resident_bytes: 8,
        };
        let mut materialize = |weights: &mut Weights| weights.materialize();
        let error = read_logical_weights(&path, &packed, LogicalReadMode::Eager(&mut materialize))
            .err()
            .expect("a packed plan entry must refuse on an engine with no packed leg");
        let error = error.to_string();
        assert!(error.contains("model.q.weight"), "{error}");
        assert!(error.contains("Packed"), "{error}");
        assert!(
            error.contains("dense residency policy"),
            "the refusal must name the fix: {error}"
        );
    }

    /// **sc-20651 blocker 5: a scalar-fp8 scale companion must hold exactly one value.**
    ///
    /// `scalar_fp8` means ONE per-tensor scale. The decode multiplied by whatever the companion
    /// held, and MLX broadcasts: a `[cols]` companion would broadcast across rows and a `[rows, 1]`
    /// one across columns, so a per-row or per-column scale convention mis-read as scalar-fp8
    /// decoded to plausible-but-wrong weights instead of refusing. The dtype check that was there
    /// cannot catch it — both conventions are `F32`.
    #[test]
    fn scalar_fp8_refuses_a_multi_element_scale_companion() {
        let dir = fixture_dir();
        let path = dir.path().join("broadcast-scale.safetensors");
        write_safetensors(
            &path,
            &[
                // Undescribed fp8 (the plain cast), so the plan gives it `ScalarScaleSource::Unit`
                // and no companion of its own.
                ("model.q", "F8_E4M3", &[2, 4], vec![0x38; 8]),
                // A per-COLUMN F32 vector that is a legitimate weight of this fixture; the plan
                // names it as an ordinary dense row.
                (
                    "model.zscale",
                    "F32",
                    &[4],
                    [2.0_f32, 2.0, 2.0, 2.0]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &StripModel).expect("plan");

        // Re-point the fp8 layer's scale at that four-element vector — the shape a per-column
        // convention would have. `zscale` sorts after `q`, so it is still resident when `q`
        // decodes.
        let mut broadcast = plan.clone();
        let fp8 = broadcast
            .tensors
            .iter_mut()
            .find(|tensor| tensor.physical_key == "model.q")
            .expect("the fp8 layer is planned");
        fp8.codec = TensorCodecSpec::ScalarFp8 {
            scale: ScalarScaleSource::Companion {
                physical_key: "model.zscale".to_string(),
            },
            input_scale: None,
            full_precision_matrix_mult: false,
        };

        let mut materialize = |weights: &mut Weights| weights.materialize();
        let error =
            read_logical_weights(&path, &broadcast, LogicalReadMode::Eager(&mut materialize))
                .err()
                .expect("a broadcastable multi-element scale must refuse")
                .to_string();
        assert!(error.contains("model.zscale"), "{error}");
        assert!(error.contains("ONE F32 scalar"), "{error}");
        assert!(error.contains("4 elements"), "{error}");
    }

    /// **sc-20651 major 3: an undeclared block-padded layer plans but must not materialize.**
    ///
    /// The MLX half of the same rule the Candle engine applies, from the same gen-core refusal, so
    /// both engines say the same thing about the same checkpoint. MXFP8 storage is 32-padded on
    /// both axes and the file records no true geometry, so an adapter that declares nothing leaves
    /// the plan carrying the *padded* grid — and decoding it hands the model padding as weights.
    /// Planning still succeeds because a plan is also a pricing artifact.
    #[test]
    fn an_undeclared_block_padded_layer_plans_but_refuses_to_materialize() {
        let dir = fixture_dir();
        let path = dir.path().join("mxfp8-undeclared.safetensors");
        let (rows, cols) = (32_usize, 64_usize);
        let descriptor = br#"{"format": "mxfp8"}"#;
        let scale_shape = gen_core::mxfp8_scale_shape([rows, cols]);
        write_safetensors(
            &path,
            &[
                (
                    "model.q.weight",
                    "F8_E4M3",
                    &[rows, cols],
                    vec![0x38; rows * cols],
                ),
                (
                    "model.q.weight_scale",
                    "U8",
                    &scale_shape,
                    vec![127_u8; scale_shape.iter().product()],
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );

        let undeclared = plan_logical_weights(&path, &StripModel)
            .expect("an undeclared padded checkpoint still PLANS: pricing is not materialization");
        assert_eq!(
            undeclared.tensors[0].undeclared_padded_storage(),
            Some([rows, cols]),
            "the plan records that its logical shape is only the stored grid"
        );

        let mut materialize = |weights: &mut Weights| weights.materialize();
        let error =
            read_logical_weights(&path, &undeclared, LogicalReadMode::Eager(&mut materialize))
                .err()
                .expect("materializing an undeclared padded grid must be refused")
                .to_string();
        assert!(error.contains("model.q.weight"), "{error}");
        assert!(error.contains("block-padded"), "{error}");
        assert!(error.contains("declares no logical shape"), "{error}");
    }

    #[test]
    fn read_refuses_a_tensor_set_that_changed_since_planning() {
        let dir = fixture_dir();
        let path = dir.path().join("drift.safetensors");
        write_safetensors(&path, &[("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0]))]);
        let plan = plan_logical_weights(&path, &StripModel).unwrap();
        write_safetensors(
            &path,
            &[
                ("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0])),
                ("model.extra", "BF16", &[1], bf16_bytes(&[3.0])),
            ],
        );
        let error = read_logical_weights(&path, &plan, LogicalReadMode::Deferred)
            .err()
            .expect("a changed tensor set must refuse");
        assert!(
            error
                .to_string()
                .contains("tensor set changed since planning"),
            "{error}"
        );
    }

    #[test]
    fn dense_codec_refuses_a_substituted_dtype() {
        // A plan that claims dense bf16 for a tensor the backend actually loads as f32 must not
        // pass through the codec — the codec guards the representation it reports.
        let dir = fixture_dir();
        let path = dir.path().join("subst.safetensors");
        write_safetensors(
            &path,
            &[("model.w", "F32", &[1], 1.0_f32.to_le_bytes().to_vec())],
        );
        let plan = LogicalWeightPlan {
            mapping_id: "strip-model-test",
            tensors: vec![LogicalTensorPlan {
                logical_key: "w".to_owned(),
                physical_key: "model.w".to_owned(),
                encoding: WeightEncoding::DenseBf16,
                shape: vec![1],
                source_bytes: 4,
                codec_id: DENSE_BF16_CODEC.codec_id,
                resident_encoding: WeightEncoding::DenseBf16,
                codec: TensorCodecSpec::Dense,
                residency: PlannedResidency {
                    mode: ResidencyMode::Dense,
                    resident_bytes: 2,
                },
                transform: None,
            }],
            companions: Vec::new(),
            source_bytes: 4,
        };
        let error = read_logical_weights(&path, &plan, LogicalReadMode::Deferred)
            .err()
            .expect("dtype substitution must refuse");
        assert!(
            error.to_string().contains("planned as dense-bf16"),
            "{error}"
        );
    }

    /// sc-20385 review: `measure_residency` used to add the plan's own `companion.resident_bytes`
    /// into the measured total, so `receipt == plan` was a copy of the plan on that row rather than
    /// a cross-check of it. This backend plans dense only and retains no companion, so the honest
    /// form is to assert the invariant: a plan claiming a retained companion refuses.
    #[test]
    fn a_plan_that_retains_a_companion_refuses_because_this_backend_measures_none() {
        let dir = fixture_dir();
        let path = dir.path().join("retained-companion.safetensors");
        write_safetensors(
            &path,
            &[
                ("model.w", "BF16", &[1], bf16_bytes(&[1.0])),
                (
                    "model.w.weight_scale",
                    "F32",
                    &[],
                    1.0_f32.to_le_bytes().to_vec(),
                ),
            ],
        );
        let plan = LogicalWeightPlan {
            mapping_id: "strip-model-test",
            tensors: vec![LogicalTensorPlan {
                logical_key: "w".to_owned(),
                physical_key: "model.w".to_owned(),
                encoding: WeightEncoding::DenseBf16,
                shape: vec![1],
                source_bytes: 2,
                codec_id: DENSE_BF16_CODEC.codec_id,
                resident_encoding: WeightEncoding::DenseBf16,
                codec: TensorCodecSpec::Dense,
                residency: PlannedResidency {
                    mode: ResidencyMode::Dense,
                    resident_bytes: 2,
                },
                transform: None,
            }],
            companions: vec![CompanionTensorPlan {
                physical_key: "model.w.weight_scale".to_owned(),
                role: CompanionRole::WeightScale,
                owner_physical_key: "model.w".to_owned(),
                source_bytes: 4,
                // A packed policy's answer — which this backend never produces and never measures.
                resident_bytes: 4,
            }],
            source_bytes: 6,
        };
        let mut materialize = |weights: &mut Weights| weights.materialize();
        let error = read_logical_weights(&path, &plan, LogicalReadMode::Eager(&mut materialize))
            .err()
            .expect("a retained companion must refuse, not be echoed back as measured");
        let error = error.to_string();
        assert!(
            error.contains("planned to retain 4 resident bytes")
                && error.contains("model.w.weight_scale"),
            "{error}"
        );

        // The same fixture with the companion consumed reads fine, so the refusal is about the
        // retained bytes and not about the companion's presence.
        let mut consumed = plan.clone();
        consumed.companions[0].resident_bytes = 0;
        let mut materialize = |weights: &mut Weights| weights.materialize();
        let weights =
            read_logical_weights(&path, &consumed, LogicalReadMode::Eager(&mut materialize))
                .expect("a consumed companion reads");
        assert_eq!(
            weights.receipt.resident_bytes(),
            consumed.resident_bytes(),
            "receipt and plan agree once nothing is retained"
        );
    }

    /// sc-20634 review: planning and reading are two separate opens, so a same-key same-dtype
    /// tensor whose SHAPE changed in between must refuse instead of loading silently. The plan's
    /// dtype still matches here, so only the shape check can catch it.
    #[test]
    fn dense_codec_refuses_a_substituted_shape() {
        let dir = fixture_dir();
        let path = dir.path().join("reshaped.safetensors");
        // On disk: 2 bf16 elements. Planned: the same key, the same dense-bf16 encoding, shape [1].
        write_safetensors(
            &path,
            &[("model.w", "BF16", &[2], vec![0x00, 0x3f, 0x00, 0x40])],
        );
        let plan = LogicalWeightPlan {
            mapping_id: "strip-model-test",
            tensors: vec![LogicalTensorPlan {
                logical_key: "w".to_owned(),
                physical_key: "model.w".to_owned(),
                encoding: WeightEncoding::DenseBf16,
                shape: vec![1],
                source_bytes: 2,
                codec_id: DENSE_BF16_CODEC.codec_id,
                resident_encoding: WeightEncoding::DenseBf16,
                codec: TensorCodecSpec::Dense,
                residency: PlannedResidency {
                    mode: ResidencyMode::Dense,
                    resident_bytes: 2,
                },
                transform: None,
            }],
            companions: Vec::new(),
            source_bytes: 2,
        };
        let error = read_logical_weights(&path, &plan, LogicalReadMode::Deferred)
            .err()
            .expect("a shape substitution must refuse");
        let error = error.to_string();
        assert!(
            error.contains("planned with stored shape [1]")
                && error.contains("loaded shape [2]")
                && error.contains("\"model.w\""),
            "{error}"
        );

        // The matching shape is accepted, so the refusal is about the substitution, not the fixture.
        let mut matching = plan.clone();
        matching.tensors[0].shape = vec![2];
        matching.tensors[0].source_bytes = 4;
        matching.tensors[0].residency.resident_bytes = 4;
        matching.source_bytes = 4;
        read_logical_weights(&path, &matching, LogicalReadMode::Deferred)
            .expect("the planned shape loads");
    }

    /// MLX's own converters agree with the gen-core spec decoders on every one of the 256 codes —
    /// independent implementations of the same tables, cross-checked bit for bit (NaN by class).
    #[test]
    fn mlx_device_decoders_agree_with_the_gen_core_references_for_all_256_codes() {
        let codes: Vec<u8> = (0..=255).collect();
        let array = Array::from_slice(&codes, &[256]);
        for (what, decoded, reference) in [
            (
                "e4m3",
                super::e4m3_bytes_to_f32(&array).unwrap(),
                gen_core::fp8_e4m3fn_to_f32 as fn(u8) -> f32,
            ),
            (
                "e5m2",
                super::e5m2_bytes_to_f32(&array).unwrap(),
                gen_core::fp8_e5m2_to_f32,
            ),
            (
                "e8m0",
                super::e8m0_bytes_to_f32(&array).unwrap(),
                gen_core::e8m0_to_f32,
            ),
        ] {
            let decoded = decoded.as_slice::<f32>();
            for (code, got) in codes.iter().zip(decoded) {
                let expected = reference(*code);
                assert!(
                    (got.is_nan() && expected.is_nan()) || got.to_bits() == expected.to_bits(),
                    "{what} code {code:#04x}: mlx {got}, reference {expected}"
                );
            }
        }
    }
}
