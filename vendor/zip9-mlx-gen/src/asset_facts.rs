//! Header-only projection of safetensors storage into the bytes MLX actually keeps resident.
//!
//! Providers own the set of tensors they quantize or materialize. This module owns the arithmetic,
//! packed-triple detection, recursive source traversal, and checked failure semantics so those facts
//! cannot drift between model families.

use std::collections::HashSet;
use std::path::Path;

use gen_core::weightsmeta::{safetensors_path_tensor_headers, SafetensorsTensorHeader};
use gen_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentProjection {
    /// Retain the tensor's existing safetensors payload.
    Stored,
    /// Pack an eligible rank-two `.weight` into MLX affine codes plus bf16 scales and biases.
    GroupQuantized { bits: i32, group_size: usize },
    /// Materialize every element as bf16 (used by Krea's plain-int8 native-file loader).
    Bfloat16,
    /// Materialize every element as f32, whatever the stored width.
    ///
    /// This is the projection for a component whose loader **upcasts unconditionally**, which is
    /// not a hypothetical: `mlx_gen_sdxl::load_vae` does `cast_all(Float32)` on every load because
    /// the SDXL VAE is fp16-unstable, and every SDXL-family tier ships the fp16 variant. Priced
    /// from stored bytes such a component is underpriced by exactly 2x at every tier (sc-15839).
    Float32,
    /// The loader consumes this source-only tensor without retaining it.
    Omit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentTensorBytes {
    pub name: String,
    pub resident_bytes: u64,
}

/// Describe the exact tensors a provider's block loader streams. Classification belongs to
/// that loader; this helper only groups already projected bytes and preserves the remainder.
pub fn streamed_weight_facts(
    tensors: &[ResidentTensorBytes],
    classify: impl Fn(&str) -> Option<(String, u32)>,
) -> Result<Option<gen_core::StreamedWeightFacts>> {
    use std::collections::BTreeMap;
    let mut resident_bytes = 0_u64;
    let mut stacks: BTreeMap<String, BTreeMap<u32, u64>> = BTreeMap::new();
    for tensor in tensors {
        let bytes = match classify(&tensor.name) {
            Some((stack, index)) => stacks.entry(stack).or_default().entry(index).or_default(),
            None => &mut resident_bytes,
        };
        *bytes = bytes
            .checked_add(tensor.resident_bytes)
            .ok_or_else(|| Error::Msg("streamed component byte count overflow".to_owned()))?;
    }
    if stacks.is_empty() {
        return Ok(None);
    }
    let stacks = stacks
        .into_iter()
        .map(|(name, blocks)| {
            if !blocks.keys().copied().eq(0..blocks.len() as u32) {
                return Err(Error::Msg(format!(
                    "streamed stack {name:?} has non-contiguous block indices"
                )));
            }
            Ok(blocks.into_values().collect())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(gen_core::StreamedWeightFacts {
        resident_bytes,
        stacks,
    }))
}

/// Split an exact dotted block-stack prefix (`layers.` or `transformer_blocks.`). A provider
/// that has nested stacks passes the full prefix selected by its loader, including the parent.
pub fn indexed_block_key(key: &str, prefix: &str) -> Option<(String, u32)> {
    let (index, _) = key.strip_prefix(prefix)?.split_once('.')?;
    Some((prefix.to_owned(), index.parse().ok()?))
}

/// Channel geometry for a provider that executes the shared layer-wise convolution decoder.
/// The provider supplies its actual upsampling schedule and compute dtype; neither is inferred
/// from the file's precision or from the denoiser. Missing headers leave the profile unknown.
pub fn layerwise_decoder_workspace(
    tensors: &[SafetensorsTensorHeader],
    block_prefix: &str,
    spatial_divisors: &[u32],
    activation_dtype_width: u32,
) -> Option<gen_core::DecoderWorkspaceFacts> {
    let channels = spatial_divisors
        .iter()
        .enumerate()
        .map(|(index, divisor)| {
            if *divisor == 0 {
                return None;
            }
            let key = format!("{block_prefix}{index}.resnets.0.conv1.weight");
            let tensor = tensors.iter().find(|tensor| tensor.name == key)?;
            if tensor.shape.len() != 4 {
                return None;
            }
            u32::try_from(tensor.shape[0])
                .ok()
                .filter(|channels| *channels > 0)
        })
        .collect::<Option<Vec<_>>>()?;
    if channels.is_empty() || activation_dtype_width == 0 {
        return None;
    }
    let input_channels = spatial_divisors
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let key = format!("{block_prefix}{index}.resnets.0.conv1.weight");
            let tensor = tensors.iter().find(|tensor| tensor.name == key)?;
            u32::try_from(tensor.shape[1])
                .ok()
                .filter(|channels| *channels > 0)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(gen_core::DecoderWorkspaceFacts {
        tiling: gen_core::DecoderTilingRealization::LayerwiseConvolution,
        activation_dtype_width,
        channels,
        input_channels,
        spatial_divisors: spatial_divisors.to_vec(),
    })
}

pub fn projected_safetensors_tensors(
    path: impl AsRef<Path>,
    projection: impl Fn(&SafetensorsTensorHeader) -> ResidentProjection,
) -> Result<Vec<ResidentTensorBytes>> {
    let tensors = safetensors_path_tensor_headers(path)?;
    projected_tensor_headers(&tensors, projection)
}

/// Project an already validated tensor-header inventory into the bytes MLX will retain.
///
/// Selected text encoders use this entry point with
/// [`gen_core::ValidatedEncoderSource::tensor_headers`] so a recursive path walk cannot price
/// nested files that neither MLX nor Candle loads as direct shards.
pub fn projected_tensor_headers(
    tensors: &[SafetensorsTensorHeader],
    projection: impl Fn(&SafetensorsTensorHeader) -> ResidentProjection,
) -> Result<Vec<ResidentTensorBytes>> {
    if tensors.is_empty() {
        return Err(Error::Msg(
            "safetensors source contains no tensor headers".to_owned(),
        ));
    }
    let packed_bases = tensors
        .iter()
        .filter_map(|tensor| tensor.name.strip_suffix(".scales").map(str::to_owned))
        .collect::<HashSet<_>>();

    tensors
        .iter()
        .map(|tensor| {
            let base = tensor.name.strip_suffix(".weight");
            let policy = if base.is_some_and(|base| packed_bases.contains(base)) {
                ResidentProjection::Stored
            } else {
                projection(tensor)
            };
            let resident_bytes = match policy {
                ResidentProjection::Stored => tensor.data_bytes,
                ResidentProjection::Omit => 0,
                ResidentProjection::Bfloat16 | ResidentProjection::Float32 => {
                    let width = if policy == ResidentProjection::Float32 {
                        4
                    } else {
                        2
                    };
                    tensor
                        .shape
                        .iter()
                        .try_fold(1_u64, |total, dimension| {
                            let dimension = u64::try_from(*dimension).map_err(|_| {
                                Error::Msg(format!(
                                    "tensor {:?} has an unrepresentable dimension",
                                    tensor.name
                                ))
                            })?;
                            total.checked_mul(dimension).ok_or_else(|| {
                                Error::Msg(format!(
                                    "tensor {:?} element count overflow",
                                    tensor.name
                                ))
                            })
                        })?
                        .checked_mul(width)
                        .ok_or_else(|| {
                            Error::Msg(format!(
                                "tensor {:?} {}-byte materialization size overflow",
                                tensor.name, width
                            ))
                        })?
                }
                ResidentProjection::GroupQuantized { bits, group_size } => {
                    let [out, input] = tensor.shape.as_slice() else {
                        return Ok(ResidentTensorBytes {
                            name: tensor.name.clone(),
                            resident_bytes: tensor.data_bytes,
                        });
                    };
                    if base.is_none()
                        || !matches!(bits, 4 | 8)
                        || group_size == 0
                        || *input < group_size
                        || *input % group_size != 0
                    {
                        tensor.data_bytes
                    } else {
                        let out = u64::try_from(*out).map_err(|_| {
                            Error::Msg(format!("tensor {:?} output overflow", tensor.name))
                        })?;
                        let input = u64::try_from(*input).map_err(|_| {
                            Error::Msg(format!("tensor {:?} input overflow", tensor.name))
                        })?;
                        let codes = out
                            .checked_mul(input)
                            .and_then(|elements| elements.checked_mul(bits as u64))
                            .map(|packed_bits| packed_bits / 8)
                            .ok_or_else(|| {
                                Error::Msg(format!(
                                    "tensor {:?} quantized code size overflow",
                                    tensor.name
                                ))
                            })?;
                        let tables = out
                            .checked_mul(input / group_size as u64)
                            .and_then(|entries| entries.checked_mul(4))
                            .ok_or_else(|| {
                                Error::Msg(format!(
                                    "tensor {:?} quantization table size overflow",
                                    tensor.name
                                ))
                            })?;
                        codes.checked_add(tables).ok_or_else(|| {
                            Error::Msg(format!("tensor {:?} resident size overflow", tensor.name))
                        })?
                    }
                }
            };
            Ok(ResidentTensorBytes {
                name: tensor.name.clone(),
                resident_bytes,
            })
        })
        .collect()
}

pub fn projected_safetensors_bytes(
    path: impl AsRef<Path>,
    projection: impl Fn(&SafetensorsTensorHeader) -> ResidentProjection,
) -> Result<u64> {
    projected_safetensors_tensors(path, projection)?
        .into_iter()
        .try_fold(0_u64, |total, tensor| {
            total.checked_add(tensor.resident_bytes).ok_or_else(|| {
                Error::Msg(format!(
                    "resident byte sum overflow at tensor {:?}",
                    tensor.name
                ))
            })
        })
}

pub fn projected_tensor_headers_bytes(
    tensors: &[SafetensorsTensorHeader],
    projection: impl Fn(&SafetensorsTensorHeader) -> ResidentProjection,
) -> Result<u64> {
    projected_tensor_headers(tensors, projection)?
        .into_iter()
        .try_fold(0_u64, |total, tensor| {
            total.checked_add(tensor.resident_bytes).ok_or_else(|| {
                Error::Msg(format!(
                    "resident byte sum overflow at tensor {:?}",
                    tensor.name
                ))
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_inventory_keeps_fixed_tensors_and_rejects_incomplete_stacks() {
        let tensor = |name: &str, resident_bytes| ResidentTensorBytes {
            name: name.into(),
            resident_bytes,
        };
        let tensors = vec![
            tensor("projection.weight", 100),
            tensor("layers.0.weight", 20),
            tensor("layers.0.scales", 2),
            tensor("layers.1.weight", 50),
        ];
        let facts = streamed_weight_facts(&tensors, |key| indexed_block_key(key, "layers."))
            .unwrap()
            .unwrap();
        assert_eq!(facts.resident_bytes, 100);
        assert_eq!(facts.stacks, vec![vec![22, 50]]);
        assert_eq!(facts.peak_bytes(Some(1)), 150);
        assert_eq!(facts.peak_bytes(None), 172);
        assert!(
            streamed_weight_facts(&[tensor("layers.1.weight", 50)], |key| indexed_block_key(
                key, "layers."
            ))
            .is_err()
        );
        assert!(streamed_weight_facts(&tensors, |_| None).unwrap().is_none());
    }

    #[test]
    fn decoder_profile_preserves_input_width_and_rejects_missing_stages() {
        let headers = [(512, 512), (512, 512), (256, 512), (128, 256)]
            .into_iter()
            .enumerate()
            .map(|(i, (out, input))| SafetensorsTensorHeader {
                name: format!("decoder.up_blocks.{i}.resnets.0.conv1.weight"),
                dtype: gen_core::weightsmeta::Dtype::F16,
                shape: vec![out, input, 3, 3],
                data_bytes: (out * input * 9 * 2) as u64,
            })
            .collect::<Vec<_>>();
        let facts =
            layerwise_decoder_workspace(&headers, "decoder.up_blocks.", &[8, 4, 2, 1], 4).unwrap();
        assert_eq!(facts.channels, [512, 512, 256, 128]);
        assert_eq!(facts.input_channels, [512, 512, 512, 256]);
        assert_eq!(facts.activation_dtype_width, 4);
        assert!(
            layerwise_decoder_workspace(&headers[..3], "decoder.up_blocks.", &[8, 4, 2, 1], 4)
                .is_none()
        );
        assert!(
            layerwise_decoder_workspace(&headers, "decoder.up_blocks.", &[8, 4, 2, 0], 4).is_none()
        );
    }

    fn write_file(path: &Path, entries: &[(&str, &str, &[usize], usize)]) {
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in entries {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn projection_is_recursive_checked_and_preserves_existing_packs() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        write_file(
            &nested.join("dense.safetensors"),
            &[
                ("dense.weight", "BF16", &[2, 64], 256),
                ("leaf.weight", "BF16", &[2, 33], 132),
            ],
        );
        std::fs::write(root.join("config.json"), b"not weights").unwrap();
        let projected =
            projected_safetensors_bytes(&root, |_| ResidentProjection::GroupQuantized {
                bits: 4,
                group_size: 64,
            })
            .unwrap();
        assert_eq!(projected, 72 + 132);

        write_file(
            &root.join("packed.safetensors"),
            &[
                ("packed.weight", "U32", &[2, 8], 64),
                ("packed.scales", "BF16", &[2, 1], 4),
                ("packed.biases", "BF16", &[2, 1], 4),
            ],
        );
        let with_pack =
            projected_safetensors_bytes(&root, |_| ResidentProjection::GroupQuantized {
                bits: 4,
                group_size: 64,
            })
            .unwrap();
        assert_eq!(with_pack, projected + 72);

        let corrupt = root.join("nested/corrupt.safetensors");
        std::fs::write(&corrupt, b"not a safetensors file").unwrap();
        assert!(projected_safetensors_bytes(&root, |_| ResidentProjection::Stored).is_err());
        assert!(projected_safetensors_bytes(&corrupt, |_| ResidentProjection::Stored).is_err());

        let missing_file = root.join("missing.safetensors");
        assert!(
            projected_safetensors_bytes(&missing_file, |_| ResidentProjection::Stored).is_err()
        );
        let empty_file = root.join("empty.safetensors");
        std::fs::write(&empty_file, []).unwrap();
        assert!(projected_safetensors_bytes(&empty_file, |_| ResidentProjection::Stored).is_err());

        let missing_dir = root.join("missing-component");
        assert!(projected_safetensors_bytes(&missing_dir, |_| ResidentProjection::Stored).is_err());
        let empty_dir = root.join("empty-component");
        std::fs::create_dir_all(&empty_dir).unwrap();
        assert!(projected_safetensors_bytes(&empty_dir, |_| ResidentProjection::Stored).is_err());
        let corrupt_dir = root.join("corrupt-component");
        std::fs::create_dir_all(&corrupt_dir).unwrap();
        std::fs::write(corrupt_dir.join("model.safetensors"), b"corrupt").unwrap();
        assert!(projected_safetensors_bytes(&corrupt_dir, |_| ResidentProjection::Stored).is_err());
    }
}
