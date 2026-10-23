//! Task-neutral H3 tensor adapter and bounded source normalization (sc-26681).
//!
//! Canonical names/shapes follow inference `81cff07c13137fea520cec0b1dc9b8f083514601`,
//! `mlx-gen-minimax-h3/src/dit/{heads,layers,block,refiner}.rs`. Native raw QKV is
//! **per-head interleaved**; reference `state_dict()` QKV is already reordered. They must not
//! be confused (`src/layout.rs` rule 2 and `src/dit/qkv.rs`). Native SwiGLU is gate-first;
//! canonical SwiGLU is value-first. Normalization copies byte ranges, never casts precision,
//! never materializes a dense tensor/model, and publishes through the shared derivative cache.
//!
//! Executable raw-to-Diffusers reference pinned at
//! https://github.com/mrbizarro/minimax-h3-mlx/blob/604306cab8694584e6096b32a1c1731bee22c43e/reference/diffusers/convert_minimax_h3_to_diffusers.py
//! Runtime state_dict exports must explicitly declare `__metadata__.h3_qkv_layout=contiguous`;
//! raw checkpoints default to the reference's interleaved source layout.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use gen_core::checkpoint_codec::{CheckpointCodecRegistry, WeightEncoding, DENSE_CODECS};
use serde_json::{json, Map, Value};

use crate::checkpoint_derivative::{
    CheckpointDerivativeKindV1, CheckpointDerivativeOutputs, CheckpointDerivativeRequestV1,
};
use crate::checkpoint_plan_store::ResolvedCheckpointV1;
use crate::lora_family::read_safetensors_header;

pub const H3_MAPPING_ID: &str = "minimax-h3-native-v1";
pub const H3_CONFIG_FILE: &str = "config.json";
pub const H3_INDEX_FILE: &str = "diffusion_pytorch_model.safetensors.index.json";

/// Apply the inspector's descriptor policy before allocating or deserializing untrusted JSON.
pub(crate) fn read_h3_descriptor(path: &Path) -> Result<Value, String> {
    let limit = crate::checkpoint_inspector::MAX_JSON_DESCRIPTOR_BYTES;
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    let oversized = || {
        format!(
            "H3 descriptor {} exceeds {limit} byte descriptor limit",
            path.display()
        )
    };
    if len > limit {
        return Err(oversized());
    }
    let mut bytes = Vec::with_capacity(len as usize);
    // Also reject growth between metadata and read without consuming the whole file.
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(oversized());
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The full architecture document providers read. Nothing here declares FL2VA/Ref2VA semantics.
#[derive(Debug, Clone)]
pub struct H3Config(pub Value);

impl H3Config {
    /// The pinned published architecture for fixtures and explicit catalog declarations.
    /// Production sources require their own sidecar/embedded config or an explicitly selected
    /// base configuration; tensor shapes and filenames never select this value implicitly.
    pub fn published() -> Self {
        Self(json!({
            "_class_name": "MiniMaxH3Transformer3DModel",
            "num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 5376,
            "num_layers": 50, "num_refiner_layers": 2, "ffn_dim": 14336,
            "in_channels": 24, "audio_in_channels": 32, "patch_size": [1,2,2],
            "text_dim": 5120, "freq_dim": 256, "time_embed_hidden_dim": 5376,
            "time_embed_dim": 2688, "rope_freq_dim": 16, "rope_theta": 10000.0,
            "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5
        }))
    }

    fn dim(&self, name: &str) -> Result<usize, String> {
        self.0
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0 && v <= i32::MAX as usize)
            .ok_or_else(|| format!("MiniMax-H3 config requires positive integer {name:?}"))
    }

    fn validate(&self) -> Result<(), String> {
        if self.0.get("_class_name").and_then(Value::as_str) != Some("MiniMaxH3Transformer3DModel")
        {
            return Err("config.json must declare MiniMaxH3Transformer3DModel".into());
        }
        if self.0.get("quantization").is_some() || self.0.get("quantization_config").is_some() {
            return Err("MiniMax-H3 dense import does not decode quantized/pruned/ConvRot/W4A8 configs; re-export F16/BF16/F32".into());
        }
        for field in [
            "num_attention_heads",
            "attention_head_dim",
            "hidden_size",
            "num_layers",
            "num_refiner_layers",
            "ffn_dim",
            "in_channels",
            "audio_in_channels",
            "text_dim",
            "freq_dim",
            "time_embed_hidden_dim",
            "time_embed_dim",
            "rope_freq_dim",
        ] {
            self.dim(field)?;
        }
        if self.dim("num_layers")? > 1024 || self.dim("num_refiner_layers")? > 1024 {
            return Err("MiniMax-H3 layer counts exceed supported inventory bounds (1024)".into());
        }
        for field in ["rope_theta", "norm_eps", "qk_norm_eps", "final_norm_eps"] {
            if !self
                .0
                .get(field)
                .and_then(Value::as_f64)
                .is_some_and(|v| v.is_finite() && v > 0.0)
            {
                return Err(format!(
                    "MiniMax-H3 config requires finite positive {field:?}"
                ));
            }
        }
        self.patch_volume()?;
        if 6 * self.dim("rope_freq_dim")? > self.dim("attention_head_dim")? {
            return Err("MiniMax-H3 RoPE rotary dimension exceeds attention_head_dim".into());
        }
        Ok(())
    }

    fn patch_volume(&self) -> Result<usize, String> {
        let patch = self
            .0
            .get("patch_size")
            .and_then(Value::as_array)
            .filter(|v| v.len() == 3)
            .ok_or("MiniMax-H3 patch_size must be three positive integers")?;
        patch
            .iter()
            .try_fold(1usize, |n, v| {
                let d = v
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|&v| v > 0)
                    .ok_or("MiniMax-H3 patch_size must be three positive integers")?;
                n.checked_mul(d).ok_or("MiniMax-H3 patch_size overflows")
            })
            .map_err(str::to_owned)
    }

    /// Complete provider-consumed inventory, including heads, norm gains, biases, trunk and refiner.
    pub fn shapes(&self) -> Result<BTreeMap<String, Vec<usize>>, String> {
        self.validate()?;
        let h = self.dim("hidden_size")?;
        let q = self
            .dim("num_attention_heads")?
            .checked_mul(self.dim("attention_head_dim")?)
            .ok_or("H3 attention width overflow")?;
        let f = self.dim("ffn_dim")?;
        let t = self.dim("time_embed_dim")?;
        let mut names = BTreeMap::new();
        let mut add = |key: String, shape: Vec<usize>| {
            names.insert(key, shape);
        };
        for (container, count, modulated) in [
            ("transformer_blocks", self.dim("num_layers")?, true),
            (
                "token_refiner.refiner_blocks",
                self.dim("num_refiner_layers")?,
                false,
            ),
        ] {
            for i in 0..count {
                for leaf in ["norm1", "norm2"] {
                    add(format!("{container}.{i}.{leaf}.weight"), vec![h]);
                }
                for leaf in ["to_q", "to_k", "to_v"] {
                    add(format!("{container}.{i}.attn.{leaf}.weight"), vec![q, h]);
                }
                add(format!("{container}.{i}.attn.to_out.0.weight"), vec![h, q]);
                for leaf in ["norm_q", "norm_k"] {
                    add(
                        format!("{container}.{i}.attn.{leaf}.weight"),
                        vec![self.dim("attention_head_dim")?],
                    );
                }
                add(
                    format!("{container}.{i}.ff.net.0.proj.weight"),
                    vec![2 * f, h],
                );
                add(format!("{container}.{i}.ff.net.2.weight"), vec![h, f]);
                if modulated {
                    // Three modality groups × six modulation parameters, per pinned block.rs.
                    add(
                        format!("{container}.{i}.adaln_proj.linear.weight"),
                        vec![18 * h, t],
                    );
                    add(
                        format!("{container}.{i}.adaln_proj.linear.bias"),
                        vec![18 * h],
                    );
                }
            }
        }
        add("token_refiner.final_norm.weight".into(), vec![h]);
        for (base, out, input) in [
            (
                "proj_in",
                h,
                self.dim("in_channels")? * self.patch_volume()?,
            ),
            ("audio_proj_in", h, self.dim("audio_in_channels")?),
            ("context_embedder", h, self.dim("text_dim")?),
            (
                "time_embedder.linear_1",
                self.dim("time_embed_hidden_dim")?,
                self.dim("freq_dim")?,
            ),
            (
                "time_embedder.linear_2",
                t,
                self.dim("time_embed_hidden_dim")?,
            ),
            ("norm_out.linear", 2 * h, t),
            (
                "proj_out",
                self.dim("in_channels")? * self.patch_volume()?,
                h,
            ),
            ("audio_proj_out", self.dim("audio_in_channels")?, h),
        ] {
            add(format!("{base}.weight"), vec![out, input]);
            add(format!("{base}.bias"), vec![out]);
        }
        add("norm_out.norm.weight".into(), vec![h]);
        Ok(names)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3QkvLayout {
    Interleaved,
    Contiguous,
}

#[derive(Debug, Clone)]
struct Tensor {
    source: usize,
    key: String,
    dtype: String,
    shape: Vec<usize>,
    offset: u64,
    /// Ordered source byte ranges relative to the tensor payload. Empty only for zero-size data.
    ranges: Vec<(u64, u64)>,
}

/// Verified source inventory and exact copy schedule. Header-only memory; no tensor payloads.
#[derive(Debug, Clone)]
pub struct H3SourceInventory {
    pub config: H3Config,
    pub native: bool,
    /// Files and namespace-wrapped exports need a canonical component directory too, even when
    /// their tensor leaves already use Diffusers spelling.
    pub requires_normalization: bool,
    sources: Vec<PathBuf>,
    tensors: BTreeMap<String, Tensor>,
}

/// Resolve only the plan's declared files; callers must first re-verify/pin source identities.
impl H3SourceInventory {
    pub fn from_resolved(resolved: &ResolvedCheckpointV1) -> Result<Self, String> {
        Self::from_resolved_with_config(resolved, None)
    }

    /// Consume a plan with an explicitly bound fallback architecture document, when needed.
    /// Source-owned configuration wins; callers seal the fallback's persisted provenance.
    pub fn from_resolved_with_config(
        resolved: &ResolvedCheckpointV1,
        bound_config: Option<H3Config>,
    ) -> Result<Self, String> {
        Self::from_files_with_config(
            &resolved
                .layers
                .iter()
                .map(|l| (l.layer.target_path.clone(), l.path.clone()))
                .collect::<Vec<_>>(),
            bound_config,
        )
    }

    pub fn from_files(files: &[(String, PathBuf)]) -> Result<Self, String> {
        Self::from_files_with_config(files, None)
    }

    /// Source-owned sidecar/embedded documents are authoritative; the bound configuration is
    /// only a fallback. Complete JSON is preserved, including provider-relevant non-shape fields.
    pub fn from_files_with_config(
        files: &[(String, PathBuf)],
        bound_config: Option<H3Config>,
    ) -> Result<Self, String> {
        let mut weights = Vec::new();
        let mut config = None;
        let mut index = None;
        let mut directory = None;
        for (name, path) in files {
            let parent = Path::new(name).parent().unwrap_or(Path::new(""));
            if directory.is_some_and(|dir| dir != parent) {
                return Err(format!("unconsumed MiniMax-H3 artifact {name:?}: all DiT files must share one directory"));
            }
            directory = Some(parent);
            match Path::new(name).file_name().and_then(|s| s.to_str()) {
                Some(H3_CONFIG_FILE) => config = Some(H3Config(read_h3_descriptor(path)?)),
                Some(H3_INDEX_FILE) => index = Some(read_h3_descriptor(path)?),
                _ if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors")) => weights.push((name,path)),
                _ => return Err(format!("unconsumed MiniMax-H3 artifact {name:?}; expected DiT safetensors/config/index")),
            }
        }
        if weights.is_empty() {
            return Err("MiniMax-H3 checkpoint carries no DiT safetensors".into());
        }
        let mut headers = Vec::new();
        for (_, path) in &weights {
            if !crate::safetensors::file_is_structurally_valid(path) {
                return Err(format!("invalid MiniMax-H3 safetensors {}", path.display()));
            }
            headers.push(read_safetensors_header(path).map_err(|e| e.to_string())?);
        }
        let native = headers.iter().any(|header| {
            header.as_object().is_some_and(|h| {
                h.keys().any(|k| {
                    let k = strip_prefix(k);
                    k.starts_with("blocks.") || k.starts_with("token_refiner.blocks.")
                })
            })
        });
        let requires_normalization = native
            // Exact-pin providers enumerate only lowercase suffixes; normalize accepted variants.
            || weights.iter().any(|(_,path)| path.extension().is_none_or(|ext| ext != "safetensors"))
            || !files.iter().any(|(name, _)| {
                Path::new(name).file_name().and_then(|s| s.to_str()) == Some(H3_CONFIG_FILE)
            })
            || headers.iter().any(|h| {
                h.as_object()
                    .is_some_and(|h| h.keys().any(|k| strip_prefix(k) != k))
            });
        let mut qkv_layout = H3QkvLayout::Interleaved;
        for header in &headers {
            if let Some(meta) = header.get("__metadata__") {
                if let Some(embedded) = meta.get("h3_config").and_then(Value::as_str) {
                    if embedded.len() as u64
                        > crate::checkpoint_inspector::MAX_JSON_DESCRIPTOR_BYTES
                    {
                        return Err(
                            "H3 h3_config metadata exceeds 16777216 byte descriptor limit".into(),
                        );
                    }
                    let embedded = H3Config(
                        serde_json::from_str(embedded).map_err(|e| format!("h3_config: {e}"))?,
                    );
                    if config.as_ref().is_some_and(|c| c.0 != embedded.0) {
                        return Err(
                            "conflicting MiniMax-H3 config.json and h3_config metadata".into()
                        );
                    }
                    config = Some(embedded);
                }
                if let Some(layout) = meta.get("h3_qkv_layout").and_then(Value::as_str) {
                    qkv_layout = match layout {
                        "interleaved" => H3QkvLayout::Interleaved,
                        "contiguous" => H3QkvLayout::Contiguous,
                        _ => return Err(format!("unsupported H3 QKV layout {layout:?}; expected interleaved raw or contiguous reference state_dict")),
                    };
                }
            }
        }
        if !native && weights.len() > 1 && config.is_none() {
            return Err("sharded Diffusers H3 requires its explicit config.json".into());
        }
        if native && weights.len() != 1 {
            return Err("native H3 import expects one safetensors file; export sharded sources in canonical Diffusers layout".into());
        }
        if weights.len() > 1 && index.is_none() {
            return Err(
                "sharded H3 source requires diffusion_pytorch_model.safetensors.index.json".into(),
            );
        }
        let config = config.or(bound_config).ok_or("MiniMax-H3 source has no config.json or embedded h3_config; explicitly select a compatible base to bind its exact configuration before import")?;
        let expected = config.shapes()?;
        let codecs = CheckpointCodecRegistry::new(DENSE_CODECS.iter().copied())
            .map_err(|e| e.to_string())?;
        let mut tensors = BTreeMap::<String, Tensor>::new();
        let mut physical = BTreeMap::<String, String>::new();
        for (source, ((name, path), header)) in weights.iter().zip(&headers).enumerate() {
            let mut file = File::open(path).map_err(|e| e.to_string())?;
            let mut len = [0u8; 8];
            file.read_exact(&mut len).map_err(|e| e.to_string())?;
            let body = 8 + u64::from_le_bytes(len);
            for (key, entry) in header.as_object().ok_or("invalid H3 header")? {
                if key == "__metadata__" {
                    continue;
                }
                let dtype = entry
                    .get("dtype")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("tensor {key:?} lacks dtype"))?;
                let (encoding,width) = match dtype {
                    "BF16" => (WeightEncoding::DenseBf16,2u64),
                    "F16" => (WeightEncoding::DenseF16,2u64),
                    "F32" => (WeightEncoding::DenseF32,4u64),
                    _ => return Err(format!("unsupported H3 tensor {key:?} encoding {dtype:?}; dense F16/BF16/F32 only, not ConvRot/pruned/W4A8")),
                };
                if codecs.for_encoding(encoding).is_none() {
                    return Err(format!("unregistered H3 dense codec for {key:?}"));
                }
                let shape: Vec<usize> = entry
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or("tensor lacks shape")?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or("invalid tensor dimension")
                    })
                    .collect::<Result<_, _>>()?;
                let start = entry["data_offsets"][0]
                    .as_u64()
                    .ok_or("tensor lacks offset")?;
                let end = entry["data_offsets"][1]
                    .as_u64()
                    .ok_or("tensor lacks offset")?;
                if let Some(previous) = physical.insert(key.clone(), (*name).clone()) {
                    return Err(format!(
                        "duplicate H3 tensor {key:?} in {previous:?} and {name:?}"
                    ));
                }
                if native && strip_prefix(key) == "rope.inv_freq" {
                    // The pinned converter consumes this known non-parameter buffer: the provider
                    // regenerates it from config. Verify values, not just its familiar name.
                    let count = config.dim("rope_freq_dim")?;
                    if dtype != "F32" || shape != [count] {
                        return Err(format!(
                            "H3 rope.inv_freq requires F32 [{count}], got {dtype} {shape:?}"
                        ));
                    }
                    let theta = config.0["rope_theta"]
                        .as_f64()
                        .ok_or("H3 rope_theta missing")? as f32;
                    file.seek(SeekFrom::Start(body + start))
                        .map_err(|e| e.to_string())?;
                    for i in 0..count {
                        let mut bytes = [0u8; 4];
                        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
                        let actual = f32::from_le_bytes(bytes);
                        let expected = theta.powf(-(i as f32 / count as f32));
                        if !actual.is_finite() || (actual - expected).abs() > expected.abs() * 2e-6
                        {
                            return Err(format!("H3 rope.inv_freq[{i}] conflicts with bound rope_theta/rope_freq_dim: {actual}, expected {expected}"));
                        }
                    }
                    continue;
                }
                let mapped = if native {
                    native_outputs(strip_prefix(key), &config, qkv_layout, &shape, width)?
                } else {
                    vec![(
                        strip_prefix(key).to_owned(),
                        shape.clone(),
                        vec![(0, end - start)],
                    )]
                };
                for (logical, logical_shape, ranges) in mapped {
                    let required = expected.get(&logical).ok_or_else(|| {
                        format!("unconsumed H3 backbone tensor {key:?} maps to unknown {logical:?}")
                    })?;
                    if &logical_shape != required {
                        return Err(format!("H3 tensor {key:?} -> {logical:?}: shape {logical_shape:?}, expected {required:?}"));
                    }
                    if let Some(previous) = tensors.get(&logical) {
                        return Err(format!(
                            "conflicting H3 canonical tensor {logical:?}: sources {:?} and {key:?}",
                            previous.key
                        ));
                    }
                    tensors.insert(
                        logical,
                        Tensor {
                            source,
                            key: key.clone(),
                            dtype: dtype.into(),
                            shape: logical_shape,
                            offset: body + start,
                            ranges,
                        },
                    );
                }
            }
        }
        if let Some(index) = index {
            let map = index
                .get("weight_map")
                .and_then(Value::as_object)
                .filter(|m| !m.is_empty())
                .ok_or("H3 index requires nonempty weight_map")?;
            for (key, shard) in map {
                let shard = shard
                    .as_str()
                    .ok_or("H3 index shard names must be strings")?;
                let actual = physical.get(key).ok_or_else(|| {
                    format!("H3 index tensor {key:?} missing from shard {shard:?}")
                })?;
                let actual = Path::new(actual)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .ok_or("invalid shard path")?;
                if actual != shard {
                    return Err(format!(
                        "H3 index tensor {key:?} names {shard:?}, actually in {actual:?}"
                    ));
                }
            }
            for (key, shard) in &physical {
                if !map.contains_key(key) {
                    return Err(format!(
                        "unconsumed H3 shard tensor {key:?} in {shard:?}: absent from weight_map"
                    ));
                }
            }
        }
        let missing: Vec<_> = expected
            .keys()
            .filter(|k| !tensors.contains_key(*k))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "incomplete H3 backbone: missing {} tensors: {}",
                missing.len(),
                missing.join(", ")
            ));
        }
        Ok(Self {
            config,
            native,
            requires_normalization,
            sources: weights.into_iter().map(|(_, p)| p.clone()).collect(),
            tensors,
        })
    }

    pub fn derivative_request(&self) -> CheckpointDerivativeRequestV1 {
        let mut outputs = vec![H3_CONFIG_FILE.into(), H3_INDEX_FILE.into()];
        outputs.extend((0..self.sources.len()).map(|i| self.shard_name(i)));
        CheckpointDerivativeRequestV1 {
            kind: CheckpointDerivativeKindV1::NormalizedLayout,
            adapter_id: H3_MAPPING_ID.into(),
            adapter_version: "1".into(),
            codec_id: "dense-f16-bf16-f32".into(),
            codec_version: "1".into(),
            backend: "portable".into(),
            outputs,
        }
    }

    fn shard_name(&self, i: usize) -> String {
        format!(
            "diffusion_pytorch_model-{:05}-of-{:05}.safetensors",
            i + 1,
            self.sources.len()
        )
    }

    /// Emit directly from seekable sources through a fixed 1 MiB buffer. Payload allocations are
    /// independent of model/tensor size; only small tensor headers and the output index are resident.
    pub fn produce(&self, outputs: &CheckpointDerivativeOutputs<'_>) -> Result<(), String> {
        let mut index = Map::new();
        let mut total = 0u64;
        let mut buffer = vec![0u8; 1024 * 1024];
        for (i, path) in self.sources.iter().enumerate() {
            let shard_name = self.shard_name(i);
            let mut header = Map::new();
            let selected: Vec<_> = self.tensors.iter().filter(|(_, t)| t.source == i).collect();
            let mut offset = 0u64;
            for (name, tensor) in &selected {
                let len: u64 = tensor.ranges.iter().map(|(_, len)| len).sum();
                header.insert((*name).clone(),json!({"dtype":tensor.dtype,"shape":tensor.shape,"data_offsets":[offset,offset+len]}));
                index.insert((*name).clone(), json!(shard_name));
                offset += len;
            }
            let mut header = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
            header.resize(header.len().div_ceil(8) * 8, b' ');
            let mut out = outputs.create(&shard_name).map_err(|e| e.to_string())?;
            out.write_all(&(header.len() as u64).to_le_bytes())
                .and_then(|_| out.write_all(&header))
                .map_err(|e| e.to_string())?;
            let mut input = File::open(path).map_err(|e| e.to_string())?;
            for (_, tensor) in selected {
                for &(start, len) in &tensor.ranges {
                    input
                        .seek(SeekFrom::Start(tensor.offset + start))
                        .map_err(|e| e.to_string())?;
                    let mut remaining = len;
                    while remaining > 0 {
                        let take = remaining.min(buffer.len() as u64) as usize;
                        input
                            .read_exact(&mut buffer[..take])
                            .and_then(|_| out.write_all(&buffer[..take]))
                            .map_err(|e| e.to_string())?;
                        remaining -= take as u64;
                    }
                }
            }
            total += offset;
        }
        serde_json::to_writer(
            outputs.create(H3_CONFIG_FILE).map_err(|e| e.to_string())?,
            &self.config.0,
        )
        .map_err(|e| e.to_string())?;
        serde_json::to_writer(
            outputs.create(H3_INDEX_FILE).map_err(|e| e.to_string())?,
            &json!({"metadata":{"total_size":total},"weight_map":index}),
        )
        .map_err(|e| e.to_string())
    }
}

fn strip_prefix(key: &str) -> &str {
    ["model.diffusion_model.", "diffusion_model."]
        .into_iter()
        .find_map(|prefix| key.strip_prefix(prefix))
        .unwrap_or(key)
}

type Output = (String, Vec<usize>, Vec<(u64, u64)>);

fn native_outputs(
    key: &str,
    config: &H3Config,
    layout: H3QkvLayout,
    shape: &[usize],
    width: u64,
) -> Result<Vec<Output>, String> {
    let size = shape
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d as u64))
        .ok_or("H3 tensor size overflow")?;
    let normalized = if let Some(rest) = key.strip_prefix("token_refiner.blocks.") {
        format!("token_refiner.refiner_blocks.{rest}")
    } else if let Some(rest) = key.strip_prefix("blocks.") {
        format!("transformer_blocks.{rest}")
    } else {
        key.into()
    };
    if let Some(base) = normalized.strip_suffix(".attn.qkv_proj.weight") {
        let q = config.dim("num_attention_heads")? * config.dim("attention_head_dim")?;
        let hidden = config.dim("hidden_size")?;
        if shape != [3 * q, hidden] {
            return Err(format!(
                "H3 fused QKV {key:?} shape {shape:?}, expected [{}, {hidden}]",
                3 * q
            ));
        }
        let row = hidden as u64 * width;
        let head_dim = config.dim("attention_head_dim")?;
        let heads = config.dim("num_attention_heads")?;
        return Ok(["to_q", "to_k", "to_v"]
            .into_iter()
            .enumerate()
            .map(|(j, leaf)| {
                let ranges = match layout {
                    H3QkvLayout::Contiguous => vec![(j as u64 * q as u64 * row, q as u64 * row)],
                    H3QkvLayout::Interleaved => (0..heads)
                        .map(|h| (((h * 3 + j) * head_dim) as u64 * row, head_dim as u64 * row))
                        .collect(),
                };
                (
                    format!("{base}.attn.{leaf}.weight"),
                    vec![q, hidden],
                    ranges,
                )
            })
            .collect());
    }
    if let Some(base) = normalized.strip_suffix(".mlp.fc1.weight") {
        let f = config.dim("ffn_dim")?;
        if shape != [2 * f, config.dim("hidden_size")?] {
            return Err(format!(
                "H3 gated FFN {key:?} has shape {shape:?}, expected [{}, {}]",
                2 * f,
                config.dim("hidden_size")?
            ));
        }
        return Ok(vec![(
            format!("{base}.ff.net.0.proj.weight"),
            shape.to_vec(),
            vec![(size / 2, size / 2), (0, size / 2)],
        )]);
    }
    let mut logical = normalized
        .replace(".attn.out_proj.", ".attn.to_out.0.")
        .replace(".attn.q_norm.", ".attn.norm_q.")
        .replace(".attn.k_norm.", ".attn.norm_k.")
        .replace(".mlp.fc2.", ".ff.net.2.");
    for (raw, canonical) in [
        ("video_patch_proj.", "proj_in."),
        ("audio_patch_proj.", "audio_proj_in."),
        ("condition_proj.", "context_embedder."),
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.video_out.", "proj_out."),
        ("final_layer.audio_out.", "audio_proj_out."),
    ] {
        if let Some(rest) = logical.strip_prefix(raw) {
            logical = format!("{canonical}{rest}");
            break;
        }
    }
    Ok(vec![(logical, shape.to_vec(), vec![(0, size)])])
}

#[cfg(test)]
#[path = "minimax_h3_codec_tests.rs"]
mod tests;
