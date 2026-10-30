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
//! <https://github.com/mrbizarro/minimax-h3-mlx/blob/604306cab8694584e6096b32a1c1731bee22c43e/reference/diffusers/convert_minimax_h3_to_diffusers.py>
//! Runtime state_dict exports must explicitly declare `__metadata__.h3_qkv_layout=contiguous`;
//! raw checkpoints default to the reference's interleaved source layout.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use gen_core::checkpoint_codec::{CheckpointCodecRegistry, WeightEncoding, DENSE_CODECS};
use gen_core::convrot_int8::{ConvRotInt8, MAX_DESCRIPTOR_BYTES};
use gen_core::w4a8_int8::W4A8Int8;
use gen_core::weightsmeta::Dtype;
use gen_core_h3::pruned_adaln::{PrunedAdaLn, PruningIdentity, TABLE_KEY, TOPOLOGY_ID};
use serde::{Deserialize, Serialize};
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
pub fn read_h3_descriptor(path: &Path) -> Result<Value, String> {
    let limit = crate::checkpoint_inspector::MAX_JSON_DESCRIPTOR_BYTES;
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    read_h3_descriptor_from_reader(file, len, limit, path)
}

fn read_h3_descriptor_from_reader(
    reader: impl Read,
    len: u64,
    limit: u64,
    path: &Path,
) -> Result<Value, String> {
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
    reader
        .take(limit + 1)
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

    /// Validate the complete architecture document without allocating its tensor inventory.
    pub fn validate(&self) -> Result<(), String> {
        if self.0.get("_class_name").and_then(Value::as_str) != Some("MiniMaxH3Transformer3DModel")
        {
            return Err("config.json must declare MiniMaxH3Transformer3DModel".into());
        }
        if self.0.get("quantization").is_some() || self.0.get("quantization_config").is_some() {
            return Err("MiniMax-H3 import requires an architecture config, not a stock packed-tier quantization policy; INT8 execution is declared by per-layer ConvRot operands".into());
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

    /// Complete inventory for the centered-curve AdaLN topology (sc-26684). The trunk, heads,
    /// norms, biases and refiner keep the full geometry; the direct `U*S` table replaces the
    /// four timestep-MLP tensors, and every AdaLN modulation projection narrows its input columns
    /// from `time_embed_dim` to `rank`. `grid`/`rank` come from the source table, never a filename.
    pub fn pruned_shapes(
        &self,
        grid: usize,
        rank: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, String> {
        if grid < 2
            || rank == 0
            || grid > i32::MAX as usize
            || rank > i32::MAX as usize
            || grid.checked_mul(rank).is_none()
        {
            return Err(format!(
                "MiniMax-H3 pruned AdaLN requires an F32 table with [grid >= 2, rank > 0], got [{grid}, {rank}]"
            ));
        }
        let mut names = self.shapes()?;
        for leaf in ["linear_1", "linear_2"] {
            for suffix in ["weight", "bias"] {
                names.remove(&format!("time_embedder.{leaf}.{suffix}"));
            }
        }
        for (name, shape) in names.iter_mut() {
            let reduced_input = name == "norm_out.linear.weight"
                || (name.starts_with("transformer_blocks.")
                    && name.ends_with(".adaln_proj.linear.weight"));
            if reduced_input {
                shape[1] = rank;
            }
        }
        names.insert(TABLE_KEY.to_owned(), vec![grid, rank]);
        Ok(names)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3QkvLayout {
    Interleaved,
    Contiguous,
}

/// Exact consumer binding between an H3 source's reduced centered-curve AdaLN topology and the
/// shared native generators (sc-26684). `identity` is the full 64-hex engine content hash over
/// every canonical operand value in contract order, at the ORIGINAL stored dtype, produced by
/// `gen_core_h3::pruned_adaln::PrunedAdaLn::identity_builder` over the same normalized bytes both
/// native caches consume. `sourceBytes` counts stored operand bytes at their dtype width and
/// `computeBytes` counts the same operands once widened to F32; they are deliberately separate
/// quantities, never interchangeable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct H3PruningBinding {
    pub topology_id: String,
    /// Full 64-hex engine identity hash of the complete reduced operand set.
    pub identity: String,
    pub grid: usize,
    pub rank: usize,
    pub source_bytes: u64,
    pub compute_bytes: u64,
}

impl H3PruningBinding {
    pub fn to_json(&self) -> Value {
        json!({
            "topologyId": &self.topology_id,
            "identity": &self.identity,
            "grid": self.grid,
            "rank": self.rank,
            "sourceBytes": self.source_bytes,
            "computeBytes": self.compute_bytes,
        })
    }

    pub fn from_value(value: &Value) -> Result<Self, String> {
        let binding: Self = serde_json::from_value(value.clone())
            .map_err(|error| format!("H3 AdaLN pruning binding: {error}"))?;
        binding.validate()?;
        Ok(binding)
    }

    /// The shared topology identity the native generators consume. A forged or partial binding
    /// still parses here; it can never pass a provider comparison against the actual source,
    /// whose identity is recomputed from operand bytes.
    pub fn engine_identity(&self) -> Result<PruningIdentity, String> {
        self.validate()?;
        let mut identity = [0u8; 32];
        let digits = self.identity.as_bytes();
        for (index, byte) in identity.iter_mut().enumerate() {
            let high = (digits[index * 2] as char).to_digit(16);
            let low = (digits[index * 2 + 1] as char).to_digit(16);
            *byte = match (high, low) {
                (Some(high), Some(low)) => (high * 16 + low) as u8,
                _ => return Err("H3 AdaLN pruning binding: identity is not 64 hex digits".into()),
            };
        }
        Ok(PruningIdentity(identity))
    }

    fn validate(&self) -> Result<(), String> {
        if self.topology_id != TOPOLOGY_ID {
            return Err(format!(
                "H3 AdaLN pruning binding: topologyId {:?} is not the current shared topology {TOPOLOGY_ID:?}",
                self.topology_id
            ));
        }
        if self.grid < 2
            || self.rank == 0
            || self.grid > i32::MAX as usize
            || self.rank > i32::MAX as usize
        {
            return Err(format!(
                "H3 AdaLN pruning binding: requires [grid >= 2, rank > 0], got [{}, {}]",
                self.grid, self.rank
            ));
        }
        let table = self
            .grid
            .checked_mul(self.rank)
            .ok_or("H3 AdaLN pruning binding: table geometry overflows")?;
        // The table itself is always F32; widened F32 operands are never narrower than stored.
        let table_bytes = (table as u64)
            .checked_mul(4)
            .ok_or("H3 AdaLN pruning binding: table byte count overflows")?;
        if self.source_bytes < table_bytes {
            return Err(format!(
                "H3 AdaLN pruning binding: sourceBytes {} cannot carry the F32 table ({table_bytes} bytes)",
                self.source_bytes
            ));
        }
        if self.compute_bytes < table_bytes {
            return Err(format!(
                "H3 AdaLN pruning binding: computeBytes {} cannot carry the F32 table ({table_bytes} bytes)",
                self.compute_bytes
            ));
        }
        if !self.compute_bytes.is_multiple_of(4) {
            return Err(format!(
                "H3 AdaLN pruning binding: computeBytes {} is not whole F32 elements",
                self.compute_bytes
            ));
        }
        // Every admitted operand is F16, BF16 or F32, so widening to F32 never shrinks the payload.
        if self.compute_bytes < self.source_bytes {
            return Err(format!(
                "H3 AdaLN pruning binding: computeBytes {} is narrower than sourceBytes {}",
                self.compute_bytes, self.source_bytes
            ));
        }
        if !self.source_bytes.is_multiple_of(2) {
            return Err(format!(
                "H3 AdaLN pruning binding: sourceBytes {} is not a whole operand byte count",
                self.source_bytes
            ));
        }
        if self.identity.len() != 64 || !self.identity.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(
                "H3 AdaLN pruning binding: identity must be the full 64-hex engine hash".into(),
            );
        }
        Ok(())
    }
}

/// The complete engine identity hex string, lowercase, one byte at a time.
fn pruning_identity_hex(identity: PruningIdentity) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut hex = String::with_capacity(64);
    for byte in identity.0 {
        hex.push(DIGITS[(byte >> 4) as usize] as char);
        hex.push(DIGITS[(byte & 15) as usize] as char);
    }
    hex
}

/// An iterable F32 view over one stored canonical operand. Reads stay inside one fixed chunk; the
/// values are decoded losslessly from the ORIGINAL stored dtype exactly as the normalization copy
/// schedule emits them (the schedule copies bytes, so canonical order equals source order).
struct OperandValues<'a> {
    reader: std::io::BufReader<File>,
    ranges: &'a [(u64, u64)],
    next_range: usize,
    base: u64,
    dtype: Dtype,
    remaining: u64,
    chunk: Vec<u8>,
    values: Vec<f32>,
    cursor: usize,
    error: Option<String>,
    check: &'a dyn Fn() -> std::io::Result<()>,
}

const OPERAND_CHUNK: usize = 64 * 1024;

impl<'a> OperandValues<'a> {
    fn open(
        source: &Path,
        tensor: &'a Tensor,
        dtype: Dtype,
        check: &'a dyn Fn() -> std::io::Result<()>,
    ) -> Result<Self, String> {
        check().map_err(|error| error.to_string())?;
        let reader = std::io::BufReader::with_capacity(
            OPERAND_CHUNK,
            File::open(source)
                .map_err(|error| format!("H3 AdaLN: {}: {error}", source.display()))?,
        );
        Ok(Self {
            reader,
            ranges: &tensor.ranges,
            next_range: 0,
            base: tensor.offset,
            dtype,
            remaining: 0,
            chunk: vec![0u8; OPERAND_CHUNK],
            values: Vec::new(),
            cursor: 0,
            error: None,
            check,
        })
    }

    /// Decode the next bounded chunk of the current operand range. False ends the operand.
    fn fill(&mut self) -> bool {
        self.values.clear();
        self.cursor = 0;
        while self.values.is_empty() {
            if let Err(error) = (self.check)() {
                self.error = Some(error.to_string());
                return false;
            }
            if self.remaining == 0 {
                let Some(&(start, length)) = self.ranges.get(self.next_range) else {
                    return false;
                };
                self.next_range += 1;
                if self
                    .reader
                    .seek(SeekFrom::Start(self.base + start))
                    .is_err()
                {
                    self.error = Some("H3 AdaLN: cannot seek the reduced operand payload".into());
                    return false;
                }
                self.remaining = length;
                if length == 0 {
                    continue;
                }
            }
            let width = self.dtype.size();
            let take = self.remaining.min(self.chunk.len() as u64) as usize / width * width;
            if take == 0 {
                self.error = Some(format!(
                    "H3 AdaLN: reduced operand bytes are not aligned to {width}-byte {} values",
                    self.dtype.as_str()
                ));
                return false;
            }
            if let Err(error) = self.reader.read_exact(&mut self.chunk[..take]) {
                self.error = Some(format!(
                    "H3 AdaLN: truncated reduced operand payload: {error}"
                ));
                return false;
            }
            if let Err(error) = (self.check)() {
                self.error = Some(error.to_string());
                return false;
            }
            self.remaining -= take as u64;
            let chunk = &self.chunk[..take];
            match self.dtype {
                Dtype::F32 => self.values.extend(
                    chunk
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|bytes| f32::from_le_bytes(*bytes)),
                ),
                Dtype::BF16 => self
                    .values
                    .extend(chunk.as_chunks::<2>().0.iter().map(|bytes| {
                        let bits = (u16::from_le_bytes(*bytes) as u32) << 16;
                        f32::from_bits(bits)
                    })),
                Dtype::F16 => self
                    .values
                    .extend(chunk.as_chunks::<2>().0.iter().map(|bytes| {
                        let bits = u16::from_le_bytes(*bytes);
                        half::f16::from_bits(bits).to_f32()
                    })),
                _ => unreachable!("inspect admits only dense F16/BF16/F32 AdaLN operands"),
            }
        }
        true
    }
}

impl Iterator for OperandValues<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.error.is_some() {
            return None;
        }
        if self.cursor == self.values.len() && !self.fill() {
            return None;
        }
        let value = self.values[self.cursor];
        self.cursor += 1;
        Some(value)
    }
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

/// Verified copy schedule; pruning values are streamed through bounded buffers, never retained.
#[derive(Debug, Clone)]
pub struct H3SourceInventory {
    pub config: H3Config,
    pub native: bool,
    /// Files and namespace-wrapped exports need a canonical component directory too, even when
    /// their tensor leaves already use Diffusers spelling.
    pub requires_normalization: bool,
    /// Descriptor-validated native symmetric INT8, never the affine MLX Q8 dialect.
    pub convrot_int8: bool,
    /// Packed codebook W4A8, possibly mixed with ConvRot INT8 projections.
    pub w4a8_int8: bool,
    /// Content identity of the reduced centered-curve AdaLN topology this source actually carries.
    /// Present only when the source stores the direct table instead of the timestep MLP.
    pub pruning: Option<H3PruningBinding>,
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
        Self::from_resolved_with_config_and_check(resolved, bound_config, &|| Ok(()))
    }

    pub fn from_resolved_with_config_and_check(
        resolved: &ResolvedCheckpointV1,
        bound_config: Option<H3Config>,
        check: &dyn Fn() -> std::io::Result<()>,
    ) -> Result<Self, String> {
        Self::from_files_with_config_and_check(
            &resolved
                .layers
                .iter()
                .map(|l| (l.layer.target_path.clone(), l.path.clone()))
                .collect::<Vec<_>>(),
            bound_config,
            check,
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
        Self::from_files_with_config_and_check(files, bound_config, &|| Ok(()))
    }

    pub fn from_files_with_config_and_check(
        files: &[(String, PathBuf)],
        bound_config: Option<H3Config>,
        check: &dyn Fn() -> std::io::Result<()>,
    ) -> Result<Self, String> {
        check().map_err(|error| error.to_string())?;
        let mut weights = Vec::new();
        let mut config = None;
        let mut index = None;
        let mut directory = None;
        for (name, path) in files {
            check().map_err(|error| error.to_string())?;
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
            check().map_err(|error| error.to_string())?;
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
        if native && weights.len() != 1 {
            return Err("native H3 import expects one safetensors file; export sharded sources in canonical Diffusers layout".into());
        }
        if weights.len() > 1 && index.is_none() {
            return Err(
                "sharded H3 source requires diffusion_pytorch_model.safetensors.index.json".into(),
            );
        }
        let config = config.or(bound_config).ok_or("MiniMax-H3 source has no config.json or embedded h3_config; explicitly select a compatible base to bind its exact configuration before import")?;
        let codecs = CheckpointCodecRegistry::new(DENSE_CODECS.iter().copied())
            .map_err(|e| e.to_string())?;
        let mut tensors = BTreeMap::<String, Tensor>::new();
        let mut physical = H3Physical::new();
        for (source, header) in headers.iter().enumerate() {
            for (key, entry) in header.as_object().ok_or("invalid H3 header")? {
                if key == "__metadata__" {
                    continue;
                }
                if let Some((previous, _)) = physical.insert(key, (source, entry)) {
                    return Err(format!(
                        "duplicate H3 tensor {key:?} in {:?} and {:?}",
                        weights[previous].0, weights[source].0
                    ));
                }
            }
        }
        // The table itself decides whether this is a reduced topology; shapes are validated
        // against that choice and never guessed from a filename or a metadata claim.
        let pruned = pruning_geometry(&physical)?;
        let mut expected = match pruned {
            Some((grid, rank)) => config.pruned_shapes(grid, rank)?,
            None => config.shapes()?,
        };
        let quantized = convrot_outputs(
            &headers, &weights, &physical, native, &config, qkv_layout, &expected,
        )?;
        for outputs in quantized.values() {
            for (logical, shape, _) in outputs {
                // The packed projection's physical shape is [out, in/2]; the codec already
                // checked its logical geometry against the architecture before scheduling it.
                expected.insert(logical.clone(), shape.clone());
            }
        }
        for (source, ((_, path), header)) in weights.iter().zip(&headers).enumerate() {
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
                    "BF16" => (Some(WeightEncoding::DenseBf16),2u64),
                    "F16" => (Some(WeightEncoding::DenseF16),2u64),
                    "F32" => (Some(WeightEncoding::DenseF32),4u64),
                    "I8" | "U8" | "F8_E4M3" if quantized.contains_key(key) => (None,1u64),
                    _ => return Err(format!("unsupported H3 tensor {key:?} encoding {dtype:?}; expected dense or descriptor-validated ConvRot INT8/W4A8 operands")),
                };
                if encoding.is_some_and(|encoding| codecs.for_encoding(encoding).is_none()) {
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
                let mapped = if let Some(outputs) = quantized.get(key) {
                    outputs.clone()
                } else if native {
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
                let (source, _) = physical.get(key.as_str()).ok_or_else(|| {
                    format!("H3 index tensor {key:?} missing from shard {shard:?}")
                })?;
                let actual = Path::new(weights[*source].0)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .ok_or("invalid shard path")?;
                if actual != shard {
                    return Err(format!(
                        "H3 index tensor {key:?} names {shard:?}, actually in {actual:?}"
                    ));
                }
            }
            for (key, (source, _)) in &physical {
                if !map.contains_key(*key) {
                    return Err(format!(
                        "unconsumed H3 shard tensor {key:?} in {:?}: absent from weight_map",
                        weights[*source].0
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
        // The complete spine proved above; only now is the reduced topology hashed from its
        // actual stored bytes, so a refused or partial topology never yields a binding.
        let sources: Vec<_> = weights.iter().map(|(_, path)| (*path).clone()).collect();
        let pruning = match pruned {
            Some((grid, rank)) => Some(pruning_binding(
                &tensors, &config, &sources, grid, rank, check,
            )?),
            None => None,
        };
        check().map_err(|error| error.to_string())?;
        Ok(Self {
            config,
            native,
            requires_normalization: requires_normalization || pruning.is_some(),
            convrot_int8: tensors.keys().any(|key| key.ends_with(".weight_scale")),
            w4a8_int8: tensors.keys().any(|key| key.ends_with(".weight_s_rel")),
            pruning,
            sources,
            tensors,
        })
    }

    pub fn source_encoding(&self) -> Option<&'static str> {
        if self.w4a8_int8 {
            Some("asym_w4a8_int8")
        } else if self.convrot_int8 {
            Some("convrot_int8")
        } else {
            None
        }
    }

    pub fn derivative_request(&self) -> CheckpointDerivativeRequestV1 {
        let mut outputs = vec![H3_CONFIG_FILE.into(), H3_INDEX_FILE.into()];
        outputs.extend((0..self.sources.len()).map(|i| self.shard_name(i)));
        CheckpointDerivativeRequestV1 {
            kind: CheckpointDerivativeKindV1::NormalizedLayout,
            adapter_id: H3_MAPPING_ID.into(),
            adapter_version: "1".into(),
            codec_id: if self.w4a8_int8 {
                crate::checkpoint_weight_facts::W4A8_INT8_CODEC_ID
            } else if self.convrot_int8 {
                crate::checkpoint_weight_facts::CONVROT_INT8_CODEC_ID
            } else {
                "dense-f16-bf16-f32"
            }
            .into(),
            codec_version: "2".into(),
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
        self.produce_with_check(outputs, &|| Ok(()))
    }

    pub fn produce_with_check(
        &self,
        outputs: &CheckpointDerivativeOutputs<'_>,
        check: &dyn Fn() -> std::io::Result<()>,
    ) -> Result<(), String> {
        check().map_err(|error| error.to_string())?;
        let mut index = Map::new();
        let mut total = 0u64;
        let total_payload_bytes = self
            .tensors
            .values()
            .flat_map(|tensor| tensor.ranges.iter())
            .try_fold(0u64, |sum, &(_, bytes)| sum.checked_add(bytes))
            .ok_or("H3 normalization payload size overflow")?;
        let mut copied_payload_bytes = 0u64;
        let copy_checkpoint = |copied_payload_bytes| {
            check().map_err(|error| {
                tracing::info!(
                    event = "h3_normalization_copy_progress",
                    state = if error.kind() == std::io::ErrorKind::Interrupted {
                        "Canceled"
                    } else {
                        "Failed"
                    },
                    copied_payload_bytes,
                    total_payload_bytes,
                    "Normalization stopped before derivative publication"
                );
                error.to_string()
            })
        };
        let mut buffer = vec![0u8; 1024 * 1024];
        for (i, path) in self.sources.iter().enumerate() {
            check().map_err(|error| error.to_string())?;
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
                        copy_checkpoint(copied_payload_bytes)?;
                        let take = remaining.min(buffer.len() as u64) as usize;
                        input
                            .read_exact(&mut buffer[..take])
                            .map_err(|e| e.to_string())?;
                        copy_checkpoint(copied_payload_bytes)?;
                        out.write_all(&buffer[..take]).map_err(|e| e.to_string())?;
                        copied_payload_bytes += take as u64;
                        if copied_payload_bytes == take as u64
                            || copied_payload_bytes % (256 * 1024 * 1024) < take as u64
                        {
                            tracing::info!(event = "h3_normalization_copy_progress", state = "Writing",
                                copied_payload_bytes, total_payload_bytes,
                                "Actual bounded normalization payload copy; derivative is not yet published");
                        }
                        copy_checkpoint(copied_payload_bytes)?;
                        remaining -= take as u64;
                    }
                }
            }
            total += offset;
        }
        check().map_err(|error| error.to_string())?;
        serde_json::to_writer(
            outputs.create(H3_CONFIG_FILE).map_err(|e| e.to_string())?,
            &self.config.0,
        )
        .map_err(|e| e.to_string())?;
        serde_json::to_writer(
            outputs.create(H3_INDEX_FILE).map_err(|e| e.to_string())?,
            &json!({"metadata":{"total_size":total},"weight_map":index}),
        )
        .map_err(|e| e.to_string())?;
        check().map_err(|error| error.to_string())
    }
}

fn strip_prefix(key: &str) -> &str {
    ["model.diffusion_model.", "diffusion_model."]
        .into_iter()
        .find_map(|prefix| key.strip_prefix(prefix))
        .unwrap_or(key)
}

type Output = (String, Vec<usize>, Vec<(u64, u64)>);

type H3Physical<'a> = BTreeMap<&'a str, (usize, &'a Value)>;

/// Timestep-MLP operands the direct centered-curve table replaces, in their canonical Diffusers
/// spelling and the accepted native export spelling.
const TIME_MLP_NAMES: [(&str, &str); 4] = [
    (
        "time_embedder.linear_1.weight",
        "time_embedder.proj_in.weight",
    ),
    ("time_embedder.linear_1.bias", "time_embedder.proj_in.bias"),
    (
        "time_embedder.linear_2.weight",
        "time_embedder.proj_out.weight",
    ),
    ("time_embedder.linear_2.bias", "time_embedder.proj_out.bias"),
];

/// Read the direct table's own geometry from headers before any operand is mapped. A present but
/// malformed table refuses; only an absent table selects the full timestep-MLP inventory.
fn pruning_geometry(physical: &H3Physical<'_>) -> Result<Option<(usize, usize)>, String> {
    let mut geometry = None;
    for (key, (_, entry)) in physical {
        if strip_prefix(key) != TABLE_KEY {
            continue;
        }
        if entry.get("dtype").and_then(Value::as_str) != Some("F32") {
            return Err(format!(
                "H3 AdaLN: {TABLE_KEY} must be stored F32, got {:?}",
                entry.get("dtype")
            ));
        }
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("H3 AdaLN: {TABLE_KEY} lacks a shape"))?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
            })
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| format!("H3 AdaLN: {TABLE_KEY} has an invalid shape"))?;
        if shape.len() != 2 || shape[0] < 2 || shape[1] == 0 {
            return Err(format!(
                "H3 AdaLN: {TABLE_KEY} must be [grid >= 2, rank > 0], got {shape:?}"
            ));
        }
        geometry = Some((shape[0], shape[1]));
    }
    if geometry.is_some() {
        for (canonical, native_spelling) in TIME_MLP_NAMES {
            if physical.keys().any(|key| {
                let key = strip_prefix(key);
                key == canonical || key == native_spelling
            }) {
                return Err(format!(
                    "H3 AdaLN: {TABLE_KEY} conflicts with full timestep MLP {canonical}"
                ));
            }
        }
    }
    Ok(geometry)
}

/// Hash the complete reduced topology from the authoritative copy schedule. Every operand is
/// decoded losslessly from its stored dtype in contract order; a missing operand, an inconsistent
/// stored dtype or a nonfinite value refuses the inventory before any execution or publication.
fn pruning_binding(
    tensors: &BTreeMap<String, Tensor>,
    config: &H3Config,
    sources: &[PathBuf],
    grid: usize,
    rank: usize,
    check: &dyn Fn() -> std::io::Result<()>,
) -> Result<H3PruningBinding, String> {
    let topology = PrunedAdaLn::inspect(
        config.dim("hidden_size")?,
        config.dim("num_layers")?,
        config.dim("time_embed_dim")?,
        |name| {
            tensors
                .get(name)
                .and_then(|tensor| Dtype::parse(&tensor.dtype).map(|d| (d, tensor.shape.clone())))
        },
    )?
    .ok_or_else(|| {
        format!("H3 AdaLN: {TABLE_KEY} is present but the reduced AdaLN topology is incomplete")
    })?;
    if (topology.grid, topology.rank) != (grid, rank) {
        return Err(format!(
            "H3 AdaLN: table geometry [{}, {}] disagrees with the inspected [{grid}, {rank}]",
            topology.grid, topology.rank
        ));
    }
    let mut builder = topology.identity_builder();
    let mut source_bytes = 0u64;
    let mut compute_bytes = 0u64;
    for operand in topology.operands() {
        let tensor = tensors
            .get(&operand.name)
            .ok_or_else(|| format!("H3 AdaLN: missing reduced operand {}", operand.name))?;
        if Dtype::parse(&tensor.dtype) != Some(operand.dtype) {
            return Err(format!(
                "H3 AdaLN: {} stored as {}, but the inspected topology declares {:?}",
                operand.name, tensor.dtype, operand.dtype
            ));
        }
        let elements = operand
            .shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or("H3 AdaLN: operand element count overflows")?;
        let source = sources
            .get(tensor.source)
            .ok_or("H3 AdaLN: operand source is outside the declared files")?;
        let mut values = OperandValues::open(source, tensor, operand.dtype, check)?;
        let hashed = builder.operand_values(&operand.name, &mut values);
        if let Some(error) = values.error.take() {
            return Err(error);
        }
        hashed?;
        let stored = (elements as u64)
            .checked_mul(operand.dtype.size() as u64)
            .ok_or("H3 AdaLN: stored operand bytes overflow")?;
        source_bytes = source_bytes
            .checked_add(stored)
            .ok_or("H3 AdaLN: stored operand bytes overflow")?;
        compute_bytes = compute_bytes
            .checked_add(
                (elements as u64)
                    .checked_mul(4)
                    .ok_or("H3 AdaLN: widened operand bytes overflow")?,
            )
            .ok_or("H3 AdaLN: widened operand bytes overflow")?;
    }
    Ok(H3PruningBinding {
        topology_id: TOPOLOGY_ID.to_owned(),
        identity: pruning_identity_hex(builder.finish()?),
        grid,
        rank,
        source_bytes,
        compute_bytes,
    })
}

/// Validate executable operands before scheduling copies. Codes are never decoded or allocated;
/// descriptor reads are bounded and row scales are scanned one value at a time.
fn convrot_outputs(
    headers: &[Value],
    files: &[(&String, &PathBuf)],
    physical: &H3Physical<'_>,
    native: bool,
    config: &H3Config,
    layout: H3QkvLayout,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<BTreeMap<String, Vec<Output>>, String> {
    let mut outputs = BTreeMap::new();
    for (source, header) in headers.iter().enumerate() {
        if header
            .get("__metadata__")
            .and_then(Value::as_object)
            .is_some_and(|metadata| {
                [
                    "_quantization_metadata",
                    "quant_format",
                    "quantization_config",
                ]
                .iter()
                .any(|key| metadata.contains_key(*key))
            })
        {
            return Err("H3 requires authoritative per-layer ConvRot descriptors; competing file-level quantization declarations are unsupported".into());
        }
        if !header
            .as_object()
            .ok_or("invalid H3 header")?
            .keys()
            .any(|key| key.ends_with(".comfy_quant"))
        {
            continue;
        }
        let mut file = File::open(files[source].1).map_err(|e| e.to_string())?;
        let mut len = [0; 8];
        file.read_exact(&mut len).map_err(|e| e.to_string())?;
        let body = 8 + u64::from_le_bytes(len);
        let shape = |entry: &Value| -> Result<Vec<usize>, String> {
            entry["shape"]
                .as_array()
                .ok_or("ConvRot operand lacks shape")?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| "invalid ConvRot shape".into())
                })
                .collect()
        };
        let offset = |entry: &Value| -> Result<u64, String> {
            entry["data_offsets"][0]
                .as_u64()
                .ok_or_else(|| "ConvRot operand lacks offset".into())
        };
        let mut reader = std::io::BufReader::with_capacity(64 * 1024, file);
        let mut descriptor_bytes = [0u8; MAX_DESCRIPTOR_BYTES];
        for (key, entry) in header.as_object().ok_or("invalid H3 header")? {
            let Some(base) = key.strip_suffix(".comfy_quant") else {
                continue;
            };
            let descriptor_shape = shape(entry)?;
            if entry["dtype"] != "U8"
                || descriptor_shape.len() != 1
                || descriptor_shape[0] == 0
                || descriptor_shape[0] > MAX_DESCRIPTOR_BYTES
            {
                return Err(format!("H3 {key}: requires bounded U8 ConvRot descriptor"));
            }
            reader
                .seek(SeekFrom::Start(body + offset(entry)?))
                .map_err(|e| e.to_string())?;
            let payload = &mut descriptor_bytes[..descriptor_shape[0]];
            reader.read_exact(payload).map_err(|e| e.to_string())?;
            let declaration: Value = serde_json::from_slice(payload)
                .map_err(|e| format!("H3 {key}: invalid projection descriptor: {e}"))?;
            match declaration.get("format").and_then(Value::as_str) {
                Some("asym_w4a8_int8") => {
                    w4a8_outputs(key, payload, physical, files, native, config, layout, expected, &mut outputs)?;
                    continue;
                }
                Some("int8_tensorwise") => {}
                other => return Err(format!("H3 {key}: unsupported projection format {other:?}; supported native formats are int8_tensorwise ConvRot and asym_w4a8_int8; export a supported checkpoint")),
            }
            let weight_key = format!("{base}.weight");
            let scale_key = format!("{base}.weight_scale");
            let (_, weight) = physical
                .get(weight_key.as_str())
                .ok_or_else(|| format!("H3 {key}: missing weight"))?;
            let (scale_source, scale) = physical
                .get(scale_key.as_str())
                .ok_or_else(|| format!("H3 {key}: missing weight_scale"))?;
            if weight["dtype"] != "I8" || scale["dtype"] != "F32" {
                return Err(format!(
                    "H3 {key}: ConvRot requires signed I8 weight and F32 row scales"
                ));
            }
            let weight_shape = shape(weight)?;
            let scale_shape = shape(scale)?;
            let descriptor = ConvRotInt8::parse(payload, &weight_shape, &scale_shape)
                .map_err(|e| format!("H3 {key}: {e}"))?;
            let mut foreign_reader;
            let (scale_reader, scale_body) = if *scale_source == source {
                (&mut reader, body)
            } else {
                let mut file = File::open(files[*scale_source].1).map_err(|e| e.to_string())?;
                let mut len = [0; 8];
                file.read_exact(&mut len).map_err(|e| e.to_string())?;
                foreign_reader = std::io::BufReader::with_capacity(64 * 1024, file);
                (&mut foreign_reader, 8 + u64::from_le_bytes(len))
            };
            scale_reader
                .seek(SeekFrom::Start(scale_body + offset(scale)?))
                .map_err(|e| e.to_string())?;
            for _ in 0..descriptor.out_features {
                let mut bytes = [0; 4];
                scale_reader
                    .read_exact(&mut bytes)
                    .map_err(|e| e.to_string())?;
                let value = f32::from_le_bytes(bytes);
                if !value.is_finite() || value <= 0.0 {
                    return Err(format!(
                        "H3 {scale_key}: requires finite positive F32 scales"
                    ));
                }
            }
            let mapped = if native {
                native_outputs(strip_prefix(&weight_key), config, layout, &weight_shape, 1)?
            } else {
                vec![(
                    strip_prefix(&weight_key).into(),
                    weight_shape.clone(),
                    vec![(0, (descriptor.out_features * descriptor.in_features) as u64)],
                )]
            };
            let mut mapped_scales = Vec::new();
            let mut mapped_descriptors = Vec::new();
            for (logical, dimensions, ranges) in &mapped {
                let supported = (logical.starts_with("transformer_blocks.")
                    || logical.starts_with("token_refiner.refiner_blocks."))
                    && (logical.contains(".attn.")
                        || logical.contains(".ff.")
                        || logical.ends_with(".adaln_proj.linear.weight"));
                if !supported || expected.get(logical) != Some(dimensions) {
                    return Err(format!(
                        "H3 {key}: quantized projection {logical:?} is not a native INT8 consumer"
                    ));
                }
                let logical_base = logical
                    .strip_suffix(".weight")
                    .ok_or("invalid ConvRot logical weight")?;
                let input = descriptor.in_features as u64;
                // Only output rows move: input rotation groups and descriptor bytes remain unchanged.
                let scale_ranges = ranges
                    .iter()
                    .map(|&(start, len)| (start / input * 4, len / input * 4))
                    .collect();
                mapped_scales.push((
                    format!("{logical_base}.weight_scale"),
                    vec![dimensions[0], 1],
                    scale_ranges,
                ));
                mapped_descriptors.push((
                    format!("{logical_base}.comfy_quant"),
                    descriptor_shape.clone(),
                    vec![(0, payload.len() as u64)],
                ));
            }
            outputs.insert(weight_key, mapped);
            outputs.insert(scale_key, mapped_scales);
            outputs.insert(key.clone(), mapped_descriptors);
        }
    }
    Ok(outputs)
}

/// Validate the shared native W4A8 contract, then schedule only row copies. Logical rows are
/// mapped first (QKV and SwiGLU); each companion applies that exact permutation at its own stored
/// row width. No unpacked int8, float projection, or model-sized scratch is ever constructed.
#[allow(clippy::too_many_arguments)]
fn w4a8_outputs(
    key: &str,
    payload: &[u8],
    physical: &H3Physical<'_>,
    files: &[(&String, &PathBuf)],
    native: bool,
    config: &H3Config,
    layout: H3QkvLayout,
    expected: &BTreeMap<String, Vec<usize>>,
    outputs: &mut BTreeMap<String, Vec<Output>>,
) -> Result<(), String> {
    let base = key
        .strip_suffix(".comfy_quant")
        .ok_or("invalid W4A8 descriptor key")?;
    let operand =
        |suffix: &str, dtype: &str| -> Result<(String, usize, Vec<usize>, &Value), String> {
            let name = format!("{base}.{suffix}");
            let (source, entry) = physical
                .get(name.as_str())
                .ok_or_else(|| format!("H3 {key}: missing W4A8 {suffix}"))?;
            if entry["dtype"].as_str() != Some(dtype) {
                return Err(format!(
                    "H3 {name}: W4A8 requires stored {dtype}, got {}",
                    entry["dtype"]
                ));
            }
            let shape = entry["shape"]
                .as_array()
                .and_then(|shape| {
                    shape
                        .iter()
                        .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| format!("H3 {name}: invalid W4A8 shape"))?;
            Ok((name, *source, shape, *entry))
        };
    let weight = operand("weight", "I8")?;
    let relative = operand("weight_s_rel", "F8_E4M3")?;
    let channel = operand("weight_s_channel", "F32")?;
    let codebook = operand("weight_codebook", "F32")?;
    let descriptor = W4A8Int8::parse(payload, &weight.2, &relative.2, &channel.2, &codebook.2)
        .map_err(|e| format!("H3 {key}: {e}"))?;
    // Shared value validators run on bounded chunks. Each chunk is a complete set of groups or
    // channels; geometry has already been validated above against the original stored operands.
    for (suffix, item, width) in [
        ("weight_s_rel", &relative, 1usize),
        ("weight_s_channel", &channel, 4),
        ("codebook", &codebook, 4),
    ] {
        let mut file = File::open(files[item.1].1).map_err(|e| e.to_string())?;
        let mut length = [0; 8];
        file.read_exact(&mut length).map_err(|e| e.to_string())?;
        let start = item.3["data_offsets"][0]
            .as_u64()
            .ok_or("W4A8 operand lacks offset")?;
        file.seek(SeekFrom::Start(8 + u64::from_le_bytes(length) + start))
            .map_err(|e| e.to_string())?;
        let mut remaining = item
            .2
            .iter()
            .try_fold(width, |n, &d| n.checked_mul(d))
            .ok_or("W4A8 operand byte count overflow")?;
        let mut buffer = [0u8; 64 * 1024];
        while remaining != 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count])
                .map_err(|e| e.to_string())?;
            let validated = if suffix == "weight_s_rel" {
                // count is a multiple of 16: each rotation group carries sixteen fp8 scales.
                W4A8Int8 {
                    out_features: count / 16,
                    in_features: 256,
                    ..descriptor
                }
                .validate_s_rel(&buffer[..count])
            } else {
                let values: Vec<_> = buffer[..count]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                if suffix == "codebook" {
                    descriptor.validate_codebook(&values)
                } else {
                    W4A8Int8 {
                        out_features: values.len(),
                        ..descriptor
                    }
                    .validate_s_channel(&values)
                }
            };
            validated.map_err(|e| format!("H3 {}: {e}", item.0))?;
            remaining -= count;
        }
    }
    let logical_shape = vec![descriptor.out_features, descriptor.in_features];
    let mapped = if native {
        native_outputs(strip_prefix(&weight.0), config, layout, &logical_shape, 1)?
    } else {
        vec![(
            strip_prefix(&weight.0).into(),
            logical_shape.clone(),
            vec![(0, (descriptor.out_features * descriptor.in_features) as u64)],
        )]
    };
    let mut packed = Vec::new();
    let mut relative_outputs = Vec::new();
    let mut channel_outputs = Vec::new();
    let mut codebook_outputs = Vec::new();
    let mut descriptors = Vec::new();
    for (logical, dimensions, ranges) in mapped {
        let supported = (logical.starts_with("transformer_blocks.")
            || logical.starts_with("token_refiner.refiner_blocks."))
            && (logical.contains(".attn.to_") || logical.contains(".ff."));
        if !supported || expected.get(&logical) != Some(&dimensions) {
            return Err(format!("H3 {key}: W4A8 projection {logical:?} is not a native packed consumer with geometry {dimensions:?}"));
        }
        let logical_base = logical
            .strip_suffix(".weight")
            .ok_or("invalid W4A8 logical weight")?;
        let rows = |row_width: u64| {
            ranges
                .iter()
                .map(|&(start, len)| {
                    (
                        start / descriptor.in_features as u64 * row_width,
                        len / descriptor.in_features as u64 * row_width,
                    )
                })
                .collect()
        };
        relative_outputs.push((
            format!("{logical_base}.weight_s_rel"),
            vec![dimensions[0], descriptor.in_features / 16],
            rows((descriptor.in_features / 16) as u64),
        ));
        channel_outputs.push((
            format!("{logical_base}.weight_s_channel"),
            vec![dimensions[0]],
            rows(4),
        ));
        codebook_outputs.push((
            format!("{logical_base}.weight_codebook"),
            vec![16],
            vec![(0, 64)],
        ));
        descriptors.push((
            format!("{logical_base}.comfy_quant"),
            vec![payload.len()],
            vec![(0, payload.len() as u64)],
        ));
        packed.push((
            logical,
            vec![dimensions[0], descriptor.in_features / 2],
            rows((descriptor.in_features / 2) as u64),
        ));
    }
    outputs.insert(weight.0, packed);
    outputs.insert(relative.0, relative_outputs);
    outputs.insert(channel.0, channel_outputs);
    outputs.insert(codebook.0, codebook_outputs);
    outputs.insert(key.into(), descriptors);
    Ok(())
}

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
