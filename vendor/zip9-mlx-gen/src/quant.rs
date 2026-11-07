//! Quantization — group-wise affine Q4/Q8, the mlx-rs equivalent of the Python mflux fork's
//! `nn.quantize(model, bits=bits)` path. mflux never passes `group_size`, so it uses MLX's
//! default of **64**. The actual quantization seam is
//! [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize), which quantizes each
//! `Linear` base in place **with the bf16-parity cast** the fork goldens require — providers route
//! through it (or their per-family loaders), so this module owns only the shared default below.
//!
//! Byte-level packing parity vs the fork (mlx 0.31) is checked in `tests/quant_parity.rs` —
//! the version-drift risk, since the crate links an older bundled MLX (mlx-rs 0.25).
//!
//! This module also owns the **group-size-parametric** packed-weight loaders ([`lin`] / [`embedding`])
//! and the **offline pre-quantization** converter primitives ([`load_dir_map`] / [`save_map`] /
//! [`quantize_map`]) that the provider crates used to copy verbatim (only their `GROUP_SIZE` differed:
//! 64 codebase-default, 32 for Boogu's `3360 = 32·105` DiT). Each provider keeps its own `GROUP_SIZE`
//! constant (and its target predicate) and passes them in (6935).

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Component, Path};

use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::adapters::AdaptableLinear;
use crate::nn::TokenEmbedding;
use crate::weights::Weights;
use crate::Result;

/// MLX's default quantization group size; mflux relies on it (never overrides). Used by the real
/// quantization seams ([`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) and
/// the Kolors ChatGLM3 quantizer).
pub const DEFAULT_GROUP_SIZE: i32 = 64;

/// Read the packed Q4/Q8 tier declared by `<root>/<component>/config.json`.
///
/// A missing config, or a valid config with no `quantization` member, denotes a dense component.
/// Every other read/JSON/marker failure is an error: a present but damaged marker must never be
/// mistaken for dense weights and silently load-time quantized. `component` is a provider-owned,
/// non-empty relative path; absolute paths and `..` are rejected.
pub fn packed_quant_bits(root: &Path, component: &str) -> Result<Option<i32>> {
    let component_path = Path::new(component);
    if component_path.as_os_str().is_empty()
        || component_path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(crate::Error::Msg(format!(
            "packed quant: component path must be a non-empty relative path without `..`, got \
             {component:?}"
        )));
    }

    packed_quant_bits_at(&root.join(component_path))
}

/// [`packed_quant_bits`] against a component directory that is already resolved — for a caller
/// holding the component path itself rather than a snapshot root plus a component name (the
/// offline pre-quantize converters, which are handed `…/transformer` directly).
pub fn packed_quant_bits_at(component_dir: &Path) -> Result<Option<i32>> {
    let config_path = component_dir.join("config.json");
    let bytes = match std::fs::read(&config_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(crate::Error::Msg(format!(
                "packed quant: read {}: {err}",
                config_path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
        crate::Error::Msg(format!(
            "packed quant: parse {}: {err}",
            config_path.display()
        ))
    })?;
    let Some(marker) = config.get("quantization") else {
        return Ok(None);
    };
    let marker = marker.as_object().ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization` must be an object",
            config_path.display()
        ))
    })?;
    let bits_i64 = marker
        .get("bits")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            crate::Error::Msg(format!(
                "packed quant: {} `quantization.bits` must be an integer",
                config_path.display()
            ))
        })?;
    let bits = i32::try_from(bits_i64).map_err(|_| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization.bits` is out of range: {bits_i64}",
            config_path.display()
        ))
    })?;
    if !matches!(bits, 4 | 8) {
        return Err(crate::Error::Msg(format!(
            "packed quant: {} declares unsupported `quantization.bits` {bits}; expected 4 or 8",
            config_path.display()
        )));
    }
    Ok(Some(bits))
}

/// Read the affine quantization group size declared by a packed component's `config.json`.
///
/// Dense components return `None`. A packed marker must declare one of the group sizes implemented
/// by MLX (`32`, `64`, or `128`); rejecting any other value here keeps corrupt or hand-edited
/// manifests from failing later inside `quantized_matmul` with an opaque backend error.
pub fn packed_quant_group_size_at(component_dir: &Path) -> Result<Option<i32>> {
    let config_path = component_dir.join("config.json");
    let bytes = match std::fs::read(&config_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(crate::Error::Msg(format!(
                "packed quant: read {}: {err}",
                config_path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
        crate::Error::Msg(format!(
            "packed quant: parse {}: {err}",
            config_path.display()
        ))
    })?;
    let Some(marker) = config.get("quantization") else {
        return Ok(None);
    };
    let marker = marker.as_object().ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization` must be an object",
            config_path.display()
        ))
    })?;
    let group_size_i64 = marker
        .get("group_size")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            crate::Error::Msg(format!(
                "packed quant: {} `quantization.group_size` must be an integer",
                config_path.display()
            ))
        })?;
    let group_size = i32::try_from(group_size_i64).map_err(|_| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization.group_size` is out of range: {group_size_i64}",
            config_path.display()
        ))
    })?;
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(crate::Error::Msg(format!(
            "packed quant: {} declares unsupported `quantization.group_size` {group_size}; expected 32, 64, or 128",
            config_path.display()
        )));
    }
    Ok(Some(group_size))
}

/// Decide whether a requested Q4/Q8 tier must be produced by load-time quantization.
///
/// Packed turnkeys must match the request exactly; a mismatch is a hard error because provider
/// `quantize` methods intentionally no-op over packed weights. Direct provider struct loaders that
/// bypass the registry/load seam also bypass this guard by design and must pin their own source tier.
pub fn needs_load_time_quant(
    root: &Path,
    component: &str,
    requested_bits: i32,
    model_id: &str,
) -> Result<bool> {
    if !matches!(requested_bits, 4 | 8) {
        return Err(crate::Error::Msg(format!(
            "{model_id}: unsupported requested quantization tier Q{requested_bits}; expected Q4 or Q8"
        )));
    }
    match packed_quant_bits(root, component)? {
        Some(packed) if packed != requested_bits => Err(crate::Error::Msg(format!(
            "{model_id}: {component}/ is a pre-quantized Q{packed} turnkey but Q{requested_bits} \
             was requested; quantize is a no-op on packed weights so the request would silently \
             serve Q{packed}. Point at a Q{requested_bits} snapshot (or a dense one)."
        ))),
        Some(_) => Ok(false),
        None => Ok(true),
    }
}

/// Derive the quant bit-width from the packed shapes at group size `group_size`: `scales` is
/// `[out, in/gs]` ⇒ `in = scales.cols·gs`; the u32-packed `weight` is `[out, in·bits/32]` ⇒
/// `bits = wq.cols·32/in`. Exact for any group-aligned Q4/Q8 pack, so the bit-width need not be
/// carried in a side manifest.
///
/// F-011: returns `Result` and validates the dtype and exact geometry a corrupt/mis-converted
/// pre-quantized snapshot would otherwise mishandle: codes must be u32, scales floating point, rows
/// must agree, and packed columns must divide exactly rather than truncating into an apparently valid
/// Q4/Q8 width. The shared load seam for every Group-B packed snapshot feeds straight off external
/// `.safetensors`, so these tensors are untrusted.
pub fn packed_bits(wq: &Array, scales: &Array, group_size: i32) -> Result<i32> {
    let sshape = scales.shape();
    let wshape = wq.shape();
    if sshape.len() != 2 || wshape.len() != 2 {
        return Err(crate::Error::Msg(format!(
            "packed quant: scales and weight must be 2-D, got scales {:?} / weight {:?}",
            sshape, wshape
        )));
    }
    if wq.dtype() != Dtype::Uint32 {
        return Err(crate::Error::Msg(format!(
            "packed quant: weight codes must be u32, got {:?}",
            wq.dtype()
        )));
    }
    if !matches!(
        scales.dtype(),
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(crate::Error::Msg(format!(
            "packed quant: scales must be floating point, got {:?}",
            scales.dtype()
        )));
    }
    if sshape[0] != wshape[0] {
        return Err(crate::Error::Msg(format!(
            "packed quant: scales and weight rows differ ({} != {})",
            sshape[0], wshape[0]
        )));
    }
    let in_dim = sshape[1].checked_mul(group_size).ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: input dimension overflow (scales cols {} × group_size {group_size})",
            sshape[1]
        ))
    })?;
    if in_dim == 0 {
        return Err(crate::Error::Msg(format!(
            "packed quant: zero input dim (scales cols {} × group_size {})",
            sshape[1], group_size
        )));
    }
    let packed_width = wshape[1].checked_mul(32).ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: packed width overflow (weight cols {} × 32)",
            wshape[1]
        ))
    })?;
    if packed_width % in_dim != 0 {
        return Err(crate::Error::Msg(format!(
            "packed quant: weight cols {} do not exactly encode scales cols {} × group_size {group_size}",
            wshape[1], sshape[1]
        )));
    }
    let bits = packed_width / in_dim;
    if !matches!(bits, 4 | 8) {
        // Name the assumed `group_size` (sc-15154). Every term here is derived FROM it, so a caller
        // that passes the wrong one gets an illegal width from a perfectly good artifact — Mage's q8
        // vision tower (packed at 64, read at 32) reported "bits 16" and was diagnosed as a bad
        // upload. A reader who can see the assumption can check it before re-hosting anything.
        return Err(crate::Error::Msg(format!(
            "packed quant: inferred bit-width {bits} ∉ {{4, 8}} \
             (weight cols {}, scales cols {} × group_size {group_size} ⇒ in_dim {in_dim}); either \
             the snapshot is corrupt/mis-converted, or it was packed at a different group_size than \
             the {group_size} this loader assumed",
            wshape[1], sshape[1]
        )));
    }
    Ok(bits)
}

/// Load `{base}` as an [`AdaptableLinear`] — **packed** (Q4/Q8) when `{base}.scales` is present (a
/// pre-quantized snapshot, bit-width inferred from the shapes at `group_size`), else **dense**. `bias`
/// additionally loads the dense `{base}.bias` (the quantization's own `{base}.biases` is distinct and
/// always loaded on the packed path). Auto-detection means one loader serves both a dense bf16 and a
/// pre-quantized snapshot with no `quantization` manifest to read. Shared by the provider crates'
/// packed loaders — each passes its own `group_size`.
pub fn lin(w: &Weights, base: &str, bias: bool, group_size: i32) -> Result<AdaptableLinear> {
    let bias = if bias {
        Some(w.require(&format!("{base}.bias"))?.clone())
    } else {
        None
    };
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size)?;
        return Ok(AdaptableLinear::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            bias,
            group_size,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        w.require(&format!("{base}.weight"))?.clone(),
        bias,
    ))
}

/// Load `{base}` as a [`TokenEmbedding`] — packed when `{base}.scales` is present, else dense (the
/// embedding analogue of [`lin`]).
pub fn embedding(w: &Weights, base: &str, group_size: i32) -> Result<TokenEmbedding> {
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size)?;
        return Ok(TokenEmbedding::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            group_size,
            bits,
        ));
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.clone(),
    ))
}

/// Read every tensor of `dir` (possibly sharded safetensors) into an owned key→`Array` map (MLX
/// arrays are ref-counted, so the clone is a handle copy, not a buffer copy). The read side of the
/// offline pre-quantization converters.
pub fn load_dir_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    Ok(w.keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect())
}

/// Materialize (`eval`) + write a key→`Array` map to a single `path` safetensors (one file — a packed
/// component is small enough not to need sharding; the loaders glob `*.safetensors`, so one file
/// replaces the source's shards). Keys are sorted before serialization so rebuilding an identical
/// validated artifact produces byte-identical files that can be bound to publication by SHA-256.
/// The write side of the converters.
pub fn save_map(path: &Path, map: &HashMap<String, Array>) -> Result<()> {
    eval(map.values().collect::<Vec<_>>())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut entries = map.iter().collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(key, _)| *key);
    Array::save_safetensors(
        entries.into_iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

/// Pack every `is_target` `{base}.weight` (group-quantizable 2-D, `in % group_size == 0`,
/// `in >= group_size`) into the triple `{base}.weight` (u32 codes) + `.scales` + `.biases` — the
/// weight cast to **bf16 first** so the pack is byte-identical to the load-time
/// [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) (and the fork's
/// `nn.quantize(bf16)`). Every other tensor (norms, 1-D, non-divisible, non-target) passes through
/// unchanged. `is_target` receives the **base** (the key minus its `.weight` suffix). The shape guard
/// keeps an odd-shaped or 1-D target dense rather than crashing `quantize`.
///
/// Each packed target is **evaluated one tensor at a time** (source first, then the packed triple;
/// sc-24114). Those evals are load-bearing, not redundant: MLX runs a safetensors `Load` on its
/// CPU stream, and a GPU `quantize` over an unread `Load` makes the Metal command buffer wait on
/// the disk read — deferred to one eval over the whole map, that trips the GPU watchdog
/// (`kIOGPUCommandBufferCallbackErrorTimeout` → `SubmissionsIgnored`) on a slow volume.
pub fn quantize_map(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
    is_target: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let base = k.strip_suffix(".weight").filter(|b| is_target(b));
        let packable = base.is_some()
            && v.shape().len() == 2
            && v.shape()[1] % group_size == 0
            && v.shape()[1] >= group_size;
        if let (Some(base), true) = (base, packable) {
            // sc-24114: consume the source bytes BEFORE the GPU quantize kernel is scheduled
            // behind them. A safetensors `Load` runs on MLX's CPU stream; a GPU consumer of a
            // not-yet-read `Load` makes `eval` encode a GPU-timeline wait on that read
            // (`Event::wait(stream)` → `encodeWait`), so a graph of hundreds of lazy
            // `Load → quantize` pairs turns into command buffers that sit on the Metal timeline
            // waiting for gigabytes of `pread` from disk — and past the GPU watchdog that is
            // `kIOGPUCommandBufferCallbackErrorTimeout`, after which every later submission of
            // the process is `SubmissionsIgnored`. Reproduced at 90 production tensors from a
            // ~300 MiB/s disk. Evaluating the source on its own stream first leaves the quantize
            // with an already-resident input (no GPU wait at all), and evaluating the packed
            // triple right away bounds the dense transient to one tensor.
            v.eval()?;
            let wbf16 = v.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
            eval([&wq, &scales, &biases])?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

// ============================================================================================
// Turnkey-assembly glue — the Group-B converter tail the provider `convert.rs` modules used to
// clone verbatim (sc-9108 / F-045). Each provider still owns its per-component pack predicates and
// its `prequantize_turnkey` wiring, but routes the config annotation, the symlink-resolving
// directory copy, and the non-weight asset tail through these shared helpers so the six converters
// stay byte-identical (they had drifted: z-image dropped `LICENSE.txt`; krea's copy skipped the
// symlink deref; sensenova never annotated its config).
// ============================================================================================

/// The canonical set of top-level non-weight assets a turnkey snapshot copies verbatim from the
/// dense source (in addition to the component dirs it packs / passes through): the diffusers pipeline
/// manifest and every license/readme variant an upstream repo might ship. A converter iterates this,
/// copying each that exists (deref'ing HF-cache symlinks). Owning it here fixes the F-045 drift where
/// z-image omitted `LICENSE.txt` and shipped license-less rehosted tiers.
pub const TURNKEY_ASSET_FILES: &[&str] = &[
    "model_index.json",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "README.md",
];

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}` block
/// added. A missing source config starts from an empty object. The written bytes are
/// `serde_json::to_string_pretty` of the merged value, byte-identical across every provider that
/// packs per-component dirs.
///
/// **This block is load-bearing, not provenance.** An earlier version of this doc said the Rust
/// loaders auto-detect packed weights via `{base}.scales` and "ignore this block" — true of the
/// *loaders*, false of everything else, and the SC-15525 review traced a real coverage gap back to
/// believing it. [`packed_quant_bits`] reads this marker and **only** this marker, so a packed
/// component shipped without it is reported as dense and three separate decisions go the wrong way:
///
/// * [`needs_load_time_quant`] answers "yes", which is what the F-144 requested-vs-packed tier guard
///   consults — so a Q4 request against an unmarked packed-Q8 component passes the guard and is then
///   served Q8 by a `quantize()` that no-ops on already-packed weights;
/// * a memory-strategy contract's "does this load leave the transformer lazy" predicate answers "no"
///   and withholds rung 4. That is exactly why `illustrious_xl_v1`/`illustrious_xl_v2` at q8 publish
///   no bounded transformer residency — see `mlx_gen_sdxl::memory_strategy::load_leaves_blocks_lazy`;
/// * anything sizing a fit from the declared tier reads the wrong one.
///
/// A packing path that omits this call produces a snapshot that *works* and is *mislabelled*, which
/// is the hardest kind to notice. Every packing path must write it.
pub fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?).map_err(|e| {
            crate::Error::Msg(format!("quant convert: parse {}: {e}", src_cfg.display()))
        })?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| crate::Error::Msg(format!("quant convert: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}

/// Recursively copy a directory's files, resolving symlinks (HF snapshots symlink into
/// `../../blobs/…`) to real bytes so the assembled tier is self-contained and HF-uploadable. This is
/// the symlink-resolving copy the six sibling converters share (the F-045 unification point where
/// krea's non-deref'ing copy is brought into line).
pub fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        // Never carry a hidden entry into an assembled tier: an AppleDouble sidecar copied in here
        // would be uploaded with the turnkey and then break `Weights::from_dir` for everyone who
        // downloads it (SceneWorks#1333).
        if gen_core::weightsmeta::is_hidden_file(&path) {
            continue;
        }
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            let real = std::fs::canonicalize(&path)?;
            std::fs::copy(&real, &target)?;
        }
    }
    Ok(())
}

/// Copy a single top-level asset `name` from `src_root` to `dst_root` if it exists, deref'ing an
/// HF-cache symlink to real bytes. Returns whether the file was present and copied. The per-file
/// building block behind [`copy_turnkey_assets`] (some converters — e.g. sensenova — need the copied?
/// signal per file).
pub fn copy_asset(src_root: &Path, dst_root: &Path, name: &str) -> Result<bool> {
    let src = src_root.join(name);
    if !src.exists() {
        return Ok(false);
    }
    let real = std::fs::canonicalize(&src)?;
    std::fs::create_dir_all(dst_root)?;
    std::fs::copy(&real, dst_root.join(name))?;
    Ok(true)
}

/// Copy every [`TURNKEY_ASSET_FILES`] entry that exists from `src_root` to `dst_root` (deref'ing
/// symlinks). The shared non-weight tail of `prequantize_turnkey`.
pub fn copy_turnkey_assets(src_root: &Path, dst_root: &Path) -> Result<()> {
    for name in TURNKEY_ASSET_FILES {
        copy_asset(src_root, dst_root, name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker_fixture(tmp: &tempfile::TempDir, body: Option<&str>) -> std::path::PathBuf {
        let root = tmp.path().join(format!(
            "mlx-gen-quant-marker-{:?}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let dir = root.join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(body) = body {
            std::fs::write(dir.join("config.json"), body).unwrap();
        }
        root
    }

    #[test]
    fn quant_tier_missing_or_unmarked_config_is_dense() {
        let tmp = tempfile::tempdir().unwrap();
        for body in [None, Some("{}"), Some(r#"{"hidden_size": 64}"#)] {
            let root = marker_fixture(&tmp, body);
            assert_eq!(packed_quant_bits(&root, "transformer").unwrap(), None);
            assert!(needs_load_time_quant(&root, "transformer", 4, "model").unwrap());
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn quant_tier_match_skips_and_mismatch_names_context() {
        let tmp = tempfile::tempdir().unwrap();
        let root = marker_fixture(&tmp, Some(r#"{"quantization":{"bits":8,"group_size":64}}"#));
        assert!(!needs_load_time_quant(&root, "transformer", 8, "lens_turbo").unwrap());
        let error = needs_load_time_quant(&root, "transformer", 4, "lens_turbo")
            .unwrap_err()
            .to_string();
        assert!(error.contains("lens_turbo"), "{error}");
        assert!(error.contains("transformer"), "{error}");
        assert!(error.contains("Q8") && error.contains("Q4"), "{error}");
        std::fs::remove_dir_all(root).ok();
    }

    /// sc-24114: `quantize_map` consumes every packed target's source bytes before it returns —
    /// the eval boundary that keeps a GPU quantize kernel from waiting on a disk read inside a
    /// Metal command buffer (`kIOGPUCommandBufferCallbackErrorTimeout` on a slow volume).
    ///
    /// Mechanical pin: load a file lazily, run `quantize_map`, then **truncate the file to zero
    /// bytes** before anything downstream evaluates. If the reads were still pending (the
    /// pre-fix graph deferred them to `save_map`'s single eval), the later eval reads a
    /// zero-length file and cannot reproduce the eager reference; if they were consumed inside
    /// `quantize_map`, the packed triple is byte-identical to the reference.
    ///
    /// *Mutation that reds this:* removing the `v.eval()` / `eval([&wq, &scales, &biases])` pair
    /// from `quantize_map`.
    #[test]
    fn quantize_map_consumes_its_source_bytes_before_returning() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let w = Array::from_slice(
            &(0..256 * 128)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            &[256, 128],
        );
        let norm = Array::ones::<f32>(&[128]).unwrap();
        Array::save_safetensors(
            [
                ("blocks.0.proj.weight", &w),
                ("blocks.0.norm.weight", &norm),
            ],
            None::<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        // The eager reference, from bytes read while the file is intact.
        let (ewq, esc, ebi) = quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), 64, 8).unwrap();
        eval([&ewq, &esc, &ebi]).unwrap();

        let lazy = load_dir_map(tmp.path()).unwrap();
        let out = quantize_map(lazy, 8, 64, |_| true).unwrap();
        // Nothing has been evaluated by this test yet: pull the source out from under the map.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();

        for (key, expected) in [
            ("blocks.0.proj.weight", &ewq),
            ("blocks.0.proj.scales", &esc),
            ("blocks.0.proj.biases", &ebi),
        ] {
            let got = out.get(key).expect(key);
            let same = mlx_rs::ops::eq(got, expected)
                .and_then(|m| m.all(None))
                .and_then(|a| a.try_item::<bool>());
            assert!(
                matches!(same, Ok(true)),
                "{key}: packed output does not match the eager reference after the source was \
                 truncated ({same:?}) — quantize_map deferred its reads"
            );
        }
    }

    #[test]
    fn packed_group_size_accepts_only_mlx_affine_geometries() {
        let tmp = tempfile::tempdir().unwrap();
        for group_size in [32, 64, 128] {
            let body = format!(r#"{{"quantization":{{"bits":8,"group_size":{group_size}}}}}"#);
            let root = marker_fixture(&tmp, Some(&body));
            assert_eq!(
                packed_quant_group_size_at(&root.join("transformer")).unwrap(),
                Some(group_size)
            );
            std::fs::remove_dir_all(root).ok();
        }
        for marker in [
            r#"{"quantization":{"bits":8}}"#,
            r#"{"quantization":{"bits":8,"group_size":"32"}}"#,
            r#"{"quantization":{"bits":8,"group_size":16}}"#,
            r#"{"quantization":{"bits":8,"group_size":2147483648}}"#,
        ] {
            let root = marker_fixture(&tmp, Some(marker));
            assert!(
                packed_quant_group_size_at(&root.join("transformer")).is_err(),
                "group-size marker must fail: {marker}"
            );
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn quant_tier_rejects_every_present_but_malformed_marker() {
        let tmp = tempfile::tempdir().unwrap();
        for body in [
            "{",
            r#"{"quantization":null}"#,
            r#"{"quantization":[]}"#,
            r#"{"quantization":{}}"#,
            r#"{"quantization":{"bits":"4"}}"#,
            r#"{"quantization":{"bits":4.0}}"#,
            r#"{"quantization":{"bits":3}}"#,
            r#"{"quantization":{"bits":2147483648}}"#,
        ] {
            let root = marker_fixture(&tmp, Some(body));
            assert!(
                packed_quant_bits(&root, "transformer").is_err(),
                "marker must fail: {body}"
            );
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn quant_tier_only_swallows_not_found_and_rejects_unsafe_components() {
        let tmp = tempfile::tempdir().unwrap();
        let root = marker_fixture(&tmp, None);
        std::fs::create_dir(root.join("transformer/config.json")).unwrap();
        assert!(packed_quant_bits(&root, "transformer").is_err());
        for component in ["", "../transformer", "/transformer", "transformer/../vae"] {
            assert!(
                packed_quant_bits(&root, component).is_err(),
                "{component:?}"
            );
        }
        assert!(needs_load_time_quant(&root, "vae", 3, "model").is_err());
        std::fs::remove_dir_all(root).ok();
    }

    /// F-011: a Q4 pack at group_size 64 derives bits == 4 from the standard shapes.
    #[test]
    fn packed_bits_derives_q4() {
        // scales [out, in/gs] = [128, 4] ⇒ in_dim 256; wq [out, in·4/32] = [128, 32] ⇒ bits 4.
        let scales = Array::zeros::<f32>(&[128, 4]).unwrap();
        let wq = Array::zeros::<u32>(&[128, 32]).unwrap();
        assert_eq!(packed_bits(&wq, &scales, 64).unwrap(), 4);
    }

    #[test]
    fn packed_bits_derives_q8() {
        // scales [out, in/gs] = [64, 2] ⇒ in_dim 128; wq [out, in·8/32] = [64, 32] ⇒ bits 8.
        let scales = Array::zeros::<f32>(&[64, 2]).unwrap();
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        assert_eq!(packed_bits(&wq, &scales, 64).unwrap(), 8);
    }

    #[test]
    fn packed_bits_rejects_1d_shapes() {
        let scales = Array::zeros::<f32>(&[64]).unwrap(); // 1-D
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("must be 2-D"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_zero_in_dim() {
        // [out, 0] scales ⇒ in_dim 0 ⇒ integer divide-by-zero today; now a typed error.
        let scales = Array::zeros::<f32>(&[64, 0]).unwrap();
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("zero input dim"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_non_q4_q8_width() {
        // in_dim 256, wq cols 24 ⇒ bits 24·32/256 = 3 ∉ {4,8}.
        let scales = Array::zeros::<f32>(&[64, 4]).unwrap(); // in_dim 256
        let wq = Array::zeros::<u32>(&[64, 24]).unwrap(); // bits 3
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("∉ {4, 8}"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_truncated_width_wrong_rows_and_non_u32_codes() {
        let scales = Array::zeros::<f32>(&[2, 2]).unwrap();
        let truncated = Array::zeros::<u32>(&[2, 33]).unwrap();
        assert!(packed_bits(&truncated, &scales, 64).is_err());

        let wrong_rows = Array::zeros::<u32>(&[3, 32]).unwrap();
        assert!(packed_bits(&wrong_rows, &scales, 64).is_err());

        let float_codes = Array::zeros::<f32>(&[2, 32]).unwrap();
        assert!(packed_bits(&float_codes, &scales, 64).is_err());
    }
}
