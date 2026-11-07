//! Weight loading — safetensors → MLX arrays by dotted key, plus file metadata, with a
//! dtype helper. No torch dependency: reads safetensors directly via mlx-rs.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype};

use crate::{Error, Result};

/// A loaded set of named tensors (dotted keys, e.g. `"layers.0.attention.to_q.weight"`)
/// plus the file's string metadata (e.g. a LoKr adapter's `networkType` / `alpha` / `rank`).
#[derive(Clone)]
pub struct Weights {
    tensors: HashMap<String, Array>,
    metadata: HashMap<String, String>,
    accessed: RefCell<std::collections::HashSet<String>>,
}

impl Weights {
    /// Build weights from an already materialized tensor map.
    pub fn from_map(tensors: HashMap<String, Array>) -> Self {
        Self {
            tensors,
            metadata: HashMap::new(),
            accessed: RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// An empty weights container — for building small in-memory fixtures (e.g. checkpoint-remap unit
    /// tests that assert `remap_*_keys` produces the aliased names without a real snapshot).
    pub fn empty() -> Self {
        Self {
            tensors: HashMap::new(),
            metadata: HashMap::new(),
            accessed: RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// Load a single `.safetensors` file (tensors + metadata).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let (tensors, metadata) = Array::load_safetensors_with_metadata(path.as_ref())?;
        Ok(Self {
            tensors,
            metadata,
            accessed: RefCell::new(std::collections::HashSet::new()),
        })
    }

    /// Evaluate every retained array before a mutation-sensitive source read is considered complete.
    ///
    /// MLX safetensors loads and the remap/cast operations layered on top of them are lazy. A caller
    /// that only constructs a [`Weights`] value inside `PinnedWeightsFile::read_unchanged` would
    /// otherwise run the post-read fingerprint check before the file-backed graph had consumed its
    /// payload. Provider loaders call this after their final normalization so both source arrays and
    /// derived casts/dequantizations are materialized while the pin guard is still active.
    ///
    /// Materializing is also the load boundary at which the GPU's view of every fresh buffer is
    /// verified against the CPU's ([`crate::coherence`], sc-22414), before any graph consumes it.
    ///
    /// Evaluation is batched at [`Self::MATERIALIZE_BATCH_BYTES`] in sorted-key order so a cold
    /// multi-gigabyte checkpoint never becomes one submission.
    pub fn materialize(&self) -> Result<()> {
        let mut named: Vec<(&str, &Array)> = self
            .tensors
            .iter()
            .map(|(key, array)| (key.as_str(), array))
            .collect();
        Self::materialize_and_verify(&mut named)
    }

    /// Upper bound on the bytes one [`Self::materialize`] / [`Self::materialize_accessed`] batch
    /// evaluates at once.
    pub const MATERIALIZE_BATCH_BYTES: usize = 512 * 1024 * 1024;

    fn materialize_and_verify(named: &mut [(&str, &Array)]) -> Result<()> {
        named.sort_unstable_by_key(|(key, _)| *key);
        let mut start = 0;
        let mut bytes = 0usize;
        for end in 0..named.len() {
            bytes = bytes.saturating_add(named[end].1.nbytes());
            if bytes >= Self::MATERIALIZE_BATCH_BYTES {
                Self::materialize_batch(&named[start..=end])?;
                start = end + 1;
                bytes = 0;
            }
        }
        if start < named.len() {
            Self::materialize_batch(&named[start..])?;
        }
        Ok(())
    }

    /// Every materialize path evaluates through here. The batch's pending safetensors reads are
    /// evaluated first, on their own: MLX runs a `Load` on its CPU stream, and evaluating a derived
    /// array (a GPU-stream cast/transpose/remap) over an unread `Load` makes the Metal command
    /// buffer wait on the disk read — on a slow or external drive past the GPU watchdog
    /// (`kIOGPUCommandBufferCallbackErrorTimeout`; sd3, ideogram and seedvr2 in the sc-24245
    /// campaign). Reading the loads first means the GPU only ever sees resident inputs.
    fn materialize_batch(batch: &[(&str, &Array)]) -> Result<()> {
        mlx_rs::transforms::eval_pending_loads(batch.iter().map(|(_, array)| *array))?;
        mlx_rs::transforms::eval(batch.iter().map(|(_, array)| *array))?;
        crate::coherence::verify_gpu_view(batch.iter().copied())
    }

    /// Evaluate only tensors read through [`Self::get`] / [`Self::require`] since the last drain.
    /// Block-window loaders use this before [`Self::remove_accessed`] so the source bytes for the
    /// current window are consumed under its immutable-file guard without evaluating the rest of the
    /// checkpoint and defeating bounded residency.
    ///
    /// Like [`Self::materialize`], this verifies the GPU's view of each window's fresh buffers
    /// against the CPU's before the window's blocks consume them (sc-22414).
    pub fn materialize_accessed(&self) -> Result<()> {
        let accessed = self.accessed.borrow();
        let mut named: Vec<(&str, &Array)> = accessed
            .iter()
            .filter_map(|key| self.tensors.get(key).map(|array| (key.as_str(), array)))
            .collect();
        Self::materialize_and_verify(&mut named)
    }

    /// [`Self::materialize`]'s evaluate-and-verify pass over arrays that have already **left** a
    /// map — the same batching and the same sc-22414 GPU-view check, for a loader whose source
    /// handles must be verified later than its build (a resident DiT verifies its block bodies only
    /// after the AdaLN eviction, so the check never raises the phase peak).
    pub fn materialize_named<'a>(named: &mut [(&'a str, &'a Array)]) -> Result<()> {
        Self::materialize_and_verify(named)
    }

    /// Handle copies of every tensor read through [`Self::get`] / [`Self::require`] so far that is
    /// still in the map, in sorted-key order. `Array` is refcounted, so these are the **same
    /// buffers** the built model holds — retaining them adds no bytes, and a loader can hand them
    /// to [`Self::materialize_named`] at the point in its residency schedule where the load must be
    /// verified without changing what is resident.
    pub fn accessed_entries(&self) -> Vec<(String, Array)> {
        let accessed = self.accessed.borrow();
        let mut named: Vec<(String, Array)> = accessed
            .iter()
            .filter_map(|key| {
                self.tensors
                    .get(key)
                    .map(|array| (key.clone(), array.clone()))
            })
            .collect();
        named.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        named
    }

    /// Load a safetensors file while decoding `F8_E4M3` payloads to bf16.
    ///
    /// MLX has no fp8 storage dtype, but does expose byte-accurate E4M3
    /// conversion. The regular loader therefore rejects fp8 files before a
    /// provider can normalize them; this opt-in loader preserves the ordinary
    /// path for all other dtypes and converts only fp8 tensor views.
    pub fn from_file_with_fp8(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_fp8_filter(path, |_| true)
    }

    /// Load only the selected tensors from an fp8-capable safetensors file.
    ///
    /// Filtering happens on tensor names before any MLX array is created. Component-staged loaders
    /// use this for fused checkpoints so a text-only phase never transiently materializes the
    /// transformer or VAE.
    pub fn from_file_with_fp8_filter(
        path: impl AsRef<Path>,
        mut include: impl FnMut(&str) -> bool,
    ) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: the mapping is read-only and remains alive until every borrowed
        // TensorView has been copied into MLX-owned array storage below.
        let bytes = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let safe = safetensors_fp8::SafeTensors::deserialize(&bytes).map_err(|error| {
            Error::from(format!("loading {}: {error}", path.as_ref().display()))
        })?;
        let mut tensors = HashMap::new();
        for (name, view) in safe.tensors() {
            if !include(&name) {
                continue;
            }
            let shape: Vec<i32> = view
                .shape()
                .iter()
                .map(|&dim| {
                    i32::try_from(dim).map_err(|_| {
                        Error::Msg(format!(
                            "loading {}: tensor {name:?} dimension {dim} exceeds MLX i32 shape range",
                            path.as_ref().display()
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            let value = if view.dtype() == safetensors_fp8::Dtype::F8_E4M3 {
                let elements = shape.iter().try_fold(1i32, |count, &dim| {
                    count.checked_mul(dim).ok_or_else(|| {
                        Error::Msg(format!(
                            "loading {}: fp8 tensor {name:?} element count exceeds MLX's i32 \
                             Array::from_slice range",
                            path.as_ref().display()
                        ))
                    })
                })?;
                if elements as usize != view.data().len() {
                    return Err(Error::Msg(format!(
                        "loading {}: fp8 tensor {name:?} shape describes {elements} elements but \
                         stores {} bytes",
                        path.as_ref().display(),
                        view.data().len()
                    )));
                }
                Array::from_slice(view.data(), &shape).from_fp8(Dtype::Bfloat16)?
            } else {
                let dtype = match view.dtype() {
                    safetensors_fp8::Dtype::BOOL => Dtype::Bool,
                    safetensors_fp8::Dtype::U8 => Dtype::Uint8,
                    safetensors_fp8::Dtype::U16 => Dtype::Uint16,
                    safetensors_fp8::Dtype::U32 => Dtype::Uint32,
                    safetensors_fp8::Dtype::U64 => Dtype::Uint64,
                    safetensors_fp8::Dtype::I8 => Dtype::Int8,
                    safetensors_fp8::Dtype::I16 => Dtype::Int16,
                    safetensors_fp8::Dtype::I32 => Dtype::Int32,
                    safetensors_fp8::Dtype::I64 => Dtype::Int64,
                    safetensors_fp8::Dtype::F16 => Dtype::Float16,
                    safetensors_fp8::Dtype::BF16 => Dtype::Bfloat16,
                    safetensors_fp8::Dtype::F32 => Dtype::Float32,
                    safetensors_fp8::Dtype::F64 => Dtype::Float64,
                    other => {
                        return Err(Error::Msg(format!(
                            "loading {}: unsupported tensor dtype {other:?}",
                            path.as_ref().display()
                        )))
                    }
                };
                // MLX copies the TensorView bytes into its own array storage.
                unsafe {
                    Array::from_raw_data(
                        view.data().as_ptr().cast::<std::ffi::c_void>(),
                        &shape,
                        dtype,
                    )
                }
            };
            tensors.insert(name.to_owned(), value);
        }
        Ok(Self {
            tensors,
            metadata: HashMap::new(),
            accessed: RefCell::new(std::collections::HashSet::new()),
        })
    }

    /// Load and merge every `.safetensors` file under `dir` (sharded checkpoints). Keys
    /// across shards are disjoint, so a plain merge reconstructs the full tensor set
    /// without parsing the index — no torch, no shard map needed.
    ///
    /// Hidden entries are skipped: macOS AppleDouble sidecars (`._model.safetensors`) carry the
    /// `.safetensors` extension but are not shards, and sort ahead of the real file — see
    /// [`gen_core::weightsmeta::is_hidden_file`].
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .filter(|p| !gen_core::weightsmeta::is_hidden_file(p))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors files in {}", dir.display()).into());
        }
        Self::from_paths(&files, dir)
    }

    /// Load a previously selected shard set with the same disjoint-key and metadata behavior as
    /// [`Self::from_dir`]. The caller chooses and validates the complete path list before loading.
    pub fn from_paths(files: &[PathBuf], source_dir: &Path) -> Result<Self> {
        if files.is_empty() {
            return Err(format!("no .safetensors files in {}", source_dir.display()).into());
        }
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for f in files {
            // Name the offending file: the underlying mlx-c error carries only its own C++ source
            // location, so a corrupt/foreign file in a shard dir was previously undiagnosable.
            let (t, m) = Array::load_safetensors_with_metadata(f)
                .map_err(|e| Error::from(format!("loading shard {}: {e}", f.display())))?;
            // Shards are expected to be disjoint; a key collision means the shard set is wrong (e.g.
            // a stray extra file in the dir) and a plain `extend` would silently let the later shard
            // win, loading a partially-wrong tensor set. Surface it instead (F-032). Metadata is
            // descriptive (per-shard `__metadata__`), so a later-wins merge there is benign.
            for (k, v) in t {
                if tensors.insert(k.clone(), v).is_some() {
                    return Err(format!(
                        "duplicate tensor key `{k}` across shards in {} (non-disjoint shard set)",
                        source_dir.display()
                    )
                    .into());
                }
            }
            metadata.extend(m);
        }
        Ok(Self {
            tensors,
            metadata,
            accessed: RefCell::new(std::collections::HashSet::new()),
        })
    }

    pub fn get(&self, key: &str) -> Option<&Array> {
        let value = self.tensors.get(key)?;
        self.accessed.borrow_mut().insert(key.to_owned());
        Some(value)
    }

    /// Get a tensor by key, returning an error (not panicking) when it is absent.
    pub fn require(&self, key: &str) -> Result<&Array> {
        let value = self
            .tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_string()))?;
        self.accessed.borrow_mut().insert(key.to_owned());
        Ok(value)
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Insert (or overwrite) a tensor under `key`. Used by checkpoint remapping (diffusers →
    /// internal names + conv-weight transposes) when loading real weights.
    pub fn insert(&mut self, key: impl Into<String>, tensor: Array) {
        self.tensors.insert(key.into(), tensor);
    }

    /// Copy the tensor at `from` to the new key `to` (no-op if `from` is absent). A convenience
    /// for the identity-but-renamed entries in a checkpoint→internal mapping.
    pub fn alias(&mut self, from: &str, to: &str) {
        if let Some(t) = self.tensors.get(from).cloned() {
            self.tensors.insert(to.to_string(), t);
        }
    }

    /// Remove and return the tensor at `key` (`None` if absent). Lets a **consuming** loader release
    /// each source tensor from the map as soon as it has been copied/quantized into the built model, so
    /// the source and the growing built model don't both stay resident — bounding the load-time
    /// unified-memory transient to ~the built size (sc-11030; the gpt-oss encoder is otherwise a
    /// 13 GB-source + 21/63 GB-built spike). MLX returns the dropped buffer to its reuse pool.
    pub fn remove(&mut self, key: &str) -> Option<Array> {
        self.tensors.remove(key)
    }

    /// Drop every tensor whose key starts with `prefix` (e.g. `"model.layers.7."` once layer 7 is
    /// built). The companion to [`remove`](Self::remove) for a whole sub-module's source tensors.
    pub fn remove_prefix(&mut self, prefix: &str) {
        self.tensors.retain(|k, _| !k.starts_with(prefix));
    }

    /// Remove every tensor successfully read through [`require`](Self::require) since the previous
    /// call. A consuming loader can then reject any requested-prefix key left behind, which makes an
    /// omitted constructor read observable instead of hiding it behind wholesale prefix deletion.
    pub fn remove_accessed(&mut self) {
        let accessed = std::mem::take(self.accessed.get_mut());
        for key in accessed {
            self.tensors.remove(&key);
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Every stored key that has **not** been read through [`get`](Self::get)/[`require`](Self::require)
    /// so far — the complement of `remove_accessed`'s tracking set. A loader-conformance test builds a
    /// candidate weight map, constructs the model against it, then asserts this is empty to prove no
    /// tensor was silently ignored (sc-18758: e.g. a config that says `ff_bias:false` must not merely
    /// skip *requiring* the bias — a test wants to also confirm nothing else in the map went unread).
    pub fn unused_keys(&self) -> Vec<&str> {
        let accessed = self.accessed.borrow();
        self.tensors
            .keys()
            .map(String::as_str)
            .filter(|k| !accessed.contains(*k))
            .collect()
    }

    /// Consume the container into its raw tensor map (drops metadata and access bookkeeping).
    /// The logical-weight reader uses this to unify the ordinary lazy loader with its fp8-view
    /// loader, which never has metadata to begin with.
    pub fn into_tensors(self) -> HashMap<String, Array> {
        self.tensors
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Cast every tensor to `dtype` in place — mirrors the way a reference loader downcasts the
    /// whole checkpoint at load (e.g. the vendored SDXL `_load_safetensor_weights(..., float16=True)`
    /// applies `v.astype(mx.float16)` to every tensor). A no-op when `dtype` already matches.
    pub fn cast_all(&mut self, dtype: Dtype) -> Result<()> {
        // Per tensor this transiently holds both the source and the cast copy until the reassignment
        // drops the source (~2× that tensor's bytes at the moment of cast). Fine for the per-tensor
        // granularity here; a chunked cast+eval would be the lever if a full-checkpoint downcast ever
        // shows up as a peak-memory problem (F-089).
        for v in self.tensors.values_mut() {
            if v.dtype() != dtype {
                *v = v.as_dtype(dtype)?;
            }
        }
        Ok(())
    }

    /// Cast tensors selected by key while preserving every unselected tensor verbatim.
    /// This is useful for mixed-storage modules whose large projection/embedding weights use a
    /// compact dtype while norms, packed codes, scales, and unrelated subtrees retain theirs.
    pub fn cast_matching(
        &mut self,
        dtype: Dtype,
        mut selected: impl FnMut(&str) -> bool,
    ) -> Result<()> {
        for (key, value) in &mut self.tensors {
            if selected(key) && value.dtype() != dtype {
                *value = value.as_dtype(dtype)?;
            }
        }
        Ok(())
    }
}

/// Cast to a target compute dtype (e.g. bf16, mirroring mflux's torch_convert downcast). A thin
/// `as_dtype` passthrough — kept as a named helper so call sites read as an explicit downcast (and
/// `as_dtype` itself no-ops when the dtype already matches) (F-089).
pub fn to_dtype(a: &Array, dtype: Dtype) -> Result<Array> {
    Ok(a.as_dtype(dtype)?)
}

/// Upcast to `f32`. The common case of [`to_dtype`] — providers that pin reductions/preprocessing
/// to f32 (e.g. scail2's CLIP/preprocess islands) call this so the intent reads at the use site.
pub fn to_f32(a: &Array) -> Result<Array> {
    Ok(a.as_dtype(Dtype::Float32)?)
}

/// Build a dotted tensor key from a `prefix` and a leaf `name`, collapsing the empty-prefix case
/// (so a root module addresses `"weight"`, a nested one `"blocks.0.weight"`). The Rust analogue of
/// the fork's `f"{prefix}.{name}"` key assembly — shared so provider loaders/text-encoders that
/// walk `tree_flatten`-style names don't each re-spell it.
pub fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_file_with_metadata() {
        // Per-process scratch dir (matching `mlx_gen_appledouble_{pid}` below) — a fixed `$TMPDIR`
        // name races a second concurrent `cargo test`.
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let path = dir.join("w.safetensors");

        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        Array::save_safetensors(vec![("blk.weight", &a)], Some(&meta), &path).unwrap();

        let w = Weights::from_file(&path).unwrap();
        assert_eq!(w.len(), 1);
        assert!(w.get("blk.weight").is_some());
        assert!(w.require("blk.weight").is_ok());
        assert!(w.require("missing").is_err());
        assert_eq!(w.metadata("networkType"), Some("lokr"));
    }

    #[test]
    fn fp8_capable_filter_materializes_only_selected_tensor_names() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let path = dir.join("components.safetensors");
        let text = Array::from_slice(&[1.0f32], &[1]);
        let transformer = Array::from_slice(&[2.0f32], &[1]);
        Array::save_safetensors(
            vec![
                ("text_encoder.weight", &text),
                ("transformer.weight", &transformer),
            ],
            None,
            &path,
        )
        .unwrap();

        let weights =
            Weights::from_file_with_fp8_filter(&path, |name| name.starts_with("text_encoder"))
                .unwrap();
        assert_eq!(weights.keys().collect::<Vec<_>>(), ["text_encoder.weight"]);
    }

    #[test]
    fn to_dtype_casts_to_bf16() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        assert_eq!(
            to_dtype(&a, Dtype::Bfloat16).unwrap().dtype(),
            Dtype::Bfloat16
        );
    }

    #[test]
    fn cast_matching_only_casts_selected_tensors() {
        let mut w = Weights::empty();
        w.insert(
            "language_model.layers.0.mlp.weight",
            Array::from_slice(&[1.0f32], &[1]),
        );
        w.insert(
            "language_model.layers.0.norm.weight",
            Array::from_slice(&[1.0f32], &[1]),
        );

        w.cast_matching(Dtype::Bfloat16, |key| key.contains("mlp"))
            .unwrap();

        assert_eq!(
            w.require("language_model.layers.0.mlp.weight")
                .unwrap()
                .dtype(),
            Dtype::Bfloat16
        );
        assert_eq!(
            w.require("language_model.layers.0.norm.weight")
                .unwrap()
                .dtype(),
            Dtype::Float32
        );
    }

    /// SceneWorks#1333 regression: `boogu_image` failed to load because its `mllm/` dir held a macOS
    /// AppleDouble sidecar next to the real shard. `._model.safetensors` has extension `safetensors`
    /// and sorts first, so `from_dir` opened it and mlx-c rejected its magic bytes with
    /// `[load_safetensors] Invalid json header length`. The sidecar must be skipped.
    #[test]
    fn from_dir_skips_appledouble_sidecar() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();

        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        Array::save_safetensors(
            vec![("blk.weight", &a)],
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        // Real AppleDouble header (magic 0x00051607, version 0x00020000).
        std::fs::write(
            dir.join("._model.safetensors"),
            [0x00, 0x05, 0x16, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00],
        )
        .unwrap();

        let w = Weights::from_dir(&dir).expect("sidecar must be skipped, not loaded");
        assert_eq!(w.len(), 1);
        assert!(w.get("blk.weight").is_some());
    }

    /// A genuinely corrupt *shard* still errors — and the message now names the file, which the bare
    /// mlx-c error (a C++ source location) did not.
    #[test]
    fn from_dir_error_names_the_offending_shard() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        std::fs::write(dir.join("model.safetensors"), [0x00, 0x05, 0x16, 0x07]).unwrap();

        let err = match Weights::from_dir(&dir) {
            Ok(_) => panic!("a corrupt shard must not load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("model.safetensors"), "unexpected: {err}");
    }

    /// `accessed_entries` returns exactly the read keys that are still mapped — a key read and then
    /// removed is not reported, an unread key never is — as refcounted handles of the same buffers.
    /// MUTATION: report every key (drop the `accessed` filter) — RED on the unread key.
    #[test]
    fn accessed_entries_are_the_read_keys_still_in_the_map() {
        let mut w = Weights::empty();
        w.insert("b.weight", Array::from_slice(&[1.0f32, 2.0], &[2]));
        w.insert("a.weight", Array::from_slice(&[3.0f32], &[1]));
        w.insert("unread.weight", Array::from_f32(9.0));
        w.insert("gone.weight", Array::from_f32(4.0));
        w.require("b.weight").unwrap();
        w.get("a.weight").unwrap();
        w.require("gone.weight").unwrap();
        w.remove("gone.weight");

        let entries = w.accessed_entries();
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["a.weight", "b.weight"], "sorted, read, still mapped");
        assert_eq!(entries[1].1.as_slice::<f32>(), &[1.0, 2.0]);

        let mut named: Vec<(&str, &Array)> = entries.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Weights::materialize_named(&mut named).unwrap();
    }

    #[test]
    fn remove_accessed_leaves_one_omitted_constructor_key_as_a_discriminator() {
        let mut w = Weights::empty();
        w.insert("vae.block.small.weight", Array::from_f32(1.0));
        w.insert("vae.block.small.bias", Array::from_f32(0.0));
        w.require("vae.block.small.weight").unwrap();
        // Mutation: constructor forgot the bias read. Exact access-draining must leave it visible.
        w.remove_accessed();
        assert!(w.get("vae.block.small.weight").is_none());
        assert!(w.get("vae.block.small.bias").is_some());
        assert_eq!(
            w.keys().filter(|key| key.starts_with("vae.block.")).count(),
            1
        );
    }

    fn is_available(a: &Array) -> bool {
        let mut available = false;
        let status = unsafe { mlx_sys::_mlx_array_is_available(&mut available, a.as_ptr()) };
        assert_eq!(status, 0);
        available
    }

    /// sc-24245: a derived (GPU-stream cast + transpose) array over a lazy safetensors `Load`,
    /// stored under an accessed key, materializes to the right values with its source read.
    #[test]
    fn materialize_accessed_evaluates_a_derived_array_over_a_lazy_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.safetensors");
        let source = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        Array::save_safetensors(vec![("w", &source)], None, &path).unwrap();

        let mut w = Weights::from_file(&path).unwrap();
        let raw = w.require("w").unwrap().clone();
        let derived = raw
            .as_dtype(Dtype::Float16)
            .unwrap()
            .transpose_axes(&[1, 0])
            .unwrap();
        w.insert("w", derived);
        assert!(!is_available(&raw));

        w.materialize_accessed().unwrap();
        let derived = w.require("w").unwrap();
        assert!(is_available(&raw));
        assert!(is_available(derived));
        let expected = Array::from_slice(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
        assert_eq!(derived.as_dtype(Dtype::Float32).unwrap(), expected);
    }
}
