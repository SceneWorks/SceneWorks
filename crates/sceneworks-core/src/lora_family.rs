//! Detects the architecture family of a LoRA file from its safetensors header.
//!
//! The detector inspects the tensor key names parsed from a safetensors
//! header and matches them against a table of architecture signatures.
//! It is deliberately conservative: it only returns a family when the
//! evidence is strong and unambiguous. Callers should treat `None` as
//! "we cannot prove the user is wrong" and accept a user-supplied family,
//! while a `Some(family)` that disagrees with a user-supplied family is
//! grounds to reject the import.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

/// Maximum allowed safetensors header size, in bytes. Matches the
/// pre-existing 16 MiB cap enforced by the rust-api.
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// Errors returned when reading a safetensors header from disk.
#[derive(Debug)]
pub enum SafetensorsHeaderError {
    /// The header bytes could not be read from disk.
    Io(std::io::Error),
    /// The file did not contain a valid safetensors header (too short,
    /// implausible length, or non-JSON contents).
    InvalidHeader,
    /// The header parsed cleanly but the file is too small to hold the tensor
    /// data the header declares — the file is truncated/incomplete (e.g. an
    /// interrupted download). `declared` is the minimum size the header implies
    /// (`8 + header_len + max(tensor data_offsets end)`), `actual` is the size
    /// on disk.
    IncompleteData { declared: u64, actual: u64 },
}

impl std::fmt::Display for SafetensorsHeaderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidHeader => formatter.write_str("invalid safetensors header"),
            Self::IncompleteData { declared, actual } => write!(
                formatter,
                "incomplete or truncated safetensors: file is {actual} bytes but the header \
                 declares tensor data requiring at least {declared} bytes"
            ),
        }
    }
}

impl std::error::Error for SafetensorsHeaderError {}

/// Reads and JSON-decodes the safetensors header from `path`. The file
/// layout is: 8-byte little-endian header length, then the JSON header,
/// then tensor data — only the header is read.
pub fn read_safetensors_header(path: &Path) -> Result<Value, SafetensorsHeaderError> {
    let metadata = fs::metadata(path).map_err(SafetensorsHeaderError::Io)?;
    if metadata.len() < 8 {
        return Err(SafetensorsHeaderError::InvalidHeader);
    }
    let mut file = fs::File::open(path).map_err(SafetensorsHeaderError::Io)?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len + 8 > metadata.len() {
        return Err(SafetensorsHeaderError::InvalidHeader);
    }
    let header_len_usize =
        usize::try_from(header_len).map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    let mut header = vec![0_u8; header_len_usize];
    file.read_exact(&mut header)
        .map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    let header = serde_json::from_slice::<Value>(&header)
        .map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    // A valid header can still front a truncated/incomplete file (an interrupted
    // download): the data section must be large enough to hold every tensor the
    // header declares. The tensor `data_offsets` are relative to the byte buffer
    // that begins right after the 8-byte length and the header JSON, so the file
    // must be at least `8 + header_len + max(data_offsets end)` bytes. Without this
    // the bad file is accepted at import and only fails cryptically at load time
    // ("invalid data offsets exceeding the size of the file"). See sc-6072.
    let declared = 8_u64
        .saturating_add(header_len)
        .saturating_add(max_tensor_data_end(&header));
    if metadata.len() < declared {
        return Err(SafetensorsHeaderError::IncompleteData {
            declared,
            actual: metadata.len(),
        });
    }
    Ok(header)
}

/// The largest `data_offsets` end across all tensor entries in a parsed
/// safetensors header — i.e. the length of the tensor data section the header
/// declares. The `__metadata__` key and any entry without a well-formed
/// two-element `data_offsets` array contribute nothing (they carry no tensor
/// bytes). Returns 0 for a header with no tensors.
fn max_tensor_data_end(header: &Value) -> u64 {
    let Some(entries) = header.as_object() else {
        return 0;
    };
    entries
        .iter()
        .filter(|(key, _)| key.as_str() != "__metadata__")
        .filter_map(|(_, tensor)| {
            tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .and_then(|offsets| offsets.get(1))
                .and_then(Value::as_u64)
        })
        .max()
        .unwrap_or(0)
}

/// Returns the first `.safetensors` file at or below `path`. When `path`
/// itself is a `.safetensors` file it is returned directly. Returns `None`
/// when no file is found or `path` is neither a file nor a directory.
///
/// Hidden entries are skipped ([`is_hidden_file`]) — the `read_dir` here is
/// *unordered*, so an AppleDouble sidecar (`._adapter.safetensors`) could
/// otherwise be returned in place of the real adapter (SceneWorks#1333).
pub fn first_safetensors_path(path: &Path) -> Option<PathBuf> {
    if path.is_file() && is_safetensors_file(path) {
        return Some(path.to_path_buf());
    }
    if !path.is_dir() {
        return None;
    }
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = fs::read_dir(current).ok()?;
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                stack.push(entry_path);
            } else if is_safetensors_file(&entry_path) {
                return Some(entry_path);
            }
        }
    }
    None
}

/// True when `path`'s file name begins with `.` — a hidden entry that is never
/// a weight or adapter file.
///
/// macOS writes an **AppleDouble sidecar** (`._<name>`) beside a file whenever
/// it must persist extended attributes on a volume with no native xattr support
/// (exFAT/FAT drives, SMB/NFS shares, cloud-sync folders); they also survive a
/// Finder copy or a zip round-trip. `._model.safetensors` has extension
/// `safetensors`, so an extension-only filter admits it — and since `.` sorts
/// first, a sorted loader opens it *before* the real file, hits its AppleDouble
/// magic, and dies on a bogus header length (SceneWorks#1333). No legitimate
/// weight file starts with `.`, so skipping hidden entries is exact.
///
/// Mirrors `gen_core::weightsmeta::is_hidden_file`, restated here because
/// `sceneworks-core` deliberately carries no gen-core dependency.
pub fn is_hidden_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

/// True when `path` is a loadable `.safetensors` file: the extension matches
/// (case-insensitively — some re-hosted checkpoints ship `.SAFETENSORS`) and the
/// entry is not hidden.
///
/// This is the predicate every directory scan should use. A bare extension test
/// admits macOS AppleDouble sidecars — see [`is_hidden_file`].
pub fn is_safetensors_file(path: &Path) -> bool {
    has_safetensors_extension(path) && !is_hidden_file(path)
}

fn has_safetensors_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
}

/// Resolves the adapter file to load from a LoRA record `dir`, preferring `declared`
/// (the plain filename the record's manifest `files` list names) when it exists,
/// otherwise falling back to [`first_safetensors_path`] (sc-10221).
///
/// A trainer leaves periodic step checkpoints (`<stem>-stepNNN.safetensors`, `save_every`
/// default 250) in the same folder as the final `<stem>.safetensors`; a bare directory
/// scan (`first_safetensors_path`, unordered `read_dir`) can therefore load an
/// under-trained checkpoint — and since `-stepNNN` sorts before `.safetensors`, a
/// checkpoint is even the likely pick. Honoring the declared final name loads the
/// intended adapter deterministically.
///
/// `declared` is treated as untrusted (it rides the job payload): only a plain in-`dir`
/// filename — no path separators, not `.`/`..` — is accepted, so a crafted `files` value
/// cannot redirect the load outside `dir`. Anything else falls through to the scan.
pub fn resolve_adapter_in_dir(dir: &Path, declared: Option<&str>) -> Option<PathBuf> {
    if let Some(name) = declared.map(str::trim).filter(|name| !name.is_empty()) {
        let is_plain = name != "."
            && name != ".."
            && !name.contains('/')
            && !name.contains('\\')
            && Path::new(name).file_name().and_then(|value| value.to_str()) == Some(name);
        if is_plain {
            let candidate = dir.join(name);
            if candidate.is_file() && is_safetensors_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    first_safetensors_path(dir)
}

/// Returns the detected family for a base model directory or file.
///
/// Detection strategy, in priority order:
/// 1. Diffusers `model_index.json` `_class_name` — canonical for diffusers
///    snapshots and the most reliable signal.
/// 2. Safetensors header architecture detection via [`detect_lora_family`] —
///    the architecture-prefix substrings (e.g. `transformer.transformer_blocks.`)
///    appear in base models as well as LoRAs, so the same detector usefully
///    classifies single-file diffusers checkpoints.
///
/// Returns `Ok(None)` when no confident signal is available — callers should
/// treat that as "unassociated" rather than as an error.
pub fn detect_model_family(path: &Path) -> Result<Option<String>, SafetensorsHeaderError> {
    if path.is_dir() {
        if let Some(family) = read_diffusers_model_index_family(path) {
            return Ok(Some(family));
        }
    } else if path.is_file()
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("model_index.json"))
    {
        if let Some(parent) = path.parent() {
            if let Some(family) = read_diffusers_model_index_family(parent) {
                return Ok(Some(family));
            }
        }
    }
    let Some(safetensors_path) = first_safetensors_path(path) else {
        return Ok(None);
    };
    let header = read_safetensors_header(&safetensors_path)?;
    Ok(detect_lora_family(&header))
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FamilyMismatch {
    pub supplied: String,
    pub detected: String,
}

/// Applies the shared import policy for detected architecture families.
///
/// A confident detector result rejects a conflicting user-supplied family; a
/// missing detector result keeps the supplied family, if any.
pub fn reconcile_detected_family(
    supplied: Option<String>,
    detected: Option<String>,
) -> Result<Option<String>, FamilyMismatch> {
    match (supplied, detected) {
        (Some(supplied), Some(detected)) => {
            // Compare on the canonical family token, not the raw string, so spelling
            // variants of one family agree: a user-supplied `krea-2` (the UI's hyphen
            // form) against a detected `krea_2` (the catalog/trainer token), or
            // ai-toolkit's separator-less `krea2`. Store the canonical token so the
            // recorded family matches `loraCompatibility.families` exactly.
            let canonical_supplied = canonical_lora_family(&supplied);
            let canonical_detected = canonical_lora_family(&detected);
            // A detected *base architecture* family also satisfies a declared, more specific
            // model family built on that architecture. A Chroma checkpoint (`chroma`) with no
            // metadata detects as `flux`; FLUX.2 [klein] / [dev] (`flux2-klein` / `flux2-dev`)
            // carry no variant signature and detect as the base `flux2`. Without this, every
            // legitimate Chroma / klein / dev download or import fails as a false mismatch.
            // Keep the *declared* family — it is the model's true identity (a klein model must
            // stay `flux2-klein`, a Chroma LoRA `chroma`), and the base is only what the
            // detector falls back to when the variant shares the base's tensor layout.
            if canonical_supplied == canonical_detected
                || detected_base_architecture_satisfies_declared(
                    &canonical_supplied,
                    &canonical_detected,
                )
            {
                Ok(Some(canonical_supplied))
            } else {
                Err(FamilyMismatch { supplied, detected })
            }
        }
        (None, Some(detected)) => Ok(Some(canonical_lora_family(&detected))),
        (Some(supplied), None) => Ok(Some(canonical_lora_family(&supplied))),
        (None, None) => Ok(None),
    }
}

/// Adds model-manifest defaults derived from the imported model type/family.
/// Existing author-supplied fields are preserved.
pub fn apply_model_manifest_defaults(
    entry: &mut Map<String, Value>,
    model_type: &str,
    family: Option<&str>,
) {
    entry
        .entry("downloads".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    entry
        .entry("defaults".to_owned())
        .or_insert_with(|| json!({}));
    entry
        .entry("limits".to_owned())
        .or_insert_with(|| json!({}));
    entry.entry("ui".to_owned()).or_insert_with(|| json!({}));

    let Some(family) = family
        .map(normalize_model_family)
        .filter(|value| !value.is_empty())
    else {
        entry
            .entry("loraCompatibility".to_owned())
            .or_insert_with(|| json!({}));
        return;
    };

    if let Some(adapter) = model_adapter_for_family(&family) {
        entry
            .entry("adapter".to_owned())
            .or_insert_with(|| Value::String(adapter.to_owned()));
    }

    let capabilities = model_capabilities_for_type_and_family(model_type, &family);
    if !capabilities.is_empty() {
        entry.entry("capabilities".to_owned()).or_insert_with(|| {
            Value::Array(
                capabilities
                    .into_iter()
                    .map(|capability| Value::String(capability.to_owned()))
                    .collect(),
            )
        });
    }

    let compatibility = entry
        .entry("loraCompatibility".to_owned())
        .or_insert_with(|| json!({}));
    if let Some(object) = compatibility.as_object_mut() {
        object
            .entry("families".to_owned())
            .or_insert_with(|| json!([family]));
    }

    apply_family_studio_surface_defaults(entry, &family);
}

/// Stamp the resolution / img2img Studio surface an imported model needs to match its builtin sibling
/// (sc-14071, epic 14015). An imported entry ships empty `limits` / `defaults` / `ui`, so without this
/// the Studio resolution picker falls back to its bare 4-option list (`selectedModel.limits.resolutions`
/// is empty) and never offers img2img. Every field is `or_insert_with`, so an author-supplied value
/// always wins — this only fills the gaps the import path leaves. Family-scoped: only families with a
/// stamped surface below are touched; every other family is left exactly as before.
///
/// **`krea-2`** mirrors the builtin `krea_2_turbo` entry (config/manifests/builtin.models.jsonc): the
/// 15-bucket ÷16-aligned ≤2048² resolution list, the 1024² default, `mlx.minMemoryGb` 48 (the ≤1536²
/// visibility floor + the >1536² memory-gate anchor, sc-13959 — an empty `mlx` block would offer 2048²
/// unconditionally on any Mac), the `ui.img2img` toggle + `img2imgStrength` slider (reference-guided
/// latent-init, resolved by the worker's `resolve_img2img_init_generic`), and — for the MLX Kontext
/// edit surface (sc-14119) — the `ui.editReferences` optional-second-image slot the Edit tab renders
/// (the `edit_image` capability is stamped by `model_capabilities_for_type_and_family`). The
/// `editReferences` copy mirrors the builtin Turbo entry (image 1 required + image 2 optional, fixed
/// order).
///
/// **`mage-flow`** (sc-15036) mirrors the builtin `mage_flow_base` entry: its 13-bucket ÷16-aligned
/// 512–2048 resolution ladder, the 1024² default, the binding `requiresDimensionsMultipleOf: 16`
/// stride (NR-MMDiT patch=1 over a 16×-downsampled latent — without it the free Width/Height
/// override would offer sizes the engine rejects), and the 30-step / CFG-5 undistilled Base
/// sampling defaults a fine-tune inherits. It deliberately declares **no** `mlx.quantize`: a
/// fine-tuned checkpoint is DENSE bf16 and the builtin q4/q8 tiers are pre-quantized artifacts with
/// an 8-bit floor on `norm_out.linear` and the text-encoder decoder layers (sc-15071) that a naive
/// load-time quantize does not reproduce — so packing one at load would render a tiled texture
/// rather than the prompt. It DOES declare `mlx.minMemoryGb`, which the builtin does not, precisely
/// because loading dense is what makes the sc-13959 >1536² gate matter here. No `ui.img2img` and no
/// `ui.editReferences`: the non-edit Mage variants advertise no conditioning at all.
fn apply_family_studio_surface_defaults(entry: &mut Map<String, Value>, family: &str) {
    match family {
        "krea-2" => apply_krea_2_studio_surface_defaults(entry),
        "mage-flow" => apply_mage_flow_studio_surface_defaults(entry),
        _ => {}
    }
}

/// The `mage-flow` Studio surface — see [`apply_family_studio_surface_defaults`].
fn apply_mage_flow_studio_surface_defaults(entry: &mut Map<String, Value>) {
    if let Some(limits) = entry
        .entry("limits".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        limits.entry("resolutions".to_owned()).or_insert_with(|| {
            json!([
                "512x512",
                "768x768",
                "1024x1024",
                "1536x1536",
                "2048x2048",
                "1280x720",
                "1536x1024",
                "2048x1024",
                "2048x512",
                "720x1280",
                "1024x1536",
                "1024x2048",
                "512x2048"
            ])
        });
        limits
            .entry("count".to_owned())
            .or_insert_with(|| json!([1, 2, 4]));
        // Binding, not advisory: both sides must be a multiple of 16 or the engine refuses the
        // geometry. An imported entry without it would let the Width/Height override offer sizes
        // that fail at generate time.
        limits
            .entry("requiresDimensionsMultipleOf".to_owned())
            .or_insert_with(|| json!(16));
    }
    if let Some(defaults) = entry
        .entry("defaults".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        defaults
            .entry("resolution".to_owned())
            .or_insert_with(|| json!("1024x1024"));
        // The undistilled Base regime a fine-tune inherits (the distilled Turbo checkpoints are a
        // different starting point and are not a training target).
        defaults
            .entry("steps".to_owned())
            .or_insert_with(|| json!(30));
        defaults
            .entry("guidanceScale".to_owned())
            .or_insert_with(|| json!(5));
        defaults
            .entry("count".to_owned())
            .or_insert_with(|| json!(1));
    }
    if let Some(mlx) = entry
        .entry("mlx".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        // The >1536² memory-gate anchor (sc-13959). Load-bearing for a fine-tune in a way it is not
        // for the builtin: `mage_flow_base` declares no `mlx.minMemoryGb` but defaults to the
        // pre-quantized `q4` tier (7.87 GB measured peak), whereas a fine-tuned checkpoint is DENSE
        // bf16 by construction — ~2.4x the resident weights — and this entry deliberately declares
        // no `mlx.quantize` (the pre-quantized tiers carry an 8-bit floor a load-time quantize does
        // not reproduce, sc-15071). With no floor, `resolutionMemory.js` has no basis to predict a
        // peak and offers 2048² unconditionally on ANY Mac — and an MLX overcommit is an
        // uncatchable SIGKILL, not a recoverable error.
        //
        // 20 GB, derived from the builtin manifest's own numbers rather than guessed:
        //   install totals  q4 7.002 · q8 9.450 · bf16 17.464 GB
        //   measured MLX unified peaks (sc-15071 note)  q4 7.87 · q8 10.11 GB
        //   peak/install    1.124 · 1.070  =>  mean 1.097
        //   bf16 predicted  17.464 x 1.097 ~= 19.2 GB
        // cross-checked against the manifest's MEASURED `candle.vramGbByTier.bf16` of 20.41 GB.
        // Rounded UP to 20: over-estimating hides a borderline bucket (safe), under-estimating
        // offers one that SIGKILLs. This is an extrapolation, not an on-device measurement — the
        // sc-15038 precedent (calibrate via `mlx::get_peak_memory`) applies to it too.
        //
        // Effect: everything at/below the 2.36 MP baseline stays unconditionally offered (including
        // 2048x1024, which at 2.10 MP is below it); only true 2048² is gated, needing a ~49 GB host
        // to clear the 0.9 headroom fraction.
        mlx.entry("minMemoryGb".to_owned())
            .or_insert_with(|| json!(20));
    }
}

/// The `krea-2` Studio surface — see [`apply_family_studio_surface_defaults`].
fn apply_krea_2_studio_surface_defaults(entry: &mut Map<String, Value>) {
    if let Some(limits) = entry
        .entry("limits".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        limits.entry("resolutions".to_owned()).or_insert_with(|| {
            json!([
                "1024x1024",
                "768x1024",
                "1024x768",
                "1280x720",
                "720x1280",
                "1216x832",
                "832x1216",
                "1152x896",
                "896x1152",
                "1536x1536",
                "2048x1152",
                "1152x2048",
                "2048x1408",
                "1408x2048",
                "2048x2048"
            ])
        });
    }
    if let Some(defaults) = entry
        .entry("defaults".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        defaults
            .entry("resolution".to_owned())
            .or_insert_with(|| json!("1024x1024"));
    }
    if let Some(mlx) = entry
        .entry("mlx".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        mlx.entry("minMemoryGb".to_owned())
            .or_insert_with(|| json!(48));
    }
    if let Some(ui) = entry
        .entry("ui".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        ui.entry("img2img".to_owned())
            .or_insert_with(|| json!(true));
        ui.entry("img2imgStrength".to_owned()).or_insert_with(|| {
            json!({
                "label": "Reference strength",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05
            })
        });
        // Kontext edit second-image slot (sc-14119, mirrors the builtin krea_2_turbo `ui` block): the
        // `krea2_identity_edit` edit optionally takes a SECOND source (image 1 required + image 2
        // optional, FIXED order). Presence of this object makes the Studio render the optional second
        // slot in Edit mode; the worker + engine (`krea_2_turbo_edit` MultiReference) cap at two and
        // preserve order. Ignored outside `edit_image`.
        ui.entry("editReferences".to_owned()).or_insert_with(|| {
            json!({
                "secondaryLabel": "Image 2 (optional)",
                "secondaryHint": "Optional — a second image combined with Image 1 in a fixed order (Image 1, then Image 2). In your instruction, refer to each subject by position (\"the person in image 1 / image 2\") or by description (\"the woman in the green jacket\") — either works."
            })
        });
    }
}

pub fn model_adapter_for_family(family: &str) -> Option<&'static str> {
    match normalize_model_family(family).as_str() {
        "z-image" => Some("z_image_diffusers"),
        "qwen-image" => Some("qwen_image"),
        "lens" => Some("lens_turbo"),
        "sensenova-u1" => Some("sensenova_u1"),
        "flux" => Some("flux_diffusers"),
        "chroma" => Some("chroma_diffusers"),
        "kolors" => Some("kolors_diffusers"),
        "sdxl" => Some("sdxl_diffusers"),
        // Mage-Flow is macOS-only native MLX (epic 14034): there is no Torch/diffusers adapter.
        // Like bernini / sd3 this label is recorded in recipe/lineage only — the job is MLX-routed
        // by engine id (builtins) or by family (a full base fine-tune, sc-15036), never
        // instantiated through a Torch adapter. Matches the builtin entries' `"adapter"`.
        "mage-flow" => Some("mlx_mage"),
        // SD3 / SD3.5 is the native-MLX port (epic 7841); there is no Torch/diffusers
        // adapter wired in SceneWorks. This label is recorded in recipe/lineage only —
        // the job is MLX-routed by engine id, never instantiated through a Torch adapter
        // (mirrors bernini / scail2). sc-7874 declares the LoRA family; engine wiring is
        // a later slice.
        "sd3" => Some("sd3"),
        "ltx-video" => Some("ltx_video"),
        "wan-video" => Some("wan_video"),
        "svd" => Some("svd_video"),
        // Bernini is macOS-only native MLX (epic 4699): there is no Torch/diffusers
        // adapter. This label is recorded in recipe/lineage only; on Mac the job is
        // MLX-routed by engine id, never instantiated through a Torch adapter.
        "bernini" => Some("bernini"),
        // SCAIL-2 (epic 5439) is likewise macOS-only native MLX (engine id
        // "scail2_14b"); no Torch/diffusers adapter. Lineage label only.
        "scail2" => Some("scail2"),
        // Anima (epic 10512) is macOS-only native MLX (Cosmos-Predict2 DiT + AnimaTextConditioner);
        // there is no Torch/diffusers adapter — the job is MLX-routed by engine id (`anima_base` /
        // `anima_aesthetic` / `anima_turbo`), never instantiated through a Torch adapter. Lineage
        // label only (mirrors sd3 / bernini / scail2).
        "anima" => Some("anima"),
        // Krea 2 (epic 14015): imported single-file Krea 2 checkpoints reuse the same MLX Krea
        // adapter the builtin krea_2 catalog entries declare (`mlx_krea`), routed to the Krea MLX
        // engine by family (sc-14108). The match scrutinee is the normalized family, so the arm keys
        // on the hyphen form `krea-2` (`normalize_model_family("krea_2")`). Builtin krea_2 entries
        // (`krea_2_turbo` / `krea_2_raw`) declare `adapter` explicitly, so
        // `apply_model_manifest_defaults` (`or_insert_with`) never overrides them — this default only
        // stamps user/imported krea_2 models that omit it.
        "krea-2" => Some("mlx_krea"),
        _ => None,
    }
}

pub fn model_capabilities_for_type_and_family(model_type: &str, family: &str) -> Vec<&'static str> {
    match (
        model_type.trim().to_ascii_lowercase().as_str(),
        normalize_model_family(family).as_str(),
    ) {
        // No `character_image`: the native Z-Image lane has no IP-Adapter/reference-conditioning
        // code, and no Z-Image IP-Adapter exists upstream
        // (sc-2005). Custom z-image models that override capabilities can still
        // re-declare it, but the family default shouldn't claim what it can't do.
        ("image", "z-image") => vec!["text_to_image", "style_variations"],
        ("image", "qwen-image") => vec!["text_to_image", "style_variations"],
        ("image", "lens") => vec!["text_to_image", "style_variations"],
        ("image", "sensenova-u1") => vec!["text_to_image", "edit_image", "vqa", "interleave"],
        ("image", "flux") => vec!["text_to_image", "style_variations"],
        ("image", "chroma") => vec!["text_to_image", "style_variations"],
        ("image", "kolors") => vec!["text_to_image", "character_image", "style_variations"],
        ("image", "sdxl") => vec!["text_to_image", "edit_image", "style_variations"],
        // SD3 / SD3.5 (epic 7841, native MLX). Text-to-image flow-matching MMDiT;
        // img2img/inpaint are exposed by the diffusers pipelines but are not wired in
        // SceneWorks yet, so the family default advertises only what the native port
        // serves today (sc-7874 declares the LoRA-compatibility family).
        ("image", "sd3") => vec!["text_to_image", "style_variations"],
        // Anima (epic 10512, native MLX) is an anime text-to-image DiT. LoRA-capable
        // (`supports_lora`/`supports_lokr`, sc-10521); no edit/inpaint or reference/IP-Adapter
        // surface, so the family default advertises only t2i + style variations (like z-image /
        // qwen-image / lens / flux).
        ("image", "anima") => vec!["text_to_image", "style_variations"],
        // Krea 2 (epic 14015): imported single-file Krea 2 checkpoints. The KreaImported lane serves
        // text-to-image (sc-14018), img2img (sc-14071 — reference-guided latent-init, exposed via the
        // `ui.img2img` toggle stamped by `apply_family_studio_surface_defaults`, NOT a capabilities
        // value: z-image owns "image_to_image" for its edit-mode img2img), AND the Kontext instruction
        // `edit_image` surface on the MLX backend (sc-14119 — the source rides as in-context tokens +
        // grounds the Qwen3-VL vision tower, driven by the `krea2_identity_edit` LoRA). `edit_image`
        // pairs with the `ui.editReferences` slot `apply_family_studio_surface_defaults` also stamps.
        // Still NOT `character_image` (no IP-Adapter/identity surface on the bare transformer). The
        // match keys on the normalized hyphen form `krea-2` (`normalize_model_family("krea_2")`). This
        // default only affects imported/user krea_2 models; the builtin krea_2 entries declare their
        // own `capabilities` explicitly, so `apply_model_manifest_defaults` never changes them.
        ("image", "krea-2") => vec!["text_to_image", "edit_image", "style_variations"],
        // Mage-Flow (epic 14034). The non-edit variants — the only ones that are a training target,
        // and therefore the only ones a full base fine-tune (sc-15036) can be derived from —
        // advertise NO conditioning on their descriptor: no reference, no multi-reference, no edit.
        // So text-to-image plus style variations, and deliberately not `edit_image` (that needs an
        // `mage_flow_edit*` checkpoint) nor `character_image` (no identity surface).
        ("image", "mage-flow") => vec!["text_to_image", "style_variations"],
        // Bernini still-image companion (epic 4699 / sc-5424): the same `Modality::Both`
        // engine the video `bernini` family uses, but the image-typed catalog id
        // (`bernini_image`) exposes only the still tasks — t2i (text→image) and i2i
        // (`edit_image`, the source-image edit via `Conditioning::Reference`). No
        // `character_image`/`style_variations` (no IP-Adapter/style surface) and no LoRA
        // (the descriptor reports `supports_lora: false`).
        ("image", "bernini") => vec!["text_to_image", "edit_image"],
        ("video", "ltx-video") => vec![
            "image_to_video",
            "text_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
        ],
        ("video", "wan-video") => vec![
            "image_to_video",
            "text_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "replace_person",
        ],
        // Stable Video Diffusion is image-conditioned only (no text prompt) and
        // does not support the timeline/replacement modes.
        ("video", "svd") => vec!["image_to_video"],
        // Bernini (epic 4699) is a Wan2.2-T2V-A14B renderer + Qwen2.5-VL semantic
        // planner whose engine descriptor is `Modality::Both` with conditioning
        // `[Reference, MultiReference, VideoClip]`. The video task surface maps onto
        // SceneWorks modes (sc-4703 / sc-5425): `text_to_video` (t2v), `video_to_video`
        // (v2v — a source clip edit), `reference_to_video` (r2v — subject reference
        // images → video), `reference_video_to_video` (rv2v — source clip + reference
        // images), `multi_video_to_video` (mv2v — multiple source clips), and `ads2v`
        // (source video + reference video + reference images). Bernini has no classic
        // still-image-to-video (its renderer is T2V, not I2V). The t2i/i2i image
        // companion is a separate image-typed catalog id (tracked under epic 4699), not
        // declared here.
        ("video", "bernini") => vec![
            "text_to_video",
            "video_to_video",
            "reference_to_video",
            "reference_video_to_video",
            "multi_video_to_video",
            "ads2v",
        ],
        // SCAIL-2 (epic 5439) is a Wan2.1-14B I2V end-to-end character-animation
        // engine: a reference character image + a driving video → an animated clip.
        // Its engine descriptor is `Modality::Video` with conditioning
        // `[Reference, Mask, MultiReference, ControlClip]`. It serves the standalone
        // `animate_character` mode (sc-5448, the worker paints the color-coded masks
        // from native SAM3) and cross-identity `replace_person` (sc-5452, the same
        // engine with replace_flag=true, as the higher-quality backend behind the
        // existing person-track replacement pipeline). LoRA is sc-5451 — not declared
        // until wired. Multi-character (paired ref+mask) awaits the engine
        // request-contract extension (sc-5583).
        ("video", "scail2") => vec!["animate_character", "replace_person"],
        _ => Vec::new(),
    }
}

pub(crate) fn normalize_model_family(family: &str) -> String {
    let normalized = family.trim().to_ascii_lowercase().replace('_', "-");
    // Collapse spelling variants the `_`→`-` step alone can't unify. ostris
    // ai-toolkit bakes Krea 2's base id into trained files as the separator-less
    // `krea2` (`ss_base_model_version: "krea2"`), which is the same family as
    // `krea-2` / `krea_2`. Kept an explicit alias, not a blind separator-strip —
    // a blind strip could merge unrelated families.
    match normalized.as_str() {
        "krea2" => "krea-2".to_owned(),
        _ => normalized,
    }
}

/// The canonical *stored* LoRA-family token for any spelling of a family.
///
/// Detection, the catalog manifests, and the trainers agree on one token per
/// family; only Krea 2 has variants in the wild — `krea2` (ai-toolkit's
/// `ss_base_model_version`), `krea-2` (the UI's hyphen form), and `krea_2` (the
/// catalog token) — all of which resolve to the catalog token `krea_2`. Every
/// other family already stores its `normalize_model_family` form, so this returns
/// that (lower-cased, `_`→`-`) unchanged for them.
pub fn canonical_lora_family(family: &str) -> String {
    let normalized = normalize_model_family(family);
    match normalized.as_str() {
        "krea-2" => "krea_2".to_owned(),
        _ => normalized,
    }
}

/// True when a *detected* base-architecture family legitimately satisfies a declared,
/// more specific model family — i.e. the declared family's weights share that base
/// architecture's tensor layout, so [`detect_model_family`] can only report the base.
/// These are exactly the families that also load the base architecture's LoRAs, so this
/// consults the same [`extra_compatible_lora_families`] registry to stay in lockstep:
///
/// * Chroma (`chroma`) is FLUX.1-schnell-derived; a metadata-less Chroma checkpoint has
///   byte-for-byte the same tensor keys as FLUX.1, so key detection reports `flux`.
/// * FLUX.2 [klein] / [dev] (`flux2-klein` / `flux2-dev`) carry no klein/dev-specific
///   tensor signature, so detection reports the base `flux2`.
///
/// The reason such a model accepts the base family's LoRAs is precisely that it *is*
/// that base architecture, which is why the accept-detection and load-LoRA relations
/// coincide. Both arguments must already be canonical (see [`canonical_lora_family`]).
/// Directional on purpose: only a detected *base* satisfies a declared *variant*, never
/// the reverse, so a genuinely wrong download/import is still a confident mismatch.
fn detected_base_architecture_satisfies_declared(declared: &str, detected: &str) -> bool {
    extra_compatible_lora_families(declared).contains(&detected)
}

fn read_diffusers_model_index_family(dir: &Path) -> Option<String> {
    let index_path = dir.join("model_index.json");
    let bytes = fs::read(&index_path).ok()?;
    let index: Value = serde_json::from_slice(&bytes).ok()?;
    let class_name = index.get("_class_name").and_then(Value::as_str)?;
    // Mage-Flow's MMDiT tensor names overlap Qwen-Image, so the directory classifier must use
    // the exact pipeline + transformer contract rather than accepting a class-name lookalike.
    if class_name.eq_ignore_ascii_case("MageFlowPipeline") {
        let config: Value =
            serde_json::from_slice(&fs::read(dir.join("transformer").join("config.json")).ok()?)
                .ok()?;
        let is_mage = config
            .get("_class_name")
            .and_then(Value::as_str)
            .is_some_and(|value| value == "MageFlow")
            && config.get("in_channels").and_then(Value::as_u64) == Some(128)
            && config.get("hidden_size").and_then(Value::as_u64) == Some(3072)
            && config.get("depth").and_then(Value::as_u64) == Some(12);
        return is_mage.then(|| "mage-flow".to_owned());
    }
    diffusers_class_name_to_family(class_name)
}

/// Maps a diffusers pipeline `_class_name` to a SceneWorks family. The map is
/// intentionally conservative — only return a family when the mapping is
/// unambiguous, and otherwise let the caller treat the model as unassociated.
pub fn diffusers_class_name_to_family(class_name: &str) -> Option<String> {
    let normalized = class_name.trim();
    let lower = normalized.to_ascii_lowercase();
    match lower.as_str() {
        "zimagepipeline"
        | "zimageimg2imgpipeline"
        | "zimageturbopipeline"
        | "zimageturboimg2imgpipeline" => Some("z-image".to_owned()),
        "qwenimagepipeline" | "qwenimageimg2imgpipeline" | "qwenimageeditpipeline" => {
            Some("qwen-image".to_owned())
        }
        "lenspipeline" => Some("lens".to_owned()),
        // Directory detection additionally validates transformer/config.json before accepting this
        // family; this mapping is also used by callers that already hold a trusted class name.
        "mageflowpipeline" => Some("mage-flow".to_owned()),
        // Anima (epic 10512). Its diffusers modular pipeline (`AnimaModularPipeline`) is the
        // `model_index.json` `_class_name` a diffusers-format Anima export would carry.
        "animamodularpipeline" => Some("anima".to_owned()),
        "fluxpipeline" | "fluximg2imgpipeline" | "fluxinpaintpipeline" => Some("flux".to_owned()),
        "chromapipeline" | "chromaimg2imgpipeline" => Some("chroma".to_owned()),
        "kolorspipeline" | "kolorsimg2imgpipeline" => Some("kolors".to_owned()),
        "wanpipeline" | "wani2vpipeline" | "wantext2videopipeline" => Some("wan-video".to_owned()),
        "ltxpipeline" | "ltxvideopipeline" | "ltximagetovideopipeline" => {
            Some("ltx-video".to_owned())
        }
        "stablediffusion3pipeline"
        | "stablediffusion3img2imgpipeline"
        | "stablediffusion3inpaintpipeline" => Some("sd3".to_owned()),
        "stablediffusionpipeline" | "stablediffusionimg2imgpipeline" => Some("sd1.5".to_owned()),
        "stablediffusionxlpipeline" | "stablediffusionxlimg2imgpipeline" => Some("sdxl".to_owned()),
        _ => None,
    }
}

/// Returns the detected LoRA architecture family or `None` if the header
/// is ambiguous, empty, or matches no known signature with confidence.
pub fn detect_lora_family(header: &Value) -> Option<String> {
    if let Some(family) = detect_metadata_family(header) {
        return Some(family);
    }
    let keys = collect_tensor_keys(header);
    if keys.is_empty() {
        return None;
    }
    // Some families expose a tensor-name segment that appears in no other family
    // we detect; one such key is enough to identify them, ahead of (and exempt
    // from) the bucket scorer's `MIN_KEY_MATCHES` floor. See [`detect_unique_key_family`].
    if let Some(family) = detect_unique_key_family(&keys) {
        return Some(family);
    }
    let bucket = detect_bucket(&keys)?;
    match bucket {
        Bucket::WanVideo => Some("wan-video".to_owned()),
        Bucket::Flux => Some("flux".to_owned()),
        Bucket::Flux2 => Some("flux2".to_owned()),
        Bucket::LtxVideo => Some("ltx-video".to_owned()),
        Bucket::Sd3 => Some("sd3".to_owned()),
        Bucket::Ideogram => Some("ideogram".to_owned()),
        Bucket::Anima => Some("anima".to_owned()),
        Bucket::Sdxl => Some("sdxl".to_owned()),
        Bucket::Sd15 => Some("sd1.5".to_owned()),
        Bucket::MmDit => disambiguate_mm_dit(&keys),
    }
}

/// Identifies a family from a single architecture-unique tensor-name segment,
/// bypassing the `MIN_KEY_MATCHES` marker-count floor the bucket scorer enforces.
///
/// The floor exists so an ambiguous handful of generic keys can never win, but it
/// also blocks sparse adapters that touch only a module or two — e.g. a Krea 2
/// `text_fusion.projector` scale LoRA, which ships just two tensors and so can
/// never reach four marker hits. When a key segment is unambiguous across the
/// entire signature table, a single occurrence is sufficient evidence, so those
/// cases are handled here first.
///
/// Kept deliberately narrow: only segments that appear in *no other* family we
/// detect belong here, because a confident-but-wrong family is grounds to reject
/// an import (see the module docs), which is worse than an inconclusive `None`.
fn detect_unique_key_family(keys: &[String]) -> Option<String> {
    // Anima (epic 10512). The Cosmos-Predict2 DiT bundles an `AnimaTextConditioner` under an
    // `llm_adapter` sub-module — a name no other detected family uses. The **turbo** distillation LoRA
    // (`anima-turbo-lora-v0.2`) trains 60 `diffusion_model.llm_adapter.blocks.<n>.…` targets alongside
    // the DiT; a single such key unambiguously identifies Anima, ahead of (and exempt from) the
    // `MIN_KEY_MATCHES` floor — so even a sparse LoKr that touches only the conditioner is detected
    // (the Ideogram-LoKr precedent, where a below-floor adapter went undetected). The DiT-only style
    // LoRA carries no `llm_adapter` key and is instead caught by the `Bucket::Anima` signature below on
    // its Cosmos adaLN-modulation markers — so both official LoRA shapes classify as `anima`, and a
    // signature that *required* `llm_adapter` (which the style LoRA lacks) is correctly avoided.
    //
    // Both the diffusers-dotted (`…llm_adapter.blocks…`) and kohya-flattened (`…_llm_adapter_blocks…`)
    // spellings are matched. `llm_adapter` appears in no other family's LoRA keys.
    if keys
        .iter()
        .any(|key| key.contains("llm_adapter.") || key.contains("_llm_adapter_"))
    {
        return Some("anima".to_owned());
    }
    // Krea 2 (epic 7565). Its DiT carries a `text_fusion` Qwen3-VL-layer aggregator
    // and a gated single-stream attention whose projection is the leaf Linear
    // `attn.to_gate` — names that appear in no other family's LoRA keys (dual-stream
    // MMDiT/Flux/SD3/Wan have none). Both the diffusers dotted form
    // (`...text_fusion.projector...`, `...attn.to_gate.lora_A...`) and the
    // kohya/flattened underscore form are matched. The family label is `krea_2`
    // (underscore) to match the catalog's `loraCompatibility.families` and the
    // `krea_2_raw_lora` trainer output exactly, since import-time reconciliation
    // compares the family string verbatim.
    //
    // The `to_gate` markers require the trailing module-boundary separator (a `.` for
    // the diffusers leaf, so the LoRA sub-weight `lora_A`/`lora_down` follows). Without
    // it, the substring would also match LTX-2's cross-modal gating tensors
    // (`...audio_to_video_attn.to_gate_logits...`, `...attn1.to_gate_logits...`), which
    // contain `attn.to_gate` as a non-boundary substring and would otherwise be
    // mis-detected as krea_2 and rejected from every LTX model.
    if keys.iter().any(|key| {
        key.contains("text_fusion")
            || key.contains("attn.to_gate.")
            || key.contains("_attn_to_gate.")
    }) {
        return Some("krea_2".to_owned());
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    WanVideo,
    Flux,
    Flux2,
    LtxVideo,
    /// Stable Diffusion 3 / 3.5 (Large, Large Turbo, Medium). A dual-stream
    /// MMDiT like Qwen-Image / Z-Image, but distinguished by its
    /// `ff`/`ff_context` feedforward + `*_context` norms + `context_embedder`
    /// naming (it never uses the `img_mlp`/`txt_mlp` dual-MLP names) and its
    /// optional dual-attention `attn2` blocks (sd3.5 only). Bucketed separately
    /// so SD3.5 community LoRAs are positively recognized rather than falling
    /// into the Qwen/Z-Image block-count disambiguation (sc-7874).
    Sd3,
    /// Ideogram 4 (epic 4725): a single-stream 34-layer flow-matching DiT with a
    /// unique `diffusion_model.layers.<n>.` block prefix and fused `attention.qkv`,
    /// `adaln_modulation`, and `feed_forward.w{1,2,3}` modules — shared with no other
    /// family we detect. Metadata (`ss_base_model_version: ideogram4`) is the primary
    /// signal; this bucket catches metadata-less ComfyUI-style exports.
    Ideogram,
    /// Anima (epic 10512): the Cosmos-Predict2 DiT anime T2I model. Its LoRAs prefix every DiT block
    /// with `diffusion_model.blocks.<n>.` (shared with native/ComfyUI Wan) but carry the
    /// Cosmos-specific `adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}` down/up modulation pairs
    /// (present in BOTH official shapes — the DiT-only style LoRA and the DiT+conditioner turbo LoRA)
    /// that no other family uses. The turbo shape additionally trains `llm_adapter.*` conditioner
    /// targets (caught earlier by `detect_unique_key_family`); this bucket keys on the adaLN markers so
    /// the DiT-only shape — which has NO `llm_adapter` key — still classifies as `anima`.
    Anima,
    MmDit,
    Sdxl,
    Sd15,
}

struct BucketSignature {
    bucket: Bucket,
    /// Every group in this list must be satisfied: at least one tensor key
    /// must contain at least one substring from the inner slice. A signature
    /// with any unmet group does not apply (score = 0). Use this to encode
    /// AND-of-OR conjunctions like "must have both `lora_te1_` and
    /// `lora_te2_` keys" for SDXL.
    require_all_of: &'static [&'static [&'static str]],
    /// If any substring here is present in any tensor key, the signature is
    /// disqualified regardless of marker score.
    disqualifiers: &'static [&'static str],
    /// Substrings that count toward the score when present in a tensor key.
    markers: &'static [&'static str],
}

/// Every block-index prefix a `blocks.<n>.` family spells its transformer blocks with:
/// diffusers (`transformer.blocks.`), native/ComfyUI (`diffusion_model.blocks.`), and the
/// kohya/musubi flattened forms (`diffusion_model_blocks_`, `lora_unet_blocks_`).
///
/// Wan **and** Anima (Cosmos-Predict2) both use this prefix set, so the prefix alone proves
/// nothing — [`COSMOS_LEAF_MODULE_MARKERS`] and [`COSMOS_ADALN_MARKERS`] are what separate
/// them. Shared by the Anima signatures so an Anima LoRA is recognized in whichever export
/// spelling its trainer produced, matching the prefixes the Wan signatures already accept.
const BLOCK_PREFIX_MARKERS: &[&str] = &[
    "diffusion_model.blocks.",
    "diffusion_model_blocks_",
    "transformer.blocks.",
    "lora_unet_blocks_",
];

/// Anima's Cosmos-Predict2 adaLN-modulation module names (`adaln_modulation_{self_attn,
/// cross_attn,mlp}.{1,2}` down/up pairs), in the spelling shared by the dotted and
/// flattened exports. No other family we detect modulates this way — Wan has none, and
/// Ideogram's is a bare `.adaln_modulation` with no `_self_attn`/`_cross_attn`/`_mlp`
/// suffix — so one of these keys positively identifies Anima.
const COSMOS_ADALN_MARKERS: &[&str] = &[
    "adaln_modulation_self_attn",
    "adaln_modulation_cross_attn",
    "adaln_modulation_mlp",
];

/// Anima's Cosmos-Predict2 *leaf module* names inside a transformer block, in both the
/// diffusers-dotted and the kohya-flattened spelling.
///
/// This is the discriminator that does not depend on which targets a trainer chose. Cosmos
/// names its attention projections `{self_attn,cross_attn}.{q,k,v}_proj` + `output_proj`
/// and its feedforward `mlp.layer1` / `mlp.layer2`; Wan — the only other family sharing
/// [`BLOCK_PREFIX_MARKERS`] — names the very same modules `{self_attn,cross_attn}.{q,k,v,o}`
/// and `ffn.0` / `ffn.2`. A Wan block key therefore can never contain one of these
/// substrings, which is what makes them safe both as an Anima requirement and as a Wan
/// disqualifier.
///
/// The `_attn.`/`_attn_` lead-in matches `self_attn` and `cross_attn` in one substring. CLIP
/// text-encoder keys (`..._self_attn_q_proj`) also contain these, so every signature using
/// this list pairs it with a block prefix requirement and disqualifies the `lora_te*`
/// text-encoder prefixes.
const COSMOS_LEAF_MODULE_MARKERS: &[&str] = &[
    "_attn.q_proj",
    "_attn.k_proj",
    "_attn.v_proj",
    "_attn.output_proj",
    "_attn_q_proj",
    "_attn_k_proj",
    "_attn_v_proj",
    "_attn_output_proj",
    ".mlp.layer",
    "_mlp_layer",
];

const SIGNATURES: &[BucketSignature] = &[
    BucketSignature {
        bucket: Bucket::Flux,
        // Flux is the only architecture that ships both double-stream and
        // single-stream transformer blocks together. The single-block prefix
        // alone is enough to identify it. Kohya-style Flux LoRAs flatten the
        // same names into `lora_unet_single_blocks_...` tensor keys.
        require_all_of: &[&["single_transformer_blocks.", "single_blocks_"]],
        disqualifiers: &[],
        markers: &[
            "single_transformer_blocks.",
            "single_blocks_",
            "double_blocks.",
            "double_blocks_",
            "transformer_blocks.",
        ],
    },
    BucketSignature {
        bucket: Bucket::Flux,
        // XLabs / x-flux Flux LoRAs adapt only the double-stream blocks through an
        // attention-processor layout (`double_blocks.<n>.processor.{qkv,proj}_lora{1,2}`)
        // and ship no single-stream keys, so the primary Flux signature (which
        // requires a single-block marker) misses them. `double_blocks.` is unique to
        // Flux among the architectures we detect; pairing it with the x-flux
        // processor/lora naming keeps this tight. Disqualified whenever single-block
        // keys are present so it never co-scores with the primary Flux signature
        // (two same-bucket scores could otherwise trip the runner-up margin and
        // return ambiguous).
        require_all_of: &[
            &["double_blocks."],
            &["qkv_lora", "proj_lora", "processor."],
        ],
        disqualifiers: &["single_transformer_blocks.", "single_blocks_"],
        markers: &["double_blocks.", "processor.", "qkv_lora", "proj_lora"],
    },
    BucketSignature {
        bucket: Bucket::Flux2,
        // FLUX.2 [klein] native/ComfyUI LoRAs share the FLUX.1 `diffusion_model.`
        // prefix and the same double_blocks/single_blocks split, but FLUX.2 shares
        // modulation ACROSS all blocks via top-level `double_stream_modulation_img`
        // / `double_stream_modulation_txt` / `single_stream_modulation` tensors,
        // whereas FLUX.1 keeps per-block `img_mod`/`txt_mod`/`modulation` inside
        // each block index. Those shared-modulation keys are unique to FLUX.2 and
        // never appear in a FLUX.1 LoRA, so requiring one cleanly separates the two
        // (the primary Flux signature can't fire here anyway — FLUX.2 uses
        // `single_blocks.` with a dot, not the `single_blocks_` underscore form).
        require_all_of: &[&["double_stream_modulation_", "single_stream_modulation."]],
        disqualifiers: &["single_transformer_blocks."],
        markers: &[
            "double_stream_modulation_",
            "single_stream_modulation.",
            "double_blocks.",
            "single_blocks.",
            ".img_attn.",
            ".txt_attn.",
        ],
    },
    BucketSignature {
        bucket: Bucket::WanVideo,
        // Diffusers-format Wan LoRAs expose their blocks under
        // `transformer.blocks.<n>.` (not `transformer.transformer_blocks.`) and
        // use either the native `self_attn`/`cross_attn`/`ffn` module names or
        // the diffusers `attn1`/`attn2` names. The `transformer.blocks.` prefix
        // marker alone scores every key; discriminating against the MMDiT-style
        // key prefix keeps Wan separate from Qwen/Z-Image.
        require_all_of: &[&["transformer.blocks."]],
        disqualifiers: &[
            "transformer.transformer_blocks.",
            "single_transformer_blocks.",
            // Anima (Cosmos-Predict2) reuses this prefix too, so disqualify on its Cosmos
            // module naming — no real Wan LoRA carries it (SceneWorks#1670). See the
            // native/ComfyUI Wan signature below for the full rationale.
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
            "_attn.q_proj",
            "_attn.k_proj",
            "_attn.v_proj",
            "_attn.output_proj",
            ".mlp.layer",
        ],
        markers: &[
            "transformer.blocks.",
            ".self_attn.",
            ".cross_attn.",
            ".ffn.",
        ],
    },
    BucketSignature {
        bucket: Bucket::WanVideo,
        // ComfyUI / native Wan checkpoints and LoRAs prefix every block key with
        // `diffusion_model.blocks.<n>.` and keep the native `self_attn`/
        // `cross_attn`/`ffn` module names. Requiring both the prefix and a Wan
        // module marker keeps this from colliding with ComfyUI Flux
        // (`diffusion_model.double_blocks.` / `diffusion_model.single_blocks.`)
        // or LTX (`diffusion_model.transformer_blocks.`), none of which contain
        // the bare `.blocks.` segment.
        require_all_of: &[
            &["diffusion_model.blocks."],
            &[".self_attn.", ".cross_attn.", ".ffn."],
        ],
        disqualifiers: &[
            "transformer.transformer_blocks.",
            "single_transformer_blocks.",
            "double_blocks.",
            // Anima (Cosmos-Predict2) shares the `diffusion_model.blocks.` prefix and the
            // `.self_attn.`/`.cross_attn.` module markers, but is a separate `Bucket::Anima`
            // (sc-10521). Disqualify on its Cosmos adaLN-modulation naming — which no real Wan LoRA
            // ever carries — so a native Anima LoRA never co-scores here and trips the runner-up
            // margin against the Anima signature.
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
            // The adaLN keys above only exist when the trainer *targeted* the modulation layers.
            // An Anima LoRA that trains just attention (and/or the MLP) — the common shape — has
            // none, and used to land here and be reported as a confident `wan-video`, hard-rejecting
            // the import (SceneWorks#1670). The Cosmos leaf module names are present in every shape,
            // and Wan spells the same modules `.q.`/`.k.`/`.v.`/`.o.` + `ffn.0`/`ffn.2`, so they can
            // never appear in a Wan LoRA — see `COSMOS_LEAF_MODULE_MARKERS`.
            "_attn.q_proj",
            "_attn.k_proj",
            "_attn.v_proj",
            "_attn.output_proj",
            ".mlp.layer",
        ],
        markers: &[
            "diffusion_model.blocks.",
            ".self_attn.",
            ".cross_attn.",
            ".ffn.",
        ],
    },
    BucketSignature {
        bucket: Bucket::WanVideo,
        // Kohya / musubi-tuner Wan LoRAs flatten the native module path into
        // underscore-delimited keys: `lora_unet_blocks_<n>_self_attn_q...`.
        // `lora_unet_blocks_` is Wan-specific — SD/SDXL UNet keys are
        // `lora_unet_down_blocks_` / `lora_unet_up_blocks_` / `lora_unet_mid_block_`,
        // never the bare `lora_unet_blocks_`. Disqualifying the SD/SDXL
        // text-encoder prefixes (which Wan lacks) prevents any collision with the
        // Sd15/Sdxl signatures, whose text-encoder keys also contain `_self_attn_`.
        require_all_of: &[
            &["lora_unet_blocks_"],
            &["_self_attn_", "_cross_attn_", "_ffn_"],
        ],
        disqualifiers: &[
            "lora_te_",
            "lora_te1_",
            "lora_te2_",
            // A kohya/musubi-flattened Anima (Cosmos-Predict2) LoRA lands on this same
            // `lora_unet_blocks_<n>_{self,cross}_attn_...` layout, so the Anima discriminators apply
            // here exactly as they do to the native/ComfyUI Wan signature above (SceneWorks#1670):
            // the Cosmos adaLN modulation, and — for the attention/MLP-only shape that trains no
            // modulation at all — the Cosmos leaf module names, which no Wan LoRA can carry.
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
            "_attn_q_proj",
            "_attn_k_proj",
            "_attn_v_proj",
            "_attn_output_proj",
            "_mlp_layer",
        ],
        markers: &["lora_unet_blocks_", "_self_attn_", "_cross_attn_", "_ffn_"],
    },
    BucketSignature {
        bucket: Bucket::LtxVideo,
        // LTX-Video uses MMDiT-style `transformer_blocks.` keys but its attention
        // submodules are named `attn1` and `attn2` (the latter is the
        // cross-attention path). It does not use the dual-stream `img_mlp` /
        // `txt_mlp` naming that Qwen-Image / Z-Image expose.
        //
        // Two prefix forms are accepted: the diffusers `transformer.transformer_blocks.`
        // and the LTX-2 native / ComfyUI `diffusion_model.transformer_blocks.` export
        // (e.g. `ltx-2.3-22b-distilled-lora-*`). The `diffusion_model.` prefix is
        // shared with FLUX.2, but FLUX.2 nests `diffusion_model.double_blocks.` /
        // `single_blocks.` (never `transformer_blocks`), and pairing the prefix with
        // the required `.attn1.` + `.attn2.` submodules keeps this unambiguous. LTX-2
        // 2.3 additionally trains cross-modal audio attention
        // (`audio_to_video_attn.to_gate_logits`, etc.); those are not `add_q_proj`
        // joint-attention keys, so the disqualifiers below still hold.
        require_all_of: &[
            &[
                "transformer.transformer_blocks.",
                "diffusion_model.transformer_blocks.",
            ],
            &[".attn1."],
            &[".attn2."],
        ],
        disqualifiers: &[
            "single_transformer_blocks.",
            ".img_mlp.",
            ".txt_mlp.",
            "add_q_proj",
            "add_k_proj",
        ],
        markers: &[
            "transformer.transformer_blocks.",
            "diffusion_model.transformer_blocks.",
            ".attn1.",
            ".attn2.",
        ],
    },
    BucketSignature {
        bucket: Bucket::Ideogram,
        // Ideogram 4 (epic 4725) native / ComfyUI export. Its single-stream 34-layer
        // DiT prefixes every block key with `diffusion_model.layers.<n>.` — a segment
        // no other detected family uses (`blocks`, `transformer_blocks`, `double_blocks`,
        // `single_blocks` are all block-word forms; the CLIP text encoder uses
        // `text_model.encoder.layers.`, a different prefix). Its modules are the fused
        // `attention.qkv` / `attention.o`, `feed_forward.w{1,2,3}`, and `adaln_modulation`
        // (full-word `attention`, not the `attn`/`self_attn`/`attn1` every other family
        // uses). Requiring the prefix AND one Ideogram-specific module marker keeps this
        // from firing on anything else; the block-word disqualifiers are belt-and-braces.
        require_all_of: &[
            &["diffusion_model.layers."],
            &[".attention.qkv", ".adaln_modulation", ".feed_forward.w"],
        ],
        disqualifiers: &[
            "transformer_blocks.",
            "double_blocks.",
            "single_blocks",
            "diffusion_model.blocks.",
        ],
        markers: &[
            "diffusion_model.layers.",
            ".attention.qkv",
            ".attention.o.",
            ".feed_forward.w1",
            ".feed_forward.w2",
            ".feed_forward.w3",
            ".adaln_modulation",
        ],
    },
    BucketSignature {
        bucket: Bucket::Anima,
        // Anima (epic 10512, sc-10521) native / ComfyUI export. Its Cosmos-Predict2 DiT prefixes every
        // block with `diffusion_model.blocks.<n>.` — a prefix it SHARES with native/ComfyUI Wan — but
        // its adaLN modulation is the Cosmos triple `adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}`
        // (down/up pairs), a naming no other detected family uses (Wan has none; Ideogram uses a bare
        // `.adaln_modulation` with no `_self_attn`/`_cross_attn`/`_mlp` suffix). Requiring the prefix
        // AND one Cosmos adaLN marker positively identifies BOTH official shapes: the DiT-only style
        // LoRA (448 targets, no `llm_adapter`) lands here; the DiT+conditioner turbo LoRA is caught
        // earlier by `detect_unique_key_family` on its `llm_adapter` keys (and would land here too).
        //
        // `llm_adapter` is a SCORING marker (only the turbo shape carries it), never REQUIRED — a
        // signature that required it would misclassify the style LoRA (which has zero adapter tensors).
        // The kohya-flattened underscore forms are matched alongside the dotted forms. Wan/Flux/LTX/
        // Ideogram block-word forms are disqualified belt-and-braces; the colliding native-Wan
        // signature additionally disqualifies on the Cosmos adaLN markers so the two never co-score.
        //
        // The accepted prefixes are the full `BLOCK_PREFIX_MARKERS` set (not just the ComfyUI
        // `diffusion_model.` forms), so a kohya/musubi-flattened (`lora_unet_blocks_`) or
        // diffusers (`transformer.blocks.`) Anima export is recognized here rather than falling
        // through to the Wan signature that accepts those same prefixes (SceneWorks#1670).
        require_all_of: &[BLOCK_PREFIX_MARKERS, COSMOS_ADALN_MARKERS],
        disqualifiers: &[
            ".ffn.",
            "_ffn_",
            "transformer_blocks.",
            "double_blocks.",
            "single_blocks",
            "diffusion_model.layers.",
            "lora_te_",
            "lora_te1_",
            "lora_te2_",
        ],
        markers: &[
            "diffusion_model.blocks.",
            "diffusion_model_blocks_",
            "transformer.blocks.",
            "lora_unet_blocks_",
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
            "llm_adapter.",
            "_llm_adapter_",
            ".self_attn.",
            ".cross_attn.",
            "_self_attn_",
            "_cross_attn_",
            ".mlp.layer",
            "_mlp_layer",
        ],
    },
    BucketSignature {
        bucket: Bucket::Anima,
        // Anima, the shape that trains no adaLN modulation (SceneWorks#1670). The signature above
        // keys on the Cosmos adaLN-modulation modules, which only exist in the file when the trainer
        // *targeted* them — the two official LoRAs do, but the ordinary attention-only (and
        // attention+MLP) LoRAs people train and share do not. Such a file carries nothing but the
        // block prefix and `{self,cross}_attn` markers, which is exactly the native/ComfyUI Wan
        // layout, so it used to be reported as a confident `wan-video` and the import hard-rejected
        // ("LoRA file appears to be a wan-video model, but family was declared as anima").
        //
        // The discriminator that survives any choice of targets is the Cosmos *leaf module* naming —
        // `{self,cross}_attn.{q,k,v}_proj` / `output_proj` and `mlp.layer{1,2}` against Wan's
        // `.{q,k,v,o}.` and `ffn.{0,2}` (see `COSMOS_LEAF_MODULE_MARKERS`). Requiring a block prefix
        // AND one such leaf name identifies Anima positively; the colliding Wan signatures disqualify
        // on the same list, so the buckets never co-score and trip the runner-up margin.
        //
        // Disqualified whenever the adaLN keys are present so this never co-scores with the signature
        // above (two same-bucket scores would fail the 1.5× margin and return ambiguous — the
        // x-flux/Flux precedent). `lora_te*` is disqualified because CLIP text-encoder keys
        // (`..._self_attn_q_proj`) also contain the flattened leaf markers.
        require_all_of: &[BLOCK_PREFIX_MARKERS, COSMOS_LEAF_MODULE_MARKERS],
        disqualifiers: &[
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
            ".ffn.",
            "_ffn_",
            "transformer_blocks.",
            "double_blocks.",
            "single_blocks",
            "diffusion_model.layers.",
            "lora_te_",
            "lora_te1_",
            "lora_te2_",
        ],
        markers: &[
            "diffusion_model.blocks.",
            "diffusion_model_blocks_",
            "transformer.blocks.",
            "lora_unet_blocks_",
            "llm_adapter.",
            "_llm_adapter_",
            "_attn.q_proj",
            "_attn.k_proj",
            "_attn.v_proj",
            "_attn.output_proj",
            "_attn_q_proj",
            "_attn_k_proj",
            "_attn_v_proj",
            "_attn_output_proj",
            ".mlp.layer",
            "_mlp_layer",
        ],
    },
    BucketSignature {
        bucket: Bucket::Sd3,
        // Stable Diffusion 3 / 3.5 (diffusers format). Like Qwen-Image / Z-Image
        // it is a dual-stream MMDiT with `transformer.transformer_blocks.<n>.`
        // joint attention (`attn.{to_q,to_k,to_v,to_out.0}` for the image stream +
        // `attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}` for the text stream),
        // but its feedforwards are named `ff` / `ff_context` and its context norms
        // `*_context` (Qwen / Z-Image use `img_mlp` / `txt_mlp` instead, and never
        // carry a `ff_context` / `*_context` / `context_embedder` key). Requiring one
        // of those SD3-only context keys cleanly separates SD3 from the Qwen/Z-Image
        // bucket. SD3.5 (not SD3.0) also trains optional dual-attention `attn2`
        // sub-blocks (joint blocks 0..=12); `attn2` is allowed here. LTX-Video also
        // uses `transformer.transformer_blocks.` + `attn1`/`attn2`, but has no
        // joint-attention `add_q_proj` (it is single-stream cross-attention), so
        // requiring `add_q_proj`/`add_k_proj` keeps SD3 distinct from LTX.
        require_all_of: &[
            &["transformer.transformer_blocks."],
            &["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
            &[".ff_context.", ".norm1_context.", "context_embedder"],
        ],
        disqualifiers: &[
            "single_transformer_blocks.",
            "transformer.blocks.",
            ".img_mlp.",
            ".txt_mlp.",
            ".attn1.",
        ],
        markers: &[
            "transformer.transformer_blocks.",
            ".attn.add_q_proj",
            ".attn.add_k_proj",
            ".attn.add_v_proj",
            ".attn.to_add_out",
            ".attn.to_q.",
            ".attn.to_k.",
            ".attn.to_v.",
            ".attn.to_out.",
            ".attn2.",
            ".ff.",
            ".ff_context.",
            ".norm1_context.",
            ".norm2_context.",
            "context_embedder",
        ],
    },
    BucketSignature {
        bucket: Bucket::Sd3,
        // kohya / sd-scripts / LyCORIS flatten the SD3 dual-stream module path into
        // underscore-delimited keys behind a `lora_unet_` / `lora_transformer_` /
        // `lycoris_` prefix, e.g.
        // `lora_transformer_transformer_blocks_0_attn_add_q_proj.lora_down.weight`
        // or `..._ff_context_net_0_proj.lora_down.weight`. The dotted SD3 signature
        // above can't see these (no `transformer.transformer_blocks.` segment).
        // `_transformer_blocks_` (underscores both sides) + the joint-attention
        // `add_q_proj` + an SD3-only context key (`_ff_context_` / `_norm1_context_` /
        // `context_embedder`) discriminates it from Qwen/Z-Image kohya (which carry
        // `_img_mlp_`/`_txt_mlp_` and never a `_ff_context_`) and from Wan/SD/SDXL
        // kohya (which do not nest `_transformer_blocks_` under joint attention).
        require_all_of: &[
            &["_transformer_blocks_"],
            &["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
            &["_ff_context_", "_norm1_context_", "context_embedder"],
        ],
        disqualifiers: &[
            "single_transformer_blocks",
            "single_blocks_",
            "_img_mlp_",
            "_txt_mlp_",
            ".img_mlp.",
            ".txt_mlp.",
            "_attn1_",
            ".attn1.",
        ],
        markers: &[
            "_transformer_blocks_",
            "_attn_add_q_proj",
            "_attn_add_k_proj",
            "_attn_add_v_proj",
            "_attn_to_add_out",
            "_attn_to_q",
            "_attn_to_k",
            "_attn_to_v",
            "_attn_to_out",
            "_attn2_",
            "_ff_net_",
            "_ff_context_",
            "_norm1_context_",
            "_norm2_context_",
            "context_embedder",
        ],
    },
    BucketSignature {
        bucket: Bucket::MmDit,
        // Dual-stream MMDiT covers Qwen-Image and Z-Image. They share a key
        // layout in current Diffusers releases; per-family disambiguation
        // happens after this bucket is selected.
        //
        // The block prefix is matched as the bare `transformer_blocks.` rather than
        // the diffusers `transformer.transformer_blocks.` so that ComfyUI-distributed
        // Qwen-Image / Qwen-Image-Edit adapters — which drop the `transformer.` module
        // prefix and key their blocks as `transformer_blocks.<n>.attn.…` /
        // `.img_mlp.` / `.txt_mlp.` / `.attn.add_{q,k}_proj.` — are detected instead of
        // falling through as family-less (sc-10506). Diffusers keys still match (the
        // dotted form contains the bare substring); the sibling variants that also
        // contain `transformer_blocks.` as a substring — Flux's `single_transformer_blocks.`
        // and LTX's `.attn2.` — are rejected by the disqualifiers below, and the required
        // dual-stream group keeps single-stream families out. The dual-MLP requirement is
        // what separates this from a bare-`transformer_blocks.` Krea adapter (attention-only,
        // detected by its `family` metadata stamp).
        require_all_of: &[
            &["transformer_blocks."],
            &[
                ".img_mlp.",
                ".txt_mlp.",
                "add_q_proj",
                "add_k_proj",
                ".to_added_q.",
                ".to_added_k.",
            ],
        ],
        disqualifiers: &[
            "single_transformer_blocks.",
            "transformer.blocks.",
            ".attn2.",
            // SD3 / SD3.5 shares the joint-attention `add_q_proj` keys but is a
            // separate bucket (it uses `ff`/`ff_context` + `*_context` norms, never
            // `img_mlp`/`txt_mlp`). Disqualify on its context-only keys so a
            // non-dual-attention SD3 LoRA never co-scores here and trips the
            // runner-up margin against the SD3 signature (sc-7874).
            ".ff_context.",
            ".norm1_context.",
            ".norm2_context.",
            "context_embedder",
        ],
        markers: &[
            "transformer_blocks.",
            ".img_mlp.",
            ".txt_mlp.",
            "add_q_proj",
            "add_k_proj",
            ".to_added_q.",
            ".to_added_k.",
        ],
    },
    BucketSignature {
        bucket: Bucket::MmDit,
        // kohya / musubi-tuner / LyCORIS (lycoris-lora) Qwen-Image & Z-Image LoRAs
        // flatten the dual-stream MMDiT module path into underscore-delimited keys
        // behind a `lora_unet_` / `lora_transformer_` / `lycoris_` prefix, e.g.
        // `lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w1` or
        // `lora_unet_transformer_blocks_0_img_mlp_net_0_proj.lora_down.weight`. The
        // dotted MMDiT signature above can't see these (no `transformer.
        // transformer_blocks.` segment). `_transformer_blocks_` (underscores on both
        // sides) is the discriminator: Wan kohya uses `_blocks_`, SD/SDXL kohya use
        // `_down_blocks_`/`_up_blocks_`/`_mid_block_`, and SDXL's nested
        // `_transformer_blocks_` never carries the joint-attention `add_{q,k}_proj`
        // or dual-stream `_img_mlp_`/`_txt_mlp_` that group two requires. Flux's
        // single-stream keys are disqualified so a (double+single) Flux LoRA never
        // lands here despite sharing `transformer_blocks` + `add_q_proj`.
        require_all_of: &[
            &["_transformer_blocks_"],
            &[
                "_img_mlp_",
                "_txt_mlp_",
                "add_q_proj",
                "add_k_proj",
                "to_added_q",
                "to_added_k",
            ],
        ],
        disqualifiers: &[
            "single_transformer_blocks",
            "single_blocks_",
            ".attn1.",
            ".attn2.",
            "_attn1_",
            "_attn2_",
            // SD3 / SD3.5 kohya/LyCORIS export shares `_transformer_blocks_` +
            // joint-attention `add_q_proj`, but is the separate SD3 bucket
            // (`_ff_context_` / `_norm1_context_` / `context_embedder`, never
            // `_img_mlp_`/`_txt_mlp_`). Disqualify on those SD3-only context keys
            // (sc-7874).
            "_ff_context_",
            "_norm1_context_",
            "_norm2_context_",
            "context_embedder",
        ],
        markers: &[
            "_transformer_blocks_",
            "_img_mlp_",
            "_txt_mlp_",
            "add_q_proj",
            "add_k_proj",
            "to_added_q",
            "to_added_k",
            "_attn_to_q",
            "_attn_to_k",
            "_attn_to_v",
        ],
    },
    BucketSignature {
        bucket: Bucket::Sdxl,
        // SDXL ships two text encoders, so kohya-style LoRAs always carry
        // both `lora_te1_` and `lora_te2_` prefixes alongside `lora_unet_`.
        require_all_of: &[&["lora_unet_"], &["lora_te1_"], &["lora_te2_"]],
        disqualifiers: &["transformer.transformer_blocks.", "transformer.blocks."],
        markers: &["lora_unet_", "lora_te1_", "lora_te2_"],
    },
    BucketSignature {
        bucket: Bucket::Sd15,
        // SD1.5 only ships a single text encoder, so kohya-style LoRAs
        // never carry the SDXL `lora_te1_` / `lora_te2_` split.
        require_all_of: &[&["lora_unet_"], &["lora_te_"]],
        disqualifiers: &[
            "lora_te1_",
            "lora_te2_",
            "transformer.transformer_blocks.",
            "transformer.blocks.",
        ],
        markers: &["lora_unet_", "lora_te_"],
    },
];

/// Minimum marker hits required for any bucket to win. Below this the file
/// is treated as ambiguous.
const MIN_KEY_MATCHES: usize = 4;

/// Best score must beat the runner-up by at least 1.5×. Encoded as a
/// rational comparison so we never need floats: best * DEN >= second * NUM.
const MARGIN_NUM: usize = 3;
const MARGIN_DEN: usize = 2;

/// Block-index threshold for Qwen-Image (larger, ~60 blocks). Values are
/// zero-indexed block numbers. Low block indices are intentionally not
/// enough to identify Z-Image: a sparse Qwen LoRA may only train early
/// blocks, and false hard rejections are worse than an inconclusive result.
const QWEN_MIN_BLOCK_INDEX: usize = 39;

/// Mage-Flow's published NR-MMDiT depth (`transformer/config.json` `depth: 12`, identical across
/// all six repos) — block indices `0..=11` and nothing above. See [`disambiguate_mm_dit`].
const MAGE_FLOW_BLOCK_COUNT: usize = 12;

fn detect_bucket(keys: &[String]) -> Option<Bucket> {
    let mut scored: Vec<(Bucket, usize)> = SIGNATURES
        .iter()
        .map(|sig| (sig.bucket, score_signature(sig, keys)))
        .collect();
    scored.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    let (best_bucket, best_score) = scored[0];
    if best_score < MIN_KEY_MATCHES {
        return None;
    }
    let second_score = scored.get(1).map(|entry| entry.1).unwrap_or(0);
    if second_score > 0 && best_score * MARGIN_DEN < second_score * MARGIN_NUM {
        // best/second < 1.5 → too close to call.
        return None;
    }
    Some(best_bucket)
}

fn score_signature(sig: &BucketSignature, keys: &[String]) -> usize {
    if keys.iter().any(|key| {
        sig.disqualifiers
            .iter()
            .any(|disqualifier| key.contains(disqualifier))
    }) {
        return 0;
    }
    let all_required_groups_satisfied = sig.require_all_of.iter().all(|group| {
        keys.iter()
            .any(|key| group.iter().any(|needle| key.contains(needle)))
    });
    if !all_required_groups_satisfied {
        return 0;
    }
    let mut score = 0_usize;
    for key in keys {
        if sig.markers.iter().any(|marker| key.contains(marker)) {
            score += 1;
        }
    }
    score
}

/// Splits the dual-stream MMDiT bucket by transformer **depth**, the only tensor-level signal that
/// separates its members.
///
/// Qwen-Image (60 blocks) is claimed from [`QWEN_MIN_BLOCK_INDEX`] up. Mage-Flow (sc-14057) is the
/// other end: its NR-MMDiT is a reparameterized Z-Image S3-DiT whose diffusers module names are
/// spelled *identically* to Qwen-Image's — `transformer_blocks.<n>.attn.to_{q,k,v}` /
/// `.attn.add_{q,k,v}_proj` / `.img_mlp.net.0.proj` / `.txt_mlp.net.2` / `.img_mod.1` /
/// `.txt_mod.1` — at the same 3072 hidden size and 24 heads, so no key or shape tells them apart.
/// What does is the published `depth: 12`: a full-coverage Mage adapter spans block indices
/// `0..=11` exactly, where Qwen-Image spans `0..=59` and Z-Image `0..=29`.
///
/// The **whole contiguous span** is required, not just `max == 11`, and that is what keeps the
/// signal honest: a sparse adapter that trains only a handful of early blocks stays inconclusive
/// (`None`) rather than being confidently mislabelled, preserving the module's "a wrong family is
/// worse than no family" contract. The residual overlap — a Qwen-Image adapter that trains exactly
/// its first twelve blocks and no others — is not a convention any trainer ships, and every
/// mainstream Qwen trainer stamps `ss_base_model_version` / `modelspec.architecture`, which
/// [`detect_metadata_family`] resolves *before* this key path ever runs.
///
/// This mirrors the base-checkpoint classifier's Mage arm
/// (`base_weights::detect_transformer_family`), which pins the same exact `0..12` index set. It
/// does **not** additionally require `.img_mod.1.`/`.txt_mod.1.` the way that arm does: a base
/// checkpoint always carries every module, whereas an adapter trains a chosen subset and most never
/// touch the modulation projections.
fn disambiguate_mm_dit(keys: &[String]) -> Option<String> {
    let blocks = transformer_block_indices(keys);
    let max_block = *blocks.iter().next_back()?;
    if max_block >= QWEN_MIN_BLOCK_INDEX {
        return Some("qwen-image".to_owned());
    }
    if max_block == MAGE_FLOW_BLOCK_COUNT - 1 && blocks.len() == MAGE_FLOW_BLOCK_COUNT {
        return Some("mage-flow".to_owned());
    }
    None
}

/// Every `N` seen in keys matching `transformer.transformer_blocks.<N>.` (or just
/// `transformer_blocks.<N>.`, or the kohya-flattened `transformer_blocks_<N>_`), de-duplicated and
/// ordered. Empty when no such key exists.
fn transformer_block_indices(keys: &[String]) -> std::collections::BTreeSet<usize> {
    keys.iter()
        .filter_map(|key| parse_block_index(key))
        .collect()
}

fn parse_block_index(key: &str) -> Option<usize> {
    // Diffusers separates with dots (`transformer_blocks.<N>.`); kohya / LyCORIS
    // flatten to underscores (`transformer_blocks_<N>_`). Accept either separator.
    let needle = "transformer_blocks";
    let mut rest = key;
    while let Some(position) = rest.find(needle) {
        let after = &rest[position + needle.len()..];
        let candidate = match after.as_bytes().first() {
            Some(b'.' | b'_') => &after[1..],
            _ => after,
        };
        let digits: String = candidate.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() {
            if let Ok(index) = digits.parse::<usize>() {
                return Some(index);
            }
        }
        rest = after;
    }
    None
}

fn detect_metadata_family(header: &Value) -> Option<String> {
    let metadata = header.get("__metadata__")?.as_object()?;
    // SceneWorks-native provenance first: our own trainers stamp the adapter header with a canonical
    // `family` token (`krea_2`, `z-image`, …) and a `baseModel` training-base id (`krea_2_raw`, …),
    // specifically so import reconciliation can validate it (see the candle/MLX Krea trainers' `save`).
    // These keys are NOT part of the kohya (`ss_base_model_version`) or diffusers (`modelspec.*`)
    // conventions checked below, so without reading them a SceneWorks-created LoRA whose tensor keys
    // match no bucket signature is left undetected — e.g. a default Krea LoRA, whose bare
    // `transformer_blocks.<n>.attn.to_{q,k,v}` keys hit no bucket and carry no `text_fusion`/`to_gate`
    // unique key. The `family` stamp is already the canonical token, so trust it directly; `baseModel`
    // is a model id, so map it through the architecture matcher alongside the kohya/diffusers keys.
    if let Some(family) = metadata
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(canonical_lora_family(family));
    }
    for key in [
        "baseModel",
        "base_model",
        "ss_base_model_version",
        "modelspec.architecture",
        "modelspec.implementation",
    ] {
        let Some(value) = metadata.get(key).and_then(Value::as_str) else {
            continue;
        };
        if let Some(family) = metadata_value_to_family(value) {
            return Some(family);
        }
    }
    None
}

/// Whether `haystack` contains `needle` starting at a **token boundary** — at the very start of
/// the string, or immediately after a non-alphanumeric byte.
///
/// Exists for the Mage arm in [`metadata_value_to_family`] (sc-14057): a plain `contains` there
/// matches the tail of `image`, so `z-image-flow` / `qwen_image_flow` / `imageflow` would all read
/// as Mage-Flow and be hard-rejected from their own models. Separators (`-`, `_`, `/`, `.`, space)
/// and string start are boundaries; a letter or digit is not.
///
/// Byte indexing is safe for any input: `match_indices` yields char-boundary offsets, and a
/// preceding UTF-8 continuation byte is not ASCII-alphanumeric, so a multi-byte neighbour is
/// treated as a boundary rather than panicking.
fn contains_token(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(position, _)| {
        position == 0 || !haystack.as_bytes()[position - 1].is_ascii_alphanumeric()
    })
}

fn metadata_value_to_family(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    // Check chroma before flux: Chroma is FLUX.1-schnell-derived, so a Chroma
    // LoRA's metadata may name both. Only metadata can distinguish the two — by
    // tensor keys a Chroma LoRA is identical to a Flux LoRA (same double/single
    // transformer blocks), so the key-based detector classifies it as `flux`.
    if normalized.contains("chroma") {
        return Some("chroma".to_owned());
    }
    // Krea 2 (epic 7565): a distinct single-stream DiT family. No SD/Flux/Qwen
    // architecture string contains "krea", so this can sit ahead of them safely.
    // The label is `krea_2` (underscore) to match the catalog/trainer convention.
    if normalized.contains("krea") {
        return Some("krea_2".to_owned());
    }
    // Ideogram 4 (epic 4725): its own MMDiT (`diffusion_model.layers.<n>.attention.qkv`
    // + `adaln_modulation`), a distinct family from every SD/Flux/Qwen architecture. The
    // ai-toolkit trainer stamps `ss_base_model_version: "ideogram4"`; no other family's
    // architecture string contains "ideogram", so match it here. The label is `ideogram`
    // to match the `ideogram_4` catalog `family` / `loraCompatibility.families`.
    if normalized.contains("ideogram") {
        return Some("ideogram".to_owned());
    }
    // Anima (epic 10512): the Cosmos-Predict2 anime DiT. Its catalog/training-base ids are
    // `anima_base` / `anima_aesthetic` / `anima_turbo` and its weight files
    // `anima-<variant>-v1.0.safetensors`, so a trainer that stamps the base id or file name
    // names the family here (SceneWorks#1670) instead of falling through to key detection,
    // where a metadata-less Anima file can be mistaken for Wan.
    //
    // Matched on a token boundary, NOT `contains("anima")`: several unrelated SDXL/SD1.5
    // architectures embed that substring — Animagine XL (`animagine-xl-3.1`) and AnimateDiff
    // (`animatediff`) most notably — and mapping one of those to `anima` would hard-reject a
    // perfectly good SDXL LoRA, the exact failure mode this whole path exists to avoid.
    if normalized == "anima"
        || normalized.starts_with("anima_")
        || normalized.starts_with("anima-")
        || normalized.contains("cosmos-predict")
        || normalized.contains("cosmos_predict")
        || normalized.contains("cosmospredict")
    {
        return Some("anima".to_owned());
    }
    // Mage-Flow (epic 14034). Catalog/training-base ids are `mage_flow_base` / `mage_flow` /
    // `mage_flow_turbo` (+ the `_edit_*` trio), repos `microsoft|SceneWorks/Mage-Flow*`, and the MLX
    // trainer stamps `family: "mage_flow"`.
    //
    // 🔴 TRAP: "mage" is a SUBSTRING OF "image" — `z-image`, `qwen-image`, `qwen_image`,
    // `ZImagePipeline` all contain it. A `contains("mage")` test (the shape used for `flux` /
    // `krea` / `ideogram`) would therefore swallow every Z-Image and Qwen-Image adapter and reject
    // it from its own model. Matched on a token boundary instead — the same defence the `anima`
    // arm above needs against `animagine-xl` / `animatediff` — so only a genuine Mage id, repo
    // name, or file name resolves here.
    //
    // The boundary rule applies to the `*flow` spellings too, not just the bare token: `contains`
    // there re-opens the identical hole one level down, since `z-image-flow`, `qwen_image_flow`
    // and `imageflow` all end in `mage-flow` / `mage_flow` / `mageflow`. This arm runs *ahead* of
    // the z-image and qwen-image arms below, so such a hit would be a confident mislabel — strictly
    // worse than the `None` those names deserve.
    if normalized == "mage"
        || normalized.starts_with("mage-")
        || normalized.starts_with("mage_")
        || contains_token(&normalized, "mage-flow")
        || contains_token(&normalized, "mage_flow")
        || contains_token(&normalized, "mageflow")
    {
        return Some("mage-flow".to_owned());
    }
    // Check flux2 before flux: FLUX.2 architecture strings ("flux2", "flux-2",
    // "flux.2", "flux_2") all contain the "flux" substring, so the generic flux
    // match below would otherwise swallow them. FLUX.2 is a distinct family with
    // its own MLX adapter — a FLUX.1 ("flux") LoRA is not interchangeable with it.
    if normalized.contains("flux2")
        || normalized.contains("flux-2")
        || normalized.contains("flux.2")
        || normalized.contains("flux_2")
    {
        return Some("flux2".to_owned());
    }
    if normalized.contains("flux") {
        return Some("flux".to_owned());
    }
    if normalized.contains("zimage") || normalized.contains("z-image") {
        return Some("z-image".to_owned());
    }
    if normalized.contains("qwen") && normalized.contains("image") {
        return Some("qwen-image".to_owned());
    }
    if normalized.contains("ltx") {
        return Some("ltx-video".to_owned());
    }
    if normalized.contains("wan") {
        return Some("wan-video".to_owned());
    }
    // Check SD3 before sdxl / sd1.5: SD3 / SD3.5 architecture strings
    // ("sd3", "sd-3", "sd3.5", "stable-diffusion-3", "stable-diffusion-3.5")
    // are a distinct dual-stream MMDiT family, not the UNet-based SDXL / SD1.5.
    // None of the SDXL / SD1.5 markers below contain a bare "sd3" or
    // "stable-diffusion-3", so ordering here is safe (sc-7874).
    if normalized.contains("sd3")
        || normalized.contains("sd-3")
        || normalized.contains("sd_3")
        || normalized.contains("stable-diffusion-3")
        || normalized.contains("stable_diffusion_3")
        || normalized.contains("stablediffusion3")
    {
        return Some("sd3".to_owned());
    }
    if normalized.contains("sdxl") {
        return Some("sdxl".to_owned());
    }
    if normalized == "sd1" || normalized == "sd1.5" || normalized.contains("stable-diffusion-v1") {
        return Some("sd1.5".to_owned());
    }
    None
}

/// What an adapter file's `__metadata__` says about itself, beyond its family (sc-14057).
///
/// Every field is optional **on purpose**. A great many third-party adapters ship none of them, and
/// the one thing this type must never do is invent a plausible-looking value: guessing `rank` from
/// `alpha` (or either from a factor shape) would record a number the file never claimed, and a
/// wrong rank is exactly the failure that makes an adapter apply at the wrong strength. Absent
/// stays absent, and callers omit the field rather than defaulting it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdapterFileMetadata {
    /// Declared network type, lower-cased — `lora` / `lokr` / `loha` / …. The worker's
    /// `classify_adapter` keys the engine's adapter `kind` off the same `networkType` stamp.
    pub network_type: Option<String>,
    /// LoRA/LoKr rank (`r`). `None` when the file does not declare one.
    pub rank: Option<u64>,
    /// LoRA/LoKr alpha. `None` when the file does not declare one — **not** silently equal to
    /// `rank`, which is only the *loader's* last-resort scaling fallback, never a recorded fact.
    pub alpha: Option<f64>,
}

impl AdapterFileMetadata {
    pub fn is_empty(&self) -> bool {
        self.network_type.is_none() && self.rank.is_none() && self.alpha.is_none()
    }
}

/// Locates the first `.safetensors` under `dir` and answers both questions its header can settle:
/// the detected architecture family, and what the file declares about itself (sc-14057).
///
/// Returns an empty pair — `(None, AdapterFileMetadata::default())` — when the directory holds no
/// safetensors at all, and the structured [`SafetensorsHeaderError`] when a file was found but its
/// header is unreadable, malformed, or fronts a truncated file. Either half of a successful answer
/// may still legitimately be absent: an unrecognized architecture is `None`, and a file with a bare
/// `{"format": "pt"}` block yields an empty [`AdapterFileMetadata`] rather than invented defaults.
///
/// This is the *post-download* counterpart to the API's file-level inspector. The two import routes
/// see the file at different times — a local/uploaded file exists when the job is queued, a
/// repo/URL import only after the transfer lands — so both need the same one-read/two-answers
/// primitive if an adapter is to be described identically whichever way it arrived.
pub fn inspect_adapter_in_dir(
    dir: &Path,
) -> Result<(Option<String>, AdapterFileMetadata), SafetensorsHeaderError> {
    let Some(safetensors_path) = first_safetensors_path(dir) else {
        return Ok((None, AdapterFileMetadata::default()));
    };
    let header = read_safetensors_header(&safetensors_path)?;
    Ok((detect_lora_family(&header), read_adapter_metadata(&header)))
}

/// The single writer for the adapter-declared fields on an imported LoRA's manifest entry
/// (sc-14057): `networkType`, `rank`, `alpha`.
///
/// One function so the local-file route (which records these at queue time, from the source file)
/// and the repo/URL route (which can only record them once the worker has downloaded the file)
/// produce the *same shape* for the same adapter. Two identical adapters differing only in how they
/// were ingested previously ended up with different manifest entries.
///
/// Semantics, deliberately:
/// - **Only declared fields are written.** An adapter that states no `alpha` records none; it does
///   not inherit `rank`. `alpha = rank` is the loader's last-resort scaling fallback, never a fact
///   about the file, and recording it would let a fallback masquerade as a stated value.
/// - **Existing values win** (`or_insert`, matching the family reconcile at the same call site). A
///   value already on the entry came from the API pre-flight or an explicit caller, and a
///   post-download re-read must not silently rewrite it.
pub fn apply_adapter_metadata_to_manifest_entry(
    entry: &mut Map<String, Value>,
    metadata: &AdapterFileMetadata,
) {
    if let Some(network_type) = &metadata.network_type {
        entry
            .entry("networkType")
            .or_insert_with(|| Value::String(network_type.clone()));
    }
    if let Some(rank) = metadata.rank {
        entry.entry("rank").or_insert_with(|| json!(rank));
    }
    if let Some(alpha) = metadata.alpha {
        entry.entry("alpha").or_insert_with(|| json!(alpha));
    }
}

/// Reads the rank / alpha / network-type an adapter file declares in its safetensors
/// `__metadata__` (sc-14057).
///
/// Three conventions are accepted, in precedence order:
///
/// 1. **SceneWorks' own trainers** — `networkType` / `rank` / `alpha` (the epic-2193 reload
///    contract; `save_lora_peft` and `save_lokr` in the inference workspace write these as
///    *strings*, since the safetensors metadata map is string-valued).
/// 2. **kohya / sd-scripts** — `ss_network_dim` / `ss_network_alpha` / `ss_network_module`.
/// 3. **diffusers `save_lora_adapter` / raw PEFT** — no top-level keys at all; `r` and
///    `lora_alpha` live inside the JSON blob at `lora_adapter_metadata` (sc-5513). Only the global
///    values are read here: PEFT's per-module `rank_pattern` / `alpha_pattern` overrides are a
///    *loader* concern (the inference `LoraAdapterMeta` resolves them per target) and have no
///    single value to record.
///
/// Numbers are accepted as JSON numbers as well as strings, because a non-conforming writer can
/// emit either. A non-positive or unparseable rank is treated as absent — it can only poison a
/// scaling denominator. A zero *alpha* is kept: that is a legitimately scale-0 adapter.
pub fn read_adapter_metadata(header: &Value) -> AdapterFileMetadata {
    let Some(metadata) = header.get("__metadata__").and_then(Value::as_object) else {
        return AdapterFileMetadata::default();
    };
    // The PEFT blob is a JSON *string* under one key; parse it once and treat it as the last
    // fallback for each field.
    let peft_blob: Option<Map<String, Value>> = metadata
        .get("lora_adapter_metadata")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|value| value.as_object().cloned());
    let lookup = |keys: &[&str]| -> Option<Value> {
        for key in keys {
            if let Some(value) = metadata.get(*key) {
                return Some(value.clone());
            }
        }
        let blob = peft_blob.as_ref()?;
        for key in keys {
            if let Some(value) = blob.get(*key) {
                return Some(value.clone());
            }
        }
        None
    };

    let network_type = lookup(&["networkType", "network_type"])
        .as_ref()
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            // LyCORIS under kohya sets `ss_network_module: "lycoris.kohya"` — which names no algo —
            // and puts the real one in the `ss_network_args` JSON blob as `algo`.
            let algo = metadata
                .get("ss_network_args")
                .and_then(Value::as_str)
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())?
                .get("algo")?
                .as_str()?
                .trim()
                .to_ascii_lowercase();
            (!algo.is_empty()).then_some(algo)
        })
        .or_else(|| {
            // Plain kohya records the *module path* (`networks.lora`). Only claim a type when the
            // path names one outright — `lycoris.kohya` names none, and guessing would be worse
            // than leaving it unstated.
            let module = metadata
                .get("ss_network_module")
                .and_then(Value::as_str)?
                .to_ascii_lowercase();
            ["lokr", "loha", "lora"]
                .into_iter()
                .find(|token| module.contains(token))
                .map(str::to_owned)
        });
    let rank = lookup(&["rank", "ss_network_dim", "r"])
        .as_ref()
        .and_then(number_from_metadata)
        .filter(|value| *value > 0.0)
        .map(|value| value as u64);
    let alpha = lookup(&["alpha", "ss_network_alpha", "lora_alpha"])
        .as_ref()
        .and_then(number_from_metadata);
    AdapterFileMetadata {
        network_type,
        rank,
        alpha,
    }
}

/// A safetensors `__metadata__` value as a number: the map is string-valued by spec (our trainers
/// write `"16"`), but PEFT's embedded JSON blob carries real numbers.
fn number_from_metadata(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
    .filter(|value| value.is_finite())
}

/// The safetensors header is a JSON object whose top-level keys are tensor
/// names plus a special `__metadata__` entry. Returns the tensor names only.
fn collect_tensor_keys(header: &Value) -> Vec<String> {
    let Some(object) = header.as_object() else {
        return Vec::new();
    };
    object
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// LoRA family-compatibility validation (epic 3018, sc-3027).
//
// Ported from the Python worker's `lora_adapters.py`
// (validate_lora_compatibility / accepted_lora_families / lora_families /
// lora_base_model) so the Rust GPU worker rejects an incompatible LoRA *before*
// a job runs, with the same message, instead of failing deep in the engine's
// strict adapter loader. Pure (no I/O): it reads the LoRA spec's *declared*
// families, exactly like the Python pre-flight.
// ---------------------------------------------------------------------------

/// Maximum LoRAs per job (matches the worker's `MAX_JOB_LORAS` / Python
/// `normalize_lora_specs`).
pub const MAX_JOB_LORAS: usize = 5;

/// Architecture families a model can load LoRAs from *in addition to* its own
/// (Python `EXTRA_COMPATIBLE_LORA_FAMILIES`). Chroma is FLUX.1-derived and shares
/// Flux's block layout, so Flux LoRAs load on Chroma (one-directional). FLUX.2
/// [klein]'s model family is `flux2-klein` but klein LoRAs are detected/declared
/// as `flux2`, so a klein model must accept `flux2` LoRAs. FLUX.2-dev's family is
/// `flux2-dev` (a separate model — Mistral3 TE + 48/48 DiT) but it shares the FLUX.2
/// transformer layout, so dev LoRAs are likewise detected/declared as `flux2` (epic 5914;
/// dev LoRA application is validated in sc-5920). Krea Realtime 14B's model family is
/// `krea-realtime`, but its checkpoint IS Wan 2.1 T2V 14B weight-for-weight — same 40 blocks, same
/// `blocks.{i}.self_attn.{q,k,v,o}` / `ffn` target names — so Wan-family LoRAs install on it
/// (sc-15015 wired the dense forward-time residual path in `mlx-gen-krea-realtime`, and it is
/// tier-agnostic over the packed Q4/Q8 bases). Declaring it HERE rather than putting `wan-video` in
/// the model's own `loraCompatibility.families` is deliberate: family membership is read by every
/// other family-keyed gate too (training bases, repo/tier resolution), and Krea Realtime is not a
/// Wan model for any of those. The relation is one-directional in the usual way — a Wan LoRA loads
/// on Krea Realtime, and a Krea-Realtime checkpoint detected by tensor keys reports the base
/// `wan-video` (there is no krea-specific signature), which is exactly what
/// [`detected_base_architecture_satisfies_declared`] needs; a Krea-Realtime LoRA is not thereby
/// declared loadable on a Wan model.
fn extra_compatible_lora_families(normalized_family: &str) -> &'static [&'static str] {
    match normalized_family {
        "chroma" => &["flux"],
        "flux2-klein" | "flux2-dev" => &["flux2"],
        "krea-realtime" => &["wan-video"],
        // SCAIL-2's DiT is Wan2.1-I2V-14B-derived and ships the raw I2V module names — the same
        // `blocks.{i}.self_attn.{q,k,v,o}` / `ffn` / qk-norm targets — so a Wan2.1-I2V LoRA's tensor
        // keys resolve against it. That is not incidental: the bundled `scail2_lightning` speed
        // toggle IS a lightx2v Wan2.1-I2V step-distill LoRA, applied cross-architecture on purpose
        // (the engine merges every compatible target and deliberately skips the one that differs,
        // the in_dim-36 `patch_embedding` vs SCAIL-2's in_dim 20). Because the file is a genuine Wan
        // LoRA, `detect_lora_family` reports `wan-video` — correctly — so without this entry the
        // job-creation gate rejects the very transplant the engine exists to perform (sc-18200).
        // Declared HERE rather than as `wan-video` in the model's own `loraCompatibility.families`
        // for the same reason as krea-realtime above: family membership is read by every other
        // family-keyed gate too (tier/repo resolution, adapter routing, training bases), and SCAIL-2
        // is not a Wan model for any of those. One-directional as usual — a Wan LoRA loads on
        // SCAIL-2; a SCAIL-2 LoRA is not thereby declared loadable on a Wan model.
        "scail2" => &["wan-video"],
        _ => &[],
    }
}

/// The set of LoRA families a model of `model_family` can load (normalized; the
/// model's own family plus its extra-compatible families). Empty when the family
/// is unknown — callers treat that as "skip validation".
pub fn accepted_lora_families(model_family: &str) -> Vec<String> {
    let normalized = normalize_model_family(model_family);
    if normalized.is_empty() {
        return Vec::new();
    }
    let mut families = vec![normalized.clone()];
    families.extend(
        extra_compatible_lora_families(&normalized)
            .iter()
            .map(|family| (*family).to_owned()),
    );
    families
}

/// The LoRA's declared compatible families (normalized, de-duplicated, sorted),
/// from the first present of `families` / `compatibleFamilies` / `modelFamilies`
/// / `compatibility.families` / `[family]`. Empty when the spec declares none
/// (an unstamped LoRA the user vouches for — never rejected on family grounds).
pub fn lora_declared_families(lora: &Value) -> Vec<String> {
    let compatibility = lora.get("compatibility").and_then(Value::as_object);
    let raw = ["families", "compatibleFamilies", "modelFamilies"]
        .into_iter()
        .find_map(|key| lora.get(key).and_then(Value::as_array).cloned())
        .or_else(|| {
            compatibility
                .and_then(|compat| compat.get("families").and_then(Value::as_array).cloned())
        })
        .or_else(|| {
            lora.get("family")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(|value| vec![Value::String(value.to_owned())])
        })
        .unwrap_or_default();
    let mut families: Vec<String> = raw
        .iter()
        .filter_map(Value::as_str)
        .map(normalize_model_family)
        .filter(|family| !family.is_empty())
        .collect();
    families.sort();
    families.dedup();
    families
}

/// The specific base model a LoRA records (e.g. `wan_2_2`, `wan_2_2_t2v_14b`), or
/// `None`. Used by the base-model gate for families where a matching family alone
/// is not enough (Python `lora_base_model`).
pub fn lora_base_model(lora: &Value) -> Option<String> {
    lora.get("baseModel")
        .or_else(|| lora.get("base_model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Families that share an architecture family but NOT a LoRA-compatible
/// architecture, so the trained base model must also match (Python
/// `_BASE_MODEL_GATED_FAMILIES`). Wan: `wan_2_2` (5B) and `wan_2_2_*_14b` (A14B)
/// are both `wan-video` but cross-applying a LoRA garbles output.
fn is_base_model_gated_family(family: &str) -> bool {
    family == "wan-video"
}

/// Whether a model id names a **14B-class** Wan-family backbone, by the catalog's own id
/// convention: the 14B entries carry a `_14b` suffix (`wan_2_2_t2v_14b`, `wan_2_2_i2v_14b`,
/// `wan_2_2_vace_fun_14b`, `scail2_14b`, `krea_realtime_14b`) and the 5B TI2V entry — the other
/// side of the split [`is_base_model_gated_family`] exists for — does not (`wan_2_2`).
///
/// A convention rather than an enumerated list on purpose: a new Wan 14B entry must not have to be
/// added here to keep working, and a hard-coded list would silently exclude it.
fn is_wan_14b_class_id(id: &str) -> bool {
    normalize_model_family(id).ends_with("-14b")
}

/// Whether a model id names an **image-to-video** Wan entry (`wan_2_2_i2v_14b`), by the same id
/// convention — an `i2v` path SEGMENT, not a substring, so a hypothetical `…si2vx…` id cannot match
/// by accident.
///
/// This is a size-class peer of the 14B T2V entries, so [`is_wan_14b_class_id`] alone admits it.
/// It must not ride the extra-compatible arm: an I2V LoRA targets `cross_attn.k_img`/`v_img`, which
/// a text-to-video backbone does not have at any surface width, and the product ALREADY treats the
/// two as non-interchangeable — `wan_2_2_i2v_14b` is refused on `wan_2_2_t2v_14b` by exact equality.
/// Admitting it on Krea Realtime (also a T2V backbone) would be the one place that inconsistency
/// existed, and it would surface as a hard engine error only AFTER a multi-GB tier fetch instead of
/// a 400 at submit.
fn is_wan_i2v_id(id: &str) -> bool {
    normalize_model_family(id)
        .split('-')
        .any(|segment| segment == "i2v")
}

/// Whether a LoRA recording trained base model `base` may load on `model_id`, for a model whose own
/// declared family is `model_family`. This is the base-model half of the compatibility gate; the
/// family half is [`accepted_lora_families`].
///
/// **Exact id equality is the ordinary answer** and the only one for a genuine Wan model: `wan_2_2`
/// (TI2V-5B) and the `*_14b` entries both declare `wan-video`, and cross-applying a LoRA between
/// them garbles the output, so a LoRA that records its base pins to that base.
///
/// The second arm exists because of the extra-compatible relation (sc-15017). Krea Realtime 14B
/// declares its OWN `krea-realtime` family and accepts `wan-video` LoRAs because its DiT is Wan 2.1
/// T2V 14B weight-for-weight — but a Wan LoRA's recorded base is a Wan id, so it can NEVER equal
/// `krea_realtime_14b`. Under exact equality alone, every base-model-stamped Wan LoRA would be
/// refused on Krea and the relation would be dead on arrival for exactly the LoRAs the app itself
/// stamps at import. So for a model that accepts `wan-video` through the registry, the gate is
/// preserved rather than dropped, on TWO axes:
///
/// * the **size class** (`is_wan_14b_class_id`) — the 5B-vs-14B split the gate was written for;
/// * the **conditioning class** (`is_wan_i2v_id`) — an I2V base is refused, because Krea Realtime is
///   a text-to-video backbone and `wan_2_2_i2v_14b` is already refused on the sibling
///   `wan_2_2_t2v_14b` by exact equality. Without this the arm would be the one place in the product
///   where an I2V stamp is admitted onto a T2V model, and the mismatch would surface as a hard
///   engine error after a multi-GB tier fetch rather than a 400 at submit.
///
/// A LoRA that records NO base model is not this function's concern — the callers fall back to
/// family gating for those, exactly as before.
pub fn base_model_satisfies_gate(model_family: &str, model_id: &str, base: &str) -> bool {
    if base == model_id {
        return true;
    }
    let normalized_family = normalize_model_family(model_family);
    extra_compatible_lora_families(&normalized_family).contains(&"wan-video")
        && is_wan_14b_class_id(model_id)
        && is_wan_14b_class_id(base)
        && (extra_compatible_backbone_is_image_conditioned(&normalized_family)
            || !is_wan_i2v_id(base))
}

/// Whether a model riding the `wan-video` extra-compatible arm has an **image-conditioned** (I2V)
/// backbone, i.e. one that actually carries the `cross_attn.k_img`/`v_img` projections an I2V LoRA
/// targets.
///
/// This exists because the I2V exclusion in [`base_model_satisfies_gate`] is not a property of Wan
/// LoRAs — it is a property of the *host*. sc-15017 introduced it for Krea Realtime, whose DiT is Wan
/// 2.1 **T2V** 14B: it has no `k_img`/`v_img` at any width, so an I2V-stamped LoRA cannot apply and
/// the exclusion is right. Applying that same rule to every extra-compatible family inverts it for an
/// image-conditioned host — SCAIL-2's DiT IS Wan2.1-**I2V**-14B-derived and does carry `k_img`/`v_img`
/// (they are adaptable targets, and the bundled `scail2_lightning` adapter patches them), so an I2V
/// base is the *exact* architectural match while a T2V base is the lossy one. Without this split,
/// forward-porting sc-18200 to a tree containing sc-15017 would refuse precisely the right LoRAs on
/// SCAIL-2 and admit the weaker ones (sc-18200 forward-port).
///
/// Both size classes still gate: `is_wan_14b_class_id` keeps the 5B TI2V base out either way, which
/// is the split the base-model gate was written for.
fn extra_compatible_backbone_is_image_conditioned(normalized_family: &str) -> bool {
    match normalized_family {
        // Wan2.1-I2V-14B-derived: 40 blocks × dim 5120 with the I2V cross-attention stack intact.
        "scail2" => true,
        // Wan 2.1 T2V 14B weight-for-weight — no image cross-attention at any width.
        "krea-realtime" => false,
        _ => false,
    }
}

/// A LoRA id for error messages: `id` / `loraId` / `lora_<n>`.
fn lora_display_id(lora: &Value, index: usize) -> String {
    lora.get("id")
        .or_else(|| lora.get("loraId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("lora_{}", index + 1))
}

/// Validate every LoRA in `loras` against `model_family` before a job runs
/// (Python `validate_lora_compatibility`). Errors on a declared family the model
/// cannot load, or — for a base-model-gated family (Wan) — a recorded base model
/// that differs from `model_id`. A LoRA that declares no family is skipped (the
/// user vouches for it). Returns the user-facing message as `Err`.
pub fn validate_lora_compatibility(
    loras: &[Value],
    model_family: Option<&str>,
    adapter_id: &str,
    model_id: Option<&str>,
) -> Result<(), String> {
    let normalized_model_family = model_family.map(normalize_model_family).unwrap_or_default();
    let accepted = model_family.map(accepted_lora_families).unwrap_or_default();
    if loras.is_empty() || accepted.is_empty() {
        return Ok(());
    }
    for (index, lora) in loras.iter().enumerate() {
        let families = lora_declared_families(lora);
        if families.is_empty() {
            continue;
        }
        let lora_id = lora_display_id(lora, index);
        // Accept when any declared family is one the model can load.
        if !families.iter().any(|family| accepted.contains(family)) {
            return Err(format!(
                "LoRA {lora_id} is not compatible with model family {normalized_model_family} for {adapter_id}."
            ));
        }
        // Base-model gating (Wan 5B vs 14B): a LoRA that records its trained base
        // model only applies to that exact model; one without falls back to family.
        if let Some(model_id) = model_id {
            if families
                .iter()
                .any(|family| is_base_model_gated_family(family))
            {
                if let Some(base) = lora_base_model(lora) {
                    // `base_model_satisfies_gate` keeps the 5B-vs-14B split while letting a
                    // Wan-14B LoRA through on a model that accepts `wan-video` via the registry
                    // (sc-15017) — its base can never equal that model's own id.
                    if !base_model_satisfies_gate(model_family.unwrap_or_default(), model_id, &base)
                    {
                        return Err(format!(
                            "LoRA {lora_id} was trained for base model {base}, not {model_id}; \
                             Wan 5B and 14B LoRAs are not interchangeable."
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn header_from_keys(keys: &[&str]) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        for key in keys {
            object.insert(
                (*key).to_owned(),
                json!({"dtype": "F16", "shape": [16, 1024], "data_offsets": [0, 32768]}),
            );
        }
        Value::Object(object)
    }

    fn write_safetensors(path: &Path, keys: &[String]) {
        // Emit a minimal valid safetensors layout: 8-byte little-endian header
        // length, then a JSON header whose entries each point at empty tensor
        // slices in the (empty) tensor section. The detector only reads the
        // header, so empty offsets are fine.
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        for key in keys {
            header.insert(
                key.clone(),
                json!({"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        std::fs::write(path, buffer).expect("write safetensors");
    }

    fn diffusers_double_stream_keys(prefix: &str, block_count: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
            for module in ["img_mlp.net.0.proj", "txt_mlp.net.0.proj"] {
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
            keys.push(format!(
                "{prefix}.transformer_blocks.{block}.attn.add_q_proj.lora_A.weight"
            ));
            keys.push(format!(
                "{prefix}.transformer_blocks.{block}.attn.add_q_proj.lora_B.weight"
            ));
        }
        keys
    }

    #[test]
    fn detects_wan_video() {
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
                keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
                keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));
    }

    #[test]
    fn detects_diffusers_wan_video() {
        // The `wan_lora` trainer (epic 1949 sc-1952) saves diffusers-format keys
        // via WanPipeline.save_lora_weights: `transformer.blocks.<n>.attn1|attn2.to_*`
        // (not the native `self_attn`/`cross_attn`/`ffn` names). These must still
        // detect as wan-video so the inference loader gates them correctly.
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in [
                "attn1.to_q",
                "attn1.to_k",
                "attn1.to_v",
                "attn1.to_out.0",
                "attn2.to_q",
                "attn2.to_k",
                "attn2.to_v",
                "attn2.to_out.0",
            ] {
                keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
                keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));
    }

    #[test]
    fn detects_comfyui_native_wan_video() {
        // ComfyUI / native Wan LoRAs prefix every block with
        // `diffusion_model.blocks.<n>.` and keep the native self_attn/cross_attn/ffn
        // module names. These contain `.blocks.` but not `transformer.blocks.`, so
        // the diffusers Wan signature misses them — the ComfyUI sibling must catch them.
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in [
                "self_attn.q",
                "self_attn.k",
                "self_attn.v",
                "self_attn.o",
                "cross_attn.q",
                "cross_attn.k",
                "ffn.0",
                "ffn.2",
            ] {
                keys.push(format!(
                    "diffusion_model.blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "diffusion_model.blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));
    }

    #[test]
    fn detects_kohya_wan_video() {
        // Kohya / musubi-tuner Wan LoRAs flatten the path into underscore-delimited
        // keys with a `lora_unet_blocks_<n>_` prefix and no text-encoder keys.
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in [
                "self_attn_q",
                "self_attn_k",
                "self_attn_v",
                "self_attn_o",
                "cross_attn_q",
                "cross_attn_k",
                "ffn_0",
                "ffn_2",
            ] {
                keys.push(format!(
                    "lora_unet_blocks_{block}_{module}.lora_down.weight"
                ));
                keys.push(format!("lora_unet_blocks_{block}_{module}.lora_up.weight"));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));
    }

    #[test]
    fn detects_flux() {
        let mut keys = Vec::new();
        for block in 0..19 {
            keys.push(format!(
                "transformer.transformer_blocks.{block}.attn.to_q.lora_A.weight"
            ));
            keys.push(format!(
                "transformer.transformer_blocks.{block}.attn.to_q.lora_B.weight"
            ));
        }
        for block in 0..38 {
            keys.push(format!(
                "transformer.single_transformer_blocks.{block}.proj_mlp.lora_A.weight"
            ));
            keys.push(format!(
                "transformer.single_transformer_blocks.{block}.proj_mlp.lora_B.weight"
            ));
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    #[test]
    fn detects_kohya_flux() {
        let mut keys = Vec::new();
        for block in 0..19 {
            for module in ["img_mlp_0", "txt_mlp_0", "img_attn_qkv", "txt_attn_qkv"] {
                keys.push(format!(
                    "lora_unet_double_blocks_{block}_{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "lora_unet_double_blocks_{block}_{module}.lora_up.weight"
                ));
            }
        }
        for block in 0..38 {
            keys.push(format!(
                "lora_unet_single_blocks_{block}_linear1.lora_down.weight"
            ));
            keys.push(format!(
                "lora_unet_single_blocks_{block}_linear1.lora_up.weight"
            ));
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    #[test]
    fn detects_metadata_family_before_keys() {
        let mut header = header_from_keys(&[
            "lora_unet_double_blocks_0_img_mlp_0.lora_down.weight",
            "lora_unet_double_blocks_0_img_mlp_0.lora_up.weight",
        ]);
        header["__metadata__"] = json!({
            "ss_base_model_version": "flux1",
            "modelspec.architecture": "flux-1-dev/lora"
        });

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    #[test]
    fn chroma_metadata_distinguishes_chroma_from_flux_keys() {
        // Chroma is FLUX.1-schnell-derived: its LoRA tensor keys are identical to
        // Flux (single/double transformer blocks), so only metadata can mark a
        // LoRA as chroma. Metadata is checked before keys and chroma before flux.
        let mut header = header_from_keys(&[
            "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        ]);
        header["__metadata__"] = json!({
            "modelspec.architecture": "chroma/lora"
        });

        assert_eq!(detect_lora_family(&header).as_deref(), Some("chroma"));
    }

    #[test]
    fn ideogram_metadata_detects_ideogram_family() {
        // An Ideogram 4 LoKr (ai-toolkit) stamps `ss_base_model_version: "ideogram4"`.
        // Its MMDiT keys (`diffusion_model.layers.<n>.attention.qkv`) match no bucket
        // signature, so without the metadata branch it detects as `None` — the failure
        // that let an Ideogram LoRA import into a krea_2 folder. Metadata is read first
        // and yields the canonical `ideogram` family token.
        let mut header = header_from_keys(&[
            "diffusion_model.layers.0.attention.qkv.lokr_w1",
            "diffusion_model.layers.0.attention.qkv.lokr_w2",
            "diffusion_model.layers.0.adaln_modulation.lokr_w2",
        ]);
        header["__metadata__"] = json!({
            "ss_base_model_version": "ideogram4"
        });

        assert_eq!(detect_lora_family(&header).as_deref(), Some("ideogram"));
    }

    #[test]
    fn ideogram_keys_detect_without_metadata() {
        // A metadata-less Ideogram export (no `ss_base_model_version`) must still
        // detect from its `diffusion_model.layers.<n>.` MMDiT keys via the bucket
        // scorer, not fall through to None.
        let mut keys = Vec::new();
        for layer in 0..6 {
            keys.push(format!(
                "diffusion_model.layers.{layer}.attention.qkv.lokr_w2"
            ));
            keys.push(format!(
                "diffusion_model.layers.{layer}.attention.o.lokr_w2"
            ));
            keys.push(format!(
                "diffusion_model.layers.{layer}.feed_forward.w1.lokr_w2"
            ));
            keys.push(format!(
                "diffusion_model.layers.{layer}.adaln_modulation.lokr_w2"
            ));
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("ideogram"));
    }

    #[test]
    fn detects_xflux_double_blocks_only() {
        // XLabs / x-flux realism-style LoRAs adapt only the double-stream blocks
        // (no single blocks, no metadata) via the attention-processor layout.
        let mut keys = Vec::new();
        for block in 0..19 {
            for module in ["qkv_lora1", "qkv_lora2", "proj_lora1", "proj_lora2"] {
                keys.push(format!(
                    "double_blocks.{block}.processor.{module}.down.weight"
                ));
                keys.push(format!(
                    "double_blocks.{block}.processor.{module}.up.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    fn flux2_klein_native_keys() -> Vec<String> {
        // Mirrors the real klein_9B_Turbo_r128.safetensors layout: native/ComfyUI
        // `diffusion_model.` prefix, 8 double blocks + 24 single blocks, and FLUX.2's
        // shared (per-stream, not per-block) modulation tensors.
        let mut keys = Vec::new();
        for block in 0..8 {
            for module in [
                "img_attn.proj",
                "img_attn.qkv",
                "txt_attn.proj",
                "txt_attn.qkv",
            ] {
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_up.weight"
                ));
            }
            for module in ["img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2"] {
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        for block in 0..24 {
            for module in ["linear1", "linear2"] {
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        for module in [
            "double_stream_modulation_img.lin",
            "double_stream_modulation_txt.lin",
            "single_stream_modulation.lin",
            "img_in",
            "txt_in",
        ] {
            keys.push(format!("diffusion_model.{module}.lora_down.weight"));
            keys.push(format!("diffusion_model.{module}.lora_up.weight"));
        }
        keys
    }

    #[test]
    fn detects_flux2_klein_native() {
        let keys = flux2_klein_native_keys();
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux2"));
    }

    #[test]
    fn flux1_native_keys_do_not_detect_as_flux2() {
        // A FLUX.1 native LoRA shares the double_blocks/single_blocks split but keeps
        // PER-BLOCK modulation (`img_mod`/`txt_mod`/`modulation`) and has none of the
        // shared `*_stream_modulation` tensors, so it must not be misread as FLUX.2.
        let mut keys = Vec::new();
        for block in 0..19 {
            for module in ["img_attn.qkv", "txt_attn.qkv", "img_mod.lin", "txt_mod.lin"] {
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        for block in 0..38 {
            for module in ["linear1", "modulation.lin"] {
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_ne!(detect_lora_family(&header).as_deref(), Some("flux2"));
    }

    #[test]
    fn flux2_metadata_wins_over_generic_flux_substring() {
        // FLUX.2 architecture strings contain "flux"; the flux2 metadata branch must
        // claim them before the generic flux match (mirrors chroma-before-flux).
        for arch in ["flux2", "flux-2", "flux.2-klein", "FLUX_2/lora"] {
            let mut header = header_from_keys(&[
                "lora_unet_double_blocks_0_img_mlp_0.lora_down.weight",
                "lora_unet_double_blocks_0_img_mlp_0.lora_up.weight",
            ]);
            header["__metadata__"] = json!({ "modelspec.architecture": arch });
            assert_eq!(
                detect_lora_family(&header).as_deref(),
                Some("flux2"),
                "architecture {arch} should map to flux2"
            );
        }
    }

    #[test]
    fn ai_toolkit_flux2_klein_base_model_version_detects_flux2() {
        // Real ai-toolkit klein LoRAs (e.g. V3_flux_klein.safetensors) train only
        // attn/mlp/linear — no `*_stream_modulation` tensors — so the key signature
        // can't fire, but they carry `ss_base_model_version: "flux2_klein_9b"`. That
        // value contains "flux", so the generic flux branch used to swallow it; the
        // flux2 branch must claim it first.
        let mut header = header_from_keys(&[
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "diffusion_model.single_blocks.0.linear1.lora_A.weight",
        ]);
        header["__metadata__"] = json!({ "ss_base_model_version": "flux2_klein_9b" });

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux2"));
    }

    #[test]
    fn flux_family_maps_to_flux_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("flux"), Some("flux_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "flux"),
            vec!["text_to_image", "style_variations"],
        );
    }

    #[test]
    fn chroma_family_maps_to_chroma_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("chroma"), Some("chroma_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "chroma"),
            vec!["text_to_image", "style_variations"],
        );
    }

    #[test]
    fn kolors_family_maps_to_kolors_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("kolors"), Some("kolors_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "kolors"),
            vec!["text_to_image", "character_image", "style_variations"],
        );
    }

    #[test]
    fn sdxl_family_maps_to_sdxl_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("sdxl"), Some("sdxl_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "sdxl"),
            vec!["text_to_image", "edit_image", "style_variations"],
        );
    }

    #[test]
    fn svd_family_maps_to_svd_video_adapter_and_image_to_video_only() {
        assert_eq!(model_adapter_for_family("svd"), Some("svd_video"));
        // SVD is image-conditioned only — no text-to-video or timeline modes.
        assert_eq!(
            model_capabilities_for_type_and_family("video", "svd"),
            vec!["image_to_video"],
        );
    }

    #[test]
    fn bernini_family_maps_to_bernini_adapter_and_full_video_task_surface() {
        assert_eq!(model_adapter_for_family("bernini"), Some("bernini"));
        // sc-4703 / sc-5425: the full Bernini video task surface — t2v + the editing
        // (`video_to_video`), reference-driven (`reference_to_video` /
        // `reference_video_to_video`), and multi-source (`multi_video_to_video` /
        // `ads2v`) modes. No still-image-to-video (renderer is T2V).
        assert_eq!(
            model_capabilities_for_type_and_family("video", "bernini"),
            vec![
                "text_to_video",
                "video_to_video",
                "reference_to_video",
                "reference_video_to_video",
                "multi_video_to_video",
                "ads2v",
            ],
        );
        // sc-5424: the image-typed companion (`bernini_image`) shares the `bernini`
        // family/adapter but exposes only the still tasks — t2i + i2i (`edit_image`).
        assert_eq!(
            model_capabilities_for_type_and_family("image", "bernini"),
            vec!["text_to_image", "edit_image"],
        );
    }

    #[test]
    fn detects_ltx_video() {
        let mut keys = Vec::new();
        for block in 0..28 {
            for module in ["attn1.to_q", "attn1.to_k", "attn2.to_q", "ff.net.0.proj"] {
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("ltx-video"));
    }

    /// The real `ltx-2.3-22b-distilled-lora-*` export: a `diffusion_model.` prefix
    /// (LTX-2 native / ComfyUI form, not the diffusers `transformer.` prefix) plus
    /// LTX-2 2.3's cross-modal audio attention, whose gating tensors are named
    /// `..._attn.to_gate_logits...`. That `attn.to_gate` substring previously tripped
    /// the Krea-2 unique-key detector (which now requires a `to_gate.` module
    /// boundary), and the diffusers-only LTX prefix meant the bucket scorer never
    /// recognized it either — so the LoRA was mis-detected as krea_2 and rejected from
    /// every LTX model. Both halves must hold: not krea_2, and positively ltx-video.
    #[test]
    fn detects_ltx2_native_with_gated_audio_attention() {
        let mut keys = Vec::new();
        for block in 0..28 {
            for module in [
                "attn1.to_q",
                "attn1.to_gate_logits",
                "attn2.to_q",
                "audio_to_video_attn.to_gate_logits",
                "video_to_audio_attn.to_gate_logits",
                "ff.net.0.proj",
            ] {
                keys.push(format!(
                    "diffusion_model.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "diffusion_model.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("ltx-video"));
    }

    #[test]
    fn detects_qwen_image_by_block_count() {
        let keys = diffusers_double_stream_keys("transformer", 60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    #[test]
    fn low_mm_dit_block_count_is_inconclusive() {
        let keys = diffusers_double_stream_keys("transformer", 24);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert!(detect_lora_family(&header).is_none());
    }

    // ---- Mage-Flow (epic 14034 / sc-14057) ---------------------------------------------------

    /// A community "all-linear" Mage-Flow adapter with no metadata. Its module names are spelled
    /// identically to Qwen-Image's, so the ONLY tensor-level evidence is the published `depth: 12`
    /// — block indices `0..=11`, complete and with nothing above.
    #[test]
    fn detects_mage_flow_by_its_exact_twelve_block_span() {
        let keys = diffusers_double_stream_keys("transformer", 12);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("mage-flow"));
    }

    /// The kohya / LyCORIS flattening of the same file (`lora_unet_transformer_blocks_0_…`) must
    /// reach the same answer — the second MMDiT signature and `parse_block_index`'s `_` separator
    /// both have to hold for a third-party export spelling.
    #[test]
    fn detects_mage_flow_in_the_kohya_flattened_spelling() {
        let mut keys = Vec::new();
        for block in 0..12 {
            for module in [
                "attn_to_q",
                "attn_to_k",
                "attn_to_v",
                "attn_add_q_proj",
                "img_mlp_net_0_proj",
                "txt_mlp_net_2",
            ] {
                for role in ["lora_down.weight", "lora_up.weight"] {
                    keys.push(format!(
                        "lora_unet_transformer_blocks_{block}_{module}.{role}"
                    ));
                }
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("mage-flow"));
    }

    /// LoKr, not LoRA: the Mage trainer advertises `supports_lokr`, and a LoKr file carries
    /// `lokr_w1`/`lokr_w2` at the same module paths. Detection keys on the module path, so the
    /// adapter kind must not change the answer.
    #[test]
    fn detects_mage_flow_lokr() {
        let mut keys = Vec::new();
        for block in 0..12 {
            for module in ["attn.to_q", "attn.add_k_proj", "img_mlp.net.0.proj"] {
                for factor in ["lokr_w1", "lokr_w2"] {
                    keys.push(format!(
                        "transformer.transformer_blocks.{block}.{module}.{factor}"
                    ));
                }
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("mage-flow"));
    }

    /// The full-span requirement is what keeps the 12-block signal honest. A sparse adapter that
    /// tops out at block 11 without covering every block below stays inconclusive rather than
    /// claiming a family it cannot prove — drop the `blocks.len()` check and this goes green,
    /// which is exactly the false-confidence this guards.
    #[test]
    fn a_partial_span_below_twelve_blocks_is_not_claimed_as_mage_flow() {
        for present in [vec![0usize, 1, 2, 11], (0..7).collect::<Vec<_>>(), vec![11]] {
            let mut keys = Vec::new();
            for block in &present {
                for module in ["attn.to_q", "attn.add_q_proj", "img_mlp.net.0.proj"] {
                    for role in ["lora_A.weight", "lora_B.weight"] {
                        keys.push(format!(
                            "transformer.transformer_blocks.{block}.{module}.{role}"
                        ));
                    }
                }
            }
            let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

            assert_eq!(
                detect_lora_family(&header),
                None,
                "a partial {present:?} span must stay inconclusive, not resolve to a family"
            );
        }
    }

    /// The other direction of the same boundary: the deeper MMDiT siblings must never land on
    /// Mage. Qwen-Image (60 blocks) stays qwen; a 30-block Z-Image-shaped span stays inconclusive.
    #[test]
    fn deeper_mm_dit_siblings_are_never_mistaken_for_mage_flow() {
        for depth in [30usize, 60] {
            let keys = diffusers_double_stream_keys("transformer", depth);
            let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
            assert_ne!(
                detect_lora_family(&header).as_deref(),
                Some("mage-flow"),
                "a {depth}-block MMDiT adapter must not detect as mage-flow"
            );
        }
        let qwen = diffusers_double_stream_keys("transformer", 60);
        let header = header_from_keys(&qwen.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    /// Ideogram's LoKr is the family that has been mis-detected before (sc-4725 / the
    /// `text_fusion` boundary regression). It must resolve to `ideogram`, never to Mage — its
    /// `diffusion_model.layers.<n>.` blocks are a different grammar entirely, and its 34 layers
    /// would be a 12-block miss anyway.
    #[test]
    fn an_ideogram_lokr_is_not_mistaken_for_mage_flow() {
        let mut keys = Vec::new();
        for layer in 0..34 {
            for module in [
                "attention.qkv",
                "attention.out",
                "adaln_modulation",
                "feed_forward.w1",
            ] {
                for factor in ["lokr_w1", "lokr_w2"] {
                    keys.push(format!("diffusion_model.layers.{layer}.{module}.{factor}"));
                }
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        let detected = detect_lora_family(&header);
        assert_ne!(detected.as_deref(), Some("mage-flow"));
        assert_eq!(detected.as_deref(), Some("ideogram"));
    }

    /// The metadata path: every spelling a Mage adapter's provenance can arrive in — our MLX
    /// trainer's `family: "mage_flow"` stamp, the catalog/training-base ids, the upstream and
    /// re-hosted repos, and a bare file stem.
    #[test]
    fn mage_flow_metadata_stamps_detect_without_tensor_evidence() {
        for (key, value) in [
            ("family", "mage_flow"),
            ("baseModel", "mage_flow_base"),
            ("baseModel", "mage_flow_edit_base"),
            ("ss_base_model_version", "mage-flow"),
            ("ss_base_model_version", "Mage"),
            ("modelspec.architecture", "microsoft/Mage-Flow-Turbo"),
            ("modelspec.implementation", "SceneWorks/Mage-Flow-Base"),
        ] {
            let mut object = serde_json::Map::new();
            object.insert("__metadata__".to_owned(), json!({ key: value }));
            object.insert(
                "some.module.lora_A.weight".to_owned(),
                json!({"dtype": "BF16", "shape": [8, 1024], "data_offsets": [0, 16384]}),
            );
            assert_eq!(
                detect_lora_family(&Value::Object(object)).as_deref(),
                Some("mage-flow"),
                "{key} = {value:?} should resolve to mage-flow"
            );
        }
    }

    /// 🔴 The trap that makes the Mage metadata arm dangerous: **"mage" is a substring of
    /// "image"**. A `contains("mage")` test — the shape `flux` / `krea` / `ideogram` use — would
    /// swallow every Z-Image and Qwen-Image adapter and hard-reject it from its own model. Loosen
    /// the token-boundary match and this test goes red on the first case.
    ///
    /// The `*-flow` rows are the ones that discriminate the *arm's own* spellings: `contains`ing
    /// the three flow forms (`mage-flow` / `mage_flow` / `mageflow`) reintroduces the exact hazard
    /// the bare-token rows already defend, because `z-image-flow`, `qwen_image_flow` and
    /// `imageflow` each end in one of them. They sit ahead of the z-image and qwen-image arms, so a
    /// hit there is a confident *mislabel* — worse than an inconclusive result — and the adapter is
    /// then hard-rejected from its own model.
    #[test]
    fn image_family_metadata_is_not_swallowed_by_the_mage_substring() {
        for (value, expected) in [
            ("z-image", "z-image"),
            ("zimage", "z-image"),
            ("Z-Image-Turbo", "z-image"),
            ("qwen-image", "qwen-image"),
            ("Qwen-Image-Edit-2509", "qwen-image"),
            ("Qwen/Qwen_Image", "qwen-image"),
            // …and the `*-flow` spellings, which a `contains("mage-flow")` test swallows whole.
            ("z-image-flow", "z-image"),
            ("Tongyi/Z-Image_Flow", "z-image"),
            ("qwen_image_flow", "qwen-image"),
        ] {
            assert_eq!(
                metadata_value_to_family(value).as_deref(),
                Some(expected),
                "{value:?} must stay {expected}, not be captured by the mage arm"
            );
        }
        // An unaffiliated `*imageflow*` name belongs to no family we know. It must stay
        // **inconclusive** rather than being claimed by Mage: `None` leaves the adapter usable
        // wherever the user says it belongs, a wrong family hard-rejects it everywhere.
        for value in ["imageflow", "ImageFlow-XL"] {
            assert_eq!(
                metadata_value_to_family(value),
                None,
                "{value:?} must stay inconclusive, not be captured by the mage arm"
            );
        }
        // And the same at the whole-header level, so ordering inside the arm chain is covered too.
        let mut object = serde_json::Map::new();
        object.insert(
            "__metadata__".to_owned(),
            json!({ "ss_base_model_version": "z-image" }),
        );
        object.insert(
            "transformer_blocks.0.attn.to_q.lora_A.weight".to_owned(),
            json!({"dtype": "BF16", "shape": [8, 1024], "data_offsets": [0, 16384]}),
        );
        assert_eq!(
            detect_lora_family(&Value::Object(object)).as_deref(),
            Some("z-image")
        );
    }

    /// A Mage adapter is offered on Mage models and nowhere else: the detected token is the
    /// verbatim `mage-flow` the catalog's `loraCompatibility.families` and the
    /// `mage_flow_*_lora` training targets declare, it survives canonicalization unchanged, and
    /// the compatibility gate accepts it on `mage-flow` while rejecting it everywhere else.
    #[test]
    fn a_mage_flow_lora_is_accepted_only_on_mage_flow_models() {
        assert_eq!(canonical_lora_family("mage-flow"), "mage-flow");
        assert_eq!(canonical_lora_family("mage_flow"), "mage-flow");
        assert_eq!(accepted_lora_families("mage-flow"), vec!["mage-flow"]);

        let lora = json!({ "id": "mage_style", "family": "mage-flow" });
        assert!(validate_lora_compatibility(
            std::slice::from_ref(&lora),
            Some("mage-flow"),
            "mage_style",
            Some("mage_flow_base")
        )
        .is_ok());
        for other in ["qwen-image", "z-image", "krea_2", "flux2"] {
            assert!(
                validate_lora_compatibility(
                    std::slice::from_ref(&lora),
                    Some(other),
                    "mage_style",
                    Some("some_model")
                )
                .is_err(),
                "a mage-flow LoRA must be refused on a {other} model"
            );
        }
        // …and the reverse: a sibling MMDiT family's LoRA is refused on a Mage model.
        for other in ["qwen-image", "z-image"] {
            let foreign = json!({ "id": "other", "family": other });
            assert!(
                validate_lora_compatibility(
                    &[foreign],
                    Some("mage-flow"),
                    "other",
                    Some("mage_flow_base")
                )
                .is_err(),
                "a {other} LoRA must be refused on a mage-flow model"
            );
        }
    }

    // ---- adapter `__metadata__` rank / alpha (sc-14057) ---------------------------------------

    fn header_with_metadata(metadata: Value) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("__metadata__".to_owned(), metadata);
        object.insert(
            "transformer_blocks.0.attn.to_q.lora_A.weight".to_owned(),
            json!({"dtype": "BF16", "shape": [8, 1024], "data_offsets": [0, 16384]}),
        );
        Value::Object(object)
    }

    #[test]
    fn reads_rank_and_alpha_from_the_sceneworks_trainer_stamp() {
        // The MLX/candle trainers write string values (the safetensors metadata map is
        // string-valued), and alpha may legitimately differ from rank.
        let header = header_with_metadata(json!({
            "networkType": "lokr",
            "rank": "16",
            "alpha": "32",
        }));
        let meta = read_adapter_metadata(&header);
        assert_eq!(meta.network_type.as_deref(), Some("lokr"));
        assert_eq!(meta.rank, Some(16));
        assert_eq!(meta.alpha, Some(32.0));
    }

    #[test]
    fn reads_rank_and_alpha_from_the_kohya_and_peft_conventions() {
        let kohya = header_with_metadata(json!({
            "ss_network_module": "networks.lora",
            "ss_network_dim": "64",
            "ss_network_alpha": "32",
        }));
        let meta = read_adapter_metadata(&kohya);
        assert_eq!(meta.network_type.as_deref(), Some("lora"));
        assert_eq!(meta.rank, Some(64));
        assert_eq!(meta.alpha, Some(32.0));

        // LyCORIS under kohya: `ss_network_module` names no algo (`lycoris.kohya`), so the type
        // comes from the `ss_network_args` blob — this is how a third-party Mage **LoKr** arrives.
        let lycoris = header_with_metadata(json!({
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": "{\"algo\": \"lokr\", \"factor\": \"-1\"}",
            "ss_network_dim": "16",
        }));
        let meta = read_adapter_metadata(&lycoris);
        assert_eq!(meta.network_type.as_deref(), Some("lokr"));
        assert_eq!(meta.rank, Some(16));
        assert_eq!(meta.alpha, None);

        // …and with no algo to read, nothing is guessed.
        let bare_lycoris = header_with_metadata(json!({ "ss_network_module": "lycoris.kohya" }));
        assert_eq!(read_adapter_metadata(&bare_lycoris).network_type, None);

        // diffusers `save_lora_adapter` / raw PEFT: no top-level keys at all, real JSON numbers
        // inside the `lora_adapter_metadata` blob (sc-5513).
        let peft = header_with_metadata(json!({
            "format": "pt",
            "lora_adapter_metadata": "{\"r\": 8, \"lora_alpha\": 16, \"target_modules\": [\"to_q\"]}",
        }));
        let meta = read_adapter_metadata(&peft);
        assert_eq!(meta.rank, Some(8));
        assert_eq!(meta.alpha, Some(16.0));
        assert_eq!(meta.network_type, None);
    }

    /// 🔴 The omitted case, handled explicitly: a file that declares no rank/alpha records NONE of
    /// them. Nothing may back-fill `alpha` from `rank` (the loader's last-resort scaling fallback
    /// is not a fact about the file) or invent a rank from a factor shape.
    #[test]
    fn an_adapter_that_declares_no_rank_or_alpha_records_neither() {
        for metadata in [
            json!({ "format": "pt" }),
            json!({}),
            // rank present, alpha absent — alpha must NOT inherit rank.
            json!({ "networkType": "lora", "rank": "8" }),
        ] {
            let declares_rank = metadata.get("rank").is_some();
            let meta = read_adapter_metadata(&header_with_metadata(metadata.clone()));
            assert_eq!(meta.alpha, None, "alpha must stay absent for {metadata}");
            if !declares_rank {
                assert_eq!(meta.rank, None, "rank must stay absent for {metadata}");
                assert!(meta.is_empty(), "{metadata} should read as no declaration");
            } else {
                assert_eq!(meta.rank, Some(8));
            }
        }
        // A header with no `__metadata__` block at all.
        let bare = header_from_keys(&["transformer_blocks.0.attn.to_q.lora_A.weight"]);
        let mut object = bare.as_object().cloned().unwrap();
        object.remove("__metadata__");
        assert!(read_adapter_metadata(&Value::Object(object)).is_empty());
    }

    /// A non-positive or unparseable rank can only poison a scaling denominator, so it is treated
    /// as absent. A zero *alpha* is a legitimate scale-0 adapter and is kept.
    #[test]
    fn a_degenerate_rank_is_dropped_but_a_zero_alpha_is_kept() {
        let meta = read_adapter_metadata(&header_with_metadata(json!({
            "rank": "0",
            "alpha": "0",
        })));
        assert_eq!(meta.rank, None);
        assert_eq!(meta.alpha, Some(0.0));

        let meta = read_adapter_metadata(&header_with_metadata(json!({ "rank": "not-a-number" })));
        assert_eq!(meta.rank, None);
    }

    // ---- the post-download inspector + manifest writer (sc-14057) ------------------------------

    /// Write a minimal valid safetensors file carrying `metadata` and `keys`.
    fn write_adapter_file(path: &Path, metadata: &str, keys: &[&str]) {
        let tensors: Vec<String> = keys
            .iter()
            .map(|key| format!(r#""{key}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#))
            .collect();
        let header = format!(r#"{{"__metadata__":{metadata},{}}}"#, tensors.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        fs::write(path, bytes).expect("write adapter");
    }

    /// 🔴 The ingest-route asymmetry this closes: the API records `networkType`/`rank`/`alpha` from
    /// the *source file*, which a URL/repo import does not have when the job is queued. The worker
    /// re-reads the landed download through this dir-level inspector, so the same adapter is
    /// described identically whichever route brought it in.
    #[test]
    fn the_post_download_inspector_reads_family_and_declared_metadata_from_a_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_adapter_file(
            &dir.path().join("adapter.safetensors"),
            r#"{"family":"mage_flow","networkType":"lokr","rank":"16","alpha":"32"}"#,
            &["transformer_blocks.0.attn.to_q.lokr_w1"],
        );

        let (family, meta) = inspect_adapter_in_dir(dir.path()).expect("inspect");
        assert_eq!(family.as_deref(), Some("mage-flow"));
        assert_eq!(meta.network_type.as_deref(), Some("lokr"));
        assert_eq!(meta.rank, Some(16));
        assert_eq!(meta.alpha, Some(32.0));

        // A directory with no safetensors is not an error — it is simply no evidence.
        let empty = tempfile::tempdir().expect("tempdir");
        let (family, meta) = inspect_adapter_in_dir(empty.path()).expect("inspect");
        assert_eq!(family, None);
        assert!(meta.is_empty());
    }

    /// The URL/repo route end to end at the manifest seam: the entry the API queued carries no
    /// declared metadata (no file existed yet), and the writer fills in exactly what the downloaded
    /// adapter states — nothing more.
    #[test]
    fn a_url_route_manifest_entry_gains_the_downloaded_adapters_declared_metadata() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_adapter_file(
            &dir.path().join("adapter.safetensors"),
            r#"{"networkType":"lokr","rank":"16"}"#,
            &["transformer_blocks.0.attn.to_q.lokr_w1"],
        );

        // Exactly the shape `queue_lora_import_job` emits for a `sourceUrl` import: no
        // `networkType` / `rank` / `alpha`, because `read_adapter_metadata` had no file to read.
        let mut entry = json!({
            "id": "mage_flow_realism",
            "name": "Realism",
            "source": { "provider": "url", "url": "https://example.test/realism.safetensors" },
        })
        .as_object()
        .cloned()
        .expect("object");
        assert!(!entry.contains_key("networkType"));

        let (_, meta) = inspect_adapter_in_dir(dir.path()).expect("inspect");
        apply_adapter_metadata_to_manifest_entry(&mut entry, &meta);

        assert_eq!(
            entry.get("networkType").and_then(Value::as_str),
            Some("lokr")
        );
        assert_eq!(entry.get("rank").and_then(Value::as_u64), Some(16));
        // 🔴 The file declares no alpha, so the entry records none — it does NOT inherit `rank`.
        assert!(
            !entry.contains_key("alpha"),
            "alpha must stay absent, not inherit rank: {entry:?}"
        );
    }

    /// The writer never rewrites a value the entry already carries — matching the `family`
    /// reconcile it sits beside. A local-file import that already recorded these facts at queue
    /// time keeps them when the worker re-reads the copied file.
    #[test]
    fn the_manifest_writer_does_not_overwrite_already_recorded_facts() {
        let mut entry = json!({ "networkType": "lora", "rank": 8, "alpha": 4.0 })
            .as_object()
            .cloned()
            .expect("object");
        apply_adapter_metadata_to_manifest_entry(
            &mut entry,
            &AdapterFileMetadata {
                network_type: Some("lokr".to_owned()),
                rank: Some(64),
                alpha: Some(128.0),
            },
        );
        assert_eq!(
            entry.get("networkType").and_then(Value::as_str),
            Some("lora")
        );
        assert_eq!(entry.get("rank").and_then(Value::as_u64), Some(8));
        assert_eq!(entry.get("alpha").and_then(Value::as_f64), Some(4.0));

        // …and an adapter that declares nothing adds no keys at all.
        let mut entry = json!({ "id": "x" }).as_object().cloned().expect("object");
        apply_adapter_metadata_to_manifest_entry(&mut entry, &AdapterFileMetadata::default());
        assert_eq!(
            entry.len(),
            1,
            "no declaration must write no keys: {entry:?}"
        );
    }

    #[test]
    fn detects_krea_projector_scale_lora() {
        // The real failing upload: a Krea 2 `text_fusion.projector` scale LoRA in
        // diffusers format ships only two tensors and empty metadata. It can never
        // reach `MIN_KEY_MATCHES`, but `text_fusion` is krea-unique, so the
        // unique-key pre-check identifies it. The label must be the verbatim
        // `krea_2` the catalog/trainer use (import reconciliation compares it raw).
        let header = header_from_keys(&[
            "transformer.text_fusion.projector.lora_A.weight",
            "transformer.text_fusion.projector.lora_B.weight",
        ]);

        assert_eq!(detect_lora_family(&header).as_deref(), Some("krea_2"));
    }

    #[test]
    fn detects_krea_by_gated_attention() {
        // A community krea LoRA that adapts the gated single-stream attention carries
        // `attn.to_gate`, a projection no other family exposes.
        let mut keys = Vec::new();
        for block in 0..28 {
            for module in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_gate",
                "attn.to_out.0",
            ] {
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("krea_2"));
    }

    #[test]
    fn detects_krea_by_metadata() {
        let mut object = serde_json::Map::new();
        object.insert(
            "__metadata__".to_owned(),
            json!({"modelspec.architecture": "krea-2-raw"}),
        );
        object.insert(
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight".to_owned(),
            json!({"dtype": "F16", "shape": [16, 1024], "data_offsets": [0, 32768]}),
        );
        let header = Value::Object(object);

        assert_eq!(detect_lora_family(&header).as_deref(), Some("krea_2"));
    }

    #[test]
    fn detects_sceneworks_trained_krea_lora_by_family_stamp() {
        // A LoRA trained IN SceneWorks by the (candle/MLX) Krea trainer. Verified against real
        // on-disk output: the header stamps `family: krea_2` + `baseModel: krea_2_raw`, and the
        // default target set (`to_q`/`to_k`/`to_v`/`to_out.0`) produces bare `transformer_blocks.
        // <n>.attn.*` keys. Those keys hit NO bucket signature (bare `transformer_blocks.`, not the
        // dotted `transformer.transformer_blocks.` the buckets require) and carry no `text_fusion`/
        // `to_gate` unique key, so before this fix the LoRA was left undetected. The `family` stamp is
        // now the authoritative signal.
        let mut object = serde_json::Map::new();
        object.insert(
            "__metadata__".to_owned(),
            json!({
                "family": "krea_2",
                "baseModel": "krea_2_raw",
                "networkType": "lora",
                "rank": "8",
                "alpha": "8",
            }),
        );
        for block in 0..28 {
            for module in ["to_q", "to_k", "to_v", "to_out.0"] {
                for role in ["lora_A.weight", "lora_B.weight"] {
                    object.insert(
                        format!("transformer_blocks.{block}.attn.{module}.{role}"),
                        json!({"dtype": "BF16", "shape": [8, 1024], "data_offsets": [0, 16384]}),
                    );
                }
            }
        }
        let header = Value::Object(object);

        assert_eq!(detect_lora_family(&header).as_deref(), Some("krea_2"));

        // The `baseModel` id alone (no `family` stamp) also resolves via the architecture matcher —
        // covers older/third-party SceneWorks-lineage files that recorded only the trained base.
        let base_only = json!({
            "__metadata__": { "baseModel": "krea_2_raw" },
            "transformer_blocks.0.attn.to_q.lora_A.weight":
                {"dtype": "BF16", "shape": [8, 1024], "data_offsets": [0, 16384]},
        });
        assert_eq!(detect_lora_family(&base_only).as_deref(), Some("krea_2"));
    }

    // ---- Anima (epic 10512, sc-10521) --------------------------------------------------------------

    /// The 448-target DiT surface (16 per block) as the official Anima LoRAs spell it (ComfyUI
    /// `diffusion_model.` prefix, PEFT `lora_A`/`lora_B`, original Cosmos module names).
    fn anima_dit_keys(blocks: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for b in 0..blocks {
            for attn in ["self_attn", "cross_attn"] {
                for proj in ["q_proj", "k_proj", "v_proj", "output_proj"] {
                    for role in ["lora_A.weight", "lora_B.weight"] {
                        keys.push(format!("diffusion_model.blocks.{b}.{attn}.{proj}.{role}"));
                    }
                }
            }
            for layer in ["mlp.layer1", "mlp.layer2"] {
                for role in ["lora_A.weight", "lora_B.weight"] {
                    keys.push(format!("diffusion_model.blocks.{b}.{layer}.{role}"));
                }
            }
            // The Cosmos adaLN-modulation down/up pairs — the Anima-unique discriminator.
            for adaln in [
                "adaln_modulation_self_attn",
                "adaln_modulation_cross_attn",
                "adaln_modulation_mlp",
            ] {
                for updown in ["1", "2"] {
                    for role in ["lora_A.weight", "lora_B.weight"] {
                        keys.push(format!(
                            "diffusion_model.blocks.{b}.{adaln}.{updown}.{role}"
                        ));
                    }
                }
            }
        }
        keys
    }

    /// The 60-target `AnimaTextConditioner` surface (10 per block) the turbo LoRA adds.
    fn anima_adapter_keys(blocks: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for b in 0..blocks {
            for attn in ["self_attn", "cross_attn"] {
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    for role in ["lora_A.weight", "lora_B.weight"] {
                        keys.push(format!(
                            "diffusion_model.llm_adapter.blocks.{b}.{attn}.{proj}.{role}"
                        ));
                    }
                }
            }
            for m in ["mlp.0", "mlp.2"] {
                for role in ["lora_A.weight", "lora_B.weight"] {
                    keys.push(format!("diffusion_model.llm_adapter.blocks.{b}.{m}.{role}"));
                }
            }
        }
        keys
    }

    #[test]
    fn detects_anima_style_lora_dit_only() {
        // `anima-greg-rutkowski-style` shape: 448 DiT targets, ZERO `llm_adapter` — so it is caught
        // by the `Bucket::Anima` signature on its Cosmos adaLN markers, NOT the unique-key path. A
        // signature that required `llm_adapter` would misclassify this file (the crux of sc-10521).
        let keys = anima_dit_keys(28);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(detect_lora_family(&header).as_deref(), Some("anima"));
    }

    #[test]
    fn detects_anima_turbo_lora_with_adapter() {
        // `anima-turbo-lora-v0.2` shape: 448 DiT + 60 `llm_adapter` = 508 targets. The `llm_adapter`
        // key identifies it via the unique-key fast path.
        let mut keys = anima_dit_keys(28);
        keys.extend(anima_adapter_keys(6));
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(detect_lora_family(&header).as_deref(), Some("anima"));
    }

    #[test]
    fn detects_sparse_anima_lokr_via_llm_adapter() {
        // A sparse LoKr that touches only a couple conditioner modules (below the MIN_KEY_MATCHES=4
        // marker floor) — the Ideogram-LoKr precedent where a below-floor adapter went undetected.
        // The `llm_adapter` unique key identifies it regardless of the floor. LyCORIS `lokr_*` factors.
        let header = header_from_keys(&[
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lokr_w1",
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lokr_w2",
        ]);
        assert_eq!(detect_lora_family(&header).as_deref(), Some("anima"));
    }

    #[test]
    fn anima_not_mistaken_for_wan_and_wan_still_detected() {
        // Anima and native/ComfyUI Wan SHARE the `diffusion_model.blocks.` prefix + `.self_attn.`/
        // `.cross_attn.` markers, so the collision must be resolved in Anima's favor for Anima keys...
        let anima = anima_dit_keys(28);
        let anima_header = header_from_keys(&anima.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(detect_lora_family(&anima_header).as_deref(), Some("anima"));
        assert_ne!(
            detect_lora_family(&anima_header).as_deref(),
            Some("wan-video")
        );

        // ...while a genuine native Wan LoRA (`.ffn.`, no Cosmos adaLN) still detects as wan-video.
        let mut wan = Vec::new();
        for b in 0..30 {
            for m in ["self_attn.q", "self_attn.k", "cross_attn.v", "ffn.0"] {
                for role in ["lora_A.weight", "lora_B.weight"] {
                    wan.push(format!("diffusion_model.blocks.{b}.{m}.{role}"));
                }
            }
        }
        let wan_header = header_from_keys(&wan.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(
            detect_lora_family(&wan_header).as_deref(),
            Some("wan-video")
        );
    }

    /// The Cosmos DiT attention/MLP surface WITHOUT the adaLN-modulation targets — the shape an
    /// ordinary (non-official) Anima LoRA has, and the one SceneWorks#1670 reports. `prefix_key`
    /// builds the export spelling under test so every prefix the Wan signatures accept is covered.
    fn anima_attention_only_keys(
        blocks: usize,
        prefix_key: fn(usize, &str, &str) -> String,
    ) -> Vec<String> {
        let mut keys = Vec::new();
        for b in 0..blocks {
            for attn in ["self_attn", "cross_attn"] {
                for proj in ["q_proj", "k_proj", "v_proj", "output_proj"] {
                    for role in ["lora_A.weight", "lora_B.weight"] {
                        keys.push(prefix_key(b, &format!("{attn}.{proj}"), role));
                    }
                }
            }
        }
        keys
    }

    fn comfy_dotted_key(block: usize, module: &str, role: &str) -> String {
        format!("diffusion_model.blocks.{block}.{module}.{role}")
    }

    fn diffusers_dotted_key(block: usize, module: &str, role: &str) -> String {
        format!("transformer.blocks.{block}.{module}.{role}")
    }

    fn kohya_flattened_key(block: usize, module: &str, role: &str) -> String {
        format!(
            "lora_unet_blocks_{block}_{}.{role}",
            module.replace('.', "_")
        )
    }

    #[test]
    fn detects_anima_lora_without_adaln_targets() {
        // SceneWorks#1670: an Anima LoRA that trains only the attention projections carries no
        // `adaln_modulation_*` and no `llm_adapter` key, so neither the adaLN signature nor the
        // unique-key path fires. It has exactly the native/ComfyUI Wan surface — `diffusion_model.
        // blocks.<n>.{self,cross}_attn.…` — and used to be detected as a confident `wan-video`,
        // which hard-rejects the import ("appears to be a wan-video model, but family was declared
        // as anima"). The Cosmos leaf module names (`q_proj`/`output_proj`, never Wan's `.q.`/`.o.`)
        // are what identify it.
        let keys = anima_attention_only_keys(28, comfy_dotted_key);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(detect_lora_family(&header).as_deref(), Some("anima"));
    }

    #[test]
    fn detects_anima_lora_without_adaln_targets_in_every_export_spelling() {
        // The same attention-only file as above in the two other prefix spellings a trainer may
        // emit — diffusers (`transformer.blocks.`) and kohya/musubi (`lora_unet_blocks_`). Both
        // prefixes are accepted by a Wan signature, so both would otherwise mis-detect.
        for build in [
            diffusers_dotted_key as fn(usize, &str, &str) -> String,
            kohya_flattened_key as fn(usize, &str, &str) -> String,
        ] {
            let keys = anima_attention_only_keys(28, build);
            let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
            assert_eq!(detect_lora_family(&header).as_deref(), Some("anima"));
        }
    }

    #[test]
    fn detects_kohya_flattened_anima_lora_with_adaln() {
        // A kohya/musubi-flattened Anima LoRA that *does* train the modulation layers. Its
        // `lora_unet_blocks_<n>_…` prefix is the one the kohya Wan signature requires, and its
        // flattened adaLN keys (`…_adaln_modulation_self_attn_1`) even contain `_self_attn_`, so
        // before this fix it scored as Wan while the Anima signature — which accepted only the
        // `diffusion_model.` prefixes — could not see it at all.
        let mut keys = anima_attention_only_keys(28, kohya_flattened_key);
        for b in 0..28 {
            for adaln in [
                "adaln_modulation_self_attn",
                "adaln_modulation_cross_attn",
                "adaln_modulation_mlp",
            ] {
                for updown in ["1", "2"] {
                    for role in ["lora_down.weight", "lora_up.weight"] {
                        keys.push(format!("lora_unet_blocks_{b}_{adaln}_{updown}.{role}"));
                    }
                }
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(detect_lora_family(&header).as_deref(), Some("anima"));
    }

    #[test]
    fn anima_metadata_base_ids_detect_without_tensor_evidence() {
        // A trainer that stamps the Anima training base / weight file names the family outright,
        // so detection never has to lean on the Wan-shaped tensor keys (SceneWorks#1670).
        for value in [
            "anima",
            "anima_base",
            "anima_turbo",
            "anima-aesthetic-v1.0",
            "cosmos-predict2-2b",
        ] {
            assert_eq!(
                super::metadata_value_to_family(value).as_deref(),
                Some("anima"),
                "{value} should name the anima family"
            );
        }
    }

    #[test]
    fn anima_metadata_match_does_not_swallow_animagine_or_animatediff() {
        // `contains("anima")` would map Animagine XL and AnimateDiff — both SDXL/SD1.5 — onto the
        // anima family and hard-reject legitimate SDXL LoRAs, so the match is token-bounded.
        // Neither names a family (an inconclusive `None` keeps the user's declared family), and
        // crucially neither is claimed as `anima`.
        assert_eq!(super::metadata_value_to_family("animagine-xl-3.1"), None);
        assert_eq!(super::metadata_value_to_family("animatediff"), None);
        assert_eq!(
            super::metadata_value_to_family("sdxl_animagine_v3").as_deref(),
            Some("sdxl")
        );
    }

    #[test]
    fn anima_family_metadata_mappings() {
        assert_eq!(super::model_adapter_for_family("anima"), Some("anima"));
        assert_eq!(
            super::model_capabilities_for_type_and_family("image", "anima"),
            vec!["text_to_image", "style_variations"]
        );
        assert_eq!(
            super::diffusers_class_name_to_family("AnimaModularPipeline").as_deref(),
            Some("anima")
        );
        // The stored/canonical family token round-trips unchanged.
        assert_eq!(super::canonical_lora_family("anima"), "anima");
    }

    /// kohya / musubi-tuner / LyCORIS export of a dual-stream MMDiT (Qwen-Image /
    /// Z-Image) adapter: module paths flattened with underscores behind `prefix`,
    /// carrying LoKr (`lokr_w1`/`lokr_w2`/`alpha`) tensors. Mirrors the real
    /// lycoris-lora file shape from sc-2626.
    fn lycoris_underscore_mmdit_keys(prefix: &str, block_count: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in [
                "attn_to_q",
                "attn_to_k",
                "attn_to_v",
                "attn_to_out_0",
                "attn_add_q_proj",
                "attn_add_k_proj",
                "attn_add_v_proj",
                "attn_to_add_out",
                "img_mlp_net_0_proj",
                "img_mlp_net_2",
                "txt_mlp_net_0_proj",
                "txt_mlp_net_2",
            ] {
                let base = format!("{prefix}_transformer_blocks_{block}_{module}");
                keys.push(format!("{base}.lokr_w1"));
                keys.push(format!("{base}.lokr_w2"));
                keys.push(format!("{base}.alpha"));
            }
        }
        keys
    }

    #[test]
    fn detects_lycoris_underscore_qwen_image_lokr() {
        // The real failing upload (sc-2626): a lycoris-lora-exported Qwen-Image LoKr
        // whose keys carry the library's default `lycoris` prefix and underscore-
        // flattened module paths — invisible to the dotted MMDiT signature.
        let keys = lycoris_underscore_mmdit_keys("lycoris", 60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    #[test]
    fn detects_kohya_underscore_qwen_image() {
        // kohya / musubi-tuner flatten with a `lora_unet` prefix instead.
        let keys = lycoris_underscore_mmdit_keys("lora_unet", 60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    #[test]
    fn low_underscore_mm_dit_block_count_is_inconclusive() {
        // Same conservative block-count gate as the dotted path: too few blocks to
        // tell Qwen from Z-Image → inconclusive rather than a wrong guess.
        let keys = lycoris_underscore_mmdit_keys("lycoris", 24);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert!(detect_lora_family(&header).is_none());
    }

    /// ComfyUI-distributed dual-stream MMDiT (Qwen-Image / Qwen-Image-Edit) adapter:
    /// the block keys carry NO `transformer.` module prefix (bare `transformer_blocks.
    /// <n>.…`) and use kohya `lora_down`/`lora_up`/`alpha` factorization. Mirrors the
    /// real `Qwen-Image-Lightning-4steps` / `Qwen-Image-Edit-2509-Lightning-4steps`
    /// files (sc-10506).
    fn comfyui_bare_prefix_mmdit_keys(block_count: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "img_mlp.net.0.proj",
                "img_mlp.net.2",
                "txt_mlp.net.0.proj",
                "txt_mlp.net.2",
            ] {
                let base = format!("transformer_blocks.{block}.{module}");
                keys.push(format!("{base}.lora_down.weight"));
                keys.push(format!("{base}.lora_up.weight"));
                keys.push(format!("{base}.alpha"));
            }
        }
        keys
    }

    #[test]
    fn detects_comfyui_bare_prefix_qwen_image() {
        // The real failing rows from sc-10452's external-root scan (sc-10506):
        // `Qwen-Image-Lightning-4steps` and `Qwen-Image-Edit-2509-Lightning-4steps`.
        // Their keys drop the `transformer.` prefix the dotted MMDiT signature used
        // to require, so before the fix both surfaced with no detected family and
        // were offered-then-refused at generate. Both share this key shape and detect
        // as `qwen-image` (Qwen-Image-Edit reuses the Qwen-Image transformer, and
        // Edit models declare `qwen-image` LoRA compatibility).
        let keys = comfyui_bare_prefix_mmdit_keys(60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    /// Mage's own row in the ComfyUI bare-prefix grammar (sc-14057). Same key shape and kohya
    /// `lora_down`/`lora_up`/`alpha` factorization as the Qwen-Image Lightning files above — the
    /// **only** thing separating the two families is the 12-block span, so this needs its own
    /// assertion: `detects_comfyui_bare_prefix_qwen_image` pins the 60-block end and stays green
    /// even if the Mage arm regresses to `None`.
    ///
    /// Two negatives are asserted in the same grammar so the test discriminates both halves of the
    /// depth rule rather than just "some Mage-ish header resolves": a shorter 0..=10 span (relax
    /// `max_block == 11` to "anything below Qwen" and it goes green) and a sparse span that merely
    /// *reaches* 11 (drop the `blocks.len()` full-span check and it goes green).
    #[test]
    fn detects_comfyui_bare_prefix_mage_flow() {
        let keys = comfyui_bare_prefix_mmdit_keys(MAGE_FLOW_BLOCK_COUNT);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(detect_lora_family(&header).as_deref(), Some("mage-flow"));

        let short = comfyui_bare_prefix_mmdit_keys(MAGE_FLOW_BLOCK_COUNT - 1);
        let header = header_from_keys(&short.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(
            detect_lora_family(&header),
            None,
            "an 11-block bare-prefix adapter must stay inconclusive, not be claimed as Mage"
        );

        // Reaches block 11 but skips most of the span — the shape a partially-trained community
        // export has. Inconclusive, never a confident Mage.
        let sparse: Vec<String> = comfyui_bare_prefix_mmdit_keys(MAGE_FLOW_BLOCK_COUNT)
            .into_iter()
            .filter(|key| {
                ["blocks.0.", "blocks.1.", "blocks.11."]
                    .iter()
                    .any(|marker| key.contains(marker))
            })
            .collect();
        let header = header_from_keys(&sparse.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(
            detect_lora_family(&header),
            None,
            "a sparse bare-prefix span that only reaches block 11 must stay inconclusive"
        );
    }

    #[test]
    fn bare_prefix_attention_only_is_not_mistaken_for_qwen() {
        // A bare-`transformer_blocks.` adapter that trains ONLY attention projections
        // (no dual `img_mlp`/`txt_mlp`, no joint `add_{q,k}_proj`) is the Krea 2 target
        // shape, NOT a dual-stream MMDiT. The relaxed prefix must not swallow it: the
        // dual-stream require-group keeps it out, so it stays inconclusive here (a real
        // Krea file is instead identified by its `family` metadata stamp).
        let mut keys = Vec::new();
        for block in 0..60 {
            for module in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
                keys.push(format!(
                    "transformer_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "transformer_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn ambiguous_mm_dit_block_count_returns_none() {
        let keys = diffusers_double_stream_keys("transformer", 32);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert!(detect_lora_family(&header).is_none());
    }

    /// Diffusers-format SD3 / SD3.5 LoRA keys. Mirrors a community SD3.5 LoRA
    /// trained via `StableDiffusion3Pipeline.save_lora_weights`: dual-stream joint
    /// attention (`attn.{to_q,to_k,to_v,to_out.0}` image stream +
    /// `attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}` text stream), the SD3
    /// `ff` / `ff_context` feedforwards and `*_context` norms. When `dual_attention`
    /// is set the first 13 joint blocks also train an `attn2` (sd3.5, not sd3.0).
    fn diffusers_sd3_keys(block_count: usize, dual_attention: bool) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "norm1_context.linear",
            ] {
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
            // SD3.5 dual-attention joint blocks (0..=12) add an `attn2` (image-stream
            // self-attention only, no joint `add_*` projections).
            if dual_attention && block <= 12 {
                for module in ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"] {
                    keys.push(format!(
                        "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
                    ));
                    keys.push(format!(
                        "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
                    ));
                }
            }
        }
        keys
    }

    #[test]
    fn detects_sd3_5_large_diffusers() {
        // SD3.5 Large: 38 joint blocks, dual attention in blocks 0..=12.
        let keys = diffusers_sd3_keys(38, true);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd3"));
    }

    #[test]
    fn detects_sd3_5_medium_diffusers() {
        // SD3.5 Medium: 24 joint blocks, dual attention in blocks 0..=12.
        let keys = diffusers_sd3_keys(24, true);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd3"));
    }

    #[test]
    fn detects_sd3_no_dual_attention_diffusers() {
        // An SD3.0-style / attention-only SD3.5 LoRA without `attn2`: it still has the
        // joint `add_q_proj` + SD3 `ff_context` context keys, so it must detect as sd3
        // and never co-score into the Qwen/Z-Image MMDiT bucket.
        let keys = diffusers_sd3_keys(38, false);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd3"));
    }

    #[test]
    fn sd3_attention_only_lora_detects_as_sd3() {
        // A sparse SD3 LoRA that trains ONLY the joint attention projections (the
        // recommended "attention-only" SD3.5 recipe) — no `ff`/`ff_context` — still
        // carries `norm1_context` (the AdaLN-Zero context modulation rides alongside
        // attention) which marks it SD3. Use the `context_embedder` top-level key as
        // the SD3-only anchor, mirroring real attention-only exports.
        let mut keys = Vec::new();
        for block in 0..38 {
            for module in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
            ] {
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        keys.push("transformer.context_embedder.lora_A.weight".to_owned());
        keys.push("transformer.context_embedder.lora_B.weight".to_owned());
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd3"));
    }

    /// kohya / sd-scripts / LyCORIS underscore-flattened SD3 keys behind `prefix`.
    fn underscore_sd3_keys(prefix: &str, block_count: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in [
                "attn_to_q",
                "attn_to_k",
                "attn_to_v",
                "attn_to_out_0",
                "attn_add_q_proj",
                "attn_add_k_proj",
                "attn_add_v_proj",
                "attn_to_add_out",
                "ff_net_0_proj",
                "ff_net_2",
                "ff_context_net_0_proj",
                "ff_context_net_2",
                "norm1_context_linear",
            ] {
                let base = format!("{prefix}_transformer_blocks_{block}_{module}");
                keys.push(format!("{base}.lora_down.weight"));
                keys.push(format!("{base}.lora_up.weight"));
            }
        }
        keys
    }

    #[test]
    fn detects_kohya_underscore_sd3() {
        let keys = underscore_sd3_keys("lora_unet", 38);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd3"));
    }

    #[test]
    fn detects_lycoris_underscore_sd3() {
        let keys = underscore_sd3_keys("lora_transformer", 24);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd3"));
    }

    #[test]
    fn sd3_keys_do_not_detect_as_qwen_or_z_image() {
        // The whole point of a dedicated SD3 bucket: SD3.5 Large has 38 blocks, which
        // is below the Qwen block-count gate (>=39) but above Z-Image's range — under
        // the old Qwen/Z-Image-only MMDiT path it would have been inconclusive. It
        // must now positively resolve to sd3, never to qwen-image or z-image.
        let keys = diffusers_sd3_keys(38, true);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());
        let detected = detect_lora_family(&header);
        assert_eq!(detected.as_deref(), Some("sd3"));
        assert_ne!(detected.as_deref(), Some("qwen-image"));
        assert_ne!(detected.as_deref(), Some("z-image"));
    }

    #[test]
    fn qwen_image_keys_do_not_detect_as_sd3() {
        // Conversely, a real Qwen-Image LoRA (img_mlp/txt_mlp, no ff_context) must
        // never be swallowed by the new SD3 signature.
        let keys = diffusers_double_stream_keys("transformer", 60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    #[test]
    fn detects_sd3_metadata_before_keys() {
        // Metadata-stamped SD3 LoRA (sd-scripts records `ss_base_model_version`).
        for arch in [
            "sd3",
            "sd3.5",
            "sd-3.5-large",
            "stable-diffusion-3.5-large/lora",
            "stable_diffusion_3_medium",
        ] {
            let mut header = header_from_keys(&[
                "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
                "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight",
            ]);
            header["__metadata__"] = json!({ "modelspec.architecture": arch });
            assert_eq!(
                detect_lora_family(&header).as_deref(),
                Some("sd3"),
                "architecture {arch} should map to sd3"
            );
        }
    }

    #[test]
    fn sd3_metadata_not_confused_with_sdxl_or_sd15() {
        // The sd3 metadata branch runs before sdxl / sd1.5; an SDXL or SD1.5 arch
        // string must still resolve to its own family (no bare "sd3" substring).
        let mut sdxl = header_from_keys(&["lora_unet_x.lora_down.weight"]);
        sdxl["__metadata__"] = json!({ "ss_base_model_version": "sdxl_base_v1-0" });
        assert_eq!(detect_lora_family(&sdxl).as_deref(), Some("sdxl"));

        let mut sd15 = header_from_keys(&["lora_unet_x.lora_down.weight"]);
        sd15["__metadata__"] = json!({ "ss_base_model_version": "sd1.5" });
        assert_eq!(detect_lora_family(&sd15).as_deref(), Some("sd1.5"));
    }

    #[test]
    fn sd3_diffusers_class_names_map_to_sd3_family() {
        assert_eq!(
            diffusers_class_name_to_family("StableDiffusion3Pipeline").as_deref(),
            Some("sd3")
        );
        assert_eq!(
            diffusers_class_name_to_family("StableDiffusion3Img2ImgPipeline").as_deref(),
            Some("sd3")
        );
        assert_eq!(
            diffusers_class_name_to_family("StableDiffusion3InpaintPipeline").as_deref(),
            Some("sd3")
        );
    }

    #[test]
    fn sd3_family_maps_to_sd3_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("sd3"), Some("sd3"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "sd3"),
            vec!["text_to_image", "style_variations"],
        );
    }

    #[test]
    fn validate_lora_compatibility_accepts_sd3_lora_on_sd3_model() {
        // A community SD3.5 LoRA declares `family: sd3` and applies on an sd3 model.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "s", "family": "sd3" })],
            Some("sd3"),
            "mlx_sd3",
            Some("sd3_5_large"),
        )
        .is_ok());
        // sd3 is NOT base-model-gated (only wan-video is): a LoRA recording a
        // different sd3 base model still applies by family match.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "s", "family": "sd3", "baseModel": "sd3_5_medium" })],
            Some("sd3"),
            "mlx_sd3",
            Some("sd3_5_large"),
        )
        .is_ok());
        // A foreign-family LoRA is still rejected on an sd3 model.
        let err = validate_lora_compatibility(
            &[json!({ "id": "sdxllora", "family": "sdxl" })],
            Some("sd3"),
            "mlx_sd3",
            Some("sd3_5_large"),
        )
        .unwrap_err();
        assert!(err.contains("sdxllora"), "got: {err}");
    }

    #[test]
    fn detects_sdxl() {
        let mut keys = Vec::new();
        for block in 0..10 {
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_down.weight"
            ));
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_up.weight"
            ));
        }
        keys.push(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned(),
        );
        keys.push(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned(),
        );
        keys.push(
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned(),
        );
        keys.push(
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned(),
        );
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sdxl"));
    }

    #[test]
    fn detects_sd15() {
        let mut keys = Vec::new();
        for block in 0..10 {
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_down.weight"
            ));
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_up.weight"
            ));
        }
        keys.push(
            "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned(),
        );
        keys.push("lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned());
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd1.5"));
    }

    #[test]
    fn empty_header_returns_none() {
        let header = json!({"__metadata__": {"format": "pt"}});
        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn unknown_keys_return_none() {
        let header =
            header_from_keys(&["weird.custom.module.weight", "another.random.tensor.bias"]);
        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn non_object_header_returns_none() {
        let header = json!(["not", "an", "object"]);
        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn diffusers_class_names_map_to_known_families() {
        assert_eq!(
            diffusers_class_name_to_family("ZImagePipeline").as_deref(),
            Some("z-image")
        );
        assert_eq!(
            diffusers_class_name_to_family("QwenImagePipeline").as_deref(),
            Some("qwen-image")
        );
        assert_eq!(
            diffusers_class_name_to_family("FluxPipeline").as_deref(),
            Some("flux")
        );
        assert_eq!(
            diffusers_class_name_to_family("ChromaPipeline").as_deref(),
            Some("chroma")
        );
        assert_eq!(
            diffusers_class_name_to_family("KolorsPipeline").as_deref(),
            Some("kolors")
        );
        assert_eq!(
            diffusers_class_name_to_family("StableDiffusionXLPipeline").as_deref(),
            Some("sdxl")
        );
        assert!(diffusers_class_name_to_family("UnknownCustomPipeline").is_none());
    }

    #[test]
    fn reconcile_detected_family_rejects_mismatches_only() {
        assert_eq!(
            reconcile_detected_family(Some("z-image".to_owned()), Some("z-image".to_owned()))
                .unwrap()
                .as_deref(),
            Some("z-image")
        );
        assert_eq!(
            reconcile_detected_family(None, Some("qwen-image".to_owned()))
                .unwrap()
                .as_deref(),
            Some("qwen-image")
        );
        assert_eq!(
            reconcile_detected_family(Some("wan-video".to_owned()), None)
                .unwrap()
                .as_deref(),
            Some("wan-video")
        );
        assert_eq!(
            reconcile_detected_family(Some("z-image".to_owned()), Some("qwen-image".to_owned()))
                .unwrap_err(),
            FamilyMismatch {
                supplied: "z-image".to_owned(),
                detected: "qwen-image".to_owned(),
            }
        );
    }

    #[test]
    fn reconcile_detected_family_accepts_base_architecture_of_derived_families() {
        // A declared model family whose weights legitimately detect only as a compatible
        // *base* architecture must reconcile (it previously failed every such download /
        // import as a false mismatch), keeping the declared family as the model's identity:
        //   - Chroma (`chroma`) is FLUX.1-derived; metadata-less weights detect as `flux`.
        //   - FLUX.2 [klein] / [dev] carry no variant signature; weights detect as `flux2`.
        let cases = [
            ("chroma", "flux"),
            ("flux2-klein", "flux2"),
            ("flux2_klein", "flux2"),
            ("flux2-dev", "flux2"),
        ];
        for (declared, detected) in cases {
            assert_eq!(
                reconcile_detected_family(Some(declared.to_owned()), Some(detected.to_owned()))
                    .unwrap()
                    .as_deref(),
                Some(canonical_lora_family(declared).as_str()),
                "declared {declared:?} vs detected {detected:?} should keep {declared:?}"
            );
        }
        // Directional: the reverse is NOT a variant relationship, so it stays a mismatch —
        // a plain `flux` weight is not a Chroma, and `flux2` is not FLUX.1 (`flux`).
        assert!(
            reconcile_detected_family(Some("flux".to_owned()), Some("chroma".to_owned())).is_err()
        );
        assert!(
            reconcile_detected_family(Some("flux".to_owned()), Some("flux2".to_owned())).is_err()
        );
        // An unrelated cross-architecture pair is still a confident mismatch.
        assert!(
            reconcile_detected_family(Some("flux2-klein".to_owned()), Some("flux".to_owned()))
                .is_err()
        );
    }

    #[test]
    fn canonical_lora_family_collapses_krea_spelling_variants() {
        for variant in ["krea_2", "krea-2", "krea2", "KREA2", " Krea-2 "] {
            assert_eq!(
                canonical_lora_family(variant),
                "krea_2",
                "{variant:?} should canonicalize to krea_2"
            );
        }
        // Unrelated families keep their normalized (hyphen) stored token.
        assert_eq!(canonical_lora_family("z_image"), "z-image");
        assert_eq!(canonical_lora_family("Wan-Video"), "wan-video");
        assert_eq!(canonical_lora_family("flux2"), "flux2");
    }

    #[test]
    fn reconcile_detected_family_unifies_krea_spelling_variants() {
        // The reported bug: a UI-supplied `krea-2` (or ai-toolkit's `krea2`) against a
        // detected `krea_2` must reconcile to the canonical `krea_2`, not be rejected.
        for supplied in ["krea-2", "krea2", "KREA_2"] {
            assert_eq!(
                reconcile_detected_family(Some(supplied.to_owned()), Some("krea_2".to_owned()))
                    .unwrap()
                    .as_deref(),
                Some("krea_2"),
                "supplied {supplied:?} vs detected krea_2 should resolve to krea_2"
            );
        }
        // A supplied-only krea variant (detection inconclusive) is canonicalized too.
        assert_eq!(
            reconcile_detected_family(Some("krea-2".to_owned()), None)
                .unwrap()
                .as_deref(),
            Some("krea_2")
        );
        // A genuine cross-family conflict still errors, reporting the raw inputs.
        assert!(
            reconcile_detected_family(Some("flux".to_owned()), Some("krea_2".to_owned())).is_err()
        );
    }

    #[test]
    fn model_manifest_defaults_follow_supported_families() {
        let mut entry = serde_json::Map::new();
        apply_model_manifest_defaults(&mut entry, "video", Some("wan_video"));

        assert_eq!(entry["adapter"], "wan_video");
        assert_eq!(
            entry["capabilities"],
            json!([
                "image_to_video",
                "text_to_video",
                "first_last_frame",
                "extend_clip",
                "video_bridge",
                "replace_person"
            ])
        );
        assert_eq!(entry["loraCompatibility"]["families"], json!(["wan-video"]));
        assert_eq!(entry["downloads"], json!([]));
    }

    #[test]
    fn imported_krea_2_gets_adapter_and_text_to_image_capability() {
        // sc-14108 (epic 14015): an imported single-file krea_2 base checkpoint whose family the
        // base-weight gate stamps must ALSO pick up the krea_2 family defaults, or it stays
        // adapter-less with an empty `capabilities` array and the Image Studio picker (which filters
        // on `text_to_image`) still hides it. `apply_model_manifest_defaults` is fed the canonical
        // catalog token `krea_2` (from `reconcile_detected_family`); it normalizes to `krea-2`
        // internally, which is the form the family-default arms key on.
        let mut entry = serde_json::Map::new();
        apply_model_manifest_defaults(&mut entry, "image", Some("krea_2"));

        // The imported model is now MLX-Krea-routed and selectable as a text-to-image model.
        assert_eq!(entry["adapter"], "mlx_krea");
        // Text-to-image + edit_image (the MLX Kontext edit surface, sc-14119) + style_variations.
        // img2img is a `ui.img2img` toggle (asserted in the sibling test), NOT a capabilities value.
        // `character_image` stays unclaimed (no IP-Adapter/identity surface on the bare transformer).
        assert_eq!(
            entry["capabilities"],
            json!(["text_to_image", "edit_image", "style_variations"])
        );
        // `loraCompatibility.families` carries the normalized token (as every other family does,
        // e.g. wan-video above) — Krea LoRAs resolve to it through `canonical_lora_family`.
        assert_eq!(entry["loraCompatibility"]["families"], json!(["krea-2"]));
    }

    #[test]
    fn imported_krea_2_gets_builtin_resolution_and_img2img_surface() {
        // sc-14071 (epic 14015): an imported krea_2 entry ships empty limits/defaults/ui, so the Studio
        // resolution picker would fall back to its 4-option list and never offer img2img. The family
        // default must stamp the SAME resolution / img2img / edit surface the builtin `krea_2_turbo`
        // entry carries (config/manifests/builtin.models.jsonc): the 15-bucket resolution list, the
        // 1024² default, `mlx.minMemoryGb` 48 (the >1536² memory-gate anchor, sc-13959), the
        // `ui.img2img` toggle + `img2imgStrength` slider, and — for the Kontext edit surface (sc-14119)
        // — the `ui.editReferences` optional-second-image slot.
        let mut entry = serde_json::Map::new();
        apply_model_manifest_defaults(&mut entry, "image", Some("krea_2"));

        // The exact 15-bucket resolution list the builtin Krea 2 Turbo ships (verbatim, same order).
        let resolutions = entry["limits"]["resolutions"]
            .as_array()
            .expect("resolutions is an array");
        assert_eq!(
            resolutions.len(),
            15,
            "the builtin Krea Turbo resolution list has 15 buckets"
        );
        assert_eq!(
            entry["limits"]["resolutions"],
            json!([
                "1024x1024",
                "768x1024",
                "1024x768",
                "1280x720",
                "720x1280",
                "1216x832",
                "832x1216",
                "1152x896",
                "896x1152",
                "1536x1536",
                "2048x1152",
                "1152x2048",
                "2048x1408",
                "1408x2048",
                "2048x2048"
            ])
        );
        assert_eq!(entry["defaults"]["resolution"], "1024x1024");
        // The ≤1536² visibility floor / >1536² memory-gate anchor — without it 2048² is offered
        // unconditionally on any Mac (sc-13959).
        assert_eq!(entry["mlx"]["minMemoryGb"], json!(48));
        // img2img is exposed as the `ui.img2img` toggle + strength slider (worker resolves it via
        // `resolve_img2img_init_generic`), NOT as a capability.
        assert_eq!(entry["ui"]["img2img"], json!(true));
        assert_eq!(entry["ui"]["img2imgStrength"]["default"], json!(0.5));
        // The Kontext edit second-image slot (sc-14119) mirrors the builtin krea_2_turbo `ui` block.
        assert_eq!(
            entry["ui"]["editReferences"]["secondaryLabel"],
            json!("Image 2 (optional)")
        );
    }

    /// sc-15036 (epic 14034 F6) — a full base fine-tune lands in the model catalog as a bare
    /// entry: an id, a name, a family, and `paths.model`. Without the family defaults it would be
    /// adapter-less with an EMPTY `capabilities` array, and the Image Studio picker (which filters
    /// on `text_to_image`) would hide the model the user just spent hours training.
    ///
    /// Also pins the Studio surface it inherits from its base: the 13-bucket resolution ladder, the
    /// 1024² default, the undistilled Base 30-step / CFG-5 regime, and — load-bearing — the BINDING
    /// `requiresDimensionsMultipleOf: 16` stride, without which the free Width/Height override
    /// offers geometry the engine rejects.
    #[test]
    fn a_fine_tuned_mage_flow_base_gets_the_family_surface_it_needs_to_be_selectable() {
        let mut entry = serde_json::Map::new();
        apply_model_manifest_defaults(&mut entry, "image", Some("mage-flow"));

        assert_eq!(entry["adapter"], "mlx_mage");
        // Text-to-image + style variations. NOT `edit_image` (that needs an `mage_flow_edit*`
        // checkpoint, which is not a training target) and NOT `character_image` (no identity
        // surface) — the non-edit Mage descriptors advertise no conditioning at all.
        assert_eq!(
            entry["capabilities"],
            json!(["text_to_image", "style_variations"])
        );
        assert_eq!(entry["loraCompatibility"]["families"], json!(["mage-flow"]));

        assert_eq!(
            entry["limits"]["resolutions"],
            json!([
                "512x512",
                "768x768",
                "1024x1024",
                "1536x1536",
                "2048x2048",
                "1280x720",
                "1536x1024",
                "2048x1024",
                "2048x512",
                "720x1280",
                "1024x1536",
                "1024x2048",
                "512x2048"
            ])
        );
        assert_eq!(entry["defaults"]["resolution"], "1024x1024");
        assert_eq!(entry["defaults"]["steps"], json!(30));
        assert_eq!(entry["defaults"]["guidanceScale"], json!(5));
        assert_eq!(
            entry["limits"]["requiresDimensionsMultipleOf"],
            json!(16),
            "the ÷16 stride is BINDING on Mage geometry, not advisory"
        );
        // Deliberately NO `mlx.quantize`: a fine-tune is dense bf16 and the pre-quantized tiers
        // carry an 8-bit floor (sc-15071) that a load-time quantize does not reproduce, so packing
        // one at load would render a tiled texture rather than the prompt.
        assert!(
            !entry["mlx"]
                .as_object()
                .expect("mlx block")
                .contains_key("quantize"),
            "a fine-tuned Mage checkpoint must not be load-time quantized by default"
        );
        // ...but it DOES declare a memory floor, and that pairing is the whole point: loading dense
        // is exactly what makes the sc-13959 >1536² gate matter. With no floor, `resolutionMemory`
        // has no basis to predict a peak and offers 2048² on ANY Mac — and an MLX overcommit is an
        // uncatchable SIGKILL. Derived from the builtin manifest's measured q4/q8 unified peaks and
        // cross-checked against its measured candle bf16 VRAM; see the arm for the arithmetic.
        assert_eq!(
            entry["mlx"]["minMemoryGb"],
            json!(20),
            "a dense fine-tune must anchor the >1536² memory gate"
        );
        // No conditioning surface is invented for it either.
        let ui = entry["ui"].as_object().expect("ui object");
        assert!(!ui.contains_key("img2img"), "Mage advertises no img2img");
        assert!(
            !ui.contains_key("editReferences"),
            "the non-edit Mage variants take no reference images"
        );
    }

    /// sc-15036 — an author-supplied value always wins over the family default, so the stamping can
    /// never overwrite a catalog entry that already declares its own surface. Discriminating: the
    /// author's one-bucket list survives while the untouched sibling keys are still filled.
    #[test]
    fn mage_flow_studio_defaults_never_overwrite_author_supplied_values() {
        let mut entry = serde_json::Map::new();
        entry.insert(
            "limits".to_owned(),
            json!({ "resolutions": ["768x768"], "requiresDimensionsMultipleOf": 32 }),
        );
        entry.insert("defaults".to_owned(), json!({ "steps": 8 }));
        apply_model_manifest_defaults(&mut entry, "image", Some("mage-flow"));

        assert_eq!(entry["limits"]["resolutions"], json!(["768x768"]));
        assert_eq!(entry["limits"]["requiresDimensionsMultipleOf"], json!(32));
        assert_eq!(entry["defaults"]["steps"], json!(8));
        // ...and the gaps are still filled.
        assert_eq!(entry["defaults"]["resolution"], "1024x1024");
        assert_eq!(entry["limits"]["count"], json!([1, 2, 4]));
    }

    #[test]
    fn non_krea_families_get_no_studio_resolution_surface() {
        // The krea-2 Studio-surface default (sc-14071) is family-scoped: every other family keeps its
        // empty `limits` / `defaults` / `ui` blocks and gains no `mlx` block, so nothing else is
        // perturbed by the new stamping.
        for (model_type, family) in [("image", "z-image"), ("video", "wan_video")] {
            let mut entry = serde_json::Map::new();
            apply_model_manifest_defaults(&mut entry, model_type, Some(family));
            assert!(
                entry["limits"]
                    .as_object()
                    .expect("limits object")
                    .is_empty(),
                "{family} limits must stay empty"
            );
            assert!(
                entry["defaults"]
                    .as_object()
                    .expect("defaults object")
                    .is_empty(),
                "{family} defaults must stay empty"
            );
            assert!(
                entry["ui"].as_object().expect("ui object").is_empty(),
                "{family} ui must stay empty"
            );
            assert!(
                entry.get("mlx").is_none(),
                "{family} must gain no mlx block"
            );
        }
    }

    #[test]
    fn detect_model_family_reads_diffusers_index() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("model_index.json"),
            br#"{"_class_name": "ZImagePipeline", "_diffusers_version": "0.27.0"}"#,
        )
        .expect("write index");
        let family = detect_model_family(temp.path()).expect("detect");
        assert_eq!(family.as_deref(), Some("z-image"));
    }

    #[test]
    fn detect_model_family_requires_exact_mage_diffusers_contract() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(temp.path().join("transformer")).expect("transformer dir");
        std::fs::write(
            temp.path().join("model_index.json"),
            br#"{"_class_name":"MageFlowPipeline"}"#,
        )
        .expect("write index");
        let config_path = temp.path().join("transformer").join("config.json");
        let valid = json!({
            "_class_name": "MageFlow",
            "in_channels": 128,
            "hidden_size": 3072,
            "depth": 12
        });
        std::fs::write(&config_path, serde_json::to_vec(&valid).unwrap()).expect("write config");
        assert_eq!(
            detect_model_family(temp.path()).expect("detect").as_deref(),
            Some("mage-flow")
        );

        for (field, wrong) in [
            ("_class_name", json!("NotMage")),
            ("in_channels", json!(64)),
            ("hidden_size", json!(2048)),
            ("depth", json!(24)),
        ] {
            let mut mutated = valid.clone();
            mutated[field] = wrong;
            std::fs::write(&config_path, serde_json::to_vec(&mutated).unwrap())
                .expect("mutate config");
            assert!(
                detect_model_family(temp.path()).expect("detect").is_none(),
                "{field} mismatch must fail closed"
            );
        }
    }

    #[test]
    fn detect_model_family_falls_back_to_header() {
        let temp = tempfile::tempdir().expect("tempdir");
        let keys = diffusers_double_stream_keys("transformer", 40);
        write_safetensors(&temp.path().join("checkpoint.safetensors"), &keys);
        let family = detect_model_family(temp.path()).expect("detect");
        assert_eq!(family.as_deref(), Some("qwen-image"));
    }

    #[test]
    fn detect_model_family_returns_none_for_empty_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let family = detect_model_family(temp.path()).expect("detect");
        assert!(family.is_none());
    }

    #[test]
    fn detect_model_family_returns_none_for_unmapped_class_name() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("model_index.json"),
            br#"{"_class_name": "ExperimentalPipeline"}"#,
        )
        .expect("write index");
        let family = detect_model_family(temp.path()).expect("detect");
        assert!(family.is_none());
    }

    /// Write a safetensors file whose header declares a single tensor spanning
    /// `[0, declared_data_len)` but whose data section on disk is only
    /// `actual_data_len` bytes — so `declared_data_len > actual_data_len`
    /// reproduces a truncated/interrupted download.
    fn write_safetensors_with_data(path: &Path, declared_data_len: u64, actual_data_len: u64) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        header.insert(
            "lora.weight".to_owned(),
            json!({"dtype": "F16", "shape": [1], "data_offsets": [0, declared_data_len]}),
        );
        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        buffer.resize(buffer.len() + actual_data_len as usize, 0_u8);
        std::fs::write(path, buffer).expect("write safetensors");
    }

    #[test]
    fn read_safetensors_header_accepts_complete_data_section() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("complete.safetensors");
        write_safetensors_with_data(&path, 32, 32);
        read_safetensors_header(&path).expect("complete file is accepted");
    }

    #[test]
    fn read_safetensors_header_accepts_trailing_padding() {
        // A file larger than the declared data section (trailing padding) is not
        // "incomplete"; only truncation is rejected.
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("padded.safetensors");
        write_safetensors_with_data(&path, 32, 64);
        read_safetensors_header(&path).expect("over-long file is accepted");
    }

    #[test]
    fn read_safetensors_header_rejects_truncated_data_section() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("truncated.safetensors");
        // Header declares 1024 bytes of tensor data, but only 16 are present.
        write_safetensors_with_data(&path, 1024, 16);
        match read_safetensors_header(&path) {
            Err(SafetensorsHeaderError::IncompleteData { declared, actual }) => {
                assert!(
                    actual < declared,
                    "actual {actual} should be below declared minimum {declared}"
                );
            }
            other => panic!("expected IncompleteData, got {other:?}"),
        }
    }

    #[test]
    fn read_safetensors_header_accepts_empty_tensors() {
        // The `write_safetensors` helper emits empty tensors (`data_offsets [0, 0]`);
        // a header-only file with no tensor bytes is complete.
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("empty.safetensors");
        write_safetensors(&path, &["lora.weight".to_owned()]);
        read_safetensors_header(&path).expect("empty-tensor file is accepted");
    }

    #[test]
    fn max_tensor_data_end_skips_metadata_and_takes_max() {
        let header = json!({
            "__metadata__": {"format": "pt"},
            "a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 100]},
            "b": {"dtype": "F16", "shape": [1], "data_offsets": [100, 420]},
        });
        assert_eq!(max_tensor_data_end(&header), 420);
        assert_eq!(max_tensor_data_end(&json!({"__metadata__": {"x": "y"}})), 0);
    }

    #[test]
    fn sensenova_u1_image_family_supports_text_to_image_and_edit() {
        let caps = model_capabilities_for_type_and_family("image", "sensenova-u1");
        assert!(caps.contains(&"text_to_image"));
        assert!(caps.contains(&"edit_image"));
    }

    // --- LoRA family-compat validation (sc-3027) ---

    #[test]
    fn accepted_lora_families_includes_extra_compatible() {
        assert_eq!(accepted_lora_families("flux"), vec!["flux".to_owned()]);
        // chroma additionally accepts flux; flux2-klein accepts flux2.
        assert_eq!(
            accepted_lora_families("chroma"),
            vec!["chroma".to_owned(), "flux".to_owned()]
        );
        assert_eq!(
            accepted_lora_families("flux2_klein"),
            vec!["flux2-klein".to_owned(), "flux2".to_owned()]
        );
        assert!(accepted_lora_families("").is_empty());
    }

    /// sc-8444 (epic 8431): Krea Realtime 14B declares its OWN `krea-realtime` family in the
    /// catalog, and picks up Wan-family LoRAs through the extra-compatible registry rather than by
    /// joining the Wan family. Both halves are asserted, and both DISCRIMINATE:
    /// * the accepted set must be exactly `[krea-realtime, wan-video]` — dropping the registry
    ///   entry leaves `[krea-realtime]` (a real Wan LoRA then gets pre-flight rejected), and
    ///   declaring `wan-video` as the model's own family instead would make the FIRST element
    ///   `wan-video`, which this pins against;
    /// * the relation stays one-directional — a Wan model must NOT thereby accept a
    ///   `krea-realtime` LoRA, which is what a symmetric edit to the registry would produce.
    #[test]
    fn krea_realtime_accepts_wan_loras_one_directionally() {
        assert_eq!(
            accepted_lora_families("krea-realtime"),
            vec!["krea-realtime".to_owned(), "wan-video".to_owned()]
        );
        // The manifest spells the family with a hyphen; the underscore spelling normalizes to the
        // same token, so the registry hit does not depend on which spelling reaches it.
        assert_eq!(
            accepted_lora_families("krea_realtime"),
            accepted_lora_families("krea-realtime")
        );
        // Not symmetric: a Wan model accepts only Wan LoRAs.
        assert_eq!(
            accepted_lora_families("wan-video"),
            vec!["wan-video".to_owned()]
        );
        // Distinct from the Krea 2 IMAGE family — same vendor, unrelated architecture.
        assert_eq!(accepted_lora_families("krea_2"), vec!["krea-2".to_owned()]);
        // And a real Wan LoRA passes the pre-flight on a Krea Realtime job.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a", "family": "wan-video" })],
            Some("krea-realtime"),
            "krea_realtime",
            Some("krea_realtime_14b"),
        )
        .is_ok());
    }

    /// 🔴 sc-15017: the family half of the gate was never the whole gate. `wan-video` is
    /// base-model GATED, and that gate keys on the LORA's declared family — so it fires on a Krea
    /// Realtime job too, where exact id equality can never hold (the LoRA records a Wan base,
    /// the model is `krea_realtime_14b`). Left alone, every base-model-STAMPED Wan LoRA — which is
    /// what SceneWorks' own importer writes — was refused on Krea while an unstamped one passed.
    ///
    /// The relaxation must not dissolve the gate: 5B and 14B stay non-interchangeable.
    #[test]
    fn base_model_gate_admits_wan_14b_on_krea_but_still_splits_5b_from_14b() {
        // Exact equality is unchanged, and it is the ONLY answer for a genuine Wan model: the
        // second arm cannot fire there, because `wan-video` has no extra-compatible entry.
        assert!(base_model_satisfies_gate(
            "wan-video",
            "wan_2_2_t2v_14b",
            "wan_2_2_t2v_14b"
        ));
        assert!(
            !base_model_satisfies_gate("wan-video", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"),
            "two 14B WAN models must still pin to their own base — the relaxation is only for the \
             extra-compatible relation, not a blanket 14B pass"
        );
        assert!(!base_model_satisfies_gate(
            "wan-video",
            "wan_2_2_t2v_14b",
            "wan_2_2"
        ));

        // Krea Realtime accepts a Wan 14B base…
        assert!(base_model_satisfies_gate(
            "krea-realtime",
            "krea_realtime_14b",
            "wan_2_2_t2v_14b"
        ));
        assert!(base_model_satisfies_gate(
            "krea-realtime",
            "krea_realtime_14b",
            "wan_2_1_t2v_14b"
        ));
        // …and REFUSES the 5B TI2V base, which is the whole reason the gate exists.
        assert!(
            !base_model_satisfies_gate("krea-realtime", "krea_realtime_14b", "wan_2_2"),
            "the 5B TI2V base has 48 latent channels — admitting it would garble the render"
        );
        // A model with no extra-compatible entry gets no relaxation, even between two 14B ids.
        // (This used to name `scail2`, which gained a `wan-video` entry in sc-18200 — so it now
        // legitimately DOES relax, and is asserted separately below. `ltx-video` has no entry.)
        assert!(!base_model_satisfies_gate(
            "ltx-video",
            "ltx_2_3_14b",
            "wan_2_2_t2v_14b"
        ));
        // 🔴 …and REFUSES an I2V base, even though it is the same 14B size class. Krea Realtime is a
        // TEXT-to-video backbone: an I2V LoRA targets `cross_attn.k_img`/`v_img`, which it does not
        // have. The product already refuses this stamp on the sibling T2V model by exact equality
        // (asserted right below), so admitting it here would be the one place that inconsistency
        // existed — and it would surface as a hard engine error AFTER a multi-GB tier fetch instead
        // of a 400 at submit.
        assert!(!base_model_satisfies_gate(
            "krea-realtime",
            "krea_realtime_14b",
            "wan_2_2_i2v_14b"
        ));
        assert!(
            !base_model_satisfies_gate("wan-video", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"),
            "the sibling T2V model this mirrors must genuinely refuse the same stamp"
        );
        // The exclusion is a path SEGMENT, not a substring: an id that merely CONTAINS the letters
        // must still pass, or a future entry could be refused for its spelling.
        assert!(base_model_satisfies_gate(
            "krea-realtime",
            "krea_realtime_14b",
            "wan_2_2_si2vx_14b"
        ));
        // And an I2V base still loads on its OWN model — the exclusion is scoped to the
        // extra-compatible arm; it does not tighten exact equality.
        assert!(base_model_satisfies_gate(
            "wan-video",
            "wan_2_2_i2v_14b",
            "wan_2_2_i2v_14b"
        ));

        // End to end through the pre-flight the worker runs.
        let stamped_14b = json!({
            "id": "origami", "family": "wan-video", "baseModel": "wan_2_2_t2v_14b"
        });
        assert!(
            validate_lora_compatibility(
                std::slice::from_ref(&stamped_14b),
                Some("krea-realtime"),
                "krea_realtime",
                Some("krea_realtime_14b"),
            )
            .is_ok(),
            "a base-model-stamped Wan 14B LoRA must run on Krea Realtime"
        );
        let stamped_5b = json!({
            "id": "ti2v_style", "family": "wan-video", "baseModel": "wan_2_2"
        });
        assert!(validate_lora_compatibility(
            std::slice::from_ref(&stamped_5b),
            Some("krea-realtime"),
            "krea_realtime",
            Some("krea_realtime_14b"),
        )
        .is_err());
        // The same 5B LoRA is still fine on its own model — so the rejection above is the SIZE
        // class, not the stamp merely being present.
        assert!(validate_lora_compatibility(
            std::slice::from_ref(&stamped_5b),
            Some("wan-video"),
            "wan_video",
            Some("wan_2_2"),
        )
        .is_ok());
    }

    /// sc-18200: SCAIL-2's DiT is Wan2.1-I2V-derived and ships the raw I2V module names, and the
    /// bundled `scail2_lightning` toggle IS a lightx2v Wan2.1-I2V LoRA — so a Wan LoRA must load on
    /// a SCAIL-2 model. Same shape as the krea-realtime entry above, and asserted the same way so
    /// it DISCRIMINATES: dropping the registry entry leaves `[scail2]` (and the job-creation gate
    /// then rejects the lightning LoRA outright), while declaring `wan-video` as SCAIL-2's own
    /// family would make the FIRST element `wan-video`, which this pins against.
    /// The base-model half, which only exists on this line (sc-15017). The `wan-video` registry entry
    /// is read by `base_model_satisfies_gate` too, so sc-18200 gained a SECOND meaning when forward-
    /// ported here — and the I2V axis sc-15017 wrote for Krea Realtime is INVERTED for SCAIL-2.
    /// Krea Realtime is a T2V backbone with no `cross_attn.k_img`/`v_img`, so an I2V stamp must be
    /// refused there. SCAIL-2 is Wan2.1-**I2V**-derived and carries those projections — the bundled
    /// lightning adapter patches them — so an I2V base is its EXACT match. Ported naively, SCAIL-2
    /// would have refused the right LoRAs and admitted the weaker ones, silently.
    #[test]
    fn scail2_base_model_gate_admits_i2v_unlike_the_t2v_backbone() {
        // The exact architectural match: a Wan I2V 14B base on SCAIL-2.
        assert!(
            base_model_satisfies_gate("scail2", "scail2_14b", "wan_2_2_i2v_14b"),
            "SCAIL-2 is I2V-derived and has k_img/v_img — an I2V base is its exact match"
        );
        // The same stamp stays refused on the T2V backbone, unchanged by this split.
        assert!(
            !base_model_satisfies_gate("krea-realtime", "krea_realtime_14b", "wan_2_2_i2v_14b"),
            "Krea Realtime has no image cross-attention; sc-15017's exclusion still holds there"
        );
        // A T2V base is still admitted on SCAIL-2 (same 40×5120 block layout, minus the img stack).
        assert!(base_model_satisfies_gate(
            "scail2",
            "scail2_14b",
            "wan_2_2_t2v_14b"
        ));
        // The size class still gates on BOTH: the 5B TI2V base is refused either way.
        assert!(
            !base_model_satisfies_gate("scail2", "scail2_14b", "wan_2_2"),
            "the 5B TI2V base is a different latent geometry — the relaxation is size-classed"
        );
    }

    #[test]
    fn scail2_accepts_wan_loras_one_directionally() {
        assert_eq!(
            accepted_lora_families("scail2"),
            vec!["scail2".to_owned(), "wan-video".to_owned()]
        );
        // Not symmetric: a Wan model does not thereby accept a scail2 LoRA.
        assert!(!accepted_lora_families("wan-video").contains(&"scail2".to_owned()));
        // The pre-flight accepts a Wan-declared LoRA on a SCAIL-2 job.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "scail2_lightning", "family": "wan-video" })],
            Some("scail2"),
            "scail2",
            Some("scail2_14b"),
        )
        .is_ok());
        // ...and still refuses an unrelated architecture.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "wrong", "family": "flux" })],
            Some("scail2"),
            "scail2",
            Some("scail2_14b"),
        )
        .is_err());
    }

    #[test]
    fn lora_declared_families_reads_first_present_source() {
        assert_eq!(
            lora_declared_families(&json!({ "family": "FLUX" })),
            vec!["flux".to_owned()]
        );
        assert_eq!(
            lora_declared_families(&json!({ "compatibility": { "families": ["sdxl"] } })),
            vec!["sdxl".to_owned()]
        );
        // `families` wins over `family`; normalized + de-duplicated + sorted.
        assert_eq!(
            lora_declared_families(&json!({ "families": ["flux2", "Flux2"], "family": "sdxl" })),
            vec!["flux2".to_owned()]
        );
        assert!(lora_declared_families(&json!({ "id": "x" })).is_empty());
    }

    #[test]
    fn validate_lora_compatibility_accepts_matching_and_extra_compatible() {
        // exact family
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a", "family": "flux" })],
            Some("flux"),
            "mlx_flux",
            Some("flux_dev"),
        )
        .is_ok());
        // flux2-klein model accepts a flux2 LoRA
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a", "family": "flux2" })],
            Some("flux2-klein"),
            "mlx_flux2",
            Some("flux2_klein_9b"),
        )
        .is_ok());
        // unstamped LoRA (no declared family) is skipped, not rejected
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a" })],
            Some("sdxl"),
            "mlx_sdxl",
            Some("sdxl"),
        )
        .is_ok());
        // unknown model family → skip (no accepted set)
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a", "family": "flux" })],
            None,
            "mlx_x",
            None,
        )
        .is_ok());
    }

    #[test]
    fn validate_lora_compatibility_rejects_incompatible_family() {
        let err = validate_lora_compatibility(
            &[json!({ "id": "fluxlora", "family": "flux" })],
            Some("sdxl"),
            "mlx_sdxl",
            Some("sdxl"),
        )
        .unwrap_err();
        assert!(err.contains("fluxlora"), "got: {err}");
        assert!(err.contains("sdxl"), "got: {err}");
    }

    #[test]
    fn validate_lora_compatibility_gates_wan_base_model() {
        // Same family (wan-video) but a LoRA trained for a different base model is rejected.
        let err = validate_lora_compatibility(
            &[json!({ "id": "w", "family": "wan-video", "baseModel": "wan_2_2_t2v_14b" })],
            Some("wan-video"),
            "mlx_wan",
            Some("wan_2_2"),
        )
        .unwrap_err();
        assert!(err.contains("not interchangeable"), "got: {err}");
        // A matching base model passes; a LoRA without a base model falls back to family.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "w", "family": "wan-video", "baseModel": "wan_2_2" })],
            Some("wan-video"),
            "mlx_wan",
            Some("wan_2_2"),
        )
        .is_ok());
        assert!(validate_lora_compatibility(
            &[json!({ "id": "w", "family": "wan-video" })],
            Some("wan-video"),
            "mlx_wan",
            Some("wan_2_2"),
        )
        .is_ok());
    }

    #[test]
    fn validate_lora_compatibility_accepts_krea_raw_lora_on_turbo() {
        // epic 7565 P3 (sc-7578): a Krea LoRA trained on Krea 2 Raw records
        // `family: krea_2` / `baseModel: krea_2_raw` and applies at Krea 2 Turbo inference
        // by family match. `krea_2` is NOT base-model-gated (only wan-video is), so the Raw
        // base model differing from the served Turbo model id does NOT reject — the Lens /
        // Z-Image train-on-base → infer-on-Turbo precedent.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "k", "family": "krea_2", "baseModel": "krea_2_raw" })],
            Some("krea_2"),
            "mlx_krea",
            Some("krea_2_turbo"),
        )
        .is_ok());
        // A foreign-family LoRA is still rejected on the Krea Turbo model.
        let err = validate_lora_compatibility(
            &[json!({ "id": "sdxllora", "family": "sdxl" })],
            Some("krea_2"),
            "mlx_krea",
            Some("krea_2_turbo"),
        )
        .unwrap_err();
        assert!(err.contains("sdxllora"), "got: {err}");
    }

    // sc-10221: resolve_adapter_in_dir prefers the manifest-declared adapter over an
    // arbitrary directory scan, so a trained LoRA loads its final adapter rather than a
    // step checkpoint sharing the folder.
    fn touch(path: &Path) {
        std::fs::File::create(path).expect("create fixture file");
    }

    #[test]
    fn resolve_adapter_in_dir_prefers_declared_over_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A trainer folder: the final adapter plus a step checkpoint that sorts BEFORE it
        // (`-` 0x2D < `.` 0x2E), so an ordered scan would grab the checkpoint.
        let final_adapter = dir.path().join("my_style.safetensors");
        touch(&dir.path().join("my_style-step250.safetensors"));
        touch(&final_adapter);
        assert_eq!(
            resolve_adapter_in_dir(dir.path(), Some("my_style.safetensors")),
            Some(final_adapter)
        );
    }

    #[test]
    fn resolve_adapter_in_dir_falls_back_when_no_or_missing_declared() {
        let dir = tempfile::tempdir().expect("tempdir");
        let only = dir.path().join("adapter.safetensors");
        touch(&only);
        // No declared name → first_safetensors_path fallback.
        assert_eq!(resolve_adapter_in_dir(dir.path(), None), Some(only.clone()));
        // Declared name that doesn't exist on disk → fallback (not an error).
        assert_eq!(
            resolve_adapter_in_dir(dir.path(), Some("does_not_exist.safetensors")),
            Some(only)
        );
    }

    #[test]
    fn resolve_adapter_in_dir_rejects_traversal_and_non_safetensors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let final_adapter = dir.path().join("final.safetensors");
        touch(&final_adapter);
        // Outside the dir: a real sibling the crafted name tries to reach via `..`.
        let outside = dir.path().parent().unwrap().join("evil.safetensors");
        touch(&outside);
        // Path-separated / traversing declared names are ignored (fall back to the scan),
        // so a crafted `files` value can't redirect the load outside the record dir.
        for crafted in ["../evil.safetensors", "sub/final.safetensors", "..", "."] {
            assert_eq!(
                resolve_adapter_in_dir(dir.path(), Some(crafted)),
                Some(final_adapter.clone()),
                "crafted name {crafted:?} must not escape the dir"
            );
        }
        // A declared non-.safetensors file is not accepted; fall back to the scan.
        touch(&dir.path().join("notes.txt"));
        assert_eq!(
            resolve_adapter_in_dir(dir.path(), Some("notes.txt")),
            Some(final_adapter)
        );
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn is_hidden_file_flags_dotfiles_and_appledouble_sidecars() {
        assert!(is_hidden_file(Path::new("._adapter.safetensors")));
        assert!(is_hidden_file(Path::new("/a/b/._model.safetensors")));
        assert!(is_hidden_file(Path::new("/a/b/.DS_Store")));
        assert!(!is_hidden_file(Path::new("adapter.safetensors")));
        assert!(!is_hidden_file(Path::new("/a/b/model.safetensors")));
        // A dot on a *directory* component is not a hidden file name.
        assert!(!is_hidden_file(Path::new("/a/.cache/model.safetensors")));
    }

    /// SceneWorks#1333: `._adapter.safetensors` (a macOS AppleDouble sidecar) carries the
    /// `.safetensors` extension, so the extension-only filter used to accept it. `first_safetensors_path`
    /// scans with an *unordered* `read_dir` and returns the first match, so the sidecar could be
    /// returned in place of the real adapter — nondeterministically, run to run.
    #[test]
    fn first_safetensors_path_skips_appledouble_sidecar() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("adapter.safetensors");
        touch(&dir.path().join("._adapter.safetensors"));
        touch(&real);
        assert_eq!(first_safetensors_path(dir.path()), Some(real));
    }

    /// A dir holding only a sidecar has no adapter at all.
    #[test]
    fn first_safetensors_path_ignores_a_lone_sidecar() {
        let dir = tempfile::tempdir().expect("tempdir");
        touch(&dir.path().join("._adapter.safetensors"));
        assert_eq!(first_safetensors_path(dir.path()), None);
    }

    /// The sidecar must also be rejected on the *declared* path, and on a direct file argument.
    #[test]
    fn resolve_adapter_in_dir_rejects_a_declared_sidecar() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("adapter.safetensors");
        let sidecar = dir.path().join("._adapter.safetensors");
        touch(&sidecar);
        touch(&real);
        // Declared name naming the sidecar → not accepted; falls back to the scan, which skips it.
        assert_eq!(
            resolve_adapter_in_dir(dir.path(), Some("._adapter.safetensors")),
            Some(real)
        );
        // Passed directly as a file path, a sidecar is still not a safetensors file.
        assert_eq!(first_safetensors_path(&sidecar), None);
    }
}
