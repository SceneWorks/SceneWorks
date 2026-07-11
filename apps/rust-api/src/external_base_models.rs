//! Surface the **base** weights sitting in an operator's ComfyUI `models/` tree as
//! assembled virtual models, read in place (epic 10451 Phase 2, sc-10667).
//!
//! Phase 1 (sc-10452, [`crate::external_loras`]) scans only `<root>/loras`. Phase 2
//! adds the sibling base subtrees — `diffusion_models/`, `unet/`, `text_encoders/`,
//! `vae/`, `checkpoints/` — and assembles a **virtual model** from the separate
//! component files, because modern ComfyUI does not fuse them: the diffusion
//! transformer, the prompt encoders, and the VAE are distinct files. A legacy
//! all-in-one `checkpoints/` file is a virtual model on its own.
//!
//! Each file is classified header-only by the sc-10662 detector
//! ([`sceneworks_core::base_weights`]) into `(family, component, quant)`. A model is
//! **anchored** by a transformer (`diffusion_models/`, `unet/`) or an all-in-one
//! checkpoint (`checkpoints/`); the text encoders and VAEs in the tree are matched
//! in as components.
//!
//! The rows mirror the external-LoRA posture and are deliberately second-class:
//! * `catalogScope: "external"` + `removable: false` — read in place, never copied,
//!   never offered for deletion.
//! * ids are namespaced `external_base_…` so they can never collide with a
//!   manifest model id.
//! * **`usable: false` for now.** The per-family remap/dequant loaders are the
//!   downstream slices (sc-10668+); until one lands, an assembled external model is
//!   surfaced with an `unusableReason` (the sc-10509 fail-closed-with-reason
//!   posture) rather than offered as a generation target. `assembly` +
//!   `components` record what was found so the loader slice and the UI can use it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sceneworks_core::base_weights::{
    detect_base_weight_file, BaseWeightDetection, ComponentRole, QuantFormat,
};
use sceneworks_core::external_roots::comfyui_base_dirs;
use sceneworks_core::lora_family::is_hidden_file;
use sceneworks_core::slug::slugify;
use serde_json::{json, Value};

/// Catalog `catalogScope` for a scanned, externally-owned base model. Models use
/// `catalogScope` (not the LoRA `scope`); `"external"` is new here (manifest models
/// are only ever `"builtin"` or `"user"`).
pub(crate) const EXTERNAL_SCOPE: &str = "external";

/// Prefix on every synthesized id, distinct from the external-LoRA `external_`
/// prefix so a base model and an adapter can never collide. Also the marker
/// `resolve_model_manifest_entry` keys on to forward an assembled row to the worker.
pub(crate) const EXTERNAL_ID_PREFIX: &str = "external_base_";

/// Reason attached to every assembled external base model until a per-family
/// load-in-place loader exists (sc-10668+). Even a fully-assembled model is not yet
/// runnable, so it is surfaced fail-closed rather than offered for generation.
const LOADER_PENDING_REASON: &str =
    "External ComfyUI base-model loading is not yet implemented (epic 10451 Phase 2).";

/// Directory nesting descended below each base subtree — the real trees are mostly
/// flat, but a couple nest a level; this is a runaway guard, not a real limit.
const MAX_SCAN_DEPTH: usize = 8;

/// Upper bound on files inspected per base subtree, so a pathological directory
/// yields a truncated list rather than a stalled API.
const MAX_FILES_PER_DIR: usize = 4096;

/// The base subtrees that **anchor** a virtual model — a file here is a model the
/// user would pick. `text_encoders/` and `vae/` are component pools, never anchors.
const ANCHOR_SUBDIRS: &[&str] = &["diffusion_models", "unet", "checkpoints"];

/// Memo of the detection for each file, keyed by path + identity on disk (size +
/// mtime). `model_catalog` rebuilds on every job-create; classifying a 20 GB base
/// file means parsing a safetensors header that can be a multi-thousand-entry JSON
/// blob, so — exactly as [`crate::external_loras::ExternalLoraCache`] does for
/// adapters — the walk and `stat` always run (added/changed/removed files appear at
/// once) but the header parse is skipped when size and mtime both match.
#[derive(Default)]
pub(crate) struct ExternalBaseModelCache {
    entries: HashMap<PathBuf, CachedDetection>,
}

#[derive(Clone)]
struct CachedDetection {
    modified: Option<SystemTime>,
    size: u64,
    detection: BaseWeightDetection,
}

impl ExternalBaseModelCache {
    fn get(
        &self,
        path: &Path,
        modified: Option<SystemTime>,
        size: u64,
    ) -> Option<BaseWeightDetection> {
        let entry = self.entries.get(path)?;
        let modified = modified?;
        let cached_modified = entry.modified?;
        (entry.size == size && cached_modified == modified).then(|| entry.detection.clone())
    }

    fn insert(
        &mut self,
        path: PathBuf,
        modified: Option<SystemTime>,
        size: u64,
        detection: BaseWeightDetection,
    ) {
        self.entries.insert(
            path,
            CachedDetection {
                modified,
                size,
                detection,
            },
        );
    }

    fn retain_seen(&mut self, seen: &HashSet<PathBuf>) {
        self.entries.retain(|path, _| seen.contains(path));
    }
}

/// A single classified base-weight file found on disk.
struct DetectedFile {
    /// Canonical path to the file (the operator's own file — never copied).
    path: PathBuf,
    /// Which base subtree it was found in (`diffusion_models`, `vae`, …).
    subdir: &'static str,
    /// Display name relative to the subtree, e.g. `Wan/high_noise` — kept so files
    /// filed under a nested folder stay distinguishable.
    name: String,
    detection: BaseWeightDetection,
}

impl DetectedFile {
    fn verdict(&self) -> Option<(&Option<String>, ComponentRole, QuantFormat)> {
        match &self.detection {
            BaseWeightDetection::Recognized(v) => Some((&v.family, v.component, v.quant)),
            BaseWeightDetection::Unrecognized { .. } => None,
        }
    }

    fn component_json(&self) -> Value {
        let (family, role, quant) = match &self.detection {
            BaseWeightDetection::Recognized(v) => (
                v.family.clone(),
                Some(v.component.as_str()),
                Some(v.quant.as_str()),
            ),
            BaseWeightDetection::Unrecognized { .. } => (None, None, None),
        };
        json!({
            "name": self.name,
            "subdir": self.subdir,
            "role": role,
            "family": family,
            "quant": quant,
            "path": self.path.display().to_string(),
        })
    }
}

/// Scan each configured root's base subtrees and return synthesized model-catalog
/// rows, one per assembled virtual model (a transformer + its matched components,
/// or an all-in-one checkpoint). Returns empty when no roots are configured — the
/// default — keeping the catalog byte-identical for every install that has not
/// opted in.
///
/// Blocking filesystem work — call from `spawn_blocking`, as `model_catalog` does
/// for the manifest install-state sweep.
pub(crate) fn scan_external_base_models(
    roots: &[PathBuf],
    cache: &mut ExternalBaseModelCache,
) -> Vec<Value> {
    let mut detected: Vec<DetectedFile> = Vec::new();
    let mut seen = HashSet::new();

    for (subdir, dir) in comfyui_base_dirs(roots) {
        // Canonicalize once: every file must stay under this, so a symlink inside
        // the subtree cannot walk the scan out of the operator's declared root.
        let Ok(canonical_dir) = std::fs::canonicalize(&dir) else {
            continue;
        };
        for file in collect_weight_files(&canonical_dir) {
            let Some(detection) = classify(&file, cache) else {
                continue;
            };
            seen.insert(file.clone());
            let name = relative_display_name(file.strip_prefix(&canonical_dir).unwrap_or(&file));
            detected.push(DetectedFile {
                path: file,
                subdir,
                name,
                detection,
            });
        }
    }
    cache.retain_seen(&seen);

    assemble_models(detected)
}

/// Depth-bounded walk returning every `.safetensors`/`.gguf` file whose canonical
/// path is still under `root`. Hidden entries are skipped (an external ComfyUI
/// folder is a prime home for macOS AppleDouble `._x` sidecars). Symlinks are
/// followed then re-checked for containment, matching the worker's confinement — a
/// link escaping the root is dropped, since a row the worker would refuse to load
/// is worse than no row. Mirrors [`crate::external_loras::collect_adapters`].
fn collect_weight_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0_usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if files.len() >= MAX_FILES_PER_DIR {
                tracing::warn!(
                    root = %root.display(),
                    limit = MAX_FILES_PER_DIR,
                    "external base-model scan hit its per-dir cap; remaining files ignored"
                );
                return files;
            }
            let path = entry.path();
            if is_hidden_file(&path) {
                continue;
            }
            let Ok(canonical) = std::fs::canonicalize(&path) else {
                continue;
            };
            if !canonical.starts_with(root) {
                continue;
            }
            if canonical.is_dir() {
                if depth < MAX_SCAN_DEPTH {
                    stack.push((canonical, depth + 1));
                }
            } else if has_weight_extension(&canonical) {
                files.push(canonical);
            }
        }
    }
    files
}

/// True when `path` names a base-weight container: a `.safetensors` or a `.gguf`
/// (case-insensitively — re-hosted checkpoints ship `.SAFETENSORS`). GGUF content
/// is confirmed by magic in [`detect_base_weight_file`]; the extension only decides
/// what to hand the detector.
fn has_weight_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("safetensors") || extension.eq_ignore_ascii_case("gguf")
        })
}

/// Classify a file, using the memo when size + mtime match. `None` when the file
/// vanished between the walk and the `stat`, or its header could not be read at all
/// (a truncated download) — such a file is dropped and, deliberately, **not**
/// memoized, so a still-downloading file is retried on the next build.
fn classify(path: &Path, cache: &mut ExternalBaseModelCache) -> Option<BaseWeightDetection> {
    let metadata = std::fs::metadata(path).ok()?;
    let size = metadata.len();
    let modified = metadata.modified().ok();
    if let Some(detection) = cache.get(path, modified, size) {
        return Some(detection);
    }
    let detection = detect_base_weight_file(path).ok()?;
    cache.insert(path.to_path_buf(), modified, size, detection.clone());
    Some(detection)
}

/// A stable, human-meaningful name from a file's path relative to its subtree:
/// `Wan/high_noise.safetensors` → `Wan/high_noise`.
fn relative_display_name(relative: &Path) -> String {
    let without_extension = relative.with_extension("");
    without_extension
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Group detected files into virtual models: each transformer/checkpoint anchor
/// becomes one row, with text encoders and VAEs matched in from the pools.
fn assemble_models(detected: Vec<DetectedFile>) -> Vec<Value> {
    // Component pools: everything the detector calls a text encoder / VAE, wherever
    // it was filed (usually `text_encoders/` and `vae/`).
    let text_encoders: Vec<&DetectedFile> = detected
        .iter()
        .filter(|file| matches!(file.verdict(), Some((_, ComponentRole::TextEncoder, _))))
        .collect();
    let vaes: Vec<&DetectedFile> = detected
        .iter()
        .filter(|file| matches!(file.verdict(), Some((_, ComponentRole::Vae, _))))
        .collect();

    let mut rows = Vec::new();
    let mut used_ids = HashSet::new();

    // Wan2.2 is a dual-expert MoE (sc-10671): pair the high/low-noise transformer files into ONE model
    // (a single expert can't run), handled entirely here — the paired indices are then skipped below.
    // The tree's UMT5 TE + VAE are folded in when present (sc-10909), read in place like the experts.
    let (wan_rows, wan_consumed) =
        assemble_wan_experts(&detected, &text_encoders, &vaes, &mut used_ids);
    rows.extend(wan_rows);

    for (index, anchor) in detected.iter().enumerate() {
        // Only files in an anchor subtree seed a model; a text encoder or VAE is a
        // component, never a standalone model.
        if !ANCHOR_SUBDIRS.contains(&anchor.subdir) {
            continue;
        }
        // Wan MoE experts were already emitted (paired or as an incomplete lone expert).
        if wan_consumed.contains(&index) {
            continue;
        }
        if let Some(row) = assemble_one(anchor, &text_encoders, &vaes, &mut used_ids) {
            rows.push(row);
        }
    }
    rows
}

/// The high/low-noise level a Wan2.2 MoE expert file names. Filename is the only discriminator (the two
/// experts are structurally identical), matching the sc-10592 external-LoRA `.high_noise`/`.low_noise`
/// pairing. Case-insensitive; `None` when neither (or both) token is present.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WanExpertLevel {
    High,
    Low,
}

/// One pairing group of Wan MoE experts (`detected` indices): the high/low-noise expert (if each was
/// found) plus every index that hashed to this group's base.
#[derive(Default)]
struct WanExpertGroup {
    high: Option<usize>,
    low: Option<usize>,
    all: Vec<usize>,
}

fn wan_expert_level(name: &str) -> Option<WanExpertLevel> {
    let lower = name.to_ascii_lowercase();
    match (lower.contains("high"), lower.contains("low")) {
        (true, false) => Some(WanExpertLevel::High),
        (false, true) => Some(WanExpertLevel::Low),
        _ => None,
    }
}

/// The pairing key for a Wan expert: its name with the high/low token normalized out, so the two
/// noise-level siblings share a base (`wan2.2_t2v_high_noise_…` and `…_low_noise_…` → the same base).
/// t2v vs i2v keep distinct bases (the `t2v`/`i2v` token survives), so they never cross-pair.
fn wan_pair_base(name: &str) -> String {
    name.to_ascii_lowercase()
        .replace("high", "")
        .replace("low", "")
}

/// A Wan transformer component JSON with its MoE role (`transformer_high` / `transformer_low`) — the
/// roles the worker's `resolve_wan_comfyui_paths` reads. Overrides the detector's plain `transformer`.
fn wan_component_json(file: &DetectedFile, role: &str) -> Value {
    let mut component = file.component_json();
    component["role"] = Value::String(role.to_owned());
    component
}

/// Pair Wan2.2 MoE transformer experts (sc-10671). Returns one row per pairing group — a **complete**
/// model with both `transformer_high` + `transformer_low` components when both noise levels are present,
/// or an **incomplete** unusable row per lone expert when its sibling is missing. Also returns the set
/// of `detected` indices consumed, so `assemble_models` skips them (a single Wan expert must never fall
/// to the snapshot-backed single-transformer path, which would wrongly mark it runnable).
///
/// The tree's **UMT5 text encoder** (a `t5` encoder, itself scaled-fp8) and **VAE** are folded into the
/// complete row as `text_encoder` / `vae` components when present (sc-10909) — the worker then reads
/// them in place (scaled-fp8 dequant / native-key remap) instead of the snapshot tier. This is purely
/// additive: the tiny tokenizer (and either component when absent) still comes from the resident
/// snapshot, so the row is `complete` and runnable regardless of whether the TE/VAE were folded.
fn assemble_wan_experts(
    detected: &[DetectedFile],
    text_encoders: &[&DetectedFile],
    vaes: &[&DetectedFile],
    used_ids: &mut HashSet<String>,
) -> (Vec<Value>, HashSet<usize>) {
    // Group the wan-video transformers by their pairing base (the name with the high/low token
    // normalized out), tracking the high/low expert index + every index in the group.
    let mut groups: std::collections::BTreeMap<String, WanExpertGroup> =
        std::collections::BTreeMap::new();
    for (index, file) in detected.iter().enumerate() {
        if !ANCHOR_SUBDIRS.contains(&file.subdir) {
            continue;
        }
        if !matches!(file.verdict(), Some((Some(fam), ComponentRole::Transformer, _)) if fam == "wan-video")
        {
            continue;
        }
        let entry = groups.entry(wan_pair_base(&file.name)).or_default();
        match wan_expert_level(&file.name) {
            Some(WanExpertLevel::High) if entry.high.is_none() => entry.high = Some(index),
            Some(WanExpertLevel::Low) if entry.low.is_none() => entry.low = Some(index),
            _ => {}
        }
        entry.all.push(index);
    }

    let mut rows = Vec::new();
    let mut consumed = HashSet::new();
    for (_base, group) in groups {
        let WanExpertGroup { high, low, all } = group;
        for index in &all {
            consumed.insert(*index);
        }
        match (high, low) {
            // Both experts present → one complete MoE model (runnable at scaled_fp8_companion).
            (Some(high), Some(low)) => {
                let anchor = &detected[high];
                let mut components = vec![
                    wan_component_json(anchor, "transformer_high"),
                    wan_component_json(&detected[low], "transformer_low"),
                ];
                // sc-10909: fold the tree's UMT5 TE (a `t5` encoder — `required_text_encoders`) and an
                // unambiguous VAE in as in-place components. Additive only; whichever is absent falls
                // back to the snapshot tier at load, so completeness/runnability is unaffected.
                if let Some(encoder) = text_encoders.iter().find(
                    |encoder| matches!(encoder.verdict(), Some((Some(fam), _, _)) if fam == "t5"),
                ) {
                    components.push(encoder.component_json());
                }
                if let [only_vae] = vaes {
                    components.push(only_vae.component_json());
                }
                let id = unique_id(&anchor.name, used_ids);
                rows.push(finish_row(
                    id,
                    anchor,
                    Some("wan-video".to_owned()),
                    "complete",
                    LOADER_PENDING_REASON.to_owned(),
                    components,
                ));
            }
            // A lone expert (missing its high/low sibling) can't run — surfaced incomplete.
            _ => {
                for index in &all {
                    let file = &detected[*index];
                    let role = match wan_expert_level(&file.name) {
                        Some(WanExpertLevel::Low) => "transformer_low",
                        _ => "transformer_high",
                    };
                    let id = unique_id(&file.name, used_ids);
                    rows.push(finish_row(
                        id,
                        file,
                        Some("wan-video".to_owned()),
                        "incomplete",
                        "Incomplete: the other Wan MoE expert (high/low-noise) was not found in the \
                         tree — both are required."
                            .to_owned(),
                        vec![wan_component_json(file, role)],
                    ));
                }
            }
        }
    }
    (rows, consumed)
}

/// Text-encoder families that satisfy a given DiT family (best-effort labels from
/// the base-weight detector), and whether the family needs a VAE. `None` for a
/// family we do not yet model — its anchor is still surfaced, but as `unassemblable`.
///
/// Deliberately conservative and grounded in what SceneWorks ships: precise
/// component pairing (exactly which encoder/VAE snapshot a family loads) is the
/// loader slice's job (sc-10668+); here we only decide "are the raw materials
/// present". VAE families are intentionally not pinned — the Wan and Qwen 3D VAEs
/// are byte-identical by key, so the detector returns no VAE family, and requiring
/// one would false-negative every assembly.
/// Families the worker loads **DiT-in-place, snapshot-backed** (epic 10451 Phase 2b,
/// sc-10670): the diffusion transformer is read from the ComfyUI tree, but the text
/// encoder / VAE / tokenizer come from a resident SceneWorks diffusers snapshot —
/// not the tree. So a bare DiT anchor is a **complete** assembly on its own; the
/// tree's own components are a different quant/key-schema (separate slices) and are
/// deliberately not folded in.
///
/// `qwen-image`: the ComfyUI Qwen2.5-VL text encoders are themselves *scaled*-fp8
/// (sc-10671) and the tree VAE uses native WAN-VAE keys (a 194-key 3D-VAE remap), so
/// both are sourced from our snapshot while the plain-fp8 DiT is read in place.
///
/// `flux2` (sc-10680): the `flux2_dev_fp8mixed` file is a bare **inline-scale fp8** DiT
/// with no text encoder / VAE — the Mistral-3 TE + AutoencoderKL-Flux2 + tokenizer are
/// sourced from our snapshot while the fp8 DiT is dequanted + read in place.
fn is_snapshot_backed(family: &str) -> bool {
    // `wan-video` is also snapshot-backed but takes the dedicated MoE-pairing path
    // ([`assemble_wan_experts`]), never the single-transformer branch this gates.
    matches!(family, "qwen-image" | "flux2")
}

/// Whether a fully-assembled external model has a worker loader for its family+quant,
/// i.e. is offerable as a generation target (the picker offers only `usable !== false`).
/// Each `(family, quant)` here corresponds to a landed load-in-place slice:
/// z-image bf16 (sc-10668) · qwen-image plain fp8_e4m3 (sc-10670) · wan-video
/// scaled_fp8_companion (sc-10671) · flux2 inline-scale fp8 (sc-10680). Everything else
/// stays fail-closed until its slice lands. Only ever true for a `complete` assembly.
fn family_quant_runnable(family: Option<&str>, quant: Option<&str>, assembly: &str) -> bool {
    assembly == "complete"
        && matches!(
            (family, quant),
            (Some("z-image"), Some("bf16"))
                | (Some("qwen-image"), Some("fp8_e4m3"))
                | (Some("wan-video"), Some("scaled_fp8_companion"))
                | (Some("flux2"), Some("fp8_inline_scale"))
        )
}

fn required_text_encoders(family: &str) -> Option<&'static [&'static str]> {
    match family {
        // Z-Image pairs with Qwen3-4B; Wan with UMT5 (a T5 encoder); FLUX.2 with
        // Mistral-3 — all reliably labelled by the detector.
        "z-image" => Some(&["qwen3"]),
        "wan-video" => Some(&["t5"]),
        "flux2" => Some(&["mistral"]),
        // Qwen-Image uses Qwen2.5-VL, which the detector does not yet label (it is
        // neither the Qwen3 q_norm/k_norm shape nor a tagged vision tower). Left out
        // deliberately rather than mis-pairing it with the Qwen3 encoder — its anchor
        // is still surfaced, as `unassemblable`, until the encoder detection lands.
        _ => None,
    }
}

/// Map a detected family to a catalog `type` so external rows group with the
/// manifest models of the same modality.
fn model_type_for_family(family: &str) -> &'static str {
    match family {
        "wan-video" | "ltx-video" => "video",
        _ => "image",
    }
}

fn assemble_one(
    anchor: &DetectedFile,
    text_encoders: &[&DetectedFile],
    vaes: &[&DetectedFile],
    used_ids: &mut HashSet<String>,
) -> Option<Value> {
    let id = unique_id(&anchor.name, used_ids);
    let mut components = vec![anchor.component_json()];

    // `assembly` records completeness; `usable` stays false regardless (no loader
    // yet). `reason` prefers the most actionable message.
    let (family, assembly, reason): (Option<String>, &str, String) = match anchor.verdict() {
        // An unreadable/unclassifiable anchor: surfaced so the user sees we found a
        // file we could not use, with the detector's reason (sc-10509 posture).
        None => {
            let detail = match &anchor.detection {
                BaseWeightDetection::Unrecognized { reason } => reason.clone(),
                BaseWeightDetection::Recognized(_) => String::new(),
            };
            (
                None,
                "unrecognized",
                format!("Unrecognized base weight: {detail}"),
            )
        }
        // An all-in-one checkpoint carries every role itself — complete standalone.
        Some((family, ComponentRole::Checkpoint, _)) => {
            (family.clone(), "complete", LOADER_PENDING_REASON.to_owned())
        }
        // A VAE or text encoder filed in an anchor subtree is a component, not a
        // model; skip it rather than surfacing a bogus row.
        Some((_, ComponentRole::Vae | ComponentRole::TextEncoder, _)) => return None,
        // A transformer needs its companion encoder + VAE matched in from the tree.
        Some((family, ComponentRole::Transformer, _)) => {
            let Some(family_name) = family.clone() else {
                return Some(finish_row(
                    id,
                    anchor,
                    None,
                    "unrecognized",
                    "Unrecognized diffusion-transformer architecture.".to_owned(),
                    components,
                ));
            };
            // Snapshot-backed families (sc-10670): the DiT is read in place; the TE + tokenizer come
            // from a resident SceneWorks snapshot (the tree's Qwen2.5-VL TE is scaled-fp8, sc-10671),
            // so a bare DiT anchor is complete on its own. The **VAE**, however, now has an in-place
            // loader (sc-10830: native WAN-VAE keys → diffusers remap), so when the tree carries an
            // unambiguous VAE it is folded into the row and read in place; ambiguous/absent falls back
            // to the snapshot VAE (still complete). Runnability stays decided by the DiT's family+quant
            // in `finish_row`.
            if is_snapshot_backed(&family_name) {
                if let [only_vae] = vaes {
                    components.push(only_vae.component_json());
                }
                return Some(finish_row(
                    id,
                    anchor,
                    Some(family_name),
                    "complete",
                    LOADER_PENDING_REASON.to_owned(),
                    components,
                ));
            }
            match required_text_encoders(&family_name) {
                None => (
                    Some(family_name),
                    "unassemblable",
                    "Component assembly for this family is not yet modeled.".to_owned(),
                ),
                Some(accepted) => {
                    let matched_encoder = text_encoders.iter().find(|encoder| {
                        matches!(encoder.verdict(), Some((Some(fam), _, _)) if accepted.contains(&fam.as_str()))
                    });
                    if let Some(encoder) = matched_encoder {
                        components.push(encoder.component_json());
                    }
                    // VAE families are byte-identical across model families (the Wan and
                    // Qwen 3D VAEs share every key), so the detector cannot label them.
                    // Completeness only requires that *a* VAE is present; a specific one
                    // is listed as a component only when the tree is unambiguous (exactly
                    // one). The loader slice resolves the exact VAE per family (sc-10668+).
                    let vae_present = !vaes.is_empty();
                    if let [only_vae] = vaes {
                        components.push(only_vae.component_json());
                    }
                    if matched_encoder.is_some() && vae_present {
                        (
                            Some(family_name),
                            "complete",
                            LOADER_PENDING_REASON.to_owned(),
                        )
                    } else {
                        let mut missing = Vec::new();
                        if matched_encoder.is_none() {
                            missing.push(format!("a {} text encoder", accepted.join("/")));
                        }
                        if !vae_present {
                            missing.push("a VAE".to_owned());
                        }
                        (
                            Some(family_name),
                            "incomplete",
                            format!(
                                "Incomplete: no {} found in the tree.",
                                missing.join(" and ")
                            ),
                        )
                    }
                }
            }
        }
    };

    Some(finish_row(id, anchor, family, assembly, reason, components))
}

fn finish_row(
    id: String,
    anchor: &DetectedFile,
    family: Option<String>,
    assembly: &str,
    reason: String,
    components: Vec<Value>,
) -> Value {
    let quant = match &anchor.detection {
        BaseWeightDetection::Recognized(v) => Some(v.quant.as_str()),
        BaseWeightDetection::Unrecognized { .. } => None,
    };
    // A model is **runnable** only when a worker loader exists for its family+quant and
    // the assembly is complete (all component paths resolved). sc-10668 wired z-image bf16;
    // sc-10670 wires qwen-image plain fp8_e4m3. Everything else stays fail-closed
    // (`usable:false` + reason) until its loader slice lands — the web picker offers only
    // `usable !== false`.
    let runnable = family_quant_runnable(family.as_deref(), quant, assembly);
    let mut row = json!({
        "id": id,
        "name": format!("{} (ComfyUI)", anchor.name),
        "type": family.as_deref().map(model_type_for_family).unwrap_or("image"),
        "catalogScope": EXTERNAL_SCOPE,
        "removable": false,
        "downloadable": false,
        "downloads": [],
        "installState": "installed",
        "installedPath": anchor.path.display().to_string(),
        "manifestPath": Value::Null,
        "source": { "path": anchor.path.display().to_string() },
        "usable": runnable,
        "unusableReason": if runnable { Value::Null } else { Value::String(reason) },
        "assembly": assembly,
        "components": components,
    });
    if let Some(family) = family {
        row["family"] = Value::String(family);
    }
    if let Some(quant) = quant {
        row["quant"] = Value::String(quant.to_owned());
    }
    row
}

/// `external_base_<slug>`, suffixed on collision so two anchors that slugify alike
/// keep distinct ids (the id is the catalog's primary key).
fn unique_id(display_name: &str, used_ids: &mut HashSet<String>) -> String {
    let base = format!(
        "{EXTERNAL_ID_PREFIX}{}",
        slugify(display_name, "model", Some(80))
    );
    let mut candidate = base.clone();
    let mut suffix = 2_usize;
    while used_ids.contains(&candidate) {
        candidate = format!("{base}_{suffix}");
        suffix += 1;
    }
    used_ids.insert(candidate.clone());
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    /// Write a safetensors file whose header declares `(name, dtype)` tensors. Only
    /// the header is read, but the declared data must be present or the header is
    /// rejected as truncated (sc-6072).
    fn write_safetensors(path: &Path, entries: &[(&str, &str)]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent dir");
        }
        let mut header = Map::new();
        for (index, (key, dtype)) in entries.iter().enumerate() {
            let start = index * 4;
            header.insert(
                (*key).to_owned(),
                json!({ "dtype": dtype, "shape": [1], "data_offsets": [start, start + 4] }),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header json");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend(std::iter::repeat(0_u8).take(entries.len() * 4));
        std::fs::write(path, bytes).expect("write safetensors");
    }

    fn z_image_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("cap_embedder.0.weight", "BF16"),
            ("noise_refiner.0.attention.qkv.weight", "BF16"),
            ("layers.0.attention.qkv.weight", "BF16"),
            ("layers.0.feed_forward.w1.weight", "BF16"),
        ]
    }
    fn qwen3_encoder_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.embed_tokens.weight", "BF16"),
            ("model.layers.0.self_attn.q_proj.weight", "BF16"),
            ("model.layers.0.self_attn.q_norm.weight", "BF16"),
            ("model.layers.0.self_attn.k_norm.weight", "BF16"),
            ("model.layers.0.mlp.gate_proj.weight", "BF16"),
        ]
    }
    fn vae_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("encoder.conv_in.weight", "F32"),
            ("encoder.down.0.block.0.conv1.weight", "F32"),
            ("decoder.conv_out.weight", "F32"),
            ("decoder.up.0.block.0.conv1.weight", "F32"),
        ]
    }
    /// A ComfyUI UMT5-XXL text encoder (`text_encoders/umt5_xxl_fp8_e4m3fn_scaled`): T5 encoder keys +
    /// a `.scale_weight` companion → detector `(t5, text_encoder, scaled_fp8_companion)`.
    fn umt5_scaled_te_keys() -> Vec<(&'static str, &'static str)> {
        vec![
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
        ]
    }

    fn models_root(temp: &Path) -> PathBuf {
        temp.join("ComfyUI").join("models")
    }

    fn scan(roots: &[PathBuf]) -> Vec<Value> {
        scan_external_base_models(roots, &mut ExternalBaseModelCache::default())
    }

    #[test]
    fn no_roots_configured_yields_no_rows() {
        assert!(scan(&[]).is_empty());
    }

    #[test]
    fn a_root_without_base_subtrees_is_empty() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        std::fs::create_dir_all(root.join("loras")).expect("mkdir");
        assert!(scan(&[root]).is_empty());
    }

    #[test]
    fn z_image_bf16_with_encoder_and_vae_assembles_complete_and_runnable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root.join("unet").join("z_image_turbo_bf16.safetensors"),
            &z_image_keys(),
        );
        write_safetensors(
            &root.join("text_encoders").join("qwen_3_4b.safetensors"),
            &qwen3_encoder_keys(),
        );
        write_safetensors(&root.join("vae").join("ae.safetensors"), &vae_keys());

        let rows = scan(&[root]);
        assert_eq!(
            rows.len(),
            1,
            "one anchored model, encoder+vae folded in as components"
        );
        let row = &rows[0];
        assert_eq!(row["family"], "z-image");
        assert_eq!(row["type"], "image");
        assert_eq!(row["catalogScope"], EXTERNAL_SCOPE);
        assert_eq!(row["removable"], false);
        assert_eq!(row["assembly"], "complete");
        // sc-10668: z-image bf16 complete is now runnable (the candle loader exists),
        // so it is usable and the picker offers it. No unusable reason.
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        assert_eq!(row["quant"], "bf16");
        assert_eq!(row["components"].as_array().unwrap().len(), 3);
        assert!(row["id"].as_str().unwrap().starts_with(EXTERNAL_ID_PREFIX));
    }

    #[test]
    fn z_image_without_encoder_is_incomplete_with_reason() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root.join("unet").join("z_image_turbo_bf16.safetensors"),
            &z_image_keys(),
        );
        write_safetensors(&root.join("vae").join("ae.safetensors"), &vae_keys());
        // No text_encoders/ at all.

        let rows = scan(&[root]);
        let row = &rows[0];
        assert_eq!(row["assembly"], "incomplete");
        assert_eq!(row["usable"], false);
        let reason = row["unusableReason"].as_str().unwrap();
        assert!(reason.contains("text encoder"), "reason: {reason}");
        // The VAE it did find is still listed as a component.
        assert_eq!(row["components"].as_array().unwrap().len(), 2);
    }

    /// A ComfyUI Qwen-Image DiT (`diffusion_models/*_fp8_e4m3fn`): dual-stream MMDiT keys under the
    /// BFL `model.diffusion_model.` prefix, all `F8_E4M3` → detector `(qwen-image, transformer,
    /// fp8_e4m3)`.
    fn qwen_image_dit_fp8_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.diffusion_model.img_in.weight", "F8_E4M3"),
            (
                "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
                "F8_E4M3",
            ),
            (
                "model.diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight",
                "F8_E4M3",
            ),
        ]
    }

    #[test]
    fn qwen_image_fp8_dit_is_snapshot_backed_complete_and_runnable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only the DiT is on disk — no tree text encoder / VAE. Snapshot-backed families (sc-10670)
        // source the TE / VAE / tokenizer from a resident SceneWorks snapshot, so a bare DiT anchor is
        // still complete and runnable.
        write_safetensors(
            &root
                .join("diffusion_models")
                .join("qwen_image_2512_fp8_e4m3fn.safetensors"),
            &qwen_image_dit_fp8_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "one anchored DiT model");
        let row = &rows[0];
        assert_eq!(row["family"], "qwen-image");
        assert_eq!(row["type"], "image");
        assert_eq!(row["quant"], "fp8_e4m3");
        assert_eq!(row["assembly"], "complete");
        // sc-10670: plain-fp8 qwen DiT has a loader → usable, no reason. No tree VAE here, so
        // components is the DiT alone (the TE is always snapshot-sourced, sc-10671).
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        assert_eq!(row["components"].as_array().unwrap().len(), 1);
    }

    /// A ComfyUI FLUX.2-dev fp8-mixed DiT (`diffusion_models/flux2_dev_fp8mixed`): BFL-native MMDiT keys
    /// with **inline-scale fp8** MLPs (`.weight` F8_E4M3 + `.weight_scale`/`.input_scale` F32 siblings)
    /// and BF16 attention/modulation → detector `(flux2, transformer, fp8_inline_scale)` (sc-10662).
    fn flux2_dev_dit_inline_scale_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("double_stream_modulation_img.lin.weight", "BF16"),
            ("single_stream_modulation.lin.weight", "BF16"),
            ("double_blocks.0.img_attn.qkv.weight", "BF16"),
            ("double_blocks.0.img_mlp.0.weight", "F8_E4M3"),
            ("double_blocks.0.img_mlp.0.weight_scale", "F32"),
            ("double_blocks.0.img_mlp.0.input_scale", "F32"),
            ("single_blocks.0.linear1.weight", "F8_E4M3"),
            ("single_blocks.0.linear1.weight_scale", "F32"),
            ("single_blocks.0.linear1.input_scale", "F32"),
        ]
    }

    #[test]
    fn flux2_inline_scale_dit_is_snapshot_backed_complete_and_runnable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only the DiT is on disk — no tree text encoder / VAE. flux2 is snapshot-backed (sc-10680): the
        // Mistral-3 TE / AutoencoderKL-Flux2 / tokenizer come from a resident SceneWorks snapshot, so a
        // bare inline-scale-fp8 DiT anchor is still complete and runnable.
        write_safetensors(
            &root
                .join("diffusion_models")
                .join("flux2_dev_fp8mixed.safetensors"),
            &flux2_dev_dit_inline_scale_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "one anchored DiT model");
        let row = &rows[0];
        assert_eq!(row["family"], "flux2");
        assert_eq!(row["type"], "image");
        assert_eq!(row["quant"], "fp8_inline_scale");
        assert_eq!(row["assembly"], "complete");
        // sc-10680: inline-scale-fp8 flux2 DiT has a loader → usable, no reason. No tree VAE here, so
        // components is the DiT alone (the TE / VAE are snapshot-sourced).
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        assert_eq!(row["components"].as_array().unwrap().len(), 1);
    }

    /// A ComfyUI Wan2.2 A14B expert (`unet/wan2.2_*_fp8_scaled`): native-Wan dual-stream keys + a
    /// `.scale_weight` companion + the `scaled_fp8` marker → detector `(wan-video, transformer,
    /// scaled_fp8_companion)`.
    fn wan_expert_scaled_fp8_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("patch_embedding.weight", "F16"),
            ("blocks.0.self_attn.q.weight", "F8_E4M3"),
            ("blocks.0.self_attn.q.scale_weight", "F32"),
            ("blocks.0.cross_attn.q.weight", "F8_E4M3"),
            ("blocks.0.ffn.0.weight", "F8_E4M3"),
            ("scaled_fp8", "F8_E4M3"),
        ]
    }

    #[test]
    fn wan_moe_high_low_experts_pair_into_one_complete_runnable_model() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "two experts pair into one MoE model");
        let row = &rows[0];
        assert_eq!(row["family"], "wan-video");
        assert_eq!(row["type"], "video");
        assert_eq!(row["quant"], "scaled_fp8_companion");
        assert_eq!(row["assembly"], "complete");
        assert_eq!(row["usable"], true);
        assert_eq!(row["unusableReason"], Value::Null);
        // Both experts as components, tagged with their MoE roles.
        let components = row["components"].as_array().unwrap();
        assert_eq!(components.len(), 2);
        let roles: Vec<&str> = components
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert!(roles.contains(&"transformer_high"));
        assert!(roles.contains(&"transformer_low"));
    }

    #[test]
    fn wan_moe_folds_in_tree_umt5_te_and_vae() {
        // sc-10909: a complete MoE pair + the tree's UMT5 TE (t5, scaled-fp8) + one VAE → both are
        // folded in as in-place `text_encoder` / `vae` components (worker reads them in place). Still
        // complete + runnable (the tokenizer stays snapshot-sourced).
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        write_safetensors(
            &root
                .join("text_encoders")
                .join("umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            &umt5_scaled_te_keys(),
        );
        write_safetensors(
            &root.join("vae").join("wan_2.1_vae.safetensors"),
            &vae_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "experts + TE + VAE → one MoE model");
        let row = &rows[0];
        assert_eq!(row["family"], "wan-video");
        assert_eq!(row["assembly"], "complete");
        assert_eq!(row["usable"], true);
        let components = row["components"].as_array().unwrap();
        let roles: Vec<&str> = components
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert!(roles.contains(&"transformer_high"));
        assert!(roles.contains(&"transformer_low"));
        // The TE + VAE ride along as in-place components (worker reads them in place, sc-10909).
        assert!(roles.contains(&"text_encoder"), "roles: {roles:?}");
        assert!(roles.contains(&"vae"), "roles: {roles:?}");
    }

    #[test]
    fn wan_lone_expert_without_sibling_is_incomplete() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only the high-noise expert — no low-noise sibling.
        write_safetensors(
            &root
                .join("unet")
                .join("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
            &wan_expert_scaled_fp8_keys(),
        );
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["family"], "wan-video");
        assert_eq!(rows[0]["assembly"], "incomplete");
        assert_eq!(rows[0]["usable"], false);
        let reason = rows[0]["unusableReason"].as_str().unwrap();
        assert!(reason.contains("MoE expert"), "reason: {reason}");
    }

    #[test]
    fn qwen_image_fp8_dit_folds_in_an_unambiguous_tree_vae() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // sc-10830: a ComfyUI Qwen-Image DiT + exactly one tree VAE → the VAE is folded into the row
        // (read in place, native WAN-VAE keys → diffusers remap) while the DiT stays snapshot-backed
        // for the TE + tokenizer. Runnable, and the VAE rides along as a `vae` component.
        write_safetensors(
            &root
                .join("diffusion_models")
                .join("qwen_image_2512_fp8_e4m3fn.safetensors"),
            &qwen_image_dit_fp8_keys(),
        );
        write_safetensors(
            &root.join("vae").join("qwen_image_vae.safetensors"),
            &vae_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "one anchored DiT model");
        let row = &rows[0];
        assert_eq!(row["family"], "qwen-image");
        assert_eq!(row["assembly"], "complete");
        assert_eq!(row["usable"], true);
        let components = row["components"].as_array().unwrap();
        assert_eq!(components.len(), 2, "DiT + the folded tree VAE");
        assert!(
            components
                .iter()
                .any(|component| component["role"] == "vae"),
            "the tree VAE rides along as a `vae` component for the in-place loader"
        );
    }

    #[test]
    fn all_in_one_checkpoint_is_complete_standalone() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // LTX all-in-one: DiT + an embedded audio_vae → ComponentRole::Checkpoint.
        write_safetensors(
            &root.join("checkpoints").join("ltx.safetensors"),
            &[
                ("model.diffusion_model.scale_shift_table", "F32"),
                ("model.diffusion_model.patchify_proj.weight", "BF16"),
                (
                    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                    "BF16",
                ),
                ("audio_vae.encoder.conv_in.conv.weight", "F32"),
                ("audio_vae.decoder.conv_out.conv.weight", "F32"),
            ],
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["family"], "ltx-video");
        assert_eq!(rows[0]["type"], "video");
        assert_eq!(rows[0]["assembly"], "complete");
        // No component pool needed: the checkpoint carries every role.
        assert_eq!(rows[0]["components"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn text_encoders_and_vaes_are_components_never_anchors() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Only encoders + VAEs, no transformer/checkpoint → nothing to anchor a model.
        write_safetensors(
            &root.join("text_encoders").join("qwen_3_4b.safetensors"),
            &qwen3_encoder_keys(),
        );
        write_safetensors(&root.join("vae").join("ae.safetensors"), &vae_keys());
        assert!(scan(&[root]).is_empty());
    }

    #[test]
    fn unrecognized_family_transformer_is_surfaced_unusable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        write_safetensors(
            &root.join("diffusion_models").join("mystery.safetensors"),
            &[("some.unknown.tensor", "BF16"), ("another.mystery", "BF16")],
        );
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["usable"], false);
        assert_eq!(rows[0]["assembly"], "unrecognized");
        assert!(rows[0].get("family").is_none());
    }

    #[test]
    fn a_family_without_a_component_spec_is_unassemblable_but_listed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        // Ideogram is a recognized DiT family with no component spec here yet.
        write_safetensors(
            &root.join("diffusion_models").join("ideo.safetensors"),
            &[
                ("embed_image_indicator.weight", "BF16"),
                ("layers.0.attention.qkv.weight", "BF16"),
                ("layers.0.adaln_modulation.weight", "BF16"),
            ],
        );
        let rows = scan(&[root]);
        assert_eq!(rows[0]["family"], "ideogram");
        assert_eq!(rows[0]["assembly"], "unassemblable");
        assert_eq!(rows[0]["usable"], false);
    }

    #[test]
    fn gguf_anchor_is_surfaced() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        let unet = root.join("unet");
        std::fs::create_dir_all(&unet).expect("mkdir");
        // GGUF magic + padding; detected by magic, classified quant=gguf.
        std::fs::write(unet.join("wan_Q4_K_S.gguf"), b"GGUF\x00\x00\x00\x00padding")
            .expect("write");
        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["quant"], "gguf");
        assert_eq!(rows[0]["usable"], false);
    }

    #[test]
    fn scan_populates_then_prunes_the_cache() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = models_root(temp.path());
        let anchor = root.join("unet").join("z_image_turbo_bf16.safetensors");
        write_safetensors(&anchor, &z_image_keys());

        let mut cache = ExternalBaseModelCache::default();
        let rows = scan_external_base_models(std::slice::from_ref(&root), &mut cache);
        assert_eq!(rows.len(), 1);
        assert_eq!(cache.entries.len(), 1);

        let again = scan_external_base_models(std::slice::from_ref(&root), &mut cache);
        assert_eq!(
            again, rows,
            "unchanged tree yields identical rows from cache"
        );

        std::fs::remove_file(&anchor).expect("remove");
        assert!(scan_external_base_models(&[root], &mut cache).is_empty());
        assert!(cache.entries.is_empty(), "vanished files pruned");
    }

    /// Manual smoke against a real ComfyUI tree — the only test over real key
    /// conventions. Ignored by default:
    ///
    /// ```text
    /// SCENEWORKS_EXTERNAL_MODEL_ROOTS='C:\Users\Michael\ComfyUI-Shared\models' \
    ///   cargo test -p sceneworks-rust-api --lib external_base_models::tests::real_comfyui_base_tree -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a real ComfyUI models tree via SCENEWORKS_EXTERNAL_MODEL_ROOTS"]
    fn real_comfyui_base_tree_smoke() {
        let roots = sceneworks_core::external_roots::parse_external_model_roots(
            std::env::var("SCENEWORKS_EXTERNAL_MODEL_ROOTS")
                .ok()
                .as_deref(),
        );
        assert!(!roots.is_empty(), "set SCENEWORKS_EXTERNAL_MODEL_ROOTS");
        let rows = scan(&roots);
        println!("\n{} external base models assembled:", rows.len());
        for row in &rows {
            println!(
                "  {:<10} {:<14} {:<8} {}",
                row.get("family")
                    .and_then(Value::as_str)
                    .unwrap_or("(none)"),
                row["assembly"].as_str().unwrap_or_default(),
                row.get("quant").and_then(Value::as_str).unwrap_or("-"),
                row["name"].as_str().unwrap_or_default(),
            );
        }
        assert!(
            !rows.is_empty(),
            "the real tree should assemble base models"
        );
    }
}
