//! Surface the LoRA adapters already sitting in an operator's ComfyUI `models/`
//! tree, read in place (epic 10451 / sc-10452).
//!
//! Every other LoRA in the catalog is *manifest-declared*: `lora_catalog` merges
//! `builtin.loras.jsonc`, `user.loras.jsonc` and a project manifest, and a LoRA
//! that no manifest names simply does not exist. Nothing scans a directory. That
//! is exactly why "just point at my ComfyUI folder" needs new code rather than a
//! config tweak — this module is that scan, producing synthetic catalog rows for
//! the adapters it finds.
//!
//! The rows are deliberately second-class:
//! * `scope: "external"` — `delete_lora` / `update_lora` reject the scope, and
//!   `lora_catalog` marks them `removable: false`. **We never offer to mutate or
//!   delete a file the user owns and we merely borrowed.**
//! * ids are namespaced `external_…` so a scanned adapter can never collide with,
//!   or shadow, a manifest entry.
//! * nothing is copied. `source.path` points straight at the operator's file; the
//!   worker's `normalize_app_managed_lora_path` admits it because the same root
//!   list is configured there.
//!
//! This lives in core so the API catalog and the worker's portable source
//! inventory run the *same* scan and therefore assign the same `external_…`
//! identities to the same files (SW-7030): the API publishes the row, the worker
//! advertises the registered digest claim for exactly that id.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::external_roots::{walk_confined, ExternalScanCache};
use crate::lora_family::{detect_lora_family, is_safetensors_file, read_safetensors_header};
use crate::slug::slugify;

/// Catalog `scope` for a scanned, externally-owned adapter.
pub const EXTERNAL_SCOPE: &str = "external";

/// Prefix on every synthesized id, so an external row can never collide with a
/// manifest-declared LoRA (whose ids are `<family>_<slug>`).
pub const EXTERNAL_ID_PREFIX: &str = "external_";

/// Directory nesting we will descend below `<root>/loras`. The real trees nest a
/// level or two (`loras/Wan/…`, `loras/Ltx2.3/…`); this is a runaway guard against
/// a pathological tree, not a real limit.
const MAX_SCAN_DEPTH: usize = 8;

/// Upper bound on adapters surfaced from one root. A user with a runaway directory
/// should get a truncated list rather than a stalled API.
const MAX_ADAPTERS_PER_ROOT: usize = 4096;

/// Cached header detections for external files.
pub type ExternalLoraCache = ExternalScanCache<Option<String>>;

/// Scan each configured root's `loras/` subdirectory and return synthesized catalog
/// rows, one per `.safetensors` adapter found. Returns empty when no roots are
/// configured (the default), which keeps the catalog byte-identical for every
/// install that has not opted in.
///
/// Blocking filesystem work — call from `spawn_blocking`, as `lora_catalog` does for
/// the manifest normalize sweep.
pub fn scan_external_loras(roots: &[PathBuf], cache: &mut ExternalLoraCache) -> Vec<Value> {
    let mut rows = Vec::new();
    let mut used_ids = HashSet::new();
    let mut seen = HashSet::new();
    for root in roots {
        let lora_dir = root.join(crate::external_roots::COMFYUI_LORA_SUBDIR);
        if !lora_dir.is_dir() {
            continue;
        }
        // Canonicalize once: every candidate must stay under this, so a symlink
        // inside the tree cannot walk us out of the root the operator declared.
        let Ok(canonical_root) = std::fs::canonicalize(&lora_dir) else {
            continue;
        };
        // A `loras/` that is itself a symlink out of the declared root is not the
        // operator's tree: both the API's confinement and the worker's would
        // refuse those paths at load time, so do not surface them at all.
        let Ok(canonical_model_root) = std::fs::canonicalize(root) else {
            continue;
        };
        if !canonical_root.starts_with(&canonical_model_root) {
            continue;
        }
        let adapters = collect_adapters(&canonical_root);
        let found = adapters.len();
        let mut detected = 0_usize;
        for adapter in adapters {
            match external_lora_row(&canonical_root, &adapter, &mut used_ids, cache) {
                Some(row) => {
                    if row.get("family").is_some() {
                        detected += 1;
                    }
                    seen.insert(adapter);
                    rows.push(row);
                }
                None => {
                    tracing::debug!(
                        path = %adapter.display(),
                        "skipping external LoRA with an unreadable safetensors header"
                    );
                }
            }
        }
        tracing::info!(
            root = %lora_dir.display(),
            found,
            detected_family = detected,
            "scanned external LoRA root"
        );
    }
    cache.retain_seen(&seen);
    rows
}

/// Depth-bounded walk of `root` returning every `.safetensors` file whose
/// canonical path is still under `root`. Hidden entries are skipped: an external
/// ComfyUI folder is exactly where macOS AppleDouble sidecars
/// (`._adapter.safetensors`) turn up, and each one would otherwise cost a header
/// read per catalog build and a slot against `MAX_ADAPTERS_PER_ROOT` — the cap
/// that silently drops real adapters once reached (SceneWorks#1333).
///
/// Symlinks are *followed*, then re-checked for containment: multi-install ComfyUI
/// setups routinely symlink a shared weights directory in, so refusing to follow
/// would hide legitimate files. A link that escapes the root is dropped instead —
/// which keeps this scan consistent with the worker's confinement check, where such
/// a path would be rejected at generation time anyway. Surfacing a row the worker
/// will later refuse to load would be worse than not surfacing it.
pub fn collect_adapters(root: &Path) -> Vec<PathBuf> {
    walk_confined(
        root,
        MAX_SCAN_DEPTH,
        MAX_ADAPTERS_PER_ROOT,
        is_safetensors_file,
    )
}

/// Build one catalog row for `adapter`. `None` when the file has no readable
/// safetensors header (truncated, or not actually safetensors) — such a file cannot
/// be loaded, so listing it would only produce a confusing failure later.
///
/// A readable header whose *family* the detector cannot identify still yields a row,
/// with `family` absent: the adapter is real and the user can see we found it. It
/// simply will not pass any model's compatibility gate, which is the honest outcome —
/// quietly dropping it would look like the scan had missed the file.
fn external_lora_row(
    root: &Path,
    adapter: &Path,
    used_ids: &mut HashSet<String>,
    cache: &mut ExternalLoraCache,
) -> Option<Value> {
    let metadata = std::fs::metadata(adapter).ok()?;
    let size = metadata.len();
    let modified = metadata.modified().ok();

    let family = match cache.get(adapter, modified, size) {
        Some(family) => family,
        None => {
            // Cache miss (new, changed, or mtime-less file): parse the header. A file
            // that will not parse is dropped entirely — and deliberately not memoized,
            // so a partially-written download is retried on the next catalog build
            // rather than being remembered as broken.
            let header = read_safetensors_header(adapter).ok()?;
            let family = detect_lora_family(&header);
            cache.insert(adapter.to_path_buf(), modified, size, family.clone());
            family
        }
    };

    let relative = adapter.strip_prefix(root).unwrap_or(adapter);
    let display_name = relative_display_name(relative);
    let id = unique_id(&display_name, used_ids);
    let file_name = adapter.file_name()?.to_str()?.to_owned();

    let mut row = json!({
        "id": id,
        "name": display_name,
        "scope": EXTERNAL_SCOPE,
        "source": { "path": adapter.display().to_string() },
        "files": [file_name],
        // The scan proved the file exists and parsed its header, so it is installed by
        // construction. There is no manifest behind it.
        "installedPath": adapter.display().to_string(),
        "installState": "installed",
        "manifestPath": Value::Null,
    });
    if let Some(family) = family {
        row["family"] = Value::String(family);
    }
    Some(row)
}

/// A stable, human-meaningful name from the adapter's path relative to `loras/`:
/// `Wan/detailz-wan.safetensors` → `Wan/detailz-wan`. Keeping the subdirectory
/// distinguishes same-named adapters filed under different families.
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

/// `external_<slug>`, suffixed on collision so two adapters that slugify alike keep
/// distinct ids (the id is the catalog's primary key and the deletion/selection handle).
fn unique_id(display_name: &str, used_ids: &mut HashSet<String>) -> String {
    let base = format!(
        "{EXTERNAL_ID_PREFIX}{}",
        slugify(display_name, "lora", Some(80))
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
