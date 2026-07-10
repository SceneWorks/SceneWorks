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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sceneworks_core::external_roots::comfyui_lora_dirs;
use sceneworks_core::lora_family::{
    detect_lora_family, is_safetensors_file, read_safetensors_header,
};
use sceneworks_core::slug::slugify;
use serde_json::{json, Value};

/// Catalog `scope` for a scanned, externally-owned adapter.
pub(crate) const EXTERNAL_SCOPE: &str = "external";

/// Prefix on every synthesized id, so an external row can never collide with a
/// manifest-declared LoRA (whose ids are `<family>_<slug>`).
const EXTERNAL_ID_PREFIX: &str = "external_";

/// Directory nesting we will descend below `<root>/loras`. The real trees nest a
/// level or two (`loras/Wan/…`, `loras/Ltx2.3/…`); this is a runaway guard against
/// a pathological tree, not a real limit.
const MAX_SCAN_DEPTH: usize = 8;

/// Upper bound on adapters surfaced from one root. A user with a runaway directory
/// should get a truncated list rather than a stalled API.
const MAX_ADAPTERS_PER_ROOT: usize = 4096;

/// Memo of the family detected for each adapter, keyed by its identity on disk.
///
/// `lora_catalog` is rebuilt on **every job-create**, and family detection means
/// parsing a safetensors header — for a real Wan adapter that is a ~1200-entry JSON
/// blob. A user with a large ComfyUI collection (the very user this feature targets)
/// would otherwise re-parse every header before every generation. The walk and the
/// `stat` stay cheap and always run, so added, removed, or *modified* files are
/// picked up immediately; only the header read is skipped, and only when size and
/// mtime both match what was parsed last time.
#[derive(Default)]
pub(crate) struct ExternalLoraCache {
    entries: HashMap<PathBuf, CachedAdapter>,
}

#[derive(Clone)]
struct CachedAdapter {
    modified: Option<SystemTime>,
    size: u64,
    /// `None` = header parsed fine but no family could be identified. Cached too,
    /// so an unidentifiable adapter is not re-parsed on every catalog build.
    family: Option<String>,
}

impl ExternalLoraCache {
    /// The cached family for `path` when the file is byte-identical (same size and
    /// mtime) to the one we parsed. Outer `None` = cache miss, parse the header.
    fn get(&self, path: &Path, modified: Option<SystemTime>, size: u64) -> Option<Option<String>> {
        let entry = self.entries.get(path)?;
        // A file whose mtime is unavailable can never be proven unchanged.
        let modified = modified?;
        let cached_modified = entry.modified?;
        (entry.size == size && cached_modified == modified).then(|| entry.family.clone())
    }

    fn insert(
        &mut self,
        path: PathBuf,
        modified: Option<SystemTime>,
        size: u64,
        family: Option<String>,
    ) {
        self.entries.insert(
            path,
            CachedAdapter {
                modified,
                size,
                family,
            },
        );
    }

    /// Drop memos for adapters that are no longer on disk, so a long-lived process
    /// that watches a churning directory does not grow without bound.
    fn retain_seen(&mut self, seen: &HashSet<PathBuf>) {
        self.entries.retain(|path, _| seen.contains(path));
    }
}

/// Scan each configured root's `loras/` subdirectory and return synthesized catalog
/// rows, one per `.safetensors` adapter found. Returns empty when no roots are
/// configured (the default), which keeps the catalog byte-identical for every
/// install that has not opted in.
///
/// Blocking filesystem work — call from `spawn_blocking`, as `lora_catalog` does for
/// the manifest normalize sweep.
pub(crate) fn scan_external_loras(roots: &[PathBuf], cache: &mut ExternalLoraCache) -> Vec<Value> {
    let mut rows = Vec::new();
    let mut used_ids = HashSet::new();
    let mut seen = HashSet::new();
    for lora_dir in comfyui_lora_dirs(roots) {
        // Canonicalize once: every candidate must stay under this, so a symlink
        // inside the tree cannot walk us out of the root the operator declared.
        let Ok(canonical_root) = std::fs::canonicalize(&lora_dir) else {
            continue;
        };
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
fn collect_adapters(root: &Path) -> Vec<PathBuf> {
    let mut adapters = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0_usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if adapters.len() >= MAX_ADAPTERS_PER_ROOT {
                tracing::warn!(
                    root = %root.display(),
                    limit = MAX_ADAPTERS_PER_ROOT,
                    "external LoRA scan hit its per-root cap; remaining files ignored"
                );
                return adapters;
            }
            let path = entry.path();
            // `canonicalize` resolves any symlink and proves existence; anything that
            // lands outside the declared root is not ours to read.
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
            } else if is_safetensors_file(&canonical) {
                adapters.push(canonical);
            }
        }
    }
    adapters
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    /// Write a syntactically valid safetensors file whose header declares `keys`.
    /// Only the header is ever read, but the declared tensor data must actually be
    /// present or `read_safetensors_header` rejects the file as truncated (sc-6072).
    fn write_safetensors(path: &Path, keys: &[&str]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent dir");
        }
        let mut header = Map::new();
        for (index, key) in keys.iter().enumerate() {
            let start = index * 4;
            header.insert(
                (*key).to_owned(),
                json!({ "dtype": "F32", "shape": [1], "data_offsets": [start, start + 4] }),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header json");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend(std::iter::repeat(0_u8).take(keys.len() * 4));
        std::fs::write(path, bytes).expect("write safetensors");
    }

    /// Key layout taken verbatim from a real ComfyUI Wan adapter
    /// (`wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors`).
    fn wan_keys() -> Vec<&'static str> {
        vec![
            "diffusion_model.blocks.0.cross_attn.k.lora_down.weight",
            "diffusion_model.blocks.0.cross_attn.k.lora_up.weight",
            "diffusion_model.blocks.0.cross_attn.v.lora_down.weight",
            "diffusion_model.blocks.0.cross_attn.v.lora_up.weight",
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
            "diffusion_model.blocks.0.self_attn.q.lora_up.weight",
            "diffusion_model.blocks.0.ffn.0.lora_down.weight",
            "diffusion_model.blocks.0.ffn.0.lora_up.weight",
        ]
    }

    fn comfy_root(temp: &Path) -> PathBuf {
        temp.join("ComfyUI").join("models")
    }

    /// Scan with a throwaway cache. Cache behaviour is asserted separately.
    fn scan(roots: &[PathBuf]) -> Vec<Value> {
        scan_external_loras(roots, &mut ExternalLoraCache::default())
    }

    #[test]
    fn no_roots_configured_yields_no_rows() {
        assert!(scan(&[]).is_empty());
    }

    #[test]
    fn a_root_without_a_loras_dir_is_skipped() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        std::fs::create_dir_all(root.join("checkpoints")).expect("mkdir");
        assert!(scan(&[root]).is_empty());
    }

    /// The real tree nests adapters under `loras/Wan/` and `loras/Ltx2.3/`, so a flat
    /// glob would miss most of them.
    #[test]
    fn scan_recurses_into_subdirectories_and_names_rows_by_relative_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        let loras = root.join("loras");
        write_safetensors(&loras.join("top-level.safetensors"), &wan_keys());
        write_safetensors(
            &loras.join("Wan").join("detailz-wan.safetensors"),
            &wan_keys(),
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 2, "both the flat and the nested adapter appear");

        let names: HashSet<&str> = rows
            .iter()
            .filter_map(|row| row.get("name").and_then(Value::as_str))
            .collect();
        assert!(names.contains("top-level"));
        assert!(
            names.contains("Wan/detailz-wan"),
            "subdir is kept in the name"
        );
    }

    /// SceneWorks#1333: an external ComfyUI folder is a prime home for macOS AppleDouble
    /// sidecars (`._adapter.safetensors`) — the folder is user-managed and often lives on
    /// an external or network volume. A sidecar carries the `.safetensors` extension, so an
    /// extension-only filter collects it. It then fails its header parse and is dropped, but
    /// only after costing a header read per catalog build and, worse, a slot against
    /// `MAX_ADAPTERS_PER_ROOT` — the cap that silently drops real adapters once reached.
    #[test]
    fn scan_skips_appledouble_sidecars() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        let loras = root.join("loras");
        write_safetensors(&loras.join("detailz-wan.safetensors"), &wan_keys());
        // A real AppleDouble header: magic 0x00051607, version 0x00020000.
        std::fs::write(
            loras.join("._detailz-wan.safetensors"),
            [0x00, 0x05, 0x16, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00],
        )
        .expect("write sidecar");

        // The sidecar must never even reach the header parse.
        let collected = collect_adapters(&std::fs::canonicalize(&loras).expect("canonical"));
        assert_eq!(
            collected.len(),
            1,
            "sidecar leaked into the adapter list: {collected:?}"
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1, "only the real adapter is surfaced");
        assert_eq!(
            rows[0].get("name").and_then(Value::as_str),
            Some("detailz-wan")
        );
    }

    /// Rows must be inert: namespaced id, external scope, no manifest, and a
    /// `source.path` pointing at the operator's own file (never a copy).
    #[test]
    fn rows_are_external_scoped_and_point_at_the_original_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        let adapter = root.join("loras").join("detailz-wan.safetensors");
        write_safetensors(&adapter, &wan_keys());

        let rows = scan(&[root]);
        let row = rows.first().expect("one row");

        assert_eq!(row["scope"], EXTERNAL_SCOPE);
        assert_eq!(row["installState"], "installed");
        assert_eq!(row["manifestPath"], Value::Null);
        assert!(row["id"]
            .as_str()
            .expect("id")
            .starts_with(EXTERNAL_ID_PREFIX));
        assert_eq!(row["files"], json!(["detailz-wan.safetensors"]));

        let source = row["source"]["path"].as_str().expect("source path");
        let canonical = adapter.canonicalize().expect("canonicalize");
        assert_eq!(Path::new(source), canonical.as_path());
        assert_eq!(
            row["installedPath"].as_str().expect("installedPath"),
            source
        );
    }

    /// The detector recognizes a real ComfyUI Wan key layout, so the row is
    /// compatibility-gated like any manifest LoRA. Guards the scanner against
    /// regressing into "everything is family-less".
    #[test]
    fn a_real_wan_key_layout_detects_its_family() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        write_safetensors(&root.join("loras").join("wan.safetensors"), &wan_keys());

        let rows = scan(&[root]);
        assert_eq!(
            rows[0].get("family").and_then(Value::as_str),
            Some("wan-video")
        );
    }

    /// A readable adapter whose family we cannot infer is still listed (with no
    /// `family`), rather than silently vanishing from a folder the user pointed us at.
    #[test]
    fn an_undetectable_family_still_yields_a_row_without_family() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        write_safetensors(
            &root.join("loras").join("mystery.safetensors"),
            &["some.unknown.tensor"],
        );

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].get("family").is_none());
        assert_eq!(rows[0]["installState"], "installed");
    }

    /// A file that is not safetensors (or is truncated) cannot be loaded, so it is
    /// dropped instead of becoming a row that fails at generation time.
    #[test]
    fn unreadable_headers_are_dropped() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        let loras = root.join("loras");
        std::fs::create_dir_all(&loras).expect("mkdir");
        std::fs::write(loras.join("garbage.safetensors"), b"not safetensors").expect("write");
        // A non-safetensors extension is not even considered.
        std::fs::write(loras.join("notes.txt"), b"hello").expect("write");

        assert!(scan(&[root]).is_empty());
    }

    /// Two adapters in different subdirectories can slugify identically; ids are the
    /// catalog's primary key, so they must stay distinct.
    #[test]
    fn colliding_slugs_get_distinct_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        let loras = root.join("loras");
        write_safetensors(&loras.join("a").join("style.safetensors"), &wan_keys());
        write_safetensors(&loras.join("b").join("style.safetensors"), &wan_keys());

        let rows = scan(&[root]);
        assert_eq!(rows.len(), 2);
        let ids: HashSet<&str> = rows
            .iter()
            .filter_map(|row| row.get("id").and_then(Value::as_str))
            .collect();
        assert_eq!(ids.len(), 2, "slug collision must not merge two adapters");
    }

    /// A cache entry is only reused when BOTH size and mtime match — the file we
    /// parsed is the file on disk. Anything else is a miss, so an edited adapter is
    /// re-read rather than served from a stale memo.
    #[test]
    fn cache_hits_require_matching_size_and_mtime() {
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000);
        let later = now + std::time::Duration::from_secs(1);
        let path = PathBuf::from("adapter.safetensors");

        let mut cache = ExternalLoraCache::default();
        cache.insert(path.clone(), Some(now), 100, Some("wan-video".to_owned()));

        assert_eq!(
            cache.get(&path, Some(now), 100),
            Some(Some("wan-video".to_owned())),
            "identical size + mtime is a hit"
        );
        assert_eq!(
            cache.get(&path, Some(later), 100),
            None,
            "newer mtime misses"
        );
        assert_eq!(
            cache.get(&path, Some(now), 101),
            None,
            "changed size misses"
        );
        assert_eq!(
            cache.get(&path, None, 100),
            None,
            "a file with no mtime can never be proven unchanged"
        );
        assert_eq!(cache.get(Path::new("other"), Some(now), 100), None);
    }

    /// A `None` family (parsed fine, unidentifiable) is memoized too — otherwise every
    /// catalog build would re-parse the adapters that can never be identified.
    #[test]
    fn cache_memoizes_an_undetected_family() {
        let now = SystemTime::UNIX_EPOCH;
        let path = PathBuf::from("mystery.safetensors");
        let mut cache = ExternalLoraCache::default();
        cache.insert(path.clone(), Some(now), 7, None);
        assert_eq!(cache.get(&path, Some(now), 7), Some(None));
    }

    /// The cache fills on scan and drops adapters that have disappeared, so a
    /// long-lived process watching a churning directory does not grow unbounded.
    #[test]
    fn scan_populates_then_prunes_the_cache() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        let adapter = root.join("loras").join("wan.safetensors");
        write_safetensors(&adapter, &wan_keys());

        let mut cache = ExternalLoraCache::default();
        let rows = scan_external_loras(std::slice::from_ref(&root), &mut cache);
        assert_eq!(rows.len(), 1);
        assert_eq!(cache.entries.len(), 1, "the adapter is memoized");

        // A second scan of the unchanged tree yields the same rows, now from cache.
        let again = scan_external_loras(std::slice::from_ref(&root), &mut cache);
        assert_eq!(again, rows);
        assert_eq!(cache.entries.len(), 1);

        std::fs::remove_file(&adapter).expect("remove adapter");
        let empty = scan_external_loras(&[root], &mut cache);
        assert!(empty.is_empty());
        assert!(cache.entries.is_empty(), "vanished adapters are pruned");
    }

    /// Manual smoke against a real ComfyUI tree — the only test that exercises the
    /// actual key conventions users ship. Ignored by default (no such tree in CI):
    ///
    /// ```text
    /// SCENEWORKS_EXTERNAL_MODEL_ROOTS='C:\Users\Michael\ComfyUI-Shared\models' \
    ///   cargo test -p sceneworks-rust-api --lib external_loras::tests::real_comfyui_tree -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a real ComfyUI models tree via SCENEWORKS_EXTERNAL_MODEL_ROOTS"]
    fn real_comfyui_tree_smoke() {
        let roots = sceneworks_core::external_roots::parse_external_model_roots(
            std::env::var("SCENEWORKS_EXTERNAL_MODEL_ROOTS")
                .ok()
                .as_deref(),
        );
        assert!(!roots.is_empty(), "set SCENEWORKS_EXTERNAL_MODEL_ROOTS");

        let rows = scan(&roots);
        println!("\n{} adapters surfaced:", rows.len());
        let mut without_family = 0;
        for row in &rows {
            let family = row.get("family").and_then(Value::as_str);
            if family.is_none() {
                without_family += 1;
            }
            println!(
                "  {:<12} {:<10} {}",
                family.unwrap_or("(none)"),
                row["scope"].as_str().unwrap_or_default(),
                row["name"].as_str().unwrap_or_default(),
            );
        }
        println!(
            "\n{without_family} of {} lack a detected family\n",
            rows.len()
        );
        assert!(!rows.is_empty(), "the real tree should surface adapters");
    }

    /// A symlink escaping the declared root is dropped: the worker's confinement would
    /// reject that path at generation time, so surfacing it would only promise a load
    /// that cannot happen.
    #[cfg(unix)]
    #[test]
    fn symlinks_escaping_the_root_are_not_surfaced() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = comfy_root(temp.path());
        let loras = root.join("loras");
        std::fs::create_dir_all(&loras).expect("mkdir");
        let outside = temp.path().join("outside");
        write_safetensors(&outside.join("escape.safetensors"), &wan_keys());
        std::os::unix::fs::symlink(
            outside.join("escape.safetensors"),
            loras.join("escape.safetensors"),
        )
        .expect("symlink");

        assert!(scan(&[root]).is_empty());
    }
}
