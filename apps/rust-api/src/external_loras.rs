//! Surface the LoRA adapters already sitting in an operator's ComfyUI `models/`
//! tree, read in place (epic 10451 / sc-10452).
//!
//! The scan itself lives in `sceneworks_core::external_loras` so the API catalog
//! and the worker's portable source inventory assign the same `external_…`
//! identities to the same files (SW-7030); this module only re-exports it under
//! the API's historical path.

#[cfg(test)]
use std::collections::HashSet;
#[cfg(test)]
use std::path::{Path, PathBuf};

#[cfg(test)]
use serde_json::{json, Value};

#[cfg(test)]
pub(crate) use sceneworks_core::external_loras::{collect_adapters, EXTERNAL_ID_PREFIX};
pub(crate) use sceneworks_core::external_loras::{
    scan_external_loras, ExternalLoraCache, EXTERNAL_SCOPE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;
    use std::time::SystemTime;

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
        bytes.extend(std::iter::repeat_n(0_u8, keys.len() * 4));
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
        assert_eq!(cache.len(), 1, "the adapter is memoized");

        // A second scan of the unchanged tree yields the same rows, now from cache.
        let again = scan_external_loras(std::slice::from_ref(&root), &mut cache);
        assert_eq!(again, rows);
        assert_eq!(cache.len(), 1);

        std::fs::remove_file(&adapter).expect("remove adapter");
        let empty = scan_external_loras(&[root], &mut cache);
        assert!(empty.is_empty());
        assert!(cache.is_empty(), "vanished adapters are pruned");
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
