use super::*;
// Reuse the ONE crate-wide HF-cache guard (sc-13834): tests here seed/resolve the cache via the
// env-first `huggingface_repo_cache_path`, so without this they would resolve into a developer's
// real HF cache when HF_HOME is set. Serialize on the same `HF_ENV_LOCK`; never add a second lock.
use crate::tests::support::isolate_hf_cache;

#[test]
fn discovery_repairs_legacy_snapshot_revisions_and_restores_cache_eligibility() {
    use sceneworks_core::model_artifacts::artifact_selection::{
        local_cache_eligibility_for_model, LocalCacheCoverage,
    };
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path();
    let repo = "owner/receipt-repair";
    let rev = "1111111111111111111111111111111111111111";
    let snapshot = huggingface_repo_cache_path(data, repo)
        .unwrap()
        .join("snapshots")
        .join(rev);
    std::fs::create_dir_all(snapshot.join("q4")).unwrap();
    std::fs::write(snapshot.join("q4/model.safetensors"), b"weights").unwrap();
    let managed = data.join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    let marker = managed.join(".sceneworks-download-complete.json");
    // A newer catalog pin must never be substituted for the snapshot actually installed.
    let model = json!({"id":"repair", "downloads":[{"provider":"huggingface", "repo":repo,
        "revision":"2222222222222222222222222222222222222222", "variant":"q4", "default":true, "files":["q4/*"]}]});
    for revision in [None, Some(Value::Null), Some(json!("")), Some(json!("  "))] {
        let mut entry = json!({"repo":repo, "modelId":"repair", "variant":"q4", "backfilled":true,
            "resolvedFiles":["q4/model.safetensors"], "customField":"preserved"});
        if let Some(revision) = revision {
            entry["snapshotRevision"] = revision;
        }
        let sibling = json!({"repo":repo,"modelId":"other","variant":"q8", "snapshotRevision":rev,
            "resolvedFiles":["q8/model.safetensors"]});
        let mut top = entry.clone();
        top["receipts"] = json!([entry, sibling]);
        std::fs::write(&marker, serde_json::to_vec(&top).unwrap()).unwrap();
        assert_eq!(
            local_cache_eligibility_for_model(&model, "macos", Some("q4"), data).coverage,
            LocalCacheCoverage::None
        );
        install_state_for(model_download_context(&model).unwrap(), &model, data);
        let bytes = std::fs::read(&marker).unwrap();
        let after: Value = serde_json::from_slice(&bytes).unwrap();
        top["snapshotRevision"] = json!(rev);
        top["receipts"][0]["snapshotRevision"] = json!(rev);
        assert_eq!(
            after, top,
            "only missing revisions change, including the mirrored top-level receipt"
        );
        assert_eq!(
            local_cache_eligibility_for_model(&model, "macos", Some("q4"), data).coverage,
            LocalCacheCoverage::Full
        );
        install_state_for(model_download_context(&model).unwrap(), &model, data);
        assert_eq!(
            std::fs::read(&marker).unwrap(),
            bytes,
            "repeat discovery is idempotent"
        );
    }
}

#[test]
fn snapshot_revision_repair_does_not_pin_a_torn_shard_set() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let repo = "owner/torn-repair";
    let root = huggingface_repo_cache_path(temp.path(), repo).unwrap();
    let snapshot = root.join("snapshots/1111111111111111111111111111111111111111");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"missing.safetensors"}}"#,
    )
    .unwrap();
    let managed = temp.path().join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    let marker = managed.join(".sceneworks-download-complete.json");
    let receipt =
        json!({"repo":repo,"modelId":"repair", "resolvedFiles":["model.safetensors.index.json"]});
    let before = serde_json::to_vec(&receipt).unwrap();
    std::fs::write(&marker, &before).unwrap();
    repair_missing_snapshot_revisions(&managed, &json!({"id":"repair"}), temp.path());
    assert_eq!(std::fs::read(&marker).unwrap(), before);
}

fn builtin_models_entry(model_id: &str) -> Value {
    let manifest = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
        .expect("valid embedded model catalog");
    manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        .cloned()
        .unwrap_or_else(|| panic!("builtin entry {model_id} present"))
}

/// The optional native film planner must describe a loadable official checkpoint, not merely a
/// repository directory. Its first pin predated the weights and held only metadata; that exact
/// snapshot plus its completed download receipt must remain unavailable. The current pin is a
/// 15-shard indexed checkpoint, and one wildcard-matching shard is still a torn install.
#[test]
fn qwen36_film_planner_requires_the_current_complete_indexed_snapshot() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path();
    let model = builtin_models_entry("film_planner_qwen3_6_27b");
    let download = model_download(&model).expect("Qwen planner download");
    let repo = download["repo"].as_str().unwrap();
    let current_revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9";
    let obsolete_revision = "70215ea87f5a6ae62bb56822396c1b42f1a18e0b";
    assert_eq!(download["revision"], current_revision);
    assert_eq!(
        download["files"],
        json!([
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-*.safetensors"
        ])
    );
    assert_eq!(
        download["breaking"], true,
        "only the unusable README-only receipt needs stale-install refusal"
    );

    let root = huggingface_repo_cache_path(data, repo).unwrap();
    let obsolete = root.join("snapshots").join(obsolete_revision);
    std::fs::create_dir_all(&obsolete).unwrap();
    std::fs::write(obsolete.join("README.md"), b"model card only").unwrap();
    let managed = data.join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&json!({
            "schemaVersion": 2,
            "repo": repo,
            "modelId": "film_planner_qwen3_6_27b",
            "variant": "default",
            "manifestFiles": [],
            "resolvedFiles": ["README.md"],
            "snapshotRevision": obsolete_revision
        }))
        .unwrap(),
    )
    .unwrap();
    let readme_only = install_state_for(model_download_context(&model).unwrap(), &model, data);
    assert!(
        !readme_only.installed,
        "an old completed receipt cannot make a metadata-only snapshot usable"
    );
    assert!(
        readme_only
            .missing_required_files
            .iter()
            .any(|file| file == "config.json"),
        "missing loader artifacts stay visible: {:?}",
        readme_only.missing_required_files
    );

    let current = root.join("snapshots").join(current_revision);
    std::fs::create_dir_all(&current).unwrap();
    for file in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        std::fs::write(current.join(file), b"{}").unwrap();
    }
    let shard_name = |ordinal: usize| format!("model-{ordinal:05}-of-00015.safetensors");
    std::fs::write(current.join("model.safetensors.index.json"), b"{not json").unwrap();
    std::fs::write(current.join(shard_name(1)), b"first shard").unwrap();
    let invalid_index = install_state_for(model_download_context(&model).unwrap(), &model, data);
    assert!(
        !invalid_index.installed
            && invalid_index.cache_incomplete
            && invalid_index
                .missing_required_files
                .iter()
                .any(|file| file == "model.safetensors.index.json"),
        "an explicitly required invalid index cannot make a wildcard shard set usable: {:?}",
        invalid_index.missing_required_files
    );

    let weight_map = (1..=15)
        .map(|ordinal| {
            (
                format!("tensor.{ordinal}"),
                Value::String(shard_name(ordinal)),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        current.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();

    let partial = install_state_for(model_download_context(&model).unwrap(), &model, data);
    assert!(
        !partial.installed && partial.cache_incomplete,
        "one shard matches the wildcard but cannot satisfy a 15-shard index; missing={:?}",
        partial.missing_required_files
    );
    assert!(
        partial
            .missing_required_files
            .iter()
            .any(|file| file == &shard_name(15)),
        "the index names the missing shards: {:?}",
        partial.missing_required_files
    );

    for ordinal in 2..=15 {
        std::fs::write(current.join(shard_name(ordinal)), b"shard").unwrap();
    }
    let complete = install_state_for(model_download_context(&model).unwrap(), &model, data);
    assert!(
        complete.installed,
        "complete indexed snapshot; missing={:?}",
        complete.missing_required_files
    );
    assert!(complete.missing_required_files.is_empty());
}

/// Manifest `files` entries are match patterns, not literal names — Eros's shared Gemma
/// encoder ships as `"gemma/*"`. Seeding must write a CONCRETE file the pattern matches:
/// `snapshot_contains_pattern` globs the snapshot, so any real name under `gemma/` reads
/// exactly like the literal `*` file this helper used to write. That literal only ever worked
/// because `*` is a legal filename on macOS/Linux; on Windows `fs::write` fails with
/// `ERROR_INVALID_NAME` (os error 123), which reddened the seeded-cache Eros tests on the dev
/// box while the Linux `cargo test` lane stayed green.
fn seeded_file_path(pattern: &str) -> String {
    pattern
        .split('/')
        .map(|segment| match segment {
            "*" => "seeded.safetensors".to_owned(),
            segment if pattern_contains_glob(segment) => {
                segment.replace('*', "seeded").replace('?', "0")
            }
            segment => segment.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn seed_manifest_download(data_dir: &FsPath, download: &Value) {
    let repo = download.get("repo").and_then(Value::as_str).unwrap();
    let revision = download.get("revision").and_then(Value::as_str).unwrap();
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots")
        .join(revision);
    for file in string_array_field(download, "files") {
        let path = snapshot.join(seeded_file_path(&file));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"legacy Eros weights").unwrap();
    }
}

fn seed_manifest_downloads(data_dir: &FsPath, model: &Value, include_co_requisites: bool) {
    for download in model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| include_co_requisites || !is_co_requisite_download(download))
    {
        seed_manifest_download(data_dir, download);
    }
}

/// A platformCleanupOnly tombstone has its `variants` projection stripped by
/// `apply_model_catalog_entry`, yet the snapshot pipeline still runs size enrichment on it.
/// The refresh must no-op for tombstones while staying a hard error for real entries —
/// this exact mismatch 500'd the whole catalog for machines holding legacy Eros weights.
/// (Platform-neutral twin of the seeded-cache Eros tests below, which reach the same tombstone
/// through a real on-disk install — see `seeded_file_path` for why that seeding is glob-aware.)
#[test]
fn size_refresh_tolerates_a_cleanup_tombstone_without_variants() {
    let mut tombstone = json!({
        "id": "ltx_2_3_eros",
        "platformCleanupOnly": true,
    });
    refresh_variant_download_sizes(&mut tombstone).expect("tombstone size refresh must be a no-op");
    assert!(
        tombstone.get("variants").is_none(),
        "tombstone refresh must not invent a variants projection"
    );

    let mut real_entry = json!({ "id": "ltx_2_3" });
    assert!(
        refresh_variant_download_sizes(&mut real_entry).is_err(),
        "a non-tombstone entry without a variants array is still a contract violation"
    );
}

/// SC-18902: Eros is a real MLX product route, but its exact-head Candle/CUDA capture was
/// unusable. Pin the complete catalog projection so a future edit cannot restore the 46.8 GB
/// Windows/Linux offer merely by changing one of the three required download rows.
#[test]
fn ltx_eros_catalog_is_macos_only_and_not_downloadable_off_macos() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let base = builtin_models_entry("ltx_2_3");
    assert_ne!(
        base.get("macOnly").and_then(Value::as_bool),
        Some(true),
        "base LTX-2.3 must remain cross-platform"
    );
    for os in ["windows", "linux"] {
        let mut candle_base = base.clone();
        retain_downloads_for_os(&mut candle_base, os);
        assert!(
            model_download_context(&candle_base)
                .expect("valid base LTX catalog entry")
                .is_some(),
            "{os}: base LTX-2.3 remains downloadable"
        );
    }

    let entry = builtin_models_entry("ltx_2_3_eros");
    assert_eq!(entry.get("macOnly").and_then(Value::as_bool), Some(true));

    let mut false_override = entry.clone();
    let object = false_override.as_object_mut().unwrap();
    object.insert("macOnly".to_owned(), Value::Bool(false));
    object.insert("type".to_owned(), Value::String("image".to_owned()));
    object.insert("downloadable".to_owned(), Value::Bool(true));
    let mut absent_override = entry.clone();
    let object = absent_override.as_object_mut().unwrap();
    object.remove("macOnly");
    object.insert("downloadable".to_owned(), Value::Bool(true));
    object.insert("usable".to_owned(), Value::Bool(true));
    for override_attempt in [false_override, absent_override] {
        let mut overridden_projection = vec![override_attempt];
        retain_models_for_os(&mut overridden_projection, "windows", temp.path()).unwrap();
        assert!(
            overridden_projection.is_empty(),
            "user platform/download fields cannot restore the withdrawn Eros product"
        );
    }

    let mut macos = entry.clone();
    retain_downloads_for_os(&mut macos, "macos");
    let macos_downloads = macos
        .get("downloads")
        .and_then(Value::as_array)
        .expect("macOS Eros downloads");
    assert_eq!(
        macos_downloads
            .iter()
            .filter_map(|download| download.get("repo").and_then(Value::as_str))
            .collect::<Vec<_>>(),
        vec![
            "SceneWorks/ltx-2.3-mlx",
            "TenStrip/LTX2.3-10Eros",
            "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments",
        ],
        "macOS retains the shared Gemma bundle, Eros checkpoint, and cond_safe adapter"
    );
    let macos_context = model_download_context(&macos)
        .expect("valid Eros download context")
        .expect("macOS Eros remains downloadable");
    let macos_state = install_state_for(Some(macos_context), &macos, temp.path());
    assert!(macos_state.downloadable);
    let mut macos_projection = vec![base.clone(), entry.clone()];
    retain_models_for_os(&mut macos_projection, "macos", temp.path()).unwrap();
    assert_eq!(
        macos_projection
            .iter()
            .filter_map(|model| model.get("id").and_then(Value::as_str))
            .collect::<Vec<_>>(),
        vec!["ltx_2_3", "ltx_2_3_eros"],
        "macOS catalog and Model Manager retain both LTX products"
    );

    for os in ["windows", "linux"] {
        let mut candle = entry.clone();
        retain_downloads_for_os(&mut candle, os);
        assert_eq!(
            candle
                .get("downloads")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|download| download.get("repo").and_then(Value::as_str))
                .collect::<Vec<_>>(),
            vec!["SceneWorks/ltx-2.3-mlx"],
            "{os}: platform filtering retains only the shared Gemma bundle, never the Eros checkpoint or adapter"
        );
        assert!(
            model_download_context(&candle)
                .expect("valid filtered catalog entry")
                .is_none(),
            "{os}: Eros must have no canonical download"
        );
        let state = install_state_for(None, &candle, temp.path());
        assert!(!state.downloadable, "{os}: Eros must not be downloadable");
        assert!(
            !state.installed,
            "{os}: filtered remote files must not resurrect a picker row"
        );
        let mut projection = vec![base.clone(), entry.clone()];
        retain_models_for_os(&mut projection, os, temp.path()).unwrap();
        assert_eq!(
            projection
                .iter()
                .filter_map(|model| model.get("id").and_then(Value::as_str))
                .collect::<Vec<_>>(),
            vec!["ltx_2_3"],
            "{os}: Model Manager/catalog must omit Eros and preserve base LTX-2.3"
        );
    }
}

/// A previously shipped Windows/Linux install remains visible only as a cleanup tombstone. The
/// tombstone cannot generate, download, repair, or update, but DELETE can reclaim both the dense
/// checkpoint and its Eros-exclusive cond_safe adapter; once removed, the row disappears.
#[tokio::test(flavor = "current_thread")]
async fn ltx_eros_legacy_install_is_cleanup_only_and_reclaimable_off_macos() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let entry = builtin_models_entry("ltx_2_3_eros");
    seed_manifest_downloads(data_dir, &entry, true);

    let mut projection = vec![entry.clone()];
    retain_models_for_os(&mut projection, "windows", data_dir).unwrap();
    assert_eq!(
        projection.len(),
        1,
        "an installed legacy Eros row is retained"
    );
    assert_eq!(
        projection[0]
            .get("platformCleanupOnly")
            .and_then(Value::as_bool),
        Some(true)
    );

    let context = model_download_context(&projection[0])
        .unwrap()
        .expect("cleanup tombstone retains its original artifact context");
    // No resolved-cache artifacts (sc-19708's availability seam): a platform tombstone is
    // judged from its own on-disk install, not from an external library's local copies.
    let projected = apply_model_catalog_entry(
        projection.pop().unwrap(),
        Some(context),
        data_dir,
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        &[],
    )
    .unwrap();
    assert_eq!(
        projected.get("installState").and_then(Value::as_str),
        Some("installed")
    );
    assert_eq!(
        projected.get("usable").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        projected.get("downloadable").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        projected.get("repairAvailable").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        projected.get("updateAvailable").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        projected.get("removable").and_then(Value::as_bool),
        Some(true)
    );
    let refusal = ensure_model_downloadable(&projected)
        .expect_err("a cleanup tombstone must reject direct download API calls");
    assert!(refusal.detail.contains("retained only"));
    let refusal = ensure_model_not_cleanup_only(&projected)
        .expect_err("a cleanup tombstone must reject direct conversion API calls");
    assert!(refusal.detail.contains("conversion"));
    assert_eq!(
        projected.get("hasVariantMatrix").and_then(Value::as_bool),
        Some(false)
    );
    for field in [
        "variants",
        "mlxConversionState",
        "mlxInstallState",
        "mlxTiers",
        "mlxTierStates",
    ] {
        assert!(
            projected.get(field).is_none(),
            "cleanup projection leaked {field}"
        );
    }

    // The snapshot pipeline runs size enrichment on every retained row AFTER the tombstone
    // strip. It must tolerate the stripped projection — a hard error here 500s the entire
    // catalog for any machine still holding legacy Eros weights in its HF cache.
    let mut enriched = projected.clone();
    let context = model_download_context(&enriched).unwrap();
    apply_model_catalog_size_fields(&mut enriched, context.as_ref(), None)
        .expect("size enrichment tolerates a cleanup tombstone");
    assert!(
        enriched.get("variants").is_none(),
        "size enrichment must not resurrect the variants projection on a tombstone"
    );

    let primary_repo = "TenStrip/LTX2.3-10Eros";
    let adapter_repo = "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments";
    let primary_cache = huggingface_repo_cache_path(data_dir, primary_repo).unwrap();
    let adapter_cache = huggingface_repo_cache_path(data_dir, adapter_repo).unwrap();
    let cleanup_paths = model_artifact_paths(&projected, data_dir);
    assert!(cleanup_paths.contains(&primary_cache));
    assert!(cleanup_paths.contains(&adapter_cache));

    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
    let removal = crate::remove_owned_artifacts(cleanup_paths, &allowed_roots, true)
        .await
        .expect("cleanup-only Eros delete succeeds");
    assert!(!removal.removed_paths.is_empty());
    assert!(!primary_cache.exists(), "primary Eros cache is reclaimed");
    assert!(
        !adapter_cache.exists(),
        "exclusive cond_safe cache is reclaimed"
    );

    let mut after_delete = vec![entry.clone()];
    retain_models_for_os(&mut after_delete, "windows", data_dir).unwrap();
    assert!(
        after_delete.is_empty(),
        "the cleanup tombstone disappears after deletion"
    );

    // A partial legacy install also needs the delete affordance, but never a repair/download.
    seed_manifest_downloads(data_dir, &entry, false);
    let mut partial = vec![entry];
    retain_models_for_os(&mut partial, "linux", data_dir).unwrap();
    assert_eq!(
        partial.len(),
        1,
        "a partial legacy install remains reclaimable"
    );
    let context = model_download_context(&partial[0]).unwrap();
    let partial = apply_model_catalog_entry(
        partial.pop().unwrap(),
        context,
        data_dir,
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        &[],
    )
    .unwrap();
    assert_eq!(
        partial.get("cacheState").and_then(Value::as_str),
        Some("incomplete")
    );
    assert_eq!(
        partial.get("repairAvailable").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        partial.get("downloadable").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        partial.get("removable").and_then(Value::as_bool),
        Some(true)
    );
    let partial_primary_cache = huggingface_repo_cache_path(data_dir, primary_repo).unwrap();
    assert!(partial_primary_cache.exists());
    let removal = crate::remove_owned_artifacts(
        model_artifact_paths(&partial, data_dir),
        &allowed_roots,
        true,
    )
    .await
    .expect("partial cleanup-only Eros delete succeeds");
    assert!(!removal.removed_paths.is_empty());
    assert!(
        !partial_primary_cache.exists(),
        "partial primary Eros cache is reclaimed"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn ltx_eros_corequisite_only_partial_is_cleanup_only_and_reclaimable_off_macos() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let entry = builtin_models_entry("ltx_2_3_eros");
    let adapter = entry
        .get("downloads")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .find(|download| download.get("cleanupWithModel").and_then(Value::as_bool) == Some(true))
        .unwrap();
    seed_manifest_download(data_dir, adapter);

    let adapter_repo = adapter.get("repo").and_then(Value::as_str).unwrap();
    let adapter_cache = huggingface_repo_cache_path(data_dir, adapter_repo).unwrap();
    let mut projection = vec![entry];
    retain_models_for_os(&mut projection, "windows", data_dir).unwrap();
    assert_eq!(
        projection.len(),
        1,
        "a co-requisite-only partial is retained"
    );
    let context = model_download_context(&projection[0]).unwrap();
    let projected = apply_model_catalog_entry(
        projection.pop().unwrap(),
        context,
        data_dir,
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        &[],
    )
    .unwrap();
    assert_eq!(
        projected.get("cacheState").and_then(Value::as_str),
        Some("incomplete")
    );
    assert_eq!(
        projected.get("removable").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        projected.get("downloadable").and_then(Value::as_bool),
        Some(false)
    );

    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
    let removal = crate::remove_owned_artifacts(
        model_artifact_paths(&projected, data_dir),
        &allowed_roots,
        true,
    )
    .await
    .expect("co-requisite-only partial delete succeeds");
    assert!(!removal.removed_paths.is_empty());
    assert!(!adapter_cache.exists(), "orphan adapter cache is reclaimed");
}

#[tokio::test(flavor = "current_thread")]
async fn ltx_eros_mixed_trash_failure_retains_a_retryable_cleanup_tombstone() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let entry = builtin_models_entry("ltx_2_3_eros");
    seed_manifest_downloads(data_dir, &entry, true);

    let primary_cache = huggingface_repo_cache_path(data_dir, "TenStrip/LTX2.3-10Eros").unwrap();
    let adapter_cache =
        huggingface_repo_cache_path(data_dir, "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments")
            .unwrap();
    let mut projection = vec![entry.clone()];
    retain_models_for_os(&mut projection, "linux", data_dir).unwrap();
    let cleanup_model = projection.pop().unwrap();
    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
    let _trash = crate::test_trash_outcomes([
        (primary_cache.clone(), true),
        (adapter_cache.clone(), false),
    ]);
    let removal = crate::remove_owned_artifacts(
        model_artifact_paths(&cleanup_model, data_dir),
        &allowed_roots,
        false,
    )
    .await
    .expect("mixed OS-trash outcome is recoverable");
    assert!(removal
        .removed_paths
        .iter()
        .any(|path| path == &primary_cache.display().to_string()));
    assert!(removal
        .trash_failed_paths
        .iter()
        .any(|path| path == &adapter_cache.display().to_string()));
    assert!(!primary_cache.exists());
    assert!(adapter_cache.exists());

    let mut retry_projection = vec![entry.clone()];
    retain_models_for_os(&mut retry_projection, "linux", data_dir).unwrap();
    assert_eq!(
        retry_projection.len(),
        1,
        "adapter residue must keep the permanent-delete retry resolvable"
    );
    let context = model_download_context(&retry_projection[0]).unwrap();
    let retry_model = apply_model_catalog_entry(
        retry_projection.pop().unwrap(),
        context,
        data_dir,
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
        &[],
    )
    .unwrap();
    assert_eq!(
        retry_model.get("cacheState").and_then(Value::as_str),
        Some("incomplete")
    );
    let retry = crate::remove_owned_artifacts(
        model_artifact_paths(&retry_model, data_dir),
        &allowed_roots,
        true,
    )
    .await
    .expect("permanent retry reclaims the retained adapter");
    assert!(!retry.removed_paths.is_empty());
    assert!(!adapter_cache.exists());

    let mut after_retry = vec![entry];
    retain_models_for_os(&mut after_retry, "linux", data_dir).unwrap();
    assert!(
        after_retry.is_empty(),
        "tombstone disappears after retry succeeds"
    );
}

#[test]
fn multi_repo_marker_filters_nested_receipts_by_requested_repo() {
    let temp = tempfile::tempdir().unwrap();
    let managed = temp.path().join("models/owner--primary");
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(managed.join(".sceneworks-download-complete.json"), serde_json::to_vec(&json!({
        "repo": "owner/corequisite",
        "receipts": [
            {"repo":"owner/primary", "resolvedFiles":["model.safetensors"], "snapshotRevision":"primary-rev"},
            {"repo":"owner/corequisite", "resolvedFiles":["encoder.safetensors"], "snapshotRevision":"dependency-rev"}
        ]
    })).unwrap()).unwrap();

    let primary = receipt_file_sets(&managed, "owner/primary", None);
    assert_eq!(
        primary,
        vec![ReceiptFileSet {
            files: vec!["model.safetensors".to_owned()],
            revision: Some("primary-rev".to_owned())
        }]
    );
    let dependency = receipt_file_sets(&managed, "owner/corequisite", None);
    assert_eq!(
        dependency,
        vec![ReceiptFileSet {
            files: vec!["encoder.safetensors".to_owned()],
            revision: Some("dependency-rev".to_owned())
        }]
    );
}

/// sc-14432: the RECEIPT lane was diffusers-only, so a flat no-`model_index` tier passed
/// `snapshot_tier_is_loadable` unconditionally. A receipt recording exactly what the `<tier>/*` glob
/// matched (the very shape the story cited as evidence of a bad re-host) therefore kept the model
/// reading installed through the usable-stale path even once the cache-health lane had correctly
/// demoted the tier. Both receipt halves — this one and `backfill_current_receipt` — now consult the
/// shared family predicate.
#[test]
fn a_torn_flat_tier_is_not_counted_as_a_usable_stale_install() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "SceneWorks/sensenova-u1-8b-mlx";
    let model = json!({
        "id": "sensenova_u1_8b",
        "family": "sensenova-u1",
        "downloads": [{ "provider": "huggingface", "repo": repo, "variant": "q4", "files": ["q4/*"] }]
    });
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    let tier = snapshot.join("q4");
    std::fs::create_dir_all(&tier).unwrap();
    // The backbone landed; the tokenizer never did, and there is no sibling tier to borrow one from.
    std::fs::write(tier.join("model.safetensors"), b"weights").unwrap();
    std::fs::write(tier.join("config.json"), b"{}").unwrap();

    let marker_dir = data_dir.join("models").join(safe_download_dir(repo));
    let marker = marker_dir.join(".sceneworks-download-complete.json");

    // A pre-existing receipt (written before this tightening, or by a download that recorded exactly
    // what the glob matched) whose every recorded file DOES exist — the usable-stale route. The tier
    // still cannot load, so it must not resurrect the install.
    std::fs::create_dir_all(&marker_dir).unwrap();
    std::fs::write(
        &marker,
        serde_json::to_vec(&json!({
            "schemaVersion": 2, "repo": repo, "modelId": "sensenova_u1_8b", "variant": "q4",
            "manifestFiles": ["q4/*"],
            "resolvedFiles": ["q4/model.safetensors", "q4/config.json"],
        }))
        .unwrap(),
    )
    .unwrap();

    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        !state.installed,
        "a receipt whose files all exist must not keep a torn flat tier installed"
    );

    // Mutation check: the tokenizer arriving makes the tier genuinely loadable, and the SAME receipt
    // now legitimately counts — proving the assertion above discriminates on loadability, not on the
    // receipt merely being present.
    std::fs::write(tier.join("tokenizer.json"), b"{}").unwrap();
    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(state.installed);
}

/// sc-20526 fixture: a diffusers quant tier whose text encoder is SHARDED. `present_shards`
/// selects which of the three declared shards actually landed, so the same builder produces both
/// the interrupted download and the genuinely complete tier.
fn write_sharded_diffusers_tier(snapshot: &FsPath, tier: &str, present_shards: &[u32]) {
    let tier_dir = snapshot.join(tier);
    std::fs::create_dir_all(&tier_dir).unwrap();
    std::fs::write(
        tier_dir.join("model_index.json"),
        serde_json::to_vec(&json!({
            "_class_name": "LensPipeline",
            "text_encoder": ["transformers", "Qwen3Model"],
            "transformer": ["diffusers", "LensTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        }))
        .unwrap(),
    )
    .unwrap();
    for (component, weights) in [
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
    ] {
        let dir = tier_dir.join(component);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join(weights), b"weights").unwrap();
    }
    let scheduler = tier_dir.join("scheduler");
    std::fs::create_dir_all(&scheduler).unwrap();
    std::fs::write(scheduler.join("scheduler_config.json"), b"{}").unwrap();

    let encoder = tier_dir.join("text_encoder");
    std::fs::create_dir_all(&encoder).unwrap();
    std::fs::write(encoder.join("config.json"), b"{}").unwrap();
    std::fs::write(
        encoder.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00003.safetensors",
            "lm_head.weight": "model-00003-of-00003.safetensors",
        }}))
        .unwrap(),
    )
    .unwrap();
    for shard in present_shards {
        std::fs::write(
            encoder.join(format!("model-0000{shard}-of-00003.safetensors")),
            b"shard",
        )
        .unwrap();
    }
}

fn sharded_tier_model(repo: &str, tier: &str) -> Value {
    json!({
        "id": "lens_turbo",
        "downloads": [{
            "provider": "huggingface", "repo": repo,
            "variant": tier, "files": [format!("{tier}/*")]
        }]
    })
}

/// sc-20526 (AC 1, AC 6): the lens_turbo bf16 shape. The tier's `<tier>/*` glob matches, its
/// `model_index.json` is present, and `text_encoder/` holds a real `.safetensors` — so every
/// pre-existing check passed — yet the index names three shards and only the LAST one landed.
/// `model.embed_tokens.weight` lives in shard 1, so the load died with "cannot find tensor".
/// Backfill must refuse to mint a receipt, and the tier must read not-installed but repairable.
#[test]
fn backfill_refuses_a_partially_downloaded_sharded_tier() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "SceneWorks/lens-turbo-mlx";
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    write_sharded_diffusers_tier(&snapshot, "bf16", &[3]);
    let model = sharded_tier_model(repo, "bf16");

    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        !state.installed,
        "a tier missing 2 of 3 text-encoder shards must not read installed"
    );
    assert!(
        state.cache_incomplete,
        "it must surface as repairable/re-downloadable, not merely absent"
    );
    assert!(
        state
            .missing_required_files
            .iter()
            .any(|file| file.contains("model-00001-of-00003.safetensors")),
        "the absent shard must be named: {:?}",
        state.missing_required_files
    );
    let marker = data_dir
        .join("models")
        .join(safe_download_dir(repo))
        .join(".sceneworks-download-complete.json");
    assert!(
        !marker.exists(),
        "backfill must not launder an interrupted download into a complete receipt"
    );

    // Mutation check: the two missing shards arriving flips every assertion above, proving the
    // check discriminates on shard completeness rather than rejecting sharded tiers wholesale.
    for shard in [1, 2] {
        std::fs::write(
            snapshot.join(format!(
                "bf16/text_encoder/model-0000{shard}-of-00003.safetensors"
            )),
            b"shard",
        )
        .unwrap();
    }
    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(state.installed, "a genuinely complete tier must pass");
    assert!(!state.cache_incomplete);
    let receipt: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    assert_eq!(receipt["backfilled"], true);
    assert!(
        receipt["resolvedFiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|file| file.as_str()
                == Some("bf16/text_encoder/model-00001-of-00003.safetensors")),
        "the honest receipt records the full shard set"
    );
}

/// sc-20526 (AC 3, AC 4, AC 6): an ALREADY-WRITTEN false-complete receipt — `backfilled: true`
/// with `resolvedFiles` listing the index plus only the shards that happened to land. This is
/// exactly what shipped to affected users. The next scan must invalidate it (no "usable stale"
/// install) AND repair the file, so nobody has to delete it by hand.
#[test]
fn existing_false_complete_receipt_is_invalidated_and_repaired() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "SceneWorks/lens-turbo-mlx";
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    write_sharded_diffusers_tier(&snapshot, "bf16", &[3]);
    let model = sharded_tier_model(repo, "bf16");

    let managed = data_dir.join("models").join(safe_download_dir(repo));
    let marker = managed.join(".sceneworks-download-complete.json");
    std::fs::create_dir_all(&managed).unwrap();
    let torn_entry = json!({
        "schemaVersion": 2, "repo": repo, "modelId": "lens_turbo", "variant": "bf16",
        "manifestFiles": ["bf16/*"],
        "resolvedFiles": [
            "bf16/model_index.json",
            "bf16/text_encoder/config.json",
            "bf16/text_encoder/model-00003-of-00003.safetensors",
            "bf16/text_encoder/model.safetensors.index.json",
        ],
        "backfilled": true,
    });
    std::fs::write(&marker, serde_json::to_vec(&torn_entry).unwrap()).unwrap();

    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        !state.installed,
        "a receipt whose recorded files all exist must still not resurrect a torn shard set"
    );
    assert!(
        !marker.exists(),
        "the false-complete receipt must be repaired away so the install self-heals"
    );

    // A receipt for a tier that IS complete survives the same sweep untouched.
    write_sharded_diffusers_tier(&snapshot, "q4", &[1, 2, 3]);
    let sound = json!({
        "schemaVersion": 2, "repo": repo, "modelId": "lens_turbo", "variant": "q4",
        "manifestFiles": ["q4/*"],
        "resolvedFiles": [
            "q4/model_index.json",
            "q4/text_encoder/model-00001-of-00003.safetensors",
            "q4/text_encoder/model-00002-of-00003.safetensors",
            "q4/text_encoder/model-00003-of-00003.safetensors",
            "q4/text_encoder/model.safetensors.index.json",
        ],
        "backfilled": true,
    });
    let seeded_files = sound["resolvedFiles"].clone();
    std::fs::write(&marker, serde_json::to_vec(&sound).unwrap()).unwrap();
    let q4 = sharded_tier_model(repo, "q4");
    assert!(install_state_for(model_download_context(&q4).unwrap(), &q4, data_dir).installed);
    // IDENTITY, not mere existence. `marker.exists()` alone proves nothing here: the tier is
    // complete, so a receipt deleted by an over-eager repair would be re-backfilled inside the
    // same call and the file would be back before the assertion ran. A re-backfill records the
    // full WALKED set (model_index, transformer, vae, scheduler, …), so the seeded five entries
    // surviving verbatim is what proves the sound receipt was never touched.
    let after: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    assert_eq!(
        after["resolvedFiles"], seeded_files,
        "repair must be positive-evidence only — a sound receipt is never deleted (nor \
         re-minted from a walk, which would list the transformer/vae/scheduler files too)"
    );
    assert_eq!(after["manifestFiles"], json!(["q4/*"]));
}

/// sc-20526 (AC 2): a WHOLE-REPO download (empty `files`) declares no per-file claim, so "some
/// payload landed" used to be the only signal. The repo's own shard index is the ground truth.
#[test]
fn whole_repo_install_validates_its_shard_index() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/whole-repo-sharded";
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }}))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(snapshot.join("model-00002-of-00002.safetensors"), b"shard").unwrap();
    let model = json!({
        "id": "whole-repo-model",
        "downloads": [{ "provider": "huggingface", "repo": repo }]
    });

    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(!state.installed, "half a shard set is not an install");
    assert!(state
        .missing_required_files
        .iter()
        .any(|file| file.contains("model-00001-of-00002.safetensors")));

    std::fs::write(snapshot.join("model-00001-of-00002.safetensors"), b"shard").unwrap();
    assert!(
        install_state_for(model_download_context(&model).unwrap(), &model, data_dir).installed,
        "a complete whole-repo shard set must pass"
    );
}

/// sc-20526 (AC 1) — the shard guard INSIDE `backfill_current_receipt`, standing on its own.
///
/// `backfill_refuses_a_partially_downloaded_sharded_tier` cannot witness it: that tier is
/// diffusers, so `diffusers_snapshot_health` already drives `cache_installed` false and
/// `install_state_for` never enters the backfill at all — delete the guard and that test still
/// passes. A flat explicit filter now also refuses the torn set at catalog-health time, so this
/// test calls the backfill seam directly to keep its independent guard covered, then proves the
/// public install state reports the same missing shard.
#[test]
fn backfill_refuses_a_flat_tier_whose_shard_index_is_torn() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/flat-sharded";
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        }}))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(snapshot.join("model-00002-of-00002.safetensors"), b"shard").unwrap();
    let model = json!({
        "id": "flat_sharded",
        "downloads": [{
            "provider": "huggingface", "repo": repo,
            "files": ["config.json", "model.safetensors.index.json", "*.safetensors"]
        }]
    });
    let managed = data_dir.join("models").join(safe_download_dir(repo));
    let marker = managed.join(".sceneworks-download-complete.json");

    let context = model_download_context(&model).unwrap().unwrap();
    backfill_current_receipt(&managed, &model, &context, data_dir);
    assert!(
        !marker.exists(),
        "backfill must independently refuse an index whose shard never landed"
    );

    let state = install_state_for(Some(context.clone()), &model, data_dir);
    assert!(
        !state.installed
            && state.cache_incomplete
            && state
                .missing_required_files
                .iter()
                .any(|file| file == "model-00001-of-00002.safetensors"),
        "filtered cache health must refuse the same torn set: {:?}",
        state.missing_required_files
    );

    // Mutation check: the absent shard arriving lets the SAME call mint the receipt, proving the
    // guard discriminates on shard completeness rather than never writing for this shape.
    std::fs::write(snapshot.join("model-00001-of-00002.safetensors"), b"shard").unwrap();
    assert!(install_state_for(Some(context), &model, data_dir).installed);
    let receipt: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    assert_eq!(receipt["backfilled"], true);
    assert!(
        receipt["resolvedFiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|file| file.as_str() == Some("model-00001-of-00002.safetensors")),
        "the honest receipt records the full shard set"
    );
}

/// sc-20526 (AC 3) — the shard filter in `receipt_files_present`, with repair kept out of the way.
///
/// `existing_false_complete_receipt_is_invalidated_and_repaired` deletes the receipt before the
/// read side ever runs, so it cannot witness that filter. Here a SECOND cached revision holds the
/// same index with a complete shard set, which is precisely the evidence
/// `receipt_entry_is_provably_torn` treats as "not provably torn" — the receipt SURVIVES the
/// sweep. It still must not count: the only revision holding every file the receipt RECORDED is
/// the torn one (rev-a), and the complete revision (rev-b) is missing `config.json`, so no single
/// snapshot satisfies both the recorded set and that set's own index.
#[test]
fn torn_receipt_surviving_repair_is_still_refused_by_the_read_side() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/two-revision-sharded";
    let root = huggingface_repo_cache_path(data_dir, repo).unwrap();
    let index_json = serde_json::to_vec(&json!({"weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }}))
    .unwrap();

    // rev-a: exactly what the receipt recorded — and its index names a shard that never landed.
    let torn = root.join("snapshots/rev-a");
    std::fs::create_dir_all(&torn).unwrap();
    std::fs::write(torn.join("config.json"), b"{}").unwrap();
    std::fs::write(torn.join("model.safetensors.index.json"), &index_json).unwrap();
    std::fs::write(torn.join("model-00002-of-00002.safetensors"), b"shard").unwrap();

    // rev-b: a COMPLETE shard set, but no `config.json` — so it can never satisfy this receipt.
    // Its existence is what makes the entry not *provably* torn, so repair leaves the file alone.
    let sound = root.join("snapshots/rev-b");
    std::fs::create_dir_all(&sound).unwrap();
    std::fs::write(sound.join("model.safetensors.index.json"), &index_json).unwrap();
    for shard in [1, 2] {
        std::fs::write(
            sound.join(format!("model-0000{shard}-of-00002.safetensors")),
            b"shard",
        )
        .unwrap();
    }

    // `tokenizer.json` exists in no revision, so the cache-health lane reads NOT installed and
    // the receipt is the only thing that could still call this model installed.
    let model = json!({
        "id": "two_revision_sharded",
        "downloads": [{
            "provider": "huggingface", "repo": repo,
            "files": ["config.json", "model.safetensors.index.json", "tokenizer.json"]
        }]
    });
    let managed = data_dir.join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    let marker = managed.join(".sceneworks-download-complete.json");
    let seeded = serde_json::to_vec(&json!({
        "schemaVersion": 2, "repo": repo, "modelId": "two_revision_sharded",
        "variant": "default",
        "manifestFiles": ["config.json", "model.safetensors.index.json", "tokenizer.json"],
        "resolvedFiles": [
            "config.json",
            "model.safetensors.index.json",
            "model-00002-of-00002.safetensors",
        ],
        "backfilled": true,
    }))
    .unwrap();
    std::fs::write(&marker, &seeded).unwrap();

    let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert_eq!(
        std::fs::read(&marker).ok().as_deref(),
        Some(seeded.as_slice()),
        "fixture precondition: a complete sibling revision makes this entry NOT provably torn, \
         so repair must leave it byte-identical — the read side has to do the refusing"
    );
    assert!(
        !state.installed,
        "the revision holding every recorded file is the torn one, so a surviving receipt must \
         not be honoured as a usable-stale install"
    );

    // Mutation check: the absent shard landing in rev-a makes the SAME surviving receipt
    // legitimately count, proving the refusal keys on shard completeness, not on the receipt.
    std::fs::write(torn.join("model-00001-of-00002.safetensors"), b"shard").unwrap();
    assert!(
        install_state_for(model_download_context(&model).unwrap(), &model, data_dir).installed,
        "a receipt whose recorded index IS complete in the revision holding its files counts"
    );
}

/// sc-20526 (AC 4) — the hard constraint on repair: PROVABLY torn, never merely unverifiable.
///
/// A backfilled, index-bearing receipt whose repo cache is not on this box at all (weights moved
/// into an external library, or a data dir restored without its HF cache) carries no evidence of
/// tearing. Deleting it would destroy the relocation proof an external-library install depends
/// on, and read-side validation already refuses to count it — so it must survive untouched.
/// Flipping `saw_torn`'s initial value to `true` turns "no evidence" into "delete it".
#[test]
fn repair_leaves_a_backfilled_receipt_whose_repo_cache_is_absent() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/relocated-to-external-library";
    let managed = data_dir.join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    let marker = managed.join(".sceneworks-download-complete.json");
    let seeded = serde_json::to_vec(&json!({
        "schemaVersion": 2, "repo": repo, "modelId": "relocated", "variant": "bf16",
        "manifestFiles": ["bf16/*"],
        "resolvedFiles": [
            "bf16/text_encoder/model-00003-of-00003.safetensors",
            "bf16/text_encoder/model.safetensors.index.json",
        ],
        "backfilled": true,
    }))
    .unwrap();
    std::fs::write(&marker, &seeded).unwrap();
    assert!(
        huggingface_repo_cache_path(data_dir, repo).is_some_and(|root| !root.exists()),
        "fixture precondition: no repo cache at all, so nothing can be proven about the shards"
    );

    repair_torn_backfilled_receipts(&managed, data_dir);

    assert_eq!(
        std::fs::read(&marker).ok().as_deref(),
        Some(seeded.as_slice()),
        "an absent repo cache is not evidence of tearing — the receipt must survive byte-identical"
    );
}

/// sc-20526 (AC 4), the second no-evidence shape: the repo cache IS here, but not one cached
/// revision holds the index the receipt recorded (a revision bump, or a cache pruned of this
/// tier). Nothing can be read off disk, so nothing is proven and the receipt stays.
#[test]
fn repair_leaves_a_backfilled_receipt_whose_index_is_in_no_snapshot() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/index-not-cached";
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    std::fs::create_dir_all(snapshot.join("q4")).unwrap();
    std::fs::write(snapshot.join("q4/model_index.json"), b"{}").unwrap();
    std::fs::write(snapshot.join("q4/some.safetensors"), b"weights").unwrap();

    let managed = data_dir.join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    let marker = managed.join(".sceneworks-download-complete.json");
    let seeded = serde_json::to_vec(&json!({
        "schemaVersion": 2, "repo": repo, "modelId": "index_not_cached", "variant": "bf16",
        "manifestFiles": ["bf16/*"],
        "resolvedFiles": [
            "bf16/text_encoder/model-00003-of-00003.safetensors",
            "bf16/text_encoder/model.safetensors.index.json",
        ],
        "backfilled": true,
    }))
    .unwrap();
    std::fs::write(&marker, &seeded).unwrap();

    repair_torn_backfilled_receipts(&managed, data_dir);

    assert_eq!(
        std::fs::read(&marker).ok().as_deref(),
        Some(seeded.as_slice()),
        "a recorded index that is in NO snapshot proves nothing — the receipt must survive \
         byte-identical"
    );

    // Mutation check: the index arriving with a shard missing IS proof, and the same call now
    // removes the entry — so the survival above is a decision about evidence, not inaction.
    let encoder = snapshot.join("bf16/text_encoder");
    std::fs::create_dir_all(&encoder).unwrap();
    std::fs::write(
        encoder.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "lm_head.weight": "model-00003-of-00003.safetensors",
        }}))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(encoder.join("model-00003-of-00003.safetensors"), b"shard").unwrap();
    repair_torn_backfilled_receipts(&managed, data_dir);
    assert!(
        !marker.exists(),
        "a recorded index that IS on disk and names a missing shard is provably torn"
    );
}

#[test]
fn complete_pre_receipt_install_is_backfilled_and_protected_after_rename() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/backfill";
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
    let original = json!({"id":"backfill-model", "downloads":[{"provider":"huggingface", "repo":repo, "files":["old.safetensors"]}]});

    let initial = install_state_for(
        model_download_context(&original).unwrap(),
        &original,
        data_dir,
    );
    assert!(initial.installed);
    let marker = data_dir
        .join("models")
        .join(safe_download_dir(repo))
        .join(".sceneworks-download-complete.json");
    let receipt: Value = serde_json::from_slice(&std::fs::read(marker).unwrap()).unwrap();
    assert_eq!(receipt["resolvedFiles"], json!(["old.safetensors"]));
    assert_eq!(receipt["backfilled"], true);

    let renamed = json!({"id":"backfill-model", "downloads":[{"provider":"huggingface", "repo":repo, "files":["new.safetensors"]}]});
    let protected = install_state_for(
        model_download_context(&renamed).unwrap(),
        &renamed,
        data_dir,
    );
    assert!(
        protected.installed,
        "backfilled exact old set remains usable"
    );
    assert!(protected.update_available, "rename is offered as an update");
}

#[test]
fn shared_repo_backfill_preserves_each_models_receipt() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/shared";
    let snapshot = huggingface_repo_cache_path(data_dir, repo)
        .unwrap()
        .join("snapshots/rev-a");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("base.safetensors"), b"base weights").unwrap();
    std::fs::write(snapshot.join("turbo.safetensors"), b"turbo weights").unwrap();
    let base = json!({
        "id": "shared-base",
        "downloads": [{
            "provider": "huggingface",
            "repo": repo,
            "files": ["base.safetensors"]
        }]
    });
    let turbo = json!({
        "id": "shared-turbo",
        "downloads": [{
            "provider": "huggingface",
            "repo": repo,
            "files": ["turbo.safetensors"]
        }]
    });

    assert!(install_state_for(model_download_context(&base).unwrap(), &base, data_dir).installed);
    assert!(install_state_for(model_download_context(&turbo).unwrap(), &turbo, data_dir).installed);

    let managed = data_dir.join("models").join(safe_download_dir(repo));
    let base_receipts = receipt_file_sets(&managed, repo, Some("shared-base"));
    let turbo_receipts = receipt_file_sets(&managed, repo, Some("shared-turbo"));
    assert_eq!(base_receipts[0].files, vec!["base.safetensors".to_owned()]);
    assert_eq!(
        turbo_receipts[0].files,
        vec!["turbo.safetensors".to_owned()]
    );
    assert_eq!(
        receipt_entries(&managed).len(),
        2,
        "backfill must merge shared-repo model receipts instead of making the first writer win"
    );
}

#[test]
fn receipt_remains_usable_when_current_manifest_file_changes() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/model";
    let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
    let snapshot = cache.join("snapshots/rev-a");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
    let managed = data_dir.join("models/owner--model");
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&json!({
            "schemaVersion": 2, "repo": repo,
            "resolvedFiles": ["old.safetensors"]
        }))
        .unwrap(),
    )
    .unwrap();

    let files = receipt_file_sets(&managed, repo, None);
    assert_eq!(files[0].files, vec!["old.safetensors".to_owned()]);
    assert!(receipt_files_present(data_dir, repo, &files[0], None));
    assert!(!huggingface_cache_health(&cache, &["new.safetensors".to_owned()]).installed);

    let ambiguous = cache.join("snapshots/rev-b");
    std::fs::create_dir_all(&ambiguous).unwrap();
    std::fs::write(ambiguous.join("old.safetensors"), b"other weights").unwrap();
    assert!(
        !receipt_files_present(data_dir, repo, &files[0], None),
        "legacy receipt must identify one snapshot"
    );

    std::fs::remove_file(snapshot.join("old.safetensors")).unwrap();
    std::fs::remove_file(ambiguous.join("old.safetensors")).unwrap();
    assert!(
        !receipt_files_present(data_dir, repo, &files[0], None),
        "torn stale install is missing"
    );
}

#[test]
fn catalog_distinguishes_usable_stale_from_torn_and_points_at_cache() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let repo = "owner/model";
    let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
    let snapshot = cache.join("snapshots/rev-a");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
    let managed = data_dir.join("models").join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed).unwrap();
    std::fs::write(
        managed.join(".sceneworks-download-complete.json"),
        serde_json::to_vec(&json!({
            "schemaVersion": 2, "repo": repo, "resolvedFiles": ["old.safetensors"]
        }))
        .unwrap(),
    )
    .unwrap();
    let model = json!({"id":"model", "downloads":[{
        "provider":"huggingface", "repo":repo, "files":["new.safetensors"]
    }]});
    let context = model_download_context(&model).unwrap().unwrap();
    let stale = install_state_for(Some(context), &model, data_dir);
    assert!(stale.installed);
    assert!(stale.update_available);
    assert_eq!(
        stale.installed_path.as_deref(),
        Some(cache.to_string_lossy().as_ref())
    );

    std::fs::remove_file(snapshot.join("old.safetensors")).unwrap();
    let torn = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(!torn.installed);
    assert!(!torn.update_available);
}

#[test]
fn breaking_and_corequisite_softness_matrix() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    for breaking in [false, true] {
        for soft in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let data_dir = temp.path();
            let repo = "owner/model";
            let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
            let snapshot = cache.join("snapshots/rev-a");
            std::fs::create_dir_all(&snapshot).unwrap();
            std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
            let managed = data_dir.join("models").join(safe_download_dir(repo));
            std::fs::create_dir_all(&managed).unwrap();
            std::fs::write(
                managed.join(".sceneworks-download-complete.json"),
                serde_json::to_vec(&json!({
                    "schemaVersion": 2, "repo": repo,
                    "resolvedFiles": ["old.safetensors"]
                }))
                .unwrap(),
            )
            .unwrap();
            let model = json!({
                "id": "model",
                "downloads": [
                    {"provider":"huggingface", "repo":repo,
                     "files":["new.safetensors"], "breaking":breaking},
                    {"provider":"huggingface", "repo":"owner/dependency",
                     "coRequisite":true, "required": if soft { "soft" } else { "hard" }}
                ]
            });
            let state =
                install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
            assert_eq!(
                state.installed,
                !breaking && soft,
                "breaking={breaking}, soft={soft}"
            );
            assert!(
                state.update_available,
                "every stale/soft combination offers an update"
            );
            if !soft {
                assert!(state
                    .missing_required_files
                    .iter()
                    .any(|file| file.contains("owner/dependency")));
            }
            if !breaking && !soft {
                let mut omitted = model.clone();
                omitted.as_object_mut().unwrap()["downloads"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("breaking");
                omitted.as_object_mut().unwrap()["downloads"][1]
                    .as_object_mut()
                    .unwrap()
                    .remove("required");
                let defaulted = install_state_for(
                    model_download_context(&omitted).unwrap(),
                    &omitted,
                    data_dir,
                );
                assert!(!defaulted.installed, "omitted required defaults to hard");
                assert!(
                    defaulted.update_available,
                    "omitted breaking defaults to false"
                );
            }
        }
    }
}

/// sc-13680: the AUDIO lane's install-state now accounts for chatterbox_tts's two HARD
/// co-requisites (the `perth` + `voice_embedding` companion weights). Previously no audio model
/// used a co-requisite, so this generic gate (shared with PiD-gemma / Wan-Lightning) was untested
/// on the audio lane. A present primary snapshot with the perth/ve co-requisites ABSENT must NOT
/// report installed (it must surface as a repairable partial), and staging both must flip it to
/// installed — proving the perth+VE rehoming is enforced end to end.
#[test]
fn chatterbox_tts_install_state_gates_on_the_perth_and_ve_co_requisites() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    // The primary chatterbox snapshot (T3 + S3Gen + tokenizer) is present…
    let primary_repo = "ResembleAI/chatterbox";
    let primary_snapshot = huggingface_repo_cache_path(data_dir, primary_repo)
        .unwrap()
        .join("snapshots/main");
    std::fs::create_dir_all(&primary_snapshot).unwrap();
    for file in ["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"] {
        std::fs::write(primary_snapshot.join(file), b"weights").unwrap();
    }

    // …the chatterbox_tts catalog shape: primary + the two hard co-requisites (ve + perth),
    // mirroring builtin.models.jsonc (componentId is inert to install-state, which gates on
    // repo/files presence).
    let model = json!({
        "id": "chatterbox_tts",
        "downloads": [
            { "provider": "huggingface", "repo": primary_repo,
              "files": ["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"] },
            { "provider": "huggingface", "repo": "ResembleAI/chatterbox",
              "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", "coRequisite": true,
              "componentId": "voice_embedding", "files": ["ve.safetensors"] },
            { "provider": "huggingface", "repo": "SceneWorks/perth-implicit",
              "revision": "80b60f9caead09b8d3b512bda0b24038f28c08ec", "coRequisite": true,
              "componentId": "perth", "files": ["perth_implicit.safetensors"] }
        ]
    });

    // Co-requisites absent → NOT installed; both missing repos surface for repair.
    let missing = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        !missing.installed,
        "a present primary with the perth+ve co-requisites absent must not report installed"
    );
    assert!(
        missing.cache_incomplete,
        "the missing hard co-requisites make this a repairable partial install"
    );
    assert!(
        missing
            .missing_required_files
            .iter()
            .any(|entry| entry.contains("SceneWorks/perth-implicit"))
            && missing
                .missing_required_files
                .iter()
                .any(|entry| entry.contains("ResembleAI/chatterbox")),
        "both missing co-requisite repos must be reported, got {:?}",
        missing.missing_required_files
    );

    // Stage BOTH co-requisites at their pinned snapshots → install-state flips to installed.
    let ve = huggingface_repo_cache_path(data_dir, "ResembleAI/chatterbox")
        .unwrap()
        .join("snapshots/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18");
    std::fs::create_dir_all(&ve).unwrap();
    std::fs::write(ve.join("ve.safetensors"), b"weights").unwrap();
    let perth = huggingface_repo_cache_path(data_dir, "SceneWorks/perth-implicit")
        .unwrap()
        .join("snapshots/80b60f9caead09b8d3b512bda0b24038f28c08ec");
    std::fs::create_dir_all(&perth).unwrap();
    std::fs::write(perth.join("perth_implicit.safetensors"), b"weights").unwrap();

    let installed = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
    assert!(
        installed.installed,
        "primary + both hard co-requisites present must report installed"
    );
    assert!(
        installed.missing_required_files.is_empty(),
        "nothing is missing once the perth+ve co-requisites are staged, got {:?}",
        installed.missing_required_files
    );
}

/// sc-13684: MMAudio is a PURE 5-component assembly, so each tier catalogs a non-coRequisite `dit`
/// primary anchor plus FIVE hard component coRequisites (clip/synchformer/dit/vae/vocoder). Because
/// install-state gates on the hard coRequisites, an mmaudio tier stays not-installed until ALL five
/// component snapshots are present — even with the primary anchor on disk. Binds to the LIVE
/// builtin manifest so a dropped/renamed component coRequisite (or a lost pinned revision) fails here.
#[test]
fn mmaudio_install_state_gates_on_all_five_component_co_requisites() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    for model_id in ["mmaudio_small_16k", "mmaudio_large_44k"] {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let model = builtin_models_entry(model_id);

        // Every component is a hard coRequisite, and mmaudio catalogs FIVE of them.
        let co_requisites = model_co_requisite_downloads(&model);
        assert_eq!(
            co_requisites.len(),
            5,
            "{model_id}: MMAudio must catalog all five component coRequisites"
        );

        // Stage ONLY the primary anchor (the `dit` file) — no component coRequisite yet.
        let context = model_download_context(&model).unwrap().unwrap();
        let anchor_snapshot = huggingface_repo_cache_path(data_dir, &context.repo)
            .unwrap()
            .join("snapshots/eb13a1a98fdbec91753775c57b074ccdfc60587c");
        std::fs::create_dir_all(&anchor_snapshot).unwrap();
        for file in &context.files {
            let path = anchor_snapshot.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"weights").unwrap();
        }

        let missing = install_state_for(Some(context.clone()), &model, data_dir);
        assert!(
            !missing.installed,
            "{model_id}: a present anchor with the five components absent must not report installed"
        );
        assert!(
            missing.cache_incomplete,
            "{model_id}: missing hard component coRequisites make this a repairable partial install"
        );

        // Stage every component coRequisite at its pinned snapshot → install-state flips to installed.
        for co_requisite in &co_requisites {
            let repo = co_requisite.get("repo").and_then(Value::as_str).unwrap();
            let revision = co_requisite
                .get("revision")
                .and_then(Value::as_str)
                .unwrap();
            let snapshot = huggingface_repo_cache_path(data_dir, repo)
                .unwrap()
                .join("snapshots")
                .join(revision);
            std::fs::create_dir_all(&snapshot).unwrap();
            for file in string_array_field(co_requisite, "files") {
                let path = snapshot.join(seeded_file_path(&file));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"weights").unwrap();
            }
        }

        let installed = install_state_for(Some(context), &model, data_dir);
        assert!(
            installed.installed,
            "{model_id}: anchor + all five component coRequisites present must report installed"
        );
        assert!(
            installed.missing_required_files.is_empty(),
            "{model_id}: nothing is missing once every component is staged, got {:?}",
            installed.missing_required_files
        );

        // Existing installs predate the upstream repo rename and keep the exact immutable CLIP
        // snapshot under the legacy cache namespace. The catalog must continue to call those
        // installs ready instead of offering a repair that would fetch the same ~3.95 GB blob
        // again; the worker uses the same validator-proven plain source and hard-links the
        // canonical namespace cache-only when the filesystem supports it.
        let rename = sceneworks_core::hf_repo_renames::MMAUDIO_DFN5B_CLIP_RENAME;
        let clip = co_requisites
            .iter()
            .find(|download| {
                download.get("repo").and_then(Value::as_str) == Some(rename.current_repo)
            })
            .expect("live MMAudio manifest has the canonical CLIP component");
        let clip_revision = clip.get("revision").and_then(Value::as_str).unwrap();
        let clip_files = string_array_field(clip, "files");
        std::fs::remove_dir_all(
            huggingface_repo_cache_path(data_dir, rename.current_repo).unwrap(),
        )
        .unwrap();
        let legacy_repo = huggingface_repo_cache_path(data_dir, rename.legacy_repo).unwrap();
        let legacy_snapshot = legacy_repo.join("snapshots").join(clip_revision);
        for file in &clip_files {
            let path = legacy_snapshot.join(seeded_file_path(file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::symlink;

                let blob = legacy_repo.join("blobs").join("clip-blob");
                std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
                std::fs::write(&blob, b"existing cached weights").unwrap();
                symlink(&blob, &path).unwrap();
            }
            #[cfg(not(unix))]
            std::fs::write(path, b"existing cached weights").unwrap();
        }

        let legacy_installed =
            install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            legacy_installed.installed,
            "{model_id}: the exact legacy CLIP snapshot remains a complete offline install"
        );
        assert!(legacy_installed.missing_required_files.is_empty());
    }
}

/// sc-25679 (E4): Iris-3B's GENERATION task is the backbone root files plus ONE hard
/// `text_encoder` component (Qwen3-VL-4B-Instruct at its pinned revision). The primary fetches
/// only the root `config.yaml` + `model.safetensors` — never the `depth/` / `upscaler/` task
/// backbones — and the entry stays not-installed (a repairable partial install) until the
/// encoder snapshot is present, so Image Studio never advertises a half-installed generator.
#[test]
fn iris_generation_install_state_gates_on_its_text_encoder_component() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path();
    let model = builtin_models_entry("iris_3b");

    let co_requisites = model_co_requisite_downloads(&model);
    assert_eq!(
        co_requisites.len(),
        1,
        "generation has exactly one component"
    );
    let encoder = &co_requisites[0];
    assert_eq!(
        encoder.get("componentId").and_then(Value::as_str),
        Some("text_encoder")
    );
    assert_eq!(
        encoder.get("repo").and_then(Value::as_str),
        Some("Qwen/Qwen3-VL-4B-Instruct")
    );

    let context = model_download_context(&model).unwrap().unwrap();
    assert_eq!(context.repo, "speridlabs/iris-3b");
    assert_eq!(context.files, ["config.yaml", "model.safetensors"]);
    let backbone = huggingface_repo_cache_path(data_dir, &context.repo)
        .unwrap()
        .join("snapshots/7445443349bc9abe3c96f01ff793e2098ca012b3");
    std::fs::create_dir_all(&backbone).unwrap();
    for file in &context.files {
        std::fs::write(backbone.join(file), b"weights").unwrap();
    }
    let partial = install_state_for(Some(context.clone()), &model, data_dir);
    assert!(
        !partial.installed,
        "backbone without its encoder is not installed"
    );
    assert!(
        partial.cache_incomplete,
        "a missing encoder is a repairable partial install"
    );

    let revision = encoder.get("revision").and_then(Value::as_str).unwrap();
    let snapshot = huggingface_repo_cache_path(data_dir, "Qwen/Qwen3-VL-4B-Instruct")
        .unwrap()
        .join("snapshots")
        .join(revision);
    for file in string_array_field(encoder, "files") {
        let path = snapshot.join(seeded_file_path(&file));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"weights").unwrap();
    }
    // The shard index is validated, so it must name the shard the glob seeded.
    let shard = seeded_file_path("model-*.safetensors");
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": { "w": shard } })).unwrap(),
    )
    .unwrap();
    let installed = install_state_for(Some(context), &model, data_dir);
    assert!(
        installed.installed,
        "backbone + encoder is installed: incomplete={} missing={:?} path={:?}",
        installed.cache_incomplete, installed.missing_required_files, installed.installed_path
    );
    assert!(
        installed.missing_required_files.is_empty(),
        "{:?}",
        installed.missing_required_files
    );
}

/// sc-25680: Iris-3B is served on Candle/CUDA too, so its catalog entry is NOT platform-scoped:
/// after the running OS's `retain_downloads_for_os` filter on Windows and Linux the backbone and
/// its `text_encoder` component both survive, and the install state gates on the encoder exactly
/// as it does on macOS (a backbone alone is a repairable partial install).
#[test]
fn iris_generation_install_state_is_the_same_on_the_candle_platforms() {
    let _env = isolate_hf_cache();
    let base = builtin_models_entry("iris_3b");
    assert_ne!(
        base.get("macOnly").and_then(Value::as_bool),
        Some(true),
        "iris_3b has a Candle lane, so it must not be labelled macOS-only"
    );
    for os in ["windows", "linux"] {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let mut model = base.clone();
        retain_downloads_for_os(&mut model, os);
        let co_requisites = model_co_requisite_downloads(&model);
        assert_eq!(co_requisites.len(), 1, "{os}: the text_encoder survives");
        assert_eq!(
            co_requisites[0].get("componentId").and_then(Value::as_str),
            Some("text_encoder"),
            "{os}"
        );
        let context = model_download_context(&model).unwrap().unwrap();
        assert_eq!(context.repo, "speridlabs/iris-3b", "{os}");
        assert_eq!(context.files, ["config.yaml", "model.safetensors"], "{os}");

        let backbone = huggingface_repo_cache_path(data_dir, &context.repo)
            .unwrap()
            .join("snapshots/7445443349bc9abe3c96f01ff793e2098ca012b3");
        std::fs::create_dir_all(&backbone).unwrap();
        for file in &context.files {
            std::fs::write(backbone.join(file), b"weights").unwrap();
        }
        let partial = install_state_for(Some(context.clone()), &model, data_dir);
        assert!(
            !partial.installed && partial.cache_incomplete,
            "{os}: backbone without its encoder is a repairable partial install"
        );

        let encoder = &co_requisites[0];
        let revision = encoder.get("revision").and_then(Value::as_str).unwrap();
        let snapshot = huggingface_repo_cache_path(data_dir, "Qwen/Qwen3-VL-4B-Instruct")
            .unwrap()
            .join("snapshots")
            .join(revision);
        for file in string_array_field(encoder, "files") {
            let path = snapshot.join(seeded_file_path(&file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"weights").unwrap();
        }
        let shard = seeded_file_path("model-*.safetensors");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({ "weight_map": { "w": shard } })).unwrap(),
        )
        .unwrap();
        let installed = install_state_for(Some(context), &model, data_dir);
        assert!(
            installed.installed,
            "{os}: backbone + encoder is installed: {:?}",
            installed.missing_required_files
        );
    }
}

/// sc-26234 (epic sc-26233): FLUX 3 Image Klein is ONE package — the DiT, the autoencoder AND
/// the Qwen3-VL text encoder all live under the one pinned `black-forest-labs/flux-3-image-klein`
/// root (`gen_core_flux3::GenerationResources` reads them from there), so the entry has exactly
/// one download and NO co-requisite component (the inverse of Iris-3B, whose encoder is a
/// separate `text_encoder` component). Install state therefore gates on the bundled
/// `text_encoder/` files of the SAME snapshot: the DiT + autoencoder alone is a repairable
/// partial install, never an installed generator.
fn seed_flux3_runtime_bundle(snapshot: &FsPath) {
    for file in [
        "config.json",
        "model.safetensors",
        "ae.safetensors",
        "text_encoder/config.json",
        "text_encoder/tokenizer.json",
        "text_encoder/tokenizer_config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
    ] {
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"weights").unwrap();
    }
    std::fs::write(snapshot.join("text_encoder/model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}})).unwrap()).unwrap();
}

// A source-produced v2 fixture receipt; runtime files alone never mint this trust identity.
fn seed_flux3_exact_receipt(data_dir: &FsPath, model: &Value) -> PathBuf {
    let row = &model["downloads"][0];
    let managed = data_dir
        .join("models")
        .join(safe_download_dir(row["repo"].as_str().unwrap()));
    std::fs::create_dir_all(&managed).unwrap();
    let path = managed.join(".sceneworks-download-complete.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({"schemaVersion":2,
        "repo":row["repo"],"modelId":model["id"],"variant":"bf16",
        "snapshotRevision":row["revision"],"resolvedFiles":["model.safetensors"]}))
        .unwrap(),
    )
    .unwrap();
    path
}

fn assert_flux3_primary_and_optional_pup(model: &Value) -> &Value {
    let base = model["id"] == "flux3_klein_base";
    let repo = if base {
        "black-forest-labs/flux-3-image-klein-base"
    } else {
        "black-forest-labs/flux-3-image-klein"
    };
    let revision = if base {
        "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4"
    } else {
        "8454017f9ebaada30c998fccd9f67a80c7580f68"
    };
    let downloads = model["downloads"].as_array().unwrap();
    let primary: Vec<_> = downloads
        .iter()
        .filter(|row| row.get("coRequisite").and_then(Value::as_bool) != Some(true))
        .collect();
    assert!(!primary.is_empty(), "at least the dense primary package");
    for row in &primary {
        assert_eq!(row["repo"], repo);
        assert_eq!(
            row["files"],
            json!([]),
            "every tier fetches the whole original"
        );
    }
    let primary_count = primary.len();
    let primary = primary
        .into_iter()
        .find(|row| row["default"] == true)
        .unwrap();
    assert_eq!(primary["provider"], "huggingface");
    assert_eq!(primary["repo"], repo);
    assert_eq!(primary["revision"], revision);
    assert_eq!(primary["default"], true);
    assert_eq!(
        primary["files"],
        json!([]),
        "whole primary repository required"
    );
    assert!(primary
        .get("required")
        .is_none_or(|value| value.as_bool() == Some(true)));
    let optional = model_co_requisite_downloads(model);
    assert!(
        optional.len() <= 1,
        "only the exact soft PUP may accompany the primary"
    );
    assert_eq!(
        downloads.len(),
        primary_count + optional.len(),
        "tier rows plus the one optional component"
    );
    for row in optional {
        assert_eq!(row["provider"], "huggingface");
        assert_eq!(row["repo"], "black-forest-labs/flux-3-image-klein-pup");
        assert_eq!(row["revision"], "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed");
        assert_eq!(row["componentId"], "prompt_upsampler");
        assert_eq!(row["required"], "soft", "no required external components");
        assert_eq!(row["files"], json!([]));
        assert_ne!(row.get("default").and_then(Value::as_bool), Some(true));
    }
    primary
}

#[test]
fn flux3_package_fixture_rejects_required_or_wrong_optional_components() {
    let model = builtin_models_entry("flux3_klein");
    let index = model["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .position(|row| row["coRequisite"] == true)
        .expect("combined catalog exposes the soft PUP");
    for (field, value) in [
        ("required", json!(true)),
        ("repo", json!("not-the-pinned-source")),
        ("revision", json!("0".repeat(40))),
        ("componentId", json!("text_encoder")),
        ("files", json!(["model.safetensors"])),
        ("coRequisite", json!(false)),
    ] {
        let mut invalid = model.clone();
        invalid["downloads"][index][field] = value;
        assert!(
            std::panic::catch_unwind(|| assert_flux3_primary_and_optional_pup(&invalid)).is_err(),
            "{field} must not weaken the exact primary/soft-PUP fixture"
        );
    }
    let mut duplicate = model.clone();
    duplicate["downloads"]
        .as_array_mut()
        .unwrap()
        .push(model["downloads"][index].clone());
    assert!(
        std::panic::catch_unwind(|| assert_flux3_primary_and_optional_pup(&duplicate)).is_err()
    );
}

#[test]
fn flux3_klein_install_state_gates_on_its_bundled_text_encoder_without_a_component() {
    for (id, repo, revision, size) in [
        (
            "flux3_klein",
            "black-forest-labs/flux-3-image-klein",
            "8454017f9ebaada30c998fccd9f67a80c7580f68",
            43_317_649_602,
        ),
        (
            "flux3_klein_base",
            "black-forest-labs/flux-3-image-klein-base",
            "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4",
            43_229_093_855,
        ),
    ] {
        assert_flux3_runtime_install(id, repo, revision, size);
    }
}

fn assert_flux3_runtime_install(
    id: &str,
    expected_repo: &str,
    expected_revision: &str,
    expected_size: u64,
) {
    let _env = isolate_hf_cache();
    let temp = tempfile::Builder::new()
        .prefix(&format!("flux3-install-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    let data_dir = temp.path();
    let model = builtin_models_entry(id);
    let primary = assert_flux3_primary_and_optional_pup(&model);
    assert!(model.get("component").is_none());
    let revision = primary["revision"].as_str().unwrap();
    assert_eq!(revision, expected_revision);
    let context = model_download_context(&model).unwrap().unwrap();
    assert_eq!(context.repo, expected_repo);
    assert!(
        context.files.is_empty(),
        "download ALL files, not this runtime fixture's minimum set"
    );
    assert_eq!(context.fallback_size_bytes, Some(expected_size));
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    let repo_root = library.repository_root(&context.repo).unwrap();
    let snapshot = library
        .snapshots_root_from_repository_root(&repo_root)
        .unwrap()
        .join(revision);
    std::fs::create_dir_all(&snapshot).unwrap();
    for file in ["config.json", "model.safetensors", "ae.safetensors"] {
        std::fs::write(snapshot.join(file), b"weights").unwrap();
    }
    let partial = install_state_for(Some(context.clone()), &model, data_dir);
    assert!(
        !partial.installed,
        "DiT + AE with the ENTIRE bundled encoder absent is not ready"
    );
    assert!(
        partial.cache_incomplete,
        "whole-repo download policy must not weaken runtime completeness"
    );
    assert!(!model_variant_states(&model, data_dir)[0].installed);
    seed_flux3_runtime_bundle(&snapshot);
    let unverified = install_state_for(Some(context.clone()), &model, data_dir);
    assert!(
        !unverified.installed
            && model_variant_states(&model, data_dir)
                .iter()
                .all(|row| !row.installed)
    );
    assert!(unverified
        .missing_required_files
        .iter()
        .any(|message| message.contains("install verification required")));
    seed_flux3_exact_receipt(data_dir, &model);
    for required in [
        "model.safetensors",
        "ae.safetensors",
        "text_encoder/tokenizer_config.json",
        "text_encoder/model-00002-of-00002.safetensors",
    ] {
        let path = snapshot.join(required);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let torn = install_state_for(Some(context.clone()), &model, data_dir);
        assert!(
            !torn.installed && torn.cache_incomplete,
            "missing {required}: {:?}",
            torn.missing_required_files
        );
        assert!(!model_variant_states(&model, data_dir)[0].installed);
        std::fs::write(&path, bytes).unwrap();
    }
    let index = snapshot.join("text_encoder/model.safetensors.index.json");
    let valid_index = std::fs::read(&index).unwrap();
    std::fs::write(
        &index,
        serde_json::to_vec(&json!({"weight_map":{"w":"../model.safetensors"}})).unwrap(),
    )
    .unwrap();
    let escaped = install_state_for(Some(context.clone()), &model, data_dir);
    assert!(
        !escaped.installed && escaped.cache_incomplete,
        "index traversal must refuse"
    );
    std::fs::write(&index, valid_index).unwrap();
    let installed = install_state_for(Some(context.clone()), &model, data_dir);
    assert!(
        installed.installed && !installed.cache_incomplete,
        "{:?}",
        installed.missing_required_files
    );
    assert!(installed.missing_required_files.is_empty());
    assert_eq!(
        std::fs::canonicalize(installed.installed_path.as_deref().unwrap()).unwrap(),
        std::fs::canonicalize(&snapshot).unwrap(),
        "the typed resolver must identify this exact selected snapshot (including Windows extended-path form)"
    );
    assert!(model_variant_states(&model, data_dir)[0].installed);
    let managed = data_dir
        .join("models")
        .join(safe_download_dir(&context.repo));
    assert!(
        managed.join(".sceneworks-download-complete.json").exists(),
        "the explicit source fixture receipt, not arbitrary cache completeness, authorizes readiness"
    );
}

#[test]
fn flux3_klein_runtime_bundle_requires_the_selected_revision_and_managed_receipt_identity() {
    for (id, repo, revision) in [
        (
            "flux3_klein",
            "black-forest-labs/flux-3-image-klein",
            "8454017f9ebaada30c998fccd9f67a80c7580f68",
        ),
        (
            "flux3_klein_base",
            "black-forest-labs/flux-3-image-klein-base",
            "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4",
        ),
    ] {
        assert_flux3_runtime_authority(id, repo, revision);
    }
}

#[test]
fn flux3_derived_catalog_readiness_requires_the_complete_original_on_both_ids() {
    use sha2::{Digest, Sha256};
    let _env = isolate_hf_cache();
    for id in ["flux3_klein", "flux3_klein_base"] {
        let temp = tempfile::tempdir().unwrap();
        let mut model = builtin_models_entry(id);
        let mut row = model["downloads"][0].clone();
        let repo = row["repo"].as_str().unwrap().to_owned();
        let revision = row["revision"].as_str().unwrap().to_owned();
        // Synthetic pins are strictly test-only, not real base conversion evidence.
        row["variant"] = json!("q8");
        row["default"] = json!(false);
        row["localDerivation"] = json!({"fromVariant":"bf16","conversion":"fixture-only",
            "weightsFile":"model.safetensors","weightsBytes":1,
            "weightsSha256":format!("{:x}", Sha256::digest(b"x")),"reason":"synthetic test"});
        model["downloads"] = json!([model["downloads"][0].clone(), row]);
        let derived = temp
            .path()
            .join("models/derived")
            .join(id)
            .join("q8/fixture-only");
        std::fs::create_dir_all(&derived).unwrap();
        std::fs::write(derived.join("model.safetensors"), b"x").unwrap();
        let q8 = |model: &Value| {
            model_variant_states(model, temp.path())
                .into_iter()
                .find(|state| state.variant == "q8")
                .unwrap()
        };
        let absent = q8(&model);
        assert!(
            !absent.installed && absent.derivation_pending == Some(true),
            "{id}"
        );
        assert!(
            absent
                .missing_required_files
                .iter()
                .any(|file| file.contains("complete pinned original")),
            "{id}"
        );
        let library = sceneworks_core::hf_home::model_source_library(temp.path());
        let repo_root = library.repository_root(&repo).unwrap();
        let root = library
            .snapshots_root_from_repository_root(&repo_root)
            .unwrap()
            .join(revision);
        seed_flux3_runtime_bundle(&root);
        assert!(
            !q8(&model).installed,
            "derived bytes + unreceipted original remain unavailable"
        );
        seed_flux3_exact_receipt(temp.path(), &model);
        assert!(
            q8(&model).installed,
            "{id}: correct original plus pinned derived bytes"
        );
        std::fs::remove_file(root.join("text_encoder/tokenizer_config.json")).unwrap();
        assert!(
            !q8(&model).installed,
            "{id}: a torn original cannot be ready"
        );
        seed_flux3_runtime_bundle(&root);
        std::fs::write(derived.join("model.safetensors"), b"tampered").unwrap();
        assert!(
            !q8(&model).installed,
            "{id}: corrupted conversion cannot be ready"
        );
    }
}

#[test]
fn flux3_runtime_bundles_do_not_alias_between_base_and_distilled() {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let library = sceneworks_core::hf_home::model_source_library(temp.path());
    let distilled = builtin_models_entry("flux3_klein");
    let base = builtin_models_entry("flux3_klein_base");
    let distilled_context = model_download_context(&distilled).unwrap().unwrap();
    let base_context = model_download_context(&base).unwrap().unwrap();
    let root = library.repository_root(&distilled_context.repo).unwrap();
    seed_flux3_runtime_bundle(
        &library
            .snapshots_root_from_repository_root(&root)
            .unwrap()
            .join("8454017f9ebaada30c998fccd9f67a80c7580f68"),
    );
    seed_flux3_exact_receipt(temp.path(), &distilled);
    assert!(install_state_for(Some(distilled_context), &distilled, temp.path()).installed);
    assert!(!install_state_for(Some(base_context), &base, temp.path()).installed);
}

fn assert_flux3_runtime_authority(id: &str, expected_repo: &str, expected_revision: &str) {
    let _env = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let model = builtin_models_entry(id);
    let context = model_download_context(&model).unwrap().unwrap();
    assert_eq!(context.repo, expected_repo);
    let library = sceneworks_core::hf_home::model_source_library(temp.path());
    let repo_root = library.repository_root(expected_repo).unwrap();
    let snapshot = library
        .snapshots_root_from_repository_root(&repo_root)
        .unwrap()
        .join(expected_revision);
    seed_flux3_runtime_bundle(&snapshot);
    let ready = || install_state_for(Some(context.clone()), &model, temp.path());
    let worker = || {
        sceneworks_worker::flux3_receipted_snapshot(
            temp.path(),
            expected_repo,
            expected_revision,
            id,
        )
    };
    assert!(
        !ready().installed && worker().is_none(),
        "complete pinned cache is not an authorized install"
    );
    assert!(model_variant_states(&model, temp.path())
        .iter()
        .all(|state| !state.installed));
    let marker = seed_flux3_exact_receipt(temp.path(), &model);
    assert!(
        ready().installed && worker().is_some(),
        "same source receipt must authorize API and worker"
    );
    assert!(model_variant_states(&model, temp.path())[0].installed);
    let valid = std::fs::read(&marker).unwrap();
    for (key, wrong) in [
        ("snapshotRevision", "b".repeat(40)),
        ("repo", "wrong/repository".into()),
        ("modelId", "wrong_model".into()),
    ] {
        let mut bad: Value = serde_json::from_slice(&valid).unwrap();
        bad[key] = json!(wrong);
        std::fs::write(&marker, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(
            !ready().installed && worker().is_none(),
            "{id}: wrong {key}"
        );
        assert!(!model_variant_states(&model, temp.path())[0].installed);
    }
    std::fs::write(&marker, b"corrupt receipt").unwrap();
    assert!(!ready().installed && worker().is_none());
    std::fs::write(&marker, &valid).unwrap();
    assert!(ready().installed);
    std::fs::write(
        snapshot.join("model.safetensors"),
        b"corrupt after the trusted tree stamp",
    )
    .unwrap();
    assert!(
        !ready().installed && worker().is_none(),
        "tree stamp drift is never repaired from current bytes"
    );
}

#[cfg(unix)]
#[test]
fn flux3_klein_runtime_bundle_accepts_its_own_hf_blob_symlinks() {
    let _env = isolate_hf_cache();
    let temp = tempfile::Builder::new()
        .prefix(&format!("flux3-blobs-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    let model = builtin_models_entry("flux3_klein");
    let context = model_download_context(&model).unwrap().unwrap();
    let repo_root = huggingface_repo_cache_path(temp.path(), &context.repo).unwrap();
    let snapshot = repo_root.join("snapshots/8454017f9ebaada30c998fccd9f67a80c7580f68");
    seed_flux3_runtime_bundle(&snapshot);
    std::fs::create_dir_all(repo_root.join("blobs")).unwrap();
    let shard = snapshot.join("text_encoder/model-00001-of-00002.safetensors");
    let blob = repo_root.join("blobs/fixture-shard");
    std::fs::rename(&shard, &blob).unwrap();
    std::os::unix::fs::symlink(&blob, &shard).unwrap();
    seed_flux3_exact_receipt(temp.path(), &model);
    assert!(
        install_state_for(Some(context), &model, temp.path()).installed,
        "authorized same-repository blob links are not path traversal"
    );
}

/// sc-26234: the package is GATED upstream (FLUX Kommunity License), pulled directly from Black
/// Forest Labs with no SceneWorks re-host, so the catalog projection must carry `gated: true`
/// and the `huggingface.co` credential host the Models screen keys its "add a Hugging Face
/// token" / "request access" affordances on — and the stored-HF-token download path with it.
#[test]
fn flux3_klein_is_gated_behind_a_huggingface_credential() {
    let model = builtin_models_entry("flux3_klein");
    let mut object = model.as_object().unwrap().clone();
    apply_gating_fields(&mut object);
    assert_eq!(object.get("gated").and_then(Value::as_bool), Some(true));
    assert_eq!(
        object.get("credentialHost").and_then(Value::as_str),
        Some("huggingface.co")
    );
    assert!(
        !model_requires_license_acknowledgment(&model),
        "a gated repo is refused upstream without a token; it needs no acknowledgment gate"
    );
    assert_eq!(
        derive_credential_host(&object).as_deref(),
        Some("huggingface.co"),
        "the download itself resolves to the same Hugging Face credential"
    );
}

/// sc-26235: the same pinned package serves native MLX and Candle/CUDA. Platform projection
/// must not hide the image entry or replace its upstream download on Windows or Linux.
#[test]
fn flux3_klein_keeps_its_native_candle_availability_and_package_on_each_platform() {
    let _env = isolate_hf_cache();
    let base = builtin_models_entry("flux3_klein");
    assert_eq!(
        base.get("macOnly").and_then(Value::as_bool),
        Some(false),
        "flux3_klein has native providers on both platforms"
    );
    for os in ["windows", "linux"] {
        let temp = tempfile::tempdir().unwrap();
        let mut projected = vec![base.clone()];
        retain_models_for_os(&mut projected, os, temp.path()).unwrap();
        assert_eq!(projected.len(), 1, "{os}: an image entry is not withdrawn");
        let mut model = projected.remove(0);
        assert_eq!(
            model.get("macOnly").and_then(Value::as_bool),
            Some(false),
            "{os}: the native Candle model remains selectable"
        );
        assert!(
            !video_model_withdrawn_on_platform("flux3_klein", &model, os),
            "{os}"
        );
        retain_downloads_for_os(&mut model, os);
        assert_flux3_primary_and_optional_pup(&model);
        let context = model_download_context(&model).unwrap().unwrap();
        assert_eq!(context.repo, "black-forest-labs/flux-3-image-klein", "{os}");
    }
}

/// sc-24214: Supra2-IMG is served on Candle too, under the same engine id, so the running OS's
/// `retain_downloads_for_os` filter keeps the IDENTICAL install on Windows and Linux as on macOS:
/// the pinned `model_final_ema.pt` primary and both pinned co-requisites (`text_encoder`, `vae`).
#[test]
fn supra2_installs_the_same_three_components_on_every_platform() {
    let base = builtin_models_entry("supra2_img");
    assert_eq!(base.get("macOnly").and_then(Value::as_bool), Some(false));
    let retained = |os: &str| {
        let mut model = base.clone();
        retain_downloads_for_os(&mut model, os);
        model
    };
    let mac = retained("macos");
    for os in ["windows", "linux"] {
        let model = retained(os);
        assert_eq!(
            model["downloads"], mac["downloads"],
            "{os}: same downloads as macOS"
        );
        let components: Vec<_> = model_co_requisite_downloads(&model)
            .iter()
            .map(|row| {
                (
                    row["componentId"].as_str().unwrap().to_owned(),
                    row["repo"].as_str().unwrap().to_owned(),
                    row["revision"].as_str().unwrap().to_owned(),
                )
            })
            .collect();
        assert_eq!(
            components,
            [
                (
                    "text_encoder".to_owned(),
                    "google/flan-t5-base".to_owned(),
                    "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2".to_owned()
                ),
                (
                    "vae".to_owned(),
                    "stabilityai/sd-vae-ft-mse".to_owned(),
                    "31f26fdeee1355a5c34592e401dd41e45d25a493".to_owned()
                ),
            ],
            "{os}"
        );
        let context = model_download_context(&model).unwrap().unwrap();
        assert_eq!(context.repo, "SupraLabs/Supra2-IMG", "{os}");
        assert_eq!(context.files, ["model_final_ema.pt"], "{os}");
    }
}

/// sc-25682 (E4): Iris-3B DEPTH is its own task install — exactly the three `depth/` export files
/// at the pinned revision, NO co-requisite (no text encoder, no generation backbone), installed
/// only when all three are present. Same on every platform (both lanes serve it).
#[test]
fn iris_depth_installs_from_its_own_task_closure_without_a_text_encoder() {
    let _env = isolate_hf_cache();
    let base = builtin_models_entry("iris_3b_depth");
    assert_eq!(base.get("type").and_then(Value::as_str), Some("utility"));
    for os in ["macos", "windows", "linux"] {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let mut model = base.clone();
        retain_downloads_for_os(&mut model, os);
        assert!(
            model_co_requisite_downloads(&model).is_empty(),
            "{os}: depth needs no text encoder or backbone component"
        );
        let context = model_download_context(&model).unwrap().unwrap();
        assert_eq!(context.repo, "speridlabs/iris-3b", "{os}");
        assert_eq!(
            context.files,
            [
                "depth/config.yaml",
                "depth/model.safetensors",
                "depth/empty_prompt.safetensors"
            ],
            "{os}"
        );
        let snapshot = huggingface_repo_cache_path(data_dir, &context.repo)
            .unwrap()
            .join("snapshots/7445443349bc9abe3c96f01ff793e2098ca012b3");
        // The GENERATION backbone's root files are not a depth install.
        std::fs::create_dir_all(snapshot.join("depth")).unwrap();
        for file in ["config.yaml", "model.safetensors"] {
            std::fs::write(snapshot.join(file), b"weights").unwrap();
        }
        for file in &context.files[..2] {
            std::fs::write(snapshot.join(file), b"weights").unwrap();
        }
        let torn = install_state_for(Some(context.clone()), &model, data_dir);
        assert!(
            !torn.installed,
            "{os}: two of three export files is not installed"
        );
        std::fs::write(snapshot.join(&context.files[2]), b"weights").unwrap();
        let installed = install_state_for(Some(context), &model, data_dir);
        assert!(
            installed.installed,
            "{os}: the complete depth export is installed: {:?}",
            installed.missing_required_files
        );
    }
}

/// sc-25684 (E4): Iris-3B RESTORATION is its own task install too — exactly the three `upscaler/`
/// export files at the pinned revision, NO co-requisite (no text encoder), installed only when all
/// three are present; neither the generation backbone's root files nor the `depth/` export make
/// it installed. Same on every platform (both lanes serve it).
#[test]
fn iris_restoration_installs_from_its_own_task_closure_without_a_text_encoder() {
    let _env = isolate_hf_cache();
    let base = builtin_models_entry("iris_3b_restore");
    assert_eq!(base.get("type").and_then(Value::as_str), Some("utility"));
    for os in ["macos", "windows", "linux"] {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let mut model = base.clone();
        retain_downloads_for_os(&mut model, os);
        assert!(
            model_co_requisite_downloads(&model).is_empty(),
            "{os}: restoration needs no text encoder or backbone component"
        );
        let context = model_download_context(&model).unwrap().unwrap();
        assert_eq!(context.repo, "speridlabs/iris-3b", "{os}");
        assert_eq!(
            context.files,
            [
                "upscaler/config.yaml",
                "upscaler/model.safetensors",
                "upscaler/empty_prompt.safetensors"
            ],
            "{os}"
        );
        let snapshot = huggingface_repo_cache_path(data_dir, &context.repo)
            .unwrap()
            .join("snapshots/7445443349bc9abe3c96f01ff793e2098ca012b3");
        // The OTHER tasks' complete installs are not a restoration install.
        std::fs::create_dir_all(snapshot.join("depth")).unwrap();
        std::fs::create_dir_all(snapshot.join("upscaler")).unwrap();
        for file in [
            "config.yaml",
            "model.safetensors",
            "depth/config.yaml",
            "depth/model.safetensors",
            "depth/empty_prompt.safetensors",
        ] {
            std::fs::write(snapshot.join(file), b"weights").unwrap();
        }
        let absent = install_state_for(Some(context.clone()), &model, data_dir);
        assert!(
            !absent.installed,
            "{os}: other tasks do not install restoration"
        );
        for file in &context.files[..2] {
            std::fs::write(snapshot.join(file), b"weights").unwrap();
        }
        let torn = install_state_for(Some(context.clone()), &model, data_dir);
        assert!(
            !torn.installed && torn.cache_incomplete,
            "{os}: two of three export files is a repairable partial install"
        );
        std::fs::write(snapshot.join(&context.files[2]), b"weights").unwrap();
        let installed = install_state_for(Some(context), &model, data_dir);
        assert!(
            installed.installed,
            "{os}: the complete restoration export is installed: {:?}",
            installed.missing_required_files
        );
    }
}

/// sc-13681 / sc-13686: each MOSS TTS entry advertises its RVQ codec as a HARD component
/// coRequisite (`moss_ttsd_v05` → XY_Tokenizer, `moss_tts_realtime` → MOSS-Audio-Tokenizer). Because
/// install-state gates on hard coRequisites, a MOSS entry stays not-installed until the codec
/// snapshot is present — even with its primary AR snapshot on disk — so the Model Manager reports a
/// repairable partial install and names the missing codec repo, then flips to installed once the
/// codec is staged. Binds to the LIVE builtin manifest so a dropped/renamed codec coRequisite or a
/// lost pinned revision fails HERE, not silently at synth time. The negative path's twin at the
/// worker's `resolve_co_requisites` seam (model_jobs::…a_missing_moss_codec_fails…) proves the JOB
/// then fails at load with the actionable error; this proves the catalog never advertises it ready.
#[test]
fn moss_tts_install_state_gates_on_the_codec_co_requisite() {
    let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
    for model_id in ["moss_ttsd_v05", "moss_tts_realtime"] {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let model = builtin_models_entry(model_id);

        // Exactly one hard codec coRequisite, keyed on the `codec` component id.
        let co_requisites = model_co_requisite_downloads(&model);
        assert_eq!(
            co_requisites.len(),
            1,
            "{model_id}: MOSS advertises exactly the codec coRequisite"
        );
        let codec = &co_requisites[0];
        assert_eq!(
            codec.get("componentId").and_then(Value::as_str),
            Some("codec"),
            "{model_id}: the coRequisite is the `codec` component"
        );
        let codec_repo = codec.get("repo").and_then(Value::as_str).unwrap();

        // Stage the primary AR snapshot (the manifest's non-coRequisite download) — codec absent.
        let context = model_download_context(&model).unwrap().unwrap();
        let primary = model
            .get("downloads")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .find(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
            .expect("MOSS entry has a primary download");
        let primary_rev = primary.get("revision").and_then(Value::as_str).unwrap();
        let primary_snapshot = huggingface_repo_cache_path(data_dir, &context.repo)
            .unwrap()
            .join("snapshots")
            .join(primary_rev);
        std::fs::create_dir_all(&primary_snapshot).unwrap();
        for file in &context.files {
            let path = primary_snapshot.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"weights").unwrap();
        }

        let missing = install_state_for(Some(context.clone()), &model, data_dir);
        assert!(
            !missing.installed,
            "{model_id}: primary present but codec absent must not report installed"
        );
        assert!(
            missing.cache_incomplete,
            "{model_id}: the missing hard codec makes this a repairable partial install"
        );
        assert!(
            missing
                .missing_required_files
                .iter()
                .any(|entry| entry.contains(codec_repo)),
            "{model_id}: the missing codec repo must be reported, got {:?}",
            missing.missing_required_files
        );

        // Stage the codec snapshot at its pinned revision → install-state flips to installed.
        let codec_rev = codec.get("revision").and_then(Value::as_str).unwrap();
        let codec_snapshot = huggingface_repo_cache_path(data_dir, codec_repo)
            .unwrap()
            .join("snapshots")
            .join(codec_rev);
        std::fs::create_dir_all(&codec_snapshot).unwrap();
        for file in string_array_field(codec, "files") {
            let path = codec_snapshot.join(seeded_file_path(&file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"weights").unwrap();
        }

        let installed = install_state_for(Some(context), &model, data_dir);
        assert!(
            installed.installed,
            "{model_id}: primary + codec present must report installed"
        );
        assert!(
            installed.missing_required_files.is_empty(),
            "{model_id}: nothing missing once the codec is staged, got {:?}",
            installed.missing_required_files
        );
    }
}

/// The three descriptor-required SDXL components (CLIP-L/bigG tokenizers + fp16-fix VAE) remain
/// Candle-only hard co-requisites on every Candle-SDXL base + InstantID. OpenPose is different: it
/// is one cross-platform soft install companion on exactly the five accepted base models.
/// Missing it must offer an update without making ordinary MLX or Candle generation uninstalled.
/// Binds both contracts to the LIVE builtin manifest and install-state implementation.
#[test]
fn sdxl_shared_components_separate_hard_candle_and_soft_cross_platform_state() {
    use std::collections::BTreeSet;

    let _env = isolate_hf_cache();
    let hard_want: BTreeSet<String> = ["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"]
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    let soft_want = BTreeSet::from(["controlnet_openpose".to_owned()]);
    let pose_models = [
        "sdxl",
        "realvisxl",
        "realvisxl_lightning",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ];

    for model_id in [
        "sdxl",
        "realvisxl",
        "realvisxl_lightning",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
        "instantid_realvisxl",
    ] {
        let entry = builtin_models_entry(model_id);
        for os in ["macos", "windows", "linux"] {
            let mut platform_entry = entry.clone();
            retain_downloads_for_os(&mut platform_entry, os);
            let co_requisites = model_co_requisite_downloads(&platform_entry);
            let ids_for = |soft: bool| -> BTreeSet<String> {
                co_requisites
                    .iter()
                    .filter(|download| {
                        (download.get("required").and_then(Value::as_str) == Some("soft")) == soft
                    })
                    .filter_map(|download| {
                        download
                            .get("componentId")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .collect()
            };
            let expected_hard = if os == "macos" {
                BTreeSet::new()
            } else {
                hard_want.clone()
            };
            assert_eq!(
                ids_for(false),
                expected_hard,
                "{model_id} on {os}: only Candle must gate on the three hard CLIP/VAE components",
            );
            let expected_soft = if pose_models.contains(&model_id) {
                soft_want.clone()
            } else {
                BTreeSet::new()
            };
            assert_eq!(
                ids_for(true),
                expected_soft,
                "{model_id} on {os}: OpenPose must be soft, cross-platform, and limited to the five accepted pose backbones",
            );

            if pose_models.contains(&model_id) {
                let data = tempfile::tempdir().unwrap();
                for download in platform_entry
                    .get("downloads")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|download| {
                        download.get("required").and_then(Value::as_str) != Some("soft")
                    })
                {
                    seed_manifest_download(data.path(), download);
                }
                let state = install_state_for(
                    model_download_context(&platform_entry).unwrap(),
                    &platform_entry,
                    data.path(),
                );
                assert!(
                    state.installed,
                    "{model_id} on {os}: missing soft OpenPose must not block ordinary generation",
                );
                assert!(
                    state.update_available,
                    "{model_id} on {os}: missing soft OpenPose must remain installable/repairable",
                );
                assert!(
                    !state.cache_incomplete && state.missing_required_files.is_empty(),
                    "{model_id} on {os}: a missing soft component is not a torn hard install; missing={:?}",
                    state.missing_required_files,
                );
            }
        }
    }
}

/// The live SCAIL-2 manifest must give off-Mac users all three shared package variants and the
/// catalog's per-tier install badge must enforce both the provider-required six-file layout and
/// the exact precision marker. This binds product advertisement, Model Manager filtering, and
/// worker loadability without duplicating weights or enabling a mixed package. The product
/// description must also advertise the reviewed Candle q4/q8/bf16 routes rather than the
/// pre-admission fail-closed wording.
#[test]
fn scail2_shared_variants_are_installable_off_macos_and_advertise_admitted_candle_tiers() {
    let _env = isolate_hf_cache();
    let data = tempfile::tempdir().unwrap();
    let original = builtin_models_entry("scail2_14b");

    for os in ["windows", "linux"] {
        let mut model = original.clone();
        retain_downloads_for_os(&mut model, os);
        let downloads = model["downloads"].as_array().unwrap();
        assert_eq!(
            downloads.len(),
            3,
            "{os} must expose all three package tiers"
        );
        let variants: std::collections::BTreeMap<_, _> = downloads
            .iter()
            .map(|download| {
                (
                    download["variant"].as_str().unwrap(),
                    download["files"].clone(),
                )
            })
            .collect();
        assert_eq!(variants["q4"], json!(["q4/*"]));
        assert_eq!(variants["q8"], json!(["q8/*"]));
        assert_eq!(variants["bf16"], json!(["bf16/*"]));
        assert_eq!(model_download(&model).unwrap()["variant"], "q4");
    }

    let mut model = original;
    retain_downloads_for_os(&mut model, "windows");
    let download = model_download(&model).unwrap();
    let repo = download["repo"].as_str().unwrap();
    let revision = download["revision"].as_str().unwrap();
    let snapshot = huggingface_repo_cache_path(data.path(), repo)
        .unwrap()
        .join("snapshots")
        .join(revision);
    let q4 = snapshot.join("q4");
    let q8 = snapshot.join("q8");
    let bf16 = snapshot.join("bf16");

    let write_tier = |dir: &FsPath, marker: Value| {
        std::fs::create_dir_all(dir).unwrap();
        for file in sceneworks_core::mlx_tier_completeness::SCAIL2_TIER_FILES {
            std::fs::write(dir.join(file), b"").unwrap();
        }
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
    };
    let health = |states: &[ModelVariantState]| {
        states
            .iter()
            .map(|state| {
                (
                    state.variant.clone(),
                    (state.installed, state.cache_incomplete),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>()
    };

    std::fs::create_dir_all(&q4).unwrap();
    std::fs::write(q4.join("dit.safetensors"), b"").unwrap();
    let partial = install_state_for(model_download_context(&model).unwrap(), &model, data.path());
    assert!(!partial.installed);
    assert!(partial.cache_incomplete);
    assert!(
        partial
            .missing_required_files
            .iter()
            .any(|file| file.contains("q4") && file.contains("incomplete")),
        "got {:?}",
        partial.missing_required_files
    );

    write_tier(
        &q4,
        json!({ "quantization": { "bits": 4, "group_size": 64 } }),
    );
    let states = model_variant_states(&model, data.path());
    assert_eq!(
        health(&states),
        std::collections::BTreeMap::from([
            ("bf16".to_owned(), (false, false)),
            ("q4".to_owned(), (true, false)),
            ("q8".to_owned(), (false, false)),
        ]),
        "one complete variant must not make never-downloaded siblings look installed or torn"
    );

    // Every q8 filename exists, but the q4 marker was mixed into the q8 directory. The coarse
    // glob sees files; the shared completeness predicate must still keep this tier disabled and
    // repairable while leaving the independently complete q4 installed.
    write_tier(
        &q8,
        json!({ "quantization": { "bits": 4, "group_size": 64 } }),
    );
    let states = model_variant_states(&model, data.path());
    assert_eq!(
        health(&states),
        std::collections::BTreeMap::from([
            ("bf16".to_owned(), (false, false)),
            ("q4".to_owned(), (true, false)),
            ("q8".to_owned(), (false, true)),
        ])
    );
    let mixed = install_state_for(model_download_context(&model).unwrap(), &model, data.path());
    assert!(mixed.installed, "the valid q4 sibling remains installed");
    assert!(
        mixed.cache_incomplete,
        "the mixed q8 tier remains repairable"
    );

    write_tier(
        &q8,
        json!({ "quantization": { "bits": 8, "group_size": 64 } }),
    );
    write_tier(&bf16, json!({}));
    let states = model_variant_states(&model, data.path());
    assert!(states.iter().all(|state| state.installed));
    assert!(states.iter().all(|state| !state.cache_incomplete));

    std::fs::remove_file(q4.join("t5_encoder.safetensors")).unwrap();
    let states = model_variant_states(&model, data.path());
    assert_eq!(
        health(&states),
        std::collections::BTreeMap::from([
            ("bf16".to_owned(), (true, false)),
            ("q4".to_owned(), (false, true)),
            ("q8".to_owned(), (true, false)),
        ]),
        "tearing q4 must not alter the independently installed q8/bf16 states"
    );

    let description = model["ui"]["description"].as_str().unwrap();
    assert!(description.contains("Candle on NVIDIA Windows/Linux"));
    assert!(description.contains("ordered reference character images"));
    assert!(description
        .contains("admit the same installable q4, q8, and bf16 package variants when installed"));
    assert!(
        !description.contains("execution remains fail-closed"),
        "the pre-admission product wording must not contradict the reviewed Candle routes"
    );
}
