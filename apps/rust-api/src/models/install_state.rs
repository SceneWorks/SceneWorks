use super::*;

// sc-4205 (F-API-12): the per-model install/cache state, formerly threaded through a
// 5-tuple that was easy to mis-order. Named fields make the catalog loop legible.
pub(super) struct ModelCatalogEntryState {
    pub(super) downloadable: bool,
    pub(super) installed_path: Option<String>,
    pub(super) installed: bool,
    pub(super) cache_incomplete: bool,
    pub(super) missing_required_files: Vec<String>,
    pub(super) update_available: bool,
    // Reuse the existing plan/source probe for the product projection, never resolve it twice.
    pub(super) imported_source_present: Option<bool>,
}

#[derive(Debug, PartialEq)]
pub(super) struct ReceiptFileSet {
    pub(super) files: Vec<String>,
    pub(super) revision: Option<String>,
}

pub(super) fn receipt_entries(managed_path: &FsPath) -> Vec<Value> {
    let Ok(bytes) = std::fs::read(managed_path.join(".sceneworks-download-complete.json")) else {
        return Vec::new();
    };
    let Ok(receipt) = serde_json::from_slice::<Value>(&bytes) else {
        return Vec::new();
    };
    receipt
        .get("receipts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![receipt])
}

pub(super) fn receipt_file_sets(
    managed_path: &FsPath,
    repo: &str,
    model_id: Option<&str>,
) -> Vec<ReceiptFileSet> {
    receipt_entries(managed_path)
        .into_iter()
        .filter_map(|entry| {
            if entry.get("repo").and_then(Value::as_str) != Some(repo) {
                return None;
            }
            // Shared repos back multiple catalog cards. A model-specific receipt must protect only
            // the card that produced it; receipts predating modelId remain generic for compatibility.
            if let (Some(expected), Some(actual)) =
                (model_id, entry.get("modelId").and_then(Value::as_str))
            {
                if actual != expected {
                    return None;
                }
            }
            let files = entry
                .get("resolvedFiles")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let revision = entry
                .get("snapshotRevision")
                .and_then(Value::as_str)
                .map(str::to_owned);
            (!files.is_empty()).then_some(ReceiptFileSet { files, revision })
        })
        .collect()
}

/// Whether a snapshot the receipt's files resolve into is an actually LOADABLE install (not merely a
/// set of files that exist). A backfill (sc-13076) records whatever was on disk, so an interrupted
/// download left a torn tier — its `model_index.json` + a stray config present, but the
/// transformer/vae weights missing — whose recorded files all exist yet cannot load. When the
/// receipt/tier files form a diffusers tier subdir (`["<tier>/*"]`, or a receipt whose files share one
/// leading dir), require that subdir to pass the same per-component weight check the cache-health path
/// uses. A non-diffusers set (no `<tier>/model_index.json`, or a flat single-variant filter) keeps the
/// prior file-existence contract.
///
/// `family_complete` is the model's shared per-tier predicate ([`no_model_index_family_predicate`]) for
/// a no-`model_index` MLX turnkey, or `None`. Without it this check was diffusers-only, so a FLAT tier
/// (SenseNova/SANA/Boogu/Anima) passed unconditionally: a backfill could mint a "complete" receipt for a
/// tier whose tokenizer never landed, and that receipt then kept the model reading installed through the
/// usable-stale path even once the cache-health lane had correctly demoted it (sc-14432).
pub(super) fn snapshot_tier_is_loadable(
    snapshot: &FsPath,
    files: &[String],
    family_complete: Option<fn(&FsPath) -> bool>,
) -> bool {
    match tier_subdir_name(files) {
        Some(tier) => {
            let tier_dir = snapshot.join(&tier);
            let diffusers_ok = !path_is_readable_file(&tier_dir.join("model_index.json"))
                || diffusers_snapshot_health(&tier_dir).installed;
            match family_complete {
                Some(complete) => diffusers_ok && complete(&tier_dir),
                None => diffusers_ok,
            }
        }
        None => true,
    }
}

pub(super) fn receipt_files_present(
    data_dir: &FsPath,
    repo: &str,
    receipt: &ReceiptFileSet,
    family_complete: Option<fn(&FsPath) -> bool>,
) -> bool {
    !receipt.files.is_empty()
        && huggingface_repo_cache_path(data_dir, repo)
            .map(|root| {
                let matches = crate::huggingface_snapshot_dirs(&root)
                    .into_iter()
                    .filter(|snapshot| {
                        receipt
                            .files
                            .iter()
                            .all(|file| snapshot.join(file).is_file())
                    })
                    // A torn tier's recorded files all exist but the install can't load — it must not
                    // count as a "usable stale" install that keeps the model falsely installed.
                    .filter(|snapshot| {
                        snapshot_tier_is_loadable(snapshot, &receipt.files, family_complete)
                    })
                    // A backfill records whatever it FOUND, so a receipt can name a shard index plus
                    // only the shards that happened to land. Every recorded index must still resolve
                    // to a complete shard set on disk, or this receipt is laundering an interrupted
                    // download into "installed" (sc-20526). Stat-only: the index paths are already in
                    // the receipt, so this costs no directory walk.
                    .filter(|snapshot| listed_shard_indexes_are_complete(snapshot, &receipt.files))
                    .collect::<Vec<_>>();
                receipt
                    .revision
                    .as_deref()
                    .map_or(matches.len() == 1, |revision| {
                        matches.iter().any(|snapshot| {
                            snapshot.file_name().and_then(|v| v.to_str()) == Some(revision)
                        })
                    })
            })
            .unwrap_or(false)
}

// Catalog entries can share a Hugging Face repo. The catalog sweep probes entries concurrently,
// so serialize only the final receipt recheck/write (not the expensive snapshot walk) to prevent
// two first-seen entries from truncating the same backfill file at once.
pub(super) static RECEIPT_BACKFILL_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// [`no_model_index_family_predicate`] for a whole manifest `model` entry — the receipt lanes carry the
/// model, not a pre-split family/id pair.
pub(super) fn model_family_tier_predicate(model: &Value) -> Option<fn(&FsPath) -> bool> {
    no_model_index_family_predicate(
        model
            .get("family")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        model.get("id").and_then(Value::as_str).unwrap_or_default(),
    )
}

/// Whether a receipt entry is PROVABLY torn: every cached snapshot that holds one of the shard
/// indexes the entry recorded is missing at least one shard that index names, and no snapshot holds
/// a complete set (sc-20526).
///
/// Deliberately positive-evidence only. A receipt whose repo cache is gone, whose files simply are
/// not where this box expects them, or that records no shard index at all is left ALONE — read-side
/// validation already refuses to count such a receipt as installed, and silently deleting a receipt
/// on ambiguous evidence would destroy the relocation proof an external-library install depends on.
pub(super) fn receipt_entry_is_provably_torn(entry: &Value, data_dir: &FsPath) -> bool {
    if entry.get("backfilled").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    let Some(repo) = entry.get("repo").and_then(Value::as_str) else {
        return false;
    };
    let indexes = entry
        .get("resolvedFiles")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|file| file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX))
        .collect::<Vec<_>>();
    if indexes.is_empty() {
        return false;
    }
    let Some(root) = huggingface_repo_cache_path(data_dir, repo) else {
        return false;
    };
    let mut saw_torn = false;
    for snapshot in crate::huggingface_snapshot_dirs(&root) {
        let present = indexes
            .iter()
            .filter(|file| path_is_readable_file(&snapshot.join(file)))
            .collect::<Vec<_>>();
        if present.is_empty() {
            continue;
        }
        // One complete snapshot is enough: cached revisions are alternatives, and a re-download
        // lands beside the torn revision it replaces rather than deleting it.
        if present.len() == indexes.len()
            && listed_shard_indexes_are_complete(
                &snapshot,
                &indexes
                    .iter()
                    .map(|file| (*file).to_owned())
                    .collect::<Vec<_>>(),
            )
        {
            return false;
        }
        saw_torn = true;
    }
    saw_torn
}

/// Repair receipts that a pre-sc-20526 backfill minted over a partially downloaded tier.
///
/// The offending shape is `backfilled: true` with a shard index whose `weight_map` names files that
/// were never downloaded — the receipt recorded the partial on-disk set as if it were complete, and
/// nothing invalidated it, so affected users needed a manual delete. Dropping the entry here lets the
/// next scan re-backfill honestly (the backfill writer early-outs while ANY receipt for the model
/// survives) and heals every other receipt consumer, since they all read this one file.
///
/// Cost: one small JSON read per managed model directory plus a stat per recorded index and shard.
/// It runs on the catalog sweep, never on the per-job hot path.
pub(super) fn repair_torn_backfilled_receipts(managed_path: &FsPath, data_dir: &FsPath) {
    let entries = receipt_entries(managed_path);
    if entries.is_empty()
        || !entries
            .iter()
            .any(|entry| receipt_entry_is_provably_torn(entry, data_dir))
    {
        return;
    }
    let _write_guard = RECEIPT_BACKFILL_WRITE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Ok(_receipt_lock) = sceneworks_core::download_receipt::lock(managed_path) else {
        return;
    };
    // Re-read under the lock: a concurrent backfill may have rewritten the file.
    let kept = receipt_entries(managed_path)
        .into_iter()
        .filter(|entry| !receipt_entry_is_provably_torn(entry, data_dir))
        .collect::<Vec<_>>();
    let receipt_path = managed_path.join(".sceneworks-download-complete.json");
    let Some(first) = kept.first().cloned() else {
        // Every entry was a false-complete: the model is not installed, so the receipt must go
        // rather than linger as a bare marker `model_is_installed` would still honour.
        let _ = std::fs::remove_file(&receipt_path);
        return;
    };
    let mut receipt = first;
    if let Some(object) = receipt.as_object_mut() {
        object.insert("receipts".to_owned(), Value::Array(kept));
    }
    let _ = sceneworks_core::download_receipt::write(&receipt_path, &receipt);
}

/// Heal legacy install identity before the catalog and local-cache eligibility read it.
pub(super) fn repair_missing_snapshot_revisions(
    managed: &FsPath,
    model: &Value,
    data_dir: &FsPath,
) {
    use sceneworks_core::download_receipt;
    let marker = managed.join(".sceneworks-download-complete.json");
    if !marker.is_file() {
        return;
    }
    let Ok(_lock) = download_receipt::lock(managed) else {
        return;
    };
    let Ok(bytes) = std::fs::read(&marker) else {
        return;
    };
    let Ok(mut receipt) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    let family_complete = model_family_tier_predicate(model);
    let repair = |entry: &mut Value| {
        if entry
            .get("modelId")
            .and_then(Value::as_str)
            .is_some_and(|id| Some(id) != model.get("id").and_then(Value::as_str))
        {
            return false;
        }
        let Some(repo) = entry.get("repo").and_then(Value::as_str) else {
            return false;
        };
        let Some(root) = huggingface_repo_cache_path(data_dir, repo) else {
            return false;
        };
        let Some(revision) = download_receipt::recover_revision(entry, &root) else {
            return false;
        };
        let Ok((_, snapshot)) = sceneworks_core::hf_home::model_source_library(data_dir)
            .discover_snapshot(repo, Some(&revision))
        else {
            return false;
        };
        let files = string_array_field(entry, "resolvedFiles");
        if !snapshot_tier_is_loadable(&snapshot, &files, family_complete)
            || !listed_shard_indexes_are_complete(&snapshot, &files)
        {
            return false;
        }
        entry["snapshotRevision"] = Value::String(revision);
        true
    };
    // The newest receipt is mirrored at the top level. Repair both representations without
    // rebuilding the envelope or dropping unrelated variants and fields.
    let mut changed = repair(&mut receipt);
    if let Some(entries) = receipt.get_mut("receipts").and_then(Value::as_array_mut) {
        for entry in entries {
            changed |= repair(entry);
        }
    }
    if changed {
        if let Err(error) = download_receipt::write(&marker, &receipt) {
            tracing::warn!(%error, path = %marker.display(), "could not repair missing snapshot revisions");
        }
    }
}

pub(super) fn backfill_current_receipt(
    managed_path: &FsPath,
    model: &Value,
    context: &DownloadContext,
    data_dir: &FsPath,
) {
    let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
    if !receipt_file_sets(managed_path, &context.repo, Some(model_id)).is_empty() {
        return;
    }
    let family_complete = model_family_tier_predicate(model);
    let receipts = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .filter_map(|entry| {
            let repo = entry.get("repo")?.as_str()?;
            let files = string_array_field(entry, "files");
            let root = huggingface_repo_cache_path(data_dir, repo)?;
            let snapshot = crate::huggingface_snapshot_dirs(&root).into_iter().find(|snapshot| {
                files.iter().all(|pattern| snapshot_contains_pattern(snapshot, pattern))
            })?;
            // Never manufacture a receipt for a torn tier: a `<tier>/*` glob matches as soon as one
            // metadata file exists, so backfilling it would record a "complete" install that cannot
            // load. Require the tier to actually hold its weights before preserving it (sc-13076), and
            // — for a no-`model_index` turnkey, where "holds its weights" is family-specific — to pass
            // the shared per-tier predicate too (sc-14432).
            if !snapshot_tier_is_loadable(&snapshot, &files, family_complete) {
                return None;
            }
            let resolved = snapshot_files(&snapshot).into_iter()
                .filter(|file| allow_pattern_matches(file, &files)).collect::<Vec<_>>();
            // Record only a file set that SATISFIES what the repo declares, never merely what was
            // found (sc-20526). A `<tier>/*` glob and a whole-repo fetch declare no per-file claim,
            // so the sharded-safetensors indexes IN the resolved set are the workable ground truth:
            // every shard their `weight_map` names must be on disk. Otherwise this would mint a
            // "complete" receipt for a tier missing ~4 GB of its text encoder (lens_turbo bf16) and
            // that receipt would then keep the model reading installed through the usable-stale path.
            if !listed_shard_indexes_are_complete(&snapshot, &resolved) {
                return None;
            }
            // Record WHICH snapshot these files were read from (sc-19712 F-5). A revision-less
            // receipt makes its whole repository unserveable from the resolved cache — there is no
            // pair to compare coverage against — so a backfilled install silently dropped out of
            // the local tier while still being promotable, occupying cache bytes it could never be
            // served from. The revision is not unknowable here: the snapshot directory selected
            // just above IS `.../snapshots/<revision>`, and `resolvedFiles` were read from it. Only
            // a genuinely immutable 40-hex name is recorded; anything else is omitted rather than
            // written as a revision the source tier could not re-resolve. The worker's
            // `establish_receipt_tree_stamp` already patches receipts this same way.
            let snapshot_revision = snapshot
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| {
                    sceneworks_core::model_artifacts::validate_immutable_revision(name).is_ok()
                })
                .map(|name| Value::String(name.to_owned()))
                .unwrap_or(Value::Null);
            (!resolved.is_empty()).then(|| json!({
                "schemaVersion": 2, "repo": repo,
                "modelId": model.get("id").cloned().unwrap_or(Value::Null),
                "variant": entry.get("variant").cloned().unwrap_or_else(|| Value::String("default".to_owned())),
                "manifestFiles": files, "resolvedFiles": resolved, "backfilled": true,
                "snapshotRevision": snapshot_revision,
            }))
        }).collect::<Vec<_>>();
    if receipts.is_empty() {
        return;
    }
    if std::fs::create_dir_all(managed_path).is_err() {
        return;
    }
    let _write_guard = RECEIPT_BACKFILL_WRITE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Ok(_receipt_lock) = sceneworks_core::download_receipt::lock(managed_path) else {
        return;
    };
    if !receipt_file_sets(managed_path, &context.repo, Some(model_id)).is_empty() {
        return;
    }
    let mut merged = receipt_entries(managed_path);
    for new_entry in receipts {
        let identity = (
            new_entry.get("repo").and_then(Value::as_str),
            new_entry.get("modelId").and_then(Value::as_str),
            new_entry.get("variant").and_then(Value::as_str),
        );
        if let Some(existing) = merged.iter_mut().find(|existing| {
            (
                existing.get("repo").and_then(Value::as_str),
                existing.get("modelId").and_then(Value::as_str),
                existing.get("variant").and_then(Value::as_str),
            ) == identity
        }) {
            *existing = new_entry;
        } else {
            merged.push(new_entry);
        }
    }
    let mut receipt = merged[0].clone();
    receipt
        .as_object_mut()
        .unwrap()
        .insert("receipts".to_owned(), Value::Array(merged));
    let _ = sceneworks_core::download_receipt::write(
        &managed_path.join(".sceneworks-download-complete.json"),
        &receipt,
    );
}

// Resolve a model's install/cache state from its (optional) download source. A
// downloadable model checks the HF cache + the SceneWorks-managed dir; a non-download
// model (a local manifest entry) checks its declared installed path; otherwise it's
// simply absent.
pub(super) fn is_flux3_runtime_model(model_id: &str) -> bool {
    matches!(model_id, "flux3_klein" | "flux3_klein_base")
}

pub(super) fn flux3_selected_revision<'a>(
    download: Option<&'a Value>,
    repo: &str,
) -> Option<&'a str> {
    let download = download?;
    if download.get("repo").and_then(Value::as_str) != Some(repo) {
        return None;
    }
    let revision = download.get("revision")?.as_str()?;
    sceneworks_core::model_artifacts::validate_immutable_revision(revision)
        .is_ok()
        .then_some(revision)
}

pub(super) fn flux3_selected_snapshot(
    data_dir: &FsPath,
    repo: &str,
    download: Option<&Value>,
    model_id: &str,
) -> Option<PathBuf> {
    let revision = flux3_selected_revision(download, repo)?;
    sceneworks_worker::flux3_receipted_snapshot(data_dir, repo, revision, model_id)
}

// Runtime readiness is separate from whole-repository download completion. A readable DiT is
// not a usable FLUX 3 bundle when its bundled encoder or an indexed shard is absent. Other
// families retain their existing cache policy; this card selects only its exact repo/revision.
pub(super) fn model_cache_health_for_download(
    model_id: &str,
    repo: &str,
    download: Option<&Value>,
    cache_path: Option<&FsPath>,
    files: &[String],
    data_dir: &FsPath,
) -> Option<HuggingFaceCacheHealth> {
    if !is_flux3_runtime_model(model_id) {
        return cache_path.map(|root| huggingface_cache_health(root, files));
    }
    let Some(revision) = flux3_selected_revision(download, repo) else {
        return Some(HuggingFaceCacheHealth::missing(vec![
            "FLUX 3 requires its selected immutable repository revision".to_owned(),
        ]));
    };
    let Some(snapshot) = flux3_selected_snapshot(data_dir, repo, download, model_id) else {
        return Some(HuggingFaceCacheHealth::missing(vec![
            "FLUX 3 install verification required: no trusted exact-revision install receipt. Install or verify this pinned package through the downloader; complete cache files alone are not selectable.".to_owned(),
        ]));
    };
    if !snapshot.is_dir() {
        return Some(HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: vec![format!("snapshots/{revision}")],
        });
    }
    if let Some(tier @ ("fp8r" | "nvfp4")) = download
        .and_then(|row| row.get("variant"))
        .and_then(Value::as_str)
    {
        if !snapshot.join("variants").join(tier).exists() {
            return Some(HuggingFaceCacheHealth {
                installed: false,
                incomplete: false,
                missing_files: vec![format!("variants/{tier}")],
            });
        }
    }
    Some(
        match sceneworks_worker::flux3_runtime_tier_bundle_check(
            &snapshot,
            download
                .and_then(|d| d.get("variant"))
                .and_then(Value::as_str)
                .unwrap_or("bf16"),
            model_id,
        ) {
            Ok(()) => HuggingFaceCacheHealth::installed(),
            Err(error) => HuggingFaceCacheHealth::missing(vec![error]),
        },
    )
}

pub(super) fn install_state_for(
    download_context: Option<DownloadContext>,
    model: &Value,
    data_dir: &FsPath,
) -> ModelCatalogEntryState {
    if model.get("family").and_then(Value::as_str) == Some("minimax-h3")
        && model
            .as_object()
            .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
            .is_some()
    {
        let entry = model.as_object().expect("imported entry");
        let checkpoint_id = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
            .expect("imported plan");
        let resolved = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir)
            .resolve(checkpoint_id);
        let mut missing = Vec::new();
        let mut bound_config = None;
        match resolved {
            Ok(resolved) => {
                if let Err(reason) =
                    sceneworks_core::minimax_h3_import::verify_entry_assembly(entry, &resolved.plan)
                {
                    missing.push(reason);
                }
                bound_config =
                    sceneworks_core::minimax_h3_import::bound_config(&resolved.plan.context)
                        .ok()
                        .flatten()
                        .map(|config| config.0);
            }
            Err(error) => missing.push(error.to_string()),
        }
        let source_present = missing.is_empty();
        let rows = model_co_requisite_downloads(model);
        if let Err(reason) =
            sceneworks_core::minimax_h3_import::selected_import_component_tier(entry)
        {
            missing.push(reason);
        }
        if rows.is_empty() {
            missing.push(
                "The import has no compatible component closure; re-import with the selected base"
                    .to_owned(),
            );
        }
        let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
            sceneworks_core::hf_home::model_source_library(data_dir),
        );
        for row in rows {
            if imported_component_presence(data_dir, &row)
                != sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence::Installed {
                missing.push(format!("{}: {}", row.get("repo").and_then(Value::as_str).unwrap_or("component"),
                    string_array_field(&row, "files").join(", ")));
            }
            if let Some(config) = &bound_config {
                if let (Some(repo), Some(revision)) = (
                    row.get("repo").and_then(Value::as_str),
                    row.get("revision").and_then(Value::as_str),
                ) {
                    if let Ok((_, snapshot)) =
                        resolver.discover_source_snapshot(repo, Some(revision))
                    {
                        if let Err(reason) = sceneworks_core::minimax_h3_import::import_component_configuration_compatible(&snapshot, &row, config) {
                            missing.push(reason);
                        }
                    }
                }
            }
        }
        return ModelCatalogEntryState {
            downloadable: false,
            installed_path: model_manifest_installed_path(model, data_dir)
                .map(|path| path.display().to_string()),
            installed: missing.is_empty(),
            cache_incomplete: !missing.is_empty(),
            missing_required_files: missing,
            update_available: false,
            imported_source_present: Some(source_present),
        };
    }
    if let Some(download_context) = download_context {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let selected_download = model_download(model);
        let managed_path = data_dir
            .join("models")
            .join(safe_download_dir(&download_context.repo));
        let cache_path = huggingface_repo_cache_path(data_dir, &download_context.repo);
        // Quant-matrix models (sc-8506/8508): the top-level install state aggregates across ALL
        // selectable tiers, not just the default one. A model that offers bf16/q8/q4 counts as
        // installed when ANY tier is fully present, and is only "incomplete" when a tier is genuinely
        // torn (partially downloaded) AND no complete tier exists. Installing a single valid tier —
        // even a non-default one — must never surface as an incomplete/repairable cache (sc-9907),
        // because that rendered a false "Cached files are incomplete" warning + Fix button on a
        // perfectly good install. Single-variant models keep the default-tier contract below.
        let mut tier_dependencies_missing = false;
        let (cache_installed, cache_incomplete, mut missing_required_files) =
            if model_has_variant_matrix(model) {
                let variants = model_variant_states(model, data_dir);
                tier_dependencies_missing =
                    variants.iter().any(|variant| variant.dependencies_missing);
                let any_installed = variants.iter().any(|variant| variant.installed);
                // A TORN tier — some of its files present but not all — is a genuine repair candidate:
                // it will fail to load, and re-downloading that tier fixes it. A never-fetched tier is
                // `missing` (`cache_incomplete == false` — nothing on disk to be torn), NOT torn, so it
                // never enters this find. That distinction is what makes it safe to surface repair even
                // when a complete sibling exists: sc-9907 (a clean single-tier install must not read
                // "incomplete + Fix" just because its OTHER tiers were never downloaded) is preserved by
                // `torn` excluding missing tiers, WITHOUT also suppressing a genuinely half-downloaded
                // sibling (sc-14431 — the old `!any_installed &&` guard hid a torn q8 behind a complete
                // q4, so chroma/sdxl/qwen/wan/lens all reported `repairAvailable: false` over a torn tier).
                let torn = variants
                    .iter()
                    .find(|variant| variant.cache_incomplete && !variant.installed);
                let incomplete = torn.is_some();
                // Report the torn tier's missing files so the model-level repair knows what to re-fetch,
                // even though a sibling tier is complete (`any_installed`).
                let missing = torn
                    .map(|variant| variant.missing_required_files.clone())
                    .unwrap_or_default();
                (any_installed, incomplete, missing)
            } else {
                let cache_health = model_cache_health_for_download(
                    model_id,
                    &download_context.repo,
                    selected_download.as_ref(),
                    cache_path.as_deref(),
                    &download_context.files,
                    data_dir,
                );
                let installed = cache_health.as_ref().is_some_and(|health| health.installed);
                let incomplete = cache_health
                    .as_ref()
                    .is_some_and(|health| health.incomplete);
                let missing = cache_health
                    .as_ref()
                    .map(|health| health.missing_files.clone())
                    .unwrap_or_default();
                // sc-13513: a single-variant no-`model_index` turnkey (Boogu's `base/` default download,
                // whose component globs match stray weights) reads coarse-installed while actually torn.
                // Downgrade so the TOP-LEVEL badge agrees with the per-variant state and the worker's
                // loader gate — otherwise `installState:"installed"`/`repairAvailable:false` renders a
                // false green over an install that then fails to load. SANA (variant-matrix) is already
                // covered by the aggregate branch above; Anima's exact-path filter catches torn coarsely.
                let family = model
                    .get("family")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if installed
                    && no_model_index_tier_is_torn(
                        family,
                        model.get("id").and_then(Value::as_str).unwrap_or_default(),
                        &download_context.files,
                        cache_path.as_deref(),
                        &managed_path,
                    )
                {
                    (false, true, vec![torn_tier_marker(&download_context.files)])
                } else {
                    (installed, incomplete, missing)
                }
            };
        // A quant-matrix model's top-level state is the aggregate of its tier-aware variant states
        // computed above (cache_installed = "any tier installed"). The repo-level managed marker must
        // NOT independently mark it installed (sc-9909): a stale .sceneworks-download-complete.json
        // left by an empty download would otherwise read the whole model as installed while every tier
        // reads missing. Single-variant models keep the repo-level managed contract.
        // sc-20526: drop any receipt a pre-fix backfill minted over a partially downloaded tier
        // BEFORE it is read, so an affected install self-heals on the next scan instead of needing a
        // manual delete — and so the backfill below can re-record the tier honestly.
        if !is_flux3_runtime_model(model_id) {
            repair_torn_backfilled_receipts(&managed_path, data_dir);
            repair_missing_snapshot_revisions(&managed_path, model, data_dir);
        }
        let receipt_file_sets = receipt_file_sets(
            &managed_path,
            &download_context.repo,
            model.get("id").and_then(Value::as_str),
        );
        let managed_installed = if is_flux3_runtime_model(model_id) {
            false // The shared receipt-resolved cache health already covers managed snapshots.
        } else {
            !model_has_variant_matrix(model)
                && receipt_file_sets.is_empty()
                && model_is_installed(&managed_path)
        };
        // Usable native FLUX 3 components do not prove every extra whole-repo file downloaded.
        // Only the actual downloader may publish that completion receipt for this source.
        if cache_installed && !is_flux3_runtime_model(model_id) {
            backfill_current_receipt(&managed_path, model, &download_context, data_dir);
        }
        let stale_files_present = !is_flux3_runtime_model(model_id)
            && !cache_installed
            && receipt_file_sets.iter().any(|receipt| {
                receipt_files_present(
                    data_dir,
                    &download_context.repo,
                    receipt,
                    model_family_tier_predicate(model),
                )
            });
        let breaking_update = model
            .get("breaking")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || model
                .get("downloads")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|entry| {
                    !is_co_requisite_download(entry)
                        && entry.get("repo").and_then(Value::as_str)
                            == Some(download_context.repo.as_str())
                        && string_array_field(entry, "files") == download_context.files
                        && entry
                            .get("breaking")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                });
        // A receipt proves the primary files exist, not that their dependencies still do.
        // Do not let it override missing dependencies. A partial nonbreaking primary update
        // may still use its complete older receipt.
        let usable_stale = stale_files_present && !breaking_update && !tier_dependencies_missing;
        let primary_installed = managed_installed || cache_installed || usable_stale;
        let installed_path = if is_flux3_runtime_model(model_id) {
            if cache_installed || (cache_incomplete && !managed_installed) {
                flux3_selected_snapshot(
                    data_dir,
                    &download_context.repo,
                    selected_download.as_ref(),
                    model_id,
                )
            } else if managed_installed {
                Some(managed_path.clone())
            } else {
                None
            }
        } else if cache_installed || cache_incomplete || usable_stale {
            cache_path.clone()
        } else {
            Some(managed_path)
        };
        // Co-requisites (sc-9696): the entry counts as installed only when the primary AND every
        // co-requisite dependency (e.g. the PiD decoder's shared gemma-2-2b-it caption encoder) are
        // cached. Gating on this keeps a feature that silently no-ops without its dependency (PiD →
        // native VAE) from advertising as ready, and a present primary with a missing/partial
        // co-requisite surfaces as a repairable partial install (cache_incomplete → repairAvailable),
        // whose repair re-runs the download job that now fetches the co-requisite too.
        let mut hard_co_requisites_installed = true;
        let mut soft_co_requisite_update = false;
        let mut co_requisite_incomplete = false;
        // Tier-scoped co-requisites (sc-14980) are gated like the tiers themselves: this state is a
        // model-level AGGREGATE that counts a quant-matrix model installed when ANY tier is fully
        // present, so requiring every tier's shared text encoder would report a complete, working q4
        // install as incomplete forever. Satisfy them when at least one tier's set is fully cached;
        // tier-agnostic co-requisites (every other family) keep the strict all-must-be-present rule.
        let tier_scoped: Vec<Value> = model_co_requisite_downloads(model)
            .into_iter()
            .filter(|download| co_requisite_variant(download).is_some())
            .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft"))
            .collect();
        // Current matrix variants already check their matching dependency sets with the primary.
        // Retain the dependency floor when a receipt is supplying an older primary instead.
        if (!model_has_variant_matrix(model) || usable_stale) && !tier_scoped.is_empty() {
            let tiers: std::collections::BTreeSet<String> = tier_scoped
                .iter()
                .filter_map(co_requisite_variant)
                .collect();
            let any_tier_complete = tiers.iter().any(|tier| {
                tier_scoped
                    .iter()
                    .filter(|download| co_requisite_variant(download).as_deref() == Some(tier))
                    .all(|download| {
                        co_requisite_cache_health(data_dir, download)
                            .is_some_and(|health| health.installed)
                    })
            });
            if !any_tier_complete {
                hard_co_requisites_installed = false;
                missing_required_files.extend(
                    tier_scoped
                        .iter()
                        .filter_map(|download| {
                            download
                                .get("repo")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .collect::<std::collections::BTreeSet<_>>(),
                );
            }
        }
        // A choice group (sc-22998, YuE2's decoder) is satisfied by ANY installed option; when none
        // is, a partially installed option (the one a repair completes), else the default, is what
        // the install reports missing.
        for co_requisite in co_requisite_rows_gating_install(
            model_co_requisite_downloads(model)
                .into_iter()
                .filter(|download| co_requisite_variant(download).is_none())
                .collect(),
            |download| co_requisite_presence(data_dir, download),
        ) {
            let Some(repo) = co_requisite.get("repo").and_then(Value::as_str) else {
                continue;
            };
            let health = co_requisite_cache_health(data_dir, &co_requisite);
            if health.as_ref().is_some_and(|health| health.installed) {
                continue;
            }
            let soft = co_requisite.get("required").and_then(Value::as_str) == Some("soft");
            if soft {
                soft_co_requisite_update = true;
            } else {
                hard_co_requisites_installed = false;
                co_requisite_incomplete |= health.as_ref().is_some_and(|health| health.incomplete);
            }
            match health
                .as_ref()
                .map(|health| health.missing_files.as_slice())
            {
                Some(missing) if !missing.is_empty() && !soft => missing_required_files
                    .extend(missing.iter().map(|file| format!("{repo}/{file}"))),
                _ if !soft => missing_required_files.push(repo.to_owned()),
                _ => {}
            }
        }
        ModelCatalogEntryState {
            downloadable: true,
            installed_path: installed_path.map(|path| path.display().to_string()),
            installed: primary_installed && hard_co_requisites_installed,
            cache_incomplete: cache_incomplete
                || (primary_installed && !hard_co_requisites_installed)
                || co_requisite_incomplete,
            missing_required_files,
            update_available: stale_files_present || soft_co_requisite_update,
            imported_source_present: None,
        }
    } else if let Some(installed_path) = model_manifest_installed_path(model, data_dir) {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: Some(installed_path.display().to_string()),
            installed: model_is_installed(&installed_path),
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
            imported_source_present: None,
        }
    } else {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: None,
            installed: false,
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
            imported_source_present: None,
        }
    }
}

/// Per co-requisite choice group, the options whose component is installed on this host
/// (`{"decoder": ["standard"]}`); `None` for a model with no choice group.
pub(super) fn installed_co_requisite_choices(model: &Value, data_dir: &FsPath) -> Option<Value> {
    use sceneworks_core::model_artifacts::artifact_selection::{
        co_requisite_choice, model_co_requisite_downloads,
    };
    let mut groups: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for download in model_co_requisite_downloads(model) {
        let Some(choice) = co_requisite_choice(&download) else {
            continue;
        };
        let options = groups.entry(choice.group).or_default();
        let installed =
            co_requisite_cache_health(data_dir, &download).is_some_and(|health| health.installed);
        if installed && !options.contains(&choice.option) {
            options.push(choice.option);
        }
    }
    (!groups.is_empty()).then(|| json!(groups))
}

/// [`co_requisite_cache_health`] as the three-way presence choice-group gating reads (sc-22998).
/// Exact-revision availability for MOSS request-selected components. A sibling cached snapshot
/// cannot make the selected package installed. Hash authentication remains the native load gate.
pub(crate) fn pinned_component_presence(
    data_dir: &FsPath,
    download: &Value,
) -> sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return CoRequisitePresence::Absent;
    };
    let Some(revision) = download.get("revision").and_then(Value::as_str) else {
        return CoRequisitePresence::Absent;
    };
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let Ok((_, snapshot)) = resolver.discover_source_snapshot(repo, Some(revision)) else {
        return CoRequisitePresence::Absent;
    };
    let files = string_array_field(download, "files");
    let health = huggingface_filtered_cache_health(&[snapshot], &files);
    if health.installed {
        CoRequisitePresence::Installed
    } else if health.incomplete {
        CoRequisitePresence::Incomplete
    } else {
        CoRequisitePresence::Absent
    }
}

pub(super) fn imported_component_presence(
    data_dir: &FsPath,
    download: &Value,
) -> sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let presence = if download.get("revision").and_then(Value::as_str).is_some() {
        pinned_component_presence(data_dir, download)
    } else {
        co_requisite_presence(data_dir, download)
    };
    if presence != CoRequisitePresence::Installed {
        return presence;
    }
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return CoRequisitePresence::Absent;
    };
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let complete = resolver
        .discover_source_snapshot(repo, download.get("revision").and_then(Value::as_str))
        .is_ok_and(|(_, snapshot)| {
            sceneworks_core::minimax_h3_import::import_component_files_complete(&snapshot, download)
        });
    if complete {
        CoRequisitePresence::Installed
    } else {
        CoRequisitePresence::Incomplete
    }
}

pub(super) fn co_requisite_presence(
    data_dir: &FsPath,
    download: &Value,
) -> sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    match co_requisite_cache_health(data_dir, download) {
        Some(health) if health.installed => CoRequisitePresence::Installed,
        Some(health) if health.incomplete => CoRequisitePresence::Incomplete,
        _ => CoRequisitePresence::Absent,
    }
}

/// Resolve install-state health for a hard/soft co-requisite, including exact cache-only aliases
/// for upstream repository renames. This deliberately does not mutate the cache while serving the
/// catalog. It lets an existing immutable legacy snapshot remain usable/installed; the worker's
/// load-time resolver then consumes the same validated plain source and hard-links the canonical
/// cache namespace from those local bytes when possible.
pub(super) fn co_requisite_cache_health(
    data_dir: &FsPath,
    download: &Value,
) -> Option<HuggingFaceCacheHealth> {
    match sceneworks_core::managed_file::ManagedFileIdentity::from_download(download) {
        Ok(Some(identity)) => {
            let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
                sceneworks_core::hf_home::model_source_library(data_dir),
            );
            let valid = resolver
                .discover_source_snapshot(&identity.repo, Some(&identity.revision))
                .map(|(_, snapshot)| snapshot.join(&identity.file))
                .is_ok_and(|path| identity.verify(&path).is_ok());
            return Some(if valid {
                HuggingFaceCacheHealth::installed()
            } else {
                HuggingFaceCacheHealth::missing(vec![identity.file])
            });
        }
        Err(_) => {
            return Some(HuggingFaceCacheHealth::missing(string_array_field(
                download, "files",
            )))
        }
        Ok(None) => {}
    }
    let repo = download.get("repo").and_then(Value::as_str)?;
    let files = string_array_field(download, "files");
    let exact_rename = download
        .get("revision")
        .and_then(Value::as_str)
        .and_then(|revision| {
            sceneworks_core::hf_repo_renames::exact_huggingface_repo_rename(repo, revision, &files)
        });
    let current = match exact_rename {
        Some(rename) => {
            exact_rename_side_cache_health(data_dir, rename.current_repo, rename, &files)
        }
        None => huggingface_repo_cache_path(data_dir, repo)
            .map(|path| huggingface_cache_health(&path, &files)),
    };
    if current.as_ref().is_some_and(|health| health.installed) {
        return current;
    }

    let exact_legacy = exact_rename.and_then(|rename| {
        exact_rename_side_cache_health(data_dir, rename.legacy_repo, rename, &files)
    });

    match (current, exact_legacy) {
        (_, Some(legacy)) if legacy.installed => Some(legacy),
        (Some(current), _) if current.incomplete => Some(current),
        (_, Some(legacy)) if legacy.incomplete => Some(legacy),
        (current, legacy) => current.or(legacy),
    }
}

pub(super) fn co_requisite_satisfied_by_exact_legacy_rename(
    data_dir: &FsPath,
    download: &Value,
) -> bool {
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return false;
    };
    let Some(revision) = download.get("revision").and_then(Value::as_str) else {
        return false;
    };
    let files = string_array_field(download, "files");
    let Some(rename) =
        sceneworks_core::hf_repo_renames::exact_huggingface_repo_rename(repo, revision, &files)
    else {
        return false;
    };
    exact_rename_side_cache_health(data_dir, rename.legacy_repo, rename, &files)
        .is_some_and(|health| health.installed)
}

/// Evaluate an exact legacy alias with the same canonical-source confinement used by the worker's
/// load-time relink. A readable symlink is not sufficient: if it escapes the legacy repository
/// cache, treating it as installed would suppress the canonical repair download even though the
/// worker must reject it.
pub(super) fn exact_rename_side_cache_health(
    data_dir: &FsPath,
    repo: &str,
    rename: &sceneworks_core::hf_repo_renames::HuggingFaceRepoRename,
    files: &[String],
) -> Option<HuggingFaceCacheHealth> {
    let repo_root = huggingface_repo_cache_path(data_dir, repo)?;
    if !huggingface_repo_cache_exists(&repo_root) {
        return Some(HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: Vec::new(),
        });
    }
    // Through the typed source-library seam rather than raw joins (sc-19711's artifact inventory
    // forbids the API constructing a model root directly; `crates/sceneworks-worker/src/model_jobs.rs`
    // is the one classified infrastructure surface allowed to). `repo_root` came from
    // `huggingface_repo_cache_path`, i.e. this same library, so the seam accepts it; `.root()` is the
    // very `huggingface_hub_cache_dir(data_dir)` this used to pass, unchanged.
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    let snapshot = library
        .snapshots_root_from_repository_root(&repo_root)
        .ok()?
        .join(rename.revision);
    if sceneworks_core::hf_repo_renames::validate_exact_huggingface_repo_rename_snapshot(
        library.root(),
        &repo_root,
        &snapshot,
        rename,
    )
    .is_some()
    {
        Some(HuggingFaceCacheHealth::installed())
    } else {
        Some(HuggingFaceCacheHealth::missing(files.to_vec()))
    }
}

/// Install-state health for a tier whose pinned revision SUPERSEDES an exact older one
/// ([`sceneworks_core::hf_revision_supersessions`]) while the tier exists ONLY at that older
/// revision — the v0.9.0 install of the Qwen-Image 2.1 q8/q4 re-host after sc-25909 republished it
/// to add each tier's NOTICE. The worker carries such a tier to the current pin cache-only at load
/// time, but only when the legacy snapshot holds every carried file at its exact size; the coarse
/// revision-agnostic probe would otherwise call a partial legacy tier installed and the render would
/// fail "not installed". Like the rename seam, serving the catalog never mutates the cache.
///
/// `None` (the ordinary probe decides) for every row without a declaration, and whenever the tier
/// is already present at the current pin.
pub(super) fn superseded_revision_tier_cache_health(
    data_dir: &FsPath,
    download: &Value,
    files: &[String],
) -> Option<HuggingFaceCacheHealth> {
    use sceneworks_core::hf_revision_supersessions::{
        exact_huggingface_revision_supersession, supersession_tier,
        validate_superseded_legacy_snapshot,
    };
    let repo = download.get("repo").and_then(Value::as_str)?;
    let revision = download.get("revision").and_then(Value::as_str)?;
    let supersession = exact_huggingface_revision_supersession(repo, revision, files)?;
    let tier = supersession_tier(supersession)?;
    let repo_root = huggingface_repo_cache_path(data_dir, repo)?;
    if !huggingface_repo_cache_exists(&repo_root) {
        return None;
    }
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    let snapshots = library
        .snapshots_root_from_repository_root(&repo_root)
        .ok()?;
    let legacy_snapshot = snapshots.join(supersession.legacy_revision);
    if snapshots
        .join(supersession.current_revision)
        .join(tier)
        .exists()
        || !legacy_snapshot.join(tier).exists()
    {
        return None;
    }
    if validate_superseded_legacy_snapshot(
        library.root(),
        &repo_root,
        &legacy_snapshot,
        supersession,
    )
    .is_some()
    {
        return Some(HuggingFaceCacheHealth::installed());
    }
    let missing = supersession
        .carried
        .iter()
        .filter(|file| {
            std::fs::metadata(legacy_snapshot.join(file.path)).map_or(true, |metadata| {
                !metadata.is_file() || metadata.len() != file.size
            })
        })
        .map(|file| file.path.to_owned())
        .collect::<Vec<_>>();
    Some(HuggingFaceCacheHealth::missing(if missing.is_empty() {
        // Every entry reads at its size yet confinement failed (an escaping link): name the tier.
        vec![format!("{tier}/*")]
    } else {
        missing
    }))
}

// Per-variant install state (sc-8508, epic 8506): a single downloadable tier of a quant-matrix
// model. `install_state_for` reports the DEFAULT variant's install state (back-compat single-variant
// contract); `model_variants` reports one of these per declared download entry so the catalog knows
// WHICH tiers are on disk, not just whether *a* variant is.
pub(super) struct ModelVariantState {
    /// The tier key: an explicit `downloads[].variant` (bf16/q8/q4), else "default" for a
    /// single-variant model (which has exactly one entry).
    pub(super) variant: String,
    /// Whether this specific tier's files are present in the HF cache.
    pub(super) installed: bool,
    /// Resolved install path for this tier (the shared repo cache root; tiers live as `files`-
    /// filtered subdirs within it). `None` when the repo has never been fetched.
    pub(super) installed_path: Option<String>,
    /// This tier's incomplete-cache signal (some but not all `files` present).
    pub(super) cache_incomplete: bool,
    /// A present primary is missing required companions; primary-only receipts cannot repair this.
    pub(super) dependencies_missing: bool,
    /// Files this tier is missing from the cache (empty when complete or absent).
    pub(super) missing_required_files: Vec<String>,
    /// This tier's estimated download size (from `downloads[].estimatedSizeBytes` /
    /// `footprint.diskSizeBytes`).
    pub(super) download_size_bytes: Option<u64>,
    /// The raw `downloads[].footprint` object (disk size + optional measured memory), passed
    /// through verbatim for the RAM-suggestion surfaces (sc-8509/8516). `Null` when absent.
    pub(super) footprint: Value,
    /// This tier is DECLARED but its artifact is not published yet (`downloads[].pendingArtifact`,
    /// sc-24112). The picker still lists it — the tier axis is real — but it can never be queued,
    /// never reads `installed`, and is rendered unavailable rather than as a download button.
    pub(super) pending_artifact: bool,
    /// Whether the PER-TIER delete can reclaim this tier on its own (never for a pending row):
    /// true when the row carries a non-empty `files` scope, OR when it is a scope-less tier that is
    /// the sole row on its repo ([`is_sole_repo_tier`]) — `qwen_image_2_1`'s bf16 tier, which IS
    /// the whole upstream snapshot and shares that repo with nothing. A scope-less row on a repo a
    /// sibling shares is the model rather than a slice of it, and `DELETE
    /// /models/:id/variants/:variant` refuses it with "delete the whole model instead", so the UI
    /// must not offer a per-tier delete for it. Both answers come from [`tier_delete_scope`]'s own
    /// rule, so the button and the route cannot disagree.
    pub(super) tier_deletable: bool,
    /// sc-22998: `Some` only for a locally derived tier (`downloads[].localDerivation`). `true` while
    /// no snapshot verified against the pinned derived weights exists at its location — the tier
    /// is then not installed; `POST /models/:id/download {variant}` fetches the original and the
    /// worker derives the tier locally (sc-22999).
    pub(super) derivation_pending: Option<bool>,
}

// Whether `model`'s `downloads` array is a quant-matrix — i.e. at least one supported entry carries
// an explicit non-empty `variant` key (q4/q8/bf16). Entry COUNT is deliberately NOT a discriminator:
// the manifest uses multiple download entries for non-tier reasons too (alternate sources, native
// fallbacks, co-requisite TE repos — e.g. PiD backbone + gemma-2-2b-it, boogu mlx+native, krea, wan,
// ltx). Those are not quant matrices, so only variant-presence flags a model as one (sc-8508).
pub(super) fn model_has_variant_matrix(model: &Value) -> bool {
    let Some(downloads) = model.get("downloads").and_then(Value::as_array) else {
        return false;
    };
    downloads
        .iter()
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .any(|entry| {
            entry
                .get("variant")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
        })
}

// The shared per-tier completeness predicate for a no-`model_index` MLX turnkey
// (`sceneworks_core::mlx_tier_completeness`), keyed on the manifest `family`, or `None` for a family
// whose per-tier completeness is already accurate — diffusers turnkeys (flux/qwen/…) via the
// `model_index.json` augmentation in `huggingface_filtered_cache_health`, or a family with no bespoke
// per-tier layout. The catalog uses this to downgrade a tier that clears the coarse `<tier>/*` glob but
// is actually torn (backbone present, text-encoder/VAE/tokenizer missing) to `incomplete`, so the
// /models report agrees with the worker's tier resolvers, which gate on these SAME predicates
// (sc-13513). Anima is included so its convert-output states (`mlx_convert_output_tier_states`) tighten;
// on its variants[] source download the exact-path `files` filter already reports torn accurately, so
// this is a no-op there.
//
// `model_id` refines the choice where one family spans two on-disk contracts: the `sensenova-u1`
// family covers both the base ids and the distilled `_fast` twins, and only the latter need the
// 8-step distill accounted for (sc-14432).
pub(super) fn no_model_index_family_predicate(
    family: &str,
    model_id: &str,
) -> Option<fn(&FsPath) -> bool> {
    use sceneworks_core::mlx_tier_completeness as tc;
    match family {
        "anima" => Some(tc::anima_tier_complete),
        "boogu" => Some(tc::boogu_tier_complete),
        "sana" => Some(tc::sana_tier_complete),
        "scail2" => Some(tc::scail2_tier_complete),
        // sc-14432: the SenseNova-U1 turnkeys ship a FLAT unified tier (backbone + config at the tier
        // root, no `model_index.json`), so without a predicate a torn tier read `installed` and then
        // failed at load — "complete but unloadable", with re-downloading the only (useless) repair.
        // The `_fast` twins additionally need the pre-merged distill marker to be loadable, so the id —
        // not the family — picks the predicate. Dispatched through the SHARED id list the worker's tier
        // resolver uses, so an id the worker would not tighten is not tightened here either.
        "sensenova-u1" => tc::sensenova_tier_predicate(model_id),
        // Both MiniMax-H3 entries load both DiT partitions. Each partition's glob can match
        // an interrupted download, so require both indexed shard sets before advertising a tier.
        "minimax-h3" if tc::minimax_h3_tier_predicate(model_id).is_some() => {
            Some(|dir| tc::minimax_h3_tier_complete(dir) && tc::minimax_h3_ref_tier_complete(dir))
        }
        _ => None,
    }
}

// Whether a no-`model_index` MLX turnkey tier that CLEARED the coarse presence check is actually TORN —
// backbone present, but a text-encoder / VAE / tokenizer component missing per the shared family
// predicate. `false` when the family has no bespoke predicate, the `files` filter isn't a single tier
// subdir (e.g. the candle whole-repo SANA entry, `files: []`), or SOME candidate location holds a
// complete tier. Drives the coarse `installed` → `incomplete` downgrade in BOTH the per-variant states
// (`model_variant_states`) and the top-level aggregate (`install_state_for`), so the /models badge, the
// variant state, and the worker's loader gate all agree (sc-13513). Every cache snapshot AND the managed
// dir is checked with `any` — a SANA repo keeps two snapshots (tiered + flat), and a complete tier in one
// location must not be dragged down by a sibling that lacks it.
pub(super) fn no_model_index_tier_is_torn(
    family: &str,
    model_id: &str,
    files: &[String],
    cache_path: Option<&FsPath>,
    managed_path: &FsPath,
) -> bool {
    let Some(complete) = no_model_index_family_predicate(family, model_id) else {
        return false;
    };
    let Some(tier) = tier_subdir_name(files) else {
        return false;
    };
    let mut tier_bases = cache_path
        .map(huggingface_snapshot_dirs)
        .unwrap_or_default();
    tier_bases.push(managed_path.to_path_buf());
    !tier_bases.iter().any(|base| complete(&base.join(&tier)))
}

// The `missingRequiredFiles` marker for a torn no-`model_index` tier (`no_model_index_tier_is_torn`),
// scoped to the tier subdir when the filter names one.
pub(super) fn torn_tier_marker(files: &[String]) -> String {
    tier_subdir_name(files)
        .map(|tier| format!("{tier}/ (incomplete: missing model components)"))
        .unwrap_or_else(|| "incomplete: missing model components".to_owned())
}

// Select the canonical source entry for each emitted variant. A single-variant model yields one
// "default" entry; a quant matrix yields one per tier. Install probing and the later live-size
// refresh share this selector so their response-array positions cannot drift.
pub(super) fn model_variant_downloads(model: &Value) -> Vec<&Value> {
    let Some(downloads) = model.get("downloads").and_then(Value::as_array) else {
        return Vec::new();
    };
    // Emitted variant keys must be unique: the single-variant case is exactly one "default", and a
    // quant matrix has one entry per distinct q4/q8/bf16 tier. Guard against a manifest that maps two
    // supported entries to the same key (unlabeled alternate sources both collapsing to "default", or
    // a duplicated `variant`) — keep the first, drop the rest, so downstream per-variant tracking
    // never emits two same-keyed states (sc-8508).
    let mut seen_variants = std::collections::HashSet::new();
    downloads
        .iter()
        // Co-requisites (sc-9696) are dependencies, not selectable tiers — never a variant state.
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .filter(|entry| {
            let key = entry
                .get("variant")
                .and_then(Value::as_str)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "default".to_owned());
            seen_variants.insert(key)
        })
        .collect()
}

// Build the per-variant install state for every selected download entry. Each entry is probed
// independently against the HF cache using that tier's own `files` filter, so the catalog can
// advertise (e.g.) bf16 installed while q4 is missing.
pub(super) fn model_variant_states(model: &Value, data_dir: &FsPath) -> Vec<ModelVariantState> {
    // sc-13513: the manifest `family` selects the shared per-tier completeness predicate for the
    // no-`model_index` MLX turnkeys (SANA/Boogu) so a torn tier reads `incomplete`, not `installed`.
    let family = model
        .get("family")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
    model_variant_downloads(model)
        .into_iter()
        .map(|entry| {
            let repo = entry
                .get("repo")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let files = string_array_field(entry, "files");
            let cache_path = huggingface_repo_cache_path(data_dir, &repo);
            let cache_health = superseded_revision_tier_cache_health(data_dir, entry, &files)
                .or_else(|| {
                    model_cache_health_for_download(
                        model_id,
                        &repo,
                        Some(entry),
                        cache_path.as_deref(),
                        &files,
                        data_dir,
                    )
                });
            let cache_installed = cache_health.as_ref().is_some_and(|health| health.installed);
            let mut cache_incomplete = cache_health
                .as_ref()
                .is_some_and(|health| health.incomplete);
            let mut missing_required_files = cache_health
                .as_ref()
                .map(|health| health.missing_files.clone())
                .unwrap_or_default();
            // The managed dir mirrors the default-download install path; a variant present there (a
            // directly-downloaded turnkey) counts as installed too — but the check must be TIER-aware.
            // A quant-matrix repo writes ONE repo-level completion marker no matter which tier was
            // fetched, so keying a per-tier "installed" on the bare marker made EVERY tier report
            // installed after any single tier's download (sc-9909). Require the tier's own files to
            // actually exist under the managed dir, not just the marker.
            let managed_path = data_dir.join("models").join(safe_download_dir(&repo));
            let managed_installed = if is_flux3_runtime_model(model_id) {
                false // The shared receipt resolver already judged this root.
            } else {
                managed_tier_installed(&managed_path, &files)
            };
            let mut installed = cache_installed || managed_installed;

            // sc-13513: the coarse `<tier>/*` glob (and the managed presence check) is satisfied by ANY
            // single nested file, so a TORN tier of a no-`model_index` MLX turnkey (SANA/Boogu — backbone
            // present, text-encoder/VAE/tokenizer missing) read `installed`. The shared family predicate
            // downgrades it to `incomplete`. Only ever downgrades a coarse `installed`; a clean-missing
            // tier is never promoted, and a family without a bespoke predicate (diffusers turnkeys,
            // accurate via the `model_index.json` augmentation) is untouched.
            if installed
                && no_model_index_tier_is_torn(
                    family,
                    model_id,
                    &files,
                    cache_path.as_deref(),
                    &managed_path,
                )
            {
                installed = false;
                cache_incomplete = true;
                let marker = torn_tier_marker(&files);
                if !missing_required_files.contains(&marker) {
                    missing_required_files.push(marker);
                }
            }

            // sc-22998: a locally derived tier's `files` are its ORIGINAL's, so the cache probe
            // above describes the original, never the tier. It is installed only when a derived
            // snapshot at its declared location verifies against the pinned derived weights (and
            // its companions, checked below, are present); otherwise it is `derivationPending`.
            let mut derivation_pending = None;
            let mut derived_path = None;
            let mut derived_exists = false;
            if sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
                entry,
            ) {
                use sceneworks_core::model_artifacts::artifact_selection::{
                    derived_snapshot_state, local_derivation, local_derivation_snapshot_dir,
                    DerivedSnapshotState,
                };
                let variant = entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let located = local_derivation(entry).and_then(|derivation| {
                    local_derivation_snapshot_dir(data_dir, model_id, variant, &derivation)
                        .map(|dir| (dir, derivation))
                });
                let state = match &located {
                    Some((dir, derivation)) => derived_snapshot_state(dir, derivation),
                    None => DerivedSnapshotState::Invalid(
                        "the tier declares a malformed local derivation".to_owned(),
                    ),
                };
                missing_required_files.clear();
                cache_incomplete = false;
                let verified = state == DerivedSnapshotState::Verified;
                derived_exists = !matches!(state, DerivedSnapshotState::Absent);
                if let DerivedSnapshotState::Invalid(why) = &state {
                    missing_required_files.push(format!("derived {variant} snapshot: {why}"));
                }
                // The companion check below still applies to a verified derived tier.
                installed = verified && (!is_flux3_runtime_model(model_id) || cache_installed || managed_installed);
                if verified && !installed {
                    missing_required_files.push("FLUX 3 derived tier requires its complete pinned original text encoder and autoencoder".to_owned());
                }
                derivation_pending = Some(!installed);
                derived_path = located
                    .filter(|_| derived_exists)
                    .map(|(dir, _)| dir.display().to_string());
            }

            // A tier is usable only with its own required companions. Checking these only at
            // model level let a complete q4 encoder certify bf16 and hid its repair action.
            // An absent primary remains absent even if shared files from another tier exist.
            let mut dependencies_missing = false;
            if installed || cache_incomplete {
                // Any installed option of a choice group satisfies it (sc-22998).
                for download in co_requisite_rows_gating_install(
                    model_co_requisite_downloads_for_variant_all_options(
                        model,
                        entry.get("variant").and_then(Value::as_str),
                    ),
                    |download| co_requisite_presence(data_dir, download),
                )
                .into_iter()
                .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft"))
                {
                    let health = co_requisite_cache_health(data_dir, &download);
                    if health.as_ref().is_some_and(|health| health.installed) {
                        continue;
                    }
                    installed = false;
                    cache_incomplete = true;
                    dependencies_missing = true;
                    let repo = download
                        .get("repo")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let missing = health
                        .map(|health| health.missing_files)
                        .filter(|files| !files.is_empty())
                        .unwrap_or_else(|| string_array_field(&download, "files"));
                    if missing.is_empty() {
                        missing_required_files.push(repo.to_owned());
                    } else {
                        missing_required_files
                            .extend(missing.iter().map(|file| format!("{repo}/{file}")));
                    }
                }
            }

            let installed_path = if is_flux3_runtime_model(model_id) {
                if cache_installed || (cache_incomplete && !managed_installed) {
                    flux3_selected_snapshot(data_dir, &repo, Some(entry), model_id)
                } else if managed_installed {
                    Some(managed_path.clone())
                } else {
                    None
                }
            } else if cache_installed || cache_incomplete {
                cache_path
            } else if managed_installed {
                Some(managed_path)
            } else {
                None
            };
            // sc-24112: a DECLARED-but-unpublished tier can never be installed, whatever the cache
            // probe above happened to find. Forcing it here rather than filtering the row out keeps
            // the tier in the picker (the axis is real) while making every downstream
            // "installed?"/"queue it?" answer false through the ONE field they all read.
            let pending_artifact = is_pending_artifact_download(entry);
            if pending_artifact {
                installed = false;
                cache_incomplete = false;
            }
            ModelVariantState {
                variant: entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "default".to_owned()),
                installed,
                installed_path: if derivation_pending.is_some() {
                    derived_path
                } else {
                    installed_path.map(|path| path.display().to_string())
                },
                cache_incomplete,
                dependencies_missing,
                missing_required_files,
                download_size_bytes: manifest_download_size_bytes(model, entry)
                    .or_else(|| variant_footprint_disk_bytes(entry)),
                footprint: entry.get("footprint").cloned().unwrap_or(Value::Null),
                pending_artifact,
                tier_deletable: !pending_artifact
                    && if derivation_pending.is_some() {
                        // Only the derived snapshot is ever deleted, so only one that exists.
                        derived_exists
                    } else {
                        !files.is_empty() || is_sole_repo_tier(model, entry)
                    },
                derivation_pending,
            }
        })
        .collect()
}

// Whether a tier's OWN artifacts live in the app-managed turnkey dir (data/models/<repo>), as opposed
// to the shared HF cache. The repo-level completion marker (.sceneworks-download-complete.json) alone
// does NOT certify a tier: a quant-matrix repo writes exactly one marker regardless of which tier was
// downloaded, so a bare-marker check reported every tier of a repo installed after any single tier's
// fetch (sc-9909). Require BOTH the marker AND — for a tier that declares a `files` filter — that the
// tier's files actually exist under the managed dir. A single-variant turnkey (empty `files`) is
// certified by the marker alone, preserving the pre-matrix contract.
pub(super) fn managed_tier_installed(managed_path: &FsPath, files: &[String]) -> bool {
    if !model_is_installed(managed_path) {
        return false;
    }
    files.is_empty()
        || files
            .iter()
            .all(|pattern| snapshot_contains_pattern(managed_path, pattern))
}

// The on-disk size a `downloads[].footprint.diskSizeBytes` declares, if any — the tier-scoped
// footprint signal (sc-8508) used as a fallback size when `estimatedSizeBytes` is absent.
pub(super) fn variant_footprint_disk_bytes(download: &Value) -> Option<u64> {
    download
        .get("footprint")
        .and_then(|footprint| footprint.get("diskSizeBytes"))
        .and_then(json_size_to_u64)
}

// Gated-model signal (sc-1898): a machine-readable `gated` flag plus the credential
// host the download requires, so the Models screen can route the user to the
// credential screen before a download will succeed. The host honors an explicit
// manifest `credentialHost` and otherwise derives from the download provider/source
// URL; `licenseUrl` passes through untouched.
pub(super) fn apply_gating_fields(object: &mut JsonObject) {
    let gated = object
        .get("gated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    object.insert("gated".to_owned(), Value::Bool(gated));
    if gated {
        let credential_host = object
            .get("credentialHost")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
            .or_else(|| derive_credential_host(object));
        object.insert(
            "credentialHost".to_owned(),
            credential_host.map(Value::String).unwrap_or(Value::Null),
        );
    }
}

// Mac UI gating (sc-3486): per-model Rust/MLX support so the web client can hide/
// disable a model with no native Mac lane in the pickers, plus (macOS only) the MLX availability +
// conversion status for models that declare an `mlx` variant. Additive fields the
// web/Docker build ignores; the client only acts on macSupport when the capabilities
// endpoint reports `macGatingActive`, so non-Mac pickers are untouched.
// Per-variant catalog fields (sc-8508): emit a `variants` array — one object per declared quant
// tier — plus a `hasVariantMatrix` boolean the web uses to decide whether to render a tier picker.
// A single-variant model gets a one-element array keyed "default" that mirrors the top-level
// install state, so the existing single-variant contract is unchanged.
pub(super) fn apply_variant_fields(object: &mut JsonObject, data_dir: &FsPath) {
    let model = Value::Object(object.clone());
    let has_matrix = model_has_variant_matrix(&model);
    let variants = model_variant_states(&model, data_dir)
        .into_iter()
        .map(|variant| {
            let mut row = json!({
                "variant": variant.variant,
                "installed": variant.installed,
                // sc-24112: a pending tier is neither installed nor installable. `"pending"` is a
                // THIRD install state rather than `"missing"` so the web can say "not published
                // yet" instead of offering a download button that would queue a fetch of a
                // revision that does not resolve.
                "installState": if variant.pending_artifact {
                    "pending"
                } else if variant.installed {
                    "installed"
                } else if variant.derivation_pending == Some(true) {
                    // sc-22998: a locally derived tier with no verified derived snapshot. A distinct
                    // state so the web offers neither a download (there is nothing to fetch) nor
                    // "missing" (which it would read as "download me").
                    "derivationPending"
                } else {
                    "missing"
                },
                "pendingArtifact": variant.pending_artifact,
                "tierDeletable": variant.tier_deletable,
                "cacheState": if variant.cache_incomplete {
                    "incomplete"
                } else if variant.installed {
                    "complete"
                } else {
                    "missing"
                },
                "installedPath": variant
                    .installed_path
                    .map(Value::String)
                    .unwrap_or(Value::Null),
                "missingRequiredFiles": variant.missing_required_files,
                "downloadSizeBytes": variant
                    .download_size_bytes
                    .map(|value| json!(value))
                    .unwrap_or(Value::Null),
                "footprint": variant.footprint,
            });
            // Emitted only on a locally derived tier, so every other model's variant rows keep
            // their existing shape.
            if let (Some(pending), Some(row)) = (variant.derivation_pending, row.as_object_mut()) {
                row.insert("derivationPending".to_owned(), Value::Bool(pending));
            }
            row
        })
        .collect::<Vec<_>>();
    object.insert("hasVariantMatrix".to_owned(), Value::Bool(has_matrix));
    object.insert("variants".to_owned(), Value::Array(variants));
}

pub(super) fn refresh_variant_download_sizes(model: &mut Value) -> Result<(), ApiError> {
    // A platformCleanupOnly tombstone strips the whole variants projection in
    // `apply_model_catalog_entry` — the missing array is its contract, not corruption, and there
    // is nothing to refresh. Keep the strict array guard below for every real catalog entry.
    if model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true) {
        return Ok(());
    }
    // `apply_variant_fields` runs in the blocking install-state sweep while live HF estimation runs
    // concurrently. Refresh only the already-built response fields after both complete: rebuilding
    // variants here would repeat filesystem probes serially and undo the intended overlap.
    let sizes = model_variant_downloads(model)
        .into_iter()
        .map(|download| {
            manifest_download_size_bytes(model, download)
                .or_else(|| variant_footprint_disk_bytes(download))
        })
        .collect::<Vec<_>>();
    let variants = model
        .get_mut("variants")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ApiError::internal("Model catalog variants must be an array"))?;
    if variants.len() != sizes.len() {
        return Err(ApiError::internal(
            "Model catalog variant count changed during size enrichment",
        ));
    }
    for (variant, size) in variants.iter_mut().zip(sizes) {
        let object = variant
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("Model catalog variant must be an object"))?;
        object.insert(
            "downloadSizeBytes".to_owned(),
            size.map(|value| json!(value)).unwrap_or(Value::Null),
        );
    }
    Ok(())
}

/// The convert-output quant tiers present under a converted MLX dir (sc-10730), highest-fidelity first.
/// Convert-at-install models (e.g. Anima) write `<converted>/<tier>/<backbone>/…` for each of
/// bf16/q8/q4 in ONE convert job — the tiers are convert OUTPUTS, not per-tier downloads. This lets the
/// Studio offer a generation-time tier picker via the decoupled `mlxTiers` catalog field WITHOUT the
/// download variant-matrix (`hasVariantMatrix`), whose `QuantDownloadPanel` would render bogus per-tier
/// download buttons for a model that has no per-tier download. Empty for a flat converted dir (no tier
/// subdirs) → the web renders no picker. Mirrors the worker tier resolvers' "tier present" probe so the
/// catalog and `anima_tier_subdir` agree on which tiers are loadable.
pub(super) fn mlx_convert_output_tiers(converted_dir: &FsPath) -> Vec<&'static str> {
    ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| tier_subdir_has_weights(&converted_dir.join(tier)))
        .collect()
}

/// Per-tier install state for a convert-at-install model's FULL possible tier set (q4/q8/bf16), so the
/// Studio picker can show EVERY tier with the un-converted ones DISABLED — the same "show all, disable
/// unavailable" rule the download-matrix `variants[]` array gives. Unlike [`mlx_convert_output_tiers`]
/// (installed tiers only, which can only ever GROW the picker), this lists all three whether or not they
/// are on disk. Three states per tier: `"installed"`/`"complete"` (loadable), `"missing"`/`"incomplete"`
/// (a TORN tier — backbone present, a component gone; repairable by re-convert), and
/// `"missing"`/`"missing"` (never converted for this tier).
///
/// Completeness is family-specific (sc-13513): a convert-at-install family with a bespoke no-`model_index`
/// layout (Anima's `diffusion_models/ + text_encoders/ + vae/`) is gated on the SAME shared per-tier
/// predicate the worker's `anima_tier_subdir` resolver uses ([`no_model_index_family_predicate`]), so a
/// torn convert output reads `incomplete` rather than `installed`. Any other convert family keeps the
/// coarse [`tier_subdir_has_weights`] backbone probe.
pub(super) fn mlx_convert_output_tier_states(
    converted_dir: &FsPath,
    family: &str,
    model_id: &str,
) -> Vec<Value> {
    let predicate = no_model_index_family_predicate(family, model_id);
    ["q4", "q8", "bf16"]
        .into_iter()
        .map(|tier| {
            let dir = converted_dir.join(tier);
            let has_backbone = tier_subdir_has_weights(&dir);
            let complete = match predicate {
                Some(complete) => complete(&dir),
                None => has_backbone,
            };
            let (install_state, cache_state) = if complete {
                ("installed", "complete")
            } else if has_backbone {
                ("missing", "incomplete")
            } else {
                ("missing", "missing")
            };
            // Convert-at-install tiers have no downloads[] row of their own, so this runtime catalog
            // state is the only truthful place to surface their per-tier size. Count the files the
            // tier can load, including symlinked shared weights: unlike converted_tier_real_bytes
            // (deletion accounting), this is a residency estimate, not reclaimable disk accounting.
            let disk_size_bytes = converted_tier_loaded_bytes(&dir);
            json!({
                "tier": tier,
                "installState": install_state,
                "cacheState": cache_state,
                "diskSizeBytes": disk_size_bytes,
            })
        })
        .collect()
}

/// Sum the bytes visible to a converted tier load, including symlinked weight files.
///
/// This intentionally differs from [`converted_tier_real_bytes`], which excludes symlinks because it
/// answers "what would deleting this tier reclaim?". The catalog memory floor instead needs "what
/// bytes can this tier hold resident?", so a shared text encoder or VAE linked into the tier must count.
/// Directory symlinks are followed because a converter may link a shared component directory rather
/// than its individual files. Canonical-directory de-duplication prevents cycles and counts a shared
/// component once even when more than one in-tier path aliases it.
pub(super) fn converted_tier_loaded_bytes(tier_dir: &FsPath) -> Option<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![tier_dir.to_path_buf()];
    let mut visited_dirs = std::collections::HashSet::new();
    let mut visited_files = std::collections::HashSet::new();
    while let Some(dir) = stack.pop() {
        let canonical = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if !visited_dirs.insert(canonical) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if is_hidden_file(&path) {
                continue;
            }
            let Ok(link_meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if link_meta.file_type().is_symlink() {
                if let Ok(target_meta) = std::fs::metadata(&path) {
                    if target_meta.is_file() {
                        let canonical =
                            std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                        if visited_files.insert(canonical) {
                            total = total.saturating_add(target_meta.len());
                        }
                    } else if target_meta.is_dir() {
                        stack.push(path);
                    }
                }
            } else if link_meta.is_dir() {
                stack.push(path);
            } else if link_meta.is_file() {
                let canonical = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if visited_files.insert(canonical) {
                    total = total.saturating_add(link_meta.len());
                }
            }
        }
    }
    (total > 0).then_some(total)
}

/// Whether a converted tier subdir holds loadable weights: a non-hidden `.safetensors` / `.index.json`
/// under a known backbone dir (`diffusion_models/` for Anima's Cosmos DiT, `transformer/` for other
/// DiTs, `unet/` for SDXL) or flat in the tier dir. A hidden `._*` AppleDouble sidecar is not a weight
/// (SceneWorks#1333), mirroring the worker resolvers.
pub(super) fn tier_subdir_has_weights(tier_dir: &FsPath) -> bool {
    if !tier_dir.is_dir() {
        return false;
    }
    let dir_has_weight = |dir: &FsPath| -> bool {
        std::fs::read_dir(dir).is_ok_and(|entries| {
            entries.flatten().any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                !name.starts_with("._")
                    && (name.ends_with(".safetensors") || name.ends_with(".index.json"))
            })
        })
    };
    // A known backbone subdir (`diffusion_models/` Anima, `transformer/` DiTs, `unet/` SDXL), or flat
    // in the tier dir itself.
    dir_has_weight(tier_dir)
        || ["diffusion_models", "transformer", "unet"]
            .into_iter()
            .any(|sub| dir_has_weight(&tier_dir.join(sub)))
}

pub(crate) fn model_is_installed(path: &FsPath) -> bool {
    path.is_dir() && path.join(".sceneworks-download-complete.json").is_file()
}

pub(crate) struct MlxCatalogStatus {
    pub(crate) install_state: &'static str,
    pub(crate) conversion_state: &'static str,
    pub(crate) converted_path: Option<PathBuf>,
    /// A newer source checkpoint is available than the one this install was converted from.
    /// True only for a converted `requiresConversion` model whose manifest `convertSourceFile`
    /// is NOT present in the `convertSourceRepo` cache (the installed converted dir carries no
    /// version stamp, so the source cache is the proxy — see `convert_source_file_cached`).
    pub(crate) update_available: bool,
    /// Why a converted artifact that carries the install marker would still be refused at load
    /// (sc-24216): the engine's own pin-contract reason, from
    /// [`sceneworks_core::converted_artifacts`]. `Some` only for a converter with a declared
    /// contract whose marked output fails it — the status then reads as needing conversion, so the
    /// Model Manager offers Convert again instead of a model that cannot load.
    pub(crate) conversion_problem: Option<String>,
}

/// Whether the named source `file` is present in any cached snapshot of `repo` — the proxy for
/// "the current manifest source has been downloaded." Keys off the manifest fields alone, so it
/// works for every convert-at-install model with no per-model logic.
pub(super) fn convert_source_file_cached(data_dir: &FsPath, repo: &str, file: &str) -> bool {
    huggingface_repo_cache_path(data_dir, repo)
        .map(|root| crate::huggingface_snapshot_dirs(&root))
        .unwrap_or_default()
        .iter()
        .any(|snapshot| snapshot.join(file).is_file())
}

/// macOS Model Manager status for a model's `mlx` variant. Returns `None` when the
/// model declares no `mlx` block.
///
/// `conversion_state`:
/// - `ready`            turnkey MLX repo (no conversion needed)
/// - `converted`        requiresConversion and the local MLX dir exists
/// - `needs_conversion` source checkpoint present, MLX dir absent
/// - `needs_source`     source checkpoint not downloaded yet
///
/// `install_state` is `installed` when the usable MLX artifact exists.
pub(crate) fn mlx_catalog_status(
    model: &serde_json::Map<String, Value>,
    data_dir: &FsPath,
) -> Option<MlxCatalogStatus> {
    let mlx = model.get("mlx").and_then(Value::as_object)?;
    let repo_cached = |repo: &str| {
        huggingface_repo_cache_path(data_dir, repo)
            .as_deref()
            .is_some_and(huggingface_repo_cache_exists)
    };
    if mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let converted_dir = data_dir.join("models").join("mlx").join(model_id);
        // mlx-video converters write a top-level config.json; the FLUX.2-klein
        // diffusers converter (sc-2235) writes a diffusers model_index.json. Either
        // marks a finished local MLX artifact.
        let marked = converted_dir.join("config.json").is_file()
            || converted_dir.join("model_index.json").is_file();
        // A marked artifact from a converter with an engine pin contract must also satisfy it
        // (manifest, pins, file sizes — metadata reads only); a truncated, edited or half-written
        // bundle is reported with the engine's reason rather than as installed (sc-24216).
        let conversion_problem = if marked {
            sceneworks_core::converted_artifacts::converted_artifact_problem(
                mlx.get("converter")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                &converted_dir,
                sceneworks_core::converted_artifacts::ArtifactCheck::Sizes,
            )
        } else {
            None
        };
        if marked && conversion_problem.is_none() {
            // The converted artifact records no source version, so use the source cache as the
            // proxy: if the manifest's current `convertSourceFile` is NOT cached, this install was
            // built from an older source → an update is available. A dir-based converter with no
            // `convertSourceFile` simply never reports an update (no false positives).
            let update_available = match (
                mlx.get("convertSourceRepo").and_then(Value::as_str),
                mlx.get("convertSourceFile").and_then(Value::as_str),
            ) {
                (Some(repo), Some(file)) if !file.trim().is_empty() => {
                    !convert_source_file_cached(data_dir, repo, file)
                }
                _ => false,
            };
            return Some(MlxCatalogStatus {
                install_state: "installed",
                conversion_state: "converted",
                converted_path: Some(converted_dir),
                update_available,
                conversion_problem: None,
            });
        }
        let source_present = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        Some(MlxCatalogStatus {
            install_state: "missing",
            conversion_state: if source_present {
                "needs_conversion"
            } else {
                "needs_source"
            },
            converted_path: None,
            update_available: false,
            conversion_problem,
        })
    } else {
        let repo_installed = mlx
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        // A turnkey model may still be served by a pre-existing local conversion at
        // <data>/models/mlx/<id> — the worker's resolve_*_model_dir prefers a local dir over
        // the turnkey download. Count that as installed too, so a model flipped from
        // requiresConversion → turnkey (sc-5599) doesn't read as "missing" for users who had
        // already converted it locally.
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let local_dir = data_dir.join("models").join("mlx").join(model_id);
        let local_installed = local_dir.join("config.json").is_file();
        Some(MlxCatalogStatus {
            install_state: if repo_installed || local_installed {
                "installed"
            } else {
                "missing"
            },
            conversion_state: "ready",
            converted_path: local_installed.then_some(local_dir),
            // Turnkey models have no local conversion to go stale (they track their repo directly).
            update_available: false,
            conversion_problem: None,
        })
    }
}

pub(crate) fn model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = model_manifest_installed_path(model, data_dir) {
        paths.push(path);
    }
    // A convert-at-install model's converted output (`<data>/models/mlx/<id>`, the `outputDir` every
    // convert job is given) is owned by this entry alone, so a whole-model delete reclaims it with
    // the source checkpoint (sc-24216). Before this the delete removed the source only and the
    // marker left behind kept the model reading as installed. Co-requisites (shared components such
    // as a text encoder or VAE) are never listed here — see `is_co_requisite_download`.
    if let Some(converted) = converted_artifact_dir(model, data_dir) {
        paths.push(converted);
    }
    let cleanup_downloads = model_download(model).into_iter().collect::<Vec<_>>();
    if model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true) {
        paths.extend(cleanup_with_model_artifact_paths(model, data_dir));
    }
    for repo in cleanup_downloads.into_iter().filter_map(|download| {
        download
            .get("repo")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }) {
        paths.push(data_dir.join("models").join(safe_download_dir(&repo)));
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, &repo) {
            paths.push(cache_path);
        }
    }
    if let Some(source_path) = model
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        let run_dir = if path.is_absolute() {
            path
        } else {
            data_dir.join(path)
        };
        // An Iris training run's checkpoint root and its `.prior` retirement (sc-25690), named by
        // the engine contract. They sit inside the run directory, which goes first so a trash
        // delete moves the run as one item; the named paths then catch any the directory missed.
        let checkpoint_dirs = crate::iris_training::iris_run_checkpoint_paths(model, &run_dir);
        paths.push(run_dir);
        paths.extend(checkpoint_dirs);
    }
    unique_paths(paths)
}

/// The converted-artifact dir a `mlx.requiresConversion` entry owns: `<data>/models/mlx/<id>`, the
/// convert job's `outputDir`. `None` for every other entry, and for an id that is not one plain path
/// component (never joined outside the conversion root).
pub(super) fn converted_artifact_dir(model: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let requires_conversion = model
        .get("mlx")
        .and_then(|mlx| mlx.get("requiresConversion"))
        .and_then(Value::as_bool)
        == Some(true);
    if !requires_conversion {
        return None;
    }
    let id = model.get("id").and_then(Value::as_str)?;
    let mut components = FsPath::new(id).components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(_)), None) => {
            Some(data_dir.join("models").join("mlx").join(id))
        }
        _ => None,
    }
}

pub(super) fn cleanup_with_model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for repo in model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| download.get("cleanupWithModel").and_then(Value::as_bool) == Some(true))
        .filter_map(|download| download.get("repo").and_then(Value::as_str))
    {
        paths.push(data_dir.join("models").join(safe_download_dir(repo)));
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, repo) {
            paths.push(cache_path);
        }
    }
    unique_paths(paths)
}

pub(crate) fn model_manifest_installed_path(model: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let raw_path = model
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if raw_path.contains("${") {
        return None;
    }
    let path = PathBuf::from(raw_path);
    Some(if path.is_absolute() {
        path
    } else {
        data_dir.join(path)
    })
}
