//! ProjectStore indexing aggregate.

use super::*;

#[derive(Default)]
pub(super) struct ReindexCounts {
    assets: u32,
    generation_sets: u32,
    timelines: u32,
}

pub(super) fn reindex_project_path(
    project_id: &str,
    project_path: &Path,
    prune_orphans: bool,
) -> ProjectStoreResult<ReindexCounts> {
    let index_mutation = AssetIndexMutation::begin_reindex(project_path)?;
    // sc-10117: rewrite inline-upscaled sidecars missing their fold link BEFORE indexing, so the
    // rebuilt index — and the Library, which reads the sidecars directly — sees the healed lineage.
    // Additive + idempotent, so it is safe to run on every reindex.
    backfill_upscale_variant_lineage(project_path)?;

    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute("delete from assets", [])?;
    transaction.execute("delete from generation_sets", [])?;
    transaction.execute("delete from timelines", [])?;
    clear_character_tables(&transaction)?;

    let mut counts = ReindexCounts::default();
    for sidecar_path in asset_sidecars(project_path)? {
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::SidecarRead);
        let asset = match read_json(&sidecar_path) {
            Ok(asset) => asset,
            Err(error) => {
                tracing::warn!(event = "asset_reindex_sidecar_invalid", sidecar = %sidecar_path.display(), error = %error, "skipping malformed asset sidecar during reindex");
                continue;
            }
        };
        if let Err(error) = index_asset_on_connection(
            &transaction,
            project_id,
            project_path,
            &asset,
            Some(&sidecar_path),
        ) {
            match error {
                // The caller's rebuild transaction is the SQLite invariant: never
                // turn a database failure into a superficially successful index.
                ProjectStoreError::Sqlite(_) => return Err(error),
                error => {
                    tracing::warn!(event = "asset_reindex_sidecar_invalid", sidecar = %sidecar_path.display(), error = %error, "skipping malformed asset sidecar during reindex");
                    continue;
                }
            }
        }
        counts.assets += 1;
    }

    let generation_set_dir = project_path.join("generation-sets");
    for entry in read_dir_paths(&generation_set_dir)? {
        if entry.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::GenerationSetRead);
        let generation_set = match read_json(&entry) {
            Ok(generation_set) => generation_set,
            Err(error) => {
                tracing::warn!(event = "generation_set_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed generation set sidecar during reindex");
                continue;
            }
        };
        let id = match required_str(&generation_set, "id") {
            Ok(id) => id,
            Err(error) => {
                tracing::warn!(event = "generation_set_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed generation set sidecar during reindex");
                continue;
            }
        };
        transaction.execute(
            "
            insert or replace into generation_sets (id, mode, model, prompt, created_at, job_id)
            values (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                id,
                optional_str(&generation_set, "mode").unwrap_or("unknown"),
                optional_str(&generation_set, "model").unwrap_or("unknown"),
                optional_str(&generation_set, "prompt").unwrap_or(""),
                optional_str(&generation_set, "createdAt").unwrap_or(""),
                optional_str(&generation_set, "jobId"),
            ],
        )?;
        counts.generation_sets += 1;
    }

    let timeline_dir = project_path.join("timelines");
    for entry in read_dir_paths(&timeline_dir)? {
        if !entry
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::TimelineRead);
        let timeline = match read_json(&entry) {
            Ok(timeline) => timeline,
            Err(error) => {
                tracing::warn!(event = "timeline_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed timeline sidecar during reindex");
                continue;
            }
        };
        let id = match required_str(&timeline, "id") {
            Ok(id) => id,
            Err(error) => {
                tracing::warn!(event = "timeline_reindex_sidecar_invalid", sidecar = %entry.display(), error = %error, "skipping malformed timeline sidecar during reindex");
                continue;
            }
        };
        let rel_path = relative_string(project_path, &entry)?;
        transaction.execute(
            "
            insert or replace into timelines (
              id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
            params![
                id,
                optional_str(&timeline, "name").unwrap_or("Timeline"),
                rel_path,
                optional_str(&timeline, "aspectRatio").unwrap_or("16:9"),
                sql_int(optional_u64(&timeline, "width").unwrap_or(1280)),
                sql_int(optional_u64(&timeline, "height").unwrap_or(720)),
                sql_int(optional_u64(&timeline, "fps").unwrap_or(30)),
                optional_f64(&timeline, "duration").unwrap_or(0.0),
                optional_str(&timeline, "createdAt").unwrap_or(""),
                optional_str(&timeline, "updatedAt")
                    .or_else(|| optional_str(&timeline, "createdAt"))
                    .unwrap_or(""),
            ],
        )?;
        counts.timelines += 1;
    }

    reindex_characters_on_connection(&transaction, project_path)?;
    transaction.execute(
        "insert or replace into project_metadata (key, value) values (?1, ?2)",
        params![ASSET_INDEX_VERSION_KEY, PROJECT_SCHEMA_VERSION.to_string()],
    )?;

    transaction.commit()?;
    if prune_orphans {
        // Keep the dirty marker and project lock until missing-media rows have
        // been removed and their surviving sidecars quarantined. Clearing it
        // after only the rebuild could resurrect a partially purged asset after
        // a crash.
        let pruned = prune_orphaned_asset_rows(project_path)?;
        counts.assets = counts.assets.saturating_sub(pruned as u32);
    }
    index_mutation.commit();
    Ok(counts)
}

/// Sidecars of pruned "purged-but-referenced" assets are moved here, out of the
/// folders `asset_sidecars` scans, so a rebuild can't resurrect them. A dotdir so
/// it reads as internal state, not user content.
pub(super) const ORPHANED_SIDECAR_DIR: &str = ".orphaned-sidecars";

/// Delete index rows whose recorded media file is gone from disk, and quarantine
/// each orphaned sidecar out of the scanned asset tree. Backs
/// [`ProjectStore::prune_orphaned_assets`]; see that method for the rationale.
/// Takes the per-project file lock so the read-then-delete is serialized against
/// concurrent indexing/mutation.
fn prune_orphaned_asset_rows(project_path: &Path) -> ProjectStoreResult<usize> {
    let _guard = lock_project_files(project_path);
    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    // (asset id, media path, orphaned sidecar to quarantine) for every row whose media is gone.
    let orphans: Vec<(String, String, Option<PathBuf>)> = {
        let mut statement =
            transaction.prepare("select id, file_path, sidecar_path from assets")?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        let mut orphans = Vec::new();
        for row in rows {
            let (asset_id, file_rel, sidecar_rel) = row?;
            let Some(file_rel) = file_rel else {
                continue;
            };
            // Normalize separators before joining — matching `normalize_asset_cached`'s
            // URL builder — so a Windows-authored sidecar (backslash paths) isn't
            // misread as missing on a POSIX host and wrongly pruned.
            let file_rel = file_rel.replace('\\', "/");
            if project_path.join(&file_rel).exists() {
                continue;
            }
            // Resolve the orphan's sidecar (the indexed path first, else derived from
            // the media path — mirroring `list_assets`) so it can be quarantined too.
            let sidecar = sidecar_rel
                .map(|rel| project_path.join(rel.replace('\\', "/")))
                .filter(|path| path.exists())
                .or_else(|| {
                    let derived = project_path
                        .join(&file_rel)
                        .with_extension("sceneworks.json");
                    derived.exists().then_some(derived)
                });
            orphans.push((asset_id, file_rel, sidecar));
        }
        orphans
    };
    for (asset_id, _, _) in &orphans {
        transaction.execute("delete from assets where id = ?1", params![asset_id])?;
    }
    transaction.commit()?;
    // The row(s) are gone from the registry; now move each orphaned sidecar OUT of
    // the scanned asset folders into `.orphaned-sidecars/` (never scanned by
    // `asset_sidecars`). This declutters the project and closes the resurrection
    // edge — a fully-orphaned project would otherwise re-add these rows via the
    // empty-table auto-reindex in `list_assets`. The sidecar is retained (moved, not
    // deleted) so the recipe metadata stays recoverable. Report a move failure after
    // attempting every orphan so dirty repair keeps its marker and retries instead
    // of clearing it while a resurrectable sidecar remains in the scanned tree.
    let mut cleanup_error = None;
    for (asset_id, media_rel, sidecar) in &orphans {
        if let Err(error) = remove_sibling_poster_no_symlinks(project_path, media_rel) {
            cleanup_error.get_or_insert(error);
        }
        let Some(sidecar) = sidecar else {
            continue;
        };
        if let Err(error) = quarantine_orphaned_sidecar(project_path, asset_id, sidecar) {
            tracing::debug!(
                event = "orphaned_sidecar_quarantine_failed",
                asset_id = %asset_id,
                sidecar = %sidecar.display(),
                error = %error,
                "could not move orphaned sidecar out of the asset tree; repair remains pending"
            );
            cleanup_error.get_or_insert(ProjectStoreError::Io(error));
        }
    }
    if let Some(error) = cleanup_error {
        return Err(error);
    }
    Ok(orphans.len())
}

/// Move an orphaned asset's sidecar into `<project>/.orphaned-sidecars/`, keyed by
/// the (charset-validated) asset id. Retained rather than deleted so the metadata
/// is recoverable; a same-key file from a prior prune is overwritten.
fn quarantine_orphaned_sidecar(
    project_path: &Path,
    asset_id: &str,
    sidecar: &Path,
) -> std::io::Result<()> {
    let dir = project_path.join(ORPHANED_SIDECAR_DIR);
    fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{asset_id}.sceneworks.json"));
    if dest.exists() {
        fs::remove_file(&dest)?;
    }
    fs::rename(sidecar, &dest)
}

pub(super) fn project_has_sidecars(project_path: &Path) -> bool {
    asset_sidecars(project_path).is_ok_and(|paths| !paths.is_empty())
}

impl ProjectStore {
    pub fn reindex_project(&self, project_id: &str) -> ProjectStoreResult<ReindexResult> {
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let counts = reindex_project_path(project_id, &project_path, true)?;
        Ok(ReindexResult {
            project_id: project_id.to_owned(),
            assets: counts.assets,
            generation_sets: counts.generation_sets,
            timelines: counts.timelines,
        })
    }

    /// Remove index rows for "purged-but-referenced" assets — rows whose recorded
    /// media file no longer exists on disk. These arise when an asset's media is
    /// deleted out-of-band, or a purge is interrupted after the file is unlinked
    /// but before the row is removed: the row (and its sidecar) linger, so
    /// `list_assets` keeps serving an asset whose `/files/` URL 404s on every open.
    /// The web UI degrades each to a placeholder (`assetMedia.jsx`), so the only
    /// symptom is server-log noise — one 404 per missing asset, every startup.
    ///
    /// Deletes the index row and moves the orphaned `*.sceneworks.json` sidecar out
    /// of the scanned asset tree into the project's `.orphaned-sidecars/` quarantine
    /// (see [`prune_orphaned_asset_rows`]). Quarantining rather than deleting keeps
    /// the recipe metadata recoverable, and moving it out of the folders
    /// `asset_sidecars` scans stops a rebuild (`reindex` / the empty-table
    /// auto-reindex in `list_assets`) from resurrecting the row. Media lives beside
    /// the sidecar under the project directory, so in the normal case a file missing
    /// from a readable project is a genuine purge. Idempotent; returns the number of
    /// rows pruned.
    pub fn prune_orphaned_assets(&self, project_id: &str) -> ProjectStoreResult<usize> {
        let project_path = self.find_project_path(project_id)?;
        prune_orphaned_asset_rows(&project_path)
    }

    /// Run [`prune_orphaned_assets`](Self::prune_orphaned_assets) across every
    /// registered project (including the reserved global libraries) that already
    /// has an initialized `project.db`. Best-effort per project: one project's
    /// failure is logged and does not abort the sweep. Intended as a startup
    /// data-integrity pass so the Library never fetches a purged asset. Returns the
    /// total number of rows pruned across all projects.
    pub fn prune_all_orphaned_assets(&self) -> ProjectStoreResult<usize> {
        let project_paths: Vec<PathBuf> = {
            let mut registry_cache = self.lock.lock();
            self.load_registry(&mut registry_cache)?
                .into_iter()
                .filter_map(|item| item.path.map(PathBuf::from))
                // Only touch projects that have been opened at least once (a DB
                // exists); never materialize a `project.db` for one that has not.
                .filter(|path| path.join("project.db").exists())
                .collect()
        };
        let mut total = 0;
        for project_path in project_paths {
            match prune_orphaned_asset_rows(&project_path) {
                Ok(count) => total += count,
                Err(error) => tracing::warn!(
                    event = "orphaned_asset_prune_project_failed",
                    project = %project_path.display(),
                    error = %error,
                    "could not prune orphaned assets for project"
                ),
            }
        }
        Ok(total)
    }
}
