//! ProjectStore asset indexing aggregate.

use super::*;

pub(super) const ASSET_INDEX_VERSION_KEY: &str = "assetIndexVersion";

pub(super) const ASSET_INDEX_DIRTY_MARKER: &str = ".asset-index-dirty";

thread_local! {
    static ACTIVE_ASSET_INDEX_MUTATIONS: RefCell<Vec<PathBuf>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn current_thread_owns_asset_index_mutation(project_path: &Path) -> bool {
    ACTIVE_ASSET_INDEX_MUTATIONS.with(|active| {
        active
            .borrow()
            .iter()
            .any(|active_path| active_path == project_path)
    })
}

/// Durable guard for a filesystem mutation whose indexed envelope is committed
/// in a separate SQLite transaction. The marker is written before the first
/// media/sidecar change and removed only after the DB transaction commits.
/// Any error or process interruption leaves it behind, so the next indexed read
/// rebuilds from disk instead of serving a stale envelope indefinitely.
pub(crate) struct AssetIndexMutation {
    project_path: PathBuf,
    marker_path: PathBuf,
    _project_guard: ReentrantMutexGuard<'static, ()>,
}

impl AssetIndexMutation {
    pub(crate) fn begin(project_path: &Path) -> ProjectStoreResult<Self> {
        let project_guard = lock_project_files(project_path);
        // Never let a new mutation overwrite and later clear a marker left by a
        // prior failed mutation. Repair that earlier filesystem/DB split first.
        ensure_project_db_ready(project_path)?;
        Self::write_marker(project_path, project_guard)
    }

    pub(super) fn begin_reindex(project_path: &Path) -> ProjectStoreResult<Self> {
        let project_guard = lock_project_files(project_path);
        Self::write_marker(project_path, project_guard)
    }

    fn write_marker(
        project_path: &Path,
        project_guard: ReentrantMutexGuard<'static, ()>,
    ) -> ProjectStoreResult<Self> {
        let marker_path = project_path.join(ASSET_INDEX_DIRTY_MARKER);
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::IndexMarkerWrite);
        write_json(
            &marker_path,
            &json!({
                "schemaVersion": 1,
                "reason": "asset-filesystem-mutation"
            }),
        )?;
        ACTIVE_ASSET_INDEX_MUTATIONS.with(|active| {
            active.borrow_mut().push(project_path.to_path_buf());
        });
        Ok(Self {
            project_path: project_path.to_path_buf(),
            marker_path,
            _project_guard: project_guard,
        })
    }

    pub(crate) fn commit(self) {
        clear_asset_index_dirty_marker(&self.marker_path);
    }
}

impl Drop for AssetIndexMutation {
    fn drop(&mut self) {
        ACTIVE_ASSET_INDEX_MUTATIONS.with(|active| {
            let mut active = active.borrow_mut();
            if let Some(index) = active
                .iter()
                .rposition(|active_path| active_path == &self.project_path)
            {
                active.remove(index);
            }
        });
    }
}

fn clear_asset_index_dirty_marker(marker_path: &Path) {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::IndexMarkerRemove);
    if let Err(error) = fs::remove_file(marker_path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::debug!(
                event = "asset_index_dirty_marker_remove_failed",
                marker = %marker_path.display(),
                error = %error,
                "asset index is committed but its repair marker remains; next read will reindex"
            );
        }
    }
}

pub(super) fn asset_index_is_dirty(project_path: &Path) -> bool {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::IndexMarkerRead);
    project_path.join(ASSET_INDEX_DIRTY_MARKER).exists()
}

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_ASSET_INDEX_DB_MUTATION: Cell<bool> = const { Cell::new(false) };
    static FAIL_NEXT_ASSET_SIDECAR_REMOVE: Cell<bool> = const { Cell::new(false) };
    static FAIL_NEXT_UPSCALE_LINEAGE_DB_READ: Cell<bool> = const { Cell::new(false) };
    static FAIL_NEXT_ASSET_POSTER_DB_READ: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(super) fn fail_next_asset_index_db_mutation() {
    FAIL_NEXT_ASSET_INDEX_DB_MUTATION.with(|fail| fail.set(true));
}

#[cfg(test)]
pub(super) fn fail_next_asset_sidecar_remove() {
    FAIL_NEXT_ASSET_SIDECAR_REMOVE.with(|fail| fail.set(true));
}

#[cfg(test)]
pub(super) fn fail_next_upscale_lineage_db_read() {
    FAIL_NEXT_UPSCALE_LINEAGE_DB_READ.with(|fail| fail.set(true));
}

#[cfg(test)]
pub(crate) fn fail_next_asset_poster_db_read() {
    FAIL_NEXT_ASSET_POSTER_DB_READ.with(|fail| fail.set(true));
}

#[cfg(test)]
fn maybe_fail_upscale_lineage_db_read() -> ProjectStoreResult<()> {
    if FAIL_NEXT_UPSCALE_LINEAGE_DB_READ.with(|fail| fail.replace(false)) {
        return Err(ProjectStoreError::Sqlite(rusqlite::Error::InvalidQuery));
    }
    Ok(())
}

#[cfg(not(test))]
fn maybe_fail_upscale_lineage_db_read() -> ProjectStoreResult<()> {
    Ok(())
}

#[cfg(test)]
pub(crate) fn maybe_fail_asset_poster_db_read() -> ProjectStoreResult<()> {
    if FAIL_NEXT_ASSET_POSTER_DB_READ.with(|fail| fail.replace(false)) {
        return Err(ProjectStoreError::Sqlite(rusqlite::Error::InvalidQuery));
    }
    Ok(())
}

#[cfg(not(test))]
pub(crate) fn maybe_fail_asset_poster_db_read() -> ProjectStoreResult<()> {
    Ok(())
}

#[cfg(test)]
pub(super) fn maybe_fail_asset_index_db_mutation() -> ProjectStoreResult<()> {
    let should_fail = FAIL_NEXT_ASSET_INDEX_DB_MUTATION.with(|fail| fail.replace(false));
    if should_fail {
        return Err(ProjectStoreError::Io(std::io::Error::other(
            "injected asset index DB mutation failure",
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn remove_asset_sidecar(path: &Path) -> ProjectStoreResult<()> {
    let should_fail = FAIL_NEXT_ASSET_SIDECAR_REMOVE.with(|fail| fail.replace(false));
    if should_fail {
        return Err(ProjectStoreError::Io(std::io::Error::other(
            "injected asset sidecar removal failure",
        )));
    }
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(not(test))]
pub(super) fn remove_asset_sidecar(path: &Path) -> ProjectStoreResult<()> {
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(not(test))]
pub(super) fn maybe_fail_asset_index_db_mutation() -> ProjectStoreResult<()> {
    Ok(())
}

/// sc-10117: heal upscale-variant lineage on inline-upscaled asset sidecars.
///
/// The Image Studio inline "Upscale" post-pass historically linked each `(Nx upscaled)` variant to
/// its base image with a bare `upscaledFrom` field that nothing read — and that was dropped at
/// sidecar-build time — so those variants never carried `extra.upscaledFromAssetId` /
/// `lineage.sourceAssetId` and never folded with their originals in the Library / Recent Batches.
/// The link was never persisted, so it is reconstructed from generation-set structure: within a
/// generation set an upscaled file `…_{index}_up{N}x.<ext>` is the upscaled variant of the base
/// `…_{index}.<ext>`. Rewrites only upscaled sidecars still missing the link (idempotent + additive)
/// and returns how many were healed. A reconstruction with no matching base is left untouched rather
/// than inventing a bogus parent.
pub(super) fn backfill_upscale_variant_lineage(project_path: &Path) -> ProjectStoreResult<usize> {
    let sidecars = asset_sidecars(project_path)?;
    // (generationSetId, relative media path) -> asset id, for every asset in the project.
    let mut base_ids: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new();
    let mut parsed: Vec<(PathBuf, Value)> = Vec::with_capacity(sidecars.len());
    for sidecar_path in sidecars {
        record_asset_list_filesystem_operation(AssetListFilesystemOperation::SidecarRead);
        let Ok(asset) = read_json(&sidecar_path) else {
            continue;
        };
        if let (Some(id), Some(genset), Some(path)) = (
            asset.get("id").and_then(Value::as_str),
            asset.get("generationSetId").and_then(Value::as_str),
            asset.pointer("/file/path").and_then(Value::as_str),
        ) {
            base_ids.insert((genset.to_owned(), path.to_owned()), id.to_owned());
        }
        parsed.push((sidecar_path, asset));
    }

    let mut healed = 0usize;
    for (sidecar_path, mut asset) in parsed {
        // Already linked (a standalone image_upscale asset, or a prior heal) — skip.
        if asset
            .pointer("/extra/upscaledFromAssetId")
            .and_then(Value::as_str)
            .is_some()
        {
            continue;
        }
        // Only touch assets the inline upscale post-pass produced (they carry an `upscale` record).
        if asset
            .pointer("/recipe/rawAdapterSettings/upscale")
            .is_none()
        {
            continue;
        }
        let (Some(genset), Some(path)) = (
            asset
                .get("generationSetId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            asset
                .pointer("/file/path")
                .and_then(Value::as_str)
                .map(str::to_owned),
        ) else {
            continue;
        };
        let Some((base_path, factor)) = strip_upscale_suffix(&path) else {
            continue;
        };
        // The reconstructed base must actually exist in the same generation set.
        let Some(base_id) = base_ids.get(&(genset, base_path)).cloned() else {
            continue;
        };
        let engine = asset
            .pointer("/recipe/rawAdapterSettings/upscale/engine")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        apply_upscale_variant_link(&mut asset, &base_id, factor, &engine);
        // A single failed rewrite must not abort the whole heal (or the reindex that calls it).
        if write_json(&sidecar_path, &asset).is_ok() {
            healed += 1;
        }
    }
    Ok(healed)
}

/// Split an upscaled media path `…_up{N}x.<ext>` into its base path `….<ext>` and the factor `N`.
/// Returns `None` when the filename doesn't carry the inline-upscale suffix.
fn strip_upscale_suffix(path: &str) -> Option<(String, u8)> {
    let (stem, ext) = path.rsplit_once('.')?;
    let marker = stem.rfind("_up")?;
    let factor: u8 = stem[marker + 3..].strip_suffix('x')?.parse().ok()?;
    Some((format!("{}.{ext}", &stem[..marker]), factor))
}

/// Stamp the fold/lineage keys the Library reads onto an upscaled asset in place, mirroring the
/// worker's standalone `image_upscale` fact (`sourceAssetId` / `parents` / `extra` markers).
pub(super) fn apply_upscale_variant_link(
    asset: &mut Value,
    base_id: &str,
    factor: u8,
    engine: &str,
) {
    let Some(object) = asset.as_object_mut() else {
        return;
    };
    if let Some(lineage) = object
        .entry("lineage")
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        lineage.insert("sourceAssetId".to_owned(), json!(base_id));
        lineage.insert("parents".to_owned(), json!([base_id]));
    }
    if let Some(extra) = object
        .entry("extra")
        .or_insert_with(|| json!({}))
        .as_object_mut()
    {
        extra.insert("isUpscaled".to_owned(), json!(true));
        extra.insert("upscaledFromAssetId".to_owned(), json!(base_id));
        extra.insert("factor".to_owned(), json!(factor));
        if !engine.is_empty() {
            extra.insert("engine".to_owned(), json!(engine));
        }
    }
}

/// Collect the upscale-fold lineage group around `asset_id` (sc-10205): the asset
/// itself, the original it was upscaled from, and transitively every upscaled
/// variant pointing back into the group. Mirrors the web's fold keys
/// (`extra.upscaledFromAssetId`, falling back to `lineage.sourceAssetId` when
/// `extra.isUpscaled` — assetVariants.js): the Library renders such a pair as ONE
/// tile, so a move of the visible asset must carry the whole group. The requested
/// id is always first. Unreadable sidecars are skipped — worst case the group
/// degrades to the single asset, never to an error.
pub(super) const UPSCALE_LINEAGE_QUERY: &str = "
    with recursive connected(id) as (
      values (?1)
      union
      select parent.upscaled_from_asset_id
        from assets parent join connected c on parent.id = c.id
       where parent.upscaled_from_asset_id is not null
      union
      select child.id
        from assets child join connected c on child.upscaled_from_asset_id = c.id
    )
    select id from connected
";

pub(super) fn upscale_lineage_group(
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Vec<String>> {
    let mut group = vec![asset_id.to_owned()];
    let connection = connect_project_db(project_path)?;
    maybe_fail_upscale_lineage_db_read()?;
    let mut statement = connection.prepare(UPSCALE_LINEAGE_QUERY)?;
    let rows = statement.query_map(params![asset_id], |row| row.get::<_, String>(0))?;
    for id in rows {
        let id = id?;
        if id != asset_id && !group.contains(&id) {
            group.push(id);
        }
    }
    Ok(group)
}

pub(super) fn index_asset(
    project_id: &str,
    project_path: &Path,
    asset: &Value,
    sidecar_path: Option<&Path>,
) -> ProjectStoreResult<()> {
    let sidecar_path = match sidecar_path {
        Some(path) => path.to_path_buf(),
        None => project_path
            .join(required_str(
                asset.get("file").ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset file is required".to_owned())
                })?,
                "path",
            )?)
            .with_extension("sceneworks.json"),
    };
    maybe_fail_asset_index_db_mutation()?;
    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    index_asset_on_connection(
        &transaction,
        project_id,
        project_path,
        asset,
        Some(&sidecar_path),
    )?;
    transaction.commit()?;
    Ok(())
}

/// Thin DB-prep wrapper over the shared resolver in `asset_index` (sc-4272).
/// `ensure_project_db_ready` backfills derived columns after a schema bump
/// before the lookup, unlike the character-store wrapper's lighter migrate.
pub(super) fn find_asset_sidecar_path(
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Option<PathBuf>> {
    ensure_project_db_ready(project_path)?;
    let connection = connect_project_db(project_path)?;
    find_asset_sidecar_path_on_connection(&connection, project_path, asset_id)
}

pub(super) fn purge_asset_record(project_path: &Path, asset_id: &str) -> ProjectStoreResult<()> {
    maybe_fail_asset_index_db_mutation()?;
    let mut connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute("delete from assets where id = ?1", params![asset_id])?;
    transaction.commit()?;
    Ok(())
}
