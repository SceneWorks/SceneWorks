//! ProjectStore asset queries aggregate.

use super::*;

impl AssetListFilesystemOperations {
    pub const fn total(self) -> u64 {
        self.registry_opens
            + self.registry_metadata_reads
            + self.registry_content_reads
            + self.path_stats
            + self.directory_scans
            + self.sidecar_reads
            + self.generation_set_reads
            + self.timeline_reads
            + self.character_reads
            + self.poster_stats
            + self.poster_reads
            + self.index_marker_reads
            + self.index_marker_writes
            + self.index_marker_removes
            + self.directory_create_calls
            + self.db_opens
    }
}

thread_local! {
    static ASSET_LIST_FILESYSTEM_OPERATIONS: RefCell<Vec<AssetListFilesystemOperations>> =
        const { RefCell::new(Vec::new()) };
}

pub(crate) fn record_asset_list_filesystem_operation(operation: AssetListFilesystemOperation) {
    ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(operations) = stack.last_mut() else {
            return;
        };
        match operation {
            AssetListFilesystemOperation::RegistryOpen => operations.registry_opens += 1,
            AssetListFilesystemOperation::RegistryMetadataRead => {
                operations.registry_metadata_reads += 1;
            }
            AssetListFilesystemOperation::RegistryContentRead => {
                operations.registry_content_reads += 1;
            }
            AssetListFilesystemOperation::PathStat => operations.path_stats += 1,
            AssetListFilesystemOperation::DirectoryScan => operations.directory_scans += 1,
            AssetListFilesystemOperation::SidecarRead => operations.sidecar_reads += 1,
            AssetListFilesystemOperation::GenerationSetRead => {
                operations.generation_set_reads += 1;
            }
            AssetListFilesystemOperation::TimelineRead => operations.timeline_reads += 1,
            AssetListFilesystemOperation::CharacterRead => operations.character_reads += 1,
            AssetListFilesystemOperation::PosterStat => operations.poster_stats += 1,
            AssetListFilesystemOperation::PosterRead => operations.poster_reads += 1,
            AssetListFilesystemOperation::IndexMarkerRead => operations.index_marker_reads += 1,
            AssetListFilesystemOperation::IndexMarkerWrite => operations.index_marker_writes += 1,
            AssetListFilesystemOperation::IndexMarkerRemove => {
                operations.index_marker_removes += 1;
            }
            AssetListFilesystemOperation::DirectoryCreateCall => {
                operations.directory_create_calls += 1;
            }
            AssetListFilesystemOperation::DbOpen => operations.db_opens += 1,
        }
    });
}

struct AssetListFilesystemOperationScope {
    active: bool,
}

impl AssetListFilesystemOperationScope {
    pub(super) fn begin() -> Self {
        ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| {
            stack
                .borrow_mut()
                .push(AssetListFilesystemOperations::default());
        });
        Self { active: true }
    }

    fn finish(mut self) -> AssetListFilesystemOperations {
        self.active = false;
        ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| stack.borrow_mut().pop().unwrap_or_default())
    }
}

impl Drop for AssetListFilesystemOperationScope {
    fn drop(&mut self) {
        if self.active {
            ASSET_LIST_FILESYSTEM_OPERATIONS.with(|stack| {
                stack.borrow_mut().pop();
            });
        }
    }
}

/// Validate the minimum structure the indexer itself requires before a cached
/// envelope can participate in hydration or de-duplication. The SQL primary key
/// is authoritative: a parseable but corrupted `asset_json.id` must never
/// impersonate another row and hide its healthy envelope via `seen_asset_ids`.
fn indexed_asset_envelope_is_valid(
    row_id: &str,
    sidecar_path: Option<&str>,
    asset: &Value,
) -> bool {
    if !is_safe_id(row_id)
        || asset.get("id").and_then(Value::as_str) != Some(row_id)
        || sidecar_path.is_some_and(|path| !is_safe_relative_path(path))
    {
        return false;
    }
    for key in ["type", "displayName", "createdAt"] {
        let Some(value) = asset.get(key).and_then(Value::as_str) else {
            return false;
        };
        if value.is_empty() {
            return false;
        }
    }
    asset
        .pointer("/file/path")
        .and_then(Value::as_str)
        .is_some_and(is_safe_relative_path)
}

impl ProjectStore {
    pub fn list_assets(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
        scope: AssetScope,
    ) -> ProjectStoreResult<Vec<Value>> {
        self.list_assets_with_filesystem_operations(
            project_id,
            include_rejected,
            include_trashed,
            scope,
        )
        .map(|(assets, _)| assets)
    }

    /// List assets and return the complete request-scoped filesystem-operation
    /// inventory used by performance tests, benchmarks, and API observability.
    ///
    /// The indexed `asset_json` envelope is authoritative on this clean path,
    /// including its embedded generation set and DB-backed poster URL.
    /// SceneWorks-owned mutations update the sidecar and index before returning.
    /// Tools that edit
    /// sidecars or generation-set JSON out of band must call
    /// [`Self::reindex_project`] (the REST equivalent is
    /// `POST /api/v1/projects/:project_id/reindex`) before expecting those edits
    /// in listings. This explicit contract includes deletion, same-size rewrites,
    /// and edits that preserve timestamps. Advertised posters are served from
    /// the same indexed DB row, so list requests do no poster filesystem work.
    pub fn list_assets_with_filesystem_operations(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
        scope: AssetScope,
    ) -> ProjectStoreResult<(Vec<Value>, AssetListFilesystemOperations)> {
        let operation_scope = AssetListFilesystemOperationScope::begin();
        let result =
            self.list_assets_filtered(project_id, include_rejected, include_trashed, scope, None);
        let operations = operation_scope.finish();
        result.map(|assets| (assets, operations))
    }

    pub(super) fn list_assets_filtered(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
        scope: AssetScope,
        asset_type: Option<&str>,
    ) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        #[cfg(test)]
        run_ensure_ready_before_lock_hook(&project_path);
        let _project_guard = lock_project_files(&project_path);
        let mut connection = connect_project_db_ready_locked(&project_path)?;
        let total = connection.query_row("select count(*) from assets", [], |row| {
            row.get::<_, i64>(0)
        })?;
        // Pre-migration projects (created before the `sidecar_path` column /
        // asset index existed) surface as EMPTY libraries even though their
        // assets are still on disk as `.sceneworks.json` sidecars (V-4). The
        // first open auto-reindexes from those sidecars so users never see a
        // silently-empty library. Idempotent: only fires when the table is
        // empty AND sidecars exist on disk, so a genuinely-empty project does
        // no work and there is no reindex loop. A failed reindex must not 500
        // the listing — fail open and return whatever the table currently
        // holds (possibly empty), which is strictly better than an error.
        if total == 0 && project_has_sidecars(&project_path) {
            if let Err(error) = reindex_project_path(project_id, &project_path, false) {
                tracing::warn!(
                    event = "asset_list_auto_reindex_failed",
                    project_id,
                    project = %project_path.display(),
                    error = %error,
                    "could not rebuild the asset index while listing; returning indexed rows"
                );
            }
            connection = connect_project_db(&project_path)?;
        }

        // The Library view hides Character Studio test outputs (sc-2024). Rows
        // with a null origin (should not occur after the schema-bump reindex)
        // fail open and stay visible.
        let origin_filter = match scope {
            AssetScope::All => " and (origin is null or origin != 'vector_workflow_intermediate')",
            AssetScope::Library => " and (origin is null or origin not in ('character_studio', 'vector_workflow_intermediate'))",
        };
        let mut statement = connection.prepare(&format!(
            "
            select id, sidecar_path, asset_json
              from assets
             where (?1 or rejected = 0)
               and (?2 or trashed = 0)
               and (?3 is null or type = ?3)
               {origin_filter}
             order by created_at desc
            "
        ))?;
        let rows = statement
            .query_map(
                params![include_rejected, include_trashed, asset_type],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        // HashSet dedup (was an O(n²) Vec linear scan) and a per-call
        // generation-set cache so a set's JSON is read once, not once per asset
        // in it (sc-4270 / F-CORE-10).
        let mut seen_asset_ids = std::collections::HashSet::new();
        let mut generation_sets = GenerationSetCache::default();
        let mut assets = Vec::new();
        for (row_id, sidecar_rel, asset_json) in rows {
            let Some(asset_json) = asset_json else {
                tracing::warn!(event = "asset_list_indexed_envelope_missing", asset_id = %row_id, "skipping asset row with no indexed envelope");
                continue;
            };
            let mut asset = match serde_json::from_str::<Value>(&asset_json) {
                Ok(asset) => asset,
                Err(error) => {
                    tracing::warn!(event = "asset_list_indexed_envelope_invalid", asset_id = %row_id, error = %error, "skipping malformed indexed asset envelope");
                    continue;
                }
            };
            if !indexed_asset_envelope_is_valid(&row_id, sidecar_rel.as_deref(), &asset) {
                tracing::warn!(event = "asset_list_indexed_envelope_invalid", asset_id = %row_id, "skipping invalid indexed asset envelope");
                continue;
            }
            if let Some(sidecar_rel) = sidecar_rel {
                if let Some(object) = asset.as_object_mut() {
                    object.insert("sidecarPath".to_owned(), Value::String(sidecar_rel));
                }
            }
            let asset = match hydrate_indexed_asset_cached(
                project_id,
                &project_path,
                asset,
                &mut generation_sets,
            ) {
                Ok(asset) => asset,
                Err(error) => {
                    tracing::warn!(event = "asset_list_indexed_envelope_hydration_failed", asset_id = %row_id, error = %error, "skipping asset whose indexed envelope could not be hydrated");
                    continue;
                }
            };
            if seen_asset_ids.insert(row_id) {
                assets.push(asset);
            }
        }
        Ok(assets)
    }

    pub fn get_asset(&self, project_id: &str, asset_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Resolve a library asset id to its absolute media file path, guarded against path traversal.
    /// Used by the saved-voice register flow to hand the reference clip to the worker embed path
    /// (sc-13517). Mirrors `get_asset`'s resolution but returns the on-disk path rather than the JSON.
    pub fn resolve_asset_media_path(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<PathBuf> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let asset = normalize_asset(project_id, &project_path, &sidecar_path)?;
        let media_rel = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if media_rel.is_empty() || !is_safe_relative_path(media_rel) {
            return Err(ProjectStoreError::BadRequest(
                "Asset media path must be project-relative".to_owned(),
            ));
        }
        crate::safe_path::resolve_local_file(&project_path, media_rel).map_err(|error| {
            ProjectStoreError::BadRequest(format!(
                "Asset media path is not a confined project file: {error}"
            ))
        })
    }

    pub(super) fn find_asset_sidecar(
        &self,
        project_path: &Path,
        asset_id: &str,
    ) -> ProjectStoreResult<PathBuf> {
        // Charset-check before any path use. This is the chokepoint every
        // asset-mutating method (get/update/delete/purge) hits first, so a
        // crafted-sidecar id like `../../x` is rejected before `delete_asset`/
        // `purge_asset` join it into a `trash/<asset_id>` dir or `remove_dir_all`
        // it (sc-4211 / F-CORE-7).
        if !is_safe_id(asset_id) {
            return Err(ProjectStoreError::BadRequest("Invalid asset id".to_owned()));
        }
        find_asset_sidecar_path(project_path, asset_id)?
            .ok_or_else(|| ProjectStoreError::NotFound("Asset not found".to_owned()))
    }
}
