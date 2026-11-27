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

/// How much of each indexed envelope a listing returns.
///
/// `Full` is the historical, default shape: every listed asset is the complete
/// envelope `get_asset` would return (recipe, embedded generation set, model
/// provenance, adapter settings). External callers (MCP tools, the film harness,
/// acceptance scripts) read those fields straight off list items, so it stays
/// the default.
///
/// `Summary` is what a catalog/grid needs: the same envelope minus the bulky
/// replay data that only an opened asset uses (see [`summarize_listed_asset`]).
/// A summary item carries `listView: "summary"`, so a client can tell it apart
/// from a full envelope and fetch `GET .../assets/:asset_id` before replaying,
/// regenerating or inspecting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssetListView {
    #[default]
    Full,
    Summary,
}

/// The marker a [`AssetListView::Summary`] item carries at `listView`.
pub const ASSET_LIST_VIEW_SUMMARY: &str = "summary";

/// `recipe.rawAdapterSettings` keys a summary keeps because catalog surfaces
/// render them on every card (face-likeness badge, VEDA attention outcome,
/// depth-map details). Each is a small object; the rest of the block (prompt
/// batches, MOSS execution reports, structured-caption trees, ...) is replay
/// data and is only on the full envelope.
const SUMMARY_RAW_ADAPTER_SETTINGS_KEYS: [&str; 4] =
    ["faceLikeness", "veda", "vedaExecution", "depth"];

/// Drop the replay-only bulk from a listed envelope, in place.
///
/// Removed: top-level and `recipe.modelProvenance`, `recipe.loras`, every
/// `recipe.rawAdapterSettings` key outside [`SUMMARY_RAW_ADAPTER_SETTINGS_KEYS`],
/// and the embedded generation set's `recipe` / `prompt` / `negativePrompt`
/// (the set's identity — id, jobId, mode, model, count, createdAt — stays).
/// Everything else is kept, notably `recipe.prompt` (catalog search),
/// `recipe.mode` / `model` / `seed` / `negativePrompt`, and the whole
/// `recipe.normalizedSettings` (character, timeline and batch membership).
pub fn summarize_listed_asset(asset: &mut Value) {
    let Some(object) = asset.as_object_mut() else {
        return;
    };
    object.remove("modelProvenance");
    if let Some(generation_set) = object
        .get_mut("generationSet")
        .and_then(Value::as_object_mut)
    {
        for key in ["recipe", "prompt", "negativePrompt"] {
            generation_set.remove(key);
        }
    }
    if let Some(recipe) = object.get_mut("recipe").and_then(Value::as_object_mut) {
        recipe.remove("loras");
        recipe.remove("modelProvenance");
        if let Some(Value::Object(mut raw)) = recipe.remove("rawAdapterSettings") {
            raw.retain(|key, _| SUMMARY_RAW_ADAPTER_SETTINGS_KEYS.contains(&key.as_str()));
            if !raw.is_empty() {
                recipe.insert("rawAdapterSettings".to_owned(), Value::Object(raw));
            }
        }
    }
    object.insert(
        "listView".to_owned(),
        Value::String(ASSET_LIST_VIEW_SUMMARY.to_owned()),
    );
}

/// The summary's SQL-side pre-prune of an indexed envelope (see `read_asset_listing`).
const SUMMARY_ENVELOPE_SQL: &str = "case when json_valid(asset_json) then json_remove(asset_json, \
     '$.modelProvenance', '$.generationSet.recipe', '$.generationSet.prompt', \
     '$.generationSet.negativePrompt', '$.recipe.loras', '$.recipe.modelProvenance') \
     else asset_json end";

fn project_listed_assets(mut assets: Vec<Value>, view: AssetListView) -> Vec<Value> {
    if view == AssetListView::Summary {
        assets.iter_mut().for_each(summarize_listed_asset);
    }
    assets
}

/// Which assets a listing includes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AssetListFilter {
    pub include_rejected: bool,
    pub include_trashed: bool,
    pub scope: AssetScope,
}

/// The result of [`ProjectStore::list_asset_changes`].
#[derive(Debug, Clone, PartialEq)]
pub struct AssetChanges {
    /// Opaque cursor to pass as `since` on the next call.
    pub cursor: String,
    /// `true` when `assets` is the complete listing (no or stale cursor): the
    /// client must REPLACE its catalog rather than merge.
    pub reset: bool,
    /// New or changed assets that match the filter, newest first.
    pub assets: Vec<Value>,
    /// Ids to drop from a merged catalog: purged, or changed out of the filter.
    /// Always empty on a reset.
    pub removed_ids: Vec<String>,
}

struct AssetListing {
    assets: Vec<Value>,
    cursor: String,
    reset: bool,
    removed_ids: Vec<String>,
}

/// `<epoch>.<seq>`: the journal epoch of the `project.db` that issued it and the
/// highest journal sequence the issuing read had seen.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AssetChangeCursor {
    epoch: String,
    seq: i64,
}

impl AssetChangeCursor {
    fn parse(cursor: &str) -> ProjectStoreResult<Self> {
        let invalid = || ProjectStoreError::BadRequest("Invalid asset change cursor".to_owned());
        let (epoch, seq) = cursor.rsplit_once('.').ok_or_else(invalid)?;
        if epoch.is_empty() || !epoch.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Err(invalid());
        }
        let seq = seq.parse::<i64>().map_err(|_| invalid())?;
        if seq < 0 {
            return Err(invalid());
        }
        Ok(Self {
            epoch: epoch.to_owned(),
            seq,
        })
    }

    fn encode(&self) -> String {
        format!("{}.{}", self.epoch, self.seq)
    }
}

type IndexedAssetRow = (String, Option<String>, Option<String>);

/// Parse, validate and hydrate indexed rows in order. Each row yields its id
/// and the served envelope, or `None` when the row cannot be served (missing,
/// malformed or invalid envelope, failed hydration, duplicate id), logged as
/// before.
fn hydrate_indexed_rows(
    project_id: &str,
    project_path: &Path,
    rows: Vec<IndexedAssetRow>,
) -> Vec<(String, Option<Value>)> {
    // HashSet dedup (was an O(n²) Vec linear scan) and a per-call
    // generation-set cache so a set's JSON is read once, not once per asset
    // in it (sc-4270 / F-CORE-10).
    let mut seen_asset_ids = std::collections::HashSet::new();
    let mut generation_sets = GenerationSetCache::default();
    let mut hydrated = Vec::with_capacity(rows.len());
    for (row_id, sidecar_rel, asset_json) in rows {
        let Some(asset_json) = asset_json else {
            tracing::warn!(event = "asset_list_indexed_envelope_missing", asset_id = %row_id, "skipping asset row with no indexed envelope");
            hydrated.push((row_id, None));
            continue;
        };
        let mut asset = match serde_json::from_str::<Value>(&asset_json) {
            Ok(asset) => asset,
            Err(error) => {
                tracing::warn!(event = "asset_list_indexed_envelope_invalid", asset_id = %row_id, error = %error, "skipping malformed indexed asset envelope");
                hydrated.push((row_id, None));
                continue;
            }
        };
        if !indexed_asset_envelope_is_valid(&row_id, sidecar_rel.as_deref(), &asset) {
            tracing::warn!(event = "asset_list_indexed_envelope_invalid", asset_id = %row_id, "skipping invalid indexed asset envelope");
            hydrated.push((row_id, None));
            continue;
        }
        if let Some(sidecar_rel) = sidecar_rel {
            if let Some(object) = asset.as_object_mut() {
                object.insert("sidecarPath".to_owned(), Value::String(sidecar_rel));
            }
        }
        let asset = match hydrate_indexed_asset_cached(
            project_id,
            project_path,
            asset,
            &mut generation_sets,
        ) {
            Ok(asset) => asset,
            Err(error) => {
                tracing::warn!(event = "asset_list_indexed_envelope_hydration_failed", asset_id = %row_id, error = %error, "skipping asset whose indexed envelope could not be hydrated");
                hydrated.push((row_id, None));
                continue;
            }
        };
        if seen_asset_ids.insert(row_id.clone()) {
            hydrated.push((row_id, Some(asset)));
        }
    }
    hydrated
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
        self.list_assets_view_with_filesystem_operations(
            project_id,
            AssetListFilter {
                include_rejected,
                include_trashed,
                scope,
            },
            AssetListView::Full,
        )
    }

    /// [`Self::list_assets_with_filesystem_operations`] in the requested
    /// [`AssetListView`]. Both views read the same indexed envelopes, so the
    /// filesystem-operation contract is identical; the summary is a projection.
    pub fn list_assets_view_with_filesystem_operations(
        &self,
        project_id: &str,
        filter: AssetListFilter,
        view: AssetListView,
    ) -> ProjectStoreResult<(Vec<Value>, AssetListFilesystemOperations)> {
        let operation_scope = AssetListFilesystemOperationScope::begin();
        let result = self
            .read_asset_listing(project_id, filter, view, None, None)
            .map(|listing| project_listed_assets(listing.assets, view));
        let operations = operation_scope.finish();
        result.map(|assets| (assets, operations))
    }

    /// The assets that changed since `since` (a cursor an earlier call returned),
    /// plus the ids a client holding that earlier listing must drop: purged
    /// assets, and assets that changed into no longer matching `filter` (for
    /// example trashed while `include_trashed` is false).
    ///
    /// Without a cursor, or with one issued by a different `project.db` (its
    /// epoch differs, e.g. the database was deleted and rebuilt), the result is
    /// the full listing with `reset: true`, so the client replaces its catalog
    /// instead of merging. Every result carries the cursor for the next call,
    /// read in the same SQLite snapshot as the rows, so a write landing between
    /// two calls is never skipped.
    ///
    /// The out-of-band sidecar edit contract of
    /// [`Self::list_assets_with_filesystem_operations`] applies unchanged: such
    /// edits surface after `reindex_project`, which re-stamps every asset.
    pub fn list_asset_changes(
        &self,
        project_id: &str,
        filter: AssetListFilter,
        view: AssetListView,
        since: Option<&str>,
    ) -> ProjectStoreResult<(AssetChanges, AssetListFilesystemOperations)> {
        let since = since
            .map(str::trim)
            .filter(|cursor| !cursor.is_empty())
            .map(AssetChangeCursor::parse)
            .transpose()?;
        let operation_scope = AssetListFilesystemOperationScope::begin();
        let result = self.read_asset_listing(project_id, filter, view, None, since.as_ref());
        let operations = operation_scope.finish();
        result.map(|listing| {
            (
                AssetChanges {
                    cursor: listing.cursor,
                    reset: listing.reset,
                    assets: project_listed_assets(listing.assets, view),
                    removed_ids: listing.removed_ids,
                },
                operations,
            )
        })
    }

    pub(super) fn list_assets_filtered(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
        scope: AssetScope,
        asset_type: Option<&str>,
    ) -> ProjectStoreResult<Vec<Value>> {
        self.read_asset_listing(
            project_id,
            AssetListFilter {
                include_rejected,
                include_trashed,
                scope,
            },
            AssetListView::Full,
            asset_type,
            None,
        )
        .map(|listing| listing.assets)
    }

    /// One read of the indexed catalog: the whole filtered listing, or — when
    /// `since` is a cursor this database issued — only the rows journaled after
    /// it. Rows, tombstones and the returned cursor come from one read snapshot.
    fn read_asset_listing(
        &self,
        project_id: &str,
        filter: AssetListFilter,
        view: AssetListView,
        asset_type: Option<&str>,
        since: Option<&AssetChangeCursor>,
    ) -> ProjectStoreResult<AssetListing> {
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

        // One deferred read transaction: the journal head, the rows and the
        // tombstones are all read from the same snapshot.
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let epoch = transaction
            .query_row(
                "select value from project_metadata where key = ?1",
                params![ASSET_CHANGE_EPOCH_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .unwrap_or_default();
        let head = transaction.query_row(
            "select coalesce(max(seq), 0) from asset_changes",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        let cursor = AssetChangeCursor {
            epoch: epoch.clone(),
            seq: head,
        }
        .encode();
        let incremental_from = since
            .filter(|since| !epoch.is_empty() && since.epoch == epoch && since.seq <= head)
            .map(|since| since.seq);

        // The Library view hides Character Studio test outputs (sc-2024). Rows
        // with a null origin (should not occur after the schema-bump reindex)
        // fail open and stay visible.
        let origin_filter = match filter.scope {
            AssetScope::All => "(origin is null or origin != 'vector_workflow_intermediate')",
            AssetScope::Library => "(origin is null or origin not in ('character_studio', 'vector_workflow_intermediate'))",
        };
        let visible = format!(
            "(?1 or rejected = 0) and (?2 or trashed = 0) and (?3 is null or type = ?3) and {origin_filter}"
        );
        // A summary never carries the replay bulk, so SQLite drops the largest parts before
        // the envelope is handed to serde (on the measured 15k-asset catalog they are ~2/3 of
        // the bytes). Only parts the options upgrade never reads are pruned here — it does not
        // descend into `loras` or provenance, and the embedded set's recipe is dropped whole —
        // so the upgraded summary is identical to summarizing the upgraded full envelope.
        // Malformed JSON is passed through untouched for the per-row validation below to skip.
        let envelope = match view {
            AssetListView::Full => "asset_json",
            AssetListView::Summary => SUMMARY_ENVELOPE_SQL,
        };
        let mut removed_ids = Vec::new();
        let rows = match incremental_from {
            None => {
                let mut statement = transaction.prepare(&format!(
                    "
                    select id, sidecar_path, {envelope}
                      from assets
                     where {visible}
                     order by created_at desc
                    "
                ))?;
                let rows = statement
                    .query_map(
                        params![filter.include_rejected, filter.include_trashed, asset_type],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, Option<String>>(1)?,
                                row.get::<_, Option<String>>(2)?,
                            ))
                        },
                    )?
                    .collect::<Result<Vec<_>, _>>()?;
                rows
            }
            Some(from) => {
                // Every journaled id after the cursor: still-visible rows are
                // returned, everything else (purged, or no longer matching the
                // filter) is reported for removal.
                let mut statement = transaction.prepare(&format!(
                    "
                    select changes.asset_id, assets.sidecar_path, {envelope},
                           assets.id is not null and {visible}
                      from asset_changes changes
                      left join assets on assets.id = changes.asset_id
                     where changes.seq > ?4
                     order by assets.created_at desc
                    "
                ))?;
                let changed = statement
                    .query_map(
                        params![
                            filter.include_rejected,
                            filter.include_trashed,
                            asset_type,
                            from
                        ],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, Option<String>>(1)?,
                                row.get::<_, Option<String>>(2)?,
                                row.get::<_, Option<bool>>(3)?.unwrap_or(false),
                            ))
                        },
                    )?
                    .collect::<Result<Vec<_>, _>>()?;
                let mut rows = Vec::with_capacity(changed.len());
                for (id, sidecar_path, asset_json, is_visible) in changed {
                    if is_visible {
                        rows.push((id, sidecar_path, asset_json));
                    } else {
                        removed_ids.push(id);
                    }
                }
                rows
            }
        };
        transaction.finish()?;
        let mut assets = Vec::with_capacity(rows.len());
        for (row_id, asset) in hydrate_indexed_rows(project_id, &project_path, rows) {
            match asset {
                Some(asset) => assets.push(asset),
                // A changed row whose envelope can no longer be served must not
                // linger in an incremental client's catalog.
                None if incremental_from.is_some() => removed_ids.push(row_id),
                None => {}
            }
        }
        Ok(AssetListing {
            assets,
            cursor,
            reset: incremental_from.is_none(),
            removed_ids,
        })
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
