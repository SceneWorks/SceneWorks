//! ProjectStore schema aggregate.

use super::*;

/// Bump whenever the project.db schema changes (new table, column, or index).
/// `apply_project_migrations` only re-runs its DDL when `PRAGMA user_version`
/// is behind this; forgetting to bump means an existing DB never gets the change.
///
/// v3: sc-2022 added the `training_datasets.character_id` column to the migration
/// but left this at 2, so any project.db already stamped at v2 hit the early-return
/// gate and never got the column — the dataset list query then failed with
/// "no such column: character_id". Bumping forces the idempotent migration to
/// replay and add the column on existing databases.
///
/// v4: sc-10117 — no schema change, but the bump forces the one-time reindex in
/// `ensure_project_db_ready`, which now heals inline-upscaled asset sidecars that
/// were missing their fold lineage (`extra.upscaledFromAssetId` / `lineage`), so
/// existing upscale pairs collapse in the Library on next open.
///
/// v5: sc-13517 — added the `saved_voices` table (Voice Clone "register a voice"
/// registry). The bump lets existing DBs pick up the new table through the gated
/// migration; the table is DB-authoritative (no sidecar), so the reindex the bump
/// triggers — which only clears the sidecar-rebuilt tables — leaves it intact.
///
/// v6: sc-14787 — rebuilt indexed asset envelopes for the authoritative clean
/// path and its durable dirty-marker repair contract.
///
/// v7: sc-14799 — materializes generation-set and poster hydration in each
/// indexed asset envelope, removing per-row filesystem reads from clean lists.
///
/// v8: sc-14799 review convergence stores poster bytes and their digest in
/// `project.db`, so advertised posters need no clean-list filesystem scan.
pub(super) const PROJECT_SCHEMA_VERSION: i64 = 8;

#[cfg(test)]
type EnsureReadyBeforeLockHook = Option<(PathBuf, std::sync::Arc<std::sync::Barrier>)>;

#[cfg(test)]
fn ensure_ready_before_lock_hooks() -> &'static Mutex<EnsureReadyBeforeLockHook> {
    static HOOK: std::sync::OnceLock<Mutex<EnsureReadyBeforeLockHook>> = std::sync::OnceLock::new();
    HOOK.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
pub(super) fn install_ensure_ready_before_lock_hook(
    project_path: PathBuf,
    barrier: std::sync::Arc<std::sync::Barrier>,
) {
    *ensure_ready_before_lock_hooks().lock() = Some((project_path, barrier));
}

#[cfg(test)]
pub(super) fn run_ensure_ready_before_lock_hook(project_path: &Path) {
    let barrier = {
        let mut hook = ensure_ready_before_lock_hooks().lock();
        match hook.as_ref() {
            Some((hook_path, _)) if hook_path == project_path => {
                hook.take().map(|(_, barrier)| barrier)
            }
            _ => None,
        }
    };
    if let Some(barrier) = barrier {
        barrier.wait();
    }
}

pub(super) fn project_schema_version(connection: &Connection) -> ProjectStoreResult<i64> {
    Ok(connection.query_row("pragma user_version", [], |row| row.get(0))?)
}

pub fn apply_project_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    if project_schema_version(connection)? >= PROJECT_SCHEMA_VERSION {
        return Ok(());
    }
    // Serialize migrations across processes, then re-read: another opener may
    // have finished the entire migration while this connection waited for the lock.
    let transaction = Transaction::new_unchecked(connection, TransactionBehavior::Immediate)?;
    if project_schema_version(&transaction)? >= PROJECT_SCHEMA_VERSION {
        return Ok(());
    }
    migrate_project_schema(&transaction)?;
    transaction.commit()?;
    Ok(())
}

fn migrate_project_schema(connection: &Connection) -> ProjectStoreResult<()> {
    connection.execute_batch(
        "
        create table if not exists project_metadata (
          key text primary key,
          value text not null
        );
        insert or ignore into project_metadata (key, value) values ('schemaVersion', '1');
        create table if not exists assets (
          id text primary key,
          type text not null,
          display_name text not null,
          file_path text not null,
          generation_set_id text,
          created_at text not null,
          favorite integer not null default 0,
          rating integer not null default 0,
          rejected integer not null default 0,
          trashed integer not null default 0
        );
        create table if not exists generation_sets (
          id text primary key,
          mode text not null,
          model text not null,
          prompt text not null,
          created_at text not null,
          job_id text
        );
        create table if not exists timelines (
          id text primary key,
          name text not null,
          file_path text not null,
          aspect_ratio text not null,
          width integer not null,
          height integer not null,
          fps integer not null,
          duration real not null default 0,
          created_at text not null,
          updated_at text not null
        );
        ",
    )?;
    ensure_column(connection, "assets", "sidecar_path", "text")?;
    // sc-2024: the originating studio (image_studio / video_studio /
    // document_studio / character_studio / upload). Existing rows are backfilled
    // by the reindex that `ensure_project_db_ready` runs on this version bump.
    ensure_column(connection, "assets", "origin", "text")?;
    // Normalized sidecar envelope + lineage keys make list hydration and
    // upscale-fold traversal indexed operations. Existing projects are
    // backfilled by the version-bump reindex below.
    ensure_column(connection, "assets", "asset_json", "text")?;
    ensure_column(connection, "assets", "source_asset_id", "text")?;
    ensure_column(connection, "assets", "upscaled_from_asset_id", "text")?;
    ensure_column(connection, "assets", "sidecar_size", "integer")?;
    ensure_column(connection, "assets", "sidecar_modified_ns", "text")?;
    ensure_column(connection, "assets", "sidecar_sha256", "text")?;
    ensure_column(connection, "assets", "poster_bytes", "blob")?;
    ensure_column(connection, "assets", "poster_sha256", "text")?;
    connection.execute_batch(
        "
        create index if not exists idx_assets_source_asset
          on assets(source_asset_id);
        create index if not exists idx_assets_upscaled_from
          on assets(upscaled_from_asset_id);
        ",
    )?;
    apply_character_migrations(connection)?;
    apply_training_dataset_migrations(connection)?;
    crate::voice_store::apply_voice_migrations(connection)?;
    // Pragma assignment cannot be parameterized; the version is a trusted const.
    connection.execute_batch(&format!("pragma user_version = {PROJECT_SCHEMA_VERSION}"))?;
    Ok(())
}

pub fn ensure_project_db_ready(project_path: &Path) -> ProjectStoreResult<()> {
    #[cfg(test)]
    run_ensure_ready_before_lock_hook(project_path);
    // Marker inspection, rebuild, orphan pruning, and marker clear must be one
    // serialized project operation. Mutation guards use the same reentrant lock,
    // so a list cannot repair and clear the marker owned by an in-flight write.
    let _project_guard = lock_project_files(project_path);
    connect_project_db_ready_locked(project_path).map(drop)
}

/// Return one ready connection while the caller holds the project lock.
///
/// The clean path keeps and reuses the connection opened for migrations/index
/// readiness, so an asset list performs one `project.db` open. A version bump or
/// dirty-marker repair rebuilds transactionally and then opens the rebuilt DB;
/// repair is intentionally outside the clean-path operation-count contract.
pub(super) fn connect_project_db_ready_locked(
    project_path: &Path,
) -> ProjectStoreResult<Connection> {
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    let indexed_version = connection
        .query_row(
            "select value from project_metadata where key = ?1",
            params![ASSET_INDEX_VERSION_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    // The schema stamp and the derived-data rebuild are separate operations.
    // Record completion only in the rebuild transaction so a failed reindex is
    // retried even though PRAGMA user_version already reached this version.
    let asset_index_dirty = asset_index_is_dirty(project_path)
        && !current_thread_owns_asset_index_mutation(project_path);
    if indexed_version.as_deref() != Some(&PROJECT_SCHEMA_VERSION.to_string()) || asset_index_dirty
    {
        drop(connection);
        // Legacy/version-bump rebuilds continue to recover sidecars even when
        // their media is temporarily unavailable. Only dirty-mutation repair
        // prunes missing-media rows before clearing the marker, closing the
        // partial-purge resurrection window.
        let project_id = read_project_summary(project_path)?.id;
        reindex_project_path(&project_id, project_path, asset_index_dirty)?;
        return connect_project_db(project_path);
    }
    Ok(connection)
}

pub(super) fn connect_project_db(project_path: &Path) -> ProjectStoreResult<Connection> {
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::DirectoryCreateCall);
    record_asset_list_filesystem_operation(AssetListFilesystemOperation::DbOpen);
    fs::create_dir_all(project_path)?;
    let connection = Connection::open(project_path.join("project.db"))?;
    crate::store_util::configure_connection(&connection, crate::store_util::Profile::Project)?;
    Ok(connection)
}

/// Open `project.db` with the shared `busy_timeout` AND run the version-gated
/// migrations, then hand back the ready connection. This is the house helper
/// external store modules (e.g. `training_store`) should use instead of a raw
/// `Connection::open(project_path.join("project.db"))`, which sets no
/// `busy_timeout` (so concurrent worker+API access hits an immediate
/// `SQLITE_BUSY` instead of waiting) and can miss migrations (sc-11202 / F-026).
/// `apply_project_migrations` is `PRAGMA user_version`-gated, so the DDL only
/// runs when the schema is actually behind — cheap on the common already-migrated
/// path.
pub(crate) fn connect_project_db_migrated(project_path: &Path) -> ProjectStoreResult<Connection> {
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    Ok(connection)
}
