use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
use std::time::Duration;

use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter, Connection, ErrorCode, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::store_util::lock_store_path;
use crate::time::utc_now;

pub const CATALOG_SCHEMA_VERSION: u32 = 1;
pub const CATALOG_DATABASE_FILE: &str = "catalog.sqlite";
pub const CATALOG_MANIFEST_FILE: &str = "catalog.json";
pub const CATALOG_REGISTRY_FILE: &str = "attached-catalogs.json";
pub const CATALOG_ARTIFACT_DIRECTORIES: &[&str] =
    &["images", "thumbnails", "embeddings", "artifacts"];

pub fn catalog_timestamp_now() -> String {
    utc_now()
}

const CATALOG_APPLICATION_ID: i32 = 0x5343_5743;
const REGISTRY_SCHEMA_VERSION: u32 = 1;
const MAX_REGISTRY_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PAGE_SIZE: u32 = 10_000;
const MAX_CONTRACT_STATE_BYTES: usize = 1024 * 1024;
const MAX_QUERY_FILTERS: usize = 16;
const MAX_FILTER_VALUES: usize = 64;
const MAX_FACET_FIELDS: usize = 16;
const MAX_FACET_VALUES: u32 = 200;
const MAX_FACET_VALUE_BYTES: u32 = 512;

const SOURCE_CONFIG_METADATA_KEY: &str = "source_config";
const ANALYZER_VERSIONS_METADATA_KEY: &str = "analyzer_versions";
const CHECKPOINTS_METADATA_KEY: &str = "checkpoints";
const PROGRESS_METADATA_KEY: &str = "processing_progress";
const PROCESSING_CONTROL_METADATA_KEY: &str = "processing_control";
const ANALYZER_CONFIG_METADATA_KEY: &str = "analyzer_config";
const PROCESSING_LEASE_FILE: &str = ".catalog-processing.lock";
const CATALOG_ID_METADATA_KEY: &str = "catalog_id";
const CATALOG_NAME_METADATA_KEY: &str = "name";
const CATALOG_CREATED_AT_METADATA_KEY: &str = "created_at";
const CATALOG_SCHEMA_VERSION_METADATA_KEY: &str = "schema_version";
const RESERVED_CATALOG_METADATA_KEYS: &[&str] = &[
    CATALOG_ID_METADATA_KEY,
    CATALOG_NAME_METADATA_KEY,
    CATALOG_CREATED_AT_METADATA_KEY,
    CATALOG_SCHEMA_VERSION_METADATA_KEY,
    SOURCE_CONFIG_METADATA_KEY,
    ANALYZER_VERSIONS_METADATA_KEY,
    CHECKPOINTS_METADATA_KEY,
    PROGRESS_METADATA_KEY,
    PROCESSING_CONTROL_METADATA_KEY,
    ANALYZER_CONFIG_METADATA_KEY,
];

pub type CatalogResult<T> = Result<T, CatalogError>;

#[derive(Debug)]
pub enum CatalogError {
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    InvalidCatalog(String),
    NotFound(String),
    AlreadyExists(String),
    Conflict(String),
    Corrupt { path: PathBuf, detail: String },
    Incompatible { found: u32, supported: u32 },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Sqlite(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::InvalidCatalog(detail) => write!(formatter, "{detail}"),
            Self::NotFound(detail) => write!(formatter, "{detail}"),
            Self::AlreadyExists(detail) => write!(formatter, "{detail}"),
            Self::Conflict(detail) => write!(formatter, "{detail}"),
            Self::Corrupt { path, detail } => {
                write!(
                    formatter,
                    "Catalog at {} is corrupt: {detail}",
                    path.display()
                )
            }
            Self::Incompatible { found, supported } => write!(
                formatter,
                "Catalog schema version {found} is newer than supported version {supported}"
            ),
        }
    }
}

impl std::error::Error for CatalogError {}

impl From<std::io::Error> for CatalogError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for CatalogError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogDescriptor {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub schema_version: u32,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachedCatalog {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub attached_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewCatalogRecord {
    pub id: String,
    pub image_path: String,
    pub thumbnail_path: Option<String>,
    pub embedding_path: Option<String>,
    pub artifact_path: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogRecord {
    pub id: String,
    pub image_path: String,
    pub thumbnail_path: Option<String>,
    pub embedding_path: Option<String>,
    pub artifact_path: Option<String>,
    pub metadata: Value,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatalogRecordPage {
    pub records: Vec<CatalogRecord>,
    pub next_cursor: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogStorageAccounting {
    pub database_bytes: u64,
    pub manifest_bytes: u64,
    pub artifact_bytes: u64,
    pub total_bytes: u64,
    pub record_count: u64,
}

/// Stable, forward-compatible source configuration stored inside the catalog.
/// Source paths are canonical absolute paths selected by the user; arbitrary
/// options stay source-kind specific without changing the lifecycle contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogSourceConfig {
    pub kind: String,
    pub paths: Vec<PathBuf>,
    #[serde(default = "empty_json_object")]
    pub options: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CatalogProcessingState {
    #[default]
    Idle,
    Running,
    Paused,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CatalogProcessingDesiredState {
    #[default]
    Running,
    Paused,
}

/// Narrow, revisioned user intent for cooperative catalog processors.
///
/// This is deliberately stored separately from [`CatalogContractState`]. A
/// pause/resume request can therefore never replace a scanner checkpoint,
/// analyzer version, source configuration, or freshly-published progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogProcessingControl {
    pub desired_state: CatalogProcessingDesiredState,
    pub revision: u64,
    pub updated_at: String,
}

impl Default for CatalogProcessingControl {
    fn default() -> Self {
        Self {
            desired_state: CatalogProcessingDesiredState::Running,
            revision: 0,
            updated_at: utc_now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogAnalyzerThresholds {
    pub person_min_confidence: f64,
    pub face_min_confidence: f64,
    pub pose_min_keypoint_confidence: f64,
    pub prominent_frame_fraction: f64,
    pub frame_edge_margin: f64,
    pub min_pose_coverage: f64,
}

impl Default for CatalogAnalyzerThresholds {
    fn default() -> Self {
        Self {
            person_min_confidence: 0.25,
            face_min_confidence: 0.65,
            pose_min_keypoint_confidence: 0.3,
            prominent_frame_fraction: 0.2,
            frame_edge_margin: 0.01,
            min_pose_coverage: 0.72,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogAnalyzerSettings {
    pub structured_analysis_enabled: bool,
    pub vision_analysis_enabled: bool,
    pub semantic_embeddings_enabled: bool,
    pub thresholds: CatalogAnalyzerThresholds,
}

impl Default for CatalogAnalyzerSettings {
    fn default() -> Self {
        Self {
            structured_analysis_enabled: true,
            vision_analysis_enabled: false,
            semantic_embeddings_enabled: false,
            thresholds: CatalogAnalyzerThresholds::default(),
        }
    }
}

/// Revisioned catalog-owned analyzer intent. Model versions and run results
/// remain processor-owned; these settings are durable user choices that later
/// orchestration can consume without racing scanner/analyzer checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogAnalyzerConfig {
    pub settings: CatalogAnalyzerSettings,
    pub revision: u64,
    pub updated_at: String,
}

impl Default for CatalogAnalyzerConfig {
    fn default() -> Self {
        Self {
            settings: CatalogAnalyzerSettings::default(),
            revision: 0,
            updated_at: utc_now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogProcessingProgress {
    pub state: CatalogProcessingState,
    pub candidate_count: u64,
    pub processed_count: u64,
    pub accepted_count: u64,
    pub rejected_count: u64,
    pub error_count: u64,
    pub message: Option<String>,
    pub updated_at: String,
}

impl Default for CatalogProcessingProgress {
    fn default() -> Self {
        Self {
            state: CatalogProcessingState::Idle,
            candidate_count: 0,
            processed_count: 0,
            accepted_count: 0,
            rejected_count: 0,
            error_count: 0,
            message: None,
            updated_at: utc_now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CatalogContractState {
    pub source_config: Option<CatalogSourceConfig>,
    #[serde(default)]
    pub analyzer_versions: BTreeMap<String, String>,
    #[serde(default)]
    pub checkpoints: BTreeMap<String, Value>,
    pub processing: CatalogProcessingProgress,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogRecordFilter {
    pub field: String,
    pub values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogFacetCount {
    pub value: String,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogFacet {
    pub field: String,
    pub values: Vec<CatalogFacetCount>,
}

/// Cross-process exclusive lease for all catalog processing and lifecycle
/// operations that must not overlap it.
///
/// The handle releases automatically on drop, including process exit. Scanner
/// implementations hold it for their complete bounded pass; detach/delete
/// acquire the same lease before changing registry or disk state.
#[derive(Debug)]
pub struct CatalogProcessingLease {
    _file: File,
}

impl CatalogProcessingLease {
    pub fn try_acquire(catalog: &Catalog) -> CatalogResult<Self> {
        use fs2::FileExt as _;

        let lock_path = catalog.root.join(PROCESSING_LEASE_FILE);
        if let Ok(metadata) = fs::symlink_metadata(&lock_path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(CatalogError::InvalidCatalog(
                    "Catalog processing lock must be a regular file".to_owned(),
                ));
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)?;
        let contended = fs2::lock_contended_error().raw_os_error();
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Self { _file: file }),
            Err(error) if error.raw_os_error() == contended => Err(CatalogError::Conflict(
                "Catalog processing is active".to_owned(),
            )),
            Err(error) => Err(CatalogError::Io(error)),
        }
    }

    pub fn is_active(catalog: &Catalog) -> CatalogResult<bool> {
        match Self::try_acquire(catalog) {
            Ok(lease) => {
                drop(lease);
                Ok(false)
            }
            Err(CatalogError::Conflict(_)) => Ok(true),
            Err(error) => Err(error),
        }
    }

    /// Reconciles a processor state left behind by a crashed process.
    ///
    /// Holding this lease proves that no live processor owns the catalog. The
    /// update is deliberately limited to the progress metadata row so scanner
    /// checkpoints, source configuration, analyzers, and user intent survive.
    pub fn reconcile_interrupted(&self, catalog: &Catalog) -> CatalogResult<bool> {
        catalog.reconcile_interrupted_processing()
    }
}

#[derive(Debug)]
pub struct Catalog {
    root: PathBuf,
    descriptor: CatalogDescriptor,
    connection: Connection,
}

impl Catalog {
    pub fn create(root: impl AsRef<Path>, name: impl Into<String>) -> CatalogResult<Self> {
        let root = root.as_ref();
        let name = validated_name(name.into())?;
        prepare_new_root(root)?;
        let root = fs::canonicalize(root)?;
        for directory in CATALOG_ARTIFACT_DIRECTORIES {
            fs::create_dir_all(root.join(directory))?;
        }

        let descriptor = CatalogDescriptor {
            id: random_hex(16)?,
            name,
            path: root.clone(),
            schema_version: CATALOG_SCHEMA_VERSION,
            created_at: utc_now(),
        };
        write_manifest(&root, &descriptor)?;

        let database_path = root.join(CATALOG_DATABASE_FILE);
        let connection = Connection::open(&database_path)
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        configure_connection(&connection, &database_path)?;
        connection
            .pragma_update(None, "application_id", CATALOG_APPLICATION_ID)
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        migrate(&connection, &database_path)?;
        write_database_metadata(&connection, &database_path, &descriptor)?;

        Ok(Self {
            root,
            descriptor,
            connection,
        })
    }

    pub fn open(root: impl AsRef<Path>) -> CatalogResult<Self> {
        let root = canonical_catalog_root(root.as_ref())?;
        let descriptor = read_manifest(&root)?;
        if descriptor.schema_version > CATALOG_SCHEMA_VERSION {
            return Err(CatalogError::Incompatible {
                found: descriptor.schema_version,
                supported: CATALOG_SCHEMA_VERSION,
            });
        }

        let database_path = root.join(CATALOG_DATABASE_FILE);
        if !database_path.is_file() {
            return Err(CatalogError::InvalidCatalog(format!(
                "Catalog database is missing at {}",
                database_path.display()
            )));
        }
        let connection = Connection::open(&database_path)
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        configure_connection(&connection, &database_path)?;
        let application_id: i32 = connection
            .pragma_query_value(None, "application_id", |row| row.get(0))
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        if application_id != CATALOG_APPLICATION_ID {
            return Err(CatalogError::Corrupt {
                path: database_path,
                detail: "database does not have the SceneWorks catalog application id".to_owned(),
            });
        }
        migrate(&connection, &database_path)?;
        validate_schema(&connection, &database_path)?;
        validate_database_metadata(&connection, &database_path, &descriptor)?;

        Ok(Self {
            root,
            descriptor,
            connection,
        })
    }

    pub fn descriptor(&self) -> &CatalogDescriptor {
        &self.descriptor
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn database_path(&self) -> PathBuf {
        self.root.join(CATALOG_DATABASE_FILE)
    }

    pub fn artifact_directory(&self, kind: &str) -> CatalogResult<PathBuf> {
        if !CATALOG_ARTIFACT_DIRECTORIES.contains(&kind) {
            return Err(CatalogError::InvalidCatalog(format!(
                "Unknown catalog artifact kind: {kind}"
            )));
        }
        Ok(self.root.join(kind))
    }

    pub fn append_records(&mut self, records: &[NewCatalogRecord]) -> CatalogResult<usize> {
        self.append_records_and_metadata(records, &[])
    }

    /// Appends a bounded record batch and advances scanner metadata in the same
    /// SQLite transaction. A process crash can therefore only leave both the
    /// records and checkpoint committed, or neither committed.
    pub fn append_records_and_metadata(
        &mut self,
        records: &[NewCatalogRecord],
        metadata: &[(&str, &str)],
    ) -> CatalogResult<usize> {
        for record in records {
            validate_record(record)?;
        }
        for (key, value) in metadata {
            validate_metadata_entry(key, value)?;
        }
        let database_path = self.database_path();
        let transaction = self
            .connection
            .transaction()
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        {
            let mut statement = transaction
                .prepare_cached(
                    "insert into catalog_records (
                        id, image_path, thumbnail_path, embedding_path, artifact_path,
                        metadata_json, created_at, updated_at
                     ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
                     on conflict(id) do update set
                        image_path = excluded.image_path,
                        thumbnail_path = excluded.thumbnail_path,
                        embedding_path = excluded.embedding_path,
                        artifact_path = excluded.artifact_path,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at",
                )
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            for record in records {
                let metadata = serde_json::to_string(&record.metadata)?;
                statement
                    .execute(params![
                        record.id,
                        record.image_path,
                        record.thumbnail_path,
                        record.embedding_path,
                        record.artifact_path,
                        metadata,
                        utc_now(),
                    ])
                    .map_err(|error| map_sqlite_error(&database_path, error))?;
            }
        }
        {
            let mut statement = transaction
                .prepare_cached(
                    "insert into catalog_metadata(key, value) values (?1, ?2)
                     on conflict(key) do update set value = excluded.value",
                )
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            for (key, value) in metadata {
                statement
                    .execute(params![key, value])
                    .map_err(|error| map_sqlite_error(&database_path, error))?;
            }
        }
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        Ok(records.len())
    }

    pub fn metadata(&self, key: &str) -> CatalogResult<Option<String>> {
        validate_metadata_key(key)?;
        self.connection
            .query_row(
                "select value from catalog_metadata where key = ?1",
                [key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| map_sqlite_error(&self.database_path(), error))
    }

    pub fn contains_record_id(&self, record_id: &str) -> CatalogResult<bool> {
        if record_id.trim().is_empty() || record_id.len() > 512 || record_id.contains('\0') {
            return Err(CatalogError::InvalidCatalog(
                "Catalog record id must contain 1 to 512 non-NUL bytes".to_owned(),
            ));
        }
        self.connection
            .query_row(
                "select exists(select 1 from catalog_records where id = ?1)",
                [record_id],
                |row| row.get(0),
            )
            .map_err(|error| map_sqlite_error(&self.database_path(), error))
    }

    pub fn page_records(&self, offset: u64, limit: u32) -> CatalogResult<Vec<CatalogRecord>> {
        validate_page_size(limit)?;
        if offset > i64::MAX as u64 {
            return Err(CatalogError::InvalidCatalog(
                "Catalog page offset is too large".to_owned(),
            ));
        }
        let database_path = self.database_path();
        let mut statement = self
            .connection
            .prepare_cached(
                "select id, image_path, thumbnail_path, embedding_path, artifact_path,
                        metadata_json, created_at, updated_at
                 from catalog_records
                 order by rowid
                 limit ?1 offset ?2",
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let rows = statement
            .query_map(params![limit, offset], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                ))
            })
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let mut records = Vec::new();
        for row in rows {
            let (
                id,
                image_path,
                thumbnail_path,
                embedding_path,
                artifact_path,
                metadata_json,
                created_at,
                updated_at,
            ) = row.map_err(|error| map_sqlite_error(&database_path, error))?;
            let metadata =
                serde_json::from_str(&metadata_json).map_err(|error| CatalogError::Corrupt {
                    path: database_path.clone(),
                    detail: format!("record {id} contains invalid metadata JSON: {error}"),
                })?;
            records.push(CatalogRecord {
                id,
                image_path,
                thumbnail_path,
                embedding_path,
                artifact_path,
                metadata,
                created_at,
                updated_at,
            });
        }
        Ok(records)
    }

    /// Reads a scale-stable page using SQLite's monotonically increasing rowid as
    /// an opaque cursor. Unlike `OFFSET`, the index can seek directly to the next
    /// row even when a catalog contains millions of records.
    pub fn page_records_after(
        &self,
        after_cursor: Option<i64>,
        limit: u32,
    ) -> CatalogResult<CatalogRecordPage> {
        self.query_records_after(after_cursor, limit, &[])
    }

    /// Reads a bounded, filtered keyset page. Filter fields address top-level or
    /// dotted metadata keys and are always passed to SQLite as bound JSON paths;
    /// neither field names nor values are interpolated into SQL.
    pub fn query_records_after(
        &self,
        after_cursor: Option<i64>,
        limit: u32,
        filters: &[CatalogRecordFilter],
    ) -> CatalogResult<CatalogRecordPage> {
        validate_page_size(limit)?;
        validate_filters(filters)?;
        let database_path = self.database_path();
        let cursor = after_cursor.unwrap_or(0);
        if cursor < 0 {
            return Err(CatalogError::InvalidCatalog(
                "Catalog page cursor cannot be negative".to_owned(),
            ));
        }
        let mut sql = String::from(
            "select rowid, id, image_path, thumbnail_path, embedding_path, artifact_path,
                    metadata_json, created_at, updated_at
             from catalog_records
             where rowid > ?",
        );
        let mut bindings = vec![SqlValue::Integer(cursor)];
        append_filter_sql(&mut sql, &mut bindings, filters);
        sql.push_str(" order by rowid limit ?");
        bindings.push(SqlValue::Integer(i64::from(limit) + 1));
        let mut statement = self
            .connection
            .prepare(&sql)
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let rows = statement
            .query_map(params_from_iter(bindings), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                ))
            })
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let mut records = Vec::new();
        let mut cursors = Vec::new();
        for row in rows {
            let (
                rowid,
                id,
                image_path,
                thumbnail_path,
                embedding_path,
                artifact_path,
                metadata_json,
                created_at,
                updated_at,
            ) = row.map_err(|error| map_sqlite_error(&database_path, error))?;
            let metadata =
                serde_json::from_str(&metadata_json).map_err(|error| CatalogError::Corrupt {
                    path: database_path.clone(),
                    detail: format!("record {id} contains invalid metadata JSON: {error}"),
                })?;
            cursors.push(rowid);
            records.push(CatalogRecord {
                id,
                image_path,
                thumbnail_path,
                embedding_path,
                artifact_path,
                metadata,
                created_at,
                updated_at,
            });
        }
        let has_more = records.len() > limit as usize;
        if has_more {
            records.pop();
            cursors.pop();
        }
        let next_cursor = has_more.then(|| {
            *cursors
                .last()
                .expect("a page with a lookahead row has a returned row")
        });
        Ok(CatalogRecordPage {
            records,
            next_cursor,
        })
    }

    /// Computes bounded facet buckets in SQLite. The database may scan matching
    /// rows, but the API process retains only `max_values_per_facet` buckets per
    /// requested field and never materializes the result set.
    pub fn facet_counts(
        &self,
        fields: &[String],
        filters: &[CatalogRecordFilter],
        max_values_per_facet: u32,
    ) -> CatalogResult<Vec<CatalogFacet>> {
        if fields.is_empty() || fields.len() > MAX_FACET_FIELDS {
            return Err(CatalogError::InvalidCatalog(format!(
                "Catalog facets require 1 to {MAX_FACET_FIELDS} fields"
            )));
        }
        if max_values_per_facet == 0 || max_values_per_facet > MAX_FACET_VALUES {
            return Err(CatalogError::InvalidCatalog(format!(
                "Catalog facet size must be between 1 and {MAX_FACET_VALUES}"
            )));
        }
        validate_filters(filters)?;
        for field in fields {
            metadata_json_path(field)?;
        }

        let database_path = self.database_path();
        let mut facets = Vec::with_capacity(fields.len());
        for field in fields {
            let json_path = metadata_json_path(field)?;
            let mut sql = String::from(
                "select cast(json_extract(metadata_json, ?) as text) as facet_value,
                        count(*) as facet_count
                 from catalog_records
                 where json_type(metadata_json, ?) is not null
                   and length(cast(json_extract(metadata_json, ?) as blob))
                       between 1 and ?",
            );
            let mut bindings = vec![
                SqlValue::Text(json_path.clone()),
                SqlValue::Text(json_path.clone()),
                SqlValue::Text(json_path),
                SqlValue::Integer(i64::from(MAX_FACET_VALUE_BYTES)),
            ];
            append_filter_sql(&mut sql, &mut bindings, filters);
            sql.push_str(
                " group by facet_value
                  order by facet_count desc, facet_value asc
                  limit ?",
            );
            bindings.push(SqlValue::Integer(i64::from(max_values_per_facet)));
            let mut statement = self
                .connection
                .prepare(&sql)
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            let rows = statement
                .query_map(params_from_iter(bindings), |row| {
                    Ok(CatalogFacetCount {
                        value: row.get(0)?,
                        count: row.get(1)?,
                    })
                })
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            let values = rows
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            facets.push(CatalogFacet {
                field: field.clone(),
                values,
            });
        }
        Ok(facets)
    }

    pub fn contract_state(&self) -> CatalogResult<CatalogContractState> {
        let mut processing: CatalogProcessingProgress = self
            .read_contract_value(PROGRESS_METADATA_KEY)?
            .unwrap_or_else(|| CatalogProcessingProgress {
                updated_at: self.descriptor.created_at.clone(),
                ..CatalogProcessingProgress::default()
            });
        if processing.updated_at.is_empty() {
            processing.updated_at = self.descriptor.created_at.clone();
        }
        Ok(CatalogContractState {
            source_config: self.read_contract_value(SOURCE_CONFIG_METADATA_KEY)?,
            analyzer_versions: self
                .read_contract_value(ANALYZER_VERSIONS_METADATA_KEY)?
                .unwrap_or_default(),
            checkpoints: self
                .read_contract_value(CHECKPOINTS_METADATA_KEY)?
                .unwrap_or_default(),
            processing,
        })
    }

    /// Persists the source/analyzer/checkpoint/progress snapshot transactionally.
    /// Scanner and analyzer stories can update this stable contract without
    /// touching SceneWorks' jobs or project databases.
    pub fn set_contract_state(&self, state: &CatalogContractState) -> CatalogResult<()> {
        let mut normalized = state.clone();
        if let Some(source) = normalized.source_config.as_mut() {
            normalize_source_config(source)?;
        }
        validate_contract_state(&normalized)?;
        let database_path = self.database_path();
        let transaction = self
            .connection
            .unchecked_transaction()
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        match &normalized.source_config {
            Some(source_config) => {
                transaction
                    .execute(
                        "insert or replace into catalog_metadata(key, value) values (?1, ?2)",
                        params![
                            SOURCE_CONFIG_METADATA_KEY,
                            serde_json::to_string(source_config)?
                        ],
                    )
                    .map_err(|error| map_sqlite_error(&database_path, error))?;
            }
            None => {
                transaction
                    .execute(
                        "delete from catalog_metadata where key = ?1",
                        [SOURCE_CONFIG_METADATA_KEY],
                    )
                    .map_err(|error| map_sqlite_error(&database_path, error))?;
            }
        }
        for (key, value) in [
            (
                ANALYZER_VERSIONS_METADATA_KEY,
                serde_json::to_string(&normalized.analyzer_versions)?,
            ),
            (
                CHECKPOINTS_METADATA_KEY,
                serde_json::to_string(&normalized.checkpoints)?,
            ),
            (
                PROGRESS_METADATA_KEY,
                serde_json::to_string(&normalized.processing)?,
            ),
        ] {
            transaction
                .execute(
                    "insert or replace into catalog_metadata(key, value) values (?1, ?2)",
                    params![key, value],
                )
                .map_err(|error| map_sqlite_error(&database_path, error))?;
        }
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&database_path, error))
    }

    pub fn processing_control(&self) -> CatalogResult<CatalogProcessingControl> {
        self.read_contract_value(PROCESSING_CONTROL_METADATA_KEY)
            .map(|control| {
                control.unwrap_or_else(|| CatalogProcessingControl {
                    updated_at: self.descriptor.created_at.clone(),
                    ..CatalogProcessingControl::default()
                })
            })
    }

    pub fn analyzer_config(&self) -> CatalogResult<CatalogAnalyzerConfig> {
        self.read_contract_value(ANALYZER_CONFIG_METADATA_KEY)
            .map(|config| {
                config.unwrap_or_else(|| CatalogAnalyzerConfig {
                    updated_at: self.descriptor.created_at.clone(),
                    ..CatalogAnalyzerConfig::default()
                })
            })
    }

    /// Atomically advances user analyzer settings without rewriting processor
    /// provenance, checkpoints, or progress.
    pub fn request_analyzer_config(
        &self,
        expected_revision: u64,
        settings: CatalogAnalyzerSettings,
    ) -> CatalogResult<CatalogAnalyzerConfig> {
        validate_analyzer_settings(&settings)?;
        let database_path = self.database_path();
        let default = CatalogAnalyzerConfig {
            updated_at: self.descriptor.created_at.clone(),
            ..CatalogAnalyzerConfig::default()
        };
        self.connection
            .execute(
                "insert or ignore into catalog_metadata(key, value) values (?1, ?2)",
                params![
                    ANALYZER_CONFIG_METADATA_KEY,
                    serde_json::to_string(&default)?
                ],
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let current_payload: String = self
            .connection
            .query_row(
                "select value from catalog_metadata where key = ?1",
                [ANALYZER_CONFIG_METADATA_KEY],
                |row| row.get(0),
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let current = serde_json::from_str::<CatalogAnalyzerConfig>(&current_payload)?;
        if current.revision != expected_revision {
            return Err(CatalogError::Conflict(
                "Catalog analyzer configuration changed; refresh and retry".to_owned(),
            ));
        }
        let next = CatalogAnalyzerConfig {
            settings,
            revision: current.revision.checked_add(1).ok_or_else(|| {
                CatalogError::Conflict("Catalog analyzer revision is exhausted".to_owned())
            })?,
            updated_at: utc_now(),
        };
        let changed = self
            .connection
            .execute(
                "update catalog_metadata set value = ?1 where key = ?2 and value = ?3",
                params![
                    serde_json::to_string(&next)?,
                    ANALYZER_CONFIG_METADATA_KEY,
                    current_payload
                ],
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        if changed != 1 {
            return Err(CatalogError::Conflict(
                "Catalog analyzer configuration changed; refresh and retry".to_owned(),
            ));
        }
        Ok(next)
    }

    /// Atomically advances user processing intent without reading or rewriting
    /// any other catalog contract field.
    pub fn request_processing_control(
        &self,
        expected_revision: u64,
        desired_state: CatalogProcessingDesiredState,
    ) -> CatalogResult<CatalogProcessingControl> {
        let database_path = self.database_path();
        let default = CatalogProcessingControl {
            updated_at: self.descriptor.created_at.clone(),
            ..CatalogProcessingControl::default()
        };
        self.connection
            .execute(
                "insert or ignore into catalog_metadata(key, value) values (?1, ?2)",
                params![
                    PROCESSING_CONTROL_METADATA_KEY,
                    serde_json::to_string(&default)?
                ],
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let current_payload: String = self
            .connection
            .query_row(
                "select value from catalog_metadata where key = ?1",
                [PROCESSING_CONTROL_METADATA_KEY],
                |row| row.get(0),
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let current = serde_json::from_str::<CatalogProcessingControl>(&current_payload)?;
        if current.revision != expected_revision {
            return Err(CatalogError::Conflict(
                "Catalog processing control changed; refresh and retry".to_owned(),
            ));
        }
        let progress_payload: Option<String> = self
            .connection
            .query_row(
                "select value from catalog_metadata where key = ?1",
                [PROGRESS_METADATA_KEY],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let actual = progress_payload
            .map(|payload| serde_json::from_str::<CatalogProcessingProgress>(&payload))
            .transpose()?
            .unwrap_or_default();
        let valid = matches!(
            (current.desired_state, actual.state, desired_state),
            (
                CatalogProcessingDesiredState::Running,
                CatalogProcessingState::Running,
                CatalogProcessingDesiredState::Paused
            ) | (
                CatalogProcessingDesiredState::Paused,
                CatalogProcessingState::Paused,
                CatalogProcessingDesiredState::Running
            ) | (
                _,
                CatalogProcessingState::Failed,
                CatalogProcessingDesiredState::Running
            )
        );
        if !valid {
            return Err(CatalogError::Conflict(
                "Catalog processing state does not allow that transition".to_owned(),
            ));
        }
        let next = CatalogProcessingControl {
            desired_state,
            revision: current.revision.checked_add(1).ok_or_else(|| {
                CatalogError::Conflict("Catalog processing revision is exhausted".to_owned())
            })?,
            updated_at: utc_now(),
        };
        let changed = self
            .connection
            .execute(
                "update catalog_metadata set value = ?1 where key = ?2 and value = ?3",
                params![
                    serde_json::to_string(&next)?,
                    PROCESSING_CONTROL_METADATA_KEY,
                    current_payload
                ],
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        if changed != 1 {
            return Err(CatalogError::Conflict(
                "Catalog processing control changed; refresh and retry".to_owned(),
            ));
        }
        Ok(next)
    }

    /// Publishes actual processor progress without replacing source, analyzer,
    /// checkpoint, or desired-control metadata.
    pub fn set_processing_progress(
        &self,
        progress: &CatalogProcessingProgress,
    ) -> CatalogResult<()> {
        validate_processing_progress(progress)?;
        let database_path = self.database_path();
        self.connection
            .execute(
                "insert or replace into catalog_metadata(key, value) values (?1, ?2)",
                params![PROGRESS_METADATA_KEY, serde_json::to_string(progress)?],
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        Ok(())
    }

    fn reconcile_interrupted_processing(&self) -> CatalogResult<bool> {
        let database_path = self.database_path();
        loop {
            let current_payload: Option<String> = self
                .connection
                .query_row(
                    "select value from catalog_metadata where key = ?1",
                    [PROGRESS_METADATA_KEY],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            let Some(current_payload) = current_payload else {
                return Ok(false);
            };
            let mut progress = serde_json::from_str::<CatalogProcessingProgress>(&current_payload)?;
            if progress.state != CatalogProcessingState::Running {
                return Ok(false);
            }
            progress.state = CatalogProcessingState::Failed;
            progress.message =
                Some("Catalog processing was interrupted; restart it to continue".to_owned());
            progress.updated_at = utc_now();
            let changed = self
                .connection
                .execute(
                    "update catalog_metadata set value = ?1 where key = ?2 and value = ?3",
                    params![
                        serde_json::to_string(&progress)?,
                        PROGRESS_METADATA_KEY,
                        current_payload
                    ],
                )
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            if changed == 1 {
                return Ok(true);
            }
        }
    }

    fn read_contract_value<T: for<'de> Deserialize<'de>>(
        &self,
        key: &str,
    ) -> CatalogResult<Option<T>> {
        let database_path = self.database_path();
        let payload: Option<String> = self
            .connection
            .query_row(
                "select value from catalog_metadata where key = ?1",
                [key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let Some(payload) = payload else {
            return Ok(None);
        };
        if payload.len() > MAX_CONTRACT_STATE_BYTES {
            return Err(CatalogError::Corrupt {
                path: database_path,
                detail: format!("catalog metadata {key} exceeds its size limit"),
            });
        }
        serde_json::from_str(&payload)
            .map(Some)
            .map_err(|error| CatalogError::Corrupt {
                path: database_path,
                detail: format!("catalog metadata {key} is invalid: {error}"),
            })
    }

    pub fn storage_accounting(&self) -> CatalogResult<CatalogStorageAccounting> {
        let database_bytes = sqlite_storage_bytes(&self.root)?;
        let manifest_bytes = fs::metadata(self.root.join(CATALOG_MANIFEST_FILE))?.len();
        let total_bytes = directory_file_bytes(&self.root)?;
        let artifact_bytes = total_bytes.saturating_sub(database_bytes + manifest_bytes);
        let record_count = self
            .connection
            .query_row("select count(*) from catalog_records", [], |row| row.get(0))
            .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
        Ok(CatalogStorageAccounting {
            database_bytes,
            manifest_bytes,
            artifact_bytes,
            total_bytes,
            record_count,
        })
    }

    pub fn sqlite_setting(&self, pragma: &str) -> CatalogResult<String> {
        if !matches!(
            pragma,
            "journal_mode" | "synchronous" | "busy_timeout" | "foreign_keys" | "temp_store"
        ) {
            return Err(CatalogError::InvalidCatalog(
                "Unsupported SQLite setting".to_owned(),
            ));
        }
        let value: rusqlite::types::Value = self
            .connection
            .pragma_query_value(None, pragma, |row| row.get(0))
            .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
        match value {
            rusqlite::types::Value::Integer(value) => Ok(value.to_string()),
            rusqlite::types::Value::Real(value) => Ok(value.to_string()),
            rusqlite::types::Value::Text(value) => Ok(value),
            rusqlite::types::Value::Null => Ok(String::new()),
            rusqlite::types::Value::Blob(_) => Err(CatalogError::Corrupt {
                path: self.database_path(),
                detail: format!("SQLite pragma {pragma} returned a blob"),
            }),
        }
    }

    pub fn close(self) {}
}

#[derive(Debug, Clone)]
pub struct CatalogRegistry {
    registry_path: PathBuf,
    #[cfg(test)]
    fail_saves: Arc<AtomicBool>,
}

impl CatalogRegistry {
    pub fn new(state_directory: impl AsRef<Path>) -> Self {
        let state_directory = absolute_path(state_directory.as_ref());
        Self {
            registry_path: state_directory.join(CATALOG_REGISTRY_FILE),
            #[cfg(test)]
            fail_saves: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn registry_path(&self) -> &Path {
        &self.registry_path
    }

    pub fn create_catalog(
        &self,
        root: impl AsRef<Path>,
        name: impl Into<String>,
    ) -> CatalogResult<Catalog> {
        let catalog = Catalog::create(root, name)?;
        self.attach_descriptor(catalog.descriptor())?;
        Ok(catalog)
    }

    pub fn open_catalog(&self, root: impl AsRef<Path>) -> CatalogResult<Catalog> {
        Catalog::open(root)
    }

    pub fn attach(&self, root: impl AsRef<Path>) -> CatalogResult<AttachedCatalog> {
        let catalog = Catalog::open(root)?;
        self.attach_descriptor(catalog.descriptor())
    }

    pub fn list(&self) -> CatalogResult<Vec<AttachedCatalog>> {
        Ok(self.load()?.catalogs)
    }

    pub fn get(&self, catalog_id: &str) -> CatalogResult<AttachedCatalog> {
        validated_catalog_id(catalog_id)?;
        self.load()?
            .catalogs
            .into_iter()
            .find(|catalog| catalog.id == catalog_id)
            .ok_or_else(|| {
                CatalogError::NotFound(format!("Attached catalog not found: {catalog_id}"))
            })
    }

    /// Resolves a catalog ID only through the lightweight attached registry and
    /// rechecks the on-disk identity. Request handlers must use this method
    /// instead of accepting a root path for catalog-id operations.
    pub fn open_attached(&self, catalog_id: &str) -> CatalogResult<Catalog> {
        let attached = self.get(catalog_id)?;
        let catalog = Catalog::open(&attached.path)?;
        if catalog.descriptor().id != catalog_id {
            return Err(CatalogError::InvalidCatalog(
                "Attached catalog identity does not match its registry entry".to_owned(),
            ));
        }
        Ok(catalog)
    }

    pub fn detach(&self, catalog_id: &str) -> CatalogResult<AttachedCatalog> {
        validated_catalog_id(catalog_id)?;
        let _guard = lock_store_path(&self.registry_path);
        let mut registry = self.load()?;
        let index = registry
            .catalogs
            .iter()
            .position(|catalog| catalog.id == catalog_id)
            .ok_or_else(|| {
                CatalogError::NotFound(format!("Attached catalog not found: {catalog_id}"))
            })?;
        let attached = registry.catalogs[index].clone();
        let catalog = Catalog::open(&attached.path)
            .ok()
            .filter(|catalog| catalog.descriptor().id == catalog_id);
        let processing_lease = catalog
            .as_ref()
            .map(CatalogProcessingLease::try_acquire)
            .transpose()?;
        if let (Some(catalog), Some(processing_lease)) = (&catalog, &processing_lease) {
            processing_lease.reconcile_interrupted(catalog)?;
        }
        let detached = registry.catalogs.remove(index);
        self.save(&registry)?;
        Ok(detached)
    }

    pub fn delete_on_disk(&self, catalog_id: &str) -> CatalogResult<AttachedCatalog> {
        validated_catalog_id(catalog_id)?;
        let _guard = lock_store_path(&self.registry_path);
        let mut registry = self.load()?;
        let index = registry
            .catalogs
            .iter()
            .position(|catalog| catalog.id == catalog_id)
            .ok_or_else(|| {
                CatalogError::NotFound(format!("Attached catalog not found: {catalog_id}"))
            })?;
        let attached = registry.catalogs[index].clone();
        let catalog = Catalog::open(&attached.path)?;
        if catalog.descriptor().id != catalog_id {
            return Err(CatalogError::InvalidCatalog(format!(
                "Refusing to delete {} because its catalog identity changed",
                attached.path.display()
            )));
        }
        let processing_lease = CatalogProcessingLease::try_acquire(&catalog)?;
        processing_lease.reconcile_interrupted(&catalog)?;
        drop(catalog);
        registry.catalogs.remove(index);
        self.save(&registry)?;
        if let Err(error) = fs::remove_dir_all(&attached.path) {
            registry.catalogs.push(attached.clone());
            registry
                .catalogs
                .sort_by(|left, right| left.id.cmp(&right.id));
            self.save(&registry)?;
            return Err(CatalogError::Io(error));
        }
        Ok(attached)
    }

    fn attach_descriptor(&self, descriptor: &CatalogDescriptor) -> CatalogResult<AttachedCatalog> {
        if self.registry_path.starts_with(&descriptor.path) {
            return Err(CatalogError::InvalidCatalog(
                "The SceneWorks catalog registry cannot be stored inside a catalog".to_owned(),
            ));
        }
        let _guard = lock_store_path(&self.registry_path);
        let mut registry = self.load()?;
        let attached = AttachedCatalog {
            id: descriptor.id.clone(),
            name: descriptor.name.clone(),
            path: descriptor.path.clone(),
            attached_at: utc_now(),
        };
        registry
            .catalogs
            .retain(|existing| existing.id != attached.id && existing.path != attached.path);
        registry.catalogs.push(attached.clone());
        registry
            .catalogs
            .sort_by(|left, right| left.id.cmp(&right.id));
        self.save(&registry)?;
        Ok(attached)
    }

    fn load(&self) -> CatalogResult<RegistryDocument> {
        if !self.registry_path.exists() {
            return Ok(RegistryDocument::default());
        }
        let metadata = fs::metadata(&self.registry_path)?;
        if metadata.len() > MAX_REGISTRY_BYTES {
            return Err(CatalogError::Corrupt {
                path: self.registry_path.clone(),
                detail: format!("registry exceeds {MAX_REGISTRY_BYTES} bytes"),
            });
        }
        let payload = fs::read(&self.registry_path)?;
        let registry: RegistryDocument =
            serde_json::from_slice(&payload).map_err(|error| CatalogError::Corrupt {
                path: self.registry_path.clone(),
                detail: error.to_string(),
            })?;
        if registry.schema_version > REGISTRY_SCHEMA_VERSION {
            return Err(CatalogError::Incompatible {
                found: registry.schema_version,
                supported: REGISTRY_SCHEMA_VERSION,
            });
        }
        Ok(registry)
    }

    fn save(&self, registry: &RegistryDocument) -> CatalogResult<()> {
        let mut payload = serde_json::to_vec_pretty(registry)?;
        payload.push(b'\n');
        if payload.len() as u64 > MAX_REGISTRY_BYTES {
            return Err(CatalogError::InvalidCatalog(format!(
                "Catalog registry exceeds its {MAX_REGISTRY_BYTES}-byte limit"
            )));
        }
        #[cfg(test)]
        if self.fail_saves.load(Ordering::SeqCst) {
            return Err(CatalogError::Io(std::io::Error::other(
                "injected registry save failure",
            )));
        }
        atomic_write(&self.registry_path, &payload)
    }

    #[cfg(test)]
    fn set_save_failure(&self, fail: bool) {
        self.fail_saves.store(fail, Ordering::SeqCst);
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RegistryDocument {
    schema_version: u32,
    catalogs: Vec<AttachedCatalog>,
}

impl Default for RegistryDocument {
    fn default() -> Self {
        Self {
            schema_version: REGISTRY_SCHEMA_VERSION,
            catalogs: Vec::new(),
        }
    }
}

fn prepare_new_root(root: &Path) -> CatalogResult<()> {
    if root.exists() {
        let metadata = fs::symlink_metadata(root)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(CatalogError::InvalidCatalog(format!(
                "Catalog location must be a real directory: {}",
                root.display()
            )));
        }
        if fs::read_dir(root)?.next().is_some() {
            return Err(CatalogError::AlreadyExists(format!(
                "Catalog location must be empty: {}",
                root.display()
            )));
        }
    } else {
        fs::create_dir_all(root)?;
    }
    Ok(())
}

fn canonical_catalog_root(root: &Path) -> CatalogResult<PathBuf> {
    if !root.exists() {
        return Err(CatalogError::NotFound(format!(
            "Catalog directory not found: {}",
            root.display()
        )));
    }
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CatalogError::InvalidCatalog(format!(
            "Catalog location must be a real directory: {}",
            root.display()
        )));
    }
    Ok(fs::canonicalize(root)?)
}

fn validated_name(name: String) -> CatalogResult<String> {
    let name = name.trim().to_owned();
    if name.is_empty() || name.chars().count() > 200 {
        return Err(CatalogError::InvalidCatalog(
            "Catalog name must contain 1 to 200 characters".to_owned(),
        ));
    }
    Ok(name)
}

fn empty_json_object() -> Value {
    Value::Object(serde_json::Map::new())
}

fn validated_catalog_id(catalog_id: &str) -> CatalogResult<()> {
    if catalog_id.len() != 32
        || !catalog_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog id has an invalid format".to_owned(),
        ));
    }
    Ok(())
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn validate_record(record: &NewCatalogRecord) -> CatalogResult<()> {
    if record.id.trim().is_empty() || record.id.len() > 512 || record.id.contains('\0') {
        return Err(CatalogError::InvalidCatalog(
            "Catalog record id must contain 1 to 512 non-NUL bytes".to_owned(),
        ));
    }
    validate_catalog_relative_path(&record.image_path, "image")?;
    for path in [
        record.thumbnail_path.as_deref(),
        record.embedding_path.as_deref(),
        record.artifact_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        validate_catalog_relative_path(path, "artifact")?;
    }
    Ok(())
}

fn validate_catalog_relative_path(path: &str, label: &str) -> CatalogResult<()> {
    use std::path::Component;

    if path.trim().is_empty() || path.contains('\0') {
        return Err(CatalogError::InvalidCatalog(format!(
            "Catalog record {label} path is required"
        )));
    }
    // Interpret both separators so a catalog created on Unix cannot persist a
    // Windows traversal (or vice versa) that becomes dangerous after relocation.
    let portable = path.replace('\\', "/");
    let path = Path::new(&portable);
    let has_windows_prefix =
        portable.as_bytes().get(1) == Some(&b':') || portable.starts_with("//");
    if has_windows_prefix
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(CatalogError::InvalidCatalog(format!(
            "Catalog record {label} path must stay within the catalog root"
        )));
    }
    Ok(())
}

fn validate_metadata_key(key: &str) -> CatalogResult<()> {
    if key.trim().is_empty() || key.len() > 256 || key.contains('\0') {
        return Err(CatalogError::InvalidCatalog(
            "Catalog metadata keys must contain 1 to 256 non-NUL bytes".to_owned(),
        ));
    }
    Ok(())
}

fn validate_metadata_entry(key: &str, value: &str) -> CatalogResult<()> {
    validate_metadata_key(key)?;
    if RESERVED_CATALOG_METADATA_KEYS.contains(&key) {
        return Err(CatalogError::InvalidCatalog(
            "Catalog internal metadata is read-only".to_owned(),
        ));
    }
    if value.len() > 1024 * 1024 || value.contains('\0') {
        return Err(CatalogError::InvalidCatalog(
            "Catalog metadata values cannot exceed 1 MiB or contain NUL".to_owned(),
        ));
    }
    Ok(())
}

fn validate_page_size(limit: u32) -> CatalogResult<()> {
    if limit == 0 || limit > MAX_PAGE_SIZE {
        return Err(CatalogError::InvalidCatalog(format!(
            "Catalog page size must be between 1 and {MAX_PAGE_SIZE}"
        )));
    }
    Ok(())
}

fn validate_filters(filters: &[CatalogRecordFilter]) -> CatalogResult<()> {
    if filters.len() > MAX_QUERY_FILTERS {
        return Err(CatalogError::InvalidCatalog(format!(
            "Catalog queries support at most {MAX_QUERY_FILTERS} filters"
        )));
    }
    for filter in filters {
        metadata_json_path(&filter.field)?;
        if filter.values.is_empty() || filter.values.len() > MAX_FILTER_VALUES {
            return Err(CatalogError::InvalidCatalog(format!(
                "Each catalog filter requires 1 to {MAX_FILTER_VALUES} values"
            )));
        }
        if filter
            .values
            .iter()
            .any(|value| value.len() > 512 || value.contains('\0'))
        {
            return Err(CatalogError::InvalidCatalog(
                "Catalog filter values cannot exceed 512 bytes or contain NUL".to_owned(),
            ));
        }
    }
    Ok(())
}

fn metadata_json_path(field: &str) -> CatalogResult<String> {
    if field.is_empty() || field.len() > 128 {
        return Err(CatalogError::InvalidCatalog(
            "Catalog metadata fields must contain 1 to 128 bytes".to_owned(),
        ));
    }
    let segments = field.split('.').collect::<Vec<_>>();
    if segments.iter().any(|segment| {
        segment.is_empty()
            || segment.len() > 64
            || !segment
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    }) {
        return Err(CatalogError::InvalidCatalog(
            "Catalog metadata fields may contain ASCII letters, numbers, '_' and '-'".to_owned(),
        ));
    }
    Ok(format!(
        "$.{}",
        segments
            .into_iter()
            .map(|segment| format!("\"{segment}\""))
            .collect::<Vec<_>>()
            .join(".")
    ))
}

fn append_filter_sql(
    sql: &mut String,
    bindings: &mut Vec<SqlValue>,
    filters: &[CatalogRecordFilter],
) {
    for filter in filters {
        let json_path =
            metadata_json_path(&filter.field).expect("catalog filters were validated before SQL");
        sql.push_str(" and cast(json_extract(metadata_json, ?) as text) in (");
        bindings.push(SqlValue::Text(json_path));
        for (index, value) in filter.values.iter().enumerate() {
            if index > 0 {
                sql.push(',');
            }
            sql.push('?');
            bindings.push(SqlValue::Text(value.clone()));
        }
        sql.push(')');
    }
}

fn normalize_source_config(source: &mut CatalogSourceConfig) -> CatalogResult<()> {
    let kind = source.kind.trim();
    if kind.is_empty()
        || kind.len() > 64
        || !kind
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog source kind has an invalid format".to_owned(),
        ));
    }
    if source.paths.is_empty() || source.paths.len() > 256 {
        return Err(CatalogError::InvalidCatalog(
            "Catalog source configuration requires 1 to 256 paths".to_owned(),
        ));
    }
    if !source.options.is_object() {
        return Err(CatalogError::InvalidCatalog(
            "Catalog source options must be an object".to_owned(),
        ));
    }
    source.kind = kind.to_owned();
    for path in &mut source.paths {
        if !path.is_absolute() {
            return Err(CatalogError::InvalidCatalog(
                "Catalog source paths must be absolute".to_owned(),
            ));
        }
        *path = fs::canonicalize(&*path).map_err(|_| {
            CatalogError::InvalidCatalog(
                "A selected catalog source path does not exist or cannot be accessed".to_owned(),
            )
        })?;
    }
    Ok(())
}

fn validate_contract_state(state: &CatalogContractState) -> CatalogResult<()> {
    if state.analyzer_versions.len() > 128
        || state.analyzer_versions.iter().any(|(name, version)| {
            name.trim().is_empty()
                || name.len() > 128
                || version.trim().is_empty()
                || version.len() > 512
        })
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog analyzer provenance is invalid".to_owned(),
        ));
    }
    if state.checkpoints.len() > 256
        || state
            .checkpoints
            .keys()
            .any(|name| name.trim().is_empty() || name.len() > 128)
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog checkpoints are invalid".to_owned(),
        ));
    }
    validate_processing_progress(&state.processing)?;
    let serialized = serde_json::to_vec(state)?;
    if serialized.len() > MAX_CONTRACT_STATE_BYTES {
        return Err(CatalogError::InvalidCatalog(format!(
            "Catalog contract state exceeds {MAX_CONTRACT_STATE_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_analyzer_settings(settings: &CatalogAnalyzerSettings) -> CatalogResult<()> {
    let thresholds = &settings.thresholds;
    for (name, value) in [
        ("personMinConfidence", thresholds.person_min_confidence),
        ("faceMinConfidence", thresholds.face_min_confidence),
        (
            "prominentFrameFraction",
            thresholds.prominent_frame_fraction,
        ),
        ("frameEdgeMargin", thresholds.frame_edge_margin),
        ("minPoseCoverage", thresholds.min_pose_coverage),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(CatalogError::InvalidCatalog(format!(
                "Catalog analyzer {name} must be between 0 and 1"
            )));
        }
    }
    if thresholds.frame_edge_margin > 0.25 {
        return Err(CatalogError::InvalidCatalog(
            "Catalog analyzer frameEdgeMargin must not exceed 0.25".to_owned(),
        ));
    }
    if !thresholds.pose_min_keypoint_confidence.is_finite()
        || thresholds.pose_min_keypoint_confidence < 0.0
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog analyzer poseMinKeypointConfidence must be finite and non-negative".to_owned(),
        ));
    }
    Ok(())
}

fn validate_processing_progress(progress: &CatalogProcessingProgress) -> CatalogResult<()> {
    if progress
        .message
        .as_ref()
        .is_some_and(|message| message.len() > 4096)
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog processing message exceeds 4096 bytes".to_owned(),
        ));
    }
    if progress.updated_at.len() > 128 {
        return Err(CatalogError::InvalidCatalog(
            "Catalog processing timestamp is invalid".to_owned(),
        ));
    }
    Ok(())
}

fn configure_connection(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    connection
        .pragma_update(None, "journal_mode", "wal")
        .map_err(|error| map_sqlite_error(database_path, error))?;
    connection
        .pragma_update(None, "synchronous", "normal")
        .map_err(|error| map_sqlite_error(database_path, error))?;
    connection
        .pragma_update(None, "foreign_keys", true)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    connection
        // A facet may encounter millions of distinct values. Spill SQLite's
        // transient GROUP BY sorter to disk instead of growing the API heap
        // with catalog cardinality.
        .pragma_update(None, "temp_store", "file")
        .map_err(|error| map_sqlite_error(database_path, error))?;
    connection
        .pragma_update(None, "cache_size", -65_536)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    connection
        .pragma_update(None, "wal_autocheckpoint", 2_000)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    Ok(())
}

fn migrate(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    let version: u32 = connection
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if version > CATALOG_SCHEMA_VERSION {
        return Err(CatalogError::Incompatible {
            found: version,
            supported: CATALOG_SCHEMA_VERSION,
        });
    }
    if version < 1 {
        connection
            .execute_batch(
                "begin immediate;
                 create table if not exists catalog_metadata (
                    key text primary key not null,
                    value text not null
                 ) strict;
                 create table if not exists catalog_records (
                    id text primary key not null,
                    image_path text not null,
                    thumbnail_path text,
                    embedding_path text,
                    artifact_path text,
                    metadata_json text not null default '{}',
                    created_at text not null,
                    updated_at text not null
                 ) strict;
                 create index if not exists idx_catalog_records_image_path
                    on catalog_records(image_path);
                 pragma user_version = 1;
                 commit;",
            )
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    Ok(())
}

fn validate_schema(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    validate_table_columns(
        connection,
        database_path,
        "catalog_metadata",
        &[("key", "TEXT", 1, 1), ("value", "TEXT", 1, 0)],
    )?;
    validate_table_columns(
        connection,
        database_path,
        "catalog_records",
        &[
            ("id", "TEXT", 1_i64, 1_i64),
            ("image_path", "TEXT", 1, 0),
            ("thumbnail_path", "TEXT", 0, 0),
            ("embedding_path", "TEXT", 0, 0),
            ("artifact_path", "TEXT", 0, 0),
            ("metadata_json", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ],
    )?;

    let index_owner: Option<String> = connection
        .query_row(
            "select tbl_name from sqlite_master
             where type = 'index' and name = 'idx_catalog_records_image_path'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let mut index_statement = connection
        .prepare("pragma index_info(idx_catalog_records_image_path)")
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let indexed_columns = index_statement
        .query_map([], |row| row.get::<_, String>("name"))
        .map_err(|error| map_sqlite_error(database_path, error))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if index_owner.as_deref() != Some("catalog_records") || indexed_columns != ["image_path"] {
        return Err(CatalogError::Corrupt {
            path: database_path.to_path_buf(),
            detail: "catalog_records image-path index has the wrong definition".to_owned(),
        });
    }
    Ok(())
}

fn validate_table_columns(
    connection: &Connection,
    database_path: &Path,
    table: &str,
    expected_columns: &[(&str, &str, i64, i64)],
) -> CatalogResult<()> {
    let sql = format!("pragma table_info({table})");
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let actual_columns = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>("name")?,
                row.get::<_, String>("type")?,
                row.get::<_, i64>("notnull")?,
                row.get::<_, i64>("pk")?,
            ))
        })
        .map_err(|error| map_sqlite_error(database_path, error))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let columns_match = actual_columns.len() == expected_columns.len()
        && actual_columns
            .iter()
            .zip(expected_columns.iter())
            .all(|(actual, expected)| {
                actual.0 == expected.0
                    && actual.1.eq_ignore_ascii_case(expected.1)
                    && actual.2 == expected.2
                    && actual.3 == expected.3
            });
    if !columns_match {
        return Err(CatalogError::Corrupt {
            path: database_path.to_path_buf(),
            detail: format!("{table} table does not match schema version 1"),
        });
    }
    Ok(())
}

fn write_database_metadata(
    connection: &Connection,
    database_path: &Path,
    descriptor: &CatalogDescriptor,
) -> CatalogResult<()> {
    let transaction = connection
        .unchecked_transaction()
        .map_err(|error| map_sqlite_error(database_path, error))?;
    for (key, value) in [
        (CATALOG_ID_METADATA_KEY, descriptor.id.as_str()),
        (CATALOG_NAME_METADATA_KEY, descriptor.name.as_str()),
        (
            CATALOG_CREATED_AT_METADATA_KEY,
            descriptor.created_at.as_str(),
        ),
    ] {
        transaction
            .execute(
                "insert or replace into catalog_metadata(key, value) values (?1, ?2)",
                params![key, value],
            )
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    transaction
        .commit()
        .map_err(|error| map_sqlite_error(database_path, error))
}

fn validate_database_metadata(
    connection: &Connection,
    database_path: &Path,
    descriptor: &CatalogDescriptor,
) -> CatalogResult<()> {
    for (key, expected) in [
        (CATALOG_ID_METADATA_KEY, descriptor.id.as_str()),
        (CATALOG_NAME_METADATA_KEY, descriptor.name.as_str()),
        (
            CATALOG_CREATED_AT_METADATA_KEY,
            descriptor.created_at.as_str(),
        ),
    ] {
        let actual: Option<String> = connection
            .query_row(
                "select value from catalog_metadata where key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| map_sqlite_error(database_path, error))?;
        if actual.as_deref() != Some(expected) {
            return Err(CatalogError::Corrupt {
                path: database_path.to_path_buf(),
                detail: format!("manifest and database {key} values do not match"),
            });
        }
    }
    Ok(())
}

fn read_manifest(root: &Path) -> CatalogResult<CatalogDescriptor> {
    let path = root.join(CATALOG_MANIFEST_FILE);
    let payload = fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CatalogError::InvalidCatalog(format!("Catalog manifest is missing: {}", path.display()))
        } else {
            CatalogError::Io(error)
        }
    })?;
    let mut descriptor: CatalogDescriptor =
        serde_json::from_slice(&payload).map_err(|error| CatalogError::Corrupt {
            path: path.clone(),
            detail: error.to_string(),
        })?;
    descriptor.path = root.to_path_buf();
    Ok(descriptor)
}

fn write_manifest(root: &Path, descriptor: &CatalogDescriptor) -> CatalogResult<()> {
    let mut payload = serde_json::to_vec_pretty(descriptor)?;
    payload.push(b'\n');
    atomic_write(&root.join(CATALOG_MANIFEST_FILE), &payload)
}

fn atomic_write(path: &Path, payload: &[u8]) -> CatalogResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let token = random_hex(8)?;
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("json");
    let temporary = path.with_extension(format!("{extension}.{token}.tmp"));
    let mut file = fs::File::create(&temporary)?;
    file.write_all(payload)?;
    file.sync_all()?;
    drop(file);
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(CatalogError::Io(error));
    }
    Ok(())
}

fn random_hex(bytes: usize) -> CatalogResult<String> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buffer = vec![0; bytes];
    getrandom::fill(&mut buffer)
        .map_err(|error| CatalogError::Io(std::io::Error::other(error.to_string())))?;
    let mut output = String::with_capacity(bytes * 2);
    for byte in buffer {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(output)
}

fn map_sqlite_error(path: &Path, error: rusqlite::Error) -> CatalogError {
    match &error {
        rusqlite::Error::SqliteFailure(inner, _)
            if matches!(
                inner.code,
                ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase
            ) =>
        {
            CatalogError::Corrupt {
                path: path.to_path_buf(),
                detail: error.to_string(),
            }
        }
        _ => CatalogError::Sqlite(error),
    }
}

fn sqlite_storage_bytes(root: &Path) -> CatalogResult<u64> {
    let mut bytes = 0_u64;
    for name in [
        CATALOG_DATABASE_FILE.to_owned(),
        format!("{CATALOG_DATABASE_FILE}-wal"),
        format!("{CATALOG_DATABASE_FILE}-shm"),
    ] {
        let path = root.join(name);
        if let Ok(metadata) = fs::symlink_metadata(path) {
            bytes = bytes.saturating_add(metadata.len());
        }
    }
    Ok(bytes)
}

fn directory_file_bytes(root: &Path) -> CatalogResult<u64> {
    let mut total = 0_u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_symlink() {
                total = total.saturating_add(metadata.len());
            } else if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn record(id: &str) -> NewCatalogRecord {
        NewCatalogRecord {
            id: id.to_owned(),
            image_path: format!("images/{id}.jpg"),
            thumbnail_path: Some(format!("thumbnails/{id}.jpg")),
            embedding_path: Some(format!("embeddings/{id}.f32")),
            artifact_path: None,
            metadata: serde_json::json!({"caption": id}),
        }
    }

    #[test]
    fn catalog_lifecycle_is_independent_of_projects_and_jobs() {
        let temporary = tempdir().expect("temp directory");
        let state = temporary.path().join("state");
        let root = temporary.path().join("selected").join("photos");
        let registry = CatalogRegistry::new(&state);
        fs::create_dir_all(&state).unwrap();
        for operational_database in ["jobs.db", "project.db"] {
            let connection = Connection::open(state.join(operational_database)).unwrap();
            connection
                .execute("create table sentinel(value text not null)", [])
                .unwrap();
            connection
                .execute("insert into sentinel(value) values ('untouched')", [])
                .unwrap();
        }

        let mut catalog = registry
            .create_catalog(&root, "Photos")
            .expect("catalog creates");
        let id = catalog.descriptor().id.clone();
        catalog
            .append_records(&[record("one"), record("two")])
            .expect("records append in a batch");
        catalog.close();

        let reopened = registry.open_catalog(&root).expect("catalog reopens");
        assert_eq!(reopened.page_records(0, 10).expect("page").len(), 2);
        assert!(root.join(CATALOG_DATABASE_FILE).is_file());
        for operational_database in ["jobs.db", "project.db"] {
            let connection = Connection::open(state.join(operational_database)).unwrap();
            let tables: Vec<String> = connection
                .prepare("select name from sqlite_master where type = 'table' order by name")
                .unwrap()
                .query_map([], |row| row.get(0))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            assert_eq!(tables, ["sentinel"]);
            assert_eq!(
                connection
                    .query_row("select value from sentinel", [], |row| row
                        .get::<_, String>(0))
                    .unwrap(),
                "untouched"
            );
        }
        reopened.close();

        let moved = temporary.path().join("moved-catalog");
        fs::rename(&root, &moved).expect("catalog directory moves while closed");
        let attached = registry.attach(&moved).expect("moved catalog reattaches");
        assert_eq!(attached.id, id);
        assert_eq!(registry.list().expect("list").len(), 1);
        assert_eq!(
            registry.list().expect("list")[0].path,
            fs::canonicalize(&moved).unwrap()
        );

        registry.detach(&id).expect("catalog detaches");
        assert!(moved.exists(), "detach must not delete catalog files");
        assert!(registry.list().expect("list").is_empty());
        assert_eq!(
            Catalog::open(&moved)
                .expect("detached catalog remains independently openable")
                .page_records(0, 10)
                .expect("page")
                .len(),
            2
        );
    }

    #[test]
    fn explicit_delete_removes_only_a_valid_attached_catalog() {
        let temporary = tempdir().expect("temp directory");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let catalog = registry
            .create_catalog(&root, "Delete me")
            .expect("catalog creates");
        let id = catalog.descriptor().id.clone();
        catalog.close();

        registry
            .delete_on_disk(&id)
            .expect("explicit delete succeeds");
        assert!(!root.exists());
        assert!(registry.list().expect("list").is_empty());
    }

    #[test]
    fn registry_failure_does_not_delete_catalog_files() {
        let temporary = tempdir().expect("temp directory");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let catalog = registry
            .create_catalog(&root, "Keep me")
            .expect("catalog creates");
        let id = catalog.descriptor().id.clone();
        catalog.close();

        registry.set_save_failure(true);
        assert!(registry.delete_on_disk(&id).is_err());
        assert!(root.join(CATALOG_DATABASE_FILE).is_file());
        assert_eq!(registry.list().expect("registry remains readable").len(), 1);
    }

    #[test]
    fn registry_contains_paths_not_catalog_records() {
        let temporary = tempdir().expect("temp directory");
        let registry = CatalogRegistry::new(temporary.path().join("normal-state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry
            .create_catalog(&root, "Registry boundary")
            .expect("catalog creates");
        catalog
            .append_records(&[record("private-record")])
            .expect("record appends");

        let registry_json =
            fs::read_to_string(registry.registry_path()).expect("registry is readable");
        assert!(registry_json.contains("Registry boundary"));
        assert!(!registry_json.contains("private-record"));
        assert!(
            fs::metadata(registry.registry_path()).unwrap().len() < 4096,
            "registry remains lightweight"
        );

        let connection = Connection::open(root.join(CATALOG_DATABASE_FILE)).unwrap();
        let column_types: Vec<String> = connection
            .prepare("pragma table_info(catalog_records)")
            .unwrap()
            .query_map([], |row| row.get(2))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(column_types.iter().all(|kind| kind == "TEXT"));
    }

    #[test]
    fn sqlite_is_tuned_for_batched_writes_and_concurrent_reads() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let mut writer = Catalog::create(&root, "Large catalog").expect("catalog creates");
        let reader = Catalog::open(&root).expect("second read connection opens");
        assert_eq!(writer.sqlite_setting("journal_mode").unwrap(), "wal");
        assert_eq!(writer.sqlite_setting("busy_timeout").unwrap(), "5000");
        assert_eq!(writer.sqlite_setting("foreign_keys").unwrap(), "1");
        assert_eq!(
            writer.sqlite_setting("temp_store").unwrap(),
            "1",
            "facet aggregation temp state must spill to files, not process memory"
        );

        let records = (0..2_500)
            .map(|index| record(&format!("record-{index:04}")))
            .collect::<Vec<_>>();
        writer
            .append_records(&records)
            .expect("large batch appends");
        assert_eq!(reader.page_records(100, 75).expect("paged read").len(), 75);
        assert_eq!(
            writer
                .storage_accounting()
                .expect("accounting")
                .record_count,
            2_500
        );

        let mut cursor = None;
        let mut ids = Vec::new();
        loop {
            let page = reader
                .page_records_after(cursor, 137)
                .expect("keyset page reads");
            if page.records.is_empty() {
                break;
            }
            cursor = page.next_cursor;
            ids.extend(page.records.into_iter().map(|record| record.id));
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(ids.len(), 2_500);
        ids.sort();
        ids.dedup();
        assert_eq!(
            ids.len(),
            2_500,
            "keyset pages neither skip nor repeat rows"
        );
    }

    #[test]
    fn storage_accounting_includes_externalized_artifacts() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let catalog = Catalog::create(&root, "Accounting").expect("catalog creates");
        fs::write(
            catalog
                .artifact_directory("images")
                .unwrap()
                .join("large.bin"),
            vec![7_u8; 16_384],
        )
        .expect("artifact writes");

        let accounting = catalog.storage_accounting().expect("accounting");
        assert!(accounting.database_bytes > 0);
        assert!(accounting.manifest_bytes > 0);
        assert!(accounting.artifact_bytes >= 16_384);
        assert_eq!(
            accounting.total_bytes,
            accounting.database_bytes + accounting.manifest_bytes + accounting.artifact_bytes
        );
    }

    #[test]
    fn incompatible_and_corrupt_catalogs_have_typed_errors() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Errors")
            .expect("catalog creates")
            .close();

        let database_path = root.join(CATALOG_DATABASE_FILE);
        let connection = Connection::open(&database_path).unwrap();
        connection
            .pragma_update(None, "user_version", CATALOG_SCHEMA_VERSION + 1)
            .unwrap();
        drop(connection);
        assert!(matches!(
            Catalog::open(&root),
            Err(CatalogError::Incompatible { .. })
        ));

        fs::write(&database_path, b"not a sqlite database").expect("database corrupts");
        assert!(matches!(
            Catalog::open(&root),
            Err(CatalogError::Corrupt { .. })
        ));
    }

    #[test]
    fn older_schema_is_migrated_on_open() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Migration")
            .expect("catalog creates")
            .close();
        let database_path = root.join(CATALOG_DATABASE_FILE);
        let connection = Connection::open(&database_path).unwrap();
        connection
            .execute("drop table catalog_records", [])
            .unwrap();
        connection.pragma_update(None, "user_version", 0).unwrap();
        drop(connection);

        Catalog::open(&root)
            .expect("older catalog migrates")
            .close();
        let connection = Connection::open(database_path).unwrap();
        let version: u32 = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(version, CATALOG_SCHEMA_VERSION);
    }

    #[test]
    fn malformed_current_schema_is_reported_as_corrupt_during_open() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Schema")
            .expect("catalog creates")
            .close();
        let connection = Connection::open(root.join(CATALOG_DATABASE_FILE)).unwrap();
        connection
            .execute("drop index idx_catalog_records_image_path", [])
            .unwrap();
        drop(connection);

        assert!(matches!(
            Catalog::open(&root),
            Err(CatalogError::Corrupt { .. })
        ));
    }

    #[test]
    fn missing_metadata_table_is_reported_as_corrupt_during_open() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Metadata schema")
            .expect("catalog creates")
            .close();
        let connection = Connection::open(root.join(CATALOG_DATABASE_FILE)).unwrap();
        connection
            .execute("drop table catalog_metadata", [])
            .unwrap();
        drop(connection);

        assert!(matches!(
            Catalog::open(&root),
            Err(CatalogError::Corrupt { .. })
        ));
    }

    #[test]
    fn same_name_index_with_wrong_definition_is_reported_as_corrupt() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Index schema")
            .expect("catalog creates")
            .close();
        let connection = Connection::open(root.join(CATALOG_DATABASE_FILE)).unwrap();
        connection
            .execute("drop index idx_catalog_records_image_path", [])
            .unwrap();
        connection
            .execute(
                "create index idx_catalog_records_image_path
                 on catalog_metadata(value)",
                [],
            )
            .unwrap();
        drop(connection);

        assert!(matches!(
            Catalog::open(&root),
            Err(CatalogError::Corrupt { .. })
        ));
    }

    #[test]
    fn oversized_registry_is_rejected_before_replacing_the_valid_file() {
        let temporary = tempdir().expect("temp directory");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        registry
            .save(&RegistryDocument::default())
            .expect("small registry saves");
        let oversized = RegistryDocument {
            schema_version: REGISTRY_SCHEMA_VERSION,
            catalogs: vec![AttachedCatalog {
                id: "oversized".to_owned(),
                name: "x".repeat(MAX_REGISTRY_BYTES as usize),
                path: PathBuf::from("catalog"),
                attached_at: utc_now(),
            }],
        };

        assert!(matches!(
            registry.save(&oversized),
            Err(CatalogError::InvalidCatalog(_))
        ));
        assert!(
            registry
                .load()
                .expect("prior registry still loads")
                .catalogs
                .is_empty(),
            "oversized serialization must not replace the previous registry"
        );
    }

    #[test]
    fn creation_rejects_nonempty_directories_to_make_delete_semantics_safe() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("existing");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("keep.txt"), "user data").unwrap();
        assert!(matches!(
            Catalog::create(&root, "Unsafe"),
            Err(CatalogError::AlreadyExists(_))
        ));
        assert_eq!(
            fs::read_to_string(root.join("keep.txt")).unwrap(),
            "user data"
        );
    }

    #[test]
    fn contract_state_and_query_facets_are_persisted_and_bounded() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let source = temporary.path().join("source.parquet");
        fs::write(&source, b"fixture").unwrap();
        let mut catalog = Catalog::create(&root, "Queryable").expect("catalog creates");
        let mut first = record("one");
        first.metadata = serde_json::json!({
            "medium": "photo",
            "personCount": 1,
            "analysis": { "fullBody": true }
        });
        let mut second = record("two");
        second.metadata = serde_json::json!({
            "medium": "illustration",
            "personCount": 1,
            "analysis": { "fullBody": false }
        });
        let mut third = record("three");
        third.metadata = serde_json::json!({
            "medium": "photo",
            "personCount": 2,
            "analysis": { "fullBody": true }
        });
        catalog.append_records(&[first, second, third]).unwrap();

        let state = CatalogContractState {
            source_config: Some(CatalogSourceConfig {
                kind: "parquet".to_owned(),
                paths: vec![source.clone()],
                options: serde_json::json!({"captionColumn": "TEXT"}),
            }),
            analyzer_versions: BTreeMap::from([(
                "person_detector".to_owned(),
                "model@sha256:abc".to_owned(),
            )]),
            checkpoints: BTreeMap::from([(
                "scan".to_owned(),
                serde_json::json!({"shard": 2, "row": 48}),
            )]),
            processing: CatalogProcessingProgress {
                state: CatalogProcessingState::Paused,
                candidate_count: 9,
                processed_count: 3,
                accepted_count: 2,
                rejected_count: 1,
                error_count: 1,
                message: Some("user requested pause".to_owned()),
                updated_at: "2026-07-26T00:00:00Z".to_owned(),
            },
        };
        catalog.set_contract_state(&state).unwrap();
        assert_eq!(
            catalog
                .contract_state()
                .unwrap()
                .source_config
                .unwrap()
                .paths,
            [fs::canonicalize(source).unwrap()]
        );

        let photo = CatalogRecordFilter {
            field: "medium".to_owned(),
            values: vec!["photo".to_owned()],
        };
        let first_page = catalog
            .query_records_after(None, 1, std::slice::from_ref(&photo))
            .unwrap();
        assert_eq!(first_page.records.len(), 1);
        assert!(
            first_page.next_cursor.is_some(),
            "a lookahead row produces a continuation cursor"
        );
        let second_page = catalog
            .query_records_after(first_page.next_cursor, 1, std::slice::from_ref(&photo))
            .unwrap();
        assert_eq!(second_page.records.len(), 1);
        assert!(
            second_page.next_cursor.is_none(),
            "the final page does not advertise an empty trailing page"
        );

        let facets = catalog
            .facet_counts(
                &["medium".to_owned(), "analysis.fullBody".to_owned()],
                &[],
                1,
            )
            .unwrap();
        assert_eq!(facets.len(), 2);
        assert_eq!(facets[0].values.len(), 1, "facet output honors its cap");
        assert_eq!(facets[0].values[0].value, "photo");
        assert_eq!(facets[0].values[0].count, 2);
    }

    #[test]
    fn record_paths_and_attached_ids_cannot_escape_catalog_scope() {
        let temporary = tempdir().expect("temp directory");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let root = temporary.path().join("catalog");
        let mut catalog = registry.create_catalog(&root, "Scoped").unwrap();
        let catalog_id = catalog.descriptor().id.clone();

        for unsafe_path in [
            "../outside.jpg",
            r"..\outside.jpg",
            "/outside.jpg",
            r"C:\outside.jpg",
        ] {
            let mut unsafe_record = record("unsafe");
            unsafe_record.image_path = unsafe_path.to_owned();
            assert!(
                matches!(
                    catalog.append_records(&[unsafe_record]),
                    Err(CatalogError::InvalidCatalog(_))
                ),
                "{unsafe_path} must be rejected"
            );
        }
        catalog.close();

        assert_eq!(
            registry.open_attached(&catalog_id).unwrap().descriptor().id,
            catalog_id
        );
        assert!(matches!(
            registry.open_attached("../../catalog"),
            Err(CatalogError::InvalidCatalog(_))
        ));
        assert!(matches!(
            registry.open_attached("00000000000000000000000000000000"),
            Err(CatalogError::NotFound(_))
        ));
    }

    #[test]
    fn atomic_scanner_metadata_cannot_overwrite_catalog_identity() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let mut catalog = Catalog::create(&root, "Reserved metadata").expect("catalog creates");
        let descriptor = catalog.descriptor().clone();
        let original = RESERVED_CATALOG_METADATA_KEYS
            .iter()
            .map(|key| {
                (
                    *key,
                    catalog
                        .metadata(key)
                        .expect("reserved metadata reads before attempted write"),
                )
            })
            .collect::<Vec<_>>();

        for (index, (key, expected)) in original.iter().enumerate() {
            let record_id = format!("reserved-{index}");
            let attempted_record = record(&record_id);
            assert!(
                matches!(
                    catalog.append_records_and_metadata(&[attempted_record], &[(*key, "tampered")]),
                    Err(CatalogError::InvalidCatalog(_))
                ),
                "{key} must be read-only"
            );
            assert_eq!(
                catalog.metadata(key).expect("reserved metadata rereads"),
                *expected,
                "{key} must remain unchanged"
            );
            assert!(
                !catalog
                    .contains_record_id(&record_id)
                    .expect("record existence checks"),
                "record and metadata writes must fail atomically for {key}"
            );
        }
        catalog.close();

        let reopened = Catalog::open(&root).expect("catalog remains reopenable");
        assert_eq!(reopened.descriptor(), &descriptor);
        assert_eq!(reopened.storage_accounting().unwrap().record_count, 0);
        for (key, expected) in original {
            assert_eq!(reopened.metadata(key).unwrap(), expected);
        }
    }

    #[test]
    fn absent_processing_state_uses_the_stable_catalog_creation_time() {
        let temporary = tempdir().expect("temp directory");
        let catalog = Catalog::create(temporary.path().join("catalog"), "Legacy state").unwrap();
        let created_at = catalog.descriptor().created_at.clone();

        let first = catalog.contract_state().unwrap().processing;
        let second = catalog.contract_state().unwrap().processing;
        assert_eq!(first.updated_at, created_at);
        assert_eq!(second.updated_at, created_at);
        assert_eq!(first, second, "repeated status reads must be stable");
    }

    #[test]
    fn high_cardinality_facets_are_disk_backed_and_exclude_oversized_keys() {
        let temporary = tempdir().expect("temp directory");
        let mut catalog =
            Catalog::create(temporary.path().join("catalog"), "Bounded facets").unwrap();
        let mut records = (0..2_048)
            .map(|index| {
                let mut item = record(&format!("record-{index:04}"));
                item.metadata = serde_json::json!({"unique": format!("value-{index:04}")});
                item
            })
            .collect::<Vec<_>>();
        let mut oversized = record("oversized");
        oversized.metadata =
            serde_json::json!({"unique": "x".repeat(MAX_FACET_VALUE_BYTES as usize + 1)});
        records.push(oversized);
        catalog.append_records(&records).unwrap();

        assert_eq!(catalog.sqlite_setting("temp_store").unwrap(), "1");
        let facets = catalog
            .facet_counts(&["unique".to_owned()], &[], 7)
            .unwrap();
        assert_eq!(
            facets[0].values.len(),
            7,
            "high-cardinality output is capped"
        );
        assert!(facets[0]
            .values
            .iter()
            .all(|bucket| bucket.value.len() <= MAX_FACET_VALUE_BYTES as usize));

        let mut oversized_only = Catalog::create(
            temporary.path().join("oversized-catalog"),
            "Oversized facets",
        )
        .unwrap();
        let mut oversized_record = record("only");
        oversized_record.metadata =
            serde_json::json!({"unique": "x".repeat(MAX_FACET_VALUE_BYTES as usize + 1)});
        oversized_only.append_records(&[oversized_record]).unwrap();
        assert!(
            oversized_only
                .facet_counts(&["unique".to_owned()], &[], 10)
                .unwrap()[0]
                .values
                .is_empty(),
            "oversized facet keys are excluded before grouping"
        );
    }

    #[test]
    fn processing_control_is_narrow_revisioned_and_preserves_concurrent_contract_updates() {
        let temporary = tempdir().expect("temp directory");
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        let catalog = Catalog::create(temporary.path().join("catalog"), "Controlled").unwrap();
        let mut contract = CatalogContractState {
            source_config: Some(CatalogSourceConfig {
                kind: "parquet".to_owned(),
                paths: vec![source.clone()],
                options: serde_json::json!({"captionColumn": "TEXT"}),
            }),
            analyzer_versions: BTreeMap::from([(
                "tagger".to_owned(),
                "model@sha256:one".to_owned(),
            )]),
            checkpoints: BTreeMap::from([("scanner".to_owned(), serde_json::json!({"row": 4}))]),
            processing: CatalogProcessingProgress {
                state: CatalogProcessingState::Running,
                message: Some("active".to_owned()),
                updated_at: catalog_timestamp_now(),
                ..CatalogProcessingProgress::default()
            },
        };
        catalog.set_contract_state(&contract).unwrap();

        // A processor publishes newer provenance/checkpoint data immediately
        // before the user request. The narrow control write must not replace it
        // with an earlier full-contract snapshot.
        contract
            .analyzer_versions
            .insert("tagger".to_owned(), "model@sha256:newer".to_owned());
        contract
            .checkpoints
            .insert("scanner".to_owned(), serde_json::json!({"row": 9}));
        catalog.set_contract_state(&contract).unwrap();
        let paused = catalog
            .request_processing_control(0, CatalogProcessingDesiredState::Paused)
            .unwrap();
        assert_eq!(paused.revision, 1);
        let preserved = catalog.contract_state().unwrap();
        assert_eq!(preserved.analyzer_versions["tagger"], "model@sha256:newer");
        assert_eq!(preserved.checkpoints["scanner"]["row"], 9);
        assert_eq!(
            preserved.source_config.unwrap().paths,
            vec![fs::canonicalize(source).unwrap()]
        );
        assert!(matches!(
            catalog.request_processing_control(0, CatalogProcessingDesiredState::Paused),
            Err(CatalogError::Conflict(_))
        ));
        assert!(matches!(
            catalog.request_processing_control(1, CatalogProcessingDesiredState::Running),
            Err(CatalogError::Conflict(_))
        ));

        let mut actual = catalog.contract_state().unwrap().processing;
        actual.state = CatalogProcessingState::Paused;
        actual.updated_at = catalog_timestamp_now();
        catalog.set_processing_progress(&actual).unwrap();
        let resumed = catalog
            .request_processing_control(1, CatalogProcessingDesiredState::Running)
            .unwrap();
        assert_eq!(resumed.revision, 2);
        assert_eq!(
            catalog.contract_state().unwrap().processing.state,
            CatalogProcessingState::Paused,
            "desired resume never claims actual running"
        );
    }

    #[test]
    fn simultaneous_processing_control_requests_have_one_conflict_loser() {
        let temporary = tempdir().expect("temp directory");
        let registry_root = temporary.path().join("state");
        let registry = CatalogRegistry::new(&registry_root);
        let catalog = registry
            .create_catalog(temporary.path().join("catalog"), "Concurrent")
            .unwrap();
        let id = catalog.descriptor().id.clone();
        catalog
            .set_processing_progress(&CatalogProcessingProgress {
                state: CatalogProcessingState::Running,
                updated_at: catalog_timestamp_now(),
                ..CatalogProcessingProgress::default()
            })
            .unwrap();
        catalog.close();

        let barrier = Arc::new(std::sync::Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let registry_root = registry_root.clone();
                let id = id.clone();
                std::thread::spawn(move || {
                    let catalog = CatalogRegistry::new(registry_root)
                        .open_attached(&id)
                        .unwrap();
                    barrier.wait();
                    catalog.request_processing_control(0, CatalogProcessingDesiredState::Paused)
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let results = handles
            .into_iter()
            .map(|handle| handle.join().expect("request thread joins"))
            .collect::<Vec<_>>();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(CatalogError::Conflict(_))))
                .count(),
            1
        );
    }

    #[test]
    fn analyzer_configuration_is_durable_validated_and_revision_guarded() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let catalog = Catalog::create(&root, "Analyzers").unwrap();
        let initial = catalog.analyzer_config().unwrap();
        assert_eq!(initial.revision, 0);
        assert!(initial.settings.structured_analysis_enabled);
        assert!(!initial.settings.vision_analysis_enabled);

        let mut settings = initial.settings;
        settings.vision_analysis_enabled = true;
        settings.semantic_embeddings_enabled = true;
        settings.thresholds.person_min_confidence = 0.4;
        let updated = catalog
            .request_analyzer_config(0, settings.clone())
            .expect("analyzer settings persist");
        assert_eq!(updated.revision, 1);
        assert_eq!(updated.settings, settings);
        assert!(matches!(
            catalog.request_analyzer_config(0, CatalogAnalyzerSettings::default()),
            Err(CatalogError::Conflict(_))
        ));

        let mut invalid = settings;
        invalid.thresholds.frame_edge_margin = 0.5;
        assert!(matches!(
            catalog.request_analyzer_config(1, invalid),
            Err(CatalogError::InvalidCatalog(_))
        ));
        catalog.close();

        let reopened = Catalog::open(root).unwrap();
        assert_eq!(reopened.analyzer_config().unwrap(), updated);
    }

    #[test]
    fn detach_and_delete_are_blocked_by_the_shared_active_processing_lease() {
        let temporary = tempdir().expect("temp directory");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let catalog = registry
            .create_catalog(temporary.path().join("catalog"), "Lifecycle")
            .unwrap();
        let id = catalog.descriptor().id.clone();
        let mut contract = catalog.contract_state().unwrap();
        contract
            .analyzer_versions
            .insert("tagger".to_owned(), "model@sha256:keep".to_owned());
        contract
            .checkpoints
            .insert("scanner".to_owned(), serde_json::json!({"row": 17}));
        contract.processing = CatalogProcessingProgress {
            state: CatalogProcessingState::Running,
            updated_at: catalog_timestamp_now(),
            ..CatalogProcessingProgress::default()
        };
        catalog.set_contract_state(&contract).unwrap();
        let lease = CatalogProcessingLease::try_acquire(&catalog).unwrap();
        assert!(matches!(
            registry.detach(&id),
            Err(CatalogError::Conflict(_))
        ));
        assert!(matches!(
            registry.delete_on_disk(&id),
            Err(CatalogError::Conflict(_))
        ));
        assert!(registry.get(&id).is_ok());
        assert!(catalog.root().exists());
        drop(lease);
        catalog.close();
        let detached = registry
            .detach(&id)
            .expect("stale running state without a lease detaches");
        let reconciled = Catalog::open(&detached.path).unwrap();
        let reconciled_contract = reconciled.contract_state().unwrap();
        let progress = reconciled_contract.processing;
        assert_eq!(progress.state, CatalogProcessingState::Failed);
        assert!(progress
            .message
            .as_deref()
            .is_some_and(|message| message.contains("interrupted")));
        assert_eq!(
            reconciled_contract.analyzer_versions["tagger"],
            "model@sha256:keep"
        );
        assert_eq!(reconciled_contract.checkpoints["scanner"]["row"], 17);
        reconciled.close();
        registry.attach(&detached.path).expect("catalog reattaches");
        registry
            .delete_on_disk(&id)
            .expect("reconciled catalog deletes safely");
        assert!(!detached.path.exists());
    }

    #[test]
    fn unavailable_or_replaced_catalog_can_be_detached_without_touching_disk() {
        let temporary = tempdir().expect("temp directory");
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let missing = registry
            .create_catalog(temporary.path().join("missing"), "Missing")
            .unwrap();
        let missing_id = missing.descriptor().id.clone();
        let missing_path = missing.root().to_owned();
        missing.close();
        fs::remove_dir_all(&missing_path).expect("catalog becomes unavailable");

        let detached_missing = registry
            .detach(&missing_id)
            .expect("missing catalog detaches from the registry");
        assert_eq!(detached_missing.id, missing_id);
        assert!(!missing_path.exists());

        let original = registry
            .create_catalog(temporary.path().join("replaced"), "Original")
            .unwrap();
        let original_id = original.descriptor().id.clone();
        let original_path = original.root().to_owned();
        original.close();
        fs::remove_dir_all(&original_path).expect("original catalog removes");
        let replacement = Catalog::create(&original_path, "Replacement").unwrap();
        let replacement_id = replacement.descriptor().id.clone();
        replacement.close();

        let detached_replaced = registry
            .detach(&original_id)
            .expect("stale identity detaches from the registry");
        assert_eq!(detached_replaced.id, original_id);
        let replacement = Catalog::open(&original_path).expect("replacement remains untouched");
        assert_eq!(replacement.descriptor().id, replacement_id);
    }
}
