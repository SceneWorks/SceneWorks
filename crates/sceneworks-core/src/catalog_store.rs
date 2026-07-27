use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
use std::time::Duration;

use rusqlite::types::Value as SqlValue;
use rusqlite::{
    params, params_from_iter, Connection, ErrorCode, OptionalExtension, Transaction,
    TransactionBehavior,
};
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
const CATALOG_DATABASE_SCHEMA_VERSION: u32 = 5;
const REGISTRY_SCHEMA_VERSION: u32 = 1;
const MAX_REGISTRY_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PAGE_SIZE: u32 = 10_000;
const MAX_CONTRACT_STATE_BYTES: usize = 1024 * 1024;
const MAX_QUERY_FILTERS: usize = 16;
const MAX_FILTER_VALUES: usize = 64;
const MAX_FACET_FIELDS: usize = 16;
const MAX_FACET_VALUES: u32 = 200;
const MAX_FACET_VALUE_BYTES: u32 = 512;
const MAX_SEARCH_QUERY_BYTES: usize = 1024;
const MAX_SAVED_VIEWS: u64 = 200;
const MAX_REJECTION_REASON_BYTES: usize = 2_048;
const CATALOG_WAL_AUTOCHECKPOINT_PAGES: u32 = 20_000;
const CATALOG_HEIGHT_INDEX_NAME: &str = "idx_catalog_search_height";
const CATALOG_RECORD_LOOKUP_BATCH_SIZE: usize = 400;

#[derive(Clone, Copy)]
struct CatalogIndexSpec {
    name: &'static str,
    columns: &'static [&'static str],
    predicate: Option<&'static str>,
}

const CATALOG_HEIGHT_INDEX_SPEC: CatalogIndexSpec = CatalogIndexSpec {
    name: CATALOG_HEIGHT_INDEX_NAME,
    columns: &["height", "record_id"],
    predicate: Some("height is not null"),
};

const CURATION_INDEX_SPECS: &[CatalogIndexSpec] = &[
    CatalogIndexSpec {
        name: "idx_catalog_search_medium",
        columns: &["medium", "record_id"],
        predicate: Some("medium is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_people",
        columns: &["person_count", "record_id"],
        predicate: Some("person_count is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_faces",
        columns: &["face_count", "record_id"],
        predicate: Some("face_count is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_full_body",
        columns: &["full_body", "record_id"],
        predicate: Some("full_body is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_qualified_full_body",
        columns: &["qualified_single_full_body", "record_id"],
        predicate: Some("qualified_single_full_body is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_pose",
        columns: &["pose_coverage", "record_id"],
        predicate: Some("pose_coverage is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_crop",
        columns: &["crop_state", "record_id"],
        predicate: Some("crop_state is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_subject_size",
        columns: &["subject_size", "record_id"],
        predicate: Some("subject_size is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_dimensions",
        columns: &["width", "height", "record_id"],
        predicate: None,
    },
    CATALOG_HEIGHT_INDEX_SPEC,
    CatalogIndexSpec {
        name: "idx_catalog_search_availability",
        columns: &["availability", "record_id"],
        predicate: None,
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_confidence",
        columns: &["analyzer_confidence", "record_id"],
        predicate: Some("analyzer_confidence is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_sample",
        columns: &["sample_key", "record_id"],
        predicate: None,
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_primary_flow",
        columns: &[
            "medium",
            "person_count",
            "qualified_single_full_body",
            "sample_key",
            "record_id",
        ],
        predicate: Some(
            "medium is not null and person_count is not null \
             and qualified_single_full_body is not null",
        ),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_content_hash",
        columns: &["content_hash", "record_id"],
        predicate: Some("content_hash is not null"),
    },
    CatalogIndexSpec {
        name: "idx_catalog_search_perceptual_hash",
        columns: &["perceptual_hash", "record_id"],
        predicate: Some("perceptual_hash is not null"),
    },
];

const SOURCE_CONFIG_METADATA_KEY: &str = "source_config";
const ANALYZER_VERSIONS_METADATA_KEY: &str = "analyzer_versions";
const CHECKPOINTS_METADATA_KEY: &str = "checkpoints";
const PROGRESS_METADATA_KEY: &str = "processing_progress";
const PROCESSING_CONTROL_METADATA_KEY: &str = "processing_control";
const ANALYZER_CONFIG_METADATA_KEY: &str = "analyzer_config";
const CURATION_PROJECTION_CHECKPOINT_METADATA_KEY: &str = "curation_projection_v1_checkpoint";
const CURATION_PROJECTION_COMPLETE_METADATA_KEY: &str = "curation_projection_v1_complete";
const CATALOG_RECORD_COUNT_METADATA_KEY: &str = "catalog_record_count";
const CURATION_BACKFILL_BATCH_SIZE: u32 = 1_000;
const CATALOG_RECORD_WRITE_BATCH_SIZE: usize = 50;
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
    CURATION_PROJECTION_CHECKPOINT_METADATA_KEY,
    CURATION_PROJECTION_COMPLETE_METADATA_KEY,
    CATALOG_RECORD_COUNT_METADATA_KEY,
];

pub type CatalogResult<T> = Result<T, CatalogError>;

#[cfg(test)]
thread_local! {
    static CURATION_BACKFILL_BEFORE_WRITE_TX_HOOK:
        std::cell::RefCell<Option<Box<dyn FnOnce()>>> = std::cell::RefCell::new(None);
    static CURATION_BACKFILL_AFTER_SOURCE_READ_HOOK:
        std::cell::RefCell<Option<Box<dyn FnOnce()>>> = std::cell::RefCell::new(None);
}

#[cfg(test)]
fn set_curation_backfill_test_hooks(
    before_write_tx: impl FnOnce() + 'static,
    after_source_read: impl FnOnce() + 'static,
) {
    CURATION_BACKFILL_BEFORE_WRITE_TX_HOOK.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(before_write_tx));
    });
    CURATION_BACKFILL_AFTER_SOURCE_READ_HOOK.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(after_source_read));
    });
}

#[cfg(test)]
fn run_curation_backfill_before_write_tx_hook() {
    CURATION_BACKFILL_BEFORE_WRITE_TX_HOOK.with(|hook| {
        if let Some(hook) = hook.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(not(test))]
fn run_curation_backfill_before_write_tx_hook() {}

#[cfg(test)]
fn run_curation_backfill_after_source_read_hook() {
    CURATION_BACKFILL_AFTER_SOURCE_READ_HOOK.with(|hook| {
        if let Some(hook) = hook.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(not(test))]
fn run_curation_backfill_after_source_read_hook() {}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CatalogSearchRequest {
    #[serde(default)]
    pub cursor: Option<String>,
    pub limit: u32,
    #[serde(default)]
    pub filters: Vec<CatalogRecordFilter>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub min_width: Option<u32>,
    #[serde(default)]
    pub max_width: Option<u32>,
    #[serde(default)]
    pub min_height: Option<u32>,
    #[serde(default)]
    pub max_height: Option<u32>,
    #[serde(default)]
    pub min_analyzer_confidence: Option<f64>,
    #[serde(default)]
    pub sample_seed: Option<u64>,
    #[serde(default)]
    pub deduplicate: bool,
    #[serde(default)]
    pub include_total: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatalogSearchPage {
    pub records: Vec<CatalogRecord>,
    pub next_cursor: Option<String>,
    pub total_count: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CatalogReviewDecision {
    #[default]
    Default,
    Include,
    Exclude,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogRecordReview {
    pub record_id: String,
    pub decision: CatalogReviewDecision,
    pub rejection_reason: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogSavedView {
    pub id: String,
    pub name: String,
    pub query: CatalogSearchRequest,
    pub revision: u64,
    pub created_at: String,
    pub updated_at: String,
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
        backfill_curation_projection(&connection, &database_path)?;
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
        backfill_curation_projection(&connection, &database_path)?;
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
        self.append_records_metadata_and_progress(records, metadata, None)
    }

    /// Atomically appends a bounded scanner batch, advances its caller-owned
    /// checkpoint metadata, and publishes typed processing progress.
    ///
    /// Reserved catalog metadata remains inaccessible through `metadata`;
    /// processor progress can only be written through its validated type.
    pub fn append_records_and_metadata_with_progress(
        &mut self,
        records: &[NewCatalogRecord],
        metadata: &[(&str, &str)],
        progress: &CatalogProcessingProgress,
    ) -> CatalogResult<usize> {
        self.append_records_metadata_and_progress(records, metadata, Some(progress))
    }

    fn append_records_metadata_and_progress(
        &mut self,
        records: &[NewCatalogRecord],
        metadata: &[(&str, &str)],
        progress: Option<&CatalogProcessingProgress>,
    ) -> CatalogResult<usize> {
        for record in records {
            validate_record(record)?;
        }
        for (key, value) in metadata {
            validate_metadata_entry(key, value)?;
        }
        if let Some(progress) = progress {
            validate_processing_progress(progress)?;
        }
        let progress_payload = progress.map(serde_json::to_string).transpose()?;
        let mut transaction_metadata =
            Vec::with_capacity(metadata.len() + usize::from(progress_payload.is_some()));
        transaction_metadata.extend_from_slice(metadata);
        if let Some(progress_payload) = progress_payload.as_deref() {
            transaction_metadata.push((PROGRESS_METADATA_KEY, progress_payload));
        }
        let database_path = self.database_path();
        let transaction = self
            .connection
            // Acquire the WAL writer slot before the existing-id read. A
            // deferred read transaction can otherwise lose a snapshot upgrade
            // race and fail immediately with SQLITE_BUSY_SNAPSHOT.
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let existing_record_ids = query_existing_record_ids(
            &transaction,
            &database_path,
            records.iter().map(|record| record.id.as_str()),
        )?;
        let mut unique_record_ids = HashSet::with_capacity(records.len());
        let added_record_count = records
            .iter()
            .filter(|record| {
                !existing_record_ids.contains(record.id.as_str())
                    && unique_record_ids.insert(record.id.as_str())
            })
            .count() as u64;
        upsert_catalog_records(&transaction, &database_path, records)?;
        upsert_search_projection(
            &transaction,
            &database_path,
            records.iter(),
            Some(&existing_record_ids),
        )?;
        upsert_catalog_metadata(
            &transaction,
            &database_path,
            &transaction_metadata,
            added_record_count,
        )?;
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
        validate_record_id(record_id)?;
        self.connection
            .query_row(
                "select exists(select 1 from catalog_records where id = ?1)",
                [record_id],
                |row| row.get(0),
            )
            .map_err(|error| map_sqlite_error(&self.database_path(), error))
    }

    pub fn existing_record_ids(&self, record_ids: &[String]) -> CatalogResult<HashSet<String>> {
        for record_id in record_ids {
            validate_record_id(record_id)?;
        }
        query_existing_record_ids(
            &self.connection,
            &self.database_path(),
            record_ids.iter().map(String::as_str),
        )
    }

    pub fn record_by_id(&self, record_id: &str) -> CatalogResult<CatalogRecord> {
        validate_record_id(record_id)?;
        let database_path = self.database_path();
        let row = self
            .connection
            .query_row(
                "select id, image_path, thumbnail_path, embedding_path, artifact_path,
                        metadata_json, created_at, updated_at
                 from catalog_records where id = ?1",
                [record_id],
                |row| {
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
                },
            )
            .optional()
            .map_err(|error| map_sqlite_error(&database_path, error))?
            .ok_or_else(|| CatalogError::NotFound("Catalog record was not found".to_owned()))?;
        Ok(CatalogRecord {
            id: row.0.clone(),
            image_path: row.1,
            thumbnail_path: row.2,
            embedding_path: row.3,
            artifact_path: row.4,
            metadata: serde_json::from_str(&row.5).map_err(|error| CatalogError::Corrupt {
                path: database_path,
                detail: format!("record {} contains invalid metadata JSON: {error}", row.0),
            })?,
            created_at: row.6,
            updated_at: row.7,
        })
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

    /// Runs the curation query exclusively against the compact indexed
    /// projection. The opaque cursor is bound to the complete query, including
    /// seed and deduplication, so it cannot be replayed against another view.
    pub fn search_records(
        &self,
        request: &CatalogSearchRequest,
    ) -> CatalogResult<CatalogSearchPage> {
        validate_search_request(request)?;
        let fingerprint = search_fingerprint(request)?;
        let cursor =
            parse_search_cursor(request.cursor.as_deref(), &fingerprint, request.sample_seed)?;
        let database_path = self.database_path();
        let search_text = fts_query(&request.text)?;
        let mut sql = String::from(
            "select records.rowid, records.id, records.image_path, records.thumbnail_path,
                    records.embedding_path, records.artifact_path, records.metadata_json,
                    records.created_at, records.updated_at, search.sample_key
             from catalog_records records
             join catalog_record_search search on search.record_id = records.id
             left join catalog_record_reviews review on review.record_id = records.id",
        );
        sql.push_str(" where 1 = 1");
        let mut bindings = Vec::new();
        append_search_predicates(&mut sql, &mut bindings, request, search_text.as_deref())?;

        let sample_start = request
            .sample_seed
            .map(|seed| stable_sample_key(&format!("seed:{seed}")));
        let (sample_phase, last_key, last_rowid) = match cursor {
            Some(SearchCursor::Row { rowid }) => (0_u8, 0_i64, rowid),
            Some(SearchCursor::Sample {
                phase,
                sample_key,
                rowid,
                ..
            }) => (phase, sample_key, rowid),
            None => (0, sample_start.unwrap_or_default(), 0),
        };
        if let Some(start) = sample_start {
            if sample_phase == 0 {
                sql.push_str(
                    " and search.sample_key >= ?
                      and (search.sample_key > ? or
                           (search.sample_key = ? and records.rowid > ?))",
                );
                bindings.extend([
                    SqlValue::Integer(start),
                    SqlValue::Integer(last_key),
                    SqlValue::Integer(last_key),
                    SqlValue::Integer(last_rowid),
                ]);
            } else {
                sql.push_str(
                    " and search.sample_key < ?
                      and (search.sample_key > ? or
                           (search.sample_key = ? and records.rowid > ?))",
                );
                bindings.extend([
                    SqlValue::Integer(start),
                    SqlValue::Integer(last_key),
                    SqlValue::Integer(last_key),
                    SqlValue::Integer(last_rowid),
                ]);
            }
            sql.push_str(" order by search.sample_key, records.rowid");
        } else {
            sql.push_str(" and records.rowid > ? order by records.rowid");
            bindings.push(SqlValue::Integer(last_rowid));
        }
        sql.push_str(" limit ?");
        bindings.push(SqlValue::Integer(i64::from(request.limit) + 1));
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
                    row.get::<_, i64>(9)?,
                ))
            })
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let mut decoded = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let has_more = decoded.len() > request.limit as usize;
        if has_more {
            decoded.pop();
        }
        let wrap_cursor = request
            .sample_seed
            .filter(|_| sample_phase == 0)
            .map(|seed| format!("v1:{fingerprint}:s:{seed}:1:-1:0"));
        let next_cursor = decoded.last().and_then(|row| {
            if has_more {
                Some(if let Some(seed) = request.sample_seed {
                    format!(
                        "v1:{fingerprint}:s:{seed}:{sample_phase}:{}:{}",
                        row.9, row.0
                    )
                } else {
                    format!("v1:{fingerprint}:r:{}", row.0)
                })
            } else {
                wrap_cursor.clone()
            }
        });
        // If a seed begins beyond the highest matching sample key, advance to
        // the wrap phase without asking the client to guess a new query.
        let next_cursor = if decoded.is_empty() {
            wrap_cursor
        } else {
            next_cursor
        };
        let mut records = Vec::with_capacity(decoded.len());
        for row in decoded {
            let metadata = serde_json::from_str(&row.6).map_err(|error| CatalogError::Corrupt {
                path: database_path.clone(),
                detail: format!("record {} contains invalid metadata JSON: {error}", row.1),
            })?;
            records.push(CatalogRecord {
                id: row.1,
                image_path: row.2,
                thumbnail_path: row.3,
                embedding_path: row.4,
                artifact_path: row.5,
                metadata,
                created_at: row.7,
                updated_at: row.8,
            });
        }
        let total_count = request
            .include_total
            .then(|| self.search_count(request, search_text.as_deref()))
            .transpose()?;
        Ok(CatalogSearchPage {
            records,
            next_cursor,
            total_count,
        })
    }

    fn search_count(
        &self,
        request: &CatalogSearchRequest,
        search_text: Option<&str>,
    ) -> CatalogResult<u64> {
        let mut sql = String::from(
            "select count(*)
             from catalog_records records
             join catalog_record_search search on search.record_id = records.id
             left join catalog_record_reviews review on review.record_id = records.id",
        );
        sql.push_str(" where 1 = 1");
        let mut bindings = Vec::new();
        append_search_predicates(&mut sql, &mut bindings, request, search_text)?;
        self.connection
            .query_row(&sql, params_from_iter(bindings), |row| row.get(0))
            .map_err(|error| map_sqlite_error(&self.database_path(), error))
    }

    pub fn search_facet_counts(
        &self,
        request: &CatalogSearchRequest,
        fields: &[String],
        max_values_per_facet: u32,
    ) -> CatalogResult<Vec<CatalogFacet>> {
        validate_search_request(request)?;
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
        let search_text = fts_query(&request.text)?;
        let database_path = self.database_path();
        let mut facets = Vec::with_capacity(fields.len());
        for field in fields {
            let (column, _) = search_column(field, "search", "review")?;
            let mut sql = String::from("select cast(");
            sql.push_str(&column);
            sql.push_str(
                " as text) facet_value, count(*)
                 from catalog_records records
                 join catalog_record_search search on search.record_id = records.id
                 left join catalog_record_reviews review on review.record_id = records.id",
            );
            sql.push_str(" where ");
            sql.push_str(&column);
            sql.push_str(" is not null");
            let mut bindings = Vec::new();
            append_search_predicates(&mut sql, &mut bindings, request, search_text.as_deref())?;
            sql.push_str(" group by facet_value order by count(*) desc, facet_value limit ?");
            bindings.push(SqlValue::Integer(i64::from(max_values_per_facet)));
            let mut statement = self
                .connection
                .prepare(&sql)
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            let values = statement
                .query_map(params_from_iter(bindings), |row| {
                    Ok(CatalogFacetCount {
                        value: row.get(0)?,
                        count: row.get(1)?,
                    })
                })
                .map_err(|error| map_sqlite_error(&database_path, error))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| map_sqlite_error(&database_path, error))?;
            facets.push(CatalogFacet {
                field: field.clone(),
                values,
            });
        }
        Ok(facets)
    }

    pub fn record_reviews(&self, record_ids: &[String]) -> CatalogResult<Vec<CatalogRecordReview>> {
        if record_ids.len() > MAX_PAGE_SIZE as usize {
            return Err(CatalogError::InvalidCatalog(
                "Too many catalog reviews requested".to_owned(),
            ));
        }
        let mut reviews = Vec::new();
        let mut statement = self
            .connection
            .prepare_cached(
                "select decision, rejection_reason, updated_at
                 from catalog_record_reviews where record_id = ?1",
            )
            .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
        for record_id in record_ids {
            validate_record_id(record_id)?;
            if let Some((decision, rejection_reason, updated_at)) = statement
                .query_row([record_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .optional()
                .map_err(|error| map_sqlite_error(&self.database_path(), error))?
            {
                reviews.push(CatalogRecordReview {
                    record_id: record_id.clone(),
                    decision: parse_review_decision(&decision)?,
                    rejection_reason,
                    updated_at,
                });
            }
        }
        Ok(reviews)
    }

    pub fn set_record_review(
        &self,
        record_id: &str,
        decision: CatalogReviewDecision,
        rejection_reason: Option<String>,
    ) -> CatalogResult<Option<CatalogRecordReview>> {
        validate_record_id(record_id)?;
        if !self.contains_record_id(record_id)? {
            return Err(CatalogError::NotFound(
                "Catalog record was not found".to_owned(),
            ));
        }
        let reason = rejection_reason
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        if reason
            .as_ref()
            .is_some_and(|value| value.len() > MAX_REJECTION_REASON_BYTES || value.contains('\0'))
        {
            return Err(CatalogError::InvalidCatalog(
                "Catalog rejection reason is invalid".to_owned(),
            ));
        }
        match decision {
            CatalogReviewDecision::Default => {
                self.connection
                    .execute(
                        "delete from catalog_record_reviews where record_id = ?1",
                        [record_id],
                    )
                    .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
                Ok(None)
            }
            CatalogReviewDecision::Include => {
                let updated_at = utc_now();
                self.connection
                    .execute(
                        "insert into catalog_record_reviews(
                            record_id, decision, rejection_reason, updated_at
                         ) values (?1, 'include', null, ?2)
                         on conflict(record_id) do update set
                            decision = 'include', rejection_reason = null,
                            updated_at = excluded.updated_at",
                        params![record_id, updated_at],
                    )
                    .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
                Ok(Some(CatalogRecordReview {
                    record_id: record_id.to_owned(),
                    decision,
                    rejection_reason: None,
                    updated_at,
                }))
            }
            CatalogReviewDecision::Exclude => {
                let reason = reason.ok_or_else(|| {
                    CatalogError::InvalidCatalog(
                        "Excluded catalog records require a rejection reason".to_owned(),
                    )
                })?;
                let updated_at = utc_now();
                self.connection
                    .execute(
                        "insert into catalog_record_reviews(
                            record_id, decision, rejection_reason, updated_at
                         ) values (?1, 'exclude', ?2, ?3)
                         on conflict(record_id) do update set
                            decision = 'exclude', rejection_reason = excluded.rejection_reason,
                            updated_at = excluded.updated_at",
                        params![record_id, reason, updated_at],
                    )
                    .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
                Ok(Some(CatalogRecordReview {
                    record_id: record_id.to_owned(),
                    decision,
                    rejection_reason: Some(reason),
                    updated_at,
                }))
            }
        }
    }

    pub fn saved_views(&self) -> CatalogResult<Vec<CatalogSavedView>> {
        let database_path = self.database_path();
        let mut statement = self
            .connection
            .prepare(
                "select id, name, query_json, revision, created_at, updated_at
                 from catalog_saved_views order by name collate nocase, id",
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let views = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, u64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })
            .map_err(|error| map_sqlite_error(&database_path, error))?
            .map(|row| {
                let (id, name, query_json, revision, created_at, updated_at) =
                    row.map_err(|error| map_sqlite_error(&database_path, error))?;
                Ok(CatalogSavedView {
                    id,
                    name,
                    query: serde_json::from_str(&query_json)?,
                    revision,
                    created_at,
                    updated_at,
                })
            })
            .collect();
        views
    }

    pub fn save_view(
        &self,
        id: Option<&str>,
        name: String,
        mut query: CatalogSearchRequest,
        expected_revision: Option<u64>,
    ) -> CatalogResult<CatalogSavedView> {
        let name = validated_saved_view_name(name)?;
        query.cursor = None;
        query.include_total = false;
        validate_search_request(&query)?;
        let query_json = serde_json::to_string(&query)?;
        let now = utc_now();
        let id = match id {
            Some(id) => {
                validate_saved_view_id(id)?;
                let expected = expected_revision.ok_or_else(|| {
                    CatalogError::Conflict(
                        "Updating a saved view requires its current revision".to_owned(),
                    )
                })?;
                let changed = self
                    .connection
                    .execute(
                        "update catalog_saved_views
                         set name = ?1, query_json = ?2, revision = revision + 1, updated_at = ?3
                        where id = ?4 and revision = ?5",
                        params![name, query_json, now, id, expected],
                    )
                    .map_err(|error| map_saved_view_write_error(&self.database_path(), error))?;
                if changed == 0 {
                    return Err(CatalogError::Conflict(
                        "Saved view changed; refresh and retry".to_owned(),
                    ));
                }
                id.to_owned()
            }
            None => {
                let count: u64 = self
                    .connection
                    .query_row("select count(*) from catalog_saved_views", [], |row| {
                        row.get(0)
                    })
                    .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
                if count >= MAX_SAVED_VIEWS {
                    return Err(CatalogError::InvalidCatalog(format!(
                        "Catalogs support at most {MAX_SAVED_VIEWS} saved views"
                    )));
                }
                let id = random_hex(16)?;
                self.connection
                    .execute(
                        "insert into catalog_saved_views(
                            id, name, query_json, revision, created_at, updated_at
                         ) values (?1, ?2, ?3, 0, ?4, ?4)",
                        params![id, name, query_json, now],
                    )
                    .map_err(|error| map_saved_view_write_error(&self.database_path(), error))?;
                id
            }
        };
        let (revision, created_at) = self
            .connection
            .query_row(
                "select revision, created_at from catalog_saved_views where id = ?1",
                [&id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
        Ok(CatalogSavedView {
            id,
            name,
            query,
            revision,
            created_at,
            updated_at: now,
        })
    }

    pub fn delete_saved_view(&self, id: &str, expected_revision: u64) -> CatalogResult<()> {
        validate_saved_view_id(id)?;
        let changed = self
            .connection
            .execute(
                "delete from catalog_saved_views where id = ?1 and revision = ?2",
                params![id, expected_revision],
            )
            .map_err(|error| map_sqlite_error(&self.database_path(), error))?;
        if changed == 0 {
            return Err(CatalogError::Conflict(
                "Saved view changed; refresh and retry".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn contract_state(&self) -> CatalogResult<CatalogContractState> {
        let database_path = self.database_path();
        let mut statement = self
            .connection
            .prepare(
                "select key, value from catalog_metadata
                 where key in (?1, ?2, ?3, ?4)",
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let mut payloads = statement
            .query_map(
                params![
                    SOURCE_CONFIG_METADATA_KEY,
                    ANALYZER_VERSIONS_METADATA_KEY,
                    CHECKPOINTS_METADATA_KEY,
                    PROGRESS_METADATA_KEY
                ],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(|error| map_sqlite_error(&database_path, error))?
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(|error| map_sqlite_error(&database_path, error))?;
        let mut processing: CatalogProcessingProgress = parse_contract_payload(
            &database_path,
            PROGRESS_METADATA_KEY,
            payloads.remove(PROGRESS_METADATA_KEY),
        )?
        .unwrap_or_else(|| CatalogProcessingProgress {
            updated_at: self.descriptor.created_at.clone(),
            ..CatalogProcessingProgress::default()
        });
        if processing.updated_at.is_empty() {
            processing.updated_at = self.descriptor.created_at.clone();
        }
        Ok(CatalogContractState {
            source_config: parse_contract_payload(
                &database_path,
                SOURCE_CONFIG_METADATA_KEY,
                payloads.remove(SOURCE_CONFIG_METADATA_KEY),
            )?,
            analyzer_versions: parse_contract_payload(
                &database_path,
                ANALYZER_VERSIONS_METADATA_KEY,
                payloads.remove(ANALYZER_VERSIONS_METADATA_KEY),
            )?
            .unwrap_or_default(),
            checkpoints: parse_contract_payload(
                &database_path,
                CHECKPOINTS_METADATA_KEY,
                payloads.remove(CHECKPOINTS_METADATA_KEY),
            )?
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
        parse_contract_payload(&database_path, key, payload)
    }

    pub fn storage_accounting(&self) -> CatalogResult<CatalogStorageAccounting> {
        let database_bytes = sqlite_storage_bytes(&self.root)?;
        let manifest_bytes = fs::metadata(self.root.join(CATALOG_MANIFEST_FILE))?.len();
        let total_bytes = directory_file_bytes(&self.root)?;
        let artifact_bytes = total_bytes.saturating_sub(database_bytes + manifest_bytes);
        let record_count = self.record_count()?;
        Ok(CatalogStorageAccounting {
            database_bytes,
            manifest_bytes,
            artifact_bytes,
            total_bytes,
            record_count,
        })
    }

    pub fn record_count(&self) -> CatalogResult<u64> {
        catalog_record_count(&self.connection, &self.database_path())
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
    validate_record_id(&record.id)?;
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

#[derive(Debug)]
struct CatalogSearchProjection {
    sample_key: i64,
    medium: Option<String>,
    person_count: Option<i64>,
    face_count: Option<i64>,
    full_body: Option<i64>,
    qualified_single_full_body: Option<i64>,
    pose_coverage: Option<f64>,
    crop_state: Option<String>,
    subject_size: Option<f64>,
    width: Option<i64>,
    height: Option<i64>,
    availability: String,
    analyzer_confidence: Option<f64>,
    content_hash: Option<String>,
    perceptual_hash: Option<String>,
    caption: String,
    tags: String,
}

fn search_projection(record: &NewCatalogRecord) -> CatalogSearchProjection {
    let metadata = &record.metadata;
    let string_at = |paths: &[&str]| {
        paths
            .iter()
            .find_map(|path| metadata.pointer(path).and_then(Value::as_str))
            .map(str::to_owned)
    };
    let integer_at = |paths: &[&str]| {
        paths.iter().find_map(|path| {
            metadata
                .pointer(path)
                .and_then(|value| value.as_i64().or_else(|| value.as_u64()?.try_into().ok()))
        })
    };
    let float_at = |paths: &[&str]| {
        paths
            .iter()
            .find_map(|path| metadata.pointer(path).and_then(Value::as_f64))
    };
    let boolean_at = |path: &str| {
        metadata
            .pointer(path)
            .and_then(Value::as_bool)
            .map(i64::from)
    };
    let content_hash = string_at(&[
        "/acquisition/contentHash",
        "/contentHash",
        "/analysis/structured/inputFingerprint",
    ])
    .filter(|value| value != "unavailable");
    CatalogSearchProjection {
        sample_key: stable_sample_key(&record.id),
        medium: string_at(&["/analysis/medium", "/medium"]),
        person_count: integer_at(&["/analysis/personCount", "/personCount"]),
        face_count: integer_at(&["/analysis/faceCount"]),
        full_body: boolean_at("/analysis/fullBody"),
        qualified_single_full_body: boolean_at("/analysis/qualifiedSingleFullBody"),
        pose_coverage: float_at(&["/analysis/poseCoverage"]),
        crop_state: string_at(&["/analysis/cropState"]),
        subject_size: float_at(&["/analysis/subjectFrameFraction"]),
        width: integer_at(&["/acquisition/width", "/width"]),
        height: integer_at(&["/acquisition/height", "/height"]),
        availability: string_at(&["/acquisition/availability", "/availability"])
            .unwrap_or_else(|| "unknown".to_owned()),
        analyzer_confidence: float_at(&[
            "/analysis/mediumConfidence",
            "/analysis/visionLanguage/medium/confidence",
            "/analysis/structured/derived/analyzer/confidence",
            "/analysis/structured/person/analyzer/confidence",
            "/analysis/structured/face/analyzer/confidence",
            "/analysis/structured/pose/analyzer/confidence",
        ]),
        content_hash,
        perceptual_hash: string_at(&["/perceptualHash", "/analysis/perceptualHash"]),
        caption: searchable_caption(metadata),
        tags: searchable_tags(metadata),
    }
}

fn upsert_catalog_records(
    transaction: &Transaction<'_>,
    database_path: &Path,
    records: &[NewCatalogRecord],
) -> CatalogResult<()> {
    let timestamp = utc_now();
    for records in records.chunks(CATALOG_RECORD_WRITE_BATCH_SIZE) {
        let mut sql = String::from(
            "insert into catalog_records (
                id, image_path, thumbnail_path, embedding_path, artifact_path,
                metadata_json, created_at, updated_at
             ) values ",
        );
        let mut bindings = Vec::with_capacity(records.len() * 8);
        for (index, record) in records.iter().enumerate() {
            if index > 0 {
                sql.push(',');
            }
            sql.push_str("(?,?,?,?,?,?,?,?)");
            bindings.extend([
                SqlValue::Text(record.id.clone()),
                SqlValue::Text(record.image_path.clone()),
                record
                    .thumbnail_path
                    .clone()
                    .map_or(SqlValue::Null, SqlValue::Text),
                record
                    .embedding_path
                    .clone()
                    .map_or(SqlValue::Null, SqlValue::Text),
                record
                    .artifact_path
                    .clone()
                    .map_or(SqlValue::Null, SqlValue::Text),
                SqlValue::Text(serde_json::to_string(&record.metadata)?),
                SqlValue::Text(timestamp.clone()),
                SqlValue::Text(timestamp.clone()),
            ]);
        }
        sql.push_str(
            " on conflict(id) do update set
                image_path = excluded.image_path,
                thumbnail_path = excluded.thumbnail_path,
                embedding_path = excluded.embedding_path,
                artifact_path = excluded.artifact_path,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at",
        );
        transaction
            .prepare_cached(&sql)
            .and_then(|mut statement| statement.execute(params_from_iter(bindings)))
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    Ok(())
}

fn upsert_catalog_metadata(
    transaction: &Transaction<'_>,
    database_path: &Path,
    metadata: &[(&str, &str)],
    added_record_count: u64,
) -> CatalogResult<()> {
    if metadata.is_empty() && added_record_count == 0 {
        return Ok(());
    }
    let mut entries = metadata
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect::<Vec<_>>();
    if added_record_count > 0 {
        entries.push((
            CATALOG_RECORD_COUNT_METADATA_KEY.to_owned(),
            added_record_count.to_string(),
        ));
    }
    for entries in entries.chunks(CATALOG_RECORD_WRITE_BATCH_SIZE) {
        let mut sql = String::from("insert into catalog_metadata(key, value) values ");
        let mut bindings = Vec::with_capacity(entries.len() * 2);
        for (index, (key, value)) in entries.iter().enumerate() {
            if index > 0 {
                sql.push(',');
            }
            sql.push_str("(?,?)");
            bindings.extend([
                SqlValue::Text(key.to_owned()),
                SqlValue::Text(value.to_owned()),
            ]);
        }
        sql.push_str(
            " on conflict(key) do update set value =
                case when excluded.key = 'catalog_record_count'
                     then cast(cast(catalog_metadata.value as integer)
                               + cast(excluded.value as integer) as text)
                     else excluded.value end",
        );
        transaction
            .prepare_cached(&sql)
            .and_then(|mut statement| statement.execute(params_from_iter(bindings)))
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    Ok(())
}

fn query_existing_record_ids<'a>(
    connection: &Connection,
    database_path: &Path,
    record_ids: impl IntoIterator<Item = &'a str>,
) -> CatalogResult<HashSet<String>> {
    let record_ids = record_ids.into_iter().collect::<Vec<_>>();
    let mut existing = HashSet::new();
    for record_ids in record_ids.chunks(CATALOG_RECORD_LOOKUP_BATCH_SIZE) {
        let mut sql = String::from("select id from catalog_records where id in (");
        let mut bindings = Vec::with_capacity(record_ids.len());
        for (index, record_id) in record_ids.iter().enumerate() {
            if index > 0 {
                sql.push(',');
            }
            sql.push('?');
            bindings.push(SqlValue::Text((*record_id).to_owned()));
        }
        sql.push(')');
        let mut statement = connection
            .prepare_cached(&sql)
            .map_err(|error| map_sqlite_error(database_path, error))?;
        let ids = statement
            .query_map(params_from_iter(bindings), |row| row.get::<_, String>(0))
            .map_err(|error| map_sqlite_error(database_path, error))?;
        for id in ids {
            existing.insert(id.map_err(|error| map_sqlite_error(database_path, error))?);
        }
    }
    Ok(existing)
}

fn upsert_search_projection<'a>(
    transaction: &Transaction<'_>,
    database_path: &Path,
    records: impl IntoIterator<Item = &'a NewCatalogRecord>,
    existing_record_ids: Option<&HashSet<String>>,
) -> CatalogResult<()> {
    let records = records.into_iter().collect::<Vec<_>>();
    // Preserve the existing last-write-wins behavior for callers that include
    // the same record id more than once, including across write chunks.
    let mut seen = HashSet::with_capacity(records.len());
    let mut unique = records
        .into_iter()
        .rev()
        .filter(|record| seen.insert(record.id.as_str()))
        .collect::<Vec<_>>();
    unique.reverse();
    for unique in unique.chunks(CATALOG_RECORD_WRITE_BATCH_SIZE) {
        let projected = unique
            .iter()
            .map(|record| search_projection(record))
            .collect::<Vec<_>>();

        let mut index_sql = String::from(
            "insert into catalog_record_search (
                record_id, sample_key, medium, person_count, face_count, full_body,
                qualified_single_full_body, pose_coverage, crop_state, subject_size,
                width, height, availability, analyzer_confidence, content_hash,
                perceptual_hash
             ) values ",
        );
        let mut index_bindings = Vec::with_capacity(unique.len() * 16);
        for (index, (record, projection)) in unique.iter().zip(&projected).enumerate() {
            if index > 0 {
                index_sql.push(',');
            }
            index_sql.push_str("(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
            index_bindings.extend([
                SqlValue::Text(record.id.clone()),
                SqlValue::Integer(projection.sample_key),
                projection
                    .medium
                    .clone()
                    .map_or(SqlValue::Null, SqlValue::Text),
                projection
                    .person_count
                    .map_or(SqlValue::Null, SqlValue::Integer),
                projection
                    .face_count
                    .map_or(SqlValue::Null, SqlValue::Integer),
                projection
                    .full_body
                    .map_or(SqlValue::Null, SqlValue::Integer),
                projection
                    .qualified_single_full_body
                    .map_or(SqlValue::Null, SqlValue::Integer),
                projection
                    .pose_coverage
                    .map_or(SqlValue::Null, SqlValue::Real),
                projection
                    .crop_state
                    .clone()
                    .map_or(SqlValue::Null, SqlValue::Text),
                projection
                    .subject_size
                    .map_or(SqlValue::Null, SqlValue::Real),
                projection.width.map_or(SqlValue::Null, SqlValue::Integer),
                projection.height.map_or(SqlValue::Null, SqlValue::Integer),
                SqlValue::Text(projection.availability.clone()),
                projection
                    .analyzer_confidence
                    .map_or(SqlValue::Null, SqlValue::Real),
                projection
                    .content_hash
                    .clone()
                    .map_or(SqlValue::Null, SqlValue::Text),
                projection
                    .perceptual_hash
                    .clone()
                    .map_or(SqlValue::Null, SqlValue::Text),
            ]);
        }
        let requires_upsert = existing_record_ids.map_or(true, |existing| {
            unique
                .iter()
                .any(|record| existing.contains(record.id.as_str()))
        });
        if requires_upsert {
            index_sql.push_str(
                " on conflict(record_id) do update set
                sample_key = excluded.sample_key,
                medium = excluded.medium,
                person_count = excluded.person_count,
                face_count = excluded.face_count,
                full_body = excluded.full_body,
                qualified_single_full_body = excluded.qualified_single_full_body,
                pose_coverage = excluded.pose_coverage,
                crop_state = excluded.crop_state,
                subject_size = excluded.subject_size,
                width = excluded.width,
                height = excluded.height,
                availability = excluded.availability,
                analyzer_confidence = excluded.analyzer_confidence,
                content_hash = excluded.content_hash,
                perceptual_hash = excluded.perceptual_hash",
            );
        }
        transaction
            .prepare_cached(&index_sql)
            .and_then(|mut statement| statement.execute(params_from_iter(index_bindings)))
            .map_err(|error| map_sqlite_error(database_path, error))?;

        let replaced = unique
            .iter()
            .filter(|record| {
                existing_record_ids.map_or(true, |existing| existing.contains(record.id.as_str()))
            })
            .collect::<Vec<_>>();
        if !replaced.is_empty() {
            let mut delete_sql = String::from(
                "delete from catalog_record_fts where rowid in (
                    select rowid from catalog_records where id in (",
            );
            let mut delete_bindings = Vec::with_capacity(replaced.len());
            for (index, record) in replaced.iter().enumerate() {
                if index > 0 {
                    delete_sql.push(',');
                }
                delete_sql.push('?');
                delete_bindings.push(SqlValue::Text(record.id.clone()));
            }
            delete_sql.push_str("))");
            transaction
                .prepare_cached(&delete_sql)
                .and_then(|mut statement| statement.execute(params_from_iter(delete_bindings)))
                .map_err(|error| map_sqlite_error(database_path, error))?;
        }

        let mut fts_sql =
            String::from("insert into catalog_record_fts(rowid, record_id, caption, tags) values ");
        let mut fts_bindings = Vec::with_capacity(unique.len() * 4);
        for (index, (record, projection)) in unique.iter().zip(&projected).enumerate() {
            if index > 0 {
                fts_sql.push(',');
            }
            fts_sql.push_str("((select rowid from catalog_records where id = ?),?,?,?)");
            fts_bindings.extend([
                SqlValue::Text(record.id.clone()),
                SqlValue::Text(record.id.clone()),
                SqlValue::Text(projection.caption.clone()),
                SqlValue::Text(projection.tags.clone()),
            ]);
        }
        transaction
            .prepare_cached(&fts_sql)
            .and_then(|mut statement| statement.execute(params_from_iter(fts_bindings)))
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum SearchCursor {
    Row {
        rowid: i64,
    },
    Sample {
        phase: u8,
        sample_key: i64,
        rowid: i64,
    },
}

fn validate_search_request(request: &CatalogSearchRequest) -> CatalogResult<()> {
    validate_page_size(request.limit)?;
    if request.limit > 200 {
        return Err(CatalogError::InvalidCatalog(
            "Catalog curation pages support at most 200 records".to_owned(),
        ));
    }
    if request.text.len() > MAX_SEARCH_QUERY_BYTES || request.text.contains('\0') {
        return Err(CatalogError::InvalidCatalog(
            "Catalog search text is invalid".to_owned(),
        ));
    }
    if request.filters.len() > MAX_QUERY_FILTERS {
        return Err(CatalogError::InvalidCatalog(format!(
            "Catalog queries support at most {MAX_QUERY_FILTERS} filters"
        )));
    }
    for filter in &request.filters {
        search_column(&filter.field, "search", "review")?;
        if filter.values.is_empty() || filter.values.len() > MAX_FILTER_VALUES {
            return Err(CatalogError::InvalidCatalog(format!(
                "Each catalog filter requires 1 to {MAX_FILTER_VALUES} values"
            )));
        }
        if filter
            .values
            .iter()
            .any(|value| value.len() > 128 || value.contains('\0'))
        {
            return Err(CatalogError::InvalidCatalog(
                "Catalog curation filter values are invalid".to_owned(),
            ));
        }
    }
    for (minimum, maximum, label) in [
        (request.min_width, request.max_width, "width"),
        (request.min_height, request.max_height, "height"),
    ] {
        if minimum
            .zip(maximum)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Err(CatalogError::InvalidCatalog(format!(
                "Catalog {label} minimum cannot exceed its maximum"
            )));
        }
    }
    if request
        .min_analyzer_confidence
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog analyzer confidence must be between 0 and 1".to_owned(),
        ));
    }
    Ok(())
}

fn search_column(
    field: &str,
    search_alias: &str,
    review_alias: &str,
) -> CatalogResult<(String, bool)> {
    match field {
        "medium" | "analysis.medium" => Ok((format!("{search_alias}.medium"), false)),
        "personCount" | "analysis.personCount" => {
            Ok((format!("{search_alias}.person_count"), true))
        }
        "faceCount" | "analysis.faceCount" => Ok((format!("{search_alias}.face_count"), true)),
        "fullBody" | "analysis.fullBody" => Ok((format!("{search_alias}.full_body"), true)),
        "qualifiedSingleFullBody" | "analysis.qualifiedSingleFullBody" => {
            Ok((format!("{search_alias}.qualified_single_full_body"), true))
        }
        "poseCoverage" | "analysis.poseCoverage" => {
            Ok((format!("{search_alias}.pose_coverage"), true))
        }
        "cropState" | "analysis.cropState" => Ok((format!("{search_alias}.crop_state"), false)),
        "subjectSize" | "analysis.subjectFrameFraction" => {
            Ok((format!("{search_alias}.subject_size"), true))
        }
        "width" => Ok((format!("{search_alias}.width"), true)),
        "height" => Ok((format!("{search_alias}.height"), true)),
        "availability" | "acquisition.availability" => {
            Ok((format!("{search_alias}.availability"), false))
        }
        "analyzerConfidence" | "analysis.mediumConfidence" => {
            Ok((format!("{search_alias}.analyzer_confidence"), true))
        }
        "reviewDecision" => Ok((
            format!("coalesce({review_alias}.decision, 'default')"),
            false,
        )),
        _ => Err(CatalogError::InvalidCatalog(format!(
            "Unsupported indexed catalog filter: {field}"
        ))),
    }
}

fn append_search_predicates(
    sql: &mut String,
    bindings: &mut Vec<SqlValue>,
    request: &CatalogSearchRequest,
    search_text: Option<&str>,
) -> CatalogResult<()> {
    append_search_predicates_for(
        sql,
        bindings,
        request,
        search_text,
        "records",
        "search",
        "review",
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_search_predicates_for(
    sql: &mut String,
    bindings: &mut Vec<SqlValue>,
    request: &CatalogSearchRequest,
    search_text: Option<&str>,
    records_alias: &str,
    search_alias: &str,
    review_alias: &str,
    include_deduplication: bool,
) -> CatalogResult<()> {
    if let Some(search_text) = search_text {
        sql.push_str(" and ");
        sql.push_str(records_alias);
        sql.push_str(
            ".rowid in (
                select rowid from catalog_record_fts
                where catalog_record_fts match ?
             )",
        );
        bindings.push(SqlValue::Text(search_text.to_owned()));
    }
    for filter in &request.filters {
        let (column, numeric) = search_column(&filter.field, search_alias, review_alias)?;
        sql.push_str(" and ");
        sql.push_str(&column);
        sql.push_str(" in (");
        for (index, value) in filter.values.iter().enumerate() {
            if index > 0 {
                sql.push(',');
            }
            sql.push('?');
            if numeric {
                let normalized = match value.as_str() {
                    "true" => SqlValue::Integer(1),
                    "false" => SqlValue::Integer(0),
                    _ => value.parse::<f64>().map(SqlValue::Real).map_err(|_| {
                        CatalogError::InvalidCatalog(format!(
                            "Catalog filter {} requires numeric values",
                            filter.field
                        ))
                    })?,
                };
                bindings.push(normalized);
            } else {
                bindings.push(SqlValue::Text(value.clone()));
            }
        }
        sql.push(')');
    }
    for (field, value, operator) in [
        ("width", request.min_width, ">="),
        ("width", request.max_width, "<="),
        ("height", request.min_height, ">="),
        ("height", request.max_height, "<="),
    ] {
        if let Some(value) = value {
            sql.push_str(" and ");
            sql.push_str(search_alias);
            sql.push('.');
            sql.push_str(field);
            sql.push(' ');
            sql.push_str(operator);
            sql.push_str(" ?");
            bindings.push(SqlValue::Integer(i64::from(value)));
        }
    }
    if let Some(value) = request.min_analyzer_confidence {
        sql.push_str(" and ");
        sql.push_str(search_alias);
        sql.push_str(".analyzer_confidence >= ?");
        bindings.push(SqlValue::Real(value));
    }
    if include_deduplication && request.deduplicate {
        sql.push_str(
            " and not exists (
                select 1
                from catalog_records duplicate_records
                join catalog_record_search duplicate
                  on duplicate.record_id = duplicate_records.id
                left join catalog_record_reviews duplicate_review
                  on duplicate_review.record_id = duplicate_records.id
                where duplicate.record_id < ",
        );
        sql.push_str(search_alias);
        sql.push_str(
            ".record_id
                  and (
                    (",
        );
        sql.push_str(search_alias);
        sql.push_str(
            ".perceptual_hash is not null
                     and duplicate.perceptual_hash = ",
        );
        sql.push_str(search_alias);
        sql.push_str(
            ".perceptual_hash)
                    or
                    (",
        );
        sql.push_str(search_alias);
        sql.push_str(
            ".content_hash is not null
                     and duplicate.content_hash = ",
        );
        sql.push_str(search_alias);
        sql.push_str(
            ".content_hash)
                  )",
        );
        append_search_predicates_for(
            sql,
            bindings,
            request,
            search_text,
            "duplicate_records",
            "duplicate",
            "duplicate_review",
            false,
        )?;
        sql.push(')');
    }
    Ok(())
}

fn fts_query(text: &str) -> CatalogResult<Option<String>> {
    let tokens = text
        .split_whitespace()
        .map(|token| token.trim_matches(|character: char| !character.is_alphanumeric()))
        .filter(|token| !token.is_empty())
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect::<Vec<_>>();
    if !text.trim().is_empty() && tokens.is_empty() {
        return Err(CatalogError::InvalidCatalog(
            "Catalog search requires letters or numbers".to_owned(),
        ));
    }
    Ok((!tokens.is_empty()).then(|| tokens.join(" AND ")))
}

fn search_fingerprint(request: &CatalogSearchRequest) -> CatalogResult<String> {
    use sha2::{Digest, Sha256};

    let mut normalized = request.clone();
    normalized.cursor = None;
    normalized.include_total = false;
    let digest = Sha256::digest(serde_json::to_vec(&normalized)?);
    Ok(format!("{digest:x}")[..24].to_owned())
}

fn parse_search_cursor(
    cursor: Option<&str>,
    expected_fingerprint: &str,
    expected_seed: Option<u64>,
) -> CatalogResult<Option<SearchCursor>> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };
    let parts = cursor.split(':').collect::<Vec<_>>();
    if parts.len() == 4 && parts[0] == "v1" && parts[1] == expected_fingerprint && parts[2] == "r" {
        let rowid = parts[3].parse::<i64>().ok().filter(|value| *value >= 0);
        return rowid
            .map(|rowid| Some(SearchCursor::Row { rowid }))
            .ok_or_else(|| CatalogError::InvalidCatalog("Catalog cursor is invalid".to_owned()));
    }
    if parts.len() == 7 && parts[0] == "v1" && parts[1] == expected_fingerprint && parts[2] == "s" {
        let seed = parts[3].parse::<u64>().ok();
        let phase = parts[4].parse::<u8>().ok().filter(|value| *value <= 1);
        let sample_key = parts[5].parse::<i64>().ok();
        let rowid = parts[6].parse::<i64>().ok().filter(|value| *value >= 0);
        if let (Some(seed), Some(phase), Some(sample_key), Some(rowid)) =
            (seed, phase, sample_key, rowid)
        {
            if expected_seed == Some(seed) {
                return Ok(Some(SearchCursor::Sample {
                    phase,
                    sample_key,
                    rowid,
                }));
            }
        }
    }
    Err(CatalogError::InvalidCatalog(
        "Catalog cursor does not match this query".to_owned(),
    ))
}

fn stable_sample_key(value: &str) -> i64 {
    // FNV-1a gives an architecture-independent stable ordering key. Masking the
    // sign bit keeps SQLite ordering and cursor serialization straightforward.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash & i64::MAX as u64) as i64
}

fn searchable_caption(metadata: &Value) -> String {
    metadata
        .get("caption")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn searchable_tags(metadata: &Value) -> String {
    let mut tags = HashSet::new();
    if let Some(membership) = metadata
        .pointer("/analysis/tagMembership")
        .and_then(Value::as_object)
    {
        tags.extend(
            membership
                .iter()
                .filter(|(_, included)| included.as_bool() == Some(true))
                .map(|(tag, _)| tag.as_str()),
        );
    }
    if let Some(values) = metadata
        .pointer("/analysis/visionLanguage/tags")
        .and_then(Value::as_array)
    {
        tags.extend(values.iter().filter_map(|value| {
            value
                .get("value")
                .and_then(Value::as_str)
                .or_else(|| value.as_str())
        }));
    }
    let mut tags = tags.into_iter().collect::<Vec<_>>();
    tags.sort_unstable();
    tags.join(" ")
}

fn validate_record_id(record_id: &str) -> CatalogResult<()> {
    if record_id.trim().is_empty() || record_id.len() > 512 || record_id.contains('\0') {
        return Err(CatalogError::InvalidCatalog(
            "Catalog record id must contain 1 to 512 non-NUL bytes".to_owned(),
        ));
    }
    Ok(())
}

fn parse_review_decision(value: &str) -> CatalogResult<CatalogReviewDecision> {
    match value {
        "default" => Ok(CatalogReviewDecision::Default),
        "include" => Ok(CatalogReviewDecision::Include),
        "exclude" => Ok(CatalogReviewDecision::Exclude),
        _ => Err(CatalogError::Corrupt {
            path: PathBuf::from(CATALOG_DATABASE_FILE),
            detail: "catalog review has an invalid decision".to_owned(),
        }),
    }
}

fn validated_saved_view_name(name: String) -> CatalogResult<String> {
    let name = name.trim();
    if name.is_empty() || name.len() > 128 || name.contains('\0') {
        return Err(CatalogError::InvalidCatalog(
            "Saved view name must contain 1 to 128 non-NUL bytes".to_owned(),
        ));
    }
    Ok(name.to_owned())
}

fn validate_saved_view_id(id: &str) -> CatalogResult<()> {
    if id.len() != 32 || !id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CatalogError::InvalidCatalog(
            "Saved view id is invalid".to_owned(),
        ));
    }
    Ok(())
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
    if (settings.vision_analysis_enabled || settings.semantic_embeddings_enabled)
        && !settings.structured_analysis_enabled
    {
        return Err(CatalogError::InvalidCatalog(
            "Catalog vision analysis and semantic embeddings require structured analysis"
                .to_owned(),
        ));
    }
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
    use rusqlite::functions::FunctionFlags;

    connection
        .create_scalar_function(
            "sceneworks_sample_key",
            1,
            FunctionFlags::SQLITE_DETERMINISTIC | FunctionFlags::SQLITE_INNOCUOUS,
            |context| {
                let record_id = context.get::<String>(0)?;
                Ok(stable_sample_key(&record_id))
            },
        )
        .map_err(|error| map_sqlite_error(database_path, error))?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let journal_mode: String = connection
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        connection
            .pragma_update(None, "journal_mode", "wal")
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
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
        // Projection-heavy scanner batches dirty several indexes at once.
        // Keep automatic checkpoints bounded (about 80 MiB at SQLite's
        // default page size) without forcing checkpoint work into every few
        // small scanner commits.
        .pragma_update(None, "wal_autocheckpoint", CATALOG_WAL_AUTOCHECKPOINT_PAGES)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    Ok(())
}

fn migrate(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    let mut version: u32 = connection
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if version > CATALOG_DATABASE_SCHEMA_VERSION {
        return Err(CatalogError::Incompatible {
            found: version,
            supported: CATALOG_DATABASE_SCHEMA_VERSION,
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
        version = 1;
    }
    if version < 2 {
        connection
            .execute_batch(
                "begin immediate;
             create table if not exists catalog_record_search (
                record_id text primary key not null
                    references catalog_records(id) on delete cascade,
                sample_key integer not null,
                medium text,
                person_count integer,
                face_count integer,
                full_body integer,
                qualified_single_full_body integer,
                pose_coverage real,
                crop_state text,
                subject_size real,
                width integer,
                height integer,
                availability text,
                analyzer_confidence real,
                content_hash text,
                perceptual_hash text
             ) strict;
             create virtual table if not exists catalog_record_fts using fts5(
                record_id unindexed, caption, tags, tokenize = 'unicode61'
             );
             create table if not exists catalog_record_reviews (
                record_id text primary key not null
                    references catalog_records(id) on delete cascade,
                decision text not null check(decision in ('default', 'include', 'exclude')),
                rejection_reason text,
                updated_at text not null
             ) strict;
             create index if not exists idx_catalog_reviews_decision
                on catalog_record_reviews(decision, record_id);
             create table if not exists catalog_saved_views (
                id text primary key not null,
                name text not null,
                query_json text not null,
                created_at text not null,
                updated_at text not null,
                revision integer not null default 0
             ) strict;
             create unique index if not exists idx_catalog_saved_views_name
                on catalog_saved_views(name collate nocase);
             commit;",
            )
            .map_err(|error| map_sqlite_error(database_path, error))?;
        if !table_has_column(
            connection,
            database_path,
            "catalog_record_search",
            "qualified_single_full_body",
        )? {
            connection
                .execute_batch(
                    "begin immediate;
                     alter table catalog_record_search rename to catalog_record_search_v1;
                     create table catalog_record_search (
                        record_id text primary key not null
                            references catalog_records(id) on delete cascade,
                        sample_key integer not null,
                        medium text,
                        person_count integer,
                        face_count integer,
                        full_body integer,
                        qualified_single_full_body integer,
                        pose_coverage real,
                        crop_state text,
                        subject_size real,
                        width integer,
                        height integer,
                        availability text,
                        analyzer_confidence real,
                        content_hash text,
                        perceptual_hash text
                     ) strict;
                     insert into catalog_record_search (
                        record_id, sample_key, medium, person_count, face_count, full_body,
                        pose_coverage, crop_state, subject_size, width, height, availability,
                        analyzer_confidence, content_hash, perceptual_hash
                     )
                     select record_id, sample_key, medium, person_count, face_count, full_body,
                            pose_coverage, crop_state, subject_size, width, height, availability,
                            analyzer_confidence, content_hash, perceptual_hash
                     from catalog_record_search_v1;
                     drop table catalog_record_search_v1;
                     commit;",
                )
                .map_err(|error| map_sqlite_error(database_path, error))?;
        }
        if !table_has_column(connection, database_path, "catalog_saved_views", "revision")? {
            connection
                .execute(
                    "alter table catalog_saved_views
                 add column revision integer not null default 0",
                    [],
                )
                .map_err(|error| map_sqlite_error(database_path, error))?;
        }
        connection
            .execute_batch(
                "begin immediate;
                 pragma user_version = 2;
                 commit;",
            )
            .map_err(|error| map_sqlite_error(database_path, error))?;
        version = 2;
    }
    if version < 3 {
        migrate_curation_indexes_v3(connection, database_path)?;
        version = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    if version < 4 {
        migrate_height_index_v4(connection, database_path)?;
        version = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    if version < 5 {
        migrate_record_count_v5(connection, database_path)?;
    }
    Ok(())
}

fn curation_index_sql(spec: CatalogIndexSpec) -> String {
    let mut sql = format!(
        "create index {} on catalog_record_search({})",
        spec.name,
        spec.columns.join(", ")
    );
    if let Some(predicate) = spec.predicate {
        sql.push_str(" where ");
        sql.push_str(predicate);
    }
    sql
}

fn migrate_curation_indexes_v3(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    let transaction = Transaction::new_unchecked(connection, TransactionBehavior::Immediate)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let current_version: u32 = transaction
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if current_version >= 3 {
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(database_path, error))?;
        return Ok(());
    }
    // The canonical set includes the height index introduced by v4. Keep the
    // v3 migration faithful to the schema that shipped before that addition.
    for spec in CURATION_INDEX_SPECS
        .iter()
        .filter(|spec| spec.name != CATALOG_HEIGHT_INDEX_NAME)
    {
        transaction
            .execute_batch(&format!(
                "drop index if exists {}; {}",
                spec.name,
                curation_index_sql(*spec)
            ))
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
    transaction
        .pragma_update(None, "user_version", 3)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    transaction
        .commit()
        .map_err(|error| map_sqlite_error(database_path, error))
}

fn migrate_height_index_v4(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    let transaction = Transaction::new_unchecked(connection, TransactionBehavior::Immediate)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let current_version: u32 = transaction
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if current_version >= 4 {
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(database_path, error))?;
        return Ok(());
    }
    transaction
        .execute_batch(&format!(
            "drop index if exists {}; {}",
            CATALOG_HEIGHT_INDEX_SPEC.name,
            curation_index_sql(CATALOG_HEIGHT_INDEX_SPEC)
        ))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    transaction
        .pragma_update(None, "user_version", 4)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    transaction
        .commit()
        .map_err(|error| map_sqlite_error(database_path, error))
}

fn migrate_record_count_v5(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    let transaction = Transaction::new_unchecked(connection, TransactionBehavior::Immediate)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let current_version: u32 = transaction
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if current_version >= 5 {
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(database_path, error))?;
        return Ok(());
    }
    transaction
        .execute(
            "insert into catalog_metadata(key, value)
             select ?1, cast(count(*) as text) from catalog_records
             where true
             on conflict(key) do update set value = excluded.value",
            [CATALOG_RECORD_COUNT_METADATA_KEY],
        )
        .map_err(|error| map_sqlite_error(database_path, error))?;
    transaction
        .pragma_update(None, "user_version", 5)
        .map_err(|error| map_sqlite_error(database_path, error))?;
    transaction
        .commit()
        .map_err(|error| map_sqlite_error(database_path, error))
}

fn table_has_column(
    connection: &Connection,
    database_path: &Path,
    table: &str,
    column: &str,
) -> CatalogResult<bool> {
    let sql = format!("pragma table_info({table})");
    connection
        .prepare(&sql)
        .and_then(|mut statement| {
            let columns = statement
                .query_map([], |row| row.get::<_, String>("name"))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(columns.iter().any(|candidate| candidate == column))
        })
        .map_err(|error| map_sqlite_error(database_path, error))
}

fn backfill_curation_projection(
    connection: &Connection,
    database_path: &Path,
) -> CatalogResult<()> {
    let complete: Option<String> = connection
        .query_row(
            "select value from catalog_metadata where key = ?1",
            [CURATION_PROJECTION_COMPLETE_METADATA_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| map_sqlite_error(database_path, error))?;
    if complete.as_deref() == Some("true") {
        return Ok(());
    }
    loop {
        // Serialize incomplete migrations before taking the record snapshot.
        // This prevents another opener from advancing the checkpoint while
        // this batch is being projected and prevents record upserts from
        // committing between the source read and projection write.
        run_curation_backfill_before_write_tx_hook();
        let transaction = Transaction::new_unchecked(connection, TransactionBehavior::Immediate)
            .map_err(|error| map_sqlite_error(database_path, error))?;
        let complete: Option<String> = transaction
            .query_row(
                "select value from catalog_metadata where key = ?1",
                [CURATION_PROJECTION_COMPLETE_METADATA_KEY],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| map_sqlite_error(database_path, error))?;
        if complete.as_deref() == Some("true") {
            transaction
                .commit()
                .map_err(|error| map_sqlite_error(database_path, error))?;
            return Ok(());
        }
        let checkpoint = transaction
            .query_row(
                "select value from catalog_metadata where key = ?1",
                [CURATION_PROJECTION_CHECKPOINT_METADATA_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| map_sqlite_error(database_path, error))?
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or_default();
        let batch = {
            let mut statement = transaction
                .prepare(
                    "select rowid, id, image_path, thumbnail_path, embedding_path,
                            artifact_path, metadata_json
                     from catalog_records
                     where rowid > ?1
                     order by rowid
                     limit ?2",
                )
                .map_err(|error| map_sqlite_error(database_path, error))?;
            statement
                .query_map(params![checkpoint, CURATION_BACKFILL_BATCH_SIZE], |row| {
                    let metadata_json: String = row.get(6)?;
                    let metadata = serde_json::from_str(&metadata_json).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            metadata_json.len(),
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })?;
                    Ok((
                        row.get::<_, i64>(0)?,
                        NewCatalogRecord {
                            id: row.get(1)?,
                            image_path: row.get(2)?,
                            thumbnail_path: row.get(3)?,
                            embedding_path: row.get(4)?,
                            artifact_path: row.get(5)?,
                            metadata,
                        },
                    ))
                })
                .and_then(Iterator::collect::<Result<Vec<_>, _>>)
                .map_err(|error| map_sqlite_error(database_path, error))?
        };
        run_curation_backfill_after_source_read_hook();
        if batch.is_empty() {
            transaction
                .execute(
                    "insert into catalog_metadata(key, value) values (?1, 'true')
                     on conflict(key) do update set value = excluded.value",
                    [CURATION_PROJECTION_COMPLETE_METADATA_KEY],
                )
                .map_err(|error| map_sqlite_error(database_path, error))?;
            transaction
                .execute(
                    "delete from catalog_metadata where key = ?1",
                    [CURATION_PROJECTION_CHECKPOINT_METADATA_KEY],
                )
                .map_err(|error| map_sqlite_error(database_path, error))?;
            transaction
                .commit()
                .map_err(|error| map_sqlite_error(database_path, error))?;
            return Ok(());
        }
        upsert_search_projection(
            &transaction,
            database_path,
            batch.iter().map(|(_, record)| record),
            None,
        )?;
        let checkpoint = batch.last().map(|(rowid, _)| *rowid).unwrap_or(checkpoint);
        transaction
            .execute(
                "insert into catalog_metadata(key, value) values (?1, ?2)
                 on conflict(key) do update set value = excluded.value",
                params![
                    CURATION_PROJECTION_CHECKPOINT_METADATA_KEY,
                    checkpoint.to_string()
                ],
            )
            .map_err(|error| map_sqlite_error(database_path, error))?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(database_path, error))?;
    }
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
    validate_table_columns(
        connection,
        database_path,
        "catalog_record_search",
        &[
            ("record_id", "TEXT", 1, 1),
            ("sample_key", "INTEGER", 1, 0),
            ("medium", "TEXT", 0, 0),
            ("person_count", "INTEGER", 0, 0),
            ("face_count", "INTEGER", 0, 0),
            ("full_body", "INTEGER", 0, 0),
            ("qualified_single_full_body", "INTEGER", 0, 0),
            ("pose_coverage", "REAL", 0, 0),
            ("crop_state", "TEXT", 0, 0),
            ("subject_size", "REAL", 0, 0),
            ("width", "INTEGER", 0, 0),
            ("height", "INTEGER", 0, 0),
            ("availability", "TEXT", 0, 0),
            ("analyzer_confidence", "REAL", 0, 0),
            ("content_hash", "TEXT", 0, 0),
            ("perceptual_hash", "TEXT", 0, 0),
        ],
    )?;
    validate_table_columns(
        connection,
        database_path,
        "catalog_record_reviews",
        &[
            ("record_id", "TEXT", 1, 1),
            ("decision", "TEXT", 1, 0),
            ("rejection_reason", "TEXT", 0, 0),
            ("updated_at", "TEXT", 1, 0),
        ],
    )?;
    validate_table_columns(
        connection,
        database_path,
        "catalog_saved_views",
        &[
            ("id", "TEXT", 1, 1),
            ("name", "TEXT", 1, 0),
            ("query_json", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
            ("revision", "INTEGER", 1, 0),
        ],
    )?;
    catalog_record_count(connection, database_path)?;
    validate_curation_indexes(connection, database_path)?;

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

fn catalog_record_count(connection: &Connection, database_path: &Path) -> CatalogResult<u64> {
    let value: Option<String> = connection
        .query_row(
            "select value from catalog_metadata where key = ?1",
            [CATALOG_RECORD_COUNT_METADATA_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| map_sqlite_error(database_path, error))?;
    value
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| CatalogError::Corrupt {
            path: database_path.to_path_buf(),
            detail: "catalog record count metadata is missing or invalid".to_owned(),
        })
}

fn parse_contract_payload<T: for<'de> Deserialize<'de>>(
    database_path: &Path,
    key: &str,
    payload: Option<String>,
) -> CatalogResult<Option<T>> {
    let Some(payload) = payload else {
        return Ok(None);
    };
    if payload.len() > MAX_CONTRACT_STATE_BYTES {
        return Err(CatalogError::Corrupt {
            path: database_path.to_path_buf(),
            detail: format!("catalog metadata {key} exceeds its size limit"),
        });
    }
    serde_json::from_str(&payload)
        .map(Some)
        .map_err(|error| CatalogError::Corrupt {
            path: database_path.to_path_buf(),
            detail: format!("catalog metadata {key} is invalid: {error}"),
        })
}

fn normalize_index_sql(sql: &str) -> String {
    sql.chars()
        .filter(|character| !character.is_ascii_whitespace())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn validate_curation_indexes(connection: &Connection, database_path: &Path) -> CatalogResult<()> {
    let mut statement = connection
        .prepare(
            "select master.name, master.tbl_name, master.sql, info.seqno, info.name
             from sqlite_master master
             left join pragma_index_info(master.name) info
             where master.type = 'index'
               and master.name glob 'idx_catalog_search_*'
             order by master.name, info.seqno",
        )
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })
        .map_err(|error| map_sqlite_error(database_path, error))?;
    let mut definitions = BTreeMap::<String, (String, Option<String>, Vec<String>)>::new();
    for row in rows {
        let (name, owner, sql, sequence, column) =
            row.map_err(|error| map_sqlite_error(database_path, error))?;
        let entry = definitions
            .entry(name)
            .or_insert_with(|| (owner, sql, Vec::new()));
        if sequence == Some(entry.2.len() as i64) {
            if let Some(column) = column {
                entry.2.push(column);
            }
        }
    }
    for spec in CURATION_INDEX_SPECS {
        let expected_sql = curation_index_sql(*spec);
        let definition_matches = definitions
            .get(spec.name)
            .is_some_and(|(owner, sql, columns)| {
                owner == "catalog_record_search"
                    && sql.as_deref().is_some_and(|sql| {
                        normalize_index_sql(sql) == normalize_index_sql(&expected_sql)
                    })
                    && columns == spec.columns
            });
        if !definition_matches {
            return Err(CatalogError::Corrupt {
                path: database_path.to_path_buf(),
                detail: format!("{} index has the wrong definition", spec.name),
            });
        }
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

fn map_saved_view_write_error(path: &Path, error: rusqlite::Error) -> CatalogError {
    if matches!(
        error.sqlite_error_code(),
        Some(ErrorCode::ConstraintViolation)
    ) {
        CatalogError::Conflict("A saved view with that name already exists".to_owned())
    } else {
        map_sqlite_error(path, error)
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

    fn replace_curation_indexes_with_dense_v2(connection: &Connection) {
        connection.execute_batch("begin immediate;").unwrap();
        for spec in CURATION_INDEX_SPECS
            .iter()
            .filter(|spec| spec.name != CATALOG_HEIGHT_INDEX_NAME)
        {
            connection
                .execute_batch(&format!(
                    "drop index if exists {}; create index {} on catalog_record_search({})",
                    spec.name,
                    spec.name,
                    spec.columns.join(", ")
                ))
                .unwrap();
        }
        connection.pragma_update(None, "user_version", 2).unwrap();
        connection.execute_batch("commit;").unwrap();
    }

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
            writer
                .connection
                .pragma_query_value(None, "wal_autocheckpoint", |row| row.get::<_, u32>(0))
                .unwrap(),
            CATALOG_WAL_AUTOCHECKPOINT_PAGES
        );
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
            .pragma_update(None, "user_version", CATALOG_DATABASE_SCHEMA_VERSION + 1)
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
        assert_eq!(version, CATALOG_DATABASE_SCHEMA_VERSION);
    }

    #[test]
    fn dense_v2_curation_indexes_upgrade_to_exact_latest_definitions() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let mut catalog = Catalog::create(&root, "V2 curation indexes").unwrap();
        catalog
            .append_records(&[curation_record(
                "preserved-record",
                "Preserved projection",
                "sha256:preserved",
                "dhash64:preserved",
            )])
            .unwrap();
        catalog.close();

        let database_path = root.join(CATALOG_DATABASE_FILE);
        let connection = Connection::open(&database_path).unwrap();
        replace_curation_indexes_with_dense_v2(&connection);
        drop(connection);

        let upgraded = Catalog::open(&root).expect("v2 catalog upgrades");
        assert_eq!(
            upgraded
                .connection
                .pragma_query_value(None, "user_version", |row| row.get::<_, u32>(0))
                .unwrap(),
            CATALOG_DATABASE_SCHEMA_VERSION
        );
        assert_eq!(
            upgraded
                .connection
                .query_row(
                    "select medium from catalog_record_search where record_id = 'preserved-record'",
                    [],
                    |row| row.get::<_, Option<String>>(0),
                )
                .unwrap()
                .as_deref(),
            Some("photograph")
        );
        assert_eq!(CURATION_INDEX_SPECS.len(), 16);
        for spec in CURATION_INDEX_SPECS {
            let actual_sql: String = upgraded
                .connection
                .query_row(
                    "select sql from sqlite_master where type = 'index' and name = ?1",
                    [spec.name],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(
                normalize_index_sql(&actual_sql),
                normalize_index_sql(&curation_index_sql(*spec)),
                "{} should have its canonical latest definition",
                spec.name
            );
        }
        let query_plan = |sql: &str| {
            upgraded
                .connection
                .prepare(&format!("explain query plan {sql}"))
                .unwrap()
                .query_map([], |row| row.get::<_, String>(3))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .join("\n")
        };
        let medium_plan =
            query_plan("select record_id from catalog_record_search where medium = 'photograph'");
        assert!(
            medium_plan.contains("idx_catalog_search_medium"),
            "sparse medium index should remain usable: {medium_plan}"
        );
        let height_equality_plan =
            query_plan("select record_id from catalog_record_search where height = 1536");
        assert!(
            height_equality_plan.contains(CATALOG_HEIGHT_INDEX_NAME),
            "height equality should seek the dedicated index: {height_equality_plan}"
        );
        let height_range_plan = query_plan(
            "select record_id from catalog_record_search
             where height >= 1024 and height <= 2048",
        );
        assert!(
            height_range_plan.contains(CATALOG_HEIGHT_INDEX_NAME),
            "height min/max range should seek the dedicated index: {height_range_plan}"
        );
        let height_facet_plan = query_plan(
            "select height, count(*) from catalog_record_search
             where height is not null group by height",
        );
        assert!(
            height_facet_plan.contains(CATALOG_HEIGHT_INDEX_NAME),
            "height facets should scan the covering height index: {height_facet_plan}"
        );
        upgraded.close();

        let writer = Connection::open(&database_path).unwrap();
        writer.execute_batch("begin immediate;").unwrap();
        Catalog::open(&root)
            .expect("completed latest-schema open does not need a writer lock")
            .close();
        writer.execute_batch("rollback;").unwrap();
    }

    #[test]
    fn v3_catalog_adds_height_index_once_during_v4_upgrade() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "V3 catalog").unwrap().close();
        let database_path = root.join(CATALOG_DATABASE_FILE);
        let connection = Connection::open(&database_path).unwrap();
        connection
            .execute_batch(
                "drop index idx_catalog_search_height;
                 pragma user_version = 3;",
            )
            .unwrap();
        drop(connection);

        let upgraded = Catalog::open(&root).expect("v3 catalog upgrades");
        let height_sql: String = upgraded
            .connection
            .query_row(
                "select sql from sqlite_master
                 where type = 'index' and name = 'idx_catalog_search_height'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            normalize_index_sql(&height_sql),
            normalize_index_sql(&curation_index_sql(CATALOG_HEIGHT_INDEX_SPEC))
        );
        assert_eq!(
            upgraded
                .connection
                .pragma_query_value(None, "user_version", |row| row.get::<_, u32>(0))
                .unwrap(),
            CATALOG_DATABASE_SCHEMA_VERSION
        );
        upgraded.close();
    }

    #[test]
    fn v4_catalog_backfills_durable_record_count_during_v5_upgrade() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let mut catalog = Catalog::create(&root, "V4 catalog").unwrap();
        catalog.append_records(&[record("preserved")]).unwrap();
        catalog.close();
        let database_path = root.join(CATALOG_DATABASE_FILE);
        let connection = Connection::open(&database_path).unwrap();
        connection
            .execute(
                "delete from catalog_metadata where key = ?1",
                [CATALOG_RECORD_COUNT_METADATA_KEY],
            )
            .unwrap();
        connection.pragma_update(None, "user_version", 4).unwrap();
        drop(connection);

        let upgraded = Catalog::open(&root).expect("v4 catalog upgrades to v5");
        assert_eq!(upgraded.storage_accounting().unwrap().record_count, 1);
        assert_eq!(
            upgraded
                .connection
                .pragma_query_value(None, "user_version", |row| row.get::<_, u32>(0))
                .unwrap(),
            CATALOG_DATABASE_SCHEMA_VERSION
        );
        upgraded.close();
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
    fn wrong_curation_index_predicate_is_reported_as_corrupt() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Curation index schema")
            .expect("catalog creates")
            .close();
        let connection = Connection::open(root.join(CATALOG_DATABASE_FILE)).unwrap();
        connection
            .execute_batch(
                "drop index idx_catalog_search_medium;
                 create index idx_catalog_search_medium
                    on catalog_record_search(medium, record_id)
                    where medium is null;",
            )
            .unwrap();
        drop(connection);

        assert!(matches!(
            Catalog::open(&root),
            Err(CatalogError::Corrupt { .. })
        ));
    }

    #[test]
    fn malformed_current_height_index_is_reported_as_corrupt() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Height index schema")
            .expect("catalog creates")
            .close();
        let connection = Connection::open(root.join(CATALOG_DATABASE_FILE)).unwrap();
        connection
            .execute_batch(
                "drop index idx_catalog_search_height;
                 create index idx_catalog_search_height
                    on catalog_record_search(height, width, record_id)
                    where height is not null;",
            )
            .unwrap();
        drop(connection);

        assert!(matches!(
            Catalog::open(&root),
            Err(CatalogError::Corrupt { .. })
        ));
    }

    #[test]
    fn malformed_current_record_count_is_reported_as_corrupt() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Record count schema")
            .expect("catalog creates")
            .close();
        let connection = Connection::open(root.join(CATALOG_DATABASE_FILE)).unwrap();
        connection
            .execute(
                "update catalog_metadata set value = 'invalid' where key = ?1",
                [CATALOG_RECORD_COUNT_METADATA_KEY],
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

    fn curation_record(id: &str, caption: &str, hash: &str, perceptual: &str) -> NewCatalogRecord {
        NewCatalogRecord {
            id: id.to_owned(),
            image_path: format!("images/{id}.jpg"),
            thumbnail_path: Some(format!("thumbnails/{id}.jpg")),
            embedding_path: None,
            artifact_path: None,
            metadata: serde_json::json!({
                "caption": caption,
                "acquisition": {
                    "availability": "available",
                    "contentHash": hash,
                    "width": 1024,
                    "height": 1536
                },
                "analysis": {
                    "medium": "photograph",
                    "personCount": 1,
                    "faceCount": 1,
                    "fullBody": true,
                    "qualifiedSingleFullBody": true,
                    "cropState": "uncropped",
                    "poseCoverage": 0.91,
                    "subjectFrameFraction": 0.62,
                    "perceptualHash": perceptual,
                    "visionLanguage": {
                        "medium": { "confidence": 0.97 }
                    },
                    "tagMembership": { "full_body": true, "outdoors": true }
                }
            }),
        }
    }

    #[test]
    fn curation_projection_migration_matches_fresh_upserts_and_fts_is_current() {
        let temporary = tempdir().expect("temp directory");
        let legacy_root = temporary.path().join("legacy");
        let fresh_root = temporary.path().join("fresh");
        let legacy = Catalog::create(&legacy_root, "Legacy").unwrap();
        let descriptor = legacy.descriptor().clone();
        legacy.close();

        let legacy_db = legacy_root.join(CATALOG_DATABASE_FILE);
        let raw = Connection::open(&legacy_db).unwrap();
        raw.execute_batch(
            "drop table catalog_record_search;
             drop table catalog_record_fts;
             drop table catalog_record_reviews;
             drop table catalog_saved_views;
             create table catalog_record_search (
                record_id text primary key not null
                    references catalog_records(id) on delete cascade,
                sample_key integer not null,
                medium text,
                person_count integer,
                face_count integer,
                full_body integer,
                pose_coverage real,
                crop_state text,
                subject_size real,
                width integer,
                height integer,
                availability text,
                analyzer_confidence real,
                content_hash text,
                perceptual_hash text
             ) strict;
             create index idx_catalog_search_primary_flow
                on catalog_record_search(
                    medium, person_count, full_body, sample_key, record_id
                );
             delete from catalog_metadata
                where key in (
                    'curation_projection_v1_checkpoint',
                    'curation_projection_v1_complete'
                );
             pragma user_version = 1;",
        )
        .unwrap();
        let mut record = curation_record("same-id", "placeholder", "sha256:one", "dhash64:1111");
        record.metadata["caption"] =
            serde_json::json!(format!("{} migration_suffix", "long-caption ".repeat(120)));
        record.metadata["analysis"]["visionLanguage"]["tags"] =
            serde_json::json!([{"value": "generated_only_tag"}]);
        raw.execute(
            "insert into catalog_records(
                id, image_path, thumbnail_path, embedding_path, artifact_path,
                metadata_json, created_at, updated_at
             ) values (?1, ?2, ?3, null, null, ?4, ?5, ?5)",
            params![
                record.id,
                record.image_path,
                record.thumbnail_path,
                serde_json::to_string(&record.metadata).unwrap(),
                utc_now(),
            ],
        )
        .unwrap();
        drop(raw);
        // The public manifest remains schema-compatible; opening performs the
        // additive backfill and validates all new tables.
        let manifest = read_manifest(&legacy_root).unwrap();
        assert_eq!(manifest.id, descriptor.id);
        assert_eq!(manifest.schema_version, descriptor.schema_version);
        let migrated = Catalog::open(&legacy_root).unwrap();

        let mut fresh = Catalog::create(&fresh_root, "Fresh").unwrap();
        fresh.append_records(std::slice::from_ref(&record)).unwrap();
        let migrated_key: i64 = migrated
            .connection
            .query_row(
                "select sample_key from catalog_record_search where record_id='same-id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let fresh_key: i64 = fresh
            .connection
            .query_row(
                "select sample_key from catalog_record_search where record_id='same-id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(migrated_key, fresh_key);

        let request = CatalogSearchRequest {
            limit: 10,
            text: "migration_suffix generated_only_tag".to_owned(),
            ..CatalogSearchRequest::default()
        };
        assert_eq!(migrated.search_records(&request).unwrap().records.len(), 1);
        assert_eq!(fresh.search_records(&request).unwrap().records.len(), 1);

        let mut updated = record;
        updated.metadata["caption"] = serde_json::json!("A blue jacket");
        updated.metadata["analysis"]["tagMembership"] = serde_json::json!({"studio": true});
        fresh.append_records(&[updated]).unwrap();
        assert!(fresh.search_records(&request).unwrap().records.is_empty());
        assert_eq!(
            fresh
                .search_records(&CatalogSearchRequest {
                    limit: 10,
                    text: "blue studio".to_owned(),
                    ..CatalogSearchRequest::default()
                })
                .unwrap()
                .records
                .len(),
            1
        );
    }

    #[test]
    fn batched_projection_preserves_last_write_updates_and_sparse_indexes() {
        let temporary = tempdir().expect("temp directory");
        let mut catalog =
            Catalog::create(temporary.path().join("catalog"), "Batched projection").unwrap();
        let records = (0..51)
            .map(|index| {
                let id = if index == 0 || index == 50 {
                    "duplicate-across-chunks".to_owned()
                } else {
                    format!("record-{index:02}")
                };
                let caption = if index == 50 {
                    "latest_duplicate_caption".to_owned()
                } else if index == 0 {
                    "old_duplicate_caption".to_owned()
                } else {
                    format!("unique_caption_{index}")
                };
                let mut record = curation_record(
                    &id,
                    &caption,
                    &format!("sha256:{index}"),
                    &format!("dhash64:{index}"),
                );
                record.image_path = format!("images/write-{index}.jpg");
                if index == 50 {
                    record.metadata["analysis"]["medium"] = serde_json::json!("illustration");
                }
                record
            })
            .collect::<Vec<_>>();
        catalog.append_records(&records).unwrap();
        let persisted = catalog.record_by_id("duplicate-across-chunks").unwrap();
        assert_eq!(persisted.image_path, "images/write-50.jpg");
        assert_eq!(persisted.metadata["caption"], "latest_duplicate_caption");
        assert_eq!(catalog.storage_accounting().unwrap().record_count, 50);
        assert_eq!(
            catalog
                .connection
                .query_row(
                    "select medium, content_hash from catalog_record_search
                     where record_id = 'duplicate-across-chunks'",
                    [],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .unwrap(),
            ("illustration".to_owned(), "sha256:50".to_owned())
        );
        assert_eq!(
            catalog
                .search_records(&CatalogSearchRequest {
                    limit: 10,
                    text: "latest_duplicate_caption".to_owned(),
                    ..CatalogSearchRequest::default()
                })
                .unwrap()
                .records
                .len(),
            1
        );
        assert!(catalog
            .search_records(&CatalogSearchRequest {
                limit: 10,
                text: "old_duplicate_caption".to_owned(),
                ..CatalogSearchRequest::default()
            })
            .unwrap()
            .records
            .is_empty());

        catalog
            .connection
            .execute(
                "update catalog_records
                 set created_at = '2000-01-01T00:00:00Z',
                     updated_at = '2000-01-01T00:00:00Z'
                 where id = 'duplicate-across-chunks'",
                [],
            )
            .unwrap();
        let mut existing = curation_record(
            "duplicate-across-chunks",
            "mixed_existing_update",
            "sha256:existing-update",
            "dhash64:existing-update",
        );
        existing.metadata["analysis"]["medium"] = serde_json::json!("illustration");
        let new = curation_record(
            "mixed-new-record",
            "mixed_new_insert",
            "sha256:mixed-new",
            "dhash64:mixed-new",
        );
        catalog.append_records(&[existing, new]).unwrap();
        let persisted_existing = catalog.record_by_id("duplicate-across-chunks").unwrap();
        assert_eq!(
            persisted_existing.metadata["caption"],
            "mixed_existing_update"
        );
        assert_eq!(persisted_existing.created_at, "2000-01-01T00:00:00Z");
        assert_ne!(persisted_existing.updated_at, "2000-01-01T00:00:00Z");
        let persisted_new = catalog.record_by_id("mixed-new-record").unwrap();
        assert_eq!(persisted_new.metadata["caption"], "mixed_new_insert");
        assert_eq!(catalog.storage_accounting().unwrap().record_count, 51);
        assert_eq!(
            catalog
                .connection
                .query_row(
                    "select medium, content_hash from catalog_record_search
                     where record_id = 'duplicate-across-chunks'",
                    [],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .unwrap(),
            (
                "illustration".to_owned(),
                "sha256:existing-update".to_owned()
            )
        );
        for text in ["mixed_existing_update", "mixed_new_insert"] {
            assert_eq!(
                catalog
                    .search_records(&CatalogSearchRequest {
                        limit: 10,
                        text: text.to_owned(),
                        ..CatalogSearchRequest::default()
                    })
                    .unwrap()
                    .records
                    .len(),
                1
            );
        }
        assert!(catalog
            .search_records(&CatalogSearchRequest {
                limit: 10,
                text: "latest_duplicate_caption".to_owned(),
                ..CatalogSearchRequest::default()
            })
            .unwrap()
            .records
            .is_empty());
        let medium_index_sql: String = catalog
            .connection
            .query_row(
                "select sql from sqlite_master
                 where type = 'index' and name = 'idx_catalog_search_medium'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            medium_index_sql.contains("where medium is not null"),
            "analysis-only facet indexes must stay sparse: {medium_index_sql}"
        );
    }

    #[test]
    fn curation_sampling_dedup_reviews_and_saved_views_are_reproducible() {
        let temporary = tempdir().expect("temp directory");
        let mut catalog =
            Catalog::create(temporary.path().join("catalog"), "Curation").expect("catalog");
        let records = (0..12)
            .map(|index| {
                let duplicate = index == 5;
                let content_hash = if duplicate {
                    "sha256:0".to_owned()
                } else {
                    format!("sha256:{index}")
                };
                let perceptual_hash = if duplicate {
                    "dhash64:0".to_owned()
                } else {
                    format!("dhash64:{index}")
                };
                curation_record(
                    &format!("record-{index:02}"),
                    &format!("Full body person {index}"),
                    &content_hash,
                    &perceptual_hash,
                )
            })
            .collect::<Vec<_>>();
        catalog.append_records(&records).unwrap();
        let base = CatalogSearchRequest {
            limit: 4,
            filters: vec![
                CatalogRecordFilter {
                    field: "medium".to_owned(),
                    values: vec!["photograph".to_owned()],
                },
                CatalogRecordFilter {
                    field: "personCount".to_owned(),
                    values: vec!["1".to_owned()],
                },
                CatalogRecordFilter {
                    field: "qualifiedSingleFullBody".to_owned(),
                    values: vec!["true".to_owned()],
                },
            ],
            sample_seed: Some(41),
            deduplicate: true,
            ..CatalogSearchRequest::default()
        };
        let first = catalog.search_records(&base).unwrap();
        let repeated = catalog.search_records(&base).unwrap();
        assert_eq!(first.records, repeated.records);
        assert!(first.next_cursor.is_some());
        assert_eq!(first.total_count, None, "counting is explicitly opt-in");
        let plan = catalog
            .connection
            .prepare(
                "explain query plan
                 select record_id from catalog_record_search
                 where medium='photograph' and person_count=1
                   and qualified_single_full_body=1
                 order by sample_key, record_id limit 48",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join(" ");
        assert!(
            plan.contains("idx_catalog_search_primary_flow"),
            "primary flow must stay index-backed: {plan}"
        );
        let second = catalog
            .search_records(&CatalogSearchRequest {
                cursor: first.next_cursor.clone(),
                ..base.clone()
            })
            .unwrap();
        assert!(first
            .records
            .iter()
            .all(|record| second.records.iter().all(|next| next.id != record.id)));
        assert!(matches!(
            catalog.search_records(&CatalogSearchRequest {
                text: "changed".to_owned(),
                cursor: first.next_cursor,
                ..base.clone()
            }),
            Err(CatalogError::InvalidCatalog(_))
        ));
        let other_seed = catalog
            .search_records(&CatalogSearchRequest {
                sample_seed: Some(278_020),
                ..base.clone()
            })
            .unwrap();
        assert_ne!(
            first
                .records
                .iter()
                .map(|record| &record.id)
                .collect::<Vec<_>>(),
            other_seed
                .records
                .iter()
                .map(|record| &record.id)
                .collect::<Vec<_>>()
        );

        assert!(catalog
            .set_record_review(
                "record-00",
                CatalogReviewDecision::Exclude,
                Some("cropped feet".to_owned())
            )
            .unwrap()
            .is_some());
        assert_eq!(
            catalog.record_reviews(&["record-00".to_owned()]).unwrap()[0]
                .rejection_reason
                .as_deref(),
            Some("cropped feet")
        );
        assert!(catalog
            .set_record_review(
                "record-00",
                CatalogReviewDecision::Include,
                Some("ignored".into())
            )
            .unwrap()
            .unwrap()
            .rejection_reason
            .is_none());
        assert!(catalog
            .set_record_review("record-00", CatalogReviewDecision::Default, None)
            .unwrap()
            .is_none());

        let view = catalog
            .save_view(None, "Full body photos".to_owned(), base.clone(), None)
            .unwrap();
        assert!(view.query.cursor.is_none());
        let updated = catalog
            .save_view(
                Some(&view.id),
                "Reviewed photos".to_owned(),
                base,
                Some(view.revision),
            )
            .unwrap();
        assert!(matches!(
            catalog.save_view(
                Some(&view.id),
                "Stale edit".to_owned(),
                updated.query.clone(),
                Some(view.revision)
            ),
            Err(CatalogError::Conflict(_))
        ));
        assert!(matches!(
            catalog.save_view(
                None,
                "reviewed photos".to_owned(),
                updated.query.clone(),
                None,
            ),
            Err(CatalogError::Conflict(_))
        ));
        let other = catalog
            .save_view(None, "Other view".to_owned(), updated.query.clone(), None)
            .unwrap();
        assert!(matches!(
            catalog.save_view(
                Some(&other.id),
                "Reviewed photos".to_owned(),
                updated.query,
                Some(other.revision),
            ),
            Err(CatalogError::Conflict(_))
        ));
    }

    #[test]
    fn curation_deduplication_and_full_body_qualification_use_the_matched_set() {
        let temporary = tempdir().expect("temp directory");
        let mut catalog =
            Catalog::create(temporary.path().join("catalog"), "Scoped dedup").unwrap();
        let mut outside = curation_record(
            "record-a-outside",
            "Unrelated illustration",
            "sha256:shared",
            "dhash64:shared",
        );
        outside.metadata["analysis"]["medium"] = serde_json::json!("illustration");
        outside.metadata["acquisition"]["width"] = serde_json::json!(320);
        let matching = curation_record(
            "record-z-matching",
            "Target suffix",
            "sha256:shared",
            "dhash64:shared",
        );
        let mut visible_only = curation_record(
            "record-visible-only",
            "Visible body",
            "sha256:visible",
            "dhash64:visible",
        );
        visible_only.metadata["analysis"]["qualifiedSingleFullBody"] = serde_json::json!(false);
        catalog
            .append_records(&[outside, matching, visible_only])
            .unwrap();

        let scoped = catalog
            .search_records(&CatalogSearchRequest {
                limit: 10,
                text: "target".to_owned(),
                filters: vec![CatalogRecordFilter {
                    field: "medium".to_owned(),
                    values: vec!["photograph".to_owned()],
                }],
                min_width: Some(1000),
                deduplicate: true,
                ..CatalogSearchRequest::default()
            })
            .unwrap();
        assert_eq!(
            scoped
                .records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["record-z-matching"]
        );

        let full_body = catalog
            .search_records(&CatalogSearchRequest {
                limit: 10,
                filters: vec![CatalogRecordFilter {
                    field: "fullBody".to_owned(),
                    values: vec!["true".to_owned()],
                }],
                ..CatalogSearchRequest::default()
            })
            .unwrap();
        assert!(full_body
            .records
            .iter()
            .any(|record| record.id == "record-visible-only"));
        let qualified = catalog
            .search_records(&CatalogSearchRequest {
                limit: 10,
                filters: vec![CatalogRecordFilter {
                    field: "qualifiedSingleFullBody".to_owned(),
                    values: vec!["true".to_owned()],
                }],
                ..CatalogSearchRequest::default()
            })
            .unwrap();
        assert!(!qualified
            .records
            .iter()
            .any(|record| record.id == "record-visible-only"));
    }

    #[test]
    fn completed_curation_migration_reopens_without_taking_a_writer_lock() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        Catalog::create(&root, "Reopen").unwrap().close();
        let database_path = root.join(CATALOG_DATABASE_FILE);
        let writer = Connection::open(&database_path).unwrap();
        writer.execute_batch("begin immediate;").unwrap();
        let reopened = Catalog::open(&root).expect("completed migration is read-compatible");
        reopened.close();
        writer.execute_batch("rollback;").unwrap();
    }

    #[test]
    fn concurrent_curation_migrations_serialize_checkpointed_batches() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let mut catalog = Catalog::create(&root, "Concurrent migration").unwrap();
        let records = (0..1_005)
            .map(|index| {
                curation_record(
                    &format!("record-{index:04}"),
                    &format!("Concurrent record {index}"),
                    &format!("sha256:{index}"),
                    &format!("dhash64:{index}"),
                )
            })
            .collect::<Vec<_>>();
        catalog.append_records(&records).unwrap();
        catalog.close();
        let database_path = root.join(CATALOG_DATABASE_FILE);
        let raw = Connection::open(&database_path).unwrap();
        raw.execute(
            "delete from catalog_metadata where key in (?1, ?2)",
            params![
                CURATION_PROJECTION_CHECKPOINT_METADATA_KEY,
                CURATION_PROJECTION_COMPLETE_METADATA_KEY
            ],
        )
        .unwrap();
        raw.execute_batch(
            "delete from catalog_record_search;
             delete from catalog_record_fts;",
        )
        .unwrap();
        drop(raw);

        let barrier = Arc::new(std::sync::Barrier::new(4));
        let handles = (0..3)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let root = root.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    Catalog::open(root).expect("concurrent migration opens")
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        for handle in handles {
            handle.join().unwrap().close();
        }

        let raw = Connection::open(&database_path).unwrap();
        let projected: u64 = raw
            .query_row("select count(*) from catalog_record_search", [], |row| {
                row.get(0)
            })
            .unwrap();
        let searchable: u64 = raw
            .query_row("select count(*) from catalog_record_fts", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(projected, records.len() as u64);
        assert_eq!(searchable, records.len() as u64);
        assert_eq!(
            raw.query_row(
                "select value from catalog_metadata where key = ?1",
                [CURATION_PROJECTION_COMPLETE_METADATA_KEY],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
            "true"
        );
        assert!(
            raw.query_row(
                "select value from catalog_metadata where key = ?1",
                [CURATION_PROJECTION_CHECKPOINT_METADATA_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .unwrap()
            .is_none(),
            "completed migration removes its checkpoint"
        );
    }

    #[test]
    fn curation_migration_cannot_overwrite_a_racing_record_update_with_stale_projection() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let mut catalog = Catalog::create(&root, "Racing update").unwrap();
        let original = curation_record(
            "racing-record",
            "old_race_caption",
            "sha256:racing",
            "dhash64:racing",
        );
        catalog.append_records(&[original]).unwrap();
        catalog.close();
        let database_path = root.join(CATALOG_DATABASE_FILE);
        let writer = Connection::open(&database_path).unwrap();
        writer
            .execute(
                "delete from catalog_metadata where key in (?1, ?2)",
                params![
                    CURATION_PROJECTION_CHECKPOINT_METADATA_KEY,
                    CURATION_PROJECTION_COMPLETE_METADATA_KEY
                ],
            )
            .unwrap();
        writer
            .execute_batch(
                "delete from catalog_record_search;
                 delete from catalog_record_fts;
                 begin immediate;",
            )
            .unwrap();
        let mut updated = curation_record(
            "racing-record",
            "fresh_race_suffix",
            "sha256:racing",
            "dhash64:racing",
        );
        updated.metadata["analysis"]["tagMembership"] = serde_json::json!({"fresh_race_tag": true});
        writer
            .execute(
                "update catalog_records
                 set metadata_json = ?1, updated_at = ?2
                 where id = 'racing-record'",
                params![serde_json::to_string(&updated.metadata).unwrap(), utc_now()],
            )
            .unwrap();

        let (before_tx, before_rx) = std::sync::mpsc::channel();
        let (allow_tx, allow_rx) = std::sync::mpsc::channel();
        let (source_read_tx, source_read_rx) = std::sync::mpsc::channel();
        let opener_root = root.clone();
        let handle = std::thread::spawn(move || {
            set_curation_backfill_test_hooks(
                move || {
                    before_tx.send(()).unwrap();
                    allow_rx.recv().unwrap();
                },
                move || source_read_tx.send(()).unwrap(),
            );
            Catalog::open(opener_root).expect("migration waits for racing update")
        });
        before_rx.recv().unwrap();
        allow_tx.send(()).unwrap();
        assert!(
            matches!(
                source_read_rx.recv_timeout(Duration::from_millis(500)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ),
            "the source snapshot must wait until the active writer commits"
        );
        writer.execute_batch("commit;").unwrap();
        source_read_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("source snapshot proceeds after the writer commits");
        handle.join().unwrap().close();

        let reopened = Catalog::open(&root).unwrap();
        assert_eq!(
            reopened
                .search_records(&CatalogSearchRequest {
                    limit: 10,
                    text: "fresh_race_suffix fresh_race_tag".to_owned(),
                    ..CatalogSearchRequest::default()
                })
                .unwrap()
                .records
                .len(),
            1
        );
        assert!(reopened
            .search_records(&CatalogSearchRequest {
                limit: 10,
                text: "old_race_caption".to_owned(),
                ..CatalogSearchRequest::default()
            })
            .unwrap()
            .records
            .is_empty());
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
    fn scanner_records_checkpoint_and_progress_roll_back_together() {
        let temporary = tempdir().expect("temp directory");
        let root = temporary.path().join("catalog");
        let mut catalog = Catalog::create(&root, "Atomic scanner state").unwrap();
        let original_progress = catalog.contract_state().unwrap().processing;
        catalog
            .connection
            .execute_batch(
                "create trigger fail_scanner_progress
                 before insert on catalog_metadata
                 when new.key = 'processing_progress'
                 begin
                     select raise(abort, 'injected scanner progress failure');
                 end;",
            )
            .unwrap();
        let progress = CatalogProcessingProgress {
            state: CatalogProcessingState::Running,
            candidate_count: 1,
            processed_count: 1,
            accepted_count: 1,
            message: Some("scanner batch".to_owned()),
            updated_at: catalog_timestamp_now(),
            ..CatalogProcessingProgress::default()
        };

        assert!(matches!(
            catalog.append_records_and_metadata_with_progress(
                &[record("atomic-scanner-record")],
                &[("scanner.test.checkpoint", r#"{"row":1}"#)],
                &progress,
            ),
            Err(CatalogError::Sqlite(_))
        ));
        assert!(!catalog.contains_record_id("atomic-scanner-record").unwrap());
        assert_eq!(catalog.metadata("scanner.test.checkpoint").unwrap(), None);
        assert_eq!(
            catalog.contract_state().unwrap().processing,
            original_progress
        );
        assert_eq!(catalog.record_count().unwrap(), 0);
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
