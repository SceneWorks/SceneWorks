//! Restart-safe, bounded indexing of LAION-style Parquet metadata into an
//! isolated [`Catalog`].
//!
//! This scanner deliberately stores only metadata and source identities. It
//! does not download images or create a SceneWorks training dataset.

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use sceneworks_core::catalog_store::{
    catalog_timestamp_now, Catalog, CatalogError, CatalogProcessingDesiredState,
    CatalogProcessingLease, CatalogProcessingProgress, CatalogProcessingState, CatalogRegistry,
    NewCatalogRecord,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

const CHECKPOINT_KEY: &str = "scanner.parquet.checkpoint.v1";
const DEFAULT_BATCH_SIZE: usize = 1_000;
const MAX_BATCH_SIZE: usize = 10_000;
const MAX_CAPTION_CHARS: usize = 2_048;
const MAX_IMAGE_EDGE: u32 = 16_384;
const MAX_COLUMN_NAME_CHARS: usize = 256;
const MAX_FILTER_TERMS: usize = 100;
const MAX_FILTER_TERM_CHARS: usize = 256;
pub const MAX_PARQUET_SHARDS: usize = 4_096;
const MAX_PARQUET_DIRECTORY_ENTRIES: usize = 100_000;

#[derive(Debug)]
pub enum CatalogParquetScanError {
    Io(std::io::Error),
    Parquet(parquet::errors::ParquetError),
    Catalog(CatalogError),
    Json(serde_json::Error),
    InvalidSource(String),
    CheckpointMismatch(String),
    Busy(String),
    Interrupted(String),
}

impl std::fmt::Display for CatalogParquetScanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Parquet(error) => write!(formatter, "{error}"),
            Self::Catalog(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::InvalidSource(detail)
            | Self::CheckpointMismatch(detail)
            | Self::Busy(detail)
            | Self::Interrupted(detail) => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for CatalogParquetScanError {}

impl From<std::io::Error> for CatalogParquetScanError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<parquet::errors::ParquetError> for CatalogParquetScanError {
    fn from(error: parquet::errors::ParquetError) -> Self {
        Self::Parquet(error)
    }
}

impl From<CatalogError> for CatalogParquetScanError {
    fn from(error: CatalogError) -> Self {
        Self::Catalog(error)
    }
}

impl From<serde_json::Error> for CatalogParquetScanError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub type CatalogParquetScanResult<T> = Result<T, CatalogParquetScanError>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct CatalogParquetScanOptions {
    pub url_column: Option<String>,
    pub caption_column: Option<String>,
    pub min_width: Option<u32>,
    pub min_height: Option<u32>,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub caption_includes: Vec<String>,
    pub caption_excludes: Vec<String>,
    pub require_people: bool,
    pub max_unsafe_score: Option<f64>,
    pub min_aesthetic_score: Option<f64>,
    /// Transaction size. This does not change the semantic scan identity.
    pub batch_size: usize,
    /// Optional per-invocation row budget, used by schedulers to yield and
    /// resume long scans. This does not change the semantic scan identity.
    pub max_rows: Option<u64>,
}

impl Default for CatalogParquetScanOptions {
    fn default() -> Self {
        Self {
            url_column: None,
            caption_column: None,
            min_width: None,
            min_height: None,
            max_width: Some(MAX_IMAGE_EDGE),
            max_height: Some(MAX_IMAGE_EDGE),
            caption_includes: Vec::new(),
            caption_excludes: Vec::new(),
            require_people: false,
            max_unsafe_score: None,
            min_aesthetic_score: None,
            batch_size: DEFAULT_BATCH_SIZE,
            max_rows: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogParquetScanCounts {
    pub scanned: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub duplicate: u64,
    pub malformed: u64,
    pub reasons: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogParquetCheckpoint {
    pub source_signature: String,
    pub options_signature: String,
    /// Index of the next shard to read.
    pub shard: usize,
    /// Index of the next row group to read within `shard`.
    pub row_group: usize,
    /// Index of the next row to read within `row_group`.
    pub row: u64,
    pub complete: bool,
    pub counts: CatalogParquetScanCounts,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogParquetScanReport {
    pub checkpoint: CatalogParquetCheckpoint,
    pub source_shards: Vec<PathBuf>,
    pub paused: bool,
}

/// An attached scanner session that retains the shared processing lease across
/// every bounded pass. This prevents status reconciliation or lifecycle calls
/// from observing a lease-free gap while durable progress still says running.
pub struct AttachedCatalogParquetScanDriver {
    catalog: Catalog,
    _lease: CatalogProcessingLease,
}

impl AttachedCatalogParquetScanDriver {
    pub fn try_start(
        registry: &CatalogRegistry,
        catalog_id: &str,
    ) -> CatalogParquetScanResult<Self> {
        let catalog = registry.open_attached(catalog_id)?;
        let lease = CatalogProcessingLease::try_acquire(&catalog).map_err(|error| match error {
            CatalogError::Conflict(_) => CatalogParquetScanError::Busy(format!(
                "Catalog {} already has active processing.",
                catalog.descriptor().id
            )),
            error => CatalogParquetScanError::Catalog(error),
        })?;
        Ok(Self {
            catalog,
            _lease: lease,
        })
    }

    pub fn scan_pass(
        &mut self,
        source: &Path,
        options: &CatalogParquetScanOptions,
    ) -> CatalogParquetScanResult<CatalogParquetScanReport> {
        self.scan_pass_with_cancel(source, options, || false)
    }

    /// Runs one bounded pass and observes process cancellation at every durable
    /// batch boundary. The checkpoint is committed before interruption is
    /// published, so a later public restart resumes without replaying rows.
    pub fn scan_pass_with_cancel<F>(
        &mut self,
        source: &Path,
        options: &CatalogParquetScanOptions,
        should_cancel: F,
    ) -> CatalogParquetScanResult<CatalogParquetScanReport>
    where
        F: Fn() -> bool,
    {
        let result =
            scan_parquet_catalog_under_lease(&mut self.catalog, source, options, &should_cancel);
        if let Err(error) = &result {
            publish_scan_failure(&self.catalog, error);
        }
        result
    }

    /// Marks a between-pass cancellation durably before releasing the lease.
    pub fn publish_interrupted(&self) {
        publish_scan_failure(
            &self.catalog,
            &CatalogParquetScanError::Interrupted(
                "Catalog scan interrupted by server shutdown; restart it to continue.".to_owned(),
            ),
        );
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticOptions<'a> {
    url_column: &'a Option<String>,
    caption_column: &'a Option<String>,
    min_width: Option<u32>,
    min_height: Option<u32>,
    max_width: Option<u32>,
    max_height: Option<u32>,
    caption_includes: Vec<String>,
    caption_excludes: Vec<String>,
    require_people: bool,
    max_unsafe_score: Option<f64>,
    min_aesthetic_score: Option<f64>,
}

/// Indexes a bounded pass of Parquet rows into `catalog`.
///
/// Calling this again with the same source and semantic options resumes at the
/// exact next shard/row-group/row. A different source or filter configuration is
/// rejected rather than silently mixing two scans in one catalog.
pub fn scan_attached_parquet_catalog(
    registry: &CatalogRegistry,
    catalog_id: &str,
    source: &Path,
    options: &CatalogParquetScanOptions,
) -> CatalogParquetScanResult<CatalogParquetScanReport> {
    let mut driver = AttachedCatalogParquetScanDriver::try_start(registry, catalog_id)?;
    driver.scan_pass(source, options)
}

#[cfg(test)]
fn scan_parquet_catalog(
    catalog: &mut Catalog,
    source: &Path,
    options: &CatalogParquetScanOptions,
) -> CatalogParquetScanResult<CatalogParquetScanReport> {
    let _lease = CatalogProcessingLease::try_acquire(catalog).map_err(|error| match error {
        CatalogError::Conflict(_) => CatalogParquetScanError::Busy(format!(
            "Catalog {} already has active processing.",
            catalog.descriptor().id
        )),
        error => CatalogParquetScanError::Catalog(error),
    })?;
    let result = scan_parquet_catalog_under_lease(catalog, source, options, &|| false);
    if let Err(error) = &result {
        publish_scan_failure(catalog, error);
    }
    result
}

fn publish_scan_failure(catalog: &Catalog, error: &CatalogParquetScanError) {
    if let Ok(mut progress) = catalog.contract_state().map(|state| state.processing) {
        progress.state = CatalogProcessingState::Failed;
        progress.message = Some(match error {
            CatalogParquetScanError::Interrupted(detail) => detail.clone(),
            _ => "Parquet scan failed; inspect the worker logs for details".to_owned(),
        });
        progress.updated_at = catalog_timestamp_now();
        let _ = catalog.set_processing_progress(&progress);
    }
}

fn scan_parquet_catalog_under_lease(
    catalog: &mut Catalog,
    source: &Path,
    options: &CatalogParquetScanOptions,
    should_cancel: &dyn Fn() -> bool,
) -> CatalogParquetScanResult<CatalogParquetScanReport> {
    validate_options(options)?;
    let shards = parquet_shards_with_cancel(source, should_cancel)?;
    let source_signature = source_signature(&shards, should_cancel)?;
    let options_signature = options_signature(options)?;
    let mut checkpoint = load_checkpoint(catalog, &source_signature, &options_signature)?;
    if checkpoint.complete {
        publish_progress(
            catalog,
            &checkpoint,
            CatalogProcessingState::Completed,
            Some("Parquet scan complete"),
        )?;
        return Ok(CatalogParquetScanReport {
            checkpoint,
            source_shards: shards,
            paused: false,
        });
    }
    if catalog.processing_control()?.desired_state == CatalogProcessingDesiredState::Paused {
        publish_progress(
            catalog,
            &checkpoint,
            CatalogProcessingState::Paused,
            Some("Paused by user"),
        )?;
        return Ok(CatalogParquetScanReport {
            checkpoint,
            source_shards: shards,
            paused: true,
        });
    }
    publish_progress(
        catalog,
        &checkpoint,
        CatalogProcessingState::Running,
        Some("Scanning Parquet metadata"),
    )?;
    if should_cancel() {
        return Err(CatalogParquetScanError::Interrupted(
            "Catalog scan interrupted by server shutdown; restart it to continue.".to_owned(),
        ));
    }

    let batch_size = options.batch_size.clamp(1, MAX_BATCH_SIZE);
    let row_budget = options.max_rows.unwrap_or(u64::MAX);
    let mut rows_this_pass = 0_u64;
    let mut pending = Vec::with_capacity(batch_size);
    let mut pending_ids = HashSet::with_capacity(batch_size);

    for (shard_index, shard) in shards.iter().enumerate().skip(checkpoint.shard) {
        let reader = SerializedFileReader::new(File::open(shard)?)?;
        let row_group_start = if shard_index == checkpoint.shard {
            checkpoint.row_group
        } else {
            0
        };
        for row_group_index in row_group_start..reader.num_row_groups() {
            let row_start =
                if shard_index == checkpoint.shard && row_group_index == checkpoint.row_group {
                    checkpoint.row
                } else {
                    0
                };
            let row_group = reader.get_row_group(row_group_index)?;
            let rows = row_group.get_row_iter(None)?;
            for (row_index, row) in rows.enumerate().skip(row_start as usize) {
                if rows_this_pass >= row_budget {
                    let paused = flush_batch(catalog, &mut pending, &checkpoint)?;
                    if should_cancel() && !paused {
                        return Err(CatalogParquetScanError::Interrupted(
                            "Catalog scan interrupted by server shutdown; restart it to continue."
                                .to_owned(),
                        ));
                    }
                    if !paused {
                        publish_progress(
                            catalog,
                            &checkpoint,
                            CatalogProcessingState::Running,
                            Some("Parquet scan queued for its next bounded pass"),
                        )?;
                    }
                    return Ok(CatalogParquetScanReport {
                        checkpoint,
                        source_shards: shards,
                        paused,
                    });
                }
                rows_this_pass = rows_this_pass.saturating_add(1);
                checkpoint.counts.scanned = checkpoint.counts.scanned.saturating_add(1);
                checkpoint.shard = shard_index;
                checkpoint.row_group = row_group_index;
                checkpoint.row = row_index as u64 + 1;

                match row {
                    Ok(row) => match catalog_record_for_row(
                        &row,
                        shard,
                        shard_index,
                        row_group_index,
                        row_index as u64,
                        options,
                    ) {
                        RowOutcome::Accept(record) => {
                            if pending_ids.contains(&record.id)
                                || catalog.contains_record_id(&record.id)?
                            {
                                increment_duplicate(&mut checkpoint.counts);
                            } else {
                                pending_ids.insert(record.id.clone());
                                pending.push(record);
                                checkpoint.counts.accepted =
                                    checkpoint.counts.accepted.saturating_add(1);
                            }
                        }
                        RowOutcome::Reject(reason) => {
                            increment_rejected(&mut checkpoint.counts, reason)
                        }
                        RowOutcome::Malformed(reason) => {
                            increment_malformed(&mut checkpoint.counts, reason)
                        }
                    },
                    Err(_) => increment_malformed(&mut checkpoint.counts, "row_decode"),
                }

                if pending.len() >= batch_size || checkpoint.counts.scanned % batch_size as u64 == 0
                {
                    if flush_batch(catalog, &mut pending, &checkpoint)? {
                        return Ok(CatalogParquetScanReport {
                            checkpoint,
                            source_shards: shards,
                            paused: true,
                        });
                    }
                    if should_cancel() {
                        return Err(CatalogParquetScanError::Interrupted(
                            "Catalog scan interrupted by server shutdown; restart it to continue."
                                .to_owned(),
                        ));
                    }
                    pending_ids.clear();
                }
            }
            checkpoint.row_group = row_group_index + 1;
            checkpoint.row = 0;
            if flush_batch(catalog, &mut pending, &checkpoint)? {
                return Ok(CatalogParquetScanReport {
                    checkpoint,
                    source_shards: shards,
                    paused: true,
                });
            }
            if should_cancel() {
                return Err(CatalogParquetScanError::Interrupted(
                    "Catalog scan interrupted by server shutdown; restart it to continue."
                        .to_owned(),
                ));
            }
            pending_ids.clear();
        }
        checkpoint.shard = shard_index + 1;
        checkpoint.row_group = 0;
        checkpoint.row = 0;
        if flush_batch(catalog, &mut pending, &checkpoint)? {
            return Ok(CatalogParquetScanReport {
                checkpoint,
                source_shards: shards,
                paused: true,
            });
        }
        if should_cancel() {
            return Err(CatalogParquetScanError::Interrupted(
                "Catalog scan interrupted by server shutdown; restart it to continue.".to_owned(),
            ));
        }
        pending_ids.clear();
    }

    checkpoint.complete = true;
    flush_batch(catalog, &mut pending, &checkpoint)?;
    publish_progress(
        catalog,
        &checkpoint,
        CatalogProcessingState::Completed,
        Some("Parquet scan complete"),
    )?;
    Ok(CatalogParquetScanReport {
        checkpoint,
        source_shards: shards,
        paused: false,
    })
}

/// Whole-word/phrase matcher shared with the existing materializing importer.
/// Punctuation and case are ignored, but substrings inside larger words are not.
pub(crate) fn caption_matches_term(caption: &str, term: &str) -> bool {
    let caption_tokens = word_tokens(caption);
    let term_tokens = word_tokens(term);
    !term_tokens.is_empty()
        && caption_tokens
            .windows(term_tokens.len())
            .any(|window| window == term_tokens)
}

pub(crate) fn parquet_shards(source: &Path) -> CatalogParquetScanResult<Vec<PathBuf>> {
    parquet_shards_with_cancel(source, &|| false)
}

fn parquet_shards_with_cancel(
    source: &Path,
    should_cancel: &dyn Fn() -> bool,
) -> CatalogParquetScanResult<Vec<PathBuf>> {
    if !source.is_absolute() {
        return Err(CatalogParquetScanError::InvalidSource(
            "Parquet source path must be absolute.".to_owned(),
        ));
    }
    let canonical = std::fs::canonicalize(source)?;
    let metadata = std::fs::metadata(&canonical)?;
    let mut shards = if metadata.is_file() {
        if !has_parquet_extension(&canonical) {
            return Err(CatalogParquetScanError::InvalidSource(
                "Parquet source file must end in .parquet.".to_owned(),
            ));
        }
        vec![canonical]
    } else if metadata.is_dir() {
        let mut discovered = Vec::new();
        for (entry_index, entry) in std::fs::read_dir(&canonical)?.enumerate() {
            if should_cancel() {
                return Err(preflight_interrupted());
            }
            if entry_index >= MAX_PARQUET_DIRECTORY_ENTRIES {
                return Err(CatalogParquetScanError::InvalidSource(format!(
                    "Parquet source directory exceeds the {MAX_PARQUET_DIRECTORY_ENTRIES}-entry \
                     validation limit; split it into smaller source directories."
                )));
            }
            let path = entry?.path();
            if path.is_file() && has_parquet_extension(&path) {
                discovered.push(std::fs::canonicalize(path)?);
                if discovered.len() > MAX_PARQUET_SHARDS {
                    return Err(CatalogParquetScanError::InvalidSource(format!(
                        "Parquet source contains more than {MAX_PARQUET_SHARDS} shards; combine \
                         shards or split them across catalogs."
                    )));
                }
            }
        }
        discovered
    } else {
        Vec::new()
    };
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(CatalogParquetScanError::InvalidSource(
            "Parquet source contains no .parquet shards.".to_owned(),
        ));
    }
    Ok(shards)
}

/// Validates the complete source/options plan without mutating a catalog.
///
/// API callers use this before creating a catalog or advancing desired
/// processing state, so semantic errors are synchronous rather than background
/// failures after a successful lifecycle response.
pub fn validate_catalog_parquet_scan_plan(
    source: &Path,
    options: &CatalogParquetScanOptions,
) -> CatalogParquetScanResult<()> {
    validate_catalog_parquet_scan_plan_with_cancel(source, options, || false)
}

/// Cancellation-aware variant used by request preflight. Cancellation is
/// checked between directory entries and shard metadata reads so abandoned or
/// timed-out requests stop consuming the bounded validation pool promptly.
pub fn validate_catalog_parquet_scan_plan_with_cancel<F>(
    source: &Path,
    options: &CatalogParquetScanOptions,
    should_cancel: F,
) -> CatalogParquetScanResult<()>
where
    F: Fn() -> bool,
{
    validate_options(options)?;
    let shards = parquet_shards_with_cancel(source, &should_cancel)?;
    for shard in shards {
        if should_cancel() {
            return Err(preflight_interrupted());
        }
        let reader = SerializedFileReader::new(File::open(shard)?)?;
        let available_columns = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|column| column.name().to_ascii_lowercase())
            .collect::<HashSet<_>>();
        for (field, requested) in [
            ("urlColumn", options.url_column.as_deref()),
            ("captionColumn", options.caption_column.as_deref()),
        ] {
            if let Some(requested) = requested {
                if !available_columns.contains(&requested.trim().to_ascii_lowercase()) {
                    return Err(CatalogParquetScanError::InvalidSource(format!(
                        "{field} does not exist in every Parquet shard."
                    )));
                }
            }
        }
    }
    Ok(())
}

fn preflight_interrupted() -> CatalogParquetScanError {
    CatalogParquetScanError::Interrupted(
        "Parquet validation was canceled before it completed.".to_owned(),
    )
}

fn validate_options(options: &CatalogParquetScanOptions) -> CatalogParquetScanResult<()> {
    if options.batch_size == 0 || options.batch_size > MAX_BATCH_SIZE {
        return Err(CatalogParquetScanError::InvalidSource(format!(
            "Parquet catalog batch size must be between 1 and {MAX_BATCH_SIZE}."
        )));
    }
    if options.max_rows == Some(0) {
        return Err(CatalogParquetScanError::InvalidSource(
            "Parquet catalog row budget must be positive.".to_owned(),
        ));
    }
    for (field, column) in [
        ("urlColumn", options.url_column.as_deref()),
        ("captionColumn", options.caption_column.as_deref()),
    ] {
        if column.is_some_and(|column| {
            let column = column.trim();
            column.is_empty()
                || column.chars().count() > MAX_COLUMN_NAME_CHARS
                || column.chars().any(char::is_control)
        }) {
            return Err(CatalogParquetScanError::InvalidSource(format!(
                "{field} must be a non-empty column name of at most {MAX_COLUMN_NAME_CHARS} characters."
            )));
        }
    }
    for (field, terms) in [
        ("captionIncludes", &options.caption_includes),
        ("captionExcludes", &options.caption_excludes),
    ] {
        if terms.len() > MAX_FILTER_TERMS {
            return Err(CatalogParquetScanError::InvalidSource(format!(
                "{field} accepts at most {MAX_FILTER_TERMS} terms."
            )));
        }
        if terms.iter().any(|term| {
            let term = term.trim();
            term.is_empty() || term.chars().count() > MAX_FILTER_TERM_CHARS
        }) {
            return Err(CatalogParquetScanError::InvalidSource(format!(
                "{field} terms must contain 1 to {MAX_FILTER_TERM_CHARS} characters."
            )));
        }
    }
    for (name, value) in [
        ("maxUnsafeScore", options.max_unsafe_score),
        ("minAestheticScore", options.min_aesthetic_score),
    ] {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(CatalogParquetScanError::InvalidSource(format!(
                "{name} must be finite."
            )));
        }
    }
    for (name, minimum, maximum) in [
        ("width", options.min_width, options.max_width),
        ("height", options.min_height, options.max_height),
    ] {
        if [minimum, maximum]
            .into_iter()
            .flatten()
            .any(|value| value == 0 || value > MAX_IMAGE_EDGE)
        {
            return Err(CatalogParquetScanError::InvalidSource(format!(
                "Parquet catalog {name} bounds must be between 1 and {MAX_IMAGE_EDGE}."
            )));
        }
        if minimum
            .zip(maximum)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Err(CatalogParquetScanError::InvalidSource(format!(
                "Parquet catalog minimum {name} cannot exceed maximum {name}."
            )));
        }
    }
    if options
        .max_unsafe_score
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(CatalogParquetScanError::InvalidSource(
            "maxUnsafeScore must be between 0 and 1.".to_owned(),
        ));
    }
    Ok(())
}

fn load_checkpoint(
    catalog: &Catalog,
    source_signature: &str,
    options_signature: &str,
) -> CatalogParquetScanResult<CatalogParquetCheckpoint> {
    let Some(payload) = catalog.metadata(CHECKPOINT_KEY)? else {
        return Ok(CatalogParquetCheckpoint {
            source_signature: source_signature.to_owned(),
            options_signature: options_signature.to_owned(),
            shard: 0,
            row_group: 0,
            row: 0,
            complete: false,
            counts: CatalogParquetScanCounts::default(),
        });
    };
    let checkpoint: CatalogParquetCheckpoint = serde_json::from_str(&payload)?;
    if checkpoint.source_signature != source_signature {
        return Err(CatalogParquetScanError::CheckpointMismatch(
            "Parquet source changed after this catalog scan started.".to_owned(),
        ));
    }
    if checkpoint.options_signature != options_signature {
        return Err(CatalogParquetScanError::CheckpointMismatch(
            "Parquet scan filters changed after this catalog scan started.".to_owned(),
        ));
    }
    Ok(checkpoint)
}

fn flush_batch(
    catalog: &mut Catalog,
    pending: &mut Vec<NewCatalogRecord>,
    checkpoint: &CatalogParquetCheckpoint,
) -> CatalogParquetScanResult<bool> {
    let payload = serde_json::to_string(checkpoint)?;
    catalog.append_records_and_metadata(pending, &[(CHECKPOINT_KEY, payload.as_str())])?;
    pending.clear();
    let paused =
        catalog.processing_control()?.desired_state == CatalogProcessingDesiredState::Paused;
    publish_progress(
        catalog,
        checkpoint,
        if paused {
            CatalogProcessingState::Paused
        } else {
            CatalogProcessingState::Running
        },
        Some(if paused {
            "Paused by user"
        } else {
            "Scanning Parquet metadata"
        }),
    )?;
    Ok(paused)
}

fn publish_progress(
    catalog: &Catalog,
    checkpoint: &CatalogParquetCheckpoint,
    state: CatalogProcessingState,
    message: Option<&str>,
) -> CatalogParquetScanResult<()> {
    let previous = catalog.contract_state()?.processing;
    catalog.set_processing_progress(&CatalogProcessingProgress {
        state,
        candidate_count: previous.candidate_count.max(checkpoint.counts.scanned),
        processed_count: checkpoint.counts.scanned,
        accepted_count: checkpoint.counts.accepted,
        rejected_count: checkpoint
            .counts
            .rejected
            .saturating_add(checkpoint.counts.duplicate),
        error_count: checkpoint.counts.malformed,
        message: message.map(str::to_owned),
        updated_at: catalog_timestamp_now(),
    })?;
    Ok(())
}

enum RowOutcome {
    Accept(NewCatalogRecord),
    Reject(&'static str),
    Malformed(&'static str),
}

#[allow(clippy::too_many_arguments)]
fn catalog_record_for_row(
    row: &Row,
    shard: &Path,
    shard_index: usize,
    row_group: usize,
    row_index: u64,
    options: &CatalogParquetScanOptions,
) -> RowOutcome {
    let url_aliases = requested_aliases(
        options.url_column.as_deref(),
        &["url", "sample_url", "image_url"],
    );
    let caption_aliases = requested_aliases(
        options.caption_column.as_deref(),
        &["text", "caption", "description"],
    );
    let Some(url) = row_string_alias(row, &url_aliases) else {
        return RowOutcome::Malformed("missing_url");
    };
    let Some(caption) = row_string_alias(row, &caption_aliases) else {
        return RowOutcome::Malformed("missing_caption");
    };
    let url = url.trim().replace("&amp;", "&");
    let caption = caption.trim();
    if url.is_empty() {
        return RowOutcome::Malformed("blank_url");
    }
    if caption.is_empty() {
        return RowOutcome::Malformed("blank_caption");
    }
    if !valid_public_http_url(&url) {
        return RowOutcome::Reject("invalid_url");
    }

    let width = match filtered_dimension(
        row_u32_alias(row, &["width", "image_width", "original_width"]),
        options.min_width.is_some() || options.max_width.is_some(),
        "missing_width",
        "invalid_width",
    ) {
        Ok(value) => value,
        Err(reason) => return RowOutcome::Malformed(reason),
    };
    let height = match filtered_dimension(
        row_u32_alias(row, &["height", "image_height", "original_height"]),
        options.min_height.is_some() || options.max_height.is_some(),
        "missing_height",
        "invalid_height",
    ) {
        Ok(value) => value,
        Err(reason) => return RowOutcome::Malformed(reason),
    };
    if width.is_some_and(|value| {
        options.min_width.is_some_and(|minimum| value < minimum)
            || options.max_width.is_some_and(|maximum| value > maximum)
    }) || height.is_some_and(|value| {
        options.min_height.is_some_and(|minimum| value < minimum)
            || options.max_height.is_some_and(|maximum| value > maximum)
    }) {
        return RowOutcome::Reject("dimensions");
    }
    if !options.caption_includes.is_empty()
        && !options
            .caption_includes
            .iter()
            .any(|term| caption_matches_term(caption, term))
    {
        return RowOutcome::Reject("caption_include");
    }
    if options
        .caption_excludes
        .iter()
        .any(|term| caption_matches_term(caption, term))
    {
        return RowOutcome::Reject("caption_exclude");
    }
    if options.require_people && !caption_describes_people(caption) {
        return RowOutcome::Reject("people");
    }

    let unsafe_score = match filtered_score(
        row_probability_alias(
            row,
            &[
                "punsafe",
                "unsafe_score",
                "nsfw_score",
                "nsfw",
                "safety_score",
            ],
        ),
        options.max_unsafe_score.is_some(),
        "missing_safety",
        "invalid_safety",
    ) {
        Ok(value) => value,
        Err(reason) => return RowOutcome::Malformed(reason),
    };
    if options
        .max_unsafe_score
        .zip(unsafe_score)
        .is_some_and(|(maximum, score)| score > maximum)
    {
        return RowOutcome::Reject("safety");
    }
    let aesthetic_score = match filtered_score(
        row_f64_alias(
            row,
            &["aesthetic", "aesthetic_score", "aesthetic_score_laion_v2"],
        ),
        options.min_aesthetic_score.is_some(),
        "missing_aesthetic",
        "invalid_aesthetic",
    ) {
        Ok(value) => value,
        Err(reason) => return RowOutcome::Malformed(reason),
    };
    if options
        .min_aesthetic_score
        .zip(aesthetic_score)
        .is_some_and(|(minimum, score)| score < minimum)
    {
        return RowOutcome::Reject("aesthetic");
    }

    let digest = Sha256::digest(url.as_bytes());
    let id = format!("pq_{}", hex_prefix(&digest, 32));
    RowOutcome::Accept(NewCatalogRecord {
        image_path: format!("images/{id}"),
        id,
        thumbnail_path: None,
        embedding_path: None,
        artifact_path: None,
        metadata: json!({
            "url": url,
            "caption": caption.chars().take(MAX_CAPTION_CHARS).collect::<String>(),
            "width": width,
            "height": height,
            "unsafeScore": unsafe_score,
            "aestheticScore": aesthetic_score,
            "source": {
                "shard": shard,
                "shardIndex": shard_index,
                "rowGroup": row_group,
                "row": row_index,
            }
        }),
    })
}

fn increment_reason(counts: &mut CatalogParquetScanCounts, reason: &str) {
    let count = counts.reasons.entry(reason.to_owned()).or_default();
    *count = count.saturating_add(1);
}

fn increment_rejected(counts: &mut CatalogParquetScanCounts, reason: &'static str) {
    counts.rejected = counts.rejected.saturating_add(1);
    increment_reason(counts, reason);
}

fn increment_malformed(counts: &mut CatalogParquetScanCounts, reason: &'static str) {
    counts.malformed = counts.malformed.saturating_add(1);
    increment_reason(counts, reason);
}

fn increment_duplicate(counts: &mut CatalogParquetScanCounts) {
    counts.duplicate = counts.duplicate.saturating_add(1);
    increment_reason(counts, "duplicate_url");
}

fn requested_aliases<'a>(requested: Option<&'a str>, defaults: &'a [&'a str]) -> Vec<&'a str> {
    requested
        .filter(|value| !value.trim().is_empty())
        .map(|value| vec![value])
        .unwrap_or_else(|| defaults.to_vec())
}

fn find_field<'a>(row: &'a Row, aliases: &[&str]) -> Option<&'a Field> {
    row.get_column_iter()
        .find(|(name, _)| {
            aliases
                .iter()
                .any(|requested| name.eq_ignore_ascii_case(requested))
        })
        .map(|(_, field)| field)
}

fn row_string_alias(row: &Row, aliases: &[&str]) -> Option<String> {
    match find_field(row, aliases)? {
        Field::Str(value) => Some(value.clone()),
        Field::Bytes(value) => std::str::from_utf8(value.data()).ok().map(str::to_owned),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum MetadataField<T> {
    Missing,
    Invalid,
    Value(T),
}

fn row_u32_alias(row: &Row, aliases: &[&str]) -> MetadataField<u32> {
    let Some(field) = find_field(row, aliases) else {
        return MetadataField::Missing;
    };
    let value = match field {
        Field::Byte(value) => u32::try_from(*value).ok(),
        Field::Short(value) => u32::try_from(*value).ok(),
        Field::Int(value) => u32::try_from(*value).ok(),
        Field::Long(value) => u32::try_from(*value).ok(),
        Field::UByte(value) => Some((*value).into()),
        Field::UShort(value) => Some((*value).into()),
        Field::UInt(value) => Some(*value),
        Field::ULong(value) => u32::try_from(*value).ok(),
        Field::Float(value) => exact_u32((*value).into()),
        Field::Double(value) => exact_u32(*value),
        Field::Str(value) => value.trim().parse().ok(),
        Field::Bytes(value) => std::str::from_utf8(value.data())
            .ok()
            .and_then(|value| value.trim().parse().ok()),
        _ => None,
    };
    match value {
        Some(value) => MetadataField::Value(value),
        None => MetadataField::Invalid,
    }
}

fn row_f64_alias(row: &Row, aliases: &[&str]) -> MetadataField<f64> {
    let Some(field) = find_field(row, aliases) else {
        return MetadataField::Missing;
    };
    let value = match field {
        Field::Byte(value) => Some((*value).into()),
        Field::Short(value) => Some((*value).into()),
        Field::Int(value) => Some((*value).into()),
        Field::Long(value) => Some(*value as f64),
        Field::UByte(value) => Some((*value).into()),
        Field::UShort(value) => Some((*value).into()),
        Field::UInt(value) => Some((*value).into()),
        Field::ULong(value) => Some(*value as f64),
        Field::Float(value) if value.is_finite() => Some((*value).into()),
        Field::Double(value) if value.is_finite() => Some(*value),
        Field::Str(value) => parse_metadata_score(value),
        Field::Bytes(value) => std::str::from_utf8(value.data())
            .ok()
            .and_then(parse_metadata_score),
        _ => None,
    };
    match value {
        Some(value) if value.is_finite() => MetadataField::Value(value),
        _ => MetadataField::Invalid,
    }
}

fn row_probability_alias(row: &Row, aliases: &[&str]) -> MetadataField<f64> {
    match row_f64_alias(row, aliases) {
        MetadataField::Value(value) if (0.0..=1.0).contains(&value) => MetadataField::Value(value),
        MetadataField::Value(_) => MetadataField::Invalid,
        other => other,
    }
}

fn exact_u32(value: f64) -> Option<u32> {
    (value.is_finite() && value >= 0.0 && value <= u32::MAX as f64 && value.fract() == 0.0)
        .then_some(value as u32)
}

fn filtered_dimension(
    field: MetadataField<u32>,
    required: bool,
    missing_reason: &'static str,
    invalid_reason: &'static str,
) -> Result<Option<u32>, &'static str> {
    match field {
        MetadataField::Value(value) => Ok(Some(value)),
        MetadataField::Missing if required => Err(missing_reason),
        MetadataField::Invalid if required => Err(invalid_reason),
        MetadataField::Missing | MetadataField::Invalid => Ok(None),
    }
}

fn filtered_score(
    field: MetadataField<f64>,
    required: bool,
    missing_reason: &'static str,
    invalid_reason: &'static str,
) -> Result<Option<f64>, &'static str> {
    match field {
        MetadataField::Value(value) => Ok(Some(value)),
        MetadataField::Missing if required => Err(missing_reason),
        MetadataField::Invalid if required => Err(invalid_reason),
        MetadataField::Missing | MetadataField::Invalid => Ok(None),
    }
}

fn parse_metadata_score(value: &str) -> Option<f64> {
    let normalized = value.trim().to_ascii_lowercase();
    normalized.parse().ok().or(match normalized.as_str() {
        "safe" | "sfw" | "false" => Some(0.0),
        "unlikely" => Some(0.25),
        "unsure" => Some(0.5),
        "unsafe" | "nsfw" | "true" | "porn" => Some(1.0),
        _ => None,
    })
}

fn valid_public_http_url(value: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(value) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
        && url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none()
}

fn caption_describes_people(caption: &str) -> bool {
    const PEOPLE_TERMS: &[&str] = &[
        "person",
        "people",
        "human",
        "man",
        "men",
        "woman",
        "women",
        "boy",
        "boys",
        "girl",
        "girls",
        "child",
        "children",
        "lady",
        "gentleman",
        "portrait",
        "self portrait",
    ];
    PEOPLE_TERMS
        .iter()
        .any(|term| caption_matches_term(caption, term))
}

fn word_tokens(value: &str) -> Vec<String> {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn has_parquet_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("parquet"))
}

fn source_signature(
    shards: &[PathBuf],
    should_cancel: &dyn Fn() -> bool,
) -> CatalogParquetScanResult<String> {
    let mut digest = Sha256::new();
    for shard in shards {
        if should_cancel() {
            return Err(preflight_interrupted());
        }
        let metadata = std::fs::metadata(shard)?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
            .map(|value| value.as_nanos())
            .unwrap_or_default();
        digest.update(shard.to_string_lossy().as_bytes());
        digest.update([0]);
        digest.update(metadata.len().to_le_bytes());
        digest.update(modified.to_le_bytes());
    }
    Ok(hex_prefix(&digest.finalize(), 64))
}

fn options_signature(options: &CatalogParquetScanOptions) -> CatalogParquetScanResult<String> {
    let semantic = SemanticOptions {
        url_column: &options.url_column,
        caption_column: &options.caption_column,
        min_width: options.min_width,
        min_height: options.min_height,
        max_width: options.max_width,
        max_height: options.max_height,
        caption_includes: normalized_terms(&options.caption_includes),
        caption_excludes: normalized_terms(&options.caption_excludes),
        require_people: options.require_people,
        max_unsafe_score: options.max_unsafe_score,
        min_aesthetic_score: options.min_aesthetic_score,
    };
    let payload = serde_json::to_vec(&semantic)?;
    Ok(hex_prefix(&Sha256::digest(payload), 64))
}

fn normalized_terms(terms: &[String]) -> Vec<String> {
    let mut terms = terms
        .iter()
        .map(|term| word_tokens(term).join(" "))
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn hex_prefix(bytes: &[u8], chars: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(chars);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        if output.len() == chars {
            break;
        }
        output.push(HEX[(byte & 0x0f) as usize] as char);
        if output.len() == chars {
            break;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::basic::Compression;
    use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int32Type, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use std::sync::Arc;

    #[derive(Clone)]
    struct TestRow<'a> {
        url: &'a str,
        caption: &'a str,
        width: i32,
        height: i32,
        unsafe_score: f64,
        aesthetic: f64,
        marker: i64,
    }

    fn write_rows(path: &Path, row_groups: &[Vec<TestRow<'_>>]) {
        let schema = Arc::new(
            parse_message_type(
                "message laion {
                    REQUIRED BINARY UrL (UTF8);
                    REQUIRED BINARY TeXt (UTF8);
                    REQUIRED INT32 WiDtH;
                    REQUIRED INT32 HeIgHt;
                    REQUIRED DOUBLE PuNsAfE;
                    REQUIRED DOUBLE AeStHeTiC_ScOrE;
                    REQUIRED INT64 marker;
                }",
            )
            .expect("test schema"),
        );
        let properties = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build(),
        );
        let mut writer = SerializedFileWriter::new(
            File::create(path).expect("test parquet creates"),
            schema,
            properties,
        )
        .expect("writer creates");
        for rows in row_groups {
            let mut row_group = writer.next_row_group().expect("row group creates");
            let mut column_index = 0;
            while let Some(mut column) = row_group.next_column().expect("column advances") {
                match column_index {
                    0 | 1 => {
                        let values = rows
                            .iter()
                            .map(|row| {
                                if column_index == 0 {
                                    ByteArray::from(row.url)
                                } else {
                                    ByteArray::from(row.caption)
                                }
                            })
                            .collect::<Vec<_>>();
                        column
                            .typed::<ByteArrayType>()
                            .write_batch(&values, None, None)
                            .expect("strings write");
                    }
                    2 | 3 => {
                        let values = rows
                            .iter()
                            .map(|row| {
                                if column_index == 2 {
                                    row.width
                                } else {
                                    row.height
                                }
                            })
                            .collect::<Vec<_>>();
                        column
                            .typed::<Int32Type>()
                            .write_batch(&values, None, None)
                            .expect("ints write");
                    }
                    4 | 5 => {
                        let values = rows
                            .iter()
                            .map(|row| {
                                if column_index == 4 {
                                    row.unsafe_score
                                } else {
                                    row.aesthetic
                                }
                            })
                            .collect::<Vec<_>>();
                        column
                            .typed::<DoubleType>()
                            .write_batch(&values, None, None)
                            .expect("doubles write");
                    }
                    6 => {
                        let values = rows.iter().map(|row| row.marker).collect::<Vec<_>>();
                        column
                            .typed::<Int64Type>()
                            .write_batch(&values, None, None)
                            .expect("longs write");
                    }
                    other => panic!("unexpected column index: {other}"),
                }
                column.close().expect("column closes");
                column_index += 1;
            }
            row_group.close().expect("row group closes");
        }
        writer.close().expect("writer closes");
    }

    fn write_rows_without_scores(path: &Path, rows: &[TestRow<'_>], include_dimensions: bool) {
        let schema = Arc::new(
            parse_message_type(if include_dimensions {
                "message laion {
                    REQUIRED BINARY URL (UTF8);
                    REQUIRED BINARY TEXT (UTF8);
                    REQUIRED INT32 WIDTH;
                    REQUIRED INT32 HEIGHT;
                }"
            } else {
                "message laion {
                    REQUIRED BINARY URL (UTF8);
                    REQUIRED BINARY TEXT (UTF8);
                }"
            })
            .expect("test schema"),
        );
        let mut writer = SerializedFileWriter::new(
            File::create(path).expect("test parquet creates"),
            schema,
            Arc::new(WriterProperties::builder().build()),
        )
        .expect("writer creates");
        let mut row_group = writer.next_row_group().expect("row group creates");
        let mut column_index = 0;
        while let Some(mut column) = row_group.next_column().expect("column advances") {
            if column_index < 2 {
                let values = rows
                    .iter()
                    .map(|row| {
                        ByteArray::from(if column_index == 0 {
                            row.url
                        } else {
                            row.caption
                        })
                    })
                    .collect::<Vec<_>>();
                column
                    .typed::<ByteArrayType>()
                    .write_batch(&values, None, None)
                    .expect("strings write");
            } else {
                let values = rows
                    .iter()
                    .map(|row| {
                        if column_index == 2 {
                            row.width
                        } else {
                            row.height
                        }
                    })
                    .collect::<Vec<_>>();
                column
                    .typed::<Int32Type>()
                    .write_batch(&values, None, None)
                    .expect("dimensions write");
            }
            column.close().expect("column closes");
            column_index += 1;
        }
        row_group.close().expect("row group closes");
        writer.close().expect("writer closes");
    }

    fn row(url: &'static str, caption: &'static str) -> TestRow<'static> {
        TestRow {
            url,
            caption,
            width: 1024,
            height: 1024,
            unsafe_score: 0.01,
            aesthetic: 7.0,
            marker: 0,
        }
    }

    #[test]
    fn whole_word_people_matching_excludes_known_false_positives() {
        assert!(caption_describes_people("portrait of a woman"));
        assert!(caption_matches_term(
            "a young woman, outdoors",
            "young woman"
        ));
        assert!(!caption_matches_term("a watchman tower", "man"));
        assert!(!caption_describes_people("Watchman graphic novel cover"));
        assert!(!caption_describes_people(
            "BMW model photographed in a studio"
        ));
        assert!(!caption_describes_people("Tasmania wilderness panorama"));
    }

    #[test]
    fn catalog_scan_lease_is_exclusive_and_releases_for_same_or_changed_options() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let source = temporary.path().join("source.parquet");
        write_rows(
            &source,
            &[vec![row("https://example.com/one.jpg", "one person")]],
        );
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let first_catalog = registry
            .create_catalog(temporary.path().join("catalog"), "Lease")
            .expect("catalog creates");
        let catalog_id = first_catalog.descriptor().id.clone();
        let lease =
            CatalogProcessingLease::try_acquire(&first_catalog).expect("first lease acquires");
        first_catalog.close();

        for options in [
            CatalogParquetScanOptions::default(),
            CatalogParquetScanOptions {
                require_people: true,
                ..CatalogParquetScanOptions::default()
            },
        ] {
            let error = scan_attached_parquet_catalog(&registry, &catalog_id, &source, &options)
                .expect_err("concurrent scan must not reach checkpoint or writes");
            assert!(matches!(error, CatalogParquetScanError::Busy(_)));
        }
        let untouched = registry
            .open_attached(&catalog_id)
            .expect("catalog reopens while lease is held");
        assert_eq!(
            untouched.metadata(CHECKPOINT_KEY).expect("metadata reads"),
            None,
            "contended scans must not load or publish a checkpoint"
        );
        assert_eq!(
            untouched
                .storage_accounting()
                .expect("accounting")
                .record_count,
            0,
            "contended scans must not mix records"
        );
        untouched.close();

        drop(lease);
        let report = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions::default(),
        )
        .expect("lease releases on drop");
        assert!(report.checkpoint.complete);
        assert_eq!(report.checkpoint.counts.accepted, 1);
    }

    #[test]
    fn scanner_cooperatively_pauses_at_a_batch_boundary_and_resumes_without_claiming_early() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let source = temporary.path().join("source.parquet");
        write_rows(
            &source,
            &[vec![
                row("https://example.com/one.jpg", "one"),
                row("https://example.com/two.jpg", "two"),
            ]],
        );
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let catalog = registry
            .create_catalog(temporary.path().join("catalog"), "Controlled")
            .expect("catalog creates");
        let catalog_id = catalog.descriptor().id.clone();
        let lease = CatalogProcessingLease::try_acquire(&catalog).expect("processor lease");
        catalog
            .set_processing_progress(&CatalogProcessingProgress {
                state: CatalogProcessingState::Running,
                message: Some("scanner active".to_owned()),
                updated_at: catalog_timestamp_now(),
                ..CatalogProcessingProgress::default()
            })
            .expect("actual running publishes");
        let pause = catalog
            .request_processing_control(0, CatalogProcessingDesiredState::Paused)
            .expect("pause intent persists narrowly");
        assert_eq!(pause.revision, 1);
        assert_eq!(
            catalog.contract_state().unwrap().processing.state,
            CatalogProcessingState::Running,
            "requesting pause does not pretend the processor has acknowledged it"
        );
        drop(lease);
        catalog.close();

        let paused = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions {
                batch_size: 1,
                ..CatalogParquetScanOptions::default()
            },
        )
        .expect("scanner consumes pause");
        assert!(paused.paused);
        assert!(!paused.checkpoint.complete);
        let catalog = registry.open_attached(&catalog_id).unwrap();
        assert_eq!(
            catalog.contract_state().unwrap().processing.state,
            CatalogProcessingState::Paused
        );
        let resume = catalog
            .request_processing_control(1, CatalogProcessingDesiredState::Running)
            .expect("resume intent persists");
        assert_eq!(resume.revision, 2);
        assert_eq!(
            catalog.contract_state().unwrap().processing.state,
            CatalogProcessingState::Paused,
            "resume remains desired-only until a processor actually starts"
        );
        catalog.close();

        let resumed = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions {
                batch_size: 1,
                ..CatalogParquetScanOptions::default()
            },
        )
        .expect("scanner resumes");
        assert!(!resumed.paused);
        assert!(resumed.checkpoint.complete);
        assert_eq!(resumed.checkpoint.counts.accepted, 2);
        assert_eq!(
            registry
                .open_attached(&catalog_id)
                .unwrap()
                .contract_state()
                .unwrap()
                .processing
                .state,
            CatalogProcessingState::Completed
        );
    }

    #[test]
    fn resumes_at_exact_row_group_and_row_without_duplicate_records() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let source = temporary.path().join("source");
        std::fs::create_dir(&source).expect("source creates");
        write_rows(
            &source.join("part-00000.parquet"),
            &[
                vec![
                    row("https://example.com/1.jpg", "one person"),
                    row("https://example.com/2.jpg", "two people"),
                ],
                vec![
                    row("https://example.com/3.jpg", "three women"),
                    row("https://example.com/4.jpg", "four men"),
                ],
            ],
        );
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let catalog = registry
            .create_catalog(temporary.path().join("catalog"), "Resume")
            .expect("catalog creates");
        let catalog_id = catalog.descriptor().id.clone();
        catalog.close();
        let first = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions {
                max_rows: Some(3),
                batch_size: 2,
                ..CatalogParquetScanOptions::default()
            },
        )
        .expect("first bounded scan");
        assert!(!first.checkpoint.complete);
        assert_eq!(
            (
                first.checkpoint.shard,
                first.checkpoint.row_group,
                first.checkpoint.row
            ),
            (0, 1, 1)
        );
        assert_eq!(first.checkpoint.counts.accepted, 3);
        let yielded_catalog = registry.open_attached(&catalog_id).unwrap();
        assert_eq!(
            yielded_catalog.contract_state().unwrap().processing.state,
            CatalogProcessingState::Running,
            "incomplete bounded work remains durably recoverable as running"
        );
        yielded_catalog.close();

        let mismatch = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions {
                require_people: true,
                ..CatalogParquetScanOptions::default()
            },
        )
        .expect_err("changed filters cannot silently mix scans");
        assert!(matches!(
            mismatch,
            CatalogParquetScanError::CheckpointMismatch(_)
        ));

        let resumed = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions {
                max_rows: None,
                batch_size: 1,
                ..CatalogParquetScanOptions::default()
            },
        )
        .expect("scan resumes");
        assert!(resumed.checkpoint.complete);
        assert_eq!(resumed.checkpoint.counts.scanned, 4);
        assert_eq!(resumed.checkpoint.counts.accepted, 4);
        assert_eq!(resumed.checkpoint.counts.duplicate, 0);
        let catalog = registry
            .open_attached(&catalog_id)
            .expect("attached catalog opens");
        assert_eq!(catalog.storage_accounting().unwrap().record_count, 4);
        let page = catalog.page_records_after(None, 10).expect("records page");
        assert_eq!(page.records[2].metadata["source"]["rowGroup"], 1);
        assert_eq!(page.records[2].metadata["source"]["row"], 0);
    }

    #[test]
    fn case_insensitive_columns_filters_and_reason_counts_are_stable() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let source = temporary.path().join("filters.parquet");
        let mut invalid_dimensions = row("https://example.com/small.jpg", "a person");
        invalid_dimensions.width = 100;
        let mut unsafe_row = row("https://example.com/unsafe.jpg", "a person");
        unsafe_row.unsafe_score = 0.9;
        let mut ugly = row("https://example.com/ugly.jpg", "a person");
        ugly.aesthetic = 2.0;
        let mut negative_width = row("https://example.com/negative.jpg", "a person");
        negative_width.width = -1;
        let mut invalid_safety = row("https://example.com/nan-safety.jpg", "a person");
        invalid_safety.unsafe_score = f64::NAN;
        let mut invalid_aesthetic = row("https://example.com/nan-aesthetic.jpg", "a person");
        invalid_aesthetic.aesthetic = f64::NAN;
        write_rows(
            &source,
            &[vec![
                row("https://example.com/ok.jpg", "portrait of a woman"),
                row("https://example.com/ok.jpg", "duplicate woman"),
                row("ftp://example.com/no.jpg", "a woman"),
                invalid_dimensions,
                unsafe_row,
                ugly,
                row("https://example.com/watchman.jpg", "Watchman in Tasmania"),
                row(
                    "https://example.com/excluded.jpg",
                    "a person with watermark",
                ),
                row("", "a person"),
                row("https://example.com/blank.jpg", ""),
                negative_width,
                invalid_safety,
                invalid_aesthetic,
            ]],
        );
        let mut catalog =
            Catalog::create(temporary.path().join("catalog"), "Filters").expect("catalog creates");
        let report = scan_parquet_catalog(
            &mut catalog,
            &source,
            &CatalogParquetScanOptions {
                min_width: Some(512),
                min_height: Some(512),
                caption_includes: vec!["person".to_owned(), "woman".to_owned()],
                caption_excludes: vec!["watermark".to_owned()],
                require_people: true,
                max_unsafe_score: Some(0.5),
                min_aesthetic_score: Some(5.0),
                batch_size: 2,
                ..CatalogParquetScanOptions::default()
            },
        )
        .expect("scan succeeds");
        let counts = report.checkpoint.counts;
        assert_eq!(counts.scanned, 13);
        assert_eq!(counts.accepted, 1);
        assert_eq!(counts.duplicate, 1);
        assert_eq!(counts.rejected, 6);
        assert_eq!(counts.malformed, 5);
        assert_eq!(counts.reasons["invalid_url"], 1);
        assert_eq!(counts.reasons["dimensions"], 1);
        assert_eq!(counts.reasons["safety"], 1);
        assert_eq!(counts.reasons["aesthetic"], 1);
        assert_eq!(counts.reasons["caption_include"], 1);
        assert_eq!(counts.reasons["caption_exclude"], 1);
        assert_eq!(counts.reasons["blank_url"], 1);
        assert_eq!(counts.reasons["blank_caption"], 1);
        assert_eq!(counts.reasons["invalid_width"], 1);
        assert_eq!(counts.reasons["invalid_safety"], 1);
        assert_eq!(counts.reasons["invalid_aesthetic"], 1);
    }

    #[test]
    fn configured_filters_reject_missing_metadata_and_invalid_ranges() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let no_dimensions = temporary.path().join("no-dimensions.parquet");
        let no_scores = temporary.path().join("no-scores.parquet");
        let rows = [row("https://example.com/one.jpg", "one person")];
        write_rows_without_scores(&no_dimensions, &rows, false);
        write_rows_without_scores(&no_scores, &rows, true);

        let scan = |name: &str,
                    source: &Path,
                    options: CatalogParquetScanOptions|
         -> CatalogParquetScanCounts {
            let registry = CatalogRegistry::new(temporary.path().join(format!("{name}-state")));
            let catalog = registry
                .create_catalog(temporary.path().join(format!("{name}-catalog")), name)
                .expect("catalog creates");
            let catalog_id = catalog.descriptor().id.clone();
            catalog.close();
            scan_attached_parquet_catalog(&registry, &catalog_id, source, &options)
                .expect("scan reports malformed metadata")
                .checkpoint
                .counts
        };

        let missing_dimensions = scan(
            "missing-dimensions",
            &no_dimensions,
            CatalogParquetScanOptions::default(),
        );
        assert_eq!(missing_dimensions.malformed, 1);
        assert_eq!(missing_dimensions.reasons["missing_width"], 1);

        let missing_safety = scan(
            "missing-safety",
            &no_scores,
            CatalogParquetScanOptions {
                max_unsafe_score: Some(0.5),
                ..CatalogParquetScanOptions::default()
            },
        );
        assert_eq!(missing_safety.malformed, 1);
        assert_eq!(missing_safety.reasons["missing_safety"], 1);

        let missing_aesthetic = scan(
            "missing-aesthetic",
            &no_scores,
            CatalogParquetScanOptions {
                min_aesthetic_score: Some(5.0),
                ..CatalogParquetScanOptions::default()
            },
        );
        assert_eq!(missing_aesthetic.malformed, 1);
        assert_eq!(missing_aesthetic.reasons["missing_aesthetic"], 1);

        for options in [
            CatalogParquetScanOptions {
                min_width: Some(100),
                max_width: Some(99),
                ..CatalogParquetScanOptions::default()
            },
            CatalogParquetScanOptions {
                min_height: Some(100),
                max_height: Some(99),
                ..CatalogParquetScanOptions::default()
            },
            CatalogParquetScanOptions {
                max_unsafe_score: Some(1.1),
                ..CatalogParquetScanOptions::default()
            },
            CatalogParquetScanOptions {
                batch_size: 0,
                ..CatalogParquetScanOptions::default()
            },
            CatalogParquetScanOptions {
                max_rows: Some(0),
                ..CatalogParquetScanOptions::default()
            },
            CatalogParquetScanOptions {
                url_column: Some(" ".to_owned()),
                ..CatalogParquetScanOptions::default()
            },
            CatalogParquetScanOptions {
                caption_includes: vec![String::new()],
                ..CatalogParquetScanOptions::default()
            },
        ] {
            assert!(matches!(
                validate_options(&options),
                Err(CatalogParquetScanError::InvalidSource(_))
            ));
        }
    }

    #[test]
    fn checkpoint_crosses_shards_and_source_mutation_is_rejected() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let source = temporary.path().join("source");
        std::fs::create_dir(&source).expect("source creates");
        write_rows(
            &source.join("part-00000.parquet"),
            &[vec![row("https://example.com/one.jpg", "one person")]],
        );
        write_rows(
            &source.join("part-00001.parquet"),
            &[vec![row("https://example.com/two.jpg", "two people")]],
        );
        let registry = CatalogRegistry::new(temporary.path().join("state"));
        let catalog = registry
            .create_catalog(temporary.path().join("catalog"), "Shards")
            .expect("catalog creates");
        let catalog_id = catalog.descriptor().id.clone();
        catalog.close();
        let first = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions {
                max_rows: Some(1),
                ..CatalogParquetScanOptions::default()
            },
        )
        .expect("first shard scans");
        assert_eq!(
            (
                first.checkpoint.shard,
                first.checkpoint.row_group,
                first.checkpoint.row
            ),
            (1, 0, 0)
        );
        let resumed = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions::default(),
        )
        .expect("second shard resumes");
        assert!(resumed.checkpoint.complete);
        assert_eq!(resumed.checkpoint.counts.accepted, 2);
        let catalog = registry
            .open_attached(&catalog_id)
            .expect("catalog reopens");
        let page = catalog.page_records_after(None, 10).expect("records page");
        assert_eq!(page.records[1].metadata["source"]["shardIndex"], 1);
        catalog.close();

        write_rows(
            &source.join("part-00002.parquet"),
            &[vec![row("https://example.com/three.jpg", "three women")]],
        );
        let error = scan_attached_parquet_catalog(
            &registry,
            &catalog_id,
            &source,
            &CatalogParquetScanOptions::default(),
        )
        .expect_err("changed shard set cannot resume");
        assert!(matches!(
            error,
            CatalogParquetScanError::CheckpointMismatch(_)
        ));
    }

    #[test]
    fn validation_cancellation_stops_directory_enumeration() {
        let temporary = tempfile::tempdir().expect("temp directory");
        let source = temporary.path().join("source");
        std::fs::create_dir(&source).expect("source creates");
        for index in 0..20 {
            File::create(source.join(format!("{index:02}.parquet"))).expect("shard creates");
        }
        let checks = std::sync::atomic::AtomicUsize::new(0);
        let error = validate_catalog_parquet_scan_plan_with_cancel(
            &source,
            &CatalogParquetScanOptions::default(),
            || checks.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= 3,
        )
        .expect_err("cancellation stops validation");
        assert!(matches!(error, CatalogParquetScanError::Interrupted(_)));
        assert!(
            checks.load(std::sync::atomic::Ordering::SeqCst) <= 5,
            "validation must stop promptly after cancellation"
        );
    }
}
