//! Bounded, restart-safe acquisition of images referenced by an attached catalog.
//!
//! This is deliberately a catalog utility, not a SceneWorks project or training
//! dataset materializer. URL availability and cached-content identity live under
//! each record's `acquisition` metadata; analyzer output remains independent.

use crate::downloads::{fetch_public_source_url_bytes_with_options, PublicSourceUrlFetchOptions};
use crate::{WorkerError, WorkerResult};
use fs2::FileExt as _;
use futures_util::future::join_all;
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageFormat, ImageReader};
use sceneworks_core::catalog_store::{
    Catalog, CatalogError, CatalogRecord, CatalogRegistry, NewCatalogRecord,
};
use sceneworks_core::time::utc_now;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{Cursor, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const FETCH_CHECKPOINT_KEY: &str = "fetch.checkpoint.v1";
const FETCH_LOCK_FILE: &str = ".catalog-image-fetch.lock";
const COMPLETION_DIRECTORY: &str = "fetch-completions";
const TEMP_DIRECTORY: &str = ".fetch-tmp";
const MAX_CONCURRENCY: usize = 32;
const MAX_PAGE_SIZE: u32 = 1_000;
const MAX_IMAGE_BYTES: usize = 256 * 1024 * 1024;
const MAX_IN_FLIGHT_BYTES: usize = 1024 * 1024 * 1024;
const MAX_IMAGE_EDGE: u32 = 32_768;
const MAX_DECODED_PIXELS: u64 = 268_435_456;
const MAX_RETRIES: u32 = 8;

#[derive(Debug)]
pub enum CatalogImageFetchError {
    Catalog(CatalogError),
    Io(std::io::Error),
    Json(serde_json::Error),
    Image(image::ImageError),
    InvalidOptions(String),
    Busy(String),
}

impl std::fmt::Display for CatalogImageFetchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Catalog(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Image(error) => write!(formatter, "{error}"),
            Self::InvalidOptions(detail) | Self::Busy(detail) => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for CatalogImageFetchError {}

impl From<CatalogError> for CatalogImageFetchError {
    fn from(value: CatalogError) -> Self {
        Self::Catalog(value)
    }
}

impl From<std::io::Error> for CatalogImageFetchError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for CatalogImageFetchError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<image::ImageError> for CatalogImageFetchError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
}

pub type CatalogImageFetchResult<T> = Result<T, CatalogImageFetchError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CatalogImageFetchOptions {
    /// Stop once this many usable images exist in the catalog.
    pub accepted_target: u64,
    pub concurrency: usize,
    pub page_size: u32,
    pub max_file_bytes: usize,
    /// Pessimistic encoded-byte budget shared by all active requests.
    pub max_in_flight_bytes: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
    pub connect_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub hop_timeout_ms: u64,
    pub max_retries: u32,
    pub retry_backoff_ms: u64,
    /// Maximum artifact bytes in this catalog. Existing SQLite/manifest bytes
    /// do not count toward this cache policy.
    pub max_cache_bytes: u64,
    pub discard_rejected_originals: bool,
    /// `None` disables thumbnails.
    pub thumbnail_max_edge: Option<u32>,
    pub max_thumbnail_bytes: usize,
}

impl Default for CatalogImageFetchOptions {
    fn default() -> Self {
        Self {
            accepted_target: 1_000,
            concurrency: 8,
            page_size: 128,
            max_file_bytes: 32 * 1024 * 1024,
            max_in_flight_bytes: 256 * 1024 * 1024,
            max_width: 16_384,
            max_height: 16_384,
            max_pixels: 100_000_000,
            connect_timeout_ms: 10_000,
            read_timeout_ms: 30_000,
            hop_timeout_ms: 60_000,
            max_retries: 2,
            retry_backoff_ms: 250,
            max_cache_bytes: 100 * 1024 * 1024 * 1024,
            discard_rejected_originals: true,
            thumbnail_max_edge: Some(256),
            max_thumbnail_bytes: 512 * 1024,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogImageFetchCounts {
    pub candidates: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub failures: u64,
    pub retries: u64,
    pub cache_hits: u64,
    pub duplicate_content: u64,
    pub tampered: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogImageFetchCheckpoint {
    pub requested_accepted_target: u64,
    pub exhausted: bool,
    pub counts: CatalogImageFetchCounts,
    pub updated_at: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogImageFetchReport {
    pub checkpoint: CatalogImageFetchCheckpoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcquisitionState {
    status: String,
    availability: String,
    content_hash: Option<String>,
    cache_path: Option<String>,
    thumbnail_path: Option<String>,
    bytes: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    attempts: u64,
    retries: u64,
    failure_reason: Option<String>,
    updated_at: String,
}

impl Default for AcquisitionState {
    fn default() -> Self {
        Self {
            status: "pending".to_owned(),
            availability: "unknown".to_owned(),
            content_hash: None,
            cache_path: None,
            thumbnail_path: None,
            bytes: None,
            width: None,
            height: None,
            attempts: 0,
            retries: 0,
            failure_reason: None,
            updated_at: utc_now(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompletionMarker {
    record_id: String,
    content_hash: String,
    cache_path: String,
    thumbnail_path: Option<String>,
    bytes: u64,
    width: u32,
    height: u32,
}

#[derive(Debug)]
struct CatalogFetchLease {
    _file: File,
}

impl CatalogFetchLease {
    fn acquire(catalog: &Catalog) -> CatalogImageFetchResult<Self> {
        let path = catalog.root().join(FETCH_LOCK_FILE);
        if let Ok(metadata) = std::fs::symlink_metadata(&path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(CatalogImageFetchError::InvalidOptions(format!(
                    "Catalog fetch lock must be a regular file: {}",
                    path.display()
                )));
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        let contended = fs2::lock_contended_error().raw_os_error();
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Self { _file: file }),
            Err(error) if error.raw_os_error() == contended => {
                Err(CatalogImageFetchError::Busy(format!(
                    "Catalog {} already has an active image fetch.",
                    catalog.descriptor().id
                )))
            }
            Err(error) => Err(error.into()),
        }
    }
}

/// Fetches image candidates for one attached catalog.
///
/// The catalog ID is resolved exclusively through `CatalogRegistry`; callers
/// cannot supply an arbitrary cache root. A catalog-wide advisory lease prevents
/// two worker processes from mutating acquisition state concurrently.
pub async fn fetch_attached_catalog_images(
    registry: &CatalogRegistry,
    catalog_id: &str,
    options: &CatalogImageFetchOptions,
) -> CatalogImageFetchResult<CatalogImageFetchReport> {
    fetch_attached_catalog_images_with(
        registry,
        catalog_id,
        options,
        |url, fetch_options| async move {
            fetch_public_source_url_bytes_with_options(&url, &fetch_options).await
        },
    )
    .await
}

async fn fetch_attached_catalog_images_with<F, Fut>(
    registry: &CatalogRegistry,
    catalog_id: &str,
    options: &CatalogImageFetchOptions,
    fetcher: F,
) -> CatalogImageFetchResult<CatalogImageFetchReport>
where
    F: Fn(String, PublicSourceUrlFetchOptions) -> Fut + Clone,
    Fut: Future<Output = WorkerResult<Vec<u8>>>,
{
    validate_options(options)?;
    let mut catalog = registry.open_attached(catalog_id)?;
    let _lease = CatalogFetchLease::acquire(&catalog)?;
    prepare_cache_directories(&catalog)?;
    cleanup_stale_temps(&catalog)?;

    let mut checkpoint = CatalogImageFetchCheckpoint {
        requested_accepted_target: options.accepted_target,
        exhausted: true,
        counts: CatalogImageFetchCounts::default(),
        updated_at: utc_now(),
    };
    let semaphore = Arc::new(Semaphore::new(options.max_in_flight_bytes));
    let mut cursor = None;
    let mut target_reached = false;

    loop {
        let page = catalog.page_records_after(cursor, options.page_size)?;
        if page.records.is_empty() {
            break;
        }
        cursor = page.next_cursor;
        let mut candidate_index = 0;
        while candidate_index < page.records.len() {
            let mut pending = Vec::new();
            while candidate_index < page.records.len() && pending.len() < options.concurrency {
                let mut record = page.records[candidate_index].clone();
                candidate_index += 1;
                checkpoint.counts.candidates = checkpoint.counts.candidates.saturating_add(1);

                if target_reached {
                    // Fetch-target stopping never skips retention reconciliation.
                    // Rejected records need no network or decode work, even when
                    // their cached file was concurrently removed as the last
                    // shared reference. Accepted/pending candidates remain for a
                    // later request once the requested target has been met.
                    if !record_is_accepted(&record) {
                        if let Some(marker) = recover_completion_marker(&catalog, &record)? {
                            let prior = acquisition_state(&record);
                            apply_available(
                                &mut catalog,
                                &mut record,
                                &marker,
                                prior.attempts,
                                prior.retries,
                                &mut checkpoint,
                            )?;
                        }
                        checkpoint.counts.rejected = checkpoint.counts.rejected.saturating_add(1);
                        if options.discard_rejected_originals {
                            discard_rejected_original(&mut catalog, &mut record, &mut checkpoint)?;
                        }
                    } else if matches!(
                        acquisition_state(&record).status.as_str(),
                        "available" | "rejected"
                    ) {
                        remove_completion_marker(&catalog, &record.id)?;
                    }
                    continue;
                }

                if let Some(marker) = recover_completion_marker(&catalog, &record)? {
                    let prior = acquisition_state(&record);
                    apply_available(
                        &mut catalog,
                        &mut record,
                        &marker,
                        prior.attempts,
                        prior.retries,
                        &mut checkpoint,
                    )?;
                    account_completed_record(&mut catalog, &mut record, options, &mut checkpoint)?;
                    if checkpoint.counts.accepted >= options.accepted_target {
                        checkpoint.exhausted = false;
                        target_reached = true;
                    }
                    continue;
                }

                match validate_completed_record(&catalog, &record)? {
                    CompletedValidation::Valid => {
                        checkpoint.counts.cache_hits =
                            checkpoint.counts.cache_hits.saturating_add(1);
                        account_completed_record(
                            &mut catalog,
                            &mut record,
                            options,
                            &mut checkpoint,
                        )?;
                        if checkpoint.counts.accepted >= options.accepted_target {
                            checkpoint.exhausted = false;
                            target_reached = true;
                        }
                    }
                    CompletedValidation::Tampered => {
                        checkpoint.counts.tampered = checkpoint.counts.tampered.saturating_add(1);
                        pending.push(record);
                    }
                    CompletedValidation::Incomplete => pending.push(record),
                }
            }

            if pending.is_empty() {
                continue;
            }
            let futures = pending.into_iter().map(|record| {
                fetch_candidate(
                    record,
                    options.clone(),
                    Arc::clone(&semaphore),
                    fetcher.clone(),
                )
            });
            for outcome in join_all(futures).await {
                let mut record = outcome.record;
                let prior = acquisition_state(&record);
                checkpoint.counts.retries = checkpoint
                    .counts
                    .retries
                    .saturating_add(u64::from(outcome.attempts.saturating_sub(1)));
                // Decode one completed response at a time. The concurrent
                // download futures retain encoded-byte permits, while decoded
                // raster memory is therefore bounded by one maxPixels image.
                let result = outcome
                    .result
                    .and_then(|download| validate_image(download.bytes, options, download.permit));
                match result {
                    Ok(fetched) => {
                        let marker = persist_fetched(
                            &catalog,
                            &record,
                            &fetched,
                            options,
                            &mut checkpoint.counts,
                        )?;
                        apply_available(
                            &mut catalog,
                            &mut record,
                            &marker,
                            prior.attempts.saturating_add(u64::from(outcome.attempts)),
                            prior
                                .retries
                                .saturating_add(u64::from(outcome.attempts.saturating_sub(1))),
                            &mut checkpoint,
                        )?;
                        account_completed_record(
                            &mut catalog,
                            &mut record,
                            options,
                            &mut checkpoint,
                        )?;
                    }
                    Err(error) => {
                        checkpoint.counts.failures = checkpoint.counts.failures.saturating_add(1);
                        apply_failure(
                            &mut catalog,
                            &mut record,
                            prior.attempts.saturating_add(u64::from(outcome.attempts)),
                            prior
                                .retries
                                .saturating_add(u64::from(outcome.attempts.saturating_sub(1))),
                            &bounded_failure_reason(&error),
                            &mut checkpoint,
                        )?;
                    }
                }
                if checkpoint.counts.accepted >= options.accepted_target {
                    checkpoint.exhausted = false;
                    target_reached = true;
                }
            }
        }
        if page.next_cursor.is_none() {
            break;
        }
    }

    checkpoint.updated_at = utc_now();
    persist_checkpoint(&mut catalog, &checkpoint)?;
    Ok(CatalogImageFetchReport { checkpoint })
}

struct FetchOutcome {
    record: CatalogRecord,
    attempts: u32,
    result: WorkerResult<FetchedDownload>,
}

struct FetchedDownload {
    bytes: Vec<u8>,
    permit: OwnedSemaphorePermit,
}

#[derive(Debug)]
struct FetchedImage {
    bytes: Vec<u8>,
    format: ImageFormat,
    image: DynamicImage,
    width: u32,
    height: u32,
    _permit: OwnedSemaphorePermit,
}

async fn fetch_candidate<F, Fut>(
    record: CatalogRecord,
    options: CatalogImageFetchOptions,
    semaphore: Arc<Semaphore>,
    fetcher: F,
) -> FetchOutcome
where
    F: Fn(String, PublicSourceUrlFetchOptions) -> Fut,
    Fut: Future<Output = WorkerResult<Vec<u8>>>,
{
    let Some(url) = record
        .metadata
        .get("url")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return FetchOutcome {
            record,
            attempts: 1,
            result: Err(WorkerError::InvalidPayload(
                "catalog record has no image URL".to_owned(),
            )),
        };
    };
    let permit_count =
        u32::try_from(options.max_file_bytes).expect("validated max file byte limit fits in u32");
    let permit = match semaphore.acquire_many_owned(permit_count).await {
        Ok(permit) => permit,
        Err(_) => {
            return FetchOutcome {
                record,
                attempts: 1,
                result: Err(WorkerError::Canceled(
                    "catalog image memory budget closed".to_owned(),
                )),
            };
        }
    };
    let fetch_options = PublicSourceUrlFetchOptions {
        max_bytes: options.max_file_bytes,
        connect_timeout: Duration::from_millis(options.connect_timeout_ms),
        read_timeout: Duration::from_millis(options.read_timeout_ms),
        hop_timeout: Duration::from_millis(options.hop_timeout_ms),
    };
    let mut attempts = 0;
    loop {
        attempts += 1;
        match fetcher(url.clone(), fetch_options).await {
            Ok(bytes) => {
                return FetchOutcome {
                    record,
                    attempts,
                    result: Ok(FetchedDownload { bytes, permit }),
                };
            }
            Err(_error) if attempts <= options.max_retries => {
                let multiplier = 1_u64 << attempts.saturating_sub(1).min(8);
                let delay = options
                    .retry_backoff_ms
                    .saturating_mul(multiplier)
                    .min(30_000);
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            Err(error) => {
                return FetchOutcome {
                    record,
                    attempts,
                    result: Err(error),
                };
            }
        }
    }
}

fn validate_image(
    bytes: Vec<u8>,
    options: &CatalogImageFetchOptions,
    permit: OwnedSemaphorePermit,
) -> WorkerResult<FetchedImage> {
    let reader = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|error| WorkerError::InvalidPayload(format!("image format failed: {error}")))?;
    let format = reader
        .format()
        .ok_or_else(|| WorkerError::InvalidPayload("image format is unknown".to_owned()))?;
    if !matches!(
        format,
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP
    ) {
        return Err(WorkerError::InvalidPayload(
            "image format must be PNG, JPEG, or WebP".to_owned(),
        ));
    }
    let (width, height) = reader.into_dimensions().map_err(|error| {
        WorkerError::InvalidPayload(format!("image dimensions failed: {error}"))
    })?;
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if width == 0
        || height == 0
        || width > options.max_width
        || height > options.max_height
        || pixels > options.max_pixels
    {
        return Err(WorkerError::InvalidPayload(format!(
            "image dimensions {width}x{height} exceed configured limits"
        )));
    }
    let image = image::load_from_memory_with_format(&bytes, format)
        .map_err(|error| WorkerError::InvalidPayload(format!("image decode failed: {error}")))?;
    Ok(FetchedImage {
        bytes,
        format,
        image,
        width,
        height,
        _permit: permit,
    })
}

fn persist_fetched(
    catalog: &Catalog,
    record: &CatalogRecord,
    fetched: &FetchedImage,
    options: &CatalogImageFetchOptions,
    counts: &mut CatalogImageFetchCounts,
) -> CatalogImageFetchResult<CompletionMarker> {
    let hash = hex_digest(&fetched.bytes);
    let extension = match fetched.format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
        ImageFormat::WebP => "webp",
        _ => unreachable!("validated image format"),
    };
    let cache_path = format!("images/sha256/{}/{}.{}", &hash[..2], hash, extension);
    let destination = safe_catalog_file(catalog, &cache_path, true)?;
    let already_present = destination.exists();
    if already_present {
        if validate_cached_file(&destination, &hash, fetched.bytes.len() as u64).is_ok() {
            counts.duplicate_content = counts.duplicate_content.saturating_add(1);
        } else {
            // The path is content-addressed, so a mismatch is tampering or an
            // interrupted external write. The catalog lease makes replacement
            // exclusive and the new bytes are independently hash-verified.
            std::fs::remove_file(&destination)?;
            atomic_create(&destination, &fetched.bytes)?;
        }
    } else {
        let projected = catalog
            .storage_accounting()?
            .artifact_bytes
            .saturating_add(fetched.bytes.len() as u64);
        if projected > options.max_cache_bytes {
            return Err(CatalogImageFetchError::InvalidOptions(format!(
                "catalog image cache would exceed its {} byte limit",
                options.max_cache_bytes
            )));
        }
        atomic_create(&destination, &fetched.bytes)?;
    }

    // Persist the full-size completion before optional thumbnail work. A crash
    // or thumbnail encoder failure can then resume from the verified original
    // without downloading it again.
    let base_marker = CompletionMarker {
        record_id: record.id.clone(),
        content_hash: hash.clone(),
        cache_path: cache_path.clone(),
        thumbnail_path: None,
        bytes: fetched.bytes.len() as u64,
        width: fetched.width,
        height: fetched.height,
    };
    persist_completion_marker(catalog, &base_marker, options.max_cache_bytes)?;

    let thumbnail_path = if let Some(max_edge) = options.thumbnail_max_edge {
        let relative = format!("thumbnails/sha256/{}/{}.jpg", &hash[..2], hash);
        let path = safe_catalog_file(catalog, &relative, true)?;
        if !path.exists() {
            let thumbnail = fetched.image.thumbnail(max_edge, max_edge).to_rgb8();
            let mut encoded = Vec::new();
            JpegEncoder::new_with_quality(&mut encoded, 82).encode_image(&thumbnail)?;
            let projected = catalog
                .storage_accounting()?
                .artifact_bytes
                .saturating_add(encoded.len() as u64);
            if encoded.len() > options.max_thumbnail_bytes || projected > options.max_cache_bytes {
                return Ok(base_marker);
            }
            atomic_create(&path, &encoded)?;
        }
        Some(relative)
    } else {
        None
    };
    let marker = CompletionMarker {
        record_id: record.id.clone(),
        content_hash: hash,
        cache_path,
        thumbnail_path,
        bytes: fetched.bytes.len() as u64,
        width: fetched.width,
        height: fetched.height,
    };
    persist_completion_marker(catalog, &marker, options.max_cache_bytes)?;
    Ok(marker)
}

fn apply_available(
    catalog: &mut Catalog,
    record: &mut CatalogRecord,
    marker: &CompletionMarker,
    attempts: u64,
    retries: u64,
    checkpoint: &mut CatalogImageFetchCheckpoint,
) -> CatalogImageFetchResult<()> {
    let state = AcquisitionState {
        status: if record_is_accepted(record) {
            "available".to_owned()
        } else {
            "rejected".to_owned()
        },
        availability: "available".to_owned(),
        content_hash: Some(marker.content_hash.clone()),
        cache_path: Some(marker.cache_path.clone()),
        thumbnail_path: marker.thumbnail_path.clone(),
        bytes: Some(marker.bytes),
        width: Some(marker.width),
        height: Some(marker.height),
        attempts,
        retries,
        failure_reason: None,
        updated_at: utc_now(),
    };
    set_acquisition_state(record, &state)?;
    record.image_path = marker.cache_path.clone();
    record.thumbnail_path.clone_from(&marker.thumbnail_path);
    persist_record_and_checkpoint(catalog, record, checkpoint)?;
    remove_completion_marker(catalog, &record.id)
}

fn apply_failure(
    catalog: &mut Catalog,
    record: &mut CatalogRecord,
    attempts: u64,
    retries: u64,
    reason: &str,
    checkpoint: &mut CatalogImageFetchCheckpoint,
) -> CatalogImageFetchResult<()> {
    let mut state = acquisition_state(record);
    state.status = "failed".to_owned();
    state.availability = "unavailable".to_owned();
    state.attempts = attempts;
    state.retries = retries;
    state.failure_reason = Some(reason.to_owned());
    state.updated_at = utc_now();
    set_acquisition_state(record, &state)?;
    persist_record_and_checkpoint(catalog, record, checkpoint)
}

fn account_completed_record(
    catalog: &mut Catalog,
    record: &mut CatalogRecord,
    options: &CatalogImageFetchOptions,
    checkpoint: &mut CatalogImageFetchCheckpoint,
) -> CatalogImageFetchResult<()> {
    if record_is_accepted(record) {
        checkpoint.counts.accepted = checkpoint.counts.accepted.saturating_add(1);
    } else {
        checkpoint.counts.rejected = checkpoint.counts.rejected.saturating_add(1);
        if options.discard_rejected_originals {
            discard_rejected_original(catalog, record, checkpoint)?;
        }
    }
    remove_completion_marker(catalog, &record.id)
}

fn discard_rejected_original(
    catalog: &mut Catalog,
    record: &mut CatalogRecord,
    checkpoint: &mut CatalogImageFetchCheckpoint,
) -> CatalogImageFetchResult<()> {
    let mut state = acquisition_state(record);
    if state.availability == "discarded" {
        state.status = "rejected".to_owned();
        state.updated_at = utc_now();
        set_acquisition_state(record, &state)?;
        return persist_record_and_checkpoint(catalog, record, checkpoint);
    }
    if let Some(path) = state.cache_path.as_deref() {
        let references = cache_path_references(catalog, &record.id, path)?;
        if !references.has_live_reference && !references.has_pending_reference {
            let file = safe_catalog_file(catalog, path, false)?;
            if file.exists() {
                std::fs::remove_file(file)?;
            }
            for mut other in references.rejected_references {
                let mut other_state = acquisition_state(&other);
                other_state.status = "rejected".to_owned();
                other_state.availability = "discarded".to_owned();
                other_state.updated_at = utc_now();
                set_acquisition_state(&mut other, &other_state)?;
                other.image_path = format!("images/{}", other.id);
                persist_record_and_checkpoint(catalog, &other, checkpoint)?;
            }
            state.availability = "discarded".to_owned();
            record.image_path = format!("images/{}", record.id);
        } else {
            // Content-addressed files may be shared by duplicate URLs/content.
            // Never delete another record's backing file.
            state.availability = "shared".to_owned();
        }
    }
    state.status = "rejected".to_owned();
    state.updated_at = utc_now();
    set_acquisition_state(record, &state)?;
    persist_record_and_checkpoint(catalog, record, checkpoint)
}

struct CachePathReferences {
    has_live_reference: bool,
    has_pending_reference: bool,
    rejected_references: Vec<CatalogRecord>,
}

fn cache_path_references(
    catalog: &Catalog,
    record_id: &str,
    cache_path: &str,
) -> CatalogImageFetchResult<CachePathReferences> {
    let mut references = CachePathReferences {
        has_live_reference: false,
        has_pending_reference: false,
        rejected_references: Vec::new(),
    };
    let mut cursor = None;
    loop {
        let page = catalog.page_records_after(cursor, MAX_PAGE_SIZE)?;
        for other in page.records {
            if other.id == record_id {
                continue;
            }
            let state = acquisition_state(&other);
            if state.cache_path.as_deref() != Some(cache_path) || state.availability == "discarded"
            {
                continue;
            }
            if record_is_accepted(&other) {
                references.has_live_reference = true;
            } else if state.status != "rejected" {
                // This record has a rejection result but has not itself reached
                // the retention reconciliation step yet. Preserve the shared
                // file until it is processed later in the same/page-next run.
                references.has_pending_reference = true;
            } else {
                references.rejected_references.push(other);
            }
        }
        let Some(next) = page.next_cursor else {
            return Ok(references);
        };
        cursor = Some(next);
    }
}

fn persist_record_and_checkpoint(
    catalog: &mut Catalog,
    record: &CatalogRecord,
    checkpoint: &mut CatalogImageFetchCheckpoint,
) -> CatalogImageFetchResult<()> {
    checkpoint.updated_at = utc_now();
    let payload = serde_json::to_string(checkpoint)?;
    let update = NewCatalogRecord {
        id: record.id.clone(),
        image_path: record.image_path.clone(),
        thumbnail_path: record.thumbnail_path.clone(),
        embedding_path: record.embedding_path.clone(),
        artifact_path: record.artifact_path.clone(),
        metadata: record.metadata.clone(),
    };
    catalog.append_records_and_metadata(&[update], &[(FETCH_CHECKPOINT_KEY, payload.as_str())])?;
    Ok(())
}

fn persist_checkpoint(
    catalog: &mut Catalog,
    checkpoint: &CatalogImageFetchCheckpoint,
) -> CatalogImageFetchResult<()> {
    let payload = serde_json::to_string(checkpoint)?;
    catalog.append_records_and_metadata(&[], &[(FETCH_CHECKPOINT_KEY, payload.as_str())])?;
    Ok(())
}

enum CompletedValidation {
    Valid,
    Tampered,
    Incomplete,
}

fn validate_completed_record(
    catalog: &Catalog,
    record: &CatalogRecord,
) -> CatalogImageFetchResult<CompletedValidation> {
    let state = acquisition_state(record);
    if state.status == "rejected"
        && matches!(state.availability.as_str(), "discarded" | "shared")
        && !record_is_accepted(record)
    {
        return Ok(CompletedValidation::Valid);
    }
    if state.status != "available"
        && !(state.status == "rejected" && state.availability == "shared")
    {
        return Ok(CompletedValidation::Incomplete);
    }
    let (Some(path), Some(hash), Some(bytes)) = (
        state.cache_path.as_deref(),
        state.content_hash.as_deref(),
        state.bytes,
    ) else {
        return Ok(CompletedValidation::Tampered);
    };
    let path = safe_catalog_file(catalog, path, false)?;
    if !path.is_file() || validate_cached_file(&path, hash, bytes).is_err() {
        return Ok(CompletedValidation::Tampered);
    }
    Ok(CompletedValidation::Valid)
}

fn record_is_accepted(record: &CatalogRecord) -> bool {
    record
        .metadata
        .get("analysis")
        .and_then(|analysis| analysis.get("accepted"))
        .and_then(Value::as_bool)
        != Some(false)
}

fn acquisition_state(record: &CatalogRecord) -> AcquisitionState {
    record
        .metadata
        .get("acquisition")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

fn set_acquisition_state(
    record: &mut CatalogRecord,
    state: &AcquisitionState,
) -> CatalogImageFetchResult<()> {
    let object = record.metadata.as_object_mut().ok_or_else(|| {
        CatalogImageFetchError::InvalidOptions(format!(
            "catalog record {} metadata must be an object",
            record.id
        ))
    })?;
    object.insert("acquisition".to_owned(), serde_json::to_value(state)?);
    Ok(())
}

fn validate_options(options: &CatalogImageFetchOptions) -> CatalogImageFetchResult<()> {
    if options.accepted_target == 0 {
        return Err(CatalogImageFetchError::InvalidOptions(
            "acceptedTarget must be positive".to_owned(),
        ));
    }
    if options.concurrency == 0 || options.concurrency > MAX_CONCURRENCY {
        return Err(CatalogImageFetchError::InvalidOptions(format!(
            "concurrency must be between 1 and {MAX_CONCURRENCY}"
        )));
    }
    if options.page_size == 0 || options.page_size > MAX_PAGE_SIZE {
        return Err(CatalogImageFetchError::InvalidOptions(format!(
            "pageSize must be between 1 and {MAX_PAGE_SIZE}"
        )));
    }
    if options.max_file_bytes == 0
        || options.max_file_bytes > MAX_IMAGE_BYTES
        || options.max_file_bytes > u32::MAX as usize
        || options.max_in_flight_bytes < options.max_file_bytes
        || options.max_in_flight_bytes > MAX_IN_FLIGHT_BYTES
    {
        return Err(CatalogImageFetchError::InvalidOptions(
            "encoded image and in-flight byte limits are invalid".to_owned(),
        ));
    }
    if options.max_width == 0
        || options.max_height == 0
        || options.max_width > MAX_IMAGE_EDGE
        || options.max_height > MAX_IMAGE_EDGE
        || options.max_pixels == 0
        || options.max_pixels > MAX_DECODED_PIXELS
    {
        return Err(CatalogImageFetchError::InvalidOptions(
            "decoded image dimension limits are invalid".to_owned(),
        ));
    }
    if [
        options.connect_timeout_ms,
        options.read_timeout_ms,
        options.hop_timeout_ms,
    ]
    .contains(&0)
        || options.max_retries > MAX_RETRIES
        || options.max_cache_bytes == 0
        || options.thumbnail_max_edge == Some(0)
        || options.max_thumbnail_bytes == 0
    {
        return Err(CatalogImageFetchError::InvalidOptions(
            "fetch timeout, retry, cache, or thumbnail limits are invalid".to_owned(),
        ));
    }
    Ok(())
}

fn prepare_cache_directories(catalog: &Catalog) -> CatalogImageFetchResult<()> {
    for relative in [
        "images/sha256",
        "images/.fetch-tmp",
        "thumbnails/sha256",
        "thumbnails/.fetch-tmp",
        "artifacts/fetch-completions",
        "artifacts/.fetch-tmp",
    ] {
        safe_catalog_directory(catalog, relative, true)?;
    }
    Ok(())
}

fn cleanup_stale_temps(catalog: &Catalog) -> CatalogImageFetchResult<()> {
    for relative in [
        "images/.fetch-tmp",
        "thumbnails/.fetch-tmp",
        "artifacts/.fetch-tmp",
    ] {
        let directory = safe_catalog_directory(catalog, relative, false)?;
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() && entry.file_name().to_string_lossy().ends_with(".tmp") {
                std::fs::remove_file(entry.path())?;
            }
        }
    }
    Ok(())
}

fn safe_catalog_directory(
    catalog: &Catalog,
    relative: &str,
    create: bool,
) -> CatalogImageFetchResult<PathBuf> {
    let path = safe_relative_join(catalog.root(), relative)?;
    let mut current = catalog.root().to_path_buf();
    for component in Path::new(relative).components() {
        let Component::Normal(component) = component else {
            return Err(CatalogImageFetchError::InvalidOptions(
                "catalog artifact path must be relative".to_owned(),
            ));
        };
        current.push(component);
        if !current.exists() {
            if create {
                std::fs::create_dir(&current)?;
            } else {
                return Ok(path);
            }
        }
        let metadata = std::fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(CatalogImageFetchError::InvalidOptions(format!(
                "catalog cache directory must be a real directory: {}",
                current.display()
            )));
        }
    }
    Ok(path)
}

fn safe_catalog_file(
    catalog: &Catalog,
    relative: &str,
    create_parent: bool,
) -> CatalogImageFetchResult<PathBuf> {
    let path = safe_relative_join(catalog.root(), relative)?;
    let parent_relative = Path::new(relative)
        .parent()
        .and_then(Path::to_str)
        .ok_or_else(|| {
            CatalogImageFetchError::InvalidOptions("catalog cache path has no parent".to_owned())
        })?;
    safe_catalog_directory(catalog, parent_relative, create_parent)?;
    if let Ok(metadata) = std::fs::symlink_metadata(&path) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(CatalogImageFetchError::InvalidOptions(format!(
                "catalog cache file must be a regular file: {}",
                path.display()
            )));
        }
    }
    Ok(path)
}

fn safe_relative_join(root: &Path, relative: &str) -> CatalogImageFetchResult<PathBuf> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|component| {
            !matches!(component, Component::Normal(_))
                || component.as_os_str().to_string_lossy().contains('\0')
        })
    {
        return Err(CatalogImageFetchError::InvalidOptions(
            "catalog cache path must contain only normal relative components".to_owned(),
        ));
    }
    Ok(root.join(path))
}

fn atomic_create(path: &Path, bytes: &[u8]) -> CatalogImageFetchResult<()> {
    atomic_write_with_hook(path, bytes, false, || Ok(()))
}

fn atomic_replace(path: &Path, bytes: &[u8]) -> CatalogImageFetchResult<()> {
    atomic_write_with_hook(path, bytes, true, || Ok(()))
}

fn atomic_write_with_hook<F>(
    path: &Path,
    bytes: &[u8],
    replace: bool,
    before_rename: F,
) -> CatalogImageFetchResult<()>
where
    F: FnOnce() -> std::io::Result<()>,
{
    let parent = path.parent().ok_or_else(|| {
        CatalogImageFetchError::InvalidOptions("cache path has no parent".to_owned())
    })?;
    let temp_directory = parent
        .ancestors()
        .find(|candidate| {
            candidate
                .file_name()
                .is_some_and(|name| name == "images" || name == "thumbnails" || name == "artifacts")
        })
        .map(|artifact_root| artifact_root.join(TEMP_DIRECTORY))
        .unwrap_or_else(|| parent.join(TEMP_DIRECTORY));
    std::fs::create_dir_all(&temp_directory)?;
    let temp = temp_directory.join(format!("{}.tmp", uuid::Uuid::new_v4()));
    let temp_metadata = std::fs::symlink_metadata(&temp_directory)?;
    if temp_metadata.file_type().is_symlink() || !temp_metadata.is_dir() {
        return Err(CatalogImageFetchError::InvalidOptions(format!(
            "catalog cache temporary directory must be a real directory: {}",
            temp_directory.display()
        )));
    }
    let result = (|| -> CatalogImageFetchResult<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        sync_parent_directory(&temp);
        before_rename()?;
        if !replace && path.exists() {
            return Err(CatalogImageFetchError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("cache destination already exists: {}", path.display()),
            )));
        }
        match std::fs::rename(&temp, path) {
            Ok(()) => {
                sync_parent_directory(path);
                Ok(())
            }
            Err(error) if path.exists() => {
                if replace {
                    Err(CatalogImageFetchError::Io(std::io::Error::new(
                        error.kind(),
                        format!(
                            "atomic cache replacement failed while preserving {}: {error}",
                            path.display()
                        ),
                    )))
                } else {
                    Err(CatalogImageFetchError::Io(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        format!(
                            "cache destination appeared during atomic write: {} ({error})",
                            path.display()
                        ),
                    )))
                }
            }
            Err(error) => Err(error.into()),
        }
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result
}

fn sync_parent_directory(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
    }
}

fn completion_marker_path(catalog: &Catalog, record_id: &str) -> CatalogImageFetchResult<PathBuf> {
    let id_hash = hex_digest(record_id.as_bytes());
    safe_catalog_file(
        catalog,
        &format!("artifacts/{COMPLETION_DIRECTORY}/{id_hash}.json"),
        true,
    )
}

fn remove_completion_marker(catalog: &Catalog, record_id: &str) -> CatalogImageFetchResult<()> {
    let path = completion_marker_path(catalog, record_id)?;
    if path.exists() {
        std::fs::remove_file(&path)?;
        sync_parent_directory(&path);
    }
    Ok(())
}

fn persist_completion_marker(
    catalog: &Catalog,
    marker: &CompletionMarker,
    max_cache_bytes: u64,
) -> CatalogImageFetchResult<()> {
    let path = completion_marker_path(catalog, &marker.record_id)?;
    let payload = serde_json::to_vec(marker)?;
    let existing_bytes = std::fs::metadata(&path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let projected = catalog
        .storage_accounting()?
        .artifact_bytes
        .saturating_sub(existing_bytes)
        .saturating_add(payload.len() as u64);
    if projected > max_cache_bytes {
        return Err(CatalogImageFetchError::InvalidOptions(format!(
            "catalog completion metadata would exceed its {max_cache_bytes} byte artifact limit"
        )));
    }
    if path.exists() {
        let existing: CompletionMarker = serde_json::from_slice(&std::fs::read(&path)?)?;
        if existing == *marker {
            return Ok(());
        }
        return atomic_replace(&path, &payload);
    }
    atomic_create(&path, &payload)
}

fn recover_completion_marker(
    catalog: &Catalog,
    record: &CatalogRecord,
) -> CatalogImageFetchResult<Option<CompletionMarker>> {
    let state = acquisition_state(record);
    if matches!(state.status.as_str(), "available" | "rejected") {
        return Ok(None);
    }
    let path = completion_marker_path(catalog, &record.id)?;
    if !path.is_file() {
        return Ok(None);
    }
    let payload = std::fs::read(&path)?;
    let marker: CompletionMarker = match serde_json::from_slice(&payload) {
        Ok(marker) => marker,
        Err(_) => {
            std::fs::remove_file(&path)?;
            return Ok(None);
        }
    };
    if marker.record_id != record.id {
        std::fs::remove_file(&path)?;
        return Ok(None);
    }
    let cached = safe_catalog_file(catalog, &marker.cache_path, false)?;
    if validate_cached_file(&cached, &marker.content_hash, marker.bytes).is_err() {
        std::fs::remove_file(&path)?;
        return Ok(None);
    }
    Ok(Some(marker))
}

fn validate_cached_file(path: &Path, expected_hash: &str, bytes: u64) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "cached image metadata mismatch",
        ));
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if format!("{:x}", hasher.finalize()) != expected_hash {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "cached image hash mismatch",
        ));
    }
    Ok(())
}

fn bounded_failure_reason(error: &WorkerError) -> String {
    error.to_string().chars().take(512).collect()
}

fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use tempfile::TempDir;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(width, height, Rgb([12, 34, 56])));
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("encode fixture");
        bytes
    }

    fn catalog_fixture() -> (TempDir, CatalogRegistry, String) {
        let root = TempDir::new().expect("temp root");
        let registry = CatalogRegistry::new(root.path().join("registry.json"));
        let mut catalog = registry
            .create_catalog(root.path().join("catalog"), "fetch-test")
            .expect("create catalog");
        let id = catalog.descriptor().id.clone();
        catalog
            .append_records(&[NewCatalogRecord {
                id: "pq_test".to_owned(),
                image_path: "images/pq_test".to_owned(),
                thumbnail_path: None,
                embedding_path: None,
                artifact_path: None,
                metadata: json!({"url": "https://example.com/image.png", "caption": "test"}),
            }])
            .expect("append record");
        (root, registry, id)
    }

    fn catalog_with_urls(urls: &[&str]) -> (TempDir, CatalogRegistry, String) {
        let root = TempDir::new().expect("temp root");
        let registry = CatalogRegistry::new(root.path().join("registry.json"));
        let mut catalog = registry
            .create_catalog(root.path().join("catalog"), "fetch-test")
            .expect("create catalog");
        let id = catalog.descriptor().id.clone();
        let records = urls
            .iter()
            .enumerate()
            .map(|(index, url)| NewCatalogRecord {
                id: format!("pq_{index:04}"),
                image_path: format!("images/pq_{index:04}"),
                thumbnail_path: None,
                embedding_path: None,
                artifact_path: None,
                metadata: json!({"url": url, "caption": format!("candidate {index}")}),
            })
            .collect::<Vec<_>>();
        catalog.append_records(&records).expect("append records");
        (root, registry, id)
    }

    fn set_analysis_acceptance(registry: &CatalogRegistry, id: &str, accepted: &[bool]) {
        let mut catalog = registry.open_attached(id).expect("open");
        let records = catalog
            .page_records(0, accepted.len() as u32)
            .expect("page")
            .into_iter()
            .zip(accepted)
            .map(|(mut record, accepted)| {
                record.metadata["analysis"] =
                    json!({"accepted": accepted, "reason": "lifecycle-test"});
                NewCatalogRecord {
                    id: record.id,
                    image_path: record.image_path,
                    thumbnail_path: record.thumbnail_path,
                    embedding_path: record.embedding_path,
                    artifact_path: record.artifact_path,
                    metadata: record.metadata,
                }
            })
            .collect::<Vec<_>>();
        catalog.append_records(&records).expect("persist analysis");
    }

    #[test]
    fn image_validation_rejects_dimension_bomb_before_decode() {
        let bytes = png(32, 32);
        let permit = Arc::new(Semaphore::new(bytes.len()))
            .try_acquire_many_owned(bytes.len() as u32)
            .expect("permit");
        let options = CatalogImageFetchOptions {
            max_width: 16,
            max_height: 16,
            max_pixels: 256,
            ..CatalogImageFetchOptions::default()
        };
        let error = validate_image(bytes, &options, permit).expect_err("dimensions rejected");
        assert!(error.to_string().contains("dimensions"));
    }

    #[test]
    fn completion_marker_recovers_cache_without_network() {
        let (_root, registry, id) = catalog_fixture();
        let catalog = registry.open_attached(&id).expect("open");
        prepare_cache_directories(&catalog).expect("prepare");
        let record = catalog.page_records(0, 1).expect("page").remove(0);
        let bytes = png(8, 8);
        let hash = hex_digest(&bytes);
        let relative = format!("images/sha256/{}/{}.png", &hash[..2], hash);
        let path = safe_catalog_file(&catalog, &relative, true).expect("path");
        atomic_create(&path, &bytes).expect("cache");
        let marker = CompletionMarker {
            record_id: record.id.clone(),
            content_hash: hash.clone(),
            cache_path: relative,
            thumbnail_path: None,
            bytes: bytes.len() as u64,
            width: 8,
            height: 8,
        };
        persist_completion_marker(&catalog, &marker, u64::MAX).expect("base marker");
        let thumbnail_relative = format!("thumbnails/sha256/{}/{}.jpg", &hash[..2], hash);
        let thumbnail = safe_catalog_file(&catalog, &thumbnail_relative, true).expect("thumbnail");
        atomic_create(&thumbnail, b"bounded thumbnail").expect("write thumbnail");
        let marker = CompletionMarker {
            thumbnail_path: Some(thumbnail_relative),
            ..marker
        };
        // Simulates the crash window after the thumbnail is written but before
        // SQLite is updated: the richer marker must replace the base marker.
        persist_completion_marker(&catalog, &marker, u64::MAX).expect("updated marker");
        assert_eq!(
            recover_completion_marker(&catalog, &record)
                .expect("recover")
                .expect("present")
                .thumbnail_path,
            marker.thumbnail_path
        );
    }

    #[tokio::test]
    async fn interrupted_marker_replacement_preserves_old_resume_marker() {
        let (_root, registry, id) = catalog_fixture();
        let catalog = registry.open_attached(&id).expect("open");
        prepare_cache_directories(&catalog).expect("prepare");
        let record = catalog.page_records(0, 1).expect("page").remove(0);
        let bytes = png(8, 8);
        let hash = hex_digest(&bytes);
        let relative = format!("images/sha256/{}/{}.png", &hash[..2], hash);
        let cache = safe_catalog_file(&catalog, &relative, true).expect("cache path");
        atomic_create(&cache, &bytes).expect("cache");
        let old = CompletionMarker {
            record_id: record.id.clone(),
            content_hash: hash,
            cache_path: relative,
            thumbnail_path: None,
            bytes: bytes.len() as u64,
            width: 8,
            height: 8,
        };
        persist_completion_marker(&catalog, &old, u64::MAX).expect("old marker");
        let replacement = CompletionMarker {
            thumbnail_path: Some("thumbnails/sha256/new.jpg".to_owned()),
            ..old.clone()
        };
        let marker_path = completion_marker_path(&catalog, &record.id).expect("marker path");
        let error = atomic_write_with_hook(
            &marker_path,
            &serde_json::to_vec(&replacement).expect("serialize"),
            true,
            || Err(std::io::Error::other("injected interruption")),
        )
        .expect_err("replacement interrupted");
        assert!(error.to_string().contains("injected interruption"));
        let durable: CompletionMarker =
            serde_json::from_slice(&std::fs::read(&marker_path).expect("old marker survives"))
                .expect("old marker valid");
        assert_eq!(durable, old);
        drop(catalog);

        let network_calls = Arc::new(Mutex::new(0_u64));
        let options = CatalogImageFetchOptions {
            accepted_target: 1,
            thumbnail_max_edge: None,
            ..CatalogImageFetchOptions::default()
        };
        let report = fetch_attached_catalog_images_with(&registry, &id, &options, {
            let network_calls = Arc::clone(&network_calls);
            move |_url, _options| {
                let network_calls = Arc::clone(&network_calls);
                async move {
                    *network_calls.lock().expect("calls") += 1;
                    Err(WorkerError::InvalidPayload(
                        "durable marker must avoid redownload".to_owned(),
                    ))
                }
            }
        })
        .await
        .expect("resume");
        assert_eq!(report.checkpoint.counts.accepted, 1);
        assert_eq!(*network_calls.lock().expect("calls"), 0);
    }

    #[test]
    fn tampered_cache_is_not_a_resume_hit() {
        let (_root, registry, id) = catalog_fixture();
        let mut catalog = registry.open_attached(&id).expect("open");
        prepare_cache_directories(&catalog).expect("prepare");
        let mut record = catalog.page_records(0, 1).expect("page").remove(0);
        let bytes = png(8, 8);
        let hash = hex_digest(&bytes);
        let relative = format!("images/sha256/{}/{}.png", &hash[..2], hash);
        let path = safe_catalog_file(&catalog, &relative, true).expect("path");
        atomic_create(&path, &bytes).expect("cache");
        let marker = CompletionMarker {
            record_id: record.id.clone(),
            content_hash: hash,
            cache_path: relative,
            thumbnail_path: None,
            bytes: bytes.len() as u64,
            width: 8,
            height: 8,
        };
        let mut checkpoint = CatalogImageFetchCheckpoint {
            requested_accepted_target: 1,
            exhausted: false,
            counts: CatalogImageFetchCounts::default(),
            updated_at: utc_now(),
        };
        apply_available(&mut catalog, &mut record, &marker, 1, 0, &mut checkpoint).expect("state");
        std::fs::write(path, b"tampered").expect("tamper");
        assert!(matches!(
            validate_completed_record(&catalog, &record).expect("validate"),
            CompletedValidation::Tampered
        ));
    }

    #[test]
    fn cache_paths_reject_parent_and_symlink_escape() {
        let (_root, registry, id) = catalog_fixture();
        let catalog = registry.open_attached(&id).expect("open");
        assert!(safe_catalog_file(&catalog, "../outside", true).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                std::env::temp_dir(),
                catalog.root().join("images").join("escape"),
            )
            .expect("symlink");
            assert!(safe_catalog_file(&catalog, "images/escape/file", true).is_err());
        }
    }

    #[test]
    fn stale_temp_cleanup_is_scoped_to_owned_directories() {
        let (_root, registry, id) = catalog_fixture();
        let catalog = registry.open_attached(&id).expect("open");
        prepare_cache_directories(&catalog).expect("prepare");
        let owned = catalog.root().join("images/.fetch-tmp/crash.tmp");
        let unrelated = catalog.root().join("images/keep.tmp");
        std::fs::write(&owned, b"partial").expect("owned");
        std::fs::write(&unrelated, b"keep").expect("unrelated");
        cleanup_stale_temps(&catalog).expect("cleanup");
        assert!(!owned.exists());
        assert!(unrelated.exists());
    }

    #[test]
    fn acquisition_metadata_is_separate_from_analysis() {
        let (_root, registry, id) = catalog_fixture();
        let catalog = registry.open_attached(&id).expect("open");
        let mut record = catalog.page_records(0, 1).expect("page").remove(0);
        record.metadata["analysis"] = json!({"accepted": false, "tag": "kept"});
        set_acquisition_state(
            &mut record,
            &AcquisitionState {
                status: "failed".to_owned(),
                failure_reason: Some("dead URL".to_owned()),
                ..AcquisitionState::default()
            },
        )
        .expect("state");
        assert_eq!(record.metadata["analysis"]["tag"], "kept");
        assert_eq!(record.metadata["acquisition"]["status"], "failed");
    }

    #[tokio::test]
    async fn accepted_target_continues_past_failures_and_resume_skips_completed_content() {
        let (_root, registry, id) = catalog_with_urls(&[
            "https://example.com/dead",
            "https://example.com/malformed",
            "https://example.com/good-1",
            "https://example.com/good-2",
            "https://example.com/not-needed",
        ]);
        let good = png(12, 10);
        let calls = Arc::new(Mutex::new(BTreeMap::<String, u64>::new()));
        let fetcher = {
            let calls = Arc::clone(&calls);
            let good = good.clone();
            move |url: String, _options: PublicSourceUrlFetchOptions| {
                let calls = Arc::clone(&calls);
                let good = good.clone();
                async move {
                    *calls
                        .lock()
                        .expect("calls lock")
                        .entry(url.clone())
                        .or_default() += 1;
                    if url.ends_with("/dead") {
                        Err(WorkerError::InvalidPayload("connection refused".to_owned()))
                    } else if url.ends_with("/malformed") {
                        Ok(b"not an image".to_vec())
                    } else {
                        Ok(good)
                    }
                }
            }
        };
        let options = CatalogImageFetchOptions {
            accepted_target: 2,
            concurrency: 2,
            page_size: 2,
            max_retries: 1,
            retry_backoff_ms: 1,
            thumbnail_max_edge: None,
            ..CatalogImageFetchOptions::default()
        };
        let first = fetch_attached_catalog_images_with(&registry, &id, &options, fetcher.clone())
            .await
            .expect("first pass");
        assert_eq!(first.checkpoint.counts.accepted, 2);
        assert_eq!(first.checkpoint.counts.failures, 2);
        assert_eq!(first.checkpoint.counts.retries, 1);
        assert!(!first.checkpoint.exhausted);
        assert_eq!(
            calls
                .lock()
                .expect("calls")
                .get("https://example.com/not-needed")
                .copied()
                .unwrap_or(0),
            0
        );
        let before = calls.lock().expect("calls").clone();

        let second = fetch_attached_catalog_images_with(&registry, &id, &options, fetcher)
            .await
            .expect("resume pass");
        assert_eq!(second.checkpoint.counts.accepted, 2);
        assert_eq!(second.checkpoint.counts.cache_hits, 2);
        let after = calls.lock().expect("calls");
        assert_eq!(
            after["https://example.com/good-1"],
            before["https://example.com/good-1"]
        );
        assert_eq!(
            after["https://example.com/good-2"],
            before["https://example.com/good-2"]
        );
    }

    #[tokio::test]
    async fn duplicate_content_uses_one_content_addressed_file() {
        let (_root, registry, id) =
            catalog_with_urls(&["https://one.example/image", "https://two.example/image"]);
        let bytes = png(9, 7);
        let expected_bytes = bytes.clone();
        let options = CatalogImageFetchOptions {
            accepted_target: 2,
            concurrency: 1,
            thumbnail_max_edge: None,
            ..CatalogImageFetchOptions::default()
        };
        let report =
            fetch_attached_catalog_images_with(&registry, &id, &options, move |_url, _options| {
                let bytes = bytes.clone();
                async move { Ok(bytes) }
            })
            .await
            .expect("fetch duplicates");
        assert_eq!(report.checkpoint.counts.accepted, 2);
        assert_eq!(report.checkpoint.counts.duplicate_content, 1);
        let catalog = registry.open_attached(&id).expect("open");
        let hash_root = catalog.root().join("images/sha256");
        let image_files = walk_regular_files(&hash_root);
        assert_eq!(image_files, 1);
        let database = std::fs::read(catalog.database_path()).expect("read SQLite");
        assert!(
            !database
                .windows(expected_bytes.len())
                .any(|window| window == expected_bytes),
            "encoded image bytes must stay in filesystem artifacts, never SQLite blobs"
        );
    }

    #[tokio::test]
    async fn low_quota_duplicate_catalog_compacts_completion_markers() {
        let urls = (0..24)
            .map(|index| format!("https://example.com/duplicate-{index}"))
            .collect::<Vec<_>>();
        let url_refs = urls.iter().map(String::as_str).collect::<Vec<_>>();
        let (_root, registry, id) = catalog_with_urls(&url_refs);
        let acceptance = (0..urls.len())
            .map(|index| index % 2 == 1)
            .collect::<Vec<_>>();
        set_analysis_acceptance(&registry, &id, &acceptance);
        let bytes = png(9, 7);
        let options = CatalogImageFetchOptions {
            accepted_target: acceptance.iter().filter(|accepted| **accepted).count() as u64,
            concurrency: 1,
            thumbnail_max_edge: None,
            max_cache_bytes: bytes.len() as u64 + 1_024,
            ..CatalogImageFetchOptions::default()
        };
        let report =
            fetch_attached_catalog_images_with(&registry, &id, &options, move |_url, _options| {
                let bytes = bytes.clone();
                async move { Ok(bytes) }
            })
            .await
            .expect("bounded duplicates");
        assert_eq!(report.checkpoint.counts.accepted, 12);
        assert_eq!(report.checkpoint.counts.rejected, 12);
        let catalog = registry.open_attached(&id).expect("open");
        assert_eq!(
            walk_regular_files(&catalog.root().join("artifacts/fetch-completions")),
            0,
            "completion markers are crash journals, not permanent per-record artifacts"
        );
        assert!(
            catalog
                .storage_accounting()
                .expect("accounting")
                .artifact_bytes
                <= options.max_cache_bytes
        );
        assert_eq!(walk_regular_files(&catalog.root().join("images/sha256")), 1);
    }

    #[tokio::test]
    async fn duplicate_cache_hit_cannot_bypass_completion_marker_quota() {
        let (_root, registry, id) = catalog_with_urls(&["https://example.com/existing-duplicate"]);
        let bytes = png(9, 7);
        {
            let catalog = registry.open_attached(&id).expect("open");
            prepare_cache_directories(&catalog).expect("prepare");
            let hash = hex_digest(&bytes);
            let relative = format!("images/sha256/{}/{}.png", &hash[..2], hash);
            let path = safe_catalog_file(&catalog, &relative, true).expect("cache path");
            atomic_create(&path, &bytes).expect("preexisting content");
        }
        let options = CatalogImageFetchOptions {
            accepted_target: 1,
            thumbnail_max_edge: None,
            max_cache_bytes: bytes.len() as u64,
            ..CatalogImageFetchOptions::default()
        };
        let error =
            fetch_attached_catalog_images_with(&registry, &id, &options, move |_url, _options| {
                let bytes = bytes.clone();
                async move { Ok(bytes) }
            })
            .await
            .expect_err("marker must respect quota");
        assert!(error.to_string().contains("completion metadata"));
        let catalog = registry.open_attached(&id).expect("reopen");
        assert_eq!(
            walk_regular_files(&catalog.root().join("artifacts/fetch-completions")),
            0
        );
        assert!(
            catalog
                .storage_accounting()
                .expect("accounting")
                .artifact_bytes
                <= options.max_cache_bytes
        );
    }

    #[tokio::test]
    async fn rejected_record_discards_original_but_keeps_thumbnail_and_analysis() {
        let (_root, registry, id) = catalog_with_urls(&["https://example.com/rejected-image"]);
        {
            let mut catalog = registry.open_attached(&id).expect("open");
            let mut record = catalog.page_records(0, 1).expect("page").remove(0);
            record.metadata["analysis"] = json!({"accepted": false, "reason": "vision-filter"});
            catalog
                .append_records(&[NewCatalogRecord {
                    id: record.id,
                    image_path: record.image_path,
                    thumbnail_path: record.thumbnail_path,
                    embedding_path: record.embedding_path,
                    artifact_path: record.artifact_path,
                    metadata: record.metadata,
                }])
                .expect("persist analysis");
        }
        let bytes = png(32, 24);
        let options = CatalogImageFetchOptions {
            accepted_target: 1,
            discard_rejected_originals: true,
            thumbnail_max_edge: Some(16),
            ..CatalogImageFetchOptions::default()
        };
        let report =
            fetch_attached_catalog_images_with(&registry, &id, &options, move |_url, _options| {
                let bytes = bytes.clone();
                async move { Ok(bytes) }
            })
            .await
            .expect("fetch");
        assert_eq!(report.checkpoint.counts.rejected, 1);
        assert!(report.checkpoint.exhausted);
        let catalog = registry.open_attached(&id).expect("reopen");
        let record = catalog.page_records(0, 1).expect("page").remove(0);
        let state = acquisition_state(&record);
        assert_eq!(state.status, "rejected");
        assert_eq!(state.availability, "discarded");
        assert_eq!(record.metadata["analysis"]["reason"], "vision-filter");
        assert!(!catalog
            .root()
            .join(state.cache_path.expect("cache path"))
            .exists());
        assert!(catalog
            .root()
            .join(state.thumbnail_path.expect("thumbnail path"))
            .is_file());
    }

    #[tokio::test]
    async fn later_analysis_rejection_discards_an_already_completed_original() {
        let (_root, registry, id) = catalog_with_urls(&["https://example.com/later-rejected"]);
        let bytes = png(20, 18);
        let options = CatalogImageFetchOptions {
            accepted_target: 1,
            thumbnail_max_edge: None,
            ..CatalogImageFetchOptions::default()
        };
        fetch_attached_catalog_images_with(&registry, &id, &options, {
            let bytes = bytes.clone();
            move |_url, _options| {
                let bytes = bytes.clone();
                async move { Ok(bytes) }
            }
        })
        .await
        .expect("initial fetch");
        let cache_path = {
            let catalog = registry.open_attached(&id).expect("open");
            let record = catalog.page_records(0, 1).expect("page").remove(0);
            catalog
                .root()
                .join(acquisition_state(&record).cache_path.expect("cache path"))
        };
        assert!(cache_path.is_file());
        set_analysis_acceptance(&registry, &id, &[false]);
        let network_calls = Arc::new(Mutex::new(0_u64));
        let report = fetch_attached_catalog_images_with(&registry, &id, &options, {
            let network_calls = Arc::clone(&network_calls);
            move |_url, _options| {
                let network_calls = Arc::clone(&network_calls);
                async move {
                    *network_calls.lock().expect("calls") += 1;
                    Err(WorkerError::InvalidPayload(
                        "network must not be used for a valid cache hit".to_owned(),
                    ))
                }
            }
        })
        .await
        .expect("reconcile rejection");
        assert_eq!(report.checkpoint.counts.rejected, 1);
        assert_eq!(*network_calls.lock().expect("calls"), 0);
        assert!(!cache_path.exists());
        let catalog = registry.open_attached(&id).expect("reopen");
        let record = catalog.page_records(0, 1).expect("page").remove(0);
        assert_eq!(acquisition_state(&record).availability, "discarded");
    }

    #[tokio::test]
    async fn shared_content_keeps_one_live_reference_then_discards_when_all_rejected() {
        let (_root, registry, id) = catalog_with_urls(&[
            "https://example.com/duplicate-a",
            "https://example.com/duplicate-b",
        ]);
        let bytes = png(14, 13);
        let options = CatalogImageFetchOptions {
            accepted_target: 2,
            concurrency: 1,
            thumbnail_max_edge: None,
            ..CatalogImageFetchOptions::default()
        };
        fetch_attached_catalog_images_with(&registry, &id, &options, {
            let bytes = bytes.clone();
            move |_url, _options| {
                let bytes = bytes.clone();
                async move { Ok(bytes) }
            }
        })
        .await
        .expect("initial duplicate fetch");
        let cache_path = {
            let catalog = registry.open_attached(&id).expect("open");
            let record = catalog.page_records(0, 1).expect("page").remove(0);
            catalog
                .root()
                .join(acquisition_state(&record).cache_path.expect("cache path"))
        };

        // The first record is encountered before the still-live second record,
        // so its rejection must not delete their shared content.
        set_analysis_acceptance(&registry, &id, &[false, true]);
        let one_target = CatalogImageFetchOptions {
            accepted_target: 1,
            ..options.clone()
        };
        fetch_attached_catalog_images_with(&registry, &id, &one_target, |_url, _options| async {
            Err(WorkerError::InvalidPayload(
                "valid cache hits must not fetch".to_owned(),
            ))
        })
        .await
        .expect("one accepted");
        assert!(cache_path.is_file());
        {
            let catalog = registry.open_attached(&id).expect("open");
            let records = catalog.page_records(0, 2).expect("page");
            assert_eq!(acquisition_state(&records[0]).availability, "shared");
            assert_eq!(acquisition_state(&records[1]).availability, "available");
        }

        // Once both analysis results reject, the first rejected/shared record
        // is not a live reference. The last live record discards the file and
        // both records are reconciled to `discarded`.
        set_analysis_acceptance(&registry, &id, &[false, false]);
        let report = fetch_attached_catalog_images_with(
            &registry,
            &id,
            &one_target,
            |_url, _options| async {
                Err(WorkerError::InvalidPayload(
                    "rejected cache records must not refetch".to_owned(),
                ))
            },
        )
        .await
        .expect("all rejected");
        assert_eq!(report.checkpoint.counts.rejected, 2);
        assert!(report.checkpoint.exhausted);
        assert!(!cache_path.exists());
        let catalog = registry.open_attached(&id).expect("reopen");
        for record in catalog.page_records(0, 2).expect("page") {
            assert_eq!(acquisition_state(&record).availability, "discarded");
        }
    }

    #[tokio::test]
    async fn accepted_target_between_rejected_duplicates_does_not_strand_cleanup() {
        let (_root, registry, id) = catalog_with_urls(&[
            "https://example.com/rejected-a",
            "https://example.com/accepted-target",
            "https://example.com/rejected-b",
        ]);
        let duplicate = png(12, 12);
        let accepted = png(13, 12);
        let options = CatalogImageFetchOptions {
            accepted_target: 3,
            concurrency: 1,
            thumbnail_max_edge: None,
            ..CatalogImageFetchOptions::default()
        };
        fetch_attached_catalog_images_with(&registry, &id, &options, {
            let duplicate = duplicate.clone();
            let accepted = accepted.clone();
            move |url, _options| {
                let bytes = if url.ends_with("accepted-target") {
                    accepted.clone()
                } else {
                    duplicate.clone()
                };
                async move { Ok(bytes) }
            }
        })
        .await
        .expect("initial fetch");
        set_analysis_acceptance(&registry, &id, &[false, true, false]);
        let (duplicate_path, accepted_path) = {
            let catalog = registry.open_attached(&id).expect("open");
            let records = catalog.page_records(0, 3).expect("page");
            (
                catalog.root().join(
                    acquisition_state(&records[0])
                        .cache_path
                        .expect("duplicate path"),
                ),
                catalog.root().join(
                    acquisition_state(&records[1])
                        .cache_path
                        .expect("accepted path"),
                ),
            )
        };
        let one_target = CatalogImageFetchOptions {
            accepted_target: 1,
            ..options
        };
        let report = fetch_attached_catalog_images_with(
            &registry,
            &id,
            &one_target,
            |_url, _options| async {
                Err(WorkerError::InvalidPayload(
                    "completed records must not fetch".to_owned(),
                ))
            },
        )
        .await
        .expect("retention reconciliation");
        assert_eq!(report.checkpoint.counts.accepted, 1);
        assert_eq!(report.checkpoint.counts.rejected, 2);
        assert!(!duplicate_path.exists());
        assert!(accepted_path.is_file());
        let catalog = registry.open_attached(&id).expect("reopen");
        let records = catalog.page_records(0, 3).expect("page");
        assert_eq!(acquisition_state(&records[0]).availability, "discarded");
        assert_eq!(acquisition_state(&records[2]).availability, "discarded");
    }

    #[tokio::test]
    async fn tampered_completed_cache_is_refetched_and_repaired() {
        let (_root, registry, id) = catalog_with_urls(&["https://example.com/image"]);
        let bytes = png(10, 10);
        let options = CatalogImageFetchOptions {
            accepted_target: 1,
            thumbnail_max_edge: None,
            ..CatalogImageFetchOptions::default()
        };
        fetch_attached_catalog_images_with(&registry, &id, &options, {
            let bytes = bytes.clone();
            move |_url, _options| {
                let bytes = bytes.clone();
                async move { Ok(bytes) }
            }
        })
        .await
        .expect("initial fetch");
        let cache_path = {
            let catalog = registry.open_attached(&id).expect("open");
            let record = catalog.page_records(0, 1).expect("page").remove(0);
            catalog
                .root()
                .join(acquisition_state(&record).cache_path.expect("cache path"))
        };
        std::fs::write(&cache_path, b"tampered").expect("tamper");
        let calls = Arc::new(Mutex::new(0_u64));
        let report = fetch_attached_catalog_images_with(&registry, &id, &options, {
            let bytes = bytes.clone();
            let calls = Arc::clone(&calls);
            move |_url, _options| {
                let bytes = bytes.clone();
                let calls = Arc::clone(&calls);
                async move {
                    *calls.lock().expect("calls") += 1;
                    Ok(bytes)
                }
            }
        })
        .await
        .expect("repair");
        assert_eq!(report.checkpoint.counts.tampered, 1);
        assert_eq!(*calls.lock().expect("calls"), 1);
        assert!(validate_cached_file(&cache_path, &hex_digest(&bytes), bytes.len() as u64).is_ok());
    }

    #[test]
    fn fetch_lease_is_cross_handle_exclusive() {
        let (_root, registry, id) = catalog_fixture();
        let first_catalog = registry.open_attached(&id).expect("first open");
        let second_catalog = registry.open_attached(&id).expect("second open");
        let _first = CatalogFetchLease::acquire(&first_catalog).expect("first lease");
        let error = CatalogFetchLease::acquire(&second_catalog).expect_err("must contend");
        assert!(matches!(error, CatalogImageFetchError::Busy(_)));
    }

    fn walk_regular_files(root: &Path) -> usize {
        let mut count = 0;
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in std::fs::read_dir(directory).expect("read cache tree") {
                let entry = entry.expect("entry");
                let metadata = entry.metadata().expect("metadata");
                if metadata.is_dir() {
                    pending.push(entry.path());
                } else if metadata.is_file() {
                    count += 1;
                }
            }
        }
        count
    }
}
