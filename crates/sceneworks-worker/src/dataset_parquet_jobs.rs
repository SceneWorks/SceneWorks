//! Bounded, resumable materialization of LAION-style URL/caption Parquet shards.
//!
//! The metadata can span billions of rows, so this job deliberately reads only a
//! bounded candidate window, downloads concurrently, and commits successful images
//! to an empty SceneWorks dataset in one API-side transaction.

use super::*;
use crate::downloads::fetch_public_source_url_bytes;
use futures_util::stream::{self, StreamExt};
use image::GenericImageView;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use std::collections::HashSet;
use std::fs::File;
use std::io::Cursor;

const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_IMAGE_EDGE: u32 = 16_384;
const MAX_IMAGE_DECODE_ALLOC: u64 = 256 * 1024 * 1024;
const MAX_CAPTION_CHARS: usize = 2_048;
const MAX_SCAN_CANDIDATES: usize = 500_000;
const CANDIDATE_MULTIPLIER: usize = 10;
const CANCEL_MESSAGE: &str = "Parquet dataset import canceled.";

#[derive(Clone, Debug)]
struct Candidate {
    url: String,
    caption: String,
    item_id: String,
}

#[derive(Debug)]
struct MaterializedItem {
    id: String,
    path: PathBuf,
    display_name: String,
    caption: String,
    width: u32,
    height: u32,
}

pub(crate) async fn run_dataset_parquet_import_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let project_id = required_payload_string(&job.payload, "projectId")?.to_owned();
    let dataset_id = required_payload_string(&job.payload, "datasetId")?.to_owned();
    let project_root = PathBuf::from(required_payload_string(&job.payload, "projectRoot")?);
    let dataset_root = PathBuf::from(required_payload_string(&job.payload, "datasetRoot")?);
    let source_path = PathBuf::from(required_payload_string(&job.payload, "sourcePath")?);
    let url_column = required_payload_string(&job.payload, "urlColumn")?.to_owned();
    let caption_column = required_payload_string(&job.payload, "captionColumn")?.to_owned();
    let dataset_version = payload_u32(&job.payload, "datasetVersion", 0);
    let max_items = payload_usize(&job.payload, "maxItems", 1_000).clamp(1, 25_000);
    let min_edge = payload_u32(&job.payload, "minEdge", 512).min(8_192);
    let concurrency = payload_usize(&job.payload, "concurrency", 16).clamp(1, 64);
    let caption_includes = normalized_filter_terms(&job.payload, "captionIncludes");
    let caption_excludes = normalized_filter_terms(&job.payload, "captionExcludes");

    if !project_root.is_absolute() || !dataset_root.is_absolute() || !source_path.is_absolute() {
        return Err(WorkerError::InvalidPayload(
            "Parquet import paths must be absolute.".to_owned(),
        ));
    }
    let canonical_project_root = std::fs::canonicalize(&project_root)?;
    let canonical_dataset_root = std::fs::canonicalize(&dataset_root)?;
    if !canonical_dataset_root.starts_with(&canonical_project_root) {
        return Err(WorkerError::InvalidPayload(
            "Parquet dataset root is outside its project.".to_owned(),
        ));
    }
    if !job.id.starts_with("job_")
        || !job
            .id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return Err(WorkerError::InvalidPayload(
            "Parquet import job id is invalid.".to_owned(),
        ));
    }
    let staging_root = canonical_dataset_root.join("parquet-imports").join(&job.id);
    tokio::fs::create_dir_all(&staging_root).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.02,
            "Scanning Parquet metadata.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let scan_limit = max_items
        .saturating_mul(CANDIDATE_MULTIPLIER)
        .max(max_items)
        .min(MAX_SCAN_CANDIDATES);
    let scan_source = source_path.clone();
    let scan_url_column = url_column.clone();
    let scan_caption_column = caption_column.clone();
    let scan_caption_includes = caption_includes.clone();
    let scan_caption_excludes = caption_excludes.clone();
    let scan_task = tokio::task::spawn_blocking(move || {
        read_candidates(
            &scan_source,
            &scan_url_column,
            &scan_caption_column,
            scan_limit,
            min_edge,
            &scan_caption_includes,
            &scan_caption_excludes,
        )
    });
    let candidates =
        heartbeat_while_blocking(api, settings, &job.id, "Parquet metadata scan", scan_task)
            .await?;
    if candidates.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "No usable URL/caption rows were found in columns {url_column:?} and {caption_column:?}."
        )));
    }

    let mut materialized = Vec::with_capacity(max_items.min(candidates.len()));
    let mut failed = 0usize;
    // One network wave per batch keeps progress/cancel checkpoints bounded by
    // the download read timeout rather than several serial waves.
    let batch_size = concurrency;
    for candidate_batch in candidates.chunks(batch_size) {
        check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
        let remaining = max_items.saturating_sub(materialized.len());
        if remaining == 0 {
            break;
        }
        let attempts: Vec<Candidate> = candidate_batch
            .iter()
            .take(remaining.saturating_add(concurrency))
            .cloned()
            .collect();
        let root = staging_root.clone();
        let mut results = stream::iter(attempts.into_iter().map(|candidate| {
            let root = root.clone();
            async move { materialize_candidate(candidate, &root, min_edge).await }
        }))
        .buffer_unordered(concurrency);
        while let Some(result) = results.next().await {
            match result {
                Ok(item) if materialized.len() < max_items => materialized.push(item),
                Ok(_) => {}
                Err(error) => {
                    failed = failed.saturating_add(1);
                    tracing::debug!(
                        event = "dataset_parquet_item_skipped",
                        jobId = %job.id,
                        error = %error
                    );
                }
            }
        }
        let attempted = materialized.len().saturating_add(failed);
        let progress = 0.08
            + 0.82
                * (materialized.len() as f64 / max_items as f64)
                    .max(attempted as f64 / candidates.len() as f64)
                    .min(1.0);
        let snapshot = update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Downloading,
                ProgressStage::Downloading,
                progress,
                &format!(
                    "Imported {} of {} requested images ({} skipped).",
                    materialized.len(),
                    max_items,
                    failed
                ),
                None,
                None,
                None,
            ),
        )
        .await?;
        if snapshot.cancel_requested {
            mark_job_canceled(api, &job.id, CANCEL_MESSAGE).await?;
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
    }
    if materialized.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "None of the {} candidate image URLs produced a decodable image at least {min_edge}px on each edge.",
            candidates.len()
        )));
    }

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Importing,
            0.94,
            "Committing imported images to the training dataset.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let items = materialized
        .iter()
        .map(|item| {
            let relative = item
                .path
                .strip_prefix(&canonical_project_root)
                .map_err(|_| {
                    WorkerError::InvalidPayload(
                        "Parquet import output is outside its project.".to_owned(),
                    )
                })?;
            Ok(json!({
                "id": item.id,
                "path": relative.to_string_lossy().replace('\\', "/"),
                "displayName": item.display_name,
                "caption": {
                    "text": item.caption,
                    "source": "imported",
                    "triggerWords": [],
                },
                "width": item.width,
                "height": item.height,
            }))
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    let committed: Value = api
        .post_json(
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/parquet-import-finalize"
            ),
            &json!({
                "jobId": job.id,
                "datasetVersion": dataset_version,
                "items": items,
            }),
        )
        .await?;

    if let Err(error) = tokio::fs::remove_dir_all(&staging_root).await {
        tracing::warn!(
            event = "dataset_parquet_staging_cleanup_failed",
            jobId = %job.id,
            path = %staging_root.display(),
            error = %error
        );
    }
    let mut result = JsonObject::new();
    result.insert("datasetId".to_owned(), json!(dataset_id));
    result.insert("sourcePath".to_owned(), json!(source_path));
    result.insert("importedItemCount".to_owned(), json!(materialized.len()));
    result.insert("skippedItemCount".to_owned(), json!(failed));
    result.insert("scannedCandidateCount".to_owned(), json!(candidates.len()));
    result.insert("captionIncludes".to_owned(), json!(caption_includes));
    result.insert("captionExcludes".to_owned(), json!(caption_excludes));
    result.insert(
        "datasetVersion".to_owned(),
        committed.get("version").cloned().unwrap_or(Value::Null),
    );
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!(
                "Imported {} images from Parquet metadata.",
                materialized.len()
            ),
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

fn payload_usize(payload: &JsonObject, field: &str, default: usize) -> usize {
    payload
        .get(field)
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        })
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
}

fn normalized_filter_terms(payload: &JsonObject, field: &str) -> Vec<String> {
    payload_string_array(payload, field)
        .into_iter()
        .map(|term| term.trim().to_lowercase())
        .filter(|term| !term.is_empty())
        .collect()
}

fn parquet_files(source: &Path) -> WorkerResult<Vec<PathBuf>> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    let mut files = std::fs::read_dir(source)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case("parquet"))
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Parquet source directory contains no .parquet files.".to_owned(),
        ));
    }
    Ok(files)
}

fn read_candidates(
    source: &Path,
    url_column: &str,
    caption_column: &str,
    limit: usize,
    min_edge: u32,
    caption_includes: &[String],
    caption_excludes: &[String],
) -> WorkerResult<Vec<Candidate>> {
    let files = parquet_files(source)?;
    let mut candidates = Vec::with_capacity(limit.min(16_384));
    let mut seen_urls = HashSet::new();
    let mut observed_columns = Vec::new();
    for path in files {
        let reader = SerializedFileReader::new(File::open(&path)?).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Could not read Parquet shard {}: {error}",
                path.display()
            ))
        })?;
        let rows = reader.get_row_iter(None).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Could not scan Parquet shard {}: {error}",
                path.display()
            ))
        })?;
        for row in rows {
            let row = row.map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "Could not decode a row in {}: {error}",
                    path.display()
                ))
            })?;
            if observed_columns.is_empty() {
                observed_columns.extend(
                    row.get_column_iter()
                        .map(|(name, _)| name.to_owned())
                        .take(32),
                );
            }
            let url = row_string(&row, url_column);
            let caption = row_string(&row, caption_column);
            let (Some(url), Some(caption)) = (url, caption) else {
                continue;
            };
            let url = url.trim().replace("&amp;", "&");
            let caption = caption.trim();
            let caption_lower = caption.to_lowercase();
            let width = row_u32(&row, "WIDTH");
            let height = row_u32(&row, "HEIGHT");
            if url.is_empty()
                || caption.is_empty()
                || (!caption_includes.is_empty()
                    && !caption_includes
                        .iter()
                        .any(|term| caption_lower.contains(term)))
                || caption_excludes
                    .iter()
                    .any(|term| caption_lower.contains(term))
                || width.zip(height).is_some_and(|(width, height)| {
                    width < min_edge
                        || height < min_edge
                        || width > MAX_IMAGE_EDGE
                        || height > MAX_IMAGE_EDGE
                })
                || !matches!(
                    reqwest::Url::parse(&url)
                        .ok()
                        .as_ref()
                        .map(reqwest::Url::scheme),
                    Some("http" | "https")
                )
                || !seen_urls.insert(url.clone())
            {
                continue;
            }
            let digest = Sha256::digest(url.as_bytes());
            candidates.push(Candidate {
                url,
                caption: caption.chars().take(MAX_CAPTION_CHARS).collect(),
                item_id: format!("pq_{}", hex_prefix(&digest, 20)),
            });
            if candidates.len() >= limit {
                return Ok(candidates);
            }
        }
    }
    if candidates.is_empty() && !observed_columns.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "Parquet columns {url_column:?}/{caption_column:?} were not found or contained no usable rows. Available columns include: {}.",
            observed_columns.join(", ")
        )));
    }
    Ok(candidates)
}

fn row_u32(row: &parquet::record::Row, requested: &str) -> Option<u32> {
    row.get_column_iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(requested))
        .and_then(|(_, field)| match field {
            Field::Byte(value) => u32::try_from(*value).ok(),
            Field::Short(value) => u32::try_from(*value).ok(),
            Field::Int(value) => u32::try_from(*value).ok(),
            Field::Long(value) => u32::try_from(*value).ok(),
            Field::UByte(value) => Some((*value).into()),
            Field::UShort(value) => Some((*value).into()),
            Field::UInt(value) => Some(*value),
            Field::ULong(value) => u32::try_from(*value).ok(),
            _ => None,
        })
}

fn row_string(row: &parquet::record::Row, requested: &str) -> Option<String> {
    row.get_column_iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(requested))
        .and_then(|(_, field)| match field {
            Field::Str(value) => Some(value.clone()),
            Field::Bytes(value) => std::str::from_utf8(value.data()).ok().map(str::to_owned),
            _ => None,
        })
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

async fn materialize_candidate(
    candidate: Candidate,
    staging_root: &Path,
    min_edge: u32,
) -> WorkerResult<MaterializedItem> {
    let target = staging_root.join(format!("{}.png", candidate.item_id));
    if target.is_file() {
        if let Ok((width, height)) = image::image_dimensions(&target) {
            if width >= min_edge && height >= min_edge {
                return Ok(materialized_item(candidate, target, width, height));
            }
        }
    }
    let bytes = fetch_public_source_url_bytes(&candidate.url, MAX_IMAGE_BYTES).await?;
    let target_for_write = target.clone();
    let dimensions = tokio::task::spawn_blocking(move || -> WorkerResult<(u32, u32)> {
        let mut reader = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("dataset image format failed: {error}"))
            })?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_IMAGE_EDGE);
        limits.max_image_height = Some(MAX_IMAGE_EDGE);
        limits.max_alloc = Some(MAX_IMAGE_DECODE_ALLOC);
        reader.limits(limits);
        let image = reader.decode().map_err(|error| {
            WorkerError::InvalidPayload(format!("dataset image decode failed: {error}"))
        })?;
        let (width, height) = image.dimensions();
        if width < min_edge || height < min_edge {
            return Err(WorkerError::InvalidPayload(format!(
                "dataset image is {width}x{height}, below the {min_edge}px minimum edge"
            )));
        }
        let temporary = target_for_write.with_extension("png.partial");
        image
            .save_with_format(&temporary, image::ImageFormat::Png)
            .map_err(|error| {
                WorkerError::Io(std::io::Error::other(format!(
                    "dataset image encode failed: {error}"
                )))
            })?;
        if target_for_write.exists() {
            std::fs::remove_file(&target_for_write)?;
        }
        std::fs::rename(&temporary, &target_for_write)?;
        Ok((width, height))
    })
    .await
    .map_err(|error| task_join_error("Parquet image decode", error))??;
    Ok(materialized_item(
        candidate,
        target,
        dimensions.0,
        dimensions.1,
    ))
}

fn materialized_item(
    candidate: Candidate,
    path: PathBuf,
    width: u32,
    height: u32,
) -> MaterializedItem {
    MaterializedItem {
        display_name: format!("{}.png", candidate.item_id),
        id: candidate.item_id,
        path,
        caption: candidate.caption,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::basic::Compression;
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use std::sync::Arc;

    #[test]
    fn deterministic_item_id_hex_prefix_has_requested_length() {
        let digest = Sha256::digest(b"https://example.com/image.jpg");
        let prefix = hex_prefix(&digest, 20);
        assert_eq!(prefix.len(), 20);
        assert!(prefix.chars().all(|value| value.is_ascii_hexdigit()));
    }

    #[test]
    fn reads_snappy_laion_url_and_text_columns_case_insensitively() {
        let temp = tempfile::tempdir().expect("temp directory");
        let path = temp.path().join("part-00000.snappy.parquet");
        let schema = Arc::new(
            parse_message_type(
                "message laion {
                    REQUIRED BINARY URL (UTF8);
                    REQUIRED BINARY TEXT (UTF8);
                }",
            )
            .expect("test schema"),
        );
        let properties = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build(),
        );
        let file = File::create(&path).expect("test parquet creates");
        let mut writer =
            SerializedFileWriter::new(file, schema, properties).expect("writer creates");
        let mut row_group = writer.next_row_group().expect("row group creates");
        for values in [
            [
                "https://example.com/one.jpg?a=1&amp;b=2",
                "https://example.com/logo.png",
            ],
            ["a cinematic portrait", "a portrait company logo"],
        ] {
            let mut column = row_group
                .next_column()
                .expect("column advances")
                .expect("column exists");
            let values = values.into_iter().map(ByteArray::from).collect::<Vec<_>>();
            column
                .typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .expect("column writes");
            column.close().expect("column closes");
        }
        row_group.close().expect("row group closes");
        writer.close().expect("writer closes");

        let candidates = read_candidates(
            temp.path(),
            "url",
            "text",
            10,
            0,
            &["portrait".to_owned()],
            &["logo".to_owned()],
        )
        .expect("rows read");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].url, "https://example.com/one.jpg?a=1&b=2");
        assert_eq!(candidates[0].caption, "a cinematic portrait");
        assert!(candidates[0].item_id.starts_with("pq_"));
    }
}
