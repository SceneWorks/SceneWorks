//! Native subject-mask generation for training datasets (epic 2123, sc-2126).
//!
//! The `dataset_subject_mask` job runs the shipped SAM3 text-concept ("person") segmenter over every
//! image of a training dataset and POSTs one single-channel PNG mask per image to rust-api's
//! content-hash-keyed mask sidecar (`.../subject-masks`, stored by
//! `sceneworks_core::training_subject_masks`). The SAM3 model is built (and quantized) ONCE per job;
//! each image is then segmented on its own (a one-frame PCS pass — dataset images are unrelated
//! stills, so there is no track to carry between them) and every detected person is unioned into the
//! subject: white = subject, black = background, at the image's own dimensions. Masks are POSTed in
//! bounded chunks ([`SUBJECT_MASK_POST_CHUNK_BYTES`]) so a large dataset never exceeds the route's
//! body limit after all the GPU work is done.
//!
//! The correctness linchpin mirrors the face pass: an image where SAM3 finds **no person** still gets
//! a mask — an all-black one — so "examined, nothing found" is a stored, flaggable state rather than
//! an absent record that reads as "not processed". The store records it as `empty`.
//!
//! MLX on macOS (`person_segment_sam3`), candle off-Mac (`person_segment_sam3_candle`); on a platform
//! with neither, a precise unsupported error. The segmenter is injected into
//! [`generate_subject_masks`] so the per-image loop, union and encoding are unit-tested without SAM3
//! weights.

use super::*;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::CancelFlag;

/// The segmentation space stamped on the POSTed masks; rust-api refuses any other value.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SUBJECT_MASK_SPACE: &str = "sam3-person";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CANCEL_MESSAGE: &str = "Subject mask generation canceled by user.";
/// Largest JSON-encoded mask payload per sidecar POST. Masks go up in chunks of at most this many
/// bytes (rust-api's `/subject-masks` route accepts up to 256 MiB per body), so a dataset of any
/// size never 413s after the GPU work. One mask is at most 32 MiB raw (~43 MiB base64), so a single
/// mask always fits a chunk. The store treats each POST as an incremental write.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SUBJECT_MASK_POST_CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct SubjectMaskItem {
    image_path: PathBuf,
    content_hash: String,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct SubjectMaskRecord {
    content_hash: String,
    /// Single-channel PNG at the image's dimensions.
    png: Vec<u8>,
    /// True when no person was detected (every pixel 0).
    empty: bool,
}

/// Union every detected person's binary mask (row-major `width*height`, 0/255) into one subject
/// mask. No masks → an all-black mask. A mask of the wrong length is an engine error, never padded.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn union_person_masks(masks: &[Vec<u8>], width: u32, height: u32) -> WorkerResult<Vec<u8>> {
    let len = width as usize * height as usize;
    let mut subject = vec![0u8; len];
    for mask in masks {
        if mask.len() != len {
            return Err(WorkerError::Engine(format!(
                "subject mask: person mask has {} values, expected {width}x{height}",
                mask.len()
            )));
        }
        for (out, &value) in subject.iter_mut().zip(mask) {
            *out = (*out).max(value);
        }
    }
    Ok(subject)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn encode_mask_png(mask: Vec<u8>, width: u32, height: u32) -> WorkerResult<Vec<u8>> {
    let image = image::GrayImage::from_raw(width, height, mask)
        .ok_or_else(|| WorkerError::Engine("subject mask buffer size mismatch".to_owned()))?;
    let mut out = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|error| WorkerError::Engine(format!("subject mask encode: {error}")))?;
    Ok(out)
}

/// Decodes dataset image `index` for the segmenter (see [`generate_subject_masks`]).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
type ImageLoader = Box<dyn FnMut(usize) -> WorkerResult<image::RgbImage> + Send>;

/// Turns image `index` and every person SAM3 found on it into its stored record (see
/// [`generate_subject_masks`]).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
type MaskFinisher =
    Box<dyn FnMut(usize, &image::RgbImage, Vec<Vec<u8>>) -> WorkerResult<SubjectMaskRecord> + Send>;

/// Segment every item with ONE call to `segment_batch(count, load, finish)` — the backend's
/// `segment_persons_per_image`, which builds the SAM3 session once and, per image, calls `load`,
/// runs a one-frame "person" propagate and hands the detected masks to `finish`. Here `load` decodes
/// the item and `finish` unions every person, encodes the PNG and reports the item's index on `tx`.
/// The segmenter is injected so the loop is testable without weights.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn generate_subject_masks<B>(
    items: Vec<SubjectMaskItem>,
    cancel: CancelFlag,
    tx: tokio::sync::mpsc::Sender<usize>,
    segment_batch: B,
) -> WorkerResult<Vec<SubjectMaskRecord>>
where
    B: FnOnce(usize, ImageLoader, MaskFinisher) -> WorkerResult<Vec<SubjectMaskRecord>>,
{
    if cancel.is_cancelled() {
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    let count = items.len();
    let paths: Vec<PathBuf> = items.iter().map(|item| item.image_path.clone()).collect();
    let hashes: Vec<String> = items.into_iter().map(|item| item.content_hash).collect();
    let load: ImageLoader = Box::new(move |index| {
        let path = &paths[index];
        Ok(crate::image_decode::decode_image_any(path)
            .map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "subject mask image {}: {error}",
                    path.display()
                ))
            })?
            .to_rgb8())
    });
    let finish: MaskFinisher = Box::new(move |index, image, persons| {
        let (width, height) = image.dimensions();
        let subject = union_person_masks(&persons, width, height)?;
        let empty = subject.iter().all(|&value| value == 0);
        let record = SubjectMaskRecord {
            content_hash: hashes[index].clone(),
            png: encode_mask_png(subject, width, height)?,
            empty,
        };
        // A closed channel means the consumer loop returned early (POST failure / 409): trip the
        // flag so the batch bails at its next per-image check instead of running unheard (sc-8804,
        // F-003).
        if tx.blocking_send(index).is_err() {
            cancel.cancel();
        }
        Ok(record)
    });
    segment_batch(count, load, finish)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_dataset_subject_mask_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let items = subject_mask_items(settings, &job.payload)?;
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Subject mask job has no images to mask.".to_owned(),
        ));
    }
    let backend = backend_label(&settings.gpu_id);
    let total = items.len();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.04,
            "Preparing subject mask job.",
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    // Resolve-only (sc-17629): the SAM3 Model Manager install, or an actionable install error.
    let (model_path, tokenizer_path) =
        crate::person_segment_sam3_common::require_segmenter_weights(settings)?;
    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.08,
            "Loading SAM3 person segmenter.",
            None,
            backend,
        ),
    )
    .await?;

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<usize>(64);
    let blocking = spawn_subject_mask_segmentation(
        model_path,
        tokenizer_path,
        items,
        cancel.clone(),
        tx,
        job.id.clone(),
    );

    let dataset_id = required_payload_string(&job.payload, "datasetId")?.to_owned();
    let cfg = AnalysisJobConfig {
        endpoint_suffix: "subject-masks",
        space: SUBJECT_MASK_SPACE,
        cancel_message: CANCEL_MESSAGE,
        saving_message: "Saving subject masks.",
        join_error_label: "dataset subject mask task join",
        item_message: &|index, total| format!("Masked image {} of {}.", index + 1, total),
        post_chunk_bytes: Some(SUBJECT_MASK_POST_CHUNK_BYTES),
    };
    run_batched_analysis_job(
        api,
        settings,
        job,
        &cfg,
        total,
        backend,
        cancel,
        rx,
        blocking,
        subject_mask_records_payload,
        |records, responses| {
            let empty = records.iter().filter(|record| record.empty).count();
            let result = subject_mask_result(&dataset_id, records.len(), empty, &responses)?;
            let skipped = result["skippedContentHashes"]
                .as_array()
                .map_or(0, Vec::len);
            let mut message = format!(
                "Masked {} image(s); {empty} with no person detected.",
                records.len()
            );
            if skipped > 0 {
                message.push_str(&format!(
                    " {skipped} skipped: the image left the dataset while masking."
                ));
            }
            Ok(analysis_progress(
                JobStatus::Completed,
                ProgressStage::Completed,
                1.0,
                &message,
                Some(result),
                backend,
            ))
        },
    )
    .await?;
    Ok(())
}

/// Segment `items` with ONE SAM3 session on a blocking thread (the model is built + quantized once,
/// not per image), reporting each finished item's index on `tx`. Shared by the
/// `dataset_subject_mask` job and the training run's subject-mask prepass (sc-2124).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn spawn_subject_mask_segmentation(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    items: Vec<SubjectMaskItem>,
    cancel: CancelFlag,
    tx: tokio::sync::mpsc::Sender<usize>,
    job_id: String,
) -> tokio::task::JoinHandle<WorkerResult<Vec<SubjectMaskRecord>>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<SubjectMaskRecord>> {
        emit_event(
            "dataset_subject_mask_start",
            json!({ "jobId": job_id, "space": SUBJECT_MASK_SPACE }),
        );
        let engine_cancel = cancel.clone();
        let records = generate_subject_masks(items, cancel, tx, |count, load, finish| {
            #[cfg(target_os = "macos")]
            let masks = crate::person_segment_sam3::segment_persons_per_image(
                model_path,
                tokenizer_path,
                count,
                load,
                finish,
                Some(engine_cancel),
            );
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let masks = crate::person_segment_sam3_candle::segment_persons_per_image(
                model_path,
                tokenizer_path,
                count,
                load,
                finish,
                Some(engine_cancel),
            );
            masks
        })?;
        emit_event(
            "dataset_subject_mask_complete",
            json!({ "jobId": job_id, "space": SUBJECT_MASK_SPACE }),
        );
        Ok(records)
    })
}

/// Cancel copy of a training run stopped during its subject-mask prepass.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const PREPASS_CANCEL_MESSAGE: &str = "LoRA training canceled while generating subject masks.";

/// The training run's subject-mask prepass (sc-2124). When the API found dataset images without a
/// subject mask on a run that reads them (subject-masked loss — on by default for the Z-Image
/// character presets — or the subject-restricted normal loss), it stamped their work list on the
/// job as `subjectMaskPrepass` (the `dataset_subject_mask` item shape). This segments exactly those
/// images with SAM3, stores the masks through the same `/subject-masks` sidecar the mask job uses,
/// and returns; the training preflight that runs next re-reads the dataset's masks and still refuses
/// an image left without a usable one (e.g. SAM3 found no person in it), naming it. No prepass key
/// (every image already masked, or no masked technique) is a no-op. Progress stays inside the
/// preparing band, below the 0.05 the training run starts from.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_training_subject_mask_prepass(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let Some(prepass) = job.payload.get("subjectMaskPrepass") else {
        return Ok(());
    };
    let prepass = prepass.as_object().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Training payload.subjectMaskPrepass must be an object.".to_owned(),
        )
    })?;
    let items = subject_mask_items(settings, prepass)?;
    if items.is_empty() {
        return Ok(());
    }
    let project_id = required_payload_string(prepass, "projectId")?;
    let dataset_id = required_payload_string(prepass, "datasetId")?;
    let backend = backend_label(&settings.gpu_id);
    let total = items.len();
    update_job(
        api,
        &job.id,
        crate::training_jobs::training_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.01,
            &format!("Generating subject masks for {total} image(s) with SAM3."),
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, PREPASS_CANCEL_MESSAGE).await?;
    let (model_path, tokenizer_path) =
        crate::person_segment_sam3_common::require_segmenter_weights(settings)?;
    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<usize>(64);
    let blocking = spawn_subject_mask_segmentation(
        model_path,
        tokenizer_path,
        items,
        cancel.clone(),
        tx,
        job.id.clone(),
    );
    let records = stream_batched_records(
        api,
        settings,
        job,
        BatchedStream {
            cancel_message: PREPASS_CANCEL_MESSAGE,
            join_error_label: "training subject mask task join",
        },
        cancel,
        rx,
        blocking,
        |index| {
            crate::training_jobs::training_progress(
                JobStatus::Preparing,
                ProgressStage::Preparing,
                0.01 + 0.03 * ((index + 1) as f64 / total as f64),
                &format!("Generated subject mask {} of {total}.", index + 1),
                None,
                backend,
            )
        },
    )
    .await?;
    post_dataset_sidecar(
        api,
        project_id,
        dataset_id,
        "subject-masks",
        SUBJECT_MASK_SPACE,
        subject_mask_records_payload(&records),
        Some(SUBJECT_MASK_POST_CHUNK_BYTES),
    )
    .await?;
    Ok(())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn subject_mask_records_payload(records: &[SubjectMaskRecord]) -> Vec<Value> {
    use base64::Engine as _;
    records
        .iter()
        .map(|record| {
            json!({
                "contentHash": record.content_hash,
                "maskPng": base64::engine::general_purpose::STANDARD.encode(&record.png),
            })
        })
        .collect()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn subject_mask_result(
    dataset_id: &str,
    masked: usize,
    empty: usize,
    responses: &[Value],
) -> WorkerResult<JsonObject> {
    // Fold the per-chunk sidecar responses: masks stored, and the hashes skipped because their image
    // left the dataset while the job ran. A response without a numeric `stored` is a contract break.
    let mut stored = 0u64;
    let mut skipped = Vec::new();
    for response in responses {
        stored += response
            .get("stored")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                WorkerError::Engine(format!(
                    "subject mask sidecar response has no stored count: {response}"
                ))
            })?;
        if let Some(hashes) = response.get("skipped").and_then(Value::as_array) {
            skipped.extend(hashes.iter().cloned());
        }
    }
    let mut result = JsonObject::new();
    result.insert("space".to_owned(), json!(SUBJECT_MASK_SPACE));
    result.insert("datasetId".to_owned(), json!(dataset_id));
    result.insert("maskedItemCount".to_owned(), json!(masked));
    result.insert("emptyMaskCount".to_owned(), json!(empty));
    result.insert("stored".to_owned(), json!(stored));
    result.insert("skippedContentHashes".to_owned(), Value::Array(skipped));
    Ok(result)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn subject_mask_items(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Vec<SubjectMaskItem>> {
    let dataset_root = payload
        .get("datasetRoot")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Subject mask payload.datasetRoot must be an app-managed dataset path.".to_owned(),
            )
        })?;
    let items = payload
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Subject mask payload.items must be an array.".to_owned())
        })?;
    items
        .iter()
        .map(|item| {
            let object = item.as_object().ok_or_else(|| {
                WorkerError::InvalidPayload("Subject mask item must be an object.".to_owned())
            })?;
            let field = |key: &str| {
                object
                    .get(key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            };
            let item_id = field("itemId").ok_or_else(|| {
                WorkerError::InvalidPayload("Subject mask item is missing itemId.".to_owned())
            })?;
            let content_hash = field("contentHash")
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "Subject mask item {item_id} is missing contentHash."
                    ))
                })?
                .to_owned();
            let image_path = field("imagePath").ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "Subject mask item {item_id} is missing imagePath."
                ))
            })?;
            let image_path = resolve_dataset_item_path(
                settings,
                dataset_root,
                image_path,
                &format!("Subject mask item {item_id} imagePath"),
            )?;
            Ok(SubjectMaskItem {
                image_path,
                content_hash,
            })
        })
        .collect()
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
pub(crate) async fn run_dataset_subject_mask_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Subject mask generation (SAM3) needs the macOS MLX backend or the candle backend \
         (build with --features backend-candle)."
            .to_owned(),
    ))
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod tests;
