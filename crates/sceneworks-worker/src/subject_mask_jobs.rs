//! Native subject-mask generation for training datasets (epic 2123, sc-2126).
//!
//! The `dataset_subject_mask` job runs the shipped SAM3 text-concept ("person") segmenter over every
//! image of a training dataset and POSTs one single-channel PNG mask per image to rust-api's
//! content-hash-keyed mask sidecar (`.../subject-masks`, stored by
//! `sceneworks_core::training_subject_masks`). Each image is segmented on its own (a one-frame PCS
//! pass — dataset images are unrelated stills, so there is no track to carry between them) and every
//! detected person is unioned into the subject: white = subject, black = background, at the image's
//! own dimensions.
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

/// Segment every item with `segment` (image → every detected person's binary mask at the image's
/// dimensions), union, encode, and report each finished item's index on `tx`. The SAM3 segmenter is
/// injected so the loop is testable without weights.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn generate_subject_masks<S>(
    items: Vec<SubjectMaskItem>,
    cancel: CancelFlag,
    tx: tokio::sync::mpsc::Sender<usize>,
    mut segment: S,
) -> WorkerResult<Vec<SubjectMaskRecord>>
where
    S: FnMut(&image::RgbImage) -> WorkerResult<Vec<Vec<u8>>>,
{
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        let image = crate::image_decode::decode_image_any(&item.image_path)
            .map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "subject mask image {}: {error}",
                    item.image_path.display()
                ))
            })?
            .to_rgb8();
        let (width, height) = image.dimensions();
        let persons = segment(&image)?;
        let subject = union_person_masks(&persons, width, height)?;
        let empty = subject.iter().all(|&value| value == 0);
        out.push(SubjectMaskRecord {
            content_hash: item.content_hash,
            png: encode_mask_png(subject, width, height)?,
            empty,
        });
        // A closed channel means the consumer loop returned early (POST failure / 409): trip the
        // flag so the loop bails instead of running unheard (sc-8804, F-003).
        if tx.blocking_send(index).is_err() {
            cancel.cancel();
        }
    }
    Ok(out)
}

/// SAM3 "person" segmentation of one still: a one-frame PCS pass, returning every detected person's
/// mask at the image's dimensions (empty when nobody is found).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn sam3_person_masks(
    model_path: &Path,
    tokenizer_path: &Path,
    image: &image::RgbImage,
    cancel: &CancelFlag,
) -> WorkerResult<Vec<Vec<u8>>> {
    let frame = gen_core::Image {
        width: image.width(),
        height: image.height(),
        pixels: image.as_raw().clone(),
    };
    #[cfg(target_os = "macos")]
    let all = crate::person_segment_sam3::segment_all_persons_in_memory(
        model_path,
        tokenizer_path,
        std::slice::from_ref(&frame),
        Some(cancel.clone()),
        None,
    )?;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let all = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
        model_path,
        tokenizer_path,
        std::slice::from_ref(&frame),
        Some(cancel.clone()),
        None,
    )?;
    Ok(all
        .per_frame
        .into_iter()
        .next()
        .unwrap_or_default()
        .into_iter()
        .map(|(_, mask)| mask)
        .collect())
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
    let blocking_cancel = cancel.clone();
    let job_id = job.id.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<Vec<SubjectMaskRecord>> {
        emit_event(
            "dataset_subject_mask_start",
            json!({ "jobId": job_id, "space": SUBJECT_MASK_SPACE }),
        );
        let engine_cancel = blocking_cancel.clone();
        let records = generate_subject_masks(items, blocking_cancel, tx, |image| {
            sam3_person_masks(&model_path, &tokenizer_path, image, &engine_cancel)
        })?;
        emit_event(
            "dataset_subject_mask_complete",
            json!({ "jobId": job_id, "space": SUBJECT_MASK_SPACE }),
        );
        Ok(records)
    });

    let dataset_id = required_payload_string(&job.payload, "datasetId")?.to_owned();
    let cfg = AnalysisJobConfig {
        endpoint_suffix: "subject-masks",
        space: SUBJECT_MASK_SPACE,
        cancel_message: CANCEL_MESSAGE,
        saving_message: "Saving subject masks.",
        join_error_label: "dataset subject mask task join",
        item_message: &|index, total| format!("Masked image {} of {}.", index + 1, total),
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
        |records, stored| {
            let empty = records.iter().filter(|record| record.empty).count();
            analysis_progress(
                JobStatus::Completed,
                ProgressStage::Completed,
                1.0,
                &format!(
                    "Masked {} image(s); {empty} with no person detected.",
                    records.len()
                ),
                Some(subject_mask_result(
                    &dataset_id,
                    records.len(),
                    empty,
                    stored,
                )),
                backend,
            )
        },
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
fn subject_mask_result(dataset_id: &str, masked: usize, empty: usize, stored: Value) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("space".to_owned(), json!(SUBJECT_MASK_SPACE));
    result.insert("datasetId".to_owned(), json!(dataset_id));
    result.insert("maskedItemCount".to_owned(), json!(masked));
    result.insert("emptyMaskCount".to_owned(), json!(empty));
    result.insert(
        "stored".to_owned(),
        stored.get("stored").cloned().unwrap_or(Value::Null),
    );
    result
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
