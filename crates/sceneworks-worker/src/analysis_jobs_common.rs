//! Shared dataset parsing, analysis progress and sidecar scaffold.
use super::*;

#[cfg(native_lane)]
use gen_core::CancelFlag;

/// Strict analysis jobs reject malformed items and normalize text. Upscale historically skips
/// malformed entries and preserves their raw strings; keep that supported contract explicit.
pub(crate) enum DatasetItemPolicy {
    #[cfg(native_lane)]
    Strict,
    SkipMalformed,
}

pub(crate) struct DatasetItem<'a> {
    pub object: &'a JsonObject,
    pub item_id: &'a str,
    settings: &'a Settings,
    dataset_root: &'a str,
    label: &'a str,
    normalize: bool,
}

impl DatasetItem<'_> {
    #[cfg(native_lane)]
    pub(crate) fn required(&self, field: &str) -> WorkerResult<&str> {
        payload::required_trimmed_str(
            self.object,
            field,
            &format!("{} item {} is missing {field}.", self.label, self.item_id),
        )
    }

    pub(crate) fn image_path(&self) -> WorkerResult<PathBuf> {
        let path = if self.normalize {
            payload::required_trimmed_str(
                self.object,
                "imagePath",
                &format!("{} item {} is missing imagePath.", self.label, self.item_id),
            )?
        } else {
            self.object
                .get("imagePath")
                .and_then(Value::as_str)
                .expect("validated upscale path")
        };
        resolve_dataset_item_path(
            self.settings,
            self.dataset_root,
            path,
            &format!("{} item {} imagePath", self.label, self.item_id),
        )
    }
}

/// Single owner for datasetRoot/items/object/itemId validation and iteration. The mapper retains
/// each lane's additional fields and validation order, including content hashes and captions.
pub(crate) fn parse_dataset_items<T>(
    settings: &Settings,
    payload: &JsonObject,
    label: &str,
    policy: DatasetItemPolicy,
    mut parse: impl FnMut(DatasetItem<'_>) -> WorkerResult<T>,
) -> WorkerResult<Vec<T>> {
    let root = payload::required_trimmed_str(
        payload,
        "datasetRoot",
        &format!("{label} payload.datasetRoot must be an app-managed dataset path."),
    )?;
    let normalize = !matches!(policy, DatasetItemPolicy::SkipMalformed);
    let items = match payload.get("items").and_then(Value::as_array) {
        Some(items) => items,
        None if !normalize => return Ok(Vec::new()),
        None => {
            return Err(WorkerError::InvalidPayload(format!(
                "{label} payload.items must be an array."
            )))
        }
    };
    dataset_items(items, label, normalize)
        .map(|entry| {
            let (object, item_id) = entry?;
            parse(DatasetItem {
                object,
                item_id,
                settings,
                dataset_root: root,
                label,
                normalize,
            })
        })
        .collect()
}

fn dataset_items<'a>(
    items: &'a [Value],
    label: &'a str,
    normalize: bool,
) -> impl Iterator<Item = WorkerResult<(&'a JsonObject, &'a str)>> {
    items.iter().filter_map(move |item| {
        let Some(object) = item.as_object() else {
            return normalize.then(|| {
                Err(WorkerError::InvalidPayload(format!(
                    "{label} item must be an object."
                )))
            });
        };
        if normalize {
            Some(
                payload::required_trimmed_str(
                    object,
                    "itemId",
                    &format!("{label} item is missing itemId."),
                )
                .map(|id| (object, id)),
            )
        } else {
            let id = object.get("itemId").and_then(Value::as_str)?;
            let path = object.get("imagePath").and_then(Value::as_str)?;
            (!id.is_empty() && !path.is_empty()).then_some(Ok((object, id)))
        }
    })
}

/// Per-item progress scaling shared by every batched analysis job: item `index` (0-based) of `total`
/// maps to `0.12 + 0.78 * ((index + 1) / total)`, leaving `0.04/0.08` for prepare/load and `0.94/1.0`
/// for save/complete. Extracted so the one magic ramp is defined (and unit-tested) once (sc-8836).
#[cfg(native_lane)]
pub(crate) fn item_progress(index: usize, total: usize) -> f64 {
    0.12 + 0.78 * ((index + 1) as f64 / total.max(1) as f64)
}

/// Build a batched-analysis-job [`ProgressRequest`] with the shared field defaults. Every stage of both
/// analysis jobs stamped these identical constructors (`analysis_progress` / `face_progress`); this is
/// the one copy (sc-8836).
#[cfg(native_lane)]
pub(crate) fn analysis_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    progress_payload_for_backend(status, stage, progress, message, result, backend)
}

/// The per-run knobs the shared [`run_batched_analysis_job`] scaffold needs: the sidecar endpoint +
/// embedding space, the cancel message, the saving-stage message, and the per-item progress-message
/// builder. Everything that differs between the CLIP and face analysis jobs (beyond the item shape,
/// the model loading, and the record fold) is expressed here.
#[cfg(native_lane)]
pub(crate) struct AnalysisJobConfig<'a> {
    /// The `.../training/datasets/{dataset_id}/<endpoint_suffix>` sidecar path segment
    /// (`analysis-embeddings` for CLIP, `face-embeddings` for the face stack).
    pub endpoint_suffix: &'a str,
    /// The stable embedding-space tag POSTed alongside the records (guards the sidecar ingest merge).
    pub space: &'a str,
    /// The user-facing cancel message both the in-loop poll and the blocking task raise.
    pub cancel_message: &'a str,
    /// Saving-stage status message (e.g. "Saving embeddings.").
    pub saving_message: &'a str,
    /// The `task_join_error` label if the blocking analysis task panics — kept per-run so each job
    /// preserves its original distinct diagnostic string (`"dataset analysis task join"` for CLIP,
    /// `"dataset face analysis task join"` for the face stack).
    pub join_error_label: &'a str,
    /// Progress-line message for item `index` (0-based) of `total` (e.g. "Analyzed image 3 of 10.").
    /// `Send + Sync` so the enclosing job future stays `Send` (rust-api `tokio::spawn`s the loop).
    pub item_message: &'a (dyn Fn(usize, usize) -> String + Send + Sync),
    /// `Some(max)` POSTs the records to the sidecar in several requests, each holding items whose
    /// JSON encoding totals at most `max` bytes (an item larger than `max` travels alone), so a
    /// large batch of bulky records (e.g. base64 subject-mask PNGs) never exceeds the route's body
    /// limit after all the GPU work is done. The sidecar must therefore treat each POST as an
    /// incremental write. `None` sends every record in one POST.
    pub post_chunk_bytes: Option<usize>,
}

/// Split `items` into consecutive POST chunks whose JSON-encoded items total at most `max_bytes`
/// (see [`AnalysisJobConfig::post_chunk_bytes`]); `None` → one chunk. Always at least one chunk,
/// so an empty batch still makes its (empty) POST.
#[cfg(native_lane)]
pub(crate) fn chunk_post_items(items: Vec<Value>, max_bytes: Option<usize>) -> Vec<Vec<Value>> {
    let Some(max_bytes) = max_bytes else {
        return vec![items];
    };
    let mut chunks = Vec::new();
    let mut chunk = Vec::new();
    let mut chunk_bytes = 0usize;
    for item in items {
        // `,` separator included: a chunk's items serialize to at most `max_bytes` as a JSON array body.
        let bytes = item.to_string().len() + 1;
        if !chunk.is_empty() && chunk_bytes + bytes > max_bytes {
            chunks.push(std::mem::take(&mut chunk));
            chunk_bytes = 0;
        }
        chunk_bytes += bytes;
        chunk.push(item);
    }
    if !chunk.is_empty() || chunks.is_empty() {
        chunks.push(chunk);
    }
    chunks
}

/// Drive one batched dataset-analysis job to completion once its records-producing blocking task is
/// spawned. Owns the shared scaffold (sc-8836): binds `blocking` to `cancel` through [`StreamSupervisor`],
/// runs the `select!` stream loop that scales per-item progress into `0.12..0.90` and polls cancel on
/// each heartbeat tick, joins the records on clean exit, POSTs them (folded through `records_payload`)
/// to the sidecar (in [`AnalysisJobConfig::post_chunk_bytes`] chunks when set), and emits the completed
/// update built by `completed` from the records and every sidecar response, in POST order. The caller has already parsed
/// items, emitted the preparing/loading progress, created the `(tx, rx)` pair, and spawned `blocking`
/// (whose per-item `tx.blocking_send(index)` drives `rx`).
///
/// The per-item `index` a producer sends (0-based) is reported through [`item_progress`]. Returns the
/// joined records so the caller can derive its completed-result fields (e.g. a with-face count).
#[cfg(native_lane)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_batched_analysis_job<R, P, C>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    cfg: &AnalysisJobConfig<'_>,
    total: usize,
    backend: &str,
    cancel: CancelFlag,
    rx: tokio::sync::mpsc::Receiver<usize>,
    blocking: tokio::task::JoinHandle<WorkerResult<Vec<R>>>,
    records_payload: P,
    completed: C,
) -> WorkerResult<Vec<R>>
where
    P: FnOnce(&[R]) -> Vec<Value>,
    C: FnOnce(&[R], Vec<Value>) -> WorkerResult<ProgressRequest>,
{
    let records = stream_batched_records(
        api,
        settings,
        job,
        BatchedStream {
            cancel_message: cfg.cancel_message,
            join_error_label: cfg.join_error_label,
        },
        cancel,
        rx,
        blocking,
        |index| {
            analysis_progress(
                JobStatus::Running,
                ProgressStage::Running,
                item_progress(index, total),
                &(cfg.item_message)(index, total),
                None,
                backend,
            )
        },
    )
    .await?;
    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.94,
            cfg.saving_message,
            None,
            backend,
        ),
    )
    .await?;
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let dataset_id = required_payload_string(&job.payload, "datasetId")?;
    let responses = post_dataset_sidecar(
        api,
        project_id,
        dataset_id,
        cfg.endpoint_suffix,
        cfg.space,
        records_payload(&records),
        cfg.post_chunk_bytes,
    )
    .await?;
    update_job(api, &job.id, completed(&records, responses)?).await?;
    Ok(records)
}

/// POST `items` to the dataset sidecar `.../training/datasets/{dataset_id}/{endpoint_suffix}` as
/// `{ space, items }` bodies — in [`AnalysisJobConfig::post_chunk_bytes`]-style chunks when
/// `post_chunk_bytes` is set — returning every response in POST order. Shared by the batched
/// analysis jobs and the training run's subject-mask prepass (sc-2124).
#[cfg(native_lane)]
pub(crate) async fn post_dataset_sidecar(
    api: &ApiClient,
    project_id: &str,
    dataset_id: &str,
    endpoint_suffix: &str,
    space: &str,
    items: Vec<Value>,
    post_chunk_bytes: Option<usize>,
) -> WorkerResult<Vec<Value>> {
    let path =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/{endpoint_suffix}");
    let mut responses = Vec::new();
    for chunk in chunk_post_items(items, post_chunk_bytes) {
        let response: Value = api
            .post_json(&path, &json!({ "space": space, "items": chunk }))
            .await?;
        responses.push(response);
    }
    Ok(responses)
}

/// The cancel copy and join-error label of one [`stream_batched_records`] run.
#[cfg(native_lane)]
pub(crate) struct BatchedStream<'a> {
    pub cancel_message: &'a str,
    pub join_error_label: &'a str,
}

/// Supervise a records-producing blocking task until it finishes (sc-8836; extracted for the
/// training subject-mask prepass, sc-2124): binds `blocking` to `cancel` via a
/// [`StreamSupervisor`], posts `progress_for(index)` for each per-item `index` the producer sends on
/// `rx`, heartbeats and polls cancel on each tick, and joins the records once the channel closes. A
/// cancel trips the flag, waits for the task to stop, then posts the terminal `Canceled` and
/// returns [`WorkerError::Canceled`].
#[cfg(native_lane)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn stream_batched_records<R, F>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    stream: BatchedStream<'_>,
    cancel: CancelFlag,
    mut rx: tokio::sync::mpsc::Receiver<usize>,
    blocking: tokio::task::JoinHandle<WorkerResult<Vec<R>>>,
    progress_for: F,
) -> WorkerResult<Vec<R>>
where
    F: Fn(usize) -> ProgressRequest,
{
    let mut supervisor =
        StreamSupervisor::new(cancel, blocking, progress_report_interval(settings), None);
    let handling = async {
        loop {
            match supervisor
                .next(api, settings, &job.id, Some(&mut rx))
                .await?
            {
                StreamSignal::Event(index) => {
                    update_job(api, &job.id, progress_for(index)).await?;
                }
                StreamSignal::Finished => break,
                StreamSignal::CancelRequested | StreamSignal::ProgressTick => {}
            }
        }
        Ok(())
    }
    .await;
    let (canceled, result) = supervisor.finish_stream(handling, &mut rx).await?;
    let result = result.map_err(|error| task_join_error(stream.join_error_label, error))?;
    if canceled {
        mark_job_canceled(api, &job.id, stream.cancel_message).await?;
        return Err(WorkerError::Canceled(stream.cancel_message.to_owned()));
    }
    result
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn item_progress_scales_into_the_running_band() {
        // The shared per-item ramp: first of 10 lands just above the 0.12 floor, last lands at 0.90.
        let total = 10;
        assert!((item_progress(0, total) - (0.12 + 0.78 * 0.1)).abs() < 1e-12);
        assert!((item_progress(total - 1, total) - 0.90).abs() < 1e-12);
        // Monotonic across the batch.
        for i in 1..total {
            assert!(item_progress(i, total) > item_progress(i - 1, total));
        }
    }

    #[test]
    fn post_items_chunk_under_the_byte_budget_in_order() {
        // Each item encodes to 12 bytes (`{"k":"xxx"}` = 11 + the separator).
        let items: Vec<Value> = (0..5).map(|i| json!({ "k": format!("x{i:02}") })).collect();
        assert_eq!(items[0].to_string().len() + 1, 12);
        let chunks = chunk_post_items(items.clone(), Some(25));
        assert_eq!(
            chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![2, 2, 1],
            "two 12-byte items fit 25 bytes, a third would not"
        );
        assert_eq!(chunks.concat(), items, "every item, once, in order");
        // An item larger than the budget travels alone rather than being dropped.
        assert_eq!(chunk_post_items(items.clone(), Some(5)).len(), 5);
        // No budget → one POST; an empty batch still makes its (empty) POST.
        assert_eq!(chunk_post_items(items.clone(), None), vec![items]);
        assert_eq!(
            chunk_post_items(Vec::new(), Some(25)),
            vec![Vec::<Value>::new()]
        );
    }

    #[test]
    fn item_progress_guards_a_zero_total() {
        // A degenerate `total == 0` must not divide by zero (callers reject empty item lists upstream,
        // but the ramp stays total-safe regardless).
        assert!(item_progress(0, 0).is_finite());
    }

    #[test]
    fn analysis_progress_stamps_backend_and_number() {
        let request = analysis_progress(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            "half",
            None,
            "mlx",
        );
        assert_eq!(request.message, "half");
        assert_eq!(request.backend.as_deref(), Some("mlx"));
        assert!(request.result.is_none());
    }
}

#[cfg(test)]
mod parsing_tests {
    use super::*;

    #[test]
    fn dataset_items_preserve_strict_and_skip_malformed_contracts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut settings = crate::tests::test_settings(String::new(), None);
        settings.data_dir = dir.path().to_owned();
        let root = dir.path().join("datasets/ds-1");
        let payload = json!({"datasetRoot": root, "items": [null, {},
            {"itemId": "", "imagePath": "one.png"},
            {"itemId": " raw-id ", "imagePath": "one.png"}]})
        .as_object()
        .cloned()
        .expect("payload");
        let items = parse_dataset_items(
            &settings,
            &payload,
            "Dataset upscale",
            DatasetItemPolicy::SkipMalformed,
            |item| Ok((item.item_id.to_owned(), item.image_path()?)),
        )
        .expect("upscale skips malformed");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, " raw-id ");
        assert_eq!(
            items[0].1,
            dir.path()
                .canonicalize()
                .expect("root")
                .join("datasets/ds-1/one.png")
        );
        #[cfg(native_lane)]
        {
            assert!(parse_dataset_items(
                &settings,
                &payload,
                "Caption",
                DatasetItemPolicy::Strict,
                |item| Ok(item.item_id.to_owned())
            )
            .is_err());
            let valid = json!({"datasetRoot": root, "items": [{"itemId": " id ", "imagePath": "one.png", "contentHash": " hash "}]}).as_object().cloned().expect("payload");
            let items = parse_dataset_items(
                &settings,
                &valid,
                "Dataset analysis",
                DatasetItemPolicy::Strict,
                |item| {
                    Ok((
                        item.item_id.to_owned(),
                        item.required("contentHash")?.to_owned(),
                        item.image_path()?,
                    ))
                },
            )
            .expect("strict item");
            assert_eq!((&*items[0].0, &*items[0].1), ("id", "hash"));
        }
        let escaped =
            json!({"datasetRoot": root, "items": [{"itemId": "id", "imagePath": "../escape.png"}]})
                .as_object()
                .cloned()
                .expect("payload");
        assert!(
            parse_dataset_items(
                &settings,
                &escaped,
                "Dataset upscale",
                DatasetItemPolicy::SkipMalformed,
                |item| item.image_path()
            )
            .is_err(),
            "malformed-skip policy never skips path confinement failures"
        );
    }
}
